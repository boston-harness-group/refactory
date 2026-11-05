"""Replay model client for offline SkillOpt replay tests.

Feeds arm A's recorded model outputs back to arm B's SkillOpt strategy,
matching by content (system + normalised prompt) regardless of call order.

FND-skillopt-source-a-2: ``_build_prompt_from_messages()`` at
skillopt/model/claude_backend.py:83-142 assembles the transport wrapping.
FND-skillopt-source-a-6: system prompts are stored raw in calls.jsonl.
FND-skillopt-source-a-8 / FND-recording-format-1: stdout is JSON with a
``result`` field.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from refactory_dsh.model import CompletionTimeout

from refactory_packages.skillopt.trace import current_call_key


# ── normaliser ─────────────────────────────────────────────────────────────

# Tail instructions A's transport wrapper appends (anchored to end).
_TAIL_RE = re.compile(
    r"\n\n(?:Answer the latest user request\.|Return only JSON matching the provided schema\.)\s*$"
)


def normalize_recorded_prompt(prompt: str) -> str:
    """Strip A's Claude Code transport wrapping from a recorded prompt.

    Steps (from the outside in):
    1. Strip trailing tail instruction.
    2. Strip leading ``History:\\n`` prefix.
    3. Strip ``- User: `` prefix from the first content line.
    4. Un-indent continuation lines (remove first 2 spaces).
    5. Strip trailing whitespace.
    """
    # 1. Strip trailing tail instruction.
    text = _TAIL_RE.sub("", prompt)

    # 2. Strip leading ``History:\n`` prefix.
    if text.startswith("History:\n"):
        text = text[len("History:\n"):]
    else:
        return text.rstrip()

    # 3. Strip ``- User: `` prefix from the first content line.
    if text.startswith("- User: "):
        text = text[len("- User: "):]
    elif text.startswith("- User:"):
        text = text[len("- User:"):]
    else:
        return text.rstrip()

    # 4. Un-indent continuation lines (remove the first 2 spaces).
    lines = text.split("\n")
    result_lines = [lines[0]]
    for line in lines[1:]:
        if line.startswith("  "):
            result_lines.append(line[2:])
        else:
            result_lines.append(line)
    text = "\n".join(result_lines)

    # 5. Strip trailing whitespace.
    return text.rstrip()


# ── exceptions ─────────────────────────────────────────────────────────────


class ReplayMismatch(Exception):
    """No recorded call matches the request."""

    def __init__(
        self,
        request_system: str,
        request_prompt: str,
        nearest_system: str,
        nearest_prompt: str,
        diff_text: str,
    ) -> None:
        self.request_system = request_system
        self.request_prompt = request_prompt
        self.nearest_system = nearest_system
        self.nearest_prompt = nearest_prompt
        self.diff_text = diff_text
        super().__init__(
            f"No recorded call matches the request.\n"
            f"Nearest diff:\n{diff_text[:2000]}"
        )


class ReplayExhausted(Exception):
    """All recorded outputs for an identical (system, prompt) pair consumed."""

    def __init__(self, system: str, prompt: str, consumed_count: int) -> None:
        self.system = system
        self.prompt = prompt
        self.consumed_count = consumed_count
        super().__init__(
            f"All {consumed_count} recorded outputs for this (system, prompt) "
            f"pair have been consumed."
        )


# ── matcher ────────────────────────────────────────────────────────────────


def match_key(system: str, prompt: str) -> tuple[str, str]:
    """The key a call is matched on: system and prompt, each without trailing whitespace.

    A called the model through the Claude Code CLI, and its recorded system prompts lose
    the trailing newline SkillOpt's ``_build_system`` keeps (the rollout_system.md
    template ends with one, rollout.py:16-21, and ``load_prompt`` does not strip it).
    Arm B reproduces SkillOpt's bytes, so the same system prompt differs by that one
    newline, and an exact key mismatched every solver call of the first s42 replay.
    Trailing whitespace is transport, not content; the recorded user prompts are already
    stripped the same way (`normalize_recorded_prompt`, step 5).
    """
    return system.rstrip(), prompt.rstrip()


class ReplayedCallFailure(RuntimeError):
    """A's call failed every attempt; the replay fails it the same way.

    SkillOpt's Claude backend retries a failed CLI call and then raises
    ``RuntimeError(f"Claude backend failed after {retries} retries: {last_err}")``
    (skillopt/model/claude_backend.py:309-324), ``last_err`` being
    ``Claude CLI exited with code {rc}`` when the CLI wrote nothing to stderr
    (:282). A records each failed attempt in calls.jsonl with ``rc != 0``.
    """


#: A's own timeouts: SkillOpt's rollout reports an item as ``task-timeout-{N}s``
#: when its future outlives the task timeout (livemathematicianbench/rollout.py:
#: 394-399, :420-432), whatever the model does afterwards. The answer that
#: arrives later is still in calls.jsonl -- a *late answer*, never served. The
#: replay raises refactory's own ``CompletionTimeout`` for that call, exactly
#: the exception a live arm-B call raises when it runs out of time, so B
#: records it as it would record a live timeout.
A_TIMEOUT_MESSAGE = (
    "model completion timed out: arm A's call outlived its task timeout "
    "(its late answer is calls.jsonl line {line})"
)


@dataclass
class _Recording:
    line: int
    at: float
    result: str
    ok: bool
    rc: int
    late: bool = False


#: SkillOpt's attempts per call: the LiveMath rollout asks with ``retries=5``
#: (rollout.py:227, :241); every optimizer call with ``retries=3``
#: (reflect.py:359, :456; aggregate.py:50, :248; clip.py:77;
#: slow_update.py:431; meta_skill.py:67).
SOLVER_ATTEMPTS = 5
OPTIMIZER_ATTEMPTS = 3


class CallMatcher:
    """Index A's calls.jsonl by (system, normalised_prompt) → list of recordings.

    Matching is by content regardless of order (A's solver calls ran
    concurrently). A per-key positional counter disambiguates genuinely
    identical (system, prompt) pairs.

    A recording is one *attempt*. A's failed attempts (``rc != 0``) are
    replayed as A's retries: failures followed by a success within the call's
    attempts serve the success; failures that use up the attempts, or the key's
    recordings, raise :class:`ReplayedCallFailure`. A recording marked late
    (:meth:`mark_late`, from A's own timeout results) raises
    refactory's ``CompletionTimeout`` (:data:`A_TIMEOUT_MESSAGE`) and is never served. None of these is a divergence.

    ``unused_recordings()`` tracks per-recording consumption: if A has 3
    identical calls and B only consumes 1, the 2 remaining are reported.
    Late answers are reported apart (:meth:`late_answers`).
    """

    def __init__(self, calls_jsonl_path: str | Path) -> None:
        path = Path(calls_jsonl_path)
        self._index: dict[tuple[str, str], list[_Recording]] = {}
        self._counters: dict[tuple[str, str], int] = {}
        self._by_line: dict[int, tuple[tuple[str, str], _Recording]] = {}
        #: Failed attempts replayed as A's retries, and late answers withheld.
        self.failed_attempts_replayed = 0
        self.late_answers_withheld: list[int] = []
        #: The latest calls.jsonl line B was served (-1 before the first).
        self.last_line_served = -1
        #: The first call that did not match, in the order B made them. A replay ends
        #: there: everything after it follows a different trajectory. Calls made after
        #: it raise at once (`ReplayMismatch`), so a failure the run isolates -- one
        #: item's solver call, which scores the item 0 and goes on (#554) -- cannot
        #: hide the divergence behind a later, misleading one.
        self.first_divergence: ReplayMismatch | ReplayExhausted | None = None

        self._total_recordings = 0
        with open(path) as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                system = entry.get("system", "")
                normalised = normalize_recorded_prompt(entry.get("prompt", ""))
                stdout_data = json.loads(entry["stdout"])
                rc = int(entry.get("rc", 0) or 0)
                recording = _Recording(
                    line=line_no,
                    at=float(entry.get("at", 0.0) or 0.0),
                    result=stdout_data.get("result", ""),
                    ok=rc == 0 and not stdout_data.get("is_error"),
                    rc=rc,
                )
                key = match_key(system, normalised)
                self._index.setdefault(key, []).append(recording)
                self._by_line[line_no] = (key, recording)
                self._total_recordings += 1

        for key, items in self._index.items():
            items.sort(key=lambda r: r.at)  # chronological
            self._counters[key] = 0

    @property
    def total_recordings(self) -> int:
        """Total number of individual recordings loaded."""
        return self._total_recordings

    def recordings_for(self, system: str, prompt: str) -> list[tuple[int, str, bool]]:
        """``(line, result, ok)`` of every recording under this key, chronologically."""
        return [(r.line, r.result, r.ok) for r in self._index.get(match_key(system, prompt), [])]

    def mark_late(self, lines: list[int]) -> None:
        """Mark recordings (calls.jsonl line numbers) as A's late answers."""
        for line in lines:
            self._by_line[line][1].late = True

    def lookup(self, system: str, prompt: str, *, attempts: int | None = None) -> str:
        """Look up the next recorded output for this (system, prompt) pair.

        ``prompt`` is unframed: A's prompts are normalised at index time, and
        :class:`ReplayClient` takes the framing off B's solver prompts before asking.
        ``attempts`` is how many tries A's client made per call; ``None``
        allows any number of failed attempts before a success.
        """
        if self.first_divergence is not None:
            raise ReplayMismatch(system, prompt, "", "",
                                 "the replay already diverged at an earlier call")
        key = match_key(system, prompt)
        if key not in self._index:
            # Find the nearest match for diagnostics.
            nearest_sys, nearest_prompt = self._nearest(key[1], key[0])
            diff = "\n".join(difflib.unified_diff(
                key[1].splitlines(),
                nearest_prompt.splitlines(),
                fromfile="request",
                tofile="nearest_recorded",
                lineterm="",
            ))
            if key[0] != nearest_sys:
                # A system-only difference leaves the prompt diff empty; show it.
                diff += "\n" + "\n".join(difflib.unified_diff(
                    key[0].splitlines(), nearest_sys.splitlines(),
                    fromfile="request system", tofile="nearest_recorded system", lineterm="",
                ))
            self.first_divergence = ReplayMismatch(system, prompt, nearest_sys, nearest_prompt, diff)
            raise self.first_divergence

        recordings = self._index[key]
        idx = self._counters[key]
        if idx >= len(recordings):
            self.first_divergence = ReplayExhausted(system, prompt, len(recordings))
            raise self.first_divergence

        failed: list[_Recording] = []
        while idx < len(recordings):
            recording = recordings[idx]
            idx += 1
            if recording.late:
                self._counters[key] = idx
                self.failed_attempts_replayed += len(failed)
                self.late_answers_withheld.append(recording.line)
                raise CompletionTimeout(A_TIMEOUT_MESSAGE.format(line=recording.line))
            if recording.ok:
                self._counters[key] = idx
                self.failed_attempts_replayed += len(failed)
                self.last_line_served = max(self.last_line_served, recording.line)
                return recording.result
            failed.append(recording)
            if attempts is not None and len(failed) >= attempts:
                break
        self._counters[key] = idx
        self.failed_attempts_replayed += len(failed)
        raise ReplayedCallFailure(
            f"Claude backend failed after {attempts or len(failed)} retries: "
            f"Claude CLI exited with code {failed[-1].rc}"
        )

    def consumed_count(self) -> int:
        """Recordings B's calls reproduced: answers served and failed attempts
        replayed. Late answers are not counted (:meth:`late_answers`)."""
        return sum(self._counters.values()) - len(self.late_answers_withheld)

    def late_answers(self) -> list[dict[str, Any]]:
        """A's late answers after its own timeout, withheld from B."""
        out = []
        for line in self.late_answers_withheld:
            key, recording = self._by_line[line]
            out.append({"line": line, "system": key[0][:80], "prompt": key[1][:80]})
        return out

    def unused_after_last_served(self) -> int:
        """Unused recordings A made after the last one B was served: A's calls
        past the point where B's run ended (e.g. its post-training evaluations)."""
        count = 0
        for key, recordings in self._index.items():
            for recording in recordings[self._counters[key]:]:
                if recording.line > self.last_line_served:
                    count += 1
        return count

    def unused_before_last_served(self) -> list[dict[str, Any]]:
        """Unused recordings A made before the last one B was served: calls A
        made while B was still running that B never asked for."""
        unused: list[dict[str, Any]] = []
        for key, recordings in self._index.items():
            for recording in recordings[self._counters[key]:]:
                if recording.line <= self.last_line_served:
                    unused.append({"line": recording.line, "system": key[0][:80],
                                   "prompt": key[1][:80]})
        return sorted(unused, key=lambda item: item["line"])

    def consumed_for(self, system: str, prompt: str) -> int:
        """How many recordings for this (system, prompt) pair are already used.

        The public read of the per-key positional counter: ``ReplayClient``
        logs it as the index of the recording it is about to serve, and a
        caller must not reach into ``_counters`` to get it.
        """
        return self._counters.get(match_key(system, prompt), 0)

    def recorded_prompts(self, system: str) -> list[str]:
        """Every normalised prompt recorded under *system*, in index order.

        Read-only: used to recover the order in which A presented a set of
        items inside one prompt (see :class:`RecordedPatchOrder`).
        """
        wanted = system.rstrip()
        return [prompt for recorded_system, prompt in self._index if recorded_system == wanted]

    def unused_recordings(self) -> list[tuple[str, str, int]]:
        """Return (system, prompt, remaining_count) for each key with unused recordings.

        Tracks per-recording consumption: if A has 3 identical calls for a key
        and B consumed only 1, reports (system, prompt, 2) — the 2 remaining.
        A key B never asked for at all is reported with remaining_count = total.
        """
        unused: list[tuple[str, str, int]] = []
        for key, recordings in self._index.items():
            remaining = len(recordings) - self._counters[key]
            if remaining > 0:
                unused.append((key[0], key[1], remaining))
        return unused

    def _nearest(self, prompt: str, system: str = "") -> tuple[str, str]:
        """Find the nearest (system, prompt) key, for the divergence report.

        Searches the keys recorded under the same system when there are any,
        else all of them, and scores by ``SequenceMatcher`` over *lines*. A
        character-level ratio is quadratic in prompt length: on the ~130 KB
        slow-update prompts, which differ from their nearest recording only in
        places, it ran for over a quarter of an hour on one divergence.
        """
        wanted = system.rstrip()
        keys = [key for key in self._index if key[0] == wanted] or list(self._index)
        lines = prompt.splitlines()
        best_ratio = -1.0
        best_key = ("", "")
        for key in keys:
            ratio = difflib.SequenceMatcher(None, lines, key[1].splitlines()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_key = key
        return best_key


# ── A's analyst-patch order, read from its merge prompts ───────────────────

#: The header ``prompts.format_merge_user`` puts in front of the serialised
#: patch array.  Split on it to reach the array without re-deriving the prompt.
_MERGE_HEADER = "\n## Patches to merge ("

def _split_merge_prompt(prompt: str) -> tuple[str, list[Any]] | None:
    """Split a merge prompt into (text before the array, the parsed array).

    ``prompts.format_merge_user`` ends with ``json.dumps(patches, indent=2)``,
    so the array runs to the end of the prompt.  Candidate split points are
    tried from the end, because a skill document or an edit's content may
    itself contain the header text.  Returns ``None`` when the prompt has no
    parseable trailing array.
    """
    search_from = len(prompt)
    while True:
        at = prompt.rfind(_MERGE_HEADER, 0, search_from)
        if at < 0:
            return None
        line_end = prompt.find("\n", at + 1)
        if line_end >= 0:
            prefix = prompt[:line_end + 1]
            try:
                parsed = json.loads(prompt[line_end + 1:])
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return prefix, parsed
        search_from = at


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _permutation(target: list[str], source: list[str]) -> list[int] | None:
    """Indices into *source* that put it in *target*'s order, or ``None``."""
    taken: set[int] = set()
    order: list[int] = []
    for wanted in target:
        for index, candidate in enumerate(source):
            if index not in taken and candidate == wanted:
                order.append(index)
                taken.add(index)
                break
        else:
            return None
    return order if len(order) == len(source) else None


# ── replay-only analyst-patch ordering ─────────────────────────────────────


class RecordedPatchOrder:
    """Replay-only: hand B's merge its analyst patches in A's completion order.

    SkillOpt appends analyst patches as their futures complete
    (reflect.py:643-650), so the order is a race. It shows in the merge prompt
    and, when A's merge call failed, in the fallback concatenation of the
    patches' edits (aggregate.py:70-76). The order is set on B's patch list at
    the source, so B's merge prompt is matched byte for byte, never rewritten
    on the replay side. A's order is read from
    its own recorded merge prompt (served or failed) whose patch array holds
    exactly B's patches; without one, B's order stands.

    It reaches arm B only through ``SkillOptLoop``'s ``patch_order`` option,
    passed by the replay runner alone.
    """

    def __init__(self, matcher: CallMatcher, systems: tuple[str, ...]) -> None:
        self._matcher = matcher
        self._systems = systems
        #: One entry per call: epoch, kind, and whether A's order was found.
        self.applied: list[dict[str, Any]] = []

    def __call__(self, epoch: int, kind: str, patches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        b_canonical = [_canonical(patch) for patch in patches]
        for system in self._systems:
            for recorded in self._matcher.recorded_prompts(system):
                split = _split_merge_prompt(recorded)
                if split is None:
                    continue
                a_canonical = [_canonical(patch) for patch in split[1]]
                if sorted(a_canonical) != sorted(b_canonical):
                    continue
                order = _permutation(a_canonical, b_canonical)
                if order is None:
                    continue
                self.applied.append({"epoch": epoch, "kind": kind, "applied": True, "order": order})
                return [patches[i] for i in order]
        self.applied.append({"epoch": epoch, "kind": kind, "applied": False})
        return patches


# ── replay-only rollout ordering ───────────────────────────────────────────


class RecordedRolloutOrder:
    """Replay-only: hand B's analysts the train records in A's completion order.

    **Why this exists, and why it is not a fix to the algorithm.** SkillOpt's
    LiveMath rollout runs the items on a thread pool and appends each result
    as its future completes (``wait(FIRST_COMPLETED)``,
    ``skillopt/envs/livemathematicianbench/rollout.py:383-436``).  The reflect
    stage splits that list into failures and successes and shuffles each with
    the batch seed (``skillopt/gradient/reflect.py:558-561``,
    ``_shuffle_for_minibatch`` at :471-482), so which items share an analyst
    group depends on which model call returned first.  Arm B's records come in
    the Run's item order, so the same seed yields different groups.  A's order
    is a race, not code to port.

    A writes ``results.jsonl`` in that same completion order (rollout.py:418),
    one ``steps/step_NNNN/rollout/`` per step, and LiveMath runs one step per
    epoch (``A_CONFIG_REQUIRED`` in the runner refuses anything else), so
    epoch *e*'s order is ``steps/step_{e:04d}/rollout/results.jsonl``.

    It reaches arm B only through ``SkillOptLoop``'s ``rollout_order`` option,
    which is ``None`` unless a caller passes a callable, and only the replay
    runner does.  The loop accepts nothing but a permutation of its own
    records.  When A's step has a different set of ids, the records are
    returned unchanged, so the content difference still reaches the matcher.
    """

    def __init__(self, a_out_dir: str | Path) -> None:
        self._a_out = Path(a_out_dir)
        #: One entry per epoch asked: whether A's order was applied, and why not.
        self.applied: list[dict[str, Any]] = []

    def recorded_ids(self, epoch: int) -> list[str] | None:
        path = self._a_out / "steps" / f"step_{epoch:04d}" / "rollout" / "results.jsonl"
        if not path.exists():
            return None
        ids: list[str] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                ids.append(str(json.loads(line)["id"]))
        return ids

    def __call__(self, epoch: int, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        order = self.recorded_ids(epoch)
        b_ids = [str(r.get("id", "")) for r in records]
        if order is None:
            self.applied.append({"epoch": epoch, "applied": False, "reason": "no recorded rollout"})
            return records
        if sorted(order) != sorted(b_ids) or len(set(order)) != len(order):
            self.applied.append({"epoch": epoch, "applied": False, "reason": "different items"})
            return records
        by_id = {str(r.get("id", "")): r for r in records}
        self.applied.append({"epoch": epoch, "applied": True, "order": order})
        return [by_id[item_id] for item_id in order]


# ── A's timeouts, from its own rollout results ─────────────────────────────


def late_answer_lines(a_out_dir: str | Path, matcher: CallMatcher) -> dict[str, Any]:
    """The calls.jsonl lines that are A's late answers after its own timeout.

    For every ``task-timeout-*`` result in any ``results.jsonl`` under A's
    ``out/`` (livemathematicianbench/rollout.py:373, written at :431), the
    timed-out ``process_one`` still finished later and wrote the item's
    ``target_system_prompt.txt``, ``target_user_prompt.txt`` and
    ``conversation.json`` (rollout.py:254-284). Those name the call's key, and
    the recording whose answer is the conversation's message is the late one.

    Returns ``{"lines": [...], "events": [...], "unresolved": [...]}``; an event
    whose late answer cannot be found is listed as unresolved, not guessed.
    """
    a_out = Path(a_out_dir)
    lines: list[int] = []
    events: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for results in sorted(a_out.rglob("results.jsonl")):
        for raw in results.read_text().splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            reason = str(row.get("fail_reason") or "")
            if not reason.startswith("task-timeout"):
                continue
            where = str(results.parent.relative_to(a_out))
            event = {"rollout": where, "id": str(row.get("id")), "fail_reason": reason}
            found = _late_line(results.parent / "predictions", str(row.get("id")), matcher)
            if found is None:
                unresolved.append(event)
                continue
            event["line"] = found
            events.append(event)
            lines.append(found)
    return {"lines": lines, "events": events, "unresolved": unresolved}


def _late_line(predictions: Path, item_id: str, matcher: CallMatcher) -> int | None:
    # skillopt/envs/task_output.py:16-24: unsafe ids map to "_mapped-<sha256>".
    candidates = [predictions / item_id, predictions / (
        "_mapped-" + hashlib.sha256(item_id.encode("utf-8")).hexdigest()
    )]
    for directory in candidates:
        try:
            system = (directory / "target_system_prompt.txt").read_text(encoding="utf-8")
            user = (directory / "target_user_prompt.txt").read_text(encoding="utf-8")
            conversation = json.loads((directory / "conversation.json").read_text())
        except (OSError, ValueError):
            continue
        answers = {
            str(entry.get("content"))
            for entry in conversation
            if isinstance(entry, dict) and entry.get("type") == "message"
        }
        for line, result, ok in matcher.recordings_for(system, user):
            if ok and result in answers:
                return line
    return None


# ── replay client ──────────────────────────────────────────────────────────


class ReplayClient:
    """Model client that replays A's recorded outputs.

    Implements the executor's agent-node protocol:
    ``for_node()``, ``with_tool_policy()``, ``with_wall()``, ``route()``,
    and ``complete()`` with the live client's signature
    (``DshModelClient.complete``: ``system``, ``workdir``, ``cwd``, ``env``,
    ``confine``, nothing else), so a call the live client would refuse fails
    here too (#493 G30).

    A solver prompt is matched after taking off the Claude backend framing, as A's
    recordings are indexed; an optimizer prompt too with ``unframe_optimizer``
    (arm B's ``claude_backend_wrapper``), and otherwise exactly as B wrote it.
    """

    def __init__(self, matcher: CallMatcher, *, unframe_optimizer: bool = False) -> None:
        self.matcher = matcher
        #: Take the Claude backend framing off optimizer prompts too: arm B runs with
        #: ``claude_backend_wrapper`` on.
        self.unframe_optimizer = unframe_optimizer
        self.match_log: list[dict[str, Any]] = []
        #: The first fault of the replay itself (not a divergence, not A's event).
        self.harness_error: str | None = None

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        workdir: Any = None,
        cwd: Any = None,
        env: Any = None,
        confine: bool = False,
    ) -> str:
        """Answer a request from A's recordings."""
        # Every strategy call is an optimizer call, made inside trace.ask (which
        # exposes its key); a solver call (the walker's) had the rollout's attempts.
        call_key = current_call_key()
        if call_key is None or self.unframe_optimizer:
            # The solve node sends the question framed as A's Claude backend framed it
            # (livemath ``ops.claude_backend_prompt``), and so does every optimizer call
            # with ``claude_backend_wrapper`` on; A's recordings are indexed with that
            # framing taken off, so take it off B's prompt the same way.
            prompt = normalize_recorded_prompt(prompt)
        idx = self.matcher.consumed_for(system, prompt)
        attempts = OPTIMIZER_ATTEMPTS if call_key else SOLVER_ATTEMPTS
        entry = {
            "key": (system[:80], prompt[:80]),
            "index": idx,
            "call_key": call_key,
        }
        try:
            result = self.matcher.lookup(system, prompt, attempts=attempts)
        except (CompletionTimeout, ReplayedCallFailure) as event:
            entry["a_event"] = f"{type(event).__name__}: {event}"
            self.match_log.append(entry)
            raise
        except (ReplayMismatch, ReplayExhausted):
            raise
        except Exception as exc:
            # Arm B carries on past a failed optimizer call, as SkillOpt does
            # (trace.ask), so a fault in the replay itself would otherwise look
            # like A's environment. Keep the first one for the report.
            if self.harness_error is None:
                self.harness_error = f"{type(exc).__name__}: {exc}"
            raise
        self.match_log.append(entry)
        return result

    def for_node(self, node_id: str, role: str | None = None) -> "ReplayClient":
        """No-op — returns self."""
        return self

    def with_tool_policy(self, policy: str) -> "ReplayClient":
        """No-op — returns self."""
        return self

    def with_wall(self, seconds: int | float) -> "ReplayClient":
        """No-op — returns self."""
        return self

    def route(self) -> dict[str, str]:
        """Return ``{'worker_backend': 'dsh'}`` to pass the tool_policy gate."""
        return {"worker_backend": "dsh"}
