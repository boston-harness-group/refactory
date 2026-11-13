"""The baselines: a random edit, and a model that reads the gradient.

Both implement the same contract, ``propose`` returning typed edits, so the
engine and the validating loop are identical for either. The difference is only
where the ideas come from, which is what makes the two comparable.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import replace
from typing import Any, Callable, Sequence

from refactory_dsh.nn import edits as nn_edits

from .base import (
    READER_CONCURRENCY, UNREADABLE_PROPOSAL, Candidate, Optimizer, Proposal, ProposalContext,
    considered_text, trajectory_groups, unreadable_proposal,
)

def text_only(model: Any) -> Any:
    """``model`` with ``tool_policy="none"``, when the client supports one.

    Every optimizer call -- the proposal and each reader -- is a prompt in and JSON
    out. On an agentic route (``claude_code``) the model would otherwise get tools in a
    working directory next to the run's records. The run's own client is left alone.
    """
    with_policy = getattr(model, "with_tool_policy", None)
    return with_policy("none") if callable(with_policy) else model


#: How many times one optimizer call is attempted, and how long the loop waits after a
#: failed attempt: SkillOpt's Claude backend (``retries=3``; ``min(2 ** attempt, 15)``
#: seconds after each failed attempt, claude_backend.py:314-324), as the SkillOpt port
#: (``refactory_packages/skillopt/trace.py`` ``ask``) retries it.
CALL_ATTEMPTS = 3


def retry_delay(attempt: int) -> float:
    """Seconds waited after failed attempt ``attempt`` (from 0): 1, 2, 4."""
    return min(2 ** attempt, 15)


#: How a retry waits (tests replace it).
_sleep = time.sleep
#: How often a retry's wait looks for a stop (`AutoresearchOptimizer._wait`).
STOP_POLL_S = 0.25


def _fatal() -> tuple[type[BaseException], ...]:
    """What ends a call at once, never retried: a spent budget, a cancelled run, a
    missing model choice."""
    from concurrent.futures import CancelledError

    from refactory_dsh.budgets import BudgetExhausted
    from refactory_dsh.model import ModelChoiceError

    return (BudgetExhausted, CancelledError, ModelChoiceError)


#: A reader's role: go through one group of whole trajectories and report patterns.
READER_SYSTEM = (
    "You read an AI pipeline's trajectories on evaluation items and report what you find. "
    "You reply with JSON only."
)
READER_ANSWER = """Write one JSON object and nothing else:

{"failure_patterns": [{"pattern": "<what goes wrong, in one sentence>",
                       "step": "<the step it starts in, or \"unclear\">",
                       "items": ["<item id>", ...],
                       "evidence": ["<a short verbatim quote from a trajectory>", ...]}],
 "success_patterns": [{"pattern": "<what the passing items do right>", "step": "...",
                       "items": [...], "evidence": [...]}]}

Rules:
- Every pattern names the items it covers and quotes them; no pattern without evidence.
- Cover every item answered wrong; group items that fail the same way.
- Do not propose changes: say what happens, not what to do.
"""


def _elite(parents: Sequence[Candidate], ctx: ProposalContext) -> Sequence[Candidate]:
    """The top ``ctx.elite``: exactly what the loop passed before the pool widened.

    The loop hands a proposer the whole evaluated population now, so a policy-based
    optimizer can select for itself. An optimizer with no policy of its own still
    wants the elite, and this reproduces `Loop.leaderboard` rather than approximating
    it: the same filter (a candidate with no score is not a parent), the same
    ordering (score descending, **id as the tie-break**), and the same fallback when
    nothing has scored yet. Getting any of those subtly wrong changes which parents
    are bred from, which changes the search, which is how a duplicate stopped being
    refused one layer earlier than it should have been.
    """
    val = dict(getattr(ctx, "val_scores", None) or {})
    if val:
        # Under the validation gate the elite is the best on val, a tie by the higher
        # train score, then the order handed over (the loop's pool order). A candidate
        # whose train run failed still has a val score, so the incumbent stays a
        # parent even then.
        ranked = sorted(
            (candidate for candidate in parents if candidate.id in val),
            key=lambda candidate: (-val[candidate.id],
                                   -(candidate.score if candidate.score is not None
                                     else float("-inf"))),
        )
        if ranked:
            return ranked[: max(1, ctx.elite)]
    ranked = sorted(
        (candidate for candidate in parents if candidate.score is not None),
        key=lambda candidate: (-(candidate.score or 0.0), candidate.id),
    )
    if not ranked:
        # `Loop.select_parents` falls back to the population, in its own order.
        return list(parents)[: max(1, ctx.elite)]
    return ranked[: max(1, ctx.elite)]


class RandomOptimizer(Optimizer):
    """Sample edits uniformly from the enabled catalogue.

    Structural edits need blocks to insert, so a mode that declares none is only
    reachable through its parameters. The sampler says so by returning fewer
    proposals rather than by inventing an operation.
    """

    name = "random"

    def __init__(self, *, k: int = 3, seed: int = 0) -> None:
        super().__init__(k=k)
        self.seed = seed

    def rng(self, ctx: ProposalContext, n: int) -> random.Random:
        """A stream derived from the seed and the generation, not carried forward.

        Derived rather than accumulated so a proposal sequence is a pure function of
        the run's configuration and how far it has got. That is what lets a run be
        resumed and reproduce what an uninterrupted run would have done, and it costs
        nothing: a carried stream has no value beyond being reproducible.
        """
        return random.Random(f"{self.seed}:{ctx.generation}:{ctx.epoch}:{n}")

    def propose(
        self, ctx: ProposalContext, n: int, parents: Sequence[Candidate]
    ) -> list[Proposal]:
        if not parents:
            return []
        # `parents` is the evaluated population since issue #32, so that a policy-based
        # optimizer can select from it. This one has no parent policy: it takes the
        # elite the loop used to hand it, which keeps its proposals what they always
        # were and stops widening the pool from silently changing every run.
        parents = _elite(parents, ctx)
        rng = self.rng(ctx, n)
        preferred = [name for name in ctx.preferred_edits if name in set(nn_edits.enabled(ctx.preset))]
        proposals: list[Proposal] = []
        for index in range(max(1, n)):
            parent = parents[index % len(parents)]
            edit = None
            if preferred and index % 2 == 0:
                # Every other proposal goes to what the author suggested. A prior,
                # not a restriction: the sampler still covers the catalogue.
                edit = self._preferred_edit(rng.choice(preferred), parent, ctx, rng)
            if edit is None:
                edit = nn_edits.random_edit(
                    parent.state,
                    preset=ctx.preset,
                    insertable=ctx.insertable,
                    rng=rng,
                )
            if edit is None:
                continue
            proposals.append(
                Proposal(edit=edit, parent=parent.id, rationale="sampled at random")
            )
        return proposals


    def _preferred_edit(
        self, op: str, parent: Candidate, ctx: ProposalContext, rng: random.Random
    ) -> dict[str, Any] | None:
        """A random edit restricted to one operation, or ``None`` if none fits."""
        for _ in range(20):
            edit = nn_edits.random_edit(
                parent.state,
                preset="broad",
                insertable=ctx.insertable,
                rng=rng,
            )
            if edit is not None and edit.get("op") == op:
                return edit
        return None


def checked_patterns(payload: Any, items: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    """A reader's patterns with every cited item id checked against the group it read.

    A reader sees only its own group, so an id outside it is a slip (a val id, a
    misremembered one) and must not reach the proposal call as evidence: it is dropped
    from ``items`` and kept under ``not_in_group``, which the request says.
    """
    group = {str(item) for item in items}
    out: dict[str, list[dict[str, Any]]] = {}
    for kind in ("failure_patterns", "success_patterns"):
        patterns: list[dict[str, Any]] = []
        for pattern in (payload.get(kind) if isinstance(payload, dict) else None) or ():
            if not isinstance(pattern, dict):
                continue
            cited = [str(i) for i in pattern.get("items") or ()]
            patterns.append({**pattern, "items": [i for i in cited if i in group],
                             "not_in_group": [i for i in cited if i not in group]})
        out[kind] = patterns
    return out


class AutoresearchOptimizer(Optimizer):
    """Ask a model to read the traces and propose the next update.

    The prompt carries the write surface, the edit catalogue, the insertable
    blocks, the leaderboard and the evidence, and — on a retry — the reasons the
    previous attempt was refused. Telling the model exactly what was wrong is what
    makes the loop converge in one or two attempts instead of five.
    """

    name = "autoresearch"
    uses_model = True

    def __init__(self, client: Any, *, k: int = 3, system: str = "") -> None:
        super().__init__(k=k)
        if client is None:
            raise ValueError("the autoresearch optimizer needs a model client")
        self._client = text_only(client)
        self._completion_options: dict[str, Any] = {}
        self._system = system or (
            "You are the optimizer of a search over workflows. You reply with JSON only."
        )
        #: This round's analysis (``ProposalContext.analysts`` on; `_analyse`), for the
        #: loop to journal; ``None`` when no analyst ran.
        self.last_analysis: dict[str, Any] | None = None

    def bind_model(self, model: Any, *, workdir: Any = None) -> None:
        import inspect
        self._client = text_only(model)
        # Keep the original minimal ModelClient protocol compatible. DSH supports
        # an owned workspace, so its proposer receipts belong to this search.
        if workdir is not None and "workdir" in inspect.signature(model.complete).parameters:
            self._completion_options = {"workdir": workdir}

    def _read(self, ctx: ProposalContext, groups: list[list[tuple[str, str]]]
              ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Readers go through the item groups, concurrently: their findings, and the
        groups that could not be read. A group read before (a retry after a refusal)
        is not read again."""
        from concurrent.futures import ThreadPoolExecutor
        import hashlib

        cache: dict[str, dict[str, Any]] = getattr(self, "_reader_cache", {})
        self._reader_cache = cache
        # Kept in the run directory too, so a resumed run reuses what it paid for. The
        # key is the group's whole text: the edited candidate's records for exactly
        # these items.
        from pathlib import Path
        store = (Path(self.trace_path).parent / "reader_findings"
                 if self.trace_path is not None else None)
        steps = ", ".join(f"`{n.get('id')}`" for n in ctx.nodes) or "(not listed)"

        def read(index: int, group: list[tuple[str, str]]) -> tuple[dict | None, dict | None]:
            items = [item for item, _block in group]
            body = "\n\n".join(block for _item, block in group)
            key = hashlib.sha256(body.encode()).hexdigest()
            if key in cache:
                return cache[key], None
            if store is not None and (store / f"{key}.json").exists():
                try:
                    saved = json.loads((store / f"{key}.json").read_text())
                    cache[key] = {**saved, **checked_patterns(saved, items), "group": index}
                    return cache[key], None
                except (OSError, json.JSONDecodeError):
                    pass
            prompt = (f"# Reading trajectories -- group {index} of {len(groups)}\n\n"
                      f"Mode: `{ctx.mode_name}`. The pipeline's steps: {steps}.\n\n"
                      f"Below are {len(items)} train item(s) of the candidate being edited, "
                      "each in full: its input, every step's output in the order they ran, "
                      "the expected answer, the prediction and the grader's reason.\n\n"
                      f"{body}\n\n## Answer\n{READER_ANSWER}")
            call = {"kind": "reader", "epoch": ctx.epoch, "generation": ctx.generation,
                    "group": index, "items": items, "system": READER_SYSTEM, "prompt": prompt}
            try:
                reply = self._client.complete(prompt, system=READER_SYSTEM,
                                              **self._completion_options)
                payload = extract_json(reply)
            except Exception as exc:
                self.record_call(**call, reply=locals().get("reply"), error=f"{type(exc).__name__}: {exc}")
                return None, {"group": index, "items": items, "error": f"{type(exc).__name__}: {exc}"}
            found = {"group": index, "items": items, **checked_patterns(payload, items)}
            self.record_call(**call, reply=reply, error=None,
                             patterns=len(found["failure_patterns"]) + len(found["success_patterns"]))
            cache[key] = found
            if store is not None:
                try:
                    store.mkdir(parents=True, exist_ok=True)
                    (store / f"{key}.json").write_text(json.dumps(found, default=str))
                except OSError:
                    pass
            return found, None

        with ThreadPoolExecutor(max_workers=max(1, min(READER_CONCURRENCY, len(groups)))) as pool:
            results = list(pool.map(lambda pair: read(*pair), enumerate(groups, 1)))
        findings = [found for found, _failed in results if found is not None]
        failures = [failed for _found, failed in results if failed is not None]
        return findings, failures

    def _complete(self, call: dict[str, Any], text: str, system: str, gate: Any = None
                  ) -> tuple[str | None, str | None, bool, int]:
        """One optimizer call, retried as SkillOpt retries it (``CALL_ATTEMPTS``): the
        reply, or ``None`` and the last error; whether the provider refused the prompt as
        longer than the model's context; and how many attempts were made. Any exception
        from an attempt is a failed attempt, recorded in the trace (with its ``attempt``,
        from 1) and followed by a wait of ``retry_delay``. A spent budget, a cancelled run
        or a missing model choice raise at once (`_fatal`). A context overflow is not
        tried again at the same length: the caller retries it shorter. A reply is never
        retried, however unusable: reading it is the caller's. ``gate`` (a semaphore)
        bounds the calls in flight, not the waits."""
        from contextlib import nullcontext

        from . import analysts

        error = None
        for attempt in range(1, CALL_ATTEMPTS + 1):
            try:
                with gate if gate is not None else nullcontext():
                    reply = self._client.complete(text, system=system, **self._completion_options)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self.record_call(**call, attempt=attempt, reply=None, error=error)
                if isinstance(exc, _fatal()):
                    raise
                if analysts.context_overflow(error):
                    return None, error, True, attempt
                self._wait(retry_delay(attempt - 1))
                continue
            return reply, None, False, attempt
        return None, error, False, CALL_ATTEMPTS

    def _wait(self, seconds: float) -> None:
        """A retry's wait, cut short by a stop: the client's own cancellation scopes
        (``DshModelClient.with_cancellation``) or the loop's ``stop_check`` (a person's
        stop request), checked every ``STOP_POLL_S``: a person's stop raises ``RunStopped``
        (the search then ends as a stop), a cancelled scope ``CancelledError``; either way
        the call is not attempted again. With neither, ``_sleep``."""
        from concurrent.futures import CancelledError

        from .base import RunStopped

        events = tuple(getattr(self._client, "_cancellations", ()) or ())
        check = getattr(self, "stop_check", None)
        if not events and not callable(check):
            _sleep(seconds)
            return
        left = float(seconds)
        while True:
            if callable(check) and check():
                raise RunStopped("a person asked the run to stop during an optimizer call's "
                                 "retry wait; not attempted again")
            if any(event.is_set() for event in events):
                raise CancelledError("the optimizer call was cancelled during its retry "
                                     "wait; not attempted again")
            if left <= 0:
                return
            step = min(left, STOP_POLL_S)
            if events:
                events[-1].wait(step)
            else:
                _sleep(step)
            left -= step

    def _review_steps(self, ctx: ProposalContext) -> list[dict[str, Any]]:
        """Each added step (``ctx.added_steps``) with a reviewer's verdict on whether its
        output holds work its inputs lacked: one call per step that recorded text output,
        up to ``STEP_REVIEW_STEPS``, concurrently, each recorded (kind ``step_review``)
        and cached like the readers' findings, in memory and in the run directory, so a
        retry or a resumed run pays for none again. A failed call is recorded and the
        step is shown with ``review_error``; it never fails the proposal."""
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path

        from . import analysts, feedback

        cache: dict[str, dict[str, Any]] = self.__dict__.setdefault("_review_cache", {})
        store = (Path(self.trace_path).parent / "step_reviews"
                 if self.trace_path is not None else None)
        reviewable = [index for index, entry in enumerate(ctx.added_steps)
                      if entry.get("samples")][: feedback.STEP_REVIEW_STEPS]

        def review(index: int) -> tuple[int, dict[str, Any]]:
            entry = ctx.added_steps[index]
            prompt = feedback.review_prompt(entry, ctx.mode_name)
            key = feedback.review_key(prompt)
            if key in cache:
                return index, {"review": cache[key]}
            if store is not None and (store / f"{key}.json").exists():
                try:
                    saved = feedback.stored_review((store / f"{key}.json").read_text())
                except OSError:
                    saved = None
                if saved is not None:
                    cache[key] = saved
                    return index, {"review": saved}
            parse = (feedback.parse_decision_review if entry.get("decision")
                     else feedback.parse_review)

            def attempt(text: str, retry: bool) -> tuple[Any, str | None, bool]:
                """The review, or its error and whether the model call itself (not the
                reading of its reply) failed as longer than the model's context. The
                call is retried as every optimizer call is (`_complete`)."""
                call = {"kind": "step_review", "epoch": ctx.epoch,
                        "generation": ctx.generation, "step": entry.get("step"),
                        "candidate": entry.get("candidate"),
                        "items": [s.get("item") for s in entry.get("samples") or ()],
                        "system": feedback.STEP_REVIEW_SYSTEM, "prompt": text,
                        **({"context_retry": True} if retry else {})}
                reply, error, overflow, made = self._complete(
                    call, text, feedback.STEP_REVIEW_SYSTEM)
                if reply is None:
                    return None, error, overflow
                try:
                    found = parse(extract_json(reply))
                except Exception as exc:  # a failed review is shown as such, never fatal
                    error = f"{type(exc).__name__}: {exc}"
                    self.record_call(**call, attempt=made, reply=reply, error=error)
                    return None, error, False
                self.record_call(**call, attempt=made, reply=reply, error=None,
                                 verdict=found["verdict"])
                return found, None, False

            found, error, overflow = attempt(prompt, False)
            if found is None and overflow:
                # Once, at half the length the provider refused, as an analyst is retried;
                # not at all when the request cannot be made shorter.
                smaller = feedback.review_prompt(entry, ctx.mode_name,
                                                 room=max(1, len(prompt) // 2))
                if len(smaller) < len(prompt):
                    found, error, _overflow = attempt(smaller, True)
                else:
                    error = (f"{error}; the request could not be made shorter, so it was not "
                             "tried again")
            if found is None:
                return index, {"review_error": error}
            cache[key] = found
            if store is not None:
                try:
                    store.mkdir(parents=True, exist_ok=True)
                    (store / f"{key}.json").write_text(json.dumps(found))
                except OSError:
                    pass
            return index, {"review": found}

        reviewed: dict[int, dict[str, Any]] = {}
        if reviewable:
            with ThreadPoolExecutor(max_workers=max(1, min(READER_CONCURRENCY,
                                                           len(reviewable)))) as pool:
                reviewed = dict(pool.map(review, reviewable))
        return [{**entry, **reviewed.get(index, {})}
                for index, entry in enumerate(ctx.added_steps)]

    def _model_id(self) -> str:
        """The optimizer route's model, for the analyst caches' keys: an answer from
        one model is not reused for another."""
        route = getattr(self._client, "route", None)
        try:
            found = route() if callable(route) else {}
        except Exception:  # a client that cannot say is keyed by its type alone
            found = {}
        found = found if isinstance(found, dict) else {}
        return json.dumps({key: found.get(key) for key in
                           ("worker_backend", "provider", "model", "effort")} | {
                              "client": type(self._client).__name__}, sort_keys=True)

    def _ask(self, ctx: ProposalContext, system: str, prompt: str, fields: dict[str, Any],
             parse: Any, counts: dict[str, int], rebuild: Any = None,
             missing: str = 'the answer has no "findings" list'
             ) -> tuple[Any, str | None, bool, bool]:
        """One analyst-stage call (`_analyse`, and the memory call, `write_memory`):
        ``parse(payload)`` of the reply, or ``None`` and why (``missing`` when ``parse``
        finds nothing), whether it was retried after a context overflow, and whether
        it was answered from the cache (nothing paid now). Cached by
        the optimizer's model and the call's whole text, in memory and in the run
        directory (``analyst_findings/``); a call is recorded in the optimizer trace with
        what it found. At most ``analysts.CONCURRENCY`` calls of a stage are in flight
        at once, merges included. A failed call is attempted again as SkillOpt retries
        it (`_complete`: up to ``CALL_ATTEMPTS``, waiting ``retry_delay``); an
        unreadable reply is not. A call the provider refuses as longer than the model's
        context (`analysts.context_overflow`) is retried with ``rebuild(room)``, the
        same request within half the failed prompt's length, unless that is no shorter;
        its answer is cached under
        the original call, so a retry or a resumed run pays for neither again. A spent
        budget, a cancelled run or a missing model choice stop the proposal (as the
        SkillOpt port's ``ask``); any other failure is recorded and returned, never
        raised."""
        import hashlib
        import threading
        from pathlib import Path

        from . import analysts

        cache: dict[str, str] = self.__dict__.setdefault("_analyst_cache", {})
        lock = self.__dict__.setdefault("_analyst_lock", threading.Lock())
        gate = self.__dict__.setdefault("_analyst_gate",
                                        threading.BoundedSemaphore(analysts.CONCURRENCY))
        store = (Path(self.trace_path).parent / "analyst_findings"
                 if self.trace_path is not None else None)
        key = hashlib.sha256(f"{self._model_id()}\n{system}\n{prompt}".encode()).hexdigest()
        held = cache.get(key)
        if held is None and store is not None and (store / f"{key}.json").exists():
            try:
                held = json.loads((store / f"{key}.json").read_text())
            except (OSError, json.JSONDecodeError):
                held = None
        if isinstance(held, dict) and isinstance(held.get("reply"), str):
            found = parse(extract_json(held["reply"]))
            if found is not None:
                with lock:
                    counts["cached"] += 1
                cache[key] = held
                retried = bool(held.get("context_retry"))
                if retried and rebuild is not None:
                    rebuild(max(1, len(prompt) // 2))  # the retried request's own record
                return found, None, retried, True

        def attempt(text: str, retry: bool) -> tuple[Any, str | None, str | None, bool]:
            """The call's findings, error and reply, and whether the model call itself
            (not the reading of its reply) failed as longer than the model's context."""
            call = {**fields, "epoch": ctx.epoch, "generation": ctx.generation,
                    "system": system, "prompt": text,
                    **({"context_retry": True} if retry else {})}
            try:
                reply, error, overflow, made = self._complete(call, text, system, gate=gate)
            except BaseException:
                with lock:
                    counts["calls"] += 1
                raise
            with lock:
                counts["calls"] += made
            if reply is None:
                # Only a failed model call can say the prompt did not fit: an unreadable
                # reply's error quotes the reply, whose prose may name a context window.
                return None, error, None, overflow
            try:
                found = parse(extract_json(reply))
                if found is None:
                    raise ValueError(missing)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self.record_call(**call, attempt=made, reply=reply, error=error)
                return None, error, None, False
            self.record_call(**call, attempt=made, reply=reply, error=None, findings=found)
            return found, None, reply, False

        found, error, reply, overflow = attempt(prompt, False)
        retried = False
        if found is None and rebuild is not None and overflow:
            # Once, at half the length the provider refused, re-clipped the same way;
            # not at all when the request cannot be made shorter.
            smaller = rebuild(max(1, len(prompt) // 2))
            if len(smaller) < len(prompt):
                retried = True
                found, error, reply, _overflow = attempt(smaller, True)
            else:
                error = (f"{error}; the request could not be made shorter, so it was not "
                         "tried again")
        if found is None:
            return None, error, retried, False
        # The answer, under the original call's key: the same call made again (a retry,
        # a resumed run) is answered from here, retried or not.
        held = {"reply": reply, **({"context_retry": True} if retried else {})}
        cache[key] = held
        if store is not None:
            try:
                store.mkdir(parents=True, exist_ok=True)
                (store / f"{key}.json").write_text(json.dumps(held))
            except OSError:
                pass
        return found, None, retried, False

    def write_memory(self, *, epoch: int, generation: int,
                     build: Callable[[int], tuple[str, dict[str, int]]]) -> dict[str, Any]:
        """The epoch's memory call (`optim.memory`): ``build(room)`` makes the request
        within ``room`` characters and says what it clipped. One call of the optimizer's
        model, traced (kind ``memory``), retried and cached as an analyst call is
        (`_ask`), retried at half size after a context overflow. Returns ``{"note"
        ({"text", "reasoning"}) or None, "error", "context_retry", "cached", "calls",
        "clipped_items", "clipped_chars", "prompt_chars"}``; a spent budget, a cancelled
        run or a missing model choice raise."""
        from . import memory

        prompt, clip = build(memory.MEMORY_ROOM)
        clip = dict(clip)
        counts = {"calls": 0, "cached": 0}

        def smaller(room: int) -> str:
            text, used = build(room)
            clip.update(used)
            return text

        found, error, retried, cached = self._ask(
            ProposalContext(epoch=epoch, generation=generation), memory.MEMORY_SYSTEM, prompt,
            {"kind": "memory"}, memory.parse_note, counts, rebuild=smaller,
            missing='the answer has no "memory" note')
        return {"note": found, "error": error, "context_retry": retried, "cached": cached,
                "calls": counts["calls"], "clipped_items": int(clip.get("items") or 0),
                "clipped_chars": int(clip.get("chars") or 0), "prompt_chars": len(prompt)}

    def _round_analysis(self, ctx: ProposalContext) -> dict[str, Any] | None:
        """The analysis of this proposal step, made once: keyed by the epoch, the
        generation, the edited candidate, its train records, the stage's settings and
        the optimizer's model (`analysts.round_key`), so a retry in the same step reuses
        it and pays for no call (the memory note it read is the epoch's, the same for
        every attempt). Kept in memory and in
        ``analyst_findings/round-<key>.json``, so a resumed run reuses it too."""
        from pathlib import Path

        from . import analysts

        key = analysts.round_key(ctx, self._model_id())
        held = self.__dict__.get("_held_analysis")
        if held is not None and held[0] == key:
            return held[1]
        store = (Path(self.trace_path).parent / "analyst_findings" / f"round-{key}.json"
                 if self.trace_path is not None else None)
        if store is not None and store.exists():
            try:
                saved = json.loads(store.read_text())
            except (OSError, json.JSONDecodeError):
                saved = None
            if isinstance(saved, dict) and saved.get("key") == key:
                # Made and journaled by an earlier process: nothing was paid now, so no
                # call of it counts as retried or clipped now either.
                analysis = {**saved, "cached": saved.get("cached", 0) + saved.get("calls", 0),
                            "calls": 0,
                            **{part: [{**entry, "cached": True} for entry in saved.get(part) or ()]
                               for part in ("batches", "merges")}}
                self._held_analysis = (key, analysis)
                return analysis
        analysis = self._analyse(ctx, key)
        if analysis is not None:
            self._held_analysis = (key, analysis)
            failed = (any(b.get("error") for b in analysis["batches"])
                      or any(m.get("error") for m in analysis["merges"]))
            # A step with a failed call is kept for its retries, not for a resumed run,
            # which tries the failed calls again (the others from their own cache).
            if store is not None and not failed:
                try:
                    store.parent.mkdir(parents=True, exist_ok=True)
                    store.write_text(json.dumps(analysis, default=str))
                except OSError:
                    pass
        return analysis

    def _analyse(self, ctx: ProposalContext, key: str = "") -> dict[str, Any] | None:
        """The analyst stage (`optim.analysts`, ``ctx.analysts``): failure and success
        analysts over batches of the edited candidate's train items, concurrently, then
        the failure-first merge into one ranked list. ``None`` without items. When a
        call stops the stage (a spent budget), ``last_analysis`` holds what it made and
        paid so far, with the ``error``, for the loop to journal, and the error is
        raised."""
        from concurrent.futures import ThreadPoolExecutor

        from . import analysts

        conf = analysts.settings(**{k: v for k, v in ctx.analysts.items()
                                    if k in ("group_size", "merge_batch_size", "max_batches")})
        failures, successes = analysts.split(ctx.item_records, int(ctx.epoch))
        if not failures and not successes:
            return None
        plan, left = analysts.batches(failures, successes, group_size=conf["group_size"],
                                      max_batches=conf["max_batches"])
        pipeline = analysts.pipeline_text(ctx)
        totals = {kind: sum(1 for k, _i, _g in plan if k == kind) for kind in analysts.KINDS}
        analysed = {str(r.get("id")) for _k, _i, group in plan for r in group}
        counts = {"calls": 0, "cached": 0}
        merges: list[dict[str, Any]] = []
        done: list[dict[str, Any]] = []

        def made(findings: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
            return {"key": key, "epoch": ctx.epoch, "generation": ctx.generation,
                    "candidate": ctx.parent, **conf,
                    "batches": [{k: v for k, v in b.items() if k != "found"} for b in done],
                    "merges": sorted(merges, key=lambda m: (m["step"], m["level"], m["batch"])),
                    "not_analysed": left, "findings": list(findings), **counts, **extra}

        def analyse(batch: tuple[str, int, list[dict[str, Any]]]) -> dict[str, Any]:
            kind, index, group = batch
            items = [str(r.get("id")) for r in group]
            prompt, clip = analysts.analyst_batch(ctx, kind, index, totals[kind], group,
                                                  pipeline)

            def smaller(room: int) -> str:
                # The same batch re-clipped within ``room``; its clipping is the one
                # the analyst read, so it is the one recorded.
                text, clip_used = analysts.analyst_batch(ctx, kind, index, totals[kind],
                                                         group, pipeline, room=room)
                clip.update(clip_used)
                return text

            found, error, retried, cached = self._ask(
                ctx, analysts.FAILURE_SYSTEM if kind == "failure" else analysts.SUCCESS_SYSTEM,
                prompt,
                {"kind": f"analyst_{kind}", "group": index, "items": items},
                lambda payload: analysts.parse_findings(payload, kind, items), counts,
                rebuild=smaller)
            return {"kind": kind, "index": index, "items": items, "found": found or [],
                    "findings": len(found or ()), "clipped": clip["items"],
                    "clipped_chars": clip["chars"],
                    **({"context_retry": True} if retried else {}),
                    **({"cached": True} if cached else {}),
                    "error": error}

        def merge_lists(step: str, lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
            """SkillOpt's hierarchical merge (``aggregate.py``): lists without findings
            are dropped, one list is not merged, and each level merges batches of
            ``merge_batch_size``; an unreadable merge keeps every input finding."""
            current = [found for found in lists if found]
            level = 0
            while len(current) > 1:
                level += 1
                size = conf["merge_batch_size"]
                chunks = [current[i:i + size] for i in range(0, len(current), size)]

                def merge(numbered: tuple[int, list[list[dict[str, Any]]]],
                          level: int = level) -> list[dict[str, Any]]:
                    number, chunk = numbered
                    if len(chunk) == 1:
                        return chunk[0]
                    found, error, retried, cached = self._ask(
                        ctx, analysts.merge_system(step),
                        analysts.merge_prompt(ctx, step, chunk, level, pipeline),
                        {"kind": f"analyst_merge_{step}", "level": level, "batch": number,
                         "lists": len(chunk)},
                        lambda payload: analysts.parse_findings(payload, step, sorted(analysed),
                                                                limit=None), counts,
                        rebuild=lambda room: analysts.fit_merge_prompt(ctx, step, chunk, level,
                                                                       room))
                    merges.append({"step": step, "level": level, "batch": number,
                                   "lists": len(chunk), "fallback": found is None,
                                   **({"context_retry": True} if retried else {}),
                                   **({"cached": True} if cached else {}),
                                   "error": error})
                    return found if found is not None else [f for part in chunk for f in part]

                with ThreadPoolExecutor(max_workers=max(1, min(analysts.CONCURRENCY,
                                                               len(chunks)))) as pool:
                    current = list(pool.map(merge, enumerate(chunks, 1)))
            return current[0] if current else []

        try:
            with ThreadPoolExecutor(max_workers=max(1, min(analysts.CONCURRENCY,
                                                           len(plan)))) as pool:
                done[:] = list(pool.map(analyse, plan))
            with ThreadPoolExecutor(max_workers=2) as pool:
                failure_side, success_side = pool.map(
                    lambda kind: merge_lists(kind, [b["found"] for b in done
                                                    if b["kind"] == kind]),
                    analysts.KINDS)
            if failure_side and success_side:
                found, error, retried, cached = self._ask(
                    ctx, analysts.merge_system("final"),
                    analysts.merge_prompt(ctx, "final", [failure_side, success_side], 0,
                                          pipeline),
                    {"kind": "analyst_merge_final", "level": 0, "batch": 1, "lists": 2},
                    lambda payload: analysts.parse_findings(payload, "failure",
                                                            sorted(analysed), limit=None),
                    counts,
                    rebuild=lambda room: analysts.fit_merge_prompt(
                        ctx, "final", [failure_side, success_side], 0, room))
                merges.append({"step": "final", "level": 0, "batch": 1, "lists": 2,
                               "fallback": found is None,
                               **({"context_retry": True} if retried else {}),
                               **({"cached": True} if cached else {}),
                               "error": error})
                # SkillOpt's fallback: the failure findings first, then the success ones.
                ranked = found if found is not None else [*failure_side, *success_side]
            else:
                ranked = failure_side or success_side
        except BaseException as exc:
            # What was made and paid before the stop, for the loop to journal.
            self.last_analysis = made([], error=f"{type(exc).__name__}: {exc}")
            raise
        return made(ranked)

    def propose(
        self, ctx: ProposalContext, n: int, parents: Sequence[Candidate]
    ) -> list[Proposal]:
        self.last_analysis = None
        if ctx.added_steps:
            ctx = replace(ctx, added_steps=self._review_steps(ctx))
        if ctx.analysts and ctx.item_records:
            # The analysts read every item in full and their ranked findings take the
            # readers' place, as SkillOpt's analysts are its only reading of the traces.
            self.last_analysis = self._round_analysis(ctx)
            ctx = replace(ctx, analysis=self.last_analysis)
        groups = trajectory_groups(ctx.item_records, ctx.evidence_budget) if ctx.item_records else []
        if len(groups) > 1 and ctx.analysis is None:
            findings, failures = self._read(ctx, groups)
            ctx = replace(ctx, reader_findings=findings, reader_failures=failures)
        ctx = replace(ctx, proposals=max(1, n))
        question = (
            f"{ctx.render()}\n\nPropose at most {max(1, n)} update(s). "
            "Reply with the JSON object described under 'Answer' and nothing else."
        )
        call = {"kind": "proposal", "epoch": ctx.epoch, "generation": ctx.generation,
                "system": self._system, "prompt": question}
        try:
            text = self._client.complete(question, system=self._system, **self._completion_options)
        except Exception as exc:
            self.record_call(**call, reply=None, proposals=0, error=f"{type(exc).__name__}: {exc}")
            raise
        try:
            payload, malformed = answer_json(text)
        except Exception as exc:
            self.record_call(**call, reply=text, proposals=0, error=f"{type(exc).__name__}: {exc}")
            raise
        if malformed:
            # Nothing of a broken answer is read: an object nested in it is a fragment
            # (D-s42 epoch 5 read the first edit of a two-proposal answer as the whole
            # answer, dropped the other proposal and the diagnosis, and counted nothing
            # unreadable). One refused stand-in per proposal asked for, so the loop
            # counts them and asks again, with why, while it has attempts left.
            self.last_diagnosis = self.last_considered = ""
            why = (f"your whole answer was not valid JSON ({malformed}), so nothing in it "
                   "was read: none of its proposals, nor its diagnosis"
                   "; answer again with one valid JSON object")
            unreadable = [Proposal(edit={UNREADABLE_PROPOSAL: why}) for _ in range(max(1, n))]
            self.record_call(**call, reply=text, proposals=0, unreadable=len(unreadable),
                             diagnosis="", considered="", error=f"malformed answer: {malformed}")
            return unreadable
        self.last_diagnosis = str(payload.get("diagnosis", "") or "")
        self.last_considered = considered_text(payload.get("considered"))
        parent_ids = {parent.id for parent in parents}
        # A proposal that names no parent belongs to the pipeline the request showed
        # ("The pipeline you may edit" is ``ctx.parent``'s). Left unset, the loop
        # applied it to the first parent it was handed, which without the validation
        # gate is the seed: edits written for the leader landed on the seed.
        described = ctx.parent if ctx.parent in parent_ids else None

        proposals: list[Proposal] = []
        # An entry with no usable edit is refused, not dropped: the loop counts it and
        # the next request says why. Listed first, so a batch filled by the readable
        # ones cannot skip it.
        unreadable: list[Proposal] = []
        listed, missing = _answered_proposals(payload)
        if missing:
            unreadable.append(unreadable_proposal(payload, missing))
        for raw in listed:
            steps, why = _proposal_edits(raw)
            if not steps:
                unreadable.append(unreadable_proposal(raw, why))
                continue
            parent = raw.get("parent")
            proposals.append(
                Proposal(
                    edits=tuple(steps),
                    rationale=str(raw.get("rationale", "") or ""),
                    prediction=str(raw.get("prediction", "") or ""),
                    confidence=_confidence(raw.get("confidence")),
                    parent=parent if parent in parent_ids else described,
                )
            )
        proposals = [*unreadable, *proposals[: max(1, n)]]
        self.record_call(**call, reply=text, proposals=len(proposals) - len(unreadable),
                         unreadable=len(unreadable),
                         diagnosis=self.last_diagnosis, considered=self.last_considered,
                         error=None)
        return proposals


def _parse_edit(edit: Any) -> dict[str, Any] | None:
    if isinstance(edit, str):
        try:
            edit = json.loads(edit)
        except json.JSONDecodeError:
            return None
    return edit if isinstance(edit, dict) and "op" in edit else None


#: The keys of an answered proposal that are about the proposal, not its edit.
PROPOSAL_FIELDS = ("rationale", "prediction", "confidence", "parent")
#: The keys that hold an answered proposal's edit (``_proposal_edits``).
EDIT_KEYS = ("edit", "edits", "edit_json")


def _answered_proposals(payload: dict[str, Any]) -> tuple[list[Any], str]:
    """The proposal entries of an answer, and why there are none when it has no
    ``proposals`` list: its ``proposals`` (one object counts as one entry), else a
    single ``proposal``, else the answer itself when it is one proposal (an edit
    written at its top level, or an ``edit``/``edits`` beside the diagnosis). An empty
    ``proposals`` list is an answer that proposes nothing, not a malformed one."""
    if "proposals" in payload or "proposal" in payload:
        listed = payload["proposals"] if "proposals" in payload else payload["proposal"]
        if not listed:
            return [], ""
        return (list(listed) if isinstance(listed, (list, tuple)) else [listed]), ""
    if "op" in payload or any(key in payload for key in EDIT_KEYS):
        return [payload], ""
    return [], 'the answer has no "proposals" list'


def _op_fields(op: Any) -> tuple[str, ...] | None:
    """The fields the catalogue's ``op`` takes, or ``None`` for an op it does not have."""
    for entry in nn_edits.catalogue("broad"):
        if entry["op"] == op:
            return tuple(entry["fields"])
    return None


def _proposal_edits(raw: Any) -> tuple[list[dict[str, Any]], str]:
    """The edits one answered proposal carries, in order, or none and why: its
    ``edits`` list (a compound proposal), else its single ``edit``, else -- an edit
    written at the proposal's top level (``{"op": ..., "rationale": ...}``) -- the
    fields its op takes (``_op_fields``). An entry that is not an edit
    drops the whole proposal: applying the rest would score something the model did
    not propose. More edits than the run allows are kept, so the loop refuses them
    with the reason (``E-too-many-edits``) the next request shows."""
    if not isinstance(raw, dict):
        return [], f"a proposal is a JSON object, not {type(raw).__name__}"
    listed = raw.get("edits")
    if isinstance(listed, list) and listed:
        steps = [_parse_edit(edit) for edit in listed]
        bad = [index for index, step in enumerate(steps, 1) if step is None]
        if bad:
            return [], (f"entry {bad[0]} of its {len(steps)} edits is not an edit object "
                        "naming its op")
        return steps, ""  # type: ignore[return-value]
    if "op" in raw and not any(key in raw for key in EDIT_KEYS):
        # Only what the op takes: a proposal-level key ("id", "diagnosis") left in
        # would be read as the edit's own (an "id" naming a node to remove). An op the
        # catalogue lacks keeps its fields less the proposal's, for E-op to name.
        fields = _op_fields(raw["op"])
        return [{key: value for key, value in raw.items()
                 if (key == "op" or key in fields if fields is not None
                     else key not in PROPOSAL_FIELDS)}], ""
    single = _parse_edit(raw.get("edit") or raw.get("edit_json"))
    if single is not None:
        return [single], ""
    if any(key in raw for key in EDIT_KEYS):
        return [], "its edit is not an edit object naming its op"
    return [], "it carries no edit (no \"edit\", \"edits\" or \"op\")"


def _confidence(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def extract_json(text: str) -> dict[str, Any]:
    """The first JSON object in ``text``, fenced or bare.

    A model's answer is prose with an object in it often enough that refusing
    anything but a bare object would waste most of the tokens it just spent.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("the model returned nothing")
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    for chunk in [*fenced, text]:
        found = _first_object(chunk)
        if found is not None:
            return found
    raise ValueError(f"no JSON object in the model's answer: {text[:400]!r}")


def answer_json(text: str) -> tuple[dict[str, Any], str]:
    """A proposer's answer object and ``""``; or ``{}`` and why the answer's object is
    not valid JSON (the parser's message, where, and the text there). Unlike
    `extract_json`, an object nested inside a broken one is never taken for the answer:
    it is a fragment of it (one edit of one proposal). The whole reply is read first,
    its top-level objects by their own braces (strings skipped, so a code fence inside
    a prompt the answer proposes cannot end it); the answer is the object with a
    ``proposals`` list (`_is_answer`), as the answer format says, wherever it is (a
    fenced one too). Only when no such object parses is the reply read as a top-level
    array of proposal objects: every one of its last such array (`_top_array`), or
    refused whole; with such an array, no object inside or after it is ever read alone
    as the answer. When
    the reply's object is broken, a fenced object is read only when it is a whole
    answer: an example edit fenced before a cut-off answer is not the answer, and
    neither is an example array. No JSON object at all raises ``ValueError``, as
    `extract_json` does."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("the model returned nothing")
    found, problem = _outermost_object(text)
    if _is_answer(found):
        return found, ""  # type: ignore[return-value]
    for chunk in re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL):
        try:
            loaded = json.loads(chunk.strip())
        except json.JSONDecodeError:
            continue
        if _is_answer(loaded):
            return loaded, ""
    listed, refused = _top_array(text)
    if listed is not None:
        return {"proposals": listed}, ""
    if refused:
        return {}, refused
    if found is not None:
        return found, ""
    if problem:
        return {}, problem
    raise ValueError(f"no JSON object in the model's answer: {text[:400]!r}")


#: How a JSON object starts: a brace, then a key. Braces in prose (``{a, b}``) do not.
_JSON_START = re.compile(r"\{\s*(\"|\})")
#: How a top-level array of objects starts: a bracket, then a brace, then a key (prose
#: such as ``[{op, knob}]`` does not).
_ARRAY_START = re.compile(r"\[\s*\{\s*\"")


def _is_answer(loaded: Any) -> bool:
    """Whether a JSON value is a whole answer: an object with a ``proposals`` list."""
    return isinstance(loaded, dict) and isinstance(loaded.get("proposals"), list)


def _top_spans(text: str) -> list[tuple[str, int, int | None]]:
    """The reply's top-level JSON spans, in order, none inside another: ``("array" |
    "object", start, end)`` for each array of objects (`_ARRAY_START`) and each object
    (`_JSON_START`) outside them; ``end`` is ``None`` for one never closed, the last."""
    spans: list[tuple[str, int, int | None]] = []
    at = 0
    while True:
        found = [m for m in (_ARRAY_START.search(text, at), _JSON_START.search(text, at)) if m]
        if not found:
            return spans
        match = min(found, key=lambda m: m.start())
        kind = "array" if text[match.start()] == "[" else "object"
        end = _object_end(text, match.start(), *(("[", "]") if kind == "array" else ("{", "}")))
        spans.append((kind, match.start(), end))
        if end is None:
            return spans
        at = end + 1


def _top_array(text: str) -> tuple[list[dict[str, Any]] | None, str]:
    """The proposals of a reply whose JSON is a top-level array of objects (``[{...},
    {...}]``, bare or fenced): every entry of its LAST such array (a draft list before a
    final one is a draft), or none and why it cannot be read. ``(None, "")`` when the
    reply has no such array. A JSON object after the last array leaves the answer
    unclear, so the reply is refused rather than read as that object, or as one entry of
    the array (an object inside an array is never the answer here)."""
    spans = _top_spans(text)
    arrays = [span for span in spans if span[0] == "array"]
    if not arrays:
        return None, ""
    _kind, start, end = arrays[-1]
    if end is None:
        return None, (f"its top-level array opened at character {start + 1} is never "
                      f"closed (cut off? it ends: …{' '.join(text[-120:].split())})")
    after = [span for span in spans if span[0] == "object" and span[1] > end]
    if after:
        _k, at, closed = after[0]
        if closed is None:
            return None, (f"its object opened at character {at + 1} is never closed (cut "
                          f"off? it ends: …{' '.join(text[-120:].split())})")
        return None, (f"a JSON object (at character {at + 1}) follows its top-level list of "
                      "proposals, so which one is the answer is unclear; nothing was read. "
                      'Write one object with a "proposals" list')
    try:
        loaded = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        return None, (f"its top-level array is not valid JSON ({exc.msg} at line {exc.lineno} "
                      f"column {exc.colno})")
    if not loaded or not all(isinstance(entry, dict) for entry in loaded):
        return None, ("its top-level array holds something other than proposal objects; "
                      'write one object with a "proposals" list')
    return list(loaded), ""


def _outermost_object(text: str) -> tuple[dict[str, Any] | None, str]:
    """The answer among ``text``'s top-level ``{...}`` objects (scanned by their own
    braces, never into one), and why there is none.

    The last one carrying a ``proposals`` list is the answer; without one, the first
    object, unless a span that starts like a JSON object failed to parse: then the
    answer is broken and no lone fragment after it (an edit, a proposal) is read. An
    object left open ends the scan (everything after it is inside it). Braces in prose
    (``{a, b}``) are skipped."""
    first: dict[str, Any] | None = None
    answer: dict[str, Any] | None = None
    problem = ""
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        looks_json = _JSON_START.match(text, start) is not None
        if end is None:
            if not looks_json:
                start = text.find("{", start + 1)
                continue
            problem = problem or (f"its object opened at character {start + 1} is never "
                                  f"closed (cut off? it ends: …{' '.join(text[-120:].split())})")
            break
        try:
            loaded = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            if looks_json:
                at = start + exc.pos
                problem = problem or (
                    f"{exc.msg} at line {exc.lineno} column {exc.colno}, here: "
                    f"…{' '.join(text[max(start, at - 120):at].split())} ⟨HERE⟩ "
                    f"{' '.join(text[at:at + 60].split())}…")
        else:
            if isinstance(loaded, dict):
                first = loaded if first is None else first
                if _is_answer(loaded):
                    answer = loaded
        start = text.find("{", end + 1)
    if answer is not None:
        return answer, ""
    if problem:
        return None, problem
    return first, ""


def _object_end(text: str, start: int, opens: str = "{", closes: str = "}") -> int | None:
    """The index of the ``}`` closing the ``{`` at ``start`` (strings skipped), or
    ``None`` when it is never closed; ``opens``/``closes`` ``[``/``]`` for an array."""
    depth = 0
    in_string = escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opens:
            depth += 1
        elif char == closes:
            depth -= 1
            if depth == 0:
                return index
    return None


def _first_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        loaded = json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(loaded, dict):
                        return loaded
                    break
        start = text.find("{", start + 1)
    return None
