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
import json
import re
from pathlib import Path
from typing import Any


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


class CallMatcher:
    """Index A's calls.jsonl by (system, normalised_prompt) → list of results.

    Matching is by content regardless of order (A's solver calls ran
    concurrently). A per-key positional counter disambiguates genuinely
    identical (system, prompt) pairs.

    ``unused_recordings()`` tracks per-recording consumption: if A has 3
    identical calls and B only consumes 1, the 2 remaining are reported.
    """

    def __init__(self, calls_jsonl_path: str | Path) -> None:
        path = Path(calls_jsonl_path)
        self._index: dict[tuple[str, str], list[str]] = {}
        self._counters: dict[tuple[str, str], int] = {}
        # Track total recordings per key, so unused_recordings can report
        # individual un-consumed recordings even when the key was partially
        # consumed.
        self._totals: dict[tuple[str, str], int] = {}

        # Load and index all entries, sorted by timestamp within each key.
        entries_by_key: dict[tuple[str, str], list[tuple[float, str]]] = {}
        self._total_recordings = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                system = entry.get("system", "")
                raw_prompt = entry.get("prompt", "")
                normalised = normalize_recorded_prompt(raw_prompt)
                stdout_data = json.loads(entry["stdout"])
                result_text = stdout_data["result"]
                at = entry.get("at", 0.0)
                key = (system, normalised)
                entries_by_key.setdefault(key, []).append((at, result_text))
                self._total_recordings += 1

        for key, items in entries_by_key.items():
            items.sort(key=lambda t: t[0])  # chronological
            self._index[key] = [result for _, result in items]
            self._counters[key] = 0
            self._totals[key] = len(items)

    @property
    def total_recordings(self) -> int:
        """Total number of individual recordings loaded."""
        return self._total_recordings

    def lookup(self, system: str, prompt: str) -> str:
        """Look up the next recorded output for this (system, prompt) pair.

        B sends raw prompts (no wrapping), so no normalisation at lookup time.
        A's prompts are normalised at index time, producing the same raw form.
        """
        key = (system, prompt)
        if key not in self._index:
            # Find the nearest match for diagnostics.
            nearest_key = self._nearest(prompt)
            nearest_sys, nearest_prompt = nearest_key
            diff = "\n".join(difflib.unified_diff(
                prompt.splitlines(),
                nearest_prompt.splitlines(),
                fromfile="request",
                tofile="nearest_recorded",
                lineterm="",
            ))
            raise ReplayMismatch(system, prompt, nearest_sys, nearest_prompt, diff)

        outputs = self._index[key]
        idx = self._counters[key]
        if idx >= len(outputs):
            raise ReplayExhausted(system, prompt, len(outputs))

        self._counters[key] = idx + 1
        return outputs[idx]

    def consumed_count(self) -> int:
        """Total number of individual recordings consumed so far."""
        return sum(self._counters.values())

    def unused_recordings(self) -> list[tuple[str, str, int]]:
        """Return (system, prompt, remaining_count) for each key with unused recordings.

        Tracks per-recording consumption: if A has 3 identical calls for a key
        and B consumed only 1, reports (system, prompt, 2) — the 2 remaining.
        A key B never asked for at all is reported with remaining_count = total.
        """
        unused: list[tuple[str, str, int]] = []
        for key, total in self._totals.items():
            consumed = self._counters[key]
            remaining = total - consumed
            if remaining > 0:
                unused.append((key[0], key[1], remaining))
        return unused

    def _nearest(self, prompt: str) -> tuple[str, str]:
        """Find the nearest (system, prompt) key by SequenceMatcher ratio."""
        best_ratio = -1.0
        best_key = ("", "")
        for key in self._index:
            ratio = difflib.SequenceMatcher(None, prompt, key[1]).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_key = key
        return best_key


# ── replay client ──────────────────────────────────────────────────────────


class ReplayClient:
    """Model client that replays A's recorded outputs.

    Implements the executor's agent-node protocol:
    ``for_node()``, ``with_tool_policy()``, ``with_wall()``, ``route()``,
    and ``complete()`` accepting ``call_key`` and ``workdir`` kwargs.
    """

    def __init__(self, matcher: CallMatcher) -> None:
        self.matcher = matcher
        self.match_log: list[dict[str, Any]] = []

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        call_key: Any = None,
        workdir: Any = None,
        cwd: Any = None,
        env: Any = None,
        confine: bool = False,
        **kwargs: Any,
    ) -> str:
        """Answer a request from A's recordings."""
        idx = self.matcher._counters.get((system, prompt), 0)
        result = self.matcher.lookup(system, prompt)
        self.match_log.append({
            "key": (system[:80], prompt[:80]),
            "index": idx,
            "call_key": call_key,
        })
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
