"""What a structural edit did to the items it changed, from the item records alone.

A child's item counts ("3 fixed, 7 broken") say that an edit moved items, not how. When
the child has steps its parent's graph does not, the records say how: which of those
new steps ran on the broken (and the fixed) items, what a gate among them answered,
which step the child called more often than the parent did on the same item, and --
where the Task can read a reply as an answer (`task.reply_judge`, `Task.reply_correct`)
-- whether that step's first reply was already right before it ran again. Computed,
never asked of a model, and generic: steps are compared by name, verdicts are the
gates' own, and only the Task reads a reply.

On S1-s45 the loop a proposer added broke 8 items; on 5 of them the solver's first reply
was right, a judge sent it back, and its next reply was wrong. The request had said only
"broken 8"; the proposer and memory then guessed at why.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Collection, Mapping, Sequence

#: How a reply is read as an item's answer: ``(reply, metadata) -> bool | None``.
ReplyJudge = Callable[[str, Mapping[str, Any]], "bool | None"]


def _by_item(records: Sequence[Mapping[str, Any]] | None) -> dict[str, Mapping[str, Any]]:
    return {str(r.get("id")): r for r in records or () if isinstance(r, Mapping)}


def _agents(record: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Each model call of an item, in order: ``(step, reply)``."""
    return [(str(e.get("node")), str(e.get("response") or ""))
            for e in record.get("exchanges") or () if isinstance(e, Mapping)]


def _visits(record: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """Each op or code-gate visit of an item, in order: ``(step, kind, verdict)``."""
    return [(str(s.get("node")), str(s.get("kind") or ""), str(s.get("verdict") or ""))
            for s in record.get("steps") or () if isinstance(s, Mapping)]


def _ran(record: Mapping[str, Any]) -> set[str]:
    return {node for node, _ in _agents(record)} | {node for node, _, _ in _visits(record)}


class _Reader:
    """The Task's reading of replies, counting the ones it could not read and keeping the
    first error a reading raised, so neither drops out unsaid."""

    def __init__(self, judge: ReplyJudge | None) -> None:
        self.judge = judge
        self.errors: list[str] = []

    def __call__(self, reply: str, meta: Mapping[str, Any]) -> bool | None:
        if self.judge is None:
            return None
        try:
            verdict = self.judge(reply, meta)
        except Exception as exc:  # noqa: BLE001 - said in the line (`effect_lines`)
            self.errors.append(f"{type(exc).__name__}: {exc}")
            return None
        return None if verdict is None else bool(verdict)


def _group(ids: Sequence[str], parent: Mapping[str, Mapping[str, Any]],
           child: Mapping[str, Mapping[str, Any]], new_steps: Collection[str],
           read: _Reader) -> dict[str, Any]:
    new: Counter[str] = Counter()
    said: dict[str, Counter[str]] = {}
    reran: Counter[str] = Counter()
    unread_runs: Counter[str] = Counter()
    first_right: Counter[str] = Counter()
    then_changed: Counter[str] = Counter()
    never_right: Counter[str] = Counter()
    unread: Counter[str] = Counter()
    parent_failed = 0
    for item in ids:
        before, after = parent[item], child[item]
        if before.get("error_text"):
            # The parent's run of the item failed: nothing ran to compare with.
            parent_failed += 1
            continue
        new.update(step for step in _ran(after) if step in new_steps)
        for gate in {node for node, kind, _ in _visits(after) if kind == "gate"} & set(new_steps):
            said.setdefault(gate, Counter()).update(
                {verdict for node, kind, verdict in _visits(after)
                 if node == gate and kind == "gate" and verdict})
        calls: dict[str, list[str]] = {}
        for node, reply in _agents(after):
            calls.setdefault(node, []).append(reply)
        before_calls = Counter(node for node, _ in _agents(before))
        meta = after.get("metadata") if isinstance(after.get("metadata"), Mapping) else {}
        for node, replies in calls.items():
            if len(replies) < 2 or len(replies) <= before_calls[node]:
                continue
            reran[node] += 1
            if before_calls[node] > 1:
                # The parent already called it more than once on this item (a step in a
                # nested pass, a loop of its own): which reply is "first" is not known.
                unread_runs[node] += 1
                continue
            if read.judge is None:
                continue
            readings = [read(reply, meta) for reply in replies]
            first, last = readings[0], readings[-1]
            if first is None:
                unread[node] += 1
            elif first:
                first_right[node] += 1
                if last is False:
                    then_changed[node] += 1
            elif not any(readings):
                never_right[node] += 1
    return {"items": len(ids), "parent_failed": parent_failed, "new": dict(new),
            "said": {g: dict(c) for g, c in said.items()}, "reran": dict(reran),
            "unread_runs": dict(unread_runs), "first_right": dict(first_right),
            "then_changed": dict(then_changed), "never_right": dict(never_right),
            "unread": dict(unread)}


def _says_something(group: Mapping[str, Any]) -> bool:
    return bool(group["new"] or group["said"] or group["reran"] or group["parent_failed"])


def edit_effect(parent_records: Sequence[Mapping[str, Any]] | None,
                child_records: Sequence[Mapping[str, Any]] | None,
                judge: ReplyJudge | None = None, *,
                new_steps: Collection[str]) -> dict[str, Any]:
    """For the shared items a child broke (parent right, child wrong) and fixed (the
    reverse): which of ``new_steps`` (the child graph's steps its parent's lacks) ran on
    them, the verdicts those gates gave (per verdict, the items it was given on at least
    once), the steps the child called more often than the parent on the same item (and
    at least twice), and with ``judge``, of those, on how many the first reply was
    already right, then changed to a wrong one, or was never right. An item whose
    parent run failed with an error is left out and counted. A group with nothing to
    say is left out; empty when neither has anything, or the edit added no step."""
    if not new_steps:
        return {}
    parent, child = _by_item(parent_records), _by_item(child_records)
    shared = [i for i in child if i in parent]
    broken = [i for i in shared if parent[i].get("passed") and not child[i].get("passed")]
    fixed = [i for i in shared if not parent[i].get("passed") and child[i].get("passed")]
    read = _Reader(judge)
    groups = {kind: _group(ids, parent, child, set(new_steps), read)
              for kind, ids in (("broken", broken), ("fixed", fixed)) if ids}
    out: dict[str, Any] = {kind: g for kind, g in groups.items() if _says_something(g)}
    if not out:
        return {}
    out["reads_replies"] = judge is not None
    if read.errors:
        out["read_errors"] = {"count": len(read.errors), "first": read.errors[0]}
    return out


#: What each group is, as the lines say it.
_GROUPS = {"broken": "broken item(s) (parent right, child wrong)",
           "fixed": "fixed item(s) (parent wrong, child right)"}


def _counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{count} ({step})"
                     for step, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def effect_lines(effect: Mapping[str, Any]) -> list[str]:
    """`edit_effect` as a request's lines: one per group with something to say, each
    count of that group's items. Nothing when the edit added no step that ran on them."""
    lines = []
    for kind, label in _GROUPS.items():
        group = effect.get(kind)
        if not group:
            continue
        parts = []
        if group["new"]:
            parts.append("steps new in the child ran on " + _counts(group["new"]))
        for gate, verdicts in sorted(group["said"].items()):
            parts.append(f"gate {gate} gave " + " and ".join(
                f"{verdict} on {count}" for verdict, count in sorted(
                    verdicts.items(), key=lambda kv: -kv[1]))
                + " (an item counted under each verdict it got at least once)")
        for step, runs in sorted(group["reran"].items()):
            parts.append(f"{step} ran more often than in the parent on {runs}")
            if group["unread_runs"].get(step):
                parts.append(f"on {group['unread_runs'][step]} of those the parent already "
                             f"called {step} more than once, so its first reply is not known")
            readable = runs - group["unread_runs"].get(step, 0)
            if not readable:
                continue
            if group["unread"].get(step, 0) == readable and effect.get("reads_replies"):
                parts.append(f"{step}'s first reply was not readable as an answer on any of "
                             f"those {readable}")
                continue
            if not effect.get("reads_replies"):
                parts.append(f"whether {step}'s first reply was right is not known: the task "
                             "reads no reply as an answer")
                continue
            parts.append(
                f"of those {readable}, {step}'s first reply was already right on "
                f"{group['first_right'].get(step, 0)} and changed to a wrong one on "
                f"{group['then_changed'].get(step, 0)} of those; never right on "
                f"{group['never_right'].get(step, 0)}"
                + (f"; not readable as an answer on {group['unread'][step]}"
                   if group["unread"].get(step) else "")
                + " (the task's own reading of a reply)")
        if group["parent_failed"]:
            parts.append(f"{group['parent_failed']} left out: the parent's run of the item "
                         "failed with an error, so nothing ran to compare with")
        lines.append(f"How the edit moved its {group['items']} {label}: "
                     + "; ".join(parts) + ".")
    if lines and effect.get("read_errors"):
        errors = effect["read_errors"]
        lines.append(f"The task's reading of a reply failed {errors['count']} time(s), "
                     f"first: {errors['first']}.")
    return lines
