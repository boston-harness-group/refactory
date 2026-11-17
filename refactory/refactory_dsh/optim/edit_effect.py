"""What a structural edit did to the items it changed, from the item records alone.

A child's item counts ("3 fixed, 7 broken") say that an edit moved items, not how. When
the child ran steps its parent did not, the records say how: which of the new steps ran
on the broken (and the fixed) items, what a gate among them answered, which step ran
more than once, and -- where the Task can read a reply as an answer
(`task.reply_judge`, `Task.reply_correct`) -- whether that step's first reply was
already right before it ran again. Computed, never asked of a model, and generic: steps
are compared by name, verdicts are the gates' own, and only the Task reads a reply.

On S1-s45 the loop a proposer added broke 8 items; on 5 of them the solver's first reply
was right, a judge sent it back, and its next reply was wrong. The request had said only
"broken 8"; the proposer and memory then guessed at why.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Mapping, Sequence

#: How a reply is read as an item's answer: ``(reply, metadata) -> bool | None``.
ReplyJudge = Callable[[str, Mapping[str, Any]], "bool | None"]


def _by_item(records: Sequence[Mapping[str, Any]] | None) -> dict[str, Mapping[str, Any]]:
    return {str(r.get("id")): r for r in records or () if isinstance(r, Mapping)}


def _agents(record: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Each model call of an item, in order: ``(step, reply)``."""
    return [(str(e.get("node")), str(e.get("response") or ""))
            for e in record.get("exchanges") or () if isinstance(e, Mapping)]


def _gates(record: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Each code-gate visit of an item, in order: ``(gate, verdict)``."""
    return [(str(s.get("node")), str(s.get("verdict") or ""))
            for s in record.get("steps") or ()
            if isinstance(s, Mapping) and s.get("kind") == "gate"]


def _steps(record: Mapping[str, Any]) -> set[str]:
    return {node for node, _ in _agents(record)} | {node for node, _ in _gates(record)}


def _group(ids: Sequence[str], parent: Mapping[str, Mapping[str, Any]],
           child: Mapping[str, Mapping[str, Any]], judge: ReplyJudge | None) -> dict[str, Any]:
    new: Counter[str] = Counter()
    said: dict[str, Counter[str]] = {}
    reran: Counter[str] = Counter()
    first_right: Counter[str] = Counter()
    then_changed: Counter[str] = Counter()
    never_right: Counter[str] = Counter()
    for item in ids:
        before, after = parent[item], child[item]
        added = _steps(after) - _steps(before)
        new.update(added)
        for gate in {g for g, _ in _gates(after) if g in added}:
            said.setdefault(gate, Counter()).update(
                {verdict for g, verdict in _gates(after) if g == gate and verdict})
        replies: dict[str, list[str]] = {}
        for node, reply in _agents(after):
            replies.setdefault(node, []).append(reply)
        meta = after.get("metadata") if isinstance(after.get("metadata"), Mapping) else {}
        for node, texts in replies.items():
            if len(texts) < 2:
                continue
            reran[node] += 1
            if judge is None:
                continue
            first, last = judge(texts[0], meta), judge(texts[-1], meta)
            if first:
                first_right[node] += 1
                if last is False:
                    then_changed[node] += 1
            elif first is False and not any(judge(t, meta) for t in texts):
                never_right[node] += 1
    return {"items": len(ids), "new": dict(new), "said": {g: dict(c) for g, c in said.items()},
            "reran": dict(reran), "first_right": dict(first_right),
            "then_changed": dict(then_changed), "never_right": dict(never_right)}


def edit_effect(parent_records: Sequence[Mapping[str, Any]] | None,
                child_records: Sequence[Mapping[str, Any]] | None,
                judge: ReplyJudge | None = None) -> dict[str, Any]:
    """For the shared items a child broke (parent right, child wrong) and fixed (the
    reverse): the steps new in the child that ran on them, the verdicts those gates gave
    (per verdict, the items it was given on at least once), the steps that ran more than
    once, and with ``judge``, of those, on how many items the first reply was already
    right, then changed to a wrong one, or was never right. Empty when the child ran no
    step its parent did not on any of those items."""
    parent, child = _by_item(parent_records), _by_item(child_records)
    shared = [i for i in child if i in parent]
    broken = [i for i in shared if parent[i].get("passed") and not child[i].get("passed")]
    fixed = [i for i in shared if not parent[i].get("passed") and child[i].get("passed")]
    out = {kind: _group(ids, parent, child, judge)
           for kind, ids in (("broken", broken), ("fixed", fixed)) if ids}
    return out if any(group["new"] for group in out.values()) else {}


#: What each group is, as the lines say it.
_GROUPS = {"broken": "broken item(s) (parent right, child wrong)",
           "fixed": "fixed item(s) (parent wrong, child right)"}


def effect_lines(effect: Mapping[str, Any]) -> list[str]:
    """`edit_effect` as a request's lines: one per group, each count of that group's
    items. Nothing when the edit added no step that ran on them."""
    lines = []
    for kind, label in _GROUPS.items():
        group = effect.get(kind)
        if not group:
            continue
        n = group["items"]
        parts = []
        if group["new"]:
            parts.append("steps new in the child ran on " + ", ".join(
                f"{count} ({step})" for step, count in sorted(group["new"].items(),
                                                             key=lambda kv: (-kv[1], kv[0]))))
        for gate, verdicts in sorted(group["said"].items()):
            parts.append(f"gate {gate} said " + ", ".join(
                f"{verdict} on {count}" for verdict, count in sorted(verdicts.items(),
                                                                     key=lambda kv: -kv[1])))
        for step, count in sorted(group["reran"].items()):
            parts.append(f"{step} ran more than once on {count}")
            if step in group["first_right"] or step in group["never_right"]:
                parts.append(
                    f"{step}'s first reply was already right on "
                    f"{group['first_right'].get(step, 0)}, and changed to a wrong one on "
                    f"{group['then_changed'].get(step, 0)} of those; never right on "
                    f"{group['never_right'].get(step, 0)} (the task's own reading of a reply)")
        lines.append(f"How the edit moved its {n} {label}: " + "; ".join(parts) + ".")
    return lines
