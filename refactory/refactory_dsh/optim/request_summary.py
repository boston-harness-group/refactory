"""Computed summary for the proposer's request: aggregate tables from the run's own
records, inserted after the score/policy header.

Five tables, each stating its source:

1. **Noise**: rescore spreads -- every candidate scored more than once on the same items.
2. **Item stability**: per-item pass rates across all scorings in this run.
3. **By group**: pass rates per metadata group, when the task declares one.
4. **By step**: for the candidate being edited, where items first went wrong and where
   steps gave no answer.
5. **History by kind of change**: every evaluated candidate's edits classified, with the
   score change against its parent and the gate outcome.

Domain-free: nothing here is specific to any benchmark or task.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .base import fmt_score


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "(none)"
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    rule = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def _cell(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text.replace("|", "\\|").replace("\n", " ")


# ── 1. noise ──────────────────────────────────────────────────────────────────


def _noise_lines(noise: Mapping[str, Any] | None) -> list[str]:
    """Rescore spreads and per-item variability, from repeated scorings."""
    if noise is None:
        return ["No repeat scoring of one graph yet; differences of a few items may be "
                "noise."]
    spreads = noise.get("spreads") or []
    if not spreads:
        return ["No repeat scoring of one graph yet; differences of a few items may be "
                "noise."]
    lines = ["Source: repeated train scorings of the same candidate graph in this run.",
             _table(["candidate", "scorings", "spread (items)"],
                    [[s["candidate"], s["epochs"], fmt_score(s["spread"])]
                     for s in spreads])]
    return lines


# ── 2. item stability ────────────────────────────────────────────────────────


def _item_stability_lines(stability: Mapping[str, Any] | None) -> list[str]:
    if stability is None:
        return []
    always = stability.get("always_pass", 0)
    never = stability.get("never_pass", 0)
    sometimes = stability.get("sometimes_pass") or []
    total = always + never + len(sometimes)
    lines = [f"Source: pass/fail of each train item across every candidate scored in "
             f"this run ({stability.get('scorings', '?')} scoring(s), {total} item(s)).",
             f"Always pass: {always}. Never pass: {never}. "
             f"Sometimes pass: {len(sometimes)}."]
    if sometimes:
        lines.append("Items that sometimes pass are where the outcome is uncertain "
                     "(the same item passes under some candidates and fails under others):")
        lines.append(_table(
            ["item", "pass rate", "scorings"],
            [[s["item"], f"{s['pass_rate']:.0%}", s["scorings"]] for s in sometimes[:30]]))
    return lines


# ── 3. by group ───────────────────────────────────────────────────────────────


def _by_group_lines(by_group: Mapping[str, Any] | None) -> list[str]:
    if by_group is None or not by_group.get("groups"):
        return []
    field = by_group["field"]
    groups = by_group["groups"]
    lines = [f"Source: pass rate per `{field}` group for the candidate being edited "
             "and the best candidate.",
             _table(["group", "items", "edited", "best"],
                    [[g["group"], g.get("items", ""), fmt_score(g.get("edited")),
                      fmt_score(g.get("best"))] for g in groups])]
    return lines


# ── 4. by step ────────────────────────────────────────────────────────────────


def _by_step_lines(by_step: Mapping[str, Any] | None) -> list[str]:
    if by_step is None or not by_step.get("steps"):
        return []
    steps = by_step["steps"]
    lines = ["Source: for the candidate being edited, analyst attribution and no-answer "
             "causes per step."]
    headers = ["step", "first wrong", "no answer"]
    time_headers: list[str] = []
    if any(s.get("median_s") is not None for s in steps):
        time_headers = ["median s", "p90 s"]
    lines.append(_table(
        [*headers, *time_headers],
        [[s["step"], s.get("first_wrong", ""), s.get("no_answer", ""),
          *(([fmt_score(s.get("median_s")), fmt_score(s.get("p90_s"))]
             if time_headers else []))]
         for s in steps]))
    return lines


# ── 5. history by kind of change ──────────────────────────────────────────────


def _change_history_lines(history: Sequence[Mapping[str, Any]] | None) -> list[str]:
    if not history:
        return []
    lines = ["Source: every evaluated candidate's own edits classified by kind, with the "
             "train score change against its parent and the gate outcome.",
             _table(["kind", "count", "mean change", "best", "worst"],
                    [[h["kind"], h["count"], _signed(h.get("mean_change")),
                      _signed(h.get("best")), _signed(h.get("worst"))]
                     for h in history])]
    return lines


def _signed(value: Any) -> str:
    if value is None or not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:+.4g}"


# ── the section ───────────────────────────────────────────────────────────────


def section_lines(summary: Mapping[str, Any] | None) -> list[str]:
    """The whole computed summary section for the proposal request; nothing without data."""
    if not summary:
        return []
    lines = ["## Computed summary"]
    subsections = [
        ("### Noise (rescore spread)", _noise_lines(summary.get("noise"))),
        ("### Stability across items", _item_stability_lines(summary.get("item_stability"))),
        ("### By group", _by_group_lines(summary.get("by_group"))),
        ("### By step", _by_step_lines(summary.get("by_step"))),
        ("### History by kind of change",
         _change_history_lines(summary.get("change_history"))),
    ]
    for heading, body in subsections:
        if body:
            lines += ["", heading, *body]
    return lines


# ── build from loop state ─────────────────────────────────────────────────────


def build_noise(records_by_epoch: Mapping[str, Mapping[int, Sequence[Mapping]]],
                train_item_records: Mapping[str, Sequence[Mapping]],
                ) -> dict[str, Any] | None:
    """Noise data from repeated scorings of the same candidate."""
    spreads = []
    for cid, epochs in records_by_epoch.items():
        if len(epochs) < 2:
            continue
        scores = []
        for records in epochs.values():
            values = [float(r["score"]) for r in records
                      if isinstance(r.get("score"), (int, float))]
            if values:
                scores.append(sum(values) / len(values))
        if len(scores) >= 2:
            spreads.append({"candidate": cid, "epochs": len(scores),
                            "spread": round(max(scores) - min(scores), 4)})
    return {"spreads": spreads} if spreads else None


def build_item_stability(train_item_records: Mapping[str, Sequence[Mapping]],
                         ) -> dict[str, Any] | None:
    """Item stability across all candidates scored in this run."""
    item_results: dict[str, list[bool]] = {}
    n_scorings = 0
    for records in train_item_records.values():
        n_scorings += 1
        for r in records:
            item_id = str(r.get("id", ""))
            if not item_id:
                continue
            item_results.setdefault(item_id, []).append(bool(r.get("passed")))
    if not item_results:
        return None
    always_pass = 0
    never_pass = 0
    sometimes: list[dict[str, Any]] = []
    for item_id, results in sorted(item_results.items()):
        passes = sum(results)
        total = len(results)
        rate = passes / total if total else 0
        if rate == 1.0:
            always_pass += 1
        elif rate == 0.0:
            never_pass += 1
        else:
            sometimes.append({"item": item_id, "pass_rate": round(rate, 3),
                              "scorings": total})
    sometimes.sort(key=lambda s: abs(s["pass_rate"] - 0.5))
    return {"always_pass": always_pass, "never_pass": never_pass,
            "sometimes_pass": sometimes, "scorings": n_scorings}


def build_by_group(item_records: Sequence[Mapping[str, Any]],
                   best_records: Sequence[Mapping[str, Any]] | None,
                   ) -> dict[str, Any] | None:
    """Pass rate per metadata group for the edited and best candidate."""
    field = _find_group_field(item_records)
    if field is None:
        return None
    edited_groups = _group_rates(item_records, field)
    best_groups = _group_rates(best_records or [], field) if best_records else {}
    groups = []
    for name in sorted(edited_groups):
        eg = edited_groups[name]
        bg = best_groups.get(name, {})
        groups.append({"group": name, "items": eg["total"],
                       "edited": eg["rate"], "best": bg.get("rate")})
    return {"field": field, "groups": groups} if groups else None


def _find_group_field(records: Sequence[Mapping[str, Any]]) -> str | None:
    """Find a categorical metadata field from item records."""
    counts: dict[str, int] = {}
    for r in records:
        meta = r.get("item_meta") if isinstance(r.get("item_meta"), Mapping) else {}
        for key, value in meta.items():
            if isinstance(value, str) and value:
                counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    best = max(counts, key=lambda k: counts[k])
    if counts[best] < 2:
        return None
    n_distinct = len({(r.get("item_meta") or {}).get(best) for r in records
                      if isinstance(r.get("item_meta"), Mapping)
                      and (r.get("item_meta") or {}).get(best)})
    if n_distinct < 2 or n_distinct > len(records):
        return None
    return best


def _group_rates(records: Sequence[Mapping[str, Any]], field: str
                 ) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for r in records:
        meta = r.get("item_meta") if isinstance(r.get("item_meta"), Mapping) else {}
        value = meta.get(field)
        if not isinstance(value, str) or not value:
            value = "(none)"
        g = groups.setdefault(value, {"passed": 0, "total": 0})
        g["total"] += 1
        if r.get("passed"):
            g["passed"] += 1
    return {name: {**g, "rate": round(g["passed"] / g["total"], 3) if g["total"] else 0}
            for name, g in groups.items()}


def build_by_step(item_records: Sequence[Mapping[str, Any]],
                  findings: Sequence[Mapping[str, Any]] | None,
                  ) -> dict[str, Any] | None:
    """Per-step attribution from analyst findings and no-answer causes."""
    step_data: dict[str, dict[str, int]] = {}
    for r in item_records:
        for step, info in sorted((r.get("missing_steps") or {}).items()):
            if isinstance(info, dict):
                s = step_data.setdefault(step, {"first_wrong": 0, "no_answer": 0})
                s["no_answer"] += 1
    if findings:
        for f in findings:
            steps = f.get("steps") if isinstance(f.get("steps"), Mapping) else {}
            first = steps.get("first_wrong")
            if isinstance(first, str) and first:
                items = f.get("items") or []
                s = step_data.setdefault(first, {"first_wrong": 0, "no_answer": 0})
                s["first_wrong"] += len(items)
    if not step_data:
        return None
    return {"steps": [{"step": name, **counts}
                      for name, counts in sorted(step_data.items(),
                                                  key=lambda x: -(x[1]["first_wrong"]
                                                                   + x[1]["no_answer"]))]}


def build_change_history(history: Sequence[Mapping[str, Any]],
                         ) -> list[dict[str, Any]] | None:
    """Every evaluated candidate's edits classified by kind, with score change."""
    from .feedback import change_kind

    kinds: dict[str, list[float]] = {}
    for entry in history:
        edits = entry.get("edits") or entry.get("own") or []
        kind = change_kind(edits)
        score = entry.get("score")
        parent_score = entry.get("parent_score")
        if isinstance(score, (int, float)) and isinstance(parent_score, (int, float)):
            delta = score - parent_score
            kinds.setdefault(kind, []).append(delta)
        else:
            kinds.setdefault(kind, [])
    if not kinds:
        return None
    result = []
    for kind, deltas in sorted(kinds.items(), key=lambda x: -len(x[1])):
        entry: dict[str, Any] = {"kind": kind, "count": len(deltas)}
        if deltas:
            entry["mean_change"] = round(sum(deltas) / len(deltas), 4)
            entry["best"] = round(max(deltas), 4)
            entry["worst"] = round(min(deltas), 4)
        result.append(entry)
    return result or None
