"""What the optimizer learns across a run, beyond its recent candidates.

Three request sections, each behind a ``LoopConfig`` switch:

- **Every change so far, by kind** (``run_ledger``): every judged candidate of the run
  with the kind of change it made (``change_kind``), what it touched, its train
  fixed/broken counts against its parent and the gate's outcome word, after a tally by
  kind. Without it the optimizer sees only its last few candidates and cannot tell that
  a kind of change has failed every time it was tried.
- **Your lessons so far** (``lessons``): the optimizer's own short notes on which
  changes help here, written in its answer and shown back next round, so what it
  learned is carried forward rather than re-derived (SkillOpt's slow update does this
  for its skill text).
- **Did the added steps contribute?** (``step_review``): for each step a candidate of
  the last round added, the train items it fixed and broke, how often the final answer
  (what the grader read) changed, and a reviewer's verdict on whether the step's output holds anything
  its inputs lacked.

Domain-free by construction: they read the graph, a score per item and each step's
output as plain text. No answer or option is parsed, and only train records and the
gate's outcome words are used -- never a validation item or score.
"""

from __future__ import annotations

import json
import re
import statistics
from typing import Any, Mapping, Sequence

from .base import _by_item, _clip, _clip_middle, _table, edit_change

#: The kinds a change is filed under, in the order the tally lists them.
CHANGE_KINDS = ("prompt/setting", "added agent step", "added code step", "split",
                "loop/gate", "parallel", "removal/rewire", "other")
#: When one proposal makes several kinds of change, the kind it is filed under: the
#: first of these it makes (the most specific structure first).
_KIND_PRIORITY = ("split", "parallel", "loop/gate", "added code step", "added agent step",
                  "removal/rewire", "other", "prompt/setting")
_KNOB_OPS = {"set_knob", "add_knob", "remove_knob"}
_REWIRE_OPS = {"bypass", "remove_node", "remove_edge", "add_edge", "reorder"}
_NODE_OPS = {"insert_after", "insert_before", "add_node", "wrap", "duplicate"}
#: The gate's outcome words that mean a candidate was kept (judged and kept, the
#: current one, or -- under a pool -- in it or once in it).
KEPT_WORDS = ("kept", "current", "in pool", "left pool")

#: Rows "Every change so far, by kind" lists in full, newest kept; older ones are only
#: counted in the tally. And the most characters its table may take.
LEDGER_ROWS = 40
LEDGER_CHARS = 9000
LEDGER_RATIONALE_CHARS = 140
LEDGER_TOUCHED = 4

#: The most characters of lessons kept and shown.
LESSONS_CHARS = 1500

#: Steps reviewed per request and train items each review reads. A review reads each
#: item's texts (the step's input as sent, its output, the next reader's output) whole;
#: only when together they pass ``STEP_REVIEW_ROOM`` characters (the analysts' room,
#: ``analysts.ANALYST_ROOM``) are the longest cut, head and tail, to one length and never
#: below ``STEP_REVIEW_FLOOR``, each cut said (`review_prompt`).
STEP_REVIEW_STEPS = 8
STEP_REVIEW_ITEMS = 4
STEP_REVIEW_ROOM = 400000
STEP_REVIEW_FLOOR = 2000
STEP_REVIEW_QUOTE_CHARS = 240
STEP_REVIEW_NOTE_CHARS = 300
#: The most characters "Did the added steps contribute?" gives full entries; the steps
#: past it get one short line each.
STEP_REVIEW_CHARS = 9000
#: The verdicts a review may give.
STEP_VERDICTS = ("adds_new_work", "restates_inputs", "contradicts_inputs", "no_effect")
#: The verdicts of a decision step's review (`decision_step`), from how often it
#: decided as its inputs recommended (`parse_decision_review`).
DECISION_VERDICTS = ("follows_inputs", "overrides_inputs", "no_recommendation")


# ── change kinds ────────────────────────────────────────────────────────────


def _node_kind(node: Any, state: Mapping[str, Any] | None = None) -> str:
    """``agent``, ``code`` (an op that runs code the edit wrote), ``gate`` or another
    node kind, from a node object or, by id, from ``state``'s graph."""
    from refactory_dsh import snippet

    if isinstance(node, str) and state is not None:
        node = ((state.get("graph") or {}).get("nodes") or {}).get(node)
    if not isinstance(node, Mapping):
        return ""
    kind = str(node.get("kind") or "")
    if kind == "op" and (node.get("code") or node.get("call") in (snippet.RUN, snippet.GATE)):
        return "code"
    return kind


def _op_kind(edit: Mapping[str, Any], state: Mapping[str, Any] | None) -> str:
    op = edit.get("op")
    if op in _KNOB_OPS:
        return "prompt/setting"
    if op == "split":
        return "split"
    if op == "parallel":
        return "parallel"
    if op == "loop":
        return "loop/gate"
    if op in _REWIRE_OPS:
        return "removal/rewire"
    if op in _NODE_OPS:
        # A duplicate copies its target: the copy's kind, or the target's.
        kind = _node_kind(edit.get("node") if op != "duplicate" else edit.get("new_id"), state)
        if not kind and op == "duplicate":
            kind = _node_kind(edit.get("target"), state)
        if kind == "gate":
            return "loop/gate"
        if kind == "code":
            return "added code step"
        if kind == "agent":
            return "added agent step"
    return "other"


def change_kind(edits: Any, state: Mapping[str, Any] | None = None) -> str:
    """The kind of change a candidate's own edits make (``CHANGE_KINDS``), from the edit
    ops and the kinds of the nodes they add (looked up in ``state``, the candidate's
    own, when an edit names a node by id): one kind per candidate, the most specific
    structure it makes (``_KIND_PRIORITY``)."""
    steps = [e for e in (edits if isinstance(edits, (list, tuple)) else [edits])
             if isinstance(e, Mapping)]
    if not steps:
        return "other"
    kinds = {_op_kind(edit, state) for edit in steps}
    return next(kind for kind in _KIND_PRIORITY if kind in kinds)


def touched(edits: Any) -> list[str]:
    """What a candidate's own edits touch, in order, each once: the knobs it sets and
    the steps it adds, removes, rewires or targets."""
    out: list[str] = []

    def add(name: Any) -> None:
        if isinstance(name, str) and name and name not in out:
            out.append(name)

    for edit in edits if isinstance(edits, (list, tuple)) else [edits]:
        if not isinstance(edit, Mapping):
            continue
        if edit.get("op") in _KNOB_OPS:
            add(edit.get("name"))
            continue
        for field in ("node", "gate"):
            if isinstance(edit.get(field), Mapping):
                add(edit[field].get("id"))
        for field in ("into", "branches"):
            for node in edit.get(field) or ():
                if isinstance(node, Mapping):
                    add(node.get("id"))
        for field in ("new_id", "id", "target", "first", "second"):
            add(edit.get(field))
        if edit.get("op") in ("add_edge", "remove_edge") and edit.get("source"):
            add(f"{edit.get('source')}->{edit.get('target')}")
    return out


# ── the run ledger ──────────────────────────────────────────────────────────


def ledger_entry(candidate: Any, parent: Any, *, record: Mapping[str, Any] | None,
                 parent_records: Sequence[Mapping[str, Any]] | None,
                 child_records: Sequence[Mapping[str, Any]] | None,
                 gate: str | None, same_seed: bool = True,
                 integrity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One row of "Every change so far, by kind": train records and the gate's word
    only. A child whose parent was not scored with its round's seed (``same_seed``
    false) gets no fixed/broken counts: the difference would be partly the seed's.
    ``integrity`` is the integrity judge's verdict on it (`optim.integrity`), when the
    judge is on; the row then carries it in a few words."""
    edits = candidate.own_edits()
    change = (edit_change(parent_records, child_records)
              if parent is not None and child_records and same_seed else None)
    shared = bool(change and change["shared"])
    return {
        "id": candidate.id, "round": (record or {}).get("epoch"),
        "parent": candidate.parent, "kind": change_kind(edits, candidate.state),
        "touched": touched(edits),
        "fixed": len(change["fixed"]) if shared else None,
        "broken": len(change["broken"]) if shared else None,
        "shared": change["shared"] if change else 0,
        "gate": gate, "same_seed": bool(same_seed),
        "rationale": str((record or {}).get("rationale") or ""),
        **({"integrity": _integrity_cell(integrity)} if integrity else {}),
    }


def _integrity_cell(record: Mapping[str, Any]) -> str:
    from .integrity import cell

    return cell(record)


def _net(entry: Mapping[str, Any]) -> int | None:
    fixed, broken = entry.get("fixed"), entry.get("broken")
    return fixed - broken if isinstance(fixed, int) and isinstance(broken, int) else None


def _signed(value: int | None) -> str:
    return "n/a" if value is None else f"{value:+d}"


def _touched_text(names: Sequence[str]) -> str:
    shown = ", ".join(names[:LEDGER_TOUCHED])
    return shown + (f" +{len(names) - LEDGER_TOUCHED} more" if len(names) > LEDGER_TOUCHED else "")


def ledger_lines(entries: Sequence[Mapping[str, Any]], *, rows: int = LEDGER_ROWS,
                 chars: int = LEDGER_CHARS) -> list[str]:
    """"Every change so far, by kind": a tally line per kind over the whole run, then the
    newest candidates in full, oldest first, within ``rows`` and ``chars``; the older
    ones are only in the tally. Nothing for no entries."""
    if not entries:
        return []
    gated = any(entry.get("gate") for entry in entries)
    # The integrity judge's column, only when it judged something (`optim.integrity`).
    judged = any("integrity" in entry for entry in entries)
    tally: list[str] = []
    for kind in CHANGE_KINDS:
        of_kind = [entry for entry in entries if entry.get("kind") == kind]
        if not of_kind:
            continue
        nets = [net for net in map(_net, of_kind) if net is not None]
        parts = [f"tried {len(of_kind)}"]
        if gated:
            parts.append(f"kept {sum(1 for e in of_kind if e.get('gate') in KEPT_WORDS)}")
        parts.append(f"best train fixed-broken {_signed(max(nets) if nets else None)}")
        tally.append(f"- {kind}: " + ", ".join(parts))
    lines = ["## Every change so far, by kind",
             "Every candidate this run has evaluated, by the kind of change its own edits "
             "made. `fixed/broken` compares it with its parent on the train items both were "
             "scored with the same seed, in the round the candidate was evaluated (parent "
             "wrong and child right / the reverse; `n/a (different seed)` when the parent was "
             "not scored in that round)"
             + ("; `gate` is what the validation gate decided" if gated else "")
             + ("; `integrity` is the integrity review's verdict on the change (does its "
                "benefit depend on how the data, grader or environment happen to be built?)"
                if judged else "")
             + ". Read it for which kinds of change have paid off in this pipeline and which "
             "have not, before choosing yours.",
             *tally]
    headers = ["id", "round", "parent", "kind", "touched", "train fixed/broken",
               *(["gate"] if gated else []), *(["integrity"] if judged else []), "rationale"]

    def row(entry: Mapping[str, Any]) -> list[Any]:
        counts = (f"{entry['fixed']}/{entry['broken']} of {entry.get('shared')}"
                  if _net(entry) is not None else
                  "n/a (different seed)" if entry.get("same_seed") is False else "n/a")
        return [entry.get("id"), entry.get("round", ""), entry.get("parent"), entry.get("kind"),
                _touched_text(list(entry.get("touched") or ())), counts,
                *([entry.get("gate") or ""] if gated else []),
                *([entry.get("integrity") or "not judged"] if judged else []),
                _clip(entry.get("rationale"), LEDGER_RATIONALE_CHARS) or "(none)"]

    shown = list(entries)[-max(1, rows):]
    table = _table(headers, [row(entry) for entry in shown])
    while len(shown) > 1 and len(table) > chars:
        shown = shown[1:]
        table = _table(headers, [row(entry) for entry in shown])
    older = len(entries) - len(shown)
    if older:
        lines.append(f"The {older} oldest are counted in the tally above only; the newest "
                     f"{len(shown)} follow, oldest first.")
    return [*lines, table]


# ── lessons ─────────────────────────────────────────────────────────────────


def lessons_text(value: Any) -> str:
    """An answer's ``lessons`` as text within ``LESSONS_CHARS``: a string as written, a
    list as one line per entry; empty when it carries none."""
    if isinstance(value, (list, tuple)):
        text = "\n".join(f"- {str(entry).strip()}" for entry in value if str(entry).strip())
    elif isinstance(value, str):
        text = value.strip()
    else:
        text = ""
    return text if len(text) <= LESSONS_CHARS else text[:LESSONS_CHARS - 1] + "…"


def lessons_lines(lessons: Mapping[str, Any] | None) -> list[str]:
    """"Your lessons so far": the latest lessons the optimizer wrote, with when, and,
    when a later answer could not be read (``not_updated``), that they were not updated
    then and why; nothing before it has written any or had an answer unread."""
    lessons = lessons or {}
    text = lessons_text(lessons.get("text"))
    stale = lessons.get("not_updated") or {}
    if not text and not stale:
        return []
    lines = ["## Your lessons so far"]
    if text:
        lines.append(f"What you wrote under `lessons` in epoch {lessons.get('epoch', '?')} "
                     f"(generation {lessons.get('generation', '?')}). Check them against the "
                     "results since; keep, revise or drop them in your answer.")
    if stale:
        lines.append(f"NOT UPDATED in epoch {stale.get('epoch', '?')} (generation "
                     f"{stale.get('generation', '?')}): your answer then could not be read "
                     f"({_clip(stale.get('why'), 300)}), so its lessons were lost"
                     + ("; these are the earlier ones, unchanged." if text
                        else "; you have none kept yet."))
    return [*lines, *([text] if text else [])]


# ── added steps ─────────────────────────────────────────────────────────────


#: Node kinds that only route (a fork and its join): not listed as added steps.
PLUMBING_KINDS = ("fork", "join")


def added_steps(parent_state: Mapping[str, Any] | None,
                child_state: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """The steps the child has that its parent does not, as ``(id, kind)`` in the
    child's graph order: whatever edit made them (an insert, a split's halves, a
    fork's branches, a loop's gate). A fork and its join only route, so they are left
    out (``PLUMBING_KINDS``)."""
    def nodes(state: Mapping[str, Any] | None) -> Mapping[str, Any]:
        return ((state or {}).get("graph") or {}).get("nodes") or {}

    before = nodes(parent_state)
    return [(str(node_id), _node_kind(node)) for node_id, node in nodes(child_state).items()
            if node_id not in before and _node_kind(node) not in PLUMBING_KINDS]


def _final_output(record: Mapping[str, Any]) -> str | None:
    """The last recorded step output of an item, whitespace-normalised; ``None`` when
    the item recorded none."""
    exchanges = [e for e in record.get("exchanges") or () if isinstance(e, Mapping)]
    if not exchanges:
        return None
    return " ".join(str(exchanges[-1].get("response") or "").split())


def _answer(record: Mapping[str, Any]) -> str | None:
    """The answer the grader read for an item (its record's ``prediction``), whitespace-
    and case-normalised and not parsed any further; ``None`` when none is recorded."""
    value = record.get("prediction")
    if value is None:
        return None
    return " ".join(str(value).split()).lower()


def _numbers(exchanges: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[float]:
    out = []
    for exchange in exchanges:
        for key in keys:
            value = exchange.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out.append(float(value))
                break
    return out


_FACT = re.compile(r"\{\{\s*([A-Za-z_][\w.-]*)\s*\}\}")


def downstream(graph: Mapping[str, Any] | None, step: str) -> set[str] | None:
    """The steps that read what ``step`` writes or produces, directly or through other
    steps (a file in ``reads``/``required_reads``, a fact in ``inputs`` or as a
    ``{{name}}`` in a prompt), from a state's graph; ``None`` without one."""
    nodes = ((graph or {}).get("graph") or {}).get("nodes") if graph else None
    if not isinstance(nodes, Mapping) or step not in nodes:
        return None

    def outputs(node: Mapping[str, Any]) -> set[str]:
        return {str(v) for key in ("writes", "produces") for v in node.get(key) or ()}

    def inputs(node: Mapping[str, Any]) -> set[str]:
        names = {str(v) for key in ("reads", "required_reads", "inputs")
                 for v in node.get(key) or ()}
        for key in ("prompt_template", "system_prompt_template"):
            names |= set(_FACT.findall(str(node.get(key) or "")))
        return names

    found: set[str] = set()
    frontier = outputs(nodes[step])
    while frontier:
        nxt: set[str] = set()
        for node_id, node in nodes.items():
            if node_id == step or node_id in found or not isinstance(node, Mapping):
                continue
            if inputs(node) & frontier:
                found.add(str(node_id))
                nxt |= outputs(node)
        frontier = nxt
    return found


def step_evidence(step: str, *, parent_records: Sequence[Mapping[str, Any]] | None,
                  child_records: Sequence[Mapping[str, Any]] | None,
                  items: int = STEP_REVIEW_ITEMS,
                  graph: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """What the child's train records show of one added step, against its parent on the
    train items both were scored on: the candidate's fixed and broken items, how many
    shared items' final answer changed (the ``prediction`` the grader read, whitespace-
    and case-normalised; where either item recorded none, the final step's output text,
    whitespace-normalised, counted apart), the
    step's median duration and tokens when its records carry them, and up to ``items``
    samples of what the step was given, what it wrote and what the next recorded step that
    reads its output wrote (by ``graph``'s data flow, so a sibling fork branch is not
    taken for it; without a graph, the next recorded step), changed outcomes first, then
    by id."""
    readers = downstream(graph, step)
    change = edit_change(parent_records, child_records)
    parent, child = _by_item(parent_records), _by_item(child_records)
    shared = sorted(set(parent) & set(child))
    # Reasoning text differs between runs on nearly every item, so the answer the grader
    # read is compared; the output text only for items that recorded no answer.
    compared = changed = text_compared = text_changed = 0
    flipped: set[str] = set()
    for item in shared:
        before, after = _answer(parent[item]), _answer(child[item])
        if before is not None and after is not None:
            compared += 1
            changed += before != after
            if before != after:
                flipped.add(item)
            continue
        before, after = _final_output(parent[item]), _final_output(child[item])
        if before is None or after is None:
            continue
        text_compared += 1
        text_changed += before != after
        if before != after:
            flipped.add(item)
    ran = []
    for item in sorted(child):
        exchanges = [e for e in child[item].get("exchanges") or () if isinstance(e, Mapping)]
        at = next((i for i, e in enumerate(exchanges) if e.get("node") == step), None)
        if at is not None:
            ran.append((item, exchanges, at))
    own = [exchanges[at] for _item, exchanges, at in ran]
    durations = _numbers(own, ("duration_s", "elapsed_s", "wall_s"))
    tokens = _numbers(own, ("output_tokens", "tokens"))
    order = {item: rank for rank, item in enumerate([*change["fixed"], *change["broken"]])}
    samples = []
    for item, exchanges, at in sorted(ran, key=lambda r: (order.get(r[0], len(order)), r[0]))[:items]:
        later = exchanges[at + 1:]
        after = next((e for e in later if readers is None or e.get("node") in readers), None)
        samples.append({
            "item": item,
            "input": str(exchanges[at].get("prompt") or ""),
            "output": str(exchanges[at].get("response") or ""),
            "next_step": str(after.get("node") or "") if after else "",
            "next_output": str(after.get("response") or "") if after else "",
        })
    visits = {item: [v for v in child[item].get("steps") or ()
                     if isinstance(v, Mapping) and v.get("node") == step]
              for item in sorted(child)}
    visits = {item: found for item, found in visits.items() if found}
    return {
        "step": step, "shared": change["shared"], "fixed": len(change["fixed"]),
        "broken": len(change["broken"]), "compared": compared, "final_changed": changed,
        "text_compared": text_compared, "text_changed": text_changed,
        "ran": len(ran),
        "median_s": round(statistics.median(durations), 1) if durations else None,
        "median_tokens": round(statistics.median(tokens)) if tokens else None,
        "samples": samples,
        **({"instructions": instructions} if samples and (instructions := step_instructions(
            graph, step)) else {}),
        **({"visits": _visit_facts(visits, set(parent), flipped, change)} if visits else {}),
        **({"decision": True} if samples and decision_step(graph, step, samples) else {}),
    }


def step_instructions(state: Mapping[str, Any] | None, step: str) -> str:
    """A step's own system prompt as the candidate runs it: the value of the knob its
    ``system_prompt_template`` is bound to, else the template; empty without one."""
    nodes = ((state or {}).get("graph") or {}).get("nodes") or {}
    node = nodes.get(step) if isinstance(nodes, Mapping) else None
    if not isinstance(node, Mapping):
        return ""
    knobs = (state or {}).get("knobs") or {}
    for bound, knob in node.get("field_bindings") or ():
        value = (knobs.get(knob) or {}).get("value") if isinstance(knobs, Mapping) else None
        if bound == "system_prompt_template" and isinstance(value, str):
            return value.strip()
    return str(node.get("system_prompt_template") or "").strip()


def fit_samples(samples: Sequence[Mapping[str, Any]], room: int = STEP_REVIEW_ROOM
                ) -> list[dict[str, Any]]:
    """A review's samples with their texts whole when together they fit ``room``; else
    the longest cut head and tail to one length (water-filling, `analysts.item_cap`,
    never below ``STEP_REVIEW_FLOOR``), each cut saying how much it left out."""
    from .analysts import item_cap

    keys = ("input", "output", "next_output")
    cap = item_cap([len(str(s.get(key) or "")) for s in samples for key in keys], room,
                   floor=STEP_REVIEW_FLOOR)
    if cap is None:
        return [dict(s) for s in samples]
    return [{**s, **{key: _clip_middle(s.get(key), cap) for key in keys if s.get(key)}}
            for s in samples]


def decision_step(graph: Mapping[str, Any] | None, step: str,
                  samples: Sequence[Mapping[str, Any]] = ()) -> bool:
    """Whether ``step`` makes the pipeline's decision: no model step reads what it wrote,
    directly or through other steps (by ``graph``'s data flow; only code, gates or the
    grader do), and on the sampled items no other step's model call followed it. Its
    job is to settle on an answer, so whether it repeats its inputs' wording says little;
    whether it decides as they recommended says what it adds."""
    readers = downstream(graph, step)
    if readers is None:
        return False
    nodes = ((graph or {}).get("graph") or {}).get("nodes") or {}
    if any(_node_kind(nodes.get(reader)) == "agent" for reader in readers):
        return False
    return all(not sample.get("next_step") for sample in samples)


#: The most item ids a code step's or gate's line names per group.
VISIT_IDS = 8


def _visit_facts(visits: Mapping[str, Sequence[Mapping[str, Any]]], in_parent: set[str],
                 flipped: set[str], change: Mapping[str, Any]) -> dict[str, Any]:
    """What an added op or code gate did on each item it ran on, from the records'
    ``steps``: an op's items with output (``fired``) and without; a gate's items by its
    first verdict, and the items it was visited on more than once (``relooped``). Each
    group with its items' final-answer changes and fixed/broken against the parent (on
    items the parent was scored on too)."""
    fixed, broken = set(change.get("fixed") or ()), set(change.get("broken") or ())

    def group(items: Sequence[str]) -> dict[str, Any]:
        shared = [item for item in items if item in in_parent]
        return {"items": list(items), "shared": len(shared),
                "changed": sum(item in flipped for item in shared),
                "fixed": sum(item in fixed for item in shared),
                "broken": sum(item in broken for item in shared)}

    kind = "gate" if any(v.get("kind") == "gate" for vs in visits.values() for v in vs) else "op"
    out: dict[str, Any] = {"kind": kind, "ran": len(visits)}
    if kind == "op":
        fired = [item for item, vs in visits.items() if any(not v.get("empty") for v in vs)]
        out["fired"] = group(fired)
        out["silent"] = group([item for item in visits if item not in fired])
        outputs = {v.get("digest") for vs in visits.values() for v in vs if not v.get("empty")}
        out["distinct_outputs"] = len(outputs)
        example = next((v.get("output") for item in fired for v in visits[item]
                        if not v.get("empty") and v.get("output")), "")
        if example:
            out["example"] = _clip(example, STEP_REVIEW_QUOTE_CHARS)
        return out
    routes: dict[str, list[str]] = {}
    for item, vs in visits.items():
        routes.setdefault(str(vs[0].get("verdict") or "?"), []).append(item)
    out["routes"] = {label: group(items) for label, items in
                     sorted(routes.items(), key=lambda kv: (-len(kv[1]), kv[0]))}
    out["relooped"] = group([item for item, vs in visits.items() if len(vs) > 1])
    return out


STEP_REVIEW_SYSTEM = ("You review one step of an AI pipeline against what it was given. "
                      "You reply with JSON only.")


def review_prompt(entry: Mapping[str, Any], mode_name: str = "") -> str:
    """The review call for one added step: its input as sent, its output and the next
    step's output on up to ``STEP_REVIEW_ITEMS`` train items, and domain-free questions.
    A decision step (``entry["decision"]``) is asked about instead (`decision_prompt`)."""
    if entry.get("decision"):
        return decision_prompt(entry, mode_name)
    step = entry.get("step")
    blocks = []
    for sample in fit_samples(entry.get("samples") or ()):
        block = [f"## Item {sample.get('item')}",
                 f"### What `{step}` was given (its prompt as sent)", sample.get("input") or "(empty)",
                 f"### What `{step}` wrote", sample.get("output") or "(empty)"]
        if sample.get("next_step"):
            block += [f"### What the next step that reads it, `{sample['next_step']}`, wrote",
                      sample.get("next_output") or "(empty)"]
        else:
            block.append("(No recorded step after it reads what it wrote.)")
        blocks.append("\n".join(block))
    return "\n\n".join([
        f"# Reviewing an added step: `{step}`",
        (f"Mode: `{mode_name}`. " if mode_name else "")
        + f"Candidate `{entry.get('candidate')}` added the {entry.get('node_kind') or ''} step "
        f"`{step}` to the pipeline ({entry.get('kind')}). Below, on {len(blocks)} train "
        "item(s), is what the step was given, what it wrote, and what the next step that "
        "reads its output wrote.",
        *_instructions_block(entry),
        "Compare the step's output with what it was given: does it contain facts, checks, "
        "computations or alternatives its inputs lacked? Then look at the next step's "
        "output: does it use what this step wrote? Judge the step's own contribution, not "
        "whether the final answer is right, and do not solve the items yourself.",
        *blocks,
        "## Answer\nWrite one JSON object and nothing else:\n\n"
        '{"verdict": "adds_new_work" | "restates_inputs" | "contradicts_inputs" | "no_effect",\n'
        ' "example": "<a short verbatim quote from the step\'s output that shows it>",\n'
        ' "note": "<one sentence>"}\n\n'
        "- adds_new_work: its output holds facts, checks, computations or alternatives its "
        "inputs lacked.\n"
        "- restates_inputs: its output mostly repeats or rephrases what it was given.\n"
        "- contradicts_inputs: its output disagrees with what it was given (say what about "
        "in the note).\n"
        "- no_effect: the next step does not use what it wrote, or there is nothing to "
        "judge.",
    ])


def _instructions_block(entry: Mapping[str, Any]) -> list[str]:
    """The reviewed step's own instructions (its system prompt), when it has any."""
    if not entry.get("instructions"):
        return []
    return [f"## `{entry.get('step')}`'s own instructions (its system prompt)\n"
            f"{entry['instructions']}"]


def decision_prompt(entry: Mapping[str, Any], mode_name: str = "") -> str:
    """The review call for an added decision step (`decision_step`): on each sampled
    item, what its inputs recommended and what it decided. Repeating the chosen answer
    is such a step's job, so the question is whether it ever decides otherwise."""
    step = entry.get("step")
    blocks = [f"## Item {sample.get('item')}\n"
              f"### What `{step}` was given (its prompt as sent)\n{sample.get('input') or '(empty)'}\n"
              f"### What `{step}` wrote\n{sample.get('output') or '(empty)'}"
              for sample in fit_samples(entry.get("samples") or ())]
    return "\n\n".join([
        f"# Reviewing an added decision step: `{step}`",
        (f"Mode: `{mode_name}`. " if mode_name else "")
        + f"Candidate `{entry.get('candidate')}` added the {entry.get('node_kind') or ''} step "
        f"`{step}` to the pipeline ({entry.get('kind')}). No later model step reads what it "
        "writes: it makes the pipeline's decision, so repeating the chosen answer is its "
        f"job. Below, on {len(blocks)} train item(s), is what it was given and what it wrote.",
        *_instructions_block(entry),
        "For each item, say what its inputs recommended (the answer or decision the earlier "
        "steps' notes in its prompt point to; \"none\" when they disagree or recommend "
        "nothing) and what the step decided, and whether the two are the same. Judge the "
        "step's decision against its inputs, not whether it is right, and do not solve "
        "the items yourself.",
        *blocks,
        "## Answer\nWrite one JSON object and nothing else:\n\n"
        '{"items": [{"item": "<id>", "inputs_recommend": "<answer, or none>", '
        '"decided": "<answer>", "same": true | false}],\n'
        ' "example": "<a short verbatim quote from the step\'s output where it departs from '
        'its inputs, or empty>",\n'
        ' "note": "<one sentence>"}',
    ])


def parse_decision_review(payload: Any) -> dict[str, Any]:
    """A decision step's review: of the items whose inputs recommended something, on
    how many (``agree`` of ``of``) it decided the same; its verdict from
    ``DECISION_VERDICTS``. Raises ``ValueError`` on an answer without an items list."""
    if not isinstance(payload, Mapping) or not isinstance(payload.get("items"), list):
        raise ValueError('the decision review has no "items" list')
    rows = [row for row in payload["items"] if isinstance(row, Mapping)]
    if not rows:
        raise ValueError('the decision review\'s "items" list is empty')
    clear = [row for row in rows
             if str(row.get("inputs_recommend") or "").strip().lower() not in ("", "none")]
    agree = sum(row.get("same") is True for row in clear)
    verdict = ("no_recommendation" if not clear else
               "follows_inputs" if agree == len(clear) else "overrides_inputs")
    return {"verdict": verdict, "agree": agree, "of": len(clear), "items": len(rows),
            "example": _clip(payload.get("example"), STEP_REVIEW_QUOTE_CHARS),
            "note": _clip(payload.get("note"), STEP_REVIEW_NOTE_CHARS)}


def parse_review(payload: Any) -> dict[str, Any]:
    """A review answer, checked: a verdict from ``STEP_VERDICTS`` and a bounded example
    and note, or a kept decision review (`parse_decision_review`) as it was stored.
    Raises ``ValueError`` on an answer without a known verdict."""
    if not isinstance(payload, Mapping):
        raise ValueError("the review is not a JSON object")
    if payload.get("verdict") in DECISION_VERDICTS and all(
            isinstance(payload.get(key), int) for key in ("agree", "of", "items")):
        return {key: payload[key] for key in ("verdict", "agree", "of", "items")} | {
            "example": _clip(payload.get("example"), STEP_REVIEW_QUOTE_CHARS),
            "note": _clip(payload.get("note"), STEP_REVIEW_NOTE_CHARS)}
    verdict = re.sub(r"[\s-]+", "_", str(payload.get("verdict") or "").strip().lower())
    if verdict not in STEP_VERDICTS:
        raise ValueError(f"unknown verdict {payload.get('verdict')!r}; expected one of "
                         + ", ".join(STEP_VERDICTS))
    return {"verdict": verdict,
            "example": _clip(payload.get("example"), STEP_REVIEW_QUOTE_CHARS),
            "note": _clip(payload.get("note"), STEP_REVIEW_NOTE_CHARS)}


def step_review_lines(entries: Sequence[Mapping[str, Any]], *,
                      chars: int = STEP_REVIEW_CHARS) -> list[str]:
    """"Did the added steps contribute?": one entry per step the last round's candidates
    added, with its train counts and its review; nothing for none."""
    if not entries:
        return []
    lines = ["## Did the added steps contribute?",
             "Each step a candidate of the last round added, on the train items it shares "
             "with its parent: the candidate's fixed/broken items, how often the final answer "
             "(what the grader read) changed, and a reviewer's verdict (one call per step, on up to "
             f"{STEP_REVIEW_ITEMS} train items) on whether the step's output holds facts, "
             "checks, computations or alternatives its inputs lacked. A step that only "
             "restates its inputs adds a model call and little else. A step that makes the "
             "decision (no later model step reads it) is judged instead by how often it "
             "decides as its inputs recommended. A code step or gate gets no reviewer: its "
             "measured behaviour is shown (on which items a code step gave output or a gate "
             "routed each way or relooped, and how the final answer moved on them)."]
    used, compact = sum(len(line) + 1 for line in lines), 0
    for entry in entries:
        head = (f"- `{entry.get('step')}` ({entry.get('node_kind') or 'step'}, added by "
                f"{entry.get('candidate')}, {entry.get('kind')}"
                + (f"; gate {entry['gate']}" if entry.get("gate") else "") + "): ")
        if entry.get("same_seed") is False:
            facts = ["fixed/broken and answer changes n/a (different seed: the parent was not "
                     "scored in the round this candidate was evaluated)"]
        elif entry.get("shared"):
            facts = [f"{entry.get('fixed')} fixed / {entry.get('broken')} broken on "
                     f"{entry.get('shared')} shared train item(s)"]
        else:
            facts = ["no train item shared with its parent"]
        if entry.get("compared"):
            share = entry["final_changed"] / entry["compared"]
            facts.append(f"final answer changed on {entry['final_changed']}/{entry['compared']} "
                         f"({share:.0%}; the answer the grader read)")
        if entry.get("text_compared"):
            share = entry["text_changed"] / entry["text_compared"]
            facts.append(f"final output text changed on {entry['text_changed']}/"
                         f"{entry['text_compared']} ({share:.0%}; "
                         + ("items that recorded no answer, so " if entry.get("compared") else
                            "no answer recorded, so ")
                         + "the last step's output text was compared)")
        if entry.get("median_s") is not None:
            facts.append(f"median {entry['median_s']} s")
        if entry.get("median_tokens") is not None:
            facts.append(f"median {entry['median_tokens']} tokens")
        if isinstance(entry.get("visits"), Mapping):
            facts.append(visit_text(entry["visits"], entry.get("same_seed") is not False))
        review = entry.get("review")
        if isinstance(review, Mapping):
            said = f"review: {_verdict_text(review)}"
            if review.get("example"):
                said += f' -- "{review["example"]}"'
            if review.get("note"):
                said += f" -- {review['note']}"
        elif entry.get("review_error"):
            said = f"review: not available ({_clip(entry['review_error'], 160)})"
        elif entry.get("visits"):
            said = "no reviewer (a code step or gate: the measured behaviour above)"
        elif not entry.get("ran"):
            said = ("review: none (the step recorded no model call and its records hold no "
                    "visit of it: a fork or join step, a step that never ran, or records "
                    "made before visits were kept)")
        else:
            said = "review: none (more added steps than are reviewed per round)"
        line = head + "; ".join(facts) + "; " + said
        if used + len(line) + 1 > chars:
            # Past the section's room: what each remaining step did, in one short line.
            line = (f"- `{entry.get('step')}` (added by {entry.get('candidate')}): "
                    + ("n/a (different seed)" if entry.get("same_seed") is False else
                       f"{entry.get('fixed')} fixed / {entry.get('broken')} broken")
                    + (f"; final answer changed {entry['final_changed']}/{entry['compared']}"
                       if entry.get("compared") and entry.get("same_seed") is not False else "")
                    + (f"; review: {_verdict_text(review)}" if isinstance(review, Mapping)
                       else ""))
            if not compact:
                lines.append(f"(The section is kept within {chars} characters: the remaining "
                             "steps in one line each.)")
            compact += 1
        lines.append(line)
        used += len(line) + 1
    return lines


def _verdict_text(review: Mapping[str, Any]) -> str:
    """A review's verdict as shown: a decision step's says how often it decided as its
    inputs recommended."""
    if review.get("verdict") in DECISION_VERDICTS:
        if not review.get("of"):
            return (f"no_recommendation (its inputs recommended no one answer on any of the "
                    f"{review.get('items')} item(s) read)")
        return (f"{review['verdict']} (decides as its inputs recommended on {review['agree']} "
                f"of {review['of']} item(s) where they recommended one)")
    return str(review.get("verdict"))


def _ids(items: Sequence[str]) -> str:
    shown = ", ".join(items[:VISIT_IDS])
    return shown + (f" and {len(items) - VISIT_IDS} more" if len(items) > VISIT_IDS else "")


def _moved(group: Mapping[str, Any], counts: bool) -> str:
    """How the final answer moved on a group's items shared with the parent."""
    if not counts:
        return ""
    if not group.get("shared"):
        return " (none shared with the parent)"
    return (f" (final answer changed on {group['changed']}/{group['shared']}; "
            f"{group['fixed']} fixed / {group['broken']} broken)")


def visit_text(visits: Mapping[str, Any], counts: bool = True) -> str:
    """An added code step's or gate's measured behaviour (`_visit_facts`), in one clause:
    on which items it gave output or routed each way, and how the final answer moved
    there; ``counts`` off (a parent scored on another seed) leaves the movement out."""
    ran = visits.get("ran", 0)
    if visits.get("kind") == "gate":
        routes = [f"`{label}` on {len(group['items'])}: {_ids(group['items'])}"
                  + _moved(group, counts)
                  for label, group in (visits.get("routes") or {}).items()]
        relooped = visits.get("relooped") or {}
        text = (f"ran on {ran} item(s); first verdict " + "; ".join(routes)
                + f"; relooped (visited more than once) on {len(relooped.get('items') or ())}")
        if relooped.get("items"):
            text += f": {_ids(relooped['items'])}" + _moved(relooped, counts)
        return text
    fired, silent = visits.get("fired") or {}, visits.get("silent") or {}
    text = f"gave output on {len(fired.get('items') or ())} of {ran} item(s) it ran on"
    if fired.get("items"):
        text += (f": {_ids(fired['items'])}" + _moved(fired, counts)
                 + f", {visits.get('distinct_outputs', 0)} distinct output(s)")
        if visits.get("example"):
            text += f', e.g. "{visits["example"]}"'
    if silent.get("items"):
        text += f"; empty output on {len(silent['items'])}" + _moved(silent, counts)
    return text


def review_key(prompt: str) -> str:
    import hashlib

    return hashlib.sha256(prompt.encode()).hexdigest()


def stored_review(text: str) -> dict[str, Any] | None:
    try:
        saved = json.loads(text)
    except json.JSONDecodeError:
        return None
    try:
        return parse_review(saved)
    except ValueError:
        return None


#: The paragraph ``structure_guidance`` adds under "How nodes compose".
STRUCTURE_GUIDANCE = (
    "What makes a new step pay off: a step helps when it adds work or information the "
    "pipeline does not already have -- an independent attempt that does not see earlier "
    "steps' conclusions, several attempts combined by a vote or a check, a deterministic "
    "computation or check in code, or a retry sent back with a specific reason. A step "
    "that re-reads the same notes with the same model tends to repeat their mistakes. "
    "Prefer a structural change that changes how the work is done over one that delivers "
    "an instruction a prompt edit could deliver.")
