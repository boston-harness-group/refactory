"""Show the proposer what its new or changed steps produced on the smoke items.

After a proposal's smoke run passes, this module builds a short review block from
the smoke run's per-item records and asks the proposer whether the output is what
it intended.  The proposer replies keep, a revised proposal, or drop.

``review_needed`` decides whether a candidate gets a review (only proposals that
add or change a step); ``build_review_block`` extracts per-step facts, files,
replies, errors and warning flags; ``smoke_review_turn`` makes the model call and
parses the reply.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from refactory_dsh.nn.mode import RunResult

REVIEW_BLOCK_CAP = 12000
FACT_SIZE_WARN = 20000
FILE_SIZE_WARN = 20000


def _nodes(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    graph = state.get("graph") or {}
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    if isinstance(nodes, dict):
        return dict(nodes)
    if isinstance(nodes, list):
        return {str(n.get("id", i)): n for i, n in enumerate(nodes)}
    return {}


def _knob_values(state: Mapping[str, Any]) -> dict[str, Any]:
    return {name: (entry or {}).get("value")
            for name, entry in (state.get("knobs") or {}).items()}


def _node_signature(node: dict[str, Any], knobs: dict[str, Any]) -> dict[str, Any]:
    sig: dict[str, Any] = {
        "kind": node.get("kind"),
        "call": node.get("call"),
        "evaluator": node.get("evaluator"),
        "prompt_template": node.get("prompt_template"),
        "reads": tuple(sorted(node.get("reads") or ())),
        "writes": tuple(sorted(node.get("writes") or ())),
        "produces": tuple(sorted(node.get("produces") or ())),
    }
    for field_name, knob_name in node.get("field_bindings") or ():
        sig[f"bound:{field_name}"] = knobs.get(knob_name)
    if node.get("code"):
        sig["code"] = node["code"]
    return sig


def changed_steps(
    parent: Any, child: Any,
) -> tuple[set[str], set[str]]:
    """The child's new and changed step ids compared to its parent.

    Returns ``(added, changed)``.  A step is changed when any of its kind, call,
    evaluator, prompt template, reads, writes, produces, bound knob values or
    code differ between parent and child.
    """
    if parent is None:
        return set(), set()
    parent_nodes = _nodes(parent.state)
    child_nodes = _nodes(child.state)
    parent_knobs = _knob_values(parent.state)
    child_knobs = _knob_values(child.state)
    added = set(child_nodes) - set(parent_nodes)
    changed: set[str] = set()
    for nid in set(child_nodes) & set(parent_nodes):
        if _node_signature(parent_nodes[nid], parent_knobs) != _node_signature(
            child_nodes[nid], child_knobs
        ):
            changed.add(nid)
    return added, changed


def review_needed(parent: Any, child: Any) -> bool:
    """Whether a candidate needs a smoke output review: it adds or changes a step."""
    added, changed_set = changed_steps(parent, child)
    return bool(added or changed_set)


def _truncate(text: str, limit: int) -> tuple[str, int]:
    full = len(text)
    if full <= limit:
        return text, full
    return text[:limit] + f"\n… [{full - limit} chars truncated]", full


def _file_reads(node: dict[str, Any]) -> set[str]:
    return set(node.get("reads") or ())


def _step_data_from_records(
    item_records: Sequence[dict[str, Any]],
    step_ids: set[str],
    child_nodes: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Per-step, per-item data from item records."""
    out: dict[str, list[dict[str, Any]]] = {sid: [] for sid in step_ids}
    for record in item_records:
        item_id = str(record.get("id", ""))
        for visit in record.get("steps") or ():
            node_id = visit.get("node")
            if node_id not in step_ids:
                continue
            entry: dict[str, Any] = {
                "item": item_id,
                "kind": visit.get("kind"),
                "output": visit.get("output", ""),
                "output_len": len(str(visit.get("output", ""))),
                "empty": visit.get("empty", False),
            }
            if visit.get("kind") == "gate":
                entry["verdict"] = visit.get("verdict", "")
                entry["reason"] = visit.get("reason", "")
            out[node_id].append(entry)
        for exchange in record.get("exchanges") or ():
            node_id = exchange.get("node")
            if node_id not in step_ids:
                continue
            reply_text = str(exchange.get("response") or "")
            out.setdefault(node_id, [])
            exists = any(e.get("item") == item_id and e.get("kind") == "agent"
                         for e in out[node_id])
            if not exists:
                out[node_id].append({
                    "item": item_id,
                    "kind": "agent",
                    "reply": reply_text[:2000],
                    "reply_len": len(reply_text),
                })
        if record.get("error_text"):
            error_text = str(record["error_text"])
            first_line = error_text.split("\n", 1)[0]
            for sid in step_ids:
                if sid in first_line:
                    out.setdefault(sid, [])
                    out[sid].append({
                        "item": item_id,
                        "kind": "error",
                        "error": error_text[:1000],
                    })
        if record.get("missing_steps"):
            for sid, info in (record["missing_steps"] or {}).items():
                if sid in step_ids:
                    out.setdefault(sid, [])
                    out[sid].append({
                        "item": item_id,
                        "kind": "no_answer",
                        "cause": info.get("cause", "unknown") if isinstance(info, dict) else str(info),
                    })
    return out



def _check_identical_files(
    step_ids: set[str],
    child_nodes: dict[str, dict[str, Any]],
    smoke_workdir: Any,
    item_records: Sequence[dict[str, Any]],
) -> list[str]:
    """Check if any step wrote a file identical to one it read."""
    from pathlib import Path

    flags: list[str] = []
    if smoke_workdir is None:
        return flags
    workdir = Path(smoke_workdir)
    for record in item_records:
        item_id = str(record.get("id", ""))
        for sid in step_ids:
            node = child_nodes.get(sid, {})
            reads = set(node.get("reads") or ())
            writes = set(node.get("writes") or ())
            if not reads or not writes:
                continue
            from refactory_dsh.items import pass_dir_of
            index = 1
            pdir = pass_dir_of(workdir, next(
                (nid for nid, n in child_nodes.items() if n.get("kind") == "data"), ""),
                item_id, index)
            if not pdir.is_dir():
                for candidate_dir in workdir.glob(f"*/{item_id}*"):
                    if candidate_dir.is_dir():
                        pdir = candidate_dir
                        break
                else:
                    continue
            for written_name in writes:
                for read_name in reads:
                    written = pdir / written_name
                    read_file = pdir / read_name
                    if not written.exists() or not read_file.exists():
                        continue
                    try:
                        if written.read_bytes() == read_file.read_bytes():
                            flags.append(
                                f"⚠ step `{sid}` on item {item_id}: "
                                f"wrote `{written_name}` byte-identical to "
                                f"its input `{read_name}`"
                            )
                    except OSError:
                        continue
    return flags


def build_review_block(
    parent: Any,
    child: Any,
    result: RunResult,
    smoke_workdir: Any = None,
) -> str | None:
    """Build the review block shown to the proposer.

    Returns ``None`` when no review is needed (no added or changed steps).
    """
    added, changed_set = changed_steps(parent, child)
    step_ids = added | changed_set
    if not step_ids:
        return None

    child_nodes = _nodes(child.state)
    item_records = list(result.item_records or ())
    if not item_records:
        return None

    step_data = _step_data_from_records(item_records, step_ids, child_nodes)
    flags = _check_identical_files(step_ids, child_nodes, smoke_workdir, item_records)

    for sid, visits in step_data.items():
        if not visits:
            flags.append(f"⚠ step `{sid}`: produced no output on any smoke item")
        for v in visits:
            output_len = v.get("output_len", 0) or v.get("reply_len", 0)
            if output_len > FACT_SIZE_WARN:
                flags.append(
                    f"⚠ step `{sid}` on item {v.get('item')}: output is {output_len:,} "
                    f"characters (> {FACT_SIZE_WARN:,})"
                )
            if v.get("empty"):
                flags.append(f"⚠ step `{sid}` on item {v.get('item')}: output is empty")
            if v.get("kind") == "no_answer":
                flags.append(
                    f"⚠ step `{sid}` on item {v.get('item')}: gave no answer "
                    f"({v.get('cause', 'unknown')})"
                )

    lines: list[str] = ["## Smoke output review"]
    lines.append("")
    lines.append(
        f"Your proposal {'added' if added else 'changed'} "
        + (f"{len(step_ids)} step(s)" if len(step_ids) > 1 else f"step `{next(iter(step_ids))}`")
        + f". A smoke run on {len(item_records)} train item(s) passed. "
        "Here is what those steps produced:"
    )

    for sid in sorted(step_ids):
        label = "new" if sid in added else "changed"
        node = child_nodes.get(sid, {})
        kind = node.get("kind", "unknown")
        lines.append("")
        lines.append(f"### `{sid}` ({label}, {kind})")
        visits = step_data.get(sid, [])
        if not visits:
            lines.append("No output recorded.")
            continue
        for v in visits:
            item = v.get("item", "?")
            if v.get("kind") == "op":
                output, full_len = _truncate(str(v.get("output", "")), 2000)
                lines.append(f"- Item `{item}`: {output}")
                if full_len > 2000:
                    lines.append(f"  (full length: {full_len:,} chars)")
            elif v.get("kind") == "gate":
                lines.append(f"- Item `{item}`: verdict={v.get('verdict')}, "
                             f"reason={v.get('reason', '')[:500]}")
            elif v.get("kind") == "agent":
                reply, full_len = _truncate(str(v.get("reply", "")), 2000)
                lines.append(f"- Item `{item}` reply: {reply}")
                if full_len > 2000:
                    lines.append(f"  (full length: {v.get('reply_len', full_len):,} chars)")
            elif v.get("kind") == "error":
                lines.append(f"- Item `{item}` error: {v.get('error', '')[:500]}")
            elif v.get("kind") == "no_answer":
                lines.append(f"- Item `{item}`: no answer ({v.get('cause', 'unknown')})")

    for record in item_records:
        lines.append("")
        lines.append(f"Item `{record.get('id')}`: score={record.get('score')}, "
                     f"passed={record.get('passed')}")

    if flags:
        lines.append("")
        lines.append("### Warnings")
        for flag in dict.fromkeys(flags):
            lines.append(flag)

    lines.append("")
    lines.append(
        "Does this output do what you intended? Reply with one of:\n"
        "- `keep` — proceed with this proposal as-is\n"
        "- A revised JSON proposal (same format as your original) — it will go "
        "through the same checks and smoke run\n"
        "- `drop` — withdraw this proposal"
    )

    block = "\n".join(lines)
    if len(block) > REVIEW_BLOCK_CAP:
        block = block[:REVIEW_BLOCK_CAP] + f"\n… [{len(block) - REVIEW_BLOCK_CAP} chars truncated]"
    return block


KEEP = "keep"
DROP = "drop"
REVISED = "revised"


def parse_review_reply(reply: str) -> tuple[str, dict[str, Any] | None]:
    """Parse a review reply: ``(decision, payload)``.

    ``decision`` is one of ``KEEP``, ``DROP`` or ``REVISED``.  For ``REVISED``,
    ``payload`` is the parsed JSON object.
    """
    text = reply.strip()
    if not text:
        return KEEP, None
    lower = text.lower().strip().strip("`").strip()
    if lower == "keep":
        return KEEP, None
    if lower == "drop":
        return DROP, None
    try:
        obj = _extract_json(text)
        if obj is not None and isinstance(obj, dict):
            return REVISED, obj
    except Exception:
        pass
    if "keep" in lower[:20]:
        return KEEP, None
    if "drop" in lower[:20]:
        return DROP, None
    return KEEP, None


def _extract_json(text: str) -> Any:
    for start in range(len(text)):
        if text[start] == "{":
            for end in range(len(text), start, -1):
                if text[end - 1] == "}":
                    try:
                        return json.loads(text[start:end])
                    except json.JSONDecodeError:
                        continue
    return None


def review_call_text(block: str) -> str:
    """The prompt for the review call: the block plus instructions."""
    return block
