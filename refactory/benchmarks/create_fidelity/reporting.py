"""Offline source graph diagrams, stage mapping and raw evidence links."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
from urllib.parse import quote


def nodes(graph: dict) -> dict:
    value = graph.get("nodes", {})
    return value if isinstance(value, dict) else {n["id"]: n for n in value}


def diagram(graph: dict, lane: str) -> str:
    items = nodes(graph)
    if not items:
        return '<p>No source graph available.</p>'
    edges = [dict(e) for e in graph.get("edges", [])]
    for name, node in items.items():
        edges += [{"source": name, "target": t, "condition": "fork"} for t in node.get("targets", [])]
        edges += [{"source": s, "target": name, "condition": "join"} for s in node.get("sources", [])]
    forward = [e for e in edges if e.get("condition") != "reloop" and e["source"] in items and e["target"] in items]
    ranks = {n: 0 for n in items}
    for _ in range(len(items)):
        changed = False
        for e in forward:
            value = min(len(items), ranks[e["source"]] + 1)
            if value > ranks[e["target"]]:
                ranks[e["target"]] = value
                changed = True
        if not changed:
            break
    levels = {r: sorted(n for n in items if ranks[n] == r) for r in set(ranks.values())}
    width, step = 780, 82
    positions = {n: (width * (i + 1) / (len(group) + 1), 40 + rank * step)
                 for rank, group in levels.items() for i, n in enumerate(group)}
    marker = quote(lane, safe="") + "-arrow"
    svg = [f'<svg viewBox="0 0 {width} {(max(ranks.values()) + 1) * step}" role="img" aria-label="{html.escape(lane)} source graph">',
           f'<defs><marker id="{marker}" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="none" stroke="#64748b"/></marker></defs>']
    for edge in edges:
        if edge["source"] not in positions or edge["target"] not in positions:
            continue
        x1, y1 = positions[edge["source"]]
        x2, y2 = positions[edge["target"]]
        loop = edge.get("condition") == "reloop"
        bend = 32 if loop else (x1 + x2) / 2
        d = f'M{x1},{y1+16} C{bend},{y1+40} {bend},{y2-40} {x2},{y2-17}'
        label = edge.get("condition") or ""
        color = "#b45309" if loop else "#94a3b8"
        svg.append(f'<path d="{d}" fill="none" stroke="{color}" marker-end="url(#{marker})"><title>{html.escape(edge["source"] + " → " + edge["target"] + " " + label)}</title></path>')
    for name, node in items.items():
        x, y = positions[name]
        kind = node.get("kind", node.get("_type", ""))
        color = "#fef3c7" if "gate" in kind.lower() else "#dbeafe" if "agent" in kind.lower() else "#e2e8f0"
        owner = node.get("gate_kind", node.get("evaluator_type", node.get("role", "native")))
        dash = ' stroke-dasharray="5 3"' if node.get("blocking") is False else ""
        svg.append(f'<a href="#gap-{quote(name)}"><rect x="{x-108}" y="{y-17}" width="216" height="34" rx="7" fill="{color}" stroke="#64748b"{dash}/><text x="{x}" y="{y+4}" text-anchor="middle" font-size="12">{html.escape(name)}</text><title>{html.escape(str(kind) + "; owner=" + str(owner) + "; blocking=" + str(node.get("blocking", True)))}</title></a>')
    return ''.join(svg) + '</svg>'


def annotated_html(path: Path, report: dict, oracle: dict) -> str:
    lanes = report["lanes"]
    def evidence_link(evidence: str, lane: dict) -> str:
        filename, _, fragment = evidence.partition("#")
        target = Path(filename)
        if not target.is_absolute():
            target = Path(lane.get("evidence_root", path)) / target
        href = quote(os.path.relpath(target, path), safe="/") + ("#" + quote(fragment, safe="=") if fragment else "")
        return f'<a href="{href}">{html.escape(evidence)}</a>'
    headers = ''.join('<th>' + html.escape(lane["lane"]) + '</th>' for lane in lanes)
    graphs = []
    for lane in lanes:
        g = lane.get("graph", {})
        graphs.append('<section><h2>' + html.escape(lane["lane"]) + '</h2><p>' + html.escape(lane.get("graph_origin", "not supplied")) +
                      f' · {len(nodes(g))} nodes / {len(g.get("edges", []))} edges</p><details><summary>Source identity and preflight</summary><pre>' +
                      html.escape(json.dumps({"identity": lane.get("identity", {}), "preflight": lane.get("preflight", {}), "evaluation": lane.get("evaluation", {})}, indent=2)) + '</pre></details>' + diagram(g, lane["lane"]) + '</section>')
    gaps = []
    source_url = f'https://github.com/boston-harness-group/remote-factory/blob/{oracle["source_commit"]}/factory/workflow/contributed/create_v2/workflow.py'
    for name, expected in oracle["graph"]["nodes"].items():
        cells = []
        for lane in lanes:
            node = nodes(lane.get("graph", {})).get(name)
            if node is None:
                description = "Absent at this stage ID; a renamed stage needs explicit obligation mapping."
            else:
                description = (f'{node.get("kind", node.get("_type"))}; owner={node.get("gate_kind", node.get("evaluator_type", node.get("role", "native")))}; '
                               f'blocking={node.get("blocking", True)}; timeout={node.get("timeout_s", node.get("timeout", "inherited"))}')
                if name.endswith("_director") and node.get("kind") == "op":
                    description += "; composite DSH planner/children/quality-review mapping"
            cells.append('<td>' + html.escape(description) + '</td>')
        gaps.append(f'<tr id="gap-{quote(name)}"><th><a href="{source_url}">{html.escape(name)}</a><small>{html.escape(expected["_type"])}</small></th>{"".join(cells)}</tr>')
    requirements = list(dict.fromkeys(r["requirement_id"] for lane in lanes for key in ("capability", "fidelity") for r in lane[key]))
    body = []
    regression_map = report.get("regression_links", {})
    # Fix links refer to the current implementation, including regressions added
    # after an old failed trial. Frozen run identities remain separate above.
    revision = regression_map.get("fix_ref", "codex/create-fidelity")
    repository_url = "https://github.com/boston-harness-group/refactory-dsh/blob/" + quote(revision, safe="/") + "/"
    for requirement in requirements:
        cells = []
        for lane in lanes:
            r = next((r for key in ("capability", "fidelity") for r in lane[key] if r["requirement_id"] == requirement),
                     {"status": "unavailable", "reason": "No evidence", "evidence": []})
            cells.append(f'<td class="{r["status"]}"><strong>{r["status"]}</strong><p>{html.escape(r["reason"])}</p>' + ''.join(evidence_link(e, lane) for e in r["evidence"]) + '</td>')
        fixes = []
        for item in regression_map.get("rows", []):
            if not any(requirement == prefix or prefix.endswith(".") and requirement.startswith(prefix)
                       for prefix in item["requirements"]):
                continue
            fixes.append('<details><summary>' + html.escape(item["owner"]) + '</summary><p>' +
                         html.escape(item["fix"]) + '</p><a href="' + repository_url + quote(item["source"], safe="/") +
                         '">Current candidate fix</a><a href="' + repository_url + quote(item["regression"], safe="/") +
                         '">Controlled regression</a></details>')
        body.append(f'<tr><th>{html.escape(requirement)}{"".join(fixes)}</th>{"".join(cells)}</tr>')
    style = 'body{font:15px system-ui;margin:2rem;color:#172033}table{border-collapse:collapse;width:100%;margin:2rem 0}td,th{padding:.7rem;border:1px solid #ccd4df;vertical-align:top;text-align:left}.passed{background:#dcfce7}.failed{background:#fee2e2}.unavailable,.not_exercised{background:#f1f5f9}p{max-width:65rem}td a,small{display:block;overflow-wrap:anywhere}.graphs{display:flex;gap:1rem}.graphs section{flex:1;min-width:320px;border:1px solid #cbd5e1;padding:1rem}.graphs svg{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}h2{font-size:20px}th{max-width:220px}:target{background:#fef3c7}'
    return ('<!doctype html><meta charset="utf-8"><title>Create fidelity evidence</title><style>' + style +
            '</style><h1>Paired Create evidence</h1><p>' + html.escape(report["interpretation"]) +
            '</p><p>Overall live acceptance: <strong>' + report['status'] +
            '</strong>. Static source graphs are not execution evidence. Click a stage to inspect its source mapping. Dashed nodes are non-blocking; orange edges are repairs.</p><div class="graphs">' + ''.join(graphs) +
            '</div><h2>Source stage mapping</h2><table><tr><th>Pinned reference stage</th>' + headers + '</tr>' + ''.join(gaps) +
            '</table><h2>Independent capability and authoring evidence</h2><table><tr><th>Requirement</th>' + headers + '</tr>' + ''.join(body) + '</table>')
