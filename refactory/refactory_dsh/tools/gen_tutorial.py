"""Generate the HTML tutorial from the code, so it cannot drift.

A hand-written tour of sixteen modes, two v2 pipelines and eleven loop options is
wrong the first time somebody adds a node or a flag. This reads the real
catalogue, the real graphs, the real registries and the real command lines, and
writes `docs/tutorial.html`; `--check` fails when the file on disk differs, so the
page is a fact about the code rather than a memory of it.

    python -m refactory_dsh.tools.gen_tutorial          # write
    python -m refactory_dsh.tools.gen_tutorial --check  # verify

The prose (what the outer loop is, how to ask the CEO) is static and lives here;
the tables, the diagrams, the option lists and the verb lists are read from
`console.catalogue`, each mode's `build()`, `strategies`, `optim`, `evaluators`,
`nn.edits.PRESETS`, the two command lines, and the tool's own verb table.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from refactory_dsh import console, evaluators, optim, registry
from refactory_dsh import strategies as strategies_module
from refactory_dsh.loop import LoopConfig
from refactory_dsh.nn import edits as edits_module
from refactory_dsh.nn.graph import Node
from refactory_dsh.paths import REPO_ROOT

OUT = Path(__file__).resolve().parents[2] / "docs" / "tutorial.html"

#: Where the packages this repository *ships* live. The registry also discovers the
#: packages a person created in `.factory/packages`, and a page generated from those
#: would differ on every machine — so the tutorial documents what a clone contains,
#: not what this checkout happens to have made.
COMMITTED = Path(__file__).resolve().parents[2] / "refactory_packages"

#: The tool's generated contract table, which the "ask the CEO" section documents.
#: `2b14f72eff` (#258) replaced the hand-written `VERBS` table in `index.ts` with
#: this file, so the page reads the table the tool actually ships.
TOOL_SOURCE = REPO_ROOT / "packages" / "refactory" / "tool-refactory" / "src" / "generated-help.ts"

#: Which node kinds a reader has to understand, in the order the legend shows them.
KINDS: tuple[tuple[str, str], ...] = (
    ("op", "an operation: deterministic Python, no model"),
    ("agent", "an agent task: may use tools and multiple model turns"),
    ("gate", "a gate: routes on a verdict — `fn` computes it, `user` asks a person"),
    ("fork", "a fork: the branches start together"),
    ("join", "a join: the barrier every branch must reach"),
    ("data", "a data node: the subgraph runs once per item"),
)

#: The outer loop's options, with the explanation a reader needs. `parse_options`
#: proves the table is complete against the real command line.
LOOP_OPTIONS: tuple[tuple[str, str, str], ...] = (
    ("MODE", "the mode to optimize, as `<package>/<name>`", "—"),
    ("--epochs", "how many rounds of propose → validate → evaluate → select", "2"),
    ("--batch-size", "candidates proposed per epoch", "4"),
    ("--optimizer", "how the next candidates are proposed (table below)", "autoresearch"),
    ("--strategy", "which outer loop drives the search (table below)", "the harness's own"),
    ("--evaluator", "which runtime scores a candidate: `walker` (fast) or `dsh` (ships)", "walker"),
    ("--preset", "which edits the optimizer may use: `broad`, `structure`, `conservative`, `knob_only`", "broad"),
    ("--seed", "the search RNG seed; model responses can still vary", "0"),
    ("--elite", "how many parents each round proposes against", "3"),
    ("--max-evaluations", "a hard ceiling on evaluated candidates, whatever the epochs say", "64"),
    ("--attempts", "retries per batch after a refusal, before the batch is short", "3"),
    ("--runs-root", "the run directory to write; read it back with `console --run <this path>`", ".factory/loop/<mode>-<time>"),
    ("--intent", "what the run is for, in your words; seeds the intent ledger the agents read", "empty"),
    ("--project", "the project root packages resolve against", "the checkout"),
    ("--model", "model for proposals and agent nodes in both evaluators; inspect actual worker receipts", "the default route"),
)

#: The loop's own steps, in order. The default strategy's names, and what each does.
LOOP_STEPS: tuple[tuple[str, str], ...] = (
    ("init", "read the mode's state, freeze what was frozen, write the run's first frame"),
    ("forward", "evaluate the seed, so the search has a score to beat"),
    ("propose", "the optimizer proposes a batch of edits — typed, never free text"),
    ("validate", "every candidate is checked before it may cost a run (V2, V7, `E-shape`, …)"),
    ("backward", "score the batch and turn it into a gradient over the knobs and nodes"),
    ("optimizer_step", "the optimizer's own state advances (elites, population, archive)"),
    ("select_parents", "pick what the next batch is proposed against"),
    ("report", "the leaderboard, the rejections, and what the optimizer actually received"),
)

#: The run's own verbs, as the console spells them, with the question each answers.
CONSOLE_VERBS: tuple[tuple[str, str], ...] = (
    ("modes", "what can this checkout run"),
    ("strategies", "the named outer loops, optimizers and evaluators"),
    ("runs", "every run under the root, newest first"),
    ("start <mode>", "begin a run in its own process"),
    ("status --run <id>", "where a run is, and the events since an offset"),
    ("artifacts --run <id>", "what a run left on disk"),
    ("leaderboard --run <id>", "the candidates by score, each with its diff against its parent"),
    ("candidate <id> --run <id>", "one candidate and exactly what it changed"),
    ("best --run <id>", "the winner, with the state that scored it"),
    ("adopt --run <id> --as <pkg>", "write the winner as a new package, input untouched"),
    ("steer --run <id> --note …", "read the intent ledger, or add what you said to it"),
    ("stop --run <id>", "ask a run to stop"),
    ("registry list|verify|smoke|add|import|remove|enable|disable", "manage packages of modes"),
    ("create|edit|design", "author with a model (see the two v2 sections)"),
)


def esc(value: Any) -> str:
    """Text, safe in an element and in an attribute."""
    return html.escape(str(value), quote=True)


def code(value: Any) -> str:
    """A code span, escaped."""
    return f"<code>{esc(value)}</code>"


# ── the code as data ───────────────────────────────────────────────────────


def mode_shape(identifier: str, project: Path) -> dict[str, Any]:
    """One mode's graph: its nodes, its edges, and the phase walk a reader follows."""
    mode = registry.resolve(project, identifier)[1]()
    graph = mode.build()
    nodes = list(graph.nodes.values())
    return {
        "nodes": nodes,
        "edges": list(graph.edges),
        "knobs": {name: getattr(knob, "description", "") for name, knob in (mode.declare() or {}).items()},
        "unscored_reason": getattr(mode, "unscored_reason", ""),
        "agents": sum(1 for node in nodes if node.kind == "agent"),
        "walk": _walk(graph),
    }


def _walk(graph: Any) -> list[tuple[int, Node]]:
    """The graph as a reader walks it: depth-first, a fork's branches indented."""
    nodes = {node.id: node for node in graph.nodes.values()}
    outgoing: dict[str, list[str]] = {}
    for edge in graph.edges:
        outgoing.setdefault(edge.source, []).append(edge.target)
    seen: set[str] = set()
    walked: list[tuple[int, Node]] = []

    def visit(node_id: str, depth: int) -> None:
        if node_id in seen or node_id not in nodes:
            return
        seen.add(node_id)
        node = nodes[node_id]
        walked.append((depth, node))
        if node.kind == "fork":
            for target in node.targets:
                visit(target, depth + 1)
        if node.kind == "data":
            # The subgraph is reached through the node, not through an edge.
            visit(node.subgraph_entry, depth + 1)
        for target in outgoing.get(node_id, []):
            visit(target, depth)

    visit(graph.start, 0)
    return walked


def catalogue(project: Path) -> list[dict[str, Any]]:
    """Every *shipped* mode, with the console's own entry and its graph's shape.

    Workspace packages (what `create` wrote into `.factory/packages`) are left out on
    purpose: the page is committed, so it has to be the same page on every machine.
    """
    entries: list[dict[str, Any]] = []
    for entry in console.catalogue(project=project):
        where = registry.where(str(entry["package"]), Path(project))
        if where is None or not Path(where).resolve().is_relative_to(COMMITTED.resolve()):
            continue
        shape = mode_shape(entry["id"], project)
        entries.append({**entry, **shape})
    return entries


def committed_listing() -> str:
    """`registry list`, restricted to the packages this repository ships.

    The tutorials index quotes the listing; quoting a machine's own output means the
    page is wrong on the next machine, so the quote is this function's output and a
    test keeps the two equal.
    """
    lines: list[str] = []
    for name in sorted(registry.installed_names(COMMITTED.parent)):
        try:
            package = registry.load(name, COMMITTED.parent)
        except Exception:
            continue
        if Path(registry.where(name, COMMITTED.parent) or "").resolve() != (COMMITTED / name).resolve():
            continue
        lines.append(f"{name}/  {package.description}")
        for mode in sorted(package.modes, key=lambda m: m.name):
            lines.append(f"  {f'{name}/{mode.name}':<38} {getattr(mode, 'description', '')}")
    return "\n".join(lines) + "\n"


def named_registry(module: Any, attribute: str) -> list[tuple[str, str]]:
    """A registry's names with the first line of each class's docstring."""
    found: dict[str, type] = getattr(module, attribute, {}) or {}
    out: list[tuple[str, str]] = []
    for name, cls in sorted(found.items()):
        doc = (getattr(cls, "__doc__", "") or "").strip().splitlines()
        out.append((name, doc[0].strip() if doc else ""))
    return out


def parse_options(help_text: str) -> list[str]:
    """Every long flag in an argparse `--help`, in the order it lists them."""
    return re.findall(r"(?m)^\s+(--[a-z][a-z-]*)", help_text)


def command_help(module: str) -> str:
    """One module's real `--help`, as the user would see it."""
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True, text=True, cwd=str(REPO_ROOT / "refactory"),
    )
    return result.stdout + result.stderr


def _generated_object(text: str, marker: str) -> dict[str, Any]:
    """The JSON object literal `text` assigns to `marker`, read as JSON.

    `gen_discovery` emits `generated-help.ts` with `json.dumps`, so the literal is
    valid JSON and can be decoded directly rather than pattern-matched as TypeScript.
    """
    start = text.index("{", text.index(marker) + len(marker))
    value, _end = json.JSONDecoder().raw_decode(text, start)
    if not isinstance(value, dict):
        raise ValueError(f"{marker!r} is not an object literal")
    return value


def tool_verbs() -> list[tuple[str, str]]:
    """The `refactory` tool's verbs and purposes, read from its generated table.

    The table lives in `generated-help.ts` (`CHAT`), not in `index.ts`;
    reading it keeps this page tied to the contract the tool ships.
    """
    chats = _generated_object(TOOL_SOURCE.read_text(), "export const CHAT = ")
    return [(str(entry["verb"]), str(entry["purpose"])) for entry in chats.values()]


# ── rendering ──────────────────────────────────────────────────────────────


def kind_chip(kind: str) -> str:
    """One node's kind, as a coloured chip."""
    return f'<span class="chip chip-{esc(kind)}">{esc(kind)}</span>'


def render_pipeline(shape: dict[str, Any]) -> str:
    """The graph as a readable pipeline: phases, with a fork's branches grouped."""
    rows: list[str] = []
    index = 0
    walk = shape["walk"]
    while index < len(walk):
        depth, node = walk[index]
        detail = ""
        if node.kind == "op":
            detail = code(node.call.split(":")[-1])
        elif node.kind == "agent":
            detail = '<span class="muted">a model call, run as a subagent</span>'
        elif node.kind == "gate":
            detail = code(f"{node.gate_kind} gate")
            if node.gate_kind == "user":
                detail += ' <strong>— a person answers this one</strong>'
        elif node.kind in ("fork", "data"):
            nested: list[str] = []
            index += 1
            while index < len(walk) and walk[index][0] > depth:
                nested_depth, nested_node = walk[index]
                nested.append(f'<li>{node_row(nested_node, nested_depth - depth)}</li>')
                index += 1
            label = "branches, together" if node.kind == "fork" else "each item runs this"
            detail = f' <span class="muted">— {esc(label)}</span>'
            rows.append(
                f'<li class="phase phase-{esc(node.kind)}"><div class="phase-head">{kind_chip(node.kind)}'
                f'<span class="phase-id">{esc(node.id)}</span>{detail}</div>'
                f'<ul class="branches">{"".join(nested)}</ul></li>'
            )
            continue
        rows.append(node_row(node, depth, detail))
        index += 1
    return '<ol class="pipeline">' + "".join(rows) + "</ol>"


def node_row(node: Node, depth: int, detail: str = "") -> str:
    """One node in a pipeline walk."""
    if not detail:
        if node.kind == "op":
            detail = code(node.call.split(":")[-1])
        elif node.kind == "agent":
            detail = '<span class="muted">a model call, run as a subagent</span>'
        elif node.kind == "gate":
            detail = code(f"{node.gate_kind} gate")
            if node.gate_kind == "user":
                detail += ' <strong>— a person answers this one</strong>'
    writes = ", ".join(node.writes)
    produces = ", ".join(node.produces)
    trail = []
    if writes:
        trail.append(f"writes {code(writes)}")
    if produces:
        trail.append(f"produces {code(produces)}")
    return (
        f'<li class="phase" style="--depth:{depth}">'
        f'<div class="phase-head">{kind_chip(node.kind)}<span class="phase-id">{esc(node.id)}</span>'
        f'<span class="phase-detail">{detail}</span></div>'
        f'{f"<div class=\'phase-trail\'>{chr(32).join(trail)}</div>" if trail else ""}'
        "</li>"
    )


def render_mode_table(entries: list[dict[str, Any]]) -> str:
    """Every mode, with what it is and whether it calls a model."""
    rows = []
    for entry in sorted(entries, key=lambda item: item["id"]):
        knobs = ", ".join(entry["knobs"]) or "<span class='muted'>none</span>"
        calls = (
            f'<span class="yes">{entry["agents"]} agent node(s)</span>'
            if entry["agents"] else '<span class="muted">none — deterministic</span>'
        )
        action = entry.get("action", "start")
        badge = (
            '<span class="badge badge-run">run it</span>' if action == "start"
            else f'<span class="badge badge-author">{esc(action)}</span>'
        )
        rows.append(
            "<tr>"
            f'<td><code>{esc(entry["id"])}</code><br><span class="muted">{esc(entry["package"])}</span></td>'
            f'<td>{esc(entry["description"])}</td>'
            f'<td>{badge}</td>'
            f"<td>{calls}</td>"
            f'<td class="knobs">{knobs}</td>'
            "</tr>"
        )
    return (
        '<table class="modes"><thead><tr>'
        "<th>mode</th><th>what it does</th><th>to use it</th><th>model calls</th><th>knobs</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def render_kinds() -> str:
    """The node-kind legend."""
    return "".join(
        f'<li>{kind_chip(kind)} <span class="muted">{esc(meaning)}</span></li>' for kind, meaning in KINDS
    )


def render_named(rows: list[tuple[str, str]], first: str, second: str) -> str:
    """A two-column table of names and their own descriptions."""
    body = "".join(
        f"<tr><td><code>{esc(name)}</code></td><td>{esc(description) or '<span class=\'muted\'>—</span>'}</td></tr>"
        for name, description in rows
    )
    return f"<table><thead><tr><th>{esc(first)}</th><th>{esc(second)}</th></tr></thead><tbody>{body}</tbody></table>"


def render_options() -> str:
    """The outer loop's options, with defaults."""
    body = "".join(
        f"<tr><td><code>{esc(flag)}</code></td><td>{esc(explanation)}</td><td><code>{esc(default)}</code></td></tr>"
        for flag, explanation, default in LOOP_OPTIONS
    )
    return (
        "<table><thead><tr><th>option</th><th>what it does</th><th>default</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def render_steps() -> str:
    """The loop's eight steps."""
    body = "".join(
        f'<li><code>{esc(name)}</code> <span class="muted">— {esc(what)}</span></li>' for name, what in LOOP_STEPS
    )
    return f'<ol class="steps">{body}</ol>'


def render_presets() -> str:
    """The edit presets and the operations each one allows."""
    body = "".join(
        f"<tr><td><code>{esc(name)}</code></td><td>{esc(', '.join(sorted(preset.get('ops', ())))) if isinstance(preset, dict) else esc(preset)}</td></tr>"
        for name, preset in sorted(edits_module.PRESETS.items())
    )
    return f"<table><thead><tr><th>preset</th><th>operations allowed</th></tr></thead><tbody>{body}</tbody></table>"


def render_console_verbs() -> str:
    """The console's verbs, with the question each answers."""
    body = "".join(
        f"<tr><td><code>{esc(verb)}</code></td><td>{esc(question)}</td></tr>" for verb, question in CONSOLE_VERBS
    )
    return f"<table><thead><tr><th>command</th><th>the question it answers</th></tr></thead><tbody>{body}</tbody></table>"


def render_ceo_verbs() -> str:
    """The `refactory` tool's verbs, read from its own table."""
    body = "".join(
        f"<tr><td><code>{esc(verb)}</code></td><td>{esc(help_text)}</td></tr>" for verb, help_text in tool_verbs()
    )
    return f"<table><thead><tr><th>action</th><th>what it does</th></tr></thead><tbody>{body}</tbody></table>"


def example_block(label: str, command: str, output: str) -> str:
    """One command with the output it produced, on one run."""
    return (
        f'<figure class="example"><figcaption>{esc(label)}</figcaption>'
        f'<div class="cmd"><pre><code>{esc(command)}</code></pre>'
        f'<button class="copy" type="button" data-copy>copy</button></div>'
        f'<pre class="out"><code>{esc(output)}</code></pre></figure>'
    )


def render_recipes() -> str:
    """Every documented command, from the one place that also checks them.

    The page and `check_recipes` read the same recipes, so a command cannot be shown
    without also being runnable -- and the tier badge says honestly which ones cost a
    model call.
    """
    from refactory_dsh.tools import recipes as recipes_module

    blocks: list[str] = []
    for group, group_recipes in recipes_module.by_group().items():
        rows = []
        for recipe in group_recipes:
            badge = {
                "cheap": '<span class="badge badge-run">offline</span>',
                "model": '<span class="badge badge-author">model calls</span>',
                "manual": '<span class="badge">by hand</span>',
            }[recipe.tier]
            rows.append(
                f'<h4>{esc(recipe.title)} {badge}</h4>'
                f'<div class="cmd"><pre><code>{esc(recipe.command)}</code></pre>'
                f'<button class="copy" type="button" data-copy>copy</button></div>'
                f'<p class="muted">{esc(recipe.expect)}</p>'
            )
        blocks.append(
            f'<details open><summary>{esc(recipes_module.GROUP_TITLES[group])}</summary>'
            f'{"".join(rows)}</details>'
        )
    return "".join(blocks)


def render_task_hooks() -> str:
    """The Task protocol's three hooks, from the class that declares them."""
    from refactory_dsh.task import Task

    rows = []
    for name in ("instances", "setup", "verify"):
        hook = getattr(Task, name)
        doc = (hook.__doc__ or "").strip().splitlines()
        rows.append((f"{name}()", doc[0] if doc else ""))
    return render_named(rows, "hook", "what it is for")


def render_ledger_kinds() -> str:
    """The intent ledger's entry kinds, from the module that owns them."""
    from refactory_dsh import steering

    kinds = [
        ("seed", steering.SEED_KIND if hasattr(steering, "SEED_KIND") else "seed",
         "what the run is for, in the asker's words; written once, before the lowering"),
        ("note", "info | correction | constraint",
         "something said while it runs; a constraint must be respected, not merely known"),
        ("verdict", "verdict",
         "the answer a gate was routed on, so a resumed run replays it instead of asking again"),
    ]
    return render_named([(k, f"{what} — {why}") for k, what, why in kinds], "entry", "what it carries")


def render_drift_gates() -> str:
    """Which gate proves what, so a reader knows what "tested" means here."""
    rows = (
        ("uv run python -m pytest refactory/tests -q", "the glue: the language, the modes, the loop, the console"),
        ("pnpm run test", "the eight packages: the spine, the three bundles, the console and the client tab"),
        ("uv run python -m pytest refactory/tests -q --run-integration", "one mode on both runtimes, compared fact by fact"),
        ("uv run python -m refactory_dsh.tools.check_recipes", "every command this page shows, executed"),
        ("uv run python -m refactory_dsh.tools.gen_tutorial --check", "this page matches the code it describes"),
        ("uv run python -m refactory_dsh.tools.gen_api_reference --check", "the API reference matches the signatures"),
    )
    return render_named(rows, "gate", "what it proves")


def render(project: Path | None = None) -> str:
    """The whole page."""
    project = Path(project or REPO_ROOT)
    entries = catalogue(project)
    create = next(entry for entry in entries if entry["id"] == "default_pack/create")
    design = next(entry for entry in entries if entry["id"] == "default_pack/design")
    loop_help = command_help("refactory_dsh.loop")
    create_help = command_help("refactory_dsh.create")
    outers = named_registry(strategies_module, "STRATEGIES")
    optimizers = named_registry(optim, "REGISTRY")
    evaluator_rows = named_registry(evaluators, "REGISTRY")

    return TEMPLATE.format(
        modes=render_mode_table(entries),
        mode_count=len(entries),
        agent_modes=sum(1 for entry in entries if entry["agents"]),
        kinds=render_kinds(),
        create_pipeline=render_pipeline(create),
        create_knobs=render_named(
            [(name, description) for name, description in create["knobs"].items()],
            "create knob", "what it does",
        ),
        design_pipeline=render_pipeline(design),
        design_knobs=render_named(
            [(name, description) for name, description in design["knobs"].items()],
            "design knob", "what it does",
        ),
        strategies=render_named(outers, "strategy", "what it is"),
        optimizers=render_named(optimizers, "optimizer", "how it proposes"),
        evaluators=render_named(evaluator_rows, "evaluator", "which runtime scores"),
        presets=render_presets(),
        options=render_options(),
        steps=render_steps(),
        console_verbs=render_console_verbs(),
        recipes=render_recipes(),
        dataset_pipeline=render_pipeline(
            next(entry for entry in entries if entry["id"] == "dataset_probe/dataset")
        ),
        task_hooks=render_task_hooks(),
        ledger_kinds=render_ledger_kinds(),
        drift_gates=render_drift_gates(),
        ceo_verbs=render_ceo_verbs(),
        ceo_actions=len(tool_verbs()),
        loop_help=esc(loop_help),
        create_help=esc(create_help),
        loop_defaults=render_named(
            [(field, f"default {LoopConfig.__dataclass_fields__[field].default!r}") for field in (
                "epochs", "batch_size", "optimizer", "edit_preset", "seed", "elite",
                "max_evaluations", "attempts", "evaluator", "dsh_timeout_s",
            )],
            "LoopConfig field", "default",
        ),
        use_example=example_block(
            "the same step on the runtime that ships: a real DSH run, its artifacts on disk",
            ".venv/bin/python -m refactory_dsh.interactive counting_pack/fixed_step \\\n"
            '  --intent "count up to five"',
            '{\n  "mode": "counting_pack-fixed_step",\n  "session_id": "session-a48dee17-…",\n'
            '  "workdir": "<checkout>/.factory/interactive/counting_pack-fixed_step",\n'
            '  "final": "The workflow graph is already at **done** — all four nodes\n'
            '            (`start`, `step`, `check`, `finish`) completed, with the\n'
            '            `check` gate retrying twice and then proceeding."\n}\n\n'
            '// .factory/interactive/counting_pack-fixed_step/result.json\n'
            '{\n  "final_count": 6,\n  "steps_taken": 3,\n  "closeness": 0.5\n}\n'
            "// plus facts.json, trace.json, state.json, intent-ledger.md and graphs/",
        ),
        counter_example=example_block(
            "counting_pack/fixed_step, one epoch of four candidates — seconds, no model",
            ".venv/bin/python -m refactory_dsh.loop counting_pack/fixed_step \\\n"
            "  --epochs 1 --batch-size 4 --optimizer random --seed 5 --runs-root /tmp/tour/run",
            '{\n  "mode": "counting_pack/fixed_step",\n  "optimizer": "random",\n'
            '  "evaluator": "walker",\n  "bound": "epochs",\n  "evaluations": 4,\n'
            '  "seed_score": 0.46,\n  "best_score": 0.96,\n  "raw_best_score": 1.0,\n'
            '  "scoring": {"compared": "penalized", "parsimony": 0.01, ...},\n'
            '  "holdout": {"train": 1.0, "holdout": null, "overfit": false, ...},\n'
            '  "structural_wins": 1,\n'
            '  "rejection_codes": {"E-shape": 3, "V2": 3, "V7": 6, ...},\n'
            '  "report": "/tmp/tour/run/report.md"\n}',
        ),
    )


#: The page. Static prose and layout; every table and diagram arrives as a slot.
TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>refactory — create a mode, use it, optimize it, use it again</title>
<style>
  :root {{
    --bg: #fbfaf8; --fg: #1c1b19; --muted: #6b6862; --line: #e3ded6; --card: #fff;
    --accent: #4a5cc4; --accent-soft: #eceefb; --code-bg: #f5f3ef;
    --op: #55606e; --agent: #4a5cc4; --gate: #b06a00; --fork: #0f766e; --join: #0f766e; --data: #2f7d32;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #16171a; --fg: #eceae5; --muted: #a09c94; --line: #2e3034; --card: #1e2023;
      --accent: #9aa7ff; --accent-soft: #232744; --code-bg: #232529;
      --op: #9aa5b1; --agent: #9aa7ff; --gate: #e0a83c; --fork: #5eead4; --join: #5eead4; --data: #86d98a;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--fg);
    font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}
  a {{ color: var(--accent); }}
  code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.9em; }}
  code {{ background: var(--code-bg); padding: 0.1em 0.35em; border-radius: 4px; }}
  pre {{ background: var(--code-bg); padding: 0.9rem 1rem; border-radius: 10px; overflow-x: auto; margin: 0; }}
  pre code {{ background: none; padding: 0; }}
  .wrap {{ display: grid; grid-template-columns: 250px minmax(0, 1fr); gap: 2.5rem;
           max-width: 1240px; margin: 0 auto; padding: 0 1.5rem 6rem; }}
  header.hero {{ grid-column: 1 / -1; padding: 3.5rem 0 1.5rem; border-bottom: 1px solid var(--line); }}
  header.hero h1 {{ margin: 0 0 0.5rem; font-size: 2.1rem; letter-spacing: -0.02em; }}
  header.hero p {{ margin: 0; color: var(--muted); max-width: 62ch; }}
  .chips {{ display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 1.2rem; }}
  .chips span {{ background: var(--accent-soft); color: var(--accent); border-radius: 999px;
                 padding: 0.25rem 0.7rem; font-size: 0.82rem; font-weight: 600; }}
  nav {{ position: sticky; top: 1.5rem; align-self: start; font-size: 0.9rem; }}
  nav ol {{ list-style: none; margin: 0; padding: 0; }}
  nav li {{ margin: 0.35rem 0; }}
  nav a {{ text-decoration: none; color: var(--muted); }}
  nav a:hover {{ color: var(--accent); }}
  main {{ min-width: 0; }}
  section {{ padding: 2.5rem 0 1rem; border-bottom: 1px solid var(--line); }}
  section:last-child {{ border-bottom: 0; }}
  h2 {{ font-size: 1.5rem; margin: 0 0 0.4rem; letter-spacing: -0.01em; }}
  h3 {{ font-size: 1.1rem; margin: 2rem 0 0.5rem; }}
  .lede {{ color: var(--muted); margin: 0 0 1.2rem; max-width: 74ch; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.92rem; }}
  th, td {{ text-align: left; padding: 0.55rem 0.7rem; border-bottom: 1px solid var(--line); vertical-align: top; }}
  th {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
  tbody tr:hover {{ background: var(--card); }}
  td.knobs {{ color: var(--muted); }}
  .muted {{ color: var(--muted); }}
  .yes {{ color: var(--agent); font-weight: 600; }}
  .chip {{ display: inline-block; min-width: 3.6rem; text-align: center; border-radius: 999px;
           padding: 0.05rem 0.5rem; font-size: 0.75rem; font-weight: 700; color: #fff;
           background: var(--op); letter-spacing: 0.02em; }}
  .chip-agent {{ background: var(--agent); }}
  .chip-gate {{ background: var(--gate); }}
  .chip-fork, .chip-join {{ background: var(--fork); }}
  .chip-data {{ background: var(--data); }}
  .badge {{ border-radius: 6px; padding: 0.1rem 0.45rem; font-size: 0.78rem; font-weight: 600; }}
  .badge-run {{ background: var(--accent-soft); color: var(--accent); }}
  .badge-author {{ background: #fdf0d8; color: #8a5a00; }}
  @media (prefers-color-scheme: dark) {{ .badge-author {{ background: #3a2f16; color: #e0a83c; }} }}
  .pipeline, .steps, .legend {{ list-style: none; margin: 1rem 0; padding: 0; }}
  .pipeline .phase {{ border-left: 2px solid var(--line); padding: 0.45rem 0 0.45rem 1rem; margin-left: 0.35rem; }}
  .pipeline .phase:last-child {{ border-left-color: transparent; }}
  .phase-head {{ display: flex; gap: 0.6rem; align-items: baseline; flex-wrap: wrap; }}
  .phase-id {{ font-weight: 600; }}
  .phase-detail {{ color: var(--muted); font-size: 0.9rem; }}
  .phase-trail {{ color: var(--muted); font-size: 0.82rem; margin-top: 0.15rem; }}
  .phase-fork {{ border-left-color: var(--fork); }}
  .branches {{ list-style: none; margin: 0.4rem 0 0.2rem; padding: 0 0 0 1.2rem; border-left: 2px dashed var(--fork); }}
  .branches .phase {{ border-left: 0; padding-left: 0; margin-left: 0; }}
  .legend li {{ margin: 0.3rem 0; }}
  .steps li {{ margin: 0.4rem 0; }}
  .example {{ margin: 1.5rem 0; }}
  .example figcaption {{ font-size: 0.85rem; color: var(--muted); margin-bottom: 0.4rem; }}
  .cmd {{ position: relative; }}
  .cmd .copy {{ position: absolute; top: 0.5rem; right: 0.5rem; border: 1px solid var(--line);
                background: var(--card); color: var(--muted); border-radius: 6px; font-size: 0.75rem;
                padding: 0.15rem 0.5rem; cursor: pointer; }}
  .out {{ margin-top: 0.5rem; }}
  details {{ border: 1px solid var(--line); border-radius: 10px; padding: 0.6rem 1rem; margin: 1rem 0; background: var(--card); }}
  summary {{ cursor: pointer; font-weight: 600; }}
  .callout {{ border-left: 3px solid var(--accent); background: var(--card); padding: 0.9rem 1.1rem;
              border-radius: 0 10px 10px 0; margin: 1.2rem 0; }}
  .callout.warn {{ border-left-color: var(--gate); }}
  .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
  @media (max-width: 900px) {{ .wrap {{ grid-template-columns: 1fr; }} nav {{ display: none; }} .two {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="wrap">
  <header class="hero">
    <h1>Create a mode → use it → optimize it → use it again</h1>
    <p>Everything in this checkout in one page: what every mode is, how <strong>create-v2</strong> and
    <strong>design-v2</strong> build one, what the <strong>outer loop</strong> is and every option it takes, and how
    to ask the CEO for any of it in the GUI. Commands are the real ones, output is the real output.</p>
    <div class="chips">
      <span>{mode_count} modes</span>
      <span>{agent_modes} call a model</span>
      <span>2 runtimes: walker &amp; DSH</span>
      <span>{ceo_actions} CEO actions</span>
    </div>
  </header>

  <nav>
    <ol>
      <li><a href="#shape">The four steps</a></li>
      <li><a href="#mode">What a mode is</a></li>
      <li><a href="#modes">Every mode</a></li>
      <li><a href="#semantics">How it runs</a></li>
      <li><a href="#create">create-v2</a></li>
      <li><a href="#design">design-v2</a></li>
      <li><a href="#loop">The outer loop</a></li>
      <li><a href="#dataset">Datasets and Tasks</a></li>
      <li><a href="#steer">Steering</a></li>
      <li><a href="#drift">Keeping it honest</a></li>
      <li><a href="#read">Reading a run</a></li>
      <li><a href="#gui">Asking the CEO (GUI)</a></li>
      <li><a href="#commands">Every command</a></li>
      <li><a href="#where">Where things live</a></li>
    </ol>
  </nav>

  <main>
    <section id="shape">
      <h2>The four steps</h2>
      <p class="lede">One mode, four commands, and every other page in this repository is the detail behind one
      of them.</p>
      <table>
        <thead><tr><th>step</th><th>command</th><th>what you have at the end</th></tr></thead>
        <tbody>
          <tr><td><strong>create</strong></td>
              <td><code>uv run refactory ceo "&lt;what it does&gt;" --mode create --name my_pack</code></td>
              <td>a package on disk, registered, its mode smoke-scored</td></tr>
          <tr><td><strong>use it</strong></td>
              <td><code>uv run refactory ceo my_pack/&lt;mode&gt; --intent "&lt;what you want&gt;"</code></td>
              <td>the mode ran on DSH, with its artifacts, and a person answered any gate</td></tr>
          <tr><td><strong>optimize it</strong></td>
              <td><code>uv run refactory optimize my_pack/&lt;mode&gt; --epochs 4 --batch-size 6 --optimizer random --seed 5</code></td>
              <td>a leaderboard, a report, and the edits that improved it</td></tr>
          <tr><td><strong>use it again</strong></td>
              <td><code>uv run refactory adopt --run &lt;run id&gt; --as my_tuned</code></td>
              <td>the winner as a mode of its own, smoke-scored by the registry</td></tr>
        </tbody>
      </table>
      {use_example}
      <p class="muted"><code>interactive</code> runs the mode on DSH under the CEO, with your terminal
      attached so a <code>user</code> gate reaches <em>you</em>; the run's directory holds
      <code>facts.json</code>, <code>trace.json</code>, <code>state.json</code> and the intent ledger.
      <code>--workdir</code> is optional and is resolved to an absolute path, because DSH takes it as the
      session's working directory.</p>
      <div class="callout warn">
        <strong>The one thing that trips everybody up.</strong> Apart from <code>create</code>,
        <code>edit</code> and <code>design</code>, every mode in the box is deterministic <code>op</code> and
        <code>gate</code> nodes — they are offline fixtures, not examples of what a mode looks like. A mode calls
        a model when its graph has <code>agent</code> nodes, and <strong>only the brief decides that</strong>: ask
        for "an agent that drafts …" and the created mode makes that call. A mode with an agent node also needs a
        model <em>at eval time</em>, so <code>registry smoke</code> is not how you use it — it fails with
        <code>agent node '…' needs a model client</code>. Use <code>interactive</code>, the panel, or the CEO.
      </div>
    </section>

    <section id="mode">
      <h2>What a mode is</h2>
      <p class="lede">A Python class: a <strong>graph</strong> of nodes with typed <strong>knobs</strong>, and a
      <code>score()</code>. The graph is the architecture, the knobs are the parameters, and both are one
      serializable state dict — which is what lets an optimizer propose "set <code>grid</code> to 18" and
      "duplicate <code>accept</code>" as typed edits rather than as text.</p>
      <h3>The node kinds</h3>
      <ul class="legend">{kinds}</ul>
      <p>Only <code>agent</code> nodes call a model. Everything else is deterministic, which is why the fixtures
      are cheap to run and the loop can evaluate hundreds of candidates.</p>
    </section>

    <section id="modes">
      <h2>Every mode</h2>
      <p class="lede">Read from the registry: this is the same JSON the <code>refactory</code> tool, the GUI panel
      and the CEO see. <span class="badge badge-run">run it</span> means the outer loop can search it;
      <span class="badge badge-author">create/edit/design</span> means it <em>makes</em> something and is run by
      its own command instead.</p>
      {modes}
    </section>

    <section id="create">
      <h2>Create</h2>
      <p class="lede"><code>default_pack/create</code> is the current public authoring pipeline, with a
      research phase, a strategy phase a <strong>person</strong> approves, a build, three reviews in parallel, an
      audit of their evidence, then registration and a smoke run. Its output is a package of modes, written as
      Python.</p>
      <p><code>default_pack/create_fidelity</code> is an opt-in candidate restoring the pinned
      remote-factory create-v2 graph and director protocol. Its controlled tests do not establish live
      model fidelity. It supports portable packages; update and plugin variants remain unsupported.
      Public Create retains its existing behavior until the paired live benchmark is complete.</p>
      <h3>The pipeline, as the graph declares it</h3>
      {create_pipeline}
      <h3>What it produces</h3>
      <p>A package directory in the workspace (<code>.factory/packages/&lt;name&gt;</code>) holding
      <code>__init__.py</code>, the mode module and its ops module; registered, validated, and smoke-scored. The
      intended <em>structure</em> of the mode is decided by the design node and checked by the validate gate
      before any review runs.</p>
      <h3>Its knobs</h3>
      {create_knobs}
      <h3>Running it</h3>
      <figure class="example">
        <figcaption>a real create, start to finish</figcaption>
        <div class="cmd"><pre><code>.venv/bin/python -m refactory_dsh.create \\
  --name annealing_pack \\
  --brief "hill-climb a greedy circle packing by accepting non-worsening moves" \\
  --extends circle_packing/adaptive \\
  --verdict gate_strategy=proceed</code></pre>
        <button class="copy" type="button" data-copy>copy</button></div>
      </figure>
      <p><code>--verdict</code> is the person's answer to <code>gate_strategy</code>, and dropping it is the one
      mistake worth naming: the run stops at that gate and tells you so, because a creation that approved its own
      plan is not a reviewed creation. On a paused run,
      <code>console steer --verdict gate_strategy=proceed</code> records the same answer and a resume replays
      it.</p>
      <div class="callout">
        <strong>Ask for a model call in the brief.</strong> "Its graph must start with an agent node that
        drafts …" is the difference between a mode that calls a model and one that does not. Ask for a
        <code>data</code> node and a <code>Task</code> (<code>--data-node --task my_pack.tasks:MyTask</code>) and
        the validate gate <em>checks</em> you got one instead of taking your word for it.
      </div>
      <details><summary>Every option, from the command's own <code>--help</code></summary><pre><code>{create_help}</code></pre></details>
    </section>

    <section id="design">
      <h2>design-v2</h2>
      <p class="lede"><code>default_pack/design</code> is design-v2: the same v2 head — research, strategy, a
      person's approval — but its output is a checked <strong>artifact</strong> rather than a package. The
      <code>design_doc</code> node writes the design the code is then written from, and the check gate runs
      <em>your</em> command to decide whether the artifact works.</p>
      <h3>The pipeline</h3>
      {design_pipeline}
      <h3>Its knobs</h3>
      {design_knobs}
      <h3>Running it</h3>
      <figure class="example">
        <figcaption>a design whose proof is a command you name</figcaption>
        <div class="cmd"><pre><code>.venv/bin/python -m refactory_dsh.design \\
  --name wordcount_tool \\
  --brief "a CLI that counts words, with tests" \\
  --check "pytest -q" \\
  --verdict gate_strategy=proceed</code></pre>
        <button class="copy" type="button" data-copy>copy</button></div>
      </figure>
    </section>

    <section id="loop">
      <h2>The outer loop</h2>
      <p class="lede">A mode is one point in a space of modes: its knobs can be re-set and its graph can be
      re-wired. The outer loop searches that space. It proposes typed edits, refuses the ones that break the
      graph, evaluates what survives, and keeps what improves — and it reports what it refused, so a run that
      found nothing says why.</p>
      <h3>The eight steps</h3>
      {steps}
      <h3>What it is not</h3>
      <p>It is not "ask a model to rewrite the mode". Every edit is from a closed catalogue of operations
      (<code>set_knob</code>, <code>duplicate</code>, <code>insert_after</code>, …), every candidate is validated
      <em>before</em> it may cost a run, and the score comes from running the candidate, not from an opinion about
      it. The one model-driven optimizer, <code>autoresearch</code>, is given the traces and proposes from the
      same catalogue as everyone else.</p>

      <h3>Every option</h3>
      {options}
      <p class="muted">Proven complete against the command's own <code>--help</code> by the generator's test.</p>
      <details><summary>The real <code>--help</code></summary><pre><code>{loop_help}</code></pre></details>
      <details><summary>What the config defaults to</summary>{loop_defaults}</details>

      <h3>Strategies — which outer loop drives the search</h3>
      <p>A strategy is the <em>shape</em> of the loop: one pipeline, or N branches that merge, or a staged
      curriculum, or a tree. They are ports of SRF's own loops, and the differences are in the docstrings they
      carry.</p>
      {strategies}

      <h3>Optimizers — how the next candidates are proposed</h3>
      <p>An optimizer is the <em>policy</em>: which parent to breed from, how far to move, whether to keep a
      population. <code>random</code> samples the catalogue, <code>hill_climb</code> mutates the best so far, and
      <code>autoresearch</code> asks a model to read the traces.</p>
      {optimizers}

      <h3>Evaluators — which runtime scores a candidate</h3>
      {evaluators}
      <div class="callout">
        <code>walker</code> is fast and in-process, so it is what a search uses. <code>dsh</code> is the runtime
        that ships; the two are kept honest by running the same mode on both and comparing the facts. Use
        <code>--evaluator dsh</code> when the question is "does this survive on the real runtime", not "is this
        better".
      </div>

      <h3>Edit presets — which operations are allowed</h3>
      {presets}

      <h3>A run you can do in seconds</h3>
      {counter_example}
      <p>Read it back with <code>console leaderboard --run /tmp/tour/run</code>: every row is a candidate with the
      diff against its parent, so the table says <em>what changed</em>, not just which number is bigger.</p>
    </section>

    <section id="read">
      <h2>Reading a run</h2>
      <p class="lede">A run is a directory. Everything a person asks about it is a console verb, which is also
      what the tool and the GUI panel call.</p>
      {console_verbs}
      <div class="callout">
        <strong>Adopting the winner is the whole point.</strong> <code>adopt</code> writes the winning state as a
        new package and leaves the input untouched, so the tuned mode is a mode like any other: smoke it, run it,
        or put it back in the loop.
      </div>
    </section>

    <section id="gui">
      <h2>Asking the CEO (the GUI)</h2>
      <p class="lede">In the GUI you do not type these commands; you ask, and the CEO calls the
      <code>refactory</code> tool — the same console, the same JSON. Its coding tools are disabled on purpose: it
      can choose and start a run, read it, manage packages and author a mode, and it cannot wander off and
      implement something itself.</p>

      <h3>What to type</h3>
      <table>
        <thead><tr><th>you say</th><th>what the CEO does</th></tr></thead>
        <tbody>
          <tr><td>"What modes can I run?"</td><td><code>modes</code> — the catalogue with each mode's parameters</td></tr>
          <tr><td>"List the outer loops and optimizers"</td><td><code>strategies</code> — then it can choose settings by name</td></tr>
          <tr><td>"Start a run of counting_pack/fixed_step with the random optimizer"</td><td><code>start</code> with <code>settings.optimizer</code>, returning the run id</td></tr>
          <tr><td>"Where is that run?"</td><td><code>status</code> — state, epoch, best score, and what happened since</td></tr>
          <tr><td>"Show me the leaderboard"</td><td><code>leaderboard</code> — candidates with their diffs against their parents</td></tr>
          <tr><td>"Which change won, and what did it change?"</td><td><code>best</code>, then <code>candidate</code> for one row</td></tr>
          <tr><td>"Keep the winner as a package called my_tuned"</td><td><code>adopt</code> — writes the package, input untouched</td></tr>
          <tr><td>"This run is for making the packing tighter"</td><td><code>steer</code> — writes the intent ledger the agents read</td></tr>
          <tr><td>"Create a mode that drafts a blurb with an agent and checks it"</td><td><code>create</code> — the model-driven authoring mode</td></tr>
          <tr><td>"Show me the packages, then disable annealing_pack"</td><td><code>registry</code> with <code>list</code>, then <code>disable</code></td></tr>
        </tbody>
      </table>

      <h3>Every action it has</h3>
      <p>Read from the tool's own verb table, so this list is the tool's list.</p>
      {ceo_verbs}

      <h3>The panel, if you would rather click</h3>
      <p>Start the GUI from this checkout (<code>uv run refactory ui</code>), open the printed token URL, pick the
      workspace, create a session, then <strong>Open the sidebar → Modes</strong>. A blank session can open
      the panel without an introductory model call. Choose <strong>Start</strong> to optimize a scored mode
      or <strong>Run once</strong> to execute an enabled mode, including authoring modes. Supply its required
      knobs first. The installed profile composes the Modes panel and the console it reads.</p>
      <p>Attach an exact run directory to stream durable graph/worker activity, see pending questions and
      use boundary controls. Reading a saved run does not execute it; recovery requires explicit resume.
      See <a href="interfaces.md">Terminal and GUI</a> for controls and
      <a href="release-validation-20260926.md">release validation</a> for the remaining fast-mode acceptance
      gap. New workers default to <a href="models.md#model-provider-versus-coding-agent">full native permission
      bypass</a>; saved policies are retained.</p>

      <div class="callout warn">
        <strong>A person's gate is still a person's.</strong> If a mode has a <code>user</code> gate (create's
        <code>gate_strategy</code> is one), the runtime asks <em>you</em> — the terminal answerer in
        <code>interactive</code>, or the GUI's own dialog. The CEO cannot answer it, and neither can the model.
      </div>
    </section>

    <section id="semantics">
      <h2>How a mode becomes something that runs</h2>
      <p class="lede">One page of semantics, because everything else follows from it: a
      mode is data, the data is lowered to one IR, and two runtimes execute that IR.</p>
      <svg viewBox="0 0 900 260" role="img" aria-label="A mode is lowered to an IR, which two runtimes execute and report facts from" style="width:100%;height:auto;margin:1rem 0">
        <defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="currentColor"/></marker></defs>
        <g fill="none" stroke="currentColor" stroke-width="1.5" marker-end="url(#arrow)" opacity="0.7">
          <path d="M210,70 H290"/><path d="M450,70 H530"/>
          <path d="M490,120 V160"/><path d="M490,200 V232"/>
        </g>
        <g font-size="13" fill="currentColor">
          <rect x="20" y="30" width="190" height="80" rx="10" fill="var(--card)" stroke="var(--line)"/>
          <text x="34" y="56" font-weight="700">a mode (Python)</text>
          <text x="34" y="76">nodes · knobs · score()</text>
          <text x="34" y="96" opacity="0.7">state_dict() → JSON</text>

          <rect x="300" y="30" width="150" height="80" rx="10" fill="var(--card)" stroke="var(--line)"/>
          <text x="314" y="56" font-weight="700">the IR</text>
          <text x="314" y="76">Workflow.to_dict()</text>
          <text x="314" y="96" opacity="0.7">lowering.py</text>

          <rect x="540" y="18" width="170" height="46" rx="10" fill="var(--card)" stroke="var(--line)"/>
          <text x="554" y="46">walker — in-process</text>
          <rect x="540" y="76" width="170" height="46" rx="10" fill="var(--card)" stroke="var(--line)"/>
          <text x="554" y="104">DSH — the one that ships</text>

          <rect x="360" y="160" width="260" height="40" rx="10" fill="var(--card)" stroke="var(--line)"/>
          <text x="374" y="185">facts.json — what the run produced</text>
          <rect x="360" y="212" width="260" height="40" rx="10" fill="var(--card)" stroke="var(--line)"/>
          <text x="374" y="237">score(facts) — the number</text>
        </g>
      </svg>
      <ul>
        <li><strong>The mode is data.</strong> Its architecture and its parameters are one
        serializable dict, so an optimizer can propose <code>set_knob</code> or
        <code>duplicate</code> as a typed edit, and a candidate is a value you can diff —
        not a code change somebody has to read.</li>
        <li><strong>The IR is the one interface.</strong> <code>lowering.py</code> turns the
        state into <code>Workflow.to_dict()</code>; the runtime knows nothing about
        <code>Mode</code>, <code>Knob</code> or <code>score()</code>. That is why the same
        graph runs on both runtimes and why &quot;a mode means the same thing&quot; is a
        testable claim rather than a hope.</li>
        <li><strong>Facts, then a number.</strong> A node declares what it reads, writes and
        produces; the executor enforces it, and <code>score()</code> reads the facts. A run
        that lied about its artifacts fails instead of scoring.</li>
      </ul>
    </section>

    <section id="dataset">
      <h2>Datasets, Tasks and data nodes</h2>
      <p class="lede">A mode can walk a subgraph once per <em>item</em> — one benchmark
      instance, one row of a file, one directory — and a <strong>Task</strong> can supply
      those items and judge each pass.</p>
      {dataset_pipeline}
      <p>Each pass writes the item to <code>current_item.json</code> in the run directory, so
      an op or an agent node reads the item it is working on rather than being told it in a
      prompt. Reusable harnesses take items from a <code>source_path</code> read as
      <code>directory</code>/<code>jsonl</code>/<code>csv</code>, or the bound Task's
      <code>instances()</code>; <code>limit</code> caps them; a named source that is missing is
      refused by name, because an empty selection and an unbound source must not look alike.
      The verdicts are recorded as <code>item_scores</code>, and <code>score()</code> reads
      what the Task decided instead of a number the graph guessed.</p>
      <h3>The Task protocol</h3>
      {task_hooks}
      <p>Both runtimes load items through one implementation
      (<code>nn.executor.load_items</code>), so the items a mode selects do not depend on which
      runtime is running it, and the equivalence test runs one data-driven mode on each and
      compares the facts. <a href="../docs/datasets.md">datasets.md</a> is the long form.</p>
    </section>

    <section id="steer">
      <h2>Steering a run</h2>
      <p class="lede">A run is accountable to what it was asked for. The <strong>intent
      ledger</strong> is where that lives: it is seeded before the lowering, folded into every
      agent node's prompt, and read back when the run is judged.</p>
      <figure class="example">
        <figcaption>steering is a console verb, so the CEO does the same thing when you ask in words</figcaption>
        <div class="cmd"><pre><code>uv run refactory ceo blurb/writer --intent "a blurb about teapots"
uv run refactory steer --run &lt;id&gt; --note "keep the step small" --kind constraint
uv run refactory steer --run &lt;id&gt; --verdict gate_strategy=proceed</code></pre>
        <button class="copy" type="button" data-copy>copy</button></div>
      </figure>
      {ledger_kinds}
      <ul>
        <li><strong>The seed is written before the lowering</strong>, because the lowering is
        what carries it into the prompts; seeding afterwards would produce a run nobody steered.</li>
        <li><strong>A verdict is a decision the run routed on.</strong> Writing it down is what
        makes a resumed run the same run: re-asking would produce a second answer to a question
        already answered, and inventing one would produce a result nobody can account for.</li>
        <li><strong>Fidelity is checked, not assumed.</strong> A ledger that is missing, empty or
        holding the placeholder is reported, and given the run's agent outputs, a node that cited
        no entry — or a dangling one — is caught. Whether the work <em>satisfies</em> the intent
        still needs somebody to read it; that is what the gates are for.</li>
      </ul>
    </section>

    <section id="drift">
      <h2>Keeping it honest</h2>
      <p class="lede">Two runtimes, generated documents and a page of commands are three ways
      for a repository to start lying. Each one has a gate, and the gate is the answer to
      &quot;is this tested?&quot;.</p>
      {drift_gates}
      <div class="callout">
        <strong>The equivalence test is the interesting one.</strong> It runs the same mode on
        the walker and on DSH and compares the facts, not the score: the two runtimes may differ
        in <em>how</em> they run a graph — one process against many, sequential branches against
        concurrent — and must not differ in what the run produced. Where they do differ, it is
        written down in the spine's README and the Agent Notes rather than discovered later.
      </div>
    </section>

    <section id="commands">
      <h2>Every command, and what it should produce</h2>
      <p class="lede">The page and the checker read one list:
      <code>refactory_dsh/tools/recipes.py</code>. `uv run python -m
      refactory_dsh.tools.check_recipes` executes the offline ones and fails when any
      no longer does what this page says; <code>--tier model</code> runs the ones that
      make real model calls.</p>
      {recipes}
    </section>

    <section id="where">
      <h2>Where things live</h2>
      <table>
        <thead><tr><th>what</th><th>where</th></tr></thead>
        <tbody>
          <tr><td>the entry point</td><td><code>refactory/refactory_dsh/cli.py</code> — one verb per question, dispatching onto the module that owns it</td></tr>
          <tr><td>the install</td><td><code>pyproject.toml</code> + <code>uv.lock</code> (`uv sync`), <code>install.sh</code>, <code>refactory-dsh</code></td></tr>
          <tr><td>the modes</td><td><code>refactory/refactory_packages/</code> — <code>default_pack</code> is the authoring one</td></tr>
          <tr><td>the walker (debug runtime)</td><td><code>refactory/refactory_dsh/nn/</code> — graph, knobs, validation, executor</td></tr>
          <tr><td>the outer loop</td><td><code>refactory/refactory_dsh/loop.py</code>, <code>optim/</code>, <code>strategies.py</code></td></tr>
          <tr><td>the console (one JSON for everybody)</td><td><code>refactory/refactory_dsh/console.py</code></td></tr>
          <tr><td>the runtime that ships</td><td><code>packages/workflow/workflow-spine/</code> (the cursor) and <code>packages/bundle/graph/</code></td></tr>
          <tr><td>the CEO preset the GUI mounts</td><td><code>packages/bundle/refactory/</code> — the <code>ceo</code> composition, declared inline and set as the roster default</td></tr>
          <tr><td>the GUI's refactory surface</td><td><code>packages/bundle/refactory-web/</code> — the console, picker and brand rows over the pinned web app</td></tr>
          <tr><td>the conversion between them</td><td><code>refactory/refactory_dsh/lowering.py</code> → <code>Workflow.to_dict()</code></td></tr>
          <tr><td>the CEO's tool</td><td><code>packages/refactory/tool-refactory/</code></td></tr>
          <tr><td>the GUI panel</td><td><code>packages/client/ui-refactory/</code> + <code>packages/refactory/refactory-console/</code></td></tr>
          <tr><td>the tutorials, in prose</td><td><code>refactory/docs/tutorials/</code></td></tr>
          <tr><td>the documented commands, and their checker</td><td><code>refactory/refactory_dsh/tools/recipes.py</code> + <code>check_recipes.py</code></td></tr>
          <tr><td>the generators</td><td><code>refactory/refactory_dsh/tools/gen_tutorial.py</code>, <code>gen_api_reference.py</code></td></tr>
          <tr><td>datasets and Tasks</td><td><code>refactory/docs/datasets.md</code></td></tr>
        </tbody>
      </table>
      <p class="muted">This page is generated by <code>python -m refactory_dsh.tools.gen_tutorial</code> from the
      code it describes, and <code>refactory/tests/test_docs_tutorial.py</code> fails when it is stale.</p>
    </section>
  </main>
</div>
<script>
  for (const button of document.querySelectorAll('[data-copy]')) {{
    button.addEventListener('click', async () => {{
      const text = button.parentElement.querySelector('code').innerText
      try {{ await navigator.clipboard.writeText(text); button.textContent = 'copied' }}
      catch {{ button.textContent = 'select it' }}
      setTimeout(() => {{ button.textContent = 'copy' }}, 1200)
    }})
  }}
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    """Write the page, or check that the file on disk is current."""
    parser = argparse.ArgumentParser(description="Generate the HTML tutorial.")
    parser.add_argument("--check", action="store_true", help="fail when the file differs instead of writing it")
    args = parser.parse_args(argv)
    rendered = render()
    if args.check:
        if not OUT.exists() or OUT.read_text() != rendered:
            print(f"gen-tutorial: {OUT} is stale; run `python -m refactory_dsh.tools.gen_tutorial`")
            return 1
        print("gen-tutorial: up to date")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(rendered)
    print(f"gen-tutorial: wrote {OUT}")
    return 0


if __name__ == "__main__":  # pragma: no cover - the entry point
    raise SystemExit(main())
