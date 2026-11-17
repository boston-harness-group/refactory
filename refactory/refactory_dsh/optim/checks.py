"""Checks a candidate gets before it is scored, beyond the validator's rules.

Both are run by `Optimizer.attempt`, the path every strategy's proposals take, after the
validator and lowering have accepted the state:

- `answered_gate_refusals` (``E-gate-kind``): a gate someone answers (a person, or the
  agent driving the run) that the edit adds, or turns an existing gate into, when nobody
  answers gates while the run's candidates run. Such a gate passes every rule and then
  fails every item that reaches it, on both runtimes.
- `code_dry_run` (``E-code-run``): each code node the edit adds or changes is run in the
  sandbox on the parent's recorded inputs for a few items (the files and facts its run
  left: ``runs/<parent>/passes/<data node>/<item>/``), before any item is scored with it.
  Code that raises on them, or returns what its node cannot use, is refused with the
  error and the line. When there is nothing to run it on -- no recorded inputs, inputs the
  candidate's new steps make, facts a step in a data pass reads (item directories do not
  reliably keep them), no sandbox -- it is skipped, and the candidate carries a note saying so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

#: How many of the parent's recorded items a new or changed code node is tried on.
DRY_RUN_ITEMS = 3


def _graph(state: Mapping[str, Any]) -> Any:
    from refactory_dsh.nn.graph import Graph

    try:
        return Graph.from_state(dict(state.get("graph") or {}))
    except Exception:  # noqa: BLE001 - an unreadable graph is the validator's to report
        return None


def answered_gate_refusals(parent_state: Mapping[str, Any], state: Mapping[str, Any]) -> list[str]:
    """``E-gate-kind`` for each gate of ``state`` answered by someone (`ANSWERED_GATES`)
    that ``parent_state`` does not have answered the same way. The parent's own are not
    the edit's doing."""
    from refactory_dsh.nn.node_schema import ANSWERED_GATES

    parent, graph = _graph(parent_state), _graph(state)
    if graph is None:
        return []
    before = {node_id: node.gate_kind for node_id, node in (parent.nodes.items() if parent else ())
              if node.kind == "gate"}
    return [f"E-gate-kind: gate {node_id!r} is answered by {node.gate_kind!r}, and nobody "
            "answers a gate while this run's candidates run (they run headless), so every item "
            "that reaches it would fail; make it a `fn` gate: a verdict gate "
            "(\"evaluator\": \"refactory_dsh.snippet:verdict\") reading what a reviewing step "
            "wrote, or a code gate"
            for node_id, node in sorted(graph.nodes.items())
            if node.kind == "gate" and node.gate_kind in ANSWERED_GATES
            and before.get(node_id) != node.gate_kind]


def _recorded_dirs(recorded: Mapping[str, Any], graph: Any, node_id: str
                   ) -> tuple[list[Path], bool]:
    """Where the parent's run left the inputs ``node_id`` would see, and whether they are a
    data pass's item directories (the node is in a pass) rather than the run directory."""
    for data_id, data in graph.nodes.items():
        if data.kind == "data" and node_id in graph.data_body(data_id):
            return [Path(p) for p in (recorded.get("passes") or {}).get(data_id) or ()], True
    run = recorded.get("run")
    return ([Path(run)] if run else []), False


def _inputs(node: Any, folder: Path) -> tuple[dict[str, Any] | None, str]:
    """The ``inp`` the node's code would get in ``folder``, or ``None`` and why not: a
    declared read or input the recorded run does not have (made by a step the candidate
    adds, or a file that run never wrote). Facts come from the run directory's
    ``facts.json``; an item directory's is not the item's (`code_dry_run`)."""
    files: dict[str, str] = {}
    missing = [name for name in node.reads if not (folder / name).is_file()]
    try:
        facts_all = json.loads((folder / "facts.json").read_text())
    except (OSError, ValueError):
        facts_all = {}
    facts_all = facts_all if isinstance(facts_all, dict) else {}
    missing += [name for name in node.inputs if name not in facts_all]
    if missing:
        return None, f"the recorded item has no {', '.join(missing)}"
    for name in node.reads:
        try:
            files[name] = (folder / name).read_text()
        except (UnicodeDecodeError, IsADirectoryError):
            continue
    return {"files": files, "facts": {name: facts_all[name] for name in node.inputs}}, ""


def code_dry_run(parent_id: str, parent_state: Mapping[str, Any], state: Mapping[str, Any],
                 nodes: Sequence[str], recorded: Callable[[str], Mapping[str, Any]] | None,
                 sandbox: tuple[bool, str] | None = None) -> tuple[list[str], list[str]]:
    """Run each code node in ``nodes`` (new or changed against the parent) on up to
    `DRY_RUN_ITEMS` of the parent's recorded items: ``(refusals, notes)``. ``recorded``
    gives a candidate's recorded inputs (``{"run": dir, "passes": {data node: [item
    dirs]}}``); ``None`` (a caller with no run directory) skips every node, with a note."""
    from refactory_dsh import snippet

    graph = _graph(state)
    if graph is None or not nodes:
        return [], []
    if recorded is None:
        return [], [f"code dry run skipped for {', '.join(nodes)}: no recorded run inputs here"]
    try:
        found = dict(recorded(parent_id) or {})
    except Exception as exc:  # noqa: BLE001 - a run directory that cannot be read is a skip
        found = {}
        why = f"{type(exc).__name__}: {exc}"
    else:
        why = ""
    ok, reason = sandbox if sandbox is not None else snippet.available()
    if not ok:
        return [], [f"code dry run skipped for {', '.join(nodes)}: no sandbox here ({reason})"]
    values = {name: (entry or {}).get("value") for name, entry in (state.get("knobs") or {}).items()}
    refusals: list[str] = []
    notes: list[str] = []
    for node_id in nodes:
        node = graph.nodes.get(node_id)
        if node is None:
            continue
        folders, in_pass = _recorded_dirs(found, graph, node_id)
        if in_pass and node.inputs:
            # An item directory keeps the item's files, not reliably its facts: on the
            # walker its facts.json is a copy of the run's taken when the pass began
            # (`items.prepare_pass_dir`), so the first item has none and the next the one
            # before's.
            notes.append(f"code dry run of {node_id!r} skipped: it reads the fact(s) "
                         f"{', '.join(node.inputs)}, and the parent's item directories do not "
                         "reliably keep each item's facts (a walker run keeps none)")
            continue
        dirs = [d for d in folders if d.is_dir()][:DRY_RUN_ITEMS]
        if not dirs:
            notes.append(f"code dry run of {node_id!r} skipped: the parent {parent_id} has no "
                         "recorded inputs" + (f" ({why})" if why else "")
                         + " for where it runs")
            continue
        spec = snippet.spec(node, [e.condition for e in graph.out_edges(node_id)], values)
        ran, skipped = 0, []
        for folder in dirs:
            inp, missing = _inputs(node, folder)
            if inp is None:
                skipped.append(f"{folder.name}: {missing}")
                continue
            try:
                result = snippet.execute(spec["code"], inp, hide=(str(folder),))
                if node.kind == "gate":
                    snippet.checked_verdict(node_id, result, spec)
                else:
                    snippet.checked_result(node_id, result, node.writes, node.produces)
            except snippet.SandboxUnavailable as exc:
                if isinstance(exc, snippet.NoSandbox):
                    # No sandbox at all refuses, as any failure does; a busy machine
                    # that did not start one in time only skips the item.
                    refusals.append(
                        f"E-code-run: code node {node_id!r} failed when run on the parent's "
                        f"recorded inputs for item {folder.name!r}: {str(exc)[:1500]}. Its "
                        f"inputs there: files {sorted(inp['files'])}, facts "
                        f"{sorted(inp['facts'])}; read each one's shape before using it")
                    break
                skipped.append(f"{folder.name}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 - any failure is what the dry run is for
                refusals.append(
                    f"E-code-run: code node {node_id!r} failed when run on the parent's recorded "
                    f"inputs for item {folder.name!r}: {str(exc)[:1500]}. Its inputs there: files "
                    f"{sorted(inp['files'])}, facts {sorted(inp['facts'])}; read each one's "
                    "shape before using it")
                break
            ran += 1
        if refusals:
            break
        if not ran:
            notes.append(f"code dry run of {node_id!r} skipped: " + "; ".join(skipped))
    return refusals, notes


def recorded_inputs(run_dir: Path) -> dict[str, Any]:
    """A candidate's recorded inputs in its run directory (``runs/<id>``): the directory
    itself, and each data node's finished item directories (``passes/<node>/<item>``, a
    ``.done`` marker), sorted. Empty when the run kept none."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return {}
    passes: dict[str, list[str]] = {}
    root = run_dir / "passes"
    if root.is_dir():
        for node_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            done = sorted(str(item) for item in node_dir.iterdir()
                          if item.is_dir() and (item / ".done").exists())
            if done:
                passes[node_dir.name] = done
    return {"run": str(run_dir), "passes": passes}
