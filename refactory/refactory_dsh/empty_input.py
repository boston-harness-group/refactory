"""Run a created mode once with its input empty, and refuse one that invents input.

A created mode that finds no input and generates stand-in data instead -- a synthetic
table, a sampled set, a simulated file -- exits 0 and reports on something nobody gave
it. The canonical smoke cannot see that: it runs the mode on real input. So after the
smoke passes, create mode runs the mode once more with every input it can point
somewhere empty, and judges the run by what it did:

* it stopped with an error of its own -- **pass**, the mode refused to run without input;
* it finished and wrote none of the files its graph declares -- **pass**, nothing was
  reported;
* it finished and wrote results -- **finding**: the results came from no input;
* it was stopped, or could not start (a timeout, a missing model) -- **inconclusive**,
  recorded and not held against the mode.

An input is something a run can point elsewhere: a sample knob whose value is a path
(the person's real input for the smoke), a declared input (``requires_input``) the
mode takes as a knob, and a data node that reads a ``source_path``. A data node bound
to a Task takes its items from the Task, so it has no empty equivalent and is left as
it is; a mode with no input of any of these kinds is not applicable and is recorded so.

The run goes through `sandbox.smoke`, the same contained path and the same time limit
the canonical smoke uses, so it costs one more short run of the mode and no new way
of running one.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

#: The receipt the smoke gate writes, beside ``smoke.json``.
RECEIPT = "empty-input.json"

#: Where the empty stand-ins for the inputs are made, inside the run's own directory.
EMPTY_ROOT = ".empty-input"

#: Empty files a data node of each format reads as no items.
_EMPTY_SOURCE = {"jsonl": "items.jsonl", "csv": "items.csv"}


def describe(identifier: str, project: Path, **kwargs: Any) -> dict[str, Any]:
    """The mode's knobs, data nodes and declared writes, read where loading it is safe."""
    from refactory_dsh import sandbox

    code = (
        "import json\n"
        "from pathlib import Path\n"
        "from refactory_dsh import registry\n"
        f"package, mode_class = registry.resolve(Path({str(project)!r}), {identifier!r})\n"
        "mode = mode_class()\n"
        "graph = mode.graph()\n"
        "print(json.dumps({\n"
        "    'knobs': sorted((mode.state_dict().get('knobs') or {}).keys()),\n"
        "    'data_nodes': [{'node': n.id, 'task_ref': n.task_ref, 'source_path': n.source_path,\n"
        "                    'source_format': n.source_format}\n"
        "                   for n in graph.nodes.values() if n.kind == 'data'],\n"
        "    'writes': sorted({w for n in graph.nodes.values() for w in n.writes}),\n"
        "}))\n"
    )
    result = sandbox.run_sandboxed(code, workdir=Path(project), env=dict(os.environ), **kwargs)
    if not result.ok:
        raise ValueError(f"the mode could not be described for the empty-input check: "
                         f"{result.problem()}")
    return dict(result.payload)


def _as_path(value: str, project: Path) -> Path | None:
    """The existing path a knob value names, or ``None`` when it names none."""
    text = str(value).strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    for path in ((candidate,) if candidate.is_absolute() else (project / candidate, candidate)):
        if path.exists():
            return path
    return None


def plan(
    described: dict[str, Any], samples: dict[str, str], declared: tuple[str, ...] | list[str],
    *, empty_root: Path, project: Path,
) -> dict[str, Any]:
    """What the empty-input run points where, and the inputs it empties, in words.

    Makes the empty stand-ins under ``empty_root``: an empty directory for a directory
    input, an empty file of the same name for a file input, and an empty source of the
    node's own format for a data node.
    """
    knob_names = set(described.get("knobs") or ())
    knobs: dict[str, str] = {}
    sources: dict[str, dict[str, str]] = {}
    inputs: list[str] = []

    def empty_dir(label: str) -> Path:
        path = empty_root / label
        path.mkdir(parents=True, exist_ok=True)
        return path

    for name, value in samples.items():
        real = _as_path(value, project)
        if real is None:
            continue
        if real.is_dir():
            knobs[name] = str(empty_dir(f"knob-{name}").resolve())
        else:
            stand_in = empty_dir(f"knob-{name}") / real.name
            stand_in.write_text("")
            knobs[name] = str(stand_in.resolve())
        inputs.append(f"knob {name} (sample {value}) -> empty")
    for name in declared:
        if name in knob_names and name not in samples:
            knobs[name] = str(empty_dir(f"knob-{name}").resolve())
            inputs.append(f"declared input knob {name} -> empty directory")
    for node in described.get("data_nodes") or ():
        if node.get("task_ref") or not node.get("source_path"):
            continue
        node_id = str(node["node"])
        source_format = str(node.get("source_format") or "directory")
        folder = empty_dir(f"data-{node_id}")
        if source_format in _EMPTY_SOURCE:
            source = folder / _EMPTY_SOURCE[source_format]
            source.write_text("")
        else:
            source = folder
        sources[node_id] = {"source_path": str(source.resolve()), "source_format": source_format}
        inputs.append(f"data node {node_id} ({source_format} source) -> empty")
    return {"knobs": knobs, "sources": sources, "inputs": inputs}


def produced_results(workdir: Path, writes: list[str] | tuple[str, ...]) -> list[str]:
    """The files the mode's graph declares that the run left in its directory."""
    from refactory_dsh.nn.executor import ITEMS_OVERRIDE

    found: list[str] = []
    for declared in writes:
        relative = str(declared).strip().rstrip("/")
        if not relative or relative.startswith(EMPTY_ROOT) or relative == ITEMS_OVERRIDE:
            continue
        if (Path(workdir) / relative).exists():
            found.append(relative)
    return sorted(found)


def verdict(*, ok: bool, timed_out: bool, error: str, produced: list[str]) -> tuple[str, str]:
    """``(status, reason)`` for one empty-input run; see the module docstring."""
    from refactory_dsh.authoring_evidence import failure_kind

    if timed_out:
        return "inconclusive", "the empty-input run was stopped before it ended"
    if not ok:
        kind = failure_kind(error)
        if kind != "code_failure":
            return "inconclusive", f"the empty-input run could not run ({kind}): {error}"
        return "pass", f"the mode stopped on empty input: {error or '(no message)'}"
    if produced:
        return "finding", (
            "with its input empty the mode finished and wrote results "
            f"({', '.join(produced)}). A created mode must not fabricate, simulate or "
            "sample stand-in input data when its real input is missing or empty; it must "
            "stop with an error that names the missing input"
        )
    return "pass", "the mode finished on empty input and wrote none of its declared files"


def run(
    identifier: str, workdir: Path, *, project: Path, samples: dict[str, str],
    declared: tuple[str, ...] | list[str], model: bool, timeout_s: int,
    model_context: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run the mode once with its inputs empty and return the receipt."""
    from refactory_dsh import sandbox
    from refactory_dsh.items import write_source

    workdir = Path(workdir)
    described = describe(identifier, project)
    planned = plan(described, samples, declared, empty_root=workdir / EMPTY_ROOT,
                   project=project)
    receipt: dict[str, Any] = {"mode": identifier, "workdir": str(workdir.resolve()),
                               "inputs": planned["inputs"]}
    if not planned["knobs"] and not planned["sources"]:
        return {**receipt, "status": "not_applicable",
                "reason": "the mode has no input a run can point somewhere empty: no sample "
                          "knob names a path, no declared input is a knob, and no data node "
                          "reads a source_path"}
    for node_id, source in planned["sources"].items():
        write_source(workdir, node_id=node_id, **source)
    result = sandbox.smoke(
        identifier, workdir, project=project, model=model,
        knobs={**samples, **planned["knobs"]}, model_context=model_context,
        timeout_s=timeout_s,
    )
    ok = result.ok and result.payload.get("exit_ok") is not False
    error = result.problem() if not result.ok else str(result.payload.get("error") or "")
    produced = produced_results(workdir, described.get("writes") or ())
    status, reason = verdict(ok=ok, timed_out=result.timed_out, error=error, produced=produced)
    return {**receipt, "status": status, "reason": reason, "exit_ok": ok,
            "timed_out": result.timed_out, "returncode": result.returncode,
            "elapsed_s": result.elapsed_s, "error": error, "produced": produced,
            "knobs": planned["knobs"], "sources": planned["sources"]}
