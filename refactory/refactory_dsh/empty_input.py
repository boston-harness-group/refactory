"""Run a created mode once with its input empty, and refuse one that invents input.

A created mode that finds no input and generates stand-in data instead -- a synthetic
table, a sampled set, a simulated file -- exits 0 and reports on something nobody gave
it. The canonical smoke cannot see that: it runs the mode on real input. So after the
smoke passes, create mode runs the mode once more with its declared inputs empty, and
judges the run by what it did:

* it stopped with an error of its own (a non-zero exit) -- **pass**, the mode refused to
  run without input;
* it finished (exit 0) -- **finding**, whatever it wrote or reported: a file, an empty
  file, a "no input" status, or nothing but facts and a score. No rule can tell an
  honest status from a fabricated verdict ("plate passes QC"), and a fabricated result
  need not be written to a file at all, so the one rule is the one the API states: a
  mode with no input stops with an error naming what is missing;
* it was stopped, could not start, or could not be described (a timeout, a missing
  model) -- **inconclusive**, recorded and warned about, not held against the mode.

Only what the run declares as input is emptied, never an output or a setting:

* a knob the brief declares as input (``requires_input``, the ``--input`` names) that
  the mode has. With a sample (``--sample``) or a default that names an existing
  folder or file, it points at an empty folder or an empty file of the same name;
  with any other value it is the empty string;
* a data node that reads a ``source_path``, which reads an empty source of its format.

Every other knob keeps its value: a sample for an output folder, a configuration file
or a reference database is not input, and emptying it would make a fabricating mode
crash and pass. A path the graph declares as a write is never emptied. A data node
bound to a Task takes its items from the Task, so it has no empty equivalent.

A mode with no declared input at all is **not applicable**. A mode whose undeclared
knob defaults to an existing path that looks like input -- a file or folder under the
project or the person's home folder; never a system or scratch location, an executable,
the run folder or a path the graph writes -- is **inconclusive** and is not run: emptying the declared input would still leave it
reading real data, and a run on real data says nothing about empty input. Both are
warnings, never findings.

The run goes through `sandbox.smoke`, the same contained path the canonical smoke
uses. Describing the mode and running it share one smoke budget, so the check never
takes longer than one smoke.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

#: The receipt the smoke gate writes, beside ``smoke.json``.
RECEIPT = "empty-input.json"

#: Where the empty stand-ins for the inputs are made, inside the run's own directory.
EMPTY_ROOT = ".empty-input"

#: Empty files a data node of each format reads as no items.
_EMPTY_SOURCE = {"jsonl": "items.jsonl", "csv": "items.csv"}

#: The longest describing the mode may take, within the smoke budget.
DESCRIBE_TIMEOUT_S = 180

#: The statuses a person is warned about: the check did not judge the mode.
UNJUDGED = ("inconclusive", "not_applicable")

#: System locations: a default pointing under one names a tool, not input.
_SYSTEM_PREFIXES = ("/usr", "/bin", "/sbin", "/opt", "/etc", "/Library", "/System",
                    "/Applications")


class Finding(ValueError):
    """The mode reported on empty input: a repairable code finding, typed.

    The smoke gate sends it back to the builder as it is. Its reason quotes file names
    the mode chose, so it is never classified by its words.
    """

    def __init__(self, receipt: dict[str, Any]):
        self.receipt = receipt
        super().__init__(f"empty-input check: {receipt.get('reason')}")


def describe(identifier: str, project: Path, *, timeout_s: int = DESCRIBE_TIMEOUT_S,
             **kwargs: Any) -> dict[str, Any]:
    """The mode's knobs and their values, data nodes and declared writes.

    Read in the sandbox, where loading the mode is safe. Raises `ValueError` when the
    mode cannot be described in ``timeout_s``.
    """
    from refactory_dsh import sandbox

    code = (
        "import json\n"
        "from pathlib import Path\n"
        "from refactory_dsh import registry\n"
        f"package, mode_class = registry.resolve(Path({str(project)!r}), {identifier!r})\n"
        "mode = mode_class()\n"
        "graph = mode.graph()\n"
        "knobs = mode.state_dict().get('knobs') or {}\n"
        "print(json.dumps({\n"
        "    'knobs': sorted(knobs.keys()),\n"
        "    'values': {name: (entry or {}).get('value') for name, entry in knobs.items()},\n"
        "    'required': sorted(name for name, entry in knobs.items()\n"
        "                       if (entry or {}).get('spec', {}).get('required')),\n"
        "    'data_nodes': [{'node': n.id, 'task_ref': n.task_ref, 'source_path': n.source_path,\n"
        "                    'source_format': n.source_format}\n"
        "                   for n in graph.nodes.values() if n.kind == 'data'],\n"
        "    'writes': sorted({w for n in graph.nodes.values() for w in n.writes}),\n"
        "}, default=str))\n"
    )
    result = sandbox.run_sandboxed(code, workdir=Path(project), env=dict(os.environ),
                                   timeout_s=timeout_s, **kwargs)
    if not result.ok:
        raise ValueError(f"the mode could not be described for the empty-input check: "
                         f"{result.problem()}")
    return dict(result.payload)


def _as_path(value: Any, project: Path) -> Path | None:
    """The existing path a knob value names, or ``None`` when it names none."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    # A relative value is the project's, as the mode is run from it on both runtimes;
    # never the current directory, which differs between them.
    path = candidate if candidate.is_absolute() else Path(project) / candidate
    return path if path.exists() else None


def _temp_roots() -> tuple[Path, ...]:
    """Scratch locations: what a knob defaulting there points at is not input."""
    import tempfile

    roots = {Path("/tmp"), Path("/var/tmp"), Path("/dev/shm"), Path(tempfile.gettempdir())}
    return tuple(sorted({root.resolve() for root in roots if root.exists()}))


def _under(path: Path, roots: Any) -> bool:
    return any(path == root or path.is_relative_to(root) for root in roots)


def _looks_like_input(path: Path, *, project: Path, run_root: Path) -> bool:
    """Whether an existing path an undeclared knob defaults to could be real input.

    Only a file or folder under the project, or under the person's home folder, could
    be; never the project or home folder as a whole, anything under the run folder, a
    scratch location (`_temp_roots`), a system location (`_SYSTEM_PREFIXES`) or an
    executable file. So ``/``, ``/usr/bin/env`` and ``/tmp`` never stop the check.
    """
    resolved, root = Path(path).resolve(), Path(project).resolve()
    home = Path.home().resolve()
    if resolved in (root, home) or _under(resolved, (Path(run_root).resolve(),)):
        return False
    if resolved.is_file() and os.access(resolved, os.X_OK):
        return False
    if _under(resolved, (root,)):
        return True
    system = tuple(Path(prefix).resolve() for prefix in _SYSTEM_PREFIXES if Path(prefix).exists())
    return (_under(resolved, (home,)) and not _under(resolved, _temp_roots())
            and not _under(resolved, system))


def _normal(value: Any) -> str:
    return str(value).strip().rstrip("/").removeprefix("./")


def infer_inputs(
    described: dict[str, Any],
    *, project: Path, run_root: Path,
) -> list[str]:
    """Knob names that are inputs, inferred from the mode's structure as a last resort.

    Called only when neither ``--input`` nor the plan declared any inputs.  A
    ``required=True`` knob is inferred only when its value is empty or blank, or
    when its effective value is an existing input-like path (`_looks_like_input`).
    A required knob with a non-empty default that is not a path into the project is
    a *setting* (``model_name``, ``api_endpoint``), never an input.  A knob whose
    value only looks like a path is left for ``plan()``'s ``undeclared`` list, which
    is inconclusive and never emptied.

    Data-node sources are handled separately by `plan`.

    Returns the inferred names in sorted order, or an empty list when nothing qualifies.
    """
    writes = {_normal(w) for w in described.get("writes", ()) if _normal(w)}
    values = dict(described.get("values") or {})
    required = set(described.get("required") or ())
    inputs: list[str] = []
    for name in sorted(described.get("knobs") or ()):
        value = values.get(name)
        if name in writes or (isinstance(value, str) and _normal(value) in writes):
            continue
        if name in required:
            default_blank = not str(value or "").strip()
            if default_blank:
                inputs.append(name)
                continue
            real = _as_path(value, project)
            if real is not None and _looks_like_input(
                    real, project=project, run_root=run_root):
                inputs.append(name)
    return inputs


def _infer_reasons(
    names: list[str], described: dict[str, Any],
    *, project: Path, run_root: Path,
) -> list[str]:
    """Why each inferred name qualified, for the run warning."""
    values = dict(described.get("values") or {})
    reasons: list[str] = []
    for name in names:
        value = values.get(name)
        if not str(value or "").strip():
            reasons.append(f"{name} (required, default empty)")
        else:
            reasons.append(f"{name} (required, default is an input-like path)")
    return reasons


def plan(
    described: dict[str, Any], samples: dict[str, str], declared: tuple[str, ...] | list[str],
    *, empty_root: Path, project: Path, run_root: Path | None = None,
) -> dict[str, Any]:
    """What the empty-input run points where, and why, in words.

    Makes the empty stand-ins under ``empty_root``. ``knobs`` and ``sources`` are what
    the run overrides; ``inputs`` says what was emptied, ``kept`` what was declared but
    left alone and why, and ``undeclared`` the knobs whose default reads an existing
    path that looks like input (`_looks_like_input`) the run does not declare as input.
    ``run_root`` is the run's folder (default: the folder holding ``empty_root``).
    """
    run_root = Path(run_root) if run_root is not None else Path(empty_root).parent
    knob_names = set(described.get("knobs") or ())
    values = dict(described.get("values") or {})
    writes = {_normal(write) for write in described.get("writes") or () if _normal(write)}
    knobs: dict[str, str] = {}
    sources: dict[str, dict[str, str]] = {}
    inputs: list[str] = []
    kept: list[str] = []

    def empty_dir(label: str) -> Path:
        path = empty_root / label
        path.mkdir(parents=True, exist_ok=True)
        return path

    input_knobs: list[str] = []
    for name in dict.fromkeys(declared):
        if name not in knob_names:
            kept.append(f"declared input {name} is not a knob of the mode")
            continue
        value = samples[name] if name in samples else values.get(name)
        if name in writes or (isinstance(value, str) and _normal(value) in writes):
            kept.append(f"declared input {name} names a file the graph writes; left as given")
            continue
        input_knobs.append(name)
        real = _as_path(value, project)
        if real is not None and real.is_dir():
            knobs[name] = str(empty_dir(f"knob-{name}").resolve())
            inputs.append(f"input knob {name} ({value}) -> empty folder")
        elif real is not None:
            stand_in = empty_dir(f"knob-{name}") / real.name
            stand_in.write_text("")
            knobs[name] = str(stand_in.resolve())
            inputs.append(f"input knob {name} ({value}) -> empty file")
        else:
            knobs[name] = ""
            inputs.append(f"input knob {name} ({value!r}) -> empty string")
    for node in described.get("data_nodes") or ():
        if node.get("task_ref") or not node.get("source_path"):
            continue
        node_id = str(node["node"])
        if _normal(node["source_path"]) in writes:
            kept.append(f"data node {node_id} reads a file the graph writes; left as given")
            continue
        source_format = str(node.get("source_format") or "directory")
        folder = empty_dir(f"data-{node_id}")
        if source_format in _EMPTY_SOURCE:
            source = folder / _EMPTY_SOURCE[source_format]
            source.write_text("")
        else:
            source = folder
        sources[node_id] = {"source_path": str(source.resolve()), "source_format": source_format}
        inputs.append(f"data node {node_id} ({source_format} source) -> empty")
    undeclared: list[str] = []
    for name in sorted(knob_names - set(input_knobs) - set(samples)):
        value = values.get(name)
        real = _as_path(value, project)
        if (real is None or _normal(value) in writes
                or not _looks_like_input(real, project=project, run_root=run_root)):
            continue
        undeclared.append(f"{name} (default {value})")
    return {"knobs": knobs, "sources": sources, "inputs": inputs, "kept": kept,
            "undeclared": undeclared}


def produced_results(workdir: Path, writes: list[str] | tuple[str, ...]) -> list[str]:
    """The files the mode's graph declares that the run left in its directory.

    Recorded in the receipt so a person sees what the run wrote; the verdict does not
    depend on it.
    """
    from refactory_dsh.nn.executor import ITEMS_OVERRIDE

    found: list[str] = []
    for declared in writes:
        relative = str(declared).strip().rstrip("/")
        if not relative or relative.startswith(EMPTY_ROOT) or relative == ITEMS_OVERRIDE:
            continue
        if (Path(workdir) / relative).exists():
            found.append(relative)
    return sorted(found)


def verdict(*, ok: bool, timed_out: bool, error: str,
            produced: list[str] | tuple[str, ...] = ()) -> tuple[str, str]:
    """``(status, reason)`` for one empty-input run; see the module docstring.

    ``produced`` only names, in the reason, the declared files the run wrote.
    """
    from refactory_dsh.authoring_evidence import failure_kind

    if timed_out:
        return "inconclusive", "the empty-input run was stopped before it ended"
    if not ok:
        kind = failure_kind(error)
        if kind != "code_failure":
            return "inconclusive", f"the empty-input run could not run ({kind}): {error}"
        return "pass", f"the mode stopped on empty input: {error or '(no message)'}"
    wrote = f" and wrote {', '.join(produced)}" if produced else ""
    return "finding", (
        f"with its input empty the mode exited 0{wrote}. A created mode must not "
        "fabricate, simulate or sample stand-in input data when its real input is "
        "missing or empty, nor report a result or a status in its place: it must stop "
        "with an error that names the missing input (a non-zero exit)"
    )


def run(
    identifier: str, workdir: Path, *, project: Path, samples: dict[str, str],
    declared: tuple[str, ...] | list[str], model: bool, timeout_s: int,
    model_context: dict[str, str] | None = None, run_root: Path | None = None,
    smoke_items: int = 0, split: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the mode once with its inputs empty and return the receipt.

    ``run_root`` is the run's folder, a scratch location for `plan` (default: the
    empty-input run's own ``workdir``).

    ``smoke_items`` and ``split`` (``holdout``, ``parts_manifest``) are the canonical
    smoke's sample, passed on as they are: a Task-bound node keeps its items, and they
    are the same train items the smoke ran, never a val or test item.

    ``timeout_s`` is the smoke's inactivity limit. Describing the mode and the run share
    one smoke's wall clock (`sandbox.smoke_wall_s`): the describe gets at most
    `DESCRIBE_TIMEOUT_S` of the smoke's backstop and the run's backstop is what is left,
    so the check fits one run of the smoke gate's envelope.
    """
    from refactory_dsh import sandbox
    from refactory_dsh.items import write_source

    from refactory_dsh import target_guard

    workdir = Path(workdir)
    receipt: dict[str, Any] = {"mode": identifier, "workdir": str(workdir.resolve()),
                               "declared": list(declared), "timeout_s": int(timeout_s)}
    backstop = sandbox.smoke_backstop_s(int(timeout_s))
    started = time.monotonic()
    try:
        described = describe(identifier, project,
                             timeout_s=max(1, min(DESCRIBE_TIMEOUT_S, int(backstop))))
    except Exception as exc:  # noqa: BLE001 - any failure to describe leaves it unjudged
        return {**receipt, "status": "inconclusive",
                "reason": f"the mode could not be described, so the empty-input run did "
                          f"not start: {type(exc).__name__}: {exc}"}
    if not declared:
        inferred_list = infer_inputs(described, project=project,
                                     run_root=run_root or workdir)
        if inferred_list:
            declared = tuple(inferred_list)
            receipt["declared"] = list(declared)
            reasons = _infer_reasons(inferred_list, described, project=project,
                                     run_root=run_root or workdir)
            target_guard.warn(
                "the empty-input check inferred these inputs from the mode's structure "
                f"(neither --input nor the plan declared any): {', '.join(reasons)}"
            )
    planned = plan(described, samples, declared, empty_root=workdir / EMPTY_ROOT,
                   project=project, run_root=run_root or workdir)
    receipt.update({key: planned[key] for key in ("inputs", "kept", "undeclared")})
    if not planned["knobs"] and not planned["sources"]:
        return {**receipt, "status": "not_applicable",
                "reason": "the mode declares no input the check can empty: no declared "
                          "input (--input or required knob) is a knob of the mode, and "
                          "no data node reads a source_path"}
    if planned["undeclared"]:
        return {**receipt, "status": "inconclusive",
                "reason": "not run: these knobs are not declared inputs but default to an "
                          f"existing path, so an empty-input run would still read it: "
                          f"{', '.join(planned['undeclared'])}. Declare each input with "
                          "--input, or give the knob a --sample"}
    remaining = int(backstop) - math.ceil(time.monotonic() - started)
    if remaining < 1:
        return {**receipt, "status": "inconclusive",
                "reason": "describing the mode used the whole smoke budget, so the "
                          "empty-input run did not start"}
    for node_id, source in planned["sources"].items():
        write_source(workdir, node_id=node_id, **source)
    result = sandbox.smoke(
        identifier, workdir, project=project, model=model,
        knobs={**samples, **planned["knobs"]}, model_context=model_context,
        timeout_s=int(timeout_s), backstop_s=remaining, smoke_items=int(smoke_items),
        **(split or {}),
    )
    ok = result.ok and result.payload.get("exit_ok") is not False
    error = result.problem() if not result.ok else str(result.payload.get("error") or "")
    produced = produced_results(workdir, described.get("writes") or ())
    status, reason = verdict(ok=ok, timed_out=result.timed_out, error=error,
                             produced=produced)
    return {**receipt, "status": status, "reason": reason, "exit_ok": ok,
            "timed_out": result.timed_out, "returncode": result.returncode,
            "elapsed_s": result.elapsed_s, "run_backstop_s": remaining, "error": error,
            "sample": result.payload.get("sample"),
            "produced": produced,
            "knobs": planned["knobs"], "sources": planned["sources"]}


def warning(receipt: dict[str, Any] | None) -> str:
    """The line a person reads when the check did not judge the mode; empty otherwise."""
    if not receipt or receipt.get("status") not in UNJUDGED:
        return ""
    status = str(receipt["status"]).replace("_", " ")
    return f"the empty-input check was {status} for {receipt.get('mode')}: {receipt.get('reason')}"
