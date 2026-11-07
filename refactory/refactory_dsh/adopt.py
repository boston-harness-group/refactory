"""Adopt a run's winner as a mode of its own.

D15 in the decision record:

    **The loop's output is a new artifact, never a mutation of its input.**
    `create`, `edit`, `design` and **an adopting loop** all produce something new,
    so the baseline stays measurable.

`create`, `edit` and `design` exist. The adopting loop did not, and the consequence
was that a run's winner was unreachable: `state.json`'s population records what the
best candidate *scored*, `console candidate` returns a **diff** rather than a state,
and there is no command that turns either into something runnable. The full state is
in `journal.jsonl` — re-running it reproduces the score exactly — but reading a
journal by hand is not an interface.

**What adopting produces.** A package whose mode is the winner's state, pinned. The
mode subclasses the one that was searched, so its ops, its knobs and its `score()`
are the same code, and only the state differs. That is what makes the comparison
meaningful: the baseline and the adoption are the same mode at two points in the
search, which is exactly what D15's "so the baseline stays measurable" asks for.

**What it never does.** It does not touch the source package. Ordinary adoption
reads the run; a legacy pending finished-run selection is first recovered into
its journal so adoption cannot silently ignore that choice. `tests/test_adoption.py` asserts the source is
byte-identical afterwards, because "produces something new" is the whole decision and
a test that only checks a file appeared would not notice it being violated.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from refactory_dsh import registry

#: Where the pinned state and the provenance of an adoption live, beside the mode.
ADOPTED = ".adopted.json"

#: A package's source file. Small on purpose: everything else is inherited.
MODE_FILE = "mode.py"


class AdoptionError(RuntimeError):
    """A winner cannot be adopted. Always names what is wrong."""


def _check_name(as_name: str) -> None:
    """Refuse a destination name that is not the name of a package.

    `adopt` writes one directory under the workspace root, so the name is one path
    component and nothing else -- and it is a *package* name, which is the thing the
    written `__init__.py` will declare. Both rules are the package's own, so a name
    this accepts cannot produce a package the registry refuses to import.
    """
    from refactory_dsh.nn import Package, PackageError

    try:
        Package(name=as_name)
    except PackageError as error:
        raise AdoptionError(f"{as_name!r} is not a package name: {error}") from error


def _journal_entries(run_root: Path) -> list[dict[str, Any]]:
    from refactory_dsh.history import JOURNAL

    path = Path(run_root) / JOURNAL
    if not path.exists():
        raise AdoptionError(f"no {JOURNAL} in {run_root}; nothing recorded a state")
    entries: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            entries.append(loaded)
    return entries


def _score_by_id(entries: list[dict[str, Any]]) -> dict[str, float]:
    scored: dict[str, float] = {}
    for entry in entries:
        if entry.get("event") == "score" and entry.get("id") is not None:
            candidate_id = str(entry["id"])
            if entry.get("exit_ok") is not True:
                scored.pop(candidate_id, None)
                continue
            try:
                value = entry.get("score")
                if value is None or isinstance(value, bool) or not math.isfinite(float(value)):
                    scored.pop(candidate_id, None)
                    continue
                scored[candidate_id] = float(value)
            except (TypeError, ValueError):
                continue
    return scored


def _rejected_ids(entries: list[dict[str, Any]]) -> set[str]:
    """The candidates a person dismissed while the run was live.

    Read from the same journal as the scores, because this function and
    ``Loop.report`` have to name the *same* winner: a run whose best candidate was
    rejected from the panel still records its score, and adopting it would write the
    one answer the person had already read and refused. The last recorded list wins,
    since the run writes its whole keep/reject state with every control act.
    """
    rejected: set[str] = set()
    for entry in entries:
        if entry.get("event") != "control":
            continue
        if "rejected" in entry:
            rejected = {str(value) for value in entry.get("rejected") or ()}
            continue
        if entry.get("action") != "reject":
            continue
        for candidate_id in entry.get("targets") or ():
            rejected.add(str(candidate_id))
        if entry.get("target"):
            rejected.add(str(entry["target"]))
    return rejected


def best(run_root: Path) -> dict[str, Any]:
    """The winner: the highest-scoring candidate whose state the run recorded.

    Read from the journal, because that is the only artifact that carries a
    candidate's *state*. `report.json` carries the score, and `state.json`'s
    population carries the edits, and neither can be run.

    The order is the run report's own -- score descending, then id ascending
    (`Loop.leaderboard`) -- so this and `report.json` name the same candidate.
    Insertion order would pick the seed on a tie and adopt the baseline as the
    result while the report named a structural winner. Candidates the run recorded
    as rejected are skipped, for the same reason: the report skips them too.
    """
    run_root = Path(run_root)
    entries = _journal_entries(run_root)
    gated = _gate_winner(run_root, entries)
    if gated is not None:
        return gated
    scores = _score_by_id(entries)
    rejected = _rejected_ids(entries)

    scored: list[tuple[float, str, dict[str, Any], list[Any]]] = []
    for entry in entries:
        if entry.get("event") != "candidate":
            continue
        state = entry.get("state")
        if not isinstance(state, dict):
            continue
        candidate_id = str(entry.get("id") or "")
        if candidate_id in rejected:
            continue
        score = scores.get(candidate_id)
        if score is None:
            continue
        scored.append((score, candidate_id, state, list(entry.get("edits") or ())))

    if not scored:
        raise AdoptionError(
            f"{run_root} recorded no candidate that ran successfully, scored and kept its state; "
            "there is nothing to adopt"
        )
    winner_score, winner_id, winner_state, winner_edits = min(
        scored, key=lambda candidate: (-candidate[0], candidate[1])
    )
    return {
        "run": str(run_root),
        "candidate": winner_id,
        "score": winner_score,
        **_score_basis(entries, winner_id, winner_score),
        "edits": winner_edits,
        "state": winner_state,
    }


def _gate_winner(run_root: Path, entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The winner of a default-loop run with ``val_gate``: the validation gate's best.

    The report names the best on val as the winner (``Loop.report``), so adopting
    takes the same candidate, whatever its train score (it has none when its train
    run failed). ``None`` for a run without the gate.
    """
    try:
        config = json.loads((run_root / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        config = next((entry.get("config") or {} for entry in entries
                       if entry.get("event") == "start"), {})
    if not config.get("val_gate"):  # refused with a custom strategy, which keeps its own gate
        return None
    gate: dict[str, Any] = {}
    for entry in entries:
        if entry.get("event") == "strategy_state":
            gate = (entry.get("state") or {}).get("val_gate") or {}
    winner_id = gate.get("best_candidate_id")
    found = next((entry for entry in reversed(entries) if entry.get("event") == "candidate"
                  and str(entry.get("id")) == str(winner_id)
                  and isinstance(entry.get("state"), dict)), None)
    if not winner_id or found is None:
        raise AdoptionError(
            f"{run_root} ran with the validation gate, which recorded no best candidate "
            "with a state; there is nothing to adopt"
        )
    score = float(gate.get("best_score") or 0.0)
    train = _score_by_id(entries).get(str(winner_id))
    return {
        "run": str(run_root),
        "candidate": str(winner_id),
        "score": score,
        "score_basis": "val",
        "score_note": (f"{_number(score)} on val, the validation gate's best; train "
                       + (_number(train) if train is not None else "none (its train run failed)")),
        "edits": list(found.get("edits") or ()),
        "state": found["state"],
    }


def _number(value: float) -> str:
    return format(round(float(value), 6), "g")


def _score_basis(entries: list[dict[str, Any]], candidate_id: str, score: float) -> dict[str, Any]:
    """What a winner's score is: the search's, with its cost penalty, beside the raw one.

    `package smoke` reports the mode's own (raw) score, so without this the adopted
    package's two numbers disagree with no reason given (0.875 here, 0.955 there).
    """
    detail = next((entry for entry in reversed(entries) if entry.get("event") == "score"
                   and str(entry.get("id")) == candidate_id), {})
    raw = detail.get("raw_score")
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        # A journal from before the cost penalty: its score is the evaluator's own.
        return {"score_note": f"{_number(score)} as the run recorded it; no raw score to compare"}
    if round(float(raw) - float(score), 9) == 0:
        return {"raw_score": raw, "score_note": f"{_number(score)} raw; the search took no cost penalty"}
    penalty = ""
    if isinstance(detail.get("parsimony"), (int, float)) and isinstance(detail.get("nodes"), int):
        penalty = f" ({_number(detail['parsimony'])} x {detail['nodes']} nodes)"
    return {"raw_score": raw,
            "score_note": f"{_number(score)} with the cost penalty{penalty}; raw {_number(raw)}"}


def _mode_source(
    package: str, mode_name: str, state: dict[str, Any], project: Path
) -> str:
    """The adopted package's mode: the searched class, pinned to the winner's state.

    Inherited rather than regenerated. A regenerated mode would be a *new*
    implementation that happens to score the same, and then a comparison against the
    baseline would be measuring two things at once.
    """
    import pprint

    return (
        '"""An adopted mode: the winner of a run, pinned.\n\n'
        f"Adopted from `{package}/{mode_name}`. The class is the one that was\n"
        "searched, so its ops, its knobs and its `score()` are the same code; only\n"
        "the state differs, and the state is what the search found. Written by\n"
        "`refactory_dsh.adopt`; edit it like any other mode.\n"
        '"""\n\n'
        "from __future__ import annotations\n\n"
        "from typing import Any\n\n"
        f"from refactory_packages.{package}.mode import {_class_name(package, mode_name, project)}\n\n"
        "#: The winning state. A dict literal rather than a sidecar file so the mode\n"
        "#: is reproducible from the source alone.\n"
        f"PINNED: dict[str, Any] = {pprint.pformat(state, width=88, sort_dicts=True)}\n\n\n"
        f"class Adopted({_class_name(package, mode_name, project)}):\n"
        '    """The winner of a run, as a mode of its own."""\n\n'
        f"    name = {mode_name!r}\n"
        "    adopted = True\n\n"
        "    def __init__(self, state: dict[str, Any] | None = None) -> None:\n"
        "        super().__init__(state=state if state is not None else PINNED)\n"
    )


def _class_name(package: str, mode_name: str, project: Path) -> str:
    """The class named by a mode, read from the package rather than guessed.

    ``project`` is required rather than defaulted: the searched package may be a
    workspace package, and the run happened in its own process, so this one has to
    install that root itself or the import fails for a mode it just searched.
    """
    loaded = registry.load(package, project)
    for mode_class in loaded.modes:
        if mode_class.name == mode_name:
            return mode_class.__name__
    raise AdoptionError(
        f"{package} has no mode {mode_name!r}; known: "
        f"{[m.name for m in loaded.modes]}"
    )


def _check_holdout(run_root: Path, candidate_id: str, *, allow_overfit: bool) -> None:
    """Require a completed audit for this winner, including an explicit no-holdout result."""
    path = run_root / "holdout.json"
    try:
        audit = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AdoptionError(
            "no readable held-out audit is recorded; finish the search before adopting"
        ) from error
    if not isinstance(audit, dict) or audit.get("winner") != candidate_id:
        raise AdoptionError("the held-out audit does not describe the current winner; re-audit it before adopting")
    status = audit.get("status")
    if status is None:
        # Older journals did not name audit status. Accept their actual measured
        # score or explicit lack of a split, never a failed/unavailable measurement.
        instances = audit.get("instances") or {}
        if audit.get("holdout") is not None:
            status = "measured"
        elif instances.get("holdout") == 0 or str(audit.get("note", "")).startswith("the mode has no data node"):
            status = "not_applicable"
    if status not in ("measured", "not_applicable"):
        raise AdoptionError(f"cannot adopt: {audit.get('note') or 'the required held-out audit is unavailable'}")
    if audit.get("overfit") and not allow_overfit:
        raise AdoptionError(
            f"the winner's held-out gap {audit.get('gap')} exceeds {audit.get('threshold')}; "
            "pass --allow-overfit to adopt it anyway"
        )


def adopt(
    run_root: Path,
    *,
    as_name: str,
    project: Path | None = None,
    mode: str = "",
    allow_overfit: bool = False,
) -> dict[str, Any]:
    """Write the winner as a new package, honoring a legacy pending selection.

    ``mode`` is the identifier the run searched, as ``<package>/<name>``. It is
    read from the run's own journal when not given, so a caller does not have to
    remember what a run was about. A completed held-out audit must describe the
    winner. ``allow_overfit`` acknowledges a measured gap; it does not override
    an execution failure or a missing audit.
    A legacy finished-run Keep/Reject is recovered before selecting the winner;
    it changes selection records, without evaluating or proposing candidates.
    """
    run_root = Path(run_root)
    project = Path(project or Path("."))
    # The name is checked by the same rule the package it writes is held to, and it
    # is checked here rather than after the directory exists: a name that is not a
    # package name used to fail as a half-written package, or as whatever the
    # filesystem said about a path that is not one component.
    _check_name(as_name)
    from refactory_dsh.control import ControlError
    from refactory_dsh.finished_control import recover_selection
    from refactory_dsh.loop import LoopError

    try:
        recovered = recover_selection(run_root, project=project)
    except (ControlError, LoopError, ValueError, OSError) as error:
        raise AdoptionError(f"could not recover the pending selection: {error}") from error
    winner = best(run_root)
    _check_holdout(run_root, winner["candidate"], allow_overfit=allow_overfit)

    if not mode:
        for entry in _journal_entries(run_root):
            if entry.get("event") == "start" and entry.get("mode"):
                mode = str(entry["mode"])
                break
    if "/" not in mode:
        raise AdoptionError(
            f"cannot tell which mode {run_root} searched; pass it as <package>/<name>"
        )
    package, mode_name = mode.split("/", 1)

    destination = registry.workspace_root(project) / as_name
    if destination.exists():
        raise AdoptionError(
            f"{destination} already exists; adopting never overwrites, so choose "
            "another name or remove it first"
        )

    destination.mkdir(parents=True)
    # A half-written package is worse than none: adopting never overwrites, so a
    # residue makes the name unusable, and the catalogue warns about a package whose
    # mode module was never written. Written first, moved into place only when every
    # file is there.
    try:
        (destination / "__init__.py").write_text(
            '"""A package holding one adopted mode.\n\n'
            f"Produced by `refactory_dsh.adopt` from the run at {run_root}.\n"
            '"""\n\n'
            "from __future__ import annotations\n\n"
            "from refactory_dsh.nn import Package\n\n"
            "from .mode import Adopted\n\n"
            f"PACKAGE = Package(\n"
            f"    name={as_name!r},\n"
            f"    description='The winner of a run of {mode}, as a mode of its own.',\n"
            "    modes=(Adopted,),\n"
            f"    ops={('refactory_packages.' + package + '.ops',)!r},\n"
            ")\n"
        )
        (destination / MODE_FILE).write_text(
            _mode_source(package, mode_name, winner["state"], project)
        )
        (destination / ADOPTED).write_text(
            json.dumps(
                {
                    "adopted_from": mode,
                    "run": str(run_root),
                    "candidate": winner["candidate"],
                    "score": winner["score"],
                    "edits": winner["edits"],
                    "resolved_profile": json.loads((run_root / "resolved-profile.json").read_text()) if (run_root / "resolved-profile.json").is_file() else None,
                    "holdout": json.loads((run_root / "holdout.json").read_text()),
                },
                indent=2,
            )
        )
    except Exception:
        import shutil

        shutil.rmtree(destination, ignore_errors=True)
        raise

    registry.install_workspace(project)
    loaded = registry.load(as_name, project)
    result = {
        "package": as_name,
        "mode": f"{as_name}/{loaded.modes[0].name}",
        "adopted_from": mode,
        "candidate": winner["candidate"],
        "score": winner["score"],
        **{key: winner[key] for key in ("raw_score", "score_note") if key in winner},
        "edits": winner["edits"],
        "package_root": str(destination),
    }
    if recovered is not None:
        result["recovered_control"] = recovered
    return result
