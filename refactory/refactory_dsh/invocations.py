"""Start a fresh graph without treating earlier outputs as current results.

Explicit redo has its own invalidation contract. Preparing/opening a graph is a
new invocation: archive the old run's owned outputs and runtime facts, retain
external inputs and archive the steering ledger, and publish the new lowering together.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from refactory_dsh import lowering, run_state
from refactory_dsh.nn.executor import ITEMS_OVERRIDE
from refactory_dsh.run_artifacts import archive_root, overlaps, owned_outputs
from refactory_dsh.steering import LEDGER as STEERING_LEDGER, installing_to
from refactory_dsh.run_ownership import LOCK, ownership

# ── constants that core and product share ──────────────────────────────────────
# Moved here (core) so that neither side needs to import the other to name them.
# Product modules re-export from their original locations for backward compat.

#: Where a run records the worktree it works in.
WORKTREE_RECORD = Path(".refactory") / "worktree.json"

#: Where an attempt records the files it is about to write.
PENDING = Path(".refactory") / "materialise-pending.json"

#: The script reviewers run commands through.
WRAPPER = Path(".refactory") / "in-worktree"

#: The HOME a check is given.
CHECK_HOME = Path(".refactory") / "check-home"

#: The HOME the unchanged base is prepared and checked with.
BASE_CHECK_HOME = Path(".refactory") / "base-check-home"

#: The HOME the check copy is prepared and checked with (`repo_change._check_copy`): a
#: setup that keeps what it wrote into HOME (a profile link to the checkout) must point at
#: the copy, not at the run's worktree.
CHECK_COPY_HOME = Path(".refactory") / "check-copy-home"

#: How long one install may take before it is reported as failed.
INSTALL_TIMEOUT_S = 1800

#: Suppress git hook managers so installs and commands do not rewrite hooks (#347).
HOOK_SUPPRESS: dict[str, str] = {"LEFTHOOK": "0", "HUSKY": "0"}

#: Where a run keeps its target's baseline.
TARGET_RECORD = Path(".refactory") / "target.json"

#: What a run that was approved up front told the person delivery would do.
DELIVERY_SHOWN = Path(".refactory") / "delivery-shown.json"

#: What a node that calls run_check writes.
CHECK_WRITES = ("changes.diff", "test-guardedness.json", "changed-functions.json")

#: What `design_ops.prepare_check` records about the attempt the check and its reviews
#: run on, side by side; a review that starts before the check records binds to it.
CHECK_PREP = "check-prep.json"

#: What the check beside the reviews decided; `design_ops.gate_check` routes on it.
CHECK_VERDICT = "check-verdict.json"

#: Every self-check `design_ops.prepare_check` runs, one JSON object a line, appended.
SELF_CHECK_RECORD = "self-check-record.jsonl"

#: What a node that calls run_check produces.
CHECK_PRODUCES = ("commit", "files", "inherited_failures", "changed_functions")

INVOCATION = Path('.refactory/invocation.json')

#: Where the console records about a run it started.
CONSOLE_FILE = "console.json"

#: The nodes whose delivery must carry authoring evidence.
REVIEWERS = ("health_checker", "code_reviewer", "qa_director", "overwatch", "focused_qa")

#: The callable names of delivery's entry points, used by graph_control to find
#: the deliver node by its callable rather than by its id or a command substring.
DELIVER_ENTRY_POINTS = frozenset({"refactory_dsh.delivery:deliver_design"})

#: Where a run records its authoring evidence.
EVIDENCE = "authoring-evidence.json"


def _inputs(workdir: Path, declarations: dict[str, Any]) -> list[Path]:
    nodes = [node for node in declarations.values() if isinstance(node, dict)]

    def relative(path: Path) -> Path:
        if path.is_absolute() and path.is_relative_to(workdir):
            return path.relative_to(workdir)
        return path

    def canonical(path: Path) -> Path:
        return relative((workdir / path).resolve())

    writes = [canonical(Path(name)) for node in nodes for name in node.get('writes', [])]
    inputs: list[Path] = []
    for node in nodes:
        for name in node.get('reads', []):
            path = Path(name)
            resolved = canonical(path)
            if not any(overlaps(resolved, output) for output in writes):
                # Preserve both an alias and its target. A lexical-only match can
                # archive a directory still read through an in-run symlink or .. .
                inputs.extend([relative(path), resolved])
    # A runtime source override is an input even when an older graph owned the
    # directory it now points at. Never archive that input out from under the run.
    sources = [node.get('data', {}).get('source_path') for node in nodes]
    override = workdir / ITEMS_OVERRIDE
    if override.exists():
        payload = json.loads(override.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f'{ITEMS_OVERRIDE} must contain an object')
        sources.extend(entry.get('source_path') for entry in payload.values() if isinstance(entry, dict))
    for source in sources:
        if not source:
            continue
        path = Path(source)
        resolved = path.resolve() if path.is_absolute() else (workdir / path).resolve()
        if resolved.is_relative_to(workdir):
            inputs.extend([relative(path), resolved.relative_to(workdir)])
    inputs.extend([Path(ITEMS_OVERRIDE), LOCK])
    return inputs


def prepare(
    state: dict[str, Any], workdir: Path, *, name: str, transform: Callable,
    resource_limits: dict[str, Any] | None = None,
    intent: str = "",
) -> Path:
    """Stage a valid lowering, archive prior ownership, then install or roll back."""
    with ownership(workdir):
        workdir = Path(workdir).resolve()
        runtime = workdir / '.refactory'
        if runtime.is_symlink() or not runtime.resolve().is_relative_to(workdir):
            raise ValueError('cannot prepare a run with .refactory outside its work directory')
        runtime.mkdir(parents=True, exist_ok=True)
        declaration_path = workdir / lowering.DECLARATIONS
        prior: dict[str, Any] = {}
        if declaration_path.exists():
            prior = json.loads(declaration_path.read_text())
            if not isinstance(prior, dict):
                raise ValueError('existing run declarations must contain an object; use a new work directory')
        elif ((workdir / 'facts.json').exists() or (runtime / 'spine-status.json').exists()) \
                and not run_state.walked(workdir):
            # A graph walked without lowering (`nn.execute`: an outer-loop candidate, a
            # user's graph) leaves its facts and status but no declarations; its files
            # are archived below like any earlier run's.
            raise ValueError('existing run has no ownership declarations; use a new work directory')
        writes = [name for node in prior.values() if isinstance(node, dict) for name in node.get('writes', [])]
        targets, retained = owned_outputs(workdir, writes, _inputs(workdir, lowering.node_declarations(state)))
        metadata = [Path(STEERING_LEDGER), Path('facts.json'), Path('graphs'), Path('.memory'), Path(lowering.DECLARATIONS),
                    Path('.refactory/spine-status.json'), Path('.refactory/agent-delivery'),
                    Path('.refactory/events.jsonl'), Path('.refactory/spine-checkpoint.json'),
                    Path('.refactory/graph-controls'), Path('.refactory/graph-control-receipt.json'),
                    Path('.refactory/resource-usage.json'), Path('.refactory/workers'),
                    Path('.refactory/resume-request.json'), Path('.refactory/run-config.json'),
                    Path('.refactory/progress.json'),
                    TARGET_RECORD, WORKTREE_RECORD, PENDING, WRAPPER,
                    CHECK_HOME, BASE_CHECK_HOME, DELIVERY_SHOWN, INVOCATION]
        for relative in metadata:
            path = workdir / relative
            if not path.exists() and not path.is_symlink():
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(workdir):
                raise ValueError(f'cannot archive runtime path {str(relative)!r} outside the run')
            targets.append(relative)
        # A runtime directory can subsume an output already selected beneath it.
        # Move each tree only once, shallowest first.
        selected: list[Path] = []
        for relative in sorted(set(targets), key=lambda path: (len(path.parts), str(path))):
            if not any(parent in relative.parents for parent in selected):
                selected.append(relative)
        targets = selected
        identifier = uuid.uuid4().hex
        archive = archive_root(workdir, 'invocations') / identifier
        record = {'id': identifier, 'mode': name, 'started': datetime.now(timezone.utc).isoformat(),
                  'previous': str(archive.relative_to(workdir)) if targets else None,
                  'archived': [str(path) for path in targets], 'retained_inputs': retained}
        # Nothing from the old invocation moves until the complete new lowering exists.
        with tempfile.TemporaryDirectory(prefix='prepare-', dir=runtime) as temporary:
            stage = Path(temporary)
            # Seed the ledger into the stage so the fold reads the fresh intent.
            if intent:
                from refactory_dsh import steering as steering_mod
                steering_mod.Ledger.seed(stage, intent)
            # Folded from the stage, cited where the ledger will be installed (dm-229).
            with installing_to(workdir):
                lowering.write_mode_graph(
                    state, stage / 'graphs', name=name,
                    transform=lambda workflow: transform(workflow, stage_dir=stage),
                )
            lowering.write_declarations(state, stage)
            (stage / INVOCATION).write_text(json.dumps(record, indent=2) + '\n')
            fresh_metadata: list[Path] = []
            if resource_limits is not None:
                from refactory_dsh.budgets import LEDGER, RunBudget

                RunBudget.create(stage, resource_limits)
                fresh_metadata.append(LEDGER)
            moved: list[Path] = []
            installed: list[Path] = []
            try:
                if targets:
                    archive.mkdir(parents=True)
                    (archive / 'archive.json').write_text(json.dumps(record, indent=2) + '\n')
                for relative in targets:
                    destination = archive / 'artifacts' / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    (workdir / relative).rename(destination)
                    moved.append(relative)
                ledger_install = [Path(STEERING_LEDGER)] if (stage / STEERING_LEDGER).exists() else []
                for relative in [Path('graphs'), Path(lowering.DECLARATIONS), INVOCATION, *fresh_metadata, *ledger_install]:
                    (stage / relative).rename(workdir / relative)
                    installed.append(relative)
            except Exception:
                for relative in reversed(installed):
                    path = workdir / relative
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                for relative in reversed(moved):
                    (archive / 'artifacts' / relative).rename(workdir / relative)
                raise
        return workdir / 'graphs'
