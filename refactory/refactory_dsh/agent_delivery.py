"""Agent artifact delivery and context receipts shared by the walker and DSH.

Receipts establish which context was supplied and which bytes were delivered. They
are not model acknowledgements or evidence that the work satisfies that context.
Artifact bytes and the model's final response are retained separately, unchanged.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from refactory_dsh import target_guard
from refactory_dsh.invocations import EVIDENCE, REVIEWERS, WORKTREE_RECORD

ROOT = Path('.refactory/agent-delivery')

# ── feedback re-fill ──────────────────────────────────────────────────────────


def fill_feedback(workdir: Path, node: str, values: dict[str, Any], *,
                  feedback_keys: tuple[str, ...] | list[str] | None = None,
                  ) -> tuple[dict[str, Any], list[str]]:
    """Re-derive missing feedback keys from persisted files before rendering.

    Core function that resolves the ``refactory.feedback_refill`` entry-point
    group.  Called at the three render sites (worker_node, nn/executor,
    agent_delivery.main) before ``render()``.

    Parameters
    ----------
    workdir : Path
        The run directory.
    node : str
        The node id, used to read ``declarations.json[node]["feedback_keys"]``
        when *feedback_keys* is ``None``.
    values : dict
        The ``{**knobs, **facts}`` dict that ``render`` will use.
    feedback_keys : tuple or list or None
        When a tuple or list, used as the key list directly.  When ``None``,
        reads the node's keys from ``declarations.json`` (for the worker route
        and agent_delivery.main).
    """
    if feedback_keys is None:
        # Read from declarations.json
        decl_path = Path(workdir) / ".refactory" / "declarations.json"
        if decl_path.is_file():
            try:
                import json as _json_mod
                decls = _json_mod.loads(decl_path.read_text())
                node_decl = decls.get(node) or {}
                feedback_keys = tuple(node_decl.get("feedback_keys") or ())
            except (OSError, ValueError, KeyError):
                feedback_keys = ()
        else:
            feedback_keys = ()

    if not feedback_keys:
        return values, []

    # Check if all keys are filled
    missing = [k for k in feedback_keys if not values.get(k)]
    if not missing:
        return values, []

    # Resolve the entry point
    import importlib.metadata
    eps = importlib.metadata.entry_points(group='refactory.feedback_refill')
    if not eps:
        # No entry point installed — refuse with "re-run uv sync"
        raise RuntimeError(
            "feedback keys are unfilled but no refactory.feedback_refill entry point is "
            "installed; re-run `uv sync` or `./install.sh`"
        )

    ep = list(eps)[0]
    refill_fn = ep.load()
    return refill_fn(Path(workdir), feedback_keys, values)

#: The heading a reloop's feedback travels under, ahead of the node's prompt.
FEEDBACK_HEADING = '## Feedback for this visit'


def with_feedback(prompt: str, feedback: str) -> str:
    """``prompt`` with the feedback a reloop left for this visit, said once.

    A gate that sends a node back says why, and the node's next prompt must carry it.
    Most prompts already do -- their template renders the gate's reason from a fact
    (design mode's `emit`: "What went wrong last time") -- and repeating it under a
    second heading only doubles a long check log. So the feedback is added, under
    :data:`FEEDBACK_HEADING` ahead of the prompt, unless the rendered prompt already
    says it (:func:`_rendered`). Both runtimes deliver through here: the spine's worker
    with the feedback file the spine wrote, the walker with the reason its gate gave.
    """
    feedback = (feedback or '').strip()
    if not feedback or _rendered(feedback, prompt):
        return prompt
    return f'{FEEDBACK_HEADING}\n\n{feedback}\n\n{prompt}'


#: How a gate's reason over the runner's limit is clipped: head, this marker, tail
#: (`run._clipped`, and `reasonOf` in workflow-spine's `execute.ts`).
_CLIPPED = re.compile(r'\n… \(clipped from \d+ characters\) …\n')


def _rendered(feedback: str, prompt: str) -> bool:
    """Whether ``prompt`` already says ``feedback``: its lines appear, whole and in a row,
    among the prompt's. Text that merely occurs inside a longer line is not the same
    words -- a note "add tests" is not said by "add tests for every change". A reason
    clipped to the runner's limit is said when its head and its tail both are, the head
    ending and the tail starting where the clip cut a line."""
    lines = [line.rstrip() for line in prompt.split('\n')]
    clipped = _CLIPPED.split(feedback)
    if len(clipped) == 2:
        head, tail = clipped
        return _lines_in(head, lines, cut_last=True) and _lines_in(tail, lines, cut_first=True)
    return _lines_in(feedback, lines)


def _lines_in(text: str, lines: list[str], *, cut_first: bool = False, cut_last: bool = False) -> bool:
    """Whether ``text``'s lines are a run of ``lines``; a cut first line need only end one,
    a cut last line only begin one."""
    wanted = [line.rstrip() for line in text.split('\n')]
    last = len(wanted) - 1
    for start in range(len(lines) - last):
        for offset, want in enumerate(wanted):
            have = lines[start + offset]
            cut_start, cut_end = cut_first and offset == 0, cut_last and offset == last
            same = (want in have if cut_start and cut_end else have.endswith(want) if cut_start
                    else have.startswith(want) if cut_end else have == want)
            if not same:
                break
        else:
            return True
    return False


CONTRACT_START = '<!-- refactory worker execution -->'
CONTRACT_END = '<!-- end refactory worker execution -->'

_UNSET: object = object()
_worktree_state: Any | None | object = _UNSET
_delivery_identity: Any | None | object = _UNSET
_skip_receipt: Any | None | object = _UNSET


def get_worktree_state() -> Any | None:
    """Return the registered worktree-state namespace, discovering it via entry points on first call."""
    global _worktree_state
    if _worktree_state is _UNSET:
        eps = importlib.metadata.entry_points(group='refactory.worktree_state')
        if len(eps) > 1:
            raise RuntimeError(
                f"expected at most one 'refactory.worktree_state' entry point, "
                f"got {len(eps)}: {[ep.name for ep in eps]}"
            )
        if eps:
            ep = next(iter(eps))
            _worktree_state = ep.load()
        else:
            _worktree_state = None
    return _worktree_state


def get_delivery_identity() -> Any | None:
    """Return the registered delivery-identity callable, discovering it via entry points on first call."""
    global _delivery_identity
    if _delivery_identity is _UNSET:
        eps = importlib.metadata.entry_points(group='refactory.delivery_identity')
        if len(eps) > 1:
            raise RuntimeError(
                f"expected at most one 'refactory.delivery_identity' entry point, "
                f"got {len(eps)}: {[ep.name for ep in eps]}"
            )
        if eps:
            ep = next(iter(eps))
            _delivery_identity = ep.load()
        else:
            _delivery_identity = None
    return _delivery_identity


def get_skip_receipt() -> Any | None:
    """Return the registered skip-receipt recorder, discovering it via entry points on first call."""
    global _skip_receipt
    if _skip_receipt is _UNSET:
        eps = importlib.metadata.entry_points(group='refactory.skip_receipt')
        if len(eps) > 1:
            raise RuntimeError(
                f"expected at most one 'refactory.skip_receipt' entry point, "
                f"got {len(eps)}: {[ep.name for ep in eps]}"
            )
        if eps:
            ep = next(iter(eps))
            _skip_receipt = ep.load()
        else:
            _skip_receipt = None
    return _skip_receipt


def _execution_contract(workdir: Path, node: str, prompt: str, reads: list[str],
                        writes: list[str], model_context: dict[str, Any] | None,
                        *, set_aside: dict[str, str] | None = None,
                        workspace: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    """Orient one node without making its worker discover the delivery protocol.

    A node that works in a workspace (``workspace``, `workspace.Workspace.describe`) is
    told where it is and how its work leaves it: its diff becomes the answer file, or its
    outputs are copied out of it. Its outputs are listed in the workspace, since that is
    where it writes them; the run directory is outside what its tools may change.
    """
    context = model_context or {}
    # Absolute path to refactory's own Python, for nested dispatch (worker_node,
    # sandboxed validation). This is NOT the target's interpreter.
    python = str(Path(context['python_bin']) / 'python') if context.get('python_bin') else sys.executable
    # Capture the real import root before any rebase (dm-159 correction 4.2).
    real_refactory_source = str(Path(__file__).resolve().parent)
    execution = {
        'node': node, 'workdir': str(workdir),
        # In name order on both runtimes: the lowered graph keeps a node's reads as a set,
        # and the gate's review scope is fingerprinted over them sorted (`Ledger.review`).
        'inputs': [str((workdir / name).resolve()) for name in sorted(reads)],
        'outputs': [str(_path(workdir, name)) for name in writes],
        'python': python, 'refactory_source': real_refactory_source,
    }
    if workspace is not None:
        place = Path(workspace['path'])
        execution['workspace'] = dict(workspace)
        if workspace.get('collect') == 'diff':
            # The answer is derived from the diff; the rest are the final response.
            execution['outputs'] = []
            execution['derived'] = {writes[0]: 'the workspace\'s diff', **{
                name: 'your final response' for name in writes[1:]}} if writes else {}
        else:
            execution['outputs'] = [str(place / name) for name in writes]
    if context.get('runtime_root'):
        execution['runtime_root'] = str(context['runtime_root'])

    # Rebase paths when the target IS the driver checkout (dm-159).
    ws = get_worktree_state()
    wt_found = ws.recorded(workdir) if ws else None
    if wt_found is not None:
        target_root = Path(wt_found.target)
        worktree_path = Path(wt_found.path)
        src = Path(execution['refactory_source'])
        if src.is_relative_to(target_root):
            execution['refactory_source'] = str(worktree_path / src.relative_to(target_root))
        if 'runtime_root' in execution:
            rt = Path(execution['runtime_root'])
            if rt.is_relative_to(target_root):
                execution['runtime_root'] = str(worktree_path / rt.relative_to(target_root))
    di = get_delivery_identity()
    if di is None and node in REVIEWERS and (workdir / EVIDENCE).exists():
        raise RuntimeError(
            "the delivery-identity entry point is not installed but this node "
            "needs authoring evidence; re-run `uv sync` or `./install.sh`"
        )
    if node in REVIEWERS:
        authoring = di(workdir) if di else None
        if authoring is not None:
            execution['authoring'] = authoring
    if set_aside:
        execution['set_aside'] = set_aside
    # A repeated view may already contain our block. Replace only that block;
    # the task and the current steering ledger remain intact.
    start = prompt.find(CONTRACT_START)
    end = prompt.find(CONTRACT_END, start)
    if start >= 0 and end >= start:
        suffix = prompt[end + len(CONTRACT_END):]
        if start == 0 and suffix.startswith('\n\n'):
            suffix = suffix[2:]
        prompt = prompt[:start] + suffix
    instructions = (
        'Execute this workflow node only. The paths below are exact; relative task paths use workdir. '
        'Use the supplied task, API and reference source first; inspect additional code when needed for this task. '
        'The runtime owns delivery receipts and advancing the graph. Its graphs, worker specs, delivery history '
        'and adjacent run reports are not completion instructions; inspect them only if this task explicitly needs them. '
        'Perform the checks required by the current task. After delivering the required outputs, return immediately '
        'with a brief completion response; do not execute later workflow stages or wait for a receipt. '
        'Do not duplicate a file artifact in the final response.'
    )
    if not (workspace is not None and workspace.get('collect') == 'diff'):
        # A node whose answer is its workspace's diff declares no output for it to write.
        instructions += ' If no output file is declared, return the requested result.'
    if set_aside:
        instructions += (
            ' This step\'s previous outputs were moved to the paths listed in set_aside.'
            ' Read them there when the task revises earlier work, and write each output afresh at its declared path.'
        )
    if workspace is not None:
        instructions += _workspace_instructions(workspace, writes)
    # Worktree-aware contract instructions (dm-159).
    if wt_found is not None and workspace is not None:
        # A workspace node reads, changes and runs the repository in its workspace,
        # never in the run's worktree or the target.
        instructions += (
            f' The run\'s worktree at {wt_found.path} holds the attempts as committed;'
            f' do not change it, and never change the target at {wt_found.target}.'
            ' The python above is for nested dispatch only.'
            f' It imports refactory from {real_refactory_source}, not from your workspace.'
        )
    elif wt_found is not None:
        tg_baseline = target_guard._read(workdir)
        is_output_only = tg_baseline is not None and node in tg_baseline.get('output_only', ())
        instructions += (
            f' Read the repository in the worktree at {wt_found.path},'
            f' and give paths relative to it.'
            f' Write only the paths in `outputs` below, in the run directory.'
            f' Never change the target at {wt_found.target}.'
        )
        if is_output_only:
            instructions += ' Do not change the worktree.'
        # In-worktree wrapper (invocations.WRAPPER).
        from refactory_dsh.invocations import WRAPPER
        wrapper_path = Path(workdir) / WRAPPER
        if wrapper_path.exists():
            instructions += (
                f' Run repository commands only through {wrapper_path}.'
            )
        # Python interpreter warning — name the real import root (dm-159 correction 4.2).
        instructions += (
            ' The python above is for nested dispatch only.'
            f' It imports refactory from {real_refactory_source},'
            ' so a refactory tool run with it writes there from any directory,'
            ' and such a write is refused.'
        )
        if is_output_only:
            instructions += (
                " Do not run generators; the run's generate and check steps do."
            )
    block = f'{CONTRACT_START}\n{instructions}\n{json.dumps(execution, indent=2)}\n{CONTRACT_END}'
    return f'{block}\n\n{prompt}', execution


def _workspace_instructions(workspace: dict[str, Any], writes: list[str]) -> str:
    """What a workspace node is told about where it works and how its work is taken."""
    place = workspace['path']
    kind = workspace.get('kind')
    if kind == 'clone':
        where = (f' Your workspace is {place}: a clone of the run\'s worktree at its '
                 f'{"base" if workspace.get("of") == "base" else "last attempt"}, with its'
                 ' dependencies installed and its setup and generate commands run.'
                 ' Read, change and run the repository there (its own virtual environment and'
                 ' package binaries come first on PATH), and give paths relative to it.')
    else:
        where = (f' Your workspace is {place}, a new empty folder: work and run commands there,'
                 ' and give paths relative to it.')
    sandbox = str(workspace.get('sandbox') or '')
    if sandbox.startswith('workspace-write'):
        where += ' Your tools may change files only there and in the temporary directory.'
    if workspace.get('problems'):
        where += (' Preparing it reported a problem (listed under workspace.problems below); the check'
                  ' runs the same steps, so fix it there if your change caused it.')
    if workspace.get('normalised'):
        where += (' Your last answer changed the line endings or the blank lines at the ends of'
                  f' {", ".join(workspace["normalised"])}: an emission carries each file with LF line'
                  ' endings and one final newline, and the attempt holds them that way.')
    if workspace.get('collect') == 'diff' and writes:
        taken = (f' When you are done, stop: the workspace\'s diff is your answer. The runtime writes'
                 f' {writes[0]} from it, so do not write {writes[0]} yourself, and leave in the'
                 ' workspace only the change you mean: every file you add, change or delete there is'
                 ' part of it, except its .refactory folder, environments (.venv, node_modules) and tool'
                 ' caches (__pycache__ and the like). Text files only:'
                 ' a binary file, a symlink or a file-mode change is refused.')
        if len(writes) > 1:
            taken += f' Your final response is saved as {", ".join(writes[1:])}.'
    else:
        taken = (' Write each output at its path relative to your workspace (listed in `outputs`);'
                 ' the runtime copies it to the run directory when you stop.')
    return where + taken


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(workdir: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'agent output {name!r} must be relative to its run directory')
    path = workdir / relative
    if not path.resolve().is_relative_to(workdir.resolve()):
        raise ValueError(f'agent output {name!r} resolves outside its run directory')
    return path


def _fingerprint(workdir: Path, name: str) -> dict[str, Any] | None:
    path = _path(workdir, name)
    if not path.exists():
        return None
    if path.is_dir():
        # Delivery's own audit writes cannot make a declared directory look new.
        files = [p for p in sorted(path.rglob('*')) if p.is_file()
                 and not p.is_relative_to(workdir / ROOT)]
        return {'kind': 'directory', 'inode': path.stat().st_ino, 'files': {
            str(p.relative_to(path)): _fingerprint(workdir, str(p.relative_to(workdir)))
            for p in files
        }}
    stat = path.stat()
    return {'kind': 'file', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns,
            'inode': stat.st_ino}


def _set_aside(workdir: Path, record: dict[str, Any], names: list[str],
               subfolder: str = 'refused') -> str | None:
    """Move an attempt's outputs out of the run, kept beside its receipt.

    Left in place, a refused output is what the redo's worker finds, and it answers that
    the file is already right rather than writing it afresh. On the CEO's route the redo
    is judged against the snapshot from before the refused attempt, where that same file
    would count as newly delivered.

    *subfolder* selects the archive directory under the receipt: ``refused`` for
    target-guard refusals, ``previous`` for outputs set aside before a re-dispatch.
    """
    if not names:
        return None
    parent = workdir / ROOT / record['id'] / subfolder
    archive = parent / str(sum(1 for _ in parent.glob('*')))
    moved: list[Path] = []
    for name in sorted(names, key=lambda value: len(Path(value).parts)):
        relative = Path(name)
        if any(parent in relative.parents for parent in moved):
            continue
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        _path(workdir, name).rename(destination)
        moved.append(relative)
    return str(archive.relative_to(workdir))


def _set_aside_stale_outputs(workdir: Path, node: str, reads: list[str],
                              writes: list[str]) -> dict[str, str] | None:
    """Move a previous attempt's surviving write-only outputs before a new dispatch.

    On re-dispatch — after a materialise refusal, a failed check, a CEO redo or
    a rerun — each declared output that is write-only (not also read) and still
    exists on disk is moved under the latest receipt's .refactory/agent-delivery/
    folder, so the worker writes each one fresh.  An output the node also reads
    stays in place (it must be read before writing anyway).  An output with no
    receipt from this node is left alone (some other step wrote it).
    """
    write_only = [w for w in writes if w not in reads]
    if not write_only:
        return None
    # Find the latest receipt for this node (the one whose outputs we are moving).
    latest: dict[str, Any] | None = None
    for receipt_path in sorted((workdir / ROOT).glob('*/receipt.json')):
        try:
            receipt = json.loads(receipt_path.read_text())
            if receipt.get('node') != node:
                continue
            if receipt.get('status') not in ('delivered', 'failed', 'superseded'):
                continue
            if latest is None or receipt.get('started', '') >= latest.get('started', ''):
                latest = receipt
        except (OSError, ValueError, KeyError):
            continue
    if latest is None:
        return None
    to_move = [name for name in write_only if _path(workdir, name).exists()]
    if not to_move:
        return None
    archive = _set_aside(workdir, latest, to_move, subfolder='previous')
    if archive is None:
        return None
    # Absolute, like the contract's inputs and outputs.
    return {name: str(workdir / archive / name) for name in to_move}


def _attempt_failed(workdir: Path, record: dict[str, Any]) -> bool:
    """Whether a finish for this pending delivery already ran and refused it."""
    try:
        return json.loads((workdir / record['receipt']).read_text()).get('status') == 'failed'
    except (OSError, ValueError, KeyError):
        return False


def _pending(workdir: Path, node: str) -> Path:
    return workdir / ROOT / 'pending' / (hashlib.sha256(node.encode()).hexdigest() + '.json')


def _watch_checkouts(workdir: Path) -> dict[str, Any]:
    """Snapshot refactory's own checkouts that are not the run's target (dm-164).

    Found from the directory this module, and so the contract's Python, imports
    refactory from -- the contract's ``refactory_source`` -- and from the run's record.
    """
    return target_guard.watch_checkouts(workdir, target_guard.checkouts(workdir, Path(__file__).resolve().parent))


def _save_checkout_baselines(directory: Path, checkouts: dict[str, Any] | None) -> None:
    """Keep each watched checkout's uncommitted bytes for a lossless restore; a redo's replace the last."""
    if (directory / 'checkouts').exists():
        (directory / 'checkouts').rename(directory / f'checkouts-{uuid.uuid4().hex[:8]}')
    for path, snapshot in (checkouts or {}).items():
        if snapshot.get('changes'):
            unsaved = target_guard.save_baseline(Path(path), snapshot['changes'],
                                                 target_guard.checkout_baseline(directory, path))
            if unsaved:
                # A refusal leaves these where they are (`restore_target`), so say so now.
                target_guard.warn(f"refactory's own checkout {path} has uncommitted files whose bytes could "
                                  f"not be kept, so they cannot be put back if this node changes them: "
                                  f"{target_guard._named(unsaved)}")


def begin(workdir: Path, node: str, prompt: str, writes: list[str], *, reuse: bool = False,
          reads: list[str] | None = None, model_context: dict[str, Any] | None = None,
          workspace: dict[str, Any] | None = None) -> dict[str, Any]:
    """Snapshot outputs and supply the current ledger before dispatching one agent.

    DSH can report the same due node repeatedly; reuse keeps its original output
    snapshot until completion rather than mistaking an in-flight write for old data.
    A changed prompt or live ledger is supplied afresh and archived in the context
    history. A DSH view records context supplied to the CEO, not proof of forwarding.
    """
    from refactory_dsh.steering import Ledger, parse_entries, prompt_with_ledger

    workdir = Path(workdir).resolve()
    ws = get_worktree_state()
    if ws is None and (workdir / WORKTREE_RECORD).exists():
        raise RuntimeError(
            "the worktree record exists but no worktree-state entry point is installed; "
            "re-run `uv sync` or `./install.sh`"
        )
    pending = _pending(workdir, node)
    previous = None
    if reuse and pending.exists():
        previous = json.loads(pending.read_text())
        if previous['writes'] != writes:
            raise ValueError(f'agent {node!r} output declarations changed during delivery')
        if ((previous.get('target') is not None or previous.get('checkouts'))
                and _attempt_failed(workdir, previous)):
            # A refused attempt keeps its record for the retry. Judging the retry against
            # the snapshot from before the first attempt would refuse it for changes the
            # person already saw, and chose to keep, every time.
            previous['target'] = target_guard.watch(workdir, node)
            rewatched = previous['target']
            if rewatched is not None and rewatched.get('changes') and (rewatched.get('also') or 'also' not in rewatched):
                baseline_dir = workdir / ROOT / previous['id'] / 'target-baseline'
                if baseline_dir.exists():
                    baseline_dir.rename(baseline_dir.with_name(f"target-baseline-{uuid.uuid4().hex[:8]}"))
                target_guard.save_baseline(Path(rewatched['root']), rewatched['changes'], baseline_dir)
            previous['checkouts'] = _watch_checkouts(workdir)
            _save_checkout_baselines(workdir / ROOT / previous['id'], previous['checkouts'])
            pending.write_text(json.dumps(previous, indent=2) + '\n')
    ledger = Ledger(workdir)
    ledger_text = ledger.path.read_text() if ledger.exists() else ""
    entries = parse_entries(ledger_text)
    annotated = prompt_with_ledger(prompt, ledger, entries=entries)
    set_aside = _set_aside_stale_outputs(workdir, node, reads or [], writes) if previous is None else None
    annotated, execution = _execution_contract(workdir, node, annotated, reads or [], writes, model_context,
                                               set_aside=set_aside, workspace=workspace)
    if previous is not None and previous['prompt'] == annotated:
        return previous
    before = previous['before'] if previous is not None else {name: _fingerprint(workdir, name) for name in writes}
    # Taken here because every route dispatches through this bracket (walker, worker
    # bridge, CEO subagent), so a change any agent node of a worktree run makes to
    # the target is attributed to that node alone, not to whatever ran since
    # prepare (#319, dm-159).
    target = previous.get('target') if previous is not None else target_guard.watch(workdir, node)
    # Refactory's own checkout, when it is not the target: every agent node, any route (dm-164).
    checkouts = previous.get('checkouts') if previous is not None else _watch_checkouts(workdir)
    identifier = previous['id'] if previous is not None else uuid.uuid4().hex
    directory = workdir / ROOT / identifier
    directory.mkdir(parents=True, exist_ok=True)
    # Save pre-node target bytes for lossless restore (dm-159).
    # Only a snapshot taken just now is saved: a reused in-flight dispatch keeps its
    # pre-node copies, which the agent may already have changed in the target.
    if (previous is None and target is not None and target.get('changes')
            and (target.get('also') or 'also' not in target)):
        baseline_dir = directory / 'target-baseline'
        target_guard.save_baseline(Path(target['root']), target['changes'], baseline_dir)
    if previous is None:
        _save_checkout_baselines(directory, checkouts)
    contexts = list(previous.get('contexts', [])) if previous is not None else []
    context = {'at': _now(), 'entries': entries, 'execution': execution,
               'prompt': str(ROOT / identifier / f'context-{len(contexts)}.prompt.txt'),
               'ledger': str(ROOT / identifier / f'context-{len(contexts)}.ledger.md')}
    (workdir / context['prompt']).write_text(annotated)
    (workdir / context['ledger']).write_text(ledger_text)
    contexts.append(context)
    (directory / 'prompt.txt').write_text(annotated)
    if ledger_text:
        (directory / 'ledger.md').write_text(ledger_text)
    prompt_hash = hashlib.sha256(annotated.encode()).hexdigest()
    record = {'id': identifier, 'node': node,
              'started': previous['started'] if previous is not None else context['at'], 'writes': writes,
              'before': before, 'prompt': annotated, 'entries': entries, 'contexts': contexts,
              'execution': execution, 'target': target, 'checkouts': checkouts, 'prompt_hash': prompt_hash,
              'receipt': str(ROOT / identifier / 'receipt.json')}
    (directory / 'receipt.json').write_text(json.dumps({
        **{key: value for key, value in record.items() if key not in ('prompt', 'target', 'checkouts')},
        'status': 'pending', 'artifacts': before,
    }, indent=2) + '\n')
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.write_text(json.dumps(record, indent=2) + '\n')
    return record


def finish(workdir: Path, node: str, *, final_response: str | None = None,
           capture_final: bool = False,
           sent_prompt_hash: str | None = None,
           sent_prompt: str | None = None,
           unresolved_keys: list[str] | None = None) -> dict[str, Any]:
    """Preserve fresh tool-written files; capture final text only as the fallback.

    Every required output must be newly delivered, including unchanged bytes that
    were actually rewritten (stat metadata changes). Pre-existing untouched files
    cannot establish success. No model text is stripped or reformatted.
    """
    workdir = Path(workdir).resolve()
    ws = get_worktree_state()
    if ws is None and (workdir / WORKTREE_RECORD).exists():
        raise RuntimeError(
            "the worktree record exists but no worktree-state entry point is installed; "
            "re-run `uv sync` or `./install.sh`"
        )
    pending = _pending(workdir, node)
    if not pending.exists():
        raise ValueError(f'agent {node!r} was not dispatched; no delivery snapshot exists')
    record = json.loads(pending.read_text())
    directory = workdir / ROOT / record['id']
    if final_response is not None:
        (directory / 'final-response.txt').write_text(final_response)
    writes = record['writes']
    after = {name: _fingerprint(workdir, name) for name in writes}
    fresh = {name for name in writes if after[name] is not None and after[name] != record['before'][name]}
    captured = None
    if capture_final and writes and writes[0] not in fresh and final_response is not None:
        first = _path(workdir, writes[0])
        if first.is_dir():
            raise ValueError(f'agent {node!r} returned text for directory output {writes[0]!r}')
        first.parent.mkdir(parents=True, exist_ok=True)
        first.write_text(final_response)
        captured = writes[0]
        after[captured] = _fingerprint(workdir, captured)
        fresh.add(captured)
    missing = [name for name in writes if name not in fresh]
    fail_changes, warn_changes = target_guard.strayed(
        workdir, record.get('target'))
    # Refactory's own checkout, where it is not the target, is refused for every agent
    # node, a run with or without a worktree alike; `target_guard.CHECKOUT_POLICY` says
    # what happens to its files (dm-164).
    checkout_changes = target_guard.strayed_checkouts(workdir, record.get('checkouts'))
    # A launcher checkout that is also refactory's is refused below, not only warned about.
    warn_changes = [change for change in warn_changes
                    if not any(change.startswith(f"in {path}, ") for path in checkout_changes)]
    set_aside = (_set_aside(workdir, record,
                            [name for name in writes if name in fresh])
                 if fail_changes or checkout_changes else None)
    # Restore the target losslessly on refusal (dm-159).
    restored_info = None
    wt_path = None
    target_before = record.get('target')
    if fail_changes and target_before is not None and (target_before.get('also') or 'also' not in target_before):
        # A worktree run only (a target-only snapshot, or an output-only one with `also`):
        # a run without a worktree is refused exactly as before dm-159.
        if target_before.get('also'):
            wt_path = next(iter(target_before['also']), None)
        else:
            ws_ref = get_worktree_state()
            wt_found_ref = ws_ref.recorded(workdir) if ws_ref else None
            if wt_found_ref is not None:
                wt_path = wt_found_ref.path
        delivery_dir = workdir / ROOT / record['id']
        baseline_dir = delivery_dir / 'target-baseline'
        restored_info = target_guard.restore_target(
            Path(target_before['root']), target_before,
            baseline_dir, workdir)
    checkouts_settled = {path: target_guard.settle_checkout(path, record['checkouts'][path], directory, workdir,
                                                             node=record['node'])
                         for path in checkout_changes}
    all_changes = fail_changes + warn_changes
    receipt = {key: value for key, value in record.items() if key not in ('prompt', 'target', 'checkouts')}
    receipt['prompt_hash'] = record.get('prompt_hash')
    receipt['prompt_path'] = str(ROOT / record['id'] / 'prompt.txt')
    if sent_prompt_hash is not None:
        receipt['sent_prompt_hash'] = sent_prompt_hash
    if sent_prompt is not None:
        sent_path = directory / 'sent-prompt.txt'
        sent_path.write_text(sent_prompt)
        receipt['sent_prompt_path'] = str(ROOT / record['id'] / 'sent-prompt.txt')
    if unresolved_keys:
        receipt['unresolved_keys'] = unresolved_keys
    receipt.update(finished=_now(), status='failed' if missing or fail_changes or checkout_changes else 'delivered',
                   delivery='context supplied; semantic compliance not verified',
                   artifacts=after, captured_final=captured,
                   final_response=str(ROOT / record['id'] / 'final-response.txt') if final_response is not None else None,
                   missing_or_unchanged=missing, **({'target_changes': all_changes} if all_changes else {}),
                   **({'checkout_changes': checkout_changes} if checkout_changes else {}),
                   **({'set_aside': set_aside} if set_aside else {}))
    (directory / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    pending.unlink()
    if warn_changes:
        target_guard.warn(
            f"while agent {record['node']!r} ran, files outside its "
            f"worktree changed: {'; '.join(warn_changes)}. The run works "
            "in its own worktree, so these changes are not its result; "
            "check whether another session or person made them.")
    # Refused even when the declared output arrived: the output may be a summary of
    # edits that never pass through the step that writes the artifact.
    if fail_changes or checkout_changes:
        tg_bl = target_guard._read(workdir)
        reasons = []
        if fail_changes:
            is_oo = (restored_info is not None and tg_bl is not None
                     and record['node'] in tg_bl.get('output_only', ()))
            reasons.append(target_guard.refusal(
                record['node'], record['target']['root'],
                fail_changes, writes, set_aside=set_aside,
                worktree_path=wt_path, restored=restored_info,
                is_output_only=is_oo))
        if checkout_changes:
            reasons.append(target_guard.checkout_refusal(
                record['node'], (tg_bl or {}).get('root'), checkout_changes, checkouts_settled,
                set_aside=None if fail_changes else set_aside))
        raise ValueError(' '.join(reasons))
    if missing:
        raise ValueError(
            f'agent {record["node"]!r} did not deliver required outputs: '
            f'{", ".join(missing)} (missing or unchanged from before dispatch)')
    # Check whether the git configuration or hooks the run's worktree shares
    # with its target changed during this step (#347).  Beside the existing
    # target-guard bracket.
    wt_found = ws.recorded(workdir) if ws else None
    if wt_found is not None and ws.shared_changed(wt_found):
        wt_msg = (f"the git configuration or hooks the run's worktree shares "
                  f"with {wt_found.target} changed during {node}; check them "
                  f"before your next commit there")
        if 'also' in (record.get('target') or {}):
            # Output-only node (full snapshot with also/warn_also tracking):
            # a shared-state change fails the step.
            raise ValueError(f"agent {node!r} changed shared git state: {wt_msg}")
        else:
            target_guard.warn(wt_msg)
    return receipt


def invalidate(workdir: Path, nodes: list[str], reason: str) -> None:
    """Retire in-flight snapshots when an explicit skip or redo invalidates work."""
    for node in nodes:
        pending = _pending(workdir, node)
        if not pending.exists():
            continue
        record = json.loads(pending.read_text())
        receipt = workdir / ROOT / record['id'] / 'receipt.json'
        previous = json.loads(receipt.read_text())
        previous.update(status='superseded', finished=_now(), reason=reason)
        receipt.write_text(json.dumps(previous, indent=2) + '\n')
        pending.unlink()


def receipts_in(workdir: Path) -> dict[str, dict[str, Any]]:
    """Latest receipts, with a separate comparison to the current artifact state.

    A later deterministic node may legitimately enrich an agent's output. That is
    reported as a modification, not as proof that the earlier delivery failed.
    """
    workdir = Path(workdir)
    found: dict[str, dict[str, Any]] = {}
    for path in sorted((workdir / ROOT).glob('*/receipt.json')):
        try:
            receipt = json.loads(path.read_text())
            node = receipt['node']
            previous = found.get(node)
            if previous is not None and previous['started'] >= receipt['started']:
                continue
            current = {name: _fingerprint(workdir, name) for name in receipt['artifacts']}
            receipt['current_artifacts'] = current
            receipt['missing_now'] = [name for name, value in current.items() if value is None]
            receipt['current'] = current == receipt['artifacts']
            found[node] = receipt
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return found


def main() -> int:
    """JSON stdin protocol for the configured DSH dispatch/completion bridge."""
    try:
        request = json.load(sys.stdin)
        workdir = Path.cwd()
        if request['action'] == 'begin':
            import re as _re
            from refactory_dsh.run import _knobs, _read_facts
            from refactory_dsh.nn.executor import RunContext, render
            prompt = request['prompt']
            context = RunContext(workdir, knobs=_knobs(), facts=_read_facts(workdir))
            # Re-fill feedback keys before rendering — a failure to re-derive
            # a declared key refuses the dispatch (dm-120).
            values = {**context.knobs, **context.facts}
            values, refilled_keys = fill_feedback(workdir, request['node'], values)
            context = RunContext(workdir, knobs=context.knobs, facts={**context.facts, **{k: values[k] for k in (refilled_keys or [])}})
            rendered = render(prompt, context)
            # Detect unresolved template keys from the template itself (not
            # substituted values).  Record as a warning, not a refusal: the
            # steering ledger folds the brief verbatim into every template,
            # so a brief containing {{word}} must not block begin (#347, C2).
            # Strip the STEERING_START…STEERING_END block before scanning so
            # keys inside the brief (folded in via the ledger) are not warned.
            from refactory_dsh.steering import STEERING_START, STEERING_END
            scan_prompt = prompt
            st_start = scan_prompt.find(STEERING_START)
            st_end = scan_prompt.find(STEERING_END, st_start) if st_start >= 0 else -1
            if st_start >= 0 and st_end >= st_start:
                scan_prompt = scan_prompt[:st_start] + scan_prompt[st_end + len(STEERING_END):]
            template_keys = _re.findall(r'\{\{\s*([A-Za-z_]\w*)\s*\}\}', scan_prompt)
            unresolved = [k for k in template_keys if k.strip() not in values and k not in (refilled_keys or [])]
            result = begin(workdir, request['node'], rendered, request['writes'], reuse=True,
                           reads=request.get('reads'))
            # The bridge reads the prompt; the target snapshot is this module's own state.
            result = {key: value for key, value in result.items() if key != 'target'}
            if unresolved:
                result['unresolved_keys'] = unresolved
        elif request['action'] == 'finish':
            sent_prompt_hash = request.get('sent_prompt_hash')
            sent_prompt = request.get('sent_prompt')
            unresolved_keys = request.get('unresolved_keys')
            result = finish(workdir, request['node'],
                            sent_prompt_hash=sent_prompt_hash,
                            sent_prompt=sent_prompt,
                            unresolved_keys=unresolved_keys)
        elif request['action'] == 'invalidate':
            nodes = request['nodes']
            reason = request['reason']
            if (not isinstance(nodes, list)
                    or not all(isinstance(node, str) for node in nodes)
                    or not isinstance(reason, str) or not reason.strip()):
                raise ValueError('invalidate needs node ids and a non-empty reason')
            invalidate(workdir, nodes, reason)
            result = {'requested': nodes}
        elif request['action'] == 'reviewers':
            result = {'reviewers': list(REVIEWERS)}
        elif request['action'] == 'record-skip':
            recorder = get_skip_receipt()
            if recorder is None:
                raise RuntimeError(
                    "no 'refactory.skip_receipt' entry point is installed, so a person's "
                    "skip cannot be recorded; re-run `uv sync` in this checkout"
                )
            result = recorder(workdir, str(request.get('node', '')), request.get('provenance', {}))
        elif request['action'] == 'pending-prompt':
            node = request['node']
            pending = _pending(workdir, node)
            if pending.exists():
                record = json.loads(pending.read_text())
                result = {'prompt': record.get('prompt', '')}
            else:
                result = {'prompt': ''}
        else:
            raise ValueError('unknown agent delivery action')
        print(json.dumps(result))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'agent delivery: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
