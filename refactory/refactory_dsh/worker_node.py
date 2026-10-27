"""Graph-owned agent dispatch bridge used by CLI and native GUI single runs."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
from typing import Any


def _resolve_pin_python(workdir: Path) -> str | None:
    """Read the driver's ``pin_checkout`` from run-config.json, return pin Python or ``None``."""
    config_path = workdir / '.refactory' / 'run-config.json'
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text())
        pin_co = config.get('driver', {}).get('pin_checkout')
        if pin_co:
            return str(Path(pin_co) / '.venv' / 'bin' / 'python')
    except Exception:
        pass
    return None


def _check_driver_mid_run(workdir: Path) -> str | None:
    """Return an error message if the driver moved and no pin exists, else ``None``.

    Called at worker dispatch time (brief R5): a process started mid-run after
    the driver moved either uses the pin or refuses with a message naming the
    run, the recorded commit and the command that pins it.
    """
    config_path = workdir / '.refactory' / 'run-config.json'
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text())
        driver = config.get('driver')
        if not driver:
            return None
        if driver.get('pin_checkout'):
            return None  # already pinned; handled by _resolve_pin_python
        commit = driver.get('commit')
        checkout = driver.get('checkout')
        if not commit or not checkout:
            return None
        import subprocess as _sp
        result = _sp.run(
            ['git', '-C', checkout, 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip() != commit:
            return (
                f'the driver moved since run {workdir} launched '
                f'(recorded {commit[:12]}, now {result.stdout.strip()[:12]}). '
                f'Pin with: refactory graph-control resume --run {workdir}'
            )
    except Exception:
        pass
    return None


def execute(spec: dict[str, Any], workdir: Path) -> dict[str, Any]:
    # When pinned, override model_context so workers use the pin's paths
    # for nested dispatch and DSH CLI resolution.
    _config_path = workdir / '.refactory' / 'run-config.json'
    if _config_path.exists() and spec.get('model_context'):
        try:
            _cfg = json.loads(_config_path.read_text())
            _pin_co = _cfg.get('driver', {}).get('pin_checkout')
            if _pin_co:
                spec = {**spec, 'model_context': {
                    **spec['model_context'],
                    'runtime_root': _pin_co,
                    'python_bin': str(Path(_pin_co) / '.venv' / 'bin'),
                }}
        except Exception:
            pass
    from refactory_dsh.agent_delivery import _path
    from refactory_dsh.model import DshModelClient
    if spec.get('schema_version') != 1:
        raise ValueError('unsupported worker spec version')
    for name in spec['reads']:
        if not _path(workdir, name).exists():
            raise ValueError(f'required worker input is missing: {name}')
    from refactory_dsh.run import NODE_ENV, _knobs, _read_facts
    manifest = json.loads(os.environ.get(NODE_ENV, '{}'))
    prompt_file = manifest.get('prompt_file')
    feedback_file = manifest.get('feedback_file')
    prompt = spec['prompt']
    if prompt_file:
        prompt_path = Path(prompt_file)
        if prompt_path.exists():
            prompt = prompt_path.read_text(encoding='utf-8')
    feedback = ''
    if feedback_file:
        fb_path = Path(feedback_file)
        if fb_path.exists():
            feedback = fb_path.read_text(encoding='utf-8')

    node = spec['node_id']
    from refactory_dsh.nn.executor import RunContext, render
    from refactory_dsh.agent_delivery import fill_feedback
    context = RunContext(workdir, knobs=_knobs(), facts=_read_facts(workdir))
    # Re-fill feedback keys before rendering — a failure to re-derive
    # a declared key refuses the dispatch (dm-120).
    values = {**context.knobs, **context.facts}
    values, refilled_keys = fill_feedback(
        workdir, node, values,
        feedback_keys=spec.get("feedback_keys"),
    )
    if refilled_keys:
        context = RunContext(workdir, knobs=context.knobs, facts={**context.facts, **{k: values[k] for k in refilled_keys}})
    # The node's own role, as the walker's session records it; a spec from before the
    # role was written keeps the generic one.
    client = DshModelClient(**spec['model_context']).for_node(node, role=spec.get('role', 'worker'))
    # A node that declares a workspace works there (`workspace.py`): prepared before the
    # delivery bracket opens, so the contract can name it, and collected into the node's
    # outputs before the bracket closes, so delivery checks what the workspace produced.
    from refactory_dsh import workspace as workspaces
    from refactory_dsh.agent_delivery import with_feedback
    from refactory_dsh.node_retry import deliver
    runtime_root = (client.context() or {}).get('runtime_root') if spec.get('workspace') else None

    def dispatch_workspace(retrying: bool):
        return workspaces.dispatched(
            workdir, node, spec.get('workspace'), knobs=context.knobs, facts=context.facts,
            route=client.route(), fresh=retrying, runtime_root=Path(runtime_root) if runtime_root else None)

    def ask(attempt, sent, place):
        if place is None:
            return attempt.complete(sent, workdir=workdir)
        return workspaces.ask(place, attempt, sent, workdir=workdir)

    output, receipt = deliver(
        workdir, node, prompt=with_feedback(render(prompt, context), feedback),
        writes=spec['writes'], reads=spec['reads'], client=client, ask=ask,
        capture_final=client.route()["worker_backend"] == "dsh", rule=spec.get('retry'),
        timeout_s=spec['model_context'].get('node_wall_s'), budget=getattr(client, '_budget', None),
        workspace=dispatch_workspace)
    for check in spec.get('post_checks', []):
        artifact = _path(workdir, check['path'])
        if check.get('must_exist', True) and not artifact.exists():
            raise ValueError(f'worker artifact check failed: {check["path"]} missing')
        if artifact.is_file():
            if artifact.stat().st_size < check.get('min_size', 0):
                raise ValueError(f'worker artifact check failed: {check["path"]} too small')
            for text in check.get('must_contain', []):
                if text not in artifact.read_text():
                    raise ValueError(f'worker artifact check failed: {check["path"]} missing required text')
    from refactory_dsh.run import declarations, update_facts
    produces = spec.get('produces')
    if produces is None:
        produces = declarations(workdir, node)['produces']
    if produces:
        value = output
        if spec['writes']:
            try:
                value = _path(workdir, spec['writes'][0]).read_text()
            except (UnicodeDecodeError, IsADirectoryError):
                pass
        # Only this fact: a sibling branch may have written others since this one began.
        update_facts(workdir, {produces[0]: value})
    return receipt


def main() -> int:
    try:
        workdir = Path.cwd()
        # ── mid-run driver check (brief R5) ──
        # If the run is pinned, re-exec under the pin's Python so the
        # worker loads the pinned code.  If not pinned but the driver
        # moved, refuse with a diagnostic message.
        _pin_py = _resolve_pin_python(workdir)
        if _pin_py:
            # Compare sys.prefix (the venv root) with the pin's venv, not
            # os.path.realpath: uv venvs symlink to the shared interpreter,
            # so realpath resolves both to the same binary and the switch
            # never fires.  Re-exec with ``-m refactory_dsh.worker_node``
            # so the pin's package is loaded, not sys.argv[0] (which under
            # ``-m`` is the driver's worker_node.py file path).
            _pin_venv = str(Path(_pin_py).parent.parent)
            if sys.prefix != _pin_venv:
                # Guard against infinite re-exec: if we already re-exec'd
                # once and are still not on the pin, something is broken.
                if os.environ.get('_REFACTORY_WORKER_REEXEC'):
                    print(
                        f'worker: re-exec did not land on the pin '
                        f'(expected prefix {_pin_venv}, got {sys.prefix}); '
                        f'the pin venv may be broken',
                        file=sys.stderr,
                    )
                    return 1
                if os.path.isfile(_pin_py):
                    os.environ['_REFACTORY_WORKER_REEXEC'] = '1'
                    os.execv(_pin_py, [_pin_py, '-m', 'refactory_dsh.worker_node', *sys.argv[1:]])
                print(
                    f'worker: pinned Python {_pin_py} not found; '
                    f'run `refactory pin list` to inspect',
                    file=sys.stderr,
                )
                return 1
        else:
            _drift_msg = _check_driver_mid_run(workdir)
            if _drift_msg:
                print(f'worker: {_drift_msg}', file=sys.stderr)
                return 1
        path = Path(sys.argv[1])
        serialized = path.read_text()
        if path.parent.name == 'worker-specs':
            import hashlib
            if hashlib.sha256(serialized.encode()).hexdigest() != path.stem:
                raise ValueError('worker spec was modified after graph preparation; refusing execution')
        spec = json.loads(serialized)
        try:
            receipt = execute(spec, Path.cwd())
            print(json.dumps({'node': receipt['node'], 'status': receipt['status']}))
            return 0
        except Exception as error:
            if spec.get('optional'):
                node_id = spec.get('node_id', 'unknown')
                reviews_dir = workdir / 'reviews'
                reviews_dir.mkdir(parents=True, exist_ok=True)
                (reviews_dir / f'{node_id}-missing.txt').write_text(
                    f'{type(error).__name__}: {error}\n')
                # dm-232: settle the pending delivery marker at the point
                # the branch is recorded missing, so it never goes stale.
                try:
                    from refactory_dsh.agent_delivery import invalidate as _invalidate
                    _invalidate(workdir, [node_id], f'optional branch missing: {error}')
                except Exception as settle_error:
                    # The branch stays missing either way; a marker left pending only
                    # makes the next resume ask for `graph-control acknowledge`.
                    print(f'refactory: run warning: could not settle the delivery marker of '
                          f'optional {node_id}: {type(settle_error).__name__}: {settle_error}; '
                          f'a later resume asks for graph-control acknowledge', file=sys.stderr)
                print(json.dumps({'node': node_id, 'status': 'missing'}))
                return 0
            raise
    except Exception as error:
        print(f'worker: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
