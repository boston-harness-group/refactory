"""Worker routes and durable receipts over DSH's named-provider registry.

A worker performs one node. Graph order, output checks and gates stay with
Refactory. Capabilities describe this adapter, not everything a product can do.
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKENDS = ('dsh', 'codex', 'claude_code', 'openshell')
PERMISSIONS = {
    'dsh': ('skip', 'default'),
    'codex': ('skip', 'default', 'never', 'approve-for-me', 'dangerously-bypass-approvals-and-sandbox'),
    'claude_code': ('skip', 'default', 'dontAsk', 'acceptEdits', 'auto', 'plan', 'bypassPermissions'),
    'openshell': ('skip', 'default'),
}
SKIP_PERMISSIONS = {'codex': 'dangerously-bypass-approvals-and-sandbox',
                    'claude_code': 'bypassPermissions'}
PACKAGES = {'codex': '@deepseek-ai/dsh-subagent-codex',
            'claude_code': '@deepseek-ai/dsh-subagent-claude-code'}
OPENCODE_PROVIDER = 'opencode-go'
OPENCODE_API_KEY_ENV = 'OPENCODE_API_KEY'
OPENCODE_SESSION_HEADER = 'x-opencode-session'


def capabilities(backend: str) -> dict[str, Any]:
    if backend not in BACKENDS:
        raise ValueError(f'unknown worker backend {backend!r}; choose {", ".join(BACKENDS)}')
    return {'model_selection': True, 'provider_selection': backend == 'dsh',
            'reasoning_effort': backend == 'dsh', 'usage': backend == 'dsh',
            'cancel': True, 'resume': False, 'inflight_steering': False,
            'images': False, 'user_questions': False,
            'steering': 'next-node-boundary', 'permissions': list(PERMISSIONS[backend]),
            'progress': 'model-request' if backend == 'dsh' else 'lifecycle'}


def validate_route(route: dict[str, Any]) -> dict[str, Any]:
    route = {**route}
    backend = route.setdefault('worker_backend', 'dsh')
    capabilities(backend)
    unknown = set(route) - {'worker_backend', 'provider', 'model', 'effort', 'permission_mode',
                             'openshell_gateway', 'openshell_image'}
    if unknown:
        raise ValueError(f'unknown worker route fields: {sorted(unknown)}')
    if any(not isinstance(value, str) for value in route.values()):
        raise ValueError('worker route fields must be strings')
    if backend != 'dsh' and (route.get('provider') or route.get('effort')):
        raise ValueError(f'{backend} uses native provider settings and does not support provider/effort overrides')
    if route.get('permission_mode', 'skip') not in PERMISSIONS[backend]:
        raise ValueError(f'unsupported permission mode for {backend}: {route.get("permission_mode")}')
    return route


def resolve_route(base: dict[str, Any], *, role: str = 'worker', node: str = '',
                  role_routes: dict[str, Any] | None = None,
                  node_routes: dict[str, Any] | None = None) -> dict[str, Any]:
    """Explicit node overrides role overrides run; runtime supplies empty fields."""
    route = dict(base)
    for override in ((role_routes or {}).get(role, {}), (node_routes or {}).get(node, {})):
        if not isinstance(override, dict):
            raise ValueError('role/node worker route overrides must be objects')
        # A backend switch must not inherit a different product's route.
        if override.get('worker_backend', route.get('worker_backend')) != route.get('worker_backend'):
            # These two policies have the same meaning across every backend.
            # Preserve an explicit native default in historical profiles; only
            # backend-specific permission values need a fresh default.
            permission = route.get('permission_mode')
            route.update(provider='', model='', effort='',
                         permission_mode=permission if permission in ('default', 'skip') else 'skip')
        route.update(override)
    return validate_route(route)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def event(workdir: Path, record: dict[str, Any], state: str, summary: str) -> None:
    directory = workdir / '.refactory'
    directory.mkdir(parents=True, exist_ok=True)
    payload = {'schema_version': 1, 'run_id': str(workdir.resolve()), 'at': now(),
               'kind': 'worker', 'worker_id': record['worker_id'],
               'node_id': record.get('node_id'), 'state': state, 'summary': summary,
               'route': record.get('actual_route'), 'usage': record.get('usage')}
    with (directory / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(payload, separators=(',', ':')) + '\n')


def begin(workdir: Path, route: dict[str, Any], *, node: str = '', role: str = 'worker') -> Path:
    identifier = 'worker-' + uuid.uuid4().hex
    path = workdir / '.refactory' / 'workers' / (identifier + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {'schema_version': 1, 'worker_id': identifier, 'run_id': str(workdir.resolve()),
              'node_id': node, 'role': role, 'cwd': str(workdir.resolve()), 'started': now(),
              'state': 'starting', 'requested_route': route, 'actual_route': None,
              'capabilities': capabilities(route['worker_backend']),
              'usage': {'status': 'unavailable', 'tokens': None},
              'runtime': {'backend': route['worker_backend'], 'version': None}}
    path.write_text(json.dumps(record, indent=2) + '\n')
    event(workdir, record, 'starting', f"Starting {route['worker_backend']} worker")
    return path


def finish(path: Path, state: str, *, error: str | None = None) -> dict[str, Any]:
    record = json.loads(path.read_text())
    record.update(state=state, finished=now())
    if error:
        record['error'] = error
    path.write_text(json.dumps(record, indent=2) + '\n')
    event(path.parent.parent.parent, record, state, error or f'Worker {state}')
    return record


def permission_patch(route: dict[str, Any]) -> list[dict[str, Any]]:
    """Scope DSH bypass to this runtime, without leaking through child env."""
    if route.get('worker_backend', 'dsh') == 'dsh' and route.get('permission_mode', 'skip') == 'skip':
        return [{'id': 'sandbox-policy', 'config': {'mode': 'danger-full-access'}},
                {'id': 'approval', 'config': {'policy': 'never'}}]
    return []


def _ensure_opencode_session(providers: dict[str, Any], route_provider: str) -> dict[str, Any]:
    """Inject x-opencode-session for OpenCode worker routes.

    The built-in opencode-go provider wraps every request with
    withOpenCodeSessionHeader, but provider_config creates a generic provider
    that lacks that wrapper. When the route selects opencode-go and the
    provider entry doesn't already carry the header, inject a per-run session
    id so workers are distinguishable from each other and from the GUI session.
    """
    if route_provider != OPENCODE_PROVIDER:
        return providers
    entry = providers.get(OPENCODE_PROVIDER, {'apiKeyEnv': OPENCODE_API_KEY_ENV})
    headers = entry.get('headers', {})
    if any(k.lower() == OPENCODE_SESSION_HEADER for k in headers):
        return providers
    patched = dict(entry)
    patched['headers'] = {**headers, OPENCODE_SESSION_HEADER: f'refactory-worker-{uuid.uuid4().hex[:12]}'}
    return {**providers, OPENCODE_PROVIDER: patched}


def provider_rows(route: dict[str, Any], provider_config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Declare selected provider routes to a runtime that cannot see profiles.

    A nested runtime boots the headless profile, which cannot see provider
    profiles declared in another profile's patch (the web session's, say).
    Declared provider routes ride the worker patch so the same selection is
    resolvable in the nested composition; credentials stay in the native
    store their apiKeyEnv names. When the route selects opencode-go, a
    per-worker session header is injected automatically.
    """
    if route.get('worker_backend', 'dsh') != 'dsh':
        return []
    providers = _ensure_opencode_session(dict(provider_config or {}), route.get('provider', ''))
    for name, cfg in list(providers.items()):
        cfg = dict(cfg)
        # Agent-node completions can think for minutes before streaming
        # tokens; disable the idle timeout so long reasoning isn't killed.
        cfg.setdefault('streamIdleTimeoutMs', None)
        providers[name] = cfg
    if providers:
        return [{'id': 'llm-pi-ai', 'config': {'providers': providers}}]
    return []


def write_patch(path: Path, runtime_root: Path, route: dict[str, Any],
                provider_config: dict[str, Any] | None = None) -> None:
    """Compose installed DSH services, never a parallel agent wire protocol."""
    observer = Path(__file__).with_name('worker_observer.mjs')
    rows: list[dict[str, Any]] = [{'insert': [
        {'id': 'refactory-worker-observer', 'name': str(observer)}]}]
    rows.extend(permission_patch(route))
    rows.extend(provider_rows(route, provider_config))
    inner_backend = route['worker_backend']
    if inner_backend == 'openshell':
        inner_backend = 'dsh'
    if inner_backend != 'dsh':
        package = PACKAGES[inner_backend]
        provider = runtime_root / 'node_modules' / package
        if not (provider / 'package.json').exists():
            raise ValueError(f'{package} is not installed; install optional worker dependencies in {runtime_root}')
        config: dict[str, Any] = {'providerName': 'refactory-worker'}
        if route.get('model'):
            config['model'] = route['model']
        permission = route.get('permission_mode', 'skip')
        if permission != 'default':
            config['permissionMode'] = SKIP_PERMISSIONS[route['worker_backend']] if permission == 'skip' else permission
        rows.append({'insert': [{'id': 'refactory-external-worker', 'name': str(provider / 'lib' / 'index.js'), 'config': config}]})
        # A patch's name is an identity guard, not a replacement operation.
        # Disable the ordinary driver and insert the named-provider driver.
        rows.append({'id': 'headless-runner', 'disabled': True})
        rows.append({'insert': [{'id': 'refactory-worker-driver',
                                'name': str(Path(__file__).with_name('worker_bridge.mjs')), 'config': {}}]})
    # JSON is valid YAML; this avoids embedding untrusted model ids into !!js.
    path.write_text(json.dumps(rows, indent=2) + '\n')


def lower_workflow(workflow: Any, workdir: Path, model_context: dict[str, Any] | None,
                   declarations: dict[str, Any] | None = None) -> Any:
    """Resolve explicit node workers before graph dispatch, preserving all edges.

    Each dispatched AgentNode carries a deterministic command the runtime runs.
    The original prompt stays in the node so wf_edit can amend it.
    """
    if model_context is None:
        return workflow
    import hashlib
    import shlex
    from refactory_dsh.ir import AgentNode
    from refactory_dsh.model import DshModelClient

    # Routing/serialization does not dispatch work. The fresh invocation installs
    # its ledger atomically with this lowering, so it may not exist yet.
    routing_context = dict(model_context)
    budget_ledger = routing_context.pop('budget_ledger', None)
    client = DshModelClient(**routing_context)
    result = workflow.model_copy(deep=True)
    directory = Path(workdir) / '.refactory' / 'worker-specs'
    for name, node in list(result.nodes.items()):
        if not isinstance(node, AgentNode):
            continue
        selected = client.for_node(name, role=node.role.value)
        if node.tools:
            raise ValueError(f'worker node {name!r} declares tool restrictions this bridge cannot enforce')
        directory.mkdir(parents=True, exist_ok=True)
        selected_context = selected.context()
        if budget_ledger:
            selected_context['budget_ledger'] = budget_ledger
        selected_context.update(role_routes={}, node_routes={})
        if node.timeout is not None:
            # The node's own bound is its model session's wall (`Node.timeout_s`).
            selected_context['node_wall_s'] = node.timeout
        is_optional = bool((declarations or {}).get(name, {}).get('optional'))
        # The worker captures an answer it was not written into the first write, and reads
        # the produced fact from it (`agent_delivery.finish`, `worker_node.execute`), as the
        # walker does with the node's first declared write. The IR's writes are a set, so
        # that output comes from the declarations, and the rest stay sorted.
        writes = sorted(node.writes)
        answer = (declarations or {}).get(name, {}).get('answer_file')
        if answer in writes:
            writes = [answer, *(item for item in writes if item != answer)]
        payload = {'schema_version': 1, 'node_id': name,
                                    'prompt': node.prompt_template, 'reads': sorted(node.reads),
                                    'writes': writes,
                                    'produces': list((declarations or {}).get(name, {}).get('produces', [])),
                                    'post_checks': [check.model_dump() for check in node.post_checks],
                                    'model_context': selected_context}
        if is_optional:
            payload['optional'] = True
        serialized = json.dumps(payload, indent=2, sort_keys=True) + '\n'
        spec = directory / (hashlib.sha256(serialized.encode()).hexdigest() + '.json')
        spec.write_text(serialized)
        # The dispatch command wraps one model completion with its declared retry
        # budget, and the spine must outlast that budget or the node dies at the
        # host shell's default clamp with the work still running (#289). A node
        # timeout left unset derives from the selected route: the same envelope the
        # dispatched session runs under, computed once so the layers cannot
        # disagree about when a node's work ends (#307).
        from refactory_dsh.model import DEFAULT_TIMEOUT_S, NODE_DISPATCH_OVERHEAD_S, node_session_budget_s
        attempts = max(1, int(selected_context.get('attempts') or 1))
        per_completion = int(selected_context.get('timeout_s') or DEFAULT_TIMEOUT_S)
        if node.timeout is not None:
            # The session inside stops at the node's own bound; the dispatch keeps
            # the overhead a derived envelope has, so the session ends first.
            command_timeout_s = node.timeout + NODE_DISPATCH_OVERHEAD_S
        else:
            command_timeout_s = node_session_budget_s(attempts, per_completion)
        node.dispatch_command = f'{sys.executable} -m refactory_dsh.worker_node ' + shlex.quote(str(spec))
        node.command_timeout_s = command_timeout_s
    return result


def environment(context: dict[str, Any] | None) -> dict[str, str]:
    """Nonsecret resolved context for graph-owned calls.

    Does not set PATH or PYTHONPATH; the target's environment is applied by
    model._environment() at node dispatch time.
    """
    if not context:
        return {}
    result = {'REFACTORY_WORKER_CONTEXT': json.dumps(context, separators=(',', ':'))}
    # Tool subprocesses keep these ordinary runtime variables through DSH's
    # credential scrub.  PATH carries the target's Python when a worktree
    # exists (set by model._environment at dispatch time); PYTHONPATH is not
    # set, so a target that is refactory itself is tested from its own tree.
    if context.get('python_bin'):
        bindir = str(context['python_bin'])
        result.update(VENV_BIN=bindir, VIRTUAL_ENV=str(Path(bindir).parent))
    if context.get('budget_ledger'):
        result['REFACTORY_BUDGET_LEDGER'] = str(context['budget_ledger'])
    return result


def observer_patch(path: Path, route: dict[str, Any],
                   provider_config: dict[str, Any] | None = None) -> None:
    rows = [{'insert': [{'id': 'refactory-worker-observer',
                        'name': str(Path(__file__).with_name('worker_observer.mjs'))}]}]
    rows.extend(permission_patch(route))
    # The coordinator's runtime is as nested as a node's: without this row the
    # graph profile cannot resolve the selected provider and silently dispatches
    # its own default instead.
    rows.extend(provider_rows(route, provider_config))
    path.write_text(json.dumps(rows) + '\n')


def account_usage(budget: Any, receipt: dict[str, Any], *, final: bool = False) -> None:
    """Keep measured DSH calls while exposing incomplete delegated coverage."""
    usage = receipt.get('usage', {})
    identifier = receipt['worker_id']
    node_id = receipt.get('node_id', '')
    if usage.get('incomplete_nested_usage'):
        budget.record_usage(identifier, usage.get('known_tokens'), status='measured', node_id=node_id)
        budget.record_usage(identifier + ':unmetered-descendants', None, status='unavailable', node_id=node_id)
    elif final or usage.get('status') == 'measured':
        budget.record_usage(identifier, usage.get('tokens'), status=usage.get('status', 'unavailable'), node_id=node_id)
    # Check per-step warning threshold after recording.
    if node_id:
        node_total = budget.node_tokens(node_id)
        budget.warn_if_exceeded(node_id, node_total)


class GraphTelemetry:
    """One coordinator receipt; node subprocesses own distinct worker receipts."""
    def __init__(self, workdir: Path, context: dict[str, Any], budget: Any = None):
        self.budget = budget
        route = {'worker_backend': 'dsh', 'provider': '', 'model': '', 'effort': '', 'permission_mode': 'default'}
        if context.get('worker_backend', 'dsh') == 'dsh':
            route.update({key: str(context.get(key) or '') for key in ('provider', 'model', 'effort')})
            route['permission_mode'] = context.get('permission_mode') or 'default'
        self.path = begin(workdir, route, role='coordinator')
        self.patch = self.path.with_suffix('.patch.yml')
        observer_patch(self.patch, route, provider_config=context.get('provider_config'))
        self.context = context

    def command(self, argv: list[str]) -> list[str]:
        result = list(argv)
        index = result.index('--profile')
        result[index:index] = ['--patch', str(self.patch)]
        return result

    def environment(self) -> dict[str, str]:
        return {**environment(self.context), 'DSH_HOME': self.context['dsh_home'],
                'REFACTORY_WORKER_RECEIPT': str(self.path),
                'REFACTORY_RUNTIME_ROOT': self.context['runtime_root']}

    def account(self, *, final: bool = False) -> None:
        if self.budget is None:
            return
        receipt = json.loads(self.path.read_text())
        usage = receipt.get('usage', {})
        if usage.get('incomplete_nested_usage') or usage.get('status') == 'measured' or (final and receipt.get('actual_route')):
            account_usage(self.budget, receipt, final=final)

    def checkpoint(self, *, final: bool = False) -> None:
        self.account(final=final)
        if self.budget is not None:
            self.budget.checkpoint()

    def finish(self, state: str, error: str | None = None) -> None:
        self.account(final=True)
        finish(self.path, state, error=error)


def run_graph_command(argv: list[str], *, workdir: Path, context: dict[str, Any],
                      timeout: float | None = None, env: dict[str, str] | None = None,
                      stop_condition: Any | None = None,
                      **kwargs: Any) -> Any:
    """Inherit terminal IO while polling graph resource and coordinator receipts."""
    if context.get('worker_backend') == 'openshell':
        from refactory_dsh.openshell_backend import run_in_openshell
        return run_in_openshell(argv, workdir=workdir, context=context,
                                timeout=timeout, env=env,
                                stop_condition=stop_condition, **kwargs)
    import os
    import subprocess
    import time
    from refactory_dsh.budgets import RunBudget
    from refactory_dsh.owned_process import _termination_as_exit, shutdown, STOP_CONDITION_GRACE_S

    environment_values = {**os.environ, **(env or {})}
    ledger = context.get('budget_ledger') or environment_values.get('REFACTORY_BUDGET_LEDGER')
    budget = RunBudget.from_path(ledger) if ledger else None
    telemetry = GraphTelemetry(workdir, context, budget)
    environment_values.update(telemetry.environment())
    try:
        telemetry.checkpoint()
    except BaseException as error:
        telemetry.finish('budget_exhausted', str(error))
        raise
    with _termination_as_exit():
        try:
            process = subprocess.Popen(telemetry.command(argv), env=environment_values,
                                       start_new_session=os.name == 'posix', **kwargs)
        except BaseException as error:
            telemetry.finish('failed', str(error))
            raise
        deadline = time.monotonic() + timeout if timeout else None
        try:
            while True:
                telemetry.checkpoint()
                if deadline is not None and time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(argv, timeout)
                # dm-93: poll stop_condition; when terminal, wait the grace
                # period and re-check before acting.  A transient failed ->
                # executing transition (a node retry) must not end the owner.
                if stop_condition is not None:
                    try:
                        should_stop = stop_condition()
                    except Exception:
                        should_stop = False
                    if should_stop:
                        # Wait grace period; the process may exit on its own
                        try:
                            code = process.wait(timeout=STOP_CONDITION_GRACE_S)
                            break
                        except subprocess.TimeoutExpired:
                            pass
                        # Re-check: only stop if condition still holds after grace
                        try:
                            still_stopped = stop_condition()
                        except Exception:
                            still_stopped = False
                        if not still_stopped:
                            continue  # no longer terminal; keep polling
                        # Still terminal — close stdin and terminate
                        if process.stdin is not None:
                            try:
                                process.stdin.close()
                            except OSError:
                                pass
                        try:
                            code = process.wait(timeout=5)
                            break
                        except subprocess.TimeoutExpired:
                            shutdown(process, label='graph runtime')
                            telemetry.finish('stopped', 'stop condition (terminal status)')
                            return subprocess.CompletedProcess(argv, process.returncode or 1)
                try:
                    code = process.wait(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException as error:
            shutdown(process, label='graph runtime')
            state = 'timed_out' if isinstance(error, subprocess.TimeoutExpired) else (
                'budget_exhausted' if type(error).__name__ == 'BudgetExhausted' else 'cancelled')
            telemetry.finish(state, str(error))
            raise
    try:
        telemetry.checkpoint(final=True)
    except BaseException as error:
        telemetry.finish('budget_exhausted', str(error))
        raise
    telemetry.finish('completed' if code == 0 else 'failed')
    return subprocess.CompletedProcess(argv, code)
