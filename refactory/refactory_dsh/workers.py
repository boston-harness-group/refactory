"""Worker routes and durable receipts over DSH's named-provider registry.

A worker performs one node. Graph order, output checks and gates stay with
Refactory. Capabilities describe this adapter, not everything a product can do.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from refactory_dsh.tool_policy import no_effect

BACKENDS = ('dsh', 'codex', 'claude_code', 'openshell')
CLAUDE_CODE_EFFORT_LEVELS = frozenset({"low", "medium", "high", "xhigh", "max"})
#: Claude Code's filesystem settings sources (the SDK's ``settingSources``), as the
#: ``claude --setting-sources`` flag spells them; ``none`` loads none of them.
CLAUDE_CODE_SETTING_SOURCES = frozenset({"user", "project", "local"})
#: The oldest Claude Code each model family runs on, by model-id prefix; Claude Code
#: refuses a newer model at its first call ("version X or newer is required"), after
#: the run's setup. A model no prefix matches is not checked.
CLAUDE_CODE_MINIMUM_VERSIONS = (
    (('claude-opus-5', 'claude-sonnet-5', 'claude-haiku-5', 'claude-fable-5'), '2.1.280'),
)
CLAUDE_EXECUTABLE_ENV = 'REFACTORY_CLAUDE_EXECUTABLE'
CLAUDE_AGENT_SDK = '@anthropic-ai/claude-agent-sdk'
ROUTE_FIELDS = frozenset({'worker_backend', 'provider', 'model', 'effort', 'permission_mode',
                          'openshell_gateway', 'openshell_image', 'max_completion_tokens',
                          'setting_sources', 'claude_executable', 'max_thinking_tokens',
                          'max_turns', 'bash_timeout_ms', 'bash_output_chars'})

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
WORKSPACE_COMMAND_TIMEOUT_MS = 600_000

OPENCODE_PROVIDER = 'opencode-go'
OPENCODE_API_KEY_ENV = 'OPENCODE_API_KEY'
OPENCODE_SESSION_HEADER = 'x-opencode-session'

#: Where a DSH process refactory starts keeps its sessions (`session_store_rows`):
#: ``run`` (the default) gives each process its own store beside its receipt, in the
#: run directory; ``shared`` leaves them in ``$DSH_HOME/sessions``, as before.
SESSIONS_ENV = 'REFACTORY_DSH_SESSIONS'
SESSION_STORES = ('run', 'shared')


def capabilities(backend: str) -> dict[str, Any]:
    if backend not in BACKENDS:
        raise ValueError(f'unknown worker backend {backend!r}; choose {", ".join(BACKENDS)}')
    return {'model_selection': True, 'provider_selection': backend == 'dsh',
            'reasoning_effort': backend in ('dsh', 'claude_code'), 'usage': backend == 'dsh',
            'cancel': True, 'resume': False, 'inflight_steering': False,
            'images': False, 'user_questions': False,
            'system_prompt': backend in ('dsh', 'claude_code'),
            'steering': 'next-node-boundary', 'permissions': list(PERMISSIONS[backend]),
            'progress': 'model-request' if backend == 'dsh' else 'lifecycle'}


def validate_route(route: dict[str, Any]) -> dict[str, Any]:
    route = {**route}
    backend = route.setdefault('worker_backend', 'dsh')
    capabilities(backend)
    unknown = set(route) - ROUTE_FIELDS
    if unknown:
        raise ValueError(f'unknown worker route fields: {sorted(unknown)}')
    if any(not isinstance(value, str) for value in route.values()):
        raise ValueError('worker route fields must be strings')
    if backend != 'dsh' and route.get('provider'):
        raise ValueError(f'{backend} uses native provider settings and does not support provider overrides')
    if route.get('effort') and backend not in ('dsh', 'claude_code'):
        raise ValueError(f'{backend} does not support effort overrides')
    if backend == 'claude_code' and route.get('effort') and route['effort'] not in CLAUDE_CODE_EFFORT_LEVELS:
        accepted = ', '.join(sorted(CLAUDE_CODE_EFFORT_LEVELS))
        raise ValueError(f'claude_code effort must be one of {accepted}; got {route["effort"]!r}')
    if route.get('setting_sources'):
        if backend != 'claude_code':
            raise ValueError(f'{backend} does not support setting_sources; only claude_code loads Claude Code settings')
        sources = route['setting_sources'].split(',')
        if route['setting_sources'] != 'none' and not set(sources) <= CLAUDE_CODE_SETTING_SOURCES:
            accepted = ', '.join(sorted(CLAUDE_CODE_SETTING_SOURCES))
            raise ValueError(f'claude_code setting_sources must be "none" or a comma list of {accepted}; '
                             f'got {route["setting_sources"]!r}')
    if route.get('claude_executable'):
        if backend != 'claude_code':
            raise ValueError(f'{backend} does not support claude_executable; only claude_code runs a Claude Code executable')
        # A pure check: whether the file exists and runs is `claude_code_preflight`'s.
        if not os.path.isabs(route['claude_executable']):
            raise ValueError(f'claude_code claude_executable must be an absolute path; '
                             f'got {route["claude_executable"]!r}')
    if route.get('max_thinking_tokens'):
        if backend != 'claude_code':
            raise ValueError(f'{backend} does not support max_thinking_tokens; only claude_code '
                             'takes a thinking budget')
        if not route['max_thinking_tokens'].isdigit() or int(route['max_thinking_tokens']) <= 0:
            raise ValueError('claude_code max_thinking_tokens is a positive whole number of tokens; '
                             f'got {route["max_thinking_tokens"]!r}')
    for key, what in (('max_turns', 'turns'), ('bash_timeout_ms', 'milliseconds'),
                      ('bash_output_chars', 'characters')):
        if route.get(key):
            if backend != 'claude_code':
                raise ValueError(f'{backend} does not support {key}; only claude_code takes it')
            if not route[key].isdigit() or int(route[key]) <= 0:
                raise ValueError(f'claude_code {key} is a positive whole number of {what}; '
                                 f'got {route[key]!r}')
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
            for key in ('setting_sources', 'claude_executable', 'max_thinking_tokens', 'max_turns',
                        'bash_timeout_ms', 'bash_output_chars'):
                if key in route:
                    route[key] = ''
        route.update(override)
    return validate_route(route)


class StepRoutes:
    """The routes a run names -- its own, and each role's and node's in its route maps --
    which bound what a step of a searched graph may choose for itself (`Node.route`): its
    model, effort and output cap, each a value one of those routes uses with the same
    backend and provider. A step is never offered a model, effort or cap the run's author
    did not name."""

    def __init__(self, base: dict[str, Any], role_routes: dict[str, Any] | None = None,
                 node_routes: dict[str, Any] | None = None) -> None:
        self.base = validate_route(dict(base))
        self.role_routes = dict(role_routes or {})
        self.node_routes = dict(node_routes or {})
        self.routes = [self.base,
                       *(resolve_route(self.base, role=role, role_routes=self.role_routes)
                         for role in sorted(self.role_routes)),
                       *(resolve_route(self.base, node=node, node_routes=self.node_routes)
                         for node in sorted(self.node_routes))]

    @classmethod
    def of(cls, model: Any) -> 'StepRoutes | None':
        """A model client's routes, or None for a client that has none (a scripted one)."""
        route = getattr(model, 'route', None)
        if not callable(route):
            return None
        return cls(route(), getattr(model, '_role_routes', None), getattr(model, '_node_routes', None))

    def step_route(self, node: str, role: str) -> dict[str, Any]:
        """The route a step runs on before its own: its role's, then its node's."""
        return resolve_route(self.base, role=role, node=node, role_routes=self.role_routes,
                             node_routes=self.node_routes)

    def values(self, under: dict[str, Any]) -> dict[str, list[str]]:
        """Each field a step may set, with the values the run's routes on ``under``'s
        backend and provider use."""
        from refactory_dsh.nn.graph import STEP_ROUTE_FIELDS
        same = [r for r in self.routes if (r.get('worker_backend'), r.get('provider', ''))
                == (under.get('worker_backend'), under.get('provider', ''))]
        return {key: sorted({str(r[key]) for r in same if r.get(key)}) for key in STEP_ROUTE_FIELDS}

    def offered(self) -> dict[str, list[str]]:
        """What a step on the run's own route may choose: each field's values other than
        the run's own. Empty when there is nothing to choose."""
        offered = {key: [v for v in values if v != str(self.base.get(key) or '')]
                   for key, values in self.values(self.base).items()}
        return {key: values for key, values in offered.items() if values}

    def refusal(self, node: str, role: str, declared: dict[str, str]) -> str | None:
        """Why ``node`` may not run on ``declared``, or None when every field it sets is a
        value the run's routes use and not what the step runs on already."""
        under = self.step_route(node, role)
        allowed = self.values(under)
        for key, value in declared.items():
            if key not in allowed:
                return (f"step {node!r} sets route {key} {value!r}, which a step may not "
                        "choose for itself")
            if value not in allowed[key]:
                return (f"step {node!r} sets route {key} {value!r}, which no route of this run "
                        f"uses; allowed: {', '.join(allowed[key]) or 'none'}")
            if value == str(under.get(key) or ''):
                return (f"step {node!r} sets route {key} {value!r}, which it runs on already; "
                        "leave it out")
        return None


def claude_executable_choice(route: dict[str, Any]) -> str:
    """The Claude Code executable a claude_code route names, else `CLAUDE_EXECUTABLE_ENV`'s;
    empty keeps the one bundled with the Agent SDK."""
    return route.get('claude_executable') or os.path.expanduser(os.environ.get(CLAUDE_EXECUTABLE_ENV, ''))


def claude_code_minimum(model: str) -> str:
    """The oldest Claude Code version ``model`` runs on, or empty when none is known.

    A ``provider/`` prefix and a ``[...]`` suffix (``claude-opus-5-5[1m]``) are ignored."""
    name = model.rsplit('/', 1)[-1].split('[', 1)[0].strip().lower()
    for prefixes, minimum in CLAUDE_CODE_MINIMUM_VERSIONS:
        if name.startswith(prefixes):
            return minimum
    return ''


def _version_tuple(version: str) -> tuple[int, ...] | None:
    """The leading X.Y.Z of ``version`` (``2.1.281-rc`` is 2.1.281), or None without one."""
    match = re.match(r'\s*(\d+)\.(\d+)\.(\d+)', version)
    return tuple(int(part) for part in match.groups()) if match else None


def agent_roles(state: dict[str, Any]) -> dict[str, str]:
    """Each agent node of a mode state and the worker role it is dispatched under.

    The same ``Node.role`` the walker (`nn.executor`) and the spine's lowering
    (`lower_workflow`) pass to `DshModelClient.for_node`; a fork's branches and a data
    pass's steps are nodes of the same graph, so every node a run dispatches is here.
    """
    from refactory_dsh.nn.graph import Graph

    graph = Graph.from_state(state.get('graph') or {})
    return {node.id: node.role for node in graph.nodes.values() if node.kind == 'agent'}


def bundled_claude_code(runtime_root: Path | str) -> dict[str, str] | None:
    """The Claude Code bundled with the Agent SDK the installed adapter resolves:
    ``path`` (empty when no platform package holds one), ``version`` (the SDK's
    ``claudeCodeVersion``, read without starting it) and ``sdk_version``. None when
    the optional adapter is not installed."""
    adapter = Path(runtime_root) / 'node_modules' / PACKAGES['claude_code']
    if not (adapter / 'package.json').exists():
        return None
    # pnpm links the adapter's own dependencies beside it in its store directory.
    sdk = adapter.resolve().parent.parent / CLAUDE_AGENT_SDK
    try:
        metadata = json.loads((sdk / 'package.json').read_text())
    except (OSError, ValueError):
        metadata = {}
    sdk = sdk.resolve()
    # The binary ships in a per-platform sibling package (claude-agent-sdk-darwin-arm64, ...).
    binaries = sorted(path for path in sdk.parent.glob(sdk.name + '-*/claude') if path.is_file())
    return {'path': str(binaries[0]) if binaries else '',
            'version': str(metadata.get('claudeCodeVersion') or ''),
            'sdk_version': str(metadata.get('version') or '')}


#: The environment switches that turn Claude Code's compaction off; the adapter drops them
#: from a claude_code worker's environment (`claude_code_compaction`).
COMPACTION_SWITCHES = ('DISABLE_AUTO_COMPACT', 'DISABLE_COMPACT', 'DISABLE_MICROCOMPACT')


def claude_code_compaction(runtime_root: Path | str) -> dict[str, Any] | None:
    """Whether claude_code workers compact, and what of the person's they override.

    ``forced`` says whether the installed adapter starts Claude Code with the flag setting
    ``autoCompactEnabled: true`` (it is patched to; False means the patch is not applied).
    ``person`` is the person's own ``autoCompactEnabled`` in the global config Claude Code
    reads (`_claude_global_config`; None when unset), which a worker would otherwise read
    through its real HOME and which is never changed; ``dropped`` are the switches in this environment a worker does not get. None
    when the optional adapter is not installed.
    """
    adapter = Path(runtime_root) / 'node_modules' / PACKAGES['claude_code']
    try:
        source = (adapter / 'lib' / 'index.js').read_text()
    except OSError:
        return None
    try:
        person = json.loads(_claude_global_config().read_text()).get('autoCompactEnabled')
    except (OSError, ValueError, AttributeError):
        person = None
    return {'forced': 'settings: { autoCompactEnabled: true }' in source,
            'person': person if isinstance(person, bool) else None,
            'dropped': [name for name in COMPACTION_SWITCHES if name in os.environ]}


def _claude_global_config() -> Path:
    """The global config file Claude Code reads, found as it finds it: a legacy
    ``.config.json`` in its config directory (``CLAUDE_CONFIG_DIR``, else ``~/.claude``)
    when there is one, else ``.claude.json`` in ``CLAUDE_CONFIG_DIR``, else in HOME."""
    configured = os.environ.get('CLAUDE_CONFIG_DIR')
    legacy = Path(configured or Path.home() / '.claude') / '.config.json'
    return legacy if legacy.exists() else Path(configured or Path.home()) / '.claude.json'


def claude_code_version(executable: str, *, exact: bool = False) -> str:
    """The X.Y.Z ``executable --version`` starts with, or empty when it says none. With
    ``exact``, the whole version it reports (its first word: ``2.1.299-beta``, say), for a
    pin (`Mode.claude_code_versions`), which a suffix does not meet."""
    try:
        done = subprocess.run([executable, '--version'], capture_output=True, text=True,
                              timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return ''
    match = re.match(r'\s*(\d+\.\d+\.\d+)', done.stdout)
    if done.returncode != 0 or not match:
        return ''
    return done.stdout.split()[0] if exact else match.group(1)


#: How a refusal says to choose another Claude Code.
_CLAUDE_EXECUTABLE_FIX = (f'--claude-executable /path/to/claude, {CLAUDE_EXECUTABLE_ENV}, '
                          f'or "claude_executable" in refactory.workers.json')


def claude_code_preflight(model: Any, runtime_root: Path | str | None = None, *,
                          roles: dict[str, str] | None = None,
                          pins: dict[str, str] | None = None) -> None:
    """Refuse, before the run directory exists, a claude_code route whose Claude Code
    cannot run its model: a chosen executable that is missing, not executable or whose
    version is unreadable, or a version older than the model needs
    (`CLAUDE_CODE_MINIMUM_VERSIONS`). ``model`` is a client or its `context()`. No model
    is called.

    ``roles`` maps each agent node of the run's graph to its worker role
    (`agent_roles`): each node is checked on the route it will run on, its node route
    over its own role's. Without a graph the run route, each role route and each node
    route (as role ``worker``) are checked. Only a Claude Code problem is refused: a
    combination that is not a valid route is left for dispatch to report.

    ``pins`` maps an agent node to the Claude Code version it must run on
    (`Mode.claude_code_versions`): with ``roles``, a claude_code route for that node whose
    Claude Code reports another version, or none, is refused.
    """
    # Another ModelClient (a test's fake, a library caller's own) carries no route
    # config to read; its own readiness contract applies.
    context = model.context() if callable(getattr(model, 'context', None)) else model
    if not isinstance(context, dict):
        return
    base = {key: value for key, value in context.items() if key in ROUTE_FIELDS and value}
    role_routes, node_routes = context.get('role_routes') or {}, context.get('node_routes') or {}

    def resolved(where: str, role: str = 'worker', node: str = '') -> tuple[str, dict[str, Any]] | None:
        try:
            route = resolve_route(base, role=role, node=node, role_routes=role_routes,
                                  node_routes=node_routes)
        except ValueError:
            return None
        return where, {**route, '_node': node}

    if roles is None:
        candidates = [resolved('the run', role=''),
                      *(resolved(f'role {role!r}', role=role) for role in role_routes),
                      *(resolved(f'node {node!r}', node=node) for node in node_routes)]
    else:
        candidates = [*(resolved(f'node {node!r} (role {role!r})', role=role, node=node)
                        for node, role in roles.items()),
                      # A route map entry for a node the graph does not have yet.
                      *(resolved(f'node {node!r}', node=node) for node in node_routes if node not in roles)]
    routes = [found for found in candidates if found is not None and found[1]['worker_backend'] == 'claude_code']
    if not routes:
        return
    root = runtime_root or context.get('runtime_root')
    if not root:
        from refactory_dsh.paths import dsh_root
        root = dsh_root()
    versions: dict[str, str] = {}
    for where, route in routes:
        chosen = claude_executable_choice(route)
        if chosen and not os.path.isfile(chosen):
            raise ValueError(f'the claude_code worker for {where} is set to run Claude Code at {chosen}, '
                             f'which does not exist; point {_CLAUDE_EXECUTABLE_FIX} at a Claude Code '
                             f'executable')
        if chosen and not os.access(chosen, os.X_OK):
            raise ValueError(f'the claude_code worker for {where} is set to run Claude Code at {chosen}, '
                             f'which is not executable; point {_CLAUDE_EXECUTABLE_FIX} at a Claude Code '
                             f'executable')
        model_id = route.get('model', '')
        minimum = claude_code_minimum(model_id)
        pinned = (pins or {}).get(route['_node'], '') if roles is not None else ''
        if not minimum and not pinned:
            continue
        if chosen:
            if chosen not in versions:
                versions[chosen] = claude_code_version(chosen, exact=True)
            # The exact version for a pin; its leading X.Y.Z for a model's minimum.
            reported, described = versions[chosen], chosen
            leading = re.match(r'\d+\.\d+\.\d+', reported)
            version = leading.group(0) if leading else ''
            if not version:
                needs = (f'model {model_id} needs Claude Code {minimum} or newer' if minimum
                         else f'its mode pins Claude Code {pinned}')
                raise ValueError(f'the claude_code worker for {where} is set to run Claude Code at {chosen}, '
                                 f'but `{chosen} --version` reported no version, and {needs}. Point '
                                 f'{_CLAUDE_EXECUTABLE_FIX} at a Claude Code executable')
        else:
            bundled = bundled_claude_code(root)
            if not bundled or not bundled['version']:
                if pinned:
                    raise ValueError(f'the claude_code worker for {where} must run Claude Code {pinned}, '
                                     f'which its mode pins, and no Claude Code is named or bundled; point '
                                     f'{_CLAUDE_EXECUTABLE_FIX} at Claude Code {pinned}')
                continue  # dispatch reports a missing adapter; nothing to compare against
            version = reported = bundled['version']
            described = (f'{bundled["path"] or "the Agent SDK"} (bundled with {CLAUDE_AGENT_SDK} '
                         f'{bundled["sdk_version"]})')
        if pinned and reported != pinned:
            raise ValueError(f'the claude_code worker for {where} would run Claude Code {reported} at '
                             f'{described}, and its mode pins Claude Code {pinned} (its system prompt '
                             f'and tool text change between versions). Point {_CLAUDE_EXECUTABLE_FIX} '
                             f'at Claude Code {pinned}')
        if not minimum:
            continue
        found = _version_tuple(version)
        if found is None:
            continue  # a version string with no X.Y.Z says nothing to compare
        if found < _version_tuple(minimum):
            raise ValueError(f'the claude_code worker for {where} would run Claude Code {version} at '
                             f'{described}, and model {model_id} needs Claude Code {minimum} or newer. '
                             f'Choose a newer Claude Code with {_CLAUDE_EXECUTABLE_FIX}, or update it')


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


def begin(workdir: Path, route: dict[str, Any], *, node: str = '', role: str = 'worker',
          tool_policy: str = '') -> Path:
    identifier = 'worker-' + uuid.uuid4().hex
    path = workdir / '.refactory' / 'workers' / (identifier + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {'schema_version': 1, 'worker_id': identifier, 'run_id': str(workdir.resolve()),
              'node_id': node, 'role': role, 'cwd': str(workdir.resolve()), 'started': now(),
              'state': 'starting', 'requested_route': route, 'actual_route': None,
              'capabilities': capabilities(route['worker_backend']),
              'usage': {'status': 'unavailable', 'tokens': None},
              'runtime': {'backend': route['worker_backend'], 'version': None}}
    # ── no-tools receipt fields ─────────────────────────────────────────
    if no_effect(tool_policy):
        record['requested_tool_policy'] = tool_policy
        # Nothing is enforced until the runtime says so: DSH's no-tools plugin at the
        # provider request, Claude Code's adapter from the options it sent the SDK and
        # the tools the SDK reported. A worker that never got that far enforced nothing.
        record['enforced_tool_policy'] = None
        record['tool_observation_available'] = False
        # ``none`` permits no tool; ``refused`` offers Claude Code's own, each refused by
        # dontAsk where it needs permission (`tool_policy`).
        record['tools_permitted' if tool_policy == 'none' else 'tools_offered'] = None
        record['tool_policy_source'] = 'not-yet-observed'
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


def permission_patch(route: dict[str, Any], *, workspace_root: Path | None = None) -> list[dict[str, Any]]:
    """Scope DSH bypass to this runtime, without leaking through child env.

    A node that works in a workspace (`workspace.py`) gets DSH's ``workspace-write``
    policy rooted at it instead of full access: its tools may change files there (and
    in the platform's temporary area) and nowhere else, and nothing asks, because
    nobody answers a worker. The caller passes ``workspace_root`` only where DSH can
    enforce the policy (`workspace.confinement`); elsewhere the node keeps today's full
    access and the run records why.
    """
    if route.get('worker_backend', 'dsh') == 'dsh' and route.get('permission_mode', 'skip') == 'skip':
        if workspace_root is not None:
            return [{'id': 'sandbox-policy', 'config': {'mode': 'workspace-write',
                                                        'workspaceRoot': str(workspace_root)}},
                    {'id': 'approval', 'config': {'policy': 'never'}}]
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


def session_store_choice(value: str | None = None) -> str:
    """``run`` or ``shared``: an explicit value, else `SESSIONS_ENV`, else ``run``."""
    choice = (value or os.environ.get(SESSIONS_ENV) or 'run').strip()
    if choice not in SESSION_STORES:
        raise ValueError(f'{SESSIONS_ENV} must be one of {", ".join(SESSION_STORES)}; got {choice!r}')
    return choice


def session_store(receipt: Path, choice: str | None = None) -> Path | None:
    """The DSH session store of the process a receipt records, or None for the shared one.

    DSH keeps every session it runs, and its JSONL backend finds one by id by probing
    every project directory under its root (three lookups each), checks the whole root
    once per process, and its projection cache loads every stored session's record at
    start-up. Under the one shared ``$DSH_HOME`` all of that grows with every run on the
    machine: 52k stored sessions made each helper 3.7 GB and filled the kernel's dentry
    cache with billions of failed lookups. A store per process holds only its own
    sessions, so its cost no longer depends on what ran before. Nothing in refactory
    reads a worker's DSH session back; the store stays beside the receipt as evidence,
    and goes with the run directory.
    """
    if session_store_choice(choice) == 'shared':
        return None
    return receipt.with_suffix('.dsh')


def session_store_rows(store: Path | None) -> list[dict[str, Any]]:
    """Point DSH's session log and its storage domains (the projection cache) at ``store``.

    The two roots the base profile puts under ``$DSH_HOME`` (``sessions``, ``storages``).
    DSH_HOME itself is unchanged: credentials, settings and profiles stay the person's,
    and the run's evaluation context (`profiles`), which names DSH_HOME, is the same.
    """
    if store is None:
        return []
    return [{'id': 'session-persistence-jsonl', 'config': {'root': str(store / 'sessions')}},
            {'id': 'storage-json', 'config': {'root': str(store / 'storages')}}]


def session_patch(workdir: Path) -> Path | None:
    """A patch that gives a coordinator started without a model client its own store.

    `GraphTelemetry` gives a coordinator with a client its store through its observer
    patch; a library caller that starts the spine with no client (`run_on_dsh` without
    ``model``) gets this one, named like a receipt so the store sits with the others.
    None when the shared store is chosen (`SESSIONS_ENV`): there is no context to carry it.
    """
    # Named as a receipt would be; no receipt is written, so receipt readers see none.
    receipt = Path(workdir) / '.refactory' / 'workers' / ('coordinator-' + uuid.uuid4().hex + '.json')
    store = session_store(receipt)
    if store is None:
        return None
    path = receipt.with_suffix('.patch.yml')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session_store_rows(store)) + '\n')
    return path


def write_patch(path: Path, runtime_root: Path, route: dict[str, Any],
                provider_config: dict[str, Any] | None = None, *,
                workspace_root: Path | None = None, tool_home: Path | None = None,
                system_prompt: str = "", tool_policy: str = "",
                seal: dict[str, Any] | None = None,
                session_store: Path | None = None) -> None:
    """Compose installed DSH services, never a parallel agent wire protocol.

    ``workspace_root`` confines the node's tools to its workspace (`permission_patch`).
    ``tool_home`` is the DSH home the node's shell is told (`workspace.tool_dsh_home`):
    the DSH process keeps its own, and a command run in a workspace never reaches it.
    ``system_prompt`` is the rendered system text; when non-empty, a composition plugin
    is inserted that injects it as a system-prompt section.
    ``session_store`` is where this process keeps its DSH sessions (`session_store_rows`);
    None keeps DSH's shared store.
    ``seal`` (``tools``, ``runner``) seals the agent (`refactory_dsh.seal`): DSH's local
    sandbox provider runs each command under ``runner``, and the agent has ``tools`` only.
    On the claude_code backend (``seal["backend"]``, Claude Code's own tools) the adapter
    does: it offers and allows ``tools`` only, in ``workspace_root``, and its hook runs each
    Bash command under ``runner`` and refuses a Read outside the workspace.
    """
    observer = Path(__file__).with_name('worker_observer.mjs')
    rows: list[dict[str, Any]] = [{'insert': [
        {'id': 'refactory-worker-observer', 'name': str(observer)}]}]
    if system_prompt and route['worker_backend'] in ('dsh', 'openshell'):
        # An external worker's adapter takes the text from its own config instead.
        plugin = Path(__file__).with_name('system_prompt_plugin.mjs')
        rows.append({'insert': [{'id': 'refactory-system-prompt', 'name': str(plugin)}]})
    rows.extend(permission_patch(route, workspace_root=workspace_root))
    rows.extend(session_store_rows(session_store))
    if tool_home is not None:
        rows.append({'id': 'shell-env', 'config': {'dshHome': str(tool_home)}})
        if route['worker_backend'] in ('dsh', 'openshell'):
            # DSH otherwise promotes expired commands to unbounded background jobs.
            # Apply to workspace tools only, not the spine's whole-node executor.
            rows.extend([
                {'id': 'bash-sandbox', 'config': {
                    'timeoutMs': WORKSPACE_COMMAND_TIMEOUT_MS,
                    'maxTimeoutMs': WORKSPACE_COMMAND_TIMEOUT_MS}},
                {'id': 'tool-bash', 'config': {
                    'enableRunInBackground': False, 'promoteOnTimeout': False}},
            ])
    claude_seal = seal is not None and seal.get('backend') == 'claude_code'
    if claude_seal:
        from refactory_dsh.seal import CLAUDE_CODE_TOOLS
        if (workspace_root is None or route.get('worker_backend') != 'claude_code'
                or not seal.get('tools') or not set(seal['tools']) <= set(CLAUDE_CODE_TOOLS)
                or not seal.get('runner') or tool_policy):
            raise ValueError("a step with Claude Code's tools runs sealed on the claude_code "
                             "worker backend, confined to its workspace, with no tool policy")
    elif seal is not None:
        from refactory_dsh.seal import FAILURE
        if workspace_root is None or route.get('worker_backend', 'dsh') != 'dsh':
            raise ValueError('a sealed agent runs on the dsh worker backend, confined to its workspace')
        rows.append({'id': 'sandbox', 'config': {'runnerCommand': list(seal['runner']),
                                                 'runnerFailureSignatures': [FAILURE.strip()]}})
        plugin = Path(__file__).with_name('tools_plugin.mjs')
        rows.append({'insert': [{'id': 'refactory-tools', 'name': str(plugin),
                                 'config': {'allow': list(seal['tools'])}}]})
    # ── no-tools enforcement ─────────────────────────────────────────────
    if tool_policy == "none" and route.get('worker_backend', 'dsh') == 'dsh':
        plugin = Path(__file__).with_name('no_tools_plugin.mjs')
        rows.append({'insert': [{'id': 'refactory-no-tools', 'name': str(plugin)}]})
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
        if inner_backend == 'claude_code':
            # The Claude Code adapter takes the node's effort, system prompt, tool policy,
            # settings sources and executable from its own config ``env``, never from this
            # process's environment; it strips them before starting Claude Code. Codex reads
            # none: both runtimes refuse those for codex before dispatch (`capabilities`).
            env = {name: value for name, value in (
                ('REFACTORY_MODEL_EFFORT', route.get('effort', '')),
                ('REFACTORY_SYSTEM_PROMPT', system_prompt),
                ('REFACTORY_TOOL_POLICY', tool_policy),
                ('REFACTORY_SETTING_SOURCES', route.get('setting_sources', '')),
                ('REFACTORY_CLAUDE_EXECUTABLE', claude_executable_choice(route)),
                # How long a step waits on background tasks that report nothing; the
                # adapter's default (600 s) unless this process's environment sets it.
                ('REFACTORY_BACKGROUND_QUIET_S', os.environ.get('REFACTORY_BACKGROUND_QUIET_S', '')),
                ('REFACTORY_MAX_THINKING_TOKENS', route.get('max_thinking_tokens', '')),
                ('REFACTORY_MAX_TURNS', route.get('max_turns', '')),
                ('REFACTORY_BASH_TIMEOUT_MS', route.get('bash_timeout_ms', '')),
                ('REFACTORY_BASH_OUTPUT_CHARS', route.get('bash_output_chars', '')),
                # A sealed step (`refactory_dsh.seal`, Claude Code's own tools): the tools it
                # is offered, its workspace, and the runner its hook puts before each command.
                ('REFACTORY_SEALED_TOOLS', ','.join(seal['tools']) if claude_seal else ''),
                ('REFACTORY_SEAL_WORKSPACE', str(workspace_root) if claude_seal else ''),
                ('REFACTORY_SEAL_RUNNER', json.dumps(list(seal['runner'])) if claude_seal else ''),
            ) if value}
            if env:
                config['env'] = env
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

    A node that declares a workspace is refused on the CEO-subagent route (no
    ``model_context``): the driving agent's subagent runs where the CEO runs, so nothing
    could give it a workspace or collect its output from one.
    """
    from refactory_dsh.ir import AgentNode
    if model_context is None:
        declared = sorted(name for name, node in workflow.nodes.items()
                          if isinstance(node, AgentNode) and node.workspace)
        if declared:
            raise ValueError(
                f"agent node(s) {', '.join(repr(name) for name in declared)} declare a workspace, which "
                "only a dispatched worker can give them; this graph has no worker model context, so "
                "they would run as the CEO's subagent, in the CEO's own directory. Run it with a "
                "worker route (the default for `refactory run` and the GUI).")
        routed = sorted(name for name, node in workflow.nodes.items()
                        if isinstance(node, AgentNode) and node.route)
        if routed:
            raise ValueError(
                f"agent node(s) {', '.join(repr(name) for name in routed)} declare a route of their "
                "own, which only a dispatched worker can take; this graph has no worker model "
                "context, so they would run on the CEO's model. Run it with a worker route.")
        return workflow
    import hashlib
    import shlex
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
        if node.route is not None:
            selected = selected.with_step_route(node.route)
        from refactory_dsh.workspace import for_sealed_step, is_sealed
        node_state = {'kind': 'agent', 'tools': list(node.tools or ()),
                      'tool_policy': node.tool_policy,
                      'workspace': getattr(node, 'workspace', None)}
        # A sealed step runs on the dsh backend with the run's model, as on the walker.
        selected = for_sealed_step(selected, name, is_sealed(node_state), node_state['tools'])
        # The node's tool policy is part of its client: a session without tools may
        # retry on any backend (`DshModelClient.retry_attempts`), so its envelope
        # below reserves every attempt.
        selected = selected.with_tool_policy(getattr(node, "tool_policy", None) or "")
        if no_effect(node.tool_policy) and node.tools:
            raise ValueError(f'worker node {name!r} declares tool_policy="{node.tool_policy}" and also declares tools, which is a contradiction')
        if node.tools and not is_sealed(node_state):
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
        payload = {'schema_version': 1, 'node_id': name, 'role': node.role.value,
                                    'prompt': node.prompt_template, 'reads': sorted(node.reads),
                                    'writes': writes,
                                    'produces': list((declarations or {}).get(name, {}).get('produces', [])),
                                    'post_checks': [check.model_dump() for check in node.post_checks],
                                    'model_context': selected_context}
        if node.system_prompt:
            payload['system_prompt'] = node.system_prompt
        if getattr(node, 'tool_policy', None):
            payload['tool_policy'] = node.tool_policy
        if getattr(node, 'keeps_tool_markup', None):
            payload['keeps_tool_markup'] = True
        if node.tools:
            payload['tools'] = list(node.tools)
        if is_optional:
            payload['optional'] = True
        optional_reads = (declarations or {}).get(name, {}).get('optional_reads')
        if optional_reads:
            # Reads it runs without when absent (`workspace.optional_reads`).
            payload['optional_reads'] = list(optional_reads)
        retry = getattr(node, 'retry', None)
        if retry and int(retry.get('times', 0)) <= 0:
            retry = None
        if retry:
            # The worker runs the node's retry rule (`node_retry.deliver`), inside the
            # envelope reserved for all its attempts below.
            payload['retry'] = retry
        if getattr(node, 'missing_on_crash', None):
            # Spent retries leave the step missing, not failed (`worker_node.main`).
            payload['missing_on_crash'] = True
        if getattr(node, 'falls_back', None):
            # A genuine failure leaves it missing for its fallback gate (`worker_node.main`).
            payload['falls_back'] = True
        if node.workspace:
            # Only when declared, so the spec of a node without one -- named by its
            # hash -- is the bytes it was before workspaces.
            payload['workspace'] = dict(node.workspace)
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
        attempts = selected.retry_attempts()
        per_completion = int(selected_context.get('timeout_s') or DEFAULT_TIMEOUT_S)
        # A session that reports activity has no wall unless its node declares one: it ends
        # on inactivity, or at the harness backstop at the latest (`model.DEFAULT_BACKSTOP_S`),
        # so the dispatch outlasts the backstop, not a per-completion wall.
        envelope_s = (selected.backstop_limit() + NODE_DISPATCH_OVERHEAD_S if selected.observes_activity()
                      else node_session_budget_s(attempts, per_completion))
        # A step with no tools that writes out tool calls is asked once more, inside this
        # same dispatch (`node_retry.deliver`): one more session, under the wall of the
        # attempt it replaces, with the startup and shutdown time any session has.
        from refactory_dsh.node_retry import node_allowance_s, refuses_tool_markup
        markup = refuses_tool_markup(node.tool_policy, getattr(node, 'keeps_tool_markup', None))
        if retry:
            # Every attempt's session -- the first under the node's bound, each retry
            # longer by its factor -- plus the one dispatch overhead,
            # so the retry runs inside the command the spine started.
            if node.timeout is not None:
                command_timeout_s = (node_allowance_s(node.timeout, retry, markup_retry=markup)
                                     + (1 + markup) * NODE_DISPATCH_OVERHEAD_S)
                if 'crash' in retry['on']:
                    # A crashed session's retry is a new dispatch of its own: each one
                    # gets the startup and shutdown time the first one has.
                    command_timeout_s += int(retry['times']) * NODE_DISPATCH_OVERHEAD_S
            elif 'timeout' in retry['on']:
                raise ValueError(f'worker node {name!r} declares a retry but no timeout')
            else:
                # A crash retry with no node bound runs each session under the route's
                # envelope, so the dispatch reserves one per attempt.
                command_timeout_s = (1 + int(retry['times']) + markup) * envelope_s
        elif node.timeout is not None:
            # The session inside stops at the node's own bound; the dispatch keeps
            # the overhead a derived envelope has, so the session ends first.
            command_timeout_s = (node_allowance_s(node.timeout, None, markup_retry=markup)
                                 + (1 + markup) * NODE_DISPATCH_OVERHEAD_S)
        else:
            command_timeout_s = (1 + markup) * envelope_s
        if node.workspace:
            # The workspace is prepared before the session starts -- a clone, its
            # installs, generate and setup -- so the dispatch outlasts the session by
            # that much more per attempt, or the spine stops a retry still preparing.
            from refactory_dsh.workspace import preparation_budget_s
            command_timeout_s += (1 + (int(retry["times"]) if retry else 0) + markup) * preparation_budget_s(
                node.workspace, workflow.knob_values.get('check_timeout_s'))
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
                   provider_config: dict[str, Any] | None = None, *,
                   session_store: Path | None = None) -> None:
    rows = [{'insert': [{'id': 'refactory-worker-observer',
                        'name': str(Path(__file__).with_name('worker_observer.mjs'))}]}]
    rows.extend(permission_patch(route))
    # The coordinator's CEO and its in-process subagents are DSH sessions too.
    rows.extend(session_store_rows(session_store))
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
        # The context alone decides: a client folds `SESSIONS_ENV` into the context when it
        # is created, and a context without the field means ``run``. (A node command's
        # worker, rebuilt from its spec, falls back to the variable instead: see models.md.)
        observer_patch(self.patch, route, provider_config=context.get('provider_config'),
                       session_store=session_store(self.path, context.get('dsh_sessions') or 'run'))
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
