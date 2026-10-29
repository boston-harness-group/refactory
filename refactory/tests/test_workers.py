from __future__ import annotations
import json
from pathlib import Path
import subprocess
import threading

import pytest

from refactory_dsh import workers
from refactory_dsh.budgets import BudgetExhausted, ResourceLimits, RunBudget
from refactory_dsh.model import DshModelClient


def test_route_precedence_is_per_node_and_does_not_mutate_parent():
    client = DshModelClient(provider='p', model='base',
                            role_routes={'builder': {'model': 'build'}},
                            node_routes={'one': {'model': 'specific'}, 'two': {'worker_backend': 'codex', 'model': 'codex-model'}})
    assert client.for_node('one', 'builder').route()['model'] == 'specific'
    assert client.for_node('other', 'builder').route()['model'] == 'build'
    assert client.for_node('two', 'builder').route() == {'worker_backend': 'codex', 'provider': '', 'model': 'codex-model', 'effort': '', 'permission_mode': 'skip'}
    assert client.route()['model'] == 'base'
    assert DshModelClient(**client.context()).context() == client.context()


@pytest.mark.parametrize('route', [{'worker_backend': 'codex', 'provider': 'openai'},
                                   {'worker_backend': 'codex', 'effort': 'high'},
                                   {'worker_backend': 'codex', 'permission_mode': 'invalid'},
                                   {'worker_backend': 'invalid'}])
def test_unsupported_routes_fail_before_launch(route):
    with pytest.raises(ValueError):
        DshModelClient(**route)


def test_receipt_preserves_unavailable_usage_and_request_is_not_actual(monkeypatch, tmp_path):
    def complete(self, cmd, run_dir, stop_at=None, budget_s=None, task=None, target_workdir=None):
        assert '--patch' in cmd
        assert self._environment()['REFACTORY_WORKER_RECEIPT']
        return subprocess.CompletedProcess(cmd, 0, 'done', '')
    monkeypatch.setattr(DshModelClient, '_run_completion', complete)
    assert DshModelClient(provider='p', model='m').for_node('build').complete('task', workdir=tmp_path) == 'done'
    receipt = json.loads(next((tmp_path / '.refactory/workers').glob('*.json')).read_text())
    assert receipt['state'] == 'completed'
    assert receipt['node_id'] == 'build'
    assert receipt['requested_route']['model'] == 'm'
    assert receipt['actual_route'] is None
    assert receipt['usage'] == {'status': 'unavailable', 'tokens': None}
    assert len((tmp_path / '.refactory/events.jsonl').read_text().splitlines()) == 2


def test_external_token_budget_fails_without_launch(tmp_path):
    budget = RunBudget.open(tmp_path, ResourceLimits(max_tokens=100))
    with pytest.raises(BudgetExhausted, match='telemetry'):
        DshModelClient(worker_backend='codex').with_budget(budget).complete('task', workdir=tmp_path)
    assert not (tmp_path / '.refactory/workers').exists()


def test_measured_nested_usage_is_counted_once(monkeypatch, tmp_path):
    budget = RunBudget.open(tmp_path, ResourceLimits(max_tokens=100))
    def complete(self, cmd, run_dir, stop_at=None, budget_s=None, task=None, target_workdir=None):
        receipt = json.loads(self._worker_receipt.read_text())
        receipt['usage'] = {'status': 'measured', 'tokens': 80, 'calls': [{'tokens': 30}, {'tokens': 50}]}
        self._worker_receipt.write_text(json.dumps(receipt))
        self._checkpoint()
        self._checkpoint()
        return subprocess.CompletedProcess(cmd, 0, 'done', '')
    monkeypatch.setattr(DshModelClient, '_run_completion', complete)
    DshModelClient().with_budget(budget).complete('task', workdir=tmp_path)
    assert budget.snapshot()['known_tokens'] == 80


def test_unknown_usage_cannot_complete_token_limited_run(monkeypatch, tmp_path):
    budget = RunBudget.open(tmp_path, ResourceLimits(max_tokens=100))
    monkeypatch.setattr(DshModelClient, '_run_completion', lambda self, cmd, run_dir, stop_at=None, budget_s=None, task=None, target_workdir=None: subprocess.CompletedProcess(cmd, 0, 'done', ''))
    with pytest.raises(BudgetExhausted, match='unavailable'):
        DshModelClient().with_budget(budget).complete('task', workdir=tmp_path)
    assert json.loads(next((tmp_path / '.refactory/workers').glob('*.json')).read_text())['state'] == 'budget_exhausted'


def test_lowered_bridge_preserves_edges_and_declared_outputs(tmp_path):
    from refactory_dsh.ir import AgentNode, AgentRole, Workflow
    graph = Workflow(name='tiny', nodes={'build': AgentNode(id='build', role=AgentRole.BUILDER,
                prompt_template='write result.txt', writes={'result.txt'})}, edges=[], start_node='build')
    context = DshModelClient(worker_backend='codex', model='chosen').context()
    lowered = workers.lower_workflow(graph, tmp_path, context)
    assert lowered.edges == graph.edges
    assert lowered.nodes['build'].writes == {'result.txt'}
    assert 'refactory_dsh.worker_node' in lowered.nodes['build'].dispatch_command
    assert graph.nodes['build'].prompt_template == 'write result.txt'
    spec = json.loads(next((tmp_path / '.refactory/worker-specs').glob('*.json')).read_text())
    assert spec['model_context']['worker_backend'] == 'codex'
    assert spec['writes'] == ['result.txt']
    # The worker opens its session with the node's role, as the walker does.
    assert spec['role'] == AgentRole.BUILDER.value


def test_lowered_node_timeout_covers_the_declared_completion_budget(tmp_path):
    from refactory_dsh.ir import AgentNode, AgentRole, Workflow
    from refactory_dsh.model import DEFAULT_TIMEOUT_S

    def graph(timeout=None):
        return Workflow(name='tiny', nodes={'build': AgentNode(id='build', role=AgentRole.BUILDER,
                    prompt_template='write result.txt', writes={'result.txt'}, timeout=timeout)},
                        edges=[], start_node='build')

    # An unset node timeout derives from the route's declared budget, so the
    # spine's node envelope outlasts the retries the model client may take
    # instead of dying at the host shell's default clamp (#289).
    dsh = workers.lower_workflow(graph(), tmp_path / 'dsh', DshModelClient().context())
    attempts = DshModelClient().context()['attempts']
    assert dsh.nodes['build'].command_timeout_s == attempts * DEFAULT_TIMEOUT_S + 600

    # A native backend reports a single attempt, so its envelope is one
    # completion plus overhead.
    codex = workers.lower_workflow(graph(), tmp_path / 'codex',
                                   DshModelClient(worker_backend='codex', model='m').context())
    assert codex.nodes['build'].command_timeout_s == DEFAULT_TIMEOUT_S + 600

    # An explicit node timeout is the author's own bound on the model session: the
    # dispatched session stops there, and the command keeps the dispatch overhead
    # beyond it, as a derived envelope does, so the session ends first (#307).
    explicit = workers.lower_workflow(graph(timeout=123), tmp_path / 'explicit', DshModelClient().context())
    assert explicit.nodes['build'].command_timeout_s == 123 + 600
    spec_path = explicit.nodes['build'].dispatch_command.rsplit(' ', 1)[1].strip("'")
    assert json.loads(Path(spec_path).read_text())['model_context']['node_wall_s'] == 123


def test_graph_checks_worker_outputs_not_success_sentence(monkeypatch, tmp_path):
    from refactory_dsh.worker_node import execute
    monkeypatch.setattr(DshModelClient, 'complete', lambda *args, **kwargs: 'I succeeded')
    spec = {'schema_version': 1, 'node_id': 'build', 'reads': [], 'writes': ['one.txt', 'two.txt'],
            'prompt': 'write files', 'model_context': {}}
    with pytest.raises(ValueError, match='two.txt'):
        execute(spec, tmp_path)


def test_pinned_provider_metadata_callbacks_do_not_need_model_calls():
    root = Path(__file__).resolve().parents[2]
    if not (root / 'node_modules/@deepseek-ai/dsh-subagent-codex').exists():
        pytest.skip('optional worker packages not installed')
    subprocess.run(['node', str(Path(__file__).with_name('worker_metadata_test.mjs')), str(root)], check=True)


def test_real_patch_composition_dispatches_codex_provider_before_any_product_start(tmp_path):
    root = Path(__file__).resolve().parents[2]
    if not (root / 'node_modules/@deepseek-ai/dsh-subagent-codex').exists():
        pytest.skip('optional worker packages not installed')
    for backend in ('codex', 'claude_code'):
        workers.write_patch(tmp_path / (backend + '.json'), root, DshModelClient(worker_backend=backend).route())
    subprocess.run(['node', str(Path(__file__).with_name('worker_dispatch_test.mjs')), str(root), str(tmp_path)], check=True)


def test_flock_takes_its_lock_without_a_diagnostic_report():
    """Workers died with exit -6 mid-step: the pinned flock generated a diagnostic report
    to tell glibc from musl, and a report aborts the process when a socket is closing."""
    root = Path(__file__).resolve().parents[2]
    subprocess.run(['node', str(Path(__file__).with_name('flock_libc_test.mjs')), str(root)], check=True)


def test_observer_keeps_nested_external_usage_incomplete_without_model_calls():
    subprocess.run(['node', str(Path(__file__).with_name('worker_observer_test.mjs'))], check=True)


def test_worker_patch_carries_declared_provider_profiles(tmp_path):
    """A declared provider route must ride the patch into the nested runtime."""
    gateway = {"apiKeyEnv": "LITELLM_API_KEY", "api": "anthropic-messages",
               "baseURL": "https://gateway.example/v1",
               "models": [{"id": "gateway-model", "name": "gateway-model"}]}
    route = DshModelClient(provider='gateway', model='gateway-model',
                           provider_config={'gateway': gateway}).route()
    workers.write_patch(tmp_path / 'with-provider.json', tmp_path, route, provider_config={'gateway': gateway})
    rows = json.loads((tmp_path / 'with-provider.json').read_text())
    row = next(entry for entry in rows if entry.get('id') == 'llm-pi-ai')
    patched_gateway = {**gateway, 'streamIdleTimeoutMs': None}
    assert row['config'] == {'providers': {'gateway': patched_gateway}}
    # Without a declaration the patch composes exactly what it did before.
    workers.write_patch(tmp_path / 'without-provider.json', tmp_path, DshModelClient().route())
    assert not any(entry.get('id') == 'llm-pi-ai'
                   for entry in json.loads((tmp_path / 'without-provider.json').read_text()))
    # The composed real headless tree resolves the declared route, not NO_ADAPTER.
    root = Path(__file__).resolve().parents[2]
    subprocess.run(['node', str(Path(__file__).with_name('worker_provider_config_test.mjs')),
                    str(root), str(tmp_path)], check=True)


@pytest.mark.parametrize('provider_config', [{'other': {'apiKeyEnv': 'X'}},
                                             {'opencode-go': {'apiKeyEnv': 'OPENCODE_API_KEY'}}],
                         ids=['provider-undeclared', 'declared-without-header'])
def test_opencode_route_gets_a_session_header(provider_config):
    """OpenCode workers need x-opencode-session; provider_config providers lack it."""
    rows = workers.provider_rows({'worker_backend': 'dsh', 'provider': 'opencode-go'}, provider_config)
    header = rows[0]['config']['providers']['opencode-go']['headers']['x-opencode-session']
    assert header.startswith('refactory-worker-') and len(header) > len('refactory-worker-')


def test_a_declared_opencode_session_header_is_kept_exactly():
    config = {'opencode-go': {'apiKeyEnv': 'OPENCODE_API_KEY',
                              'headers': {'x-opencode-session': 'custom'}}}
    rows = workers.provider_rows({'worker_backend': 'dsh', 'provider': 'opencode-go'}, config)
    assert rows[0]['config']['providers']['opencode-go']['headers'] == {'x-opencode-session': 'custom'}


def test_other_provider_routes_get_no_opencode_entry():
    config = {'anthropic': {'apiKeyEnv': 'ANTHROPIC_API_KEY'}}
    rows = workers.provider_rows({'worker_backend': 'dsh', 'provider': 'anthropic'}, config)
    assert rows[0]['config']['providers'] == {'anthropic': {**config['anthropic'], 'streamIdleTimeoutMs': None}}


def test_opencode_session_header_uses_correct_credential_name():
    """The auto-created entry must use OPENCODE_API_KEY, matching the built-in provider."""
    route = {'worker_backend': 'dsh', 'provider': 'opencode-go'}
    rows = workers.provider_rows(route, None)
    oc = rows[0]['config']['providers']['opencode-go']
    assert oc['apiKeyEnv'] == 'OPENCODE_API_KEY'


def test_opencode_session_ids_are_unique_per_call():
    """Each provider_rows call should produce a distinct session id."""
    route = {'worker_backend': 'dsh', 'provider': 'opencode-go'}
    ids = set()
    for _ in range(10):
        rows = workers.provider_rows(route, None)
        header = rows[0]['config']['providers']['opencode-go']['headers']['x-opencode-session']
        ids.add(header)
    assert len(ids) == 10


def test_coordinator_patch_carries_declared_provider_profiles(tmp_path):
    """The coordinator's graph runtime is as nested as a node's headless one.

    Its patch omitted the declared provider routes, so a run selected to use a
    gateway provider dispatched the graph profile's own default instead — a
    project routed to a litellm gateway silently billed a DeepSeek account.
    """
    gateway = {"apiKeyEnv": "LITELLM_API_KEY", "api": "anthropic-messages",
               "baseURL": "https://gateway.example/v1",
               "models": [{"id": "gateway-model", "name": "gateway-model"}]}
    context = DshModelClient(provider='gateway', model='gateway-model',
                             provider_config={'gateway': gateway}).context()
    telemetry = workers.GraphTelemetry(tmp_path, context, None)
    rows = json.loads(telemetry.patch.read_text())
    row = next(entry for entry in rows if entry.get('id') == 'llm-pi-ai')
    patched_gateway = {**gateway, 'streamIdleTimeoutMs': None}
    assert row['config'] == {'providers': {'gateway': patched_gateway}}
    # Without a declaration the coordinator patch composes exactly what it did.
    telemetry = workers.GraphTelemetry(tmp_path, DshModelClient().context(), None)
    rows = json.loads(telemetry.patch.read_text())
    assert not any(entry.get('id') == 'llm-pi-ai' for entry in rows)


def test_provider_rows_disables_stream_idle_timeout_by_default():
    cfg = {"apiKeyEnv": "KEY", "baseURL": "https://example.com"}
    rows = workers.provider_rows({'worker_backend': 'dsh'}, {'myProvider': cfg})
    providers = rows[0]['config']['providers']
    assert providers['myProvider']['streamIdleTimeoutMs'] is None
    assert providers['myProvider']['apiKeyEnv'] == 'KEY'


def test_provider_rows_preserves_explicit_stream_idle_timeout():
    cfg = {"apiKeyEnv": "KEY", "streamIdleTimeoutMs": 600000}
    rows = workers.provider_rows({'worker_backend': 'dsh'}, {'myProvider': cfg})
    assert rows[0]['config']['providers']['myProvider']['streamIdleTimeoutMs'] == 600000


def test_provider_rows_does_not_mutate_caller_config():
    cfg = {"apiKeyEnv": "KEY"}
    original = dict(cfg)
    workers.provider_rows({'worker_backend': 'dsh'}, {'myProvider': cfg})
    assert cfg == original


def test_partial_nested_usage_retains_known_total_but_refuses_token_work(tmp_path):
    budget = RunBudget.open(tmp_path, ResourceLimits(max_tokens=200))
    receipt = {'worker_id': 'dsh-one', 'usage': {'status': 'unavailable', 'tokens': None,
               'known_tokens': 24, 'incomplete_nested_usage': True}}
    workers.account_usage(budget, receipt)
    workers.account_usage(budget, receipt)
    assert budget.snapshot()['known_tokens'] == 24
    assert budget.snapshot()['tokens'] is None
    with pytest.raises(BudgetExhausted, match='unavailable'):
        budget.checkpoint()


def test_external_worker_must_deliver_the_only_output(monkeypatch, tmp_path):
    from refactory_dsh.worker_node import execute
    monkeypatch.setattr(DshModelClient, 'complete', lambda *args, **kwargs: 'I succeeded')
    spec = {'schema_version': 1, 'node_id': 'build', 'reads': [], 'writes': ['only.txt'],
            'prompt': 'write file', 'model_context': {'worker_backend': 'codex'}}
    with pytest.raises(ValueError, match='only.txt'):
        execute(spec, tmp_path)
    assert not (tmp_path / 'only.txt').exists()


def test_worker_specs_keep_old_invocation_route_immutable(tmp_path):
    from refactory_dsh.ir import AgentNode, AgentRole, Workflow
    graph = Workflow(name='tiny', nodes={'build': AgentNode(id='build', role=AgentRole.BUILDER,
                prompt_template='write result.txt', writes={'result.txt'})}, edges=[], start_node='build')
    first = workers.lower_workflow(graph, tmp_path, DshModelClient(model='first').context())
    second = workers.lower_workflow(graph, tmp_path, DshModelClient(model='second').context())
    assert first.nodes['build'].dispatch_command != second.nodes['build'].dispatch_command
    specs = [json.loads(path.read_text()) for path in (tmp_path/'.refactory/worker-specs').glob('*.json')]
    assert {spec['model_context']['model'] for spec in specs} == {'first','second'}


def test_worker_bridge_renders_knobs_and_preserves_agent_facts(monkeypatch, tmp_path):
    from refactory_dsh.worker_node import execute
    (tmp_path / '.refactory').mkdir()
    (tmp_path / '.refactory/declarations.json').write_text(json.dumps({'build': {'produces': ['answer']}}))
    monkeypatch.setenv('SRF_KNOBS', json.dumps({'number': 42}))
    seen = []
    def complete(self, prompt, **kwargs):
        seen.append(prompt)
        return '42'
    monkeypatch.setattr(DshModelClient, 'complete', complete)
    execute({'schema_version': 1, 'node_id': 'build', 'reads': [], 'writes': ['only.txt'],
             'prompt': 'Return {{number}}', 'model_context': {}}, tmp_path)
    assert len(seen) == 1 and seen[0].endswith('Return 42')
    assert str(tmp_path / 'only.txt') in seen[0]
    assert json.loads((tmp_path/'facts.json').read_text())['answer'] == '42'


def test_worker_uses_prompt_file_over_spec_prompt(monkeypatch, tmp_path):
    """A worker reads its prompt from the node's prompt_file when the manifest names one."""
    from refactory_dsh.worker_node import execute
    (tmp_path / '.refactory').mkdir()
    (tmp_path / '.refactory/declarations.json').write_text(json.dumps({'build': {'produces': []}}))
    seen = []
    def complete(self, prompt, **kwargs):
        seen.append(prompt)
        return 'done'
    monkeypatch.setattr(DshModelClient, 'complete', complete)
    prompt_dir = tmp_path / '.refactory' / 'prompts'
    prompt_dir.mkdir(parents=True)
    prompt_file = prompt_dir / 'build.md'
    prompt_file.write_text('EDITED prompt text', encoding='utf-8')
    manifest = json.dumps({'id': 'build', 'type': 'AgentNode', 'command': 'echo', 'reads': [], 'writes': [], 'prompt_file': str(prompt_file)})
    monkeypatch.setenv('DSH_WORKFLOW_NODE', manifest)
    execute({'schema_version': 1, 'node_id': 'build', 'reads': [], 'writes': [],
             'prompt': 'ORIGINAL prompt text', 'model_context': {}}, tmp_path)
    assert len(seen) == 1
    assert 'EDITED prompt text' in seen[0]
    assert 'ORIGINAL prompt text' not in seen[0]


def test_worker_falls_back_to_spec_prompt_without_prompt_file(monkeypatch, tmp_path):
    """A worker uses the spec's prompt when the manifest names no prompt_file."""
    from refactory_dsh.worker_node import execute
    (tmp_path / '.refactory').mkdir()
    (tmp_path / '.refactory/declarations.json').write_text(json.dumps({'build': {'produces': []}}))
    seen = []
    def complete(self, prompt, **kwargs):
        seen.append(prompt)
        return 'done'
    monkeypatch.setattr(DshModelClient, 'complete', complete)
    manifest = json.dumps({'id': 'build', 'type': 'AgentNode', 'command': 'echo', 'reads': [], 'writes': []})
    monkeypatch.setenv('DSH_WORKFLOW_NODE', manifest)
    execute({'schema_version': 1, 'node_id': 'build', 'reads': [], 'writes': [],
             'prompt': 'ORIGINAL prompt text', 'model_context': {}}, tmp_path)
    assert len(seen) == 1
    assert 'ORIGINAL prompt text' in seen[0]


def test_completion_timeout_stops_owned_runtime_and_records_reason(tmp_path):
    import os
    import sys
    script = tmp_path / 'owned-runtime'
    script.write_text(f'#!{sys.executable}\nimport signal,time\nfrom pathlib import Path\ndef stop(*args):\n Path("stopped").write_text("yes")\n raise SystemExit(0)\nsignal.signal(signal.SIGTERM,stop)\nPath("started").write_text("yes")\ntime.sleep(30)\n')
    script.chmod(0o700)
    with pytest.raises(RuntimeError, match='timed out'):
        DshModelClient(dsh_cli=str(script), timeout_s=1, attempts=1).complete('ignored', workdir=tmp_path)
    assert (tmp_path/'stopped').read_text() == 'yes'
    receipt = json.loads(next((tmp_path/'.refactory/workers').glob('*.json')).read_text())
    assert receipt['state'] == 'timed_out'


def test_cancellation_receipt_does_not_cancel_another_client(monkeypatch, tmp_path):
    from concurrent.futures import CancelledError
    event = threading.Event()
    first = DshModelClient().with_cancellation(event)
    second = DshModelClient()
    event.set()
    monkeypatch.setattr(DshModelClient, '_run_completion', lambda self, cmd, run_dir, stop_at=None, budget_s=None, task=None, target_workdir=None: subprocess.CompletedProcess(cmd, 0, 'done', ''))
    with pytest.raises(CancelledError):
        first.complete('first', workdir=tmp_path/'first')
    assert second.complete('second', workdir=tmp_path/'second') == 'done'
    first_receipt = json.loads(next((tmp_path/'first/.refactory/workers').glob('*.json')).read_text())
    second_receipt = json.loads(next((tmp_path/'second/.refactory/workers').glob('*.json')).read_text())
    assert first_receipt['state'] == 'cancelled'
    assert second_receipt['state'] == 'completed'
    assert first_receipt['worker_id'] != second_receipt['worker_id']


def test_deterministic_nested_environment_keeps_route_and_limits(monkeypatch, tmp_path):
    budget = RunBudget.open(tmp_path, ResourceLimits(max_tokens=200))
    client = DshModelClient(provider='p', model='m', effort='high', timeout_s=10,
                            role_routes={'builder': {'worker_backend': 'codex', 'model': 'code'}}).with_budget(budget)
    for key,value in workers.environment(client.context()).items():
        monkeypatch.setenv(key,value)
    nested = DshModelClient()
    assert nested.context() == client.context()
    assert nested.for_node('build','builder').route()['worker_backend'] == 'codex'


@pytest.mark.parametrize('python_command', ['python3', 'declared'])
def test_worker_shell_keeps_selected_python_through_actual_dsh_scrub(monkeypatch, tmp_path, python_command):
    import shutil
    import sys
    root = Path(__file__).resolve().parents[2]
    modules = list((root / 'node_modules/.pnpm').glob('@deepseek-ai+dsh-subprocess@*/node_modules/@deepseek-ai/dsh-subprocess/lib/index.js'))
    if not modules:
        pytest.skip('pinned DSH subprocess package not installed')
    node = shutil.which('node')
    import os
    # The interpreter is selected by a path with `..` in it, as `../.venv/bin/python` gives
    # (dm-269): the client normalises it, and the scrub keeps the normalised directory.
    bindir = str(Path(os.path.abspath(sys.executable)).parent.resolve())
    selected = os.path.join(bindir, '..', os.path.basename(bindir))
    assert '..' in Path(selected).parts and os.path.samefile(selected, bindir)
    # A narrow PATH still has to reach the shell the worker spawns and the selected
    # interpreter. `/usr/bin:/bin` is a distro assumption NixOS does not satisfy, so
    # pin the two directories the test actually needs instead.
    bash_dir = os.path.dirname(shutil.which('bash') or '')
    monkeypatch.setenv('PATH', os.pathsep.join(part for part in (bindir, bash_dir) if part))
    monkeypatch.delenv('PYTHONPATH', raising=False)
    monkeypatch.setenv('REFACTORY_TEST_SECRET', 'not-a-real-secret')
    client = DshModelClient(python_bin=selected)
    assert client.context()['python_bin'] == bindir
    from refactory_dsh.sandbox import scrub
    recreated = DshModelClient(**client.context())
    # Context survives even the narrower generated-package environment scrub.
    with monkeypatch.context() as isolated:
        isolated.setenv('VENV_BIN', '/not/the/selected/environment')
        assert DshModelClient(**recreated.context()).context()['python_bin'] == bindir
    expression = ('import json,os,sys,refactory_dsh; print(json.dumps({"executable":sys.executable,'
                  '"module":refactory_dsh.__file__,"venv":os.environ.get("VENV_BIN")}))')
    import shlex
    from refactory_dsh.agent_delivery import begin
    delivery = begin(tmp_path, 'runtime-check', 'Import the installed API.', [], model_context=client.context())
    executable = delivery['execution']['python'] if python_command == 'declared' else python_command
    command = shlex.quote(executable) + ' -c ' + shlex.quote(expression)
    script = (f'import {{scrubbedParentEnv}} from {json.dumps(modules[0].as_uri())};'
              'import {spawnSync} from "node:child_process";'
              'const env=scrubbedParentEnv(); if("REFACTORY_TEST_SECRET" in env)throw Error("secret leaked");'
              f'const p=spawnSync("bash",["-c",{json.dumps(command)}],{{env,encoding:"utf8"}});'
              'process.stdout.write(p.stdout);process.stderr.write(p.stderr);process.exit(p.status??1);')
    result = subprocess.run([node, '--input-type=module', '-e', script], cwd=tmp_path,
                            env=client._environment(), capture_output=True, text=True, check=True)
    actual = json.loads(result.stdout)
    assert Path(actual['executable']).parent == Path(bindir)
    assert actual['venv'] == bindir, "the scrub did not forward the selected Python, normalised"
    assert Path(actual['module']).resolve() == root / 'refactory/refactory_dsh/__init__.py'
    assert 'REFACTORY_TEST_SECRET' not in scrub(client._environment())
    # After the environment fix, PYTHONPATH is absent from the scrubbed env.
    assert 'PYTHONPATH' not in scrub(client._environment())


@pytest.mark.parametrize('python_command', ['python3', 'declared'])
def test_agent_tool_subprocess_imports_target_package_not_refactory_source(monkeypatch, tmp_path, python_command):
    """A real target venv's packages are importable and refactory_dsh from the
    driver's source tree is not on the import path."""
    import os
    import shlex
    import shutil
    import sys
    root = Path(__file__).resolve().parents[2]
    modules = list((root / 'node_modules/.pnpm').glob(
        '@deepseek-ai+dsh-subprocess@*/node_modules/@deepseek-ai/dsh-subprocess/lib/index.js'))
    if not modules:
        pytest.skip('pinned DSH subprocess package not installed')
    node = shutil.which('node')
    # Create a real target venv with a tiny sentinel package.
    target_root = tmp_path / 'target'
    target_root.mkdir()
    target_venv = target_root / '.venv'
    subprocess.run([sys.executable, '-m', 'venv', str(target_venv)], check=True)
    target_bindir = str(target_venv / 'bin')
    # Write a small sentinel module into the target venv's site-packages.
    site_dir = list(target_venv.glob('lib/python*/site-packages'))[0]
    sentinel = site_dir / 'target_sentinel'
    sentinel.mkdir()
    (sentinel / '__init__.py').write_text('PRESENT = True\n')
    # Build the worker environment simulating what model._environment() produces
    # when a worktree exists: PATH leads with the target's .venv/bin, no PYTHONPATH.
    bash_dir = os.path.dirname(shutil.which('bash') or '')
    monkeypatch.setenv('PATH', os.pathsep.join(part for part in (target_bindir, bash_dir) if part))
    monkeypatch.delenv('PYTHONPATH', raising=False)
    monkeypatch.setenv('REFACTORY_TEST_SECRET', 'not-a-real-secret')
    client = DshModelClient(python_bin=target_bindir)
    expression = (
        'import json,sys;'
        'import target_sentinel;'
        'rdsh_ok=False;rdsh_file="";'
        'exec("try:\\n import refactory_dsh\\n rdsh_ok=True;rdsh_file=refactory_dsh.__file__\\nexcept ImportError:\\n pass");'
        'print(json.dumps({"sentinel":target_sentinel.PRESENT,"rdsh_ok":rdsh_ok,"rdsh_file":rdsh_file}))'
    )
    from refactory_dsh.agent_delivery import begin
    delivery = begin(tmp_path, 'runtime-check', 'Import the target package.', [], model_context=client.context())
    executable = delivery['execution']['python'] if python_command == 'declared' else 'python3'
    command = shlex.quote(executable) + ' -c ' + shlex.quote(expression)
    from refactory_dsh.sandbox import scrub
    script = (f'import {{scrubbedParentEnv}} from {json.dumps(modules[0].as_uri())};'
              'import {spawnSync} from "node:child_process";'
              'const env=scrubbedParentEnv(); if("REFACTORY_TEST_SECRET" in env)throw Error("secret leaked");'
              f'const p=spawnSync("bash",["-c",{json.dumps(command)}],{{env,encoding:"utf8"}});'
              'process.stdout.write(p.stdout);process.stderr.write(p.stderr);process.exit(p.status??1);')
    result = subprocess.run([node, '--input-type=module', '-e', script], cwd=tmp_path,
                            env=client._environment(), capture_output=True, text=True, check=True)
    actual = json.loads(result.stdout)
    # 1. The target package imports successfully.
    assert actual['sentinel'] is True, "target_sentinel should be importable"
    # 2. refactory_dsh either cannot be imported (not in target venv) or its __file__
    #    does not point to refactory's source tree.
    if actual['rdsh_ok']:
        assert str(root) not in actual['rdsh_file'], (
            f"refactory_dsh should NOT resolve to the driver's source tree, got {actual['rdsh_file']}")
    # 3. PYTHONPATH is not in the scrubbed environment.
    assert 'PYTHONPATH' not in scrub(client._environment())


def test_self_hosted_refactory_target_resolves_to_target_copy(monkeypatch, tmp_path):
    """For a target that IS a refactory checkout, the agent's python imports
    the target's refactory_dsh, not the driver's."""
    import os
    import shlex
    import shutil
    import sys
    root = Path(__file__).resolve().parents[2]
    modules = list((root / 'node_modules/.pnpm').glob(
        '@deepseek-ai+dsh-subprocess@*/node_modules/@deepseek-ai/dsh-subprocess/lib/index.js'))
    if not modules:
        pytest.skip('pinned DSH subprocess package not installed')
    node = shutil.which('node')
    # Create a real venv and a fake refactory_dsh in the target.
    target_root = tmp_path / 'target'
    target_root.mkdir()
    target_venv = target_root / '.venv'
    subprocess.run([sys.executable, '-m', 'venv', str(target_venv)], check=True)
    target_bindir = str(target_venv / 'bin')
    # Install a sentinel refactory_dsh in the target venv.
    site_dir = list(target_venv.glob('lib/python*/site-packages'))[0]
    pkg_dir = site_dir / 'refactory_dsh'
    pkg_dir.mkdir()
    (pkg_dir / '__init__.py').write_text('SENTINEL = "target_copy"\n')
    # Build the worker environment.
    bash_dir = os.path.dirname(shutil.which('bash') or '')
    monkeypatch.setenv('PATH', os.pathsep.join(part for part in (target_bindir, bash_dir) if part))
    monkeypatch.delenv('PYTHONPATH', raising=False)
    client = DshModelClient(python_bin=target_bindir)
    expression = 'import json,refactory_dsh; print(json.dumps({"file": refactory_dsh.__file__}))'
    from refactory_dsh.agent_delivery import begin
    delivery = begin(tmp_path, 'runtime-check', 'Import refactory_dsh.', [], model_context=client.context())
    executable = delivery['execution']['python']
    command = shlex.quote(executable) + ' -c ' + shlex.quote(expression)
    script = (f'import {{scrubbedParentEnv}} from {json.dumps(modules[0].as_uri())};'
              'import {spawnSync} from "node:child_process";'
              'const env=scrubbedParentEnv();'
              f'const p=spawnSync("bash",["-c",{json.dumps(command)}],{{env,encoding:"utf8"}});'
              'process.stdout.write(p.stdout);process.stderr.write(p.stderr);process.exit(p.status??1);')
    result = subprocess.run([node, '--input-type=module', '-e', script], cwd=tmp_path,
                            env=client._environment(), capture_output=True, text=True, check=True)
    actual = json.loads(result.stdout)
    # __file__ must be under the target, not under the driver's source tree.
    assert str(target_root) in actual['file'], (
        f"refactory_dsh.__file__ should be under {target_root}, got {actual['file']}")
    assert str(root / 'refactory') not in actual['file'], (
        f"refactory_dsh.__file__ must NOT be from the driver's source, got {actual['file']}")


def test_worker_node_dispatch_uses_absolute_interpreter(tmp_path):
    """lower_workflow() dispatches worker_node with sys.executable, not bare python."""
    import sys
    from refactory_dsh.ir import AgentNode, AgentRole, Workflow
    wf = Workflow(
        name='test',
        nodes={'test_node': AgentNode(
            id='test_node', prompt_template='Do nothing.',
            reads=set(), writes=set(), role=AgentRole.BUILDER,
        )},
        edges=[],
        start_node='test_node',
    )
    client = DshModelClient()
    from refactory_dsh.workers import lower_workflow
    lowered = lower_workflow(wf, tmp_path, client.context())
    dispatch = lowered.nodes['test_node'].dispatch_command
    # Must start with an absolute path, not bare 'python'
    assert dispatch.startswith(sys.executable), (
        f"dispatch_command should start with sys.executable ({sys.executable}), got: {dispatch}")
    assert '/python' in dispatch.split()[0] or dispatch.split()[0] == sys.executable
    # Verify the absolute path can actually run refactory_dsh
    result = subprocess.run(
        [sys.executable, '-c', 'import refactory_dsh; print(refactory_dsh.__file__)'],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_create_mode_check_still_imports_refactory_dsh(tmp_path):
    """Create-mode validation through run_sandboxed still works after the change.

    run_sandboxed uses sys.executable (absolute path) and sets PYTHONPATH to
    refactory's source, so refactory_dsh is always importable there. This test
    passes both before and after the change; it guards against regression.
    """
    import os
    from refactory_dsh import sandbox as _sandbox
    code = 'import json, refactory_dsh; print(json.dumps({"file": refactory_dsh.__file__}))'
    result = _sandbox.run_sandboxed(code, workdir=tmp_path, env=dict(os.environ))
    assert result.ok, f"run_sandboxed should succeed: {result.stderr}"
    assert 'refactory_dsh' in result.payload.get('file', ''), (
        f"refactory_dsh should be importable in run_sandboxed: {result.payload}")


def test_refactory_env_puts_refactorys_interpreter_first_and_keeps_the_scrubbed_rest(monkeypatch):
    """The environment a check of a refactory artifact runs with: refactory's interpreter
    first however the shell's PATH reads, and everything scrub keeps (HOME, LANG) kept."""
    import os
    from refactory_dsh import sandbox as _sandbox
    from refactory_dsh.paths import venv_bin
    bindir = str(venv_bin())
    monkeypatch.setenv('PATH', os.pathsep.join(p for p in os.environ.get('PATH', '').split(os.pathsep) if p != bindir))
    monkeypatch.setenv('HOME', '/home/someone')
    monkeypatch.setenv('SOME_API_KEY', 'secret')
    env = _sandbox.refactory_env()
    assert env['PATH'].split(os.pathsep)[0] == bindir
    assert env['PATH'].split(os.pathsep).count(bindir) == 1
    assert env['HOME'] == '/home/someone'
    assert 'SOME_API_KEY' not in env


def test_coordinator_and_node_account_separately_without_double_count(tmp_path):
    budget = RunBudget.open(tmp_path, ResourceLimits(max_tokens=200))
    telemetry = workers.GraphTelemetry(tmp_path,DshModelClient().context(),budget)
    telemetry.checkpoint(final=True)
    assert budget.snapshot()['workers'] == {}
    record=json.loads(telemetry.path.read_text())
    record.update(actual_route={'model':'m'},usage={'status':'measured','tokens':70})
    telemetry.path.write_text(json.dumps(record))
    budget.record_usage('node-one',80,status='measured')
    telemetry.checkpoint()
    telemetry.checkpoint(final=True)
    telemetry.finish('completed')
    assert budget.snapshot()['known_tokens'] == 150


def test_coordinator_unknown_usage_fails_a_token_limited_run(tmp_path):
    budget = RunBudget.open(tmp_path, ResourceLimits(max_tokens=200))
    telemetry=workers.GraphTelemetry(tmp_path,DshModelClient().context(),budget)
    record=json.loads(telemetry.path.read_text())
    record['actual_route']={'model':'m'}
    telemetry.path.write_text(json.dumps(record))
    telemetry.checkpoint() # In flight; usage may not have arrived yet.
    with pytest.raises(BudgetExhausted,match='unavailable'):
        telemetry.checkpoint(final=True)


def test_coordinator_token_limit_stops_only_its_owned_process(tmp_path):
    import sys
    script=tmp_path/'graph-runtime'
    script.write_text(f'#!{sys.executable}\nimport os,json,signal,time\nfrom pathlib import Path\ndef stop(*args):\n Path("stopped").write_text("yes")\n raise SystemExit(0)\nsignal.signal(signal.SIGTERM,stop)\np=Path(os.environ["REFACTORY_WORKER_RECEIPT"])\nr=json.loads(p.read_text())\nr.update(actual_route={{"model":"m"}},usage={{"status":"measured","tokens":50}})\np.write_text(json.dumps(r))\ntime.sleep(30)\n')
    script.chmod(0o700)
    budget=RunBudget.open(tmp_path,ResourceLimits(max_tokens=30))
    context=DshModelClient().with_budget(budget).context()
    with pytest.raises(BudgetExhausted,match='token budget'):
        workers.run_graph_command([str(script),'--profile','graph'],workdir=tmp_path,context=context,cwd=tmp_path,timeout=3)
    assert (tmp_path/'stopped').read_text() == 'yes'
    receipt=json.loads(next((tmp_path/'.refactory/workers').glob('*.json')).read_text())
    assert receipt['state'] == 'budget_exhausted'
    assert budget.snapshot()['known_tokens'] == 50


def test_graph_worker_refuses_modified_dispatch_contract(tmp_path):
    import os
    import sys
    spec_dir=tmp_path/'worker-specs'
    spec_dir.mkdir()
    spec=spec_dir/('0'*64+'.json')
    spec.write_text(json.dumps({'schema_version':1,'node_id':'tampered'}))
    result=subprocess.run([sys.executable,'-m','refactory_dsh.worker_node',str(spec)],capture_output=True,text=True)
    assert result.returncode == 1
    assert 'modified after graph preparation' in result.stderr


def test_worker_node_optional_exits_zero_on_failure(tmp_path, monkeypatch, capsys):
    """A worker spec with optional: true exits 0 with status 'missing' when execute raises."""
    import sys
    from refactory_dsh import worker_node

    spec = {
        'schema_version': 1,
        'node_id': 'verifier',
        'optional': True,
        'reads': [],
        'writes': [],
        'prompt': 'test',
        'model_context': {},
    }
    spec_path = tmp_path / 'spec.json'
    spec_path.write_text(json.dumps(spec))

    monkeypatch.setattr(sys, 'argv', ['worker_node', str(spec_path)])
    monkeypatch.chdir(tmp_path)

    def _failing_execute(s, w):
        raise RuntimeError('deliberate test error')
    monkeypatch.setattr(worker_node, 'execute', _failing_execute)

    exit_code = worker_node.main()
    assert exit_code == 0, f'optional worker must exit 0, got {exit_code}'

    captured = capsys.readouterr()
    output = json.loads(captured.out.strip())
    assert output['status'] == 'missing'
    assert output['node'] == 'verifier'

    missing_file = tmp_path / 'reviews' / 'verifier-missing.txt'
    assert missing_file.exists(), 'missing file must be written'
    assert 'RuntimeError' in missing_file.read_text()


def test_worker_node_optional_settles_pending_marker(tmp_path, monkeypatch, capsys):
    """dm-232: optional worker that fails settles the pending delivery marker."""
    import hashlib
    import sys
    from refactory_dsh import worker_node

    node_id = 'verifier'
    spec = {
        'schema_version': 1,
        'node_id': node_id,
        'optional': True,
        'reads': [],
        'writes': [],
        'prompt': 'test',
        'model_context': {},
    }
    spec_path = tmp_path / 'spec.json'
    spec_path.write_text(json.dumps(spec))

    # Create a pending delivery marker
    pending_hash = hashlib.sha256(node_id.encode()).hexdigest()
    delivery_id = 'test-delivery-001'
    pending_dir = tmp_path / '.refactory' / 'agent-delivery' / 'pending'
    pending_dir.mkdir(parents=True, exist_ok=True)
    pending_file = pending_dir / f'{pending_hash}.json'
    pending_file.write_text(json.dumps({'id': delivery_id, 'node': node_id}))

    # Create the receipt file that invalidate() will update
    receipt_dir = tmp_path / '.refactory' / 'agent-delivery' / delivery_id
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_file = receipt_dir / 'receipt.json'
    receipt_file.write_text(json.dumps({
        'node': node_id, 'status': 'pending',
        'started': '2026-01-01T00:00:00Z', 'artifacts': {},
    }))

    monkeypatch.setattr(sys, 'argv', ['worker_node', str(spec_path)])
    monkeypatch.chdir(tmp_path)

    def _failing_execute(s, w):
        raise RuntimeError('deliberate test error')
    monkeypatch.setattr(worker_node, 'execute', _failing_execute)

    exit_code = worker_node.main()
    assert exit_code == 0, f'optional worker must exit 0, got {exit_code}'

    # The pending marker should be invalidated (removed)
    assert not pending_file.exists(), 'pending marker should be invalidated after optional branch recorded missing'

    # The receipt should be updated to superseded
    updated = json.loads(receipt_file.read_text())
    assert updated['status'] == 'superseded', f'receipt status should be superseded, got {updated["status"]}'


def test_run_node_optional_settles_pending_marker(tmp_path, monkeypatch):
    """dm-232: run_node for optional op settles the pending delivery marker."""
    import hashlib
    from refactory_dsh import run as run_mod

    workdir = tmp_path / 'run'
    workdir.mkdir()
    refactory_dir = workdir / '.refactory'
    refactory_dir.mkdir()

    node_id = 'verifier'
    (refactory_dir / 'declarations.json').write_text(json.dumps({
        node_id: {
            'reads': [],
            'writes': ['reviews/verification.md'],
            'produces': ['verification'],
            'optional': True,
        }
    }))
    (workdir / 'facts.json').write_text('{}')

    # Create a pending delivery marker
    pending_hash = hashlib.sha256(node_id.encode()).hexdigest()
    delivery_id = 'test-delivery-002'
    pending_dir = refactory_dir / 'agent-delivery' / 'pending'
    pending_dir.mkdir(parents=True, exist_ok=True)
    pending_file = pending_dir / f'{pending_hash}.json'
    pending_file.write_text(json.dumps({'id': delivery_id, 'node': node_id}))

    # Create receipt
    receipt_dir = refactory_dir / 'agent-delivery' / delivery_id
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_file = receipt_dir / 'receipt.json'
    receipt_file.write_text(json.dumps({
        'node': node_id, 'status': 'pending',
        'started': '2026-01-01T00:00:00Z', 'artifacts': {},
    }))

    call_count = 0
    def _always_fail(ctx):
        nonlocal call_count
        call_count += 1
        raise RuntimeError('deliberate test failure')

    monkeypatch.setattr(run_mod, 'resolve', lambda target: _always_fail)
    monkeypatch.setenv('REFACTORY_PROJECT', str(tmp_path))

    result = run_mod.run_node(
        'test_module:always_fail',
        workdir=workdir,
        node_id=node_id,
    )
    assert result == {}, f'optional op must return empty dict, got {result!r}'

    # Pending marker should be invalidated (removed)
    assert not pending_file.exists(), 'pending marker should be invalidated'

    # Receipt should be updated to superseded
    updated = json.loads(receipt_file.read_text())
    assert updated['status'] == 'superseded'


def test_worker_node_warns_when_the_marker_cannot_be_settled(tmp_path, monkeypatch, capsys):
    """dm-232: a marker that cannot be settled is a run warning, not silence; the branch stays missing."""
    import sys
    from refactory_dsh import agent_delivery, worker_node

    spec_path = tmp_path / 'spec.json'
    spec_path.write_text(json.dumps({'schema_version': 1, 'node_id': 'verifier', 'optional': True,
                                     'reads': [], 'writes': [], 'prompt': 'test', 'model_context': {}}))
    monkeypatch.setattr(sys, 'argv', ['worker_node', str(spec_path)])
    monkeypatch.chdir(tmp_path)

    def _failing_execute(s, w):
        raise RuntimeError('model completion timed out after 1s')

    def _unsettled(workdir, nodes, reason):
        raise OSError('receipt is unreadable')

    monkeypatch.setattr(worker_node, 'execute', _failing_execute)
    monkeypatch.setattr(agent_delivery, 'invalidate', _unsettled)
    assert worker_node.main() == 0
    err = capsys.readouterr().err
    assert 'refactory: run warning: could not settle the delivery marker of optional verifier' in err
    assert 'receipt is unreadable' in err
    assert (tmp_path / 'reviews/verifier-missing.txt').exists()


def test_run_node_warns_when_the_marker_cannot_be_settled(tmp_path, monkeypatch, capsys):
    """dm-232: run.py's optional path reports a marker it could not settle as a run warning."""
    from refactory_dsh import agent_delivery
    from refactory_dsh import run as run_mod

    workdir = tmp_path / 'run'
    (workdir / '.refactory').mkdir(parents=True)
    (workdir / '.refactory' / 'declarations.json').write_text(json.dumps({
        'verifier': {'reads': [], 'writes': [], 'produces': [], 'optional': True}}))
    (workdir / 'facts.json').write_text('{}')

    def _always_fail(ctx):
        raise RuntimeError('deliberate test failure')

    def _unsettled(workdir, nodes, reason):
        raise OSError('receipt is unreadable')

    monkeypatch.setattr(run_mod, 'resolve', lambda target: _always_fail)
    monkeypatch.setattr(agent_delivery, 'invalidate', _unsettled)
    monkeypatch.setenv('REFACTORY_PROJECT', str(tmp_path))
    assert run_mod.run_node('test_module:always_fail', workdir=workdir, node_id='verifier') == {}
    err = capsys.readouterr().err
    assert 'refactory: run warning: could not settle the delivery marker of optional verifier' in err
    assert 'receipt is unreadable' in err


class _TimesOutFor:
    """A model client whose session for one node runs past its wall; others answer."""

    def __init__(self, slow: str, node: str = '') -> None:
        self.slow, self.node = slow, node

    def for_node(self, node, role='worker'):
        return _TimesOutFor(self.slow, node)

    def complete(self, prompt, *, system='', workdir=None):
        if self.node == self.slow:
            raise RuntimeError('model completion timed out after 1s; requested runtime shutdown; not retried')
        return 'ok'


def _walker_review_fork():
    from refactory_dsh.nn import Graph, Node

    graph = Graph()
    graph.add(Node('fork', 'fork', targets=('verifier', 'reviewer'), optional_targets=('verifier',)))
    graph.add(Node('verifier', 'agent', prompt_template='Verify.', timeout_s=1))
    graph.add(Node('reviewer', 'agent', prompt_template='Review.'))
    graph.add(Node('join', 'join', sources=('reviewer',)))
    graph.connect('fork', 'join')
    graph.connect('verifier', 'join')
    graph.connect('reviewer', 'join')
    return graph


@pytest.mark.parametrize('concurrent', [False, True])
def test_walker_optional_branch_settles_its_pending_marker(tmp_path, concurrent):
    """dm-232 on the walker: an optional agent whose session timed out never reached
    `finish`, so the fork settles the delivery it began when it records the branch
    missing. Otherwise a later recovery of the directory refuses on the marker.

    A custom client runs the fork's branches on the caller's thread; the shipped DSH
    client runs them concurrently, here against a stand-in DSH whose verifier session
    outlives the verifier's one-second wall."""
    from refactory_dsh.nn.executor import Executor, RunContext

    workdir = tmp_path / 'run'
    workdir.mkdir()
    ctx = RunContext(workdir, knobs={}, facts={}, task={})
    if concurrent:
        dsh = tmp_path / 'dsh'
        dsh.write_text('#!/bin/sh\nif grep -q Verify; then exec sleep 60; fi\necho ok\n')
        dsh.chmod(0o755)
        ctx.model = DshModelClient(worker_backend='dsh', dsh_cli=str(dsh), attempts=1)
    else:
        ctx.model = _TimesOutFor('verifier')
    executor = Executor(_walker_review_fork(), ctx)
    executor.run(start='fork')
    assert any(step.node == 'verifier' and (step.reason or '').startswith('missing:') for step in executor.steps)
    assert list((workdir / '.refactory/agent-delivery/pending').glob('*.json')) == []
    receipts = [json.loads(p.read_text()) for p in (workdir / '.refactory/agent-delivery').glob('*/receipt.json')]
    assert {r['node']: r['status'] for r in receipts}['verifier'] == 'superseded'


def test_walker_warns_when_an_optional_marker_cannot_be_settled(tmp_path, monkeypatch, caplog):
    from refactory_dsh import agent_delivery
    from refactory_dsh.nn.executor import Executor, RunContext

    def _unsettled(workdir, nodes, reason):
        raise OSError('receipt is unreadable')

    monkeypatch.setattr(agent_delivery, 'invalidate', _unsettled)
    workdir = tmp_path / 'run'
    workdir.mkdir()
    ctx = RunContext(workdir, knobs={}, facts={}, task={})
    ctx.model = _TimesOutFor('verifier')
    with caplog.at_level('WARNING', logger='refactory_dsh.nn.fork'):
        Executor(_walker_review_fork(), ctx).run(start='fork')
    assert 'could not settle the delivery markers of verifier: receipt is unreadable' in caplog.text


def test_worker_node_non_optional_exits_one_on_failure(tmp_path, monkeypatch, capsys):
    """A worker spec without optional still exits 1 when execute raises."""
    import sys
    from refactory_dsh import worker_node

    spec = {
        'schema_version': 1,
        'node_id': 'verifier',
        'reads': [],
        'writes': [],
        'prompt': 'test',
        'model_context': {},
    }
    spec_path = tmp_path / 'spec.json'
    spec_path.write_text(json.dumps(spec))

    monkeypatch.setattr(sys, 'argv', ['worker_node', str(spec_path)])
    monkeypatch.chdir(tmp_path)

    def _failing_execute(s, w):
        raise RuntimeError('deliberate test error')
    monkeypatch.setattr(worker_node, 'execute', _failing_execute)

    exit_code = worker_node.main()
    assert exit_code == 1, f'non-optional worker must exit 1, got {exit_code}'

    missing_file = tmp_path / 'reviews' / 'verifier-missing.txt'
    assert not missing_file.exists(), 'missing file must NOT be written for non-optional'


def test_run_node_optional_op_records_missing(tmp_path, monkeypatch):
    """run_node for an optional op node records it as missing instead of raising."""
    from refactory_dsh import run as run_mod

    workdir = tmp_path / 'run'
    workdir.mkdir()
    refactory_dir = workdir / '.refactory'
    refactory_dir.mkdir()
    (refactory_dir / 'declarations.json').write_text(json.dumps({
        'verifier': {
            'reads': [],
            'writes': ['reviews/verification.md'],
            'produces': ['verification'],
            'optional': True,
        }
    }))
    (workdir / 'facts.json').write_text('{}')

    call_count = 0
    def _always_fail(ctx):
        nonlocal call_count
        call_count += 1
        raise RuntimeError('deliberate test failure')

    monkeypatch.setattr(run_mod, 'resolve', lambda target: _always_fail)
    monkeypatch.setenv('REFACTORY_PROJECT', str(tmp_path))

    result = run_mod.run_node(
        'test_module:always_fail',
        workdir=workdir,
        node_id='verifier',
    )
    assert result == {}, f'optional op must return empty dict, got {result!r}'
    assert call_count == 2, 'must attempt initial + one retry before recording missing'

    missing_file = workdir / 'reviews' / 'verifier-missing.txt'
    assert missing_file.exists(), 'missing file must be written'
    assert 'RuntimeError' in missing_file.read_text()


def test_workspace_shell_kills_hangs_and_remains_usable(tmp_path):
    root = Path(__file__).resolve().parents[2]
    if not (root / "node_modules/@deepseek-ai/dsh").exists():
        pytest.skip("DSH packages not installed")
    patch = tmp_path / "workspace.json"
    workers.write_patch(patch, root, DshModelClient().route(), tool_home=tmp_path / "home")
    subprocess.run(["node", str(Path(__file__).with_name("workspace_command_timeout_test.mjs")),
                    str(root), str(patch)], check=True, timeout=30)


def test_command_limit_is_scoped_to_dsh_workspace_tools(tmp_path):
    for backend, workspace in [("dsh", False), ("codex", True), ("claude_code", True)]:
        patch = tmp_path / (backend + ".json")
        root = Path(__file__).resolve().parents[2]
        if backend != "dsh" and not (root / "node_modules" / workers.PACKAGES[backend]).exists():
            continue
        workers.write_patch(patch, root, DshModelClient(worker_backend=backend).route(),
                            tool_home=tmp_path if workspace else None)
        assert not any(row.get("id") in ("bash-sandbox", "tool-bash")
                       for row in json.loads(patch.read_text()))
