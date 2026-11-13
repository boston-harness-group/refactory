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


def test_validate_route_accepts_max_completion_tokens_string():
    """validate_route accepts max_completion_tokens when it is a string."""
    route = workers.validate_route({
        "worker_backend": "dsh",
        "max_completion_tokens": "16384",
    })
    assert route["max_completion_tokens"] == "16384"


def test_validate_route_rejects_max_completion_tokens_integer():
    """validate_route rejects max_completion_tokens when it is an integer
    (PIT-003: route fields must be strings)."""
    import pytest

    with pytest.raises(ValueError, match="must be strings"):
        workers.validate_route({
            "worker_backend": "dsh",
            "max_completion_tokens": 16384,
        })


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


def _bridge_prompt(tmp_path, startup: dict, stdin: bytes) -> dict:
    """Run the real worker bridge (fake services) and return the prompt it dispatched."""
    bridge = Path(__file__).resolve().parents[1] / 'refactory_dsh' / 'worker_bridge.mjs'
    (tmp_path / 'startup.json').write_text(json.dumps(startup))
    out = tmp_path / 'out.json'
    subprocess.run(['node', str(Path(__file__).with_name('worker_bridge_stdin_test.mjs')),
                    str(bridge), str(tmp_path / 'startup.json'), str(out)],
                   input=stdin, check=True, timeout=60, capture_output=True)
    return json.loads(out.read_text())


# Multi-byte characters of every width, so a 64 KB pipe chunk boundary splits some of them.
_LARGE_PROMPT = ('é ünïcödé — 棋盘 ♞♛ 🐴🏁 ' * 20000) + 'end'


def test_the_bridge_reads_the_prompt_from_stdin_when_the_task_is_a_dash(tmp_path):
    """The launcher sends large prompts on stdin with task "-"; the external-worker
    config disables the headless runner that resolves it, so the bridge must."""
    data = _LARGE_PROMPT.encode('utf-8')
    assert len(data) > 500_000
    sent = _bridge_prompt(tmp_path, {'task': '-'}, data)
    assert sent == {'code': 0, 'prompts': [_LARGE_PROMPT]}


def test_the_bridge_reads_stdin_when_no_task_is_given(tmp_path):
    sent = _bridge_prompt(tmp_path, {}, 'write answer.txt — ✓'.encode('utf-8'))
    assert sent == {'code': 0, 'prompts': ['write answer.txt — ✓']}


def test_the_bridge_passes_a_literal_task_through_and_ignores_stdin(tmp_path):
    sent = _bridge_prompt(tmp_path, {'task': 'write answer.txt with 42'}, b'not the prompt')
    assert sent == {'code': 0, 'prompts': ['write answer.txt with 42']}



@pytest.mark.parametrize('stdin', [b'', b'  \n\t'])
def test_the_bridge_refuses_an_empty_prompt_on_stdin(tmp_path, stdin):
    """As the headless runner does: an empty task fails loud, never reaches a worker."""
    sent = _bridge_prompt(tmp_path, {'task': '-'}, stdin)
    assert sent == {'code': 1, 'prompts': []}


def _patch_rows(tmp_path, route, **kwargs):
    from refactory_dsh.workers import write_patch
    root = Path(__file__).resolve().parents[2]
    patch = tmp_path / 'worker.patch.yml'
    write_patch(patch, root, route, **kwargs)
    return json.loads(patch.read_text())


def _external_config(rows):
    [row] = [row for row in rows if 'insert' in row and row['insert'][0]['id'] == 'refactory-external-worker']
    return row['insert'][0]['config']


def test_claude_code_patch_carries_node_settings_in_adapter_config(tmp_path):
    """The adapter reads effort, system prompt, tool policy and settings sources from
    its config env only; the DSH process env never reached it (#493 G40)."""
    route = {'worker_backend': 'claude_code', 'model': 'm', 'effort': 'low',
             'permission_mode': 'dontAsk', 'setting_sources': 'user,project'}
    try:
        config = _external_config(_patch_rows(tmp_path, route, system_prompt='Be brief', tool_policy='none'))
    except ValueError as error:
        pytest.skip(str(error))
    assert config['env'] == {'REFACTORY_MODEL_EFFORT': 'low', 'REFACTORY_SYSTEM_PROMPT': 'Be brief',
                             'REFACTORY_TOOL_POLICY': 'none', 'REFACTORY_SETTING_SOURCES': 'user,project'}
    plain = _external_config(_patch_rows(tmp_path, {'worker_backend': 'claude_code', 'model': 'm'}))
    assert 'env' not in plain


def test_codex_patch_carries_no_claude_code_settings(tmp_path):
    """Codex's adapter reads none of them; both runtimes refuse them before dispatch."""
    try:
        config = _external_config(_patch_rows(tmp_path, {'worker_backend': 'codex', 'model': 'm'},
                                              system_prompt='Be brief'))
    except ValueError as error:
        pytest.skip(str(error))
    assert 'env' not in config


def test_setting_sources_route_field_is_claude_code_only():
    from refactory_dsh.workers import resolve_route, validate_route
    assert validate_route({'worker_backend': 'claude_code', 'setting_sources': 'user,project'})['setting_sources'] == 'user,project'
    assert validate_route({'worker_backend': 'claude_code', 'setting_sources': 'none'})['setting_sources'] == 'none'
    with pytest.raises(ValueError, match='setting_sources must be'):
        validate_route({'worker_backend': 'claude_code', 'setting_sources': 'user,everything'})
    with pytest.raises(ValueError, match='codex does not support setting_sources'):
        validate_route({'worker_backend': 'codex', 'setting_sources': 'user'})
    # A node that switches product does not inherit Claude Code's settings sources.
    switched = resolve_route({'worker_backend': 'claude_code', 'setting_sources': 'user,project'},
                             node='n', node_routes={'n': {'worker_backend': 'dsh', 'provider': 'p', 'model': 'm'}})
    assert switched['setting_sources'] == ''


def test_setting_sources_crosses_the_worker_context(monkeypatch):
    """Carried in the client's context like max_completion_tokens, so a dispatched
    worker (the spine's worker_node) keeps the node route's value."""
    from refactory_dsh.model import DshModelClient
    client = DshModelClient(worker_backend='claude_code', model='m', routes_resolved=True,
                            node_routes={'solve': {'setting_sources': 'user,project'}}).for_node('solve')
    assert client.route()['setting_sources'] == 'user,project'
    monkeypatch.setenv('REFACTORY_WORKER_CONTEXT', json.dumps(client.context()))
    assert DshModelClient().route()['setting_sources'] == 'user,project'
    monkeypatch.delenv('REFACTORY_WORKER_CONTEXT')
    # The spine's worker rebuilds its client from the spec's context as arguments.
    capped = DshModelClient(worker_backend='claude_code', model='m', routes_resolved=True,
                            node_routes={'solve': {'setting_sources': 'user', 'max_completion_tokens': '64'}}
                            ).for_node('solve')
    rebuilt = DshModelClient(**capped.context()).route()
    assert (rebuilt['setting_sources'], rebuilt['max_completion_tokens']) == ('user', '64')



def test_a_dsh_client_in_a_claude_code_workers_child_does_not_inherit_its_setting_sources(tmp_path, monkeypatch):
    """REFACTORY_WORKER_CONTEXT reaches a Claude Code worker's children (DSH keeps
    REFACTORY_* through its env scrub); a DSH client there takes no Claude Code field."""
    # A run-level route, as `refactory.workers.json` or the launch sets it.
    parent = DshModelClient(worker_backend='claude_code', model='m', routes_resolved=True,
                            setting_sources='user,project')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('REFACTORY_WORKER_CONTEXT', json.dumps(parent.context()))
    child = DshModelClient(worker_backend='dsh', provider='p', model='m2')
    assert 'setting_sources' not in child.route()
    assert 'setting_sources' not in child.context()
    # The same product still inherits it, as effort is.
    assert DshModelClient(worker_backend='claude_code').route()['setting_sources'] == 'user,project'


def test_claude_executable_route_field_is_claude_code_only_and_absolute():
    from refactory_dsh.workers import resolve_route, validate_route
    route = validate_route({'worker_backend': 'claude_code', 'claude_executable': '/opt/claude/bin/claude'})
    assert route['claude_executable'] == '/opt/claude/bin/claude'
    with pytest.raises(ValueError, match='codex does not support claude_executable'):
        validate_route({'worker_backend': 'codex', 'claude_executable': '/opt/claude/bin/claude'})
    with pytest.raises(ValueError, match='claude_executable must be an absolute path'):
        validate_route({'worker_backend': 'claude_code', 'claude_executable': 'bin/claude'})
    # A node that switches product does not inherit Claude Code's executable.
    switched = resolve_route({'worker_backend': 'claude_code', 'claude_executable': '/opt/claude/bin/claude'},
                             node='n', node_routes={'n': {'worker_backend': 'dsh', 'provider': 'p', 'model': 'm'}})
    assert switched['claude_executable'] == ''


def test_claude_code_patch_carries_a_chosen_executable(tmp_path, monkeypatch):
    """The route's executable, else REFACTORY_CLAUDE_EXECUTABLE, reaches the adapter's
    config env."""
    route = {'worker_backend': 'claude_code', 'model': 'm', 'claude_executable': '/opt/claude/bin/claude'}
    try:
        config = _external_config(_patch_rows(tmp_path, route))
    except ValueError as error:
        pytest.skip(str(error))
    assert config['env'] == {'REFACTORY_CLAUDE_EXECUTABLE': '/opt/claude/bin/claude'}
    monkeypatch.setenv('REFACTORY_CLAUDE_EXECUTABLE', '/env/claude')
    config = _external_config(_patch_rows(tmp_path, {'worker_backend': 'claude_code', 'model': 'm'}))
    assert config['env'] == {'REFACTORY_CLAUDE_EXECUTABLE': '/env/claude'}
    # The route's own choice wins over the environment's.
    assert _external_config(_patch_rows(tmp_path, route))['env']['REFACTORY_CLAUDE_EXECUTABLE'] \
        == '/opt/claude/bin/claude'
    monkeypatch.delenv('REFACTORY_CLAUDE_EXECUTABLE')
    assert 'env' not in _external_config(_patch_rows(tmp_path, {'worker_backend': 'claude_code', 'model': 'm'}))
    # The background-task quiet bound rides the same env, only when this process sets it.
    monkeypatch.setenv('REFACTORY_BACKGROUND_QUIET_S', '30')
    assert _external_config(_patch_rows(tmp_path, {'worker_backend': 'claude_code', 'model': 'm'}))['env'] \
        == {'REFACTORY_BACKGROUND_QUIET_S': '30'}


def test_claude_executable_crosses_the_worker_context_and_defaults_stay_unchanged(monkeypatch):
    """Carried in the client's context like setting_sources, only when chosen; the
    environment variable applies to claude_code clients only."""
    plain = DshModelClient(worker_backend='claude_code', model='m', routes_resolved=True)
    assert 'claude_executable' not in plain.route() and 'claude_executable' not in plain.context()
    client = DshModelClient(worker_backend='claude_code', model='m', routes_resolved=True,
                            node_routes={'solve': {'claude_executable': '/opt/claude'}}).for_node('solve')
    assert client.route()['claude_executable'] == '/opt/claude'
    assert DshModelClient(**client.context()).route()['claude_executable'] == '/opt/claude'
    monkeypatch.setenv('REFACTORY_WORKER_CONTEXT', json.dumps(client.context()))
    assert DshModelClient().route()['claude_executable'] == '/opt/claude'
    # A DSH client in a Claude Code worker's child takes none of a run-level choice.
    monkeypatch.delenv('REFACTORY_WORKER_CONTEXT')
    parent = DshModelClient(worker_backend='claude_code', model='m', routes_resolved=True,
                            claude_executable='/opt/claude')
    monkeypatch.setenv('REFACTORY_WORKER_CONTEXT', json.dumps(parent.context()))
    assert 'claude_executable' not in DshModelClient(worker_backend='dsh', provider='p', model='m2').context()
    monkeypatch.delenv('REFACTORY_WORKER_CONTEXT')
    monkeypatch.setenv('REFACTORY_CLAUDE_EXECUTABLE', '~/bin/claude')
    assert DshModelClient(worker_backend='claude_code', model='m', routes_resolved=True).route()[
        'claude_executable'] == str(Path.home() / 'bin' / 'claude')
    assert 'claude_executable' not in DshModelClient(provider='p', model='m').route()


def _fake_claude(tmp_path, version):
    path = tmp_path / f'claude-{version}'
    path.write_text(f'#!/bin/sh\necho "{version} (Claude Code)"\n')
    path.chmod(0o755)
    return path


def _fake_runtime(tmp_path, claude_code_version='2.1.263'):
    """A runtime root whose adapter resolves an Agent SDK bundling ``claude_code_version``."""
    root = tmp_path / 'runtime'
    adapter = root / 'node_modules' / '@deepseek-ai' / 'dsh-subagent-claude-code'
    adapter.mkdir(parents=True)
    (adapter / 'package.json').write_text('{}')
    sdk = root / 'node_modules' / '@anthropic-ai' / 'claude-agent-sdk'
    sdk.mkdir(parents=True)
    (sdk / 'package.json').write_text(json.dumps({'version': '0.3.263', 'claudeCodeVersion': claude_code_version}))
    binary = root / 'node_modules' / '@anthropic-ai' / 'claude-agent-sdk-darwin-arm64' / 'claude'
    binary.parent.mkdir()
    binary.write_text('')
    return root


def test_claude_code_minimum_versions_cover_the_claude_5_family():
    from refactory_dsh.workers import claude_code_minimum
    for model in ('claude-opus-5-5', 'anthropic/claude-sonnet-5', 'claude-haiku-5-1[1m]', 'claude-fable-5'):
        assert claude_code_minimum(model) == '2.1.280', model
    for model in ('claude-opus-4-6', 'claude-test', '', 'gpt-5.5'):
        assert claude_code_minimum(model) == '', model


def test_bundled_claude_code_is_read_from_the_sdk_package_without_starting_it(tmp_path):
    from refactory_dsh.workers import bundled_claude_code
    root = _fake_runtime(tmp_path, '2.1.999')
    found = bundled_claude_code(root)
    assert found == {'path': str(root / 'node_modules/@anthropic-ai/claude-agent-sdk-darwin-arm64/claude'),
                     'version': '2.1.999', 'sdk_version': '0.3.263'}
    assert bundled_claude_code(tmp_path / 'empty') is None


def test_preflight_refuses_an_old_chosen_claude_code_for_a_claude_5_model(tmp_path):
    from refactory_dsh.workers import claude_code_preflight
    old = _fake_claude(tmp_path, '2.1.263')
    client = DshModelClient(worker_backend='claude_code', model='claude-opus-5-5', routes_resolved=True,
                            claude_executable=str(old), runtime_root=_fake_runtime(tmp_path))
    with pytest.raises(ValueError) as refused:
        claude_code_preflight(client)
    message = str(refused.value)
    for part in (f'Claude Code 2.1.263 at {old}', 'model claude-opus-5-5', '2.1.280 or newer',
                 '--claude-executable /path/to/claude', 'REFACTORY_CLAUDE_EXECUTABLE',
                 'claude_executable" in refactory.workers.json'):
        assert part in message
    # A new enough one passes, and a model without a known minimum is not checked.
    claude_code_preflight(DshModelClient(worker_backend='claude_code', model='claude-opus-5-5',
                                         routes_resolved=True, claude_executable=str(_fake_claude(tmp_path, '2.1.300'))))
    claude_code_preflight(DshModelClient(worker_backend='claude_code', model='claude-opus-4-6',
                                         routes_resolved=True, claude_executable=str(old)))


def test_preflight_refuses_the_bundled_claude_code_for_a_claude_5_model(tmp_path):
    from refactory_dsh.workers import claude_code_preflight
    root = _fake_runtime(tmp_path, '2.1.263')
    client = DshModelClient(worker_backend='claude_code', model='claude-opus-5-5', routes_resolved=True,
                            runtime_root=root)
    with pytest.raises(ValueError, match=r'Claude Code 2\.1\.263 at .*claude-agent-sdk-darwin-arm64/claude '
                                         r'\(bundled with @anthropic-ai/claude-agent-sdk 0\.3\.263\)'):
        claude_code_preflight(client)
    claude_code_preflight(DshModelClient(worker_backend='claude_code', model='claude-opus-5-5',
                                         routes_resolved=True, runtime_root=_fake_runtime(tmp_path / 'new', '2.1.300')))


def test_preflight_checks_role_and_node_routes_and_skips_other_backends(tmp_path):
    from refactory_dsh.workers import claude_code_preflight
    root = _fake_runtime(tmp_path)
    dsh = DshModelClient(provider='p', model='m', routes_resolved=True, runtime_root=root)
    claude_code_preflight(dsh)
    on_node = DshModelClient(provider='p', model='m', routes_resolved=True, runtime_root=root, node_routes={
        'build': {'worker_backend': 'claude_code', 'model': 'claude-sonnet-5'}})
    with pytest.raises(ValueError, match="worker for node 'build'.*model claude-sonnet-5"):
        claude_code_preflight(on_node)
    on_role = DshModelClient(provider='p', model='m', routes_resolved=True, runtime_root=root, role_routes={
        'builder': {'worker_backend': 'claude_code', 'model': 'claude-opus-5-5'}})
    with pytest.raises(ValueError, match="worker for role 'builder'"):
        claude_code_preflight(on_role.context())


def test_preflight_refuses_a_missing_or_silent_chosen_executable(tmp_path):
    from refactory_dsh.workers import claude_code_preflight
    missing = DshModelClient(worker_backend='claude_code', model='claude-opus-4-6', routes_resolved=True,
                             claude_executable=str(tmp_path / 'nothing'))
    with pytest.raises(ValueError, match='which does not exist'):
        claude_code_preflight(missing)
    silent = tmp_path / 'silent'
    silent.write_text('#!/bin/sh\nexit 0\n')
    silent.chmod(0o755)
    with pytest.raises(ValueError, match='reported no version'):
        claude_code_preflight(DshModelClient(worker_backend='claude_code', model='claude-opus-5-5',
                                             routes_resolved=True, claude_executable=str(silent)))


def test_the_design_walker_refuses_an_old_claude_code_before_the_run_directory_exists(tmp_path, monkeypatch, capsys):
    """`--worker-backend claude_code --model claude-opus-5-5` once failed at the first
    model call, after setup and the worktrees; now it is refused at launch."""
    from refactory_dsh import design
    monkeypatch.chdir(tmp_path)
    workdir = tmp_path / 'the-run'
    code = design.main(['--name', 'tool', '--brief', 'Change tool.py.', '--check', 'true', '--no-worktree',
                        '--workdir', str(workdir), '--worker-backend', 'claude_code',
                        '--model', 'claude-opus-5-5', '--claude-executable', str(_fake_claude(tmp_path, '2.1.263'))])
    assert code == 1
    assert 'Claude Code 2.1.263' in capsys.readouterr().err
    assert not workdir.exists()


def test_an_external_worker_gets_its_system_prompt_from_config_not_the_environment(tmp_path):
    rows = _patch_rows(tmp_path, {'worker_backend': 'claude_code', 'model': 'm'}, system_prompt='Be brief') \
        if (Path(__file__).resolve().parents[2] / 'node_modules/@deepseek-ai/dsh-subagent-claude-code/package.json').exists() else None
    if rows is None:
        pytest.skip('Claude Code adapter not installed')
    assert not any(row.get('insert', [{}])[0].get('id') == 'refactory-system-prompt' for row in rows)
    assert _external_config(rows)['env']['REFACTORY_SYSTEM_PROMPT'] == 'Be brief'
    dsh_rows = _patch_rows(tmp_path, {'worker_backend': 'dsh', 'provider': 'p', 'model': 'm'}, system_prompt='Be brief')
    assert any(row.get('insert', [{}])[0].get('id') == 'refactory-system-prompt' for row in dsh_rows)
    client = DshModelClient(worker_backend='claude_code', model='m', routes_resolved=True)
    client._system_text = 'Be brief'
    assert 'REFACTORY_SYSTEM_PROMPT' not in client._environment()
    dsh = DshModelClient(worker_backend='dsh', provider='p', model='m', routes_resolved=True)
    dsh._system_text = 'Be brief'
    assert dsh._environment()['REFACTORY_SYSTEM_PROMPT'] == 'Be brief'


_NO_TOOLS_HARNESS = """
import { pathToFileURL } from 'node:url'
const handlers = {}
const plugin = await import(pathToFileURL(process.argv[2]).href)
plugin.apply({ tools: { guard() {} }, on: (name, handler) => { handlers[name] = handler } })
if (process.argv[3] === 'request') await handlers['llm/stream']({ tools: [] }, async () => 'sent')
"""


@pytest.mark.parametrize('request_made', [False, True])
def test_a_dsh_no_tools_worker_is_enforced_only_once_the_plugin_sees_a_request(tmp_path, request_made):
    """begin() records the request; only no_tools_plugin.mjs, at the provider request
    it guards, records the policy as enforced."""
    import os
    route = {'worker_backend': 'dsh', 'provider': 'p', 'model': 'm', 'effort': '', 'permission_mode': 'skip'}
    path = workers.begin(tmp_path, route, node='n', tool_policy='none')
    harness = tmp_path / 'no_tools_harness.mjs'
    harness.write_text(_NO_TOOLS_HARNESS)
    plugin = Path(workers.__file__).with_name('no_tools_plugin.mjs')
    subprocess.run(['node', str(harness), str(plugin), 'request' if request_made else 'none'], check=True,
                   timeout=60, env={**os.environ, 'REFACTORY_WORKER_RECEIPT': str(path)})
    record = json.loads(path.read_text())
    assert record['requested_tool_policy'] == 'none'
    if request_made:
        assert (record['enforced_tool_policy'], record['tools_permitted'], record['tool_policy_source']) == (
            'none', [], 'dsh-provider-request')
    else:
        assert (record['enforced_tool_policy'], record['tool_policy_source']) == (None, 'not-yet-observed')


def test_the_spine_launcher_refuses_an_old_claude_code_before_the_run_directory_exists(tmp_path, monkeypatch):
    """`refactory ceo` (`interactive.run`) checks it with the run's model, before
    `ownership` creates the run directory."""
    from refactory_dsh import dsh_profile, interactive, steer_probe
    monkeypatch.setattr(interactive, 'require_build', lambda **kw: None)
    monkeypatch.setattr(steer_probe, 'find_steer_module', lambda: 'refactory.steer')
    monkeypatch.setattr(dsh_profile, 'require', lambda **kw: None)
    workdir = tmp_path / 'the-run'
    with pytest.raises(ValueError, match=r'Claude Code 2\.1\.263 .*model claude-opus-5-5'):
        interactive.run('default_pack/design_fast', workdir=workdir, worker_backend='claude_code',
                        model='claude-opus-5-5', claude_executable=str(_fake_claude(tmp_path, '2.1.263')))
    assert not workdir.exists()


def test_preflight_lets_through_routes_only_some_roles_or_nodes_combine():
    """A role that switches product and a node override that only the run's backend takes
    are a valid configuration: each node runs on its own role's route. The check refused
    them by resolving every node under every role (review of #637)."""
    from refactory_dsh.workers import claude_code_preflight
    for override in ({'setting_sources': 'user'}, {'effort': 'high'}, {'permission_mode': 'acceptEdits'}):
        client = DshModelClient(worker_backend='claude_code', model='claude-opus-4-6', routes_resolved=True,
                                role_routes={'qa': {'worker_backend': 'codex'}}, node_routes={'build': override})
        client.for_node('build', role='builder')
        claude_code_preflight(client)
        claude_code_preflight(client, roles={'build': 'builder', 'review': 'qa'})
        # A combination dispatch would refuse is dispatch's to report, not the preflight's.
        claude_code_preflight(client, roles={'build': 'qa'})


def test_preflight_checks_each_node_on_its_own_roles_route(tmp_path):
    """A node override applies under the node's own role only: on a DSH node it is not a
    Claude Code route, and on a claude_code role's node it is."""
    from refactory_dsh.workers import claude_code_preflight
    old = str(_fake_claude(tmp_path, '2.1.263'))
    client = DshModelClient(provider='p', model='m', routes_resolved=True,
                            role_routes={'qa': {'worker_backend': 'claude_code', 'model': 'claude-sonnet-4-6',
                                                'claude_executable': old}},
                            node_routes={'design': {'model': 'claude-opus-5-5'}})
    claude_code_preflight(client)
    claude_code_preflight(client, roles={'design': 'designer', 'check': 'qa'})
    with pytest.raises(ValueError, match=r"node 'design' \(role 'qa'\) would run Claude Code 2\.1\.263 .*claude-opus-5-5"):
        claude_code_preflight(client, roles={'design': 'qa'})


def test_agent_roles_are_the_roles_dispatch_uses():
    """Every agent node of a shipped mode, fork branches included, with its `Node.role`."""
    from refactory_dsh import registry
    from refactory_dsh.nn.graph import Graph
    from refactory_dsh.paths import REPO_ROOT
    from refactory_dsh.workers import agent_roles
    state = registry.resolve(REPO_ROOT, 'default_pack/design')[1]().state_dict()
    graph = Graph.from_state(state['graph'])
    roles = agent_roles(state)
    assert roles == {node.id: node.role for node in graph.nodes.values() if node.kind == 'agent'}
    assert len(roles) > 3 and len(set(roles.values())) > 1


def test_a_version_with_a_suffix_is_compared_on_its_leading_numbers(tmp_path):
    from refactory_dsh.workers import _version_tuple, claude_code_preflight
    assert _version_tuple('2.1.281-rc') == (2, 1, 281)
    assert _version_tuple('2.1.281') == (2, 1, 281)
    assert _version_tuple('next') is None
    claude_code_preflight(DshModelClient(worker_backend='claude_code', model='claude-opus-5-5', routes_resolved=True,
                                         runtime_root=_fake_runtime(tmp_path / 'rc', '2.1.281-rc.1')))
    with pytest.raises(ValueError, match=r'2\.1\.263-beta'):
        claude_code_preflight(DshModelClient(worker_backend='claude_code', model='claude-opus-5-5',
                                             routes_resolved=True,
                                             runtime_root=_fake_runtime(tmp_path / 'beta', '2.1.263-beta')))


def test_the_console_open_path_refuses_an_old_claude_code_in_prepare(tmp_path):
    """`console.py` opens a run through `interactive.prepare`, so the check is there too,
    before the run directory exists."""
    from refactory_dsh import interactive
    workdir = tmp_path / 'the-run'
    with pytest.raises(ValueError, match=r'Claude Code 2\.1\.263 .*model claude-opus-5-5'):
        interactive.prepare('default_pack/design_fast', workdir=workdir, model_context={
            'worker_backend': 'claude_code', 'model': 'claude-opus-5-5',
            'claude_executable': str(_fake_claude(tmp_path, '2.1.263'))})
    assert not workdir.exists()


def test_the_optimizer_loop_refuses_an_old_claude_code_before_its_run_directory(tmp_path):
    from refactory_dsh.loop import Loop, LoopConfig, LoopError
    from refactory_dsh.paths import REPO_ROOT
    root = tmp_path / 'search'
    client = DshModelClient(worker_backend='claude_code', model='claude-opus-5-5', routes_resolved=True,
                            claude_executable=str(_fake_claude(tmp_path, '2.1.263')))
    with pytest.raises(LoopError, match=r"Claude Code 2\.1\.263 .*model claude-opus-5-5"):
        Loop('circle_packing/adaptive', LoopConfig(optimizer='autoresearch', runs_root=root), project=REPO_ROOT,
             model=client)
    assert not root.exists()


@pytest.mark.parametrize('entry', ['create', 'edit'])
def test_create_and_edit_refuse_an_old_claude_code_before_anything_runs(entry, tmp_path, monkeypatch, capsys):
    import importlib
    main = importlib.import_module(f'refactory_dsh.{entry}').main
    monkeypatch.chdir(tmp_path)
    workdir = tmp_path / 'the-run'
    given = (['--single'] if entry == 'create' else ['--mode', 'circle_packing/adaptive'])
    code = main(['--name', 'tool', '--brief', 'Make a tool.', *given, '--workdir', str(workdir),
                 '--worker-backend', 'claude_code', '--model', 'claude-opus-5-5',
                 '--claude-executable', str(_fake_claude(tmp_path, '2.1.263'))])
    assert code == 1
    assert 'Claude Code 2.1.263' in capsys.readouterr().err
    assert not workdir.exists()
