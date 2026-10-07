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
                                   {'worker_backend': 'claude_code', 'effort': 'high'},
                                   {'worker_backend': 'codex', 'permission_mode': 'invalid'},
                                   {'worker_backend': 'invalid'}])
def test_unsupported_routes_fail_before_launch(route):
    with pytest.raises(ValueError):
        DshModelClient(**route)


def test_receipt_preserves_unavailable_usage_and_request_is_not_actual(monkeypatch, tmp_path):
    def complete(self, cmd, run_dir):
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
    def complete(self, cmd, run_dir):
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
    monkeypatch.setattr(DshModelClient, '_run_completion', lambda self, cmd, run_dir: subprocess.CompletedProcess(cmd, 0, 'done', ''))
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
    assert 'refactory_dsh.worker_node' in lowered.nodes['build'].command
    assert graph.nodes['build'].prompt_template == 'write result.txt'
    spec = json.loads(next((tmp_path / '.refactory/worker-specs').glob('*.json')).read_text())
    assert spec['model_context']['worker_backend'] == 'codex'
    assert spec['writes'] == ['result.txt']


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

    # An explicit node timeout is the author's own bound and passes through.
    explicit = workers.lower_workflow(graph(timeout=123), tmp_path / 'explicit', DshModelClient().context())
    assert explicit.nodes['build'].command_timeout_s == 123


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
    assert row['config'] == {'providers': {'gateway': gateway}}
    # Without a declaration the patch composes exactly what it did before.
    workers.write_patch(tmp_path / 'without-provider.json', tmp_path, DshModelClient().route())
    assert not any(entry.get('id') == 'llm-pi-ai'
                   for entry in json.loads((tmp_path / 'without-provider.json').read_text()))
    # The composed real headless tree resolves the declared route, not NO_ADAPTER.
    root = Path(__file__).resolve().parents[2]
    subprocess.run(['node', str(Path(__file__).with_name('worker_provider_config_test.mjs')),
                    str(root), str(tmp_path)], check=True)


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
    assert first.nodes['build'].command != second.nodes['build'].command
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
    monkeypatch.setattr(DshModelClient, '_run_completion', lambda self, cmd, run_dir: subprocess.CompletedProcess(cmd, 0, 'done', ''))
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
    bindir = str(Path(sys.executable).parent)
    # A narrow PATH still has to reach the shell the worker spawns and the selected
    # interpreter. `/usr/bin:/bin` is a distro assumption NixOS does not satisfy, so
    # pin the two directories the test actually needs instead.
    import os
    bash_dir = os.path.dirname(shutil.which('bash') or '')
    monkeypatch.setenv('PATH', os.pathsep.join(part for part in (bindir, bash_dir) if part))
    monkeypatch.delenv('PYTHONPATH', raising=False)
    monkeypatch.setenv('REFACTORY_TEST_SECRET', 'not-a-real-secret')
    client = DshModelClient(python_bin=bindir)
    from refactory_dsh.sandbox import scrub
    recreated = DshModelClient(**client.context())
    # Context survives even the narrower generated-package environment scrub.
    with monkeypatch.context() as isolated:
        isolated.setenv('VENV_BIN', '/not/the/selected/environment')
        assert DshModelClient(**recreated.context()).context()['python_bin'] == bindir
    expression = 'import json,sys,refactory_dsh; print(json.dumps({"executable":sys.executable,"module":refactory_dsh.__file__}))'
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
    assert Path(actual['module']).resolve() == root / 'refactory/refactory_dsh/__init__.py'
    assert 'REFACTORY_TEST_SECRET' not in scrub(client._environment())


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
