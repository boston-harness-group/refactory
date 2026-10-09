"""Worker permissions stay scoped and durable without starting a model runtime."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from refactory_dsh import console, interactive, workers
from refactory_dsh.budgets import RunBudget
from refactory_dsh.model import DshModelClient


PROJECT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('permission', ['skip', 'default'])
def test_dsh_worker_and_coordinator_overlay_do_not_change_child_environment(tmp_path, monkeypatch, permission):
    monkeypatch.delenv('REFACTORY_WORKER_CONTEXT', raising=False)
    monkeypatch.setenv('DSH_PERMISSION_MODE', 'workspace-write')
    seen = []

    def inspect_completion(client, argv, _workdir, _stop_at=None, _budget_s=None):
        # Read the actual generated launch overlay, not only its helper output.
        index = len(argv) - 1 - argv[::-1].index('--patch')
        seen.extend(json.loads(Path(argv[index + 1]).read_text()))
        assert client._environment()['DSH_PERMISSION_MODE'] == 'workspace-write'
        return subprocess.CompletedProcess(argv, 0, 'done', '')

    monkeypatch.setattr(DshModelClient, '_run_completion', inspect_completion)
    client = DshModelClient(permission_mode=permission)
    assert client.complete('No real completion is permitted.', workdir=tmp_path) == 'done'
    assert 'DSH_PERMISSION_MODE' not in workers.environment(client.context())
    telemetry = workers.GraphTelemetry(tmp_path, client.context())
    coordinator = json.loads(telemetry.patch.read_text())
    assert json.loads(telemetry.path.read_text())['requested_route']['permission_mode'] == permission
    for overlay in (seen, coordinator):
        policies = {row['id']: row['config'] for row in overlay if row.get('id') in ('sandbox-policy', 'approval')}
        if permission == 'skip':
            assert policies['sandbox-policy']['mode'] == 'danger-full-access'
            assert policies['approval']['policy'] == 'never'
        else:
            assert policies == {}
    # A conservative child retains ambient native policy even after a skip parent.
    monkeypatch.setenv('REFACTORY_WORKER_CONTEXT', json.dumps(client.context()))
    child = DshModelClient(permission_mode='default')
    assert child.route()['permission_mode'] == 'default'
    assert child._environment()['DSH_PERMISSION_MODE'] == 'workspace-write'


@pytest.mark.parametrize(('parent', 'arguments', 'expected'), [
    ({'worker_backend': 'dsh', 'permission_mode': 'skip'}, {}, 'skip'),
    ({'worker_backend': 'dsh', 'permission_mode': 'skip'}, {'permission_mode': 'default'}, 'default'),
    ({'worker_backend': 'dsh', 'provider': 'old-provider', 'model': 'old-model', 'effort': 'high',
      'permission_mode': 'default'}, {'worker_backend': 'codex'}, 'default'),
    ({'worker_backend': 'dsh', 'provider': 'old-provider', 'model': 'old-model', 'effort': 'high',
      'permission_mode': 'skip'}, {'worker_backend': 'codex'}, 'skip'),
    ({'worker_backend': 'codex', 'permission_mode': 'never'}, {}, 'never'),
    ({'worker_backend': 'claude_code', 'permission_mode': 'acceptEdits'}, {'worker_backend': 'codex'}, 'skip'),
    ({'worker_backend': 'claude_code', 'permission_mode': 'acceptEdits'},
     {'worker_backend': 'codex', 'permission_mode': 'default'}, 'default'),
])
def test_explicit_native_policy_and_same_backend_inheritance(monkeypatch, parent, arguments, expected):
    monkeypatch.setenv('REFACTORY_WORKER_CONTEXT', json.dumps(parent))
    client = DshModelClient(**arguments)
    assert client.route()['permission_mode'] == expected
    if arguments.get('worker_backend', parent['worker_backend']) != parent['worker_backend']:
        assert all(client.route()[key] == '' for key in ('provider', 'model', 'effort'))
    assert DshModelClient(**client.context()).route() == client.route()


@pytest.mark.parametrize(('backend', 'permission', 'expected'), [
    ('dsh', 'default', 'default'), ('dsh', 'skip', 'skip'),
    ('claude_code', 'acceptEdits', 'skip'),
])
@pytest.mark.parametrize('route_kind', ['role_routes', 'node_routes'])
def test_backend_switch_preserves_shared_permissions_only(backend, permission, expected, route_kind):
    routes = {route_kind: {'builder': {'worker_backend': 'codex'}}}
    client = DshModelClient(worker_backend=backend, permission_mode=permission, **routes)
    selected = client.for_node('builder', 'builder')
    assert selected.route()['worker_backend'] == 'codex'
    assert selected.route()['permission_mode'] == expected
    assert client.route()['permission_mode'] == permission


@pytest.mark.parametrize(('context', 'permission'), [(None, 'skip'), ({'permission_mode': 'default'}, 'default')])
def test_fresh_prepare_persists_effective_policy_in_every_worker_spec(tmp_path, monkeypatch, context, permission):
    monkeypatch.delenv('REFACTORY_WORKER_CONTEXT', raising=False)
    interactive.prepare('default_pack/create', workdir=tmp_path, project=PROJECT, model_context=context)
    saved = json.loads((tmp_path / '.refactory/run-config.json').read_text())['model_context']
    assert saved['permission_mode'] == permission
    specs = list((tmp_path / '.refactory/worker-specs').glob('*.json'))
    assert specs
    for path in specs:
        assert json.loads(path.read_text())['model_context']['permission_mode'] == permission


def test_console_open_propagates_the_persisted_fresh_default(tmp_path, monkeypatch):
    monkeypatch.delenv('REFACTORY_WORKER_CONTEXT', raising=False)
    opened = console._open_mode('default_pack/create', workdir=tmp_path, project=PROJECT)
    saved = json.loads((tmp_path / '.refactory/run-config.json').read_text())['model_context']
    assert saved['permission_mode'] == 'skip'
    assert json.loads(opened['commandEnv']['REFACTORY_WORKER_CONTEXT']) == saved
    assert 'DSH_PERMISSION_MODE' not in opened['commandEnv']


@pytest.mark.parametrize(('historical', 'expected'), [
    (None, 'default'),
    ({}, 'default'),
    ({'worker_backend': 'dsh'}, 'default'),
    ({'worker_backend': 'dsh', 'permission_mode': None}, 'default'),
    ({'worker_backend': 'dsh', 'permission_mode': 'default'}, 'default'),
    ({'worker_backend': 'dsh', 'permission_mode': 'skip'}, 'skip'),
])
def test_recovery_retains_recorded_policy_without_rewriting_history(tmp_path, monkeypatch, historical, expected):
    monkeypatch.delenv('REFACTORY_WORKER_CONTEXT', raising=False)
    name, _ = interactive.prepare('counting_pack/fixed_step', workdir=tmp_path, project=PROJECT,
                                  resource_limits={'max_elapsed_s': 60})
    config = tmp_path / '.refactory/run-config.json'
    saved = json.loads(config.read_text())
    saved['model_context'] = historical
    config.write_text(json.dumps(saved))
    original = config.read_bytes()
    RunBudget.open(tmp_path).pause()
    (tmp_path / '.refactory/spine-checkpoint.json').write_text(json.dumps({
        'run_id': str(tmp_path), 'graph': {'name': name}, 'status': 'paused', 'file_fingerprints': {},
    }))
    seen = []
    monkeypatch.setattr(workers, 'run_graph_command', lambda *args, **kwargs:
                        seen.append(kwargs) or SimpleNamespace(returncode=0))
    # Exercises the real recovery validation; only process launch is intercepted.
    assert interactive.run('counting_pack/fixed_step', workdir=tmp_path, resume=True, headless=True) == 0
    assert seen[0]['context']['permission_mode'] == expected
    assert config.read_bytes() == original
    assert (tmp_path / '.refactory/resume-request.json').exists()
    assert RunBudget.open(tmp_path).snapshot()['started_at'] is None


def test_legacy_null_context_without_budget_pins_native_policy_for_nested_workers(tmp_path, monkeypatch):
    monkeypatch.delenv('REFACTORY_WORKER_CONTEXT', raising=False)
    name, _ = interactive.prepare('counting_pack/fixed_step', workdir=tmp_path, project=PROJECT)
    config = tmp_path / '.refactory/run-config.json'
    saved = json.loads(config.read_text())
    saved.update(model_context=None, resource_limits=None)
    config.write_text(json.dumps(saved))
    original = config.read_bytes()
    (tmp_path / '.refactory/spine-checkpoint.json').write_text(json.dumps({
        'run_id': str(tmp_path), 'graph': {'name': name}, 'status': 'paused', 'file_fingerprints': {},
    }))
    monkeypatch.setenv('REFACTORY_WORKER_CONTEXT', json.dumps({
        'worker_backend': 'dsh', 'permission_mode': 'skip',
        'role_routes': {'builder': {'worker_backend': 'codex', 'permission_mode': 'skip'}},
        'node_routes': {'repair': {'worker_backend': 'claude_code', 'permission_mode': 'bypassPermissions'}},
    }))
    seen = []

    def inspect_launch(*args, **kwargs):
        seen.append(kwargs)
        context = json.loads(kwargs['env']['REFACTORY_WORKER_CONTEXT'])
        assert context == kwargs['context']
        assert context['worker_backend'] == 'dsh'
        assert context['permission_mode'] == 'default'
        assert context['role_routes'] == context['node_routes'] == {}
        with monkeypatch.context() as nested:
            nested.setenv('REFACTORY_WORKER_CONTEXT', kwargs['env']['REFACTORY_WORKER_CONTEXT'])
            client = DshModelClient()
            assert client.route()['permission_mode'] == 'default'
            for node, role in [('build', 'builder'), ('repair', 'worker')]:
                route = client.for_node(node, role).route()
                assert route['worker_backend'] == 'dsh'
                assert route['permission_mode'] == 'default'
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(workers, 'run_graph_command', inspect_launch)
    monkeypatch.setattr(interactive, 'run_owned', lambda *args, **kwargs: pytest.fail('legacy resume must pin its worker context'))
    assert interactive.run('counting_pack/fixed_step', workdir=tmp_path, resume=True, headless=True) == 0
    assert len(seen) == 1
    assert config.read_bytes() == original
    assert not (tmp_path / '.refactory/resource-usage.json').exists()
