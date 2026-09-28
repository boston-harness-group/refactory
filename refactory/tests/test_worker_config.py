"""Project routes are resolved at launch and pinned through all child executions."""
import json
from pathlib import Path

import pytest

from refactory_dsh import console, interactive, profiles
from refactory_dsh.model import DshModelClient
from refactory_dsh.worker_config import FILENAME, resolve


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for key in ('REFACTORY_WORKER_CONTEXT', 'REFACTORY_WORKER_BACKEND', 'REFACTORY_MODEL_ID',
                'REFACTORY_MODEL_PROVIDER', 'REFACTORY_MODEL_EFFORT'):
        monkeypatch.delenv(key, raising=False)


def write(project, settings):
    project.mkdir(parents=True, exist_ok=True)
    (project / FILENAME).write_text(json.dumps(settings))


@pytest.mark.parametrize('backend', ['dsh', 'codex', 'claude_code'])
def test_project_default_and_explicit_dsh(tmp_path, backend, monkeypatch):
    monkeypatch.setenv('REFACTORY_MODEL_ID', 'dsh-model')
    monkeypatch.setenv('REFACTORY_MODEL_PROVIDER', 'dsh-provider')
    write(tmp_path, {'worker_backend': backend, 'model': 'selected'})
    selected = DshModelClient(project=tmp_path)
    assert selected.route()['worker_backend'] == backend
    assert selected.route()['model'] == 'selected'
    assert selected.route()['permission_mode'] == 'skip'
    if backend != 'dsh':
        assert selected.route()['provider'] == ''
        explicit = DshModelClient(project=tmp_path, worker_backend='dsh')
        assert explicit.route()['model'] == explicit.route()['provider'] == ''
    assert selected.context()['routing_provenance']['file']['loaded']
    assert resolve(tmp_path)['settings']['worker_backend'] == backend


def test_coder_only_and_role_node_precedence(tmp_path):
    write(tmp_path, {'worker_backend': 'dsh', 'model': 'ceo-worker',
                    'role_routes': {'worker': {'worker_backend': 'claude_code'}},
                    'node_routes': {'emit': {'worker_backend': 'codex', 'model': 'coder'}}})
    client = DshModelClient(project=tmp_path)
    assert client.for_node('emit').route()['worker_backend'] == 'codex'
    assert client.for_node('emit').route()['model'] == 'coder'
    assert client.for_node('review').route()['worker_backend'] == 'claude_code'
    assert client.for_node('review').route()['model'] == ''
    assert client.for_node('research', 'researcher').route()['worker_backend'] == 'dsh'
    assert DshModelClient(project=tmp_path, role_routes={}, node_routes={}).for_node('emit').route()['worker_backend'] == 'dsh'


def test_saved_and_nested_contexts_ignore_later_project_and_environment(tmp_path, monkeypatch):
    project, elsewhere = tmp_path / 'project', tmp_path / 'artifact'
    write(project, {'worker_backend': 'codex', 'permission_mode': 'default'})
    saved = DshModelClient(project=project).context()
    write(project, {'worker_backend': 'claude_code'})
    write(elsewhere, {'worker_backend': 'dsh', 'model': 'wrong'})
    monkeypatch.setenv('REFACTORY_WORKER_CONTEXT', json.dumps(saved))
    monkeypatch.setenv('REFACTORY_MODEL_ID', 'wrong-environment')
    for client in (DshModelClient(), DshModelClient(project=elsewhere), DshModelClient(**saved)):
        assert client.route() == {key: saved[key] for key in client.route()}
    assert not resolve(elsewhere)['provenance']['file']['loaded']
    monkeypatch.delenv('REFACTORY_WORKER_CONTEXT')
    assert DshModelClient(project=project).route()['worker_backend'] == 'claude_code'
    assert DshModelClient(project=elsewhere).route()['model'] == 'wrong'


@pytest.mark.parametrize('settings, match', [
    ({'worker_backend': 'missing'}, 'unknown worker backend'),
    ({'worker_backend': 'codex', 'provider': 'openai'}, 'provider/effort'),
    ({'worker_backend': 'claude_code', 'effort': 'high'}, 'provider/effort'),
    ({'worker_backend': 'codex', 'permission_mode': 'acceptEdits'}, 'permission'),
    ({'model': 42}, 'strings'), ({'api_key': 'not-a-real-key'}, 'unknown routing fields'),
    ({'node_routes': {'emit': []}}, 'overrides must be objects'),
    ({'role_routes': []}, 'must be an object'),
])
def test_invalid_defaults_never_fall_back(tmp_path, settings, match):
    write(tmp_path, settings)
    with pytest.raises(ValueError, match=match):
        DshModelClient(project=tmp_path)


def test_console_chat_and_terminal_preparation_share_project_route(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    write(project, {'node_routes': {'emit': {'worker_backend': 'codex'}}})
    opened = console._open_mode('default_pack/create', project=project, workdir=tmp_path / 'chat')
    chat = json.loads(opened['commandEnv']['REFACTORY_WORKER_CONTEXT'])
    interactive.prepare('default_pack/create', project=project, workdir=tmp_path / 'terminal')
    terminal = json.loads((tmp_path / 'terminal/.refactory/run-config.json').read_text())['model_context']
    assert chat == terminal
    for run in ('chat', 'terminal'):
        specs = [json.loads(p.read_text()) for p in (tmp_path / run / '.refactory/worker-specs').glob('*.json')]
        emit = next(spec for spec in specs if spec['node_id'] == 'emit')
        assert emit['model_context']['worker_backend'] == 'codex'
        assert all(spec['model_context']['worker_backend'] == 'dsh' for spec in specs if spec['node_id'] != 'emit')


def test_profiles_override_project_and_refresh_keeps_snapshot(tmp_path):
    write(tmp_path, {'worker_backend': 'codex'})
    first = profiles.resolve(tmp_path, 'counting_pack/fixed_step')
    assert first['settings']['worker_backend'] == 'codex'
    profiles.save(tmp_path, 'explicit', {'settings': {'worker_backend': 'dsh'}})
    selected = profiles.resolve(tmp_path, 'counting_pack/fixed_step', name='explicit')
    assert selected['settings']['worker_backend'] == 'dsh'
    write(tmp_path, {'worker_backend': 'claude_code'})
    refreshed = profiles.refresh(tmp_path, first['mode'], first['settings'], first)
    assert refreshed['settings'] == first['settings']
    assert refreshed['worker_configuration'] == first['worker_configuration']
    assert refreshed['context_hash'] == first['context_hash']


def test_profile_creation_captures_defaults_and_switch_clears_profile_model(tmp_path):
    write(tmp_path, {'worker_backend': 'dsh', 'provider': 'provider', 'model': 'dsh-model'})
    saved = profiles.save(tmp_path, 'from-file', {'settings': {}})
    assert saved['settings']['provider'] == 'provider'
    resolved = profiles.resolve(tmp_path, 'counting_pack/fixed_step', name='from-file',
                                overrides={'worker_backend': 'codex'})
    assert resolved['settings']['worker_backend'] == 'codex'
    assert resolved['settings']['provider'] == resolved['settings']['model'] == ''
    assert resolved['worker_configuration']['provenance']['fields']['worker_backend'] == 'explicit-launch'


def test_direct_mode_resume_uses_recorded_route_not_a_changed_file(tmp_path, monkeypatch):
    from refactory_packages.counting_pack.mode import FixedStep
    from refactory_dsh.nn import executor
    write(tmp_path, {'worker_backend': 'codex', 'permission_mode': 'default'})
    client = DshModelClient(project=tmp_path)
    run = tmp_path / 'run'
    FixedStep().run(run, model=client)
    path = run / executor.PROGRESS
    record = json.loads(path.read_text())
    assert record['model_context']['worker_backend'] == 'codex'
    record.update(node=FixedStep().graph().start, outcome='pending')
    path.write_text(json.dumps(record))
    write(tmp_path, {'worker_backend': 'claude_code'})
    seen = []
    monkeypatch.setattr(executor, 'execute', lambda *args, **kwargs: seen.append(kwargs['model'].context()))
    FixedStep.resume(run, model=DshModelClient(project=tmp_path))
    assert seen[0]['worker_backend'] == 'codex'
    assert seen[0]['permission_mode'] == 'default'


GATEWAY = {"apiKeyEnv": "LITELLM_API_KEY", "api": "anthropic-messages",
           "baseURL": "https://gateway.example/v1",
           "models": [{"id": "gateway-model", "name": "gateway-model"}]}


def test_provider_config_flows_to_context_and_worker_specs(tmp_path):
    write(tmp_path, {'provider': 'gateway', 'model': 'gateway-model',
                     'provider_config': {'gateway': GATEWAY}})
    client = DshModelClient(project=tmp_path)
    assert client.context()['provider_config'] == {'gateway': GATEWAY}
    assert client.context()['routing_provenance']['fields']['provider_config'] == 'project-file'
    # The declared route survives the subprocess boundary: a worker spec rebuilt
    # from the saved context resolves the same provider profile.
    rebuilt = DshModelClient(**client.context())
    assert rebuilt.context()['provider_config'] == {'gateway': GATEWAY}


def test_provider_config_survives_saved_and_nested_contexts(tmp_path, monkeypatch):
    project, elsewhere = tmp_path / 'project', tmp_path / 'artifact'
    write(project, {'provider_config': {'gateway': GATEWAY}})
    saved = DshModelClient(project=project).context()
    write(project, {'provider_config': {'other': GATEWAY}})
    write(elsewhere, {'provider_config': {'wrong': GATEWAY}})
    monkeypatch.setenv('REFACTORY_WORKER_CONTEXT', json.dumps(saved))
    assert DshModelClient().context()['provider_config'] == {'gateway': GATEWAY}
    monkeypatch.delenv('REFACTORY_WORKER_CONTEXT')
    assert DshModelClient(project=project).context()['provider_config'] == {'other': GATEWAY}


def test_explicit_launch_clears_project_provider_config(tmp_path):
    write(tmp_path, {'provider_config': {'gateway': GATEWAY}})
    # Route maps replace their lower layer as a whole, and so does the
    # provider-profile map: an explicit empty object is a real choice.
    explicit = DshModelClient(project=tmp_path, provider_config={})
    assert explicit.context()['provider_config'] == {}
    assert DshModelClient(project=tmp_path, provider_config={'launch': GATEWAY}).context() \
        ['provider_config'] == {'launch': GATEWAY}


@pytest.mark.parametrize('settings, match', [
    ({'provider_config': []}, 'must be an object keyed by nonempty provider names'),
    ({'provider_config': {'': GATEWAY}}, 'must be an object keyed by nonempty provider names'),
    ({'provider_config': {'gateway': 'https://gateway.example'}}, 'provider_config.gateway must be an object'),
    ({'provider_config': {'gateway': {'apiKey': 'not-a-real-key'}}}, 'credential value'),
    ({'provider_config': {'gateway': {'api_key': 'not-a-real-key'}}}, 'credential value'),
])
def test_invalid_provider_config_never_falls_back(tmp_path, settings, match):
    write(tmp_path, settings)
    with pytest.raises(ValueError, match=match):
        DshModelClient(project=tmp_path)
