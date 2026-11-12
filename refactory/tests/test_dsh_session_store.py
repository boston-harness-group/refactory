"""Each DSH process a run starts keeps its sessions in its own store, in the run.

DSH's JSONL session backend finds a session by probing every project directory under
its root, and its projection cache loads every stored session's record at start-up.
Under the one shared ``$DSH_HOME/sessions`` that cost grew with every run a machine had
done: 52k stored sessions made each helper 3.7 GB and filled the kernel's dentry cache
with billions of failed lookups, which took a shared host down.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from refactory_dsh import workers
from refactory_dsh.model import CompletionFailed, DshModelClient

ROOT = Path(__file__).resolve().parents[2]


def _rows(patch: Path) -> dict[str, dict]:
    return {row['id']: row for row in json.loads(patch.read_text()) if 'id' in row}


def _worker_patch(cmd: list[str]) -> Path:
    """The per-completion patch: the last ``--patch`` (the first is the shared node overlay)."""
    patches = [cmd[index + 1] for index, arg in enumerate(cmd) if arg == '--patch']
    return Path(patches[-1])


def _complete(monkeypatch, tmp_path, client, **kwargs):
    seen = {}

    def run(self, cmd, run_dir, stop_at=None, budget_s=None, task=None, target_workdir=None):
        seen['cmd'], seen['receipt'], seen['env'] = cmd, self._worker_receipt, self._environment()
        store = workers.session_store(self._worker_receipt, getattr(self, '_dsh_sessions', None))
        if store is not None:
            # What DSH writes there: a session log under its encoded cwd.
            (store / 'sessions' / '--cwd--' / 'session-1').mkdir(parents=True)
        return subprocess.CompletedProcess(cmd, 0, 'done', '')
    monkeypatch.setattr(DshModelClient, '_run_completion', run)
    assert client.complete('task', **kwargs) == 'done'
    return seen


def test_each_completion_keeps_its_sessions_beside_its_receipt(monkeypatch, tmp_path):
    monkeypatch.delenv(workers.SESSIONS_ENV, raising=False)
    home = tmp_path / 'home'
    seen = _complete(monkeypatch, tmp_path, DshModelClient(provider='p', model='m', dsh_home=home),
                     workdir=tmp_path / 'run')
    store = seen['receipt'].with_suffix('.dsh')
    assert store.parent == tmp_path / 'run' / '.refactory' / 'workers'
    rows = _rows(_worker_patch(seen['cmd']))
    assert rows['session-persistence-jsonl']['config'] == {'root': str(store / 'sessions')}
    assert rows['storage-json']['config'] == {'root': str(store / 'storages')}
    # DSH_HOME is untouched: credentials, settings and profiles stay the person's.
    assert seen['env']['DSH_HOME'] == str(home.resolve())
    # The store stays with the run, and a receipt reader still sees only receipts.
    assert (store / 'sessions').is_dir()
    assert [path.suffix for path in store.parent.glob('*.json')] == ['.json']


def test_a_completion_without_a_run_directory_deletes_its_store(monkeypatch, tmp_path):
    monkeypatch.delenv(workers.SESSIONS_ENV, raising=False)
    monkeypatch.setenv('REFACTORY_MODEL_SCRATCH', str(tmp_path))
    seen = _complete(monkeypatch, tmp_path, DshModelClient(provider='p', model='m'))
    store = seen['receipt'].with_suffix('.dsh')
    assert store.parent == tmp_path / 'refactory-model' / '.refactory' / 'workers'
    assert not store.exists()
    assert seen['receipt'].exists()


def test_the_coordinator_keeps_its_sessions_in_the_run(monkeypatch, tmp_path):
    monkeypatch.delenv(workers.SESSIONS_ENV, raising=False)
    telemetry = workers.GraphTelemetry(tmp_path, DshModelClient(provider='p', model='m').context(), None)
    store = telemetry.path.with_suffix('.dsh')
    rows = _rows(telemetry.patch)
    assert rows['session-persistence-jsonl']['config'] == {'root': str(store / 'sessions')}
    assert rows['storage-json']['config'] == {'root': str(store / 'storages')}


def test_shared_keeps_dshs_own_store_and_travels_to_dispatched_workers(monkeypatch, tmp_path):
    monkeypatch.setenv(workers.SESSIONS_ENV, 'shared')
    client = DshModelClient(provider='p', model='m')
    context = client.context()
    assert context['dsh_sessions'] == 'shared'
    # The choice rides the worker spec's context, which the dispatched worker is rebuilt
    # from, so it holds without the variable in the worker's environment.
    monkeypatch.delenv(workers.SESSIONS_ENV)
    rebuilt = DshModelClient(**context)
    assert rebuilt.context() == context
    seen = _complete(monkeypatch, tmp_path, rebuilt, workdir=tmp_path)
    rows = _rows(_worker_patch(seen['cmd']))
    assert 'session-persistence-jsonl' not in rows and 'storage-json' not in rows
    telemetry = workers.GraphTelemetry(tmp_path, context, None)
    assert 'session-persistence-jsonl' not in _rows(telemetry.patch)


def test_the_default_leaves_a_saved_context_and_the_evaluation_context_as_they_were(monkeypatch, tmp_path):
    monkeypatch.delenv(workers.SESSIONS_ENV, raising=False)
    # Worker specs are named by their bytes; a default context gains no field.
    assert 'dsh_sessions' not in DshModelClient(provider='p', model='m').context()
    # The run's evaluation context (what `--resume` compares) never names the store, so
    # neither the default nor the shared choice changes it: a run resumes either way.
    from refactory_dsh import profiles
    identities = []
    for choice in ('run', 'shared'):
        monkeypatch.setenv(workers.SESSIONS_ENV, choice)
        identities.append(profiles.runtime_identity({}, tmp_path, uses_models=True,
                                                    model_client=DshModelClient(provider='p', model='m')))
    assert identities[0] == identities[1]
    assert 'dsh_sessions' not in json.dumps(identities[0])


def test_a_node_command_takes_the_choice_from_the_context_it_inherits(monkeypatch):
    """A client built under an inherited worker context (`REFACTORY_WORKER_CONTEXT`) takes
    the choice from it when no explicit value is given."""
    monkeypatch.delenv(workers.SESSIONS_ENV, raising=False)
    monkeypatch.setenv('REFACTORY_WORKER_CONTEXT', json.dumps({'dsh_sessions': 'shared'}))
    assert DshModelClient(provider='p', model='m').context()['dsh_sessions'] == 'shared'


def test_the_coordinator_follows_the_context_not_the_environment(monkeypatch, tmp_path):
    """A resumed spine run whose saved context is the default keeps per-process stores for
    its coordinator, as its node commands do, whatever the resuming shell sets."""
    monkeypatch.delenv(workers.SESSIONS_ENV, raising=False)
    saved = DshModelClient(provider='p', model='m').context()
    monkeypatch.setenv(workers.SESSIONS_ENV, 'shared')
    telemetry = workers.GraphTelemetry(tmp_path, saved, None)
    assert _rows(telemetry.patch)['session-persistence-jsonl']['config'] == {
        'root': str(telemetry.path.with_suffix('.dsh') / 'sessions')}


def _spine_without_a_client(monkeypatch, tmp_path):
    """Start `run_on_dsh` with no model client and return the argv it spawns."""
    from refactory_dsh import dsh_profile, registry
    from refactory_dsh.dsh_runtime import run_on_dsh
    from refactory_dsh.paths import REPO_ROOT
    from tests_support import _profile

    class Process:
        stdout, stderr, returncode = iter(()), iter(()), None

        def wait(self, timeout=None):
            self.returncode = 1
            return 1

    state = registry.resolve(REPO_ROOT, "circle_packing/adaptive")[1]().state_dict()
    home = tmp_path / "home"
    _profile(home, "graph", ["@deepseek-ai/dsh-base", dsh_profile.GRAPH_BUNDLE])
    monkeypatch.setenv("DSH_HOME", str(home))
    monkeypatch.delenv("REFACTORY_MODEL_DSH_HOME", raising=False)
    seen = {}

    def capture(command, **kwargs):
        seen["command"] = command
        return Process()
    monkeypatch.setattr("refactory_dsh.dsh_runtime.subprocess.Popen", capture)
    run_on_dsh(state, tmp_path / "run", name="probe")
    return seen["command"]


def test_a_coordinator_started_without_a_client_keeps_its_own_store(monkeypatch, tmp_path):
    monkeypatch.delenv(workers.SESSIONS_ENV, raising=False)
    argv = _spine_without_a_client(monkeypatch, tmp_path)
    assert argv.index("--patch") < argv.index("--profile")
    patch = Path(argv[argv.index("--patch") + 1])
    store = patch.parent / patch.name.replace(".patch.yml", ".dsh")
    assert patch.parent == tmp_path / "run" / ".refactory" / "workers"
    assert _rows(patch) == {
        'session-persistence-jsonl': {'id': 'session-persistence-jsonl', 'config': {'root': str(store / 'sessions')}},
        'storage-json': {'id': 'storage-json', 'config': {'root': str(store / 'storages')}}}
    # It is not a receipt: receipt readers still see none.
    assert not list(patch.parent.glob('*.json'))


def test_each_client_less_coordinator_gets_its_own_store(monkeypatch, tmp_path):
    """Two coordinators of one run directory (a rerun, a resume) must not share a store:
    a shared store is the cost this avoids, and a second process would take the first's
    session lock."""
    monkeypatch.delenv(workers.SESSIONS_ENV, raising=False)
    first, second = workers.session_patch(tmp_path), workers.session_patch(tmp_path)
    assert first != second
    assert _rows(first)['session-persistence-jsonl'] != _rows(second)['session-persistence-jsonl']
    assert _rows(first)['storage-json'] != _rows(second)['storage-json']


def test_a_coordinator_without_a_client_honours_shared(monkeypatch, tmp_path):
    monkeypatch.setenv(workers.SESSIONS_ENV, 'shared')
    assert "--patch" not in _spine_without_a_client(monkeypatch, tmp_path)


def test_an_unknown_choice_fails_before_launch(monkeypatch):
    monkeypatch.setenv(workers.SESSIONS_ENV, 'elsewhere')
    with pytest.raises(ValueError, match=workers.SESSIONS_ENV):
        DshModelClient(provider='p', model='m')


def test_real_dsh_writes_the_session_into_the_run_and_nothing_into_dsh_home(monkeypatch, tmp_path):
    """The pinned DSH, booted the way a completion boots it, honours the patch.

    The provider is one no adapter serves, so the run fails before any model call, after
    DSH has stored the session it opened.
    """
    if not (ROOT / 'node_modules/@deepseek-ai/dsh/lib/bin.js').exists():
        pytest.skip('DSH packages not installed')
    monkeypatch.delenv(workers.SESSIONS_ENV, raising=False)
    monkeypatch.delenv('DSH_CLI', raising=False)
    home, run = tmp_path / 'home', tmp_path / 'run'
    home.mkdir()
    client = DshModelClient(provider='no-such-provider', model='none', dsh_home=home,
                            runtime_root=ROOT, dsh_cli='', attempts=1, timeout_s=120)
    with pytest.raises(CompletionFailed):
        client.complete('say hi', workdir=run)
    [receipt] = (run / '.refactory' / 'workers').glob('*.json')
    store = receipt.with_suffix('.dsh')
    assert list((store / 'sessions').glob('*/session-*/session.*.jsonl*'))
    assert list((store / 'storages' / 'session_projcache' / 'sessions').glob('session-*.json'))
    assert not (home / 'sessions').exists() and not (home / 'storages').exists()
