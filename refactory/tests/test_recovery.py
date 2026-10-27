import hashlib
import json
import threading
from types import SimpleNamespace

import pytest

from refactory_dsh import recovery
from refactory_dsh.run_ownership import ownership


def saved_run(root, status='paused'):
    runtime = root / '.refactory'
    runtime.mkdir()
    source = root / 'mode.py'
    source.write_text('answer = 42\n')
    config = {'mode': 'demo/run', 'project': str(root),
              'configuration_fingerprints': {str(source): recovery.fingerprint(source)}}
    (runtime / 'run-config.json').write_text(json.dumps(config))
    payload = {'run_id': str(root), 'graph': {'name': 'demo-run'}, 'status': status,
               'file_fingerprints': {}}
    (runtime / 'spine-checkpoint.json').write_text(json.dumps(payload))
    return source


def test_start_releases_ownership_before_launch(monkeypatch, tmp_path):
    saved_run(tmp_path)

    def popen(argv, **kwargs):
        acquired, errors = [], []

        def child():
            try:
                with ownership(tmp_path):
                    acquired.append(True)
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=child)
        thread.start()
        thread.join()
        assert not errors
        assert acquired
        assert 'refactory_dsh.supervisor' in argv
        return SimpleNamespace(pid=123)

    monkeypatch.setattr('subprocess.Popen', popen)
    assert recovery.start(tmp_path)['pid'] == 123


@pytest.mark.parametrize('status', ['executing', 'unexpected', None])
def test_unknown_outcome_never_produces_resume_request(tmp_path, status):
    saved_run(tmp_path, status)
    with pytest.raises(ValueError, match='inspect side effects'):
        recovery.prepare(tmp_path, 'demo-run')
    assert not (tmp_path / '.refactory/resume-request.json').exists()


def test_failed_status_refused_without_acknowledge_flag(tmp_path):
    saved_run(tmp_path, 'failed')
    with pytest.raises(ValueError, match='resume --acknowledge-failure'):
        recovery.prepare(tmp_path, 'demo-run')
    assert not (tmp_path / '.refactory/resume-request.json').exists()


def test_failed_status_accepted_with_acknowledge_flag(tmp_path):
    saved_run(tmp_path, 'failed')
    recovery.prepare(tmp_path, 'demo-run', acknowledge_failed=True)
    request = json.loads((tmp_path / '.refactory/resume-request.json').read_text())
    checkpoint = tmp_path / '.refactory/spine-checkpoint.json'
    assert request['checkpoint_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert request['acknowledge_failed'] is True


def test_child_prepare_honours_acknowledged_failed_from_file(tmp_path):
    saved_run(tmp_path, 'failed')
    # Parent's call: writes resume-request.json with acknowledge_failed: true
    recovery.prepare(tmp_path, 'demo-run', acknowledge_failed=True)
    request_path = tmp_path / '.refactory/resume-request.json'
    assert request_path.exists()
    parent_request = json.loads(request_path.read_text())
    assert parent_request['acknowledge_failed'] is True

    # Child's call: no acknowledge_failed argument, reads from file
    recovery.prepare(tmp_path, 'demo-run')
    child_request = json.loads(request_path.read_text())
    assert child_request['acknowledge_failed'] is True
    checkpoint = tmp_path / '.refactory/spine-checkpoint.json'
    assert child_request['checkpoint_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def test_child_prepare_refuses_failed_without_file_acknowledgement(tmp_path):
    saved_run(tmp_path, 'failed')
    # No pre-existing resume-request.json, no acknowledge_failed argument
    with pytest.raises(ValueError, match='resume --acknowledge-failure'):
        recovery.prepare(tmp_path, 'demo-run')
    assert not (tmp_path / '.refactory/resume-request.json').exists()


def test_child_prepare_refuses_stale_acknowledgement(tmp_path):
    """A resume-request.json whose checkpoint_sha256 does not match the current
    checkpoint bytes is not honoured — the request belongs to a different checkpoint."""
    saved_run(tmp_path, 'failed')
    # Write a resume-request.json with acknowledge_failed but a wrong sha256
    request_path = tmp_path / '.refactory/resume-request.json'
    request_path.write_text(json.dumps({
        'checkpoint_sha256': 'stale-sha256-that-does-not-match',
        'acknowledge_failed': True,
    }))
    with pytest.raises(ValueError, match='resume --acknowledge-failure'):
        recovery.prepare(tmp_path, 'demo-run')


def test_changed_source_and_unknown_worker_refuse_recovery(tmp_path):
    source = saved_run(tmp_path)
    source.write_text('answer = 0\n')
    with pytest.raises(ValueError, match='configuration.*changed'):
        recovery.prepare(tmp_path, 'demo-run')
    source.write_text('answer = 42\n')
    pending = tmp_path / '.refactory/agent-delivery/pending'
    pending.mkdir(parents=True)
    (pending / 'worker.json').write_text('{}')
    with pytest.raises(ValueError, match='worker outcome is unknown'):
        recovery.prepare(tmp_path, 'demo-run')


def test_recovery_request_binds_validated_bytes(tmp_path):
    saved_run(tmp_path)
    recovery.prepare(tmp_path, 'demo-run')
    checkpoint = tmp_path / '.refactory/spine-checkpoint.json'
    request = json.loads((tmp_path / '.refactory/resume-request.json').read_text())
    assert request['checkpoint_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def test_receipt_with_no_pending_marker_does_not_block_recovery(tmp_path):
    """A refused step whose receipt was written does not block recovery.

    Only a dispatch with no receipt at all (a pending marker with no corresponding
    finished receipt) is 'unknown'. A refused step has both a receipt and — after the
    fix — no pending marker, so recovery proceeds.
    """
    saved_run(tmp_path)
    # Create a delivery record WITH a receipt but WITHOUT a pending marker
    delivery_dir = tmp_path / '.refactory' / 'agent-delivery' / 'test-step-id'
    delivery_dir.mkdir(parents=True)
    (delivery_dir / 'receipt.json').write_text(json.dumps({
        'id': 'test-step-id', 'node': 'design_doc', 'status': 'failed',
        'started': '2026-10-17T00:00:00+00:00',
        'finished': '2026-10-17T00:01:00+00:00',
        'artifacts': {}, 'writes': [],
    }))
    # No pending marker exists — recovery should succeed
    recovery.prepare(tmp_path, 'demo-run')
    assert (tmp_path / '.refactory/resume-request.json').exists()


def test_configuration_fingerprints_ignore_only_interpreter_cache(monkeypatch, tmp_path):
    from refactory_dsh import registry, lowering
    package = tmp_path / 'package'
    package.mkdir()
    (package / 'mode.py').write_text('a=1')
    (tmp_path / 'graphs').mkdir()
    declarations = tmp_path / lowering.DECLARATIONS
    declarations.parent.mkdir(exist_ok=True)
    declarations.write_text('{}')
    monkeypatch.setattr(registry, 'where', lambda *args: package)
    class Mode:
        pass
    monkeypatch.setattr(registry, 'resolve', lambda *args: (SimpleNamespace(ops=()), Mode))
    before = recovery.configuration_fingerprints(tmp_path, 'package/run', tmp_path)
    (package / '__pycache__').mkdir()
    (package / '__pycache__/mode.pyc').write_bytes(b'cache')
    assert recovery.configuration_fingerprints(tmp_path, 'package/run', tmp_path) == before
    (package / 'data.json').write_text('{}')
    assert recovery.configuration_fingerprints(tmp_path, 'package/run', tmp_path) != before

def test_set_aside_never_moves_the_runs_own_configuration(tmp_path):
    """The configuration is identity, not an output: a changed one stays in place for the
    resume to refuse, while a half-written output beside it is still set aside."""
    saved_run(tmp_path, 'failed')
    config = tmp_path / '.refactory/run-config.json'
    config.write_text('{"mode": "demo/run", "knobs": ["edited"]}')
    output = tmp_path / 'design.md'
    output.write_text('half written\n')
    directory, files = recovery.set_aside(
        tmp_path, fingerprints={'.refactory/run-config.json': 'stale', 'design.md': 'stale'},
        markers=[], status='failed', owner_pid=None)
    assert [entry['original_path'] for entry in files] == ['design.md']
    assert config.is_file()
    assert not output.exists()
    assert (directory / 'outputs/design.md').is_file()
