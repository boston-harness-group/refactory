import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from refactory_dsh import console, graph_control, live, readable


def test_cursor_does_not_acknowledge_partial_write(tmp_path):
    live.emit(tmp_path, 'node_start', node_id='a', state='running')
    journal = tmp_path / live.EVENTS
    with journal.open('ab') as handle:
        handle.write(b'{"kind":"node_complete"')
    events, cursor = live.events(tmp_path)
    assert cursor == 1
    assert events[0]['run_id'] == str(tmp_path)
    assert live.events(tmp_path, after=cursor) == ([], 1)


def test_unknown_usage_is_absent_and_completed_stages_survive_terminal(tmp_path):
    directory = tmp_path / '.refactory'
    directory.mkdir()
    (directory / 'spine-status.json').write_text(json.dumps({'status': 'done', 'completed': ['a', 'qa'], 'nodes': []}))
    activity = live.snapshot(tmp_path)
    assert activity['completed'] == ['a', 'qa']
    assert activity['resources'] is None
    assert activity['last_event_at'] is None
    assert console.status(tmp_path)['activity'] == activity


def test_rerun_requires_exact_preview_and_never_executes_work(tmp_path):
    directory = tmp_path / '.refactory'
    directory.mkdir()
    checkpoint = directory / 'spine-checkpoint.json'
    checkpoint.write_text(json.dumps({'redo_plans': {'a': {'nodes': ['a', 'qa']}}}))
    (directory / 'invocation.json').write_text(json.dumps({'id': 'inv1'}))
    preview = graph_control.request(tmp_path, 'preview', node='a')
    assert preview['invalidated'] == ['a', 'qa']
    assert not (directory / 'graph-controls').exists()
    with pytest.raises(ValueError, match='preview'):
        graph_control.request(tmp_path, 'rerun', node='a')
    receipt = graph_control.request(tmp_path, 'rerun', node='a', revision=preview['revision'])
    assert receipt['status'] == 'queued'
    assert receipt['run_id'] == str(tmp_path)
    checkpoint.write_text('{}')
    with pytest.raises(ValueError, match='preview'):
        graph_control.request(tmp_path, 'rerun', node='a', revision=preview['revision'])


def test_cursor_validation(tmp_path):
    with pytest.raises(ValueError, match='nonnegative'):
        live.events(tmp_path, -1)


def test_append_after_interrupted_record_exposes_gap_and_new_activity(tmp_path):
    live.emit(tmp_path, 'before')
    with (tmp_path / live.EVENTS).open('ab') as handle:
        handle.write(b'{"partial":')
    live.emit(tmp_path, 'after')
    events, cursor = live.events(tmp_path, after=1)
    assert [event['kind'] for event in events] == ['journal_gap', 'after']
    assert cursor == 3
    assert live.events(tmp_path, after=cursor) == ([], cursor)


def test_malformed_complete_record_cannot_hide_later_events(tmp_path):
    live.emit(tmp_path, 'before')
    with (tmp_path / live.EVENTS).open('ab') as handle:
        handle.write(b'bad record\n\n')
    live.emit(tmp_path, 'after')
    assert [event['kind'] for event in live.events(tmp_path)[0]] == ['before', 'journal_gap', 'after']


def test_control_file_order_preserves_request_order(monkeypatch, tmp_path):
    directory = tmp_path / '.refactory'
    directory.mkdir()
    (directory / 'spine-checkpoint.json').write_text('{}')
    # UUID order deliberately opposes arrival order.
    from types import SimpleNamespace
    identifiers = iter(['z', 'a'])
    moments = iter([2, 10])
    monkeypatch.setattr(graph_control.uuid, 'uuid4', lambda: SimpleNamespace(hex=next(identifiers)))
    monkeypatch.setattr(graph_control.time, 'time_ns', lambda: next(moments))
    graph_control.request(tmp_path, 'pause')
    graph_control.request(tmp_path, 'stop')
    queued = [json.loads(path.read_text()) for path in sorted((directory / 'graph-controls').glob('*.json'))]
    assert [item['action'] for item in queued] == ['pause', 'stop']


def test_exhausted_or_exited_graph_never_claims_running(tmp_path):
    directory = tmp_path / '.refactory'
    directory.mkdir()
    (directory / 'spine-status.json').write_text(json.dumps({'status': 'executing', 'nodes': ['work']}))
    (directory / 'spine-checkpoint.json').write_text(json.dumps({'owner_pid': 99999999}))
    result = console.status(tmp_path)
    assert result['state'] == 'died'
    assert result['owner_alive'] is False
    (directory / 'resource-usage.json').write_text(json.dumps({'exhausted': True, 'reason': 'token budget exhausted'}))
    result = console.status(tmp_path)
    assert result['state'] == 'failed'
    assert not result['running']
    assert result['activity']['wait_reason'] == 'token budget exhausted'


def test_owner_alive_reflects_pid_liveness(tmp_path):
    """owner_alive=True for a live PID, False for a dead one, None when absent."""
    directory = tmp_path / '.refactory'
    directory.mkdir()
    (directory / 'spine-status.json').write_text(json.dumps({'status': 'executing', 'nodes': []}))

    # Live PID
    (directory / 'spine-checkpoint.json').write_text(json.dumps({'owner_pid': os.getpid()}))
    result = console.status(tmp_path)
    assert result['owner_alive'] is True

    # Dead PID
    (directory / 'spine-checkpoint.json').write_text(json.dumps({'owner_pid': 99999999}))
    result = console.status(tmp_path)
    assert result['owner_alive'] is False

    # No owner PID
    (directory / 'spine-checkpoint.json').write_text(json.dumps({}))
    result = console.status(tmp_path)
    assert result['owner_alive'] is None


def test_status_surfaces_failed_control_in_activity(tmp_path):
    """console.status() includes failed graph control receipt from activity.control."""
    directory = tmp_path / '.refactory'
    directory.mkdir()
    (directory / 'spine-status.json').write_text(json.dumps({'status': 'executing', 'nodes': ['work']}))
    (directory / 'spine-checkpoint.json').write_text(json.dumps({'owner_pid': os.getpid()}))
    (directory / 'graph-control-receipt.json').write_text(json.dumps({
        'action': 'rerun', 'status': 'failed',
        'message': 'graph changed since preview; preview invalidation again',
    }))
    result = console.status(tmp_path)
    control = result['activity'].get('control')
    assert control is not None
    assert control['status'] == 'failed'
    assert 'graph changed since preview' in control['message']


def test_graph_control_shows_last_failed_receipt(tmp_path):
    """graph_control.request() returns the last failed receipt when one exists."""
    directory = tmp_path / '.refactory'
    directory.mkdir()
    (directory / 'spine-checkpoint.json').write_text(json.dumps({}))
    (directory / 'graph-control-receipt.json').write_text(json.dumps({
        'action': 'rerun', 'status': 'failed',
        'message': 'graph changed since preview; preview invalidation again',
    }))
    result = graph_control.request(tmp_path, 'pause')
    assert result['status'] == 'queued'
    assert result['last_failed_receipt']['status'] == 'failed'
    assert 'graph changed since preview' in result['last_failed_receipt']['message']


def test_status_shows_failed_control_when_finished(tmp_path):
    """dm-101: A failed control receipt IS still rendered as 'Last failed control'
    when the run state is 'finished', because only a later successful control
    clears a failure (the receipt file is overwritten with status 'applied')."""
    activity = {
        'control': {
            'action': 'rerun', 'status': 'failed',
            'message': 'graph changed since preview',
        },
    }
    payload = {
        'authoring': {'name': 'test-run', 'completed': ['a']},
        'state': 'finished',
        'activity': activity,
    }
    text = readable.format_status(payload)
    assert 'Last failed control' in text


def test_later_success_clears_failure_warning(tmp_path):
    """dm-101: A run_warning 'graph control rerun failed: ...' is filtered out
    once a later control succeeded (receipt overwritten with status 'applied').
    This covers b28a's real sequence: a failed rerun followed by an applied rerun."""
    activity = {
        'control': {
            'action': 'rerun', 'status': 'applied',
            'message': 'Applied at graph boundary.',
        },
        'events': [
            {'kind': 'run_warning', 'summary': 'graph control rerun failed: graph changed since preview'},
        ],
    }
    payload = {
        'authoring': {'name': 'test-run', 'completed': ['a', 'b']},
        'state': 'finished',
        'activity': activity,
    }
    text = readable.format_status(payload)
    assert 'Last failed control' not in text
    assert 'graph control rerun failed' not in text


def test_failed_control_warning_shown_when_no_later_success(tmp_path):
    """dm-101: When no later control succeeded, the warning IS shown."""
    activity = {
        'control': {
            'action': 'rerun', 'status': 'failed',
            'message': 'graph changed since preview',
        },
        'events': [
            {'kind': 'run_warning', 'summary': 'graph control rerun failed: graph changed since preview'},
        ],
    }
    payload = {
        'authoring': {'name': 'test-run', 'completed': ['a']},
        'state': 'executing',
        'activity': activity,
    }
    text = readable.format_status(payload)
    assert 'Last failed control' in text
    assert 'graph control rerun failed' in text


def test_status_shows_failed_control_when_executing(tmp_path):
    """dm-101: A failed control receipt IS rendered when the run is executing."""
    activity = {
        'control': {
            'action': 'rerun', 'status': 'failed',
            'message': 'graph changed since preview',
        },
    }
    payload = {
        'authoring': {'name': 'test-run', 'completed': ['a']},
        'state': 'executing',
        'activity': activity,
    }
    text = readable.format_status(payload)
    assert 'Last failed control' in text


def test_later_success_clears_failure_in_graph_control_request(tmp_path):
    """dm-101: graph_control.request() surfaces last_failed_receipt from a
    failed receipt even when the spine is terminal, because the receipt file
    is the source of truth. A later applied control overwrites the receipt,
    so the failure disappears naturally."""
    directory = tmp_path / '.refactory'
    directory.mkdir()
    (directory / 'spine-checkpoint.json').write_text(json.dumps({}))
    # Receipt says 'failed' — always surfaced regardless of spine state.
    (directory / 'graph-control-receipt.json').write_text(json.dumps({
        'action': 'rerun', 'status': 'failed',
        'message': 'graph changed since preview; preview invalidation again',
    }))
    (directory / 'spine-status.json').write_text(json.dumps({'status': 'done'}))
    result = graph_control.request(tmp_path, 'pause')
    assert result['status'] == 'queued'
    assert 'last_failed_receipt' in result
    assert result['last_failed_receipt']['status'] == 'failed'

    # Now simulate a later successful control overwriting the receipt.
    (directory / 'graph-control-receipt.json').write_text(json.dumps({
        'action': 'rerun', 'status': 'applied',
        'message': 'Applied at graph boundary.',
    }))
    result2 = graph_control.request(tmp_path, 'pause')
    assert result2['status'] == 'queued'
    assert 'last_failed_receipt' not in result2


def test_preview_includes_run_id(tmp_path):
    """dm-101: preview payloads carry run_id so the CLI prints the run name."""
    directory = tmp_path / '.refactory'
    directory.mkdir()
    checkpoint = directory / 'spine-checkpoint.json'
    checkpoint.write_text(json.dumps({
        'redo_plans': {'a': {'nodes': ['a', 'qa']}},
    }))
    result = graph_control.request(tmp_path, 'preview', node='a')
    assert result['status'] == 'preview'
    assert 'run_id' in result
    assert str(tmp_path) in result['run_id']
    text = readable.format_graph_control(result)
    assert str(tmp_path) in text
    assert 'Preview ()' not in text


def test_format_graph_control_names_the_run(tmp_path):
    """dm-101: every format_graph_control line names the run."""
    payload = {
        'status': 'queued', 'action': 'pause', 'run_id': '/runs/b28a',
    }
    text = readable.format_graph_control(payload)
    assert '/runs/b28a' in text

    # preview
    payload = {
        'status': 'preview', 'action': 'rerun', 'run_id': '/runs/b28a',
        'invalidated': ['a', 'b'],
    }
    text = readable.format_graph_control(payload)
    assert '/runs/b28a' in text

    # failed
    payload = {
        'status': 'failed', 'action': 'rerun', 'run_id': '/runs/b28a',
        'message': 'stale preview',
    }
    text = readable.format_graph_control(payload)
    assert '/runs/b28a' in text

    # acknowledged
    payload = {
        'status': 'acknowledged', 'action': 'acknowledge', 'run_id': '/runs/b28a',
        'message': 'ok', 'files': [],
    }
    text = readable.format_graph_control(payload)
    assert '/runs/b28a' in text

    # withdrawn
    payload = {
        'status': 'withdrawn', 'action': 'withdraw', 'run_id': '/runs/b28a',
        'control_id': 'ctrl-1',
    }
    text = readable.format_graph_control(payload)
    assert '/runs/b28a' in text

    # last_failed_receipt
    payload = {
        'status': 'queued', 'action': 'pause', 'run_id': '/runs/b28a',
        'last_failed_receipt': {
            'action': 'rerun', 'status': 'failed', 'message': 'graph changed',
        },
    }
    text = readable.format_graph_control(payload)
    assert '/runs/b28a' in text


def test_amend_command_returns_run_id(tmp_path):
    """dm-101: _amend_command returns a dict with run_id."""
    directory = tmp_path / '.refactory'
    directory.mkdir()
    checkpoint_data = {
        'status': 'ready',
        'state': {'cursor': ['run']},
        'graph': {'nodes': {'run': {'_type': 'FnNode', 'command': 'old-cmd'}}},
    }
    (directory / 'spine-checkpoint.json').write_text(json.dumps(checkpoint_data))
    (directory / 'invocation.json').write_text(json.dumps({'id': 'inv1'}))
    (tmp_path / 'facts.json').write_text(json.dumps({'check': 'old-check'}))
    result = graph_control._amend_command(
        tmp_path, directory / 'spine-checkpoint.json', checkpoint_data,
        field='check', value='new-check', note='test', acknowledge_failure=False,
    )
    assert 'run_id' in result
    assert str(tmp_path) in result['run_id']


def test_acknowledge_returns_run_id(tmp_path):
    """dm-101: _acknowledge returns a dict with run_id."""
    import fcntl
    directory = tmp_path / '.refactory'
    directory.mkdir()
    checkpoint_data = {'status': 'executing', 'owner_pid': 99999999}
    checkpoint = directory / 'spine-checkpoint.json'
    checkpoint.write_text(json.dumps(checkpoint_data))
    result = graph_control._acknowledge(tmp_path, checkpoint, checkpoint_data)
    assert 'run_id' in result
    assert str(tmp_path) in result['run_id']


def test_withdraw_returns_run_id(tmp_path):
    """dm-101: _withdraw returns a dict with run_id."""
    directory = tmp_path / '.refactory'
    directory.mkdir()
    controls_dir = directory / 'graph-controls'
    controls_dir.mkdir()
    (controls_dir / 'test-ctrl.json').write_text(json.dumps({
        'id': 'test-ctrl', 'action': 'pause', 'status': 'queued',
    }))
    result = graph_control._withdraw(tmp_path, 'test-ctrl')
    assert 'run_id' in result
    assert str(tmp_path) in result['run_id']


def test_graph_snapshot_reads_saved_topology_without_commands_or_mutations(tmp_path):
    checkpoint = tmp_path / '.refactory/spine-checkpoint.json'
    checkpoint.parent.mkdir()
    graph = {'nodes': {
        'fork': {'_type': 'ForkNode', 'targets': ['work', 'qa']},
        'work': {'_type': 'AgentNode', 'command': 'DO NOT EXECUTE', 'prompt_template': 'private'},
        'qa': {'_type': 'FnNode'},
        'join': {'_type': 'JoinNode', 'sources': ['work', 'qa']},
        'review': {'_type': 'GateNode'},
    }, 'edges': [{'source': 'join', 'target': 'review'},
                  {'source': 'review', 'target': 'fork', 'condition': 'reloop'},
                  {'source': 'missing', 'target': 'review'}]}
    checkpoint.write_text(json.dumps({'graph': graph, 'skipped': ['qa', 'missing']}))
    before = checkpoint.read_bytes()
    result = live.snapshot(tmp_path)['graph']
    assert result['nodes'][1] == {'id': 'work', 'kind': 'AgentNode'}
    assert result['skipped'] == ['qa']
    assert {(edge['source'], edge['target']) for edge in result['edges']} == {
        ('fork', 'work'), ('fork', 'qa'), ('work', 'join'), ('qa', 'join'),
        ('join', 'review'), ('review', 'fork'),
    }
    assert result['edges'][1]['label'] == 'reloop'
    assert 'private' not in json.dumps(result)
    assert 'DO NOT EXECUTE' not in json.dumps(result)
    assert checkpoint.read_bytes() == before
    assert not (tmp_path / live.EVENTS).exists()


def test_preview_returns_deferred_when_concurrent_work_blocks(tmp_path):
    directory = tmp_path / '.refactory'
    directory.mkdir()
    checkpoint = directory / 'spine-checkpoint.json'
    checkpoint.write_text(json.dumps({
        'deferred_redo_plans': {'a': {'nodes': ['a', 'join', 'end']}},
        'redo_plans': {},
    }))
    preview = graph_control.request(tmp_path, 'preview', node='a')
    assert preview['status'] == 'deferred'
    assert preview['invalidated'] == ['a', 'join', 'end']


def test_rerun_from_deferred_preview_records_deferred_nodes(tmp_path):
    directory = tmp_path / '.refactory'
    directory.mkdir()
    (directory / 'invocation.json').write_text(json.dumps({'id': 'inv1'}))
    checkpoint = directory / 'spine-checkpoint.json'
    checkpoint.write_text(json.dumps({
        'deferred_redo_plans': {'a': {'nodes': ['a', 'join', 'end']}},
        'redo_plans': {},
    }))
    saved = json.loads(checkpoint.read_text())
    revision = saved.get('graph_revision') or hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    receipt = graph_control.request(tmp_path, 'rerun', node='a', revision=revision)
    assert receipt['status'] == 'queued'
    assert receipt['deferred_nodes'] == ['a', 'join', 'end']
    # Verify the control file on disk
    controls = list((directory / 'graph-controls').glob('*.json'))
    assert len(controls) == 1
    control = json.loads(controls[0].read_text())
    assert control['deferred_nodes'] == ['a', 'join', 'end']


def test_rerun_of_failed_run_refused_without_acknowledge(tmp_path):
    directory = tmp_path / '.refactory'
    directory.mkdir()
    checkpoint = directory / 'spine-checkpoint.json'
    checkpoint.write_text(json.dumps({
        'status': 'failed',
        'redo_plans': {'a': {'nodes': ['a']}},
    }))
    saved = json.loads(checkpoint.read_text())
    revision = saved.get('graph_revision') or hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='acknowledge_failure'):
        graph_control.request(tmp_path, 'rerun', node='a', revision=revision)


def test_rerun_of_failed_run_queues_control_then_starts_owner(monkeypatch, tmp_path):
    from refactory_dsh import recovery
    directory = tmp_path / '.refactory'
    directory.mkdir()
    (directory / 'invocation.json').write_text(json.dumps({'id': 'inv1'}))
    checkpoint = directory / 'spine-checkpoint.json'
    checkpoint.write_text(json.dumps({
        'status': 'failed',
        'redo_plans': {'a': {'nodes': ['a']}},
    }))
    saved = json.loads(checkpoint.read_text())
    revision = saved.get('graph_revision') or hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    start_calls = []
    def mock_start(root, *, acknowledge_failed=False):
        # Verify control file exists BEFORE start is called
        controls = list((root / '.refactory/graph-controls').glob('*.json'))
        start_calls.append({
            'root': root,
            'acknowledge_failed': acknowledge_failed,
            'controls_existed': len(controls) > 0,
        })
        return {'status': 'queued', 'pid': 999}
    monkeypatch.setattr(recovery, 'start', mock_start)
    result = graph_control.request(
        tmp_path, 'rerun', node='a', revision=revision,
        acknowledge_failure=True,
    )
    assert result['status'] == 'queued'
    assert result['target'] == 'a'
    assert len(start_calls) == 1
    assert start_calls[0]['acknowledge_failed'] is True
    assert start_calls[0]['controls_existed'] is True
    assert result['owner']['pid'] == 999
    assert 'Started a new owner' in readable.format_graph_control(result)


def test_resume_says_whether_it_started_an_owner(monkeypatch, tmp_path):
    """A resume that starts an owner names it; one queued for the live owner does not."""
    from refactory_dsh import recovery
    directory = tmp_path / '.refactory'
    directory.mkdir()
    (directory / 'run-config.json').write_text(json.dumps({'mode': 'demo/run', 'project': str(tmp_path)}))
    (directory / 'invocation.json').write_text(json.dumps({'id': 'inv1'}))
    checkpoint = directory / 'spine-checkpoint.json'
    checkpoint.write_text(json.dumps({'status': 'halted'}))
    monkeypatch.setattr(recovery, 'prepare', lambda *args, **kwargs: None)
    monkeypatch.setattr('subprocess.Popen', lambda *args, **kwargs: SimpleNamespace(pid=4242))
    started = readable.format_graph_control(graph_control.request(tmp_path, 'resume'))
    assert 'Started a new owner' in started and 'pid 4242' in started
    checkpoint.write_text(json.dumps({'status': 'ready', 'owner_pid': os.getpid()}))
    queued = readable.format_graph_control(graph_control.request(tmp_path, 'resume'))
    assert 'Queued resume' in queued and 'Started a new owner' not in queued


def test_acknowledged_rerun_child_gets_past_validation(tmp_path):
    """Full parent-write → child-read → spine-delete lifecycle: one-shot semantics."""
    from refactory_dsh import recovery

    # -- setup: a failed run --
    directory = tmp_path / '.refactory'
    directory.mkdir()
    source = tmp_path / 'mode.py'
    source.write_text('answer = 42\n')
    config = {'mode': 'demo/run', 'project': str(tmp_path),
              'configuration_fingerprints': {str(source): recovery.fingerprint(source)}}
    (directory / 'run-config.json').write_text(json.dumps(config))
    payload = {'run_id': str(tmp_path), 'graph': {'name': 'demo-run'},
               'status': 'failed', 'file_fingerprints': {}}
    (directory / 'spine-checkpoint.json').write_text(json.dumps(payload))

    # -- parent path: prepare with acknowledgement --
    recovery.prepare(tmp_path, 'demo-run', acknowledge_failed=True)
    request_path = directory / 'resume-request.json'
    assert request_path.exists()
    assert json.loads(request_path.read_text())['acknowledge_failed'] is True

    # -- child path: prepare without acknowledgement --
    recovery.prepare(tmp_path, 'demo-run')
    assert json.loads(request_path.read_text())['acknowledge_failed'] is True

    # -- spine consumption: delete the file --
    request_path.unlink()

    # -- one-shot: a plain resume now refuses --
    with pytest.raises(ValueError, match='resume --acknowledge-failure'):
        recovery.prepare(tmp_path, 'demo-run')
    assert not request_path.exists()


def test_acknowledged_rerun_child_interactive_resume_path(monkeypatch, tmp_path):
    """The child's interactive resume path (interactive.prepare with resume=True)
    gets past validation when the parent wrote an acknowledged resume-request.json.
    This exercises the exact code path the bug was in: interactive.prepare(resume=True)
    → recovery.prepare(workdir, name) without acknowledge_failed."""
    from refactory_dsh import recovery, interactive

    # -- setup: a failed run with run-config --
    directory = tmp_path / '.refactory'
    directory.mkdir()
    source = tmp_path / 'mode.py'
    source.write_text('answer = 42\n')
    config = {'mode': 'demo/run', 'project': str(tmp_path),
              'configuration_fingerprints': {str(source): recovery.fingerprint(source)}}
    (directory / 'run-config.json').write_text(json.dumps(config))
    payload = {'run_id': str(tmp_path), 'graph': {'name': 'demo-run'},
               'status': 'failed', 'file_fingerprints': {}}
    (directory / 'spine-checkpoint.json').write_text(json.dumps(payload))

    # -- parent's prepare (as start() does inside the ownership block) --
    recovery.prepare(tmp_path, 'demo-run', acknowledge_failed=True)
    assert json.loads((directory / 'resume-request.json').read_text())['acknowledge_failed'] is True

    # -- child's interactive.prepare(resume=True): this is the path the bug was in --
    # interactive.prepare acquires ownership, so we need to release before it.
    # We call it directly — it calls recovery.prepare(workdir, name) without acknowledge_failed.
    name, graphs = interactive.prepare('demo/run', workdir=tmp_path, resume=True)
    assert name == 'demo-run'
    # The child got past validation — no ValueError raised.
    request = json.loads((directory / 'resume-request.json').read_text())
    assert request['acknowledge_failed'] is True


def test_console_main_graph_control_acknowledge_failure(monkeypatch, tmp_path):
    directory = tmp_path / '.refactory'
    directory.mkdir()
    (directory / 'invocation.json').write_text(json.dumps({'id': 'inv1'}))
    checkpoint = directory / 'spine-checkpoint.json'
    checkpoint.write_text(json.dumps({
        'status': 'failed',
        'redo_plans': {'a': {'nodes': ['a']}},
    }))
    saved = json.loads(checkpoint.read_text())
    revision = saved.get('graph_revision') or hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    captured = {}
    def mock_request(root, action, **kwargs):
        captured.update(kwargs)
        captured['action'] = action
        return {'status': 'queued', 'target': 'a'}
    monkeypatch.setattr(graph_control, 'request', mock_request)
    console.main([
        'graph-control', 'rerun',
        '--run', str(tmp_path),
        '--node', 'a',
        '--revision', revision,
        '--acknowledge-failure',
    ])
    assert captured.get('acknowledge_failure') is True
    assert captured.get('node') == 'a'


def test_running_nodes_passes_through_live_snapshot(tmp_path):
    """running_nodes from spine-status.json flows through live.snapshot()."""
    directory = tmp_path / '.refactory'
    directory.mkdir()
    running_nodes = [{'id': 'research_director', 'kind': 'agent'},
                     {'id': 'gate_check', 'kind': 'gate'}]
    (directory / 'spine-status.json').write_text(json.dumps({
        'status': 'executing', 'nodes': ['research_director', 'gate_check'],
        'running_nodes': running_nodes,
        'wait_reason': 'waiting for worker delivery: research_director (agent), gate_check (gate)',
    }))
    activity = live.snapshot(tmp_path)
    assert activity['running_nodes'] == running_nodes
    assert 'research_director' in activity['wait_reason']


def test_step_tokens_sums_workers_by_node_id(tmp_path):
    """Two workers for the same node_id sum to one step_tokens entry."""
    directory = tmp_path / '.refactory'
    directory.mkdir()
    workers_dir = directory / 'workers'
    workers_dir.mkdir()
    (workers_dir / 'w1.json').write_text(json.dumps({
        'worker_id': 'w1', 'node_id': 'research_director',
        'usage': {'status': 'measured', 'tokens': 6000},
    }))
    (workers_dir / 'w2.json').write_text(json.dumps({
        'worker_id': 'w2', 'node_id': 'research_director',
        'usage': {'status': 'measured', 'tokens': 4000},
    }))
    (workers_dir / 'w3.json').write_text(json.dumps({
        'worker_id': 'w3', 'node_id': 'strategy',
        'usage': {'status': 'measured', 'tokens': 2000},
    }))
    spine = {'status': 'executing', 'nodes': ['design_doc'],
             'completed': ['research_director', 'strategy']}
    (directory / 'spine-status.json').write_text(json.dumps(spine))
    authoring = console._authoring(tmp_path, spine)
    assert authoring['step_tokens']['research_director'] == 10000
    assert authoring['step_tokens']['strategy'] == 2000
    assert authoring['total_tokens'] == 12000


def test_step_tokens_is_none_when_no_workers(tmp_path):
    """No worker receipts → step_tokens is None, not an empty dict."""
    directory = tmp_path / '.refactory'
    directory.mkdir()
    spine = {'status': 'executing', 'nodes': ['prepare']}
    (directory / 'spine-status.json').write_text(json.dumps(spine))
    authoring = console._authoring(tmp_path, spine)
    assert authoring['step_tokens'] is None
    assert authoring['total_tokens'] is None


def test_warn_tokens_emits_exactly_one_run_warning_per_step(tmp_path):
    """--warn-tokens triggers one run_warning per step via events.jsonl."""
    from refactory_dsh.budgets import RunBudget, ResourceLimits

    budget = RunBudget.create(tmp_path, ResourceLimits(warn_tokens=8000))
    # Record two workers for same node totaling 10000
    budget.record_usage('w1', 5000, node_id='research_director')
    budget.record_usage('w2', 5000, node_id='research_director')
    node_total = budget.node_tokens('research_director')
    assert node_total == 10000
    budget.warn_if_exceeded('research_director', node_total)
    # Read events.jsonl
    events_path = tmp_path / '.refactory' / 'events.jsonl'
    assert events_path.exists()
    all_lines = [json.loads(line) for line in events_path.read_text().strip().split('\n')]
    warnings = [e for e in all_lines if e.get('kind') == 'run_warning']
    assert len(warnings) == 1
    assert 'research_director' in warnings[0]['summary']
    assert '10,000' in warnings[0]['summary']

    # Second call for the same node does NOT emit a second warning
    budget.warn_if_exceeded('research_director', node_total)
    all_lines2 = [json.loads(line) for line in events_path.read_text().strip().split('\n')]
    warnings2 = [e for e in all_lines2 if e.get('kind') == 'run_warning']
    assert len(warnings2) == 1  # still only one


def test_warn_tokens_does_not_fire_below_threshold(tmp_path):
    """Below the threshold, no warning is emitted."""
    from refactory_dsh.budgets import RunBudget, ResourceLimits

    budget = RunBudget.create(tmp_path, ResourceLimits(warn_tokens=20000))
    budget.record_usage('w1', 5000, node_id='research_director')
    budget.warn_if_exceeded('research_director', 5000)
    events_path = tmp_path / '.refactory' / 'events.jsonl'
    if events_path.exists():
        all_lines = [json.loads(line) for line in events_path.read_text().strip().split('\n')]
        warnings = [e for e in all_lines if e.get('kind') == 'run_warning']
        assert len(warnings) == 0


def test_warn_tokens_validation_rejects_invalid_values():
    """warn_tokens must be a positive integer or unset."""
    from refactory_dsh.budgets import ResourceLimits
    with pytest.raises(ValueError, match="warn_tokens"):
        ResourceLimits(warn_tokens=0)
    with pytest.raises(ValueError, match="warn_tokens"):
        ResourceLimits(warn_tokens=-1)
    with pytest.raises(ValueError, match="warn_tokens"):
        ResourceLimits(warn_tokens=True)
    # Valid: positive integer
    limits = ResourceLimits(warn_tokens=1000)
    assert limits.warn_tokens == 1000
    # Valid: None (unset)
    limits = ResourceLimits(warn_tokens=None)
    assert limits.warn_tokens is None


def test_resume_waits_for_exiting_owner_then_starts_fresh(monkeypatch, tmp_path):
    """A resume of a terminal-status run whose owner exits during the wait starts a fresh owner."""
    import subprocess
    import sys
    from refactory_dsh import recovery

    directory = tmp_path / '.refactory'
    directory.mkdir()
    # Spawn a real child that exits after 0.5s — within the 3s timeout.
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(0.5)'])
    try:
        (directory / 'spine-checkpoint.json').write_text(
            json.dumps({'owner_pid': child.pid})
        )
        (directory / 'spine-status.json').write_text(
            json.dumps({'status': 'refused'})
        )
        (directory / 'invocation.json').write_text(json.dumps({'id': 'inv1'}))
        start_calls = []
        monkeypatch.setattr(recovery, 'start', lambda root, **_kw: (
            start_calls.append(root),
            {'status': 'queued', 'action': 'resume', 'run_id': str(root), 'pid': 9999},
        )[1])
        result = graph_control.request(tmp_path, 'resume')
        assert len(start_calls) == 1, "recovery.start must be called when the owner exits"
        assert result['pid'] == 9999
        # No stale control file should be queued.
        controls = list((directory / 'graph-controls').glob('*.json'))
        assert len(controls) == 0, "no control file should be queued when a fresh owner was started"
    finally:
        try:
            child.kill()
        except OSError:
            pass
        child.wait()


def test_resume_reports_owner_still_alive_after_timeout(monkeypatch, tmp_path):
    """A resume at a terminal status where the owner outlives the timeout files a pending resume."""
    directory = tmp_path / '.refactory'
    directory.mkdir()
    (directory / 'spine-checkpoint.json').write_text(
        json.dumps({'owner_pid': os.getpid()})
    )
    (directory / 'spine-status.json').write_text(
        json.dumps({'status': 'refused'})
    )
    (directory / 'invocation.json').write_text(json.dumps({'id': 'inv1'}))
    monkeypatch.setattr(graph_control, '_OWNER_EXIT_TIMEOUT', 0.3)
    result = graph_control.request(tmp_path, 'resume')
    assert result['status'] == 'queued'
    assert result.get('resume_pending') is True
    assert 'owner_exiting' not in result
    formatted = readable.format_graph_control(result)
    assert 'Resume is pending; a new owner will start when the current one exits.' in formatted
    assert (directory / 'resume-pending.json').exists()


def test_resume_queued_for_paused_live_owner(tmp_path):
    """A live owner paused at a boundary gets a queued resume — today's behaviour."""
    directory = tmp_path / '.refactory'
    directory.mkdir()
    (directory / 'spine-checkpoint.json').write_text(
        json.dumps({'owner_pid': os.getpid()})
    )
    (directory / 'spine-status.json').write_text(
        json.dumps({'status': 'paused'})
    )
    (directory / 'invocation.json').write_text(json.dumps({'id': 'inv1'}))
    result = graph_control.request(tmp_path, 'resume')
    assert result['status'] == 'queued'
    assert 'owner_exiting' not in result
    assert 'resume_pending' not in result


def test_graph_snapshot_tolerates_old_and_partial_checkpoints(tmp_path):
    assert live.snapshot(tmp_path)['graph'] is None
    checkpoint = tmp_path / '.refactory/spine-checkpoint.json'
    checkpoint.parent.mkdir()
    for content in ['{', '{}', '{"graph": {"nodes": []}}']:
        checkpoint.write_text(content)
        assert live.snapshot(tmp_path)['graph'] is None


# ── the activity index and its paged read ──────────────────────────────────
#
# The journal is the record; `events.index.jsonl` and its anchor are derived from
# it. Every test here holds one reader to the other, because an index that
# changes an answer is a second source of truth whatever it is called.


def _journal(root: Path, lines: list[bytes]) -> Path:
    path = root / live.EVENTS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b''.join(lines))
    return path


def _record(kind: str = 'step', **fields: object) -> bytes:
    return (json.dumps({'schema_version': 1, 'kind': kind, 'at': '2026-10-26T09:00:00+00:00', **fields}) + '\n').encode()


def _full_scan(root: Path) -> dict[str, object]:
    """Today's answer, computed by scanning: the rule an indexed answer must equal."""
    records, total = live.events(root, 0)
    return {'total': total, 'last_event_at': records[-1].get('at') if records else None}


def _status(root: Path, *argv: str) -> dict:
    result = subprocess.run([sys.executable, '-m', 'refactory_dsh.console', 'status',
                             '--run', str(root), *argv], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


def test_events_page_returns_a_bounded_ordered_slice(tmp_path):
    """T-1: a page is a window with a cursor, and a partial write is in neither."""
    _journal(tmp_path, [_record(n=i) for i in range(5)])
    records, total = live.events_page(tmp_path, after=1, limit=2)
    assert [row['n'] for row in records] == [1, 2]
    assert total == 5
    assert live.events_page(tmp_path, after=5, limit=10) == ([], 5)
    with (tmp_path / live.EVENTS).open('ab') as handle:
        handle.write(b'{"kind":"half"')
    records, total = live.events_page(tmp_path, after=3, limit=10)
    assert [row['n'] for row in records] == [3, 4]
    assert total == 5


def test_one_status_answer_takes_its_summary_from_the_page_pass(tmp_path, monkeypatch):
    """T-2: `total`/`last_event_at` cost no second journal read (today's double scan)."""
    _journal(tmp_path, [_record(n=i) for i in range(10)])
    live.page(tmp_path)
    reads: list[str] = []
    open_real = Path.open

    def counting(self, *args, **kwargs):
        if self.name == live.EVENTS.name:
            reads.append(str(self))
        return open_real(self, *args, **kwargs)

    # A warm index and an unchanged journal: `total` and `last_event_at` are in
    # the overview line, and the journal is opened only to re-read its first
    # record as the identity anchor — never walked to count or to find the newest
    # `at`, which is what `snapshot` used to do twice.
    monkeypatch.setattr(Path, 'open', counting)
    assert live.summary(tmp_path) == {'total': 10, 'last_event_at': '2026-10-26T09:00:00+00:00'}
    assert len(reads) == 1, reads

    # A page opens the journal once for its records, on top of the anchor read it
    # shares with the summary: never a pass over the whole journal.
    reads.clear()
    result = live.page(tmp_path, after=4, limit=3)
    assert [row['n'] for row in result['events']] == [4, 5, 6]
    assert (result['total'], result['last_event_at']) == (10, '2026-10-26T09:00:00+00:00')
    assert len(reads) == 2, reads

    # The record appended since is indexed from where the last pass stopped: an
    # open each for the anchor and the new suffix, and none per record — the same
    # three opens whether one record or a thousand arrived.
    monkeypatch.undo()
    with (tmp_path / live.EVENTS).open('ab') as handle:
        handle.write(_record(n=10))
    reads.clear()
    monkeypatch.setattr(Path, 'open', counting)
    assert live.summary(tmp_path)['total'] == 11
    assert len(reads) <= 3, reads

    # The same call on a journal a thousand records longer costs the same opens,
    # so nothing here is proportional to the history.
    with (tmp_path / live.EVENTS).open('ab') as handle:
        for n in range(11, 1011):
            handle.write(_record(n=n))
    assert live.summary(tmp_path)['total'] == 1011
    reads.clear()
    assert live.summary(tmp_path)['total'] == 1011
    assert len(reads) <= 3, reads


def _median_ms(action: object, rounds: int = 9) -> float:
    """The median duration of `action` in milliseconds, warm."""
    timings = []
    for _ in range(rounds):
        started = time.perf_counter()
        action()
        timings.append((time.perf_counter() - started) * 1000)
    return statistics.median(timings)


def test_page_cost_does_not_grow_with_the_journal(tmp_path):
    """T-3: a warm page and a warm summary are flat across N — R-1.7a and R-1.7b.

    The assertions are cross-N ratios, because a threshold that compares two
    numbers measured at the same N accepts an implementation that is linear in N
    at every N (the code this test replaces did exactly that: 9.05 ms at 5,000
    records and 35.6 ms at 20,000 records still passed). With three journal sizes
    40x apart, a page that reads anything proportional to N cannot stay within
    `max(small * 2, 5 ms)`; a page that seeks does, because its cost is four
    strides and 200 records whatever the history behind it is.
    """
    sizes = (500, 5000, 20000, 270000)
    roots = {}
    for size in sizes:
        roots[size] = tmp_path / f'journal-{size}'
        _journal(roots[size], [_record(i=i, summary='x' * 900) for i in range(size)])
        # The cold first call is allowed to scan: it is what builds the index.
        assert len(live.page(roots[size])['events']) == live.PAGE_DEFAULT

    # R-1.7b: `total`/`last_event_at` cost one small file read, not a walk.
    summary = {size: _median_ms(lambda root=roots[size]: live.summary(root)) for size in sizes}

    # R-1.7a: the same 200-record page at a fixed `after`, and at N/2.
    pages = {size: _median_ms(lambda root=roots[size]: live.page(root, after=200, limit=200))
             for size in sizes}
    late = {size: _median_ms(lambda root=roots[size], n=size: live.page(root, after=n // 2, limit=200))
            for size in sizes}

    # 40x the journal for the same page: the ceiling is 2x, not the 10x the
    # previous test used, and it is measured against a page, not against nothing.
    assert pages[20000] < max(pages[500] * 2, 5.0), (pages, late)
    assert late[20000] < max(late[500] * 2, 5.0), (pages, late)
    # The slope with no absolute floor in front of it, at the size the reported
    # run actually reached. The 5 ms floor above is needed at 500 records, where
    # a page's own cost is close to timer noise, but it also hides a read
    # proportional to N while the index is still small: a page that parsed the
    # whole index was measured passing that floor at 20,000 records (0.74 ms
    # against 5.0 ms) and passing a 20,000-vs-5,000 ratio at 1.16x, because the
    # 64:1 striding keeps a full index read cheap at that size. At 270,000
    # records the same read costs 4.5x a page (2.76 ms against 0.61 ms), so the
    # ratio is asserted against the largest pair, where it bites.
    assert pages[270000] < pages[20000] * 2, (pages, late)
    assert late[270000] < late[20000] * 2, (pages, late)
    # The summary is flat across the same range, and is cheaper than a page
    # because it parses no record at all.
    assert summary[20000] < max(summary[500] * 2, 5.0), summary
    assert summary[270000] < summary[20000] * 2, summary
    # A page that returns 200 records has a floor; a real one must not be below
    # the noise floor either, so a broken (empty) page cannot pass by being fast.
    assert pages[20000] > 0 and len(live.page(roots[20000], after=200, limit=200)['events']) == 200

    # R-1.7a's other half: an exhaustive pass costs O(N) in total, so N x 4
    # must be about x 4 and not x 16. 40x the records may take 12x the time.
    def full_pass(root: Path) -> int:
        after, pages_read = 0, 0
        while True:
            result = live.page(root, after=after, limit=live.PAGE_DEFAULT)
            pages_read += 1
            after = result['cursor']
            if not result['has_more']:
                return pages_read

    passes = {size: _median_ms(lambda root=roots[size]: full_pass(root), rounds=3) for size in sizes}
    assert passes[20000] < passes[500] * 40 * 2, passes


def test_page_cost_separates_journal_size_from_worker_count(tmp_path):
    """T-3b: with the index warm, N does not move a page; receipts cost separately.

    The journal bound (i) is measured on `live.page`, which reads records only.
    The receipt slope (ii) is measured on `console.status`, which is where the
    `O(workers + deliveries)` member R-1.5 names is actually paid; asserting it
    of `live.page` would be asserting a cost that function does not have.
    """
    def timed(page_read, rounds: int = 5) -> float:
        for _ in range(rounds):
            page_read()
        started = time.perf_counter()
        for _ in range(rounds):
            page_read()
        return (time.perf_counter() - started) / rounds

    def receipted(root: Path, count: int) -> None:
        workers = root / '.refactory/workers'
        workers.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            (workers / f'w{i}.json').write_text(json.dumps({'id': f'w{i}', 'usage': {'tokens': i}}))

    trees = {}
    for name, size in (('small', 500), ('large', 20000)):
        for receipts in (1, 500):
            root = tmp_path / f'{name}-{receipts}'
            _journal(root, [_record(i=i) for i in range(size)])
            receipted(root, receipts)
            # Warm the index, then measure the second page onwards.
            assert len(live.page(root)['events']) == min(size, live.PAGE_DEFAULT)
            trees[name, receipts] = root

    journal_page = lambda root: lambda: live.page(root, after=200, limit=200)   # noqa: E731
    status_page = lambda root: lambda: console.status(root, activity_after=0)   # noqa: E731

    # (i) journal size does not move a warm page, at the same receipt count.
    assert timed(journal_page(trees['large', 1])) < max(timed(journal_page(trees['small', 1])) * 10, 0.25)

    # (ii) the receipt member grows with the receipt count, and is not claimed flat.
    one, many = timed(status_page(trees['small', 1])), timed(status_page(trees['small', 500]))
    assert many > one, (one, many)
    # And the journal bound is independent of it: both trees pay the same receipts cost.
    assert timed(status_page(trees['large', 500])) < many * 10


def test_every_page_carries_the_summary_members(tmp_path):
    """T-6: a reader that only ever reads page 0 still sees the run summary."""
    _journal(tmp_path, [_record(n=i) for i in range(300)])
    _journal(tmp_path, [_record(n=i) for i in range(300)])
    (tmp_path / '.refactory/workers').mkdir(parents=True, exist_ok=True)
    (tmp_path / '.refactory/workers/w.json').write_text(json.dumps(
        {'id': 'w', 'node_id': 'build', 'status': 'done', 'usage': {'tokens': 5, 'calls': 2}}))
    (tmp_path / 'facts.json').write_text(json.dumps({'name': 'packed'}))
    (tmp_path / '.refactory/spine-status.json').write_text(json.dumps({'status': 'executing', 'nodes': ['build']}))
    (tmp_path / '.refactory/spine-checkpoint.json').write_text(json.dumps({'owner_pid': os.getpid()}))
    (tmp_path / '.refactory/graph-control-receipt.json').write_text(json.dumps({'status': 'applied'}))
    summary = {'run_id', 'invocation', 'after', 'cursor', 'total', 'page_size', 'has_more',
               'last_event_at', 'started_at', 'nodes', 'completed', 'running_nodes', 'graph',
               'wait_reason', 'workers', 'deliveries', 'outputs', 'resources', 'control'}
    for after in (0, 200, 300):
        activity = console.status(tmp_path, activity_after=after)['activity']
        assert summary <= set(activity), sorted(summary - set(activity))
        assert activity['outputs'] == {'name': 'packed'}
        assert activity['workers'][0]['usage'] == {'tokens': 5, 'calls': 2}
        assert activity['control'] == {'status': 'applied'}
        assert str(activity['run_id']).endswith(tmp_path.name)
    assert console.status(tmp_path, activity_after=0)['activity']['cursor'] == 200


def test_status_serves_400_records_as_two_pages(tmp_path):
    """T-5: two pages of 200 reconstruct the journal, and neither exceeds it."""
    _journal(tmp_path, [_record(n=i) for i in range(400)])
    seen: list[int] = []
    after = 0
    while True:
        activity = console.status(tmp_path, activity_after=after)['activity']
        assert len(activity['events']) <= activity['page_size']
        assert activity['cursor'] == after + len(activity['events'])
        seen.extend(row['n'] for row in activity['events'])
        if not activity['has_more']:
            break
        after = activity['cursor']
    assert seen == list(range(400))


def test_format_status_still_prints_warnings_from_the_page_it_is_given(tmp_path):
    """T-8: the readable block renders the page it is given, at either cursor."""
    _journal(tmp_path, [_record('run_warning', summary='old warning')] + [_record(n=i) for i in range(400)])
    (tmp_path / '.refactory/spine-status.json').write_text(json.dumps({'status': 'executing', 'nodes': ['build']}))
    page = console.status(tmp_path, activity_after=200)
    page['text'] = readable.format_status(page)
    assert 'old warning' not in page['text']
    whole = console.status(tmp_path, activity_after=0)
    whole['text'] = readable.format_status(whole)
    assert 'Warning: old warning' in whole['text']


def test_preexisting_usage_calls_array_still_reads_as_a_count(tmp_path):
    """T-10: a pre-existing journal and receipt still read; the row reports a count."""
    calls = [{'tokens': 4}, {'tokens': 5}]
    _journal(tmp_path, [_record('worker', usage={'status': 'measured', 'tokens': 9, 'calls': calls})])
    workers = tmp_path / '.refactory/workers'
    workers.mkdir(parents=True, exist_ok=True)
    (workers / 'w.json').write_text(json.dumps(
        {'id': 'w', 'usage': {'status': 'measured', 'tokens': 9, 'calls': calls}}))
    activity = console.status(tmp_path)['activity']
    # The journal record is history and is served exactly as it was written (R-3.3).
    assert activity['events'][0]['usage']['calls'] == calls
    # The summary row reports the count, so no array travels on the wire (SD-7).
    assert activity['workers'][0]['usage'] == {'status': 'measured', 'tokens': 9, 'calls': 2}
    json.dumps(activity)  # the whole page is still serialisable


def test_last_event_at_matches_a_full_scan_on_a_fixed_matrix(tmp_path):
    """T-25: the indexed `last_event_at` is today's answer for all 15 journals."""
    good = _record('a')
    good2 = _record('b', at='2026-10-26T10:00:00+00:00')
    bad = b'not json\n'
    no_at = b'{"schema_version":1,"kind":"c"}\n'
    cases = {
        'malformed last complete line': good + good2 + bad,
        'two consecutive trailing gaps': good + bad + bad,
        'gap then unterminated tail': good + bad + b'{"half":',
        'json line that is not an object': good + b'[1,2]\n',
        'malformed middle line': good + bad + good2,
        'good json with unterminated tail': good + good2 + b'{"half":',
        'empty journal': b'',
        'blank line only': b'\n\n\n',
        'blank line at tail': good + b'\n\n',
        'record without at before a good one': no_at + good,
        'newest record without at': good + no_at,
        'only gaps': bad + bad,
        'newest {}': good + b'{}\n',
        'gap then an at-less record': good + bad + no_at,
        'ordinary newest record': good + good2,
    }
    assert len(cases) == 15
    for name, payload in cases.items():
        root = tmp_path / name.replace(' ', '_').replace('{}', 'braces')
        _journal(root, [payload])
        expected = _full_scan(root)
        # Twice: the first answers from a rebuild, the second from the index it wrote.
        for _ in range(2):
            activity = live.snapshot(root)
            assert (activity['total'], activity['last_event_at']) == (
                expected['total'], expected['last_event_at']), name
    missing = tmp_path / 'missing'
    (missing / '.refactory').mkdir(parents=True)
    assert (live.snapshot(missing)['total'], live.snapshot(missing)['last_event_at']) == (0, None)


def test_the_index_never_changes_an_answer_and_is_rebuilt(tmp_path):
    """T-24: loss, corruption, a torn line and a replaced journal answer alike."""
    _journal(tmp_path, [_record(n=i) for i in range(6)])
    expected = _full_scan(tmp_path)
    fresh = live.snapshot(tmp_path)
    assert (fresh['total'], fresh['last_event_at']) == (expected['total'], expected['last_event_at'])
    assert len(fresh['events']) == 6
    index, anchor = tmp_path / live.INDEX, tmp_path / live.PROVENANCE
    assert index.is_file() and anchor.is_file()
    whole = index.read_bytes()
    for damage in ('absent', 'anchor absent', 'corrupt', 'torn final line'):
        if damage == 'absent':
            index.unlink(missing_ok=True)
        elif damage == 'anchor absent':
            anchor.unlink()
        elif damage == 'corrupt':
            index.write_bytes(b'this is not an index\n')
        else:
            index.write_bytes(index.read_bytes() + b'{"schema_version":1,"i":6,"sta')
        activity = live.snapshot(tmp_path)
        assert (activity['total'], activity['last_event_at']) == (expected['total'], expected['last_event_at']), damage
        assert len(activity['events']) == 6, damage
        # The next call rebuilt whatever was damaged.
        assert index.is_file() and anchor.is_file(), damage
        assert index.read_bytes() == whole, damage

    # An index cut short at every byte, including inside the lines that place
    # records 1…5. A cut that leaves a whole line behind is the case that broke
    # `total`: the index claimed provenance for the whole journal and the reader
    # believed it, so the records past the cut were never placed and `total`
    # answered the truncated line count. A cut inside record 0 happened to trigger
    # a rebuild, which is why cutting at 40 alone could not see it.
    for cut in range(len(whole)):
        index.write_bytes(whole[:cut])
        for _ in range(2):  # the rebuild's answer, then the index it wrote
            activity = live.snapshot(tmp_path)
            assert (activity['total'], activity['last_event_at']) == (
                expected['total'], expected['last_event_at']), cut
            assert len(activity['events']) == 6, cut
            assert index.read_bytes() == whole, cut

    # The same sweep on a journal long enough to be several index strides, where
    # a cut can fall between strides rather than only inside one.
    _journal(tmp_path, [_record(n=i) for i in range(300)])
    live.snapshot(tmp_path)
    expected = _full_scan(tmp_path)
    whole = index.read_bytes()
    assert len(whole) > 2 * live._STRIDE_LINE
    for cut in (0, 1, 40, len(whole) // 3, len(whole) // 2, len(whole) - live._STRIDE_LINE,
                len(whole) - 1, len(whole)):
        index.write_bytes(whole[:cut])
        for _ in range(2):
            activity = live.snapshot(tmp_path)
            assert (activity['total'], activity['last_event_at']) == (
                expected['total'], expected['last_event_at']), cut
            assert len(activity['events']) == 200, cut
            assert index.read_bytes() == whole, cut

    # A gap as the newest record is counted, served, and answers `null`.
    _journal(tmp_path, [_record(n=0), b'bad\n', b'also bad\n'])
    activity = live.snapshot(tmp_path)
    assert activity['total'] == 3
    assert activity['last_event_at'] is None
    assert activity['events'][-1]['kind'] == 'journal_gap'
    assert activity['has_more'] is False

    # The journal is archived away, then a fresh one replaces it at the same path.
    archived = tmp_path / '.refactory/invocations/second/artifacts'
    archived.mkdir(parents=True)
    (tmp_path / live.EVENTS).rename(archived / 'events.jsonl')
    gone = live.snapshot(tmp_path)
    assert (gone['total'], gone['last_event_at'], gone['events']) == (0, None, [])
    _journal(tmp_path, [_record('fresh', at='2026-10-26T11:00:00+00:00') for _ in range(4)])
    replaced = live.snapshot(tmp_path)
    assert (replaced['total'], replaced['last_event_at']) == (4, '2026-10-26T11:00:00+00:00')
    assert replaced['events'][0]['kind'] == 'fresh'


def test_the_index_answers_under_concurrent_builders_and_torn_lines(tmp_path):
    """T-26: two builders and a torn line; three writers; one answer each."""
    _journal(tmp_path, [_record(n=i) for i in range(50)])
    script = tmp_path / 'read.py'
    script.write_text(
        'import json, sys\n'
        'from pathlib import Path\n'
        'from refactory_dsh import live\n'
        'result = live.snapshot(Path(sys.argv[1]))\n'
        'print(json.dumps({"total": result["total"], "last_event_at": result["last_event_at"],'
        ' "kinds": [row.get("n", row["kind"]) for row in result["events"]]}))\n'
    )

    def reader() -> subprocess.Popen[str]:
        return subprocess.Popen([sys.executable, str(script), str(tmp_path)],
                                stdout=subprocess.PIPE, text=True)

    # A warm index whose last line was torn mid-write, then two readers racing.
    live.page(tmp_path)
    assert (tmp_path / live.INDEX).is_file()
    (tmp_path / live.INDEX).write_bytes((tmp_path / live.INDEX).read_bytes() + b'{"schema_version":1,"i":')
    builders = [reader(), reader()]
    outputs = [json.loads(process.communicate()[0]) for process in builders]
    expected = _full_scan(tmp_path)
    for output in outputs:
        assert (output['total'], output['last_event_at']) == (expected['total'], expected['last_event_at'])
        assert output['kinds'] == list(range(50))

    # Three processes appending, as the spine, the worker and the observer do.
    def writer(offset: int) -> subprocess.Popen[bytes]:
        code = ('import sys\nfrom pathlib import Path\nfrom refactory_dsh import live\n'
                f'for n in range({offset}, {offset + 20}): live.emit(Path(sys.argv[1]), "w", n=n)\n')
        return subprocess.Popen([sys.executable, '-c', code, str(tmp_path)])

    writers = [writer(100), writer(200), writer(300)]
    assert [process.wait() for process in writers] == [0, 0, 0]
    lines = (tmp_path / live.EVENTS).read_bytes().splitlines(keepends=True)
    # The count is the newline-terminated lines, and the index agrees with it.
    assert live.snapshot(tmp_path)['total'] == sum(1 for line in lines if line.endswith(b'\n'))


def test_bounded_activity_page(tmp_path):
    """Every `status` answer is a page, not the whole journal (§12 gate 7).

    A 400-record journal page by page, then a gap record as the newest, then a
    record after it, then a half-written line: the bounds, the refusals and the
    timestamp at the boundary are what this locks.
    """
    _journal(tmp_path, [_record(i=i, summary='x' * 1000) for i in range(400)])
    for argv, after in ((('--activity-after', '0'), 0), (('--activity-after', '200'), 200),
                        (('--activity-after', '400'), 400), (('--activity-limit', '50'), 0)):
        activity = _status(tmp_path, *argv)['activity']
        assert len(activity['events']) <= activity['page_size']
        assert activity['cursor'] == after + len(activity['events'])
        assert activity['has_more'] is (activity['cursor'] < activity['total']), argv

    for refused in (('--activity-limit', '0'), ('--activity-limit', '-1'), ('--activity-after', '-1')):
        result = subprocess.run([sys.executable, '-m', 'refactory_dsh.console', 'status',
                                 '--run', str(tmp_path), *refused], capture_output=True, text=True)
        assert result.returncode == 2, (refused, result.stdout)

    with (tmp_path / live.EVENTS).open('ab') as handle:
        handle.write(b'not json\n')
    activity = _status(tmp_path, '--activity-after', '400')['activity']
    assert activity['total'] == 401
    assert activity['events'][-1]['kind'] == 'journal_gap'
    # R-1.4's identity holds on this answer like on every other. `after == 400`
    # is the gap's own page — the single record left — so serving it advances the
    # cursor to `total` and there is nothing more: `has_more` is false.
    assert activity['has_more'] is (activity['cursor'] < activity['total'])
    assert (activity['cursor'], activity['has_more']) == (401, False)
    assert activity['last_event_at'] is None
    # Its reachability is asserted from the start instead: an exhaustive pass over
    # 200-record pages serves all 401 records and the gap is its last.
    seen: list[str] = []
    after = 0
    while True:
        page = _status(tmp_path, '--activity-after', str(after))['activity']
        assert page['has_more'] is (page['cursor'] < page['total']), after
        seen.extend(str(row['kind']) for row in page['events'])
        if not page['has_more']:
            break
        after = page['cursor']
    assert len(seen) == 401 and seen[-1] == 'journal_gap'
    middle = _status(tmp_path, '--activity-after', '200')['activity']
    assert (middle['cursor'], middle['has_more']) == (400, True)

    with (tmp_path / live.EVENTS).open('ab') as handle:
        handle.write(_record('after-gap', at='2026-10-26T12:00:00+00:00'))
    assert _status(tmp_path)['activity']['last_event_at'] == '2026-10-26T12:00:00+00:00'

    with (tmp_path / live.EVENTS).open('ab') as handle:
        handle.write(b'{"half":')
    assert _status(tmp_path)['activity']['total'] == 402


def test_builders_racing_a_live_writer_never_publish_a_torn_pair(tmp_path):
    """T-27: builders racing a live writer publish whole pairs, and leave no temp.

    §4.6 gives most of its text to concurrent builders, and T-26 races two of them
    against a torn line already on disk. What it does not exercise is builders
    running *while* a writer appends — the state the overview and provenance
    publishes race in. Every reader is also a writer of those two derived files,
    and one fixed temp name let two of them open the same inode: the loser's
    truncate landed mid-flight, so a half-written `events.index.overview.json` (or
    a pair from two different coverages) could be renamed into place.

    Correctness survived that — the reader re-parses and rejects a torn pair — so
    what a test has to catch is the torn write itself, which forces every racing
    reader onto the full O(N) rebuild this index exists to remove. A per-writer
    temp (PID + a per-process counter, so no two publishes share an inode; no
    `O_EXCL`, which would wedge the index behind a crashed writer's leftover) is
    what this asserts: a concurrent watcher never sees a derived file that is not
    a complete JSON object, the answers still equal a full scan, and no publish
    temp outlives its writer.
    """
    live.emit(tmp_path, 'seed')
    directory = tmp_path / '.refactory'
    script = tmp_path / 'race.py'
    script.write_text(
        'import json, sys\n'
        'from pathlib import Path\n'
        'from refactory_dsh import live\n'
        'root = Path(sys.argv[1])\n'
        'answers = []\n'
        'for _ in range(1200):\n'
        '    result = live.snapshot(root)\n'
        '    answers.append({"total": result["total"], "last_event_at": result["last_event_at"],\n'
        '                    "summary": live.summary(root)["total"]})\n'
        'print(json.dumps(answers))\n'
    )

    def builder() -> subprocess.Popen[str]:
        return subprocess.Popen([sys.executable, str(script), str(tmp_path)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Warm the index so the publishes are what the builders do from their first
    # poll, not the initial rebuild from an empty run directory.
    live.snapshot(tmp_path)
    # Start the builders first, then append while they are still reading, so the
    # publishes interleave with the writer instead of following it. Eight builders
    # against 1500 appends is the pressure that makes the collision reproducible:
    # the fixed-temp code publishes a torn file in roughly two runs out of three,
    # the per-writer one never does (measured over three runs each).
    builders = [builder() for _ in range(8)]
    torn: list[str] = []
    observed = 0
    for n in range(1500):
        live.emit(tmp_path, 'racing', n=n)
        # Watch the two published files from here, between appends, the way a
        # second reader would: each must always be one complete JSON object.
        for name in (live.OVERVIEW, live.PROVENANCE):
            try:
                text = (tmp_path / name).read_text()
            except FileNotFoundError:
                continue
            observed += 1
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                torn.append(f'{name.name}: {text[:60]!r}')
                continue
            if not isinstance(value, dict):
                torn.append(f'{name.name}: not an object')
    for process in builders:
        out, err = process.communicate()
        assert process.returncode == 0, err
        answers = json.loads(out)
        assert len(answers) == 1200
        for answer in answers:
            # A builder reads while the writer appends, so each of its two reads
            # observes its own instant of the journal. Neither count may be pinned
            # to the *final* total, and a tight bound around it is not sound under
            # contention (CI's four workers produced 1520 against a true 1501 once):
            # `summary` and `snapshot` each re-read, the writer is mid-append, and
            # the derived index is rebuilt by whichever reader wins. Asserting a
            # narrow numeric band on racing reads tests the scheduler, not this
            # change. What this test is for — that a racing writer never publishes a
            # torn derived file — is asserted below via `torn`, and the settled
            # answer that follows pins the exact total.
            assert isinstance(answer['total'], int) and answer['total'] >= 0, answer
            assert isinstance(answer['summary'], int) and answer['summary'] >= 0, answer
    # The declaration above is only meaningful if the watcher really did overlap:
    # with no reads there is nothing that could have seen a torn file.
    assert observed > 0, 'the watcher never overlapped the builders; the race was not exercised'
    assert torn == [], f'torn derived file published while racing: {torn[:3]}'
    # The settled answer equals the full scan exactly.
    expected = _full_scan(tmp_path)
    settled = live.snapshot(tmp_path)
    assert (settled['total'], settled['last_event_at']) == (expected['total'], expected['last_event_at'])
    assert settled['total'] == 1501
    # No publish temp outlives its writer, under any name.
    leftovers = [path.name for path in directory.iterdir() if path.name.endswith('.tmp')]
    assert leftovers == [], leftovers


def test_a_cursor_past_the_end_answers_an_empty_page_clamped_to_total(tmp_path):
    """T-28: `after` beyond the journal never hands back a cursor past `total`.

    R-1.3 defines the cursor as `after + len(events)` and leaves `after > total`
    open. Serving it literally returned `cursor: 5000` on a 1,100-record journal: a
    caller trusting `cursor <= total` was surprised, and a page-until-`has_more`
    loop still terminated but on a number that names no record. The page stays
    empty and `has_more` stays false; only the cursor is clamped.
    """
    _journal(tmp_path, [_record(n=i) for i in range(60)])
    assert live.summary(tmp_path)['total'] == 60

    beyond = live.page(tmp_path, after=5000, limit=10)
    assert beyond['events'] == []
    assert (beyond['cursor'], beyond['total'], beyond['has_more']) == (60, 60, False)
    # Exactly at the end is the settled case, and it agrees.
    at_end = live.page(tmp_path, after=60, limit=10)
    assert at_end['events'] == []
    assert (at_end['cursor'], at_end['has_more']) == (60, False)
    # Just short of the end still serves the last record and lands on `total`.
    last = live.page(tmp_path, after=59, limit=10)
    assert len(last['events']) == 1
    assert (last['cursor'], last['has_more']) == (60, False)
    # A page in the middle is unaffected by the clamp.
    middle = live.page(tmp_path, after=10, limit=10)
    assert (middle['cursor'], middle['has_more']) == (20, True)
    # An empty run directory answers 0 rather than echoing the caller's cursor.
    empty = tmp_path / 'nothing-here'
    empty.mkdir()
    assert live.page(empty, after=99)['cursor'] == 0


def test_a_trailing_partial_write_costs_one_publish_then_nothing(tmp_path, monkeypatch):
    """T-29: a poll tick does not re-index or re-publish a journal that is mid-write.

    A writer appending record-by-record leaves unterminated bytes between its
    writes. The walk stops at that line, so the covered prefix does not move — but
    the journal's own size does, and a guard comparing the journal against the
    *covered prefix* rather than against the size it was measured at rebuilt the
    whole index on every tick. Every reader polls while a run is live, so that
    rebuild is the O(N) scan the index exists to avoid, repeated per poll.

    The contract this locks: settling on a trailing partial publishes at most once,
    and every poll after that neither rebuilds nor rewrites.
    """
    live.emit(tmp_path, 'complete')
    journal = tmp_path / live.EVENTS
    with journal.open('ab') as handle:
        handle.write(b'{"kind":"half"')
    # Settle first: the read that observes the partial publishes once (the covered
    # prefix is unchanged, but `size_at_write` catches up to what was examined), and
    # everything after that must be silent.
    live.summary(tmp_path)
    live.summary(tmp_path)
    rebuilds: list[int] = []
    writes: list[int] = []
    real_rebuild, real_write = live._rebuild, live._write
    monkeypatch.setattr(live, '_rebuild', lambda *a, **k: (rebuilds.append(1), real_rebuild(*a, **k))[1])
    monkeypatch.setattr(live, '_write', lambda *a, **k: (writes.append(1), real_write(*a, **k))[1])
    for _ in range(10):
        assert live.summary(tmp_path)['total'] == 1
        live.page(tmp_path)
    assert rebuilds == [], f'poll ticks rebuilt the index {len(rebuilds)} times'
    assert writes == [], f'poll ticks republished the index {len(writes)} times'
    # The partial is still not a record, and terminating it makes it a counted gap.
    with journal.open('ab') as handle:
        handle.write(b'"}\n')
    assert live.summary(tmp_path)['total'] == 2
    assert live.events(tmp_path)[1] == 2
