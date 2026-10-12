from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

import pytest

from refactory_dsh import lowering, workers
from refactory_dsh.model import DshModelClient
from refactory_dsh.nn import Graph, Node
from refactory_dsh.nn.executor import execute
from refactory_dsh.steering import Ledger


def fake_wire(monkeypatch, callback):
    """Replace only external model transport; all runtime dispatch/receipts remain."""
    def wire(self, cmd, run_dir, stop_at=None, budget_s=None):
        receipt = json.loads(self._worker_receipt.read_text())
        receipt['actual_route'] = {**self.route(), 'source': 'controlled-external-IO'}
        self._worker_receipt.write_text(json.dumps(receipt))
        return subprocess.CompletedProcess(cmd, 0, callback(self, cmd[-1], run_dir, budget_s), '')
    monkeypatch.setattr(DshModelClient, '_run_completion', wire)


def verdict(prompt, value='proceed'):
    start = prompt.index('Runtime evidence (read current artifacts before judging):\n')
    request, _ = json.JSONDecoder().raw_decode(prompt[start:].split('\n', 1)[1])
    return json.dumps({key: request[key] for key in ('schema_version', 'gate_id', 'input_digest', 'evidence_digest')}
                      | {'verdict': value, 'rationale': 'Reviewed exact current evidence.'})


def test_optional_stage_allowance_roundtrip_and_real_dispatch(monkeypatch, tmp_path):
    legacy = Node('legacy', 'agent', prompt_template='old')
    assert 'timeout_s' not in legacy.to_state()
    explicit = Node('build', 'agent', prompt_template='deliver', writes=('result.md',), timeout_s=1800)
    assert Node.from_state(explicit.to_state()).timeout_s == 1800
    with pytest.raises(ValueError, match='positive integer'):
        Node('bad', 'agent', prompt_template='bad', timeout_s=True)
    graph = Graph()
    graph.add(explicit)
    seen = []
    fake_wire(monkeypatch, lambda client, prompt, root, wall: seen.append(wall) or 'delivered')
    result = execute(graph, tmp_path, model=DshModelClient(attempts=3))
    assert result.exit_ok and seen == [1800]
    receipt = json.loads(next((tmp_path / '.refactory/workers').glob('*.json')).read_text())
    assert receipt['effective_deadlines']['stage_timeout_s'] == 1800
    lowered = workers.lower_workflow(lowering.to_workflow({'graph': graph.to_state()}), tmp_path / 'native', DshModelClient().context())
    spec = json.loads(next((tmp_path / 'native/.refactory/worker-specs').glob('*.json')).read_text())
    assert lowered.nodes['build'].command_timeout_s == 1800
    assert spec['model_context']['node_timeout_s'] == 1800


def test_agent_gate_keeps_question_model_owner_and_strict_fresh_verdict(monkeypatch, tmp_path):
    graph = Graph()
    graph.add(Node('review', 'agent', prompt_template='Write the review.', reads=('spec.md',), writes=('review.md',)))
    graph.add(Node('choose', 'gate', gate_kind='agent', prompt_template='Original CEO criteria.', reads=('review.md',), timeout_s=3600))
    graph.add(Node('accepted', 'agent', prompt_template='accepted', writes=('accepted.md',)))
    graph.add(Node('repair', 'agent', prompt_template='repair', writes=('repair.md',)))
    graph.connect('review', 'choose')
    graph.connect('choose', 'accepted', 'proceed')
    graph.connect('choose', 'repair', 'reloop')
    (tmp_path / 'spec.md').write_text('exact input')
    seen = []
    def answer(client, prompt, root, wall):
        seen.append((client._node_id, wall, prompt))
        return verdict(prompt) if client._node_id == 'choose' else 'complete substantive output'
    fake_wire(monkeypatch, answer)
    result = execute(graph, tmp_path, facts={'create_fidelity': True}, model=DshModelClient(effort='high'))
    assert result.exit_ok and (tmp_path / 'accepted.md').exists()
    assert result.model_calls == 3
    assert 'Original CEO criteria.' in seen[1][2] and seen[1][1] == 3600
    receipt = json.loads(next((tmp_path / '.refactory/create-fidelity-gates').glob('*/verdict.json')).read_text())
    assert receipt['owner'] == 'agent' and receipt['actual_route']['worker_backend'] == 'dsh'
    assert receipt['evidence']['deliveries']['review']['execution']['input_fingerprints']['spec.md']


@pytest.mark.parametrize('mutation', ['inputs', 'response'])
def test_stale_gate_evidence_or_verdict_fails_without_advancing(monkeypatch, tmp_path, mutation):
    graph = Graph()
    graph.add(Node('choose', 'gate', gate_kind='agent', prompt_template='criteria', reads=('review.md',)))
    graph.add(Node('accepted', 'agent', prompt_template='done', writes=('accepted.md',)))
    graph.connect('choose', 'accepted', 'proceed')
    (tmp_path / 'review.md').write_text('first')
    def answer(client, prompt, root, wall):
        response = verdict(prompt)
        if mutation == 'inputs':
            (root / 'review.md').write_text('changed during evaluation')
        else:
            response = response.replace(json.loads(response)['input_digest'], 'stale')
        return response
    fake_wire(monkeypatch, answer)
    result = execute(graph, tmp_path, facts={'create_fidelity': True}, model=DshModelClient())
    assert not result.exit_ok and 'stale' in result.error
    assert not (tmp_path / 'accepted.md').exists()


def test_human_approval_is_one_use_and_bound_to_strategy(tmp_path):
    graph = Graph()
    gate = graph.add(Node('gate_strategy', 'gate', gate_kind='user', reads=('strategy.md',)))
    graph.add(Node('done', 'op', call='refactory_dsh.examples:noop'))
    graph.connect(gate.id, 'done', 'proceed')
    (tmp_path / 'strategy.md').write_text('version one')
    ledger = Ledger.seed(tmp_path, 'Build requested behavior.')
    review = ledger.visit(gate.id, gate.reads)
    ledger.verdict(gate.id, 'proceed', review=review)
    # Stop after routing, without invoking an unrelated op.
    result = execute(graph, tmp_path, facts={'create_fidelity': True}, max_steps=1)
    assert result.facts['gate_strategy_verdict'] == 'proceed'
    second = execute(graph, tmp_path, facts={'create_fidelity': True}, verdicts={gate.id: 'proceed'})
    assert not second.exit_ok and 'fresh human approval' in second.error
    ledger.verdict(gate.id, 'proceed', review=review)
    (tmp_path / 'strategy.md').write_text('version two')
    stale = execute(graph, tmp_path, facts={'create_fidelity': True})
    assert not stale.exit_ok and 'fresh human approval' in stale.error
    ledger.verdict(gate.id, 'proceed', approval=True)
    assert 'standing approval' in execute(graph, tmp_path, facts={'create_fidelity': True}).error
    progress = json.loads((tmp_path / '.refactory/progress.json').read_text())
    assert progress['node'] == 'gate_strategy', 'a failed approval boundary must remain resumable'


@pytest.mark.parametrize('precheck_ok,archive_fails', [(True, False), (False, False), (True, True)])
def test_nonblocking_archive_overlaps_builder_settles_and_preserves_failure(monkeypatch, tmp_path, precheck_ok, archive_fails):
    graph = Graph()
    graph.add(Node('archivist_plan', 'agent', prompt_template='archive', writes=('archive.md',), blocking=False))
    graph.add(Node('builder', 'agent', prompt_template='build', writes=('build.md',)))
    graph.connect('archivist_plan', 'builder')
    builder_started = threading.Event()
    archive_finished = threading.Event()
    def answer(client, prompt, root, wall):
        if client._node_id == 'archivist_plan':
            assert builder_started.wait(2), 'builder was blocked by advisory archive'
            if archive_fails:
                raise RuntimeError('archive publication failed')
            archive_finished.set()
            return 'archive delivered'
        builder_started.set()
        return 'build delivered'
    fake_wire(monkeypatch, answer)
    result = execute(graph, tmp_path, facts={'create_fidelity': True, 'create_fidelity_precheck_ok': precheck_ok}, model=DshModelClient())
    assert result.exit_ok is (precheck_ok and not archive_fails)
    if not archive_fails:
        assert archive_finished.is_set() and (tmp_path / 'archive.md').read_text() == 'archive delivered'
    else:
        assert 'archive publication failed' in result.error


def test_unbudgeted_owned_cli_timeout_performs_runtime_shutdown(tmp_path):
    import sys
    from refactory_dsh.owned_process import run_owned
    script = tmp_path / 'runtime.py'
    script.write_text('import signal,time\nfrom pathlib import Path\ndef stop(*args):\n Path("stopped").write_text("runtime cleanup settled")\n raise SystemExit(0)\nsignal.signal(signal.SIGTERM,stop)\ntime.sleep(30)\n')
    with pytest.raises(subprocess.TimeoutExpired):
        run_owned([sys.executable, str(script)], cwd=tmp_path, timeout=0.2)
    assert (tmp_path / 'stopped').read_text() == 'runtime cleanup settled'


def test_unbudgeted_graph_shutdown_keeps_forced_cleanup_unknown(monkeypatch, tmp_path):
    import sys
    from refactory_dsh import owned_process
    script = tmp_path / 'runtime.py'
    script.write_text('import signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\ntime.sleep(30)\n')
    monkeypatch.setattr(owned_process, 'SHUTDOWN_GRACE_S', 0.05)
    with pytest.raises(RuntimeError, match='child cleanup unconfirmed'):
        workers.run_graph_command([sys.executable, str(script), '--profile', 'graph'], workdir=tmp_path,
                                  context=DshModelClient().context(), cwd=tmp_path, timeout=0.1)
    receipt = json.loads(next((tmp_path / '.refactory/workers').glob('*.json')).read_text())
    assert receipt['state'] == 'unknown' and 'cleanup unconfirmed' in receipt['error']


def test_foreground_failure_retains_unknown_background_cleanup_and_refuses_mode_resume(monkeypatch, tmp_path):
    import sys
    from refactory_dsh import owned_process
    from refactory_dsh.nn.mode import Mode

    graph = Graph()
    graph.add(Node('archivist_plan', 'agent', prompt_template='archive', blocking=False))
    graph.add(Node('builder', 'agent', prompt_template='build'))
    graph.connect('archivist_plan', 'builder')
    runtime = tmp_path / 'runtime.py'
    runtime.write_text('import signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\ntime.sleep(30)\n')
    monkeypatch.setattr(owned_process, 'SHUTDOWN_GRACE_S', 0.05)
    archive_started = threading.Event()
    def answer(client, prompt, root, wall):
        if client._node_id == 'archivist_plan':
            archive_started.set()
            workers.run_graph_command([sys.executable, str(runtime), '--profile', 'graph'], workdir=tmp_path,
                                      context=DshModelClient().context(), cwd=tmp_path, timeout=0.1)
        assert archive_started.wait(2)
        raise RuntimeError('foreground builder failed')
    fake_wire(monkeypatch, answer)
    result = execute(graph, tmp_path, facts={'create_fidelity': True}, model=DshModelClient())
    assert not result.exit_ok and 'foreground builder failed' in result.error
    assert 'advisory worker failed' in result.error and 'cleanup unconfirmed' in result.error
    progress = json.loads((tmp_path / '.refactory/progress.json').read_text())
    assert progress['node'] == 'builder' and progress['outcome'] == 'unknown'
    receipts = [json.loads(path.read_text()) for path in (tmp_path / '.refactory/workers').glob('*.json')]
    assert any(receipt['state'] == 'unknown' for receipt in receipts)
    with pytest.raises(ValueError, match='unknown side-effect outcome'):
        Mode.resume(tmp_path)


def test_native_gate_command_crosses_real_runner_model_and_receipt_boundaries(tmp_path):
    import os
    import sys
    graph = Graph()
    graph.add(Node('init', 'op', call='refactory_dsh.create_fidelity_ops:init_user_intent'))
    graph.add(Node('choose', 'gate', gate_kind='agent', prompt_template='Original CEO criteria.', reads=('report.md',), timeout_s=5))
    graph.add(Node('done', 'agent', prompt_template='done'))
    graph.connect('init', 'choose')
    graph.connect('choose', 'done', 'proceed')
    state = {'graph': graph.to_state()}
    lowering.write_declarations(state, tmp_path)
    (tmp_path / 'facts.json').write_text(json.dumps({'create_fidelity': True}))
    (tmp_path / 'report.md').write_text('Full substantive current report, including acceptance evidence.')
    script = tmp_path / 'model-cli'
    script.write_text(f'#!{sys.executable}\n' + '''import json,os,sys
from pathlib import Path
receipt=Path(os.environ['REFACTORY_WORKER_RECEIPT'])
record=json.loads(receipt.read_text())
record['actual_route']={**record['requested_route'],'source':'controlled-external-IO'}
receipt.write_text(json.dumps(record))
prompt=sys.argv[-1]
assert 'Full substantive current report' in prompt
start=prompt.index('Runtime evidence (read current artifacts before judging):\\n')
request,_=json.JSONDecoder().raw_decode(prompt[start:].split('\\n',1)[1])
answer={k:request[k] for k in ('schema_version','gate_id','input_digest','evidence_digest')}
answer.update(verdict='proceed',rationale='Reviewed full current report.')
print(json.dumps(answer))
''')
    script.chmod(0o700)
    client = DshModelClient(dsh_cli=str(script), effort='high')
    workflow = workers.lower_workflow(lowering.to_workflow(state), tmp_path, client.context(), lowering.node_declarations(state))
    gate = workflow.nodes['choose']
    assert gate.evaluator_type == 'agent' and gate.evaluator_role.value == 'ceo'
    assert gate.command_timeout_s > 5
    env = {**os.environ, **workers.environment(client.context()), 'REFACTORY_WORKDIR': str(tmp_path),
           'PYTHONPATH': str(Path(__file__).resolve().parents[1])}
    proc = subprocess.run([sys.executable, '-m', 'refactory_dsh.run', 'refactory_dsh.create_fidelity_gates:evaluate_native', '--node', 'choose'],
                          cwd=tmp_path, env=env, text=True, capture_output=True, timeout=20)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == 'proceed'
    assert json.loads((tmp_path / 'facts.json').read_text())['choose_verdict'] == 'proceed'
    receipt = json.loads(next((tmp_path / '.refactory/workers').glob('*.json')).read_text())
    assert receipt['role'] == 'ceo' and receipt['effective_deadlines']['stage_timeout_s'] == 5


def test_native_human_bridge_publishes_digest_bound_receipt_and_expires_on_change(tmp_path):
    import os
    import sys
    from refactory_dsh.create_fidelity_gates import human_current_approval
    from refactory_dsh.nn.executor import RunContext
    reads = ('.factory/strategy/current.md', '.factory/strategy/user-intent.md')
    for name in reads:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Approved exact behavior.')
    (tmp_path / 'facts.json').write_text(json.dumps({'create_fidelity': True}))
    ledger = Ledger.seed(tmp_path, 'Build requested behavior.')
    review = ledger.review('gate_strategy', reads)
    ledger.verdict('gate_strategy', 'proceed', review=review)
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])}
    request = {'action': 'begin', 'gate': 'gate_strategy', 'reads': reads, 'labels': ['proceed', 'reloop']}
    proc = subprocess.run([sys.executable, '-m', 'refactory_dsh.steering'], cwd=tmp_path, env=env,
                          input=json.dumps(request), text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)['verdict'] == 'proceed'
    ctx = RunContext(tmp_path)
    assert human_current_approval(ctx)
    (tmp_path / reads[0]).write_text('Materially different behavior.')
    assert not human_current_approval(ctx)


def test_native_pending_revision_confirms_after_its_own_feedback_without_duplicate_append(tmp_path):
    import os
    import sys
    reads = ('.factory/strategy/current.md', '.factory/strategy/user-intent.md')
    for name in reads:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Original human requirements.')
    (tmp_path / 'facts.json').write_text(json.dumps({'create_fidelity': True}))
    ledger = Ledger.seed(tmp_path, 'Original human requirements.')
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])}
    def bridge(request):
        proc = subprocess.run([sys.executable, '-m', 'refactory_dsh.steering'], cwd=tmp_path, env=env,
                              input=json.dumps(request), text=True, capture_output=True, timeout=10)
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout)
    base = {'gate': 'gate_strategy', 'reads': reads, 'labels': ['proceed', 'reloop']}
    began = bridge({**base, 'action': 'begin'})
    ledger.note('correction', 'Acceptance must include an invalid-input preservation check.')
    ledger.verdict('gate_strategy', 'reloop', review=began['review'])
    polled = bridge({**base, 'action': 'poll', 'review': began['review']})
    assert polled['verdict'] == 'reloop' and polled['current_review'] != began['review']
    confirmed = bridge({**base, 'action': 'confirm', 'review': polled['current_review'],
                        'receipt_id': polled['receipt_id'], 'verdict': 'reloop'})
    assert confirmed['confirmed']
    assert (tmp_path / reads[1]).read_text().count('Acceptance must include') == 1
    assert len(list((tmp_path / '.refactory/create-fidelity-gates').glob('*/verdict.json'))) == 1


def test_native_interactive_revision_requires_and_records_current_human_feedback(tmp_path):
    import os
    import sys
    reads = ('.factory/strategy/current.md', '.factory/strategy/user-intent.md')
    for name in reads:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Original human requirements.')
    (tmp_path / 'facts.json').write_text(json.dumps({'create_fidelity': True}))
    ledger = Ledger.seed(tmp_path, 'Original human requirements.')
    request = {'action': 'answer', 'gate': 'gate_strategy', 'reads': reads, 'labels': ['proceed', 'reloop'],
               'review': ledger.review('gate_strategy', reads), 'verdict': 'reloop'}
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])}
    def invoke(payload):
        return subprocess.run([sys.executable, '-m', 'refactory_dsh.steering'], cwd=tmp_path, env=env,
                              input=json.dumps(payload), text=True, capture_output=True, timeout=10)
    rejected = invoke(request)
    assert rejected.returncode == 1 and 'current feedback' in rejected.stderr
    assert not ledger.verdicts()
    accepted = invoke({**request, 'feedback': 'Keep every original requirement; test both invocations.'})
    assert accepted.returncode == 0, accepted.stderr
    assert 'Keep every original requirement' in (tmp_path / reads[1]).read_text()
