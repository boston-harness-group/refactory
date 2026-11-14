from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from refactory_dsh import agent_delivery, nn
from refactory_dsh.model import DshModelClient
from refactory_dsh.steering import Ledger


def test_contract_keeps_task_steering_and_exact_paths_without_exposing_settings(tmp_path):
    Ledger.seed(tmp_path, 'Preserve the public API and run the requested check.')
    task = 'Write result.json after checking input.py.\n'
    context = {'python_bin': str(Path(sys.executable).parent), 'runtime_root': '/selected/runtime',
               'api_key': 'never-in-a-prompt', 'permission_mode': 'skip'}
    first = agent_delivery.begin(tmp_path, 'check', task, ['result.json'], reads=['input.py'], model_context=context)
    prompt = first['prompt']
    assert task in prompt and 'Preserve the public API' in prompt
    assert 'never-in-a-prompt' not in prompt and 'permission_mode' not in prompt
    assert first['execution'] == {
        'node': 'check', 'workdir': str(tmp_path), 'inputs': [str(tmp_path / 'input.py')],
        'outputs': [str(tmp_path / 'result.json')], 'python': str(Path(sys.executable).parent / 'python'),
        'refactory_source': str(Path(agent_delivery.__file__).parent), 'runtime_root': '/selected/runtime',
    }
    repeated = agent_delivery.begin(tmp_path, 'check', task, ['result.json'], reads=['input.py'], model_context=context, reuse=True)
    assert repeated == first
    assert (tmp_path / first['contexts'][0]['prompt']).read_text() == prompt
    assert json.loads((tmp_path / first['receipt']).read_text())['execution'] == first['execution']


def test_brief_final_response_does_not_replace_delivered_artifact(tmp_path):
    artifact = '{"answer":42}\n'
    first = agent_delivery.begin(tmp_path, 'build', 'Write result.json.', ['result.json'])
    (tmp_path / 'result.json').write_text(artifact)
    receipt = agent_delivery.finish(tmp_path, 'build', final_response='Done.', capture_final=True)
    assert receipt['status'] == 'delivered' and receipt['captured_final'] is None
    assert (tmp_path / 'result.json').read_text() == artifact
    assert (tmp_path / receipt['final_response']).read_text() == 'Done.'
    assert receipt['execution'] == first['execution']


def test_a_non_output_only_step_that_changes_hooks_warns_naming_the_step(tmp_path, capsys):
    """Per-step shared_changed check in finish(): warning path (#347)."""
    import subprocess
    from refactory_dsh import target_guard, worktree
    from tests_support import git_repo

    repo = git_repo(tmp_path / 'repo')
    run = tmp_path / 'run'
    found = worktree.create(repo, 'HEAD', 'hook0000warn', run)
    target_guard.record(run, repo, output_only=('emit',))
    agent_delivery.begin(run, 'research_director', 'Research.', ['research-plan.json'])
    # Modify a hook file in the shared hooks directory
    hooks = Path(subprocess.run(['git', '-C', str(repo), 'rev-parse', '--path-format=absolute',
                                 '--git-common-dir'], capture_output=True, text=True,
                                check=True).stdout.strip()) / 'hooks'
    hooks.mkdir(parents=True, exist_ok=True)
    (hooks / 'pre-commit').write_text('#!/bin/sh\nexit 0\n')
    (run / 'research-plan.json').write_text('[]\n')
    receipt = agent_delivery.finish(run, 'research_director')
    assert receipt['status'] == 'delivered'
    err = capsys.readouterr().err
    assert 'changed during research_director' in err
    assert target_guard.WARNING.strip(':').strip() in err


def test_an_output_only_step_that_changes_hooks_fails_the_step(tmp_path):
    """Per-step shared_changed check in finish(): failure path for output-only (#347)."""
    import subprocess
    from refactory_dsh import target_guard, worktree
    from tests_support import git_repo

    repo = git_repo(tmp_path / 'repo')
    run = tmp_path / 'run'
    found = worktree.create(repo, 'HEAD', 'hook1111fail', run)
    target_guard.record(run, repo, output_only=('emit', 'design_doc'))
    agent_delivery.begin(run, 'design_doc', 'Write the design.', ['design.md'])
    hooks = Path(subprocess.run(['git', '-C', str(repo), 'rev-parse', '--path-format=absolute',
                                 '--git-common-dir'], capture_output=True, text=True,
                                check=True).stdout.strip()) / 'hooks'
    hooks.mkdir(parents=True, exist_ok=True)
    (hooks / 'pre-commit').write_text('#!/bin/sh\nexit 0\n')
    (run / 'design.md').write_text('the design\n')
    import pytest
    with pytest.raises(ValueError, match='changed shared git state'):
        agent_delivery.finish(run, 'design_doc')


def test_a_step_that_does_not_change_hooks_is_not_warned(tmp_path, capsys):
    """No false positive when hooks are unchanged (#347)."""
    from refactory_dsh import target_guard, worktree
    from tests_support import git_repo

    repo = git_repo(tmp_path / 'repo')
    run = tmp_path / 'run'
    found = worktree.create(repo, 'HEAD', 'hook2222safe', run)
    target_guard.record(run, repo, output_only=('emit',))
    agent_delivery.begin(run, 'research_director', 'Research.', ['research-plan.json'])
    (run / 'research-plan.json').write_text('[]\n')
    receipt = agent_delivery.finish(run, 'research_director')
    assert receipt['status'] == 'delivered'
    err = capsys.readouterr().err
    assert 'changed during' not in err
    assert 'changed shared git state' not in err


def test_shared_state_check_does_not_run_without_a_worktree(tmp_path, capsys):
    """The check only fires when worktree.recorded returns a Worktree (#347)."""
    import subprocess
    from refactory_dsh import target_guard

    root = tmp_path / 'target'
    root.mkdir()
    subprocess.run(['git', '-C', str(root), 'init', '-q'], check=True, capture_output=True)
    (root / 'model.py').write_text('ORIGINAL = True\n')
    run = tmp_path / 'run'
    target_guard.record(run, root, output_only=('emit',))
    agent_delivery.begin(run, 'research_director', 'Research.', ['research-plan.json'])
    # Create hooks dir and modify hooks -- but no worktree record, so no shared check
    hooks = root / '.git' / 'hooks'
    hooks.mkdir(parents=True, exist_ok=True)
    (hooks / 'pre-commit').write_text('#!/bin/sh\nexit 0\n')
    (run / 'research-plan.json').write_text('[]\n')
    receipt = agent_delivery.finish(run, 'research_director')
    assert receipt['status'] == 'delivered'
    err = capsys.readouterr().err
    assert 'changed shared git state' not in err
    assert 'changed during' not in err


def test_refused_step_writes_receipt_and_removes_pending_marker(tmp_path):
    """A refused step (target guard refusal) writes its receipt and removes its marker."""
    from refactory_dsh import target_guard, worktree
    from tests_support import git_repo

    repo = git_repo(tmp_path / 'repo')
    run = tmp_path / 'run'
    found = worktree.create(repo, 'HEAD', 'pend0000clean', run)
    target_guard.record(run, repo, output_only=('emit', 'design_doc'))
    record = agent_delivery.begin(run, 'design_doc', 'Write the design.', ['design.md'])
    # Modify the target repository to trigger target guard refusal
    import subprocess as _sp
    (repo / 'intruder.txt').write_text('unexpected change\n')
    _sp.run(['git', '-C', str(repo), 'add', 'intruder.txt'], check=True, capture_output=True)
    _sp.run(['git', '-C', str(repo), 'commit', '-q', '-m', 'intruder'], check=True, capture_output=True)
    (run / 'design.md').write_text('the design\n')
    pending = run / agent_delivery.ROOT / 'pending'
    assert any(pending.glob('*.json'))
    import pytest as _pytest
    with _pytest.raises(ValueError):
        agent_delivery.finish(run, 'design_doc')
    # Receipt is written
    receipt = json.loads((run / record['receipt']).read_text())
    assert receipt['status'] == 'failed'
    # Pending marker is gone
    assert not any(pending.glob('*.json'))


def test_missing_outputs_writes_receipt_and_removes_pending_marker(tmp_path):
    """A step with missing outputs writes its receipt and removes its pending marker."""
    record = agent_delivery.begin(tmp_path, 'build', 'Write output.txt.', ['output.txt'])
    # Do NOT write output.txt — it will be missing
    pending = tmp_path / agent_delivery.ROOT / 'pending'
    assert any(pending.glob('*.json'))
    import pytest as _pytest
    with _pytest.raises(ValueError, match='did not deliver'):
        agent_delivery.finish(tmp_path, 'build')
    receipt = json.loads((tmp_path / record['receipt']).read_text())
    assert receipt['status'] == 'failed'
    assert not any(pending.glob('*.json'))


def test_planning_steps_are_output_only_in_design_but_not_in_a_plain_locate(tmp_path, monkeypatch):
    """Mode-supplied output_only through locate(): mode isolation (#347)."""
    import subprocess
    from refactory_dsh import design_ops, target_guard, worktree
    from refactory_dsh.design import DesignMode
    from tests_support import FakeModel, git_repo

    repo = git_repo(tmp_path / 'repo')
    monkeypatch.setenv('REFACTORY_PROJECT', str(repo))

    # A design-mode run records planning steps as output-only
    design_run = tmp_path / 'design_run'
    ctx_design = nn.RunContext(design_run, model=FakeModel([]), knobs={'worktree': False},
                               task={'name': 'guard_tool', 'brief': 'a tool', 'check': 'true'})
    ctx_design.facts.update(design_ops.prepare_design(ctx_design))
    recorded_design = json.loads((design_run / target_guard.RECORD).read_text())
    for step in design_ops.PLANNING_STEPS:
        assert step in recorded_design['output_only'], f'{step} must be output-only in design'
    assert 'emit' in recorded_design['output_only']

    # A plain locate (not through prepare_design) does NOT include planning steps
    from refactory_dsh import repo_change
    plain_run = tmp_path / 'plain_run'
    ctx_plain = nn.RunContext(plain_run, model=FakeModel([]), knobs={'worktree': False},
                              task={'name': 'other_tool', 'brief': 'a tool', 'check': 'true'})
    ctx_plain.facts.update(repo_change.locate(ctx_plain, 'other_tool',
                                              dict(ctx_plain.task or {})))
    recorded_plain = json.loads((plain_run / target_guard.RECORD).read_text())
    assert recorded_plain['output_only'] == ['emit'], \
        f'plain locate must have only emit, got {recorded_plain["output_only"]}'
    for step in design_ops.PLANNING_STEPS:
        assert step not in recorded_plain['output_only'], \
            f'{step} must NOT be output-only in a plain locate'


def test_execution_contract_rebases_paths_for_worktree_runs(tmp_path):
    """Test 6: the execution contract names the worktree and rebases paths when the target is the driver."""
    from refactory_dsh import target_guard, worktree

    target = tmp_path / "target"
    wt = tmp_path / "worktree"
    run = tmp_path / "run"
    target.mkdir(parents=True)
    wt.mkdir(parents=True)
    run.mkdir(parents=True)

    # Write a worktree record.
    (run / ".refactory").mkdir(parents=True, exist_ok=True)
    import json as _json
    (run / ".refactory" / "worktree.json").write_text(_json.dumps({
        "path": str(wt), "target": str(target),
        "branch": "refactory/run-test", "base": "abc123",
        "head": "abc123", "run": str(run),
    }))

    # Record a target guard baseline.
    target_guard.record(run, target, output_only=("emit",), also=(wt,))

    # Build the contract with refactory_source inside the target.
    import importlib
    real_source = str(Path(agent_delivery.__file__).resolve().parent)
    # Only test when the real source IS inside the target; simulate this by
    # creating a fake source path under target.
    fake_source = target / "refactory" / "refactory_dsh"
    fake_source.mkdir(parents=True, exist_ok=True)
    (fake_source / "__init__.py").write_text("")

    original_file = agent_delivery.__file__
    try:
        agent_delivery.__file__ = str(fake_source / "agent_delivery.py")
        context = {'python_bin': str(Path(sys.executable).parent),
                    'runtime_root': str(target)}
        record = agent_delivery.begin(run, "emit", "Write emit.txt.", ["emit.txt"],
                                       model_context=context)
        execution = record["execution"]
        # refactory_source rebased onto worktree.
        assert str(wt) in execution["refactory_source"]
        # runtime_root rebased onto worktree.
        assert str(wt) in execution["runtime_root"]
        # Contract text mentions worktree and warns about python.
        prompt = record["prompt"]
        assert f"worktree at {wt}" in prompt
        assert f"Never change the target at {target}" in prompt
        assert "nested dispatch only" in prompt
        # The python warning names the real import root (inside the target).
        assert str(fake_source) in prompt
    finally:
        agent_delivery.__file__ = original_file


def test_python_and_native_worker_dispatch_share_execution_contract(tmp_path, monkeypatch):
    from refactory_dsh.worker_node import execute

    seen = []
    def complete(self, prompt, *, system='', workdir=None):
        seen.append(prompt)
        (workdir / 'answer.txt').write_text('42')
        return 'Done.'
    monkeypatch.setattr(DshModelClient, 'complete', complete)
    client = DshModelClient(python_bin=Path(sys.executable).parent)

    class OneNode(nn.Mode):
        name = 'contract'
        def build(self):
            graph = nn.Graph()
            graph.add(nn.Node('build', 'agent', prompt_template='Read input.txt; write answer.txt.',
                              reads=('input.txt',), writes=('answer.txt',)))
            return graph

    contexts = []
    for backend in ('python', 'native'):
        workdir = tmp_path / backend
        workdir.mkdir()
        (workdir / 'input.txt').write_text('42')
        if backend == 'python':
            result = OneNode().run(workdir, model=client)
            assert result.exit_ok, result.error
        else:
            execute({'schema_version': 1, 'node_id': 'build', 'reads': ['input.txt'],
                     'writes': ['answer.txt'], 'produces': [], 'prompt': 'Read input.txt; write answer.txt.',
                     'model_context': client.context()}, workdir)
        receipt = agent_delivery.receipts_in(workdir)['build']
        context = receipt['execution']
        assert context['workdir'] == str(workdir)
        assert context['inputs'] == [str(workdir / 'input.txt')]
        assert context['outputs'] == [str(workdir / 'answer.txt')]
        assert seen[-1].endswith('Read input.txt; write answer.txt.')
        contexts.append({key: value for key, value in context.items() if key not in ('workdir', 'inputs', 'outputs')})
    assert contexts[0] == contexts[1]


def test_redispatch_moves_previous_emit_txt_aside(tmp_path):
    """After a completed delivery, begin() again moves emit.txt under agent-delivery."""
    from refactory_dsh.repo_change import EMIT_FILE

    record = agent_delivery.begin(tmp_path, 'emit', 'Write emit.txt.', [EMIT_FILE])
    (tmp_path / EMIT_FILE).write_text('===FILE: tool.py===\nprint(1)\n')
    agent_delivery.finish(tmp_path, 'emit', final_response='Done.')
    # emit.txt is still in the run directory after finish
    assert (tmp_path / EMIT_FILE).exists()
    # A new begin() for the same node moves the stale output
    record2 = agent_delivery.begin(tmp_path, 'emit', 'Write emit.txt again.', [EMIT_FILE])
    assert not (tmp_path / EMIT_FILE).exists(), 'stale emit.txt must be gone before re-dispatch'
    # The moved file is under the previous receipt's previous/ folder
    moved = tmp_path / agent_delivery.ROOT / record['id'] / 'previous'
    assert moved.exists() and list(moved.rglob(EMIT_FILE))
    # The new dispatch's before snapshot shows no emit.txt
    assert record2['before'][EMIT_FILE] is None
    # The execution contract's set_aside field points to the moved copy
    assert 'set_aside' in record2['execution']
    assert EMIT_FILE in record2['execution']['set_aside']
    # A fresh write is treated as newly delivered
    (tmp_path / EMIT_FILE).write_text('===FILE: tool.py===\nprint(2)\n')
    receipt = agent_delivery.finish(tmp_path, 'emit', final_response='Done again.')
    assert receipt['status'] == 'delivered'


def test_redispatch_after_failed_delivery_starts_clean(tmp_path):
    """After a failed delivery, begin() again moves a surviving emit.txt aside."""
    from refactory_dsh.repo_change import EMIT_FILE

    record = agent_delivery.begin(tmp_path, 'emit', 'Write emit.txt.', [EMIT_FILE])
    # Do NOT write emit.txt — finish will raise for missing output
    with pytest.raises(ValueError, match='did not deliver'):
        agent_delivery.finish(tmp_path, 'emit')
    # Simulate the file appearing after the failed delivery (e.g. captured_final
    # wrote it, or the runner planted it for a retry)
    (tmp_path / EMIT_FILE).write_text('stale content\n')
    # A new begin() for the same node moves the stale output
    record2 = agent_delivery.begin(tmp_path, 'emit', 'Write emit.txt again.', [EMIT_FILE])
    assert not (tmp_path / EMIT_FILE).exists(), 'stale emit.txt must be gone before re-dispatch'
    moved = tmp_path / agent_delivery.ROOT / record['id'] / 'previous'
    assert moved.exists() and list(moved.rglob(EMIT_FILE))
    assert record2['before'][EMIT_FILE] is None


def test_redispatch_of_non_emit_node_moves_its_output_aside(tmp_path):
    """A redispatch of a node writing design.md moves it aside and reports it in set_aside."""
    record = agent_delivery.begin(tmp_path, 'design_doc', 'Write the design.', ['design.md'])
    (tmp_path / 'design.md').write_text('the design\n')
    agent_delivery.finish(tmp_path, 'design_doc', final_response='Done.')
    assert (tmp_path / 'design.md').exists()
    # A new begin() moves the write-only output aside
    record2 = agent_delivery.begin(tmp_path, 'design_doc', 'Revise the design.', ['design.md'])
    assert not (tmp_path / 'design.md').exists(), 'write-only output must be moved aside on re-dispatch'
    # The moved file is under the previous receipt's previous/ folder
    moved = tmp_path / agent_delivery.ROOT / record['id'] / 'previous'
    assert moved.exists() and list(moved.rglob('design.md'))
    # The execution contract's set_aside field points to the moved copy
    assert 'set_aside' in record2['execution']
    assert 'design.md' in record2['execution']['set_aside']
    prev_path = tmp_path / record2['execution']['set_aside']['design.md']
    assert prev_path.read_text() == 'the design\n'
    # The before snapshot shows design.md is gone
    assert record2['before']['design.md'] is None
    # A fresh write is treated as newly delivered
    (tmp_path / 'design.md').write_text('revised design\n')
    receipt = agent_delivery.finish(tmp_path, 'design_doc', final_response='Revised.')
    assert receipt['status'] == 'delivered'


def test_redispatch_reviewer_node_finds_previous_copy_in_contract(tmp_path):
    """A reviewer re-dispatched finds its previous output moved and its path in set_aside."""
    original_content = '# Code Review\nAll looks good.\n'
    record = agent_delivery.begin(tmp_path, 'code_reviewer', 'Review the code.', ['review.md'])
    (tmp_path / 'review.md').write_text(original_content)
    agent_delivery.finish(tmp_path, 'code_reviewer', final_response='Done.')
    assert (tmp_path / 'review.md').exists()
    # Re-dispatch
    record2 = agent_delivery.begin(tmp_path, 'code_reviewer', 'Review the code again.', ['review.md'])
    # review.md is gone from the workdir
    assert not (tmp_path / 'review.md').exists()
    # The execution contract's set_aside dict maps 'review.md' to a path under agent-delivery
    set_aside = record2['execution']['set_aside']
    assert 'review.md' in set_aside
    previous_path = tmp_path / set_aside['review.md']
    assert '.refactory/agent-delivery/' in set_aside['review.md']
    # The previous copy is readable at that path with the original content
    assert previous_path.read_text() == original_content
    # The before snapshot shows no review.md
    assert record2['before']['review.md'] is None
    # The prompt tells the worker about the moved outputs
    assert 'previous outputs were moved' in record2['prompt']
    assert 'write each output afresh' in record2['prompt']
    # A fresh review.md counts as delivered
    (tmp_path / 'review.md').write_text('# Code Review v2\nNeeds changes.\n')
    receipt = agent_delivery.finish(tmp_path, 'code_reviewer', final_response='Done again.')
    assert receipt['status'] == 'delivered'


def test_redispatch_read_write_file_stays_in_place(tmp_path):
    """A node that reads and writes the same file keeps it on re-dispatch."""
    (tmp_path / 'state.json').write_text('{"step": 0}\n')
    record = agent_delivery.begin(tmp_path, 'derive', 'Update state.', ['state.json'],
                                  reads=['state.json'])
    (tmp_path / 'state.json').write_text('{"step": 1}\n')
    agent_delivery.finish(tmp_path, 'derive', final_response='Done.')
    assert (tmp_path / 'state.json').exists()
    # Re-dispatch: state.json stays because it is also a read
    record2 = agent_delivery.begin(tmp_path, 'derive', 'Update state again.', ['state.json'],
                                   reads=['state.json'])
    assert (tmp_path / 'state.json').exists(), 'read-and-write file must stay in place'
    assert (tmp_path / 'state.json').read_text() == '{"step": 1}\n'
    # No set_aside in the execution contract (nothing moved)
    assert record2['execution'].get('set_aside') is None
    # The before snapshot shows the file is present
    assert record2['before']['state.json'] is not None


def test_first_dispatch_leaves_other_steps_output_in_place(tmp_path):
    """A first dispatch does not move a file another step wrote."""
    # Simulate another step having written design.md
    (tmp_path / 'design.md').write_text('another step wrote this\n')
    # First begin() for design_doc — no prior receipt exists
    record = agent_delivery.begin(tmp_path, 'design_doc', 'Write the design.', ['design.md'])
    assert (tmp_path / 'design.md').exists(), 'no-receipt file must stay in place'
    assert (tmp_path / 'design.md').read_text() == 'another step wrote this\n'
    # set_aside is absent or None
    assert record['execution'].get('set_aside') is None


def test_main_bridge_renders_templates_before_begin(tmp_path, monkeypatch):
    """The bridge's begin action renders {{templates}} before calling begin()."""
    import io
    (tmp_path / 'facts.json').write_text(json.dumps({'artifact_name': 'rendered_value'}))
    Ledger.seed(tmp_path, 'test')
    payload = json.dumps({
        'action': 'begin', 'node': 'emit',
        'prompt': 'Build {{artifact_name}}.', 'writes': ['out.txt'], 'reads': [],
    })
    monkeypatch.setattr('sys.stdin', io.StringIO(payload))
    monkeypatch.chdir(tmp_path)
    code = agent_delivery.main()
    assert code == 0
    # The recorded prompt.txt should contain the rendered value
    pending_dir = tmp_path / '.refactory/agent-delivery/pending'
    pending_files = list(pending_dir.glob('*.json'))
    assert len(pending_files) == 1
    record = json.loads(pending_files[0].read_text())
    assert 'rendered_value' in record['prompt']
    assert '{{artifact_name}}' not in record['prompt']


def test_main_bridge_warns_unrendered_templates(tmp_path, monkeypatch):
    """An unknown template key in the mode's own template yields a warning, not a refusal."""
    import io
    (tmp_path / 'facts.json').write_text(json.dumps({}))
    Ledger.seed(tmp_path, 'test')
    payload = json.dumps({
        'action': 'begin', 'node': 'emit',
        'prompt': 'Build {{unknown_key}}.', 'writes': ['out.txt'], 'reads': [],
    })
    captured = io.StringIO()
    monkeypatch.setattr('sys.stdin', io.StringIO(payload))
    monkeypatch.setattr('sys.stdout', captured)
    monkeypatch.chdir(tmp_path)
    code = agent_delivery.main()
    assert code == 0
    data = json.loads(captured.getvalue())
    assert 'unresolved_keys' in data
    assert 'unknown_key' in data['unresolved_keys']


def test_main_bridge_accepts_templates_inside_substituted_values(tmp_path, monkeypatch):
    """A brief containing {{templates}} is accepted when substituted into the prompt."""
    import io
    (tmp_path / 'facts.json').write_text(json.dumps({
        'brief': 'Fix the {{templates}} issue',
    }))
    Ledger.seed(tmp_path, 'test')
    payload = json.dumps({
        'action': 'begin', 'node': 'emit',
        'prompt': 'Brief: {{brief}}', 'writes': ['out.txt'], 'reads': [],
    })
    monkeypatch.setattr('sys.stdin', io.StringIO(payload))
    monkeypatch.chdir(tmp_path)
    code = agent_delivery.main()
    assert code == 0
    # The rendered prompt should contain the brief's text including {{templates}}
    pending_dir = tmp_path / '.refactory/agent-delivery/pending'
    pending_files = list(pending_dir.glob('*.json'))
    assert len(pending_files) == 1
    record = json.loads(pending_files[0].read_text())
    assert '{{templates}}' in record['prompt']


def test_main_bridge_no_warning_for_keys_in_steering_ledger(tmp_path, monkeypatch):
    """A ledger entry containing {{x}} does not produce an unresolved key warning."""
    import io
    from refactory_dsh.steering import STEERING_START, STEERING_END
    (tmp_path / 'facts.json').write_text(json.dumps({'artifact_name': 'ok'}))
    Ledger.seed(tmp_path, 'test brief with {{x}} inside')
    # The steering ledger folds the brief (which contains {{x}}) into the template.
    # The scan should strip the STEERING block and not warn about {{x}}.
    payload = json.dumps({
        'action': 'begin', 'node': 'emit',
        'prompt': 'Build {{artifact_name}}.', 'writes': ['out.txt'], 'reads': [],
    })
    captured = io.StringIO()
    monkeypatch.setattr('sys.stdin', io.StringIO(payload))
    monkeypatch.setattr('sys.stdout', captured)
    monkeypatch.chdir(tmp_path)
    code = agent_delivery.main()
    assert code == 0
    data = json.loads(captured.getvalue())
    # Should not contain 'x' as an unresolved key
    unresolved = data.get('unresolved_keys', [])
    assert 'x' not in unresolved


def test_main_bridge_finish_records_sent_prompt(tmp_path, monkeypatch):
    """begin then finish through main() records sent_prompt_hash, sent_prompt_path and unresolved_keys in receipt."""
    import io
    (tmp_path / 'facts.json').write_text(json.dumps({'artifact_name': 'val'}))
    Ledger.seed(tmp_path, 'test')

    # Step 1: begin
    begin_payload = json.dumps({
        'action': 'begin', 'node': 'emit',
        'prompt': 'Build {{artifact_name}}.', 'writes': ['out.txt'], 'reads': [],
    })
    begin_out = io.StringIO()
    monkeypatch.setattr('sys.stdin', io.StringIO(begin_payload))
    monkeypatch.setattr('sys.stdout', begin_out)
    monkeypatch.chdir(tmp_path)
    assert agent_delivery.main() == 0
    begin_data = json.loads(begin_out.getvalue())
    assert 'prompt' in begin_data

    # Write the output file
    (tmp_path / 'out.txt').write_text('done\n')

    # Step 2: finish with sent_prompt_hash, sent_prompt and unresolved_keys
    import hashlib
    sent_text = 'The full prompt including Build val.'
    sent_hash = hashlib.sha256(sent_text.encode()).hexdigest()
    finish_payload = json.dumps({
        'action': 'finish', 'node': 'emit',
        'sent_prompt_hash': sent_hash,
        'sent_prompt': sent_text,
        'unresolved_keys': ['some_key'],
    })
    finish_out = io.StringIO()
    monkeypatch.setattr('sys.stdin', io.StringIO(finish_payload))
    monkeypatch.setattr('sys.stdout', finish_out)
    assert agent_delivery.main() == 0

    # Verify receipt.json
    delivery_dirs = list((tmp_path / '.refactory/agent-delivery').glob('*/receipt.json'))
    assert len(delivery_dirs) == 1
    receipt = json.loads(delivery_dirs[0].read_text())
    assert receipt['sent_prompt_hash'] == sent_hash
    assert 'sent_prompt_path' in receipt
    assert receipt['unresolved_keys'] == ['some_key']
    # Verify sent-prompt.txt was written
    sent_path = delivery_dirs[0].parent / 'sent-prompt.txt'
    assert sent_path.exists()
    assert sent_path.read_text() == sent_text


def test_receipt_records_prompt_hash(tmp_path):
    """begin() and finish() include prompt_hash in the receipt."""
    Ledger.seed(tmp_path, 'test')
    record = agent_delivery.begin(tmp_path, 'emit', 'Write emit.txt.', ['emit.txt'])
    assert 'prompt_hash' in record
    assert len(record['prompt_hash']) == 64  # SHA-256 hex
    (tmp_path / 'emit.txt').write_text('content\n')
    receipt = agent_delivery.finish(tmp_path, 'emit')
    assert receipt.get('prompt_hash') == record['prompt_hash']


def test_reviewers_action_returns_reviewers_list(tmp_path, monkeypatch):
    """The reviewers bridge action returns the REVIEWERS list."""
    import io
    from refactory_dsh import authoring_evidence
    payload = json.dumps({'action': 'reviewers'})
    monkeypatch.setattr('sys.stdin', io.StringIO(payload))
    monkeypatch.chdir(tmp_path)
    import subprocess
    result = subprocess.run(
        [sys.executable, '-m', 'refactory_dsh.agent_delivery'],
        input=payload, capture_output=True, text=True, cwd=str(tmp_path),
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert set(data['reviewers']) == set(authoring_evidence.REVIEWERS)


def test_reused_pending_dispatch_moves_nothing(tmp_path):
    """A reused pending dispatch does not move files — it keeps the original snapshot."""
    record = agent_delivery.begin(tmp_path, 'emit', 'Write emit.txt.', ['emit.txt'])
    # File is written but not finished — dispatch is still pending
    (tmp_path / 'emit.txt').write_text('in progress\n')
    # Reuse the pending dispatch (same prompt → early return)
    record2 = agent_delivery.begin(tmp_path, 'emit', 'Write emit.txt.', ['emit.txt'], reuse=True)
    # Same dispatch reused — nothing moved
    assert record2['id'] == record['id']
    assert (tmp_path / 'emit.txt').exists(), 'reused dispatch must not move files'


def test_two_runs_on_one_repository_one_run_s_setup_installs_hooks_the_other_is_unaffected(tmp_path):
    """o55-dm149d: another run's setup and another session's branch must not fail this run's step.

    Run B's setup installs a hook the way lefthook and husky's postinstall do (into
    `git rev-parse --git-path hooks`), and a session makes a tracking branch in the
    person's checkout. Run A's output-only step is still delivered, and a real change
    to the shared hooks still fails it, naming the file.
    """
    import subprocess
    from refactory_dsh import repo_change, target_guard, worktree
    from tests_support import git, git_repo

    repo = git_repo(tmp_path / 'repo')
    git(repo, 'remote', 'add', 'origin', 'https://example.invalid/repo.git')
    git(repo, 'update-ref', 'refs/remotes/origin/main', 'HEAD')
    run_a, run_b = tmp_path / 'run-a', tmp_path / 'run-b'
    worktree.create(repo, 'HEAD', 'twoA0000aaaa', run_a)
    found_b = worktree.create(repo, 'HEAD', 'twoB0000bbbb', run_b)
    target_guard.record(run_a, repo, output_only=('design_doc',))
    agent_delivery.begin(run_a, 'design_doc', 'Write the design.', ['design.md'])

    install = ('hooks=$(git rev-parse --git-path hooks) && mkdir -p "$hooks" && '
               'printf "#!/bin/sh\\nexit 0\\n" > "$hooks/pre-commit" && chmod +x "$hooks/pre-commit"')
    home = tmp_path / 'home-b'
    home.mkdir()
    env = worktree.check_environment(Path(found_b.path), home)
    _found, problems, failed = repo_change.prepare_commands(
        repo_change._PrepCtx({'setup': install}, workdir=run_b), found_b, env, home, 60, None,
        amend=False)
    assert problems == [] and failed is None
    installed = Path(git(Path(found_b.path), 'rev-parse', '--path-format=absolute',
                         '--git-path', 'hooks')) / 'pre-commit'
    assert installed.is_file(), "run B's setup installed its hook in its own hooks"
    git(repo, 'worktree', 'add', '-q', '-b', 'fix/other-session', str(tmp_path / 'session'),
        'origin/main')

    (run_a / 'design.md').write_text('the design\n')
    assert agent_delivery.finish(run_a, 'design_doc')['status'] == 'delivered'

    agent_delivery.begin(run_a, 'design_doc', 'Write the design again.', ['design.md'])
    shared_hooks = Path(subprocess.run(
        ['git', '-C', str(repo), 'rev-parse', '--path-format=absolute', '--git-common-dir'],
        capture_output=True, text=True, check=True).stdout.strip()) / 'hooks'
    shared_hooks.mkdir(exist_ok=True)
    (shared_hooks / 'pre-commit').write_text('#!/bin/sh\nexit 0\n')
    (run_a / 'design.md').write_text('the design, again\n')
    with pytest.raises(ValueError, match=r'changed shared git state: .*\(hook pre-commit added\)'):
        agent_delivery.finish(run_a, 'design_doc')
