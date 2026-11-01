"""The lowered native path must admit initial fast workers before any model call."""
import json

import pytest

from refactory_dsh import fast_ops, lowering, workers
from refactory_dsh.fast import CreateFastMode, EditFastMode, DesignFastMode
from refactory_dsh.model import DshModelClient
from refactory_dsh.nn.executor import RunContext
from refactory_dsh.worker_node import execute
from tests_support import FakeModel


#: Design's prepare checks the model credential (dm-257): the test's own home has a placeholder key.
pytestmark = pytest.mark.usefixtures("configured_model_home")


@pytest.mark.parametrize('mode_class,node_id', [
    (CreateFastMode, 'strategy'), (CreateFastMode, 'emit'),
    (EditFastMode, 'emit'), (DesignFastMode, 'strategy'), (DesignFastMode, 'emit')])
def test_initial_fast_inputs_reach_lowered_worker_dispatch(tmp_path, monkeypatch, mode_class, node_id):
    monkeypatch.setenv('REFACTORY_PROJECT', str(tmp_path))
    mode = mode_class()
    ctx = RunContext(tmp_path / 'run', model=FakeModel([]), knobs={**mode.values(), 'single_shot': True},
                     task={'name': 'admission_control', 'brief': 'a tiny reusable greeting', 'check': 'true'})
    prepare = fast_ops.prepare_artifact if mode_class is DesignFastMode else fast_ops.prepare_package
    ctx.facts.update(prepare(ctx))
    if mode_class is not DesignFastMode:
        ctx.facts.update(fast_ops.survey_package(ctx))
    if node_id == 'emit' and mode_class is not EditFastMode:
        ctx.write_text('implementation-plan.txt', 'Implement the brief, then review the requested check.')
        ctx.facts['implementation_plan'] = ctx.read_text('implementation-plan.txt')
    (ctx.workdir / 'facts.json').write_text(json.dumps(ctx.facts))
    workers.lower_workflow(lowering.to_workflow(mode.state_dict()), ctx.workdir, DshModelClient().context())
    spec = next(json.loads(path.read_text()) for path in (ctx.workdir / '.refactory/worker-specs').glob('*.json')
                if json.loads(path.read_text())['node_id'] == node_id)
    assert all((ctx.workdir / name).exists() for name in spec['reads'])
    assert 'no repair requested' in (ctx.workdir / 'repair.txt').read_text()
    def dispatch(*args, **kwargs):
        raise RuntimeError('admission passed; deterministic stop before provider')
    monkeypatch.setattr(DshModelClient, 'complete', dispatch)
    with pytest.raises(RuntimeError, match='admission passed'):
        execute(spec, ctx.workdir)


def test_a_worktree_off_design_fast_run_reaches_focused_qa_on_the_spine(tmp_path, monkeypatch):
    """Each step runs through the node runner, as the spine runs it. A worktree-off run
    writes no test-guardedness.json, so focused_qa may not declare it: a declared input
    that is missing refuses the dispatch (dm-152)."""
    import io
    import sys

    from refactory_dsh.run import main
    from tests_support import render

    monkeypatch.setenv('REFACTORY_PROJECT', str(tmp_path))
    run = tmp_path / 'run'
    run.mkdir()
    mode = DesignFastMode()
    lowering.write_declarations(mode.state_dict(), run)
    monkeypatch.setenv('REFACTORY_KNOBS', json.dumps({
        **mode.values(), 'name': 'greeting', 'brief': 'a tiny greeting', 'check': 'true', 'worktree': False}))

    def step(target: str, node: str, kind: str) -> None:
        monkeypatch.setenv('REFACTORY_WORKDIR', str(run))
        monkeypatch.setenv('DSH_WORKFLOW_NODE', json.dumps({'id': node, 'type': kind}))
        err, saved = io.StringIO(), sys.stderr
        sys.stderr = err
        try:
            code = main([target, '--node', node])
        finally:
            sys.stderr = saved
        assert code == 0, f'{node}: {err.getvalue()}'

    step('refactory_dsh.fast_ops:prepare_artifact', 'prepare', 'op')
    (run / 'implementation-plan.txt').write_text('Write greeting.py.\n')
    (run / 'emit.txt').write_text(render({'greeting.py': "print('hi')\n"}))
    step('refactory_dsh.fast_ops:materialise_artifact', 'materialise', 'op')
    step('refactory_dsh.fast_ops:gate_emit', 'gate_emit', 'gate')
    step('refactory_dsh.fast_ops:focused_check', 'focused_check', 'op')
    # test-guardedness.json is declared in the node's writes and written through
    # ctx.write_json with a placeholder when there is no worktree.
    assert (run / 'test-guardedness.json').exists(), 'test-guardedness.json should be written'
    content = json.loads((run / 'test-guardedness.json').read_text())
    assert 'unavailable' in content, 'non-worktree guardedness should be a placeholder'

    workers.lower_workflow(lowering.to_workflow(mode.state_dict()), run, DshModelClient().context())
    spec = next(json.loads(path.read_text()) for path in (run / '.refactory/worker-specs').glob('*.json')
                if json.loads(path.read_text())['node_id'] == 'focused_qa')

    def dispatch(*args, **kwargs):
        raise RuntimeError('admission passed; deterministic stop before provider')
    monkeypatch.setattr(DshModelClient, 'complete', dispatch)
    with pytest.raises(RuntimeError, match='admission passed'):
        execute(spec, run)
