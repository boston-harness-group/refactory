"""A nested search and its validators consume the enclosing allocation."""
import json
from pathlib import Path

import child_patches
from refactory_dsh import nn
from refactory_dsh.budgets import LEDGER, RunBudget
from refactory_dsh.loop import Loop, LoopConfig
from test_loop import _BoundedInt, _patch_mode
from test_loop_as_mode import _mode


def spend(ctx):
    assert ctx.model is None
    ctx.budget.record_usage(str(ctx.workdir), 1)
    ctx.write_text('proof.txt', 'measured control\n')


def finish(ctx):
    ctx.write_text('validated.txt', 'ran\n')


class MeteredControl(nn.Mode):
    name = 'metered'
    def build(self):
        graph = nn.Graph()
        graph.add(nn.Node('work', 'op', call=f'{__name__}:spend', writes=('proof.txt',)))
        graph.add(nn.Node('validation', 'op', role='validation', call=f'{__name__}:finish', writes=('validated.txt',)))
        graph.connect('work', 'validation')
        return graph
    def score(self, run):
        return 1.0


def test_nested_search_reuses_allocation_without_pausing_or_extending(tmp_path, monkeypatch):
    budget = RunBudget.create(tmp_path / 'outer', {'max_tokens': 10, 'validation_reserve_tokens': 2})
    budget.record_usage('before', 3)
    started = budget.snapshot()['started_at']
    monkeypatch.setenv('REFACTORY_BUDGET_LEDGER', str(budget.path))
    with _patch_mode(_BoundedInt):
        loop = Loop('fake/bounded', LoopConfig(epochs=1, batch_size=1, optimizer='random',
                    max_tokens=1000, runs_root=tmp_path / 'inner'), project=Path('.'))
        report = loop.run()
    assert report['best'] is not None
    assert loop.model is None
    assert loop.budget.path == budget.path
    assert not (tmp_path / 'inner' / LEDGER).exists()
    state = budget.snapshot()
    assert state['tokens'] == 3 and state['limits']['max_tokens'] == 10
    assert state['started_at'] == started and state['phase'] == 'work'


def test_candidate_validator_keeps_holdout_reserve_and_stays_ineligible(tmp_path):
    with _patch_mode(MeteredControl):
        loop = Loop('fake/metered', LoopConfig(epochs=1, batch_size=1, optimizer='random',
                    max_tokens=2, validation_reserve_tokens=1, runs_root=tmp_path), project=Path('.'))
        report = loop.run()
    assert report['best'] is None and report['exit_ok'] is False
    # The budget cut the candidate's run: it is not scored as a failure but left
    # unevaluated (named in the report) for a resume with more budget.
    assert report['failed_candidates'] == [] and report['unevaluated']
    stopped = [json.loads(line) for line in (tmp_path / 'journal.jsonl').read_text().splitlines()
               if json.loads(line).get('event') == 'budget_stopped']
    assert len(stopped) == 1 and 'reserve retained' in stopped[0]['error']
    assert not list((tmp_path / 'runs').rglob('validated.txt'))
    assert loop.budget.snapshot()['tokens'] == 1


class _MeteredPackage:
    name = 'fake'
    description = ''

    def mode(self, name):
        return MeteredControl


def resolve_metered(project, identifier):
    """`registry.resolve`, for `fake/metered`: the loop's ops resolve their target with it."""
    return _MeteredPackage(), MeteredControl


def test_loop_mode_nodes_rehydrate_the_same_shared_budget(tmp_path, monkeypatch):
    import refactory_dsh.registry as registry

    mode = _mode(target='fake/metered', epochs=1, batch_size=1, max_evaluations=1)
    budget = RunBudget.create(tmp_path / 'outer', {'max_tokens': 10, 'validation_reserve_tokens': 2})
    # The loop's steps run in the walker's op children: the fake target reaches them there.
    child_patches.for_test(monkeypatch).setattr(registry, 'resolve', resolve_metered)
    result = mode.run(tmp_path / 'outer', shared_budget=budget)
    assert result.exit_ok, result.error
    assert budget.snapshot()['tokens'] == 1
    assert budget.snapshot()['phase'] == 'work'
    assert not (tmp_path / 'outer/loop' / LEDGER).exists()


def test_custom_model_without_usage_is_refused_before_walker_call(tmp_path):
    class Custom:
        calls = 0
        def complete(self, prompt, *, system=''):
            self.calls += 1
            return 'fictional result'
    class Agent(nn.Mode):
        name = 'agent'
        def build(self):
            graph = nn.Graph()
            graph.add(nn.Node('agent', 'agent', prompt_template='write answer', writes=('answer.txt',)))
            return graph
    model = Custom()
    result = Agent().run(tmp_path, model=model, resource_limits={'max_tokens': 10})
    assert not result.exit_ok and 'no token telemetry' in result.error
    assert model.calls == 0
    assert RunBudget.open(tmp_path).snapshot()['tokens'] is None


def test_custom_proposer_is_refused_at_call_but_deterministic_target_still_runs(tmp_path):
    class Custom:
        calls = 0
        def complete(self, prompt, *, system=''):
            self.calls += 1
            return '{"proposals": []}'
    model = Custom()
    with _patch_mode(_BoundedInt):
        loop = Loop('fake/bounded', LoopConfig(epochs=2, batch_size=1, optimizer='autoresearch',
                    max_tokens=10, runs_root=tmp_path), project=Path('.'), model=model)
        report = loop.run()
    assert model.calls == 0
    assert report['evaluations'] == 1
    assert 'no token telemetry' in report['bound']
    assert report['proposer']['all_failed'] is True
    assert report['proposer']['failure_count'] == 1
    assert report['resources']['tokens'] is None
