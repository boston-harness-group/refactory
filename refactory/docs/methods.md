# Adding an optimization method

refactory is meant to be the PyTorch of harnesses: an optimization method (an outer
loop, a parent policy, a proposer) is put together from building blocks that already
exist, not written as a private copy of them. This page lists those blocks, where they
live, and the rule for adding one.

## The rule

1. **Compose what exists.** A new method uses the core parent pickers, proposers,
   validation gate, final evaluation, budgets and trace below, rather than writing
   its own.
2. **Register through the package route.** A new parent picker or proposer goes in a
   package's `Package.optimizers`; a new outer loop goes in `Package.strategies`. Both are
   then selectable by name through the normal settings (`optimizer`, `parent_picker`,
   `strategy`), with no new CLI flag and no edit to a core table.
3. **A genuinely new general piece goes in core, once.** If the method needs something
   other methods could use (a selection rule, a frontier, a sampler), put it in
   `refactory_dsh` (one copy) and make the method import it. A method package keeps
   only what is specific to that method.
4. **A faithful port says so and pins it.** A port that must reproduce an upstream
   algorithm exactly says so in its module docstring and pins its behaviour with a
   fixed-seed test that compares the whole decision trace, so moving a shared piece
   into core cannot change it unnoticed.

Review checks this: a new method reuses the core pickers, proposers, gate, final
evaluation and trace, and any helper it duplicated has been moved to core.

## The building blocks

| block | where | what it gives a method |
|---|---|---|
| parent pickers: `archive`, `evolutionary`, `islands`, `hill_climb`, `map_elites` | `refactory_dsh/optim/policies.py` (`ParentSelector`) | which candidate to edit next. Write `select` only. Each picker gets `ProposalContext.val_scores` (one val mean per candidate) and `ProposalContext.val_item_scores` (each candidate's per-item val scores, keyed by the run's HMAC of the item id, never test). It draws from `ParentSelector.rng`, so a resumed run picks again what it picked before |
| proposers: `autoresearch` (the model proposer), `random` | `refactory_dsh/optim/random_optimizer.py` (`Optimizer`) | the edits. Write `propose`. The base class validates every proposal before it costs an evaluation. A proposer that calls a model gets the run's model through `bind_model` |
| picker + proposer pairing | `refactory_dsh/optim/compose.py` (`PickedParent`) | a picker chooses the parent and any proposer writes the edits (`parent_picker` setting) |
| registration | `optim.register` (core's own entries), `Package.optimizers`, `Package.strategies` (`refactory_dsh/nn/package.py`), both found by `registry.contributed` | selectable by name. `optim.resolve` / `strategies.resolve` look in core first, then in packages. Profiles validate both |
| the strategy API | `refactory_dsh/outerloop.py` (`OuterLoop`, `Run`) | `Run.attempt` (apply and validate a proposal), `Run.evaluate` / `Run.evaluate_many` (`EvalRequest`: a candidate, a part, chosen items, a seed), `Run.part_item_scores` (per-item val scores, HMAC-keyed, val only), `Run.shared_scores`, budgets (`Run.budget_left`, `Run.rollouts_left`, `BudgetStopped`), `Run.keep`, `Run.stop`, `Run.save_strategy_state` |
| validation gate | `outerloop.ValidationGate` | accept a candidate only on a strictly higher val score; the incumbent; shared-item scoring when an incident left an item out |
| final evaluation | `outerloop.FinalEvaluation` | score the winner and the seed on test once, at the end |
| strategy trace | `refactory_dsh/strategy_trace.py` (`TraceRecorder`, `ask`) | one JSONL trace of decisions and model calls, with retries and call keys, so a run can be replayed |
| early stop | `OuterLoop.should_stop` | a hook a strategy overrides to end the search early |
| train minibatches | `train_minibatch` setting (default loop); `EvalRequest.items` (strategies) | score a candidate on a sample of train rather than all of it |

## Worked examples

- **GEPA** (`refactory_packages/gepa`): GEPA reflection (its prompt, its
  instruction extractor, its merge) plus the core blocks: `Run.part_item_scores`
  for its per-item Pareto frontier, `ValidationGate`, `FinalEvaluation` and
  `TraceRecorder`. The frontier functions are about to become a core `pareto`
  parent picker that GEPA also uses, as the next step of this work.
- **SkillOpt** (`refactory_packages/skillopt`): its analysts, merge and slow update
  (`SkillOptProposer`), plus `ValidationGate`, `FinalEvaluation`, `TraceRecorder`
  and `Run.evaluate_many`. Its early stop is the `should_stop` hook.

## A picker and a proposer in a package

```python
from refactory_dsh.nn import Package
from refactory_dsh.optim import Optimizer, ParentSelector, Proposal


class Freshest(ParentSelector):
    """Breed from the newest candidate."""

    name = "freshest"
    policy = "the newest candidate"

    def select(self, ctx, n, parents):
        return list(parents[-1:]) * n


class SetTone(Optimizer):
    name = "set_tone"

    def __init__(self, *, k=3, seed=0):
        super().__init__(k=k)

    def propose(self, ctx, n, parents):
        return [Proposal(edit={"op": "set_knob", "name": "tone", "value": "terse"},
                         parent=parents[0].id)]


PACKAGE = Package(name="my_methods", optimizers=(Freshest, SetTone))
```

Then a run selects them by name: `--optimizer set_tone`, or `parent_picker: freshest`
in a profile, paired with any proposer. A picker keeps `population = True` (the
default) to get every judged candidate under `selection: pool`, and set it to `False`
for a single-line picker, as `hill_climb` does, which keeps `selection: checkpoint`
as its default. `tests/test_package_optimizers.py` runs exactly this route end to end.

More guides (benchmark packages, edit operations, node kinds and judges) will follow.
