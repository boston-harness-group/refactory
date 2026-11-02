# Plan: GEPA and MIPROv2 as outer-loop options

Status: **proposed**. Adds two research optimizers to the outer loop and the
evidence that they are implemented correctly. AFlow, ADAS and a TextGrad-style
credit-assignment operator are follow-ups that reuse the Phase 0 plumbing.

## Why these two

The loop already has quality-diversity and evolutionary search (`map_elites`,
`islands`, `evolutionary`, `archive`). What it lacks is the two ideas that make
prompt optimization sample-efficient:

- **GEPA** (Agrawal et al., 2025): reflective rewriting of one module's prompt
  from execution traces and grader feedback, with a per-instance Pareto frontier
  that keeps specialists alive instead of collapsing to the mean.
- **MIPROv2** (Opsahl-Ong et al., 2024): bootstrapped few-shot demonstrations
  plus grounded instruction proposals, searched jointly per module with TPE.

Sample efficiency matters here because an evaluation is expensive: an Opus
baseline on eleven chess puzzles took about fifty minutes.

## Where the codebase stands

- `strategies.Reflective` is labelled "SRF `gepa`" but is not GEPA: its merge
  copies knob values between parents, and it has no Pareto frontier, no
  trace-driven reflective mutation and no minibatch acceptance gate.
- Per-item scores already exist: data nodes record `item_scores`
  (`items.py`, `nn/executor.py`). The frontier's input is computed and not used.
- `ProposalContext.traces` carries one aggregate gradient, not per-item traces.
- The edit catalogue has no few-shot demo edit.
- Holdout splitting exists (`task_module.write_subset`, `Loop.audit_holdout`).
- The proposer already resolves through `for_node("optimizer", role="optimizer")`,
  so `role_routes` can give the optimizer a different model from the workers.

## Decisions

| Question | Decision | Why |
|---|---|---|
| Replace `reflective`? | No. Add `gepa` alongside it. | `reflective` is a documented SRF port with tests and results; changing its behaviour invalidates them. Keeping both gives a direct ablation. Alias or deprecate later if `gepa` dominates. |
| Optuna? | Yes, as the optional extra `refactory-dsh[optim]`. | DSPy's MIPROv2 uses Optuna's TPE, so parity has one fewer variable. Its dependency tree is too heavy for core. `mipro` imports it lazily and fails loud with the install command. |
| Parity task | HotpotQA, distractor setting. | Both papers report on it; it is multi-module, so per-module reflection and merge are exercised; distractor paragraphs ship with each question, so no retrieval server. GSM8K is single-module and near ceiling. |
| Models | Task: GLM on the `dsh` worker. Optimizer: Sonnet, fixed across every optimizer. Confirmation: Opus. | The comparison grid is large; GLM is cheap, already routed, and leaves headroom. Both methods assume a strong proposer, and a varying proposer would confound the comparison. |

## Phase 0: Shared groundwork

Tracked in #564.

1. **Per-item score vector** on `Candidate`, read from the `item_scores` fact
   and keyed by item id.
2. **Grader feedback**: optional `feedback: str` on `VerifyResult`, so a task can
   say why an item failed. Reflection without it is guesswork.
3. **Per-item traces** in `ProposalContext`: input, output, score and feedback
   per item, bounded.
4. **Minibatch evaluation**: the loop can score a candidate on a named subset of
   train items. Generalize `write_subset`.
5. **Unified budget accounting**: item-evaluations, model calls and tokens,
   counted identically for every optimizer and recorded in
   `resolved-profile.json` and the run report.
6. **Leakage guard**: holdout ids are tagged, and a test fails if one reaches the
   proposal context, traces, demos or a Pareto set.
7. **Optimizer route**: test that `role_routes: {"optimizer": ...}` reaches the
   proposer, and record the task and optimizer routes in the report.
8. **Fake-model simulation harness**: a deterministic model whose output depends
   on features of the prompt, plus two planted tasks:
   - *category-specialist*: item categories each need a different instruction;
   - *demo-sensitive*: only specific demonstrations help.

Exit: the new tests pass, the existing optimizer tests are unchanged, and the
leakage test fails when a holdout item is deliberately leaked.

## Phase 1: GEPA (`gepa` strategy)

Tracked in #565.

- Split train into a feedback set (minibatches for reflection) and a Pareto set.
- **Frontier**: candidates that are best on at least one Pareto item. Dominated
  candidates are pruned.
- **Parent sampling**: weighted by the number of Pareto items a candidate wins.
- **Reflective mutation**: choose one prompt slot round-robin over agent nodes;
  give the proposer the current prompt with minibatch traces and feedback; it
  returns a new prompt, applied as `set_knob`.
- **Acceptance gate**: a child is kept only if it beats its parent on the same
  minibatch, and is then scored on the full Pareto set.
- **System-aware merge**: take node-level components from two lineages that share
  an ancestor and changed different nodes.
- **Artifacts**: lineage, frontier per iteration, and every reflection prompt and
  response, saved to the run directory.

Tests: each invariant on the fake model. On the category-specialist task the
frontier keeps a specialist per category, and greedy selection keeps fewer.

## Phase 2: MIPROv2 (`mipro` optimizer)

Tracked in #566.

- **Demo edit**: a `set_demos` op or a `demos` knob kind, rendered through
  `{{demos}}`, with its own render test.
- **Bootstrap**: run the seed mode on train items and keep only traces that pass
  the grader; build K candidate demo sets per node.
- **Instruction proposals**: N per node, grounded on a dataset summary, a program
  summary, the bootstrapped demos and a sampled tip.
- **Search**: Optuna TPE over (instruction, demo set) per node, on minibatches,
  with the incumbent re-scored on the full set every M trials.

Tests: demos come only from passing train traces; the proposal prompt contains
its grounding; a seed reproduces a run; the search finds the planted demos on the
demo-sensitive task.

## Phase 3: Reference parity with DSPy

Tracked in #567.

- Port a HotpotQA distractor program to an `nn.Mode`: select paragraphs, write a
  second-hop query, select again, answer. Score with exact match and F1.
- A script under `experiments/` runs `dspy.GEPA`, `dspy.MIPROv2`, `gepa` and
  `mipro` with the same task model, optimizer model, split and rollout budget,
  three seeds each.

Exit: our holdout gains are within the seed confidence interval of DSPy's, the
learning curves have a similar shape, and the budget splits the same way between
proposal and evaluation.

## Phase 4: Benchmarks and ablations

Tracked in #568.

- **Chess v2**: 150+ puzzles stratified by Lichess rating, a fixed holdout of 50
  or more, and python-chess scoring that accepts any line that forces mate. It
  needs a tracked package of its own.
- **A structure-sensitive benchmark**: a small SWE-bench-lite slice, or authoring
  modes scored on their regression tests.
- **Grid**: `gepa`, `mipro`, `reflective`, `evolutionary`, `hill_climb` and
  `random`, at an equal budget, with three seeds, on GLM. Then the top two on Opus.
- **Ablations**, with the expected direction:
  - GEPA with greedy selection instead of the frontier: no better;
  - GEPA with score-only feedback: clearly worse;
  - MIPRO with instructions only, and with demos only: each no better than full;
  - both methods against `random` at an equal budget: both win.

An implementation whose ablations come out the wrong way is treated as a bug
until shown otherwise.

## Sequencing

Phase 0 comes first. Phases 1 and 2 can run in parallel, each followed by its half
of Phase 3. Phase 4 comes last.
