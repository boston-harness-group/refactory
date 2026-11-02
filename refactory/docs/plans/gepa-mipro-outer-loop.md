# Plan: GEPA and MIPROv2 as outer-loop options

Status: **proposed**. Adds two research optimizers to the outer loop, plus the
evidence that they are implemented correctly. Both are built the way SkillOpt is
built (#559): as strategy packages on `OuterLoop` and the public `Run` API. AFlow,
ADAS and a TextGrad-style credit-assignment operator are follow-ups.

## Why these two

The loop has quality-diversity and evolutionary search (`map_elites`, `islands`,
`evolutionary`, `archive`). It has trace-driven text optimization of a single
skill document (`skillopt`). What it lacks:

- **GEPA** (Agrawal et al., 2025): reflective rewriting of *one module's* prompt
  at a time, with a per-instance Pareto frontier that keeps specialists alive
  instead of collapsing to the mean, and a merge that combines modules from
  different lineages.
- **MIPROv2** (Opsahl-Ong et al., 2024): bootstrapped few-shot demonstrations
  plus grounded instruction proposals, searched jointly per module with TPE.

Sample efficiency matters because an evaluation is expensive: an Opus baseline on
eleven chess puzzles took about fifty minutes.

## What `main` already provides

Most of the plumbing these optimizers need landed with #554, #556, #558 and #559:

| Need | Provided by |
|---|---|
| Per-item scores and outcomes | `Run.item_records(candidate)`: `id`, `score`, `passed`, `prediction`, `label`, `label_text`, `fail_reason`, `error_text`/`error_kind`, `metadata` |
| Per-item, per-node traces | Each record's `exchanges` holds `{node, prompt, response}` for every agent call, so a failure can be traced to a node |
| Grader feedback | `VerifyResult.details`/`prediction` and the record's `fail_reason` |
| Minibatch evaluation | `Run.evaluate(candidate, items=[...])`, restricted to train ids |
| Leakage protection | `LoopConfig.parts_manifest` train/val/test split. `item_records` and `part_ids` refuse val/test. Val/test are evaluated in full and exposed only as scores (`Run.val_scores`) |
| Budget | `max_evaluations`, `max_rollouts` and `Run.budget_left()`. `max_tokens` and per-trial token counts |
| Prompt edits | `set_knob` and `text_patch` (#556) |
| Resumability | `strategy_state_dict` / `load_strategy_state` and `Run.save_strategy_state` (#556) |
| Model-call record/replay | SkillOpt's `TraceRecorder` with stable call keys, and `tests/test_replay_harness.py` |
| Packaging | `Package.strategies` with `required_ops`; discovered by `strategies.resolve` and `--strategy` |

What is missing for these two methods:

- SkillOpt optimizes one knob named `skill`. GEPA needs to cycle over every
  agent node's prompt slot.
- There is no Pareto frontier over per-item scores, and no merge across lineages.
- There is no few-shot demo edit.
- Strategies receive the loop's model through `configure(model=...)`, so a
  strategy's own calls cannot yet be routed to a different model from the workers.

`strategies.Reflective` is labelled "SRF `gepa`" but is not GEPA. Its merge
copies knob values, and it has no frontier, no trace-driven mutation and no
minibatch gate. It stays as the SRF port.

## Decisions

| Question | Decision | Why |
|---|---|---|
| Where the code lives | `refactory_packages/gepa` and `refactory_packages/mipro`, as `OuterLoop` subclasses using only the public `Run` API | Same seam as SkillOpt and the core/product boundary. Nothing is added to the SRF `STRATEGIES` table. |
| Replace `reflective`? | No | It is a documented SRF port with tests and results. Keeping it gives a direct ablation. |
| Shared code with SkillOpt | Factor out trajectory formatting, tolerant JSON parsing and the trace recorder where GEPA reuses them. Do this in coordination with the SkillOpt owner. | Avoids a second copy of the same helpers in a sibling package. |
| Optuna | Optional extra `refactory-dsh[optim]`, imported lazily by `mipro` | DSPy's MIPROv2 uses Optuna TPE, which reduces parity variables. Its dependency tree is too heavy for core. |
| Parity task | HotpotQA, distractor setting | Both papers report on it. It is multi-module, needs no retrieval server, and is far from the ceiling. |
| Models | Task: GLM on the `dsh` worker. Optimizer: Sonnet, fixed for every strategy. Confirmation: Opus. | The comparison grid is large. A varying proposer would confound it. |
| Fidelity evidence | Cite the reference implementation file:line for every default, as SkillOpt does, plus record/replay and DSPy parity | Two independent kinds of evidence that the port is faithful |

## Phase 0: Remaining groundwork

Tracked in #564.

1. **Optimizer route for strategy packages**: a strategy's own model calls
   resolve through the `optimizer` role, so `role_routes.optimizer` can differ
   from the workers. The task and optimizer routes are recorded in the report.
2. **Token accounting check**: per-candidate token counts reach the report for
   strategy-package runs, so budgets are comparable across strategies.
3. **Planted-task fake model**: a deterministic model whose output depends on
   features of the prompt. It drives two tasks:
   - *category-specialist*: item categories each need a different instruction;
   - *demo-sensitive*: only specific demonstrations help.
   It complements record/replay: replay pins a strategy's behaviour, while
   planted tasks check that it finds a known optimum.
4. **Feedback sufficiency**: confirm that `details`/`fail_reason` carry enough
   for reflection on chess and HotpotQA. Add a field only if they do not.

Exit: tests for 1–3 pass, and existing strategy and SkillOpt tests are unchanged.

## Phase 1: GEPA (`refactory_packages/gepa`)

Tracked in #565.

- Split train ids (`Run.part_ids`) into a feedback set and a Pareto set. The val
  part remains the acceptance gate.
- **Frontier**: candidates that are best on at least one Pareto item, using
  `item_records` scores. Dominated candidates are pruned.
- **Parent sampling**: weighted by the number of Pareto items a candidate wins.
- **Reflective mutation**: choose one agent node's prompt slot round-robin. Give
  the proposer that node's exchanges plus scores and feedback from a minibatch
  (`evaluate(items=...)`). Apply the rewrite as `text_patch` or `set_knob`.
- **Acceptance gate**: keep a child only if it beats its parent on the same
  minibatch, then score it on the full Pareto set.
- **System-aware merge**: take node-level prompt slots from two lineages that
  share an ancestor and changed different nodes.
- **State and artifacts**: the frontier and lineage go into the strategy state
  dict, so a run resumes. Reflection prompts and responses go through the trace
  recorder.

Tests: each invariant on the planted category task (the frontier keeps a
specialist per category, and greedy selection keeps fewer), plus a replay test
pinning one run.

## Phase 2: MIPROv2 (`refactory_packages/mipro`)

Tracked in #566.

- **Demo edit**: a `set_demos` op or a `demos` knob kind, rendered through
  `{{demos}}`, with its own render test. This is the one new core edit. It is
  declared in `required_ops`.
- **Bootstrap**: demos come only from `item_records` with `passed=True`, using the
  relevant node's `exchanges`. Build K demo sets per node.
- **Instruction proposals**: N per node, grounded on a dataset summary, a program
  summary, the bootstrapped demos and a sampled tip.
- **Search**: Optuna TPE over (instruction, demo set) per node on minibatches,
  with the incumbent re-scored on the full train set every M trials and gated on
  val at the end.
- Trial history and the Optuna study go into the strategy state, so a run resumes.

Tests: demos come only from passing train records; the proposal prompt contains
its grounding; a seed reproduces a run; the search finds the planted demos.

## Phase 3: Fidelity and parity

Tracked in #567.

- **Citations**: every default in `gepa` and `mipro` cites the reference
  implementation (`gepa` package or `dspy.GEPA`, and `dspy.MIPROv2`) by file and
  line, as SkillOpt does.
- **DSPy parity**: port a HotpotQA distractor program to an `nn.Mode` (select
  paragraphs, second-hop query, select again, answer; scored by EM/F1). A script
  under `experiments/` runs `dspy.GEPA`, `dspy.MIPROv2`, `gepa` and `mipro` with
  the same models, split and rollout budget, three seeds each.

Exit: our val gains are within the seed confidence interval of DSPy's, the
learning curves have a similar shape, and the budget splits the same way between
proposal and evaluation.

## Phase 4: Benchmarks and ablations

Tracked in #568.

- **Chess v2**: 150+ puzzles stratified by Lichess rating, held out through a
  `parts_manifest`, with python-chess scoring that accepts any line that forces
  mate. It needs a tracked package of its own.
- **A structure-sensitive benchmark**: a small SWE-bench-lite slice, or authoring
  modes scored on their regression tests.
- **Grid**: `gepa`, `mipro`, `skillopt`, `reflective`, `evolutionary`,
  `hill_climb` and `random`, at an equal rollout budget, with three seeds, on GLM.
  Then the top two on Opus.
- **Ablations**, with the expected direction:
  - GEPA with greedy selection instead of the frontier: no better;
  - GEPA with score-only feedback (exchanges withheld): clearly worse;
  - GEPA with merge disabled: no better;
  - MIPRO with instructions only, and with demos only: each no better than full;
  - every method against `random` at an equal budget: wins.

Comparing `reflective`, `skillopt` and `gepa` side by side separates what each
reflective design buys. An ablation that comes out the wrong way is treated as a
bug until shown otherwise.

## Sequencing

Phase 0 comes first and is small. Phases 1 and 2 can run in parallel, each
followed by its half of Phase 3. Phase 4 comes last. Coordinate with the SkillOpt
owner before Phase 1 starts, since the shared helpers are factored out then.
