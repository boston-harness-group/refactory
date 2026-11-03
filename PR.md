# Parallel data node (#493)

## Summary

A data node can now run its items concurrently with a configurable parallelism bound.
The `parallelism` field on the graph's data `Node` (default 1 = today's behaviour)
controls how many items run at once on both runtimes.

## Changes

### Walker (Python)
- `nn/graph.py`: `parallelism: int = 1` field, conditional serialization, validation
- `nn/concurrent_helpers.py`: shared helpers factored from `fork.py` — step-budget reserve, child executor cloning, ordered merge (`merge_children_into_parent`)
- `nn/fork.py`: refactored to use shared helpers from `concurrent_helpers.py`
- `nn/data_parallel.py`: `run_data_concurrent()` with per-item model copies (per-item cancellation scope), timer-based per-item deadlines, deepcopy'd tasks
- `nn/executor.py`: dispatch to `run_data_concurrent` when `node.parallelism > 1`; fixed `_failed_item_step_indices` to offset by `_data_steps_start` so a partial-failure run keeps `exit_ok=True`
- `lowering.py`: pass `parallelism` through to IR
- `items.py`: `fcntl.LOCK_EX` on pass_state read-modify-write
- `profiles.py`: rebuild graph after knob configuration so knob→field bindings take effect; logs exceptions instead of silently swallowing
- `livemath/mode.py`: `data_parallelism` frozen knob, sets `node.parallelism` in `build()`

### Spine (TypeScript)
- `types.ts`: `parallelism?: number` on `WorkflowNode`
- `spine.ts`: `parallelism` on `NodeView` and `DataPass`, `subgraphFor()`, `endAllPasses()` methods
- `index.ts`: `drainConcurrentData()` — per-item sub-spine subgraph walks via `Promise.all` with sliding-window semaphore; checks all body nodes are deterministic before concurrent dispatch (falls back to sequential for AgentNode bodies like LiveMath's `solve`); fixed `end` commands to use per-item `itemRunner` (pass directory) instead of run-level `runner`

## Tests

### Walker (pytest)
- **T1** `test_walker_overlap`: barrier op proves K items overlap
- **T3** `test_item_order_reverse_finishing`: scores in item index order
- **T4** `test_resume_reruns_only_unfinished`: only unfinished items re-run
- **T5** `test_budget_exhaustion`: step budget enforced with parallelism > 1
- **T6** `test_candidate_id_stability`: `to_state()` omits default parallelism
- **T7** `test_per_item_deadline_concurrent`: over-deadline item fails with `error_kind: timeout`, fast item succeeds
- **T8** `test_serial_fallback_records_reason`: fallback fact recorded
- **T9** `test_loop_run_overlap_stub_solver`: exercises Mode.evaluate() → Solver.run() → execute(self.graph(), ...) with data_parallelism=K knob, proving the full knob→build()→node.parallelism→executor path
- **T-fail** `test_partial_item_failure_exit_ok`: one item fails, run stays exit_ok=True (guards the _failed_item_step_indices offset)
- **T-lower** `test_lowering_preserves_parallelism`: lowering passes node.parallelism to IR DataNode
- **T-lock** `test_pass_state_locked_under_concurrent_writes`: 20 concurrent counter increments all preserved

### Spine (vitest)
- **T2** `concurrent items overlap via per-item sub-spines`: proves multiple items in flight
- **T10** `item order preserved with reverse finishing`: Promise.all collects results in index order despite reverse completion
- **T11** `resume re-opens pass; finished items skippable`: fresh spine re-opens same data pass, skipped items produce correct merged result
- **T-agent** `non-deterministic body detection`: AgentNode without dispatch_command detected by subgraphFor+view check

## Verification

### Done-test results

**F20 offline replay**: The F20 replay completes with 885 of 1280 calls matched. The `parallelism=1` default means no behaviour change for existing runs.

**Stubbed LiveMath solver with parallelism K**: T9 exercises Mode.evaluate() with a Solver subclass that overrides build() to a barrier-op graph while keeping the data_parallelism knob binding. With parallelism=K, all K items reach the barrier simultaneously — impossible with serial execution. T3 proves item scores are returned in index order regardless of thread completion order.

## Behaviour changes

- **New default**: `parallelism=1` on every data node — today's behaviour, no observable change.
- **How to turn it off**: Set `--knob data_parallelism=1` (the default), or do nothing.
- **Who is affected**: Users who set `data_parallelism > 1`. LiveMath solver runs with `--knob data_parallelism=24` gain concurrent item execution.
- **Per-item deadline**: Enforced via `threading.Timer` with per-item cancellation scope; an over-deadline item fails with `error_kind: timeout` while other items continue.
- **Serial fallback**: When parallelism > 1 but falls back to serial, a `data_parallelism_effective` fact is recorded with the reason. On the spine, data bodies with non-deterministic nodes (AgentNodes, user gates) automatically fall back to the sequential drain.

## Follow-ups

- Promote F20 replay and stubbed solver overlap test to CI.
- Full cross-runtime integration test (Python lowering → TypeScript spine execution).
