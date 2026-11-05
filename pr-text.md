# Fork/join inside a data node's per-item pass (F21)

## Summary

A data node's per-item pass can contain a fork and join on both the Python walker and the TS spine. This PR adds comprehensive tests proving it on both runtimes, a validation guard (V13) that refuses the unsupported inverse shape (data-inside-fork) at validation/lowering time, and cross-runtime equivalence assertions. Per-item deadlines, the shared step cap, stop/pause, and resume mid-pass are all tested with a fork inside the pass.

## Changes

- `refactory/refactory_dsh/nn/validate.py` — V13 rule: reject data nodes inside fork branches.
- `refactory/refactory_dsh/lowering.py` — SWEEP-1 data-inside-fork check at lowering time.
- `packages/workflow/workflow-spine/src/spine.ts` — `validateGraph` data-inside-fork check.
- `packages/workflow/workflow-spine/tests/graph-shapes.ts` — `forkInsideData` shape, `dataInsideForkGraph` export; `dataInsideBranch` moved out of invariant array.
- `packages/workflow/workflow-spine/tests/graph-set.spec.ts` — Data-inside-fork refusal test.
- `packages/workflow/workflow-spine/tests/spine.spec.ts` — Manual fork-inside-data-pass state-machine test.
- `refactory/tests/loop_ops.py` — `conditional_fail`, `halt_branch_on_first_item`, `write_item_result`, `halt_branch_on_first_item_spine`, `halt_gate_on_first_item` ops.
- `refactory/tests/test_data_parallel.py` — All new walker and two-runtime tests below.

## Tests

| Test | Runtime | What it proves |
|---|---|---|
| `test_fork_inside_data_op_branches[1,3]` | walker | Fork→[op,op]→join→exit in data body completes |
| `test_fork_inside_data_agent_branch[1,3]` | walker | Agent node in fork branch with tool_policy="none", d_exit→Task.verify scoring chain |
| `test_fork_inside_data_failed_branch[1,3]` | walker | Failed branch: item scores 0, error_kind=="RuntimeError" |
| `test_fork_inside_data_deadline[1,3]` | walker | Per-item deadline with fork: verify timeout fires, fork walk does not timeout |
| `test_fork_inside_data_step_cap[1,3]` | walker | Shared step cap covers fork branches inside data pass |
| `test_fork_inside_data_stop[1,3]` | walker | HaltStop inside fork branch stops the run |
| `test_fork_inside_data_pause` | walker | HaltStop from fn_gate after fork/join pauses the run |
| `test_fork_inside_data_resume` | walker | Stop mid-fork on item 0, resume, same item_scores as uninterrupted |
| `test_fork_inside_data_two_runtimes[1,3]` | both | Same item_records, item_scores, merged facts on walker and spine |
| `test_fork_inside_data_resume_spine` | spine | Stop mid-fork on item 0, resume from checkpoint, same item_scores |
| `test_data_inside_fork_validation` | walker | V13 issue + LoweringError for data-inside-fork |
| `graph-set.spec.ts` forkInsideData | spine | Shape invariants (completion ordering, join-after-branches, redo) |
| `graph-set.spec.ts` data-inside-fork refusal | spine | validateGraph reports data-inside-fork issue |
| `spine.spec.ts` fork inside a data pass | spine | Manual startPass/complete/endPass drive |

## Verification

```bash
pnpm test && .venv/bin/python -m pytest -q -n 4 refactory/tests
```

## Follow-ups

- Per-item deadline covering the fork walk phase (currently the deadline is disarmed for the walk, covering setup/verify only; the fork runs under the step cap and budget).
- Finer-grained walker resume mid-fork (currently replays the whole item).
- Cross-data-body reloop validation in `crossContainerReloopIssues`.
- Spine single-pass field → stack refactor for nested data-inside-fork-inside-data.
- The spine does not enforce step caps — it is a stateless state machine; step caps are a walker concept. The spine graph-set invariants test the fork-inside-data shape through all completion orderings.

## Behaviour changes

**Data-inside-fork graphs are now refused at validation/lowering time** on both runtimes:
- Python `validate()`: V13 issue
- Python `to_workflow()`: `LoweringError`
- TS `validateGraph()`: issue with node and message

Previously, such graphs passed validation but crashed at walker runtime (`_branches()` at fork.py:80). The spine could walk them but will now flag them for parity.

**Who is affected**: Only a graph author who places a data node as a fork branch target. No production graph uses this shape.

**How to turn it off**: Restructure the graph so the data node is outside the fork, or place the fork inside the data body (the supported shape).
