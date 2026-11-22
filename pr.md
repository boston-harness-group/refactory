# Per-step chain excusal for chained check commands

Supersedes refactory#721.

## Summary

When a check command chains steps with `&&` (e.g. `tok.sh && pytest X && grep -q Y f`), an early-failing step stops the chain and leaves the base incomplete. This change adds a per-step marker re-run that runs all chain steps in one shell unconditionally, records each step's exit code via stdout markers, and applies a same-outcome excusal rule: a test failure in step k is inherited only when every earlier step ended the same way on both sides and step k's runner summary is accounted on both sides. Non-test step failures are never excused.

## Changes

### `refactory/refactory_dsh/base_check.py`

- **`_chain_steps`**: Splits commands on top-level `&&`, `||`, `;` only, keeping pipelines and subshells whole. Records the reason (heredoc, keyword, backtick) when it refuses to split, via an optional `reason` parameter. Commands with heredocs, `if`/`case`/`for`/`while`, or backtick substitutions spanning operators are not split.
- **`_build_marker_script`**: Constructs a shell script running each step unconditionally with `exec 2>&1` to merge stderr into stdout and unique marker lines capturing `$?`.
- **`_parse_step_results`**: Parses marker lines from stdout into per-step dicts with output, exit code, operator.
- **`_step_would_run`**: Operator gating for `||` only — `&&` and `;` steps are never dropped.
- **`_run_chain_markers`**: Shared helper that builds, runs, and parses the marker script through `sandbox.run_command`.
- **`_runner_from_output`**: Detects pytest/vitest from output patterns when the step text does not name a runner.
- **`_per_step_comparison`**: Core function that re-runs the attempt with markers, compares per-step exit codes and summaries against the base's cached per-step results, and refines the inherited/new classification.
- **`_try_base`**: Conditional second marker run on the base when the check failed and is incomplete. Enriches `failures`, recomputes `_base_complete`, and caches per-step results in a `steps` field.
- **`filter_check`**: Dispatches to `_per_step_comparison` for chain commands when `all_excused` is True.
- **`excusal_blocker`**: Added optional `step_results` keyword for per-step accounting; existing callers unchanged.

### `refactory/tests/test_chain_excusal.py` (new)

Unit tests, scenario tests, cache integration tests, and a real-shell fidelity test.

## Tests

### Scenario tests (brief's done-when cases)
- `test_chain_tok_sh_pytest_grep`: S inherited, N and grep stay the attempt's.
- `test_chain_build_vitest_different_outcome`: T stays the attempt's (different step 0 outcome).
- `test_chain_cd_pytest_state_carries`: Failure inherited with `pkg/` prefix test ids.
- `test_chain_pytest_ruff_pytest_attempt_reruns`: S inherited, N (integ) is new.
- `test_heredoc_not_split_base_incomplete`: `_chain_steps` returns None; reason recorded.
- `test_passing_base_one_run`: No marker re-run; `_run_chain_markers` not called.
- `test_unavailable_timed_out_base_inherits_nothing`: Environment failure blocks inheritance.

### Unit tests
- `test_chain_steps_*`: Splitting, operators, heredoc, keywords, backticks, quotes, subshells, reason recording.
- `test_runner_from_output_*`: Output-based runner detection for pytest and vitest.
- `test_marker_script_construction_and_parsing`: Round-trip of build + parse.
- `test_step_would_run_operators`: `||` gating only, `&&`/`;` always run.
- `test_excusal_blocker_backwards_compat`: Existing callers unchanged.
- `test_or_chain_operator_gating`: `||` steps excluded when previous exited 0.

### Integration tests
- `test_single_shell_fidelity_real`: Real `sandbox.run_command` with `cd` and `export` state carrying.
- `test_no_files_in_run_worktree`: Attempt marker run uses check copy, not run worktree.
- `test_cache_round_trip`, `test_old_cache_without_steps`, `test_cache_hit_does_not_rerun_base`.
- `test_unaccounted_base_step_blocks_inheritance`: `_accounted` gate prevents inheritance.
- `test_and_gating_would_break_inheritance`: `&&` gating is NOT applied; S still inherited.

## Verification

### Mutation checks (each reverts the code and watches a scenario test fail)
- **Dropping the same-outcome rule**: `test_chain_build_vitest_different_outcome` fails (T incorrectly inherited when build exited differently).
- **Running steps in separate shells**: `test_single_shell_fidelity_real` fails (cd/export state does not carry).
- **Dropping the attempt re-run**: `test_chain_pytest_ruff_pytest_attempt_reruns` fails (N never discovered).
- **Marking base_complete without `_accounted`**: `test_unaccounted_base_step_blocks_inheritance` fails (S incorrectly inherited from unaccounted base).
- **Re-adding `&&` gating**: `test_and_gating_would_break_inheritance` fails (pytest failures excluded after failed tok.sh).
- **Re-running the base on a cache hit**: `test_cache_hit_does_not_rerun_base` fails (cached steps present but base re-run would occur).

### Runtime coverage

Both runtimes reach all four ops through the same code path:

| Runtime | Entry point | How it reaches the change |
|---------|-------------|---------------------------|
| Walker in-process | `prepare_check`, `check_beside_reviews`, `gate_check`, `fast_ops.focused_check` | → `filter_check` → `run_base_check` → `_try_base` |
| Spine via `python -m refactory_dsh.run` | `prepare_check`, `check_beside_reviews`, `gate_check`, `fast_ops.focused_check` | → `filter_check` → `run_base_check` → `_try_base` |

## Follow-ups

- Remove the dead `_MARKER_RE_TEMPLATE` constant and the unused `command` parameter in `_run_chain_markers`.
- Consider per-step timeout reporting when individual steps are slow.
- The `steps` cache field stores per-step output (truncated to 5000 chars); monitor cache size.
