# SkillOpt parity: output cap, budget binding, epoch failure, and winner selection

Closes boston-harness-group/refactory#493 rows G31, G32, G36, G37.

## Summary

Four parity gaps between SkillOpt arm B (refactory) and arm A (microsoft/SkillOpt) found in the 2026-11-02/03 shakedowns cause incorrect run outcomes: uncapped solver output tokens (G31), strategy model calls that bypass the run budget (G36), all-failed runs that report success (G32), and the run's winner diverging from the strategy's val-selected best (G37). The cheap parts of G33 (report header showing wrong optimizer name and epoch count) are fixed alongside G37.

## Changes

### G37 — Winner selection

A custom strategy can now declare its own best candidate via `OuterLoop.best_candidate(population)`. `SkillOptLoop` records the accepted candidate's id where `_best_skill` is set (loop.py:236, :496), persists it in `strategy_state_dict` / `load_strategy_state` (resume), and returns the candidate by id lookup from the population. If the strategy's best id is not in the population, `ValueError` is raised and `report()` sets `exit_ok=False` with a clear message — never falls back silently to the leaderboard. `report.json`, `report.md`, and the holdout audit all use the strategy's pick when provided. Default strategies return `None`, preserving today's leaderboard-first behaviour.

### G36 — Budget binding

`Loop.bind_budget()` now calls `strategy.bind_model(proposer)` on the custom strategy, giving it the budget-bound, optimizer-role-routed client. `OuterLoop.bind_model()` is a new overridable method (default: `self.model = model`). All seven SkillOpt call categories (analyst, merge_failure, merge_success, merge_final, rank, slow_update, meta_skill) go through `trace.ask(self.model, ...)` which reads `self.model` at call time, so they inherit the budget binding without any SkillOptLoop override.

### G32 — Epoch failure

When every optimizer call in an epoch fails, `SkillOptLoop._record_epoch_calls()` now calls `run.stop()` and returns `True`, causing the strategy to save state and return early. In the shared engine, `report()` checks `proposer.get("all_failed")` and forces `exit_ok=False` with a clear error naming the last failure. There is no knob — a run whose optimizer never answered is never a success.

### G31 — Output cap (livemath mode knob)

The livemath `Solver` mode adds a `max_completion_tokens` class variable (default 16384, 0 = no cap). `Solver.run()` injects it into the solve node's route via `_node_routes`. The route field propagates through `DshModelClient.route()`, `for_node()`, `context()`, `validate_route()`, and `ROUTE_FIELDS`. Every other mode and every design-mode worker keeps today's behaviour (no cap).

**G31 evidence (DSH route/request):** DSH's `agent/request` hook returns `ModelSelection` (provider, model, reasoningEffort). `maxOutputTokens`/`maxTokens` is NOT part of `ModelSelection` — it lives in `AgentOptions`/`LlmCallConfig` (dsh-subagent/lib/types/child-agent.js:243, :495), which the hook cannot set. The Python route carries the value so a future DSH release can read it, but the JS observer does not inject `maxOutputTokens` today. G31 is wired in Python only; the JS observer is left unchanged.

### G33 — Report header (cheap fix alongside G37)

When a custom strategy is set, `report()` uses the strategy's `name` and `_epoch` count instead of the infrastructure optimizer's.

## Tests

- **G37**: `test_best_candidate_returns_val_gated_winner_by_id`, `test_strategy_best_candidate_overrides_report_winner`, `test_holdout_audit_uses_strategy_winner`, `test_best_candidate_id_not_in_population_errors`, `test_best_candidate_id_persisted_across_resume`, `test_default_loop_winner_is_leaderboard_first`
- **G36**: `test_budget_exhausted_stops_optimizer_calls`, `test_budget_bound_model_survives_resume`, `test_bind_budget_without_custom_strategy_does_not_error`
- **G32**: Updated `test_failed_optimizer_calls_fall_back_like_skillopt` (now asserts early stop and `exit_ok=False`), `test_partial_failure_does_not_early_stop`, updated `test_a_run_with_an_unreachable_proposer_reports_failure` (asserts `exit_ok=False`)
- **G31**: `test_max_completion_tokens_appears_in_route`, `test_max_completion_tokens_omitted_when_unset`, `test_for_node_propagates_max_completion_tokens`, `test_validate_route_accepts_max_completion_tokens_string`, `test_validate_route_rejects_max_completion_tokens_integer`, `test_livemath_mode_knob_sets_solver_route`
- **G33**: `test_report_shows_strategy_name_and_epoch_count`

## Verification

F20 offline replay: `replay_runner.py` over `~/skillopt/runs/skillopt-train-20261023-e10-s42` completed with **885 of 1280 calls matched** (`LIVEMATH_DATA_PATH=$HOME/skillopt/data/livemath_raw`).

## Behaviour changes

1. **All-failed runs exit non-zero (G32).** When every optimizer call in a run (default loop) or every optimizer call in an epoch (SkillOpt) fails, the run now exits non-zero with a clear error naming the last failure. Previously it would complete all epochs and exit 0. There is no knob to disable this; a run whose optimizer never answered is never a success.

2. **LiveMath solver output cap defaults to 16384 (G31).** The livemath mode knob `max_completion_tokens` (default 16384, 0 = no cap) is applied to the solver node's route only. Every other mode and every design-mode worker keeps today's behaviour (no cap). Affects livemath/solver runs only. Set the mode knob `max_completion_tokens: 0` to disable.

3. **Strategy-declared winners override leaderboard ranking (G37).** When a custom strategy's `best_candidate(population)` returns a candidate, it becomes the run's `best` in `report.json`, `report.md`, and the holdout audit. If the strategy's best candidate id is not in the population, the run reports `exit_ok=False` with a clear error — never falls back silently to the leaderboard. The default strategy returns `None` — no change for default-loop runs.

4. **Strategy model calls are budget-tracked (G36).** After `bind_budget()`, the custom strategy's `self.model` is the budget-bound, optimizer-role-routed client. `BudgetExhausted` can now fire from strategy model calls. Affects any run with `--max-tokens` or `--max-elapsed-s` and a custom strategy.

## Runtime coverage

The outer loop (`loop.py`, `outerloop.py`, `skillopt/`) is Python only; the spine (TS graph runtime) is not touched. `worker_observer.mjs` is DSH worker glue, tested by `pnpm test`. No spine change means no TS runtime testing is needed beyond the existing `pnpm test` suite.

## Follow-ups

- Wire `maxOutputTokens` in `worker_observer.mjs` when DSH adds the field to `ModelSelection` in the `agent/request` hook.
- G33 holdout corpus count (177 vs train ids) — deferred as not cheap.
