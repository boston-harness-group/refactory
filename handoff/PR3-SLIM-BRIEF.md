# Slim structure PR3 — brief (user-approved 2026-11-16)

Hand-made PR, built and tested on the laptop. Source: `handoff/easy-structure-plan.md` §9, PR3 row — only part of it.

## In
1. **Any outer loop.**
   - Operator sampling for strategies with no model proposer (`random_edit`, BestOfN).
   - SkillOpt declares itself text-only.
   - A `rebase` seam stub for the GEPA-style strategy.
   - `test_strategy_operators.py`: every registered strategy takes every operator, or declares that it opts out.
2. **`configure` as a bindable knob for two settings only:** `max_iterations` (loop limit) and branch `optional`.
   - Extend `BINDABLE_FIELDS` and make lowering resolve each one.
   - Test: walker value = lowered value = spine behaviour.
3. **Strict coverage test** (`nn/coverage.py`). Nothing stays "planned"; each item is covered, or excluded on purpose with a reason.

## Deferred (record as follow-ups in `coverage.py` DECISIONS and the PR body)
- `timeout_s` / `retry` / `on_exhausted` knobs. They mostly trade cost and reliability, invite fail-fast gaming (against "a time-out counts as a failure"), and touch `node_retry` in both runtimes.
- Workspace kind, model role.
- The per-sub-item spike.

## Size and rules
- About 500–700 lines including tests.
- Docs regenerated and checked, and related suites run, before any push.
- Independent review, then `safe-merge.sh`. Title ends "(hand-made)"; short body with a Behaviour changes section.
- Not blocking: the clean-LiveMath structure arms were launched 2026-11-16 00:34 UTC from main `f8857b6f59`.
