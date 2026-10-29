# SPEC_APPROVED handoff: INT-001 activity verbosity

workspace:
  source_root: /Users/rawhad/3_Resources/external_libs/refactory-dsh
  worktree_root: /Users/rawhad/3_Resources/external_libs/refactory-dsh/.hai-worktrees/INT-001-activity-verbosity
  branch: hai/INT-001-activity-verbosity
  base_commit: 02a86790a6acc263d77d0f6588d62bcbfda6a055

session_id: session-a0e27736-9599-4b1e-9464-a8adf017b3d8

spec_path: docs/refactory-activity-verbosity-spec.md
spec_version: revision 4
spec_sha256: 5f4475e289eccbf092d0618dbad2cd0673fd167f4806d44d6f1d299abec346a0
hai_state_path: .hai/state.yaml
hai_state_sha256: 6dbab33ce31ab94c066595a59e3429f8e5e82368c41d8d31f11d844502cc604d
hai_captured_through: 6e8a42f8-e885-4cc9-ac51-08b8e449e0a0
hai_version_or_commit: working-tree state (untracked in base commit; validated by .hai-validate.mjs)

## trace_coverage

R-1 (bounded/paged journal read) -> INT-001, DEC-001
R-2 (three levels, standard default, full history reachable at every level) -> INT-001, DEC-001, DEC-002
R-3 (no cumulative usage.calls in new journal records) -> INT-001
R-4 (cursor continuity across attach/watch/refresh) -> INT-001
R-5 (truncation reported distinctly from absent JSON) -> INT-001
R-6 (two-way compatibility: new panel/old console, old panel/new console, index never a dependency) -> INT-001

Reviewer-verified: every R-* trace is supporting (not decorative); DEC-001 satisfied at every
level including compact (R-2.4 keeps the paging control); DEC-002 exact (three levels, standard
default per R-2.1 / §4.5 / T-21). No normative obligation lacks a trace. No behavior change is
required for answers the current code already gives correctly (`total` and `last_event_at`
agree with today's code 15/15 in the reviewer's matrix).

## automated_tests

Mapping in spec §10: T-1..T-26 (+T-3b, T-18b, T-23a/T-23b, T-24, T-25, T-26), each citing
R/UC/F IDs. Gate commands in spec §12 (all use <worktree_root>/.venv).

## manual_use_cases

Defined in spec §9 (UC-1..UC-12) and checked in §11 (1..11). Reviewer-owned integration
runs them in M4.

## done_gates

Spec §12 gates 1-8. Gate 7 (`test_bounded_activity_page`) is RED until built; everything
else is green at baseline (62+53 python, 1/1 observer, 266 vitest, typecheck exit 0).

## unresolved_soft_assumptions

ASM-001..ASM-005 remain soft in spec §7.1 with explicit "what changes if false"; none is a
requirement.

## reviewer_pass_evidence

Round 4 independent SPEC_REVIEWER: verdict PASS on sha256 5f4475e289eccbf092d0618dbad2cd0673fd167f4806d44d6f1d299abec346a0.
Both round-3 SPEC_BLOCKERs verified fixed by execution (H-1: 15/15 gap matrix agrees with
today; H-2: executable model of identity checks 1-5 answers today's values for control,
append, archival, fresh-short, same-size rewrite, same-inode-larger). Prior rounds:
r1 FAIL (F-1..F-4 blockers), r2 FAIL (G-1..G-3), r3 FAIL (H-1, H-2).

Residual WARN/NIT carried forward and disclosed (not blocking):
- J-1 NIT: revision log says the round-3 walk diverged 6/15; reviewer measured 7/15.
- J-2 WARN: §4.6 calls a partly-written line "truncated at the last complete line"; for
  worker_observer.mjs's appendFileSync a mid-record split survives as a journal_gap. No
  answer regression versus today; builder should state the served/gap wording.
- J-3 WARN: dual numbering `[R-6.5.2]` marker vs the real `R-6.6` clause.
- J-4 NIT: SD-3's "100 calls in 5.6 s" is optimistic (reviewer measured 10.65 s); shape holds.
- J-5 WARN: T-23a names run-stream.client.spec.ts, which only covers contribution codecs;
  T-23b's RunActivity "Outputs and validation" assertion is a NEW test in
  body.client.spec.tsx (RefactoryBody.tsx:305 mounts RunActivity).
- J-6 NIT: gate 7's prose still mentions the old `G-7` label.

## exact_command_list

    cd <worktree_root> && node .hai-validate.mjs .hai/state.yaml
    cd <worktree_root>/refactory && ../.venv/bin/python -m pytest tests/test_live_activity.py tests/test_gui_loop.py tests/test_observe.py tests/test_console_projects.py -q
    cd <worktree_root>/refactory && node --test tests/worker_observer_test.mjs
    cd <worktree_root> && pnpm vitest run packages/refactory/refactory-console/tests packages/refactory/tool-refactory/tests packages/client/ui-refactory/tests packages/bundle/refactory-web/tests
    cd <worktree_root> && pnpm typecheck
    cd <worktree_root> && .venv/bin/python -m refactory_dsh.tools.gen_discovery --check
    cd <worktree_root>/refactory && ../.venv/bin/python -m pytest tests/test_live_activity.py -q -k test_bounded_activity_page
