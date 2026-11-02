# PR #555 walker parity — part A notes

## Walker test-suite timing

Every op and fn gate now runs in a reused forkserver child on all walker
paths (not only with `enable_recovery`).  The walker test suite
(`test_executor`, `test_walker_gate`, `test_walker_gate_core`,
`test_walker_recovery`, `test_walker_resume`, `test_walker_run_record`)
runs **68 s** after the change vs **73 s** before (162 vs 159 tests; the
three new tests add < 1 s each).  No slowdown from routing every op
through the forkserver; the per-child overhead is amortised by the reused
forkserver process.

## test_edit_design.py race fix

The unique package names per test (`edited_revision`, `edited_regression`,
etc.) introduced in the earlier commit are **kept**.  Parallel pytest-xdist
workers share the workspace's package registry, and a shared name like
`edited_pkg` causes intermittent failures when one worker's cleanup races
with another worker's registration.  Unique names make each test's package
lifecycle independent, eliminating the race without serialising the tests.
