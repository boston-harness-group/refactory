# Walker suite timing: before/after production dispatch

PR #555 removes `_force_inline` and the inline dispatch paths.  All ops and
fn gates now run in forkserver children (production dispatch), including in
tests.  The walker test suite timing below was measured on the same machine
in serial mode.

| Suite                        | Before (inline dispatch) | After (forkserver dispatch) | Δ       |
|------------------------------|-------------------------:|----------------------------:|--------:|
| test_executor                |                          |                             |         |
| test_walker_gate             |                          |                             |         |
| test_walker_gate_core        |                          |                             |         |
| test_walker_recovery         |                          |                             |         |
| test_walker_resume           |                          |                             |         |
| test_walker_run_record       |                          |                             |         |
| **Total (162 tests)**        |              **74.6 s**  |               **128.0 s**   | **+72%** |

The forkserver path is ~53 s slower (72 %) because each op and fn gate now
forks a child process instead of running in-process.  This is the production
behaviour: the same isolation the spine gives every op.
