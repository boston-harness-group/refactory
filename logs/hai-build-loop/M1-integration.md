# Integration evidence: M1 console contract

## Environment
- Workspace: source_root `/Users/rawhad/3_Resources/external_libs/refactory-dsh`,
  worktree_root `<source>/.hai-worktrees/INT-001-activity-verbosity`,
  branch `hai/INT-001-activity-verbosity`, base_commit `02a86790a6acc263d77d0f6588d62bcbfda6a055`.
- Tested HEAD: `cfd2d2495c` (spec rev-4 commit) + uncommitted M1 working-tree changes,
  `git diff --stat` = 12 files, +1824/−253, plus untracked `logs/`.
- Spec: `docs/refactory-activity-verbosity-spec.md`, revision 6,
  sha256 `ad48b75350a20d08eaa1e131ebed2b4e880c5835d656a4ec2553ce55774887ac`.
- State: `.hai/state.yaml` sha256 `6dbab33ce31ab94c066595a59e3429f8e5e82368c41d8d31f11d844502cc604d`,
  `captured_through 6e8a42f8-e885-4cc9-ac51-08b8e449e0a0`, validator `valid: …`, exit 0.
- Runtime: `<worktree_root>/.venv` (CPython 3.12.10, refactory_dsh installed editable),
  Node 24.1.0, pnpm workspace `node_modules` present.
- Offline; no live/external calls were made (no authorization sought or needed).

## Round history (provenance)
- **Build round 1** (builder `509f3227`): implemented M1; all five gates green.
- **Build review round 1** (reviewer `4c8e7b55`): **FAIL** — M1-1 `total` under-reported on a
  validity-surviving short index; M1-2 warm per-page cost was O(N) (R-1.7a/SD-3 undelivered).
- **Spec loop** (`b72c0b90`): revision 6 approved — legalises the third derived file
  (`events.index.overview.json`); rejected a false HUMAN_ESCALATION by exec'ing HEAD.
- **Build round 2** (builder `2aaf25b7`): fixed both blockers, found a second latent bug
  (`size_at_write` recorded file end, not counted-prefix end), mutation-tested T-3.
- **Build review round 2**: the first re-reviewer (`99c7b0b5`) and its replacement
  (`23793e5c`) both **failed without a verdict** (platform-level subagent crash, no closing
  message). The tree was verified intact between attempts (spec hash, 12-file diffstat, venv
  import). Per the worktree contract the orchestrator then executed the checks itself with
  its own fixtures; this is recorded as such, not as an independent reviewer PASS.

## Commands and results (orchestrator-executed, this round)
| Command | Exit/result | Evidence |
|---|---|---|
| `node .hai-validate.mjs .hai/state.yaml` | valid, exit 0 | 1 intent, 2 decisions, 10 research |
| `pytest tests/test_live_activity.py tests/test_gui_loop.py tests/test_observe.py tests/test_console_projects.py -q` | **129 passed** in 22.96s | exit 0 |
| `node --test tests/worker_observer_test.mjs` | tests 1 / pass 1 / fail 0 | 61 ms |
| `pytest tests/test_live_activity.py -q -k test_bounded_activity_page` | **1 passed**, 51 deselected | gate 7 |
| `python -m refactory_dsh.tools.gen_discovery --check` | current, exit 0 | gate 6 |
| `python /tmp/m1_fix1.py` (truncation sweep) | **PASS** (51 cuts, 0 mismatches) | see below |
| `python /tmp/m1_fix2.py` (warm cost) | **PASS** (1.01× across 40× N) | see below |

## FIX 1 — index truncation (M1-1)
Journals of 6 / 40 / 300 records; index warmed, then truncated at 17 byte positions each
(0, 1, 17, 40, thirds, tenths, size−1, size). At every cut, both the rebuilding call and the
following warm call were compared against the full scan (`live.events`) on `total`,
`last_event_at` and served records.
- 6 records (index 1,053 B), 40 records (1,053 B), 300 records (5,265 B): **all cuts ok**.
- Result: `RESULT: PASS`, 51 cuts checked, 0 mismatches.
- Pre-fix this class produced `total=20` on a 40-record journal; no longer reproducible.

## FIX 2 — warm cost (M1-2), medians of 9, warm index
| N | page(after=200) | page(after=N/2) | summary | total |
|---|---|---|---|---|
| 500 | 0.627 ms | 0.625 ms | 0.108 ms | 500 |
| 5,000 | 0.824 ms | 0.611 ms | 0.106 ms | 5,000 |
| 20,000 | 0.634 ms | 0.634 ms | 0.109 ms | 20,000 |

Cross-N ratios: page(200) **1.01×**, page(N/2) **1.01×**, summary **1.01×** across 40× N.
Pre-fix the same measurements were 3.94× / 4.17× and a N²-shaped full pass (4.6→3,484 ms).
R-1.7a/R-1.7b are delivered.

## Test-power probe (finding M1-W1, WARN, not a blocker)
Runtime-patched `live.page` to additionally parse the whole index (the pre-fix shape) and
applied T-3's own shipped threshold `page(20k) < max(page(500)*2, 5.0)`:

| N | index size | fixed | with O(N) read | slowdown |
|---|---|---|---|---|
| 500 | 8.2 KiB | 0.60 ms | 0.62 ms | 1.03× |
| 20,000 | 321.9 KiB | 0.60 ms | 0.74 ms | 1.25× |
| 270,000 | 4,338.5 KiB | 0.61 ms | 2.48 ms | 4.05× |

The injected O(N) read **still passes** T-3, because the 64:1 striding makes a full index read
cheap in absolute terms and the `max(…, 5.0)` floor dominates at N=20,000. The test is
therefore under-powered, not vacuous: at the issue's own 269,900-record scale the regression is
a visible 4.05×, but T-3 as written would not fail on it.
Disposition: **WARN carried to M4** — strengthen T-3 to assert the cross-N ratio directly
(e.g. `page(20k) < page(500) * 2` without the absolute floor, or add a 270k size and compare
against a fixed ceiling derived from N=500 measured in the same run) so a future O(N) index
read cannot regress silently. Does not block M1: R-1.7 is actually delivered.

## Residual UX / manual checks
- Console-side use cases (UC-1, UC-4, UC-5, UC-6, UC-11, UC-12) are covered by gate 2's tests;
  the interactive terminal feel of `status --follow` is deferred to M4's manual checklist (§11).

## Cleanup
- Throwaway fixtures lived in `tempfile.TemporaryDirectory()` and `/tmp`; no processes started,
  no servers, no ports held. `/tmp/base-check` is the reviewer-sanctioned base-commit worktree
  used for regression comparison; it is outside the review tree and left for reuse.
