# Integration evidence: M2 boundary and service

## Environment
- Workspace: source_root `/Users/rawhad/3_Resources/external_libs/refactory-dsh`,
  worktree_root `<source>/.hai-worktrees/INT-001-activity-verbosity`,
  branch `hai/INT-001-activity-verbosity`, base_commit `02a86790a6acc263d77d0f6588d62bcbfda6a055`.
- Tested tree: HEAD `7da92c75c7` (M1 accepted) + uncommitted M2 changes,
  `git diff --stat` = **7 files, +483/−23**.
- Spec: revision 6, sha256 `ad48b75350a20d08eaa1e131ebed2b4e880c5835d656a4ec2553ce55774887ac`
  (unchanged before and after M2).
- State: `.hai/state.yaml` sha256 `6dbab33c…604d`, validator `valid: …`, exit 0; `.hai/` unmodified.
- Runtime: worktree `.venv` (refactory_dsh editable), Node 24.1.0, pnpm `node_modules`.
- Offline. No live/external model calls. M2's "live" suite spawns the real local console.

## Round history
- **Builder round 1** (`fd03a0ec`): implemented M2; gates green; no blockers.
- **Verification**: executed by the orchestrator (independent re-run + independent mutation
  testing). Reviewer subagents had crashed on the preceding milestone; this round the
  orchestrator verified directly and records that provenance rather than implying a
  reviewer PASS it does not have.

## Commands and results (orchestrator re-run, not builder-reported)
| Command | Result | Evidence |
|---|---|---|
| `pnpm vitest run packages/refactory/refactory-console/tests packages/refactory/tool-refactory/tests packages/client/ui-refactory/tests packages/bundle/refactory-web/tests` | **19 files / 339 tests passed**, exit 0, 14.49s | gate 4 |
| `pnpm typecheck` | exit 0, no output | gate 5 |
| `node .hai-validate.mjs .hai/state.yaml` | valid, exit 0 | gate 1 (builder) |
| pytest ×4 suites | 129 passed, exit 0 | gate 2 (builder) |
| `node --test tests/worker_observer_test.mjs` | 1 pass / 0 fail | gate 3 (builder) |
| `gen_discovery --check` | current, exit 0 | gate 6 (builder) |
| `pytest -k test_bounded_activity_page` | 1 passed | gate 7 (builder) |

## Mutation testing (the checks that separate real coverage from theatre)
Each mutation applied to the real source, the specific suite run, then the source restored
and verified byte-identical (`diff -q` against backups) with suites green again.

| Mutation | Expected failing test | Actual |
|---|---|---|
| Delete `activityAfter = status.activity?.cursor ?? activityAfter` (`index.ts:674`) | T-15 cursor advance | **only** `× advances its own journal cursor across polls instead of restarting the page` — 1 failed / 48 passed |
| `if (false && read.lossy)` (`console.ts:355`) | T-11 truncation | **only** `× reports a payload the collection cap cut as truncation, naming the cap and the retained tail` — 1 failed / 68 passed |
| `if (suppressed === undefined \|\| true) return result` (`console.ts:274`) | T-13 fallback | **only** `× keeps a run readable against a console that refuses the paging options, retrying once` — 1 failed / 48 passed |

Restore verification: both files `diff -q` clean; re-run `158 passed / 5 files`.
Contrast with the M1 cost test: that one did NOT fail under its own mutation and was carried
as WARN M1-W1. The M2 tests genuinely discriminate.

## Live-console paging (proves the live test is not silently on the fallback path)
Fixture: temp run root, 250-record journal, `.refactory/invocation.json` written.
`python -m refactory_dsh.console status --run <dir> --activity-after 200 --activity-limit 200`
→ `after=200 cursor=250 total=250 page_size=200 has_more=False n_events=50`
`last_event_at=2026-10-26T09:04:09+00:00`.
The real console accepts both options and returns a correct page, so the fallback path is not
what the live suite is exercising.

## Use cases covered
| Spec use case | Method | Result | Evidence |
|---|---|---|---|
| UC-7 truncation failure | fake seam with `lossy` reader (T-11), service translation (T-14) | pass | vitest T-11/T-14 |
| UC-8 absent JSON | complete non-JSON payload keeps `the console printed no JSON:` verbatim (T-11 second half) | pass | vitest |
| UC-9 new panel / old console | exit-2 `unrecognized arguments` → one retry without the flags, `activity.paged === false` (T-13) | pass | vitest |
| UC-10 old panel / new console | new `RefactoryActivity` members are optional (T-23a codec shape) | pass | vitest |
| R-4.2 cursor continuity | watchStream advances from the answer's own `activity.cursor` (T-15) | pass | mutation-verified |

## Live calls and timings
None. No model/service calls were made; no authorization was required or sought.

## Log review
- No server, port, or long-lived process started by this milestone; nothing to stop.
- Builder reported three `contract.spec.ts` `spawnSync` 5s timeouts on a loaded host early in
  its session; the same files pass in ~1.1s idle, `contract.spec.ts` is unmodified by M2, and
  the orchestrator's re-runs were green. Recorded as environmental, not code.

## Failures and retries
- Round 1 review of M1 (prior milestone) — not this one.
- No M2 failures; no retries beyond the designed single console fallback (T-13 asserts
  exactly 2 spawns, and a repeat refusal does not loop).

## Residual human UX checks
- §11 item 7 (a real cap overflow with `collectBytes: 1024`) remains an **M4** manual check.
  At M2 the truncation verdict is proved at the boundary with a fake `lossy` reader and at the
  service with a fake console; the end-to-end overflow is not yet exercised.

## Cleanup
- Fixtures in `/tmp` and `mktemp -d`; mutation backups `/tmp/{con,idx}.bak` removed after
  restore verification; source restored byte-identical. No processes left running.
