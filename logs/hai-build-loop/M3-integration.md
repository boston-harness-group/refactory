# Integration evidence: M3 panel

## Environment
- Workspace: source_root `/Users/rawhad/3_Resources/external_libs/refactory-dsh`,
  worktree_root `<source>/.hai-worktrees/INT-001-activity-verbosity`,
  branch `hai/INT-001-activity-verbosity`, base_commit `02a86790a6acc263d77d0f6588d62bcbfda6a055`.
- Tested tree: HEAD `a09ba3cfe1` (M1+M2 accepted) + uncommitted M3 changes,
  `git diff --stat` = **11 files, +1359/−44**, all under `packages/client/ui-refactory/`.
- Spec: revision 6, sha256 `ad48b75350a20d08eaa1e131ebed2b4e880c5835d656a4ec2553ce55774887ac`
  (unchanged). `docs/`, `.hai/`, `refactory/`, `packages/refactory/` untouched by M3.
- State: `.hai/state.yaml` sha256 `6dbab33c…604d`, validator `valid: …`, exit 0.
- Runtime: worktree `.venv`, Node 24.1.0, pnpm `node_modules`. Offline; no live calls.

## Round history
- **Builder round 1** (`66bbfe1d`): implemented M3; gates 4/5 green; 1/2/3/6/7 re-verified.
- **Verification**: orchestrator-executed (independent gate re-run, source inspection,
  mutation re-run). Reviewer subagents crashed on M1; this round the orchestrator verified
  directly and records that provenance rather than implying a reviewer PASS it does not have.

## Commands and results (orchestrator re-run)
| Command | Result | Evidence |
|---|---|---|
| `pnpm vitest run packages/refactory/refactory-console/tests packages/refactory/tool-refactory/tests packages/client/ui-refactory/tests packages/bundle/refactory-web/tests` | **19 files / 366 tests passed**, exit 0, 6.11s | gate 4 |
| `pnpm typecheck` | exit 0, no output | gate 5 |
| builder-reported, re-verified | gate 1 validator exit 0; gate 2 129 passed; gate 3 1/0; gate 6 exit 0; gate 7 1 passed | builder report |

## DEC-001 / DEC-002 satisfaction (the milestone's real purpose)
- **DEC-002 (three levels, standard default)**: `store.ts:118`
  `export const DEFAULT_ACTIVITY_LEVEL: RefactoryActivityLevel = 'standard'`; fresh tab state
  seeds `activityLevel: DEFAULT_ACTIVITY_LEVEL` (`store.ts:368`).
- **DEC-001 (full history kept, never tail-only)**: the held history is accumulated without a
  cap — `store.ts:604` `events: [...prefix, ...held.events]` — and `complete` is set only by an
  answer carrying `has_more === false`. The paging control is drawn at **every** level, so
  `compact` does not make the history unreachable (R-2.4).
- **Rendering is bounded, holding is not** (the G-2 correction from spec review):
  `HistoryWindow` (`RunActivity.tsx:116-149`) draws `events.slice(first, first + RENDER_WINDOW)`
  only, with the drawn slice translated so the scrollbar keeps the real history length. DOM
  rows are therefore bounded by `RENDER_WINDOW` (200) regardless of N, while the records held
  are unbounded. This is exactly the "reachable, not all-on-screen" contract.

## Cursor continuity (R-4.2/F-2/F-3) — three bugs fixed by the builder
| Bug | Fix | Test |
|---|---|---|
| Refresh sent the progress cursor | sends `{ after: max(held.through, status.activity.cursor) }` | T-17 |
| `watchStream` opened at 0 (the `face.ts:408` bug) | opens at page 0's `activity.cursor` | T-16 |
| `activityEarlier` corrupted the forward cursor | earlier-page control reads the held history's first index, not `total` | T-19 |

## Mutation testing (orchestrator re-ran a subset; builder ran five)
| Mutation | Caught by |
|---|---|
| cap `HeldHistory.events` at 200 (verbose becomes "last N") | 4 tests: attach-continuity, both verbose-exhaust tests, page-arrival |
| render every held record (the original `RunActivity.tsx:41-42` bug) | 3 tests incl. T-18b |
| drop the paging control at `compact` | T-18 compact test |
| refresh passes only `status.cursor` | T-17 |
| `watchStream` restarts at 0 | T-16 stream test |

All sources restored and `diff -q` verified byte-identical afterwards (builder). The
orchestrator independently confirmed the shipped assertions exist and read the bounding code:
`body.client.spec.tsx:346-394` asserts `rows(view).length ≤ RENDER_WINDOW + 2` at N=5,000 and
after scrolling to row 4,600 (T-18b).

## Use cases covered
| Spec use case | Method | Result |
|---|---|---|
| UC-1 live attach | T-16 face: attach starts both cursors at 0, polls advance | pass |
| UC-2 large run in verbose | T-18/T-18b: exhaustive paging to `complete`, 200-row DOM bound at N=5,000 | pass |
| UC-3 level switching | T-18: three levels, standard default, compact keeps the history control | pass |
| UC-5 journal growing while paging | T-18 exhaust-resumable, T-19 earlier-page walk | pass |
| R-6.3 older console | T-20: unpaged note shown, not silently presented as the whole history | pass |
| R-5 failure lines | truncation (`capBytes`/`retainedBytes`) vs no-JSON rendered distinctly | pass |

## Live calls and timings
None.

## Log review
- No server or long-lived process started. `pnpm build` was NOT run (browser bundle) — the
  aggregate `tsc -b` covers types; a real bundle build is deferred to M4's gate sweep.

## Failures and retries
- None in M3. No retries beyond the designed M2 console fallback.

## Residual human UX checks (carried to M4, disclosed not claimed)
1. **§11 manual checklist (browser-level) was NOT executed** — items 1-7 need a live GUI and a
   real large run. Not claimed as done.
2. **Virtual-scroll thumb on a 269,900-record run is sub-pixel** (viewport height is
   `N × ROW_HEIGHT`); a thumb scale factor was deliberately not added since no clause requires
   it and it would change the row↔scrollTop mapping. M4 UX look.
3. **`compact` hides `activity.resources`** per the §2 R-2 table's literal wording. If the
   reviewer reads that row as per-worker usage only, it is a one-line change.
4. **`readStatusStream` is declared but not populated** in production wiring (`index.ts`
   unchanged by M3); the panel works today through the `Remote.status` path. The precise
   journal-range reader remains an M4/M2-shaped follow-up.
5. Truncation copy uses a new locale key `activity.truncated`; a locally-introduced network
   failure has no `details` and falls back to the generic failure line by design.

## Cleanup
- The probe spec `zz-scale.probe.spec.tsx` was created under `tests/` and **deleted** after the
  run (`rm` verified; `git status` shows it absent). Fixtures otherwise in-memory. No processes
  left running.
