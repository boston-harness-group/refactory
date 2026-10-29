# Integration evidence: M4 integration (final)

## Environment
- Workspace: source_root `/Users/rawhad/3_Resources/external_libs/refactory-dsh`,
  worktree_root `<source>/.hai-worktrees/INT-001-activity-verbosity`,
  branch `hai/INT-001-activity-verbosity`, base_commit `02a86790a6acc263d77d0f6588d62bcbfda6a055`.
- Tested tree: HEAD `2840cfd2b0` (M1+M2+M3 accepted) + this milestone's T-3 strengthening
  (`refactory/tests/test_live_activity.py`, uncommitted at the time of the sweep).
- Spec: revision 6, sha256 `ad48b75350a20d08eaa1e131ebed2b4e880c5835d656a4ec2553ce55774887ac`.
- State: `.hai/state.yaml` sha256 `6dbab33c…604d`, validator valid, exit 0.
- Runtime: worktree `.venv` (CPython 3.12.10, refactory_dsh editable), Node 24.1.0.
- **Offline. No live model/API calls were made; none were authorized or needed.**

## Round history
- M1: builder round 1 → review FAIL (2 blockers) → builder round 2 → orchestrator verification
  (both reviewer subagents crashed); log at `M1-integration.md`.
- M2: builder → orchestrator verification with 3 independent mutations; `M2-integration.md`.
- M3: builder → orchestrator verification; `M3-integration.md`.
- M4 (this file): integrated gate sweep + console-side manual checklist + WARN M1-W1 closure.

## Commands and results — all eight done gates in ONE integrated run
| Gate | Command | Result |
|---|---|---|
| 1 | `node .hai-validate.mjs .hai/state.yaml` | valid, **exit 0** |
| 2 | `pytest tests/test_live_activity.py tests/test_gui_loop.py tests/test_observe.py tests/test_console_projects.py -q` | **129 passed**, 23.93 s |
| 3 | `node --test tests/worker_observer_test.mjs` | **pass 1 / fail 0** |
| 4 | `pnpm vitest run packages/refactory/refactory-console/tests packages/refactory/tool-refactory/tests packages/client/ui-refactory/tests packages/bundle/refactory-web/tests` | **19 files / 366 tests passed** |
| 5 | `pnpm typecheck` | **exit 0** |
| 6 | `python -m refactory_dsh.tools.gen_discovery --check` | **exit 0** |
| 7 | `pytest tests/test_live_activity.py -q -k test_bounded_activity_page` | **1 passed**, 51 deselected |
| 8 | gates 2–5 green together (above) + no `usage.calls` arrays | **verified**, see below |

## WARN M1-W1 closed — the cost test now discriminates
At M1 the cost assertion `page(20k) < max(page(500)*2, 5.0)` **passed under its own mutation**
(an injected whole-index read measured 0.74 ms against a 5.0 ms floor). Root cause: the 64:1
striding makes a full index read cheap in absolute terms at 20,000 records.

Resolution in this milestone (`refactory/tests/test_live_activity.py`): the journal sizes are now
`(500, 5000, 20000, 270000)` — 270,000 being the scale of the run that prompted the issue — and
the slope is asserted **with no absolute floor** on the largest pair.

Mutation evidence (whole-index read re-injected into `live.page`, then reverted):
| Scale | Correct page | Mutated page | Ratio | Result |
|---|---|---|---|---|
| 20k vs 5k | 0.78 ms | 0.66–0.80 ms | 1.16× | passed — mutation invisible |
| **270k vs 20k** | **0.61 ms** | **2.70 ms** | **3.56×** | **FAILS** `assert 2.696 < 0.807*2` |

The strengthened test passes on the correct implementation (1 passed in 4.87 s) and fails on the
mutation, both verified. `live.py` restored byte-identical (`diff -q` clean); full python suite
after restore: **52 passed**.

## Console-side manual checklist (§11 items executed without a browser)
Fixture: a 5,000-record journal with `usage.calls` stored as an integer per record.

| Item | Method | Observed | Result |
|---|---|---|---|
| 9 — cold vs warm + the three derived files | two successive `console status` | cold 151 ms, warm 124 ms; `events.index.jsonl` 83,187 B, `events.index.provenance.json` 351 B, `events.index.overview.json` 96 B | pass |
| 2 — verbose reaches the end | page until `has_more: false` | 25 pages, 5,000 records seen, `total` 5,000, **complete** | pass |
| 5 — old journal with `usage.calls` still opens | `console status` on it | opens, records preserved, `outputs` present. **Corrected during PR preparation:** the journal's own event records are served *unchanged* (R-3.3 — `events[0].usage.calls` stays a list, no migration), while the worker *summary* row reports `calls` as a count (SD-7, §4.3). The earlier wording conflated the two and implied events were rewritten; they are not, and must not be. | pass |
| 8 — journal growth linear in requests | 20/80/160/320 requests | 123.5 / 123.8 / 124.7 / 125.3 bytes per record; **0** `"calls":[` arrays | pass |
| 8 — quadratic → linear quantified | old shape simulated | a single 320-request record would carry 320 call entries (12,167 B); whole old journal ≈1,959,127 B vs **72,646 B** measured → **27× smaller** | pass |
| 10 — replacement answered from the new journal | replace `events.jsonl` wholesale | before `(5000, …09:23:19)`, after `(1, '2027-01-01T00:00:00+00:00')` | pass |

Paging bounds the payload: `--activity-limit 200` → 101,890 B of events, `--activity-limit 1000` →
509,890 B (`outputs` and `workers` 2 B each here), so the wire is bounded by `page_size`, not by N.

## NOT verified (disclosed, not claimed)
1. **§11 item 7 — a real cap overflow.** Not proven end-to-end. The boundary is proven with a
   fake `lossy` reader at the seam (T-11) and with a fake console at the service (T-14); two
   attempts to drive a real `collectBytes: 1024` overflow failed because the probe was
   constructed against `Console` incorrectly (it requires a context with a subprocess service,
   `TypeError: Cannot read properties of undefined (reading 'pythonBin')`). The probes were
   removed. **Owner:** whoever next touches the console boundary; the correct harness is the
   existing `service-live.spec.ts` pattern, which spawns the real python console.
2. **Browser-level panel items (1, 3, 4, 6, 11)** — need a live GUI and a real long run.
   Automation covers their logic (T-16…T-22, T-18b, T-20), but a human should still confirm the
   feel, the virtual-scroll thumb at 269,900 records, and the level control's keyboard use.
3. **`pnpm build`** (browser bundle) was not run; `tsc -b` covers types.

## Residual WARN/NIT carried forward
- **M1-W1** — CLOSED (above).
- **`compact` hides `activity.resources`** per the §2 R-2 table's literal wording (spec-sanctioned).
- **`readStatusStream`** is declared but not populated in production wiring; the panel works
  through the `Remote.status` path. A precise journal-range reader remains future work (SD-9).
- **Virtual-scroll thumb** is sub-pixel on a 269,900-record run; no clause requires scaling it.
- SD-4's "36 MB per poll" wording conflates payload with the collection cap (pre-existing).

## Use cases
UC-1…UC-12 are covered by the gate-2/gate-4 suites (mapped per milestone in
`M1/M2/M3-integration.md`); UC-2's real-scale case is now additionally protected by the
270,000-record cost assertion.

## Cleanup
- All probe files removed: `gate8.probe.spec.ts`, `zz-cap.probe.spec.ts`,
  `packages/client/ui-refactory/tests/zz-scale.probe.spec.tsx`, `zz-cap.probe.spec.ts`
  (`git ls-files --others` shows no probe files; `git status` shows only the intended T-3 change).
- Mutations reverted with `diff -q` proof (`live.py` byte-identical).
- Fixtures in `mktemp -d`; no servers, ports, or background processes left running.

## Regression picture (verified against base `02a86790a6`, not assumed)

The full suite was run twice on a loaded host: **93 failed / 3182 passed / 16 skipped in 55:54**.
On a quiet host the same suite reported 15 failed / 3246 passed, and the base commit reported
23 failed. The count is load-sensitive, so the failure *set* was compared module by module
against a throwaway base-commit worktree (`/tmp/base-check` at `02a86790a6`):

| Module (scope-adjacent) | Build | Base | Verdict |
|---|---|---|---|
| `test_live_activity.py` | **52 passed** | 40 passed | +12 new tests, none broken |
| `test_observe.py` | 22 passed | 22 passed | unchanged |
| `test_console_projects.py` | 2 passed | 1 passed | +1 new |
| `test_gui_loop.py` | 53 passed | 52 passed | +1 new |
| `test_interactive.py` | 5 failed | **5 failed (same tests)** | pre-existing |
| `test_cli.py` | **50 passed** | 18 failed / 32 passed | base tree lacks deps; build is better |

Modules outside scope, each compared the same way: `test_workers.py` +
`test_worker_permission_contexts.py` 9 failed on **both** sides; `test_builder_probe.py` 4 failed
on **both**; `test_resource_lifecycle.py` 3 failed on **both**; `test_model_timeout.py` 1 failed
on **both**; `test_edit_design.py` 7 failed (build) vs **11 failed** (base).

`test_additive_launcher.py` needed a note: it **skips** in the base worktree because there is no
`node_modules` there, but fails identically when `node_modules` is linked in (verified, then the
link was removed). It depends on `node_modules/.bin/dsh` generating a web profile manifest, so it
is pre-existing and environment-dependent, not introduced here.

**Conclusion: no failure set introduced by INT-001.** Every module in scope is either unchanged,
strictly better, or has an identical pre-existing failure set. The 93-vs-23 spread is load
sensitivity in probe/cleanup tests measured on a host running concurrent builds, not a regression.

---

# Continuation (2026-10-27): the three unverified items, closed

The loop was restarted against the same execution tree (worktree verified, `source_changes: []`,
`.hai/state.yaml` unchanged at `6dbab33c…`, spec unchanged at revision 6 `ad48b753…`). All three
items the first pass disclosed as unproven were closed rather than re-reported.

## Item 1 — a REAL `collectBytes` overflow (was: not proven end-to-end) — CLOSED

The earlier attempts failed because the probe constructed `Console` without a context that
provides a subprocess service. The correct harness is the one `service-live.spec.ts` already
uses: a `Context` with `LocalSubprocessRuntime`, then the service over it.

Added as a **permanent** test in `packages/refactory/refactory-console/tests/service-live.spec.ts`
(a one-off probe was not kept):
`it('names a real capped payload as truncation, not as missing JSON')` — a real console
subprocess over a 400-record journal with `collectBytes: 2048`.

Observed (real process, not a fake reader):
```
code:    refactory/console-truncated
message: the console's output was truncated at the 2048-byte collection cap before it
         printed JSON (only 2048 bytes were retained); raise collectBytes or page the read
         <exact argv>
details: { capBytes: 2048, retainedBytes: 2048, argv: [...] }
says "printed no JSON": NO
```

**Mutation-proven**: setting `if (false && read.lossy)` in `console.ts:355` makes exactly this
test fail with `expected 'refactory/console-failed' to be 'refactory/console-truncated'` —
i.e. it reproduces the reported issue's symptom — and the suite returns to 10 passed after the
source is restored byte-identically.

## Item 3 — `pnpm build` (was: not run) — CLOSED, and it exposed a real gate gap

Running it failed first:
```
gen-api-reference: refactory/docs/api/reference.md is stale
[ELIFECYCLE] Command failed with exit code 1.
```

**Root cause: this change caused it.** `console.status` gained `activity_after`/`activity_limit`,
so the generated API reference was stale. The spec's §12 gates included
`gen_discovery --check` but **not** `gen_api_reference --check` or `pnpm build`, so a stale
generated artifact shipped green through the entire first pass. That is a gate gap, not a doc nit.

Regenerated with `pnpm run docs:generate` and reviewed the diff:
- `refactory/docs/api/reference.md` — `status` signature gains `activity_after`/`activity_limit`
- `refactory/docs/tutorial.html` — the `status` action text now names the two flags
- `refactory/refactory_dsh/data/documentation.json`, `packages/refactory/tool-refactory/src/generated-help.ts` — the generated help/docs identity
- lowering snapshots and the design-graph fixture: **unchanged**

Then `pnpm build` → **exit 0** (`docs:check` + `typecheck` + both `tsdown` faces, including
`@deepseek-ai/dsh-client-ui-refactory/client` at 231.98 kB). `pnpm run docs:check` → exit 0.

**Recommendation for the spec:** §12 should include `pnpm run docs:check` (or `pnpm build`) in
the gate list. Left as a recorded finding rather than silently amended into the accepted spec.

## Item 2 — browser-level panel checks (was: deferred to a human) — CLOSED as far as automation can

The first pass screenshotted static states. This pass drove the real gestures through the real
component and asserted the outcomes:

| Gesture | Method | Observed | Result |
|---|---|---|---|
| Switch level by clicking the radio | `fireEvent.click` on the real `<input type="radio">` | `standard` → `compact`; 3 radios `compact,standard,verbose` | pass |
| Scroll a whole window in verbose | set `scrollTop = 24 × 200`, `fireEvent.scroll` | drawn rows move **0–199 → 200–399**, still exactly **200** rows | pass |
| Scroll container spans history at N=5,000 | read `style.height` | **120,000 px** (`5000 × 24`) | pass |
| Compact keeps history reachable (DEC-001) | render at `compact` | history disclosure **present**, "Load earlier activity" **present** | pass |
| Compact hides outputs (R-2 table) | render at `compact` | `[data-refactory-activity-outputs]` **absent** | pass |

Still a **genuine human check**, not claimed by automation: the *feel* of virtual scrolling in the
running GUI, and the sub-pixel scrollbar thumb on a 269,900-record run.

## Gates after the continuation (all re-run on this tree)
| Gate | Result |
|---|---|
| 1 state valid | unchanged `6dbab33c…`, valid |
| 2 python ×4 | **129 passed** |
| 3 observer | **pass 1 / fail 0** |
| 4 vitest | **19 files / 367 tests passed** (was 366; +1 real-overflow test) |
| 5 typecheck | exit 0 |
| 6 discovery | exit 0 |
| 7 bounded activity page | 1 passed |
| **8 `pnpm build`** | **exit 0** (new: `docs:check` green) |

## Cleanup
- Throwaway probes removed (`zz-cap-probe.spec.ts`, `zz-interact.probe.spec.tsx`); only the
  permanent test in `service-live.spec.ts` remains. `git status` shows 5 intentional files.
- Mutations reverted with `diff -q` proof; `lib/` build output is gitignored.

---

# Live GUI verification (2026-10-27)

The panel was finally checked in a **running GUI serving this branch**, not only in jsdom.
This closes the last human-shaped gap for the *logic*; the remaining human check is only taste
(the feel of virtual scrolling), not behaviour.

## How the checkout is deployed with its own harness home
The repository ships a launcher (`./refactory-dsh`) that boots DSH over the pinned harness. The
machine-state parts live in a DSH home, and the source checkout used an **isolated one**
(`<checkout>/.dsh-home`, gitignored) rather than the global `~/.dsh`, so Refactory's sessions,
profiles and credentials do not mix with the global harness:

- `DSH_HOME=<checkout>/.dsh-home` — honoured by `dsh_profile.dsh_home()` (`dsh_profile.py:189-192`: explicit env value, then `~/.dsh`);
- `profiles/{graph,refactory}` hold **link:** dependencies pointing at this checkout's
  `packages/bundle/*`, so the GUI loads the checkout's own bundles;
- credentials come from that home's `.credentials.yaml` / `.env`.

To serve **this branch** instead of `main`, the same pattern was repeated in the worktree:

```
cd <worktree_root>
mkdir -p .dsh-home                                   # copy credentials in, chmod 600
DSH_HOME=$PWD/.dsh-home <provision the 3 profiles>   # graph, refactory (init from web), refactory-web
pnpm build                                           # bundles carry the branch's panel code
DSH_HOME=$PWD/.dsh-home ./refactory-dsh --no-open --port 3081
```

Verification that the served app is this branch, not `main`:
- the profile's dependencies resolve to `<worktree_root>/packages/bundle/...`;
- the Modes panel header reads **"Modes project · INT-001-activity-verbosity"**;
- `packages/client/ui-refactory/lib/client.js` (231,983 B, built at 10:05) contains the new
  panel identifiers.

## Observed in the live GUI (Playwright, real server on 127.0.0.1:3081)
Fixture: a synthetic but realistic run at `/tmp/demo-run` — 4,200 journal records, a three-node
graph checkpoint, one worker receipt, `facts.json` — attached through the panel's
"Existing run directory → Read this run" field.

| Level | Radio selected | DOM rows | History | "Outputs and validation" | Console errors |
|---|---|---|---|---|---|
| default (none clicked) | **standard** | **200** | 200/4200 | shown | none |
| after clicking Verbose | verbose | **200** | 200/4200 | shown | none |
| after clicking Compact | compact | **200** | 200/4200 | **hidden** | none |

Reading of that table against the intent:
- **DEC-002** — three levels, and the default really is `standard` with nothing clicked.
- **DEC-001** — at *every* level, including compact, the history stays reachable (`200/4200`),
  so no level is a tail.
- **R-2.5 / SD-12** — the DOM holds 200 rows whatever is held or selected.
- The page-size and total shown are the console's real paged answer for this run
  (`total=4200 events=200 has_more=True page_size=200` from `console status`).

Screenshots: `/tmp/pr-shots/live-{standard,verbose,compact}.png` (panel section, cropped).

## Still a human check
Whether virtual scrolling *feels* right, and whether the scrollbar thumb is usable on a
269,900-record run. Both are taste/usability judgements that a screenshot cannot settle.

---

# Layout defect on a live run, and its fix (2026-10-27)

Found by a human reading a live 437-record run in the panel: the activity history opened with a
large empty gap above its first row. Three separate causes compounded.

| Cause | How it was found | Measured before | After |
|---|---|---|---|
| Row height was assumed, not enforced | `RunActivity.tsx` declares `ROW_HEIGHT = 24` with the comment "a row index is a scroll position", but `.recorded > li` was `padding: 8px 0` in an `8px`-gap flex column | row 35.59px + 8px gap = **43.59px** | **24px**, gap 0 |
| Reserved height disagreed with the content | `clientHeight 320`, `scrollHeight 15,503` vs inline `height: 11,592px` | mismatch | **12,696 = 12,696** |
| The viewport was never moved to the drawn slice | the slice was translated by `first * ROW_HEIGHT` while the element stayed `scrollTop 0` | first row **7,896px** below the viewport top | **0px** |

Fix:
- rows are fixed-height (`height: 24px; box-sizing: border-box`), gap 0, and a long record
  ellipsises instead of wrapping — the translate offset and the scroll-to-index mapping are
  arithmetic on `ROW_HEIGHT`, so a row that grew put every later row out of step with the scrollbar;
- the viewport is positioned at the drawn slice when the history **opens or the level changes**
  (`key={level}` + a ref), deliberately **not** on a page arrival, which would yank a reader who
  is looking further back;
- `standard` still opens at the newest records (SD-2) and both levels follow the reader once they
  scroll (SD-12).

Verified on the live run (529 records by then): `scrollTop 7896`, `firstRowTop == viewportTop`
(both 2834.1), newest records drawn, **no console errors**.

Regression test added (`body.client.spec.tsx`): the slice index the element reports and the offset
it is positioned at must agree, which is exactly what the reported gap violated. The first attempt
at this fix used a `ref` keyed on `level + events.length`; it broke an existing scroll test by
repositioning on every page arrival, which is why the key carries the level only.

**Lesson worth keeping:** the panel's DOM-bound tests all passed throughout — 199 of them — while
the layout was unreadable, because none asserted anything about position. A bound on *how many*
rows are drawn says nothing about *where* they are.
