# Devlogs

## 2026-10-26 — INT-001: the activity view keeps the full history at three levels

**Intent** (`.hai/state.yaml` INT-001, DEC-001, DEC-002): a watched run's activity
view keeps the full activity history and renders it at compact, standard, or
verbose detail, defaulting to standard. DEC-001 forbids replacing the history
with a tail-only view; DEC-002 fixes the three level names and the default.

**Why it mattered.** Issue 511 reported the panel failing on a long `create` run
with "the console printed no JSON", then "Run type is not recognized yet", while
the run and `refactory status --follow` stayed healthy. Three causes stacked:

1. `worker_observer.mjs` wrote the receipt's full cumulative `usage.calls` array
   into every journal event, so the journal grew quadratically (RES-003). A
   320-request worker alone produced a 12 KB record; the whole journal was ~1.9 MB
   where the same run now writes 72 KB.
2. `console.status` built its activity from `live.snapshot(run_root)` with no
   cursor, so `activity.events` was the entire journal, and `snapshot` read it
   twice. A 269,900-record run answered with a 73,953,525-byte payload.
3. The console runner collected at most 4 MiB and reported any undecodable stdout
   as "the console printed no JSON", so a truncated payload was indistinguishable
   from a console that printed nothing.

**What shipped** (spec `docs/refactory-activity-verbosity-spec.md`, revision 6):

- A derived, disposable index beside `events.jsonl`: a strided offset index for
  seeking (one fixed-width line per 64 records), an overview line carrying `total`
  and the newest `at`, and a one-line provenance anchor. A warm page costs
  0.61 ms and a warm summary 0.09 ms at 5,000 / 20,000 / 270,000 records — flat.
  Deleting the files loses nothing (R-6.5); a cold call may scan.
- `console status --activity-after/--activity-limit`, on `status` and
  `status --follow`, with `total` counting records regardless of paging and a
  page bounded by `page_size`. The follow loop advances instead of re-reading.
- The observer records a call *count*, not the cumulative array. Old journals
  that carry arrays stay readable with no migration.
- The console boundary tells a payload the collection cap cut apart from one that
  printed no JSON, naming the cap and the retained tail — the facts the subprocess
  seam actually delivers (it exposes no byte total, and spill is unconfigured).
- An older console that refuses the paging flags is retried once without them and
  the answer is marked unpaged.
- The panel offers compact / standard / verbose with **standard** as the default,
  keeps every record (uncapped), pages to `has_more: false`, and draws at most 200
  rows whatever the history size, so the scrollbar spans the run while the DOM
  stays bounded.

**Decisions worth remembering**

- **The spec had to reproduce today's answers, not improve them.** Revision 3
  defined `last_event_at` as "the newest record that has an `at`", which changes
  the answer for a journal whose newest record is a `journal_gap`. Revision 4
  anchors it to the shipped expression (`all_events[-1].get('at')`), verified
  against HEAD on a 15-case matrix. A spec that silently changes correct behavior
  is a regression wearing a contract's clothes.
- **Three derived files, not one.** Folding `total`/`last_at` into the existing
  provenance anchor was measured and rejected: three independent processes append
  to the journal and only `live.emit` inspects the previous byte, so an in-process
  anchor can never track the other two — a stale anchor would be normal, and would
  answer a wrong `total`. §1.2 was amended to name the third file explicitly.
- **A rejected escalation.** A reviewer filed HUMAN_ESCALATION claiming the spec
  redefined a human-visible value. Running HEAD's own `snapshot` over the matrix
  showed HEAD agrees with the rule 15/15; the premise was false and nothing was
  owed to the human. Escalating a two-way engineering trade inverts the authority
  model.
- **A disclosed limitation, not a hidden one.** A shape-consistent but wrong
  `last_at` in the overview is served, because the anchor digests record 0 only.
  Authenticating it per call would reintroduce the O(N) read the index removes, so
  §4.6 states the limitation and deletion restores the true answer.

**Testing notes (the part that generalizes)**

- A test that cannot fail is worse than no test. The M1 cost assertion
  `page(20k) < max(page(500)*2, 5.0)` **passed with the O(N) read re-injected**:
  the 64:1 striding keeps a whole-index read cheap in absolute terms, and the 5 ms
  floor swallowed it. Strengthened at M4 by asserting the slope **with no absolute
  floor** at 270,000 records, where the mutation costs 3.56× a page and now fails.
- Mutation-test every claim that a test "proves" something. M2/M3 each shipped
  mutations that failed exactly their intended test and nothing else; that is the
  check that separates coverage from coverage theatre.

**Not verified** (disclosed in `logs/hai-build-loop/M4-integration.md`)

- A real `collectBytes` overflow end-to-end: proven with a fake `lossy` reader at
  the seam and a fake console at the service, but not driven through a real
  subprocess. Two probe attempts were malformed and removed.
- Browser-level manual checks (feel, keyboard use, the virtual-scroll thumb at
  269,900 rows): the logic is covered by tests; the human checks are still open.
- `pnpm build` for the browser bundle was not run (`tsc -b` covers types).

**Continuation, 2026-10-27 — the three disclosed gaps closed**

- **A real `collectBytes` overflow now has a permanent test.** The first pass proved truncation
  only with a fake reader and a fake console; the end-to-end probe had failed because it built
  `Console` without a subprocess-providing context. Using the `LocalSubprocessRuntime` harness
  `service-live.spec.ts` already had, a real console subprocess with `collectBytes: 2048` now
  answers `refactory/console-truncated` naming the cap, and does not say "printed no JSON".
  Mutation-proven: disabling the `lossy` branch makes it fail with `console-failed`, which is
  exactly the reported symptom.
- **`pnpm build` was red, and this change caused it.** `gen_api_reference --check` reported
  `refactory/docs/api/reference.md` stale because `console.status` gained the two paging
  parameters. §12's gates included `gen_discovery --check` but not `gen_api_reference --check`
  or `pnpm build`, so a stale generated artifact passed the whole first loop. Regenerated the
  four generated files (reviewed; lowering snapshots unchanged) and the build is green.
  **Gate lesson: a generated-artifact gate that is not in the done-gates is not a gate.**
- **Panel gestures are now asserted, not just screenshotted.** Clicking the level radio moves
  `standard` → `compact`; scrolling a whole window moves the drawn rows from 0–199 to 200–399
  while still drawing exactly 200; the scroll container is `N × 24 px` (120,000 px at 5,000);
  compact keeps the history control and drops the outputs disclosure. The *feel* of virtual
  scrolling and the sub-pixel thumb at 269,900 records remain genuine human checks.

**Future work**

- `readStatusStream` is declared but not populated: the panel reaches paged status
  through `Remote.status`. A real journal-range reader would remove the per-poll
  round trip.
- Scale the virtual-scroll thumb so a 269,900-record run has a usable scrollbar.
- SD-4's "36 MB per poll" conflates payload size with the collection cap.
