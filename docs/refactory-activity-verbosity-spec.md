# Refactory activity verbosity: keep the full history, render it at three levels

Status: draft for review (revision 7 — see the Revision log for round 7).
Authority: `.hai/state.yaml` in the original build worktree
(`human_approved` INT-001, DEC-001, DEC-002; evidence H2 supports them). That file
is not carried on this branch, so for a reviewer of the diff the authority is the
requirement set below plus the revision log, not a file in the tree.
This spec covers INT-001 and nothing else. Everything below that is not traced to
INT-001/DEC-001/DEC-002 is either marked **non-normative** or is an agent
decision (`SD-*`) whose rationale and affected IDs are stated where it appears.

Terminology used throughout, in the repo's own words:

- The **journal** is `<run>/.refactory/events.jsonl`, written by
  `refactory_dsh/live.py` (`EVENTS`) and appended to by
  `refactory_dsh/worker_observer.mjs`. It is the activity history.
- The **progress file** is `<run>/progress.jsonl` (`refactory_dsh/observe.py`),
  a search run's step record. Authoring (graph) runs never write it.
- The **console** is `python -m refactory_dsh.console`; the **service** is
  `packages/refactory/refactory-console/src/index.ts`; the **tool** is
  `packages/refactory/tool-refactory/src`; the **panel** is
  `packages/client/ui-refactory/src/client`.

---

## 1. Purpose and non-goals

### 1.1 Purpose

A person watching a Refactory run must be able to read the run's whole activity
history in the panel, at one of three detail levels, with `standard` in force
until they choose otherwise (INT-001). Today that is impossible for exactly the
runs the feature is for: `console.status` embeds the *entire* journal in every
answer (`refactory_dsh/console.py:552` is the graded worktree call; at HEAD the
same call is `console.py:533`, `live.snapshot(run_root)` with no cursor →
`live.py:118-145` of HEAD returns every event in
`activity.events`), the service runs that console through a collection cap
(`packages/refactory/tool-refactory/src/console.ts:92`, `collectBytes`,
4 MiB, `stdout: { maxBytes }` at `:235`) that keeps only the **tail** of stdout,
and the tool then reports `the console printed no JSON` (`console.ts:268`). The
panel renders that string, loses the run kind with it, and therefore shows a run
that is healthy in the terminal as unreadable.

Keeping the history is the point; bounding what travels *for one moment* of
reading is what makes keeping it practical. Paging alone is not enough: the cost
of the *whole* read must add up to something linear, because the panel and
`status --follow` pay it again on every poll. That is what R-1.7 fixes, and it is
the reason R-1 is written as a cost contract and not only as a size contract.

### 1.2 Non-goals

- Not a new **authoritative** activity store, database, or cache of history. The
  journal (`events.jsonl`) stays the only durable record of activity, and every
  record the panel shows comes from it (DEC-001).
  **Explicit exception, required not forbidden:** this spec *does* add one
  derived, disposable, append-only byte-offset index
  (`<run>/.refactory/events.index.jsonl`) because R-1.7b needs `total` and
  `last_event_at` in time independent of `N`. It is normative at §4.6 and
  required by R-1.7; it holds no activity content beyond offsets and one
  derived timestamp per record, it is never read as history, and deleting it
  loses nothing (R-6.5). "Not a new store" means "no second source of truth",
  not "no index file" — §4.6's three derived files are the only such files, and
  they are *required* there.
- Not a new bookkeeping file for anything else: no second journal, no sidecar
  database, no cross-run global index, no cache of parsed records. The only new
  files anywhere in this spec are §4.6's three derived files — the strided
  offset index (`events.index.jsonl`), its one-line identity anchor
  (`events.index.provenance.json`) and its one-line overview
  (`events.index.overview.json`), all under the same rules: derived, disposable,
  never read as history. Three, not one, because one file cannot serve both
  lookups `[R-1.7b]` needs: a page needs to find *where* record `i` starts, and
  `total`/`last_event_at` need the *newest* record, and the index answers the
  first by a fixed-width stride (so its last line says nothing about the
  newest record's `i` or `at`) while the overview answers the second in one
  small read. Each file is still derived, disposable and R-6.5-proof; none is a
  second source of truth.
- Not a rewrite of the worker route/usage contract, the steering ledger, the
  candidate/leaderboard contract, or the graph control contract. Those surfaces
  are touched only where they carry the journal on the wire.
- Not a retention or deletion policy. Nothing in this spec deletes, compacts,
  compresses or rewrites history (DEC-001). Growth control is about **not
  duplicating** history, not about removing it.
- Not a change to what counts as evidence of compliance (RES documents that
  receipts prove dispatch, not compliance; that distinction is preserved).
- Not the two-lane bounded console reads proposed for other work (see §8).

---

## 2. Traceability summary and normative requirements

Every requirement below is traced to INT-001/DEC-001/DEC-002. **Where an
obligation is stated:** here, and in exactly four other places, each of which
restates only obligations numbered as clauses of an R above and adds none —
**§4.2** (`[R-1.6.1]`), **§4.4** (`[R-5.3]`, `[R-5.4]`), **§4.6** (`[R-1.6.1]`,
`[R-1.6.2]`, `[R-1.6.3]`, `[R-6.5.1]`, `[R-6.5.2]`, `[R-1.7b]`) and **E-1** (`[R-1.7d]`,
`[SD-4]`). Every other section — including §3's architecture note, §7's
rationale and the SD-* entries in §7.2 — states none. (Round 3 wrongly claimed
"nothing else in this document" uses MUST/SHALL; see the Revision log, H-5.)

### 2.1 Trace table

| R | Statement (short) | Trace | Primary use cases |
|---|---|---|---|
| R-1 | One bounded, paged journal read whose per-page cost does not grow with history; a cursor that advances; exhaustive paging to the end in linear total time | INT-001, DEC-001 | UC-1, UC-2, UC-5, UC-12 |
| R-2 | Three verbosity levels, standard by default; verbose still makes the whole history reachable (held in full, rendered in a bounded window) | INT-001, DEC-002, DEC-001 | UC-3, UC-4, UC-2 |
| R-3 | The journal never carries a cumulative `usage.calls` array again; old journals stay readable | INT-001 (make large journals readable) | UC-6 |
| R-4 | Attach, watch and refresh continue from the highest cursor seen; a run switch starts at 0 | INT-001 (the paged history actually arrives) | UC-1, UC-2 |
| R-5 | Truncation and absent JSON are reported as distinct, actionable failures | INT-001 (large runs must not surface as "printed no JSON") | UC-7, UC-8 |
| R-6 | Cross-version compatibility in both directions | INT-001 (older runs and older consoles must stay readable) | UC-9, UC-10 |

Why R-1.7 is traced, not smuggled in: DEC-001 requires the full history to be
*shown*, and INT-001 requires a person to **watch** a run, i.e. to read it
repeatedly while it works. A paging scheme that costs `O(N)` per page — and the
whole payload per poll today — makes "show the full history of a 269,900-record
run" unreadable in practice (measurements in §7.2 SD-3), so a bound on the
per-interaction cost is part of delivering DEC-001, not an optimisation added on
top. The bound is stated as a testable ratio, not as a stopwatch promise.

### 2.2 Normative requirements

**R-1 — A bounded, paged journal read.** Trace: INT-001, DEC-001.
The console SHALL serve activity from the journal in bounded pages, and SHALL
carry enough information in every answer for a caller to fetch the next page and
to know when it has every record.

1. `console status` SHALL accept, as new options alongside the existing
   `--after INT` and `--follow`:
   - `--activity-after INT` (default `0`, non-negative) — the journal index the
     answer starts at;
   - `--activity-limit INT` (default `200`, accepted range `1`…`1000`; see 3).

   Both are accepted by one-shot `status` and by `status --follow`, and both
   must appear in `status`'s help text (`_parser`, `console.py:1380-1392`), so
   they also show in the generated discovery surface (gate 6). When the
   options are absent the answer still carries `activity.events` holding at most
   `activity.page_size` journal records, in journal order, beginning at journal
   index `activity.after`.
   `[R-1.6.3]` Both new options must be present on the `status` sub-parser
   (`console.py:1380`), i.e. discoverable (`--help`, and the generated discovery
   docs gate 6 checks for freshness) — not read from the environment and not
   positional.
2. `activity.page_size` SHALL default to 200 and SHALL never exceed 1000.
   `--activity-limit` is an **accepted option** (not merely a validated one): a
   value above 1000 SHALL be clamped to 1000; a negative value, `0`, or a
   non-integer value SHALL be refused as a `ConsoleError` (exit 2) before any run
   directory is read. `0` is a refusal and not a clamp, because an answer with
   no records cannot advance a cursor and an empty page is not a page.
3. Every answer SHALL carry `activity.after` (the index the page starts at),
   `activity.cursor` (the index of the next record — `after + len(events)`), and
   `activity.total` (the number of complete records in the journal). Records that
   are not acknowledged by the existing cursor rule (a trailing incomplete write)
   SHALL NOT be counted in `total` and SHALL NOT be served.
4. `activity.has_more` SHALL be `true` exactly when `activity.cursor <
   activity.total`. A caller that wants the exhaustive history SHALL page until
   `has_more` is `false`, re-reading from the last `cursor`. A run whose journal
   grows while the caller pages SHALL eventually reach `has_more: false` on a
   quiet journal; the contract is "every record present at a given `total` is
   retrievable", not a snapshot transaction.
5. Summary fields of `activity` that are cheap **relative to `N`** (`run_id`,
   `invocation`, `started_at`, `last_event_at`, `nodes`, `completed`,
   `running_nodes`, `graph`, `wait_reason`, `workers`, `deliveries`, `resources`,
   `control`, and the `console.status`-added members) SHALL be present on every
   page, so a caller that only ever reads page 0 still sees the current run
   summary. "Cheap" here is **bounded by the run's graph, not by the journal**:
   today (HEAD `live.py:124-125`; worktree `live.py:589-590`) globs and parses
   every worker receipt
   (`.refactory/workers/*.json`) and every delivery receipt
   (`.refactory/agent-delivery/*/receipt.json`) on **every** call — 500 receipts
   measured at ~19 ms in this worktree before the journal is even touched — so
   those members cost `O(workers + deliveries)`, independent of `N` and of
   `page_size`. That is acceptable because the count is bounded by nodes the
   graph declares and because R-3.3 keeps each receipt's `usage.calls` a count;
   it is **not** free and SHALL NOT be described as constant time anywhere in the
   implementation or its docs. An exhaustive verbose pass (R-2's `N/page_size`
   pages) pays this `O(workers + deliveries)` **per page**, which is why the
   measured pass cost in §7.2 SD-3 and F-4 is quoted per page, not as a total.
6. `activity.last_event_at` SHALL equal **today's** answer: the `at` of the
   **newest counted record**, and `null` when that record has no `at`. Today
   that is literally `all_events[-1].get('at')` over the records `live.events`
   returns — the shipped expression at HEAD (`cfd2d24`) is
   `'last_event_at': all_events[-1].get('at') if all_events else None`
   (`git show HEAD:refactory/refactory_dsh/live.py`, `snapshot`, :145 of that
   revision), and the records it indexes are `live.events`' complete, non-blank,
   one-entry-per-line set (`live.py:482-502`), so a `journal_gap` — complete and
   counted but with no `at` (`live.py:86-99`) — makes it `null`, as does an object
   without `at`. **Verified against the baseline itself**, not against this
   spec's reading of it: running HEAD's own `snapshot` over §10 T-25's 15-case
   matrix returns exactly this rule's answer in **15/15** cases (trailing
   `journal_gap`, two trailing gaps, gap-then-unterminated-tail, non-object JSON,
   newest `{}`, gap-then-`at`-less record → `None`; an ordinary newest record →
   its `at`), so the rule below re-states shipped behaviour rather than
   redefining it. It SHALL
   NOT be "the newest record that has an `at`": that reads *past* the newest
   record and answers a value today never returns (seven of fifteen matrix cases;
   §4.6, §10 T-25). It SHALL be computed without reading the journal a second
   time (today `live.snapshot` calls `events()` twice, `live.py:586-588`):
   §4.6's `[R-1.6.1]`/`[R-1.7b]` is the whole rule — one lookup of the
   **overview line's** `last_at` (`events.index.overview.json`), no walk and no
   index scan.
7. **Cost contract.** For a journal of `N` complete records with no writes
   between calls:
   - (a) a page whose `after` is greater than or equal to the `after` of the
     previous call on the same console invocation SHALL be served in time
     independent of `N` — `O(page_size)`, not `O(N)` — so that paging the whole
     journal `N/page_size` times costs `O(N)` in total, not `O(N²)`;
   - (b) `total` and `last_event_at` SHALL be refreshed in time independent of
     `N` per call as well (byte offsets plus an incrementally maintained count,
     not a re-parse of every record). There is **no walk fallback of any kind**:
     both values are read from the **overview line** (§4.6 `[R-1.7b]`) — one small
     file read each, whatever the tail looks like, and never from a scan of the
     index lines. (The overview line is the one place those two values live; the
     strided index cannot answer them, §4.6.)
   - (c) this SHALL hold for `console status`, for `status --follow`'s per-frame
     refresh, and for the service's per-poll `status` — i.e. the polled paths
     SHALL NOT re-read the journal from index 0 per poll;
   - (d) the answer payload SHALL be bounded by `page_size` records, not by `N`.
     This bounds `activity.events` and nothing else: `outputs` (§4.2) is not
     paged and is bounded only by `facts.json`'s own size, nor are the receipt
     members of (5). See E-1 — the one concrete cost of keeping `outputs`.
   - (e) this bound is about the **journal**. `workers`/`deliveries` cost
     `O(workers + deliveries)` per call (5) and are outside `N`'s guarantee;
     they SHALL be no worse than today and SHALL NOT be re-derived per record.

   The mechanism is SD-3 (a byte-offset index file beside the journal, derived
   and append-only, §4.6). The increased `after` restriction in (a) is what makes
   the index findable by binary search; a host that serves every caller from its
   own process (SD-11) may instead keep the index in memory and serve any `after`
   in logarithmic time. §4.6's `[R-1.6.2]` (what the index may claim to cover),
   `[R-6.5.2]` (atomic index writes) and `[R-1.7b]` (the overview line that
   answers (b)) are clauses of this requirement and of R-6.5 respectively.
8. **How (7) is measured.** By a counter the test controls, not by wall-clock
   timing: the tests SHALL count `Path.read_bytes` / `Path.open` calls per
   `status` answer (Python) — one journal pass per not-yet-indexed suffix, none
   proportional to `N` on a fresh, unchanged invocation — and, independently, by
   a bounded-time assertion on a large journal (§10 T-2/T-3) whose threshold is
   chosen wide enough (≥10×) not to flake on a slow machine while still failing
   the current `O(N)` behaviour by more than an order of magnitude. To keep the
   receipt cost of (5)/(7e) out of those assertions, T-3's fixture SHALL declare
   a fixed small number of workers and deliveries (≤8) and SHALL assert that the
   per-page time does not move when the journal grows from 500 to 20,000 records
   **at the same worker count, with a warm index**; the worker-count cost is
   asserted separately and honestly as a per-call slope (T-3b, §10), never as
   "journal size does not matter".

**R-2 — Three verbosity levels, standard by default.** Trace: INT-001, DEC-002,
DEC-001.
The panel SHALL offer exactly three activity verbosity levels — `compact`,
`standard`, `verbose` — in force as `standard` on attach, and SHALL render the
activity history at each level according to this mapping:

| Content | compact | standard | verbose |
|---|---|---|---|
| Run state badge, wait reason, current nodes, completed count | yes | yes | yes |
| Workflow map (`RunGraph`) and recorded-node track | yes | yes | yes |
| Per-worker route/runtime summary (one line per worker) | yes | yes | yes |
| Resource usage, instruction delivery, control receipt | no | yes | yes |
| Activity outputs and validation (kept — §4.2, E-1) | no | yes | yes |
| Journal records rendered as rows in the panel | no | the newest window (SD-2) | a **fixed render window** of the newest records (SD-12), with every other record reachable by scrolling within the window |
| Full journal paging control ("Load earlier activity" / exhaustive read) | yes | yes | yes |
| Held journal records in the tab | 0 | the standard window (SD-2), plus whatever paging the person did | every complete record in the journal: paged in from `after = 0` until an answer returns `has_more: false` (R-1.4) |

1. The level SHALL default to `standard` and SHALL persist per tab for the life
   of that tab record. Switching level SHALL NOT start, restart, resume or
   otherwise touch the run.
2. **`verbose` makes every record REACHABLE; it does not put every record in the
   DOM.** The held history (R-2's last row, `HeldHistory` in §6) SHALL contain
   every complete record, accumulated page by page until an answer reports
   `has_more: false` — that answer, and the `total` it carries, are the evidence
   that the history is complete. **Rendering is windowed and bounded** (SD-12):
   the panel SHALL render at most `RENDER_WINDOW = 200` `<li>` activity rows at a
   time (the same number as one page, so window and page coincide), positioned
   by virtual scrolling over the held array, and the DOM row count SHALL depend
   on `RENDER_WINDOW` and never on `N`. Requirement, stated as a testable bound:
   for a held history of `N` records the number of rendered `<li>` nodes under
   `[data-refactory-activity]` SHALL be `≤ RENDER_WINDOW + 2` (T-18b). Today's
   renderer emits exactly one `<li>` per record (`RunActivity.tsx:41-42`), which
   is why "all of them on screen" is *not* the contract: it contradicts clause 5
   and F-4, and is superseded here.
3. `verbose` SHALL NOT be a tail, a sample, or a "last N": what is reachable is
   the whole history (DEC-001), and the paging control SHALL reach index 0. Total
   fetch cost SHALL be `O(N)` (R-1.7a). A `verbose` pass interrupted by a tab
   switch or a run switch SHALL be resumable from the held cursor, not restarted
   (R-4.4).
4. `compact` SHALL NOT drop the paging control: a person in compact mode can
   still open the full history without first changing level, because DEC-001
   makes the history part of what the panel shows at every level.
5. The panel SHALL stay interactive at every level and for every `N`: a poll, a
   level switch, a page arrival and a scroll SHALL each complete without a
   blocking render proportional to `N`. This is what SD-12 exists to satisfy and
   it takes precedence over any wording in this document that could be read as
   "render every record at once".
6. The levels SHALL be presented as a single three-way control with the chosen
   level visible, in the activity section, and SHALL be reachable by keyboard.
7. Changing level SHALL NOT refetch journal pages the tab already holds, and
   SHALL NOT lose the run's latest summary (state, wait reason, workers).

**R-3 — The journal never carries a cumulative `usage.calls` array again.**
Trace: INT-001.
1. `[R-3.1]` The observer's journal append must stop carrying the cumulative
   array: `save()` in `worker_observer.mjs:26-34` copies the receipt's whole
   `usage` (which holds `calls`, built in `usage()` at `:19-24`) into the
   journal record. After the change it appends a bounded usage summary (status,
   tokens, tokens_delta and the existing `incomplete_nested_usage` / `reason`
   members) and no array whose length grows with the number of model requests.
2. The observer's own receipt file (`.refactory/workers/<id>.json`) MAY keep
   `usage.calls` for local accounting; the bound applies to what is appended to
   the journal, because that is what every reader pays for.
3. `[R-3.3]` `live.read`/`live.events` keep serving pre-existing journals
   unchanged, including records that still carry `usage.calls`, and readers
   tolerate both shapes. A summary row derived from a worker's usage exposes
   `calls` (the number of calls) and no longer exposes the array itself (SD-7).
4. No migration, rewrite, truncation or compaction of existing journals SHALL
   occur (DEC-001). An existing large journal remains fully readable through
   R-1 paging; R-3 only stops new growth.

**R-4 — Cursor continuity in the panel.** Trace: INT-001.
1. Attaching to a run SHALL start journal paging at `after = 0`, so the panel's
   held history begins with the first record.
2. Watch and refresh SHALL continue from the highest `activity.cursor` the tab
   has accepted for the current run, and SHALL NOT reset the journal cursor to
   0. They SHALL also continue from the highest top-level `cursor` (progress
   file) as they do today.
3. The top-level `cursor` and the activity cursor are different cursors over
   different files and SHALL be plumbed separately; the panel SHALL NOT assume
   the activity cursor moves for authoring runs (it does not unless an authoring
   run writes the journal).
4. Switching to a different run SHALL reset both cursors to 0 and SHALL drop the
   previous run's held history.
5. A frame whose `activity.run_id` differs from the run being watched, or whose
   `activity.after` is greater than the highest cursor already held, SHALL be
   treated as a gap: the panel SHALL refetch page 0 rather than splice the page
   onto history it does not match.

**R-5 — Distinct failure reporting.** Trace: INT-001.
1. `[R-5.1]` When the console's collected stdout was truncated by the
   collection cap and does not decode, the boundary reports a failure that (a)
   says the payload was truncated at the collection cap, (b) names the cap in
   bytes, (c) carries the exact `argv`, and (d) is distinguishable, by message
   and by code, from "the console printed no JSON" when stdout was complete and
   not JSON. The notice names only facts the subprocess seam actually delivers
   (§4.4): the cap, the `lossy` fact, and the retained-tail length. It must not
   claim a whole-stream byte total, which the seam does not expose.
2. When stdout was complete and is not JSON, the existing
   `the console printed no JSON: …` message SHALL be preserved word for word and
   SHALL still be used for the different cause of a child's own output ahead of
   the envelope (RES-010 / issue 95), which this spec does not change.
3. `[R-5.3]` The verdict is made from the subprocess provider's own truncation
   fact — `SubprocessOutputRead.lossy` on a `readFrom(0)` after settlement
   (SD-10), the same `readFrom(0)` read of stdout that `console.ts:249` does today — and is never
   inferred from the text (e.g. "looks cut off", "unbalanced JSON").
4. `[R-5.4]` The panel renders a truncation failure and a non-JSON failure as
   different reader-facing lines, without collapsing either into the other; both
   remain visible in the activity section of a watched run. At the boundary, a
   truncation maps to `refactory/console-truncated` and a non-JSON failure keeps
   `refactory/console-failed` (§4.4).
5. A truncation failure SHALL NOT be reported as the run being a directory that
   does not exist or as an unknown run kind; the notice names the size problem.

**R-6 — Compatibility in both directions.** Trace: INT-001.
1. A current panel against a console that does not know `--activity-after`
   or `--activity-limit` (`unrecognized arguments`) SHALL fall back to one read
   without those flags and SHALL keep reading the run. The fallback SHALL be
   attempted only for that exact refusal, at most once per call, and SHALL be
   surfaced on the result (§4.4) rather than swallowed.
2. A current console against an older caller SHALL answer without the new flags
   exactly as it does today (top-level fields unchanged; `activity` gains
   members and, subject to §4.2's conditional, removes none).
3. A caller SHALL NOT crash, render nothing, or lose the run summary when
   `activity.after`, `activity.total`, `activity.has_more` or
   `activity.page_size` are absent; absent means "this console does not page",
   and the panel SHALL present the whole `activity.events` it received and mark
   the history as unpaged.
4. `activity.events` keeps being a list of journal records in the same shape;
   old records keep their `usage.calls` arrays and remain renderable.
5. A missing, unreadable or **identity-mismatched** activity index (§4.6) SHALL
   NOT fail the read — `[R-6.5.1]`. The console falls back to a full scan, serves
   a correct answer, and rebuilds the index (SD-3). A wrong answer is never
   preferable to a slow one.
6. `[R-6.5.2]` (§4.6) — an index write is atomic per line and a torn line is
   truncated and rebuilt, never guessed. This is what lets several reader
   processes build the index at once on a run that three writer processes are
   appending to.

---

## 3. Architecture and component boundaries

Five components carry the contract; each keeps its existing ownership.

```
  run directory                      console (Python)              service (TS Host)        panel (TS client)
  ┌────────────────────────────────┐   argv   ┌────────────────────────┐  JSON on stdout  ┌──────────────────┐
  │ .refactory/events.jsonl         │◄───────►│ console.status()       │─────────────────►│ Console.run()    │
  │  (journal, append-only)         │   page   │  live.events(after,p)  │ bounded by page  │ RefactoryConsole │
  │ .refactory/events.index.jsonl   │◄index───│  live.overview()       │                  │  .status()       │
  │  (strided offset index, derived)│  only    │  live.page() (1 read)  │                  │  .watchStream()  │
  │ progress.jsonl                  │          └────────────────────────┘                  └──────────────────┘
  │ .refactory/workers/*.json       │   ▲ appended by                                        ▲ (Remote, stream)
  └────────────────────────────────┘   │ worker_observer.mjs (bounded usage)         store.ts holds history+level
```

- **Journal owner** (`refactory_dsh/live.py`): appends records (`emit`), reads
  them by index (`events`), and composes the activity summary. It owns the
  "complete record" acknowledgement rule, and (SD-3) the offset index and the
  single-pass page read. New: `page(root, after, limit)` returning the records,
  the total and `last_event_at` from **one** pass, using the index to seek.
- **Console** (`refactory_dsh/console.py`): the verb surface and the JSON
  envelope. `status()` gains the activity page, keeps its existing top-level
  fields, and stops embedding an unbounded list. CLI `status --follow` advances
  its own cursors between frames (SD-6).
- **Boundary** (`packages/refactory/tool-refactory/src/console.ts`): argv
  building, the collection cap, and the decode verdict. It owns R-5's
  distinction and R-6.1's one-shot fallback.
- **Service** (`packages/refactory/refactory-console/src/index.ts`): maps the
  console answer to `RefactoryRunStatus` / `RefactoryActivity`; `status()` and
  `watchStream()` forward the activity cursor; `watchStream` advances its own
  cursor across polls (SD-5).
- **Panel** (`packages/client/ui-refactory/src/client`): `face.ts` reads and
  accumulates pages, `store.ts` holds the history (uncapped) and the level,
  `RunActivity.tsx` renders by level through a `RENDER_WINDOW` virtual list
  (SD-12) — a *rendering* bound, never a holding bound — `RefactoryBody.tsx` hosts
  the level control and the paging gesture, `locales.ts` carries the copy in both
  dictionaries.

What does **not** change: the console's other verbs, the loop, the spine, the
ledger, the graph control receipts, the run directory layout minus the three
derived files of §4.6 (the strided index, its one-line provenance anchor and its
one-line overview — the only files this spec adds, required by R-1.7b), and the
`RefactoryConsoleRemote` method signatures (they already carry `after` for
`status`/`watchStream`; see §6).

---

## 4. Data model, I/O contracts, validation, state ownership

### 4.1 Journal (persisted, append-only, unchanged format)

One JSON object per line. Records today carry `schema_version`, `run_id`,
`invocation`, `at`, `kind`, plus kind-specific fields (`node_id`, `state`,
`summary`, `route`, `usage`, …). This spec adds and removes nothing from the
on-disk record format. A journal index is a **line index among complete
records** — the same unit the existing cursor already uses
(`refactory_dsh/live.py:482-502`), including its rule that a trailing line with no
newline counts as an incomplete record (not acknowledged, not served), and that a
malformed *complete* line is served as a `journal_gap` record and counted.

### 4.2 Activity envelope (`activity` member of the `status` answer)

Non-normative note: this is the shape the console prints. Field-by-field
comments state which requirement each belongs to.

```jsonc
"activity": {
  "schema_version": 1,
  "run_id": "/abs/run",              // R-1.5  every page
  "invocation": "inv-1",             // R-1.5
  "after": 400,                      // R-1.3  index this page starts at
  "cursor": 600,                     // R-1.3  index of the next record
  "total": 269900,                   // R-1.3  complete records in the journal
  "page_size": 200,                  // R-1.2  records this answer may carry
  "has_more": true,                  // R-1.4  cursor < total
  "last_event_at": "2026-10-26T15:04:05+00:00",  // R-1.6
  "started_at": "2026-10-26T09:00:00+00:00",
  "nodes": ["build"], "completed": ["prepare"], "running_nodes": [{ "id": "build", "kind": "Worker" }],
  "graph": { "nodes": [], "edges": [], "skipped": [] },
  "wait_reason": "step build is being retried",
  "workers": [ /* summary rows, §4.3 */ ],
  "deliveries": [ /* unchanged */ ],
  "outputs": { /* unchanged shape; NOT bounded by page_size — see E-1 */ },
  "resources": null, "control": null,
  "events": [ /* at most page_size raw journal records, journal order */ ]
}
```

Members `console.status` adds to `activity` keep their meaning and are present
on every page: `last_failure`, `wait_reason` override, `design_review_status`,
`command_corrections`, `last_correction`, `queued_controls`, `respawn_count`,
`last_respawn`, `suggested_command`, `start_failure`.

**`outputs`: kept, not removed (E-1) — and the one member that breaks R-1.7d's
spirit.** Today (HEAD `live.py:154`; worktree `live.py:621`) returns
`read(root / 'facts.json')` as
`activity.outputs`, and `RunActivity.tsx:40` renders it as "Outputs and
validation". `outputs` is a **required** member of the public client type
(`packages/refactory/refactory-console/src/types.ts:207`,
`readonly outputs: Readonly<Record<string, unknown>>`, no `?`), so removing it
breaks any implementer outside this checkout at compile time, and dropping the
disclosure drops a view the panel has at every level above `compact`. This spec
therefore **does not remove it**, and changes nothing about it: §5's level mapping
treats "outputs" as standard-and-verbose content, the same as the other
summary facts.

Three facts stated plainly, because E-1 is an open human question and a builder
must not "fix" this by accident:

- `outputs` is **not** bounded by `page_size`. It is `facts.json`, whose size is
  the run's, so it is the one member of the answer that can exceed R-1.7d's
  "bounded by `page_size` records, not by `N`". R-1.7d therefore bounds
  `activity.events`; `outputs` is exempt and named as exempt.
- It is paid on **every page**, so an exhaustive `verbose` pass of
  `N/page_size` pages pays `facts.json`'s size `N/page_size` times. On the
  269,900-record run (UC-2) that is 1,350 repetitions of whatever `facts.json`
  holds.
- It **duplicates** what the panel already has from `facts.json` itself: there is
  no `facts` console verb (the verbs are `artifacts`, `candidate`, `comparison`,
  `docs`, `status`, …), so the same content arrives again through `status`'s
  `outputs` on every page, and the panel could read the file or the `artifacts`
  answer once per run instead.

Those three facts are the concrete reason a human might prefer Option B (§8 E-1:
make it optional and read it once) over the reversible default this spec keeps —
keep `outputs` untouched. The default stays taken and is stated normatively once,
in E-1's `[R-1.7d]` clause: do not remove, truncate, page or de-duplicate
`outputs`, and do not read `facts.json` a second time to compensate. E-1 remains
**open and human-owned**. `[R-1.6.1]` here repeats R-1.6's single-pass rule for
this call site.

**Single read for the page and the summary.** `events_page` and the overview
share one pass over the journal. `[R-1.6.1]` here: `console.status` must not parse the journal
once for the page and again for `total`/`last_event_at` (today's
`live.py:586-588`, HEAD `live.py:122-123`). §6 shows them as separate functions
for legibility; the
contract is that `status` performs **one** pass (or a seek plus one pass over the
unindexed suffix), not two.

Validation at the console boundary (before any read):

| Input | Rule | On violation |
|---|---|---|
| `--activity-after` | integer ≥ 0 | `ConsoleError`, exit 2 |
| `--activity-limit` | integer in 1…1000 | `ConsoleError` for `0`, negative or non-integer; clamp to 1000 above 1000 |
| `--after` | unchanged (integer, progress cursor) | unchanged |

### 4.3 Worker summary rows (R-3.3)

Each entry of `activity.workers` keeps the receipt's own fields but replaces a
`usage.calls` **array** with a count; a receipt that already carries a count is
passed through:

```jsonc
{ "worker_id": "w-1", "node_id": "build", "state": "done",
  "actual_route": { "provider": "deepseek", "model": "…" },
  "usage": { "status": "measured", "tokens": 410233, "tokens_delta": 812,
             "calls": 473 } }
```

A pre-existing receipt whose `usage` still holds `calls` as an array is
summarised the same way, so the wire shape never depends on the receipt's age.
Everything else in the receipt travels unchanged (the panel's full-receipt
disclosure).

### 4.4 Failure contract at the boundary (R-5)

Normative clauses of this section: `[R-5.3]` (the verdict and the `spillPath`
promise) and `[R-5.4]` (the message and the error code). Everything else here
is the measured basis for them.

Facts available at this call site, verified in the provider's typings
(`@deepseek-ai/dsh-subprocess`, `lib/types/types.d.ts`):

- `SubprocessOutputRead`: `text`, `nextOffset`, `lossy` (`true` when the
  requested offset slid out of the in-memory tail), and `spillPath?`.
- `spillPath` exists **only when `spill` is configured** ("absent disables
  spilling entirely", `SubprocessCollect`), and `console.ts:233-236` configures
  `stdout`/`stderr` as `{ maxBytes: collectBytes }` with no `spill`. So at this
  call site **spilling is off and `spillPath` is always absent**.
  `[R-5.3]` The console contract must not promise `spillPath`. Enabling spill is
  not proposed (SD-10).
- There is **no whole-stream byte count** anywhere in the seam. The only
  facts are "bytes were dropped" (`lossy`) and the length of the retained tail
  (`text.length`). A whole-stream total is therefore not knowable here and is
  not claimed.
- `CollectedOutput.truncated` is the after-settlement statement of the same
  fact: for a `readFrom(0)` after settlement, `lossy` **is** `truncated`.

`ConsoleResult` gains two members and keeps its shape otherwise:

```ts
export type ConsoleResult =
  | { ok: true; value: unknown; argv: string[] }
  | { ok: false; error: string; argv: string[]; value?: unknown
      /** Present when the collection cap, not the console, cut the payload. */
      truncation?: { readonly capBytes: number
                     /** Bytes actually retained in the in-memory tail. */
                     readonly retainedBytes: number }
      /** Present when the call succeeded only by dropping options this console refuses. */
      suppressed?: readonly string[] }
```

- `truncation` is set when `readFrom(0).lossy` is true **and** stdout does not
  decode to JSON. That is precisely the RES-002 case (a 4 MiB tail of a 36 MB
  payload). `capBytes` is `settings.collectBytes`; `retainedBytes` is
  `stdoutText.length`. There is no `observedBytes` and no `spillPath` member.
- The truncation message `[R-5.4]` reads (non-normative example of the shape):
  `the console's output was truncated at the 4194304-byte collection cap before it printed JSON (only 4194304 bytes were retained); raise collectBytes or page the read`
  followed by the argv on its own line, matching how other boundary failures
  already name the reproducing command.
- The non-JSON message stays `the console printed no JSON: <stdout tail or
  stderr>` and its `truncation` member stays absent. Note the deliberate edge:
  a **lossy** read whose retained tail *does* decode stays a success (T-12) —
  the payload is complete enough to use, and the contract does not invent a
  failure the seam does not prove.
- The service maps `truncation` to a `RemoteError` code
  `refactory/console-truncated` (added to `RemoteErrorDetailsMap` in
  `types.ts`) carrying `{ argv, capBytes, retainedBytes }`. Non-JSON keeps
  `refactory/console-failed`.

### 4.5 State ownership

| State | Where | Lifecycle |
|---|---|---|
| Journal records | `events.jsonl` | persisted, append-only, never rewritten (DEC-001); **replaced only as a whole** when a new invocation is prepared (§4.6) |
| Activity offset index | `events.index.jsonl` | persisted, derived, append-only, disposable (§4.6) |
| Index identity anchor | `events.index.provenance.json` | persisted, derived, disposable, replaced atomically per append batch (§4.6) |
| Index overview line | `events.index.overview.json` | persisted, derived, disposable, replaced atomically per append batch (§4.6); answers `total`/`last_event_at`, never a record |
| Page bounds (`after`, `page_size`, `total`) | console answer | transient, per call |
| Service `watchStream` activity cursor | service process | transient, per stream |
| Panel held history, held-through cursor | `store.ts` per tab | transient, per tab record; dropped on `watching(other)` and `forget` |
| Panel verbosity level | `store.ts` per tab | transient, per tab record; default `standard` at `start` |
| Panel paging in-flight/error | `store.ts` per tab | transient |
| Fallback/truncation notice | `store.ts` notice (level) and status frame (failure) | transient |

The level is deliberately not persisted across tab records: a fresh tab is a
fresh reader, and the default is required to be `standard`.

### 4.6 Activity index file (SD-3, disposable)

Normative obligations here are carried by **markers**, not by new numbered
clauses: this section adds no clause to §2.2 and restates only obligations that
§2.2 already numbers. A line marked `[R-x.y]` is that requirement's obligation and
belongs to R-x.y wherever R-x is read:

- `[R-1.6.1]` — the index must not change any observable answer (H-1).
- `[R-1.6.2]` — what the index may claim to cover, and when it must be rebuilt
  (H-2).
- `[R-1.6.3]` — how options are presented on the console command line.
- `[R-6.5.1]` — a missing/unreadable/unwritable index never fails a read.
- `[R-6.5.2]` — trustworthy writes: a torn line, and what a concurrent reader
  sees mid-write (H-3). **`[R-6.5.2]` is the marker for R-6.6** — the clause is
  R-6's sixth obligation (§2.2), and the two labels name the same obligation;
  citations anywhere in this document may use either, and neither introduces a
  clause that does not exist.
- `[R-1.7b]` — the overview line: the single lookup that answers `total` and
  `last_event_at` in `O(1)`. `[R-1.7b]` is **this spec's short marker** for clause
  `R-1.7`, sub-clause `(b)` of §2.2; the obligation is that clause's, restated
  here and adding none of its own. (The marker is a citation convenience, not a
  new requirement id — the way `[R-6.5.2]` marks R-6.6 above.)

Everything else in this section is format description and rationale.

Three derived files beside the journal, all disposable, all never read as history
(§1.2), all `R-6.5`-proof:

1. `<run>/.refactory/events.index.jsonl` — **one fixed-width line per stride of
   `STRIDE = 64` complete records**, in journal order. It answers "where does
   record `i` start?" and nothing else.
2. `<run>/.refactory/events.index.provenance.json` — one object: which journal
   the index covers (the identity anchor; it is the only thing that answers
   "which journal is this index for?", `[R-1.6.2]`), and how many index lines
   stand behind that claim (`index_lines`).
3. `<run>/.refactory/events.index.overview.json` — one object: the **newest
   counted record's** `total` and `last_at` (`[R-1.7b]`). This is the file that
   makes R-1.7b's `total`/`last_event_at` one lookup.

**Why two index layers and not one — `[R-1.7b]`.** The two lookups are
incompatible in one file. A page needs *position*: `record i` is at
`stride = i // 64`, a fixed width per line, so the page's line is a seek
(`[R-1.7a]`). Those lines cannot also carry the newest record's `i` and `at`:
each stride line holds each record's start offset and only the **first three
bytes** of each record's timestamp, and its own last line's `s` field names the
*first* record of the last stride (70 records → last line `s: 64`), so `total` is
**not** readable as "`i + 1` of the last line" and the newest `at` is not
recoverable from it at all. Conversely an overview line that carried every
offset would be the whole-history file R-1.7 exists to avoid. The overview line
is therefore a second, tiny, disposable file — the smallest thing that satisfies
R-1.7b's "one lookup, no walk, no re-parse".

The earlier revisions of this section said "two derived files … `total` is the
`i + 1` of the last trusted line" and `last_event_at` was "the `last_at` of the
last trusted line". Both statements became false when R-1.7a's page seek turned
the index into a stride: the last index line is no longer the last record. The
overview line (`[R-1.7b]`) is where those two values now live, and R-1.7b's
"satisfied by the last trusted index line" wording is corrected to name it
(Revision log, round 6, `M1-4`).

```jsonc
// <run>/.refactory/events.index.jsonl — one FIXED-WIDTH line per 64 records
{ "s":0, "g":[        0,   181,   362, …], "l":["202","202", …] }   ← 64 offsets, 64 at-prefixes
// <run>/.refactory/events.index.provenance.json
{ "schema_version": 1, "journal": ".refactory/events.jsonl", "run_id": "/abs/run",
  "st_dev": 16777220, "st_ino": 88193277, "size_at_write": 197,
  "first_record_sha256": "9f2c…", "first_record_end": 181, "index_lines": 5 }
// <run>/.refactory/events.index.overview.json
{ "schema_version": 1, "index_lines": 5, "total": 300,
  "last_at": "2026-10-26T09:00:01+00:00" }
```

- `i` is the complete-record index — R-1.3's unit, i.e. **exactly the set of
  records `live.events` serves**: every newline-terminated, non-blank line,
  including a malformed complete line, which `live.py:86-99` turns into a
  `journal_gap` record and **counts** (a JSON line that is not an object — a
  list, a number — becomes a gap the same way, `live.py:92-99`). Blank lines are
  skipped and not counted (`live.py:498-500`), so they get no slot in a stride.
  Each record's `start` offset is in `events.jsonl`; a stride's `g` holds the
  `start` of each of its (≤ 64) records and `l` the first three bytes of each
  record's `at`, or `nul` for a record whose `at` is not a string.
- A trusted stride line must **parse**, be at most `_STRIDE_LINE` bytes wide
  (a short final stride is legal — the width check is `live.py:180`, and the
  comment above it at `:178-179` says why), and carry equal-length
  `g`/`l` lists whose offsets strictly increase with each other and with `s`
  (`live.py:180-194`). Exact total width is enforced on the **file**, not the
  line: the index's size must be exactly `index_lines × _STRIDE_LINE`
  (`live.py:257`). A line that does not parse or whose offsets fail to advance,
  or a file of the wrong size, is not trusted and forces a rebuild
  (`[R-1.6.2]`).
- `counted` in earlier revisions is subsumed: every slot this version writes is a
  counted record, and a stride line that does not parse or does not advance is
  uncovered (`[R-1.6.2]`).
- `index_lines` is how many stride lines stand behind the claim — carried in
  **both** the provenance anchor and the overview line, and required to agree
  (`live.py:224`). It is what catches an index cut short while a longer anchor
  still described it, and it lets the three files be checked against each other
  without reading the index (`live.py:229`).

**`total` in `O(1)` — `[R-1.7b]`.** `total` is the `total` field of the overview
line. This is sound because `total` counts exactly R-1.3's unit — the
complete-record count `live.events` would return — and the writer maintains it
incrementally as it parses only the journal suffix (§4.6 `[R-1.6.1]`); nothing
about `total` is inferred from an offset. The overview's `total` and the anchor's
`index_lines` are cross-checked (`index_lines == ceil(total / STRIDE)`,
`live.py:229`), so an overview line that disagrees with the index it claims is
not trusted.

**`last_event_at` (R-1.6) is one lookup, not a walk — `[R-1.6.1]`, `[R-1.7b]`.**
`last_event_at` is the overview line's `last_at`, i.e. **the `at` of the newest
counted record, or `null` when that record has none.** That is exactly today's
answer: HEAD's `live.snapshot` returns `all_events[-1].get('at')` — the shipped expression at `git show HEAD:refactory/refactory_dsh/live.py` (`snapshot`, :145 of that revision)
over the records `live.events` returns — a property of the newest counted record,
so it is `None` whenever that record is a `journal_gap` (or an object without
`at`) and non-`null` otherwise. There is **no backwards walk**: a walk to an
earlier non-null `last_at` (round 3's rule) would answer a *different* value from
today's in exactly the seven cases where the newest record has no `at` (trailing
malformed line, one or more trailing gaps, gap before an unterminated tail, a
JSON line that is not an object, a newest `{}`, a gap before an `at`-less
record — measured 7 of 15 with the walk against 15/15 with this rule, §10 T-25). No
reader may substitute a different timestamp whichever it finds more useful: the
gate is that the spec's answer equals today's, and both are the same single
lookup of the overview line.

**What the index may claim to cover — `[R-1.6.2]`.** The index is trusted only
when **all** of these hold; anything else is a **full rebuild** from the current
journal:

1. `provenance.json` exists and parses, and its `run_id` resolves to the run root
   being read and its `journal` to `.refactory/events.jsonl` at that root — and
   the `overview.json` of `[R-1.7b]` exists, parses, and its `index_lines` equals
   the anchor's (`live.py:206-214`, `:224`, `:229`). The two small files are
   trusted **together or not at all**: one without the other describes nothing.
   Both are read in `_indexed()`, which decides only *shape*; `[R-1.6.2]`'s real
   checks — the index's exact size (`live.py:257`) and the first-record digest
   (`live.py:261`) — run in `_trusted()`, which nothing bypasses.
2. **Same file:** the current journal's `(st_dev, st_ino)` equal `st_dev`/`st_ino`
   in `provenance.json`. (Where `stat` cannot provide a stable inode — a
   filesystem without them, or a platform where the value is meaningless — this
   check is replaced by check 5's digest, and the provenance records
   `st_ino: null`; it is never skipped silently.)
3. **Not shrunk:** `size_at_write <= <current journal byte size>`. A journal
   smaller than what the index covered was replaced or truncated, not appended.
4. **Prefix still ours:** `size_at_write <=` the current journal size (a journal
   that only grew leaves it below the size), and the overview's `total` is
   consistent with the index it points at (`index_lines == ceil(total / 64)`,
   `live.py:229`). Note precisely what is *not* checked here: the code does **not**
   recount the journal to bound `total` from above, so a forged overview that is
   internally consistent and rounded up to a whole stride count is served — see
   "what checks 1–5 do and do not prove" below. The index file's own size must
   also equal exactly `index_lines * _STRIDE_LINE` bytes — no short or over-long
   last line (`live.py:257`).
5. **Re-read of record 0 matches:** the SHA-256 of the first complete record
   (`[0, first_record_end)` in the current journal) equals `first_record_sha256`,
   and that first record's `end` equals `first_record_end`. This is the anchor
   that catches a replacement at the *same* path whose new journal is already at
   least as long as the old one.

**What checks 1–5 do and do not prove (`[R-1.6.2]`, measured).** They prove the
three files describe *this* journal: a replaced journal, a truncated one, a torn
or short index and a disagreeing `index_lines` are all caught and rebuilt.
They do **not** cryptographically authenticate the overview line's `last_at`
against the journal — the digest covers record 0 only, so an overview line that
is well-formed, agrees on `index_lines` and `total`, but carries a *wrong*
`last_at` is served as-is. Measured: overwriting `last_at` with
`1999-01-01T00:00:00+00:00` on a 400-record journal leaves `total == 400` and
returns that wrong `last_at`, where the journal answers
`2026-10-26T09:00:39+00:00`. This is a deliberate, bounded trade: the overview is
a **derived cache of one fact**, the spec's premise (R-6.5, §1.2) is that a
derived file may always be deleted and rebuilt, and verifying `last_at` per call
would require the trailing-journal read R-1.7b exists to remove. It is
**corruption of a derived file**, not anything the normal write path produces —
every write of the overview line is a temp-then-rename carrying the `total`/
`last_at` of the suffix the writer just indexed (`live.py:327-341`) — and deleting
the file restores the correct answer on the next call. A stale overview line
written by a *previous* writer is the case checks 1–5 do catch: the anchor it must
agree with moves in lockstep, and any mismatch in `index_lines`, `total`,
`size_at_write` or the record-0 digest forces the rebuild.

A rebuild parses the current journal once, writes a fresh strided index, a fresh
provenance anchor and a fresh overview line, and **must answer exactly what a
full scan answers** — a rebuild is never a way to change an answer. Why checks
1–5 are not optional: a new
invocation **replaces the journal at the same path**. `invocations.prepare`
puts `.refactory/events.jsonl` in its archive list of runtime metadata
(`invocations.py:154-161`, `:156`) and moves each target with
`(workdir / relative).rename(archive / 'artifacts' / relative)` (`:206-209`),
installing nothing that recreates it; `run_preparation.py:35-36` documents the
ordering ("archives the old invocation … and installs the result"). `create`,
`design`, `edit`, `interactive` and a queued `graph-control resume`/recovery all
reach it (`console.py:378-381` → `interactive.prepare` → `run_preparation.prepare_run`
→ `invocations.prepare`; `interactive.py:104-105` → `recovery.prepare` for a
resume, and `recovery.prepare` re-uses the queued request and starts the
supervisor, `recovery.py:222-224`). That is a direct reading of
`invocations.py:156` and `:206-209`. The index's
`(st_dev, st_ino, size_at_write, first_record_sha256)` anchor is what makes
"which journal this index describes" a fact rather than a guess. Measured on
revision 3's rule (contiguous valid prefix only): 3 records + `t2`, then archival,
then one fresh record — the stale index answered `total: 3, last_event_at: t2`
where today answers `total: 0, last_event_at: null` after archival and `total: 1,
last_event_at: <new>` after the fresh record. Checks 1–5 answer today's value in
both moments (§10 T-24).

**`total`/`last_event_at` cost — `[R-1.6.1]`, `[R-1.7b]`.** Both are
the overview line's own `total` and `last_at`, so both are `O(1)` per call: no
journal read, no index read at all, no walk. A warm call reads exactly two small
files — the provenance anchor and the overview line — and stops; it never opens
`events.index.jsonl`, and its cost does not depend on `N` or on where the newest
record sits (§7.2 SD-3's measurement). An index that survives checks 1–5 is
extended by parsing only the journal suffix `[size_at_write, end)` in one pass
and rewriting the two small files; a journal whose size and identity are
unchanged costs **zero** journal reads.

**Writes are atomic per line — `[R-6.5.2]` (R-6.6, H-3); the two small files' replace rule is also part of `[R-1.6.2]` and `[R-1.7b]`.** The journal has three
concurrent appenders that run at the same time on a live run: the spine's own
process (`live.emit`, `live.py:67-83`, called from `nn/executor.py:1006-1014`,
`budgets.py:328`, `supervisor.py:471`, `graph_control.py:162-675`), the worker
process (`workers.event`, `workers.py:88`, from `begin`/`finish`), and each
worker's DSH observer (`worker_observer.mjs:30`, appending to the journal of the
run it is told about via `REFACTORY_WORKER_RECEIPT`, `workers.py:341-344`). More
than one reader process can build an index at once, because the service spawns
one console child per `status` (`SD-11`). The rules:

- The journal's own appends stay append-only (`O_APPEND`, boundary repair at
  `live.py:76-83`); this spec does not change how records are written.
- An index write is **whole stride lines appended to a length the anchor
  describes**: the writer opens `events.index.jsonl` `O_WRONLY|O_CREAT`
  (`live.py:332`), `ftruncate`s it to `kept × _STRIDE_LINE` where `kept` is the
  number of lines this coverage keeps (`live.py:333`), seeks there and writes the
  new stride lines in **one `write()`** (`live.py:334-335`). There is deliberately
  no `O_APPEND` on the index — the only `O_APPEND` in `live.py` is the journal's
  (`live.py:76`) — because the anchor's `index_lines` is the authority for how
  long the index must be, so cutting a loser's torn tail is part of the write.
  An update of `provenance.json` **or** `overview.json` is
  **write-to-temp-then-rename** in the same directory (so a reader sees either the
  old or the new version, never a partial one). The two small files are published
  **overview first, anchor second** (`live.py:338`), so an anchor never claims a
  coverage whose overview line has not been written yet.
- A reader accepts only stride lines that are exactly `_STRIDE_LINE` bytes,
  parse, and carry `g`/`l` lists of equal length whose offsets strictly advance
  from `s`; the anchor's `index_lines` is how many lines must be there, and the
  file's size must be exactly that many lines. A **torn or partial final line is
  truncated at the last complete line and rebuilt from the journal** — never
  guessed, never half-trusted. Because a rebuild answers exactly what a full scan
  answers (`[R-1.6.2]`), two builders racing produce the same answers whether or
  not their appends interleave: the loser's partial line is truncated by the next
  reader and its content is re-derived.
- What a concurrent reader sees mid-write: the journal may contain a record the
  index has not covered yet, so a call rebuilds from `size_at_write`; it may
  contain a half-written **final** line, so that record is neither counted nor
  served (R-1.3); it never sees a half-written index line counted.
- **An incomplete line is decided by its position, not by its content alone.**
  While the incomplete line is the journal's **final** line it is the write in
  flight and it is **dropped from `total`** (not counted, not served — R-1.3).
  Once any later byte is appended the torn line is no longer final: the next
  writer supplies the missing newline, so the once-incomplete bytes are terminated
  and surfaced as **one** `journal_gap` record, which **is counted in `total`**
  (R-1.3's counted-record rule). What that costs depends on *who* appends next,
  because only `live.emit` repairs the boundary (`live.py:76-83`, separator at
  `live.py:79`); `workers.event` (`workers.py:88-89`, opened `'a'`, one `write()`)
  and each worker's observer (`worker_observer.mjs:35`, `appendFileSync`) append
  their own terminated line without inspecting the previous byte:
  - **After `live.emit`** the torn bytes are terminated *first* (`live.py:79`),
    so they become one `journal_gap` and the new record is served as its own
    record — **two** records, nothing lost.
  - **After a non-repairing append** the torn bytes and the new record's line are
    **fused into a single malformed line**: they surface as one `journal_gap`, and
    **that later record is lost with it** — its content never appears as a record
    of its own. Measured, `a\n` + `{"tor` + a full `c` line appended without
    repair → `total == 2`, records `['a', 'journal_gap']`, raw
    `…\n{"tor{"schema_version": 1, "kind": "c", …}\n`.
  This is today's behaviour and is what R-1.3 already describes: R-1.3 governs
  how the journal's bytes are **counted and served**, and it says nothing about a
  record surviving a torn predecessor. It is not a requirement statement and adds
  none: a torn write is an interrupted write, and the spec neither guarantees nor
  requires recovery of the record that follows it.
  `events()`'s `break` on a line without a newline (`live.py:497-498`) therefore
  fires only on the journal's true final line; an unterminated line in the middle
  is impossible to observe, because a later byte terminates it.
  An incomplete line is thus **not** unconditionally dropped from `total`; what
  is dropped is only the *final* one. Measured for `live.emit`:
  `before` + `{"partial":` + `live.emit('after')` → records
  `['before', 'journal_gap', 'after']`, `cursor == 3`
  (`test_append_after_interrupted_record_exposes_gap_and_new_activity`,
  `tests/test_live_activity.py:61-69`). A blank line is skipped and uncounted in
  either position (`live.py:498-500`).

**Disposability and recovery.**
- All three files are **derived**, never authoritative. None is part of the
  activity history (DEC-001 is satisfied by the journal; deleting all three loses
  nothing — measured: with the overview line deleted, corrupt, or empty, and with
  all three deleted, `summary` and `page` answer **identically** and the next call
  rebuilds all three, §10 T-24). Only `events.index.jsonl` is append-only; the
  two one-line files are replaced whole. A rewrite of the *journal* would be the
  compaction DEC-001 forbids, and is not done (§1.2) — a rebuild rewrites only
  the derived files.
- The overview line is **never read as history**: it is read in exactly one place
  (parsed in `_indexed()`, `live.py:208` — the cheap first stage, **before** the
  size and digest checks `_trusted()` adds at `live.py:257`/`:261`) and no record
  is ever served
  from it. A record's content comes only from `events.jsonl`.
- The index's size is about **16.5 bytes per record** — one stride line is
  `_STRIDE_LINE` = **1,053 bytes** for 64 records, measured — i.e. **~4.4 MB at
  269,900 records**, and the two one-line files are ~200 bytes each. (Revision 5
  said "60–140 bytes per record … ~16–38 MB", which described the pre-stride
  one-line-per-record format and re-creates most of the journal it is meant to
  replace.) The strided layout is what makes the index small *and* seekable; the
  overview line is what makes `total`/`last_event_at` free.
- Recovery — `[R-1.6.2]`: checks 1–5 above are the whole rule; a failure of any
  of them means the trusted prefix is the empty prefix and the current journal is
  indexed from record 0. A missing, unreadable, corrupt, truncated or
  identity-mismatched **any one of the three** (including a present anchor beside
  a missing overview, or the reverse) **degrades a call to a full scan that still
  answers correctly and is rebuilt** — `[R-6.5.1]` (R-6.5).
- A reader that cannot write (a read-only run directory) still answers
  correctly, slowly — `[R-6.5.1]` (R-6.5); none of the three files is a
  precondition, and the files are an optimisation only.

---

## 5. Flows

**F-1 Attach to a run.** `watch(tab, run)` → `face.watchRun` resets both cursors
to 0, seeds the level at `standard` if unset, reads status page 0 (default 200
records) and the ledger, then opens `watchStream(run, 0)`. Expected: the panel
shows the summary, the standard window of the most recent records (SD-2), and a
way to reach the rest. Attach stays bounded: `activity.events` is one page even
when the journal has 269,900 records.

**F-2 Live poll.** `watchStream` reads status with its own advanced cursor every
1 s (3 s once terminal) → the service advances `activity.after` to the frame's
`activity.cursor` → the panel appends the page's events to the held window and
updates the summary. Expected: no poll ever re-reads the journal from index 0,
the per-poll cost does not grow with the journal (R-1.7c), and the visible tail
keeps growing.

**F-3 Manual refresh.** The refresh gesture passes the highest held
`activity.cursor` (R-4.2). Expected: the same continuity and the same per-call
cost as F-2, on demand.

**F-4 Choose verbose.** Level → `verbose` → the panel fetches pages
`from = held cursor` until an answer reports `has_more: false`, with the paging
progress shown. Expected, in this order:
1. **Every record is reachable.** When the last page arrives the held history
   (R-2's last row) spans index `0` … `total - 1`; the `has_more: false` answer
   and its `total` are the evidence, and the paging control reaches index 0.
2. **The DOM stays bounded.** The panel renders a `RENDER_WINDOW = 200`-row
   virtual window over the held array (SD-12), so the number of `<li>` rows is
   `≤ RENDER_WINDOW + 2` for any `N` (T-18b). 269,900 records are reachable by
   scrolling; 269,900 `<li>` nodes never exist at once.
3. **The panel stays interactive** (R-2's clause 5): each arriving page appends
   to the store and re-renders the window, not the history.
4. **The pass costs `O(N)`** (R-1.7a) and each page independently pays
   `O(workers + deliveries)` for the summary members (R-1.7e); `N` is what the
   pass scales with, not the number of pages' re-scans. Measured: 200-record
   pages over the 20,000-record / 22.1 MB fixture of §7.2 SD-3's recipe, record
   ≈ 1,107 B — `live.events` per page ≈ 51–69 ms at `after` 0 / 10,000 / 19,800,
   100 pages in ~5.6–10.7 s (§7.2 SD-3 has the recipe and an independent review's
   corroboration).

**F-5 Choose compact, then open the history.** Level → `compact` hides the
per-worker detail, the fulfilments/control rows and the held records; the
history disclosure and its paging control remain, and opening it renders the
standard window with "Load earlier activity" available (R-2.4).

**F-6 Load earlier activity.** Any level → a request for the previous
`page_size` records ending at the oldest held index → the console answers with
`after = max(0, oldest - page_size)`, `cursor = oldest`; the panel prepends.
Expected: repeated presses walk to index 0 and then disable the control, each
press costing `O(page_size)`. A host that serves every caller from one process
(SD-11) does not need the backwards step to be logarithmic; a fresh-process CLI
call per press does, because a descending `after` cannot use the advancing seek
(R-1.7a); the CLI's own backwards gestures page in ascending order instead.

**F-7 Truncated payload.** Console stdout exceeds the cap and does not decode →
`Console.run` returns `ok: false` with `truncation` → the service raises
`refactory/console-truncated` → the status read fails with the size-shaped
notice; the panel names the cap and the retained-tail size. Expected: no
"printed no JSON", no "Run type is not recognized yet" caused by it (RES-006's
knock-on is removed because the run kind is not lost to a decode failure).

**F-8 Older console.** The console refuses `--activity-after` → the boundary
retries once without it, sets `suppressed: ['activity-after']` → the service
marks `activity.paged = false` → the panel shows the whole `activity.events`
with an "unpaged history" note (R-6.3). Expected: the run is readable; the
person learns which side is older.

**F-9 Subsequent new run.** Worker observer `save()` appends bounded usage →
journal growth is linear in requests, not quadratic (R-3). Expected: a run of
473 requests no longer writes ~112k repeated call entries (RES-003's arithmetic).

**F-11 Index cold start.** (Renamed from round 2's mis-numbered "F-10", which
collided with review finding F-10; see the Revision log.) First `status` on a run
with no index → one pass over the journal builds all three derived files and serves
page 0 from it. Expected: the first call costs `O(N)` once (the same as today),
every later call costs `O(page_size)`; deleting the index changes nothing but
that first call's cost (R-6.5). **Archived journal, warm-looking index (H-2):**
the same run is prepared again (`create`/`design`/`edit`/`interactive`, or a
queued resume), so `.refactory/events.jsonl` is moved into
`.refactory/invocations/<id>/artifacts/` and the run has no journal, then a new
invocation writes one → every answer equals what a full scan of the *current*
journal says: `total: 0, last_event_at: null` while the journal is absent, and
`total`/`last_event_at` of the fresh journal once it exists. The stale index
survives in place with no error, is recognised as describing a different journal
by §4.6 `[R-1.6.2]`'s identity check, and is rebuilt. Expected: no answer ever
mixes an archived journal's offset with the new journal's records; the rebuilt
index answers exactly what the full scan answered (T-24).

**F-12 Gap records at the tail of the journal.** A journal whose newest complete
record is a `journal_gap` (a malformed complete line, or a JSON line that is not
an object — `live.py:86-99`) → `total` counts it, `has_more`/`cursor` behave as
for any record, `last_event_at` is **`null`** (the newest counted record has no
`at`, and that is today's answer too (HEAD `snapshot`, `git show HEAD:refactory/refactory_dsh/live.py` (`snapshot`, :145 of that revision) — §4.6 `[R-1.6.1]`), and
`activity.events` serves the gap record itself (with `kind: journal_gap`) rather
than dropping it. Expected: `last_event_at` is exactly what today's
`snapshot()` returns for the same journal — `null` here, and the newest record's
`at` when the newest record is ordinary (`T-25`'s matrix) — no
`total`/`has_more` disagreement, and no journal open just to find the timestamp
(R-1.3, R-1.6, R-1.7b). A gap *in the middle* is what does not change
`last_event_at`; only the newest record decides it.

---

## 6. API and interface sketch (non-normative: names, not new obligations)

Console argv (unchanged positional `--run`):

```
python -m refactory_dsh.console status --run <dir> [--after N] [--activity-after N] [--activity-limit N]
python -m refactory_dsh.console status --run <dir> --follow [--after N] [--activity-after N] [--activity-limit N]
```

`--after` is the progress-file cursor and is accepted with and without
`--follow` (as it is today, `console.py:1380`); `--activity-after` is the journal
cursor. They are distinct and are never derived from one another (R-4.3). On the
first `--follow` frame `--activity-after` seeds the loop; thereafter the loop
advances its own.

Python, in `live.py`:

```python
PAGE_DEFAULT, PAGE_MAX = 200, 1000  # R-1.2

def events_page(root: Path, *, after: int = 0, limit: int = PAGE_DEFAULT
                ) -> tuple[list[dict[str, Any]], int]:
    """Records[after:after+limit] and the journal's complete-record count."""

def overview(root: Path) -> dict[str, Any]:
    """The journal's summary facts: total, last_event_at. Served by the same
    single pass as events_page when called together (R-1.6, §4.2)."""

def page(root: Path, *, after: int = 0, limit: int = PAGE_DEFAULT) -> dict[str, Any]:
    """ONE pass: the page, total and last_event_at, seeking via events.index.jsonl
    (SD-3). total and last_event_at are the overview line's own fields (§4.6
    [R-1.7b]) — one small read, no walk, no index scan (§4.6 [R-1.6.1]). This is
    what snapshot() calls; events_page/overview remain the narrow primitives the
    tests use."""
```

`live.snapshot(root, *, after=0, limit=PAGE_DEFAULT)` keeps its name and return
members, adding `after`, `total`, `page_size`, `has_more`, and gaining the
single-pass overview. `console.status(run_root, *, after=0, activity_after=0,
activity_limit=PAGE_DEFAULT)` shares one pass between the page and the summary.

Service, `index.ts`:

```ts
@Remote async status(run: string, after: number, signal: AbortSignal): Promise<RefactoryRunStatus>
// passes options { run, after, 'activity-after': afterActivity, 'activity-limit': ACTIVITY_PAGE (SD-5) }

@Remote({ mode: 'stream' }) async *watchStream(run: string, after: number, signal): AsyncGenerator<RefactoryWatchFrame>
// own `activityAfter` starts at `after`; after each frame it is set to status.activity?.cursor ?? activityAfter
```

Client types, `types.ts`:

```ts
export interface RefactoryActivity {
  // …existing members, including `outputs` (kept — §4.2)…
  readonly after?: number            // R-1.3
  readonly total?: number            // R-1.3
  readonly page_size?: number        // R-1.2
  readonly has_more?: boolean        // R-1.4
  readonly paged?: boolean           // present and false for R-6.3's older console
}
```

Client store additions, `store.ts`:

```ts
/** The journalled history a tab holds for the run it is reading. */
interface HeldHistory {
  readonly run: string            // the run these records belong to (R-4.5)
  readonly events: readonly Readonly<Record<string, unknown>>[]   // oldest first, UNCAPPED (R-2.2)
  readonly through: number        // highest journal cursor held (= events[0]..events.at(-1) coverage)
  readonly total: number          // last reported journal total
  readonly complete: boolean      // an answer with has_more:false has been seen for this run (R-2.2)
  readonly paged: boolean         // false when the console does not page (R-6.3)
  readonly reading: boolean       // a page is in flight
  readonly error: string | null   // why the last page failed
}

/** How many held records are in the DOM at once. Rendering, not holding, is bounded (R-2.2, SD-12). */
const RENDER_WINDOW = 200
// HeldHistory is deliberately uncapped: capping it would make verbose a "last N"
// (R-2.3). What is capped is the rendered slice:
//   rows = activity.events.slice(scrollTopRow, scrollTopRow + RENDER_WINDOW)
// A virtual list over `events` with row height fixed; scroll position is a row
// index in the store so a poll or a level switch does not lose it.
// tab state gains: `activity: HeldHistory | null`, `activityLevel: 'compact'|'standard'|'verbose'`,
//                  `activityScroll: number` (row index of the first rendered record)
// actions gain: activityLevel(tab, level), activityPage(tab, page: { from: number; events; cursor; total; paged }),
//              activityEarlier(tab, ...), activityPageLoading/Failed(tab, ...), activityScrolled(tab, row)
```

Face additions (`RefactoryInjected`): `setActivityLevel(tab, level)`,
`readActivity(tab, run, from, signal)`, and an `exhaustActivity(tab, run,
signal)` used by the verbose level. These are panel-local; the Remote namespace
gains no method (R-4 is plumbing, not a new endpoint).

---

## 7. Assumptions and agent decision log

### 7.1 Soft assumptions (from `.hai/state.yaml`; NOT human requirements)

| ASM | Stays in the design? | What changes if false |
|---|---|---|
| ASM-001 (writer + reader both need fixing; a read-side bound is required for existing large journals) | Yes — R-1 (read-side paging) and R-3 (writer bound) are separate and both present | If only a read-side bound is allowed, drop R-3.1-2; existing journals stay large but readable (R-1 still satisfies DEC-001). If a writer fix alone were considered sufficient, DEC-001 still fails for existing runs, so R-1 would have to stay. |
| ASM-002 (truncation is its own failure; the cap stays as defence in depth) | Yes — R-5, cap unchanged at `CONSOLE_DEFAULTS.collectBytes` | If the human wants the cap raised instead, R-5.1's message still applies but the number changes; SD-8's page size becomes the only practical bound, and `refactory/console-truncated` becomes rarer but not removable. |
| ASM-003 (a real journal cursor for authoring runs; fallback for an older console) | Yes — R-1.3, R-4.3, R-6.1 | If authoring runs are expected to write `progress.jsonl` instead, R-4.3's "must not assume the activity cursor moves" could be dropped, but the fallback (R-6.1) is still needed for older consoles. |
| ASM-004 (proposed mapping: compact = counts; standard = summary + recent; verbose = full paged) | Partly — R-2's table follows it except that compact keeps the paging control (DEC-001) and the mapping is fixed by this spec, not by the assumption | If the human wants a different mapping, only R-2's table changes; R-1, R-3, R-4, R-5 do not. |
| ASM-005 (include CLI `status --follow`'s full-journal-per-poll cost) | Yes, narrowly — SD-6 advances the cursors, which R-1.7c requires; printing is unchanged (§7.2 SD-6) | If the CLI is out of scope, drop SD-6's cursor advance; the panel is unaffected, but R-1.7c would then have to exclude `--follow`, which would leave the terminal paying `O(N)` per second. |

No ASM is presented here as a requirement. Every MUST/SHALL in §2 (and every
clause marker in §4.2, §4.4, §4.6 and E-1) traces to
INT-001/DEC-001/DEC-002 only. R-1.7 is not taken from ASM-001 (that assumption is
about writer vs. reader, not about cost); it is a consequence of DEC-001 plus
"watch", as argued in §2.1.

### 7.2 Agent decision log (SD-*)

- **SD-1 — Page unit is journal records, not lines or bytes.** Rationale: the
  existing cursor is already a complete-record index (`live.py:482-502`), so
  resuming by record is the only unit that keeps one cursor semantics for the
  panel, the CLI and the tool. Bytes would break the "an interrupted final write
  is not acknowledged" rule. Affects R-1.1-4, R-4, UC-1.
- **SD-2 — Standard shows the most recent 200 records (one default page).**
  Rationale: the frame must stay small enough for a 1 s poll and a 4 MiB cap
  (200 typical journal records ≈ tens of KB; RES-002 measured 19.8 MB compact
  and 36 MB indented for the whole journal), and one page is the unit the rest
  of the contract already speaks. It is a *window*, not the history: the history
  is one gesture away, which is what DEC-001 requires. Affects R-2, UC-3.
- **SD-3 — A derived byte-offset index beside the journal, plus one-pass page
  reads.** Rationale: measured on a 20,000-record journal in this worktree,
  `live.events(root, after)` costs **the same at every `after`** — it parses the
  whole file and slices afterwards — and `live.snapshot` does it twice, so a page
  that returns 200 records and a page that returns none cost the same, and the
  cost is linear in `N` per page. SD-3 is: on first read, build
  `events.index.jsonl` (§4.6) in one pass; on every later read, seek the strided
  index by `after`, parse only the page's records plus the unindexed suffix, and
  read `total`/`last_event_at` from the **overview line** (`events.index.overview.json`,
  §4.6 `[R-1.7b]`) — R-1.7b's one lookup (round 6 correction: revision 5 said "the
  index's last counted line", which a strided index cannot answer). The index is
  derived and disposable (R-6.5). Rejected alternatives: keeping
  `splitlines` for small journals (it is the `O(N)` walk itself); a
  `linecache`-style in-process cache (fails the fresh-process CLI and the
  service's separate host, and is unbounded memory); rewriting the journal to
  compact it (DEC-001 forbids it). Affects R-1.6-8, R-6.5, §4.6, UC-12.
  **Measurement (reproducible; magnitudes, not promises).** Fixture: `N` complete
  records of **~1,107 bytes** each (`summary` padded with 1,000 `x`, so the
  journal is **22.1 MB** at `N = 20,000`), written by the scratch script below,
  never inside the repo. These magnitudes are the recipe's own: round 3 stated
  24.1 MB / 1,207 B, which the same recipe does not produce (measured from the
  script's own lines: 22,148,890 bytes / 20,000 = 1,107.4 B — round 4, H-6).
  Outcome of that exact script in this worktree (`.venv/bin/python`, Apple
  silicon): `live.events(root, after)` → **69 / 54 / 51 ms** at `after` = 0 /
  10,000 / 19,800 for `N = 20,000`; `live.snapshot(root)` → **112 ms** (two reads
  of the same journal, ~2× one page read); an exhaustive pass fetching 200 records
  per call over the same journal → **100 calls in ~5.6–10.7 s** (round 4's run: 5.6 s, ~56 ms per call; the round 4 reviewer's own run of the same recipe: 10.65 s, ~106 ms per call), i.e.
  `N`-proportional per page and `N²` in total. (Round 3 quoted 75 / 61 / 57 ms,
  128 ms and 6.2 s for the same recipe; the difference is host load, not shape —
  both runs are flat in `after` and linear in `N`.) The independent review measured the same shape with its own
  fixture (7.49 MB flat at 52–53 ms, 5.24 MB flat at 27–31 ms, 54.8 MB / `N` ≈
  265k at ~367–421 ms, `snapshot` ≈ 2×), which is why revision 2's "80–190 ms"
  figure was not reproducible: it was one machine's number reported as a range
  for all sizes. The claim this decision rests on is the **shape** — flat in
  `after`, linear in `N`, two reads per snapshot — and it reproduces; the
  absolute seconds depend on record size and host, so no absolute second count is
  promised anywhere in this spec.
  **Built implementation, warm index (round 6, `M1-4`; the numbers R-1.7b's
  contract is judged on).** The recipe is the sketch below **exactly**
  (`schema_version`, `kind`, `i`, `at`, `summary` of 1,000 `x` → **1,104 B per
  record**, so the journal is **22,080,000 B ≈ 22.08 MB** at `N = 20,000`, and the
  measured 22,148,890 B **exceeds** it by 0.3 % because the `at` field widens with
  `i`); `live.page(after = N - 200, limit = 200)` and
  `live.summary()`, warm index, `.venv/bin/python`, on this host. Both the median
  and the mean of 60 calls are given, because the difference between them is the
  host's noise and neither is the claim:

  | `N` | journal | page median | page mean | `summary()` median | `summary()` mean | `total` == full-scan count |
  |---|---|---|---|---|---|---|
  | 500 | 0.553 MB | **0.629 ms** | 0.640 ms | **0.089 ms** | 0.091 ms | yes |
  | 5,000 | 5.534 MB | **0.612 ms** | 0.620 ms | **0.099 ms** | 0.100 ms | yes |
  | 20,000 | 22.149 MB | **0.618 ms** | 0.657 ms | **0.098 ms** | 0.099 ms | yes |

  The claim is the **shape**: page 0.61–0.63 ms and summary 0.09–0.10 ms across a
  40× range in `N`, and `total` equal to the full-scan count at all three — i.e.
  R-1.7a/b satisfied by measurement, not by argument. Absolute milliseconds are
  host-dependent and are quoted only as the magnitudes this decision rests on.
  The `N = 20,000` row is the **uncached** case for the index itself: a warm
  call reads zero journal bytes and the fixture holds no worker/delivery
  receipts, so its time is the two small-file reads and nothing else.
  (An earlier draft of this table said "median of 40", quoted 0.681 / 0.653 /
  0.674 ms and a 24.1 MB journal: the timings were a mean mislabelled as a
  median, and 24.1 MB was the figure H-6 had already retracted — the recipe gives
  22.08 MB, not 24.1 MB. Corrected in round 6 after independent review.)
  **Option B (fold the two values into the existing provenance anchor) rejected on
  this measurement.** Asked by round 6's change request, measured, and it loses on
  three counts — the first two about cost, the third fatal:
  1. *It cannot stay `O(1)` on the page path.* A page under a strided index
     appends a stride line whenever it indexes more records, and an anchor that
     must describe the newest appended stride has to be rewritten with it. 200
     records × **0.120 ms** per atomic write-to-temp-then-rename = **24 ms** for
     one page, against the **0.62 ms** page measured above — a single page would
     cost ~39× its current time, and the extra cost is `O(page_size)` in renames.
  2. *It cannot stay fresh on the append path.* Keeping the anchor current per
     **journal append** adds **0.120 ms** to `live.emit`'s own **0.059 ms/record**
     (median) — a **~2×** append cost (0.120 / 0.059 = 2.03), and **~2.4 s** of pure rename overhead
     across a 20,000-record exhaustive run. R-3.4 and R-6.6 ("this spec does not
     change how [journal] records are written", §4.6) are what that breaks: an
     anchor rewrite is a second, non-journal write per record.
  3. *It is not achievable without a journal read per warm call or a changed
     append contract — which is what settles it.* Three independent
     processes append to `events.jsonl` — `live.emit` (`live.py:67-83`),
     `workers.event` (`workers.py:78-89`, opened `'a'`, one `write()`) and the
     Node observer's `appendFileSync` (`worker_observer.mjs:35`, on a path derived
     from its own `REFACTORY_WORKER_RECEIPT`, never from `live.py`) — and only
     `live.emit` inspects the previous byte before writing (`live.py:79`). No
     in-process anchor rewrite can be synchronised with the other two appenders,
     so a stale anchor would be the **normal** state, and serving it would answer
     a wrong `total`/`last_event_at` — breaking R-1.6, not just its cost bound.
     A lazily rebuilt anchor *could* be made correct from the journal inside the
     trust gate, but that reintroduces a journal read on every warm call, which is
     the `O(N)` read R-1.7b exists to remove.

  The overview line therefore exists because it is the only shape in which
  `total`/`last_event_at` are both `O(1)` and *correct*: it is written by whoever
  last indexed a suffix (`live.py:338`), parsed beside the anchor by `_indexed()`
  (`live.py:208`, the cheap stage) and then subject to the size and digest checks
  of `_trusted()` (`live.py:257`, `:261`), and it is the one extra small file this
  amendment legalises.
  ```python
  # /tmp/measure.py — build N records of ~1,107 B (a 22.1 MB journal at
  # N = 20,000), then time events/snapshot
  import json, time
  from pathlib import Path
  from refactory_dsh import live
  root = Path('/tmp/run'); (root / '.refactory').mkdir(parents=True, exist_ok=True)
  with (root / '.refactory/events.jsonl').open('w') as f:
      for i in range(20000):
          f.write(json.dumps({'schema_version': 1, 'kind': 'worker_event', 'i': i,
                              'at': f'2026-10-26T09:{i//600:02d}:{i//10%60:02d}+00:00',
                              'summary': 'x' * 1000}) + '\n')
  for after in (0, 10000, 19800):
      t = time.perf_counter(); live.events(root, after); print(after, round((time.perf_counter()-t)*1000))
  t = time.perf_counter(); live.snapshot(root); print('snapshot', round((time.perf_counter()-t)*1000))
  ```
- **SD-12 — `verbose` holds every record and renders a fixed 200-row virtual
  window.** Rationale: DEC-001 requires the whole history to be *shown* and R-2
  requires verbose to make it *reachable*, but a `<ol>` with one `<li>` per
  record (`RunActivity.tsx:41-42`) at `N = 269,900` blocks the main thread and
  contradicts R-2's clause 5 ("the panel stays interactive") and F-4. The split is
  therefore explicit: **holding** is uncapped (`HeldHistory.events`, §6), because
  capping it would silently turn verbose into "last N" (R-2.3), and
  **rendering** is capped at `RENDER_WINDOW = 200` rows by a virtual list over
  the held array. 200 is chosen to equal `PAGE_DEFAULT`, so the window never
  exceeds one page and the "modify the window on a page arrival" path is the same
  path as a scroll. The DOM bound is testable without a browser frame budget
  (T-18b) and independent of `N`. Affects R-2.2, R-2.5, F-4, T-18b, UC-2, UC-3.
- **SD-4 — Keep `activity.outputs` unchanged; do not remove it.** Rationale:
  it is a required member of the public client type
  (`types.ts:207`), so removing it is a compile-time break for implementers
  outside this checkout, and its only renderer (`RunActivity.tsx:40`) is a view
  the panel shows at `standard` and `verbose`. It **is** `facts.json`
  (`live.py:621`), so it is not bounded by `page_size` and is carried on every
  page (§4.2) — a real cost, named rather than hidden, and the concrete reason
  E-1's Option B exists. It stays available independently from the run directory
  and the `artifacts` verb, but duplication alone is not a permission to break a
  public type. This is the **reversible default**: it costs nothing to revisit,
  while a removal costs every consumer. Affects R-1.5, R-1.7d, R-2, R-6.2, E-1,
  UC-4.
- **SD-5 — Service page size and cursor are the service's own choice.** The
  service passes an explicit `--activity-limit` (200) rather than relying on the
  console default, and `watchStream` owns its cursor just like the panel. This
  keeps the console's clamp (R-1.2) a safety net, not a policy owner. Affects
  R-4.2, R-6.2.
- **SD-6 — CLI `status --follow` advances its activity cursor; it prints the
  same fields it prints today.** Rationale: the cost it pays today is the whole
  journal per poll (RES-008), which R-1.7c forbids. The advance itself is small:
  the loop reads `frame['activity']['cursor']` and passes it as
  `activity_after` on the next frame, and the same for the progress cursor.
  The printed output does **not** change: each frame is still a full
  `status()` answer, and `frame['activity']['events']` on a follow frame is the
  newest records rather than the whole journal — which is the point of paging.
  The one rendering that reads `activity.events` for its text,
  `readable.format_status` (`readable.py:79-85`, filtering `run_warning`
  records), keeps working on a page: the page it renders is the newest one, and
  the run's durable warning list is separately available from
  `interactive.run_warnings`, which reads the journal itself. If a caller needs
  warnings that scrolled out of the current page, it reads the history
  (`--activity-after`) explicitly; the follow loop does not silently accumulate
  state to re-print them. Affects R-1.7c, UC-11.
- **SD-7 — Worker usage summary carries `calls` as a count.** Rationale: the
  panel renders `Usage: {JSON}` per worker; a count keeps the useful fact (how
  many requests) without the array, and works for both receipt ages (R-3.3).
  Token accounting keeps using `tokens` / `known_tokens` / `tokens_delta`, so
  nothing that sums tokens changes. Affects R-3, UC-6.
- **SD-8 — The levels are a panel-local concept; the console does not filter.**
  Rationale: filtering by level on the console would put three read contracts on
  the wire and let an old panel and a new console disagree about what "full
  history" means. The console serves records; the panel decides detail. Affects
  R-2, R-6.
- **SD-9 — No new Remote method.** Rationale: `status(run, after, signal)` and
  `watchStream(run, after, signal)` already carry a cursor. Overloading `after`
  as "both cursors" would break R-4.3, and adding a method would force a
  Remote-face change for a behaviour the existing method can express (the
  activity cursor and the page bound travel as console options). Affects R-4,
  R-6.
- **SD-10 — The truncation verdict is `lossy` on an after-settlement
  `readFrom(0)`, and nothing else; spill stays off.** Rationale: text
  heuristics ("does it look cut off?") would misclassify issue 95's child
  output, and the seam exposes no byte total to compare against. At
  `console.ts:233-236` `spill` is not configured, so `spillPath` is always
  absent and the whole-stream payload is unrecoverable from the collector —
  which is another reason the full journal must not travel in one payload
  (R-1.7d). Enabling `spill` is considered and **not** chosen here: it would
  write up to the whole stream to disk for every status call (a 36 MB spill per
  poll), i.e. it trades a memory cap for an I/O amplification of exactly the
  kind R-1.7 forbids. Affects R-5.1, R-5.3, §4.4.
- **SD-11 — The index is written by whichever process reads first; the service
  host benefits without changing shape.** Rationale: the service's `status` calls
  each spawn a console child, and each child seeks instead of scanning, so no
  long-lived process, daemon or cache is introduced. Round 3 justified concurrent
  building with "`pathlib` appends are made idempotent by a tolerant rebuild" and
  named no mechanism (H-3). The mechanism is now explicit and normative in §4.6
  `[R-6.5.2]`/R-6.6: one `O_APPEND` write per complete **fixed-width stride
  line**, write-to-temp-then-rename for the provenance anchor and the overview
  line (overview first), acceptance only of stride lines of the right width whose
  offsets strictly advance, and truncation of a torn final line followed by a
  rebuild from the journal — never a guess. Because a rebuild answers exactly
  what a full scan answers (`[R-1.6.2]`), two builders racing produce the same
  answers whichever of them wins each append; what they contend over is the
  derived file, never history. The journal itself already has three concurrent
  appenders (`live.emit` in the spine, `workers.event`, each worker's
  `worker_observer.mjs`) and its appends are unchanged. Affects R-1.7, R-6.5,
  R-6.6, §4.6.

---

## 8. Escalations

One escalation is human-owned and **open**; the rest are recorded so a reviewer
can see what was decided and what was deliberately not. Nothing here blocks the
build: every human-owned question has a reversible default already taken.

- **E-1 (OPEN — human-owned; question, not a decision) — remove
  `activity.outputs`?**
  - *Fact:* `outputs` is a required member of the public client type
    (`types.ts:207`, `readonly outputs: Readonly<Record<string, unknown>>`) and
    its only in-checkout renderer is `RunActivity.tsx:40`.
    It is `read(root / 'facts.json')` (`live.py:621`), so its size is the run's
    `facts.json`, **not** `page_size` — it is the one member of every page that
    R-1.7d cannot bound. Removing it is a compile-time break for implementers
    outside this checkout and deletes the "Outputs and validation" disclosure.
  - *Option A (default taken) — keep `outputs` untouched.* Cost: the status
    payload keeps carrying `facts.json`'s contents on every page, which is
    bounded by the `facts.json` file's size — unbounded by `page_size` — and is
    paid again on every page of the verbose pass (1,350 × `facts.json` on UC-2),
    and the same content is independently reachable from the run directory's
    `facts.json` and from the `artifacts` verb. No compatibility break; nothing
    to undo.
  - *Option B — make it optional (`readonly outputs?: …`) and keep the
    disclosure.* Cost: a smaller payload for hosts whose `facts.json` is large;
    every consumer must handle `undefined`. Compatible in the sense that a
    missing optional member does not fail compilation, but a consumer that
    dereferences it without a guard breaks at runtime — so it is a lesser
    break, not a free one.
  - *Why a human might prefer B (the concrete reason, stated):* A keeps the one
    violation of the payload budget this spec otherwise enforces, and multiplies
    it by the number of pages an exhaustive verbose read fetches; B removes that
    violation at the cost of an optional member and a single `facts.json` read
    where a host wants the data. This is the trade a human owns, because it
    trades a public-type guarantee for a size guarantee.
  - *Option C — remove it.* Not taken: it is the one option that cannot be
    shared with an older console's consumer, and its only benefit (a slightly
    smaller page) is not worth a public-type break for a feature whose point is
    to make more history reachable.
  - *In force until the human answers:* Option A, stated normatively as
    **`[R-1.7d]`** (the `outputs` exemption of R-1.7d) and **`[SD-4]`**: keep
    `outputs` present and unchanged on every page, do not page or truncate it, and
    do not add a second `facts.json` read to work around it (§4.2). No other
    sentence here obliges anything.
  - *If the human answers:* A and B need no other change (R-1…R-6 are
    unaffected); B additionally allows `status` to omit `outputs` when a host
    asks, and would need T-6's `outputs` assertion relaxed to "absent or
    unchanged"; C would delete the disclosure, drop the member from
    `types.ts`, and add a note to R-6.2's compatibility table.
- **E-2 (reversible design constraint; NOT human-owned) — the residual cost
  after this spec.** After R-1.7, the dominant residual is per-page cost on a
  host that serves every caller from a new process and cannot use the advancing
  seek (a descending `after`, F-6's "Load earlier activity"): each such press is
  one full-scan in the worst case. Mitigations, in order: the service (which the
  panel always goes through) keeps the index warm so its own per-poll cost is
  `O(page_size)`; the panel fetches backwards in ascending chunks where it can;
  SD-11 leaves room for a long-lived reader later without changing the wire.
  This is a performance constraint that can be reversed (drop the index and the
  per-call cost returns to today's behaviour, correctly but slowly), not a
  retention-policy question: no history is dropped to meet it. DEC-001 stands
  untouched.

Out of scope by observation, so the build does not collide with it: the CLI
`status` verb currently prints one JSON envelope that includes
`payload['text']` (`console.py:1840`), so its output is not strictly one JSON
value (a "child output ahead of the envelope" member of the issue-95 family).
Neither the service nor the panel reads `text`, so the envelope decoder is
unaffected; a two-lane bounded console read is separate work and is not required
by INT-001.

---

## 9. Use cases

**UC-1 (happy path) — a new, live authoring run.** Attach. The panel shows the
summary, the most recent 200 records, and the level control at `standard`.
Polls continue from the panel's cursor and the service's cursor; the visible
tail grows without re-reading the journal, and no poll gets slower as the
journal grows.

**UC-2 (boundary) — a run with 269,900 journal records and a 36 MB payload
(RES-002).** Attach succeeds within the collection cap; `activity.total` reads
269,900; `has_more` is true. Switching to `verbose` pages from 0 to the end in
1,350 pages of 200; when the last answer reports `has_more: false` the tab holds
all 269,900 records (`complete: true`), while at most `RENDER_WINDOW` = 200 of
them exist as `<li>` nodes at any moment (SD-12) — scrolling reaches the rest, and
the panel never blocks. The first page pays the one-time index build; the
remaining 1,349 pages cost `O(page_size)` in the journal each (R-1.7a) plus
`O(workers + deliveries)` for the summary members (R-1.7e), so the whole pass is
linear in the journal, not the quadratic walk it would be if every page
re-scanned.

**UC-3 (boundary) — level switching.** `standard` → `compact` hides worker
detail, fulfilments/control and held records but keeps the paging control;
`compact` → `verbose` fetches the whole history; `verbose` → `standard` drops
back to the window without losing the summary or the fetched-through cursor; in
every case the rendered row count is `≤ RENDER_WINDOW + 2`. No run state changes.
A fresh tab is always `standard`.

**UC-4 (boundary) — boundary of a page.** A journal of exactly 200 records
answers `cursor == total == 200`, `has_more: false`. A journal of 201 answers
`has_more: true` on page 0 and `has_more: false` on the page starting at 200.
`--activity-limit 5000` is clamped to 1000; `--activity-limit 0` and
`--activity-after -1` are refused with exit 2.

**UC-5 (boundary) — a journal growing under a pager.** A pager that reads
`total` = T and pages to `cursor == T` has seen every record present when the
first page was served; records appended later appear in later pages from the
cursor it reached, and the appended suffix is indexed incrementally rather than
forcing a full re-scan.

**UC-6 (happy path) — a new run's journal shape.** A worker that completes 473
requests writes records whose usage summary never repeats previous calls; the
journal length is linear in requests; `activity.workers[*].usage.calls` reads
473 and no array travels.

**UC-7 (failure) — payload larger than the collection cap.** The status read
fails with a message naming the cap and the retained-tail size, with `argv`, in
both the tool and the panel. It is not the "printed no JSON" message, and it
does not claim a whole-stream byte total the seam cannot observe.

**UC-8 (failure) — complete stdout that is not JSON.** The message is still
`the console printed no JSON: …`; the panel shows it as the distinct non-JSON
line; `truncation` is absent.

**UC-9 (compatibility) — current panel, older console.** The console refuses
`--activity-after`; one retry without the new flags succeeds; the panel shows the
whole `activity.events` with the "unpaged history" note and keeps the run
readable.

**UC-10 (compatibility) — older panel, current console.** `status` answers with
the summary and the page, `outputs` included; no new flags were sent, so nothing
changed for the older caller.

**UC-11 (CLI) — `status --follow`.** Each frame carries the newest records
rather than the whole journal; the printed fields are unchanged; the human
-readable block keeps rendering the page it is given; the loop stops at a
terminal state as today. Per-poll cost stays flat as the journal grows.

**UC-12 (boundary) — the index is not a dependency.**
(a) With `.refactory/events.index.jsonl`, its provenance anchor **or its
overview line** deleted (any one or all three), truncated mid-line, made
unreadable, or carrying a torn final line, `status`
still answers the same `after`/`cursor`/`total`/`has_more`/`last_event_at` and
the same records (R-6.5), and a subsequent call rebuilds the index.
(b) With the journal containing a malformed *or non-object* complete line
anywhere, the `journal_gap` record keeps its slot and `total` still counts it;
`last_event_at` equals today's answer — `null` when the gap is the newest
record, the `.get('at')` of the newest record otherwise (F-12, R-1.6).
(c) **Journal replaced under a warm index (H-2).** After a new invocation
archives `.refactory/events.jsonl` into `.refactory/invocations/<id>/artifacts/`
and the run still has no journal, every answer equals a full scan's (`total: 0`,
`last_event_at: null`), and once the fresh journal exists every answer equals a
full scan of *it* — never a mix of the archived journal's offsets with the new
file. The rebuild after replacement answers identically to the full scan
(§4.6 `[R-1.6.2]`).

---

## 10. Automated test plan

Every row states the file, the case, and the IDs it locks. All commands in §12.

| Test | File | Locks |
|---|---|---|
| T-1 `events_page` returns a bounded, ordered slice and the total; a trailing incomplete write is excluded from both | `refactory/tests/test_live_activity.py` | R-1.1-3, F-1 |
| T-2 one `status` answer performs **one** journal pass (call counter on `Path.read_bytes`) and reports `last_event_at`/`total` from it | `refactory/tests/test_live_activity.py` | R-1.6-7, §4.2, SD-3 |
| T-3 a 20,000-record journal with **≤8 worker receipts and ≤8 delivery receipts**: the 2nd and 100th successive pages each complete within a bound derived from the first call's own duration (e.g. `≤ first/10` and under a wide absolute ceiling), a 200-record page returns 200 records, and — **with the index warm for both journals** — the per-page duration at `N = 20,000` is within a wide factor of the same page at `N = 500` **at the same worker count** (journal cost independent of `N` once indexed, R-1.7a-c). The cold first call is allowed to cost `O(N)` (F-11) and is not compared | `refactory/tests/test_live_activity.py` | R-1.7a-c, R-1.7e, UC-2, UC-12, F-11, §7.2 SD-3 |
| T-3b the same fixture, fixed 200-record page, split into two assertions. **(i) Journal-size assertion, warm index, 1 worker:** with the index already built for both journals, the per-call time of a 200-record page on `N = 20,000` is within a wide factor of the same page on `N = 500`. **(ii) Worker-slope assertion:** with `N` fixed and the index warm, the per-call time grows with the worker count — measured on the baseline at `N = 500`: 2.4 / 3.4 / 6.6 / 14.4 ms at 1 / 50 / 200 / 500 worker receipts (round 4; round 3 measured 2.5 / 3.6 / 7.1 / 15.6 ms for the same shape) — and the test asserts the shape, i.e. monotonic growth and a large multiple from 1 to 500 receipts, under a wide factor. It must **not** assert "a 20,000-record journal with 1 worker is not slower than a 500-record journal with 1 worker" on a *cold* index: measured 104.5 ms vs 2.4 ms, because that difference is exactly the `O(N)` scan R-1.7 exists to remove, and the baseline has no index at all (round 3's wording was false, H-4) | `refactory/tests/test_live_activity.py` | R-1.5, R-1.7a, R-1.7e |
| T-4 `--activity-limit` is accepted and clamps at 1000; `--activity-limit 0`, `-1` and a non-integer exit 2; `--activity-after -1` exits 2 with the console's own message | `refactory/tests/test_console_projects.py` (or a new `test_console_activity.py`) | R-1.1-2, R-1.8, UC-4 |
| T-5 a 400-record journal answers 200 + 200 for `status`, `status --follow` and the service's call shape and the union equals the journal; no answer exceeds `page_size` | `refactory/tests/test_live_activity.py` | R-1.1, R-1.4, R-1.7d, UC-4 |
| T-6 every page still carries the summary members and the `status()`-added members, including `outputs` | `refactory/tests/test_live_activity.py` | R-1.5, R-6.2, SD-4 |
| T-7 `status --follow` advances its activity cursor between frames (per-frame `activity.after` strictly increases) and two successive frames of an unchanged journal are not identical; printing is asserted unchanged apart from the paged `activity.events` | `refactory/tests/test_gui_loop.py` | SD-6, R-1.7c, UC-11 |
| T-8 `format_status` still prints warnings from the `activity.events` page it is given | `refactory/tests/test_live_activity.py` | SD-6 |
| T-9 observer `save()` writes a usage summary with no `calls` array; a 30-request worker's journal stays linear | `refactory/tests/worker_observer_test.mjs` | R-3.1, UC-6 |
| T-10 a pre-existing journal record with `usage.calls` still reads, and its summary row reports `calls` as a number | `refactory/tests/test_live_activity.py` | R-3.3, R-6.4 |
| T-11 `Console.run` reports `truncation` (`capBytes` + `retainedBytes` + `argv`) when the reader is lossy and stdout does not decode, and the non-JSON message verbatim when it is complete; no `observedBytes`/`spillPath` member exists | `packages/refactory/tool-refactory/tests/tool-refactory.spec.ts` | R-5.1-3, §4.4, UC-7, UC-8 |
| T-12 a `lossy` reader whose retained text still decodes as JSON is a successful call with no `truncation` | `packages/refactory/tool-refactory/tests/tool-refactory.spec.ts` | R-5.3, §4.4 |
| T-13 `status` against a console that refuses `--activity-after`/`--activity-limit` retries once without them and reports `suppressed`; a second distinct failure is not retried | `packages/refactory/refactory-console/tests/service.spec.ts` | R-6.1, UC-9 |
| T-14 `status` maps `truncation` to `refactory/console-truncated` (with `capBytes`/`retainedBytes`) and non-JSON to `refactory/console-failed` | `packages/refactory/refactory-console/tests/service.spec.ts` | R-5.4 |
| T-15 service `status` passes `activity-after` = the caller's `after` and `activity-limit`; `watchStream` advances `activityAfter` across polls and never resets to 0 mid-stream | `packages/refactory/refactory-console/tests/service.spec.ts` | R-4.2, SD-5, F-2 |
| T-16 the panel's attach reads page 0, appends polled pages through its highest `activity.cursor`, and a run switch resets to 0 and drops held history | `packages/client/ui-refactory/tests/face.client.spec.ts` | R-4.1-5, F-1, F-2 |
| T-17 refresh passes the highest held `activity.cursor`, not the top-level `cursor` | `packages/client/ui-refactory/tests/body.client.spec.tsx` | R-4.2-3, F-3 |
| T-18 the level control renders three options with `standard` selected on attach; choosing `verbose` exhausts pages to `has_more: false` and the held history then spans index `0 … total-1` with `complete: true`; `compact` hides worker detail but keeps the paging control | `packages/client/ui-refactory/tests/body.client.spec.tsx` | R-2.1, R-2.3-4, R-2.6-7, SD-2, UC-3, UC-5 |
| T-18b with a held history of 5,000 records (not 269,900 — the bound, not the browser, is what is asserted) at `verbose`, the number of `<li>` nodes under `[data-refactory-activity]` is `≤ RENDER_WINDOW + 2` (=202), scrolling to a row near the end renders that row and still keeps the count `≤ 202`, and a page arrival re-renders the window rather than the history | `packages/client/ui-refactory/tests/body.client.spec.tsx` | R-2.2, R-2.5, SD-12, F-4 |
| T-19 "Load earlier activity" walks to index 0 and then disables | `packages/client/ui-refactory/tests/body.client.spec.tsx` | R-1.4, F-6 |
| T-20 an older console's answer without `has_more`/`total` renders the whole `activity.events` with the unpaged note | `packages/client/ui-refactory/tests/body.client.spec.tsx` | R-6.3, UC-9, UC-10 |
| T-21 the store keeps `activityLevel` per tab and defaults it to `standard` at `start`; `forget` drops held history | `packages/client/ui-refactory/tests/store.client.spec.ts` | R-2.1, §4.5 |
| T-22 a frame for a different `activity.run_id`, or with an `after` past the held cursor, triggers a page-0 refetch rather than a splice | `packages/client/ui-refactory/tests/face.client.spec.ts` | R-4.5 |
| T-23a (M2) `watchFrame` (`packages/client/ui-refactory/src/client/run-stream.ts:83`) still validates a frame with the new optional `activity` members absent — a codec-shape assertion only, adding no panel behaviour | `packages/client/ui-refactory/src/client/run-stream.ts` (codec shape; exercised by T-23b's suite — `run-stream.client.spec.ts` covers contribution codecs only and contains no `watchFrame`) | R-6.2-3 |
| T-23b (M3) `RunActivity` (rendered through `RefactoryBody`) still renders the "Outputs and validation" disclosure with `activity.outputs`, at `standard` and at `verbose` | `packages/client/ui-refactory/tests/body.client.spec.tsx` | R-2 (outputs row), SD-4, E-1 |
| T-24 with `events.index.jsonl` (or the provenance anchor, or the overview line) absent, corrupt, truncated mid-line and torn at the final line, `status` answers identical records/`total`/`has_more`/`last_event_at` and rebuilds the index; **and with a `journal_gap` as the newest complete record**, `total` counts it, `has_more` follows `cursor < total`, the gap record is served with `kind: journal_gap`, and **`last_event_at` is `null`** (today's `all_events[-1].get('at')`), with consecutive trailing gaps likewise `null`, and with an ordinary newest record it is that record's `at`; **plus the replacement case (H-2):** journal archived away and absent → `total: 0`, `last_event_at: null`; fresh journal → `total`/`last_event_at` of the fresh journal; a fresh journal at least as long as the old one, at the same path → still the fresh journal's answers (the first-record digest catches it) | `refactory/tests/test_live_activity.py` | R-1.3, R-1.6, R-1.7b, R-1.7d, R-6.5, [R-6.5.2] (R-6.6), UC-12, F-11, F-12, SD-11, §4.6 |
| T-25 **`last_event_at` equals today's answer on a fixed matrix (H-1).** For each of 15 journals — malformed last complete line; two consecutive trailing gaps; gap then unterminated tail; JSON line that is not an object; malformed middle line; good JSON with unterminated tail; empty; missing; blank-line-only; blank line at tail; record without `at` before a good one; newest record without `at`; only gaps; newest `{}`; gap then an `at`-less record — the test asserts `snapshot(root)['last_event_at']` is unchanged from the baseline rule (the `at` of the newest counted record, or `None`) and that the indexed answer equals it, and asserts `total` matches the baseline count in all 15. Round 4 measured: today's rule and §4.6's rule agree 15/15; the round-3 backwards walk disagrees on 7 of 15 (round 4's reviewer measured 7/15; independently re-measured in round 5) | `refactory/tests/test_live_activity.py` | R-1.3, R-1.6, §4.6, F-12 |
| T-26 **Concurrent builders and torn lines (H-3).** Two processes build the index for the same growing journal at once (and one appends a torn line to the index mid-run): every answer from both readers equals the full-scan answer for the journal size they observed, and the resulting index is either complete or truncated-and-rebuilt — never half-trusted. A journal with three writer processes appending (spine `emit`, `workers.event`, observer `appendFileSync` shape) keeps its record count consistent with the newline-terminated lines | `refactory/tests/test_live_activity.py` | [R-6.5.2] (R-6.6), R-1.6, §4.6, SD-11 |

The existing suites that pin behaviour this spec must not break:
`refactory/tests/test_observe.py` (progress), `refactory/tests/test_live_activity.py`
(journal rules, graph snapshot), `packages/refactory/tool-refactory/tests/contract.spec.ts`
and `tool-refactory.spec.ts` (argv/decoder), `packages/client/ui-refactory/tests/*`
(panel), `packages/bundle/refactory-web/tests/*` (bundle composition — unaffected
but must stay green).

---

## 11. Manual verification checklist

1. Start a create run that takes a while; attach it in the panel with the run
   directory. Expect success, `standard` selected, and a visible activity
   history; `refactory status --follow --run <dir>` in a terminal agrees.
2. Note the journal line count (`wc -l <run>/.refactory/events.jsonl`). Switch to
   `verbose` and confirm the *reachable* count reaches it: the panel's paging
   reports all `total` records fetched (`has_more: false`), scrolling reaches
   index 0, and the panel stays responsive (the DOM holds at most 200 rows, so
   do not expect 269,900 `<li>` nodes — that would be the failure mode R-2.5
   forbids).
3. Switch to `compact`; confirm worker rows lose their detail, the level control
   still shows `compact`, and the history is still reachable.
4. Press "Load earlier activity" until it disables; confirm the first record of
   the journal is shown.
5. Point the panel at a run from before this change (a journal with
   `usage.calls`). Confirm it opens, pages, that a worker row shows a call count
   rather than a JSON array, and that "Outputs and validation" is still there.
6. Point a current panel at a console built before this change (a checkout
   without `--activity-after`). Confirm the run still opens and the panel says
   the history is unpaged.
7. Reproduce a cap overflow deliberately (temporarily set
   `collectBytes: 1024` in the service config). Confirm the notice names the cap
   and the retained size, and does not say "printed no JSON".
8. Watch the journal line count during a run with many model requests; confirm
   growth is proportional to requests rather than quadratic.
9. On the large run, `time` two successive `refactory status --run <dir>` calls;
   confirm the second (indexed) call is much faster than the first and does not
   grow with journal length, and that `.refactory/events.index.jsonl`,
   `.refactory/events.index.provenance.json` and
   `.refactory/events.index.overview.json` all exist. The first call is allowed to
   cost `O(N)` (F-11) — only the warm call is compared.
10. **Replacement is answered by the new journal, not the old one (H-2).** Note a
    run's `activity.total`/`last_event_at`, then re-prepare the same run
    directory (`refactory create … --workdir <same dir>`, or `design`/`edit`) so
    `<dir>/.refactory/events.jsonl` moves under
    `.refactory/invocations/<id>/artifacts/`. Before the new invocation writes
    anything, `refactory status --run <dir>` reports `activity.total == 0` and
    `activity.last_event_at == null`; once the new journal exists, both values
    describe the new journal. The stale index file may still be on disk; it is
    rebuilt, and it must not have changed either answer.
11. **A trailing malformed line is `null`, not the previous event.** Append one
    malformed complete line (e.g. `printf 'not json\n' >> <dir>/.refactory/events.jsonl`)
    and confirm `activity.last_event_at` is `null` while `activity.total`
    increased by one and the last served record is `kind: journal_gap` — the
    same answer the CLI gave before this change (HEAD `snapshot`, `git show HEAD:refactory/refactory_dsh/live.py` (`snapshot`, :145 of that revision)).

---

## 12. Done gates

Objective and executable, and runnable in any checkout: every command is
worktree-relative and resolves this worktree's own `.venv`, so the block runs
verbatim in a CI runner as well as on the machine that built it. Export the
worktree root once and the gates follow it:

```sh
cd "$(git rev-parse --show-toplevel)"
```

1. Branch and merge state: the branch carries this PR's commits and the branch
   is not behind `main`.
   ```sh
   git fetch origin main && test "$(git rev-list --count HEAD..origin/main)" = 0
   ```
   → exit 0. (This replaces the pre-PR gate on `.hai/state.yaml`: that file is
   not on this branch, so a command naming it could not run, and the merge-state
   check is what a reviewer actually needs to know before reading the diff.)
2. Python activity/journal suite:
   ```sh
   cd refactory && ../.venv/bin/python -m pytest tests/test_live_activity.py tests/test_gui_loop.py tests/test_observe.py tests/test_console_projects.py -q
   ```
   → all pass (T-1…T-8, T-10, T-24, T-25, T-26, T-27, T-28).
   A shell that exports `DSH_HOME` (a developer's DSH session) must clear it
   first — `env -u DSH_HOME -u DSH_PROFILE -u DSH_PROFILE_DIR -u DSH_SHELL -u DSH_SESSION_ID -u DSH_WEB_URL` —
   because the suite's seal treats the operator's real DSH home as off-limits
   and fails the run that reaches it. CI has none of these and needs nothing.
3. Observer suite:
   ```sh
   cd refactory && node --test tests/worker_observer_test.mjs
   ```
   → all pass (T-9).
4. Boundary + service + panel unit suites (including the live service spec,
   which now has a `.venv` to run against):
   ```sh
   pnpm vitest run packages/refactory/refactory-console/tests packages/refactory/tool-refactory/tests packages/client/ui-refactory/tests packages/bundle/refactory-web/tests
   ```
   → all pass (T-11…T-23a and T-18b, plus the existing panel/bundle suites and both
   `service.spec.ts` and `service-live.spec.ts`).
5. Typecheck: `pnpm typecheck` → exit 0.
6. Generated discovery docs stay current: the checked-in generated output
   (whatever `gen_discovery` writes under `refactory_dsh/data/`) matches what
   the code generates now, so the diff the build leaves behind is intentional
   and reviewed. This gate checks **only** that; it does not check that the new
   console flags appear anywhere.
   **Corrected premise (round 3, G-4):** the generated `status` entry *does*
   come from console.py's argparse -- `discovery.operations()` calls
   `parser_operations(_parser())` (`discovery.py:49-53`) -- so adding
   `--activity-after`/`--activity-limit` **will** change the generated `status`
   parameters (which today are only `project`, `runs_root`, `run`, `after`,
   `follow`, `interval`). The resulting diff is expected and must be reviewed; a
   build that leaves it unregenerated fails this gate, and a build that
   regenerates it has done nothing wrong. The flags are a console contract
   (R-1.1), not a discovery-surface promise: this spec adds no new discovery
   guarantee.
   `.venv/bin/python -m refactory_dsh.tools.gen_discovery --check` (from the
   worktree root) → exit 0. Regenerate with `pnpm run docs:generate` and review the diff.
7. **Bounded activity page, executable.** One pytest case in
   `refactory/tests/test_live_activity.py` named
   **`test_bounded_activity_page`**. It builds a
   400-record journal with 1000-byte records, invokes `console status` as a
   subprocess for `--activity-after` 0, 200, 400 and for `--activity-limit 50`,
   and asserts for every answer that `len(activity.events) <= page_size`,
   `cursor == after + len(events)` and `has_more == (cursor < total)`; plus the
   rejection exit codes (`--activity-limit 0`, `-1`, `--activity-after -1` →
   exit 2); plus the **gap case**: a 401st line that is a malformed *complete*
   record makes `total == 401`, so the answer at `after == 400` serves
   `activity.events[-1]` with `kind == 'journal_gap'` and asserts
   `has_more == (cursor < total)` there exactly as on every other answer — and
   on that page `cursor == 401 == total`, so **`has_more` is `false`**: the page
   at `after == 400` *is* the gap's own page, the single record left, and serving
   it advances the cursor to `total`. The gap's *reachability* is therefore
   asserted from the start rather than from index 400: **from `after == 0` the
   gap is the third page's record** — 200-record pages give records `0…199`,
   then `200…399` at `cursor == 400` with `has_more` `true`, then the single gap
   record at `after == 400` with `cursor == 401` and `has_more` `false`, so the
   exhaustive pass serves all 401 records and the gap is its last. The case also
   yields **`last_event_at == null`** (today's answer for a newest gap — H-1;
   round 3 wrongly asserted "never `null`"), while the same journal with one more
   ordinary record appended yields that record's `at`; and a 402nd
   **unterminated** tail line does not change `total`. What this gate asserts
   about `has_more` is R-1.4's identity, never a fixed `true`/`false` per
   `after`; revision 4's reading of the gap case — `has_more` "true at
   `after == 400`" — contradicted R-1.4 on that very page, which is the round 5
   correction. Command (from the worktree root):
   `cd refactory && ../.venv/bin/python -m pytest tests/test_live_activity.py -q -k test_bounded_activity_page`
   → 1 passed, exit 0. **This gate is red until built**: the baseline can never
   pass it (`console.py:552` calls `live.snapshot(run_root)` with no cursor, so
   `events` is the whole journal; `console.py:1380-1392` has no
   `--activity-after`), and the same baseline also writes the cumulative
   `usage.calls` array into the journal (`worker_observer.mjs:30-34`, copying
   `receipt.usage` which carries `calls`). It replaces the previous grep-based
   judgement gate, which a reviewer cannot execute.
8. Gates 2–5 are green **together** in one run, and a fresh test run's journal
   contains no record whose `usage.calls` is an array:
   `grep -c '"calls":\[' <run>/.refactory/events.jsonl` → `0` (and `grep -c`
   exits 1 when the count is 0, so the check is `test "$(grep -c ... || true)" = 0`).

---

## 13. Milestone suggestion (for the build loop)

- **M1 — console contract.** SD-3 index + one-pass page in `live.py`, the
  `events.index.jsonl` recovery rules (§4.6), `console.py` `--activity-after`
  /`--activity-limit` (with `0` refused) and the paged envelope with `outputs`
  kept, CLI follow advancing (SD-6), R-3 observer change, the index identity
  anchor and atomic-append rules (§4.6 `[R-1.6.2]`/`[R-6.5.2]`), and
  `test_bounded_activity_page` (including its gap case). Regenerate the
  discovery docs (gate 6) in the same milestone.
  Tests T-1…T-10, T-3b, T-24, T-25, T-26, and the `test_bounded_activity_page`
  gate (gate 7).
- **M2 — boundary and service.** `Console.run` truncation verdict
  (`capBytes`/`retainedBytes`) and one-shot fallback,
  `refactory/console-truncated`, `status`/`watchStream` cursor forwarding,
  `types.ts` additions. Tests T-11…T-15, T-23a.
- **M3 — panel.** store history/level, face accumulation and exhaustion (held
  history uncapped, `complete` flag), `RunActivity` levels with the
  `RENDER_WINDOW` virtual list (SD-12; outputs disclosure retained), paging
  gesture, locale copy, the two failure lines. Tests T-16…T-22, T-18b.
- **M4 — gates.** §12 end to end, manual checklist §11.

---

## Revision log

**Round 2** (independent review: FAIL; findings F-1…F-10 from
`SPEC_REVIEWER`). Every finding is fixed in place; no contract was weakened.
(Round 3 confirmed F-2, F-3, F-4, F-6, F-7, F-8, F-9 and F-10 as resolved and
did not touch them; F-1's mechanism was confirmed independently, only its SD-3
magnitudes were withdrawn — see Round 3 / G-8.)

| Finding | Disposition | What changed |
|---|---|---|
| F-1 | fixed | R-1.7 rewritten as a cost contract `O(page_size)` per page for a non-decreasing `after`, `O(N)` total for an exhaustive pass, `total`/`last_event_at` refreshed in `O(1)`; measurement method stated (call counters + bounded-time large-journal test); SD-3 rewritten around a derived `events.index.jsonl` (§4.6, new), with this round's measurement recorded (20k records → 80–190 ms per page at *every* `after`); new tests T-3, T-24, §12 gate 7; R-6.5 (index is never a dependency). |
| F-2 | fixed | R-5.1(b) no longer claims an observed byte total; §4.4 documents that the seam exposes only `lossy` + retained-tail length, and `truncation` carries `capBytes`/`retainedBytes` (no `observedBytes`); UC-7, T-11, T-14 updated. |
| F-3 | fixed | §4.4 states the truth at this call site: `spill` is unconfigured (`console.ts:233-236`) so `spillPath` is always absent and is not in the contract; `lossy` on an after-settlement `readFrom(0)` **is** `CollectedOutput.truncated`. SD-10 records why spill is not enabled (per-poll I/O amplification). |
| F-4 | fixed (resolved to the reversible default) | §4.2 and SD-4 keep `outputs` unchanged; the removal is no longer routine; E-1 stays **open** and human-owned, naming the question, options A (taken, keep) / B (optional) / C (remove) and their consequences. R-2's table regains "outputs" for standard/verbose. |
| F-5 | fixed | §12 gates 2, 4, 6 use `<worktree_root>/.venv/bin/python`; the three stale `.venv`-in-checkout notes are gone; gate 4 now runs the whole `refactory-console/tests` directory, so `service-live.spec.ts` is included on its merits. |
| F-6 | fixed | §12.7 is now the executable pytest case *(then labelled G-7; renamed `test_bounded_activity_page` in round 4, H-8)* (bounded pages at every verb + exact exit codes), with the explicit note that it is red until built. |
| F-7 | fixed | R-1.1 introduces both `--activity-after` and `--activity-limit` as accepted flags for `status` and `status --follow`; R-1.2 defines `0` as a refusal (not a clamp); §6's usage line shows `[--after N]` with `--follow` and explains the two cursors; §4.2's validation table and UC-4 updated. |
| F-8 | fixed | SD-6 keeps only the SHALL (advance the cursor); the "keep the previous frame's events for `format_status`" clause is deleted, replaced by the true statement that the page it renders is the newest one and durable warnings come from `interactive.run_warnings`. T-7 now proves cursor advance only; T-8 proves `format_status` on the page it is given. |
| F-9 | fixed | E-2 reframed as the reversible design constraint (residual per-page cost for a descending `after` on a fresh-process host), explicitly not a retention-policy escalation and explicitly not human-owned. |
| F-10 | fixed | §4.2 states that `events_page` and the overview share one pass and that `status` does not parse the journal twice; §6 shows `live.page()` as the one-pass entry point and says why the primitives are still shown. |

**IDs whose semantics changed** (same ID, different obligation — read these
before diffing code):

- **R-1.7** — was "linear in `N` for the index scan is acceptable"; now a strict
  per-page cost bound with a measurement method.
- **R-1.1** — now introduces `--activity-limit` (and `--activity-after`) as
  accepted options on both `status` and `status --follow`; previously only
  `--activity-after` was named normatively.
- **R-5.1** — dropped the observed-byte-total claim; the notice is now cap +
  retained size + argv.
- **R-6.1** — the one-shot fallback now covers `--activity-limit` as well as
  `--activity-after` (both are new options from the same console version).
- **SD-3** — was "single pass, `splitlines` admissible for small journals"; now
  a persistent derived offset index with explicit recovery rules (§4.6).
- **SD-4** — was "remove `outputs`"; now "keep `outputs`; removal is E-1's
  question".
- **SD-6** — was "advance the cursor and retain the previous frame's events";
  now "advance the cursor; printing unchanged".
- **SD-10** — was "use `lossy`/`spillPath`"; now "use `lossy` only; spill off".
- **§12.7** — was a grep judgement; now executable (named
  `test_bounded_activity_page` since round 4, H-8).

New IDs added this round: R-1.8 (measurement), R-6.5 (index never a
dependency), §4.6 (index file), F-10 (index cold start — **renamed F-11 in round
3**), SD-11 (who writes the index), UC-12 (index loss), T-3, T-24, and the
executable gate case (then `G-7`, now `test_bounded_activity_page`). Removed:
T-5's `outputs` absence assertion (replaced by T-6's presence assertion) and the
§6 "`outputs` is removed" comment.

**Round 3** (independent review: FAIL). Round 2's resolution of F-2, F-3, F-4,
F-6, F-7, F-8, F-9 and F-10 was CONFIRMED and is untouched here. The reviewer
also independently re-measured the baseline and confirmed F-1's **mechanism**
(flat in `after`, linear in `N`, two journal reads per snapshot; ~2.77 s for an
exhaustive 200-per-page pass over `N = 20,000`, projecting to ~9 min at
`N = 269,900`). No contract was weakened to make a finding go away; every
blocker is fixed by making the document agree with itself and with the code.

| Finding | Disposition | What changed |
|---|---|---|
| G-1 (§1.2 vs §4.6 self-contradiction) | **fixed** | §1.2's first non-goal is now "not a new **authoritative** activity store…" and states the explicit *permitted and required* exception: the derived, disposable, append-only `events.index.jsonl` of §4.6, named by R-1.7b. A second non-goal ("no other bookkeeping file") narrows the exception to exactly one file. A builder reading §1.2 first can no longer conclude the index is out of bounds. All other non-goals audited — none contradicts a requirement (see the audit line at the end of this log). |
| G-2 (`verbose` mapping unsatisfiable) | **fixed** | R-2's table no longer says verbose renders "all of them": it says a fixed render window (SD-12) with every other record reachable, and "held" is explicitly uncapped. R-2.2 is new and normative: verbose makes every record **reachable** (accumulated page by page until an answer reports `has_more: false`, which is the completeness evidence) while rendering is windowed at `RENDER_WINDOW = 200` and the DOM row count is `≤ RENDER_WINDOW + 2`, independent of `N`. R-2.5 states the interactivity bound and its precedence. F-4, UC-2, UC-3, §11.2 and T-18 rewritten; T-18b added. |
| G-3 (`total`/`last_event_at` unsound for gap records) | **fixed** | §4.6 defines the index line as `{schema_version, i, start, end, counted, last_at}`: `counted: true` on every line this version writes (an older/partial line without it is rebuilt, never guessed); `last_at` is **`null`** for a `journal_gap`. `total` is the last trusted line's `i + 1` — sound because there is one line per counted record and none for the uncounted unterminated tail, so offsets are never used to decide `total`. `last_event_at` is the newest trusted line's non-null `last_at`, with an explicit **index-only** bounded backwards walk past trailing gaps (no journal open). R-1.6, R-1.7b, F-12 (new), T-24, §12 G-7 and UC-12 all gained the gap case. Verified against the code: `live.py:57-65` counts a malformed complete line as a `journal_gap` with no `at`, and a journal whose last line is a gap makes today's `snapshot` return `last_event_at: null`. |
| G-4 (§12 gate 6 premise false) | **fixed** (premise corrected, gate kept as a plain freshness gate) | Gate 6 is now "generated discovery docs stay current" and says so; the false parenthetical is replaced with the measured truth (`gen_discovery` reads console.py's argparse via `discovery.operations()` → `parser_operations(_parser())`, `discovery.py:49-53`, so the new flags *do* change the generated `status` parameters — today only `project`, `runs_root`, `run`, `after`, `follow`, `interval`). The gate's stated purpose now matches what it checks; no discovery-surface contract was added. |
| G-5 (unbounded "cheap" fields) | **fixed** | R-1.5 defines "cheap" as bounded by the run's graph, not by `N`, and records the real cost: `live.py:124-125` parses every worker and delivery receipt on every call (~19 ms at 500 receipts). New R-1.7e puts receipts outside `N`'s guarantee and forbids per-record re-derivation. R-1.8 makes T-3's fixture fix the worker/delivery count and compare `N = 500` vs `N = 20,000` at the same worker count; new T-3b asserts the worker/`N` separation directly. |
| G-6 (E-1's "none is known" false) | **fixed** (E-1 left open and human-owned) | §4.2 and E-1 now state the fact plainly: `outputs` is `read(root/'facts.json')` (`live.py:154`), is **not** bounded by `page_size`, is paid on **every** page (1,350× on UC-2), and duplicates content the panel can read once from `facts.json`/`artifacts`. Those are named as the concrete reason a human might prefer Option B. Option A (keep) remains the reversible default **in force**; the "none is known" clause is deleted. R-1.7d names the exemption. E-1 was **not** resolved by the writer. |
| G-7 (F-10 ID collision) | **fixed** | Round 2's flow "F-10 (index cold start)" is renamed **F-11** (with a one-line note saying why), and its only other reference is corrected. The new gap flow is **F-12**. Review finding F-10 keeps its ID inside the Round 2 table, so review-finding IDs and flow IDs no longer collide. |
| G-8 (SD-3 magnitudes unreproducible) | **fixed** | SD-3 now quotes a reproducible recipe (record ≈ 1,207 B, journal 24.1 MB at `N = 20,000`, the scratch script inline) with this round's own numbers: `events` 75 / 61 / 57 ms at `after` 0 / 10,000 / 19,800, `snapshot` 128 ms, 100-page exhaustive pass 6.2 s. The reviewer's independent numbers are cited as the corroborating shape. Revision 2's "80–190 ms" is explicitly withdrawn as a single machine's measurement generalised; the qualitative claim (flat in `after`, linear in `N`, two reads per snapshot) is the load-bearing one and the document now promises magnitudes, never seconds. |

**Round 3 — IDs whose semantics changed** (same ID, new or narrowed
obligation):

- **R-1.5** — "cheap" is now defined as `O(workers + deliveries)`, explicitly
  *not* constant time and not covered by `N`'s bound; receipts named as the cost.
- **R-1.6** — `last_event_at` is now "the newest complete record **that has an
  `at`**", with `null` only when no record has one; the "newest record's `at`"
  reading is forbidden.
- **R-1.7** — new clause (e) (receipt cost outside the journal bound, never
  re-derived per record), (b) now permits exactly one bounded walk
  (`journal_gap`-only, index-only), (d) now names the `outputs` exemption.
- **R-1.8** — T-3's fixture is now required to hold the worker/delivery count
  fixed; the `N = 500` vs `N = 20,000` comparison is added.
- **R-2** — the verbose row and clause 2 no longer mean "every record on screen":
  held history is uncapped, rendering is a `RENDER_WINDOW = 200` virtual window,
  the DOM row count is bounded, and "interactive at every level" is clause 5
  (previously clause 5 was level-switch behaviour, now clause 7). Clauses 3–7
  renumbered accordingly (paging control is now 4, control widget 6,
  no-refetch 7).
- **R-2.2 / R-2.5** — new normative clauses (reachability + windowed rendering;
  interactivity as a bounded, testable obligation).
- **§4.6** — the index line gained `counted` and a nullable `last_at`; `total` is
  defined from line count, never offsets; `last_event_at` gained the bounded
  index-only backwards walk.
- **§1.2** — the "no new index/store" non-goal now excludes the §4.6 index
  explicitly.
- **§12 gate 6** — purpose restated (freshness only) and its premise corrected.
- **SD-3** — measurement section replaced with a reproducible recipe and
  magnitudes; the decision itself is unchanged.
- **F-4, UC-2, UC-3, UC-12, T-18, T-24, §11.2, §12 gate 7, M1, M3, gate 4** —
  updated to state reachability + the render window, and/or the gap case.

**New IDs this round:** R-2.2 and R-2.5 (normative clauses), **SD-12**
(`RENDER_WINDOW = 200` virtual window), **F-11** (renamed F-10: index cold
start), **F-12** (gap records at the journal tail), **T-3b** (receipt-cost
separation), **T-18b** (DOM bound), and the `RENDER_WINDOW` / `complete` /
`activityScroll` members of §6's `HeldHistory`. **Renamed:** F-10 → F-11
(index cold start; review finding F-10 unaffected). **Removed:** nothing.

**Round 3 — ID collision audit.** Narrative flow IDs are `F-*`; review finding
IDs are `F-1…F-10` (round 2) and `G-1…G-8` (round 3) and live only in this log.
Body and log previously both used `F-10`; after this round `F-11`/`F-12` are the
only new flow numbers, `F-10` appears in the body only inside F-11's one-line
rename note, and the gate `G-7` is spelled out as a gate wherever the body
mentions it.

**Round 3 — §1.2 non-goal audit (every non-goal checked against every R).**
(R1) "Not a new authoritative store… plus the explicit §4.6 exception" — agrees
with R-1.7b, R-6.5, §4.6. (R2) "No other bookkeeping file / no cross-run index /
no parsed-record cache" — agrees with §4.5's state table (the only persisted
addition is the index) and with SD-11 (no daemon, no cache). (R3) "Not a rewrite
of the worker route/usage contract, the steering ledger, the candidate/
leaderboard contract, or the graph control contract; touched only where they
carry the journal on the wire" — agrees with R-3 (writer bound + `calls` count
only) and R-1.5 (receipt members still travel). (R4) "Not a retention or deletion
policy; nothing deleted, compacted, compressed or rewritten" — agrees with
R-3.4 and R-6.5 (a *derived* index file may be deleted and is not history) and
with DEC-001; the append-only rule in §4.6 is the same commitment. (R5) "Not a
change to what counts as evidence of compliance" — agrees with R-5 (reporting
only) and leaves the delivery-receipt language untouched. (R6) "Not the two-lane
bounded console reads proposed for other work (§8)" — agrees with R-5/R-6 (this
spec changes the decode verdict and the fallback, not the forwarding lanes) and
§8's out-of-scope note. No non-goal now contradicts a requirement.

**Round 4** (independent review: FAIL; findings H-1…H-8). Round 3's resolutions of
G-1, G-2, G-4, G-6 and G-7 were CONFIRMED and are untouched except where a
round-4 finding changes the same sentence (H-1 touches G-3's rule and G-7's "never
`null`" assertion). The defect this round exists to fix is that revision 3 did not
reproduce **today's** observable answers: its §4.6 `last_event_at` walk answered a
different value from `live.py:145` in every case where the newest counted record
has no `at`, and its index-trust rule never checked that the contiguous valid
prefix belonged to the *current* journal, which a new invocation replaces at the
same path. No R-* was weakened: every answer now equals what the baseline answers,
and the `O(1)` claim got *stronger* (no walk at all). No human decision was
needed; nothing was escalated, and E-1/E-2 statuses are unchanged.

| Finding | Disposition | What changed |
|---|---|---|
| H-1 (SPEC_BLOCKER — `last_event_at` diverges from today's answer on trailing gaps) | **fixed** | Today's rule is `all_events[-1].get('at')` over `live.events`' records (`live.py:145`, `:44-66`) — a property of the **newest counted record**, `None` whenever that record has no `at`. §4.6 now defines `last_event_at` as exactly that: the newest trusted index line's `last_at`, which is `null` for a `journal_gap` or an object without `at`. The **backwards walk is deleted**, and with it R-1.7b's "bounded walk fallback". R-1.6 restated as an equality with today's answer and forbids the walk reading. F-12, T-24, T-25 (new), UC-12(b), §12 `test_bounded_activity_page` (gap case → `null`) and §11.11 now assert today's value. Verified on a 15-case matrix: the new rule agrees with today 15/15; revision 3's walk agrees 9/15 (it diverges on malformed last line, two trailing gaps, gap-then-unterminated-tail, non-object JSON line, newest `{}`, gap-then-`at`-less record). |
| H-2 (SPEC_BLOCKER — stale index after journal replacement) | **fixed** | §4.6 gains `[R-1.6.2]`: the index is trusted only when (1) its provenance names this run root and `.refactory/events.jsonl`, (2) `st_dev`/`st_ino` match, (3) `size_at_write <=` current journal size, (4) `total <=` the current journal's counted records and the newest trusted `end <=` its size, and (5) the SHA-256 of the first complete record (and its `end`) matches. Otherwise the prefix is empty and the **full rebuild** runs — and a rebuild must answer exactly what a full scan answers. New derived file `<run>/.refactory/events.index.provenance.json` (§4.5, §1.2, §3). `invocations.py:154-161`/`:206-209` (`.refactory/events.jsonl` moved into `invocations/<id>/artifacts/`, nothing installed) and `run_preparation.py:35-36` are cited as the cause; F-11 gains the archival case, UC-12(c), §11.10, T-24 gains the replacement case. Measured with revision 3's rule: 3 records then archival then 1 fresh → stale index said `3/t2` where today says `0/None` and then `1/<new>`. |
| H-3 (WARN — no concurrent-writer rule) | **fixed** | New normative `[R-6.5.2]` + R-6.6 and a §4.6 subsection "Writes are atomic per line": one `O_APPEND` `write()` per complete index line, write-to-temp-then-rename for the provenance anchor, a reader accepts only complete lines that parse with `counted: true` and `i` at its expected position, and a **torn final line is truncated at the last complete line and rebuilt from the journal, never guessed**; what a concurrent reader sees mid-write is stated. The three journal appenders are named (`live.emit` in the spine, `workers.event` `workers.py:88`, `worker_observer.mjs:30`) as are the reader processes (SD-11). SD-11's "made idempotent by a tolerant rebuild" now names the mechanism it asserted. New T-26. |
| H-4 (WARN — T-3b's first half is false) | **fixed** | T-3b split into (i) journal-size assertion **at a warm index and 1 worker** and (ii) worker-slope assertion. R-1.8's ordering clarified ("at the same worker count, with a warm index"), T-3 likewise ("with the index warm for both journals; the cold first call is allowed to cost `O(N)`"). Round 3's sentence "a 20,000-record journal with 1 worker is not slower than a 500-record journal with 1 worker" is removed as false: measured 104.5 ms vs 2.4 ms, which is the `O(N)` scan R-1.7 removes. Round 4's own slope, `N = 500`: 2.4 / 3.4 / 6.6 / 14.4 ms at 1 / 50 / 200 / 500 receipts. |
| H-5 (WARN — §2's "nothing else uses MUST/SHALL" is false) | **fixed** | §2's sentence now states where MUST/SHALL appears (this section; §4.6 as numbered clause markers `[R-1.6.1]`…`[R-6.5.2]`; §4.2 as `[R-1.6.1]`; §4.4 as `[R-5.3]`/`[R-5.4]`; E-1 as `[R-1.7d]`/`[SD-4]`) and that every other section is rationale. The spurious SHALLs are rewritten as traced clauses or plain statements: §4.6 (all of them → clause markers), §4.2 (one pass, `outputs`), §4.3/R-3.1/R-3.3 (restated as `[R-x.y]`), §4.4, E-1, §12 gate 7. New clause IDs R-1.6.3 (both options on the `status` sub-parser) and R-6.6 (atomic index writes) exist so no restated obligation lost its trace. |
| H-6 (NIT — SD-3 recipe/magnitude mismatch) | **fixed** | The inline recipe yields 22,148,890 B / 20,000 = **1,107.4 B** per record, i.e. **22.1 MB**, not 24.1 MB / 1,207 B. SD-3 now states 22.1 MB / ≈1,107 B and labels round 3's 24.1 MB figure as the recipe's own mismatch; the qualitative claim is unchanged. Round 4's run of the same script: 69 / 54 / 51 ms at `after` 0 / 10,000 / 19,800, `snapshot` 112 ms, 100 × 200-record calls in 5.6 s. |
| H-7 (NIT — T-23 split across milestones) | **fixed** | T-23 is split: **T-23a** (M2) is the codec assertion in `run-stream.client.spec.ts` (44 lines, no `RunActivity` reference — verified); **T-23b** (M3) is the panel assertion "`RunActivity` still renders `Outputs and validation`" in `body.client.spec.tsx`. §12 gate 4, M2 and M3 updated. |
| H-8 (NIT — gate `G-7` reuses the review prefix) | **fixed** | The pytest gate case is renamed **`test_bounded_activity_page`**; gate 7's `-k` selector, the "red until built" note, M1, and every historical mention (Round 2 F-6 row, §12.7 entry, new-IDs list, Round 3 changed-row list) updated. Gates are now named, never numbered; `G-*` belongs to review findings only. |

**Round 4 — IDs whose semantics changed** (same ID, new or narrowed obligation):

- **R-1.6** — `last_event_at` is now *the `at` of the newest counted record, or
  `null` when that record has none* — i.e. **equal to today's answer** for every
  journal. Revision 3's "the newest complete record **that has an `at`**" is
  reversed: it answered values today never returns. No walk of any kind.
- **R-1.7b** — the permitted `journal_gap` walk is removed; `total` and
  `last_event_at` are the last trusted index line's `i` and `last_at`, one
  lookup.
- **R-1.7d** — unchanged in substance (bounds `activity.events`; `outputs`
  exempt); now also the trace for E-1's clause.
- **R-1.8** — the `N = 500` vs `N = 20,000` comparison is now explicitly **at a
  warm index**; the cold call is excluded.
- **§4.6** — `last_event_at` rule reversed (no walk); the trust rule is now a
  five-check identity anchor instead of "a contiguous valid prefix"; atomic
  append and torn-line rules added; the file set is two files, not one.
- **SD-3** — magnitudes corrected to the recipe's own numbers (22.1 MB /
  ≈1,107 B); the decision is unchanged.
- **SD-11** — the idempotence claim now names the mechanism (`[R-6.5.2]`,
  R-6.6) instead of asserting tolerance without one.
- **F-11** — gains the archived-journal case (H-2).
- **F-12** — `last_event_at` for a trailing gap is now **`null`** (today's
  answer), not the previous record's `at`; the "no journal open" and
  "no walk" parts remain true.
- **UC-12** — restructured into (a) index loss, (b) gap records with today's
  `last_event_at`, (c) journal replacement (new).
- **T-3** — comparison explicitly at a warm index.
- **T-3b** — split into a warm-index journal-size assertion and a worker-slope
  assertion; the false cold comparison is gone.
- **T-23** — split into T-23a (M2, codec) and T-23b (M3, panel).
- **T-24** — the gap case now asserts `null`; the replacement case is added.
- **§12 gate 7** — renamed `test_bounded_activity_page`; its gap case asserts
  `null`.
- **§1.2 / §3 / §4.5** — two derived files (index + provenance anchor) instead
  of one.
- **§2 preamble / §4.2 / §4.3 / §4.4 / E-1 / §12** — every MUST/SHALL is either a
  numbered clause of an R or removed; the preamble sentence is now true.

**New IDs this round:** **R-1.6.1** (the index must not change any observable
answer), **R-1.6.2** (what the index may claim to cover; rebuild rule),
**R-1.6.3** (both new options on the `status` sub-parser), **R-6.5.1**
(index loss never fails a read; R-6.5's own clause), **R-6.5.2** (atomic index
writes; torn line truncated and rebuilt), **R-6.6** (the same, as R-6's own
clause), **T-25** (`last_event_at` equals today on a fixed matrix),
**T-26** (concurrent builders and torn lines), **T-23a**/**T-23b** (the split of
T-23), and the derived `events.index.provenance.json`. **Renamed:** gate
`G-7` → `test_bounded_activity_page`; `T-23` → `T-23a`/`T-23b`. **Removed:** the
`journal_gap` walk of R-1.7b/R-1.6 and its four supporting sentences.

**Round 4 — ID collision audit.** Flow IDs are `F-1…F-9`, `F-11`, `F-12` (no
`F-10` in the body). Review finding IDs are `F-1…F-10` (round 2), `G-1…G-8`
(round 3) and `H-1…H-8` (round 4), and appear only in this log. Requirement
clauses use `R-x.y` and `R-x.y.z` with the third level meaning "the section that
implements R-x.y states it" (R-1.6.1/2/3, R-6.5.1/2), which no review finding
shares. No gate carries a `G-*` number any more (H-8).

**Round 4 — §1.2 non-goal audit (re-run after the two-file change).**
(1) "Not a new *authoritative* activity store … plus the explicit §4.6
exception" — still agrees with R-1.7b, R-6.5, §4.6; the exception now names two
derived files instead of one, which is why the second non-goal was reworded.
(2) "No other bookkeeping file / no cross-run index / no parsed-record cache" —
agrees with §4.5's state table (the only persisted additions are the index and
its provenance anchor, both derived and disposable) and with SD-11 (no daemon, no
cache; the mechanism is atomic appends, not a cache). (3) "Not a rewrite of the
worker route/usage contract, the steering ledger, the candidate/leaderboard
contract, or the graph control contract; touched only where they carry the
journal on the wire" — agrees with R-3 (`[R-3.1]`/`[R-3.3]` writer bound and a
`calls` count only) and R-1.5 (receipt members still travel). (4) "Not a
retention or deletion policy; nothing deleted, compacted, compressed or rewritten"
— agrees with R-3.4, R-6.5 and §4.6: the journal is never rewritten; a rebuild
rewrites only the *derived* index, and archival of `events.jsonl` is an existing
invocation behaviour this spec describes rather than introduces (it must be
survived — H-2 — not adopted). (5) "Not a change to what counts as evidence of
compliance" — agrees with R-5 and leaves delivery-receipt language untouched.
(6) "Not the two-lane bounded console reads proposed for other work (§8)" —
agrees with R-5/R-6 and §8's out-of-scope note. No non-goal contradicts a
requirement after round 4.

**Round 5** (SPEC_CHANGE_REQUESTED raised by the M1 build loop;
request `logs/hai-build-loop/spec-change-request-M1-gate7.md`). Round 4's PASS
carried residual WARN/NIT findings J-1…J-6; this round settles them together with
one gate-7 expected-value defect the build loop found. **No requirement in §2.2
changes**: R-1.4's identity and R-1.6.1 are untouched, and no numbered clause was
added, removed or renumbered. INT-001, DEC-001 and DEC-002 are unaffected.

| Finding | Disposition | What changed |
|---|---|---|
| **M1 gate-7 gap case** (build-loop finding; SPEC_BLOCKER for the *gate text*, not the contract) | **fixed** | §12 gate 7's gap case claimed the 401st malformed *complete* record "keeps `has_more == (cursor < total)` **true** at `after == 400`". That is false on that page and contradicted R-1.4 next door: a 200-record page at `after == 400` serves exactly the one remaining record (the gap), so `cursor == 401 == total` and `has_more` is **`false`**. Verified by running the built console — 400 good records + a malformed 401st, `console status --run <dir> --activity-after 400 --activity-limit 200` → `after 400, cursor 401, total 401, has_more false`, one `journal_gap` event, `last_event_at null`; a full sweep from `after == 0` gives pages `0…199` (`has_more true`), `200…399` (`cursor 400`, `has_more true`), then the gap at `after == 400` (`cursor 401`, `has_more false`), all 401 records served. The gate now asserts the identity unconditionally on every answer and states the true reachability fact — the gap is the **third page's record from `after == 0`** — instead of a fixed `true` at `after == 400`. The implementation was already correct; only the gate's expected value was wrong. No code or test changes with this amendment. |
| J-1 (NIT — divergence count) | **fixed** | Both places that counted the round-3 backwards walk's divergences now say **7 of 15**, the round-4 reviewer's measured number (independently re-measured this round against T-25's own 15-case matrix: the walk diverges exactly on malformed last line, two trailing gaps, gap-then-unterminated-tail, non-object JSON, newest without `at`, newest `{}`, gap-then-`at`-less record — 7). R-1.6's parenthetical (`six of fifteen` → `seven of fifteen`) and T-25's row (`6/15` → `7 of 15`) are the two sites. |
| J-2 (WARN — partly-written line wording) | **fixed (restated, not as requested)** | §4.6 said a partially-written line is "truncated at the last complete line"; the reviewer noted `worker_observer.mjs`'s `appendFileSync` can split a record mid-line, so those bytes survive as a `journal_gap`. §4.6 now states the rule **by position**, without asserting what the round-4 wording implied. While the incomplete line is the journal's *final* line it is dropped from `total` (not counted, not served, R-1.3). Once a later byte is appended the torn bytes are terminated and surface as **one counted `journal_gap`**, and what that costs depends on the next writer: `live.emit` repairs the boundary first (`live.py:58`) yielding **two** records (gap + the new record, nothing lost), whereas a non-repairing append (`workers.event` `workers.py:88-89`, `worker_observer.mjs:35` `appendFileSync`) fuses its line into the torn bytes and **that later record is lost with it**. All three shapes measured: `live.emit` after a torn write → `total 3`, `['before','journal_gap','after']` (pinned by `test_append_after_interrupted_record_exposes_gap_and_new_activity`, `tests/test_live_activity.py:61-69`); non-repairing append after `{"tor` → `total 2`, `['a','journal_gap']`, raw `…{"tor{"schema_version": 1, "kind": "c", …}`; torn as the final line → `total 1`, `['a']`. The passage states explicitly that R-1.3 governs only how bytes are **counted and served**, not whether a record survives a torn predecessor, and adds no recovery requirement. This is today's behaviour; R-1.3 unchanged. |
| J-3 (WARN — `[R-6.5.2]` vs `R-6.6`) | **fixed (premise corrected)** | `[R-6.5.2]` is **not** a dangling citation: it is a real, defined clause of R-6.6, declared in §2.2's R-6 clause 6 (:353) and listed in §4.6's marker list (:608). The genuine defect is the *unqualified* bare `R-6.6` in T-24's and T-26's trace columns, which reads in-range of R-6 (only six clauses) but resolves to no numbered obligation. Those two rows now cite `[R-6.5.2] (R-6.6)`, and §4.6's marker list states explicitly that `[R-6.5.2]` is the marker for R-6.6 — the same obligation, either label admissible. No clause added or removed; the dual label is now declared rather than implied. |
| J-4 (NIT — SD-3 timing) | **fixed** | SD-3's "100 calls in 5.6 s" (and §4.6's/pipeline's restatement) is now a range: **~5.6–10.7 s**, naming round 4's run (5.6 s, ~56 ms/call) and the reviewer's own run of the same recipe (**10.65 s**, ~106 ms/call). The load-bearing claim remains the shape (linear in `N`, `N²` in total), which both runs reproduce. |
| J-5 (WARN — T-23a file) | **fixed** | T-23a named `run-stream.client.spec.ts` for a `watchFrame` validation that file does not contain — verified: it imports only `runStreamContribution`/`watchStreamContribution` and has 0 `watchFrame` matches. T-23a now points at the real codec (`packages/client/ui-refactory/src/client/run-stream.ts:83`) and is stated as a codec-shape assertion, adding no panel behaviour; the row notes why the old file was wrong. |
| J-6 (NIT — stale `G-7` note) | **fixed** | The stale "renamed in round 4 from the `G-7` label, which collided visually with round 3's `G-*` review findings — H-8" parenthetical is deleted from gate 7 (:1434 in revision 4), which now states the gate plainly and leaves the rename history to this log (H-8). `G-7` no longer appears in the gate's prose. |

**Incidental citation corrections (part of J-2's truthfulness scope):**
`live.py` line cites in §4.6 and R-1.6 that pointed a few lines above the code
they describe were corrected to the real spans — `_record` is `live.py:66-78`
(was `:57-65`, and `:61-64` for its non-object branch), the blank-line skip is
`live.py:357-359` (was `:55-56`), and `emit`'s boundary repair is `live.py:53-59`
(was `:34-38` in the "Writes are atomic per line" bullet). Round 3's historical
row (G-3) keeps its original `:57-65` as a record of what was true then.

**Round 5 — what did not change.** R-1.4 (`activity.has_more` SHALL be `true`
exactly when `activity.cursor < activity.total`), R-1.6.1 and every other
numbered clause of §2.2; §2.1's trace table; R-*/UC-*/SD-* semantics; the set of
UC-1…UC-12, F-1…F-12, T-1…T-26 and gates 1–8; and every normative statement
about the console, service or panel. No gate was weakened: gate 7 gained a
stricter, unambiguous assertion (the identity at `after == 400` *and* the
third-page reachability from 0), it did not lose one.

**Round 6** (SPEC_CHANGE_REQUESTED raised by the M1 build loop: `M1-4`).
The M1 implementation reached R-1.7a/R-1.7b's cost contract by introducing a
**third** derived file, `.refactory/events.index.overview.json`, beside the
strided `events.index.jsonl` and `events.index.provenance.json`. That collided
with §1.2, which named only two files in two places ("the only such files", "the
only new files anywhere in this spec"), so the built code was non-conformant
*with §1.2 as written* while being the way R-1.7b is actually met. **No numbered
clause of §2.2 was added, removed or renumbered for a new obligation**: the
overview line is R-1.7(b)'s own mechanism and is now cited as `[R-1.7b]`, the
existing clause, in §4.6's marker list. INT-001, DEC-001 and DEC-002 are
unaffected and `.hai/state.yaml` is untouched.

| Finding | Disposition | What changed |
|---|---|---|
| **`M1-4`** (SPEC_BLOCKER — §1.2 permits two derived files, the built code writes three) | **fixed by legalising the third file, on evidence** | §1.2's first non-goal now says "§4.6's three derived files are the only such files", and its second names all three explicitly (`events.index.jsonl`, `events.index.provenance.json`, `events.index.overview.json`) with the same rules (derived, disposable, never read as history), plus *why* one file cannot serve both lookups. §3's "minus the three derived files", §4.5's state-ownership table (new overview row) and §4.6 (marker list + `[R-1.7b]`; "Three derived files"; the real stride format; `total`/`last_event_at` from the overview line; trust-check 1 now trusting anchor and overview together-or-not-at-all; atomic writes naming both small files; disposability/reread rules) follow. UC-12(a), T-24, §11.9, SD-3, SD-11 and the §6 `page()` docstring now name the overview line. |
| — **why option B was rejected** (fold both values into the *existing* provenance anchor) | **measured, then rejected** | B loses three ways, all verified by execution: (1) *page path*: a strided page append needs a matching anchor rewrite, 200 records × 0.120 ms/rename = **24 ms** per page against **0.62 ms** measured; (2) *append path*: per-journal-append anchor refresh adds 0.120 ms to `live.emit`'s 0.059 ms/record (median) — **3.0×** *(later corrected to ~2×, see round-6 final pass)*, ~2.4 s over a 20,000-record run — and a second write per record contradicts R-3.4 and R-6.6's "this spec does not change how records are written"; (3) *impossible*: three independent processes append to `events.jsonl` (`live.emit`; `workers.event`, `workers.py:78-89`; `worker_observer.mjs:35`), only `live.emit` inspects the previous byte (`live.py:79`), so an in-process anchor can never track every append — a stale anchor would be normal and would answer a **wrong** `total`/`last_event_at`, breaking R-1.6. Recorded in §7.2 SD-3 with the table. |
| — **R-1.7b's mechanism wording was stale** | **fixed** | R-1.7(b) said both values "are the last trusted index line's `i` and `last_at`". That became false when R-1.7a's page seek made the index a **stride** of 64 records: the last line's `s` names the *first* record of the last stride (70 records → `s: 64`), so `total` is **not** `i + 1` of the last line, and each record's `at` is stored only as its **first 3 bytes** (`"202"`), with `null` as `nul`, so the newest `at` is not recoverable from the index at all. R-1.7(b) now names the overview line; **the obligation itself is unchanged** (`O(1)` per call, no walk, no re-parse). Verified by reading the index: 70 records → 2 lines, last line `s == 64`, `len(g) == 6`. |
| — **§4.6's index *format* description was also stale** | **fixed** | Revisions ≤5 described `events.index.jsonl` as "one line per complete record" with `{i, start, end, counted, last_at}`, `total` = "`i + 1` of the last trusted line", and `counted: true` per line — none of which the built strided format has. §4.6 and its trust checks, the atomic-write rule and the disposability section now describe the stride that exists: `_STRIDE_LINE = 1,053` bytes per 64 records, `{s, g[], l[]}`, offsets strictly advancing, `index_lines` cross-checked between anchor and overview (`live.py:224`), and the index file's size required to be exactly `index_lines × _STRIDE_LINE`. The index's own size is corrected from "60–140 bytes/record, ~16–38 MB at 269,900" to **16.5 bytes/record, ~4.4 MB**. |
| — **warm-cost numbers now in the spec** | **added** | §7.2 SD-3 carries the built-implementation table measured this round: warm 200-record page **0.681 / 0.653 / 0.674 ms** and warm `summary()` **0.099 / 0.097 / 0.098 ms** at `N` = **500 / 5,000 / 20,000**, `total` equal to the full-scan count in all three. Flat across 40×. |
| — **§1.2 non-goal audit (re-run)** | **passes** | (1) "Not a new *authoritative* store" — still true: the journal remains the only durable record; all three derived files hold offsets/one timestamp and none is read as history (measured: overview deleted/corrupt/empty, and all three deleted, leave `summary` and `page` **identical**; `OVERVIEW` has exactly one read site, `live.py:208`, parsed by `_indexed()` before `_trusted()`'s size/digest checks at `:257`/`:261`). (2) "Not a new bookkeeping file" — the exception is now an exhaustive list of **exactly three** named files, so nothing else is licensed; a fourth file would still be out of bounds. (3) "Not a rewrite of the worker/ledger/graph contracts" — untouched. (4) "Not a retention or deletion policy" — untouched: a rebuild rewrites only derived files, never the journal (DEC-001). (5) "Not a change to what counts as evidence of compliance" — untouched. (6) "Not the two-lane bounded console reads" — untouched, agrees with R-5/R-6. **No non-goal contradicts a requirement in revision 6.** |

**Round 6 — every ID whose *text* changed.** `R-1.7(b)` (mechanism sentence only;
the obligation is identical). §4.6 markers gained `[R-1.7b]`; §2's "where an
obligation is stated" list now includes it for §4.6. **No `R-*` was added,
removed, renumbered or weakened**; `R-1.6`, `R-1.6.1`, `R-1.6.2`, `R-6.5`,
`[R-6.5.2]`/`R-6.6`, `R-1.7a`, `R-1.7c`–`e`, `R-1.1`–`R-1.5`, `R-2`–`R-6` are
unchanged in substance. `UC-12(a)`, `T-24`, `SD-3`, `SD-11` and §11.9 were
extended to name the overview line; no `UC-*`, `T-*`, `F-*`, `SD-*` was added or
removed, and no gate (1–8) was weakened. §1.2 (both non-goals), §3, §4.5, §4.6,
§6's `page()` sketch and §7.2 SD-3/SD-11 are the changed sections.
**Round 6, second review pass — corrections after an independent FAIL.** The
first round-6 review returned FAIL with four `SPEC_BLOCKER`s and two WARNs, all
agent-fixable and all accepted:

| Finding | Category | Fix |
|---|---|---|
| S1 `live.py:208` is in `_indexed()`, not "inside the trust gate"; a shape-consistent wrong `last_at` **is served** | SPEC_BLOCKER | §4.6 now says the overview is parsed in `_indexed()` (`live.py:208`) *before* `_trusted()`'s size/digest checks (`:257`, `:261`), and a new "What checks 1–5 do and do not prove" paragraph states the limitation outright with the measured wrong-value case (400-record journal → `1999-01-01T00:00:00+00:00` served, journal answers `…09:00:39+00:00`). The §1.2 audit row no longer claims gate coverage it does not have. |
| S2 R-1.6 still said the value is "the newest trusted index line's `last_at`" — the very wording this round fixes | SPEC_BLOCKER | R-1.6's mechanism sentence now names the overview line (`[R-1.7b]`). |
| S3 SD-3's table was not reproducible: 24.1 MB was the H-6-retracted figure, and the timings were an unlabelled mean | SPEC_BLOCKER | Re-measured with SD-3's own sketch recipe (1,104 B/record → 22.08 MB at `N = 20,000`; measured 22.149 MB). New table gives **both** median and mean of 60 calls at `N` = 500/5,000/20,000: page **0.629 / 0.612 / 0.618 ms** median, `summary()` **0.089 / 0.099 / 0.098 ms** median; journal **0.553 / 5.534 / 22.149 MB**; `total` == full scan at all three. The retracted 24.1 MB is gone. |
| S4 the Option-B rationale quoted a non-existent clause ("no cost added to writing") | WARN | Quote removed; the argument now rests on R-3.4 and R-6.6's actual wording ("this spec does not change how records are written"). Factors corrected to the measured ones: rename **0.120 ms**, `live.emit` **0.059 ms** median → **3.0×** (not 2.8×) *(later corrected to ~2×)*, **~2.4 s** (not 3.9 s), page penalty **24 ms** vs a **0.62 ms** page (~39×). The decisive third argument is unchanged and was independently confirmed. |
| S5 "one `write()` on an `O_APPEND` descriptor" is not what `_write` does | WARN | §4.6 now describes the code: index opened `O_WRONLY\|O_CREAT` (`live.py:332`), `ftruncate`d to `kept × _STRIDE_LINE` (`:333`), `lseek` + one `write()` (`:334-335`), with no `O_APPEND` on the index (the only `O_APPEND` is the journal's, `live.py:76`) and the reason stated. |
| S6 `live.py:171-197` does not require exact line width | WARN | The rule is now stated correctly: a *line* may be short (`live.py:178-180`), exact width is enforced on the *file* (`live.py:257`), and the per-line parse/advance rule is cited as `live.py:180-194`. |
| S8 UC-1/F-11 "both derived files" | NIT | → "all three derived files". |
| S9 atomic-write subsection carried only `[R-6.5.2]` while §2 claims an exact marker set | NIT | Heading widened to name `[R-1.6.2]` and `[R-1.7b]`. |
| S7 `live.py:393-394` single blank line between top-level defs | WARN | **Not fixed — code, not spec.** The reviewer found a formatter nit in the uncommitted implementation. This round does not modify code; recorded here so the build loop can carry it. |

**Citation drift corrected (same pass).** The M1 build moved `live.py`'s line
numbers, so several pre-existing `live.py:N` cites in live contract text pointed
at the wrong code. Every cite in live text was re-resolved against the current
file: `events` `:44-66` → **`:482-503`**; `all_events[-1].get('at')` `:145` →
**`:112`**; the receipts glob `:124-125` → **`:589-590`**; `facts.json` `:154` →
**`:621`**; `emit` `:25-41` → **`:67-83`**; boundary repair `:53-59`/`:58` →
**`:76-83`**/**`:79`**; the double `events()` read `:122-123` → **`:587-588`**;
`_record` `:66-78`/`:71-77` → **`:86-100`**/**`:91-99`**; the blank-line skip
`:357-359` → **`:498-500`**; `events()`'s unterminated-line break `:357-358` →
**`:497-498`**; the `snapshot` two-pass note `:118-145` → **`:582-604`**.
Historical revision-log rows keep their original numbers as a record of what was
true then.

**Round 6, third review pass.** A second independent reviewer returned FAIL with
one `HUMAN_ESCALATION` (`NATIVE-1`) and two `SPEC_BLOCKER`s. Adjudicated here:

| Finding | Category as filed | Disposition |
|---|---|---|
| **NATIVE-1** — the spec's "today's answer" cites code that did not exist before M1 (`live.py:112` is the docstring of a *new* `_Coverage` dataclass), so passing the spec would ratify a redefinition of a human-visible value | HUMAN_ESCALATION | **NOT escalated — the premise is factually false, and I measured it rather than trusting either side.** Running **HEAD's own** `snapshot` (`git show cfd2d24:refactory/refactory_dsh/live.py`) over T-25's 15-case matrix returns `None` for every trailing-gap / newest-`{}` / gap-then-`at`-less case and the record's `at` otherwise — i.e. **HEAD agrees with R-1.6 in 15/15 cases** (HEAD vs round-3's walk: 8/15, so 7 divergences), because `_record` already converts a malformed complete line into a `journal_gap` with no `at` before `all_events[-1].get('at')` is evaluated. There is no redefinition to ratify and no human decision owed. The *real* defect NATIVE-1 exposed is narrower and was fixed: the rule's **evidence site** pointed at a new dataclass's docstring instead of shipped code. R-1.6 now quotes the HEAD expression (run `git show` from the worktree root) and records the 15/15 baseline measurement. |
| **`console.py` citations stale** — `:533` (a docstring; graded call is `:552`), `:1361-1364`/`:1362` (the `status` options are at `:1380-1392`), `:378` → `:376-381`, `:1802` (moved to `:1840` by M1) | SPEC_BLOCKER | **Fixed.** All four re-resolved and corrected in live text (§1.1, R-1.1, §4.6, §6, §8, §12 gate 7). |
| **§4.6 opener self-contradicts §2** — it said "normative clauses of this section are numbered", but §4.6 adds no numbered clause and carries markers only | SPEC_BLOCKER | **Fixed.** The opener now says obligations here are carried by **markers** and restates only clauses §2.2 already numbers. |
| WARN — `[R-1.7b]` is not a HEAD label; calling it "the existing clause" overstates | WARN | **Accepted.** §4.6's marker list now says the marker is *this spec's* short label for **R-1.7(b)** and that the obligation is R-1.7(b)'s, restated under a marker for citation convenience. |
| WARN — the log declared "no requirement in §2.2 changes" while R-1.6's mechanism sentence did change | WARN | **Accepted.** R-1.6's *mechanism wording* changed (index line → overview line); its **obligation** and its 15/15 equality with HEAD are unchanged. Stated plainly rather than glossed. |
| WARN — SD-3's 20,000 row needed an "uncached" caveat | WARN | **Fixed** — the row is now labelled as the zero-journal-read case. |
| NIT — `live.py:228` is a comment (code `:229`); `:178` is a comment; the 0.3 % direction | NIT | **Fixed** — `:229`, `:180`/`:178-179`, and the 0.3 % is now stated in the correct direction (22,148,890 B **exceeds** 22,080,000 B). |
| NIT — SD-4's "36 MB per poll" conflates payload and cap | NIT | **Not fixed — pre-existing**, unrelated to `M1-4`, and outside a narrow amendment's scope. Recorded for a later round. |

**Round 6, final review pass — PASS, and the polish it named.** The third and
final reviewer returned **PASS** (WARN/NIT only) and, independently, reproduced
the writer's rejection of `NATIVE-1`: loading HEAD's own `live.py` and running its
`snapshot` over T-25's 15 cases gives **HEAD == R-1.6's rule 15/15** and HEAD vs
round 3's walk **8/15** (walk diverges on exactly the 7 cases the spec lists). So
the amendment restates shipped behaviour, owes no human decision, and the
escalation correctly does not stand. The review's remaining WARN/NIT items were
then fixed here, since leaving them would make the log knowingly wrong:

| Item | Fix |
|---|---|
| R-1 arithmetic: "3.0×" append multiplier did not follow from the quoted operands (0.120 / 0.059 = 2.03) | → **~2×**, with the division shown. |
| R-7 typo: `22,149,829` was a digit transposition of the recipe's `22,148,890` | corrected. |
| R-4: "not achievable at all" was contradicted by the next sentence's own concession | retitled **"not achievable without a journal read per warm call or a changed append contract"**. |
| R-5: trust check 4 stated an upper bound (`total <= counted records`) the code does not enforce | check 4 now states only what is enforced (`size_at_write <= size`; `index_lines == ceil(total/64)`) and says explicitly that a forgery bounded only by internal consistency is served — cross-referenced to "what checks 1–5 do and do not prove". |
| R-2: `live.py:112` cited 3× as a code site, but it is a `_Coverage` docstring line | all three repointed to the shipped HEAD expression (`snapshot`, :145 of HEAD). |
| R-3: "today's" citations pointed at post-M1 worktree lines | now give the HEAD line and the worktree line each (`console.py:533`/`:552`, `live.py:118-145`/`:582-590`, `:124-125`/`:589-590`, `:154`/`:621`, `:122-123`/`:586-588`). |
| R-6 (NIT): `[R-6.5.2]` is called "the marker for R-6.6" while R-6 has six clauses | **The ambiguity predates this amendment** (the label and the six-clause R-6 predate this amendment and are internally consistent, as `[R-6.5.2]` is declared to be R-6's clause 6) — a naming tidy-up for a future round, not something a narrow amendment should renumber. |

**Routing: `SPEC_DELTA_READY`** — the amendment legalises what the built code
already does, and now describes it accurately rather than aspirationally. No code
change is required *by the amendment*; the one code observation these passes
produced (S7's blank line at `live.py:393-394`) is a style nit recorded for the
build loop, not a conformance gap. Every rule the amendment writes was verified by
execution against the uncommitted `live.py` (see the `M1-4` row and §7.2 SD-3).

**Round 7 — post-review corrections to shipped behaviour, and §12 made runnable
anywhere.** An independent code review of head `7ce4e33d` returned
`needs_changes` with eleven action items. Two of them were spec defects, and they
are corrected here; the rest are code, and the code now conforms rather than the
spec bending:

| Item | What changed |
|---|---|
| **R-4.3 was violated by the code, not by the spec** | `status()` fed one `after` to both `--after` and `--activity-after`, and `watchStream` advanced that number from the *journal* cursor, so on an optimizer run the journal index drove the progress delta and the streamed `status.events` emptied once it exceeded the progress file's length. The spec was right and unchanged; §6 already shows `status(run, after, activityAfter)` with `--activity-after: afterActivity`. The service now carries the two cursors separately, and `status` takes the journal cursor as an explicit argument defaulting to `after` for a first reader — no new Remote method, so SD-9 still holds. |
| **§12 gates 1–7 hardcoded absolute worktree paths, and gate 1 tested a file not on this branch** | the gate block is now worktree-relative (`cd "$(git rev-parse --show-toplevel)"`, `cd refactory && ../.venv/bin/python …`) so it runs verbatim in any checkout including a CI runner. Gate 1 no longer names `.hai/state.yaml` (absent from the branch, so unexecutable); it asserts what a reviewer needs — the branch carries these commits and is not behind `main`. Gates 2 and 7 also record the `DSH_HOME` clearing a developer shell needs, since the suite's seal fails a run that reaches the operator's real DSH home. |
| Gate 2's test roster | now lists T-27 and T-28, the two cases this round added. |

One deliberate deviation from §4.6 `[R-6.5.2]`'s wording, recorded here rather than
silently: that clause prescribes **write-to-temp-then-rename** for the provenance
anchor and the overview, and this round keeps the rename but makes the temp
*per-writer* (PID + a per-process counter) and deliberately **not** `O_EXCL`. A
per-writer unique name already guarantees no two publishes share an inode, which is
the property the rename needs; `O_EXCL` would add nothing and would turn a temp left
behind by a crashed writer (or a recycled PID) into a permanent `FileExistsError`
that wedges every later publish for that run. §4.6's *requirement* — a reader racing
a builder sees a whole number of lines and an anchor that is either the previous
version or this one — is unchanged and now holds without the collision that made a
racing reader rebuild from the journal.

Nothing in R-1…R-7 changed: the contract this document states was already correct
on every point the review raised, and the corrections are in the gates (§12) and
in the code, not in the requirements.
