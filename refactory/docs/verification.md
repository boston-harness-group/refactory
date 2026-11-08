# Verification

The repository stopped copying the DeepSeek Harness, and the gates that came with the
copy — `pnpm run test:docs` and its fifteen checks, the snapshot suite, the bilingual
pairing gate, `benchmarks/`, `website/` and `.agents/` — went with it. This document
decides what verifies the eight packages and the glue instead, and what a contributor runs
before pushing. It is the answer to issue #84, written before the deletion.

## The commands

Run everything from the repository root.

```bash
uv run python -m pytest refactory/tests -q                    # the glue's suite
uv run python -m pytest refactory/tests -q --run-integration  # ... plus the real-DSH runs
uv run python -m pytest refactory/tests -q --toolchain-gate   # which real tools ran, and which did not
uv run python -m refactory_dsh.tools.gen_api_reference --check # docs/api/reference.md matches the code
uv run python -m refactory_dsh.tools.gen_tutorial --check      # the tutorial matches the code
uv run python -m refactory_dsh.tools.check_recipes             # every documented command, executed
```

```bash
pnpm run typecheck             # tsc over the eight packages, host face then client face
pnpm run build                 # typecheck, then the host and client bundle passes
pnpm run test                  # the eight packages' vitest suites
pnpm run test:gui              # the client packages' suites under jsdom
pnpm run lint                  # oxlint
```

## What replaced each deleted gate

| the deleted gate | its replacement | what the replacement proves |
|---|---|---|
| `pnpm run test:docs`, fifteen harness documentation checks | `refactory/tests/test_ship_ready.py`, `test_docs_api.py`, `test_docs_tutorial.py`, the two generator `--check` commands, and `check_recipes` | every command a shipped document shows runs as written, every path it names exists, and the generated API reference and tutorial are current |
| the bilingual pairing gate | `refactory/tests/test_docs_english_only.py` | no document under `refactory/` or the eight packages has a `.zh.md` side or a `README.i18n.yaml` that nothing keeps aligned |
| `pnpm run test:gui` over the harness client suites | `pnpm run test` and `pnpm run test:gui` | the eight packages' own vitest suites pass, including the two client packages under jsdom |
| the recorded-session snapshot suite | the real-DSH runs: `pytest --run-integration` and `check_recipes` | a mode runs end to end on the pinned harness and leaves facts on disk |
| `benchmarks/`, `website/`, `.agents/` | none | removed with the harness; this repository publishes no benchmark gate, no documentation site, and no agent-note corpus |

The harness's per-file 100% coverage gate is also gone. `pnpm run test:coverage` runs the
eight packages' suites with coverage, and no threshold is enforced, so coverage is a number
a contributor reads rather than a gate that fails.

## The web surface and the pins it needs

`refactory ui` boots the re:factory-owned `refactory` profile, which is initialized
from the pinned harness's `web` template. The web surface is a refactory-owned bundle, and
one pnpm patch in `patches/` supplies a model-route change, declared under
`patchedDependencies` in `pnpm-workspace.yaml`:

- `@deepseek-ai/dsh-refactory-web` is the browser surface layer. It mounts the
  `refactory-console` Host service, the `ui-refactory` right-sidebar picker and the
  `ui-brand-refactory` brand occupants, disables `ui-brand-official`, and keeps the
  `wf_next` exclusion on the repeat-tool-reminder row. The client-module system
  discovers a browser plugin by scanning already-mounted Loader entries, so a
  package that declares `dsh.client` and is an active entry is picked up without
  editing the upstream `@deepseek-ai/dsh-web-app` manifest. Its manifest declares
  our occupant packages — and `@deepseek-ai/dsh-workflow-spine`, which the `ceo`
  preset mounts — so the loader resolves them the way the deleted pnpm patch did.
- `@deepseek-ai/dsh-headless` accepts `--provider` and `--model` and makes a named
  selection the model-selection middleware's current one, so
  `refactory_dsh/model.py`'s completion client uses the model a caller named.

One dependency is vendored rather than patched: `vendor/node-addon-system` replaces
`@deepseek-ai/node-addon-system` through an override, because its `flock` generated a
diagnostic report to tell glibc from musl, and a report aborts the process when a
socket is closing (DSH workers died with `exit -6` mid-step). A pnpm patch did not
reach installs whose store already held the package. Its README says when to remove it.

The `ceo` preset is **not** a patch. `packages/bundle/refactory` declares it as an
inline `@deepseek-ai/dsh-agent-preset` row in `presets/ceo.patch.yml`, the way the
harness declares its own shipped presets, and `cordis.patch.yml` makes it the registry
default. At this pin no preset tree is discovered and no configured `roots` are
involved. The declaration mounts `tool-refactory`, the re:factory persona, and the
**interactive workflow spine**. The spine is mounted with `interactiveOpen`, so the
chat's CEO can open a mode with `wf_open` and drive it with `wf_next`; it runs one
deterministic node per call, refuses a caller that is not the session that opened the
run, and answers a `user` gate through `userQuestions`. The spine's own guard denies
the CEO's coding tools, so the CEO cannot implement a node by hand, and mounting it
inside the preset is what keeps its tools and that guard on the preset's own agents: a
host row reaches every top-level agent on the host. The CLI's `graph` profile does mount
it as a host row: there the only top-level agent is the CEO its runner boots. The patch
targets the `agent-preset-registry` row that the `dsh-web-app` layer inserts, so this bundle
must be applied after `@deepseek-ai/dsh-web-app`; `dsh plugin add` appends a newly
installed bundle after the profile's template bundles, which is what an ordinary profile
install relies on.

The console resolves `repoRoot` the same way: its row reads
`ctx.refactoryWebStartup.repoRoot`, which the `refactory-web` bundle's startup plugin
resolves from its own location. `process.cwd()` would name whoever launched the GUI,
not this checkout.

Two DSH profiles are machine state this checkout owns, and both are created by
`./install.sh` (or healed on boot by `./refactory-dsh`): `graph`, which the Python glue
runs modes on, and `refactory`, which the GUI boots. `refactory` is **not** the
harness's `web`: it is initialized from that shipped template with `dsh --profile
refactory --from-default-profile web --dump-config`, so the harness's own `web`
profile keeps exactly its template bundle list and a plain `dsh --profile web` session
sees none of re:factory's preset, console or brand rows.

`dsh_profile.py` owns a layer table (`graph`, the surface-independent `presets`, the
browser `web`) and a target table over it (`graph`, `refactory`, and the attached
`martty`), emits the ordered `init`/`add`/`remove`/`clean` plan that `install.sh` and
`./refactory-dsh` both execute, and detects a harness profile an older install
mutated. `refactory doctor` reports one row per target and names the exact command for
whichever layer is short — `refactory` without the preset layer has no `ceo` preset
while the composition still names `ceo` as the roster default, so every session the
GUI creates without naming one would stop with `agent-preset/not-found`; without the
web layer the console and picker rows never mount, so the GUI boots with no refactory
panel. A harness profile an older install polluted is reported with its migration,
`dsh plugin --profile web remove @deepseek-ai/dsh-refactory
@deepseek-ai/dsh-refactory-web`; doctor states it rather than rewriting machine state.

A profile manifest carries two independent lists, and the plan reads both. `remove`
names only packages the manifest also declares as dependencies, because `pnpm remove`
removes nothing at all when one of its names is missing from that list — the whole
call fails, and under the shell scripts' `set -e` every later row is skipped, which
is how an unsafe manifest order used to leave the launcher unable to boot. A selection
that is no longer a dependency is reachable by neither `pnpm remove` nor the harness's
own `reconcile()`, which keeps a name absent from `dependencies` rather than dropping
it; the `clean` row is the manifest rewrite that clears it, and it runs before the
profile's plugin rows so a stale entry cannot make their reconcile throw. The rewrite
holds the same writer lock `dsh plugin` takes around every profile mutation, and
writes the pre-image before replacing the file, so it cannot overwrite a concurrent
plugin operation. It resolves the lock helpers through the selected pinned harness
and its plugin manager. Backups remain beside the manifest as
`package.json.refactory-backup-<uuid>`, with mode `0600`; provisioning reports their
paths. Cleanup preserves unrelated fields and the order of all retained selections.
Desired layers missing a dependency entry receive an `add` row even when already
selected. Obsolete dependencies are removed on retry even if an interrupted run
already cleared their selections.

`refactory/tests/test_dsh_profile_migration.py` executes the real launcher in
isolated homes for all four retired-profile states, selected-layer dependency
repair, interrupted retirement and fresh provisioning. It checks ordered final
bundle lists, retained unrelated state and backups, two successful runs, and an
empty final plan. Its lock test uses a readiness handshake with the pinned writer
lock: both cleanup and the pinned plugin operation must respect the held lock, and
cleanup must preserve changes committed by that writer. These tests run in the
default Python suite when Node, pnpm and the pinned install are present, including
CI after `install.sh`.

An installation that predates the `refactory-web` → `refactory` profile rename and the
`@deepseek-ai/dsh-refactory-presets` → `@deepseek-ai/dsh-refactory` bundle rename is
migrated by that same plan: re:factory's layers are cleared out of the directory the
profile rename left behind, and the retired name stops being selected in each profile
an older install left it in. `doctor` reports a harness-owned profile an older install
polluted with the full set it carries, retired name included, so the command it prints
names everything the operator has to clear rather than only the surviving layer. The
old directory is left on disk — a DSH profile cannot be moved, because its
`node_modules` holds absolute symlinks under `.dsh-module-fallback` — and what remains
there is an inert copy of the shipped `web` template that nothing boots. re:factory
never mutates an attached profile, so under Martty the removal is only reported, as a
`doctor` row naming `dsh plugin --profile martty remove
@deepseek-ai/dsh-refactory-presets`.

The second surface is **attached**, not owned. re:factory cannot recreate Martty's stack
(`dsh-base` + `martty` + `martty-balance`, the last a machine-local link), so it never
creates or mutates that profile. The documented path is to add the surface-independent
layer to the profile the operator already has: `dsh plugin --profile martty add
<checkout>/packages/bundle/refactory`. That is the preset and the workflow tools
only; `@deepseek-ai/dsh-refactory-web` is a browser surface and is not mounted under
Martty. `refactory doctor` reports the missing layer once `martty` exists, and stays
silent about the profile otherwise.

`pnpm run test:acceptance` runs that plan into a throwaway home, checks the harness's own
`web` template is exactly as the template defines it and mounts none of re:factory's rows,
then boots `refactory` and checks the preset reaches the roster as an inline declaration,
including that the installed `@deepseek-ai/dsh-base` and `@deepseek-ai/dsh-web-app` layers
declare no `ceo` preset of their own, and that the web bundle mounts the console, picker
and brand rows with a `repoRoot` resolved to the checkout (`refactoryConsole/project`) and
a live mode catalogue (`refactoryConsole/modes`). A scripted model route then makes the CEO
spawn a subagent through `subagent` and through `subagent_fork`, on `refactory` and on a
fresh profile carrying the `graph` layer, and checks each child's own tool catalog: the
delegation rows' allow-list, with no shell and nothing that starts a run. The `graph`
CEO's own catalog is checked the same way as the GUI CEO's: the `wf_*` tools, nothing its
`ceoDeny` names, and no shell, `write`, `edit`, `workflow` or `ralph`.

The gateway discovers the console at runtime from the `@Remote` markers on the mounted
service and binds arguments by the method parameter names, so the console answers without
the generated `./typert` and `./remote` artifacts. The picker's client face sends those
same names through the Connection's generic RPC,
`connection.rpc.call('/api', 'refactoryConsole/<method>', { args })`, because the pinned
Client Remote assembly is a prebuilt bundle that mounts a namespace service only for the
harness's own packages; `ui-refactory` therefore injects `connection`, not
`remote.refactoryConsole`. `refactory-console/src/types.ts` states the client face, and
`ui-refactory/tests/remote.client.spec.ts` pins each endpoint and argument name. The
published Typert generator emits the artifacts only when its own protocol package is a
workspace project, which a pinned install does not have.

### The harness that serves the profile

`refactory doctor` compares the harness this repository pins with the `dsh` on `PATH`
(`REFACTORY_DSH` overrides the resolved launcher) and names both versions when they
disagree. When the versions match it also compares the strict `TypertCodec` face each
harness's `@deepseek-ai/dsh-typert-protocol` declares. That second comparison is the
point: the harness fork's tree and the published `0.1.6-alpha.1` both report
`0.1.6-alpha.1`, so a version check alone was blind to `e459e32637` changing `schema`
to `create()`. `pnpm run test:acceptance` boots the pinned launcher; this check is what
notices that the operator's `dsh` is a different tree. `python -m
refactory_dsh.harness_check` runs it on its own and is the CI step on a plain clone,
where no separate `dsh` exists and the check is green.

The loader's boot audit reports a failed activation as a count and a package name
(`1 entry did not activate`) without the cause, which is how the incident stayed
invisible until the browser console was read. That is upstream loader behavior and is
out of scope here; this check is the downstream alternative. Recorded on #279.

## The stream, the live session and control

A run says what it is doing while it does it, and that single stream is what every
surface reads.

- `refactory_dsh/events.py` owns the event vocabulary and the two levels: the meaningful
  events a watcher gets by default, and verbose, which adds every remaining event and a
  candidate's own output. `tests/test_events.py` pins the filter.
- The outer loop prints meaningful events as they happen; the final JSON stays on stdout
  and `progress.jsonl` stays the durable record. `tests/test_loop.py` asserts a watched
  run prints, a library run does not, and verbose is a superset.
- `run_on_dsh` tees a candidate's child output while still collecting it, because
  `best_score` is parsed out of the runner's last JSON object. `test_dsh_profile.py`
  pins both from one stub child.
- `ceo` is a live session by default: the graph profile is asked for `--chat`, which
  prints each turn and reads the person's lines between them; `--headless` builds the
  older one-shot argv and nothing else differs. `test_interactive.py` pins the argv
  equality, and a real terminal run is the by-hand check.
- Control travels as its own request, not as prose: `control.py` owns a small file in
  the run directory, the loop honors pause/resume/stop at the generation boundary, and
  `refactory steer --pause/--resume/--stop`, `refactoryConsole.steer` and the panel's
  Steering buttons all write the same request. The intent ledger is the other channel:
  the proposer reads it fresh before every generation, so a note steers the next one.
- The same boundary carries the three population acts, and they are one-shot: `keep`
  pins a candidate into `select_parents`, `reject` (with `--lineage`) takes it out of
  `leaderboard` so it cannot be proposed from, reported best or adopted, and `adopt`
  writes the current winner as a package without ending the search. Rejection is a
  selection fact rather than a rewrite of history — `state.json` keeps the score — and
  `adopt.best` reads the same journal so the two never name different winners.
  `tests/test_control.py` pins each act, the consumption of the request, the lineage
  walk, the resume replay and the adopter's agreement with the report.

## What a contributor runs before pushing

Change the glue or a document: `uv run python -m pytest refactory/tests -q`, then the two
generator `--check` commands and `check_recipes` when a document, a CLI verb, or the API
reference changed.

Change one of the eight packages: `pnpm run build`, `pnpm run test`, and `pnpm run lint`.
`pnpm run build` is what puts `lib/` on disk, and the graph profile loads `lib/`, so a
change that is only typechecked has not reached a run.

Change anything that reaches DSH: `uv run python -m pytest refactory/tests -q
--run-integration`. That is the only check that drives the pinned harness rather than the
in-process walker. `refactory package smoke` does not touch DSH; it is the walker.

Change a patch: `pnpm patch <name>@<version>`, edit the printed directory, then
`pnpm patch-commit <directory>`; the patch file and `pnpm-workspace.yaml` change together.
A patch is applied by `pnpm install`, so verify it in a clean clone, not only here.

## How a document's claim is checked

`test_ship_ready.py` reads the `bash` blocks a shipped document shows and refuses a module,
a flag, or a path that does not exist. `check_recipes` executes
`refactory_dsh/tools/recipes.py`, the one list the HTML tour also renders, so a command the
documentation shows is a command that was run. A claim about a run's output is checked the
same way the documentation's reader checks it: by running the command and reading the file.

## What this does not cover

The web picker has no automated browser test. Its layers are checked by hand against a
running server: the boot manifest lists the client rows, the page reports no
`Failed to load plugins` entry (a client plugin parked on a service nothing provides is how
one shipped), `refactoryConsole/modes` over `/api` returns the catalogue, `session/create`
with no preset returns `agentPreset: ceo`, and expanding the right sidebar and pressing
**Modes** renders the picker. `ui-refactory/tests/remote.client.spec.ts` covers the call
mapping, so a wrong endpoint or argument name fails a test; a future change to the
refactory bundles or to the harness's client assembly can still break the page without a
red test.

The in-session drive is covered by `workflow-spine/tests/open.spec.ts` (open claims the
run, one node per `wf_next`, the refusals, the post-drain open view, a data pass crossing
calls, and the configured-graph path unchanged) and by hand in a browser: the CEO calls
`wf_open` then one `wf_next` per node, no `bash` call appears in the transcript, the run
reaches `done` with every selected item graded and the graph epilogue completed. A data
pass retains its materialized item across agent handoffs on both configured and
interactive graphs; `Task.setup` runs once per item. A completed exit or terminal
refusal is closed and graded before a public terminal status is returned, so refusal
can contribute a failed grade without silently losing the remaining dataset. An explicit
halt without a forward edge, or a reloop cap exhausted at a gate whose `on_exhausted` is
`halt`, instead stops the whole run: its interrupted pass remains inspectable, is not
graded, and does not start the next item. The terminal response carries `halted` with
the gate and reason, and the status file records `halted`. A gate in a pass that does not
declare `halt` ends only its item when its cap runs out: `continue` (the default) goes on
along `proceed` and the item is graded as it stands, with the gate recorded in
`gate_exhausted`; `fail_item` (the default when the gate has no forward `proceed` edge)
grades the item failed through `end --failed`. Either way the next item starts.

An active run still holds one exclusive claim. After completion or halt, its owner
can inspect it until another successful `wf_open` replaces the claim; the next run
may belong to the same session or another one. Automatically derived work directories
are unique, so reopening the same mode preserves earlier artifacts. The detached
search (`refactory start`) is untouched. This is in-process lifecycle behavior, not
restart/resume persistence.

The Typert Remote faces are kept in step by hand. The generated artifacts would make that
impossible to get wrong, and issue #85 carries why they are not generated.

### Cursor integrity

Completion and gate verdicts must name a node in the current cursor. A rejected
completion or invalid verdict leaves the cursor and its history unchanged; a past
completion does not authorize running that node's successor again. The data-pass
runner alone restores the enclosing data continuation after grading its final item.
When the composition runs commands, the CEO cannot certify a deterministic node or
automatic gate with `wf_done` or `wf_verdict`: `wf_next` runs its actual operation or
evaluator. A literal `halt` on a currently due automatic gate with no forward halt
edge is recorded as cancellation, not a measured gate result.

Explicit prompt edits, skip/unskip and redo remain steering actions. To recover a
halted run, redo a completed agent deliberately; a verdict cannot implicitly reopen
a halted gate outside the cursor. Restart/resume and pausing remain separate contracts.

### User-gate decisions after a revision

A terminal or GUI answer authorizes one gate visit against the exact declared review
inputs. Returning after an upstream agent is redone, or changing an input while a question is pending,
requires a fresh question. The intent ledger keeps the old answer as history; it is
not reusable authorization. `console steer --verdict gate=value` queues a one-use
answer for the current lowered gate inputs, and expires if they change before use.
Explicit `ceo --approve` and Python/authoring `verdicts` retain their separate meaning:
standing upfront authorization for that run. Old records without an explicit scope
are displayed but cannot prove current approval.

For actual-use validation, run the shipped `review_pack/review`, answer its `approve`
gate, then open the mode again in the completed run directory. Confirm a new question
appears even when `draft.md` is unchanged. Changing `draft.md` while that question is
pending must refuse the old answer without routing, then allow a fresh review and
answer. Validate both terminal and native browser channels; no substitute mode is
needed. Also run with explicit `--approve approve=proceed` to verify unattended
upfront authorization still works.


## Bounded release acceptance

Agree on the feature, exact success conditions, budget and stop condition before model-backed acceptance. Test both CLI and GUI for a feature exposed in both; API success alone is not evidence of native GUI behavior. Record actual provider/model/backend and the revision under test.

Separate framework execution and control, declared artifact checks, and generated-output quality. Reuse saved artifacts for unchanged routing/reporting contracts, and reserve fresh model runs for behavior that needs them. A timeout is an incomplete run, even when its failure reporting is correct. A no-improvement search is not a framework failure.

Documentation-only changes need link/command checks, generated-reference freshness when relevant, and visual inspection of changed graphics; they do not require rerunning authoring campaigns. Keep normal commit/push checks. Do not turn a wording correction into an unbounded model-quality exercise.
