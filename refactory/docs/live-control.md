# Live activity, control and recovery

A run directory is the durable identity shared by the terminal, native panel and owning runtime. Attach a viewer using the exact returned directory. Opening, closing or reconnecting a viewer does not execute work or cancel the run.

## Read live work

```bash
uv run refactory status --run /absolute/run --follow
uv run refactory runs
uv run refactory steer --run /absolute/run
```

The native Modes panel offers **Runs and pending questions** and **Existing run directory → Read this run**. Attached views stream persisted changes. Activity includes due/completed nodes, last event time, wait reason, worker receipts, requested and observed routes, available usage, instruction-delivery state, recorded outputs, **run warnings** and **owner process liveness**. No completion percentage is estimated from elapsed time. A missing observation remains unavailable.

The status includes `owner_alive` (true when the checkpoint's owner process is responding, false when it has exited, absent when no owner is recorded) and failed graph control receipts visible in the activity's `control` field. `refactory status` prints whether the owner process is alive and, when the last graph control failed, a line naming the failure.

`spine-status.json` now includes a `running_nodes` array naming each executing node and its kind (`agent`, `operation`, `gate`, `control`). The `wait_reason` field names the running node(s) and their kinds — e.g. `"waiting for worker delivery: research_director (agent)"`.

Each usage event in `.refactory/events.jsonl` carries `tokens_delta` (the token count of the single request that triggered this event) alongside the cumulative `tokens` total. Summing `tokens_delta` across events for one worker gives its total; summing `tokens` across events overcounts.

The authoring status includes `step_tokens`: per-step token totals computed by summing worker receipts by `node_id`. A step with retries sums to one entry. `total_tokens` is the sum across all steps.

Graph activity is appended to `.refactory/events.jsonl`; checkpoints, worker receipts and the resource ledger live beside it. Historical runs may lack newer records. An interrupted journal write is reported as a gap rather than fabricated progress. A worker context being prepared, a dispatch being observed, and an output satisfying a correction are different facts.

## Questions and ownership

Pending user gates retain their prompt and accepted verdicts.

**Live mode (no `--headless`).** When a run reaches a user gate, the terminal shows the gate's own prompt (including the delivery line, with the run's facts filled in), the accepted verdicts, and asks the person to pick one. After choosing a verdict, the person may add an optional multi-line note (ending with a line containing only `.`); the note is recorded as steering in the intent ledger — a correction with reloop, a constraint with proceed.

**Headless mode.** A headless DSH run stops at a user gate and prints the exact commands to answer it later. A walker run (``python -m refactory_dsh.design`` etc.) stops with ``GateStop`` and prints the same commands:

```
refactory: <gate> is waiting for a person.
  <the gate's prompt>
  Answer it with one of <verdicts> (add --note "..." to say why):
    refactory steer --run <dir> --verdict <gate>=<verdict>
  Then continue the run:
    refactory graph-control resume --run <dir>
```

You can also answer via the native panel, the owning CEO session, or an explicit verdict:

```bash
uv run refactory steer --run /absolute/run --verdict gate_name=proceed
```

The allowed verdict comes from that gate. The answer is scoped to the run and invocation, not to whichever chat is visible. Reading or answering a persisted question does not grant a second session permission to drive the graph. Each CEO session owns its workflow; concurrent runs need distinct work directories. A native question left pending after a disconnect stays pending until a valid answer reaches the owner. A restarted owner must pass the recovery checks below.

### Steering note kinds

Steering notes added with `refactory steer --kind info` record context that agents
see but the audit does not require evidence for. Use `--kind correction` or
`--kind constraint` for asks the audit must cover.

Use `--scope process` for notes about the run itself (e.g. "add a row to the
audit table", "record timestamps in a specific format"). A process note remains
visible to agents and is recorded in the evidence payload, but does not become
an auditable requirement — because the audit cannot cite itself as evidence for
a directive about its own behavior. Process notes are reported on in a separate
non-blocking section of the intent audit and appear in the run report and pull
request, so a person sees each one and whether it was followed.

### Steering sources

A steering note carries a `--source` (`user-relayed` or `ceo`).

- **`user-relayed`** (the CLI default) marks the person's own words. These are
  auditable requirements.
- **`ceo`** (the `wf_steer` tool's default) marks the CEO's own observations.
  These are context the agent sees, not auditable requirements. They render
  under "CEO observations" with an explicit non-auditable disclaimer.

When the CEO relays the person's words through `wf_steer`, it must set
`source: "user-relayed"` so they become requirements the audit checks. Without
it the words are silently treated as CEO observations. Legacy ledger entries
without a source annotation default to `user-relayed`.

```bash
uv run refactory steer --run /absolute/run --note "fix the padding" --source user-relayed
```

## Mid-turn messages

When the CEO is mid-turn, a message you type is delivered immediately as
steering rather than queued until the turn finishes. The refactory-web
composition sets the conversation plugin's `busyEnter` to `steer`, which
overrides the upstream default (`queue`). DSH settings resolve
"schema defaults → composition base → user layer" (dsh-settings types.d.ts),
so a person's saved Enter-behaviour choice in the GUI still wins.

To change the behaviour, open the conversation plugin's settings (found in
the GUI under **Settings → Conversation → Enter Behaviour**) and choose
**Queue** or **Steer**. The saved choice overrides the composition default.

DSH's `resolveSubmitMode` sends the saved preference on plain Enter and its
opposite on the Cmd/Ctrl chord.  With `steer` as the composition default,
**plain Enter steers** (delivers the message immediately as a steering note)
and **Cmd+Enter** (macOS) / **Ctrl+Enter** (other platforms) **queues** the
message until the turn finishes.  A saved choice of **Queue** reverses this:
plain Enter queues, Cmd/Ctrl+Enter steers.

## Apply a control at its boundary

Optimizer notes, keep/reject and pause/resume/stop apply at generation boundaries. Graph controls apply at the next graph boundary and can wait for admitted work to settle. External workers accept new instructions at a subsequent node boundary, not as an in-flight prompt edit. A queued control receipt is a request, not confirmation of application.

The native panel's **Graph controls** offers pause/resume/stop. The console exposes the same operations:

```bash
uv run refactory graph-control pause --run /absolute/run
uv run refactory graph-control resume --run /absolute/run
uv run refactory graph-control stop --run /absolute/run
uv run refactory graph-control skip --run /absolute/run --node check
```

- `pause --node NODE` arms the pause before that node only: every node before it runs, and a route that never reaches it never holds.
- `stop` on a paused run whose owner has exited starts an owner, as `resume` does, and that owner applies the stop before it dispatches anything. The command reports `applied` once the run reads `halted`, and `stopping` if that takes longer.
- When a control needs an owner and recovery refuses to start one (`stop` on a paused run, `rerun --acknowledge-failure`, `amend-command --acknowledge-failure`, `redeliver`), the control is withdrawn, not left queued, so a later owner never applies it unasked. The refusal says what to do, typically `acknowledge` first.

Process cancellation is separate: `refactory stop --run RUN_DIR` requests interruption of the run's current owner process (from the console launch record, or from the spine checkpoint when the launch record is absent or stale, matching how `refactory status` determines `owner_alive`). Inspect the receipt and subsequent activity for confirmation; sending a signal is not proof that every worker terminated. Stopping a reader does not issue either kind of stop.

A control that cannot be applied — a stale revision, an unsupported action, a node that is not safely replayable — is recorded with `status: failed` in the receipt. The failure reason is also journaled as a run warning, visible in `refactory status` and in the receipt shown by `refactory graph-control`'s own output. A failed control stops being reported to the person once a later control on the same run succeeds, because the receipt file is overwritten with `status: applied`. The failure remains in the journal for audit; only the display is replaced by the later success. The owner removes the control file once it applies or refuses the control. A deferred rerun stays queued until concurrent work joins. `refactory status` lists the queued controls, so a person sees what a next owner would apply. Every graph-control confirmation includes the run's name.

To change completed work, choose a completed node and prompt correction, press **Preview affected results**, inspect the invalidated nodes, then **Confirm correction and rerun**. The console equivalent is `graph-control preview --node NODE`, followed by `graph-control rerun --node NODE --text "CORRECTION" --revision REVISION` with the returned revision and the same `--run`. The owner refuses a stale preview. Prior evidence is archived and dependent results invalidated before execution; a note alone does not silently rewrite completed work. Unsafe replay plans, including unsettled concurrent work, are refused.

The revision is the graph revision the spine stores with its checkpoint. It covers the graph's structure and progress, not its owner, status or pauses, so an acknowledge, a pause or an owner restart keeps a preview current; the owner recomputes it from the live graph to compare. A rerun applies behind a person's pause, and the run then holds before the reopened node until it is resumed. The CEO's own redo still refuses while a person's pause holds.

## Resuming after the driver moved

When a run's driver checkout moves to a new commit (or its fingerprinted files are
edited) between launch and resume, the run creates a **pin**: a prepared git
worktree of the original commit under `~/.refactory/pins/<commit>/checkout`. The
pin has its own `uv sync`, `pnpm install` and `pnpm build`, and a DSH profile
named `graph-pin-<commit12>` provisioned in the person's own `$DSH_HOME` — so
credentials stay where they are and DSH_HOME is never changed.

The resume path detects drift by comparing the recorded commit and configuration
fingerprints against the current checkout.  Only fingerprints under the driver
checkout (by path component, not string prefix) count as driver drift; the run's
own graphs, `.factory/` artefacts, and project packages are not driver drift
and are still validated normally — a changed run graph raises the same "prepared
configuration … changed" refusal as before.

When drift is found, the owner process re-execs under the pin's Python
interpreter. After re-exec, `REPO_ROOT` resolves to the pin, so every
path — `dsh_command()`, `venv_bin()`, the node commands — naturally points
to the pin's code. The loop-breaking guard: if `REPO_ROOT` already equals
the pin checkout, re-exec is skipped.

A dirty driver (uncommitted changes to machinery paths at launch) records a
unified diff. When the pin is created, `git apply` applies that diff to the pin's
worktree, so the pin matches the exact state the run was launched with. A
conflicting diff refuses with a message naming the commit and the conflict.

Pins are shared: two runs at the same commit use the same pin. `refactory pin
list` shows all pins; `refactory pin prune` removes those no live or resumable run
refers to.

## Which project a run uses

`wf_open` passes `session.header.cwd` as `--project` to the open command, so a
session opened in another directory runs with that directory as its project.
`startup.ts` reads `REFACTORY_PROJECT` from the environment when set and falls
back to `process.cwd()` only when absent, so the CLI's `--project` is no longer
overridden.

## Recover a known boundary

The run's status includes whether its owner process is alive. A failed run with a live owner may still apply queued controls and does not need recovery — wait for it or send another control. A failed run whose owner is not alive cannot apply pending controls; explicit recovery (resume) is needed to start a fresh owner from the saved checkpoint.

### Retry status (dm-95)

When a node command fails, `spine-status.json` records status `ready` with a `last_failure` field containing `{ node, error, at }`, not run-level `failed`. The graph is ready to retry the same boundary. The CEO-facing failure payload in `wf_next` uses `node` for the node id, `command` for the command string, and `error` containing the `<id> exited with code N` prefix followed by the real reason from stderr. `refactory status` shows "step X is being retried" when `last_failure` is present and the owner is alive. Run-level `failed` means the graph run has ended — it appears only for infrastructure errors (an uncaught exception from the graph's drain loop) or when the owner exits without retrying the failed node.

When a step runs successfully, `wf_next`'s `ran` field carries a facts summary instead of the full `facts.json` object. The summary is `{ _summary: true, path: "facts.json", keys, sizes, inline }`: values whose JSON serialization is ≤ 512 characters are inlined in `inline`; larger values are reported by key and byte size only in `sizes`. The 512-character limit keeps short strings, numbers, and small arrays visible while excluding large blobs (a design run's `facts.json` is about 37 KB). The CEO's coding tools are disabled while driving a graph, so the full value of a large fact is not accessible through `wf_next`.

When the owner exits with status `ready` and `last_failure` present, the supervisor promotes the status to `failed` before deciding whether to respawn. The promoted failure names the node and its error. The supervisor does not respawn it: the run genuinely ended because a node failure was not retried. It still respawns a real crash (an owner killed or dead while `executing`).

Consumers that need finality should use the `done` or `failed` status (which now unambiguously means the run ended), not a transient status during execution.

### Resume a failed due step (dm-97)

When a node fails and its checkpoint records status `failed`, resume requires acknowledgement:

```bash
uv run refactory graph-control resume --acknowledge-failure --run /absolute/run
```

The acknowledgement is bound to the checkpoint's SHA-256 hash, so it cannot apply to a later failure. Without the flag, the refusal names this exact command. The failed node is re-dispatched from the saved checkpoint.

For node command failures that wrote `ready` (dm-95), a plain `resume` works after promotion because the promoted `failed` checkpoint requires `--acknowledge-failure`.


A live owner's resume control releases its boundary pause. If the old graph owner is gone, explicit resume can start a fresh owned process from the saved checkpoint. When the run's status is terminal (done, halted, failed, refused, or cancelled) and the recorded owner is still alive, resume waits briefly for it to exit. If the owner exits during the wait, a fresh owner is started automatically. If the owner is still alive after the wait, the resume is filed as a pending resume (`.refactory/resume-pending.json`) and honoured automatically when the old owner finishes exiting — no second command is needed. It preserves the run identity, completed work, route configuration and consumed resources. Larger resource limits are explicit allocations, not a reset of usage. `refactory graph-control resume --run <dir>` is the recovery command:

```bash
uv run refactory graph-control resume --run /absolute/run
```

It internally runs the interactive entry point with `--resume`.

Recovery requires a recognized checkpoint, a stopped owner, matching prepared configuration and unchanged fingerprinted inputs/outputs. It refuses a dispatch whose outcome is unknown — a pending marker with no written receipt; a refused step whose receipt was written is not unknown. It also refuses a failed or unknown last-node outcome. In-flight external product sessions are not resumed. Python executor partial-fork recovery is also refused because successful sibling facts cannot yet be restored safely.

A process killed by a signal during a check, environment install, generate or setup command is retried once; if it dies again, the run reports it as an infrastructure failure naming the command and the signal and halts without spending an attempt. This includes shell-wrapped signals (exit 134/135/139 from `./install.sh` or similar). When the base comparison is unavailable (a crash that cannot be rebuilt), the base environment is discarded so the next check gets a fresh start, the run continues without excusals, and reports this in `check.txt` and the PR's Verification section. A convergence extra-attempt is recorded in the run and the PR; convergence is judged from the recorded check output, not the feedback text.

An emission that changes nothing is refused back to emit, including when materialise is replayed on an emit.txt the last attempt already holds. The refusal says the emission changed nothing, not that an edit's text is missing, and the branch holds one attempt per answer that changed something.

Inspect side effects and outputs when continuation is refused. Use explicit invalidation where a valid replay plan exists, or start a new run deliberately. Reopening the mode is not an implicit retry. No mechanism here can infer whether an unacknowledged external side effect happened before a crash.

### Supervision

When an owner process exits and the run's spine status is neither terminal nor deliberately paused, the supervisor automatically restarts the owner from the saved checkpoint. This eliminates the need for manual `graph-control resume` after a model outage, a crash, or a killed process.

**Terminal states** (no respawn): `done`, `cancelled`, `failed`. Note: `failed` now appears only for infrastructure errors or when a node failure is promoted after the owner exits without retrying (dm-95). Node command failures write `ready` with `last_failure` during execution.
**Deliberately paused states** (no respawn): `halted`, `paused`, `waiting_user`, `refused`.
**Supervisable states** (respawn if owner is dead): `ready` (without `last_failure`), `executing`.

| Aspect | Detail |
|--------|--------|
| Respawn limit | `REFACTORY_MAX_RESPAWNS` environment variable (default 3, 0 disables). |
| History | Each respawn is logged to `.refactory/supervision-history.json` and to `.refactory/events.jsonl`. |
| Detached process | The supervisor (`python -m refactory_dsh.supervisor`) outlives the CLI or GUI that launched it. One per run, held by `.refactory/supervisor.lock`. `console.json` keeps the supervisor's PID; the supervisor forwards SIGINT/SIGTERM to its child. |
| Transient vs deterministic | A `recovery.prepare()` refusal (e.g. "last node outcome is unknown or failed") is deterministic — recorded as a start failure and not retried. Only transient exits (crash, kill, model outage) are respawned. |
| Start-failure recording | An owner that exits before advancing the spine status writes `.refactory/start-failure.json` with the reason. `refactory status` reports `state: failed` with the reason, instead of the stale spine state. |
| Single-resume | A resume requested while the old owner is still exiting is filed as `.refactory/resume-pending.json` and honoured automatically when the old owner finishes exiting. No second command needed (dm-81). |
| Headless inline | A fresh headless run calls the unified `supervise()` function with `first_exit_code`, sharing the lock, signal forwarding, resume-pending consumption, and start-failure recording with the detached host. The inline host supervises only after the run's ownership is released (dm-94), so a consumed resume can start a new owner. |
| Terminal-status exit | A headless run's owner exits once its graph reaches a terminal status (done, failed, cancelled): the coordinator does not continue on stdin, whatever the profile. `graph-control stop` on a run whose graph is terminal and whose owner lingers stops the owner directly via SIGTERM. |
| Stop-requested guard | After each child exit, `needs_supervision` checks `.refactory/stop-requested.json`; a stop requested at or after the child's start time prevents respawn. An older stop does not block a resumed run. |
| Resume-pending validation | `resume-pending.json` is consumed only when its `checkpoint_sha256` matches the current `spine-checkpoint.json`; a mismatched pending is deleted and reported in `start-failure.json`. |
| Nesting prevention | The `supervisor.lock` prevents a supervised child from running its own supervision loop. A child's attempt to call `supervise()` fails the non-blocking flock and returns immediately. |
| Supervisor killed | If the supervisor itself is killed (`SIGKILL`), the owner child continues in its own session. The run is then unsupervised — exactly as today. A manual `refactory graph-control resume` is the fallback. |

### Correct a run's command

When a run's setup, check or generate command is wrong for the work, correct it
with `amend-command` and continue:

```bash
uv run refactory graph-control amend-command --run /absolute/run \
    --check 'pnpm test && python -m pytest -q tests' --note 'added pytest'
```

Exactly one of `--check`, `--setup`, `--generate` is required. The change is
recorded in `facts.json` as an explicit, timestamped entry in `command_history`,
with the old value, the new value and the person's note. It appears in
`refactory status` and in the pull request's Verification section. It is never
a silent overwrite of facts.json.

The handler finds which node to rerun from the graph's declarations — the
earliest due or completed node whose operation reads the amended fact — and
queues a rerun control with `source: 'amend-command'`. The spine consumes it
at its next boundary using `commandCorrection` semantics, which accept due
nodes that have not yet completed. Forward descendants are invalidated through
the existing redo machinery. The step that failed with the old command is
retried with the new one, without anyone editing the checkpoint.

Follow with `refactory graph-control resume --run /absolute/run` to start a new
owner. For a failed checkpoint, add `--acknowledge-failure` to the amend-command.

### Acknowledge an executing, failed, paused or stopped checkpoint

When the owner dies while nodes are running, the checkpoint stays `executing`.
When a run is stopped during an agent node, the checkpoint may be `failed` with
a pending dispatch under `.refactory/agent-delivery/pending/`. In either case,
recovery refuses the checkpoint. Acknowledge it as your statement that you
inspected the side effects:

```bash
uv run refactory graph-control acknowledge --run /absolute/run
```

This writes an acknowledgement bound to the checkpoint's SHA, moves pending
delivery markers and outputs whose fingerprints differ from the checkpoint's
recorded values under `.refactory/set-aside/<timestamp>/`, never deleting them,
and lists them in the command's output and in a `manifest.json` inside the
directory.

For a fork with some branches completed before the crash: completed branches'
outputs match the checkpoint's recorded fingerprints and stay in place.
Only in-flight branches' partial outputs differ and are set aside. After
resume, the spine re-dispatches only the incomplete branches from the cursor.
When all branches complete, the join reads outputs from both the preserved
completed branches and the re-dispatched ones.

For a `failed` checkpoint with pending dispatches (the stopped-run scenario),
acknowledge moves the pending dispatches to set-aside and writes
`acknowledge_failed: true` in `resume-request.json`. For an `executing`
checkpoint, it writes `acknowledge_executing: true` as before. Both are
SHA-bound. After acknowledge, `resume` continues the run without passing the
flag again.

A `paused` checkpoint whose owner has exited is acknowledged the same way, when
a pending dispatch makes recovery refuse it (dm-232): the markers are set aside
and the acknowledgement carries only the checkpoint's SHA, since a pause needs
no failure acknowledged. An optional fork branch that is recorded missing, a
verifier past its time limit say, settles its own marker on both runtimes, so
that case no longer needs it.

`graph-control stop` alone lets a dispatched agent's worker finish and applies at
the next boundary. When the worker dies while its owner lives on, though (a
`refactory stop` SIGINT that reaches the worker, a kill, a crash), the spine
records the failed node as retryable: the checkpoint is `ready` (or `halted`), the
worker's marker stays pending, and recovery refuses it (dm-275). Acknowledge takes
such a stopped checkpoint when a pending marker is there, the same way, with only
the checkpoint's SHA; a stopped checkpoint with nothing pending has nothing to
acknowledge and is refused. On a stopped checkpoint, a `stop` still queued for the
owner that exited is set aside with the markers and journalled as withdrawn: the
run already stopped, and the owner `resume` starts would otherwise apply it and
halt again. On an executing, failed or paused checkpoint a queued stop stays
queued for the next owner. So stop, `acknowledge`, `resume` continues the run with
no file moved by hand. The
walker has no graph controls: `graph-control` refuses a run without a spine
checkpoint.

Acknowledge refuses while the owner or the supervisor is alive, and an owner running
as another user (whose signal probe is refused) counts as alive. When the
supervisor stops on the dm-70 refusal, `start-failure.json` and
`refactory status` name this command as the next step.

### Withdraw a queued control

A queued control that was never applied can be removed:

```bash
uv run refactory graph-control withdraw --run /absolute/run --node <control_id>
```

This refuses if the control has already been applied (receipt `status: applied`).
Use `refactory status` to see which controls are queued.

### Replay a failed run

When a run's owner has exited after a failure, `--resume` refuses the
checkpoint unless `--acknowledge-failure` is passed. To replay a specific
node, preview and confirm the rerun with the failure acknowledged:

```bash
uv run refactory graph-control preview --run /absolute/run --node NODE
uv run refactory graph-control rerun --run /absolute/run \
  --node NODE --text "CORRECTION" --revision REVISION --acknowledge-failure
```

The `--acknowledge-failure` flag confirms that the person has inspected the
failed node's side effects. For a simpler retry of the failed due node
(without replaying a specific node), use:

```bash
uv run refactory graph-control resume --acknowledge-failure --run /absolute/run
```

Both commands start a new owner process. `resume --acknowledge-failure`
re-dispatches the failed due node from the checkpoint. `rerun` queues a
rerun control file, then starts a new owner that applies the queued rerun
at its first boundary. Both require the checkpoint's SHA-256 hash to match,
so neither can apply to a later failure.

### Redelivering

Bring a finished run's pull request up to date with main in one command:

    uv run refactory graph-control redeliver --run /absolute/run

This merges the current main, regenerates, re-checks and pushes, starting the
owner process itself. It is equivalent to `rerun --node deliver` plus `resume`,
without needing `--text`, `--revision` or a preview.

When the run failed at the deliver node (the usual case after a timeout),
`redeliver` passes the failure acknowledgement automatically. If the failure
is at another node, `redeliver` refuses with a message naming the due node.

When delivery is interrupted between its local merge and its push, the run
recognises its own merge commit on retry and resets to the reviewed commit
before re-merging. No manual `git reset` is needed. A commit that is truly
outside the run (not delivery's own merge) is still refused.

If the owner died during delivery, `graph-control resume` restarts it. If main
has moved further since the last delivery attempt, use `redeliver` to merge the
current main.

### Automatic conflict resolution

When delivery merges the target branch and an authored file conflicts, the run
no longer pushes the reviewed commit behind main. Instead:

1. Delivery records the conflict (which files, their base/ours/theirs content,
   and what main changed) and raises `NotDelivered` — nothing is pushed.
2. The `delivery_gate` evaluator reads the conflict facts and records them in
   `pending-merge.json` (main's tip, the conflicted paths). It does **not**
   start a merge — the worktree stays clean while the builder works. It
   returns `Verdict("reloop")`.
3. The run goes back to `emit`. The builder receives the conflict detail via
   `{{validation}}` (base, ours, theirs, and what main changed for each file).
4. Materialise reads `pending-merge.json`, starts `git merge --no-commit
   <target_tip>`, resolves generated-file conflicts to theirs (main's side),
   applies the emission, and either commits the merge or aborts it — all in
   one deterministic step. An emission that leaves conflict markers or omits a
   conflicted path is refused, and the merge is aborted.
5. The check, the base comparison and the reviews run on the merged result
   before delivery tries again.
6. The loop is bounded by `check_attempts` (default 3). A run that cannot
   resolve stops, naming the conflicting files. `graph-control redeliver`
   restarts it.

The merge exists only inside materialise (`_commit_emission`). A stop at any
point — a crash at emit, a refused emission, budget exhaustion, or a person's
stop — leaves the worktree clean with no MERGE_HEAD. After a crash between
the merge commit and the record write, `require_current` adopts the existing
merge commit rather than refusing.

After delivery pushes a merge commit, it advances the worktree record (head and
base) to the merge, so a later attempt builds on it and the next push
fast-forwards. Delivery only ever pushes a head that contains main.

`rerun` with `--text` on a node that takes no text (an op or a gate) is refused
with `"this node takes no text; drop --text"`.

### Incomplete cursor drain

A run whose cursor empties before reaching a graph sink — for example after
a redo that dropped the work due, or a skip that bypassed the remaining
path — halts with the names of the unreached sinks. The halt is visible in
`refactory status` (state `halted`, finished) and makes `refactory ceo`
exit non-zero. A sink is a node the scheduler has nowhere to go after: not a
node with an outgoing edge, not a fork branch or a director's child (they
continue at the join or the director's successor), and not a data node's
per-item exit (it returns to its data node).

### Check gates

A check that could not run appears as "did not run" in the PR Verification
section and in the run report, derived from `validation.status` in
`authoring-evidence.json`. The CEO may ask the person how to proceed (naming
what stopped the check and what would unblock it), but it never recommends
skipping a check gate. A person who wants to skip the check uses
`refactory graph-control skip --run <dir> --node <check_node>` from the CLI.
The GUI panel does not offer a skip button; use the CLI command. A person's
skip records `source: 'person'` provenance in the spine checkpoint and is
accepted by `require_completion`. A CEO's `wf_edit` skip records
`source: 'ceo'` and is refused.

### Forward-only invalidation

Replaying a node invalidates only its forward path. Nodes reached exclusively
through a gate's reloop back-edge are not invalidated or archived. Example:
replaying `overwatch` invalidates `{overwatch, gate_overwatch, report,
deliver}` but not `{emit, materialise, check, reviews, synth}`.

The forward path of a node inside a fork branch, or of a director's child,
continues where the scheduler would take it: the fork's join, or the
director's successor. Replaying `code_reviewer` invalidates
`{code_reviewer, join_qa, synthesize_qa, gate_qa, ...}` and leaves the other
reviewers' answers in place; once `code_reviewer` runs again, the join and the
gate after it run again too.

A verdict (proceed, reloop, or any forward outcome) inside a fork branch
replaces only the gate's own cursor position; sibling positions and their
container barriers remain intact. A reloop whose target lies outside its
gate's innermost container (fork or expanded director) is refused at graph
load time and at verdict time with a descriptive error naming the gate, its
container, and the external target.

### Scoped corrections

The `--text` on a rerun sets per-visit feedback, not a permanent prompt
amendment. The feedback is shown to the driver, cleared when the agent
completes, and superseded when a gate reloops with its own findings. A
one-shot correction ("emit these two files and change nothing else") applies
to that visit only and does not override a later review's repair.

A dispatched agent node's prompt carries that visit's feedback (rerun text
and a gate's reloop reason) once, for any graph and any mode: under a heading
"## Feedback for this visit" ahead of the prompt, unless the rendered prompt
already says it as whole lines (a template that renders the gate's reason, as
design mode's `emit` does; for a reason clipped to the runner's limit, its head
and its tail). A note found only inside a longer line is not said. In design mode
a send-back reloops to `pick_corrections` (dm-226), so the gate's reason is that
gate's visit feedback, cleared when it decides; emit gets the reason only through
the facts its prompt renders (`validation`, `qa_gate`, `overwatch_review`). `agent_delivery.with_feedback` decides, on both runtimes: the
walker hands it its gate's reason for the reloop target's next visit. The
feedback sidecar is written atomically by
the spine's `drain()` and read by the worker node before dispatch. It is
removed (or omitted from the manifest) when no feedback exists for the
visit, so a stale file is never read. Concurrent item passes of one node
may share the sidecar because they carry the same feedback text.

### Gate refusal reason persistence

When a gate refused work and sent it back (reloop), or halted at its
iteration limit, the gate's last refusal reason is preserved beside its
verdicts. On a rerun of the node the gate sent work back to, that reason
is shown as the node's feedback — the node sees what the gate said, even
when the person reruns without `--text`. When the person supplies `--text`,
the gate's reason appears first and the person's text follows it. A gate
that proceeded clears its stored reason, so a rerun after a passed gate
shows no stale refusal.

### Goal completion

Goal completion requires `done` status in spine-status.json. A
`tools/pre-execute` listener in the spine denies `update_goal complete` when
the recorded status is not `done` and no person-issued `completion_override`
matches the current status. The person can override via
`refactory graph-control confirm-completion`, which writes a
`completion_override` bound to the current status and invocation into
spine-status.json. The override is cleared when `recordStatus()` writes a
different status, so a resumed or new invocation does not inherit it.

### CEO-dispatched prompt integrity

The CEO's prompt recording listener attributes prompts to due nodes by the
delivery key: the node id from the execution contract embedded in prompt.txt,
keyed by run and node. A prompt naming no due node's key is recorded under
none and no records are cleared, so an unrelated `send_message` does not
refuse faithful reviewers. Only top-level CEO calls are recorded; a
reviewer subagent's own `send_message` is not recorded as a CEO prompt.

### CEO redo and attempt limits

When the CEO redoes a completed node through `wf_redo`, the redo preserves
the run's attempt counters and gate iteration counts. A gate that was on
iteration 2 of 3 stays there; the check gate's `attempts_used` is not reset.
The kind-specific counters `check_attempts_used`, `review_attempts_used`,
`check_repairs_used` and `review_repairs_used` are likewise preserved, because
`items.py` preserves every fact ending in `'attempts_used'` or `'repairs_used'`.
This prevents an infinite redo loop where the CEO redoes emit after the check
gate exhausts its attempts, resetting the counter each time.

Once a gate's iteration limit has halted the run, or a node has failed,
`wf_redo` is refused. The refusal names
`refactory graph-control rerun --acknowledge-failure` as the way a person
spends more attempts. Any `graph-control rerun` by a person resets all
counters — both gate iteration counts and attempt facts — as their decision
to spend more attempts.

### Deferred replay

When concurrent work (a fork's parallel branches) blocks a replay, preview
returns status `deferred` with the invalidation plan. Confirm the deferred
preview to queue the replay. It stays queued until concurrent work joins,
then applies if the invalidation plan still matches, or fails with "plan
changed" if it does not.

### Gate replay

Gate replay is not needed. Forward-only invalidation limits the blast radius:
replaying the upstream agent (e.g. `overwatch` for `gate_overwatch`) re-runs
only that agent, this gate and what follows, not the loop body. The refusal
message names this alternative.

### Writer and reader inventory

| File | Module | Function / Line | Reads / Writes | `last_failure` handling |
|------|--------|-----------------|----------------|------------------------|
| spine-status.json | workflow-spine/src/index.ts | `recordStatus` (767) | **Writer** | Writes `last_failure` when `failedNode` param is provided |
| spine-status.json | refactory_dsh/run_state.py | `spine_status` (48) | Reads `status` | Does not read `last_failure` |
| spine-status.json | refactory_dsh/run_state.py | `promote_unretried_failure` | Reads/writes `status`, `last_failure` | Promotes `ready`+`last_failure` to `failed` |
| spine-status.json | refactory_dsh/console.py | `_spine` (403) | Reads all fields | Returns full dict |
| spine-status.json | refactory_dsh/console.py | `status` (494) | Reads `status`, `last_failure` | Adds `last_failure` to activity, sets wait_reason |
| spine-status.json | refactory_dsh/dsh_runtime.py | `_spine_status` (267) | Reads `status`, `error` | Not affected; `ready`→"did not finish" is correct |
| spine-status.json | refactory_dsh/graph_control.py | `request` (52,84) | Reads `status` | Not affected |
| spine-status.json | refactory_dsh/interactive.py | `_terminal_stop` (348) | Reads `status` via `spine_status()` | `ready` is not terminal → no false stop |
| spine-status.json | refactory_dsh/interactive.py | line 393 | Reads `status`, `error` | Not affected; reads halted reason |
| spine-status.json | refactory_dsh/invocations.py | lines 84,89 | Checks existence only | Not affected |
| spine-status.json | refactory_dsh/live.py | `snapshot` (120) | Reads all fields | Includes `last_failure` in return, overrides `wait_reason` |
| spine-status.json | refactory_dsh/repo_change.py | line 62 | Reads `status` | `ready` not in FINISHED → run still active |
| spine-status.json | workflow-spine/src/index.ts | C1 `tools/pre-execute` listener | Reads `status`, `completion_override` | Denies `update_goal complete` when status ≠ done and no matching override |

Release checks exercised live CLI observation and native gate answers, pause/resume, reviewed replay and owned stop. Separate registered dead-owner recovery checks passed through public CLI resume and the native Resume control, preserving completed artifacts, saved configuration and accumulated resources while executing only the remaining node. An unavailable-provider guard prevented model execution. See the [validation record](release-validation-20260926.md).

## Target selection and uncommitted changes

The GUI lets a person pick a run's target repository or project from the registered projects list (the same registry as `refactory projects`). The `target` knob names the repository; an empty value starts a new project. A target `<select>` appears in the **Configure a run** section of `RefactoryBody` when registered projects exist and the chosen mode declares the `target` parameter (design and design_fast; not create or edit).

The `include_uncommitted` knob (off by default) opts a run into starting from the person's uncommitted changes. When set, modified tracked files, staged files, untracked unignored files and deleted tracked files are copied into the worktree and committed as the run's base. The run records each file with its kind and content hash. The PR's Verification says "The base included uncommitted changes from the working tree." The person's checkout is never changed.

## cwd escape detection

When a check's command changes its working directory outside the run's worktree, the check outcome includes `cwd_escaped` naming the directory. This is advisory: the check still passes or fails on its own exit code. Detection uses an EXIT trap that compares the shell's final `$PWD` with the worktree root (both realpath'd, prefix check), writing a sentinel to a file rather than stdout or stderr. A `cd` into a subdirectory of the worktree is not flagged. The trap fires even after `exit`, and the command's original exit status is preserved.

## Worktree and staging isolation

Design and design_fast modes use a git worktree of the target repository, on a branch `refactory/run-<id>`. Create, create_fast, edit and edit_fast modes stage their package in a per-run directory (`<run>/.factory-staging/.factory/packages/<name>/`), so two parallel runs never collide. At registration, the staged package is copied atomically (rename the old aside, rename the new in, then delete the old) to the shared `.factory/packages/<name>/`. If the shared package's state differs from what prepare recorded — including when it now exists but did not before — registration refuses, naming the other run (read from `.origin.json`). After a successful register the origin stamp in `.origin.json` identifies the registering run so a reloop does not refuse itself. Modes that only write into their run directory (loop, optimize) need no isolation.

[Profiles and candidate evidence](optimization-profiles.md) · [Fast modes and resource recovery](budgets-fast-modes.md) · [Worker capabilities](models.md#model-provider-versus-coding-agent)
