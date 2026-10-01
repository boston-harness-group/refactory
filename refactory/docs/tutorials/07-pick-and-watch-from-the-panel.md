# Pick and watch from the panel

See [Get started](../getting-started.md) for installation and [Terminal and GUI](../interfaces.md) for current interface coverage. The other tutorials use the shell. This one uses the browser, because the panel is
where a person chooses a run rather than composing the command for it: a mode, an
outer loop, an optimizer and an evaluator are all **described choices** read from the
same console the shell and the model use.

What you end with: a run chosen, started and watched from the right Sidebar, and the
same run readable from the shell by the id the panel showed you.

**The panel reads the console, so it is the CLI's surface in a different shape.** If a
choice is not in `uv run refactory modes`, it is not in the picker; if
the console refuses something, the panel shows the console's own reason.

---

## Start the GUI from this checkout

The repository's `refactory-web` profile composes the panel built at
`packages/client/ui-refactory/lib/client.js`:

```bash
uv run refactory ui --port 3103 --no-open
```

```
dsh web: http://127.0.0.1:3103/?token=<per-process token>
```

Open the printed URL, token and all. The token is per-process and the bare
`http://127.0.0.1:3103/` answers `401` without it.

The panel needs the same checkout the rest of these tutorials use: the bundle row
resolves `repoRoot` from the directory the server starts in, and the console it runs
is `.venv/bin/python -m refactory_dsh.console`.

## Open the panel

![Refactory’s blue Modes panel in a narrow dark layout](../assets/panel-20260927/after-configured-dark-narrow.png)

1. **Pick the workspace.** The app remembers one; a fresh profile starts with the
   workspace list empty and asks you to add a directory. Choose this repository.
2. **Select New Session.** Its panel entry is available while the conversation is
   still blank; opening the panel does not require a message or a model call.
3. **Press the expand control** at the top right of the conversation (its tooltip
   is `Open the sidebar`), then the **Modes** tab.

The panel is the right Sidebar's own tab, beside the files view, so it keeps its place
while you read the conversation.

The new-chat preset picker offers **CEO** for guided Refactory workflows and **Minimal** for a lightweight agent with a persistent shell. **Settings → Agent presets** offers the same two choices and lets you choose the default for new conversations. CEO coordinates agents, follows the workflow graph, and brings questions and results back to you.

Each CEO chat owns its own `wf_open` run. You can leave one chat paused at a
review gate and drive another mode in a different chat; state, edits and steering
stay with the calling session. A session must finish its current run before
opening another. Use distinct run directories across sessions (omitting `workdir`
derives one from the session); overlapping directories are refused before lowering,
including a completed run another session still has open for inspection.

## Find the controls

**Configure a run** has two views: **Run** for one execution and **Optimize** for searching variations of a scored mode. The views share **Package → Mode**, inputs and worker routing. Switching views never starts work, clears values or changes the watched run. The selected view also survives splitting the dock.

**Run** contains **Run once**. **Optimize** contains **Start optimization**, visible **Search limits**, and expandable **Search method · optional overrides** and **Saved profiles and advanced search**. Authoring modes explain why they cannot be optimized. Expand **Worker routing and shared resources** for backend, provider, model, effort, permission policy, optional budgets and role/node overrides. Hidden views and folded sections retain their controls and values. No new resource cap is introduced.

Each field has a **?** help button. Hover, focus with the keyboard, or click to see its explanation; Escape dismisses it. Mode input help comes from the mode's own declaration. The **brief** appears first and spans the form width. Clicking it opens a large multiline editor. **Apply brief** or **⌘/Ctrl+Enter** saves the draft; **Cancel** or **Escape** discards unapplied changes and returns focus to the brief. Applying text does not launch work.

![The expanded brief editor](../assets/panel-ux-20260927/brief-editor-dark.png)

**Activity & results** follows the explicitly selected run. Its workflow map reads actual nodes and connections from that run’s saved checkpoint, including forks, joins and retry paths. An active node pulses; waiting, paused, failed and completed nodes retain explicit labels. **Follow active** keeps the current node in view; turn it off to inspect another area. Select a node or focus it and press Enter for its full name/state. Reduced-motion settings stop the pulse. Older runs without topology retain the recorded-node list. Neither view invents a completion percentage or claims that execution completion means validation passed. Pending questions appear in a banner with a direct **Review questions** link, above configuration, and their answers are at the top of **Steering**. Detailed receipts, outputs and history expand separately. Graph correction still requires **Correct completed work → Preview affected results → Confirm correction and rerun**.

In **Steering**, click **View full ledger** to open the original brief and all past notes, corrections, constraints and gate records. Entries show their recorded timestamps and complete text, including blank lines and indentation. Search finds earlier text, kinds or timestamps; clearing the search restores the full history. The viewer follows incoming updates, retains the last read history during refresh, and returns focus to its trigger when closed with **Close ledger** or Escape.

Add multiline guidance under **What you want to say**, choose its kind, then press **Add a note**. The draft clears after the host confirms it was recorded; a failure keeps the draft for retry. Notes append to this run's ledger and reach workers at supported boundaries. They do not rewrite past entries or rerun completed work. Gate answers and graph correction remain their own explicit actions.

**Packages of modes** is an expandable section below the run controls. A wider panel places configuration beside activity; a narrow or split panel uses one column. Both themes retain visible keyboard focus, wrapping paths and the same controls.

## What each control does

| control | what it does | the console call behind it |
|---|---|---|
| **Package → Mode** | choose a package, then one of its described modes; dispatch still uses the full package/mode identifier | `console modes` |
| **Outer loop**, **Optimizer**, **Evaluator** | the three level-two registries; each entry carries the console's own description, and `Use the default` leaves the console's default in place | `console strategies` |
| **The mode's knobs** | one field per knob the chosen mode declares, shaped by the kind it declares: a knob with declared values is a choice, a `bool` is a choice over `true`/`false`, a number is bounded by its declared range, an open `str` is text. What the fields show is what the run sends, including a knob left at its declared default | `console start <mode> --knob NAME=VALUE` |
| **Search budget** | epochs, candidates per epoch and maximum candidate evaluations; invalid values disable Start. These are search settings, separate from mode knobs. Final holdout checks are additional; these are not token/time limits | `console start <mode> --epochs … --batch-size … --max-evaluations …` |
| **Start optimization** | starts the chosen mode in its own process and names the run it began | `console start <mode> --strategy … --optimizer … --evaluator … --knob …` |
| **Run once** | shows a unique run directory during launch, then reports the run's own status, due nodes and facts; **Watch and steer this run** attaches its live status and ledger | `console run <mode> --workdir <unique-directory> --knob NAME=VALUE` |
| **Activity & results** | streams the attached run’s durable activity, node/worker state, checks and available usage; **Refresh status** requests a snapshot | `console status --run <id>`, `console stop --run <id>` |
| **Existing run directory → Read this run** | attach the exact directory reported by CEO chat or the terminal; read its status and steer its ledger | `console status --run <directory>`, `console steer --run <directory> --gates` |
| **Packages of modes** | per package: **Enable**, **Disable**, **Remove**, and **Import a package** from a directory or git URL | `registry list`, `enable`, `disable`, `remove`, `import` |

Behaviours to know before you press something:

- **Controls appear as they become usable.** The section headings are always there,
  but **Start optimization** and **Run once** appear in their respective views once a mode is chosen, and **Refresh status** and **Stop the
  run** appear once a run is being read. A panel that looks short is waiting for a
  choice, not broken.

- **Disabled packages offer Enable.** Their run buttons are disabled, and enabling
  the package refreshes its modes. Authoring modes use **Run once** after their required knobs are supplied;
  **Start optimization** remains the optimization action for scored modes.
- **A user gate remains the user's decision.** Attach the run to see its pending
  question and permitted answers, then submit the verdict through the native
  controls or owning chat. The panel does not invent an answer. A live owner
  consumes it at its boundary; if the owner has stopped, use explicit **resume**
  only when the saved boundary passes recovery checks. Graph and optimizer
  controls retain their separate timing and ownership rules.
- **Task grades are separate from a mode score.** A completed run shows recorded
  per-item pass/fail counts when present. If no aggregate mode score was produced,
  the panel says so; it does not invent an average or keep promising a future score.
- **A one-shot has its own identity while it works.** Its path appears while preparing; **Watch and steer this run** becomes available once the fresh graph is durable. That button selects the run for the status and Steering sections. Launching it does not redirect a different run you deliberately attached. The same owned console call returns its final result or refusal; ending its stream requests cancellation through the runtime's subprocess owner. Preparation alone does not certify model authentication or success.
- **Attached activity updates live.** Persisted events update the selected run's
  nodes, workers, waits and outputs. Reconnect reads the same saved run; it does
  not relaunch it. No completion percentage is inferred from elapsed time.

## The run the panel started is an ordinary run

The frame names the run it is reading, in full. Copy that path and read it from the
shell — the same commands the other tutorials use:

```bash
uv run refactory status --run /tmp/tour2/run
```

```json
{
  "run_id": "run",
  "root": "/tmp/tour2/run",
  "mode": "counting_pack/fixed_step",
  "state": "finished",
  "events": [
    {"step": "init", "epoch": 0, "generation": 0, "evaluations": 0, "population": 4,
     "epochs": 1, "batch_size": 4}
  ]
}
```

```bash
uv run refactory leaderboard --run /tmp/tour2/run
```

```json
[
  {"id": "e76f85ded100", "parent": "30b45d283597", "score": 1.0,
   "edits": [{"op": "set_knob", "name": "target", "value": 6}],
   "structural": false, "exit_ok": true}
]
```

And the winner becomes a mode of its own, exactly as it would from the shell:

```bash
uv run refactory adopt --run /tmp/tour2/run --as my_tuned
```

```json
{
  "package": "my_tuned",
  "mode": "my_tuned/fixed_step",
  "adopted_from": "counting_pack/fixed_step",
  "candidate": "e76f85ded100",
  "score": 1.0,
  "package_root": "<checkout>/.factory/packages/my_tuned"
}
```

`registry smoke my_tuned/fixed_step` then reports `exit_ok true, score 1.0` — the run
the panel started left the same artifacts, and the same read-back works, because both
surfaces run one process boundary.

## Choosing from the shell first, when that is easier

The panel is a picker over JSON you can read yourself, which is the fastest way to find
out what is on offer before opening a browser:

```bash
uv run refactory modes | head -14
```

```json
[
  {
    "id": "adopted_exact/exact",
    "package": "adopted_exact",
    "description": "establish closed forms of sums with sympy and score what checked out",
    "startable": true,
    "action": "start",
    ...
  }
]
```

`startable` and `action` are what the panel reads to decide whether a mode is a run
(`start`) or an authoring verb (`create`, `edit`, `design`).

---

## Where the panel fits

- **The shell** — [Optimize it](02-optimize-it.md) — the loop, its report, and a
  candidate's diff.
- **The conversation** — [Drive it from the GUI](05-drive-it-from-the-gui.md) — the
  same acts asked for in words, which the CEO performs with the `refactory` tool.
- **Authoring** — [Author with a model](03-author-with-a-model.md) — `create`, `edit`
  and `design`, available through **Run once** or stepwise chat. Fast variants
  have [completed bounded CLI/native GUI acceptance](../budgets-fast-modes.md#acceptance-status).
- **A person at a gate** — [Run it with a person](06-run-it-with-a-person.md).
- **Datasets** — [Datasets and Tasks](../datasets.md) — a mode that iterates items,
  and the Task that supplies and judges them.

## When the panel is not there

- **Nothing appears in the Sidebar.** The profile in force does not compose
  `@deepseek-ai/dsh-client-ui-refactory` — for example a plain `dsh --profile web`
  session, which the harness owns and re:factory never writes to. Start the GUI with
  `uv run refactory ui` (or `./refactory-dsh`) to boot the re:factory-owned
  `refactory-web` profile; `refactory doctor` names the layer to add if it is short.
  Rebuilding with `pnpm run build` refreshes the artifact but does not change the
  composition.
- **The panel loads but every read fails.** The console could not be reached: the
  message the Host reports carries the console's own words and the argv it ran, so
  `uv run refactory modes` by hand from the same directory reproduces it.
- **A package you just imported is missing.** Press the reload control at the top of
  the panel; the catalogue is read once per open.
