# Hardening: what the audit found, and what holds it now

## Failure digest, feedback re-fill and builder probe (dm-120, dm-121)

### The failure digest

Before this change, the builder received raw check output — up to 500 lines of
stack traces, passing-test lines and source-map noise — or, after a redo, the
literal `{{validation}}` placeholder unfilled.

The **failure digest** (`check_digest.py`) parses check output per runner (pytest,
vitest and node's test runner; unknown runners fall back to the raw tail).  Each
failing test's id, assertion or error message and a short traceback are extracted,
with ANSI codes, passing-test lines and source-map warnings stripped.  The digest
is stored in `check-record.json`, which is a file (not a fact) and therefore
survives a redo or rerun.

### Feedback re-fill

Each builder's emit node declares its feedback keys as a `Node.feedback_keys`
field.  Before rendering, `agent_delivery.fill_feedback` — called at the three
render sites (worker_node, nn/executor, agent_delivery.main) — checks whether any
declared key is absent from the facts.  When one is:

1. Re-derive from the producing gate's persisted file (`check-record.json`,
   `validation.txt`, `reviews/qa-gate.txt`, `reviews/overwatch-gate.txt`,
   `repair.txt`).
2. Fall back to `feedback-initial.json` (prepare-time recorded values).
3. Refuse only when both are missing.

The re-fill is resolved through the `refactory.feedback_refill` entry-point group,
following the same pattern as `refactory.environment_overlay`.  Core modules
(`agent_delivery`, `worker_node`, `nn.executor`) import nothing from the product;
they reach the refill logic only through the entry point.

### Agent workspaces

An agent node can declare where it works: `workspace` on the node (`nn.graph`,
saved only when it is not the default, `none`), carried by the IR, the run's
declarations and the worker spec. `scratch` is a fresh empty folder per dispatch;
`clone` (`of: attempt` or `base`) is a prepared clone of the run's worktree.
`collect: files` copies the node's declared writes out of the workspace;
`collect: diff` writes the clone's diff into the node's first write. Both runtimes
run it through one module, `refactory_dsh/workspace.py`: the spine's worker
(`worker_node.execute`) and the walker (`Executor._visit_agent`) prepare the
workspace before the delivery bracket opens and collect from it before it closes, so
`agent_delivery.finish` checks what the workspace produced. A workspace is refused on
the CEO-subagent route (`workers.lower_workflow` with no worker model context): nothing
there could give the node a workspace or collect from it. The spine only passes the
declaration through.

Design mode's builder (`emit`, in design and design_fast) works in a clone of the
last attempt and collects its diff: it edits the repository there, runs the tests its
change touches (the test files it changed or added, and the tests importing the modules
it changed) and the brief's own example or CLI path as often as it needs, and stops. Its
prompt tells it to run `refactory-targeted-tests` before it stops and fix what fails, and
says the full check runs after it answers, so it does not run the whole suite itself: in the
live trial (dm-149) its full-suite runs took 11.6 of its 27 minutes, and in design runs A1 and
A2 its whole-suite runs under its own `timeout` were killed with no output, up to nine times
a session. That command (`targeted_tests.py`, put on the clone's PATH by `clones.prepare`, so
both runtimes have it) reads the workspace's current diff as materialise will commit it,
selects with the self-check's own function (`design_ops.self_check_selection`), runs that
selection with the check's options kept, and prints what it selected and the result; so the
failures the self-check would send the builder back for, to a new cold session, are found in
the session it is in. It sets no limit of its own. Materialise,
the check, the reviews and the target guard read the derived `emit.txt` as they read
an emitted one. create, edit and fast-create keep the emission-only builder.

- **Prepared once per run.** The clone (`clones.prepare`, the product's half,
  reached through the `refactory.workspace_clone` entry point) shares the target's
  objects (`git clone --shared`), keeps its own refs and index, and has no remote and
  no hooks. It persists across commands, fix-ups and attempts. When materialise has
  committed what it derived, the clone already holds the new head: the next dispatch
  verifies that (the clone's change from the new head, less what its preparation
  made, is empty) and only moves its base (`verified`). A head that did not move keeps
  the clone as the agent left it (`kept`), so a refused emission's work is still there
  to fix; any other head resets the clone to it (`reset`). A file the agent added counts
  as held when its content and mode are the commit's. Installs run again only when a
  dependency manifest or lockfile anywhere in the clone changes
  (`clones.dependency_fingerprint`), setup with them or after a reset, and generate on
  every dispatch. A step that fails does not stop the node: the agent is told, and the
  check runs the same steps. A pending merge from delivery is started in the clone, its
  generated conflicts resolved to main's side as materialise resolves them; a merge the
  agent finished is kept across a refused round, and one materialise gave up is
  dropped, so main's changes never come back as the agent's own.
- **Run.** The model's working directory is the workspace; its receipt and patch stay
  in the run directory (`DshModelClient.complete(cwd=...)`). Its environment is the
  clone's (`worktree.check_environment`): the clone's own `.venv` and
  `node_modules/.bin` first, the run worktree's taken off PATH, and HOME at the
  clone's `.refactory/home`, with pnpm's store, uv's cache and the XDG cache inside it
  (the person's are read-only to it). Its shell is told a DSH home of its own inside the
  workspace (`workers.write_patch`'s `shell-env` row), and its `BASH_ENV` file unsets the
  run's variables (`REFACTORY_*`, `SRF_*`, `VENV_BIN`, `PYTHONPATH`) before each command,
  as the check never has them; the DSH process keeps them for its observer. Its DSH
  sandbox policy is `workspace-write` rooted at the workspace, with approval `never`
  (`workers.permission_patch`), where a DSH runner works here (bubblewrap or Landlock on
  Linux, Seatbelt on macOS; probed once). Where none does, a workspace node on the dsh
  backend is refused, since its shell would have full access while it is asked to install
  and run things, unless `REFACTORY_WORKSPACE_SANDBOX=unconfined` chooses that; the
  dispatch record and a warning then say so. A native backend (codex, Claude Code) runs
  under its own sandbox and keeps the person's HOME, where its CLI finds its login.
- **Collect.** `derive_emission` writes whole-file `===FILE:` and `===DELETE:` blocks
  from `git diff --raw --no-renames <base>` and the untracked files, leaving out
  `.refactory/`, the environments and tool caches, and what the preparation made while
  the agent left it as made. A binary file, a symlink, a mode change, a new executable,
  a submodule, a line the emission would read as a block header, or (without a
  worktree) a deletion is refused, as is a path an emission header cannot carry (with
  whitespace, `=`, a quote, a backtick or a backslash) or one holding a control or hidden
  format character (the same `unprintable_refusal` an emitted block gets, dm-278), a
  file over 50 MB, a change over 200 MB in all, and a workspace with no change: the
  refusal is left beside the emission
  (`.refactory/workspace-refusal.json`, keyed by its digest) and materialise sends it
  back to the builder like any refused emission. Line endings and end-of-file newlines
  are normalised by the blocks, with a warning, and the next dispatch tells the agent
  which files. Only the repository's own ignore rules apply: the clone ignores a
  person's global ignore file.
- **Clean-up.** At the run's end, however it ended: the walker and the DSH evaluator
  (the `refactory.run_end` entry point) and a spine run's owner. A workspace a
  dispatch holds is left.
- **OpenShell (experimental).** On the `openshell` worker backend the same workspace is
  packed (`openshell_backend.pack_workspace`, after the clone takes its own copy of the
  objects it borrowed), the agent (Claude Code, the binary the gateway's provider admits)
  runs inside the sandbox, and the folder comes back (`download_artifacts`) to the same
  collect step (`workspace.sync_back`, which refuses a folder that is not the workspace's
  copy). It has had no live run, so it is refused unless
  `REFACTORY_OPENSHELL_WORKSPACE=1` opts in. Not yet: packing `.venv` and
  `node_modules`, the node's model and usage for `claude -p`, and the remote stop's PID
  path.

### The builder probe

The emit prompt's "do not apply or test your change anywhere", which create, edit
and fast-create still carry, exists because direct edits to the worktree bypass
materialise and the check. The builder probe gave design's builder a safe way to test
before its workspace did; its prompt no longer names it. Since dm-133, the probe is
also available to `design_doc` via `--command` (no `--emission`); the clone
runs at the run's base with `emission=None`, and the same isolation guarantees
apply:

- **Standalone clone** (`clones.clone_shared`, `git clone --shared`): not a linked
  worktree, so no `.git` registration; its own refs and index, the target's objects
  shared.  `origin` removed after clone.
- **Hooks off**: `core.hooksPath=/dev/null`, `LEFTHOOK=0`, `HUSKY=0`.
- **Isolated HOME and DSH_HOME**: per-probe directories under `probe-<id>/`.
- **PATH stripped**: the run worktree's `.venv/bin` and `node_modules/.bin` are
  removed so tools resolve from the clone, not from the run's worktree.
- **Lifecycle**: cold probes are deleted in a `finally` block. Warm probes
  (`--test` via the CLI) are kept on disk between probes for reuse — the clone
  directory survives with `.venv` and `node_modules` intact — and are removed
  by `sweep_warm_clones`: at `materialise`, and at the run's end by the walker
  (`nn.executor.execute`, through the `refactory.run_end` entry point), by the
  owner of a spine run started from the CLI (`interactive.run`) and by the DSH
  evaluator (`dsh_runtime.evaluate_on_dsh`). A run the GUI's DSH server drives
  has no Python owner, so nothing removes a clone it leaves before materialise.
  A clone another probe is using is skipped; one that cannot be removed is
  named in a warning. Leaked cold probes are swept by `sweep_dead_probes`
  (owner PID check).
- **Not an attempt**: no `commit_attempt`, no attempt counter increment.
- **target_guard preserved**: the existing test at `test_design_worktree.py:667`
  continues to pass.  A cold probe is removed before `finish()`; a warm clone
  stays under the run directory's `.refactory/` until materialise or the
  run's end.

#### Why each isolation guarantee exists

| Guarantee | Reason |
|---|---|
| Standalone clone | dm-96: a PR's checks registered 59 worktrees in the target's .git |
| Origin removed | A push from the probe could reach the shared .git |
| Hooks off | pnpm and lefthook rewrite shared hooks (dm-96) |
| Isolated HOME | Tool caches stay out of the worktree's diff |
| PATH stripped | A missing tool from the clone must not resolve into the run's worktree |
| Lifecycle | Cold probes deleted in finally; warm probes reused within the attempt, cleaned by materialise and at run's end; a SIGKILL leaves a warm clone that the next materialise or end-of-run cleanup removes |

- **Vitest mixed-ID routing**: when a probe command includes both bare spec IDs
  (whole files) and structured `file > suite > test` IDs, the named files run
  whole — more tests than strictly asked, never fewer. Splitting into two
  `&&`-joined invocations was rejected because a failing first invocation would
  silently skip the second (the failure class #502 already fixed for the
  self-check's chunks), and `failfast.targeted_prepass` passes bare spec IDs
  as `test_ids`, so the split would reach the self-check.

#### What is not isolated

Installs use the person's real HOME (`worktree.py:1229-1230`), as the check's
installs do.  `~/.npmrc`, `~/.netrc`, uv configuration and keyrings are how most
private indexes sign in.


Every issue closed before this pass was re-read against the code and probed for its
own acceptance criteria — by running them, not by grepping for identifiers. Five
were not finished, four of those while being marked done. This records each one, the
evidence, the fix, and the test that keeps it fixed, so the same claim cannot be
made twice.

Two probes also came back **clean**, and they are recorded because an audit that
only finds fault is not an audit.

---

## The method

For each issue: read its acceptance criteria as written, then write a probe that
would fail if the criterion were unmet, and run it. Grepping for a name proves a
name exists. Running the behaviour is what found these.

```bash
cd refactory && ../.venv/bin/python -m pytest tests/ -q
```

---

## 1. `check_package` executed generated code in the driver (#23)

**What the issue said.** create-mode writes Python and then something decides
whether it is usable, "which means loading it, which means executing model-written
code". The table named `create_ops.check_package: imports the generated package to
validate it — none — in-process, full access`.

**What the code did.** The docstring said *"A generated package is **never
imported here**"*. The body's tail was:

```python
package = load_fresh(name)              # importlib.import_module
for ops_module in package.ops:
    importlib.import_module(ops_module)  # model code, again
for mode_class in package.modes:
    problems.extend(_check_mode(mode_class))  # constructs the mode
```

Only the *structural* half ran in a subprocess. The semantic half — is the name
right, do the ops import, is the mode constructible, does it satisfy every
validator rule — ran in the driver with full access.

**The fix.**

- `sandbox.verify_sandboxed` now performs the whole check in its subprocess,
  including the package name, each declared ops module, and each mode's
  construction, score, knobs and validator rules.
- `register` and `smoke` needed the manifest only to *name* what was registered and
  which mode to smoke, and were loading the package to get it. They use the new
  `sandbox.describe_package`, which reads it out of process.
- `load_fresh` and `_check_mode` are deleted. No function in the driver loads a
  generated package any more.

**Held by.** `tests/test_create.py`: a test patches `importlib.import_module` to
record any import of the package's module prefix, runs `check_package` to a clean
result, and asserts the recording is empty. A second does the same for
`describe_package`.

**Known limit, stated rather than implied.** `registry.catalogue` and `registry.load`
still import packages to describe them, so *listing* a generated package still runs
it. That is a different boundary from create-mode's validation, and it is not fixed
here.

---

## 2. Run retention was declared and never applied (#24)

**What the issue said.** `memory(namespace, kind, retention)` where `retention` is
`run`, `project` or `global` — "the difference between a memory that dies with the
run and one that accumulates".

**What the code did.** `MemoryStore.forget_run_scoped` existed, with a docstring
reading *"Called when the run ends"*, and **nothing called it**. So a run-scoped
namespace accumulated exactly like a project one.

The cost is sharper than untidy files. `_evaluate` re-evaluates candidates when a
run resumes, and a run-scoped memory left on disk is read the second time — so a
resumed run would not be the run it resumed.

**A second gap, found while fixing it.** Memory did not cross to DSH **at all**:
`lowering` dropped it and `run.py` built its `RunContext` without a store. A mode
declaring a cross-run prior therefore worked in the walker and silently kept nothing
on the runtime that ships. Nothing reported the difference, because the running
example declares no memory.

**The fix.**

- `Loop._evaluate` forgets in a `finally`, so a crash or a stop applies the policy
  too, not only a clean finish.
- `lowering.write_declarations` carries the mode's memory declarations beside the
  node facts; `lowering.read_memory_declarations` reads them back; and `run.py`
  builds the store from them, with the same roots the walker uses.

**Held by.** `tests/test_declarations.py`: a run-scoped namespace is gone after
`forget_run_scoped`; project and global namespaces survive it; the loop's
`_evaluate` contains the call **and** a `finally`; the declarations survive the
lowering; and the node runner builds a store.

---

## 3. The strategies recorded their decision instead of applying it (#20)

**What the issue said.** "The deliverable includes all the SRF candidates
implemented on the new API. The abstraction is proven by the ports, not by the tests
for the abstraction."

**What the code did.** Five strategies took their distinguishing branch, counted it,
logged it, and called the ordinary path:

```python
def epoch_step(self, run=None, *, branch="mutate"):
    if branch == "merge": self.merges += 1
    else:                 self.mutations += 1
    run.log(f"branch={branch}")
    return super().epoch_step(run)     # the merge is not a merge
```

`reflective_sampler` stored `last_reflection` and read it nowhere.
`adaptive_meta` computed `meta` and logged it. `research_then_optimize` stored
`research_note` — `grep` found only the two lines that set and logged it.
`staged_curriculum` logged the stage and ran the ordinary step.

Measured, not argued: running all fourteen on identical input, **six produced
byte-identical leaderboards to `default`**.

The tests passed because they asserted the *counter* —
`test_reflective_takes_a_merge_branch_and_a_mutate_branch` checks
`loop.merges >= 1` — and a branch that changes nothing still increments a counter.

**The fix.** `ProposalContext.strategy` is a new channel carrying what the outer
loop decided for this iteration, so a decision reaches the proposer instead of only
the log. Each strategy now acts:

| strategy | what it now does |
|---|---|
| `adaptive_meta` | widens the batch by 2 on a stall — a real budget decision |
| `reflective` | builds a genuine merge: one parent's knobs taken from another, and it falls back to a mutation when the parents differ structurally rather than in their settings, because a no-op merge is refused as `E-noop` and loses the slot |
| `reflective_sampler` | drops a proposal that repeats an edit the run already refused |
| `research_then_optimize` | publishes its conclusion into the context |
| `staged_curriculum` | publishes which stage is running |

**Held by.** Six tests that assert the **effect**: the batch is wider, the merged
proposal takes the other parent's setting, a refused edit is not re-proposed, and
the decision is present in the context a proposer receives.

**Honest limit.** With `optimizer=random` the model-facing decisions (a research
note, a curriculum stage) reach a proposer that cannot read them, so four strategies
still match `default` on the tiny example. The channel is real; showing the
difference needs a model proposer.

---

## 4. A revision recorded a name, not a state (#13)

**What the issue said.** Provenance of *"what it was edited from, **at which
state**"*.

**What the code did.** `_revision_source` correctly handed the model the source
files. The record then said `revision_of: "circle_packing/adaptive"` and nothing
else. A mode is not a fixed thing — the outer loop edits it — so that names a class
which may have changed since, and the revision cannot be reproduced or diffed
against what it actually started from.

**The fix.** `edit()` resolves the source mode once, records its whole
`revision_of_state` plus a `revision_of_digest` (the same `candidate_id` the loop
uses, so a revision and a candidate that are the same state agree), and returns the
digest.

**Held by.** A test that runs a revision, reads the origin file, asserts the
recorded state is a full state dict, and that the digest matches both the result and
`candidate_id(state)`.

---

## 5. The lowering dropped `callable_name` (#21)

**What the issue said.** *"`FnNode.callable_name` already exists; do not invent a
command string."*

**What the code did.** Emitted `command: "python -m refactory_dsh.run <mod>:<fn>"`
and left `callable_name` unset. `callable_name` and `post_checks` appeared zero
times in `lowering.py`.

**Where the issue was right, and where it was wrong.** `callable_name` *is* a real
`FnNode` field — `factory/workflow/primitives.py`, round-tripped by
`factory/workflow/parity.py` — so the lowering was dropping something the language
has. But **DSH's spine does not model it**: `WorkflowNode` has `command`,
`evaluator_type` and `evaluator_command`, and `callable_name` appears nowhere in the
spine's source, which executes the command. A graph carrying only `callable_name`
would be a graph whose nodes never run.

**The fix.** The lowering emits **both**: `callable_name`, so the IR stays faithful
to the language and refactory's own tooling can read it, and `command`, because that
is what the runtime we ship executes.

**Held by.** Two tests: every op node carries both fields, and every declared
`callable_name` resolves.

**Not adopted, stated rather than skipped.** `AgentNode.post_checks` and refactory's
`ForkNode`/`JoinNode` are still unused. `post_checks` would duplicate a check the
executor already performs; adopting fork/join is a real design change, not a
fidelity fix, and is not done. Design's `emit` still needs none: an `emit.txt`
`materialise` cannot apply is sent back to `emit` by the check (#361).

---

## 6. The documented commands, run as written (#31)

Ship readiness is not a feeling about the code; it is the claim that a person who has
never seen this repository can install it, follow the docs, and get the promised
result. That claim is testable, so it is a test: `tests/test_ship_ready.py` reads the
shipped documents, extracts every command line from their `bash` blocks, and runs it.

The gate checks four things, and each one found something:

| check | what it would catch |
|---|---|
| every `python -m refactory_dsh.X` a doc names imports | a tutorial pointing at a deleted module |
| every module's `--help` accepts the flags the doc shows | a flag renamed in code but not in the prose |
| the README links the architecture and the tutorials | a stranger with nowhere to go next |
| the docs the README promises exist on disk | a link to a file that was moved |

**The bug it found.** `python -m refactory_dsh.sandbox --help` printed its usage line
to **stderr** and exited **2**. Two things were wrong with that, and the second is the
worse one: a reader who asks for help is told they made an error, and a program that
captures stdout — which is how the doc gate itself reads a help page — saw nothing at
all. So the flag table was empty and the gate could not check the sandbox CLI, which
is where an unknown flag had to be caught by hand.

**The fix.** `sandbox.main` gained `usage()`, which is the one string both paths
print, and an explicit branch: `-h`/`--help` prints it to **stdout** and returns
**0**. The two cases that are genuinely misuse keep their old behaviour, because they
are a different decision and were already deliberate:

| invocation | stream | status | why |
|---|---|---|---|
| `-h`, `--help` | stdout | 0 | a question, answered |
| no arguments | stderr | 2 | nothing was asked for; `test_the_sandbox_cli_needs_an_instruction` pins this |
| an unknown flag | stderr | 2 | not a command |

**Held by.** A test that runs `--help` against `sandbox`, `loop` and `console` and
requires both a zero status and non-empty stdout. It is not vacuous: the pre-fix
behaviour (status 2, empty stdout) fails exactly that assertion, which was run both
ways when the fix was made.

**The second finding was in the README, not the code.** It described the layout but
never linked `docs/architecture.md` or `docs/tutorials/`, so the entry point a
stranger actually opens did not lead anywhere. The README now carries a Documentation
table — architecture, tutorials, API, tool packing, hardening, contributing.

**Also regenerated.** `docs/api/reference.md` is generated from the code and had
drifted from it; the new `usage()` export made that visible. `test_docs_api.py` fails
when the two disagree, so the reference is checked rather than trusted.

**And the tracker's own commit field was not checked.** The gate asked a done item for
a non-empty `commit`, which `"(pending)"` satisfies, so six closed items carried one
and the field proved nothing. A recorded commit exists so a reader can go and look at
the change; `test_a_recorded_commit_is_one_git_actually_knows` now runs `git cat-file
-e` on every one, and all 23 done items resolve. Verified both ways: planting a
made-up hash fails the gate, and removing it passes.

**The install itself was run, not inferred.** `./install.sh` is the one command a
stranger executes, and everything above only proves the commands *after* it work. It
was run against this checkout: `pnpm install --frozen-lockfile`, `pnpm run build`,
the venv, the editable installs, and the registry listing all
completed with status 0. The build was then checked for the damage a build can do —
`git status` is clean, the Python suite is unchanged, the DSH-side specs pass, and the
doc gates pass — because "it exited 0" and "it left the tree working" are different
claims and only the second one matters.

**What this still does not prove.** It is not a clean clone: the venv already existed,
so the script's fresh-clone branches were not exercised. A machine with neither was not available here.

---

## 7. The sandbox made an installed toolchain unreachable (#30)

Found by installing the tool and running the path, which is the only way it could be
found. Every Lean test in the repository **skipped** on this machine, with a reason
naming the missing binary — an honest skip, and one that hid a bug, because the
skipped tests were the only thing that would have executed the checker.

**What happened.** Lean 4.33.1 was installed. `registry smoke lean_pack/prover`
returned **score 1.0**: a real proof, accepted by the real checker, in one attempt.
`sandbox.smoke lean_pack/prover` returned **score 0.0** with `exit_code 1` and:

```text
error: no default toolchain configured. run `elan default stable` to install
```

Same mode, same machine, same theorem, two answers.

**The cause.** `run_sandboxed` sets `HOME` to the run's working directory. That is
correct containment — code we did not write should not read a user's dotfiles — but
elan finds its toolchains through `HOME`, so the sandbox pointed the checker at
`<workdir>/.elan`, which is empty. Proven by isolating the variable: `lake env lean`
on a trivial file succeeds with the real `HOME` and fails with `HOME=<workdir>`, and
in the failing case it *creates* `<workdir>/.elan/settings.toml` as a side effect.

This is not a Lean bug. rustup, pyenv, nvm, gvm and sdkman all locate themselves
through `HOME`, so containment alone breaks every per-user toolchain manager, and the
failure is a wrong statement about the machine: "no default toolchain configured" on a
machine where the toolchain is installed and working.

**The fix, in the declaration.** The manifest already existed to answer "can this
package run here"; it now also carries what the tool needs *at run time*. `Tool`
gains `env`, a tuple of `(name, value)` pairs, and `toolchain.tool_env(package)`
collects them from the package's **present** tools. `sandbox.smoke` forwards them into
the run. `lean_pack` declares `ELAN_HOME`.

Three properties make it hold rather than paper over:

| property | why it is the right shape |
|---|---|
| the value is resolved where the declaration is read | package code cannot compute it — by the time an op runs, `HOME` is the working directory |
| absent tools contribute nothing | forwarding for a tool that is not installed adds environment a run cannot use |
| applied *after* containment, and still secret-filtered | `HOME` stays the working directory, and a declaration cannot smuggle a credential in |

**Held by.** Four tests. One drives the mechanism on a synthetic package, so it holds
on a machine without Lean. One asserts the same probe sees **no** `ELAN_HOME` without
forwarding — the half that makes the first mean something. One asserts the secret
filter still applies to a declaration. One runs `sandbox.smoke` on the real mode with
the real checker. Every one of them was checked against the unfixed code: disabling
forwarding reproduces `score 0.0` and the elan message, so none passes vacuously.

**A test that had to change, and why that is the finding rather than a nuisance.**
`test_a_declared_tool_that_is_missing_names_the_fix_when_required` asserted that
`preflight("lean_pack")` raises `ToolMissing`. It passed on a bare machine and failed
the moment the declared tool was actually installed — it had quietly asserted that
Lean is *not* here. It now asserts the conditional truth, and the mechanism is tested
against a synthetic absent tool. `test_the_lean_declaration_names_the_toolchain_home`
is machine-dependent by design and carries `@needs_lean`.

**Verified, finally.** The Lean path had never run. It has now: `verified: true`,
`attempts: 1`, strategy `omega`, `exit_code 0`, and the axiom audit reports
`[propext, Quot.sound]` — no `sorryAx`, so the `#print axioms` check that separates a
proof from a `sorry` is exercising real output rather than a stub's. The suite runs
**562 passed, 7 skipped with Lean** and **558 passed, 11 skipped without it**; both
were run, because CI has no Lean and this checkout now does.

**Still unverified.** mathlib is not installed, so the mathlib problem and its
`[[require]] path` remain untested. The core-library problem is the one that ran.

---

## 8. Two defects mathlib found, and the optimizer SRF ships that we did not run

Closing #30 left two things unverified and one thing unported. All three are the same
kind of gap: a claim that was true of the tests and not yet true of the code.

### `mathlib` was installed, and the problem it exists for did not run

The package declared mathlib, the tool probe reported it, and the mathlib problem had
**never been executed** — the tests skipped with a reason naming the missing path. So
mathlib was installed: a `mathlib4` checkout, `lake exe cache get`, 7.4 GB. The
problem then scored **0.0**, and the reason had nothing to do with mathlib.

**Defect one: the statement did not parse.**

```lean
theorem sum_odd_eq_sq (n : Nat) : ∑ i in Finset.range n, (2 * i + 1) = n ^ 2 := by
```

```text
error: unexpected token 'in'; expected ','
```

`∑ i in s, f i` is the form most introductions show. This Mathlib does not accept it:
Lean reads `∑ i` as an ordinary binder and then wants a comma. The form that compiles
is `∑ i ∈ s, f i`, alongside the explicit `s.sum (fun i => ...)`, which was checked
side by side in one file to be certain which of them was wrong.

Worth being exact about why nothing caught it. Every test of this mode used a **stub
checker**, and a stub accepts whatever it is handed. A syntax error in the statement
is invisible to a test that never parses the statement.

**Defect two: the timeout abandoned the checker, and a timeout reads like a refutation.**

`timeout_s` defaulted to 60s. The first invocation against mathlib pays for importing
the library, which took longer than 60s cold; the same proof then took 22s warm. So
the run reported `verified: false` for a theorem that is provable, on a machine where
it was about to be proved. The default is now 180s, with the reason recorded at the
knob: a fact that costs a run its score must clear the cost of starting.

**A third thing, found while reading the record it produced.** The run's
`lean_version` fact came from probing `lean` on the PATH, and the project's checker is
not that compiler: a project requiring mathlib takes a `lean-toolchain` from the
dependency, so `lake env lean` ran **4.34.0-rc2** while the PATH had **4.33.1**. The
record named a compiler that decided nothing. `scaffold_project` now asks the checker
in the project directory once, right after the project exists, and `result.json`
carries `checker_version` — the compiler that actually decided.

**Held by.** Three tests: one pins the `∈` form and the `BigOperators` preamble with
neither Lean nor mathlib installed, so the syntax cannot drift back; one pins the
timeout default against the cost of a large import; one runs the real problem and
requires `statement_ok == 1`, `verified == true`, no `sorryAx`, and a
`checker_version` in the durable record. The first two are ungated — they are the
regression guards. The third is gated on `MATHLIB_PATH` and **ran**, not skipped.

### SRF's own optimizer was vendored and never used

`srf/lab/map_elites.py` is not one of SRF's thirteen modes; it is an optimizer SRF
keeps beside them — a multi-emitter MAP-Elites over mode and knob space, whose
distinguishing property is that it keeps **one champion per cell of a feature grid**
rather than one best candidate overall. Nothing in this repository referenced it. The
thirteen modes were ported and this was not, so the count of "SRF ports" was complete
against a list that omitted it.

Ported on the loop's own vocabulary: features are graph size, whether the candidate
was reached by moving structure or only settings, and the step budget it spends; the
archive refuses a weaker candidate in an occupied niche and accepts one in an empty
niche; selection draws parents from across occupied niches.

**It was put in the wrong bucket, and the count papered over it.** It was added as a
fifteenth `OuterLoop`, and `SRF_LAB_PORT_COUNT = 1` existed to make the totals add up
around that placement. MAP-Elites is a search policy, not a control flow.

### The two buckets were not what the code said

`outerloop.py` says *"most research ideas live here [in the proposer]"*. The registry
disagreed: every SRF-derived name was an `OuterLoop`, and the `Optimizer` registry
held two hand-written entries and no port at all.

The cause was structural, and finding it is what made the fix possible. `Loop`
handed a proposer `select_parents(self.config.elite)` — the top few — and
`Candidate.describe()` carries a score and the edits that produced it but **not the
state**. A proposer therefore saw only the elite's states. Island selection, a niche
grid and a tournament over an archive all need candidates the loop was not handing
over, so those five could not have been proposers however they were filed.

**So the contract changed, not the filing.** The loop hands over the evaluated
population and the optimizer selects; `ProposalContext.elite` carries the count an
optimizer with no parent policy should take, so widening the pool does not silently
change what the existing two propose. Four of SRF's thirteen modes — `scs`, `evox`,
`openevolve`, `karpathy` — moved to `optim/policies.py` as proposers, and
`map_elites` joined them there.

**The split is asserted** rather than described: `13 - len(SRF_OPTIMIZER_PORTS)` is
the loop count, every optimizer port must be in the optimizer registry, no name may
be in both buckets, and every registered name must be one or the other.

**Two defects came out of doing it**, both found by the new tests rather than by
reasoning:

- `MapElitesSelector.features` read `state["nodes"]`, and the nodes are at
  `state["graph"]["nodes"]`. `.get` returns `None` rather than raising, so every
  candidate landed in one cell — a niche grid that is a single bucket and looks like
  it is working. Caught by a test that aims at a known-empty niche.
- `island_of` derived the island from `hash(candidate.id)`, and Python salts string
  hashing per process, so the same candidate sat in a different island on a resumed
  run than it did when it was evaluated. Inherited from the original `IslandArchive`;
  invisible until a run is resumed or a test is run twice. It uses a digest now.

**Held by.** `test_optimizer_policies.py` — one test per policy, each naming the
difference that is the reason it exists, plus the two defect regressions and a test
that the loop still hands over the population. On the standard §3 standard, an
optimizer with no parent policy is asserted to breed from the elite *by id*, because
that ordering is what the loop used before and getting it subtly wrong changed the
search enough to lose a refusal a layer early.

---

## 9. Half of a fix, closed as if it were whole (#36)

Worth recording because it is the same failure the rest of this document audits, made
while auditing.

**What was claimed.** Issue #36's acceptance criteria list seven things, the first
being *"`lowering` can emit `user` (and `agent`) gates; a mode can declare where it
wants one"*. The ledger, the transform, the seeding guard, the fidelity gate and the
inert-when-unused property were all built and tested. The gates were not, and the
issue was closed with "landed in …" above a list that read as complete.

**What the code said.** `lowering.py` hardcoded `evaluator_type="fn"`, and `nn.Node`
had no gate kind at all. **A mode could not declare a pause point**, which is the
mechanism the whole steering design rests on: an intent ledger nobody can be asked
about is a file.

**The fix.** `Node.gate_kind` — `fn`, `user`, or `agent` — with three rules that make
the choice real rather than decorative:

- `fn` requires an evaluator, and the other two **refuse** one, because a gate with
  two answerers has an answer that depends on the runtime.
- `lowering` emits the declared kind and emits a command **only** for `fn`.
- A walker that meets a `user` gate **stops** and records the gate visit, so
  `refactory steer --verdict` is accepted. The person answers with
  `refactory steer --run <dir> --verdict <gate>=<verdict> [--note ...]`, then
  continues with `--resume <dir>`. The answer arrives through explicit
  `RunContext.verdicts` standing authorization, or a one-use ledger decision
  scoped to the current review inputs. Interactive answers are consumed; a later
  gate visit asks again rather than replaying historical approval.

**And a second defect, found by an existing test rather than by this work.** The first
version of `Node.to_state` serialized `gate_kind` unconditionally, which changed
**every mode's state dict** — and therefore every candidate id the loop has ever
recorded, so a resumed run would no longer match itself and the search moved for
reasons that had nothing to do with the search. `test_no_batch_slot_is_silently_lost`
caught it as a vanished batch slot. A field at its default is now omitted.

**Held by.** Six tests: a mode can declare a person-answered gate and it reaches the
IR; a walker run stops at one; a supplied verdict lets it proceed; an unknown
verdict is still refused; a gate cannot have two answerers; and the default is not
serialized. Verified both ways — restoring the hardcoded `fn` fails the first.

---

## 10. Planning steps changed the person's git hooks (#347)

**What happened.** In a design run, while `design_doc` ran, an install in the
run's worktree re-ran the repository's `lefthook install` postinstall. The hooks
in `.git/hooks/` (pre-commit, pre-merge-commit, pre-push) are shared by every
worktree of the repository, so the person's checkout changed before anything was
approved. The target guard held only `emit` to its declared output; the run
reported the change only at materialise, 45 minutes later.

**The four-part fix.**

1. **Suppress hook managers.** `HOOK_SUPPRESS` in `worktree.py` sets
   `LEFTHOOK=0` and `HUSKY=0` in every environment a run uses: `_install_env`,
   `check_environment` (and therefore the wrapper), and `model._environment`
   (every agent worker). An install or command cannot trigger a hook manager.
2. **Per-step shared-state detection.** `agent_delivery.finish`, beside the
   existing target-guard bracket, checks `worktree.shared_changed` after every
   agent step. A change to the shared hooks or configuration is a run warning
   naming the step; for output-only steps (planning steps and emit), it fails
   the step.
3. **Planning guard.** Design's `prepare_design` passes its planning steps
   (`research_director`, `pitfalls_researcher`, `strategy_director`,
   `synthesize_strategy`, `design_doc`) to `repo_change.locate` as
   `output_only`, which merges them
   with `OUTPUT_ONLY = ("emit",)` at `target_guard.record`. The watch/strayed
   bracket that catches emit editing the target now catches a planning step
   doing so. `OUTPUT_ONLY` stays `("emit",)` — design's steps are design's own.
4. **Per-worktree hook isolation.** Every worktree refactory creates gets its own
   hooks path via `extensions.worktreeConfig` and a per-worktree `core.hooksPath`
   pointing to an isolated directory inside the worktree's git dir. A hook manager
   that runs inside the worktree (via `git rev-parse --git-path hooks`) writes into
   this isolated directory, not the shared `.git/hooks/`. This closes the gap where
   a hook manager that ignores its suppress variable still rewrites the shared hooks.

**Held by.**

- `test_worktree.py`: `test_check_environment_suppresses_hook_managers`,
  `test_the_wrapper_suppresses_hook_managers`.
- `test_agent_delivery.py`: per-step shared-state check for output-only and
  non-output-only steps, mode isolation between design and plain `locate`.
- `test_target_guard.py`: a planning step registered as output-only that
  changes the target is refused.
- `test_design_worktree.py`: end-to-end — a planning step that changes shared
  hooks fails the design run.
- `test_worktree.py`: `test_worktree_gets_isolated_hooks_path_and_hook_manager_leaves_target_unchanged`,
  `test_hook_isolation_skipped_when_target_has_core_bare`,
  `test_hook_isolation_skipped_when_target_has_core_worktree`,
  `test_no_hooks_overrides_worktree_hooks_path`,
  `test_config_lock_does_not_block_isolate_hooks_when_extension_already_set`.
- `repo_guard.py` (pytest) and `vitest.global-setup.ts` (vitest): session-level
  guards that fail the test suite when any test changes the repository's hooks,
  config, or worktree list (attributed to the session). Session-wide
  `REFACTORY_HOME` prevents subprocesses from writing worktrees into the real
  refactory home. The guard reports itself inactive, with the reason, when it
  cannot take a snapshot (e.g. not a git repository, or python not found).
- `test_worktree.py`: `test_shared_state_digest_matches_old_formula` (digest
  backward compatibility), `test_session_guard_resolves_symlinked_tmp_root`
  (symlinked temp root resolution), `test_session_guard_reports_inactive_on_non_git_dir`,
  `test_vitest_guard_reports_inactive_when_python_missing`,
  `test_vitest_guard_reports_inactive_on_non_git_dir` (inactive guard reporting).

**Known limit.** A hook manager that writes `core.hooksPath` itself, as husky
does with `git config core.hooksPath .husky/_`, writes the common config even
with isolation on. `HOOK_SUPPRESS` (`HUSKY=0`) stops husky, and `shared_changed`
reports any such change. Hook isolation cannot be enabled when the target's
common config sets `core.bare=true` or `core.worktree`; for such targets the run
falls back to `HOOK_SUPPRESS` and `shared_changed`, and emits a warning naming
the reason.

---

## Two probes that came back clean

Recorded because the point of an audit is accuracy in both directions.

- **`#16`, str knobs.** Probed by sampling a str knob 300 times through
  `edits.random_edit`, and by reading `Loop._other_value`. Both route through one
  `Knob.sample`, and the docstring says why: *"One sampler for the whole system, so
  the starting sweep and the random optimizer agree about what is reachable."* A str
  knob with declared values samples among them; one without gets `STR_MUTATIONS`.
  Implemented.
- **`#25`, path-aware metric integrity.** `nn.fact_produced`'s docstring and body
  both require the producer to lie on a `start → terminal` path. Implemented. (A
  first probe of mine was inconclusive — the off-path node I built was caught by
  `V2` for being unreachable, which is a different rule — and the probe was the
  thing that was wrong.)

---

## 11. Subagent write confinement (#347, C0)

**What the gap was.** Subagents the CEO spawns (via `subagent` and `subagent_fork`)
had no shell (#360) but could `write` or `edit` any path. A subagent could change
the person's checkout or the run's worktree directly, bypassing materialise, the
check, and the reviews.

**The mechanism.** A `tools/pre-execute` listener in `packages/workflow/workflow-spine`
intercepts every `write` and `edit` call. When the calling session is a delegated
child — identified by `delegationDepth > 0` or the presence of `parentSession` in
the session header (NOT by `header.origin`, which the session types document as
presentation metadata only) — the listener walks the parent chain via `parentSession`
to find the session that opened the run (in `openedBySession` for the GUI CEO preset,
or the configured graph's `config.workdir` for the CLI graph profile).

The target path is resolved against the calling session's sandbox policy workspace
root (when a sandboxPolicy service is mounted), falling back to `header.cwd`, exactly
as dsh-tool-fs does. Both the run directory and the resolved target are normalised with
`realpathSync` (where the path exists) via the existing `canonicalWorkdir` helper,
handling discrepancies such as `/var` vs `/private/var`.

If the normalised target does not start with the normalised run directory (plus a path
separator, or equal to it), the call is denied with a message naming the run directory
the subagent may write to instead.

**Behaviour matrix.**

| Scenario | Result |
|---|---|
| Subagent writes inside the run directory | Allowed |
| Subagent writes outside the run directory | Denied |
| Subagent edits outside the run directory | Denied |
| Non-subagent (top-level CEO) writes anywhere | Allowed (the CEO's own `write`/`edit` are separately denied by `ceoDeny`) |
| Subagent whose parent has no open run | Allowed freely |
| Non-write/edit tool from a subagent | Allowed |
| Grandchild subagent (helper's own helper) | Confined; the parent chain is walked |
| `subagent_fork` child | Confined; identified by `parentSession` |

**What it does not cover.**

- **Symlinks.** The listener uses `realpathSync` where paths exist, but a non-existent
  target being written can only resolve its parent directory. A symlink created after
  the check (TOCTOU) is not caught. dsh-tool-fs's own backend owns deeper symlink policy.
- **The CEO's own tools.** Unchanged; `ceoDeny` removes `write` and `edit` from the
  CEO's own scope.
- **Workers.** Workers run in their own headless runtime, not through the spine plugin's
  tool pipeline.

**Held by.** `packages/workflow/workflow-spine/tests/confine-writes.spec.ts`: nineteen
test cases covering both configured-graph and interactiveOpen modes, path traversal,
relative-path cwd resolution, fork children, grandchildren, the deny message content,
and the fail-open case when no run is open.

---

## 14. Post-build findings must cite files the builder can change

**What the issue is.** A post-build reviewer finding that cites only a run document (e.g. `design.md`) sends the run back to `emit`, which cannot change that file. The run produces an empty attempt, which `#369` refuses. The run must be stopped and replayed by hand. b24ar_hooks_refine's attempt 2 was blocked this way: all three reviews cited design.md's backtick-wrapped markers as their only finding, the code itself was complete.

**The rule.** A finding blocks only when the step it is sent back to can fix it. Post-build gate evaluators classify each blocking finding by the structured form `<path>: <text>` at its start. When that path matches a file declared in a node's `writes` (from `.refactory/declarations.json`), the finding is about a run document and is downgraded to a non-blocking note naming the owning step. Every other finding — free-form text, or a path not in any node's `writes` — stays blocking. A finding like `BLOCKING: the change does not implement design.md's Rule 2` stays blocking because it does not start with a structured path; a finding like `BLOCKING: design.md: site markers are in backticks` is downgraded because `design.md` is declared in `design_doc`'s writes.

**Owner derivation.** Owners are derived from the running graph's node `writes` declarations in `.refactory/declarations.json`, not from a hard-coded map. This holds for any graph: design, create, design_fast, edit, refine, and a user's own mode with its own documents. A path no node writes is not a run document. `_run_doc_owners()` in `create_ops.py` builds the mapping at gate time.

**Which gates enforce this.**

- **`gate_qa`** and **`gate_overwatch`** (both in `create_ops.py`) enforce this. Both call `scope_filter_findings()` on the raw blocking findings before deciding whether to reloop.
- **`gate_design_review`** (`design_ops.py`) does not need it — it reads no review prose, and reloops to `design_doc` only when `design.md` has no `## Changes`.
- **`check_artifact`** (`design_ops.py`) does not need it — its findings come from test runner output (assertion errors, test names), not from free-form review text parsed by `blocking_findings()`.
- **`fast_ops.gate_qa`** (`fast_ops.py`) does not need it — it uses structured JSON checks from `focused-qa.json`, not `blocking_findings()` on review prose.
- **`fast_ops.gate_emit`** (`fast_ops.py`) does not need it — it reloops on materialise refusals, not review findings.

**Held by.** `test_gate_stale_findings.py`: `test_classify_finding_structured_worktree_file_stays_blocking`, `test_classify_finding_structured_run_doc_downgraded`, `test_classify_finding_freeform_stays_blocking`, `test_classify_finding_no_path_stays_blocking`, `test_run_doc_owners_from_declarations`, `test_scope_filter_with_custom_graph_document`, `test_classify_finding_exact_path_match`, `test_scope_filter_mixed_findings`, `test_gate_qa_downgrades_design_md_finding`, `test_gate_qa_keeps_worktree_finding_blocking`, `test_gate_overwatch_downgrades_design_md_finding`.

---

## Build-loop stop signal and refusal bound (dm-202, dm-203)

### The b36b evidence

b36b_cc_sysprompt (2026-10-23) built for 3 h 10 min and cost $178.  Its
build-loop limit fired at 17:22 ("review reloop limit reached after 3 review
cycle(s)"), but the spine recorded it as status `ready` with `last_failure`
(dm-95).  The CEO redid past it twice, spending $68 and 1 h 26 min after the
limit had stopped the run.

The recorded error began mid-sentence because the spine kept the tail of
stderr and the reason was at the head (dm-203).

### The fix

**`BuildLoopStop(ValueError)`** in `refactory_dsh.run`: the exception type
all build-loop limit stops raise.  Eight existing `raise ValueError` in
`create_ops.repair` (5) and `fast_ops._repair` (3) now raise
`BuildLoopStop` with the same messages.

**Exit code 3** in `refactory_dsh.run.main()`: `BuildLoopStop` exits 3 with
the `--- refactory failure ---` marker and `format_failure_detail` on
stderr.  Not retried by `run_node` (alongside `BudgetExhausted`).

**The spine** maps exit code 3 with the marker to status `failed` (terminal),
not `ready` (retryable).  `wf_redo` refuses, and only
`refactory graph-control rerun --acknowledge-failure` can proceed.

**Refusal bound:** `consecutive_refusals` fact, `refusal_attempts` knob
(default 3).  Counted by `_count_refusal` in `repo_change`, called from
both `repo_change.materialise` and `create_ops.materialise`.  A refusal
still spends no attempt (#330 E5); the bound is separate.

**`format_failure_detail`:** reason (first line, ≤500 chars), then FAILED
test ids via `parse_test_failures`, then clipped tail, total ≤2000 chars.
`failureDetailOf` on the spine extracts post-marker text, falls back to
head+tail for non-refactory commands.

### Tests that hold each fix

- `test_build_loop_stop.py::test_build_loop_stop_is_a_value_error`
- `test_build_loop_stop.py::test_main_exits_3_for_build_loop_stop`
- `test_build_loop_stop.py::test_main_exits_1_for_other_exceptions`
- `test_build_loop_stop.py::test_run_node_does_not_retry_build_loop_stop`
- `test_build_loop_stop.py::test_walker_stop_reaches_run_result_error`
- `test_build_loop_stop.py::test_consecutive_refusals_end_the_run`
- `test_build_loop_stop.py::test_refusal_bound_holds_across_node_processes`
- `test_build_loop_stop.py::test_format_failure_detail_reason_first`
- `test_build_loop_stop.py::test_format_failure_detail_short_message_comes_back_whole`
- `test_build_loop_stop.py::test_format_failure_detail_1200_char_message_no_repeat`
- `test_build_loop_stop.py::test_a_short_message_with_a_test_id_comes_back_exactly_whole`
- `test_build_loop_stop.py::test_a_design_check_limit_ends_the_walker_run_with_the_stop`: design mode runs its check beside the reviews; `gate_check` raises the check's limit as a `BuildLoopStop` again
- `test_build_loop_stop.py::test_a_design_check_limit_exits_3_on_the_spine`: the same path through the node runner, one process per node, ending at exit 3
- `test_build_loop_stop.py::test_every_limit_raise_in_the_build_loop_is_a_stop`: every `raise` in `create_ops.repair` and `fast_ops._repair` is a `BuildLoopStop`
- `test_build_loop_stop.py::test_an_applied_emission_resets_the_refusal_count_across_processes`
- `test_build_loop_stop.py::test_an_applied_package_resets_the_create_refusal_count`
- `test_build_loop_stop.py::test_create_materialise_stops_at_the_bound`
- `test_build_loop_stop.py::test_the_refusal_bound_follows_its_knob`
- `test_build_loop_stop.py::test_a_run_lowered_before_the_refusal_count_still_materialises`
- `test_build_loop_stop.py::test_a_ceo_redo_keeps_the_refusal_count`
- the spine's correction spec: exit 3 with the marker records `failed` and the CEO's tools refuse; exit 3 without it, or exit 1, records `ready`; the text after the marker is kept exactly; a person's rerun of a completed upstream node after a stop applies and the run continues
- `correction.spec.ts: "exit code 3 with marker records failed and CEO tools refuse"`
- `correction.spec.ts: "exit code 3 without marker records ready (dm-95 unchanged)"`
- `correction.spec.ts: "exit code 1 records ready (dm-95 regression guard)"`
- `correction.spec.ts: "acknowledged rerun after exit-3 stop runs the stopped node again"`
- `correction.spec.ts: "failureDetailOf with run warnings before marker shows reason first"`

## Emission parsing and materialise (#494, dm-208, dm-215)

**Emission path validation.** The path in a `===FILE:`, `===EDIT:` or `===DELETE:`
header may contain any character except whitespace, `=`, backtick, single quote and
double quote. Structural safety is enforced after parsing, by `_artifact_paths` in
`repo_change` (design and design_fast) and `_package_relative` in `create_ops` (create,
create_fast, edit and edit_fast): no absolute path, no `..`, no backslash, no trailing
`/`, and no control or hidden format character. A line that starts like a header but
does not match the format is refused with the offending line quoted, in both parsers.

**Control and hidden format characters are refused by name (dm-208).** A path holding
a character of Unicode category Cc (the C0 controls, DEL, the C1 controls), a
bidirectional control (U+061C, U+200E, U+200F, U+202A to U+202E, U+2066 to U+2069), a
zero-width space (U+200B), a word joiner (U+2060) or a byte-order mark (U+FEFF) is
refused, naming the character, for example `U+202E RIGHT-TO-LEFT OVERRIDE`. A terminal,
a diff or a review hides such a character or shows the name as something else, so the
file a reviewer reads about would not be the file written. Other format characters are
allowed: U+200C and U+200D join letters in Persian names and emoji sequences. A name
that is not UTF-8 (a workspace file such as `tool\xff.py`, which git decodes to a lone
surrogate) is refused the same way, naming the byte, rather than crashing collect (dm-278).
`unprintable_refusal` (in the core module `emitted_paths`, re-exported by `repo_change`)
is the one check, for any place that takes a path from an agent: materialise, create
mode, and a workspace's collect step (dm-278).

**A malformed header goes back to the builder in every mode (dm-215).** In create and
the fast modes, as in design mode, `materialise` records a malformed header as
`emission_refused`, and the gate sends the reason back to `emit` with the line quoted;
the run does not end there. So does a path create mode cannot place inside the package
(absolute, `..`, or a refused character): every path is placed before the package
directory is cleared. The refusal bound above still applies.

**Emitted paths are literal pathspecs (dm-215).** Every worktree git call runs with
`GIT_LITERAL_PATHSPECS=1`: those through `worktree.git`, which delivery's go through
too, and the check's own in `base_check` and `failfast`. A path such as
`app/[id]/page.ts` names that one file and is never read as a pattern, so deleting it
can no longer delete `app/i/page.ts`.

Tests: `test_design_worktree.py::test_a_path_holding_a_control_or_format_character_is_refused_by_name`,
`test_create.py::test_an_emit_path_holding_a_control_or_format_character_is_refused_by_name`,
`test_design_worktree.py::test_an_emission_materialise_cannot_apply_goes_back_to_emit_with_the_reason`,
`test_create.py::test_create_fast_mode_loops_back_on_malformed_header`,
`test_create.py::test_create_fast_sends_a_path_it_cannot_place_back_to_emit` and
`test_worktree.py::test_delete_bracket_path_does_not_glob_expand`.

## Judges decide QA in design mode (b37b, measure4-dm149, b39b)

### The evidence

Design mode decided its QA by parsing reviewer prose for `BLOCKING` lines
(`create_ops.blocking_findings`, read by `synthesize_qa`, `gate_qa` and `gate_overwatch`)
and by a deterministic correction audit inside `gate_overwatch`. Both ended runs that had
nothing left to fix:

- **b37b** spent all three QA rounds, about three hours, on prose the parser read as
  findings: a health check's heading `## 11. BLOCKING: <path>: format preserved in all
  prompts`, and a QA report whose sentence wrapped onto a line that began with the
  quoted marker. **measure4-dm149** lost two of its three rounds the same way, to a
  QA report's tally of the parser's own cases, lines such as ``- `BLOCKING:` → 1 finding ✓``.
- **b39b** failed at `gate_overwatch` after the overwatch stated PASS. The correction
  audit refused it for two sites: one "not in the diff", where the gate correction had
  dropped the guard planned there, so the change rightly left the code alone; and one
  "not reached by any test".

### The rule

The reviews report; judges decide. A judge is a graph node: an agent that writes a JSON
verdict, and a deterministic gate that reads that verdict and never scans prose for
markers (`refactory_dsh.judges`). Both runtimes run the same gate, a Python node, so the
decision is the same on the walker and the spine, and a scripted model drives it in
tests.

- **The check stays binding.** `gate_check` routes before either judge runs. A failing
  check sends the build back to emit, and no judge can pass it.
- **What holds a change back** is remote-factory design-v2's rule, the same for both
  judges: an ask in the brief or a correction the person gave that was not built, the
  check failing, or a critical or high defect in product code (wrong behaviour, data
  loss, a crash, a security problem). Each must-fix item carries its `kind` (`ask`,
  `check`, `defect`, `test`, `docs`, `style`), its `severity` (`critical`, `high`,
  `medium`, `low`) and what it `breaks` (`ask`, `correction`, `health`, `none`). An item
  the judge labels `test`, `docs` or `style` is a follow-up whatever its severity (weak or
  tautological assertions, missing edge-case tests, docs wording, style); an `ask` or
  `check` item always holds the change back; a `defect` holds it back when it is critical
  or high, or breaks an ask, a correction or the health check. An item with no severity is
  high and one with no kind is a `defect`, so a missing label never lets a defect through.
  A gate sends the change back only when its judge says `redo` and names an item that holds
  it back (`judges.blocks`). Otherwise the change goes on and the items are follow-ups on
  the pull request.
- **The QA judge.**
  - `design_ops.synthesize_qa` starts each round. It sets the last round's judging files
    aside (`qa-judge.json`, `overwatch-verdict.json`, `intent-audit.json`,
    `correction-audit.json`, kept under `.refactory/judging/`), so a judge or audit a
    person skips leaves nothing for its gate or the overwatch to mistake for this
    round's. It puts this attempt's three reviews and, when it reported, the verifier's
    into `reviews/qa-synthesized.md` exactly as written, with the check's result, the
    diff stat and the design's Changes section. Nothing reads them for markers.
  - `qa_judge` reads that, the whole `design.md`, the brief, the corrections, what
    sent the attempt back and the validation receipt. It writes `qa-judge.json`:
    `{"verdict": "proceed"|"redo", "must_fix": [{"source", "where", "what", "why",
    "severity", "breaks"}], "follow_ups": [...], "dismissed": [{"source", "finding",
    "why"}]}`. A finding that does not hold against the code, or that only quotes a
    format, is dismissed with its reason.
- **`design_ops.gate_qa`.**
  - A `redo` that names an item holding the change back goes to emit, with the must-fix
    items as the feedback. It spends a review attempt (`create_ops.repair`), so at
    `review_attempts` it is a `BuildLoopStop`.
  - Otherwise the change goes on to the correction audit, and the follow-ups and any
    must-fix items left go to the pull request.
  - Two checks stay structural, as before. A review that left no report, or whose
    sign-off is not current, sends the change back so the reviews run again. A review
    line starting `UNAVAILABLE:` stops the run.
- **An unreadable verdict is read again, judge only.** A judge's file is read when its
  only JSON content is exactly one object: bare, fenced, or among prose. Two or more
  objects are unreadable, because which one is the verdict would be a guess.
  - When the file is missing or unreadable, the gate answers `reread`. That edge goes
    back to the judge alone, with no builder round and no review attempt, and the parse
    error is in the judge's prompt (`qa_judge_reread`, `overwatch_reread`).
  - A code-side count (`qa_judge_rereads`, `overwatch_rereads`, reset each round) caps
    it at the frozen `judge_rereads` knob, 1 by default.
  - Still unreadable, the QA gate goes on with a note, because the overwatch still
    judges. The overwatch gate goes on with a note that acceptance was not judged, and
    the pull request's Reviews line does not say the audit passed.
  - A must-fix item whose `what` starts `UNAVAILABLE:` stops the run: it is how each
    judge is told to report a missing tool or credential. A file with no readable
    verdict and a line starting `UNAVAILABLE:` stops it too, as create's overwatch gate
    does; beside a readable verdict, such a line decides nothing.
- **The judges' prompts carry no reviewer markers.** The overwatch gets
  `JUDGE_REVIEW_CONTEXT` (`wire_reviews(..., judge_nodes=("overwatch",))`), not the
  reviewers' context, which asks for `UNAVAILABLE:` lines in a report and for errors to
  be resolved "in your own report". The QA judge gets no review context.
- **Both judges see what sent the attempt back.** That is `review_feedback`, which
  `prepare_check` sets before the reviews and keeps across the self-check's fix-ups,
  as the reviewers beside the check get it. `validation` would not do: by the time a
  judge runs it holds only this round's check and QA results.
- **The evidence map moves to the QA judge.** The correction audit measures the sites
  the map (`intent-audit.json`) names for each correction, so the map has to exist before
  the overwatch reads the audit. `wire_reviews(..., audit_node="qa_judge")` gives the QA
  judge the map instructions the overwatch carried; create keeps them on its overwatch.
- **The correction audit is evidence.** `design_ops.audit_corrections` runs between
  `gate_qa` and the overwatch. It runs the audit `gate_overwatch` ran
  (`correction_audit_problems`), with `removed_sites`, and reports a cited site the diff
  adds nothing to by why (`base_check.site_outside_change`):
  - `removed` when the change deletes its code;
  - `unchanged` when the change leaves it as the base has it. The audit measures that as
    not met, for the overwatch to decide, never as a pass, because b39b's dropped guard
    and a correction the change never applied look the same;
  - a site past the end of its file, or in a file that does not exist, names no code and
    stays "not in the diff".

  The audit writes `correction-audit.json` and routes nothing. The overwatch's prompt
  carries its summary (`correction_audit`): per clause and site, what it measured as not
  met, the map's own problems, and the checks to confirm by hand. A skipped audit leaves
  the overwatch nothing to read but a note, and the overwatch declares none of these
  files as inputs, so a skip cannot stall it on the spine.
- **The overwatch judges.** It writes `overwatch-verdict.json`:
  `{"verdict": "pass"|"redo", "must_fix": [...], "follow_ups": [...],
  "corrections": [{"id", "met", "evidence"}]}`. `design_ops.gate_overwatch` sends the
  change to emit, spending a review attempt, when:
  - a correction is marked `"met": false`, whatever the verdict. A pass with an unmet
    correction does not ship, and the builder's feedback names the correction and the
    overwatch's evidence;
  - or the verdict is `redo` and it names an item that holds the change back, or a
    correction it could not judge.

  Otherwise the change goes to the report. The follow-ups, any must-fix items left, and
  every correction the overwatch listed without a decision or did not list at all go to
  the pull request, named.
- **No free audit round.** The `audit` edge from `gate_overwatch` back to the overwatch,
  which re-ran it when its own map was malformed, is gone from the design graph. The
  map's problems are in the audit summary the overwatch reads.
- **The report** (`design_ops.report_judged`) requires every review's sign-off to be
  current. The map's shape and the audit's results were the overwatch's evidence, and
  they do not refuse the report a second time.
- **The pull request.** Both judges' follow-ups reach it through `record_follow_ups`
  (`reviews/follow-ups.json`), as before. Its corrections line comes from the overwatch's
  verdict ("Corrections, as the overwatch judged them"), not from the audit's statuses,
  and its Reviews line counts only a readable `pass` as the audit's (`delivery._body`).
- **The verifier** stays in the review fork, design-blind, as one more input to the QA
  judge, because it found a real gap on b39a's round 4. Its report is no longer parsed,
  and `reviews/verification-unresolved.md` is not written in design mode. A verifier
  that times out is simply absent from the judge's input.
- **Both runtimes capture an answer the same way.** When a judge answers in its message
  instead of writing its file, the answer is captured into the node's first declared
  output, `qa-judge.json` or `overwatch-verdict.json`. The walker always did this. On
  the spine the IR keeps writes as a set, and the worker captured into the first file in
  sorted order (`intent-audit.json` for the QA judge). The declarations now carry each
  agent's `answer_file` (`lowering.node_declarations`), and `workers.lower_workflow`
  puts it first in the worker spec. This holds for every mode: create's and edit's
  overwatch now has an answer captured into its report, `reviews/overwatch.md`, which
  its gate reads, where the spine used to put it in `intent-audit.json`.
- **Reviewer prompts are unchanged.** The `BLOCKING` marker and the `[core]` tags decide
  nothing in design mode. create, edit and the fast modes keep `create_ops.gate_qa` and
  `create_ops.gate_overwatch` exactly as they were.

### Tests that hold it

All in `test_qa_judge.py`, through the real design graph with a scripted model unless
named otherwise:

- **Prose and markers.**
  - `test_prose_quoting_the_marker_does_not_send_the_change_back` (b37b);
  - `test_the_synthesis_carries_every_review_unread_and_the_verifier_only_when_it_reported`;
  - `test_the_judges_answer_in_their_verdicts_not_in_review_markers`.
- **Routing.**
  - `test_a_judge_redo_sends_its_must_fix_to_the_builder_once`;
  - `test_the_judges_see_what_sent_the_attempt_back`;
  - `test_a_failing_check_goes_back_before_any_judge`;
  - `test_an_overwatch_redo_counts_toward_the_review_limit_and_stops_at_it`;
  - `test_only_what_holds_the_change_back_sends_it_back` (the severity bar);
  - `test_a_minor_overwatch_redo_goes_on_but_an_unmet_correction_still_goes_back`.
- **Corrections.**
  - `test_b39b_removed_unchanged_and_untested_sites_with_a_passing_overwatch_reach_the_report`;
  - `test_an_unapplied_correction_is_not_met_and_a_pass_that_says_so_goes_back`;
  - `test_an_overwatch_pass_with_an_unmet_correction_goes_back_naming_it`;
  - `test_an_overwatch_redo_for_a_correction_tells_the_builder_which_and_why`;
  - `test_a_pass_lists_what_it_did_not_judge_and_the_must_fix_it_left`;
  - `test_a_proceed_lists_the_must_fix_items_it_left`;
  - `test_a_site_outside_the_change_is_removed_only_when_the_change_deletes_it`.
- **Unreadable verdicts.**
  - `test_a_judgment_is_one_json_object_and_a_redo_names_what_to_fix`;
  - `test_a_verdict_among_prose_is_read`;
  - `test_a_judge_whose_verdict_cannot_be_read_is_asked_again_with_no_builder_round`;
  - `test_a_verdict_still_unreadable_goes_on_with_a_note_the_pull_request_carries`;
  - `test_the_rereads_follow_their_knob`;
  - `test_an_unreadable_overwatch_verdict_is_read_again_then_goes_on_with_a_note`.
- **Structural checks.**
  - `test_a_judge_that_needs_an_unavailable_tool_stops_the_run`;
  - `test_a_review_that_says_a_tool_is_unavailable_stops_the_run`;
  - `test_a_review_that_left_nothing_sends_the_change_back_whatever_the_judge_says`;
  - `test_a_stale_sign_off_sends_the_change_back_whatever_the_judge_says`;
  - `test_the_report_refuses_a_sign_off_that_is_no_longer_current`.
- **Rounds.**
  - `test_each_round_starts_with_nothing_an_earlier_round_judged`;
  - `test_the_audit_writes_its_file_every_round_even_with_nothing_to_measure`.
- **Graphs.**
  - `test_create_and_edit_keep_their_gates`;
  - `test_the_design_graph_has_judges_and_no_free_audit_round`.

  The lowering snapshots hold the rest of create's and edit's graphs.
- **The spine node runner**, each node run as the spine runs it, as its own process
  through `refactory_dsh.run`:
  - `test_gate_qa_on_the_spine_routes_on_the_judge_file`;
  - `test_gate_qa_on_the_spine_stops_at_the_review_limit`;
  - `test_gate_qa_on_the_spine_proceeds_without_a_judge_file`;
  - `test_the_qa_tail_runs_node_by_node_on_the_spine`;
  - `test_a_skipped_judge_never_reads_an_earlier_rounds_verdict_on_the_spine`;
  - `test_an_unmet_correction_reaches_the_builders_prompt_on_the_spine`.
- **The spine's worker:**
  `test_a_judge_that_answers_in_its_message_is_captured_into_its_verdict_file`, for
  both judges and create's overwatch.
- `test_an_unavailable_line_beside_a_readable_verdict_decides_nothing`,
  `test_each_round_gets_its_own_reread`.
- **The spine itself:** `spine.spec.ts` walks the lowered design graph through
  `qa_judge`, `audit_corrections`, both gates and both `reread` edges.

## What this does not cover

- Listing a generated package still imports it (§1).
- The reflective/curriculum differences need a model proposer to show (§3).
- `post_checks` and fork/join are unused (§5).
- The GUI is not audited here at all. Its defects were tracked as issues #60–#73, and the panel's behaviour is held by `packages/client/ui-refactory/tests/` and [tutorial 7](tutorials/07-pick-and-watch-from-the-panel.md).

## 15. Check sandbox file-size limit, truncation guard, signal classification and attempt counters (#330, #348 E8, #349)

## 12. Check sandbox file-size limit, truncation guard, signal classification and attempt counters (#330, #348 E8, #349)

**File-size limit raised.**  The check's `RLIMIT_FSIZE` was 64 MB (`DEFAULT_FILE_MB = 64`) while installs got 16 GB (`INSTALL_FILE_MB = 16384`).  A check that wrote a coverage database, build artifact, sqlite file or test fixture over 64 MB was killed by SIGXFSZ and misread as a code failure.  `DEFAULT_FILE_MB` is now 16384, matching `INSTALL_FILE_MB`, because a check runs the project's full test suite, which writes the same class of files an install does.

**Truncation guard shared.**  Every materialiser now calls `refuse_truncated` from `repo_change.py` before writing a file: `repo_change.materialise` (design, design_fast via `fast_ops.materialise_artifact`), `create_ops.materialise` (create, create_fast, edit, edit_fast).  The function was renamed from `_refuse_truncated` to `refuse_truncated` so the import from `create_ops` is a public API.

**Shell-wrapped signal classification.**  Shell-wrapped signal deaths (exit 128+N) for resource-limit signals are now classified by `check_failure_kind` in `authoring_evidence.py`:

- **137 (SIGKILL), 152 (SIGXCPU), 153 (SIGXFSZ)** are classified as `'uncertain'` (infrastructure), stopping the run without spending a code attempt.  These come from the kernel, the OOM killer or the sandbox's own limits and cannot be the change's fault.
- **134 (SIGABRT), 135 (SIGBUS), 139 (SIGSEGV)** remain `'code_failure'` because they can be caused by a native extension the change broke or a C assertion it triggers.  `run_command` retries them once (as it does for direct signal deaths), and if they persist, the base comparison decides whether the crash is the change's fault or infrastructure.

This is an exit-code-only check with no text scanning.

**Separate check and review counters.**  Check failures and review reloops are counted separately: `check_attempts_used` / `check_attempts`, `review_attempts_used` / `review_attempts`, under the existing `attempts_used` / `attempts` overall cap.  Fast modes keep one `repair_attempts` limit (default 1) but record which kind (`check_repairs_used`, `review_repairs_used`) spent it and name that kind in the stop message.  `refactory status` shows all counters.

**Tests.** `tests/test_check_sandbox.py` has one test per fix: a 100 MB file write passes, exit 137/152/153 are classified as uncertain, exit 134/135/139 are code_failure, exit 1 is still code_failure, the create materialiser refuses truncated text, review reloops do not consume check attempts, and each limit's stop message names itself.

## 16. Delivery envelope covers the merged check (dm-107)

## 13. Delivery envelope covers the merged check (dm-107)

The deliver node's `command_timeout_s` is computed at lowering as
`_check_envelope_s + INSTALL_TIMEOUT_S + NODE_DISPATCH_OVERHEAD_S`. At defaults
(`check_timeout_s=180`): `6540 + 1800 + 600 = 8940 s` (≈ 2 h 29 min). This covers
the full merged-check envelope (three installs, generate, setup and the check
itself) plus one additional install cycle for the merged tree's dependencies and
overhead for the merge and regeneration steps.

The predicate is declaration-based: any op that declares `check_timeout_s` in its
knobs and does not produce `CHECK_PRODUCES` (i.e. is not itself the check node) gets
the delivery envelope. This follows the convention `_is_check_node` uses for the
check node itself.

Without this, the spine's default 600 s kills delivery when the merged check takes
more than 10 minutes. The b27c_recovery run (`check_timeout_s=1800`) was killed at
10 minutes with the merge committed locally and nothing pushed.

**Held by.** `test_lowering.py`: `test_deliver_node_gets_delivery_envelope` asserts
the lowered value equals the formula; `test_deliver_timeout_exceeds_spine_default`
asserts it exceeds 600 s with `check_timeout_s=1800`;
`test_synthetic_deliver_op_gets_delivery_envelope` asserts that a custom op under
another name that declares the knob gets the envelope.

## 24. Delivery conflict resolution loop (dm-102)

When delivery merges the target branch and an authored file conflicts, the run
loops back to the builder automatically instead of pushing behind main.

**The loop.** `deliver → delivery_gate → emit → materialise`. When
`delivery_conflict_files` is non-empty, the `delivery_gate` evaluator records
the conflict (main's tip, the conflicted paths, and the conflict detail) in
`pending-merge.json`, sets `validation` with the conflict description, and
returns `reloop`. The worktree stays clean while the builder works.
Materialise reads `pending-merge.json`, runs `git merge --no-commit
<target_tip>`, resolves generated-file conflicts to theirs (main's side),
applies the emission, and either commits the merge or aborts it — all in one
deterministic step. Materialise refuses an emission that leaves conflict
markers. When MERGE_HEAD is present, `git commit` creates a merge commit.
The check, base comparison and reviews run on the merged result before
delivery tries again.

**The merge exists only inside materialise.** The delivery gate does not
start a merge. It records what materialise needs, then the builder works on
a clean worktree. Materialise's `_commit_emission` is the single place
where `git merge --no-commit` runs, and its except/finally blocks abort it
on any failure. A stop at any point — a crash at emit, a refused or empty
emission, budget exhaustion, or a person's stop — leaves the worktree clean.

**Safety invariants.**

- *Nothing pushes without main.* `_deliver` raises `NotDelivered` on conflict,
  check_failed or abandoned; `push_head` is never set to `found.head` as a
  fallback. ls-remote failure on an existing repository raises instead of
  falling through.
- *Conflict-marker refusal.* `_commit_emission` scans each emitted file for
  `<<<<<<<` / `>>>>>>>` (excluding `<<<<<<< SEARCH` / `>>>>>>> REPLACE`) and
  raises `AttemptRefused` before committing.
- *Every conflicted path in the emission.* During a merge, `_commit_emission`
  refuses before writing unless every `--diff-filter=U` path appears in the
  emission as ===FILE: or ===EDIT:.
- *Worktree never left mid-merge.* The merge runs only inside
  `_commit_emission`. On any failure (edit application, marker check, missing
  path, commit failure), the except block aborts the merge and deletes
  `pending-merge.json`. `_abort_pending_merge` at the top of the delivery_gate
  evaluator cleans up any stale merge from a crashed materialise. After a
  crash between the merge commit and the record write, `require_current`
  adopts the existing merge commit.
- *Loop bound.* `delivery_gate` counts reloops against `check_attempts`
  (default 3). At the limit, it raises an error naming the conflicting files
  without starting a merge. `graph-control redeliver` accepts a run stopped
  there.
- *Delivery advance (dm-116).* When delivery pushes a merge commit, the
  finally block advances the worktree record (head = merge, base = main's
  tip) so a later push fast-forwards.

**Tests.**

- `test_two_runs_conflict_reloop`: two runs editing the same function; the
  second's delivery reloops and the pushed head is a merge commit containing
  main; the merge's second parent is main's tip.
- `test_merge_failure_leaves_worktree_clean`: materialise crashes during a
  merge; no MERGE_HEAD remains.
- `test_emission_with_conflict_markers_refused`: a file with markers is
  refused; a normal SEARCH/REPLACE block succeeds.
- `test_emission_missing_conflicted_path_aborts_merge`: an emission omitting
  a conflicted path aborts the merge and leaves the worktree clean.
- `test_materialise_base_moves_to_main_tip`: removing the base-move logic
  fails this test.
- `test_base_changed_authored_file_conflict`: delivery does not push on
  authored-file conflict; conflict facts are set.
- `test_merged_result_fails_check`: delivery does not push on failed merged
  check; the worktree is not marked succeeded.
- `test_edit_resolves_conflict_keeping_mains_lines`: an ===EDIT: block
  resolves a merge conflict by keeping main's lines.
- `test_ls_remote_failure_stops_existing_repo`: ls-remote failure on an
  existing repository stops delivery instead of pushing without main.
- `test_delivery_advances_worktree_after_merge_push`: after delivery pushes a
  merge commit, the worktree record advances so a later push fast-forwards.
- `test_apply_corrections_no_ledger_leaves_design_unchanged`: with no ledger,
  the agent does not run and design.md is unchanged.
- `test_unresolved_conflict_stops_after_budget`: a builder that never resolves
  stops after check_attempts, with a clean worktree (no MERGE_HEAD, no
  unmerged entries).

## 17. Goal completion requires a finished run (#347, C1)

Goals were marked complete from runs that had failed before review. `spine.next()`
reports a halted run as `done`, so the CEO believed it. The fix: a `tools/pre-execute`
listener in the spine denies `update_goal complete` when spine-status.json records a
non-`done` status. A dedicated `confirm-completion` control in `graph_control.request()`,
reachable only via the CLI (not through any CEO tool), writes a `completion_override`
bound to the current status and invocation into spine-status.json. The override is
cleared when `recordStatus()` writes a different status. The refusal steers the CEO
toward `update_goal blocked` or ending the turn. The C0 write confinement listener also
denies subagent writes to `.refactory/`, so the override cannot be forged.

When the override is valid and `update_goal complete` is allowed through, a
`tools/post-execute` listener appends the override details (confirmed status and
timestamp) to the tool result content, recording it on the goal's completion.

**Held by.** `goal-completion.spec.ts`: tests covering halted/failed/done status,
matching/mismatching override, override recording on the goal, override clearing on
status change, interactive mode workdir resolution, and forgery prevention via
`.refactory/` write denial.
`confine-writes.spec.ts`: test that subagent writes to `.refactory/` are denied.

## 18. Cancelled approval rewrite (#347, C3)

Session #319 treated a cancelled `ask_user_question` as permission and implemented,
committed and pushed without opening a mode. The fix: a `tools/post-execute` listener
rewrites `ask_user_question` cancellation errors (matching `error.info.code` against
`ASK_CANCELLED`, `ASK_ABORTED`, `ASK_TIMED_OUT`) to the unambiguous message "Not
approved: stop and wait for the person."

**Held by.** `cancel-approval.spec.ts`: five tests covering the three cancellation codes,
a successful result, and cancellation of a different tool.

## 19. CEO-dispatched prompt integrity (#347, C2)

When the CEO ran a graph's agent step through its own subagent, the prompt could arrive
with unrendered `{{templates}}`, and retries were rewritten: in #317 the CEO told
reviewers to conclude "PASS, no BLOCKING" and handed overwatch a pre-filled audit. The
receipt did not show what was sent. The fix: `agent_delivery.main()` renders templates
before `begin()` (matching `worker_node.py`). Unresolved template keys from the template
itself (not substituted values) are recorded as a warning in the bridge result and in the
receipt, not refused — the steering ledger folds the brief verbatim into every template,
so a brief containing `{{word}}` must not block begin. A spine `tools/pre-execute`
listener records each prompt the CEO sends through `subagent`, `subagent_fork`, and
`send_message`, keyed by run (workdir) and node. Prompts are cleared on delivery completion
and on fresh dispatch (not on reuse, so parallel reviewers keep their recorded prompts).
For review and audit nodes, `wf_done` checks only the latest recorded prompt for the
node's current delivery and refuses when it does not contain the pending delivery's
prompt.txt.  Prompts are recorded under the due node whose pending prompt.txt they
contain, not under every due node, so a rewritten retry targeting one node cannot
satisfy another.  The spine fails closed: when the reviewer list cannot be fetched,
all nodes are treated as reviewers; when the in-memory prompt cache is empty after a
restart, it re-reads `prompt.txt` from the pending delivery via a bridge action.  The
receipt records `prompt_hash`, `prompt_path`, `sent_prompt_hash`, `sent_prompt_path`
and, when present, `unresolved_keys`.  Unresolved template keys returned by the bridge
are recorded as run warnings in events.jsonl (once per delivery) and stored in the
receipt.  The sent prompt text is written to `<delivery>/sent-prompt.txt`.

**Held by.** `test_agent_delivery.py`: `test_main_bridge_renders_templates_before_begin`,
`test_main_bridge_warns_unrendered_templates`,
`test_main_bridge_accepts_templates_inside_substituted_values`,
`test_receipt_records_prompt_hash`,
`test_reviewers_action_returns_reviewers_list`. `prompt-recording.spec.ts`: nine tests
for wf_done validation of reviewer nodes including rewritten-retry refusal, in-delivery
rewrite refusal, parallel reviewer acceptance, reviewer-list-fetch fail-closed behavior,
faithful-A/rewritten-A′/faithful-B refusal, unresolved template key recording,
unrelated send_message not clearing records, and subagent calls not being recorded.
Prompts are attributed to due nodes by the delivery key (the node id from the execution
contract embedded in prompt.txt), keyed by run and node. A prompt that names no due
node's key is recorded under none and no records are cleared.

## 20. Sacred Rules 8 and 9

Sacred Rules 8 (do not do another agent's job) and 9 (do not skip QA verification) from
remote-factory's CEO instructions were not in the spine's CEO instructions. Both rules
are now in the spine's `systemPrompt.section` at `index.ts`, which both GUI and CLI CEOs
read. Rule 8: the CEO is an orchestrator that delegates all technical work to graph agents
and never takes over their work. Rule 9: every change that produces a PR must pass through
the graph's review steps.

**Held by.** `plugin.spec.ts`: `'CEO system prompt includes delegation and review
discipline rules (Sacred Rules 8/9)'` asserts the section contains the key phrases.

## 21. Case-insensitive .refactory write-deny

On a case-insensitive filesystem (macOS default), a subagent write to
`<run>/.REFACTORY/spine-status.json` would pass the `.refactory/` check and
overwrite the real file, forging `status: done` or a `completion_override`. The
C0 write-confinement listener now compares the resolved target against the
`.refactory` directory using a case-insensitive prefix check (lowercased
comparison), so `.REFACTORY`, `.Refactory` and any other casing is denied.

**Held by.** `confine-writes.spec.ts`:
`'denies a subagent writing inside .REFACTORY/ (case-insensitive bypass) (#347, C1)'`.

## 22. Check-not-run reporting (#363)

When a check could not run, the CEO recommended "Skip check gate and continue
(Recommended)". The fix: `delivery.py`'s `_body()` derives the Check line from the
`validation.status` field in `authoring-evidence.json` (set to `'pass'` only when the
check passes at `design_ops.py:428-429`). When validation is absent, the PR says "did
not run"; when validation has a non-pass status, it says "failed". The Reviews line
checks audit requirement rows: every row must have status `pass` and at least one row
must exist. The run report (`design_ops.py`) derives `"checked"` from the same
evidence. The CEO instructions explicitly forbid recommending skipping a check gate.

**Held by.** `test_delivery.py`: `test_pr_verification_says_check_did_not_run_when_validation_fact_is_absent`,
`test_pr_verification_says_check_passed_when_validation_fact_is_set`,
`test_pr_reviews_line_says_audit_did_not_pass_on_fail_row`,
`test_pr_reviews_line_says_audit_did_not_pass_on_unverified_row`,
`test_pr_reviews_line_not_passed_on_excluded_row_without_user_change`,
`test_pr_reviews_line_not_passed_on_empty_requirements`,
`test_pr_check_line_says_failed_when_validation_failed`,
and `test_run_report_checked_false_when_check_did_not_run`.

---

### Signal-crash classification at preparation call sites (dm-105)

`run_command` already retries once on direct signals (rc < 0) and on shell-wrapped
SIGABRT/SIGBUS/SIGSEGV (exit 134/135/139). A new helper `is_signal_crash(result)`
classifies what survives that retry: it returns `(True, signal_label)` when the
return code is a direct signal or a shell-wrapped signal in `_AMBIGUOUS_SHELL_EXITS`.

Every preparation call site — install (`worktree.ensure_environment`), generate and
setup (`repo_change.prepare_commands`), and delivery's post-merge generate and
merged-tree check (`delivery._regenerate_merge`, `_merge_target`, `_check_merged`)
— calls `is_signal_crash` after `run_command` returns with `not result.ok`. When
the crash persists, the call site normalises the `prep_failure` dict's `returncode`
to a negative signal number so `check_failure_kind` returns `'uncertain'`
(infrastructure), and the run stops without spending a code attempt. A shell-wrapped
command whose child aborts and that exits 134 (as `./install.sh` did) is classified
the same way: `run_command` retries once, and on the second crash `is_signal_crash`
detects exit 134 in `_AMBIGUOUS_SHELL_EXITS`.

In delivery, a signal crash in `_check_merged` or in `_merge_target`'s generate
stops the run as infrastructure (raises ValueError) rather than continuing to push
the reviewed commit. Crash detection uses the structured `infrastructure` flag on
`MergeResult` and the second element of `_check_merged`'s return tuple, not text
scanning for "killed by SIG".

### Base comparison discard-and-rebuild on crash (dm-105)

When any step of the base comparison — install, generate, setup or the check
command itself — crashes (a signal that survives the retry), `run_base_check`
discards `BASE_CHECK_HOME`, the base worktree and the cache file, and rebuilds
from scratch. On double failure, the base environment is discarded again (HOME,
worktree, cache) so the next check gets a fresh start, and the result is returned
with `environment_failure` set but never cached. The run continues without
excusals, and the "continued without excusals" note appears in `check.txt`, the
evidence, and the PR's Verification section.

A setup that leaves a marker in HOME before crashing makes the next check's
setup exit 1 on reuse. This is exactly the b27a rerun failure. Discarding HOME
after a double crash prevents it.

Deterministic (non-crash) failures — a non-zero exit that repeats — are cached
as before. The base is prepared at most twice per attempt, so the check node's
time limit still holds.

### Rolling convergence (dm-109)

After the regular check attempts are spent, another check attempt is granted each
time the attempt's parsed failure set is non-empty, a strict subset of the previous
attempt's set, contains no failure that was not in an earlier attempt, and the
runner completed fully (excusal_blocker returns None on the recorded check output,
not the feedback text).  The extension stops the first time the set does not
strictly shrink.  Hard cap: at most 8 check attempts in total (counting the regular
ones), controlled by the `check_attempt_cap` knob (default 8, declared on design,
design_fast, create, edit and their fast variants).  Setting it to 0 or to a value
at or below the regular check attempts disables the extension.  An empty failure set
is not progress.  The overall `attempts` cap does not refuse an attempt this rule
granted: granted attempts are counted separately via `convergence_extras_used`.
The count of granted attempts and the failure count at each granted attempt are
recorded in the run and the PR's Verification section.

Runner completion is judged from the `check_blocker_history` fact, which records
the excusal_blocker result computed from the full check output at each attempt.
This avoids false refusals when the feedback text (gate_qa's stderr/stdout)
contains warning lines that excusal_blocker would misinterpret.

### Repair feedback augmentation (dm-109)

When a check fails, the builder's feedback includes: (a) the design's `[left: ...]`
sites with their reasons, telling the builder not to change them but to fix the
tests or callers instead; (b) broken guarantees — tests that fail on the attempt,
are absent from the base's failure set, live in files the attempt did not change,
and the base completed fully.

In fast modes, the [left] sites and broken guarantees are stored in the
`focused-check.json` payload's `robustness_notes` field by `focused_check`, and
prepended to the builder's feedback by `gate_qa`.

## 15. Independent verification step (dm-84, #400)

Every reviewer in a design run reads design.md, so gaps the design missed
propagate unchallenged through health check, code review, QA and overwatch.
The evidence log (comment 6252312877 on #400, 54 entries) records every gap
the reviews passed and the method that found it. Of 54 entries, only two
needed the design; the rest were found by design-blind methods:

- **M1** Adversarial inputs
- **M2** Trace a real case
- **M3** Where else does this pattern appear
- **M11** Verify claims against code
- **M12** Shared state
- **M13** Boundary values
- **M15** Weakened tests

For a change to an existing repository the three code reviewers apply M2, M3, M11 and
M12 to every changed function (`REPO_CHANGE_REVIEW_INSTRUCTIONS`), so the verifier
applies only M1, M2, M13 and M15 there, and its time goes to what only a reader
without the design finds. A new artifact's reviewers do not apply M3, M11 or M12, and
fix_pack has no reviewers, so in those the verifier applies all seven.
`prepare_design` sets the `verification_methods` fact by `deliverable_kind`.

The verifier is given the brief, the diff and the repository — never
design.md. It runs in parallel with the reviews as an optional branch of
the review fork, bounded at `DESIGN_VERIFIER_TIMEOUT_S` (1,200 s) in design mode and
`VERIFIER_TIMEOUT_S` (600 s) in design_fast, on both runtimes: the node declares
`timeout_s`, the spine gives its dispatched session that wall (`node_wall_s`) plus the
usual dispatch overhead, and the walker gives its model client the same wall. Its
prompt asks it to finish in about seven minutes and to write each finding as it finds
it, since a session stopped at the wall leaves nothing. 1,200 s is what a medium change
needs: before any bound, 22 of 23 verifier sessions on changes of 500 diff lines or
more finished within it (median 740 s), while under 600 s, 45 of 88 sessions timed out.
In design mode a verifier that times out is retried once with twice the limit
(`Node.retry`, `refactory_dsh.node_retry`). A verifier that fails, returns nothing or
runs out of time again is recorded as missing, and the QA synthesis both judges read
says so, with the reason (`VERIFIER MISSING (timed out twice)`).
fix_pack's verifier is inline on the main path, where running out of time would fail
the run, so it keeps the route's envelope.

**It never gets a builder round of its own.** When a review finds a core defect, the
verifier's findings ride along in the same feedback, with the instruction to answer
each under `## Verifier findings` in emit.txt (fixed, or rebutted with a cited
file:line). Otherwise `gate_qa` writes them, with any rebuttal, to
`reviews/verification-unresolved.md`. Overwatch reads that file when it exists (it is
not a declared input, because on the spine a declared input must exist, and it does
not in a run whose verifier found nothing) and makes a finding
that breaks an ask or existing behaviour, unrebutted, its own core finding, which
sends the build back; the pull request lists the rest. In the fast modes, with no
Overwatch, the pull request lists them. A round with no verifier finding empties the
file, so an earlier round's findings are never reported as current.

This replaced an advisory builder round (#475) that the verifier took whenever the
reviews passed and it did not: a full emit, check and review cycle spent on findings
no other gate judged.

In design mode none of this parsing happens: the verifier's report goes, whole, to the
QA judge with the three reviews, which weighs it like any other review. See "Judges
decide QA in design mode".

### Why design-blind

If the verifier read design.md, it would share the same blind spots as
every other reviewer. By working from the brief and the code alone, it
catches gaps the design missed, which is the failure mode the evidence log
documents.

### Pipeline position

Full mode (the check runs beside the reviews; see "The check beside the reviews" below):
```
prepare_check →(proceed) fork_qa → [health_checker, code_reviewer, qa_director, verifier, check] → join_qa → gate_check →(proceed) synthesize_qa → qa_judge → gate_qa
```

Fast mode (design_fast):
```
focused_check → gate_check →(proceed) fork_qa → [focused_qa, verifier] → join_qa → gate_qa
```

### Measurement

The replay harness (`python -m refactory_dsh.replay_harness`) scores the
verifier against the evidence log. See docs/replay-harness.md.

## 19. cwd escape detection (#351 I3)

**What should happen.** When a check's command changes directory outside the run's
worktree, the check record and the reviewers' evidence say so, naming the directory.
Advisory, not a failure.

**How it works.** `sandbox.run_command` accepts an optional `worktree_root`
parameter. When given, it wraps the command in a shell that installs an EXIT trap
comparing the final `$PWD` with the worktree root (both realpath'd, prefix check).
The trap writes to a sentinel file, not stdout or stderr, so the command's output
is unchanged. `SandboxResult.cwd_outside` carries the escaped directory, which
`design_ops` and `fast_ops` record as `cwd_escaped` in the check outcome.

**What it does not catch.** An intermediate `cd` that returns before the command
ends. A subshell that `cd`s. A symlink whose target resolves outside the worktree
is caught by `pwd -P` because the trap resolves both paths.

**Test.** `test_cwd_escape.py`: `cd /tmp` is flagged; `cd <subdir>` is not; `exit 3`
keeps status 3; the sentinel never reaches stdout or stderr.

## 20. Worktree isolation for create/edit modes (#351)

**What should happen.** Two parallel create runs of the same package name both
succeed in their own staging directories.

**How it works.** `create_ops.prepare_create()` sets up a staging directory at
`<run>/.factory-staging/.factory/packages/<name>/`. `materialise()` writes there,
`validate()` imports from there (via `registry.install_workspace` with
`project_override`), and `register()` copies atomically to the shared
`.factory/packages/<name>/` — refusing if the shared package changed since prepare.

**Test.** `test_staging_isolation.py`: two runs through `prepare_create`, then
`register`; the second register refuses naming the first, and the first's package is
intact; a reloop does not refuse itself; an edit whose source changed mid-run refuses;
an edit whose `revision_of` differs from its name copies from the source.

## 21. Symlink safety in copy_uncommitted (#351)

**What should happen.** `copy_uncommitted` must not follow symlinks.  A tracked
absolute symlink that was retargeted must not overwrite the old target outside the
worktree.  An untracked symlink to a credentials file must be recreated as a link,
not have its contents committed.

**How it works.** `worktree.copy_uncommitted` checks `source.is_symlink()` before
reading contents.  Symlinks are recreated with `os.readlink` + `os.symlink`; the
hash records the link target string, not the file content.  Before writing any
destination, an existing symlink at that path is unlinked so writes never follow
a stale link.  Regular-file destinations whose resolved path escapes the worktree
(e.g. a parent component is a symlink) are refused.

**Test.** `test_uncommitted_changes.py`:
`test_symlink_retarget_leaves_old_target_untouched` — a retargeted tracked
symlink is recreated as a link and the old target is untouched;
`test_untracked_symlink_outside_repo_recreated_as_link` — an untracked symlink
to an external file is recreated as a link, not as the file's contents.

## 22. Staged package validation resolves other workspace packages (#351)

**What should happen.** A staged package whose ops import from another workspace
package (e.g. `from refactory_packages.base_ws import ops`) validates and
registers, as on main.

**How it works.** `check_package` passes `shared_project=project_for_nodes()` to
`verify_sandboxed` when `project_override` is given.  The subprocess installs the
shared workspace behind the staging root via
`registry.install_workspace(shared, project_override=staging)`, so the staged
copy wins for its own name and every other package still resolves.

**Test.** `test_staging_isolation.py:test_staged_package_can_import_shared_workspace_package`
— a staged package that imports `HELPER` from a shared workspace package validates
without import errors.

## 23. Copy-on-register atomicity (#351)

**How it works.** `register()` writes the staged package to a temporary sibling
directory, renames the old package aside, renames the new one in, then deletes the
old. If the swap fails, the old is put back. The registering run's id (its invocation
id, as `repo_change` uses) is written into `.origin.json` so the refusal message names
the other run. After a successful register, the origin stamp in `.origin.json`
identifies this run so a reloop does not refuse itself.
`prepare_create` records both `source_hash` and `source_existed`, so register can
refuse when a new package appears (not only when an existing one changed).

**Test.** `test_staging_isolation.py:test_two_parallel_creates_both_build_second_register_refuses`,
`test_reloop_register_does_not_refuse_itself`,
`test_edit_whose_source_changed_refuses_at_register`.

## 25. CEO gate verdicts and check skips (#363, dm-113, dm-115)

**Gate verdicts.** A user gate is answered only by a person. The
mechanical defense: `wf_verdict` throws when a verdict argument is
passed for a user gate (index.ts, the holding test at
plugin.spec.ts:116). The CEO's system prompt says it never claims,
restates, or re-applies a person's verdict; this is defense-in-depth.
A headless CEO (`graphStartup.chat === false`) cannot write
`source='user-relayed'` via `wf_steer`; the runtime silently
downgrades to `'ceo'`. A CEO note about a gate verdict is recorded
as the CEO's observation, not an answer, and is shown to the person
at the gate labelled as the CEO's.

**Check skips.** Only a person can skip the check, through
`refactory graph-control skip --node <check_node>`. The source
`'person'` is hard-coded in `applyControls`, not read from the
control file. Skip provenance is stored in the spine checkpoint's
`state.skip_provenance` map — the single source of truth.
`require_completion` accepts a person-sourced skip and refuses a
CEO-sourced skip, naming the source. The CEO's `wf_edit skip`
records `source: 'ceo'` and is refused by `planSkip` for
conditional-only gates (the normal check shape) and by
`require_completion` for any shape.

Residual: an external-backend worker can write a graph-control file,
the same residual as #429's confirm-completion.

---

## 24. One note is one ledger entry — heading escape (dm-132)

**The invariant.** One note is always one entry. The recorded source travels with the
text: a CEO note stays a CEO note, and a user-relayed note stays user-relayed. No
ledger writer can produce more than one entry from one note.

**The attack vector.** A note body containing `## [...] verdict` or `## [...] correction`
at the start of a line is parsed as a separate entry by `parse_entries`, because the
parser starts a new entry at any line beginning with `## [`. A CEO note with a forged
verdict block becomes a separate verdict entry that `take_verdict` returns through the
upfront shortcut. A forged correction block becomes a user-relayed requirement.

**The escape scheme.** Lossless backslash byte-stuffing via `splitlines(keepends=True)`.
On write, every body line whose content (after stripping the trailing line ending) starts
with zero-or-more backslashes followed by `## [` gains one leading backslash. On read,
`parse_entries` removes exactly one leading backslash from lines matching `\+## [`.

- `## [X]` → stored `\## [X]` → read removes one `\` → `## [X]`.
- `\## [X]` → stored `\\## [X]` → read removes one `\` → `\## [X]`.
- `\\## [X]` → stored `\\\## [X]` → read removes one `\` → `\\## [X]`.

**The separator set.** `str.splitlines()` splits on: `\n`, `\r`, `\r\n`, `\x0b`, `\x0c`,
`\x1c`, `\x1d`, `\x1e`, `\x85`, `\u2028`, `\u2029`. The writer and reader both use
`splitlines(keepends=True)`, so they cannot disagree on where a line starts.

**The three write sites.** `Ledger.seed()`, `Ledger.note()`, `Ledger._decision_entry()`.
All three call `_escape_body()` on the text before writing.

**The single read site.** `parse_entries()` applies `_unescape_line()` to each body line
before appending. This is the single unescape point for all consumers (14+ call sites
including `entries()`, `take_verdict()`, `verdicts()`, `render_entries()`,
`requirements()`, `identity()`, and `agent_delivery.begin()`).

**Verdict single-line check.** `verdict()` rejects a gate or value that spans multiple
lines under any line separator (not just `\n`), using `len(x.splitlines()) != 1`.

**Both runtimes.** The ledger is shared by the DSH runtime and the walker runtime. Both
read through `parse_entries` and both write through the three `Ledger` methods, so the
fix applies to both runtimes.

**The legacy `\## [` edge case.** A pre-existing line starting with `\## [` in an old
ledger will lose one backslash on read. This is theoretical — no plausible legacy content
starts with `\## [`.

**Test.** `test_steering.py`: `test_forged_verdict_in_ceo_note_does_not_produce_verdict`
(parametrized over 6 separators), `test_forged_correction_in_ceo_note_not_a_requirement`
(parametrized over 6 separators), `test_seed_with_forged_heading_stays_one_entry`,
`test_decision_entry_body_escaped`, `test_literal_backslash_heading_round_trips`,
`test_verdict_rejects_multiline_gate_or_value`,
`test_existing_ledger_parses_unchanged`, `test_legacy_backslash_heading_edge_case`,
`test_escape_does_not_affect_other_markdown_headings`.

---

Two things that were on this list have moved off it, which is the point of keeping it:

- **mathlib is no longer untested.** It was installed and the mathlib problem ran, and
  doing so found two defects that no stub could have found (§8). The mathlib tests are
  gated on `MATHLIB_PATH`: they run where mathlib is present and skip with the reason
  where it is not, so a CI machine without it is honest rather than green by omission.
- **SRF's own optimizer is no longer unused.** `map_elites` is ported, counted
  separately from the thirteen modes, and asserted by name (§8).


## 25. Per-correction audit (#431)

**How it works.** `correction_audit_problems()` in `authoring_evidence.py` runs
at every post-build gate (`gate_overwatch` in create and edit, `gate_qa` in fast
modes). In design mode it runs as an op of its own before the overwatch and blocks
nothing; a site the change deletes is `removed`, and one it leaves as the base has it is
`unchanged`, measured as not met for the overwatch to decide; see "Judges decide QA in
design mode". For each correction or constraint in the ledger it reads the overwatch's
`intent-audit.json`, finds the clause-level `sites` arrays, and checks three
things for every clause:

1. **Site in diff.** The cited `file:line` must fall inside a hunk of the
   run's `git diff base..HEAD`. A `not_in_diff` result is a failure.
2. **Test reachability via site revert.** `compute_site_reverts` (§27) reverts
   only the overlapping hunks, runs the named test, and records whether it
   fails. A test that still passes after the revert does not reach the site.
   A code site with an empty `tests` list fails the clause.
3. **Evidence scope.** `patch_scope` (§26) checks whether the named test
   patches the project's own modules on the tested path. A test that does
   is `local`, not `connected`.

**Measured failures block; unmeasured checks are advisory.** A clause is BLOCKING
when the audit measured a miss:
- the site is `not_in_diff`;
- the test still passes after the revert (`pass`: it does not reach the site),
  unless the site is documentation (below);
- the cited test ran no tests (`no_tests`: a wrong test id);
- a code site has no tests and its change is not comments only (a documentation
  site in the diff needs none, below);
- the test patches the changed function (§26);
- any status the audit does not know.

The message names the correction id, the clause, the site and the test.

A check the audit could not measure is advisory instead, and so is a
documentation site. The run proceeds, and the pull request's Verification
section lists it under "Correction-audit checks to confirm by hand (advisory)"
for a reviewer to check. It is also in
`correction-audit.json` (clause status `advisory`) and `reviews/overwatch-gate.txt`
or `fast-report.json`. These cases are:
- `not_run`: the time cap ran out, the test timed out or was killed, the patch
  did not apply, or git failed;
- `collection_error`: the revert broke test collection, so it measured nothing;
- `import_only` or `new_file` with no caught sibling: the hunk only adds new
  definitions, so reverting it can only break an import;
- a `doc` clause naming a changed file that is not documentation (wording in a
  code file);
- a site in the diff in a prose document (`.md`, `.markdown`, `.rst`, `.adoc`),
  whether no test catches its revert or none is cited. A document is read, not
  run, so no test is expected to reach it, and the diff already shows the site
  changed. A lab run spent a whole builder round adding a test that read a "See
  also" link back from a Markdown file. A documentation site that is not in the
  diff still blocks. A `.txt` file or a file under `docs/` is not prose for this
  rule: it may be data, a fixture or code, and its uncaught revert still blocks.
  A Markdown file that drives behaviour, such as an agent prompt, is prose here,
  so the pull request is where a reviewer checks it.

A measured miss is never hidden by an unmeasured check: a wrong test id or an
unknown result on any test of a site blocks, even when another test of the same
site was not run.

A site whose change is comments only passes without a test, since no test can
catch a comment (`comment_only`). Each added line is judged in place in HEAD's
file: Python with `tokenize`, TypeScript and JavaScript (not JSX) with a scan that
knows strings, template literals and regular expressions. A `#` or `//` inside a
string, a block or JSDoc comment followed by code, a pointer dereference such as
`*p = 0;` and a directive such as `//go:embed` are code. A file of any other kind
is never comment-only.

**Withdrawn corrections.** A correction whose audit row has `status == "excluded"`
with a `user_change` pointing to a later ledger entry is skipped — it was
withdrawn. This reuses the existing exclusion validation path.

**Evidence forms.** Not every clause has a code site:
- **prohibition:** requires a `pattern` field; code confirms the pattern is
  absent from the `+` lines of the diff. Missing `pattern` fails the clause.
- **doc:** requires a `path` field naming a changed documentation file
  (`.md`, `.rst`, `.txt`, or under `docs/`), with or without a `:line`. A path
  to a non-doc file passes as advisory when that file changed, and fails when it
  did not.
- **design:** requires a non-empty `heading` field that exists in the current
  `design.md`. Missing or unknown heading fails the clause.
- **test:** cites a code site plus the test; the revert runs against the code
  site, not the test file. A site inside a test file is not a code site: it
  passes when it is in the diff (`test_file`) and fails when it is not.

A clause whose `sites` is not a list fails with "sites must be a list".

**New-file sites.** A site in a file the base lacks (`new_file` result) or a
site whose revert fails only with an import error on a name the reverted hunks
define (`import_only` result) passes when the same clause has a sibling site in
an existing file whose revert fails its test. Without one it is advisory.

**Test.** `test_correction_audit.py::test_correction_audit_unbuilt_blocks`,
`test_correction_audit_no_base_does_not_block`,
`test_large_ledger_past_the_cap_proceeds_with_an_advisory` (full design, through
`gate_overwatch` and the pull-request text),
`test_design_fast_past_the_cap_proceeds_with_an_advisory`,
`test_a_measured_miss_still_blocks_beside_unmeasured_pairs`,
`test_a_comment_only_site_needs_no_test`,
`test_a_doc_clause_on_a_changed_code_file_is_advisory`,
`test_a_documentation_site_needs_no_test_but_must_be_in_the_diff`.

## 26. Evidence scope enforcement (#431)

**How it works.** `patch_scope.py` checks each named test for patches of the
project's own modules on the tested path. It checks the specific test function
plus the fixtures it requests (including conftest fixtures), not the whole file.

Scope definitions (from `code_reviewer.md` category 8, quoted as written):
- **local**: a unit test exists that tests the stated outcome in isolation
- **connected**: a test exists that calls a real entry point crossing a module
  boundary, mocking only external I/O
- **assembled**: a smoke test or startup check verifies end-to-end behavior
- **artifact**: the artifact exists and has correct content

This repository's rules:
- a test that patches the project's own modules on the tested path is local,
  not connected;
- assembled means driving a mode's `run()` with no project-module patch on the
  path;
- artifact means reading the delivered output.

**Allowlist.** Targets that are always allowed: clocks (`time.time`,
`datetime.datetime.now`, etc.), randomness (`random.*`, `Math.random`) and
environment (`os.environ`, `process.env`).

**Detection.** Python: `monkeypatch.setattr` (string and object forms),
`monkeypatch.setenv`/`delenv` (allowed), `unittest.mock.patch`,
`unittest.mock.patch.object`, `MagicMock` assignments. TypeScript: `vi.mock`,
`vi.spyOn` with regex-based detection. Project modules are relative imports
plus workspace package names.

**Test.** `test_patch_scope.py` (11 tests covering each detection pattern).

## 27. Site-revert checks (#431)

**How it works.** `compute_site_reverts()` in `base_check.py` mechanically
reverts each changed site to the base and runs the named test to check whether
it catches the change.

1. Get the diff hunks via `git diff -U0 base HEAD -- file`.
2. Find hunks whose added-line range overlaps the site's line range.
3. For pure-deletion hunks, the site line is the hunk position or the next line.
4. Build a reverse patch of only the overlapping hunks.
5. Apply with `git apply --reverse --unidiff-zero` (the `--unidiff-zero` flag
   handles pure-addition hunks).
6. Run the named test with `pytest <file>::<name>` or `vitest run <file> -t
   <name>`, in the attempt's worktree under `worktree.check_environment`.
7. Classify the result:
   - `fail` → the test assertion fails, the site is caught ✓
   - `pass` → the test still passes, the site is NOT caught ✗
   - `collection_error` → import/collection error, not a catch
   - `import_only` → fails only due to import of a name the reverted hunk
     defines, not a catch
   - `no_tests` → the cited test does not exist (pytest exit 4 or 5)
   - `not_run` → the time cap ran out, the test timed out or was killed, the
     patch did not apply, or git failed

   Before any revert, a site in a test file is recorded as `test_file`, and a site
   whose added lines are all comments as `comment_only`. Neither is reverted.
8. Restore the file with `git checkout HEAD -- <file>` in a `finally` block.

**Test-file exclusion.** A site inside a test file (`test_*.py`, `*_test.py`,
`*.spec.ts`, or under a `tests/` directory) is not a code site. It is checked
against the diff and never reverted.

**Cache.** Measured results are cached per `(HEAD, path:line, test_name)` in
`.refactory/site-revert-cache.json`, a dedicated file separate from
`base-check.json`. A `not_run` result is never cached, so a later gate measures
the pair. An `unavailable` entry cached by an earlier version is measured again.

**Time cap.** Total revert time is capped below the gate's DSH timeout
(`SITE_REVERT_TOTAL_S`, 240 s; `SITE_REVERT_PAIR_S`, 60 s per pair). Past the cap,
the remaining pairs are `not_run`, which is advisory (§25). The cap is checked only
after the site itself: a site that is not in the diff, or whose path lies outside the
run's worktree, is `not_in_diff` whatever the time.

**Measuring honestly.** Each revert runs `python -B -m pytest`, so a module reverted
twice within one second never runs from a stale `.pyc`. A test that passes is never a
`collection_error`, whatever it prints, and a cited test that is not a test id is
`no_tests`.

**Blocking rule.** A measured miss is BLOCKING (§25). The only exemptions are:
1. A `## Site revert exemptions` section in `design.md`.
2. An explicit no-base record (`{"no_base": true}`) — modes without a base skip
   diff and revert checks.

**Test.** `test_correction_audit.py`.

## 28. Non-core findings become follow-ups (dm-171)

**What the issue was.** On 2026-10-22 no lab run finished on its own. Reviewers sent
builds back for findings that were real but outside the change: a missing extra test,
a doc, an edge case the brief never asked for. Each cost a full builder round and a
review cycle, and a run that spent its attempts on them failed with the change's
core already working. b34b built 9 times with clean reviews before its last round;
b33c built 8 times as each round's reviewers found new gaps.

**The rule.** A build is held back only by a core defect. Reviewers end the first
line of each BLOCKING finding with `[core]` (an ask, a criterion, a correction or
constraint, or a regression) or `[non-core]`; `create_ops.SEVERITY_RULE` is the one
instruction every post-build reviewer prompt carries, and `create_ops.is_non_core`
the one parser. An untagged or doubly tagged finding is core, and so is one whose tag
sits on a continuation line: a reviewer that writes an unmet ask and then a
`- docs also missing [non-core]` line has written a core finding. `create_ops.gate_qa`,
`create_ops.gate_overwatch` and `fast_ops.gate_qa` all split findings with
`split_core`:

- core findings send the build back as before, with the non-core ones riding along;
- non-core findings alone never cost a round, never spend a review attempt, and
  never stop a run;
- Overwatch is told to state PASS when only non-core findings remain
  (`OVERWATCH_VERDICT_RULE`), so a FAIL always blocks — it may rest on an ask named
  only in prose. A PASS with non-core findings proceeds only when the intent audit
  passes every requirement, and the evidence and correction audits hold;
- in the fast modes, a failed check whose name ends with the tag is non-core too.

What a gate proceeds with is recorded by `record_follow_ups` and reaches the pull
request through `delivery._body`.

Design mode reads no tags: its QA judge and overwatch apply the same rule as judges and
list the follow-ups themselves ("Judges decide QA in design mode").

**The cap.** A full-mode run has `attempts` builder rounds (`check_attempts` in design;
three by default), shared by check failures and review reloops. The verifier never
adds a round of its own (§15).

**Held by.** `test_follow_ups.py`.

## Pin integrity

### Repository guard

The `repo_guard` pytest plugin (loaded via `conftest.py`) snapshots the
repository's worktree list, hooks and git config at session start and compares
at session end. Any change attributed to the test session fails the suite.

Pin tests use temporary git repositories created with `git init` in `tmp_path`.
No test creates a worktree in the real repository under test, so the guard never
fires for pin operations.

### Hooks isolation during pin preparation

A pin's `pnpm install` runs prepare scripts (such as `lefthook install`) that
can rewrite the driver's shared hooks. Pin creation isolates hooks:

1. `extensions.worktreeConfig` is enabled on the driver (the same extension
   run worktrees already enable, and the canary checks already expect).
2. `core.hooksPath` is set to `<gitdir>/hooks.isolated` as a per-worktree
   config on the pin (the same path `worktree._isolate_hooks` uses for run
   worktrees), so hooks write to the pin's own git dir, not the driver's
   shared hooks.
3. The `LEFTHOOK=0` and `HUSKY=0` environment variables are set during
   `pnpm install` and `pnpm build`.

### Dirty driver pinning

When a driver has uncommitted changes to machinery paths at launch, the unified
diff is recorded in the driver record (`dirty_diff`). When a pin is created,
`git apply` applies that diff to the pin's worktree. If the diff conflicts with
the pin's commit (the checkout has moved and the changes overlap), pin creation
refuses with a `RuntimeError` naming the commit and the conflicting paths.

### File lock for concurrent pin creation

`create_pin()` acquires an exclusive `flock` on
`<refactory_home>/pins/<key>.lock` before preparing a pin. Concurrent callers
block until the first one finishes, then find the prepared pin and return it.
A rebuild (`require_build`) is also refused when a live run uses the checkout:
the run is registered on its driver at launch, and `_refuse_if_live_runs`
checks every registered run's owner lock before allowing a build.

### In-place pin preparation

The pin is prepared directly at its final path (`<refactory_home>/pins/<key>/`),
guarded by an exclusive file lock (`<key>.lock`). The `pin.json` file is the
readiness marker: it is written only after all preparation steps (worktree
creation, hooks isolation, dirty diff application, `uv sync`, `pnpm install`,
`pnpm build`, and profile provisioning) succeed. A failure at any point removes
the pin directory and propagates the error. Concurrent callers block on the lock
and, after the first one finishes, find the prepared pin via `pin.json` and
return it immediately.

### Worker dispatch under a pin

The dispatched worker (`worker_node.py`) compares `sys.prefix` (the venv root)
with the pin's venv, not `os.path.realpath(sys.executable)`.  A uv venv's Python
is a symlink to the shared uv interpreter, so `realpath` resolves both the
driver's and the pin's binaries to the same target and the re-exec would never
fire.  When the prefixes differ, the worker re-execs with
`[pin_py, '-m', 'refactory_dsh.worker_node', *sys.argv[1:]]` so the pin's
package is loaded rather than the driver's `worker_node.py` file path that
`sys.argv[0]` holds under `-m` dispatch.

A re-exec loop guard prevents infinite re-execs when the pin's venv is broken:
the environment variable `_REFACTORY_WORKER_REEXEC` is set before `os.execv`.
If the re-exec'd process still finds itself on the wrong venv, it fails with
a diagnostic message naming both interpreters instead of re-exec'ing again.

### Drift scope

`detect_drift` counts only fingerprint paths under the driver checkout as
driver drift, using `Path.is_relative_to` on resolved paths so a sibling
directory like `<checkout>-runs/` is never counted.  Paths under `.factory/`
inside the checkout (the run's own graphs and artefacts) are also excluded
from driver drift.

Changes to non-driver paths — the run's own `graphs/` directory, project
packages — are not driver drift and do not trigger pinning.  Recovery's
`prepare()` and the supervisor's `prepare()` calls still refuse those changes
with "prepared configuration … changed; dependent results need explicit
invalidation" even when the driver has moved and a pin is being created.
The `skip_driver_paths` parameter tells `prepare()` which fingerprint paths
to skip (the driver's); all others are still validated.

### Stale-build handling on resume

`interactive.run()` passes `check_only=True` to `require_build()` on resume
and respawn.  A stale build on a resume child warns to stderr instead of
attempting a rebuild.  Where a rebuild is attempted (a fresh launch with no live
runs), a failed `pnpm run build` is caught and logged as a warning rather than
stopping the launch.

### Supervisor drift detection

The supervisor loop calls `recovery.prepare()` at two sites: the resume-pending
path (where a person explicitly resumes) and the transient-exit respawn path.
Both sites detect drift before `prepare()` and, when the driver has moved,
resolve the pin and pass `skip_driver_paths=` the driver checkout to
`prepare()`.  This skips only the driver's fingerprints; non-driver
fingerprints (the run's graphs, project packages) are still validated and
raise the "prepared configuration … changed" refusal as on main.  The
re-exec'd owner child validates against the pin's files after re-exec.

---

## 28. Target write protection for worktree runs (dm-159)

### The incident

On 2026-10-21, a worktree run's builder wrote 14 tracked files and added 6 new
files directly into the target checkout instead of its worktree. Every process
started on the driver checkout during that window imported the edited code. Three
defenses failed together:

1. **A prompt edit dropped the scope rules.** The CEO called `wf_edit set-prompt`
   on emit, replacing the whole template and dropping `{{emit_scope}}` (the paths,
   the `===FILE:`/`===EDIT:` format, and "Write nothing but emit.txt").
2. **The execution contract pointed at the target.** `_execution_contract` names
   `refactory_source` and `runtime_root` from `Path(__file__)`. For a run whose
   target is its own driver checkout, those paths lie inside the target, and the
   agent wrote there.
3. **The guard only warned.** `target_guard.strayed()` routed target-root changes
   to the warning list when `also` (the worktree) was set. `finish()` wrote a run
   warning and the receipt said "delivered".

### Why a target write is refused

An agent's changes must go through materialise: `emit.txt` → `materialise` →
committed on the run's branch → the check → the reviews → `deliver`. A direct
write to the target bypasses every step after emit.

### The three fixes

1. **`strayed()` refuses target changes for worktree runs, and every agent is
   guarded.** When `also` is set (a worktree exists), target-root changes go to
   the fail list. Non-output-only agents get a target-only guard. The target is
   actively restored to its pre-agent state on refusal, using bytes saved at
   `watch()` time.
2. **`_execution_contract` rebases paths onto the worktree.** When
   `refactory_source` or `runtime_root` lies inside the run's target, it is
   rebased onto the worktree. The contract text tells the agent to write there,
   names the target as off-limits, warns that the python interpreter is for nested
   dispatch only, and names the in-worktree wrapper for repository commands.
3. **`amendPrompt` refuses a `set-prompt` that drops placeholders.** When the
   current template contains `{{...}}` patterns and the replacement text drops
   any, the edit is refused with an error naming the missing placeholders and
   suggesting `append-prompt`.

### Scope

The refusal applies to every agent node of a worktree run, not only output-only
nodes. Reviewers, the verifier, overwatch, apply_corrections and
expanded researchers are all guarded.

Concurrent runs on one target refuse each other's in-flight nodes when either
strays. A write that finishes after the node's `finish()` is reported only as a
later warning.

In a self-targeting run, a pull or commit in the driver checkout during an
output-only node now refuses that node. The driver checkout must stay still
while runs are in flight.

The driver or pin checkout, when it is not the target, is watched too: see
[Refactory's own checkout](#32-refactorys-own-checkout-dm-164).

### Behaviour changes

| Change | Applies to |
|---|---|
| `strayed()` target-root changes → fail (Python, `target_guard.py`) | Both runtimes (walker and spine-driven DSH) |
| Guard all worktree-run agents (Python, `target_guard.py`) | Both runtimes |
| Target restoration on refusal (Python, `target_guard.py`) | Both runtimes |
| `_execution_contract` path rebase + worktree instructions (Python, `agent_delivery.py`) | Both runtimes |
| `amendPrompt` placeholder guard (TypeScript, `spine.ts`) | Spine runtime only (the walker has no `wf_edit`) |
| CEO instructions and `${toolPrefix}_edit` description update (TypeScript, `index.ts`) | Spine runtime only |


## Fail-fast targeted pre-pass (dm-128)

When the previous check recorded failing tests that were not inherited from
the base, the next check runs those tests first through the
targeted-command builder (`build_probe_command`), with a short timeout of
`min(check_timeout_s // 3, 120)`.

In `prepare_check` the pre-pass is now part of the self-check (b41a, below): the last check's failing tests join the self-check's selection, and
`prepare_check` has one test-running path. A test that failed in the last check
and still fails now gets up to `self_check_fixups` (2) fix-ups, spending no
attempt, before the early stop spends one; before, it spent one at once. The
pre-pass's other callers, `check_artifact` and design_fast's `focused_check`,
pass no selection and are unchanged.

**Early stop**: the pre-pass stops early — skipping the full check — only
when **all** of the following hold:

1. `result.prep_failure is None` (no install/generate/setup failure; a
   change-caused prep failure has `change_caused_prep` set instead and
   `prep_failure` absent, so it also passes this condition).
2. `check_failure_kind(result) == "code_failure"` (structural classification).
3. Exit code is not 4 or 5 (pytest "no tests collected" / "no tests matched").
4. `parse_test_failures` on the targeted output, minus the inherited set,
   includes at least one of the targeted test IDs.

**Fall-through**: any other outcome runs the full check, as before.  This
includes: all targeted tests passing, pytest exit 4 or 5 (renamed/removed
test), a timeout, no targeted command being possible, a preparation step
failure (install, generate, setup), vitest "No test files found", and a
signal kill.

**Timeout**: the pre-pass never shortens the full run's time limit.  The
full check always receives the original `check_timeout_s`.

**Evidence**: `check-record.json` gains `check_mode` (`"targeted"` or
`"full"`) and `targeted_command` (the targeted command string, or `null`).
The outcome dict passed to `evidence.record()` also includes these fields.

**Convergence exclusion**: targeted runs are excluded from
`check_failures_history` and `check_blocker_history` via
`record_check_history()`, which guards on `check_mode != "targeted"`.
A targeted early stop also prevents `repair()` / `_repair()` from granting
convergence extras via the `early_stop` parameter.

The pre-pass is a second call to `run_check` within the same node.  The
existing check envelope budgets `3 × INSTALL_TIMEOUT_S + 3 × budget`.  The
pre-pass uses at most `min(budget // 3, 120)` plus one install (normally a
cache hit), which the `3 × INSTALL_TIMEOUT_S` slack absorbs.  The total of
pre-pass + full check ≤ `4/3 × budget`, well within the `3 × budget` in the
envelope.  b33b extracts the envelope and will adjust if needed.


## Self-check inside the attempt (b41a)

### Why

Two runs on 2026-10-24 (b39a, b39b) failed round 1 only on mechanical failures,
and each cost a full round: a second emit and a second review fork.

- **b39a**: 5 of 2,841 pytest tests failed: a lowering snapshot not regenerated,
  two tests pinning the old node list, and two whose scripted answers fell out of
  order. The mechanical round cost about $60 and 37 min.
- **b39b**: 14 of 2,835 failed: a new core-to-product import (the boundary test),
  ten "every mode validates" tests (V7: a knob no node consumed), two snapshot
  tests and one test pinning the replaced behaviour. That round cost about $90 and
  49 min.

### What runs, and when

The self-check runs on every applied emission, before the review fork: after
`prepare_attempt` has installed, run `generate` (amending what it writes into the
attempt) and run `setup`. The repository's `docs:generate` now also regenerates the
lowering snapshots (`regenerate_snapshots`), so the run's generate step keeps them
current. It runs in the attempt's prepared worktree through
`repo_change.run_check(prepared=True)`.

- **An attempt's first emission** runs the self-check on the tests the change touches
  (the selection below), before the review fork; the brief's own example is left to the
  review round's full check.
- **A fix-up's emission** runs the selection below in chunks within the self-check's
  time limit (`check_digest.build_probe_command`), in `prepare_check`
  (`design_ops._fixup_self_check`).

### The selection (`test_selection`)

The engine reads the repository's own imports; it names no test or file.

- **The tests the last self-check found failing,** first, as ids.
- **Engine tests, when the emission changed a hub.** A hub is a source
  module at least a quarter of the check's Python test files import (here the
  `nn` package, the registry and the design graph). The engine tests are the
  structural tests, which import no module of the repository (they read it as
  files: the boundary record, the docs), and the tests that import every hub,
  the ones built on the whole engine (lowering smoke, the edit-design tests with
  their mode validation, the design worktree tests).
- **The test files the fix-up changed**: the diff from the commit the last self-check
  ran on, not the whole attempt's, which the first self-check already covered
  (`design_ops._fixup_changes`; the record's `how_selected.fixup_changed`).
- **The last check's failures** that are not inherited (dm-128's set), as ids.
- **The tests that import a module the fix-up changed**, depth 1 (`from package import name`
  resolves to the submodule), and the structural tests when a source changed but no
  hub did: **cheapest first**, so a slow file at the limit does not hide cheap ones
  behind it. A file's cost is the seconds an earlier self-check of the run timed it
  at, else its size, put in seconds at the rate the finished files ran. A chunk run
  by a single pytest command gets `--junitxml` (a file outside the worktree, removed
  after), and a file costs the sum of its tests' seconds there. Without per-test
  seconds only a chunk of one file times that file; a chunk of several times none of
  them. A time is a lower bound when the chunk was stopped at the limit or ran only
  some of the file's tests, by id: a finished time beats it, a higher bound replaces
  it, and it never puts a file's cost below its size estimate. While no file there has
  finished, sizes have no rate in seconds to weigh a bound against, so the files with
  only a bound run after the others.

Only tests the check's pytest steps collect are selected (their path arguments),
and the hubs are counted among them. A TypeScript runner gets only the spec files
the emission touched: the check's own `vitest run` has no related-file selection
(`--related` is an unknown option to it, and `vitest related` is a different
command), so a source change selects no spec, and a Python-only selection never
starts the vitest step. Docs (`.md`, `.rst`, `.txt`) and generated files select
nothing; a docs-only emission runs nothing.

A test file is passed whole to its runner (`build_probe_command(test_files=…)`):
pytest and vitest both take a file path where they take a test id.

### The cut

The selection is cut, in order, into about `SELF_CHECK_CHUNKS` (6) chunks of at most
`SELF_CHECK_CHUNK_FILES` (4) tests, so a chunk the limit stops loses little, each its
own targeted command and each holding one runner's tests: the targeted command
keeps the check's `&&` between steps, so a failing vitest step would otherwise
stop the Python files beside it. The limit is `min(self_check_timeout_s, check_timeout_s // 3)` (300 s
with the lab's 1,800 s check). No chunk starts once the limit is spent; a chunk
still running at the limit is stopped and does not count toward the self-check's
pass/fail decision, because a killed runner names no failure; every finished chunk's
failures do count. A killed runner writes no per-test seconds either, so a stopped
chunk's elapsed time is a lower bound only when it held one file. The files of a
stopped chunk therefore run one per chunk on the run's later self-checks, until a
chunk finishes each (`test_selection.run_alone`): each gets a time or a lower bound
of its own, and a file that has only a bound cannot take other files down with it
when the limit stops it again (dm-247). A chunk that reports
nothing (no failure named, its environment failed) ran nothing that counts, so its
tests are recorded as not run. A module that does not import is that file's
failure (pytest's `ERROR <file>` line); pytest then stops before the chunk's other
tests, so they run again without it. The record says what ran and what was cut,
and how long each chunk took.

### Only the change's failures go back

Design mode has one rule for what is inherited, in the self-check and in the whole
check beside the reviews alike (`base_check.design_comparison`, b48a): a failure the
unchanged base fails too is the base's, whatever file the attempt touched and whatever
design.md says, unless the brief or an approved correction names the test, by its id,
its name or its file, as a whole word or path (`base_check.names_test`: `test_a` is not
named by `test_all_paths`). A test the run exists to fix goes back while it fails. The
fast modes keep the three-condition rule (`base_check.compare_failures`). Every failure
whose file is on the base is looked up there (`base_check.targeted_base_failures`):
the check's cached base result when there is one; otherwise only those tests run in
the base worktree base_check prepares,
after the run's generate and setup there, with the base's own HOME. What a base
run learns about a test is kept (`.refactory/base-targeted.json`), with whether the
base was prepared for that base commit, setup and generate, so a fix-up runs only
new tests there and does not prepare the base again. A failure id the parser cut
(a parametrized test whose parameter holds a space, `test_x[a b]` read as
`test_x[a`) cannot be passed back to pytest, so its whole file runs instead, on the
base and on a fix-up's rerun. A test in a file the attempt added has no base to fail
on. A base that cannot be prepared, runs out of time or names no result is
unavailable, and then nothing is inherited.

### The fix-up, and its bound

The change's failures go back to emit through a reloop, as a refused emission does
(`prepare_check`'s for a fix-up's self-check, `first_self_check`'s for the first), by way
of `pick_corrections`, which every send-back to the builder passes (dm-226):
`validation` carries them to emit on both runtimes; the verdict's reason is
`pick_corrections`' visit feedback, cleared when it decides, and never reaches emit. The
feedback leads with "self-check fix-up N of M", then the failing ids, their digest,
the asks the early asks judge found not built, the brief and the approved corrections
whole (the brief cut at about 6,000 characters, with a note) with what to do about a
test that pins behaviour they change (dm-270: update it, keeping what it guards, and
name it in the completion message as `Updated tests: <test id> (follows <the ask or
correction>)`; otherwise fix the code), the design's `[left: …]` constraint lines
(R5), and the failures that are the base's. Nothing is spent: `attempts_used`,
`check_attempts_used` and `review_attempts_used` do not move, and no review runs.
The next emission is amended into the attempt (`worktree.commit_attempt(amending=True)`,
which stages with the attempt's guards and commits through `worktree.amend`), so an
attempt stays one commit. The reviewers still see what sent the attempt back
before its fix-ups (`review_feedback`).

- At most `self_check_fixups` (2) fix-ups an attempt, and `self_check_fixups ×
  check_attempts` (6) a run (`self_check_fixup_count`, `total_self_check_fixups_used`).
- Past either bound a still-failing self-check is the early stop, which spends
  the attempt (`repair(…, early_stop=True)`); b39b's limits then apply unchanged.
- A refused fix-up emission counts toward b39b's refusal bound like any other. A
  fix-up that takes the attempt back to the commit it builds on is refused too
  ("the emission undoes the whole attempt"), rather than amended into an empty
  commit, which git refuses. When the generate step's amendment is what empties the
  attempt (it regenerates the only files the attempt still changed), it is
  intercepted as ``empty_diff_fixup``: if the fix-up budget allows, the attempt
  loops back to emit; otherwise the build early-stops as exhausted.
- A result that says nothing (a chunk stopped at the limit, a runner it cannot
  target, nothing collected, a chunk failure that names no test) goes on to the
  reviews and the whole check. A first self-check whose run fails without naming a
  test (a build error) is the attempt's failure, not a pass: it goes back with what
  the check printed.

prepare_check's `max_iterations` follows the knobs (`design.prepare_check_iterations`,
applied by `DesignMode.graph` and `DesignMode.state_dict`, so the walker and the
lowered graph get the same number). Its reloops are fix-ups, early stops and
refusals (the first self-check's go through `prepare_check`, below, under the same
bound): applied emissions are at most the attempts (`check_attempt_cap` when it is
above `check_attempts`) plus the run's fix-ups, each after at most
`refusal_attempts − 1` refusals, so the limit is
(attempts + fix-ups + 1) × refusals + fix-ups + attempts + 1: 45 with the defaults
((8 + 6 + 1) × 2 + 6 + 8 + 1), 102 at the knobs' upper bounds. Both runtimes count it
per gate (the walker's `_count_retry`, the spine's `reloopCap`), and b39b's stop
always ends the run first.

### The record

Each self-check appends one line to `self-check-record.jsonl` in the run
directory: the attempt, the commit, the fix-ups before it, the selection and how
it was made, what ran and what was cut, the commands, the limit, the duration,
the outcome (`pass`, `fixup`, `exhausted`, `inherited`, `inconclusive`,
`nothing selected`, `empty_diff_fixup`), the failing ids, the ones inherited and
what the base said; a first self-check's also what the early asks judge said
(`judge`: `judged`, or `failed` with why) and the items it found not built
(`not_built`), and a whole check that failed without naming a test (`unparsed`). Each chunk's timing carries its tests' seconds when it has them,
and `how_selected.alone` names the files run one per chunk.
It also prints one `refactory: self-check …` line, as a gate logs. `refactory
status` shows a fix-up through the lines it already prints: the gate's reloop
reason and `validation`.

### Not-found retry

When pytest exits 4 because of stale test ids in a chunk, the self-check drops
those ids (matched against the chunk's own names by path suffix and bracket
prefix, so ``tests/test_x.py::test_y`` matches ``refactory/tests/test_x.py::test_y``
and a bracket-less ``tests/test_x.py::test_p`` covers
``refactory/tests/test_x.py::test_p[2.5]``) and retries the chunk's remaining
tests.  When the rerun succeeds, its failures are recorded; when no time remains
for the rerun, a lower-bound timing entry is recorded and the chunk is cut.

### The first self-check, beside the early asks judge (b48a)

**Why.** b47a_model_source (2026-10-26) spent three build rounds after approval. Its
first self-check ran 82 files in chunks, found 25 failures and cut 10 files at the
300 s limit; the fix-up then found 5 more in a file the cut had hidden. Its refusal
function was defined but never called for three rounds, its tests passing, until the
QA judge saw it.

**The graph.** `prepare_check` runs the self-check (touched tests in chunks) and
proceeds to `fork_qa`, which runs the full check beside the three reviews:

- **`check`** (op) runs the brief's whole check in a check copy beside the reviews
  (`design_ops.check_beside_reviews`), recording its decision for `gate_check`.
- **`early_asks_check`** (agent, an optional branch, no workspace of its own) reads
  `changes.diff` against the brief's asks and the approved corrections and writes
  `early-asks.json`: for each ask and correction, built or not built, with
  `path:line` evidence for a built one. It judges presence, not quality: the QA
  judge's bar is unchanged. On the spine the worker dispatches it, as it does every
  agent node.

**The self-check's send-back** (after `join_self_check`) sends the attempt's own failures
and every item the judge found not built back as one fix-up, with the self-check's
budget and limits: the not-built items even when every test passes. Past that budget,
failures are the early stop, which spends the attempt, and the not-built items go with
them; not-built items alone go on to the reviews instead, listed in
`qa-synthesized.md`'s `## Asks and corrections not built`, which the QA judge and the
overwatch both read (dm-276: they used to spend the attempts unreviewed and end in a
draft marked check failing though the check passed). A verdict that
is missing, is not one JSON object, lists no ask (or no correction when the person
gave some), or calls an item built without a `path:line` is unreadable, never "all
built" (`design_ops.parse_early_asks`): the judge is asked again (`reread` to the
fork, which runs again; `first_self_check` keeps its result, so the check does not run
twice) as often as `judge_rereads` allows (1), then the attempt goes on without a
verdict, with `judge.status: failed` in the record and a run warning. A failed judge
branch is recorded missing with its error (`reviews/early_asks_check-missing.txt`,
written by the worker on the spine and by the fork on the walker) and read the same
way.

**Updated tests (dm-270).** The builder names each test it updated, and the change it
follows, on one line of its completion message (`Updated tests: …`).
`design_materialise` keeps the attempt's lines (`updated-tests.json`, the
`updated_tests` fact); the code reviewer and the QA director read them as they start,
and the QA judge reads them, beside the test files that failed in a self-check and
changed since, in `qa-synthesized.md`'s `## Updated tests`. A test weakened to pass is
a must-fix unless the update follows an ask or correction. Markdown around the line
(`**Updated tests:**`, a code span) is not part of what it names.

### Tests

`refactory/tests/test_first_self_check.py`: the graph on both runtimes, the judge's
verdict (unreadable, never all built), every failure in round 1 with a slow file a
chunked run cuts, the reuse and each field of its key, no reuse after a fix-up, a whole
check that names no test, the one inheritance rule through both checks (a touched test
file, a test named only in design.md, one the brief or a correction names, a name that
only starts a word), b47a's first two self-checks replayed from its records, the judge
on the walker and through the worker on the spine (not built with failing tests and
with passing ones, asked again, failing, never ending the run), the message, the
updated tests, and a fix-up's selection.
`refactory/tests/test_self_check.py`: the loop through the walker's design graph
(a stale generated file and a pinned test fixed inside the attempt; a clean
emission; the bound and the early stop; the worst case with two refusals before
every applied emission ending in the build-loop stop, at the defaults and at the
knobs' bounds; a fix-up that undoes the attempt; a fix-up generate empties; the
whole check still binding after a cut self-check), the base comparison (a new
failure goes back and the base's does not; a base out of time, a base run naming
nothing and a base whose generate fails let nothing be inherited; the base's own
HOME; knowledge from another base not reused; the base prepared once), the cut
with real chunks (a slow file at the limit, a module that does not import), an
inconclusive self-check, a docs-only emission, the spine across node processes,
the run cap, the amendment path, design_fast and `check_artifact` unchanged, the
selection's order, and b39a's and b39b's attempt-1 changes as fixtures.


## Stuck detection (dm-128)

When the same non-inherited test fails with the same assertion on two
consecutive checks of either kind (targeted or full), compared on the tests
both checks ran, the builder's feedback includes a stuck notice naming the
test and showing its last change (`git diff HEAD^..HEAD`) from the attempt's
worktree.

"Same assertion" means the same `>` source line from the assertion block,
even when the values on that line differ.  When no `>` line exists, the full
assertion string is compared.

The diff is retrieved from the worktree (`worktree.recorded(ctx.workdir).path`),
never from the run directory.  If the per-file diff is empty, the whole
`HEAD^..HEAD` diff is shown, bounded to 200 lines.  The exact `path:NN:`
lookup for per-file targeting is documented as a known limitation — the
whole `HEAD^..HEAD` fallback is the documented behaviour (dm-166).

Guards:

- No previous failure (status ≠ "fail"): no notice.
- Same commit (retry/resume): no notice.

After a refused emission, the next check runs in full because the refused
record has `status: "refused"` (not `"fail"`).  Preserving the prior failing
record across a refused emission is tracked as dm-166.


## pnpm test targeting (dm-128)

`build_probe_command` now targets `pnpm test` commands when
`package.json`'s `test` script contains `vitest`.  The command is built as
`{step} {files} {pattern_args}` with:

- **No `--` separator** — with `--`, vitest ignores the file filters.
- **Single `-t` with `|`-joined patterns** — vitest supports regex in `-t`,
  so multiple test names are joined with `|`.

If the `test` script is not vitest (e.g. jest), the function returns a
refusal and the pre-pass falls through to a full check.

The `root` parameter (defaulting to the process's cwd) specifies where to
read `package.json`, so the function works correctly on DSH where the
process's cwd is the run directory, not the worktree.

The existing vitest branch also uses a single `-t` with `|`-joined patterns
instead of multiple `-t` flags (vitest exits 1 when given several `-t` options).


## Change-caused preparation failures (dm-104)

When a generate, setup or install command fails while preparing the attempt's
worktree, the failure is compared against the unchanged base before it is
reported.  A failure the base does not have — or one that names files the
attempt changed where the base's does not — is the change's failure: the
exact error goes to the builder for repair, labelled with the step (generate,
setup or install), and spends an attempt like any failed check.

The base comparison follows the same isolation rules as the existing test
comparison (`run_base_check`):

- The base's prep runs in its own HOME (`BASE_CHECK_HOME`), not the
  attempt's `CHECK_HOME`, so a setup that writes into HOME does not share
  state with the attempt.
- A crashed or timed-out base prep is discarded and retried once (R2).
  A double crash returns the failure to today's handling (infrastructure).
- The base's prep outcome is cached per run, keyed by `cache_key(base,
  setup, generate, "")`.  Crashes and timeouts are never cached.

Infrastructure faults — a signal kill (exit < 0 or shell-wrapped 134/135/139),
a timeout, a missing executable (exit 126/127), or a resource exit
(`_RESOURCE_EXIT_CODES`: 137, 152, 153) — bypass the base comparison entirely
and keep today's handling.

The base comparison is never run from the targeted pre-pass (#463), because
the pre-pass timeout is too short for a full base preparation.

When a change-caused preparation failure is detected:

1. The error text is stripped of ANSI codes and noise via `_clean` from
   `check_digest.py`.
2. The failure is recorded in convergence history via `record_check_history`
   with `check_mode="full"`, so R6 does not decide on stale history.
3. In design mode, the failure goes to `repair()` directly, skipping
   `filter_check`, the unbuilt diagnosis, the digest, stuck detection and
   design-contract augmentation.
4. In fast mode, the failure goes to `focused-check.json` with
   `prep_triage=True` and the formatted error as `digest`.
   `gate_verification` routes it to `_repair()` with the same text as
   design mode, skipping the "requested runtime check failed" wrapping.
5. A failed preparation is never reported as "unbuilt packages" because
   the early return fires before the unbuilt diagnosis.

Cross-references: #432 (infrastructure failure handling), #453 (convergence
and noise stripping), #463 (targeted pre-pass).

### Behaviour changes

All changes (fail-fast pre-pass, stuck detection, pnpm test targeting,
change-caused preparation failures) apply to both runtimes — the TS spine
runs the same Python functions as subprocesses.  The change-caused prep
triage runs identically under `run.run_node` and the walker because the
callables (`check_artifact` and `focused_check`) are the same Python
functions.

**Scope**: design and design_fast only.  create_fast / edit_fast /
fix_pack's check gate and package check keep today's behaviour.  They become
a follow-up, filed separately.

**New optional fields in `check-record.json`**:

- `check_mode` (`"targeted"` or `"full"`)
- `targeted_command` (string or null)
- `inherited` (sorted list of inherited test IDs, on failure)
- `commit` (the current commit hash)

No new CLI flags or environment variables.

### Steer correction ordering (dm-178)

`console._steer` now writes verdicts before the note, aligning with the interactive
bridge's `answer` action which already writes verdict-first.  A correction or
constraint given in the same `refactory steer` call as a `gate_strategy=proceed`
verdict is collected by `pick_gate_corrections` and applied to `design.md` before the
build, on both runtimes.

**Residual risk, accepted:** on a live DSH run, a 500 ms spine poll landing between
`_steer`'s two writes (verdict then note) could insert a `verdict-use` before the
note, causing `pick_gate_corrections` to stop collecting before it sees the note.
The window is microseconds — `_steer` writes both entries in the same Python call
without yielding — and has not been observed in practice.

### Behaviour changes

`_steer` writes verdicts before the note (both runtimes).  `pick_gate_corrections`
uses `steering.parse_entries` and collects every correction/constraint with product
scope and user-relayed source between the proceed verdict and the next gate-visit or
verdict-use.  `apply_corrections` no longer declares
`produces=("design_corrections_applied",)`, so the count fact set by
`pick_gate_corrections` survives in `facts.json`.


## Base worktree integrity (dm-162)

### The problem

`compute_test_guardedness` copies the attempt's changed test files and helpers
into the shared base worktree, runs them, then unlinks every copied path.
For a file that already existed in the base commit (a *changed* test, not a
*new* one), the unlink deletes the base's own tracked file instead of
restoring it.  Because `_ensure_base_worktree` only checked HEAD commit
identity — not file cleanliness — subsequent callers (`run_base_check`,
`compare_prep_failure`) ran on the corrupted tree.  The corrupted result could
be cached in `base-check.json` or `base-prep.json` and replayed on resume.

Twice in 11 hours (b33a\_r1 and b34b), this turned a repairable check failure
into a run-ending environment failure that had to be fixed by hand.

### Three layers of defence

#### Layer 1: restore-in-finally in `compute_test_guardedness`

Before copying, `git ls-tree <base> -- <paths>` (with `--literal-pathspecs`)
determines which destination paths exist in the base commit.  The entire
copy → test → cleanup flow is wrapped in `try / finally`.  In the finally
block:

- pre-existing tracked files are restored via `git checkout <base> -- <paths>`
  (never HEAD of another tree);
- pre-existing ignored files have their bytes saved before copying and written
  back;
- only files that were *new* (not in the base commit) are unlinked.

The `finally` fires for `BudgetExhausted`, `KeyboardInterrupt`, sandbox
timeouts (`timed_out=True`) and any other exception.  It does not fire for
SIGKILL or a DSH node timeout that kills the process — only Layer 2 covers
those cases.

#### Layer 2: dirty-base detection and repair in `_ensure_base_worktree`

A new helper `_check_base_clean` runs at the single choke point all three
callers share — inside `_ensure_base_worktree`, after the worktree exists and
HEAD is verified, before `_isolate_hooks` and `ensure_environment`.

Algorithm (following `worktree._restore`):

1. `git --no-optional-locks status --porcelain=v1 -z --untracked-files=normal`
2. If dirty:
   - `git reset -q --hard <base_commit>` with `NO_HOOKS`
   - `git clean -fdq -e /<name>` for each entry in the base install record's
     `installed` list (names with trailing `/` stripped)
3. Re-check that the status is empty.  On failure, rebuild the base worktree
   (remove and re-add) — never let a failed reset become a cached environment
   failure.
4. Warn only about restored tracked files via `target_guard.warn` (the
   `run warning:` prefix the spine promotes to run warnings).

Installs (node\_modules, .venv, build outputs) survive because they are either
gitignored (invisible to `--untracked-files=normal`) or excluded by the `-e`
arguments from the install record.

#### Layer 3: `base_clean` cache guard

Both `base-check.json` and `base-prep.json` carry a `base_clean` field:

- `_read_cache` and `_read_prep_cache` reject entries where
  `data.get("base_clean") is not True` — dirty-base entries and old-format
  entries without the field are treated as misses (safe re-run).
- `_write_cache` and `_write_prep_cache` skip writing when `base_clean=False`
  (warn via `target_guard.warn` and return).
- Crashed results remain never-cached (the `_crashed` guard takes precedence).

The `base_clean` value comes from the status taken right after the reset in
`_check_base_clean` (recorded when `_ensure_base_worktree` ran), not a
post-check re-verification.  This avoids three problems: a `Path` that makes
`json.dumps` raise `TypeError`, attachment only on the success path (missing
the environment-failure paths), and `__pycache__` making a project without
`.gitignore` always dirty after its own check.

### Why repair-and-warn is correct here

The contributing guide says "the validator refuses; it never repairs"
(`contributing.md:94`), but that rule applies to *graph state* validation.
The base worktree is *environment state* — a cache reconstructable from the
base commit.  Repairing it and warning is the same pattern
`worktree._restore` uses for the run's own worktree after a check.

### Behaviour changes

All changes apply to both runtimes — the walker (`Mode.run`) and the DSH
runner (`run.main`) — because the logic is in `base_check.py` which both
runtimes call.

1. **Restore after guardedness (Layer 1):** `compute_test_guardedness` now
   restores pre-existing files from the base commit and removes new files in
   a `finally` block, instead of unlinking all copied files.
2. **Dirty-base reset (Layer 2):** `_ensure_base_worktree` now checks and
   repairs dirty base worktrees using `git reset --hard` + `git clean -fdq`
   with install-record exclusions, before any caller uses the base.
3. **Reset failure rebuilds (C2):** A failed reset in `_check_base_clean`
   rebuilds the base worktree from scratch instead of becoming a cached
   environment failure.
4. **Warning channel (C3):** Dirty-base warnings are emitted via
   `target_guard.warn` (not bare `print`), so they appear as `run warning:`
   lines the spine promotes.  `base_reset` is recorded in
   `check-record.json`.
5. **Cache guard (Layer 3):** `_read_cache` and `_read_prep_cache` reject
   entries without `base_clean is True`.  `_write_cache` and
   `_write_prep_cache` skip writing when `base_clean=False`.
6. **Old cache entries re-run:** Existing cache files without `base_clean`
   become misses (safe re-run).

## The check beside the reviews

**Why.** Design mode ran its check, then its reviews. On the 2026-10-22 measurements
the check took about 12 minutes on every run, and the reviews 2.5 (the happy canary)
to 10 minutes (dm-149), so the reviews' whole time was added to every round.

**How it works now.** Three nodes replace the one `check` gate:

1. **`prepare_check`** (gate, before the review fork) settles the commit the check and
   the reviewers both see: `repo_change.prepare_attempt` installs the environment, runs
   the `generate` command (its tracked-file changes amended into the attempt) and the
   `setup` build in the run's worktree. Nothing moves the commit after this. Then the
   self-check ("Self-check inside the attempt"): every attempt runs it on the tests the
   change touches, before the review fork, and an attempt that fails it goes back to the
   builder without a review. It does not run the brief's own example (the canary's marker
   grep, the wordcount check): a broken example is caught by the review round's full check,
   and costs that round. A preparation that fails
   is judged right there by the whole sequential check (`check_artifact`), which reports
   it with its triage as before. It first clears what a last attempt prepared or
   decided, records the attempt in `check-prep.json`, and sets `review_feedback` (what
   sent the build back) and `review_baseline` (the requirements) for the reviewers.
2. **`check`** (op, a branch of `fork_qa`, beside the three reviews and the verifier)
   runs the brief's check on that commit in the run's **check copy**
   (`run_check(beside_reviews=True)`): a detached worktree of its own at the attempt's
   commit, beside the run's worktree and its base worktree, its environment prepared
   there and its hooks and HOME its own (`.refactory/check-copy-home`), so a setup that
   keeps what it wrote into HOME points at the copy. The reviewers read and run the delivered files in the
   run's worktree meanwhile; neither sees, or deletes, what the other leaves. The copy is
   kept across attempts (moved to the new commit, the last check's files put back, its
   installed environment kept) and removed with the base worktree. The op records its
   decision, with the check command it ran, in `check-verdict.json`; a node inside a
   fork may not send the run outside it. A check that raises (an unrepairable failure)
   is recorded and raised by `gate_check`, because the node runner retries a failed op
   and a retry would be a second full check.
3. **`gate_check`** (gate, after `join_qa`) routes on that decision: a failed check goes
   back to emit with its feedback, as before, and that attempt's reviews are not read. A
   person's skip of the check proceeds on its skip receipt for this attempt. With no
   decision for this attempt and command -- the check command was corrected after it
   ran (`graph-control amend-command` reruns this gate), or the check never recorded
   one -- the gate runs the whole check itself, bound to the same attempt, so the
   reviews beside it stay current. A person may skip `gate_check` itself, as they could
   the old gate: it produces `runtime_evidence`, so its skip leaves a skip receipt.

**What the reviewers see.** The check's receipt does not exist when they start, so the
three reviews do not read `authoring-evidence.json` or the facts the check writes. Their
context is `BESIDE_CHECK_REVIEW_CONTEXT`: the check runs beside them and they must not
rerun it; a test that also fails on the unchanged base is not the change's defect (the
check compares its failures with the base); they write scratch files outside the run's
worktree; plus `review_baseline` and `review_feedback`. The verifier no longer reads
`test-guardedness.json`, which the check writes, and is told the check runs beside it.
Overwatch runs after the check and still audits its receipt.

**What a sign-off is bound to.** A reviewer that starts before the check records is
bound to the commit, intent and inputs it saw and to that check's attempt
(`delivery_identity` returns `identity` and `check_attempt`). The evidence names the
attempt it ran on (`validation.check_attempt`), and `current_problems` accepts such a
sign-off only when both match; any other check, commit or intent leaves it stale. A
reviewer that starts after the check recorded is bound to the receipt, as before. A
person's skip receipt carries the attempt when `check-prep.json` is for the commit the
run is at.

**Facts on the spine.** The spine runs a fork's branches at once, and each node read
the facts when it started and rewrote all of `facts.json` when it finished, so a check
finishing after its reviews would have put back the facts they wrote meanwhile. A node
now writes back only the facts it changed (`run.update_facts`), as does an agent's
completion (`worker_node`) and a data pass (`items._write_facts`): all through
`run.edit_facts`, under `.refactory/facts.lock`, with the file replaced whole. A facts
file that cannot be read is refused rather than overwritten. The walker runs up to five
fork branches at once (`nn/fork.MAX_WORKERS`), the design review fork's width, and a
serial fork no longer stops at a failed optional branch: as on the concurrent path, it
is missing and the branches after it run.

## Design scale classification (dm-200)

### Evidence

39 of the last 40 design runs on the lab were classed scoped because their briefs
named a file.  Run b37a_disputed was a structural change across six graphs, classed
scoped because its brief named files; it needed six corrections at the gate and
failed to converge at ~$765.

### The rule

`classify_scale` (`design_ops.py`) runs after `gate_research` proceeds.  It reads
every direction's `footprint` object from `research-plan.json` and applies a
three-part rule: the change is scoped only when it touches at most two existing
functions, adds nothing new, and changes no shared engine.  Anything else is full.

The classifier fails toward full, the path that loses nothing:
- a direction that is not an object, or has no footprint, makes the run full;
- a footprint must have exactly `existing_functions`, `new_items` and
  `shared_engine`, each a list; a missing, misnamed or extra key is full;
- each touched function is one `"path:line name"` entry with a real line number,
  and functions are counted by file, name and line, so same-named methods in one
  file count separately;
- a cited path must name a file in the target's checkout (the run's worktree,
  never the run directory). Research may cite it from the repository root, from a
  package, or by a bare file name, and it counts when exactly one tracked file
  ends with it. An absolute path or one with `..` never counts.

The brief is never read by the classifier.

### The reason fact

`research_scope_reason` records why the run was classed as it was:
- scoped: the functions it touches, and that it adds nothing and changes no engine;
- full: the first thing that made it full, or that the research recorded no usable
  footprint;
- refine: that it is a refinement.

The reason appears on `gate_strategy`'s card, in `design_review_status`'s output,
and in both of the CLI's prints when a run stops at a user gate.

### Tests that hold it

- `test_design_scale.py::test_classify_scale_scoped_footprint`
- `test_design_scale.py::test_classify_scale_no_footprint_means_full`
- `test_design_scale.py::test_classify_scale_too_many_functions`
- `test_design_scale.py::test_classify_scale_adds_new_items`
- `test_design_scale.py::test_classify_scale_shared_engine`
- `test_design_scale.py::test_classify_scale_refine_stays_scoped`
- `test_design_scale.py::test_classify_scale_brief_cannot_classify_itself`
- `test_design_scale.py::test_doc_only_research_with_empty_footprints_is_scoped`
- `test_design_scale.py::test_footprint_entry_without_evidence_is_full`
- `test_design_scale.py::test_structural_brief_naming_files_is_full`
- `test_design_scale.py::test_one_line_doc_fix_stays_scoped`
- `test_design_scale.py::test_refine_run_stays_scoped`
- `test_design_scale.py::test_classify_scale_reason_in_design_review_status`
- `test_design_scale.py::test_classify_scale_through_run`
- `test_design_scale.py::test_same_named_functions_in_different_files_count_separately`
- `test_design_scale.py::test_same_named_functions_in_one_file_count_separately`
- `test_design_scale.py::test_a_footprint_needs_exactly_its_three_keys`
- `test_design_scale.py::test_a_function_citation_needs_a_line_number_and_one_name`
- `test_design_scale.py::test_a_citation_outside_the_checkout_never_counts`
- `test_design_scale.py::test_citations_resolve_against_the_real_checkout_however_research_writes_them`
- `test_design_scale.py::test_two_functions_stay_scoped_and_three_do_not`
- `test_design_scale.py::test_a_rerun_reclassifies_a_scoped_run_from_its_footprint`
- `test_design_scale.py::test_a_plan_in_prose_and_fences_is_read`
- `test_design_scale.py::test_a_direction_that_is_not_an_object_is_full`
- `test_design_scale.py::test_a_shared_engine_that_is_not_a_list_is_full`
- `test_design_scale.py::test_research_director_footprint_instruction_clauses`
- `test_design_short_path.py::test_the_design_graph_routes_on_the_scale`
- `test_interactive.py::test_prepare_design_sets_full_even_when_brief_names_a_file`

### Light refine

A design-mode refine (`--refine <PR>`) follows up an open pull request.  By
default (`refine_design=light`), the `route_refine_design` gate skips the
research, pitfalls and strategy phases and goes from `prepare` straight to
`design_doc`.  The gate writes sentinel facts and files — research, pitfalls,
strategy_plan, strategy — so every downstream node finds its declared inputs on
disk on the spine, and gives `design_doc` the pull request as `refine_context`:
its description, framed as other people's words and capped at 6,000
characters, and its diff stat and changed files from where its head left its
base branch, with the base read from GitHub by URL (never from a
remote-tracking branch, nor from the run's worktree, whose base is the pull
request's head).  When the base cannot be read the design is told the diff is
unavailable, never that it is empty, and a warning says why.  With
`refine_design=full`, or on any non-refine run, the gate returns `proceed` and
the existing pipeline runs unchanged.

Prepare reads the refine from the task or, failing that, the `refine` knob,
as `delivery.refine_task` does.  On the spine `refine` is a knob and a node's
task names only the node, so the task's `refine` key, which the first version
read alone, never fired there: every refine walked the full phase, and
`classify_scale`'s refine shortcut never ran.

The `refine_design` knob is frozen: it cannot change after the run starts.
The default is `light`.  The CLI flag is `--refine-design light|full`, which
is refused without `--refine`.

If the person reloops at `gate_strategy`, the existing reloop edge goes to
`synthesize_strategy`, which gets the pull request block as its
`strategy_plan` along with the person's note.  The run reaches `design_doc`,
which still gets `refine_context`, and the gate again, without running
research, pitfalls or `strategy_director`.

#### Tests that hold it

- `test_design_refine_spine.py::test_the_spine_takes_the_light_route_when_refine_is_a_knob`
- `test_design_refine_spine.py::test_a_full_refine_on_the_spine_is_classified_as_a_refine`
- `test_design_refine.py::test_a_light_refine_designs_from_the_pull_request_and_delivers_onto_it`
- `test_design_refine.py::test_a_light_refine_says_the_diff_is_unavailable_when_the_base_cannot_be_read`
- `test_design_refine.py::test_a_light_refine_frames_the_pull_requests_description_and_caps_it`
- `test_design_refine.py::test_a_light_refine_relooped_at_its_gate_keeps_the_pull_request_in_the_next_design`
- `test_design_refine.py::test_a_full_refine_walks_the_design_phase_with_no_pull_request_block`
- `test_design_refine.py::test_non_refine_unchanged`
- `test_delivery.py::test_the_command_line_hands_refine_design_to_the_run`
- `test_delivery.py::test_the_design_command_applies_refine_design_and_refuses_it_without_refine`
- `graph-set.spec.ts`: `designGraphRefineLight`, and `designGraphRefineLightRelooped` for the
  person's reloop

**Tests.** `test_check_beside_reviews.py`: the graph and the lowered workflow; what the
reviews read and see; scripted runs whose check passes, and whose first check fails
(two emits, the second carrying the failure, both attempts reviewed); `gate_check`'s
routes, including a person's skip; the in-fork check refusing an attempt it was not
prepared for and not running twice; a corrected check command judged again; the check
running in its own copy, which follows the attempt; `prepare_check` clearing the last
attempt's records, sending back before the reviews, and judging a failed preparation at
once; sign-off binding; the facts lock, an unreadable facts file refused, and a node
writing back only the facts it changed; and a serial fork running past a failed optional
branch. `spine.spec.ts` drives the fixture through the new fork and both `gate_check`
routes.

## 29. Marker placement rule (dm-198)

**What the issue was.** On 2026-10-23, two dm-149 measurement runs
(measure3-dm149_20261023_0351 and measure4-dm149_20261023_1337) failed at
`gate_qa` with "review reloop limit reached after 3 review cycle(s)". In
both, a reviewer wrote the blocking marker while explaining a parser check,
and `create_ops.blocking_findings` read it as a core finding that came back
every round.

- In measure3, the code reviewer wrote a line opening with the inline code
  `BLOCKING:` followed by explanatory text about the parser.
- In measure4, the health checker logged its parser checks as list items
  containing the marker form.

No reviewer prompt said where the marker may appear, and the prompts
themselves showed the finding form inside a code span, teaching reviewers
the quoted form.

**The rule.** `create_ops.MARKER_RULE` tells every reviewer to write the
blocking marker only at the start of a finding line, in its plain form,
to call it "the blocking marker" when discussing it, to write every real
finding as a plain line outside any fenced code block, and to put an
example of the format inside one, because fenced blocks are not read as
findings. It is a prompt rule, so
it makes a misread less likely rather than impossible. Every site that
showed the finding format in a code span now shows it as plain text. The
parser (`blocking_findings`) is unchanged: a real finding at the start of a
line, outside a fenced block, a blockquote, or a section marked as a
previous or superseded review, still blocks.

**Held by.** `test_follow_ups.py::test_every_reviewer_carries_marker_rule`,
`test_follow_ups.py::test_every_lowered_node_that_asks_for_findings_carries_the_rule`,
`test_follow_ups.py::test_marker_rule_keeps_its_wording`,
`test_follow_ups.py::test_verifier_prompt_carries_marker_rule`,
`test_follow_ups.py::test_no_reviewer_prompt_shows_blocking_in_code_span`,
`test_follow_ups.py::test_verifier_prompt_no_code_span_blocking`,
`test_follow_ups.py::test_no_lowered_prompt_uses_the_marker_in_prose`,
`test_gate_stale_findings.py::test_fenced_code_blocking_is_skipped`,
`test_gate_stale_findings.py::test_reviewer_prompts_ask_for_structured_blocking_format`.

## 30. Draft delivery (dm-224)

### What it is

A draft delivery opens a pull request with `gh pr create --draft` when the
`draft_delivery` fact is set on the run.  The fact carries:

- `reason` (string): the limit or stop that ended the build (e.g. "review
  reloop limit reached after 3 review cycle(s)").
- `unresolved` (list of dicts): each item has `text`, `source` (the judge or
  gate that named it, or `"delivery"`), and `rounds` (the attempts that named
  it, possibly empty).
- `check_failing` (bool) and `check_failure` (string), when the check fails on
  the draft's head.

A node that sets `draft_delivery` must declare it in its `produces` (the
spine refuses an undeclared fact) and write a mapping in the shape above; a
fact that is not a mapping is read as no draft, with a run warning
(`delivery.draft_fact`), so report and deliver behave as without it.

### Who sets it (b45a)

Design mode's limit stops, in its own gate functions (`design_ops`), and
nowhere create, edit or the fast modes run: `create_ops.repair` and
`repo_change._count_refusal` still raise `BuildLoopStop` there. When the run
has an applied commit whose check evidence is current, `_draft_for_stop` sets
the fact and the stop takes its own way to the report:

| Stop | Where | Then |
|---|---|---|
| The check fails at its limit, or at the overall attempts limit | `prepare_check`, `gate_check` | `draft`: straight to the report, marked check failing; the QA judge and the overwatch judge only a change whose check passed |
| The review limit, or the overall attempts limit, at the QA judge | `gate_qa` | `proceed`: on through the correction audit and the overwatch |
| The same at the overwatch | `gate_overwatch` | `proceed`: to the report; after the QA judge's stop its items join that draft and no attempt is spent |
| The refusal bound | `design_materialise`, then `prepare_check` | `draft`: to the report, marked check failing when that commit's check failed |

The unresolved items are what each judge left unresolved as of its latest
verdict (`qa_unresolved`, kept by `gate_qa`, and `overwatch_unresolved`, kept by
`gate_overwatch`). gate_qa keeps its stop in `qa_draft`, which only it produces, and
gate_overwatch folds it into `draft_delivery`, so a redo of the overwatch keeps the
QA judge's draft. A
refused emission writes its reason to `check.txt`, and a failing check adds its
triage there after its evidence is recorded; `authoring_evidence.restore_receipts`
puts the recorded `check.txt` back from the evidence's validation history
before the draft is decided. A stop with an applied commit and current check
evidence ends in a draft pull request, and the run ends done; any other stop still
ends the run with no pull request (a stop whose latest commit has no current
evidence is dm-272).

### Why

Four runs between 2026-10-23 and 2026-10-24 passed their check but hit a
build-loop limit and ended with nothing delivered:

- **measure3-dm149_20261023_0351** and **measure4-dm149_20261023_1337**: each
  stopped at gate_qa with "review reloop limit reached after 3 review
  cycle(s)", with its latest check record a pass, and opened no pull request.
- **b37b_marker_rule** (dm-223): attempt 3 passed the full check, then stopped
  at gate_qa's review limit. Delivered by hand as #495.
- **b39b_progress** (2026-10-24): attempt 3 passed the full check, gate_qa
  found no core defect, overwatch's verdict was PASS.  Overwatch's correction
  audit then sent it to repair, which reached the overall attempts limit.
  Delivered by hand as #496.

### What report and deliver do on a draft run

- `report_artifact` calls `require_current_evidence` (the evidence-currency
  checks factored out of `current_problems` via `_evidence_currency_problems`,
  and its skip checks via `_skip_problems`) instead of `require_completion`.
  It requires current evidence for the pushed commit but not the review
  sign-off or a passing check.  A check skipped by anyone but a person is
  refused, as on the complete path, and so is a run with no check evidence at
  all, whether or not the `runtime_evidence` fact is set.  (Without the fact,
  `require_completion` is unchanged, including its acceptance of a legacy run
  with neither evidence nor that fact.)
- `deliver_design` opens the PR with `--draft` and puts the draft's section
  first in its description, between `<!-- delivery:draft -->` and
  `<!-- /delivery:draft -->`: "## Check failing" and what the check said when
  it fails (the title then starts `[check failing] `), then "## Unresolved
  review findings", listing the reason and each unresolved item.  Every value
  in the section goes through `_public` (credential values hidden) on one
  line, as every other field of the description.  Delivering again replaces
  only what is between the markers (dm-246), so a person's text, prose with
  no heading included, stays.
- A later delivery with no draft fact marks a draft that delivery made ready
  for review (`gh pr ready`), takes the section out, and takes `[draft] ` and
  `[check failing] ` off the title. A person's own draft is left a draft.
- Delivery refuses a branch whose head has the same files as its base: a run
  whose whole change is empty opens no pull request.
- An open PR for the branch is converted with `gh pr ready --undo`, after the
  push, so a refusal from there on keeps the push and the PR in the record.
  GitHub has no draft PRs for private repositories on some plans: where `gh pr
  create --draft` is refused, the PR opens ready for review with its title
  prefixed `[draft] `; where `gh pr ready --undo` is refused, the section is
  still written.  Either way the record's `draft_note` and the delivery result
  say so.
- On a base conflict or failing merged-result check, delivery drops the merge
  (back to the run's own commit), records it as the first unresolved item, and
  continues to push and open the draft.
- Every other `NotDelivered` (closed PR, push refused, abandoned merge) keeps
  today's behaviour, and is neither recorded nor reported as a draft: nothing
  was delivered.  A merged PR is recorded merged, as today.
- `evaluate_delivery_conflicts` returns `"proceed"` immediately (never
  reloops) on a draft run.
- The run ends done, not failed.

### Tests

- `refactory/tests/test_delivery.py::test_draft_passing_check_opens_draft_pr`
- `refactory/tests/test_delivery.py::test_draft_failing_check_opens_draft_pr`
- `refactory/tests/test_delivery.py::test_draft_no_evidence_refused`
- `refactory/tests/test_delivery.py::test_no_draft_fact_requires_signoff`
- `refactory/tests/test_delivery.py::test_draft_conflict_with_base`
- `refactory/tests/test_delivery.py::test_draft_failing_merged_check`
- `refactory/tests/test_delivery.py::test_draft_existing_pr_converted_to_draft`
- `refactory/tests/test_delivery.py::test_draft_closed_pr_opens_no_second_pr`
- `refactory/tests/test_delivery.py::test_draft_conflict_existing_open_pr`
- `refactory/tests/test_delivery.py::test_draft_no_pr_mode`
- `refactory/tests/test_delivery.py::test_spine_draft_delivery_across_processes`
- `refactory/tests/test_delivery.py::test_draft_pr_body_hides_credentials_and_keeps_each_item_on_one_line`
- `refactory/tests/test_delivery.py::test_draft_without_any_check_evidence_is_refused`
- `refactory/tests/test_delivery.py::test_draft_with_stale_identity_is_refused`
- `refactory/tests/test_delivery.py::test_a_draft_github_refuses_opens_ready_with_its_title_marked`
- `refactory/tests/test_delivery.py::test_a_pull_request_github_will_not_make_a_draft_keeps_its_delivery`
- `refactory/tests/test_delivery.py::test_a_draft_whose_description_gh_cannot_edit_still_records_its_push`
- `refactory/tests/test_delivery.py::test_delivering_a_draft_again_replaces_its_unresolved_section`
- `refactory/tests/test_delivery.py::test_a_draft_refinement_turns_the_refined_pull_request_into_a_draft`
- `refactory/tests/test_delivery.py::test_a_draft_whose_pull_request_merged_is_recorded_merged`
- `refactory/tests/test_delivery.py::test_a_draft_whose_merge_was_abandoned_is_not_succeeded`
- `refactory/tests/test_delivery.py::test_a_draft_whose_push_is_refused_keeps_the_work_local_as_without_the_fact`
- b45a, through the real design graph: `test_delivery.py::test_a_review_limit_at_gate_qa_with_a_passing_attempt_delivers_a_draft`,
  `::test_a_check_that_still_fails_at_its_limit_delivers_a_draft_marked_check_failing`,
  `::test_a_run_with_no_applied_commit_delivers_nothing`,
  `::test_the_refusal_bound_after_an_applied_commit_delivers_that_commit_as_a_draft`,
  `::test_a_review_limit_at_gate_overwatch_delivers_a_draft`,
  `::test_a_persons_description_survives_two_draft_deliveries`,
  `::test_a_full_delivery_after_a_draft_makes_it_ready_with_no_section`,
  `::test_a_run_whose_whole_change_is_empty_opens_no_pull_request`; through the node
  runner: `test_build_loop_stop.py::test_a_design_check_limit_goes_to_the_report_as_a_draft_on_the_spine`,
  `::test_the_refusal_bound_after_an_applied_commit_goes_to_the_report_on_the_spine`,
  `::test_a_review_limit_at_gate_qa_goes_on_as_a_draft_on_the_spine`,
  `::test_a_review_limit_at_gate_overwatch_goes_to_the_report_as_a_draft_on_the_spine`.
- `refactory/tests/test_delivery.py::test_a_draft_fact_that_is_not_a_mapping_is_no_draft`
- `refactory/tests/test_authoring_evidence.py::test_ceo_skip_is_refused_on_the_draft_path`
- `refactory/tests/test_authoring_evidence.py::test_person_skip_is_accepted_on_the_draft_path`

## 31. One design pass

### Evidence

On full runs b39a, b39b, b40a and b41a (2026-10-24/25) the design phase took 81–87
minutes, against a target of about 35–40 (and under 10–15 for a small change).

- `design_doc` was written three or four times, about 32–35 minutes. `check_design_claims`
  sent it back for a line that had drifted ("quote not found near line 940 … found at
  line 944"), for a quote with "…" in it, and for an ambiguous bare file name
  ("run.py is ambiguous"): 13–33 minutes per run.
- The design-review agent took 9–23 minutes per round, and the `design_doc` rewrite its
  BLOCKING findings triggered made it 38–54 minutes per full run. It still missed what
  the person at the gate had to correct.

### The rule

Design mode's design phase now runs like remote-factory's: research, strategy and
design go straight to the person's gate, and that gate is the design check.

- **No design reviewer.** The `design_review` agent node and the `route_design_review`
  gate are removed, not renamed: `design_doc → check_design_claims → gate_design_review →
  gate_strategy` at every scale. A reviewer that cost 9–23 minutes a round and still
  missed what the person corrected was not earning its place before a person who reads
  the design anyway.
- **A light template.** `design_doc` must write `## Summary`, `## Changes` (one bullet
  per file and what changes) and `## Tests` (repo-relative ids and what each proves);
  `apply_corrections` adds `## Corrections at approval`. Every other section is optional,
  and every consumer of one treats its absence as a design without it
  (`design_contract.REQUIRED_SECTIONS`; docs/models.md has the table).
- **One pass.** `gate_design_review` sends a design back only when `## Changes` is missing
  or empty (`design_contract.BUILD_SECTIONS`), at most `design_review_attempts` times
  (default 1). It counts rewrites (`design_review_rewrites`), not visits, so a design the
  person sends back at `gate_strategy` keeps its rewrite. Everything else it finds is
  advisory and goes on the card; an optional section a design leaves out is not
  checked. It never reads a review.
- **The claims check resolves drift.** `check_design_claims` is an op, not a gate, so no
  edge can send the design back from it.
  - A quoted line that begins within ±20 lines (`QUOTE_WINDOW`) of its citation counts,
    and the note says "cited N, found at M"; the nearest match wins, and blank lines
    inside a quote that spans lines are skipped.
  - A quote with "…" or "..." matches where its parts appear in order within the lines
    it cites plus two (`ELLIPSIS_SLACK`), never more than fifteen
    (`ELLIPSIS_MAX_LINES`). A claim citing one line gets three: room for a call
    wrapped over a line or two, not for `load(` paired with the `) -> None:` of the
    method below it. A seven-line signature cited by its range (`run.py:15-21`) gets
    nine. The first part must hold a word, because `(…)` would match any parenthesised
    line; the parts after it may be punctuation that closes what it opened
    (`def helper(…):`, `f'{sys.executable} -m … ...'`).
  - Only the window is searched, with the few lines after it that a quote beginning on
    its last line could run into. For the card, the lines elsewhere in the file where an
    unverified quote (one without an ellipsis) does appear are found in one linear pass.
    On a 1.2 MB file the earlier whole-file search took 279 s.
  - A bare or package-relative path names the one tracked file that ends with it. This
    is `design_ops.cited_candidates`, the resolver `classify_scale` uses (dm-200). A path
    that is absolute, holds `..`, or whose real path (a tracked symlink followed) is
    outside the checkout names nothing, so no line from outside the checkout reaches the
    card.
  - What still cannot be confirmed is an **unverified claim**: a quote not in the window,
    a path not in the checkout or naming several files, an identifier the file does not
    define, or a probe id `probes.jsonl` does not hold. It is listed first on the card,
    as high severity.
- **Research and pitfalls never fail a design run.** Design mode's `gate_research`
  (`design_ops.gate_design_research`) and `gate_pitfalls` each send their step back once
  only when it produced nothing parseable, then proceed with whatever exists. Everything
  else — a plan outside 3–5 directions, an empty field, a pitfall with bad evidence — is
  listed on the card (`research_advisory`, `pitfalls_advisory`). Create mode keeps
  `create_ops.gate_research` exactly as it was; both use `create_ops.research_problems`.
  `research_attempts` and `pitfalls_attempts` are gone from design mode.
- **Optional sections stay optional for corrections and pitfalls.** Delivery restores
  the pre-correction design only when it had `## Summary` or `## Changes` and the
  corrections lost it, so a design without a Summary keeps its corrections. A design
  without `## Anti-patterns to avoid` is not checked against the pitfalls, and the card
  names the high-risk ones that leaves unchecked. For one without `## Rule and sites`,
  a site is a `path:line` it cites anywhere, the entry itself included, whose file is
  in the checkout or named in `## Changes`: a pitfall answered with the site that
  handles it counts, and a made-up `nowhere/made_up.py:999` does not. (Since dm-262 the
  pitfalls are not checked at all: the card quotes the lines that cite each one.)

### The card

`design_review_status` builds the card, in this order:

1. `Design review: PASS` or `FAIL, N unresolved`, with the section still missing.
2. `Scale: <scoped|full> — <reason>`.
3. `Unverified claims (high, advisory): N`.
4. `Claims check (advisory): N notes`.
5. `Other advisories: N`, with the research and pitfalls notes prefixed.

`gate_strategy`'s question opens with the card.

### Both runtimes

The gates and the claims check are Python on both runtimes; the spine runs each as
`python -m refactory_dsh.run`. Each node declares the facts it sets: `claims_check` and
`claims_check_feedback` (`check_design_claims`), `research_advisory` and
`pitfalls_advisory` (the two research gates). The design probe still reaches
`design_doc`: `steering.design_probe_steps` finds the writer whose output a person's gate
reads and a check between it and its reloop gate reads too, so it needs no reviewer.
The probe's call-site review line (`design_probe_review`) went with the reviewer.

`claims_check_attempts`, `research_attempts` and `pitfalls_attempts` are gone from
design mode, because nothing reads them; validation V7 refuses an unused knob.

### Tests that hold it

- `test_design_one_pass.py::test_no_reviewer_runs_between_the_design_and_the_gate` (full, scoped)
- `test_design_one_pass.py::test_drift_ellipsis_and_a_bare_name_cost_no_round`
- `test_design_one_pass.py::test_an_unverified_claim_is_listed_first_and_costs_no_round`
- `test_design_one_pass.py::test_a_design_missing_changes_goes_back_exactly_once`
- `test_design_one_pass.py::test_a_rewrite_is_counted_not_a_visit`
- `test_design_one_pass.py::test_the_gate_runs_through_the_node_runner` (proceed, reloop)
- `test_design_one_pass.py::test_a_light_design_is_read_as_a_contract_design`
- `test_design_one_pass.py::test_corrections_keep_a_light_design_and_restore_one_that_lost_its_changes`
- `test_design_one_pass.py::test_corrections_restore_only_a_section_the_design_had`
- `test_delivery.py::test_gate_corrections_on_a_design_without_a_summary_reach_the_pr`
- `test_design_one_pass.py::test_research_with_one_direction_proceeds_with_a_card_note`
- `test_design_one_pass.py::test_unparseable_research_goes_back_once_then_proceeds`
- `test_design_one_pass.py::test_unparseable_pitfalls_go_back_once_then_proceed`
- `test_design_one_pass.py::test_pitfalls_with_bad_entries_proceed_with_the_problems_listed`
- `test_design_one_pass.py::test_create_research_gate_is_unchanged`
- `test_design_claims.py::test_misplaced_quote_within_the_window_is_a_note`
- `test_design_claims.py::test_quote_beyond_the_window_is_unverified`
- `test_design_claims.py::test_an_ellipsis_quote_matches_its_parts_in_order`
- `test_design_claims.py::test_an_ellipsis_quote_needs_its_parts_inside_the_window`
- `test_design_claims.py::test_an_ellipsis_quote_does_not_pair_unrelated_code`
- `test_design_claims.py::test_an_ellipsis_quote_over_one_or_two_lines_still_verifies`
- `test_design_claims.py::test_an_ellipsis_quote_spans_its_cited_lines_plus_two_and_never_more_than_fifteen`
- `test_design_claims.py::test_an_ellipsis_quote_may_end_in_punctuation`
- `test_design_claims.py::test_a_long_signature_cited_by_one_line_is_unverified`
- `test_design_claims.py::test_the_ellipsis_claims_from_design_runs_against_their_real_lines`
- `test_design_claims.py::test_quote_beyond_the_window_names_at_most_three_lines`
- `test_design_claims.py::test_a_large_file_is_searched_only_near_the_citation`
- `test_design_claims.py::test_a_path_leading_out_of_the_checkout_is_unverified`
- `test_design_claims.py::test_a_quote_that_begins_inside_the_window_counts`
- `test_design_claims.py::test_the_nearest_match_in_the_window_is_the_one_noted`
- `test_design_claims.py::test_a_quote_spanning_a_blank_line_matches`
- `test_design_claims.py::test_a_bare_or_package_relative_path_names_its_one_tracked_file`
- `test_design_scale.py::test_only_a_missing_or_empty_changes_section_sends_the_design_back`
- `test_design_scale.py::test_a_light_design_passes_with_nothing_to_list`
- `test_design_scale.py::test_a_light_design_names_the_high_risk_pitfalls_it_never_mentions`
- `test_design_scale.py::test_a_tracked_symlink_out_of_the_checkout_names_nothing`
- `test_pitfalls.py::test_the_gate_lists_quotes_and_names_high_risk_pitfalls_never_mentioned`
- `test_pitfalls.py::test_a_design_without_an_anti_patterns_section_is_quoted_all_the_same`
- `test_design_scale.py::test_every_scale_gets_one_rewrite_for_a_missing_section`
- `test_design_contract_extensions.py::test_prompt_contract_requires_only_summary_changes_and_tests`
- `test_design_contract_extensions.py::test_an_optional_section_left_out_is_not_checked`
- `spine.spec.ts`: "a scoped change skips the perspectives step, and no reviewer runs at either scale"


## 32. Refactory's own checkout (dm-164)

### The problem

An agent that runs a refactory tool with the contract's Python imports refactory from
the checkout that Python belongs to: the driver, or for a pinned run the pin. The tool
writes there from any directory. On 2026-10-21 (b32l) an emit builder ran a generator
that way and wrote into the driver checkout. The target guard watched only the target
and the run's worktree, and gave the launcher checkout a warning, so nothing refused it.

### The rule

Every agent node of a run that recorded a target (`target_guard.record`) gets a
snapshot of refactory's own checkout when that checkout is not the target, on every
route (walker, the spine's worker bridge, the CEO subagent), because the snapshot is
taken in `agent_delivery.begin`. A change there fails the node, and the message names
the checkout.

The snapshot holds the checkout's **tracked** files that differ from HEAD, and HEAD
itself (`_checkout_state`, one `git status --porcelain=v2 --branch -uno`). Untracked
files are not watched: an editor's swap or lock file, a log appended through an open
handle, a new notes file and a copied tree were where false refusals came from, and a
generator rewrites tracked files. The target guard's own snapshot is unchanged.

`target_guard.checkouts` finds the checkouts:

- **The contract's own import root.** `refactory_source`, the directory
  `agent_delivery` is imported from, is what the contract's Python imports. On the
  walker (`python -m refactory_dsh.design` and the outer loop) that is the driver. On
  the spine, a pinned run's worker re-executes under the pin's Python
  (`worker_node.main`), so it is the pin.
- **The run's record.** `refactory ceo` records the driver in
  `.refactory/run-config.json` (`pin.record_driver`), and a pinned resume adds
  `pin_checkout` (`pin.resolve_pin`). The pin is watched when the run is pinned, the
  driver otherwise.

Only git checkouts count, and not a package a checkout ignores. A checkout that is,
holds or lies in the target or the run's worktree is left to the target guard, so a run
on refactory itself is unchanged. Nor is a checkout `REFACTORY_GUARD_UNWATCHED` names,
but only inside a pytest test (`PYTEST_CURRENT_TEST`, which the processes a test starts
inherit): the suite names its own checkout there (`tests/conftest.py`), so its scripted
agents never snapshot it. Anywhere else the variable is ignored, with one
`refactory: warning:` line, because it would switch the guard off without a word.

### What a refusal does to the files

One setting decides, `target_guard.CHECKOUT_POLICY`; the user chose `restore-if-clean` on 2026-10-27:

| Policy | What it does |
|---|---|
| `restore` (A) | Puts the changed files back as they were before the node (`restore_target`); the changed copies are kept under the delivery's `checkouts/` folder |
| `report` (B) | Leaves the files as they are and names them with what the node changed: a diff from their bytes at the node's start, not from HEAD |
| `restore-if-clean` (C, the default) | Restores when the checkout had no uncommitted tracked change when the node began, and reports otherwise |

Either way the person is told: a run warning, and a line in the checkout's own log,
`~/.refactory/target-guard/<checkout hash>/restores.log`, naming the files and where
their changed bytes went. A file whose pre-node bytes were not kept (larger than a
baseline holds) is never moved away: `restore_target` decides how a path is given back
before it moves anything, and leaves it in place when nothing can. A commit is not
undone; the message prints the `git -C <checkout> reset --keep <head>` that undoes it.
A checkout git cannot read after the node is named in a warning, not refused.

### Cost

One `git status` per snapshot, two per agent node: 54 ms median on the lab driver, of
which git's index refresh is 44 ms, and about 11 ms in a fresh worktree.

### Tests that hold it

`test_target_guard_checkouts.py`: the walker and spine refusals of a driver and a pin
write, a commit, a run on refactory itself, legitimate writes, untracked files, a file
whose bytes were not kept, an unreadable checkout, a parallel stray, both retry routes,
the launcher that is the driver, which checkouts are watched, the suite's own checkout,
and each policy.
