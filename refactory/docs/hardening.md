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

### The builder probe

The emit prompt's "do not apply or test your change anywhere" exists because
direct edits to the worktree bypass materialise and the check.  The builder probe
gives a safe alternative.  Since dm-133, the probe is also available to
`design_doc` and `design_review` via `--command` (no `--emission`); the clone
runs at the run's base with `emission=None`, and the same isolation guarantees
apply:

- **Standalone clone** (`git clone --no-local --depth 1`): not a linked worktree.
  No `.git` registration.  `origin` removed after clone.
- **Hooks off**: `core.hooksPath=/dev/null`, `LEFTHOOK=0`, `HUSKY=0`.
- **Isolated HOME and DSH_HOME**: per-probe directories under `probe-<id>/`.
- **PATH stripped**: the run worktree's `.venv/bin` and `node_modules/.bin` are
  removed so tools resolve from the clone, not from the run's worktree.
- **Deleted afterwards**: the probe directory is removed in a `finally` block.
  Leaked probes are swept by `materialise` (owner PID check).
- **Not an attempt**: no `commit_attempt`, no attempt counter increment.
- **target_guard preserved**: the existing test at `test_design_worktree.py:667`
  continues to pass.  The probe cleans up before `finish()`.

#### Why each isolation guarantee exists

| Guarantee | Reason |
|---|---|
| Standalone clone | dm-96: a PR's checks registered 59 worktrees in the target's .git |
| Origin removed | A push from the probe could reach the shared .git |
| Hooks off | pnpm and lefthook rewrite shared hooks (dm-96) |
| Isolated HOME | Tool caches stay out of the worktree's diff |
| PATH stripped | A missing tool from the clone must not resolve into the run's worktree |
| Deleted afterwards | A tool timeout can SIGKILL the probe; materialise sweeps leaked dirs |

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
- **`gate_design_review`** (`design_ops.py`) does not need it — it reloops to `design_doc`, which can fix `design.md`.
- **`check_artifact`** (`design_ops.py`) does not need it — its findings come from test runner output (assertion errors, test names), not from free-form review text parsed by `blocking_findings()`.
- **`fast_ops.gate_qa`** (`fast_ops.py`) does not need it — it uses structured JSON checks from `focused-qa.json`, not `blocking_findings()` on review prose.
- **`fast_ops.gate_emit`** (`fast_ops.py`) does not need it — it reloops on materialise refusals, not review findings.

**Held by.** `test_gate_stale_findings.py`: `test_classify_finding_structured_worktree_file_stays_blocking`, `test_classify_finding_structured_run_doc_downgraded`, `test_classify_finding_freeform_stays_blocking`, `test_classify_finding_no_path_stays_blocking`, `test_run_doc_owners_from_declarations`, `test_scope_filter_with_custom_graph_document`, `test_classify_finding_exact_path_match`, `test_scope_filter_mixed_findings`, `test_gate_qa_downgrades_design_md_finding`, `test_gate_qa_keeps_worktree_finding_blocking`, `test_gate_overwatch_downgrades_design_md_finding`.

---

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
the review fork, bounded at `VERIFIER_TIMEOUT_S` (600 s) on both runtimes: the
node declares `timeout_s`, the spine gives its dispatched session that wall
(`node_wall_s`) plus the usual dispatch overhead, and the walker gives its model
client the same wall. Its prompt asks it to finish in about seven minutes and to write
each finding as it finds it, since a session stopped at the wall leaves nothing. A
verifier that fails, returns nothing or runs out of time is recorded as missing.
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

### Why design-blind

If the verifier read design.md, it would share the same blind spots as
every other reviewer. By working from the brief and the code alone, it
catches gaps the design missed, which is the failure mode the evidence log
documents.

### Pipeline position

Full mode:
```
check →(proceed) fork_qa → [health_checker, code_reviewer, qa_director, verifier] → join_qa → synthesize_qa → gate_qa
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
at every post-build gate (`gate_overwatch` in full modes, `gate_qa` in fast
modes). For each correction or constraint in the ledger it reads the overwatch's
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

**Fail closed.** Any status other than a known catch (`fail`) is a failure,
never a pass. This includes `not_in_diff`, `unavailable`, `collection_error`,
`import_only` (without a caught sibling) and `pass` (the test did not catch
the revert).

A clause that fails any check is BLOCKING. The message names the correction id,
the clause, the site and the test.

**Withdrawn corrections.** A correction whose audit row has `status == "excluded"`
with a `user_change` pointing to a later ledger entry is skipped — it was
withdrawn. This reuses the existing exclusion validation path.

**Evidence forms.** Not every clause has a code site:
- **prohibition:** requires a `pattern` field; code confirms the pattern is
  absent from the `+` lines of the diff. Missing `pattern` fails the clause.
- **doc:** requires a `path` field naming a changed documentation file
  (`.md`, `.rst`, `.txt`, or under `docs/`). A path to a non-doc file fails.
- **design:** requires a non-empty `heading` field that exists in the current
  `design.md`. Missing or unknown heading fails the clause.
- **test:** cites a code site plus the test; the revert runs against the code
  site, not the test file. A site inside a test file is not a code site.

A clause whose `sites` is not a list fails with "sites must be a list".

**New-file sites.** A site in a file the base lacks (`new_file` result) or a
site whose revert fails only with an import error on a name the reverted hunks
define (`import_only` result) passes only when the same clause has a sibling
site in an existing file whose revert fails its test.

**Test.** `test_correction_audit.py::test_correction_audit_unbuilt_blocks`,
`test_correction_audit_no_base_does_not_block`.

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
   - `unavailable` → timed out, signalled, test-file site, no tests ran
8. Restore the file with `git checkout HEAD -- <file>` in a `finally` block.

**Test-file exclusion.** A site inside a test file (`test_*.py`, `*_test.py`,
`*.spec.ts`, or under a `tests/` directory) is not a code site.

**Cache.** Results are cached per `(HEAD, path:line, test_name)` in
`.refactory/site-revert-cache.json`, a dedicated file separate from
`base-check.json`.

**Time cap.** Total revert time is capped below the gate's DSH timeout (240 s
default). Past the cap, remaining pairs are reported as `unavailable`.

**Blocking rule.** Missing or unavailable revert results are BLOCKING. The only
exemptions are:
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
nodes. Reviewers, the verifier, overwatch, design_review, apply_corrections and
expanded researchers are all guarded.

Concurrent runs on one target refuse each other's in-flight nodes when either
strays. A write that finishes after the node's `finish()` is reported only as a
later warning.

In a self-targeting run, a pull or commit in the driver checkout during an
output-only node now refuses that node. The driver checkout must stay still
while runs are in flight.

The driver or pin checkout is not watched when it is not the target. That is
tracker dm-164, a follow-up, not this run.

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
