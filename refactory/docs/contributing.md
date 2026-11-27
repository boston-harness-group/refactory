# Contributing

For anyone — person or coding agent — changing `refactory/`. It assumes you have
read [the architecture overview](architecture.md), and it is about the rules that
are not visible from the code alone: which invariants must hold, what will break
if you move something, and how to check your work.

## The commands

Run from the repository root.

```bash
uv run python -m pytest refactory/tests/ -q                    # the suite
uv run python -m pytest refactory/tests/ -q -n 8               # ... on eight workers
uv run python -m pytest refactory/tests/ -q --run-integration  # ... plus the real-DSH runs
uv run python -m pytest refactory/tests/ -q -n 4 --shard 2/4   # ... one of CI's four shards

`-n` is a command-line choice, not a setting: `--toolchain-gate` must stay in one process
(it refuses `-n` rather than passing a suite it did not check), and the per-site base
check runs a single test under a time cap.
uv run python -m refactory_dsh.tools.gen_api_reference --check # docs match the code
```

### Run the tests your change touches

CI is the binding check: a pull request's four shards run every test once. While you
iterate, run only what the diff can reach, on eight workers:

```bash
pnpm test:changed             # select for your diff and run it
pnpm test:changed --list      # ... print the selection, run nothing
pnpm test:changed -- -x -k x  # ... pass pytest its own options
```

`scripts/test-changed` selects with design mode's own self-check selection
(`refactory_dsh/test_selection.py`): the test files the diff touched, the test files that
import a module it changed, and the engine and structural tests when it changes a hub. A
change to a file every test reads (`conftest.py`, `pytest.ini`, `pyproject.toml`, a pytest
plugin, `uv.lock`) selects the whole suite, because no import graph says what it reaches.
A changed fixture, document, root config or script selects the test files that read it.
The command runs the Python suite only: a change under `packages/` needs `pnpm test`. Run
the whole suite at most once before pushing, on eight workers; `--toolchain-gate` is the
one command that must stay in a single process.

`tests/test_ir_parity.py` checks every shipped mode's IR: that the state is valid,
that `to_dict()` round-trips through `from_dict()`, and that a field-set snapshot
catches unintentional additions or removals. It runs in the default suite.

The DSH side, when you touch `packages/`:

```bash
pnpm run build                 # typecheck, then the host and client bundle passes
pnpm run test                  # the eight packages' vitest suites
pnpm run lint                  # oxlint
pnpm run typecheck
```

The gates the harness copy carried are gone with it. [Verification](verification.md)
names what replaced each one, what a contributor runs before pushing, and what is not
covered.

`.work/todo.json` tracks the work, and `tests/test_work_complete.py` **fails while
any item is open**. Each item records its issue number, the command that verifies
it, and the observed result; that test also checks the command names files that
exist, because two items once named test files that never existed and nobody
noticed.

## The environment, and what it hides

**Lean and mathlib are optional, and their absence is silent.** The Lean tests skip when `lean` is not on `PATH`; the mathlib tests skip when `MATHLIB_PATH` is unset. A skipped test cannot fail, so a green suite on a machine without them says nothing about the checker path — which is how two real defects hid. Every test that exercises a real external tool is marked `toolchain`, and one command reports which ran, which skipped and why, and fails if any did not run:

```bash
uv run python -m pytest refactory/tests -q --toolchain-gate
```

No per-PR job installs Lean or mathlib (mathlib is a 7.4 GB checkout, and this suite is not in the DSH CI matrix), so those paths are verified on developer machines by that gate. A machine without the tools is expected to fail it and to say what it did not verify.

**A DSH profile is machine state, not a checkout artifact.** It is created with `dsh plugin --profile <name> add <package>`. This checkout owns two: `graph`, which the Python glue runs modes on, and `refactory`, which the GUI boots over the harness's own `web` template — the harness's `web` profile is never written to, so a plain `dsh --profile web` session shows no re:factory rows. `refactory` needs both the `ceo` preset layer and the web-surface layer — without the preset every session it creates without naming one stops, and without the surface the console and picker rows never mount. `install.sh` provisions both idempotently, `./refactory-dsh` heals a missing layer before booting, and `REFACTORY_SKIP_DSH_PROFILE=1` skips both. `refactory_dsh/dsh_profile.py` is the one place that answers "is the profile there, does it point to this checkout, and what fixes it" — it owns the layer and target tables, emits the `init`/`add`/`remove` plan the installer and the launcher execute, and detects a harness profile an old install mutated; `refactory doctor`, the runner and the tests all read it. The profile links to one checkout at a time. A run from a different checkout (a second clone, a worktree, a contributor's branch) will refuse and name the `dsh plugin` command that relinks the profile. A contributor working from a second checkout must relink or re-run `install.sh`. Martty is an **attached** target: the operator adds the preset layer to the profile they already have, and re:factory never creates it. Without the profile, anything that runs a mode on DSH stops at the first call.

**`install.sh` provisions the default install, not the optional parts.** It creates
`.venv` from `uv.lock`'s default (`dev`) group and installs the pinned pnpm workspace
with the eight packages' `lib/` built, then creates the DSH profiles above. It does
not create `refactory/vendor/srf/graphs`, the legacy graph directory for the low-level
`dsh --profile graph run` path; the public
`uv run refactory ceo <package>/<mode>` route supplies its own graph directory.
`.github/workflows/ci.yml` runs `install.sh` for the default install.

**CI runs the Python suite in four shards.** Each job runs `--shard I/4`: the suite
packed by its CI seconds (`refactory/tests/ci_durations.json`), so the four together
run every test once. `scripts/check-shards.py` checks the plan in the fast-checks job,
and after the shards the `shards-ran-the-suite` job checks what they really ran: their
JUnit ids together must be the whole collection, each once, and every shard's runner
must have collected the same suite. The pieces are whole files, except a file longer
than four minutes, whose test functions (each with all its parametrizations) are packed
one by one. A piece weighs what the durations file says, never what was collected, so
a case that only some runners collect (the code-node sandbox's) cannot shift the plan.
Within a job a file stays on one worker, so a file that long still holds up a worker;
splitting it by theme, as the delivery and design-worktree tests are (helpers in
`_delivery_support.py` and `_design_worktree_support.py`), helps more. Refresh the
seconds when the shard times drift apart: `uv run python scripts/ci-durations.py
--latest 5` (the last five green main pushes; the nightly builds the same file as its
`ci-durations` artifact), then open a pull request with the file. A new file counts as
the median file until then, a new function in a long file as that file's median
function.
Beside the shards run a fast-checks job (docs, typecheck, ruff for syntax errors and
undefined names, the boundary and registry tests), the TypeScript job and the macOS
job; the structure-edit harness's long tier runs nightly (`nightly.yml`).

**Use managed credentials for nested model execution.** Save the provider key through **Settings → Models → Edit → API key → Apply** in the DSH home used by the run. Exported keys and repository `.env` can serve the launching process, but nested smoke deliberately scrubs credential environment variables and changes `HOME`; the product carries the nonsecret DSH-home location so DSH can resolve its managed store. See [Models](models.md#credentials-for-nested-execution) for precedence and setup, and use `refactory doctor --provider deepseek-official --model deepseek-flash` to check downstream configuration before authoring. This check does not authenticate the key; unsupported custom runtimes report an explicit unverified warning. Deterministic modes, searches with `--optimizer random`, and offline tests do not need model credentials.

## The invariants

These are the rules the system is built on. Breaking one is not a style problem;
each exists because its absence caused a specific failure.

### The state is the authority after genesis

`build()` runs **once**, to produce the starting state. Every later architecture
comes from the state dict, because that is what the optimizer edits.

- Never read a knob's value in `build()`. A knob is consumed by a node at run time.
- Never make `build()` depend on I/O, the clock, or randomness — it must be pure,
  or `build() -> state -> build()` stops being an identity.
- `score()` and `invariants()` are **methods**, never state. That is what makes
  them unreachable by a proposal, and it is the whole protection. Do not put them
  in the state dict, and do not make a candidate able to influence them.

### A candidate is judged before it costs anything

`Optimizer.attempt` applies and validates the edit *before* the loop evaluates
anything. A proposal that would produce an invalid graph costs no budget.

- A new rule goes in `nn/validate.py` and in the `RULES` table. `tests/test_integrity.py`
  asserts the table and the emitted codes are the same set **in both directions**,
  so a rule that is emitted but unpublished fails the suite.
- The validator **refuses; it never repairs.** A repair hides the edit that caused
  the problem and makes the next failure unattributable.
- Macros are literal compositions of primitives. If you add one, express it as an
  `expand` into primitives rather than as new application logic, or it can produce
  a state no sequence of primitives could.

### Every knob is reachable, and every file is produced

- `V7` refuses a declared knob no node consumes **and** a node consuming an
  undeclared knob. A knob that looks like a search axis and is not is worse than
  no knob: the search spends batches learning nothing.
- `V6` refuses a node reading a file no node on a start-to-terminal path writes.
- `V11` refuses a declared port no node writes.
- A `str` knob may be declared `required=True`. Its default is then only a
  placeholder: a run with no value for it (in its knobs or its task), or an empty
  or blank one, is refused before its run directory exists, naming the knob -- by
  `Mode.run`, and by `interactive.prepare` for a headless run. A session a person
  opens may start without it, since the value can come in the conversation; there
  only a value given explicitly blank is refused. V9, which validates a mode's
  structure and starting state, refuses only a blank value a candidate sets.
- `Loop.unreachable_knobs()` reports knobs that *this optimizer* cannot move even
  though they are declared and consumed. It warns rather than refuses, because a
  knob-only optimizer on a structural mode is a legitimate pairing.

### The journal is the only durable state

`journal.jsonl` is append-only and replayed. Do not add a second snapshot of the
population and keep it consistent by hand — that is what the system this replaced
did, and it produced real bugs.

- Anything a later step needs must be **journaled**, not derived. `generation`,
  `refusal_pressure`, and `bound` are all journaled for this reason: a resumed run
  that guessed them would take a different path.
- An invalid candidate is recorded as a `Rejection` with its reasons, never
  dropped in silence. `E-duplicate` exists because a silently dropped duplicate
  loses a batch slot and makes the run report a batch it did not try.

### The two runtimes must agree

The walker is a debug runtime; DSH ships. Two tests run the same thing both ways:

- `tests/test_equivalence.py` runs a deterministic mode on the walker and on a real DSH
  session and compares the score and facts (`--run-integration`, real model calls).
- `tests/test_runtime_equivalence.py` runs one scripted design run (a gate answered with
  a correction, a self-check fix-up, the reviews, a pull request) through each runtime's
  real entry point, and compares visits, verdicts, facts, run files, prompts, model routes,
  commits and the delivery.

Its known differences are named parity gaps (`GAPS`), each a strict xfail. When you
close one, remove it from `SCENARIO_GAPS` in the same change; a difference that no gap
explains fails the test.

- If you change the executor, change nothing that makes a node behave differently
  on DSH. The lowering in `lowering.py` is the seam, and `refactory_dsh/ir.py` owns
  the IR it lowers into. `tests/test_ir_parity.py` checks every shipped mode's IR
  for validity, round-trip fidelity and field-set stability (a field added or
  removed without updating the snapshot fails the suite).
- The IR has **no field for the facts a node produces**, so `write_declarations`
  writes them beside the graph and the runner reads them from the run directory.
  That is a known wart, not an oversight to "clean up" silently.

## The failure modes worth remembering

Each of these happened, and the shape of the code is the response.

- **A severed gradient.** The optimizer was not shown what the run learned.
  `backward()` produces the gradient and it is part of the API; `knob_attribution`
  includes `untried`, because *a value that failed* and *a value nobody tried* are
  different facts and a prompt that conflates them keeps re-proposing what it
  already knows.
- **Mutating operators that destroyed graphs.** Some rewrites produced structures
  that could not run. Hence: macros expand to primitives, the validator refuses
  rather than repairs, and validation happens before budget is spent.
- **A surface that lied.** Structural wins were counted from failed runs, and
  duplicates vanished. `structural_wins` requires `exit_ok` and a score above the
  candidate's parent (under the validation gate, a val score above the parent's);
  duplicates are refused
  at the optimizer layer and `attempt == 0` is asserted in the tests.
- **Steps that deleted their own run.** When the loop became a mode, each node
  rehydrates from the journal — and `resume=False` means "start over", so the
  first version deleted the journal between two nodes of the loop's own run. Any
  node that reads the run must resume.

## Where to put a change

| you want to | change |
|---|---|
| a new mode | a package under `refactory_packages/` |
| a new validator rule | `nn/validate.py` `RULES` + the check, then the anti-drift test |
| a new edit operation | `nn/edits.py`: a primitive, or a macro that expands to primitives |
| a new parent picker | subclass `optim.ParentSelector`, write `select`, list it in a package's `Package.optimizers` ([Adding an optimization method](methods.md)) |
| a new proposer | subclass `optim.Optimizer`, write `propose`, list it in a package's `Package.optimizers` ([Adding an optimization method](methods.md)); a proposer in the library itself goes in `optim.REGISTRY` through `optim.register` |
| a new outer loop | subclass `outerloop.OuterLoop`, override the steps, list it in a package's `Package.strategies` |
| a new optimization method | a package that composes the core blocks (pickers, proposers, `ValidationGate`, `FinalEvaluation`, the strategy trace), not a new block of code in `strategies.py`: see [Adding an optimization method](methods.md) |
| a new artifact | declare a `Port` on the mode **and** write it from a node |
| a new way to read a run | `observe.py` (readers) or `history.py` (finished runs) |
| the loop's own steps | `loop.py` — and expect `loop_mode/loop` to need matching changes |

## Review checklist for a new method

- A new optimization method reuses the core pickers, proposers, validation gate,
  final evaluation and strategy trace ([Adding an optimization method](methods.md)).
- Any helper it duplicated from core, or that other methods could use, is moved to
  core (one copy) and imported by the method.
- A faithful port pins its behaviour with a fixed-seed identical-trace test.
- Growth of an oversized module (see [What not to do](#what-not-to-do)), or a new
  module past the 1,500-line limit, is a finding.

## Documentation

- The [API reference](api/reference.md) is **generated**; never edit it. Run
  `uv run python -m refactory_dsh.tools.gen_api_reference`. `tests/test_docs_api.py`
  fails when it is stale, when a documented symbol has no prose, or when a
  document names a module that does not import.
- The architecture overview's rule table is checked against `nn.RULES`, so adding
  a rule means updating the overview too.
- A pull request that changes what the optimizer's calls are sent (the proposal, the
  analysts and their merges, memory and its comparison, the step review) includes the
  before/after size table from `uv run python -m refactory_dsh.tools.request_sizes
  <run dir> ...` on a run (or a replay of one, `python -m refactory_dsh.tools.replay_round`) made before the change and one made
  after it.
  `tests/test_request_guardrails.py` fails on repeated text and on sections out of
  order, never on size: a request may grow when it says more.
- Prose belongs in the overview and the tutorials. A comment belongs where the
  reason for the code is local to it.

## What not to do

- **Do not add a second runtime.** The walker is for debugging; DSH ships. A third
  executor is a third set of behaviours to keep in agreement.
- **Do not silently accept an invalid candidate** to keep a batch full.
- **Do not default a missing referent.** `Package` resolution refuses a bare mode
  name rather than guessing a package, and the registry refuses a name that exists
  in both roots rather than picking one.
- **Do not repair a state.** Refuse it and report the reason.
- **Do not grow an oversized module.** Keep modules small and single-purpose: about
  1,500 lines per module and about 800 per class is the threshold past which a split
  is due. The following modules are already well past that limit and must not grow
  by more than a small, localised fix — new behaviour goes in a new or existing
  focused module, imported from there:

  | module | lines |
  |---|---|
  | `refactory_dsh/loop.py` | ~9,100 |
  | `refactory_dsh/optim/base.py` | ~5,900 |
  | `refactory_dsh/nn/core_ops.py` | ~4,700 |
  | `refactory_dsh/nn/executor.py` | ~2,700 |

  A split is a pure move first: code moved verbatim, old import paths re-exported
  until callers migrate, the full suite and lowering snapshots unchanged. Behaviour
  changes go in separate PRs.

## Public capability coverage

See [discovery](discovery.md) for the shared parser/mode contract, semantic review
metadata, offline generation and extension checklist. Run `pnpm run docs:generate`
after reviewed changes and `pnpm run docs:check` before building. New public
operations, mode schemas and worker capabilities require current explanations,
examples and explicit CLI/chat/panel routes.
