# The refactory system, as built

Audience: someone who has to change this code, and needs to know what the pieces
are and why they are shaped that way. The descriptions follow the implementation; example outputs illustrate their recorded configuration and are not a guarantee for every model or workload.

This describes the system that exists now. It replaces an earlier one built on
`refactory_dsh.ops.scaffold` and `refactory_dsh.ops.outer_loop`, which are gone.
The requirements that earlier system failed to meet are recorded at the end, under
[Why it looks like this](#why-it-looks-like-this).

---

## 1. The one-sentence version

A **mode** is a Python class whose architecture and parameters are serializable
data, and the **outer loop** is a program that searches over those data for a
better one. The vocabulary is PyTorch's, deliberately.

## 2. The PyTorch mapping

| PyTorch | here | where |
|---|---|---|
| `nn.Module` | `Mode` | `refactory_dsh/nn/mode.py` |
| `nn.Parameter` | `Knob` | `refactory_dsh/nn/knob.py` |
| `state_dict()` | `Mode.state_dict()` | `mode.py` |
| `load_state_dict()` | `Mode.load_state_dict()`, strict | `mode.py` |
| `torch.optim` | `Optimizer` | `refactory_dsh/optim/` |
| forward pass | `Mode.run()` | `mode.py` |
| **loss / cost function** | **`Evaluator`** | `refactory_dsh/evaluators.py` |
| loss value | `Loop.loss_step()` | `refactory_dsh/loop.py` |
| backward pass | `Loop.backward()` | `loop.py` |
| `optimizer.step()` | `Loop.step()` | `loop.py` |
| training loop | `Loop` | `loop.py` |
| `torch.save` | `history.read_run` / the journal | `history.py`, `loop.py` |

**Three extension points, all seams.** `Optimizer` proposes and cannot run
anything; `OuterLoop` owns the search; `Evaluator` answers the one question the
loop's loss is built on — *given a mode and a state, run it and tell me what it
scored*. Each is a base class with a registry, so a new one is a new class and not
a new branch. The evaluator was a string compared against a literal inside
`Loop._evaluate` until issue #34.

**One deliberate divergence.** In PyTorch a state dict is parameters only; the
class is the architecture. Here the state dict carries **architecture as well as
parameters** — `{version, graph, knobs, ports, memory}` — because the outer loop
searches structure too. The consequence is stated once and matters everywhere:

> After genesis, the **state** is the authority, not the class.
> `Mode.build()` runs once, to produce the starting state; from then on the
> optimizer edits the state as data and never re-runs `build()`.

A knob is therefore consumed by a node at run time and is never read by `build()`
to choose structure. The validator refuses a knob that no node consumes (`V7`), so
the two mechanisms cannot be confused.

`score()` and `invariants()` are **methods**, and are not in the state. No
proposal can reach them. That protection is structural, not defensive.

## 3. The state model

```python
state = {
  "version": 1,
  "graph":   {"start": "task_input", "nodes": {...}, "edges": [...]},
  "knobs":   {"grid": {"value": 10, "spec": {...}}},
  "ports":   [...],   # what the run publishes
  "memory":  [...],   # named namespaces, each with a retention policy
}
```

Serialize/parse is **canonical**: nodes sorted by id, edges by
`(source, target, condition)`. That is what makes `build() -> state -> build()` an
identity rather than an approximation — `refactory_dsh/nn/graph.py`.

The layer below a `Mode` is refactory's own language — `Graph`, `Node`, `Edge`,
`Workflow.to_dict()`. The graph is *data*: an optimizer edits this, never Python
source, which is what makes a candidate revertible, diffable and exactly validatable.

## 4. The validator

`refactory_dsh/nn/validate.py`. A candidate is judged *before* anything runs, so
an invalid proposal costs no evaluation budget. The codes are published to the
optimizer's prompt via `rules_text()`.

| code | rule |
|---|---|
| `V0` | the state must be readable: it needs a graph, and the graph must parse |
| `V1` | the graph must name a start node, and it must exist |
| `V2` | every node must be reachable from the start node |
| `V3` | the graph must have a terminal (a node with no out-edge) |
| `V4` | a non-gate node has at most one out-edge, and it carries no verdict |
| `V5` | a gate has at least two out-edges, each with a condition, and no unconditional one |
| `V6` | every file a node reads must be written by a node on a path from start to a terminal |
| `V7` | every knob a node consumes must be declared, and every declared knob must be consumed |
| `V8` | every call and evaluator must resolve to a real callable |
| `V9` | every knob value must be of the declared kind and within its bounds; a required knob must be non-empty |
| `V10` | every cycle passes a gate; after explicit `reloop` edges are removed, each cyclic region has one gate that owns its retry budget |
| `V11` | every declared port must be a file some node on a path writes |
| `V12` | a fork's targets, a join's sources and a data node's subgraph must name nodes that exist |
| `I:` | the mode's own rule, stated in `Mode.invariants()` |

Two properties of the implementation are themselves tested:

- The `RULES` table and the codes the code actually emits are the **same set, in
  both directions** — an AST test, so a new rule cannot be added without
  publishing it (`refactory/tests/test_integrity.py`).
- `on_path_nodes` is what `V6` and `V11` are computed against, so a node that is
  unreachable neither satisfies a read nor validates a port.

The validator is **naive on purpose**: it rejects, it never repairs. Repairing
hides the edit that caused the problem.

## 5. The edit algebra

`refactory_dsh/nn/edits.py`. Edits are typed data, applied to a state:

- **17 operations**: `add_node`, `remove_node`, `replace_node`, `set_node_field`,
  `add_edge`, `remove_edge`, `set_knob`, `add_knob`, `remove_knob`, `insert_after`,
  `insert_before`, `wrap`, `bypass`, `duplicate`, `split`, `reorder`, `loop`.
- The first nine are primitives; the rest are **macros** — literal compositions of
  primitives, so a macro cannot express anything a sequence of primitives could not.
- **4 presets** declare which operations a run may use: `broad` (17),
  `structure` (17), `conservative` (4), `knob_only` (3).

```bash
python -c "from refactory_dsh.nn import edits as E; print(E.names())"
```

## 6. The extension points

There are three kinds of thing that can be plugged into a run, and they are separate
on purpose:

- **`Optimizer`** — proposes. `propose()` returns `Proposal`s; the harness
  validates and evaluates them. It cannot run anything, so it cannot waste budget.
  Seven ship: `random`, `hill_climb`, `evolutionary`, `archive`, `islands`, `map_elites`, and `autoresearch` (which asks a model).
- **`OuterLoop`** — owns the search. `run()` is `init -> Run -> strategy ->
  report`, and every step a strategy would change (`select_parents`, `propose`,
  `batch_size`, `accepts`, `update`, `should_stop`, `on_epoch_end`) is a method to
  override rather than code to copy.
- **`Evaluator`** — runs a mode and reports what it scored: the loop's loss. Two
  ship. `walker` runs the graph in this process and is the fast one a search needs;
  `dsh` lowers the state to IR and runs it on the runtime that ships, at the cost of
  a process boot per candidate. `out_of_process` says which is which without
  matching on a name.

```bash
python -c "from refactory_dsh import evaluators as E; print(E.names())"
# ['dsh', 'walker']
```

All three take the same shape: a base class, a `REGISTRY` keyed by `name`, a
`create(name, ...)` that fails loud on an unknown one, and a `register()` for a new
implementation. `tests/test_evaluators.py` asserts that a third evaluator needs no
edit to `loop.py`, because that is what a seam is for.

`Run` is what a strategy is handed: `attempt` (apply a typed edit and validate it,
microseconds, no budget) and `evaluate` (one inner run, one unit of budget). The
API makes it impossible to blur those two.

**SRF's thirteen modes split across both buckets**, and the split is asserted so a
port cannot go missing or be filed wrongly:

```bash
python -c "import refactory_dsh.strategies as S, refactory_dsh.optim as O; \
print('loops  ', S.names()); print('propose', O.names())"
# loops   ['adaptive_meta', 'autoscientists', 'best_of_n', 'default', 'reflective',
#          'reflective_sampler', 'research_then_optimize', 'staged_curriculum',
#          'staged_v1', 'tree_search']
# propose ['archive', 'autoresearch', 'evolutionary', 'hill_climb', 'islands',
#          'map_elites', 'random']
```

Four of the thirteen are **optimizers** — `scs`, `evox`, `openevolve` and `karpathy`
— plus `map_elites`, which lives in `srf/lab/` rather than among the modes and is
counted by `SRF_LAB_PORT_COUNT`. The other nine are outer loops. `SRF_PORT_COUNT` is
the total, so `13 - len(SRF_OPTIMIZER_PORTS)` is the loop count rather than a number
kept in step by hand.

They were all loops until issue #32, and the cause was the contract rather than the
filing: `Loop.optimizer_step` passed a proposer only the elite, and
`Candidate.describe()` carries a score and the edits that produced it but **not the
state**. A policy that needs a wider pool — islands, a niche grid, a tournament over
an archive — could not be written as a proposer. The loop hands over the evaluated
population now and the optimizer selects, which is why `ProposalContext` carries
`elite`: an optimizer with no parent policy of its own takes that many, so widening
the pool does not silently change what it proposes.

## 7. The loop

`refactory_dsh/loop.py`. The eight steps are named like a training loop, and they
are also the nodes of `loop_mode/loop` (see §9), which is what lets a further loop
search the loop.

| step | signature | what it does |
|---|---|---|
| `init` | `() -> None` | seed the population, refuse a target that cannot run |
| `forward` | `() -> int` | evaluate the batch; the only step that spends budget |
| `loss_step` | `() -> dict` | turn the batch into one number |
| `backward` | `() -> dict` | read the traces into the gradient |
| `optimizer_step` | `(run=None) -> (candidates, rejections)` | propose, validated before anything runs |
| `step` | `(candidates, run=None) -> list` | install offspring that are genuinely new points |
| `scheduler` | `(*, evaluated=0, proposed=0) -> bool` | close the epoch, say what bound the run |
| `report` | `() -> dict` | the visibility surface |

A candidate is a state dict plus its lineage. Evaluating one is a forward pass:
materialize the mode from the state, execute its graph, score the run. An invalid
candidate never reaches `forward` at all — it is refused with a reason, and the
reason is reported (`summarise_reasons`, `I:` first).

Confirmed by running the running example:

```bash
uv run refactory optimize circle_packing/adaptive \
  --epochs 2 --batch-size 4 --optimizer random --seed 5 --runs-root /tmp/walk
```

seed `0.795` → best `0.875` (`raw_best_score` `0.955`: the compared score is the raw one minus `0.01` per node), bound `epochs`, 8 inner runs, 1 structural win
(`set_knob grid=18` + `duplicate accept`), and refusals across
`V2 V3 V4 V5 V6 V7 V10`, `I:fact_produced`, `E-shape` — none of which cost a run.

## 8. Durability, and what a run leaves

A run is a **directory**, because a reader should not need a service to read it.
`refactory_dsh/history.py`, `observe.py`.

The loop's `--runs-root` names that directory. The console's `--runs-root` names a
**parent** of run directories, because `console start` gives every run a
subdirectory; to read a run the loop wrote directly, pass `console --run <the
directory>`.

- `journal.jsonl` — every event, append-only. This is the run's durable state, and
  the **only** place a candidate's whole state is written down. Resume replays it
  (`Loop(config.resume=True)`).
- `run.json` — the layout version. A reader fails loudly on a foreign layout
  rather than half-reading it (`RunVersionError`, `RUN_VERSION`).
- `report.json` / `report.md` — the visibility surface.
- `gradient.json`, `loss.json`, `state.json`, `config.json`.
- `progress.jsonl` — one line per candidate evaluated, not per epoch, so a reader
  can tail a run that is still going.
- `manifest.json` — what to render and as what; `rejections.jsonl`.
- `runs/<candidate-id>/candidate.json` — per-candidate measurement.

The exception worth knowing: `report.json`'s `best` is a `describe()` — id, parent,
score, edits — and carries **no state dict**. To get a candidate's architecture you
read the journal. That is a known gap; see §11.

## 9. The two runtimes, and the tower

- The **walker** (`refactory_dsh/nn/executor.py`) is a debug runtime: it executes a
  graph in-process.
- **DSH** is the runtime that ships. `refactory_dsh/lowering.py` lowers a mode
  state to the IR `refactory_dsh/ir.py` owns and writes `<mode>.graph.json` for the
  science bundle, which mounts `workflow-spine` and runs each node. The reference
  that IR was copied from is a test-time oracle: `tests/test_ir_parity.py` compares
  the two byte for byte over every shipped mode and asks the reference's validator
  to accept the result, recording the two modes it refuses as limits of that
  validator. It is installed only with `uv sync --group reference`.

A debug runtime that disagrees with the real one is worse than none, so two tests run
the same thing both ways and compare:

- `refactory/tests/test_equivalence.py` runs a deterministic mode (`circle_packing/adaptive`)
  on the walker and on a real DSH session, and compares the score and the facts. It makes
  real model calls, so it runs only with `--run-integration`.
- `refactory/tests/test_runtime_equivalence.py` runs one scripted design run on each
  runtime through its real entry point: `python -m refactory_dsh.design` on the walker,
  `refactory ceo` on the spine, with the real lowering, spine and workers, and one fake
  `dsh` answering by node id. It compares node visits and outcomes, gate verdicts, facts,
  run files, each agent's prompt and model route, the commits pushed and the pull request.
  It runs in the ordinary suite.

  Every difference it finds is either equal or explained by a named parity gap in its
  `GAPS`, each a strict xfail. **When you close a gap, remove it from `SCENARIO_GAPS`**:
  its xfail starts passing, which fails the suite until you do.

What the two share, rather than each doing its own way:

- **One prompt.** An agent node is sent its rendered template and nothing else: its
  notes say what it does (its event summary, a gate's question), and are not part of
  its prompt. A reloop's feedback is said once (`agent_delivery.with_feedback`: added
  unless the prompt already carries it as whole lines), and
  the execution block lists its inputs in name order, as the lowered graph keeps them
  and as `Ledger.review` fingerprints a gate's. The session is opened with the node's
  role on both (`worker_node` reads it from the worker spec).
- **One set of run files.** The walker writes `facts.json` (`run.write_facts`) and
  `.refactory/spine-status.json` (`run_state.write_status`) as the spine does, keeping
  the completed nodes and each gate's last verdict by the spine's rules
  (`nn/record.py`). A gate's verdict is the run's record, in the spine's checkpoint and
  the walker's progress record (and `RunResult.gate_verdicts`), never a fact; a gate out
  of reloops is recorded `halt` on both. Each status file names its `runtime` (`walker`
  or `spine`) and `owner_pid`. A walked run has no checkpoint, so whether it is still
  alive is its progress's and its walking process's to say (`run_state.walked`,
  `run_state.owner_pid`): the worktree, artifact-name and console readers ask those, the
  console reports its kind as `walker` and offers no graph controls, graph-control
  refuses it naming the walker's own resume command, and `invocations.prepare` archives a
  walked directory's files for a new run even when it was walked without declarations.
- **One set of run flags** (`run_flags`, dm-134). `refactory ceo` and the walker's mains
  (`python -m refactory_dsh.design | create | edit`) add the same flags, and turn them
  into the same `NAME=VALUE` knob list, model route and resource limits: the brief, the
  name and the check travel as knobs on both, not as the walker's task. `--verdict` is
  an alias of `--approve`, and `--attempts N` of the mode's attempts knob
  (`check_attempts` for design, `attempts` for create and edit, `repair_attempts` for
  their `_fast` variants). A flag a runtime cannot honour is refused with the
  equivalent named (the walker never waits at a gate, so it refuses `--timeout` and
  names `--max-elapsed-s`; an authoring run refuses `--intent` and `--source`), and a
  knob a mode does not declare is refused by name on both.
- **One run configuration** (dm-160). Both write `.refactory/run-config.json` through
  `run_preparation.prepare_run`: the mode, project, knobs, model route, resource
  limits, configuration fingerprints and driver record. Three fields name the runtime
  (`runtime`, `command`, `resume_command`); a spine run also records `owner_argv`, the
  fragment its supervisor restarts the owner with, and a walked run, which has no
  supervisor, records none.
- **One resume check** (dm-160). The walker's progress outcome is read as the spine's
  status (`recovery.WALKER_STATUSES`: pending→ready, completed→done, failed→failed,
  running or unknown→executing), and `recovery.resume_status_check` decides with the
  spine's allowed set and messages; only the command it names is the runtime's own. A
  walked run is refused, as the spine's is, while a dispatched worker's outcome is
  unknown, when a declared input or output changed since its progress record
  fingerprinted it (at every boundary, as the spine's checkpoint does), or when its
  prepared configuration changed. The driver checkout's own files are not held to their
  launch fingerprints on either: the spine re-runs under its pin, and the walker, which
  has none, goes on with the code checked out now, as it always has. `--acknowledge-failure` on the walker's mains is the
  spine's `graph-control acknowledge` plus `resume --acknowledge-failure` in one: it
  sets aside what the boundary cannot vouch for (`recovery.set_aside`, shared with
  graph-control), records the acknowledgement bound to the progress record, and the
  resume consumes it.
- **A gate waiting for a person is not a visit.** The walker reads a user gate's answer
  before the visit starts, so a gate nobody answered leaves the run `refused` with no
  started or failed node, as the spine logs none.

**A run that stops is a failure with a name, on both.** On DSH the graph runner reads
the cursor's own status after the CEO's turn: anything but `done` writes a line naming
the status and the nodes still due and exits non-zero, so an abandoned graph cannot
report a score as if it had run. The CEO cannot abandon one in the first place — the
graph profile mounts the subagent tool foreground-only, because backgrounding a node's
subagent is what abandoning a graph looks like from the inside. On the walker the same
rule is structural: a node that fails ends the run with `exit_ok` false and the node's
name in `error`.

```bash
uv run python -m pytest refactory/tests/test_equivalence.py -q --run-integration
# 7 passed
```

#### Walker failure policy and limits

When `enable_recovery` is `True` (the default for all walker authoring runs and
outer-loop evaluations):

- **FnNode-only retry.** An op node (`kind="op"`) that raises gets one retry.
  Gates (`kind="gate"`) are timed but never retried (their reloop mechanism via
  `max_iterations` is the graph-level retry). An agent node is retried only by
  its own declared rule (`Node.retry`, below), on both runtimes;
  `DshModelClient` has its own transport retry with exponential backoff. Fork,
  join, and data nodes are structural.

- **Declared node recovery.** An agent node may declare
  `retry={"on": ["timeout"], "times": 1, "timeout_factor": 2}`: a session that
  times out is run once more, in a fresh session with twice its `timeout_s`
  (which a retry requires). The rule
  is the graph's, never a CEO's judgment, so a headless run recovers as a
  supervised one does. Both runtimes run it through `node_retry.deliver`: the
  walker from `_visit_agent`, the spine from the worker every dispatched agent
  runs (`worker_node.execute`), whose `command_timeout_s` reserves every
  attempt's limit plus the dispatch overhead (`node_retry.node_allowance_s`).
  Each attempt goes through the delivery bracket and its target guard; a
  timed-out attempt that changed the target is refused, not retried, and
  otherwise its outputs are set aside so the retry writes them afresh. A retry
  checks the run's budget first, is clamped to what is left of the node's
  envelope and the run's wall, and is recorded as a `node_retried` event that
  `console status` counts. Only a node that declares `outputs_only` (its
  outputs are its own declared report files) may declare a retry; validation
  refuses one on a node that may change the run's worktree or the target, and a
  timed-out attempt during which the run's worktree changed is not retried
  (another branch of its fork may have written there). Design mode
  declares it on the verifier only, the one agent seen to time out; a node is
  added when it is seen to time out.

- **Fact restore.** Before each op attempt, `ctx.facts` is deep-copied. On
  failure, facts are restored from the snapshot before retry.

- **Per-op timeout.** Derived from the shared `command_timeout_s()` formula in
  `lowering.py` (matching the DSH spine's lowering classification) clamped to
  the remaining budget with a 1 s floor. Nodes without a classification get
  DSH's 3600 s default (only for classified nodes). Ops and fn gates with a timeout run in a child process
  forked from a preloaded forkserver (via a module-level
  `multiprocessing.get_context("forkserver")` context whose
  `set_forkserver_preload` is called with `refactory_dsh.nn.executor`,
  `refactory_dsh.run`, and the modules the graph's ops and fn gates name).
  The forkserver is a separate single-threaded process, so forking from it is
  safe while the walker runs fork branches on threads; it has the same
  picklable-arguments contract as spawn. The parent kills the child at the
  limit via session-grouped SIGTERM→SIGKILL escalation (SIGTERM process group,
  ≥5 s grace, SIGKILL process group). An op that exceeds its timeout raises
  `OpTimeoutError` naming the node id with outcome `"failed"` (not `"unknown"`).

- **Child-process context.** The child gets the same context DSH's runner
  builds, via the shared `build_op_context()` helper: memory store, declared
  reads, declared facts. No budget — the parent's kill is the enforcement.
  Model identity (client kind and DSH context) flows to the child so
  prepare/create_ops decide correctly. Exceptions round-trip with type, message
  and traceback.

- **Budget readers with None.** Under `enable_recovery`, ops get no
  `ctx.budget`. Readers: `fast_ops.py:111,226,307,573` (skip budget check);
  `base_check.py:425,582,721` (skip budget-clamped timeout);
  `repo_change.py:1095,1148` (skip budget check); `delivery.py:1426` (skip
  budget check); `design_ops.py:636` (skip budget check);
  `create_ops.py:1592` (skip budget check).

- **Budget integration.** `BudgetExhausted` is never retried. `OpTimeoutError`
  is never retried. Per-op timeout is clamped to `remaining_s()`, so
  `--max-elapsed-s` cannot be overrun by a single op.

- **Fork branches.** `enable_recovery` is copied to fork-child `Executor`
  instances. Branch ops get the same retry and timeout policy.

- **DSH runtime parity.** `run_node()` in `run.py` (the DSH spine's per-node
  runner) applies the same one-retry policy for FnNode ops with fact restore.
  Gate callables are never retried. This ensures one policy for both runtimes.

#### Walker recovery in the outer loop

`WalkerEvaluator` enables `enable_recovery` by default. One transient op failure
does not end a candidate's evaluation, and one hung op does not hang the search.
`DshEvaluator` uses the spine's own `command_timeout_s` enforcement and does not
use `enable_recovery`. Child-process kill ensures clean trial boundaries: no
leaked processes or state between candidates.

**The tower.** `refactory_packages/loop_mode` expresses the loop itself as a mode:
eight step-nodes, a gate that loops back to `forward` or leaves for `report`, and
`score()` reading the best score the inner run reached. Because it is scorable, a
further loop can search the loop's own settings. It reaches the same best score and
evaluation count as `Loop` on the same seed, and runs end to end on DSH:

```bash
uv run python -m pytest refactory/tests/test_loop_as_mode.py -q --run-integration
# 21 passed
```

**And it has been searched.** `batch_size` is a real lever on the inner result —
deterministic on seed 0, a wider batch reaches a better inner score — so a run over
the tower has something to find:

```bash
uv run refactory optimize loop_mode/loop --epochs 1 --batch-size 3 \
  --optimizer random --seed 0 --runs-root /tmp/tower
# seed 0.705 -> best 0.87, raw 0.96, by `set_knob batch_size 4 -> 8`
```

That is D14's claim demonstrated rather than described: the loop's own settings are a
search surface, and searching them moved the inner result. `test_loop_as_mode.py`
runs the search and asserts the lever, because the test that stood here asserted only
that the knobs were *visible* — expressiveness, not use.

## 10. The meta-modes

Three modes whose *output* is a mode or an artifact, in the package `default_pack`,
which is the package a checkout starts with.
All three produce something **new** and never mutate their input.

| mode | also an entry point | input | output |
|---|---|---|---|
| `default_pack/create` | `uv run refactory ceo "…" --mode create --task <module:Class>\|--single` | a brief, and which of the two things it is building | a package: `__init__.py`, `mode.py`, `ops.py` |
| `default_pack/edit` | `uv run refactory ceo "…" --mode edit` | a brief + an existing mode | a revised package, recording `revision_of` **and the state it came from** |
| `default_pack/design` | `uv run refactory ceo "…" --mode design` | a brief | a checked software artifact |

They are discovered and validated like any other mode, and they are also entry
points because a creation is a thing you run once from a shell as often as from a
loop.

`create` and `edit` are remote-factory's **create-v2**, and `design` is its
**design-v2**: a research director chooses what has to be learned and has a
subagent answer each question, a strategy director chooses the perspectives and
records which asks each one covers, `synthesize_strategy` compiles them into one
intent-faithful plan, and design-v2 adds `design_doc` — the design document the
code is written from. As in remote-factory's design mode, research, strategy and
design go straight to the person's gate: **no reviewer** sits between `design_doc` and
`gate_strategy`, at either scale. Two deterministic steps run there. The
`check_design_claims` op verifies each claim in the subject-at-citation-quote format:
that the cited path names a file (as written, or the one tracked file that ends with it,
by the resolver `classify_scale` uses, and never a file outside the checkout), that the quoted line is within ±20 lines of the
citation (a quote with an ellipsis matches where its parts appear in order there,
within the cited lines plus two and never more than fifteen, the first part holding a
word; a line that drifted gets the note "cited N, found at M"), and that named identifiers from
the claim's subject are present (Python via `ast`, TypeScript via word-boundary search).
What it cannot confirm is an **unverified claim**; it never sends the design back. Then
`gate_design_review` checks the design itself.

**One design pass.** The design is written to a light template: `## Summary`,
`## Changes` (one bullet per file and what changes) and `## Tests` (repo-relative ids
and what each proves); `apply_corrections` adds `## Corrections at approval` when the
person corrected the design. Every other section is optional — Failure trace, Ask
mapping, Rule and sites with its markers, the two exemption sections, Anti-patterns,
Docs, Runtime coverage — and every consumer of one treats its absence as a design
without it. `gate_design_review` sends a design back only when `## Changes` is missing
or empty, at most `design_review_attempts` times (default 1). Everything else is
**advisory** and shown on the approval gate's card: unverified claims first, then the
claims check's notes, then a missing `## Summary` or `## Tests`, a citation past the end
of its file, a quotation the brief does not contain (or differs from only in
punctuation), rule-site marker and title formatting, and a high-risk pitfall the
design never mentions. The pitfalls are quoted, not judged: the card lists each one
with up to two of the design lines that cite it, under their headings, or "not
mentioned" (dm-262). An optional section a design leaves out is not checked. Research and perspective findings
the design does not cite are recorded in the `finding_coverage` fact and not listed at
all. When its rewrite is spent the gate proceeds to `gate_strategy` with the gap shown —
it never reads a review and never fails the run. In design mode the research and
pitfalls gates never fail the run either: each sends its step back once only when it
produced nothing parseable, and lists anything else it finds on the card (create mode's
research gate is unchanged). docs/hardening.md, "One design pass", has the measurements
behind this.

**Design scale.** Scale is decided after research by a deterministic `classify_scale`
op node placed between `gate_research` and `pitfalls_researcher`. `prepare_design`
sets `research_scope` to `scoped` for a `--refine` run (with
`research_scope_reason` = `"refinement of an open pull request"`), or `full` with an
empty reason otherwise. After `gate_research` proceeds, `classify_scale` reads every
direction's `footprint` object from `research-plan.json` and applies a three-part
rule: the change is scoped only when it touches at most two existing functions, adds
nothing new, and changes no shared engine. Anything else is full; so is a direction
with no footprint key, or an entry without file:line evidence. In a new or empty
project (a checkout with no tracked file) everything is new: it is scoped when it adds
at most three new items and changes no shared engine, and full otherwise. The brief is never
read by the classifier. The `research_scope_reason` fact records why the run was
classed as it was, and the approval card's scale line shows it ("Scale: scoped —
<reason>"). A scoped change gets at most five pitfalls, and one step is left out:
`route_strategy` sends it from the pitfalls straight to `synthesize_strategy`, which
plans "the simplest thing that could pass the check" from the research and pitfalls
with no perspectives step. The routing gate answers `scoped` or `proceed` (also what a
person's skip of a gate sends on the spine), which both runtimes follow as edge labels.
Non-refine research is guided to 3 to 5 directions. A full design keeps the
perspectives step. At either scale the design cites a research finding only where it changes a
decision; the findings stay in `research/` and `strategy/` for the design writer to
read.

**Light refine.** A refine skips more than the perspectives step. `route_refine_design`,
right after `prepare`, answers `light` on a refine (the default, `refine_design=light`)
and the run goes straight to `design_doc`: no research, pitfalls or strategy agent runs.
The gate writes those steps' files and facts as "skipped" sentinels, so every later node
finds its declared reads on both runtimes, and gives `design_doc` the pull request it
follows up as `refine_context`: its description (framed as other people's words and
capped at 6,000 characters), and its diff stat and changed files against its base branch
as GitHub has it now ("unavailable" when that cannot be read, never "empty"). A person's
reloop at `gate_strategy` goes to `synthesize_strategy`, which gets the same block as its
`strategy_plan`, and the next design still gets `refine_context`. With
`refine_design=full` (`--refine-design full`), or on any run that is not a refine, the
gate answers `proceed` and the full phase runs. Prepare reads the refine from the task
(the walker) or the `refine` knob (the spine, where a node's task names only the node).

The `deliverable_kind` fact, computed by `prepare_design`, distinguishes repository
changes from artifacts. The `review_instructions` fact renders `REPO_CHANGE_REVIEW_INSTRUCTIONS`
(composed from verification methods M2, M3, M11, M12) or `ARTIFACT_REVIEW_INSTRUCTIONS`
into each reviewer's prompt via `{{review_instructions}}`. The `changed_functions` module
maps diff hunk line ranges onto function spans via `ast` (Python) and hunk-range entries
(TypeScript). It marks each entry `test` or `generated`, and keeps only the run's own
changes when the branch merged its base in. The product entries are rendered into prompts
as `{{changed_functions}}`.

For the post-build QA gate, `gate_qa` validates each of the three review files independently,
naming the reviewer and function in any refusal. Missing repository questions are
advisory: the gate proceeds, nothing is re-run, and the pull request lists them. The fast `gate_qa` validates `focused-qa.json`'s
`repo_questions` key and strips `[repo-questions]`-prefixed problems from `failure_kind`'s
word scan. `gate_strategy` now opens with a `design_review_status` fact, the approval
card: "Design review: PASS" or "Design review: FAIL, N unresolved" with the section
still missing, then the scale, the unverified claims, the claims check's notes and the
other advisories,
carried by fact substitution to all template-rendered surfaces (terminal, GUI,
headless refusal, `refactory status`). A **person answers `gate_strategy`** at that point, because a
run that can approve its own plan is not reviewed at all, so a walker run stops at
the gate and records the visit; a headless DSH run stops and prints the steer and resume commands;
the caller can supply the verdict up front via the ledger. The build then passes structural admission and canonical runtime validation inside the bounded repair cycle. Before the review fork, a **self-check** runs: an attempt's first runs the whole check once, beside an **early asks judge** that reads the diff against the brief's asks and the approved corrections, and a fix-up's runs the last failures and the tests that what the fix-up changed can break, in chunks within a time limit. The attempt's own failures and the asks the judge found not built go back to the builder together as a fix-up, amended into the same attempt, without spending an attempt or running the reviews; a failure the base shares is inherited unless the brief or a correction names it, in the self-check and in the whole check alike (docs/hardening.md, "Self-check inside the attempt"). In design mode the check runs **beside the reviews**, as a branch of the review fork, on the commit `prepare_check` settles first, reusing the first self-check's run while the attempt is unchanged, and `gate_check` routes on it after the join (docs/hardening.md, "The check beside the reviews"). Registration supplies execution visibility, not completion. Then three
reviews (a health check, a code review and an adversarial QA pass) run **in
parallel** and merge into one synthesis. In create and edit, the QA gate reads its
blocking findings, and Overwatch audits them against the brief, accepted criteria/design, latest intent ledger and actual final validation receipts. In design mode **judges decide**: a QA judge reads the synthesis whole and writes a JSON verdict (proceed, or redo with what must be fixed) and the evidence map; a correction audit measures each correction's sites against the diff and its tests, and decides nothing; Overwatch reads that audit and writes its own JSON verdict. Each gate reads its judge's JSON and never scans prose. Only an ask, a correction, a failed health check or a critical or high defect sends the build back, a correction Overwatch finds unmet always does, and a judge whose verdict cannot be read is asked again once, judge only (docs/hardening.md, "Judges decide QA in design mode"). An **independent verifier** runs in parallel with the reviews as a branch of the review fork, applying design-blind methods to the brief, diff and repository without reading design.md, within ten minutes — for a repository change only those the code reviewers do not apply (M1, M2, M13, M15), for a new artifact all seven; see docs/hardening.md §15. Overwatch judges its findings; in design mode the QA judge reads its report with the reviews. A **failure digest** is built from the check output per runner (pytest, vitest, node's test runner) and stored in `check-record.json`, which survives redo; the builder receives the digest rather than the raw output. The **builder works in a workspace**: a prepared clone of the last attempt, where it changes the code and runs the tests its change touches and the brief's own example (the check runs the full suite), and whose diff the runtime writes into `emit.txt` (docs/hardening.md, "Agent workspaces"). A **builder probe** is available during design_doc: design agents can run commands (`--command`) in a throwaway clone to verify mechanisms the design relies on. At the post-build gates, **per-correction audit** checks each gate correction item by item: for every correction in the ledger it records the site in the diff, the test that reaches that site (via mechanical hunk-level revert), and the evidence scope (via AST-based patch-target detection); see docs/hardening.md §25-27. A QA or
overwatch reloop goes back to the emit step, not to the plan a person approved.

**Core and non-core findings.** In create, edit and the fast modes, every post-build reviewer (the three reviews, Overwatch,
and the fast modes' focused QA) ends the first line of each BLOCKING finding with `[core]` when it breaks
an ask in the brief, an acceptance criterion, a correction or constraint the person
gave, or existing behaviour, and with `[non-core]` otherwise: extra tests, docs, edge
cases beyond the asks, refactors and style. An untagged finding, or one whose tag is
anywhere but the end of its first line, is core, so a missing or misplaced tag never
lets a core defect through; a finding one reviewer calls core is core even if another
calls it non-core. A fast-mode check of something non-core that fails names itself
with the tag. Only a core defect sends the build back; the
round's non-core findings ride along in the same feedback, for the builder to fix if
it is cheap. A round whose findings are all non-core proceeds without spending a
review attempt. Overwatch is told to state `VERDICT: PASS` when only non-core
findings remain, so its FAIL always blocks; a PASS with non-core findings proceeds
only when the intent audit passes every requirement and the evidence and correction
audits hold. The non-core findings a gate proceeded with are
the run's **follow-ups**: `reviews/follow-ups.json` and `reviews/follow-ups.md` list
them, `reviews/finding-sources.json` gives each finding its `class` and the fate
`follow-up`, and the pull request carries a "Follow-ups (non-core findings)" section
while its Verification says the reviews found no core defect. The delivery result
reads "…, with N follow-up(s)". The verifier never gets a builder round of its own:
its findings ride along with a core repair, and otherwise go to Overwatch, which sends
the build back only for one that breaks an ask, and to the pull request (see
docs/hardening.md §15). It is bounded at ten minutes on both runtimes.

**Finding coverage.** Each research and perspective file carries finding ids in the format `FND-<slug>-<n>` (e.g. `FND-api-surface-1`). `gate_design_review` records, in the `finding_coverage` fact, which finding ids from `research/*.md` and `strategy/*.md` the design cites, read from the detail files rather than the summaries (`research-plan.json`, `strategy-plan.json`). A design cites a finding only where it changes a decision, so an uncited finding is neither listed at the person's gate nor a reason to send the design back; the findings stay in their files for the reviewer. Create mode's `review_design` gate runs the same check against the `findings` field of `strategy.json`; create's review raises when its attempts run out.

All authoring review uses the original brief, approved design and explicit user steering
as its acceptance baseline. User-approved non-goals remain valid; an author or
reviewer cannot waive a requested behavior by documenting a new limitation.
Observed noncompliance stays blocking unless the user explicitly changes that
requirement. Coordinators pass the returned node prompt in full, and the three
reviews receive both the brief and design. These are prompt and context safeguards,
not automatic semantic enforcement: fresh modeled acceptance is needed to assess
whether reviewers follow them, and existing artifacts/reviews are not repaired.

The [authoring validation contract](authoring-validation.md) defines execution ownership, current-source/intent bindings, requirement maps, unavailable dependencies and the Edit comparison branch.

The parallel reviews are a **fork/join** pair, and a mode may also declare a
**`data`** node that runs a subgraph once per item; both are node kinds in the mode
vocabulary (`Node.targets`, `Node.sources`, the DataNode's subgraph fields), lowered
to the IR's own fork/join/data nodes and checked by `V12`. Both runtimes execute a
fork by running each branch to its barrier. The Python walker schedules up to four
independent flat branches concurrently, isolates their contexts, refuses conflicting
declarations, and merges successful facts in target order before the join. A failed
fork keeps its pre-fork facts and resume position; its completed file artifacts remain
available for audit. See the [fork limits](concepts.md) for unsupported nested and
shared-memory scheduling fallback. Both runtimes execute a `data` node by loading the items
(`inline_items`, or a `directory`, `jsonl` or `csv` source, then the node's limit)
and running the subgraph once per item with the item written to `current_item.json`.
The loader is one implementation (`nn.executor.load_items`); the walker runs the whole
item loop in one process, while DSH's spine drives it from outside through
`refactory_dsh.items list|begin|end`, because a subgraph the cursor enters has to be
walked one node at a time.

### The exceptions this package needs

`default_pack` is where the mode contract is deliberately stretched, and each stretch
is a declaration rather than a special case in the library:

- **Unscored, on purpose.** `create` and `design` do not override `score()`, because
  a creation is judged by whether what it produced works. Because `is_scored()` is
  derived, an unscored mode would otherwise be indistinguishable from one whose
  author forgot, so a mode says why with `Mode.unscored_reason`. The registry and
  the sandbox accept a stated reason and still refuse silence, and the outer loop
  still refuses such a mode as a search target — you cannot hill-climb on "did it
  work".
- **Their ops live in the library.** `create_ops` and `design_ops` ship with
  `refactory_dsh`, because a mode that writes a package needs the validator and the
  registry. A package may name any importable ops module. Where a design run works,
  and how its change lands and is checked, is `repo_change`'s, which any mode that
  changes a repository attaches (`fix_pack` is one); design and design_fast are two
  such modes, not special cases of the engine.
- **They need a model.** No knob makes them run without one, so a run with no client
  fails at the first agent node rather than producing nothing.

## 11. Steering a run you are using

Two ways to change a run, and they are different acts on different things:

| | what changes | whose job |
|---|---|---|
| **structure** | the mode's graph and knobs | the outer loop, and a result is adopted as a new mode |
| **steering** | what the agents know | the person, while *using* the mode |

**The graph does not change when you steer.** That is what steering means during use:
you add context, correct something, or forbid something, and the run accounts for it.

The carrier is the **intent ledger** — an append-only, timestamped file in the run
directory, seeded from `--intent` and appended to while the run goes:

```bash
uv run refactory optimize circle_packing/adaptive --intent "make it better, no new deps"
uv run refactory steer --run <run> --note "no numpy" --kind constraint
uv run refactory steer --run <run>          # read it back
```

Three kinds, because they are not the same act: `info` (context), `correction`
(something is wrong), `constraint` (must be respected, not merely known).

**How it reaches the agents.** `steering.for_run` folds the ledger into **every**
agent node's prompt before a run, and **both runtimes call it**: the walker in
`execute`, and DSH as the transform `write_mode_graph` applies. Every node, not one
the author remembered to wire, so a mode author cannot be the thing that makes
steering work — which is also why this is a transform over the IR and not a field on
`nn.Node`. Modes stay Pythonic and unchanged, and `to_workflow` knows nothing about
any of it. A note added to the ledger mid-run therefore reaches the agent nodes that
run afterwards, which is what `console steer --note` is for.

**A search carries the ledger into each candidate.** `for_run` folds in the ledger of
the work directory it is handed, and the loop evaluates a candidate in
`runs/<id>/` while the ledger sits at the run root, so `Loop._carry_ledger` copies it
down before each evaluation. Without that copy an `optimize --intent` or a
`steer --note` was written, reported by `steer`, and read by nothing — in the one
place steering is most wanted, because a search is what runs for hours. The copy is
per candidate so the directory says what the person had said when it ran.

**Steering command supply (product/core boundary).** The workflow spine's
`wf_steer` tool requires a `steerCommand` and `steerArgs`. These are product
values — the product module `refactory_dsh.console` owns the steer verb. To
keep core clean, the product registers a `refactory.steer` Python entry point
(in `pyproject.toml`); the core module `refactory_dsh.steer_probe` discovers it
at import time via `importlib.metadata.entry_points`. At graph startup,
`startup.ts` runs `python -m refactory_dsh.steer_probe` and publishes the
result through the `graphStartup` Cordis service. The core YAML
(`cordis.patch.yml`) reads the steer fields via `!!js` expressions, so no core
file names a product Python module as a literal string. A core-only deploy
(without the entry point) boots successfully: startup publishes `steerCommand:
''` and logs a warning; `wf_steer` is not registered, and the spine degrades
gracefully. A stale build (new YAML, old startup.ts) is detected by the YAML's
own `!!js` comma expression, which warns once naming `pnpm run build`. The
interactive driver's preflight (`interactive.py`) calls the same `find_steer_module()`
helper in-process; a fresh start without the entry point refuses before the run
directory is created, naming `uv sync`.

**A person's gate answer is recorded and consumed for that review.** The ledger
keeps the decision and the hashes of the gate's declared review inputs. An
interactive answer or public `steer --verdict` authorizes one matching gate visit;
changed inputs expire an unused answer, and a later visit asks again even when the
files are identical. Legacy unscoped answers remain history, not authorization.
Explicit upfront `--approve` or `RunContext.verdicts` remains standing authorization
from the caller. A ledger-backed upfront approval remains active when that same
work directory is resumed; it has no separate invocation identifier. Without a
current scoped answer or standing authorization, a
walker run **stops** at the gate; a headless DSH run stops and prints the steer and resume commands.

**It is inert when unused.** An empty ledger renders the empty string and `annotate`
returns the workflow it was given. A run nobody steered is the run it would have been,
which is what lets the interactive and headless paths be compared at all.

**Steering context travels beside artifacts, not inside their syntax.**
`fidelity(ledger)` reports a ledger that is missing, empty, or holding the
placeholder; `Ledger.seed` refuses the old `"Build something useful."` default.
At agent dispatch, the runtime supplies the current ledger and records that context
under `.refactory/agent-delivery/<id>/`, alongside the exact prompt. Repeated DSH
node views retain the original file snapshot and archive any changed prompt/ledger
context; those receipts show what the runtime supplied to the CEO, not proof that
the CEO forwarded it verbatim to a worker. Completion
records the declared outputs' hashes and whether final text or a tool write supplied
them. `console steer` exposes these receipts and reports whether their output fingerprints still
match (a later node may legitimately enrich them); outputs that disappeared are
reported separately as missing. This establishes context delivery, not that the model read, understood, or
satisfied it. Old runs' inline `Serves:` citations remain readable, but new nodes are
not asked to append prose to Python, JSON, or other structured artifacts.

**Tool-written artifacts and final responses are separate.** The walker snapshots
declared outputs before calling the model. A newly written or changed output remains
unchanged when the model returns a status message. If the first declared output was
not freshly written, the walker retains its existing final-text capture contract for
that first file; additional declared outputs must have been delivered. A file left
untouched from an earlier pass cannot satisfy that requirement. A real rewrite to
identical bytes is recognized from file metadata as well as its hash. The exact final
response is kept separately as `final-response.txt`, never stripped or repaired, and
a text fact is populated from the delivered first artifact when one is declared
(binary or directory outputs retain the original final-response fact).
The file-writing DSH path checks declared delivery before accepting agent completion;
its original worker response remains in the DSH session transcript rather than being
invented or copied by the CEO into a receipt. Neither path automatically certifies the
meaning or format of arbitrary file contents; downstream checks still own that work.
Explicit structural controls such as skip and director expansion retain their existing
semantics; a graph edit is not an ordinary verified agent delivery.

A note given with a `proceed` verdict at `gate_strategy`, in the same steer call or a
later one before resuming, becomes a gate correction: `pick_gate_corrections` collects
it and `apply_corrections` rewrites `design.md` before the build, on both the walker
and DSH runtimes. A correction given later, while the change is built or reviewed, is
one too (dm-226): every send-back to the builder goes through `pick_corrections`, which
applies a correction it has not applied yet before the builder's next round, and
`gate_overwatch` sends back a change built without one rather than pass it. `steer`
says which will happen, or that a run past its final review will not build it. The rewrite covers every part a correction touches, `## Changes` and
`## Docs` included, because the builder reads the whole document. The pull request
lists the corrections, one short bullet each, under "Corrections given at approval",
after its Follow-ups.

### A run with a person at a gate — the command and what it showed

`refactory_packages/review_pack` exists for this: `review` is
`task_input -> draft -> check(fn) -> approve(user) -> publish|reject`. No earlier
mode declared a gate a person answers, so the path had nothing to exercise it.

```bash
uv run refactory ceo review_pack/review \
  --knob 'topic=circle packing' \
  --intent "draft a note about circle packing and let me approve it"
```

Run against this checkout, with the terminal answering `1` (proceed) at
`[gate approve]`:

- **The IR carries who answers.** `approve` lowers to `evaluator_type="user"` with
  **no** `evaluator_command`; `check` lowers to `"fn"` **with** one.
- **The runtime asks, the model cannot answer.** The spine refuses to route a `user`
  gate without an answerer, and ignores the model's `verdict` argument: a headless
  run of the same mode stops at the gate with
  `the graph did not finish: gate approve is answered by a person and could not be
  asked: no user-questions answerer accepted the request`. Before this, the CEO's
  invented `proceed` was routed and the run reported "approved by the person".
- **The answer is recorded without granting future approval.** The terminal and
  browser answer is recorded as consumed in the run's ledger. Reopening the same
  work directory asks a fresh question. If the draft changes while a question is
  pending, the old answer is refused and the current draft needs a new answer.
  Explicit upfront approval is the supported way to grant standing authorization.
- **A mid-run note reaches the agents.** `console steer --run <workdir> --note …
  --kind constraint` appends to the ledger, and `for_run` is what folds it into the
  agent nodes of a run that starts afterwards.

## 12. Known gaps

Recorded here rather than in a backlog, because each one is a place the system
promises more than it delivers.

- **The fidelity gate cannot judge the work.** `steering.fidelity` checks the ledger
  precondition and current delivery receipts (or legacy inline citations). A receipt
  establishes the context supplied and the artifact bytes delivered, not semantic
  compliance. Whether the work **satisfies** the intent still requires reviewing the
  actual output against the request.
- **Two suites are load-flaky.** `packages/boot/app-boot/tests/hmr-config.spec.ts`
  and `user-patches.spec.ts` poll file watchers with 20s and 10s deadlines and fail
  under full-suite contention while passing standalone; `hmr-config` is not in
  `processBoundTests` while its sibling `app-boot.spec.ts` is.

**Not a gap, but worth knowing.** Four strategies — `research_then_optimize`,
`staged_curriculum`, `staged_v1`, `autoscientists` — make a decision that is only
readable by a proposer that reads `ProposalContext.strategy`. `random` does not read
it, by definition: it samples the catalogue. So under `--optimizer random` those four
behave as `default` does. That is the optimizer doing its job, not the strategy
failing; a decision needs a reader, and a sampling proposer is not one. Pair them with
`--optimizer autoresearch` to see the difference.

## Why it looks like this

The system this replaced failed in ways worth remembering, and the shape above is
the response to them. These are the findings that survived the rewrite:

- **A search has to be given something *informative*, not merely something
  *valid*.** The first real run was correct in every structural sense and useless
  as a search: the starting population was sampled across the whole edit
  catalogue, so `set_knob` — three of the operations — was drowned by structural
  edits, and epoch one evaluated candidates that differed only in structure and
  all scored the same. A validator can only reject what is invalid; it cannot make
  a batch worth learning from. The response is that `init` **sweeps** instead of
  sampling: one candidate per movable knob, each at a value the default is not, so
  the first epoch produces a gradient with contrast in it. The same distinction is
  why `knob_attribution` carries `untried` — a value that failed and a value
  nobody tried are different facts.
- **The gradient was severed and everything downstream was blind.** The optimizer
  was not given what the run had learned. The response is that `backward()`
  produces a gradient that is an explicit part of the API, `knob_attribution`
  includes **untried** values (a value that failed and a value nobody tried are
  different facts), and the report says what the optimizer actually received.
- **Knobs that lie.** A declared parameter nothing consumed looked like a search
  axis and was not. The response is `V7`, and `unreachable_knobs()` reported as a
  warning naming the lens it is invisible through.
- **Mutation operators that destroy the graph.** Some rewrites produced structures
  that could not run, and a repair pass hid it. The response is that macros are
  literal compositions of primitives, the validator is naive and refuses rather
  than repairs, and every candidate is validated *before* it costs a run.
- **Visibility defects that made the surface lie.** Structural wins counted failed
  runs; duplicates silently vanished, so a batch tried fewer points than it
  claimed. The response is `E-duplicate` refused at the optimizer layer
  (`attempt == 0` asserted), `structural_wins` requiring `exit_ok`, and the
  progress/report split.
- **Requirements the run owner raised** while using the deleted system: artifacts
  invisible in the GUI; a created mode with no identity or home; created modes not
  runnable from where they land; nothing registered by name. Those are what
  `#14` (package lifecycle and registry), `#17` (the GUI), and the `ports`/
  `manifest` surface answer.

The two handoff notes those findings were extracted from lived in
`refactory/docs/learnings/`. They were 933 lines describing the mechanics of the
replaced system — module paths, presets, and run directories that no longer exist
— so they were removed once this section had absorbed what was still true: the
requirements they raised, and the lessons above. Keeping them would have made a
newcomer's first reading a description of the wrong system.

## Where to go next

- [The concepts](concepts.md) — the nouns first: a package, a mode, the outer loop
  and the GUI, with pointers into the detail here.
- [API reference](api/index.md) — every public module and symbol.
- [Tutorials](tutorials/index.md) — the running example, built up.
- [Packing a tool that is not Python](packing-tools.md) — how a package declares a
  toolchain it cannot import, how that requirement is checked before a run spends
  budget, and how its absence is reported.
- [Contributing](contributing.md) — for a coding agent or a person changing this:
  the commands, the environment and what it hides, the invariants, and what not to do.
- [Hardening](hardening.md) — what an adversarial audit of the finished work found,
  and what holds each fix in place.

### Fresh graph invocation ownership

Terminal `ceo` and console/GUI `wf_open` share `interactive.prepare`. Preparation
stages a complete lowering before retiring any previous state. Prior declared
outputs and runtime facts, run-retention memory, graph declarations/status and agent-delivery receipts
are archived under `.refactory/invocations/<id>/artifacts/`, with an `archive.json`
manifest; a failed installation restores the prior paths. Current facts start empty.
The new mode's external inputs, explicit item selection and intent ledger remain
in place, including documented standing upfront approvals. This is a fresh graph
invocation, not a resume or an implicit redo. Unmanaged files and declared cross-run
memory are not swept away; only recorded output ownership and runtime paths are
retired. An old run lacking ownership declarations refuses reuse rather than
guessing which files to move.

## Public capability coverage

See [discovery](discovery.md) for the shared parser/mode contract, semantic review
metadata, offline generation and extension checklist. Run `pnpm run docs:generate`
after reviewed changes and `pnpm run docs:check` before building. New public
operations, mode schemas and worker capabilities require current explanations,
examples and explicit CLI/chat/panel routes.

## Core/product boundary

The codebase is partitioned into **core** and **product** modules, in preparation
for splitting into two repositories (`refactory-core` and `refactory`).

**Core** contains the DSL (`nn/`), the engines (`dsh_runtime`, `evaluators`),
the workers (`model`, `workers`), the outer loop (`loop`, `outerloop`, `optim/`),
run state and control (`events`, `invocations`, `graph_control`, etc.),
`sandbox`, and `toolchain`.

**Product** contains the CLI (`cli`, `console`), modes (`design`, `create`,
`edit`, `fast`), delivery infrastructure (`repo_change`, `worktree`, `delivery`),
and supporting modules (`brief_guard`, etc.).

`target_guard` was reclassified as **core**: its only non-stdlib import is
`invocations.py` (core), and its functions (`watch`, `strayed`, `warn`,
`refusal`) are used by `agent_delivery` (core) and `repo_change`
(product → core, allowed).

**Dependency direction:** product → core only. A core module must never import a
product module.

### Classifying a new module

Add the module's dotted name to `refactory/refactory_dsh/data/boundary.json`,
in either `core` or `product`. Run `python -m pytest refactory/tests/test_boundary.py`
to verify. An unclassified module fails the test.

### Allow-listing a temporary leak

When a core module must temporarily import a product module (e.g. a concurrent
PR is changing both), add an entry to `allowed_leaks` in `boundary.json`:

    "refactory_dsh.<core>": [
      {"imports": "refactory_dsh.<product>", "reason": "<why>", "until": "<when removed>"}
    ]

The test verifies the leak still exists; once the import is removed, a stale
allow-list entry — one whose leak no longer exists — causes a test failure until
the entry is removed.

### Registering from product

When core needs to call product logic, product registers a callback that core
discovers at call time. The `importlib.metadata` entry-point mechanism is
preferred because it is independent of import order and of which process runs
the code.

Example: `worktree.py` provides a `_worktree_environment_overlay` function
registered as an entry point in `pyproject.toml`. `model.py`'s
`get_environment_overlay()` discovers and caches it on first use. With no
registration (core-only install), the overlay is `None` and the block is
skipped. A product registration is available only in processes whose
installed distribution declares the entry point.

### Product-supplied configuration

Two additional patterns let core consume product-defined behaviour without
naming product modules by string:

1. **Node properties (lowering).** Lowering detects smoke nodes by
   `SMOKE_PRODUCES.issubset(node.produces) and 'smoke_timeout_s' in node.knobs`
   — a two-factor predicate matching the `_is_check_node` pattern
   (`CHECK_PRODUCES` + `check_timeout_s`). A mode that wants the smoke
   envelope declares these in its node. Core reads the properties; it never
   matches a product module name. Product modes: `CreateMode` at
   `create.py:37`, `EditMode` at `edit.py:31`.

2. **Run-record fields (supervisor).** The supervisor reads `owner_argv`
   from `run-config.json`, written by `interactive.prepare()` at
   `interactive.py:139`. The stored argv omits `--workdir`; the supervisor
   appends the current root at call time, keeping its current-root behaviour
   and avoiding stale absolute paths. For backward compatibility with runs
   created before this change, the supervisor discovers the owner module
   from the `refactory.owner_module` entry point.  The supervisor fallback
   is the only owner-module lookup; it takes the first entry point without
   caching.

**Behaviour changes:** Behaviour is unchanged on both runtimes
(`python -m refactory_dsh.design` and `refactory ceo`). After pulling:
`uv sync` is needed to register the new entry point
(`refactory.owner_module`). Runs created before this change respawn through
the `refactory.owner_module` entry point, so `uv sync` is required. No
other manual action required. The `owner_argv` field in `run-config.json`
is written automatically by `interactive.prepare()` on new runs. Existing
in-progress runs use the entry-point fallback, which reproduces today's
behaviour exactly.

### Entry-point groups

These groups register product behaviour that core discovers at call time:

| Group | pyproject.toml key | Discoverer | Degradation | Stale-install guard |
|---|---|---|---|---|
| `refactory.environment_overlay` | `worktree = "…worktree:_worktree_environment_overlay"` | `model.get_environment_overlay()` | `None` → skip overlay | Yes — `RuntimeError` when worktree record exists |
| `refactory.worktree_state` | `worktree = "…worktree:_worktree_state_bridge"` | `agent_delivery.get_worktree_state()` | `None` → recorded returns None, skip shared_changed | Yes — `RuntimeError` when worktree record exists at workdir |
| `refactory.delivery_identity` | `authoring = "…authoring_evidence:delivery_identity"` | `agent_delivery.get_delivery_identity()` | `None` + not reviewer or no EVIDENCE → skip enrichment | Conditional — `RuntimeError` when node in REVIEWERS AND EVIDENCE file exists |
| `refactory.owner_module` | `interactive = "refactory_dsh.interactive"` | `supervisor._build_owner_argv()` | `None` → record start failure, supervisor exits cleanly | No — used only as fallback for old runs without `owner_argv` |
| `refactory.run_end` | `warm_clones = "…builder_probe:sweep_warm_clones"`, `workspaces = "…workspace:sweep"` | `nn.executor._run_end_cleanup()`, at the end of the walker's `execute` and of `dsh_runtime.evaluate_on_dsh` | none registered → no run-end cleanup; every registered cleanup runs, and one that fails is logged, never raised | No — a stale install leaves a builder probe's warm clone, or an agent's workspace, on disk after the run |
| `refactory.workspace_clone` | `clones = "…clones:prepare"` | `workspace.get_clone_preparer()` | `None` → a node that declares a clone workspace is refused | Yes — `RuntimeError` naming `uv sync` when a clone workspace is declared |

Each discoverer follows the same pattern: a module-level sentinel, at-most-one
`entry_points()` lookup, cache, and `None`. The stale-install guard fails
loudly when the entry point is missing but the run's state requires it.
`refactory.run_end` is the exception: `_run_end_cleanup` reads the group on
every call (under a millisecond, once per run) and runs every entry in it.

### Process traces

Every process that calls `model._environment()` gets the overlay through the
same entry-point discovery:

| Process | Path to model | Interpreter |
|---|---|---|
| Worker | `worker_node.execute` → `model` | .venv/bin/python |
| Interactive owner | `interactive.py` → `model` | .venv/bin/python |
| Console | `console` → `model` | .venv/bin/python |
| Supervisor child | reads `owner_argv` from run-config.json (the interactive owner) → `model` | sys.executable |
| FnNode ops | `python -m refactory_dsh.run` (startup.ts) → `model` | PATH python |
| Evaluator model judge | `evaluators.py` → sandbox → `model` | sys.executable (PYTHONPATH=refactory/) |

Every process that calls `agent_delivery.begin`/`finish` uses the worktree-state
and delivery-identity entry points:

| Process | Path to agent_delivery | Entry points used |
|---|---|---|
| Worker (`worker_node.execute`) | `worker_node` → `agent_delivery.begin`/`finish` | environment_overlay (via model), worktree_state, delivery_identity |
| Walker (`python -m refactory_dsh.design`) | in-process → `agent_delivery.begin`/`finish` | same |
| DSH bridge (`main()` via stdin) | `worker_bridge.mjs` → `agent_delivery.main()` | delivery_identity (for `_execution_contract`), worktree_state (for `finish`) |
| Console | `console.py` → `graph_control.request` | (reaches `recovery` for `CONSOLE_FILE`; `graph_control` for `DELIVER_ENTRY_POINTS` — both now core constants) |
| Graph-control commands | `graph_control.request('redeliver')` | (uses `DELIVER_ENTRY_POINTS` from `invocations`) |
| Interactive owner | `interactive.py` → `model` | environment_overlay |
| Supervisor's child | spawns `-m refactory_dsh.interactive`; reaches agent_delivery only through worker_node and the bridge | same as worker |

Sandbox children run `sys.executable` with `PYTHONPATH=refactory/` and keep
`REFACTORY_WORKDIR`. They find the entry point through the venv's dist-info or
`refactory/refactory_dsh.egg-info`, and `entry_points()` deduplicates the two,
so the more-than-one guard does not misfire.

**After pulling this change, every checkout must run `uv sync` (or
`./install.sh`)**, or every agent node in a worktree run stops with:

    RuntimeError: the worktree record exists but no environment overlay is
    installed; re-run `uv sync` or `./install.sh`

The same applies to the two new entry-point groups (`refactory.worktree_state`
and `refactory.delivery_identity`). A stale install that is missing any of these
produces a `RuntimeError` naming `uv sync` when the run's state requires the
entry point (e.g. a worktree record exists, or the node is a reviewer with an
evidence file).

The run's setup syncs only the worktree's own venv (`worktree.py`, first check
via `repo_change.py`). It does not re-install when only `refactory/pyproject.toml`
changes, because the install key covers the root manifests. The driver's
interpreter is synced only by `install.sh`; the `refactory-dsh` launcher runs
`.venv/bin/python` without syncing.

### Running the boundary test

    python -m pytest refactory/tests/test_boundary.py -v

The test enforces the boundary statically (AST scan), at runtime (import scan),
on TypeScript packages, and checks that allowed exceptions are still present
and that non-import references are tracked.
