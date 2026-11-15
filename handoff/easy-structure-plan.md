# Plan: structural changes as trivial as a prompt patch

**The idea.** The proposer gets a small set of generic, shape-level operators that say nothing about what a step is for: insert, remove, parallelize, serialize, rewire inputs, loop with a check, route and unroute, split and merge, add or remove a fork branch, configure a setting.
- Each operator is one line: what kind of step, where, and its text (a prompt or code).
- The system does all the wiring.

The value is in purpose-free infrastructure:
- automatic data wiring;
- automatic repair;
- catching breakage before scoring;
- a DSL coverage map with a test;
- support for every outer loop.

Purpose-level shapes are examples in the request, built from the core operators, never prescribed: several attempts and a choice, check and send back, an independent attempt, simulation then analysis, a literature pass beside an experiment, iterative fitting, a tool step before reasoning. The aim is that general tasks in science and other domains are not steered toward preconceived purposes.

**Scope.** The mechanics of writing a structural edit. The request's framing and structural evidence are another PR's job.

**Code read** at origin/main c52027bfdc (line refs are `refactory/refactory_dsh/...` unless noted) and remote-factory main `factory/outer_loop/mutations.py`. LiveMath runs appear only as friction data. The design, tests and examples are domain-neutral.

## 1. What the runs show

### Counts

| run set | proposer calls | structural proposals | refused (code) | evaluated, then broke items |
|---|---|---|---|---|
| round3/3a/3b D2s (the 12 runs, 61 calls) | 61 | 2 written, +1 repair | 1 (E-precondition: split, V7 orphan knobs) | 0 broke; both scored no gain (0.40 to 0.40, 0.40 to 0.31 with 3 timeouts) |
| round2 (A2, D2, D2j, D2s) | 72 | 6 | 0 structural (6 E-integrity, all prompt-only) | 0 |
| older structure-heavy arms (analysts D, feedback B2/C, full B, brerun B) | ~150 | 98 candidates | 3 refused (E-too-many-edits, E-noop, one V15 in smoke) | 3 failed on every item (34/35 or 33/35); 1 lost 11 items to the sandbox |

- In the 12 runs, the `considered` note mentions structure in 47 of 61 calls. Reasons given for not choosing it: risky or untested 21, cost or timeouts 15, the earlier memory note said no 13, **the mechanics 12**, not the bottleneck 10 (keyword count; a note can give several).
- What a passing structural edit costs to write: new nodes average about 1,400 characters of JSON (114 new nodes, 160k characters). Recent passing proposals were 700 to 4,500 characters. A prompt patch is one `set_knob` line.
- Older arms: structural candidates averaged +0.018 train change against their parent (43 up, 40 down of 94), knob candidates +0.030 (129 up, 79 down of 241). Parents were often scored on another epoch's sample, so treat these as rough.

### Representative refusals and mechanics complaints

1. round3b D2s-s45 epoch 4, `split` of `notes_prompts` + two `remove_knob`:
   "E-precondition: 'notes_prompts' is the only node reading knob(s) ['hypotheses_task', 'precision_task'], which a split would leave read by nothing (rule V7) ... [edit 1 of 3: split]".
   The bundle's edits 2 and 3 removed exactly those knobs. The precondition is checked before the later edits run (`nn/edits.py:1186`), so a correct bundle was refused. The repair call then gave up the split and added a code op that cuts `## Your Task` out of the op-built prompt with a regex.
2. feedback C-s43: a bundle that removed two review nodes and added a code step: "E-too-many-edits: the proposal bundles 5 edits and this run allows at most 4".
3. smoke B: "V15 [meta_check]: needs {{compare_notes}}, {{hypotheses_notes}}, {{precision_notes}}, which some path to it does not produce first". The proposer used file names as facts.
4. Mechanics in `considered`: "too many edits needed to modify fork/join" (3a s42 e6); "requires 4+ edits with high validation risk (modifying fork/join targets, adding reads to decide_prompt op)" (3b s42 e2); "abandoned due to risk of breaking unknown prompt format" (3b s45 e2); "requires multiple node additions exceeding edit limits" (3b s45 e3).

### Structural edits that passed the validator and then broke

| candidate | edit | what happened |
|---|---|---|
| feedback C-s44 675882f4e0ff, brerun B-s45 3819a22f7869 | `loop` with a `gate_kind: "agent"` gate | 33-34/35 items: "gate ... is answered by 'agent' and this run has no verdict for it. A headless run cannot answer a gate". The schema still offers `gate (agent)`; only `user` is hidden (`nn/node_schema.py:148-178`); the walker fails at run time (`nn/executor.py:2185-2215`). **Still true on main.** |
| feedback C-s43 bb5cfe503102 | code op reading `current_item.json` | 34/35 items: "AttributeError: 'list' object has no attribute 'items'". The proposer guessed the file's shape (`choices` is a list). Nothing runs a new code op before the candidate is scored. |
| round3a D2s-s45 1a3f6a52fef8 | new `challenge` agent | Prompt reads "## Compare Notes\n{{compare_prompt}}": `compare_prompt` is compare's input, not its notes. Nothing caught it; the step saw the wrong text (score unchanged). |
| brerun B-s42 7818d5efadd6 | code op | 11 items: "the sandbox did not start within 60 s" (lab infrastructure, not wiring). |

### Friction in today's mechanics (found while reading)

- **A repair call does not see the proposal it is repairing.** The retry request lists only the refusal reasons (`optim/base.py:926-933`). The full request (about 280k characters, about 4.6 min per call in round3b) is rebuilt, and the proposer writes again from scratch. The repair loop already exists: `propose_valid` makes up to `attempts` (3) calls with the reasons fed back (`optim/base.py:2752-2815`, `loop.py:237`). Analyst findings are cached across retries (`optim/random_optimizer.py:762`).
- **Conflicting guidance about files.** The composition text says a new agent "cannot read a file: hand it what it needs as {{facts}}" (`optim/base.py:1172`). The table of available inputs says declared reads are pasted into the prompt (`optim/base.py:1996`), and they are (`agent_delivery.py:545-551`). Proposers built code ops to turn files into facts when they did not need to.
- **The verifier example takes three edits:** `loop`, `insert_before` for the reviewer, and a code op to turn the step's file into a fact (`optim/base.py:1265-1277`).
- **`parallel` can only add a new fork.** It cannot add a branch to an existing fork, and "the successor's declarations do not change", so the proposer has to wire the outputs to a reader by hand (`nn/edits.py:1478-1484`). The LiveMath graph already has `step_fork`, so "add an independent branch" means editing the fork's targets, the join's sources, the edges and a consumer knob.
- **Nothing says which facts are free of model output.** To make a step independent, the proposer has to work out by itself that `user_prompt` and `compare_prompt` carry only the item, while `hypotheses_prompt` already contains compare's notes.
- **`bypass` does not handle fork branches or the readers of the removed step's files.** The validator then refuses, and the proposer has to fix it by hand.


## 2. Core operators

### The rule for every operator

- **One line.** The proposer says *what kind* (agent, op, code), *where* (a step id), and *what it does* (prompt or code text). Everything else is optional.
- **No hand wiring.** The proposer never writes edges, fork/join plumbing, file-to-fact ops, reader wiring or knob declarations. Raw primitives (`add_node`, `add_edge`, `add_knob`, ...) stay available for anything the operators do not express. Operators are shortcuts, not a closed library.
- **Same contract downstream.** Steps downstream of a change see the same file and fact names as before, unless the operator is about changing that (rewire, remove).
- **Knobs for tuning.** Every text the operator creates goes into a knob (`<id>__task`, `<id>__system_prompt`, `<id>__code`) for later `set_knob` tuning.
- **One change.** Each operator counts as one change toward the 4-change cap (macro, `optim/edit_count.py`).
- **Default text.** Every text field has a default (the neighbour's text, or a neutral line from the operator's spec), so programmatic strategies can apply any operator (section 8).

### The operators

| operator: what the proposer writes | what the system does (wiring) | extends today's | triviality notes |
|---|---|---|---|
| **insert**: `{"op":"insert","after":"X","kind":"agent","text":"..."}` (or `"before"`; `kind` `op` with `call`, or `code` with Python) | fresh id; own knobs; one output file named after the id; inputs wired automatically (data wiring below); output reaches a reader | `insert_after`/`insert_before` + node object | today about 1,400 characters of node JSON; target ≤ about 300 |
| **remove**: `{"op":"remove","step":"X"}` | reconnects predecessors to successors; takes X out of a fork's targets and the join's sources; folds a one-branch fork; drops X's files from optional reads; refuses with the exact reader if a required input would vanish | `bypass` | gates included: `remove` on a gate keeps one verdict's path (`"keep":"<label>"`), so it doubles as **unroute** |
| **parallelize**: `{"op":"parallelize","steps":["A","B"]}` (existing consecutive steps), or `{"op":"parallelize","after":"X","new":[{kind,text},...]}` (new steps) | puts the steps between a fork and a join when neither reads the other's output (refused with the dependency otherwise); keeps every reader's inputs | `parallel` (new steps only today) | from remote-factory `parallelize` |
| **serialize**: `{"op":"serialize","fork":"F"}` | runs the branches one after another in their declared order; removes the fork and join | none | from remote-factory `serialize` |
| **branch** / **unbranch**: `{"op":"branch","fork":"F","kind":"agent","text":"...","optional":true}`; `{"op":"remove","step":"S"}` for a branch | adds a step to an existing fork (targets, sources, edges); `optional` puts it in `optional_targets`; its output reaches a reader after the join | none today | the most-asked missing operator in the friction data |
| **rewire**: `{"op":"rewire","step":"X","sees":["task","<step id>",...]}` | rebuilds what X reads: `task` = the task's own inputs (facts with no model step upstream), a step id = that step's outputs. Delivered through `collect` (below). X keeps its id, outputs and knobs (internally a remove and re-add with the same id). Reports any knob that no longer reaches a model. | none | **question 1** (changes an existing step's inputs) |
| **loop**: `{"op":"loop","step":"X","judge":"agent","text":"<what to check>"}`; `judge` `code` with Python, or an existing callable | a gate after X: an agent reviewer with a verdict gate, a code gate, or an fn gate; `reloop` goes back to X with the reason as feedback; `proceed` goes on; `max_iterations` becomes a knob (default 2). Never an agent- or user-answered gate. | `loop` + hand-written gate and reviewer | today 2-3 edits plus file-to-fact plumbing |
| **route**: `{"op":"route","after":"X","judge":"agent","text":"<how to choose>","to":{"<label>":"<step id or {kind,text}>",...}}` | a gate after X; one labelled edge per branch; branches rejoin at X's old successor; refuses when a branch would leave a later step's required input unwritten | none (about 4 raw edits today) | |
| **split**: `{"op":"split","step":"X","into":[{kind,text},{kind,text}]}`; one node in `into` = **replace** | the first new step takes X's inputs, the second X's outputs; X's own knobs go; knobs only X read stay judged at the end of the bundle (fixes refusal 1) | `split` (full node objects today) | |
| **merge**: `{"op":"merge","steps":["A","B"],"text":"..."}` | one step with A's inputs and B's outputs; refused when another step reads A's outputs | none | |
| **configure**: `{"op":"configure","step":"X","setting":"timeout_s","value":1200}` | declares a bound knob `X__<setting>` with a spec generated from the field (type, bounds) and binds it; later tuned with plain `set_knob`. Allowed settings = `BINDABLE_FIELDS` only: knobs, not a free field edit. | `set_knob` (prompts, code, `max_iterations`, `parallelism` only today) | **question 3** for the safety-loosening settings |

### Thought goes into the design; the edit line stays one line

The triviality targets are about **mechanics only**: writing, wiring, validity and repair. Coming up with a good structure still takes the proposer's thought, and the plan asks for it explicitly.
- **Proposal format.** A structural proposal carries, beside its one-line edit, three required fields in the answer format (`answer_instructions`, `optim/base.py:1360`; wording coordinated with the framing PR):
  - `addresses`: which failure it targets, citing the per-step evidence by step id and item ids from the request;
  - `flow_change`: how it changes the information flow or the way the work is done (which step sees what, what runs beside or after what);
  - `prediction`: a check that can be made later: which steps' outputs or which items should change, and in what direction.
- A structural proposal missing these is not refused (the gate still judges it on score), but the trace marks it, and the replay counts it.
- **Predictions are checked.** After the candidate is evaluated, the prediction is compared with what happened on those steps and items. The result goes into "What the last edit changed" (`edit_change`, `optim/base.py:2270`), so the next proposal sees whether the reasoning held, not only the score.
- **Not a success measure:** shorter `considered` notes or less reasoning. A trivial edit line with weak reasoning is a failure of the quality check below.

### Automatic data wiring (the infrastructure that makes the lines short)

- **Inputs of a new step.**
  - Default: what its neighbour had, plus the neighbour's outputs: for `after X`, X's outputs and the task inputs; for a branch, the fork predecessor's.
  - The proposer may name others with `"sees"`.
  - Files reach a model step through **`collect`**: a plain Python op that reads named files (each optional), writes one fact with a header per file, and shows a long or binary file as its path and size. A missing upstream output (a failed optional branch, a model that ran out of output) does not crash the reader; today a required pasted read does (`agent_delivery.py:548`).
  - Code steps get the same files in `inp["files"]`.
- **Outputs reach a reader.** A new step's fact is placed where the next model step downstream fills facts: an agent template knob, or a knob read by a fills-facts op (V20). The proposer can name `"to":"<knob or step>"`. The candidate summary says where it went. Today the proposer hand-writes a `set_knob` patch with `{{fact}}` and gets V15/V20 refusals when it guesses wrong.
- **Naming.** Ids, file names (`<id>.txt` or the declared extension) and fact names are generated, collision-free, and stable for the same proposal on the same parent (so state ids deduplicate).
- **What each step can use.** The "What each node has" table gains a "made by" column and marks task inputs. A step's input prompt can no longer be mistaken for its output (the `{{compare_prompt}}` mistake in section 1).
- **Effect class.** Running a step beside a copy of itself, or rewiring a tool step, is allowed only when the step is safe to rerun (`retry_safe`, `nn/graph.py:93`: no tools, outputs only, or a fresh workspace per dispatch). Otherwise it is refused with the reason and the fix (give it a workspace). Copies keep the source's tool policy and workspace kind: **question 2**.

### Ordinary building blocks (optional)

`collect` (files to one fact) and `select` (copy a chosen branch's files to a step's output names, given a label in a verdict file or fact) are plain ops in `nn/operators.py`.
- The proposer may use them in any composition, or ignore them.
- No combiner is prescribed, and there are no vote or normalisation semantics in the core.
- A proposer that wants a vote writes it as a code step.

### Examples in the request (compositions, none preferred)

The request explains each operator in one line with one example. Then it shows a handful of compositions built **only from core operators**, with neutral placeholder ids, under the line "none is the preferred shape: compose your own":
- several attempts and a choice: `parallelize` new copies of `<step>`, then `insert` a judge after the join, then `insert` a `select` op;
- check and send back: `loop` `<step>` with an agent or code judge;
- an independent attempt: `branch` with `sees: ["task"]`;
- a simulation step then an analysis step: `insert` a code or op step, then an agent after it;
- a literature or background pass beside an experiment step: `parallelize` the two;
- iterative fitting: `loop` a fit step with a code judge on a residual threshold knob;
- a tool step before reasoning: `insert` a tool-using agent before `<step>` (where the run allows tools);
- dropping a step that adds cost: `remove`.

The examples rotate, and they are rendered from the current graph's ids, as `composition_section` does today (`optim/base.py:1147`), so no single shape reads as the house style.

## 3. Compared with remote-factory's mutations

`factory/outer_loop/mutations.py` (remote-factory main) has shape-level operators over edges: `insert_node` (before or after, with agent params inferred from a nearby node), `remove_node`, `redirect_edge`, `parallelize` (existing sequential nodes into fork/join), `serialize` (fork/join back into sequence), plus parameter, prompt and knob mutations. A deterministic `validate_and_repair` then:
- prunes unreachable nodes;
- refuses cycles with no gate condition;
- refuses joins with missing sources;
- refuses broken fork/join under a data node;
- **silently drops reads that nothing upstream writes**.

| | remote-factory | this plan |
|---|---|---|
| shape operators | insert, remove, redirect, parallelize, serialize | **keep all five** (redirect becomes `rewire` for data, raw `add_edge`/`remove_edge` for control); add loop, route/unroute, branch/unbranch, split/merge, configure |
| what is wired | edges only; reads/writes left to the node | **adds data-flow wiring**: inputs, file-to-fact delivery, outputs reaching a reader, facts and files and V15/V20 by construction |
| repair | deterministic, silent (a dropped read changes what a step sees without anyone knowing) | **keep deterministic repair, but only where the intent is unambiguous, and report every fix in the trace**; otherwise refuse with the reason and run the slim repair call that shows the refused proposal |
| node settings | `mutate_params` / `mutate_knob` free edits | settings only through bound knobs (`configure`), the standing rule |
| runtimes | one | both walker and spine, with tests on each |
| who may call it | the RF loop | every strategy (section 8), including programmatic ones through default texts |

## 4. Automatic repair and catching breakage before scoring

Today a refused proposal gets up to 2 more full-size calls (`attempts` = 3; `Loop.propose`, `loop.py:3497-3545`), each about 4-5 min. Each call sees the reason but not the refused edit (`optim/base.py:926-933`).

### Deterministic fixes

These run with no model call, only when the intent is unambiguous, and every one is reported in the trace and in the candidate summary ("repaired: ..."). They never silently change what a step sees, which is the difference from remote-factory's dropped reads.
- A slot that names a file instead of a fact (`{{notes}}` where `notes.txt` exists upstream and no fact of that name does) is wired through `collect`.
- A knob-orphan or other end-state rule is judged on the bundle's final state, not per edit (fixes refusal 1 in section 1).
- An edit that sets a knob to its current value inside a bundle is dropped. A patch whose target text is not found is refused, with that reason.
- An id collision is renamed.
- An operator target that names a step's output file instead of the step is resolved to the step, when exactly one step writes that file.

### Slim repair call

For anything else that is about mechanics (E-precondition, E-shape, E-node-text, E-binding, E-too-many-edits, V-rules, E-lowering; not E-integrity, E-noop or E-duplicate, which are about the idea):
- The call sends the refused proposal, the reason, the pipeline table, the "What each node has" table, the operator list and the node schemas: about 25-35k characters instead of about 280k, with no trajectories. The answer is the fixed proposal only.
- Bounds: at most 2 slim calls per refused proposal and 3 per epoch, about 1-1.5 min each (estimated). If both fail, the next full call runs as today.
- The trace gets `kind: repair`; the epoch line gets "refused N, repaired M".

The full retry request also shows the refused proposal from now on.

### Catch before scoring

- `E-gate-kind`: refuse any agent- or user-answered gate in runs with nobody to answer, and leave them out of the schema there. Today they pass and fail every item (section 1).
- **Code dry run:** run every new or changed code step in the sandbox on 2-3 of the parent's recorded per-item inputs (the run keeps `runs/<parent>/passes/<data>/<item>/`). Refuse on an exception, with the traceback. If there is no sandbox or no recorded inputs, skip and note it.
- **Wiring smoke**, optional and cheap: walk the candidate on one recorded item with model steps answered from the parent's recorded replies (the replay pattern), checking only that every step gets its declared inputs. This catches wrong-input wiring with no model call. Build it only if the replay shows silent wiring errors remain after the operators.

## 5. Both runtimes

Operators expand only into node kinds both runtimes run today: agents, plain ops (`python -m refactory_dsh.run <call>` on the spine), code ops and gates (`lowering.snippet_spec`), verdict and fn gates, forks (nested), gates routing to several labelled edges, and loops inside branches and data passes. The new settings that `configure` binds must be resolved by lowering exactly as `max_iterations` is (`lowering.py:327-329`), so the spine gets values and needs no TypeScript change. `_lowering_refusal` (`optim/base.py:144`) stays the last guard.

Tests:
- Fuzz (`tests/structure_fuzz.py`): the generator gains every operator, with default texts. The quick tier (walker) checks the invariants; the long tier compares walker and spine facts (`check_spine`, `:914`); mutation checks switch off each operator's wiring and must catch it.
- Named walker-vs-spine tests per operator on the synthetic seeds (`tests/structure_fuzz_graphs.py`), in the `_spine` drain pattern (`tests/test_livemath_blocks.py:771-789`), at parallelism 1 and 3. They cover:
  - `loop` with each judge kind;
  - `route` to three branches inside a data pass;
  - `branch` on an existing and nested fork, including an optional branch that fails;
  - `parallelize` and `serialize` round trip;
  - `rewire` to task inputs;
  - `split`, `merge` and `remove` of a branch and of a gate;
  - each new `configure` setting (walker value = lowered value).
- New synthetic seeds: one with a tool-using step in a workspace, one whose output is a non-text file. Today's seeds are text-only.

Still untested on either runtime (settle in these tests):
- a step reading and rewriting the same file;
- a fact produced twice on one path;
- `select` after a workspace step's file collection;
- a gate loop inside a branch inside a data pass.

**Runtime gap:** agent- and user-answered gates in headless runs (neither runtime; refused). A per-sub-item map is not in the DSL; see section 7.

## 6. Measurement and triviality targets

The user's bar: **a structural change must be trivial to write and wire**. The targets below are mechanics only; the design itself gets the proposer's thought, and is checked by the quality check after the table.

| # | target | how it is measured | baseline | delivered by |
|---|---|---|---|---|
| 1 | a structural change is one line of JSON in the common cases: **median structural edit size ≤ about 2× the median prompt patch** in the same requests | characters of each structural proposal's edits vs `set_knob` proposals | about 1,400 characters per new node; 700-4,500 per structural proposal | PR2 |
| 2 | **no hand-written wiring** for core operators: no edges, fork/join plumbing, file-to-fact ops or reader `set_knob` slots written by the proposer | share of structural proposals that contain any of these | most structural proposals today (each new agent comes with reader patches; forks are hand-edited) | PR2 (operators), PR1 (`collect`, reader placement) |
| 3 | **first-try validity ≥90%** of structural proposals before repair; **≥98% after automatic repair** | replay: validate each reply through `Run.attempt` on its parent | recent runs 2 of 3 first-try; older arms about 97 of 100, but by writing long careful JSON | PR2 (validity by construction), PR1 (repair) |
| 4 | **zero silent breakage**: no structural candidate that passes validation and then fails most items from wiring | replay: walk each accepted structural candidate on 3 recorded train items (walker, model replies replayed or a stub, so no new model spend) and count wiring failures; in runs, error kinds per structural candidate | 3 of 98 candidates failed nearly every item; ≥1 wrong-input wiring | PR1 (dry run, gate refusal), PR2 (wiring by construction) |
| 5 | **the request explains each core operator in one line with one example**, and every composition example uses only core operators | a unit test on the rendered request: one line + one example per operator; each composition example parses, uses only core ops, and applies cleanly on every synthetic seed | examples today use raw node objects and 2-3 edits | PR2 |

**Replay stages,** each on the same recorded requests: main, then the request PR (framing, in flight), then PR1, then PR2, then PR3. This separates the framing's effect from the mechanics'.

**Data:**
- LiveMath: 61 calls from round3/3a/3b plus 28 older B-arm calls.
- **Non-LiveMath:** the Countdown runs behind `~/work/hack-judge-replay` (12 runs, 66 proposals: 42 prompt, 21 code, 3 structure). Their extract `raw/countdown_pc.jsonl.gz` has the prompts removed, so the replay re-reads the requests from the lab run dirs that `scripts/extract_lab.py` lists. Also the circle-packing run (11 proposals).
- Each new dataset's requests are added as soon as it has run (tool-using, file-output, retrieval).
- 2 samples per request on claude-opus-4-6 through the lab proxy. No test items are used.

**Also reported:** structural proposal rate (no target; structure is not forced); the share of `considered` notes that give the mechanics as the reason (about 12/61 today, target about 0); time per refused proposal (+4.6 min today, target ≤3 min).

**Quality check in the replay (the design, not the mechanics):**
- the share of structural proposals whose `addresses` cites per-step evidence (step ids and item ids that exist in that request); target ≥90%;
- the share whose `flow_change` names a concrete change in what a step sees or does; judged by a rubric read on a sample;
- for the evaluated candidates (score sanity check and runs), prediction hit rate: did the predicted steps or items move in the predicted direction? Reported per dataset, with no target at first.
- A replay where mechanics targets are met but these fall is a regression.

**Score sanity check:** walker-evaluate the top 6-8 replayed structural candidates per dataset against their parent on train plus val, 1 seed. Not evidence for changing defaults (that needs ≥3 datasets and ≥4 seeds).

**In runs:** the next outer-loop run on any dataset tracks the same five targets from `journal.jsonl` and the run log, plus timeouts added by new steps and val-gate acceptance of structural vs knob candidates, with the usual early bug sweeps.

## 7. DSL coverage

Enumerated from `nn/graph.py`: `NODE_KINDS` (op, agent, gate, fork, join, data), `GATE_KINDS` (fn, user, agent), `ON_EXHAUSTED`, `RETRY_ON`, `WORKSPACE_KINDS`, `BINDABLE_FIELDS`, and every `Node` and `Edge` field. Also the composition patterns both runtimes run.

The standing rule holds: an existing node changes only through knobs (bound fields) and topology. Today anything else on an existing node can be reached only by removing and re-adding it (3-6 raw edits, and it breaks fork membership): effectively the removed `replace_node`.

| construct | today: create / change / remove | after the plan | gap left |
|---|---|---|---|
| agent, op, code step | insert via node object / prompt and code knobs / `bypass` | `insert` one-liner / knobs / `remove`; `split` with one node = replace (a different callable) | none |
| fn gate (code, verdict, callable) and `max_iterations` | `loop` + hand-written gate and reviewer / `__code` knob / no move | `loop`, `route` / knobs, `configure max_iterations` / `remove` with `keep` (unroute) | none |
| agent- or user-answered gate | offered, fails headless | refused in headless runs (`E-gate-kind`) | excluded by design |
| edges, labels, `halt` | raw `add_edge`/`remove_edge` | raw ops; `route`/`loop` create labels with their judge's instructions | none |
| fork / join, targets / sources | `parallel` (new steps only) / none / none | `parallelize` (existing or new), `serialize`, `branch`, `remove` of a branch | none |
| `optional_targets` | raw new fork only | `branch ... "optional": true`; `configure` a branch's `optional` | none |
| nested forks, loops inside branches, routing to different branches | raw only | operators compose them; fuzz + named tests | none |
| `data` node (source, limit, parallelism) | parallelism knob | unchanged: grader-side and protected | excluded by design |
| map: a step once per sub-item an earlier step produced | not in the DSL. The spine has director expansion (`workflow-spine/src/types.ts:9`, `spine.ts:1296`) that neither `nn.Node` nor the walker has. A nested `data` node with an upstream `source_path` is accepted by the walker's loader (`nn/executor.py:550-575`), but V6 does not see that dependency and no test covers it. | spike only | **question 4** |
| `reads` / `required_reads` / `inputs` / `produces` / `writes` | new nodes only | inputs via `rewire` and automatic wiring; outputs fixed by contract | `rewire` waits on question 1 |
| prompts, system prompts, code | knobs | knobs | none |
| `timeout_s`, `retry` (on/times/factor), `on_exhausted` | new nodes only | `configure` (bound knobs; `retry` as three sub-knobs; `timeout_s` never below the route's default; `on_exhausted` without `halt`) | none |
| `workspace`, `tools`/`tool_policy`, `outputs_only`, `role` | new nodes only (tools refused unless allowed) | `configure`, only after your decision | **question 3** (and question 2 for copies) |
| `feedback_keys` | new agents only | raw on new nodes only | deliberate: an advanced runtime contract |
| `blocking`, `score_effect`, `notes`, `resource_role`, protected nodes' `call`/`evaluator` | — | — | excluded by design: lowering or infrastructure flags with no search value, or grader-side |
| merge two steps | none | `merge` | none |

### Coverage test

`tests/test_dsl_coverage.py` reads one map, `nn/coverage.py`.
- **Generated.** Every entry of `NODE_KINDS`, `GATE_KINDS` and `BINDABLE_FIELDS`, every `dataclasses.fields(Node)`, every edge form and every composition pattern must have an entry: the operators or raw ops that create, change and remove it, or `EXCLUDED` with a reason from a fixed list (grader-side, no headless answerer, infrastructure flag, awaiting a user decision, planned). A new DSL feature without an operator or an explicit exclusion fails the test.
- **Behavioural.** Each covered entry carries a generated one-line example. It is applied through `structure_fuzz.Applier` (the outer loop's attempt path) on each synthetic seed it fits, and the construct must appear or change in a state that validates and lowers. The fuzz generator draws from the same examples, so the long tier checks walker and spine on them.
- **Docs.** A coverage table in `docs/` is generated from the map and checked to match, as the rules text is.

## 8. Any outer loop

Every strategy has to get the operators, their validation, repair and the structural evidence: the built-in ones in `refactory_dsh/strategies.py` (default `Loop`, BestOfN, Reflective, ReflectiveSampler, AdaptiveMeta, TreeSearch, StagedCurriculum, ResearchThenOptimize, Autoscientists) and package strategies (`package_strategies`/`register`, `strategies.py:853,958`; e.g. `refactory_packages/skillopt`). The design does this by putting each piece where every strategy's proposals already pass.

### Where each piece lives

| piece | shared path it lives in | why every strategy gets it |
|---|---|---|
| operators, preconditions, expansion, `configure` specs, coverage map | `nn/edits.py` (catalogue, `apply`, `precheck`), `nn/operators.py`, `nn/coverage.py` | every candidate of every strategy is made by `Run.attempt` (`outerloop.py:116`), which calls `Optimizer.attempt` (`optim/base.py:2836`), which calls `nn_edits.apply`. This includes BestOfN's and Autoscientists' direct edits (`strategies.py:256,764,778`), the default `install` (`outerloop.py:847-855`) and SkillOpt's (`refactory_packages/skillopt/loop.py:727,802`, `proposer.py:280-344`). |
| validation and the new refusals (`E-gate-kind`, code dry run, end-state checks) | `Optimizer.attempt` + `nn/validate.py` | the same single path |
| structural evidence, the operators in the request, "made by" / "(task input)", the refused proposal shown back | `ProposalContext` and its `render` (`optim/base.py`), filled by `Loop._context` (`loop.py:5128`) | every model-written proposal of a built-in strategy goes through `optimizer.propose(ctx, ...)`: the default `Loop.propose` (`loop.py:3417`), and `_sample_from_optimizer` for BestOfN and Autoscientists (`strategies.py:313`). Reflective, ReflectiveSampler, AdaptiveMeta and TreeSearch call `super().propose`. `edit_kind_decision` (`strategies.py:130`) already reaches the request as the round's decision. |
| slim repair call | the model proposer (`AutoresearchOptimizer`, `optim/random_optimizer.py:940`), triggered from the retry loops in `Loop.propose` (`loop.py:3497-3545`) and `Optimizer.propose_valid` (`optim/base.py:2752`) | the strategy's loop owns *when* to retry; the proposer owns *how* to ask. Every strategy that retries gets slim repair, and any `Optimizer` subclass can override it. |

### Strategies that bypass a path today, and what they need

- **Strategies that write edits themselves:**
  - BestOfN without a model (`_variant`, `strategies.py:269-285`: one random `set_knob`);
  - Reflective's GEPA merge (`strategies.py:417`: knob values from a second parent; falls back when the parents differ in structure);
  - Autoscientists' merge (`strategies.py:801`);
  - the random optimizer (`nn_edits.random_edit`, `nn/edits.py:1739`).

  They pass validation (good), but they never emit structure, and the random sampler needs a mode's `insertable` blocks to build any node. They need operator sampling (next subsection) and, for the merges, the structural merge seam below.
- **SkillOpt** (package): has its own prompts and is text-only by design. It gets validation through `run.attempt` automatically. It never sees the operators, and should not: it declares itself text-only (`edit_preset="knob_only"` or a class flag) and the coverage test accepts that declaration explicitly instead of silently.
- **Any future package strategy with its own prompt:** it gets the operators' text from one shared renderer (`composition_section` + the catalogue table, made callable on its own), so it does not re-describe the DSL.

### Operators without a model (programmatic and evolutionary strategies)

Every operator must apply with no free text. Each text field defaults to the neighbour's text or to a neutral line in the operator's spec, and each value to one sampled from the generated knob spec (`Knob.sample`):
- `insert` and `branch` default to a copy of the neighbour's kind and text;
- `loop` defaults to the neutral check line;
- `route` picks existing successors as branches;
- `parallelize`, `serialize`, `remove`, `split`, `merge` and `rewire` need no text beyond their defaults.

`random_edit` gains an operator branch: pick an operator valid at a random node (its precondition passes), fill it with defaults or values sampled from the generated knob specs, and apply it. Parallel edits stop being excluded (`_NOT_SAMPLED`, `nn/edits.py:1736`). An evolutionary strategy then mutates texts afterwards with plain `set_knob` on the knobs the operator created. Text can also be sampled from existing knobs: a copy's prompt drawn from another node's current value is allowed, as long as it lands in its own knob (E-binding still holds).

### Test: every strategy can take an operator

`tests/test_strategy_operators.py`, parametrised over `strategies.available()`, which includes registered package strategies.
1. Build each strategy on a synthetic seed with a scripted model (and the random sampler).
2. Hand it one proposal per operator through `run.attempt`; assert it becomes a candidate whose state contains the operator's expansion.
3. For strategies whose optimizer `uses_model`, capture the rendered request and assert that it lists every core operator and the composition section. A strategy that declares itself text-only (SkillOpt) must say so explicitly, and the test asserts the declaration instead.
4. For non-model strategies, assert that `random_edit` reaches every operator within a seeded budget (the fuzz generator's "reaches every operation" test, `tests/test_structure_fuzz.py:64`, extended to operators).

### Seams the GEPA-style strategy will need

These come from the validation-selection plan's pool.
- **Structural merge.** Today's merges cross knob values only. GEPA's merge takes one parent's structure and another's settings, or both parents' modules. The seam is `nn_edits.rebase(edits, onto)`: replay parent B's own edits since the common ancestor (`Candidate.edits`, journaled) onto parent A through `Run.attempt`, refusing cleanly on conflict. Operators make this practical: they are few, deterministic and named by node id, unlike a long list of raw edges.
- **Per-operator provenance.** Each candidate records which operators made it (from `edit_count`), so a pool or Pareto front can be analysed by kind of change, and `edit_kind_decision` can use operator kinds instead of only "settings vs pipeline".
- **Deterministic expansion.** The same operator on the same parent gives the same state id (`candidate_id`), so a pool deduplicates across strategies and resumes.
- **Per-item scores for every structural candidate** stay on the shared evaluate path, so Pareto selection over items needs nothing operator-specific.


## 9. PRs (3, plus a lab replay script that stays in ~/experiments)

| PR | scope | main files | tests | risks | size |
|---|---|---|---|---|---|
| **PR1: the infrastructure** (build first) | All of it in shared paths, so every strategy gets it:<br>• data wiring helpers: `collect`, `select`, task-input marks, "made by" column, reader placement;<br>• deterministic fixes, reported in the trace;<br>• retry shows the refused proposal; slim repair call with bounds;<br>• catch before scoring: `E-gate-kind`, code dry run, end-state checks (the split fix);<br>• `nn/coverage.py` map + `test_dsl_coverage.py` with today's gaps marked "planned", so nothing new slips in;<br>• the stale "cannot read a file" line (with the framing PR). | `optim/base.py`, `nn/edits.py`, `nn/node_schema.py`, `nn/validate.py`, new `nn/operators.py` (collect, select), `optim/random_optimizer.py`, `loop.py`, `snippet.py`, `nn/coverage.py` | one unit test per fix and refusal on synthetic graphs; the three recorded failures as regression fixtures; a scripted-model repair test; fuzz quick tier unchanged | dry run needs the parent's per-item dirs (skip and note when pruned); slim repair may miss context (falls back to full) | about 1,000-1,300 lines incl. tests |
| **PR2: the core operators** | `insert`, `remove` (incl. unroute), `parallelize`, `serialize`, `branch`, `loop`, `route`, `split` (incl. replace), `merge`, `rewire` (if question 1 is yes):<br>• each a macro over existing primitives, with default texts;<br>• the one-line-per-operator request text and composition examples built only from operators;<br>• the `addresses` / `flow_change` / `prediction` fields and the prediction check in "What the last edit changed";<br>• edit-count entries; Behaviour changes section (operators appear in every structural run's catalogue; `knob_only` and `conservative` presets unchanged);<br>• coverage map entries flipped from "planned" to covered. | `nn/edits.py`, `nn/operators.py`, `optim/edit_count.py`, `optim/base.py`, docs | fuzz generator + mutation checks per operator; named walker-vs-spine tests per operator (section 5); request render test (target 5); new tool/workspace and file-output seeds | the four untested runtime behaviours (section 5); the effect-class rule may refuse what an environment could support | about 1,500-2,000 lines incl. tests |
| **PR3: settings and any outer loop** | `configure` for the safe settings (`timeout_s`, `retry`, `on_exhausted`, `max_iterations`, branch `optional`), plus any you approve (question 3):<br>• `BINDABLE_FIELDS` extended; lowering resolving each;<br>• operator sampling in `random_edit`; text-only declaration (SkillOpt); `test_strategy_operators.py`;<br>• `rebase` seam stub for the GEPA-style strategy;<br>• the per-sub-item spike as a test-only branch (question 4);<br>• the coverage test made strict (no "planned" left). | `nn/graph.py`, `lowering.py`, `nn/edits.py`, `nn/coverage.py`, `strategies.py` (declarations only), docs | each new bound setting: walker value = lowered value = spine behaviour; every registered strategy takes every operator; coverage test strict | binding `retry` and `timeout_s` touches `node_retry`: test both runtimes | about 900-1,300 lines incl. tests |

**Which PR delivers which target:**

| target | PR |
|---|---|
| 1 (edit size) | PR2 |
| 2 (no hand wiring) | PR1 (wiring helpers) + PR2 (operators) |
| 3 (validity) | PR2 (by construction) + PR1 (repair) |
| 4 (no silent breakage) | PR1 (dry run, gate refusal) + PR2 (wiring by construction) |
| 5 (request text) | PR2 |
| quality check | PR2 (fields, prediction check) |

**Order:**
1. PR1, then the replay (main, request PR, PR1);
2. PR2, then the replay;
3. PR3;
4. one real run.

PR1 helps raw edits even before PR2, and its traps exist today.

## Open questions for the user

1. **Rewire an existing step's inputs.** May `rewire` change what an existing step reads (internally a remove and re-add with the same id and knobs)? Recommendation: yes. It is a data-flow change, not a node-field edit, and it was the most-wanted change in the friction data.
2. **Tools for copies.** May a copy of a tool-using step (`parallelize` new copies, `branch` with a tool step) keep its source's tools, if it gets its own workspace? Today any new node with tools is refused unless the run allows it. Recommendation: yes, judged as its source.
3. **Which settings `configure` may change.** The safe set needs no decision. For the safety-loosening ones, recommendation:
   - `workspace` to `scratch`/`clone`: yes;
   - `role`, among the run's declared routes: yes;
   - `tools`/`tool_policy` and `outputs_only`: no, for now.

   Excluded in the coverage map until you decide.
4. **Per-sub-item experiment.** Spend a spike (nested data node fed by an upstream file, on both runtimes) before any map operator? Recommendation: yes, inside PR3. The spine's director expansion has no walker counterpart either way.
5. **Build route** (carried from before). PR1 hand-made; PR2 and PR3 by design run or hand-made?
