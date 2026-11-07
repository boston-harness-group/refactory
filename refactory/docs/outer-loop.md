# Outer-loop recovery

## WalkerEvaluator

`WalkerEvaluator` enables `enable_recovery` by default on every `mode.evaluate()`
call.  This means:

- **Op retry.** A transient failure in an FnNode op is retried once with
  fact restore before it is recorded as a failure.
- **Per-op timeout.** Every op and fn gate runs in a child process forked from
  a preloaded forkserver, with a timeout derived from `command_timeout_s()`
  (the same formula lowering uses), clamped to the remaining budget with a 1 s
  floor.  The forkserver preloads the graph's op and gate modules, so per-op
  overhead is ~0.01 s instead of the ~0.28 s a fresh spawn costs.
- **Clean trial boundaries.** The child process is killed at the limit
  via session-grouped SIGTERM→SIGKILL escalation.  No leaked processes or
  state survive between candidates.

## DshEvaluator

`DshEvaluator` runs candidates through the DSH spine, which enforces its own
`command_timeout_s` on each node's subprocess.  It does not use
`enable_recovery`.

## The default loop's validation gate (`--val-gate 1`)

With `--val-gate 1` and a parts manifest with a val part, the default loop evaluates
candidates the way the SkillOpt strategy does, using the same `ValidationGate` and
`FinalEvaluation` from `outerloop.py`. Only the optimizer, where edit ideas come from,
differs. Without `--val-gate` nothing below applies.

It follows SkillOpt's single line, where one current candidate is the only parent, so
it refuses, before anything is rolled out, the optimizers that breed from a population
of parents (`archive`, `evolutionary`, `islands`, `map_elites`) and the strategies that
merge parents or pick among sibling samples outside the gate (`autoscientists`,
`best_of_n`, `reflective`). Run those without `--val-gate`. The `random`, `autoresearch`
and `hill_climb` optimizers run with it.

- **Train seed.** Train rollouts in epoch `e` use SkillOpt's seed, `seed + e*1000 + 1`
  (`outerloop.train_batch_seed`). Val and test rollouts use `seed`.
- **Gate.** Every candidate evaluated on train is rolled out on val, including one whose
  train run failed, and is accepted only when its val score strictly beats the gate's
  current candidate (the last one accepted), not its parent. With `--batch-size` above 1
  the decisions depend on the batch's order: a candidate accepted earlier in the batch
  raises the bar for the ones after it. SkillOpt gates one candidate per epoch and has
  no equivalent.
- **Parent and winner.** As in SkillOpt, the gate's current candidate (also the best on
  val) is the parent every optimizer edits, with any accepted candidate a person kept,
  whether or not its train run succeeded. The leaderboard lists the accepted candidates a
  person has not rejected, ranked by val score. Its first is the winner: the report, the
  held-out audit and `adopt` all name it, and the report's `best_score` is its val score
  (`best_score_basis`), with its train score in `best_train_score`.
- **Last epoch.** The batch proposed in the last epoch is evaluated on train (with the
  next epoch's seed) and gated before the run ends, so N epochs gate N proposed batches.
- **What the optimizer reads.** Every row it sees (recent results, the history, the
  gradient's leaderboard) carries the gate's decision: `current` (the candidate it
  edits), `kept`, `rejected` or `not gated`. The per-knob attribution counts, for each
  value, how many of its candidates were kept, rejected or not yet gated. The request
  says it edits the `current` candidate, and that a `rejected` one did not generalise.
  The history's best is the best kept candidate. Val scores are never shown to it;
  scores there are train scores. That includes "Previous runs of this mode": a gated
  sibling run's report gives its winner's val score as `best_score`, so the summary
  carries that run's `best_train_score` instead.
- **What a person reads.** Each epoch's progress line says how many candidates the
  gate kept and rejected that epoch, with the val best and current scores, instead of
  a train-score "improved".
- **Final evaluation.** After a normal end (the epoch bound or an early stop), with a
  test part, the final candidate's val check runs, then the seed, the best and the final
  candidate on test with `seed`. A run that ends on a spent evaluation, rollout, time or
  token budget, or a person's stop, scores neither the last proposals nor the test part,
  as SkillOpt; the report keeps the real stop reason (a spent budget wins over an early
  stop in the same epoch) and says why it skipped. The report's `strategy_report`
  holds `final_selection`, `baseline_test`, `best_test`, `final_test` and the two deltas
  (`test_eval` says `done`, `skipped` or `incomplete`, and `test_eval_reason` why). The
  held-out audit then compares the winner's val and test scores, so a `--holdout 0` run
  is audited. Each phase is saved: a resumed run continues it. Without a test part it is
  skipped with that reason.
- **Overfit audit.** The val-vs-test audit counts only the gap the search added:
  (winner val − winner test) − (seed val − seed test), using the seed's val score and its
  test score (`baseline_test`, which the final evaluation always runs when there is a
  test part). That is compared with a fixed 0.15. The gap's standard error,
  sqrt(v(1−v)/n_val + t(1−t)/n_test), is reported beside it for information and does
  not move the threshold.
  - **Seed winner:** never overfit; the note says why.
  - **No seed test score** (a skipped or incomplete final evaluation): the audit is
    `unmeasured`. It neither passes nor fails the run, and `adopt` refuses the winner
    until the audit is measured.
  - **Recorded fields:** `holdout.json` and the report give `gap`, `seed_gap`,
    `added_gap`, `threshold` and `standard_error`.
  - **Train vs holdout:** that audit uses the same fixed threshold and the same seed
    rule; it has no seed holdout score to subtract, so an optimized winner's whole gap
    counts.
- **Early stop.** `--early-stop-patience N` (0, off, by default) stops the run after N
  epochs in a row whose gating did not raise the best val score.
  `--early-stop-at-full-val 1` stops it once the best val score reaches 1.0. The first
  epoch gates the starting population, which sets the best the count starts from
  (SkillOpt's initial val score) and is never a reason to stop. Each later epoch gates
  the batch the previous one proposed, and is checked right after that gating, before
  it proposes, as SkillOpt checks at an epoch's end; so a seed already at 1.0 still gets
  one batch proposed and gated. Both settings need `--val-gate 1`, are saved with the
  run and are kept on resume.

## Train minibatches (`--train-minibatch N`)

By default every candidate is scored on the whole train part. With
`--train-minibatch N` and a train part of more than N items, each epoch samples N train
ids with SkillOpt's per-epoch seed (`train_batch_seed`). It needs `--val-gate 1`:
without the gate the winner and the parents are picked by train scores, which would
then come from different samples, so the run is refused before anything is rolled
out.

- **One sample per epoch.** Every candidate scored in that epoch uses the same sample,
  so candidates in a round are comparable. The sample changes from epoch to epoch, and
  each epoch's ids are journaled (`train_minibatch`).
- **The evidence is that minibatch.** When the candidate the optimizer edits was scored
  in an earlier epoch, it is scored again on this epoch's minibatch, so the proposal
  evidence is that minibatch, as SkillOpt rolls out its current candidate on each
  step's batch. Its score is left alone; only the records the optimizer reads change.
  This re-score is an evaluation like any other: it counts toward the evaluation,
  rollout, time and token budgets, and it is journaled (`rescore`).
- **Resume.** A resumed run samples the same ids. Resuming with a different N is
  refused.
- **Unaffected.** Val and test are not sampled.

`0`, or a train part of N items or fewer, scores the whole train part as before.

## What the optimizer sees, and what it said

`ProposalContext.render()` is the request every model optimizer reads. Besides the
counts and the error-cause and wrong-answer tables for the candidate being edited, it
shows **every** train item of that candidate in full, in every proposal request: all
failures and all successes, never a sample. Each item shows its input (the item metadata's `question`,
or `problem`, `task_description`, `instruction`, `input` or `prompt`, else the first
step's prompt) and its `choices`. Then it shows each step's output from the record's
`exchanges`, in the order the pass ran them (a fork's branches in branch order), then
the expected answer (`label` and `label_text`), the prediction and the grader's reason.

- **Order.** Items answered wrong come first, then those that failed with an error
  (with their error, whether or not any step ran; they are in the cause table too),
  then those that passed, each group by id. With repeats each item is shown once, from
  its first repeat.
- **Full detail, one call's budget.** Items are shown whole, every step output
  unclipped. One call's evidence budget is 120000 characters (`TRAJECTORY_BUDGET`;
  the context's `evidence_budget`). When every item fits, they all go inline into the
  one proposal call.
- **Readers, when they do not fit.** The items are split into as few balanced groups
  of whole items as fit the budget, in the same order, each item in exactly one group.
  The `autoresearch` optimizer sends each group to a reader call: the same model and
  route, prompt in and JSON out, no edits. A reader returns failure and success
  patterns, each with its item ids, short verbatim quotes and the step it implicates.
  Reader calls run concurrently (`READER_CONCURRENCY`, 4) and are recorded in
  `optimizer_trace.jsonl` with `kind: "reader"`. Their findings are kept in
  `<run root>/reader_findings/`, keyed by the group's whole text (the edited
  candidate's records for exactly those items), so a retry after a refusal and a
  resumed run reuse them instead of paying again.
- **No tools.** Every optimizer call, the proposal and each reader, is made with
  `tool_policy="none"`: prompt in, JSON out, no tools in a working directory beside
  the run's records. This is the same mechanism SkillOpt's calls use; the run's own
  client keeps its tools, and a token-limited run's budget guard stays in place. Only
  the `dsh` and `claude_code` backends can run a call without tools, so a run whose
  optimizer role is routed elsewhere (`codex`, `openshell`) is refused at launch,
  before anything runs. The message says how to route the role, for example
  `--role-routes '{"optimizer": {"worker_backend": "claude_code"}}'`. It never runs
  the optimizer with tools.
- **What the proposal call then gets.** The readers' findings, verbatim but bounded:
  at most 12 patterns per reader, each pattern's text at most 400 characters, at most 3
  quotes of at most 240 characters and 40 item ids, and the whole findings block at
  most 40% of the budget (`READER_*`). Then a
  table of every item: outcome, expected, predicted, the grader's reason, the error.
  Then as many items in full as the rest of the budget holds, failures and passes
  alternating. A reader call that fails is noted, with its items, and the proposal
  goes ahead with the findings it has. An optimizer that calls no reader gets the
  table and the items in full.
- **Clipping.** Only an item longer than the whole budget on its own is clipped, head
  and tail, with a note.
- **Size.** For `livemath/skill_graph` with 35 LiveMath-sized train items (about 528k
  characters in full): 5 reader calls of 7 items (about 106.5k characters each), then
  one proposal call of about 135k characters (about 34k tokens): the trajectory section
  is about 105k and the pipeline with its node specs about 18k. That is 6 calls.

"Evidence from the traces" shows the gradient without the noise. The leaderboard copy
is already "Recent results". Per-trial worker receipts and run-directory paths
(`measurements`, `workers`, `workdir`, `trials`) are left out wherever they appear.
Long keys and texts, such as a text knob's value in `knob_attribution`, are shown by
their head and length. A `measurement_summary` gives the trial runs, failures,
timeouts and the top failure causes instead.
- **Train only.** The records are the edited candidate's train records (the loop
  never hands val or test records to an optimizer).

The pipeline section also shows each node of the edited graph **in full**, as a node
object that `add_node`, `insert_after`, `insert_before`, `wrap`, `split` and `loop` take.
A model copies one, gives it a new id, changes its prompt, reads, writes, produces or
knobs, and uses it verbatim.

- **Bound fields.** A field bound to a knob shows the knob's current value, since the
  stored field is not what runs, and says to edit the knob instead. The copies leave out
  `field_bindings` and the bound knobs.
- **Clipping.** Long texts keep their head and tail within 1200 characters
  (`NODE_SPEC_TEXT_CHARS`). A clipped text must not be copied, and the prompt says so.
- **Size.** The section, nodes and schemas together, fits 30000 characters
  (`NODE_SPEC_SECTION_CHARS`). First the texts shrink, then nodes are left out, last
  first, and named.

The edit layer enforces the same for every edit that adds a node (`add_node`,
`insert_after`, `insert_before`, `wrap`, `split`, `loop`):

- **No shared knobs.** A new node may not bind a knob another node binds or reads
  (`E-binding`), so a copy cannot silently run the original's prompt.
- **Own prompt knob.** A new agent node gets its own `<id>__system_prompt` knob holding
  its `system_prompt_template`, as a mode's agent nodes get one when it is built.
- **Text checks.** A node text carrying the clip marker
  (`[... N characters omitted ...]`), or a `{{` never closed by `}}`, is refused
  (`E-node-text`). JSON braces in a prompt are fine.
- **Removing a node.** A removed node takes its own knobs with it: the ones it binds
  that carry its id (`<id>__...`) and no other node reads. The state then still passes
  V7, and the same id can be added again. A declared knob that only the removed node
  read stays declared, so the validator refuses the state, and setting a knob no node
  reads is refused before evaluation (`E-knob`).
- **Duplicates.** `duplicate` gives the copy its own knob for every field the original
  binds, declared with the original knob's spec and current value.

After the nodes come the **node schemas**, generated by `refactory_dsh.nn.node_schema`
from the `Node` dataclass and its validator, so they cannot drift:

- **Field table.** Every field with its type, default, the node kinds that take it and
  its one-line meaning (the field's `#:` comment).
- **Per kind.** For each kind (`op`, `agent`, `fork`, `join`, `data`, and `gate` once
  per gate kind `fn`, `user`, `agent`): the fields the validator requires and the ones
  it refuses, found by building nodes, and a minimal valid example. A kind whose
  refusal names no field says "see the validator" instead of breaking the request.

`call` and `evaluator` must name an importable `module:function` (rule V8), so a new op
or fn-gate node can only reuse a callable an existing node already names. In the edit
catalogue, every field that takes a node (`node`, `gate`, and `into` for two) says so.

Two more sections carry what SkillOpt's slow update and carried-forward notes give its
optimizer, at no extra compute (train records only):

- **What the last edit changed.** The latest children of the edited candidate (its last
  batch), or the edited candidate itself when it has none yet, are compared with their
  parent item by item, on the train items both were scored on. It shows how many items
  were fixed (parent wrong, child right) and broken (the reverse), and the largest score
  rises and drops among the rest. Each changed item is shown with both trajectories, as
  far as 60000 characters hold (`EDIT_CHANGE_BUDGET`); the rest go in a table. It also
  shows the edit that made the child and the gate's decision on it, and warns that the
  two were scored in different rounds with different seeds, so some changes may be
  noise. Without a parent (the seed) or without shared items, it says so.
- **Your earlier reasoning and what happened.** Up to 6 recent candidates: the line
  from the seed to the edited candidate, and the latest rejected children. Each shows
  the rationale and prediction the optimizer gave when proposing it, the round's
  diagnosis, the gate's decision, its train score and its fixed and broken counts
  against its parent. Texts are clipped to 600 characters. Each proposal keeps the
  diagnosis of the call that produced it, and the proposals' own words are journaled,
  so a resumed run keeps them.

A proposal refused as a duplicate (`E-duplicate`) is told which candidate it repeats
(its id), so the retry can look it up and move on. This applies with or without the
gate.

Every call the `autoresearch` optimizer makes is appended to
`<run root>/optimizer_trace.jsonl` as it happens. Each line records the epoch, the
generation, the proposal round, the system prompt, the prompt, the reply, how many
proposals were parsed, the diagnosis, and the error for a call that failed. A resumed
run appends to the file; a fresh run in the same directory starts it over, as it does
the journal. A trace line that cannot be written (a full disk) is a warning; the
proposals are kept. The other library optimizers call no model; SkillOpt records its own
calls in `skillopt_trace.jsonl`.

## LiveMath skill graph (`livemath/skill_graph`)

Arm C of the outer-loop study starts from arm A's seed skill, converted by hand into an
explicit graph. Inside LiveMath's per-item data pass, `compare` analyses every option;
`precision` and `hypotheses` then run in parallel on compare's notes; `decide` reads all
three sets of notes and answers in the baseline solver's format. LiveMath's unchanged
grader scores decide's answer. Each agent runs the solver's system prompt with only its
own seed-skill section, and the step task texts carry no method guidance. Only `decide`
answers: in the three analysis steps the solver's answer format is replaced by an output
format asking for notes and no answer, so no analysis step commits to an option that
decide then follows.

Each step has two open text knobs, one per piece of text it sends:
`<step>__system_prompt` (`compare__system_prompt`, ...) is the step's whole system prompt,
seeded with the solver's system prompt and that step's section, and `<step>_task`
(`compare_task`, ...) is the task text ending its user message. The question and the
earlier steps' notes are data, not knobs. With the defaults decide's system prompt is
the solver's, byte for byte. `max_completion_tokens`, `data_parallelism`,
`use_theorem` and `use_sketch` are frozen. An item's theorem states the correct option,
so the last two stay off. Each item takes 4 model calls.

`data`, `extract_answer` and `finish` are protected (`SkillGraph.protected_nodes`): a
search edits the steps between them, never the pass, the answer extraction or the run's
end. See "Structure edits" below.

## SkillOpt over several text knobs

`--strategy skillopt --strategy-option knobs=compare__system_prompt,precision__system_prompt,hypotheses__system_prompt,decide__system_prompt`
runs SkillOpt's procedure on all four system prompts at once (arm C1); add the `_task`
knobs to the list to let it edit the task texts too. Without `knobs`,
the strategy optimizes the single `skill` knob exactly as before.

- **One document.** The analysts, the merges, the ranking, the slow update and the meta
  skill all see one document. A header names the graph's agent steps in run order. Each
  knob's text then follows, in `knobs` order, under a
  `<!-- SECTION <knob>: <description> -->` line. Each step runs once per epoch, so call
  keys and trace entries stay unique.
- **Back to the knobs.** The ranked patch becomes one `set_knob` patch per knob it
  touches, each applied through `Run.attempt`:
  - An edit may name its section with `"section": "<knob>"`. The header, and a note
    added to the analyst, merge and rank system prompts, tell the model to do so
    whenever its target text appears in more than one section, to keep an edit's
    section through the merges, and that edits in different sections are different
    text regions; the ranker sees each edit's section. A named
    edit applies to that knob only: it is refused (`skipped_target_not_in_section`) when
    its target is elsewhere in the document, and an append or a target found nowhere
    lands at the end of that knob. A section that names no knob is refused
    (`skipped_unknown_section`). The section field is not passed on to the knob's patch.
  - An unnamed edit goes to the one knob whose text holds its target. When several
    knobs hold it (the four skill_graph system prompts share most of their text), the
    edit is refused (`skipped_ambiguous_section`) and its outcome names the candidate
    sections; there is no first match.
  - An unnamed append, or a target found nowhere, goes to the last knob. That knob
    ends the document, so the edit lands where SkillOpt would put it. An `insert_after`
    whose target is only in the header or a SECTION line counts as not found, so
    SkillOpt's rule appends it.
  - An edit is refused on its own, never split, when its target lies across a section
    boundary, in the header or in a SECTION line, or when its content carries a SECTION
    line. The trace reports it as `skipped_cross_section` or
    `skipped_section_line_in_content`, and the rest of the patch applies.
- **Refused edits take no rank slot.** Before ranking, the merged edits are routed
  against the current knobs and the refused ones are dropped (trace
  `section_refused`), so the edit budget goes to edits that can apply.
- **One knob is unchanged.** With a single knob there is no header, no section
  note and no section routing: arm B's optimizer prompts are byte-identical to before.
- **The slow update.** The SLOW_UPDATE placeholder and guidance live at the end of the
  last knob (`decide__system_prompt` here), where SkillOpt keeps them: at the end of the document.
- **Refused knobs.** A knob that is missing, frozen or not a text knob is refused before
  the run spends anything. A run resumed with different `knobs` is refused too.
- **Trajectories.** LiveMath's trajectory view labels each message with the step that
  wrote it (`[compare] …`) when a record has several steps.

## Structure edits: what an accepted edit guarantees

A structure edit reaches the search only through `Optimizer.attempt` (which `Run.attempt`
and the default loop call): the macro's precondition, the edit itself, the mode's
protected nodes, the validator, and lowering. An edit that passes all of them is a
candidate; any other gets the first refusal's code and reason back. What an accepted
state is promised, on any graph shape:

- **It lowers.** The spine's lowering accepts it. A child whose parent lowers and which
  does not is refused `E-lowering`, with lowering's reason.
- **Forks pair with joins (V16).** A fork has one unconditional out-edge, to its join,
  and each join belongs to one fork. Branches are disjoint, end at the join or at a dead
  end, never lead back to the fork or past the join, and declare no shared facts, no fact
  a sibling's prompt reads, and no conflicting files (lowering and V16 read the same
  rules, `nn.validate.fork_branch_conflicts`). A gate in a branch reloops only inside
  it. A join is entered only from its fork's branches and waits only for nodes every run
  of their branch visits.
- **Reads are produced first (V6, V15).** A file a node reads is written upstream of it
  on every path that reaches it. An op or a gate may also read what a node in a loop
  with it wrote the round before, or what its data node's pass wrote (it may check for
  one). An agent's inputs are required: its reads, and every `{{name}}` its prompts fill
  in that is not a knob, are there on every path to it, and a data pass's own files and
  facts stay in the pass. A `tool_policy: none` agent declares exactly one write.
- **Passes are closed (V12).** A data node's entry reaches its exit, which is not a gate
  and has no out-edge. A pass is entered only through its data node, never by an edge
  from the data node or from outside, and never leads back to it.
- **Verdicts are unambiguous (V5).** No two out-edges of a gate carry the same verdict.
- **Each agent runs its own prompt.** An agent node an edit adds gets its own
  `<id>__system_prompt` knob, a copy never binds the original's knob (`E-binding`), and
  every declared knob is read (V7).
- **Protected nodes are untouched.** A mode names them in `Mode.protected_nodes()`
  (default none). An edit that removes one, changes its declaration or out-edges, moves a
  knob it reads, or lets a run finish without one that every run used to visit is refused
  `E-protected`. A new node may still go in front of one. The optimizer's request lists
  them.

### The invariant harness

`tests/structure_fuzz.py` checks those promises by generating random structure edits
from a seed, over LiveMath's skill graph and seven small graphs
(`tests/structure_fuzz_graphs.py`: a fork and join, a nested fork, the three gate kinds,
a gate loop, a data pass, a pipeline of ops, and several terminals). The edits cover
every primitive and macro, node copies taken from the node JSON the optimizer is shown
(new ids, new prompts), and malformed edits. Each walk applies them through the same
attempt path, and every accepted state is checked independently of the validator:
lowering, a walker run with a scripted model (it must end at a terminal or a halt), the
spine beside the walker on a sample (same facts and item scores; graphs with a user or an
agent gate are walker-only, as the test drain answers no gate), the knob ownership, the
dataflow, the fork pairing and the protected nodes.

A violation is shrunk to the fewest edits that still show it and can be saved, with
`structure_fuzz.save_case`, to `tests/fixtures/structure_fuzz/`; each case there replays
in the normal suite and must now be refused or sound. The same `seed` tries the same
edits in the same order.

- **Quick tier**, in every run: `pytest refactory/tests/test_structure_fuzz.py` (300
  edits on the walker, about 10 s), the corpus, and one mutation check per invariant,
  which switches that invariant's protection off and requires the harness to find a
  violation.
- **Long tier**, by hand: thousands of edits per seed, with the spine on every twelfth
  accepted state, the corpus on the spine, and the spine's mutation check. It needs the
  installed workspace (`pnpm install`).

```bash
uv run python -m pytest refactory/tests/test_structure_fuzz.py --run-structure-fuzz-long \
    -k long_tier   # STRUCTURE_FUZZ_EDITS=4000 STRUCTURE_FUZZ_SEEDS=1,2 by default
STRUCTURE_FUZZ_REPORT=/tmp/fuzz.json   # optional: the counts and findings as JSON
```
