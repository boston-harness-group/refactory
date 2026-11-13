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

## The default loop's validation gate (on by default with val items)

When the run's parts manifest (`--parts-manifest`) has val items, the default loop
evaluates candidates the way the SkillOpt strategy does, using the same
`ValidationGate` and `FinalEvaluation` from `outerloop.py`; only the optimizer, where
edit ideas come from, differs. No flag is needed: an unset `val_gate` (the command
line's, a profile's or the library's default) turns the gate on when there are val
items. `--val-gate 0` turns it off, and `--val-gate 1` insists on it (refused without
val items). `config.json` records the decision, and the report's `selection` section
says why (`val_gate_basis`).

- **No val items.** A run with no parts manifest, or one without a val part, selects
  on train scores and says so: "no validation items: selecting on train scores", in its
  events, its `strategy.log` and its report (`selection.warning`). With `--val-gate 0`
  the report says the run selected on train though it had val items.
- **`--holdout`.** The validation set for selection is always the manifest's val part.
  The `--holdout` fraction of a Task's items (a run without a manifest) is never
  selected on: it stays the end-of-run audit's independent split. With a manifest the
  audit's held-out items are the val part, so under the gate the audit runs on the part
  the search selected on. It then gives no overfit verdict and rolls nothing out: it
  says that no independent check was possible (`holdout.split` is `val`,
  `holdout.independent` is false, `status` is `not_applicable`, and the report names
  the split). With a test part the audit compares val with test, which nothing selects
  on (`split` `test`). A resumed run keeps the `val_gate`, `selection` and `pool_size`
  its `config.json` recorded, whatever its profile or the defaults now say; a resume
  command that gives one of them explicitly with another value is refused, naming both.
  A launch that leaves `selection` unset records the rule it resolved (`single` or
  `pool`).
- **Custom strategies** (SkillOpt) keep their own gate: an unset `val_gate` stays off
  for them, and `--val-gate 1` is refused.

Train is the proposer's feedback, validation decides every choice and test is scored
once at the end. The validation average is the fitness for the parents, the pool and
the winner, in every strategy, optimizer and parent picker; the proposer never sees a
val score. `--selection` says how it selects:

- **`single`**: SkillOpt's line. A candidate is kept only if its val score strictly
  beats the incumbent's (the gate's current candidate), and the incumbent is the only
  parent. The default for the default loop with a single-parent optimizer (`random`,
  `autoresearch`, `hill_climb`, alone or as `--parent-picker hill_climb`), the
  SkillOpt-matched setting; its decisions are those of the gate before pools existed.
- **`pool`**: entry is permissive. Every judged candidate may enter a pool of the
  `--pool-size` (8 by default) best by val score, with any tied with the last of them.
  Parents come from the pool, the incumbent first, and each strategy chooses among
  them by val (a tie by train): `random` takes the best on val, and a model proposer
  may name any pool member as the parent. The population optimizers and pickers
  (`archive`, `evolutionary`, `islands`, `map_elites`) are handed every judged
  candidate a person has not rejected instead, and apply their own rule by val: the
  archive's own limit (12), tournaments, the best of each island, each niche's
  champion. So a new structure ranked below the pool keeps its niche or island. The incumbent still changes only on a strictly higher val score (a tie
  goes to the higher train score, then the earlier candidate), and it is the winner. The
  default for every other strategy and for the population optimizers and pickers.

Strategies under the gate: `reflective` merges the two best of the pool on val and
chooses prompt or structure from the val gains of past children over their parents
(the gate's cache, rejected children included); `best_of_n` and `autoscientists` score
the seed and send their samples and branches through the gate (the merge base is the
best branch on val), and their natural ends (`samples_exhausted`, `merged`) run the
final evaluation; `adaptive_meta` measures a stall on the best val score; `tree_search`
debugs a failed candidate whether or not the gate kept it, and its notes name the leader
without a val score. Every report has a `selection` section: the setting, the pool, its
lines of descent (the seed's child each member descends from) and edit kinds, the share
of children bred from the incumbent, and the budget (train, val and test scorings,
rollouts, tokens, wall-clock). The SkillOpt strategy keeps its own gate.

- **Train seed.** Train rollouts in epoch `e` use SkillOpt's seed, `seed + e*1000 + 1`
  (`outerloop.train_batch_seed`). Val and test rollouts use `seed`.
- **Gate.** Every candidate evaluated on train is rolled out on val, including one whose
  train run failed, and must strictly beat the gate's current candidate (the last one
  accepted), not its parent. The candidates of one round (`--batch-size` above 1, or a
  `--portfolio` round's one per edit level) are judged together, each against the
  current candidate as the round starts, so the order they were proposed in decides
  nothing: of those that beat it, the one with the highest val score is accepted and
  becomes current; a tie goes to the higher train score, then to the one proposed first.
  The others are rejected, and one that beat the bar but lost to a sibling is marked
  shadowed in the report. SkillOpt gates one candidate per epoch, where the two rules
  agree; its own strategy keeps its gate unchanged.
- **Parent and winner.** Under `single`, as in SkillOpt, the gate's current candidate
  (also the best on val) is the parent every optimizer edits, with any accepted
  candidate a person kept, whether or not its train run succeeded, and the leaderboard
  lists the accepted candidates a person has not rejected, ranked by val score. Under
  `pool` the leaderboard is the pool, the incumbent first. Its first is the winner: the
  report, the held-out audit and `adopt` all name it, and the report's `best_score` is
  its val score (`best_score_basis`), with its train score in `best_train_score`.
- **Fresh evidence.** As SkillOpt rolls out its current candidate on the train part
  every epoch, the candidate the optimizer edits is scored again at the start of each
  epoch's proposal step, on that epoch's train sample (the whole train part, or the
  `--train-minibatch` sample) with that epoch's seed. A candidate created and scored
  this epoch is not scored twice. The optimizer, and every reader, reads these fresh
  records; the reader cache is keyed by the records' text, so new records are read
  again. Only the evidence changes: the candidate keeps its id, its gate val score and
  its train score, which is the score the leaderboard, the history and the gradient
  show (its first train evaluation's). The re-score counts as an evaluation, a
  measurement and its rollouts, is charged to the time and token budgets, and is
  journaled (`rescore`, with its epoch and seed); a resumed run reads its epoch back and
  does not repeat it. A re-score that fails, or returns fewer items than it ran, is no
  evidence: the previous records stay, the epoch is not marked, and it is journaled as
  `rescore_failed` (still counted as an evaluation). The re-score runs in the round
  itself: when an epoch evaluates children of the candidate the last proposal step
  edited, that parent is scored on the epoch's train sample with them. The children
  keep the `--max-concurrent-evaluations` slots: the re-score joins the round's last
  wave when that wave leaves a slot free (two portfolio children and three slots, the
  live shape), else it runs after the children, alone. Either way it is a request of a
  `Run.evaluate_many` batch: its items count against `--max-items-in-flight` with the
  children's, and a stop ends it with them (a child's error is the one raised). It goes
  in only when there is room for every child still to score and it: `--max-evaluations`
  must hold all of them, and `--max-rollouts`, when set, each child's train and val
  rollouts and then the re-score's train rollouts, so a re-score never takes what a
  child needs to be scored or judged. It is skipped when it already has this epoch's
  records.
  So every judged child pairs with its parent on the same seed, a round that keeps a
  child included; a round that keeps none finds the parent re-scored and does not score
  it again after gating, and a re-score that failed in the round is not tried again in
  that epoch. The run's last proposals, scored after the last epoch, get no parent
  re-score (no request reads their comparison). Train records are kept per epoch, so
  "What the last edit changed", "Your earlier reasoning" and the run-wide feedback
  sections compare a child with its parent's records from the child's epoch (the same
  seed); a pair from different epochs is marked as such. `--fresh-evidence 0`
  turns it off (the minibatch re-score of `--train-minibatch` stays). On LiveMath's
  35-item train part this is one train evaluation of 35 rollouts per epoch, the epochs
  whose gate kept a child included; with 3 concurrent evaluations (two portfolio
  children and the re-score) the round's wall clock is unchanged. Every gated strategy
  that runs epochs, and a parent picker, pays about one extra train evaluation per
  round that keeps a child; under a fixed `--max-evaluations` the run then gets fewer
  rounds. `best_of_n` and `autoscientists` are unchanged.
- **Last epoch.** The batch proposed in the last epoch is evaluated on train (with the
  next epoch's seed) and gated before the run ends, so N epochs gate N proposed batches.
- **What the optimizer reads.** Every row it sees (recent results, the history, the
  gradient's leaderboard) carries the gate's decision: `current` (the incumbent),
  `kept` (accepted as the incumbent when judged), `rejected` or `not gated`, under
  `pool` as under `single`: pool members are not labelled as such. The request's
  "Recent results" table lists the pool, though, so a `rejected` row there is one that
  scored well on val. Under `pool` the request adds that a parent may be any candidate
  that scored well on val, or with a population parent picker (`archive`,
  `evolutionary`, `islands`, `map_elites`) any judged candidate, a rejected one
  included. The report, `state.json`
  and the log label pool runs in full: `current`, `in pool` (in the pool now), `left
  pool` (in it after the gating that judged it, pushed out since), `rejected` (never in
  it, or rejected by a person), `below pool` (never in it, but a population picker still
  breeds from every judged candidate) or `not gated`. "Recent results" lists the current
  candidate first and the rest by train score, then age, never in val order. The
  request's first line, and the line over "What this run has already tried", give the
  train score of the current candidate, labelled as such. The per-knob attribution
  counts, for each value, how many of its candidates had each decision. The request
  says it edits the `current` candidate, and that a `rejected` one was not shown better
  on validation (with the gate's resolution, one item = 1/N of the val score, when the
  score is the items' mean); a rejected change is never re-proposed unchanged, but one
  whose train fixed/broken counts (when shown) were strongly positive may be built on or
  combined with edits to other steps in the same failure path. "Your earlier reasoning"
  shows the full edits of the latest such rejected candidate (more fixed than broken):
  every step's wiring whole, its free text sharing 4000 characters
  (`REJECTED_EDITS_CHARS`).
  The history's best is the best kept candidate. Val scores are never shown to it;
  scores there are train scores. That includes "Previous runs of this mode": a gated
  sibling run's report gives its winner's val score as `best_score`, so the summary
  carries that run's `best_train_score` instead (as for any `best_score_basis` other
  than train), and a bound whose text names a held-out part (test, holdout, held-out,
  hold out; "seed scored on train, val and test") is left out: no test or holdout
  number or wording reaches it.
- **What a person reads.** Each epoch's progress line says how many candidates the
  gate kept and rejected that epoch, with the val best and current scores, instead of
  a train-score "improved".
- **Final evaluation.** After a normal end (the epoch bound or an early stop), with a
  test part, the final candidate's val check runs, then the seed, the best and the final
  candidate on test with `seed`; a best or final that is the seed reuses the seed's
  test score (one rollout, its delta exactly 0), as a final that is the best reuses the
  best's. Before this, a seed that won was rolled out twice and its delta was rollout
  noise. A run that ends on a spent evaluation, rollout, time or
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
  one batch proposed and gated. Both settings need the gate (val items, or `--val-gate 1`), are saved with the
  run and are kept on resume.

## An evaluation that dies with the run is not a result (exit 75, `--resume`)

A failed item is the candidate's result only when it ran and failed. An item whose
process died with the run has an unknown outcome (`items.interrupted_item`). That is
an item whose model calls kept failing on the wire through the client's retries (a
proxy or provider outage), whose completion process was killed by SIGKILL or SIGTERM
(exit -9 or -15: the host's out-of-memory killer, a stop from outside), whose product
worker was killed (`signal: SIGKILL`), whose op child exited before it reported, or
whose model call the run's stop cancelled. Only an item with no verdict qualifies (an
error, or "no answer" because a step's answer is missing). An item the Task graded is
always its result. A model timeout, a wrong or garbled answer, an output limit, a
product's own failure and any other signal (SIGABRT from a runtime out of heap,
SIGSEGV) are the candidate's failures, as before. A wire failure is only ever called
one when the session's last cause is the wire's. The model's streamed reasoning (which
may say "429") never decides it.

- **In-run retries.** A train evaluation, a val or test rollout or a fresh-evidence
  re-score with such an item, or whose run failed outside any item for one of these
  reasons, is tried again inside the run after 2 minutes, then after 5 more
  (`INTERRUPTED_RETRY_WAITS_S`). There is no wall-clock cap: each attempt has the run's
  usual limits. Each attempt is journaled as `evaluation_interrupted` (attempt, outcome,
  causes, how many items died) and shown. Before an attempt runs again, the earlier
  attempt's folder is moved to the run's archive.
- **The stop.** Still interrupted after its retries, the evaluation stops the run with
  exit status **75** (`EX_TEMPFAIL`, `EXIT_INTERRUPTED`) and `"resumable": true` in the
  error JSON. Exit 2 means a refused configuration. Nothing of the evaluation is
  recorded: no score, records, val occasion, re-score or rollouts. The candidate stays
  unevaluated and its epoch open. `refactory optimize ... --resume` evaluates it again
  from the start and gates it as any other.
- **The cap.** Each stop counts for that candidate and part (a re-score counts per
  epoch), from the journal, so a resume keeps the count. An in-run retry is not a stop.
  The run stops on the first two. On the third (`INTERRUPTION_CAP`), the evaluation is
  recorded as failed, with the error "`<part>` evaluation interrupted 3 times; recorded
  as failed", and the run goes on. An evaluation that dies every time therefore cannot
  hold a run in a stop-and-resume loop.
- **Receipts are kept.** When an unevaluated candidate's train evaluation or an epoch's
  re-score runs again, in a retry or after a resume, what the earlier attempt left (its
  per-item passes and receipts) is moved to `.refactory/archive/<UTC time>/runs/<id>/...`
  inside the run (`evaluation_archived` in the journal), never deleted. A candidate's
  `occasions/` stay in place.
- **Seed results.** A `--seed-results` part with such an item is a miss: it is
  measured instead.
- **Runtimes.** The item records come from the same code on both runtimes, and a model
  call fails with the same text in the spine's worker as in the walker. Known gap: when
  the spine's own worker or op process is killed and leaves no stderr, the spine records
  only `exited with code null`, which names no signal. Such an item is still counted as
  the candidate's failure on the spine, where the walker counts it as interrupted.
- **Out of scope.** An interrupted evaluation runs again from its start. Resuming inside
  forks, loops and per-item passes is separate work.

## Train minibatches (`--train-minibatch N`)

By default every candidate is scored on the whole train part. With
`--train-minibatch N` and a train part of more than N items, each epoch samples N train
ids with SkillOpt's per-epoch seed (`train_batch_seed`). It needs the gate (val items, or `--val-gate 1`):
without the gate the winner and the parents are picked by train scores, which would
then come from different samples, so the run is refused before anything is rolled
out.

- **One sample per epoch.** Every candidate scored in that epoch uses the same sample,
  so candidates in a round are comparable. The sample changes from epoch to epoch, and
  each epoch's ids are journaled (`train_minibatch`).
- **The evidence is that minibatch.** When the candidate the optimizer edits was scored
  in an earlier epoch, it is scored again on this epoch's minibatch (the gate's fresh
  evidence, above, on the minibatch), so the proposal
  evidence is that minibatch, as SkillOpt rolls out its current candidate on each
  step's batch. Its score is left alone; only the records the optimizer reads change.
  This re-score is an evaluation like any other: it counts toward the evaluation,
  rollout, time and token budgets, and it is journaled (`rescore`).
- **Resume.** A resumed run samples the same ids. Resuming with a different N is
  refused.
- **Unaffected.** Val and test are not sampled.

`0`, or a train part of N items or fewer, scores the whole train part as before.

## Compound proposals (`--max-edits-per-proposal N`)

A proposal may carry an ordered list of edits, `{"edits": [...]}`, instead of one
`edit`. They are applied together to the parent as one candidate, which is evaluated
and gated once, so a fix that needs coordinated changes (several steps' prompts that
must agree, or a new node and the knob that feeds it) is judged as one idea rather than
as a chain of single edits each judged alone. SkillOpt's round patches change several
sections at once; this is the default loop's equivalent.

- **Checks.** Every edit's shape, op and permission are checked first; then each edit
  is prechecked and applied in order, to the state the ones before it left (a
  `set_knob` on `<id>__system_prompt` therefore goes after the edit that adds node
  `<id>`). The combined state is then held to every state rule once: frozen knobs, tool
  policy, protected nodes, the validator, lowering, no-op and duplicate (on the combined
  state, so the same point reached in another order is a duplicate). An edit the result
  does not depend on -- one that changes nothing, or that a later edit overwrites or
  undoes -- is refused `E-noop` with its index. A refusal about one edit names it,
  `[edit k of n: op]`, after the rule's code; one about the combined state names none. A
  list longer than N is refused `E-too-many-edits`.
- **Cap.** `--max-edits-per-proposal` (default 4, SkillOpt's edit budget) caps the
  list; `1` is single edits only, as before.
- **Who emits lists.** The `autoresearch` request says it may bundle up to N edits when a
  fix needs coordinated changes, with a JSON example (not shown when N is 1). When it
  asks for one proposal (one candidate per epoch), it asks that the proposal's edits
  cover every step the diagnosis names on the failure path, since an idea left for a
  separate proposal is never run; when it asks for several, separate ideas go in
  separate proposals. The random
  and population optimizers emit single edits. A proposal with one `edit` works as
  before.
- **Reading an answer.** A proposal whose edit sits at its own top level
  (`{"op": "split", ..., "rationale": ...}`) is read as that edit, keeping only the
  fields its op takes (a proposal-level `id` or `diagnosis` is dropped); an answer with
  no `proposals` list is read the same way when it is one proposal (`proposal`, an
  `edit`/`edits`, or a bare edit at its top level). An entry that still yields no edit,
  or an answer with nothing to read, is refused `E-proposal-shape` with the reason: it
  is counted with the epoch's refusals (and in the refusal pressure that can add an
  attempt) and shown under "Why your last proposals were refused" in the retry and the
  next request, never dropped. An empty `proposals` list proposes nothing.
- **Records.** The candidate's `edits` hold its lineage's edits in order;
  `Candidate.step_edits` says how many its own proposal added and `own_edits()` returns
  them. The proposal record, the journal (`candidate` with `step_edits`, `proposal`),
  "What the last edit changed" and "Your earlier reasoning" list every edit, its op and
  what it names whole. Wherever past edits are shown, the wiring of the edit and of each
  node it carries (ids, kind, reads, writes, produces, call, evaluator, gate fields, ...;
  `EDIT_WIRING_FIELDS`) is never cut; only free text (prompts, code, knob values, and
  anything under a `value`, `patch`, `config` or `spec`) is shortened, with
  "… (N more chars)", or "… (N chars)" alone when a text's share is tiny. The texts
  share a fixed budget: 400 characters per field of an edit in those two sections
  (`EDIT_FIELD_CHARS`; 300 in the reasoning entries), short texts whole and long ones
  cut evenly, never more than the budget in all. "Recent results" and "What this run
  has already tried" show each candidate's lineage of edits compactly: a text
  `set_knob` as old -> new length and a line diff against the value it replaced (the
  previous edit to that knob, else the seed's value), lines joined with ` ⏎ `; a patch's
  ops; a structural edit with its wiring whole and its texts sharing 600 characters
  (`HISTORY_EDIT_CHARS`), at most 200 each; a diff, value or patch at most 600
  characters, with "... (N more chars)" when cut. A refused list is recorded as
  `{"edits": [...]}`. The loop-as-a-graph path (`write_pending`/`read_pending`) carries
  `step_edits` and the edit reports.
- **Attribution.** The per-knob attribution credits knob values by state: every knob's
  value in a compound candidate is credited with that candidate's score (and its gate
  decision), so each knob a bundle changed gets the credit, shared with the others.

## Grader-derived data stays on the grader side (V17)

No node outside the grader side may consume grader-derived data. The rule is generic: it
names no fact, file, Task or benchmark. The validator computes a provenance set from the
graph and the declarations (`nn.validate.grader_derived`):

- what each data node publishes after its pass (its `produces`: the pass's graded
  records, under whatever name), never the per-item input its pass reads;
- the files the runtime's grading writes (`items.GRADER_FILES`: the run's facts, each
  finished item's record and marker, the collected records);
- what a Task declares its grader leaves where nodes could find it
  (`Task.grader_outputs`);
- everything a node the mode declares as grading writes or produces
  (`Mode.grader_nodes()`);
- and, to a fixpoint, everything any node that consumes one of these writes or produces:
  a grade copied through an op, a file or a fact is still the grade.

A node consumes what its prompts read when rendered -- every `{{name}}` slot and the
name every `{% if %}`/`{% elif %}` condition tests (bare `name`, `name == '...'`,
`name != '...'`: a condition tells the agent something about the value without printing
it), in its prompts and in knob-bound fields -- its `feedback_keys`, its reads and its
required_reads, and its `inputs` (the facts a code node or a verdict gate takes in). The
grader side is the data nodes, the mode's protected nodes and its grader nodes; any
other agent, op or gate consuming grader-derived data is refused `V17`, on every
candidate and every shipped or created mode (`registry`, the creation sandbox). The rule
runs in Python before anything is lowered: the spine only runs states that already
passed it, so it needs no copy of the rule.

Grader status follows a node's id, so the grader nodes are protected for edits as the
protected nodes are (`Mode.edit_protected()`: no edit may remove, rewire or replace them,
`E-protected`), and every knob a protected or grader node reads or binds holds: an edit
that sets or removes one is refused `E-frozen`, naming the node. Otherwise a search
could rebuild a grader node under its id, or rewrite a protected agent's prompt to copy
answers into what is scored. (Freezing those knobs breaks no shipped mode: LiveMath's
protected nodes read only knobs that are frozen already.)

Routing on the grade is allowed: a protected or grader gate may send a run back (a
reloop on a failed check). That is control flow the mode's author wrote, not data a
search-written node reads; the nodes after the gate learn only which branch ran.

V17 covers what the graph declares, nothing more. What the graph does not declare, the rule cannot see. An agent with tools can read any
file in its working directory, the runtime's grading files included (`items.GRADER_FILES`;
the runtime does not hide them from it, since modes such as `build` run tool-enabled
agents inside a pass that need the working directory as it is); an agent an edit adds runs without
tools (`E-new-node-tools`), so tools are only on agents the mode's author gave them,
and those are trusted as the author's. An author's op may read facts it does not
declare; it is the mode's own code. A code node an edit adds is not: its code gets only
its declared reads and `inputs`, sandboxed (see "Code nodes" below), so V17 sees
everything it takes in. (Authored ops reading undeclared facts or files remain a known
gap of V17, filed as a follow-up.)

## What the optimizer sees, and what it said

`ProposalContext.render()` is the request every model optimizer reads. Besides the
counts and the error-cause and wrong-answer tables for the candidate being edited, it
shows **every** train item of that candidate in full, in every proposal request: all
failures and all successes, never a sample. Each item shows its input (the item metadata's `question`,
or `problem`, `task_description`, `instruction`, `input` or `prompt`, else the first
step's prompt) and its `choices`. Then it shows each step's output from the record's
`exchanges`, in the order the pass ran them (a fork's branches in branch order), then
the expected answer (`label` and `label_text`), the prediction and the grader's reason.

- **Answer rules.** Built from the run's edit catalogue and the node schema, with
  placeholders only. When the run permits a structural op, the rules show a structural
  example (an op that puts a new node after an existing step, its `node` built from the
  schema's required fields) beside the parameter-move example, and the bundling example
  shows two prompt knobs first and then that node together with its own
  `<new step id>__system_prompt` knob, so neither kind leads; a run that permits
  parameter moves only sees no structural example. A new node is written from
  the node schema; the mode's blocks, when it declares any, may be used verbatim, and a
  mode with none says so ("New nodes"). The size cost states the run's parsimony per
  node; only when every train evaluation's score was the mean of its items' scores
  (the loop checks) does it add what one item is worth (1/N), since a mode's score may
  be any metric of the run; otherwise it says to compare it with the score differences
  between candidates.
  The template carries no task or domain vocabulary (`test_proposal_presentation`).

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
  Then as many items in full as the rest of the budget holds: first the items the
  readers cited, one per pattern in turn (failure patterns first), so every pattern has
  an item in full before any has a second; then failures and passes alternating. A reader call that fails is noted, with its items, and the proposal
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
- **Own prompt knobs.** A new agent node gets its own `<id>__system_prompt` knob holding
  its `system_prompt_template`, as a mode's agent nodes get one when it is built, and its
  own `<id>__task` knob holding its `prompt_template`, so a later round can `set_knob`
  either.
- **Split replaces.** `split` replaces an agent or op with two new steps: the first takes
  its in-edges (and the start, or a pass's entry), the second its out-edge (and a pass's
  exit), and the step goes with its own knobs. It is refused for a gate, fork, join or
  data node, a fork branch or join source, and when it would leave a knob read by
  nothing.
- **Where `{{fact}}` is filled in (V20).** An agent cannot read a file: a file's text
  becomes a fact through a code op that reads it and `produces` the fact, or when the
  agent writing it declares `produces`. A `{{fact}}` is filled in an agent's
  `prompt_template` and `system_prompt_template` (its `<id>__task` and
  `<id>__system_prompt` knobs) and in a knob an op marked `nn.fills_facts` reads, as
  LiveMath's skill graph marks the ops that read its `<step>_task` knobs; there a slot is
  one of the op's inputs (V15, V17). A slot an edit leaves in any other knob would reach
  its reader as written, so it is refused -- whether the edit added the slot or removed
  or rewired the step that filled it. It is judged by node, knob and slot against the
  parent: a slot the parent already left unfilled at the same node and knob (a design
  brief asking for a `{{name}}` renderer) is the author's or the user's text and is
  left be. Seeds and knobs a person sets (`--knob`) are not judged by V20; a test holds
  every shipped mode's seed to having no unfilled slot.
  The request names the knobs such ops read.
- **Lost writes (V19).** A no-tools agent or code op that sets no fact, whose every file
  a no-tools agent writes again on every path before any node reads or may read it (a
  Python op or gate, an agent with tools, a data pass), is refused.
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
  noise. Without a parent (the seed) or without shared items, it says so. When some of
  the child's train items failed with an error (the item's run failed, not a wrong
  answer), it counts them by kind and the node that raised them, with the first message
  of each kind (300 characters, `ITEM_ERROR_CHARS`). Both runtimes start a failed node's
  error with the line `<node>: <Kind>: <message>`, which is what is read (the spine
  takes it from the node runner's own account after its failure marker, or from a
  worker's error line, and keeps the node's stderr tail below it, less the runner's
  stderr copy of that account); a fork's
  nested error counts at its innermost node. An errored item the parent also got
  wrong is neither fixed nor broken, so without this a structural child that fails for a
  mechanical reason would look like one that just scored worse.
- **Your earlier reasoning and what happened.** Up to 6 recent candidates: the line
  from the seed to the edited candidate, and the latest rejected children. Each shows
  the rationale and prediction the optimizer gave when proposing it, the round's
  diagnosis (and what that round's answer said it considered, when it did), the gate's
  decision, its train score and its fixed and broken counts against its parent, and its
  train item errors counted the same way. Texts are clipped to 600 characters. Each proposal keeps the diagnosis of the call that
  produced it, and the proposals' own words are journaled, so a resumed run keeps them.
- **What an answer considered.** The answer format has an optional, short
  `considered`: the kinds of change the optimizer weighed for its diagnosis (only the
  kinds the run permits are named) and why it chose its own. It asks for no particular
  kind. When given (a string, a list or an object, kept to 1500 characters) it goes in
  the call's trace line, the proposal's journal record, the round's
  `analysis/generation-<n>.md` and the report's "What the optimizer weighed"; an answer
  without it is read as before.

A proposal refused as a duplicate (`E-duplicate`) is told which candidate it repeats
(its id), so the retry can look it up and move on. This applies with or without the
gate.

Every call the `autoresearch` optimizer makes is appended to
`<run root>/optimizer_trace.jsonl` as it happens. Each line records the epoch, the
generation, the proposal round, the system prompt, the prompt, the reply, how many
proposals were parsed, the diagnosis, what the answer considered, and the error for a
call that failed. A resumed
run appends to the file; a fresh run in the same directory starts it over, as it does
the journal. A trace line that cannot be written (a full disk) is a warning; the
proposals are kept. The other library optimizers call no model; SkillOpt records its own
calls in `skillopt_trace.jsonl`.

### Run-wide feedback: changes by kind, lessons, added-step reviews

Three more request sections (`optim/feedback.py`) carry what the run has taught so
far, beyond the recent candidates above. Each has a switch (a profile setting, or `0`/`1`
on the command line), none changes how anything is scored, and none reads anything but
the graph, train records and the gate's outcome words: no answer or option is parsed,
and no validation item or score is shown. They hold for every strategy that builds this
request (the default loop, `reflective`, `best_of_n`, `autoscientists`, a parent
picker paired with `autoresearch`); SkillOpt builds its own requests and is unchanged.
The Python walker and the optimizer only; nothing here touches the TypeScript spine.

- **Every change so far, by kind** (`--run-ledger`, default `1`). Every candidate the
  run has evaluated (under the gate, every one it judged), oldest first: id, round,
  parent, the kind of change its own edits made, the knobs and steps they touched, its
  train fixed/broken counts against its parent on shared items, the gate's word and its
  rationale (140 characters). Under the gate the counts compare the candidate's records
  from the round it was evaluated in (not a later re-score of it as the edited
  candidate) with its parent's from that same round, so both share a seed; when the
  parent was not scored in that round the row says `n/a (different seed)` and the
  tally leaves it out. The kind comes from the edit ops and the kinds of node
  they add: `prompt/setting` (knob edits only), `added agent step`, `added code step`
  (an op whose code the edit wrote), `split`, `loop/gate`, `parallel`,
  `removal/rewire` (bypass, remove, reorder, edges) or `other`; a proposal that makes
  several is filed under its most specific structure. A tally line per kind comes
  first (tried, kept, best train fixed−broken). The newest 40 rows (within 9000
  characters) are listed; older ones are only in the tally.
- **Your lessons so far** (`--lessons`, default `1`). The answer format has an optional
  `lessons` field: what the results so far teach about which changes help here, keeping,
  revising or dropping earlier lessons. The latest (at most 1500 characters) is kept with
  the epoch and generation it was written in, journaled (`optimizer_lessons`), restored
  on resume and shown back in the next request; an answer without it keeps the previous
  lessons.
- **Did the added steps contribute?** (`--step-review`, default `1`). For every step a
  candidate of the last round added (an insert, a split's halves, a fork's branches, a
  loop's gate; not the fork and join themselves), on the train items it shares with its
  parent: the candidate's fixed and broken items, how many items' final answer changed
  (the item record's `prediction`, the answer the grader read, whitespace- and
  case-normalised and not parsed further; reasoning text differs from run to run on
  nearly every item, so it is not what is compared; an item with no recorded answer
  falls back to the last step's output text, counted apart and labelled so), and the
  step's median duration and tokens when its records carry them. These counts follow
  the ledger's same-round rule; without a same-round parent they are `n/a (different
  seed)`. Then one call of the optimizer's model per step that recorded
  text output (at most 8 per request): it reads, on up to 4 train items (changed outcomes
  first), the step's prompt as sent, its output and the output of the next recorded step
  that reads it (through the graph's reads, inputs and `{{fact}}` slots, so a sibling
  fork branch is not taken for it), and answers
  `adds_new_work`, `restates_inputs`, `contradicts_inputs` or `no_effect` with a short
  quote and a note. The call is recorded in `optimizer_trace.jsonl` (kind `step_review`)
  and cached in memory and in `<run root>/step_reviews/` (cleared by a fresh run in the
  same directory, as the trace is), so a retry or a resumed run pays for none again.
  The section keeps full entries within 9000 characters; steps past that get one line
  of counts each. A failed or unreadable review is recorded and the step is shown
  without a verdict; it never fails the proposal.
- **What makes a new step pay off** (`--structure-guidance`, default `0`). When on,
  "How nodes compose" adds one generic paragraph: a step helps when it adds work or
  information the pipeline lacks (an independent attempt, several attempts combined by
  a vote or a check, a computation or check in code, a retry with a specific reason);
  one that re-reads the same notes with the same model tends to repeat their mistakes.
  Off by default, for an experiment.

The readers' success patterns (what the passing items do right) have their own heading
after the failure patterns, up to 4 per reader whatever its failure patterns, and a
quarter of the findings' room the failure patterns cannot take.

### The analyst stage: SkillOpt's per-round analysis (`--analysts`)

`--analysts 1` (default `0`) gives the model proposer the analysis SkillOpt's reflect
stage gives its patch step (`optim/analysts.py`; the SkillOpt port's `analysts.py` and
`merge.py`). Before each proposal of `autoresearch` (alone, under `--portfolio`, or
paired with a parent picker), on the train records the request carries:

- **Failure analysts.** Every item not passed (answered wrong, no answer, or an error),
  each once, shuffled with the epoch as seed and cut into batches of
  `--analyst-group-size` (8, SkillOpt's `minibatch_size`); one optimizer-model call per
  batch. At most 8 calls of the stage are in flight at once, merges included (the two
  sides' merges run side by side), so 35 train items (5 or 6 batches) take one wave of
  analysts. Each reads the pipeline (its nodes and edges, each node in
  full with the settings bound to it, every other setting's value) and each item's full
  trace: its input, every step's input and output in the order they ran, the expected
  answer, the prediction, the score and the grader's feedback. A call's prompt has
  400000 characters of room (`ANALYST_ROOM`): the optimizer model (Opus 4.6) reads 200k
  tokens, of which Claude Code's own system prompt and the reserved reply take about
  30-40k, and traces dense in notation run 2.5-3.5 characters a token, so 400000
  characters is about 115-160k tokens. The batch's items share what the rest of the
  prompt leaves: a batch that fits is never
  clipped, and one that does not has its longest items cut to one common length (never
  below 4000 characters), step inputs first, then outputs. SkillOpt never clips, so the
  clipped items and the characters left out are counted. A call the provider still
  refuses as longer than the model's context ("prompt is too long", "context length",
  ...) is retried once within half the refused prompt's length, re-clipped the same
  way (a merge has its pipeline texts clipped instead; its findings stay whole); a
  retry still too long fails the batch as any failed call does. It
  writes at most 4 findings (SkillOpt's analyst edit budget): what goes wrong, where
  (steps, edges, settings, or the structure), the change that would fix it, the items it
  covers (an id outside the batch is dropped), its impact and quotes.
- **Success analysts.** The passed items likewise (seed + 1): what made them work, where
  it comes from, and what a change must preserve.
- **Merge and rank.** Failure findings merge level by level in batches of
  `--analyst-merge-batch-size` lists (8), then success findings; a final merge with
  failure priority returns one de-duplicated list ranked by items covered and impact.
  As in SkillOpt, one list is not merged, the final merge runs only when both sides
  found something, and an unreadable merge keeps every input finding (failures first).
- **The request** shows "What the analysts found, ranked" (up to 16 findings, 30000
  characters) before the item results, and the analysts take the readers' place: no
  reader call runs, and when the items do not fit one request the items they cite are
  shown in full first. SkillOpt's analysts are its only reading of the trajectories, so
  running both would give this loop more analysis than SkillOpt's.
- `--analyst-max-batches` (0, no limit, as SkillOpt) caps the analyst calls per round,
  failure batches first; the request says how many items went unanalysed.

The analysis is made once per proposal step: it is keyed by the epoch, the generation,
the edited candidate, its train records, the stage's settings and the optimizer's model,
not by the lessons, so a retry after a refusal (whose request carries the lessons the
refused answer wrote) reuses it and makes no call; the analysts and merges read the
lessons of the step's first attempt. The step's analysis is also kept in
`<run root>/analyst_findings/round-<key>.json` when every call of it succeeded, so a
resumed run reuses it; a step with a failed call is made again on resume, its successful
calls answered from their own cache, and that analysis (the one its proposal sees) is
journaled too, with the calls it paid. An analysis is journaled again for its step only
when the earlier one was incomplete (a failed batch or merge, or stopped by the budget)
and this one differs from it.

Every call is recorded in `optimizer_trace.jsonl` (kinds `analyst_failure`,
`analyst_success`, `analyst_merge_failure`, `analyst_merge_success`,
`analyst_merge_final`) with its parsed findings, and cached by the optimizer's model and
its whole text, in memory and in `<run root>/analyst_findings/` (cleared by a fresh
run); a retried call's answer is cached under the original call, with the retry
recorded. Each step's batches, merges, call counts, clipped items (and characters left
out), context retries and ranked findings are journaled once (`analyst_findings`,
restored on resume), also when the proposal call after it fails, and when a spent budget
stops the stage part way (with what it paid and the `error`). report.md's "What the
analysts found" gives the rounds, the calls, the clipped items, the context retries and
the latest round's top 5 findings. A failed analyst call is shown and never fails the
proposal; a spent budget, a cancelled run or a missing model choice stop it, as in the
SkillOpt port. The calls go through the optimizer's model route and count in the run's
budget like the readers'.

A run with `--analysts 1` whose optimizer asks no model (the random proposer, a parent
policy alone) is refused when it starts. A custom strategy (`--strategy skillopt`)
builds its own optimizer requests: setting it warns, and the stage runs only for
proposals the strategy asks of the loop's model proposer.

The prompts name no dataset or task, ask for nothing beyond what SkillOpt's analysts
ask, and see train records only. Off, no call is made and every request is
byte-identical to a run without the stage; the four settings are excluded from the
settings digest, so a run resumes whichever way they are set.

Per round with SkillOpt's settings, for F failed and S passed train items:
ceil(F/8) + ceil(S/8) analyst calls, then the merges (one per side with more than one
list, at most 8 lists each, plus the final one when both sides found something), then
the proposal: on 35 train items, about 5-6 analyst and 3 merge calls, as SkillOpt's
analysts and merges make. SkillOpt's rank call (it trims its merged edits to the edit
budget), slow update and meta skill are not run: the proposer chooses and writes the
edits, and its `lessons` are this loop's optimizer-side memory (they are shown to the
analysts and merges, as SkillOpt shows its meta skill).

Where the analysis still differs from SkillOpt's: SkillOpt's LiveMath arm uses
LiveMath-specific analyst prompts (`livemath/skillopt_prompts/analyst_error.md`), while
these are generic by design (no dataset hints); SkillOpt never clips a trace (here a
batch over the call's room has its longest items clipped, and counted); SkillOpt shows
each item's task type (here the item's input, and its choices when it has them); and SkillOpt's analysts never
see an errored item (it has no conversation to show), while here it is a failure shown
with its error.

### The integrity judge (`--integrity-judge`, default `off`)

An optimizer paid by a score finds whatever raises it, and some of that is not the task
done better but a regularity of how the data, the grader or the environment happens to
be built. The integrity judge (`optim/integrity.py`) asks one question of each
proposal: *would this change still help if the dataset, grader or environment had been
built differently, or does its benefit depend on how they happen to be built?* Its
prompt names no dataset, task or answer format.

- **When it is called.** Off by default: no call, nothing recorded, and the request,
  the report and the run's settings digest are as before. On, every proposal that
  became a valid candidate is judged in `Run.attempt` (`Loop._integrity_check`), so
  every strategy's proposals pass it: the default loop and its optimizers (random,
  autoresearch, a parent picker, the portfolio, the pool), `best_of_n`, `reflective`,
  `autoscientists` and a custom strategy such as SkillOpt. The starting population's
  sweep of single knob moves is not judged, and a state already let through is not
  judged again. One call of the optimizer's model (its route, without tools, charged
  to the run's token and time budgets; without a budget, the run model's optimizer
  route) per proposal, recorded in `optimizer_trace.jsonl` (kind `integrity_proposal`),
  and one more when the run reports (from `Loop.run` or the graph-run loop's report
  node, never a report-only replay): the winner's combined change from the seed (kind
  `integrity_final`), so an exploit assembled across several edits is caught. A run
  with the judge on and no model stops at the start; a Task that cannot be loaded is
  warned about; a failed judgement is said once on stderr and counted in the report.
  Verdicts are journaled (`integrity_verdict`, with what the judge read), cached by
  prompt and restored on resume.
- **What it reads.** The proposal's edits in full (within 20000 characters), what they
  do to the pipeline's state (each changed setting as a unified diff, each added step
  whole, removed steps, wiring), what each setting it touches is for (the mode's own
  description, kind, allowed values, default, old and new value, the steps that read
  it), the optimizer's diagnosis, rationale and predicted effect (each one missing is
  said to be missing: "no rationale given"), the pipeline before the change, everything
  earlier changes left in the candidate (the starting pipeline against the candidate,
  within 16000 characters, when the parent is not the seed), and a summary of the
  parent's train records the proposer saw (outcome, reference result, what the pipeline
  produced, the start of each input, within 6000 characters). It judges the candidate
  as it would run, and says whether each finding is about what this change adds
  (`source: new`) or only about content an earlier change left (`inherited`). Never a
  validation or test item.
- **What it answers.** `{"verdict": "clean" | "suspect" | "exploit", "categories": [...],
  "evidence": ["<quote>"], "property": "...", "source": "new" | "inherited",
  "reason": "..."}` with categories from a
  fixed generic set: `answer_leak`, `dataset_artifact`, `grader_exploit`,
  `environment_tamper`, `task_bypass` and `train_overfit` (a general rule learned from
  training failures is not overfitting). An `exploit` is a change whose benefit depends
  on a property of how the data, the grader or the environment was built, which the
  judge names (`property`), and whose trigger has nothing to do with doing the task; an
  exploit that names no property is read as `suspect`. The judge is told to say
  `suspect` when unsure, and given legitimate counter-examples (general know-how such as
  "re-check each intermediate result before using it"). Evidence items that
  are not strings are kept as JSON text; a reply that is not valid JSON is read field
  by field and marked `salvaged` -- also when the first object a reply holds is not the
  verdict (a quoted `{}` inside a reply that is not valid JSON).
- **What it does: a policy per category**, `off` (not asked about), `warn` (recorded
  and shown; the change is still evaluated) or `veto` (refused before evaluation with
  `E-integrity`, which the next request shows with the judge's reason). The veto rule:
  the verdict is `exploit` or `suspect`, its finding is not `inherited` (a missing or
  other `source` counts as about what this change adds), and it names a category whose
  policy is `veto`. These never veto: an `inherited` finding (recorded and shown as a
  warning: a harmless child of a flagged parent is evaluated; only exactly
  `inherited`, in any case or spacing, counts), a category whose policy is `warn` or
  `off`, a verdict that names no category, a failed call, a malformed or salvaged
  reply: a failure is recorded as a judge error and shown as such.
- **What slipped past once is not vetoed later.** A change that got through -- the
  judge failed or its reply was salvaged, its category only warns, the judge was off
  when it was proposed, or the finding was about inherited content -- stays in the
  candidates built on it, and their proposals are not vetoed for it (an inherited
  finding never vetoes). Only the final audit of the winner's combined change from the
  seed reports it.
- **Setting it.** `--integrity-judge on` (or `veto`): every category `veto`.
  `--integrity-judge flag` (or `warn`): every category `warn`. An object sets
  categories one by one, in a profile or as JSON on the command line:
  `--integrity-judge '{"mode": "on", "policy": {"answer_leak": "off", "train_overfit": "warn"}}'`.
  A category's policy, most specific first: the run's setting for that category; a
  Task's allowance for it (`off`); the run's default (a bare value, or the object's
  `mode`/`default`), which covers every other category and overrides the Tasks'
  policies, so no Task raises a run's `flag` to veto; and, only when the run names no
  default (an object with just `policy`), the Tasks' policy, else `veto`.
- **What a task declares.** A Task (`refactory_dsh/task.py`) may set
  `integrity_allowances = {category: "why it is legitimate here"}`, e.g.
  `{"answer_leak": "the task is to recall facts about these items"}` or
  `{"grader_exploit": "the output format is the objective"}`: that category is off for
  the run unless the run names it, and the judge is told the reason under "What this
  task allows". `integrity_policy = {category: off|warn|veto}` gives a category's level
  when the run names no default. Where two Tasks of a graph disagree, the more lenient
  wins: an allowance beats any policy, warn beats veto. An unknown category or policy
  word stops the run.
- **Where the verdicts show.** The change ledger gains an `integrity` column (the
  verdict of the proposal that was evaluated; `(inherited)` marks a finding about
  earlier content); the request gains "Integrity review of your proposals": a tally,
  then the newest 8 suspect or exploit verdicts, each with its categories, quote, reason
  and whether it was refused (within 4000 characters) -- a refusal (an exploit or
  suspect verdict about what the proposal added) does not count as progress, a verdict
  that was only flagged (a warn category, an inherited finding) asks whether the rule
  holds in general; the
  report (`integrity` in `report.json`, "Integrity review" in `report.md`) gives the
  policy, the counts, every flagged proposal and the winner's combined-change verdict.
  The final verdict is reported, never acted on.
- **The graph-run loop** (`loop_mode`) takes the setting from its frozen
  `integrity_judge` knob (a bare word or a JSON object as text); a malformed value is
  refused naming the knob.
- **SkillOpt** proposals are judged, each knob's patch separately, and can be vetoed;
  SkillOpt's own requests do not show the verdicts. A SkillOpt run resumed inside an
  epoch's patch phase with the judge newly turned on judges the rebuilt patch; if it
  vetoes it, the rebuild fails as any refused rebuild does -- resume with the setting
  the run was started with.
- **Offline.** `python -m refactory_dsh.optim.integrity judge --proposal FILE` judges
  saved proposals with the same prompt a run builds (`--print-prompt` prints them and
  calls nothing; `--model`, `--provider`, `--worker-backend`, `--effort` pick the
  route). FILE is a JSON object, a JSON list or JSON lines; each entry carries `edits`
  (a list, or one `edit`) and optionally `parent_state` (with it the state diff is
  computed), `state`, `seed_state`, `rationale`, `prediction`, `diagnosis`,
  `train_records`, `mode_name`, `parent`, `candidate`, `integrity_judge` (the run
  setting, `on` by default), `task_policy`, `allowances`, and an `id` and `label`
  echoed in the output. A `"scope": "cumulative"` entry carries `seed_state`, `state`,
  `steps` (`{"id", "edits", "rationale"}` in order) and `train_records`. One JSON line
  per entry comes out: `id`, `label`, `verdict`, `categories`, `evidence`, `property`,
  `reason`, `vetoed`, `error`. `python -m refactory_dsh.optim.integrity export RUN_DIR`
  writes a past run's proposals in that format: for a run with the judge on, every
  judged proposal (vetoed ones and SkillOpt's per-knob patches included) from its
  verdict record, with the run's setting and the Tasks' declaration, the parent's
  train records as they stood then and the live verdict (`live`), so each entry
  rebuilds the run's own prompt; for a run without it, the optimizer's candidates.

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
- **Reads are produced first (V6, V15).** A file a node reads is written by a node on a
  path from the start to a terminal. An op or a gate may read a file that is not there
  yet (an optional read), unless it lists the read in `required_reads`: a required read
  is written upstream on every path to it, and both runtimes refuse to run the node while
  one is missing. An agent's inputs are all required: its reads, and every `{{name}}`
  its prompts fill in that is not a knob, not in its `feedback_keys` and not inside an
  `{% if name %}` on the same name, are there on every path to it. A data pass's own
  files and facts stay in the pass. A `tool_policy: none` agent declares exactly one
  write. LiveMath's skill graph declares every read of its four ops as required.
- **Passes are closed (V12).** A data node's entry reaches its exit, which is not a gate
  and has no out-edge. A pass is entered only through its data node, never by an edge
  from the data node or from outside, and never leads back to it.
- **Verdicts are unambiguous (V5).** No two out-edges of a gate carry the same verdict.
- **Each agent runs its own prompt.** An agent node an edit adds gets its own
  `<id>__system_prompt` knob, a copy never binds the original's knob (`E-binding`), and
  every declared knob is read (V7).
- **Items reach the nodes without their answer.** A Task names the item fields only its
  grader may see (`Task.grader_fields`, default none). `current_item.json`, the file the
  graph's nodes read, never carries them, on either runtime; the grader gets the whole
  item. A field in `Task.revealed_by` is shown when the run's knob for it is on.
  LiveMath hides `correct_choice`, `label`, `label_text`, `theorem` and `sketch`; its
  frozen `use_theorem` and `use_sketch` reveal the theorem and the sketch.
- **New agents run without tools.** An agent node an edit adds gets `tool_policy: none`
  unless it asks for tools (`tool_policy: null` or a `tools` list), and asking is refused
  `E-new-node-tools` unless the run allows it (`LoopConfig.allow_new_node_tools`, off by
  default). A node whose id the seed has keeps the seed's policy.
- **Protected nodes are untouched.** A mode names them in `Mode.protected_nodes()`
  (default none). An edit that removes one, changes its declaration or out-edges, moves a
  knob it reads, or lets a run finish without one that every run used to visit is refused
  `E-protected`. A new node may still go in front of one. A proposal's edits are judged
  together on the state they leave, so "remove a step, then put a code op in its place"
  is accepted even though the first edit alone would let a run skip a protected node;
  a bundle that does leave one skippable is refused, naming the bundle and the edit
  from which no later edit restores it (`[the n edits together; from edit k (op) on,
  ...]`). The optimizer's request lists them.

### Code nodes, verdict gates and `parallel`: composing the four primitives

The graph has four kinds of step: agents, code, gates and forks. An optimizer composes
them freely through edits. Before this change, a new op or fn gate could only reuse a
`module:function` that an existing node already named (V8). So an edit could not add code
or a gate, and a 2-branch fork took about six edits against a default cap of four. Three
generic callables in `refactory_dsh.snippet` and one macro close that gap:

- **Code op** (`"call": "refactory_dsh.snippet:run"`, with `code`): Python that defines
  `main(inp)`. `inp` is `{"files": {each declared read that exists: its text}, "facts":
  {each name in its inputs: its value}}`. `main` returns `{"facts": {...}, "files":
  {...}}`, and each one must be declared in the node's `produces` / `writes` (anything else
  fails the node). An edit that adds a code op binds its code to the node's own
  `<id>__code` knob, the way a new agent gets `<id>__system_prompt`. So one edit adds the
  node and its code, and a later `set_knob` rewrites the code alone.
- **Code gate** (`"evaluator": "refactory_dsh.snippet:gate"`, with `code`): `main(inp)`
  returns `{"verdict": <one of the gate's out-edge conditions, or halt>, "reason": <text>}`.
- **Verdict gate** (`"evaluator": "refactory_dsh.snippet:verdict"`, no code): the verdict
  is read from the gate's first `required_reads` file, or else its first `inputs` fact.
  That is whatever an agent or a code op upstream wrote there. Leading markdown or quotes
  (`**`, `#`, `>`, `"`, `:`) are skipped, and the word is matched against the
  out-edge conditions in any case. The verdict is the text's first word when that is an
  answer, and the rest of the text is the reason. Otherwise it is the last non-empty
  line, when that line is an answer alone (`proceed`, `proceed.`, `**proceed**`) or an
  answer then `:`, `—`, `–`, `-` or `.` and more (`> reloop — show the step`); the reason
  is the text before that line, then what follows the separator. That is the usual
  "reason first, verdict last" review. Nothing in between is read: "Do not proceed" names
  no verdict, and neither does "I would not proceed; reloop", nor a last line "reloop if
  unsure". A line that is a list item (`- Halt: no`, `1. proceed`) is never a verdict,
  first or last, and neither is a code fence (a verdict inside a fenced block). When
  neither the first word nor the last line is an answer, the node fails with an error
  naming both and the answers; it never falls back to a default (in an outer-loop run,
  that item's run fails). So does a text whose first word and last line name different
  verdicts, alone or with a reason ("Halt? No. ...", then `proceed — nothing to fix`):
  neither is picked. This is how an agent decides a gate: a reviewer
  agent writes `proceed` or `reloop`, followed by why, or its review and then the word on
  the last line.
- **Answers a gate may give:** each out-edge condition, and `halt`. V5 refuses a code or
  verdict gate whose out-edge condition is not a single word (`needs work`, `ok.`), so the
  edit is refused instead of every run failing when the gate answers. The
  router (`route_verdict`, the spine's `Spine.verdict`) takes `halt` with no halt edge as
  the end of the whole run, so a code or verdict gate may answer it too. `proceed` with no
  proceed edge is not offered: the router would send it down an unconditional edge, and a
  gate has none (V5).
- **`parallel`** (a macro with fields `target` and `branches`): 2-8 new agent or op nodes
  run at once after `target`, between a fork and a join. The join leads on to target's old
  successor. Branches must write distinct files and set distinct facts; otherwise the edit
  is refused before anything runs, and V16 holds the result. The successor's declarations
  do not change, so nothing reads the branches' outputs until an edit says so: insert a
  node after the join that reads their files or facts, or use a branch's fact as
  `{{name}}` in a later step's prompt knob. Each branch gets its own knobs, as any new
  node does. The fork and join ids (`<target>_fork`, `<target>_join`) avoid every node
  and branch id.

A reloop delivers the gate's reason to the step it sends control back to, as feedback.
This is the existing `with_feedback` path, on both runtimes. The gate's `max_iterations`
bounds how often a reloop can happen (V10). A gate asked to reloop once more than that
does what its `on_exhausted` says, the same on both runtimes (`Graph.on_exhausted`, which
the lowering hands to the spine):

- `continue` takes the gate's `proceed` edge, so the item goes on with its last attempt
  and is graded as it stands. V5 refuses it on a gate with no `proceed` edge.
- `fail_item` fails only the data-pass item the gate is in, as a failed node does: the
  item gets a failed record and its siblings and the run go on.
- `halt` halts the whole run, so no later item runs.

Undeclared, a gate inside a data pass continues (or fails its item when it has no
`proceed` edge), and a gate outside one halts, as before. Outside a pass there is no item,
so `fail_item` halts there too. So a check-and-retry loop the optimizer adds can never
stop the evaluation of a whole candidate; only a graph that declares `halt` does that.
Design mode's gates are outside any pass, so they halt as before.

`loop` also works on the last step of a pass (a step with no out-edge): it first adds a
no-op step after it (`refactory_dsh.nn.loops:loop_end`, which becomes the pass's exit),
and the gate's `proceed` edge leads there. Each of the three callables is an ordinary
node callable, and both runtimes invoke it the same way: the spine through
`python -m refactory_dsh.run`, the walker in its forkserver child. The node's declaration
(code, inputs, labels) is built from the graph in one place (`lowering.snippet_spec`), so
the two runtimes cannot disagree.

**The sandbox.** The sandbox is the security boundary. A code node's code runs under
these limits:

- `unshare -Urnmp --fork --kill-child --mount-proc`: its own user, network, mount and PID
  namespaces. There is no network, and a fresh `/proc` shows only the sandbox's own
  processes, so no host command line or environment is readable.
- `no_new_privs`, set before anything else runs.
- An empty tmpfs over `/home`, `/root`, `/tmp`, `/var/tmp`, `/dev/shm`, `/run` (and
  `/var/run`), `/var/lib`, `/mnt`, `/opt`, `/proc/sys`, and the run and project
  directories wherever they are. So no service socket and no person's or run's file is
  reachable.
- `/usr/bin/python3 -I -S`: the standard library only.
- JSON on stdin and stdout.
- An environment of `PATH`, `HOME=/tmp` and the language settings only: no knob, no node
  manifest, no credential.
- Limits of 512 MB of memory, 10 CPU seconds, 1 MB files and 32 processes. A limit that
  cannot be set fails the node.
- 10 s of wall time for the code, counted once the sandbox has started (its driver says
  so first). After that the whole process group is killed, and the PID namespace with
  it, so nothing the snippet started outlives it. Starting the sandbox has its own
  allowance of 60 s (`snippet.STARTUP_S`); running over that is the sandbox being
  unavailable, reported as such, not the code running over.

The snippet never gets the context, the run directory, other facts or the knobs. The
wrapper writes its outputs through the context, so declared writes and facts are still
enforced. The host's `/etc` and `/usr` stay readable (the interpreter needs them).

The sandbox fails closed. It is unavailable when `unshare` or the system Python is
missing, or when the probe snippet does not run (`snippet.available()`). In that case a
code node fails, the request does not describe code nodes, and an edit that adds one is
refused `E-code-nodes`. The probe's answer, passing or failing, is kept for 5 minutes
(`snippet.PROBE_CACHE_S`) in a per-user file every process reads, so a busy machine pays
one probe, not one per op; and one request's answer decides both whether it describes
code nodes and whether it refuses one (`ProposalContext.sandbox`). The answer only says
whether to try: code still runs only inside the sandbox. A failed code op is retried once, as every op is: always on the
spine, and on the walker when recovery is on. A snippet is pure, so the retry changes
nothing but time.

**Checks before anything runs.** V18 is defence in depth; the sandbox is the boundary.
It checks the code statically, without running it. The
code must parse, define `main` and be at most 4096 bytes. It may import only `re`,
`json`, `math`, `collections`, `itertools`, `statistics`, `fractions`, `unicodedata` and
`string`. It may not use `open`, `exec`, `eval`, `compile`, `__import__`,
`getattr`/`setattr`/`delattr`, `globals`/`locals`/`vars`, or any dunder name or dunder
attribute. Attribute names are allowlisted: the public methods of the builtin value types
and of what the allowed modules hand out, plus those modules' public names that are not
themselves modules (`re.enum`, `json.decoder` and `statistics.random` are refused). String
formatting is refused (`format`, `format_map`, `string.Formatter`, `string.Template`),
because it reaches attributes by name where the dunder ban cannot see it. `inputs` join V15 (each one is produced upstream on every path) and V17 (they
count as consumed, so grader-derived inputs are refused). An edit whose code holds a value
only the grader may see is refused `E-secret`, without echoing the value. Such a value is
a train item's grader-only field, such as its answer (`task.grader_secrets`), and the
comparison ignores case and whitespace. The scan is best effort: it catches a value
pasted into code, not one rebuilt from pieces. It skips values under 8 characters, such
as a single-letter choice label, because that would refuse ordinary code. The control
that holds is that grader fields never reach a code node's inputs: V17 refuses
grader-derived inputs, and items reach nodes without their grader fields. The scan covers
code only; scanning prompts is a follow-up.

**Turning it off.** Code nodes are on by default wherever the run permits structural
edits. `--allow-code-nodes 0` (`LoopConfig.allow_code_nodes`) refuses every new code node
with `E-code-nodes` and leaves code nodes out of the request. Verdict gates and `parallel`
stay, because they run no optimizer-written code. A run that keeps the default produces
the same digest as before, so it still resumes.

**What the optimizer is told.** The request's "How nodes compose" section
(`optim.base.composition_section`) describes the four kinds: what each reads and writes,
how a gate gets its verdict, and how loops and forks are formed. It adds at most two short
examples built from this graph's own ids: an agent-verdict loop around a step, and a code
op that computes a fact for a later step. Code nodes appear in the section only when the
run offers them. The section prescribes nothing: the examples are compositions, not a
menu.

### The invariant harness

`tests/structure_fuzz.py` checks those promises by generating random structure edits
from a seed, over LiveMath's skill graph and seven small graphs
(`tests/structure_fuzz_graphs.py`: a fork and join, a nested fork, the three gate kinds,
a gate loop, a data pass, a pipeline of ops, and several terminals). The edits cover
every primitive and macro, node copies taken from the node JSON the optimizer is shown
(new ids, new prompts), and malformed edits. Each walk applies them through the same
attempt path, and every accepted state is checked two ways.

- **Observed**, by running the real code and looking at what it did: lowering; a walker
  run with a scripted model, which must end at a terminal or a halt; the spine beside
  the walker on a sample (same facts and item scores; graphs with a user or an agent gate
  are walker-only, as the test drain answers no gate); no agent sent an unfilled
  `{{name}}`; no node ran without one of its `required_reads`; and no grader-only item
  value in any file a node can read or any prompt, found by content. The synthetic ops read their declared inputs; an op's input that is
  not there is counted (`notes.op_read_absent`), not called a violation, since V6 lets an
  op read an optional file.
- **Spec**, a second statement of the rules beside the validator: knob ownership,
  dataflow, fork pairing, protected nodes, new agents' tools and code nodes (each added
  code node's code parses, defines `main`, carries no hostile template, and is bound to
  its own knob). The generator writes benign code ops and code gates, hostile code that
  V18 must refuse, and code that passes V18 and fails at run time on purpose (counted as
  `notes.code_node_failed_at_run_time`); `parallel` is generated with 2-3 branches, some
  racing on a file. A code op's facts are compared on both runtimes by the spine check.

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
