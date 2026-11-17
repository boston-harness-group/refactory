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

## The default loop's validation gate (on whenever there are val items)

When the run's parts manifest (`parts_manifest`) has val items, the default loop
evaluates candidates the way the SkillOpt strategy does, using the same
`ValidationGate` and `FinalEvaluation` from `outerloop.py`; only the optimizer, where
edit ideas come from, differs. It is not a setting: there is no flag or profile key,
and the run decides from its manifest. `config.json` records the decision (`val_gate`),
and the report's `selection` section says why (`val_gate_basis`). A profile saved with
a `val_gate` key (before the switch was removed; `false` was the default then) still
loads: the key is dropped.

- **No val items.** A run with no parts manifest, or one without a val part, selects
  on train scores and says so: "no validation items: selecting on train scores", in its
  events, its `strategy.log` and its report (`selection.warning`).
- **`holdout`.** The validation set for selection is always the manifest's val part.
  The `holdout` fraction of a Task's items (a run without a manifest) is never
  selected on: it stays the end-of-run audit's independent split. With a manifest the
  audit's held-out items are the val part, so under the gate the audit runs on the part
  the search selected on. It then gives no overfit verdict and rolls nothing out: it
  says that no independent check was possible (`holdout.split` is `val`,
  `holdout.independent` is false, `status` is `not_applicable`, and the report names
  the split). With a test part the audit compares val with test, which nothing selects
  on (`split` `test`). A resumed run keeps the `selection` and `pool_size` its
  `config.json` recorded, whatever its profile or the defaults now say; a resume
  command that gives one of them explicitly with another value is refused, naming both.
  A run that recorded `val_gate` false although it had val items (started with the old
  `--val-gate 0`) is refused on resume: start a new run.
  A launch that leaves `selection` unset records the rule it resolved (`single` or
  `pool`).
- **Custom strategies** (SkillOpt) keep their own gate: the default loop's stays off
  for them.

Train is the proposer's feedback, validation decides every choice and test is scored
once at the end. The validation average is the fitness for the parents, the pool and
the winner, in every strategy, optimizer and parent picker; the proposer never sees a
val score. `selection` says how it selects:

- **`single`**: SkillOpt's line. A candidate is kept only if its val score strictly
  beats the incumbent's (the gate's current candidate), and the incumbent is the only
  parent. The default for the default loop with a single-parent optimizer (`random`,
  `autoresearch`, `hill_climb`, alone or as `parent_picker: hill_climb`), the
  SkillOpt-matched setting; its decisions are those of the gate before pools existed.
- **`pool`**: entry is permissive. Every judged candidate may enter a pool of the
  `pool_size` (8 by default) best by val score, with any tied with the last of them.
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
  `portfolio` round's one per edit level) are judged together, each against the
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
  `train_minibatch` sample) with that epoch's seed. A candidate created and scored
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
  keep the `max_concurrent_evaluations` slots: the re-score joins the round's last
  wave when that wave leaves a slot free (two portfolio children and three slots, the
  live shape), else it runs after the children, alone. Either way it is a request of a
  `Run.evaluate_many` batch: its items count against `--max-items-in-flight` with the
  children's, and a stop ends it with them (a child's error is the one raised). It goes
  in only when there is room for every child still to score and it: `--max-evaluations`
  must hold all of them, and `max_rollouts`, when set, each child's train and val
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
  seed); a pair from different epochs is marked as such. It is always on under the
  gate (there is no switch). On LiveMath's
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
  held-out audit then compares the winner's val and test scores, so a `holdout: 0` run
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
- **Early stop.** `early_stop_patience: N` (0, off, by default) stops the run after N
  epochs in a row whose gating did not raise the best val score.
  `early_stop_at_full_val` on stops it once the best val score reaches 1.0. The first
  epoch gates the starting population, which sets the best the count starts from
  (SkillOpt's initial val score) and is never a reason to stop. Each later epoch gates
  the batch the previous one proposed, and is checked right after that gating, before
  it proposes, as SkillOpt checks at an epoch's end; so a seed already at 1.0 still gets
  one batch proposed and gated. Both settings need the gate (val items), are saved with the
  run and are kept on resume.

## An evaluation that dies with the run is not a result (exit 75, `--resume`)

A failed item is the candidate's result only when it ran and failed. An item whose
process died with the run has an unknown outcome (`items.interrupted_item`). That is
an item whose model calls kept failing on the wire through the client's retries (a
proxy or provider outage, or a network outage longer than the client waits one out:
2 hours, or its node's wall; `models.md`), whose completion process was killed by SIGKILL or SIGTERM
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
  recorded: no score, records, val occasion, re-score or rollouts. What `refactory
  optimize ... --resume` does then depends on what stopped:
  - **A train evaluation:** the candidate stays unevaluated and its epoch open, so the
    resume evaluates it again from the start and gates it as any other.
  - **A val or test rollout:** the candidate's train score stands. The interrupted
    rollout is no val or test score, and the resume rolls it out again in a fresh
    occasion.
  - **A fresh-evidence re-score:** the parent keeps its earlier records. The resume
    tries the re-score again only while that epoch's children are still unevaluated
    (a known gap, listed as a follow-up).
- **The cap.** Each stop counts for that candidate and part (a re-score counts per
  epoch), from the journal, so a resume keeps the count. An in-run retry is not a stop.
  The run stops on the first two. On the third (`INTERRUPTION_CAP`), the evaluation is
  recorded as failed, with the error "`<part>` evaluation interrupted 3 times; recorded
  as failed", and the run goes on. An evaluation that dies every time therefore cannot
  hold a run in a stop-and-resume loop.
- **Receipts are kept.** When an unevaluated candidate's train evaluation or an epoch's
  re-score runs again, in a retry or after a resume, what the earlier attempt left (its
  per-item passes and receipts) is moved to `.refactory/archive/<UTC time>/runs/<id>/...`
  inside the run (`evaluation_archived` in the journal), never deleted.
  - A train attempt's archive leaves the candidate's `occasions/` in place.
  - A val or test rollout retried in the run is archived from its own occasion folder.
    After a resume it runs in a fresh occasion, and the interrupted occasion stays
    where it was.
  - A re-score's folder (`occasions/train_epoch_<n>`) is archived whole.
- **Seed results.** A `--seed-results` part with such an item is a miss: it is
  measured instead.
- **Runtimes.** The item records come from the same code on both runtimes, and a model
  call fails with the same text in the spine's worker as in the walker. Known gap: when
  the spine's own worker or op process is killed and leaves no stderr, the spine records
  only `exited with code null`, which names no signal. Such an item is still counted as
  the candidate's failure on the spine, where the walker counts it as interrupted.
- **Out of scope.** An interrupted evaluation runs again from its start. Resuming inside
  forks, loops and per-item passes is separate work.

## Train minibatches (`train_minibatch: N`)

By default every candidate is scored on the whole train part. With
`train_minibatch: N` and a train part of more than N items, each epoch samples N train
ids with SkillOpt's per-epoch seed (`train_batch_seed`). It needs the gate (val items):
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

## Compound proposals (up to 4 changes)

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
  state, so the same point reached in another order is a duplicate). A `set_knob` that
  gives a knob the value it already holds is dropped, and said (see "Wiring blocks,
  repair and checks before scoring" below). Any other edit the result does not depend
  on -- one that changes nothing, or that a later edit overwrites or undoes -- is refused
  `E-noop` with its index. A refusal about one edit names it,
  `[edit k of n: op]`, after the rule's code; one about the combined state names none. A
  list that makes more changes than the cap is refused `E-too-many-edits`, saying how
  many changes and edits it makes and how they are counted.
- **Cap.** 4 changes per proposal, SkillOpt's edit budget (`MAX_EDITS_PER_PROPOSAL`, a
  constant, not a run setting), counted by logical change (`optim/edit_count.py`): a
  structural edit and the edits that only connect it count as ONE change -- the new (or
  removed) step's own prompt and settings (an `add_knob`/`set_knob`/`remove_knob` of a
  knob it ALONE binds or reads; a knob another step binds or reads too counts on its
  own), its DIRECT neighbours' prompt lines that refer to it (a
  `set_knob` of a knob a step one hop away binds or reads -- reading what it writes,
  writing what it reads, or on an edge with it, in the graph the structural edit itself
  leaves -- whose every changed line names the
  step or a file or fact it writes), and edges between it and those neighbours. Every
  other edit counts as one more, and two structural edits are two changes. Only one-hop
  neighbours qualify, so one change cannot sprawl over the pipeline. A limit of one (a
  caller's `max_edits=1`) still means single edits.
- **Counts per candidate.** Each candidate's proposal record (journal `proposal`,
  report.json `proposal_outcomes`) carries `edit_count`: its changes (`logical`), raw
  edits (`raw`), structural and knob edits, and the knob text characters it changed
  (the lines added and removed); report.md's "Proposal versus outcome" shows them.
- **Who emits lists.** The `autoresearch` request says it may bundle up to 4 changes
  when a fix needs coordinated changes, and how they are counted, with a JSON example.
  When it
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
  next request, never dropped. An empty `proposals` list proposes nothing. An answer
  whose outermost object is not valid JSON (a list left open, a reply cut off) is read
  as nothing: no object nested in it (one edit of one proposal) is taken for the
  answer. The reply is read by its own braces first (so a code fence inside a proposed
  prompt cannot cut it), the last top-level object with a `proposals` LIST being the
  answer (an object whose `proposals` is not a list, or an example `proposal`, is not);
  after a broken answer, a fenced object is read only when it is a whole answer (a
  fenced example edit before a cut-off answer is not the answer). Only when no object
  with a `proposals` list parses anywhere in the reply is it read as a top-level array
  of objects (`[{"...": ...}, ...]`, bare or fenced, a bracket then a brace then a key):
  its LAST such list (a draft list before a final one is a draft), every entry; a list
  holding anything else, or left open, is refused whole, and so is a reply with a JSON
  object after its last list (which is the answer is unclear). An object inside or after
  such a list is never read alone as the answer. Brackets in prose (`[{op, knob}]`) and
  an example list before an answer are not such a list. Each proposal the request asked for is refused `E-proposal-shape`, saying
  where the JSON broke, so the round asks again within its attempts and the refusals
  are counted.
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
A record also keeps, under `steps` and in run order, each op visit of the item's pass
(the facts it returned and the declared files it wrote: a 600-character preview, a
digest and whether all of it was empty) and each code-gate visit (its verdict and
reason); `exchanges` stays one entry per model call. Both runtimes record them (the
spine through its runner, the walker in memory).

- **Order: the pipeline first.** The request leads with what the optimizer may change:
  after the score line and what the score measures, "The pipeline you may edit" (one
  line saying the structure and each step's settings are levers of the same standing, or
  that the run permits settings only; then the nodes, edges and what each node has
  before it), "Where the pipeline fails, step by step" (below, when the analysts or the
  readers ran), "The operations you may use" and "How nodes compose". Only then "Each
  step's settings" (every knob with its exact current value, which a patch quotes: a
  text with a line break or a `|`, which a table cell cannot show as it is, is listed
  there by reference and shown whole in a fence after the table), "Each step in full,
  and the node schema", the author's guidance and "The rules the validator checks",
  introduced neutrally: a proposal that breaks one comes back with the rule's code and
  reason, to be fixed in a later answer, at the cost of the answer only. Nothing is
  dropped by the order; the evidence sections follow as before. All of this is the
  shared request (`ProposalContext.render`), so every outer loop that asks the model
  proposer gets it: the default loop (alone, in a portfolio or under a parent picker),
  `best_of_n`, `reflective`, `reflective_sampler`, `adaptive_meta`, `tree_search`,
  `staged_curriculum`, `research_then_optimize` and `autoscientists`. The SkillOpt
  package strategy writes its own text-only requests and is unchanged.
- **Answer rules.** Built from the run's edit catalogue and the node schema, with
  placeholders only. When the run permits a structural op, the rules show a structural
  example (an op that puts a new node after an existing step, its `node` built from the
  schema's required fields) beside the parameter-move example, and the bundling example
  shows two prompt knobs first and then that node together with its own
  `<new step id>__system_prompt` knob, so neither kind leads; a run that permits
  parameter moves only sees no structural example. A text parameter such as a prompt is
  changed by a patch by default, as SkillOpt's patch mode edits its skill only by local
  append, insert, replace and delete patches: the rules say to patch the parts the
  evidence says to change and keep the rest, and the bundling example patches its two
  prompt knobs. A whole new `value` stays available, for a setting that is not text, a
  new step's own prompt, or a prompt that has to be replaced outright. The rules ask for
  each change to be grounded in the evidence, kind-neutrally: a setting, a prompt or a
  structural change, whose effect the optimizer can state -- the failure it addresses
  (the step where it starts), how it changes what a step sees or how the work is done,
  and a checkable prediction -- the kind chosen by the diagnosed cause, neither kind the
  safer default. Nothing asks for structure for its own sake. A new node is written from
  the node schema; the mode's blocks, when it declares any, may be used verbatim, and a
  mode with none says so ("New nodes"). The size cost states the run's parsimony per
  node; only when every train evaluation's score was the mean of its items' scores
  (the loop checks) does it add what one item is worth (1/N), since a mode's score may
  be any metric of the run; otherwise it says to compare it with the score differences
  between candidates.
  The template carries no task or domain vocabulary (`test_proposal_presentation`).
- **What the score measures.** Said from the run's settings (`objective_text`), under
  the request's first line and in every analyst, merge and memory call. With
  `parsimony` 0: the score is the task's own score and nothing else; the loop adds no
  cost term, so inference cost, latency and the number of steps or model calls are not
  part of the score unless the task's own score measures them. When, in addition, the
  score is the items' mean, the "Evidence from the traces" table leaves out its `cost`
  row (seconds and inner model calls per candidate), which gradient.json, the journal
  and the report keep. With `parsimony` above 0 it states the size cost as scored (raw
  score minus parsimony per node); the `cost` row stays. Under the validation gate,
  either way, it says how the gate decides (the analyst, merge and memory calls carry no
  gate note of their own): on the task's validation score, which carries no size cost,
  keeping a candidate only when it beats the current one; a train score only breaks a
  tie between candidates that beat the current one by the same validation score, and a
  tie with the current candidate is rejected. The proposal request (not the analyst,
  merge or memory calls) adds the run's **compute policy**, only what the score text
  does not already say: with `parsimony` 0, cost is not a reason for or against a change
  (and, when the score is the items' mean, per-item cost is reported separately). With
  `parsimony` above 0 the size-cost lines stand alone. Nothing steers toward any kind of
  change. The validation gate's rule is said here once; the results' gate note points to
  it.

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
  `role_routes`: `{"optimizer": {"worker_backend": "claude_code"}}`. It never runs
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

"Evidence from the traces" shows the gradient without the noise. Its first row,
"structural changes", says in plain words what the run has done with structure: how many
candidates made a structural change and how many did not, how many structural proposals
the validator refused, the operations used, and, when none was tried, "No structural
change has been tried in this run yet" (a fact about the run, not a judgement of
structure; `structural_yield` stays in gradient.json and the report). Before the loop's
first gradient (a strategy's first request) the row is counted from the evaluated
candidates the request lists, when that list is whole. The leaderboard copy
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
  rises and drops among the rest. Every changed item is shown whole, parent beside
  child: the parent's trajectory, then the child's verdict and each step's reply, a reply
  the same as the parent's said so and the input not repeated; size is never a reason to
  leave one out (only a single text over `ITEM_TEXT_MAX` is cut, and said). It also
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
calls in `skillopt_trace.jsonl`, one line per attempt (`attempt`, from 1): as SkillOpt
does, a failed optimizer call is tried up to 3 times, waiting 1, 2 and 4 s after each
failure, before its fallback; a spent budget, a cancellation or a missing model choice
is never retried. The loop's own analyst, merge, memory and step-review calls are
retried the same way (`AutoresearchOptimizer._complete`): up to 3 attempts, waiting 1,
2 and 4 s after each failed one, any exception from an attempt counting, each attempt
traced with its `attempt`; a spent budget, a cancellation or a missing model choice
stop at once. So is the proposal call (`AutoresearchOptimizer._ask_proposal`), where a
reply with no JSON object at all (empty, or prose only) also counts as a failed attempt;
a reply whose JSON object is broken is an answer, refused and asked again with why
(`E-proposal-shape`), not an attempt retry, and a proposal call whose attempts all fail
ends the round as a failed one, as a failed proposal call did before (it used to have
one attempt, so one transient failure lost the round's proposal). Otherwise a reply is
never retried, however unusable: the analyst, merge, memory and step-review calls parse
what they get, and the proposal call retries only a reply with no JSON at all; a call
refused as longer than
the model's context is not tried again at the same length but once at half of it, and
not at all when the request cannot be made shorter. A retry's wait ends early on a stop
(the client's cancellation scope, or a person's `refactory stop` request, checked every
0.25 s), and the call is not attempted again. A person's stop then ends the search as a
stop at an epoch boundary does: the stop is the run's bound (journaled, with
`stopped_inside_step`), report.md is written, and the open epoch's step -- the memory
pass, or the proposal round, which is not counted as a failed one -- is made again on
resume. In the memory pass, a cancelled client scope or a missing model choice ends the
run with its error, after journaling it as the bound and writing the report; in a
proposal round they are counted as that round's failure, as any failed proposal is. These attempts stack
on the model client's own: a `DshModelClient` completion already makes up to 3 attempts
on a transport failure (`model.DEFAULT_ATTEMPTS`), waiting up to 60 s in all between
them (`RETRY_WINDOW_S`), so one analyst, merge, memory, step-review or proposal call may in the
worst case start 9 model sessions, each up to the client's own timeout, and wait up to
3 x 60 + 1 + 2 + 4 = 187 s between them. A network, DNS or credential-refresh outage
uses none of these attempts, the SkillOpt port's or the loop's: the model client waits it
out inside the attempt (up to 2 hours, `models.md`), and the SkillOpt port's trace line
lists the outages an attempt waited out as `network_waits`. The bound is waited once, not
once per attempt: after a call has waited the whole 2 hours, an outage met by the next
call (that call's next attempt included) fails at once until some call gets through. So
the worst case of one optimizer call in an outage that never ends is about 2 hours plus
the attempts' own waits (1 + 2 + 4 s), not 3 x 2 hours.

### Item traces as files (`traces/`, written, not read)

After every evaluation (a candidate's train evaluation, a re-score, a memory rollout, a
val or test rollout, the holdout audit), on either runtime, each item's whole record is
also written as files under the run root (`refactory_dsh/traces.py`, called from
`Loop._measure`), as SkillOpt keeps `predictions/<task>/conversation.json` per rollout,
with one file per step since a candidate is a multi-step graph:

```
traces/<candidate id>/<part>-<occasion>[-seed<seed>][-eval<n>][-repeat<n>]/<item id>/
  NN-<node>.json   one per model call, in run order: node, prompt and response, whole
  steps.json       the op and code-gate visits, when the item has any
  item.json        the record without its exchanges and steps (the original id in "id"),
                   written last: an item folder with it is complete
```

- **Names.** The occasion is the evaluation folder's name without the part (`epoch_3`
  for a re-score, `0` for the first val rollout, `memory_epoch_2_<key>`), and is left
  out for a candidate's own train evaluation and for the holdout. An item id a file name
  cannot hold as it is (`/`, `:`, `..`) is renamed as pass folders are
  (`items.sanitize_item_id`, with a digest of the original).
- **One folder per recorded evaluation.** `-eval<n>` numbers the second and later
  evaluations the journal records in the same evaluation folder (`score`, `rescore`,
  `rescore_failed`, `memory_rollout`), such as two train rollouts of one candidate at
  one seed (SkillOpt's slow update). The count is taken from the journal's events, live
  and on resume, and nothing new is journaled. An attempt the journal has not recorded
  (an in-run retry, or a resume running an interrupted or budget-stopped evaluation
  again) rewrites its own folder: each file atomically, with the call files it no
  longer has removed. A fresh run in the same directory removes `traces/`, as it starts
  the journal over.
- **Failures.** An item that cannot be written is skipped, the others are written, and
  the evaluation logs one warning line. Text that cannot be encoded (a lone surrogate)
  is written escaped.
- **Leftovers.** A process killed mid-write can leave a hidden `.<name>.*.tmp` file
  beside the file it was writing. A rewritten evaluation that has fewer items than the
  attempt before it leaves that attempt's other item folders in place.
- **Size.** About 44k characters per LiveMath item. Train traces repeat what the
  journal's `train_records` already holds, so a run's train records take about twice
  the disk space.

Nothing reads these files yet: not the loop, the optimizer, the analysts, the memory
pass, the reports or a resume. A test holds that outside `traces.py` no string literal of
the Python package names `traces`, `item.json` or `steps.json` (docstrings aside, and
beyond a listed set of existing prose and a Task's own `item.json`), only the loop
imports the writer, and the TypeScript sources do not name them. They add no journal
event, and the journal stays the run's record. They are there for people and for a
later trace-reading analyst.

### Run-wide feedback: changes by kind, the memory note, added-step reviews

Three more request sections (`optim/feedback.py`, `optim/memory.py`) carry what the run has taught so
far, beyond the recent candidates above. Each has a switch (a profile setting, or `0`/`1`
on the command line), none changes how anything is scored, and none reads anything but
the graph, train records and the gate's outcome words: no answer or option is parsed,
and no validation item or score is shown. They hold for every strategy that builds this
request (the default loop, `reflective`, `best_of_n`, `autoscientists`, a parent
picker paired with `autoresearch`); SkillOpt builds its own requests and is unchanged.
The Python walker and the optimizer only; nothing here touches the TypeScript spine.

- **Every change so far, by kind** (`run_ledger`, default `1`). Every candidate the
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
- **Optimizer memory** (`lessons`, default `1`). The memory pass's note (below, "The
  memory pass"): the loop's cross-round memory, shown in the request, to every analyst
  and to every merge. The proposer writes no `lessons` any more; the answer format has
  no such field.
- **Did the added steps contribute?** (`step_review`, default `1`). For every step a
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
  text output (at most 8 per request): it reads the step's own system prompt and, on up
  to 4 train items (changed outcomes first), the step's prompt as sent, its output and
  the output of the next recorded step that reads it (through the graph's reads, inputs
  and `{{fact}}` slots, so a sibling fork branch is not taken for it), each text whole
  unless together with the request's frame, instructions and system prompt they pass
  400000 characters (then the longest are cut head and tail to one length, the cut said,
  and when the step's own instructions leave the items less than 2000 characters each
  they are cut too; a call the provider refuses as too long is retried once at half its
  length, never when that request would be no shorter), and answers
  `adds_new_work`, `restates_inputs`, `contradicts_inputs` or `no_effect` with a short
  quote and a note. A step that makes the decision -- something reads what it writes
  (or it ends the data pass, for the grader) but no model step does, not even one a
  code gate's reloop hands it to -- is asked instead, item by item, what its
  inputs recommended and what it decided, and is shown as `follows_inputs`,
  `overrides_inputs` or `no_recommendation` with "decides as its inputs recommended on N
  of M item(s)": repeating the chosen answer is its job. A code step or gate gets no
  call: from the records' `steps`, the line says on which items a code step gave
  output (and how many distinct outputs), or which way a gate routed each item first
  and which items it sent back, each group with how the final answer moved there
  (changed, fixed, broken against the parent). The call is recorded in `optimizer_trace.jsonl` (kind `step_review`)
  and cached in memory and in `<run root>/step_reviews/` (cleared by a fresh run in the
  same directory, as the trace is), so a retry or a resumed run pays for none again.
  The section keeps full entries within 9000 characters; steps past that get one line
  of counts each. A failed call is retried as every optimizer call is (above); a failed
  or unreadable review is recorded and the step is shown without a verdict; it never
  fails the proposal. Without a data node, a step no edge leaves (the graph's own end,
  read by the scorer) counts as ending the pass, so a final agent step added there is
  asked the decision question.
- **What makes a new step pay off** (`structure_guidance`, default `0`). When on,
  "How nodes compose" adds one generic paragraph: what a new step adds depends on the
  failure it addresses, with examples (an independent attempt, several attempts combined
  by a vote or a check, a computation or check in code, a retry with a specific reason,
  a critic, a step that reformats or summarises for the next one); design a structural
  change from the evidence (the failure it addresses, how it changes
  what each step sees or how the work is done, what it should change); choose between a
  prompt edit and a structural change by the diagnosed cause, neither the safer default. Off by default, for an experiment.

The readers' success patterns (what the passing items do right) have their own heading
after the failure patterns, up to 4 per reader whatever its failure patterns, and a
quarter of the findings' room the failure patterns cannot take.

### The analyst stage: SkillOpt's per-round analysis (`analysts`)

`analysts` on (default `0`) gives the model proposer the analysis SkillOpt's reflect
stage gives its patch step (`optim/analysts.py`; the SkillOpt port's `analysts.py` and
`merge.py`). Before each proposal of `autoresearch` (alone, under `portfolio`, or
paired with a parent picker), on the train records the request carries:

- **Failure analysts.** Every item not passed (answered wrong, no answer, or an error),
  each once, shuffled with the epoch as seed and cut into batches of up to
  `analyst_group_size` (8, SkillOpt's `minibatch_size`); one optimizer-model call per
  batch. At most 8 calls of the stage are in flight at once, merges included (the two
  sides' merges run side by side), so 35 train items (5 or 6 batches) take one wave of
  analysts. Each reads the pipeline in the proposer's words, from the same renderers:
  the pipeline view and each step's settings with their values (a paragraph the steps'
  texts share read once, since an analyst never patches) -- no node table or node JSON
  -- and each item's full trace: its input, every step's input and output in the order
  they ran, the expected answer, the prediction, the score and the grader's feedback. Each item's heading
  carries the item fields its Task declares (`Task.item_metadata`, read by
  `Task.item_meta`; LiveMath shows `task_type` exactly as SkillOpt's analysts see it,
  rollout.py:132: the item's first theorem type, else `math_mcq`), never a
  field the Task names in `grader_fields`. A call's prompt has
  400000 characters of room (`ANALYST_ROOM`): the optimizer model (Opus 4.6) reads 200k
  tokens, of which Claude Code's own system prompt and the reserved reply take about
  30-40k, and traces dense in notation run 2.5-3.5 characters a token, so 400000
  characters is about 115-160k tokens. The room counts the whole request: the system
  prompt, the frame with the pipeline, and the items. Items are never cut to fit it:
  walking the shuffled items in order, a batch is closed early, before 8, when the next
  item whole would take its request past the room, and that item starts the next
  batch. So every item is in exactly one batch, whole, and long traces make more calls
  rather than shorter evidence. The room is the one setting tied to the model's
  context: a model that reads more makes fewer calls and loses nothing, one that reads
  less makes more. SkillOpt never clips, and neither does this, with one last resort:
  an item longer than the room on its own (one item's trace was at most about 230000
  characters in live runs) is a batch of its own, cut head and tail, step inputs first,
  then outputs, and counted (clipped items and characters left out). A call the provider still
  refuses as longer than the model's context ("prompt is too long", "context length",
  ...) is retried once within half the refused prompt's length, its longest items cut
  to one common length (never below 4000 characters) and counted (a merge has its
  pipeline texts clipped instead; its findings stay whole), and
  not at all when that request would be no shorter; a retry still too long fails the
  batch as any failed call does. Any other failed call is attempted up to 3 times with
  SkillOpt's waits (above). Only the model call's
  own failure is read this way: an answer that cannot be read is a failed batch even
  when its prose mentions a context window. The report counts clipped items and
  context retries only for calls paid in that step, so a step remade after a resume,
  whose calls come from the cache, does not count them again. It
  writes at most 4 findings (SkillOpt's analyst edit budget): what goes wrong, where
  (steps, edges, settings, or the structure), the change that would fix it, the items it
  covers (an id outside the batch is dropped), its impact and quotes. Optionally, how
  the steps took part (`steps`), as far as the items show it: `first_wrong` (the step
  where things first went wrong; in a loop or a long multi-turn run, the first step that
  did), `overrode` (later steps whose output replaced good work from an earlier step
  with worse) and `repeated` (steps whose output repeated another step's output without
  adding to it: an earlier step, a parallel branch, or their own earlier attempt in a
  loop). These describe what happened, not why. It assumes
  no graph shape: any node id (an agent with or without tools, a code op, a gate, a
  branch) may fill a role, and a part the items cannot show is left out. Only string
  step ids are kept (backticks and spaces stripped); a missing or malformed `steps` is
  `{}` and never fails the batch.
  Merges carry it through (each part's union).
- **Success analysts.** The passed items likewise (seed + 1): what made them work, where
  it comes from, and what a change must preserve.
- **Merge and rank.** Failure findings merge level by level in batches of
  `analyst_merge_batch_size` lists (8), then success findings; a final merge with
  failure priority returns one de-duplicated list ranked by items covered and impact.
  As in SkillOpt, one list is not merged, the final merge runs only when both sides
  found something, and an unreadable merge keeps every input finding (failures first).
- **Where the pipeline fails, step by step.** Right after the pipeline, the request
  aggregates every failure finding per step (`analysts.step_map`): for each step, in the
  pipeline's flow order (`analysts.flow_order`: the edges' topological order with
  loop-closing edges set aside, not the sorted node list; the search for those edges
  starts from entry steps, then other steps, then gates, so a loop's gate comes after
  the step it guards whatever its id), the failed items and findings
  that name it (`where`), that say things first went wrong there, that it replaced good
  work from an earlier step with worse, and that it repeated other steps' output without
  adding to it. An item counts once per cell however many findings cite it; a step id
  the pipeline lacks is ignored and does not count as attribution; success findings do
  not count. The caption says the map counts all failure findings, and, when there are
  more findings than the ranked list shows (16), that the list shows only the top ones.
  Findings without `steps` still count in the first column, and the section says how
  many carry no attribution. When a step replaced earlier good work or repeated others'
  output and the run permits structural operations, one line says such a cause can lie
  in what the step is told or in what it is given and how the work is split between
  steps, with examples of both (instructions on what to keep and when to depart from
  it; independent attempts; a vote or a check; a check that sends a step back; a code
  step when code nodes are allowed), and to choose by the evidence. With the analysts
  off, the map comes from the readers' failure patterns (the step each starts in, its
  items) when readers ran: only its first column (readers give no roles), counting only
  the patterns that name a step (not `unclear`), its caption pointing at "What the
  readers found"; with neither, there is no map. Each ranked failure finding
  also shows its attribution (`steps: first went wrong at ...`); success findings show
  none.
- **The request** shows "What the analysts found, ranked" (up to 16 findings, 30000
  characters) before the item results. Each finding is shown whole, as SkillOpt shows
  its analysts' edits: its text, what to change and what to keep, where (a structure
  note too), every evidence quote, line breaks kept; only when the findings pass the
  section's room are the longest texts cut to one length (never below 100 characters),
  each cut marked `… [cut: N chars]`. The analysts read the operations as the proposer
  does (`base.operation_lines`, the same rows), so a finding names only an operation the
  proposer is shown. An analyst's reply whose top-level JSON object does not parse is
  said to be invalid JSON, with the parser's message and where (`reply_object`); it is
  not repaired, and no object nested inside it is read as the answer. The readers, the step
reviews and the integrity judge read replies the same way (the judge's own salvage of a
broken reply, marked so it never vetoes, is unchanged). The analysts' item texts show an errored item's error once (not
  again as the grader's reason). The analysts take the readers' place: no
  reader call runs, and when the items do not fit one request the items they cite are
  shown in full first. SkillOpt's analysts are its only reading of the trajectories, so
  running both would give this loop more analysis than SkillOpt's.
- `analyst_max_batches` (0, no limit, as SkillOpt) caps the analyst calls per round,
  failure batches first; the request says how many items went unanalysed.
- More batches mean more lists to merge, but each merge call still reads at most
  `analyst_merge_batch_size` lists of at most 4 findings each, so a merge request stays
  as long as before (about 35000 characters in live runs); there are only more of them.

The analysis is made once per proposal step: it is keyed by the epoch, the generation,
the edited candidate, its train records, the stage's settings, the optimizer's model and
the room (which sets the batches),
so a retry after a refusal reuses it and makes no call; the analysts and merges read the
epoch's memory note, the same for every attempt of the step. The step's analysis is also kept in
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

A run with `analysts` on whose optimizer asks no model (the random proposer, a parent
policy alone) is refused when it starts. A custom strategy (`--strategy skillopt`)
builds its own optimizer requests: setting it warns, and the stage runs only for
proposals the strategy asks of the loop's model proposer.

The prompts name no dataset or task, ask for nothing beyond what SkillOpt's analysts
ask except the optional step attribution (`steps`, above), and see train records only. Off, no call is made and every request is
byte-identical to a run without the stage; the four settings are excluded from the
settings digest, so a run resumes whichever way they are set.

Per round with SkillOpt's settings, for F failed and S passed train items:
ceil(F/8) + ceil(S/8) analyst calls when every batch of 8 fits the room (more when long
traces close batches early), then the merges (one per side with more than one
list, at most 8 lists each, plus the final one when both sides found something), then
the proposal: on 35 train items, about 5-6 analyst and 3 merge calls, as SkillOpt's
analysts and merges make. SkillOpt's rank call (it trims its merged edits to the edit
budget) and slow update are not run: the proposer chooses and writes the edits. Its
meta skill is the memory pass (below), whose note the analysts and merges read, as
SkillOpt's read its meta skill.

Where the analysis still differs from SkillOpt's: SkillOpt's LiveMath arm uses
LiveMath-specific analyst prompts (`livemath/skillopt_prompts/analyst_error.md`), while
these are generic by design (no dataset hints); SkillOpt's batches always hold 8 items
(here a batch holds fewer when 8 whole items would not fit one call; no trace is cut,
except one longer than the room on its own, counted); SkillOpt shows
each item's task type (here the fields the item's Task declares, `Task.item_metadata`,
with its input and its choices when it has them); and SkillOpt's analysts never
see an errored item (it has no conversation to show), while here it is a failure shown
with its error. A failure finding here may also say how the steps took part (`steps`),
which SkillOpt's single-skill analysts have no use for; the SkillOpt strategy's own
analyst prompts (`refactory_packages/skillopt`) are unchanged.

### The memory pass: SkillOpt's meta skill (`lessons`, default `1`)

Once per epoch, from epoch 2, before a model proposer (`autoresearch`, alone, under
`portfolio` or paired with a parent picker) writes the epoch's proposal, the
optimizer's model writes ONE memory note (`optim/memory.py`), as SkillOpt's meta skill
(`skillopt/optimizer/meta_skill.py`, `trainer.py:2088-2199`) does:

- **Which two pipelines.** SkillOpt compares the previous epoch's last-step skill with
  this epoch's. Here: the candidate current when the previous epoch proposed (journaled
  `epoch_current`) and the one current now -- the validation gate's current, or without
  the gate the train leader. Each epoch scores and gates what the one before proposed,
  then the pass runs, then the proposal, so the note is read by the proposal it precedes
  and every later one until the next pass. When nothing replaced the current, SkillOpt
  rolls the same skill out twice; here the two rollouts are, under the gate, that
  candidate's previous-epoch rollout and this epoch's, and without the gate (one seed
  for the run) its rollout when it was scored and a fresh one made for the pass; each
  side is labelled with which it is (the comparison then shows run-to-run noise, and the
  request says so).
- **Which items.** SkillOpt's 20 (`slow_update_samples`): the train ids, sorted,
  shuffled with `seed + epoch * 2000`, the first 20 (all of them when there are fewer).
- **What it costs.** Each side's records come from the epoch's own rollouts with the
  epoch's train seed (rather than SkillOpt's `seed + epoch * 2000`): under the gate the
  edited candidate is re-scored on every epoch's train sample (fresh evidence) and the
  epoch's children are scored on it, so on the whole train part a pass usually costs no
  evaluation. Only sampled items a side has no record of are rolled out
  (`memory_rollout`, an evaluation counted against the evaluation, rollout, time and
  token budgets, journaled with its records so a resume reuses them; the candidate's own
  records and score are untouched): with `train_minibatch` up to 20 rollouts per side,
  and without the gate (one seed for the run) 20 rollouts when the current did not
  change. A cap or a spent budget leaves the missing items out; the pass compares what
  both sides hold. Plus one optimizer-model call per epoch, or, when the evidence does
  not fit one call (below), a few comparison calls and a synthesis call.
- **What the call reads.** Both pipelines (as an analyst reads one: nodes, edges, each
  node in full, every setting) and what changed between them: from their nearest common
  ancestor, each candidate on the way to either with its own edits and rationale (said
  plainly when they are on different branches, as a sibling train leader or a pool's
  pick can be, or share no ancestor), or the one pipeline when unchanged; what the score
  measures (as every optimizer request says it, below); "What was tried this epoch":
  every change the optimizer proposed in the round the start-of-epoch pipeline opened,
  newest first -- each candidate evaluated, with its kind (structural or knob, and the
  change kind of the run ledger), its edits in short (each op and what it touches, no
  values), its train score, its train fixed/broken counts against its parent and the
  gate's word (with the integrity verdict when the judge is on), then each proposal
  refused before it ran (`E-integrity` with its verdict and categories,
  `E-proposal-shape`, `E-noop`, `E-too-many-edits` and the rest) with why. These are
  the facts the proposal request's history, change ledger, integrity and refusal
  sections already show; never a validation score. The section is at most 8000
  characters (`TRIED_CHARS`); past it the entries listed last are cut -- refusals first,
  then the oldest evaluated -- and the cut says how many. So a pass in an epoch that
  kept nothing sees what was tried rather than "no changes" (a round that proposed
  nothing says so). Then the
  previous note with SkillOpt's "reflect on whether it improved or harmed"; and the
  comparison as SkillOpt formats it: a summary, then regressions (right -> wrong,
  highest priority), persistent failures, improvements -- each item headed by what it
  asks (SkillOpt's `Task <id>: <question>`), with both outcomes,
  answers and grader reasons, its declared metadata and both full traces (every step's
  input and output, as the analysts render them) -- then stable successes without
  traces, as SkillOpt shows them. SkillOpt never clips, and neither does this pass.
  When the whole request, system prompt included, fits 400000 characters (the
  analysts' room, the setting tied to the model's context), it is one call, exactly
  SkillOpt's. When it does not (20 items with both full traces often pass 1M
  characters), the evidence is split, not cut:
  - **Comparison calls.** The items in the comparison's order (regressions, persistent
    failures, improvements, then stable successes, still without traces) fill a call
    until the next item's pair whole would pass the room; every item is in exactly one
    call, both traces whole, in SkillOpt's comparison format. Each call also reads the
    pipelines and what changed between them, exactly as the single call shows them (so
    an observation can say where in the pipeline a problem starts), what the score
    measures and the counts over all items, and writes up to 8 observations for the
    future optimizer: what regressed, improved or still fails, and why, each citing its
    items with short quotes. Its prompt is the memory prompt's, adapted: address the
    future optimizer, use the traces' evidence, report what the traces show without
    judging kinds of change in general (the synthesis weighs them against what was
    tried), no instructions for the pipeline's steps. Domain-free.
  - **One synthesis call.** The single call's request -- both pipelines, what changed,
    what was tried, the objective, the previous note with SkillOpt's reflect wording
    -- with every comparison call's observations in place of the traces. It writes
    the note exactly as the single call does, under the same system prompt with its
    rules (one line says the comparison was read in parts). The comparison calls are
    asked for at most 6000 characters each; a part's observations past 12000 are cut
    to fit, the longest texts first, each cut marked `… [cut: N chars]`. When the
    pipelines are so large that they and the observations pass the room, the
    pipelines' texts are clipped head and tail to fit. Both cuts are counted
    (`synthesis_cuts` in the journal and report.json, a line in report.md).
  A model that reads more makes fewer calls and loses nothing; with a large enough
  context the pass is always the single call. Only an item pair longer than the room
  on its own is cut (head and tail, each cut said, "to fit the memory request's room")
  and counted. A failed comparison call fails the pass (no note from part of the
  evidence). A person's stop, a spent budget or a cancelled run starts no further
  comparison call (likewise for analyst and merge calls). Any call refused as too long
  is retried once at half the length (a
  comparison call with its traces cut to fit, counted; the synthesis with its pipeline
  texts clipped, the observations whole). The prompt is SkillOpt's `meta_skill.md` for a graph: address
  the future optimizer, what changes help or hurt here, regression risks, revise or drop
  what did not help, no instructions for the pipeline's own steps -- plus one line
  SkillOpt needs no counterpart of, having one kind of edit: judge only kinds of change
  with evidence in front of it (this epoch's tries, or what the previous note records
  of earlier ones); any other kind is untested, with no advice for or against it and
  no list of them. Domain-free; train items only.
- **What it writes.** `{"reasoning", "memory"}`. SkillOpt sets no length (its real runs
  wrote 3.4-5.8k characters); the note is asked for and kept within 6000 characters,
  a longer one cut with the cut said.
- **Who reads it.** Every analyst call and every merge call (with SkillOpt's
  `format_meta_skill_context` wording: prefer it when the evidence is ambiguous, do not
  force it where the traces contradict it) and the proposal request ("Optimizer
  memory"). It replaces the proposer's former `lessons`; the run ledger and the step
  reviews stay. It never edits the graph.
- **When it fails.** Where SkillOpt then has no meta skill, the loop keeps the previous
  note, marked NOT UPDATED in that epoch with why, in every request that shows it; the
  next pass that writes one clears the mark. The call is retried as every optimizer call
  is. A spent budget, a cancelled run or a missing model choice is no answer about the
  memory: no pass is recorded, so a resume makes the pass; a spent budget or a person's
  stop ends the search as it does elsewhere, and the run's report is written either way.
- **Records.** The call is traced (`optimizer_trace.jsonl`, kind `memory`; a split
  pass's comparison calls are kind `memory_compare`) and cached in
  `<run root>/analyst_findings/` like an analyst call; each pass is journaled
  (`optimizer_memory`: the two candidates, whether they are one, the sample, the items
  compared and missing, the counts per category, the evaluations and rollouts it made,
  the comparison calls (`parts`) when split, the calls' clipping and retry, and the
  note); a resume replays the note and the passes and does not make a pass again. A
  pass stopped part way is made again on resume: its answered calls come from their
  cache, and the synthesis, whose request holds every observation, is reused only when
  every comparison answered the same. report.json's `optimizer_memory` and report.md's
  "Optimizer memory" list every pass and the latest note.

`lessons` `0` turns it off: no pass, no note, no section. A custom strategy (the
SkillOpt port) keeps its own meta skill and runs no pass; an optimizer that asks no
model (random, a parent policy alone) has none.

### The integrity judge (`integrity_judge`, default `off`)

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
- **A quote backstop on `inherited`.** The judge's own label is all that keeps an
  `inherited` finding from vetoing, so it is checked, conservatively, since a wrong
  flip vetoes a sound change. The finding counts as `new` whatever the judge said only
  when all of these hold: every one of its evidence quotes found in the candidate is
  in none of the parent's fields (a finding that also quotes text the parent had stays
  `inherited`); quotes are matched within one field (a setting's value, a step's
  field), never across two; at least one of them has a word of four letters or more
  that is nowhere in the parent, so a change that only deletes or reorders text never
  counts; and the parent's state is known. Matching folds case, collapses whitespace
  and drops diff markers at line starts (`optim.integrity.backstop`). A flipped
  finding vetoes if its category is `veto`; the record keeps the judge's label as
  `judged_source` and the quote as `added_quote`. It changes nothing for a `warn` or
  `off` category, a judge error or a salvaged reply, and an offline replay applies it
  the same way.
- **Anything that slipped past once is never vetoed later.** A change that got through
  is kept in every candidate built on it, and later proposals are never vetoed for it:
  the judge only vetoes what a proposal itself adds. It is only reported, by the final
  audit of the winner's combined change from the seed. This holds for every way a change
  gets through: the judge failed, its reply was salvaged, its category only warns, the
  judge was off when it was proposed, or the judge called the finding inherited.
- **Setting it.** `integrity_judge: on` (or `veto`): every category `veto`.
  `integrity_judge: flag` (or `warn`): every category `warn`. An object sets
  categories one by one, in a profile or as JSON on the command line:
  `integrity_judge`: `{"mode": "on", "policy": {"answer_leak": "off", "train_overfit": "warn"}}`.
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
the solver's, byte for byte. `max_completion_tokens` and `data_parallelism` are frozen.
No knob shows an item's theorem or proof sketch (SkillOpt's `use_theorem` and
`use_sketch` are not ported): the theorem states the correct option. Each item takes 4
model calls.

`data`, `extract_answer` and `finish` are protected (`SkillGraph.protected_nodes`): a
search edits the steps between them, never the pass, the answer extraction or the run's
end. See "Structure edits" below.

## SkillOpt with its generic analyst prompts (`analyst_prompts=generic`)

SkillOpt loads each prompt from the env's own `prompts/<name>.md` when it has one, else
from its generic `skillopt/prompts/<name>.md` (`envs/base.py:278-329`). Its LiveMath env
overrides the two analyst prompts (`analyst_error`, `analyst_success`) with
LiveMath-specific ones, and the solver's `rollout_system`; the merge, ranking, slow
update and meta skill prompts are always the generic ones. The port reads a mode
package's overrides the same way (`livemath/skillopt_prompts/`).

`--strategy skillopt --strategy-option analyst_prompts=generic` (or
`"analyst_prompts": "generic"` in a profile's `strategy_options`) gives the analysts
SkillOpt's generic prompts instead, copied byte for byte into
`refactory_packages/skillopt/generic_prompts/`, so a run measures SkillOpt without
per-task analyst guidance. Only the two analyst system prompts change: the user
messages (with the edit budget L the generic prompts name), the solver side and every
other optimizer prompt are the same. The default `task` sends exactly what it did
before. Any other value is refused when the run starts. The value is in the run's
`config.json` (`strategy_options`, when given) and in the report's `strategy_report`
(`analyst_prompts`).

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
  item; no knob shows a grader-only field. LiveMath hides `correct_choice`, `label`,
  `label_text`, `theorem` and `sketch`.
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
- A fixed environment, nothing from the parent: `PATH`, `HOME` and `TMPDIR` (`/tmp`), and
  `LANG` and `LC_ALL` at `C.UTF-8` (`snippet.LOCALE`); no knob, no node manifest, no
  credential. macOS has the same names, with `HOME` and `TMPDIR` in its scratch directory.
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

**On macOS**, which has no namespaces, the sandbox is Seatbelt: `/usr/bin/sandbox-exec`
with a deny-by-default profile (`snippet._seatbelt_profile`), running the interpreter
refactory runs on, `-I -S`, in one process:

- No network of any kind (TCP, DNS, Unix sockets).
- Reads only of `/` itself, `/dev/null`, `/usr/lib`, `/System/Library`, the interpreter,
  its shared library, its standard library and a fresh scratch directory (stricter than
  Linux, where `/etc` and `/usr` stay readable). Homes, keychains, `/tmp`, runs and
  projects are refused, not just hidden. A standard library that holds the home directory
  is refused.
- Writes only to the scratch directory and `/dev/null`. The scratch directory is the
  code's `HOME` and working directory, is held to 8 MB and 1,024 entries (Linux's tmpfs
  size), and is deleted after the step.
- No other process: fork, exec (only the interpreter itself may start), signals and other
  processes' information (their arguments and environment, the process list, their paths)
  are refused, so the process cap is one and no process is visible but its own. IOKit
  registry properties (hardware identifiers such as the serial number) are refused too.
- The same fixed environment, 10 CPU seconds, 1 MB files and 10 s wall clock (its process
  group killed).
- 512 MB of memory kept by a watchdog on the process's footprint, polled every 20 ms
  (macOS does not enforce `RLIMIT_AS`): the code is killed within about one poll of
  passing it, so it can briefly hold a little more.

System V IPC (message queues, shared memory) is reachable on both platforms (Linux's
sandbox has no IPC namespace either), and an object a step makes outlives it; closing
that is a follow-up for both.

The sandbox fails closed. It is unavailable when `unshare` or the system Python is
missing (on macOS, `sandbox-exec`, or an interpreter whose install holds the home
directory), or when the probe snippet does not run (`snippet.available()`). In that case a
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
edits. `allow_code_nodes` off (`LoopConfig.allow_code_nodes`) refuses every new code node
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

### Wiring blocks, repair and checks before scoring

Writing and wiring a structural edit is mechanics; what to change stays the proposer's
call, and nothing below changes what a proposal asks for. All of it sits on the shared
path every outer loop's proposals take (`Run.attempt` and `Optimizer.attempt`, the
model proposer's retry), so every strategy gets it.

**Two plain wiring ops** (`refactory_dsh.nn.operators`), ordinary node callables that run
the same way on both runtimes:

- `collect` (`"call": "refactory_dsh.nn.operators:collect"`): reads the files the node
  lists in `reads`, each one optional, and sets its one `produces` fact to their text,
  one `### <file>` header per file. A file that was not written is said to be missing
  (a failed optional branch, a step with no answer) instead of failing the step; a file
  over 40,000 characters, or a binary one, is shown as its path and size. This is how
  files reach a model step as one `{{fact}}`.
- `file_text` (`"call": "refactory_dsh.nn.operators:file_text"`): sets its one `produces`
  fact to the whole text of its one read, which it requires (both runtimes refuse to run
  it while the file is missing). No header and no length cap.
- `select` (`"call": "refactory_dsh.nn.operators:select"`): its first read holds a label
  (the first word, or the last line alone, as a verdict gate reads one); the label names
  another of its reads by file name or stem, and that file is copied to the node's first
  write (and its fact, when it declares one). A label naming no file fails the step.

Neither is prescribed: no combiner, vote or normalisation is built in, and a proposer
that wants a vote writes it as a code step.

**What each node has.** The request's table of what each node can use writes each fact
and file as the pipeline view does (`name@maker`, `@run` when no step makes it), so a
fact has one label everywhere. Nodes with the same row share it, and a row that holds
all of another's names is said as that row plus the rest ("as at `X`, plus ..."). A
step's input is no longer mistaken for its output. The composition text says an agent without tools sees its prompt and the
files it lists in `reads` (pasted in after the prompt), as both runtimes deliver them.
`nn.operators.place_output` puts a new step's fact into the prompt knob of the next model
step downstream (or a named one), and says where; the core operators below place their
new steps' outputs the same way.

**Mechanical fixes, said** (`optim.repair`). Made by the attempt path with no model call,
only where what the proposal meant is unambiguous; the candidate's edits are the fixed
ones (so they replay to its state), its `repairs` name each fix, and the optimizer trace
gets a `repair` line:

- an edit's `target` (or `first`, `second`, a core operator's `step`, `after`, `before`
  or `fork`, a `remove_node`'s `id`, an edge's ends) that names a file is read as the step
  that writes it, when exactly one step does;
- a new node whose id is taken gets a fresh one, when nothing else names that id. When the
  id is the edit's own `target`, `first` or `second` (which may mean "replace this step",
  something the language does not do by adding one), or another edit of the proposal names
  it, the edit is refused `E-node` with why;
- in a bundle, a `set_knob` to the value the knob already holds is dropped;
- a `{{name}}` an added or changed step fills, where nothing makes the fact `name` and
  exactly one file there is called `name` (or `name.<ext>`), gets a `file_text` op before
  the step that sets `{{name}}` to that file's whole text. Not for a fork's branch or a
  join's source, and not when the op would rewire a protected step (the same edit written
  by hand would be refused `E-protected`): then the validator refuses the slot (V15).

Anything else is refused with the exact reason. A `set_knob` patch entry whose target is
not in the knob, or is empty where its action needs one, is now refused `E-patch`, naming
the entry and its target, unless the patch says what to do then (`config.on_missing`, as
SkillOpt's patches do); before, it was skipped and the rest applied. None of these fixes
has a switch; they act only on proposals that would otherwise be refused.

**The slim repair call.** A proposal refused for how it is written -- a validator rule, a
precondition, an edit's shape, node text, a binding, a knob, a patch target, lowering,
`E-gate-kind` or `E-code-run` -- goes back to the model proposer with the
reasons and only what fixing it needs: the proposal as written (rationale and prediction
included), the pipeline's nodes, edges and what each node has, the current text of each
knob it edits, the node schemas, the rules and the operations. No trajectories, history
or framing. The answer is the fixed proposal only, and it keeps the original's rationale
and prediction. A refusal about the idea (`E-integrity`, `E-noop`, `E-duplicate`), a
run's policy (`E-protected`, `E-frozen`, `E-policy`, tools, code nodes off), the
grader-side rule V17, or the change cap (`E-too-many-edits`: which changes to drop is the
proposer's call) is never sent. At most 2 slim calls per refused proposal and 3 per epoch
(`repair.REPAIR_CALLS_PER_PROPOSAL`, `REPAIR_CALLS_PER_EPOCH`); a fix goes through every
check a proposal does. When they produce no candidate, the last fix's refusal is kept with the
proposal's reasons (marked "the proposal as a slim repair call fixed it"), so a refusal of
the fix -- an integrity veto, say -- reaches later requests, and the loop asks again with
the full request, as before; that request now shows the latest refused proposals as
written under "Why your last proposals were refused". The trace records each call (kind `repair`)
and its outcome (`repair_outcome`); the refusal is journaled with `repaired: <candidate>`;
the epoch line reads `refused N (...), repaired M (...) in K repair call(s)` (a repaired
proposal counts once, among the accepted) and says how many accepted candidates carry
mechanical fixes. A repaired refusal is counted as repaired, not refused, in the run's
refusal tallies (`refusal_codes`, `refused_total`, the report's `rejection_codes` beside
`repaired_codes`) and carries `repaired` in `rejections.jsonl`; a portfolio round counts
the fix's edit level. Each epoch's call count is journaled (`repair_calls`), so a resumed
run keeps the bound. `Optimizer.repair` makes none by default; the `autoresearch` proposer
makes the slim call, and any optimizer may override it. `Optimizer.propose_valid` repairs
the same way (it has no production caller; strategies go through `Loop.install`).

**Checks before scoring** (`optim.checks`), after the validator and lowering accept a
state:

- `E-gate-kind`: a gate a person or an agent answers (`gate_kind` `user` or `agent`) that
  the edit adds, or turns an existing gate into, is refused: an outer loop's candidates
  run headless, nobody answers it, and every item reaching it would fail, on both
  runtimes. The node schemas such a request shows leave both kinds out and say why. The
  parent's own gates are not judged.
- `E-code-run`: each code node the edit adds or changes is run in the sandbox on up to 3
  of the parent's recorded items (the files and facts its run left in
  `runs/<parent>/passes/<data node>/<item>/`, or the run directory for a node outside a
  pass). Code that raises on them, or returns what its node cannot use, is refused with
  the error, the line it failed at and the inputs it had. When nothing can be run -- the
  parent kept no recorded items, an input is made by a step the candidate adds, the node
  is in a data pass and reads facts (an item directory does not reliably keep its item's
  facts: on the walker its `facts.json` is the run's, copied when the pass began), or
  there is no sandbox -- it is skipped and the candidate's `notes` say so (journaled, and in the
  proposal record's `checks`).

A knob only a split step read is no longer judged by `split`'s precondition: the
proposal's later edits may remove it or give it a reader, and the end state is held to V7
as every rule is.

**Turning it off.** Nothing here is a setting. A run whose proposals never trip these
rules behaves as before; the fixes and checks act only on proposals that would otherwise
be refused or fail every item.

### The core structural operators: one line, wired by the system

`refactory_dsh.nn.core_ops` adds ten operators to the edit catalogue, each a macro over the
same seven primitives, checked by a precondition and validated like any edit, so every
outer loop gets them through the one attempt path. They say nothing about what a step is
for. The proposer writes what kind of step (`agent`, `code`, or `op` with a `call`), where
(a step id) and its text (a prompt, or code); everything else is optional:

| operator | one line | what the system does |
|---|---|---|
| `insert` | `{"op": "insert", "after": "<step>", "kind": "agent", "text": "..."}` (or `"before"`; `"takes_over": "<earlier step>"`) | a fresh id; the new step sees what the step it sits after sees plus that step's outputs (before a step: what that step sees); its output is wired as **Wiring** below says; before a fork's branch, it takes the branch's place |
| `remove` | `{"op": "remove", "step": "<step>"}`; on a gate, `"keep": "<verdict>"` | predecessors to successor; a fork's branch leaves the fork's targets and the join's sources, and a fork left with one branch is folded away; an operator's own helper steps (its collect step, a loop's judge) go too, so do the slots it put in other steps' prompts and the knobs only it read; refused with the exact reader when a step still needs what only it makes. On a gate, the kept verdict's path stays and steps only the others reached go |
| `parallelize` | `{"op": "parallelize", "steps": ["A", "B"]}`, `{"steps": ["A"], "copies": 2}`, or `{"after": "X", "new": [{"kind": ..., "text": ...}, ...]}` | consecutive steps (neither reading the other's output), a step beside copies of itself (agents only), or new steps, between a new fork and join; copies' and new steps' outputs are wired after the join as **Wiring** below says |
| `serialize` | `{"op": "serialize", "fork": "<fork>"}` | the branches one after another in their declared order; the fork and join go |
| `branch` | `{"op": "branch", "fork": "<fork>", "kind": ..., "text": ..., "optional": true}` | a new branch: the fork's targets, the join's sources and edges; given `optional`, its first step binds it to a knob of its own (`<step>__optional`), which `set_knob` moves; it sees what the fork's predecessor sees and makes; its output is wired after the join as **Wiring** below says, guarded (`{% if %}`) when `optional` is given |
| `rewire` | `{"op": "rewire", "step": "<step>", "sees": ["task", "<step id>"]}` | what an agent or code step sees: `task` is the task's own input, once, as the model steps receive it (never a step's output or a knob's text: an op's output carries the knobs it declares in `knob_outputs`, else every knob it reads), a step id that step's outputs; slots for what it no longer sees leave its prompt knob, new ones are added; same id, outputs and knobs |
| `loop` | `{"op": "loop", "step": "<step>", "judge": "agent", "text": "<what to check>"}` (judge `code`, or a `module:function`) | a judge after the step (an agent whose reply a verdict gate reads, a code gate, or an fn gate) that sends it back with the reason; `max_iterations` (optional in the edit, 2, 1-5) is a knob of its own afterwards (`<gate>__max_iterations`); never a gate a person or an agent answers. The written-out form (`target` and a `gate` node) stays |
| `route` | `{"op": "route", "after": "<step>", "judge": "agent", "text": "...", "to": {"<label>": "<step id>" or {kind, text}}}` | a judge and a gate after the step, one labelled edge per route; new steps rejoin at the step's old successor, their outputs read only when there |
| `split` | `{"op": "split", "step": "<step>", "into": [{kind, text}, {kind, text}]}`; one entry replaces it | the first takes the step's inputs, the last its outputs (file and fact names), the step and its prompt knobs go, a mode knob it also read moves to the last. The written-out form (`target` and two node objects) stays |
| `merge` | `{"op": "merge", "steps": ["A", "B"], "text": "..."}` | one step with A's inputs and B's outputs; refused when another step reads what A makes |

**What a core operator holds to.** From the 2026-11-17 expressibility audit
(`audit/easy-dsl-gaps.md` on the lab):

- `insert` takes `takes_over: <step>`: the new step writes that earlier step's outputs, so
  what read them (what is scored) reads the new step's, and the step's own go under new
  names (`<step>.txt`, `<step>`) for the steps before the new one. "Several attempts and
  a choice" uses it, so the chosen attempt is what is graded. The step taken over must
  have run whenever the new one runs (a fork runs every branch, a route one), so a step on
  a sibling branch or a route's skipped branch is refused. Also refused: a frozen knob, or
  a knob a later step shares, that names its output; a step that names it in its code or
  feedback keys; an agent without tools taking over a step that writes two files; and,
  with the run's protected and grading nodes, rewiring one (`E-protected`).
- A `parallelize` copy is the step itself under a new id (`core_ops._clone_agent`): the
  same prompt, system prompt, role, limits and tools, each bound text in a knob of the
  copy's own with the same value, frozen where the original's is, and no knob the step
  lacks. It is sent byte for byte what the step is sent, on both runtimes, so it takes no
  `sees`.
- A step a bundle adds keeps the reader the bundle gave it: an edit of the same bundle
  that removes its last reader (a whole `set_knob` value dropping its `{{slot}}`) is
  refused `E-unread`, naming that edit; a new step whose output reaches no step is said in
  the candidate's "As built" (a step ending its pass or the run is not: the Task reads it).
- A field an operator would ignore is refused, naming the fields it takes
  (`core_ops._unknown_fields`). That covers a role or timeout on `insert`, a tool list
  (`tools` is true or false),
  and a field no step takes inside `new`, `into` or a route's `to`. `insert` lists
  `system`, which it takes.
- An operator after which a kept step would run, on some path, without an input it had on
  every path before is refused (`core_ops.lost_inputs`). An example is a route that skips
  the steps making what a later step reads. An agent's prompt is read: a `{{slot}}`
  inside `{% if name %}` is optional there, as the validator holds it (V15). An op or a
  gate is code that cannot be read, so it keeps every read and input it has, since it may
  need what it declares optional; only the slots of the knobs it fills honour their
  guards. A step whose inputs cannot be read is said, never skipped.

**Wiring.** Facts reach a model step as `{{slots}}` in its own prompt knob, files through
a `collect` step in front of it (so a missing or binary file is said, not a crash); a
code step gets both in `inp`. An agent's reply that is also its fact is handed over once,
as the fact. A new step's output (a file `<id>.txt` and a fact `<id>`) goes into the
prompt knob of the next model step downstream -- an agent whose task prompt is a knob of
its own, or an op that fills facts into exactly one text knob -- or where `"to"` says; with neither, the candidate says nothing reads it yet.
Every text an operator creates goes into a knob of its own (`<id>__task`, `<id>__code`);
every text has a default (the neighbour's, or a neutral line), so a strategy with no model
can apply any operator; ids, files and facts are generated, and the same operator on the
same parent makes the same state. Each operator counts as one change toward the cap.
The candidate says what was wired (`Candidate.notes`, "Wired by the system: ..." in the
request), including a new step whose output no step reads yet.

**Tools.** A copy of a step that uses tools (a `parallelize` copy, or an `insert`/`branch`
whose default is a copy of one) keeps its tools only in a fresh workspace of its own
(`scratch`): the attempt path lets a new agent run with the same tools a step of the seed
runs with when it works in such a workspace (`E-new-node-tools` otherwise, unless the run
allows tools for new nodes). A step that uses tools in the run directory is not run
beside a copy of itself, rewired or merged; the refusal says to give it a workspace.

**Each kind tried, against the noise.** A kind tried once with a small result read as a
run-long ban: after one loop broke 2 items, S1-s43's note said "AVOID loops"; S4-s44's,
after one branch, "DON'T add more parallel analysis branches". Yet one of S1's 17
validation items is 0.059 of the validation mean, so a single try is inside the noise
either way. The memory call now gets, per kind tried, the tries and each one's train
fixed/broken counts and the gate's word, with the noise they sit in: one validation item's
share (`1/|val|`, from the split) and, when the run re-scored the parent, how far its train
score moved (`memory.kind_tally_lines`). The counts are said there once; the tried entries
keep their edits and score. `MEMORY_SYSTEM` says a kind tried once inside that noise is no
evidence against the kind: the note says what went wrong in that try, from how it moved its
items, never "avoid" it. The untested-kinds rule stands. The proposer sees the same tally,
kept with the note it was written from, beside the note (`memory.request_lines`).

**How a structural edit moved its items.** A child's counts ("1 fixed, 8 broken") say
that an edit moved items, not how. For a child whose graph has steps its parent's lacks,
the records say it (`optim.edit_effect`): for its broken and its fixed items, which of
those new steps ran on them, what each new gate answered (an item counted under each
verdict it got at least once), which step the child called more often than the parent did
on the same item and -- where the mode's Task reads a reply as an answer
(`Task.reply_correct`, found by `task.reply_judge`) -- of those, on how many the first
reply was already right and then changed to a wrong one, or was never right. A reply the
Task cannot read, a reading that fails, a Task that reads no reply and an item whose
parent run failed are each said, never dropped. It is computed from the records, with no
model call, and it is generic: steps are compared by name, verdicts are the gates' own,
and only the Task reads a reply. "What the last edit changed" shows it under each child's counts (a rejected
child too), and memory's "What was tried this epoch" under each structural change. On the
stopped S1-s45 run the loop broke 8 items; on 5, the solver's first reply was right before
the judge sent it back.

**The request.** Under "Structural operators", one line and one example per operator,
with this graph's ids; then compositions built only from them -- several attempts and a
choice (`parallelize` copies, an `insert` judge after the join, an `insert` `select` op),
check and send back (`loop`), an independent attempt (`branch` or `parallelize` with
`sees: ["task"]`), a simulation then an analysis (`insert` code, then an agent), a
background pass beside an experiment (`parallelize` new), iterative fitting (`loop` with a
code judge), a tool step before reasoning (only when the run allows tools for new steps),
and dropping a step -- under "none is the preferred shape -- compose your own", rotated by
generation. The answer format asks a structural proposal for `addresses` (the failure,
citing steps and items by id), `flow_change` (which step sees what) and a checkable
`prediction`; one missing them is judged anyway and marked (`reasoning_missing` in its
proposal record). "What the last edit changed" puts each child's prediction beside a
mechanical check: what happened to the items it names (fixed, broken, score up or down,
unchanged) and the steps it names (how many items a step's reply changed on, or that it is
new or gone). Two facts are said in numbers: "Time per item" (how long the edited
candidate's train items took each -- median, 90th percentile and
longest, from each item record's own `duration_s`, which both runtimes stamp from the
item's pass starting (`items.item_duration`), over the items the harness did not time out;
never its wall time over its items, which its slowest item sets), and cost: with no cost
term (parsimony 0),
that cost is not a reason for or against a change; with one, a "cost per item" column of
each candidate's measured model cost over its train items.

**A harness timeout is not a target.** An item the harness ended for time
(`base.harness_timeout`: an `error_kind`, or a step's `missing_steps` cause, of `timeout`
-- a session ended on inactivity, the item deadline -- or `harness_incident`, the
backstop; never read from an error's text) still scores what it scored, 0, but no request reads it as the
pipeline's doing. "Item results" lists it once, by id, as `HARNESS_TIMEOUT`; it is in no
error row, no item shown in full, no time and no analyst's batch, and the analysts are told
not to propose changes aimed at timeouts or retries for them. A wall a step declares
(`timeout_s`, `declared_timeout`) is the pipeline's design and an ordinary failure. "How nodes compose" says it
in one sentence. The SkillOpt port's own requests are SkillOpt's and unchanged.

**The request's order.** The proposal request (`ProposalContext.render`) gives the right
things in the right order, each once: what the score measures; the pipeline, once, as the
pipeline view; what this run has tried and how each attempt did (recent results, every
candidate so far, the traces' numbers, the last edit's children and the reasoning behind
recent candidates with what each edit built, the run's feedback and memory, earlier runs,
the latest refusals); what the analysts found, failures and successes, with the per-step
failure map; the edits it may make (the operations, each said once -- the written-out
ones in a table, then the core operators one line and one example each, the same rows the
analysts read -- how nodes compose, which apply to each step, the steps' settings, the
author's guidance); the items; then a reference at the end (what each node has before it
runs, each step in full with the node schema, the validator's rules; the nodes and edges
only when the pipeline view could not be drawn); and the
answer's form last. Items are chosen rather than cut, and size is never a reason to cut
one: every item is shown when all of them together are under `SHOW_ALL_BELOW` (120k
characters; it only decides that, and never cuts); otherwise a table lists every item,
then items chosen by a rule (`_shown_items`), never by size, are shown whole -- one failure per failure finding that
cites a known item, in rank order; then one contrasting success per success finding (the
cited and same-kind ones first), or, with no success findings, one per kind of failure
shown, each marked as a success. Without findings, one failure per kind of failure (its
outcome and declared kind) and a success beside each kind. Each
item gives its verdict, what was expected and what it gave, the grader's reason, then its
input and each step's output, each fenced with its headings demoted, so nothing in an item
reads as a section of the request. Only a single text too large to send whole
(`ITEM_TEXT_MAX`, 60k characters) is cut, and it says so in place,
"[… N chars omitted]", never a path (the proposer has no tools); each cut (item, step,
chars) is in `ProposalContext.omissions` and in the proposal call's `optimizer_trace.jsonl`
entry (`omissions`).

**One item's evidence.** Every call that reads train items -- the analysts, memory's
comparison, the proposer -- renders an item the same way (`optim.evidence.render_item`).
A step's prompt re-embeds the task's input and the earlier steps' replies, and the same
instructions recur in every item, so an item shown as each step's prompt plus its reply
grew with the square of its steps (61% of an S4 analyst batch was repeated text). Now the
input appears once per item; a step says by name what its prompt took from what is already
shown (`Sees: the input, compare's reply`) and only the part of its prompt the request does
not show elsewhere (`Also in its prompt`: a reason a judge sent it back with, say), then its
full reply, each fenced with headings demoted. A step's instructions are its settings,
under "Each step's settings", in every call that shows items: the proposer's, the
analysts' and memory's (`base.settled_for`; an analyst or memory request counts its
settings as shown, clipping and table cells included, `analysts.shown_settled`). Across the
items of one request a text shown for an earlier item is said as "same as item X, step Y"
(`evidence.dedup`) when that is shorter. Nothing is lost: every reply and every distinct
text appears once in the request. With its labelled choices shown, an item's expected
answer is its label (its choice line says the text), and the table of every item gives the
whole text only for an item not shown in full. An item's error is said once, under "Errors,
by cause", which a full item points to. The repeat guard reads inside fences too
(`request_sizes.repeated_sentences(fenced=True)`, the settings section counted once), so a
step's instructions in its settings and again in an item fail it. Two renderings of the same
item -- memory's two rollouts -- show the input and the expected answer once (the first),
and each is headed by its rollout, so "same as item X (Start of epoch), step Y" says which;
"What the last edit changed" shows the parent's item whole and the child's verdict and
replies beside it, a reply the child kept said as "same as the parent's". Generic: lines of
text are compared, with no knowledge of the task or the pipeline.

**The pipeline view.** "The pipeline you may edit" opens with the pipeline in one line per
step, generated from the graph the edits act on (`nn.pipeline_view`), read in the core
operators' own words. Reading mirrors writing: every operation in the catalogue names the
readers that render what it builds (`EditSpec.reads_as`: `step`, `sees`, `told`, `loop`,
`route`, `parallel`, `edges`), and a test fails on an operation without a reader or a reader
no operation names. A step's line says what it `sees` (each input as `"sees"` takes it,
`name@maker`), what it is `told` (each knob whose text reaches it, its opening words; `*`
frozen, `?` read by an op with several outputs for all of them; a code step's code) and
what it writes (`->`); forks are `parallel fork { ... } -> join` (`[optional]` branches); a
judge and its gate are `loop(step, judge ..., max N -> N+1 runs)` or `route(after step,
judge ..., {label: step})`, each label's steps until the ways meet; data passes are
`per item (data) { ... }`; `[protected]` steps are marked. A plain op is folded into the
step it feeds only when it is a pure assembler (no tools, not code, protected, the
grader's, or a step a gate names) whose outputs only model steps read, directly or through
an operator's collect step; it shows as `name (via op: its inputs)`. An agent with tools
sees the run dir (or its own workspace). What no reader recognises is drawn generically
(kind, sees, writes, `edges ->`) and counted: no hand-written seed graph needs one, nor does
any graph core operators build from them (`tests/test_pipeline_view.py`, with a round trip:
an operator changes only the lines it describes, which name its handles, drawn by the
readers it declares). Read from its lines alone, the view says exactly the graph's edges
(checked on every seed and its draws): a way that meets a step drawn already says
`-> step (above)`, a way that ends the run inside a block says `-> end` (so what follows the
block does not read as following it), and a loop's gate with more than one way forward
lists each after its loop line. A loop that does not go on with its last attempt past its
cap says so (`past the cap: fail_item` or `halt`, from `Graph.on_exhausted`). About 1.5k
characters on the S4 pipeline. Python only: the spine does not render requests. A parser
from the view back to a graph is not built.

**What an edit built.** A proposal says in prose what its edit does (`flow_change`); the
expansion can differ (a loop left at its default cap, a branch that sees the task taking in
more than the question). The candidate's proposal record carries `wiring`: its pipeline
view's lines that are new (`+`) or changed (`~`) against its parent's, and the steps gone
(`- removed`), read from the two graphs (`pipeline_view.changed_lines`), never from the
edit's text. The next request shows them under "As built" right after the proposal's own
"Flow change, as proposed", in "What the last edit changed" (and in "Your earlier
reasoning" for a candidate that section does not show; one it shows is pointed to), so a
description that says otherwise is seen against what ran, in the same words as the
pipeline view (at most 12 lines a candidate, the rest counted).

**Naming inputs.** `"sees"` takes `"task"`, step ids, or single fact or file names there
before the new step (that input alone); the catalogue says so, a name that is not there is
refused with the names that are, and each fork's row in the operator table lists what a new
branch there can name, `(+knob k)` marking an output that carries text knob `k`.

**What `"task"` gives.** `operators.task_inputs` is the one definition the operator line,
the pipeline view, the step-carries table and the operators' wiring read: the task's own
input, once, as the model steps receive it. A name is out when an agent makes it, when an
op makes it with a text knob's value, or when a node that takes such a name in makes it.
Of the rest, it is what is there before some model step and made by an op of the mode
(never a data node's per-item record, such as `current_item.json`, nor a pass-through
copy by `collect`, `file_text` or `select`); when a model step takes some of those in
directly, only those, so the same question built twice (a message and its framed prompt)
is given once. Which output of a multi-output op carries which knob is the mode's
declaration, the op node's `knob_outputs` (`{output: [knobs]}`, every output named, each
knob one the op reads). An op that reads a knob and declares nothing carries it in every
output, and the view marks it `?`. Both runtimes accept the field and refuse a bad one
(the walker's `Node`, the spine's `validateGraph`); lowering carries it into the IR's
`FnNode` with the op's `knobs` and `produces` beside it, and into the node declarations.
LiveMath's seeds declare it: S4's `prepare` (`user_prompt` carries nothing,
`compare_prompt` carries `compare_task`) and S1's `format_prompt` (`user_prompt` and
`solver_prompt` carry nothing, `skill_section` carries `skill`). So "task" is
`user_prompt` on S4 and `solver_prompt` on S1. A `rewire` that would leave a step seeing
what it sees now is refused.

**What each step carries, and which operators apply to it.** After "Structural operators"
the request has one row per step (`operator_fit_section` in `optim/base.py`): what a model
step's prompt and reads carry, each with its maker, a fact a plain op builds followed back
through the plain ops that built it to its sources (model steps, what the run was given,
knobs; at most 4 ops deep, `CARRIES_DEPTH`). No op says which of its outputs a knob feeds,
so an op with several outputs has its knobs marked as the op's
(`precision_prompt (op notes_prompts, which reads compare_notes.txt (compare), knob(s) precision_task, hypotheses_task (the op's, for all 2 of its outputs))`),
and which core operators apply to the step as written with their default arguments and
which are refused, in the refusal's own words (equal reasons grouped). Each verdict is the
run's own attempt on the edited candidate (`Run.mechanics`: the catalogue, then
`Optimizer.attempt` with the run's protected and grading steps, frozen and guarded knobs,
wiring and validation; everything `Run.attempt` checks but the model-judged integrity
check), recording nothing and without the code dry run (a sandbox process for each edit
on each request): an accepted edit that adds or changes code is marked "(not dry-run)",
and proposing it runs the dry run as any attempt does. A request built outside a run has
no section. A refusal that
points to replacing a step names the exact edit (`{"op": "split", "step": ..., "into":
[one step]}`) only when an attempt would accept it (`checked_hint`), and otherwise says why
replacing it is refused too. That check skips the code dry run (a sandbox process on every
request), so an accepted code replacement is marked "(not dry-run)"; proposing it runs
the dry run as any attempt does. The section fits 7000 characters: refusals are cut to their
first clause, then rows are left out, last first, and counted among the listed steps.
The lead also names what `"sees": ["task"]` gives a new step on this pipeline, in the
words the `rewire` line uses (`operators.TASK_MEANING`, computed by
`operators.task_inputs`). The `loop` line states the verdict format (`proceed` or
`reloop`, then why, which reaches the step sent back), what the judge sees by default (what
the step read and wrote) and the cap (`max_iterations` in the edit, then the knob
`<gate>__max_iterations`: send-backs, so at most that many runs plus one; default 2, 1 to
5, after which the run goes on). A fork's row says what a new branch that sees the task
takes in there.

**Turning it off.** The operators are in the `broad` and `structure` presets; the
`conservative` and `knob_only` presets are unchanged, and a run that allows only some
operations (`allowed_ops`) sees only those. The raw primitives and the written-out macros
stay as they were.

### What the edits can reach: the DSL coverage map

`refactory_dsh.nn.coverage` maps every construct of the graph language (each node kind,
gate kind, bindable field, `Node` field, edge form and composition both runtimes run) to
the operations that create it, change it on an existing node and remove it, or to the
reason none does: the grader side, a gate nobody answers headless, an infrastructure
flag, a step's identity, a user's decision, or `deferred` by the user. Nothing may be
left planned. `tests/test_dsl_coverage.py` generates the list of constructs from the
code and fails on one without an entry or left planned, checks every operation named exists, and applies
an example of each covered node kind, edge form and composition through the attempt path,
and of each change or removal the core operators cover.
The table below is generated from the map (`coverage.table_markdown()`), and the test
checks it matches. It records the user's decisions (`coverage.DECISIONS`): an operator may
change what an existing step reads; a copy of a step that uses tools keeps its tools only
with its own fresh workspace; a loop's limit and a branch's optional are settings bound to
knobs, which `set_knob` moves (`loop` makes the limit's knob, `branch` with `"optional"`
the branch's), never tools or `outputs_only`. Nothing is left planned: timeouts, retries,
`on_exhausted`, the workspace kind, the model role and the per-sub-item experiment are
deferred, with why in `coverage.DECISIONS["deferred"]`.

<!-- dsl-coverage:start -->
| construct | create | change on an existing node | remove |
|---|---|---|---|
| node kind: `op` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `duplicate`, `insert`, `parallelize`, `branch`, `route`, `merge` | `set_knob`, `rewire` | `remove_node`, `bypass`, `remove` |
| node kind: `agent` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `duplicate`, `insert`, `parallelize`, `branch`, `route`, `merge` | `set_knob`, `rewire` | `remove_node`, `bypass`, `remove` |
| node kind: `gate` | `add_node`, `wrap`, `loop`, `route` | `set_knob` | `remove_node`, `add_edge`, `remove` |
| node kind: `fork` | `parallel`, `parallelize`, `add_node` | `branch`, `remove`, `parallelize`, `insert` | `serialize`, `remove` |
| node kind: `join` | `parallel`, `parallelize`, `add_node` | `branch`, `remove`, `parallelize`, `insert` | `serialize`, `remove` |
| node kind: `data` | *grader-side* | `set_knob` | *grader-side* |
| gate kind: `fn` | `add_node`, `wrap`, `loop`, `route` | `set_knob` | `remove_node`, `add_edge`, `remove` |
| gate kind: `user` | *no headless answerer: refused E-gate-kind, and left out of the schema* | *no headless answerer: refused E-gate-kind, and left out of the schema* | `remove_node`, `add_edge`, `remove` |
| gate kind: `agent` | *no headless answerer: refused E-gate-kind, and left out of the schema* | *no headless answerer: refused E-gate-kind, and left out of the schema* | `remove_node`, `add_edge`, `remove` |
| bindable field: `prompt_template` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | `set_knob` | - |
| bindable field: `system_prompt_template` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | `set_knob` | - |
| bindable field: `code` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | `set_knob` | - |
| bindable field: `max_iterations` | `add_node`, `wrap`, `loop`, `route` | `set_knob` | - |
| bindable field: `parallelism` | *grader-side* | `set_knob` | - |
| bindable field: `optional` | `branch` | `set_knob` | - |
| node field: `id` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | *identity* | - |
| node field: `kind` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | *identity* | - |
| node field: `call` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | `split` | - |
| node field: `evaluator` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | `split` | - |
| node field: `gate_kind` | `add_node`, `wrap`, `loop`, `route` | *identity* | - |
| node field: `knobs` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | *deferred: binding a new setting on an existing step* | - |
| node field: `reads` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | `rewire` | - |
| node field: `required_reads` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | `rewire` | - |
| node field: `writes` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | *identity: a step's outputs are what its readers rely on: replace the step (split) to change them* | - |
| node field: `produces` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | *identity: a step's outputs are what its readers rely on: replace the step (split) to change them* | - |
| node field: `prompt_template` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | `set_knob` | - |
| node field: `system_prompt_template` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | `set_knob` | - |
| node field: `role` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | *deferred: the model role among the run's routes* | - |
| node field: `tools` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | *excluded by decision: tools are not a configure setting* | - |
| node field: `max_iterations` | `add_node`, `wrap`, `loop`, `route` | `set_knob` | - |
| node field: `on_exhausted` | `add_node`, `wrap`, `loop`, `route` | *deferred: a timeout or retry setting as a knob* | - |
| node field: `notes` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | *infrastructure flag* | - |
| node field: `blocking` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | *infrastructure flag* | - |
| node field: `score_effect` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | *infrastructure flag* | - |
| node field: `targets` | `parallel`, `parallelize`, `add_node` | `branch`, `remove`, `parallelize`, `insert` | - |
| node field: `optional_targets` | `branch`, `add_node` | `set_knob` | - |
| node field: `sources` | `parallel`, `parallelize`, `add_node` | `branch`, `remove`, `parallelize`, `insert` | - |
| node field: `subgraph_entry` | *grader-side* | *grader-side* | - |
| node field: `subgraph_exit` | *grader-side* | *grader-side* | - |
| node field: `task_ref` | *grader-side* | *grader-side* | - |
| node field: `source_path` | *grader-side* | *grader-side* | - |
| node field: `source_format` | *grader-side* | *grader-side* | - |
| node field: `inline_items` | *grader-side* | *grader-side* | - |
| node field: `limit` | *grader-side* | *grader-side* | - |
| node field: `parallelism` | *grader-side* | `set_knob` | - |
| node field: `parallelism_knob` | *grader-side* | *grader-side* | - |
| node field: `resource_role` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | *infrastructure flag* | - |
| node field: `feedback_keys` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | *infrastructure flag* | - |
| node field: `timeout_s` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | *deferred: a timeout or retry setting as a knob* | - |
| node field: `retry` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | *deferred: a timeout or retry setting as a knob* | - |
| node field: `outputs_only` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | *excluded by decision: not a configure setting* | - |
| node field: `workspace` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | *deferred: the workspace kind* | - |
| node field: `tool_policy` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | *excluded by decision: tools are not a configure setting* | - |
| node field: `field_bindings` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | *deferred: binding a new setting on an existing step* | - |
| node field: `code` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | `set_knob` | - |
| node field: `inputs` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | `rewire` | - |
| node field: `knob_outputs` | `add_node`, `insert_after`, `insert_before`, `wrap`, `split`, `parallel`, `loop`, `insert`, `parallelize`, `branch`, `route`, `merge` | *excluded by decision: the mode's declaration of what its op does* | - |
| edge form: `plain` | `add_edge`, `insert_after`, `insert_before`, `insert` | `remove_edge`, `add_edge`, `serialize`, `remove` | `remove_edge`, `remove` |
| edge form: `verdict` | `add_edge`, `loop`, `wrap`, `route` | `remove_edge`, `add_edge` | `remove_edge`, `remove` |
| edge form: `reloop` | `loop`, `add_edge` | `remove_edge`, `add_edge` | `remove_edge`, `remove` |
| edge form: `halt` | `add_node`, `loop` | `set_knob` | `remove_node`, `remove` |
| composition: `nested fork` | `parallel`, `parallelize`, `branch` | `branch`, `remove`, `parallelize`, `insert` | `serialize`, `remove` |
| composition: `loop inside a fork branch` | `loop` | `set_knob` | `remove_node`, `add_edge`, `remove` |
| composition: `loop inside a data pass` | `loop` | `set_knob` | `remove_node`, `add_edge`, `remove` |
| composition: `gate routing to several steps` | `route`, `add_node`, `add_edge` | `remove_edge`, `add_edge` | `remove` |
| composition: `a step per sub-item` | *deferred: the per-sub-item experiment* | *deferred: the per-sub-item experiment* | *deferred: the per-sub-item experiment* |
<!-- dsl-coverage:end -->

### The invariant harness

`tests/structure_fuzz.py` checks those promises by generating random structure edits
from a seed, over LiveMath's skill graph and seven small graphs
(`tests/structure_fuzz_graphs.py`: a fork and join, a nested fork, the three gate kinds,
a gate loop, a data pass, a pipeline of ops, and several terminals). The edits cover
every primitive and macro, node copies taken from the node JSON the optimizer is shown
(new ids, new prompts), and malformed edits. Each walk applies them through the same
attempt path (with its mechanical fixes; the harness answers every gate, so it lets edits
add gates a person or an agent answers, which an outer loop refuses), and every accepted
state is checked two ways.

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
