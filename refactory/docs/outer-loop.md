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

## LiveMath skill graph (`livemath/skill_graph`)

Arm C of the outer-loop study starts from arm A's seed skill, converted by hand into an
explicit graph. Inside LiveMath's per-item data pass, `compare` analyses every option;
`precision` and `hypotheses` then run in parallel on compare's notes; `decide` reads all
three sets of notes and answers in the baseline solver's format. LiveMath's unchanged
grader scores decide's answer. Each agent runs the solver's system prompt with only its
own seed-skill section, and the step task texts carry no method guidance.

The four section knobs (`compare_skill`, `precision_skill`, `hypotheses_skill`,
`decide_skill`) are the only open knobs. Each step's system prompt,
`max_completion_tokens`, `data_parallelism`, `use_theorem` and `use_sketch` are frozen.
An item's theorem states the correct option, so the last two stay off. Each item takes
4 model calls.
