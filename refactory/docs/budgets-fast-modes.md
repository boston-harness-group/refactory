# Fast authoring and shared run budgets

`default_pack/create_fast`, `default_pack/edit_fast` and `default_pack/design_fast`
are ordinary, inspectable modes for small, purposeful tasks. They omit research
and reviewer fan-out and allow one repair by default. The existing `create`,
`edit` and `design` modes remain available with their full graphs.

| Mode | Sequential work and required validation |
|---|---|
| Create fast | One strategist plans the minimum reusable graph, knobs, score and artifact invariants; one coder emits the package; one QA agent reviews it. Structural admission and canonical runtime smoke remain required. |
| Edit fast | One coder makes the small revision directly, then one QA agent checks it. Task bindings, inputs and artifact contracts are preserved; structural admission and canonical smoke remain required; comparison defaults on but `allow_regression=true` skips the original execution/comparison branch. |
| Design fast | One strategist, one coder and one QA agent produce a small checked change: commits on the run's own branch, in a worktree of the repository `target` names or of a new project, as Design works. The required command runs there. |

The strategy is a concise plan, not a separate research or approval workflow.
Create and Edit produce reusable packages. Design produces the artifact itself
and is not an optimization target.

## Acceptance status

All three fast modes completed their required checks and reports in separate
CLI and native GUI runs. CLI Create and native Edit each exercised one actual
structural repair; Edit retained its measured score. Both interfaces also showed
an explicit unresolved known-defect control with zero permitted repairs.

The final checks used six-minute elapsed limits and one repair. Successful CLI
Design retained a 150,000-token cap; the user chose no cumulative-token cap for
the subsequent checks. All usage remained measured. The earlier capped failures
and a native Create import failure are retained alongside the corrected success.
These results establish the sampled artifact contracts, not token efficiency or
arbitrary project quality. See the
[recorded outcomes](release-validation-20260926.md#fast-mode-outcomes).

Those Design runs worked on a copied project. Fast Design's move into a worktree
was accepted separately, in CLI runs on 2026-10-16: a new project; a change to
this repository checked with vitest, whose first check was failed on purpose, so
its repair was a second commit built on the first; a run with its worktree off;
and one copying a `source_project`. Native GUI runs of the worktree paths have
not been made yet.

## Choose the graph

In the terminal, choose `--mode create_fast`, `--mode edit_fast` or
`--mode design_fast` with `refactory ceo`. In the native GUI, select the same mode
from the mode catalogue and configure its knobs, then use **Run once**. The public
CLI forwards `--check` to the declared `check` knob for all fast variants and
honors `--timeout` for authoring. The direct authoring commands also accept `--fast`:

```bash
uv run python -m refactory_dsh.design --fast \
  --name tiny-helper --brief 'Write helper.py with add(a, b) and a small assertion test.' \
  --check 'python helper.py' --repair-attempts 1 --max-elapsed-s 360
```

Fast Design works where Design works, through the same steps
([`repo_change`](tutorials/01-write-a-mode.md#if-your-mode-changes-a-repository)).
For a change to a repository you have, pass `--target <its root>` (the `target`
knob): the run gets a worktree of it on its own branch, from a pinned base
(`--base`), and your checkout stays as it is. Without one it starts a new project
in `~/refactory-projects/<name>`. Each attempt is a commit, written in `===FILE:`,
`===EDIT:` and `===DELETE:` blocks, and the check runs in the worktree's own
environment after `--generate` (whose tracked-file changes are amended into the attempt) and `--setup`; see [models](models.md). Fast Design has no approval
gate, so it opens no pull request: its work stays on the run's branch, which
`design.json` names. To open one, push the branch yourself; Design's delivery
pushes only a base GitHub already has, so it cannot deliver that branch for you.
`--no-worktree` writes `.factory/artifacts/<name>` instead.

To change a directory that is not a git repository, pass
`--source-project /absolute/project` (the `source_project` knob) instead of
`--target`. Refactory copies the source to the artifact directory and applies
emitted whole files there. It does not replace the source project. Dependency
directories, Git metadata, previous run output and `.env*` files are excluded;
source symlinks are refused, and so are edits and deletions.

`repair_attempts` is one shared bound across structural admission, canonical runtime validation and focused QA,
and, in fast Design, an emission materialise could not apply, from zero to three. It
counts repairs after the first implementation, so one permits at most two
implementations. The required Design `check` runs where the run works, in its
worktree or its artifact directory, as a command bounded by `check_timeout_s`
rather than by memory, as every check is (see [models](models.md)). Create/Edit can also name an explicit
`check`; an empty check is reported as not requested, not passed. Focused QA asks
the worker to inspect the emitted or changed files, requested behavior and relevant
regressions, using the actual command result already supplied. It records evidence
for each check and an explicit list of checks not run. It does not edit the
implementation or repeat downstream runtime stages. A required check not run
cannot count as a pass.

Canonical smoke runs before focused QA. Its real receipts join the requested check result in the QA context. The worker must map every brief/criteria/ledger obligation to current evidence in `intent-audit.json`; it cannot waive a requirement or silently repair source. Missing dependencies stop with an unblock explanation. Changed source or intent invalidates the earlier receipt/sign-off. See [authoring validation](authoring-validation.md).

Read `authoring-evidence.json`, `intent-audit.json`, `focused-check.json`, `focused-qa.json`, `fast-report.json`, `site-revert-results.json` and `correction-audit.json` in the run
directory. They separate command results, model-authored review, structural
admission and runtime smoke. Exhausted repair produces an unresolved report and
no success report. A model's favorable review is not proof of artifact quality.

## Allocate resources

Run-wide token and elapsed limits are **optional and unset by default**. Fast mode
is defined by its smaller graph and repair bound, not a universal token quota.
The release-check cutoff was a test allocation. Read [Choosing a budget](budget-strategy.md)
for the usage breakdown, tradeoffs and revised acceptance strategy before selecting
another limit.

Elapsed time and tokens are separate from candidate counts, graph steps and one
completion's hang timeout (`--model-timeout`, default 1800 seconds). CLI controls are `--model-timeout`, `--max-elapsed-s`, `--max-tokens`,
`--validation-reserve-s` and `--validation-reserve-tokens`. The GUI exposes the
same effective run settings.

`--model-timeout` sets how long one model completion may take (default 1800s) where
a call reports no activity: a CEO turn, a console call, a Codex node. An agent node's session on DSH or Claude Code has no wall: it ends after `REFACTORY_MODEL_INACTIVITY_S` (default 600) seconds with no streamed frame, thinking included, and no running tool command, or at the harness backstop `REFACTORY_MODEL_BACKSTOP_S` (default 14400 s, a harness incident), unless the node declares its own `timeout_s`.
Such a node's spine limit is the backstop + 600 (a Codex node's: `attempts × model-timeout + 600`); the per-node
`command_timeout_s` overrides the spine's global `commandTimeoutMs`
(`execute.ts:344`). `--max-elapsed-s` is a deliberate wall-clock ceiling that can
end a step mid-completion. `streamIdleTimeoutMs` is disabled by default
(`workers.py:162`) but can undercut `--model-timeout` if explicitly enabled by a
provider config.

Python graphs accept the same fields:

```python
run = mode.run(
    workdir,
    model=client,
    resource_limits={"max_elapsed_s": 360, "max_tokens": 30000,
                     "validation_reserve_s": 60, "validation_reserve_tokens": 5000},
)
```

The values above illustrate accounting controls; they are not an allocation known to
complete fast authoring.

The shared `.refactory/resource-usage.json` ledger aggregates coordinator and
nested-worker usage, retries and validation. Repeated cumulative receipts for
the same worker replace its previous total instead of counting it twice. The
outer loop reserves validation resources before admitting another candidate;
required final validation can use the reserve. Fast package smoke and fast
Design's check are explicit validation stages. Fast graph nodes declare their resource
phase independently of the worker role. Design uses final-validation resources for
its check, QA, gate and report; Create/Edit use final-validation resources for canonical smoke and subsequent checks, QA and reporting. A repair returns to the work allocation. Mode authors can set
`nn.Node.resource_role` to `work` or `final_validation`; omission retains the prior
role-based/inherited behavior.

Elapsed limits are checked at graph boundaries and while owned model/check
processes run. Under `enable_recovery` (the default for walker authoring runs
and outer-loop evaluations), classified ops and fn gates with a derived timeout run in a
child process forked from a preloaded forkserver and are killed at the limit
via session-grouped SIGTERM→SIGKILL escalation; ops in the child get no
`ctx.budget`, and the parent's kill is the enforcement. Token limits are enforced
at provider usage boundaries. Each worker's in-flight provider request may
overshoot; this is not an exact provider-side token cap. A backend without token
telemetry refuses a token-limited launch. Missing usage or pricing is shown as
unavailable (`null`), and estimated cost remains an estimate. Candidate counts
never stand in for token or time measurement.

An exhausted budget before required validation is incomplete/unvalidated, not a
success. Owned processes are asked to shut down and partial artifacts remain.
The ledger stops charging elapsed time when the owning run pauses or finishes.

## Resume deliberately

At a completed node boundary, supported resume retains completed work and its
consumption. An explicit larger allocation can extend the limit; it does not
reset usage. The direct authoring commands accept `--resume RUN_DIR` with
`--fast` for a saved fast graph. Python uses
`ModeClass.resume(workdir, model=client, resource_limits={...})`.

An interrupted node with an unknown side-effect outcome is never automatically
replayed. Inspect its artifacts before an explicit restart. Changed graph
instructions or a changed Design source-project snapshot invalidate dependent
results and require a new run. Partial artifacts are evidence, not a claim that
the interrupted node completed.

The Python executor conservatively refuses partial-fork recovery because its
saved record does not yet support restoring individual branch facts. It does
not replay successful sibling branches or silently retry an unknown worker.

The final-validation reserve belongs to the outer run. During optimization, a
candidate's internal validation nodes cannot spend the winner holdout reserve.
Nested searches reuse their enclosing ledger and do not reset usage, extend the
allocation or pause the enclosing clock. Deterministic candidates share elapsed
limits without requiring a model client; no provider usage remains unavailable.

Focused QA is bound to a digest of the materialized output after the requested
check. If QA edits those files, the gate requires another implementation/check
cycle within the same repair limit; a passing review cannot silently approve a
changed, unchecked artifact.

Custom model clients used with token caps must implement `with_budget(budget)`
and report measured or explicitly estimated usage into that shared ledger.
Clients without this protocol are refused at the completion boundary, including
optimizer proposals. Supplying such a client to a deterministic graph does not
force a model call or invent usage. Existing uncapped clients stay compatible.
