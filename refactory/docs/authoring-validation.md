# Authoring validation and intent audits

Create/Edit execute what they built before reviewers sign off. One canonical
runtime operation owns the current revision's smoke, including its generated
score and any model/Task work. Design's declared acceptance command has the same
place in its graph. Listing or registering a package only makes it available for
execution; completion requires successful validation and current review.

```mermaid
flowchart TD
  A[Brief, research, accepted strategy/design] --> B[Coder]
  B --> C[Materialize and structural admission]
  C --> D[Stage/register package]
  D --> E[Canonical smoke]
  E --> F{Repairable failure?}
  F -->|yes, within shared bound| B
  F -->|dependency, cancellation, uncertainty or exhaustion| X[Stop with diagnostics and unblock/recovery action]
  F -->|passed| G[Separate health, code and adversarial QA]
  G --> H[QA gate]
  H -->|repair within shared bound| B
  H --> I[Overwatch: intent versus final evidence]
  I --> J[Audit gate]
  J -->|missing evidence or defect, within bound| B
  J -->|current evidence and PASS| K[Completion report]
```

Fast Create retains a strategist, coder and one focused QA worker. Fast Edit
retains coder then focused QA. Their smoke now precedes focused QA and shares
`repair_attempts` with structural admission and QA failures. There is no added
Overwatch worker or fan-out. Full Design retains separate health/code/adversarial
reviews and Overwatch after its acceptance command; fast Design retains focused
QA after that command.

## Runtime receipts and repair

`create_ops.gate_smoke` executes `smoke` once for each package attempt. Agent modes
receive a model for that execution; they no longer make a preliminary incomplete
keyless run. `smoke.json` records mode, sample inputs, run directory, status, score
when measured, stdout/stderr, elapsed time and failure information. Distinct smoke
directories preserve the outputs of earlier attempts. `revision.json` separately
records comparison/provenance. Design retains its command and actual output.

After the smoke passes, the gate runs the mode once more with its input empty
(`refactory_dsh.empty_input`). Only declared input is emptied: a knob named in the
brief's inputs (`--input`) points at an empty folder or file when its sample or
default names one, and is the empty string otherwise; a data node reading a
`source_path` reads an empty source of its format. Other knobs keep their values,
so an output folder or a settings file given with `--sample` is never emptied, and
a path the graph writes is left alone. The check then judges the run:

- it stopped with its own error: pass;
- it finished and every declared file it wrote is empty: pass. Empty means no
  content, a CSV/TSV header with no rows, or JSON (or JSONL lines) holding only
  `null`, `[]`, `{}`, `""`, `0`, flags, or the status strings `status`, `reason`,
  `error` and `message`, so `{"status": "no input", "rows": 0}` is an honest report;
- it finished and wrote a result: a repairable finding, sent back to the builder.

The check is inconclusive when the run timed out, the model was unavailable, the
mode could not be described, or an undeclared knob defaults to an existing path
(then the mode is not run, since it would still read that data). It is not
applicable when the mode declares no input. Both are run warnings and appear in
`create.json`'s `warnings`. Describing the mode and the run share one smoke budget.
`empty-input.json` holds the receipt, and `create.json` carries it beside a
`readme` entry that warns when the package has no README.

An edit starts from the package it revises: the emission's `===FILE:` blocks
replace or add files, a `===DELETE: <path>===` block removes one, and every
other file (the README, data, anything not Python) is kept. Each attempt starts
from the source package, not from the previous attempt. A symlink that leaves the
package is dropped, never copied as the file it points at; one inside the package
stays a link.

`authoring-evidence.json` binds the validation result to hashes of the delivered
source, authoritative brief/strategy/design/plan inputs, applicable ledger entries,
validation settings, receipt files and canonical runtime output files. Each validation receipt also survives in
`.refactory/validation-history/`. Earlier failed or superseded records remain
historical; they cannot certify a repaired revision. This local evidence contract
does not promise exhaustive tracking of external services or every transitive
dependency outside the delivered package.

Repairable implementation/check failures return concrete diagnostics to the
coder. Full modes share an overall attempt cap across admission, runtime and
review failure; within that cap, check failures and review reloops are counted
separately (`check_attempts_used` against `check_attempts`,
`review_attempts_used` against `review_attempts`), so a review reloop does not
consume a check attempt. Fast modes use their existing total repair allowance
but record which kind (`check_repairs_used`, `review_repairs_used`) spent it.
Missing tools/credentials, cancellation, uncertain worker outcomes/timeouts and
resource exhaustion stop with their actual outcome. The artifact and evidence
remain for explicit unblock/recovery; repeating code generation cannot install an
unavailable validation dependency.

The shared resource ledger accounts for work, validation and repair, including
nested model smoke. Review and completion use the validation allocation; a repair
returns to work admission. Token and elapsed caps remain optional and unset by
default. The repair bound controls attempts, not a fabricated token quota.

## What reviewers and Overwatch receive

The common agent-delivery path supplies the latest ledger and captures the
artifact/context identity at dispatch. Health, code and adversarial QA remain
independent full-mode roles. They inspect canonical receipts and delivered output,
then use targeted checks for uncovered requirements. They do not each repeat the
whole model/Task feed.

### Repository questions (design and design_fast only)

For repository-change runs (those with a `target`, `refine`, or `source_project`),
each reviewer must answer three structured questions per changed function:

1. **Duplicates** — where else the repository does the same thing, with `path:line` evidence.
2. **Callers** — who calls it and from which user path, with `path:line` evidence.
3. **Shared state** — what it shares with what already runs (state, files, facts, locks), with `path:line` evidence.

Each answer must include `path:line` evidence, or state `absent: <what was searched>`
when the answer is genuinely "nowhere" or "shares nothing".

The `changed-functions.json` file is computed by the `changed_functions` module, which
maps diff hunk line ranges onto function spans via `ast` (Python) and hunk-range entries
(TypeScript). It is written at `repo_change.py` materialise (unconditional) and
`_check_in_worktree` (conditional). The `changed_functions` fact is rendered into each
reviewer prompt as `{{changed_functions}}`.

`gate_qa` and the fast `gate_qa` refuse a reviewer output that lacks an entry for a
changed function, or gives an answer with no `path:line` evidence. Nothing asks them at
design time: design mode has no design reviewer (docs/hardening.md, "One design
pass").

### Marker placement rule

Every reviewer prompt that can report BLOCKING findings carries
`create_ops.MARKER_RULE`, which tells the reviewer to write the blocking
marker only at the start of a finding line in its plain form, never to
wrap it in backticks, a code span or quotation marks, to write every real
finding outside any fenced code block, and to put an example of the format
inside one, because the parser skips fenced blocks.
This makes it less likely that the `blocking_findings` parser reads an
explanatory mention as a real finding. The parser does not enforce the
rule, so a review that ignores it can still trip the parser (tracker dm-220).

### Claim checking

The code_reviewer checks each factual claim about existing code at `file:line`,
including uncited premises; a false claim is BLOCKING in the review output. At design
time, `check_design_claims` checks the design's cited claims mechanically and lists
what it cannot confirm as unverified claims on the approval card; it never sends the
design back. Uncited-claim checking is prompt-only.

### Deliverable-kind conditioning

The `deliverable_kind` fact is computed at prepare time: `"repository_change"` when the
run names a `target`, `refine`, or `source_project`; `"artifact"` otherwise. It uses
`task.get(k) or ctx.knob(k, "")` for each key, so it works on both runtimes.

Repository-change runs get `REPO_CHANGE_REVIEW_INSTRUCTIONS` (code-change review content
composed from verification methods M2, M3, M11, M12, plus repository questions and
claim-checking instructions). Artifact runs get `ARTIFACT_REVIEW_INSTRUCTIONS` (the
rendering and exchange-format text previously inline in health_checker and qa_director).

### Unanswered repository questions

Reviewers answer repository questions (duplicates, callers and shared state, each with
`path:line` evidence or an `absent:` reason) for the run's changed **product** functions
only:

- Test files and generated files are left out. A generated file says so in its first
  lines ("Generated … Do not edit"), as `generated-help.ts` does.
- Code the branch merged in from its base is left out. When a run merges its base after
  it began, a function counts only if it also changed relative to the merged-in commit
  (`changed_functions.merged_upstream`).

Missing repository questions are advisory at every gate, and nothing is re-run for
them: they are the reviewers' bookkeeping, not a defect in the change. A post-build
gate used to re-run every reviewer once when they were its only problem; on a measured
run (dm-149, 2026-10-22) that cost ten minutes and left the same answer missing. The
gate records them in `reviews/qa-gate.txt` or fast mode's `fast-report.json`, and the pull request's Verification section lists them. They
never send a build back, re-run a review, or end a run.


Overwatch receives the original brief, accepted strategy/criteria and design,
current applicable ledger, merged reviews and final runtime evidence. It writes
`reviews/overwatch.md` plus `intent-audit.json`. Fast focused QA writes its existing
report plus the same intent map. The map covers every baseline id: `brief`,
`strategy`, explicit `criterion-N` rows when supplied, `design`/`plan`, and current
product-scoped `ledger-N` instructions, including the initial idea.
Process-scoped notes (`--scope process`) are recorded in the evidence payload
under `process_notes` but are not baseline requirements. Compound asks must be decomposed into separate rows even
when they share an id. Each row identifies the requirement, status and supporting
individual file paths with concrete command/output or source findings. Directories,
line-number suffixes and the audit itself cannot serve as evidence references.

For every constraint or correction row, each clause must include a `sites` array:
`"sites": [{"site": "path:line", "tests": ["test_file::test_name"], "scope": "connected"}]`.
Each site names the exact file:line in the diff that builds this clause, the test(s)
that reach it, and the evidence scope (`local`, `connected`, `assembled`,
`artifact`). A multi-clause correction must have sites on each clause, not a single
site on the row. The gate's `correction_audit_problems` checks these sites
mechanically: it verifies the site is in the diff, reverts it and runs the named
test, and enforces the declared scope.

Required behavior without adequate evidence is failed or unverified with a
precise reason. A README limitation or reviewer/coordinator waiver cannot alter
the baseline. An exclusion must cite an explicit user change and quote its exact
text (semantic review must still establish that it really authorizes the exclusion). A missing required
consumer or validation tool stays unverified with its exact unblock action.

The gate requires coverage, substantive evidence references, a final exact PASS
verdict without blockers, and current delivery receipts. A changed artifact,
receipt, validation input or later applicable user instruction invalidates old
sign-off. Completion checks freshness again. Explicit graph redo/recovery still
owns restarting completed nodes and archiving dependent state.

These checks establish traceability and freshness. They cannot mechanically prove
that an agent understood every sentence, that a cited command establishes the
claimed behavior, or that a generated artifact is good. Those remain semantic
review responsibilities, tested with actual modeled acceptance as well as controls.

## Edit comparison policy

```mermaid
flowchart TD
  A[Revised mode passes canonical execution] --> B[Capture source identity/state; check freshness]
  B --> C{allow_regression?}
  C -->|true| D[Skip original execution and numerical comparison]
  C -->|false, comparable| E[Execute original on accepted sample inputs]
  E --> F[Compare scores; refuse a lower revised score]
  C -->|false, not comparable| G[Record why comparison is unmeasured]
  D --> H[Persist provenance and proceed to review]
  F --> H
  G --> H
```

The default is still false. With true, no original smoke runs, no delta is
computed and the revision is not classified as regressed/non-regressed.
`before_score` and `regressed` are null; `after_score` describes only the revised
measurement. `revision_note` explicitly says comparison was intentionally skipped.
The source state/digest and provenance remain, and revised correctness/QA still
run. With false, existing comparable/non-comparable behavior and numerical refusal
remain. This is not a new evaluation-editing framework or a performance-improvement
requirement.

CEO uses `--knob allow_regression=true`, the Python/headless Edit command uses
`--allow-regression`, chat's Edit tool uses `allowRegression`, and the panel exposes
the declaration's boolean knob and help. All reach the same `_revision` operation;
historical report fields stay readable.

Implementation: [graphs](../refactory_dsh/create.py),
[fast graphs](../refactory_dsh/fast.py), [operations](../refactory_dsh/create_ops.py),
[evidence contract](../refactory_dsh/authoring_evidence.py),
[delivery](../refactory_dsh/agent_delivery.py),
[controls](../tests/test_authoring_evidence.py).

The [2026-09-27 acceptance record](authoring-validation-acceptance-20260927.md)
separates successful terminal/chat runs, the panel's negative control, incomplete
full-chat attempts, measured usage and deterministic coverage.
