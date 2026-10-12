# Supplemental operator evidence v1

This is a validation interface, not a passing attestation. Keep all documents
under the operator-only review directory denied to every authoring lane.
`OperatorEvidenceContext` is supplied by the operator/harness, never derived
from a lane's claimed reviewer or user identity. It includes the actual trial,
lane, reviewer and all lane-writable roots.

Use `review_template(project, context, kind)` to obtain an **unfilled** document
with the current complete raw inventory and reference hashes. It writes nothing.
The returned `reviewed_at`, `verdict`, and `rationale` are null; null fields cannot
pass. Save completed reviews under `CONTEXT.root/LANE/semantic-review.json`,
`human-approval.json`, or `readme-review.json`. All file references are relative
to their specified owner and symlinks are rejected.

Every document binds schema version, kind, trial/lane/project/reviewer identity,
reference bytes, the current complete raw evidence hash inventory, review time,
an actual `passed`/`failed`/`unavailable` verdict and reviewer rationale. Re-scoring
changed raw evidence requires a new review of those bytes. Missing evidence
stays unavailable; inconsistent or stale supplied evidence fails.

For semantic review, `source_obligations()` returns each source stage/role rubric
ID, full original instruction text, role, stage ID and source hash. Assess every
rubric entry against the actual outputs for this concrete task. Each assessment
needs a verdict, substantive rationale, and exact excerpts from actual assigned
model inputs **and** role outputs. Command/output evidence can be cited as part
of those reports. A supplied source prompt or the workflow's own PASS claim is
not itself an independent semantic assessment. Citation format:

Scope conditional duties explicitly in the assessment rationale, quoting the
source condition and actual assigned task/project evidence. For example, the
researcher role says `Mode 2: Research (used in Improve mode)`; its Improve-only
search counts and report headings are not automatically unconditional Create
duties. Review the four researcher mode conditions against the actual invocation,
retain the full source-role hash, and assess the shared investigation/synthesis
responsibilities plus the tailored Create-stage directions. Conditional
applicability never permits omitting the entire role assessment or its actual
input/output evidence.

```json
{
  "obligation_id": "SOURCE_OBLIGATION_ID",
  "source_sha256": "FROZEN_SOURCE_HASH",
  "verdict": null,
  "rationale": null,
  "evidence": [
    {"path": "ACTUAL_INPUT_PROMPT_PATH", "sha256": "ACTUAL_HASH", "quote": "ACTUAL_EXCERPT"},
    {"path": "ACTUAL_ROLE_OUTPUT_PATH", "sha256": "ACTUAL_HASH", "quote": "ACTUAL_EXCERPT"}
  ]
}
```

Human identity supplements require an actual human reply to the specific review
question. The operator's trusted user references are a mapping from actual user
turn reference to `question_reference`, `decision`, `approver`, `strategy_digest`
and `input_paths`. Obtain these from the trusted user channel, not lane JSON.
The document adds these same fields plus `user_turn_reference`,
`answer_evidence_path` (archived `raw/approvals/...json`),
`request_evidence_path` (`approval-required.json`) and `native_receipt_path`.
The evaluator independently matches the current approved bytes, one-use request,
archived actual decision and human native-gate acceptance receipt. A label in
`--approver` does not supply a trusted user turn. No platform login or cryptographic
authentication claim is inferred from an operator attestation.

README review adds `artifact_path`, `readme_path`, `readme_sha256`,
`package_sha256` (current unchanged native package inventory), and exactly two
`executions`: `standalone` and `native`, each with an operator-owned `receipt_path`
and its `sha256`. Execute the actual authored examples with explicitly recorded
path substitutions. Preserve input, output, stdout and stderr under operator
ownership. An execution receipt has this unfilled shape:

```json
{
  "trial_id": null, "lane_id": null, "reviewer_id": null,
  "kind": null, "documented_command": null, "substitutions": {},
  "argv": [], "cwd": null, "started": null, "finished": null,
  "returncode": null, "package_before": {}, "package_after": {},
  "execution_status": null, "reason": null,
  "diagnostic": {"stream": null, "quote": null},
  "input_file": null, "output_file": null, "run_directory": null,
  "input": {"path": null, "sha256": null},
  "output": {"path": null, "sha256": null}, "expected_output": null,
  "stdout": {"path": null, "sha256": null},
  "stderr": {"path": null, "sha256": null}
}
```

The documented command must occur in README; actual argv must match that command
after only explicit path/interpreter placeholders (`INPUT`, `<PROJECT>`, `$PYTHON`,
`${RUN}`, `/path/to/project`, or `/absolute/path/to/project`). Whole command names, flags and literal
`booking_conflicts.py` filenames cannot be substituted. The resolved standalone
script must be the actual inspected artifact's `booking_conflicts.py`. Native output is `conflicts.json`
inside the recorded invocation directory. Use the frozen overlap or touching
example inputs; observed preserved input/output bytes must match those cases.
All pre/post package inventories must match native acceptance's inspected package.
Preserved input/output snapshots must still match the actual invocation files.
For an honestly failed or unavailable execution, record `execution_status`, a
precise `reason` and an exact diagnostic excerpt in preserved stdout/stderr or
output. Absent output is allowed only when the actual invocation file is absent
and the execution/review is nonpassing. Command, argv, cwd, input, time, streams
and unchanged-package bindings still apply. A nonzero status or missing output
can never certify a passing review. The review verdict must match the aggregate
execution statuses; unavailable runtime setup remains distinct from an observed
computation failure.
Ordinary shell backslash-newline continuation and an initial `cd DIR` followed
by newline or `&&` are normalized only for argument/path binding. The preserved
documented command and actual shell invocation bytes remain unchanged.
The operator executes and records commands; the evaluator executes no code.

`evaluator_provenance()` records evaluator version, validator/scorecard/reference
hashes and the bounded claim. A later report over existing raw trials must retain
the original trial identity and add this evaluator identity, review-context hash,
review-document hash and reviewer identity. It is a new evaluation of existing
evidence, never an amendment to the frozen execution or retroactive model proof.
