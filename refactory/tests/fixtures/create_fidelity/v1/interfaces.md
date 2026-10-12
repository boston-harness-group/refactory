# Candidate integration interface v1

Candidate name: `create_fidelity`, Python class
`refactory_dsh.create_fidelity.CreateFidelityMode`. Public Create, Design and Fast
retain their current graph identities. Expected topology and inherited prompts
are pinned in `reference.json`; conformance does not derive expectations from
the candidate.

Graph ownership: `create_fidelity.py`, `create_fidelity_ops.py`, and candidate
prompt data. Director ownership: `create_fidelity_directors.py` and its tests.
Gate/lifecycle ownership: `create_fidelity_gates.py`, `nn/graph.py`,
`nn/executor.py`, `lowering.py`, `workers.py`, `worker_observer.mjs`,
`model.py`, `owned_process.py`, and their focused tests. Parent owns benchmark,
contract tests, CLI/registry integration, and integration fixes.

Director stages use native op nodes with the original IDs and role `ceo`, calling
`refactory_dsh.create_fidelity_directors:research_director`, `strategy_director`,
and `qa_director`. This is a documented composite-stage mapping, not permission
to eliminate the director model. Each function takes RunContext and returns no
facts. All communication is through declared files. Stage-local DSH planning,
real child dispatch, quality review/retry, and complete report publication are
mandatory. The graph declares directory writes for each composite.

Run-relative paths retain the reference `.factory/strategy/`, `.factory/reviews/`,
and `.factory/archive/` names. Director plans: research-plan.json and
strategy-plan.json under strategy/, qa-plan.json under reviews/. Reports are
under strategy/research/, strategy/perspectives/, and reviews/testers/.
Each director publishes `<stage_id>-findings.json` and `<stage_id>-findings.md`
under `.factory/`; these contain every child report's substantive content and
provenance. Quality summaries are separate from JSON plans.

Director knobs: `child_attempts` (default 2), `director_parallelism` (default 7).
Research 3..7, strategy 2..5, QA custom 2..5 plus `workflow-validate` and
`cli-integration`. Planner may use original direction/perspective/approach,
slug and prompt fields; normalize into typed tasks before dispatch. Require
DSH for director/researcher/strategist/reviewer execution. External builders
are optional only. No missing-report synthesis by a parent.

Gate adapter: agent gates retain `gate_kind="agent"`, original IDs, role `ceo`,
and source criteria in `prompt_template`. Candidate executor routes them through
`create_fidelity_gates.evaluate(ctx, node)` for strict model verdicts and
integrity guards. Human gate remains human and consumes fresh digest-bound
approval. Candidate-only behavior is selected by `create_fidelity` fact set by
init; avoid changing old graph/serialization defaults. Gate functions used by
native lowering must preserve logical evaluator ownership in receipts.

Add optional `timeout_s` to native Node without serializing unset defaults.
Per-stage declared allowances apply in both Python and native lowering.
Non-blocking archives must settle before terminal result, report errors, and
must never turn a failed final precheck into success. Gate HALT is a branch
label where an edge exists. The candidate final precheck records its outcome;
the build archivist diagnoses a failed outcome and completion stays failed.

All benchmark events retain raw evidence. Normalized events use schema_version,
trial_id, stage_id, visit, attempt, role, kind, task_id, parent_task_id,
plan_digest, input_digest, output_digest, requested_route, actual_route,
status, causal_predecessors, effective_deadlines, raw_evidence. Missing evidence
is unavailable or not_exercised, never passed. Human approvals are bound to the
exact strategy digest; no unconditional proceed answers.

Reviewed repair-path mapping: AI gate critique is written to
`.factory/reviews/repair-feedback.md` and delivered in full to builder and QA.
It does not rewrite human intent during builder-only repairs. Human strategy
rejection still appends the person's feedback to original intent and returns
through strategy planning and fresh approval.
