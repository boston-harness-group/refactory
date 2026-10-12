# Paired Create benchmark: build and execute the same native mode

Protocol v1, October 12, 2026. Implementation instructions for issue #310. This document specifies a harness to build; it does not claim the harness or live comparison is complete.

Route addendum, October 12: after the installed provider rejected `gpt-6.1-sol`,
the user explicitly authorized Sol 6 at high effort as a fallback. This is a
new frozen trial configuration, not retroactive Sol 6.1 acceptance. Actual
`gpt-6-sol` root and child receipts remain mandatory for that trial. The shared
brief, checker and artifact requirements are unchanged.

Authoring-context addendum, October 12: context v2 explicitly supplies the
external checker ownership already required below to every authoring lane.
In trial v6, the generated configuration instead made the intentionally hidden
checker an internal prerequisite, and health correctly reported that configured
threshold unmet. `authoring-boundary.md` now distinguishes local Create checks
from mandatory operator acceptance after delivery. Its own digest and each
fully assembled authoring brief are frozen in new trials. The shared public
booking requirements and independent checker bytes are unchanged; v6 retains
its original context and results. This clarification cannot make a missing
checker or incomplete authoring run pass either scorecard.

## Purpose and scope

Use remote-factory create-v2 and refactory-dsh Create to author `booking_conflicts_bench` from the same requirements. Invoke each generated mode through its own native runtime and compare both delivered capability and authoring fidelity. Keep the application small, offline and deterministic so the experiment primarily exposes differences in Create. Full Create, no fast-mode substitution. No UI, deployment, service, dataset iteration or optimization harness.

## Shared authoring brief

Create a reusable native mode named `booking_conflicts_bench` that accepts the runtime parameter `bookings_path`, reads that JSON file and writes `conflicts.json` into that invocation's run directory. A model may author the mode; executing the generated booking computation must not require a model. Package a Python 3.11+ standard-library CLI named `booking_conflicts.py` and a README alongside the framework's native mode artifacts.

| ID | Required behavior |
|---|---|
| B01 | Input is a JSON array of bookings; an empty array is valid. |
| B02 | Each booking has nonempty string `id` and `room`, and integer `start`/`end` minute offsets with `0 <= start < end <= 1440`. Booleans are not integers for this contract; extra fields may be ignored. |
| B03 | Booking IDs are globally unique across rooms. |
| B04 | Conflicts require exactly equal room strings and half-open interval overlap: `max(start1,start2) < min(end1,end2)`. Touching endpoints do not conflict. |
| B05 | Output is exactly an object with `conflicts`: unique ID pairs, IDs sorted within pairs and pairs sorted lexicographically. Input permutation does not change parsed output. |
| B06 | CLI is `python booking_conflicts.py --input INPUT.json --output OUTPUT.json`, works from an unrelated working directory with absolute paths, and returns 0 on valid input. |
| B07 | Invalid input returns nonzero with useful stderr, creates no output when absent, and leaves an existing CLI output byte-for-byte unchanged. A native mode invocation must propagate failure honestly without claiming a valid result. |
| B08 | Mode is discoverable, inspectable, validated, documented and runnable through the framework's actual native entry points. Framework-specific artifact formats are allowed. |
| B09 | Invoke the same generated package twice on distinct valid input files in separate run directories, without package modifications. Observe distinct expected results and correctly located `conflicts.json` files. |
| B10 | README documents installation/discovery and a native invocation with runtime `bookings_path`, plus the standalone CLI and ordinary examples. |

Both authoring lanes receive this content unchanged, plus only their framework-specific invocation instructions. Generated work stays local to disposable benchmark projects; no benchmark child may publish GitHub artifacts or modify framework source. Engine implementation and its eventual PR are separate from generated test projects.

Example native executions: input A is `a/A/[0,10)` and `b/A/[5,15)`, expecting `[["a","b"]]`; input B is `c/A/[0,10)` and `d/A/[10,20)`, expecting `[]`. A third invocation with duplicate IDs must fail. The full independent CLI checker covers 56 cases, including invalid-input preservation. Freeze its bytes and assert the expected case count; do not treat 0 tests as success.

## Harness architecture and isolation

Place versioned benchmark assets in the repository's test infrastructure, with a documented command for contract-only checks, controlled traces, live baseline collection, candidate runs and report generation. Pick concrete paths after inspecting updated upstream conventions. Ordinary CI must not accidentally start paid model runs; live execution is explicit.

Use one driver and thin remote/DSH adapters for setup, full Create launch, approval delivery, artifact discovery, native mode execution and trace normalization. Adapters must not author missing output, synthesize missing research reports, bypass gates or patch the engine to make the trial pass. Persist raw events alongside normalized events so adapter mistakes are auditable. Follow each framework's supported entry point and prove the intended mode and revision actually ran; changing from a skill route to a Python executor is a distinct trial configuration.

Separate read-isolated lane roots and artifacts. Same-machine directories alone are not read isolation. Configure the available permission/sandbox boundary and test that cross-lane artifact reads are denied before claiming an independent pair. Allow explicit read-only framework source/dependencies and the common brief. The independent checker and aggregate comparison run under the operator outside the authoring lanes. Fail or clearly mark an unavailable isolation guarantee; do not relabel it as achieved. Pin the shared public requirements and independently evaluate the generated output; hash the checker, inputs and artifacts into evidence.

Three immutable identities: `remote-reference`, `dsh-baseline` (updated upstream before fixes), `dsh-candidate` (each tested patch). Record commit, patch digest, dependencies, launch route, runtime, requested/actual provider/model/effort per role, prompt/brief/checker digests, source graph digest, deadlines, approvals and interventions. DSH directors/researchers/reviewers always execute in DSH. Optional external builder adapters are separate and recorded.

## Comparators and evidence

Delivered capability and authoring fidelity are separate scorecards, and both must pass. Compare B01–B10 and every obligation in the refreshed source-derived Create manifest. Reference vs candidate comparison uses allowed semantic behavior and partial-order dependencies, not byte-equal code or a total event sequence. Research with 3 children and research with 7 can both satisfy the same policy; independent peers may complete in either order.

Suggested normalized event fields: schema version, run/trial ID, stage ID, visit/attempt, semantic role, event kind, parent/child task ID, plan digest, input/output digests, route, verdict/status, causal predecessors, effective deadline layer and raw-evidence reference. Preserve actual child report content as artifacts; a filename list is insufficient evidence of consumption. Human strategy approval identifies the actual strategy digest and authorized approver. Do not insert unconditional approval responses.

Suggested report row: requirement ID, expected obligation, each lane's status and evidence links, first demonstrated divergence, known reference defect/native mapping if any, source location, regression test, issue/owner and candidate fix revision. Statuses are `passed`, `failed`, `unavailable`, `not_exercised`; missing evidence is never passed. Report the earliest provable causal departure, not a speculative root cause inferred from the first log difference.

Machine-readable JSON is the source for a short Markdown report and annotated HTML: side-by-side logical graphs, required stages and roles, actual child fanout, gate/repair paths and links to capability results. Keep authoring outcome separate from generated-mode outcome; a usable file left by an incomplete authoring run is not a completed Create run.

Self-check the harness with deliberately incorrect output (for example, treating touching endpoints as overlap), missing child evidence and an interrupted run. The corresponding checks must fail or become unavailable. Controlled orchestration tests must cross real graph/dispatch/join/report boundaries; fake external model/I/O responses only. Keep scripted trace evidence distinct from live model results.

## Development loop and release evidence

1. Refresh upstream, reconcile local changes, materialize the updated reference and review its delta. Freeze the manifest, brief, independent checker, adapter configuration and environment.
2. Run remote reference and pre-fix DSH in parallel after route/isolation preflight. Use Sol 6.1/high for substantive DSH runs and verify actual root/child routing. Record unmatched remote routes; do not infer causal quality/speed differences from them. Never silently downgrade to Mini.
3. Produce the initial gap matrix even if a lane is incomplete. Convert each repairable gap into a focused contract/controlled-trace regression before changing production behavior. An unavailable required route blocks live acceptance, but need not block offline implementation.
4. Implement the three parallel tracks in #310 against shared interfaces. Reuse the frozen remote evidence for short iterations; rerun affected controlled tests and integrated candidate runs at meaningful milestones. Inspect full child-report delivery and gate ownership, not just node presence.
5. Once controlled checks pass, run a fresh remote/candidate pair. Both modes must be generated and invoked; score all requirements independently. Reference failure is an explicit finding, not permission to weaken the contract. Preserve a failed paired status if the reference cannot complete.
6. Add real update-existing/plugin/provider-swap trials only after the initial new-mode trial is stable and before claiming those capabilities verified. Do not expand the initial application to make it a broad benchmark suite.

Use original documented per-stage allowances and an explicitly recorded overall cap suitable for all stages. The old 180s/900s experiment limits do not apply. Distinguish idle timeout, wall timeout, remaining run budget and shutdown grace. Choose and record actual caps at preflight; stop owned work on expiry and preserve evidence. A retry after a route/budget/contract change is a new trial, not an invisible continuation. No generated-artifact hand repairs.

One pair provides diagnostic evidence for a concrete task and configuration. It does not establish statistical superiority, universal provider compatibility or absence of all semantic drift. Deterministic contract tests enforce the broader structural/behavioral boundary continuously.
