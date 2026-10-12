# Paired Create benchmark

Run from the repository root with `PYTHONPATH=refactory`. This driver does not
launch model work unless `live` is explicitly selected and preflight succeeds.
Use isolated source checkouts or git archives at the revisions in
`tests/fixtures/create_fidelity/v1/baseline.json`. Do not point it at a mutable
original checkout. No benchmark operation repairs authored deliverables.

```sh
python -m benchmarks.create_fidelity.harness prepare /tmp/create-pair \
  --reference /path/to/pinned/remote-factory \
  --baseline /path/to/pinned/refactory-dsh \
  --candidate /path/to/candidate/refactory-dsh \
  --dsh-python /path/to/dsh/.venv/bin/python \
  --remote-python /path/to/remote/.venv/bin/python \
  --runtime-root /path/to/installed/refactory-dsh \
  --hidden /path/to/historical/evidence
python -m benchmarks.create_fidelity.harness preflight /tmp/create-pair \
  --route-receipts /path/to/actual/sol61-high-probe
python -m benchmarks.create_fidelity.harness live /tmp/create-pair
```

Preparation freezes source bytes, lockfiles, brief, checker, topology contract,
runtime route, and timeout allowances. Changing any source requires a new trial.
Authoring context v2 also supplies the same explicit external-check ownership
to every lane, with a separate frozen digest. The public booking brief and
56-case checker stay unchanged. The checker runs under the operator after
authoring; internal gates must still evaluate actual local tests and defects,
while final benchmark acceptance cannot pass without the external checker.
Older trials retain their original context and cannot gain this clarification
through re-scoring.
Supply `--baseline-runtime-root` and `--candidate-runtime-root` to keep each
worker launcher anchored to its frozen checkout. These override the common
`--runtime-root`. Installed dependency bytes, symlink bindings and launcher
bytes have a separate identity checked before and after execution. Shared
published dependencies may be copied into a disposable snapshot; workspace
bindings must point into the corresponding frozen checkout. Do not use the
original dirty checkout as an unrecorded runtime dependency.
The declared 12-hour full-mode ceiling is an initial cap, not a calibrated
completion estimate. Director stages retain 3600 seconds; builder and Overwatch
retain 1800. Timeout records distinguish the layer that stopped execution and
whether process cleanup was confirmed.

`live` runs the reference and updated DSH baseline concurrently. Candidate work
can be invoked separately with `--lanes dsh-candidate`. The request and input
files are identical except for explicitly disclosed native runtime mappings.
The reference uses its native role defaults; those aliases are not evidence of
actual served models. DSH requires observed Sol 6.1/high for directors and
children. An unavailable route blocks the launch, without fallback.

The user authorized Sol 6/high after Sol 6.1 was unavailable. Use
`--model gpt-6-sol --route-authorization "User approved Sol 6/high fallback on 2026-10-12"`
when preparing that distinct trial. Its receipts are evaluated against Sol 6,
and must not be described as Sol 6.1 evidence.

Each lane pauses for real human approval after producing a strategy. Inspect
`LANE/project/approval-required.json` and the strategy and intent files,
then deliver exactly that decision:

```sh
python -m benchmarks.create_fidelity.harness approve /tmp/create-pair dsh-candidate \
  --strategy-digest DIGEST_FROM_PENDING_REQUEST --verdict proceed \
  --approver "human reviewer"
```

`reloop` requires feedback. Decisions bind exact strategy and intent bytes and
are consumed once. Never preapprove or fabricate the human's answer.
The pending request names its exact `input_paths`: candidate/reference use
`.factory/strategy/current.md` and `user-intent.md`; baseline uses `strategy.json`
and `brief.json`. Baseline recovery uses its documented native
`Mode.run(start=checkpoint.node, facts=checkpoint.facts)` API because the pinned
`Mode.resume` rejects newly added human ledger context. This mapping is recorded
in `raw/resume-*.json`; the native gate still consumes the fresh answer itself.

After authoring, verify delivered code with native discovery and distinct
invocation directories, then run the unchanged standalone checker:

```sh
python -m benchmarks.create_fidelity.harness native /tmp/create-pair dsh-candidate
python -m benchmarks.create_fidelity.harness check /tmp/create-pair dsh-candidate
python -m benchmarks.create_fidelity.harness report /tmp/create-pair
```

The report includes JSON, Markdown and annotated HTML under `operator/report/`.
Static source graphs and runtime observations are labeled separately. Missing
traces, semantic reviews, receipts or invocations cannot become passing evidence.
The unchanged 56-case checker is independently scored from native package reuse.
Do not compare speed or cost causally from one diagnostic pair.

Independent semantic, human-decision and README reviews belong under
`operator/reviews/LANE/`, outside every authoring project. The optional
`operator/review-context.json` identifies the operator reviewer and allowlists
actual user decisions in `trusted_user_references`, grouped by lane. It must
come from the real review conversation, never from a worker's output or a
claimed approver name. Without that evidence the relevant rows stay unavailable.
The operator review validator binds source obligations, current raw file hashes,
actual command receipts and the exact consumed approval. An unfilled review
template is not a passing review. See the [operator evidence schema](operator_evidence.md)
for the required fields and unfilled templates.

Report generation records the evaluator version, source hashes and review-context
hash in `operator/report/evaluator.json` and the JSON/HTML comparison. This
separates later independent evaluation from the frozen code that authored the
artifact; re-scoring cannot rewrite a trial's source or runtime identity. Reviews
must follow native/CLI verification, because any later change to the raw evidence
invalidates their hash binding. Observed deadline snapshots are available only
for runs whose worker runtime actually recorded them; older trials cannot gain
missing timing telemetry through re-scoring.

OS isolation currently requires macOS Seatbelt. It proves peer/evidence content
denial in child processes and private runtime writes. It is not a claim of a
general adversarial security boundary. Keep all historical evidence roots in
`--hidden` and never copy result artifacts between lane directories.
