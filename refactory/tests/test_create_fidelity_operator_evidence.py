"""Synthetic operator attestations test validation, never certify live output."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from benchmarks.create_fidelity import approval
from benchmarks.create_fidelity.contract import digest
from benchmarks.create_fidelity.evidence import authoring_score, capability_score, hash_tree
from benchmarks.create_fidelity.operator_evidence import (
    FILES, OperatorEvidenceContext, _command_words, _dispatch_evidence, _expanded_command,
    review_template, source_obligations, supplemental_row,
)
from test_create_fidelity_orchestration import begin_trial, approve_and_resume


@pytest.fixture(scope="module")
def completed_trial(tmp_path_factory):
    patch = pytest.MonkeyPatch()
    temporary = tmp_path_factory.mktemp("operator-evidence-controlled")
    graph, ctx, _, _ = begin_trial(temporary, patch)
    pending = approval.request(ctx.workdir, graph.nodes["gate_strategy"].prompt_template)
    approval.deliver(ctx.workdir, digest=pending["strategy_digest"], verdict="proceed", approver="test human", feedback="")
    approval.take(ctx.workdir, pending)
    outcome, _, _ = approve_and_resume(graph, ctx)
    assert outcome.exit_ok, outcome.error
    manifests = {stage: json.loads((ctx.workdir / f".factory/{stage}-findings.json").read_text())
                 for stage in ("research_director", "strategy_director", "qa_director")}
    yield ctx.workdir, manifests
    patch.undo()


def context(tmp_path, project, **changes):
    values = dict(root=tmp_path / "operator", trial_id="controlled-test-trial", lane_id="dsh-candidate",
                  reviewer_id="test independent operator", lane_roots=(project,))
    values.update(changes)
    return OperatorEvidenceContext(**values)


def filled_template(project, ctx, kind):
    record = review_template(project, ctx, kind)
    record.update(reviewed_at=datetime.now(timezone.utc).isoformat(), verdict="passed",
                  rationale="Synthetic unit-test review; this does not certify any live model result.")
    return record


def save(ctx, kind, record):
    path = ctx.root / ctx.lane_id / FILES[kind]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return path


def semantic_review(project, ctx):
    record = filled_template(project, ctx, "semantic")
    for assessment in record["assessments"]:
        duty = source_obligations()[assessment["obligation_id"]]
        inputs, outputs = _dispatch_evidence(project, duty["stage_id"], duty["role"])
        assert inputs and outputs, duty
        assessment.update(verdict="passed", rationale="Controlled fixture contains the supplied input and the actual assigned role output.")
        for paths in (inputs, outputs):
            name = next(name for name in sorted(paths) if (project / name).is_file() and (project / name).read_text().strip())
            quote = next(line for line in (project / name).read_text().splitlines() if line.strip())
            assessment["evidence"].append({"path": name, "sha256": digest(project / name), "quote": quote})
    return record


def test_optional_reviews_missing_and_unfilled_templates_never_pass(completed_trial, tmp_path):
    project, _ = completed_trial
    ctx = context(tmp_path, project)
    for kind, identifier in (("semantic", "F.obligations.semantic"), ("identity", "F.approval.identity"), ("readme", "B10")):
        assert supplemental_row(identifier, kind, project, ctx)["status"] == "unavailable"
        template = review_template(project, ctx, kind)
        assert template["verdict"] is None
        save(ctx, kind, template)
        assert supplemental_row(identifier, kind, project, ctx)["status"] == "failed"


def test_actual_protocol_allows_explicit_complete_operator_rubric(completed_trial, tmp_path):
    project, manifests = completed_trial
    ctx = context(tmp_path, project)
    save(ctx, "semantic", semantic_review(project, ctx))
    scored = {r["requirement_id"]: r for r in authoring_score([], {"exit_ok": True}, director_manifests=manifests,
                                                               evidence_root=project, operator_evidence=ctx)}
    row = scored["F.obligations.semantic"]
    assert row["status"] == "passed", row
    assert row["evaluator"]["reviewer_id"] == ctx.reviewer_id
    assert row["evaluator"]["operator_document_sha256"]
    assert row["evaluator"]["validator_sha256"] and row["evaluator"]["reference_sha256"]


@pytest.mark.parametrize("problem", ["stale-inventory", "stale-excerpt", "missing-duty", "blank-rationale", "wrong-trial", "wrong-reviewer", "contradictory-verdict"])
def test_semantic_review_rejects_stale_incomplete_or_self_certifying_claims(completed_trial, tmp_path, problem):
    project, _ = completed_trial
    ctx = context(tmp_path, project)
    record = semantic_review(project, ctx)
    if problem == "stale-inventory": record["raw_evidence_sha256"]["intent-ledger.md"] = "0" * 64
    elif problem == "stale-excerpt": record["assessments"][0]["evidence"][0]["sha256"] = "0" * 64
    elif problem == "missing-duty": record["assessments"].pop()
    elif problem == "blank-rationale": record["assessments"][0]["rationale"] = ""
    elif problem == "wrong-trial": record["trial_id"] = "other trial"
    elif problem == "wrong-reviewer": record["reviewer_id"] = "lane author"
    else: record["assessments"][0]["verdict"] = "failed"
    save(ctx, "semantic", record)
    assert supplemental_row("F.obligations.semantic", "semantic", project, ctx)["status"] == "failed"


def test_lane_authored_supplement_and_symlink_to_it_are_rejected(completed_trial, tmp_path):
    project, _ = completed_trial
    unsafe = context(tmp_path, project, root=project / "claimed-operator")
    record = semantic_review(project, unsafe)
    path = save(unsafe, "semantic", record)
    try:
        assert supplemental_row("F.obligations.semantic", "semantic", project, unsafe)["status"] == "failed"
        safe = context(tmp_path, project)
        safe.root.mkdir()
        (safe.root / safe.lane_id).symlink_to(path.parent, target_is_directory=True)
        assert supplemental_row("F.obligations.semantic", "semantic", project, safe)["status"] == "failed"
    finally:
        path.unlink()
        path.parent.rmdir()
        unsafe.root.rmdir()


def identity_review(project, ctx):
    record = filled_template(project, ctx, "identity")
    request = json.loads((project / "approval-required.json").read_text())
    receipt = next(path for path in (project / ".refactory/create-fidelity-gates").glob("*/verdict.json")
                   if json.loads(path.read_text()).get("owner") == "human")
    record.update(user_turn_reference="test-user-reply", question_reference="test-review-question", decision="proceed",
                  approver="test human", strategy_digest=request["strategy_digest"], input_paths=request["input_paths"],
                  answer_evidence_path=request["answer_path"], request_evidence_path="approval-required.json",
                  native_receipt_path=str(receipt.relative_to(project)))
    return record


def test_actual_user_reference_must_be_external_and_match_consumed_native_approval(completed_trial, tmp_path):
    project, manifests = completed_trial
    ctx = context(tmp_path, project)
    record = identity_review(project, ctx)
    save(ctx, "identity", record)
    assert supplemental_row("F.approval.identity", "identity", project, ctx)["status"] == "failed"
    trusted = {key: record[key] for key in ("question_reference", "decision", "approver", "strategy_digest", "input_paths")}
    ctx = context(tmp_path, project, trusted_user_references={"test-user-reply": trusted})
    rows = {r["requirement_id"]: r for r in authoring_score([], {"exit_ok": True}, director_manifests=manifests,
                                                          evidence_root=project, operator_evidence=ctx)}
    assert rows["F.approval.identity"]["status"] == "passed", rows["F.approval.identity"]
    bad = copy.deepcopy(record)
    bad["decision"] = "reloop"
    save(ctx, "identity", bad)
    assert supplemental_row("F.approval.identity", "identity", project, ctx)["status"] == "failed"


def test_readme_bare_boolean_is_not_evidence():
    assert capability_score(None, {"readme_verified": True})[-1]["status"] == "unavailable"


def test_documented_absolute_placeholders_and_shell_continuations_bind_actual_cli(tmp_path):
    project = tmp_path / "project"
    command = 'cd /tmp\npython /absolute/path/to/project/.factory/packages/booking_conflicts_bench/booking_conflicts.py \\\n  --input /absolute/path/to/bookings.json --output /absolute/path/to/result.json'
    expanded = _expanded_command(command, {'/absolute/path/to/project': str(project),
        '/absolute/path/to/bookings.json': str(tmp_path / 'input.json'), '/absolute/path/to/result.json': str(tmp_path / 'output.json')})
    words, cwd = _command_words(expanded, str(tmp_path))
    assert cwd == Path('/tmp')
    assert words == ['python', str(project / '.factory/packages/booking_conflicts_bench/booking_conflicts.py'),
                     '--input', str(tmp_path / 'input.json'), '--output', str(tmp_path / 'output.json')]
    assert command.startswith('cd /tmp\n'), 'the source command remains unchanged'


def test_documented_native_variables_do_not_add_a_newline_argument(tmp_path):
    command = '"$PY" -m refactory_dsh.create_fidelity_ops run booking_conflicts_bench/run \\\n  --project "$PROJECT" --workdir "$RUN_DIR" --knob "bookings_path=$INPUT"'
    expanded = _expanded_command(command, {'$PY': sys.executable, '$PROJECT': str(tmp_path),
                                           '$RUN_DIR': str(tmp_path / 'run'), '$INPUT': str(tmp_path / 'input.json')})
    words, _ = _command_words(expanded, str(tmp_path))
    assert '\n' not in words
    assert words[-1] == 'bookings_path=' + str(tmp_path / 'input.json')
    with pytest.raises(ValueError, match='placeholders'):
        _expanded_command(command, {'--project': str(tmp_path)})
    with pytest.raises(ValueError, match='placeholders'):
        _expanded_command('python /absolute/path/booking_conflicts.py', {'/absolute/path/booking_conflicts.py': str(tmp_path / 'other.py')})


@pytest.fixture
def readme_trial(tmp_path, request):
    native_failure = getattr(request, "param", None)
    project = tmp_path / "project"
    artifact = project / ".factory/packages/booking_conflicts_bench"
    artifact.mkdir(parents=True)
    (artifact / "booking_conflicts.py").write_text('''import argparse,json,os
from pathlib import Path
def conflicts(data):
    return {"conflicts": sorted(sorted([a["id"],b["id"]]) for i,a in enumerate(data) for b in data[i+1:]
        if a["room"]==b["room"] and max(a["start"],b["start"])<min(a["end"],b["end"]))}
def compute(ctx):
    if os.environ.get("TEST_NATIVE_FAILURE")=="execution": raise RuntimeError("native computation failed")
    ctx.write_json("conflicts.json",conflicts(json.loads(Path(ctx.knobs["bookings_path"]).read_text())))
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--input"); p.add_argument("--output"); a=p.parse_args()
    Path(a.output).write_text(json.dumps(conflicts(json.loads(Path(a.input).read_text()))))
''')
    (artifact / "native.py").write_text('''import argparse,os,sys
if os.environ.get("TEST_NATIVE_FAILURE")=="setup":
    print("configured graph profile unavailable",file=sys.stderr); raise SystemExit(1)
from pathlib import Path
from refactory_dsh.nn import Graph,Node,Mode,Knob
class BookingMode(Mode):
    @classmethod
    def declare(cls): return {"bookings_path":Knob("",kind="str",frozen=True)}
    def build(self):
        g=Graph(); g.add(Node("compute","op",call="booking_conflicts:compute",writes=("conflicts.json",))); return g
p=argparse.ArgumentParser(); p.add_argument("--bookings-path"); p.add_argument("--run-directory"); a=p.parse_args()
m=BookingMode(); state=m.state_dict(); state["knobs"]["bookings_path"]["value"]=a.bookings_path; m.load_state_dict(state)
r=m.run(Path(a.run_directory)); print(r.exit_ok)
if r.error: print(r.error,file=sys.stderr)
raise SystemExit(0 if r.exit_ok else 1)
''')
    commands = {"standalone": f"{sys.executable} SCRIPT/booking_conflicts.py --input INPUT --output OUTPUT",
                "native": f"{sys.executable} SCRIPT/native.py --bookings-path INPUT --run-directory RUN"}
    (artifact / "README.md").write_text("Installation: use Python with the framework installed.\n\n" + "\n\n".join(commands.values()))
    ctx = context(tmp_path, project)
    ctx.root.mkdir()
    package = hash_tree(artifact)
    input_file = ctx.root / "input.json"
    input_file.write_text(json.dumps([{"id": "a", "room": "A", "start": 0, "end": 10},
                                    {"id": "b", "room": "A", "start": 5, "end": 15}]))
    executions = []
    for kind, command in commands.items():
        output = ctx.root / ("standalone-output.json" if kind == "standalone" else "native-run/conflicts.json")
        substitutions = {"SCRIPT": str(artifact), "INPUT": str(input_file), "OUTPUT": str(output), "RUN": str(output.parent)}
        expanded = command
        for token, value in substitutions.items(): expanded = expanded.replace(token, value)
        started = datetime.now(timezone.utc).isoformat()
        cp = subprocess.run(shlex.split(expanded), cwd=ctx.root, capture_output=True, text=True, timeout=10,
                            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                                 "TEST_NATIVE_FAILURE": native_failure if kind == "native" and native_failure else ""})
        assert cp.returncode == (1 if kind == "native" and native_failure else 0), cp.stderr
        streams = {}
        for stream in ("stdout", "stderr"):
            path = ctx.root / f"{kind}-{stream}.txt"
            path.write_text(getattr(cp, stream))
            streams[stream] = {"path": path.name, "sha256": digest(path)}
        receipt = {"trial_id": ctx.trial_id, "lane_id": ctx.lane_id, "reviewer_id": ctx.reviewer_id, "kind": kind,
                   "documented_command": command, "substitutions": substitutions, "argv": shlex.split(expanded),
                   "cwd": str(ctx.root), "started": started, "finished": datetime.now(timezone.utc).isoformat(),
                   "returncode": cp.returncode, "package_before": package, "package_after": hash_tree(artifact),
                   "input_file": str(input_file), "output_file": str(output), "run_directory": str(output.parent),
                   "input": {"path": input_file.name, "sha256": digest(input_file)},
                   "output": {"path": str(output.relative_to(ctx.root)), "sha256": digest(output)} if output.exists() else None,
                   "expected_output": {"conflicts": [["a", "b"]]}, **streams}
        if kind == "native" and native_failure:
            receipt.update(execution_status="unavailable" if native_failure == "setup" else "failed",
                           reason="Configured native graph profile is absent" if native_failure == "setup" else "Actual native operation failed",
                           diagnostic={"stream": "stderr", "quote": "configured graph profile unavailable" if native_failure == "setup" else "native computation failed"})
        receipt_path = ctx.root / f"{kind}-command.json"
        receipt_path.write_text(json.dumps(receipt))
        executions.append({"kind": kind, "receipt_path": receipt_path.name, "sha256": digest(receipt_path)})
    record = filled_template(project, ctx, "readme")
    if native_failure:
        record["verdict"] = "unavailable" if native_failure == "setup" else "failed"
        record["rationale"] = "Actual setup diagnostic prevented the documented native route" if native_failure == "setup" else "Documented native execution returned a real computation failure"
    record.update(artifact_path=str(artifact.relative_to(project)), readme_path=str((artifact / "README.md").relative_to(project)),
                  readme_sha256=digest(artifact / "README.md"), package_sha256=package, executions=executions)
    save(ctx, "readme", record)
    native = {"package_before_import": package, "package_unchanged": True}
    return project, artifact, ctx, record, native


def test_readme_check_requires_actual_documented_cli_and_native_mode_execution(readme_trial):
    project, _, ctx, _, native = readme_trial
    row = capability_score(None, native, evidence_root=project, operator_evidence=ctx)[-1]
    assert row["status"] == "passed", row
    assert row["evaluator"]["operator_document_sha256"]


@pytest.mark.parametrize("readme_trial,expected", [("setup", "unavailable"), ("execution", "failed")], indirect=["readme_trial"])
def test_nonpassing_readme_review_retains_real_setup_or_computation_failure_without_output(readme_trial, expected):
    project, _, ctx, _, native = readme_trial
    row = capability_score(None, native, evidence_root=project, operator_evidence=ctx)[-1]
    assert row["status"] == expected, row
    assert row["evaluator"]["operator_document_sha256"], "valid nonpassing evidence must be retained as a reviewed result"
    assert "Operator evidence rejected" not in row["reason"]


@pytest.mark.parametrize("readme_trial", ["setup", "execution"], indirect=True)
def test_nonzero_missing_output_cannot_be_reclassified_as_passing_readme(readme_trial):
    project, _, ctx, record, native = readme_trial
    record["verdict"] = "passed"
    save(ctx, "readme", record)
    assert capability_score(None, native, evidence_root=project, operator_evidence=ctx)[-1]["status"] == "failed"


@pytest.mark.parametrize("problem", ["output-mismatch", "changed-argv", "stale-readme", "changed-package", "failed-command",
                                     "whole-command-substitution", "flag-substitution", "redirected-script"])
def test_readme_review_cannot_hide_failed_stale_or_unrelated_execution(readme_trial, problem):
    project, artifact, ctx, record, native = readme_trial
    item = record["executions"][0]
    path = ctx.root / item["receipt_path"]
    receipt = json.loads(path.read_text())
    if problem == "output-mismatch":
        output = ctx.root / receipt["output"]["path"]
        output.write_text('{"conflicts": []}')
        receipt["output"]["sha256"] = digest(output)
    elif problem == "changed-argv": receipt["argv"] = [sys.executable, "-c", "print('made up success')"]
    elif problem == "stale-readme": record["readme_sha256"] = "0" * 64
    elif problem == "changed-package":
        (artifact / "booking_conflicts.py").write_text("print('not the inspected package')")
        record["raw_evidence_sha256"] = hash_tree(project)
    elif problem == "whole-command-substitution":
        receipt["substitutions"] = {receipt["documented_command"]: str(ctx.root / "unrelated.py")}
    elif problem == "flag-substitution": receipt["substitutions"]["--output"] = str(ctx.root / "other-output")
    elif problem == "redirected-script":
        other = ctx.root / "other-implementation"
        other.mkdir()
        (other / "booking_conflicts.py").write_text("print('unrelated implementation')")
        receipt["substitutions"]["SCRIPT"] = str(other)
        receipt["argv"] = [word.replace(str(artifact), str(other)) for word in receipt["argv"]]
    else: receipt["returncode"] = 1
    path.write_text(json.dumps(receipt))
    item["sha256"] = digest(path)
    save(ctx, "readme", record)
    assert capability_score(None, native, evidence_root=project, operator_evidence=ctx)[-1]["status"] == "failed"
