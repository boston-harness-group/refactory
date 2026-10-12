"""Validate optional operator reviews; never manufacture a review or decision.

The caller supplies the trusted reviewer and user-turn context independently of
lane artifacts. Directory isolation is the harness's preflight responsibility.
These checks reject every lane-owned path, but cannot authenticate a platform
account or establish an OS permission boundary by parsing JSON.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
from typing import Any, Mapping

from .contract import FIXTURES, digest, reference

VERSION = "create-fidelity-operator-evidence/v1"
FILES = {"semantic": "semantic-review.json", "identity": "human-approval.json", "readme": "readme-review.json"}


@dataclass(frozen=True)
class OperatorEvidenceContext:
    root: Path
    trial_id: str
    lane_id: str
    reviewer_id: str
    lane_roots: tuple[Path, ...]
    # Allowlisted actual user decisions, provided by the operator channel,
    # never discovered from a model-writable project or from the review itself.
    trusted_user_references: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def evaluator_provenance() -> dict:
    return {"version": VERSION, "validator_sha256": digest(Path(__file__)),
            "scorecard_sha256": digest(Path(__file__).with_name("evidence.py")),
            "deadline_validator_sha256": digest(Path(__file__).with_name("deadline_evidence.py")),
            "contract_reader_sha256": digest(Path(__file__).with_name("contract.py")),
            "reference_sha256": digest(FIXTURES / "reference.json"),
            "scope": "Independent operator review of this task and frozen evidence; no universal semantic or account-authentication claim"}


def _path(root: Path, name: Any) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("evidence paths must be relative and remain within their declared owner")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()) or any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(root)):
        raise ValueError("evidence path escapes its owner or uses a symlink")
    return path


def source_obligations() -> dict[str, dict]:
    """Source blocks are rubric entries, not proof that an output satisfied them."""
    source = reference()
    roles = {Path(name).stem: text for name, text in source["role_prompts"].items()}
    obligations = {}
    for stage, node in source["graph"]["nodes"].items():
        if node["_type"] == "AgentNode":
            text, role = node["prompt_template"], node["role"]
        elif node.get("evaluator_type") == "agent":
            text, role = node["gate_prompt"], "ceo"
        else:
            continue
        for identifier, assigned_role, body in [(f"stage.{stage}", role, text),
                                                (f"role.{stage}.{role}", role, roles[role])]:
            obligations[identifier] = {"stage_id": stage, "role": assigned_role, "text": body,
                                       "source_sha256": hashlib.sha256(body.encode()).hexdigest()}
        child = {"research_director": "researcher", "strategy_director": "strategist", "qa_director": "adversarial_tester"}.get(stage)
        if child:
            body = roles[child]
            obligations[f"role.{stage}.{child}"] = {"stage_id": stage, "role": child, "text": body,
                                                   "source_sha256": hashlib.sha256(body.encode()).hexdigest()}
    return obligations


def review_template(project: Path, context: OperatorEvidenceContext, kind: str) -> dict:
    """Return an unfilled review scaffold and observed hashes; write nothing."""
    from .evidence import hash_tree
    result = {"schema_version": 1, "kind": kind, "trial_id": context.trial_id,
              "lane_id": context.lane_id, "project": str(project.resolve()), "reviewer_id": context.reviewer_id,
              "reference_sha256": digest(FIXTURES / "reference.json"), "raw_evidence_sha256": hash_tree(project),
              "reviewed_at": None, "verdict": None, "rationale": None}
    if kind == "semantic":
        result["assessments"] = [{"obligation_id": name, "source_sha256": duty["source_sha256"],
                                   "verdict": None, "rationale": None, "evidence": []}
                                  for name, duty in source_obligations().items()]
    return result


def _read(root: Path, name: str) -> dict:
    value = json.loads(_path(root, name).read_text())
    if not isinstance(value, dict):
        raise ValueError("operator evidence must be an object")
    return value


def _owned_document(project: Path, context: OperatorEvidenceContext, kind: str) -> tuple[Path, dict]:
    from .evidence import hash_tree
    root = context.root.resolve()
    if not context.lane_roots or project.resolve() not in {p.resolve() for p in context.lane_roots}:
        raise ValueError("operator context must enumerate this and all lane-writable roots")
    if any(root.is_relative_to(p.resolve()) or p.resolve().is_relative_to(root) for p in context.lane_roots):
        raise ValueError("operator evidence must be outside every lane-writable root")
    if any(not isinstance(value, str) or not value.strip() for value in (context.trial_id, context.lane_id, context.reviewer_id)):
        raise ValueError("trusted trial/lane/reviewer context is missing")
    path = _path(root, f"{context.lane_id}/{FILES[kind]}")
    document = _read(root, str(path.relative_to(root)))
    expected = {"schema_version": 1, "kind": kind, "trial_id": context.trial_id, "lane_id": context.lane_id,
                "project": str(project.resolve()), "reviewer_id": context.reviewer_id,
                "reference_sha256": digest(FIXTURES / "reference.json"), "raw_evidence_sha256": hash_tree(project)}
    if any(document.get(key) != value for key, value in expected.items()):
        raise ValueError("operator review identity, source, or current raw evidence hashes are stale or inconsistent")
    datetime.fromisoformat(document["reviewed_at"])
    _decision(document)
    return path, document


def _decision(record: dict) -> None:
    if record.get("verdict") not in {"passed", "failed", "unavailable"} or not isinstance(record.get("rationale"), str) or not record["rationale"].strip():
        raise ValueError("review needs an actual verdict and substantive reviewer rationale")


def _dispatch_evidence(project: Path, stage: str, role: str) -> tuple[set[str], set[str]]:
    inputs, outputs = set(), set()
    workers = {record["node_id"]: record for path in (project / ".refactory/workers").glob("*.json")
               for record in [_read(project, str(path.relative_to(project)))] if record.get("state") == "completed"}
    for path in (project / ".refactory/agent-delivery").glob("*/receipt.json"):
        delivery = _read(project, str(path.relative_to(project)))
        node = delivery.get("node", "")
        if (node != stage and not node.startswith(stage + ":")) or workers.get(node, {}).get("role") != role or delivery.get("status") != "delivered":
            continue
        inputs.update(item["prompt"] for item in delivery.get("contexts", []))
        for name, artifact in delivery.get("artifacts", {}).items():
            if artifact.get("kind") == "file":
                outputs.add(name)
            elif artifact.get("kind") == "directory":
                outputs.update(str(Path(name) / child) for child in artifact.get("files", {}))
    for path in (project / ".refactory/create-fidelity-gates").glob("*/verdict.json"):
        record = _read(project, str(path.relative_to(project)))
        if record.get("gate_id") == stage and record.get("owner") == "agent" and role == "ceo":
            inputs.add(str(path.with_name("prompt.txt").relative_to(project)))
            outputs.add(str(path.with_name("model-response.txt").relative_to(project)))
    return inputs, outputs


def _semantic(project: Path, document: dict) -> list[str]:
    obligations = source_obligations()
    assessments = document.get("assessments")
    if not isinstance(assessments, list) or len(assessments) != len(obligations) or {a.get("obligation_id") for a in assessments} != set(obligations):
        raise ValueError("semantic review must assess every frozen source stage/role obligation exactly once")
    refs = []
    for assessment in assessments:
        _decision(assessment)
        duty = obligations[assessment["obligation_id"]]
        if assessment.get("source_sha256") != duty["source_sha256"]:
            raise ValueError("assessment changed its frozen source obligation")
        inputs, outputs = _dispatch_evidence(project, duty["stage_id"], duty["role"])
        citations = assessment.get("evidence", [])
        cited = set()
        for citation in citations:
            name, quote = citation["path"], citation.get("quote")
            body = _path(project, name).read_text()
            if not isinstance(quote, str) or not quote.strip() or quote not in body or digest(_path(project, name)) != citation.get("sha256"):
                raise ValueError("semantic review excerpt or command evidence does not match the actual bytes")
            cited.add(name)
            refs.append(name)
        if not cited & inputs or not cited & outputs:
            raise ValueError("each semantic duty needs actual assigned model-input and role-output excerpts")
    statuses = {a["verdict"] for a in assessments}
    expected = "failed" if "failed" in statuses else "unavailable" if "unavailable" in statuses else "passed"
    if document["verdict"] != expected:
        raise ValueError("semantic aggregate verdict contradicts the individual source-duty assessments")
    return refs


def _identity(project: Path, document: dict, context: OperatorEvidenceContext) -> list[str]:
    turn = document.get("user_turn_reference")
    trusted = context.trusted_user_references.get(turn, {}) if isinstance(turn, str) else {}
    keys = ("question_reference", "decision", "approver", "strategy_digest", "input_paths")
    if not trusted or any(document.get(key) != trusted.get(key) for key in keys) or not all(document.get(key) for key in keys):
        raise ValueError("human approval does not match the operator's independently supplied actual user turn/question and decision")
    if document["decision"] != "proceed" or document["verdict"] != "passed":
        if document["verdict"] == "passed":
            raise ValueError("a human rejection cannot be certified as an approval")
        return []
    answer_path = document["answer_evidence_path"]
    answer = _read(project, answer_path)
    inputs = {name: digest(_path(project, name)) for name in document["input_paths"]}
    strategy_digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    if (answer.get("source") != "explicit-human" or answer.get("verdict") != "proceed"
            or answer.get("approver") != document["approver"] or answer.get("strategy_digest") != strategy_digest
            or strategy_digest != document["strategy_digest"] or answer.get("input_paths") != document["input_paths"]):
        raise ValueError("archived human decision or current approved input bytes are inconsistent")
    request_path = document["request_evidence_path"]
    request = _read(project, request_path)
    if (request.get("status") != "consumed" or request.get("answer_path") != answer_path
            or request.get("created_unix") != answer.get("request_created_unix")):
        raise ValueError("actual human request and archived answer do not show one-use consumption")
    receipt_path = document["native_receipt_path"]
    receipt = _read(project, receipt_path)
    review = hashlib.sha256(json.dumps({"gate": "gate_strategy", "files": inputs}, sort_keys=True).encode()).hexdigest()
    if (receipt.get("gate_id") != "gate_strategy" or receipt.get("owner") != "human"
            or receipt.get("verdict") != "proceed" or receipt.get("status") != "answered" or receipt.get("input_digest") != review):
        raise ValueError("consumed native gate receipt does not match the actual human-approved strategy digest")
    return [answer_path, request_path, receipt_path, *inputs]


def _expanded_command(command: str, substitutions: dict) -> str:
    expanded = command
    for token, value in substitutions.items():
        placeholder = (isinstance(token, str) and re.fullmatch(
            r"(?:[A-Z][A-Z0-9_]*|<[A-Za-z][A-Za-z0-9_]*>|\$\{[A-Za-z][A-Za-z0-9_]*\}|\$[A-Za-z][A-Za-z0-9_]*|/path/to/[A-Za-z0-9_/-]+|/absolute/[A-Za-z0-9_./-]+)", token))
        if (not placeholder or "booking_conflicts.py" in token or not isinstance(value, str)
                or ".." in Path(token).parts or not Path(value).is_absolute() or re.search(r"[\n\r;`$|&]", value)):
            raise ValueError("README substitutions are limited to explicit path/interpreter placeholders")
        pattern = re.compile(r"(?<![\w-])" + re.escape(token) + r"(?!\w)")
        # Unused parameter values are harmless; they cannot replace command
        # names, flags or a literal authored implementation filename.
        expanded = pattern.sub(lambda _: value, expanded)
    return expanded


def _command_words(expanded: str, cwd: str) -> tuple[list[str], Path]:
    # Shell continuations join source lines before lexical argument parsing.
    # Preserve the original README/receipt command; normalize only for binding.
    logical = expanded.replace("\\\r\n", "").replace("\\\n", "").strip()
    directory = Path(cwd)
    if not directory.is_absolute():
        raise ValueError("command cwd must identify the actual absolute invocation directory")
    first, separator, remainder = logical.partition("\n")
    prelude = shlex.split(first)
    if separator and len(prelude) == 2 and prelude[0] == "cd":
        selected = Path(prelude[1])
        directory = selected if selected.is_absolute() else directory / selected
        logical = remainder
    words = shlex.split(logical)
    if len(words) >= 4 and words[0] == "cd" and words[2] == "&&":
        selected = Path(words[1])
        directory = selected if selected.is_absolute() else directory / selected
        words = words[3:]
    if any(word in ("&&", ";", "|", "||") for word in words):
        raise ValueError("README receipt must isolate the actual documented invocation command")
    return words, directory


def _readme(project: Path, document: dict, context: OperatorEvidenceContext, native: dict) -> list[str]:
    artifact = _path(project, document["artifact_path"])
    readme = _path(project, document["readme_path"])
    text = readme.read_text()
    if digest(readme) != document.get("readme_sha256"):
        raise ValueError("reviewed README differs from the authored current bytes")
    from .evidence import hash_tree
    if any(path.is_symlink() for path in artifact.rglob("*")):
        raise ValueError("README artifact identity cannot omit linked implementation bytes")
    package = hash_tree(artifact)
    if not package or document.get("package_sha256") != package or package != native.get("package_before_import") or not native.get("package_unchanged"):
        raise ValueError("README execution evidence is not bound to the unchanged native artifact")
    executions = document.get("executions", [])
    if not isinstance(executions, list) or len(executions) != 2 or {r.get("kind") for r in executions} != {"standalone", "native"}:
        raise ValueError("README review needs actual standalone and native command execution receipts")
    refs = [document["readme_path"]]
    observed_statuses = []
    for item in executions:
        execution = _read(context.root, item["receipt_path"])
        if digest(_path(context.root, item["receipt_path"])) != item.get("sha256"):
            raise ValueError("operator command receipt changed after review")
        if any(execution.get(key) != value for key, value in {"trial_id": context.trial_id, "lane_id": context.lane_id,
                                                             "reviewer_id": context.reviewer_id, "kind": item["kind"]}.items()):
            raise ValueError("README execution belongs to a different operator, trial, lane or command kind")
        command = execution.get("documented_command")
        if not isinstance(command, str) or not command.strip() or command not in text:
            raise ValueError("executed command was not actually authored in README")
        expanded = _expanded_command(command, execution.get("substitutions", {}))
        argv = execution.get("argv")
        if argv not in (shlex.split(expanded), ["/bin/sh", "-c", expanded]):
            raise ValueError("actual argv does not execute the documented command with its recorded substitutions")
        if not execution.get("cwd") or execution.get("package_before") != package or execution.get("package_after") != package:
            raise ValueError("README command omitted its invocation context or modified the authored package")
        words, cwd = _command_words(expanded, execution["cwd"])
        if item["kind"] == "standalone":
            scripts = [Path(word) if Path(word).is_absolute() else cwd / word for word in words if Path(word).name == "booking_conflicts.py"]
            if len(scripts) != 1 or scripts[0].resolve() != (artifact / "booking_conflicts.py").resolve():
                raise ValueError("README standalone command did not execute the actual authored booking_conflicts.py")
        if datetime.fromisoformat(execution["finished"]) < datetime.fromisoformat(execution["started"]):
            raise ValueError("command execution interval is invalid")
        streams = {}
        for stream in ("stdout", "stderr"):
            details = execution[stream]
            if digest(_path(context.root, details["path"])) != details["sha256"]:
                raise ValueError("actual command stdout/stderr evidence changed")
            streams[stream] = _path(context.root, details["path"]).read_text(errors="replace")
        output = execution["output"]
        returncode = execution.get("returncode")
        status = execution.get("execution_status", "passed" if type(returncode) is int and returncode == 0 else "failed")
        if status not in {"passed", "failed", "unavailable"}:
            raise ValueError("README execution status is invalid")
        actual_output = Path(execution["output_file"])
        if not actual_output.is_absolute():
            raise ValueError("actual invocation output path must be absolute")
        if output is None:
            if status == "passed" or actual_output.exists():
                raise ValueError("passing/existing README output cannot be omitted from preserved evidence")
            observed_output = None
        else:
            out = _path(context.root, output["path"])
            if digest(out) != output["sha256"] or digest(actual_output) != output["sha256"]:
                raise ValueError("README execution output does not match its preserved actual result")
            streams["output"] = out.read_text(errors="replace")
            try:
                observed_output = json.loads(streams["output"])
            except ValueError:
                if status == "passed":
                    raise ValueError("passing README output is not valid JSON")
                observed_output = None
        if status == "passed":
            if type(returncode) is not int or returncode != 0 or observed_output != execution["expected_output"]:
                raise ValueError("a passing README execution needs actual zero status and the expected preserved output")
        else:
            reason, diagnostic = execution.get("reason"), execution.get("diagnostic", {})
            quote = diagnostic.get("quote")
            if (not isinstance(reason, str) or not reason.strip() or not isinstance(quote, str) or not quote.strip()
                    or quote not in streams.get(diagnostic.get("stream"), "")):
                raise ValueError("nonpassing README execution needs a precise reason and preserved diagnostic excerpt")
            if type(returncode) is not int and not execution.get("timed_out"):
                raise ValueError("nonpassing README execution omitted its actual process outcome")
            if document["verdict"] == "passed":
                raise ValueError("failed or unavailable execution cannot certify a passing README review")
        observed_statuses.append(status)
        if execution["expected_output"] not in ({"conflicts": [["a", "b"]]}, {"conflicts": []}):
            raise ValueError("README execution must use a frozen independently specified booking example")
        input_record = execution["input"]
        data = _read_input(context.root, input_record)
        actual_input = Path(execution["input_file"])
        if not actual_input.is_absolute() or digest(actual_input) != input_record["sha256"]:
            raise ValueError("preserved README input does not match the actual invocation file bytes")
        required = [{"id": "a", "room": "A", "start": 0, "end": 10}, {"id": "b", "room": "A", "start": 5, "end": 15}]
        touching = [{"id": "c", "room": "A", "start": 0, "end": 10}, {"id": "d", "room": "A", "start": 10, "end": 20}]
        if (data, execution["expected_output"]) not in ((required, {"conflicts": [["a", "b"]]}), (touching, {"conflicts": []})):
            raise ValueError("observed input/output does not bind a frozen booking example")
        output_file = execution.get("output_file")
        output_location = output_file if item["kind"] == "standalone" else execution.get("run_directory")
        if not isinstance(output_location, str) or not output_location or output_location not in expanded or execution.get("input_file") not in expanded:
            raise ValueError("README argv does not name the actual runtime input and per-invocation output location")
        if item["kind"] == "native" and output_file != str(Path(output_location) / "conflicts.json"):
            raise ValueError("native README execution did not write conflicts.json in its invocation directory")
        if item["kind"] == "standalone" and not all(token in expanded for token in ("booking_conflicts.py", "--input", "--output")):
            raise ValueError("standalone README example does not use the promised CLI")
        refs.append(str(_path(context.root, item["receipt_path"])))
    expected = "failed" if "failed" in observed_statuses else "unavailable" if "unavailable" in observed_statuses else "passed"
    if document["verdict"] != expected:
        raise ValueError("README review contradicts its actual execution statuses")
    return refs


def _read_input(root: Path, details: dict) -> Any:
    path = _path(root, details["path"])
    if digest(path) != details["sha256"]:
        raise ValueError("actual README invocation input changed")
    return json.loads(path.read_text())


def supplemental_row(identifier: str, kind: str, project: Path | None, context: OperatorEvidenceContext | None,
                     *, prerequisites: Mapping[str, str] | None = None, native: dict | None = None) -> dict:
    provenance = evaluator_provenance()
    result = {"requirement_id": identifier, "status": "unavailable", "reason": "No operator-owned supplemental review was supplied",
              "evidence": [], "evaluator": provenance}
    if project is None or context is None:
        return result
    try:
        project = project.resolve()
        path, document = _owned_document(project, context, kind)
        if prerequisites and any(status != "passed" for status in prerequisites.values()):
            raise ValueError("operator supplement cannot replace missing or failed raw protocol evidence")
        refs = (_semantic(project, document) if kind == "semantic" else _identity(project, document, context) if kind == "identity"
                else _readme(project, document, context, native or {}))
        result.update(status=document["verdict"], reason=document["rationale"], evidence=[str(path), *sorted(set(refs))])
        provenance.update(operator_document_sha256=digest(path), trial_id=context.trial_id, lane_id=context.lane_id,
                          reviewer_id=context.reviewer_id, trust_boundary="Operator context and files outside enumerated lane roots; platform authentication is not inferred")
    except FileNotFoundError as error:
        result["reason"] = f"Required operator review/evidence is absent: {error}"
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as error:
        result.update(status="failed", reason=f"Operator evidence rejected: {error}")
    return result
