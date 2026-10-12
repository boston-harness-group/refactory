"""Original CEO gates with strict model verdicts and fresh evidence guards.

The guards establish provenance, not semantic correctness. The original CEO
question still belongs to a DSH model; a marker in a report cannot approve it.
Human strategy decisions use the existing one-use review ledger.
"""
from __future__ import annotations

import hashlib
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from refactory_dsh.nn.graph import Node
from refactory_dsh.nn.executor import ExecutionError, RunContext, _ask_model
from refactory_dsh.steering import Ledger, SteeringError

ROOT = Path(".refactory/create-fidelity-gates")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _inputs(ctx: RunContext, reads: tuple[str, ...]) -> dict[str, Any]:
    from refactory_dsh.agent_delivery import _path

    files = {}
    for name in sorted(reads):
        path = _path(ctx.workdir, name)
        if path.is_dir():
            files[name] = {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(path.rglob("*")) if p.is_file()}
        else:
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return files


def _evidence(ctx: RunContext, node: Node) -> tuple[dict[str, Any], list[str]]:
    from refactory_dsh.agent_delivery import receipts_in

    inputs = _inputs(ctx, node.reads)
    problems = [f"missing declared input: {name}" for name, value in inputs.items() if value is None]
    receipts = {}
    for producer, receipt in receipts_in(ctx.workdir).items():
        relevant = set(receipt.get("writes", [])) & set(node.reads)
        # QA synthesis consumes these original report producers before the CEO.
        if node.id in ("gate_qa", "gate_doc_freshness", "gate_overwatch"):
            required_reviews = {
                ".factory/reviews/health-check.md", ".factory/reviews/code-review.md",
            }
            if node.id in ("gate_doc_freshness", "gate_overwatch"):
                required_reviews.add(".factory/reviews/overwatch-latest.md")
            relevant |= set(receipt.get("writes", [])) & required_reviews
        if not relevant:
            continue
        receipts[producer] = {key: receipt.get(key) for key in ("id", "node", "started", "finished", "status", "execution", "contexts")}
        receipts[producer]["artifacts"] = {name: receipt["artifacts"].get(name) for name in sorted(relevant)}
        receipts[producer]["current_artifacts"] = {name: receipt["current_artifacts"].get(name) for name in sorted(relevant)}
        if receipt.get("status") != "delivered" or receipts[producer]["artifacts"] != receipts[producer]["current_artifacts"]:
            problems.append(f"stale or undelivered evidence from {producer}")
        before = receipt.get("execution", {}).get("input_fingerprints")
        if before is None:
            problems.append(f"producer {producer} has no input digest evidence")
        else:
            current = _inputs(ctx, tuple(before))
            if current != before:
                problems.append(f"producer {producer} consumed stale inputs")
    evidence: dict[str, Any] = {"inputs": inputs, "deliveries": receipts}
    expected = {"gate_build": ("builder",), "gate_qa": ("health_checker", "code_reviewer"),
                "gate_overwatch": ("overwatch",), "gate_doc_freshness": ("overwatch",)}.get(node.id, ())
    problems.extend(f"missing producer receipt: {producer}" for producer in expected if producer not in receipts)
    if node.id in ("gate_build", "gate_qa", "gate_overwatch", "gate_doc_freshness"):
        from refactory_dsh.create_fidelity_ops import package_digest, validate_native_package, _native_root
        try:
            name = str(ctx.facts["name"])
            project = Path(ctx.facts["artifact_project"])
            root = _native_root(project, name)
            admission = ctx.read_json(".factory/reviews/native-admission.json", {})
            current_digest = package_digest(root)
            evidence["native_package"] = {"package_digest": current_digest, "admission": admission}
            if not admission.get("ok") or admission.get("package_digest") != current_digest:
                problems.append("native admission is missing, failed, or stale against current package")
            if node.id == "gate_build":
                requirements = tuple(p for p in str(ctx.facts.get("requires_input", "")).split(",") if p)
                check = validate_native_package(project, name, requires_input=requirements)
                ctx._gate_validation_evidence = getattr(ctx, "_gate_validation_evidence", {})
                ctx._gate_validation_evidence[node.id] = check
                # Raw validation is retained in the gate record but only stable
                # identity enters freshness checks; subprocess timings can differ.
                evidence["native_package"]["validation"] = {key: check[key] for key in ("ok", "problems", "package_digest")}
                if not check["ok"]:
                    problems.extend(check["problems"])
        except (KeyError, ValueError, OSError) as error:
            problems.append(f"native package evidence unavailable: {error}")
    return evidence, problems


def current_approval(ctx: RunContext, gate: str) -> bool:
    """Whether the latest CEO approval still certifies these exact input bytes."""
    from refactory_dsh.lowering import read_declarations
    entry = read_declarations(ctx.workdir).get(gate, {})
    logical = entry.get("logical_node")
    if logical is None:
        logical = getattr(ctx, "_gate_nodes", {}).get(gate)
    if logical is None:
        return False
    node = logical if isinstance(logical, Node) else Node.from_state(logical)
    if node.gate_kind == "user":
        return human_current_approval(ctx, node=node)
    records = []
    for path in (ctx.workdir / ROOT).glob("*/verdict.json"):
        record = json.loads(path.read_text())
        if record.get("gate_id") == gate:
            records.append((path.stat().st_mtime_ns, record))
    if not records:
        return False
    record = max(records, key=lambda item: item[0])[1]
    evidence, problems = _evidence(ctx, node)
    return not problems and record.get("verdict") == "proceed" and record.get("evidence_digest") == _digest(evidence)


def human_current_approval(ctx: RunContext, *, node: Node | None = None) -> bool:
    node = node or Node("gate_strategy", "gate", gate_kind="user",
                        reads=(".factory/strategy/current.md", ".factory/strategy/user-intent.md"))
    records = []
    for path in (ctx.workdir / ROOT).glob("*/verdict.json"):
        record = json.loads(path.read_text())
        if record.get("gate_id") == node.id and record.get("owner") == "human":
            records.append((path.stat().st_mtime_ns, record))
    if not records:
        return False
    record = max(records, key=lambda item: item[0])[1]
    return record.get("verdict") == "proceed" and record.get("input_digest") == Ledger(ctx.workdir).review(node.id, node.reads)


def _record(ctx: RunContext, record: dict[str, Any], *, raw: str = "", prompt: str = "") -> None:
    directory = ctx.workdir / ROOT / record["id"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "verdict.json").write_text(json.dumps(record, indent=2) + "\n")
    (directory / "model-response.txt").write_text(raw)
    (directory / "prompt.txt").write_text(prompt)


def human_verdict(ctx: RunContext, node: Node, labels: list[str]) -> str:
    """Consume a fresh review answer; standing or plain string approvals refuse."""
    ledger = Ledger(ctx.workdir)
    review = ledger.visit(node.id, node.reads)
    verdict = _take_human(ledger, node.id, review, labels)
    if not verdict:
        raise ExecutionError(f"gate {node.id!r} needs a fresh human approval bound to review {review}")
    if verdict == "reloop":
        _human_feedback(ctx, ledger, node.id)
    record = {"schema_version": 1, "id": uuid.uuid4().hex, "gate_id": node.id,
              "owner": "human", "input_digest": review, "verdict": verdict,
              "status": "answered", "source": "intent-ledger.md"}
    _record(ctx, record)
    return verdict


def _feedback(ctx: RunContext, gate: str, text: str, *, human: bool = False) -> None:
    from datetime import datetime, timezone
    path = ctx.path(".factory/strategy/user-intent.md" if human else ".factory/reviews/repair-feedback.md")
    if not path.is_file():
        raise ExecutionError("repair feedback needs the declared feedback artifact")
    with path.open("a") as stream:
        source = "Human feedback" if human else "Model critique (does not change the approved user request)"
        stream.write(f"\n## [{datetime.now(timezone.utc).isoformat()}] {source} at {gate}\n{text}\n")


def _human_feedback(ctx: RunContext, ledger: Ledger, gate: str) -> None:
    entries = ledger.entries()
    feedback = next((entry["text"] for entry in reversed(entries) if entry["kind"] == "correction"), "")
    if not feedback.strip():
        raise ExecutionError("strategy revision requires the human's correction in the intent ledger")
    _feedback(ctx, gate, feedback, human=True)


def _take_human(ledger: Ledger, gate: str, review: str, labels: list[str]) -> str:
    latest = next((ledger._fields(entry) for entry in reversed(ledger.entries())
                   if entry["kind"] == "verdict" and ledger._fields(entry).get("gate") == gate), {})
    if latest.get("scope") == "run":
        raise SteeringError(f"candidate gate {gate!r} requires digest-bound approval; standing approval is refused")
    return ledger.take_verdict(gate, review, labels)


def evaluate(ctx: RunContext, node: Node) -> str:
    """Ask the original CEO, then enforce freshness on the selected branch."""
    labels = list(getattr(ctx, "_gate_labels", ()))
    if not labels:
        from refactory_dsh.lowering import read_declarations
        labels = list(read_declarations(ctx.workdir).get(node.id, {}).get("gate_labels", []))
    if node.gate_kind == "user":
        return human_verdict(ctx, node, labels)
    if node.gate_kind != "agent" or not node.prompt_template:
        raise ExecutionError("candidate model gate needs the original agent gate criteria")
    if ctx.model is None:
        raise ExecutionError(f"gate {node.id!r} needs a DSH model client")
    client = ctx.model
    if callable(getattr(client, "for_node", None)):
        client = client.for_node(node.id, role=node.role)
    if callable(getattr(client, "with_node_timeout", None)):
        client = client.with_node_timeout(node.timeout_s)
    route = client.route() if callable(getattr(client, "route", None)) else {}
    if route.get("worker_backend") != "dsh":
        raise ExecutionError("Create fidelity CEO gates require DSH execution")
    evidence, problems = _evidence(ctx, node)
    input_digest = _digest(evidence["inputs"])
    evidence_digest = _digest(evidence)
    record = {"schema_version": 1, "id": uuid.uuid4().hex, "gate_id": node.id,
              "owner": "agent", "role": node.role, "input_digest": input_digest,
              "evidence_digest": evidence_digest, "evidence": evidence,
              "guard_problems": problems, "requested_route": route, "status": "running",
              "effective_deadlines": client.effective_deadlines() if callable(getattr(client, "effective_deadlines", None)) else {}}
    request = {"schema_version": 1, "gate_id": node.id, "input_digest": input_digest,
               "evidence_digest": evidence_digest, "verdict_labels": labels,
               "guard_problems": problems, "evidence": evidence}
    from refactory_dsh.create_fidelity_ops import agent_prompt
    prompt = agent_prompt(ctx, node) + "\n\nRuntime evidence (read current artifacts before judging):\n" + json.dumps(request, indent=2)
    prompt += ("\nReturn ONLY a JSON object with exactly schema_version (1), gate_id, input_digest, "
               "evidence_digest, verdict (one supplied label), and rationale (nonempty string). "
               "Echo the digests exactly. Guard problems cannot authorize proceed. Do not modify artifacts.")
    raw = ""
    workers_before = set((ctx.workdir / ".refactory/workers").glob("*.json"))
    try:
        raw = _ask_model(client, prompt, system="You are the original Create CEO gate evaluator.", workdir=ctx.workdir)
        workers = [json.loads(path.read_text()) for path in sorted(
            set((ctx.workdir / ".refactory/workers").glob("*.json")) - workers_before)]
        record["worker_receipts"] = workers
        matched = [receipt for receipt in workers if receipt.get("node_id") == node.id]
        record["actual_route"] = matched[-1].get("actual_route") if matched else None
        if not matched or not record["actual_route"] or record["actual_route"].get("worker_backend") != "dsh":
            raise ExecutionError("CEO gate actual DSH route is unavailable or inconsistent")
        answer = json.loads(raw)
        expected = {"schema_version", "gate_id", "input_digest", "evidence_digest", "verdict", "rationale"}
        if not isinstance(answer, dict) or set(answer) != expected:
            raise ExecutionError("model gate response is not strict verdict JSON")
        if (type(answer["schema_version"]) is not int or answer["schema_version"] != 1
                or answer["gate_id"] != node.id or answer["input_digest"] != input_digest
                or answer["evidence_digest"] != evidence_digest
                or answer["verdict"] not in labels
                or not isinstance(answer["rationale"], str) or not answer["rationale"].strip()):
            raise ExecutionError("model gate response has invalid labels, identity, rationale or stale digests")
        after, _ = _evidence(ctx, node)
        if _digest(after) != evidence_digest:
            raise ExecutionError("gate evidence changed during evaluation; verdict is stale")
        verdict = answer["verdict"]
        if problems and verdict == "proceed":
            if "reloop" not in labels:
                raise ExecutionError("integrity guards refused model approval: " + "; ".join(problems))
            verdict = "reloop"
        record.update(status="guard_refused" if problems else "answered", requested_verdict=answer["verdict"],
                      verdict=verdict, rationale=answer["rationale"])
        if verdict == "reloop":
            feedback = answer["rationale"]
            if problems:
                feedback += "\nIntegrity evidence: " + "; ".join(problems)
            _feedback(ctx, node.id, feedback)
        return verdict
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["raw_native_validation"] = getattr(ctx, "_gate_validation_evidence", {}).get(node.id)
        _record(ctx, record, raw=raw, prompt=prompt)


def evaluate_native(ctx: RunContext) -> str:
    """Command bridge preserving an agent gate's logical node and CEO ownership."""
    from refactory_dsh.lowering import read_declarations
    from refactory_dsh.model import DshModelClient

    entry = read_declarations(ctx.workdir).get(ctx.node_id, {})
    node = Node.from_state(entry["logical_node"])
    ctx.model = DshModelClient()
    verdict = evaluate(ctx, node)
    ctx._internal_fact(node.id + "_verdict", verdict)
    return verdict


def human_main() -> int:
    """The native begin/poll/answer protocol with candidate approval policy."""
    try:
        request = json.load(sys.stdin)
        ledger = Ledger(Path.cwd())
        gate = str(request["gate"])
        review = ledger.review(gate, request["reads"])
        labels = request["labels"]
        if request["action"] == "begin":
            review = ledger.visit(gate, request["reads"])
            result = {"review": review, "verdict": _take_human(ledger, gate, review, labels)}
        elif request["action"] == "poll":
            if review != request["review"]:
                raise SteeringError("strategy inputs changed while awaiting approval")
            result = {"review": review, "verdict": _take_human(ledger, gate, review, labels)}
        elif request["action"] == "answer":
            if review != request["review"] or request["verdict"] not in labels:
                raise SteeringError("strategy answer is stale or invalid")
            if request["verdict"] == "reloop":
                feedback = request.get("feedback", "")
                if not isinstance(feedback, str) or not feedback.strip():
                    raise SteeringError("strategy revision requires the human's current feedback")
                ledger.note("correction", feedback.strip())
            ledger.verdict(gate, request["verdict"], review=review, answered=True)
            result = {"review": review, "verdict": request["verdict"]}
        elif request["action"] == "confirm":
            if review != request["review"]:
                raise SteeringError("strategy inputs changed after the pending human answer")
            receipt_id = request.get("receipt_id", "")
            if not isinstance(receipt_id, str) or len(receipt_id) != 32 or any(c not in "0123456789abcdef" for c in receipt_id):
                raise SteeringError("pending human answer has no valid receipt identity")
            receipt = RunContext(Path.cwd()).read_json(str(ROOT / receipt_id / "verdict.json"), {})
            if (receipt.get("owner") != "human" or receipt.get("gate_id") != gate
                    or receipt.get("verdict") != request["verdict"] or receipt.get("status") != "answered"):
                raise SteeringError("pending human answer receipt is inconsistent")
            if request["verdict"] == "proceed" and receipt.get("input_digest") != review:
                raise SteeringError("pending human approval is stale")
            result = {"review": review, "verdict": request["verdict"], "confirmed": True}
        else:
            raise SteeringError("unknown human gate action")
        if result.get("verdict") and request["action"] != "confirm":
            if result["verdict"] == "reloop":
                _human_feedback(RunContext(Path.cwd()), ledger, gate)
            identifier = uuid.uuid4().hex
            _record(RunContext(Path.cwd()), {"schema_version": 1, "id": identifier,
                    "gate_id": gate, "owner": "human", "input_digest": review,
                    "verdict": result["verdict"], "status": "answered", "source": "intent-ledger.md"})
            result.update(receipt_id=identifier, current_review=ledger.review(gate, request["reads"]))
        result["protocol"] = "create_fidelity_human_v1"
        print(json.dumps(result))
        return 0
    except Exception as error:
        print(f"human gate: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(human_main())
