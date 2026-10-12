"""Fail-closed evidence normalization and independent dual scorecards."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .contract import FIXTURES, reference
from .deadline_evidence import dispatch_deadline_row
from .operator_evidence import OperatorEvidenceContext, supplemental_row

STATUSES = {"passed", "failed", "unavailable", "not_exercised"}


def hash_tree(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()
            and "__pycache__" not in p.parts and ".git" not in p.parts}


def row(identifier: str, status: str, reason: str, evidence: list[str] | None = None) -> dict:
    if status not in STATUSES:
        raise ValueError(f"unknown evidence status: {status}")
    return {"requirement_id": identifier, "status": status, "reason": reason,
            "evidence": evidence or []}


def route_score(receipts: list[dict], *, model: str = "gpt-6.1-sol", effort: str = "high") -> dict:
    if not receipts:
        return row("F.route", "unavailable", "No served root/child route receipts")
    missing = [r for r in receipts if not r.get("actual_route")]
    if missing:
        return row("F.route", "unavailable", "A worker did not report its served configuration")
    wrong = [r for r in receipts if r["actual_route"].get("worker_backend") != "dsh"
             or r["actual_route"].get("model") != model
             or r["actual_route"].get("effort") != effort
             or r["actual_route"].get("source") != "runtime-request"
             or any((r.get("requested_route") or {}).get(key)
                    and r["actual_route"].get(key) != r["requested_route"][key]
                    for key in ("worker_backend", "provider", "model", "effort"))]
    return row("F.route", "failed" if wrong else "passed",
               f"Observed configuration differs from requested routing or DSH {model}/{effort}" if wrong else
               f"All supplied worker receipts observed DSH {model}/{effort}; stage coverage scored separately")


def normalize(raw: list[dict], *, trial_id: str, source: str) -> list[dict]:
    """Keep raw indexes and explicit observations; never infer a successful missing stage."""
    events = []
    visits: dict[str, int] = {}
    for index, event in enumerate(raw):
        stage = event.get("node_id") or event.get("node") or event.get("stage_id")
        if not stage:
            continue
        raw_kind = event.get("kind") or event.get("event") or event.get("type") or "observation"
        kind = {"node.started": "node_started", "node.completed": "node_completed", "node.failed": "node_failed",
                "NodeStarted": "node_started", "NodeCompleted": "node_completed", "NodeFailed": "node_failed"}.get(raw_kind, raw_kind)
        status = event.get("state") or event.get("status") or {
            "node_started": "running", "node_completed": "completed", "node_failed": "failed"}.get(kind, "unknown")
        if kind in {"node_started", "NodeStarted"}:
            visits[stage] = visits.get(stage, 0) + 1
        events.append({"schema_version": 1, "trial_id": trial_id, "stage_id": stage,
                       "visit": visits.get(stage, 1), "attempt": event.get("attempt"),
                       "role": event.get("role"), "kind": kind, "raw_kind": raw_kind, "status": status,
                       "task_id": event.get("task_id"), "parent_task_id": event.get("parent_task_id"),
                       "plan_digest": event.get("plan_digest"), "input_digest": event.get("input_digest"),
                       "output_digest": event.get("output_digest"),
                       "requested_route": event.get("requested_route"),
                       "actual_route": event.get("actual_route") or event.get("route"),
                       "causal_predecessors": event.get("causal_predecessors", []),
                       "effective_deadlines": event.get("effective_deadlines"),
                       "verdict": event.get("verdict"), "raw_evidence": f"{source}#record={index}"})
    return events


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _evidence_path(root: Path, name: Any) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("raw evidence path must be relative to its run")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("raw evidence path escapes its run")
    return path


def _read_evidence(root: Path, name: Any) -> dict:
    value = json.loads(_evidence_path(root, name).read_text())
    if not isinstance(value, dict):
        raise ValueError("raw evidence must be a JSON object")
    return value


def _verify_director_envelope(root: Path, report: dict, *, stage: str, visit: int,
                              run_id: str, role: str, task_id: str,
                              plan_normalization: bool = False) -> list[str]:
    if (report.get("stage_id") != stage or report.get("visit") != visit or report.get("run_id") != run_id
            or report.get("parent_task_id") != stage or report.get("task_id") != task_id
            or report.get("role") != role or report.get("status") != "completed"
            or type(report.get("attempt")) is not int or report["attempt"] < 1):
        raise ValueError("report identity/status does not bind this run, stage, visit, role and attempt")
    node = f"{stage}:v{visit}:{task_id}:a{report['attempt']}"
    if report.get("node_id") != node:
        raise ValueError("report dispatch identity does not bind its attempt")
    body = report.get("body")
    if not isinstance(body, str) or report.get("output_digest") != hashlib.sha256(body.encode()).hexdigest():
        raise ValueError("embedded report body does not match its output digest")
    output = _evidence_path(root, report.get("output"))
    if not plan_normalization and output.read_text() != body:
        raise ValueError("current report file differs from embedded substantive content")
    raw = report.get("raw_evidence")
    if (not isinstance(raw, dict) or any(not isinstance(raw.get(key), list) or len(raw[key]) != 1
                                        for key in ("worker_receipts", "delivery_receipts"))):
        raise ValueError("report needs one raw worker and delivery receipt")
    worker = _read_evidence(root, raw["worker_receipts"][0])
    delivery = _read_evidence(root, raw["delivery_receipts"][0])
    if (worker.get("node_id") != node or worker.get("role") != role or worker.get("state") != "completed"
            or worker.get("run_id") != str(root.resolve()) or worker.get("native_observations")):
        raise ValueError("raw worker did not complete the assigned DSH role in this run")
    requested, actual = report.get("requested_route"), report.get("actual_route")
    if (not isinstance(requested, dict) or not isinstance(actual, dict)
            or requested.get("worker_backend") != "dsh" or actual.get("worker_backend") != "dsh"
            or worker.get("requested_route") != requested or worker.get("actual_route") != actual
            or any(requested.get(key) and requested[key] != actual.get(key) for key in ("provider", "model", "effort"))):
        raise ValueError("raw requested/actual DSH route differs from the report")
    if delivery.get("node") != node or delivery.get("status") != "delivered" or delivery.get("missing_or_unchanged"):
        raise ValueError("raw delivery is stale, incomplete or belongs to another child")
    artifact = delivery.get("artifacts", {}).get(report["output"], {})
    if artifact.get("kind") != "file" or artifact.get("sha256") != report["output_digest"]:
        raise ValueError("raw delivered artifact does not match the report digest")
    stat = output.stat()
    observed = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns, "inode": stat.st_ino}
    if not plan_normalization and any(artifact.get(key) != value for key, value in observed.items()):
        raise ValueError("report output changed since raw delivery")
    contexts = delivery.get("contexts")
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("raw delivery omits its supplied model input")
    prompt = _evidence_path(root, contexts[-1].get("prompt"))
    if report.get("input_digest") != hashlib.sha256(prompt.read_bytes()).hexdigest():
        raise ValueError("input digest does not match the actual supplied model context")
    if delivery.get("execution", {}).get("node") != node or not isinstance(delivery.get("execution", {}).get("input_fingerprints"), dict):
        raise ValueError("raw delivery omits exact input fingerprint evidence")
    return [*raw["worker_receipts"], *raw["delivery_receipts"], str(prompt.relative_to(root))]


def _director_evidence_score(name: str, manifest: dict | None, root: Path | None,
                             lower: int, upper: int, mandatory: list[str]) -> dict:
    identifier = f"F.children.{name}"
    if not manifest:
        return row(identifier, "unavailable", "No independently dispatched child manifest")
    if root is None:
        return row(identifier, "unavailable", "Raw director worker/delivery evidence was not provided for independent inspection")
    root = Path(root)
    try:
        if (manifest.get("status") != "completed" or manifest.get("stage_id") != name
                or type(manifest.get("visit")) is not int or manifest["visit"] < 1
                or not isinstance(manifest.get("run_id"), str) or not manifest["run_id"]):
            raise ValueError("director manifest is incomplete or lacks bound stage/run/visit identity")
        plan_path = ".factory/reviews/qa-plan.json" if name == "qa_director" else (
            ".factory/strategy/research-plan.json" if name == "research_director" else ".factory/strategy/strategy-plan.json")
        plan = _read_evidence(root, plan_path)
        if plan.get("stage_id") != name or manifest.get("plan_digest") != _canonical_digest(plan):
            raise ValueError("director plan digest does not match the published typed plan")
        tasks, reports = plan.get("tasks"), manifest.get("reports")
        if not isinstance(tasks, list) or not isinstance(reports, list):
            raise ValueError("director plan/reports are not typed arrays")
        ids = [report.get("task_id") for report in reports]
        custom = [task_id for task_id in ids if task_id not in mandatory]
        if (not lower <= len(custom) <= upper or any(not isinstance(task_id, str) or not task_id for task_id in ids)
                or len(set(ids)) != len(ids) or not set(mandatory) <= set(ids)):
            raise ValueError("director child count, unique identities or mandatory testers are invalid")
        by_id = {task.get("id"): task for task in tasks}
        if len(by_id) != len(tasks) or set(by_id) != set(ids):
            raise ValueError("reports do not exactly cover independently planned children")
        proposal = _read_evidence(root, f".factory/directors/{name}/v{manifest['visit']}/plan-receipt.json")
        proposed = json.loads(proposal.get("body", ""))
        if (proposal.get("plan_digest") != manifest["plan_digest"] or not isinstance(proposed, dict)
                or not isinstance(proposed.get("tasks"), list)
                or len(proposed["tasks"]) != len(custom)):
            raise ValueError("typed plan is not bound to the actual director proposal")
        proposed_ids = set()
        for original in proposed["tasks"]:
            task_id = original.get("id", original.get("slug"))
            if task_id in proposed_ids or task_id not in by_id or task_id in mandatory:
                raise ValueError("director proposal has duplicate, unknown or mandatory custom child IDs")
            proposed_ids.add(task_id)
            actual_task = by_id[task_id]
            focus = original.get("focus", original.get("direction", original.get("perspective", original.get("approach"))))
            if (actual_task.get("slug") != original.get("slug") or actual_task.get("focus") != focus
                    or actual_task.get("prompt") != original.get("prompt")
                    or actual_task.get("coverage") != original.get("coverage")
                    or actual_task.get("role") != original.get("role", actual_task.get("role"))
                    or actual_task.get("output") != original.get("output", actual_task.get("output"))):
                raise ValueError("published child task differs from what the director model proposed")
        refs = _verify_director_envelope(root, proposal, stage=name, visit=manifest["visit"], run_id=manifest["run_id"],
                                        role="ceo", task_id="plan", plan_normalization=True)
        reviews = manifest.get("quality_reviews")
        if not isinstance(reviews, list) or not reviews or reviews[-1].get("coverage_complete") is not True:
            raise ValueError("director quality review did not establish full coverage")
        final = reviews[-1]
        decisions = final.get("decisions")
        if (not isinstance(decisions, list) or len(decisions) != len(ids)
                or {decision.get("id") for decision in decisions} != set(ids)
                or any(decision.get("verdict") != "accept" for decision in decisions)
                or any(decision.get("outcome") != "passed" for decision in decisions if decision.get("id") in mandatory)):
            raise ValueError("director did not accept each child and pass both mandatory QA testers")
        quality = _read_evidence(root, final.get("raw_receipt"))
        quality_answer = json.loads(quality.get("body", ""))
        if quality_answer != {key: value for key, value in final.items() if key != "raw_receipt"}:
            raise ValueError("published quality verdict differs from the actual director model response")
        if quality.get("plan_digest") != manifest["plan_digest"]:
            raise ValueError("director quality response belongs to another plan")
        refs.extend(_verify_director_envelope(root, quality, stage=name, visit=manifest["visit"], run_id=manifest["run_id"], role="ceo", task_id="quality"))
        role = {"research_director": "researcher", "strategy_director": "strategist", "qa_director": "adversarial_tester"}[name]
        minimum_bytes = 100 if name == "strategy_director" else 50
        for report in reports:
            task = by_id[report["task_id"]]
            if (len(report.get("body", "").encode()) <= minimum_bytes or task.get("role") != role
                    or report.get("output") != task.get("output") or report.get("task_digest") != _canonical_digest(task)
                    or report.get("plan_digest") != manifest["plan_digest"]):
                raise ValueError("child body, task binding, assigned output or plan digest is invalid")
            refs.extend(_verify_director_envelope(root, report, stage=name, visit=manifest["visit"], run_id=manifest["run_id"], role=role, task_id=report["task_id"]))
        if len({report["output_digest"] for report in reports}) != len(reports):
            raise ValueError("independent children delivered duplicate report bodies")
        return row(identifier, "passed", "Independently verified complete child work, route, delivery and director quality receipts", refs)
    except (OSError, FileNotFoundError) as error:
        return row(identifier, "unavailable", f"Raw director evidence is unavailable: {error}")
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        return row(identifier, "failed", f"Director evidence verification failed: {error}")


def _raw_records(root: Path, pattern: str) -> list[tuple[str, dict]]:
    return [(str(path.relative_to(root)), _read_evidence(root, str(path.relative_to(root))))
            for path in sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime_ns)]


def _delivery_prompt(root: Path, pair: tuple[str, dict], workers: list[tuple[str, dict]] | None = None) -> tuple[str, list[str]]:
    path, receipt = pair
    if receipt.get("status") != "delivered" or receipt.get("missing_or_unchanged"):
        raise ValueError("model input belongs to an undelivered worker")
    contexts = receipt.get("contexts")
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("delivery omitted its supplied model context")
    prompt_path = contexts[-1].get("prompt")
    prompt = _evidence_path(root, prompt_path).read_text()
    if receipt.get("execution", {}).get("node") != receipt.get("node"):
        raise ValueError("delivery execution identity differs from its worker")
    if not isinstance(receipt.get("execution", {}).get("input_fingerprints"), dict):
        raise ValueError("delivery omitted its exact input fingerprints")
    refs = [path, prompt_path]
    if workers is not None:
        matching = [(worker_path, worker) for worker_path, worker in workers
                    if worker.get("node_id") == receipt.get("node") and worker.get("state") == "completed"
                    and worker.get("run_id") == str(root) and worker.get("cwd") == str(root)
                    and receipt.get("started", "") <= worker.get("started", "")
                    and worker.get("finished", "") <= receipt.get("finished", "")]
        if not matching:
            raise FileNotFoundError("supplied context lacks an actual completed worker within its delivery interval")
        refs.extend(worker_path for worker_path, _ in matching)
    return prompt, refs


def _latest_delivery(deliveries: list[tuple[str, dict]], node: str) -> tuple[str, dict]:
    matches = [pair for pair in deliveries if pair[1].get("node") == node]
    if not matches:
        raise FileNotFoundError(f"no raw delivery for {node}")
    return max(matches, key=lambda pair: str(pair[1].get("started", "")))


def _body_supplied(prompt: str, body: str) -> bool:
    # Complete bodies may be supplied as raw Markdown or as JSON string values.
    return body in prompt or any(json.dumps(body, ensure_ascii=ascii_only) in prompt for ascii_only in (False, True))


def _verified_gate(root: Path, pair: tuple[str, dict]) -> tuple[str, list[str]]:
    path, receipt = pair
    parent = Path(path).parent
    prompt_path, response_path = str(parent / "prompt.txt"), str(parent / "model-response.txt")
    prompt = _evidence_path(root, prompt_path).read_text()
    answer = json.loads(_evidence_path(root, response_path).read_text())
    guarded = receipt.get("status") == "guard_refused"
    if (receipt.get("owner") != "agent" or receipt.get("status") not in ("answered", "guard_refused")
            or not isinstance(answer, dict) or answer.get("gate_id") != receipt.get("gate_id")
            or answer.get("verdict") != receipt.get("requested_verdict", receipt.get("verdict"))
            or answer.get("input_digest") != receipt.get("input_digest")
            or answer.get("evidence_digest") != receipt.get("evidence_digest")
            or answer.get("rationale") != receipt.get("rationale")):
        raise ValueError("gate verdict does not match its actual strict model response")
    if guarded and (receipt.get("verdict") != "reloop" or not receipt.get("guard_problems")):
        raise ValueError("guard override omitted its actual integrity failures")
    evidence = receipt.get("evidence")
    # Runtime gate digests deliberately use ASCII JSON, unlike director digests.
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if not isinstance(evidence, dict) or digest(evidence) != receipt.get("evidence_digest"):
        raise ValueError("gate verdict evidence digest is invalid")
    if digest(evidence.get("inputs")) != receipt.get("input_digest"):
        raise ValueError("gate input digest is invalid")
    return prompt, [path, prompt_path, response_path]


def _protocol_rows(root: Path | None, events: list[dict], manifests: dict,
                   operator_evidence: OperatorEvidenceContext | None = None) -> list[dict]:
    identifiers = ("F.obligations", "F.consumption", "F.approval", "F.repair", "F.deadlines")
    if root is None:
        return [row(identifier, "not_exercised" if identifier == "F.repair" and not any(e.get("verdict") == "reloop" for e in events) else "unavailable",
                    "No raw run evidence for independent protocol inspection") for identifier in identifiers]
    root = Path(root).resolve()
    try:
        deliveries = _raw_records(root, ".refactory/agent-delivery/*/receipt.json")
        gates = _raw_records(root, ".refactory/create-fidelity-gates/*/verdict.json")
        workers = _raw_records(root, ".refactory/workers/*.json")
    except (OSError, ValueError, TypeError) as error:
        return [row(identifier, "unavailable", f"Raw evidence inventory could not be read: {error}") for identifier in identifiers]

    def inspect(identifier, fn):
        try:
            reason, refs = fn()
            return row(identifier, "passed", reason, sorted(set(refs)))
        except (OSError, FileNotFoundError) as error:
            return row(identifier, "unavailable", f"Required raw evidence is absent: {error}")
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            return row(identifier, "failed", f"Independent protocol verification failed: {error}")

    def obligations():
        source, refs = reference(), []
        roles = {Path(path).stem: text for path, text in source["role_prompts"].items()}
        for node, definition in source["graph"]["nodes"].items():
            if definition.get("_type") == "AgentNode":
                targets = [node]
                if node.endswith("_director"):
                    manifest = manifests.get(node)
                    if not manifest:
                        raise FileNotFoundError(f"no director manifest for inherited {node} duties")
                    targets = [f"{node}:v{manifest['visit']}:plan:a1", f"{node}:v{manifest['visit']}:quality:a{len(manifest['quality_reviews'])}"]
                    targets.extend(report["node_id"] for report in manifest["reports"])
                for target in targets:
                    pair = _latest_delivery(deliveries, target)
                    prompt, evidence = _delivery_prompt(root, pair, workers)
                    matching = [worker for _, worker in workers if worker.get("node_id") == target and worker.get("state") == "completed"]
                    if not matching:
                        raise FileNotFoundError(f"no completed raw worker for {target}")
                    role = matching[-1].get("role")
                    expected_role = definition.get("role") if target == node or ":plan:" in target or ":quality:" in target else {"research_director": "researcher", "strategy_director": "strategist", "qa_director": "adversarial_tester"}[node]
                    if role != expected_role or matching[-1].get("run_id") != str(root):
                        raise ValueError("raw worker role/run identity differs from the inherited source owner")
                    if role not in roles or roles[role] not in prompt:
                        raise ValueError(f"actual {target} context omitted full inherited {role} duties")
                    if target == node or ":plan:" in target or ":quality:" in target:
                        if definition["prompt_template"] not in prompt:
                            raise ValueError(f"actual {target} context omitted source stage duties")
                    refs.extend(evidence)
            elif definition.get("evaluator_type") == "agent":
                matching = [pair for pair in gates if pair[1].get("gate_id") == node]
                if not matching:
                    raise FileNotFoundError(f"no raw CEO evaluation for {node}")
                prompt, evidence = _verified_gate(root, matching[-1])
                if roles["ceo"] not in prompt or definition["gate_prompt"] not in prompt:
                    raise ValueError(f"actual CEO gate {node} omitted inherited duties/criteria")
                refs.extend(evidence)
        return "Full inherited role and stage instructions were supplied to actual workers/CEO gates; semantic adequacy is scored separately", refs

    def consumption():
        refs = []
        def supplied_input(prompt, receipt, name):
            body = _evidence_path(root, name).read_text()
            fingerprints = receipt.get("execution", {}).get("input_fingerprints", {})
            if fingerprints.get(name) != hashlib.sha256(body.encode()).hexdigest() or not _body_supplied(prompt, body):
                raise ValueError(f"actual {receipt.get('node')} context omitted or consumed stale input {name}")
        # All fixed source agents must consume the exact current declared input
        # bodies. Composite director envelopes are checked separately below.
        for name, definition in reference()["graph"]["nodes"].items():
            if definition.get("_type") == "AgentNode" and not name.endswith("_director"):
                pair = _latest_delivery(deliveries, name)
                prompt, evidence = _delivery_prompt(root, pair, workers)
                for input_name in definition["reads"]:
                    supplied_input(prompt, pair[1], input_name)
                refs.extend(evidence)
        for name, lower, upper, mandatory in (("research_director", 3, 7, []), ("strategy_director", 2, 5, []), ("qa_director", 2, 5, ["workflow-validate", "cli-integration"])):
            manifest = manifests.get(name)
            checked = _director_evidence_score(name, manifest, root, lower, upper, mandatory)
            if checked["status"] == "unavailable":
                raise FileNotFoundError(checked["reason"])
            if checked["status"] != "passed":
                raise ValueError(checked["reason"])
            refs.extend(checked["evidence"])
            quality = _read_evidence(root, manifest["quality_reviews"][-1]["raw_receipt"])
            consumers = [quality["node_id"]]
            if name == "research_director":
                next_manifest = manifests["strategy_director"]
                consumers.append(f"strategy_director:v{next_manifest['visit']}:plan:a1")
            elif name == "strategy_director":
                consumers.append("synthesize_strategy")
            else:
                synthesized = _evidence_path(root, ".factory/reviews/qa-synthesized.md").read_text()
                if any(report["body"] not in synthesized for report in manifest["reports"]):
                    raise ValueError("QA synthesis omitted substantive child report content")
                consumers.append("overwatch")
            for consumer in consumers:
                prompt, evidence = _delivery_prompt(root, _latest_delivery(deliveries, consumer), workers)
                if any(not _body_supplied(prompt, report["body"]) for report in manifest["reports"]):
                    raise ValueError(f"actual {consumer} context omitted full child report bodies")
                refs.extend(evidence)
        synthesized_path = ".factory/reviews/qa-synthesized.md"
        synthesized = _evidence_path(root, synthesized_path).read_text()
        for name in (".factory/reviews/health-check.md", ".factory/reviews/code-review.md"):
            if _evidence_path(root, name).read_text() not in synthesized:
                raise ValueError("QA synthesis omitted the complete health/code branch reports")
        qa_gate = next((pair for pair in reversed(gates) if pair[1].get("gate_id") == "gate_qa"), None)
        if qa_gate is None:
            raise FileNotFoundError("no raw QA gate context")
        prompt, evidence = _verified_gate(root, qa_gate)
        if synthesized not in prompt:
            raise ValueError("CEO QA evaluation did not receive the complete synthesized reports")
        refs.extend([synthesized_path, *evidence])
        return "Raw model contexts prove complete digest-bound reports reached director review, strategy synthesis, QA synthesis, Overwatch and the CEO QA gate", refs

    def approval():
        text = _evidence_path(root, "intent-ledger.md").read_text()
        entries = []
        for section in re.split(r"(?m)^## \[", text)[1:]:
            heading, _, body = section.partition("\n")
            _, _, kind = heading.partition("] ")
            fields = {key.strip(): value.strip() for line in body.splitlines() for key, separator, value in [line.partition(":")] if separator}
            entries.append((kind.strip(), fields))
        answers = [fields for kind, fields in entries if kind == "verdict" and fields.get("gate") == "gate_strategy"]
        if not answers:
            raise FileNotFoundError("no human strategy answer in the raw ledger")
        answer = answers[-1]
        inputs = {name: hashlib.sha256(_evidence_path(root, name).read_bytes()).hexdigest() for name in reference()["graph"]["nodes"]["gate_strategy"]["reads"]}
        review = hashlib.sha256(json.dumps({"gate": "gate_strategy", "files": inputs}, sort_keys=True).encode()).hexdigest()
        if answer.get("verdict") != "proceed" or answer.get("scope") != review or answer.get("source") not in ("steer", "interactive"):
            raise ValueError("human answer is unconditional, non-proceed, or stale against the exact strategy/intent bytes")
        uses = [fields for kind, fields in entries if kind == "verdict-use" and fields.get("id") == answer.get("id")]
        if answer.get("consumed") == "yes":
            if answer.get("source") != "interactive" or uses:
                raise ValueError("interactive approval has invalid replay/consumption evidence")
        elif len(uses) != 1 or uses[0].get("result") != "used":
            raise ValueError("human pending approval was not consumed exactly once")
        receipts = [pair for pair in gates if pair[1].get("gate_id") == "gate_strategy" and pair[1].get("owner") == "human"]
        if not receipts:
            raise FileNotFoundError("no native human gate acceptance receipt")
        receipt = receipts[-1][1]
        if receipt.get("verdict") != "proceed" or receipt.get("input_digest") != review:
            raise ValueError("native human gate receipt does not bind the latest exact approved revision")
        return "Raw human answer and native gate receipt bind the current exact strategy/intent digest and prove one-use consumption; authenticated approver identity is a separate limitation", ["intent-ledger.md", receipts[-1][0], *inputs]

    def repair():
        reloops = [pair for pair in gates if pair[1].get("owner") == "agent" and pair[1].get("verdict") == "reloop"]
        if not reloops:
            if any(e.get("verdict") == "reloop" for e in events) or len([pair for pair in deliveries if pair[1].get("node") == "builder"]) > 1:
                raise FileNotFoundError("normalized reloop lacks a raw gate decision")
            return None
        refs = []
        builders = sorted([pair for pair in deliveries if pair[1].get("node") == "builder"], key=lambda pair: pair[1].get("started", ""))
        for pair in reloops:
            _, evidence = _verified_gate(root, pair)
            gate_time = _evidence_path(root, pair[0]).stat().st_mtime
            after = [builder for builder in builders if datetime.fromisoformat(builder[1]["started"]).timestamp() > gate_time]
            before = [builder for builder in builders if datetime.fromisoformat(builder[1]["finished"]).timestamp() <= gate_time]
            if not before or not after:
                raise FileNotFoundError("reloop has no preceding completed build and subsequent repair dispatch")
            prompt, builder_refs = _delivery_prompt(root, after[0], workers)
            if pair[1]["rationale"] not in prompt:
                raise ValueError("repair worker did not receive the actual gate feedback")
            def authored_bytes(builder):
                artifact = builder[1].get("artifacts", {}).get(".factory/packages/", {})
                files = artifact.get("files")
                if not isinstance(files, dict):
                    raise FileNotFoundError("build receipts omit actual authored package bytes")
                return {name: details.get("sha256") for name, details in files.items()
                        if "__pycache__" not in Path(name).parts and not name.endswith((".origin.json", ".native-admission.json"))}
            if authored_bytes(before[-1]) == authored_bytes(after[0]):
                raise ValueError("repair changed no authored package/document bytes")
            completed_workers = [worker for _, worker in workers if worker.get("node_id") == "builder" and worker.get("state") == "completed" and worker.get("started", "") >= after[0][1]["started"]]
            if not completed_workers:
                raise FileNotFoundError("repair delivery lacks a completed actual builder worker")
            refs.extend([*evidence, before[-1][0], *builder_refs])
        return "Observed raw CEO reloop decisions dispatched fresh builders with the exact critique and changed actual authored bytes; repair correctness remains subject to independent acceptance", refs

    def deadlines():
        if not workers:
            raise FileNotFoundError("no actual worker deadline receipts")
        refs = []
        allowances = {name: definition.get("timeout") for name, definition in reference()["graph"]["nodes"].items()}
        for path, worker in workers:
            if worker.get("run_id") != str(root) or worker.get("cwd") != str(root):
                raise ValueError("worker deadline receipt belongs to another invocation")
            limits = worker.get("effective_deadlines")
            if not isinstance(limits, dict) or not {"completion_timeout_s", "stage_timeout_s", "declared_stage_timeout_s", "shutdown_grace_s", "run_budget"} <= set(limits):
                raise FileNotFoundError("worker omitted effective completion/stage/grace/budget deadline layers")
            for key in ("completion_timeout_s", "stage_timeout_s", "shutdown_grace_s"):
                if type(limits[key]) not in (int, float) or limits[key] <= 0:
                    raise ValueError("effective deadline is invalid")
            node = worker.get("node_id", "")
            stage = node.split(":", 1)[0]
            expected = allowances.get(stage)
            if expected and ":" not in node and limits["declared_stage_timeout_s"] != expected:
                raise ValueError(f"worker {node} did not retain the frozen source stage allowance")
            if expected and ":" in node and (not limits["declared_stage_timeout_s"] or limits["stage_timeout_s"] > expected):
                raise ValueError("director dispatch wall exceeds or omits its declared enclosing allowance")
            if worker.get("state") in ("starting", "running") and not worker.get("finished"):
                raise FileNotFoundError("worker is still active; terminal settlement evidence is not yet available")
            if worker.get("state") not in ("completed", "failed", "cancelled", "timed_out", "budget_exhausted") or not worker.get("finished"):
                raise ValueError("worker remained in flight or its shutdown/cleanup is unconfirmed")
            elapsed = (datetime.fromisoformat(worker["finished"]) - datetime.fromisoformat(worker["started"])).total_seconds()
            if elapsed < 0 or elapsed > limits["stage_timeout_s"] + limits["shutdown_grace_s"] + 2:
                raise ValueError("observed worker interval exceeded its recorded wall/shutdown envelope")
            refs.append(path)
        return "Raw actual workers retain source allowances and effective completion/stage/shutdown/budget layers, and every supplied worker settled; idle/remaining-budget telemetry and expiry enforcement are separate limitations", refs

    rows = [inspect("F.obligations", obligations), inspect("F.consumption", consumption), inspect("F.approval", approval)]
    try:
        result = repair()
        rows.append(row("F.repair", "not_exercised", "No raw AI gate reloop branch occurred; successful completion does not exercise repair") if result is None else row("F.repair", "passed", result[0], result[1]))
    except (OSError, FileNotFoundError) as error:
        rows.append(row("F.repair", "unavailable", f"Repair evidence is incomplete: {error}"))
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        rows.append(row("F.repair", "failed", f"Repair protocol evidence is invalid: {error}"))
    statuses = {item["requirement_id"]: item["status"] for item in rows}
    rows.extend([inspect("F.deadlines", deadlines),
                 supplemental_row("F.obligations.semantic", "semantic", root, operator_evidence,
                                  prerequisites={key: statuses[key] for key in ("F.obligations", "F.consumption")}),
                 supplemental_row("F.approval.identity", "identity", root, operator_evidence,
                                  prerequisites={"F.approval": statuses["F.approval"]}),
                 dispatch_deadline_row(workers),
                 row("F.deadlines.expiry", "not_exercised" if workers and all(worker.get("state") == "completed" for _, worker in workers) else "unavailable",
                     "Successful settled workers do not prove shutdown on expiry; failed/expired states need independent owned-process cleanup evidence")])
    return rows


def authoring_score(events: list[dict], outcome: dict, *, director_manifests: dict | None = None,
                    workers: list[dict] | None = None, lane: str = "dsh", evidence_root: Path | None = None,
                    expected_model: str = "gpt-6.1-sol", operator_evidence: OperatorEvidenceContext | None = None) -> list[dict]:
    latest = {}
    lifecycle = {"node_started", "NodeStarted", "node_completed", "NodeCompleted", "node_failed", "NodeFailed"}
    for event in events:
        if event["kind"] in lifecycle:
            latest[event["stage_id"]] = event
    rows = []
    for name in reference()["graph"]["nodes"]:
        last = latest.get(name, {})
        kind = last.get("kind")
        status = "passed" if kind in {"node_completed", "NodeCompleted"} else "failed" if kind in {"node_failed", "NodeFailed"} else "unavailable" if last else "not_exercised"
        reason = "Latest stage visit completed" if status == "passed" else "Latest stage visit failed" if status == "failed" else "Latest stage visit started but has no terminal receipt" if last else "No stage visit was observed"
        evidence = [event["raw_evidence"] for event in events if event["stage_id"] == name]
        if status == "passed" and any(event["kind"] in {"node_failed", "NodeFailed"} for event in events if event["stage_id"] == name):
            reason += "; earlier failed visits are retained in raw evidence"
        rows.append(row(f"F.stage.{name}", status, reason, evidence))
    for name, lower, upper, mandatory in [("research_director", 3, 7, []),
                                          ("strategy_director", 2, 5, []),
                                          ("qa_director", 2, 5, ["workflow-validate", "cli-integration"])]:
        rows.append(_director_evidence_score(name, (director_manifests or {}).get(name), evidence_root,
                                              lower, upper, mandatory))
    barrier = []
    for index, event in enumerate(events):
        if event["stage_id"] == "synthesize_qa" and event["kind"] in {"node_started", "NodeStarted"}:
            # Bound the observation by the most recent fork visit, permitting any
            # completion order among its independent branches.
            preceding = events[:index]
            start = max((i for i, e in enumerate(preceding) if e["stage_id"] == "fork_qa" and
                         e["kind"] in {"node_started", "NodeStarted"}), default=-1)
            joined = {e["stage_id"] for e in preceding[start + 1:] if e["kind"] in {"node_completed", "NodeCompleted"}}
            barrier.append({"health_checker", "code_reviewer", "qa_director", "join_qa"} <= joined)
    rows.append(row("F.qa_barrier", "passed" if barrier and all(barrier) else "failed" if barrier else "not_exercised",
                    "All three QA branches and join must complete before synthesis"))
    rows.extend(_protocol_rows(evidence_root, events, director_manifests or {}, operator_evidence))
    rows.append(route_score(workers or [], model=expected_model) if lane.startswith("dsh") else
                row("F.route", "unavailable", "Remote route receipt must be observed; role aliases are not served models"))
    status = "passed" if outcome.get("exit_ok") is True or outcome.get("success") is True else (
        "failed" if outcome else "unavailable")
    rows.append(row("F.completion", status, "Native authoring outcome, separate from delivered artifacts"))
    return rows


def capability_score(checker: dict | None, native: dict | None, *, evidence_root: Path | None = None,
                     operator_evidence: OperatorEvidenceContext | None = None) -> list[dict]:
    checker, native = checker or {}, native or {}
    cli_status = "unavailable"
    if checker.get("total") == 56 and len(checker.get("cases", [])) == 56:
        cli_status = "passed" if checker.get("passed") == 56 and all(c.get("passed") is True for c in checker["cases"]) else "failed"
    elif checker.get("status") == "passed":
        cli_status = "failed"
    rows = [row(f"B{i:02d}", cli_status, "Independent 56-case CLI checker" if cli_status != "unavailable" else "CLI evidence missing or incomplete") for i in range(1, 7)]
    invocations = native.get("invocations", [])
    valid = {r.get("input"): r for r in invocations if r.get("input") in {"overlap", "touching"}}
    invalid = next((r for r in invocations if r.get("input") == "invalid-duplicate"), {})
    native_error = invalid.get("exit_ok") is False and invalid.get("output_exists") is False
    rows.append(row("B07", "passed" if cli_status == "passed" and native_error else "failed" if invalid else "unavailable", "CLI preservation and native invalid-input failure"))
    boundary = native.get("native_boundary", {}).get("status") == "passed"
    stable = (native.get("package_unchanged") is True and native.get("package_before_import")
              and native["package_before_import"] == native.get("package_after_inspection")
              == native.get("package_after_native"))
    rows.append(row("B08", "passed" if native.get("discovered") and native.get("validated") and native.get("inspected") and boundary and stable else "failed" if native and boundary else "unavailable", "Native discovery/inspection/validation within verified model-free boundary; imports may not mutate the package"))
    expected = {"overlap": {"conflicts": [["a", "b"]]}, "touching": {"conflicts": []}}
    good = boundary and stable and len(valid) == 2 and all(r.get("exit_ok") and r.get("output") == expected[k] and
                                  r.get("package_before") == r.get("package_after") and
                                  r.get("package_before") == native.get("package_before_import") and
                                  r.get("package_before") for k, r in valid.items())
    if good:
        good = valid["overlap"].get("run_directory") != valid["touching"].get("run_directory")
    rows.append(row("B09", "passed" if good else "failed" if invocations else "unavailable", "Two distinct native invocations of an unchanged package"))
    # README evidence requires authored instructions actually exercised by native adapter.
    rows.append(supplemental_row("B10", "readme", evidence_root, operator_evidence, native=native))
    return rows


def aggregate(rows: list[dict]) -> str:
    statuses = {r["status"] for r in rows}
    if not rows or "unavailable" in statuses:
        return "failed" if "failed" in statuses else "unavailable"
    if "failed" in statuses:
        return "failed"
    return "not_exercised" if "not_exercised" in statuses else "passed"


def write_report(path: Path, lanes: list[dict]) -> dict:
    path.mkdir(parents=True, exist_ok=True)
    regression_links = json.loads((FIXTURES / "regression-links.json").read_text())
    report = {"schema_version": 1, "benchmark_version": 1, "lanes": lanes,
              "regression_links": regression_links,
              "status": aggregate([r for lane in lanes for key in ("capability", "fidelity") for r in lane[key]]),
              "interpretation": "Diagnostic pair only; no causal speed/quality ranking. Missing evidence is never a pass."}
    (path / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    names = [lane["lane"] for lane in lanes]
    requirements = list(dict.fromkeys(r["requirement_id"] for lane in lanes for key in ("capability", "fidelity") for r in lane[key]))
    md = ["# Paired Create evidence", "", f"Overall: **{report['status']}**", "", "| Requirement | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for requirement in requirements:
        statuses = []
        for lane in lanes:
            r = next((r for key in ("capability", "fidelity") for r in lane[key] if r["requirement_id"] == requirement), row(requirement, "unavailable", "No evidence"))
            statuses.append(r["status"])
        md.append("| " + requirement + " | " + " | ".join(statuses) + " |")
    (path / "report.md").write_text("\n".join(md) + "\n")
    from .reporting import annotated_html
    (path / "comparison.html").write_text(annotated_html(path, report, reference()))
    return report
