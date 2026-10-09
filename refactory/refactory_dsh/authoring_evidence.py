"""Current runtime receipts and requirement traceability for authoring.

The runtime checks identity, coverage and evidence references. Workers still own
semantic review: a well-formed evidence map is not proof of correctness.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any
import uuid

EVIDENCE = "authoring-evidence.json"
AUDIT = "intent-audit.json"
REVIEWERS = ("health_checker", "code_reviewer", "qa_director", "overwatch", "focused_qa")
INPUTS = ("brief.json", "strategy.json", "design.md", "implementation-plan.txt")
SETTINGS = ("sample_knobs", "allow_regression", "check", "task_ref", "single_shot", "requires_input")

REVIEW_CONTEXT = (
    "\n## Authoritative validation and intent baseline\n{{runtime_evidence}}\n"
    "Prior validation/review diagnostics (resolve any evidence-format errors in your own report): "
    "{{repair_feedback}}\n{{qa_gate}}\n{{overwatch_review}}\n"
    "The runtime has already executed canonical smoke/acceptance on this revision. "
    "Read its receipt and actual output files; do not repeat a full model/Task feed. "
    "Use targeted spot checks only for a concrete uncovered requirement. A numeric "
    "score, source inspection and requirement coverage are different evidence. "
    "The brief, accepted criteria/design and current user intent ledger are binding. "
    "Only explicit user changes or exclusions can alter them; an author, coordinator, "
    "reviewer or README cannot waive an ask. Missing required evidence is failed or "
    "unverified, never pending-but-passed. Do not modify the implementation. "
    "If a required tool/credential is unavailable, write `UNAVAILABLE:` with the "
    "dependency and exact unblock action; do not request recoding or unchanged retries.\n"
)

AUDIT_INSTRUCTIONS = (
    "\nWrite intent-audit.json as {\"requirements\":[{\"id\":\"baseline id\", "
    "\"requirement\":\"the concrete ask(s) checked\", \"status\":\"pass|fail|unverified|excluded\", "
    "\"evidence\":[{\"path\":\"actual file path\", \"detail\":\"command/output or precise finding\"}], "
    "\"reason\":\"explanation\", \"user_change\":\"ledger id or brief, only for an explicit user exclusion\", "
    "\"user_quote\":\"exact text of that explicit exclusion\"}]}. "
    "Cover EVERY baseline id in authoring-evidence.json, decomposing compound asks "
    "into separate rows with the same id when needed. Map every ask, accepted criterion, "
    "design obligation and current ledger instruction to actual current evidence. "
    "Evidence paths are relative to the run directory or absolute delivered file paths. "
    "Cite existing individual files, never directories, line-number suffixes or intent-audit.json itself. "
    "Write every probe fixture you create into the run directory, under a path "
    "namespaced to your own node (for example `scratch/<node id>/<name>`), never /tmp "
    "or any other outside location and never a peer's shared name: concurrent "
    "reviewers stage probes beside each other, so a flat shared name can be clobbered "
    "by a peer. The audit accepts only files inside the run "
    "directory or the delivered artifact, so a fixture kept elsewhere cannot be cited and "
    "its verification is lost. "
    "A pass needs a supporting file and specific evidence; an exclusion needs the "
    "actual explicit user change id, exact user_quote and reason. Report omissions as fail/unverified "
    "with precise reasons, even if other reviewers passed.\n"
)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def source_files(root: Path) -> dict[str, str]:
    ignored = {".git", ".factory", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".origin.json"}
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in ignored or part.startswith(".env") for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"cannot attest a symlink in the delivered source: {relative}")
        if path.is_file():
            files[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def intent(workdir: Path) -> list[dict[str, Any]]:
    from refactory_dsh.steering import Ledger

    # Gate visits/consumption are bookkeeping, not later user instructions.
    return [entry for entry in Ledger(workdir).entries()
            if entry["kind"] in ("Initial Idea", "info", "correction", "constraint", "verdict")]


def identity(workdir: Path, root: str) -> dict[str, Any]:
    return {"root": root, "files": source_files(Path(root)), "intent": intent(workdir),
            "inputs": {name: hashlib.sha256((workdir / name).read_bytes()).hexdigest()
                       if (workdir / name).is_file() else None for name in INPUTS}}


def requirements(ctx: Any) -> list[dict[str, str]]:
    result = [{"id": "brief", "text": str(ctx.facts.get("brief") or ctx.read_text("brief.json"))}]
    for name, key in (("strategy.json", "strategy"), ("design.md", "design"),
                      ("implementation-plan.txt", "plan")):
        if not ctx.exists(name):
            continue
        raw = ctx.read_text(name)
        result.append({"id": key, "text": raw})
        if key == "strategy":
            try:
                strategy = json.loads(raw)
            except ValueError:
                strategy = {}
            if isinstance(strategy, dict):
                criteria = strategy.get("acceptance_criteria", [])
                if isinstance(criteria, list):
                    result.extend({"id": f"criterion-{index + 1}", "text": str(item)}
                                  for index, item in enumerate(criteria))
    for index, entry in enumerate(intent(ctx.workdir)):
        if entry["kind"] in ("Initial Idea", "info", "correction", "constraint"):
            result.append({"id": f"ledger-{index + 1}", "text": entry["text"], "kind": entry["kind"]})
    return result


def record(ctx: Any, outcome: dict[str, Any]) -> dict[str, Any]:
    root = str(ctx.facts.get("artifact") or ctx.facts.get("package_path") or "")
    if not root or not Path(root).is_dir():
        raise ValueError("runtime evidence needs the delivered package/artifact directory")
    payload = {"schema": 1, "id": uuid.uuid4().hex,
               "identity": identity(ctx.workdir, root), "requirements": requirements(ctx),
               "settings": {key: ctx.knob(key, None) for key in SETTINGS},
               "receipts": {name: hashlib.sha256((ctx.workdir / name).read_bytes()).hexdigest()
                            for name in ("smoke.json", "revision.json", "focused-check.json", "check.txt")
                            if (ctx.workdir / name).is_file()},
               "outputs": runtime_outputs(ctx), "validation": outcome}
    ctx.write_json(EVIDENCE, payload)
    # Immutable historical receipts survive repairs and explicit runtime redo.
    history = ctx.workdir / ".refactory" / "validation-history" / payload["id"]
    history.mkdir(parents=True, exist_ok=True)
    (history / EVIDENCE).write_text(json.dumps(payload, indent=2) + "\n")
    for name in ("smoke.json", "revision.json", "check.txt", "focused-check.json", AUDIT):
        path = ctx.workdir / name
        if path.is_file():
            (history / name).write_bytes(path.read_bytes())
    ctx.set_fact("runtime_evidence", json.dumps(payload))
    return payload


def runtime_outputs(ctx: Any) -> dict[str, Any]:
    """Bind the canonical execution's actual files as well as its JSON receipt."""
    smoke = ctx.read_json("smoke.json", {}) if ctx.exists("smoke.json") else {}
    root = Path(smoke["workdir"]) if smoke.get("workdir") else None
    if root is None or not root.is_dir() or not root.resolve().is_relative_to(ctx.workdir.resolve()):
        return {}
    return {"root": str(root), "files": source_files(root)}


def delivery_identity(workdir: Path) -> dict[str, str] | None:
    path = workdir / EVIDENCE
    if not path.exists():
        return None
    evidence = json.loads(path.read_text())
    return {"receipt": digest(evidence),
            "identity": digest(identity(workdir, evidence["identity"]["root"]))}


def current_problems(ctx: Any, workers: tuple[str, ...] = ()) -> list[str]:
    """Legacy reports remain readable; new evidence cannot authorize old sign-off."""
    if not ctx.exists(EVIDENCE):
        return ["authoritative runtime evidence is missing"] if ctx.facts.get("runtime_evidence") else []
    evidence = ctx.read_json(EVIDENCE, {})
    now = delivery_identity(ctx.workdir)
    problems = []
    if now is None or now["identity"] != digest(evidence.get("identity")):
        problems.append("artifact, accepted inputs or user intent changed after validation; rerun validation and reviews")
    if evidence.get("validation", {}).get("status") != "pass":
        problems.append("required runtime validation did not pass")
    if evidence.get("settings") != {key: ctx.knob(key, None) for key in SETTINGS}:
        problems.append("validation inputs or comparison policy changed after execution")
    for name, expected in evidence.get("receipts", {}).items():
        path = ctx.workdir / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            problems.append(f"authoritative receipt {name} changed after validation")
    if "outputs" in evidence and evidence["outputs"] != runtime_outputs(ctx):
        problems.append("canonical runtime output files changed after validation")
    if workers:
        from refactory_dsh.agent_delivery import receipts_in

        receipts = receipts_in(ctx.workdir)
        for worker in workers:
            receipt = receipts.get(worker, {})
            if (receipt.get("status") != "delivered" or not receipt.get("current")
                    or receipt.get("execution", {}).get("authoring") != now):
                problems.append(f"{worker} has no current delivery/sign-off for this artifact, intent and runtime receipt")
    return problems


def audit_problems(ctx: Any) -> list[str]:
    if not ctx.exists(EVIDENCE):
        return []
    evidence = ctx.read_json(EVIDENCE, {})
    try:
        audit = ctx.read_json(AUDIT, {})
    except ValueError:
        audit = {}
    rows = audit.get("requirements", []) if isinstance(audit, dict) else []
    baseline = {item["id"]: item for item in evidence.get("requirements", [])}
    changes = {key for key, value in baseline.items() if value.get("kind") in ("Initial Idea", "correction", "constraint")}
    changes.add("brief")
    covered = set()
    problems = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] not in baseline:
            problems.append("intent audit contains an unknown requirement id")
            continue
        key = row["id"]
        covered.add(key)
        if row.get("status") == "excluded":
            change = row.get("user_change")
            quote = row.get("user_quote")
            if (not isinstance(change, str) or change not in changes or not row.get("reason") or not isinstance(quote, str)
                    or not quote.strip() or quote not in baseline.get(change, {}).get("text", "")):
                problems.append(f"{key}: exclusion has no explicit user change; reviewers cannot waive requirements")
            continue
        refs = row.get("evidence")
        if row.get("status") != "pass" or not row.get("requirement") or not isinstance(refs, list) or not refs:
            problems.append(f"{key}: required evidence is failed or unverified: {row.get('reason', 'no supporting evidence')}")
            continue
        for ref in refs:
            path = Path(str(ref.get("path", ""))) if isinstance(ref, dict) else Path()
            path = (ctx.workdir / path).resolve()
            roots = (ctx.workdir.resolve(), Path(evidence["identity"]["root"]).resolve())
            if (not isinstance(ref, dict) or not ref.get("detail") or not path.is_file()
                    or not any(path.is_relative_to(root) for root in roots)
                    or path.name == AUDIT):
                problems.append(f"{key}: missing or invalid evidence reference {str(path)!r}; "
                                "cite an existing individual file within the run or delivered artifact, with detail")
    problems.extend(f"{key}: no requirement evidence in intent audit" for key in sorted(set(baseline) - covered))
    return problems


def failure_kind(detail: str, *, timed_out: bool = False, returncode: int | None = None) -> str:
    lowered = detail.lower()
    if any(word in lowered for word in ("budget exhausted", "resource exhausted", "reserve retained", "budgetexhausted")):
        return "resource_exhausted"
    if "cancelled" in lowered or "canceled" in lowered:
        return "cancelled"
    if timed_out or "timed out" in lowered or "timeouterror" in lowered or returncode is not None and returncode < 0:
        return "uncertain"
    if re.search(r"ModuleNotFoundError: No module named ['\"]refactory_packages\.", detail):
        return "code_failure"
    if (returncode in (126, 127) or re.search(r"UNAVAILABLE:", detail, re.I)
            or any(word in lowered for word in ("credentials", "api key", "unauthorized", "no model",
                                                "modulenotfounderror", "command not found", "model completion failed"))):
        return "unavailable"
    return "code_failure"


def stop_unrepairable(kind: str, detail: str) -> None:
    if kind != "code_failure":
        raise ValueError(f"validation {kind}: {detail}. Preserve the artifact; "
                         "restore the named dependency/credentials or inspect worker/resource state before explicit recovery; code repair was not attempted")


def wire_reviews(graph: Any) -> None:
    """The full graphs retain distinct health, code, adversarial and audit roles."""
    from dataclasses import replace

    # A repair must re-enter work admission rather than spend validation reserve.
    graph.nodes["emit"] = replace(graph.nodes["emit"], resource_role="work")
    for name in ("fork_qa", "join_qa", "synthesize_qa", "gate_qa", "gate_overwatch", "report"):
        graph.nodes[name] = replace(graph.nodes[name], resource_role="final_validation")
    for name in REVIEWERS[:-1]:
        node = graph.nodes[name]
        audit = name == "overwatch"
        graph.nodes[name] = replace(
            node, prompt_template=node.prompt_template + REVIEW_CONTEXT + (AUDIT_INSTRUCTIONS if audit else ""),
            reads=(*node.reads, EVIDENCE), writes=(*node.writes, AUDIT) if audit else node.writes,
            resource_role="final_validation",
        )
    for name in ("gate_qa", "gate_overwatch"):
        node = graph.nodes[name]
        graph.nodes[name] = replace(node, reads=(*node.reads, EVIDENCE, *((AUDIT,) if name == "gate_overwatch" else ())), max_iterations=12,
                                   writes=(*node.writes, "validation.txt"),
                                   produces=(*node.produces, "validation", "attempts_used"))
    report = graph.nodes["report"]
    graph.nodes["report"] = replace(report, reads=(*report.reads, EVIDENCE, AUDIT))


def require_completion(ctx: Any) -> None:
    workers = ("focused_qa",) if "repair_attempts" in ctx.knobs else REVIEWERS[:-1]
    problems = current_problems(ctx, workers) + audit_problems(ctx)
    if problems:
        raise ValueError("authoring sign-off is incomplete or stale: " + "; ".join(problems))
