"""Explicit human answer delivery, bound to the exact pending strategy bytes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time


DEFAULT_INPUTS = (".factory/strategy/current.md", ".factory/strategy/user-intent.md")


def strategy_digest(project: Path, inputs=DEFAULT_INPUTS) -> str:
    paths = [project / name for name in inputs]
    if not paths or any(not p.resolve().is_relative_to(project.resolve()) for p in paths):
        raise ValueError("Approval inputs must be files inside the authoring project")
    return hashlib.sha256(json.dumps({str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in paths}, sort_keys=True).encode()).hexdigest()


def request(project: Path, prompt: str, *, inputs=DEFAULT_INPUTS) -> dict:
    record = {"gate": "gate_strategy", "strategy_digest": strategy_digest(project, inputs),
              "input_paths": list(inputs),
              "prompt": prompt, "status": "awaiting-human", "created_unix": time.time()}
    (project / "approval-required.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def deliver(project: Path, *, digest: str, verdict: str, approver: str, feedback: str) -> dict:
    pending = json.loads((project / "approval-required.json").read_text())
    if not approver.strip() or verdict not in {"proceed", "reloop", "halt"}:
        raise ValueError("An identified human and explicit verdict are required")
    if verdict == "reloop" and not feedback.strip():
        raise ValueError("A strategy revision requires the human's feedback")
    inputs = pending.get("input_paths", DEFAULT_INPUTS)
    if pending["strategy_digest"] != digest or strategy_digest(project, inputs) != digest:
        raise ValueError("Strategy/intent changed; review the new revision before answering")
    if pending.get("status") != "awaiting-human":
        raise ValueError("This approval request was already consumed; wait for a new gate visit")
    record = {"gate": "gate_strategy", "strategy_digest": digest, "verdict": verdict,
              "input_paths": list(inputs),
              "approver": approver, "feedback": feedback, "answered_unix": time.time(),
              "request_created_unix": pending["created_unix"], "source": "explicit-human"}
    path = project / "human-answer.json"
    if path.exists():
        raise ValueError("Pending answer already exists; it must be consumed before another answer")
    path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def take(project: Path, pending: dict) -> dict | None:
    path = project / "human-answer.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    archive = project / "raw/approvals"
    archive.mkdir(parents=True, exist_ok=True)
    target = archive / f"{time.time_ns()}.json"
    path.replace(target)
    inputs = pending.get("input_paths", DEFAULT_INPUTS)
    if (tuple(record.get("input_paths", DEFAULT_INPUTS)) != tuple(inputs)
            or record.get("strategy_digest") != strategy_digest(project, inputs)
            or record.get("strategy_digest") != pending["strategy_digest"]
            or record.get("request_created_unix") != pending["created_unix"]):
        raise ValueError("Human answer is stale or belongs to another gate visit")
    if record.get("source") != "explicit-human" or not record.get("approver"):
        raise ValueError("Unattributed gate answer cannot authorize a strategy")
    (project / "approval-required.json").write_text(json.dumps({
        **pending, "status": "consumed", "answer_path": str(target.relative_to(project)),
    }, indent=2) + "\n")
    return record
