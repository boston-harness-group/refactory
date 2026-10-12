"""DSH-owned dynamic director stages for the Create fidelity candidate.

The CEO proposes work and judges its quality. Python validates that proposal,
routes real child agents, joins their receipts, and publishes their complete
reports. No model call or child report is replaced by a parent summary.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from refactory_dsh.nn.executor import ExecutionError, Executor, RunContext, render
from refactory_dsh.nn.graph import Graph, Node


@dataclass(frozen=True)
class Stage:
    id: str
    role: str
    minimum: int
    maximum: int
    plan: str
    reports: str
    reads: tuple[str, ...]
    minimum_bytes: int


RESEARCH = Stage("research_director", "researcher", 3, 7,
                 ".factory/strategy/research-plan.json", ".factory/strategy/research",
                 (".factory/strategy/study-combined.md", ".factory/strategy/user-intent.md"), 50)
STRATEGY = Stage("strategy_director", "strategist", 2, 5,
                 ".factory/strategy/strategy-plan.json", ".factory/strategy/perspectives",
                 (".factory/research_director-findings.json", ".factory/research_director-findings.md",
                  ".factory/strategy/research-plan.json", ".factory/strategy/study-combined.md",
                  ".factory/strategy/user-intent.md"), 100)
QA = Stage("qa_director", "adversarial_tester", 2, 5,
           ".factory/reviews/qa-plan.json", ".factory/reviews/testers",
           (".factory/reviews/builder-latest.md", ".factory/strategy/current.md",
            ".factory/strategy/user-intent.md", ".factory/reviews/repair-feedback.md"), 50)
MANDATORY = ("workflow-validate", "cli-integration")
SLUG = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
# Coverage IDs are references to user requirements, not filenames or routing
# identifiers. Preserve source IDs such as B01 and REQ:B02.integer exactly.
ITEM_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


def _digest(value: Any) -> str:
    data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExecutionError(message)


def _json(text: str, label: str) -> Any:
    try:
        return json.loads(text)
    except (ValueError, TypeError) as error:
        raise ExecutionError(f"{label} must be strict JSON, without fences or prose") from error


def _items(value: Any, label: str) -> list[dict[str, Any]]:
    _require(isinstance(value, list), f"{label} must be an array")
    items: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        # Original intent_items strings retain their text; object metadata is
        # copied intact rather than silently erased by typed normalization.
        item = {"id": f"intent-{index + 1}", "text": raw} if isinstance(raw, str) else dict(raw) if isinstance(raw, dict) else {}
        _require(isinstance(item.get("id"), str) and bool(ITEM_ID.fullmatch(item["id"])),
                 f"{label} item ID must be 1..128 ASCII letters/digits with optional . _ : - separators")
        _require(isinstance(item.get("text"), str) and bool(item["text"].strip()), f"{label} has empty item text")
        items.append(item)
    _require(len({item["id"] for item in items}) == len(items), f"{label} has duplicate IDs")
    return items


def normalize_plan(raw: Any, stage: Stage, required: list[dict[str, Any]]) -> dict[str, Any]:
    """Admit typed tasks before routing anything; K excludes QA's fixed testers."""
    _require(isinstance(raw, dict), "director plan must be an object with tasks and coverage_items")
    _require(set(raw) <= {"schema_version", "stage_id", "tasks", "intent_items", "coverage_items"}, "unknown director plan fields")
    _require(type(raw.get("schema_version", 1)) is int and raw.get("schema_version", 1) == 1
             and raw.get("stage_id", stage.id) == stage.id, "director plan identity mismatch")
    tasks = raw.get("tasks")
    _require(isinstance(tasks, list) and stage.minimum <= len(tasks) <= stage.maximum,
             f"{stage.id} needs {stage.minimum}..{stage.maximum} custom children")
    intent = _items(raw.get("intent_items", []), "intent_items")
    coverage = _items(raw.get("coverage_items", []), "coverage_items")
    _require(bool(coverage), "coverage_items must identify the required work")
    by_id = {item["id"]: item for item in coverage}
    for item in required:
        _require(item["id"] in by_id and by_id[item["id"]]["text"] == item["text"], f"plan dropped or changed required coverage {item['id']}")
    if stage is STRATEGY:
        _require(bool(intent), "strategy plan must extract every user ask into intent_items")
    for item in intent:
        _require(item["id"] in by_id and by_id[item["id"]]["text"] == item["text"], f"intent item {item['id']} missing from coverage_items")
    normalized = []
    allowed = {"id", "slug", "role", "focus", "direction", "perspective", "approach", "prompt", "output", "coverage"}
    for raw_task in tasks:
        _require(isinstance(raw_task, dict) and set(raw_task) <= allowed, "task must have only typed task fields")
        slug = raw_task.get("slug")
        _require(isinstance(slug, str) and bool(SLUG.fullmatch(slug)) and slug not in MANDATORY, "invalid, unsafe, or reserved task slug")
        identifier = raw_task.get("id", slug)
        _require(isinstance(identifier, str) and bool(SLUG.fullmatch(identifier)) and identifier not in MANDATORY, "invalid or reserved child ID")
        role = raw_task.get("role", stage.role)
        _require(role == stage.role, f"{stage.id} children must use role {stage.role}")
        focus = raw_task.get("focus", raw_task.get("direction", raw_task.get("perspective", raw_task.get("approach"))))
        prompt = raw_task.get("prompt")
        _require(isinstance(focus, str) and bool(focus.strip()) and isinstance(prompt, str) and bool(prompt.strip()), "task focus and tailored prompt must be nonempty text")
        output = f"{stage.reports}/{slug}.md"
        _require(raw_task.get("output", output) == output, "child output must match its assigned report path")
        covers = raw_task.get("coverage")
        _require(isinstance(covers, list) and bool(covers) and all(isinstance(c, str) and c in by_id for c in covers)
                 and len(covers) == len(set(covers)), "task coverage must name distinct declared coverage IDs")
        normalized.append({"id": identifier, "slug": slug, "role": role, "focus": focus,
                           "prompt": prompt, "output": output, "coverage": covers, "mandatory": False})
    for key in ("id", "slug", "output"):
        _require(len({task[key] for task in normalized}) == len(normalized), f"duplicate child {key}")
    covered = {item for task in normalized for item in task["coverage"]}
    _require(set(by_id) <= covered, f"plan has uncovered items: {sorted(set(by_id) - covered)}")
    return {"schema_version": 1, "stage_id": stage.id, "tasks": normalized,
            "intent_items": intent, "coverage_items": coverage}


def _load_prompt(role: str, node: str = "") -> str:
    from refactory_dsh.create_fidelity_prompts import load_prompt
    return load_prompt(role, node)


def _route(client: Any, node: str, role: str) -> Any:
    _require(client is not None and callable(getattr(client, "for_node", None)) and callable(getattr(client, "route", None)),
             "composite directors require a routed DSH model client")
    selected = client.for_node(node, role=role)
    _require(selected.route().get("worker_backend") == "dsh", f"{node}: {role} must execute in DSH")
    return selected


class _BoundRoute:
    """Keep a resolved logical role/node route while assigning a dispatch ID."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def for_node(self, node: str, role: str = "worker") -> Any:
        _require(callable(getattr(self.client, "with_identity", None)), "DSH client must preserve a resolved route when binding dispatch identity")
        return self.client.with_identity(node, role)


def _receipts(workdir: Path, node: str) -> tuple[list[str], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    worker_paths, workers, delivery_paths, deliveries = [], [], [], []
    for path in sorted((workdir / ".refactory/workers").glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            # A concurrent peer may currently be creating/updating its receipt.
            # This agent's complete() has already settled its own receipt; if
            # that receipt is corrupt, absence below still fails this dispatch.
            continue
        if not isinstance(record, dict):
            continue
        if record.get("node_id") == node:
            worker_paths.append(str(path.relative_to(workdir)))
            workers.append(record)
    for path in sorted((workdir / ".refactory/agent-delivery").glob("*/receipt.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        if record.get("node") == node:
            delivery_paths.append(str(path.relative_to(workdir)))
            deliveries.append(record)
    return worker_paths, workers, delivery_paths, deliveries


def _run_agent(ctx: RunContext, node: str, role: str, prompt: str, output: str,
               reads: tuple[str, ...], identity: dict[str, Any], timeout_s: int = 1800) -> dict[str, Any]:
    logical_id = identity["stage_id"] if role == "ceo" else identity["task_id"]
    selected = _route(ctx.model, logical_id, role)
    deadline = getattr(ctx, "_director_deadline", None)
    if deadline is not None:
        remaining = deadline - time.monotonic()
        _require(remaining > 0, f"{identity['stage_id']} composite stage deadline expired")
        timeout_s = min(timeout_s, max(1, math.ceil(remaining)))
        if callable(getattr(selected, "with_deadline_observation", None)):
            selected = selected.with_deadline_observation("composite_stage", deadline)
    if callable(getattr(selected, "with_node_timeout", None)):
        selected = selected.with_node_timeout(timeout_s)
    requested = selected.route()
    graph = Graph()
    graph.add(Node(id=node, kind="agent", role=role, prompt_template=prompt,
                   reads=reads, writes=(output,), notes=f"Create fidelity {role} task",
                   **({"timeout_s": timeout_s} if "timeout_s" in Node.__dataclass_fields__ else {})))
    # Role/task templates are rendered before raw input bodies are appended.
    # Empty interpolation values keep embedded report code such as {{name}}
    # byte-for-byte intact through the Executor's normal render boundary.
    child_ctx = RunContext(ctx.workdir, task=ctx.task,
                           model=_BoundRoute(selected), enforce_writes=ctx.enforce_writes)
    child_ctx.budget = ctx.budget
    executor = Executor(graph, child_ctx, max_steps=1)
    # The enclosing Executor owns progress.json. The branch walker still crosses
    # real dispatch and agent-delivery boundaries without racing that marker.
    executor._reserve_step = lambda: True
    error = executor._walk_branch(node, None)
    worker_paths, workers, delivery_paths, deliveries = _receipts(ctx.workdir, node)
    actual = workers[-1].get("actual_route") if workers else None
    supplied_input_digest = identity.get("input_digest")
    if len(deliveries) == 1 and deliveries[0].get("contexts"):
        context_path = deliveries[0]["contexts"][-1].get("prompt")
        if context_path and ctx.exists(context_path):
            # Hash the actual rendered, ledger-bearing delivery, including the
            # execution contract, rather than its pre-render task template.
            supplied_input_digest = _digest(ctx.path(context_path).read_bytes())
    if not error:
        if len(workers) != 1 or workers[0].get("state") != "completed":
            error = "missing, duplicate, or failed worker receipt"
        elif not isinstance(actual, dict) or actual.get("worker_backend") != "dsh":
            error = "worker receipt does not establish actual DSH execution"
        elif any(requested.get(field) and actual.get(field) != requested[field]
                 for field in ("provider", "model", "effort")):
            error = "actual worker provider/model/effort differs from the requested route"
        elif workers[0].get("native_observations"):
            error = "DSH non-builder worker delegated to an external worker backend"
        elif workers[0].get("requested_route") != requested:
            error = "worker requested route differs from the resolved route"
        elif len(deliveries) != 1 or deliveries[0].get("status") != "delivered":
            error = "missing, duplicate, or stale delivery receipt"
    body = ctx.read_text(output) if ctx.exists(output) and ctx.path(output).is_file() else ""
    return {**identity, "input_digest": supplied_input_digest, "node_id": node, "role": role, "requested_route": requested,
            "actual_route": actual, "status": "failed" if error else "completed", "error": error,
            "output": output, "output_digest": _digest(body.encode()), "body": body,
            "raw_evidence": {"worker_receipts": worker_paths, "delivery_receipts": delivery_paths},
            "effective_deadlines": selected.effective_deadlines() if callable(getattr(selected, "effective_deadlines", None)) else {"stage_timeout_s": timeout_s},
            "steps": [{"node": step.node, "kind": step.kind, "error": step.error} for step in executor.steps]}


def _input_bundle(ctx: RunContext, stage: Stage) -> tuple[dict[str, str], list[dict[str, Any]]]:
    bodies = {}
    for name in stage.reads:
        _require(ctx.exists(name) and ctx.path(name).is_file(), f"{stage.id} missing input {name}")
        bodies[name] = ctx.read_text(name)
    if stage is STRATEGY:
        upstream = _json(bodies[".factory/research_director-findings.json"], "research findings")
        _require(upstream.get("status") == "completed" and bool(upstream.get("reports")), "strategy requires completed research findings")
        for report in upstream["reports"]:
            _require(isinstance(report.get("body"), str) and bool(report["body"].strip()), "research findings omit substantive report content")
    if stage is RESEARCH:
        required = [{"id": "workflow-patterns", "text": "Existing native workflow patterns and graph construction"},
                    {"id": "node-types", "text": "Node types, data flow, gate logic and recovery"},
                    {"id": "mode-requirements", "text": "Mode purpose and user requirements"}]
    elif stage is STRATEGY:
        required = [{"id": "architecture", "text": "Concrete graph node IDs, types and edge wiring"},
                    {"id": "testing-verification", "text": "Explicit acceptance criteria and verification evidence"}]
    else:
        # Pin every explicit checklist item before the planner extracts any
        # additional prose/table criteria. It may add criteria but cannot omit
        # the input's machine-identifiable acceptance obligations.
        checks = re.findall(r"^\s*[-*]\s+\[[ xX]\]\s+(.+)$", bodies[".factory/strategy/current.md"], re.MULTILINE)
        required = [{"id": f"acceptance-{i + 1}", "text": text.strip()} for i, text in enumerate(checks)]
        required += [{"id": "user-intent", "text": "Verify implementation against every original user ask"}]
    return bodies, required


def _context_text(bodies: dict[str, str]) -> str:
    return "\n\n".join(f"=== COMPLETE INPUT {path} ===\n{body}" for path, body in bodies.items())


def _mandatory_tasks(ctx: RunContext) -> list[dict[str, Any]]:
    # The graph owns the native spelling of validation/discovery/invocation.
    from refactory_dsh.create_fidelity_ops import native_instructions
    native = native_instructions(ctx)
    prompts = {
        "workflow-validate": "Run the generated mode through the actual native validation pipeline. Report exact commands, exit codes and full output. Inspect the native graph, compare node/edge counts with current.md, and call its Python graph validator directly. Zero validation issues are required.",
        "cli-integration": "Verify the generated mode through actual native discovery and CLI inspection. Report exact commands, exit codes and full output. Verify a nonempty description, visible nodes/edges and generated SKILL.md documentation. Test registration and portability from an unrelated working directory. For plugin packages, verify package entry points, editable installation and uninstall cleanup, and that no upstream source was changed. For project-local modes, verify the package is local to the requested output folder and discoverable from the documented project path.",
    }
    return [{"id": slug, "slug": slug, "role": QA.role, "focus": slug,
             "prompt": f"{prompt}\n\n{native}", "output": f"{QA.reports}/{slug}.md",
             "coverage": [], "mandatory": True} for slug, prompt in prompts.items()]


def _plan_prompt(ctx: RunContext, stage: Stage, bodies: dict[str, str], required: list[dict[str, Any]]) -> str:
    return (
        render(_load_prompt("ceo", stage.id), ctx) + "\n\nDSH COMPOSITE PHASE: PLAN ONLY.\n"
        "The runtime launches and joins all workers. Do not launch workers yourself. "
        "The original director's substantive planning/coverage duties apply. Original factory command spellings "
        "are reference vocabulary; use the supplied native API. JSON plans and quality prose are separate artifacts.\n"
        f"Write strict JSON to {stage.plan}. The object has schema_version:1, stage_id:{stage.id!r}, "
        "tasks, intent_items, coverage_items. Each task has id and slug (unique kebab-case), role, focus, "
        "a tailored prompt, output and coverage (list of coverage item IDs). Each coverage/intent item "
        "has id and text; preserve additional metadata. Task id and slug must be lower-case kebab-case. "
        "Coverage/intent item IDs are case-sensitive requirement references: use 1..128 ASCII letters/digits "
        "with optional dot, underscore, colon or dash; no spaces, slashes or control characters. "
        "Preserve existing requirement IDs EXACTLY, including case and punctuation; do not lowercase or renumber them. "
        "Task coverage must cite these exact IDs. Extract ALL original user asks as intent_items, "
        "include them unchanged in coverage_items, and assign each to at least one child. "
        "For QA extract ALL acceptance criteria, including prose and tables, as coverage_items. "
        f"Choose {stage.minimum}..{stage.maximum} custom tasks with role {stage.role!r}; outputs must be "
        f"{stage.reports}/<slug>.md. QA mandatory workflow-validate and cli-integration are added by DSH "
        "outside this custom count; do not include them. For strategy include explicit testing/verification "
        "and architecture coverage and preserve each ask in relevant strategist prompts.\n"
        "Required coverage items (may add others, must not change or drop these):\n"
        + json.dumps(required, ensure_ascii=False) + "\n\n" + _context_text(bodies)
    )


def _child_prompt(ctx: RunContext, stage: Stage, task: dict[str, Any], plan: dict[str, Any], bodies: dict[str, str],
                  identity: dict[str, Any], retry: str = "") -> str:
    assigned = [item for item in plan["coverage_items"] if item["id"] in task["coverage"]]
    asks = [item for item in plan["intent_items"] if item["id"] in task["coverage"]]
    return (
        render(_load_prompt(task["role"]), ctx) + "\n\nDSH CHILD ASSIGNMENT\n" + render(task["prompt"], ctx)
        + f"\nWrite your complete substantive report to {task['output']}. Include concrete findings, "
        "evidence, limitations and assigned coverage IDs. Perform your role's actual investigation/tests; "
        "a plan or completion summary is insufficient. Work only on this assignment.\n"
        + "Identity: " + json.dumps(identity, sort_keys=True) + "\n"
        + "Assigned coverage: " + json.dumps(assigned, ensure_ascii=False) + "\n"
        + "IMPORTANT: The user specifically asked for: " + json.dumps(asks, ensure_ascii=False)
        + ". Your work MUST address each assigned ask. Do not substitute your own ideas for what the user asked for.\n"
        + (f"DIRECTOR RETRY INSTRUCTIONS FOR THIS SAME CHILD:\n{render(retry, ctx)}\n" if retry else "")
        + _context_text(bodies)
    )


def _review_prompt(ctx: RunContext, stage: Stage, plan: dict[str, Any], reports: list[dict[str, Any]], bodies: dict[str, str]) -> str:
    return (
        render(_load_prompt("ceo", stage.id), ctx) + "\n\nDSH COMPOSITE PHASE: QUALITY REVIEW ONLY.\n"
        "Assess the complete reports below as the original director. Check substantive work, "
        "orthogonality, original role duties, required coverage, architecture detail, explicit acceptance "
        "criteria and every original user ask. Compare reports to the inputs, not just plan metadata. "
        "Failed/thin/missing/stale/duplicate outputs must receive retry, never accept. "
        "DSH will retry the SAME child with your precise retry_prompt, within its fixed bound. "
        "Do not write or invent missing child findings. QA mandatory testers must establish successful "
        "actual native validation/CLI checks before receiving outcome passed.\n"
        "Return strict JSON with decisions (exactly one per child), summary (nonempty quality prose), "
        "coverage_complete (boolean). Each decision: id, verdict (accept or retry), reason (nonempty), "
        "retry_prompt (required nonempty for retry), outcome (passed, findings, or failed). "
        "If any intent/acceptance item is missing, set coverage_complete false and retry the responsible child.\n"
        + "NORMALIZED PLAN:\n" + json.dumps(plan, ensure_ascii=False)
        + "\nCOMPLETE JOINED CHILD REPORTS AND RECEIPTS:\n" + json.dumps(reports, ensure_ascii=False)
        + "\n\n" + _context_text(bodies)
    )


def _validate_review(raw: Any, tasks: list[dict[str, Any]], reports: list[dict[str, Any]]) -> dict[str, Any]:
    _require(isinstance(raw, dict) and set(raw) == {"decisions", "summary", "coverage_complete"}, "quality review has invalid typed fields")
    _require(isinstance(raw["summary"], str) and bool(raw["summary"].strip()) and isinstance(raw["coverage_complete"], bool), "quality review needs prose summary and boolean coverage_complete")
    decisions = raw["decisions"]
    _require(isinstance(decisions, list) and len(decisions) == len(tasks), "quality review must assess every child exactly once")
    by_id = {report["task_id"]: report for report in reports}
    seen = set()
    for decision in decisions:
        _require(isinstance(decision, dict) and set(decision) <= {"id", "verdict", "reason", "retry_prompt", "outcome"}, "invalid child quality decision fields")
        identifier = decision.get("id")
        _require(isinstance(identifier, str) and identifier in by_id and identifier not in seen, "unknown or duplicate quality child ID")
        seen.add(identifier)
        verdict = decision.get("verdict")
        _require(verdict in ("accept", "retry") and isinstance(decision.get("reason"), str) and bool(decision["reason"].strip()), "quality verdict/reason invalid")
        _require(decision.get("outcome") in ("passed", "findings", "failed"), "quality decision requires outcome")
        if verdict == "retry":
            _require(isinstance(decision.get("retry_prompt"), str) and bool(decision["retry_prompt"].strip()), "retry requires director-authored instructions")
        else:
            _require(by_id[identifier]["status"] == "completed", "director cannot accept an invalid child report")
            if identifier in MANDATORY:
                _require(decision["outcome"] == "passed", "mandatory native QA checks did not pass")
    _require(raw["coverage_complete"] or any(d["verdict"] == "retry" for d in decisions), "incomplete coverage must order a child retry")
    return raw


def _publish(ctx: RunContext, stage: Stage, base: dict[str, Any], plan: dict[str, Any],
             reports: list[dict[str, Any]], attempts: list[dict[str, Any]], reviews: list[dict[str, Any]],
             status: str, error: str | None = None) -> None:
    payload = {**base, "status": status, "error": error, "plan_digest": _digest(plan),
               "intent_items": plan["intent_items"], "coverage_items": plan["coverage_items"],
               "reports": reports, "attempts": attempts, "quality_reviews": reviews}
    ctx.write_json(f".factory/{stage.id}-findings.json", payload)
    text = [f"# {stage.id} findings\n\nStatus: {status}\n", "## Intent items\n" + json.dumps(plan["intent_items"], indent=2, ensure_ascii=False)]
    for report in reports:
        provenance = {key: value for key, value in report.items() if key != "body"}
        text.append(f"## {report['task_id']}\n\nProvenance:\n```json\n{json.dumps(provenance, indent=2)}\n```\n\n{report['body']}")
    text.append("## Director quality assessment\n\n" + "\n\n".join(review["summary"] for review in reviews))
    ctx.write_text(f".factory/{stage.id}-findings.md", "\n\n".join(text) + "\n")
    ctx.write_text(f".factory/{stage.id}-quality.md", "\n\n".join(review["summary"] for review in reviews) + "\n")


def _direct(ctx: RunContext, stage: Stage) -> None:
    if ctx.model is None:
        # Native op lowering reconstructs its client from the resolved worker
        # context; unlike the Python walker it does not pass a model object.
        from refactory_dsh.model import DshModelClient
        ctx.model = DshModelClient(project=ctx.workdir)
    original_model = ctx.model
    allowance = 3600
    if callable(getattr(original_model, "effective_deadlines", None)):
        allowance = original_model.effective_deadlines().get("declared_stage_timeout_s") or allowance
    stopped = threading.Event()
    if callable(getattr(original_model, "with_cancellation", None)):
        ctx.model = original_model.with_cancellation(stopped)
    ctx._director_deadline = time.monotonic() + allowance
    timer = threading.Timer(allowance, stopped.set)
    timer.daemon = True
    timer.start()
    from refactory_dsh.owned_process import _termination_as_exit
    # During fanout model calls run on worker threads, so their own shutdown
    # scopes cannot install the owner's main-thread SIGTERM handler. Keep the
    # whole composite inside that scope: an external stop unwinds the join,
    # signals the existing cancellation events and awaits owned runtimes/tools.
    with _termination_as_exit():
        try:
            _direct_stage(ctx, stage)
        finally:
            timer.cancel()
            ctx.model = original_model
            del ctx._director_deadline


def _direct_stage(ctx: RunContext, stage: Stage) -> None:
    _route(ctx.model, stage.id, "ceo")
    child_attempts = ctx.knob("child_attempts", 2)
    parallelism = ctx.knob("director_parallelism", 7)
    _require(type(child_attempts) is int and 1 <= child_attempts <= 5, "child_attempts must be 1..5")
    _require(type(parallelism) is int and 1 <= parallelism <= 7, "director_parallelism must be 1..7")
    bodies, required = _input_bundle(ctx, stage)
    input_digest = _digest(bodies)
    state_path = f".factory/directors/{stage.id}/visits.json"
    visits = ctx.read_json(state_path, default={})
    visit = int(visits.get("visits", 0)) + 1
    ctx.write_json(state_path, {"visits": visit})
    directory = f".factory/directors/{stage.id}/v{visit}"
    run_id = str(ctx.task.get("trial_id") or ctx.facts.get("trial_id") or ctx.workdir.resolve())
    base = {"schema_version": 1, "run_id": run_id, "trial_id": run_id, "stage_id": stage.id,
            "visit": visit, "input_digest": input_digest}
    plan_identity = {**base, "kind": "director_plan", "task_id": "plan", "parent_task_id": stage.id,
                     "attempt": 1, "plan_digest": None, "task_digest": _digest({"stage": stage.id, "required": required})}
    proposal = _run_agent(ctx, f"{stage.id}:v{visit}:plan:a1", "ceo", _plan_prompt(ctx, stage, bodies, required),
                          stage.plan, stage.reads, plan_identity, timeout_s=3600)
    ctx.write_json(f"{directory}/plan-receipt.json", proposal)
    _require(proposal["status"] == "completed", f"{stage.id} planning failed: {proposal['error']}")
    validation = {**base, "kind": "director_plan_validation", "plan_path": stage.plan,
                  "raw_plan_receipt": f"{directory}/plan-receipt.json", "required_coverage": required,
                  "custom_child_count": {"minimum": stage.minimum, "maximum": stage.maximum},
                  "item_id_pattern": ITEM_ID.pattern, "task_slug_pattern": SLUG.pattern}
    try:
        plan = normalize_plan(_json(proposal["body"], "director plan"), stage, required)
    except ExecutionError as error:
        ctx.write_json(f"{directory}/plan-validation.json", {**validation, "status": "failed", "error": str(error)})
        raise
    ctx.write_json(f"{directory}/plan-validation.json", {**validation, "status": "passed", "error": None,
                  "custom_children": len(plan["tasks"])})
    if stage is QA:
        plan["tasks"].extend(_mandatory_tasks(ctx))
    ctx.write_json(stage.plan, plan)
    plan_digest = _digest(plan)
    proposal["plan_digest"] = plan_digest
    ctx.write_json(f"{directory}/plan-receipt.json", proposal)
    # Resolve every requested route before any child can have effects.
    for task in plan["tasks"]:
        _route(ctx.model, task["id"], task["role"])
    current: dict[str, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    counts = {task["id"]: 0 for task in plan["tasks"]}
    pending = {task["id"]: "" for task in plan["tasks"]}
    cancelled = threading.Event()
    model = ctx.model.with_cancellation(cancelled) if callable(getattr(ctx.model, "with_cancellation", None)) else ctx.model

    def dispatch(task: dict[str, Any], retry: str) -> dict[str, Any]:
        identifier = task["id"]
        counts[identifier] += 1
        attempt = counts[identifier]
        identity = {**base, "kind": "child_report", "task_id": identifier, "parent_task_id": stage.id,
                    "attempt": attempt, "plan_digest": plan_digest, "task_digest": _digest(task)}
        prompt = _child_prompt(ctx, stage, task, plan, bodies, identity, retry)
        identity["input_digest"] = _digest(prompt.encode())
        local = RunContext(ctx.workdir, knobs=ctx.knobs, facts=ctx.facts, task=ctx.task,
                           model=model, enforce_writes=ctx.enforce_writes)
        local.budget = ctx.budget
        local._director_deadline = ctx._director_deadline
        record = _run_agent(local, f"{stage.id}:v{visit}:{identifier}:a{attempt}", task["role"], prompt,
                            task["output"], stage.reads, identity)
        if record["status"] == "completed" and len(record["body"].encode()) <= stage.minimum_bytes:
            record.update(status="invalid", error=f"thin report: requires more than {stage.minimum_bytes} bytes")
        if record["status"] == "completed" and not record["body"].strip():
            record.update(status="invalid", error="empty report")
        local.write_json(f"{directory}/{identifier}-a{attempt}.json", record)
        return record

    round_number = 0
    try:
        while pending:
            round_number += 1
            selected_tasks = [task for task in plan["tasks"] if task["id"] in pending]
            with ThreadPoolExecutor(max_workers=min(parallelism, len(selected_tasks))) as pool:
                futures = {pool.submit(dispatch, task, pending[task["id"]]): task["id"] for task in selected_tasks}
                try:
                    for future in as_completed(futures):
                        report = future.result()
                        current[report["task_id"]] = report
                        history.append(report)
                except BaseException:
                    cancelled.set()
                    for future in futures:
                        future.cancel()
                    raise
            reports = [current[task["id"]] for task in plan["tasks"]]
            seen: dict[str, list[dict[str, Any]]] = {}
            for report in reports:
                if report["status"] == "completed":
                    seen.setdefault(report["output_digest"], []).append(report)
            for group in seen.values():
                if len(group) > 1:
                    for report in group:
                        report.update(status="invalid", error="duplicate child report content")
            join = {**base, "kind": "director_join", "attempt": round_number, "plan_digest": plan_digest,
                    "status": "completed", "reports": reports}
            join_path = f"{directory}/join-{round_number}.json"
            ctx.write_json(join_path, join)
            review_path = f"{directory}/quality-{round_number}.json"
            review_identity = {**base, "kind": "director_quality", "task_id": "quality", "parent_task_id": stage.id,
                               "attempt": round_number, "plan_digest": plan_digest,
                               "task_digest": _digest({"reports": [report["task_id"] for report in reports], "coverage_items": plan["coverage_items"]})}
            prompt = _review_prompt(ctx, stage, plan, reports, bodies)
            review_identity["input_digest"] = _digest(prompt.encode())
            receipt = _run_agent(ctx, f"{stage.id}:v{visit}:quality:a{round_number}", "ceo", prompt,
                                 review_path, (*stage.reads, stage.plan, join_path), review_identity, timeout_s=3600)
            ctx.write_json(f"{directory}/quality-{round_number}-receipt.json", receipt)
            _require(receipt["status"] == "completed", f"director quality review failed: {receipt['error']}")
            review = _validate_review(_json(receipt["body"], "quality review"), plan["tasks"], reports)
            review["raw_receipt"] = f"{directory}/quality-{round_number}-receipt.json"
            reviews.append(review)
            pending = {decision["id"]: decision["retry_prompt"] for decision in review["decisions"] if decision["verdict"] == "retry"}
            exhausted = [identifier for identifier in pending if counts[identifier] >= child_attempts]
            _require(not exhausted, f"{stage.id} exhausted same-child retries: {', '.join(exhausted)}")
        _require(reviews[-1]["coverage_complete"], "director quality assessment did not establish complete coverage")
    except BaseException as error:
        reports = [current[task["id"]] for task in plan["tasks"] if task["id"] in current]
        _publish(ctx, stage, base, plan, reports, history, reviews, "failed", str(error))
        raise
    _publish(ctx, stage, base, plan, reports, history, reviews, "completed")


def research_director(ctx: RunContext) -> None:
    _direct(ctx, RESEARCH)


def strategy_director(ctx: RunContext) -> None:
    _direct(ctx, STRATEGY)


def qa_director(ctx: RunContext) -> None:
    _direct(ctx, QA)
