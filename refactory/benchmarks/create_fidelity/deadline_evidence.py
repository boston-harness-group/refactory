"""Verify recorded dispatch clock observations without claiming expiry coverage."""
from __future__ import annotations

from datetime import datetime
import math


def _number(value, label, *, optional=False):
    if value is None and optional:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return value


def _same(actual, expected, label):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f"{label} differs from the observed clocks")
    elif not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(f"{label} differs from the observed clocks")


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError("dispatch timestamp missing or invalid")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("dispatch timestamp must include its timezone")
    return result


def _sample(sample, deadlines, worker, attempt):
    if not isinstance(sample, dict):
        raise ValueError("dispatch clock observation is not an object")
    required = {"attempt", "sampled_at", "sample_phase", "stage_remaining_s", "stage_clock",
                "run_remaining_s", "run_budget_snapshot", "enclosing_deadlines",
                "effective_remaining_s", "effective_limiters", "idle_timer", "enforcement_layers"}
    if not required <= set(sample):
        raise FileNotFoundError("dispatch observation omitted remaining-clock or idle-policy fields")
    if type(sample["attempt"]) is not int or sample["attempt"] != attempt:
        raise ValueError("dispatch attempts are missing, duplicated or out of order")
    if sample["sample_phase"] != "pre_transport_dispatch":
        raise ValueError("remaining allowance was not observed at pre-dispatch")
    observed = _timestamp(sample["sampled_at"])
    if observed < _timestamp(worker["started"]) or (worker.get("finished") and observed > _timestamp(worker["finished"])):
        raise ValueError("dispatch timestamp lies outside its worker interval")
    stage = _number(sample["stage_remaining_s"], "stage remaining")
    wall = _number(deadlines.get("stage_timeout_s"), "stage timeout")
    if wall == 0 or stage > wall:
        raise ValueError("dispatch remaining stage allowance exceeds its wall")
    clock = sample["stage_clock"]
    if not isinstance(clock, dict) or (clock.get("scope"), clock.get("status")) not in (
            ("absolute_stage", "active"), ("per_completion", "window_not_started")):
        raise ValueError("stage clock does not distinguish its active wall from an unstarted completion window")
    if clock["scope"] == "per_completion":
        _same(stage, _number(deadlines.get("completion_timeout_s"), "completion timeout"), "completion window")
    run = _number(sample["run_remaining_s"], "run remaining", optional=True)
    resources = sample["run_budget_snapshot"]
    ledger = deadlines.get("run_budget")
    if resources is None:
        if ledger is not None or run is not None:
            raise ValueError("configured run ledger has no dispatch snapshot")
    else:
        if not isinstance(resources, dict) or not ledger or resources.get("path") != ledger:
            raise ValueError("dispatch budget snapshot belongs to a different ledger")
        elapsed = _number(resources.get("elapsed_s"), "elapsed run time")
        maximum = _number(resources.get("max_elapsed_s"), "elapsed limit", optional=True)
        reserve = _number(resources.get("validation_reserve_s"), "validation reserve")
        phase = resources.get("phase")
        if phase not in ("work", "final_validation"):
            raise ValueError("unknown dispatch budget phase")
        retained = _number(resources.get("retained_validation_reserve_s"), "retained reserve")
        _same(retained, 0 if phase == "final_validation" else reserve, "retained reserve")
        if maximum is not None and (maximum == 0 or reserve >= maximum):
            raise ValueError("invalid elapsed allocation or validation reserve")
        expected = None if maximum is None else max(0, maximum - retained - elapsed)
        _same(run, expected, "remaining run allowance")
        _same(_number(resources.get("remaining_s"), "snapshot remaining", optional=True), expected, "snapshot remaining")
        if resources.get("elapsed_limit_status") != ("inactive" if maximum is None else "active"):
            raise ValueError("elapsed timer policy does not match the configured limit")
    clocks = {"stage": stage}
    if run is not None:
        clocks["run_budget"] = run
    enclosing = sample["enclosing_deadlines"]
    if not isinstance(enclosing, list):
        raise ValueError("enclosing timer observations must be a list")
    for timer in enclosing:
        if not isinstance(timer, dict) or not isinstance(timer.get("scope"), str) or not timer["scope"] or timer["scope"] in clocks:
            raise ValueError("enclosing timer scope is absent, duplicated or collides with a worker clock")
        if not isinstance(timer.get("enforcement"), str) or not timer["enforcement"].strip():
            raise ValueError("enclosing timer enforcement layer is unlabeled")
        clocks[timer["scope"]] = _number(timer.get("remaining_s"), "enclosing timer remaining")
    node = worker["node_id"]
    if not isinstance(node, str):
        raise ValueError("worker node identity is invalid")
    if ":" in node and node.split(":", 1)[0] in ("research_director", "strategy_director", "qa_director") and "composite_stage" not in clocks:
        raise FileNotFoundError("director dispatch omitted its enclosing composite cancellation timer")
    effective = _number(sample["effective_remaining_s"], "effective remaining")
    _same(effective, min(clocks.values()), "effective remaining allowance")
    limiters = sample["effective_limiters"]
    expected_limiters = [name for name, remaining in clocks.items() if remaining == min(clocks.values())]
    if not isinstance(limiters, list) or limiters != expected_limiters:
        raise ValueError("recorded minimum limiter differs from the active clocks")
    idle = sample["idle_timer"]
    if (not isinstance(idle, dict) or idle.get("status") != "inactive" or idle.get("timeout_s") is not None
            or idle.get("scope") != "refactory_dispatch_owner" or not isinstance(idle.get("policy"), str) or not idle["policy"].strip()):
        raise ValueError("idle timer policy is absent or differs from the dispatch owner's inactive policy")
    layers = sample["enforcement_layers"]
    if not isinstance(layers, dict) or any(not isinstance(layers.get(key), str) or not layers[key].strip()
                                         for key in ("stage", "run_budget", "enclosing_deadlines")):
        raise ValueError("deadline enforcement layers are unlabeled")
    return observed


def dispatch_deadline_row(workers: list[tuple[str, dict]]) -> dict:
    refs = []
    status, reason = "passed", "Raw per-transport dispatch samples record stage and run time remaining, retained validation reserve, enclosing cancellation clocks, effective minimum and inactive owner idle policy; expiry enforcement is assessed separately."
    try:
        actual = [(path, worker) for path, worker in workers
                  if not (worker.get("node_id") == "" and worker.get("role") == "coordinator")]
        if not actual:
            raise FileNotFoundError("no individual worker dispatch observations; coordinator receipts are excluded")
        for path, worker in actual:
            refs.append(path)
            deadlines = worker.get("effective_deadlines")
            if not isinstance(deadlines, dict) or not isinstance(deadlines.get("dispatches"), list) or not deadlines["dispatches"]:
                raise FileNotFoundError("worker receipt lacks actual pre-dispatch remaining-clock samples")
            previous = None
            for attempt, sample in enumerate(deadlines["dispatches"], 1):
                observed = _sample(sample, deadlines, worker, attempt)
                if previous is not None and observed < previous:
                    raise ValueError("dispatch observation timestamps run backward")
                previous = observed
    except (OSError, FileNotFoundError) as error:
        status, reason = "unavailable", f"Dispatch deadline telemetry is incomplete: {error}"
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        status, reason = "failed", f"Dispatch deadline telemetry is invalid: {error}"
    return {"requirement_id": "F.deadlines.remaining", "status": status, "reason": reason, "evidence": refs}
