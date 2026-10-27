"""One durable resource ledger shared by a run and its owned workers.

Elapsed limits are checked between nodes and while owned model processes run.
Token limits are checked at provider usage boundaries: an in-flight provider
request can overshoot by that request's usage. A backend without usage telemetry
must refuse a token-limited launch; absence is never measured zero.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterator


LEDGER = Path(".refactory/resource-usage.json")
ENV_LEDGER = "REFACTORY_BUDGET_LEDGER"


class BudgetExhausted(RuntimeError):
    """Required work did not fit; no success or validation is implied."""


class _UnmeteredModel:
    """Keep deterministic use possible, but refuse an unaccountable completion."""

    def __init__(self, model: Any, budget: Any) -> None:
        self._model = model
        self._budget = budget

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def with_budget(self, budget: Any) -> Any:
        return type(self)(self._model, budget)

    def for_node(self, node_id: str, role: str = "worker") -> Any:
        route = getattr(self._model, "for_node", None)
        model = route(node_id, role=role) if callable(route) else self._model
        return type(self)(model, self._budget)

    def complete(self, prompt: str, *, system: str = "", workdir: Any = None, **workspace: Any) -> str:
        self._budget.require_usage_capability(False)
        raise BudgetExhausted("custom model lacks shared token accounting")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Identical resource controls for the direct authoring entry points."""
    parser.add_argument("--max-elapsed-s", type=float, default=None, help="shared elapsed seconds, including nested work and validation")
    parser.add_argument("--max-tokens", type=int, default=None, help="shared tokens; requires provider usage telemetry")
    parser.add_argument("--validation-reserve-s", type=float, default=0, help="elapsed seconds retained for required final validation")
    parser.add_argument("--validation-reserve-tokens", type=int, default=0, help="tokens retained for required final validation")
    parser.add_argument("--warn-tokens", type=int, default=None,
                        help="per-step token warning threshold; emits a run_warning once per step (does not stop the run)")


def arguments(args: argparse.Namespace) -> dict[str, Any] | None:
    limits = ResourceLimits.from_dict(vars(args))
    return asdict(limits) if limits.configured else None


@dataclass(frozen=True)
class ResourceLimits:
    max_elapsed_s: float | None = None
    max_tokens: int | None = None
    validation_reserve_s: float = 0
    validation_reserve_tokens: int = 0
    warn_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_elapsed_s", "max_tokens"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive or unset")
        for name in ("validation_reserve_s", "validation_reserve_tokens"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in ("max_tokens", "validation_reserve_tokens"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or int(value) != value):
                raise ValueError(f"{name} must be an integer")
        if self.warn_tokens is not None and (isinstance(self.warn_tokens, bool)
                or int(self.warn_tokens) != self.warn_tokens or self.warn_tokens <= 0):
            raise ValueError("warn_tokens must be a positive integer or unset")
        if self.max_elapsed_s is not None and self.validation_reserve_s >= self.max_elapsed_s:
            raise ValueError("validation_reserve_s must be less than max_elapsed_s")
        if self.max_tokens is not None and self.validation_reserve_tokens >= self.max_tokens:
            raise ValueError("validation_reserve_tokens must be less than max_tokens")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ResourceLimits:
        return cls(**{name: values[name] for name in cls.__dataclass_fields__ if name in values})

    @property
    def configured(self) -> bool:
        return self.max_elapsed_s is not None or self.max_tokens is not None or self.warn_tokens is not None


class RunBudget:
    """Cross-process, idempotent accounting; all descendants use the same path.

    A provider receipt's worker_id identifies one cumulative usage total. Report
    individual child totals or an inclusive parent total, never both. Calling
    record_usage repeatedly for one worker replaces its total rather than adding
    it again. A lock covers read/modify/write; the JSON replacement is atomic.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()

    @classmethod
    def create(cls, workdir: Path, limits: ResourceLimits | dict[str, Any] | None = None) -> RunBudget:
        """A fresh owner must not silently inherit an old, possibly paused ledger."""
        ledger = cls(Path(workdir) / LEDGER)
        if isinstance(limits, dict):
            limits = ResourceLimits.from_dict(limits)
        with ledger._locked() as state:
            if state:
                raise ValueError("this directory already has a resource ledger; use a new run directory or explicit resume")
            state.update({"schema": 1, "limits": asdict(limits or ResourceLimits()),
                          "started_at": time.time(), "elapsed_before_s": 0.0,
                          "workers": {}, "phase": "work", "exhausted": False,
                          "reason": None})
        return ledger

    @classmethod
    def open(cls, workdir: Path, limits: ResourceLimits | dict[str, Any] | None = None) -> RunBudget:
        ledger = cls(Path(workdir) / LEDGER)
        if isinstance(limits, dict):
            limits = ResourceLimits.from_dict(limits)
        with ledger._locked() as state:
            if not state:
                state.update({"schema": 1, "limits": asdict(limits or ResourceLimits()),
                              "started_at": time.time(), "elapsed_before_s": 0.0,
                              "workers": {}, "phase": "work", "exhausted": False,
                              "reason": None})
            elif limits is not None and state["limits"] != asdict(limits):
                raise ValueError("run resource limits are immutable; extend them explicitly before resume")
        return ledger

    @classmethod
    def from_path(cls, path: str | Path) -> RunBudget:
        ledger = cls(Path(path))
        if not ledger.path.is_file():
            raise ValueError(f"resource ledger does not exist: {ledger.path}")
        return ledger

    @classmethod
    def inherited(cls, model: Any = None) -> RunBudget | None:
        budget = getattr(model, "_budget", None)
        if budget is not None:
            return budget
        context = model.context() if callable(getattr(model, "context", None)) else {}
        path = context.get("budget_ledger") or os.environ.get(ENV_LEDGER)
        return cls.from_path(path) if path else None

    @contextmanager
    def _locked(self) -> Iterator[dict[str, Any]]:
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                state = json.loads(self.path.read_text()) if self.path.exists() else {}
                yield state
                pending = self.path.with_suffix(f".{os.getpid()}.tmp")
                pending.write_text(json.dumps(self._summary(state), indent=2, sort_keys=True) + "\n")
                pending.replace(self.path)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    @property
    def limits(self) -> ResourceLimits:
        return ResourceLimits.from_dict(self.snapshot()["limits"])

    @staticmethod
    def _summary(state: dict[str, Any]) -> dict[str, Any]:
        workers = list(state.get("workers", {}).values())
        known = [entry for entry in workers if entry.get("tokens") is not None]
        unavailable = any(entry.get("tokens") is None for entry in workers)
        # No provider completion yet is not a measured zero-token completion.
        status = ("unavailable" if not workers or unavailable else
                  "estimated" if any(entry["status"] == "estimated" for entry in known) else "measured")
        elapsed = float(state.get("elapsed_before_s", 0))
        if state.get("started_at") is not None:
            elapsed += max(0, time.time() - state["started_at"])
        costs = [entry.get("estimated_cost_usd") for entry in workers]
        return {
            **state, "elapsed_s": round(elapsed, 6),
            "tokens": sum(entry["tokens"] for entry in known) if workers and not unavailable else None,
            "known_tokens": sum(entry["tokens"] for entry in known),
            "token_status": status,
            "estimated_cost_usd": sum(costs) if costs and all(cost is not None for cost in costs) else None,
            "cost_status": "estimated" if costs and all(cost is not None for cost in costs) else "unavailable",
            "enforcement": "elapsed: node boundaries and owned process polling; tokens: usage boundaries, one in-flight request per worker may overshoot",
        }

    def snapshot(self) -> dict[str, Any]:
        with self._locked() as state:
            return self._summary(state)

    def remaining_s(self, phase: str | None = None) -> float | None:
        state = self.snapshot()
        maximum = state["limits"].get("max_elapsed_s")
        if maximum is None:
            return None
        reserve = state["limits"].get("validation_reserve_s", 0) if (phase or state["phase"]) != "final_validation" else 0
        return max(0, maximum - reserve - state["elapsed_s"])

    def set_validation_owner(self, owner: str) -> None:
        if owner not in ("graph", "search"):
            raise ValueError("validation owner must be graph or search")
        with self._locked() as state:
            state["validation_owner"] = owner

    @contextmanager
    def retain_validation_reserve(self) -> Iterator[None]:
        """Nested evaluation cannot promote an enclosing run's resource phase."""
        with self._locked() as state:
            state["validation_holds"] = int(state.get("validation_holds", 0)) + 1
        try:
            yield
        finally:
            with self._locked() as state:
                state["validation_holds"] = max(0, int(state.get("validation_holds", 0)) - 1)

    def checkpoint_node(self, phase: str | None = None) -> None:
        """A candidate's internal validator cannot spend search holdout reserve."""
        state = self.snapshot()
        if state.get("validation_owner", "graph") == "search" or state.get("validation_holds", 0):
            phase = None
        self.checkpoint(phase)

    def checkpoint(self, phase: str | None = None) -> None:
        reason = None
        with self._locked() as state:
            if phase is not None:
                if phase not in ("work", "final_validation"):
                    raise ValueError("resource phase must be work or final_validation")
                state["phase"] = phase
            summary = self._summary(state)
            limits = state["limits"]
            work = state["phase"] != "final_validation"
            time_limit = limits.get("max_elapsed_s")
            token_limit = limits.get("max_tokens")
            if time_limit is not None and summary["elapsed_s"] >= time_limit - (limits.get("validation_reserve_s", 0) if work else 0):
                reason = "elapsed-time budget exhausted" + (" (final validation reserve retained)" if work and limits.get("validation_reserve_s") else "")
            if token_limit is not None:
                if state.get("workers") and summary["token_status"] == "unavailable":
                    reason = "token usage unavailable; token-limited work cannot continue"
                elif summary["known_tokens"] >= token_limit - (limits.get("validation_reserve_tokens", 0) if work else 0):
                    reason = "token budget exhausted" + (" (final validation reserve retained)" if work and limits.get("validation_reserve_tokens") else "")
            state["exhausted"] = reason is not None
            state["reason"] = reason
        if reason:
            raise BudgetExhausted(reason)

    def bind_model(self, model: Any) -> Any:
        """Attach the accounting protocol or guard unsupported costly calls."""
        if model is None:
            return None
        if callable(getattr(model, "with_budget", None)):
            return model.with_budget(self)
        return _UnmeteredModel(model, self) if self.limits.max_tokens is not None else model

    def require_usage_capability(self, available: bool) -> None:
        if self.limits.max_tokens is not None and not available:
            with self._locked() as state:
                state["exhausted"] = True
                state["reason"] = "backend has no token telemetry; token-limited launch refused"
            raise BudgetExhausted("backend has no token telemetry; token-limited launch refused")

    def record_usage(self, worker_id: str, tokens: int | None, *, estimated_cost_usd: float | None = None,
                     status: str = "measured", node_id: str = "") -> None:
        if not worker_id or status not in ("measured", "estimated", "unavailable"):
            raise ValueError("usage needs a worker identity and measured/estimated/unavailable status")
        if tokens is not None and (isinstance(tokens, bool) or int(tokens) != tokens or tokens < 0):
            raise ValueError("tokens must be a nonnegative integer or unavailable")
        if estimated_cost_usd is not None and (not math.isfinite(estimated_cost_usd) or estimated_cost_usd < 0):
            raise ValueError("estimated cost must be nonnegative or unavailable")
        with self._locked() as state:
            state.setdefault("workers", {})[worker_id] = {
                "tokens": tokens, "status": status if tokens is not None else "unavailable",
                "estimated_cost_usd": estimated_cost_usd,
            }
            if node_id:
                state.setdefault("node_workers", {})[worker_id] = node_id

    def node_tokens(self, node_id: str) -> int:
        """Sum tokens for all workers belonging to the given node_id."""
        with self._locked() as state:
            total = 0
            for worker_id, node in state.get("node_workers", {}).items():
                if node == node_id:
                    entry = state.get("workers", {}).get(worker_id, {})
                    tokens = entry.get("tokens")
                    if isinstance(tokens, (int, float)) and tokens >= 0:
                        total += int(tokens)
            return total

    def warn_if_exceeded(self, node_id: str, node_tokens: int) -> None:
        """Emit a run_warning once per step when it crosses warn_tokens."""
        should_warn = False
        warn_tokens = None
        with self._locked() as state:
            warn_tokens = state.get("limits", {}).get("warn_tokens")
            if warn_tokens is None or node_tokens < warn_tokens:
                return
            warned = set(state.get("warned_nodes", []))
            if node_id in warned:
                return
            warned.add(node_id)
            state["warned_nodes"] = sorted(warned)
            should_warn = True
        # Emit outside the lock to avoid holding it during I/O.
        if should_warn:
            # Find the run directory from the budget's own ledger path.
            workdir = self.path.parent.parent
            from refactory_dsh import live
            live.emit(workdir, "run_warning", node_id=node_id, state="warning",
                      summary=f"step {node_id} reached {node_tokens:,} tokens (threshold: {warn_tokens:,})")

    def pause(self) -> None:
        """Stop charging idle time after the owning run ends or pauses."""
        with self._locked() as state:
            state["elapsed_before_s"] = self._summary(state)["elapsed_s"]
            state["started_at"] = None

    def resume(self, limits: ResourceLimits | None = None) -> None:
        """Retain consumption; only an explicit caller may change the allocation."""
        with self._locked() as state:
            if limits is not None:
                state["limits"] = asdict(limits)
            if state.get("started_at") is None:
                state["started_at"] = time.time()
            state["exhausted"] = False
            state["reason"] = None
        try:
            self.checkpoint()
        except BudgetExhausted:
            self.pause()
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read/check a shared run resource ledger")
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--phase", choices=("work", "final_validation"))
    args = parser.parse_args(argv)
    budget = RunBudget.from_path(args.ledger)
    try:
        if args.check:
            budget.checkpoint_node(args.phase)
    except BudgetExhausted as exc:
        print(json.dumps({"status": "budget_exhausted", "error": str(exc), "resources": budget.snapshot()}))
        return 2
    print(json.dumps(budget.snapshot()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
