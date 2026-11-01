"""The two extension points: what proposes, and what runs the search.

There are two kinds of thing people write here, and they need different seams.

**Type 1 — a proposer.** Subclasses :class:`Optimizer` and implements ``propose``.
It is handed the run's gradient and the **evaluated population**, chooses parents if
it has a policy about that, and returns typed edits. It cannot run anything, so it
cannot spend budget. A proposer works with any outer loop that calls ``propose``.

Choosing a parent is part of this, and that is a correction: until issue #32 the
loop passed only the elite, and ``Candidate.describe()`` carries no state, so a
policy that needed a wider pool -- islands, a niche grid, a tournament over an
archive -- could not be written here and had to be an outer loop instead. Five SRF
methods were loops for that reason and only that reason. See
:mod:`refactory_dsh.optim.policies`.

**Type 2 — an outer loop.** Subclasses :class:`OuterLoop` and overrides its
methods. Some strategies are not "propose better", they are a different control
flow: a switch inside an iteration, a fork with no loop at all, a staged curriculum.
No number of policy hooks on a fixed loop can express those, because they disagree
about what an iteration *is*.

The two guarantees hold for both, and neither can be overridden:

- an invalid edit can never be evaluated
- a refusal costs microseconds and no budget

`Run` is the harness: the mode, the primitives, the budget, the history. A strategy
is written against it, so what a strategy can do is what the harness offers rather
than what it happens to reach for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import structlog

from refactory_dsh.optim import Candidate, Optimizer, Proposal, Rejection
from refactory_dsh.optim.base import summarise_reasons

log = structlog.get_logger()


class OuterLoopError(RuntimeError):
    """A search cannot proceed. Always names what is wrong."""


@dataclass
class Run:
    """Everything a search strategy is built from.

    Cheap and expensive are deliberately distinct: ``attempt`` applies a typed edit
    and validates it for microseconds and no budget, ``evaluate`` runs the mode and
    spends one unit. Every strategy is built on that distinction, so the API makes
    it impossible to blur.

    The loop is kept as a private attribute (``_loop``) so that optimizer-facing
    code cannot reach val/test records or workdirs through it.
    """

    _loop: Any
    log_lines: list[str] = field(default_factory=list)
    stopped: str | None = None

    # ── the cheap primitive ────────────────────────────────────────────────
    def attempt(self, proposal: Proposal, parent: Candidate) -> Candidate | list[str]:
        """Apply a typed edit and validate it. Spends no budget."""
        from refactory_dsh.nn import edits
        op_name = proposal.edit.get("op")
        # Unknown and unpermitted are two facts with two fixes; see Loop._refinement.
        refused = edits.unknown(op_name)
        if refused is not None:
            reason = [refused]
            self._loop.rejections.append(Rejection(1, proposal.edit, reason, rationale=proposal.rationale))
            return reason
        allowed = self._loop.config.allowed_ops or edits.enabled(self._loop.config.edit_preset)
        if op_name not in allowed:
            reason = ["E-policy: operation is outside this run's permitted edits"]
            self._loop.rejections.append(Rejection(1, proposal.edit, reason, rationale=proposal.rationale))
            return reason
        outcome = self._loop.optimizer.attempt(
            proposal,
            parent,
            self._loop.invariants,
            True,
            self._loop.config.frozen,
            self._loop._tried(),  # noqa: SLF001 - the harness owns the population
            seed_state=getattr(self._loop, '_original_seed_state', None),
        )
        if isinstance(outcome, list):
            self._loop.rejections.append(
                Rejection(attempt=1, edit=proposal.edit, reasons=outcome,
                          rationale=proposal.rationale)
            )
        return outcome

    # ── the expensive primitive ────────────────────────────────────────────
    def evaluate(
        self,
        candidate: Candidate,
        *,
        part: str = "train",
        items: list[str] | None = None,
        seed: int | None = None,
    ) -> Candidate:
        """Run the mode and score it. Spends one unit of budget.

        When ``part`` is ``'train'``, delegates to the existing train evaluation.
        When ``part`` is ``'val'`` or ``'test'``, evaluates on that part in a
        fresh occasion directory without touching ``candidate.score``.

        When a manifest exists and ``items`` is provided, the items must be a
        subset of the manifest's ids for that part.
        """
        manifest = getattr(self._loop, '_parts_manifest', None)
        if part == "train":
            if self._loop.evaluations >= self._loop.config.max_evaluations:
                raise OuterLoopError(
                    "the evaluation budget is spent; a strategy must ask budget_left() "
                    "before evaluating"
                )
            # Validate train items against manifest when both exist.
            if manifest is not None and items is not None and "train" in manifest:
                allowed = set(manifest["train"])
                extra = set(items) - allowed
                if extra:
                    raise OuterLoopError(
                        f"items {sorted(extra)} are not in the manifest's train ids; "
                        f"an optimizer may pick a train minibatch but never val/test ids as 'train'"
                    )
            self._loop._evaluate(candidate, seed=seed, items=items)  # noqa: SLF001
        elif part in ("val", "test"):
            # Validate items against manifest when both exist.
            if manifest is not None and items is not None and part in manifest:
                allowed = set(manifest[part])
                extra = set(items) - allowed
                if extra:
                    raise OuterLoopError(
                        f"items {sorted(extra)} are not in the manifest's {part} ids"
                    )
            self._loop._evaluate_part(candidate, part, items, seed)  # noqa: SLF001
        else:
            raise OuterLoopError(f"unknown part {part!r}; expected 'train', 'val' or 'test'")
        return candidate

    def item_records(self, candidate: Candidate, *, part: str = "train") -> list[dict[str, Any]]:
        """Return train records for the given candidate.

        Val and test records are excluded from optimizer access by design.
        """
        if part != "train":
            raise OuterLoopError("only train records are accessible to an optimizer")
        return list(self._loop._train_item_records.get(candidate.id, []))  # noqa: SLF001

    def val_scores(self, candidate: Candidate) -> list[dict[str, Any]]:
        """Score-only view of val occasions: part, occasion, score, exit_ok, error, duration_s.

        No records or workdirs are exposed.
        """
        results = self._loop._part_results.get(candidate.id, {}).get("val", [])  # noqa: SLF001
        return [
            {k: v for k, v in occ.items() if k in ("part", "occasion", "score", "exit_ok", "error", "duration_s")}
            for occ in results
        ]

    def keep(self, candidate: Candidate) -> Candidate:
        """Add a candidate to the search. A strategy that evaluates must also keep.

        Separate from ``evaluate`` because a strategy may evaluate a point and
        decide not to keep it -- a fork whose branches are all discarded is a
        legitimate search.
        """
        self._loop.step([candidate], run=self)  # noqa: SLF001
        self._loop.update(self, candidate)  # noqa: SLF001
        return candidate

    # ── the state of the search ────────────────────────────────────────────
    def seed(self) -> Candidate:
        return self._loop.population[0]

    def history(self) -> list[Candidate]:
        return [candidate for candidate in self._loop.population if candidate.evaluated]

    def parents(self, n: int, *, by: str = "score") -> list[Candidate]:
        return self._loop.select_parents(n, by=by)

    def budget_left(self) -> int:
        left = max(0, self._loop.config.max_evaluations - self._loop.evaluations)
        if hasattr(self._loop, 'rollouts') and getattr(self._loop.config, 'max_rollouts', None) is not None:
            rollout_left = max(0, self._loop.config.max_rollouts - self._loop.rollouts)
            left = min(left, rollout_left)
        return left

    def rejections(self) -> list[str]:
        return summarise_reasons(
            [reason for rejection in self._loop.rejections for reason in rejection.reasons]
        )

    def state(self) -> dict[str, Any]:
        return self._loop.state_dict()

    def knob_surface(self) -> list[dict[str, Any]]:
        return self._loop._context().knobs  # noqa: SLF001

    # ── stopping, and saying what happened ────────────────────────────────
    def stop(self, reason: str) -> None:
        self.stopped = reason or "stopped"

    def check_control(self) -> str | None:
        """Check for pause and stop requests from a person.

        Call at iteration boundaries — the top of a strategy's while-loop —
        not from inside evaluation callbacks. Returns ``'stop'`` when a person
        asked the run to stop, or ``None`` otherwise. A pause blocks here
        until released; the caller does not see it.

        One-shot population acts (keep, reject, adopt) are honoured and
        consumed. This delegates to the loop's control handler, so the
        consume-once semantics are preserved.
        """
        return self._loop._apply_control()  # noqa: SLF001

    def save_strategy_state(self) -> None:
        """Journal the strategy's current state immediately.

        Call from inside a custom ``strategy()`` override to persist state
        at any point — not only at the epoch boundary.  The journaled state
        is the return value of ``strategy_state_dict()`` at call time.
        Replay restores the last journaled ``strategy_state`` event,
        regardless of whether it was written by this method or by the
        epoch-end journal.
        """
        self._loop._save_strategy_state()  # noqa: SLF001

    def log(self, message: str) -> None:
        self.log_lines.append(str(message))
        (self._loop.root / "strategy.log").parent.mkdir(parents=True, exist_ok=True)
        with (self._loop.root / "strategy.log").open("a") as handle:
            handle.write(str(message) + "\n")


class OuterLoop:
    """Base class for a search. Override the steps, or ``run`` entirely."""

    #: The name this loop is selected by.
    name: str = "default"
    #: Whether it needs a model, so construction does not have to know.
    requires_model: bool = False

    #: Steps a subclass may override. Each has a working default.
    overridable = (
        "strategy", "select_parents", "propose", "batch_size", "install", "accepts",
        "update", "should_stop", "on_epoch_end", "report", "memory",
        "strategy_state_dict", "load_strategy_state",
    )

    def configure(
        self, *, model: Any = None, model_name: str = "", provider: str = "auto", **options: Any
    ) -> None:
        """Take what the run has, rather than the constructor knowing per name."""
        self.model = model
        self.model_name = model_name
        self.provider = provider
        self.options = dict(options)

    # ── the entry point, in one place ──────────────────────────────────────
    def run(self) -> dict[str, Any]:
        """Initialise, drive the strategy, and report.

        Written once so every strategy ends the same way: a run always reports the
        bound it stopped for, and never ends for an unstated reason.
        """
        self.init()
        run = Run(_loop=self)
        self.strategy(run)
        if run.stopped:
            self.bound = str(run.stopped)
        return self.report()

    # ── what a subclass writes ─────────────────────────────────────────────
    def strategy(self, run: Run) -> None:
        """Drive the search. The one method a control-flow strategy implements.

        The default loop proposes a batch against the current parents, keeps what it
        can, evaluates it, and decides whether the epoch is over. A strategy that is
        a different control flow overrides this and calls the same primitives: a fork
        with no loop at all, a switch inside an iteration, a staged curriculum.
        """
        raise NotImplementedError

    def select_parents(self, n: int, *, by: str = "score") -> list[Candidate]:
        raise NotImplementedError

    def propose(self, run: Run, parents: Sequence[Candidate], n: int) -> list[Proposal]:
        raise NotImplementedError

    def batch_size(self, run: Run) -> int:
        return int(run._loop.config.batch_size)  # noqa: SLF001

    def install(self, run: Run, proposals: Sequence[Proposal], parents: Sequence[Candidate]) -> list[Candidate]:
        """Turn proposals into candidates, refusing what cannot run."""
        accepted: list[Candidate] = []
        for proposal in proposals:
            parent = next((p for p in parents if p.id == proposal.parent), parents[0])
            outcome = run.attempt(proposal, parent)
            if isinstance(outcome, Candidate):
                accepted.append(outcome)
        return accepted

    def accepts(self, run: Run, candidate: Candidate) -> bool:
        """Whether a candidate is kept. A loser may still be archived by ``update``."""
        return True

    def update(self, run: Run, candidate: Candidate) -> None:
        """Add a candidate to the search's state."""

    def should_stop(self, run: Run) -> str | None:
        return run.stopped

    def on_epoch_end(self, run: Run) -> None:
        """Called once per iteration, after the scheduler."""

    def report(self, run: Run) -> dict[str, Any]:
        return {}

    def memory(self) -> Iterable[Any]:
        """Declared memory namespaces, if this loop keeps anything across runs."""
        return ()

    def strategy_state_dict(self) -> dict[str, Any]:
        """The strategy's own state, journaled each epoch for resume.

        Separate from ``state_dict()`` (which is the search's read-only view):
        a strategy may track convergence history, temperature schedules, or
        multi-phase progress that the search itself does not model.

        Override and return a JSON-serialisable dict. The default is empty,
        making this opt-in: strategies that do not need state do not override.
        The returned dict is journaled after each epoch and whenever
        ``run.save_strategy_state()`` is called, and restored by
        ``load_strategy_state`` during replay.
        """
        return dict(self._strategy_state)

    def load_strategy_state(self, state: dict[str, Any]) -> None:
        """Restore strategy state from a journaled dict.

        Called during the journal walk (replay), each time a
        ``strategy_state`` event is encountered.  The last such event wins.
        Override to restore instance variables from the dict. The default
        stores it in ``_strategy_state``.
        """
        self._strategy_state = dict(state)


def run_strategy(loop: Any) -> dict[str, Any]:
    """Run an outer loop and return its report, with the bound it stopped for.

    A run never ends for an unstated reason: whatever a strategy returns as its
    stop reason reaches the report, and a strategy that says nothing says why not.
    """
    run = Run(_loop=loop)
    loop.optimizer_run = run
    loop.run_strategy(run)
    if run.stopped:
        loop.bound = str(run.stopped)
    return loop.report()
