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
from typing import Any, Callable, Iterable, Sequence

import structlog

from refactory_dsh.optim import Candidate, Optimizer, Proposal, Rejection
from refactory_dsh.optim.base import summarise_reasons

log = structlog.get_logger()


@dataclass(frozen=True)
class EvalRequest:
    """One evaluation request for :meth:`Run.evaluate_many`."""
    candidate: Candidate
    part: str = "train"
    items: list[str] | None = None
    seed: int | None = None


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
        allowed = self._loop.permitted_ops()
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
        ``items`` may pick a train minibatch: a subset of the manifest's train ids,
        or, with no manifest, of the seed's train split -- never held-out ids.

        When ``part`` is ``'val'`` or ``'test'``, evaluates the manifest's whole part
        in a fresh occasion directory without touching ``candidate.score``. ``items``
        is refused there: an optimizer that chose its val subset would choose its
        own exam.

        Refused once the evaluation or rollout budget is spent, for every part.
        """
        loop = self._loop
        manifest = getattr(loop, '_parts_manifest', None)
        if part not in ("train", "val", "test"):
            raise OuterLoopError(f"unknown part {part!r}; expected 'train', 'val' or 'test'")
        max_rollouts = getattr(loop.config, "max_rollouts", None)
        if max_rollouts is not None and loop.rollouts >= max_rollouts:
            raise OuterLoopError(
                f"the rollout budget is spent ({loop.rollouts} of {max_rollouts}); "
                "a strategy must ask budget_left() before evaluating"
            )
        if part == "train":
            if loop.evaluations >= loop.config.max_evaluations:
                raise OuterLoopError(
                    "the evaluation budget is spent; a strategy must ask budget_left() "
                    "before evaluating"
                )
            if items is not None:
                if manifest is not None:
                    if "train" not in manifest:
                        raise OuterLoopError(
                            "the parts manifest has no entry for 'train'; "
                            "items cannot be provided for a part not in the manifest"
                        )
                    allowed = set(manifest["train"])
                    where = "the manifest's train ids"
                else:
                    seed_state = loop.population[0].state if loop.population else candidate.state
                    allowed = loop._expected_item_ids(seed_state, "train")[0]  # noqa: SLF001
                    where = "the train split"
                extra = set(str(i) for i in items) - allowed
                if extra:
                    raise OuterLoopError(
                        f"items {sorted(extra)} are not in {where}; "
                        f"an optimizer may pick a train minibatch but never val/test ids as 'train'"
                    )
            loop._evaluate(candidate, seed=seed, items=items)  # noqa: SLF001
            return candidate
        if items is not None:
            raise OuterLoopError(
                f"items cannot be chosen for {part!r}: val and test are always evaluated "
                f"in full from the parts manifest"
            )
        if manifest is None:
            raise OuterLoopError(
                f"no parts manifest exists; evaluating on {part!r} requires one "
                f"(LoopConfig.parts_manifest)"
            )
        if part not in manifest:
            raise OuterLoopError(
                f"the parts manifest has no {part!r} entry; add one to evaluate on {part!r}"
            )
        loop._evaluate_part(candidate, part, None, seed)  # noqa: SLF001
        return candidate

    # ── concurrent evaluation ──────────────────────────────────────────────
    def evaluate_many(
        self,
        requests: list[EvalRequest],
        *,
        on_result: Callable[[int, Candidate], None] | None = None,
    ) -> list[Candidate]:
        """Evaluate several independent candidates concurrently.

        Returns results in request order regardless of completion order.
        Requests naming the same candidate id run sequentially in request order;
        requests with distinct candidate ids run concurrently.  Budget is
        reserved atomically for the whole batch before any evaluation starts.

        At most one ``part="train"`` request per candidate id: train records
        are kept per candidate (``item_records``), so a second train run of the
        same candidate would overwrite the first before the caller could read
        it.  Such a batch is refused before anything runs or any budget is
        reserved; call ``evaluate`` for each run and read ``item_records``
        between.  Val and test occasions are kept per occasion, so several
        val/test requests for one candidate are allowed (run in order).

        ``on_result(index, candidate)`` fires in completion order, if provided.
        """
        import threading
        from collections import defaultdict
        from concurrent.futures import ThreadPoolExecutor, as_completed

        loop = self._loop
        manifest = getattr(loop, '_parts_manifest', None)

        if not requests:
            return []

        # ── validate every request ────────────────────────────────────────
        for i, req in enumerate(requests):
            if req.part not in ("train", "val", "test"):
                raise OuterLoopError(
                    f"request {i}: unknown part {req.part!r}; expected 'train', 'val' or 'test'"
                )
            if req.part == "train" and req.items is not None:
                if manifest is not None:
                    if "train" not in manifest:
                        raise OuterLoopError(
                            f"request {i}: the parts manifest has no entry for 'train'; "
                            "items cannot be provided for a part not in the manifest"
                        )
                    allowed = set(manifest["train"])
                    where = "the manifest's train ids"
                else:
                    seed_state = loop.population[0].state if loop.population else req.candidate.state
                    allowed = loop._expected_item_ids(seed_state, "train")[0]  # noqa: SLF001
                    where = "the train split"
                extra = set(str(x) for x in req.items) - allowed
                if extra:
                    raise OuterLoopError(
                        f"request {i}: items {sorted(extra)} are not in {where}"
                    )
            if req.part != "train":
                if req.items is not None:
                    raise OuterLoopError(
                        f"request {i}: items cannot be chosen for {req.part!r}: "
                        "val and test are always evaluated in full from the parts manifest"
                    )
                if manifest is None:
                    raise OuterLoopError(
                        f"request {i}: no parts manifest exists; evaluating on {req.part!r} "
                        "requires one (LoopConfig.parts_manifest)"
                    )
                if req.part not in manifest:
                    raise OuterLoopError(
                        f"request {i}: the parts manifest has no {req.part!r} entry"
                    )

        # ── one train request per candidate ───────────────────────────────
        train_ids_seen: set[str] = set()
        for req in requests:
            if req.part != "train":
                continue
            if req.candidate.id in train_ids_seen:
                raise OuterLoopError(
                    f"evaluate_many cannot evaluate candidate {req.candidate.id} on train "
                    "twice in one call: its train records would be overwritten; call "
                    "run.evaluate for each and read run.item_records between"
                )
            train_ids_seen.add(req.candidate.id)

        # ── atomic budget reservation ─────────────────────────────────────
        train_count = sum(1 for r in requests if r.part == "train")
        total_rollouts = 0
        for req in requests:
            if req.part == "train":
                if req.items is not None:
                    total_rollouts += max(len(req.items), 1)
                elif manifest is not None and "train" in manifest:
                    total_rollouts += max(len(manifest["train"]), 1)
                else:
                    seed_state = loop.population[0].state if loop.population else req.candidate.state
                    ids = loop._expected_item_ids(seed_state, "train")[0]  # noqa: SLF001
                    total_rollouts += max(len(ids), 1)
            else:
                if manifest is not None and req.part in manifest:
                    total_rollouts += max(len(manifest[req.part]), 1)
                else:
                    total_rollouts += 1

        with loop._state_lock:
            max_rollouts = getattr(loop.config, "max_rollouts", None)
            if train_count > 0:
                if loop.evaluations + train_count > loop.config.max_evaluations:
                    raise OuterLoopError(
                        "the evaluation budget is spent; a strategy must ask budget_left() "
                        "before evaluating"
                    )
            if max_rollouts is not None:
                if loop.rollouts + total_rollouts > max_rollouts:
                    raise OuterLoopError(
                        f"the rollout budget is spent ({loop.rollouts} of {max_rollouts}); "
                        "a strategy must ask budget_left() before evaluating"
                    )
            # Pre-increment evaluations for train requests so concurrent
            # budget_left() calls see the reservation.
            loop.evaluations += train_count

        # ── ensure item gate / slot dir ───────────────────────────────────
        loop._init_item_gate()  # noqa: SLF001

        # ── group requests by candidate id ────────────────────────────────
        # Requests with the same candidate id (at most one train, plus any
        # val/test occasions) run sequentially in request order; distinct
        # candidate ids run concurrently.
        groups: dict[str, list[int]] = defaultdict(list)
        for i, req in enumerate(requests):
            groups[req.candidate.id].append(i)

        results: list[Candidate | None] = [None] * len(requests)
        errors: list[BaseException | None] = [None] * len(requests)
        # on_result errors are tracked separately: they must not release
        # the evaluation's budget because the evaluation itself succeeded.
        on_result_errors: list[BaseException | None] = [None] * len(requests)
        failed_train_count = 0
        skipped_train_count = 0
        failed_train_lock = threading.Lock()

        def _run_one(index: int) -> None:
            req = requests[index]
            try:
                if req.part == "train":
                    loop._evaluate(  # noqa: SLF001
                        req.candidate, seed=req.seed, items=req.items,
                        _budget_reserved=True,
                        item_gate=getattr(loop, '_item_gate', None),
                    )
                else:
                    loop._evaluate_part(  # noqa: SLF001
                        req.candidate, req.part, req.items, req.seed,
                        item_gate=getattr(loop, '_item_gate', None),
                    )
                results[index] = req.candidate
            except BaseException as exc:
                errors[index] = exc
                if req.part == "train":
                    with failed_train_lock:
                        nonlocal failed_train_count
                        failed_train_count += 1
                return

            # on_result is called only after a successful evaluation.
            # An exception here must NOT release the evaluation's budget
            # (it ran and was paid for).  Record it and re-raise after
            # all evaluations settle.
            if on_result is not None:
                try:
                    on_result(index, req.candidate)
                except BaseException as exc:
                    on_result_errors[index] = exc

        def _run_group(indices: list[int]) -> None:
            """Run a group of requests for the same candidate sequentially."""
            for i, idx in enumerate(indices):
                _run_one(idx)
                if errors[idx] is not None:
                    # Count skipped train requests so their pre-reserved
                    # budget is un-reserved below.
                    for skipped_idx in indices[i + 1:]:
                        if requests[skipped_idx].part == "train":
                            with failed_train_lock:
                                nonlocal skipped_train_count
                                skipped_train_count += 1
                    break

        max_workers = getattr(loop.config, 'max_concurrent_evaluations', 4)
        # Each distinct candidate group is one concurrent unit
        group_list = list(groups.values())

        if len(group_list) == 1:
            # Single candidate or single request: run inline
            _run_group(group_list[0])
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(group_list))) as pool:
                futures = {pool.submit(_run_group, g): g for g in group_list}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except BaseException:
                        pass  # errors recorded in errors[]

        # ── un-reserve budget for failed and skipped train evals ──────────
        unreserve = failed_train_count + skipped_train_count
        if unreserve > 0:
            with loop._state_lock:
                loop.evaluations -= unreserve

        # ── re-raise the first error ──────────────────────────────────────
        # Evaluation errors take priority over on_result errors.
        first_error = next((e for e in errors if e is not None), None)
        first_on_result_error = next((e for e in on_result_errors if e is not None), None)
        error_to_raise = first_error or first_on_result_error
        if error_to_raise is not None:
            raise error_to_raise

        return [r for r in results if r is not None]  # type: ignore[misc]

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
        return self.part_scores(candidate, "val")

    def part_scores(self, candidate: Candidate, part: str) -> list[dict[str, Any]]:
        """Score-only view of a candidate's ``'val'`` or ``'test'`` occasions, oldest first.

        The fields ``val_scores`` gives, for either held-out part. A strategy that
        evaluates on test after its search (as SkillOpt does) reads the score it
        reports here; no records or workdirs are exposed.
        """
        if part not in ("val", "test"):
            raise OuterLoopError(f"part_scores covers 'val' and 'test', not {part!r}")
        results = self._loop._part_results.get(candidate.id, {}).get(part, [])  # noqa: SLF001
        return [
            {k: v for k, v in occ.items() if k in ("part", "occasion", "score", "exit_ok", "error", "duration_s")}
            for occ in results
        ]

    def has_part(self, part: str) -> bool:
        """Whether the parts manifest declares ``part``. Its ids stay withheld."""
        manifest = getattr(self._loop, '_parts_manifest', None)  # noqa: SLF001
        return manifest is not None and part in manifest

    def keep(self, candidate: Candidate) -> Candidate:
        """Add a candidate to the search. A strategy that evaluates must also keep.

        Separate from ``evaluate`` because a strategy may evaluate a point and
        decide not to keep it -- a fork whose branches are all discarded is a
        legitimate search.
        """
        self._loop.step([candidate], run=self)  # noqa: SLF001
        self._loop.update(self, candidate)  # noqa: SLF001
        return candidate

    # ── the run directory ───────────────────────────────────────────────────
    @property
    def root(self) -> Path:
        """The run's root directory.

        A strategy may write trace files or other artefacts beside the
        journal.  Exposed so a strategy does not reach through ``_loop``.
        """
        return self._loop.root

    def part_ids(self, part: str = "train") -> list[str]:
        """Item ids declared for *part* in the parts manifest.

        Returns an empty list when no manifest exists.  Only ``'train'``
        ids are exposed to optimiser-facing code; ``'val'`` and ``'test'``
        ids are withheld for the same reason ``item_records`` refuses them.
        """
        if part != "train":
            raise OuterLoopError("only train ids are accessible to an optimizer")
        manifest = getattr(self._loop, '_parts_manifest', None)  # noqa: SLF001
        if manifest is None:
            return []
        return list(manifest.get(part, []))

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

    def record_proposal_round(self, error: str | None = None) -> None:
        """Count one model call a custom strategy made to propose, and its failure.

        A custom strategy calls its model directly, not through the loop's
        ``propose``, so the run report's proposer health (``loop._proposer_health``:
        rounds, failures, ``all_failed``) would otherwise read zero rounds. A
        strategy that survives a failed call (``error`` names it) records it here,
        so a run whose model never answered says so in its report.
        """
        optimizer = self._loop.optimizer
        optimizer.proposal_rounds += 1
        if error is not None:
            optimizer.proposal_failures.append(str(error))

    def log(self, message: str) -> None:
        self.log_lines.append(str(message))
        (self._loop.root / "strategy.log").parent.mkdir(parents=True, exist_ok=True)
        with (self._loop.root / "strategy.log").open("a") as handle:
            handle.write(str(message) + "\n")


class OuterLoop:
    """Base class for a custom strategy — the **custom strategy seam**.

    A custom strategy subclasses ``OuterLoop`` and overrides the hooks in
    :attr:`overridable`: ``configure``, ``strategy(run)``,
    ``strategy_state_dict``, ``load_strategy_state``, and others.

    The loop machinery (``Loop``) is the caller.  It constructs the strategy
    object, calls ``configure(model=…, **strategy_options)`` after
    construction, creates a ``Run`` and passes it to ``strategy(run)``, and
    calls ``strategy_state_dict()`` / ``load_strategy_state()`` for
    persistence.

    The strategy keeps its own state as plain instance attributes.  It
    interacts with the loop machinery **only** through ``Run``'s public API
    (``run.evaluate()``, ``run.save_strategy_state()``, ``run.seed()``, etc.).
    It never accesses ``Loop`` internals — no ``__getattr__`` proxy, no
    monkey-patching, no ``_infra`` attribute.

    A second strategy (e.g. a GEPA-like population strategy) must be able to
    use this seam unchanged, proving it is generic.

    The contract is validated at subclass creation time by
    ``__init_subclass__``.
    """

    #: The name this loop is selected by.
    name: str = "default"
    #: Whether it needs a model, so construction does not have to know.
    requires_model: bool = False

    #: Edit operations the strategy's own proposals use. A run that names no
    #: ``allowed_ops`` permits these on top of its edit preset, so the strategy
    #: works without a profile setting; an explicit ``allowed_ops`` still wins.
    required_ops: tuple[str, ...] = ()

    #: Steps a subclass may override. Each has a working default.
    overridable = (
        "configure",
        "strategy", "select_parents", "propose", "batch_size", "install", "accepts",
        "update", "should_stop", "on_epoch_end", "report", "memory",
        "strategy_state_dict", "load_strategy_state",
        "bind_model", "best_candidate", "run_error",
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate that a custom strategy does not proxy Loop internals."""
        super().__init_subclass__(**kwargs)
        # A custom OuterLoop subclass must not define __getattr__ (proxy) or
        # carry a _infra attribute.  Loop itself is exempt because it IS the
        # infrastructure.
        from_outerloop = getattr(cls, '_is_loop_infrastructure', False)
        if from_outerloop:
            return
        if "__getattr__" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} defines __getattr__; a custom strategy must "
                f"not proxy Loop internals — use Run's public API instead"
            )

    def configure(
        self, *, model: Any = None, model_name: str = "", provider: str = "auto", **options: Any
    ) -> None:
        """Take what the run has, rather than the constructor knowing per name."""
        self.model = model
        self.model_name = model_name
        self.provider = provider
        self.options = dict(options)

    def bind_model(self, model: Any) -> None:
        """Receive the run's budget-bound, role-routed model client.

        Called by ``Loop.bind_budget()`` after the budget is attached, replacing
        whatever ``configure()`` stored.  Override when a strategy caches derived
        clients from ``self.model`` during ``configure()``; the default just
        reassigns ``self.model``.
        """
        self.model = model

    def best_candidate(self, population: "list[Candidate]") -> "Candidate | None":
        """The strategy's own best, when it differs from leaderboard rank.

        Return ``None`` to use the leaderboard's first entry (today's behaviour).
        A strategy that gates winners by a criterion the leaderboard does not
        know (e.g. validation score) overrides this.

        ``population`` is the list of candidates the Loop has evaluated.
        """
        return None

    def run_error(self) -> str | None:
        """Why the strategy's run failed, though it stopped normally; ``None`` if it did not.

        ``Run.stop`` only names the bound. A strategy that stops because its search
        cannot go on (e.g. every optimizer call of an epoch failed) returns the
        reason here, and the report says the run failed with it as the error.
        """
        return None

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
