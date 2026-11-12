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

Selection on held-out parts is offered here too, for any strategy:
:class:`ValidationGate` (accept on a strict val improvement, val scores cached per
whole candidate, the best on val) and :class:`FinalEvaluation` (the post-training
val check and test rollouts). The default loop uses the gate with ``val_gate``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import structlog

from refactory_dsh.optim import Candidate, Optimizer, Proposal, Rejection
from refactory_dsh.optim.base import proposal_refusal, summarise_reasons

log = structlog.get_logger()


@dataclass(frozen=True)
class EvalRequest:
    """One evaluation request for :meth:`Run.evaluate_many`."""
    candidate: Candidate
    part: str = "train"
    items: list[str] | None = None
    seed: int | None = None
    #: A train re-score of an already evaluated candidate as fresh evidence for
    #: ``epoch`` (``Loop._rescore``): its records are replaced and kept by epoch, its
    #: score and id stay. It shares the batch's evaluation and rollout reservation,
    #: item gate and stop signal. The default loop's own requests only.
    rescore_epoch: int | None = None


class OuterLoopError(RuntimeError):
    """A search cannot proceed. Always names what is wrong."""


class BudgetStopped(OuterLoopError):
    """A time or token budget refused or cut an evaluation before it finished.

    Nothing is decided from it: a train evaluation leaves its candidate
    unevaluated, and a val rollout (flagged ``budget_stopped`` in its occasion) is
    no val score, so a gate leaves the candidate unjudged. A resume with more
    budget scores it again. The run's bound is the spent budget: ``reason`` holds the
    budget's own reason (e.g. ``elapsed-time budget exhausted``), which a strategy
    stops with (``Run.stop``).
    """

    def __init__(self, message: str, *, reason: str = "") -> None:
        super().__init__(message)
        self.reason = reason or message


def train_batch_seed(seed: int, epoch: int, batch_idx: int = 0) -> int:
    """SkillOpt's train-batch seed: ``seed + epoch*1000 + batch_idx + 1``.

    The seed an epoch's train rollouts use (SkillOpt's ``plan_train_epoch``,
    trainer.py:1170-1177, :1242; dataloader.py:278), so a Task that shuffles per
    (item, seed) shuffles differently each epoch. Val and test rollouts keep the
    run's own ``seed``. Used by SkillOpt and by the default loop with ``val_gate``.
    """
    return seed + epoch * 1000 + batch_idx + 1


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
        """Apply a typed edit -- or a compound proposal's edits, in order, as one
        candidate -- and validate it. Spends no evaluation budget.

        With ``integrity_judge`` on, a proposal that became a valid candidate is also
        judged (``Loop._integrity_check``): one call of the optimizer's model, charged
        to the run's token and time budgets; under ``veto`` an ``exploit`` is refused
        here (``E-integrity``), so every strategy's proposals pass the same judge."""
        # Unknown and unpermitted are two facts with two fixes; see Loop._refinement.
        reason = proposal_refusal(
            proposal, permitted=self._loop.permitted_ops(),
            max_edits=getattr(self._loop.config, "max_edits_per_proposal", None))
        if reason:
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
            protected=getattr(self._loop, "protected", ()),
            graders=getattr(self._loop, "graders", ()),
            allow_new_node_tools=bool(getattr(self._loop.config, "allow_new_node_tools", False)),
        )
        check = getattr(self._loop, "_integrity_check", None)
        if not isinstance(outcome, list) and callable(check):
            outcome = check(proposal, parent, outcome)
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

    @contextmanager
    def final_scoring(self) -> Iterator[None]:
        """Score inside it in the budget's ``final_validation`` phase: the end-of-run
        val and test scoring (``FinalEvaluation``) may spend ``validation_reserve_s``
        and ``validation_reserve_tokens``, which the search's own work leaves alone."""
        loop = self._loop
        before = getattr(loop, "_final_scoring", False)
        loop._final_scoring = True  # noqa: SLF001 - the harness owns the budget phase
        try:
            yield
        finally:
            loop._final_scoring = before  # noqa: SLF001

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
        The rollout check is all-or-nothing too: the batch is refused when the
        rollouts of all its requests together would pass ``max_rollouts``, which
        is stricter than ``evaluate`` (refused only once the budget is spent).
        Budget of a train request that fails or never starts is returned.

        At most one ``part="train"`` request per candidate id: train records
        are kept per candidate (``item_records``), so a second train run of the
        same candidate would overwrite the first before the caller could read
        it.  Such a batch is refused before anything runs or any budget is
        reserved; call ``evaluate`` for each run and read ``item_records``
        between.  Val and test occasions are kept per occasion, so several
        val/test requests for one candidate are allowed (run in order).

        ``on_result(index, candidate)`` fires in completion order, if provided;
        calls never overlap.  Called on the main thread, a stop (SIGTERM) ends
        the call: no request starts after it, the spine runs in flight are shut
        down, a walker evaluation starts no new item, and every evaluation it
        ended raises instead of recording a result (no on_result fires for it,
        and its train budget is returned).
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

        # Everything after the reservation runs under one try/finally, so an
        # error anywhere (the item gate, a stop signal) returns the budget of
        # every train request that did not run, and of every one that failed.
        results: list[Candidate | None] = [None] * len(requests)
        errors: list[BaseException | None] = [None] * len(requests)
        # on_result errors are tracked separately: they must not release
        # the evaluation's budget because the evaluation itself succeeded.
        on_result_errors: list[BaseException | None] = [None] * len(requests)
        counts_lock = threading.Lock()
        # Set when a stop reaches this call: no request starts after it, no
        # on_result fires for one it ended, and a stopped evaluation raises.
        stop_event = threading.Event()
        from refactory_dsh.owned_process import EvaluationStopped, StoppableGate
        # The walker's item gate, wrapped to refuse new items after a stop;
        # set once the gate exists (below).
        stoppable_gate: StoppableGate | None = None
        # on_result calls (and the strategy-state saves they make) never overlap.
        on_result_lock = threading.Lock()
        started_train = 0
        failed_train = 0

        def _run_one(index: int) -> None:
            nonlocal started_train, failed_train
            req = requests[index]
            if stop_event.is_set():
                errors[index] = EvaluationStopped("the request was not started: evaluate_many was stopped")
                return
            if req.part == "train":
                with counts_lock:
                    started_train += 1
            try:
                if req.rescore_epoch is not None:
                    loop._rescore(  # noqa: SLF001
                        req.candidate, req.rescore_epoch, reserved=True,
                        item_gate=stoppable_gate, item_slots=True, stop_event=stop_event)
                elif req.part == "train":
                    loop._evaluate(  # noqa: SLF001
                        req.candidate, seed=req.seed, items=req.items,
                        _budget_reserved=True,
                        item_gate=stoppable_gate,
                        item_slots=True, stop_event=stop_event,
                    )
                else:
                    loop._evaluate_part(  # noqa: SLF001
                        req.candidate, req.part, req.items, req.seed,
                        item_gate=stoppable_gate,
                        item_slots=True, stop_event=stop_event,
                    )
                results[index] = req.candidate
            except BaseException as exc:
                errors[index] = exc
                if req.part == "train":
                    with counts_lock:
                        failed_train += 1
                return

            # on_result is called only after a successful evaluation.
            # An exception here must NOT release the evaluation's budget
            # (it ran and was paid for).  Record it and re-raise after
            # all evaluations settle.
            if on_result is not None:
                try:
                    with on_result_lock:
                        if stop_event.is_set():
                            return
                        on_result(index, req.candidate)
                except BaseException as exc:
                    on_result_errors[index] = exc

        def _run_group(indices: list[int]) -> None:
            """Run a group of requests for the same candidate sequentially.

            A failure skips the group's later requests; a skipped train
            request never starts, so its reserved budget is returned.
            """
            for idx in indices:
                _run_one(idx)
                if errors[idx] is not None:
                    break

        try:
            # ── ensure item gate / slot dir ───────────────────────────────
            seed_candidate = loop.population[0] if loop.population else requests[0].candidate
            loop._init_item_gate(seed_candidate)  # noqa: SLF001
            gate = getattr(loop, '_item_gate', None)
            if gate is not None:
                stoppable_gate = StoppableGate(gate, stop_event)

            # ── group requests by candidate id ────────────────────────────
            # Requests with the same candidate id (at most one train, plus any
            # val/test occasions) run sequentially in request order; distinct
            # candidate ids run concurrently.
            groups: dict[str, list[int]] = defaultdict(list)
            for i, req in enumerate(requests):
                groups[req.candidate.id].append(i)
            group_list = list(groups.values())
            max_workers = getattr(loop.config, 'max_concurrent_evaluations', 4)

            if len(group_list) == 1:
                # Single candidate or single request: run inline
                _run_group(group_list[0])
            else:
                from refactory_dsh.dsh_runtime import shutdown_tracked_runs
                from refactory_dsh.owned_process import _termination_as_exit
                # On the main thread a SIGTERM becomes SystemExit here (pool
                # threads cannot install the handler run_on_dsh uses); spines
                # the pool started are shut down before the pool is joined.
                with _termination_as_exit():
                    pool = ThreadPoolExecutor(max_workers=min(max_workers, len(group_list)))
                    futures = [pool.submit(_run_group, g) for g in group_list]
                    try:
                        for future in as_completed(futures):
                            try:
                                future.result()
                            except BaseException:
                                pass  # errors recorded in errors[]
                    except BaseException:
                        stop_event.set()
                        for future in futures:
                            future.cancel()
                        shutdown_tracked_runs()
                        raise
                    finally:
                        pool.shutdown(wait=True)
        finally:
            # ── un-reserve budget for failed and never-started train evals ──
            unreserve = (train_count - started_train) + failed_train
            if unreserve > 0:
                with loop._state_lock:
                    loop.evaluations -= unreserve

        # ── re-raise the first error ──────────────────────────────────────
        # Evaluation errors take priority over on_result errors, and a real failure
        # over a budget stop, so a crash is never reported as a clean budget stop.
        first_error = next((e for e in errors if e is not None and not isinstance(e, BudgetStopped)),
                           next((e for e in errors if e is not None), None))
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
            {k: v for k, v in occ.items()
             if k in ("part", "occasion", "score", "exit_ok", "error", "duration_s", "budget_stopped")}
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

        A kept candidate is journaled, as the default loop's are, so a resumed run
        holds it again (``candidate`` finds it by id), evaluated or not.
        """
        added = self._loop.step([candidate], run=self)  # noqa: SLF001
        if added:
            self._loop._journal_candidate(candidate)  # noqa: SLF001
        self._loop.update(self, candidate)  # noqa: SLF001
        return candidate

    def candidate(self, candidate_id: str) -> Candidate | None:
        """The candidate the search holds under ``candidate_id``, or ``None``.

        A candidate's id is its whole state's (the graph and every knob value), so
        a strategy saves ids rather than knob values and gets the candidate back
        here after a resume: the seed, an evaluated candidate, or one it kept and
        has not evaluated yet.
        """
        return next((c for c in self._loop.population if c.id == candidate_id), None)

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
        """Evaluations the search may still start: none once the evaluation, rollout,
        elapsed-time or token budget is spent (a refused evaluation counts nothing, so
        a strategy looping on this count must see the time and token budgets here)."""
        budget = getattr(self._loop, "budget", None)
        if budget is not None:
            from refactory_dsh.budgets import BudgetExhausted
            try:
                budget.checkpoint()
            except BudgetExhausted:
                return 0
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
        try:
            self.strategy(run)
        except BudgetStopped as exc:
            # A strategy's evaluation the time or token budget stopped: the search
            # ends on that budget and the strategy's own ending (finish_strategy) runs.
            run.stop(exc.reason)
        self.finish_strategy(run)
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

    def finish_strategy(self, run: Run) -> None:
        """What runs after ``strategy`` and before the report: nothing here. The default
        loop scores the batch its last epoch proposed (``Loop.finish_strategy``), for
        every built-in strategy, so none of them pays for a round it never scores."""

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


# ── selection on held-out parts, for any strategy ─────────────────────────


class RoundStopped(Exception):
    """A rollout refused part way through :meth:`ValidationGate.judge_round`: the
    decisions about the candidates scored before it, and the refusal (``error``)."""

    def __init__(self, accepted: list[Any], rejected: list[Any], error: Exception) -> None:
        super().__init__(str(error))
        self.accepted, self.rejected, self.error = accepted, rejected, error


@dataclass(frozen=True)
class Verdict:
    """What :meth:`ValidationGate.judge` decided about one candidate."""

    accepted: bool
    #: The candidate's val score.
    score: float
    #: Whether the score came from the gate's cache (no rollout).
    cached: bool
    #: The score it had to beat.
    incumbent: float
    #: Whether it became the best on val.
    best: bool


class ValidationGate:
    """Accept a candidate only on a strict val improvement, and keep the best on val.

    The gate holds the **current** candidate (the last one accepted, or one a
    strategy carries its score to) and the **best** one on val, each as an id and a
    score. Val scores are cached per whole candidate: the key is ``Candidate.id``,
    the id of its state (the graph and every knob value), so a candidate is rolled
    out on val once, whichever knob or node an edit changed.

    A Type 2 strategy (SkillOpt) makes one and calls it; the default loop uses one
    when ``LoopConfig.val_gate`` is on. Its state is plain JSON
    (``state_dict``/``load_state_dict``), journaled with the strategy's own.
    Every rollout goes through ``Run.evaluate`` with ``seed``.
    """

    def __init__(self, *, seed: int | None = None) -> None:
        self.seed = seed
        self.cache: dict[str, float] = {}
        self.current_id: str | None = None
        self.current_score: float = 0.0
        self.best_id: str | None = None
        self.best_score: float = 0.0

    @staticmethod
    def val_score(run: Run, candidate: Candidate) -> float:
        """A candidate's val score: its first val rollout's, 0.0 when that failed. A
        rollout a budget stopped (``budget_stopped``) is not a val score and is skipped."""
        scores = [s for s in run.val_scores(candidate) if not s.get("budget_stopped")]
        raw = scores[0].get("score") if scores else None
        return float(raw) if raw is not None else 0.0

    def start(self, run: Run, candidate: Candidate) -> float:
        """Roll the starting candidate out on val, unless its val score is cached
        (``prefetch``); it becomes current and best."""
        if candidate.id in self.cache:
            score = self.cache[candidate.id]
        else:
            run.evaluate(candidate, part="val", seed=self.seed)
            score = self.val_score(run, candidate)
            self.cache[candidate.id] = score
        self.current_id = self.best_id = candidate.id
        self.current_score = self.best_score = score
        return score

    def score(self, run: Run, candidate: Candidate) -> tuple[float, bool]:
        """The candidate's val score and whether it was cached; rolled out once."""
        if candidate.id in self.cache:
            return self.cache[candidate.id], True
        run.evaluate(candidate, part="val", seed=self.seed)
        score = self.val_score(run, candidate)
        self.cache[candidate.id] = score
        return score, False

    def prefetch(self, run: Run, candidates: Sequence[Candidate]) -> None:
        """Roll the candidates not yet cached out on val at once (``Run.evaluate_many``)
        and cache each score, so ``start``, ``score`` and ``judge`` reuse it.

        Fewer than two to roll out, or a batch the harness refuses
        (``OuterLoopError``), rolls nothing: the in-order calls roll them out, and
        refuse, as before. A rollout that finished is cached even when another raised.
        """
        todo = list({c.id: c for c in candidates if c.id not in self.cache}.values())
        if len(todo) < 2:
            return

        def cache(_index: int, candidate: Candidate) -> None:
            self.cache[candidate.id] = self.val_score(run, candidate)

        try:
            run.evaluate_many([EvalRequest(candidate=c, part="val", seed=self.seed) for c in todo],
                              on_result=cache)
        except OuterLoopError:
            return

    def judge(self, run: Run, candidate: Candidate, *, against: Candidate | None = None) -> Verdict:
        """Score ``candidate`` on val and accept it only if it beats the incumbent.

        The incumbent is the current candidate, or ``against`` when given (a
        population search compares a child with its parent). An equal score is
        rejected. An accepted candidate becomes current, and best when it beats
        the best. Keeping it in the search stays the strategy's call.
        """
        incumbent = self.current_score if against is None else self.score(run, against)[0]
        score, cached = self.score(run, candidate)
        accepted = score > incumbent
        best = False
        if accepted:
            self.current_id, self.current_score = candidate.id, score
            best = self.offer_best(candidate.id, score)
        return Verdict(accepted=accepted, score=score, cached=cached, incumbent=incumbent, best=best)

    def judge_round(self, run: Run, candidates: Sequence[Candidate]) -> tuple[list[Candidate], list[Candidate]]:
        """Judge one round's candidates together: ``(accepted, rejected)``, in order.

        Each is scored on val against the same bar, the current candidate's score as
        the round starts; the order they were proposed in cannot raise it. Of those that
        strictly beat it, the one with the highest val score is accepted and becomes
        current (and best, when it beats the best); ties go to the higher train score
        (``Candidate.score``; a failed train run counts lowest), then to the earlier.
        The others are rejected, a candidate that beat the bar but lost to a sibling
        included. A rollout that raises (``OuterLoopError``) stops the round: what was
        scored before it is judged, then the error propagates, and the rest stay
        unjudged.
        """
        incumbent = self.current_score
        scored: list[tuple[Candidate, float]] = []
        error: OuterLoopError | None = None
        for candidate in candidates:
            try:
                scored.append((candidate, self.score(run, candidate)[0]))
            except OuterLoopError as exc:
                error = exc
                break
        better = [(index, candidate, score) for index, (candidate, score) in enumerate(scored)
                  if score > incumbent]
        winner = None
        if better:
            def train(candidate: Candidate) -> float:
                return float(candidate.score) if candidate.score is not None else float("-inf")

            _index, winner, score = max(
                better, key=lambda entry: (entry[2], train(entry[1]), -entry[0]))
            self.current_id, self.current_score = winner.id, score
            self.offer_best(winner.id, score)
        accepted = [winner] if winner is not None else []
        rejected = [candidate for candidate, _score in scored if candidate is not winner]
        if error is not None:
            raise RoundStopped(accepted, rejected, error)
        return accepted, rejected

    def carry(self, candidate: Candidate, *, remember: bool = False) -> None:
        """Make ``candidate`` current at the current score, without a rollout.

        For a candidate a strategy derives from the current one and accepts by its
        own rule (SkillOpt's slow update). ``remember`` caches that score for it.
        """
        self.current_id = candidate.id
        if remember:
            self.cache[candidate.id] = self.current_score

    def offer_best(self, candidate_id: str, score: float) -> bool:
        """Make the candidate the best on val if ``score`` beats the best's."""
        if score > self.best_score:
            self.best_id, self.best_score = candidate_id, score
            return True
        return False

    def best_in(self, population: Sequence[Candidate]) -> Candidate | None:
        """The best candidate, looked up by id; ``None`` before any val score.

        Raises ``ValueError`` when the population does not hold it, so a report
        never names another winner in its place.
        """
        if self.best_id is None:
            return None
        for candidate in population:
            if candidate.id == self.best_id:
                return candidate
        raise ValueError(
            f"the val-best candidate {self.best_id!r} is not in the population of "
            f"{len(population)} candidates"
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "val_cache": dict(self.cache),
            "current_candidate_id": self.current_id,
            "current_score": self.current_score,
            "best_candidate_id": self.best_id,
            "best_score": self.best_score,
        }

    def load_state_dict(self, data: dict[str, Any]) -> None:
        self.cache = {str(k): float(v) for k, v in (data.get("val_cache") or {}).items()}
        self.current_id = data.get("current_candidate_id")
        self.current_score = float(data.get("current_score", 0.0))
        self.best_id = data.get("best_candidate_id")
        self.best_score = float(data.get("best_score", 0.0))


class FinalEvaluation:
    """After a search: the final candidate's val check, then three test rollouts.

    SkillOpt's post-training block (trainer.py:2241-2441), for any strategy with a
    :class:`ValidationGate`:

    1. ``final_selection``: the gate's current (final) candidate on val -- the best's
       score when it is the best, else a fresh val rollout; a score above the best's
       makes it the best.
    2. ``baseline_test``, ``best_test``, ``final_test``: the seed, the best and the
       final candidate on test, concurrently through ``Run.evaluate_many``; the best
       reuses the seed's score when it is the seed, and the final the best's when it
       is the best, or the seed's when it is the seed.

    Every rollout uses ``seed``. Each finished phase is kept (``completed``) and
    ``save`` is called, so a resumed run continues where it stopped; the decisions go
    to ``record(step_type, **fields)`` in that order, never ahead of an unfinished
    earlier phase. ``result`` holds the scores by phase name, ``best_test_delta`` and
    ``final_test_delta`` (each against the baseline), ``status`` (``running``,
    ``done``, ``skipped``, ``incomplete``) and ``reason``. A rollout the harness
    refuses (``OuterLoopError``, e.g. a spent rollout budget) ends it ``incomplete``.
    """

    PHASES = ("final_selection", "baseline_test", "best_test", "final_test", "final_done")
    TEST_PHASES = ("baseline_test", "best_test", "final_test")

    def __init__(self, *, record: Callable[..., Any] | None = None, label: str = "outer loop") -> None:
        self.record = record or (lambda step_type, **fields: None)
        self.label = label
        self.result: dict[str, Any] = {}
        self.completed: set[str] = set()
        self.traced: set[str] = set()

    @property
    def phase(self) -> str:
        """The furthest finished phase, in order; empty before any."""
        return max(self.completed, key=self.PHASES.index) if self.completed else ""

    def skip(self, run: Run, reason: str, save: Callable[[], None]) -> None:
        """Record that the evaluation does not run, and why."""
        self.result = {"status": "skipped", "reason": reason}
        self.record("test_eval", action="skipped", reason=reason)
        run.log(f"{self.label}: no post-training test evaluation: {reason}")
        save()

    def run(self, run: Run, gate: ValidationGate, *, seed: int | None, save: Callable[[], None]) -> None:
        """Run the pending phases; see the class docstring. They are the run's final
        scoring, so they may spend the budget's validation reserve (``Run.final_scoring``)."""
        with run.final_scoring():
            self._run(run, gate, seed=seed, save=save)

    def _run(self, run: Run, gate: ValidationGate, *, seed: int | None, save: Callable[[], None]) -> None:
        if not self.completed:
            self.result, self.traced = {}, set()
        result = self.result
        result["status"] = "running"

        def finished(phase: str) -> None:
            self.completed.add(phase)
            self._sync_test_scores()
            save()

        try:
            if "final_selection" not in self.completed:
                if gate.current_id == gate.best_id:
                    result["final_selection"] = gate.best_score
                    self.record("final_selection_eval", action="reused", score=gate.best_score)
                else:
                    final = self._held(run, gate.current_id)
                    run.evaluate(final, part="val", seed=seed)
                    score = self._latest_score(run, final, "val")
                    result["final_selection"] = score
                    promoted = score is not None and gate.offer_best(final.id, score)
                    self.record(
                        "final_selection_eval", action="promoted" if promoted else "evaluated",
                        score=score, best_score=gate.best_score,
                    )
                finished("final_selection")

            # (phase, candidate) for each pending test phase; None reuses the score of
            # the earlier phase that rolled the same candidate out (``reuses``): a best
            # or final that is the seed takes the baseline's, a final that is the best
            # the best's. So a seed that wins is scored on test once and its deltas are
            # 0, not the noise between two rollouts of the same candidate.
            plan: list[tuple[str, Candidate | None]] = []
            seed_id = run.seed().id
            reuses: dict[str, str] = {}
            if gate.best_id == seed_id:
                reuses["best_test"] = "baseline_test"
            if gate.current_id == gate.best_id:
                reuses["final_test"] = "best_test"
            elif gate.current_id == seed_id:
                reuses["final_test"] = "baseline_test"
            held = {"baseline_test": seed_id, "best_test": gate.best_id,
                    "final_test": gate.current_id}
            for phase in self.TEST_PHASES:
                if phase not in self.completed:
                    plan.append((phase, None if phase in reuses
                                 else self._held(run, held[phase])))
            requests = [EvalRequest(candidate=cand, part="test", seed=seed)
                        for _phase, cand in plan if cand is not None]
            request_phases = [phase for phase, cand in plan if cand is not None]
            phases = result.setdefault("test_phases", {})
            try:
                if requests:
                    def _on_result(index: int, candidate: Candidate) -> None:
                        phases[request_phases[index]] = {
                            "score": self._latest_score(run, candidate, "test"),
                            "candidate": candidate.id,
                        }
                        finished(request_phases[index])

                    run.evaluate_many(requests, on_result=_on_result)
                for phase, cand in plan:
                    if cand is None:  # the same candidate as an earlier phase: reuse it
                        source = reuses[phase]
                        earlier = phases.get(source) or {}
                        phases[phase] = {"score": earlier.get("score", result.get(source)),
                                         "reused": True, "from": source}
                        finished(phase)
            finally:
                self._trace_test_phases(save)
        except OuterLoopError as error:
            result["status"] = "incomplete"
            result["reason"] = str(error)
            self.record("test_eval", action="incomplete", reason=str(error))
            run.log(f"{self.label}: post-training evaluation stopped: {error}")
            save()
            return
        for delta, phase in (("best_test_delta", "best_test"), ("final_test_delta", "final_test")):
            result[delta] = (
                result[phase] - (result.get("baseline_test") or 0)
                if result.get(phase) is not None else None
            )
        result["status"] = "done"
        finished("final_done")

    @staticmethod
    def _held(run: Run, candidate_id: str | None) -> Candidate:
        candidate = run.candidate(candidate_id) if candidate_id else None
        if candidate is None:
            raise OuterLoopError(f"the search does not hold candidate {candidate_id!r}")
        return candidate

    @staticmethod
    def _latest_score(run: Run, candidate: Candidate, part: str) -> float | None:
        """The candidate's score on its latest ``part`` rollout; ``None`` if it failed."""
        scores = run.part_scores(candidate, part)
        raw = scores[-1].get("score") if scores else None
        return float(raw) if raw is not None else None

    def _sync_test_scores(self) -> None:
        """Write the finished test phases' scores into ``result`` in phase order,
        whatever order they finished in."""
        result = self.result
        phases = result.get("test_phases") or {}
        held = {phase: result.pop(phase) for phase in self.TEST_PHASES if phase in result}
        for phase in self.TEST_PHASES:
            if phase in phases:
                result[phase] = phases[phase].get("score")
            elif phase in held:
                result[phase] = held[phase]

    def _trace_test_phases(self, save: Callable[[], None]) -> None:
        """Record the finished test phases not yet recorded, in phase order,
        stopping at the first unfinished one."""
        phases = self.result.get("test_phases") or {}
        wrote = False
        for phase in self.TEST_PHASES:
            if phase in self.traced:
                continue
            if phase not in self.completed or phase not in phases:
                break
            entry = phases[phase]
            if entry.get("reused"):
                # ``final_test_reused`` as before; the best (the seed) reusing the
                # baseline is ``best_test_reused``.
                self.record("test_eval", action=f"{phase}_reused", score=entry.get("score"))
            else:
                self.record("test_eval", action=phase, score=entry.get("score"),
                            candidate=entry.get("candidate", ""))
            self.traced.add(phase)
            wrote = True
        if wrote:
            save()

    def state_dict(self) -> dict[str, Any]:
        return {
            "final_eval": dict(self.result),
            "completed_phases": sorted(self.completed),
            "traced_phases": sorted(self.traced),
        }

    def load_state_dict(self, data: dict[str, Any]) -> None:
        self.result = dict(data.get("final_eval") or {})
        self.completed = set(data.get("completed_phases") or ())
        self.traced = set(data.get("traced_phases") or ())


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
