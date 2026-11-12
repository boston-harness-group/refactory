"""The optimizer loop, named like one.

    init      - build the starting population from the mode's seed state
    forward   - run each candidate in the batch
    loss      - turn the batch into a number
    backward  - read the traces and compute the gradient
    optimizer - propose the next update, validating before anything is run
    step      - apply it, producing offspring
    scheduler - decide whether the epoch is over, and say what bound it
    report    - the visibility surface

A candidate is a state dict plus its lineage. Evaluating one is a forward pass:
materialize the mode from the state, execute its graph, and score the run. An
invalid candidate never reaches ``forward`` at all, so it costs no budget — it is
refused by the optimizer's loop with a reason, and that reason is reported.

Two things shape what a candidate is worth. It is evaluated on the part of its
Task's items the run directory names, never on all of them, so the search cannot be
examined on the items it was tuned against; and what it scores is the evaluator's
number minus ``parsimony`` per node of its graph, so a search with no size cost does
not grow the harness for free. Both numbers are recorded, and once the search is
over the winner is re-evaluated on the items no candidate saw.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import shutil
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass
from contextlib import nullcontext
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Sequence

from refactory_dsh import history as run_history
from refactory_dsh import events
from refactory_dsh import observe as progress_file
from refactory_dsh import task as task_module
from refactory_dsh.evaluators import create as create_evaluator
from refactory_dsh import registry
from refactory_dsh.nn import Knob, PackageError
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn import validate_state
from refactory_dsh.optim import (
    Candidate,
    Optimizer,
    Proposal,
    ProposalContext,
    Rejection,
    RandomOptimizer,
)
from refactory_dsh.optim import create as create_optimizer
from refactory_dsh.optim.base import (
    DEFAULT_MAX_EDITS, available_before, edit_level, proposal_refusal, summarise_reasons,
)
from refactory_dsh.outerloop import (
    BudgetStopped, EvalRequest, FinalEvaluation, OuterLoop, OuterLoopError, Run, ValidationGate, train_batch_seed,
)
from refactory_dsh.paths import project_root

#: The append-only record of everything a run did. Resume replays it.
JOURNAL = run_history.JOURNAL


#: The offspring a proposal node found, waiting for the node that installs them.
PENDING = "pending.json"

#: The train-minus-holdout gap above which the audit calls a winner overfit. The
#: reference's number, so a run here and a run there agree about what overfit means.
OVERFIT_THRESHOLD = 0.15
#: Why a seed winner is never called overfit.
SEED_WINNER_NOTE = ("the winner is the seed, which the search did not change, so the gap "
                    "is the difference between the parts, not overfitting")


def gap_standard_error(first: float, first_items: int, second: float,
                       second_items: int) -> float | None:
    """The standard error of a gap between two scores, for information only.

    Each score is read as a pass rate over its part (clipped to [0, 1], the largest
    variance a score in that range can have): SE = sqrt(p1(1-p1)/n1 + p2(1-p2)/n2).
    It is reported beside the fixed ``OVERFIT_THRESHOLD`` so a reader can see how much
    of a gap small parts make by chance; it does not move the threshold. ``None``
    without both sizes.
    """
    if not first_items or not second_items:
        return None

    def variance(score: float, items: int) -> float:
        p = min(1.0, max(0.0, float(score)))
        return p * (1.0 - p) / items

    return round(math.sqrt(variance(first, first_items) + variance(second, second_items)), 6)

#: Where the run writes how the winner did on the items no candidate saw.
HOLDOUT_FILE = "holdout.json"
#: The optimizer's model calls (system, prompt, reply, parsed proposals, error).
OPTIMIZER_TRACE = "optimizer_trace.jsonl"


class LoopError(RuntimeError):
    """The run cannot proceed. Always names what is wrong with the mode."""



def _item_sources(state: Mapping[str, Any]) -> dict[str, tuple[Any, ...]]:
    """Which items each data node of a state would read.

    The fields a harness's class of work lives in: the Task it binds, the source it
    reads, the limit it applies, and whether it declares item_scores. Nothing else in
    the graph says what the run is examined on.
    """
    graph = state.get("graph") or {}
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    if isinstance(nodes, dict):
        entries = list(nodes.values())
    elif isinstance(nodes, list):
        entries = nodes
    else:
        entries = []
    found: dict[str, tuple[Any, ...]] = {}
    for node in entries:
        if not isinstance(node, dict) or node.get("kind") != "data":
            continue
        found[str(node.get("id"))] = (
            node.get("task_ref"), node.get("source_path"), node.get("source_format"),
            node.get("limit", 0),
            "item_scores" in (node.get("produces") or []),
        )
    return found


#: Optimizers and parent pickers that breed from a population of parents (an archive,
#: a tournament, islands, niche champions). Under ``val_gate`` they default to the
#: ``pool`` selection (``default_selection``).
POPULATION_OPTIMIZERS = ("archive", "evolutionary", "islands", "map_elites")

#: ``LoopConfig.selection`` values under ``val_gate``: empty picks the default
#: (``default_selection``); ``single`` is SkillOpt's line; ``pool`` keeps the best
#: candidates on validation as parents.
SELECTIONS = ("", "single", "pool")


#: What a run without validation items says, in its log and its report, when it
#: falls back to selecting on train scores (``LoopConfig.val_gate`` left unset).
NO_VALIDATION = "no validation items: selecting on train scores"

#: ``LoopConfig.portfolio`` values: off, or one candidate per edit level.
PORTFOLIOS = ("", "prompt+pipeline")


def default_selection(strategy: str, optimizer: str, picker: str = "") -> str:
    """The ``selection`` a ``val_gate`` run gets when it names none: ``single`` for the
    default loop with a single-parent optimizer (the SkillOpt-matched setting, as before
    pools existed), ``pool`` for every other strategy, optimizer or parent picker."""
    if (strategy == "default" and optimizer not in POPULATION_OPTIMIZERS
            and picker not in POPULATION_OPTIMIZERS):
        return "single"
    return "pool"


def _train_key(candidate: Candidate) -> float:
    """A candidate's train score for breaking a val tie; a failed train run counts lowest."""
    return float(candidate.score) if candidate.score is not None else float("-inf")


@dataclass
class LoopConfig:
    """How a run is shaped."""

    epochs: int = 2
    batch_size: int = 4
    optimizer: str = "autoresearch"
    edit_preset: str = "broad"
    seed: int = 0
    elite: int = 3
    #: A hard ceiling on inner runs. ``epochs`` normally binds first; whichever
    #: does is reported, so a run never ends for an unstated reason.
    max_evaluations: int = 64
    #: The fraction of a Task's instances held out of the search and used to audit
    #: the winner. One instance cannot be split, which the audit records as a fact
    #: instead of failing the run.
    holdout: float = task_module.DEFAULT_HOLDOUT
    #: What each node of a candidate's graph costs its score. The reference's
    #: coefficient: without a size cost a search has no reason not to grow the
    #: harness, and the report would not say whether a candidate won on merit or on
    #: size.
    parsimony: float = 0.01
    runs_root: Path = Path(".factory/loop")
    #: Wall-clock ceiling for one inner run, enforced by the executor's step cap.
    max_steps: int = 2000
    #: Attempts the optimizer gets per batch before it stops asking.
    attempts: int = 3
    #: Knobs the operator freezes for this run, whatever the candidate says.
    frozen: tuple[str, ...] = ()
    #: What this run is *for*, as the person asking put it. Seeded into the run's
    #: intent ledger, which every agent node in an interactive run reads. Empty means
    #: nobody said, and the ledger is left alone rather than filled with a default --
    #: that default is BUG-05, and a ledger holding one is a ledger a fidelity check
    #: cannot check. See :mod:`refactory_dsh.steering`.
    intent: str = ""
    #: Knob values the search starts from, as `NAME=VALUE`. A mode whose inputs are
    #: frozen knobs (`source_root`, `test_command`) cannot be searched at all without
    #: this: every candidate would run with the declared defaults, score the same
    #: thing, and the run would report a search that found nothing.
    knobs: dict[str, Any] = dataclasses.field(default_factory=dict)
    #: How many evaluated candidates the optimizer is shown. Bounded on purpose:
    #: an unbounded history in a prompt is how a search stops thinking.
    history_limit: int = 200
    #: How many previous runs to summarise as a prior.
    prior_runs: int = 3
    #: Where a ``global``-retention memory lives. Injected so a test can put it
    #: somewhere temporary instead of the home directory.
    memory_global_root: Path | None = None
    #: Which runtime runs a candidate: "walker" (fast, for search) or "dsh" (the
    #: runtime that ships). Both score identically; the equivalence tests keep it so.
    evaluator: str = "walker"
    #: Exclude the run steering ledger from candidate model prompts; controls remain binding.
    omit_steering: bool = False
    #: Wall-clock ceiling for one inner run when the evaluator is DSH.
    dsh_timeout_s: int = 900
    #: Carry on from the journal in the run directory instead of starting over.
    #: A run reproduces what an uninterrupted one would have done exactly when its
    #: optimizer is deterministic, which is stated rather than assumed.
    resume: bool = False
    #: Write ``progress.jsonl`` as the run goes. Off for a run nobody is watching:
    #: the file is written per candidate, not per epoch, so it is the one artifact
    #: whose cost scales with the batch.
    progress: bool = True
    #: Print the run's events while it goes. Off for a library caller (a test, an
    #: embedded loop): the command line turns it on, because a search that says
    #: nothing until it finishes reads as a hang.
    print_events: bool = False
    #: Print every event, not only the meaningful ones, and pass a candidate's own
    #: run output through. A caller debugging a search wants the firehose.
    verbose: bool = False
    #: How often a paused run re-reads its control file, in seconds. Configuration
    #: only so a test does not have to wait a real second for each generation.
    control_poll_s: float = 1.0
    strategy: str = "default"
    repeats: int = 1
    allowed_ops: tuple[str, ...] = ()
    worker_backend: str = "dsh"
    provider: str = ""
    model: str = ""
    effort: str = ""
    permission_mode: str = "skip"
    #: Claude Code's settings sources for every worker (``claude --setting-sources``):
    #: ``none`` or a comma list of user, project and local. Empty keeps the adapter's
    #: default. claude_code only; ``node_routes``/``role_routes`` override it per node.
    setting_sources: str = ""
    role_routes: dict[str, Any] = dataclasses.field(default_factory=dict)
    node_routes: dict[str, Any] = dataclasses.field(default_factory=dict)
    provider_config: dict[str, Any] = dataclasses.field(default_factory=dict)
    max_elapsed_s: float | None = None
    max_tokens: int | None = None
    validation_reserve_s: float = 0
    validation_reserve_tokens: int = 0
    max_rollouts: int | None = None
    #: How many evaluations run at once: in evaluate_many, and in the default loop's
    #: rounds (a round's train evaluations, then its val-gate rollouts). 1 runs them
    #: one at a time. Default 4.
    max_concurrent_evaluations: int = 4
    #: Run-wide cap on items evaluated concurrently across all concurrent
    #: evaluations.  ``None`` means use the mode's data_parallelism.
    max_items_in_flight: int | None = None
    #: Path to a parts manifest (parts.json) with train/val/test id lists.
    #: When set, the loop copies it to the run root and every evaluation draws
    #: items from it.  The optimizer cannot choose val/test items.
    parts_manifest: str | None = None
    #: Path to a seed-results file (``python -m refactory_dsh.seed_cache export``).
    #: The seed's first evaluation on each part it covers is read from it instead
    #: of run, when the run's score context matches. See :mod:`refactory_dsh.seed_cache`.
    seed_results: str | None = None
    #: The default loop's validation gate (``outerloop.ValidationGate``), with
    #: SkillOpt's evaluation protocol: train rollouts use SkillOpt's per-epoch seed
    #: (``outerloop.train_batch_seed``); every candidate evaluated on train, one that
    #: failed there included, is rolled out on the parts manifest's val part (with
    #: ``seed``), and its val score is the fitness every choice reads (``selection``):
    #: the parents, the pool and the winner; the last epoch's proposals are scored
    #: before the run ends; and with a test part ``outerloop.FinalEvaluation`` runs at
    #: the end. The winner is the incumbent: the best on val, replaced only by a
    #: strictly higher val score. Needs a manifest with a val part; a custom strategy
    #: (SkillOpt) keeps its own gate instead.
    #:
    #: ``None`` (the default) turns it on when the parts manifest has val items and the
    #: strategy is a built-in one, and off otherwise: a run with no validation items
    #: selects on train scores and says so (``NO_VALIDATION``). ``True`` or ``False``
    #: is honoured as given (``--val-gate 1``/``0``). ``Loop.init`` replaces ``None``
    #: with the decision before anything runs, so ``config.json`` records it.
    val_gate: bool | None = None
    #: Train items per epoch: 0 (the default) scores every candidate on the whole train
    #: part. N > 0, when the train part has more than N items, samples N train ids each
    #: epoch with SkillOpt's per-epoch seed (``train_batch_seed``), scores every
    #: candidate of that epoch on that sample, and re-scores the edited candidate on it
    #: when its records are from another epoch, so the proposal evidence is that
    #: epoch's minibatch (as SkillOpt rolls out its current candidate on each step's
    #: batch). Val and test are unaffected. Each epoch's ids are journaled.
    train_minibatch: int = 0
    #: With ``val_gate``: at the start of each epoch's proposal step, re-score the
    #: candidate the optimizer edits on that epoch's train sample (the whole train part,
    #: or the ``train_minibatch`` sample) with that epoch's seed, unless its records
    #: already come from this epoch (it was just created and scored), so the optimizer
    #: and its readers work from fresh trajectories every epoch, as SkillOpt rolls out
    #: its current candidate every epoch. The re-score replaces only the evidence
    #: records: the candidate's id, train score and gate val score stay. 0 turns it
    #: off (the minibatch re-score of ``train_minibatch`` then stays on). No effect
    #: without ``val_gate``.
    fresh_evidence: bool = True
    #: The most edits one proposal may bundle (``Proposal.edits``): applied together,
    #: in order, as one candidate the gate judges once. 1 allows single edits only.
    max_edits_per_proposal: int = DEFAULT_MAX_EDITS
    #: Whether an agent node an edit adds may run with tools. Off by default: a new
    #: agent node runs with ``tool_policy: none``, and one that asks for tools is
    #: refused (``E-new-node-tools``).
    allow_new_node_tools: bool = False
    #: Whether an edit may add a code node: an op or a gate whose Python the optimizer
    #: writes, run sandboxed (`refactory_dsh.snippet`). On by default; it takes effect
    #: only where the run permits structural edits, and a machine that cannot sandbox
    #: the code refuses code nodes either way (``E-code-nodes``).
    allow_code_nodes: bool = True
    #: With ``val_gate``: stop after this many epochs in a row whose gating did not
    #: raise the best val score (SkillOpt's ``early_stop_patience``). 0 is off.
    early_stop_patience: int = 0
    #: With ``val_gate``: stop once the best val score reaches 1.0 (SkillOpt's
    #: ``early_stop_at_full_val``).
    early_stop_at_full_val: bool = False
    resolved_profile: dict[str, Any] = dataclasses.field(default_factory=dict)
    #: Strategy-specific options passed to the strategy's ``configure()``
    #: method as keyword arguments.  Set from ``--strategy-option`` on the
    #: CLI or directly from Python.
    strategy_options: dict[str, Any] = dataclasses.field(default_factory=dict)
    #: A parent policy (``archive``, ``evolutionary``, ``islands``, ``hill_climb``,
    #: ``map_elites``) that picks the candidate each step edits, while ``optimizer``
    #: writes the edits (``optim.compose.PickedParent``). Empty (the default): the
    #: optimizer alone, as before.
    parent_picker: str = ""
    #: ``prompt+pipeline``: each proposal round asks the model proposer for one
    #: candidate per edit level -- its best prompt/knob-level change and its best
    #: pipeline-level change -- and evaluates both, so the gate, not the model's
    #: confidence, chooses between them. A strategy that decides one level for a
    #: round (``strategy_decision["edit_kind"]``) gets that one only. A portfolio
    #: round's batch is one candidate per level asked (it replaces ``batch_size``);
    #: the report counts the evaluations beyond ``batch_size``. Empty (the default):
    #: off.
    portfolio: str = ""
    #: With ``val_gate``, how the validation average selects (``SELECTIONS``):
    #:
    #: - ``single``: SkillOpt's line. A candidate is kept only if its val score strictly
    #:   beats the incumbent's (the gate's current candidate), and the incumbent (with
    #:   any candidate a person kept) is the only parent.
    #: - ``pool``: every judged candidate may enter a pool of the ``pool_size`` best by
    #:   val score (ties at the edge kept). Parents come from the pool, each strategy
    #:   and parent picker choosing among them by val. The incumbent still changes only
    #:   on a strictly higher val score (a tie goes to the higher train score, then the
    #:   earlier candidate), and it is the winner.
    #:
    #: Empty (the default) picks ``default_selection``: ``single`` for the default loop
    #: with a single-parent optimizer, ``pool`` for every other strategy and for the
    #: population optimizers and pickers. No effect without ``val_gate``.
    selection: str = ""
    #: The pool's size under ``selection`` ``pool``: the candidates with the best val
    #: scores, plus any tied with the last of them.
    pool_size: int = 8

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["runs_root"] = str(self.runs_root)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoopConfig":
        """Rebuild a configuration from one a run wrote.

        Json turned the tuple fields into lists and the path into a string. Both
        are restored here rather than at each reader, so one place knows what the
        on-disk form of a config is.
        """
        known = {field.name for field in dataclasses.fields(cls)}
        restored: dict[str, Any] = {}
        for key, value in data.items():
            if key not in known:
                continue
            if key == "runs_root":
                restored[key] = Path(value)
            elif key in ("frozen", "allowed_ops"):
                restored[key] = tuple(value or ())
            elif key == "memory_global_root":
                restored[key] = Path(value) if value else None
            else:
                restored[key] = value
        return cls(**restored)


@dataclass
class EpochSummary:
    """One epoch, as the report and the scheduler describe it.

    Kept as a record rather than recomputed from the population: what an epoch
    did is a fact about that epoch, and the population at the end of a run can no
    longer answer how many candidates a middle epoch proposed.
    """

    epoch: int
    evaluated: int
    best: float | None
    mean: float | None
    improved: bool
    #: This epoch's proposals, every round of them: each was accepted or refused, so
    #: ``proposed == accepted + refused``, as the progress line says (dm-265).
    proposed: int
    #: This epoch's proposals accepted for evaluation.
    accepted: int
    #: This epoch's proposals refused.
    refused: int
    #: The candidates the run holds once the epoch is done, every epoch's together.
    population: int
    #: Refusals over the whole run so far.
    refused_total: int
    #: The structural candidates among `population`.
    structural_in_population: int


class Loop(OuterLoop):
    """The default outer loop: the eight steps over one mode.

    Every step a different strategy would change is a method here, so a strategy
    subclasses this and overrides one of them rather than copying the loop.

    When ``_custom_strategy`` is set (a pure ``OuterLoop`` subclass), the
    Loop delegates ``strategy``, ``strategy_state_dict``, and
    ``load_strategy_state`` to that object — the **custom strategy seam**.
    The strategy sees only ``Run``'s public API and never the Loop's internals.
    """

    name = "default"
    _is_loop_infrastructure = True
    #: The strategy options this strategy reads (``strategies.check_options``): the
    #: default loop reads none. Only a class's own declaration counts, so a subclass
    #: that declares nothing is not checked.
    accepted_options: tuple[str, ...] | None = ()

    def bind_budget(self, budget: Any) -> None:
        """Share accounting even when a deterministic target has no model client."""
        self.budget = budget
        if budget is None:
            return
        if self.model is not None:
            self.model = budget.bind_model(self.model)
            self.evaluator.model = self.model
            proposer = self.model.for_node("optimizer", role="optimizer") if callable(getattr(self.model, "for_node", None)) else self.model
            self.optimizer.bind_model(proposer, workdir=self.root / "proposer")
            if self._custom_strategy is not None:
                self._custom_strategy.bind_model(proposer)
        self.evaluator.budget = budget

    def run(self) -> dict[str, Any]:
        """Own strategy execution; a report-only replay does not start a search."""
        from refactory_dsh.control import running_search
        from refactory_dsh.budgets import BudgetExhausted, RunBudget, ResourceLimits, LEDGER

        ledger_path = self.root / LEDGER
        inherited_budget = self.budget or RunBudget.inherited(self.model)
        self._owns_resource_budget = inherited_budget is None
        if self._owns_resource_budget and ledger_path.exists() and not self.config.resume:
            raise LoopError("this run already has a resource ledger; use --resume or choose a new run directory")
        with running_search(self.root):
            limits = ResourceLimits.from_dict(self.config.to_dict())
            self.budget = inherited_budget or (RunBudget.from_path(ledger_path) if ledger_path.exists() else RunBudget.open(self.root, limits))
            if self.config.resume and self._owns_resource_budget:
                try:
                    self.budget.resume(limits)
                except BudgetExhausted as error:
                    self.budget.pause()
                    raise LoopError(str(error)) from error
            if self._owns_resource_budget:
                self.budget.set_validation_owner("search")
            self.bind_budget(self.budget)
            self._executing_strategy = True
            try:
                try:
                    result = super().run()
                except (BudgetExhausted, BudgetStopped) as exc:
                    # BudgetStopped: a strategy's evaluation the budget stopped
                    # (``Run.evaluate``); the bound is the budget _measure recorded.
                    self.bound = self.bound or str(exc)
                    self._journal("bound", bound=self.bound)
                    result = self.report()
                # A same-bound resume stops before the scheduler. Persist that
                # final reason too, so a later report-only replay retains it.
                self._journal("bound", bound=self.bound)
                return result
            finally:
                if self._owns_resource_budget:
                    self.budget.pause()
                self._executing_strategy = False

    def __init__(
        self,
        mode_id: str,
        config: LoopConfig | None = None,
        *,
        project: Path | None = None,
        model: Any = None,
        optimizer: Optimizer | None = None,
    ) -> None:
        self.mode_id = mode_id
        self.config = config or LoopConfig()
        self.project = Path(project or Path("."))
        self.root = Path(self.config.runs_root)
        self.model = model
        # A library caller may provide a configured client directly. Its actual
        # requested routes must agree with the saved effective settings, just as
        # clients constructed by the CLI do.
        if callable(getattr(model, "context", None)):
            client_context = model.context()
            for key in ("worker_backend", "provider", "model", "effort", "permission_mode", "setting_sources", "role_routes", "node_routes", "provider_config"):
                if key in client_context:
                    setattr(self.config, key, client_context[key])
        if not isinstance(self.config.repeats, int) or not 1 <= self.config.repeats <= 10:
            raise LoopError("repeats must be an integer from 1 to 10")
        minibatch = self.config.train_minibatch
        if isinstance(minibatch, bool) or not isinstance(minibatch, int) or minibatch < 0:
            raise LoopError("train_minibatch must be a nonnegative integer (0 is the whole train part)")
        max_edits = self.config.max_edits_per_proposal
        if isinstance(max_edits, bool) or not isinstance(max_edits, int) or max_edits < 1:
            raise LoopError("max_edits_per_proposal must be a positive integer (1 is single edits)")
        self.measurements = 0
        self.rollouts = 0
        self.holdout_measurements: dict[str, Any] = {}
        self.budget: Any = None
        self._owns_resource_budget = False
        #: Set while the end-of-run scoring runs (``Run.final_scoring``): its
        #: measurements are in the ``final_validation`` phase and may spend the reserve.
        self._final_scoring = False
        # One printer for the run: a caller that asked for events sees them while
        # the search goes, and a library caller prints nothing at all.
        self._printer = events.Printer(verbose=self.config.verbose) if self.config.print_events else None

        try:
            package, self.mode_class = registry.resolve(self.project, mode_id)
        except PackageError as exc:
            # A mode that cannot be resolved is a loop that cannot be set up, so
            # it is reported as one rather than as a registry error.
            raise LoopError(str(exc)) from exc
        self.package = package
        self.mode = self.mode_class()
        self.invariants = self.mode.invariants()
        #: Nodes no edit may remove, rewire or replace (`Mode.protected_nodes`).
        self.protected = tuple(self.mode.edit_protected())
        #: Nodes the mode declares as grading (`Mode.grader_nodes`): V17's grader side.
        self.graders = tuple(self.mode.grader_nodes())
        self.parameters = self.mode.parameters()

        if optimizer is not None:
            self.optimizer: Optimizer = optimizer
        elif self.config.optimizer == "autoresearch":
            self.optimizer = create_optimizer(
                self.config.optimizer, client=model, k=self.config.attempts
            )
        else:
            self.optimizer = create_optimizer(
                self.config.optimizer, seed=self.config.seed, k=self.config.attempts
            )
        if self.config.parent_picker:
            from refactory_dsh.optim import PickedParent, ParentSelector
            if isinstance(self.optimizer, (ParentSelector, PickedParent)):
                raise LoopError(
                    f"the {self.optimizer.name} optimizer picks parents itself, so it cannot "
                    f"take the {self.config.parent_picker} parent picker; pair the picker "
                    "with a proposer (--optimizer autoresearch or random)")
            try:
                picker = create_optimizer(self.config.parent_picker, seed=self.config.seed,
                                          k=self.config.attempts)
                self.optimizer = PickedParent(picker, self.optimizer)  # type: ignore[arg-type]
            except ValueError as exc:
                raise LoopError(str(exc)) from exc
        if self.config.selection not in SELECTIONS:
            raise LoopError(f"unknown selection {self.config.selection!r}; known: "
                            f"{', '.join(s for s in SELECTIONS if s)}")
        pool_size = self.config.pool_size
        if isinstance(pool_size, bool) or not isinstance(pool_size, int) or pool_size < 1:
            raise LoopError("pool_size must be a positive integer")
        if self.config.portfolio not in PORTFOLIOS:
            raise LoopError(f"unknown portfolio {self.config.portfolio!r}; known: "
                            f"{', '.join(p for p in PORTFOLIOS if p)}")
        if self.config.portfolio and not getattr(self.optimizer, "uses_model", False):
            raise LoopError(f"portfolio {self.config.portfolio} asks a model proposer for one "
                            f"candidate per edit level; the {self.optimizer.name} optimizer "
                            "asks no model (use --optimizer autoresearch)")
        #: This round's edit levels (``portfolio``) and the ones already accepted.
        self._levels_wanted: list[str] = []
        self._levels_done: list[str] = []
        # The run's cap on edits per proposal, enforced where every proposal is applied.
        self.optimizer.max_edits = int(self.config.max_edits_per_proposal)
        # Whether edits may add code nodes, and what their code may never contain.
        self.optimizer.allow_code_nodes = bool(self.config.allow_code_nodes)
        self.optimizer.grader_secrets = self._grader_secrets

        # How a candidate is run, and therefore how it is scored. Built here from
        # this loop's own settings rather than switched on inside `_evaluate`, so a
        # third evaluator is a new class in `evaluators.py` and not a new branch.
        self.evaluator = create_evaluator(
            self.config.evaluator,
            config=self.config,
            model=model,
            project=self.project,
        )
        self.population: list[Candidate] = []
        self.rejections: list[Rejection] = []
        #: Candidates a person pinned, in the order they were pinned. A keep is not
        #: a score: it changes which candidates the next proposals are made against,
        #: so a good point that a newer batch outscored still gets followed up.
        self.kept: list[str] = []
        #: Candidates a person dismissed, in the order they dismissed them. The
        #: population still records what they scored -- a run does not rewrite its own
        #: history -- but nothing selects them again.
        self.rejected: list[str] = []
        #: What `adopt-best-now` wrote, as the payloads `adopt.adopt` returned or the
        #: refusals it raised. Kept so the panel can show a person what their act did.
        self.adoptions: list[dict[str, Any]] = []
        self.proposals: dict[str, dict[str, Any]] = {}
        self.epoch = 0
        self.generation = 0
        self._epoch_open = False
        self.evaluations = 0
        self.bound = ""
        self.attempts_used: list[int] = []
        self._refusal_pressure = 0.0
        self.loss: dict[str, Any] = {}
        self.gradient: dict[str, Any] = {}
        #: What the outer loop decided this iteration, read by `_context()`. A
        #: plain `Loop` decides nothing beyond the bound; a strategy that branches
        #: writes what it chose here, so the choice reaches the proposal instead of
        #: only the log.
        self.strategy_decision: dict[str, Any] = {}
        self.epochs: list[EpochSummary] = []
        #: The held-out audit of the winner, computed once by :meth:`audit_holdout`
        #: and reported by :meth:`report`. ``None`` means it has not been computed.
        self.holdout_audit: dict[str, Any] | None = None
        #: The original seed state, for tool-policy admission checks. Set at init,
        #: restored from the journal on resume. Not derived from population[0].
        self._original_seed_state: dict[str, Any] | None = None
        #: The strategy's own state, backing store for the default
        #: strategy_state_dict / load_strategy_state implementations.
        self._strategy_state: dict[str, Any] = {}
        #: Loop-private storage for per-part evaluation data (records, workdirs,
        #: scores).  Keyed by (candidate_id, part_name).  This data never reaches
        #: optimizer-facing APIs.
        self._part_results: dict[str, dict[str, list[dict[str, Any]]]] = {}
        #: Loop-private per-candidate, per-part occasion counters.
        self._occasion_counters: dict[str, dict[str, int]] = {}
        #: Loop-private per-candidate train item records.
        self._train_item_records: dict[str, list[dict[str, Any]]] = {}
        #: The loop-level parts manifest (train/val/test id lists), loaded from
        #: config.parts_manifest or the run root.
        self._parts_manifest: dict[str, list[str]] | None = None
        #: The loaded ``--seed-results`` file, and the parts it has already served.
        self._seed_cache: dict[str, Any] | None = None
        self._seed_cache_used: set[str] = set()
        #: Parts the cache covers that its candidate was measured on for real,
        #: because the evaluation asked for other items (a train minibatch).
        self._seed_cache_missed: set[str] = set()
        #: The candidate the current optimizer step edits, recomputed on every
        #: ``optimizer_step`` (from ``select_parents(1)``, unless a strategy set
        #: ``strategy_decision["parent"]``). Loop-private: it decides whose train
        #: records the proposal context carries, and is not a strategy decision.
        self._editing_parent: str | None = None
        #: ``train_minibatch``: each epoch's sampled train ids, and the epoch whose
        #: minibatch each candidate's train records come from.
        self._minibatches: dict[int, list[str]] = {}
        self._records_epoch: dict[str, int] = {}
        #: Each candidate's train records by the epoch (and so the seed) they come from,
        #: so a child is compared with its parent scored in the same epoch.
        self._records_by_epoch: dict[str, dict[int, list[dict[str, Any]]]] = {}
        #: The proposal attempt within the current optimizer step (``propose``).
        self._proposal_attempt = 0
        #: Each accepted proposal's rationale and prediction, by candidate id.
        self._proposal_notes: dict[str, dict[str, Any]] = {}
        #: Whether every train evaluation so far scored the mean of its items' scores
        #: (``None`` until one did or did not): only then is one item worth 1/N of the
        #: score, which the proposal request may then say.
        self._score_is_item_mean: bool | None = None
        #: A pure OuterLoop subclass whose hooks this Loop drives.  When set,
        #: ``strategy()``, ``strategy_state_dict()``, and
        #: ``load_strategy_state()`` delegate to this object.
        self._custom_strategy: OuterLoop | None = None
        #: Reentrant lock protecting concurrent access to post-measure
        #: state mutations (budget counters, records, journal writes).
        self._state_lock: threading.RLock = threading.RLock()
        #: Run-wide item-in-flight limiter (walker: BoundedSemaphore).
        self._item_gate: threading.BoundedSemaphore | None = None
        #: Whether _init_item_gate has been called.
        self._item_gate_initialized: bool = False
        #: The validation gate when ``config.val_gate`` is on (see ``_gate``), and
        #: the candidates it accepted and rejected, in order.
        self._val_gate: ValidationGate | None = None
        #: Why ``val_gate`` is what it is (``_resolve_val_gate``), and the warning a run
        #: that fell back to train selection gives.
        self._val_gate_basis = "explicit"
        self.selection_warning = ""
        #: What a resume put back from the run's record (``_keep_recorded_run_shape``).
        self._resumed_shape: dict[str, Any] = {}
        self._gate_accepted: list[str] = []
        self._gate_rejected: list[str] = []
        #: The ids each gating judged, in order (``_pool_entered``).
        self._gate_rounds: list[list[str]] = []

    def _init_item_gate(self, candidate: Candidate | None = None) -> None:
        """Create the item-in-flight limiter on the first evaluate_many call.

        The limit is ``max_items_in_flight``, defaulting to the largest
        ``effective_parallelism`` over the data nodes of ``candidate`` (the
        seed), which honours a ``parallelism_knob`` such as LiveMath's
        ``data_parallelism``; 1 when there is no data node.  The walker gets a
        ``BoundedSemaphore``; the spine a slot directory, emptied here: nothing
        is in flight in this Loop yet, and a slot left by an earlier process
        whose pid was reused would otherwise never be reclaimed.
        """
        if self._item_gate_initialized:
            return
        max_items = self.config.max_items_in_flight
        if max_items is None:
            source = candidate if candidate is not None else (
                self.population[0] if self.population else None)
            mode = self.mode_class(state=source.state) if source is not None else self.mode
            knobs = mode.values()
            levels = [
                node.effective_parallelism(knobs)
                for node in mode.graph().nodes.values() if node.kind == "data"
            ]
            max_items = max(levels, default=1)
        if self.config.evaluator == "walker":
            self._item_gate = threading.BoundedSemaphore(max_items)
        else:
            slot_dir = self.root / ".refactory" / "item-slots"
            shutil.rmtree(slot_dir, ignore_errors=True)
            slot_dir.mkdir(parents=True, exist_ok=True)
        self._resolved_max_items_in_flight = max_items
        self._item_gate_initialized = True

    def search_name(self) -> str:
        """What runs the search, as the run header and report name it: an attached
        custom strategy's own name, else the optimizer's (#493 G33)."""
        attached = getattr(self, "_custom_strategy", None)
        return str(attached.name) if attached is not None else self.optimizer.name

    def _code_nodes_offered(self, sandbox: tuple[bool, str] | None = None) -> bool:
        """Whether the request describes code nodes: the run allows them and this
        machine can sandbox them (``sandbox``, the request's `snippet.available`
        answer; asked when not given)."""
        if not self.config.allow_code_nodes:
            return False
        if sandbox is None:
            from refactory_dsh import snippet

            sandbox = snippet.available()
        return bool(sandbox[0])

    def _sandbox_answer(self) -> tuple[bool, str] | None:
        """The machine's sandbox answer for one request (`snippet.available`), taken once
        so the offer and the refusals agree; ``None`` when the run allows no code nodes."""
        if not self.config.allow_code_nodes:
            return None
        from refactory_dsh import snippet

        return snippet.available()

    def _grader_secrets(self) -> frozenset:
        """The train items' grader-only values (`task.grader_secrets`), loaded once, the
        first time a candidate carries code (``E-secret``)."""
        cached = self.__dict__.get("_grader_secret_values")
        if cached is None:
            from refactory_dsh.task import TRAIN, grader_secrets

            try:
                cached = frozenset(grader_secrets(self.mode.state_dict(), part=TRAIN))
            except Exception:  # noqa: BLE001 - a Task that cannot load has no items to leak
                cached = frozenset()
            self.__dict__["_grader_secret_values"] = cached
        return cached

    def permitted_ops(self) -> list[str]:
        """The edit operations this run may apply.

        An explicit ``allowed_ops`` is the whole answer. Otherwise the edit
        preset's operations, plus the ``required_ops`` the running strategy
        declares -- an attached custom strategy's, or this loop's own class's.
        """
        if self.config.allowed_ops:
            return list(self.config.allowed_ops)
        permitted = list(nn_edits.enabled(self.config.edit_preset))
        attached = getattr(self, "_custom_strategy", None)
        strategy = attached if attached is not None else self
        for op in getattr(strategy, "required_ops", ()):
            if op not in permitted:
                permitted.append(op)
        return permitted

    def set_custom_strategy(self, strategy: OuterLoop) -> None:
        """Attach a custom strategy (pure OuterLoop subclass) to this Loop.

        The Loop will delegate ``strategy``, ``strategy_state_dict`` and
        ``load_strategy_state`` calls to the strategy object.  The strategy
        sees only ``Run``'s public API.
        """
        self._custom_strategy = strategy

    # ── init ───────────────────────────────────────────────────────────────
    def _seed_ledger(self) -> None:
        """Write the run's intent ledger, when somebody said what the run is for.

        Nothing is written when ``intent`` is empty: a ledger holding a default is one
        a fidelity check cannot check, which is BUG-05, and a run nobody steered needs
        no ledger rather than an empty one that looks like a statement. Refuses a
        placeholder here rather than at the gate, because the person who typed it is
        the one who can fix it.
        """
        if not self.config.intent.strip():
            return
        from refactory_dsh.steering import Ledger

        ledger = Ledger(self.root)
        if ledger.exists():
            # A resumed run keeps what it was already told, including anything said
            # since. Re-seeding would put a second origin in the ledger.
            return
        Ledger.seed(self.root, self.config.intent)

    def init(self) -> None:
        """Seed the population and refuse to search a mode that cannot be run."""
        seed_state = self.mode.state_dict()
        for name, value in self.config.knobs.items():
            # `configure`, not the editor: the search may not move a frozen knob, but
            # the caller deciding what this run operates on is not the search.
            seed_state = nn_edits.configure(seed_state, name, value)
        if self.config.knobs:
            # The loop's own mode is what the report and the seed candidate's parent
            # read, so it has to carry the same configuration as the population.
            self.mode.load_state_dict(seed_state)
        issues = validate_state(
            seed_state, invariants=self.invariants, resolve_calls=True,
            grader_side=(*self.protected, *self.graders), graders=self.graders,
        )
        if issues:
            raise LoopError(
                f"the seed state of {self.mode_id} is not runnable; "
                f"first problem: {issues[0]}"
            )
        unreachable = self.unreachable_knobs()
        if unreachable:
            # Loud, because the failure is silent: the run reports full batches over
            # a surface smaller than the mode declared.
            print(
                json.dumps({
                    "warning": "optimizer_cannot_reach_some_knobs",
                    "optimizer": self.optimizer.name,
                    "knobs": unreachable,
                    "detail": "declared and movable, but this optimizer cannot produce "
                              "a different value for them",
                }),
                file=sys.stderr,
            )
        if not self.mode.is_scored():
            raise LoopError(
                f"{self.mode_id} does not override score(), so the loop has no "
                "signal to search"
            )

        self.root.mkdir(parents=True, exist_ok=True)
        from refactory_dsh import profiles
        self._keep_recorded_run_shape()
        settings = {k: v for k, v in self.config.to_dict().items() if k not in profiles.RUNTIME_FIELDS}
        snapshot = profiles.refresh(self.project, self.mode_id, settings, self.config.resolved_profile, model_client=self.model) if self.config.resolved_profile else profiles.resolve(
            self.project, self.mode_id, overrides=settings, validate_settings=False, model_client=self.model)
        snapshot_path = self.root / profiles.SNAPSHOT
        if self.config.resume and snapshot_path.exists():
            previous = json.loads(snapshot_path.read_text())
            # Historical evidence is usable only with the same implementation,
            # input, route and contract. Runtime addressing is deliberately absent.
            if previous.get("context_hash") != snapshot.get("context_hash"):
                raise LoopError("evaluation context changed; start a new run instead of reusing cached scores")
            if snapshot.get("profile") and (snapshot.get("evaluation_context", {}).get("routing") or {}).get("status") == "runtime_defaults_unresolved":
                raise LoopError("the saved profile's inherited model route is unresolved; start a new run with an explicit model before reusing evidence")
        self.config.resolved_profile = snapshot
        if not self.config.resume:
            snapshot_path.write_text(json.dumps(snapshot, indent=2, default=str))
        else:
            self._journal("resume_configuration", resolved_profile=snapshot)
        self._resolve_val_gate()
        if self.config.val_gate and self._custom_strategy is None and not self.config.selection:
            # Recorded as resolved, so config.json says the rule the run selects by and
            # a resume compares real values (``_keep_recorded_run_shape``).
            self.config.selection = self.selection_mode()
        saved = self.root / "config.json"
        if self.config.resume and saved.exists():
            try:
                before = int(json.loads(saved.read_text()).get("train_minibatch") or 0)
            except (OSError, ValueError, json.JSONDecodeError):
                before = self.config.train_minibatch
            if before != self.config.train_minibatch:
                # The epochs already scored used the old samples; a different size would
                # compare candidates scored on different-sized minibatches.
                raise LoopError(
                    f"this run scores minibatches of {before} train items; resume it with "
                    f"train_minibatch {before}, or start a new run")
        saved.write_text(
            json.dumps({"mode": self.mode_id, **self.config.to_dict()}, indent=2, default=str)
        )
        # Recorded for every run, so its seed results can be exported for later ones.
        from refactory_dsh import seed_cache
        context = seed_cache.score_context(self.config, self.mode_class)
        (self.root / seed_cache.CONTEXT_FILE).write_text(json.dumps(context, indent=2, default=str))
        if self.config.seed_results:
            try:
                self._seed_cache = seed_cache.load(self.config.seed_results, context)
            except seed_cache.SeedCacheError as exc:
                raise LoopError(str(exc)) from exc
        self._seed_ledger()
        # Every model call the optimizer makes, appended as it happens; a fresh run
        # in this directory starts it over, as it does the journal.
        self.optimizer.trace_path = self.root / OPTIMIZER_TRACE
        if not self.config.resume:
            self.optimizer.trace_path.unlink(missing_ok=True)
        resumed = self._replay_if_asked()
        (self.root / run_history.RUN_FILE).write_text(
            json.dumps(
                {
                    "version": run_history.RUN_VERSION,
                    "mode": self.mode_id,
                    "resumed": resumed,
                    "journal": JOURNAL,
                },
                indent=2,
            )
        )
        # Load the loop-level parts manifest if configured or already on disk.
        self._load_parts_manifest()
        if self._seed_cache is not None:
            # A cache that can never apply stops the run here, rather than every
            # evaluation quietly measuring for real.
            try:
                seed_cache.check_applies(
                    self.config.seed_results, self._seed_cache, Candidate.from_state(seed_state).id,
                    {part: self._expected_item_ids(seed_state, part)[0]
                     for part in (self._seed_cache.get("parts") or {})})
            except seed_cache.SeedCacheError as exc:
                raise LoopError(str(exc)) from exc
        self._check_optimizer_route()
        if self.config.val_gate:
            if self._custom_strategy is not None:
                raise LoopError(
                    f"val_gate is the default loop's gate; the {self._custom_strategy.name} "
                    "strategy keeps its own, so turn val_gate off"
                )
            if not (self._parts_manifest and task_module.VAL in self._parts_manifest):
                raise LoopError(
                    "val_gate needs a parts manifest with a 'val' part (--parts); "
                    "give the run one, or turn val_gate off"
                )
        elif (self.config.train_minibatch
              and len(self._train_ids()) > self.config.train_minibatch):
            # Without the gate the winner and the parents are picked by train score,
            # and on minibatches those come from different samples.
            raise LoopError(
                f"train_minibatch {self.config.train_minibatch} scores each epoch on a "
                "different train sample, so train scores of different epochs cannot pick "
                "the winner or the parents; turn val_gate on (--val-gate 1), or score the "
                "whole train part (train_minibatch 0)")
        elif self.config.early_stop_patience or self.config.early_stop_at_full_val:
            raise LoopError(
                "early_stop_patience and early_stop_at_full_val measure the validation "
                "gate's best score; turn val_gate on, or leave them off"
            )
        elif self.config.selection:
            raise LoopError(
                f"selection {self.config.selection} chooses by validation score, which only "
                "the validation gate measures, and this run has none (val_gate 0, or a "
                "parts manifest without val items); give it val items, or leave selection "
                "empty"
            )

        if resumed:
            if getattr(self, "_executing_strategy", False) and self.bound:
                # A previous stop reason is history, not the new strategy's
                # decision. Its current bounds are checked before another epoch.
                previous_bound, self.bound = self.bound, ""
                self._journal("bound", bound="", previous_bound=previous_bound,
                              reason="explicit search resume")
            self._write_state()
            self._progress("init", resumed=True, population=len(self.population),
                           **({"warning": self.selection_warning} if self.selection_warning else {}),
                           epochs=self.config.epochs, batch_size=self.config.batch_size,
                           optimizer=self.search_name())
            return

        self._journal("start", mode=self.mode_id, config=self.config.to_dict())
        seed = Candidate.from_state(seed_state)
        # Persist the original seed state for tool-policy admission checks.
        # Anchored here and restored on resume, not derived from population[0].
        self._original_seed_state = seed_state
        self._journal("original_seed", state=seed_state)
        self.population = [seed]
        self._journal_candidate(seed)
        self._seed_population(seed)
        self._write_state()
        self._progress(
            "init",
            # The run's events say it (the command line prints them): no val items.
            **({"warning": self.selection_warning} if self.selection_warning else {}),
            resumed=False,
            population=len(self.population),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            optimizer=self.search_name(),
        )

    #: The settings that shape how a run selects, which a resumed run keeps as it
    #: recorded them (``_keep_recorded_run_shape``): the epochs already run selected so.
    RESUME_KEEPS = ("val_gate", "selection", "pool_size")

    def _keep_recorded_run_shape(self) -> None:
        """On resume, put back the run's recorded ``RESUME_KEEPS`` settings from its
        ``config.json``, before anything reads them.

        A setting the resume command leaves out takes the recorded value, whatever the
        profile or the defaults now say (said in the run's log when they differ). One
        the resume command gives explicitly with a different value is refused, as a
        different ``train_minibatch`` is: the run would select its later epochs by
        another rule than its earlier ones. Which settings a command gave explicitly is
        the launch snapshot's ``overrides``; a loop built in code without one counts a
        value other than ``LoopConfig``'s default as given. A ``selection`` left empty
        is compared as the default it resolves to."""
        self._resumed_shape = {}
        saved = self.root / "config.json"
        if not self.config.resume or not saved.exists():
            return
        try:
            recorded = json.loads(saved.read_text())
        except (OSError, ValueError):
            return
        if not isinstance(recorded, dict):
            return
        overrides = (self.config.resolved_profile or {}).get("overrides")
        defaults = LoopConfig()
        for key in self.RESUME_KEEPS:
            value = recorded.get(key)
            if key == "val_gate" and not isinstance(value, bool):
                continue
            if key == "selection" and value not in SELECTIONS:
                continue
            if key == "pool_size" and (isinstance(value, bool) or not isinstance(value, int)
                                       or value < 1):
                continue
            current = getattr(self.config, key)
            if key == "val_gate" and current is not None:
                current = bool(current)
            if self._shape_value(key, current) == self._shape_value(key, value):
                continue
            given = (key in overrides if isinstance(overrides, dict)
                     else current != getattr(defaults, key))
            if given:
                flag = "--" + key.replace("_", "-")
                raise LoopError(
                    f"this run was started with {key} {self._shown(key, value)} and this "
                    f"resume asks for {key} {self._shown(key, current)}; resume it without "
                    f"{flag} (or with {flag} {self._shown(key, value)}), or start a new run")
            setattr(self.config, key, value)
            self._resumed_shape[key] = value
            Run(_loop=self).log(f"resume: keeping the run's recorded {key} {value!r} "
                                f"(this launch said {current!r})")

    def _shown(self, key: str, value: Any) -> str:
        """A ``RESUME_KEEPS`` value as its flag takes it: ``val_gate`` as 1 or 0, an
        empty ``selection`` as the default it resolves to."""
        if isinstance(value, bool):
            return str(int(value))
        if key == "selection" and not value:
            return f"{self._shape_value(key, value)} (unset: the default)"
        return str(value)

    def _shape_value(self, key: str, value: Any) -> Any:
        """A ``RESUME_KEEPS`` value as it acts: an empty ``selection`` is the default
        it resolves to."""
        if key == "selection" and not value:
            picker = getattr(getattr(self.optimizer, "picker", None), "name", "")
            return default_selection(self.name, self.config.optimizer, picker)
        return value

    def _resolve_val_gate(self) -> None:
        """Decide ``val_gate`` when the run left it unset (``None``): on when the parts
        manifest it will load has val items and the strategy is a built-in one (a
        custom strategy such as SkillOpt keeps its own gate), off otherwise. Off for
        want of val items, the run says it selects on train scores (``NO_VALIDATION``),
        in its events and its strategy log now, and in its report. An explicit value is
        kept. Runs before ``config.json`` is written, so the run records the decision."""
        self._val_gate_basis = "explicit"
        self.selection_warning = ""
        if "val_gate" in self._resumed_shape:
            # Restored by _keep_recorded_run_shape: the epochs already run selected so.
            self._val_gate_basis = "resumed: the run's recorded decision"
        if self.config.val_gate is not None:
            self.config.val_gate = bool(self.config.val_gate)
            return
        if self._custom_strategy is not None:
            self.config.val_gate = False
            self._val_gate_basis = f"the {self._custom_strategy.name} strategy keeps its own gate"
            return
        has_val = bool(self._manifest_val_ids())
        self.config.val_gate = has_val
        self._val_gate_basis = ("default: the parts manifest has val items" if has_val
                                else "default: no val items")
        if not has_val:
            self.selection_warning = NO_VALIDATION
            Run(_loop=self).log(f"warning: {NO_VALIDATION}")

    def _manifest_val_ids(self) -> list[str]:
        """The val ids of the parts manifest this run will load (``parts_manifest``, else
        the run root's ``parts.json``); empty when there is none or it cannot be read
        (``_load_parts_manifest`` then refuses an unreadable one)."""
        source = (Path(self.config.parts_manifest) if self.config.parts_manifest
                  else self.root / task_module.MANIFEST)
        try:
            data = json.loads(source.read_text())
        except (OSError, ValueError):
            return []
        ids = data.get(task_module.VAL) if isinstance(data, dict) else None
        return [str(item) for item in ids] if isinstance(ids, list) else []

    def _check_optimizer_route(self) -> None:
        """Refuse, before anything runs, an optimizer that calls a model on a route that
        cannot run it without tools: its calls are prompt in, JSON out
        (``optim.random_optimizer.text_only``), and running it with tools is never a
        silent fallback."""
        from refactory_dsh.model import TOOL_POLICY_NONE_BACKENDS

        if not getattr(self.optimizer, "uses_model", False) or self.model is None:
            return
        route_for = getattr(self.model, "for_node", None)
        client = route_for("optimizer", role="optimizer") if callable(route_for) else self.model
        route = client.route() if callable(getattr(client, "route", None)) else {}
        backend = str((route or {}).get("worker_backend") or "dsh")
        if backend in TOOL_POLICY_NONE_BACKENDS:
            return
        raise LoopError(
            f"the {self.optimizer.name} optimizer's calls run without tools (tool_policy "
            f"'none'), which only the {', '.join(TOOL_POLICY_NONE_BACKENDS)} backends "
            f"enforce, and the optimizer role is routed to {backend}; route the optimizer "
            "role to one of them, e.g. --role-routes "
            "'{\"optimizer\": {\"worker_backend\": \"claude_code\"}}'")

    def _load_parts_manifest(self) -> None:
        """Load and install the loop-level parts manifest.

        If ``config.parts_manifest`` names a file, copy it to the run root as
        ``parts.json`` and load it.  If the run root already has a ``parts.json``
        (e.g. from a previous run being resumed), load that.
        """
        dest = self.root / task_module.MANIFEST
        if self.config.parts_manifest:
            source = Path(self.config.parts_manifest)
            if not source.exists():
                raise LoopError(
                    f"parts_manifest {self.config.parts_manifest!r} does not exist"
                )
            self.root.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
        if not dest.exists():
            return
        # A manifest that cannot be read is refused rather than ignored: falling back
        # to the fraction split would grade the run on items nobody chose, and
        # overlapping parts would let val/test items into the optimizer's train set.
        try:
            data = json.loads(dest.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise LoopError(f"parts manifest {dest} is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise LoopError(f"parts manifest {dest} must be a JSON object of id lists")
        parts = {
            k: [str(x) for x in v]
            for k, v in data.items()
            if isinstance(v, list)
        }
        names = [name for name in (task_module.TRAIN, task_module.VAL, task_module.TEST) if name in parts]
        for index, first in enumerate(names):
            for second in names[index + 1:]:
                shared = set(parts[first]) & set(parts[second])
                if shared:
                    raise LoopError(
                        f"parts manifest {dest}: {first} and {second} share items "
                        f"{sorted(shared)}; train, val and test must be disjoint"
                    )
        self._parts_manifest = parts

    def _seed_population(self, seed: Candidate) -> None:
        """Fill the starting batch with neighbours of the seed.

        A population of one would spend the first epoch evaluating a single point
        and look like a search while doing almost nothing. It is also swept rather
        than sampled: one candidate per movable knob, each at a value the default
        is not, so the first epoch produces a gradient with *contrast* in it. A
        purely structural starting population gives the optimizer plenty of
        candidates and no evidence about which value of anything is worth having.
        """
        wanted = max(0, int(self.config.batch_size) - 1)
        if not wanted:
            return
        sampler = RandomOptimizer(seed=self.config.seed)
        sampler.allow_code_nodes = bool(self.config.allow_code_nodes)
        sampler.grader_secrets = self._grader_secrets
        edits = self._exploration_edits(seed, wanted)
        insertable = [node.to_state() for node in self.mode.insertable()]
        rng = random.Random(f"{self.config.seed}:sweep")
        while len(edits) < wanted:
            edit = nn_edits.random_edit(
                seed.state,
                preset=self.config.edit_preset,
                insertable=insertable,
                rng=rng,
            )
            if edit is None:
                break
            edits.append(edit)

        for edit in edits:
            if edit.get("op") not in self.permitted_ops():
                continue
            proposal = Proposal(edit=edit, parent=seed.id, rationale="starting population")
            outcome = sampler.attempt(
                proposal, seed, self.invariants, True, self.config.frozen, self._tried(),
                seed_state=self._original_seed_state, protected=self.protected,
                graders=self.graders,
                allow_new_node_tools=self.config.allow_new_node_tools,
            )
            if isinstance(outcome, Candidate):
                self.population.append(outcome)
                self._journal_candidate(outcome)
                self.proposals.setdefault(outcome.id, {
                    "id": outcome.id,
                    "parent": outcome.parent,
                    "edits": outcome.edits,
                    "rationale": "swept for the starting population",
                    "prediction": "",
                    "confidence": 0.0,
                    "origin": "seed_population",
                })
            else:
                self.rejections.append(Rejection(1, edit, outcome))

    def _exploration_edits(self, seed: Candidate, wanted: int) -> list[dict[str, Any]]:
        """One parameter move per movable knob, at a value the default is not."""
        rng = random.Random(self.config.seed)
        entries = seed.state.get("knobs") or {}
        frozen = set(self.config.frozen)
        edits: list[dict[str, Any]] = []
        for name in sorted(entries):
            if len(edits) >= wanted:
                break
            knob = Knob.from_spec(entries[name]["spec"])
            if knob.frozen or name in frozen:
                continue
            value = _other_value(knob, entries[name]["value"], rng)
            if value is None:
                continue
            edits.append({"op": "set_knob", "name": name, "value": value})
        return edits

    # ── forward ────────────────────────────────────────────────────────────
    def forward(self, *, epoch: int | None = None) -> int:
        """Evaluate every candidate in the batch that has not run yet.

        ``epoch`` (default: the current one) picks the train seed under ``val_gate``.
        The time and token budgets are checked before each wave of candidates: once one
        is spent the rest stay unscored (the report names them) and the spent budget is
        the run's bound, rather than a measurement refused by it being recorded as a
        failure.

        The candidates are independent, so up to ``max_concurrent_evaluations`` of them
        run at once (``Run.evaluate_many``, under the run-wide item limit). A wave holds
        only candidates the one-after-another order would also have started: the
        evaluation budget and, with ``max_rollouts``, the most rollouts each can add
        decide its size, so the same candidates are scored, with the same items and
        seed. ``max_concurrent_evaluations`` 1 runs them one at a time, as before.

        Under an elapsed-time or token budget the budget is checked before each wave,
        so the last wave may finish up to one wave of evaluations past the limit (one
        at a time overshoots by one evaluation). A candidate whose evaluation the
        budget refused or cut (``BudgetStopped``) stays unevaluated, for a resume.
        """
        from refactory_dsh.budgets import BudgetExhausted

        batch = [candidate for candidate in self.population if not candidate.evaluated]
        at = self.epoch if epoch is None else epoch
        seed = self._train_seed(at)
        index = 0
        while index < len(batch):
            if self.evaluations >= self.config.max_evaluations:
                break
            if self.config.max_rollouts is not None and self.rollouts >= self.config.max_rollouts:
                break
            if self.budget:
                try:
                    self.budget.checkpoint("work" if self._owns_resource_budget else None)
                except BudgetExhausted as exc:
                    self.bound = str(exc)
                    break
            items = self._minibatch(at)
            each = len(items) if items is not None else len(self._train_ids())
            size = self._wave_size(min(len(batch) - index,
                                       self.config.max_evaluations - self.evaluations), each)
            wave = batch[index:index + size]
            try:
                if len(wave) == 1:
                    self._evaluate(wave[0], seed=seed, items=items)
                    self._forwarded(wave[0], index, len(batch), at, items)
                else:
                    start = index
                    Run(_loop=self).evaluate_many(
                        [EvalRequest(candidate=candidate, part=task_module.TRAIN, items=items, seed=seed)
                         for candidate in wave],
                        on_result=lambda position, candidate: self._forwarded(
                            candidate, start + position, len(batch), at, items),
                    )
            except BudgetStopped as exc:
                self.bound = self.bound or str(exc)
                break
            index += len(wave)
            if self.budget and self.budget.snapshot().get("exhausted"):
                break
        return len(batch)

    def _wave_size(self, wanted: int, items_each: int) -> int:
        """How many of the next ``wanted`` evaluations may run at once (at least one).

        At most ``max_concurrent_evaluations``. With ``max_rollouts`` set, only as many
        as the one-after-another order would also start: each is counted at the most
        rollouts it can add (its items times ``repeats``), and the wave also fits
        ``evaluate_many``'s own rollout check; a single evaluation runs alone, as before.
        """
        size = max(1, min(wanted, int(self.config.max_concurrent_evaluations or 1)))
        cap = self.config.max_rollouts
        if cap is None:
            return size
        each = max(int(items_each), 1)
        count = 0
        while (count < size and self.rollouts + count * each * self.config.repeats < cap
               and self.rollouts + (count + 1) * each <= cap):
            count += 1
        return max(count, 1)

    def _forwarded(self, candidate: Candidate, index: int, of: int, epoch: int,
                   items: list[str] | None) -> None:
        """What ``forward`` records after one candidate's train evaluation."""
        if items is not None or self._fresh_evidence():
            # Which epoch's sample and seed its records come from, so the proposal
            # step re-scores it only when they are another epoch's.
            with self._state_lock:
                self._records_epoch[candidate.id] = epoch
                self._records_by_epoch.setdefault(candidate.id, {})[epoch] = list(
                    self._train_item_records.get(candidate.id, []))
                self._journal("train_records_epoch", id=candidate.id, epoch=epoch)
        self._progress(
            "forward",
            candidate=candidate.id,
            index=index,
            of=of,
            score=candidate.score,
            exit_ok=candidate.exit_ok,
            error=candidate.error,
            duration_s=candidate.duration_s,
        )

    def _train_seed(self, epoch: int) -> int:
        """The seed a train rollout in ``epoch`` uses.

        A Task that shuffles per (item, seed) -- LiveMath's answer choices -- meets
        that seed. Without ``val_gate`` it is the run's seed, so every candidate of
        the run meets the same shuffle; with it, SkillOpt's per-epoch train seed
        (``train_batch_seed``), as SkillOpt's train rollouts use.
        """
        if self.config.val_gate:
            return train_batch_seed(self.config.seed, epoch)
        return self.config.seed

    def _train_ids(self) -> list[str]:
        """The train part's item ids: the manifest's, else the Task's train split."""
        manifest = self._parts_manifest
        if manifest and task_module.TRAIN in manifest:
            return sorted(str(item) for item in manifest[task_module.TRAIN])
        state = self.population[0].state if self.population else self.mode.state_dict()
        return sorted(self._expected_item_ids(state, task_module.TRAIN)[0])

    def _minibatch(self, epoch: int) -> list[str] | None:
        """The train ids every candidate of ``epoch`` is scored on, or ``None`` for the
        whole train part. Sampled once per epoch with ``train_batch_seed`` (so a resumed
        run samples the same ids) and journaled."""
        size = self.config.train_minibatch
        if not size:
            return None
        if epoch in self._minibatches:
            return self._minibatches[epoch]
        ids = self._train_ids()
        if len(ids) <= size:
            return None
        sample = sorted(random.Random(train_batch_seed(self.config.seed, epoch)).sample(ids, size))
        self._minibatches[epoch] = sample
        self._journal("train_minibatch", epoch=epoch, ids=sample)
        return sample

    def _fresh_evidence(self) -> bool:
        """Whether the edited candidate is re-scored each epoch on the whole train part
        (``fresh_evidence`` under ``val_gate``, the default loop's own gate)."""
        return bool(self.config.val_gate and self.config.fresh_evidence
                    and self._custom_strategy is None)

    def _refresh_edited_records(self) -> None:
        """Fresh evidence: score the edited candidate on this epoch's train sample with
        this epoch's seed when its train records come from another epoch, so the
        proposal evidence (and every reader's input) is this epoch's rollout.

        On under ``val_gate`` with ``fresh_evidence`` (the whole train part, or the
        ``train_minibatch`` sample), and with a minibatch in any case. A candidate
        created and scored this epoch already has this epoch's records and is not
        scored again; a resumed run reads the epoch from the journal, so it does not
        either. Its score is left alone: only the records the optimizer reads change.
        """
        batch = self._minibatch(self.epoch)
        if not batch and not self._fresh_evidence():
            return
        edited = self.strategy_decision.get("parent") or self._editing_parent
        if not edited or self._records_epoch.get(str(edited)) == self.epoch:
            return
        parent = next((c for c in self.population if c.id == edited and c.evaluated), None)
        if parent is None:
            return
        if self.config.max_rollouts is not None and self.rollouts >= self.config.max_rollouts:
            return
        if self.evaluations >= self.config.max_evaluations:
            return
        if self.budget:
            # The time and token budgets, checked as every scoring path checks them: a
            # spent budget stops the re-score here, before anything runs.
            from refactory_dsh.budgets import BudgetExhausted
            try:
                self.budget.checkpoint("work" if self._owns_resource_budget else None)
            except BudgetExhausted as exc:
                self.bound = str(exc)
                return
        workdir = self.root / "runs" / parent.id / "occasions" / f"train_epoch_{self.epoch}"
        if batch:
            task_module.make_manifest(workdir, train_ids=batch)
        elif self._parts_manifest is not None and (self.root / task_module.MANIFEST).exists():
            # The whole train part, as the candidate's first evaluation read it.
            workdir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.root / task_module.MANIFEST, workdir / task_module.MANIFEST)
        result, measurements = self._measure(parent, workdir, task_module.TRAIN,
                                             seed=self._train_seed(self.epoch), items=batch)
        if not measurements.get("count"):
            # The measurement's own budget checkpoint stopped it before a trial ran:
            # nothing was scored, so nothing is counted and the records stay.
            return
        expected = len(batch) if batch else len(self._train_ids())
        scored = {str(record.get("id")) for record in result.item_records}
        if not (result.exit_ok and measurements.get("complete") and len(scored) >= expected):
            # A failed or partial re-score is no evidence: the previous records stay, the
            # epoch is not marked (a resume in this epoch tries again), and what it spent
            # is counted and journaled like any evaluation.
            with self._state_lock:
                self.evaluations += 1
                if not measurements.get("cached_from"):
                    self.rollouts += len(result.item_records) or 1
                self._journal("rescore_failed", id=parent.id, epoch=self.epoch,
                              seed=self._train_seed(self.epoch), exit_ok=result.exit_ok,
                              error=result.error, records=len(scored), expected=expected,
                              measurements=measurements)
                self._journal("rollouts", rollouts=self.rollouts)
            self._progress("rescore", candidate=parent.id, epoch=self.epoch, failed=True,
                           error=result.error, records=len(scored), expected=expected,
                           duration_s=result.duration_s)
            return
        with self._state_lock:
            # An evaluation like any other: it counts toward the evaluation budget (the
            # time and token budgets are charged inside the measurement), and it is
            # journaled so a resumed run counts it too.
            self.evaluations += 1
            self._journal("rescore", id=parent.id, epoch=self.epoch,
                          seed=self._train_seed(self.epoch), items=len(batch or ()) or None,
                          score=result.score if result.exit_ok else None,
                          exit_ok=result.exit_ok, measurements=measurements)
            self._train_item_records[parent.id] = list(result.item_records)
            self._records_epoch[parent.id] = self.epoch
            self._records_by_epoch.setdefault(parent.id, {})[self.epoch] = list(result.item_records)
            if not measurements.get("cached_from"):
                self.rollouts += len(result.item_records) or 1
            self._journal("train_records", id=parent.id, records=list(result.item_records))
            self._journal("train_records_epoch", id=parent.id, epoch=self.epoch)
            self._journal("rollouts", rollouts=self.rollouts)
        self._progress("rescore", candidate=parent.id, epoch=self.epoch,
                       score=result.score if result.exit_ok else None,
                       records=len(result.item_records), duration_s=result.duration_s)

    def _carry_ledger(self, workdir: Path) -> None:
        """Put the run's ledger where the candidate's own nodes look for it.

        The run's ledger lives at the run root; a candidate is evaluated in
        ``runs/<id>/``; and the steering transform folds in the ledger of the work
        directory it is handed. Without this the ledger a person seeded with
        ``--intent``, or appended to with ``refactory steer --note`` while the search
        was going, was written, reported by ``steer``, and read by no node — in the one
        place steering is most wanted, because a search is what runs for hours.

        Copied rather than searched for up the tree, so each candidate directory stays
        a self-contained record of what the person had said when it was evaluated. A
        note added while a candidate is running reaches the next candidate, exactly as
        it reaches the next node of an interactive run.
        """
        from refactory_dsh.steering import LEDGER

        source = self.root / LEDGER
        if not source.exists():
            return
        workdir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, workdir / LEDGER)

    def _write_subset(self, workdir: Path, part: str, *, seed: int | None = None) -> None:
        """Name the part of the items this candidate is evaluated on.

        Written into the candidate's own run directory before the evaluator starts,
        and read by the item loader of either runtime. It is not a knob: a search
        that could set it would choose its own exam, and there would be no item left
        for the audit to hold out.
        """
        task_module.write_subset(workdir, part, self.config.holdout, seed=seed)

    def _evaluate(
        self, candidate: Candidate, *, seed: int | None = None, items: list[str] | None = None,
        _budget_reserved: bool = False,
        item_gate: threading.BoundedSemaphore | None = None,
        item_slots: bool = False,
        stop_event: threading.Event | None = None,
    ) -> None:
        """Run one candidate through the evaluator, and record what it scored.

        This is the loop's forward pass and the evaluator is its loss: what the
        candidate scored is whatever the evaluator says it scored. Which evaluator
        ran it is not this method's business -- see :mod:`refactory_dsh.evaluators`
        for the two that ship and what makes them agree.

        The evaluator runs the train part of the item selection, and the loop takes
        the size cost off afterwards. ``raw_score`` keeps the evaluator's number, so
        the report can say whether a candidate won on merit or on size.

        When ``_budget_reserved`` is True, the caller has already pre-incremented
        ``self.evaluations``; this method skips that increment.
        """
        workdir = self.root / "runs" / candidate.id
        mode = self.mode_class(state=candidate.state)
        # Copy the loop-level manifest so select_subset finds it.
        if self._parts_manifest is not None:
            workdir.mkdir(parents=True, exist_ok=True)
            manifest_src = self.root / task_module.MANIFEST
            if manifest_src.exists():
                shutil.copyfile(manifest_src, workdir / task_module.MANIFEST)
        # When explicit items are given, write a per-candidate manifest.
        if items is not None:
            workdir.mkdir(parents=True, exist_ok=True)
            task_module.make_manifest(workdir, train_ids=items)
        result, measurements = self._measure(
            candidate, workdir, task_module.TRAIN, seed=seed, items=items,
            item_gate=item_gate, item_slots=item_slots, stop_event=stop_event,
        )
        if measurements.get("budget_stopped"):
            self._budget_stopped(candidate, task_module.TRAIN, result, measurements)
        nodes = node_count(candidate.state)
        candidate.raw_score = result.score if result.exit_ok else None
        item_scores = [float(r["score"]) for r in result.item_records
                       if isinstance(r.get("score"), (int, float)) and not isinstance(r.get("score"), bool)]
        if result.exit_ok and item_scores:
            # A mode's score may be any metric of the run; it is a per-item mean only
            # when it equals the mean of the item scores, every time.
            mean = math.isclose(sum(item_scores) / len(item_scores), float(result.score),
                                rel_tol=1e-9, abs_tol=1e-9)
            with self._state_lock:
                self._score_is_item_mean = mean and self._score_is_item_mean is not False
        candidate.score = result.score - self.config.parsimony * nodes if result.exit_ok else None
        candidate.mean_score = candidate.raw_score
        candidate.measurements = measurements
        candidate.facts = _trim(result.facts)
        candidate.exit_ok = result.exit_ok
        candidate.error = result.error
        candidate.duration_s = result.duration_s
        candidate.model_calls = result.model_calls
        candidate.evaluated = True
        with self._state_lock:
            self._train_item_records[candidate.id] = list(result.item_records)
            if not _budget_reserved:
                self.evaluations += 1
            if not measurements.get("cached_from"):
                self.rollouts += len(result.item_records) or 1
            self._journal("train_records", id=candidate.id,
                          records=list(result.item_records))
            self._journal("rollouts", rollouts=self.rollouts)
            self._journal(
                "score",
                id=candidate.id,
                score=candidate.score,
                raw_score=candidate.raw_score,
                nodes=nodes,
                parsimony=self.config.parsimony,
                # The rollout seed, so an export can say what its records were measured
                # with (``seed_cache``).
                seed=seed,
                exit_ok=candidate.exit_ok,
                error=candidate.error,
                duration_s=candidate.duration_s,
                model_calls=candidate.model_calls,
                facts=candidate.facts,
                measurements=measurements,
            )
            self._write_candidate(candidate, workdir, mode=mode)

    def _budget_stopped(self, candidate: Candidate, part: str, result: Any,
                        measurements: dict[str, Any]) -> None:
        """A train evaluation a time or token budget refused or cut: what it ran counts
        toward rollouts and is journaled (``budget_stopped``), but the candidate stays
        unevaluated, so a resume with more budget scores it. Raises ``BudgetStopped``."""
        with self._state_lock:
            if not measurements.get("cached_from"):
                self.rollouts += len(result.item_records)
            self._journal("budget_stopped", id=candidate.id, part=part, error=result.error,
                          measurements=measurements)
            self._journal("rollouts", rollouts=self.rollouts)
        raise BudgetStopped(f"the budget stopped {candidate.id}'s {part} evaluation: {result.error}",
                            reason=self.bound or str(result.error or ""))

    def _evaluate_part(
        self, candidate: Candidate, part: str,
        items: list[str] | None, seed: int | None,
        *,
        item_gate: threading.BoundedSemaphore | None = None,
        item_slots: bool = False,
        stop_event: threading.Event | None = None,
    ) -> None:
        """Evaluate on val or test in a fresh occasion directory (F18)."""
        with self._state_lock:
            counters = self._occasion_counters.setdefault(candidate.id, {})
            counter = counters.get(part, 0)
            counters[part] = counter + 1
            self._journal("occasion_counter", id=candidate.id, part=part,
                          counter=counters[part])
        workdir = self.root / "runs" / candidate.id / "occasions" / f"{part}_{counter}"
        workdir.mkdir(parents=True, exist_ok=True)
        task_module.write_subset(workdir, part, self.config.holdout, seed=seed)
        # Copy or build the manifest so select_subset finds it.
        if items is not None:
            task_module.make_manifest(workdir, train_ids=items if part == "train" else [],
                                     val_ids=items if part == "val" else None,
                                     test_ids=items if part == "test" else None)
        elif self._parts_manifest is not None:
            manifest_src = self.root / task_module.MANIFEST
            if manifest_src.exists():
                shutil.copyfile(manifest_src, workdir / task_module.MANIFEST)
        self._carry_ledger(workdir)
        result, measurements = self._measure(
            candidate, workdir, part, seed=seed, items=items,
            item_gate=item_gate, item_slots=item_slots, stop_event=stop_event,
        )
        # Score-only occasion record for optimizer-facing storage (no records,
        # no workdir).
        occasion_public = {
            "part": part, "occasion": counter,
            "score": result.score if result.exit_ok else None,
            "exit_ok": result.exit_ok, "error": result.error,
            "duration_s": result.duration_s,
            # A rollout a time or token budget refused or cut: no score of the
            # candidate's (``ValidationGate.val_score`` skips it).
            **({"budget_stopped": True} if measurements.get("budget_stopped") else {}),
        }
        with self._state_lock:
            self._part_results.setdefault(candidate.id, {}).setdefault(part, []).append(
                occasion_public
            )
            self._journal("part_result", id=candidate.id, part=part, seed=seed,
                          result=occasion_public)
            if not measurements.get("cached_from"):
                self.rollouts += len(result.item_records) or 1
            self._journal("rollouts", rollouts=self.rollouts)
        if result.item_records:
            (workdir / "item_records.json").write_text(
                json.dumps(result.item_records, indent=2, default=str))
        if measurements.get("budget_stopped"):
            raise BudgetStopped(f"the budget stopped {candidate.id}'s {part} rollout: {result.error}",
                                reason=self.bound or str(result.error or ""))

    def _measure(
        self, candidate: Candidate, workdir: Path, part: str, *,
        seed: int | None = None, items: list[str] | None = None,
        item_gate: threading.BoundedSemaphore | None = None,
        item_slots: bool = False,
        stop_event: threading.Event | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Independent bounded trials; any failed contract makes a candidate ineligible."""
        from refactory_dsh.nn.mode import RunResult
        from refactory_dsh.budgets import BudgetExhausted

        # Loop-owned item channel: expected ids computed from the seed's state
        # and the loop root, not from the candidate's trial directory.
        seed_state = self.population[0].state if self.population else candidate.state
        expected_ids, seed_declares_item_scores = self._expected_item_ids(seed_state, part)
        # When an explicit items subset was requested, narrow expected_ids to
        # just those items so _verify_item_set does not fail on the "missing"
        # items that were intentionally excluded.
        if items is not None:
            expected_ids = set(str(i) for i in items)

        cached = self._cached_measure(candidate, workdir, part, expected_ids, seed)
        if cached is not None:
            return cached

        trials = []
        budget_cut = False  # a time or token budget refused or cut a trial
        all_item_records: list[dict[str, Any]] = []  # accumulate across repeats
        last = RunResult(exit_ok=False, error="no measurement completed")
        required = (self.config.resolved_profile.get("contract") or {}).get("required_artifacts", [])
        for index in range(self.config.repeats):
            trial_dir = workdir if index == 0 else workdir / "trials" / str(index + 1)
            self._carry_ledger(trial_dir)
            self._write_subset(trial_dir, part, seed=seed)
            # Copy the manifest to trial sub-directories so select_subset
            # finds it there too (repeats > 1 creates separate directories).
            manifest = workdir / task_module.MANIFEST
            if trial_dir != workdir and manifest.exists():
                trial_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(manifest, trial_dir / task_module.MANIFEST)
            before = self.budget.snapshot() if self.budget else {}
            try:
                if self.budget:
                    final = part == task_module.HOLDOUT or getattr(self, "_final_scoring", False)
                    self.budget.checkpoint(("final_validation" if final else "work")
                                           if self._owns_resource_budget else None)
                with self.budget.retain_validation_reserve() if self.budget else nullcontext():
                    eval_kwargs: dict[str, Any] = {}
                    if stop_event is not None and self.evaluator.out_of_process:
                        eval_kwargs["stop_event"] = stop_event
                    if item_gate is not None:
                        eval_kwargs["item_gate"] = item_gate
                    elif (item_slots and self.evaluator.out_of_process
                          and hasattr(self, '_resolved_max_items_in_flight')):
                        # Spine, under evaluate_many only: the shared slot
                        # directory. A plain Run.evaluate launches as before.
                        slot_dir = self.root / ".refactory" / "item-slots"
                        eval_kwargs["item_slots_dir"] = str(slot_dir)
                        eval_kwargs["item_slots"] = self._resolved_max_items_in_flight
                    result = self.evaluator.evaluate(
                        self.mode_class(state=candidate.state), candidate.state,
                        trial_dir, **eval_kwargs,
                    )
            except BudgetExhausted as exc:
                last = RunResult(exit_ok=False, error=str(exc))
                self.bound = str(exc)
                budget_cut = True
                break
            after = self.budget.snapshot() if self.budget else {}
            budget_exhausted_after_trial = False
            if self.budget:
                try:
                    self.budget.checkpoint()
                except BudgetExhausted as exc:
                    self.bound = str(exc)
                    budget_exhausted_after_trial = True
            # Post-trial item-ownership verification (works with both evaluators).
            self._verify_item_set(expected_ids, seed_declares_item_scores, trial_dir, part, result)
            missing = [path for path in required if not (trial_dir / path).exists()]
            valid_score = isinstance(result.score, (int, float)) and math.isfinite(result.score)
            if missing or not valid_score:
                result.exit_ok = False
                result.error = result.error or ("required artifacts missing: " + ", ".join(missing) if missing else "score is unmeasured or nonfinite")
            trial = {"trial": index + 1, "score": result.score if result.exit_ok else None,
                     "exit_ok": result.exit_ok, "error": result.error, "duration_s": result.duration_s,
                     "model_calls": result.model_calls, "required_contract": {"passed": result.exit_ok, "missing_artifacts": missing},
                     "tokens": after.get("known_tokens", 0) - before.get("known_tokens", 0) if after.get("token_status") in ("measured", "estimated") else None,
                     "cost": after["estimated_cost_usd"] - (before.get("estimated_cost_usd") or 0) if after.get("estimated_cost_usd") is not None else None,
                     "workdir": str(trial_dir)}
            trial["workers"] = []
            for receipt_path in sorted(trial_dir.glob("**/.refactory/workers/*.json")):
                try:
                    receipt = json.loads(receipt_path.read_text())
                    trial["workers"].append({key: receipt.get(key) for key in ("worker_id", "node_id", "role", "state", "actual_route", "runtime", "usage")})
                except (OSError, json.JSONDecodeError):
                    continue
            trials.append(trial)
            with self._state_lock:
                self.measurements += 1
            # Each record says which repeat graded it: joined across repeats, the
            # same item id appears once per repeat, and without the index two
            # gradings of one item read as a duplicate.
            all_item_records.extend({**record, "repeat": index + 1} for record in result.item_records)
            last = result
            if budget_exhausted_after_trial and not result.exit_ok:
                budget_cut = True  # the budget ran out inside the trial and ended it
            if not result.exit_ok:
                break
            if budget_exhausted_after_trial:
                budget_cut = budget_cut or len(trials) < self.config.repeats
                break
        scores = [t["score"] for t in trials if t["score"] is not None]
        complete = len(trials) == self.config.repeats and all(t["exit_ok"] for t in trials)
        summary = {"requested": self.config.repeats, "count": len(trials), "successful": len(scores),
                   "mean": statistics.mean(scores) if scores else None,
                   "stdev": statistics.stdev(scores) if len(scores) > 1 else None,
                   "min": min(scores) if scores else None, "max": max(scores) if scores else None,
                   "complete": complete, "trials": trials, "context_hash": self.config.resolved_profile.get("context_hash"),
                   "tokens": sum(t["tokens"] for t in trials) if trials and all(isinstance(t["tokens"], (int, float)) for t in trials) else None,
                   "cost": sum(t["cost"] for t in trials) if trials and all(isinstance(t["cost"], (int, float)) for t in trials) else None,
                   "interpretation": "observed sample mean; improvement is not a statistical proof"}
        if budget_cut and not complete:
            summary["budget_stopped"] = True
        last.score = float(summary["mean"] or 0) if complete else 0.0
        last.exit_ok = complete
        last.duration_s = sum(t["duration_s"] for t in trials)
        last.model_calls = sum(t["model_calls"] for t in trials)
        # Every trial's item records count toward rollouts and are kept
        # per occasion (brief ask 1: "every trial's items count toward
        # rollouts and the records of every repeat are kept per occasion").
        last.item_records = all_item_records
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "measurements.json").write_text(json.dumps(summary, indent=2))
        return last, summary

    def _cached_measure(
        self, candidate: Candidate, workdir: Path, part: str, expected_ids: set[str],
        seed: int | None = None,
    ) -> tuple[Any, dict[str, Any]] | None:
        """The ``--seed-results`` entry for this evaluation, if one applies.

        Only the cache's own candidate, only on the exact item set it recorded and
        with the rollout seed it was measured with (a Task may shuffle an item by it:
        a gated run's train rollouts use SkillOpt's per-epoch seed, an ungated run's
        the run's own), and only once per part: a later occasion is a fresh measurement. The
        recorded run directory is copied in, so an optimizer that reads a
        candidate's pass files finds them as if the evaluation had run here.
        """
        from refactory_dsh.nn.mode import RunResult

        cache = self._seed_cache
        if cache is None or candidate.id != cache.get("candidate"):
            return None
        entry = (cache.get("parts") or {}).get(part)
        if entry is None:
            return None
        recorded = entry["result"]
        # Every trial the cache holds; a file written before trials were exported
        # holds one, with its records on the result.
        trials = entry.get("trials") or [{"score": recorded["score"], "duration_s": recorded.get("duration_s"),
                                          "model_calls": recorded.get("model_calls"),
                                          "workdir": entry.get("workdir"),
                                          "item_records": recorded.get("item_records") or []}]
        other_seed = entry.get("seed") != seed
        if (set(entry.get("items") or ()) != expected_ids or other_seed
                or len(trials) != self.config.repeats):
            # The cache's own candidate on a part it covers, measured on other items
            # (a train minibatch) or with another rollout seed: said once per part, so
            # the run shows why the cache was not used.
            with self._state_lock:
                if part not in self._seed_cache_missed and part not in self._seed_cache_used:
                    self._seed_cache_missed.add(part)
                    reason = ("other items" if set(entry.get("items") or ()) != expected_ids
                              else f"rollout seed {seed}, cached {entry.get('seed')}" if other_seed
                              else f"{len(trials)} cached trial(s), {self.config.repeats} requested")
                    self._journal("seed_cache_miss", id=candidate.id, part=part, reason=reason,
                                  cached_items=len(entry.get("items") or ()), items=len(expected_ids))
            return None
        # Claim the part under the state lock: concurrent evaluations
        # (``Run.evaluate_many``) must not both replay the same entry.
        with self._state_lock:
            if part in self._seed_cache_used:
                return None
            self._seed_cache_used.add(part)
        ignore = shutil.ignore_patterns(
            "occasions", "trials", task_module.MANIFEST, "subset.json", "measurements.json")
        source = Path(entry.get("workdir") or "")
        if source.is_dir():
            shutil.copytree(source, workdir, dirs_exist_ok=True, ignore=ignore)
            for number in range(2, len(trials) + 1):
                # Each later trial's run directory, where a live run keeps it.
                if (source / "trials" / str(number)).is_dir():
                    shutil.copytree(source / "trials" / str(number), workdir / "trials" / str(number),
                                    dirs_exist_ok=True, ignore=ignore)
        records = [{**record, "repeat": number}
                   for number, trial in enumerate(trials, start=1) for record in trial["item_records"]]
        result = RunResult(
            exit_ok=True, score=float(recorded["score"]), error=None,
            duration_s=float(recorded.get("duration_s") or 0.0),
            model_calls=int(recorded.get("model_calls") or 0),
            facts=dict(recorded.get("facts") or {}),
            item_records=records,
            workdir=workdir,
        )
        origin = f"{cache.get('source')}#{part}"
        scores = [float(t["score"]) if t.get("score") is not None else result.score for t in trials]
        summary = {"requested": self.config.repeats, "count": len(trials), "successful": len(trials),
                   "mean": result.score, "stdev": statistics.stdev(scores) if len(scores) > 1 else None,
                   "min": min(scores), "max": max(scores), "complete": True,
                   "trials": [{"trial": number, "score": score, "exit_ok": True, "error": None,
                               "duration_s": float(trial.get("duration_s") or 0.0),
                               "model_calls": int(trial.get("model_calls") or 0),
                               "required_contract": {"passed": True, "missing_artifacts": []},
                               "tokens": None, "cost": None,
                               "workdir": str(workdir if number == 1 else workdir / "trials" / str(number)),
                               "workers": [], "cached_from": origin}
                              for number, (trial, score) in enumerate(zip(trials, scores), start=1)],
                   "context_hash": self.config.resolved_profile.get("context_hash"),
                   "tokens": None, "cost": None, "cached_from": origin,
                   "interpretation": "recorded by an earlier run under the same score context"}
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "measurements.json").write_text(json.dumps(summary, indent=2))
        with self._state_lock:
            self._journal("seed_cache_hit", id=candidate.id, part=part, source=origin,
                          items=len(result.item_records), trials=len(trials), score=result.score)
        return result, summary

    def _write_candidate(
        self, candidate: Candidate, workdir: Path, *, mode: Any = None
    ) -> None:
        workdir.mkdir(parents=True, exist_ok=True)
        nodes = node_count(candidate.state)
        (workdir / "candidate.json").write_text(
            json.dumps(
                {
                    "id": candidate.id,
                    "parent": candidate.parent,
                    # Both numbers, so a reader of one candidate directory can tell
                    # what the evaluator measured from what the search compared.
                    "score": candidate.score,
                    "raw_score": candidate.raw_score,
                    "nodes": nodes,
                    "parsimony": self.config.parsimony,
                    "size_cost": round(self.config.parsimony * nodes, 6),
                    "exit_ok": candidate.exit_ok,
                    "error": candidate.error,
                    "duration_s": candidate.duration_s,
                    "edits": candidate.edits,
                    "facts": candidate.facts,
                    "model_calls": candidate.model_calls,
                    "measurements": candidate.measurements,
                    # What the run published, against what the mode declared: a port
                    # that is missing is a promise the mode did not keep. Written for
                    # every candidate whichever evaluator ran it, because a reader
                    # comparing two runs should not have to know how they were run.
                    "ports": {
                        port.name: (workdir / port.path).exists()
                        for port in (mode.port_declarations() if mode is not None else ())
                    },
                },
                indent=2,
                default=str,
            )
        )

    # ── loss ───────────────────────────────────────────────────────────────
    def _seed_score(self) -> float | None:
        """The seed's score, the point the search started from; None until it ran and scored."""
        seed = self.population[0] if self.population else None
        return seed.score if seed is not None and seed.evaluated and seed.exit_ok else None

    def loss_step(self) -> dict[str, Any]:
        evaluated = self._evaluated()
        scored = [c for c in evaluated if c.exit_ok and c.score is not None]
        scores = [c.score for c in scored]
        raw_scores = [c.raw_score for c in scored if c.raw_score is not None]
        best = max(scores) if scores else None
        mean = (sum(scores) / len(scores)) if scores else None
        # What this epoch is compared with: the last epoch's best, or for the first epoch
        # the seed's score, the point the search started from (dm-265).
        previous = self.loss.get("best") if self.loss else self._seed_score()
        self.loss = {
            "epoch": self.epoch,
            "batch": len(evaluated),
            # The scores the search compared, and the evaluator's own numbers beside
            # them, so an epoch's numbers can be read either way.
            "scores": scores,
            "raw_scores": raw_scores,
            "mean": mean,
            "best": best,
            "raw_best": max(raw_scores) if raw_scores else None,
            "parsimony": self.config.parsimony,
            "best_ever": max([s for s in (previous, best) if s is not None], default=None),
            "improved": bool(best is not None and previous is not None and best > previous),
            "run_failures": [c.id for c in self.population if c.evaluated and not c.exit_ok],
        }
        self._write("loss.json", self.loss)
        self._journal("loss", **self.loss)
        self._progress("loss", **self.loss)
        return self.loss

    # ── backward ───────────────────────────────────────────────────────────
    def backward(self) -> dict[str, Any]:
        """Turn the traces into the gradient the optimizer reads."""
        evaluated = [c for c in self._evaluated() if c.exit_ok and c.score is not None]
        self.gradient = {
            "epoch": self.epoch,
            "scoring": self.scoring(),
            "knob_attribution": self._knob_attribution(evaluated),
            "cost": self._cost(),
            "measurement_summary": self._measurement_summary(),
            "run_failures": self.loss.get("run_failures", []),
            "rejection_codes": self._rejection_codes(),
            "rejection_reasons": [r for rej in self.rejections[-8:] for r in rej.reasons],
            "accepted_op_mix": self._op_mix(),
            "structural": self._structural_stats(),
            "leaderboard": [self._row(c) for c in self.leaderboard(8)],
        }
        self._write("gradient.json", self.gradient)
        self._progress(
            "backward",
            knobs=len(self.gradient["knob_attribution"]),
            failures=len(self.gradient["run_failures"]),
        )
        return self.gradient

    def _knob_attribution(self, evaluated: Sequence[Candidate]) -> dict[str, Any]:
        """What each parameter value was worth, over everything evaluated so far.

        The score attributed here is the one the search compares -- the evaluator's
        number minus the run's size cost -- so a value is credited with what choosing
        it did to the search's objective rather than to the raw measurement.

        ``untried`` is part of the gradient, not decoration: an optimizer reading
        only the values that have run has no way to tell a value that failed from
        one nobody has tried, and will keep re-proposing what it already knows.
        """
        buckets: dict[str, dict[str, list[float]]] = {}
        # With the validation gate, what it decided about the candidates behind each
        # value, so a value whose candidates were all rejected is not read as a gain.
        decisions: dict[str, dict[str, dict[str, int]]] = {}
        # Each knob's declaration as the candidates carry it (the latest wins), so a
        # knob an edit added is enumerated like one the mode declares.
        specs: dict[str, dict[str, Any]] = {}
        # A frozen knob (the mode's run settings, or one the operator froze) is not the
        # optimizer's to move, so its values are not attributed, nor its untried ones offered.
        frozen = set(self.config.frozen) | {
            name for name, knob in self.parameters.items() if getattr(knob, "frozen", False)}
        for candidate in evaluated:
            decision = self._gate_decision(candidate.id)
            for name, entry in (candidate.state.get("knobs") or {}).items():
                if isinstance(entry.get("spec"), dict):
                    specs[name] = entry["spec"]
                    if entry["spec"].get("frozen"):
                        frozen.add(name)
                buckets.setdefault(name, {}).setdefault(str(entry["value"]), []).append(
                    float(candidate.score)
                )
                if decision is not None:
                    counts = decisions.setdefault(name, {}).setdefault(
                        str(entry["value"]), {"kept": 0, "rejected": 0, "not gated": 0})
                    counts["kept" if decision in ("kept", "current") else decision] += 1
        attribution: dict[str, Any] = {}
        for name, values in sorted(buckets.items()):
            if name in frozen:
                continue
            attribution[name] = {
                "tried": {
                    value: {
                        "n": len(scores),
                        "mean": round(sum(scores) / len(scores), 6),
                        "max": round(max(scores), 6),
                        **({"gate": decisions[name][value]} if name in decisions else {}),
                    }
                    for value, scores in sorted(values.items())
                },
                "untried": self._untried(name, values, specs.get(name)),
            }
        return attribution

    def _untried(self, name: str, tried: dict[str, list[float]],
                 spec: Mapping[str, Any] | None = None) -> list[Any]:
        """The declared values of a knob that have not been evaluated.

        A knob with a finite set -- an enum, or a ``str`` with declared values --
        enumerates it. An ``int`` with both bounds declared is just as finite and
        just as readable, so it enumerates its range too. An open-ended numeric
        knob has no set to enumerate and answers empty. ``spec`` is the declaration
        the candidates carry, read before the mode's: a knob an edit added has no
        declaration in the mode.
        """
        knob = Knob.from_spec(dict(spec)) if spec else self.parameters.get(name)
        if knob is None:
            return []
        finite = knob.accepted()
        if finite:
            return [member for member in finite if str(member) not in tried]
        if knob.kind == "int" and knob.lo is not None and knob.hi is not None:
            return [
                value
                for value in range(int(knob.lo), int(knob.hi) + 1)
                if str(value) not in tried
            ]
        return []

    def _efficiency(self) -> dict[str, Any]:
        """How many attempts it took per candidate that could actually run."""
        proposed = len(self.proposals)
        attempts = sum(self.attempts_used)
        return {
            "rounds": attempts,
            "batches": len(self.attempts_used),
            "effective_k": sorted(set(self.attempts_used)),
            "candidates": proposed,
            "attempts_per_candidate": (
                round(attempts / proposed, 3) if proposed else None
            ),
            "why": "k starts at the configured attempts and rises by one when the "
                   "previous batch was refused more than it was accepted, because "
                   "that is the signal that the optimizer is close to something "
                   "legal rather than out of ideas",
        }

    def _proposer_health(self) -> dict[str, Any]:
        """Whether the proposer answered at all, said where a reader will see it.

        A failed round is caught and the run continues, which is right: a model
        that is down should not kill a run. But the run then looks exactly like one
        where the model answered and had nothing useful, because both end with a
        score that did not move and no rejections. That reads as a finding about
        the mode when it is a fact about the machine, so the count is reported and
        a run that never got an answer says so in one line.
        """
        failures = list(getattr(self.optimizer, "proposal_failures", []))
        rounds = int(getattr(self.optimizer, "proposal_rounds", 0))
        all_failed = bool(rounds) and len(failures) >= rounds
        if all_failed:
            finding = (
                f"every one of the {rounds} proposal round(s) failed, so this run is "
                "not a search result: nothing was proposed, and the score did not "
                "move because the proposer never answered."
            )
        elif failures:
            finding = (
                f"{len(failures)} of {rounds} proposal round(s) failed; the rest "
                "were proposed and evaluated normally."
            )
        else:
            finding = ""
        return {
            "optimizer": self.optimizer.name,
            "rounds": rounds,
            "failures": failures[:5],
            "failure_count": len(failures),
            "all_failed": all_failed,
            "finding": finding,
        }

    def _dominant_refusal(self) -> dict[str, Any]:
        """The commonest refusal, and what it says about the catalogue.

        A rule that refuses most of what the optimizer proposes is a finding about
        the ops it was offered, not about the optimizer. Counting the codes and
        stopping there leaves that unsaid.
        """
        counts = self._rejection_codes()
        if not counts:
            return {"code": None, "count": 0, "share": 0.0, "finding": ""}
        code, count = max(counts.items(), key=lambda item: item[1])
        total = sum(counts.values())
        share = round(count / total, 3)
        finding = ""
        if share >= 0.5 and code.startswith("E-"):
            finding = (
                f"{share:.0%} of refusals were {code}. That is a statement about the "
                "operations the optimizer was offered, not about the optimizer: the "
                "catalogue asks for something these proposals cannot satisfy, so the "
                "field descriptions or the preconditions are the thing to fix."
            )
        elif share >= 0.5 and code.startswith("V"):
            finding = (
                f"{share:.0%} of refusals were {code}. The optimizer is proposing edits "
                "that break a rule it does not have to guess at -- the rules are in its "
                "prompt -- so either the mode's graph invites that edit or the rule needs "
                "stating more plainly."
            )
        return {"code": code, "count": count, "share": share, "finding": finding}

    def _convergence(self) -> dict[str, Any]:
        """Whether the optimizer has run out of ideas within the surface it sees.

        A no-op or a duplicate means the same thing: every edit it can think of
        lands on a point that is already the case. That is a convergence symptom,
        not one more refusal.
        """
        counts = self._rejection_codes()
        spent = {code: counts.get(code, 0) for code in ("E-noop", "E-duplicate")}
        total = sum(spent.values())
        if not total:
            return {"reason": None, "detail": ""}
        accepted = sum(1 for c in self.population if c.edits)
        if total > accepted:
            return {
                "reason": "surface_exhausted",
                "detail": (
                    f"{total} proposal(s) landed on a state that already exists and "
                    f"{accepted} produced something new, so the optimizer is out of "
                    "ideas within the surface it can see; the surface needs widening, "
                    "not more attempts"
                ),
            }
        return {"reason": None, "detail": "", "noops": total}

    def _cost(self) -> dict[str, Any]:
        durations = [c.duration_s for c in self._evaluated()]
        if not durations:
            return {"n": 0}
        calls = sum(c.model_calls for c in self._evaluated())
        return {
            "n": len(durations),
            "mean_s": round(statistics.fmean(durations), 4),
            "max_s": round(max(durations), 4),
            "total_s": round(sum(durations), 4),
            "inner_model_calls": calls,
        }

    def _measurement_summary(self) -> dict[str, Any]:
        """The evaluations' trials in a few numbers, for the optimizer: how many ran,
        how many failed, and the failures by cause. No receipts and no paths."""
        trials = [trial for c in self._evaluated()
                  for trial in (c.measurements or {}).get("trials") or ()]
        causes: dict[str, int] = {}
        for trial in trials:
            if not trial.get("exit_ok"):
                cause = " ".join(str(trial.get("error") or "unknown").split())[:120]
                causes[cause] = causes.get(cause, 0) + 1
        return {
            "trial_runs": len(trials),
            "failed": sum(causes.values()),
            "timeouts": sum(n for cause, n in causes.items() if "timeout" in cause.lower()
                            or "timed out" in cause.lower()),
            "failures_by_cause": dict(sorted(causes.items(), key=lambda kv: -kv[1])[:6]),
        }

    def _rejection_codes(self) -> dict[str, int]:
        return _codes(self.rejections)

    def _op_mix(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for candidate in self.population:
            for edit in candidate.edits:
                op = str(edit.get("op", "?"))
                counts[op] = counts.get(op, 0) + 1
        return dict(sorted(counts.items()))

    def _structural_stats(self) -> dict[str, Any]:
        structural = [c for c in self.population if c.is_structural()]
        knob_only = [c for c in self.population if not c.is_structural()]
        structural_rejections = [
            r for r in self.rejections
            if any(isinstance(step, dict) and nn_edits.is_macro(step.get("op", ""))
                   for step in Proposal(edit=r.edit).steps)
        ]
        return {
            "structural_candidates": len(structural),
            "knob_candidates": len(knob_only),
            "structural_refusals": len(structural_rejections),
            "structural_yield": _yield(len(structural), len(structural_rejections)),
            "accepted_ops": self._op_mix(),
        }

    # ── optimizer ──────────────────────────────────────────────────────────
    def optimizer_step(
        self, run: Run | None = None
    ) -> tuple[list[Candidate], list[Rejection]]:
        """Propose a batch and turn it into candidates, through the overridable steps.

        Going through ``propose`` and ``install`` rather than straight to the
        optimizer is what lets a strategy change one of them and keep the other.
        """
        run = run or Run(_loop=self)
        # The pair a proposer is handed, and why it is the whole population rather
        # than the elite: `describe()` carries a score and the edits that produced
        # it, not the state, so a candidate the loop does not hand over cannot be
        # bred from. Passing only the elite made parent selection impossible inside
        # an optimizer, which is why every SRF method that selects parents had to be
        # a loop instead. Selection is part of "how to search", so the optimizer gets
        # the pool and chooses from it (issue #32).
        parents = [candidate for candidate in self.population if candidate.evaluated]
        if self.config.val_gate:
            parents = self._gated_parents()
        if not parents:
            parents = self.select_parents(self.config.elite)
        if not parents:
            return [], []
        # The parent this step edits, so ProposalContext.item_records carries its
        # records, not board[0]'s (brief ask 5). Recomputed every step: the leader
        # moves between steps, and a parent fixed at the first step would hand every
        # later proposal a stale candidate's records. A strategy may name the parent
        # in strategy_decision; otherwise select_parents(1) -- which respects kept
        # candidates and is overridable -- decides. Kept out of strategy_decision,
        # which reaches the proposal as the strategy's own choice.
        explicit = self.strategy_decision.get("parent")
        # An optimizer that picks its own parent (a parent picker paired with a
        # proposer) picks it here, before the request is built, so the request
        # describes the candidate its proposals are applied to.
        choose = getattr(self.optimizer, "choose_parent", None)
        picked = None if explicit or not callable(choose) else choose(
            ProposalContext(epoch=self.epoch, generation=self.generation,
                            mode_name=self.mode_id, elite=int(self.config.elite),
                            val_scores=self._val_scores()), parents)
        if explicit:
            self._editing_parent = str(explicit)
        elif picked is not None:
            self._editing_parent = picked.id
        else:
            selected = self.select_parents(1)
            self._editing_parent = selected[0].id if selected else None
        self._refresh_edited_records()
        before = len(self.rejections)
        self._levels_wanted, self._levels_done = self._round_levels(), []
        self._proposal_attempt = 0
        proposals = self.propose(run, parents, self.batch_size(run))
        candidates = self.install(run, proposals, parents)
        rejections = self.rejections[before:]
        for rejection in rejections:
            self._journal(
                "rejection",
                attempt=rejection.attempt,
                edit=rejection.edit,
                reasons=rejection.reasons,
            )
        for candidate in candidates:
            self._journal_candidate(candidate)
            notes = self._proposal_notes.get(candidate.id) or {}
            record = self.proposals.setdefault(candidate.id, {
                "id": candidate.id,
                "parent": candidate.parent,
                # Every edit its proposal applied: one, or a compound proposal's list.
                "edits": candidate.own_edits(),
                "rationale": str(notes.get("rationale") or ""),
                "prediction": str(notes.get("prediction") or ""),
                "confidence": float(notes.get("confidence") or 0.0),
                "origin": "optimizer",
                **({"level": notes["level"]} if notes.get("level") else {}),
                "diagnosis": str(notes.get("diagnosis")
                                 or getattr(self.optimizer, "last_diagnosis", "") or ""),
                # The kinds of change its own answer weighed, when it said (optional);
                # never the optimizer's latest, which may be a later retry's.
                **({"considered": str(notes["considered"])} if notes.get("considered")
                   else {}),
                "epoch": self.epoch,
            })
            # Journaled, so a resumed run still shows the optimizer its own reasoning.
            self._journal("proposal", **record)
        self._write_analysis()
        self._append_rejections(rejections)
        self._progress(
            "optimizer",
            # This epoch's proposals, every round `install` asked for: each one was either
            # accepted for evaluation or refused, so the three counts add up (dm-265).
            proposed=len(candidates) + len(rejections),
            accepted=len(candidates),
            refused=len(rejections),
            # Why this epoch's were refused, beside the count it explains.
            epoch_refusal_codes=_codes(rejections),
            # The running tally, so a watcher sees which rule is dominating without
            # opening the run directory. Same derivation as the report's.
            refusal_codes=self._rejection_codes(),
            children=[c.id for c in candidates],
        )
        return candidates, rejections

    def _portfolio_report(self) -> dict[str, Any]:
        """What ``portfolio`` cost and produced, from the journaled proposal records.

        ``by_level``: candidates proposed, evaluated and kept by the validation gate
        (``kept``: accepted, under ``pool`` as the incumbent; with ``in_pool`` under
        ``pool``: in the pool now), and the evaluations they cost (a train run, plus
        each val or test occasion).
        ``rounds``: each round's candidates in the order they were proposed, with the
        gate's decision (``_gate_decision``) and whether it was ``accepted``; a
        candidate the gate did not accept although its val score beat its parent's,
        because a sibling of the same round scored higher on val (or tied and won the
        tie), is ``shadowed``.
        ``extra_evaluations``: what the candidates beyond ``batch_size`` in each round
        cost."""
        from refactory_dsh.optim.base import EDIT_LEVELS

        by_id = {c.id: c for c in self.population}
        order = {c.id: index for index, c in enumerate(self.population)}
        records = sorted((r for r in self.proposals.values() if r.get("level")),
                         key=lambda r: order.get(r["id"], len(order)))
        cache = self._gate().cache if self.config.val_gate and self._custom_strategy is None else {}

        def cost(candidate: Candidate | None) -> int:
            if candidate is None or not candidate.evaluated:
                return 0
            return 1 + sum((self._occasion_counters.get(candidate.id) or {}).values())

        pool = self.selection_mode() == "pool"
        accepted = set(self._gate_accepted)
        fields = ("proposed", "evaluated", "kept", *(("in_pool",) if pool else ()), "evaluations")
        by_level = {level: dict.fromkeys(fields, 0) for level in EDIT_LEVELS}
        rounds: dict[Any, list[dict[str, Any]]] = {}
        for record in records:
            candidate = by_id.get(record["id"])
            entry = by_level.setdefault(str(record["level"]), dict.fromkeys(fields, 0))
            entry["proposed"] += 1
            decision = self._gate_decision(record["id"], detail=True)
            if candidate is not None and candidate.evaluated:
                entry["evaluated"] += 1
            entry["evaluations"] += cost(candidate)
            # Accepted by the gate: under single, kept; under pool, made the incumbent
            # (a pool member that was never the incumbent was not accepted).
            kept = record["id"] in accepted
            if kept:
                entry["kept"] += 1
            if pool and decision in ("current", "in pool"):
                entry["in_pool"] += 1
            siblings = rounds.setdefault(record.get("epoch"), [])
            val, parent_val = cache.get(record["id"]), cache.get(record.get("parent") or "")
            shadowed = (record["id"] in self._gate_rejected and val is not None
                        and parent_val is not None and val > parent_val and any(
                            r["id"] in accepted
                            for r in records if r.get("epoch") == record.get("epoch")
                            and r["id"] != record["id"]))
            siblings.append({"id": record["id"], "level": record["level"],
                             "evaluated": bool(candidate is not None and candidate.evaluated),
                             "gate": decision, "kept": kept, "shadowed": shadowed,
                             "val": val, "parent_val": parent_val,
                             "evaluations": cost(candidate)})
        base = int(self.config.batch_size)
        return {"setting": self.config.portfolio, "batch_size": base,
                "selection": self.selection_mode(),
                "rounds": len(rounds), "by_level": by_level,
                "per_round": [{"epoch": epoch, "candidates": entries}
                              for epoch, entries in rounds.items()],
                "extra_evaluations": sum(entry["evaluations"] for entries in rounds.values()
                                         for entry in entries[base:])}

    def _selection_report(self) -> dict[str, Any]:
        """How this run selected, what its pool holds, and what the search cost.

        ``setting``: ``single`` or ``pool`` under ``val_gate``, ``train`` without it.
        ``pool``: the candidates the next round would choose parents from (the pool;
        under ``single`` the incumbent and what a person kept; without the gate the
        ``pool_size`` best by train score). ``lines_of_descent``: how many distinct
        lines the pool holds, a line being the seed's child a candidate descends from
        (the seed is a line of its own), with ``largest_line_share``. ``edit_kinds``:
        the pool's candidates by the level of the edit that made each one (``seed``,
        ``prompt_knob``, ``pipeline``). ``parents_from_incumbent``: of the candidates
        the run added, the share whose parent was the incumbent when they were added.
        ``budget``: train scorings (evaluations, re-scores included), val and test
        scorings, tokens and wall-clock, so arms are compared at equal cost rather
        than equal rounds.
        """
        from refactory_dsh.optim.base import edit_level

        setting = self.selection_mode() or ("train" if self._custom_strategy is None else "")
        if setting == "train":
            val_ids = (task_module.manifest_part_ids(self._parts_manifest, task_module.VAL)
                       if self._parts_manifest is not None else self._manifest_val_ids())
            warning = (NO_VALIDATION if not val_ids else
                       "val_gate 0: selecting on train scores, though the run has val items")
        else:
            warning = ""
        if setting == "pool":
            members = self._judged_ranked() if self._population_policy() else self._pool()
        elif setting == "single":
            members = self._gated_parents()
        else:
            members = self.leaderboard(int(self.config.pool_size))
        by_id = {c.id: c for c in self.population}

        def line(candidate: Candidate) -> str:
            seen: set[str] = set()
            while candidate.parent and candidate.parent in by_id and candidate.id not in seen:
                seen.add(candidate.id)
                parent = by_id[candidate.parent]
                if not parent.parent:
                    return candidate.id
                candidate = parent
            return candidate.id

        lines: dict[str, int] = {}
        kinds: dict[str, int] = {}
        for candidate in members:
            lines[line(candidate)] = lines.get(line(candidate), 0) + 1
            kind = edit_level(candidate.own_edits()) if candidate.parent else "seed"
            kinds[kind] = kinds.get(kind, 0) + 1
        choices = dict(self._strategy_state.get("parent_from_incumbent") or {})
        counters = [counts for counts in self._occasion_counters.values()]
        snapshot = self.budget.snapshot() if self.budget else {}
        return {
            "setting": setting,
            **({"warning": warning} if warning else {}),
            "val_gate": bool(self.config.val_gate),
            "val_gate_basis": self._val_gate_basis,
            "pool_size": int(self.config.pool_size) if setting == "pool" else None,
            # What ``pool`` lists: the top pool_size, or every judged candidate for a
            # population optimizer or picker.
            "parent_set": ("every judged candidate" if setting == "pool" and self._population_policy()
                           else setting),
            "pool": [c.id for c in members],
            "lines_of_descent": len(lines),
            "largest_line_share": (round(max(lines.values()) / len(members), 3)
                                   if members else None),
            "edit_kinds": dict(sorted(kinds.items())),
            "parents_from_incumbent": (round(sum(1 for v in choices.values() if v) / len(choices), 3)
                                       if choices else None),
            "children": len(choices),
            "budget": {
                "train_scorings": self.evaluations,
                "val_scorings": sum(int(c.get(task_module.VAL, 0)) for c in counters),
                "test_scorings": sum(int(c.get(task_module.TEST, 0)) for c in counters),
                "rollouts": self.rollouts,
                "tokens": snapshot.get("tokens"),
                "known_tokens": snapshot.get("known_tokens"),
                "elapsed_s": snapshot.get("elapsed_s"),
            },
        }

    def _write_analysis(self) -> None:
        diagnosis = str(getattr(self.optimizer, "last_diagnosis", "") or "")
        if not diagnosis:
            return
        considered = str(getattr(self.optimizer, "last_considered", "") or "")
        path = self.root / "analysis" / f"generation-{self.generation}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"# Generation {self.generation} — {self.optimizer.name}\n\n{diagnosis}\n"
            + (f"\n## Considered\n\n{considered}\n" if considered else "")
        )

    def _append_rejections(self, rejections: Sequence[Rejection]) -> None:
        if not rejections:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "rejections.jsonl").open("a") as handle:
            for rejection in rejections:
                handle.write(
                    json.dumps(
                        {
                            "attempt": rejection.attempt,
                            "edit": rejection.edit,
                            "rationale": rejection.rationale,
                            "reasons": rejection.reasons,
                            "codes": rejection.codes(),
                        },
                        default=str,
                    )
                    + "\n"
                )

    def _exam_moved(self, candidate: Candidate) -> str | None:
        """Whether a candidate changed which items it would be graded on.

        The seed names the class of work. A candidate that rebinds a data node's Task
        or source chooses a different exam, and the held-out split is derived from the
        items a candidate actually has, so it would be graded on its own choice. A
        person's route to other items is a run's ``--source``.
        """
        if not self.population:
            return None
        before = _item_sources(self.population[0].state)
        after = _item_sources(candidate.state)
        changed = sorted(
            node for node in set(before) | set(after) if before.get(node) != after.get(node)
        )
        if not changed:
            return None
        return (
            f"E-exam: the candidate changed which items it would be graded on "
            f"({', '.join(changed)}); the class of work is the seed's, and a run points "
            f"a harness at other items with --source"
        )

    # ── item-ownership verification ──────────────────────────────────────

    def _expected_item_ids(self, state: dict[str, Any], part: str) -> tuple[set[str], bool]:
        """The item ids the loop expects a trial to grade, and whether item_scores is declared.

        Computed from the loop root and the seed's state — the loop-owned channel
        the candidate cannot write. Falls back silently when the mode cannot build
        or has no data nodes with items.

        When a loop-level manifest exists, val/test ids are read from it instead
        of calling ``partition().get(part)`` (which is always empty for val/test).
        """
        # When a manifest exists, read the part's ids directly from it (the
        # holdout audit's are its val ids: ``task.manifest_part_ids``).
        listed = (task_module.manifest_part_ids(self._parts_manifest, part)
                  if self._parts_manifest is not None else None)
        if listed is not None:
            manifest_ids = set(listed)
            # Still need to determine seed_declares_item_scores
            declares = False
            if state:
                try:
                    graph = self.mode_class(state=state).build()
                    for node in graph.nodes.values():
                        if node.kind == "data" and "item_scores" in node.produces:
                            declares = True
                            break
                except Exception:
                    pass
            return manifest_ids, declares

        if not state:
            return set(), False
        try:
            graph = self.mode_class(state=state).build()
        except Exception:
            return set(), False

        from refactory_dsh.nn.executor import full_items

        ids: set[str] = set()
        seed_declares_item_scores = False
        nodes = sorted(
            (node for node in graph.nodes.values() if node.kind == "data"),
            key=lambda node: node.id,
        )
        for node in nodes:
            if "item_scores" in node.produces:
                seed_declares_item_scores = True
            try:
                items = full_items(node, self.root)
            except Exception:
                continue
            if not items:
                continue
            selected = task_module.partition(items, self.config.holdout).get(part, [])
            for i, item in enumerate(selected):
                ids.add(str(item.get("id") or i))
        return ids, seed_declares_item_scores

    def _verify_item_set(
        self,
        expected_ids: set[str],
        seed_declares_item_scores: bool,
        trial_dir: Path,
        part: str,
        result: "RunResult",
    ) -> None:
        """Post-trial check: verify the actually-graded ids match the expected set.

        Three sub-checks:
        1. Subset integrity (backstop): the trial-dir subset.json must match the
           loop's part and holdout.
        2. Skip-loophole closure: when the seed declares item_scores, a trial that
           produces none is refused as E-exam.
        3. Item-set comparison: extra ids are refused; missing ids are refused only
           when they have no recorded per-item failure.
        """
        # 1. Subset integrity backstop
        subset = task_module.read_subset(trial_dir)
        expected_subset = {"part": part, "holdout": self.config.holdout}
        if subset and (subset.get("part") != expected_subset["part"]
                       or subset.get("holdout") != expected_subset["holdout"]):
            result.exit_ok = False
            result.error = (
                f"E-exam: subset.json was tampered with: expected {expected_subset}, "
                f"got {{'part': {subset.get('part')!r}, 'holdout': {subset.get('holdout')!r}}}"
            )
            return

        # 2. Skip-loophole closure
        has_item_scores = result.facts is not None and "item_scores" in result.facts
        item_scores = result.facts.get("item_scores") if has_item_scores else None
        if seed_declares_item_scores and not has_item_scores:
            result.exit_ok = False
            result.error = (
                "E-exam: the seed declares item_scores but the trial produced none; "
                "a trial may not avoid grading by producing no item_scores"
            )
            return

        # 3. Item-set comparison
        if not result.exit_ok or item_scores is None or not expected_ids:
            return

        graded_ids = {str(s["id"]) for s in item_scores if isinstance(s, dict) and "id" in s}

        extra = graded_ids - expected_ids
        if extra:
            result.exit_ok = False
            result.error = (
                f"E-exam: the trial graded ids outside the allowed set: "
                f"{sorted(extra)}"
            )
            return

        missing = expected_ids - graded_ids
        if missing:
            # A missing item that has a recorded per-item failure is accepted
            # (consistent with F4: failed items score 0 and keep their error).
            # A missing item with no failure record is refused.
            if result.error is not None:
                # The run had an error; missing items are attributed to it.
                return
            result.exit_ok = False
            result.error = (
                f"E-exam: expected ids were not graded and have no recorded failure: "
                f"{sorted(missing)}"
            )

    # ── step ───────────────────────────────────────────────────────────────
    def step(
        self, candidates: Sequence[Candidate], run: Run | None = None
    ) -> list[Candidate]:
        """Add offspring that are genuinely new points in the search."""
        known = {candidate.id for candidate in self.population}
        added: list[Candidate] = []
        for candidate in candidates:
            moved = self._exam_moved(candidate)
            if moved is not None:
                self.rejections.append(
                    Rejection(
                        attempt=0,
                        edit=_edit_record(candidate),
                        reasons=[moved],
                    )
                )
                continue
            if candidate.id in known:
                # The optimizer's loop refuses these now, so reaching here means a
                # strategy bypassed it. Record it rather than losing the slot in
                # silence: a batch that tried fewer points than it claimed is a
                # result nobody can trust.
                self.rejections.append(
                    Rejection(
                        attempt=0,
                        edit=_edit_record(candidate),
                        reasons=[
                            f"E-duplicate: the candidate's state already exists as "
                            f"candidate {candidate.id}, so its batch slot was lost"
                        ],
                    )
                )
                continue
            known.add(candidate.id)
            if candidate.parent and self._custom_strategy is None:
                # Whether its parent was the incumbent when it was added, for the
                # report's pool diversity; in the strategy state, so a resume keeps it.
                self._strategy_state.setdefault("parent_from_incumbent", {})[candidate.id] = (
                    candidate.parent == self._incumbent_id())
            self.population.append(candidate)
            added.append(candidate)
        self.generation += 1
        self._journal(
            "step",
            generation=self.generation,
            # Journaled, not derived: the adaptive k depends on it, so a resumed
            # run that guessed it would propose something else.
            refusal_pressure=round(self._refusal_pressure, 6),
        )
        self._write_state()
        self._progress("step", added=len(added), population=len(self.population))
        return added

    # ── scheduler ──────────────────────────────────────────────────────────
    def scheduler(self, *, evaluated: int = 0, proposed: int = 0, accepted: int = 0,
                  refused: int = 0) -> bool:
        """Close the epoch, and say whether there is another.

        The epoch is opened by the loop, not here: a resume has to know which
        iteration it is in, and an epoch that is only counted once it is finished
        cannot say that.
        """
        batch = [c for c in self._evaluated() if c.score is not None]
        scores = [c.score for c in batch if c.exit_ok and c.score is not None]
        summary = EpochSummary(
            epoch=self.epoch,
            evaluated=evaluated,
            best=max(scores) if scores else None,
            mean=(sum(scores) / len(scores)) if scores else None,
            improved=bool(self.loss.get("improved")),
            proposed=proposed,
            accepted=accepted,
            refused=refused,
            population=len(self.population),
            refused_total=len(self.rejections),
            structural_in_population=sum(1 for c in self.population if c.lineage_is_structural()),
        )
        self.epochs.append(summary)
        self._write_state()

        if self.config.val_gate and self.bound:
            # A budget a measurement found spent is why the run stops; with the
            # validation gate that reason decides whether the run is scored further.
            pass
        elif self.epoch >= self.config.epochs:
            self.bound = "epochs"
        elif self.evaluations >= self.config.max_evaluations:
            self.bound = "evaluations"
        elif self.config.max_rollouts is not None and self.rollouts >= self.config.max_rollouts:
            self.bound = "rollouts"
        if self.bound:
            # Journaled, not derived: the report states what bound the run, and a
            # run whose steps are separate processes reads that from the log.
            self._journal("bound", bound=self.bound)
        gate_fields: dict[str, Any] = {}
        if self.config.val_gate and self._custom_strategy is None:
            # A person reads this line: say what the gate decided, with its val scores,
            # rather than a train-score "improved" the gate may have rejected.
            gate = self._gate()
            gate_fields = {"gate": dict(getattr(self, "_epoch_gate", {}) or {}),
                           "val_best": gate.best_score if gate.best_id else None,
                           "val_current": gate.current_score if gate.current_id else None}
        self._progress(
            "scheduler",
            epoch=self.epoch,
            best=summary.best,
            mean=summary.mean,
            improved=summary.improved,
            bound=self.bound,
            another=not self.bound,
            **gate_fields,
        )
        return not self.bound

    # ── the loop ───────────────────────────────────────────────────────────
    # ── the steps a strategy may override ──────────────────────────────────
    def select_parents(self, n: int, *, by: str = "score") -> list[Candidate]:
        """The candidates a proposal is made against. Overridable, not fixed.

        Falls back to the population when nothing has been scored yet: a strategy
        that asks for parents before the first epoch must get the seed, not an empty
        list, or the only way to write a strategy is to know the order of the loop.

        A kept candidate is always among them, whether or not it is still in the top
        `n`. That is the whole of `keep`: without it, elite selection is a cull and a
        person's "follow this one up" would be silently dropped the moment a newer
        batch outscores it.
        """
        chosen = self.leaderboard(n)
        pinned = [
            candidate
            for candidate in self.leaderboard()
            if candidate.id in self.kept and candidate not in chosen
        ]
        if chosen or pinned:
            return chosen + pinned
        return [c for c in self.population[:n] if c.id not in self.rejected]

    def propose(self, run: Run, parents: Sequence[Candidate], n: int) -> list[Proposal]:
        """One round of proposals. The default delegates to the optimizer.

        A failed round is survived and recorded rather than raised. The run is the
        only place the evaluations so far are reported, so letting a model
        exception escape loses the whole run to a fact about the machine -- and a
        run that ends with no report looks, from outside, exactly like one that
        never started. What was refused and why is the optimizer's business;
        whether the optimizer answered at all is the loop's, and it is recorded in
        ``Optimizer.proposal_failures`` so the report can say which happened.
        """
        from refactory_dsh.budgets import BudgetExhausted

        self.optimizer.proposal_rounds += 1
        # Which attempt of this generation's proposal round this is: a retry after a
        # refusal reads another sample of the parent's trajectories.
        self._proposal_attempt += 1
        try:
            context = self._context()
            context.attempt = self._proposal_attempt
            return self.optimizer.propose(context, n, parents)
        except BudgetExhausted as exc:
            self.optimizer.proposal_failures.append(f"{type(exc).__name__}: {exc}")
            raise
        except Exception as exc:  # a proposer that is down is not a crash
            detail = f"{type(exc).__name__}: {exc}"
            self.optimizer.proposal_failures.append(detail)
            run.log(f"{self.optimizer.name} could not propose: {detail}")
            # In the progress stream too, where a person watching the run looks: a
            # round that never reached the model otherwise reads as one that did and
            # proposed nothing.
            self._progress("proposal_failed", optimizer=self.optimizer.name,
                           parent=self.strategy_decision.get("parent") or self._editing_parent,
                           attempt=self._proposal_attempt,
                           error=type(exc).__name__, detail=str(exc)[:500])
            return []

    def batch_size(self, run: Run) -> int:
        levels = self._round_levels()
        return len(levels) if levels else int(self.config.batch_size)

    def _round_levels(self) -> list[str]:
        """The edit levels this round asks for under ``portfolio``: every level, or the
        one a strategy decided for the round; none without a portfolio."""
        from refactory_dsh.optim.base import EDIT_LEVELS

        if not self.config.portfolio:
            return []
        kind = self.strategy_decision.get("edit_kind")
        return [kind] if kind in EDIT_LEVELS else list(EDIT_LEVELS)

    def install(
        self, run: Run, proposals: Sequence[Proposal], parents: Sequence[Candidate]
    ) -> list[Candidate]:
        """Turn proposals into candidates, asking again with the reasons.

        This is the validating loop the whole design rests on: apply, refuse with a
        coded reason, feed the reasons back, and only ever hand a candidate to
        evaluation. Overriding it is how a strategy changes how many times it asks.
        """
        accepted: list[Candidate] = []
        accepted_ids: set[str] = set()
        wanted = self.batch_size(run)
        # `k` adapts: one more round when the last batch was refused more than it was
        # accepted, because that is the signal that the optimizer is close to
        # something legal rather than out of ideas.
        cap = max(1, self.config.attempts) + (1 if self._refusal_pressure >= 0.5 else 0)
        rounds = 0
        before_all = len(self.rejections)
        for _attempt in range(1, cap + 1):
            rounds += 1
            if len(accepted) >= wanted:
                break
            before = len(self.rejections)
            for proposal in proposals:
                if len(accepted) >= wanted:
                    break
                # An op that is not in the catalogue at all is a different fact from
                # one that exists but is outside this run's permitted edits. Calling
                # both E-policy hides which fix the refusal is asking for. Each edit of
                # a compound proposal is checked, and a refusal names the one.
                refused = proposal_refusal(proposal, permitted=self.permitted_ops(),
                                           max_edits=self.config.max_edits_per_proposal)
                level = edit_level(proposal.steps)
                missing = [l for l in self._levels_wanted if l not in self._levels_done]
                if not refused and self._levels_wanted and level not in missing:
                    refused = [f"E-level: this round asks for one candidate per edit level "
                               f"({', '.join(self._levels_wanted)}); this proposal is a {level} "
                               f"change, which is "
                               + ("already covered" if level in self._levels_done
                                  else "not asked for")
                               + (f"; still missing: {', '.join(missing)}" if missing else "")]
                if refused:
                    self.rejections.append(Rejection(rounds, proposal.edit, refused))
                    continue
                parent = next((p for p in parents if p.id == proposal.parent), parents[0])
                outcome = run.attempt(proposal, parent)
                if isinstance(outcome, Candidate):
                    if outcome.id in accepted_ids:
                        # Two proposals in one batch that reach the same state. The
                        # population does not hold the first one yet, so the attempt
                        # above could not see it, and `step` would drop the second
                        # without saying why. Refused here, where the optimizer is
                        # still listening.
                        self.rejections.append(
                            Rejection(
                                attempt=rounds,
                                edit=proposal.edit,
                                reasons=[
                                    f"E-duplicate: this state is candidate {outcome.id}, "
                                    "already accepted from the same batch, so evaluating it "
                                    "again would spend a batch slot on a point already scored"
                                ],
                                rationale=proposal.rationale,
                            )
                        )
                        continue
                    accepted_ids.add(outcome.id)
                    accepted.append(outcome)
                    if self._levels_wanted:
                        self._levels_done.append(level)
                    # Its own words, for "Your earlier reasoning" in later requests.
                    # The diagnosis of the call that produced this proposal: a later
                    # retry in the same round replaces the optimizer's last one.
                    self._proposal_notes[outcome.id] = {
                        "rationale": proposal.rationale, "prediction": proposal.prediction,
                        "confidence": proposal.confidence,
                        "diagnosis": str(getattr(self.optimizer, "last_diagnosis", "") or ""),
                        "considered": str(getattr(self.optimizer, "last_considered", "") or ""),
                        **({"level": level} if self._levels_wanted else {})}
            # Another response is useful only if a validation round remains.
            if len(accepted) >= wanted or len(self.rejections) == before or _attempt >= cap:
                break
            proposals = self.propose(run, parents, wanted - len(accepted))
            if not proposals:
                break
        refused = len(self.rejections) - before_all
        self._refusal_pressure = refused / max(1, refused + len(accepted))
        self.attempts_used.append(rounds)
        return accepted

    def strategy(self, run: Run) -> None:
        """The default control flow: repeated epochs, each one step of the loop.

        `should_stop` is consulted first so a strategy can end for a reason of its
        own -- a plateau, a convergence -- without replacing the loop.

        When a custom strategy is attached (``_custom_strategy``), delegates
        to it — the **custom strategy seam**.
        """
        if self._custom_strategy is not None:
            return self._custom_strategy.strategy(run)
        self._search(run)

    #: How this strategy's search ends normally, on its own epoch limit: the run then
    #: scores the batch its last epoch proposed (``finish_strategy``). A strategy whose
    #: schedule is its own (a curriculum) adds the reason it stops with.
    normal_ends: tuple[str, ...] = ("epochs",)

    def _ended(self, run: Run | None) -> str:
        """Why the search ended: the bound the loop recorded, else the strategy's stop
        reason (a strategy that drives its own epochs stops with ``"epochs"`` whatever
        bound ended them)."""
        return str(self.bound or (run.stopped if run is not None else "") or "")

    def finish_strategy(self, run: Run) -> None:
        """After any built-in strategy: score what the last epoch proposed, when the
        search ended normally. Under ``val_gate`` that is ``_finish_gated_search`` (the
        last proposals scored and gated, then the post-training evaluation); without
        it, ``score_last_batch``. A custom strategy keeps its own ending."""
        if self._custom_strategy is not None:
            return
        if self.config.val_gate:
            self._finish_gated_search(run)
        else:
            self.score_last_batch(run)

    def score_last_batch(self, run: Run | None = None) -> None:
        """Without ``val_gate``: score the last epoch's proposals once the search ends
        on its epoch bound, as ``_finish_gated_search`` does under the gate.

        Each epoch scores the batch the one before proposed and then proposes the
        next, so a run of N epochs makes N proposal rounds; without this the last
        round was paid for and never scored. ``forward`` keeps the evaluation and
        rollout caps and charges what it runs. After any other bound (a person's
        stop, a spent evaluation, rollout, time or token budget) the proposals stay
        unscored and the report names them.

        Public because the loop is also a graph (``loop_mode``): its report node runs
        this, so the graph scores what the loop scores.
        """
        if self.config.val_gate:
            return
        if self._ended(run) in self.normal_ends and self._unevaluated():
            self.forward(epoch=self.epoch + 1)

    def _search(self, run: Run) -> None:
        """Epochs until a bound, a stop reason or a person's stop."""
        while True:
            if self._apply_control() == "stop":
                run.stop("a person asked the run to stop")
                return
            reason = self.should_stop(run)
            if reason:
                run.stop(reason)
                return
            if not self.epoch_step(run):
                return

    def _apply_control(self) -> str | None:
        """Honor what another process asked for, at the boundary.

        A pause blocks here until the request changes, so a person can hold a search
        between generations without killing it. The poll interval is configuration
        only so a test does not have to wait a second for each step.

        The population acts -- keep, reject, adopt -- are honored here too and then
        consumed, because this is the only place a request is safe (never inside an
        evaluation) and because one act is one act: a boundary that adopted the same
        winner every epoch would be a bug that looks like enthusiasm.
        """
        from refactory_dsh.control import Control, ONE_SHOT

        control = Control(self.root)
        paused_budget = None
        while True:
            request = control.read_request()
            action = str(request["action"]) if request is not None else ""
            if action != "pause" and paused_budget is not None:
                paused_budget.resume()
                paused_budget = None
            if request is None:
                return None
            if action == "pause":
                # Only the allocation's owner can stop its clock: a nested search
                # may share resources with enclosing work that is still running.
                if self.budget and self._owns_resource_budget and paused_budget is None:
                    self.budget.pause()
                    paused_budget = self.budget
                self._progress("paused")
                time.sleep(max(0.01, float(self.config.control_poll_s)))
                continue
            if action == "resume":
                # The pause it released is over; clearing keeps "absent means run"
                # the only reading of the file.
                control.clear()
                return None
            if action == "stop":
                self._progress("control", action="stop")
                return "stop"
            outcome = self._honor(request)
            if action in ONE_SHOT:
                control.clear()
            self._progress("control", **outcome)
            return None

    def _honor(self, request: dict[str, Any]) -> dict[str, Any]:
        """Carry out one population act, and say what it did.

        Every outcome is a fact rather than an exception. A run that dies because a
        person asked it to keep a candidate they mistyped has turned their control
        into a way to lose the search, and the file the request travels in has no
        way to say "that id was wrong" other than the journal.
        """
        action = str(request.get("action") or "")
        target = str(request.get("target") or "")
        lineage = bool(request.get("lineage"))
        known = {candidate.id for candidate in self.population}
        from refactory_dsh.control import request_identity

        outcome: dict[str, Any] = {"action": action, "request": request_identity(request)}
        if request.get("recovered"):
            outcome.update(recovered=True, requested_at=request.get("at"), status="applied")
        if target:
            outcome["target"] = target
        if lineage:
            outcome["lineage"] = True

        if action not in ("keep", "reject"):
            return self._adopt_now(str(request.get("as") or ""), outcome)

        targets = self._lineage_ids(target) if lineage else [target]
        outcome["targets"] = targets
        unknown = [candidate_id for candidate_id in targets if candidate_id not in known]
        applied: list[str] = []
        if action == "reject":
            for candidate_id in targets:
                if candidate_id in known and candidate_id not in self.rejected:
                    self.rejected.append(candidate_id)
                    applied.append(candidate_id)
                if candidate_id in self.kept:
                    self.kept.remove(candidate_id)
        else:
            # A keep is also the undo of a reject: a candidate is in and protected,
            # or out, and two flags saying both would be a state no reader can act on.
            for candidate_id in targets:
                if candidate_id in self.rejected:
                    self.rejected.remove(candidate_id)
                if candidate_id in known and candidate_id not in self.kept:
                    self.kept.append(candidate_id)
                    applied.append(candidate_id)

        outcome["applied"] = applied
        if unknown:
            outcome["unknown"] = unknown
        self._journal("control", **outcome, kept=list(self.kept), rejected=list(self.rejected))
        self._write_state()
        return outcome

    def _lineage_ids(self, target: str) -> list[str]:
        """The candidate and every descendant of it, in population order.

        "Stop spending budget around this branch" means the branch: rejecting one
        point while its descendants keep proposing from it is not what a person who
        read the lineage asked for.
        """
        children: dict[str, list[str]] = {}
        for candidate in self.population:
            if candidate.parent:
                children.setdefault(candidate.parent, []).append(candidate.id)
        found = [target]
        queue = [target]
        while queue:
            for child in children.get(queue.pop(0), ()):
                if child not in found:
                    found.append(child)
                    queue.append(child)
        return found

    def _adopt_now(self, as_name: str, outcome: dict[str, Any]) -> dict[str, Any]:
        """Write the current winner as a package, without ending the search.

        The winner is read from the journal the run is writing right now, which is
        the only artifact that carries a candidate's *state*, so this works while the
        run is live. An explicit live adoption audits that winner first; ordinary
        search evaluations remain training-only. A refusal is recorded as an event:
        adopting never overwrites, and
        a second adopt is a person's mistake rather than a reason to lose the search.
        """
        from refactory_dsh.adopt import adopt as adopt_winner

        # The identifier's package part, not `self.package`: that is the loaded
        # `Package` object, and its `str()` is a dataclass repr -- which became a
        # default name the filesystem refused with "File name too long" the first
        # time the panel adopted without naming a package.
        name = as_name or f"{self.mode_id.split('/', 1)[0]}_adopted"
        outcome["as"] = name
        try:
            self.audit_holdout()
            written = adopt_winner(
                self.root, as_name=name, project=self.project, mode=self.mode_id
            )
            outcome.update(
                package=written["package"],
                mode=written["mode"],
                candidate=written["candidate"],
                score=written["score"],
                package_root=written["package_root"],
            )
        except Exception as exc:  # a refusal is a fact, not the end of the run
            outcome["error"] = f"{type(exc).__name__}: {exc}"
        self.adoptions.append(outcome)
        self._journal("control", **outcome, kept=list(self.kept), rejected=list(self.rejected))
        self._write_state()
        return outcome

    def should_stop(self, run: Run) -> str | None:
        """A reason to stop before the bound, or ``None``. The bound is the default."""
        if run.stopped:
            return run.stopped
        if self.budget:
            from refactory_dsh.budgets import BudgetExhausted
            try:
                self.budget.checkpoint("work" if self._owns_resource_budget else None)
            except BudgetExhausted as exc:
                return str(exc)
        # A resumed completed epoch must not buy another proposal round. An
        # interrupted open epoch may still finish its bookkeeping at this limit.
        if not self._epoch_open and self.config.val_gate:
            early = self._early_stop_reason()
            if early:
                return early
        if not self._epoch_open and self.epoch >= self.config.epochs:
            return "epochs"
        if not self._epoch_open and self.evaluations >= self.config.max_evaluations:
            return "max_evaluations"
        return None

    def accepts(self, run: Run, candidate: Candidate) -> bool:
        """Whether a validated candidate is kept. A loser can still be archived."""
        return True

    def update(self, run: Run, candidate: Candidate) -> None:
        """Record a kept candidate. This is where an archive lives."""
        self.archive.add(candidate) if hasattr(self, "archive") else None

    def on_epoch_end(self, run: Run) -> None:
        """Called once per iteration, after the scheduler."""

    def open_epoch(self) -> None:
        """Open the epoch the following steps belong to. Idempotent.

        Public because the eight steps are also the nodes of a graph: an executor
        that runs one node per process has to open the epoch in the node that
        begins it, and reaching for the flag would make the graph depend on private
        state instead of on this method.
        """
        if self._epoch_open:
            return
        self.epoch += 1
        self._journal("epoch_open", epoch=self.epoch)
        self._epoch_open = True

    def close_epoch(self) -> None:
        """Close the epoch, so the next step opens the next one."""
        self._epoch_open = False
        self._journal("epoch_end", epoch=self.epoch)

    def strategy_state_dict(self) -> dict[str, Any]:
        if self._custom_strategy is not None:
            return self._custom_strategy.strategy_state_dict()
        return super().strategy_state_dict()

    def load_strategy_state(self, state: dict[str, Any]) -> None:
        if self._custom_strategy is not None:
            return self._custom_strategy.load_strategy_state(state)
        super().load_strategy_state(state)
        self._val_gate = None  # rebuilt from the restored state by _gate()

    def _gate(self) -> ValidationGate:
        """The default loop's validation gate, restored from the strategy state.

        Its state, with the ids it accepted and rejected, is the ``val_gate`` entry of
        the loop's strategy state, journaled after every decision.
        """
        if self._val_gate is None:
            saved = dict(self._strategy_state.get("val_gate") or {})
            gate = ValidationGate(seed=self.config.seed)
            gate.load_state_dict(saved)
            self._gate_accepted = list(saved.get("accepted") or [])
            self._gate_rejected = list(saved.get("rejected") or [])
            self._gate_rounds = [list(r) for r in saved.get("rounds") or []]
            self._val_gate = gate
        return self._val_gate

    def _gate_population(self, run: Run) -> None:
        """Judge each candidate evaluated on train that the gate has not judged.

        The seed is rolled out on val and accepted. Every later candidate the gate has
        not judged is one round (``ValidationGate.judge_round``): each is judged against
        the same bar, the gate's current candidate as the round starts, so the order
        siblings were proposed in decides nothing. Of those whose val score strictly
        beats it, the highest is accepted and becomes current; a tie goes to the higher
        train score, then to the one proposed first. The rest are rejected, one that
        beat the bar but lost to a sibling included (the report marks it shadowed). A
        candidate whose train run failed is rolled out on val too and judged the same
        way. A refused rollout (a spent rollout budget) judges what was scored before
        it and leaves the rest unjudged; an unjudged candidate is not a parent or a
        winner. SkillOpt gates one candidate per epoch, where both rules agree.

        The val rollouts are rolled out first, up to ``max_concurrent_evaluations`` at
        once (``_prefetch_val``), and the decisions are then taken from the gate's cache.
        """
        from refactory_dsh.outerloop import RoundStopped

        gate = self._gate()
        judged = set(self._gate_accepted) | set(self._gate_rejected)
        pending = [c for c in self.population if c.evaluated and c.id not in judged]
        cached = len(gate.cache)
        self._prefetch_val(run, gate, pending)
        if len(gate.cache) != cached:
            # Journaled now, so a run killed before the decisions rolls none again.
            self._store_gate_state()
            self._save_strategy_state()
        if pending and gate.best_id is None:
            try:
                gate.start(run, pending[0])
            except OuterLoopError as error:
                run.log(f"val gate stopped: {error}")
                return
            self._strategy_state["gate_start_epoch"] = self.epoch
            self._gate_accepted.append(pending[0].id)
            self._gate_rounds.append([pending[0].id])
            self._store_gate_state()
            self._save_strategy_state()
            pending = pending[1:]
        if not pending:
            return
        try:
            accepted, rejected = gate.judge_round(run, pending)
        except RoundStopped as stop:
            accepted, rejected = stop.accepted, stop.rejected
            run.log(f"val gate stopped: {stop.error}")
        decided = {c.id for c in accepted} | {c.id for c in rejected}
        for candidate in pending:
            if candidate.id in decided:
                (self._gate_accepted if candidate in accepted else self._gate_rejected).append(
                    candidate.id)
        if decided:
            self._gate_rounds.append([c.id for c in pending if c.id in decided])
        self._store_gate_state()
        self._save_strategy_state()

    def _prefetch_val(self, run: Run, gate: ValidationGate, pending: Sequence[Candidate]) -> None:
        """Roll the candidates the gate will judge out on val, in waves of up to
        ``max_concurrent_evaluations`` (``ValidationGate.prefetch``), so the in-order
        decisions read cached scores. Each val rollout depends only on its candidate
        and the run's seed, so the scores are the ones the decisions would have rolled.

        A wave holds only rollouts the in-order gating would also have made (with
        ``max_rollouts``, see ``_wave_size``); whatever it leaves, or a wave that did
        not score every candidate, is rolled out by the decisions, one at a time.
        """
        manifest = self._parts_manifest or {}
        each = len(manifest.get(task_module.VAL) or ())
        todo = [c for c in pending if c.id not in gate.cache]
        while len(todo) > 1:
            size = self._wave_size(len(todo), each)
            if size < 2 or (self.config.max_rollouts is not None
                            and self.rollouts >= self.config.max_rollouts):
                return
            wave, todo = todo[:size], todo[size:]
            gate.prefetch(run, wave)
            if any(c.id not in gate.cache for c in wave):
                return

    def best_score_fields(self, winner: Candidate) -> dict[str, Any]:
        """The report's ``best_score`` for ``winner``: with ``val_gate`` its val score,
        said so, with its train score (none when its train run failed) beside it;
        otherwise its train score, as before."""
        if not self.config.val_gate or self._custom_strategy is not None:
            return {"best_score": winner.score}
        gate = self._gate()
        val = gate.cache.get(winner.id, gate.best_score if winner.id == gate.best_id else None)
        return {"best_score": val, "best_score_basis": "val: the validation gate's score",
                "best_train_score": winner.score}

    def _gated_parents(self) -> list[Candidate]:
        """The parents under ``val_gate``, for every optimizer and strategy.

        ``single`` (SkillOpt's rule): the gate's current candidate (the last one
        accepted, which is also the best on val), whether or not its train run
        succeeded, with any accepted candidate a person kept. ``pool``: the pool
        (``_pool``, the incumbent first), with any judged candidate a person kept; a
        population optimizer or parent picker (``archive``, ``evolutionary``,
        ``islands``, ``map_elites``: ``ParentSelector.population``) gets every judged
        candidate instead (``_judged_ranked``) and applies its own rule to them (niche
        champions, island bests, its archive's own limit, tournaments), so a new
        structure ranked below the pool keeps its niche. The optimizer chooses by val
        (``ProposalContext.val_scores``), so which candidate is edited never depends on
        a train score from another round's seed or sample.
        """
        gate = self._gate()
        if self.selection_mode() == "pool":
            pool = self._judged_ranked() if self._population_policy() else self._pool()
            judged = set(self._gate_accepted) | set(self._gate_rejected)
            return pool + [c for c in self._evaluated() if c.id in self.kept and c.id in judged
                           and c not in pool and c.id not in self.rejected]
        board = self.leaderboard()
        current = next((c for c in board if c.id == gate.current_id), None)
        first = [current] if current is not None else board[:1]
        return first + [c for c in board if c.id in self.kept and c not in first]

    def _population_policy(self) -> bool:
        """Whether this run's parents are picked by a population policy (the optimizer
        itself, or its parent picker), which gets every judged candidate."""
        policy = getattr(self.optimizer, "picker", self.optimizer)
        return bool(getattr(policy, "population", False))

    def selection_mode(self) -> str:
        """How the validation average selects in this run (``LoopConfig.selection``):
        ``single`` or ``pool`` under the default loop's ``val_gate``; empty without it
        (train scores select) or with a custom strategy (which keeps its own gate).
        An explicit setting wins; then the one the run's gate state recorded (so a
        reader that rebuilds a finished run as a plain ``Loop`` reads the run's own
        rule); then ``default_selection``."""
        if not self.config.val_gate or self._custom_strategy is not None:
            return ""
        if self.config.selection:
            return self.config.selection
        stored = (self._strategy_state.get("val_gate") or {}).get("selection")
        if stored in ("single", "pool"):
            return str(stored)
        picker = getattr(getattr(self.optimizer, "picker", None), "name", "")
        return default_selection(self.name, self.config.optimizer, picker)

    def _pool(self) -> list[Candidate]:
        """The pool under ``selection`` ``pool``: the judged candidates with the
        ``pool_size`` best val scores, plus any tied with the last of them, minus what a
        person rejected. Ranked by val score, a tie by the higher train score (a failed
        train run lowest), then the earlier candidate; the incumbent (the gate's current
        candidate) is first, since it changes only on a strictly higher val score.

        Rebuilt from the gate's cache each time it is read: a val score never changes,
        so the pool is the same set a pool updated round by round would hold, and a
        resumed run holds it again without storing it.
        """
        gate = self._gate()
        ranked = self._top(self._judged_ranked())
        incumbent = next((c for c in ranked if c.id == gate.current_id), None)
        if incumbent is not None:
            ranked = [incumbent] + [c for c in ranked if c is not incumbent]
        return ranked

    def _top(self, ranked: list[Candidate]) -> list[Candidate]:
        """``ranked`` (best val first) cut to the pool: the ``pool_size`` best val
        scores, plus any tied with the last of them."""
        gate = self._gate()
        size = max(1, int(self.config.pool_size))
        if len(ranked) > size:
            edge = gate.cache[ranked[size - 1].id]
            ranked = [c for c in ranked if gate.cache[c.id] >= edge]
        return ranked

    def _pool_entered(self) -> set[str]:
        """The judged candidates that were in the pool after the gating that judged them.

        A val score never changes and later gatings only add candidates, so a candidate
        out of the pool now either entered it then and was pushed out since (``left
        pool``) or never entered it. Rebuilt gating by gating from the gate's cache and
        the ids each gating judged (``rounds`` in the gate state). Every judged
        candidate counts, one a person rejected later included: it held its place when
        the others were judged, so a later rejection does not turn them into ``left
        pool``. A run recorded before ``rounds`` existed groups by proposal epoch.
        """
        gate = self._gate()
        key = (len(self.population), len(self._gate_accepted), len(self._gate_rejected),
               len(gate.cache), len(self._gate_rounds), int(self.config.pool_size))
        cached = getattr(self, "_pool_entered_memo", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        judged = set(self._gate_accepted) | set(self._gate_rejected)
        by_id = {c.id: c for c in self.population}
        order = {c.id: index for index, c in enumerate(self.population)}
        rounds = [[cid for cid in r if cid in by_id and cid in gate.cache]
                  for r in self._gate_rounds]
        if not self._gate_rounds:
            members = [c for c in self._evaluated() if c.id in judged and c.id in gate.cache]
            rounds, last = [], object()
            for candidate in members:
                epoch = (self.proposals.get(candidate.id) or {}).get("epoch")
                if rounds and epoch is not None and epoch == last:
                    rounds[-1].append(candidate.id)
                else:
                    rounds.append([candidate.id])
                last = epoch
        entered: set[str] = set()
        seen: list[Candidate] = []
        for ids in rounds:
            seen.extend(by_id[cid] for cid in ids)
            top = {c.id for c in self._top(sorted(
                seen, key=lambda c: (-gate.cache[c.id], -_train_key(c), order[c.id])))}
            entered |= {cid for cid in ids if cid in top}
        self._pool_entered_memo = (key, entered)
        return entered

    def _judged_ranked(self) -> list[Candidate]:
        """Every judged candidate a person has not rejected (each has a val score),
        ranked as the pool ranks: val score, a tie by train score (a failed train run
        lowest), then the earlier; the incumbent first."""
        gate = self._gate()
        judged = set(self._gate_accepted) | set(self._gate_rejected)
        order = {c.id: index for index, c in enumerate(self.population)}
        ranked = sorted(
            (c for c in self._evaluated()
             if c.id in judged and c.id in gate.cache and c.id not in self.rejected),
            key=lambda c: (-gate.cache[c.id], -_train_key(c), order[c.id]))
        incumbent = next((c for c in ranked if c.id == gate.current_id), None)
        if incumbent is not None:
            ranked = [incumbent] + [c for c in ranked if c is not incumbent]
        return ranked

    def _incumbent_id(self) -> str | None:
        """The candidate a parent counts as "the incumbent" for (``selection`` report):
        the gate's current candidate under ``val_gate``, else the train leader."""
        if self.config.val_gate and self._custom_strategy is None:
            return self._gate().current_id
        board = self.leaderboard(1)
        return board[0].id if board else None

    def _store_gate_state(self) -> None:
        """Put the gate's state and decisions in the strategy state (not journaled)."""
        self._strategy_state["val_gate"] = {
            **self._gate().state_dict(),
            "accepted": list(self._gate_accepted), "rejected": list(self._gate_rejected),
            "rounds": [list(r) for r in self._gate_rounds],
            # Which rule ranked the winner, so adopt names the one the report names.
            "selection": self.selection_mode(), "pool_size": int(self.config.pool_size),
        }

    # ── val_gate: SkillOpt's early stop and post-training evaluation ────────
    #: The bounds a search ends on normally, as SkillOpt's ``_TRAINED``. After any
    #: other (a person's stop, a spent evaluation, rollout, time or token budget)
    #: the last proposals are not scored and the post-training evaluation is
    #: skipped, with the reason recorded.
    _TRAINED = ("epochs", "perfect_val", "patience_exhausted")

    def _count_stale_epoch(self) -> None:
        """SkillOpt's stale-epoch count, once per epoch after its gating: an epoch
        whose gating did not raise the best val score above the best seen so far is
        stale; one that did resets the count.

        The epoch that started the gate gated the starting population, which is
        SkillOpt's initial val score: it sets the best the count measures from and
        is not counted. Each later epoch gates the batch the one before proposed,
        so it is counted as SkillOpt counts the epoch that proposed that batch.
        """
        if not (self.config.early_stop_patience or self.config.early_stop_at_full_val):
            return
        state = self._strategy_state
        if state.get("stale_counted_epoch") == self.epoch:
            return
        gate = self._gate()
        state["stale_counted_epoch"] = self.epoch
        if state.get("gate_start_epoch", self.epoch) == self.epoch:
            state["early_stop_best"] = gate.best_score
            return
        best = float(state.get("early_stop_best", gate.best_score))
        if gate.best_score > best + 1e-9:
            state["early_stop_best"] = gate.best_score
            state["stale_epochs"] = 0
        else:
            state["stale_epochs"] = int(state.get("stale_epochs", 0)) + 1

    def _early_stop_reason(self) -> str:
        """SkillOpt's early stop: ``perfect_val`` once the best val score is 1.0 (with
        ``early_stop_at_full_val``), ``patience_exhausted`` after
        ``early_stop_patience`` stale epochs in a row; empty otherwise.

        Checked only once an epoch that gated a proposed batch has been counted, as
        SkillOpt checks at an epoch's end: a seed already at 1.0 on val still gets
        one batch proposed and gated."""
        state = self._strategy_state
        counted = state.get("stale_counted_epoch")
        if counted is None or counted == state.get("gate_start_epoch"):
            return ""
        if self.config.early_stop_at_full_val and self._gate().best_score >= 1.0 - 1e-9:
            return "perfect_val"
        patience = self.config.early_stop_patience
        if patience > 0 and int(state.get("stale_epochs", 0)) >= patience:
            return "patience_exhausted"
        return ""

    def _final(self) -> FinalEvaluation:
        """The post-training evaluation, restored from the strategy state."""
        final = FinalEvaluation(
            record=lambda step_type, **fields: self._journal(
                "final_evaluation", step_type=step_type, **fields),
            label="default loop",
        )
        final.load_state_dict(self._strategy_state.get("final_evaluation") or {})
        return final

    def _finish_gated_search(self, run: Run) -> None:
        """What SkillOpt does once training ends normally, under ``val_gate``.

        The last epoch's proposals are evaluated on train and gated, so a run of N
        epochs gates every batch it proposed. Then, with a test part,
        ``FinalEvaluation`` runs: the final candidate's val check, then the seed,
        the best and the final candidate on test with the run seed. Each phase is
        saved, so a resumed run continues where it stopped; a search resumed with
        more epochs evaluates again. A search stopped by a time or token budget that
        held a validation reserve back (``validation_reserve_s``/``_tokens``) leaves
        its last proposals unscored and runs the final evaluation on the reserve. A
        search that ended any other way is not scored further, and the evaluation
        records why it was skipped.
        """
        from refactory_dsh.budgets import RESERVE_RETAINED

        ended = self._ended(run)
        reserved = ended.endswith(RESERVE_RETAINED)
        state = self._strategy_state
        if (state.get("final_evaluation") or {}).get("search_epoch") != self.epoch:
            state.pop("final_evaluation", None)
        final = self._final()

        def save() -> None:
            self._store_gate_state()
            state["final_evaluation"] = {**final.state_dict(), "search_epoch": self.epoch}
            self._save_strategy_state()

        if final.result.get("status") == "done":
            return
        if ended not in self._TRAINED + self.normal_ends and not reserved:
            final.result = {"status": "skipped",
                            "reason": f"the search did not finish: it ended on {ended!r}"}
            save()
            return
        if self._unevaluated() and not reserved:
            self.forward(epoch=self.epoch + 1)
            self._gate_population(run)
            # The last batch's scores and gate decisions, which no later epoch's
            # scheduler writes: without this state.json shows them "not gated".
            self._write_state()
        if not run.has_part(task_module.TEST):
            final.skip(run, "the parts manifest has no 'test' part", save)
            return
        final.run(run, self._gate(), seed=self.config.seed, save=save)

    def final_evaluation(self) -> dict[str, Any]:
        """The post-training evaluation's figures, for the report (``val_gate``)."""
        result = self._final().result
        if not result and self.bound:
            # The search stopped before the gated search could finish (a spent budget
            # raised out of a measurement).
            result = {"status": "skipped",
                      "reason": f"the search did not finish: it ended on {self.bound!r}"}
        report: dict[str, Any] = {
            "best_selection": self._gate().best_score if self._gate().best_id else None,
            "test_eval": result.get("status") or "not_run",
        }
        if result.get("reason"):
            report["test_eval_reason"] = result["reason"]
        for key in ("final_selection", "baseline_test", "best_test", "final_test",
                    "best_test_delta", "final_test_delta"):
            report[key] = result.get(key)
        return report

    def _save_strategy_state(self) -> None:
        """Journal the strategy's current state unconditionally.

        Called after close_epoch and by run.save_strategy_state(). Journals
        even when empty, because skipping empty dicts would let a resume
        restore a stale non-empty state from an earlier epoch.
        """
        strategy_state = self.strategy_state_dict()
        self._journal("strategy_state", state=strategy_state)

    def epoch_step(self, run: Run | None = None) -> bool:
        """One iteration, and whether there is another.

        The epoch is opened here and closed by the scheduler, and both are journaled.
        A run that died inside an iteration therefore resumes *into* it rather than
        opening it again, which would count the same epoch twice.
        """
        self.open_epoch()
        evaluated = self.forward()
        early = ""
        before = (len(self._gate_accepted), len(self._gate_rejected))
        if self.config.val_gate:
            self._gate_population(run or Run(_loop=self))
            # What this epoch's gating decided, for the epoch's progress line.
            self._epoch_gate = {"kept": len(self._gate_accepted) - before[0],
                                "rejected": len(self._gate_rejected) - before[1]}
            # SkillOpt's early stop, checked right after the epoch's gating so a
            # stopping run proposes nothing it will not score.
            self._count_stale_epoch()
            early = self._early_stop_reason()
        self.loss_step()
        self.backward()
        candidates: list[Candidate] = []
        rejections: list[Rejection] = []
        if not early:
            candidates, rejections = self.optimizer_step(run)
            kept = self.step([c for c in candidates if self.accepts(run, c)], run=run)
            for candidate in kept:
                self.update(run, candidate)
        another = self.scheduler(
            evaluated=evaluated, proposed=len(candidates) + len(rejections),
            accepted=len(candidates), refused=len(rejections),
        )
        if early:
            # A budget a measurement found spent stays the reason (the scheduler kept
            # it); an early stop replaces only the epoch bound.
            if self.bound in ("", "epochs"):
                self.bound = early
                self._journal("bound", bound=early)
            another = False
        self.on_epoch_end(run)
        self.close_epoch()
        self._save_strategy_state()
        return another

    # ── report ─────────────────────────────────────────────────────────────
    def leaderboard(self, limit: int | None = None) -> list[Candidate]:
        """Everything the run kept, ranked, minus what a person rejected. With
        ``val_gate`` on: only what the validation gate accepted (one whose train run
        failed included), ranked by val score.

        The exclusion is here rather than in the population because rejection is a
        selection fact, not a historical one: `state.json` still records what the
        candidate scored, and every reader that *chooses* -- the proposer's parents,
        the report's winner, the held-out audit, the atomic adopt -- goes through
        this one ranking, so a dismissed branch cannot come back as the best answer.
        """
        if self.config.val_gate and self.selection_mode() == "pool":
            # The pool, the incumbent first: what every choice under ``pool`` reads.
            pool = self._pool()
            return pool[:limit] if limit else pool
        if self.config.val_gate:
            # The candidates the gate accepted, a failed train run included, ranked
            # by their val score: what the gate selects on, not the train score.
            gate = self._gate()  # restores the decisions on a resumed or replayed run
            accepted = set(self._gate_accepted)
            ordered = sorted(
                [c for c in self._evaluated() if c.id in accepted and c.id not in self.rejected],
                key=lambda c: (-gate.cache.get(c.id, 0.0), c.id),
            )
            return ordered[:limit] if limit else ordered
        ordered = sorted(
            [
                c
                for c in self._evaluated()
                if c.exit_ok and c.score is not None and c.id not in self.rejected
            ],
            key=lambda c: (-(c.score or 0.0), c.id),
        )
        return ordered[:limit] if limit else ordered

    def _evaluated(self) -> list[Candidate]:
        return [c for c in self.population if c.evaluated]

    def _unevaluated(self) -> list[str]:
        """Candidates the run kept but never ran, because it stopped at its bound.

        The pipeline proposes the next epoch's batch at the end of an epoch. A run that
        ends normally (its epoch bound, or a strategy's own ``normal_ends``) scores that
        batch before it reports (``finish_strategy``); one stopped by any other bound
        holds a batch that was installed and never evaluated. Named in the report so it cannot be read as "all of this ran", and
        so a model-driven proposal round whose output was dropped is visible.
        """
        return [c.id for c in self.population if not c.evaluated]

    def _row(self, candidate: Candidate, *, detail: bool = False) -> dict[str, Any]:
        """One candidate as a reader sees it: both scores, and what its size cost.

        The leaderboard, the lineage and the records the optimizer reads all go
        through here, so none of them reports a score without the cost that was
        taken off it. ``detail``: the gate's pool labels (``_gate_decision``), for the
        report, ``state.json`` and the log only; never for anything the optimizer reads.
        """
        nodes = node_count(candidate.state)
        row = {
            **candidate.describe(),
            "nodes": nodes,
            "size_cost": round(self.config.parsimony * nodes, 6),
        }
        gate = self._gate_decision(candidate.id, detail=detail)
        if gate is not None:
            row["gate"] = gate
        return row

    def _gate_decision(self, candidate_id: str, *, detail: bool = False) -> str | None:
        """What the validation gate decided about a candidate: ``current`` (the
        incumbent), ``kept`` (accepted as the incumbent when judged, since replaced),
        ``rejected`` or ``not gated``. ``None`` without ``val_gate``. Never a val score.

        That is all the optimizer reads, under ``pool`` as under ``single``: which
        candidates the pool holds follows their val ranking, so it tells no more than
        ``single`` does. With ``detail`` (the report, ``state.json``, the log), under
        ``pool`` a candidate not current says where it stands in the pool: ``in pool``
        (in it now), ``left pool`` (in it after the gating that judged it, pushed out
        since), ``rejected`` (never in it, or rejected by a person) or ``below pool``
        (never in it, but a population policy still breeds from every judged
        candidate)."""
        if not self.config.val_gate or self._custom_strategy is not None:
            return None
        gate = self._gate()
        if candidate_id == gate.current_id:
            return "current"
        if detail and self.selection_mode() == "pool":
            judged = set(self._gate_accepted) | set(self._gate_rejected)
            if candidate_id not in judged:
                return "not gated"
            if candidate_id in self.rejected:
                return "rejected"
            if any(c.id == candidate_id for c in self._pool()):
                return "in pool"
            if candidate_id in self._pool_entered():
                return "left pool"
            return "below pool" if self._population_policy() else "rejected"
        if candidate_id in self._gate_accepted:
            return "kept"
        if candidate_id in self._gate_rejected:
            return "rejected"
        return "not gated"

    def scoring(self) -> dict[str, Any]:
        """How a candidate's score is built, said where the numbers are recorded: the
        evaluator's raw number when there is no size cost (``parsimony`` 0), else that
        number minus the size cost."""
        if not self.config.parsimony:
            return {
                "compared": "raw",
                "parsimony": self.config.parsimony,
                "detail": "candidates are ranked by raw_score: parsimony is 0, so there "
                          "is no size cost and score equals raw_score",
            }
        return {
            "compared": "penalized",
            "parsimony": self.config.parsimony,
            "detail": (
                "candidates are ranked by raw_score - parsimony * nodes; every row "
                "carries both, so a candidate that won on merit is told from one "
                "that won on size"
            ),
        }

    def report(self) -> dict[str, Any]:
        # The audit runs before the payload is built: it needs the winner, and the
        # report is where its numbers are read.
        board = self.leaderboard()

        # G37: a custom strategy may declare its own best candidate.
        strategy = self._custom_strategy
        strategy_best = None
        strategy_error = None
        if strategy is not None:
            # A run the strategy says failed (G32) reports that reason first.
            strategy_error = strategy.run_error() or None
            try:
                strategy_best = strategy.best_candidate(self.population)
            except Exception as e:
                strategy_error = strategy_error or str(e)
        # With the validation gate the leaderboard is the accepted candidates a person
        # has not rejected, ranked by val: its first is the winner, as adopt reads it.
        # A custom strategy's own figures (``OuterLoop.report``), e.g. SkillOpt's
        # post-training test scores; empty for the library's strategies.
        strategy_report: dict[str, Any] = {}
        if strategy is None and self.config.val_gate:
            strategy_report = self.final_evaluation()
        if strategy is not None:
            try:
                strategy_report = dict(strategy.report(Run(_loop=self)) or {})
            except Exception as e:
                strategy_error = strategy_error or f"the strategy's report failed: {e}"
        best = strategy_best if strategy_best is not None else (board[0] if board else None)

        audit = self.audit_holdout(winner=best)
        # When the winner has explicit val/test parts, use audit_parts for a
        # like-with-like overfit comparison (both unseen by the optimizer).
        gated = strategy is None and self.config.val_gate
        if best is not None:
            parts_audit = self.audit_parts(best)
            if parts_audit is not None:
                audit = parts_audit
                if gated:
                    # Adopting reads holdout.json: it checks the audit the report shows.
                    self.holdout_audit = parts_audit
                    self._write(HOLDOUT_FILE, parts_audit)

        # G32: a run whose every proposal round failed is not a success.
        proposer = self._proposer_health()

        exit_ok = (
            best is not None and audit.get("winner") == best.id
            and audit.get("status") in ("measured", "not_applicable", "unmeasured")
            and not audit.get("overfit")
            and not proposer.get("all_failed")
            and strategy_error is None
        )

        # G33: when a custom strategy is set, report its name and epoch count.
        optimizer_name = self.search_name()
        epochs_completed = getattr(strategy, '_epoch', self.epoch) if strategy is not None else self.epoch

        payload = {
            "exit_ok": exit_ok,
            "error": (
                None if exit_ok else
                strategy_error if strategy_error else
                proposer.get("finding", "every optimizer call failed") if proposer.get("all_failed") else
                "no successfully evaluated candidate is eligible to win" if best is None else
                f"the winner overfit ({_holdout_line(audit)})" if audit.get("overfit") else
                str(audit.get("note") or "the required held-out audit is unavailable")
            ),
            "mode": self.mode_id,
            "bound": self.bound,
            "epochs": self.config.epochs,
            "epochs_completed": epochs_completed,
            "batch_size": self.config.batch_size,
            "optimizer": optimizer_name,
            "evaluator": self.config.evaluator,
            "evaluations": self.evaluations,
            "measurements": self.measurements,
            "resolved_profile": self.config.resolved_profile,
            # Which of the seed's first measurements were read from --seed-results
            # instead of run, so a reader never mistakes a reused score for a fresh one.
            "seed_results": ({"file": str(self.config.seed_results),
                              "source": (self._seed_cache or {}).get("source"),
                              "candidate": (self._seed_cache or {}).get("candidate"),
                              "parts_reused": sorted(self._seed_cache_used),
                              "parts_measured": sorted(self._seed_cache_missed - self._seed_cache_used)}
                             if self.config.seed_results else None),
            "resources": self.budget.snapshot() if self.budget else None,
            "population": len(self.population),
            "unevaluated": self._unevaluated(),
            "best": self._row(best, detail=True) if best else None,
            "best_score": best.score if best else None,
            "raw_best_score": best.raw_score if best else None,
            "seed_score": self.population[0].score if self.population and self.population[0].exit_ok else None,
            "seed_raw_score": self.population[0].raw_score if self.population and self.population[0].exit_ok else None,
            "improvement": {"status": "observed" if best and self.population and self.population[0].score is not None and best.score is not None and best.score > self.population[0].score else "none",
                            "interpretation": "sample means are observations, not proof of a stochastic improvement"},
            "scoring": self.scoring(),
            "leaderboard": [self._row(c, detail=True) for c in board],
            "failed_candidates": [self._row(c, detail=True) for c in self._evaluated()
                                  if not c.exit_ok],
            "kept": list(self.kept),
            "rejected": list(self.rejected),
            "adoptions": list(self.adoptions),
            "lineage": self._lineage(best),
            "structural_wins": self._structural_wins(board),
            "accepted_op_mix": self._op_mix(),
            "structural": self._structural_stats(),
            "rejection_codes": self._rejection_codes(),
            "rejection_reasons": [r for rej in self.rejections for r in rej.reasons][:40],
            "knob_attribution": self.gradient.get("knob_attribution", {}),
            "ports": self.mode.describe()["ports"],
            "memory": self.mode.describe()["memory"],
            "cost": self._cost(),
            "efficiency": self._efficiency(),
            "dominant_refusal": self._dominant_refusal(),
            "convergence": self._convergence(),
            "knob_coverage": {
                "unreachable": self.unreachable_knobs(),
                "detail": "declared and movable, but the selected optimizer cannot "
                          "produce a different value for it",
            },
            "proposal_outcomes": self._proposal_outcomes(),
            "epochs_summary": [asdict(summary) for summary in self.epochs],
            "proposer": proposer,
            "holdout": audit,
            "strategy_report": strategy_report,
            **({"portfolio": self._portfolio_report()} if self.config.portfolio else {}),
            "selection": self._selection_report(),
        }
        if gated and best is not None:
            payload.update(self.best_score_fields(best))
        self._write("report.json", payload)
        self._write_text("report.md", self._render_report(payload))
        self._write("manifest.json", progress_file.artifact_manifest(self.root))
        self._progress(
            "report",
            best_score=payload["best_score"],
            raw_best_score=payload["raw_best_score"],
            overfit=bool(audit.get("overfit")),
            evaluations=self.evaluations,
            bound=self.bound,
        )
        return payload

    # ── the held-out audit ─────────────────────────────────────────────────
    def audit_holdout(self, *, winner: "Candidate | None" = None) -> dict[str, Any]:
        """Re-evaluate the winner on the items no candidate saw, once.

        The search evaluated every candidate on the train part, so this is the only
        place the harness meets the held-out part. The train score is the winner's own
        recorded score and is not penalised: the size cost is a search preference,
        while this is a measurement of the same mode on items it was not tuned
        against. A winner that only works on its training items is reported rather
        than rewarded.

        When ``winner`` is provided (e.g. by a custom strategy's
        ``best_candidate``), it overrides the leaderboard's first entry.
        """
        if winner is None:
            board = self.leaderboard()
            winner = board[0] if board else None
        winner_id = winner.id if winner is not None else None
        if self.holdout_audit is not None and self.holdout_audit.get("winner") == winner_id:
            return self.holdout_audit

        items, data_nodes = self._audit_items(
            winner.state if winner is not None else {}
        )
        manifest = self._parts_manifest
        if manifest is not None:
            # The parts manifest decides the parts: the search ran its train ids,
            # the audit runs its val ids, and its test ids are never touched here.
            # The fraction then only says whether to audit: --holdout 0 turns it off.
            held = (task_module.manifest_part_ids(manifest, task_module.HOLDOUT) or ()
                    if self.config.holdout > 0 else ())
            instances: dict[str, int | None] = {
                task_module.TRAIN: len(task_module.manifest_part_ids(manifest, task_module.TRAIN) or ()),
                task_module.HOLDOUT: len(held),
            }
        else:
            instances = (
                dict(task_module.counts(items, self.config.holdout))
                if items
                else {task_module.TRAIN: None, task_module.HOLDOUT: None}
            )
        train = winner.raw_score if winner is not None else None
        holdout: float | None = None
        note = ""
        status = "unavailable"
        if winner is None:
            note = "the run has no successfully evaluated winner to audit"
        elif not data_nodes:
            status = "not_applicable"
            note = (
                "the mode has no data node, so it iterates no items and there is "
                "nothing to hold out"
            )
        elif manifest is not None and instances[task_module.HOLDOUT] == 0:
            status = "not_applicable"
            note = (
                "the run's holdout is 0, so no holdout was scored"
                if self.config.holdout <= 0 else
                "the parts manifest names no val items, so there is no holdout: the "
                "audit never runs the manifest's test items"
            )
        elif manifest is not None and self._audit_split().get("independent") is False:
            # The audit's items are the val part the search selected on: a train-vs-val
            # gap there is no overfit verdict, so none is given and nothing is rolled out.
            # The val-vs-test audit (``audit_parts``) replaces this when there is a test
            # part and the winner was scored on it.
            status = "not_applicable"
            note = (
                "no independent check was possible: the audit's held-out items are the val "
                "part the search selected on, and "
                + ("the winner was not scored on the test part"
                   if task_module.TEST in manifest else "the run has no test part")
            )
        elif not items and manifest is None:
            note = (
                "the items this mode iterates could not be read outside a candidate's "
                "run directory, so no holdout was scored"
            )
        elif instances[task_module.HOLDOUT] == 0:
            status = "not_applicable"
            note = (
                "the selection holds one item or none, so it cannot be split; the "
                "winner was searched on every item it has and no holdout exists"
            )
        else:
            holdout, note = self._holdout_score(winner)
            status = "measured" if holdout is not None else "failed"

        gap = (
            train - holdout
            if train is not None and holdout is not None
            else None
        )
        threshold = OVERFIT_THRESHOLD
        se = (gap_standard_error(train, int(instances.get(task_module.TRAIN) or 0),
                                 holdout, int(instances.get(task_module.HOLDOUT) or 0))
              if gap is not None else None)
        seed_winner = (winner is not None and bool(self.population)
                       and winner.id == self.population[0].id)
        overfit = None if gap is None else (False if seed_winner else gap > threshold)
        audit: dict[str, Any] = {
            "status": status,
            "train": train,
            "holdout": holdout,
            "gap": gap,
            "overfit": overfit,
            "threshold": threshold,
            "standard_error": se,
            "threshold_basis": (f"fixed {OVERFIT_THRESHOLD}; standard_error (from the part "
                                "sizes) is for information"),
            "instances": instances,
            "winner": winner.id if winner is not None else None,
            "measurements": self.holdout_measurements,
            **self._audit_split(),
        }
        if gap is not None and seed_winner:
            note = SEED_WINNER_NOTE
        if note:
            audit["note"] = note
        self.holdout_audit = audit
        self._write(HOLDOUT_FILE, audit)
        self._journal("holdout", **audit)
        self._progress(
            "holdout",
            train=train,
            holdout=holdout,
            gap=gap,
            overfit=audit["overfit"],
        )
        return audit

    def audit_parts(self, winner: Candidate) -> dict[str, Any] | None:
        """Overfit check comparing val vs test scores (like with like).

        When the winner has been evaluated on both 'val' and 'test' parts,
        the overfit check compares the latest val score vs the latest test
        score.  Both are unseen by the optimizer, so the comparison is
        *like with like*.

        Returns ``None`` when the winner does not have both val and test.
        """
        candidate_results = self._part_results.get(winner.id, {})
        val_parts = candidate_results.get("val", [])
        test_parts = candidate_results.get("test", [])
        if not val_parts or not test_parts:
            return None

        val_score = val_parts[-1].get("score")
        test_score = test_parts[-1].get("score")

        if val_score is None or test_score is None:
            return {
                "status": "failed",
                "val": val_score,
                "test": test_score,
                "winner": winner.id,
                "note": self._missing_part_note(
                    [(part, parts[-1]) for part, parts, score in
                     (("val", val_parts, val_score), ("test", test_parts, test_score))
                     if score is None]),
            }

        # The gap the optimization added: the winner's val-test gap minus the seed's on
        # the same parts (the seed's val score and its test score, e.g. FinalEvaluation's
        # baseline_test). A val part easier than the test part shows in the seed too.
        gap = val_score - test_score
        seed = self.population[0] if self.population else None
        seed_results = self._part_results.get(seed.id, {}) if seed is not None else {}
        seed_val = (seed_results.get("val") or [{}])[-1].get("score")
        seed_test = (seed_results.get("test") or [{}])[-1].get("score")
        seed_gap = (seed_val - seed_test if seed_val is not None and seed_test is not None
                    else None)
        manifest = self._parts_manifest or {}
        se = gap_standard_error(val_score, len(manifest.get(task_module.VAL) or ()),
                                test_score, len(manifest.get(task_module.TEST) or ()))
        audit: dict[str, Any] = {
            "status": "measured",
            "val": val_score,
            "test": test_score,
            "gap": gap,
            "seed_gap": seed_gap,
            "threshold": OVERFIT_THRESHOLD,
            "standard_error": se,
            "threshold_basis": (f"fixed {OVERFIT_THRESHOLD} on the gap the search added; "
                                "standard_error (from the val and test part sizes) is for "
                                "information"),
            "winner": winner.id,
            "split": task_module.TEST, "independent": True,
            "split_note": "the test part, which nothing selected on",
        }
        if seed is not None and winner.id == seed.id:
            audit.update(added_gap=0.0, overfit=False, note=SEED_WINNER_NOTE)
            return audit
        if seed_gap is None:
            # Without the seed's own gap the search's part of it cannot be told apart:
            # neither a pass nor a failure.
            audit.update(status="unmeasured", added_gap=None, overfit=None,
                         note=("unmeasured: the seed's "
                               + ("test" if seed_val is not None else "val")
                               + " score is missing, so the gap the search added is unknown"))
            return audit
        added = gap - seed_gap
        audit.update(added_gap=added, overfit=added > OVERFIT_THRESHOLD)
        return audit

    def _audit_split(self) -> dict[str, Any]:
        """Which items the holdout audit runs, and whether they are independent of the
        selection. With a parts manifest the audit's held-out items are its val part
        (``task.manifest_part_ids``): under ``val_gate`` that is the part the search
        selected on, so the audit is not independent, and the report says so rather
        than present it as a check on unseen items. Without a manifest they are the
        Task's ``--holdout`` fraction, which nothing selects on."""
        if self._parts_manifest is None:
            return {"split": task_module.HOLDOUT, "independent": True,
                    "split_note": "the --holdout fraction of the Task's items, which nothing "
                                  "selected on"}
        if self._custom_strategy is not None:
            return {"split": task_module.VAL, "independent": None,
                    "split_note": f"the val part; the {self._custom_strategy.name} strategy "
                                  "keeps its own selection, which may read it"}
        if self.config.val_gate:
            return {"split": task_module.VAL, "independent": False,
                    "split_note": "the val part, which the search selected on, so this is "
                                  "not an independent check; add a test part for one"}
        return {"split": task_module.VAL, "independent": True,
                "split_note": "the val part, which nothing selected on (train scores selected)"}

    def _missing_part_note(self, missing: Sequence[tuple[str, dict[str, Any]]]) -> str:
        """Why the winner's val or test score is missing: the part and its cause, and,
        when a time or token budget ran out during it, which budget and the reserve
        that keeps the end-of-run scoring its time."""
        causes = []
        for part, occasion in missing:
            error = str(occasion.get("error") or "")
            if "elapsed-time budget exhausted" in error:
                cause = (f"the elapsed-time budget (max_elapsed_s {self.config.max_elapsed_s:g}s) "
                         f"ran out during the winner's {part} scoring"
                         if self.config.max_elapsed_s else
                         f"the elapsed-time budget ran out during the winner's {part} scoring")
                if not self.config.validation_reserve_s:
                    cause += "; validation_reserve_s keeps time for the end-of-run scoring"
            elif "token budget exhausted" in error:
                cause = f"the token budget ran out during the winner's {part} scoring"
                if not self.config.validation_reserve_tokens:
                    cause += "; validation_reserve_tokens keeps tokens for the end-of-run scoring"
            else:
                cause = f"the winner's {part} scoring failed" + (f": {error}" if error else "")
            causes.append(cause)
        parts = " and ".join(part for part, _ in missing)
        return f"{parts} score is missing: " + "; ".join(causes)

    def _holdout_score(self, winner: Candidate) -> tuple[float | None, str]:
        """Run the winner on the held-out part, in a directory of its own.

        Its own directory rather than the candidate's, so the score the search
        recorded stays what it was and a reader can still see both runs.
        """
        workdir = self.root / "holdout" / winner.id
        try:
            workdir.mkdir(parents=True, exist_ok=True)
            # The manifest travels with the audit, so its loader selects the
            # manifest's holdout ids rather than splitting every item the Task has.
            manifest = self.root / task_module.MANIFEST
            if self._parts_manifest is not None and manifest.exists():
                shutil.copyfile(manifest, workdir / task_module.MANIFEST)
            result, self.holdout_measurements = self._measure(
                winner, workdir, task_module.HOLDOUT, seed=self.config.seed)
        except Exception as exc:  # a failed audit is recorded, not raised
            return None, f"the holdout run failed: {type(exc).__name__}: {exc}"
        if not result.exit_ok:
            return None, f"the holdout run failed: {result.error}"
        return float(result.score), ""

    def _audit_items(self, state: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
        """The whole item selection, and how many data nodes the mode has.

        The whole selection is loaded rather than the part this run evaluates, so the
        counts do not depend on which part the call is made from. The first data node
        whose items can be read from the run root supplies them; a node bound to a
        source this directory does not have is skipped, and a mode with no data node
        answers with no items and no nodes, which are two different facts.
        """
        if not state:
            return [], 0
        try:
            graph = self.mode_class(state=state).build()
        except Exception:  # a mode that cannot build has no items to count
            return [], 0
        from refactory_dsh.nn.executor import full_items

        nodes = sorted(
            (node for node in graph.nodes.values() if node.kind == "data"),
            key=lambda node: node.id,
        )
        for node in nodes:
            try:
                items = full_items(node, self.root)
            except Exception:  # a data node bound to a source this directory lacks
                continue
            if items:
                return items, len(nodes)
        return [], len(nodes)

    def _structural_wins(self, board: list[Candidate]) -> list[dict[str, Any]]:
        """Structural candidates that scored above the parent they were bred from.

        A structural candidate that merely ran is not a win: counting those made
        every structural edit that did not crash look like progress. Under the default
        loop's ``val_gate`` "scored above" is the gate's val score (the selection's
        fitness), not the train score, and each win carries both val scores. This is
        the report's, never the proposer's: the request shows no val score.
        """
        by_id = {c.id: c for c in self.population}
        wins: list[dict[str, Any]] = []
        gated = self.config.val_gate and self._custom_strategy is None
        cache = self._gate().cache if gated else {}
        for candidate in board:
            parent = by_id.get(candidate.parent or "")
            if not candidate.is_structural() or parent is None:
                continue
            if gated:
                val, parent_val = cache.get(candidate.id), cache.get(parent.id)
                if val is not None and parent_val is not None and val > parent_val:
                    wins.append({**self._row(candidate, detail=True), "val": val,
                                 "parent_val": parent_val})
            elif (candidate.exit_ok and candidate.score is not None and parent.exit_ok
                  and parent.score is not None and candidate.score > parent.score):
                wins.append(self._row(candidate, detail=True))
        return wins

    def _lineage(self, candidate: Candidate | None) -> list[dict[str, Any]]:
        if candidate is None:
            return []
        by_id = {c.id: c for c in self.population}
        chain: list[dict[str, Any]] = []
        seen: set[str] = set()
        while candidate is not None and candidate.id not in seen:
            seen.add(candidate.id)
            chain.append(self._row(candidate, detail=True))
            candidate = by_id.get(candidate.parent or "")
        return list(reversed(chain))

    def _proposal_outcomes(self) -> list[dict[str, Any]]:
        """Every accepted proposal, with what actually happened to the score.

        Both the compared score and the evaluator's number are recorded, with their
        deltas against the parent, so a proposal that grew the graph shows the size
        cost beside the gain it bought.
        """
        by_id = {c.id: c for c in self.population}
        out: list[dict[str, Any]] = []
        for candidate_id, record in sorted(self.proposals.items()):
            candidate = by_id.get(candidate_id)
            if candidate is None or not candidate.evaluated:
                continue
            parent = by_id.get(candidate.parent or "")
            parent_score = parent.score if parent is not None else None
            parent_raw = parent.raw_score if parent is not None else None
            delta = (
                round((candidate.score or 0.0) - parent_score, 6)
                if parent_score is not None and candidate.score is not None
                else None
            )
            raw_delta = (
                round((candidate.raw_score or 0.0) - parent_raw, 6)
                if parent_raw is not None and candidate.raw_score is not None
                else None
            )
            out.append(
                {
                    **record,
                    "score": candidate.score,
                    "raw_score": candidate.raw_score,
                    "parent_score": parent_score,
                    "parent_raw_score": parent_raw,
                    "delta": delta,
                    "raw_delta": raw_delta,
                    "exit_ok": candidate.exit_ok,
                    "error": candidate.error,
                }
            )
        return out

    def _render_report(self, payload: dict[str, Any]) -> str:
        lines = [
            f"# Outer loop — {payload['mode']}",
            "",
            f"Optimizer `{payload['optimizer']}`, {payload['epochs_completed']} epoch(s) "
            f"of {payload['epochs']}, batch {payload['batch_size']}, "
            f"{payload['evaluations']} inner run(s). "
            f"Stopped because **{payload['bound']}** bound the run.",
            "",
            f"Seed score **{payload['seed_score']}** → best **{payload['best_score']}**"
            f" (raw {payload['seed_raw_score']} → {payload['raw_best_score']}). "
            f"The compared score is the raw one minus "
            f"{payload['scoring']['parsimony']} per node of the candidate's graph.",
            *(["", f"The seed's first measurement on {', '.join(payload['seed_results']['parts_reused'])} "
               f"was reused from {payload['seed_results']['source']} (--seed-results), not run again."]
              if (payload.get("seed_results") or {}).get("parts_reused") else []),
            "",
            "## Held-out audit",
            "",
            _holdout_sentence(payload.get("holdout") or {}),
            "",
            *_strategy_report_section(payload.get("strategy_report") or {}),
            *_portfolio_section(payload.get("portfolio") or {}),
            *_selection_section(payload.get("selection") or {}),
            "## Leaderboard",
            "",
            _md_table(
                ["id", "parent", "score", "raw", "nodes", "structural", "edits", "ok"],
                [
                    [
                        c["id"], c.get("parent"), c.get("score"), c.get("raw_score"),
                        c.get("nodes"), c.get("structural"),
                        json.dumps(c.get("edits"), default=str), c.get("exit_ok"),
                    ]
                    for c in payload["leaderboard"]
                ],
            ),
            "",
            "## Structural candidates that beat their parent",
            "",
            *(["On the validation gate's val score (`val` against `parent val`); `score` is "
               "the train score.", ""] if (payload.get("selection") or {}).get("setting")
              in ("single", "pool") else []),
            _md_table(
                ["id", "score", "raw", "nodes",
                 *(["val", "parent val"] if any("val" in c for c in payload["structural_wins"])
                   else []), "edits"],
                [[c["id"], c.get("score"), c.get("raw_score"), c.get("nodes"),
                  *([c.get("val"), c.get("parent_val")]
                    if any("val" in w for w in payload["structural_wins"]) else []),
                  json.dumps(c.get("edits"), default=str)]
                 for c in payload["structural_wins"]],
            ),
            "",
            "## Why proposals were refused",
            "",
            payload["dominant_refusal"]["finding"] or "_No dominant refusal._",
            "",
            *(
                [
                    "## Did the proposer answer?",
                    "",
                    payload["proposer"]["finding"],
                    "",
                    *["- " + " ".join(failure.split())[:300]  # one bullet each, even multi-line
                      for failure in payload["proposer"]["failures"]],
                    "",
                ]
                if payload["proposer"]["failure_count"]
                else []
            ),
            "## Did the search run out of ideas?",
            "",
            payload["convergence"]["detail"] or "_No: proposals were still producing "
            "new points._",
            "",
            f"Attempts per candidate that could run: "
            f"{payload['efficiency']['attempts_per_candidate']} "
            f"(effective k {payload['efficiency']['effective_k']}).",
            "",
            "## Refusals by rule",
            "",
            _md_table(
                ["rule", "count"], [[code, count] for code, count in payload["rejection_codes"].items()]
            ),
            "",
            "## Accepted edits by operation",
            "",
            _md_table(
                ["op", "count"],
                [[op, count] for op, count in payload["accepted_op_mix"].items()],
            ),
            "",
            f"Structural yield: {payload['structural']['structural_candidates']} "
            f"candidate(s) from {payload['structural']['structural_refusals']} "
            f"refusal(s) — {payload['structural']['structural_yield']}.",
            "",
            "## What the mode publishes",
            "",
            _md_table(
                ["port", "path", "media type", "description"],
                [[p["name"], p["path"], p["media_type"], p.get("description", "")]
                 for p in payload["ports"]],
            ),
            "",
            "## What the mode may remember",
            "",
            _md_table(
                ["namespace", "kind", "retention", "description"],
                [[m["namespace"], m["kind"], m["retention"], m.get("description", "")]
                 for m in payload["memory"]],
            ),
            "",
            "## Inner runs",
            "",
            f"{payload['evaluations']} candidate(s) evaluated, total inner wall clock "
            f"{payload['cost'].get('total_s')}s, and "
            f"{payload['cost'].get('inner_model_calls')} model call(s) made inside the "
            "graphs themselves (an inserted agent node costs these).",
            "",
            *(
                [
                    f"{len(payload['unevaluated'])} candidate(s) were proposed for the "
                    "next epoch and never evaluated, because the run stopped at its "
                    "bound: "
                    + ", ".join(f"`{candidate}`" for candidate in payload["unevaluated"])
                    + ".",
                    "",
                ]
                if payload.get("unevaluated")
                else []
            ),
            "## Proposal versus outcome",
            "",
            _md_table(
                ["id", "origin", "parent_score", "score", "delta", "raw_score",
                 "raw_delta", "rationale"],
                [
                    [
                        p["id"], p.get("origin"), p.get("parent_score"), p.get("score"),
                        p.get("delta"), p.get("raw_score"), p.get("raw_delta"),
                        (p.get("rationale") or "")[:80],
                    ]
                    for p in payload["proposal_outcomes"]
                ],
            ),
            "",
            *_considered_lines(payload["proposal_outcomes"]),
            "## Lineage of the best",
            "",
            _md_table(
                ["id", "score", "raw", "nodes", "edits"],
                [[c["id"], c.get("score"), c.get("raw_score"), c.get("nodes"),
                  json.dumps(c.get("edits"), default=str)]
                 for c in payload["lineage"]],
            ),
            "",
        ]
        return "\n".join(lines)

    # ── context ────────────────────────────────────────────────────────────
    def _context(self) -> ProposalContext:
        # Under ``pool`` the whole pool: every parent a proposal may name is shown.
        board = self.leaderboard(None if self.selection_mode() == "pool" else 8)
        described, graph_note = self._describe_parent(board)
        sandbox = self._sandbox_answer()
        return ProposalContext(
            epoch=self.epoch,
            generation=self.generation,
            mode_name=self.mode_id,
            best_score=(board[0].score if board else None),
            mean_score=self.loss.get("mean"),
            mean_over=len(self.loss.get("scores") or ()),
            knobs=[
                {"name": name, **spec,
                 "reachable": self._reachable(name, spec, described["knob_states"])}
                for name, spec in described["knobs"].items()
            ],
            nodes=described["nodes"],
            edges=described["edges"],
            node_states=described.get("node_states", []),
            # What each node has before it, so a prompt names a fact that is there.
            available=described.get("available", {}),
            wiring_warnings=list(described.get("warnings") or ()),
            # Candidates run headless: no person answers a gate while they run.
            person_gates=False,
            leaderboard=[self._row(c) for c in self._request_order(board)],
            history=self._history(),
            tried=self._tried(),
            elite=int(self.config.elite),
            history_truncated=self._history_truncated(),
            prior_runs=run_history.prior_summaries(
                self.root.parent, limit=self.config.prior_runs, exclude=self.root
            ),
            traces=[self.gradient] if self.gradient else [],
            catalogue=[op for op in nn_edits.catalogue(self.config.edit_preset)
                       if not self.config.allowed_ops or op["op"] in self.config.allowed_ops],
            insertable=[node.to_state() for node in self.mode.insertable()],
            frozen=list(self.config.frozen),
            protected=list(self.protected),
            rejections=summarise_reasons(
                [
                    reason
                    for rejection in self.rejections[-8:]
                    for reason in rejection.reasons
                ]
            ),
            preset=self.config.edit_preset,
            strategy={**dict(self.strategy_decision), **({"search_contract": self.config.resolved_profile["contract"]} if self.config.resolved_profile.get("contract") else {})},
            # Read now, not at construction: the point of steering a search is that a
            # note added while it runs reaches the next proposal.
            steering=self._steering_block(),
            item_records=self._parent_train_records(board),
            parent_item_records=self._parent_train_records(board),
            graph_note=graph_note,
            edit_pairs=self._edit_pairs(board),
            reasoning=self._reasoning_entries(board),
            max_edits=int(self.config.max_edits_per_proposal),
            parsimony=float(self.config.parsimony),
            code_nodes=self._code_nodes_offered(sandbox),
            sandbox=sandbox,
            score_is_item_mean=bool(self._score_is_item_mean),
            val_items=self._gate_val_items(),
            seed_knob_values=self._seed_knob_values(),
            parent=self._edited_id(board),
            val_scores=self._val_scores(),
            selection=self.selection_mode(),
            population_parents=(self.selection_mode() == "pool" and self._population_policy()),
            edit_levels=[l for l in self._levels_wanted if l not in self._levels_done],
            strategy_name=str(getattr(self._custom_strategy, "name", "") or self.name),
        )

    def _request_order(self, board: list[Candidate]) -> list[Candidate]:
        """The request's "Recent results" rows in an order no val score decides.

        Under ``val_gate`` the leaderboard is ranked by val score, so listing it as
        ranked would tell the proposer the val ranking it must not see. The rows go
        instead: the gate's current candidate first (it is marked ``current`` anyway),
        then by train score (a failed train run lowest), then the earlier candidate.
        Without the gate the leaderboard is the train ranking already and is kept."""
        if not self.config.val_gate or self._custom_strategy is not None:
            return list(board)
        current = self._gate().current_id
        order = {c.id: index for index, c in enumerate(self.population)}
        return sorted(board, key=lambda c: (c.id != current, -_train_key(c),
                                            order.get(c.id, len(order))))

    def _val_scores(self) -> dict[str, float]:
        """The gate's val score per judged candidate under ``val_gate`` (parent
        policies rank by it); empty otherwise."""
        if not self.config.val_gate or self._custom_strategy is not None:
            return {}
        return dict(self._gate().cache)

    def _root_state(self) -> dict[str, Any] | None:
        """The root candidate's state (the seed every lineage starts from); ``None``
        before there is one."""
        root = next((c for c in self.population if not c.parent), None)
        return root.state if root is not None else None

    def _seed_knob_values(self) -> dict[str, Any]:
        """The root candidate's knob values, which every candidate's edit lineage
        starts from; empty before there is one."""
        root = next((c for c in self.population if not c.parent), None)
        knobs = (root.state or {}).get("knobs") if root is not None else None
        return {str(name): entry.get("value") for name, entry in (knobs or {}).items()
                if isinstance(entry, dict)}

    def _gate_val_items(self) -> int:
        """How many validation items the gate's score is the mean of, for the request's
        gate note: the parts manifest's val part, under the default loop's
        ``val_gate`` when every score so far is the items' mean; 0 otherwise."""
        if (not self.config.val_gate or self._custom_strategy is not None
                or not self._score_is_item_mean or self._parts_manifest is None):
            return 0
        return len(task_module.manifest_part_ids(self._parts_manifest, task_module.VAL) or ())

    def _edited_id(self, board: list[Candidate]) -> str | None:
        parent_id = self.strategy_decision.get("parent") or self._editing_parent
        return str(parent_id) if parent_id else (board[0].id if board else None)

    def _edit_pairs(self, board: list[Candidate]) -> list[dict[str, Any]]:
        """The latest evaluated children of the edited candidate, each with its parent's
        train records; or, when it has none yet, the edited candidate against its own
        parent. Train records only.

        The latest children are every evaluated child proposed in the same epoch as the
        last one (its round: all siblings, however many a round makes -- a portfolio
        makes one per edit level whatever ``batch_size`` says); a child with no proposal
        record counts in the last ``batch_size``."""
        edited = self._edited_id(board)
        by_id = {c.id: c for c in self.population}
        if edited is None or edited not in by_id:
            return []
        children = [c for c in self.population if c.parent == edited and c.evaluated]
        chosen = self._last_round(children) or [by_id[edited]]
        pairs = []
        for child in chosen:
            parent = by_id.get(child.parent or "")
            parent_records, same = (self._records_like(parent.id, child.id) if parent
                                    else ([], None))
            pairs.append({
                "child": child.id, "parent": parent.id if parent else None,
                "edit": child.own_edits(), "gate": self._gate_decision(child.id),
                "child_records": list(self._train_item_records.get(child.id, [])),
                "parent_records": parent_records, "same_seed": same,
            })
        return pairs

    def _last_round(self, children: Sequence[Candidate]) -> list[Candidate]:
        """The children proposed in the epoch of the last of them (population order),
        in order; without a proposal epoch, the last ``batch_size``."""
        if not children:
            return []
        epoch = (self.proposals.get(children[-1].id) or {}).get("epoch")
        if epoch is None:
            return list(children[-max(1, int(self.config.batch_size)):])
        return [c for c in children if (self.proposals.get(c.id) or {}).get("epoch") == epoch]

    def _records_like(self, parent_id: str, child_id: str) -> tuple[list[dict[str, Any]], bool]:
        """The parent's train records to compare with the child's: from the child's
        epoch (the same seed) when the parent was scored then, else its latest; and
        whether the two share a seed. Without ``val_gate`` every train rollout uses the
        run's seed, so they always do."""
        latest = list(self._train_item_records.get(parent_id, []))
        if not self.config.val_gate:
            return latest, True
        at = self._records_epoch.get(child_id)
        history = self._records_by_epoch.get(parent_id, {})
        if at is not None and at in history:
            return list(history[at]), True
        return latest, at is not None and self._records_epoch.get(parent_id) == at

    def _reasoning_entries(self, board: list[Candidate]) -> list[dict[str, Any]]:
        """The optimizer's own reasoning about recent candidates -- the line from the
        seed to the edited candidate, and the latest refused children -- beside what
        happened to each. Train scores, train item errors and gate decisions only."""
        from refactory_dsh.optim.base import REASONING_ENTRIES, edit_change, item_errors

        by_id = {c.id: c for c in self.population}
        line: list[Candidate] = []
        node = by_id.get(self._edited_id(board) or "")
        while node is not None and node.parent:
            line.append(node)
            node = by_id.get(node.parent)
        line.reverse()
        rejected = [c for c in self.population if c.evaluated and c not in line
                    and self._gate_decision(c.id) == "rejected"]
        chosen = (line + rejected[-REASONING_ENTRIES:])
        chosen.sort(key=lambda c: self.population.index(c))
        entries = []
        for candidate in chosen[-REASONING_ENTRIES:]:
            record = self.proposals.get(candidate.id) or {}
            if record.get("origin") != "optimizer":
                continue
            parent = by_id.get(candidate.parent or "")
            parent_records, same = (self._records_like(parent.id, candidate.id)
                                    if parent is not None else ([], None))
            records = (list(self._train_item_records.get(candidate.id) or ())
                       if candidate.evaluated else [])
            change = (edit_change(parent_records, records)
                      if parent is not None and candidate.evaluated else None)
            entries.append({
                "id": candidate.id, "epoch": record.get("epoch"),
                # What this edit added: its new or changed steps judged against its
                # parent, less what the parent already had.
                "warnings": [w for w in _wiring_warnings(
                                 candidate.state, parent.state if parent is not None else None)
                             if parent is None
                             or w not in _wiring_warnings(parent.state, parent.state)],
                "edit": candidate.own_edits(), "rationale": record.get("rationale"),
                "prediction": record.get("prediction"), "diagnosis": record.get("diagnosis"),
                "considered": record.get("considered"),
                "gate": self._gate_decision(candidate.id), "score": candidate.score,
                "same_seed": same,
                "fixed": len(change["fixed"]) if change and change["shared"] else None,
                "broken": len(change["broken"]) if change and change["shared"] else None,
                "errors": item_errors(records), "items": len(records),
            })
        return entries

    def _describe_parent(self, board: list[Candidate]) -> tuple[dict[str, Any], str]:
        """The graph and knobs of the candidate this step edits, and a note when not.

        The parent is chosen as for its records (``_parent_train_records``): the
        strategy's, else the one ``optimizer_step`` selected, else the leader. The
        seed's description would show a graph the edits no longer apply to once a
        structural candidate is the parent. The seed is described only when no
        parent is known, or when the parent's state cannot be built -- and then the
        note says so, in the prompt and in the run's progress, rather than passing
        the seed's graph off as the parent's.
        """
        by_id = {c.id: c for c in self.population}
        parent_id = self.strategy_decision.get("parent") or self._editing_parent
        parent = by_id.get(str(parent_id)) if parent_id else None
        if parent is None and board:
            parent = board[0]
        def described(mode: Any) -> dict[str, Any]:
            # The nodes in full too, in node-state form, for a structure edit to copy.
            # And its knob states (spec + value): a node an earlier edit added brings
            # its own knobs, which the seed does not declare.
            whole = mode.state_dict()
            nodes = (whole.get("graph") or {}).get("nodes") or {}
            return {**mode.describe(), "node_states": [dict(state) for state in nodes.values()],
                    "knob_states": dict(whole.get("knobs") or {}),
                    "available": available_before(whole.get("graph") or {}),
                    "warnings": _wiring_warnings(whole, self._root_state())}

        if parent is None:
            return described(self.mode), ""
        try:
            return described(self.mode_class(state=parent.state)), ""
        except Exception as exc:
            note = (f"the parent candidate {parent.id} could not be built "
                    f"({type(exc).__name__}: {exc}); the seed's pipeline is shown instead")
            self._progress("warning", candidate=parent.id, warning=note)
            return described(self.mode), note

    def _parent_train_records(self, board: list[Candidate]) -> list[dict[str, Any]]:
        """Train records for the parent the strategy selected for editing.

        Priority:
        1. ``strategy_decision["parent"]`` when a strategy named one, else the
           parent ``optimizer_step`` chose this step from ``select_parents(1)``.
        2. The most recently installed candidate's parent — what the optimizer
           actually bred from in the previous round.  This is a better signal
           than ``board[0]`` when the optimizer picks a non-best parent.
        3. ``board[0]`` — leaderboard best as a last resort.
        """
        # Prefer the parent the strategy explicitly selected.
        parent_id = self.strategy_decision.get("parent") or self._editing_parent
        if parent_id and parent_id in self._train_item_records:
            return list(self._train_item_records[parent_id])
        # Fall back to the most recently installed candidate's parent:
        # what the optimizer actually bred from.
        for c in reversed(self.population):
            if c.parent and c.parent in self._train_item_records:
                return list(self._train_item_records[c.parent])
        # Last resort: leaderboard best.
        if not board:
            return []
        return list(self._train_item_records.get(board[0].id, []))

    def _steering_block(self) -> str:
        """The run's intent ledger as the proposer reads it, empty when there is none."""
        from refactory_dsh.steering import Ledger

        return Ledger(self.root).render()

    def _reachable(self, name: str, spec: dict[str, Any],
                   knob_states: Mapping[str, Any] | None = None) -> bool:
        """Whether the selected optimizer can move this knob at all.

        The knob is read from ``knob_states`` -- the edited candidate's own knobs --
        before the seed's: a node added by an earlier edit (a new agent's
        ``<id>__system_prompt``, a code node's ``<id>__code``, a duplicate's or a
        split's copies) declares knobs the seed never had.
        """
        entry = (knob_states or {}).get(name) or self.mode.state_dict()["knobs"].get(name)
        if entry is None:
            raise KeyError(f"knob {name!r} is declared neither by the edited candidate "
                           "nor by the mode")
        knob = Knob.from_spec(entry["spec"])
        if knob.frozen or name in self.config.frozen:
            return False
        return self.optimizer.can_move(knob, spec.get("value"))

    def unreachable_knobs(self) -> list[str]:
        """Movable knobs the selected optimizer cannot move, named.

        Reported rather than tolerated: a search over four of five declared knobs
        looks exactly like a search over five.
        """
        described = self.mode.describe()["knobs"]
        return sorted(
            name
            for name, spec in described.items()
            if not spec.get("frozen")
            and name not in self.config.frozen
            and not self._reachable(name, spec)
        )

    def state_dict(self) -> dict[str, Any]:
        """The search's own state: where it is, and what it is searching.

        Distinct from a mode's `state_dict`, which is the thing being searched. A
        strategy that wants to inspect the search gets this.
        """
        return {
            "mode": self.mode_id,
            "epoch": self.epoch,
            "generation": self.generation,
            "evaluations": self.evaluations,
            "bound": self.bound,
            "evaluator": self.config.evaluator,
            "optimizer": self.optimizer.name,
            "seed": self.mode.state_dict(),
            "scoring": self.scoring(),
            "population": [self._row(candidate) for candidate in self.population],
        }

    def _tried(self) -> list[str]:
        """Every state id that exists, so a duplicate can be refused."""
        return sorted(candidate.id for candidate in self.population)

    def _history(self) -> list[dict[str, Any]]:
        """The evaluated candidates, newest last, bounded."""
        evaluated = [self._row(c) for c in self.population if c.evaluated]
        return evaluated[-self.config.history_limit :]

    def _history_truncated(self) -> bool:
        return sum(1 for c in self.population if c.evaluated) > self.config.history_limit

    # ── handing state between processes ────────────────────────────────────
    def write_pending(self, candidates: Sequence[Candidate]) -> Path:
        """Write the offspring one step produced, for the step that installs them.

        A graph node that proposes and a graph node that installs are two
        processes, so what the first found has to be written down. That is a cost
        of expressing the loop as a graph; the alternative is one node that does
        both, which hides the seam rather than removing it.
        """
        return self._write(
            PENDING,
            [
                {
                    "id": candidate.id,
                    "parent": candidate.parent,
                    "edits": candidate.edits,
                    "edit_reports": candidate.edit_reports,
                    "step_edits": candidate.step_edits,
                    "state": candidate.state,
                }
                for candidate in candidates
            ],
        )

    def read_pending(self) -> list[Candidate]:
        """The offspring the proposal node left, as candidates again."""
        path = self.root / PENDING
        if not path.exists():
            return []
        return [
            Candidate.from_state(
                record["state"], parent=record.get("parent"), edits=record.get("edits") or (),
                edit_reports=record.get("edit_reports") or (),
                step_edits=int(record.get("step_edits", 1)),
            )
            for record in json.loads(path.read_text())
        ]

    def config_from_root(self) -> "LoopConfig":
        """This run's own configuration, as the run directory recorded it."""
        return LoopConfig.from_dict(json.loads((self.root / "config.json").read_text()))

    # ── the journal, and resume ────────────────────────────────────────────
    def _journal(self, event: str, **payload: Any) -> None:
        """Append one event. The journal is the run's durable state.

        Append-only and replayed rather than snapshotted: a snapshot of the
        population has to be kept consistent with the counters, and the old glue's
        rehydrate did exactly that and produced real bugs. One log means one source
        of truth, and the same one the report already reads.

        Thread-safe: the RLock prevents interleaved JSONL lines when
        concurrent evaluations write to the journal simultaneously.
        """
        with self._state_lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / JOURNAL).open("a") as handle:
                handle.write(json.dumps({"event": event, **payload}, default=str) + "\n")

    def _journal_candidate(self, candidate: Candidate) -> None:
        self._journal(
            "candidate",
            id=candidate.id,
            parent=candidate.parent,
            edits=candidate.edits,
            edit_reports=candidate.edit_reports,
            step_edits=candidate.step_edits,
            state=candidate.state,
        )

    def _replay_if_asked(self) -> bool:
        """Rebuild the run from its journal, if there is one and resume was asked for."""
        journal = self.root / JOURNAL
        if not self.config.resume:
            if journal.exists():
                journal.unlink()
            return False
        if not journal.exists():
            raise LoopError(
                f"resume was asked for but {journal} does not exist, so there is "
                "nothing to carry on from"
            )
        self._replay(journal)
        return True

    def _replay(self, journal: Path) -> None:
        """Rebuild the population, the counters and the refusals from the log."""
        by_id: dict[str, Candidate] = {}
        order: list[str] = []
        # A strategy may evaluate a candidate before it keeps it (``Run.keep``
        # journals it then), so its score can come before the candidate itself.
        early_scores: dict[str, dict[str, Any]] = {}
        rescores = 0  # re-scores of the edited candidate (fresh evidence), each an evaluation
        rescore_measurements = 0

        def apply_score(candidate: Candidate, record: dict[str, Any]) -> None:
            candidate.score = record.get("score")
            # A journal written before the size cost existed has no `raw_score`;
            # its `score` is the evaluator's own number, which is then the raw
            # one too.
            raw = record.get("raw_score", record.get("score"))
            candidate.raw_score = None if raw is None else float(raw)
            # Old records without an explicit outcome cannot establish a
            # successful run. Keep them inspectable but ineligible, matching
            # adoption and console winner selection.
            candidate.exit_ok = record.get("exit_ok") is True
            candidate.error = record.get("error")
            candidate.duration_s = float(record.get("duration_s") or 0.0)
            candidate.model_calls = int(record.get("model_calls") or 0)
            candidate.facts = dict(record.get("facts") or {})
            candidate.measurements = dict(record.get("measurements") or {})
            candidate.evaluated = True

        for line in journal.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A half-written last line is what a crash looks like. Stopping at
                # it is the point of an append-only log.
                break
            event = record.get("event")
            if event == "candidate":
                # F14: refuse journal entries that contain removed edit ops.
                _removed_ops = {"set_node_field", "replace_node"}
                for _e in (record.get("edits") or ()):
                    if isinstance(_e, dict) and _e.get("op") in _removed_ops:
                        raise LoopError(
                            f"cannot resume: journal contains edit op {_e['op']!r} "
                            "which was removed in F14; start a new optimization run"
                        )
                    folded = nn_edits.removed(_e.get("op")) if isinstance(_e, dict) else None
                    if folded is not None:
                        raise LoopError(
                            f"cannot resume: journal contains edit op {_e['op']!r}, "
                            f"which no longer exists ({folded}); start a new "
                            "optimization run"
                        )
                candidate = Candidate.from_state(
                    record["state"],
                    parent=record.get("parent"),
                    edits=record.get("edits") or (),
                    edit_reports=record.get("edit_reports") or (),
                    # A journal written before compound proposals: one edit each.
                    step_edits=int(record.get("step_edits", 1)),
                )
                if candidate.id not in by_id:
                    by_id[candidate.id] = candidate
                    order.append(candidate.id)
                    if candidate.id in early_scores:
                        apply_score(candidate, early_scores.pop(candidate.id))
            elif event == "score":
                candidate = by_id.get(record.get("id"))
                if candidate is None:
                    early_scores[str(record.get("id"))] = record
                    continue
                apply_score(candidate, record)
            elif event == "rejection":
                self.rejections.append(
                    Rejection(
                        attempt=int(record.get("attempt") or 0),
                        edit=dict(record.get("edit") or {}),
                        reasons=list(record.get("reasons") or []),
                    )
                )
            elif event == "epoch":
                self.epoch = int(record.get("epoch") or self.epoch)
            elif event == "epoch_open":
                self.epoch = int(record.get("epoch") or self.epoch)
                self._epoch_open = True
            elif event == "epoch_end":
                self.epoch = int(record.get("epoch") or self.epoch)
                self._epoch_open = False
            elif event == "step":
                # The generation is what the proposal stream is derived from, so it
                # is journaled rather than inferred: guessing it makes a resumed run
                # propose something else.
                self.generation = int(record.get("generation") or self.generation)
                self._refusal_pressure = float(record.get("refusal_pressure") or 0.0)
            elif event == "bound":
                self.bound = str(record.get("bound") or "")
            elif event == "loss":
                self.loss = {
                    key: value
                    for key, value in record.items()
                    if key != "event"
                }
            elif event == "original_seed":
                # Restore the original seed state for tool-policy admission.
                self._original_seed_state = record.get("state")
            elif event == "control":
                # Rebuilt rather than read from `state.json`: the journal is the run's
                # durable state, and a resumed run that forgot a rejection would
                # resurrect a branch the person had already dismissed.
                self._replay_control(record)
            elif event == "strategy_state":
                if "state" in record:
                    self.load_strategy_state(record["state"])
            elif event in ("rescore", "rescore_failed"):
                rescores += 1
                rescore_measurements += int((record.get("measurements") or {}).get("count", 1))
            elif event == "train_minibatch":
                self._minibatches[int(record["epoch"])] = [str(i) for i in record.get("ids") or ()]
            elif event == "train_records_epoch":
                cid, at = str(record.get("id")), int(record.get("epoch") or 0)
                self._records_epoch[cid] = at
                self._records_by_epoch.setdefault(cid, {})[at] = list(
                    self._train_item_records.get(cid, []))
            elif event == "train_records":
                cid = record.get("id")
                if cid is not None:
                    self._train_item_records[cid] = list(record.get("records") or [])
            elif event == "part_result":
                cid = record.get("id")
                part_name = record.get("part")
                result_data = record.get("result")
                if cid is not None and part_name is not None and result_data is not None:
                    self._part_results.setdefault(cid, {}).setdefault(part_name, []).append(result_data)
            elif event == "occasion_counter":
                cid = record.get("id")
                part_name = record.get("part")
                counter_val = record.get("counter")
                if cid is not None and part_name is not None and counter_val is not None:
                    self._occasion_counters.setdefault(cid, {})[part_name] = int(counter_val)
            elif event == "rollouts":
                self.rollouts = int(record.get("rollouts") or 0)
            elif event == "seed_cache_hit":
                if record.get("part"):
                    self._seed_cache_used.add(str(record["part"]))
            elif event == "seed_cache_miss":
                if record.get("part"):
                    self._seed_cache_missed.add(str(record["part"]))

        if not order:
            raise LoopError(f"{journal} holds no candidate, so there is nothing to resume")
        self.population = [by_id[node_id] for node_id in order]
        self.evaluations = sum(1 for c in self.population if c.evaluated) + rescores
        self.measurements = (sum(c.measurements.get("count", 1) for c in self.population if c.evaluated)
                             + rescore_measurements)
        self.proposals = {
            candidate.id: {
                "id": candidate.id,
                "parent": candidate.parent,
                "edits": candidate.edits,
                "rationale": "replayed from the journal",
                "prediction": "",
                "confidence": 0.0,
                "origin": "seed_population" if candidate.parent else "seed_population",
            }
            for candidate in self.population
        }
        for line in journal.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "proposal" and record.get("id") in self.proposals:
                self.proposals[record["id"]] = {k: v for k, v in record.items() if k != "event"}

    def _replay_control(self, record: dict[str, Any]) -> None:
        """Restore one person's act while the journal is being replayed.

        The recorded `kept`/`rejected` lists are the authority when they are present,
        because they are the run's own ending state; the per-target branch covers a
        journal that recorded only what was asked for.
        """
        if "kept" in record or "rejected" in record:
            self.kept = [str(candidate_id) for candidate_id in record.get("kept") or ()]
            self.rejected = [
                str(candidate_id) for candidate_id in record.get("rejected") or ()
            ]
        elif record.get("action") in ("keep", "reject"):
            action = str(record["action"])
            asked = list(record.get("targets") or ())
            if record.get("target"):
                asked.append(record["target"])
            for candidate_id in (str(value) for value in asked):
                if action == "reject":
                    if candidate_id not in self.rejected:
                        self.rejected.append(candidate_id)
                    if candidate_id in self.kept:
                        self.kept.remove(candidate_id)
                else:
                    if candidate_id in self.rejected:
                        self.rejected.remove(candidate_id)
                    if candidate_id not in self.kept:
                        self.kept.append(candidate_id)

        if record.get("action") == "adopt":
            self.adoptions.append(
                {
                    key: value
                    for key, value in record.items()
                    if key not in ("event", "kept", "rejected")
                }
            )

    # ── persistence ────────────────────────────────────────────────────────
    def _write(self, name: str, payload: Any) -> Path:
        return self._write_text(name, json.dumps(payload, indent=2, default=str))

    def _write_text(self, name: str, text: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def _progress(self, step: str, **fields: Any) -> None:
        """One line of "where the run is", readable while it is still running.

        The journal already records everything, but it is replayed rather than
        tailed and a step is only interesting once you know which one it was. This
        is the same facts at the granularity a reader watches: one line per
        candidate evaluated, not one per epoch, because an epoch can be an hour.
        """
        if not self.config.progress:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        record = {
            "step": step,
            "epoch": self.epoch,
            "generation": self.generation,
            "evaluations": self.evaluations,
            "time": round(time.time(), 3),
            **fields,
        }
        with (self.root / progress_file.PROGRESS).open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        if self._printer is not None:
            self._printer(record)

    def _write_state(self) -> None:
        self._write(
            "state.json",
            {
                "mode": self.mode_id,
                "epoch": self.epoch,
                "generation": self.generation,
                "evaluations": self.evaluations,
                "scoring": self.scoring(),
                # A person's two lists, so a panel and a live `leaderboard` read them
                # without replaying the journal.
                "kept": list(self.kept),
                "rejected": list(self.rejected),
                "adoptions": list(self.adoptions),
                "population": [self._row(candidate, detail=True)
                               for candidate in self.population],
            },
        )


def _wiring_warnings(state: Mapping[str, Any] | None,
                     baseline: Mapping[str, Any] | None = None) -> list[str]:
    """A state's wiring warnings (`nn.validate.wiring_warnings`), its steps judged against
    ``baseline``'s (new or changed ones only) when given; none for a state whose graph
    does not parse (the validator refuses it on its own)."""
    from refactory_dsh.nn.graph import Graph
    from refactory_dsh.nn.validate import wiring_warnings

    try:
        before = (Graph.from_state((baseline or {}).get("graph") or {})
                  if baseline is not None else None)
        return wiring_warnings(Graph.from_state((state or {}).get("graph") or {}), before)
    except Exception:  # noqa: BLE001 - a broken graph is V0's to report, not this note's
        return []


def _other_value(knob: Knob, current: Any, rng: random.Random) -> Any:
    """A value a knob accepts that is not the one it currently holds.

    One sampler for the whole system, so the starting sweep and the random
    optimizer agree about what is reachable.
    """
    return knob.sample(rng, current)


def node_count(state: dict[str, Any]) -> int:
    """How many nodes a candidate's graph holds, which is what the size cost counts."""
    graph = state.get("graph") or {}
    return len(graph.get("nodes") or {})


def _considered_lines(outcomes: Sequence[dict[str, Any]]) -> list[str]:
    """The report's record of what the optimizer weighed for each accepted proposal (its
    answer's optional ``considered``), so every round's choice of change can be audited;
    nothing when no answer said."""
    rows = [(p.get("id"), p.get("epoch"), " ".join(str(p["considered"]).split()))
            for p in outcomes if p.get("considered")]
    if not rows:
        return []
    return ["## What the optimizer weighed", "",
            "The kinds of change each proposal's answer said it weighed, and why it chose "
            "its own.", "",
            *(f"- `{cid}` (epoch {epoch}): {text}" for cid, epoch, text in rows), ""]


def _holdout_line(audit: dict[str, Any]) -> str:
    """One line saying what the held-out audit found, for `refactory optimize`."""
    if not audit:
        return "holdout: this run recorded no audit of the winner"
    if "val" in audit and "test" in audit:
        # ``audit_parts``: the winner's latest val score against its latest test score.
        if audit["val"] is None or audit["test"] is None:
            return f"holdout: {audit.get('note') or 'no val or test score'} (val and test parts)"
        verdict = ("unmeasured" if audit.get("status") == "unmeasured"
                   else "overfit" if audit.get("overfit") else "holds up")
        added = audit.get("added_gap")
        added_text = (f", added by the search {added:.4f}" if isinstance(added, (int, float))
                      else "")
        se = audit.get("standard_error")
        return (
            f"holdout: val {audit['val']:.4f}, test {audit['test']:.4f}, "
            f"gap {audit.get('gap'):.4f}{added_text} against threshold {audit.get('threshold')}"
            + (f" (SE {se})" if se is not None else "")
            + f" — {verdict} (val and test parts)"
            + (f"; {audit['note']}" if audit.get("note") else "")
        )
    instances = audit.get("instances") or {}
    train = audit.get("train")
    holdout = audit.get("holdout")
    split = (
        f"{instances.get('train')} train / {instances.get('holdout')} holdout instance(s)"
    )
    if holdout is None or train is None:
        return f"holdout: {audit.get('note') or 'no holdout score'} ({split})"
    verdict = "overfit" if audit.get("overfit") else "holds up"
    se = audit.get("standard_error")
    return (
        f"holdout: train {train:.4f}, holdout {holdout:.4f}, "
        f"gap {audit.get('gap'):.4f} against threshold {audit.get('threshold')}"
        + (f" (SE {se})" if se is not None else "")
        + f" — {verdict} ({split})"
        + (f"; {audit['note']}" if audit.get("note") and audit.get("overfit") is False else "")
    )


def _portfolio_section(portfolio: dict[str, Any]) -> list[str]:
    if not portfolio:
        return []
    pool = portfolio.get("selection") == "pool"
    rows = [[level, entry["proposed"], entry["evaluated"], entry["kept"],
             *([entry.get("in_pool", 0)] if pool else []), entry["evaluations"]]
            for level, entry in portfolio["by_level"].items()]
    lines = ["## Portfolio", "",
             f"`{portfolio['setting']}`: {portfolio['rounds']} round(s) asked for one candidate "
             f"per edit level; the candidates beyond the batch size of {portfolio['batch_size']} "
             f"cost {portfolio['extra_evaluations']} evaluation(s) (train and val runs). The "
             "gate judges a round's candidates together, each against the current candidate "
             "as the round starts, and accepts the one with the highest val score (ties: higher "
             "train score, then the one proposed first)"
             + (" as the new incumbent, if it beats the current one strictly" if pool else "")
             + ": a candidate that beat its parent on val but lost to a sibling is marked "
             "shadowed."
             + (" Under pool selection every judged candidate may also enter the pool of the "
                "best on val: `in pool` is in it now, `left pool` was in it when judged and "
                "was pushed out since, `rejected` never entered it (`below pool` when the "
                "parent picker still breeds from every judged candidate). `accepted` marks a "
                "former incumbent." if pool else ""), "",
             _md_table(["level", "proposed", "evaluated",
                        "accepted as incumbent" if pool else "kept by the gate",
                        *(["in pool now"] if pool else []), "evaluations"], rows),
             ""]
    round_rows = []
    for entry in portfolio.get("per_round") or []:
        for candidate in entry["candidates"]:
            outcome = candidate["gate"] or ("evaluated" if candidate["evaluated"]
                                            else "not evaluated")
            if pool and candidate.get("kept") and candidate["gate"] != "current":
                outcome = f"accepted, {outcome}"
            if candidate["shadowed"]:
                outcome = f"shadowed, {outcome}" if pool else "shadowed"
            round_rows.append([entry["epoch"], candidate["level"], candidate["id"], outcome,
                               candidate["val"], candidate["parent_val"]])
    if round_rows:
        lines += [_md_table(["round", "level", "candidate", "outcome", "val", "parent val"],
                            round_rows), ""]
    return lines


def _selection_section(selection: dict[str, Any]) -> list[str]:
    """How the run selected, its pool's diversity and its budget, as a report section."""
    if not selection or not selection.get("setting"):
        return []
    setting = selection["setting"]
    what = {
        "pool": (f"`pool`: the validation average is the fitness; the pool keeps the "
                 f"{selection.get('pool_size')} best candidates on validation (ties at the edge "
                 "kept) and parents come from it; the incumbent, the winner, changes only on a "
                 "strictly higher validation score."),
        "single": ("`single`: SkillOpt's line; a candidate is kept only on a strictly higher "
                   "validation score than the incumbent's, and the incumbent is the only parent."),
        "train": "`train`: no validation gate, so train scores choose the parents and the winner.",
    }.get(setting, f"`{setting}`")
    budget = selection.get("budget") or {}
    share = selection.get("parents_from_incumbent")
    return ["## Selection", "", what, "",
            *([f"**Warning: {selection['warning']}.**", ""] if selection.get("warning") else []),
            _md_table(["pool", "lines of descent", "largest line", "edit kinds",
                       "parents from the incumbent"],
                      [[len(selection.get("pool") or ()), selection.get("lines_of_descent"),
                        selection.get("largest_line_share"),
                        ", ".join(f"{k} {v}" for k, v in (selection.get("edit_kinds") or {}).items()),
                        "n/a" if share is None else f"{share} of {selection.get('children')}"]]),
            "",
            _md_table(["train scorings", "val scorings", "test scorings", "rollouts", "tokens",
                       "wall-clock s"],
                      [[budget.get("train_scorings"), budget.get("val_scorings"),
                        budget.get("test_scorings"), budget.get("rollouts"),
                        budget.get("tokens") if budget.get("tokens") is not None
                        else budget.get("known_tokens"), budget.get("elapsed_s")]]),
            ""]


def _strategy_report_section(report: dict[str, Any]) -> list[str]:
    """A custom strategy's own figures as a report section, or nothing when it has none."""
    if not report:
        return []
    return ["## Strategy report", "",
            _md_table(["field", "value"], [[key, json.dumps(value, default=str) if isinstance(value, (dict, list)) else value]
                                           for key, value in report.items()]),
            ""]


def _holdout_sentence(audit: dict[str, Any]) -> str:
    """The audit as one paragraph for the report, saying both numbers and the gap, and
    which split it ran on."""
    if not audit:
        return "_This run recorded no audit of the winner._"
    line = _holdout_line(audit)
    if audit.get("split_note"):
        line += f". Audit split: {audit['split_note']}"
    if audit.get("overfit"):
        return (
            f"{line}. Every candidate was searched on the train part, so the winner "
            "may be a harness that only works on the items it was tuned against."
        )
    return f"{line}."


def _yield(accepted: int, refused: int) -> str:
    total = accepted + refused
    if not total:
        return "n/a"
    return f"{100.0 * accepted / total:.0f}%"


def _edit_record(candidate: Candidate) -> dict[str, Any]:
    """What a candidate's proposal applied, in a refusal's form: its edit, or
    ``{"edits": [...]}`` for a compound proposal."""
    own = candidate.own_edits()
    if not own:
        return {}
    return own[0] if len(own) == 1 else {"edits": own}


def _codes(rejections: Sequence[Rejection]) -> dict[str, int]:
    """How many refusals each rule code made, by code."""
    counts: dict[str, int] = {}
    for rejection in rejections:
        for code in rejection.codes():
            counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items()))


def _trim(facts: dict[str, Any], limit: int = 400) -> dict[str, Any]:
    """Keep the facts, but not an unbounded model answer inside the trace."""
    out: dict[str, Any] = {}
    for key, value in facts.items():
        if isinstance(value, str) and len(value) > limit:
            out[key] = value[:limit] + "…"
        else:
            out[key] = value
    return out


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_(none)_"
    head = "| " + " | ".join(headers) + " |"
    rule = "|" + "|".join("---" for _ in headers) + "|"
    body = [
        "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |"
        for row in rows
    ]
    return "\n".join([head, rule, *body])


def run(
    mode_id: str,
    *,
    config: LoopConfig | None = None,
    project: Path | None = None,
    model: Any = None,
) -> dict[str, Any]:
    """Build a loop and run it to its bound."""
    loop = Loop(mode_id, config, project=project, model=model)
    return loop.run()


def _coerce_strategy_value(raw: str) -> Any:
    """Coerce a CLI strategy-option value to int, float, bool, or string."""
    if raw.lower() in ("true", "yes"):
        return True
    if raw.lower() in ("false", "no"):
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def main(argv: list[str] | None = None, *, _client_factory: Any = None) -> int:
    """``python -m refactory_dsh.loop <package>/<mode> [options]``.

    ``_client_factory`` replaces ``DshModelClient`` when given: it receives the
    same keyword arguments, so a caller (``refactory grid``) or a test can see the
    resolved route and answer with a stub.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m refactory_dsh.loop",
        description="Run the optimizer loop on a mode.",
    )
    parser.add_argument("mode", help="the mode to optimize, as <package>/<name>")
    from refactory_dsh import profiles, strategies
    profiles.add_arguments(parser)
    parser.add_argument("--runs-root", default=".factory/loop")
    parser.add_argument("--project", default=str(project_root()))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--parts", default=None, dest="parts_manifest",
                        help="path to a parts.json manifest with train/val/test id lists")
    parser.add_argument("--resolved-config", default="", help="exact resolved snapshot prepared by the console")
    args = parser.parse_args(argv)
    project = Path(args.project)
    from refactory_dsh.model import DshModelClient, ModelChoiceError, require_model_choice

    def settle() -> dict[str, Any]:
        if args.resolved_config:
            snapshot = json.loads(Path(args.resolved_config).read_text())
            if snapshot.get("mode") != args.mode:
                raise profiles.ProfileError("resolved snapshot belongs to another mode")
            return snapshot
        # `--parts`, `--strategy-option` and `--knob` are overrides like any other: a
        # setting the command line leaves out keeps the profile's value.
        overrides = profiles.cli_overrides(args)
        prior_path = Path(args.runs_root) / "config.json"
        if args.resume and not args.profile and prior_path.exists():
            previous = json.loads(prior_path.read_text())
            if previous.get("mode") != args.mode:
                raise profiles.ProfileError("the resume directory belongs to another mode")
            previous_settings = {k: v for k, v in previous.items() if k in profiles.defaults()}
            for key in Loop.RESUME_KEEPS:
                # Refused here, before the settings are checked together, with the
                # loop's wording (``Loop._keep_recorded_run_shape``): a recorded
                # selection with --val-gate 0 would fail that check less clearly.
                was, asked = previous.get(key), overrides.get(key)
                if asked is None or was in (None, ""):
                    continue
                if key == "val_gate":
                    was, asked = int(bool(was)), int(bool(asked))
                if asked != was:
                    flag = "--" + key.replace("_", "-")
                    raise profiles.ProfileError(
                        f"this run was started with {key} {was} and this resume asks for "
                        f"{key} {asked}; resume it without {flag} (or with {flag} {was}), "
                        "or start a new run")
            return profiles.refresh(project, args.mode, profiles.normalize({**previous_settings, **overrides}),
                                    previous.get("resolved_profile") or {}, overrides=overrides)
        return profiles.resolve(project, args.mode, name=args.profile,
                                version=args.profile_version, overrides=overrides)

    def client(snapshot: dict[str, Any]) -> DshModelClient:
        settings = snapshot["settings"]
        return (_client_factory or DshModelClient)(routes_resolved=True, routing_provenance=(snapshot.get("worker_configuration") or {}).get("provenance"), **{key: settings[key] for key in
            ("worker_backend", "provider", "model", "effort", "permission_mode", "setting_sources", "role_routes", "node_routes", "provider_config")
            })

    try:
        snapshot = settle()
        model = client(snapshot)
        # A search whose graph or optimizer calls a model settles its route before the
        # first candidate runs (dm-258); one that calls none (`--evaluator walker` over
        # deterministic nodes, a random optimizer) needs no choice at all.
        if ((snapshot.get("evaluation_context") or {}).get("routing") or {}).get("status") \
                != "no_model_required_by_current_graph_or_optimizer":
            gated = require_model_choice(model, interactive=sys.stdin.isatty())
            if gated.route() != model.route() and not args.resolved_config:
                # The first-run choice was just stored: resolve again, so the search's
                # recorded context names the route its candidates run on.
                snapshot = settle()
                gated = client(snapshot)
            model = gated
        settings = snapshot["settings"]
        config = LoopConfig.from_dict({**settings, "runs_root": args.runs_root,
                                      "print_events": True, "verbose": args.verbose,
                                      "resume": args.resume, "resolved_profile": snapshot})
    except ModelChoiceError as exc:
        print(f"refactory: {exc}", file=sys.stderr)
        return 1
    except (profiles.ProfileError, ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    try:
        loop = strategies.create(
            config.strategy, args.mode, config,
            strategy_options=config.strategy_options,
            project=project, model=model,
        )
        report = loop.run()
    except LoopError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    audit = report.get("holdout") or {}
    # One line a person reads without opening the run directory. On stderr, because
    # everything this command prints on stdout is the one JSON object a caller parses.
    print(_holdout_line(audit), file=sys.stderr)
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "exit_ok": report["exit_ok"],
                "error": report["error"],
                "optimizer": report["optimizer"],
                "evaluator": loop.config.evaluator,
                "bound": report["bound"],
                "evaluations": report["evaluations"],
                "seed_score": report["seed_score"],
                "best_score": report["best_score"],
                "raw_best_score": report["raw_best_score"],
                "scoring": report["scoring"],
                "holdout": audit,
                "structural_wins": len(report["structural_wins"]),
                "rejection_codes": report["rejection_codes"],
                "accepted_op_mix": report["accepted_op_mix"],
                **({"strategy_report": report["strategy_report"]} if report.get("strategy_report") else {}),
                "report": str(Path(args.runs_root) / "report.md"),
            },
            indent=2,
            default=str,
        )
    )
    return 0 if report["exit_ok"] else 1


if __name__ == "__main__":  # pragma: no cover - the entry point
    raise SystemExit(main())
