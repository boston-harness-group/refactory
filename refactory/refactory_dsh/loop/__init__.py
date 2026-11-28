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
import functools
import json
import math
import random
import re
import shutil
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass
from contextlib import nullcontext
from pathlib import Path
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

if TYPE_CHECKING:
    from refactory_dsh.nn.mode import RunResult

from refactory_dsh import history as run_history
from refactory_dsh import events
from refactory_dsh import observe as progress_file
from refactory_dsh import task as task_module
from refactory_dsh.evaluators import create as create_evaluator
from refactory_dsh import registry
from refactory_dsh.nn import Knob, PackageError
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn import validate_state
from refactory_dsh.nn import pipeline_view
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
    MAX_EDITS_PER_PROPOSAL, NEW_STEP_TOOLS_OFF, RunStopped, available_before, edit_level, proposal_refusal,
    refused_entries,
    repair_losses,
    summarise_reasons,
)
from refactory_dsh.optim.memory import report as memory_report
from refactory_dsh.optim.memory import report_lines as memory_report_lines
from refactory_dsh.outerloop import (
    BudgetStopped, EvalRequest, FinalEvaluation, OuterLoop, OuterLoopError, Run, ValidationGate, train_batch_seed,
)
from refactory_dsh.paths import project_for_nodes, project_root
from refactory_dsh.workers import StepRoutes

from refactory_dsh.loop.config import EXIT_INTERRUPTED, EpochSummary, EvaluationInterrupted, HOLDOUT_FILE, INCIDENT_RERUNS, INCIDENT_RERUN_DIR, INTERRUPTED_RETRY_WAITS_S, INTERRUPTION_CAP, JOURNAL, LoopConfig, LoopError, NOT_APPLIED, NO_VALIDATION, OPTIMIZER_TRACE, OVERFIT_THRESHOLD, PENDING, PORTFOLIOS, SEED_WINNER_NOTE, SELECTIONS, SMOKE_CODE, _UNSET, _item_sources, _new_steps, _train_key, _values_before, default_selection, gap_standard_error, item_time_spread, opaque_item_key, single_parent
from refactory_dsh.loop.context import _ContextMixin
from refactory_dsh.loop.evaluate import _EvaluateMixin, _new_wiring_warnings, _other_value, _wiring_warnings
from refactory_dsh.loop.forward import _ForwardMixin
from refactory_dsh.loop.incidents import _IncidentsMixin
from refactory_dsh.loop.parts import _PartsMixin
from refactory_dsh.loop.report import REPORT_ANALYST_FINDINGS, _DISPATCH_RECORDS, _ReportMixin, _analysts_report, _analysts_section, _changes_cell, _codes, _considered_lines, _edit_record, _headline, _holdout_line, _holdout_sentence, _incident_lines, _integrity_section, _markdown_text, _md_table, _portfolio_section, _selection_section, _strategy_report_section, _trim, _uncapped_lines, _yield, node_count, uncapped_sealed_steps
from refactory_dsh.loop.resume import _ResumeMixin
from refactory_dsh.loop.selection import _SelectionMixin


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
    # `refactory optimize`'s own short list: every other setting comes from a profile.
    profiles.add_cli_arguments(parser)
    parser.set_defaults(project=str(project_root()), runs_root=".factory/loop")
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
        # `--strategy-option` and `--knob` are overrides like any other: a setting the
        # command line leaves out keeps the profile's value.
        overrides = profiles.cli_overrides(args)
        prior_path = Path(args.runs_root) / "config.json"
        if args.resume and not args.profile and prior_path.exists():
            previous = json.loads(prior_path.read_text())
            if previous.get("mode") != args.mode:
                raise profiles.ProfileError("the resume directory belongs to another mode")
            previous_settings = {k: v for k, v in previous.items() if k in profiles.defaults()}
            # The recorded ``RESUME_KEEPS`` settings stay: no flag of this command sets one.
            return profiles.refresh(project, args.mode,
                                    profiles.normalize({**previous_settings, **overrides}, project),
                                    previous.get("resolved_profile") or {}, overrides=overrides)
        return profiles.resolve(project, args.mode, name=args.profile,
                                version=args.profile_version, overrides=overrides)

    def client(snapshot: dict[str, Any]) -> DshModelClient:
        settings = snapshot["settings"]
        return (_client_factory or DshModelClient)(routes_resolved=True, routing_provenance=(snapshot.get("worker_configuration") or {}).get("provenance"), **{key: settings[key] for key in
            ("worker_backend", "provider", "model", "effort", "permission_mode", "setting_sources", "role_routes", "node_routes", "provider_config")
            }, **({"claude_executable": ce} if (ce := settings.get("claude_executable")) else {}))

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
        # A snapshot written before a setting was removed still carries it; the run
        # decides val_gate itself (``Loop._resolve_val_gate``).
        settings = {k: v for k, v in settings.items() if k not in profiles.OBSOLETE_SETTINGS}
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
    except EvaluationInterrupted as exc:
        # Not a refused configuration: the run is whole and resumable.
        print(json.dumps({"error": str(exc), "resumable": True}), file=sys.stderr)
        return EXIT_INTERRUPTED
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


class Loop(_ResumeMixin, _PartsMixin, _ForwardMixin, _EvaluateMixin, _IncidentsMixin, _SelectionMixin, _ContextMixin, _ReportMixin, OuterLoop):
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
            stop_check = getattr(self.model, "with_stop_check", None)
            if callable(stop_check):
                # A call waiting out a network outage ends its wait when a person asks
                # the run to stop (`RunStopped`), so the stop is not held up for hours.
                self.model = stop_check(self._stop_requested)
            self.evaluator.model = self.model
            proposer = self.model.for_node("optimizer", role="optimizer") if callable(getattr(self.model, "for_node", None)) else self.model
            self.optimizer.bind_model(proposer, workdir=self.root / "proposer")
            self._judge_model = proposer
            if self._custom_strategy is not None:
                self._custom_strategy.bind_model(proposer)
        self.evaluator.budget = budget

    def run(self) -> dict[str, Any]:
        """Own strategy execution; a report-only replay does not start a search."""
        from concurrent.futures import CancelledError

        from refactory_dsh.control import running_search
        from refactory_dsh.budgets import BudgetExhausted, RunBudget, ResourceLimits, LEDGER
        from refactory_dsh.model import ModelChoiceError

        if (self.config.analysts and self._custom_strategy is None
                and not getattr(self.optimizer, "uses_model", False)):
            # Checked here, not at construction, against the optimizer the run uses.
            raise LoopError(f"analysts read the train items for a model proposer; the "
                            f"{self.optimizer.name} optimizer asks no model, so they would "
                            "never run (use --optimizer autoresearch, or analysts off)")
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
            self.audit_on_report = True
            try:
                if self._integrity_on():
                    self._integrity_policy()  # an unreadable declaration stops the run here
                    if self._judge_client() is None:
                        raise LoopError("integrity_judge is on but this run has no model to "
                                        "judge with (the judge uses the optimizer's model "
                                        "route); turn it off or give the run a model")
                try:
                    result = super().run()
                except (CancelledError, ModelChoiceError) as exc:
                    # A cancelled model client or a missing model choice ends the run,
                    # but it still says why and leaves its report. (A person's stop,
                    # `RunStopped`, ended the search inside ``super().run()``.)
                    self.bound = self.bound or f"{type(exc).__name__}: {exc}"
                    self._journal("bound", bound=self.bound)
                    self.report()
                    raise
                except (BudgetExhausted, BudgetStopped) as exc:
                    # BudgetStopped: a strategy's evaluation the budget stopped
                    # (``Run.evaluate``); the bound is the budget _measure recorded.
                    self.bound = self.bound or str(exc)
                    self._journal("bound", bound=self.bound)
                    result = self.report()
                # A same-bound resume stops before the scheduler. Persist that
                # final reason too, so a later report-only replay retains it, unless
                # the journal already ends on it (the scheduler or an early stop wrote it).
                if self.__dict__.get("_journaled_bound", _UNSET) != self.bound:
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
        self.project = Path(project or project_for_nodes())
        self.root = Path(self.config.runs_root)
        self.model = model
        # A library caller may provide a configured client directly. Its actual
        # requested routes must agree with the saved effective settings, just as
        # clients constructed by the CLI do.
        if callable(getattr(model, "context", None)):
            client_context = model.context()
            for key in ("worker_backend", "provider", "model", "effort", "permission_mode", "setting_sources", "claude_executable", "role_routes", "node_routes", "provider_config"):
                if key in client_context:
                    setattr(self.config, key, client_context[key])
        if not isinstance(self.config.repeats, int) or not 1 <= self.config.repeats <= 10:
            raise LoopError("repeats must be an integer from 1 to 10")
        minibatch = self.config.train_minibatch
        if isinstance(minibatch, bool) or not isinstance(minibatch, int) or minibatch < 0:
            raise LoopError("train_minibatch must be a nonnegative integer (0 is the whole train part)")
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
        if callable(getattr(model, "context", None)):
            # A Claude Code too old for a node's model is refused here, before the run
            # directory exists, not at the first candidate's first call: each seed node
            # on its own role's route, and the proposer on the optimizer's.
            from refactory_dsh.workers import agent_roles, claude_code_preflight
            try:
                claude_code_preflight(model.context(), roles={
                    **agent_roles(self.mode.state_dict()), "optimizer": "optimizer"},
                    pins=self.mode.claude_code_versions())
            except ValueError as error:
                raise LoopError(str(error)) from error
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
                self.config.optimizer, self.project, seed=self.config.seed,
                k=self.config.attempts)
        if self.config.parent_picker:
            from refactory_dsh.optim import PickedParent, ParentSelector
            if isinstance(self.optimizer, (ParentSelector, PickedParent)):
                raise LoopError(
                    f"the {self.optimizer.name} optimizer picks parents itself, so it cannot "
                    f"take the {self.config.parent_picker} parent picker; pair the picker "
                    "with a proposer (--optimizer autoresearch or random)")
            try:
                picker = create_optimizer(self.config.parent_picker, self.project,
                                          seed=self.config.seed, k=self.config.attempts)
                self.optimizer = PickedParent(picker, self.optimizer)  # type: ignore[arg-type]
            except ValueError as exc:
                raise LoopError(str(exc)) from exc
        if self.config.selection not in SELECTIONS:
            raise LoopError(f"unknown selection {self.config.selection!r}; known: "
                            f"{', '.join(s for s in SELECTIONS if s)}")
        picker_name = getattr(getattr(self.optimizer, "picker", None), "name", "")
        #: ``default_selection`` for this run, decided once: a package optimizer's
        #: class is looked up by discovery, which is not a per-step cost.
        self._default_selection = default_selection(
            self.name, self.config.optimizer, picker_name, self.project)
        if (self.config.selection == "checkpoint"
                and self._default_selection != "checkpoint"):
            raise LoopError(
                "selection checkpoint trains one line of candidates, so it needs the default "
                f"strategy and a single-parent optimizer; this run has the {self.name} "
                f"strategy, the {self.config.optimizer} optimizer"
                + (f" and the {picker_name} parent picker" if picker_name else "")
                + "; use selection single or pool")
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
        # The cap on edits per proposal, enforced where every proposal is applied.
        self.optimizer.max_edits = MAX_EDITS_PER_PROPOSAL
        # Whether edits may add code nodes, and what their code may never contain.
        self.optimizer.allow_code_nodes = bool(self.config.allow_code_nodes)
        # The routes a step may choose its own model, effort and cap from (`Node.route`).
        self.optimizer.step_routes = StepRoutes.of(model)
        # Whether a new step's tools can run sealed in this run: one answer for the
        # request's offer and an attempt's acceptance (`workspace.sealed_steps`).
        from refactory_dsh.workspace import sealed_step, sealed_steps

        self.optimizer.sealed_tools = self._sealed_answer(model)
        self.optimizer.new_step_tools = bool(self.config.new_step_tools)
        # Each new sealed step on its own route, as its dispatch takes it.
        self.optimizer.sealed_check = functools.partial(sealed_step, model)
        self.optimizer.grader_secrets = self._grader_secrets
        # A new or changed code node is dry-run on its parent's recorded item inputs
        # (`optim.checks.code_dry_run`), found in the parent's run directory.
        self.optimizer.recorded_inputs = self._recorded_inputs

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
        #: The loop's memory note (``LoopConfig.lessons``, `optim.memory`): ``{"text",
        #: "epoch", "generation", ...}`` and ``not_updated`` after a failed pass,
        #: journaled as ``optimizer_memory`` with each epoch's pass (``_memory_passes``).
        self._memory: dict[str, Any] = {}
        self._memory_passes: list[dict[str, Any]] = []
        #: The candidate current when each epoch proposed (``epoch_current``): the next
        #: epoch's memory pass compares it with the one current then.
        self._epoch_currents: dict[int, str] = {}
        #: Rollouts the memory pass made for sampled items a side lacked, by (candidate,
        #: side key: ``e<epoch>`` for a side read from that epoch's rollouts, ``fresh<epoch>``
        #: for a fresh rollout of an unchanged current without the gate): ``{id: record}``.
        self._memory_rollouts: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
        #: The proposals each epoch's proposal round refused before they ran, by that
        #: epoch (journaled with each ``rejection``), for the next memory pass's "What was
        #: tried this epoch" (`_tried_in_round`).
        self._round_refusals: dict[int, list[Rejection]] = {}
        #: How the Tasks' declared item metadata is read for the analysts
        #: (`task.item_metadata_reader`), held once as a one-element tuple.
        self._item_metadata: tuple[Any] | None = None
        #: Each round's analysis the analyst stage journaled (``LoopConfig.analysts``;
        #: `optim.analysts.journal_entry` with the epoch and generation), for report.md.
        self._analyses: list[dict[str, Any]] = []
        #: Every integrity verdict of the run (``LoopConfig.integrity_judge``), in order:
        #: each proposal's (scope ``proposal``) and the winner's combined change
        #: (``final``), journaled as ``integrity_verdict``; and the verdicts by the
        #: prompt they answered (`optim.integrity.key`), so a retry or a resumed run
        #: pays for no judgement twice. A failed judgement is recorded, never cached.
        self._integrity: list[dict[str, Any]] = []
        self._integrity_cache: dict[str, dict[str, Any]] = {}
        #: The judge's model: the optimizer's route (``bind_budget``), so its calls are
        #: charged to the run's budget like the proposer's. ``integrity_client`` set by
        #: a caller replaces it.
        self._judge_model: Any = None
        self.integrity_client: Any = None
        #: Whether ``report`` may judge the winner's combined change from the seed (a
        #: model call): set by ``run`` and by the graph-run loop's report node
        #: (``loop_mode``), never by a report-only replay.
        self.audit_on_report = False
        #: Whether a failed judgement was already said on stderr (once per process).
        self._judge_error_said = False
        #: The (candidate, epoch) re-scores tried in this process (`_rescore`), so a
        #: round's in-batch re-score that failed is not tried again after gating; a
        #: resumed run tries again.
        self._rescore_tried: set[tuple[str, int]] = set()
        #: Loop-private storage for per-part evaluation data (records, workdirs,
        #: scores).  Keyed by (candidate_id, part_name).  This data never reaches
        #: optimizer-facing APIs.
        self._part_results: dict[str, dict[str, list[dict[str, Any]]]] = {}
        #: Each trial's harness incidents (`_settle_incidents`), for the report.
        self._harness_incidents: list[dict[str, Any]] = []
        #: Loop-private per-candidate, per-part occasion counters.
        self._occasion_counters: dict[str, dict[str, int]] = {}
        #: Under ``checkpoint``, the candidates whose val rollout beside their train run
        #: (``_train_beside_val``) a budget stopped in this process, with the stop's
        #: message: the gate takes it as the refused rollout rather than rolling again.
        self._val_stopped_beside: dict[str, str] = {}
        #: How many evaluations the journal records per evaluation folder (relative to
        #: the root), counted from its events live and on replay (`_note_recorded`); it
        #: numbers the trace folders (`_write_traces`) and is never journaled.
        self._recorded_evaluations: dict[str, int] = {}
        #: Loop-private per-candidate train item records.
        self._train_item_records: dict[str, list[dict[str, Any]]] = {}
        #: Every train run's outcomes, ``(candidate id, [(item id, passed), ...])`` in
        #: scoring order, from the ``train_records`` events live and on replay
        #: (`_note_recorded`): what an item's history is read from (`_with_item_history`).
        self.__dict__.setdefault("_train_runs", [])
        #: The smoke check's verdict on each candidate it ran (``smoke`` events, live and
        #: replayed), so no graph is smoked twice, and how many smoke runs each had.
        self._smoke_verdicts: dict[str, dict[str, Any]] = {}
        self._smoke_runs: dict[str, int] = {}
        #: The wiring review of each candidate whose edit added wiring warnings
        #: (``wiring_review`` events, live and replayed; `_wiring_check`), by the id of
        #: the candidate reviewed and of the one it became, so none is asked twice.
        self._wiring_reviews: dict[str, dict[str, Any]] = {}
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
        #: The per-run secret behind `opaque_item_key`, made on first use and
        #: journaled (``item_key_secret``) so a resumed run keys items the same way.
        self._item_key_secret: str | None = None
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
        #: When True, ``_smoke_check`` returns the candidate without running it;
        #: ``install`` sets it for ``batch_size`` > 1 and smokes the batch after.
        self._defer_smoke: bool = False
        #: The validation gate when ``config.val_gate`` is on (see ``_gate``), and
        #: the candidates it accepted and rejected, in order.
        self._val_gate: ValidationGate | None = None
        #: Why ``val_gate`` is what it is (``_resolve_val_gate``), and the warning a run
        #: that fell back to train selection gives.
        self._val_gate_basis = "explicit"
        self.selection_warning = ""
        #: What a resume put back from the run's record (``_keep_recorded_run_shape``).
        self._resumed_shape: dict[str, Any] = {}
        #: How many times each evaluation (``<id>:<part>[@<rescore epoch>]``) stopped the
        #: run as interrupted, or was recorded failed at the cap (`INTERRUPTION_CAP`);
        #: rebuilt from the journal on resume.
        self._interrupted_stops: dict[str, int] = {}
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

    def _tools_offered(self) -> bool:
        """Whether the request shows a tool step: on the answer an attempt refuses one by
        (`workspace.sealed_steps`, the optimizer's ``sealed_tools``)."""
        if getattr(self.optimizer, "sealed_tools", None) is None:
            self.optimizer.sealed_tools = self._sealed_answer(self.model)
        return bool(self.optimizer.sealed_tools[0])

    def _sealed_answer(self, model: Any) -> tuple[bool, str]:
        """Whether a new step may run its tools sealed in this run, and why not: off when
        the run turns new steps' tools off (``new_step_tools``), else whether the seal can
        run here (`workspace.sealed_steps`)."""
        if not self.config.new_step_tools:
            return False, NEW_STEP_TOOLS_OFF
        from refactory_dsh.workspace import sealed_steps

        return sealed_steps(model)

    def _tools_not_offered(self) -> str:
        """Why the request offers no tool step, for the run's start summary and journal; empty
        when it offers one, or when the run permits no operator that adds a step."""
        from refactory_dsh.optim.base import tool_step_ops

        if self._tools_offered() or not tool_step_ops(self.permitted_ops()):
            return ""
        return str(self.optimizer.sealed_tools[1] or "the seal cannot run here")

    def _sandbox_answer(self) -> tuple[bool, str] | None:
        """The machine's sandbox answer for one request (`snippet.available`), taken once
        so the offer and the refusals agree; ``None`` when the run allows no code nodes."""
        if not self.config.allow_code_nodes:
            return None
        from refactory_dsh import snippet

        return snippet.available()

    def _recorded_inputs(self, candidate_id: str) -> dict[str, Any]:
        """``candidate_id``'s recorded run inputs (`optim.checks.recorded_inputs`): its run
        directory and its data passes' finished item directories."""
        from refactory_dsh.optim.checks import recorded_inputs

        return recorded_inputs(self.root / "runs" / str(candidate_id))

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

    def _train_gold_feedback(self) -> bool:
        """Whether this loop's own train evaluations ask for gold feedback: when the
        run's Task offers it (`task.offers_train_feedback`), as SkillOpt's train rollouts
        (``SkillOptLoop``) do, so both arms' train passes get the same information. Never
        val or test (`task.write_subset` refuses it), and not the memory pass's rollouts,
        SkillOpt's slow-update rollouts' counterpart. Read once; a Task that cannot be
        loaded fails the run here rather than run it without the feedback."""
        cached = self.__dict__.get("_offers_train_feedback")
        if cached is None:
            cached = task_module.offers_train_feedback(self.mode.state_dict())
            self.__dict__["_offers_train_feedback"] = cached
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
        if self.config.analysts:
            import warnings

            # Its own requests (SkillOpt's included) do not go through the model
            # proposer, so the stage runs only where it calls ``Run.propose``.
            warnings.warn(
                f"analysts is on, but the {getattr(strategy, 'name', type(strategy).__name__)} "
                "strategy builds its own optimizer requests: the analysts run only for "
                "proposals it asks of the loop's model proposer (Run.propose), if any",
                RuntimeWarning, stacklevel=2)

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
        """The evaluated candidates, newest last, bounded; each with how many of its edits
        are its own, its first As-built lines (cached: a candidate's graph never changes)
        and the epoch it was proposed in."""
        by_id = {c.id: c for c in self.population}
        built = self.__dict__.setdefault("_built_lines", {})
        rows = []
        for c in [c for c in self.population if c.evaluated][-self.config.history_limit:]:
            parent = by_id.get(c.parent or "")
            if c.id not in built:
                built[c.id] = (pipeline_view.changed_lines(
                    parent.state, c.state, self.protected, self.graders)[:3]
                    if parent is not None else [])
            rows.append({**self._row(c), "own": c.step_edits, "built": built[c.id],
                         "epoch": (self.proposals.get(c.id) or {}).get("epoch")})
        return rows

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
                    **({"repairs": candidate.repairs} if candidate.repairs else {}),
                    **({"notes": candidate.notes} if candidate.notes else {}),
                }
                for candidate in candidates
            ],
        )

    def read_pending(self) -> list[Candidate]:
        """The offspring the proposal node left, as candidates again."""
        path = self.root / PENDING
        if not path.exists():
            return []
        out = []
        for record in json.loads(path.read_text()):
            candidate = Candidate.from_state(
                record["state"], parent=record.get("parent"), edits=record.get("edits") or (),
                edit_reports=record.get("edit_reports") or (),
                step_edits=int(record.get("step_edits", 1)),
            )
            candidate.repairs = list(record.get("repairs") or ())
            candidate.notes = list(record.get("notes") or ())
            out.append(candidate)
        return out

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
            self._note_recorded(event, payload)

    def _journal_candidate(self, candidate: Candidate) -> None:
        self._journal(
            "candidate",
            id=candidate.id,
            parent=candidate.parent,
            edits=candidate.edits,
            edit_reports=candidate.edit_reports,
            step_edits=candidate.step_edits,
            state=candidate.state,
            **({"repairs": candidate.repairs} if candidate.repairs else {}),
            **({"notes": candidate.notes} if candidate.notes else {}),
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


if __name__ == "__main__":  # pragma: no cover - the entry point
    raise SystemExit(main())
