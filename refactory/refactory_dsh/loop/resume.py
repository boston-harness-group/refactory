
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

from refactory_dsh.loop.config import JOURNAL, LoopConfig, LoopError, NO_VALIDATION, OPTIMIZER_TRACE, SELECTIONS
from refactory_dsh.loop.evaluate import _other_value



class _ResumeMixin:

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
        self._note_removed_settings()
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
        sources = self._package_sources()
        if self.config.resume and saved.exists():
            self._check_package_sources(saved, sources)
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
            json.dumps({"mode": self.mode_id, **self.config.to_dict(),
                        "package_sources": sources}, indent=2, default=str)
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
            # The step reviews it cached go with it (`AutoresearchOptimizer._review_steps`).
            shutil.rmtree(self.root / "step_reviews", ignore_errors=True)
            # And the analyst stage's (`AutoresearchOptimizer._ask`).
            shutil.rmtree(self.root / "analyst_findings", ignore_errors=True)
            # And the item traces, numbered by this journal (`_write_traces`).
            from refactory_dsh import traces
            traces.clear(self.root)
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
        # Without the gate (no val items, or a custom strategy's own gate) the run selects
        # on train scores, which some settings cannot work with.
        ungated = not self.config.val_gate
        if (ungated and self.config.train_minibatch
                and len(self._train_ids()) > self.config.train_minibatch):
            # Without the gate the winner and the parents are picked by train score,
            # and on minibatches those come from different samples.
            raise LoopError(
                f"train_minibatch {self.config.train_minibatch} scores each epoch on a "
                "different train sample, so train scores of different epochs cannot pick "
                "the winner or the parents; give the run val items (a parts_manifest with a "
                "val part) so the validation gate selects, or score the whole train part "
                "(train_minibatch 0)")
        if ungated and (self.config.early_stop_patience or self.config.early_stop_at_full_val):
            raise LoopError(
                "early_stop_patience and early_stop_at_full_val measure the validation "
                "gate's best score, and this run has no validation gate (no val items, or a "
                "custom strategy with its own gate); give it val items (a parts_manifest with "
                "a val part), or leave them off"
            )
        if ungated and self.config.selection:
            raise LoopError(
                f"selection {self.config.selection} chooses by validation score, which only "
                "the validation gate measures, and this run has none (no val items, or a "
                "custom strategy with its own gate); give it val items, or leave selection "
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
                           **({"tools_not_offered": why_not} if (why_not := self._tools_not_offered())
                              else {}),
                           epochs=self.config.epochs, batch_size=self.config.batch_size,
                           optimizer=self.search_name())
            return

        self._journal("start", mode=self.mode_id, config=self.config.to_dict())
        # A tool step the request cannot offer is said once, never lost silently.
        tools_not_offered = self._tools_not_offered()
        if tools_not_offered:
            self._journal("tools_not_offered", reason=tools_not_offered)
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
            **({"tools_not_offered": tools_not_offered} if tools_not_offered else {}),
            resumed=False,
            population=len(self.population),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            optimizer=self.search_name(),
        )

    def _package_sources(self) -> dict[str, dict[str, str]]:
        """Which package, at which version, provides this run's strategy, optimizer
        and parent picker when a package does (``Package.strategies``,
        ``Package.optimizers``), by setting; the library's own are left out. Recorded
        in ``config.json``, so a resume can refuse a different implementation under
        the same name (`_check_package_sources`)."""
        from refactory_dsh import optim, registry, strategies

        found: dict[str, dict[str, str]] = {}
        for key, name, attribute, core in (
                ("strategy", self.config.strategy, "strategies", strategies.STRATEGIES),
                ("optimizer", self.config.optimizer, "optimizers", optim.REGISTRY),
                ("parent_picker", self.config.parent_picker, "optimizers", optim.REGISTRY)):
            if not name or name in core:
                continue
            entry = registry.contributions(attribute, self.project).get(name)
            if entry is not None:
                found[key] = {"name": name, "package": entry[1].name,
                              "version": entry[1].version}
        return found

    def _check_package_sources(self, saved: Path, sources: dict[str, dict[str, str]]) -> None:
        """On resume, refuse a package strategy, optimizer or parent picker that now
        comes from another package or another version than the run recorded: the
        epochs already run were searched by the recorded one. A run recorded before
        sources were kept is not checked."""
        try:
            recorded = json.loads(saved.read_text()).get("package_sources")
        except (OSError, ValueError, AttributeError):
            return
        if not isinstance(recorded, dict) or recorded == sources:
            return
        for key in sorted(set(recorded) | set(sources)):
            was, now = recorded.get(key), sources.get(key)
            if was != now:
                def shown(entry: Any) -> str:
                    return (f"{entry['name']} from package {entry['package']} "
                            f"{entry['version']}" if isinstance(entry, dict)
                            else "the library's own")
                raise LoopError(
                    f"this run's {key} was {shown(was)} and is now {shown(now)}; resume it "
                    "with the same package version, or start a new run")
    #: The settings that shape how a run selects, which a resumed run keeps as it
    #: recorded them (``_keep_recorded_run_shape``): the epochs already run selected so.
    RESUME_KEEPS = ("selection", "pool_size")

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
            if key == "selection":
                value = self._recorded_selection(recorded)
                if value not in SELECTIONS:
                    continue
            if key == "pool_size" and (isinstance(value, bool) or not isinstance(value, int)
                                       or value < 1):
                continue
            current = getattr(self.config, key)
            if self._shape_value(key, current) == self._shape_value(key, value):
                continue
            given = (key in overrides if isinstance(overrides, dict)
                     else current != getattr(defaults, key))
            if given:
                raise LoopError(
                    f"this run was started with {key} {self._shown(key, value)} and this "
                    f"resume asks for {key} {self._shown(key, current)}; resume it without "
                    f"setting {key} (or with {key} {self._shown(key, value)}), or start a new run")
            setattr(self.config, key, value)
            self._resumed_shape[key] = value
            Run(_loop=self).log(f"resume: keeping the run's recorded {key} {value!r} "
                                f"(this launch said {current!r})")

    def _recorded_selection(self, recorded: Mapping[str, Any]) -> Any:
        """The ``selection`` a resumed run's ``config.json`` records. One that records the
        gate on and no selection (or an empty one) selected by the default of its day,
        which for the shape that now defaults to ``checkpoint`` was ``single``: it
        resumes as ``single``, so its later epochs select by the rule its earlier ones
        did."""
        value = recorded.get("selection")
        if value in (None, "") and recorded.get("val_gate") is True:
            if self._default_selection == "checkpoint":
                return "single"
        return value

    def _note_removed_settings(self) -> None:
        """On resume, say in the run's log when its recorded ``config.json`` held a
        removed setting at a value other than the one every run now uses (#640):
        ``fresh_evidence`` false, or ``max_edits_per_proposal`` other than
        ``MAX_EDITS_PER_PROPOSAL``. The resumed epochs run as every run now does, so the
        run behaves differently from here on; the log is where a reader of its results
        finds out."""
        saved = self.root / "config.json"
        if not self.config.resume or not saved.exists():
            return
        try:
            recorded = json.loads(saved.read_text())
        except (OSError, ValueError):
            return
        if not isinstance(recorded, dict):
            return
        # The old command line stored switches as integers, so a run launched with
        # ``--fresh-evidence 0`` recorded ``0``; ``None`` (unset) meant the default, on.
        fresh = recorded.get("fresh_evidence")
        if fresh is not None and not isinstance(fresh, str) and not fresh:
            Run(_loop=self).log(
                "resume: this run was recorded with fresh_evidence false; that setting is "
                "gone and fresh evidence is always on, so the epochs from here on gather "
                "fresh evidence where the earlier ones did not")
        edits = recorded.get("max_edits_per_proposal")
        if edits is not None and edits != MAX_EDITS_PER_PROPOSAL:
            Run(_loop=self).log(
                f"resume: this run was recorded with max_edits_per_proposal {edits!r}; that "
                f"setting is gone and the limit is {MAX_EDITS_PER_PROPOSAL} edits per proposal, "
                "so the epochs from here on take proposals under that limit")

    def _shown(self, key: str, value: Any) -> str:
        """A ``RESUME_KEEPS`` value as the run's settings hold it: an empty ``selection``
        as the default it resolves to."""
        if key == "selection" and not value:
            return f"{self._shape_value(key, value)} (unset: the default)"
        return str(value)

    def _shape_value(self, key: str, value: Any) -> Any:
        """A ``RESUME_KEEPS`` value as it acts: an empty ``selection`` is the default
        it resolves to."""
        if key == "selection" and not value:
            return self._default_selection
        return value

    def _resolve_val_gate(self) -> None:
        """Decide ``val_gate``: on when the parts manifest the run will load has val
        items and the strategy is a built-in one (a custom strategy such as SkillOpt
        keeps its own gate), off otherwise. Off for want of val items, the run says it
        selects on train scores (``NO_VALIDATION``), in its events and its strategy log
        now, and in its report. Runs before ``config.json`` is written, so the run
        records the decision.

        It is not a setting: a value given in code that differs from the decision is
        refused, and so is resuming a run that recorded another decision -- one started
        with the gate off although it had val items (``--val-gate 0``, the default
        before #627), whose epochs selected on train scores."""
        self.selection_warning = ""
        if self._custom_strategy is not None:
            decided = False
            self._val_gate_basis = f"the {self._custom_strategy.name} strategy keeps its own gate"
        else:
            decided = bool(self._manifest_val_ids())
            self._val_gate_basis = ("the parts manifest has val items" if decided
                                    else "no val items")
        recorded = self._recorded_val_gate()
        if isinstance(recorded, bool) and recorded != decided:
            raise LoopError(
                f"this run recorded val_gate {str(recorded).lower()}, and the run now decides "
                f"{str(decided).lower()} ({self._val_gate_basis}). "
                + ("Selecting on train scores while the run has val items (--val-gate 0) "
                   "is no longer supported" if decided else
                   "Its validation gate cannot run without val items")
                + "; start a new run instead of resuming this one")
        given = self.config.val_gate
        if given is not None and bool(given) != decided:
            raise LoopError(
                f"val_gate is not a setting: this run decides {str(decided).lower()} "
                f"({self._val_gate_basis}), and {str(bool(given)).lower()} was given")
        if isinstance(recorded, bool):
            self._val_gate_basis = f"resumed: {self._val_gate_basis}"
        self.config.val_gate = decided
        if not decided and self._custom_strategy is None:
            self.selection_warning = NO_VALIDATION
            Run(_loop=self).log(f"warning: {NO_VALIDATION}")

    def _recorded_val_gate(self) -> bool | None:
        """The ``val_gate`` a resumed run recorded in its ``config.json``; ``None`` for a
        new run or one that recorded none."""
        saved = self.root / "config.json"
        if not self.config.resume or not saved.exists():
            return None
        try:
            recorded = json.loads(saved.read_text())
        except (OSError, ValueError):
            return None
        value = recorded.get("val_gate") if isinstance(recorded, dict) else None
        return value if isinstance(value, bool) else None

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

    def _seed_population(self, seed: Candidate) -> None:
        """Fill the starting batch with neighbours of the seed.

        A population of one would spend the first epoch evaluating a single point
        and look like a search while doing almost nothing. It is also swept rather
        than sampled: one candidate per movable knob, each at a value the default
        is not, so the first epoch produces a gradient with *contrast* in it. A
        purely structural starting population gives the optimizer plenty of
        candidates and no evidence about which value of anything is worth having.

        When the optimizer uses a model (autoresearch, or a parent picker wrapping
        one), the sweep is skipped: round 1 scores the seed alone, and from then
        on the model proposer writes all candidates. Swept neighbours would stay
        in the population unjudged by the integrity judge and dilute the model's
        proposals.
        """
        wanted = max(0, int(self.config.batch_size) - 1)
        if not wanted:
            return
        if getattr(self.optimizer, "uses_model", False):
            return
        sampler = RandomOptimizer(seed=self.config.seed)
        sampler.allow_code_nodes = bool(self.config.allow_code_nodes)
        sampler.step_routes = self.optimizer.step_routes
        sampler.sealed_tools = self.optimizer.sealed_tools
        sampler.new_step_tools = bool(self.config.new_step_tools)
        sampler.sealed_check = self.optimizer.sealed_check
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
            self._note_recorded(event, record)
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
                candidate.repairs = list(record.get("repairs") or ())
                candidate.notes = list(record.get("notes") or ())
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
                rejection = Rejection(
                    attempt=int(record.get("attempt") or 0),
                    edit=dict(record.get("edit") or {}),
                    reasons=list(record.get("reasons") or []),
                    repaired=str(record.get("repaired") or ""),
                    dropped=str(record.get("dropped") or ""),
                )
                self.rejections.append(rejection)
                # A journal written before rejections carried their epoch: the epoch
                # open when it was written.
                self._round_refusals.setdefault(
                    int(record["epoch"]) if record.get("epoch") is not None else self.epoch,
                    []).append(rejection)
            elif event == "repair_calls":
                # The slim repair calls an epoch made, so a resumed run keeps the bound.
                calls = self.optimizer.__dict__.setdefault("_repair_calls", {})
                epoch = int(record.get("epoch") or 0)
                calls[epoch] = max(int(calls.get(epoch, 0)), int(record.get("calls") or 0))
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
            elif event == "optimizer_memory":
                self._memory = dict(record.get("memory") or {})
                self._memory_passes.append({k: v for k, v in record.items()
                                            if k not in ("event", "at", "ts", "memory")})
            elif event == "epoch_current":
                self._epoch_currents[int(record.get("epoch") or 0)] = str(record.get("id"))
            elif event == "memory_rollout":
                rescores += 1
                rescore_measurements += int((record.get("measurements") or {}).get("count", 1))
                held = self._memory_rollouts.setdefault(
                    (str(record.get("id")),
                     str(record.get("key") or f"e{int(record.get('epoch') or 0)}")), {})
                for item in record.get("records") or ():
                    held.setdefault(str(item.get("id")), dict(item))
            elif event == "analyst_findings":
                self._analyses.append({key: value for key, value in record.items()
                                       if key not in ("event", "at", "ts")})
            elif event == "integrity_verdict":
                self._remember_integrity({k: v for k, v in record.items() if k != "event"})
            elif event == "integrity_policy":
                self.__dict__["_integrity_policy_journaled"] = {
                    k: v for k, v in record.items() if k != "event"}
            elif event in ("rescore", "rescore_failed"):
                rescores += 1
                rescore_measurements += int((record.get("measurements") or {}).get("count", 1))
            elif event == "smoke":
                # A smoke run is a measurement, never an evaluation; its verdict is kept so
                # a resumed run does not smoke the same graph again.
                cid = str(record.get("id"))
                rescore_measurements += int((record.get("measurements") or {}).get("count", 0))
                self._smoke_verdicts[cid] = {k: v for k, v in record.items() if k != "event"}
                self._smoke_runs[cid] = self._smoke_runs.get(cid, 0) + 1
            elif event == "wiring_review":
                # Kept so a resumed run does not ask about the same edit's wiring again.
                held = {k: v for k, v in record.items() if k != "event"}
                for key in (record.get("id"), record.get("became")):
                    if key:
                        self._wiring_reviews[str(key)] = held
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
            elif event == "harness_incident":
                self._note_incident(
                    {"candidate": record.get("id"),
                     **{k: record.get(k) for k in ("part", "items", "reran", "recovered",
                                                   "not_scored", "workdir")}})
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
            elif event == "item_key_secret":
                self._item_key_secret = str(record.get("secret") or "") or None
            elif event == "evaluation_interrupted":
                if record.get("outcome") in ("stopped", "recorded_failed") and record.get("key"):
                    self._interrupted_stops[str(record["key"])] = int(record.get("stops") or 0)
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
