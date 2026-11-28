
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

from refactory_dsh.loop.config import NOT_APPLIED, _train_key, _values_before, item_time_spread
from refactory_dsh.loop.evaluate import _new_wiring_warnings, _wiring_warnings



class _ContextMixin:

    # ── context ────────────────────────────────────────────────────────────
    def _context(self) -> ProposalContext:
        # Under ``pool`` the whole pool: every parent a proposal may name is shown. Under
        # ``checkpoint``, the checkpoints chosen and ordered by nothing val decided.
        board = (self._checkpoint_board(8) if self.selection_mode() == "checkpoint"
                 else self.leaderboard(None if self.selection_mode() == "pool" else 8))
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
            graders=list(self.graders),
            graph_facts=self._graph_facts(board, described),
            # A refusal a slim repair fixed is not fed back: its fix is the candidate.
            rejections=summarise_reasons(
                [
                    reason
                    for rejection in self.rejections[-8:]
                    if not rejection.repaired
                    for reason in rejection.reasons
                ]
            ),
            refused=refused_entries(self.rejections[-8:]),
            # ...but what its fix dropped of the proposal's text is: it did not land.
            repair_losses=repair_losses(self.rejections[-8:]),
            preset=self.config.edit_preset,
            strategy={**dict(self.strategy_decision), **({"search_contract": self.config.resolved_profile["contract"]} if self.config.resolved_profile.get("contract") else {})},
            # Read now, not at construction: the point of steering a search is that a
            # note added while it runs reaches the next proposal.
            steering=self._steering_block(),
            item_records=self._with_step_sources(self._with_item_history(
                self._with_item_meta(self._parent_train_records(board))),
                self._train_records_owner(board)[0]),
            parent_item_records=self._with_step_sources(self._with_item_history(
                self._with_item_meta(self._parent_train_records(board))),
                self._train_records_owner(board)[0]),
            graph_note=graph_note,
            edit_pairs=self._edit_pairs(board),
            reasoning=self._reasoning_entries(board),
            ledger=self._ledger() if self.config.run_ledger else [],
            memory=dict(self._memory) if self.config.lessons else {},
            added_steps=self._added_steps() if self.config.step_review else [],
            structure_guidance=bool(self.config.structure_guidance),
            # What the mode's author says is worth doing, and the edits to try first
            # (``Mode.guidance``/``preferred_edits``): the request has their sections.
            guidance=self.mode.guidance(),
            preferred_edits=list(self.mode.preferred_edits()),
            analysts=self._analyst_settings(),
            integrity=self._integrity_feedback(),
            max_edits=MAX_EDITS_PER_PROPOSAL,
            parsimony=float(self.config.parsimony),
            code_nodes=self._code_nodes_offered(sandbox),
            sandbox=sandbox,
            score_is_item_mean=bool(self._score_is_item_mean),
            val_items=self._gate_val_items(),
            seed_knob_values=self._seed_knob_values(),
            parent=self._edited_id(board),
            val_scores=self._val_scores(),
            val_item_scores=self._val_item_scores(),
            selection=self.selection_mode(),
            population_parents=(self.selection_mode() == "pool" and self._population_policy()),
            edit_levels=[l for l in self._levels_wanted if l not in self._levels_done],
            strategy_name=str(getattr(self._custom_strategy, "name", "") or self.name),
            new_node_tools=self._tools_offered(),
            step_routes=(self.optimizer.step_routes.offered()
                         if getattr(self.optimizer, "step_routes", None) is not None else {}),
            time_facts=self._time_facts(board),
        )

    def _checkpoint_board(self, n: int) -> list[Candidate]:
        """Under ``checkpoint``, the candidates a request is built from and lists under
        "Recent results": the current one, then the other checkpoints newest first, at most
        ``n``. The leaderboard (the best on val first, then by val) would tell the proposer
        which pipeline scored best on val, through its train score and the rows it keeps;
        val only picks the winner and the stop."""
        gate = self._gate()
        applied = set(self._gate_accepted)
        held = [c for c in self._evaluated() if c.id in applied and c.id not in self.rejected]
        current = [c for c in held if c.id == gate.current_id]
        return (current + [c for c in reversed(held) if c.id != gate.current_id])[:n]

    def _time_facts(self, board: list[Candidate]) -> dict[str, Any]:
        """Time in numbers for the request (`optim.base.time_lines`): for the candidate
        being edited, how long its
        train items took each, from each item's own record (`items.item_duration`): median,
        90th percentile and longest, over the items the harness did not time out (those
        are said once, in "Item results"). Never its wall time over its items: that is set by
        its slowest item."""
        facts: dict[str, Any] = {}
        _owner, records = self._train_records_owner(board)
        if not records:
            return facts
        from refactory_dsh.optim.base import harness_timeout

        facts.update(item_time_spread([r for r in records if not harness_timeout(r)]))
        return facts

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
        scores = dict(self._gate().cache)
        # On the items every rolled-out candidate was scored on, when a harness incident
        # left one out of any (`shared_scores`): every parent pick ranks on these.
        rolled = [cid for cid in scores if self._part_results.get(cid, {}).get("val")]
        shared = self.shared_scores(rolled) if len(rolled) > 1 else None
        return {**scores, **shared[0]} if shared is not None else scores

    def _val_item_scores(self) -> dict[str, dict[str, float]]:
        """Each judged candidate's per-item val scores (`part_item_scores`, keyed by
        `opaque_item_key`) under ``val_gate``, for a parent policy that compares
        candidates item by item (``ParentSelector.uses_item_scores``: the optimizer
        itself, or its parent picker); empty otherwise, and never a test score. Worked
        out only for such a policy, so no other run pays for it, and a proposer paired
        with the picker never sees them (``PickedParent.propose``)."""
        if not self.config.val_gate or self._custom_strategy is not None:
            return {}
        policy = getattr(self.optimizer, "picker", self.optimizer)
        if not getattr(policy, "uses_item_scores", False):
            return {}
        found: dict[str, dict[str, float]] = {}
        for cid in self._gate().cache:
            scores = self.part_item_scores(cid, "val")
            if scores:
                found[cid] = scores
        return found

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
            record = self.proposals.get(child.id) or {}
            child_records = list(self._train_item_records.get(child.id, []))
            from refactory_dsh.optim.edit_effect import edit_effect

            effect = edit_effect(parent_records, child_records, self._reply_judge(),
                                 new_steps=sys.modules["refactory_dsh.loop"]._new_steps(parent, child))
            pairs.append({
                "child": child.id, "parent": parent.id if parent else None,
                "edit": child.own_edits(), "gate": self._gate_decision(child.id),
                # The values its set_knob edits replaced, so its edits show as changes.
                "before": _values_before(parent, child.own_edits()),
                "child_records": self._with_step_sources(child_records, child),
                "parent_records": self._with_step_sources(parent_records, parent),
                "same_seed": same,
                # How a structural edit moved the items it changed (`optim.edit_effect`).
                **({"effect": effect} if effect else {}),
                # What the proposal predicted and said, for the prediction check, and
                # what the system did beside its edits (the core operators' wiring).
                **{key: record[key] for key in ("prediction", "addresses", "flow_change")
                   if record.get(key)},
                **({"notes": list(child.notes)} if child.notes else {}),
                **({"wiring": built} if parent is not None
                   and (built := pipeline_view.changed_lines(
                       parent.state, child.state, self.protected, self.graders)) else {}),
                "steps": sorted(set(((child.state.get("graph") or {}).get("nodes") or {}))
                                | set(((parent.state.get("graph") or {}).get("nodes") or {})
                                      if parent else ())),
            })
        return pairs

    def _rescore_spread(self, candidate_id: str) -> float | None:
        """How far ``candidate_id``'s train score (its items' mean) moved between the
        epochs it was scored in this run; ``None`` before a second score."""
        scores = []
        for records in (self._records_by_epoch.get(candidate_id) or {}).values():
            values = [float(r["score"]) for r in records
                      if isinstance(r.get("score"), (int, float))]
            if values:
                scores.append(sum(values) / len(values))
        return round(max(scores) - min(scores), 3) if len(scores) > 1 else None

    def _reply_judge(self) -> Any:
        """How a step's reply is read as an item's answer, from the mode's Tasks
        (`task.reply_judge`); ``None`` when no Task reads replies. Looked up once."""
        if not hasattr(self, "_reply_judge_found"):
            from refactory_dsh.task import reply_judge

            self._reply_judge_found = reply_judge(self.mode.state_dict())
        return self._reply_judge_found

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
        whether the two share a seed. Without ``val_gate``, or under ``checkpoint``, every
        train rollout uses one seed, so they always do."""
        latest = list(self._train_item_records.get(parent_id, []))
        if not self._epoch_seeds():
            return latest, True
        at = self._records_epoch.get(child_id)
        history = self._records_by_epoch.get(parent_id, {})
        if at is not None and at in history:
            return list(history[at]), True
        return latest, at is not None and self._records_epoch.get(parent_id) == at

    def _round_records(self, parent_id: str, child_id: str
                       ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
        """A child's train records from the round it was evaluated in, its parent's from
        that same round, and whether they share that round's seed.

        A kept child becomes the edited candidate and is re-scored in later rounds
        (fresh evidence), which replaces its latest records; the round it was first
        evaluated in stays in ``_records_by_epoch``, and only the parent's records from
        that round share its seed. Without ``val_gate`` every train rollout uses the
        run's seed, so the latest records of both always do. When the parent has no
        records from the child's round the pair does not share a seed (``False``) and
        the parent's latest are returned for a caller that still wants them. Under
        ``checkpoint`` (one seed, no re-score) each one's own run is its only one."""
        child_latest = list(self._train_item_records.get(child_id, []))
        parent_latest = list(self._train_item_records.get(parent_id, []))
        if not self._epoch_seeds():
            return parent_latest, child_latest, True
        rounds = self._records_by_epoch.get(child_id) or {}
        at = min(rounds) if rounds else self._records_epoch.get(child_id)
        # Its latest records are its evaluation round's until a re-score replaces them.
        child = (list(rounds[at]) if rounds and self._records_epoch.get(child_id) != at
                 else child_latest)
        parent_rounds = self._records_by_epoch.get(parent_id) or {}
        if at is not None and at in parent_rounds:
            return list(parent_rounds[at]), child, True
        return parent_latest, child, False

    def _reasoning_entries(self, board: list[Candidate]) -> list[dict[str, Any]]:
        """The optimizer's own reasoning about recent candidates -- the line from the
        seed to the edited candidate, and the latest refused children -- beside what
        happened to each. Train scores, train item errors and gate decisions only."""
        from refactory_dsh.optim.base import REASONING_ENTRIES, edit_change, pipeline_item_errors

        by_id = {c.id: c for c in self.population}
        line: list[Candidate] = []
        node = by_id.get(self._edited_id(board) or "")
        while node is not None and node.parent:
            line.append(node)
            node = by_id.get(node.parent)
        line.reverse()
        # The latest refused children: rejected by the gate, or (``checkpoint``) not applied.
        rejected = [c for c in self.population if c.evaluated and c not in line
                    and self._gate_decision(c.id) in ("rejected", NOT_APPLIED)]
        chosen = (line + rejected[-REASONING_ENTRIES:])
        chosen.sort(key=lambda c: self.population.index(c))
        entries = []
        for candidate in chosen[-REASONING_ENTRIES:]:
            record = self.proposals.get(candidate.id) or {}
            if record.get("origin") != "optimizer":
                continue
            parent = by_id.get(candidate.parent or "")
            # The child as it was judged (its evaluation round's records, not a later
            # re-score of it as the edited candidate) against its parent from that round.
            parent_records, records, same = (
                self._round_records(parent.id, candidate.id) if parent is not None
                else ([], list(self._train_item_records.get(candidate.id) or ()), None))
            records = records if candidate.evaluated else []
            change = (edit_change(parent_records, records)
                      if parent is not None and candidate.evaluated else None)
            entries.append({
                "id": candidate.id, "epoch": record.get("epoch"),
                # What this edit added: its new or changed steps judged against its
                # parent, less what the parent already had.
                "warnings": (_new_wiring_warnings(candidate.state, parent.state)
                             if parent is not None else _wiring_warnings(candidate.state)),
                "edit": candidate.own_edits(), "rationale": record.get("rationale"),
                "before": _values_before(parent, candidate.own_edits()),
                "prediction": record.get("prediction"), "diagnosis": record.get("diagnosis"),
                "addresses": record.get("addresses"), "flow_change": record.get("flow_change"),
                "notes": list(candidate.notes or ()),
                "wiring": (pipeline_view.changed_lines(
                       parent.state, candidate.state, self.protected, self.graders)
                           if parent is not None else []),
                "considered": record.get("considered"),
                "gate": self._gate_decision(candidate.id), "score": candidate.score,
                "same_seed": same,
                "fixed": len(change["fixed"]) if change and change["shared"] else None,
                "broken": len(change["broken"]) if change and change["shared"] else None,
                "errors": pipeline_item_errors(records), "items": len(records),
                # How a structural change moved the items it changed (`optim.edit_effect`),
                # for building on a rejected one; whether it added a step (reviewed under
                # "Did the added steps contribute?").
                **(self._reasoning_effect(parent, candidate, parent_records, records)
                   if parent is not None and candidate.evaluated and same
                   and candidate.is_structural() else {}),
            })
        return entries

    def _reasoning_effect(self, parent: Candidate, candidate: Candidate,
                          parent_records: list[dict[str, Any]],
                          records: list[dict[str, Any]]) -> dict[str, Any]:
        from refactory_dsh.optim.edit_effect import edit_effect
        from refactory_dsh.optim.feedback import added_steps

        effect = edit_effect(parent_records, records, self._reply_judge(),
                             new_steps=sys.modules["refactory_dsh.loop"]._new_steps(parent, candidate))
        return {**({"effect": effect} if effect else {}),
                "added": bool(added_steps(parent.state, candidate.state))}

    # ── the memory pass (``LoopConfig.lessons``; `optim.memory`) ──────────────
    def _stopped_inside(self, exc: BaseException) -> None:
        """Journal a person's stop met inside a step (`RunStopped`), as the stop it is."""
        self._journal("stopped_inside_step", epoch=self.epoch, why=str(exc))
        self._progress("control", action="stop", inside_step=True)

    def _stop_requested(self) -> bool:
        """Whether a person has asked this run to stop (``refactory stop``), read from the
        control file without consuming the request (`_apply_control` acts on it)."""
        from refactory_dsh.control import Control

        try:
            request = Control(self.root).read_request()
        except Exception:  # noqa: BLE001 - an unreadable request is not a stop
            return False
        return bool(request) and str(request.get("action")) == "stop"

    def _memory_on(self) -> bool:
        """Whether this run makes memory notes: the switch on, a model proposer that can
        write one, and the loop's own epochs (a custom strategy, such as the SkillOpt
        port, keeps its own)."""
        return (bool(self.config.lessons) and self._custom_strategy is None
                and callable(getattr(self.optimizer, "write_memory", None)))

    def _current_now(self) -> str | None:
        """The current candidate: the gate's under ``val_gate``, else the train leader."""
        return self._incumbent_id()

    def _memory_pass(self) -> None:
        """This epoch's memory pass, once (`optim.memory`): from epoch 2, the candidate
        current when the previous epoch proposed against the one current now, on
        ``MEMORY_SAMPLES`` train items, one optimizer-model call writing the note every
        later request reads. A pass already made (a resumed epoch) is not made again.
        Records the candidate current now for the next epoch's pass."""
        if not self._memory_on():
            return
        epoch, current = self.epoch, self._current_now()
        if current and epoch not in self._epoch_currents:
            self._epoch_currents[epoch] = current
            self._journal("epoch_current", epoch=epoch, id=current)
        if epoch < 2 or any(p.get("epoch") == epoch for p in self._memory_passes):
            return
        prev = self._epoch_currents.get(epoch - 1)
        current = self._epoch_currents.get(epoch, current)
        by_id = {c.id: c for c in self.population}
        if not prev or not current or prev not in by_id or current not in by_id:
            # SkillOpt skips a pass with no previous epoch's skill (the port's
            # skip_no_prev_skill): nothing to compare.
            return
        self._make_memory(epoch, by_id[prev], by_id[current])

    def _side_records(self, cid: str, epoch: int, key: str) -> dict[str, dict[str, Any]]:
        """One side of a memory pass, by item: for a side read from the run's rollouts
        (``key`` ``e<epoch>``), ``cid``'s records from ``epoch``'s rollout (that epoch's
        train seed: fresh evidence, or a candidate created then), or without the gate (one
        seed for the whole run) its records; then what memory passes rolled out for this
        side (``key``). A fresh side (``fresh<epoch>``) holds only the latter."""
        from refactory_dsh.optim.base import _by_item

        found: dict[str, dict[str, Any]] = {}
        if not key.startswith("fresh"):
            held = (self._records_by_epoch.get(cid) or {}).get(epoch)
            if held is None and not self._epoch_seeds():
                held = self._train_item_records.get(cid)
            found.update(_by_item(held or ()))
        for item, record in (self._memory_rollouts.get((cid, key)) or {}).items():
            found.setdefault(item, record)
        return found

    def _memory_rollout(self, candidate: Candidate, items: list[str], epoch: int,
                        for_epoch: int, key: str) -> int:
        """Roll ``candidate`` out on ``items`` with ``epoch``'s train seed for the memory
        pass of ``for_epoch``: counted as an evaluation, its rollouts counted, its records
        journaled (``memory_rollout``) so a resumed run reuses them; the candidate's own
        records and score are left alone. Returns the rollouts made (0 when a cap or a
        budget leaves no room)."""
        if self.evaluations >= self.config.max_evaluations:
            return 0
        if (self.config.max_rollouts is not None
                and self.rollouts + len(items) * max(1, int(self.config.repeats or 1))
                > self.config.max_rollouts):
            return 0
        if self.budget:
            from refactory_dsh.budgets import BudgetExhausted
            try:
                self.budget.checkpoint("work" if self._owns_resource_budget else None)
            except BudgetExhausted as exc:
                self.bound = str(exc)
                return 0
        workdir = self._memory_dir(candidate.id, for_epoch, key)
        self._archive_unfinished(candidate.id, workdir)

        def prepare() -> None:
            task_module.make_manifest(workdir, train_ids=items)

        result, measurements = self._settled_measure(
            candidate, workdir, task_module.TRAIN, prepare, seed=self._train_seed(epoch),
            items=items)
        if not measurements.get("count"):
            return 0
        records = [dict(r) for r in result.item_records or ()]
        with self._state_lock:
            self.evaluations += 1
            if not measurements.get("cached_from"):
                self.rollouts += len(records) or 1
            held = self._memory_rollouts.setdefault((candidate.id, key), {})
            for record in records:
                held.setdefault(str(record.get("id")), record)
            self._journal("memory_rollout", id=candidate.id, epoch=epoch, for_epoch=for_epoch,
                          key=key,
                          seed=self._train_seed(epoch), items=list(items),
                          exit_ok=result.exit_ok, error=result.error, records=records,
                          measurements=measurements)
            self._journal("rollouts", rollouts=self.rollouts)
        self._progress("memory_rollout", candidate=candidate.id, epoch=for_epoch,
                       items=len(items), records=len(records), exit_ok=result.exit_ok)
        return len(records) or 1

    def _lineage_edits(self, prev: Candidate, current: Candidate) -> dict[str, Any]:
        """What separates ``prev`` from ``current``: their nearest common ancestor and the
        candidates from it (exclusive) to each, with their own edits, oldest first
        (``{"common", "prev", "curr"}``). ``prev`` an ancestor of ``current`` leaves
        ``prev``'s path empty; with no common ancestor ``common`` is ``None`` and each
        path is its candidate's whole lineage."""
        by_id = {c.id: c for c in self.population}

        def chain(candidate: Candidate) -> list[Candidate]:
            out, at, seen = [], candidate, set()
            while at is not None and at.id not in seen:
                seen.add(at.id)
                out.append(at)
                at = by_id.get(at.parent) if at.parent else None
            return out

        def entry(candidate: Candidate) -> dict[str, Any]:
            record = self.proposals.get(candidate.id) or {}
            return {"id": candidate.id, "parent": candidate.parent,
                    "edits": candidate.own_edits(), "rationale": record.get("rationale")}

        before, after = chain(prev), chain(current)
        ids = {c.id for c in before}
        common = next((c.id for c in after if c.id in ids), None)

        def path(line: list[Candidate]) -> list[dict[str, Any]]:
            upto = [c for c in line[: next((i for i, c in enumerate(line) if c.id == common),
                                           len(line))]]
            return [entry(c) for c in reversed(upto)]

        return {"common": common, "prev": path(before), "curr": path(after)}

    def _make_memory(self, epoch: int, prev: Candidate, current: Candidate) -> None:
        """The pass itself (`_memory_pass`): sample, gather or roll out both sides (or,
        with nothing kept, take the round's rejected change, `_rejected_side`), then ask,
        keep and journal (`_make_memory_pass`)."""
        from refactory_dsh.optim import memory

        same = prev.id == current.id
        tried, refused = self._tried_in_round(epoch - 1)
        # Nothing kept: the incumbent against the round's rejected change that moved the
        # most items, on the items it moved, both traces from the round it was evaluated
        # in (one seed), so the note can say why the tried change failed rather than
        # read two rollouts of one pipeline (`_rejected_side`). Without one, SkillOpt's
        # two rollouts of the unchanged pipeline.
        rejected = self._rejected_side(current, tried) if same else None
        if rejected is not None:
            self._make_memory_pass(epoch, current, rejected[0], same=False,
                                   sample=rejected[1], records=rejected[2:4],
                                   sides=[(current, 0, "", rejected[4]),
                                          (rejected[0], 0, "", rejected[5])],
                                   evaluated=0, rolled=0, tried=tried, refused=refused,
                                   rejected=rejected[6])
            return
        if self.selection_mode() == "checkpoint":
            self._make_checkpoint_memory(epoch, prev, current, same=same, tried=tried,
                                         refused=refused)
            return
        # The two rollouts, each (candidate, the epoch whose seed it used, its key, what it
        # is). Under the gate: both candidates' rollouts of this epoch, or, when nothing
        # replaced the current, its previous epoch's and this epoch's. Without the gate
        # (one seed for the run): each candidate's rollout when it was scored, or, when
        # nothing replaced it, that rollout and a fresh one made for this pass.
        if self.config.val_gate:
            def gated(at: int) -> str:
                return f"the rollout of epoch {at} (that epoch's train seed)"

            sides = [(prev, epoch - 1 if same else epoch, f"e{epoch - 1 if same else epoch}",
                      gated(epoch - 1 if same else epoch)),
                     (current, epoch, f"e{epoch}", gated(epoch))]
        else:
            scored = "its rollout when it was scored (the run's seed)"
            sides = [(prev, epoch, f"e{epoch}", scored),
                     (current, epoch, f"fresh{epoch}" if same else f"e{epoch}",
                      f"a rollout made for this epoch's memory pass (the run's seed)"
                      if same else scored)]
        sample = memory.sample_ids(self._train_ids(), int(self.config.seed), epoch)
        evaluated = rolled = 0
        records = []
        for candidate, at, key, _label in sides:
            held = self._side_records(candidate.id, at, key)
            missing = [item for item in sample if item not in held]
            if missing:
                made = self._memory_rollout(candidate, missing, at, epoch, key)
                evaluated += 1 if made else 0
                rolled += made
                held = self._side_records(candidate.id, at, key)
            records.append(self._with_meta(held))
        self._make_memory_pass(epoch, prev, current, same=same, sample=sample,
                               records=records, sides=sides, evaluated=evaluated,
                               rolled=rolled, tried=tried, refused=refused)

    def _make_checkpoint_memory(self, epoch: int, prev: Candidate, current: Candidate, *,
                                same: bool, tried: list[dict[str, Any]],
                                refused: list[dict[str, Any]]) -> None:
        """The memory pass under ``checkpoint``: the candidate current when the last epoch
        proposed against the one applied since, each from its own train run (one seed for
        the run), on items sampled from those both ran; nothing is rolled out. With
        nothing applied and no change that was not applied moving an item
        (`_rejected_side`), there is nothing to compare: no call is made and the note is
        kept, marked not updated."""
        from refactory_dsh.optim import memory
        from refactory_dsh.optim.base import _by_item

        if same:
            self._keep_memory({"epoch": epoch, "generation": self.generation, "prev": prev.id,
                               "curr": current.id, "same": True, "evaluated": 0, "rollouts": 0,
                               "error": "nothing was applied this epoch and no change that "
                                        "was not applied moved an item"}, None)
            return
        held = [_by_item(self._train_item_records.get(c.id) or ()) for c in (prev, current)]
        sample = memory.sample_ids(sorted(set(held[0]) & set(held[1])), int(self.config.seed),
                                   epoch)
        self._make_memory_pass(epoch, prev, current, same=False, sample=sample,
                               records=[self._with_meta(h) for h in held],
                               sides=[(prev, 0, "", "its own train run"),
                                      (current, 0, "", "its own train run (the same train seed)")],
                               evaluated=0, rolled=0, tried=tried, refused=refused)

    def _with_meta(self, held: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        """Records by item, each with its declared metadata and context when the mode
        reads any (`_declared`)."""
        return {item: self._declared(rec) for item, rec in held.items()}

    def _declared(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """``record`` with what its Task declares for the optimizer: its metadata
        (``item_meta``, the item header) and its context (``item_context``, shown in full
        after its input, `evidence.render_item`), each only when a Task declares it."""
        out = dict(record)
        if self._item_meta_reader() is not None:
            out["item_meta"] = self._item_meta(out)
        context = self._item_context_reader()
        if context is not None:
            out["item_context"] = context(out)
        return out

    def _item_context_reader(self) -> Any:
        """How a record's declared context is read (`task.item_metadata_reader` with
        ``context``), once; ``None`` when no Task declares any."""
        if getattr(self, "_item_context", None) is None:
            from refactory_dsh.task import item_metadata_reader

            try:
                self._item_context = (item_metadata_reader(self.mode.state_dict(), context=True),)
            except Exception:  # noqa: BLE001 - a mode that cannot say declares nothing
                self._item_context = (None,)
        return self._item_context[0]

    def _rejected_side(self, incumbent: Candidate, tried: Sequence[Mapping[str, Any]]
                       ) -> tuple[Any, ...] | None:
        """For a memory pass whose incumbent nothing replaced: the round's evaluated,
        not-kept child of the incumbent that fixed or broke the most train items (the
        newest on a tie), with the items it changed (broken first, at most
        ``memory.MEMORY_SAMPLES``), both sides' records from its evaluation round (one
        seed, `_round_records`), each side's label, and what the request says of it
        (``{"child", "changed", "shown", "others"}``: the other rejected changes are
        under "What was tried this epoch", with their edits and how they moved their
        items, not compared here). ``None`` when no such child changed an item under a
        shared seed."""
        from refactory_dsh.optim import memory
        from refactory_dsh.optim.base import _by_item, edit_change
        from refactory_dsh.optim.feedback import KEPT_WORDS

        by_id = {c.id: c for c in self.population}
        best = None
        others = []
        for entry in tried:
            child = by_id.get(str(entry.get("id")))
            if (child is None or child.parent != incumbent.id or not entry.get("evaluated", True)
                    or entry.get("gate") in KEPT_WORDS):
                continue
            parent_records, child_records, same_seed = self._round_records(incumbent.id,
                                                                           child.id)
            if not same_seed or not child_records:
                continue
            moved = edit_change(parent_records, child_records)
            changed = [*moved["broken"], *moved["fixed"]]
            if not changed:
                continue
            others.append(child.id)
            if best is None or len(changed) >= len(best[1]):
                best = (child, changed, parent_records, child_records, moved)
        if best is None:
            return None
        child, changed, parent_records, child_records, moved = best
        shown = changed[:memory.MEMORY_SAMPLES]
        at = "its rollout in the round the change was evaluated in (one train seed for both)"
        return (child, shown, self._with_meta(_by_item(parent_records)),
                self._with_meta(_by_item(child_records)), at, at,
                {"child": child.id,
                 "noun": (memory.NOT_KEPT_NOUN if not self.config.val_gate
                          else memory.NOT_APPLIED_NOUN if not self._epoch_seeds()
                          else memory.REJECTED_NOUN),
                 "changed": len(changed), "shown": len(shown),
                 "broken": len(moved["broken"]), "fixed": len(moved["fixed"]),
                 "shared": moved["shared"], "others": [c for c in others if c != child.id]})

    def _make_memory_pass(self, epoch: int, prev: Candidate, current: Candidate, *,
                          same: bool, sample: list[str], records: Sequence[Mapping[str, Any]],
                          sides: Sequence[tuple[Any, ...]], evaluated: int, rolled: int,
                          tried: list[dict[str, Any]], refused: list[dict[str, Any]],
                          rejected: Mapping[str, Any] | None = None) -> None:
        """Ask for the note from the two sides' ``records`` on ``sample``, keep and
        journal it. ``rejected`` (`_rejected_side`): ``current`` is a rejected child of
        the unchanged incumbent ``prev``, compared on the items it changed."""
        from concurrent.futures import CancelledError

        from refactory_dsh.budgets import BudgetExhausted
        from refactory_dsh.model import ModelChoiceError
        from refactory_dsh.optim import analysts, memory

        # Each side's records with how its own graph's model steps run (`_with_step_sources`).
        records = [dict(zip(held, self._with_step_sources(held.values(), side)))
                   for held, side in zip(records, (prev, current))]
        entries, missing = memory.pairs(records[0], records[1], sample)

        view = {c.id: analysts.pipeline_text(memory.state_view(c.state),
                                            memory.PIPELINE_TEXT_CHARS)
                for c in {prev.id: prev, current.id: current}.values()}
        previous = str(self._memory.get("text") or "")
        record: dict[str, Any] = {
            "epoch": epoch, "generation": self.generation, "prev": prev.id, "curr": current.id,
            "same": same, "sample": sample, "pairs": len(entries), "missing": missing,
            "counts": memory.counts(entries), "evaluated": evaluated, "rollouts": rolled,
            **({"rejected": dict(rejected)} if rejected else {})}

        gated = self.selection_mode() or False
        # The tally of each change tried against the noise (`memory.change_tally_lines`),
        # for this call; the proposer reads every change in its own ledger.
        tally = memory.change_tally_lines(tried, val_items=len(self._manifest_val_ids()),
                                          spread=self._rescore_spread(prev.id), gated=gated)
        lineage = self._lineage_edits(prev, current)
        # The candidates whose edits the changes section shows whole (`memory.changes_lines`).
        section = (memory.changes_heading(rejected.get("noun") if rejected else None)
                   if not same else None)
        shown_in = ({str(step.get("id")): section
                     for step in [*(lineage.get("prev") or ()), *(lineage.get("curr") or ())]}
                    if section else {})
        tried_lines = memory.tried_lines(tried, refused, round_=epoch - 1,
                                         current=prev.id if rejected else current.id,
                                         gated=gated, tally=tally, shown_in=shown_in)
        objective = self._objective()

        # The request: one call when it fits the room whole, else split into comparison
        # calls and a synthesis, never cut (`memory.MemoryRequest`).
        build = memory.MemoryRequest(
            epoch=epoch, mode_name=self.mode_id,
            prev={"id": prev.id, "pipeline": view[prev.id], "rollout": sides[0][3]},
            curr={"id": current.id, "pipeline": view[current.id], "rollout": sides[1][3],
                  **({"rejected": dict(rejected)} if rejected else {})},
            changes=memory.changes_lines(lineage, homes={
                "prev": (memory.pipeline_homes(prev.state),
                         "the pipeline at the start of this epoch" if not rejected
                         else "the incumbent pipeline"),
                "curr": (memory.pipeline_homes(current.state),
                         f"the {memory.noun({'rejected': rejected})}'s pipeline" if rejected
                         else "the pipeline now")}),
            previous_note=previous, entries=entries, tried=tried_lines, objective=objective)

        outcome: dict[str, Any] = {}
        if not entries:
            outcome = {"note": None, "error": "no sampled train item had a record under both "
                                              "pipelines, so there was nothing to compare"}
        else:
            try:
                outcome = self.optimizer.write_memory(epoch=epoch, generation=self.generation,
                                                      build=build)
            except (BudgetExhausted, CancelledError, ModelChoiceError):
                # A spent budget, a cancelled run or a missing model choice is no answer
                # about the memory: no pass is recorded, so a resume makes it (its
                # rollouts are journaled), and the error goes where a proposal's would.
                raise
            except Exception as exc:  # a failed pass keeps the earlier note
                outcome = {"note": None, "error": f"{type(exc).__name__}: {exc}"}
        record.update({key: outcome[key] for key in
                       ("calls", "context_retry", "cached", "clipped_items", "clipped_chars",
                        "prompt_chars", "parts", "compare_chars", "synthesis_cuts") if key in outcome})
        note = outcome.get("note")
        if note:
            record["reasoning"] = note.get("reasoning", "")
        self._keep_memory({**record, **({"error": outcome.get("error")}
                                         if not note else {})}, note)

    def _objective(self) -> str:
        """What the score measures and the compute policy
        (`optim.base.objective_with_policy`), from this run's settings, as every
        optimizer request says it."""
        from refactory_dsh.optim.base import objective_with_policy

        return objective_with_policy(float(self.config.parsimony),
                                     item_mean=bool(self._score_is_item_mean),
                                     gated=self.selection_mode() or False)

    def _tried_in_round(self, round_: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Every change the optimizer proposed in epoch ``round_``'s proposal round, for
        the memory pass (`optim.memory.tried_lines`): the candidates it made, oldest
        first, each with what it was (its own edits with the text they gave a new step or
        judge, and a knob's changed lines against its parent, `optim.memory.edits_text`;
        where each step it added sits, `optim.feedback.placement_lines`; its kind as a
        label), its train score and fixed/broken counts against its parent
        (`optim.feedback.ledger_entry`), the gate's word
        and the integrity verdict; and the proposals refused before they ran, with why.
        The facts the proposal request's history, change ledger, integrity and refusal
        sections show: train records and the gate's word, never a validation score."""
        from refactory_dsh.optim import memory
        from refactory_dsh.optim.base import Proposal, _structural_edit
        from refactory_dsh.optim.feedback import (change_kind, knob_values, ledger_entry,
                                                  placement_lines)

        by_id = {c.id: c for c in self.population}
        tried = []
        for candidate in self.population:
            record = self.proposals.get(candidate.id) or {}
            if not candidate.parent or record.get("epoch") != round_:
                continue
            parent = by_id.get(candidate.parent)
            if parent is not None and candidate.evaluated:
                parent_records, child_records, same = self._round_records(parent.id,
                                                                          candidate.id)
            else:
                parent_records, child_records, same = [], [], True
            entry = ledger_entry(candidate, parent, record=record,
                                 parent_records=parent_records, child_records=child_records,
                                 gate=self._gate_decision(candidate.id), same_seed=same,
                                 integrity=self._integrity_of(candidate.id))
            from refactory_dsh.optim.edit_effect import edit_effect

            effect = (edit_effect(parent_records, child_records, self._reply_judge(),
                                  new_steps=sys.modules["refactory_dsh.loop"]._new_steps(by_id.get(candidate.parent or ""),
                                                       candidate))
                      if candidate.is_structural() else {})
            before = knob_values(parent.state if parent is not None else None)
            tried.append({**entry, "structural": candidate.is_structural(),
                          "edits": memory.edits_text(candidate.own_edits(), before),
                          "placed": placement_lines(parent.state, candidate.state)
                          if parent is not None and candidate.is_structural() else [],
                          "score": candidate.score, "exit_ok": candidate.exit_ok,
                          "evaluated": candidate.evaluated,
                          **({"effect": effect} if effect else {})})
        refused = []
        for rejection in self._round_refusals.get(round_) or ():
            steps = Proposal(edit=rejection.edit).steps
            readable = [s for s in steps if isinstance(s, dict) and s.get("op")]
            refused.append({
                "round": round_, "attempt": rejection.attempt,
                "kind": change_kind(readable) if readable else "",
                "structural": any(_structural_edit(s) for s in readable) if readable else None,
                "edits": memory.edits_text(steps), "reasons": list(rejection.reasons)})
        return tried, refused

    def _keep_memory(self, record: dict[str, Any], note: dict[str, Any] | None) -> None:
        """Keep a pass's note (or, when it wrote none, the earlier note marked not
        updated in this epoch and why) and journal both (``optimizer_memory``)."""
        if note:
            self._memory = {"text": note["text"], "epoch": record["epoch"],
                            "generation": record.get("generation"), "prev": record.get("prev"),
                            "curr": record.get("curr"),
                            **({"rejected": dict(record["rejected"])}
                               if record.get("rejected") else {})}
            record = {**record, "updated": True, "chars": len(note["text"])}
        else:
            self._memory = {**self._memory, "not_updated": {
                "epoch": record["epoch"], "why": str(record.get("error") or "no note")}}
            record = {**record, "updated": False}
        self._memory_passes.append(record)
        self._journal("optimizer_memory", memory=dict(self._memory), **record)
        self._progress("optimizer_memory", epoch=record["epoch"], updated=record["updated"],
                       pairs=record.get("pairs"), evaluated=record.get("evaluated"),
                       **({"error": str(record.get("error"))[:300]} if record.get("error")
                          else {}))

    def _item_meta_reader(self) -> Any:
        """How a record's declared metadata is read (`task.item_metadata_reader`), once;
        ``None`` when no Task declares any."""
        if self._item_metadata is None:
            from refactory_dsh.task import item_metadata_reader

            try:
                self._item_metadata = (item_metadata_reader(self.mode.state_dict()),)
            except Exception:  # noqa: BLE001 - a mode that cannot say declares nothing
                self._item_metadata = (None,)
        return self._item_metadata[0]

    def _item_meta(self, record: dict[str, Any]) -> dict[str, Any]:
        read = self._item_meta_reader()
        return read(record) if read is not None else {}

    def _with_item_meta(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Train records with their declared metadata (``item_meta``) for the item headers
        the analysts and the proposal's evidence show; unchanged when none is declared."""
        if self._item_meta_reader() is None and self._item_context_reader() is None:
            return records
        return [self._declared(record) for record in records]

    def _with_step_sources(self, records: Iterable[Mapping[str, Any]],
                           candidate: Candidate | str | None) -> list[dict[str, Any]]:
        """``records`` (``candidate``'s train records) each with how its graph's model steps
        run (`optim.evidence.with_step_sources`: a model's reply, the tools the step was
        given, its real tool calls where the harness records them), which the evidence the
        analysts, the proposer and the memory pass read says under each step. From the
        candidate's own graph; unchanged when the candidate is not known."""
        from refactory_dsh.optim.evidence import with_step_sources

        if isinstance(candidate, str):
            candidate = next((c for c in self.population if c.id == candidate), None)
        nodes = ((candidate.state.get("graph") or {}).get("nodes")
                 if candidate is not None else None)
        return with_step_sources(records, nodes)

    def _with_item_history(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Each failing record, and each passing one whose item failed at least once in
        this run (a flaky pass), with its item's history (``item_history``,
        `optim.evidence.item_history`), for its heading and the table of every item: how
        often the item was solved in the run's train runs and on how many graphs.
        Information only: it selects nothing.

        Read from every train run's records (`_note_recorded`, the ``train_records``
        events, so a resume rebuilds it exactly), each repeat once; smoke and memory runs,
        val, harness incidents and harness timeouts are left out. A pass that never failed,
        or an item with no run but this one, gets none."""
        from refactory_dsh.optim.evidence import item_history

        runs: dict[str, list[tuple[str, bool]]] = {}
        for cid, outcomes in self.__dict__.get("_train_runs") or ():
            for item, passed in outcomes:
                runs.setdefault(item, []).append((cid, passed))
        out = []
        for record in records:
            seen = runs.get(str(record.get("id")), [])
            solid = record.get("passed") and all(passed for _cid, passed in seen)
            said = "" if solid else item_history(seen)
            out.append({**record, "item_history": said} if said else record)
        return out

    def _edit_count(self, candidate: Candidate) -> dict[str, int] | None:
        """``candidate``'s own edits counted by logical change (`optim.edit_count`) against
        its parent's state; ``None`` without a parent in the population."""
        from refactory_dsh.optim.edit_count import count, summary

        parent = next((c for c in self.population if c.id == candidate.parent), None)
        if parent is None:
            return None
        try:
            return summary(count(candidate.own_edits(), parent.state))
        except Exception:  # noqa: BLE001 - a count is a report, never a failure
            return None

    def _analyst_settings(self) -> dict[str, int]:
        """``ProposalContext.analysts``: the analyst stage's settings, empty when off."""
        if not self.config.analysts:
            return {}
        from refactory_dsh.optim.analysts import settings

        return settings(group_size=self.config.analyst_group_size,
                        merge_batch_size=self.config.analyst_merge_batch_size,
                        max_batches=self.config.analyst_max_batches)

    def _keep_analysis(self) -> None:
        """Journal the step's analysis (``analyst_findings``) once per analysis key
        (`optim.analysts.round_key`): a retry in the same step reuses the analysis and
        is not journaled again, nor is one a resumed run reloaded from a complete
        earlier analysis. An analysis a spent budget stopped is journaled with what it
        paid and its ``error``; one remade after an incomplete one (a failed batch or
        merge, or a stop) is journaled too, so the journal holds what the proposer saw
        and every call paid."""
        from refactory_dsh.optim.analysts import journal_entry

        analysis = getattr(self.optimizer, "last_analysis", None)
        if not self.config.analysts or not analysis:
            return
        entry = {"epoch": self.epoch, "generation": self.generation, **journal_entry(analysis)}

        def complete(earlier: dict[str, Any]) -> bool:
            return (not earlier.get("error")
                    and not any(b.get("error") for b in earlier.get("batches") or ())
                    and not any(m.get("error") for m in earlier.get("merges") or ()))

        if entry.get("key") and any(a.get("key") == entry["key"] and (complete(a) or a == entry)
                                    for a in self._analyses):
            return
        self._analyses.append(entry)
        self._journal("analyst_findings", **entry)

    def _ledger(self) -> list[dict[str, Any]]:
        """Every evaluated candidate of the run as "Every change so far" lists
        it (`optim.feedback.ledger_entry`): train records against its parent's from the
        same epoch, and the gate's word. Never a val score."""
        from refactory_dsh.optim.feedback import ledger_entry

        by_id = {c.id: c for c in self.population}
        entries = []
        for candidate in self._judged():
            parent = by_id.get(candidate.parent or "")
            parent_records, child_records, same = (
                self._round_records(parent.id, candidate.id) if parent is not None
                else ([], list(self._train_item_records.get(candidate.id) or ()), True))
            entries.append(ledger_entry(
                candidate, parent, record=self.proposals.get(candidate.id),
                parent_records=parent_records, child_records=child_records,
                gate=self._gate_decision(candidate.id), same_seed=same,
                integrity=self._integrity_of(candidate.id)))
        return entries

    def _added_steps(self) -> list[dict[str, Any]]:
        """The steps the last round's candidates added, each with its train evidence
        against its parent (`optim.feedback.step_evidence`), for the proposer to review.
        The last round is the latest epoch any judged candidate the optimizer proposed
        was proposed in. Train records only."""
        from refactory_dsh.optim.feedback import added_steps, change_kind, step_evidence

        by_id = {c.id: c for c in self.population}
        proposed = [c for c in self._judged()
                    if (self.proposals.get(c.id) or {}).get("origin") == "optimizer"
                    and (self.proposals.get(c.id) or {}).get("epoch") is not None]
        if not proposed:
            return []
        last = max(int(self.proposals[c.id]["epoch"]) for c in proposed)
        entries = []
        for candidate in proposed:
            parent = by_id.get(candidate.parent or "")
            if parent is None or int(self.proposals[candidate.id]["epoch"]) != last:
                continue
            steps = added_steps(parent.state, candidate.state)
            if not steps:
                continue
            parent_records, child_records, same = self._round_records(parent.id, candidate.id)
            kind = change_kind(candidate.own_edits(), candidate.state)
            for step, node_kind in steps:
                entries.append({
                    "candidate": candidate.id, "parent": parent.id, "kind": kind,
                    "node_kind": node_kind, "gate": self._gate_decision(candidate.id),
                    "same_seed": same,
                    # Counts against a parent scored with another seed are not shown.
                    **step_evidence(step, parent_records=parent_records if same else [],
                                    child_records=child_records, graph=candidate.state)})
        return entries

    def _edited_candidate(self, board: list[Candidate]) -> Candidate | None:
        """The candidate this step edits: the strategy's, else the one ``optimizer_step``
        selected, else the leader; ``None`` when none is known."""
        by_id = {c.id: c for c in self.population}
        parent_id = self.strategy_decision.get("parent") or self._editing_parent
        parent = by_id.get(str(parent_id)) if parent_id else None
        return parent if parent is not None or not board else board[0]

    def _graph_facts(self, board: list[Candidate], described: dict[str, Any]) -> list[str]:
        """`optim.base.graph_facts` for the request: each edit tried by `Run.mechanics` on
        the edited candidate (the seed when none is known), recording nothing and without
        the code dry run (a sandbox process per edit). A fact not worked out is said in
        its row and journaled (``graph_facts_not_worked_out``)."""
        from refactory_dsh.optim.base import graph_facts

        parent = self._edited_candidate(board) or Candidate.from_state(self.mode.state_dict())
        run = Run(_loop=self)
        catalogue = [op for op in nn_edits.catalogue(self.config.edit_preset)
                     if not self.config.allowed_ops or op["op"] in self.config.allowed_ops]
        knobs = [{"name": name, **spec} for name, spec in described["knobs"].items()]
        failures: list[dict[str, str]] = []
        facts = graph_facts(described.get("node_states", []), described["edges"], knobs,
                            catalogue, lambda edit: run.mechanics(
                                Proposal(edit=edit), parent, record=False, dry_run_code=False),
                            failures)
        if failures:
            # Said in the request's rows too; journaled so the run's record keeps it.
            self._journal("graph_facts_not_worked_out", candidate=parent.id,
                          failures=failures)
        return facts

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
        parent = self._edited_candidate(board)

        def described(mode: Any) -> dict[str, Any]:
            # The nodes in full too, in node-state form, for a structure edit to copy.
            # And its knob states (spec + value): a node an earlier edit added brings
            # its own knobs, which the seed does not declare.
            whole = mode.state_dict()
            nodes = (whole.get("graph") or {}).get("nodes") or {}
            return {**mode.describe(), "node_states": [dict(state) for state in nodes.values()],
                    "knob_states": dict(whole.get("knobs") or {}),
                    "available": available_before(
                        whole.get("graph") or {},
                        {name: (entry or {}).get("value")
                         for name, entry in (whole.get("knobs") or {}).items()}),
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
        """Train records for the parent the strategy selected for editing (see
        :meth:`_train_records_owner`)."""
        return self._train_records_owner(board)[1]

    def _train_records_owner(self, board: list[Candidate]) -> tuple[str | None, list[dict[str, Any]]]:
        """The candidate whose train records stand for the parent being edited, and its records.

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
            return str(parent_id), list(self._train_item_records[parent_id])
        # Fall back to the most recently installed candidate's parent:
        # what the optimizer actually bred from.
        for c in reversed(self.population):
            if c.parent and c.parent in self._train_item_records:
                return c.parent, list(self._train_item_records[c.parent])
        # Last resort: leaderboard best.
        if not board:
            return None, []
        return board[0].id, list(self._train_item_records.get(board[0].id, []))

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
