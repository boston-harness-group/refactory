"""SkillOpt's proposer as a reusable part: typed edits to a graph's open text knobs.

Everything SkillOpt does to turn rollouts into a change of the text it optimizes,
taken out of the strategy so another strategy (a population search) can call it:

- the train minibatch seed (``batch_seed``) and the slow update's sample;
- analysts in groups of ``group_size``, failures and successes apart
  (``analysts.reflect``);
- the failure-first hierarchical merge and the ranked edit budget
  (``merge.hierarchical_merge``, ``merge.rank``, ``schedule.edit_budget``);
- the patch step, the slow update and the meta skill.

Its output is typed edits: a ``set_knob`` with a ``patch`` or a ``value``, handed to
``Run.attempt`` like any proposal. It never runs or keeps anything; the strategy
decides what is rolled out and kept. The prompts and decisions are SkillOpt's,
unchanged.

**One knob or several.** ``knobs`` names the open text knobs it optimizes
(``("skill",)`` by default). With one, SkillOpt sees that knob's text, exactly as
before. With several, SkillOpt sees ONE document, the *composite*
(:meth:`SkillOptProposer.text`): a short header naming the graph's agent steps,
then each knob's text, in ``knobs`` order, under a ``<!-- SECTION <knob>: <use> -->``
line (``<use>`` is the knob's declared description). SkillOpt's procedure runs on
that document unchanged -- one set of analysts, one merge, one ranking, one slow
update and one meta skill per epoch, so call keys, trace entries and the
proposer's state stay single and unique -- and the patch is mapped back to the
knobs (:meth:`SkillOptProposer.route`):

- an edit may name its section: ``"section": "<knob>"``, the knob of a SECTION
  line (the composite's header asks for it when a target occurs in several
  sections). A named edit goes to that knob alone: an edit whose ``target`` is in
  that knob's text applies there; one whose ``target`` is elsewhere in the
  composite (another section, the structure, across a boundary) is **refused**
  ``skipped_target_not_in_section``; an ``append``, or a target found nowhere,
  goes to the named knob, where SkillOpt's missing-target rule applies. A section
  that names no knob is refused ``skipped_unknown_section``;
- an unnamed edit whose ``target`` occurs in exactly one knob's text goes to that
  knob; one whose ``target`` occurs in several is **refused**
  ``skipped_ambiguous_section``, its outcome naming the candidate sections (never
  a silent first match);
- an ``insert_after`` whose ``target`` is only in the header or a SECTION line takes
  SkillOpt's missing-target rule: it is appended to the last knob;
- any other edit whose ``target`` occurs in the composite only across a section
  boundary, in the header or in a SECTION line, or whose ``content`` holds a SECTION line, is
  **refused** (never split): it is reported ``skipped_cross_section`` /
  ``skipped_section_line_in_content`` and the rest of the patch still applies, as
  SkillOpt skips an edit aimed at its protected region;
- every other edit (an ``append``, a missing or unfound target) goes to the LAST
  knob, which ends the composite, so it lands where SkillOpt would put it: at the
  end of the document, before the protected region.

Each knob that receives edits gets one ``set_knob`` patch with them, in order, all
applied through ``Run.attempt`` (:meth:`SkillOptProposer.attempt_patch`). The
SLOW_UPDATE placeholder and guidance live in the last knob too (the end of the
composite, where SkillOpt keeps them), written as a ``set_knob`` ``value`` of that
knob alone. Knobs must be open ``str`` knobs; a frozen or non-text knob is
refused before the run spends anything (:meth:`SkillOptProposer.check_knobs`).

The proposer's own state -- the meta skill, the edit-budget step and the last slow
update's comparison pairs -- is ``state_dict``/``load_state_dict``, journaled with
the strategy's.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable

from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn.edits import SKILLOPT_PATCH_CONFIG
from refactory_dsh.optim import Candidate, Proposal, candidate_id

from refactory_packages.skillopt import analysts, merge, prompts, schedule
from refactory_packages.skillopt.env import SkillOptEnv
from refactory_packages.skillopt.json_utils import extract_json
from refactory_packages.skillopt.trace import TraceRecorder, ask

#: The line that opens each knob's section in the composite document.
SECTION_LINE = "<!-- SECTION {knob}: {use} -->"
#: What every SECTION line starts with; an edit's content may not carry one.
SECTION_PREFIX = "<!-- SECTION "
#: Status of an edit whose target lies across a section boundary or in the structure.
CROSS_SECTION = "skipped_cross_section"
#: Status of an edit whose content would add a SECTION line.
SECTION_LINE_IN_CONTENT = "skipped_section_line_in_content"
#: An edit's optional field naming the knob (section) it applies to.
SECTION_FIELD = "section"
#: Status of an edit whose section names no knob of the composite.
UNKNOWN_SECTION = "skipped_unknown_section"
#: Status of an edit whose target is not in the section it names.
TARGET_NOT_IN_SECTION = "skipped_target_not_in_section"
#: Status of an unnamed edit whose target occurs in several sections.
AMBIGUOUS_SECTION = "skipped_ambiguous_section"
#: The edit kinds that address text by a target (``nn.edits`` set_knob patches).
TARGET_KINDS = ("replace", "delete", "insert_after", "insert_before")


class SkillOptProposer:
    """SkillOpt's optimizer steps over a candidate's open text knob(s).

    ``client`` returns the model client each call goes through (called per step,
    so a rebound model is used). ``trace`` and ``env`` are set by the strategy
    once its run starts. ``rollout_order`` and ``patch_order`` are the offline
    replay's hooks (see ``SkillOptLoop.configure``).
    """

    def __init__(
        self,
        *,
        knob: str = "skill",
        knobs: tuple[str, ...] | list[str] | None = None,
        client: Callable[[], Any] = lambda: None,
        seed: int = 42,
        max_epochs: int = 4,
        group_size: int = 8,
        merge_batch_size: int = 8,
        analyst_edit_budget: int = 4,
        edit_budget_max: int = 4,
        edit_budget_min: int = 2,
        slow_update_samples: int = 20,
        rollout_order: Callable[[int, list[dict]], Any] | None = None,
        patch_order: Callable[[int, str, list[dict]], Any] | None = None,
    ) -> None:
        #: The knobs, in composite order; ``knob`` alone when ``knobs`` is not given.
        self.knobs: tuple[str, ...] = tuple(knobs) if knobs else (knob,)
        if not self.knobs or len(set(self.knobs)) != len(self.knobs) or not all(
            isinstance(name, str) and name for name in self.knobs
        ):
            raise ValueError(f"skillopt: knobs must name distinct text knobs, not {knobs!r}")
        #: The knob that ends the composite: appends, the SLOW_UPDATE placeholder
        #: and guidance go here (the only knob with one knob).
        self.knob: str = self.knobs[-1]
        self.client = client
        self.seed = seed
        self.max_epochs = max_epochs
        self.group_size = group_size
        self.merge_batch_size = merge_batch_size
        self.analyst_edit_budget = analyst_edit_budget
        self.edit_budget_max = edit_budget_max
        self.edit_budget_min = edit_budget_min
        self.slow_update_samples = slow_update_samples
        self.rollout_order = rollout_order
        self.patch_order = patch_order
        self.trace: TraceRecorder | None = None
        self.env: SkillOptEnv = SkillOptEnv()
        # State carried across epochs.
        self.meta_skill: str = ""
        self.budget_step: int = 0
        self.comparison_pairs: list[dict] = []

    # ── reading and writing the knobs ──────────────────────────────────────

    @property
    def multi(self) -> bool:
        """Whether the proposer optimizes several knobs through the composite."""
        return len(self.knobs) > 1

    def knob_text(self, candidate: Candidate, name: str) -> str:
        """``candidate``'s value of the text knob ``name``."""
        entry = (candidate.state.get("knobs") or {}).get(name) or {}
        return str(entry.get("value", "")) if isinstance(entry, dict) else str(entry)

    def text(self, candidate: Candidate) -> str:
        """The text SkillOpt optimizes: the knob's value, or with several knobs the
        composite document (:func:`composite`)."""
        if not self.multi:
            return self.knob_text(candidate, self.knob)
        return composite(candidate.state, self.knobs)

    def check_knobs(self, state: dict[str, Any]) -> None:
        """Refuse, before anything is spent, knobs the proposer cannot edit: a knob
        the mode lacks, or (with several knobs) one that is frozen or not ``str``."""
        declared = state.get("knobs") or {}
        missing = [name for name in self.knobs if name not in declared]
        if missing:
            if not self.multi:
                raise ValueError(
                    f"skillopt: the mode has no {self.knob!r} knob to optimize; "
                    "name its text knob with the knob option"
                )
            raise ValueError(
                f"skillopt: the mode has no {', '.join(map(repr, missing))} knob(s) to "
                "optimize; the knobs option names open text knobs of the mode"
            )
        if not self.multi:
            return
        for name in self.knobs:
            spec = (declared[name] or {}).get("spec") or {}
            if spec.get("frozen"):
                raise ValueError(f"skillopt: knob {name!r} is frozen; the knobs option "
                                 "names open text knobs only")
            if spec.get("kind") != "str":
                raise ValueError(f"skillopt: knob {name!r} has kind {spec.get('kind')!r}; "
                                 "the knobs option names text (str) knobs only")

    def set_text(self, text: str, parent: Candidate, rationale: str) -> Proposal:
        """A ``set_knob`` replacing the whole value of the last knob (``knob``)."""
        return Proposal(
            edit={"op": "set_knob", "name": self.knob, "value": text},
            rationale=rationale, parent=parent.id,
        )

    def patch(self, ranked: list[dict[str, Any]], parent: Candidate, rationale: str,
              *, knob: str | None = None) -> Proposal:
        """The patch step: the ranked edits as a ``set_knob`` patch (trainer.py:1507)
        of ``knob`` (the last knob when not given)."""
        items = [_op_to_kind(edit) for edit in ranked]
        if self.multi:
            # The section only routes the edit (:meth:`route`); the knob's patch
            # carries SkillOpt's own fields.
            items = [{k: v for k, v in item.items() if k != SECTION_FIELD}
                     if isinstance(item, dict) else item for item in items]
        return Proposal(
            edit={
                "op": "set_knob",
                "name": knob or self.knob,
                "patch": items,
                "config": dict(SKILLOPT_PATCH_CONFIG),
            },
            rationale=rationale, parent=parent.id,
        )

    def route(
        self, ranked: list[dict[str, Any]], parent: Candidate,
    ) -> tuple[list[tuple[str, list[int]]], list[dict[str, Any]]]:
        """Map the ranked edits of the composite to the knobs (module docstring).

        Returns ``(per_knob, refused)``: the knobs that receive edits, in ``knobs``
        order, each with the indices (into ``ranked``) of its edits in patch order,
        and one SkillOpt-style outcome per refused edit. Edits are routed one after
        another against the knob texts as the earlier edits left them, so applying
        each knob's edits in order reproduces the routing exactly. With one knob
        every edit is that knob's.
        """
        if not self.multi:
            return [(self.knob, list(range(len(ranked))))], []
        state = parent.state
        routed: dict[str, list[int]] = {name: [] for name in self.knobs}
        refused: list[dict[str, Any]] = []
        for index, edit in enumerate(ranked):
            item = _op_to_kind(edit) if isinstance(edit, dict) else edit
            texts = {name: _value(state, name) for name in self.knobs}
            name, status, sections = _route_one(item, texts, composite(state, self.knobs),
                                                self.knobs, _structure(state, self.knobs))
            if status:
                refused.append(_refusal(index, item, status, sections))
                continue
            routed[name].append(index)
            try:
                state = nn_edits.apply(self.patch([edit], parent, "", knob=name).edit, state)
            except nn_edits.EditError:
                # ``Run.attempt`` refuses that knob's patch the same way.
                pass
        return [(name, indices) for name, indices in routed.items() if indices], refused

    def attempt_patch(
        self, run: Any, ranked: list[dict[str, Any]], parent: Candidate, rationale: str,
    ) -> tuple[Candidate | list[str], dict[str, Any] | None]:
        """The patch step through ``run.attempt``: the candidate, or the refusal
        reasons, and SkillOpt's per-edit report (``edit_apply_report.json``).

        With one knob: one ``set_knob`` patch, as SkillOpt applies one patch. With
        several: one patch per knob that receives edits (:meth:`route`), attempted
        in ``knobs`` order, each on the candidate the last one returned. A knob
        whose edits change nothing is passed over without an attempt (so no
        refusal is recorded for it); any other refusal refuses the whole step.
        When no knob changed, the step is ``E-noop`` (one refusal recorded, as
        for one knob). The candidate returned is rebuilt as a child of ``parent``
        carrying the knobs' edits (the in-between candidates are never kept), so
        the run's lineage reaches it. The report lists every ranked edit in
        order, each with the ``knob`` it went to (``None`` when refused).

        Seam: a structural edit (typed blocks) would be attempted here, after the
        text patches and at most one per round (:meth:`structural_proposal`).
        """
        if not self.multi:
            proposal = self.patch(ranked, parent, rationale)
            outcome = run.attempt(proposal, parent)
            if isinstance(outcome, list):
                if not only_noop(outcome):
                    return outcome, None
                reports: list[dict[str, Any]] = []
                nn_edits.apply(proposal.edit, parent.state, edit_reports=reports)
                return outcome, reports[0] if reports else None
            return outcome, outcome.edit_reports[-1] if outcome.edit_reports else None

        per_knob, refused = self.route(ranked, parent)
        outcomes: list[dict[str, Any]] = list(refused)
        current: Candidate = parent
        noop_reasons: list[str] = []
        # Set while ``current`` is a held candidate a knob's patch duplicated.
        duplicate_reasons: list[str] | None = None
        proposals = [
            (name, self.patch([ranked[i] for i in indices], parent,
                              f"{rationale} ({name})", knob=name), indices)
            for name, indices in per_knob
        ]
        structural = self.structural_proposal(ranked, parent)
        if structural is not None:  # pragma: no cover - nothing returns one yet
            raise NotImplementedError("skillopt: structural edits are not built yet")
        applied: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
        for name, proposal, indices in proposals:
            knob_reports: list[dict[str, Any]] = []
            try:
                after = nn_edits.apply(proposal.edit, current.state, edit_reports=knob_reports)
            except nn_edits.EditError:
                after = None  # ``run.attempt`` below refuses it and says why
            rows = knob_reports[0]["outcomes"] if knob_reports else []
            for index, row in zip(indices, rows):
                outcomes.append({**row, "index": index, "knob": name})
            if after is not None and candidate_id(after) == current.id:
                # This knob's edits change nothing: pass it over without an
                # attempt, so a step that applies records no refusal for it.
                noop_reasons.append(
                    f"E-noop: the {name} patch leaves the state identical to its parent")
                continue
            outcome = run.attempt(proposal, current)
            if isinstance(outcome, list):
                # A knob's patch that reaches a state the search holds continues
                # from that candidate; if no later knob changes it, the step's
                # result is a duplicate, refused below as with one knob.
                duplicate = _held_duplicate(run, outcome, proposal, current)
                if duplicate is None:
                    return outcome, None
                duplicate_reasons, outcome = outcome, duplicate
            else:
                duplicate_reasons = None
            applied.append((proposal.edit, knob_reports[0] if knob_reports else None))
            current = outcome
        if duplicate_reasons is not None:
            # No later knob changed the held candidate: the step's result is a
            # state the search already has.
            return duplicate_reasons, None
        report = {
            "outcomes": sorted(outcomes, key=lambda o: o["index"]),
            "noop": current.id == parent.id,
            "knobs": [name for name, _ in per_knob],
        }
        if current.id == parent.id:
            if proposals:
                # One refusal for the step, as one knob's patch records one.
                first = run.attempt(proposals[0][1], parent)
                if isinstance(first, list):
                    noop_reasons = first
            return noop_reasons or [
                "E-noop: every edit of the patch was refused or skipped, so the "
                "state is identical to its parent"
            ], report
        # The step's candidate is a child of ``parent``: the knobs' in-between
        # candidates are never kept, so the lineage would otherwise stop at one.
        parent_reports = list(parent.edit_reports) if parent.edit_reports else [
            None for _ in parent.edits]
        return Candidate.from_state(
            current.state, parent=parent.id,
            edits=[*parent.edits, *(edit for edit, _ in applied)],
            edit_reports=[*parent_reports, *(rep for _, rep in applied)],
        ), report

    def structural_proposal(self, ranked: list[dict[str, Any]], parent: Candidate) -> Proposal | None:
        """The round's one structural edit (a typed block), or ``None``.

        A seam only: typed blocks are not built, so this proposes none. When they
        are, at most one structural edit per round is attempted after the text
        patches (:meth:`attempt_patch`).
        """
        return None

    def placeholder(self, parent: Candidate) -> Proposal:
        """Epoch 1's slow update: empty SLOW_UPDATE markers (trainer.py:1809-1824),
        at the end of the last knob."""
        return self.set_text(
            prompts.inject_empty_slow_update_field(self.knob_text(parent, self.knob)), parent,
            "slow update placeholder",
        )

    def slow_update_proposal(self, guidance: str, parent: Candidate) -> tuple[Proposal, str]:
        """The slow update's force-accepted guidance (slow_update.py:85-93) as a
        ``set_knob`` value of the last knob, and the text SkillOpt would hash
        (the whole optimized text after the update)."""
        new_value = prompts.replace_slow_update_field(self.knob_text(parent, self.knob), guidance)
        proposal = self.set_text(new_value, parent, "slow update guidance")
        if not self.multi:
            return proposal, new_value
        state = {**parent.state, "knobs": {
            **parent.state.get("knobs", {}),
            self.knob: {**parent.state["knobs"][self.knob], "value": new_value},
        }}
        return proposal, composite(state, self.knobs)

    # ── minibatches ────────────────────────────────────────────────────────

    def batch_seed(self, epoch: int, batch_idx: int = 0) -> int:
        """The epoch's train-batch seed; see :func:`batch_seed`."""
        return batch_seed(self.seed, epoch, batch_idx)

    def slow_update_sample(self, train_ids: list[str], epoch: int) -> tuple[int, list[str]]:
        """The slow update's seed and sample: the first ``slow_update_samples`` train
        ids shuffled by ``seed + epoch*2000`` (trainer.py:1843-1844;
        dataloader.py:278-283 rolls them out with the same seed)."""
        slow_seed = self.seed + epoch * 2000
        n_samples = min(self.slow_update_samples, len(train_ids))
        shuffled = list(train_ids)
        random.Random(slow_seed).shuffle(shuffled)
        return slow_seed, shuffled[:max(n_samples, 0)]

    # ── the step: analysts, merge, rank ────────────────────────────────────

    def reflect(
        self, epoch: int, records: list[dict], candidate: Candidate,
    ) -> tuple[list[dict], list[dict]]:
        """The analysts over ``candidate``'s train records, in the merge's order."""
        failure, success = analysts.reflect(
            client=self.client(),
            records=self._analyst_records(epoch, records),
            skill_text=self.text(candidate),
            edit_budget=self.analyst_edit_budget,
            meta_skill=self.meta_skill,
            # The analysts shuffle with the batch's own seed
            # (trainer.py:1242, :1283 pass batch.seed as random_seed).
            seed=self.batch_seed(epoch),
            group_size=self.group_size,
            epoch=epoch,
            trace=self.trace,
            env=self.env,
            knobs=knob_values(candidate),
        )
        return self._merge_order(epoch, "failure", failure), self._merge_order(epoch, "success", success)

    def merge(self, epoch: int, text: str, failure: list[dict], success: list[dict]) -> dict:
        """The failure-first hierarchical merge of the analysts' patches."""
        merged = merge.hierarchical_merge(
            client=self.client(),
            skill_text=text,
            failure_patches=failure,
            success_patches=success,
            meta_skill=self.meta_skill,
            epoch=epoch,
            trace=self.trace,
            merge_batch_size=self.merge_batch_size,
        )
        # SkillOpt writes the merged patch to steps/step_NNNN/merged_patch.json
        # (trainer.py:1376); recording the dict here lets the F20 replay compare it
        # structurally rather than by count. Observability only.
        self.trace.record_decision(
            "merged_patch", patch=merged, n_edits=len(merge.patch_edits(merged)),
        )
        return merged

    def rank(self, epoch: int, text: str, merged: dict) -> list[dict]:
        """The ranked edits within this step's edit budget (the cosine schedule)."""
        budget = schedule.edit_budget(
            self.budget_step + 1, self.max_epochs, self.edit_budget_max, self.edit_budget_min,
        )
        self.budget_step += 1
        ranked = merge.rank(
            client=self.client(),
            skill_text=text,
            merged_edits=merge.patch_edits(merged),
            budget=budget,
            meta_skill=self.meta_skill,
            epoch=epoch,
            trace=self.trace,
        )
        # ``edits`` mirrors SkillOpt's steps/step_NNNN/ranked_edits.json
        # (trainer.py:1396, :1427), so the F20 replay compares the selected edits
        # themselves. Observability only.
        self.trace.record_decision(
            "ranked_edits", budget_step=self.budget_step, edit_budget=budget,
            n_edits=len(ranked), edits=ranked,
        )
        return ranked

    # ── end of epoch: slow update and meta skill ───────────────────────────

    def comparison(
        self, prev: tuple[list[dict], dict], curr: tuple[list[dict], dict], sample_ids: list[str],
    ) -> list[dict]:
        """The slow update's comparison pairs from two rollouts of the sample, each
        given as ``(records, knob values)``; kept for the meta skill."""
        self.comparison_pairs = comparison_pairs(
            prev[0], curr[0], sample_ids, env=self.env, prev_knobs=prev[1], curr_knobs=curr[1],
        )
        return self.comparison_pairs

    def slow_update(self, epoch: int, prev_text: str, curr_text: str) -> tuple[str, dict | None]:
        """The slow update's guidance for ``curr_text`` (slow_update.py:434-439),
        empty when the model gave none, and the parsed response."""
        prev_guidance = prompts.extract_slow_update_field(curr_text)  # trainer.py:1916
        user = prompts.format_slow_update_user(
            prev_text, curr_text, prev_guidance, prompts.format_comparison_text(self.comparison_pairs),
        )
        output = ask(
            self.client(), self.trace, "slow_update", user,
            system=prompts.SLOW_UPDATE_SYSTEM, call_key=("slow_update", epoch, 0, 0),
        )
        result = extract_json(output) if isinstance(output, str) else None
        guidance = ""
        if isinstance(result, dict) and result.get("slow_update_content"):
            guidance = str(result["slow_update_content"]).strip()
        return guidance, result

    def meta_skill_step(self, epoch: int, prev_text: str, last_step_text: str) -> None:
        """Mechanism 9, from epoch 2 (trainer.py:2083-2198): compare the previous
        epoch's text with this epoch's last-step text and write the meta skill; a
        response without content leaves none (trainer.py:384-397, :2189-2194)."""
        user = prompts.format_meta_skill_user(
            prev_text, last_step_text, self.meta_skill,
            prompts.format_comparison_text(self.comparison_pairs),
        )
        output = ask(
            self.client(), self.trace, "meta_skill", user,
            system=prompts.META_SKILL_SYSTEM, call_key=("meta_skill", epoch, 0, 0),
        )
        result = extract_json(output) if isinstance(output, str) else None  # meta_skill.py:70-75
        content = ""
        if isinstance(result, dict) and result.get("meta_skill_content"):
            content = str(result["meta_skill_content"]).strip()
        self.meta_skill = content
        self.trace.record_decision(
            "meta_skill",
            action="write_meta_skill" if content else "no_content",
            meta_skill_length=len(content),
            # meta_skill_result.json's own fields (meta_skill.py:70-75), compared
            # by the replay; SkillOpt writes them only for a skill it got.
            **({
                "reasoning": str(result.get("reasoning", "")).strip(),
                "meta_skill_content": content,
            } if content else {}),
        )

    # ── state ──────────────────────────────────────────────────────────────

    def state_dict(self) -> dict[str, Any]:
        return {
            "meta_skill_text": self.meta_skill,
            "edit_budget_step": self.budget_step,
            "comparison_pairs": self.comparison_pairs,
            "proposer_knobs": list(self.knobs),
        }

    def load_state_dict(self, data: dict[str, Any]) -> None:
        saved = data.get("proposer_knobs")
        if saved is not None and tuple(saved) != self.knobs:
            raise ValueError(
                f"skillopt: this run optimized the knobs {list(saved)}; it cannot resume "
                f"over {list(self.knobs)} -- give the same knobs option"
            )
        self.meta_skill = data.get("meta_skill_text", "")
        self.budget_step = data.get("edit_budget_step", 0)
        self.comparison_pairs = data.get("comparison_pairs", [])

    # ── replay ordering hooks ──────────────────────────────────────────────

    def _analyst_records(self, epoch: int, records: list[dict]) -> list[dict]:
        """The epoch's records in the order the analysts split and shuffle them.

        The Run's order unless ``rollout_order`` is set; then its answer, which
        must be the same records in another order.
        """
        if self.rollout_order is None:
            return records
        ordered = list(self.rollout_order(epoch, list(records)))
        by_id = {str(r.get("id", "")): r for r in records}
        ids = [str(r.get("id", "")) for r in ordered]
        if len(by_id) != len(records) or sorted(ids) != sorted(by_id):
            raise ValueError(
                "skillopt: rollout_order must return a permutation of the epoch's "
                "records; it changed which records there are"
            )
        return [by_id[item_id] for item_id in ids]

    def _merge_order(self, epoch: int, kind: str, patches: list[dict]) -> list[dict]:
        """The epoch's ``kind`` patches in the order the merge receives them.

        The analysts' group order unless ``patch_order`` is set; then its answer,
        taken only as a permutation of these same patch objects.
        """
        if self.patch_order is None or len(patches) < 2:
            return patches
        ordered = list(self.patch_order(epoch, kind, list(patches)))
        remaining = list(range(len(patches)))
        picked: list[dict] = []
        for patch in ordered:
            index = next((i for i in remaining if patches[i] is patch), None)
            if index is None:
                raise ValueError(
                    "skillopt: patch_order must return a permutation of the epoch's "
                    "patches; it changed which patches there are"
                )
            remaining.remove(index)
            picked.append(patches[index])
        if remaining:
            raise ValueError(
                "skillopt: patch_order must return a permutation of the epoch's "
                "patches; it dropped some"
            )
        return picked


def batch_seed(seed: int, epoch: int, batch_idx: int = 0) -> int:
    """SkillOpt's train-batch seed for LiveMath: ``seed + epoch*1000 + batch_idx + 1``.

    LiveMath has a dataloader, so the trainer takes each batch's own seed from
    ``plan_train_epoch`` (trainer.py:1170-1177, :1242; dataloader.py:278) and
    hands it to the analysts as ``random_seed`` (trainer.py:1283). The
    ``Random(seed + epoch*1000)`` shuffle of base seeds (datasets/base.py:98-109)
    is the dataloader-less path and does not apply.
    """
    return seed + epoch * 1000 + batch_idx + 1


def knob_values(candidate: Candidate) -> dict[str, Any]:
    """The candidate's knob values by name, as its rollout used them."""
    knobs = candidate.state.get("knobs", {}) or {}
    return {
        name: entry.get("value") if isinstance(entry, dict) else entry
        for name, entry in knobs.items()
    }


def comparison_pairs(
    prev_records: list[dict],
    curr_records: list[dict],
    sample_ids: list[str],
    *,
    env: SkillOptEnv | None = None,
    prev_knobs: dict[str, Any] | None = None,
    curr_knobs: dict[str, Any] | None = None,
) -> list[dict]:
    """Per-sample comparison entries — slow_update.py:206-269 ``build_comparison_pairs``.

    SkillOpt builds them from the two rollouts' result dicts and reads each
    trajectory from the rollout's ``conversation.json``; here both come from
    the env's rendering of the F6 records (``SkillOptEnv.rollout_result``),
    each side with the knobs it was rolled out with.
    """
    env = env if env is not None else SkillOptEnv()
    prev_by_id = {
        str(r.get("id", "")): env.rollout_result(r, dict(prev_knobs or {}))
        for r in prev_records
    }
    curr_by_id = {
        str(r.get("id", "")): env.rollout_result(r, dict(curr_knobs or {}))
        for r in curr_records
    }
    pairs: list[dict] = []
    for item_id in sample_ids:
        prev = prev_by_id.get(item_id, {})
        curr = curr_by_id.get(item_id, {})
        prev_ok = _is_result_success(prev)
        curr_ok = _is_result_success(curr)
        if not prev_ok and curr_ok:
            category = "improved"
        elif prev_ok and not curr_ok:
            category = "regressed"
        elif not prev_ok and not curr_ok:
            category = "persistent_fail"
        else:
            category = "stable_success"
        # slow_update.py:242: SkillOpt reads the task from the sampled item;
        # an env's result carries the item's question (e.g.
        # livemathematicianbench/rollout.py:131).
        item = curr or prev
        pairs.append({
            "id": item_id,
            "task": item.get("question", item.get(
                "task_description", item.get("instruction", item_id))),
            "category": category,
            "prev": _outcome(prev, prev_ok),
            "curr": _outcome(curr, curr_ok),
            "prev_trajectory": _read_trajectory(prev),
            "curr_trajectory": _read_trajectory(curr),
        })
    return pairs


def hard_score(records: list[dict]) -> float:
    """SkillOpt's ``compute_score`` hard half over a rollout (utils/scoring.py:7-23)."""
    if not records:
        return 0.0
    return sum(1.0 if record.get("passed") else 0.0 for record in records) / len(records)


def _is_result_success(res: dict) -> bool:
    """slow_update.py:182-195: ``hard`` must be a finite number in [0, 1], >= 0.5."""
    if not res or not isinstance(res, dict):
        return False
    hard_val = res.get("hard")
    if isinstance(hard_val, (bool, int, float)):
        if math.isfinite(hard_val) and 0.0 <= hard_val <= 1.0:
            return float(hard_val) >= 0.5
    return False


def _get_soft_score(res: dict) -> float:
    """slow_update.py:198-211: ``soft`` when a finite number in [0, 1], else 0.0."""
    if not res or not isinstance(res, dict):
        return 0.0
    soft_val = res.get("soft")
    if isinstance(soft_val, (bool, int, float)):
        if math.isfinite(soft_val) and 0.0 <= soft_val <= 1.0:
            return float(soft_val)
    return 0.0


def _outcome(result: dict, passed: bool) -> dict[str, Any]:
    """One side of a comparison pair — slow_update.py:246-257."""
    return {
        "hard": int(passed),
        "soft": _get_soft_score(result),
        "predicted_answer": result.get("predicted_answer", result.get("answer", "N/A")),
        "fail_reason": result.get("fail_reason", ""),
    }


def _read_trajectory(result: dict) -> str:
    """slow_update.py:116-168 ``_read_trajectory``, over a rendered result.

    A result without a conversation is SkillOpt's missing ``conversation.json``.
    Unlike ``fmt_trajectory``, entries that are not dicts are skipped.
    """
    conversation = result.get("conversation") if result else None
    if conversation is None:
        return "(trajectory not available)"
    if not conversation:
        return "(empty trajectory)"
    return prompts.fmt_trajectory([e for e in conversation if isinstance(e, dict)])


def _op_to_kind(edit: dict[str, Any]) -> dict[str, Any]:
    """Convert SkillOpt's ``"op"`` key to a set_knob patch's ``"kind"`` key.

    SkillOpt prompts produce edits with ``"op": "append"``; a refactory
    ``set_knob`` patch reads ``"kind": "append"``.
    """
    out = dict(edit)
    if "op" in out and "kind" not in out:
        out["kind"] = out.pop("op")
    return out


# ── the composite document (several knobs) ────────────────────────────────


def _value(state: dict[str, Any], name: str) -> str:
    entry = (state.get("knobs") or {}).get(name) or {}
    return str(entry.get("value", "")) if isinstance(entry, dict) else str(entry)


def _nodes(state: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = (state.get("graph") or {}).get("nodes") or {}
    values = nodes.values() if isinstance(nodes, dict) else nodes
    return [n for n in values if isinstance(n, dict) and n.get("id")]


def agent_steps(state: dict[str, Any]) -> list[list[str]]:
    """The graph's agent nodes in run order, grouped: each group's nodes sit at the
    same depth (the longest path to them over edges, fork targets, join sources and
    data-pass entries), so a fork's parallel branches share a group."""
    nodes = {str(n["id"]): n for n in _nodes(state)}
    successors: dict[str, set[str]] = {name: set() for name in nodes}

    def link(source: Any, target: Any) -> None:
        if str(source) in nodes and str(target) in nodes:
            successors[str(source)].add(str(target))

    for edge in (state.get("graph") or {}).get("edges") or ():
        if isinstance(edge, dict):
            link(edge.get("source"), edge.get("target"))
    for name, node in nodes.items():
        for target in node.get("targets") or ():
            link(name, target)
        for source in node.get("sources") or ():
            link(source, name)
        if node.get("subgraph_entry"):
            link(name, node["subgraph_entry"])
    depth = {name: 0 for name in nodes}
    for _ in range(len(nodes)):  # longest path; a cycle stops at len(nodes) rounds
        changed = False
        for source, targets in successors.items():
            for target in targets:
                if depth[target] < depth[source] + 1:
                    depth[target] = depth[source] + 1
                    changed = True
        if not changed:
            break
    groups: dict[int, list[str]] = {}
    for name in sorted(nodes):
        if nodes[name].get("kind") == "agent":
            groups.setdefault(depth[name], []).append(name)
    return [groups[d] for d in sorted(groups)]


def section_use(state: dict[str, Any], name: str) -> str:
    """Where a knob's text is used: its declared description, else the nodes that
    declare it."""
    spec = ((state.get("knobs") or {}).get(name) or {}).get("spec") or {}
    description = str(spec.get("description") or "").strip()
    if description:
        return description
    users = [str(n["id"]) for n in _nodes(state) if name in (n.get("knobs") or ())]
    return f"used by {', '.join(users)}" if users else "not used by any node"


def composite_header(state: dict[str, Any]) -> str:
    """The composite's header: what the sections are and how edits must address them."""
    steps = " -> ".join(" | ".join(group) for group in agent_steps(state)) or "(none)"
    return (
        "# Skill sections\n"
        "This skill is split into sections, one per step of a graph. The graph's "
        f"agent steps, in order (| marks steps that run in parallel): {steps}. "
        "Each section starts with a "
        "<!-- SECTION <knob>: <where it is used> --> line and is the text of that knob; "
        "only the step it names reads it. Keep every edit inside one section, and never "
        "edit this header or a SECTION line; an edit with no target (append) goes at "
        "the end of the last section. An edit may name its section by adding "
        "\"section\": \"<knob>\" (the knob of that section's SECTION line); it then "
        "applies only there, and an append with a section goes at the end of that "
        "section. An edit whose target text appears in more than one section must "
        "name its section, or it is not applied."
    )


def composite(state: dict[str, Any], knobs: tuple[str, ...] | list[str]) -> str:
    """The one document SkillOpt sees for several text knobs (module docstring):
    the header, then each knob's text verbatim under its SECTION line, separated by
    a blank line. The last knob's text ends the document."""
    parts = [composite_header(state)]
    for name in knobs:
        line = SECTION_LINE.format(knob=name, use=section_use(state, name))
        parts.append(f"{line}\n{_value(state, name)}")
    return "\n\n".join(parts)


def _structure(state: dict[str, Any], knobs: tuple[str, ...]) -> list[str]:
    """The composite's own lines: its header and each SECTION line."""
    return [composite_header(state), *(
        SECTION_LINE.format(knob=name, use=section_use(state, name)) for name in knobs)]


def _route_one(
    edit: Any, texts: dict[str, str], document: str, knobs: tuple[str, ...],
    structure: list[str] = (),  # type: ignore[assignment]
) -> tuple[str, str, list[str]]:
    """The knob one edit goes to, or ``("", status, sections)`` when it is refused;
    ``sections`` are the knobs the refusal names (where the target is, or the
    knobs a section may name).

    A named section (``SECTION_FIELD``) is the only knob the edit may go to. An
    unnamed edit goes to the one knob holding its target, and is refused when
    several do. An unnamed ``insert_after`` whose target is only in the header or a
    SECTION line changes no structure: it takes SkillOpt's missing-target rule
    (append), so it goes to the last knob like an unfound target. Any other edit
    aimed outside the sections is refused."""
    if not isinstance(edit, dict):
        return knobs[-1], "", []
    content = edit.get("content", "")
    if isinstance(content, str) and SECTION_PREFIX in content:
        return "", SECTION_LINE_IN_CONTENT, []
    target = edit.get("target", "")
    targeted = edit.get("kind") in TARGET_KINDS and isinstance(target, str) and bool(target)
    holders = [name for name in knobs if targeted and target in texts[name]]
    section = edit.get(SECTION_FIELD)
    if section is not None and section != "":
        if not isinstance(section, str) or section not in knobs:
            return "", UNKNOWN_SECTION, list(knobs)
        if targeted and section not in holders and target in document:
            return "", TARGET_NOT_IN_SECTION, holders
        return section, "", []
    if len(holders) == 1:
        return holders[0], "", []
    if len(holders) > 1:
        return "", AMBIGUOUS_SECTION, holders
    if targeted:
        if edit.get("kind") == "insert_after" and any(target in part for part in structure):
            return knobs[-1], "", []
        if target in document:
            return "", CROSS_SECTION, []
    return knobs[-1], "", []


def _refusal(index: int, edit: Any, status: str, sections: list[str] = ()) -> dict[str, Any]:  # type: ignore[assignment]
    edit = edit if isinstance(edit, dict) else {}
    target = edit.get("target", "")
    out = {
        "index": index,
        "kind": edit.get("kind"),
        "target": str(target)[:200] if target else "",
        "content_preview": str(edit.get("content", ""))[:200],
        "status": status,
        "knob": None,
    }
    message = _REFUSAL_MESSAGES.get(status)
    if message:
        out["sections"] = list(sections)
        out["message"] = message.format(section=edit.get(SECTION_FIELD),
                                        sections=", ".join(sections) or "none")
    return out


#: Why a section-routing refusal happened, for the edit report.
_REFUSAL_MESSAGES = {
    AMBIGUOUS_SECTION: ("the target occurs in several sections ({sections}); name one "
                        "with \"section\" to apply the edit"),
    TARGET_NOT_IN_SECTION: ("the target is not in section {section!r}; it occurs in: "
                            "{sections}"),
    UNKNOWN_SECTION: "section {section!r} is not one of the sections: {sections}",
}


def only_noop(reasons: list[str]) -> bool:
    """Whether every refusal reason is ``E-noop`` (the edit changed nothing)."""
    return bool(reasons) and all(str(reason).startswith("E-noop") for reason in reasons)


def _held_duplicate(run: Any, reasons: list[str], proposal: Proposal,
                    parent: Candidate) -> Candidate | None:
    """The candidate the search already holds when one knob's patch, part of a
    multi-knob step, reaches an existing state (``E-duplicate``); else ``None``."""
    if not all(str(reason).startswith("E-duplicate") for reason in reasons):
        return None
    try:
        state = nn_edits.apply(proposal.edit, parent.state)
    except nn_edits.EditError:
        return None
    return run.candidate(candidate_id(state))
