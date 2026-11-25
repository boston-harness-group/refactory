"""GepaLoop: GEPA as an OuterLoop strategy on refactory's public extension points.

A port of ``gepa.optimize`` (``gepa`` 0.1.4, PyPI; gepa-ai/gepa, MIT) with its
default configuration, which is classic GEPA (Agrawal et al., 2025): one parent, one
reflective mutation of one component and one minibatch per iteration
(``core/engine.py:557-918``).

Per iteration:

1. (merge on) Right after an iteration that accepted a child, try a merge (``merge.py``).
2. Draw a parent from the per-instance Pareto frontier, weighted by the instances it
   is best on (core's ``refactory_dsh/optim/pareto.py``, the same frontier the
   ``pareto`` parent picker draws from; the pool is ``pareto.py``), and the
   iteration's minibatch of train ids
   (``sampler.py``).
3. Roll the parent out on the minibatch; skip the iteration if every item is perfect.
4. Pick the parent's next component round robin, build its reflective dataset from
   the records (``reflection.py``), ask the reflection model for a new instruction.
5. Roll the child out on the same minibatch; keep it only if its sum beats the
   parent's (strict), then score it on the whole Pareto set and add it to the pool.

The result is the program with the highest mean on the Pareto set. After the search
it is scored on test with the seed (``eval_test``, on by default; ``FinalEvaluation``,
as SkillOpt is), which GEPA itself leaves to its caller.

Mapping to refactory:

- A **component** is an open text knob. By default, the knobs that exactly one agent
  node renders and that are not its system prompt: per-node instructions, as DSPy's
  components are its predictors' instructions. The ``components`` option names them.
- GEPA's **trainset** is the parts manifest's train ids; its **valset** (the Pareto
  set) is the val part (``pareto_set="val"``, the default, as upstream GEPA and
  ``dspy.GEPA`` are run with a valset). Val is rolled out in full and read as per-item
  scores keyed by a per-run HMAC of the item id (``Run.part_item_scores``), in the
  manifest's order; the reflection model never sees a val item.
- **Information parity.** With ``pareto_set="val"`` GEPA *selects* on val item by item:
  every accepted program is scored on all of val, and its parent draws and its result
  come from those per-item scores. The default loop and SkillOpt see val only as one
  number per candidate, at their val gate. So GEPA reads more of val than the other
  arms do; a comparison with them should say so, or run ``pareto_set="train"``.
  ``pareto_set="train"`` holds ``pareto_size`` train ids out as the Pareto set instead
  (the leakage-free variant of ``docs/plans/gepa-mipro-outer-loop.md``); the seed and
  the Pareto-best are then scored on val, and the Pareto-best is the result only if it
  beats the seed there.
- A **metric call** is an item rollout, counted by the run itself
  (``Run.rollouts_spent``). GEPA's ``max_metric_calls`` is the ``max_metric_calls``
  option, else the run's ``max_rollouts``; it is checked at the top of each iteration as
  GEPA checks it (``engine.py:731``), so the last iteration may overshoot by a few
  rollouts as GEPA's does. With ``pareto_set="train"``
  the rollouts that scoring the result on val needs are held back, and a step that
  would eat into them ends the search instead. ``max_evaluations`` (64 by default)
  counts ``Run.evaluate`` calls, about two per iteration; the run warns at the start
  when it would bind first, and the report's ``stopped_by`` says which budget did.
- Refactory refuses every rollout, test included, once ``max_rollouts`` is spent, so
  a search that spends it leaves the test phase ``incomplete`` (SkillOpt's too). Set
  ``max_metric_calls`` below ``max_rollouts`` to keep headroom for test: twice the test
  part's size (the seed and the result; once when the seed is the result), plus a
  minibatch or a val rollout for the last iteration's overshoot.

Defaults (``api.py:46-103``):
  - seed = the run's ``--seed``    (GEPA's is 0, ``api.py:94``; the ``seed`` option sets it)
  - reflection_minibatch_size = 3  (``api.py:61``, ``:351``)
  - candidate_selection_strategy = "pareto"  (``api.py:57``; also "current_best")
  - skip_perfect_score = True, perfect_score = 1.0  (``api.py:59``, ``:62``)
  - module_selector = "round_robin"  (``api.py:66``; also "all")
  - use_merge = False, max_merge_invocations = 5, merge_val_overlap_floor = 5
                                   (``api.py:68-70``)
  - reflection_prompt_template = GEPA's default (``instruction_proposal.py:13-29``)

As GEPA does:

- A reflection call that fails (after the client's retries) drops the iteration's whole
  proposal, every component's, and the iteration moves on with no child rolled out
  (``reflective_mutation.py``, ``_propose_texts_batch_safe``: the failed task returns
  ``None``).
- A proposal whose texts are all unchanged, or that rebuilds a program the search
  already holds, is still rolled out on the minibatch (GEPA does not deduplicate).
- A minibatch that repeats an item (the sampler pads an epoch's last batch) rolls the
  repeat out again, and the sums and the reflective dataset count each occurrence.

Deviations, each forced by refactory or chosen for robustness:

- A merge is scored on 5 sampled Pareto instances (``merge.py:332-392``). Val can only
  be rolled out whole, so with ``pareto_set="val"`` the merged program is rolled out on
  all of val and judged on the same 5 instances by GEPA's rule; that rollout is then its
  full evaluation when it is accepted. A merge costs ``|val|``, not 5.
- ``max_failed_iterations`` (3): the run stops after that many iterations in a row whose
  every reflection call failed. GEPA has no such stop; without it a dead endpoint would
  spend a minibatch of parent rollouts per iteration until the budget ran out.
- Minibatch sums are taken over the items both parent and child were scored on: an item
  a harness incident left unscored is no one's win or loss.
- A candidate is its state, so a program the search already holds is not added to the
  pool a second time (GEPA adds a copy and scores it on the whole Pareto set); an
  accepted duplicate is recorded as ``duplicate`` and spends no Pareto rollouts.
- The seed's two start-up rollouts with ``pareto_set="train"`` (the Pareto ids and val)
  run concurrently.
"""

from __future__ import annotations

import random
import re
from typing import Any

from refactory_dsh.budgets import RESERVE_RETAINED, BudgetExhausted
from refactory_dsh.optim import Candidate, Proposal
from refactory_dsh.outerloop import (
    BudgetStopped, EvalRequest, FinalEvaluation, OuterLoop, OuterLoopError, Run, ValidationGate,
)

from refactory_packages.gepa import merge as merge_mod
from refactory_dsh.optim.pareto import find_dominator_programs, select_from_pareto_front
from refactory_packages.gepa.pareto import ProgramPool
from refactory_packages.gepa.reflection import (
    DEFAULT_PROMPT_TEMPLATE, check_template, extract_instruction, outcome_dataset,
    reflection_prompt, reflective_dataset,
)
from refactory_packages.gepa.sampler import EpochShuffledBatchSampler
from refactory_dsh.strategy_trace import TraceRecorder, ask

#: The version of the strategy state this code writes and reads.
STATE_VERSION = 2

#: Run.attempt's refusal of a state the search already holds names that candidate.
_DUPLICATE = re.compile(r"E-duplicate: .*?candidate ([0-9a-f]+)")


class _OutOfBudget(Exception):
    """The rollout or evaluation budget is spent: the search ends."""


class GepaLoop(OuterLoop):
    """GEPA outer-loop strategy (see the module docstring)."""

    name = "gepa"
    requires_model = True
    required_ops = ("set_knob",)
    text_only = ("GEPA rewrites the text of one component (an instruction knob) at a time "
                 "with set_knob, and merges components; it never changes the graph")
    accepted_options = (
        "candidate_selection_strategy", "components", "eval_test", "feedback", "max_failed_iterations",
        "max_merge_invocations", "max_metric_calls", "merge_val_overlap_floor", "module_selector",
        "pareto_set", "pareto_size", "perfect_score", "reflection_minibatch_size",
        "reflection_prompt_template", "seed", "skip_perfect_score", "use_merge",
    )

    # ── configure ─────────────────────────────────────────────────────────

    def configure(self, *, model: Any = None, model_name: str = "", provider: str = "auto",
                  **options: Any) -> None:
        self.model = model
        self.model_name = model_name
        self.provider = provider
        self.options = dict(options)

        seed = options.get("seed")
        #: The ``seed`` option; ``None`` takes the run's ``--seed`` when the search starts.
        self.seed_option = None if seed is None else int(seed)
        self.eval_test = _flag("eval_test", options.get("eval_test", True))
        calls = options.get("max_metric_calls")
        #: GEPA's own search budget in rollouts; ``None`` searches to ``max_rollouts``.
        self.max_metric_calls = None if calls is None else _positive("max_metric_calls", calls)
        self.minibatch_size = _positive("reflection_minibatch_size",
                                        options.get("reflection_minibatch_size", 3))
        self.selection = _choice("candidate_selection_strategy",
                                 options.get("candidate_selection_strategy", "pareto"),
                                 ("pareto", "current_best"))
        self.skip_perfect_score = _flag("skip_perfect_score", options.get("skip_perfect_score", True))
        self.perfect_score = float(options.get("perfect_score", 1.0))
        self.module_selector = _choice("module_selector", options.get("module_selector", "round_robin"),
                                       ("round_robin", "all"))
        self.use_merge = _flag("use_merge", options.get("use_merge", False))
        self.max_merge_invocations = int(options.get("max_merge_invocations", 5))
        self.merge_val_overlap_floor = _positive("merge_val_overlap_floor",
                                                 options.get("merge_val_overlap_floor", 5))
        template = options.get("reflection_prompt_template") or DEFAULT_PROMPT_TEMPLATE
        check_template(str(template))
        self.template = str(template)
        self.components_option = _names(options.get("components"))
        self.pareto_set = _choice("pareto_set", options.get("pareto_set", "val"), ("val", "train"))
        size = options.get("pareto_size")
        self.pareto_size = None if size is None else _positive("pareto_size", size)
        if self.pareto_size is not None and self.pareto_set != "train":
            raise TypeError("gepa: pareto_size is for pareto_set=train")
        self.feedback = _choice("feedback", options.get("feedback", "details"), ("details", "score"))
        self.max_failed_iterations = _positive("max_failed_iterations",
                                               options.get("max_failed_iterations", 3))

        self._fresh(self.seed_option or 0)

    def _fresh(self, seed: int) -> None:
        """The search state before the first iteration, every draw seeded by ``seed``."""
        self.base_seed = int(seed)
        # One shared generator for frontier sampling, minibatches and merge (``api.py:304``),
        # and the DSPy adapter's own for picking a call (``dspy_adapter.py:111``). The
        # core frontier functions are handed this generator, never their own, so GEPA's
        # draws stay GEPA's (``test_fixed_seed_gepa_trace_is_unchanged``).
        self.rng = random.Random(self.base_seed)
        self.adapter_rng = random.Random(0)
        self.sampler = EpochShuffledBatchSampler(self.minibatch_size, self.rng)
        self.pool = ProgramPool()
        self.gate = ValidationGate(seed=self.base_seed)
        self.components: list[str] = []
        self.pareto_ids: list[str] = []
        self.i = -1
        self.search_rollouts: int | None = None
        self.stopped_by = ""
        self.val_size = 0
        self.final = FinalEvaluation(label="gepa")
        self.merges_due = 0
        self.total_merges_tested = 0
        self.last_iter_found_new_program = False
        self.merges_performed: tuple[list, list] = ([], [])
        self.stats: dict[str, int] = {}
        self.failed_streak = 0
        self.best_id: str | None = None
        self.val_reserve = 0
        self.finished = False
        self._run_error = ""
        self._trace: TraceRecorder | None = None
        self._checkpoint: dict[str, Any] = {}

    # ── the search ────────────────────────────────────────────────────────

    def strategy(self, run: Run) -> None:
        if not self.pool.candidate_ids and not self.finished:
            # A fresh search: every draw from the seed option, else the run's --seed.
            self._fresh(self.seed_option if self.seed_option is not None else run.run_seed())
        self._trace = TraceRecorder(run.root / "gepa_trace.jsonl")
        self._trace.listener = run.record_proposal_round
        self.final.record = self._trace.record_decision
        seed = run.seed()
        component_nodes = discover_components(seed.state, self.components_option)
        names = [name for name, _ in component_nodes]
        if self.components and self.components != names:
            raise OuterLoopError(f"gepa: this run optimized components {self.components}, "
                                 f"not {names}; resume it with the same components option")
        self.components = names
        self.nodes = dict(component_nodes)
        self._check_edit_limit(run)
        train_ids = run.part_ids("train")
        if not train_ids:
            raise ValueError("gepa: the minibatches need the parts manifest's train ids "
                             "(run.part_ids('train') returned none)")
        if self.pareto_set == "val" and not run.has_part("val"):
            reason = ("an empty val part" if run.part_declared("val")
                      else "no 'val' part in the parts manifest")
            raise ValueError(f"gepa: pareto_set=val needs val items, but the run has {reason}; "
                             "add val items, or set pareto_set=train")
        self.feedback_ids = self._split(train_ids)
        try:
            if not self.finished:
                if not self.pool.candidate_ids:
                    self._warn_budgets(run)
                    self._start(run, seed)
                self._search(run)
            self._finish(run)
        except _OutOfBudget:
            self.stopped_by = self._which_budget(run)
            run.log(f"gepa: the search stopped on {self.stopped_by} after "
                    f"{run.rollouts_spent()} rollouts")
            self._finish(run)
        except (BudgetStopped, BudgetExhausted) as exc:
            # A time or token budget stopped an evaluation or a reflection call.
            run.stop(getattr(exc, "reason", "") or str(exc))
            self.stopped_by = self.stopped_by or "time or token budget"
            self._finish(run)
        self._final_evaluation(run)
        self._save(run)

    def _check_edit_limit(self, run: Run) -> None:
        """Refuse up front a configuration whose proposals ``attempt`` would refuse:
        ``module_selector=all`` and a merge each set every component in one proposal."""
        limit = run.edit_limit()
        if limit is None or len(self.components) <= limit:
            return
        for on, what in ((self.module_selector == "all", "module_selector=all"),
                         (self.use_merge, "use_merge")):
            if on:
                raise ValueError(
                    f"gepa: {what} may set all {len(self.components)} components in one "
                    f"proposal, and the run allows {limit} edits per proposal; name at most "
                    f"{limit} with the components option, or use module_selector=round_robin "
                    "without merge")

    def _warn_budgets(self, run: Run) -> None:
        """Say at the start when ``max_evaluations`` would end the search before
        ``max_rollouts``: an iteration makes about two evaluations of a minibatch each."""
        rollouts = run.rollouts_left()
        if self.max_metric_calls is not None:
            rollouts = min(rollouts if rollouts is not None else self.max_metric_calls,
                           self.max_metric_calls - run.rollouts_spent())
        evaluations = run.evaluations_left()
        if rollouts is not None and evaluations * self.minibatch_size < rollouts:
            run.log(f"gepa: warning: max_evaluations allows {evaluations} more evaluations, "
                    f"about {evaluations * self.minibatch_size} item rollouts at "
                    f"{self.minibatch_size} per evaluation, under the {rollouts} rollouts "
                    "GEPA's budget leaves; the evaluation budget will end the search first. Raise "
                    "--max-evaluations to run GEPA to its rollout budget")

    def _split(self, train_ids: list[str]) -> list[str]:
        """The train ids the minibatches come from; with ``pareto_set="train"`` the
        Pareto ids are held out of them (a seeded draw of ``pareto_size``, half by
        default)."""
        if self.pareto_set == "val":
            return list(train_ids)
        if not self.pareto_ids:
            size = self.pareto_size or max(1, len(train_ids) // 2)
            if size >= len(train_ids):
                raise ValueError(f"gepa: pareto_size={size} leaves no train ids for the "
                                 f"minibatches (the manifest has {len(train_ids)})")
            self.pareto_ids = sorted(random.Random(self.base_seed).sample(list(train_ids), size),
                                     key=list(train_ids).index)
        held = set(self.pareto_ids)
        return [i for i in train_ids if i not in held]

    def _start(self, run: Run, seed: Candidate) -> None:
        """Score the seed on the Pareto set (``engine.py:593-630``); it counts against the
        budget (``state.py:703-704``)."""
        if self.pareto_set == "train":
            # The seed on the Pareto ids and, beside it, on val: the bar the result must
            # beat, and how many rollouts to keep for scoring the result on val.
            self._spend(run, len(self.pareto_ids))
            run.evaluate_many([
                EvalRequest(candidate=seed, part="train", items=list(self.pareto_ids),
                            seed=self.base_seed),
                EvalRequest(candidate=seed, part="val", seed=self.base_seed),
            ])
            records = run.item_records(seed)
            scores = _ordered(_scores(records), self.pareto_ids)
            self.gate.cache[seed.id] = ValidationGate.val_score(run, seed)
            self.gate.start(run, seed)
            self.val_size = len(run.part_item_scores(seed, "val"))
            self.val_reserve = max(1, self.val_size)
        else:
            scores = self._pareto_scores(run, seed)
        if not scores:
            raise OuterLoopError("gepa: the seed has no per-item score on the Pareto set; "
                                 "its rollout failed, so there is no frontier to start from")
        idx = self.pool.add(seed.id, [None], scores, discovered_at=0)
        self.best_id = self.pool.candidate_ids[self.pool.best()]
        self._decide("seed", program=idx, candidate=seed.id, pareto_mean=self.pool.aggregate(idx),
                     instances=len(scores))
        self._commit(run)

    def _search(self, run: Run) -> None:
        """``engine.py:731-882``."""
        while True:
            self._commit(run)
            if run.check_control() == "stop":
                run.stop("control_stop")
                return
            if self._budget(run) <= 0:
                raise _OutOfBudget
            self.i += 1
            self._trace.set_epoch(self.i)
            calls_before = (self._trace.optimizer_calls, len(self._trace.optimizer_failures))
            if self.use_merge and self.merges_due > 0 and self.last_iter_found_new_program:
                self.last_iter_found_new_program = False
                if self._merge(run):
                    continue
            if self.use_merge:
                self.last_iter_found_new_program = False
            self._reflect(run)
            calls = self._trace.optimizer_calls - calls_before[0]
            failed = len(self._trace.optimizer_failures) - calls_before[1]
            if calls and failed >= calls:
                self.failed_streak += 1
                if self.failed_streak >= self.max_failed_iterations:
                    self._run_error = (f"every reflection call failed in {self.failed_streak} "
                                       f"iterations in a row: {self._trace.optimizer_failures[-1]}")
                    run.log(f"gepa: {self._run_error}")
                    run.stop(self._run_error)
                    return
            elif calls:
                self.failed_streak = 0

    # ── reflective mutation (reflective_mutation.py:242-641) ──────────────

    def _reflect(self, run: Run) -> None:
        parent_idx = self._select_parent()
        parent = _held(run, self.pool.candidate_ids[parent_idx])
        minibatch = self.sampler.next_minibatch(self.feedback_ids, self.i)
        slots = self._rollout(run, parent, minibatch)
        records = [r for r in slots if r is not None]
        before = _slot_scores(slots)
        self._count("iterations")
        if not records:
            self._decide("skip_no_trajectories", parent=parent_idx, minibatch=minibatch)
            return
        if self.skip_perfect_score and all(
                score is not None and score >= self.perfect_score for score in before):
            self._count("skipped_perfect")
            self._decide("skip_perfect", parent=parent_idx, minibatch=minibatch)
            return
        chosen = self._select_components(parent_idx)
        texts = _component_texts(parent, self.components)
        new_texts: dict[str, str] = {}
        for name in chosen:
            nodes = self.nodes.get(name) or []
            dataset = (reflective_dataset(records, nodes, texts[name], feedback=self.feedback,
                                          rng=self.adapter_rng)
                       if nodes else outcome_dataset(records, feedback=self.feedback))
            if not dataset:
                continue
            prompt = reflection_prompt(texts[name], dataset, self.template)
            raw = ask(self.model, self._trace, "reflect", prompt, system="",
                      call_key=("reflect", self.i, self.components.index(name), 0))
            if raw is None:
                # GEPA drops the whole proposal when a component's call fails
                # (``_propose_texts_batch_safe``), and no child is rolled out.
                self._count("reflection_failed")
                self._decide("reflection_failed", parent=parent_idx, components=chosen,
                             failed=name)
                return
            new_texts[name] = extract_instruction(raw.strip())
        if not new_texts:
            self._decide("no_proposal", parent=parent_idx, components=chosen)
            return
        changed = {n: t for n, t in new_texts.items() if t != texts[n]}
        if changed:
            made = self._attempt(run, parent, changed, f"gepa iteration {self.i}: reflective "
                                                       f"update of {', '.join(changed)}")
            if made is None:
                return
            child, existing = made
        else:
            # Every text came back unchanged: GEPA still rolls the (identical) child out.
            child, existing = parent, True
        after = _slot_scores(self._rollout(run, child, minibatch))
        shared = [k for k in range(len(minibatch)) if before[k] is not None and after[k] is not None]
        old_sum = sum(before[k] for k in shared)  # type: ignore[misc]
        new_sum = sum(after[k] for k in shared)  # type: ignore[misc]
        accepted = new_sum > old_sum
        self._decide("reflect", parent=parent_idx, components=sorted(changed or new_texts),
                     minibatch=minibatch, parent_sum=old_sum, child_sum=new_sum, accepted=accepted,
                     child=child.id, unchanged=not changed, existing=existing)
        if not accepted:
            self._count("rejected")
            return
        if not self._add(run, child, [parent_idx], "reflective"):
            return
        if self.use_merge:
            self.last_iter_found_new_program = True
            if self.total_merges_tested < self.max_merge_invocations:
                self.merges_due += 1

    def _select_parent(self) -> int:
        """``candidate_selector.py:18-33``."""
        aggs = self.pool.aggregates()
        if self.selection == "current_best":
            return aggs.index(max(aggs))
        return select_from_pareto_front(self.pool.front_programs, aggs, self.rng)

    def _select_components(self, parent_idx: int) -> list[str]:
        """``component_selector.py:10-36``: round robin advances the parent's pointer now,
        whatever comes of the proposal."""
        if self.module_selector == "all":
            return list(self.components)
        pid = self.pool.pointers[parent_idx]
        self.pool.pointers[parent_idx] = (pid + 1) % len(self.components)
        return [self.components[pid]]

    # ── merge (merge.py:290-403, engine.py:768-840) ───────────────────────

    def _merge(self, run: Run) -> bool:
        """One merge attempt; ``True`` when a merged program was scored (the iteration then
        makes no reflective mutation), accepted or not."""
        aggs = self.pool.aggregates()
        candidates = find_dominator_programs(self.pool.front_programs, aggs)
        texts = [_component_texts(_held(run, cid), self.components) for cid in self.pool.candidate_ids]

        def overlap(a: int, b: int) -> bool:
            common = set(self.pool.subscores[a]) & set(self.pool.subscores[b])
            return len(common) >= self.merge_val_overlap_floor

        found = merge_mod.sample_merge(aggs, self.rng, candidates, self.merges_performed, texts,
                                       self.pool.parents, has_overlap=overlap)
        if found is None:
            self._decide("merge_none")
            return False
        merged, id1, id2, ancestor = found
        self.merges_performed[0].append((id1, id2, ancestor))
        subsample = merge_mod.select_eval_subsample(self.pool.subscores[id1], self.pool.subscores[id2],
                                                    self.rng)
        if not subsample:
            return False
        ancestor_candidate = _held(run, self.pool.candidate_ids[ancestor])
        changed = {n: t for n, t in merged.items() if t != texts[ancestor][n]}
        made = self._attempt(run, ancestor_candidate, changed,
                             f"gepa iteration {self.i}: merge of programs {id1} and {id2} "
                             f"over their ancestor {ancestor}") if changed else None
        if made is None:
            return True
        child, _existing = made
        if self.pareto_set == "val":
            full = self._pareto_scores(run, child)
            new = [full.get(k, 0.0) for k in subsample]
        else:
            full = None
            # The subsample may repeat an id (``select_eval_subsample`` pads with
            # ``rng.choices``): each occurrence is its own rollout, as in GEPA.
            new = [0.0 if s is None else s
                   for s in _slot_scores(self._rollout(run, child, subsample))]
        sums = [sum(self.pool.subscores[p][k] for k in subsample) for p in (id1, id2)]
        accepted = sum(new) >= max(sums)
        self._decide("merge", ids=[id1, id2], ancestor=ancestor, subsample=subsample,
                     parent_sums=sums, child_sum=sum(new), accepted=accepted, child=child.id)
        if accepted:
            self._add(run, child, [id1, id2], "merge", scores=full)
            self.merges_due -= 1
            self.total_merges_tested += 1
        else:
            self._count("merges_rejected")
        return True

    # ── shared steps ──────────────────────────────────────────────────────

    def _attempt(self, run: Run, parent: Candidate, texts: dict[str, str],
                 rationale: str) -> tuple[Candidate, bool] | None:
        """The candidate the new texts make from ``parent``, and whether the search
        already held it (the texts rebuild ``parent`` or another held program, which
        GEPA would still roll out); ``None`` when refused."""
        edits = [{"op": "set_knob", "name": name, "value": text} for name, text in texts.items()]
        outcome = run.attempt(Proposal(edits=tuple(edits), rationale=rationale, parent=parent.id),
                              parent)
        if isinstance(outcome, list):
            if any(str(r).startswith("E-noop") for r in outcome):
                return parent, True
            for reason in outcome:
                found = _DUPLICATE.search(str(reason))
                held = run.candidate(found.group(1)) if found else None
                if held is not None:
                    return held, True
            self._count("refused")
            self._decide("refused", reasons=outcome[:5], components=sorted(texts))
            return None
        return outcome, False

    def _add(self, run: Run, child: Candidate, parents: list[int], how: str,
             scores: dict[str, float] | None = None) -> bool:
        """Score an accepted program on the whole Pareto set and add it
        (``engine.py:517-533``); ``False`` when the pool already holds it (a candidate is
        its state, so it is not added twice, and spends no Pareto rollouts)."""
        if self.pool.index(child.id) is not None:
            self._count("duplicate")
            self._decide("duplicate", child=child.id, how=how)
            return False
        discovered = run.rollouts_spent()
        if scores is None:
            scores = self._pareto_scores(run, child)
        run.keep(child)
        idx = self.pool.add(child.id, parents, scores, discovered_at=discovered)
        self.best_id = self.pool.candidate_ids[self.pool.best()]
        self._count("accepted_" + how)
        self._decide("added", program=idx, candidate=child.id, parents=parents, how=how,
                     pareto_mean=self.pool.aggregate(idx), best=self.best_id)
        return True

    def _pareto_scores(self, run: Run, candidate: Candidate) -> dict[str, float]:
        """The candidate's per-instance scores on the Pareto set, in its order: val,
        rolled out whole, or the held-out train ids. A val rollout that failed outright
        scores 0 on every instance, as GEPA scores a program that fails."""
        if self.pareto_set == "val":
            self._spend(run, max(1, self.val_size))
            self.gate.score(run, candidate)
            scores = run.part_item_scores(candidate, "val")
            self.val_size = self.val_size or len(scores)
            if not scores and self.pool.subscores:
                scores = {key: 0.0 for key in self.pool.subscores[0]}
            return scores
        slots = self._rollout(run, candidate, self.pareto_ids)
        return _ordered(_scores([r for r in slots if r is not None]), self.pareto_ids)

    def _rollout(self, run: Run, candidate: Candidate,
                 items: list[str]) -> list[dict[str, Any] | None]:
        """Roll ``candidate`` out on train ``items`` and return each position's record
        (``None`` when it has none). An item the list repeats is rolled out once per
        occurrence, as GEPA evaluates a minibatch, in one evaluation per repeat."""
        rounds: list[list[str]] = []
        seen: dict[str, int] = {}
        for item in items:
            nth = seen[item] = seen.get(item, -1) + 1
            if nth == len(rounds):
                rounds.append([])
            rounds[nth].append(item)
        found: list[dict[str, dict[str, Any]]] = []
        for batch in rounds:
            self._spend(run, len(batch))
            run.evaluate(candidate, part="train", items=batch, seed=self.base_seed)
            wanted = set(batch)
            found.append({str(r.get("id")): r for r in run.item_records(candidate)
                          if str(r.get("id")) in wanted})
        out: list[dict[str, Any] | None] = []
        seen = {}
        for item in items:
            nth = seen[item] = seen.get(item, -1) + 1
            out.append(found[nth].get(item))
        return out

    def _budget(self, run: Run) -> int:
        """What the search may still spend at the top of an iteration, in rollouts
        (GEPA's metric calls) less the val reserve; in evaluations when the run has no
        rollout budget; nothing once any budget (time and tokens too) is spent."""
        if run.budget_left() <= 0:
            return 0
        left = run.rollouts_left()
        budget = run.budget_left() if left is None else left - self.val_reserve
        if self.max_metric_calls is not None:
            budget = min(budget, self.max_metric_calls - run.rollouts_spent())
        return budget

    def _spend(self, run: Run, n: int) -> None:
        """Before rolling ``n`` items out: stop the search once any budget is spent, and,
        with a val reserve, when the ``n`` would eat into it. Without a reserve the last
        rollouts may overshoot ``max_rollouts``, as GEPA's last iteration does."""
        if run.budget_left() <= 0:
            raise _OutOfBudget
        left = run.rollouts_left()
        if self.val_reserve and left is not None and left - self.val_reserve < n:
            raise _OutOfBudget

    def _which_budget(self, run: Run) -> str:
        """The budget that ended the search, for the report's ``stopped_by``."""
        if self.max_metric_calls is not None and run.rollouts_spent() >= self.max_metric_calls:
            return "max_metric_calls"
        left = run.rollouts_left()
        if left is not None and left <= self.val_reserve:
            return "max_rollouts" + (" (val reserve kept)" if self.val_reserve else "")
        if run.evaluations_left() <= 0:
            return "max_evaluations"
        if run.budget_left() <= 0:
            return "time or token budget"
        return "max_rollouts (the next rollout would eat into the val reserve)"

    # ── the end ───────────────────────────────────────────────────────────

    def _finish(self, run: Run) -> None:
        """With ``pareto_set="train"``, the Pareto-best must beat the seed on val; the
        result is the program with the highest Pareto mean (``core/result.py:76-78``)."""
        if self.finished or not self.pool.candidate_ids:
            return
        self.search_rollouts = run.rollouts_spent()
        best = self.pool.candidate_ids[self.pool.best()]
        if self.pareto_set == "train" and best != self.pool.candidate_ids[0]:
            candidate = _held(run, best)
            try:
                with run.final_scoring():
                    verdict = self.gate.judge(run, candidate)
            except OuterLoopError as exc:
                verdict = None
                run.log(f"gepa: the Pareto-best {best} could not be scored on val: {exc}")
            self._decide("val_check", candidate=best, accepted=bool(verdict and verdict.accepted),
                         score=verdict.score if verdict else None, seed_score=self.gate.best_score)
            best = best if verdict and verdict.accepted else self.pool.candidate_ids[0]
        self.best_id = best
        self.finished = True
        if not run.stopped:
            run.stop("budget_exhausted")
        self._commit(run)

    def _final_evaluation(self, run: Run) -> None:
        """Score the result on test (``FinalEvaluation``, as SkillOpt does): the seed and
        GEPA's result, each with the run's seed. The result is the gate's current and
        best, so the val check reuses its score and the final phase the best's."""
        if not self.finished or self.final.result.get("status") in ("done", "skipped"):
            return
        save = lambda: self._commit(run)  # noqa: E731
        reason = self._final_eval_skip(run)
        if reason:
            self.final.skip(run, reason, save)
            return
        best = self.pool.index(self.best_id) if self.best_id else None
        score = self.gate.cache.get(self.best_id or "")
        if score is None:
            score = self.pool.aggregate(best) if best is not None else 0.0
        self.gate.adopt(self.best_id, score)  # type: ignore[arg-type]
        self.final.run(run, self.gate, seed=self.base_seed, save=save)

    def _final_eval_skip(self, run: Run) -> str:
        """Why the test evaluation does not run; empty when it does."""
        if not self.eval_test:
            return "the eval_test strategy option is false"
        if not run.has_part("test"):
            return run.empty_part_reason("test")
        if self._run_error:
            return f"the search did not finish: {self._run_error}"
        if not self.best_id:
            return "the search holds no result"
        stopped = str(run.stopped or "")
        if stopped != "budget_exhausted" and not stopped.endswith(RESERVE_RETAINED):
            return f"the search did not finish: it ended on {stopped!r}"
        return ""

    def report(self, run: Run) -> dict[str, Any]:
        """GEPA's figures, then the test evaluation's: ``test_eval`` (``done``,
        ``skipped``, ``incomplete``, ``not_run``) with ``test_eval_reason``, and the
        ``baseline_test``, ``best_test`` and ``best_test_delta`` scores. ``rollouts``
        (GEPA's metric calls) counts the search's, before any test rollout."""
        aggs = self.pool.aggregates()
        best = self.pool.index(self.best_id) if self.best_id else None
        rollouts = self.search_rollouts if self.search_rollouts is not None else run.rollouts_spent()
        result = self.final.result
        report: dict[str, Any] = {
            "programs": len(self.pool.candidate_ids),
            "frontier": len(find_dominator_programs(self.pool.front_programs, aggs)) if aggs else 0,
            "iterations": self.i + 1,
            "rollouts": rollouts,
            "stopped_by": self.stopped_by or None,
            "seed": self.base_seed,
            "pareto_set": self.pareto_set,
            "components": list(self.components),
            "best_program": best,
            "best_pareto_mean": aggs[best] if best is not None else None,
            "seed_pareto_mean": aggs[0] if aggs else None,
            "merges_accepted": self.total_merges_tested,
            "test_eval": result.get("status") or "not_run",
            **{f"count_{k}": v for k, v in sorted(self.stats.items())},
        }
        if result.get("reason"):
            report["test_eval_reason"] = result["reason"]
        for key in ("baseline_test", "best_test", "best_test_delta",
                    "best_test_delta_raw", "best_test_delta_left_out"):
            if key in result:
                report[key] = result[key]
        return report

    def best_candidate(self, population: "list[Candidate]") -> "Candidate | None":
        """GEPA's result (``core/result.py:76-78``), by id; an error when the population
        does not hold it."""
        if self.best_id is None:
            return None
        for candidate in population:
            if candidate.id == self.best_id:
                return candidate
        raise ValueError(f"gepa: the best program {self.best_id!r} is not in the population "
                         f"of {len(population)} candidates")

    def run_error(self) -> str | None:
        return self._run_error or None

    # ── state ─────────────────────────────────────────────────────────────

    def _commit(self, run: Run) -> None:
        """Snapshot the state at an iteration boundary and journal it. A resumed run
        starts the next iteration from here, as GEPA saves at the top of each
        (``engine.py:731-733``)."""
        version, internal, gauss = self.rng.getstate()
        a_version, a_internal, a_gauss = self.adapter_rng.getstate()
        self._checkpoint = {
            "gepa_state_version": STATE_VERSION,
            "base_seed": self.base_seed,
            "components": list(self.components),
            "pareto_ids": list(self.pareto_ids),
            "pool": self.pool.state_dict(),
            "i": self.i,
            "rng": [version, list(internal), gauss],
            "adapter_rng": [a_version, list(a_internal), a_gauss],
            "sampler": self.sampler.state_dict(),
            "search_rollouts": self.search_rollouts,
            "stopped_by": self.stopped_by,
            "val_size": self.val_size,
            "merges_due": self.merges_due,
            "total_merges_tested": self.total_merges_tested,
            "last_iter_found_new_program": self.last_iter_found_new_program,
            "merges_performed": [[list(t) for t in self.merges_performed[0]],
                                 [[a, b, list(d)] for a, b, d in self.merges_performed[1]]],
            "stats": dict(self.stats),
            "failed_streak": self.failed_streak,
            "best_candidate_id": self.best_id,
            "val_reserve": self.val_reserve,
            "finished": self.finished,
            "run_error": self._run_error,
            **self.gate.state_dict(),
            **self.final.state_dict(),
        }
        # The gate's best is the val-gated result only with a train Pareto set; GEPA's
        # result is the Pareto-best, so say that one.
        self._checkpoint["best_candidate_id"] = self.best_id
        run.save_strategy_state()

    def _save(self, run: Run) -> None:
        self._commit(run)

    def strategy_state_dict(self) -> dict[str, Any]:
        return dict(self._checkpoint)

    def load_strategy_state(self, data: dict[str, Any]) -> None:
        if not data:
            return
        if data.get("gepa_state_version") != STATE_VERSION:
            raise ValueError(f"gepa: cannot resume strategy state version "
                             f"{data.get('gepa_state_version')!r}; this code reads {STATE_VERSION}")
        self._fresh(int(data.get("base_seed", 0)))
        self.components = list(data.get("components") or [])
        self.pareto_ids = list(data.get("pareto_ids") or [])
        self.pool = ProgramPool.load(data.get("pool") or {})
        self.i = int(data.get("i", -1))
        version, internal, gauss = data["rng"]
        self.rng.setstate((version, tuple(internal), gauss))
        version, internal, gauss = data["adapter_rng"]
        self.adapter_rng.setstate((version, tuple(internal), gauss))
        self.sampler.load_state_dict(data.get("sampler") or {})
        rollouts = data.get("search_rollouts")
        self.search_rollouts = None if rollouts is None else int(rollouts)
        self.stopped_by = str(data.get("stopped_by") or "")
        self.val_size = int(data.get("val_size", 0))
        self.merges_due = int(data.get("merges_due", 0))
        self.total_merges_tested = int(data.get("total_merges_tested", 0))
        self.last_iter_found_new_program = bool(data.get("last_iter_found_new_program", False))
        performed = data.get("merges_performed") or [[], []]
        self.merges_performed = ([tuple(t) for t in performed[0]],
                                 [(a, b, tuple(d)) for a, b, d in performed[1]])
        self.stats = {str(k): int(v) for k, v in (data.get("stats") or {}).items()}
        self.failed_streak = int(data.get("failed_streak", 0))
        self.best_id = data.get("best_candidate_id")
        self.val_reserve = int(data.get("val_reserve", 0))
        self.finished = bool(data.get("finished", False))
        self._run_error = str(data.get("run_error") or "")
        self.gate.load_state_dict(data)
        self.final.load_state_dict(data)
        self._checkpoint = dict(data)

    # ── helpers ───────────────────────────────────────────────────────────

    def _count(self, key: str) -> None:
        self.stats[key] = self.stats.get(key, 0) + 1

    def _decide(self, step: str, **fields: Any) -> None:
        if self._trace is not None:
            self._trace.record_decision(step, iteration=self.i, **fields)


def discover_components(state: dict[str, Any], wanted: tuple[str, ...] | None) -> list[tuple[str, list[str]]]:
    """The components GEPA rewrites, each with the agent nodes that render it, in the
    knobs' declared order (GEPA keeps the seed candidate's key order, ``state.py:238``).

    ``wanted`` names them; else every open text knob exactly one agent node renders that
    is not bound to its system prompt (a per-node instruction); else those system-prompt
    knobs; else every open text knob an agent node renders; else every open text knob."""
    knobs = state.get("knobs") or {}
    nodes = (state.get("graph") or {}).get("nodes") or {}
    if isinstance(nodes, list):
        nodes = {str(n.get("id")): n for n in nodes}
    users: dict[str, list[str]] = {}
    system: set[str] = set()
    for nid, node in nodes.items():
        if node.get("kind") != "agent":
            continue
        for name in node.get("knobs") or ():
            users.setdefault(str(name), []).append(str(nid))
        for binding in node.get("field_bindings") or ():
            if len(binding) == 2 and binding[0] == "system_prompt_template":
                system.add(str(binding[1]))

    def open_text(name: str) -> bool:
        spec = (knobs.get(name) or {}).get("spec") or {}
        return spec.get("kind") == "str" and not spec.get("frozen")

    if wanted:
        for name in wanted:
            if name not in knobs:
                raise ValueError(f"gepa: the mode has no {name!r} knob; the components option "
                                 "names open text knobs")
            if not open_text(name):
                raise ValueError(f"gepa: knob {name!r} is not an open text knob")
        return [(name, users.get(name, [])) for name in wanted]
    texts = [name for name in knobs if open_text(name)]
    for rule in (lambda n: len(users.get(n, [])) == 1 and n not in system,
                 lambda n: len(users.get(n, [])) == 1,
                 lambda n: bool(users.get(n))):
        found = [(name, users[name]) for name in texts if rule(name)]
        if found:
            return found
    if texts:
        # No agent node renders one (an op reads them): reflect on outcomes alone.
        return [(name, []) for name in texts]
    raise ValueError("gepa: the mode has no open text knob; name the knobs to rewrite with "
                     "the components option")


def _component_texts(candidate: Candidate, names: list[str]) -> dict[str, str]:
    knobs = candidate.state.get("knobs") or {}
    out: dict[str, str] = {}
    for name in names:
        entry = knobs.get(name)
        value = entry.get("value") if isinstance(entry, dict) else entry
        out[name] = "" if value is None else str(value)
    return out


def _scores(records: list[dict[str, Any]]) -> dict[str, float]:
    """Each scored record's score by item id (an unscored one is left out)."""
    out: dict[str, float] = {}
    for record in records:
        score = record.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            out[str(record.get("id"))] = float(score)
    return out


def _slot_scores(slots: list[dict[str, Any] | None]) -> list[float | None]:
    """Each minibatch position's score; ``None`` where the item went unscored."""
    out: list[float | None] = []
    for record in slots:
        score = record.get("score") if record is not None else None
        ok = isinstance(score, (int, float)) and not isinstance(score, bool)
        out.append(float(score) if ok else None)  # type: ignore[arg-type]
    return out


def _ordered(scores: dict[str, float], ids: list[str]) -> dict[str, float]:
    """``scores`` in the order of ``ids`` (the Pareto set's), so the frontier, built in
    key order, does not depend on the order records came back in."""
    return {item: scores[item] for item in ids if item in scores}


def _held(run: Run, candidate_id: str) -> Candidate:
    candidate = run.candidate(candidate_id)
    if candidate is None:
        raise OuterLoopError(f"gepa: the search does not hold candidate {candidate_id!r}")
    return candidate


def _positive(name: str, value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise TypeError(f"gepa: {name} must be a positive integer, not {value!r}") from None
    if number <= 0 or isinstance(value, bool):
        raise TypeError(f"gepa: {name} must be a positive integer, not {value!r}")
    return number


def _flag(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise TypeError(f"gepa: {name} must be true or false, not {value!r}")


def _choice(name: str, value: Any, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise TypeError(f"gepa: {name} must be one of {', '.join(allowed)}, not {value!r}")
    return str(value)


def _names(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    names = [n.strip() for n in value.split(",")] if isinstance(value, str) else list(value)
    if not names or not all(isinstance(n, str) and n for n in names) or len(set(names)) != len(names):
        raise TypeError(f"gepa: components must be a comma-separated list of knob names, not {value!r}")
    return tuple(names)
