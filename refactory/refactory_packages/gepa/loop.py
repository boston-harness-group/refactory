"""GepaLoop: GEPA as an OuterLoop strategy on refactory's public extension points.

A port of ``gepa.optimize`` (``gepa`` 0.1.4, PyPI; gepa-ai/gepa, MIT) with its
default configuration, which is classic GEPA (Agrawal et al., 2025): one parent, one
reflective mutation of one component and one minibatch per iteration
(``core/engine.py:557-918``).

Per iteration:

1. (merge on) Right after an iteration that accepted a child, try a merge (``merge.py``).
2. Draw a parent from the per-instance Pareto frontier, weighted by the instances it
   is best on (``pareto.py``), and the iteration's minibatch of train ids
   (``sampler.py``).
3. Roll the parent out on the minibatch; skip the iteration if every item is perfect.
4. Pick the parent's next component round robin, build its reflective dataset from
   the records (``reflection.py``), ask the reflection model for a new instruction.
5. Roll the child out on the same minibatch; keep it only if its sum beats the
   parent's (strict), then score it on the whole Pareto set and add it to the pool.

The result is the program with the highest mean on the Pareto set.

Mapping to refactory:

- A **component** is an open text knob. By default, the knobs that exactly one agent
  node renders and that are not its system prompt: per-node instructions, as DSPy's
  components are its predictors' instructions. The ``components`` option names them.
- GEPA's **trainset** is the parts manifest's train ids; its **valset** (the Pareto
  set) is the val part (``pareto_set="val"``, the default, as ``dspy.GEPA`` is run
  with a valset). Val is rolled out in full and read as per-item scores keyed by
  opaque digests (``Run.val_item_scores``); the reflection model never sees a val
  item. ``pareto_set="train"`` instead holds ``pareto_size`` train ids out as the
  Pareto set (the leakage-free variant of ``docs/plans/gepa-mipro-outer-loop.md``);
  the seed and the Pareto-best are then scored on val, and the Pareto-best is the
  result only if it beats the seed there.
- A **metric call** is a rollout: GEPA's ``max_metric_calls`` is the run's
  ``max_rollouts``, checked at the top of each iteration as GEPA checks it
  (``engine.py:731``). Refactory refuses an evaluation once the budget is spent, so
  where GEPA would overshoot inside the last iteration the run stops there.

Defaults (``api.py:46-103``):
  - seed = 0                       (``api.py:94``)
  - reflection_minibatch_size = 3  (``api.py:61``, ``:351``)
  - candidate_selection_strategy = "pareto"  (``api.py:57``; also "current_best")
  - skip_perfect_score = True, perfect_score = 1.0  (``api.py:59``, ``:62``)
  - module_selector = "round_robin"  (``api.py:66``; also "all")
  - use_merge = False, max_merge_invocations = 5, merge_val_overlap_floor = 5
                                   (``api.py:68-70``)
  - reflection_prompt_template = GEPA's default (``instruction_proposal.py:13-29``)

Deviations, each forced by refactory or chosen for robustness:

- A merge is scored on 5 sampled Pareto instances (``merge.py:332-392``). Val can only
  be rolled out whole, so with ``pareto_set="val"`` the merged program is rolled out on
  all of val and judged on the same 5 instances by GEPA's rule; that rollout is then its
  full evaluation when it is accepted. A merge costs ``|val|``, not 5.
- GEPA aborts the run on any error in an iteration (``raise_on_exception=True``,
  ``engine.py:866-867``). Here a reflection call that fails every attempt skips its
  component, and the run stops only after ``max_failed_iterations`` iterations in a row
  whose every call failed.
- Minibatch sums are taken over the items both parent and child were scored on: an item
  a harness incident left unscored is no one's win or loss.
- A child whose state the search already holds (the reflection reproduced a text) is
  not rolled out again.
"""

from __future__ import annotations

import random
from typing import Any

from refactory_dsh.budgets import BudgetExhausted
from refactory_dsh.optim import Candidate, Proposal
from refactory_dsh.outerloop import BudgetStopped, OuterLoop, OuterLoopError, Run, ValidationGate

from refactory_packages.gepa import merge as merge_mod
from refactory_packages.gepa.pareto import (
    ProgramPool, find_dominator_programs, select_from_pareto_front,
)
from refactory_packages.gepa.reflection import (
    DEFAULT_PROMPT_TEMPLATE, check_template, extract_instruction, outcome_dataset,
    reflection_prompt, reflective_dataset,
)
from refactory_packages.gepa.sampler import EpochShuffledBatchSampler
from refactory_packages.skillopt.trace import TraceRecorder, ask

#: The version of the strategy state this code writes and reads.
STATE_VERSION = 1


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
        "candidate_selection_strategy", "components", "feedback", "max_failed_iterations",
        "max_merge_invocations", "merge_val_overlap_floor", "module_selector",
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

        self.base_seed = int(options.get("seed", 0))
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

        self._fresh()

    def _fresh(self) -> None:
        """The search state before the first iteration."""
        # One shared generator for frontier sampling, minibatches and merge (``api.py:304``),
        # and the DSPy adapter's own for picking a call (``dspy_adapter.py:111``).
        self.rng = random.Random(self.base_seed)
        self.adapter_rng = random.Random(0)
        self.sampler = EpochShuffledBatchSampler(self.minibatch_size, self.rng)
        self.pool = ProgramPool()
        self.gate = ValidationGate(seed=self.base_seed)
        self.components: list[str] = []
        self.pareto_ids: list[str] = []
        self.i = -1
        self.metric_calls = 0
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
        self._trace = TraceRecorder(run.root / "gepa_trace.jsonl")
        self._trace.listener = run.record_proposal_round
        seed = run.seed()
        component_nodes = discover_components(seed.state, self.components_option)
        names = [name for name, _ in component_nodes]
        if self.components and self.components != names:
            raise OuterLoopError(f"gepa: this run optimized components {self.components}, "
                                 f"not {names}; resume it with the same components option")
        self.components = names
        self.nodes = dict(component_nodes)
        train_ids = run.part_ids("train")
        if not train_ids:
            raise ValueError("gepa: the minibatches need the parts manifest's train ids "
                             "(run.part_ids('train') returned none)")
        if self.pareto_set == "val" and not run.has_part("val"):
            raise ValueError("gepa: pareto_set=val needs a parts manifest with a 'val' part; "
                             "give the run one, or set pareto_set=train")
        self.feedback_ids = self._split(train_ids)
        try:
            if not self.finished:
                if not self.pool.candidate_ids:
                    self._start(run, seed)
                self._search(run)
            self._finish(run)
        except _OutOfBudget:
            self._finish(run)
        except BudgetStopped as exc:
            run.stop(exc.reason)
        except BudgetExhausted as exc:
            run.stop(str(exc))
        self._save(run)

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
        scores = self._pareto_scores(run, seed)
        if self.pareto_set == "train":
            # The seed's val score: the bar the result must beat, and how much of the
            # budget to keep for scoring the result on val.
            self.gate.start(run, seed)
            self.val_reserve = max(1, len(run.val_item_scores(seed)))
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
        records = self._rollout(run, parent, minibatch)
        before = _scores(records)
        self._count("iterations")
        if not records:
            self._decide("skip_no_trajectories", parent=parent_idx, minibatch=minibatch)
            return
        if self.skip_perfect_score and all(
                before.get(item) is not None and before[item] >= self.perfect_score
                for item in minibatch):
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
                continue
            new_texts[name] = extract_instruction(raw.strip())
        changed = {n: t for n, t in new_texts.items() if t != texts[n]}
        if not changed:
            self._decide("no_new_text", parent=parent_idx, components=chosen,
                         proposed=sorted(new_texts))
            return
        child = self._attempt(run, parent, changed, f"gepa iteration {self.i}: reflective "
                                                    f"update of {', '.join(changed)}")
        if child is None:
            return
        child_records = self._rollout(run, child, minibatch)
        after = _scores(child_records)
        shared = [item for item in minibatch if item in before and item in after]
        old_sum = sum(before[item] for item in shared)
        new_sum = sum(after[item] for item in shared)
        accepted = new_sum > old_sum
        self._decide("reflect", parent=parent_idx, components=sorted(changed), minibatch=minibatch,
                     parent_sum=old_sum, child_sum=new_sum, accepted=accepted, child=child.id)
        if not accepted:
            self._count("rejected")
            return
        self._add(run, child, [parent_idx], "reflective")
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
        child = self._attempt(run, ancestor_candidate, changed,
                              f"gepa iteration {self.i}: merge of programs {id1} and {id2} "
                              f"over their ancestor {ancestor}") if changed else None
        if child is None:
            return True
        if self.pareto_set == "val":
            full = self._pareto_scores(run, child)
            new = [full.get(k, 0.0) for k in subsample]
        else:
            full = None
            new_scores = _scores(self._rollout(run, child, subsample))
            new = [new_scores.get(k, 0.0) for k in subsample]
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
                 rationale: str) -> Candidate | None:
        """The candidate the new texts make from ``parent``; ``None`` when refused."""
        edits = [{"op": "set_knob", "name": name, "value": text} for name, text in texts.items()]
        outcome = run.attempt(Proposal(edits=tuple(edits), rationale=rationale, parent=parent.id),
                              parent)
        if isinstance(outcome, list):
            self._count("refused")
            self._decide("refused", reasons=outcome[:5], components=sorted(texts))
            return None
        return outcome

    def _add(self, run: Run, child: Candidate, parents: list[int], how: str,
             scores: dict[str, float] | None = None) -> None:
        """Score an accepted program on the whole Pareto set and add it
        (``engine.py:517-533``)."""
        discovered = self.metric_calls
        if scores is None:
            scores = self._pareto_scores(run, child)
        run.keep(child)
        if self.pool.index(child.id) is not None:
            self._decide("duplicate", child=child.id)
            return
        idx = self.pool.add(child.id, parents, scores, discovered_at=discovered)
        self.best_id = self.pool.candidate_ids[self.pool.best()]
        self._count("accepted_" + how)
        self._decide("added", program=idx, candidate=child.id, parents=parents, how=how,
                     pareto_mean=self.pool.aggregate(idx), best=self.best_id)

    def _pareto_scores(self, run: Run, candidate: Candidate) -> dict[str, float]:
        """The candidate's per-instance scores on the Pareto set: val, rolled out whole, or
        the held-out train ids."""
        if self.pareto_set == "val":
            self._spend(run, 1)
            self.gate.score(run, candidate)
            scores = run.val_item_scores(candidate)
            self.metric_calls += len(scores)
            return scores
        return _scores(self._rollout(run, candidate, self.pareto_ids))

    def _rollout(self, run: Run, candidate: Candidate, items: list[str]) -> list[dict[str, Any]]:
        """Roll ``candidate`` out on train ``items`` and return their records."""
        self._spend(run, len(items))
        run.evaluate(candidate, part="train", items=list(dict.fromkeys(items)), seed=self.base_seed)
        self.metric_calls += len(items)
        wanted = set(items)
        return [r for r in run.item_records(candidate) if str(r.get("id")) in wanted]

    def _budget(self, run: Run) -> int:
        """Rollouts the search may still start, the val reserve kept back."""
        return run.budget_left() - self.val_reserve

    def _spend(self, run: Run, n: int) -> None:
        if run.budget_left() <= 0:
            raise _OutOfBudget

    # ── the end ───────────────────────────────────────────────────────────

    def _finish(self, run: Run) -> None:
        """With ``pareto_set="train"``, the Pareto-best must beat the seed on val; the
        result is the program with the highest Pareto mean (``core/result.py:76-78``)."""
        if self.finished or not self.pool.candidate_ids:
            return
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

    def report(self, run: Run) -> dict[str, Any]:
        aggs = self.pool.aggregates()
        best = self.pool.index(self.best_id) if self.best_id else None
        return {
            "programs": len(self.pool.candidate_ids),
            "frontier": len(find_dominator_programs(self.pool.front_programs, aggs)) if aggs else 0,
            "iterations": self.i + 1,
            "metric_calls": self.metric_calls,
            "pareto_set": self.pareto_set,
            "components": list(self.components),
            "best_program": best,
            "best_pareto_mean": aggs[best] if best is not None else None,
            "seed_pareto_mean": aggs[0] if aggs else None,
            "merges_accepted": self.total_merges_tested,
            **{f"count_{k}": v for k, v in sorted(self.stats.items())},
        }

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
            "components": list(self.components),
            "pareto_ids": list(self.pareto_ids),
            "pool": self.pool.state_dict(),
            "i": self.i,
            "rng": [version, list(internal), gauss],
            "adapter_rng": [a_version, list(a_internal), a_gauss],
            "sampler": self.sampler.state_dict(),
            "metric_calls": self.metric_calls,
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
        self._fresh()
        self.components = list(data.get("components") or [])
        self.pareto_ids = list(data.get("pareto_ids") or [])
        self.pool = ProgramPool.load(data.get("pool") or {})
        self.i = int(data.get("i", -1))
        version, internal, gauss = data["rng"]
        self.rng.setstate((version, tuple(internal), gauss))
        version, internal, gauss = data["adapter_rng"]
        self.adapter_rng.setstate((version, tuple(internal), gauss))
        self.sampler.load_state_dict(data.get("sampler") or {})
        self.metric_calls = int(data.get("metric_calls", 0))
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
