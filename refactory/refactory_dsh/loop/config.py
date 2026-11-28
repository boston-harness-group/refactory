
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

def opaque_item_key(item_id: str, secret: str) -> str:
    """The key `Loop.part_item_scores` gives an item: an HMAC of its id under the run's
    secret (`Loop.item_key_secret`), so a strategy can line two candidates' per-item
    val scores up without learning a held-out id. Keyed rather than a bare hash, so a
    known list of dataset ids cannot be hashed to undo it; the journal, which holds
    the raw ids, is not a boundary this draws."""
    import hashlib
    import hmac

    return hmac.new(secret.encode(), f"refactory-item:{item_id}".encode(),
                    hashlib.sha256).hexdigest()[:16]

def item_time_spread(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How long train items took each, from their records' ``duration_s``: ``per_item_median_s``,
    ``per_item_p90_s``, ``per_item_max_s`` and ``timed_items`` (how many records had a
    duration); empty when none did. One slow item moves the longest, never the median."""
    durations = sorted(float(r["duration_s"]) for r in records
                       if isinstance(r.get("duration_s"), (int, float))
                       and not isinstance(r.get("duration_s"), bool))
    if not durations:
        return {}
    middle = len(durations) // 2
    median = (durations[middle] if len(durations) % 2
              else (durations[middle - 1] + durations[middle]) / 2)
    p90 = durations[min(len(durations) - 1, max(0, -(-len(durations) * 9 // 10) - 1))]
    return {"per_item_median_s": round(median), "per_item_p90_s": round(p90),
            "per_item_max_s": round(durations[-1]), "timed_items": len(durations)}

def _values_before(parent: Candidate | None, edits: Sequence[Any]) -> dict[str, Any]:
    """The parent's values of the knobs ``edits`` set (`optim.base.whole_edits`)."""
    knobs = (parent.state.get("knobs") or {}) if parent is not None else {}
    return {edit["name"]: (knobs[edit["name"]] or {}).get("value") for edit in edits
            if isinstance(edit, dict) and edit.get("op") == "set_knob"
            and isinstance(edit.get("name"), str) and edit["name"] in knobs}

def _new_steps(parent: Any, child: Any) -> set[str]:
    """The child graph's steps its parent's graph lacks (none without a parent)."""
    if parent is None:
        return set()

    def nodes(candidate: Any) -> set[str]:
        return set(((candidate.state.get("graph") or {}).get("nodes") or {}))

    return nodes(child) - nodes(parent)

class LoopError(RuntimeError):
    """The run cannot proceed. Always names what is wrong with the mode."""

class EvaluationInterrupted(LoopError):
    """An evaluation whose items died with the run's own processes, not on their answers.

    A failed item is the candidate's result only when it ran and failed. One whose
    process was killed, whose wire to the model kept failing, or that the run's stop
    cancelled has an unknown outcome (`items.interrupted_item`), so the evaluation is
    no measurement: nothing of it is recorded as the candidate's score, records or
    val result. The loop runs it again inside the run after each wait of
    `INTERRUPTED_RETRY_WAITS_S`; still interrupted, it stops the run here (exit
    `EXIT_INTERRUPTED`), as it would had its own process died. The candidate stays
    unevaluated, so ``--resume`` evaluates it again from the start -- up to
    `INTERRUPTION_CAP` stops, after which the evaluation is recorded as failed.
    """

#: The waits, in seconds, before each in-run retry of an interrupted evaluation; the
#: evaluation is tried once more after each, so ``1 + len(...)`` attempts before the run
#: stops on it. No wall-clock cap: each attempt has the run's usual limits.
INTERRUPTED_RETRY_WAITS_S: tuple[float, ...] = (120.0, 300.0)

#: How many times the same candidate's evaluation of the same part (a re-score counts
#: per epoch) may stop the run as interrupted. An in-run retry is not a stop: one stop is
#: an evaluation still interrupted after all its in-run retries. The run stops on the
#: first ``INTERRUPTION_CAP - 1``; the ``INTERRUPTION_CAP``-th is recorded as a failed
#: evaluation ("interrupted N times; recorded as failed") and the run goes on, so an
#: evaluation that dies every time cannot hold a run in a stop-and-resume loop. Counted
#: from the journal, so a resume keeps the count.
INTERRUPTION_CAP = 3

#: How many times the loop runs an item again, on the same candidate and part, when the
#: harness failed it (`items.harness_incident`: its grader or sandbox): as many as an interrupted
#: evaluation's in-run retries. Still an incident after them, the item is left out of
#: the evaluation's score -- its numerator and its denominator -- and said wherever the
#: score is shown (`Loop._settle_incidents`). An incident is the harness's failure,
#: never the candidate's wrong answer.
INCIDENT_RERUNS = len(INTERRUPTED_RETRY_WAITS_S)

#: Where a trial's incident reruns run, one numbered folder per rerun, inside the trial.
INCIDENT_RERUN_DIR = "incident_reruns"

#: The exit status of a run stopped by an interrupted evaluation (`EvaluationInterrupted`):
#: ``EX_TEMPFAIL``, safe to ``--resume``, unlike a refused configuration (2).
EXIT_INTERRUPTED = 75

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

#: ``LoopConfig.selection`` values under ``val_gate``: empty picks the default
#: (``default_selection``); ``checkpoint`` applies every change whose graph is not broken
#: and keeps the best on validation as the winner; ``single`` is SkillOpt's line;
#: ``pool`` keeps the best candidates on validation as parents.
SELECTIONS = ("", "single", "pool", "checkpoint")

#: Why a change is not applied under ``checkpoint`` (``Loop._apply_round``), short for a
#: request's gate column: its train run failed, or a sibling scored higher on train.
NOT_APPLIED = "not applied"

#: The refusal code of a change whose graph failed its smoke check (``Loop._smoke_check``);
#: a mechanics refusal, so the slim repair call may fix it (`optim.repair.MECHANICS_CODES`).
SMOKE_CODE = "E-smoke"

#: What a run without validation items says, in its log and its report, when it
#: falls back to selecting on train scores (``LoopConfig.val_gate`` left unset).
NO_VALIDATION = "no validation items: selecting on train scores"

#: ``LoopConfig.portfolio`` values: off, or one candidate per edit level.
PORTFOLIOS = ("", "prompt+pipeline")

def default_selection(strategy: str, optimizer: str, picker: str = "",
                      project: Path | None = None) -> str:
    """The ``selection`` a ``val_gate`` run gets when it names none: ``checkpoint`` for
    the default loop with a single-parent optimizer, ``pool`` for every other strategy,
    optimizer or parent picker. A SkillOpt-gate parity run names ``single``."""
    if single_parent(strategy, optimizer, picker, project):
        return "checkpoint"
    return "pool"

def single_parent(strategy: str, optimizer: str, picker: str = "",
                  project: Path | None = None) -> bool:
    """Whether a run is the default loop with a single-parent optimizer and no
    population picker (``optim.is_population``: an archive, a tournament, islands,
    niche champions, or a package's picker that declares ``population``): the only
    runs ``checkpoint`` may select."""
    from refactory_dsh import optim

    return (strategy == "default" and not optim.is_population(optimizer, project)
            and not optim.is_population(picker, project))

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
    #: The Claude Code executable every claude_code worker runs (an absolute path);
    #: empty keeps the one bundled with the Agent SDK. ``node_routes``/``role_routes``
    #: override it per node.
    claude_executable: str = ""
    role_routes: dict[str, Any] = dataclasses.field(default_factory=dict)
    node_routes: dict[str, Any] = dataclasses.field(default_factory=dict)
    provider_config: dict[str, Any] = dataclasses.field(default_factory=dict)
    max_elapsed_s: float | None = None
    max_tokens: int | None = None
    validation_reserve_s: float = 0
    validation_reserve_tokens: int = 0
    max_rollouts: int | None = None
    #: How many evaluations run at once: in evaluate_many, and in the default loop's
    #: rounds (a round's train evaluations, then its val-gate rollouts; under
    #: ``selection`` ``checkpoint`` a lone child's val rollout runs beside its train run).
    #: 1 runs them one at a time. Default 4.
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
    #: strictly higher val score. Under ``selection`` ``checkpoint`` (the default for
    #: a single-parent optimizer) the train seed is fixed instead, and only the applied
    #: candidates are rolled out on val (see ``selection``). Needs a manifest with a val
    #: part; a custom strategy (SkillOpt) keeps its own gate instead.
    #:
    #: Not a setting: the run decides it (``Loop._resolve_val_gate``). On when the parts
    #: manifest has val items and the strategy is a built-in one, off otherwise: a run
    #: with no validation items selects on train scores and says so
    #: (``NO_VALIDATION``). ``Loop.init`` replaces ``None`` with the decision before
    #: anything runs, so ``config.json`` records it. A value given in code must be the
    #: one the run decides; there is no way to turn the gate off for a run with val
    #: items.
    val_gate: bool | None = None
    #: Train items per epoch: 0 (the default) scores every candidate on the whole train
    #: part. N > 0, when the train part has more than N items, samples N train ids each
    #: epoch with SkillOpt's per-epoch seed (``train_batch_seed``), scores every
    #: candidate of that epoch on that sample, and re-scores the edited candidate on it
    #: when its records are from another epoch, so the proposal evidence is that
    #: epoch's minibatch (as SkillOpt rolls out its current candidate on each step's
    #: batch). Val and test are unaffected. Each epoch's ids are journaled.
    train_minibatch: int = 0
    #: Whether an edit may add a code node: an op or a gate whose Python the optimizer
    #: writes, run sandboxed (`refactory_dsh.snippet`). On by default; it takes effect
    #: only where the run permits structural edits, and a machine that cannot sandbox
    #: the code refuses code nodes either way (``E-code-nodes``).
    allow_code_nodes: bool = True
    #: Whether a step an edit adds may have tools (``insert``/``map``/``fallback`` with
    #: ``tools``, or a copy of a seed step that has some). On by default: such a step runs
    #: sealed, seeing only the files it reads, with no network and no grader file. Off (an
    #: ablation), the request offers no tool step and every new agent step with tools is
    #: refused (``E-new-node-tools``); a step the seed has keeps its own tool rules either way.
    new_step_tools: bool = True
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
    #: ``map_elites``, ``pareto``, or one a package contributes) that picks the
    #: candidate each step edits, while ``optimizer`` writes the edits
    #: (``optim.compose.PickedParent``). Empty (the default): the optimizer alone, as
    #: before.
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
    #: - ``checkpoint``: training as in ML. Each round's change is applied (of several,
    #:   the one with the highest train score, the earlier on a tie) unless its graph is
    #:   broken: refused by the validator, vetoed by the integrity judge, failing the
    #:   smoke check after repair, or failing its train run. Only the applied candidate
    #:   is rolled out on val, once: a checkpoint. The seed's and a round's lone child's val
    #:   run goes beside its train run, sharing ``max_items_in_flight`` (a round takes
    #:   about the slower of the two when the limit fits both); val runs after train with
    #:   several children, ``max_concurrent_evaluations`` 1, a time or token budget, too
    #:   little ``max_rollouts`` room, its val result already held (a resume), another
    #:   candidate still waiting for a gate decision (a resume), a parts manifest with no
    #:   val, or a model client other than ``DshModelClient``. Each run's tokens and cost leave out what
    #:   the other spent, for workers that write receipts. The winner is
    #:   the best checkpoint on val (a tie goes to the earlier one); the early stop counts
    #:   rounds without a new best. No parent is re-scored: train rollouts use one seed
    #:   for the whole run (the first epoch's), so a child and its parent are compared on
    #:   the same shuffle.
    #:   The default loop with a single-parent optimizer only.
    #: - ``single``: SkillOpt's line. A candidate is kept only if its val score strictly
    #:   beats the incumbent's (the gate's current candidate), and the incumbent (with
    #:   any candidate a person kept) is the only parent.
    #: - ``pool``: every judged candidate may enter a pool of the ``pool_size`` best by
    #:   val score (ties at the edge kept). Parents come from the pool, each strategy
    #:   and parent picker choosing among them by val. The incumbent still changes only
    #:   on a strictly higher val score (a tie goes to the higher train score, then the
    #:   earlier candidate), and it is the winner.
    #:
    #: Empty (the default) picks ``default_selection``: ``checkpoint`` for the default
    #: loop with a single-parent optimizer, ``pool`` for every other strategy and for the
    #: population optimizers and pickers. No effect without ``val_gate``.
    selection: str = ""
    #: The pool's size under ``selection`` ``pool``: the candidates with the best val
    #: scores, plus any tied with the last of them.
    pool_size: int = 8
    #: The model proposer's request lists every evaluated candidate of the run, not only
    #: the recent ones, under "Every change so far": what its own edits did
    #: (`optim.feedback.change_label`, the kind only as a label), its train fixed/broken
    #: counts against its parent and the gate's word, after a tally per change (never per
    #: kind). On by default.
    run_ledger: bool = True
    #: The memory pass (`optim.memory`, SkillOpt's optimizer meta skill): once per
    #: epoch from epoch 2, before a model proposer's proposal, one optimizer-model call
    #: compares the candidate current when the previous epoch proposed with the one
    #: current now on 20 sampled train items (reusing the epoch's own rollouts; only
    #: what is missing is evaluated) and writes one memory note that every analyst
    #: call, every merge call and the proposal request read. Journaled, kept across a
    #: resume; a failed pass keeps the earlier note, marked not updated. On by default.
    lessons: bool = True
    #: Each step a candidate of the last round added is reviewed before the next
    #: proposal: its train fixed/broken counts, how often the final answer changed,
    #: and one call of the optimizer's model per step on up to 4 train items
    #: (`optim.feedback.review_prompt`), shown as "Did the added steps contribute?".
    #: On by default; a model proposer only.
    step_review: bool = True
    #: "How nodes compose" also says what makes a new step pay off
    #: (`optim.feedback.STRUCTURE_GUIDANCE`). Off by default (an experiment's switch).
    structure_guidance: bool = False
    #: The analyst stage (`optim.analysts`): before each proposal of a model proposer
    #: (autoresearch, alone, in a portfolio or under a parent picker), failure and
    #: success analysts read the edited candidate's train items in full, in batches of
    #: ``analyst_group_size``, one optimizer-model call per batch, and their findings
    #: are merged in batches of ``analyst_merge_batch_size`` into one ranked list, shown
    #: as "What the analysts found, ranked" in place of the readers' summaries;
    #: journaled (``analyst_findings``) and summarised in report.md. SkillOpt's
    #: per-round analysis. Off by default: off makes no call and changes no request.
    analysts: bool = False
    #: Items per analyst call (SkillOpt's ``minibatch_size``, 8).
    analyst_group_size: int = 8
    #: Finding lists per merge call (SkillOpt's ``merge_batch_size``, 8; at least 2).
    analyst_merge_batch_size: int = 8
    #: The most analyst calls per round, failure batches first; 0 (the default, as
    #: SkillOpt) is no limit.
    analyst_max_batches: int = 0
    #: The integrity judge (`optim.integrity`): one call of the optimizer's model per
    #: proposal that became a valid candidate, asking whether its benefit would hold if
    #: the dataset, grader or environment had been built differently, and at the end one
    #: on the winner's combined change from the seed. ``off`` (the default) makes no
    #: call and changes nothing. On, every verdict is recorded and shown in the change
    #: ledger, the next request and the report, and each category has a policy: ``off``
    #: (not asked about), ``warn`` (shown; the change is still evaluated) or ``veto`` (an
    #: ``exploit`` or ``suspect`` verdict about what the change adds, naming it, is
    #: refused before evaluation, ``E-integrity``). An ``inherited`` finding or a failed
    #: judgement never vetoes. ``on``/``veto`` turn it
    #: on with every category ``veto`` by default, ``flag``/``warn`` with every one
    #: ``warn``; ``{"mode": ..., "default": ..., "policy": {category: policy}}`` sets
    #: categories one by one. A category's policy: the run's own for it, else a Task's
    #: allowance (``Task.integrity_allowances``: off), else the run's default, else (an
    #: object with no default) the Tasks' ``integrity_policy``, else veto
    #: (`optim.integrity.resolve_policy`).
    integrity_judge: str | dict[str, Any] = "off"

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

#: No value journaled yet (`_note_recorded`'s ``bound``).
_UNSET = object()
