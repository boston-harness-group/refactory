
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

from refactory_dsh.loop.config import HOLDOUT_FILE, LoopConfig, NOT_APPLIED, OVERFIT_THRESHOLD, SEED_WINNER_NOTE, gap_standard_error


def node_count(state: dict[str, Any]) -> int:
    """How many nodes a candidate's graph holds, which is what the size cost counts."""
    graph = state.get("graph") or {}
    return len(graph.get("nodes") or {})

def _changes_cell(tally: dict[str, Any] | None) -> str:
    """A proposal's edits as the cap counts them (`optim.edit_count`): its changes, then
    its raw edits, structural and knob, and the knob text characters it changed."""
    if not tally:
        return ""
    return (f"{tally.get('logical')} ({tally.get('raw')} edits: {tally.get('structural')} "
            f"structural, {tally.get('knob')} knob; {tally.get('chars')} chars)")

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

#: Where a sealed step's dispatch record sits under a candidate's run directory
#: (`workspace.prepare`), at any depth: the run, an item's pass, and every occasion
#: (``occasions/<part>_N``, ``train_epoch_N``, memory rollouts, smoke runs) and its passes.
_DISPATCH_RECORDS = "*/**/.refactory/workspaces/*/dispatch.json"

def uncapped_sealed_steps(runs: Path) -> dict[str, Any]:
    """The sealed steps under ``runs`` (a run's candidate runs) whose dispatch record says
    they ran without a whole-step cap (`seal.step_cap`): how many, and the first
    reason; empty when none did."""
    found: list[str] = []
    # ``**`` does not follow links, so a concurrent pass's alias (``.refactory/live``) is
    # not counted a second time.
    for path in sorted(Path(runs).glob(_DISPATCH_RECORDS)):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and record.get("whole_step_cap") is False:
            found.append(str(record.get("cap") or "no cap"))
    return {"dispatches": len(found), "why": found[0]} if found else {}

def _uncapped_lines(uncapped: Mapping[str, Any] | None) -> list[str]:
    """The report's one line on sealed steps that ran without a whole-step cap."""
    if not uncapped:
        return []
    return ["", f"Sealed steps ran without a whole-step memory cap "
                f"({uncapped.get('dispatches')} step(s)): {uncapped.get('why')}."]

def _incident_lines(incidents: Sequence[dict[str, Any]]) -> list[str]:
    """The report's say on harness incidents (`Loop._settle_incidents`): one line per
    evaluation that left an item out of its score, after a line saying what that means."""
    left = [entry for entry in incidents if entry.get("not_scored")]
    if not left:
        return []
    return ["", f"**{sum(len(set(e['not_scored'])) for e in left)} item(s) in {len(left)} "
                "evaluation(s) had a harness incident and are not scored**: the harness "
                "failed them (a grader or sandbox), again on every rerun, so they count in no score "
                "below, not as wrong answers.",
            *(f"- {e.get('candidate')} {e.get('part')}: {', '.join(map(str, e['not_scored']))} "
              f"(rerun {e.get('reran', 0)} time(s))" for e in left)]

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

#: The latest round's findings report.md lists.
REPORT_ANALYST_FINDINGS = 5

def _markdown_text(text: str) -> str:
    """``text`` as one line of a markdown report: whitespace and line breaks collapsed,
    and a run of backticks (a code fence a model wrote) made one, so it opens no block
    that swallows the rest of the report; a run of tildes (the other fence) likewise."""
    one = re.sub(r"`{2,}", "`", " ".join(str(text).split()))
    return re.sub(r"~{3,}", "~", one)

def _analysts_report(analyses: Sequence[dict[str, Any]], config: LoopConfig) -> dict[str, Any]:
    """What report.json says of the analyst stage: its settings, the rounds it ran, the
    model calls they made (cached ones apart), and the latest round's analysis."""
    return {"group_size": int(config.analyst_group_size),
            "merge_batch_size": int(config.analyst_merge_batch_size),
            "max_batches": int(config.analyst_max_batches),
            "rounds": len(analyses),
            "calls": sum(int(a.get("calls") or 0) for a in analyses),
            "cached": sum(int(a.get("cached") or 0) for a in analyses),
            "failed_batches": sum(1 for a in analyses for b in a.get("batches") or ()
                                  if b.get("error")),
            "clipped_items": sum(int(a.get("clipped_items") or 0) for a in analyses),
            "clipped_chars": sum(int(a.get("clipped_chars") or 0) for a in analyses),
            "context_retries": sum(int(a.get("context_retries") or 0) for a in analyses),
            "stopped": sum(1 for a in analyses if a.get("error")),
            "latest": dict(analyses[-1]) if analyses else None}

def _headline(payload: dict[str, Any]) -> list[str]:
    """The report's score line. Under ``checkpoint`` (``seed_val`` is there) like with
    like: the seed's val checkpoint against the winner's, and the seed's train score
    against the winner's; otherwise the seed's score against the best's, as before."""
    parsimony = (f"The compared score is the raw one minus "
                 f"{payload['scoring']['parsimony']} per node of the candidate's graph.")
    if "seed_val" not in payload:
        return [f"Seed score **{payload['seed_score']}** → best **{payload['best_score']}**"
                f" (raw {payload['seed_raw_score']} → {payload['raw_best_score']}). "
                + parsimony]
    seed_val = "n/a" if payload["seed_val"] is None else payload["seed_val"]
    train = ("Train, on different samples (each epoch's own `train_minibatch` items)"
             if payload.get("train_samples_differ") else "Train")
    return [f"Val checkpoint: seed **{seed_val}** → best "
            f"**{payload['best_score']}** (the winner is the best checkpoint on val). "
            f"{train}: seed **{payload['seed_score']}** → winner "
            f"**{payload.get('best_train_score')}** (raw {payload['seed_raw_score']} → "
            f"{payload['raw_best_score']}). " + parsimony]

def _analysts_section(report: dict[str, Any]) -> list[str]:
    if not report:
        return []
    lines = ["## What the analysts found", "",
             f"The analyst stage ran in {report['rounds']} proposal round(s): "
             f"{report['calls']} optimizer-model call(s) (analysts and merges; "
             f"{report['cached']} more answered from the cache), batches of up to "
             f"{report['group_size']} train items, merges of up to "
             f"{report['merge_batch_size']} lists"
             + (f", at most {report['max_batches']} analyst call(s) per round"
                if report.get("max_batches") else "")
             + (f"; {report['failed_batches']} batch(es) could not be analysed"
                if report.get("failed_batches") else "")
             + (f"; {report['clipped_items']} item trace(s) were clipped to fit an "
                f"analyst's room ({report.get('clipped_chars', 0)} characters left out)"
                if report.get("clipped_items") else "")
             + (f"; {report['context_retries']} call(s) were too long for the model's "
                "context and retried once at half the length"
                if report.get("context_retries") else "")
             + (f"; {report['stopped']} round(s) stopped before the stage finished"
                if report.get("stopped") else "")
             + ". Every round's findings are in journal.jsonl (`analyst_findings`) and each "
             "call in optimizer_trace.jsonl.", ""]
    latest = report.get("latest")
    if latest:
        from refactory_dsh.optim.analysts import finding_line

        lines += [f"Epoch {latest.get('epoch')}, generation {latest.get('generation')} "
                  f"(candidate `{latest.get('candidate')}`, {latest.get('failed_items')} "
                  f"failed and {latest.get('passed_items')} passed item(s)), the top "
                  "findings:", ""]
        found = list(latest.get("findings") or ())
        lines += [_markdown_text(finding_line(rank, entry)) for rank, entry in
                  enumerate(found[:REPORT_ANALYST_FINDINGS], 1)] or ["_No common pattern._"]
        lines.append("")
    return lines

def _portfolio_section(portfolio: dict[str, Any]) -> list[str]:
    if not portfolio:
        return []
    pool = portfolio.get("selection") == "pool"
    checkpoint = portfolio.get("selection") == "checkpoint"
    rows = [[level, entry["proposed"], entry["evaluated"], entry["kept"],
             *([entry.get("in_pool", 0)] if pool else []), entry["evaluations"]]
            for level, entry in portfolio["by_level"].items()]
    lines = ["## Portfolio", "",
             f"`{portfolio['setting']}`: {portfolio['rounds']} round(s) asked for one candidate "
             f"per edit level; the candidates beyond the batch size of {portfolio['batch_size']} "
             f"cost {portfolio['extra_evaluations']} evaluation(s) (train and val runs). "
             + ("Of a round's candidates whose train run succeeded, the one with the highest "
                "train score (ties: the one proposed first) is applied; only it is scored on "
                "val." if checkpoint else
                "The gate judges a round's candidates together, each against the current "
                "candidate as the round starts, and accepts the one with the highest val score "
                "(ties: higher train score, then the one proposed first)"
                + (" as the new incumbent, if it beats the current one strictly" if pool else "")
                + ": a candidate that beat its parent on val but lost to a sibling is marked "
                "shadowed.")
             + (" Under pool selection every judged candidate may also enter the pool of the "
                "best on val: `in pool` is in it now, `left pool` was in it when judged and "
                "was pushed out since, `rejected` never entered it (`below pool` when the "
                "parent picker still breeds from every judged candidate). `accepted` marks a "
                "former incumbent." if pool else ""), "",
             _md_table(["level", "proposed", "evaluated",
                        "accepted as incumbent" if pool else "applied" if checkpoint
                        else "kept by the gate",
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
        "checkpoint": ("`checkpoint`: each round's change is applied unless its graph is broken "
                       "(refused, vetoed, failing the smoke check or its train run); each applied "
                       "candidate is scored on validation once, and the winner is the best of "
                       "these checkpoints."),
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

def _integrity_section(report: dict[str, Any] | None) -> list[str]:
    """The report's "Integrity review" (`optim.integrity.report_lines`); nothing when the
    integrity judge was off."""
    from refactory_dsh.optim.integrity import report_lines

    return report_lines(report)

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


class _ReportMixin:

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
        if self.config.val_gate and self.selection_mode() == "checkpoint":
            # The checkpoints (every applied candidate), the best on val first (the
            # winner; of a val tie, the earlier), then by val score, then population order.
            gate = self._gate()
            accepted = set(self._gate_accepted)
            order = {c.id: index for index, c in enumerate(self.population)}
            ordered = sorted(
                [c for c in self._evaluated() if c.id in accepted and c.id not in self.rejected],
                key=lambda c: (c.id != gate.best_id, -gate.cache.get(c.id, 0.0), order[c.id]))
            return ordered[:limit] if limit else ordered
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
        # Its measured model cost per train item, when the run measured cost: what a request
        # whose objective carries a cost term shows beside each candidate.
        cost = (candidate.measurements or {}).get("cost")
        items = len(self._train_item_records.get(candidate.id) or ())
        if isinstance(cost, (int, float)) and items:
            row["cost_per_item"] = round(float(cost) / items, 6)
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
        candidate). Under ``checkpoint``: ``current`` (the last applied), ``kept``
        (applied, since replaced) or ``not applied`` (`NOT_APPLIED`), else ``not gated``."""
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
            return NOT_APPLIED if self.selection_mode() == "checkpoint" else "rejected"
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
            "repaired_codes": self._repaired_codes(),
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
            **({"analysts": _analysts_report(self._analyses, self.config)}
               if self.config.analysts else {}),
            **({"optimizer_memory": memory_report(self._memory_passes, self._memory)}
               if self._memory_passes else {}),
            "selection": self._selection_report(),
            **({"integrity": self._integrity_report(best)} if self._integrity_on() else {}),
            # Every evaluation that had an item a harness fault failed: the items,
            # those a rerun recovered, and those left out of its score (`_settle_incidents`).
            "harness_incidents": list(self._harness_incidents),
            # Sealed steps that ran without a whole-step memory and process cap, said once
            # (`seal.step_cap`, each step's dispatch record).
            **({"sealed_without_cap": uncapped}
               if (uncapped := uncapped_sealed_steps(self.root / "runs")) else {}),
        }
        if gated and best is not None:
            payload.update(self.best_score_fields(best))
        checkpoint = gated and best is not None and self.selection_mode() == "checkpoint"
        if checkpoint and self.population:
            # Under ``checkpoint`` ``best_score`` is the winner's val checkpoint: the
            # seed's own checkpoint beside it, so the headline compares val with val
            # (and the seed's train score with the winner's, ``best_train_score``).
            seed = self.population[0]
            payload["seed_val"] = self._gate().cache.get(seed.id)
            # With ``train_minibatch`` each epoch samples its own train items: the seed's
            # train score and the winner's are then on different samples.
            epochs = (self._records_epoch.get(seed.id), self._records_epoch.get(best.id))
            if None not in epochs and (self._minibatches.get(epochs[0])
                                       != self._minibatches.get(epochs[1])):
                payload["train_samples_differ"] = True
        self._write("report.json", payload)
        self._write_text("report.md", self._render_report(payload))
        self._write("manifest.json", progress_file.artifact_manifest(self.root))
        self._progress(
            "report",
            best_score=payload["best_score"],
            raw_best_score=payload["raw_best_score"],
            **({key: payload[key] for key in ("seed_val", "best_train_score", "seed_score",
                                               "train_samples_differ") if key in payload}
               if "seed_val" in payload else {}),
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
            # The fraction then only says whether to audit: holdout 0 turns it off.
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
                    "edit_count": (record.get("edit_count") or self._edit_count(candidate)
                                   if candidate.parent else None),
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
            *_headline(payload),
            *_incident_lines(payload.get("harness_incidents") or []),
            *_uncapped_lines(payload.get("sealed_without_cap")),
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
            *_analysts_section(payload.get("analysts") or {}),
            *memory_report_lines(payload.get("optimizer_memory")),
            *_selection_section(payload.get("selection") or {}),
            *_integrity_section(payload.get("integrity")),
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
                 "raw_delta", "changes", "rationale"],
                [
                    [
                        p["id"], p.get("origin"), p.get("parent_score"), p.get("score"),
                        p.get("delta"), p.get("raw_score"), p.get("raw_delta"),
                        _changes_cell(p.get("edit_count")),
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
