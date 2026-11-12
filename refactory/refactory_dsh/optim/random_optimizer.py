"""The baselines: a random edit, and a model that reads the gradient.

Both implement the same contract, ``propose`` returning typed edits, so the
engine and the validating loop are identical for either. The difference is only
where the ideas come from, which is what makes the two comparable.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import replace
from typing import Any, Sequence

from refactory_dsh.nn import edits as nn_edits

from .base import (
    READER_CONCURRENCY, Candidate, Optimizer, Proposal, ProposalContext, considered_text,
    trajectory_groups, unreadable_proposal,
)

def text_only(model: Any) -> Any:
    """``model`` with ``tool_policy="none"``, when the client supports one.

    Every optimizer call -- the proposal and each reader -- is a prompt in and JSON
    out. On an agentic route (``claude_code``) the model would otherwise get tools in a
    working directory next to the run's records. The run's own client is left alone.
    """
    with_policy = getattr(model, "with_tool_policy", None)
    return with_policy("none") if callable(with_policy) else model


#: A reader's role: go through one group of whole trajectories and report patterns.
READER_SYSTEM = (
    "You read an AI pipeline's trajectories on evaluation items and report what you find. "
    "You reply with JSON only."
)
READER_ANSWER = """Write one JSON object and nothing else:

{"failure_patterns": [{"pattern": "<what goes wrong, in one sentence>",
                       "step": "<the step it starts in, or \"unclear\">",
                       "items": ["<item id>", ...],
                       "evidence": ["<a short verbatim quote from a trajectory>", ...]}],
 "success_patterns": [{"pattern": "<what the passing items do right>", "step": "...",
                       "items": [...], "evidence": [...]}]}

Rules:
- Every pattern names the items it covers and quotes them; no pattern without evidence.
- Cover every item answered wrong; group items that fail the same way.
- Do not propose changes: say what happens, not what to do.
"""


def _elite(parents: Sequence[Candidate], ctx: ProposalContext) -> Sequence[Candidate]:
    """The top ``ctx.elite``: exactly what the loop passed before the pool widened.

    The loop hands a proposer the whole evaluated population now, so a policy-based
    optimizer can select for itself. An optimizer with no policy of its own still
    wants the elite, and this reproduces `Loop.leaderboard` rather than approximating
    it: the same filter (a candidate with no score is not a parent), the same
    ordering (score descending, **id as the tie-break**), and the same fallback when
    nothing has scored yet. Getting any of those subtly wrong changes which parents
    are bred from, which changes the search, which is how a duplicate stopped being
    refused one layer earlier than it should have been.
    """
    val = dict(getattr(ctx, "val_scores", None) or {})
    if val:
        # Under the validation gate the elite is the best on val, a tie by the higher
        # train score, then the order handed over (the loop's pool order). A candidate
        # whose train run failed still has a val score, so the incumbent stays a
        # parent even then.
        ranked = sorted(
            (candidate for candidate in parents if candidate.id in val),
            key=lambda candidate: (-val[candidate.id],
                                   -(candidate.score if candidate.score is not None
                                     else float("-inf"))),
        )
        if ranked:
            return ranked[: max(1, ctx.elite)]
    ranked = sorted(
        (candidate for candidate in parents if candidate.score is not None),
        key=lambda candidate: (-(candidate.score or 0.0), candidate.id),
    )
    if not ranked:
        # `Loop.select_parents` falls back to the population, in its own order.
        return list(parents)[: max(1, ctx.elite)]
    return ranked[: max(1, ctx.elite)]


class RandomOptimizer(Optimizer):
    """Sample edits uniformly from the enabled catalogue.

    Structural edits need blocks to insert, so a mode that declares none is only
    reachable through its parameters. The sampler says so by returning fewer
    proposals rather than by inventing an operation.
    """

    name = "random"

    def __init__(self, *, k: int = 3, seed: int = 0) -> None:
        super().__init__(k=k)
        self.seed = seed

    def rng(self, ctx: ProposalContext, n: int) -> random.Random:
        """A stream derived from the seed and the generation, not carried forward.

        Derived rather than accumulated so a proposal sequence is a pure function of
        the run's configuration and how far it has got. That is what lets a run be
        resumed and reproduce what an uninterrupted run would have done, and it costs
        nothing: a carried stream has no value beyond being reproducible.
        """
        return random.Random(f"{self.seed}:{ctx.generation}:{ctx.epoch}:{n}")

    def propose(
        self, ctx: ProposalContext, n: int, parents: Sequence[Candidate]
    ) -> list[Proposal]:
        if not parents:
            return []
        # `parents` is the evaluated population since issue #32, so that a policy-based
        # optimizer can select from it. This one has no parent policy: it takes the
        # elite the loop used to hand it, which keeps its proposals what they always
        # were and stops widening the pool from silently changing every run.
        parents = _elite(parents, ctx)
        rng = self.rng(ctx, n)
        preferred = [name for name in ctx.preferred_edits if name in set(nn_edits.enabled(ctx.preset))]
        proposals: list[Proposal] = []
        for index in range(max(1, n)):
            parent = parents[index % len(parents)]
            edit = None
            if preferred and index % 2 == 0:
                # Every other proposal goes to what the author suggested. A prior,
                # not a restriction: the sampler still covers the catalogue.
                edit = self._preferred_edit(rng.choice(preferred), parent, ctx, rng)
            if edit is None:
                edit = nn_edits.random_edit(
                    parent.state,
                    preset=ctx.preset,
                    insertable=ctx.insertable,
                    rng=rng,
                )
            if edit is None:
                continue
            proposals.append(
                Proposal(edit=edit, parent=parent.id, rationale="sampled at random")
            )
        return proposals


    def _preferred_edit(
        self, op: str, parent: Candidate, ctx: ProposalContext, rng: random.Random
    ) -> dict[str, Any] | None:
        """A random edit restricted to one operation, or ``None`` if none fits."""
        for _ in range(20):
            edit = nn_edits.random_edit(
                parent.state,
                preset="broad",
                insertable=ctx.insertable,
                rng=rng,
            )
            if edit is not None and edit.get("op") == op:
                return edit
        return None


def checked_patterns(payload: Any, items: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    """A reader's patterns with every cited item id checked against the group it read.

    A reader sees only its own group, so an id outside it is a slip (a val id, a
    misremembered one) and must not reach the proposal call as evidence: it is dropped
    from ``items`` and kept under ``not_in_group``, which the request says.
    """
    group = {str(item) for item in items}
    out: dict[str, list[dict[str, Any]]] = {}
    for kind in ("failure_patterns", "success_patterns"):
        patterns: list[dict[str, Any]] = []
        for pattern in (payload.get(kind) if isinstance(payload, dict) else None) or ():
            if not isinstance(pattern, dict):
                continue
            cited = [str(i) for i in pattern.get("items") or ()]
            patterns.append({**pattern, "items": [i for i in cited if i in group],
                             "not_in_group": [i for i in cited if i not in group]})
        out[kind] = patterns
    return out


class AutoresearchOptimizer(Optimizer):
    """Ask a model to read the traces and propose the next update.

    The prompt carries the write surface, the edit catalogue, the insertable
    blocks, the leaderboard and the evidence, and — on a retry — the reasons the
    previous attempt was refused. Telling the model exactly what was wrong is what
    makes the loop converge in one or two attempts instead of five.
    """

    name = "autoresearch"
    uses_model = True

    def __init__(self, client: Any, *, k: int = 3, system: str = "") -> None:
        super().__init__(k=k)
        if client is None:
            raise ValueError("the autoresearch optimizer needs a model client")
        self._client = text_only(client)
        self._completion_options: dict[str, Any] = {}
        self._system = system or (
            "You are the optimizer of a search over workflows. You reply with JSON only."
        )
        #: The answer's ``lessons`` (when the request asked for them), for the loop to
        #: keep and show back; empty when the answer carried none.
        self.last_lessons: str = ""
        #: This round's analysis (``ProposalContext.analysts`` on; `_analyse`), for the
        #: loop to journal; ``None`` when no analyst ran.
        self.last_analysis: dict[str, Any] | None = None

    def bind_model(self, model: Any, *, workdir: Any = None) -> None:
        import inspect
        self._client = text_only(model)
        # Keep the original minimal ModelClient protocol compatible. DSH supports
        # an owned workspace, so its proposer receipts belong to this search.
        if workdir is not None and "workdir" in inspect.signature(model.complete).parameters:
            self._completion_options = {"workdir": workdir}

    def _read(self, ctx: ProposalContext, groups: list[list[tuple[str, str]]]
              ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Readers go through the item groups, concurrently: their findings, and the
        groups that could not be read. A group read before (a retry after a refusal)
        is not read again."""
        from concurrent.futures import ThreadPoolExecutor
        import hashlib

        cache: dict[str, dict[str, Any]] = getattr(self, "_reader_cache", {})
        self._reader_cache = cache
        # Kept in the run directory too, so a resumed run reuses what it paid for. The
        # key is the group's whole text: the edited candidate's records for exactly
        # these items.
        from pathlib import Path
        store = (Path(self.trace_path).parent / "reader_findings"
                 if self.trace_path is not None else None)
        steps = ", ".join(f"`{n.get('id')}`" for n in ctx.nodes) or "(not listed)"

        def read(index: int, group: list[tuple[str, str]]) -> tuple[dict | None, dict | None]:
            items = [item for item, _block in group]
            body = "\n\n".join(block for _item, block in group)
            key = hashlib.sha256(body.encode()).hexdigest()
            if key in cache:
                return cache[key], None
            if store is not None and (store / f"{key}.json").exists():
                try:
                    saved = json.loads((store / f"{key}.json").read_text())
                    cache[key] = {**saved, **checked_patterns(saved, items), "group": index}
                    return cache[key], None
                except (OSError, json.JSONDecodeError):
                    pass
            prompt = (f"# Reading trajectories -- group {index} of {len(groups)}\n\n"
                      f"Mode: `{ctx.mode_name}`. The pipeline's steps: {steps}.\n\n"
                      f"Below are {len(items)} train item(s) of the candidate being edited, "
                      "each in full: its input, every step's output in the order they ran, "
                      "the expected answer, the prediction and the grader's reason.\n\n"
                      f"{body}\n\n## Answer\n{READER_ANSWER}")
            call = {"kind": "reader", "epoch": ctx.epoch, "generation": ctx.generation,
                    "group": index, "items": items, "system": READER_SYSTEM, "prompt": prompt}
            try:
                reply = self._client.complete(prompt, system=READER_SYSTEM,
                                              **self._completion_options)
                payload = extract_json(reply)
            except Exception as exc:
                self.record_call(**call, reply=locals().get("reply"), error=f"{type(exc).__name__}: {exc}")
                return None, {"group": index, "items": items, "error": f"{type(exc).__name__}: {exc}"}
            found = {"group": index, "items": items, **checked_patterns(payload, items)}
            self.record_call(**call, reply=reply, error=None,
                             patterns=len(found["failure_patterns"]) + len(found["success_patterns"]))
            cache[key] = found
            if store is not None:
                try:
                    store.mkdir(parents=True, exist_ok=True)
                    (store / f"{key}.json").write_text(json.dumps(found, default=str))
                except OSError:
                    pass
            return found, None

        with ThreadPoolExecutor(max_workers=max(1, min(READER_CONCURRENCY, len(groups)))) as pool:
            results = list(pool.map(lambda pair: read(*pair), enumerate(groups, 1)))
        findings = [found for found, _failed in results if found is not None]
        failures = [failed for _found, failed in results if failed is not None]
        return findings, failures

    def _review_steps(self, ctx: ProposalContext) -> list[dict[str, Any]]:
        """Each added step (``ctx.added_steps``) with a reviewer's verdict on whether its
        output holds work its inputs lacked: one call per step that recorded text output,
        up to ``STEP_REVIEW_STEPS``, concurrently, each recorded (kind ``step_review``)
        and cached like the readers' findings, in memory and in the run directory, so a
        retry or a resumed run pays for none again. A failed call is recorded and the
        step is shown with ``review_error``; it never fails the proposal."""
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path

        from . import feedback

        cache: dict[str, dict[str, Any]] = self.__dict__.setdefault("_review_cache", {})
        store = (Path(self.trace_path).parent / "step_reviews"
                 if self.trace_path is not None else None)
        reviewable = [index for index, entry in enumerate(ctx.added_steps)
                      if entry.get("samples")][: feedback.STEP_REVIEW_STEPS]

        def review(index: int) -> tuple[int, dict[str, Any]]:
            entry = ctx.added_steps[index]
            prompt = feedback.review_prompt(entry, ctx.mode_name)
            key = feedback.review_key(prompt)
            if key in cache:
                return index, {"review": cache[key]}
            if store is not None and (store / f"{key}.json").exists():
                try:
                    saved = feedback.stored_review((store / f"{key}.json").read_text())
                except OSError:
                    saved = None
                if saved is not None:
                    cache[key] = saved
                    return index, {"review": saved}
            call = {"kind": "step_review", "epoch": ctx.epoch, "generation": ctx.generation,
                    "step": entry.get("step"), "candidate": entry.get("candidate"),
                    "items": [s.get("item") for s in entry.get("samples") or ()],
                    "system": feedback.STEP_REVIEW_SYSTEM, "prompt": prompt}
            reply = None
            try:
                reply = self._client.complete(prompt, system=feedback.STEP_REVIEW_SYSTEM,
                                              **self._completion_options)
                found = feedback.parse_review(extract_json(reply))
            except Exception as exc:  # a failed review is shown as such, never fatal
                from refactory_dsh.budgets import BudgetExhausted

                if isinstance(exc, BudgetExhausted):
                    raise
                error = f"{type(exc).__name__}: {exc}"
                self.record_call(**call, reply=reply, error=error)
                return index, {"review_error": error}
            self.record_call(**call, reply=reply, error=None, verdict=found["verdict"])
            cache[key] = found
            if store is not None:
                try:
                    store.mkdir(parents=True, exist_ok=True)
                    (store / f"{key}.json").write_text(json.dumps(found))
                except OSError:
                    pass
            return index, {"review": found}

        reviewed: dict[int, dict[str, Any]] = {}
        if reviewable:
            with ThreadPoolExecutor(max_workers=max(1, min(READER_CONCURRENCY,
                                                           len(reviewable)))) as pool:
                reviewed = dict(pool.map(review, reviewable))
        return [{**entry, **reviewed.get(index, {})}
                for index, entry in enumerate(ctx.added_steps)]

    def _model_id(self) -> str:
        """The optimizer route's model, for the analyst caches' keys: an answer from
        one model is not reused for another."""
        route = getattr(self._client, "route", None)
        try:
            found = route() if callable(route) else {}
        except Exception:  # a client that cannot say is keyed by its type alone
            found = {}
        found = found if isinstance(found, dict) else {}
        return json.dumps({key: found.get(key) for key in
                           ("worker_backend", "provider", "model", "effort")} | {
                              "client": type(self._client).__name__}, sort_keys=True)

    def _ask(self, ctx: ProposalContext, system: str, prompt: str, fields: dict[str, Any],
             parse: Any, counts: dict[str, int]) -> tuple[Any, str | None]:
        """One analyst-stage call (`_analyse`): ``parse(payload)`` of the reply, or
        ``None`` and why. Cached by the optimizer's model and the call's whole text, in
        memory and in the run directory (``analyst_findings/``); a call is recorded in
        the optimizer trace with what it found. At most ``analysts.CONCURRENCY`` calls
        of a stage are in flight at once, merges included. A spent budget, a cancelled
        run or a missing model choice stop the proposal (as the SkillOpt port's
        ``ask``); any other failure is recorded and returned, never raised."""
        import hashlib
        import threading
        from concurrent.futures import CancelledError
        from pathlib import Path

        from refactory_dsh.budgets import BudgetExhausted
        from refactory_dsh.model import ModelChoiceError

        from . import analysts

        cache: dict[str, str] = self.__dict__.setdefault("_analyst_cache", {})
        lock = self.__dict__.setdefault("_analyst_lock", threading.Lock())
        gate = self.__dict__.setdefault("_analyst_gate",
                                        threading.BoundedSemaphore(analysts.CONCURRENCY))
        store = (Path(self.trace_path).parent / "analyst_findings"
                 if self.trace_path is not None else None)
        key = hashlib.sha256(f"{self._model_id()}\n{system}\n{prompt}".encode()).hexdigest()
        reply = cache.get(key)
        if reply is None and store is not None and (store / f"{key}.json").exists():
            try:
                reply = json.loads((store / f"{key}.json").read_text()).get("reply")
            except (OSError, json.JSONDecodeError, AttributeError):
                reply = None
        if isinstance(reply, str):
            found = parse(extract_json(reply))
            if found is not None:
                with lock:
                    counts["cached"] += 1
                cache[key] = reply
                return found, None
        call = {**fields, "epoch": ctx.epoch, "generation": ctx.generation,
                "system": system, "prompt": prompt}
        reply = None
        with lock:
            counts["calls"] += 1
        try:
            with gate:
                reply = self._client.complete(prompt, system=system, **self._completion_options)
            found = parse(extract_json(reply))
            if found is None:
                raise ValueError('the answer has no "findings" list')
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, (BudgetExhausted, CancelledError, ModelChoiceError)):
                self.record_call(**call, reply=reply, error=error)
                raise
            self.record_call(**call, reply=reply, error=error)
            return None, error
        self.record_call(**call, reply=reply, error=None, findings=found)
        cache[key] = reply
        if store is not None:
            try:
                store.mkdir(parents=True, exist_ok=True)
                (store / f"{key}.json").write_text(json.dumps({"reply": reply}))
            except OSError:
                pass
        return found, None

    def _round_analysis(self, ctx: ProposalContext) -> dict[str, Any] | None:
        """The analysis of this proposal step, made once: keyed by the epoch, the
        generation, the edited candidate, its train records, the stage's settings and
        the optimizer's model (`analysts.round_key`), never by the lessons, so a retry
        in the same step reuses it (the lessons it saw are those of the step's first
        attempt) and pays for no call. Kept in memory and in
        ``analyst_findings/round-<key>.json``, so a resumed run reuses it too."""
        from pathlib import Path

        from . import analysts

        key = analysts.round_key(ctx, self._model_id())
        held = self.__dict__.get("_held_analysis")
        if held is not None and held[0] == key:
            return held[1]
        store = (Path(self.trace_path).parent / "analyst_findings" / f"round-{key}.json"
                 if self.trace_path is not None else None)
        if store is not None and store.exists():
            try:
                saved = json.loads(store.read_text())
            except (OSError, json.JSONDecodeError):
                saved = None
            if isinstance(saved, dict) and saved.get("key") == key:
                # Made and journaled by an earlier process: nothing was paid now.
                analysis = {**saved, "cached": saved.get("cached", 0) + saved.get("calls", 0),
                            "calls": 0}
                self._held_analysis = (key, analysis)
                return analysis
        analysis = self._analyse(ctx, key)
        if analysis is not None:
            self._held_analysis = (key, analysis)
            failed = (any(b.get("error") for b in analysis["batches"])
                      or any(m.get("error") for m in analysis["merges"]))
            # A step with a failed call is kept for its retries, not for a resumed run,
            # which tries the failed calls again (the others from their own cache).
            if store is not None and not failed:
                try:
                    store.parent.mkdir(parents=True, exist_ok=True)
                    store.write_text(json.dumps(analysis, default=str))
                except OSError:
                    pass
        return analysis

    def _analyse(self, ctx: ProposalContext, key: str = "") -> dict[str, Any] | None:
        """The analyst stage (`optim.analysts`, ``ctx.analysts``): failure and success
        analysts over batches of the edited candidate's train items, concurrently, then
        the failure-first merge into one ranked list. ``None`` without items. When a
        call stops the stage (a spent budget), ``last_analysis`` holds what it made and
        paid so far, with the ``error``, for the loop to journal, and the error is
        raised."""
        from concurrent.futures import ThreadPoolExecutor

        from . import analysts

        conf = analysts.settings(**{k: v for k, v in ctx.analysts.items()
                                    if k in ("group_size", "merge_batch_size", "max_batches")})
        failures, successes = analysts.split(ctx.item_records, int(ctx.epoch))
        if not failures and not successes:
            return None
        plan, left = analysts.batches(failures, successes, group_size=conf["group_size"],
                                      max_batches=conf["max_batches"])
        pipeline = analysts.pipeline_text(ctx)
        totals = {kind: sum(1 for k, _i, _g in plan if k == kind) for kind in analysts.KINDS}
        analysed = {str(r.get("id")) for _k, _i, group in plan for r in group}
        counts = {"calls": 0, "cached": 0}
        merges: list[dict[str, Any]] = []
        done: list[dict[str, Any]] = []

        def made(findings: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
            return {"key": key, "epoch": ctx.epoch, "generation": ctx.generation,
                    "candidate": ctx.parent, **conf,
                    "batches": [{k: v for k, v in b.items() if k != "found"} for b in done],
                    "merges": sorted(merges, key=lambda m: (m["step"], m["level"], m["batch"])),
                    "not_analysed": left, "findings": list(findings), **counts, **extra}

        def analyse(batch: tuple[str, int, list[dict[str, Any]]]) -> dict[str, Any]:
            kind, index, group = batch
            items = [str(r.get("id")) for r in group]
            found, error = self._ask(
                ctx, analysts.FAILURE_SYSTEM if kind == "failure" else analysts.SUCCESS_SYSTEM,
                analysts.analyst_prompt(ctx, kind, index, totals[kind], group, pipeline),
                {"kind": f"analyst_{kind}", "group": index, "items": items},
                lambda payload: analysts.parse_findings(payload, kind, items), counts)
            return {"kind": kind, "index": index, "items": items, "found": found or [],
                    "findings": len(found or ()), "clipped": analysts.clipped_items(group),
                    "error": error}

        def merge_lists(step: str, lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
            """SkillOpt's hierarchical merge (``aggregate.py``): lists without findings
            are dropped, one list is not merged, and each level merges batches of
            ``merge_batch_size``; an unreadable merge keeps every input finding."""
            current = [found for found in lists if found]
            level = 0
            while len(current) > 1:
                level += 1
                size = conf["merge_batch_size"]
                chunks = [current[i:i + size] for i in range(0, len(current), size)]

                def merge(numbered: tuple[int, list[list[dict[str, Any]]]],
                          level: int = level) -> list[dict[str, Any]]:
                    number, chunk = numbered
                    if len(chunk) == 1:
                        return chunk[0]
                    found, error = self._ask(
                        ctx, analysts.merge_system(step),
                        analysts.merge_prompt(ctx, step, chunk, level, pipeline),
                        {"kind": f"analyst_merge_{step}", "level": level, "batch": number,
                         "lists": len(chunk)},
                        lambda payload: analysts.parse_findings(payload, step, sorted(analysed),
                                                                limit=None), counts)
                    merges.append({"step": step, "level": level, "batch": number,
                                   "lists": len(chunk), "fallback": found is None,
                                   "error": error})
                    return found if found is not None else [f for part in chunk for f in part]

                with ThreadPoolExecutor(max_workers=max(1, min(analysts.CONCURRENCY,
                                                               len(chunks)))) as pool:
                    current = list(pool.map(merge, enumerate(chunks, 1)))
            return current[0] if current else []

        try:
            with ThreadPoolExecutor(max_workers=max(1, min(analysts.CONCURRENCY,
                                                           len(plan)))) as pool:
                done[:] = list(pool.map(analyse, plan))
            with ThreadPoolExecutor(max_workers=2) as pool:
                failure_side, success_side = pool.map(
                    lambda kind: merge_lists(kind, [b["found"] for b in done
                                                    if b["kind"] == kind]),
                    analysts.KINDS)
            if failure_side and success_side:
                found, error = self._ask(
                    ctx, analysts.merge_system("final"),
                    analysts.merge_prompt(ctx, "final", [failure_side, success_side], 0,
                                          pipeline),
                    {"kind": "analyst_merge_final", "level": 0, "batch": 1, "lists": 2},
                    lambda payload: analysts.parse_findings(payload, "failure",
                                                            sorted(analysed), limit=None),
                    counts)
                merges.append({"step": "final", "level": 0, "batch": 1, "lists": 2,
                               "fallback": found is None, "error": error})
                # SkillOpt's fallback: the failure findings first, then the success ones.
                ranked = found if found is not None else [*failure_side, *success_side]
            else:
                ranked = failure_side or success_side
        except BaseException as exc:
            # What was made and paid before the stop, for the loop to journal.
            self.last_analysis = made([], error=f"{type(exc).__name__}: {exc}")
            raise
        return made(ranked)

    def propose(
        self, ctx: ProposalContext, n: int, parents: Sequence[Candidate]
    ) -> list[Proposal]:
        self.last_lessons = ""
        self.last_analysis = None
        if ctx.added_steps:
            ctx = replace(ctx, added_steps=self._review_steps(ctx))
        if ctx.analysts and ctx.item_records:
            # The analysts read every item in full and their ranked findings take the
            # readers' place, as SkillOpt's analysts are its only reading of the traces.
            self.last_analysis = self._round_analysis(ctx)
            ctx = replace(ctx, analysis=self.last_analysis)
        groups = trajectory_groups(ctx.item_records, ctx.evidence_budget) if ctx.item_records else []
        if len(groups) > 1 and ctx.analysis is None:
            findings, failures = self._read(ctx, groups)
            ctx = replace(ctx, reader_findings=findings, reader_failures=failures)
        ctx = replace(ctx, proposals=max(1, n))
        question = (
            f"{ctx.render()}\n\nPropose at most {max(1, n)} update(s). "
            "Reply with the JSON object described under 'Answer' and nothing else."
        )
        call = {"kind": "proposal", "epoch": ctx.epoch, "generation": ctx.generation,
                "system": self._system, "prompt": question}
        try:
            text = self._client.complete(question, system=self._system, **self._completion_options)
        except Exception as exc:
            self.record_call(**call, reply=None, proposals=0, error=f"{type(exc).__name__}: {exc}")
            raise
        try:
            payload = extract_json(text)
        except Exception as exc:
            self.record_call(**call, reply=text, proposals=0, error=f"{type(exc).__name__}: {exc}")
            raise
        self.last_diagnosis = str(payload.get("diagnosis", "") or "")
        self.last_considered = considered_text(payload.get("considered"))
        if ctx.ask_lessons:
            from .feedback import lessons_text
            self.last_lessons = lessons_text(payload.get("lessons"))
        parent_ids = {parent.id for parent in parents}
        # A proposal that names no parent belongs to the pipeline the request showed
        # ("The pipeline you may edit" is ``ctx.parent``'s). Left unset, the loop
        # applied it to the first parent it was handed, which without the validation
        # gate is the seed: edits written for the leader landed on the seed.
        described = ctx.parent if ctx.parent in parent_ids else None

        proposals: list[Proposal] = []
        # An entry with no usable edit is refused, not dropped: the loop counts it and
        # the next request says why. Listed first, so a batch filled by the readable
        # ones cannot skip it.
        unreadable: list[Proposal] = []
        listed, missing = _answered_proposals(payload)
        if missing:
            unreadable.append(unreadable_proposal(payload, missing))
        for raw in listed:
            steps, why = _proposal_edits(raw)
            if not steps:
                unreadable.append(unreadable_proposal(raw, why))
                continue
            parent = raw.get("parent")
            proposals.append(
                Proposal(
                    edits=tuple(steps),
                    rationale=str(raw.get("rationale", "") or ""),
                    prediction=str(raw.get("prediction", "") or ""),
                    confidence=_confidence(raw.get("confidence")),
                    parent=parent if parent in parent_ids else described,
                )
            )
        proposals = [*unreadable, *proposals[: max(1, n)]]
        self.record_call(**call, reply=text, proposals=len(proposals) - len(unreadable),
                         unreadable=len(unreadable),
                         diagnosis=self.last_diagnosis, considered=self.last_considered,
                         error=None)
        return proposals


def _parse_edit(edit: Any) -> dict[str, Any] | None:
    if isinstance(edit, str):
        try:
            edit = json.loads(edit)
        except json.JSONDecodeError:
            return None
    return edit if isinstance(edit, dict) and "op" in edit else None


#: The keys of an answered proposal that are about the proposal, not its edit.
PROPOSAL_FIELDS = ("rationale", "prediction", "confidence", "parent")
#: The keys that hold an answered proposal's edit (``_proposal_edits``).
EDIT_KEYS = ("edit", "edits", "edit_json")


def _answered_proposals(payload: dict[str, Any]) -> tuple[list[Any], str]:
    """The proposal entries of an answer, and why there are none when it has no
    ``proposals`` list: its ``proposals`` (one object counts as one entry), else a
    single ``proposal``, else the answer itself when it is one proposal (an edit
    written at its top level, or an ``edit``/``edits`` beside the diagnosis). An empty
    ``proposals`` list is an answer that proposes nothing, not a malformed one."""
    if "proposals" in payload or "proposal" in payload:
        listed = payload["proposals"] if "proposals" in payload else payload["proposal"]
        if not listed:
            return [], ""
        return (list(listed) if isinstance(listed, (list, tuple)) else [listed]), ""
    if "op" in payload or any(key in payload for key in EDIT_KEYS):
        return [payload], ""
    return [], 'the answer has no "proposals" list'


def _op_fields(op: Any) -> tuple[str, ...] | None:
    """The fields the catalogue's ``op`` takes, or ``None`` for an op it does not have."""
    for entry in nn_edits.catalogue("broad"):
        if entry["op"] == op:
            return tuple(entry["fields"])
    return None


def _proposal_edits(raw: Any) -> tuple[list[dict[str, Any]], str]:
    """The edits one answered proposal carries, in order, or none and why: its
    ``edits`` list (a compound proposal), else its single ``edit``, else -- an edit
    written at the proposal's top level (``{"op": ..., "rationale": ...}``) -- the
    fields its op takes (``_op_fields``). An entry that is not an edit
    drops the whole proposal: applying the rest would score something the model did
    not propose. More edits than the run allows are kept, so the loop refuses them
    with the reason (``E-too-many-edits``) the next request shows."""
    if not isinstance(raw, dict):
        return [], f"a proposal is a JSON object, not {type(raw).__name__}"
    listed = raw.get("edits")
    if isinstance(listed, list) and listed:
        steps = [_parse_edit(edit) for edit in listed]
        bad = [index for index, step in enumerate(steps, 1) if step is None]
        if bad:
            return [], (f"entry {bad[0]} of its {len(steps)} edits is not an edit object "
                        "naming its op")
        return steps, ""  # type: ignore[return-value]
    if "op" in raw and not any(key in raw for key in EDIT_KEYS):
        # Only what the op takes: a proposal-level key ("id", "diagnosis") left in
        # would be read as the edit's own (an "id" naming a node to remove). An op the
        # catalogue lacks keeps its fields less the proposal's, for E-op to name.
        fields = _op_fields(raw["op"])
        return [{key: value for key, value in raw.items()
                 if (key == "op" or key in fields if fields is not None
                     else key not in PROPOSAL_FIELDS)}], ""
    single = _parse_edit(raw.get("edit") or raw.get("edit_json"))
    if single is not None:
        return [single], ""
    if any(key in raw for key in EDIT_KEYS):
        return [], "its edit is not an edit object naming its op"
    return [], "it carries no edit (no \"edit\", \"edits\" or \"op\")"


def _confidence(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def extract_json(text: str) -> dict[str, Any]:
    """The first JSON object in ``text``, fenced or bare.

    A model's answer is prose with an object in it often enough that refusing
    anything but a bare object would waste most of the tokens it just spent.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("the model returned nothing")
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    for chunk in [*fenced, text]:
        found = _first_object(chunk)
        if found is not None:
            return found
    raise ValueError(f"no JSON object in the model's answer: {text[:400]!r}")


def _first_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        loaded = json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(loaded, dict):
                        return loaded
                    break
        start = text.find("{", start + 1)
    return None
