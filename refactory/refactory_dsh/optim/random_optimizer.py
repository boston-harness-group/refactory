"""The baselines: a random edit, and a model that reads the gradient.

Both implement the same contract, ``propose`` returning typed edits, so the
engine and the validating loop are identical for either. The difference is only
where the ideas come from, which is what makes the two comparable.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Sequence

from refactory_dsh.nn import edits as nn_edits

from .base import Candidate, Optimizer, Proposal, ProposalContext


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


class AutoresearchOptimizer(Optimizer):
    """Ask a model to read the traces and propose the next update.

    The prompt carries the write surface, the edit catalogue, the insertable
    blocks, the leaderboard and the evidence, and — on a retry — the reasons the
    previous attempt was refused. Telling the model exactly what was wrong is what
    makes the loop converge in one or two attempts instead of five.
    """

    name = "autoresearch"

    def __init__(self, client: Any, *, k: int = 3, system: str = "") -> None:
        super().__init__(k=k)
        if client is None:
            raise ValueError("the autoresearch optimizer needs a model client")
        self._client = client
        self._completion_options: dict[str, Any] = {}
        self._system = system or (
            "You are the optimizer of a search over workflows. You reply with JSON only."
        )

    def bind_model(self, model: Any, *, workdir: Any = None) -> None:
        import inspect
        self._client = model
        # Keep the original minimal ModelClient protocol compatible. DSH supports
        # an owned workspace, so its proposer receipts belong to this search.
        if workdir is not None and "workdir" in inspect.signature(model.complete).parameters:
            self._completion_options = {"workdir": workdir}

    def propose(
        self, ctx: ProposalContext, n: int, parents: Sequence[Candidate]
    ) -> list[Proposal]:
        question = (
            f"{ctx.render()}\n\nPropose at most {max(1, n)} update(s). "
            "Reply with the JSON object described under 'Answer' and nothing else."
        )
        call = {"epoch": ctx.epoch, "generation": ctx.generation, "system": self._system,
                "prompt": question}
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
        parent_ids = {parent.id for parent in parents}

        proposals: list[Proposal] = []
        for raw in payload.get("proposals") or ():
            if not isinstance(raw, dict):
                continue
            edit = raw.get("edit") or raw.get("edit_json")
            if isinstance(edit, str):
                try:
                    edit = json.loads(edit)
                except json.JSONDecodeError:
                    continue
            if not isinstance(edit, dict) or "op" not in edit:
                continue
            parent = raw.get("parent")
            proposals.append(
                Proposal(
                    edit=edit,
                    rationale=str(raw.get("rationale", "") or ""),
                    prediction=str(raw.get("prediction", "") or ""),
                    confidence=_confidence(raw.get("confidence")),
                    parent=parent if parent in parent_ids else None,
                )
            )
        proposals = proposals[: max(1, n)]
        self.record_call(**call, reply=text, proposals=len(proposals),
                         diagnosis=self.last_diagnosis, error=None)
        return proposals


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
