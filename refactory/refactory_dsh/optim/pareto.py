"""The per-item Pareto frontier: GEPA's parent selection, as a core building block.

Ported from ``gepa`` 0.1.4 (PyPI; gepa-ai/gepa, MIT): the instance front of
``core/state.py`` (``update_state_with_new_program``) and the dominance and sampling
rules of ``gepa_utils.py``. Programs are indices into a list of candidates; each
instance key (an item) keeps the set of programs with the best score there.

Two users, one copy: the ``pareto`` parent picker
(:class:`refactory_dsh.optim.policies.ParetoSelector`), which builds the frontier from
the per-item val scores the loop hands every picker
(``ProposalContext.val_item_scores``), and the GEPA strategy
(``refactory_packages/gepa``), which keeps its own pool and passes its one shared
generator, so its draws are GEPA's.

Sets of program indices are Python ``set[int]`` as in GEPA, so the orders GEPA's draws
index into (``gepa_utils.py:99-108``, built from set iteration) are the same here under
CPython.
"""

from __future__ import annotations

import random
from typing import Mapping, Sequence


def add_to_front(front_score: dict[str, float], front_programs: dict[str, set[int]],
                 idx: int, scores: Mapping[str, float]) -> None:
    """Add program ``idx`` with its per-item ``scores`` to the fronts, in place
    (``core/state.py:551-560``): an item's front becomes ``{idx}`` when it beats the best
    there (``>``) and takes it in on a tie (``==``)."""
    for key, score in scores.items():
        prev = front_score.get(key, float("-inf"))
        if score > prev:
            front_score[key] = score
            front_programs[key] = {idx}
        elif score == prev:
            front_programs[key].add(idx)


def build_fronts(subscores: Sequence[Mapping[str, float]]) -> dict[str, set[int]]:
    """The fronts of programs ``0..len(subscores)-1`` added in that order."""
    front_score: dict[str, float] = {}
    front_programs: dict[str, set[int]] = {}
    for idx, scores in enumerate(subscores):
        add_to_front(front_score, front_programs, idx, scores)
    return front_programs


def aggregate(scores: Mapping[str, float]) -> float:
    """A program's mean over the items it was scored on, ``-inf`` with none
    (``core/state.py:438-445``)."""
    return sum(scores.values()) / len(scores) if scores else float("-inf")


def is_dominated(y: int, programs: set[int], fronts: dict[str, set[int]]) -> bool:
    """``gepa_utils.py:23-34``: ``y`` is dominated when every front holding it also holds
    one of ``programs``."""
    for front in fronts.values():
        if y in front and not any(other in front for other in programs):
            return False
    return True


def remove_dominated_programs(fronts: dict[str, set[int]], scores: list[float]) -> dict[str, set[int]]:
    """``gepa_utils.py:37-75``: drop dominated programs, testing the lowest aggregate first,
    until a full pass removes none; the fronts without them."""
    freq: dict[int, int] = {}
    for front in fronts.values():
        for p in front:
            freq[p] = freq.get(p, 0) + 1
    programs = sorted(freq.keys(), key=lambda p: scores[p])
    dominated: set[int] = set()
    found = True
    while found:
        found = False
        for y in programs:
            if y in dominated:
                continue
            others = set(programs) - {y} - dominated
            if is_dominated(y, others, fronts):
                dominated.add(y)
                found = True
                break
    return {key: {p for p in front if p not in dominated} for key, front in fronts.items()}


def front_wins(fronts: dict[str, set[int]], scores: list[float]) -> dict[int, int]:
    """Each non-dominated program with the number of items it is (tied-)best on, in the
    order GEPA's draw lists them (``gepa_utils.py:99-108``)."""
    freq: dict[int, int] = {}
    for front in remove_dominated_programs(fronts, scores).values():
        for p in front:
            freq[p] = freq.get(p, 0) + 1
    return freq


def select_from_pareto_front(fronts: dict[str, set[int]], scores: list[float],
                             rng: random.Random) -> int:
    """``ParetoCandidateSelector`` (``strategies/candidate_selector.py:18-24``,
    ``gepa_utils.py:99-116``): a non-dominated program, drawn with probability proportional
    to the number of instances it is (tied-)best on; one ``rng.choice``."""
    freq = front_wins(fronts, scores)
    sampling: list[int] = [p for p, n in freq.items() for _ in range(n)]
    return rng.choice(sampling)


def find_dominator_programs(fronts: dict[str, set[int]], scores: list[float]) -> list[int]:
    """``gepa_utils.py:78-87``: every program left on a front once the dominated are gone,
    as ``list(set)``."""
    new_fronts = remove_dominated_programs(fronts, scores)
    union: set[int] = set()
    for front in new_fronts.values():
        union |= front
    return list(union)
