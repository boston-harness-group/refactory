"""GEPA's program pool and per-instance Pareto frontier.

Ported from ``gepa`` 0.1.4 (PyPI; gepa-ai/gepa, MIT): ``core/state.py`` (the pool and
its fronts) and ``gepa_utils.py`` (dominance, frontier sampling). Programs are
indices into the pool, as in GEPA; each holds a refactory candidate id.

Sets of program indices are Python ``set[int]`` as in GEPA, so the orders GEPA's
draws index into (``gepa_utils.py:99-108``, built from set iteration) are the same
in this port under CPython.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProgramPool:
    """GEPA's ``GEPAState`` pool (``core/state.py:157-177``), with the instance front only
    (``frontier_type="instance"``, the default, ``api.py:58``).

    ``subscores[k]`` maps an instance key (an opaque val key, or a train item id when the
    Pareto set is a train split) to program ``k``'s score there.
    """

    candidate_ids: list[str] = field(default_factory=list)
    parents: list[list[int | None]] = field(default_factory=list)
    subscores: list[dict[str, float]] = field(default_factory=list)
    #: Round-robin component pointer per program (``core/state.py:539-543``).
    pointers: list[int] = field(default_factory=list)
    #: Rollouts spent before each program was found (``num_metric_calls_by_discovery``).
    discovered_at: list[int] = field(default_factory=list)
    front_score: dict[str, float] = field(default_factory=dict)
    front_programs: dict[str, set[int]] = field(default_factory=dict)

    def add(self, candidate_id: str, parents: list[int | None], scores: dict[str, float],
            discovered_at: int) -> int:
        """``update_state_with_new_program`` (``core/state.py:527-582``): append the
        program; it inherits the highest pointer of its parents (``:539-543``); each
        instance's front takes it when it beats the best there (``>``) and adds it on a tie
        (``==``, ``:551-560``). Programs are never removed."""
        idx = len(self.candidate_ids)
        self.candidate_ids.append(candidate_id)
        self.discovered_at.append(discovered_at)
        self.pointers.append(max((self.pointers[p] for p in parents if p is not None), default=0))
        self.parents.append(list(parents))
        self.subscores.append(dict(scores))
        for key, score in scores.items():
            prev = self.front_score.get(key, float("-inf"))
            if score > prev:
                self.front_score[key] = score
                self.front_programs[key] = {idx}
            elif score == prev:
                self.front_programs[key].add(idx)
        return idx

    def aggregate(self, idx: int) -> float:
        """A program's mean over the instances it was scored on, ``-inf`` with none
        (``core/state.py:438-445``)."""
        scores = self.subscores[idx]
        return sum(scores.values()) / len(scores) if scores else float("-inf")

    def aggregates(self) -> list[float]:
        return [self.aggregate(i) for i in range(len(self.candidate_ids))]

    def best(self) -> int:
        """``GEPAResult.best_idx``: the highest aggregate, the first on a tie
        (``core/result.py:76-78``)."""
        aggs = self.aggregates()
        return max(range(len(aggs)), key=lambda i: aggs[i])

    def index(self, candidate_id: str) -> int | None:
        try:
            return self.candidate_ids.index(candidate_id)
        except ValueError:
            return None

    def ancestors(self, idx: int) -> set[int]:
        """Every ancestor of a program through its parent lists (``proposer/merge.py:69-115``)."""
        found: set[int] = set()
        stack = [p for p in self.parents[idx] if p is not None]
        while stack:
            p = stack.pop()
            if p in found:
                continue
            found.add(p)
            stack.extend(q for q in self.parents[p] if q is not None)
        return found

    def state_dict(self) -> dict[str, Any]:
        """The pool; the fronts are rebuilt from it in program order (``load``), which
        reproduces them exactly, ties included."""
        return {
            "candidate_ids": list(self.candidate_ids),
            "parents": [list(p) for p in self.parents],
            "subscores": [dict(s) for s in self.subscores],
            "pointers": list(self.pointers),
            "discovered_at": list(self.discovered_at),
        }

    @classmethod
    def load(cls, data: dict[str, Any]) -> "ProgramPool":
        pool = cls()
        pointers = list(data.get("pointers") or [])
        for cid, parents, scores, found in zip(
                data.get("candidate_ids") or [], data.get("parents") or [],
                data.get("subscores") or [], data.get("discovered_at") or []):
            pool.add(str(cid), list(parents), {str(k): float(v) for k, v in scores.items()},
                     int(found))
        # The pointers advanced after each program was added (round robin, ``state.py:539``).
        pool.pointers = [int(p) for p in pointers[:len(pool.candidate_ids)]]
        return pool


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


def select_from_pareto_front(fronts: dict[str, set[int]], scores: list[float],
                             rng: random.Random) -> int:
    """``ParetoCandidateSelector`` (``strategies/candidate_selector.py:18-24``,
    ``gepa_utils.py:99-116``): a non-dominated program, drawn with probability proportional
    to the number of instances it is (tied-)best on; one ``rng.choice``."""
    new_fronts = remove_dominated_programs(fronts, scores)
    freq: dict[int, int] = {}
    for front in new_fronts.values():
        for p in front:
            freq[p] = freq.get(p, 0) + 1
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
