"""GEPA's program pool.

Ported from ``gepa`` 0.1.4 (PyPI; gepa-ai/gepa, MIT): ``core/state.py`` (the pool).
Programs are indices into the pool, as in GEPA; each holds a refactory candidate id.
The per-instance frontier itself (adding to the fronts, dominance, the weighted draw)
is core's, ``refactory_dsh.optim.pareto``, which the ``pareto`` parent picker uses
too: one copy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from refactory_dsh.optim import pareto


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
        pareto.add_to_front(self.front_score, self.front_programs, idx, scores)
        return idx

    def aggregate(self, idx: int) -> float:
        """A program's mean over the instances it was scored on, ``-inf`` with none
        (``core/state.py:438-445``)."""
        return pareto.aggregate(self.subscores[idx])

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

