"""GEPA's system-aware merge.

Ported from ``gepa`` 0.1.4 ``proposer/merge.py:20-403``. A merge takes two programs
on the frontier that share an ancestor and neither descends from the other, and
builds a program component by component: where only one of them changed a
component since the ancestor it takes that change, where both changed it the one
with the higher aggregate (a coin toss on a tie).

``texts[k]`` is program ``k``'s component texts, in component order. GEPA iterates a
``set`` of component names (``merge.py:159-163``), whose order changes with each
process's string hashing; this port iterates the components in order, one of the
orders GEPA can take.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

#: ``(id1, id2, ancestor)`` and ``(id1, id2, description)``, GEPA's ``merges_performed``.
AncestorLog = tuple[int, int, int]
MergeDescription = tuple[int, int, tuple[int, ...]]


def has_desirable_components(texts: Sequence[dict[str, str]], ancestor: int, id1: int, id2: int) -> bool:
    """``merge.py:20-43``: some component one descendant kept as the ancestor had it and
    the other changed."""
    for name in texts[ancestor]:
        anc, a, b = texts[ancestor][name], texts[id1][name], texts[id2][name]
        if (anc == a or anc == b) and a != b:
            return True
    return False


def filter_ancestors(i: int, j: int, common: Sequence[int] | set[int],
                     performed: tuple[list[AncestorLog], list[MergeDescription]],
                     agg: Sequence[float], texts: Sequence[dict[str, str]]) -> list[int]:
    """``merge.py:46-66``."""
    found: list[int] = []
    for ancestor in common:
        if (i, j, ancestor) in performed[0]:
            continue
        if agg[ancestor] > agg[i] or agg[ancestor] > agg[j]:
            continue
        if not has_desirable_components(texts, ancestor, i, j):
            continue
        found.append(ancestor)
    return found


def _ancestors(parents: Sequence[Sequence[int | None]], node: int) -> list[int]:
    """``merge.py:78-85`` (``get_ancestors``), in its insertion order."""
    found: set[int] = set()

    def walk(n: int) -> None:
        for p in parents[n]:
            if p is not None and p not in found:
                found.add(p)
                walk(p)

    walk(node)
    return list(found)


def find_common_ancestor_pair(
    rng: random.Random, parents: Sequence[Sequence[int | None]], candidates: Sequence[int],
    performed: tuple[list[AncestorLog], list[MergeDescription]], agg: Sequence[float],
    texts: Sequence[dict[str, str]], max_attempts: int = 10,
) -> tuple[int, int, int] | None:
    """``merge.py:69-115``: up to ``max_attempts`` draws of a pair, neither an ancestor of
    the other, with a common ancestor ``filter_ancestors`` keeps; the ancestor drawn
    weighted by its aggregate."""
    for _ in range(max_attempts):
        if len(candidates) < 2:
            return None
        i, j = rng.sample(list(candidates), 2)
        if i == j:
            continue
        if j < i:
            i, j = j, i
        anc_i, anc_j = _ancestors(parents, i), _ancestors(parents, j)
        if j in anc_i or i in anc_j:
            continue
        common = filter_ancestors(i, j, set(anc_i) & set(anc_j), performed, agg, texts)
        if common:
            weights = [agg[a] for a in common]
            # GEPA draws weighted by aggregate (``merge.py:107-111``); when every common
            # ancestor scored 0 (a seed that fails everything) its ``rng.choices`` raises
            # and the run aborts. This port draws uniformly then (``weights=None``).
            if not any(w > 0 for w in weights):
                weights = None  # type: ignore[assignment]
            return i, j, rng.choices(list(common), k=1, weights=weights)[0]
    return None


def sample_merge(
    agg: Sequence[float], rng: random.Random, candidates: Sequence[int],
    performed: tuple[list[AncestorLog], list[MergeDescription]],
    texts: Sequence[dict[str, str]], parents: Sequence[Sequence[int | None]],
    has_overlap: Any = None, max_attempts: int = 10,
) -> tuple[dict[str, str], int, int, int] | None:
    """``sample_and_attempt_merge_programs_by_common_predictors`` (``merge.py:118-207``):
    the merged component texts, the two programs and their ancestor; ``None`` when no
    new merge is found."""
    if len(candidates) < 2 or len(parents) < 3:
        return None
    for _ in range(max_attempts):
        found = find_common_ancestor_pair(rng, parents, list(candidates), performed, agg, texts,
                                          max_attempts=max_attempts)
        if found is None:
            continue
        id1, id2, ancestor = found
        if (id1, id2, ancestor) in performed[0]:
            continue
        merged = dict(texts[ancestor])
        desc: tuple[int, ...] = ()
        for name in texts[ancestor]:
            anc, a, b = texts[ancestor][name], texts[id1][name], texts[id2][name]
            if (anc == a or anc == b) and a != b:
                src = id2 if anc == a else id1
            elif anc != a and anc != b:
                src = id1 if agg[id1] > agg[id2] else (id2 if agg[id2] > agg[id1] else rng.choice([id1, id2]))
            else:  # a == b
                src = id1
            merged[name] = texts[src][name]
            desc = (*desc, src)
        if (id1, id2, desc) in performed[1]:
            continue
        if has_overlap is not None and not has_overlap(id1, id2):
            continue
        performed[1].append((id1, id2, desc))
        return merged, id1, id2, ancestor
    return None


def select_eval_subsample(scores1: dict[str, float], scores2: dict[str, float],
                          rng: random.Random, num_subsample_ids: int = 5) -> list[str]:
    """``merge.py:258-288``: up to ``ceil(5/3)`` instances where the first is better, as
    many where the second is, then ties, topped up at random from the rest.

    GEPA's instance ids are ints ``0..n-1``, so its ``list(set & set)`` comes out in
    ascending order; the keys here are strings, whose set order changes per process,
    so they are sorted instead."""
    common = sorted(set(scores1) & set(scores2))
    p1 = [k for k in common if scores1[k] > scores2[k]]
    p2 = [k for k in common if scores2[k] > scores1[k]]
    p3 = [k for k in common if k not in p1 and k not in p2]
    n_each = max(1, math.ceil(num_subsample_ids / 3))
    selected: list[str] = []
    for bucket in (p1, p2, p3):
        if len(selected) >= num_subsample_ids:
            break
        available = [k for k in bucket if k not in selected]
        take = min(len(available), n_each, num_subsample_ids - len(selected))
        if take > 0:
            selected += rng.sample(available, k=take)
    remaining = num_subsample_ids - len(selected)
    if remaining > 0:
        unused = [k for k in common if k not in selected]
        if len(unused) >= remaining:
            selected += rng.sample(unused, k=remaining)
        elif common:
            selected += rng.choices(common, k=remaining)
    return selected[:num_subsample_ids]
