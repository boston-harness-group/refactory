"""GEPA's epoch-shuffled minibatch sampler.

Ported from ``gepa`` 0.1.4 ``strategies/batch_sampler.py:26-103``
(``EpochShuffledBatchSampler``), for the default one minibatch per iteration. It
draws from the strategy's one shared ``random.Random`` (``api.py:304``, ``:351``).
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any


class EpochShuffledBatchSampler:
    """Shuffle the train ids each epoch, pad to a whole number of minibatches with the
    least frequent ids, and hand out minibatch ``i`` of the shuffled list."""

    def __init__(self, minibatch_size: int, rng: random.Random) -> None:
        self.minibatch_size = minibatch_size
        self.rng = rng
        self.shuffled_ids: list[str] = []
        self.epoch = -1
        self.id_freqs: Counter[str] = Counter()
        self.last_trainset_size = 0

    def _update_shuffled(self, ids: list[str]) -> None:
        """``batch_sampler.py:48-69``: one ``rng.shuffle``, then pad with the least
        frequent id, the last inserted on a tie (``most_common()[::-1][0]``)."""
        self.last_trainset_size = len(ids)
        self.shuffled_ids = list(ids)
        self.rng.shuffle(self.shuffled_ids)
        self.id_freqs = Counter(self.shuffled_ids)
        mod = len(ids) % self.minibatch_size
        for _ in range((self.minibatch_size - mod) if mod else 0):
            selected = self.id_freqs.most_common()[::-1][0][0]
            self.shuffled_ids.append(selected)
            self.id_freqs[selected] += 1

    def next_minibatch(self, ids: list[str], iteration: int) -> list[str]:
        """``batch_sampler.py:71-103``: the minibatch of global iteration ``iteration``
        (GEPA's ``state.i``, which merge iterations advance too); reshuffled when that
        crosses into a new epoch of the padded list."""
        if not ids:
            raise ValueError("gepa: cannot sample a minibatch from no train ids")
        base = iteration * self.minibatch_size
        curr_epoch = 0 if self.epoch == -1 else base // max(len(self.shuffled_ids), 1)
        if not self.shuffled_ids or len(ids) != self.last_trainset_size or curr_epoch > self.epoch:
            self.epoch = curr_epoch
            self._update_shuffled(ids)
        start = base % len(self.shuffled_ids)
        return self.shuffled_ids[start:start + self.minibatch_size]

    def state_dict(self) -> dict[str, Any]:
        return {
            "shuffled_ids": list(self.shuffled_ids),
            "epoch": self.epoch,
            # Insertion order matters to the padding's tie-break: kept as pairs.
            "id_freqs": [[k, v] for k, v in self.id_freqs.items()],
            "last_trainset_size": self.last_trainset_size,
        }

    def load_state_dict(self, data: dict[str, Any]) -> None:
        self.shuffled_ids = [str(i) for i in data.get("shuffled_ids") or []]
        self.epoch = int(data.get("epoch", -1))
        self.id_freqs = Counter({str(k): int(v) for k, v in data.get("id_freqs") or []})
        self.last_trainset_size = int(data.get("last_trainset_size", 0))
