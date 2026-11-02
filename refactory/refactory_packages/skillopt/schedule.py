"""Edit-budget schedule reproducing SkillOpt's CosineScheduler.

Formula: ``CosineScheduler._compute_lr`` at scheduler.py:87-92.
Step is 1-indexed.  For ``total_steps <= 1`` the budget is always ``max_lr``
(scheduler.py:88-89).

Defaults from ``_base_/default.yaml``:
  - max_lr = 4  (line 93, ``learning_rate: 4``)
  - min_lr = 2  (line 94, ``min_learning_rate: 2``)
"""

from __future__ import annotations

import math


def edit_budget(step: int, total_steps: int, max_lr: int = 4, min_lr: int = 2) -> int:
    """Return the edit budget for the given 1-indexed step.

    Reproduces ``CosineScheduler._compute_lr`` at scheduler.py:87-92.
    """
    if total_steps <= 1:
        return max_lr
    t = min(step, total_steps) / total_steps
    lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t))
    return max(min_lr, round(lr))


def full_sequence(total_steps: int, max_lr: int = 4, min_lr: int = 2) -> list[int]:
    """Return the full edit-budget sequence for ``total_steps`` steps."""
    return [edit_budget(s, total_steps, max_lr, min_lr) for s in range(1, total_steps + 1)]
