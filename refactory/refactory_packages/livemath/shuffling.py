"""LiveMathematicianBench choice shuffling.

Ported from SkillOpt (MIT) — skillopt/envs/livemathematicianbench/dataloader.py:196-226.
"""
from __future__ import annotations

import hashlib
import random

from .grading import normalize_label as _normalize_label

# Ported from skillopt/envs/livemathematicianbench/dataloader.py:16 (MIT)
_CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G"]


# Ported from skillopt/envs/livemathematicianbench/dataloader.py:196-198 (MIT)
def item_shuffle_seed(item_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


# Ported from skillopt/envs/livemathematicianbench/dataloader.py:200-226 (MIT)
def shuffle_item_choices(item: dict, seed: int, shuffle: bool = True) -> dict:
    if not shuffle:
        return {
            **item,
            "choices": [dict(c) for c in item["choices"]],
            "correct_choice": dict(item["correct_choice"]),
        }

    shuffled_choices = [dict(c) for c in item["choices"]]
    rng = random.Random(item_shuffle_seed(str(item["id"]), seed))
    rng.shuffle(shuffled_choices)

    original_correct = _normalize_label(item["correct_choice"]["label"])
    remapped_choices: list[dict] = []
    new_correct_choice = dict(item["correct_choice"])

    for idx, choice in enumerate(shuffled_choices):
        new_label = _CHOICE_LABELS[idx]
        old_label = _normalize_label(choice["label"])
        remapped_choices.append({"label": new_label, "text": choice["text"]})
        if old_label == original_correct:
            new_correct_choice = {"label": new_label, "text": choice["text"]}

    transformed = dict(item)
    transformed["choices"] = remapped_choices
    transformed["correct_choice"] = new_correct_choice
    return transformed
