"""LiveMathematicianBench Task.

Loads items from qa_*_final.json files, normalises them as SkillOpt's
dataloader does, and verifies solver passes with SkillOpt's evaluator logic.

Ported from SkillOpt (MIT) — skillopt/envs/livemathematicianbench/dataloader.py.
"""
from __future__ import annotations

import glob as _glob
import json
import os
from pathlib import Path
from typing import Any

from refactory_dsh.task import Task, TaskInstance, VerifyResult

from .grading import evaluate
from .shuffling import shuffle_item_choices

# Ported from skillopt/envs/livemathematicianbench/dataloader.py:16 (MIT)
_CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G"]


# ── Raw data loading utilities ported from SkillOpt dataloader.py ────────


# Ported from skillopt/envs/livemathematicianbench/dataloader.py:39-61 (MIT)
def _coerce_choices(raw_choices: Any) -> list[dict]:
    if isinstance(raw_choices, list):
        choices: list[dict] = []
        for idx, item in enumerate(raw_choices):
            if isinstance(item, dict):
                label = str(item.get("label") or _CHOICE_LABELS[idx]).strip()
                text = str(item.get("text") or item.get("content") or "").strip()
            else:
                label = _CHOICE_LABELS[idx]
                text = str(item).strip()
            if text:
                choices.append({"label": label, "text": text})
        return choices

    if isinstance(raw_choices, dict):
        labels = sorted(raw_choices.keys())
        return [
            {"label": str(label).strip(), "text": str(raw_choices[label]).strip()}
            for label in labels
            if str(raw_choices[label]).strip()
        ]

    return []


# Ported from skillopt/envs/livemathematicianbench/dataloader.py:64-70 (MIT)
def _coerce_theorem_types(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if raw is None:
        return []
    text = str(raw).strip()
    return [text] if text else []


# Ported from skillopt/envs/livemathematicianbench/dataloader.py:73-74 (MIT)
def _normalize_label(text: str) -> str:
    return str(text).strip().upper().rstrip(".):")


# Ported from skillopt/envs/livemathematicianbench/dataloader.py:77-120 (MIT)
def _normalize_item(item: dict, row_idx: int, source_path: str) -> dict:
    mcq = item.get("mcq", {}) if isinstance(item.get("mcq"), dict) else {}
    question = str(mcq.get("question") or item.get("question") or "").strip()
    choices = _coerce_choices(mcq.get("choices") or item.get("choices") or [])
    correct = mcq.get("correct_choice") or item.get("correct_choice") or {}

    if isinstance(correct, dict):
        correct_label = _normalize_label(correct.get("label", ""))
        correct_text = str(correct.get("text") or "").strip()
    else:
        correct_label = _normalize_label(correct)
        correct_text = ""

    choice_by_label = {
        _normalize_label(choice["label"]): choice["text"]
        for choice in choices
    }
    if correct_label and not correct_text:
        correct_text = choice_by_label.get(correct_label, "")
    if correct_label and correct_text and correct_label not in choice_by_label:
        choices.append({"label": correct_label, "text": correct_text})
        choices.sort(
            key=lambda choice: _CHOICE_LABELS.index(choice["label"])
            if choice["label"] in _CHOICE_LABELS
            else len(_CHOICE_LABELS)
        )
        choice_by_label[correct_label] = correct_text

    month = str(item.get("month") or "").strip()
    item_no = item.get("no", row_idx + 1)
    item_id = f"{month}:{item_no}" if month else str(item_no)

    return {
        "id": item_id,
        "month": month,
        "no": item_no,
        "paper_link": str(item.get("paper_link") or "").strip(),
        "theorem": str(item.get("theorem") or "").strip(),
        "sketch": str(item.get("sketch") or "").strip(),
        "theorem_type": _coerce_theorem_types(item.get("theorem_type")),
        "question": question,
        "choices": choices,
        "correct_choice": {
            "label": correct_label,
            "text": correct_text,
        },
        "source_path": source_path,
    }


# Ported from skillopt/envs/livemathematicianbench/dataloader.py:24-36 (MIT)
def _iter_monthly_files(data_path: str) -> list[str]:
    if not data_path:
        return []
    if os.path.isfile(data_path):
        return [data_path]
    if os.path.isdir(data_path):
        nested = _glob.glob(
            os.path.join(data_path, "**", "qa_*_final.json"),
            recursive=True,
        )
        flat = _glob.glob(os.path.join(data_path, "qa_*_final.json"))
        return sorted(set(nested + flat))
    return []


# Ported from skillopt/envs/livemathematicianbench/dataloader.py:123-143 (MIT)
def load_items(data_path: str) -> list[dict]:
    """Load and normalise LiveMathematicianBench items from JSON files."""
    files = _iter_monthly_files(data_path)
    if not files:
        raise ValueError(
            "LiveMathematicianBench requires data_path to be a qa_*_final.json file "
            "or a directory containing monthly qa_*_final.json files."
        )

    items: list[dict] = []
    for path in files:
        with open(path) as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError(f"Expected JSON array in {path}, got {type(raw).__name__}")
        for row_idx, item in enumerate(raw):
            norm = _normalize_item(item, row_idx=row_idx, source_path=path)
            if norm["question"] and norm["choices"] and norm["correct_choice"]["label"]:
                items.append(norm)
    if not items:
        raise ValueError(f"No valid LiveMathematicianBench items loaded from {data_path}")
    return items


class LiveMathTask(Task):
    """LiveMathematicianBench Task — loads items and grades solver passes.

    Configuration defaults match SkillOpt's configs/livemathematicianbench/default.yaml:
    - shuffle_choices=True  (default.yaml:20)
    - use_theorem=False     (default.yaml:21)
    - use_sketch=False      (default.yaml:22)
    """

    name = "livemath"

    def __init__(
        self,
        data_path: str | None = None,
        shuffle_choices: bool = True,   # configs/livemathematicianbench/default.yaml:20
        use_theorem: bool = False,      # configs/livemathematicianbench/default.yaml:21
        use_sketch: bool = False,       # configs/livemathematicianbench/default.yaml:22
    ) -> None:
        self.data_path = data_path or os.environ.get("LIVEMATH_DATA_PATH", "")
        self.shuffle_choices = shuffle_choices
        self.use_theorem = use_theorem
        self.use_sketch = use_sketch
        self._items: list[dict] | None = None

    def _load(self) -> list[dict]:
        if self._items is None:
            self._items = load_items(self.data_path)
        return self._items

    def instances(self, seed: int | None = None) -> list[TaskInstance]:  # type: ignore[override]
        """Load items and return TaskInstance list.

        When shuffle_choices is True and seed is not None, each item's choices
        are shuffled per SkillOpt's dataloader.py:200-226, using the seed the
        outer loop passes (F18).
        """
        raw_items = self._load()
        result: list[TaskInstance] = []
        for item in raw_items:
            if self.shuffle_choices and seed is not None:
                item = shuffle_item_choices(item, seed)
            correct_choice = item["correct_choice"]
            result.append(TaskInstance(
                id=item["id"],
                metadata={
                    "label": _normalize_label(correct_choice["label"]),
                    "label_text": correct_choice["text"],
                    "question": item["question"],
                    "choices": item["choices"],
                    "correct_choice": correct_choice,
                    "theorem": item.get("theorem", ""),
                    "sketch": item.get("sketch", ""),
                    "theorem_type": item.get("theorem_type", []),
                },
            ))
        return result

    def verify(self, instance: TaskInstance, workspace: Path) -> VerifyResult:
        """Grade the solver's pass using SkillOpt's evaluator logic.

        With tool_policy=none the agent node's captured reply is the raw model
        response written to answer.txt — the text reaches extract_answer unchanged.
        """
        answer_path = workspace / "answer.txt"
        if not answer_path.exists():
            return VerifyResult(passed=False, score=0.0)
        text = answer_path.read_text()
        result = evaluate(
            text,
            instance.metadata["correct_choice"],
            instance.metadata["choices"],
        )
        return VerifyResult(
            passed=bool(result["em"]),
            score=result["em"],
            prediction=result["predicted_label"],
        )
