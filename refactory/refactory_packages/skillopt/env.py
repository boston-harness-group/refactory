"""What SkillOpt takes from its environment adapter, found in the mode's package.

SkillOpt (microsoft/SkillOpt, MIT) keeps two things per environment that its
generic optimizer reads:

- **Prompt overrides.** ``load_prompt(name, env)`` (``skillopt/prompts/__init__.py:32-58``)
  reads ``skillopt/envs/<env>/prompts/<name>.md`` when it exists and falls back
  to the generic ``skillopt/prompts/<name>.md``. The adapter derives ``env``
  from its own module path (``envs/base.py:288-296``).
- **The rollout result.** The env's rollout writes one result dict per item
  plus ``conversation.json``, ``target_system_prompt.txt`` and
  ``target_user_prompt.txt``; the generic ``fmt_minibatch_trajectories``
  (``skillopt/gradient/reflect.py:121-248``) renders them for the analysts.

Here the environment is the package that holds the run's mode, found the same
way SkillOpt finds its env -- from where the code lives, not from a setting:

- ``refactory_packages/<package>/skillopt_prompts/<name>.md`` overrides the
  generic prompt ``<name>`` (:data:`GENERIC_PROMPTS`).
- ``refactory_packages/<package>/skillopt_view.py`` may define
  ``rollout_result(record, knobs) -> dict``: the package's own rendering of one
  F6 item record (``refactory_dsh/items.py:make_item_record``) as SkillOpt's
  rollout result for that environment, with the conversation under
  ``"conversation"`` (``None`` when the env would have written no
  ``conversation.json``). Without it, :func:`default_rollout_result` renders
  the record generically.

Nothing here names a package: LiveMath's overrides are LiveMath's files.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any, Callable

from refactory_packages.skillopt import prompts

#: The directory, inside a mode's package, that holds SkillOpt prompt overrides.
PROMPT_DIR = "skillopt_prompts"

#: The module, inside a mode's package, that renders SkillOpt rollout results.
VIEW_MODULE = "skillopt_view"

#: The package root every mode's package is importable under
#: (``refactory_dsh/registry.py`` ``load``).
ROOT_PACKAGE = "refactory_packages"

#: SkillOpt's generic prompts by file name (``skillopt/prompts/<name>.md``).
GENERIC_PROMPTS: dict[str, str] = {
    "analyst_error": prompts.FAILURE_ANALYST_SYSTEM,
    "analyst_success": prompts.SUCCESS_ANALYST_SYSTEM,
    "merge_failure": prompts.MERGE_FAILURE_SYSTEM,
    "merge_success": prompts.MERGE_SUCCESS_SYSTEM,
    "merge_final": prompts.MERGE_FINAL_SYSTEM,
    "ranking": prompts.RANK_SYSTEM,
    "slow_update": prompts.SLOW_UPDATE_SYSTEM,
    "meta_skill": prompts.META_SKILL_SYSTEM,
}

RolloutView = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def package_of(mode_id: str) -> str:
    """The package part of a ``<package>/<mode>`` identifier, or ``""``."""
    head, sep, _ = str(mode_id or "").partition("/")
    return head if sep else ""


def _package_dir(package: str) -> Path | None:
    if not package:
        return None
    try:
        spec = importlib.util.find_spec(f"{ROOT_PACKAGE}.{package}")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])


class SkillOptEnv:
    """One mode's package, as SkillOpt's environment adapter."""

    def __init__(self, package: str = "") -> None:
        self.package = package
        self._dir = _package_dir(package)
        self._view: RolloutView | None = self._load_view()

    @classmethod
    def for_mode(cls, mode_id: str) -> "SkillOptEnv":
        return cls(package_of(mode_id))

    # -- prompts: skillopt/prompts/__init__.py:32-58 ------------------------

    def override_path(self, name: str) -> Path | None:
        if self._dir is None:
            return None
        return self._dir / PROMPT_DIR / f"{name}.md"

    def prompt(self, name: str) -> str:
        """The env's ``<name>.md`` when it has one, else SkillOpt's generic text.

        Mirrors ``load_prompt(name, env)``: the file is used as read, with no
        stripping. Raises ``KeyError`` when neither exists, as ``load_prompt``
        raises ``FileNotFoundError``.
        """
        path = self.override_path(name)
        if path is not None and path.is_file():
            return path.read_text(encoding="utf-8")
        if name not in GENERIC_PROMPTS:
            raise KeyError(
                f"SkillOpt prompt {name!r} not found: no {PROMPT_DIR}/{name}.md in "
                f"package {self.package!r} and no generic prompt of that name"
            )
        return GENERIC_PROMPTS[name]

    # -- rollout results ----------------------------------------------------

    def _load_view(self) -> RolloutView | None:
        if self._dir is None or not (self._dir / f"{VIEW_MODULE}.py").is_file():
            return None
        module = importlib.import_module(f"{ROOT_PACKAGE}.{self.package}.{VIEW_MODULE}")
        view = getattr(module, "rollout_result", None)
        if not callable(view):
            raise TypeError(
                f"{ROOT_PACKAGE}.{self.package}.{VIEW_MODULE} must define "
                "rollout_result(record, knobs)"
            )
        return view

    def rollout_result(self, record: dict[str, Any], knobs: dict[str, Any]) -> dict[str, Any]:
        """One item record as the env's SkillOpt rollout result."""
        if self._view is not None:
            return self._view(record, dict(knobs))
        return default_rollout_result(record, dict(knobs))


def default_rollout_result(record: dict[str, Any], knobs: dict[str, Any]) -> dict[str, Any]:
    """A generic rollout result from an F6 record, for a package with no view.

    The fields SkillOpt's formatters read (reflect.py:159-199,
    slow_update.py:242-257): pass, score and prediction; the task under
    ``question``/``task_description``/``instruction`` and its type under
    ``task_type``/``instruction_type``, from the item's own metadata, with a
    ``question`` also given as ``task_description`` for the analyst; the
    record's fail reason; one agent message per exchange; the first exchange's prompt as the
    target user prompt; and the grading as the appended ``system`` entry a
    rollout adds (e.g. livemathematicianbench/rollout.py:272-281).
    """
    meta = record.get("metadata") or {}
    exchanges = [e for e in (record.get("exchanges") or []) if isinstance(e, dict)]
    result: dict[str, Any] = {
        "id": str(record.get("id", "")),
        "hard": 1 if record.get("passed") else 0,
        "soft": record.get("score"),
        "predicted_answer": record.get("prediction"),
    }
    for key in ("question", "task_description", "instruction", "task_type", "instruction_type"):
        if key in meta:
            result[key] = meta[key]
    # The analyst's ``Task:`` line reads only task_description/instruction
    # (reflect.py:159-199), so an item that names its task ``question`` would
    # otherwise reach the analyst with no task, and with no exchange (e.g. it
    # ran out of output tokens) nothing else would show the task either.
    if "question" in result and not ("task_description" in result or "instruction" in result):
        result["task_description"] = result["question"]
    fail_reason = prompts.record_fail_reason(record)
    result["fail_reason"] = "" if record.get("passed") else fail_reason
    result["n_turns"] = len(exchanges)
    if exchanges:
        result["target_user_prompt"] = str(exchanges[0].get("prompt") or "")
    conversation: list[dict[str, Any]] = [
        {"type": "message", "turn": i, "content": str(e.get("response") or "")}
        for i, e in enumerate(exchanges, 1)
    ]
    conversation.append({
        "role": "system",
        "content": (
            "[EVALUATION RESULT]\n"
            f"Prediction: {record.get('prediction')!r}\n"
            f"Label: {record.get('label')!r}\n"
            f"Label text: {record.get('label_text')!r}\n"
            f"Score: {record.get('score')!r}\n"
            f"Passed: {record.get('passed')!r}"
        ),
    })
    result["conversation"] = conversation
    return result
