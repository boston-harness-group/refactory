"""``text_edit_style``: text knobs change by small line edits when it is ``small``.

Under ``small`` a whole-value ``set_knob`` on a text knob is refused before anything
runs, a patch of at most ``text_edit_budget`` line edits is accepted, other knobs and
structure edits are unaffected, and the request says so. ``free`` (the default) is
unchanged. Each test fails when the behaviour it names is reverted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import profiles
from refactory_dsh.loop import Loop, LoopConfig, LoopError, text_edit_refusal
from refactory_dsh.optim import Optimizer, Proposal, ProposalContext

from test_skillopt import INITIAL_SKILL, MODE_ID, SkillProbe, _patch_mode, _write_parts
from test_val_gate import LevelProbe

LINE = "When approaching a problem, break it into steps."


def _patch(n: int) -> dict[str, Any]:
    return {"op": "set_knob", "name": "skill", "patch": [
        {"kind": "insert_after", "target": LINE, "content": f"- step hint {i}"} for i in range(n)]}


class Scripted(Optimizer):
    """Each round's proposals, against the first parent; records what it was told."""

    name = "scripted_edits"

    def __init__(self, rounds: list[list[dict[str, Any]]]) -> None:
        super().__init__(k=1)
        self.rounds = list(rounds)
        self.prompts: list[str] = []

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        self.prompts.append(ctx.render())
        edits = self.rounds.pop(0) if self.rounds else []
        return [Proposal(edit=edit, parent=parents[0].id) for edit in edits]


def _run(tmp_path: Path, rounds: list[list[dict[str, Any]]], *, mode: Any = SkillProbe,
         mode_id: str = MODE_ID, **overrides: Any) -> tuple[Loop, Scripted]:
    proposer = Scripted(rounds)
    config = dict(epochs=2, batch_size=1, optimizer="random", seed=7, runs_root=tmp_path / "run",
                  holdout=0.0, attempts=3, parts_manifest=str(_write_parts(tmp_path)))
    config.update(overrides)
    with _patch_mode(mode):
        loop = Loop(mode_id, LoopConfig(**config), project=Path("."))
        loop.optimizer = proposer
        loop.run()
    return loop, proposer


def _reasons(loop: Loop) -> list[str]:
    return [reason for rejection in loop.rejections for reason in rejection.reasons]


def _skills(loop: Loop) -> list[str]:
    return [str(c.state["knobs"]["skill"]["value"]) for c in loop.population[1:]]


def test_small_refuses_a_whole_new_text_value_and_takes_a_patch(tmp_path):
    """The rewrite is refused before evaluation with a reason that reaches the
    retry; the retry's patch is accepted and keeps the rest of the text."""
    rewrite = {"op": "set_knob", "name": "skill", "value": "# A whole new skill\n"}
    loop, proposer = _run(tmp_path, [[rewrite], [_patch(1)]], text_edit_style="small")
    assert any(r.startswith("E-edit-style:") and "'patch'" in r for r in _reasons(loop))
    assert "# A whole new skill\n" not in _skills(loop)
    patched = [s for s in _skills(loop) if "- step hint 0" in s]
    assert patched and "## Verification" in patched[0], "the patch keeps the rest"
    assert "E-edit-style" in proposer.prompts[1], "the reason reaches the retry"
    assert "## How to change a text parameter" in proposer.prompts[0], "the loop's request says so"


def test_small_limits_the_line_edits_of_one_proposal(tmp_path):
    loop, _ = _run(tmp_path, [[_patch(3)], [_patch(2)]], text_edit_style="small",
                   text_edit_budget=2)
    assert any(r.startswith("E-edit-budget:") and "3 line edits" in r for r in _reasons(loop))
    assert [s.count("- step hint") for s in _skills(loop)] == [2]
    state = loop.population[0].state
    assert text_edit_refusal(_patch(4), state, style="small", budget=4) is None
    assert text_edit_refusal(_patch(5), state, style="small", budget=4).startswith("E-edit-budget")


def test_small_leaves_other_knobs_and_structure_edits_alone(tmp_path):
    loop, _ = _run(tmp_path, [[{"op": "set_knob", "name": "level", "value": 7}]],
                   mode=LevelProbe, text_edit_style="small", allowed_ops=("set_knob",))
    assert [int(c.state["knobs"]["level"]["value"]) for c in loop.population[1:]] == [7]
    assert not _reasons(loop)
    state = loop.population[0].state
    assert text_edit_refusal({"op": "insert_after", "target": "work", "node": {}}, state,
                             style="small", budget=4) is None


def test_free_is_the_default_and_takes_a_whole_new_value(tmp_path):
    rewrite = {"op": "set_knob", "name": "skill", "value": "# A whole new skill\n"}
    loop, proposer = _run(tmp_path, [[rewrite]])
    assert loop.config.text_edit_style == "free"
    assert "# A whole new skill\n" in _skills(loop) and not _reasons(loop)
    assert "## How to change a text parameter" not in proposer.prompts[0]


def test_the_request_explains_small_edits_with_a_real_knob():
    knobs = [{"name": "skill", "value": INITIAL_SKILL, "kind": "str", "values": [],
              "frozen": False, "range": None}]
    text = ProposalContext(knobs=knobs, text_edit_style="small", text_edit_budget=3).render()
    assert "## How to change a text parameter" in text and "at most 3 line edits" in text
    example = json.loads(text.split("For example:\n`", 1)[1].split("`\n", 1)[0])
    assert example["name"] == "skill" and example["patch"][0]["target"] in INITIAL_SKILL
    assert example["patch"][0]["kind"] == "replace"
    from refactory_dsh.nn import edits as nn_edits
    with _patch_mode(SkillProbe):
        state = SkillProbe().state_dict()
    applied = nn_edits.apply(example, state)  # the example is a patch that applies
    assert "<that line, improved>" in applied["knobs"]["skill"]["value"]
    assert "## How to change a text parameter" not in ProposalContext(knobs=knobs).render()


def test_the_setting_comes_from_the_command_line_and_is_kept_on_resume(tmp_path):
    parser = argparse.ArgumentParser()
    profiles.add_arguments(parser)
    args = parser.parse_args(["--text-edit-style", "small", "--text-edit-budget", "3"])
    settings = profiles.normalize({k: v for k, v in vars(args).items()
                                   if k in profiles.defaults() and v is not None})
    assert (settings["text_edit_style"], settings["text_edit_budget"]) == ("small", 3)
    for bad in ({"text_edit_style": "tiny"}, {"text_edit_budget": 0}, {"text_edit_budget": True}):
        with pytest.raises(profiles.ProfileError, match="text_edit"):
            profiles.normalize(bad)
    with _patch_mode(SkillProbe), pytest.raises(LoopError, match="text_edit_style"):
        Loop(MODE_ID, LoopConfig(runs_root=tmp_path / "bad", text_edit_style="tiny"), project=Path("."))

    loop, _ = _run(tmp_path, [[_patch(1)]], text_edit_style="small", text_edit_budget=3)
    saved = LoopConfig.from_dict(json.loads((loop.root / "config.json").read_text()))
    assert (saved.text_edit_style, saved.text_edit_budget) == ("small", 3)
    rewrite = {"op": "set_knob", "name": "skill", "value": "# rewritten\n"}
    resumed, _ = _run(tmp_path, [[rewrite]], epochs=3, resume=True,
                      text_edit_style="small", text_edit_budget=3)
    assert any(r.startswith("E-edit-style:") for r in _reasons(resumed))
