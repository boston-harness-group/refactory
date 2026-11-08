"""A candidate an earlier edit gave new nodes can be edited again (Python walker loop).

A structural edit adds knobs the seed never declared: a new agent's
``<id>__system_prompt``, a code node's ``<id>__code``, a ``parallel`` branch's own
knobs. The next proposal request is built from the edited candidate, so every knob
lookup in it must read the candidate's own knob states, falling back to the mode's.
Also: a proposal round that fails is visible in the progress stream and the report,
and the ``set_knob`` patch format names the key that holds each entry's action.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from refactory_dsh import nn, snippet
from refactory_dsh import events as run_events
from refactory_dsh.loop import Loop, LoopConfig
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.optim import Candidate, Optimizer, Proposal
from refactory_dsh.optim.base import answer_instructions
from refactory_dsh.outerloop import Run
from refactory_packages.livemath.skill_graph import SkillGraph

needs_sandbox = pytest.mark.skipif(not snippet.available()[0],
                                   reason=f"no code-node sandbox here: {snippet.available()[1]}")

LIVEMATH = "livemath/skill_graph"

AGENT = {"op": "insert_after", "target": "compare",
         "node": {"id": "extra_check", "kind": "agent", "writes": ["extra_check.txt"],
                  "prompt_template": "Check this:\n{{compare_prompt}}",
                  "system_prompt_template": "You check comparisons."}}
CODE = {"op": "insert_after", "target": "compare",
        "node": {"id": "digest", "kind": "op", "call": snippet.RUN,
                 "reads": ["compare_notes.txt"], "produces": ["compare_digest"],
                 "code": ("def main(inp):\n"
                          "    return {'facts': {'compare_digest': "
                          "inp['files'].get('compare_notes.txt', '')[:100]}}\n")}}
PARALLEL = {"op": "parallel", "target": "compare", "branches": [
    {"id": "side_a", "kind": "agent", "writes": ["side_a.txt"],
     "prompt_template": "First angle on:\n{{compare_prompt}}"},
    {"id": "side_b", "kind": "op", "call": snippet.RUN, "inputs": ["compare_prompt"],
     "writes": ["side_b.txt"], "produces": ["side_b_len"],
     "code": ("def main(inp):\n    text = inp['facts']['compare_prompt']\n"
              "    return {'facts': {'side_b_len': len(text)}, 'files': {'side_b.txt': text[:50]}}\n")},
]}


def _child_loop(tmp_path: Path, edit: dict) -> tuple[Loop, Candidate, list[str]]:
    """A loop whose population holds the seed and the child ``edit`` makes of it (built
    by the loop's own proposal path), with the child as the candidate being edited."""
    loop = Loop(LIVEMATH, LoopConfig(runs_root=tmp_path / "run", optimizer="random"),
                project=Path("."))
    seed = Candidate.from_state(loop.mode.state_dict())
    loop.population.append(seed)
    child = Run(_loop=loop).attempt(Proposal(edits=(edit,), rationale="r", parent=seed.id), seed)
    assert isinstance(child, Candidate), child
    loop.population.append(child)
    loop._editing_parent = child.id
    new = sorted(set(child.state["knobs"]) - set(seed.state["knobs"]))
    assert new, "the edit added no knob, so this test would prove nothing"
    return loop, child, new


@needs_sandbox
@pytest.mark.parametrize("edit", [AGENT, CODE, PARALLEL], ids=["agent", "code", "parallel"])
def test_the_next_request_for_a_structural_child_lists_its_new_knobs(tmp_path, edit):
    loop, _child, new = _child_loop(tmp_path, edit)
    ctx = loop._context()  # raised KeyError: the knob was looked up in the seed only
    listed = {knob["name"]: knob for knob in ctx.knobs}
    for name in new:
        assert name in listed, name
        assert listed[name]["reachable"] is True, name
    table = ctx.render().split("## Parameters you may move", 1)[1].split("\n## ", 1)[0]
    for name in new:
        assert name in table, name


def test_attribution_enumerates_a_knob_only_a_candidate_declares(tmp_path):
    """``untried`` reads the candidates' own declaration of a knob the mode lacks."""
    loop = Loop(LIVEMATH, LoopConfig(runs_root=tmp_path / "run", optimizer="random"),
                project=Path("."))
    state = loop.mode.state_dict()
    state["knobs"]["extra_mode"] = {"spec": nn.Knob("a", "enum", enum=("a", "b", "c")).spec(),
                                    "value": "a"}
    candidate = Candidate.from_state(state)
    candidate.score = 0.5
    attribution = loop._knob_attribution([candidate])
    assert attribution["extra_mode"]["untried"] == ["b", "c"]


# ── end to end: edit a kept structural child again ────────────────────────


class _Counter(nn.Mode):
    """One op writes a number; the score adds a ``bonus`` fact when a step produces one."""

    name = "counter"

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {"step": nn.Knob(2, "int", lo=1, hi=2, description="a bounded counter")}

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node("only", "op", call="loop_ops:write_number",
                          knobs=("step",), writes=("number.json",), produces=("value",)))
        return graph

    def score(self, run: nn.RunResult) -> float:
        return float(run.facts["value"]) + float(run.facts.get("bonus", 0))


BONUS = {"op": "insert_after", "target": "only",
         "node": {"id": "bonus", "kind": "op", "call": snippet.RUN, "inputs": ["value"],
                  "produces": ["bonus"],
                  "code": "def main(inp):\n    return {'facts': {'bonus': 1}}\n"}}
BONUS_TWO = "def main(inp):\n    return {'facts': {'bonus': 5}}\n"


class _TwoStep(Optimizer):
    """Adds the bonus node to the seed; then rewrites the new node's knob on the child."""

    name = "two-step"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.seen: list[list[str]] = []

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        names = [knob["name"] for knob in ctx.knobs]
        self.seen.append(names)
        parent = next((p for p in parents if "bonus__code" in p.state["knobs"]), parents[0])
        if "bonus__code" in parent.state["knobs"]:
            edit = {"op": "set_knob", "name": "bonus__code", "value": BONUS_TWO}
        else:
            edit = BONUS
        return [Proposal(edit=edit, parent=parent.id, rationale="scripted")]


@needs_sandbox
def test_a_kept_structural_child_is_proposed_applied_and_evaluated_again(tmp_path):
    from test_loop import _patch_mode

    optimizer = _TwoStep(k=1)
    with _patch_mode(_Counter):
        loop = Loop(LIVEMATH, LoopConfig(epochs=3, batch_size=1, optimizer="random", seed=5,
                                         runs_root=tmp_path / "run"),
                    project=Path("."), optimizer=optimizer)
        report = loop.run()
    assert optimizer.proposal_failures == [], optimizer.proposal_failures
    assert any("bonus__code" in names for names in optimizer.seen), optimizer.seen
    rewritten = [c for c in loop.population
                 if (c.state["knobs"].get("bonus__code") or {}).get("value") == BONUS_TWO]
    assert rewritten and all(c.evaluated for c in rewritten), "the second edit never ran"
    assert max(c.raw_score for c in rewritten) == pytest.approx(2 + 5)
    assert report["proposer"]["failure_count"] == 0


# ── a failed proposal round is visible ────────────────────────────────────


class _Broken(Optimizer):
    name = "broken"

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        raise ValueError("cannot read the request\nsecond line")


def test_a_failed_proposal_round_is_in_the_progress_stream_and_the_report(tmp_path):
    root = tmp_path / "run"
    Loop("circle_packing/adaptive",
         LoopConfig(epochs=1, batch_size=1, optimizer="random", seed=5, runs_root=root),
         project=Path("."), optimizer=_Broken(k=1)).run()
    events = [json.loads(line) for line in (root / "progress.jsonl").read_text().splitlines()]
    failed = [e for e in events if e["step"] == "proposal_failed"]
    assert failed and failed[0]["error"] == "ValueError"
    assert failed[0]["detail"] == "cannot read the request\nsecond line"
    # One bullet per failure, even for a multi-line message.
    assert "- ValueError: cannot read the request second line\n" in (root / "report.md").read_text()
    # Shown by the default (non-verbose) printer, not only with --verbose.
    line = run_events.render(failed[0])
    assert line and "proposal failed" in line
    assert "ValueError: cannot read the request second line" in line


# ── the set_knob patch format ─────────────────────────────────────────────


def _example(text: str) -> dict:
    match = re.search(r"e\.g\. (\{.*?\}\]\})", text)
    assert match, text
    return json.loads(match.group(1))


def test_the_patch_format_names_its_key_and_its_example_applies():
    catalogue = nn_edits.catalogue("broad")
    description = next(op for op in catalogue if op["op"] == "set_knob")["description"]
    rules = answer_instructions(catalogue)
    state = SkillGraph().state_dict()
    current = state["knobs"]["compare__system_prompt"]["value"]
    for text in (description, rules):
        assert '"kind": <action>' in text and 'under "kind"' in text
        example = _example(text)
        assert example["patch"][0]["kind"] == "replace"
        example["name"] = "compare__system_prompt"
        example["patch"][0].update(target=current[:20], content="PATCHED")
        out = nn_edits.apply(example, state)
        assert out["knobs"]["compare__system_prompt"]["value"].startswith("PATCHED")


def test_a_patch_entry_without_kind_is_refused_naming_the_key():
    """The action belongs under "kind" only; another key is refused, and the refusal
    (fed back to the proposer) names the key, rather than the action being guessed."""
    state = SkillGraph().state_dict()
    for key in ("action", "op"):
        with pytest.raises(nn_edits.EditError,
                           match=r"unknown edit kind None: each patch entry names its "
                                 r"action under 'kind'"):
            nn_edits.apply({"op": "set_knob", "name": "compare__system_prompt",
                            "patch": [{key: "append", "content": "TAIL"}]}, state)
