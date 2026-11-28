"""Smoke output review: show the proposer what its new/changed steps produced."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from refactory_dsh import nn, registry
from refactory_dsh.loop import Loop, LoopConfig
from refactory_dsh.nn.mode import RunResult
from refactory_dsh.optim import Candidate, Optimizer, Proposal
from refactory_dsh.optim.smoke_review import (
    FACT_SIZE_WARN,
    build_review_block,
    changed_steps,
    parse_review_reply,
    review_needed,
)

MODE = "circle_packing/adaptive"


def _config(root: Path, **overrides: object) -> LoopConfig:
    base = dict(epochs=1, batch_size=1, optimizer="random", seed=5, runs_root=root)
    base.update(overrides)
    return LoopConfig(**base)  # type: ignore[arg-type]


def _mode() -> nn.Mode:
    _package, mode_class = registry.resolve(Path("."), MODE)
    return mode_class()


def _structural_edit(mode: nn.Mode) -> dict:
    block = {node.id: node for node in mode.insertable()}["restart_on_stall"]
    return {"op": "insert_after", "target": "accept", "node": block.to_state()}


# ── unit tests for the review module ─────────────────────────────────────


class TestChangedSteps:
    def test_new_step_detected(self):
        parent = Candidate.from_state({
            "graph": {"nodes": {"a": {"id": "a", "kind": "op"}}},
            "knobs": {},
        })
        child = Candidate.from_state({
            "graph": {"nodes": {
                "a": {"id": "a", "kind": "op"},
                "b": {"id": "b", "kind": "op"},
            }},
            "knobs": {},
        })
        added, changed = changed_steps(parent, child)
        assert added == {"b"}
        assert changed == set()

    def test_changed_prompt_detected(self):
        parent = Candidate.from_state({
            "graph": {"nodes": {"a": {"id": "a", "kind": "agent",
                                      "prompt_template": "old prompt"}}},
            "knobs": {},
        })
        child = Candidate.from_state({
            "graph": {"nodes": {"a": {"id": "a", "kind": "agent",
                                      "prompt_template": "new prompt"}}},
            "knobs": {},
        })
        added, changed = changed_steps(parent, child)
        assert added == set()
        assert changed == {"a"}

    def test_no_change_no_steps(self):
        state: dict[str, Any] = {
            "graph": {"nodes": {"a": {"id": "a", "kind": "op"}}},
            "knobs": {},
        }
        parent = Candidate.from_state(state)
        child = Candidate.from_state(state)
        added, changed = changed_steps(parent, child)
        assert added == set()
        assert changed == set()


class TestReviewNeeded:
    def test_needed_for_new_step(self):
        parent = Candidate.from_state({
            "graph": {"nodes": {"a": {"id": "a", "kind": "op"}}},
            "knobs": {},
        })
        child = Candidate.from_state({
            "graph": {"nodes": {
                "a": {"id": "a", "kind": "op"},
                "b": {"id": "b", "kind": "op"},
            }},
            "knobs": {},
        })
        assert review_needed(parent, child)

    def test_not_needed_for_knob_only(self):
        parent = Candidate.from_state({
            "graph": {"nodes": {"a": {"id": "a", "kind": "op"}}},
            "knobs": {"x": {"value": 1}},
        })
        child = Candidate.from_state({
            "graph": {"nodes": {"a": {"id": "a", "kind": "op"}}},
            "knobs": {"x": {"value": 2}},
        })
        assert not review_needed(parent, child)


class TestParseReviewReply:
    def test_keep(self):
        decision, payload = parse_review_reply("keep")
        assert decision == "keep"
        assert payload is None

    def test_drop(self):
        decision, payload = parse_review_reply("drop")
        assert decision == "drop"

    def test_revised(self):
        answer = json.dumps({"proposals": [{"edit": {"op": "set_knob", "name": "x", "value": 1}}]})
        decision, payload = parse_review_reply(answer)
        assert decision == "revised"
        assert payload is not None

    def test_empty_defaults_to_keep(self):
        decision, payload = parse_review_reply("")
        assert decision == "keep"

    def test_backtick_keep(self):
        decision, payload = parse_review_reply("`keep`")
        assert decision == "keep"


class TestBuildReviewBlock:
    def _parent_child(self, *, child_node_kind: str = "op",
                      child_node_extra: dict[str, Any] | None = None):
        parent = Candidate.from_state({
            "graph": {"nodes": {"a": {"id": "a", "kind": "op"}}},
            "knobs": {},
        })
        child_node: dict[str, Any] = {"id": "b", "kind": child_node_kind}
        if child_node_extra:
            child_node.update(child_node_extra)
        child = Candidate.from_state({
            "graph": {"nodes": {
                "a": {"id": "a", "kind": "op"},
                "b": child_node,
            }},
            "knobs": {},
        })
        return parent, child

    def test_no_review_when_no_steps_changed(self):
        state: dict[str, Any] = {
            "graph": {"nodes": {"a": {"id": "a", "kind": "op"}}},
            "knobs": {},
        }
        parent = Candidate.from_state(state)
        child = Candidate.from_state(state)
        result = RunResult(item_records=[{"id": "i1", "passed": True, "score": 1.0}])
        assert build_review_block(parent, child, result) is None

    def test_review_block_includes_step_output(self):
        parent, child = self._parent_child()
        result = RunResult(item_records=[{
            "id": "i1", "passed": True, "score": 1.0,
            "steps": [{"node": "b", "kind": "op", "output": "some output", "empty": False}],
            "exchanges": [],
        }])
        block = build_review_block(parent, child, result)
        assert block is not None
        assert "some output" in block
        assert "`b`" in block
        assert "new" in block

    def test_size_warning_on_huge_output(self):
        parent, child = self._parent_child()
        huge = "x" * (FACT_SIZE_WARN + 1000)
        result = RunResult(item_records=[{
            "id": "i1", "passed": True, "score": 1.0,
            "steps": [{"node": "b", "kind": "op", "output": huge, "empty": False}],
            "exchanges": [],
        }])
        block = build_review_block(parent, child, result)
        assert block is not None
        assert "⚠" in block
        assert f"{FACT_SIZE_WARN:,}" in block

    def test_empty_output_warning(self):
        parent, child = self._parent_child()
        result = RunResult(item_records=[{
            "id": "i1", "passed": True, "score": 1.0,
            "steps": [{"node": "b", "kind": "op", "output": "", "empty": True}],
            "exchanges": [],
        }])
        block = build_review_block(parent, child, result)
        assert block is not None
        assert "empty" in block

    def test_no_answer_warning(self):
        parent, child = self._parent_child()
        result = RunResult(item_records=[{
            "id": "i1", "passed": False, "score": 0.0,
            "steps": [],
            "exchanges": [],
            "missing_steps": {"b": {"cause": "crash"}},
        }])
        block = build_review_block(parent, child, result)
        assert block is not None
        assert "no answer" in block

    def test_review_block_none_without_item_records(self):
        parent, child = self._parent_child()
        result = RunResult(item_records=[])
        assert build_review_block(parent, child, result) is None

    def test_identical_file_flag(self, tmp_path: Path):
        parent = Candidate.from_state({
            "graph": {"nodes": {
                "data": {"id": "data", "kind": "data"},
                "a": {"id": "a", "kind": "op"},
            }},
            "knobs": {},
        })
        child = Candidate.from_state({
            "graph": {"nodes": {
                "data": {"id": "data", "kind": "data"},
                "a": {"id": "a", "kind": "op"},
                "b": {"id": "b", "kind": "op",
                       "reads": ("input.txt",), "writes": ("output.txt",)},
            }},
            "knobs": {},
        })
        from refactory_dsh.items import pass_dir_of
        pdir = pass_dir_of(tmp_path, "data", "i1", 1)
        pdir.mkdir(parents=True)
        content = b"hello world"
        (pdir / "input.txt").write_bytes(content)
        (pdir / "output.txt").write_bytes(content)
        result = RunResult(item_records=[{
            "id": "i1", "passed": True, "score": 1.0,
            "steps": [{"node": "b", "kind": "op", "output": "ok", "empty": False}],
            "exchanges": [],
        }])
        block = build_review_block(parent, child, result, smoke_workdir=tmp_path)
        assert block is not None
        assert "byte-identical" in block


# ── integration tests through _smoke_review_candidates ───────────────────


class _ReviewOptimizer(Optimizer):
    """An optimizer whose review answers are scripted, with uses_model=True
    so the review path fires."""

    name = "scripted_review"
    uses_model = True

    def __init__(self, edit: dict, review_answer: str = "keep", **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self._edit = edit
        self._review_answer = review_answer
        self._system = ""
        self.review_calls: list[str] = []

    def propose(self, ctx: Any, n: int, parents: Any) -> list[Proposal]:
        return [
            Proposal(edit=self._edit, parent=parents[0].id, rationale="scripted")
            for _ in range(max(1, n))
        ]

    def _complete(self, call: dict[str, Any], text: str, system: str,
                  gate: Any = None) -> tuple[str | None, str | None, bool, int]:
        self.review_calls.append(text)
        return self._review_answer, None, False, 1


def _make_review_scenario(
    tmp_path: Path,
    review_answer: str = "keep",
) -> tuple[Loop, _ReviewOptimizer, Candidate, Candidate]:
    """Build a Loop with a structural candidate and injected smoke results so
    _smoke_review_candidates can exercise the full path."""
    runs_root = tmp_path / "run"
    mode = _mode()
    edit = _structural_edit(mode)
    opt = _ReviewOptimizer(edit, review_answer)
    loop = Loop(MODE, _config(runs_root, epochs=1), project=Path("."), optimizer=opt)
    loop.init()
    seed = loop.population[0]

    parent = seed
    proposal = Proposal(edit=edit, parent=parent.id, rationale="scripted")
    from refactory_dsh.outerloop import Run
    run = Run(_loop=loop)
    candidate = run.attempt(proposal, parent)
    assert isinstance(candidate, Candidate), f"expected Candidate, got {candidate}"

    new_node_ids = set(candidate.state.get("graph", {}).get("nodes", {}).keys()) - \
        set(parent.state.get("graph", {}).get("nodes", {}).keys())
    assert new_node_ids, "structural edit should add a node"
    new_node = next(iter(new_node_ids))

    result = RunResult(item_records=[{
        "id": "i1", "passed": True, "score": 1.0,
        "steps": [{"node": new_node, "kind": "op",
                    "output": "x" * (FACT_SIZE_WARN + 100), "empty": False}],
        "exchanges": [],
    }])
    loop._smoke_results[candidate.id] = (result, tmp_path)
    return loop, opt, parent, candidate


def test_review_keep_proceeds(tmp_path: Path):
    """A 'keep' reply lets the candidate proceed to evaluation."""
    loop, opt, parent, candidate = _make_review_scenario(tmp_path, "keep")
    from refactory_dsh.outerloop import Run
    run = Run(_loop=loop)
    proposal = Proposal(edit=candidate.own_edits()[0], parent=parent.id, rationale="scripted")
    smoke_meta = {candidate.id: (proposal, parent)}
    result = loop._smoke_review_candidates(run, [candidate], smoke_meta)
    assert len(result) == 1
    assert result[0].id == candidate.id
    assert any("Smoke output review" in c for c in opt.review_calls)


def test_review_drop_withdraws(tmp_path: Path):
    """A 'drop' reply withdraws the proposal."""
    loop, opt, parent, candidate = _make_review_scenario(tmp_path, "drop")
    from refactory_dsh.outerloop import Run
    run = Run(_loop=loop)
    proposal = Proposal(edit=candidate.own_edits()[0], parent=parent.id, rationale="scripted")
    smoke_meta = {candidate.id: (proposal, parent)}
    result = loop._smoke_review_candidates(run, [candidate], smoke_meta)
    assert len(result) == 0
    dropped = [r for r in loop.rejections if any("smoke-review" in s for s in r.reasons)]
    assert dropped, "the dropped proposal was not recorded as a rejection"


def test_review_revised_goes_through_checks(tmp_path: Path):
    """A revised reply goes through the same checks and smoke run as any proposal."""
    revised_answer = json.dumps({"edit": {"op": "set_knob", "name": "order", "value": "asc"}})
    loop, opt, parent, candidate = _make_review_scenario(tmp_path, revised_answer)
    from refactory_dsh.outerloop import Run
    run = Run(_loop=loop)
    proposal = Proposal(edit=candidate.own_edits()[0], parent=parent.id, rationale="scripted")
    smoke_meta = {candidate.id: (proposal, parent)}
    result = loop._smoke_review_candidates(run, [candidate], smoke_meta)
    assert len(result) == 1


def test_review_at_most_one_revision(tmp_path: Path):
    """At most one revision per proposal: the second reply is not asked for."""
    revised_answer = json.dumps({"edit": {"op": "set_knob", "name": "order", "value": "asc"}})
    loop, opt, parent, candidate = _make_review_scenario(tmp_path, revised_answer)
    from refactory_dsh.outerloop import Run
    run = Run(_loop=loop)
    proposal = Proposal(edit=candidate.own_edits()[0], parent=parent.id, rationale="scripted")
    smoke_meta = {candidate.id: (proposal, parent)}
    loop._smoke_review_candidates(run, [candidate], smoke_meta)
    review_calls = [c for c in opt.review_calls if "Smoke output review" in c]
    assert len(review_calls) <= 1, (
        f"the proposer was asked for a review {len(review_calls)} times; at most 1 expected"
    )


def test_review_with_huge_fact_shows_size_warning(tmp_path: Path):
    """A proposal adding a step with huge output: the proposer sees the size warning."""
    loop, opt, parent, candidate = _make_review_scenario(tmp_path, "keep")
    from refactory_dsh.outerloop import Run
    run = Run(_loop=loop)
    proposal = Proposal(edit=candidate.own_edits()[0], parent=parent.id, rationale="scripted")
    smoke_meta = {candidate.id: (proposal, parent)}
    loop._smoke_review_candidates(run, [candidate], smoke_meta)
    review_calls = [c for c in opt.review_calls if "Smoke output review" in c]
    assert review_calls, "the review call should have been made"
    assert "⚠" in review_calls[0], "the size warning should appear in the review block"
    assert f"{FACT_SIZE_WARN:,}" in review_calls[0]


def test_text_only_proposal_no_review(tmp_path: Path):
    """A pure knob edit does not trigger a review."""
    runs_root = tmp_path / "run"
    knob_edit = {"op": "set_knob", "name": "order", "value": "asc"}
    opt = _ReviewOptimizer(knob_edit, "keep")
    loop = Loop(MODE, _config(runs_root, epochs=1), project=Path("."), optimizer=opt)
    loop.init()
    seed = loop.population[0]

    proposal = Proposal(edit=knob_edit, parent=seed.id, rationale="scripted")
    from refactory_dsh.outerloop import Run
    run = Run(_loop=loop)
    candidate = run.attempt(proposal, seed)
    assert isinstance(candidate, Candidate)

    result = RunResult(item_records=[{
        "id": "i1", "passed": True, "score": 1.0,
        "steps": [{"node": "accept", "kind": "op", "output": "ok", "empty": False}],
        "exchanges": [],
    }])
    loop._smoke_results[candidate.id] = (result, tmp_path)
    smoke_meta = {candidate.id: (proposal, seed)}
    reviewed = loop._smoke_review_candidates(run, [candidate], smoke_meta)
    assert len(reviewed) == 1
    assert not opt.review_calls, "a text-only edit should not trigger a review"


def test_smoke_failure_path_unchanged(tmp_path: Path):
    """A candidate with no smoke result (failure) is passed through without review."""
    runs_root = tmp_path / "run"
    mode = _mode()
    edit = _structural_edit(mode)
    opt = _ReviewOptimizer(edit, "keep")
    loop = Loop(MODE, _config(runs_root, epochs=1), project=Path("."), optimizer=opt)
    loop.init()
    seed = loop.population[0]

    proposal = Proposal(edit=edit, parent=seed.id, rationale="scripted")
    from refactory_dsh.outerloop import Run
    run = Run(_loop=loop)
    candidate = run.attempt(proposal, seed)
    assert isinstance(candidate, Candidate)

    smoke_meta = {candidate.id: (proposal, seed)}
    reviewed = loop._smoke_review_candidates(run, [candidate], smoke_meta)
    assert len(reviewed) == 1, "candidate with no smoke result should pass through"
    assert not opt.review_calls, "no review call should be made without smoke results"


# ── mutation checks ──────────────────────────────────────────────────────


def test_mutation_removing_review_call_loses_the_review():
    """Removing the _smoke_review_candidates call makes this test fail."""
    import inspect

    from refactory_dsh.loop import Loop as LoopClass
    source = inspect.getsource(LoopClass.install)
    assert "_smoke_review_candidates" in source, (
        "Loop.install must call _smoke_review_candidates; "
        "removing it means no proposal gets its smoke output reviewed"
    )


def test_mutation_size_flag_is_checked():
    """Breaking the size warning flag makes this test fail."""
    from refactory_dsh.optim.smoke_review import FACT_SIZE_WARN as threshold

    assert threshold > 0, "FACT_SIZE_WARN must be a positive number"
    parent = Candidate.from_state({
        "graph": {"nodes": {"a": {"id": "a", "kind": "op"}}},
        "knobs": {},
    })
    child = Candidate.from_state({
        "graph": {"nodes": {
            "a": {"id": "a", "kind": "op"},
            "b": {"id": "b", "kind": "op"},
        }},
        "knobs": {},
    })
    huge = "x" * (threshold + 100)
    result = RunResult(item_records=[{
        "id": "i1", "passed": True, "score": 1.0,
        "steps": [{"node": "b", "kind": "op", "output": huge, "empty": False}],
        "exchanges": [],
    }])
    block = build_review_block(parent, child, result)
    assert block is not None
    assert "⚠" in block, (
        "the size warning flag must appear when output exceeds FACT_SIZE_WARN; "
        "breaking the flag means the proposer is not warned about huge outputs"
    )
