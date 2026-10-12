"""Controlled native recovery tests; these are not benchmark acceptance evidence."""
import json
from types import SimpleNamespace

import pytest

from benchmarks.create_fidelity import approval, lane
from refactory_dsh import nn


def baseline_inputs(ctx):
    ctx.write_json("strategy.json", {"chosen": "controlled strategy"})
    ctx.write_json("brief.json", {"brief": ctx.knob("brief")})


def baseline_done(ctx):
    ctx.write_text("done.txt", "approved")


class ControlledBaseline(nn.Mode):
    name = "create"
    unscored_reason = "Controlled approval adapter fixture"

    @classmethod
    def declare(cls):
        return {name: nn.Knob(value, kind) for name, value, kind in (
            ("name", "", "str"), ("brief", "", "str"),
            ("single_shot", False, "bool"), ("requires_input", "", "str"),
        )}

    def build(self):
        graph = nn.Graph()
        graph.add(nn.Node("init", "op", call=f"{__name__}:baseline_inputs",
                          knobs=("brief",), writes=("strategy.json", "brief.json")))
        graph.add(nn.Node("gate_strategy", "gate", gate_kind="user", reads=("strategy.json",)))
        graph.add(nn.Node("done", "op", call=f"{__name__}:baseline_done", writes=("done.txt",)))
        graph.connect("init", "gate_strategy")
        graph.connect("gate_strategy", "done", condition="proceed")
        graph.connect("gate_strategy", "init", condition="reloop")
        return graph

    @classmethod
    def resume(cls, *args, **kwargs):
        raise AssertionError("Pinned baseline needs its documented start/facts API")


def test_baseline_approval_binds_its_native_files(tmp_path):
    (tmp_path / "strategy.json").write_text('{"chosen":"one"}')
    (tmp_path / "brief.json").write_text('{"brief":"original"}')
    pending = approval.request(tmp_path, "Review", inputs=("strategy.json", "brief.json"))
    assert pending["input_paths"] == ["strategy.json", "brief.json"]
    (tmp_path / "brief.json").write_text('{"brief":"changed"}')
    with pytest.raises(ValueError, match="changed"):
        approval.deliver(tmp_path, digest=pending["strategy_digest"], verdict="proceed",
                         approver="controlled human fixture", feedback="")


def test_baseline_native_resume_consumes_fresh_answer(tmp_path, monkeypatch):
    from refactory_dsh import model, registry
    from refactory_dsh.steering import Ledger

    # The real lane owns a subprocess; register its process-local environment
    # setting with pytest so this in-process adapter exercise cannot leak it.
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    (tmp_path / "BRIEF.md").write_text("controlled fixture brief")
    monkeypatch.setattr(registry, "load", lambda *args: SimpleNamespace(mode=lambda _: ControlledBaseline))
    monkeypatch.setattr(model, "DshModelClient", lambda **kwargs: SimpleNamespace(route=lambda: {}))
    original_take = approval.take

    def controlled_human_answer(project, pending):
        assert not (project / "done.txt").exists()
        approval.deliver(project, digest=pending["strategy_digest"], verdict="proceed",
                         approver="controlled human fixture", feedback="")
        return original_take(project, pending)

    monkeypatch.setattr(approval, "take", controlled_human_answer)
    result = lane.dsh_author({"mode": "create", "model_context": {}, "total_timeout_s": 60}, tmp_path)
    assert result["exit_ok"], result
    assert (tmp_path / "done.txt").read_text() == "approved"
    assert any(entry["kind"] == "verdict-use" for entry in Ledger(tmp_path).entries())
    resumes = list((tmp_path / "raw").glob("resume-*.json"))
    assert len(resumes) == 1
    assert json.loads(resumes[0].read_text())["node"] == "gate_strategy"
    assert not (tmp_path / "human-answer.json").exists()
    pending = json.loads((tmp_path / "approval-required.json").read_text())
    assert pending["status"] == "consumed"
    with pytest.raises(ValueError, match="already consumed"):
        approval.deliver(tmp_path, digest=pending["strategy_digest"], verdict="proceed",
                         approver="controlled human fixture", feedback="")
