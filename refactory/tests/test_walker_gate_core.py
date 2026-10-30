"""End-to-end gate tests through the real CLI commands.

Unlike test_walker_gate.py which calls execute() directly and writes the ledger
by hand, these tests run design.main / console.main / create.main / edit.main
with a scripted FakeModel, so they exercise Mode.resume, console steer and the
top-level GateStop/HaltStop arms at executor.py:579-588.

Mutation check — each of these tests fails under at least one of the two
mutations:

  * Removing the top-level GateStop/HaltStop exception arms
    (executor.py:579-588) causes the gate exception to propagate as an unhandled
    error, so progress.json never records outcome="refused" and the proceed
    test's resume never sees a matching gate.

  * Reverting the graph_identity comparison at mode.py:323 to the identity
    function causes the steering-baked-in prompt templates to trigger a graph
    mismatch on resume, raising ValueError instead of completing.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from tests_support import (
    DESIGN_DOC_ANSWER,
    FakeModel,
    PITFALLS_ANSWER,
    RESEARCH_ANSWER,
    REVIEW_ANSWERS,
    VERIFICATION_ANSWER,
    create_answers,
    good_files,
    render,
)

# ── answer sequences ──────────────────────────────────────────────────────

PRE_GATE = [
    RESEARCH_ANSWER,
    PITFALLS_ANSWER,
    "perspectives",
    "one plan",
    DESIGN_DOC_ANSWER,
]

POST_GATE_PROCEED = [
    render({"tool.py": "print('built')\n"}),
    *REVIEW_ANSWERS[:3],
    VERIFICATION_ANSWER,
    REVIEW_ANSWERS[3],
]

POST_GATE_RELOOP = [
    '{"scope": "approach", "reason": "probe reloop"}',  # classify_reloop
    "one plan",
    DESIGN_DOC_ANSWER,
]


# ── monkeypatch helper ────────────────────────────────────────────────────

def _patch_model(monkeypatch, fake):
    """Replace DshModelClient so design.main()'s constructor returns *fake*.

    Using a class whose ``__new__`` returns the fake avoids the TypeError
    that a lambda causes when design_ops.py:177 calls
    ``isinstance(client, DshModelClient)``.
    """
    class _StandIn:
        def __new__(cls, **kwargs):
            return fake
    monkeypatch.setattr("refactory_dsh.model.DshModelClient", _StandIn)


# ── design core tests ─────────────────────────────────────────────────────


def test_proceed_with_note(tmp_path, monkeypatch, capsys):
    """design.main → gate_stop → steer proceed+note → resume → done."""
    from refactory_dsh import console, design

    RUN = str(tmp_path / "probe_run")
    fake = FakeModel(list(PRE_GATE))
    _patch_model(monkeypatch, fake)
    monkeypatch.chdir(tmp_path)

    # 1. Fresh run stops at gate_strategy.
    code = design.main([
        "--name", "probe_tool", "--brief", "a tool",
        "--check", "python tool.py", "--no-worktree", "--workdir", RUN,
    ])
    assert code == 2
    progress = json.loads((Path(RUN) / ".refactory/progress.json").read_text())
    assert progress["outcome"] == "refused"
    err = capsys.readouterr().err
    assert "gate_strategy" in err
    assert "halt" in err.lower() or "--resume" in err

    # 2. Steer: proceed with a note.
    fake.answers = [DESIGN_DOC_ANSWER, *POST_GATE_PROCEED]
    code2 = console.main([
        "steer", "--run", RUN,
        "--verdict", "gate_strategy=proceed",
        "--kind", "correction",
        "--note", "PROBE-NOTE use argparse",
    ])
    assert code2 == 0
    out2 = capsys.readouterr().out
    assert "python -m refactory_dsh.design --resume" in out2

    # 3. Resume completes.
    code3 = design.main(["--resume", RUN])
    assert code3 == 0

    # 4. The first prompt after the gate contains PROBE-NOTE.
    assert "PROBE-NOTE" in fake.prompts[6], (
        f"prompts[6] does not contain PROBE-NOTE: {fake.prompts[6][:200]}"
    )


def test_halt(tmp_path, monkeypatch, capsys):
    """design.main → gate_stop → steer halt → resume → halted."""
    from refactory_dsh import console, design

    RUN = str(tmp_path / "probe_run")
    fake = FakeModel(list(PRE_GATE))
    _patch_model(monkeypatch, fake)
    monkeypatch.chdir(tmp_path)

    # 1. Fresh run stops at gate_strategy.
    code = design.main([
        "--name", "probe_tool", "--brief", "a tool",
        "--check", "python tool.py", "--no-worktree", "--workdir", RUN,
    ])
    assert code == 2
    progress = json.loads((Path(RUN) / ".refactory/progress.json").read_text())
    assert progress["outcome"] == "refused"

    # 2. Steer: halt.
    code2 = console.main([
        "steer", "--run", RUN,
        "--verdict", "gate_strategy=halt",
    ])
    assert code2 == 0

    # 3. Resume: exits 2, halted.
    fake.answers = []
    code3 = design.main(["--resume", RUN])
    assert code3 == 2
    progress3 = json.loads((Path(RUN) / ".refactory/progress.json").read_text())
    assert progress3["outcome"] == "halted"


def test_reloop(tmp_path, monkeypatch, capsys):
    """design.main → gate_stop → steer reloop+note → resume → gate again."""
    from refactory_dsh import console, design

    RUN = str(tmp_path / "probe_run")
    fake = FakeModel(list(PRE_GATE))
    _patch_model(monkeypatch, fake)
    monkeypatch.chdir(tmp_path)

    # 1. Fresh run stops at gate_strategy.
    code = design.main([
        "--name", "probe_tool", "--brief", "a tool",
        "--check", "python tool.py", "--no-worktree", "--workdir", RUN,
    ])
    assert code == 2
    progress = json.loads((Path(RUN) / ".refactory/progress.json").read_text())
    assert progress["outcome"] == "refused"

    # 2. Steer: reloop with a note.
    code2 = console.main([
        "steer", "--run", RUN,
        "--verdict", "gate_strategy=reloop",
        "--note", "PROBE-RELOOP split the module",
    ])
    assert code2 == 0

    # 3. Resume: reloops back through strategy, hits gate again.
    fake.answers = list(POST_GATE_RELOOP)
    code3 = design.main(["--resume", RUN])
    assert code3 == 2
    progress3 = json.loads((Path(RUN) / ".refactory/progress.json").read_text())
    assert progress3["outcome"] == "refused"

    # 4. A later prompt contains PROBE-RELOOP.
    assert any("PROBE-RELOOP" in p for p in fake.prompts), (
        "PROBE-RELOOP did not reach any later prompt"
    )


# ── create / edit gate-stop tests ─────────────────────────────────────────


def test_create_gate_stop(tmp_path, monkeypatch, capsys):
    """create.main with a scripted model stops at gate_strategy: exit 2, no traceback."""
    from refactory_dsh import create

    RUN = str(tmp_path / "probe_create")
    answers = create_answers("probe_tool", {"probe_tool/__init__.py": "x = 1\n"})
    fake = FakeModel(answers)
    _patch_model(monkeypatch, fake)
    monkeypatch.chdir(tmp_path)

    # No --check: create declares none, so it is refused by name, as `refactory ceo` refuses it (dm-134).
    code = create.main([
        "--name", "probe_tool", "--brief", "a tool",
        "--single", "--workdir", RUN,
    ])
    assert code == 2
    progress = json.loads((Path(RUN) / ".refactory/progress.json").read_text())
    assert progress["outcome"] == "refused"
    err = capsys.readouterr().err
    assert "gate_strategy" in err
    assert "Traceback" not in err


def test_edit_gate_stop(tmp_path, monkeypatch, capsys):
    """edit.main with a scripted model stops at gate_strategy: exit 2, no traceback."""
    from refactory_dsh import edit, registry
    from refactory_dsh.repo_change import ARTIFACTS, OWNERS

    NAME = "probe_edit_pkg"
    RUN = str(tmp_path / "probe_edit")
    answers = create_answers(NAME, good_files(NAME))
    fake = FakeModel(answers)
    _patch_model(monkeypatch, fake)
    monkeypatch.chdir(tmp_path)

    # Clean up any stale state (mirrors test_edit_design._clean).
    ws_root = registry.workspace_root(Path(".")) / NAME
    shutil.rmtree(ws_root, ignore_errors=True)

    try:
        code = edit.main([
            "--mode", "circle_packing/adaptive",
            "--name", NAME,
            "--brief", "count to a target",
            "--workdir", RUN,
        ])
        assert code == 2
        progress = json.loads((Path(RUN) / ".refactory/progress.json").read_text())
        assert progress["outcome"] == "refused"
        err = capsys.readouterr().err
        assert "gate_strategy" in err
        assert "Traceback" not in err
    finally:
        # Clean up.
        shutil.rmtree(ws_root, ignore_errors=True)
        # Owner records
        owner_file = OWNERS / f"{NAME}.json"
        owner_file.unlink(missing_ok=True)


# ── profile check tests ──────────────────────────────────────────────────


def _prepare_walker_run(tmp_path, monkeypatch, *, worker_backend, headless=None):
    """Run prepare_run for a walker run whose DSH model home holds ``headless`` (or nothing)."""
    from refactory_dsh.run_preparation import prepare_run
    from refactory_dsh.nn import Graph, Node

    fake_home = tmp_path / "dsh_home"
    fake_home.mkdir()
    if headless is not None:
        (fake_home / "profiles" / "headless").mkdir(parents=True)
        for name, text in headless.items():
            (fake_home / "profiles" / "headless" / name).write_text(text)
    monkeypatch.setattr("refactory_dsh.dsh_profile.model_home", lambda: fake_home)
    monkeypatch.setattr("refactory_dsh.recovery.configuration_fingerprints",
                        lambda workdir, mode_id, project: {})
    graph = Graph()
    graph.add(Node("a", "op", call="module:fn"))
    state = {"version": 1, "graph": graph.to_state(), "knobs": {}, "ports": [], "memory": []}
    return prepare_run(
        state, tmp_path / "run", invocation_name="test", mode_id="test/test", intent="test",
        transform=lambda workflow, stage_dir=None: workflow, runtime="walker",
        command="test", resume_command="test", worker_backend=worker_backend,
        project=Path("."))


def test_dsh_missing_profile_allowed(tmp_path, monkeypatch):
    """A DSH-backed run with no headless profile yet is not refused: DSH creates the
    profile from its shipped template on first use, as a fresh install has none."""
    _prepare_walker_run(tmp_path, monkeypatch, worker_backend="dsh")


def test_dsh_broken_profile_refused(tmp_path, monkeypatch):
    """A DSH-backed run whose headless profile directory has no package.json is refused."""
    with pytest.raises(ValueError, match="has no package.json"):
        _prepare_walker_run(tmp_path, monkeypatch, worker_backend="dsh",
                            headless={"cordis.yml": "[]\n"})


def test_fake_model_missing_profile_allowed(tmp_path, monkeypatch):
    """A fake-model run is not checked for the headless profile."""
    from refactory_dsh.run_preparation import prepare_run

    # Make model_home point to a directory with no 'headless' subdirectory.
    fake_home = tmp_path / "dsh_home"
    fake_home.mkdir()
    monkeypatch.setattr("refactory_dsh.dsh_profile.model_home", lambda: fake_home)
    # Stub out downstream steps that need a real mode/project.
    monkeypatch.setattr("refactory_dsh.recovery.configuration_fingerprints",
                        lambda workdir, mode_id, project: {})

    from refactory_dsh.nn import Graph, Node
    graph = Graph()
    graph.add(Node("a", "op", call="module:fn"))
    state = {"version": 1, "graph": graph.to_state(), "knobs": {}, "ports": [], "memory": []}

    # worker_backend="" (non-DSH) — should NOT raise for the profile check.
    prepare_run(
        state, tmp_path / "run",
        invocation_name="test",
        mode_id="test/test",
        intent="test",
        transform=lambda workflow, stage_dir=None: workflow,
        runtime="walker",
        command="test",
        resume_command="test",
        worker_backend="",
        project=Path("."),
    )
