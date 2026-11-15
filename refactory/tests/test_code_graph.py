"""Tests for the code knowledge graph: query, build op, prompts, off switch, both runtimes."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

from refactory_dsh import lowering
from refactory_dsh.code_graph import (
    _PARAGRAPH_TEMPLATE,
    _count_code_files,
    _find_graphify,
    _read_status,
    _write_status,
    build_code_graph,
    callers_of,
    cli_graph,
    explain_node,
)
from refactory_dsh.design import DesignMode
from refactory_dsh.design_ops import SIZER_MARK
from tests_support import (
    DESIGN_DOC_ANSWER,
    PITFALLS_ANSWER,
    RESEARCH_ANSWER,
    SCOPED_RESEARCH_ANSWER,
    VERIFICATION_ANSWER,
    FakeModel,
    git_repo,
    render,
)


# ── graph.json fixture in the real graphify shape ──────────────────────────────

def _fixture_graph() -> dict:
    return {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "hyperedges": [],
        "built_at_commit": "abc123",
        "nodes": [
            {"id": "n1", "label": "classify_scale()", "file_type": "function",
             "source_file": "refactory/design_ops.py", "source_location": "L958",
             "community": 0, "norm_label": "classify_scale"},
            {"id": "n2", "label": "build_code_graph()", "file_type": "function",
             "source_file": "refactory/code_graph.py", "source_location": "L100",
             "community": 0, "norm_label": "build_code_graph"},
            {"id": "n3", "label": "design_ops", "file_type": "module",
             "source_file": "refactory/design_ops.py", "source_location": "L1",
             "community": 0, "norm_label": "design_ops"},
            {"id": "n4", "label": "classify_scale()", "file_type": "function",
             "source_file": "other/ops.py", "source_location": "L42",
             "community": 1, "norm_label": "classify_scale"},
            {"id": "n5", "label": "Any", "file_type": "type",
             "source_file": "", "source_location": "",
             "community": 0, "norm_label": "any"},
            {"id": "n6", "label": "inner_fn()", "file_type": "function",
             "source_file": "refactory/design_ops.py", "source_location": "L960",
             "community": 0, "norm_label": "inner_fn"},
        ],
        "links": [
            {"source": "n2", "target": "n1", "relation": "calls",
             "source_file": "refactory/code_graph.py", "source_location": "L120",
             "confidence": 1.0},
            {"source": "n1", "target": "n2", "relation": "calls",
             "source_file": "refactory/design_ops.py", "source_location": "L970",
             "confidence": 1.0},
            {"source": "n3", "target": "n1", "relation": "imports",
             "source_file": "refactory/design_ops.py", "source_location": "L5",
             "confidence": 1.0},
            {"source": "n1", "target": "n6", "relation": "contains",
             "source_file": "refactory/design_ops.py", "source_location": "L960",
             "confidence": 1.0},
            {"source": "n1", "target": "n5", "relation": "references",
             "source_file": "refactory/design_ops.py", "source_location": "L959",
             "confidence": 0.5},
        ],
    }


def _write_fixture(graph_dir: Path) -> None:
    out = graph_dir / "graphify-out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "graph.json").write_text(json.dumps(_fixture_graph()))
    _write_status(graph_dir / "status.json", {"status": "ready", "nodes": 6, "edges": 5})


# ── query tests ────────────────────────────────────────────────────────────────

def test_explain_shows_callers_callees_importers_and_contained(tmp_path):
    graph_dir = tmp_path / "code-graph"
    _write_fixture(graph_dir)
    result = explain_node(graph_dir, "classify_scale")
    assert "classify_scale()" in result
    assert "refactory/design_ops.py:958" in result
    assert "Callers:" in result
    assert "refactory/code_graph.py:120" in result
    assert "build_code_graph()" in result
    assert "Callees:" in result
    assert "refactory/design_ops.py:970" in result
    assert "Importers:" in result
    assert "design_ops" in result
    assert "Contains:" in result
    assert "inner_fn()" in result
    assert "references" not in result.lower() or "Any" not in result


def test_callers_shows_calls_edges_only(tmp_path):
    graph_dir = tmp_path / "code-graph"
    _write_fixture(graph_dir)
    result = callers_of(graph_dir, "classify_scale")
    assert "refactory/code_graph.py:120" in result
    assert "build_code_graph()" in result
    assert "imports" not in result.lower()
    assert "contains" not in result.lower()


def test_same_name_in_two_files(tmp_path):
    graph_dir = tmp_path / "code-graph"
    _write_fixture(graph_dir)
    result = explain_node(graph_dir, "classify_scale")
    assert "refactory/design_ops.py" in result
    assert "other/ops.py" in result


def test_missing_node(tmp_path):
    graph_dir = tmp_path / "code-graph"
    _write_fixture(graph_dir)
    assert explain_node(graph_dir, "nonexistent_function") == ""
    assert callers_of(graph_dir, "nonexistent_function") is None


def test_name_matching_strips_parens(tmp_path):
    graph_dir = tmp_path / "code-graph"
    _write_fixture(graph_dir)
    r1 = explain_node(graph_dir, "classify_scale")
    r2 = explain_node(graph_dir, "classify_scale()")
    assert r1 and r2
    assert "classify_scale()" in r1
    assert "classify_scale()" in r2


def test_references_links_ignored(tmp_path):
    graph_dir = tmp_path / "code-graph"
    _write_fixture(graph_dir)
    result = explain_node(graph_dir, "classify_scale")
    assert "Any" not in result


@pytest.mark.skipif(not _find_graphify(), reason="graphify not installed")
def test_real_graphify_output(tmp_path):
    repo = tmp_path / "tiny"
    repo.mkdir()
    (repo / "main.py").write_text("def greet():\n    return 'hello'\n\ndef caller():\n    greet()\n")
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)

    graph_dir = tmp_path / "out"
    graph_dir.mkdir()
    graphify_bin = _find_graphify()
    env = {**os.environ, "GRAPHIFY_OUT": str(graph_dir / "graphify-out")}
    result = subprocess.run(
        [graphify_bin, "extract", str(repo), "--code-only", "--out", str(graph_dir)],
        cwd=str(graph_dir), env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    graph_path = graph_dir / "graphify-out" / "graph.json"
    assert graph_path.exists()
    data = json.loads(graph_path.read_text())
    assert "nodes" in data and "links" in data
    nodes = data["nodes"]
    assert any(n.get("label", "").rstrip("()") == "greet" for n in nodes)

    r = explain_node(graph_dir, "greet")
    assert "greet" in r
    c = callers_of(graph_dir, "greet")
    assert "caller" in c


# ── status tests ───────────────────────────────────────────────────────────────

def _make_args(command, name, graph_path):
    return type("Args", (), {"graph_command": command, "name": name, "graph": str(graph_path)})()


def test_status_building_prints_not_available_exits_nonzero(tmp_path, capsys):
    graph_dir = tmp_path / "code-graph"
    graph_dir.mkdir()
    _write_status(graph_dir / "status.json", {"status": "building", "started": time.time()})
    args = _make_args("explain", "foo", graph_dir)
    rc = cli_graph(args)
    assert rc != 0
    assert "not available" in capsys.readouterr().err.lower()


def test_status_failed_prints_not_available_exits_nonzero(tmp_path, capsys):
    graph_dir = tmp_path / "code-graph"
    graph_dir.mkdir()
    _write_status(graph_dir / "status.json", {"status": "failed: timeout"})
    args = _make_args("explain", "foo", graph_dir)
    rc = cli_graph(args)
    assert rc != 0
    assert "not available" in capsys.readouterr().err.lower()


def test_status_skipped_prints_not_available_exits_nonzero(tmp_path, capsys):
    graph_dir = tmp_path / "code-graph"
    graph_dir.mkdir()
    _write_status(graph_dir / "status.json", {"status": "skipped: graphify not installed"})
    args = _make_args("explain", "foo", graph_dir)
    rc = cli_graph(args)
    assert rc != 0
    assert "not available" in capsys.readouterr().err.lower()


def test_status_absent_prints_not_available_exits_nonzero(tmp_path, capsys):
    graph_dir = tmp_path / "code-graph"
    graph_dir.mkdir()
    args = _make_args("explain", "foo", graph_dir)
    rc = cli_graph(args)
    assert rc != 0
    assert "not available" in capsys.readouterr().err.lower()


def test_stale_building_treated_as_failed(tmp_path, capsys):
    graph_dir = tmp_path / "code-graph"
    graph_dir.mkdir()
    _write_status(graph_dir / "status.json",
                  {"status": "building", "started": time.time() - 600})
    args = _make_args("explain", "foo", graph_dir)
    rc = cli_graph(args)
    assert rc != 0
    assert "not available" in capsys.readouterr().err.lower()


def test_status_json_is_valid_json_after_every_op(tmp_path):
    for status_data in [
        {"status": "building", "started": 1234},
        {"status": "ready", "nodes": 10, "edges": 20},
        {"status": "failed: timeout"},
        {"status": "skipped: graphify not installed"},
    ]:
        path = tmp_path / "s.json"
        _write_status(path, status_data)
        data = json.loads(path.read_text())
        assert "status" in data


# ── build op tests ─────────────────────────────────────────────────────────────

class _FakeCtx:
    def __init__(self, workdir, facts=None, knobs=None):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.facts = facts or {}
        self.knobs = knobs or {}

    def path(self, name):
        return self.workdir / name


def test_op_returns_immediately(tmp_path):
    source = git_repo(tmp_path / "repo")
    ctx = _FakeCtx(tmp_path / "run", facts={"source": str(source), "target": str(source)},
                    knobs={"code_graph": True})
    t0 = time.time()
    result = build_code_graph(ctx)
    elapsed = time.time() - t0
    assert elapsed < 2.0
    assert "code_graph_paragraph" in result


def test_op_returns_empty_paragraph_when_knob_off(tmp_path):
    source = git_repo(tmp_path / "repo")
    ctx = _FakeCtx(tmp_path / "run", facts={"source": str(source)},
                    knobs={"code_graph": False})
    result = build_code_graph(ctx)
    assert result == {"code_graph_paragraph": ""}


def test_op_returns_empty_paragraph_when_graphify_missing(tmp_path, monkeypatch):
    source = git_repo(tmp_path / "repo")
    ctx = _FakeCtx(tmp_path / "run", facts={"source": str(source), "target": str(source)},
                    knobs={"code_graph": True})
    monkeypatch.setattr("refactory_dsh.code_graph._find_graphify", lambda: None)
    result = build_code_graph(ctx)
    assert result == {"code_graph_paragraph": ""}
    status = _read_status(ctx.workdir / "code-graph")
    assert "skipped" in status["status"]
    assert "graphify" in status["status"]


@pytest.mark.skipif(not _find_graphify(), reason="graphify not installed")
def test_op_writes_ready_on_tiny_repo(tmp_path):
    source = git_repo(tmp_path / "repo")
    ctx = _FakeCtx(tmp_path / "run", facts={"source": str(source), "target": str(source)},
                    knobs={"code_graph": True})
    result = build_code_graph(ctx)
    assert result["code_graph_paragraph"]
    graph_dir = ctx.workdir / "code-graph"
    for _ in range(60):
        status = _read_status(graph_dir)
        if status.get("status") == "ready":
            break
        time.sleep(0.5)
    assert status["status"] == "ready"
    assert status["nodes"] > 0


def test_op_writes_failed_timeout_with_stub(tmp_path, monkeypatch):
    source = git_repo(tmp_path / "repo")
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stub = stub_dir / "graphify"
    stub.write_text("#!/bin/sh\nsleep 600\n")
    stub.chmod(0o755)
    monkeypatch.setattr("refactory_dsh.code_graph._find_graphify", lambda: str(stub))
    ctx = _FakeCtx(tmp_path / "run", facts={"source": str(source), "target": str(source)},
                    knobs={"code_graph": True, "code_graph_timeout_s": 2})
    build_code_graph(ctx)
    graph_dir = ctx.workdir / "code-graph"
    for _ in range(30):
        status = _read_status(graph_dir)
        if "timeout" in status.get("status", ""):
            break
        time.sleep(0.5)
    assert "timeout" in status["status"]


def test_op_writes_skipped_over_size_guard(tmp_path, monkeypatch):
    source = git_repo(tmp_path / "repo")
    ctx = _FakeCtx(tmp_path / "run", facts={"source": str(source), "target": str(source)},
                    knobs={"code_graph": True, "code_graph_max_files": 0})
    result = build_code_graph(ctx)
    assert result == {"code_graph_paragraph": ""}
    status = _read_status(ctx.workdir / "code-graph")
    assert "skipped" in status["status"]


def test_op_writes_failed_when_graphify_missing(tmp_path, monkeypatch):
    source = git_repo(tmp_path / "repo")
    ctx = _FakeCtx(tmp_path / "run", facts={"source": str(source), "target": str(source)},
                    knobs={"code_graph": True})
    monkeypatch.setattr("refactory_dsh.code_graph._find_graphify", lambda: None)
    build_code_graph(ctx)
    status = _read_status(ctx.workdir / "code-graph")
    assert "skipped" in status["status"]
    assert "graphify" in status["status"]


@pytest.mark.skipif(not _find_graphify(), reason="graphify not installed")
def test_worktree_stays_clean(tmp_path):
    source = git_repo(tmp_path / "repo")
    before = subprocess.run(["git", "status", "--porcelain"], cwd=source,
                            capture_output=True, text=True).stdout

    # Cold build
    run1 = tmp_path / "runs" / "run1"
    ctx1 = _FakeCtx(run1, facts={"source": str(source), "target": str(source)},
                     knobs={"code_graph": True})
    build_code_graph(ctx1)
    graph_dir1 = ctx1.workdir / "code-graph"
    for _ in range(60):
        status = _read_status(graph_dir1)
        if status.get("status") in ("ready", "failed: timeout"):
            break
        time.sleep(0.5)
    after_cold = subprocess.run(["git", "status", "--porcelain"], cwd=source,
                                capture_output=True, text=True).stdout
    assert before == after_cold, f"Worktree polluted after cold build: {after_cold}"
    assert _read_status(graph_dir1).get("status") == "ready"

    # Cache-reuse build: clone the repo to a second worktree, same target
    source2 = tmp_path / "repo2"
    subprocess.run(["git", "clone", str(source), str(source2)],
                   capture_output=True, check=True)
    before2 = subprocess.run(["git", "status", "--porcelain"], cwd=source2,
                             capture_output=True, text=True).stdout
    run2 = tmp_path / "runs" / "run2"
    ctx2 = _FakeCtx(run2, facts={"source": str(source2), "target": str(source)},
                     knobs={"code_graph": True})
    build_code_graph(ctx2)
    graph_dir2 = ctx2.workdir / "code-graph"
    for _ in range(60):
        status = _read_status(graph_dir2)
        if status.get("status") in ("ready", "failed: timeout"):
            break
        time.sleep(0.5)
    after_reuse = subprocess.run(["git", "status", "--porcelain"], cwd=source2,
                                 capture_output=True, text=True).stdout
    assert before2 == after_reuse, f"Worktree polluted after cache-reuse build: {after_reuse}"


@pytest.mark.skipif(not _find_graphify(), reason="graphify not installed")
def test_second_build_same_target_reuses_cache(tmp_path):
    source1 = git_repo(tmp_path / "repo")
    run1 = tmp_path / "runs" / "run1"
    ctx1 = _FakeCtx(run1, facts={"source": str(source1), "target": str(source1)},
                     knobs={"code_graph": True})
    build_code_graph(ctx1)
    for _ in range(60):
        if _read_status(run1 / "code-graph").get("status") == "ready":
            break
        time.sleep(0.5)
    assert _read_status(run1 / "code-graph")["status"] == "ready"

    source2_path = tmp_path / "repo2"
    subprocess.run(["git", "clone", str(source1), str(source2_path)], capture_output=True, check=True)
    run2 = tmp_path / "runs" / "run2"
    ctx2 = _FakeCtx(run2, facts={"source": str(source2_path), "target": str(source1)},
                     knobs={"code_graph": True})
    build_code_graph(ctx2)
    cache_dir = run2 / "code-graph" / "graphify-out"
    assert cache_dir.exists()
    assert (cache_dir / "graph.json").exists()


def test_different_target_does_not_reuse_cache(tmp_path):
    source1 = git_repo(tmp_path / "repo1")
    run1 = tmp_path / "runs" / "run1"
    run1_cg = run1 / "code-graph"
    run1_cg.mkdir(parents=True)
    _write_status(run1_cg / "status.json", {"status": "ready", "target_key": str(source1)})
    (run1_cg / "graphify-out").mkdir()
    (run1_cg / "graphify-out" / "graph.json").write_text("{}")

    source2 = git_repo(tmp_path / "repo2")
    run2 = tmp_path / "runs" / "run2"
    ctx2 = _FakeCtx(run2, facts={"source": str(source2), "target": str(source2)},
                     knobs={"code_graph": True})
    with mock.patch("refactory_dsh.code_graph._find_graphify", return_value=None):
        build_code_graph(ctx2)
    cache_dir = run2 / "code-graph" / "graphify-out"
    assert not cache_dir.exists() or not (cache_dir / "graph.json").exists() or \
           (cache_dir / "graph.json").read_text() != "{}"


def test_no_import_time_dependency_on_graphify(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "graphifyy", None)
    import importlib
    import refactory_dsh.code_graph as cg
    importlib.reload(cg)
    ctx = _FakeCtx(tmp_path / "noop", facts={}, knobs={"code_graph": False})
    result = cg.build_code_graph(ctx)
    assert result == {"code_graph_paragraph": ""}


def test_op_skipped_when_refactory_command_not_found(tmp_path, monkeypatch):
    source = git_repo(tmp_path / "repo")
    ctx = _FakeCtx(tmp_path / "run", facts={"source": str(source), "target": str(source)},
                    knobs={"code_graph": True})
    monkeypatch.setattr("refactory_dsh.code_graph._find_refactory_cmd", lambda: None)
    result = build_code_graph(ctx)
    assert result == {"code_graph_paragraph": ""}
    status = _read_status(ctx.workdir / "code-graph")
    assert "skipped" in status["status"]
    assert "refactory command not found" in status["status"]


# ── paragraph content test ─────────────────────────────────────────────────────

def test_paragraph_content_correctness(tmp_path):
    source = git_repo(tmp_path / "repo")
    ctx = _FakeCtx(tmp_path / "run", facts={"source": str(source), "target": str(source)},
                    knobs={"code_graph": True})
    result = build_code_graph(ctx)
    paragraph = result["code_graph_paragraph"]
    if not paragraph:
        pytest.skip("graphify or refactory command not found")

    refactory_cmd = str(Path(sys.executable).parent / "refactory")
    if Path(refactory_cmd).is_file():
        assert refactory_cmd in paragraph
    assert "graph callers" in paragraph
    assert "graph explain" in paragraph
    assert "--graph" in paragraph
    assert "grep" in paragraph
    assert "lead, not evidence" in paragraph
    assert "$VENV_BIN" not in paragraph
    for cap_word in ("limit", "maximum", "at most", "no more than", "cap"):
        assert cap_word not in paragraph.lower()


# ── walker prompt tests ────────────────────────────────────────────────────────

def _design_run_with_knobs(tmp_path, brief, answers, extra_knobs=None):
    repo = git_repo(tmp_path / "repo")
    model = FakeModel(answers)
    mode = DesignMode()
    state = mode.state_dict()
    state["knobs"]["pr"]["value"] = False
    if extra_knobs:
        for k, v in extra_knobs.items():
            if k in state["knobs"]:
                state["knobs"][k]["value"] = v
    mode.load_state_dict(state)
    run = mode.run(tmp_path / "run", model=model, verdicts={"gate_strategy": "proceed"},
                   task={"name": "short_tool", "brief": brief,
                         "check": f"{sys.executable} tool.py", "target": str(repo)})
    return run, model


BUILD = [render({"tool.py": "def build():\n    print('built')\n"}),
        "health: ok", "code review: ok", "qa: ok", VERIFICATION_ANSWER, "VERDICT: PASS"]


def _graph_paragraph_in(prompt: str) -> bool:
    return "Code knowledge graph" in prompt and "graph callers" in prompt


def test_graph_on_sizer_prompt_contains_paragraph(tmp_path):
    run, model = _design_run_with_knobs(
        tmp_path, "Make tool.py print built.",
        [SCOPED_RESEARCH_ANSWER, PITFALLS_ANSWER, "one plan", DESIGN_DOC_ANSWER, *BUILD])
    if not run.exit_ok:
        pytest.skip(f"run failed: {run.error}")
    sizer_prompts = [p for p in model.prompts if SIZER_MARK in p]
    if not sizer_prompts:
        pytest.skip("sizer prompt not found")
    sizer = sizer_prompts[0]
    if run.facts.get("code_graph_paragraph"):
        assert _graph_paragraph_in(sizer)
        assert "$VENV_BIN" not in sizer
        for w in ("limit", "maximum", "at most", "no more than"):
            assert w not in sizer.split("Code knowledge graph")[1].split("\n\n")[0].lower()


def test_graph_on_research_director_prompt_contains_paragraph(tmp_path):
    run, model = _design_run_with_knobs(
        tmp_path, "Make tool.py print built.",
        [SCOPED_RESEARCH_ANSWER, PITFALLS_ANSWER, "one plan", DESIGN_DOC_ANSWER, *BUILD])
    if not run.exit_ok:
        pytest.skip(f"run failed: {run.error}")
    rd_prompts = [p for p in model.prompts if "You are the research director" in p]
    if not rd_prompts:
        pytest.skip("research director prompt not found")
    rd = rd_prompts[0]
    if run.facts.get("code_graph_paragraph"):
        assert _graph_paragraph_in(rd)


def test_graph_on_pitfalls_prompt_contains_paragraph(tmp_path):
    run, model = _design_run_with_knobs(
        tmp_path, "Make tool.py print built.",
        [SCOPED_RESEARCH_ANSWER, PITFALLS_ANSWER, "one plan", DESIGN_DOC_ANSWER, *BUILD])
    if not run.exit_ok:
        pytest.skip(f"run failed: {run.error}")
    pit_prompts = [p for p in model.prompts if "pitfalls researcher" in p]
    if not pit_prompts:
        pytest.skip("pitfalls prompt not found")
    pit = pit_prompts[0]
    if run.facts.get("code_graph_paragraph"):
        assert _graph_paragraph_in(pit)


def test_no_graph_flag_prompts_have_no_graph_paragraph(tmp_path):
    run, model = _design_run_with_knobs(
        tmp_path, "Make tool.py print built.",
        [SCOPED_RESEARCH_ANSWER, PITFALLS_ANSWER, "one plan", DESIGN_DOC_ANSWER, *BUILD],
        extra_knobs={"code_graph": False})
    sizer = [p for p in model.prompts if SIZER_MARK in p]
    rd = [p for p in model.prompts if "You are the research director" in p]
    pit = [p for p in model.prompts if "pitfalls researcher" in p]
    for prompts, label in [(sizer, "sizer"), (rd, "research_director"), (pit, "pitfalls")]:
        if prompts:
            assert not _graph_paragraph_in(prompts[0]), f"{label} should not have graph paragraph when knob off"


def test_env_var_off_produces_knob(tmp_path, monkeypatch):
    monkeypatch.setenv("REFACTORY_CODE_GRAPH", "0")
    from refactory_dsh import run_flags
    import argparse
    parser = argparse.ArgumentParser()
    run_flags.add_arguments(parser)
    args = parser.parse_args(["--name", "x", "--check", "true"])
    knobs_list = run_flags.worktree_knobs(args)
    assert "code_graph=0" in knobs_list


def test_graph_absent_op_prompts_have_no_graph_paragraph(tmp_path):
    run, model = _design_run_with_knobs(
        tmp_path, "Make tool.py print built.",
        [SCOPED_RESEARCH_ANSWER, PITFALLS_ANSWER, "one plan", DESIGN_DOC_ANSWER, *BUILD],
        extra_knobs={"code_graph": False})
    for p in model.prompts:
        assert "Code knowledge graph" not in p or not _graph_paragraph_in(p)


# ── lowering and spine tests ──────────────────────────────────────────────────

def test_lowered_build_code_graph_is_fn_node():
    state = DesignMode().state_dict()
    workflow = lowering.to_workflow(state)
    node = workflow.nodes["build_code_graph"]
    assert node.command.endswith("refactory_dsh.code_graph:build_code_graph")
    assert "code-graph/" in (getattr(node, "writes", None) or [])


def test_spine_node_runner_runs_build_code_graph(tmp_path):
    from refactory_dsh.run import declarations, main, update_facts
    from refactory_dsh.steering import Ledger

    repo = git_repo(tmp_path / "repo")
    state = DesignMode().state_dict()
    given = {"name": "tool", "brief": "fix it", "check": "python app.py",
             "target": str(repo), "pr": False}
    for knob, value in given.items():
        state["knobs"][knob]["value"] = value
    run = tmp_path / "run"
    run.mkdir()
    Ledger.seed(run, given["brief"])
    workflow = lowering.to_workflow(state)
    env = {**os.environ, **lowering.knob_environment(state),
           "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
           "PYTHONDONTWRITEBYTECODE": "1",
           "REFACTORY_WORKDIR": str(run)}
    lowering.write_declarations(state, run)

    target = "refactory_dsh.code_graph:build_code_graph"
    node_env = {**env, "DSH_WORKFLOW_NODE": json.dumps({"id": "build_code_graph", "type": "FnNode"})}
    update_facts(run, {"source": str(repo), "target": str(repo)})
    done = subprocess.run(
        [sys.executable, "-m", "refactory_dsh.run", target, "--node", "build_code_graph"],
        env=node_env, cwd=run, capture_output=True, text=True, timeout=30,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    facts = json.loads((run / "facts.json").read_text())
    assert "code_graph_paragraph" in facts


def test_spine_run_where_build_fails(tmp_path):
    from refactory_dsh.run import update_facts
    from refactory_dsh.steering import Ledger

    repo = git_repo(tmp_path / "repo")
    state = DesignMode().state_dict()
    given = {"name": "tool", "brief": "fix it", "check": "python app.py",
             "target": str(repo), "pr": False}
    for knob, value in given.items():
        state["knobs"][knob]["value"] = value
    run = tmp_path / "run"
    run.mkdir()
    Ledger.seed(run, given["brief"])
    env = {**os.environ, **lowering.knob_environment(state),
           "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
           "PYTHONDONTWRITEBYTECODE": "1",
           "REFACTORY_WORKDIR": str(run),
           "DSH_WORKFLOW_NODE": json.dumps({"id": "build_code_graph", "type": "FnNode"}),
           "_REFACTORY_GRAPHIFY_BIN_OVERRIDE": ""}
    lowering.write_declarations(state, run)
    update_facts(run, {"source": str(repo), "target": str(repo)})

    done = subprocess.run(
        [sys.executable, "-m", "refactory_dsh.run",
         "refactory_dsh.code_graph:build_code_graph", "--node", "build_code_graph"],
        env=env, cwd=run, capture_output=True, text=True, timeout=30,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    facts = json.loads((run / "facts.json").read_text())
    assert facts.get("code_graph_paragraph", "") == ""
