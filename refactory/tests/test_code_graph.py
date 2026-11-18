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
    QUERY_LOG,
    usage,
    GRAPH_SUBDIR,
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
from refactory_dsh.code_graph import EXPLORER_MARK
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
    assert explain_node(graph_dir, "nonexistent_function") is None
    assert callers_of(graph_dir, "nonexistent_function") is None


def test_explain_of_a_missing_name_says_so(tmp_path, capsys):
    graph_dir = tmp_path / "code-graph"
    _write_fixture(graph_dir)
    _write_status(graph_dir / "status.json", {"status": "ready"})
    assert cli_graph(_make_args("explain", "nonexistent_function", graph_dir)) == 0
    assert "No node named 'nonexistent_function'" in capsys.readouterr().out


def test_explain_counts_unconnected_nodes_instead_of_printing_each(tmp_path):
    graph_dir = tmp_path / "code-graph"
    graph = _fixture_graph()
    for i in range(40):
        graph["nodes"].append({"id": f"thing_{i}", "label": "Thing", "source_file": f"m{i}.py",
                               "source_location": "L1"})
    graph["nodes"].append({"id": "thing_used", "label": "Thing", "source_file": "used.py",
                           "source_location": "L3"})
    graph["links"].append({"source": "thing_used", "target": graph["nodes"][0]["id"],
                           "relation": "calls", "source_file": "used.py", "source_location": "L4"})
    (graph_dir / "graphify-out").mkdir(parents=True)
    (graph_dir / "graphify-out" / "graph.json").write_text(json.dumps(graph))
    out = explain_node(graph_dir, "Thing")
    assert out.startswith("41 node(s) named 'Thing'; 1 with connections shown; 40 with none")
    assert "and 35 more" in out
    assert out.count("Thing  ") == 1  # one entry printed, not 41


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


@pytest.mark.real_graphify
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


def test_staleness_uses_the_build_s_own_timeout(tmp_path, capsys):
    graph_dir = tmp_path / "code-graph"
    graph_dir.mkdir()
    # 100 s in: stale for a 30 s build (30 + 60 < 100), still building for the 300 s default.
    _write_status(graph_dir / "status.json",
                  {"status": "building", "started": time.time() - 100, "timeout_s": 30})
    assert cli_graph(_make_args("callers", "foo", graph_dir)) == 1
    assert "stale build" in capsys.readouterr().err


def test_a_build_whose_process_died_is_failed_at_once(tmp_path, capsys):
    graph_dir = tmp_path / "code-graph"
    graph_dir.mkdir()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    _write_status(graph_dir / "status.json", {"status": "building", "started": time.time(),
                                              "timeout_s": 300, "pid": dead.pid})
    assert cli_graph(_make_args("callers", "foo", graph_dir)) == 1
    assert "ended without writing a status" in capsys.readouterr().err


def test_a_build_that_died_as_an_unreaped_child_is_failed(tmp_path, capsys):
    """On the walker the build is the walker's child: dead but unreaped, it is a zombie."""
    graph_dir = tmp_path / "code-graph"
    graph_dir.mkdir()
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        for _ in range(100):
            stat = subprocess.run(["ps", "-o", "stat=", "-p", str(child.pid)],
                                  capture_output=True, text=True).stdout.strip()
            if stat.startswith("Z"):
                break
            time.sleep(0.05)
        assert stat.startswith("Z")
        _write_status(graph_dir / "status.json", {"status": "building", "started": time.time(),
                                                  "timeout_s": 300, "pid": child.pid})
        assert cli_graph(_make_args("callers", "foo", graph_dir)) == 1
        assert "ended without writing a status" in capsys.readouterr().err
    finally:
        child.wait()


def test_a_live_build_is_still_building(tmp_path, capsys):
    graph_dir = tmp_path / "code-graph"
    graph_dir.mkdir()
    _write_status(graph_dir / "status.json", {"status": "building", "started": time.time(),
                                              "timeout_s": 300, "pid": os.getpid()})
    assert cli_graph(_make_args("callers", "foo", graph_dir)) == 1
    assert "(building)" in capsys.readouterr().err


def test_a_graphify_killed_by_a_signal_is_named(tmp_path, monkeypatch):
    source = git_repo(tmp_path / "repo")
    stub = tmp_path / "bin" / "graphify"
    stub.parent.mkdir()
    stub.write_text("#!/bin/sh\nkill -9 $$\n")
    stub.chmod(0o755)
    monkeypatch.setattr("refactory_dsh.code_graph._find_graphify", lambda: str(stub))
    ctx = _FakeCtx(tmp_path / "run", facts={"source": str(source), "target": str(source)},
                   knobs={"code_graph": True})
    build_code_graph(ctx)
    graph_dir = ctx.workdir / GRAPH_SUBDIR
    for _ in range(40):
        status = _read_status(graph_dir)
        if status.get("status", "").startswith("failed"):
            break
        time.sleep(0.25)
    assert status["status"] == "failed: graphify killed by signal 9 (likely out of memory)"


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
    status = _read_status(ctx.workdir / GRAPH_SUBDIR)
    assert "skipped" in status["status"]
    assert "graphify" in status["status"]


@pytest.mark.real_graphify
@pytest.mark.skipif(not _find_graphify(), reason="graphify not installed")
def test_op_writes_ready_on_tiny_repo(tmp_path):
    source = git_repo(tmp_path / "repo")
    ctx = _FakeCtx(tmp_path / "run", facts={"source": str(source), "target": str(source)},
                    knobs={"code_graph": True})
    result = build_code_graph(ctx)
    assert result["code_graph_paragraph"]
    graph_dir = ctx.workdir / GRAPH_SUBDIR
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
    graph_dir = ctx.workdir / GRAPH_SUBDIR
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
    status = _read_status(ctx.workdir / GRAPH_SUBDIR)
    assert "skipped" in status["status"]


def test_op_writes_failed_when_graphify_missing(tmp_path, monkeypatch):
    source = git_repo(tmp_path / "repo")
    ctx = _FakeCtx(tmp_path / "run", facts={"source": str(source), "target": str(source)},
                    knobs={"code_graph": True})
    monkeypatch.setattr("refactory_dsh.code_graph._find_graphify", lambda: None)
    build_code_graph(ctx)
    status = _read_status(ctx.workdir / GRAPH_SUBDIR)
    assert "skipped" in status["status"]
    assert "graphify" in status["status"]


@pytest.mark.real_graphify
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
    graph_dir1 = ctx1.workdir / GRAPH_SUBDIR
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
    graph_dir2 = ctx2.workdir / GRAPH_SUBDIR
    for _ in range(60):
        status = _read_status(graph_dir2)
        if status.get("status") in ("ready", "failed: timeout"):
            break
        time.sleep(0.5)
    after_reuse = subprocess.run(["git", "status", "--porcelain"], cwd=source2,
                                 capture_output=True, text=True).stdout
    assert before2 == after_reuse, f"Worktree polluted after cache-reuse build: {after_reuse}"


@pytest.mark.real_graphify
@pytest.mark.skipif(not _find_graphify(), reason="graphify not installed")
def test_second_build_same_target_reuses_cache(tmp_path):
    source1 = git_repo(tmp_path / "repo")
    run1 = tmp_path / "runs" / "run1"
    ctx1 = _FakeCtx(run1, facts={"source": str(source1), "target": str(source1)},
                     knobs={"code_graph": True})
    build_code_graph(ctx1)
    for _ in range(60):
        if _read_status(run1 / GRAPH_SUBDIR).get("status") == "ready":
            break
        time.sleep(0.5)
    assert _read_status(run1 / GRAPH_SUBDIR)["status"] == "ready"

    source2_path = tmp_path / "repo2"
    subprocess.run(["git", "clone", str(source1), str(source2_path)], capture_output=True, check=True)
    run2 = tmp_path / "runs" / "run2"
    ctx2 = _FakeCtx(run2, facts={"source": str(source2_path), "target": str(source1)},
                     knobs={"code_graph": True})
    build_code_graph(ctx2)
    cache_dir = run2 / GRAPH_SUBDIR / "graphify-out"
    assert cache_dir.exists()
    assert (cache_dir / "graph.json").exists()


def test_different_target_does_not_reuse_cache(tmp_path):
    source1 = git_repo(tmp_path / "repo1")
    run1 = tmp_path / "runs" / "run1"
    run1_cg = run1 / GRAPH_SUBDIR
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
    cache_dir = run2 / GRAPH_SUBDIR / "graphify-out"
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
    status = _read_status(ctx.workdir / GRAPH_SUBDIR)
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


@pytest.fixture
def stub_graphify(tmp_path, monkeypatch):
    """A graphify that exits at once, so the op always emits its paragraph without the real tool."""
    stub = tmp_path / "stub-bin" / "graphify"
    stub.parent.mkdir()
    stub.write_text("#!/bin/sh\nexit 0\n")
    stub.chmod(0o755)
    monkeypatch.setenv("_REFACTORY_GRAPHIFY_BIN_OVERRIDE", str(stub))
    return stub


def _prompt(model, mark: str) -> str:
    found = [p for p in model.prompts if mark in p]
    assert found, f"no prompt with {mark!r}"
    return found[0]


_SCOPED_RUN = [SCOPED_RESEARCH_ANSWER, PITFALLS_ANSWER, "one plan", DESIGN_DOC_ANSWER, *BUILD]


def test_a_graph_that_failed_is_never_offered_to_a_later_prompt(tmp_path, stub_graphify):
    run, model = _design_run_with_knobs(tmp_path, "Make tool.py print built.", list(_SCOPED_RUN),
                                        extra_knobs={"code_graph": True})
    assert run.exit_ok, run.error
    # The stub's build writes no graph: the explorer is told so and explores the files,
    # and no step after it is told to query the graph (`record_graph_context`).
    assert "## No code graph (failed: " in _prompt(model, EXPLORER_MARK)
    assert run.facts["code_graph_paragraph"] == ""
    for mark in (SIZER_MARK, "You are the research director", "pitfalls researcher"):
        assert "graph callers" not in _prompt(model, mark)


def test_the_graph_is_on_by_default(tmp_path, stub_graphify):
    assert DesignMode().state_dict()["knobs"]["code_graph"]["value"] is True
    run, model = _design_run_with_knobs(tmp_path, "Make tool.py print built.", list(_SCOPED_RUN))
    assert run.exit_ok, run.error
    assert run.facts["code_graph_paragraph"] in _prompt(model, SIZER_MARK)
    assert (tmp_path / "run" / GRAPH_SUBDIR / "status.json").exists()


def test_no_graph_flag_prompts_have_no_graph_paragraph(tmp_path, stub_graphify):
    run, model = _design_run_with_knobs(tmp_path, "Make tool.py print built.", list(_SCOPED_RUN),
                                        extra_knobs={"code_graph": False})
    assert run.exit_ok, run.error
    for mark in (SIZER_MARK, "You are the research director", "pitfalls researcher"):
        assert "Code knowledge graph" not in _prompt(model, mark), mark


def _flags(*argv):
    import argparse

    from refactory_dsh import run_flags
    parser = argparse.ArgumentParser()
    run_flags.add_arguments(parser)
    return parser.parse_args(["--name", "x", "--check", "true", *argv])


def _project(tmp_path, monkeypatch, settings=None):
    project = tmp_path / "project"
    project.mkdir()
    if settings is not None:
        (project / "refactory.workers.json").write_text(json.dumps(settings))
    monkeypatch.setenv("REFACTORY_PROJECT", str(project))
    for env in ("REFACTORY_CODE_GRAPH", "REFACTORY_CODE_GRAPH_TIMEOUT_S", "REFACTORY_CODE_GRAPH_MAX_FILES"):
        monkeypatch.delenv(env, raising=False)
    return project


def test_env_var_off_produces_knob(tmp_path, monkeypatch):
    from refactory_dsh import run_flags
    _project(tmp_path, monkeypatch)
    monkeypatch.setenv("REFACTORY_CODE_GRAPH", "0")
    assert "code_graph=0" in run_flags.knobs(_flags(), "default_pack/design")


def test_project_file_settings_become_knobs_and_win_over_the_environment(tmp_path, monkeypatch):
    from refactory_dsh import run_flags, worker_config
    settings = {"code_graph": {"enabled": False, "timeout_s": 120, "max_files": 500}}
    _project(tmp_path, monkeypatch, settings)
    monkeypatch.setenv("REFACTORY_CODE_GRAPH", "1")
    monkeypatch.setenv("REFACTORY_CODE_GRAPH_TIMEOUT_S", "900")
    found = run_flags.knobs(_flags(), "default_pack/design")
    assert {"code_graph=0", "code_graph_timeout_s=120", "code_graph_max_files=500"} <= set(found)
    assert "code_graph=1" not in found and "code_graph_timeout_s=900" not in found
    worker_config.validate(settings)  # the routing file accepts the section


@pytest.mark.parametrize("section", [{"enabled": "no"}, {"timeout_s": True}, {"colour": 1}, []])
def test_a_bad_code_graph_section_is_refused(section):
    from refactory_dsh import worker_config
    with pytest.raises(ValueError, match="code_graph"):
        worker_config.validate({"code_graph": section})


def test_an_explicit_knob_wins_over_the_settings(tmp_path, monkeypatch):
    from refactory_dsh import run_flags
    _project(tmp_path, monkeypatch, {"code_graph": {"enabled": False}})
    found = run_flags.knobs(_flags("--knob", "code_graph=1"), "default_pack/design")
    assert found.index("code_graph=1") > found.index("code_graph=0")


@pytest.mark.parametrize("mode_id, mode_class", [
    ("default_pack/design", "refactory_dsh.design:DesignMode"),
    ("default_pack/design_fast", "refactory_dsh.fast:DesignFastMode"),
])
def test_every_graph_knob_a_run_is_given_is_declared(tmp_path, monkeypatch, mode_id, mode_class):
    """A setting kept in the shell or the project file must never break a mode that has no graph."""
    import importlib

    from refactory_dsh import run_flags
    _project(tmp_path, monkeypatch, {"code_graph": {"enabled": False, "timeout_s": 60}})
    monkeypatch.setenv("REFACTORY_CODE_GRAPH", "0")
    monkeypatch.setenv("REFACTORY_CODE_GRAPH_MAX_FILES", "10")
    module, name = mode_class.split(":")
    declared = set(getattr(importlib.import_module(module), name)().knob_names())
    given = {pair.split("=", 1)[0] for pair in run_flags.knobs(_flags(), mode_id)}
    assert {k for k in given if k.startswith("code_graph")} <= declared
    if mode_id.endswith("design_fast"):
        assert not any(k.startswith("code_graph") for k in given)


def test_design_run_by_its_full_name_gets_the_settings(tmp_path, monkeypatch):
    from refactory_dsh import run_flags
    _project(tmp_path, monkeypatch, {"code_graph": {"timeout_s": 60}})
    named = run_flags.knobs(_flags(), None, named="default_pack/design")
    assert "code_graph_timeout_s=60" in named
    assert not any(k.startswith("code_graph_timeout") for k in
                   run_flags.knobs(_flags(), None, named="counting_pack/fixed_step"))
    assert "code_graph=0" in run_flags.knobs(_flags("--no-graph"), None, named="counting_pack/fixed_step")


def test_design_fast_refuses_no_graph():
    from refactory_dsh import run_flags
    problem = run_flags.refusal(_flags("--no-graph"), "default_pack/design_fast",
                                runtime=run_flags.WALKER, label="--fast")
    assert problem and "--no-graph" in problem
    assert run_flags.refusal(_flags("--no-graph"), "default_pack/design", runtime=run_flags.WALKER) is None


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
    # Run state, not an output: a declared output under .refactory/ cannot be reset.
    assert not any(".refactory" in w for w in (getattr(node, "writes", None) or []))


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


# ── query log and per-node usage ──────────────────────────────────────────────

def _log(graph_dir):
    return [json.loads(line) for line in (graph_dir / QUERY_LOG).read_text().splitlines()]


def test_every_query_is_logged_with_its_hits(tmp_path, capsys):
    graph_dir = tmp_path / "code-graph"
    _write_fixture(graph_dir)
    _write_status(graph_dir / "status.json", {"status": "ready"})
    assert cli_graph(_make_args("callers", "classify_scale", graph_dir)) == 0
    assert cli_graph(_make_args("explain", "classify_scale", graph_dir)) == 0
    assert cli_graph(_make_args("callers", "nonexistent_function", graph_dir)) == 0
    graph = _fixture_graph()
    named = {n["id"] for n in graph["nodes"] if n["label"] == "classify_scale()"}
    callers = sum(1 for link in graph["links"] if link["relation"] == "calls" and link["target"] in named)
    first, second, third = _log(graph_dir)
    assert (first["kind"], first["query"], first["hits"], first["status"]) == \
        ("callers", "classify_scale", callers, "ready")
    assert (second["kind"], second["hits"]) == ("explain", len(named))
    assert third["hits"] == 0
    assert all(isinstance(e["ms"], int) and e["at"] > 0 for e in (first, second, third))


def test_a_query_the_graph_cannot_answer_is_logged_too(tmp_path, capsys):
    graph_dir = tmp_path / "code-graph"
    graph_dir.mkdir()
    _write_status(graph_dir / "status.json", {"status": "failed: timeout"})
    assert cli_graph(_make_args("callers", "foo", graph_dir)) == 1
    (entry,) = _log(graph_dir)
    assert (entry["status"], entry["hits"]) == ("failed: timeout", 0)


def test_a_query_against_no_graph_writes_nothing(tmp_path, capsys):
    assert cli_graph(_make_args("callers", "foo", tmp_path / "nowhere")) == 1
    assert not (tmp_path / "nowhere").exists()


def _event(at, node, kind, state=None):
    from datetime import datetime, timezone
    entry = {"at": datetime.fromtimestamp(at, timezone.utc).isoformat(), "node_id": node, "kind": kind}
    if state:
        entry["state"] = state
    return json.dumps(entry)


def test_usage_counts_queries_per_node_from_the_run_s_events(tmp_path):
    run = tmp_path / "run"
    graph_dir = run / GRAPH_SUBDIR
    graph_dir.mkdir(parents=True)
    _write_status(graph_dir / "status.json", {"status": "ready", "nodes": 3})
    (run / ".refactory" / "events.jsonl").write_text("\n".join([
        _event(100, "size_change", "node_started"), _event(200, "size_change", "node_completed"),
        # The spine's worker events say the same for a dispatched agent.
        _event(300, "research_director", "worker", "starting"),
        _event(400, "research_director", "worker", "completed"),
        _event(410, "pitfalls_researcher", "node_started"),
        _event(500, "pitfalls_researcher", "node_completed"),
        # Two nodes at once (a fork): a query then names no single node.
        _event(600, "health_checker", "worker", "starting"),
        _event(650, "verifier", "worker", "starting"),
        _event(700, "health_checker", "worker", "budget_exhausted"),
        _event(800, "verifier", "worker", "completed"),
        # Still running when the report reads the log.
        _event(900, "design_doc", "node_started"),
    ]) + "\n")
    queries = [(150, None, "callers", "ready"), (160, None, "explain", "ready"),
               (350, None, "callers", "ready"), (405, None, "callers", "ready"),
               (420, None, "callers", "failed: timeout"), (660, None, "callers", "ready"),
               (750, None, "callers", "ready"),
               # A worker that named its node is believed, whatever ran at the time.
               (660, "verifier", "explain", "ready"), (950, None, "callers", "ready")]
    (graph_dir / QUERY_LOG).write_text("".join(
        json.dumps({"at": at, "node": node, "kind": kind, "query": "f", "hits": 1, "status": st,
                    "ms": 5}) + "\n" for at, node, kind, st in queries))
    assert usage(run) == {
        "status": "ready", "queries": 9, "not_available": 1,
        "by_node": {"ambiguous": 1, "design_doc": 1, "pitfalls_researcher": 1,
                    "research_director": 1, "size_change": 2, "unattributed": 1, "verifier": 2},
        "by_kind": {"callers": 7, "explain": 2}}


def test_usage_is_none_without_a_graph_and_zero_without_queries(tmp_path):
    assert usage(tmp_path) is None
    _write_status_dir = tmp_path / GRAPH_SUBDIR
    _write_status_dir.mkdir(parents=True)
    _write_status(_write_status_dir / "status.json", {"status": "skipped: graphify not installed"})
    assert usage(tmp_path) == {"status": "skipped", "queries": 0, "not_available": 0,
                               "by_node": {}, "by_kind": {}}


def test_the_report_carries_the_graph_use(tmp_path, stub_graphify):
    run, _model = _design_run_with_knobs(tmp_path, "Make tool.py print built.", list(_SCOPED_RUN))
    assert run.exit_ok, run.error
    design = json.loads((tmp_path / "run" / "design.json").read_text())
    assert design["code_graph"]["queries"] == 0
    assert design["code_graph"]["status"] in ("ready", "failed", "building")


def test_the_report_has_no_graph_entry_when_it_is_off(tmp_path, stub_graphify):
    run, _model = _design_run_with_knobs(tmp_path, "Make tool.py print built.", list(_SCOPED_RUN),
                                         extra_knobs={"code_graph": False})
    assert run.exit_ok, run.error
    assert "code_graph" not in json.loads((tmp_path / "run" / "design.json").read_text())


def test_a_query_whose_log_cannot_be_written_still_answers(tmp_path, capsys):
    graph_dir = tmp_path / "code-graph"
    _write_fixture(graph_dir)
    _write_status(graph_dir / "status.json", {"status": "ready"})
    (graph_dir / QUERY_LOG).mkdir()  # opening it to append fails
    assert cli_graph(_make_args("callers", "classify_scale", graph_dir)) == 0
    assert "build_code_graph()" in capsys.readouterr().out


def test_a_query_names_the_node_its_worker_set(tmp_path, monkeypatch, capsys):
    graph_dir = tmp_path / "code-graph"
    _write_fixture(graph_dir)
    _write_status(graph_dir / "status.json", {"status": "ready"})
    monkeypatch.setenv("REFACTORY_NODE_ID", "research_director")
    cli_graph(_make_args("callers", "classify_scale", graph_dir))
    assert _log(graph_dir)[0]["node"] == "research_director"


def test_a_node_s_worker_environment_names_the_node(monkeypatch):
    from refactory_dsh.model import DshModelClient
    monkeypatch.setenv("REFACTORY_NODE_ID", "stale_from_the_parent")
    client = DshModelClient(provider="p", model="m")
    assert "REFACTORY_NODE_ID" not in client._environment()
    assert client.for_node("size_change")._environment()["REFACTORY_NODE_ID"] == "size_change"


def test_the_background_build_lowers_its_priority():
    from refactory_dsh.code_graph import _make_build_script
    script = _make_build_script(graphify_bin="g", source="s", graph_dir="d", timeout_s=1, target_key="t")
    assert "os.nice(10)" in script
