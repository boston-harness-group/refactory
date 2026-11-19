"""The code explorer step: graph query/path, the wait, the explorer's file, and its reach.

The explorer (`explore_code`) runs before sizing and writes graph-context.md, from the code
graph when there is one and from the files when there is not; `record_graph_context`
heads it with which, and the sizer, research, strategy, design and the builder read it
once each.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from refactory_dsh import code_graph, lowering
from refactory_dsh.code_graph import (
    EXPLORER_MARK,
    EXPLORER_MISSING,
    GRAPH_CONTEXT,
    GRAPH_CONTEXT_MAX_CHARS,
    GRAPH_SUBDIR,
    QUERY_LOG,
    _write_status,
    await_code_graph,
    cli_graph,
    path_between,
    query_graph,
    record_graph_context,
)
from refactory_dsh.design import DesignMode
from refactory_dsh.design_ops import SIZER_MARK
from refactory_dsh.nn.executor import RunContext
from tests_support import (
    DESIGN_DOC_ANSWER,
    PITFALLS_ANSWER,
    SCOPED_RESEARCH_ANSWER,
    VERIFICATION_ANSWER,
    FakeModel,
    git_repo,
    render,
)


# ── a small graph in graphify's shape ─────────────────────────────────────────

def _node(nid, label, path, line, kind="code"):
    return {"id": nid, "label": label, "file_type": kind, "source_file": path,
            "source_location": f"L{line}" if line else "", "norm_label": label.rstrip("()").lower()}


def _link(source, target, relation, path="", line=0):
    return {"source": source, "target": target, "relation": relation,
            "source_file": path, "source_location": f"L{line}" if line else ""}


def _graph() -> dict:
    nodes = [
        _node("route", "route_order()", "shop/routing.py", 10),
        _node("price", "price_order()", "shop/pricing.py", 20),
        _node("tax", "apply_tax()", "shop/tax.py", 5),
        _node("ship", "ship_order()", "shop/shipping.py", 30),
        _node("mod", "pricing.py", "shop/pricing.py", 1),
        _node("any", "Any", "shop/types.py", 1),
        _node("t1", "test_price_order()", "tests/test_pricing.py", 3),
        _node("t2", "test_price_order_tax()", "tests/test_pricing.py", 9),
        _node("why", "price rationale", "shop/pricing.py", 19, kind="rationale"),
        _node("island", "unrelated()", "other/island.py", 1),
    ]
    links = [
        _link("route", "price", "calls", "shop/routing.py", 12),
        _link("price", "tax", "calls", "shop/pricing.py", 22),
        _link("mod", "price", "contains", "shop/pricing.py", 20),
        # A type reference joins everything to `Any`: a path must not take it.
        _link("ship", "any", "references", "shop/shipping.py", 31),
        _link("tax", "any", "references", "shop/tax.py", 6),
        # A test that calls both ends: a path must not step through it.
        _link("t1", "ship", "calls", "tests/test_pricing.py", 4),
        _link("t1", "tax", "calls", "tests/test_pricing.py", 5),
        _link("t2", "price", "calls", "tests/test_pricing.py", 10),
        _link("why", "price", "rationale_for", "shop/pricing.py", 19),
    ]
    return {"directed": True, "nodes": nodes, "links": links}


def _graph_dir(tmp_path: Path, status: dict | None = None) -> Path:
    graph_dir = tmp_path / "code-graph"
    (graph_dir / "graphify-out").mkdir(parents=True)
    (graph_dir / "graphify-out" / "graph.json").write_text(json.dumps(_graph()))
    _write_status(graph_dir / "status.json", status or {"status": "ready", "nodes": 10, "edges": 9})
    return graph_dir


# ── query ─────────────────────────────────────────────────────────────────────

def test_query_finds_the_code_first_and_walks_code_edges(tmp_path):
    graph_dir = _graph_dir(tmp_path)
    # test_price_order_tax() matches all three words, the code at most two: the code still
    # comes first, as tests repeat the words of what they test.
    ranked = query_graph(graph_dir, "price order tax", depth=1).splitlines()
    assert ranked[1].strip().startswith("price_order()  shop/pricing.py:20")
    assert "test_price_order_tax()" in "\n".join(ranked[1:6])
    out = query_graph(graph_dir, "price order", depth=1)
    lines = out.splitlines()
    best = lines[1:lines.index("")] if "" in lines else lines[1:]
    # A comment node is never a match.
    assert best[0].strip().startswith("price_order()  shop/pricing.py:20")
    assert not any("rationale" in line for line in lines)
    assert "price_order() -> calls -> apply_tax()  shop/tax.py:5" in out
    assert "price_order() <- calls <- route_order()  shop/routing.py:10" in out


def test_query_depth_two_reaches_further_and_skips_type_references(tmp_path):
    graph_dir = _graph_dir(tmp_path)
    one = query_graph(graph_dir, "route", depth=1)
    two = query_graph(graph_dir, "route", depth=2)
    assert "apply_tax()" not in one
    assert "[2] price_order() -> calls -> apply_tax()" in two
    assert "Any" not in two


def test_query_with_no_matching_word_is_none(tmp_path):
    graph_dir = _graph_dir(tmp_path)
    assert query_graph(graph_dir, "zebra", depth=1) is None
    assert query_graph(graph_dir, "a an", depth=1) is None


# ── path ──────────────────────────────────────────────────────────────────────

def test_path_follows_calls_with_their_locations(tmp_path):
    out = path_between(_graph_dir(tmp_path), "route_order", "apply_tax")
    assert out == ("2 hop(s):\n"
                   "  route_order()  shop/routing.py:10\n"
                   "    --calls-->  (shop/routing.py:12)\n"
                   "  price_order()  shop/pricing.py:20\n"
                   "    --calls-->  (shop/pricing.py:22)\n"
                   "  apply_tax()  shop/tax.py:5")


def test_path_never_steps_through_a_type_or_a_test(tmp_path):
    # ship_order and apply_tax meet only through `Any` (a reference) and a test.
    assert path_between(_graph_dir(tmp_path), "ship_order", "apply_tax") == ""


def test_path_may_end_at_a_test(tmp_path):
    out = path_between(_graph_dir(tmp_path), "test_price_order_tax", "apply_tax")
    assert out.startswith("2 hop(s):") and out.rstrip().endswith("apply_tax()  shop/tax.py:5")


def test_path_with_an_unknown_name_is_none(tmp_path):
    assert path_between(_graph_dir(tmp_path), "route_order", "nowhere") is None


def _args(command, graph_dir, **fields):
    return SimpleNamespace(graph=str(graph_dir), graph_command=command, **fields)


def test_the_cli_answers_query_and_path_and_logs_both(tmp_path, capsys):
    graph_dir = _graph_dir(tmp_path)
    assert cli_graph(_args("query", graph_dir, terms=["price", "order"], depth=2)) == 0
    assert "price_order()  shop/pricing.py:20" in capsys.readouterr().out
    assert cli_graph(_args("path", graph_dir, start="route_order", end="apply_tax")) == 0
    assert "2 hop(s):" in capsys.readouterr().out
    assert cli_graph(_args("path", graph_dir, start="ship_order", end="apply_tax")) == 0
    assert "No path of code edges" in capsys.readouterr().out
    entries = [json.loads(line) for line in (graph_dir / QUERY_LOG).read_text().splitlines()]
    assert [(e["kind"], e["query"]) for e in entries] == [
        ("query", "price order"), ("path", "route_order -> apply_tax"),
        ("path", "ship_order -> apply_tax")]
    assert [e["hits"] for e in entries][1:] == [2, 0]


def test_the_console_parses_query_and_path():
    from refactory_dsh.cli import build_parser

    parser = build_parser()
    query = parser.parse_args(["graph", "query", "price", "order", "--depth", "2", "--graph", "g"])
    assert (query.graph_command, query.terms, query.depth, query.graph) == (
        "query", ["price", "order"], 2, "g")
    path = parser.parse_args(["graph", "--graph", "g", "path", "a", "b"])
    assert (path.graph_command, path.start, path.end, path.graph) == ("path", "a", "b", "g")


# ── await_code_graph ──────────────────────────────────────────────────────────

def _ctx(run: Path, knobs: dict | None = None, facts: dict | None = None) -> RunContext:
    return RunContext(run, knobs=knobs or {}, facts=facts or {})


def test_await_says_ready_and_gives_the_commands(tmp_path):
    run = tmp_path / "run"
    (run / GRAPH_SUBDIR).mkdir(parents=True)
    _write_status(run / GRAPH_SUBDIR / "status.json", {"status": "ready", "nodes": 7, "edges": 9})
    (run / GRAPH_CONTEXT).write_text("from an earlier visit")
    out = await_code_graph(_ctx(run))
    assert out["code_graph_state"] == "ready: 7 nodes, 9 edges"
    for verb in ("query", "explain", "path", "callers"):
        assert f"graph {verb} " in out["code_graph_usage"]
    assert f"--graph {run / GRAPH_SUBDIR}" in out["code_graph_usage"]
    command = out["code_graph_usage"].split("`", 2)[1].split(" graph ", 1)[0]
    assert Path(command).is_absolute() and Path(command).is_file()
    for word in ("limit", "maximum", "at most", "no more than"):
        assert word not in out["code_graph_usage"].lower()
    # An earlier visit's context is never read as this one's.
    assert not (run / GRAPH_CONTEXT).exists()


@pytest.mark.parametrize("status, state", [
    ({"status": "failed: graphify exit 1"}, "failed: graphify exit 1"),
    ({"status": "skipped: graphify not installed"}, "skipped: graphify not installed"),
    (None, "absent"),
])
def test_await_with_no_graph_gives_no_commands(tmp_path, status, state):
    run = tmp_path / "run"
    (run / GRAPH_SUBDIR).mkdir(parents=True)
    if status:
        _write_status(run / GRAPH_SUBDIR / "status.json", status)
    assert await_code_graph(_ctx(run)) == {"code_graph_state": state, "code_graph_usage": ""}


def test_await_with_the_graph_off_waits_for_nothing(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    out = await_code_graph(_ctx(run, knobs={"code_graph": False}))
    assert out == {"code_graph_state": "off (code_graph=0)", "code_graph_usage": ""}


def test_await_stops_waiting_at_its_bound(tmp_path):
    import time

    run = tmp_path / "run"
    (run / GRAPH_SUBDIR).mkdir(parents=True)
    _write_status(run / GRAPH_SUBDIR / "status.json",
                  {"status": "building", "started": time.time(), "timeout_s": 300, "pid": os.getpid()})
    out = await_code_graph(_ctx(run, knobs={"code_graph_wait_s": 0}))
    assert out == {"code_graph_state": "still building after 0 s", "code_graph_usage": ""}


def test_await_picks_up_a_build_that_finishes_while_it_waits(tmp_path):
    import threading
    import time

    run = tmp_path / "run"
    status = run / GRAPH_SUBDIR / "status.json"
    status.parent.mkdir(parents=True)
    _write_status(status, {"status": "building", "started": time.time(), "timeout_s": 300,
                           "pid": os.getpid()})
    timer = threading.Timer(0.5, _write_status, (status, {"status": "ready", "nodes": 1, "edges": 0}))
    timer.start()
    try:
        out = await_code_graph(_ctx(run, knobs={"code_graph_wait_s": 30}))
    finally:
        timer.cancel()
    assert out["code_graph_state"] == "ready: 1 nodes, 0 edges"


# ── record_graph_context ──────────────────────────────────────────────────────

BODY = "## Relevant modules\n- shop/pricing.py:20 price_order\n"
#: BODY as the prompts carry it: its headings nest under their `## Code context`.
NESTED = "### Relevant modules\n- shop/pricing.py:20 price_order\n"


def _record_run(tmp_path: Path, status: dict | None, queries: list[dict] = (), body: str | None = BODY,
                facts: dict | None = None) -> tuple[Path, dict]:
    run = tmp_path / "run"
    graph_dir = run / GRAPH_SUBDIR
    graph_dir.mkdir(parents=True)
    if status:
        _write_status(graph_dir / "status.json", status)
    with open(graph_dir / QUERY_LOG, "w") as log:
        for entry in queries:
            log.write(json.dumps(entry) + "\n")
    if body is not None:
        (run / GRAPH_CONTEXT).write_text(body)
    return run, record_graph_context(_ctx(run, facts=facts or {}))


READY = {"status": "ready", "nodes": 10, "edges": 9}
EXPLORER_QUERY = {"node": "explore_code", "kind": "query", "status": "ready"}


def test_a_context_made_from_the_graph_says_so_with_its_queries(tmp_path):
    run, out = _record_run(tmp_path, READY, [EXPLORER_QUERY, EXPLORER_QUERY,
                                             {"node": "size_change", "status": "ready"},
                                             {"node": "explore_code", "status": "failed: x"}],
                           facts={"code_graph_state": "ready: 10 nodes, 9 edges"})
    assert out["graph_context"] == (
        "Made from the code graph (ready: 10 nodes, 9 edges; 2 explorer queries).\n\n" + NESTED)
    assert (run / GRAPH_CONTEXT).read_text() == out["graph_context"]
    assert out["graph_context_source"].startswith("the code graph")
    # The later steps' paragraph is for follow-ups, pointing at the context above.
    assert out["code_graph_paragraph"].startswith("## Code graph follow-ups\n")
    assert "code context above" in out["code_graph_paragraph"]
    assert f"--graph {run / GRAPH_SUBDIR}" in out["code_graph_paragraph"]


def test_a_context_from_the_files_says_why_there_was_no_graph(tmp_path):
    run, out = _record_run(tmp_path, {"status": "failed: timeout"},
                           facts={"code_graph_state": "failed: timeout",
                                  "code_graph_paragraph": "the build's paragraph"})
    assert out["graph_context"].startswith(
        "Made from file exploration: no code graph (failed: timeout).\n\n")
    # No graph to query: no paragraph telling the later steps to query it.
    assert out["code_graph_paragraph"] == ""


def test_a_ready_graph_the_explorer_did_not_query_is_file_exploration(tmp_path):
    _, out = _record_run(tmp_path, READY, facts={"code_graph_state": "ready: 10 nodes, 9 edges"})
    assert out["graph_context"].startswith(
        "Made from file exploration: the code graph was ready, but the explorer ran no queries.")


def test_a_build_still_running_keeps_the_build_s_paragraph(tmp_path):
    import time

    status = {"status": "building", "started": time.time(), "timeout_s": 300, "pid": os.getpid()}
    _, out = _record_run(tmp_path, status, facts={"code_graph_state": "still building after 120 s",
                                                  "code_graph_paragraph": "the build's paragraph"})
    assert out["code_graph_paragraph"] == "the build's paragraph"
    assert "no code graph (still building after 120 s)" in out["graph_context"]


@pytest.mark.parametrize("missing, why", [
    (None, "the explorer step wrote nothing"),
    ("timed out after 600 s", "the explorer step did not finish: timed out after 600 s"),
])
def test_no_context_carries_nothing_and_the_file_says_why(tmp_path, missing, why):
    run = tmp_path / "run"
    (run / "reviews").mkdir(parents=True)
    if missing:
        (run / EXPLORER_MISSING).write_text(missing)
    out = record_graph_context(_ctx(run))
    assert out["graph_context"] == ""
    assert out["graph_context_source"] == f"none: {why}"
    assert (run / GRAPH_CONTEXT).read_text() == f"No code context: {why}.\n"


def test_a_long_context_is_cut_and_points_at_the_file(tmp_path):
    body = "".join(f"- line {i} shop/pricing.py:{i}\n" for i in range(2000))
    run, out = _record_run(tmp_path, READY, body=body)
    assert len(out["graph_context"]) < GRAPH_CONTEXT_MAX_CHARS + 300
    assert out["graph_context"].rstrip().endswith(f"all of it is in {GRAPH_CONTEXT})")
    # Only the prompts' copy is cut: the file the builder and subagents read keeps it all.
    kept = (run / GRAPH_CONTEXT).read_text()
    assert kept.startswith("Made from ") and kept.endswith(body)
    assert "cut at" not in kept


def test_a_ready_graph_with_no_context_is_offered_without_pointing_at_one(tmp_path):
    _, out = _record_run(tmp_path, READY, body=None, facts={"code_graph_paragraph": "build's"})
    assert out["graph_context"] == ""
    paragraph = out["code_graph_paragraph"]
    assert paragraph.startswith("## Code graph\n") and "code context" not in paragraph
    assert "graph query|explain|path|callers" in paragraph


def test_queries_with_no_node_count_when_the_explorer_was_the_one_node_running(tmp_path):
    """The spine's CEO-subagent route sets no REFACTORY_NODE_ID: its queries count by when
    they were made, as `usage` counts them."""
    from datetime import datetime, timezone

    def at(seconds):
        return datetime.fromtimestamp(seconds, timezone.utc).isoformat()

    run = tmp_path / "run"
    (run / ".refactory").mkdir(parents=True)
    events = [{"at": at(100), "node_id": "explore_code", "kind": "node_started"},
              {"at": at(200), "node_id": "explore_code", "kind": "node_completed"},
              {"at": at(300), "node_id": "size_change", "kind": "node_started"},
              {"at": at(300), "node_id": "pitfalls_researcher", "kind": "node_started"},
              {"at": at(400), "node_id": "size_change", "kind": "node_completed"},
              {"at": at(400), "node_id": "pitfalls_researcher", "kind": "node_completed"}]
    (run / ".refactory" / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    queries = [{"node": None, "at": 150, "status": "ready"},
               {"node": None, "at": 160, "status": "ready"},
               {"node": None, "at": 170, "status": "failed: x"},
               {"node": None, "at": 350, "status": "ready"},  # two nodes running: neither
               {"node": None, "at": 500, "status": "ready"}]  # no node running
    run, out = _record_run(tmp_path, READY, queries, facts={"code_graph_state": "ready"})
    assert out["graph_context"].startswith("Made from the code graph (ready; 2 explorer queries).")


def test_the_wait_fits_inside_the_spine_s_op_limit():
    from refactory_dsh.design import CODE_GRAPH_WAIT_MAX_S

    knob = DesignMode.declare()["code_graph_wait_s"]
    state = DesignMode().state_dict()
    workflow = lowering.to_workflow(state)
    limit = getattr(workflow.nodes["await_code_graph"], "command_timeout_s", None) or 600
    assert knob.hi == CODE_GRAPH_WAIT_MAX_S and knob.hi + 30 <= limit


def test_the_context_s_headings_nest_under_the_prompts_heading(tmp_path):
    body = "# Map\n## Relevant modules\n- a.py:1\n#### Deep\n- not # a heading: `x ## y`\n"
    _, out = _record_run(tmp_path, READY, body=body)
    assert out["graph_context"].endswith(
        "## Map\n### Relevant modules\n- a.py:1\n##### Deep\n- not # a heading: `x ## y`\n")


def test_recording_twice_keeps_one_heading(tmp_path):
    body = "###### six\n## Relevant modules\n- a.py:1\n"
    run, first = _record_run(tmp_path, READY, [EXPLORER_QUERY], body=body,
                             facts={"code_graph_state": "ready"})
    # A level-6 heading cannot nest further and stays as written, on every visit.
    assert first["graph_context"].endswith("###### six\n### Relevant modules\n- a.py:1\n")
    second = record_graph_context(_ctx(run, facts={"code_graph_state": "ready"}))
    assert second["graph_context"] == first["graph_context"]
    assert second["graph_context"].count("Made from ") == 1


# ── the design graph ──────────────────────────────────────────────────────────

READERS = ("size_change", "research_director", "pitfalls_researcher", "strategy_director",
           "synthesize_strategy", "design_doc", "emit")


def test_the_explorer_runs_before_sizing_and_the_sizer_before_research():
    graph = DesignMode().build()
    edges = {(edge.source, edge.target) for edge in graph.edges}
    for pair in (("build_code_graph", "await_code_graph"), ("await_code_graph", "fork_explore"),
                 ("fork_explore", "join_explore"), ("join_explore", "record_graph_context"),
                 ("record_graph_context", "clear_sizing"), ("presize_scale", "research_director")):
        assert pair in edges
    assert ("build_code_graph", "clear_sizing") not in edges
    assert ("presize_scale", "await_code_graph") not in edges
    # The explorer starts from the brief alone: nothing the sizer decides reaches it.
    template = graph.nodes["explore_code"].prompt_template
    for sized in ("presize_scope", "presize_scale", "sizing.json", "research_scope"):
        assert sized not in template
    assert graph.nodes["explore_code"].reads == ("brief.json",)
    assert graph.nodes["fork_explore"].optional_targets == ("explore_code",)
    explorer = graph.nodes["explore_code"]
    assert explorer.writes[0] == GRAPH_CONTEXT
    assert explorer.prompt_template.startswith(EXPLORER_MARK)


@pytest.mark.parametrize("node", READERS)
def test_each_reader_declares_the_context_and_carries_it_once(node):
    graph = DesignMode().build()
    assert GRAPH_CONTEXT in graph.nodes[node].reads
    template = graph.nodes[node].prompt_template
    if node == "emit":
        # The builder is pointed at the file, as it is at design.md.
        assert template.count("{{graph_context}}") == 0 and template.count(GRAPH_CONTEXT) == 1
    else:
        assert template.count("{{graph_context}}") == 1


def test_both_runtimes_get_the_explorer_steps():
    state = DesignMode().state_dict()
    workflow = lowering.to_workflow(state)
    for node in ("await_code_graph", "record_graph_context"):
        assert workflow.nodes[node].command.endswith(f"refactory_dsh.code_graph:{node}")
    assert type(workflow.nodes["explore_code"]).__name__ == "AgentNode"
    declared = lowering.node_declarations(state)
    assert declared["explore_code"]["answer_file"] == GRAPH_CONTEXT
    assert declared["record_graph_context"]["produces"] == [
        "graph_context", "graph_context_source", "code_graph_paragraph"]


# ── a walker run ──────────────────────────────────────────────────────────────

CONTEXT = "## Relevant modules\n- tool.py:1 build — UNIQUE-CODE-CONTEXT-LINE\n"
NESTED_CONTEXT = CONTEXT.replace("## ", "### ")
BUILD = [render({"tool.py": "def build():\n    print('built')\n"}),
         "health: ok", "code review: ok", "qa: ok", VERIFICATION_ANSWER, "VERDICT: PASS"]
SCOPED_RUN = [SCOPED_RESEARCH_ANSWER, PITFALLS_ANSWER, "one plan", DESIGN_DOC_ANSWER, *BUILD]


def _walker_run(tmp_path, explorer: list, knobs: dict | None = None):
    repo = git_repo(tmp_path / "repo")
    model = FakeModel(list(SCOPED_RUN), explorer=explorer)
    mode = DesignMode()
    state = mode.state_dict()
    state["knobs"]["pr"]["value"] = False
    for name, value in (knobs or {}).items():
        state["knobs"][name]["value"] = value
    mode.load_state_dict(state)
    run = mode.run(tmp_path / "run", model=model, verdicts={"gate_strategy": "proceed"},
                   task={"name": "short_tool", "brief": "Make tool.py print built.",
                         "check": f"{sys.executable} tool.py", "target": str(repo)})
    return run, model


def _prompts(model, mark):
    return [p for p in model.prompts if mark in p]


@pytest.fixture
def no_graphify(monkeypatch):
    monkeypatch.setenv("_REFACTORY_GRAPHIFY_BIN_OVERRIDE", "")


def test_a_walker_run_hands_the_context_to_each_later_step_once(tmp_path, no_graphify):
    run, model = _walker_run(tmp_path, [CONTEXT])
    assert run.exit_ok, run.error
    (explorer,) = _prompts(model, EXPLORER_MARK)
    # No graphify: the explorer is told to explore the files, and why.
    assert "## No code graph (skipped: graphify not installed)" in explorer
    assert "graph query" not in explorer
    recorded = (tmp_path / "run" / GRAPH_CONTEXT).read_text()
    assert recorded.startswith(
        "Made from file exploration: no code graph (skipped: graphify not installed).\n\n")
    for mark in (SIZER_MARK, "You are the research director", "You are the pitfalls researcher",
                 "Compile the perspectives into one plan", "Write the design document"):
        (prompt,) = _prompts(model, mark)
        assert prompt.count("UNIQUE-CODE-CONTEXT-LINE") == 1, mark
        assert prompt.count("## Code context\n") == 1, mark
        assert "graph callers" not in prompt
    # The sizer is told to check the context in the code, not to trust it or the brief; the
    # context's headings nest under its own.
    (sizer,) = _prompts(model, SIZER_MARK)
    assert "starting from the code context above: confirm what it cites in the code" in sizer
    assert "### Relevant modules\n" in sizer and "\n## Relevant modules" not in sizer
    assert sizer.index("## Code context") < sizer.index("Size this change")
    (builder,) = _prompts(model, "Write the artifact `short_tool`")
    assert "UNIQUE-CODE-CONTEXT-LINE" not in builder
    assert f"the map of the code it touches from `{GRAPH_CONTEXT}`" in builder


def test_a_walker_run_whose_explorer_fails_goes_on_without_context(tmp_path, no_graphify):
    run, model = _walker_run(tmp_path, [RuntimeError("explorer session failed")])
    assert run.exit_ok, run.error
    assert run.facts["graph_context"] == ""
    assert run.facts["graph_context_source"].startswith("none: ")
    for prompt in model.prompts:
        if EXPLORER_MARK not in prompt:
            assert "## Code context" not in prompt
            assert "the map of the code it touches" not in prompt


def test_a_walker_run_with_the_graph_off_still_explores_the_files(tmp_path, no_graphify):
    run, model = _walker_run(tmp_path, [CONTEXT], knobs={"code_graph": False})
    assert run.exit_ok, run.error
    (explorer,) = _prompts(model, EXPLORER_MARK)
    assert "## No code graph (off (code_graph=0))" in explorer
    assert run.facts["graph_context"].startswith("Made from file exploration: no code graph (off")


# ── the spine's node runner ───────────────────────────────────────────────────

def test_the_spine_node_runner_runs_both_explorer_ops(tmp_path):
    from refactory_dsh.run import update_facts
    from refactory_dsh.steering import Ledger

    repo = git_repo(tmp_path / "repo")
    state = DesignMode().state_dict()
    given = {"name": "tool", "brief": "fix it", "check": "python app.py", "target": str(repo),
             "pr": False}
    for knob, value in given.items():
        state["knobs"][knob]["value"] = value
    run = tmp_path / "run"
    run.mkdir()
    Ledger.seed(run, given["brief"])
    graph_dir = _graph_dir(run / ".refactory")
    assert graph_dir == run / GRAPH_SUBDIR
    lowering.write_declarations(state, run)
    update_facts(run, {"source": str(repo), "target": str(repo)})
    workflow = lowering.to_workflow(state)
    env = {**os.environ, **lowering.knob_environment(state),
           "PYTHONPATH": str(Path(__file__).resolve().parents[1]), "PYTHONDONTWRITEBYTECODE": "1",
           "REFACTORY_WORKDIR": str(run)}

    def node(node_id: str) -> dict:
        python, *runner = workflow.nodes[node_id].command.split()
        done = subprocess.run(
            [sys.executable, *runner], cwd=run, capture_output=True, text=True, timeout=60,
            env={**env, "DSH_WORKFLOW_NODE": json.dumps({"id": node_id, "type": "FnNode"})})
        assert done.returncode == 0, done.stderr[-2000:]
        return json.loads((run / "facts.json").read_text())

    facts = node("await_code_graph")
    assert facts["code_graph_state"] == "ready: 10 nodes, 9 edges"
    # The explorer, as a worker would run it: its queries carry its node id.
    for argv in (["query", "price", "order"], ["path", "route_order", "apply_tax"]):
        done = subprocess.run(
            [sys.executable, "-m", "refactory_dsh.cli", "graph", *argv, "--graph", str(graph_dir)],
            env={**env, "REFACTORY_NODE_ID": "explore_code"}, capture_output=True, text=True,
            timeout=60)
        assert done.returncode == 0, done.stderr[-2000:]
    (run / GRAPH_CONTEXT).write_text(CONTEXT)
    facts = node("record_graph_context")
    assert facts["graph_context"] == (
        "Made from the code graph (ready: 10 nodes, 9 edges; 2 explorer queries).\n\n"
        + NESTED_CONTEXT)
    assert facts["code_graph_paragraph"].startswith("## Code graph follow-ups")


# ── a reused graph graphify refuses to update ─────────────────────────────────

@pytest.mark.parametrize("reused, status", [(True, "ready"), (False, "failed: graphify exit 1")])
def test_a_reused_graph_graphify_refuses_is_rebuilt_afresh(tmp_path, reused, status):
    """graphify refuses to update a reused graph when the update comes out smaller (seen
    on a copy from another run of the same commit); a build that started from such a copy
    retries once from nothing."""
    repo = git_repo(tmp_path / "repo")
    graph_dir = tmp_path / "code-graph"
    out = graph_dir / "graphify-out"
    out.mkdir(parents=True)
    (out / "graph.json").write_text(json.dumps(_graph()))
    stub = tmp_path / "graphify"
    stub.write_text("#!/bin/sh\n"
                    'if [ -f "$GRAPHIFY_OUT/graph.json" ]; then echo refusing >&2; exit 1; fi\n'
                    'mkdir -p "$GRAPHIFY_OUT"; echo \'{"nodes": [{"id": "a"}], "links": []}\' '
                    '> "$GRAPHIFY_OUT/graph.json"\n')
    stub.chmod(0o755)
    script = code_graph._make_build_script(graphify_bin=str(stub), source=str(repo),
                                           graph_dir=str(graph_dir), timeout_s=60,
                                           target_key="t", reused=reused)
    (graph_dir / "_build.py").write_text(script)
    subprocess.run([sys.executable, str(graph_dir / "_build.py")], timeout=60, check=False)
    state = json.loads((graph_dir / "status.json").read_text())
    assert state["status"] == status
    if reused:
        assert state["nodes"] == 1
