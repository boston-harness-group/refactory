"""The item loader and the per-item commands DSH drives a data node through.

Issue #59: a data node iterates on both runtimes, and the item loader is one
implementation. These tests pin the loader against the three source formats, the
bounds, the unbound source, and — the point of the issue — that driving the
commands by hand produces the facts the walking executor produces for the same
graph.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from refactory_dsh import items, lowering, nn
from refactory_dsh import task as task_module
from refactory_dsh.nn.executor import CURRENT_ITEM, execute, load_items


def _probe_task():
    """A Task named, set up and judged without looking at the pass."""
    from refactory_dsh import task as task_module

    class Counting(task_module.Task):
        name = "counting"

        def instances(self):
            return [
                task_module.TaskInstance(id="alpha", metadata={"split": "train"}),
                task_module.TaskInstance(id="beta", metadata={"split": "test"}),
            ]

        def setup(self, instance, workspace):
            (workspace / f"setup-{instance.id}").write_text("prepared")

        def verify(self, instance, workspace):
            return task_module.VerifyResult(
                passed=True, score=1.0 if instance.id == "alpha" else 0.5
            )

    task_module.register_task("counting", Counting)
    return Counting


def _data_graph(**fields) -> nn.Graph:
    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        writes=(CURRENT_ITEM,), produces=("item_scores",), **fields,
    ))
    graph.add(nn.Node(
        "work", "op", call="loop_ops:record_item",
        reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
    ))
    graph.add(nn.Node(
        "report", "op", call="loop_ops:write_best",
        writes=("best.json",), produces=("best",),
    ))
    graph.connect("data", "report")
    return graph


def _declared(graph: nn.Graph, workdir: Path) -> nn.Node:
    """Write the run's declarations and hand back the data node they describe."""
    lowering.write_declarations({"graph": graph.to_state()}, workdir)
    return items.data_node(workdir, "data")


# ── the loader: one implementation, both runtimes ──────────────────────


def test_the_loader_reads_jsonl_csv_and_directory(tmp_path):
    """The three source formats, read the one way both runtimes read them."""
    jsonl = tmp_path / "items.jsonl"
    jsonl.write_text('{"id": "first", "value": 1}\n\n{"value": 2}\n')
    rows = tmp_path / "items.csv"
    rows.write_text("id,value\nfirst,1\nsecond,2\n")
    directory = tmp_path / "projects"
    (directory / "one").mkdir(parents=True)
    (directory / "two").mkdir()
    (directory / "loose.txt").write_text("not a project")

    def node(**fields) -> nn.Node:
        return nn.Node("data", "data", subgraph_entry="w", subgraph_exit="w", **fields)

    assert load_items(node(source_path="items.jsonl", source_format="jsonl"), tmp_path) == [
        {"id": "first", "value": 1},
        # The id falls back to the line's position, so a blank line still counts.
        {"id": "2", "value": 2},
    ]
    assert load_items(node(source_path="items.csv", source_format="csv"), tmp_path) == [
        {"id": "first", "value": "1"},
        {"id": "second", "value": "2"},
    ]
    assert load_items(node(source_path="projects", source_format="directory"), tmp_path) == [
        {"id": "one", "path": str(directory / "one")},
        {"id": "two", "path": str(directory / "two")},
    ]


def test_the_loader_bounds_the_selection_and_refuses_an_unbound_source(tmp_path):
    """`limit` truncates; a source that was never bound is refused by name."""
    node = nn.Node(
        "data", "data", subgraph_entry="w", subgraph_exit="w",
        source_path="items.jsonl", source_format="jsonl", limit=1,
    )
    (tmp_path / "items.jsonl").write_text('{"id": "a"}\n{"id": "b"}\n')
    assert load_items(node, tmp_path) == [{"id": "a"}]

    unbound = nn.Node(
        "data", "data", subgraph_entry="w", subgraph_exit="w",
        source_path="absent.jsonl", source_format="jsonl",
    )
    with pytest.raises(nn.ExecutionError) as caught:
        load_items(unbound, tmp_path)
    assert "absent.jsonl" in str(caught.value)
    assert "bind the source" in str(caught.value)


def test_an_empty_selection_is_a_fact_and_iterates_nothing(tmp_path):
    """Zero items is a count, not a crash: the commands still answer."""
    node = _declared(_data_graph(inline_items=()), tmp_path)
    assert items.command_list(node, tmp_path) == {"count": 0, "ids": []}


# ── the three commands ────────────────────────────────────────────────


def test_begin_materialises_the_item_and_runs_the_task_setup(tmp_path):
    """The pass reads the item from disk, and the Task prepares first."""
    _probe_task()
    node = _declared(_data_graph(task_ref="counting"), tmp_path)

    assert items.command_list(node, tmp_path) == {"count": 2, "ids": ["alpha", "beta"]}
    assert items.command_begin(node, tmp_path, 1) == {
        "id": "beta", "path": None, "split": "test",
    }
    assert json.loads((tmp_path / CURRENT_ITEM).read_text())["id"] == "beta"
    assert (tmp_path / "setup-beta").exists()


def test_end_folds_the_task_verdict_into_the_run_facts(tmp_path):
    """One verdict per pass, under the fact the walking executor writes."""
    _probe_task()
    node = _declared(_data_graph(task_ref="counting"), tmp_path)

    assert items.command_end(node, tmp_path, 0) == {
        "id": "alpha", "passed": True, "score": 1.0,
    }
    items.command_end(node, tmp_path, 1)
    facts = json.loads((tmp_path / "facts.json").read_text())
    assert facts["item_scores"] == [
        {"id": "alpha", "passed": True, "score": 1.0},
        {"id": "beta", "passed": True, "score": 0.5},
    ]


def test_a_node_that_declares_no_item_scores_keeps_none(tmp_path):
    """The fact is written when the node declares it, as in the walker."""
    _probe_task()
    graph = _data_graph(task_ref="counting")
    graph.nodes["data"].produces = ()
    node = _declared(graph, tmp_path)

    items.command_end(node, tmp_path, 0)
    assert not (tmp_path / "facts.json").exists()


def test_a_task_with_no_opinion_adds_no_score(tmp_path):
    """`verify` returning `None` is the Task saying the mode's own score stands."""
    from refactory_dsh import task as task_module

    class Silent(task_module.Task):
        name = "silent"

    task_module.register_task("silent", Silent)
    node = _declared(_data_graph(task_ref="silent"), tmp_path)
    assert items.command_end(node, tmp_path, 0) == {}
    assert not (tmp_path / "facts.json").exists()


def test_an_out_of_range_item_is_refused_by_name(tmp_path):
    node = _declared(_data_graph(inline_items=({"id": "only"},)), tmp_path)
    with pytest.raises(items.ItemsError) as caught:
        items.command_begin(node, tmp_path, 4)
    assert "out of range" in str(caught.value)


def test_the_commands_refuse_a_node_that_is_not_a_data_node(tmp_path):
    """The declarations say which node is which, and a wrong id is named."""
    graph = _data_graph(inline_items=({"id": "only"},))
    lowering.write_declarations({"graph": graph.to_state()}, tmp_path)
    with pytest.raises(items.ItemsError) as caught:
        items.data_node(tmp_path, "work")
    assert "'op'" in str(caught.value)

    with pytest.raises(items.ItemsError) as caught:
        items.data_node(tmp_path, "nowhere")
    assert "not declared" in str(caught.value)


def test_the_command_line_reports_the_count_as_json(tmp_path, monkeypatch, capsys):
    """`main` is what the runtime calls: one JSON object on stdout, or a reason on stderr."""
    import refactory_dsh.run

    monkeypatch.setattr(refactory_dsh.run, "install_workspace_packages", lambda: None)
    monkeypatch.setenv("REFACTORY_WORKDIR", str(tmp_path))
    _declared(_data_graph(inline_items=({"id": "a"}, {"id": "b"})), tmp_path)

    assert items.main(["list", "--node", "data"]) == 0
    assert json.loads(capsys.readouterr().out) == {"count": 2, "ids": ["a", "b"]}

    assert items.main(["list"]) == 1
    assert "--node is required" in capsys.readouterr().err

    assert items.main(["dance"]) == 2
    assert "usage" in capsys.readouterr().err


# ── parity: the commands produce what the walker produces ─────────────


def test_driving_the_commands_produces_the_walkers_facts(tmp_path):
    """The acceptance the issue is about: same facts, either runtime.

    The walking executor runs the whole loop itself; DSH runs the subgraph from
    outside and drives the item loop through these three commands. Both must land on
    the same `item_scores`, or a mode means two different things.
    """
    _probe_task()
    graph = _data_graph(task_ref="counting")

    walked = tmp_path / "walked"
    result = execute(graph, walked)
    assert result.exit_ok, result.error

    driven = tmp_path / "driven"
    node = _declared(graph, driven)
    count = items.command_list(node, driven)["count"]
    for index in range(count):
        items.command_begin(node, driven, index)
        items.command_end(node, driven, index)

    assert json.loads((driven / "facts.json").read_text())["item_scores"] == (
        result.facts["item_scores"]
    )
    assert json.loads((driven / CURRENT_ITEM).read_text()) == json.loads(
        (walked / CURRENT_ITEM).read_text()
    ), "the last item on disk differs between the runtimes"


def test_the_declarations_carry_the_data_nodes_source(tmp_path):
    """The IR carries the source; the runtime's declarations have to as well."""
    graph = _data_graph(
        task_ref="counting", source_path="items.jsonl", source_format="jsonl", limit=3,
        inline_items=({"id": "a"},),
    )
    path = lowering.write_declarations({"graph": graph.to_state()}, tmp_path)
    entry = json.loads(path.read_text())["data"]
    assert entry["data"] == {
        "subgraph_entry": "work",
        "subgraph_exit": "work",
        "task_ref": "counting",
        "source_path": "items.jsonl",
        "source_format": "jsonl",
        "inline_items": [{"id": "a"}],
        "limit": 3,
    }


# ── the part of the items a run directory names ───────────────────────────


def test_the_loader_and_the_dsh_reader_select_the_same_part(tmp_path):
    """One helper, both runtimes: the walker's loader and `items_of` agree.

    The part is written beside the run rather than passed in, which is what makes it
    the same answer whichever runtime reads it.
    """
    node = nn.Node(
        "data", "data", subgraph_entry="w", subgraph_exit="w",
        inline_items=tuple({"id": name} for name in "abcd"),
    )
    task_module.write_subset(tmp_path, task_module.TRAIN, 0.3)
    assert [item["id"] for item in load_items(node, tmp_path)] == ["a", "b", "c"]
    assert [item["id"] for item in items.items_of(node, tmp_path)] == ["a", "b", "c"]

    task_module.write_subset(tmp_path, task_module.HOLDOUT, 0.3)
    assert [item["id"] for item in load_items(node, tmp_path)] == ["d"]
    assert [item["id"] for item in items.items_of(node, tmp_path)] == ["d"]


def test_without_a_selection_every_item_is_iterated(tmp_path):
    """A mode run outside the outer loop has no run directory naming a part."""
    node = nn.Node(
        "data", "data", subgraph_entry="w", subgraph_exit="w",
        inline_items=({"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}),
    )
    assert [item["id"] for item in load_items(node, tmp_path)] == ["a", "b", "c", "d"]


def test_both_runtimes_iterate_the_part_the_run_directory_names(tmp_path):
    """The walker and the commands DSH drives land on the same items, per part.

    The commands are what DSH runs one pass at a time, so a selection that only the
    walker honoured would make a mode mean two different things.
    """
    _probe_task()
    graph = _data_graph(task_ref="counting")
    assert [instance.id for instance in _probe_task()().instances()] == ["alpha", "beta"]

    for part, expected in (("train", ["alpha"]), ("holdout", ["beta"])):
        walked = tmp_path / f"walked-{part}"
        task_module.write_subset(walked, part, 0.5)
        result = execute(graph, walked)
        assert result.exit_ok, result.error

        driven = tmp_path / f"driven-{part}"
        task_module.write_subset(driven, part, 0.5)
        node = _declared(graph, driven)
        ids = items.command_list(node, driven)["ids"]
        for index in range(len(ids)):
            items.command_begin(node, driven, index)
            items.command_end(node, driven, index)

        assert ids == expected
        assert json.loads((driven / "facts.json").read_text())["item_scores"] == (
            result.facts["item_scores"]
        ), "the runtimes disagree about the part they iterated"


def test_a_run_can_point_a_data_node_at_items_the_mode_never_saw(tmp_path):
    """The input port, fed at run time.

    A harness for a class of work is built once and fed many times. The graph declares
    the source it was built with, and the run may name another; without this the only
    way to use a produced harness on your own items is to edit the mode.
    """
    _probe_task()
    graph = _data_graph(task_ref="counting")
    workdir = tmp_path / "run"
    workdir.mkdir()
    (workdir / "current_item.json").write_text("{}")

    mine = tmp_path / "mine"
    for name in ("gamma", "delta"):
        (mine / name).mkdir(parents=True)

    # The Task would supply alpha and beta; this run says otherwise.
    items.write_source(workdir, source_path=str(mine), source_format="directory")
    assert [item["id"] for item in load_items(_declared(graph, workdir), workdir)] == (
        ["delta", "gamma"]
    ), "the run's own source is what the node reads"


def test_an_item_directory_describes_itself_through_item_json(tmp_path):
    """A directory of items may carry each item's fields beside its content.

    The fields reach the pass the way a Task's metadata does, and the directory's own
    name stays the id, so a described item cannot rename itself out of the selection.
    """
    _probe_task()
    graph = _data_graph(task_ref="counting")
    workdir = tmp_path / "run"
    workdir.mkdir()
    (workdir / "current_item.json").write_text("{}")
    mine = tmp_path / "mine"
    (mine / "one").mkdir(parents=True)
    (mine / "one" / "item.json").write_text(json.dumps({"main_class": "A", "id": "renamed"}))

    items.write_source(workdir, source_path=str(mine), source_format="directory")
    loaded = load_items(_declared(graph, workdir), workdir)
    assert loaded == [{"id": "one", "path": str(mine / "one"), "main_class": "A"}]


def test_a_run_can_point_one_node_somewhere_and_the_rest_nowhere_specific(tmp_path):
    """An entry for the node wins over the `*` entry."""
    workdir = tmp_path / "run"
    workdir.mkdir()
    items.write_source(workdir, source_path="/general")
    (workdir / "items.json").write_text(json.dumps({
        "*": {"source_path": "/general", "source_format": "jsonl"},
        "items": {"source_path": "/specific", "source_format": "csv"},
    }))
    assert items.source_override(workdir, "items")["source_path"] == "/specific"
    assert items.source_override(workdir, "other")["source_path"] == "/general"
    assert items.source_override(tmp_path / "elsewhere", "items") == {}


@pytest.mark.parametrize('archive_artifacts', [False, True])
def test_non_data_reset_archives_only_owned_outputs_when_requested(tmp_path, archive_artifacts):
    graph = nn.Graph()
    graph.add(nn.Node('check', 'op', call='loop_ops:write_number',
                      reads=('input.txt', 'shared.txt'), writes=('checked.json', 'shared.txt'),
                      produces=('checked',)))
    graph.add(nn.Node('finish', 'op', call='loop_ops:write_number',
                      writes=('finished.txt',), produces=('finished',)))
    graph.connect('check', 'finish')
    lowering.write_declarations({'graph': graph.to_state()}, tmp_path)
    original = {'checked.json': '{"old": true}', 'finished.txt': 'old result',
                'input.txt': 'declared input', 'shared.txt': 'read before rewrite',
                'unrelated.txt': 'unowned input'}
    for name, contents in original.items():
        (tmp_path / name).write_text(contents)
    prior_facts = {'checked': True, 'finished': 'old', 'unrelated': 'keep'}
    (tmp_path / 'facts.json').write_text(json.dumps(prior_facts))
    request = {'requested': 'check', 'start': 'check', 'nodes': ['check', 'finish'], 'restartData': []}
    if archive_artifacts:
        request['archiveArtifacts'] = True
    receipt = items.command_reset(tmp_path, request)
    assert receipt['cleared_facts'] == ['checked', 'finished']
    assert json.loads((tmp_path / 'facts.json').read_text()) == {'unrelated': 'keep'}
    archive = tmp_path / receipt['archive']
    assert json.loads((archive / 'facts.json').read_text()) == prior_facts
    if archive_artifacts:
        assert receipt['archived_artifacts'] == ['checked.json', 'finished.txt']
        assert receipt['retained_inputs'] == ['shared.txt']
        for name in ('checked.json', 'finished.txt'):
            assert not (tmp_path / name).exists()
            assert (archive / 'artifacts' / name).read_text() == original[name]
    else:
        assert receipt['archived_artifacts'] == []
        assert all((tmp_path / name).read_text() == contents for name, contents in original.items())
    for name in ('input.txt', 'shared.txt', 'unrelated.txt'):
        assert (tmp_path / name).read_text() == original[name]


def test_reset_keeps_files_an_outside_node_also_writes(tmp_path):
    """A redo archives only files the invalidated nodes own outright (#317).

    ``shared.txt`` is written by *prepare* (not redone) **and** *middle*
    (redone), so the redo keeps it: archiving it would leave nothing to
    recreate it. ``internal.txt`` is written **only** by redone nodes, so it is
    archived even though *downstream* reads it; being read by a redone node is
    not a reason to keep a file.
    """
    graph = nn.Graph()
    graph.add(nn.Node('prepare', 'op', call='loop_ops:write_number',
                      writes=('shared.txt',), produces=('prepared',)))
    graph.add(nn.Node('middle', 'op', call='loop_ops:write_number',
                      reads=('shared.txt',),
                      writes=('shared.txt', 'internal.txt'),
                      produces=('computed',)))
    graph.add(nn.Node('downstream', 'op', call='loop_ops:write_number',
                      reads=('internal.txt',), writes=('result.txt',),
                      produces=('finished',)))
    graph.connect('prepare', 'middle')
    graph.connect('middle', 'downstream')
    lowering.write_declarations({'graph': graph.to_state()}, tmp_path)

    original = {
        'shared.txt': 'written by prepare and middle',
        'internal.txt': 'written only by middle',
        'result.txt': 'result content',
    }
    for name, contents in original.items():
        (tmp_path / name).write_text(contents)

    prior_facts = {'prepared': True, 'computed': 42, 'finished': 'done'}
    (tmp_path / 'facts.json').write_text(json.dumps(prior_facts))

    request = {
        'requested': 'middle',
        'start': 'middle',
        'nodes': ['middle', 'downstream'],
        'archiveArtifacts': True,
    }
    receipt = items.command_reset(tmp_path, request)

    assert 'shared.txt' in receipt['retained_inputs']
    assert (tmp_path / 'shared.txt').read_text() == original['shared.txt']

    assert 'internal.txt' in receipt['archived_artifacts']
    assert not (tmp_path / 'internal.txt').exists()
    archive = tmp_path / receipt['archive']
    assert (archive / 'artifacts' / 'internal.txt').read_text() == original['internal.txt']

    assert 'result.txt' in receipt['archived_artifacts']
    assert not (tmp_path / 'result.txt').exists()
    assert (archive / 'artifacts' / 'result.txt').read_text() == original['result.txt']

    assert receipt['cleared_facts'] == ['computed', 'finished']
    assert json.loads((tmp_path / 'facts.json').read_text()) == {'prepared': True}


# ── F4 failure rule: spine-side item failure tests ───────────────────────────


def _failing_task():
    """A Task whose verify raises on the second item."""
    from refactory_dsh import task as task_module

    class FailVerify(task_module.Task):
        name = "fail_verify"

        def instances(self):
            return [
                task_module.TaskInstance(id="alpha", metadata={"split": "train"}),
                task_module.TaskInstance(id="beta", metadata={"split": "test"}),
            ]

        def setup(self, instance, workspace):
            pass

        def verify(self, instance, workspace):
            if instance.id == "beta":
                raise ValueError("beta verification failed")
            return task_module.VerifyResult(passed=True, score=1.0)

    task_module.register_task("fail_verify", FailVerify)
    return FailVerify


def _setup_failing_task():
    """A Task whose setup raises on the first item."""
    from refactory_dsh import task as task_module

    class FailSetup(task_module.Task):
        name = "fail_setup"

        def instances(self):
            return [
                task_module.TaskInstance(id="alpha", metadata={"split": "train"}),
                task_module.TaskInstance(id="beta", metadata={"split": "test"}),
            ]

        def setup(self, instance, workspace):
            if instance.id == "alpha":
                raise ValueError("alpha setup failed")

        def verify(self, instance, workspace):
            return task_module.VerifyResult(passed=True, score=0.5)

    task_module.register_task("fail_setup", FailSetup)
    return FailSetup


def test_command_end_failure_writes_error_record(tmp_path):
    """command_end with a failing verify writes an error record to item_scores."""
    _failing_task()
    node = _declared(_data_graph(task_ref="fail_verify"), tmp_path)

    # alpha succeeds
    result_alpha = items.command_end(node, tmp_path, 0)
    assert result_alpha["passed"] is True
    assert result_alpha["score"] == 1.0

    # beta fails
    result_beta = items.command_end(node, tmp_path, 1)
    assert "error" in result_beta
    assert result_beta["passed"] is False
    assert result_beta["score"] == 0.0
    assert "beta verification failed" in result_beta["error_text"]
    assert result_beta["error_kind"] == "ValueError"

    # Check facts
    facts = json.loads((tmp_path / "facts.json").read_text())
    scores = facts["item_scores"]
    assert len(scores) == 2
    beta_record = next(s for s in scores if s["id"] == "beta")
    assert beta_record["error_text"] is not None


def test_command_begin_failure_writes_error_record(tmp_path):
    """command_begin with a failing setup returns an error signal."""
    _setup_failing_task()
    node = _declared(_data_graph(task_ref="fail_setup"), tmp_path)

    result = items.command_begin(node, tmp_path, 0)
    assert "error" in result
    assert "alpha setup failed" in result["error"]

    # The error record is in facts
    facts = json.loads((tmp_path / "facts.json").read_text())
    scores = facts["item_scores"]
    assert len(scores) == 1
    assert scores[0]["passed"] is False
    assert scores[0]["score"] == 0.0
    assert "alpha setup failed" in scores[0]["error_text"]


def test_failure_records_match_between_runtimes(tmp_path):
    """Both paths produce identical item_scores including error records."""
    _failing_task()
    graph = _data_graph(task_ref="fail_verify")

    # Walker path
    walked = tmp_path / "walked"
    result = execute(graph, walked)
    assert result.exit_ok, result.error

    # Spine (command-driven) path
    driven = tmp_path / "driven"
    node = _declared(graph, driven)
    count = items.command_list(node, driven)["count"]
    for index in range(count):
        items.command_begin(node, driven, index)
        items.command_end(node, driven, index)

    driven_facts = json.loads((driven / "facts.json").read_text())
    assert driven_facts["item_scores"] == result.facts["item_scores"]


def test_csv_no_id_column_gets_index_ids_on_spine(tmp_path):
    """CSV data with no id column gets index ids on the spine path."""
    _probe_task()
    csv_path = tmp_path / "items.csv"
    csv_path.write_text("value\nfoo\nbar\n")

    graph = _data_graph(
        task_ref="counting",
        source_path="items.csv",
        source_format="csv",
    )
    node = _declared(graph, tmp_path)
    listing = items.command_list(node, tmp_path)
    assert listing["ids"] == ["0", "1"]

    items.command_end(node, tmp_path, 0)
    items.command_end(node, tmp_path, 1)
    facts = json.loads((tmp_path / "facts.json").read_text())
    ids = [s["id"] for s in facts["item_scores"]]
    assert ids == ["0", "1"]


def test_item_scores_replace_on_rerun_spine(tmp_path):
    """item_scores uses replace-by-id: running end twice does not double-count."""
    _probe_task()
    node = _declared(_data_graph(task_ref="counting"), tmp_path)

    # First run
    items.command_end(node, tmp_path, 0)
    items.command_end(node, tmp_path, 1)

    # Second run (rerun)
    items.command_end(node, tmp_path, 0)
    items.command_end(node, tmp_path, 1)

    facts = json.loads((tmp_path / "facts.json").read_text())
    scores = facts["item_scores"]
    # One entry per item, not four
    assert len(scores) == 2
    ids = [s["id"] for s in scores]
    assert sorted(ids) == ["alpha", "beta"]


def test_all_items_failed_raises_on_last_end(tmp_path):
    """When every item fails, command_end for the last item raises ItemsError.

    This makes the spine match the walker's all-items-failed rule: exit code 1
    tells the TS drain the data node failed, producing exit_ok=False.
    """
    from refactory_dsh import task as task_module

    class AllFail(task_module.Task):
        name = "all_fail_spine"

        def instances(self):
            return [
                task_module.TaskInstance(id="alpha", metadata={"split": "train"}),
                task_module.TaskInstance(id="beta", metadata={"split": "test"}),
            ]

        def setup(self, instance, workspace):
            pass

        def verify(self, instance, workspace):
            raise ValueError(f"{instance.id} failed")

    task_module.register_task("all_fail_spine", AllFail)

    node = _declared(_data_graph(task_ref="all_fail_spine"), tmp_path)

    # Begin each item to initialize pass state tracking.
    items.command_begin(node, tmp_path, 0)
    # First item's end: writes error record, returns normally.
    result_alpha = items.command_end(node, tmp_path, 0)
    assert "error" in result_alpha

    items.command_begin(node, tmp_path, 1)
    # Second (last) item's end: after writing the error record, raises
    # ItemsError because all items have error_text.
    with pytest.raises(items.ItemsError, match="all.*failed"):
        items.command_end(node, tmp_path, 1)


def test_command_end_with_failed_reason_skips_verify(tmp_path):
    """command_end --failed skips verify and writes a failure record.

    This is the spine-side parity fix for subgraph failures: the walker never
    calls verify after a branch error, so command_end with failed_reason should
    not call verify either.
    """
    _probe_task()
    node = _declared(_data_graph(task_ref="counting"), tmp_path)

    # Call end with failed_reason — verify should be skipped.
    result = items.command_end(node, tmp_path, 0, failed_reason="subgraph crashed")
    assert "error" in result
    assert result["passed"] is False
    assert result["score"] == 0.0
    assert "subgraph crashed" in result["error_text"]
    assert result["error_kind"] == "RuntimeError"

    # The error record is in facts.
    facts = json.loads((tmp_path / "facts.json").read_text())
    scores = facts["item_scores"]
    assert len(scores) == 1
    assert scores[0]["error_text"] is not None


def test_all_fail_counts_items_not_records(tmp_path):
    """The all-fail check counts items processed, not records (Ask 5).

    3 items where only alpha fails in verify; command_end for the last item
    does NOT raise ItemsError (1 of 3 is not all).
    """
    from refactory_dsh import task as task_module

    class OneOfThreeFails(task_module.Task):
        name = "one_of_three_fails"

        def instances(self):
            return [
                task_module.TaskInstance(id="alpha"),
                task_module.TaskInstance(id="beta"),
                task_module.TaskInstance(id="gamma"),
            ]

        def setup(self, instance, workspace):
            pass

        def verify(self, instance, workspace):
            if instance.id == "alpha":
                raise ValueError("alpha failed")
            return task_module.VerifyResult(passed=True, score=1.0)

    task_module.register_task("one_of_three_fails", OneOfThreeFails)
    node = _declared(_data_graph(task_ref="one_of_three_fails"), tmp_path)

    # Drive all 3 items through begin+end.
    for i in range(3):
        items.command_begin(node, tmp_path, i)
        items.command_end(node, tmp_path, i)

    # The last item's end should NOT raise (1 of 3 is not all).
    # If it raised, the test would have failed above.
    facts = json.loads((tmp_path / "facts.json").read_text())
    scores = facts["item_scores"]
    failed = [s for s in scores if s.get("error_text") is not None]
    assert len(failed) == 1, "exactly one item should have failed"


def test_csv_id_fallback_pinned(tmp_path):
    """CSV items without an id column get index-based ids (Ask 8).

    Targets the csv_item.setdefault('id', str(csv_index)) at executor.py:416.
    Fails when that single line is reverted (CSV rows would have no 'id' key).
    """
    csv_path = tmp_path / "items.csv"
    csv_path.write_text("value\nfoo\nbar\n")

    node = nn.Node(
        "data", "data", subgraph_entry="w", subgraph_exit="w",
        source_path="items.csv", source_format="csv",
    )
    loaded = load_items(node, tmp_path)
    assert [item["id"] for item in loaded] == ["0", "1"]


def test_walker_index_fallback_pinned(tmp_path):
    """Items with no id key get index-based ids from the walker (Ask 8).

    Inline items [{'value':'x'},{'value':'y'}] with no 'id' key; run through
    the walker; item_scores ids are ['0', '1']. Targets the
    item_id = str(item.get('id') or index) at executor.py:820.
    """
    from refactory_dsh import task as task_module
    from refactory_dsh.nn.executor import CURRENT_ITEM, execute

    class PassAllTask(task_module.Task):
        name = "pass_all_index"

        def verify(self, instance, workspace):
            return task_module.VerifyResult(passed=True, score=1.0)

    task_module.register_task("pass_all_index", PassAllTask)

    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        task_ref="pass_all_index",
        inline_items=[{"value": "x"}, {"value": "y"}],
        writes=(CURRENT_ITEM,), produces=("item_scores",),
    ))
    graph.add(nn.Node(
        "work", "op", call="loop_ops:record_item",
        reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
    ))
    graph.add(nn.Node("report", "op", call="loop_ops:write_best",
                      writes=("best.json",), produces=("best",)))
    graph.connect("data", "report")

    result = execute(graph, tmp_path / "run")
    assert result.exit_ok, result.error
    scores = result.facts["item_scores"]
    assert [s["id"] for s in scores] == ["0", "1"]


def test_failed_kind_parameter(tmp_path):
    """command_end --failed-kind overrides error_kind (Ask 6).

    When failed_kind='timeout' is passed, the record's error_kind is 'timeout'
    instead of 'RuntimeError'.
    """
    _probe_task()
    node = _declared(_data_graph(task_ref="counting"), tmp_path)
    items.command_begin(node, tmp_path, 0)

    result = items.command_end(
        node, tmp_path, 0,
        failed_reason="timed out", failed_kind="timeout",
    )
    assert result["error_kind"] == "timeout"
    assert result["passed"] is False
