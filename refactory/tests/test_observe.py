"""Watching a run: progress while it happens, diffs, and a manifest.

The loop stays a Python process. What these test is that it is *readable* from
outside one: a reader that only knows the run directory can tail a run, compare a
candidate with its parent, and know which files to render.
"""

from __future__ import annotations

import json
from pathlib import Path

from refactory_dsh import observe
from refactory_dsh.loop import Loop, LoopConfig

MODE = "circle_packing/adaptive"


def _config(root: Path, **overrides: object) -> LoopConfig:
    base = dict(epochs=2, batch_size=3, optimizer="random", seed=5, runs_root=root)
    base.update(overrides)
    return LoopConfig(**base)  # type: ignore[arg-type]


def _run(tmp_path: Path, **overrides: object) -> Path:
    loop = Loop(MODE, _config(tmp_path / "run", **overrides), project=Path("."))
    loop.run()
    loop.report()
    return tmp_path / "run"


# ── progress ───────────────────────────────────────────────────────────────


def test_progress_records_every_step_of_the_loop(tmp_path):
    root = _run(tmp_path)
    steps = [record["step"] for record in observe.progress(root)]
    assert steps[0] == "init"
    assert steps[-1] == "report"
    assert set(steps) == {
        "init", "forward", "loss", "backward", "optimizer", "step", "scheduler",
        "holdout", "report",
    }
    # The held-out audit runs after the search and before the report, so a reader
    # watching the file sees it happen where it did.
    assert steps.index("scheduler") < steps.index("holdout") < steps.index("report")


def test_progress_records_one_line_per_candidate_evaluated(tmp_path):
    """An epoch can be an hour, so a per-epoch line is not progress."""
    root = _run(tmp_path)
    forward = [r for r in observe.progress(root) if r["step"] == "forward"]
    # 2 epochs x batch_size 3, then the 3 the last epoch proposed, scored before the report
    assert len(forward) == 9, forward
    assert all(r["candidate"] for r in forward)
    assert all(r["of"] == 3 for r in forward)


def test_progress_is_written_as_the_run_goes_not_at_the_end(tmp_path):
    """A reader must be able to read the file while the loop is still inside it."""
    root = tmp_path / "run"
    loop = Loop(MODE, _config(root), project=Path("."))
    loop.init()
    during = observe.progress(root)
    assert [r["step"] for r in during] == ["init"]
    loop.forward()
    assert any(r["step"] == "forward" for r in observe.progress(root))


def test_progress_can_be_switched_off(tmp_path):
    root = _run(tmp_path, progress=False)
    assert not (root / observe.PROGRESS).exists()
    assert observe.progress(root) == []


def test_progress_tail_reads_only_what_is_new(tmp_path):
    root = _run(tmp_path)
    everything = observe.progress(root)
    head = observe.progress(root, after=0)[:3]
    assert len(head) == 3
    assert observe.progress(root, after=3) == everything[3:]
    assert observe.progress(root, after=len(everything)) == []


def test_progress_lines_carry_the_counters_a_reader_needs(tmp_path):
    root = _run(tmp_path)
    for record in observe.progress(root):
        assert record["epoch"] >= 0
        assert record["evaluations"] >= 0
        assert record["time"] > 0


def test_progress_says_which_bound_stopped_the_run(tmp_path):
    root = _run(tmp_path)
    last = [r for r in observe.progress(root) if r["step"] == "scheduler"][-1]
    assert last["bound"] == "epochs"
    assert last["another"] is False


# ── diffs ──────────────────────────────────────────────────────────────────


def test_a_diff_names_the_changed_parameter():
    diff = observe.diff_states(
        {"knobs": {"grid": {"value": 18}}},
        {"knobs": {"grid": {"value": 24}}},
    )
    assert diff["knobs"] == {"grid": {"before": 18, "after": 24}}


def test_a_diff_names_added_removed_and_changed_parameters():
    diff = observe.diff_states(
        {"knobs": {"a": {"value": 1}, "b": {"value": 2}}},
        {"knobs": {"a": {"value": 9}, "c": {"value": 3}}},
    )
    assert diff["knobs"]["a"] == {"before": 1, "after": 9}
    assert diff["knobs"]["b"] == {"removed": True, "before": 2}
    assert diff["knobs"]["c"] == {"added": True, "after": 3}


def test_a_diff_names_added_and_removed_nodes_and_edges():
    before = {
        "graph": {
            "nodes": {"generate": {"op": "generate"}},
            "edges": [{"source": "start", "target": "generate"}],
        }
    }
    after = {
        "graph": {
            "nodes": {"generate": {"op": "generate"}, "critic": {"op": "critic"}},
            "edges": [
                {"source": "start", "target": "generate"},
                {"source": "generate", "target": "critic"},
            ],
        }
    }
    diff = observe.diff_states(before, after)
    assert diff["nodes"] == {"added": ["critic"], "removed": [], "changed": []}
    assert diff["edges"]["added"] == [["generate", "critic", ""]]
    assert diff["edges"]["removed"] == []


def test_a_diff_names_a_changed_node_field():
    before = {"graph": {"nodes": {"score": {"op": "score", "produces": []}}, "edges": []}}
    after = {"graph": {"nodes": {"score": {"op": "score", "produces": ["score"]}}, "edges": []}}
    diff = observe.diff_states(before, after)
    assert diff["nodes"]["changed"] == [
        {"id": "score", "field": "produces", "before": [], "after": ["score"]}
    ]


def test_a_diff_of_a_state_with_itself_is_empty():
    state = {"knobs": {"a": {"value": 1}}, "graph": {"nodes": {}, "edges": []}}
    assert observe.diff_states(state, state) == {
        "knobs": {},
        "nodes": {"added": [], "removed": [], "changed": []},
        "edges": {"added": [], "removed": []},
    }


def test_a_candidate_diff_is_read_from_the_journal_not_the_run_directory(tmp_path):
    """`candidate.json` holds what was measured; only the journal holds the state."""
    root = _run(tmp_path)
    journal = [json.loads(line) for line in (root / "journal.jsonl").read_text().splitlines()]
    candidates = [r for r in journal if r["event"] == "candidate"]
    child = next(r for r in candidates if r["parent"])
    diff = observe.candidate_diff(root, child["id"])
    assert diff["candidate"] == child["id"]
    assert diff["parent"] == child["parent"]
    assert diff["seed"] is False
    assert diff["diff"]["knobs"] or diff["diff"]["nodes"]["changed"], diff


def test_the_seed_has_nothing_to_be_compared_with(tmp_path):
    root = _run(tmp_path)
    journal = [json.loads(line) for line in (root / "journal.jsonl").read_text().splitlines()]
    seed = next(r for r in journal if r["event"] == "candidate" and not r["parent"])
    diff = observe.candidate_diff(root, seed["id"])
    assert diff["seed"] is True
    assert diff["parent"] is None


def test_an_unknown_candidate_is_an_error_not_an_empty_diff(tmp_path):
    root = _run(tmp_path)
    try:
        observe.candidate_diff(root, "nope")
    except FileNotFoundError as error:
        assert "nope" in str(error)
    else:  # pragma: no cover - the assertion is the point
        raise AssertionError("an unknown candidate must not read as an empty diff")


def test_a_diff_renders_as_markdown():
    rendered = observe.render_diff(
        observe.diff_states(
            {"knobs": {"grid": {"value": 18}}},
            {"knobs": {"grid": {"value": 24}}, "graph": {"nodes": {}, "edges": []}},
        )
    )
    assert "grid: 18 → 24" in rendered


def test_a_seed_diff_says_so_rather_than_rendering_nothing(tmp_path):
    assert "seed" in observe.render_diff({})


# ── manifest ───────────────────────────────────────────────────────────────


def test_the_manifest_lists_what_a_reader_renders(tmp_path):
    root = _run(tmp_path)
    manifest = observe.artifact_manifest(root)
    by_name = {entry["name"]: entry for entry in manifest["artifacts"]}
    assert by_name["report"]["path"] == "report.md"
    assert by_name["report"]["media_type"] == "text/markdown"
    assert by_name["progress"]["media_type"] == "application/x-ndjson"
    assert by_name["report"]["exists"] is True
    assert by_name["gradient"]["exists"] is True
    assert by_name["progress"]["exists"] is True


def test_the_manifest_is_written_by_the_run(tmp_path):
    root = _run(tmp_path)
    on_disk = json.loads((root / "manifest.json").read_text())
    assert on_disk["run"] == str(root)
    assert {e["name"] for e in on_disk["artifacts"]} == {
        e["name"] for e in observe.artifact_manifest(root)["artifacts"]
    }


def test_the_manifest_names_the_per_candidate_files(tmp_path):
    root = _run(tmp_path)
    by_name = {e["name"]: e for e in observe.artifact_manifest(root)["artifacts"]}
    candidates = by_name["candidates"]
    assert candidates["exists"] is True
    assert len(candidates["files"]) == 9  # every candidate the two epochs proposed ran
    assert all(Path(root / path).exists() for path in candidates["files"])


def test_the_manifest_marks_absent_artifacts_absent(tmp_path):
    """A reader has to tell "not written yet" from "written and empty"."""
    root = tmp_path / "fresh"
    root.mkdir()
    by_name = {e["name"]: e for e in observe.artifact_manifest(root)["artifacts"]}
    assert by_name["report"]["exists"] is False
    assert by_name["candidates"]["exists"] is False


def test_a_graph_summary_sizes_up_a_candidate_without_loading_it(tmp_path):
    root = _run(tmp_path)
    record = json.loads((root / "journal.jsonl").read_text().splitlines()[1])
    summary = observe.graph_summary(record["state"])
    assert summary.endswith("edge(s)")
    assert "no graph" == observe.graph_summary({})
