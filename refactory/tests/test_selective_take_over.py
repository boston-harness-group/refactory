"""Selective ``takes_over``: take one output while the rest pass through.

The SpreadsheetBench V2 steps pipeline is:

    prepare -> has_case -> inspect -> plan -> implement -> verify -> finalize

Each agent step writes (reply, hand-off files, session log).  ``implement``
writes ``(implement_reply.txt, output.xlsx, implement/session.json)`` and
``finalize`` reads every step's reply and session.

A proposer wants to insert a step after ``implement`` that takes over only
``output.xlsx`` (e.g. an auto-revert step), so the new step rewrites that file
while ``implement``'s reply and session pass through unchanged.  With full
``takes_over`` that is impossible: it takes all outputs, so finalize would need
the new step to produce implement's reply and session.

Selective ``takes_over`` as ``[step, output, ...]`` takes only the named
outputs; the rest keep their original names and downstream readers keep
reading them from the original step.
"""
from __future__ import annotations

from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.optim.base import _example_refusal, _with_outputs_said

import structure_fuzz_graphs as fg
from refactory_packages.spreadsheetbench_v2.steps_mode import (
    Steps, reply_path, session_path, step_writes,
)


def _steps_state() -> dict:
    return Steps().state_dict()


def _nearest_upstream_writer(nodes: dict, edges: list, reader_id: str, filename: str) -> str | None:
    """BFS backward from *reader_id*; return the nearest node that writes *filename*."""
    preds: dict[str, list[str]] = {}
    for e in edges:
        preds.setdefault(e["target"], []).append(e["source"])
    visited: set[str] = set()
    queue = list(preds.get(reader_id, []))
    while queue:
        nid = queue.pop(0)
        if nid in visited:
            continue
        visited.add(nid)
        if filename in nodes.get(nid, {}).get("writes", []):
            return nid
        queue.extend(preds.get(nid, []))
    return None


def test_selective_take_over_keeps_non_taken_outputs_for_downstream():
    """Insert a code step after implement that takes over only output.xlsx.

    - The new step writes output.xlsx (taken) and its own default outputs.
    - implement keeps implement_reply.txt and implement/session.json.
    - verify still reads output.xlsx (from the new step).
    - finalize still reads implement_reply.txt and implement/session.json
      (from implement, unchanged).
    """
    state = _steps_state()
    nodes = state["graph"]["nodes"]

    impl_writes_before = nodes["implement"]["writes"]
    assert "output.xlsx" in impl_writes_before
    assert reply_path("implement") in impl_writes_before
    assert session_path("implement") in impl_writes_before

    result = nn_edits.apply(
        {"op": "insert", "after": "implement", "kind": "code",
         "text": "def main(inp):\n    return {'files': {'output.xlsx': inp.files['output.xlsx']}}",
         "id": "auto_revert",
         "takes_over": ["implement", "output.xlsx"]},
        state)

    nodes = result["graph"]["nodes"]
    new_step = nodes["auto_revert"]
    impl_after = nodes["implement"]

    # The new step writes output.xlsx (taken) plus its own default fact output.
    assert "output.xlsx" in new_step["writes"]

    # implement keeps its reply and session under their original names.
    assert reply_path("implement") in impl_after["writes"]
    assert session_path("implement") in impl_after["writes"]

    # implement's output.xlsx is renamed (taken over), not in its writes anymore.
    assert "output.xlsx" not in impl_after["writes"]

    # verify reads output.xlsx (now from the new step, since verify is downstream).
    verify = nodes["verify"]
    assert "output.xlsx" in verify["reads"] or "output.xlsx" in verify.get("required_reads", [])

    # finalize still reads implement's reply and session by their original names.
    finalize = nodes["finalize"]
    assert reply_path("implement") in finalize["reads"]
    assert session_path("implement") in finalize["reads"]


def test_selective_take_over_refuses_nonexistent_output():
    """Taking over an output the target doesn't write is refused."""
    state = _steps_state()
    try:
        nn_edits.apply(
            {"op": "insert", "after": "implement", "kind": "code",
             "text": "def main(inp): return {}",
             "takes_over": ["implement", "no_such_file.txt"]},
            state)
        raise AssertionError("should have been refused")
    except Exception as exc:
        assert "not among its outputs" in str(exc)


def test_selective_take_over_refuses_agent_without_tools():
    """An agent without tools can't do selective takeover: it writes only its reply."""
    state = _steps_state()
    try:
        nn_edits.apply(
            {"op": "insert", "after": "implement", "kind": "agent",
             "text": "Revert output.",
             "takes_over": ["implement", "output.xlsx"]},
            state)
        raise AssertionError("should have been refused")
    except Exception as exc:
        assert "agent step without tools" in str(exc)


def test_full_take_over_still_works():
    """The existing full takeover (string form) still works unchanged."""
    state = _steps_state()

    # Full takeover of implement by a code step.
    result = nn_edits.apply(
        {"op": "insert", "after": "implement", "kind": "code",
         "text": "def main(inp):\n    return {}",
         "id": "rewriter",
         "takes_over": "implement"},
        state)

    nodes = result["graph"]["nodes"]
    new_step = nodes["rewriter"]
    impl_after = nodes["implement"]

    # Full takeover: new step gets ALL of implement's original writes.
    for w in step_writes("implement"):
        assert w in new_step["writes"], f"{w} should be in new step's writes"

    # implement's writes are all renamed.
    for w in step_writes("implement"):
        assert w not in impl_after["writes"], f"{w} should be renamed on implement"


def test_selective_take_over_graded_file_reaches_scorer():
    """output.xlsx is the graded file; selective takeover must reach the scorer."""
    state = _steps_state()

    # Taking over output.xlsx from implement: the scorer (Task) grades output.xlsx,
    # so the takeover reaches something (the scorer), and is accepted.
    result = nn_edits.apply(
        {"op": "insert", "after": "implement", "kind": "code",
         "text": "def main(inp):\n    return {'files': {'output.xlsx': inp.files['output.xlsx']}}",
         "id": "auto_revert",
         "takes_over": ["implement", "output.xlsx"]},
        state)

    # The edit succeeds; the new step's output.xlsx is what gets graded.
    assert "auto_revert" in result["graph"]["nodes"]


# ── follow-up 1: agent steps with tools ──────────────────────────────────


def test_selective_take_over_agent_with_tools():
    """An agent step WITH tools can do selective takeover (sealed, Bash)."""
    state = _steps_state()
    result = nn_edits.apply(
        {"op": "insert", "after": "implement", "kind": "agent",
         "text": "Revert output if wrong.",
         "id": "auto_revert",
         "tools": ["Bash"],
         "takes_over": ["implement", "output.xlsx"]},
        state)

    nodes = result["graph"]["nodes"]
    new_step = nodes["auto_revert"]
    impl_after = nodes["implement"]

    assert "output.xlsx" in new_step["writes"]
    assert reply_path("implement") in impl_after["writes"]
    assert session_path("implement") in impl_after["writes"]
    assert "output.xlsx" not in impl_after["writes"]


def test_selective_take_over_sealed_agent_with_tools():
    """A sealed agent step with tools can do selective takeover."""
    state = _steps_state()
    result = nn_edits.apply(
        {"op": "insert", "after": "implement", "kind": "agent",
         "text": "Revert output if wrong.",
         "id": "auto_revert",
         "tools": ["Bash"],
         "sealed": True,
         "takes_over": ["implement", "output.xlsx"]},
        state)

    nodes = result["graph"]["nodes"]
    new_step = nodes["auto_revert"]
    assert "output.xlsx" in new_step["writes"]
    assert reply_path("implement") in nodes["implement"]["writes"]


# ── follow-up 2: generality on non-SpreadsheetBench graph shapes ────────


def _fuzz_state(mode_cls: type) -> dict:
    return mode_cls().state_dict()


def test_selective_take_over_inside_fork_branch():
    """Selective takeover of a step inside a fork branch.

    ForkJoin: setup → fork(left, right_a→right_b) → join → decide → done.
    right_a writes right_a.txt and produces right_fact.  Take over only
    right_a.txt: right_b (downstream in the branch) reads it from the new
    step; decide still reads right_fact from right_a.
    """
    state = _fuzz_state(fg.ForkJoin)
    nodes = state["graph"]["nodes"]
    assert "right_a.txt" in nodes["right_a"]["writes"]
    assert "right_fact" in nodes["right_a"]["produces"]

    result = nn_edits.apply(
        {"op": "insert", "after": "right_a", "kind": "code",
         "text": "def main(inp):\n    return {'files': {'right_a.txt': 'rewritten'}}",
         "id": "patch_a",
         "takes_over": ["right_a", "right_a.txt"]},
        state)

    nodes = result["graph"]["nodes"]
    new_step = nodes["patch_a"]
    original = nodes["right_a"]

    assert "right_a.txt" in new_step["writes"]
    assert "right_a.txt" not in original["writes"]
    assert "right_fact" in original["produces"]

    right_b = nodes["right_b"]
    assert "right_a.txt" in right_b["reads"] or any(
        "right_a.txt" in (right_b.get(k) or []) for k in ("reads", "required_reads"))


def test_selective_take_over_before_fork():
    """Selective takeover of a step whose outputs feed into a fork.

    ForkJoin: setup writes brief.txt and produces brief.  Take over only
    brief.txt: both fork branches read it from the new step; brief (fact)
    still comes from setup.
    """
    state = _fuzz_state(fg.ForkJoin)
    result = nn_edits.apply(
        {"op": "insert", "after": "setup", "kind": "code",
         "text": "def main(inp):\n    return {'files': {'brief.txt': 'filtered'}}",
         "id": "filter_brief",
         "takes_over": ["setup", "brief.txt"]},
        state)

    nodes = result["graph"]["nodes"]
    assert "brief.txt" in nodes["filter_brief"]["writes"]
    assert "brief.txt" not in nodes["setup"]["writes"]
    assert "brief" in nodes["setup"]["produces"]


def test_selective_take_over_inside_loop():
    """Selective takeover of a step whose output feeds a loop body.

    GateLoop: prepare(task.txt, task) → work → review → reloop/finish.
    Take over only task.txt from prepare: work reads it from the new step;
    task (fact used in work's prompt) still comes from prepare.
    """
    state = _fuzz_state(fg.GateLoop)
    nodes = state["graph"]["nodes"]
    assert "task.txt" in nodes["prepare"]["writes"]
    assert "task" in nodes["prepare"]["produces"]

    result = nn_edits.apply(
        {"op": "insert", "after": "prepare", "kind": "code",
         "text": "def main(inp):\n    return {'files': {'task.txt': 'augmented'}}",
         "id": "augment",
         "takes_over": ["prepare", "task.txt"]},
        state)

    nodes = result["graph"]["nodes"]
    assert "task.txt" in nodes["augment"]["writes"]
    assert "task.txt" not in nodes["prepare"]["writes"]
    assert "task" in nodes["prepare"]["produces"]

    work = nodes["work"]
    assert "task.txt" in work["reads"] or any(
        "task.txt" in (work.get(k) or []) for k in ("reads", "required_reads"))


def test_selective_take_over_in_linear_pipeline():
    """Selective takeover in a straight pipeline (DraftReview).

    intake(brief.txt, brief) → draft → review → finalize → ship.
    Take over only brief.txt from intake: draft reads it from the new step;
    brief (fact used in draft's prompt) still comes from intake.
    """
    state = _fuzz_state(fg.DraftReview)
    nodes = state["graph"]["nodes"]
    assert "brief.txt" in nodes["intake"]["writes"]
    assert "brief" in nodes["intake"]["produces"]

    result = nn_edits.apply(
        {"op": "insert", "after": "intake", "kind": "code",
         "text": "def main(inp):\n    return {'files': {'brief.txt': 'redacted'}}",
         "id": "redact",
         "takes_over": ["intake", "brief.txt"]},
        state)

    nodes = result["graph"]["nodes"]
    assert "brief.txt" in nodes["redact"]["writes"]
    assert "brief.txt" not in nodes["intake"]["writes"]
    assert "brief" in nodes["intake"]["produces"]


# ── enrichment with the list form ────────────────────────────────────────


def test_with_outputs_said_names_only_selected_files():
    """_with_outputs_said with [step, output, ...] enriches the prompt to name
    only the selected files, not every file the target writes."""
    state = _steps_state()
    impl_writes = state["graph"]["nodes"]["implement"]["writes"]
    assert len(impl_writes) > 1

    edit = {"op": "insert", "after": "implement", "kind": "agent",
            "text": "Revert output.", "tools": ["Bash"],
            "takes_over": ["implement", "output.xlsx"]}
    enriched = _with_outputs_said([edit], state)
    text = enriched[0]["text"]

    assert "output.xlsx" in text
    for w in impl_writes:
        if w != "output.xlsx":
            assert w not in text, f"{w} should not appear: only selected files"


def test_with_outputs_said_full_takeover_names_all_files():
    """_with_outputs_said with the string form names every file the target writes."""
    state = _steps_state()
    edit = {"op": "insert", "after": "implement", "kind": "agent",
            "text": "Rewrite.", "takes_over": "implement"}
    enriched = _with_outputs_said([edit], state)
    text = enriched[0]["text"]

    for w in step_writes("implement"):
        assert w in text, f"{w} should appear in full-takeover enrichment"


def test_example_refusal_selective_checks_only_selected():
    """_example_refusal with [step, output, ...] checks writability of only the
    selected files, not all of the target's files."""
    state = _steps_state()
    edit_ok = {"op": "insert", "after": "implement", "kind": "agent",
               "text": "Fix output.", "tools": ["Bash"],
               "takes_over": ["implement", "output.xlsx"]}
    assert _example_refusal(state, [edit_ok]) is None

    edit_no_tools = {"op": "insert", "after": "implement", "kind": "agent",
                     "text": "Fix output.",
                     "takes_over": ["implement", "output.xlsx"]}
    reason = _example_refusal(state, [edit_no_tools])
    assert reason is not None
    assert "agent" in reason.lower()


# ── read-modify-write takeover (verify reads and writes the same file) ──


def test_selective_take_over_after_read_modify_write_step():
    """Insert a sealed code step after verify that takes over verify's output.xlsx.

    verify both READS output.xlsx (from implement) and WRITES output.xlsx
    (the corrected version).  Taking over verify's write must NOT rename
    implement's write of the same file, or verify would lose the input it
    reads from implement.

    - implement keeps writing output.xlsx (verify's read source).
    - verify's write of output.xlsx is renamed (taken over).
    - verify still reads output.xlsx (from implement, unchanged).
    - The new step reads verify's renamed output and writes output.xlsx.
    - finalize reads output.xlsx (from the new step).
    """
    state = _steps_state()
    nodes = state["graph"]["nodes"]

    assert "output.xlsx" in nodes["implement"]["writes"]
    assert "output.xlsx" in nodes["verify"]["writes"]
    assert "output.xlsx" in nodes["verify"]["reads"]

    result = nn_edits.apply(
        {"op": "insert", "after": "verify", "kind": "code",
         "text": "def main(inp):\n    return {'files': {'output.xlsx': inp.files['output.xlsx']}}",
         "id": "revert_regressions",
         "sealed": True,
         "takes_over": ["verify", "output.xlsx"]},
        state)

    nodes = result["graph"]["nodes"]
    new_step = nodes["revert_regressions"]
    impl_after = nodes["implement"]
    verify_after = nodes["verify"]

    assert "output.xlsx" in new_step["writes"]

    # implement keeps writing output.xlsx: verify's read source is not renamed.
    assert "output.xlsx" in impl_after["writes"]

    # verify's write of output.xlsx is renamed (taken over).
    assert "output.xlsx" not in verify_after["writes"]

    # verify still reads output.xlsx (from implement, unchanged).
    assert "output.xlsx" in verify_after["reads"]

    # verify keeps its other outputs.
    assert reply_path("verify") in verify_after["writes"]
    assert session_path("verify") in verify_after["writes"]

    # Wiring: verify reads output.xlsx from implement (the nearest upstream writer).
    edges = result["graph"]["edges"]
    assert _nearest_upstream_writer(nodes, edges, "verify", "output.xlsx") == "implement"

    # Wiring: finalize reads output.xlsx from the new step.
    assert _nearest_upstream_writer(nodes, edges, "finalize", "output.xlsx") == "revert_regressions"


# ── write-only target: co-writers are still renamed ──────────────────────


def test_write_only_target_co_writer_is_renamed():
    """Take over a file the target writes but does NOT read.

    DraftReview: intake → draft → review → finalize → ship.
    Insert writer_a after intake (writes memo.txt) and writer_b after finalize
    (writes memo.txt, doesn't read it).  Take over writer_b's memo.txt with a
    new step.  writer_a is a co-writer and must be renamed (the fix's exclusion
    does not fire because writer_b doesn't read memo.txt).
    """
    state = fg.DraftReview().state_dict()

    s1 = nn_edits.apply(
        {"op": "insert", "after": "intake", "kind": "code",
         "text": "def main(inp):\n    return {'files': {'memo.txt': 'v1'}}",
         "id": "writer_a", "writes": ["memo.txt"]},
        state)

    s2 = nn_edits.apply(
        {"op": "insert", "after": "finalize", "kind": "code",
         "text": "def main(inp):\n    return {'files': {'memo.txt': 'v2'}}",
         "id": "writer_b", "writes": ["memo.txt"]},
        s1)

    nodes_before = s2["graph"]["nodes"]
    assert "memo.txt" in nodes_before["writer_a"]["writes"]
    assert "memo.txt" in nodes_before["writer_b"]["writes"]
    assert "memo.txt" not in nodes_before["writer_b"].get("reads", [])

    s3 = nn_edits.apply(
        {"op": "insert", "after": "writer_b", "kind": "code",
         "text": "def main(inp):\n    return {'files': {'memo.txt': 'v3'}}",
         "id": "taker", "takes_over": ["writer_b", "memo.txt"]},
        s2)

    nodes = s3["graph"]["nodes"]

    # taker owns memo.txt now.
    assert "memo.txt" in nodes["taker"]["writes"]

    # writer_b's memo.txt is renamed (taken over).
    assert "memo.txt" not in nodes["writer_b"]["writes"]

    # writer_a is a co-writer: its memo.txt is also renamed.
    assert "memo.txt" not in nodes["writer_a"]["writes"]


# ── two co-writers, only one feeds the target's read ─────────────────────


def _two_writer_fork_state() -> dict:
    """A fork where both branches write ``shared.txt``.

    Branch A: c1 → target → reader.  Branch B: c2 (dead-end).
    ``target`` reads ``shared.txt`` from ``c1`` (read-modify-write).
    ``c2`` writes ``shared.txt`` but cannot reach ``target``.
    """
    from refactory_dsh import nn as _nn

    graph = fg._graph([
        fg.op("setup", writes=("seed.txt",)),
        _nn.Node("split", "fork", targets=("c1", "c2")),
        fg.op("c1", reads=("seed.txt",), writes=("shared.txt",)),
        fg.op("target", reads=("shared.txt",), writes=("shared.txt",)),
        fg.op("reader", reads=("shared.txt",), writes=("result.txt",)),
        fg.op("c2", writes=("shared.txt",)),
        _nn.Node("merge", "join", sources=("reader", "c2")),
        fg.op("final", reads=("result.txt",), produces=("finished",)),
    ], [
        ("setup", "split"),
        ("split", "merge"),
        ("c1", "target"),
        ("target", "reader"),
        ("reader", "merge"),
        ("c2", "merge"),
        ("merge", "final"),
    ])

    class _TwoWriterFork(_nn.Mode):
        name = "two_writer_fork"
        def build(self):
            return graph

    return _TwoWriterFork().state_dict()


def test_two_co_writers_only_feeder_is_kept():
    """Two co-writers of ``shared.txt``.  Only the one that feeds the target's
    read is kept; the other is renamed.

    Fork: branch A has c1 → target → reader, branch B has c2.
    Both c1 and c2 write shared.txt.  target reads shared.txt from c1
    (read-modify-write).

    Taking over target's shared.txt:
    - c1 feeds target's read → NOT renamed (excluded by the fix).
    - c2 is in a different branch, can't reach target → renamed.
    """
    state = _two_writer_fork_state()
    nodes_before = state["graph"]["nodes"]
    assert "shared.txt" in nodes_before["c1"]["writes"]
    assert "shared.txt" in nodes_before["c2"]["writes"]
    assert "shared.txt" in nodes_before["target"]["reads"]
    assert "shared.txt" in nodes_before["target"]["writes"]

    result = nn_edits.apply(
        {"op": "insert", "after": "target", "kind": "code",
         "text": "def main(inp):\n    return {'files': {'shared.txt': 'taken'}}",
         "id": "taker", "takes_over": ["target", "shared.txt"]},
        state)

    nodes = result["graph"]["nodes"]

    # taker owns shared.txt.
    assert "shared.txt" in nodes["taker"]["writes"]

    # c1 feeds target's read: its shared.txt is NOT renamed.
    assert "shared.txt" in nodes["c1"]["writes"]

    # c2 is in a different branch, doesn't feed the read: renamed.
    assert "shared.txt" not in nodes["c2"]["writes"]

    # target's shared.txt is renamed (taken over).
    assert "shared.txt" not in nodes["target"]["writes"]

    # target still reads shared.txt (from c1, unchanged).
    assert "shared.txt" in nodes["target"]["reads"]
