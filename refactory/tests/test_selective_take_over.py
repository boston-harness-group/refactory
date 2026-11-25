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

from refactory_packages.spreadsheetbench_v2.steps_mode import (
    Steps, reply_path, session_path, step_writes,
)


def _steps_state() -> dict:
    return Steps().state_dict()


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
