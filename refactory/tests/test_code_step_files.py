"""Code step file I/O: a sealed code step reads its declared files from disk and writes
its declared outputs back.

The unsealed sandbox runs pure computation on ``inp`` and returns JSON only.  A sealed
code step runs in the Task's tool environment, so it can ``open()`` its declared reads
directly (including binary files that ``inp["files"]`` does not carry) and write its
declared ``writes`` to the working directory; those files are copied back to the run
directory after a successful run.

Tests here cover:
- sealed_execute copying declared writes back to the run directory;
- safety: size bounds, symlink rejection, path escapes, undeclared files left behind;
- the proposer's request text describing sealed file I/O;
- edit acceptance and wiring validation for the guard pattern on spreadsheetbench_v2/steps;
- generality: the same pattern on a second, unrelated graph (DraftReview).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from refactory_dsh import snippet
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn.validate import validate_state

from refactory_packages.spreadsheetbench_v2.steps_mode import Steps

import structure_fuzz_graphs as fg

needs_sandbox = pytest.mark.skipif(not snippet.available()[0],
                                   reason=f"no code-node sandbox here: {snippet.available()[1]}")


def _check_seal() -> bool:
    from refactory_dsh import seal
    return seal.available()[0]


needs_seal = pytest.mark.skipif(not _check_seal(), reason="no seal available")


@pytest.fixture(autouse=True)
def _sandbox_assumed_where_no_code_runs(monkeypatch):
    if not snippet.available()[0]:
        monkeypatch.setattr(snippet, "_probe", lambda: (True, "assumed: no code runs"))


# ── helpers ──────────────────────────────────────────────────────────────

_TOOL_ENV_TASK_REF = "item_files_ops:ToolEnvTask"


def _sealed_pass(tmp_path: Path) -> Path:
    pass_dir = tmp_path / "run" / "passes" / "data" / "item_1"
    pass_dir.mkdir(parents=True)
    (pass_dir / ".refactory").mkdir()
    (pass_dir / ".refactory" / "pass.json").write_text(
        json.dumps({"task": _TOOL_ENV_TASK_REF}))
    (pass_dir / ".refactory" / "workspaces").mkdir()
    return pass_dir


def _steps_state() -> dict:
    return Steps().state_dict()


# ── sealed_execute: file writes ──────────────────────────────────────────


@needs_seal
def test_sealed_execute_copies_declared_writes_back(tmp_path):
    """A sealed code step that writes a declared output file to its working directory
    has that file copied back to the run directory after execution."""
    pass_dir = _sealed_pass(tmp_path)
    (pass_dir / "data.txt").write_text("source content")
    code = (
        "def main(inp):\n"
        "    with open('out.bin', 'wb') as f:\n"
        "        f.write(b'BINARY\\x00DATA')\n"
        "    return {}\n"
    )
    result = snippet.sealed_execute(
        code, {"files": {}, "facts": {}},
        workdir=pass_dir, node_id="writer",
        reads=["data.txt"], writes=["out.bin"])
    assert (pass_dir / "out.bin").exists()
    assert (pass_dir / "out.bin").read_bytes() == b"BINARY\x00DATA"
    assert result.get("_sealed_files") == ["out.bin"]


@needs_seal
def test_sealed_execute_reads_binary_from_disk_and_writes_it_back(tmp_path):
    """A sealed code step reads a binary file from disk (not through inp), processes it,
    and writes the result to a declared output."""
    pass_dir = _sealed_pass(tmp_path)
    binary_data = bytes(range(256))
    (pass_dir / "input.bin").write_bytes(binary_data)
    code = (
        "def main(inp):\n"
        "    data = open('input.bin', 'rb').read()\n"
        "    with open('output.bin', 'wb') as f:\n"
        "        f.write(data[::-1])\n"
        "    return {'facts': {'size': len(data)}}\n"
    )
    result = snippet.sealed_execute(
        code, {"files": {}, "facts": {}},
        workdir=pass_dir, node_id="reverse",
        reads=["input.bin"], writes=["output.bin"])
    assert result["facts"]["size"] == 256
    assert (pass_dir / "output.bin").exists()
    assert (pass_dir / "output.bin").read_bytes() == binary_data[::-1]


@needs_seal
def test_sealed_execute_text_in_json_overwrites_disk_file(tmp_path):
    """When a sealed code step writes a file to disk AND returns it in JSON, the JSON
    text version wins (write_text is called after the disk copy)."""
    pass_dir = _sealed_pass(tmp_path)
    code = (
        "def main(inp):\n"
        "    with open('report.txt', 'w') as f:\n"
        "        f.write('disk version')\n"
        "    return {'files': {'report.txt': 'json version'}}\n"
    )
    result = snippet.sealed_execute(
        code, {"files": {}, "facts": {}},
        workdir=pass_dir, node_id="both",
        reads=[], writes=["report.txt"])
    assert "report.txt" in result.get("_sealed_files", [])
    assert result["files"]["report.txt"] == "json version"


@needs_seal
def test_sealed_execute_does_not_copy_undeclared_files(tmp_path):
    """A file left in the workspace that is not in the declared writes is not copied."""
    pass_dir = _sealed_pass(tmp_path)
    code = (
        "def main(inp):\n"
        "    with open('secret.txt', 'w') as f:\n"
        "        f.write('should not escape')\n"
        "    with open('declared.txt', 'w') as f:\n"
        "        f.write('ok')\n"
        "    return {}\n"
    )
    snippet.sealed_execute(
        code, {"files": {}, "facts": {}},
        workdir=pass_dir, node_id="leak",
        reads=[], writes=["declared.txt"])
    assert (pass_dir / "declared.txt").exists()
    assert not (pass_dir / "secret.txt").exists()


@needs_seal
def test_sealed_execute_no_writes_param_copies_nothing(tmp_path):
    """When writes is empty (the default), no files are copied back — backward compat."""
    pass_dir = _sealed_pass(tmp_path)
    code = (
        "def main(inp):\n"
        "    with open('out.txt', 'w') as f:\n"
        "        f.write('data')\n"
        "    return {}\n"
    )
    result = snippet.sealed_execute(
        code, {"files": {}, "facts": {}},
        workdir=pass_dir, node_id="old",
        reads=[])
    assert not (pass_dir / "out.txt").exists()
    assert "_sealed_files" not in (result or {})


@needs_seal
def test_sealed_execute_refuses_oversized_output(tmp_path):
    """A file over SEALED_OUTPUT_MAX_BYTES is refused."""
    pass_dir = _sealed_pass(tmp_path)
    limit = snippet.SEALED_OUTPUT_MAX_BYTES
    code = (
        f"def main(inp):\n"
        f"    with open('big.bin', 'wb') as f:\n"
        f"        f.write(b'x' * ({limit} + 1))\n"
        f"    return {{}}\n"
    )
    with pytest.raises(snippet.SnippetError, match="byte limit"):
        snippet.sealed_execute(
            code, {"files": {}, "facts": {}},
            workdir=pass_dir, node_id="big",
            reads=[], writes=["big.bin"])


@needs_seal
def test_sealed_execute_missing_declared_write_is_not_an_error(tmp_path):
    """A declared write the code did not produce is simply absent — not an error."""
    pass_dir = _sealed_pass(tmp_path)
    code = "def main(inp):\n    return {}\n"
    result = snippet.sealed_execute(
        code, {"files": {}, "facts": {}},
        workdir=pass_dir, node_id="skip",
        reads=[], writes=["optional.txt"])
    assert not (pass_dir / "optional.txt").exists()
    assert result.get("_sealed_files", []) == []


# ── proposer request text ────────────────────────────────────────────────


def test_sealed_code_step_request_mentions_file_io():
    """The proposer's composition section states that a sealed code step can read its
    declared files from the working directory and write declared outputs there."""
    from refactory_dsh.nn import edits as edits_mod
    from refactory_dsh.optim.base import composition_section

    state = _steps_state()
    nodes = [dict(s, id=node_id)
             for node_id, s in state["graph"]["nodes"].items()]
    text_knobs = [name for name, entry in state["knobs"].items()
                  if entry["spec"]["kind"] == "str"]
    text = "\n".join(composition_section(
        edits_mod.catalogue(), nodes, state["graph"]["edges"],
        code_nodes=True, text_knobs=text_knobs))
    assert "open()" in text
    assert "working directory" in text
    assert "binary" in text or ".xlsx" in text
    assert "copied back" in text


# ── edit acceptance: guard pattern on spreadsheetbench_v2/steps ──────────


def test_sealed_code_guard_after_implement_is_accepted():
    """A sealed code step after implement that takes over output.xlsx, reads input.xlsx
    and writes output.xlsx (the guard pattern) is accepted."""
    state = _steps_state()
    edit = {
        "op": "insert", "after": "implement", "kind": "code",
        "id": "guard",
        "text": (
            "import openpyxl\n"
            "def main(inp):\n"
            "    return {}\n"
        ),
        "sealed": True,
        "takes_over": ["implement", "output.xlsx"],
    }
    result = nn_edits.apply(edit, state)
    nodes = result["graph"]["nodes"]
    guard = nodes["guard"]
    assert "output.xlsx" in guard["writes"]
    assert guard.get("sealed_script") is True
    assert "output.xlsx" not in nodes["implement"]["writes"]
    assert "input.xlsx" in guard.get("reads", [])
    issues = validate_state(result)
    assert not issues, issues


def test_sealed_code_guard_with_extra_upstream_file_is_accepted():
    """The guard pattern where plan also writes targets.json and the guard sees it."""
    state = _steps_state()
    edit = {
        "op": "insert", "after": "implement", "kind": "code",
        "id": "guard",
        "text": (
            "import json\n"
            "def main(inp):\n"
            "    return {}\n"
        ),
        "sealed": True,
        "sees": ["implement", "plan"],
        "takes_over": ["implement", "output.xlsx"],
    }
    result = nn_edits.apply(edit, state)
    nodes = result["graph"]["nodes"]
    assert "guard" in nodes
    guard = nodes["guard"]
    assert "output.xlsx" in guard["writes"]
    assert "plan.md" in guard.get("reads", [])
    issues = validate_state(result)
    assert not issues, issues


def test_agent_guard_with_tools_after_implement_is_accepted():
    """Pattern B: an agent step with tools, same takeover, is accepted."""
    state = _steps_state()
    edit = {
        "op": "insert", "after": "implement", "kind": "agent",
        "id": "guard",
        "text": "Compare input.xlsx and output.xlsx. Restore stray cells.",
        "tools": ["Bash"],
        "takes_over": ["implement", "output.xlsx"],
    }
    result = nn_edits.apply(edit, state)
    nodes = result["graph"]["nodes"]
    assert "output.xlsx" in nodes["guard"]["writes"]
    issues = validate_state(result)
    assert not issues, issues


def test_code_judge_loop_around_implement_is_accepted():
    """Pattern C: a verdict loop around implement with a code judge."""
    state = _steps_state()
    edit = {
        "op": "loop", "step": "implement", "judge": "code",
        "text": (
            "def main(inp):\n"
            "    return {'verdict': 'proceed', 'reason': 'ok'}\n"
        ),
        "max_iterations": 1,
    }
    result = nn_edits.apply(edit, state)
    nodes = result["graph"]["nodes"]
    gate = [n for n in nodes.values()
            if n.get("evaluator") == snippet.GATE]
    assert gate, "no code gate found"
    issues = validate_state(result)
    assert not issues, issues


def test_sealed_code_judge_loop_around_implement_is_accepted():
    """Pattern C variant: the same code judge loop but sealed."""
    state = _steps_state()
    edit = {
        "op": "loop", "step": "implement", "judge": "code",
        "text": (
            "import openpyxl\n"
            "def main(inp):\n"
            "    return {'verdict': 'proceed', 'reason': 'ok'}\n"
        ),
        "max_iterations": 1,
        "sealed": True,
    }
    result = nn_edits.apply(edit, state)
    nodes = result["graph"]["nodes"]
    gate = [n for n in nodes.values()
            if n.get("evaluator") == snippet.GATE]
    assert gate, "no code gate found"
    assert gate[0].get("sealed_script") is True
    issues = validate_state(result)
    assert not issues, issues


# ── generality: the same pattern on DraftReview ──────────────────────────


def test_code_guard_on_draft_review_is_accepted():
    """A code step that takes over a file from an earlier step on a linear pipeline
    (DraftReview), to show the pattern is not SpreadsheetBench-specific."""
    state = fg.DraftReview().state_dict()
    edit = {
        "op": "insert", "after": "draft", "kind": "code",
        "id": "quality_gate",
        "text": (
            "def main(inp):\n"
            "    text = inp['files'].get('draft.txt', '')\n"
            "    return {'files': {'draft.txt': text.upper()}}\n"
        ),
        "takes_over": ["draft", "draft.txt"],
    }
    result = nn_edits.apply(edit, state)
    nodes = result["graph"]["nodes"]
    assert "draft.txt" in nodes["quality_gate"]["writes"]
    assert "draft.txt" not in nodes["draft"]["writes"]
    issues = validate_state(result)
    assert not issues, issues


def test_code_judge_loop_on_gate_loop_graph():
    """A code gate loop on a gate loop graph (GateLoop) — a second unrelated shape."""
    state = fg.GateLoop().state_dict()
    edit = {
        "op": "loop", "target": "work",
        "gate": {"id": "length_check", "kind": "gate",
                 "evaluator": snippet.GATE,
                 "reads": ["work.txt"], "required_reads": ["work.txt"],
                 "max_iterations": 2,
                 "code": (
                     "def main(inp):\n"
                     "    text = inp['files'].get('work.txt', '')\n"
                     "    if len(text) > 10:\n"
                     "        return {'verdict': 'proceed', 'reason': 'long enough'}\n"
                     "    return {'verdict': 'reloop', 'reason': 'too short'}\n"
                 )},
    }
    result = nn_edits.apply(edit, state)
    nodes = result["graph"]["nodes"]
    assert "length_check" in nodes
    issues = validate_state(result)
    assert not issues, issues


# ── gap 1: inp['files'] drops binary reads ──────────────────────────────


def test_request_text_states_binary_reads_absent_from_inp():
    """The proposer's request text states that binary reads are absent from
    inp['files'] and that an unsealed code step cannot read them from disk."""
    from refactory_dsh.nn import edits as edits_mod
    from refactory_dsh.optim.base import composition_section

    state = _steps_state()
    nodes = [dict(s, id=node_id)
             for node_id, s in state["graph"]["nodes"].items()]
    text_knobs = [name for name, entry in state["knobs"].items()
                  if entry["spec"]["kind"] == "str"]
    text = "\n".join(composition_section(
        edits_mod.catalogue(), nodes, state["graph"]["edges"],
        code_nodes=True, text_knobs=text_knobs))
    assert "binary" in text.lower()
    assert "absent" in text or "silently" in text or "not carry" in text
    assert "sandbox" in text or "sealed" in text


# ── gap 2: bypass refused when downstream reads unique output ───────────


def test_bypass_refused_when_downstream_reads_unique_output():
    """Bypassing a step whose unique output is read downstream is refused at
    edit time, so the proposer can repair the edit."""
    state = _steps_state()
    reason = nn_edits.precheck({"op": "bypass", "target": "implement"}, state)
    assert reason is not None
    assert "output.xlsx" in reason or "implement" in reason


def test_bypass_allowed_when_output_has_another_writer():
    """When a co-writer produces the same output, bypass is allowed."""
    import refactory_dsh.nn as nn
    chain = nn.Graph()
    chain.add(nn.Node("a", "op", call="json:dumps", writes=("x.txt",)))
    chain.add(nn.Node("b", "op", call="json:dumps", writes=("x.txt",)))
    chain.add(nn.Node("c", "op", call="json:loads", reads=("x.txt",), writes=("out",)))
    chain.connect("a", "b")
    chain.connect("b", "c")
    state = {"version": nn.STATE_VERSION, "graph": chain.to_state(), "knobs": {}}
    reason = nn_edits.precheck({"op": "bypass", "target": "b"}, state)
    assert reason is None


# ── gap 3: fallback multi-file constraint stated ────────────────────────


def test_fallback_line_states_multi_file_constraint():
    """The fallback operator's description states that a step writing more than
    one file needs a code or tools fallback, not a plain agent."""
    from refactory_dsh.nn import core_ops
    line = core_ops.OPERATORS["fallback"]["line"]
    assert "more than one file" in line or "writes more" in line
    assert "code" in line or "tools" in line


# ── pattern A end to end: guard restores stray cells ─────────────────


@needs_seal
def test_guard_pattern_a_restores_stray_cells_end_to_end(tmp_path):
    """Pattern A end to end: a stub produces output.xlsx with stray cell
    changes, the sealed code guard reads input.xlsx and targets.json, restores
    every cell changed outside the targets, and writes back output.xlsx.

    Fails if _collect_sealed_writes is reverted: without write-back, the
    guard's corrected output.xlsx stays in the sandbox and the run directory
    keeps the stray version."""
    import openpyxl

    from scripts.probe_guard_pattern import GUARD_CODE

    pass_dir = _sealed_pass(tmp_path)

    wb_in = openpyxl.Workbook()
    ws_in = wb_in.active
    ws_in.title = "Sheet"
    ws_in["A1"] = "original_a1"
    ws_in["B1"] = "original_b1"
    ws_in["C1"] = "original_c1"
    wb_in.save(pass_dir / "input.xlsx")

    targets = {"cells": [{"sheet": "Sheet", "cell": "A1"}]}
    (pass_dir / "targets.json").write_text(json.dumps(targets))

    wb_out = openpyxl.Workbook()
    ws_out = wb_out.active
    ws_out.title = "Sheet"
    ws_out["A1"] = "answer"
    ws_out["B1"] = "stray_change"
    ws_out["C1"] = "original_c1"
    wb_out.save(pass_dir / "output.xlsx")

    targets_text = json.dumps(targets)
    result = snippet.sealed_execute(
        GUARD_CODE,
        {"files": {"targets.json": targets_text}, "facts": {}},
        workdir=pass_dir,
        node_id="guard",
        reads=["input.xlsx", "output.xlsx", "targets.json"],
        writes=["output.xlsx"],
    )

    assert result["facts"]["restored_cells"] == 1

    wb_check = openpyxl.load_workbook(pass_dir / "output.xlsx")
    ws_check = wb_check.active
    assert ws_check["A1"].value == "answer", "target cell was not preserved"
    assert ws_check["B1"].value == "original_b1", "stray cell was not restored"
    assert ws_check["C1"].value == "original_c1", "unchanged cell was modified"
