"""Operations the executor tests resolve by name.

They live here rather than in the library because they exist to exercise the
executor, not to be an operation any real mode would call.
"""

from __future__ import annotations

from typing import Any


def write_number(ctx: Any) -> dict[str, Any]:
    value = int(ctx.knob("step", 1))
    ctx.write_json("number.json", {"value": value})
    return {"value": value}


def double(ctx: Any) -> dict[str, Any]:
    data = ctx.read_json("number.json", {"value": 0})
    doubled = int(data["value"]) * 2
    ctx.write_json("doubled.json", {"value": doubled})
    return {"doubled": doubled}


def undeclared_write(ctx: Any) -> None:
    ctx.write_json("surprise.json", {})


def always_retry(ctx: Any) -> str:
    return "retry"


def three_passes(ctx: Any) -> str:
    visits = int(ctx.facts.get("passes", 0)) + 1
    ctx.set_fact("passes", visits)
    return "retry" if visits < 3 else "proceed"


def explained_passes(ctx: Any) -> str:
    """`three_passes`, saying why each retry happens."""
    from refactory_dsh.nn import Verdict

    visits = int(ctx.facts.get("passes", 0)) + 1
    ctx.set_fact("passes", visits)
    return Verdict("retry", f"pass {visits} of 3") if visits < 3 else "proceed"


def bogus_verdict(ctx: Any) -> str:
    return "nonsense"


def boom(ctx: Any) -> None:
    raise ValueError("this op fails on purpose")


def write_best(ctx: Any) -> dict[str, Any]:
    return {"best": 42.0}


def noop(ctx: Any) -> None:
    """Do nothing — a placeholder for graphs that need a start node."""
    pass


def record_item(ctx: Any) -> dict[str, Any]:
    """Append the current item's id, so a test can see one pass per item."""
    item = ctx.read_json("current_item.json", {}) or {}
    seen = ctx.read_json("seen.json", {"ids": []}) or {"ids": []}
    seen.setdefault("ids", []).append(str(item.get("id")))
    ctx.write_json("seen.json", seen)
    return {"seen": seen["ids"]}


def explained_passes_in_a_fact(ctx: Any) -> str:
    """`explained_passes`, also leaving the reason as a fact a template can render."""
    from refactory_dsh.nn import Verdict

    visits = int(ctx.facts.get("passes", 0)) + 1
    ctx.set_fact("passes", visits)
    ctx.set_fact("why", f"pass {visits} of 3")
    return Verdict("retry", f"pass {visits} of 3") if visits < 3 else "proceed"


def pause(ctx: Any) -> None:
    """Take a moment, so a sibling branch finishes first."""
    import time

    time.sleep(0.3)


def retry_each_item_once(ctx: Any) -> str:
    """Send the current item's pass back once, saying why; then go on."""
    from refactory_dsh.nn import Verdict

    item = str((ctx.read_json("current_item.json", {}) or {}).get("id"))
    tried = ctx.read_json("tried.json", []) or []
    if item in tried:
        return "proceed"
    ctx.write_json("tried.json", [*tried, item])
    return Verdict("retry", f"item {item} needs another look")


#: A reason past the runner's 4000-character limit: a whole check log, line by line.
LONG_REASON = "\n".join(f"check log line {number}: tests/test_widget.py::test_{number} failed"
                        for number in range(150))


def long_reason_once(ctx: Any) -> str:
    """Send the work back once with a reason longer than the runner keeps, left as a fact too."""
    from refactory_dsh.nn import Verdict

    visits = int(ctx.facts.get("passes", 0)) + 1
    ctx.set_fact("passes", visits)
    ctx.set_fact("why", LONG_REASON)
    return Verdict("retry", LONG_REASON) if visits < 2 else "proceed"


def always_proceed(ctx: Any) -> str:
    return "proceed"


def always_reloop(ctx: Any) -> str:
    """A gate that always says reloop, for testing iteration caps."""
    from refactory_dsh.nn import Verdict
    return Verdict("reloop", reason="needs more work")


def fail_on_second_visit(ctx: Any) -> None:
    """Pass the first time, fail every time after: a run stopped right after a reloop."""
    visits = (ctx.read_json("visits.json", 0) or 0) + 1
    ctx.write_json("visits.json", visits)
    if visits > 1:
        raise RuntimeError("this op fails on its second visit")


def reloop_with_stdout(ctx: Any) -> str:
    """A gate that prints to stdout (extra context) then returns reloop once, then proceed.

    SWEEP-4: Tests that stdout before the verdict line is used as feedback,
    matching the spine's reasonOf(stdout) behavior.
    """
    from refactory_dsh.nn import Verdict

    visits = int(ctx.facts.get("passes", 0)) + 1
    ctx.set_fact("passes", visits)
    if visits < 2:
        print("extra context from stdout")
        return Verdict("reloop", reason="needs more work")
    return "proceed"


def reloop_twice(ctx: Any) -> str:
    """`three_passes` on an explicit `reloop` edge, for a cycle with another gate in it."""
    visits = int(ctx.facts.get("passes", 0)) + 1
    ctx.set_fact("passes", visits)
    return "reloop" if visits < 3 else "proceed"


def write_relative(ctx: Any) -> dict[str, Any]:
    """SWEEP-3: Write a file using a relative path to prove ops run in the run directory."""
    from pathlib import Path as _P
    _P("relative_output.txt").write_text("written from op\n")
    return {"relative_written": True}


def check_model_is_none(ctx: Any) -> dict[str, Any]:
    """SWEEP-5: Verify ctx.model is None for ops, matching the documented API."""
    return {"model_is_none": ctx.model is None}


def warn_on_stderr(ctx: Any) -> dict[str, Any]:
    """Gap 20: Print a run warning to stderr for run_warning event test."""
    import sys as _sys
    print("run warning: something happened", file=_sys.stderr)
    return {"warned": True}


def always_halt(ctx: Any) -> str:
    """A gate that always returns halt, for testing halt inside a fork branch."""
    return "halt"


def print_to_stdout(ctx: Any) -> dict[str, Any]:
    """An op that prints to stdout, for testing that the parent forwards it."""
    print("op stdout output: visible diagnostics")
    return {}


def stdout_then_raise(ctx: Any) -> None:
    """An op that prints to stdout then raises: the run that most needs its output."""
    print("op stdout before the raise")
    raise ValueError("op failed after printing to stdout")


def stderr_then_raise(ctx: Any) -> None:
    """An op that prints to stderr then raises, for testing stderr preservation."""
    import sys as _sys
    print("op stderr output: important debug info", file=_sys.stderr)
    raise ValueError("op failed after printing to stderr")


def parity_write_relative(ctx: Any) -> dict[str, Any]:
    """SWEEP-9: Write a relative path (SWEEP-3) and check ctx.model (SWEEP-5)."""
    from pathlib import Path as _P
    _P("parity_output.txt").write_text(f"knob_val={ctx.knobs.get('test_knob', '?')}\n")
    return {"ctx_model_is_none": ctx.model is None, "knob_applied": ctx.knobs.get("test_knob", "")}


def parity_fn_gate(ctx: Any) -> str:
    """SWEEP-9: fn gate that prints to stdout and reloops once, then proceeds."""
    from refactory_dsh.nn import Verdict
    visits = int(ctx.facts.get("parity_visits", 0)) + 1
    ctx.set_fact("parity_visits", visits)
    if visits < 2:
        print("parity stdout feedback")
        return Verdict("reloop", reason="needs another round")
    return "proceed"


def parity_apply_correction(ctx: Any) -> dict[str, Any]:
    """SWEEP-9: Apply corrections from the ledger."""
    from refactory_dsh.design_ops import gate_corrections_due
    _, corrections = gate_corrections_due(ctx.workdir)
    correction_texts = [text for _, text in corrections]
    ctx.set_fact("corrections_applied", correction_texts)
    return {"corrections_applied": correction_texts}


def parity_nested_fork_fail(ctx: Any) -> dict[str, Any]:
    """SWEEP-9: An op that always fails, for the optional-branch-first-node-failure path."""
    raise ValueError("optional branch first node failure (expected)")


def parity_read_knob(ctx: Any) -> dict[str, Any]:
    """SWEEP-9: An op that reads the test_knob knob value."""
    return {"knob_read": ctx.knobs.get("test_knob", "")}


def pid_reporting_gate(ctx: Any) -> str:
    """A fn gate that records the current PID as a fact and proceeds.

    Used by test_fn_gate_in_fork_branch_uses_forkserver: if the gate runs in
    a forkserver child, its PID differs from the parent; if in-process, it matches.
    """
    import os
    ctx.set_fact("gate_pid", os.getpid())
    return "proceed"


def test_apply_corrections(ctx: Any) -> dict[str, Any]:
    """Test op that applies corrections from the ledger and records them as a fact."""
    from pathlib import Path as _P
    from refactory_dsh.design_ops import gate_corrections_due
    _, corrections = gate_corrections_due(ctx.workdir)
    correction_texts = [text for _, text in corrections]
    _P(ctx.workdir, "design.md").write_text("# Corrected design\n")
    return {"corrected": True, "correction_texts": correction_texts}


def fail_on_a(ctx: Any) -> dict[str, Any]:
    """Like record_item, but raises for item a — used to test branch errors."""
    item = ctx.read_json("current_item.json", {}) or {}
    if str(item.get("id")) == "a":
        raise RuntimeError("branch failed for item a")
    seen = ctx.read_json("seen.json", {"ids": []}) or {"ids": []}
    seen.setdefault("ids", []).append(str(item.get("id")))
    ctx.write_json("seen.json", seen)
    return {"seen": seen["ids"]}


def fail_on_last_fails(ctx: Any) -> dict[str, Any]:
    """Like record_item, but raises for item 'last_fails' — data failure equivalence."""
    item = ctx.read_json("current_item.json", {}) or {}
    if str(item.get("id")) == "last_fails":
        raise RuntimeError("branch failed for last_fails")
    seen = ctx.read_json("seen.json", {"ids": []}) or {"ids": []}
    seen.setdefault("ids", []).append(str(item.get("id")))
    ctx.write_json("seen.json", seen)
    return {"seen": seen["ids"]}


def slow_op(ctx: Any) -> dict[str, Any]:
    """Like record_item, but sleeps 3 seconds — tests deadline not armed across branch."""
    import time
    time.sleep(3)
    item = ctx.read_json("current_item.json", {}) or {}
    seen = ctx.read_json("seen.json", {"ids": []}) or {"ids": []}
    seen.setdefault("ids", []).append(str(item.get("id")))
    ctx.write_json("seen.json", seen)
    return {"seen": seen["ids"]}


def fail_on_setup_fails(ctx: Any) -> dict[str, Any]:
    """Like record_item, but raises for item 'setup_fails' — equivalence test."""
    item = ctx.read_json("current_item.json", {}) or {}
    if str(item.get("id")) == "setup_fails":
        raise RuntimeError("branch failed for setup_fails")
    seen = ctx.read_json("seen.json", {"ids": []}) or {"ids": []}
    seen.setdefault("ids", []).append(str(item.get("id")))
    ctx.write_json("seen.json", seen)
    return {"seen": seen["ids"]}


def fork_branch_write_a(ctx: Any) -> dict[str, Any]:
    """Write a unique file via a relative path inside the run directory.

    Used by test_concurrent_fork_branch_ops_write_to_run_dir: two branches
    each write a file; both must land in the run directory.  Also records PID
    so the test can verify the op ran in a child process (forkserver) rather
    than in the parent (inline os.chdir).
    """
    import os
    from pathlib import Path as _P
    import time
    # Small jitter so both branches are truly concurrent.
    time.sleep(0.05)
    _P("branch_a_output.txt").write_text("written by branch a\n")
    ctx.set_fact("branch_a_pid", os.getpid())
    return {"branch_a_written": True}


def fork_branch_write_b(ctx: Any) -> dict[str, Any]:
    """Write a unique file via a relative path inside the run directory."""
    import os
    from pathlib import Path as _P
    import time
    time.sleep(0.05)
    _P("branch_b_output.txt").write_text("written by branch b\n")
    ctx.set_fact("branch_b_pid", os.getpid())
    return {"branch_b_written": True}


def fn_gate_plain_str_with_stdout(ctx: Any) -> str:
    """A fn gate that prints to stdout and returns a plain str (not Verdict).

    Used by test_fn_gate_plain_str_stdout_is_feedback: the captured stdout
    must appear as reloop feedback.
    """
    visits = int(ctx.facts.get("plain_gate_visits", 0)) + 1
    ctx.set_fact("plain_gate_visits", visits)
    if visits < 2:
        print("stdout from plain str gate")
        return "reloop"
    return "proceed"


def error_on_item(ctx: Any) -> dict[str, Any]:
    """Like record_item, but raises for item 'error_item' — both-runtimes test."""
    item = ctx.read_json("current_item.json", {}) or {}
    if str(item.get("id")) == "error_item":
        raise RuntimeError("subgraph error for error_item")
    seen = ctx.read_json("seen.json", {"ids": []}) or {"ids": []}
    seen.setdefault("ids", []).append(str(item.get("id")))
    ctx.write_json("seen.json", seen)
    return {"seen": seen["ids"]}


# ── pass isolation ops ────────────────────────────────────────────────


def assert_isolated(ctx: Any) -> dict[str, Any]:
    """Assert that a previous item's artifacts are invisible.

    Reads ``current_item.json`` to get the item id.  If the id is not ``"a"``
    (the first item), asserts that ``marker.txt`` does NOT exist in
    ``ctx.workdir`` and that ``ctx.facts.get("leak_test")`` is ``None``.
    Then writes ``marker.txt`` and sets fact ``leak_test``.

    Fails when reverted (without isolation, the second item sees the first
    item's ``marker.txt`` and ``leak_test`` fact).
    """
    item = ctx.read_json("current_item.json", {}) or {}
    item_id = str(item.get("id"))
    if item_id != "a":
        from pathlib import Path

        marker = Path(ctx.workdir) / "marker.txt"
        assert not marker.exists(), (
            f"item {item_id!r} can see marker.txt from a previous pass"
        )
        assert ctx.facts.get("leak_test") is None, (
            f"item {item_id!r} can see leak_test={ctx.facts.get('leak_test')!r}"
        )
    ctx.write_text("marker.txt", f"from_{item_id}")
    ctx.set_fact("leak_test", f"from_{item_id}")
    return {}


def write_undeclared_fact(ctx: Any) -> dict[str, Any]:
    """Set a fact not in the node's ``produces`` — tests leak filtering."""
    ctx.set_fact("undeclared_secret", "leaked")
    return {}


def read_run_level_input(ctx: Any) -> dict[str, Any]:
    """Read ``run_input.txt`` from ``ctx.workdir`` and assert its content.

    With pass isolation the file lives at run level and is symlinked into the
    pass directory by ``prepare_pass_dir``.  Fails without the symlink.
    """
    content = ctx.read_text("run_input.txt")
    assert content == "run-level-data", f"expected 'run-level-data', got {content!r}"
    return {}


def write_scripted_answer(ctx: Any) -> dict[str, Any]:
    """Write a scripted ``<answer>B</answer>`` response to ``answer.txt``.

    Simulates what the agent ``solve`` node + model would do in the LiveMath
    solver graph.  Used by the spine drain tests so the deterministic drain
    can run the graph without an actual model.
    """
    ctx.write_text("answer.txt", "<answer>B</answer>")
    return {}


def halt_on_b(ctx: Any) -> dict[str, Any]:
    """Record the item, then halt the run when item id is ``"b"`` — once only.

    Used to test resume mid-pass: item ``"a"`` completes, the run halts on
    ``"b"``, and resume should skip ``"a"`` and re-enter ``"b"``'s pass
    directory.  On the second visit (resume) the halt does not fire because
    ``.refactory/halted_b`` was written at the run level on the first halt.
    """
    from pathlib import Path
    from refactory_dsh.nn.executor import HaltStop

    item = ctx.read_json("current_item.json", {}) or {}
    item_id = str(item.get("id"))
    seen = ctx.read_json("seen.json", {"ids": []}) or {"ids": []}
    seen.setdefault("ids", []).append(item_id)
    ctx.write_json("seen.json", seen)
    if item_id == "b":
        # Only halt on the first visit. The flag file is at the run-level
        # workdir's parent (two levels up from a pass dir, or just the workdir
        # if no isolation), using a deterministic name.
        # We look for the flag by searching up from workdir for a passes/ dir.
        wd = Path(ctx.workdir)
        # Check if we're inside a pass directory (passes/<node>/<item>_<index>/)
        if wd.parent.parent.name == "passes":
            run_dir = wd.parent.parent.parent
        else:
            run_dir = wd
        flag = run_dir / ".refactory" / "halted_b"
        if not flag.exists():
            flag.parent.mkdir(parents=True, exist_ok=True)
            flag.write_text("halted")
            raise HaltStop("data", reason="test halt on item b")
    return {"seen": seen["ids"]}


# ── pass isolation review fixes (PR #561) ─────────────────────────────

def _current_id(ctx: Any) -> str:
    item = ctx.read_json("current_item.json", {}) or {}
    return str(item.get("id"))


def write_prompt(ctx: Any) -> dict[str, Any]:
    """Upstream of the data node: leave a run-level ``prompt.txt`` for the passes."""
    ctx.write_text("prompt.txt", "from prep")
    return {}


def read_and_write_notes(ctx: Any) -> dict[str, Any]:
    """In a pass: read the run-level ``notes.md`` and directory ``data/``, then write both.

    Fails (the item errors) when the run's content is not visible in the pass.
    """
    item = str((ctx.read_json("current_item.json", {}) or {}).get("id"))
    notes = ctx.read_text("notes.md")
    assert notes.startswith("RUN"), notes
    assert ctx.read_text("data/seed.txt") == "seed", "data/ not read through"
    ctx.write_text("notes.md", notes + "+" + item)
    ctx.write_text(f"data/{item}.txt", item)
    return {}


def read_prompt(ctx: Any) -> dict[str, Any]:
    """In a pass: read the upstream ``prompt.txt``; FileNotFoundError when it is not read through."""
    assert ctx.read_text("prompt.txt") == "from prep"
    return {}


def log_and_stop_on_b(ctx: Any) -> dict[str, Any]:
    """Log each run of an item to ``$PASS_TEST_LOG``; stop the run on item ``b`` once.

    The walker stops on a halt; the spine's op runs as a command, so it stops the
    way a killed run does -- the drain dies mid-command (``PASS_TEST_RUNTIME=spine``).
    The flag file beside the log makes the stop happen once, so the resume runs b.
    """
    import os
    import signal
    from pathlib import Path

    item_id = _current_id(ctx)
    log = Path(os.environ["PASS_TEST_LOG"])
    with log.open("a") as handle:
        handle.write(item_id + "\n")
    flag = log.with_suffix(".stopped")
    if item_id == "b" and not flag.exists():
        flag.write_text("stopped")
        if os.environ.get("PASS_TEST_RUNTIME") == "spine":
            os.kill(os.getppid(), signal.SIGKILL)
            raise SystemExit(1)
        from refactory_dsh.nn.executor import HaltStop

        raise HaltStop("data", reason="test stop on item b")
    return {}


def seed_shared(ctx: Any) -> dict[str, Any]:
    """Upstream of the data node: a run-level list fact."""
    ctx.set_fact("shared", ["seed"])
    return {}


def append_shared_in_place(ctx: Any) -> dict[str, Any]:
    """In a pass: record what ``shared`` looks like, then append to it in place."""
    import json
    from pathlib import Path

    shared = ctx.facts.get("shared")
    Path(ctx.workdir, "shared_seen.json").write_text(json.dumps(shared))
    shared.append("x")
    return {}


def tamper_run_input(ctx: Any) -> dict[str, Any]:
    """In a pass: read the run-level ``run_input.txt``, then write over it.

    The node declares no writes, so write enforcement does not stop it; only the
    pass's read-through protection does.
    """
    item_id = _current_id(ctx)
    ctx.write_text("input_seen.txt", ctx.read_text("run_input.txt"))
    ctx.write_text("run_input.txt", f"tampered by {item_id}")
    return {}


def note_item(ctx: Any) -> dict[str, Any]:
    """In a pass: set the declared, non-score fact ``note`` for this item."""
    ctx.set_fact("note", f"from {_current_id(ctx)}")
    return {}



# ── F21: a fork inside a data node's per-item pass, on both runtimes ────────
#
# One graph runs on the walker and, lowered, on the spine: branch_a and branch_b
# each write a file, and the pass's exit scores the item from both, so an item
# whose branch did not run (or ran wrongly) scores differently.


def _log_visit(ctx: Any, branch: str) -> str:
    """Append ``<item>:<branch>`` to ``$PASS_TEST_LOG`` when it is set; return the item id."""
    import os
    from pathlib import Path

    item_id = _current_id(ctx)
    if os.environ.get("PASS_TEST_LOG"):
        with Path(os.environ["PASS_TEST_LOG"]).open("a") as handle:
            handle.write(f"{item_id}:{branch}\n")
    return item_id


def fork_branch_a(ctx: Any) -> dict[str, Any]:
    """Branch a: write ``a_out.txt``, 0.25 for item ``1`` and 0.5 for any other item."""
    item_id = _log_visit(ctx, "a")
    ctx.write_text("a_out.txt", "0.25" if item_id == "1" else "0.5")
    return {}


def fork_branch_b(ctx: Any) -> dict[str, Any]:
    """Branch b as an op: write ``b_out.txt`` with the text the agent branch replies."""
    _log_visit(ctx, "b")
    ctx.write_text("b_out.txt", "agent_answer")
    return {}


def fork_branch_b_fail_on_0(ctx: Any) -> dict[str, Any]:
    """Branch b that raises ``ValueError`` on item ``0`` and is :func:`fork_branch_b` otherwise."""
    if _current_id(ctx) == "0":
        _log_visit(ctx, "b-failed")
        raise ValueError("deliberate branch failure")
    return fork_branch_b(ctx)


def fork_branch_b_stop_on_1(ctx: Any) -> dict[str, Any]:
    """Branch b that stops the run once, on item ``1``, before writing its output.

    The walker stops on a halt; the spine's op runs as a command, so it stops the
    way a killed run does -- the drain dies mid-command (``PASS_TEST_RUNTIME=spine``),
    as :func:`log_and_stop_on_b` does. The flag beside the log makes it stop once.
    """
    import os
    import signal
    from pathlib import Path

    item_id = _log_visit(ctx, "b")
    flag = Path(os.environ["PASS_TEST_LOG"]).with_suffix(".stopped")
    if item_id == "1" and not flag.exists():
        flag.write_text("stopped")
        if os.environ.get("PASS_TEST_RUNTIME") == "spine":
            os.kill(os.getppid(), signal.SIGKILL)
            raise SystemExit(1)
        from refactory_dsh.nn.executor import HaltStop

        raise HaltStop("data", reason="test stop in item 1's fork")
    ctx.write_text("b_out.txt", "agent_answer")
    return {}


def fork_branch_b_control_on_1(ctx: Any) -> dict[str, Any]:
    """Branch b that, on item ``1``, files a person's graph control for the spine.

    Writes ``$FORK_TEST_CONTROL`` (``stop`` or ``pause``) as a request in the run's
    ``.refactory/graph-controls`` (the run directory is ``$FORK_TEST_RUN``, since this
    runs inside the item's pass directory), the way ``refactory graph-control`` does,
    then writes its output: the spine applies the control at its next boundary.
    """
    import json
    import os
    from pathlib import Path

    fork_branch_b(ctx)
    if _current_id(ctx) == "1":
        run = Path(os.environ["FORK_TEST_RUN"])
        invocation_file = run / ".refactory" / "invocation.json"
        invocation = (json.loads(invocation_file.read_text()).get("id")
                      if invocation_file.exists() else None)
        controls = run / ".refactory" / "graph-controls"
        controls.mkdir(parents=True, exist_ok=True)
        (controls / "001.json").write_text(json.dumps({
            "run_id": str(run), "invocation": invocation,
            "action": os.environ["FORK_TEST_CONTROL"],
        }))
    return {}


def score_from_branches(ctx: Any) -> dict[str, Any]:
    """After the join: score the item from both branches' files into ``item_result.json``.

    The score is ``a_out.txt``'s number plus 0.5 when ``b_out.txt`` says
    ``agent_answer``; a missing file adds nothing.
    """
    try:
        a = float(ctx.read_text("a_out.txt"))
    except FileNotFoundError:
        a = 0.0
    try:
        b = 0.5 if ctx.read_text("b_out.txt").strip() == "agent_answer" else 0.0
    except FileNotFoundError:
        b = 0.0
    ctx.write_json("item_result.json", {"score": a + b})
    return {}


from refactory_dsh.task import Task, TaskInstance, VerifyResult  # noqa: E402


class ForkBranchTask(Task):
    """Items ``0``, ``1``, ``2``, each scored from its pass's ``item_result.json``.

    Both runtimes load it as ``loop_ops:ForkBranchTask``. ``$FORK_TEST_SLOW_ITEM``
    names an item whose verify first sleeps ``$FORK_TEST_SLOW_SECONDS`` (the
    per-item deadline tests).
    """

    name = "fork_branch_task"

    def instances(self, **kw: Any) -> list[TaskInstance]:
        return [TaskInstance(id=str(i), metadata={}) for i in range(3)]

    def setup(self, instance: TaskInstance, workspace: Any, **kw: Any) -> None:
        pass

    def verify(self, instance: TaskInstance, workspace: Any, **kw: Any) -> VerifyResult:
        import json
        import os
        import time
        from pathlib import Path

        if instance.id == os.environ.get("FORK_TEST_SLOW_ITEM"):
            time.sleep(float(os.environ.get("FORK_TEST_SLOW_SECONDS", "0")))
        result = Path(workspace) / "item_result.json"
        if not result.exists():
            return VerifyResult(passed=False, score=0.0, details="item_result.json missing")
        score = float(json.loads(result.read_text())["score"])
        return VerifyResult(passed=score >= 1.0, score=score)


def count_visit(ctx: Any) -> dict[str, Any]:
    """Count this node's visits in visits.json, for checking a gate's reloop cap."""
    visits = int(ctx.read_json("visits.json", 0) or 0) + 1
    ctx.write_json("visits.json", visits)
    return {"visits": visits}


def barrier_item(ctx: Any) -> dict[str, Any]:
    """Wait until ``BARRIER_K`` items are in flight at once; time out otherwise.

    Passes only when the data node runs that many items concurrently, so it shows a
    node's effective parallelism on either runtime.
    """
    import os
    import time
    from pathlib import Path

    item_id = str((ctx.read_json("current_item.json", {}) or {}).get("id", "?"))
    barrier = Path(os.environ["BARRIER_DIR"])
    want = int(os.environ["BARRIER_K"])
    (barrier / f"ready_{item_id}").write_text("1")
    deadline = time.monotonic() + float(os.environ.get("BARRIER_TIMEOUT_S", "10"))
    while len(list(barrier.glob("ready_*"))) < want:
        if time.monotonic() > deadline:
            raise TimeoutError(f"only {len(list(barrier.glob('ready_*')))} of {want} items in flight")
        time.sleep(0.02)
    (barrier / f"through_{item_id}").write_text("1")
    return {"counted": item_id}


# ── a check-and-retry loop that cannot settle every item (test_loop_exhaustion) ──

#: The attempt at which each item's answer comes out right; item ``3`` never does.
EXHAUST_SOLVES_AT = {"0": 1, "1": 2, "2": 1, "3": 99}


def exhaust_solve(ctx: Any) -> dict[str, Any]:
    """Attempt the item: ``answer.txt`` is ``right`` from its `EXHAUST_SOLVES_AT` attempt on."""
    item_id = _current_id(ctx)
    attempts = int(ctx.read_json("attempts.json", 0) or 0) + 1
    ctx.write_json("attempts.json", attempts)
    right = attempts >= EXHAUST_SOLVES_AT.get(item_id, 99)
    ctx.write_text("answer.txt", "right" if right else f"wrong {attempts}")
    return {}


def exhaust_check(ctx: Any) -> Any:
    """The checker gate: proceed on a right answer, else send the solver back with why."""
    from refactory_dsh.nn import Verdict

    answer = ctx.read_text("answer.txt").strip()
    return "proceed" if answer == "right" else Verdict("reloop", reason=f"{answer} is not right")


def exhaust_score(ctx: Any) -> dict[str, Any]:
    """Grade the answer as it stands: 1.0 when right, 0.25 when wrong (a failed item is 0)."""
    answer = ctx.read_text("answer.txt").strip()
    ctx.write_json("item_result.json", {"score": 1.0 if answer == "right" else 0.25})
    return {}


class ExhaustTask(Task):
    """Items ``0``-``3``, each scored from its pass's ``item_result.json``, else from its
    ``answer.txt`` as `exhaust_score` grades it (a pass that ends at the solver)."""

    name = "exhaust_task"

    def instances(self, **kw: Any) -> list[TaskInstance]:
        return [TaskInstance(id=str(i), metadata={}) for i in range(4)]

    def setup(self, instance: TaskInstance, workspace: Any, **kw: Any) -> None:
        pass

    def verify(self, instance: TaskInstance, workspace: Any, **kw: Any) -> VerifyResult:
        import json
        from pathlib import Path

        result = Path(workspace) / "item_result.json"
        answer = Path(workspace) / "answer.txt"
        if result.exists():
            score = float(json.loads(result.read_text())["score"])
        elif answer.exists():
            score = 1.0 if answer.read_text().strip() == "right" else 0.25
        else:
            return VerifyResult(passed=False, score=0.0, details="no answer")
        return VerifyResult(passed=score >= 1.0, score=score)


def no_review_yet(ctx: Any) -> None:
    """Write the review file a step reads before any review exists."""
    ctx.write_text("review.txt", "no review yet\n")
