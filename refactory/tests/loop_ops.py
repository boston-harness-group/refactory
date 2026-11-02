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


def fail_on_second_visit(ctx: Any) -> None:
    """Pass the first time, fail every time after: a run stopped right after a reloop."""
    visits = (ctx.read_json("visits.json", 0) or 0) + 1
    ctx.write_json("visits.json", visits)
    if visits > 1:
        raise RuntimeError("this op fails on its second visit")


def reloop_twice(ctx: Any) -> str:
    """`three_passes` on an explicit `reloop` edge, for a cycle with another gate in it."""
    visits = int(ctx.facts.get("passes", 0)) + 1
    ctx.set_fact("passes", visits)
    return "reloop" if visits < 3 else "proceed"


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
