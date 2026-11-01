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


def test_apply_corrections(ctx: Any) -> dict[str, Any]:
    """Test op that applies corrections from the ledger and records them as a fact."""
    from pathlib import Path as _P
    from refactory_dsh.design_ops import gate_corrections_due
    _, corrections = gate_corrections_due(ctx.workdir)
    correction_texts = [text for _, text in corrections]
    _P(ctx.workdir, "design.md").write_text("# Corrected design\n")
    return {"corrected": True, "correction_texts": correction_texts}
