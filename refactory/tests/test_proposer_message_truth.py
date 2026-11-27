"""Each message the outer-loop proposer reads is accurate.

One test per finding from the message-truth audit, each through the real path
(a refusal raised by applying the edit, or a request text rendered by the real
builder), plus a completeness test for no-answer causes.
"""

from __future__ import annotations

from test_pipeline_view import _mode


# ── helpers ──────────────────────────────────────────────────────────────────


def _attempt(*edits):
    from refactory_dsh.optim import Candidate, Optimizer, Proposal

    mode = _mode("livemath/skill_graph")
    seed = Candidate.from_state(mode.state_dict())
    return Optimizer().attempt(
        Proposal(edits=list(edits)),
        seed,
        mode.invariants(),
        True,
        (),
        protected=mode.edit_protected(),
    )


# ── 1. takes_over refusal splits wrong-kind from no-outputs ─────────────────


def test_takes_over_a_gate_says_wrong_kind_not_no_outputs():
    """A gate is the wrong kind; the refusal must not say 'no outputs'."""
    result = _attempt(
        {"op": "loop", "step": "decide"},
        {"op": "insert", "before": "decide", "kind": "agent", "text": "Check.",
         "takes_over": "decide_check"},
    )
    assert isinstance(result, list), result
    msg = result[0]
    assert "only agent or op steps can be taken over" in msg, msg
    assert "no outputs" not in msg, msg


def test_takes_over_a_gate_via_route_says_wrong_kind():
    """The same split in _route_take_over."""
    result = _attempt(
        {"op": "loop", "step": "decide"},
        {"op": "route", "after": "compare", "judge": "agent", "text": "Way?",
         "to": {"optionA": "precision",
                "optionB": {"kind": "agent", "text": "Alt.",
                            "takes_over": "decide_check"}}},
    )
    assert isinstance(result, list), result
    msg = result[0]
    assert "only agent or op steps can be taken over" in msg, msg
    assert "no outputs" not in msg, msg


# ── 2. every recordable no-answer cause has wording ─────────────────────────


def test_every_recordable_cause_has_wording_in_no_answer_causes():
    """A cause record_spent can record must have an entry in NO_ANSWER_CAUSES,
    so the proposer never sees a raw identifier.  The set comes from code
    (RECORDABLE_CAUSES), not literals here."""
    from refactory_dsh.node_retry import RECORDABLE_CAUSES
    from refactory_dsh.optim.base import NO_ANSWER_CAUSES

    missing = RECORDABLE_CAUSES - set(NO_ANSWER_CAUSES)
    assert not missing, f"NO_ANSWER_CAUSES has no wording for: {sorted(missing)}"


def test_recordable_causes_covers_retry_on():
    """RECORDABLE_CAUSES must contain every RETRY_ON kind, so a new retry kind
    added to the graph without wording fails the suite."""
    from refactory_dsh.nn.graph import RETRY_ON
    from refactory_dsh.node_retry import RECORDABLE_CAUSES

    missing = set(RETRY_ON) - RECORDABLE_CAUSES
    assert not missing, f"RECORDABLE_CAUSES is missing RETRY_ON members: {sorted(missing)}"


def test_missing_cause_returns_only_members_of_missing_causes():
    """missing_cause() returns only members of MISSING_CAUSES (or None), so a
    new return value without wording is caught."""
    import subprocess

    from refactory_dsh.model import (
        CompletionFailed,
        CompletionTimeout,
        DeclaredTimeout,
        HarnessBackstop,
    )
    from refactory_dsh.node_retry import MISSING_CAUSES, missing_cause
    from refactory_dsh.snippet import SnippetTimeout

    cases = [
        (SnippetTimeout("x"), "step_timeout"),
        (subprocess.TimeoutExpired("x", 1), "step_timeout"),
        (HarnessBackstop("x"), "backstop"),
        (DeclaredTimeout("x"), "declared_timeout"),
        (CompletionTimeout("x"), "timeout"),
        (CompletionFailed("x", kind="worker"), None),
        (RuntimeError("something"), None),
    ]
    for error, expected in cases:
        result = missing_cause(error)
        assert result == expected, f"{type(error).__name__}: got {result!r}, want {expected!r}"
        if result is not None:
            assert result in MISSING_CAUSES, \
                f"missing_cause returned {result!r} which is not in MISSING_CAUSES"


def test_declared_timeout_and_invented_tool_calls_render():
    """The two new entries produce human wording, not the raw cause."""
    from refactory_dsh.optim.base import _item_evidence

    records = [
        {"id": "1", "passed": False, "score": 0.0, "error_text": None,
         "fail_reason": "no answer",
         "missing_steps": {"solve": {"cause": "declared_timeout", "sessions": 1,
                                     "error": "x"}}},
        {"id": "2", "passed": False, "score": 0.0, "error_text": None,
         "fail_reason": "no answer",
         "missing_steps": {"solve": {"cause": "invented_tool_calls", "sessions": 2,
                                     "error": "x"}}},
    ]
    text = "\n".join(_item_evidence(records))
    assert "ran past its own declared timeout_s" in text, text
    assert "wrote out tool calls it has no tools for" in text, text
    assert "declared_timeout" not in text.split("ran past")[0].split("|")[-1], \
        "raw identifier leaked"


# ── 3. crash explanation names the product too ──────────────────────────────


def test_no_answer_explanation_says_worker_or_product():
    """The No-answer header must say 'the worker's or the product's', not just
    'the worker's'."""
    from refactory_dsh.optim.base import _item_evidence

    records = [
        {"id": "1", "passed": False, "score": 0.0, "error_text": None,
         "fail_reason": "no answer",
         "missing_steps": {"solve": {"cause": "crash", "sessions": 3, "error": "x"}}},
    ]
    text = "\n".join(_item_evidence(records))
    assert "the worker's or the product's" in text, text


# ── 4. 'not retried' says 'unless the step opts in' ────────────────────────


def _composition_text():
    from refactory_dsh.nn import edits as nn_edits
    from refactory_dsh.optim import base as base_mod
    from refactory_dsh.optim.memory import state_view

    mode = _mode("livemath/skill_graph")
    view = state_view(mode.state_dict())
    lines = base_mod.composition_section(
        nn_edits.catalogue("broad"),
        view.node_states,
        view.edges,
        protected=mode.edit_protected(),
    )
    return "\n".join(lines)


def test_composition_section_output_limit_says_opts_in():
    """The 'running out of output' line must say 'not retried unless the step
    opts in'."""
    text = _composition_text()
    assert "running out of output" in text.lower(), "section missing"
    idx = text.lower().index("running out of output")
    section = text[idx:idx + 500]
    assert "not retried unless the step opts in" in section, section


def test_composition_section_turn_cap_says_opts_in():
    """The 'running out of turns' line must say 'not retried unless the step
    opts in'."""
    text = _composition_text()
    assert "running out of turns" in text.lower(), "section missing"
    idx = text.lower().index("running out of turns")
    section = text[idx:idx + 500]
    assert "not retried unless the step opts in" in section, section


def test_no_answer_text_says_opts_in():
    """The No-answer header must say 'not retried unless the step opts in'."""
    from refactory_dsh.optim.base import _item_evidence

    records = [
        {"id": "1", "passed": False, "score": 0.0, "error_text": None,
         "fail_reason": "no answer",
         "missing_steps": {"solve": {"cause": "output_limit", "sessions": 1,
                                     "error": "x"}}},
    ]
    text = "\n".join(_item_evidence(records))
    assert "not retried unless the step opts in" in text, text


# ── 5. fallback catalogue line matches falls_back_on ────────────────────────


def test_fallback_catalogue_names_turn_cap_and_backstop():
    """The fallback line must name the turn cap and backstop as triggers and not
    say 'never for time' broadly."""
    from refactory_dsh.nn import edits as nn_edits

    cat = nn_edits.catalogue("broad")
    fb = next(entry for entry in cat if entry["op"] == "fallback")
    desc = fb["description"]
    assert "turn cap" in desc, desc
    assert "backstop" in desc, desc
    assert "never for time" not in desc, f"'never for time' is too broad: {desc}"
