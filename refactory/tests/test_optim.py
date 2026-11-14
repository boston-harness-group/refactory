"""Optimizers: proposal, refusal with reasons, and the validating loop.

Phase 5 gate of ``docs/plans/pythonic-modes.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from refactory_dsh import nn, optim, registry
from refactory_dsh.optim import Candidate, Optimizer, Proposal, ProposalContext


def _model_sleeps_into(waits: list[float]):  # type: ignore[no-untyped-def]
    """A ``time.sleep`` stand-in recording only the model client's own sleeps; any other
    caller (a background thread left by another test) sleeps for real."""
    from refactory_dsh import model

    real_sleep = model.time.sleep

    def sleep(seconds: float) -> None:
        if sys._getframe(1).f_code.co_filename == model.__file__:
            waits.append(seconds)
        else:
            real_sleep(seconds)
    return sleep


MODE = "circle_packing/adaptive"


def _mode() -> nn.Mode:
    _package, mode_class = registry.resolve(Path("."), MODE)
    return mode_class()


def _context(mode: nn.Mode | None = None, *, preset: str = "broad") -> ProposalContext:
    mode = mode or _mode()
    described = mode.describe()
    return ProposalContext(
        epoch=1,
        generation=1,
        mode_name="circle_packing/adaptive",
        best_score=0.875,
        mean_score=0.875,
        knobs=[{"name": name, **spec} for name, spec in described["knobs"].items()],
        nodes=described["nodes"],
        edges=described["edges"],
        leaderboard=[
            {"id": "seed", "parent": None, "score": 0.875, "structural": False, "edits": []}
        ],
        traces=[{"knob_attribution": {"order": {"asc": 1.04, "desc": 0.875}}}],
        catalogue=nn.edits.catalogue(preset),
        insertable=[node.to_state() for node in mode.insertable()],
        preset=preset,
    )


class ScriptedOptimizer(Optimizer):
    """Proposes a fixed list of edits and records the feedback it was given."""

    name = "scripted"

    def __init__(self, edits: list[dict], **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._edits = list(edits)
        self.feedback_seen: list[list[str]] = []

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        self.feedback_seen.append(list(ctx.rejections))
        return [
            Proposal(edit=edit, parent=parents[0].id, rationale="scripted")
            for edit in self._edits[:n]
        ]


# ── proposal ───────────────────────────────────────────────────────────────
def test_the_random_optimizer_proposes_valid_candidates():
    mode = _mode()
    parents = [Candidate.from_state(mode.state_dict())]
    optimizer = optim.RandomOptimizer(seed=1)
    candidates, rejections = optimizer.propose_valid(
        _context(mode), 4, parents=parents, invariants=mode.invariants()
    )
    assert candidates, [r.reasons for r in rejections]
    for candidate in candidates:
        assert candidate.parent == parents[0].id
        assert nn.validate_state(
            candidate.state, invariants=mode.invariants()
        ) == []


def test_the_random_optimizer_respects_the_batch_size():
    mode = _mode()
    parents = [Candidate.from_state(mode.state_dict())]
    optimizer = optim.RandomOptimizer(seed=3)
    candidates, _ = optimizer.propose_valid(
        _context(mode), 2, parents=parents, invariants=mode.invariants()
    )
    assert len(candidates) <= 2


def test_the_random_optimizer_is_reproducible():
    mode = _mode()
    parents = [Candidate.from_state(mode.state_dict())]
    first, _ = optim.RandomOptimizer(seed=11).propose_valid(
        _context(mode), 3, parents=parents, invariants=mode.invariants()
    )
    second, _ = optim.RandomOptimizer(seed=11).propose_valid(
        _context(mode), 3, parents=parents, invariants=mode.invariants()
    )
    assert [c.id for c in first] == [c.id for c in second]


def test_a_candidate_id_is_a_function_of_the_state():
    """Two routes to the same state are the same candidate, and are not run twice."""
    mode = _mode()
    seed = mode.state_dict()
    once = nn.edits.apply({"op": "set_knob", "name": "order", "value": "asc"}, seed)
    twice = nn.edits.apply({"op": "set_knob", "name": "order", "value": "asc"}, seed)
    assert optim.candidate_id(once) == optim.candidate_id(twice)


# ── the loop ───────────────────────────────────────────────────────────────
def test_an_invalid_proposal_becomes_a_rejection_with_reasons():
    mode = _mode()
    parents = [Candidate.from_state(mode.state_dict())]
    optimizer = ScriptedOptimizer([{"op": "remove_node", "id": "exit"}], k=1)
    candidates, rejections = optimizer.propose_valid(
        _context(mode), 1, parents=parents, invariants=mode.invariants()
    )
    assert candidates == []
    assert len(rejections) == 1
    assert any("best_score" in reason for reason in rejections[0].reasons)


def test_a_frozen_knob_is_refused_with_a_reason():
    mode = _mode()
    parents = [Candidate.from_state(mode.state_dict())]
    optimizer = ScriptedOptimizer([{"op": "set_knob", "name": "seed", "value": 9}], k=1)
    _candidates, rejections = optimizer.propose_valid(
        _context(mode), 1, parents=parents, invariants=mode.invariants()
    )
    assert rejections
    assert "frozen" in rejections[0].reasons[0]


def test_an_unknown_operation_is_refused_with_the_catalogue():
    mode = _mode()
    parents = [Candidate.from_state(mode.state_dict())]
    optimizer = ScriptedOptimizer([{"op": "levitate", "target": "eval"}], k=1)
    _candidates, rejections = optimizer.propose_valid(
        _context(mode), 1, parents=parents, invariants=mode.invariants()
    )
    assert "unknown edit" in rejections[0].reasons[0]
    assert "set_knob" in rejections[0].reasons[0]


def test_the_loop_regenerates_up_to_k_times():
    mode = _mode()
    parents = [Candidate.from_state(mode.state_dict())]
    optimizer = ScriptedOptimizer([{"op": "remove_node", "id": "exit"}], k=3)
    candidates, rejections = optimizer.propose_valid(
        _context(mode), 1, parents=parents, invariants=mode.invariants()
    )
    assert candidates == []
    assert [r.attempt for r in rejections] == [1, 2, 3]


def test_the_loop_feeds_refusal_reasons_back():
    """The second attempt is told exactly what was wrong with the first."""
    mode = _mode()
    parents = [Candidate.from_state(mode.state_dict())]
    optimizer = ScriptedOptimizer([{"op": "remove_node", "id": "exit"}], k=3)
    optimizer.propose_valid(
        _context(mode), 1, parents=parents, invariants=mode.invariants()
    )
    assert optimizer.feedback_seen[0] == []
    assert optimizer.feedback_seen[1], "the second attempt got no feedback"
    assert any("best_score" in reason for reason in optimizer.feedback_seen[1]), (
        f"the metric-integrity refusal must reach the retry: {optimizer.feedback_seen[1]}"
    )


def test_the_loop_stops_as_soon_as_enough_candidates_are_valid():
    mode = _mode()
    parents = [Candidate.from_state(mode.state_dict())]
    optimizer = ScriptedOptimizer([{"op": "set_knob", "name": "order", "value": "asc"}], k=3)
    candidates, rejections = optimizer.propose_valid(
        _context(mode), 1, parents=parents, invariants=mode.invariants()
    )
    assert len(candidates) == 1
    assert rejections == []
    assert len(optimizer.feedback_seen) == 1


def test_a_valid_edit_after_a_refused_one_still_produces_a_candidate():
    mode = _mode()
    parents = [Candidate.from_state(mode.state_dict())]
    optimizer = ScriptedOptimizer(
        [{"op": "remove_node", "id": "exit"},
         {"op": "set_knob", "name": "order", "value": "asc"}],
        k=1,
    )
    candidates, rejections = optimizer.propose_valid(
        _context(mode), 2, parents=parents, invariants=mode.invariants()
    )
    assert len(candidates) == 1
    assert len(rejections) == 1


def test_rejection_codes_group_by_rule():
    rejection = optim.Rejection(
        attempt=1,
        edit={"op": "x"},
        reasons=["V5 [gate]: a gate needs at least two conditioned out-edges, has 1",
                 "V7 [rounds]: knob is declared but no node consumes it"],
    )
    assert rejection.codes() == ["V5", "V7"]


def test_an_operator_frozen_knob_is_refused_for_the_run():
    """Freezing is a property of the run: no candidate can edit it away."""
    mode = _mode()
    parents = [Candidate.from_state(mode.state_dict())]
    ctx = _context(mode)
    ctx.frozen = ["order"]
    optimizer = ScriptedOptimizer([{"op": "set_knob", "name": "order", "value": "asc"}], k=1)
    candidates, rejections = optimizer.propose_valid(
        ctx, 1, parents=parents, invariants=mode.invariants()
    )
    assert candidates == []
    assert "E-frozen" in rejections[0].codes()[0]
    assert "frozen for this run" in rejections[0].reasons[0]


def test_a_structural_edit_marks_the_candidate():
    mode = _mode()
    parents = [Candidate.from_state(mode.state_dict())]
    block = {node.id: node for node in mode.insertable()}["observe"].to_state()
    optimizer = ScriptedOptimizer(
        [{"op": "insert_after", "target": "eval", "node": block}], k=1
    )
    candidates, rejections = optimizer.propose_valid(
        _context(mode), 1, parents=parents, invariants=mode.invariants()
    )
    assert not rejections, [r.reasons for r in rejections]
    assert candidates[0].is_structural() is True
    assert parents[0].is_structural() is False


# ── the model optimizer ────────────────────────────────────────────────────
class FakeModel:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, system: str = "") -> str:
        self.prompts.append(prompt)
        return self.answers.pop(0) if self.answers else "{}"


def test_the_autoresearch_optimizer_parses_a_fenced_answer():
    answer = (
        "Here is my decision.\n```json\n"
        '{"diagnosis": "the generator repeats itself", "proposals": ['
        '{"edit": {"op": "set_knob", "name": "order", "value": "asc"},'
        ' "rationale": "asc places more circles", "prediction": "score rises",'
        ' "confidence": 0.6}]}\n```\n'
    )
    model = FakeModel([answer])
    mode = _mode()
    parents = [Candidate.from_state(mode.state_dict())]
    optimizer = optim.AutoresearchOptimizer(model, k=1)
    candidates, rejections = optimizer.propose_valid(
        _context(mode), 1, parents=parents, invariants=mode.invariants()
    )
    assert not rejections, [r.reasons for r in rejections]
    assert candidates[0].state["knobs"]["order"]["value"] == "asc"
    assert optimizer.last_diagnosis == "the generator repeats itself"


def test_the_autoresearch_prompt_carries_the_gradient_and_the_blocks():
    model = FakeModel(["{}"])
    mode = _mode()
    parents = [Candidate.from_state(mode.state_dict())]
    optim.AutoresearchOptimizer(model, k=1).propose_valid(
        _context(mode), 1, parents=parents, invariants=mode.invariants()
    )
    prompt = model.prompts[0]
    assert "Parameters you may move" in prompt
    assert "The operations you may use" in prompt
    assert "Blocks you may insert" in prompt
    assert "restart_on_stall" in prompt
    assert "set_knob" in prompt
    assert "Answer" in prompt


def test_the_autoresearch_prompt_lists_the_refusals_on_a_retry():
    model = FakeModel([
        '{"proposals": [{"edit": {"op": "remove_node", "id": "exit"}}]}',
        '{"proposals": [{"edit": {"op": "set_knob", "name": "order", "value": "asc"}}]}',
    ])
    mode = _mode()
    parents = [Candidate.from_state(mode.state_dict())]
    optimizer = optim.AutoresearchOptimizer(model, k=2)
    candidates, rejections = optimizer.propose_valid(
        _context(mode), 1, parents=parents, invariants=mode.invariants()
    )
    assert candidates
    assert len(rejections) == 1
    assert "Why your last proposals were refused" in model.prompts[1]
    assert "best_score" in model.prompts[1], "the refusal reason never reached the retry"


def test_a_model_that_returns_nothing_is_a_failed_round_not_a_crash():
    mode = _mode()
    parents = [Candidate.from_state(mode.state_dict())]
    # Every attempt of both rounds answers with no JSON (each round's proposal call
    # makes 3 attempts): an empty object would be an answer with no proposals list,
    # which is refused (E-proposal-shape), not a failed round.
    model = FakeModel(["no json here at all"] * 6)
    optimizer = optim.AutoresearchOptimizer(model, k=2)
    candidates, rejections = optimizer.propose_valid(
        _context(mode), 1, parents=parents, invariants=mode.invariants()
    )
    assert candidates == []
    assert rejections == []
    assert len(model.prompts) == 6 and model.answers == []


# ── the proposal call's attempts (SkillOpt-style, as the other optimizer calls) ──

ANSWER = ('{"diagnosis": "d", "proposals": [{"edit": {"op": "set_knob", "name": "order", '
          '"value": "asc"}, "rationale": "r", "prediction": "p", "confidence": 0.5}]}')


class ScriptedModel:
    """Each call takes the next outcome: an exception is raised, text is answered."""

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def complete(self, prompt: str, *, system: str = "") -> str:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return str(outcome)


def _proposer(monkeypatch, tmp_path, outcomes):  # type: ignore[no-untyped-def]
    from refactory_dsh.optim import random_optimizer

    waits: list[float] = []
    monkeypatch.setattr(random_optimizer, "_sleep", waits.append)
    model = ScriptedModel(outcomes)
    optimizer = optim.AutoresearchOptimizer(model, k=1)
    optimizer.trace_path = tmp_path / "trace.jsonl"
    return optimizer, model, waits


def _trace(tmp_path):  # type: ignore[no-untyped-def]
    import json

    return [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]


def test_a_proposal_call_that_fails_or_answers_no_json_is_tried_again(monkeypatch, tmp_path):
    """The proposal call is retried as SkillOpt retries its calls: an exception, then a
    reply with no JSON at all, then an answer. Each attempt is traced (and was paid),
    with waits of 1 and 2 s between them."""
    mode = _mode()
    optimizer, model, waits = _proposer(monkeypatch, tmp_path, [
        RuntimeError("model completion failed (exit 1): dsh: worker failed"),
        "I could not decide.", ANSWER])
    [proposal] = optimizer.propose(_context(mode), 1, [Candidate.from_state(mode.state_dict())])
    assert proposal.edit["value"] == "asc"
    assert model.calls == 3 and waits == [1, 2]
    lines = [line for line in _trace(tmp_path) if line["kind"] == "proposal"]
    assert [line["attempt"] for line in lines] == [1, 2, 3]
    assert "RuntimeError" in lines[0]["error"] and lines[0]["reply"] is None
    assert "no JSON object" in lines[1]["error"] and lines[1]["reply"] == "I could not decide."
    assert lines[2]["error"] is None and lines[2]["proposals"] == 1


def test_a_proposal_call_that_fails_every_attempt_raises_the_last_error(monkeypatch, tmp_path):
    mode = _mode()
    optimizer, model, waits = _proposer(monkeypatch, tmp_path, [
        RuntimeError("one"), RuntimeError("two"), "", ANSWER])
    with pytest.raises(ValueError, match="the model returned nothing"):
        optimizer.propose(_context(mode), 1, [Candidate.from_state(mode.state_dict())])
    assert model.calls == 3 and waits == [1, 2]


def test_a_malformed_proposal_answer_is_refused_and_re_asked_not_retried(monkeypatch, tmp_path):
    """A reply whose JSON object is broken is an answer: one call, refused as unreadable
    (the loop asks again with why), never an attempt retry."""
    from refactory_dsh.optim.base import UNREADABLE_PROPOSAL

    mode = _mode()
    optimizer, model, waits = _proposer(monkeypatch, tmp_path, [
        '{"diagnosis": "d", "proposals": [{"edit": {"op": "set_knob",, }]}', ANSWER])
    [proposal] = optimizer.propose(_context(mode), 1, [Candidate.from_state(mode.state_dict())])
    assert UNREADABLE_PROPOSAL in proposal.edit
    assert model.calls == 1 and waits == []
    [line] = _trace(tmp_path)
    assert line["attempt"] == 1 and line["error"].startswith("malformed answer")


@pytest.mark.parametrize("fatal", ["budget", "cancelled", "context"])
def test_fatal_and_context_failures_of_a_proposal_call_are_not_retried(monkeypatch, tmp_path, fatal):
    from concurrent.futures import CancelledError

    from refactory_dsh.budgets import BudgetExhausted

    error = {"budget": BudgetExhausted("token budget exhausted"),
             "cancelled": CancelledError("model completion cancelled"),
             "context": RuntimeError("prompt is too long: 250000 tokens > 200000 maximum")}[fatal]
    mode = _mode()
    optimizer, model, waits = _proposer(monkeypatch, tmp_path, [error, ANSWER])
    with pytest.raises(type(error)):
        optimizer.propose(_context(mode), 1, [Candidate.from_state(mode.state_dict())])
    assert model.calls == 1 and waits == []


def test_a_person_stop_during_a_proposal_retry_wait_is_a_stop(monkeypatch, tmp_path):
    from refactory_dsh.optim.base import RunStopped

    mode = _mode()
    optimizer, model, waits = _proposer(monkeypatch, tmp_path, [RuntimeError("one"), ANSWER])
    optimizer.stop_check = lambda: True
    with pytest.raises(RunStopped):
        optimizer.propose(_context(mode), 1, [Candidate.from_state(mode.state_dict())])
    assert model.calls == 1


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 2}\n```', {"a": 2}),
        ('prose before {"a": 3} prose after', {"a": 3}),
        ('{"a": {"b": "}"}}', {"a": {"b": "}"}}),
    ],
)
def test_json_extraction(text, expected):
    assert optim.extract_json(text) == expected


def test_an_unknown_optimizer_name_is_refused():
    with pytest.raises(ValueError, match="unknown optimizer"):
        optim.create("nonsense")

    # The two that predate the bucket split, and the five search policies that used
    # to be loops. Asserted by membership rather than by equality so a new optimizer
    # is an addition rather than a test failure -- but the split itself is asserted
    # in `test_strategies.py`, where a port going missing is the thing to catch.
    assert {"random", "autoresearch"} <= set(optim.names())


# ── the model client the optimizer is handed in production ─────────────────
def test_the_model_client_reports_a_timeout_instead_of_hanging(monkeypatch):
    import subprocess

    from refactory_dsh.model import DshModelClient

    def explode(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="dsh", timeout=1)

    monkeypatch.setattr(DshModelClient, "_run_completion", explode)
    with pytest.raises(RuntimeError, match="timed out"):
        DshModelClient(timeout_s=1).complete("hello")


def test_a_node_session_runs_under_the_whole_dispatch_envelope(monkeypatch):
    """A dispatched node's session deadline is the envelope the spine reserved.

    One completion's `timeout_s` cannot bound a tool-using agent session doing a
    node's work: a deadline kill is never retried, so the session gets the whole
    `attempts`-plus-overhead envelope `workers.lower_workflow` reserves — the same
    wall both layers compute, so neither can end the node before the other (#307).
    A plain completion with no node keeps the per-completion deadline.
    """
    import subprocess

    from refactory_dsh.model import DshModelClient, node_session_budget_s

    seen: dict[str, object] = {}

    def explode(self, cmd, run_dir, stop_at=None, budget_s=None, task=None, target_workdir=None):
        seen["stop_at"], seen["budget_s"] = stop_at, budget_s
        raise subprocess.TimeoutExpired(cmd="dsh", timeout=budget_s or 1)

    monkeypatch.setattr(DshModelClient, "_run_completion", explode)
    node = DshModelClient(timeout_s=1, attempts=3).for_node("research_director")
    budget = node_session_budget_s(3, 1)
    with pytest.raises(RuntimeError, match=f"timed out after {budget}s"):
        node.complete("hello")
    # The wall is absolute and shared across attempts, not per-completion.
    assert seen["stop_at"] is not None
    assert seen["budget_s"] == budget

    seen.clear()
    with pytest.raises(RuntimeError, match="timed out after 1s"):
        DshModelClient(timeout_s=1).complete("hello")
    # Each attempt is handed its own absolute deadline: here one completion's limit.
    assert seen["budget_s"] == 1


def test_the_model_client_reports_a_failed_run(monkeypatch):
    import subprocess

    from refactory_dsh.model import DshModelClient

    class Failed:
        returncode = 3
        stdout = ""
        stderr = "no credentials"

    monkeypatch.setattr(DshModelClient, "_run_completion", lambda *a, **k: Failed())
    with pytest.raises(RuntimeError, match="exit 3"):
        DshModelClient().complete("hello")


#: What dsh's headless runner writes when a turn stops at the output token limit:
#: the streamed reasoning under its header, then (with the patch) the diagnostic.
_CAPPED_REASONING = "dsh: reasoning:\nList every check. Qh7+ Kf8, then Qh8+ Ke7 and the king walks...\n"


def test_a_completion_stopped_at_its_output_limit_says_so(monkeypatch):
    """A capped completion read "dsh: reasoning:" -- the header of the streamed
    reasoning -- so neither a person nor an optimizer could tell it from any other
    failure. The cause now names the stop and the limit it hit."""
    from refactory_dsh.model import DshModelClient

    class Capped:
        returncode = 1
        stdout = ""
        stderr = _CAPPED_REASONING + "dsh: max-tokens: the model reached its output token limit before it finished\n"

    monkeypatch.setattr(DshModelClient, "_run_completion", lambda *a, **k: Capped())
    config = {"litellm": {"models": [{"id": "glm", "maxTokens": 32768}]}}
    client = DshModelClient(provider="litellm", model="glm", provider_config=config)
    with pytest.raises(RuntimeError) as raised:
        client.complete("hello")
    assert str(raised.value) == (
        "model completion failed (exit 1): dsh: max-tokens: the model reached its "
        "output token limit before it finished (32768 tokens)"
    )


def test_the_reasoning_header_is_never_the_cause():
    """Without a diagnostic after the reasoning (an unpatched runner), the cause says
    dsh gave none -- not the header, and not the model's last line of reasoning."""
    from refactory_dsh.model import NO_DIAGNOSTIC, failure_cause

    assert failure_cause(_CAPPED_REASONING) == NO_DIAGNOSTIC
    assert failure_cause(_CAPPED_REASONING + "dsh: E_AUTH: key rejected\n") == "dsh: E_AUTH: key rejected"
    # Stderr with no reasoning stream keeps the old fallbacks.
    assert failure_cause("no credentials") == "no credentials"


def test_the_output_limit_is_the_smaller_of_model_and_route():
    from refactory_dsh.model import DshModelClient

    config = {"litellm": {"models": [{"id": "glm", "maxTokens": 32768}, {"id": "other", "maxTokens": 8}]}}
    client = DshModelClient(provider="litellm", model="glm", provider_config=config)
    assert client.output_limit() == 32768
    client._max_completion_tokens = 4096  # noqa: SLF001 - a node route's cap
    assert client.output_limit() == 4096
    assert DshModelClient(provider="litellm", model="unlisted", provider_config=config).output_limit() is None


def test_a_transport_blip_is_ridden_out_over_about_a_minute(monkeypatch):
    """A blip outlasted retries one and two seconds apart and failed the run (dm-218).
    The waits double and add up to about a minute, whatever the attempt count; a clock
    that only records the waits keeps the test from waiting. (A DNS outage, as dm-218's
    was, is now waited out without using attempts: test_network_wait.py.)"""
    from refactory_dsh import model
    from refactory_dsh.model import DshModelClient

    class Blip:
        returncode = 1
        stdout = ""
        stderr = "dsh: TRANSPORT: 500 Internal Server Error"

    class Ok:
        returncode = 0
        stdout = "done\n"
        stderr = ""

    answers: list[object] = [Blip(), Blip(), Blip(), Ok()]
    waits: list[float] = []
    monkeypatch.setattr(DshModelClient, "_run_completion", lambda *a, **k: answers.pop(0))
    monkeypatch.setattr(model.time, "sleep", _model_sleeps_into(waits))
    assert DshModelClient(attempts=4).complete("hello") == "done\n"
    assert answers == [] and len(waits) == 3
    assert waits[1] == 2 * waits[0] and waits[2] == 2 * waits[1]
    assert sum(waits) == pytest.approx(60.0)

    # The default count keeps the same window, and still fails a transport that stays down.
    answers[:] = [Blip(), Blip(), Blip()]
    waits.clear()
    with pytest.raises(RuntimeError, match="on the wire 3 time"):
        DshModelClient().complete("hello")
    assert waits == pytest.approx([20.0, 40.0])

    # The window is a setting of its own, and it travels in the context a worker is built from.
    configured = DshModelClient(retry_window_s=6)
    assert "backoff_s" not in configured.context()
    answers[:] = [Blip(), Blip(), Ok()]
    waits.clear()
    assert DshModelClient(**configured.context()).complete("hello") == "done\n"
    assert waits == pytest.approx([2.0, 4.0])

    # A spec saved before the window existed keeps the schedule it was saved with.
    answers[:] = [Blip(), Blip(), Ok()]
    waits.clear()
    assert DshModelClient(backoff_s=1.0).complete("hello") == "done\n"
    assert waits == pytest.approx([1.0, 2.0])


def test_the_model_client_returns_stdout(monkeypatch):
    import subprocess

    from refactory_dsh.model import DshModelClient

    class Ok:
        returncode = 0
        stdout = '{"proposals": []}\n'
        stderr = ""

    monkeypatch.setattr(DshModelClient, "_run_completion", lambda *a, **k: Ok())
    assert DshModelClient().complete("hello") == '{"proposals": []}\n'


def test_large_prompt_is_sent_on_stdin_not_as_argument(monkeypatch):
    import subprocess

    from refactory_dsh.model import DshModelClient

    seen = {}

    def capture(self, cmd, run_dir, stop_at=None, budget_s=None, task=None, target_workdir=None):
        seen["cmd"] = cmd
        seen["task"] = task
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    monkeypatch.setattr(DshModelClient, "_run_completion", capture)
    large = "x" * (200 * 1024)  # 200 KiB, well over the 128 KiB limit
    result = DshModelClient().complete(large)
    assert result == "ok"
    assert large not in seen["cmd"]
    assert seen["cmd"][-1] == "-"
    assert seen["task"] == large


def test_stdin_prompt_reaches_the_child_process_whole(tmp_path):
    """A prompt over 128 KiB reaches the subprocess on stdin, not the command line."""
    import sys
    import textwrap

    from refactory_dsh.model import DshModelClient

    script = tmp_path / "echo_stdin.py"
    script.write_text(textwrap.dedent("""\
        import sys
        data = sys.stdin.read()
        print(len(data.encode("utf-8")))
    """))

    large = "\u00e9" * (200 * 1024)  # 200 KiB of non-ASCII
    client = DshModelClient(timeout_s=10, attempts=1)
    result = client._run_completion(
        [sys.executable, str(script)], tmp_path, task=large,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == str(len(large.encode("utf-8")))


def test_a_no_op_edit_is_refused_rather_than_re_evaluated():
    """Setting a knob to the value it already holds must not cost a batch slot."""
    mode = _mode()
    seed = mode.state_dict()
    currently = seed["knobs"]["order"]["value"]
    parents = [Candidate.from_state(seed)]
    optimizer = ScriptedOptimizer(
        [{"op": "set_knob", "name": "order", "value": currently}], k=1
    )
    candidates, rejections = optimizer.propose_valid(
        _context(mode), 1, parents=parents, invariants=mode.invariants()
    )
    assert candidates == []
    assert rejections[0].codes() == ["E-noop"]
    assert "identical to its parent" in rejections[0].reasons[0]


def test_a_real_change_is_still_accepted():
    """The no-op guard must not refuse a knob move that actually moves."""
    mode = _mode()
    parents = [Candidate.from_state(mode.state_dict())]
    optimizer = ScriptedOptimizer([{"op": "set_knob", "name": "order", "value": "asc"}], k=1)
    candidates, rejections = optimizer.propose_valid(
        _context(mode), 1, parents=parents, invariants=mode.invariants()
    )
    assert len(candidates) == 1 and not rejections


# ── history and the duplicate refusal (the history issue) ──────────────────
def test_a_proposal_that_recreates_an_existing_state_is_refused():
    """Not just the parent's state: anything the search already holds."""
    mode = _mode()
    seed = mode.state_dict()
    moved = nn.edits.apply({"op": "set_knob", "name": "order", "value": "asc"}, seed)
    parent = Candidate.from_state(moved)
    existing = Candidate.from_state(seed)
    optimizer = ScriptedOptimizer(
        [{"op": "set_knob", "name": "order", "value": seed["knobs"]["order"]["value"]}],
        k=1,
    )
    candidates, rejections = optimizer.propose_valid(
        _context(mode), 1, parents=[parent], invariants=mode.invariants(),
        seen=[parent.id, existing.id],
    )
    assert candidates == []
    assert rejections[0].codes() == ["E-duplicate"]
    assert "already exists" in rejections[0].reasons[0]


def test_a_duplicate_is_still_refused_without_the_seen_set():
    """Defence in depth: the loop always passes seen, but the gate stands alone."""
    mode = _mode()
    seed = mode.state_dict()
    parent = Candidate.from_state(seed)
    currently = seed["knobs"]["order"]["value"]
    optimizer = ScriptedOptimizer(
        [{"op": "set_knob", "name": "order", "value": currently}], k=1
    )
    _candidates, rejections = optimizer.propose_valid(
        _context(mode), 1, parents=[parent], invariants=mode.invariants()
    )
    assert rejections[0].codes() == ["E-noop"], (
        "the parent's own state is a no-op, which is the more specific refusal"
    )


def test_a_new_state_is_not_refused_as_a_duplicate():
    mode = _mode()
    seed = mode.state_dict()
    parent = Candidate.from_state(seed)
    optimizer = ScriptedOptimizer([{"op": "set_knob", "name": "order", "value": "asc"}], k=1)
    candidates, rejections = optimizer.propose_valid(
        _context(mode), 1, parents=[parent], invariants=mode.invariants(),
        seen=[parent.id],
    )
    assert len(candidates) == 1 and not rejections


# ── a proposer that is down is not a search result ─────────────────────────
#
# Filed as "a failed model call looks like a search that found nothing". Probing
# it showed the opposite defect and a worse one: `propose_valid` -- the path with
# the retry loop and the exception guard -- is called only by tests, and the loop
# goes through `Loop.propose` straight to `optimizer.propose`. So a model failure
# did not hide; it crashed the run, taking the report and every evaluation with
# it. These assert the behaviour a person actually needs: the run survives, and it
# says the proposer never answered.


class _DeadClient:
    """A model that cannot be reached, which is the first failure anyone hits."""

    def __init__(self, message: str = "401 unauthorized") -> None:
        self.message = message

    def complete(self, prompt: str, *, system: str = "") -> str:
        raise RuntimeError(self.message)


def test_a_run_with_an_unreachable_proposer_reports_failure(tmp_path):
    from refactory_dsh.loop import Loop, LoopConfig

    loop = Loop(
        "circle_packing/adaptive",
        LoopConfig(epochs=1, batch_size=2, seed=5, optimizer="autoresearch",
                   runs_root=tmp_path / "run"),
        project=Path("."),
        model=_DeadClient(),
    )
    report = loop.run()  # must not raise

    proposer = report["proposer"]
    assert proposer["failure_count"] >= 1, "the failure was not recorded"
    assert proposer["all_failed"] is True
    assert proposer["optimizer"] == "autoresearch"
    # G32: all-failed runs now exit non-zero
    assert report["exit_ok"] is False


def test_the_report_says_the_proposer_never_answered(tmp_path):
    """The distinction that matters: the machine failed, or the mode did not move."""
    from refactory_dsh.loop import Loop, LoopConfig

    root = tmp_path / "run"
    Loop(
        "circle_packing/adaptive",
        LoopConfig(epochs=1, batch_size=2, seed=5, optimizer="autoresearch",
                   runs_root=root),
        project=Path("."),
        model=_DeadClient(),
    ).run()

    text = (root / "report.md").read_text()
    assert "Did the proposer answer?" in text, (
        "the report does not mention the proposer, so a run where the model was "
        "unreachable reads as a run that found nothing"
    )
    assert "not a search result" in text
    written = (root / "report.json").read_text()
    assert "401 unauthorized" in written, (
        "the reason the proposer failed is not in the report, so a reader cannot "
        "tell a dead key from a wrong model name"
    )


def test_a_run_whose_proposer_answered_does_not_claim_a_failure(tmp_path):
    """The other half: the finding must be absent when nothing failed."""
    from refactory_dsh.loop import Loop, LoopConfig

    root = tmp_path / "run"
    report = Loop(
        "circle_packing/adaptive",
        LoopConfig(epochs=1, batch_size=2, seed=5, optimizer="random", runs_root=root),
        project=Path("."),
    ).run()

    assert report["proposer"]["failure_count"] == 0
    assert report["proposer"]["all_failed"] is False
    assert report["proposer"]["finding"] == ""
    assert "Did the proposer answer?" not in (root / "report.md").read_text(), (
        "a healthy run is claiming its proposer failed"
    )


TEARDOWN = ("dsh: worker failed: subagent-claude-code: Product subagent failure (product: "
            "Claude Code; stage: teardown; category: unknown; exit code: 0)")


def _answers(monkeypatch, *answers):  # type: ignore[no-untyped-def]
    from refactory_dsh import model
    from refactory_dsh.model import DshModelClient

    queue = list(answers)
    waits: list[float] = []
    monkeypatch.setattr(DshModelClient, "_run_completion", lambda *a, **k: queue.pop(0))
    monkeypatch.setattr(model.time, "sleep", _model_sleeps_into(waits))
    return queue, waits


def _no_tools():  # type: ignore[no-untyped-def]
    from refactory_dsh.model import DshModelClient

    # Only the dsh backend retries at all (other backends make one attempt).
    return DshModelClient(worker_backend="dsh", provider="p", model="m").with_tool_policy("none")


class _Failed:
    returncode = 1

    def __init__(self, stderr: str, stdout: str = "") -> None:
        self.stderr = stderr
        self.stdout = stdout


class _Done:
    returncode = 0
    stdout = "done\n"

    def __init__(self, stderr: str = "") -> None:
        self.stderr = stderr


def _events(run: Path, kind: str) -> list[dict]:
    import json

    path = run / ".refactory/events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [row for row in rows if row.get("kind") == kind]


def test_an_answer_whose_worker_cleanup_failed_is_kept_with_a_warning(monkeypatch, tmp_path):
    """A worker that answered and whose cleanup then failed: the answer is the
    completion's, it is not asked again, and a warning is recorded. Both shapes: the
    bridge's exit 0 with its warning line, and a runtime's exit 1 with the clean-exit
    teardown failure and the answer on stdout. With tools and without."""
    from refactory_dsh.model import TEARDOWN_EVENT, TEARDOWN_WARNING, DshModelClient

    clients = (_no_tools(), DshModelClient(worker_backend="dsh", provider="p", model="m"))
    shapes = (_Done(stderr=f"{TEARDOWN_WARNING} the worker scope did not empty\n"),
              _Failed(f"dsh: reasoning:\nthinking\n{TEARDOWN}\n", stdout="done\n"))
    for number, (client, proc) in enumerate((c, p) for c in clients for p in shapes):
        run = tmp_path / f"run{number}"
        queue, waits = _answers(monkeypatch, proc, _Done())
        assert client.complete("hello", workdir=run) == "done\n"
        assert len(queue) == 1 and waits == []
        [event] = _events(run, TEARDOWN_EVENT)
        assert event["state"] == "warning" and "the answer is kept" in event["summary"]
        assert "teardown" in event["error"] and event["cleanup"] == "unconfirmed"
        tools = client._tool_policy != "none"
        assert event["tools"] is tools
        assert ("its tool processes may still be running" in event["summary"]) is tools


@pytest.mark.parametrize("stdout, stderr", [
    ("", TEARDOWN),
    # The worker stopped without completing before its cleanup failed: no answer.
    ("partial\n", f"dsh: worker cancelled: stopped\n{TEARDOWN}"),
    # A turn whose error only quotes the teardown signature is not the worker's own
    # failure line: its partial text is not an answer.
    ("partial\n", "dsh: worker error: the tool printed (product: X; stage: teardown; "
                   "category: unknown; exit code: 0)"),
])
def test_a_teardown_failure_without_an_answer_fails_at_once(monkeypatch, tmp_path, stdout, stderr):
    from refactory_dsh.model import TEARDOWN_EVENT

    queue, waits = _answers(monkeypatch, _Failed(stderr, stdout=stdout), _Done())
    with pytest.raises(RuntimeError, match=r"model completion failed \(exit 1\): dsh: worker") as raised:
        _no_tools().complete("hello", workdir=tmp_path / "run")
    assert raised.value.kind == "worker"
    assert len(queue) == 1 and waits == []
    assert _events(tmp_path / "run", TEARDOWN_EVENT) == []


def test_each_retry_on_the_wire_is_recorded(monkeypatch, tmp_path):
    """The client still re-asks after a wire failure; each retry is a recorded event
    with its cause and attempt."""
    from refactory_dsh.model import RETRY_EVENT

    run = tmp_path / "run"
    queue, waits = _answers(monkeypatch, _Failed("dsh: TRANSPORT: 500 Internal Server Error"),
                            _Failed("dsh: TRANSPORT: 429 Too Many Requests"), _Done())
    assert _no_tools().complete("hello", workdir=run) == "done\n"
    assert queue == [] and len(waits) == 2
    events = _events(run, RETRY_EVENT)
    assert [(e["attempt"], e["attempts"], e["cause"]) for e in events] == [
        (1, 3, "dsh: TRANSPORT: 500 Internal Server Error"), (2, 3, "dsh: TRANSPORT: 429 Too Many Requests")]
    assert all(e["state"] == "retrying" for e in events)


@pytest.mark.parametrize("stderr", [
    TEARDOWN.replace("exit code: 0", "exit code: 1"),
    TEARDOWN.replace("exit code: 0", "exit code: 01"),
    TEARDOWN.replace("stage: teardown", "stage: run"),
    # "exit code: 0" in the streamed reasoning, the real diagnostic exit 1.
    "dsh: reasoning:\nthe last command ended with exit code: 0\n"
    + TEARDOWN.replace("exit code: 0", "exit code: 1"),
    # A combined startup and teardown failure: the clean exit is the startup group's.
    ("dsh: worker failed: subagent-claude-code: Product subagent failure (product: Claude "
     "Code; stage: startup; category: unknown; exit code: 0); Product subagent failure "
     "(product: Claude Code; stage: teardown; category: unknown)"),
])
def test_other_teardown_or_stage_failures_are_not_retried(monkeypatch, stderr):
    """A teardown failure after a nonzero CLI exit, a clean exit failing in another
    stage, a clean exit mentioned only in the reasoning, or a teardown group without
    an exit code: fails at once with its cause, even without tools."""
    queue, waits = _answers(monkeypatch, _Failed(stderr), _Done())
    with pytest.raises(RuntimeError, match=r"model completion failed \(exit 1\): dsh: worker failed"):
        _no_tools().complete("hello")
    assert len(queue) == 1 and waits == []


def test_a_transport_failure_is_still_retried_and_the_predicates_stay_apart(monkeypatch):
    from refactory_dsh.model import (DshModelClient, failure_cause, is_clean_exit_teardown_failure,
                                     is_retryable_failure, is_transport_failure)

    for client in (DshModelClient(), _no_tools()):
        queue, waits = _answers(monkeypatch, _Failed("dsh: TRANSPORT: 500 Internal Server Error"), _Done())
        assert client.complete("hello") == "done\n"
        assert queue == [] and len(waits) == 1
    assert not is_transport_failure(TEARDOWN), "a teardown failure is not called transport"
    assert is_clean_exit_teardown_failure(TEARDOWN.upper())
    assert not is_retryable_failure(TEARDOWN), "a teardown failure is never re-asked"
    assert failure_cause(f"noise\n{TEARDOWN}\n") == TEARDOWN


def test_a_native_session_without_tools_retries_and_one_with_tools_does_not(monkeypatch):
    """On a native backend (claude_code) a session without tools retries a transport
    failure within the configured attempts; a session with tools makes one attempt, as
    before. The configured count survives a context round trip."""
    from refactory_dsh.model import DshModelClient

    native = DshModelClient(worker_backend="claude_code")
    text = native.with_tool_policy("none")
    assert native.retry_attempts() == 1 and text.retry_attempts() == 3
    rebuilt = DshModelClient(**native.context()).with_tool_policy("none")
    assert rebuilt.retry_attempts() == 3
    for failure in ("dsh: TRANSPORT: 500 Internal Server Error",):
        queue, waits = _answers(monkeypatch, _Failed(failure), _Done())
        assert text.complete("hello") == "done\n"
        assert queue == [] and len(waits) == 1
        queue, waits = _answers(monkeypatch, _Failed(failure), _Done())
        with pytest.raises(RuntimeError, match="model completion failed"):
            native.complete("hello")
        assert len(queue) == 1 and waits == []


def test_no_tools_agent_nodes_and_the_optimizer_get_a_no_tools_client(tmp_path, monkeypatch):
    """The client a tool_policy "none" agent node calls is a no-tools client on both
    runtimes, and so is the optimizer's: the walker (`GraphExecutor`, via for_node and
    with_tool_policy), the spine (`workers.lower_workflow` writes the policy and the
    node's attempts into the worker spec that `worker_node` rebuilds its client from),
    and `text_only`. On a native backend each retries; the spine's envelope reserves
    every attempt."""
    import json

    from refactory_dsh import workers
    from refactory_dsh.ir import AgentNode, AgentRole, Workflow
    from refactory_dsh.model import DEFAULT_TIMEOUT_S, DshModelClient, node_session_budget_s
    from refactory_dsh.optim.random_optimizer import text_only

    native = DshModelClient(worker_backend="claude_code")
    assert text_only(native)._tool_policy == "none" and text_only(native).retry_attempts() == 3

    def graph(policy):  # type: ignore[no-untyped-def]
        node = AgentNode(id="answer", role=AgentRole.BUILDER, prompt_template="q",
                         writes={"answer.txt"}, tool_policy=policy)
        return Workflow(name="g", nodes={"answer": node}, edges=[], start_node="answer")

    for policy, attempts in (("none", 3), (None, 1)):
        lowered = workers.lower_workflow(graph(policy), tmp_path / str(policy), native.context())
        node = lowered.nodes["answer"]
        spec = json.loads(Path(node.dispatch_command.rsplit(" ", 1)[1].strip("'")).read_text())
        rebuilt = DshModelClient(**spec["model_context"]).for_node("answer", role="builder")
        rebuilt = rebuilt.with_tool_policy(spec.get("tool_policy") or "")  # as worker_node does
        assert (rebuilt._tool_policy == "none") is (policy == "none")
        assert rebuilt.retry_attempts() == attempts
        assert node.command_timeout_s == node_session_budget_s(attempts, DEFAULT_TIMEOUT_S)

    # The walker: the executor hands an agent node the for_node client with the node's
    # tool policy applied.
    from refactory_dsh.nn import executor as nn_executor

    seen = {}

    def ask(model, prompt, *, system, workdir):  # type: ignore[no-untyped-def]
        seen["policy"], seen["attempts"] = model._tool_policy, model.retry_attempts()
        return "A"

    monkeypatch.setattr(nn_executor, "_ask_model", ask)
    from refactory_dsh.nn.graph import Graph, Node

    node = Node(id="answer", kind="agent", prompt_template="q", tool_policy="none",
                writes=("answer.txt",))
    walk = Graph()
    walk.add(node)
    walk.start = "answer"
    executor = nn_executor.Executor(walk, nn_executor.RunContext(tmp_path / "walk", model=native))
    import os

    saved = dict(os.environ)  # the visit sets run variables; leave none behind
    try:
        executor._visit_agent(node)
    except Exception:  # noqa: BLE001 - only the client the node asked with matters here
        pass
    finally:
        os.environ.clear()
        os.environ.update(saved)
    assert seen == {"policy": "none", "attempts": 3}


@pytest.mark.parametrize("run_route, node_route, attempts", [
    # A dsh run with one node routed to claude_code (tools): that node's native session
    # makes one attempt and gets one completion's time limit (it used to inherit dsh's 3).
    ({"worker_backend": "dsh", "provider": "p", "model": "m"},
     {"worker_backend": "claude_code"}, 1),
    # A claude_code run with one node routed to dsh (tools): that node retries as a dsh
    # run's nodes do, with the time limit for every attempt (it used to be held to 1).
    ({"worker_backend": "claude_code"},
     {"worker_backend": "dsh", "provider": "p", "model": "m"}, 3),
])
def test_a_mixed_route_node_gets_its_own_routes_attempts_and_time_limit(
        tmp_path, run_route, node_route, attempts):
    """The attempt count and the spine's time limit follow the node's own route, not
    the run's default: on the spine (the lower_workflow spec, rebuilt as worker_node
    does) and on the walker (the for_node client)."""
    import json

    from refactory_dsh import workers
    from refactory_dsh.ir import AgentNode, AgentRole, Workflow
    from refactory_dsh.model import DEFAULT_TIMEOUT_S, DshModelClient, node_session_budget_s

    run = DshModelClient(**run_route, node_routes={"answer": node_route})
    node = AgentNode(id="answer", role=AgentRole.BUILDER, prompt_template="q",
                     writes={"answer.txt"})
    graph = Workflow(name="g", nodes={"answer": node}, edges=[], start_node="answer")
    lowered = workers.lower_workflow(graph, tmp_path, run.context()).nodes["answer"]
    assert lowered.command_timeout_s == node_session_budget_s(attempts, DEFAULT_TIMEOUT_S)
    spec = json.loads(Path(lowered.dispatch_command.rsplit(" ", 1)[1].strip("'")).read_text())
    rebuilt = DshModelClient(**spec["model_context"]).for_node("answer", role="builder")
    rebuilt = rebuilt.with_tool_policy(spec.get("tool_policy") or "")
    assert rebuilt.route()["worker_backend"] == node_route["worker_backend"]
    assert rebuilt.retry_attempts() == attempts
    walker = run.for_node("answer", role="builder").with_tool_policy("")
    assert walker.route()["worker_backend"] == node_route["worker_backend"]
    assert walker.retry_attempts() == attempts


# ── the answer's optional `considered` ────────────────────────────────────


@pytest.mark.parametrize("considered, recorded", [
    ('"a prompt change or a code step; the prompt, since the notes are fine"',
     "a prompt change or a code step; the prompt, since the notes are fine"),
    ('["a new agent step", "the setting: cheaper"]', "a new agent step\nthe setting: cheaper"),
    ('{"code step": "no file to read"}', '{"code step": "no file to read"}'),
    (None, ""),
    ('""', ""),
])
def test_an_answer_may_say_what_it_considered_and_it_is_traced(tmp_path, considered, recorded):
    """`considered` is optional and any shape is read: a string as it is, a list one entry
    per line, an object as JSON. It is kept beside the diagnosis and in the call's trace;
    an answer without it proposes exactly as before."""
    import json

    field = f'"considered": {considered}, ' if considered is not None else ""
    answer = ('{"diagnosis": "the order is wrong", ' + field + '"proposals": ['
              '{"edit": {"op": "set_knob", "name": "order", "value": "asc"}}]}')
    mode = _mode()
    optimizer = optim.AutoresearchOptimizer(FakeModel([answer]), k=1)
    optimizer.trace_path = tmp_path / "trace.jsonl"
    candidates, rejections = optimizer.propose_valid(
        _context(mode), 1, parents=[Candidate.from_state(mode.state_dict())],
        invariants=mode.invariants())
    assert candidates and not rejections, [r.reasons for r in rejections]
    assert optimizer.last_considered == recorded
    [call] = [json.loads(line) for line in optimizer.trace_path.read_text().splitlines()]
    assert call["considered"] == recorded and call["diagnosis"] == "the order is wrong"


def test_considered_is_bounded():
    from refactory_dsh.optim.base import CONSIDERED_CHARS, considered_text

    long = considered_text("x" * (CONSIDERED_CHARS * 2))
    assert len(long) == CONSIDERED_CHARS and long.endswith("…")


def test_the_answer_format_offers_considered_as_optional_and_names_only_permitted_kinds():
    from refactory_dsh.nn import edits as nn_edits
    from refactory_dsh.optim.base import answer_instructions

    full = answer_instructions(nn_edits.catalogue(), code_nodes=True)
    line = next(l for l in full.splitlines() if '"considered"' in l)
    assert "optional" in line and "why you chose yours" in line
    for kind in ("a prompt or setting", "a new agent step", "a code step", "a gate or loop",
                 "parallel steps"):
        assert kind in line, kind
    no_code = next(l for l in answer_instructions(nn_edits.catalogue()).splitlines()
                   if '"considered"' in l)
    assert "a code step" not in no_code and "a new agent step" in no_code
    knobs_only = next(l for l in answer_instructions(
        [op for op in nn_edits.catalogue() if op["op"] == "set_knob"], code_nodes=True
    ).splitlines() if '"considered"' in l)
    assert "(a prompt or setting)" in knobs_only
