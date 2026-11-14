"""A model call waits out a network, DNS, credential-refresh or proxy outage.

An 85-minute DNS outage made 40 optimizer calls fail with Claude Code's ``API Error:
getaddrinfo ENOTFOUND oauth2.googleapis.com``; SkillOpt's three attempts (1 + 2 + 4 s)
ran out and an epoch's analysis was lost. The model client now waits such a failure out
(`model.is_infrastructure_failure`, `DshModelClient._wait_for_network`) without using an
attempt, so the caller's retries see only real model failures.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import CancelledError
from pathlib import Path

import pytest

from refactory_dsh import model
from refactory_dsh.items import interrupted_cause
from refactory_dsh.model import (NETWORK_WAIT_EVENT, RETRY_EVENT, CompletionFailed, DshModelClient,
                                 failure_cause, is_infrastructure_failure, observe_network_waits)

#: The lab's outage, as the Claude Code worker reports it: the Vertex token refresh
#: could not resolve its host.
OAUTH_DNS = ("dsh: worker failed: subagent-claude-code: Product subagent failure (product: "
             "Claude Code; stage: query-run; category: invalid-result): the result was an error "
             "after 0 output tokens: API Error: getaddrinfo ENOTFOUND oauth2.googleapis.com")

#: A real model failure on the same route: the provider's rate limit.
RATE_LIMITED = ("dsh: worker failed: subagent-claude-code: Product subagent failure (product: "
                "Claude Code; stage: query-run; category: invalid-result): the result was an error "
                'after 0 output tokens: API Error: 429 {"type":"error","error":{"type":'
                '"rate_limit_error","message":"rate limited"}}')


@pytest.mark.parametrize("cause", [
    OAUTH_DNS,
    "dsh: TRANSPORT: getaddrinfo ENOTFOUND api.anthropic.com",
    "dsh: TRANSPORT: getaddrinfo EAI_AGAIN api.anthropic.com",
    "dsh: TRANSPORT: request to https://api.anthropic.com/v1/messages failed, reason: ENOTFOUND",
    "dsh: TRANSPORT: connect ECONNREFUSED 127.0.0.1:8082",
    "dsh: TRANSPORT: read ECONNRESET",
    "dsh: TRANSPORT: connect ETIMEDOUT 10.0.0.1:443",
    "dsh: TRANSPORT: socket hang up",
    "dsh: TRANSPORT: fetch failed",
    "dsh: TRANSPORT: Connection error.",
    "curl: (6) Could not resolve host: api.anthropic.com",
    "dsh: TRANSPORT: 502 Bad Gateway",
    "dsh: TRANSPORT: 503 Service Unavailable",
    "dsh: TRANSPORT: 504 Gateway Timeout",
    "the result was an error: API Error: 502 upstream unreachable",
    "the result was an error: API Error: Could not refresh access token: request failed",
    "the result was an error: API Error: request to https://oauth2.googleapis.com/token timed out",
    "the result was an error: API Error: Failed to refresh the access token",
    "the result was an error: token refresh failed: no response",
    "dsh: TRANSPORT: connect ENETUNREACH 2600::1:443",
])
def test_network_dns_auth_refresh_and_proxy_failures_are_infrastructure(cause):
    assert is_infrastructure_failure(cause)
    assert is_infrastructure_failure(cause.upper()), "the match ignores case"


@pytest.mark.parametrize("cause", [
    RATE_LIMITED,
    "dsh: TRANSPORT: 429 Too Many Requests",
    "the result was an error: API Error: 429 after a socket hang up",
    "dsh: TRANSPORT: rate limit exceeded",
    'the result was an error: API Error: 529 {"type":"overloaded_error"}',
    "dsh: TRANSPORT: 503 Service Unavailable: overloaded",
    "dsh: TRANSPORT: 500 Internal Server Error",
    # A status the error line reports wins over the words after it (review of #649).
    "the result was an error: API Error: 500 Internal Server Error: Connection error.",
    "the result was an error: API Error: 500 {\"error\": \"socket hang up upstream\"}",
    ("the result was an error: API Error: 403 Could not refresh access token: Permission "
     "'iam.serviceAccounts.getAccessToken' denied"),
    "dsh: TRANSPORT: 401 Unauthorized from oauth2.googleapis.com",
    "the result was an error: Could not refresh access token: permission denied",
    "dsh: TRANSPORT: HTTP/1.1 400 Bad Request (fetch failed to parse)",
    "the result was an error: oauth2.googleapis.com answered 403 to the token refresh",
    "the result was an error: Internal Server Error (the upstream socket hang up)",
    "the result was an error: Unauthorized: could not refresh access token",
    # A gateway code counts only as the error line's own status, not in free text.
    "the result was an error: the test server answered status 503 Service Unavailable",
    "the result was an error: API Error: 400 prompt is too long",
    "the result was an error: invalid_grant: token refresh failed, the credential was revoked",
    "dsh: TRANSPORT: 401 invalid x-api-key",
    "dsh: max-tokens: the model reached its output token limit before it finished",
    "Product subagent failure (stage: query-run; category: output-limit): connection error",
    # Each veto below also names the wire, so only the veto keeps it from being an outage.
    "the result was an error: the model declined (stop_reason: refusal) after a socket hang up",
    "the result was an error: invalid api key (fetch failed earlier)",
    "model completion timed out after 1800s; requested runtime shutdown",
    "dsh: worker failed: the session ended without an answer",
    "",
])
def test_model_account_and_request_failures_are_not(cause):
    assert not is_infrastructure_failure(cause)


# ── the client ──────────────────────────────────────────────────────────────


class _Failed:
    returncode = 1
    stdout = ""

    def __init__(self, stderr: str) -> None:
        self.stderr = stderr


class _Done:
    returncode = 0
    stdout = "done\n"
    stderr = ""


@pytest.fixture
def answers(monkeypatch):  # type: ignore[no-untyped-def]
    """Queue the sessions' outcomes; each wait is one recorded ``sleep`` (no real waiting)."""
    queue: list[object] = []
    waits: list[float] = []
    monkeypatch.setattr(DshModelClient, "_run_completion", lambda *a, **k: queue.pop(0))
    # ``model.time`` is the process-wide ``time`` module: record only the sleeps the model
    # client itself makes, so a background thread left by another test (polling with its
    # own short sleeps) neither pollutes the recorded waits nor stops sleeping for real.
    real_sleep = model.time.sleep

    def sleep(seconds: float) -> None:
        if sys._getframe(1).f_code.co_filename == model.__file__:
            waits.append(seconds)
        else:
            real_sleep(seconds)

    monkeypatch.setattr(model.time, "sleep", sleep)
    monkeypatch.setattr(model, "NETWORK_POLL_S", 10_000.0)
    monkeypatch.setattr(model, "_jitter", lambda: 1.0)
    return queue, waits


def _text_client(**kwargs) -> DshModelClient:  # type: ignore[no-untyped-def]
    """The LiveMath arms' optimizer and solver route: Claude Code, no tools."""
    return DshModelClient(worker_backend="claude_code", **kwargs).with_tool_policy("none")


def _events(run: Path, kind: str) -> list[dict]:
    path = run / ".refactory/events.jsonl"
    if not path.exists():
        return []
    return [event for line in path.read_text().splitlines()
            if (event := json.loads(line)).get("kind") == kind]


def test_the_lab_outage_is_waited_out_and_the_call_answers(answers, tmp_path, capsys):
    """The exact ENOTFOUND oauth2 failure, five times over: the client waits 5, 10, 20,
    40, 60 s and asks again each time, without using any of its 3 attempts, and answers
    when the network is back. The outage is traced once, from start to end, and the run
    log says it is waiting."""
    queue, waits = answers
    queue.extend([_Failed(OAUTH_DNS)] * 5 + [_Done()])
    run = tmp_path / "run"
    assert _text_client().complete("hello", workdir=run) == "done\n"
    assert queue == [] and waits == [5.0, 10.0, 20.0, 40.0, 60.0]
    assert _events(run, RETRY_EVENT) == [], "an outage uses no attempt"
    waiting, ended = _events(run, NETWORK_WAIT_EVENT)
    assert waiting["state"] == "waiting" and "ENOTFOUND oauth2.googleapis.com" in waiting["error"]
    assert ended["state"] == "recovered" and ended["failures"] == 5
    assert ended["waited_s"] == pytest.approx(135.0, abs=1.0)
    assert ended["started_at"] == waiting["started_at"] and ended["ended_at"] >= ended["started_at"]
    log = capsys.readouterr().err
    assert "waiting for network: " + failure_cause(OAUTH_DNS) in log
    # Once a minute: at the start, and again once a minute of waiting has passed.
    assert log.count("waiting for network") == 2
    assert "recovered after waiting" in log


def test_waits_do_not_count_as_attempts(answers):
    """Outages before, between and after real failures: the call still makes its three
    real attempts, and fails as a transport failure only when those run out."""
    queue, waits = answers
    wire = "dsh: TRANSPORT: 500 Internal Server Error"
    outage = "dsh: TRANSPORT: getaddrinfo EAI_AGAIN api.anthropic.com"
    queue.extend([_Failed(outage), _Failed(wire), _Failed(outage), _Failed(outage),
                  _Failed(wire), _Failed(outage), _Failed(wire)])
    client = DshModelClient(worker_backend="dsh", provider="p", model="m", attempts=3)
    with observe_network_waits() as outages:
        with pytest.raises(CompletionFailed, match=r"failed on the wire 3 time\(s\); last cause: "
                                                    r"dsh: TRANSPORT: 500") as raised:
            client.complete("hello")
    assert raised.value.kind == "transport" and queue == []
    # Each outage ends when a real failure shows the network is back.
    assert [(e["failures"], e["outcome"]) for e in outages] == [
        (1, "recovered"), (2, "recovered"), (1, "recovered")]


def test_other_failures_are_handled_as_before(answers):
    """A rate limit on the same route fails at once, as it did: no wait, no retry."""
    queue, waits = answers
    queue.extend([_Failed(RATE_LIMITED), _Done()])
    with pytest.raises(CompletionFailed, match=r"failed \(exit 1\): dsh: worker failed") as raised:
        _text_client().complete("hello")
    assert raised.value.kind == "transport" and len(queue) == 1 and waits == []


def test_a_session_with_tools_on_a_native_backend_is_still_not_replayed(answers):
    """Claude Code with tools makes one attempt (`retry_attempts`): asking again could
    repeat real work, so an outage there fails at once, as before."""
    queue, waits = answers
    for outage in (OAUTH_DNS, "dsh: TRANSPORT: socket hang up"):
        queue[:] = [_Failed(outage), _Done()]
        with pytest.raises(CompletionFailed, match="ENOTFOUND|socket hang up"):
            DshModelClient(worker_backend="claude_code").complete("hello")
        assert len(queue) == 1 and waits == []


def test_on_the_dsh_backend_a_product_detail_never_starts_a_wait_for_a_session_with_tools(answers):
    """The product's detail decides only for a session without tools; with tools only
    the wire's own errors do (`without_product_detail`)."""
    queue, waits = answers
    queue.extend([_Failed(OAUTH_DNS), _Done()])
    with pytest.raises(CompletionFailed):
        DshModelClient(worker_backend="dsh", provider="p", model="m").complete("hello")
    assert len(queue) == 1 and waits == []


def test_the_bound_raises_the_original_failure(answers, monkeypatch, tmp_path):
    """The network never comes back: once the call has waited its bound it raises the
    failure, said as a wire failure (so an evaluation reads the item as interrupted, not
    answered) and naming the last cause."""
    queue, waits = answers
    monkeypatch.setattr(model, "NETWORK_WAIT_S", 100.0)
    queue.extend([_Failed(OAUTH_DNS)] * 10)
    run = tmp_path / "run"
    with pytest.raises(CompletionFailed) as raised:
        _text_client().complete("hello", workdir=run)
    assert waits == [5.0, 10.0, 20.0, 40.0], "the next wait (60 s) would pass the bound"
    assert len(queue) == 5
    message = str(raised.value)
    assert raised.value.kind == "transport"
    assert message.startswith("model completion failed on the wire 5 time(s)")
    assert message.endswith("last cause: " + failure_cause(OAUTH_DNS))
    assert interrupted_cause(message) == "transport"
    assert [e["state"] for e in _events(run, NETWORK_WAIT_EVENT)] == ["waiting", "gave_up"]


def test_the_failed_tries_count_toward_the_bound(answers, monkeypatch):
    """The bound is the call's time on the outage, its failed tries included: each try
    here takes 50 s before failing, so a 100 s bound allows one wait."""
    queue, waits = answers
    clock = [1000.0]

    def run(*args, **kwargs):  # type: ignore[no-untyped-def]
        clock[0] += 50
        return queue.pop(0)

    monkeypatch.setattr(DshModelClient, "_run_completion", run)
    monkeypatch.setattr(model.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(model, "NETWORK_WAIT_S", 100.0)
    queue.extend([_Failed(OAUTH_DNS)] * 5)
    with pytest.raises(CompletionFailed, match="it waited 105s"):
        _text_client().complete("hello")
    assert waits == [5.0]


def test_the_default_bound_is_two_hours_of_one_minute_waits(answers):
    queue, waits = answers
    queue.extend([_Failed("dsh: TRANSPORT: fetch failed")] * 200)
    with pytest.raises(CompletionFailed, match="the most one call waits"):
        _text_client().complete("hello")
    assert 7200 - 60 < sum(waits) <= 7200 and max(waits) == 60.0


def test_a_node_wall_bounds_the_wait(answers, monkeypatch):
    """A node's own limit on its session stops the wait a minute short of it. (A clock
    the waits move, so the wall is measured in the waits' time.)"""
    queue, waits = answers
    clock = [1000.0]

    real_sleep, real_monotonic = model.time.sleep, model.time.monotonic

    def from_model() -> bool:
        # ``model.time`` is the process-wide ``time`` module: only the model client's own
        # calls see the test clock; any other thread keeps the real one.
        return sys._getframe(2).f_code.co_filename == model.__file__

    def sleep(seconds: float) -> None:
        if not from_model():
            real_sleep(seconds)
            return
        waits.append(seconds)
        clock[0] += seconds

    def monotonic() -> float:
        return clock[0] if from_model() else real_monotonic()

    monkeypatch.setattr(model.time, "monotonic", monotonic)
    monkeypatch.setattr(model.time, "sleep", sleep)
    queue.extend([_Failed(OAUTH_DNS)] * 10)
    with pytest.raises(CompletionFailed, match="until the node's time ran out"):
        _text_client().with_wall(200).complete("hello")
    # 135 s of waits leave 65 s of the 200; one more minute would leave 5.
    assert waits == [5.0, 10.0, 20.0, 40.0, 60.0]


def test_a_person_stop_ends_the_wait_as_a_stop(answers, monkeypatch, tmp_path):
    """A person asks the run to stop while a call waits for the network: the wait ends at
    the next poll with `RunStopped`, as an optimizer call's retry wait ends (#647), so the
    search ends as a stop; an item it ends reads as stopped, not answered."""
    from refactory_dsh.optim.base import RunStopped

    queue, waits = answers
    stop = [False]

    def stopping(*args, **kwargs):  # type: ignore[no-untyped-def]
        stop[0] = len(queue) <= 2
        return queue.pop(0)

    monkeypatch.setattr(DshModelClient, "_run_completion", stopping)
    queue.extend([_Failed(OAUTH_DNS)] * 4)
    run = tmp_path / "run"
    # A second check bound later does not hide the first.
    client = _text_client().with_stop_check(lambda: stop[0]).with_stop_check(lambda: False)
    with pytest.raises(RunStopped, match="a person asked the run to stop") as raised:
        client.complete("hello", workdir=run)
    assert waits == [5.0, 10.0] and len(queue) == 1
    assert interrupted_cause(f"solve: RunStopped: {raised.value}") == "stopped"
    assert [e["state"] for e in _events(run, NETWORK_WAIT_EVENT)] == ["waiting", "stopped"]


def test_a_stop_check_that_fails_is_no_stop(answers):
    queue, waits = answers
    queue.extend([_Failed(OAUTH_DNS), _Done()])

    def broken() -> bool:
        raise OSError("control file unreadable")

    assert _text_client().with_stop_check(broken).complete("hello") == "done\n"


@pytest.mark.parametrize("scope", ["own", "inherited"])
def test_cancellation_during_the_wait(monkeypatch, tmp_path, scope):
    """The call's scope -- its own, or one it inherited from a fork around it -- is
    cancelled while it waits: it stops waiting within a poll and is not asked again."""
    calls: list[int] = []

    def failing(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(1)
        return _Failed(OAUTH_DNS)

    monkeypatch.setattr(DshModelClient, "_run_completion", failing)
    monkeypatch.setattr(model, "NETWORK_FIRST_WAIT_S", 30.0)
    # Its own scope wakes the wait at once, whatever the poll; an inherited one is seen
    # at the next poll.
    monkeypatch.setattr(model, "NETWORK_POLL_S", 10.0 if scope == "own" else 0.05)
    cancelled, inner = threading.Event(), threading.Event()
    timer = threading.Timer(0.2, cancelled.set)
    timer.start()
    client = _text_client().with_cancellation(cancelled)
    if scope == "inherited":
        client = client.with_cancellation(inner)
    run = tmp_path / "run"
    started = time.monotonic()
    with pytest.raises(CancelledError):
        client.complete("hello", workdir=run)
    assert time.monotonic() - started < 5 and calls == [1]
    assert [e["state"] for e in _events(run, NETWORK_WAIT_EVENT)] == ["waiting", "cancelled"]


def test_observers_get_each_outage(answers):
    queue, waits = answers
    queue.extend([_Failed(OAUTH_DNS), _Done(), _Failed("dsh: TRANSPORT: socket hang up"),
                  _Failed("dsh: TRANSPORT: read ECONNRESET"), _Done()])
    client = _text_client()
    with observe_network_waits() as outer:
        with observe_network_waits() as inner:
            client.complete("one")
        client.complete("two")
    assert [(e["failures"], e["outcome"]) for e in inner] == [(1, "recovered")]
    assert [(e["failures"], e["outcome"]) for e in outer] == [(1, "recovered"), (2, "recovered")]
    assert outer[1]["error"] == "dsh: TRANSPORT: read ECONNRESET", "the outage's last cause"
    assert set(outer[0]) == {"started_at", "ended_at", "waited_s", "error", "failures", "outcome",
                             "usage"}


# ── the SkillOpt port's three attempts ──────────────────────────────────────


def test_skillopt_ask_sees_one_attempt_per_real_failure(answers, tmp_path):
    """SkillOpt's ``ask`` makes three attempts. Outages inside an attempt are the
    client's to wait out: here two outages and a rate limit make one failed attempt, and
    an outage then an answer the second; each attempt's trace line lists its outages."""
    from refactory_packages.skillopt.trace import TraceRecorder, ask, no_retry_waits

    queue, waits = answers
    queue.extend([_Failed(OAUTH_DNS), _Failed(OAUTH_DNS), _Failed(RATE_LIMITED),
                  _Failed("dsh: TRANSPORT: getaddrinfo EAI_AGAIN api.anthropic.com"), _Done()])
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    with no_retry_waits():
        output = ask(_text_client(), trace, "analyst", "user", system="system",
                     call_key=("analyst", 1, 0, 0))
    assert output == "done\n" and queue == []
    lines = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [line["attempt"] for line in lines] == [1, 2]
    assert "rate_limit_error" in lines[0]["error"] and "error" not in lines[1]
    assert [len(line["network_waits"]) for line in lines] == [1, 1]
    assert lines[0]["network_waits"][0]["failures"] == 2
    assert trace.optimizer_calls == 2 and len(trace.optimizer_failures) == 1


def test_skillopt_ask_without_an_outage_writes_no_network_waits(answers, tmp_path):
    from refactory_packages.skillopt.trace import TraceRecorder, ask

    queue, waits = answers
    queue.append(_Done())
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    assert ask(_text_client(), trace, "analyst", "u", system="s", call_key=("analyst", 1, 0, 0))
    [line] = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert "network_waits" not in line


# ── the loop hands its model a person's stop ────────────────────────────────


def test_the_loop_binds_a_person_stop_to_its_model_calls(tmp_path):
    from refactory_dsh.control import Control
    from refactory_dsh.loop import Loop, LoopConfig

    class Budget:
        def bind_model(self, model):  # type: ignore[no-untyped-def]
            return model

    loop = Loop("circle_packing/adaptive",
                LoopConfig(epochs=1, batch_size=1, seed=5, optimizer="random",
                           runs_root=tmp_path / "run"), project=Path("."))
    loop.model = DshModelClient(worker_backend="dsh", provider="p", model="m")
    loop.bind_budget(Budget())
    assert loop.model._stop_requested() is False
    assert loop._judge_model._stop_requested() is False
    Control(loop.root).path.parent.mkdir(parents=True, exist_ok=True)
    Control(loop.root).path.write_text(json.dumps({"action": "pause"}))
    assert loop.model._stop_requested() is False, "a pause is no stop"
    Control(loop.root).path.write_text(json.dumps({"action": "stop"}))
    assert loop.model._stop_requested() is True
    assert loop._judge_model._stop_requested() is True, "the optimizer's route has it too"
    assert loop.evaluator.model is loop.model


# ── review of #649: what decides, slow tries, the bound once, jitter ───────


def test_the_models_reasoning_never_decides_an_outage(answers):
    """A session whose streamed reasoning quotes a connection error from its own tool
    work, then ends without a diagnostic, is not the network's failure: it is not waited
    out or asked again (a session with tools would repeat its work). A real diagnostic
    after the same reasoning still is."""
    queue, waits = answers
    reasoning = ("dsh: reasoning:\nThe test database is down: connect ECONNREFUSED "
                 "127.0.0.1:5432. Let me look at the fixture.\n")
    queue.extend([_Failed(reasoning), _Done()])
    with pytest.raises(CompletionFailed):
        DshModelClient(worker_backend="dsh", provider="p", model="m").complete("hello")
    assert len(queue) == 1 and waits == []
    queue[:] = [_Failed(reasoning + "dsh: TRANSPORT: getaddrinfo ENOTFOUND api.anthropic.com"),
                _Done()]
    assert DshModelClient(worker_backend="dsh", provider="p", model="m").complete("hello") == "done\n"
    assert queue == [] and waits == [5.0]


def test_a_request_that_keeps_failing_slowly_is_not_an_outage(answers, monkeypatch):
    """Each try runs 400 s and then the proxy hangs up (the request runs past its timeout):
    the network carried every generation, so after three such tries the call fails
    instead of paying for long generations for two hours."""
    queue, waits = answers
    clock = [1000.0]

    def run(*args, **kwargs):  # type: ignore[no-untyped-def]
        clock[0] += 400
        return queue.pop(0)

    monkeypatch.setattr(DshModelClient, "_run_completion", run)
    monkeypatch.setattr(model.time, "monotonic", lambda: clock[0])
    queue.extend([_Failed("dsh: TRANSPORT: 504 Gateway Timeout")] * 10)
    with pytest.raises(CompletionFailed, match="3 of its tries each ran 300s") as raised:
        _text_client().complete("hello")
    assert waits == [5.0, 10.0] and len(queue) == 7
    assert interrupted_cause(str(raised.value)) == "transport"


def test_after_one_call_waited_the_bound_the_next_fails_at_once(answers, monkeypatch):
    """Once a call has waited the whole bound, an outage met by the next call (an
    optimizer call's next attempt) fails at once instead of waiting the bound again,
    until a call gets through; then outages are waited out again."""
    queue, waits = answers
    monkeypatch.setattr(model, "NETWORK_WAIT_S", 30.0)
    queue.extend([_Failed(OAUTH_DNS)] * 3)
    with pytest.raises(CompletionFailed, match="the most one call waits"):
        _text_client().complete("one")
    assert waits == [5.0, 10.0] and model.NETWORK_GIVEN_UP.is_set()
    queue[:] = [_Failed(OAUTH_DNS), _Done()]
    with pytest.raises(CompletionFailed, match="an earlier call already waited 30s"):
        _text_client().complete("two")
    assert waits == [5.0, 10.0] and len(queue) == 1
    assert _text_client().complete("three") == "done\n"
    assert not model.NETWORK_GIVEN_UP.is_set(), "an answer shows the network is up"
    queue[:] = [_Failed(OAUTH_DNS), _Done()]
    assert _text_client().complete("four") == "done\n"
    assert waits == [5.0, 10.0, 5.0]


def test_a_failure_that_is_not_an_outage_also_shows_the_network_is_up(answers):
    queue, waits = answers
    model.NETWORK_GIVEN_UP.set()
    queue.extend([_Failed(RATE_LIMITED)])
    with pytest.raises(CompletionFailed):
        _text_client().complete("hello")
    assert not model.NETWORK_GIVEN_UP.is_set()


def test_skillopt_ask_waits_the_bound_once_not_per_attempt(answers, monkeypatch, tmp_path):
    """The worst case of one optimizer call: its first attempt waits the bound, and its
    other two fail at once (`NETWORK_GIVEN_UP`) rather than waiting it again."""
    from refactory_packages.skillopt.trace import TraceRecorder, ask, no_retry_waits

    queue, waits = answers
    monkeypatch.setattr(model, "NETWORK_WAIT_S", 30.0)
    queue.extend([_Failed(OAUTH_DNS)] * 10)
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    with no_retry_waits():
        assert ask(_text_client(), trace, "analyst", "u", system="s",
                   call_key=("analyst", 1, 0, 0)) is None
    assert waits == [5.0, 10.0] and len(queue) == 10 - 3 - 1 - 1
    assert len(trace.optimizer_failures) == 3


def test_waits_are_jittered(answers, monkeypatch):
    queue, waits = answers
    monkeypatch.setattr(model, "_jitter", lambda: 1.2)
    queue.extend([_Failed(OAUTH_DNS)] * 3 + [_Done()])
    assert _text_client().complete("hello") == "done\n"
    assert waits == pytest.approx([6.0, 12.0, 24.0])
    monkeypatch.undo()
    draws = [model._jitter() for _ in range(2000)]
    assert 0.8 <= min(draws) < 0.85 and 1.15 < max(draws) <= 1.2
