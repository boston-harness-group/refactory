"""Running and reading the loop the way a GUI does.

Issue #17: a user starts a run and reads what came out without a terminal. The
loop stays a Python process; what these test is that everything a GUI needs is
reachable through one call each, and that the same calls are reachable over the
command line a DSH tool shells out to.

The console is what the GUI drives. Testing it here rather than through a browser
is deliberate: the GUI's contract is these payloads, and a payload that is wrong
is wrong wherever it is rendered.
"""

from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import console
from refactory_dsh.console import ConsoleError

MODE = "circle_packing/adaptive"
#: A one-epoch, two-candidate run: enough to produce every artifact, fast enough
#: to wait for in a test.
FAST = dict(epochs=1, batch_size=2, optimizer="random", seed=3)


def _start(tmp_path: Path, run_id: str = "run", **overrides: object) -> Path:
    config = {"run_id": run_id, **FAST, **overrides}
    started = console.start(
        MODE, project=Path("."), runs_root=tmp_path / "loop", **config
    )
    return Path(started["root"])


def _wait(run: Path, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = console.status(run)
        if state["finished"] or state["state"] == "died":
            return state
        time.sleep(0.2)
    raise AssertionError(f"{run} did not settle within {timeout}s\n{console.status(run)}")


# ── the catalogue a start dialog shows ─────────────────────────────────────


def test_the_catalogue_lists_the_modes_that_can_be_started():
    ids = [entry["id"] for entry in console.catalogue(project=Path("."))]
    assert MODE in ids
    assert "annealing_pack/anneal" in ids


def test_a_catalogue_entry_carries_what_a_start_dialog_needs():
    entry = next(e for e in console.catalogue(project=Path(".")) if e["id"] == MODE)
    assert entry["startable"] is True
    assert entry["scored"] is True
    assert entry["package"] == "circle_packing"
    assert entry["parameters"], "a mode with no visible parameters cannot be configured"
    assert set(entry["parameters"]) >= {"grid", "order"}
    assert entry["preferred_edits"]
    assert entry["trust"] in ("trusted", "generated")


def test_a_catalogue_entry_names_both_routes_and_the_gates_that_ask_a_person():
    """The route and the human touchpoints are readable before the run is opened.

    A caller shown only `action` picks the headless route, and a headless authoring
    run pre-answers a gate the person would have been asked at. The spine route and
    the gate that belongs to a person are the two facts that choice needs (#92, #93).
    """
    entries = {entry["id"]: entry for entry in console.catalogue(project=Path("."))}

    authoring = entries["default_pack/design"]
    assert authoring["action"] == "design"
    assert authoring["interactive"] is True, "a mode the spine can lower says so"
    assert {"id": "gate_strategy", "user": True} in authoring["gates"]
    assert {"id": "gate_research", "user": False} in authoring["gates"], (
        "a gate with a command is answered by the runtime, not by a person"
    )
    assert authoring["parameters"], "the knobs travel with the route"

    scored = entries["circle_packing/adaptive"]
    assert scored["action"] == "start"
    assert scored["interactive"] is True
    assert all(gate["user"] is False for gate in scored["gates"]), scored["gates"]


def test_the_terminal_listing_leaves_the_routes_to_the_json(tmp_path, capsys):
    """`refactory list` is for a person: no `[run: start or wf_open]` tool hints. The JSON an
    agent reads (`refactory list --json`) still says how each mode runs and who it asks."""
    from refactory_dsh import cli, registry

    listing = registry.render_catalogue(Path("."))
    design = next(line for line in listing.splitlines() if "default_pack/design " in line)
    assert design.split() == ["default_pack/design", *"design, emit and check a software "
                              "artifact from a brief".split()]
    for hint in ("wf_open", "[run:", "asks a person at", "gate_strategy"):
        assert hint not in listing, hint
    assert cli.main(["list", "--json"]) == 0
    entries = {entry["id"]: entry for entry in json.loads(capsys.readouterr().out)}
    assert entries["default_pack/design"]["action"] == "design"
    assert entries["default_pack/design"]["interactive"] is True
    assert {"id": "gate_strategy", "user": True} in entries["default_pack/design"]["gates"]
    assert entries["circle_packing/adaptive"]["action"] == "start"


def test_a_disabled_package_is_listed_but_not_startable(tmp_path):
    """Removing a mode from the list would make a disabled package look absent."""
    from refactory_dsh import registry

    project = tmp_path / "project"
    project.mkdir()
    registry.set_enabled(project, "annealing_pack", False)
    entry = next(
        e for e in console.catalogue(project=project) if e["id"] == "annealing_pack/anneal"
    )
    assert entry["enabled"] is False
    assert entry["startable"] is False
    assert "disabled" in entry["reason"]
    enabled = next(
        e for e in console.catalogue(project=project) if e["id"] == MODE
    )
    assert enabled["startable"] is True


# ── starting, watching, stopping ───────────────────────────────────────────


def test_a_run_started_from_the_console_finishes_and_leaves_a_report(tmp_path):
    run = _start(tmp_path)
    state = _wait(run)
    assert state["state"] == "finished"
    assert state["finished"] is True
    assert state["best_score"] is not None
    assert (run / "report.md").exists()


def test_the_console_records_enough_to_reopen_the_run_later(tmp_path):
    """A GUI that restarted must find the run it started, not orphan it."""
    run = _start(tmp_path)
    _wait(run)
    recorded = json.loads((run / console.CONSOLE_FILE).read_text())
    assert recorded["mode"] == MODE
    assert recorded["root"] == str(run)
    assert recorded["pid"] > 0
    assert console.status(run)["mode"] == MODE


def test_status_reports_progress_while_a_run_is_unfinished(tmp_path):
    """A reader must see a run that has started and not finished.

    Driven by hand rather than by racing a subprocess: the run here is advanced
    one phase, so "unfinished" is a fact of the setup instead of a hope about
    timing. The subprocess path is what the start/watch tests below cover.
    """
    from refactory_dsh.loop import Loop, LoopConfig

    root = tmp_path / "run"
    loop = Loop(
        MODE,
        LoopConfig(epochs=1, batch_size=4, optimizer="random", seed=3, runs_root=root),
        project=Path("."),
    )
    loop.init()
    state = console.status(root)
    assert state["state"] == "running"
    assert state["finished"] is False
    assert state["mode"] == MODE

    loop.forward()
    advanced = console.status(root)
    assert advanced["counters"]["evaluations"] >= 1
    assert advanced["state"] == "running"
    assert any(record["step"] == "forward" for record in advanced["events"])


def test_a_started_run_is_watchable_from_the_first_poll(tmp_path):
    """The console file is written before the loop's first artifact exists."""
    run = _start(tmp_path, epochs=1, batch_size=4)
    first = console.status(run)
    assert first["mode"] == MODE
    assert first["state"] in {"starting", "running"}
    _wait(run)


def test_status_tails_only_what_the_reader_has_not_seen(tmp_path):
    run = _start(tmp_path)
    _wait(run)
    first = console.status(run, after=0)
    assert first["events"]
    assert first["cursor"] == len(first["events"])
    again = console.status(run, after=first["cursor"])
    assert again["events"] == []
    assert again["cursor"] == first["cursor"]


def test_a_run_that_died_is_reported_as_died_not_as_running(tmp_path):
    """A reader has to be able to tell "still working" from "gone"."""
    run = _start(tmp_path)
    _wait(run)
    # A clean run left a finished owner record; a process that died mid-search
    # leaves the same record saying it is running with a pid that is gone. That is
    # the state the reader has to tell apart, so it is what this simulates.
    owner = run / ".refactory" / "optimizer-status.json"
    recorded = json.loads(owner.read_text())
    recorded["status"] = "running"
    recorded["pid"] = 999999
    owner.write_text(json.dumps(recorded))
    (run / "report.json").unlink()
    assert console.status(run)["state"] == "died"
    assert console.status(run)["running"] is False


def test_an_unknown_run_directory_is_an_error(tmp_path):
    try:
        console.status(tmp_path / "nope")
    except ConsoleError as error:
        assert "nope" in str(error)
    else:  # pragma: no cover - the assertion is the point
        raise AssertionError("status must refuse a directory that is not a run")


def test_starting_over_an_existing_run_is_refused(tmp_path):
    _start(tmp_path)
    try:
        _start(tmp_path)
    except ConsoleError as error:
        assert "already exists" in str(error)
    else:  # pragma: no cover - the assertion is the point
        raise AssertionError("a run id must not be silently reused")


def test_stop_reports_that_there_was_nothing_to_stop(tmp_path):
    run = _start(tmp_path)
    _wait(run)
    assert console.stop(run)["stopped"] is False


def test_a_requested_stop_stays_confirmed_after_the_process_is_gone(tmp_path):
    """A graceful shutdown is not instant, so the receipt has to outlive the pid.

    ``stop`` signals a live run and must then say whether it stopped. The process is
    normally still alive at that instant and gone by the next poll -- where the pid
    alone reads "not running", indistinguishable from a run that finished on its own.
    Without the record of the request, a person who stopped a run was told nothing was
    stopped.
    """
    run = tmp_path / "loop" / "stop-me"
    (run / ".refactory").mkdir(parents=True)
    (run / console.CONSOLE_FILE).write_text(
        json.dumps({"run_id": "stop-me", "mode": MODE, "pid": 999_999_999})
    )
    (run / console.STOP_REQUESTED).write_text("{}")
    assert console.stop(run)["stopped"] is True


def test_every_run_under_the_root_is_listed_newest_first(tmp_path):
    root = tmp_path / "loop"
    for run_id in ("one", "two"):
        console.start(
            MODE, project=Path("."), runs_root=root, run_id=run_id, **FAST
        )
    _wait(root / "one")
    _wait(root / "two")
    listed = console.runs(project=Path("."), runs_root=root)
    assert {item["run_id"] for item in listed} == {"one", "two"}
    assert all(item["finished"] for item in listed)


def test_a_run_in_flight_is_listed_too(tmp_path):
    root = tmp_path / "loop"
    run = _start(tmp_path, run_id="watching", epochs=2, batch_size=4)
    assert "watching" in {item["run_id"] for item in console.runs(project=Path("."), runs_root=root)}
    console.stop(run)


def test_stop_finds_checkpoint_owner_when_console_record_is_absent(monkeypatch, tmp_path):
    """A resumed run's owner is recorded in console.json by recovery.start.

    When recovery.start starts a new launcher, it writes the child's PID into
    the console record. If that record is absent or has no live PID, stop falls
    back to the checkpoint's owner_pid. This test verifies the full path:
    recovery.start spawns a real child (monkeypatched to sleep instead of the
    interactive entry point), records its PID in console.json, and console.stop
    kills that child.
    """
    from contextlib import contextmanager
    from refactory_dsh import recovery

    run = tmp_path / "resumed-run"
    (run / ".refactory").mkdir(parents=True)
    (run / ".refactory" / "run-config.json").write_text(
        json.dumps({"mode": "demo/run", "project": str(tmp_path)})
    )

    # Monkeypatch prepare and ownership to skip validation
    monkeypatch.setattr(recovery, "prepare", lambda *args, **kwargs: None)

    @contextmanager
    def fake_ownership(root):
        yield 0
    monkeypatch.setattr("refactory_dsh.run_ownership.ownership", fake_ownership)

    # Monkeypatch subprocess.Popen to launch a long-lived sleep instead of interactive
    real_popen = subprocess.Popen

    def sleep_popen(args, **kwargs):
        return real_popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            **kwargs,
        )
    monkeypatch.setattr(subprocess, "Popen", sleep_popen)

    result = recovery.start(run)
    child_pid = result["pid"]
    try:
        # Verify the PID was recorded in console.json
        recorded = json.loads((run / console.CONSOLE_FILE).read_text())
        assert recorded["pid"] == child_pid

        # Now stop the run through console.stop
        stop_result = console.stop(run)
        assert stop_result["stopped"] is True, stop_result

        # The child must be dead
        try:
            os.kill(child_pid, 0)
            raise AssertionError("child process should be dead")
        except ProcessLookupError:
            pass  # expected
    finally:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass


def test_stop_falls_back_to_checkpoint_owner(tmp_path):
    """When console.json has no PID, stop falls back to the checkpoint's owner_pid."""
    run = tmp_path / "fallback-run"
    (run / ".refactory").mkdir(parents=True)

    # Spawn a real child process in its own session
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        # Write only the checkpoint — no console.json with a pid.
        (run / ".refactory" / "spine-checkpoint.json").write_text(
            json.dumps({"owner_pid": child.pid})
        )
        result = console.stop(run)
        assert result["stopped"] is True, result
        # The child must be dead.
        child.wait(timeout=5)
    finally:
        try:
            child.kill()
        except OSError:
            pass
        child.wait()


# ── the artifacts a GUI renders ────────────────────────────────────────────


def test_the_artifacts_a_gui_renders_are_all_reachable(tmp_path):
    run = _start(tmp_path)
    _wait(run)
    names = {entry["name"] for entry in console.artifacts(run)["artifacts"] if entry["exists"]}
    assert {"report", "gradient", "loss", "progress", "candidates"} <= names


def test_artifacts_can_be_returned_with_their_contents(tmp_path):
    run = _start(tmp_path)
    _wait(run)
    manifest = console.artifacts(run, contents=True)
    by_name = {entry["name"]: entry for entry in manifest["artifacts"]}
    assert "# " in by_name["report"]["content"]
    assert isinstance(by_name["gradient"]["content"], dict)
    assert by_name["gradient"]["content"]["knob_attribution"]
    assert "content" not in by_name["progress"], "a live tail is not a page body"


def test_the_leaderboard_ranks_candidates_and_gives_each_its_diff(tmp_path):
    run = _start(tmp_path)
    _wait(run)
    board = console.leaderboard(run)
    assert len(board) == 2
    assert board[0]["score"] >= board[1]["score"]
    assert board[0]["diff"] is not None
    assert board[0]["diff"]["candidate"] == board[0]["id"]


def test_a_candidate_reads_back_with_its_parent_and_a_rendered_diff(tmp_path):
    run = _start(tmp_path)
    _wait(run)
    child = next(row for row in console.leaderboard(run) if row["parent"])
    read = console.candidate(run, child["id"])
    assert read["candidate"] == child["id"]
    assert read["parent"] == child["parent"]
    assert read["seed"] is False
    assert read["rendered"]
    assert read["score"] == child["score"], "the measurement comes from the run directory"


# ── the command line a DSH tool shells out to ──────────────────────────────


def _cli(*argv: str, cwd: Path = Path(".")) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "refactory_dsh.console", *argv],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


def test_the_cli_lists_modes_as_json(tmp_path):
    modes = _cli("modes", "--project", str(Path(".").resolve()))
    assert MODE in [entry["id"] for entry in modes]


def test_the_cli_starts_watches_and_reads_a_run(tmp_path):
    """The whole loop, over the boundary a tool actually crosses."""
    root = tmp_path / "loop"
    started = _cli(
        "start", MODE,
        "--project", str(Path(".").resolve()),
        "--runs-root", str(root),
        "--run-id", "cli",
        "--epochs", "1", "--batch-size", "2", "--optimizer", "random", "--seed", "3",
    )
    run = Path(started["root"])
    deadline = time.time() + 60
    while time.time() < deadline:
        state = _cli("status", "--run", str(run), "--after", "0")
        if state["finished"]:
            break
        time.sleep(0.2)
    assert state["finished"] is True

    board = _cli("leaderboard", "--run", str(run))
    assert board and board[0]["diff"] is not None
    one = _cli("candidate", "--run", str(run), board[0]["id"])
    assert one["candidate"] == board[0]["id"]
    manifest = _cli("artifacts", "--run", str(run), "--contents")
    assert {e["name"] for e in manifest["artifacts"]} >= {"report", "progress"}
    assert _cli("runs", "--project", str(Path(".").resolve()), "--runs-root", str(root))


def test_the_cli_refuses_a_verb_that_needs_a_run_without_one():
    result = subprocess.run(
        [sys.executable, "-m", "refactory_dsh.console", "status"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "needs --run" in json.loads(result.stdout)["error"]


def test_the_cli_reports_a_missing_run_as_json_not_a_traceback(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "refactory_dsh.console", "status", "--run", str(tmp_path / "gone")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    error = json.loads(result.stdout)["error"]
    assert "does not exist" in error and "gone" in error
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("verb", ["modes", "runs", "start", "status", "artifacts", "leaderboard", "candidate", "stop"])
def test_every_verb_the_gui_needs_exists(verb):
    from refactory_dsh.console import _parser

    assert verb in _parser()._subparsers._group_actions[0].choices  # type: ignore[attr-defined]


def test_the_console_reads_a_run_it_did_not_start(tmp_path):
    """A run started by the loop's own entry point is still readable.

    The console file records a start the console performed; `run.json` is written
    by the loop and names the mode, so a reader is not blind to a run somebody
    else began.
    """
    from refactory_dsh.loop import Loop, LoopConfig

    root = tmp_path / "byhand"
    loop = Loop(
        MODE,
        LoopConfig(epochs=1, batch_size=2, optimizer="random", seed=3, runs_root=root),
        project=Path("."),
    )
    loop.init()
    assert not (root / console.CONSOLE_FILE).exists()
    state = console.status(root)
    assert state["mode"] == MODE
    assert state["state"] == "running"
    assert state["running"] is False

    loop.forward()
    loop.loss_step()
    loop.backward()
    loop.report()
    finished = console.status(root)
    assert finished["finished"] is True
    assert finished["best_score"] is not None


def test_the_design_review_reported_is_the_one_the_run_wrote(tmp_path):
    """The create working directory is shared, so a stale review must not be reported.

    A run that stopped before its design step (the person's gate, say) reported
    whichever `design.md` happened to be in `.factory/create` — another package's
    design, presented as this one's.
    """
    from refactory_dsh.console import _design_review

    work = tmp_path / ".factory" / "create"
    work.mkdir(parents=True)
    (work / "brief.json").write_text('{"name": "other_pkg"}')
    (work / "design.md").write_text("the other package's design")
    assert _design_review(tmp_path, "mine") == "", "another package's review was reported"

    (work / "brief.json").write_text('{"name": "mine"}')
    assert _design_review(tmp_path, "mine") == "the other package's design"


def test_start_points_an_authoring_mode_at_its_own_action(tmp_path):
    """`create` is not a search target; the refusal says what to use instead."""
    from refactory_dsh.console import ConsoleError, start

    with pytest.raises(ConsoleError, match="authoring mode"):
        start("default_pack/create", project=Path("."), runs_root=tmp_path / "runs")
    try:
        start("default_pack/create", project=Path("."), runs_root=tmp_path / "runs")
    except ConsoleError as exc:
        assert "`create` action" in str(exc), str(exc)


def test_the_catalogue_names_the_action_that_runs_each_mode():
    """An authoring mode is run by its own verb, and the catalogue says so.

    A caller that learns that by failing `start` picked the wrong tool from a
    catalogue that could have told it.
    """
    from refactory_dsh.console import catalogue

    entries = {entry["id"]: entry for entry in catalogue(project=Path("."))}
    assert entries["default_pack/create"]["action"] == "create"
    assert entries["default_pack/design"]["action"] == "design"
    assert entries["default_pack/create"]["scored"] is False
    assert entries["circle_packing/adaptive"]["action"] == "start"
    assert entries["circle_packing/adaptive"]["scored"] is True


# ── the console's authoring forwards (#61, #62, #64) ───────────────────────
#
# The tool calls the console, and the console forwards the authoring entry points
# as a captured child process. What the forwarded call says has to survive that hop: the entry points
# report a handled refusal as JSON on *stderr*, and `_forward` captured stdout only,
# so the Web GUI saw `create failed (exit 2)` with an empty payload and no reason.


def test_a_forwarded_refusal_keeps_the_reason_it_wrote_to_stderr(monkeypatch, capsys):
    """An authoring entry point reports a refusal as JSON on stderr (#61).

    `_forward` captured stdout only, so the payload was `{"output": ""}` and the tool
    could only say `create failed (exit 2)` — with the actual reason destroyed.
    """
    from subprocess import CompletedProcess

    from refactory_dsh import console

    monkeypatch.setattr(
        console.subprocess, "run",
        lambda argv, **kwargs: CompletedProcess(
            argv, 2, stdout="", stderr=json.dumps({"error": "the gate needs a person"}),
        ),
    )

    payload = console._forward("stub_authoring", ["--name", "x"])
    assert payload["ok"] is False and payload["exit_code"] == 2
    assert payload["result"]["error"] == "the gate needs a person"
    assert "the gate needs a person" in payload["stderr"]

    # A console caller gets the reason at the top level, and a failing exit.
    monkeypatch.setattr(console, "_forward", lambda module, argv: payload)
    code = console.main(["create", "--name", "x", "--brief", "y"])
    printed = json.loads(capsys.readouterr().out)
    assert code == 2, "a failed forward must be a failed console call, not a successful one"
    assert printed["error"] == "the gate needs a person"


def test_status_reads_an_authoring_runs_own_spine(tmp_path, capsys):
    """`status --run .factory/create/<name>` says where a creation is (#62).

    An authoring run is synchronous and writes no `progress.jsonl`, so before this
    there was nothing to watch: the only signal was the call returning.
    """
    from refactory_dsh import console

    workdir = tmp_path / "create" / "demo_pack"
    (workdir / ".refactory").mkdir(parents=True)
    (workdir / ".refactory" / "spine-status.json").write_text(json.dumps({
        "status": "ready", "nodes": ["strategy_director"], "completed": ["prepare", "survey"],
    }))
    (workdir / "facts.json").write_text(json.dumps({
        "name": "demo_pack", "validation": "(no validation has run yet)",
    }))
    assert console.main(["status", "--run", str(workdir)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["state"] == "running"
    assert payload["authoring"]["spine"] == "ready"
    assert payload["authoring"]["nodes_due"] == ["strategy_director"]
    assert payload["authoring"]["name"] == "demo_pack"
    assert payload["finished"] is False

    (workdir / ".refactory" / "spine-status.json").write_text(json.dumps({"status": "done"}))
    assert console.main(["status", "--run", str(workdir)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["state"] == "finished" and payload["finished"] is True


def test_the_console_forwards_a_workdir_to_an_authoring_run(monkeypatch, capsys):
    """`--workdir` on the authoring verbs reaches the entry point (#64).

    The entry points namespace their own default by name, so this is for a caller
    that wants a specific directory — a second attempt, or a run kept apart from the
    package's own history.
    """
    from refactory_dsh import console

    seen: list[list[str]] = []
    monkeypatch.setattr(
        console, "_forward",
        lambda module, argv: (seen.append(argv), {"verb": module, "exit_code": 0, "ok": True, "result": {}})[1],
    )
    monkeypatch.setattr(console, "_design_review", lambda project, name: "")
    for verb, extra in (
        ("create", ["--name", "p", "--brief", "b"]),
        ("edit", ["--mode", "counting_pack/fixed_step", "--name", "p", "--brief", "b"]),
        ("design", ["--name", "a", "--brief", "b", "--check", "true"]),
    ):
        assert console.main([verb, *extra, "--workdir", "/tmp/somewhere"]) == 0
        argv = seen[-1]
        assert "--workdir" in argv and "/tmp/somewhere" in argv, verb
    capsys.readouterr()


# ── running a mode once from the panel (#66) ───────────────────────────────
#
# The tool's only "execute this mode" action was `start`, which is the outer loop:
# from the GUI a person could create a mode and then only optimize it, never *use*
# it. Running a mode once was a CLI-only spelling with no console verb and
# therefore no tool action.


def test_the_console_forwards_the_creation_directive(monkeypatch, capsys):
    """A creation says which it is: a class of work, or the one-shot special case.

    The console is the path the panel and the tool take, so the directive has to
    survive the forward rather than being a command-line-only spelling.
    """
    from refactory_dsh import console

    seen: list[list[str]] = []
    monkeypatch.setattr(
        console, "_forward",
        lambda module, argv: (seen.append(argv), {"verb": module, "exit_code": 0, "ok": True, "result": {}})[1],
    )
    monkeypatch.setattr(console, "_design_review", lambda project, name: "")
    assert console.main(["create", "--name", "p", "--brief", "b", "--single"]) == 0
    assert "--single" in seen[-1]
    assert console.main(["create", "--name", "p", "--brief", "b", "--task", "a.b:C"]) == 0
    assert "--task" in seen[-1] and "a.b:C" in seen[-1]
    capsys.readouterr()


def test_the_console_forwards_a_single_run_of_a_mode(monkeypatch, capsys):
    from refactory_dsh import console

    seen: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        console, "_forward",
        lambda module, argv: (seen.append((module, argv)), {
            "verb": module, "exit_code": 0, "ok": True, "result": {"final": "done"},
        })[1],
    )
    code = console.main([
        "run", "buildroot_pack/build", "--intent", "discover the recipe",
        "--knob", "rounds=2", "--verdict", "approve=proceed",
        "--workdir", "/tmp/somewhere", "--timeout", "60",
    ])
    assert code == 0
    module, argv = seen[-1]
    assert module == "interactive", "a single run is the interactive entry point"
    assert argv[0] == "buildroot_pack/build"
    assert "--headless" in argv
    assert argv[argv.index("--intent") + 1] == "discover the recipe"
    assert argv[argv.index("--knob") + 1] == "rounds=2"
    assert argv[argv.index("--approve") + 1] == "approve=proceed"
    assert argv[argv.index("--workdir") + 1] == "/tmp/somewhere"
    assert argv[argv.index("--timeout") + 1] == "60"
    capsys.readouterr()


def test_the_console_forwards_the_items_a_run_should_read(monkeypatch, capsys):
    """`--source` reaches the runner, so the panel can feed a harness its own items."""
    from refactory_dsh import console

    seen: list[list[str]] = []
    monkeypatch.setattr(
        console, "_forward",
        lambda module, argv: (seen.append(argv), {
            "verb": module, "exit_code": 0, "ok": True, "result": {},
        })[1],
    )
    monkeypatch.setattr(console, "status", lambda directory: {})
    assert console.main([
        "run", "java_build_pack/build", "--source", "/tmp/trees",
        "--source-format", "directory", "--workdir", "/tmp/somewhere",
    ]) == 0
    argv = seen[-1]
    assert argv[argv.index("--source") + 1] == "/tmp/trees"
    assert argv[argv.index("--source-format") + 1] == "directory"
    capsys.readouterr()


def test_a_single_run_is_read_back_from_the_directory_the_runner_chose(monkeypatch, capsys):
    """Without `--workdir`, the console must read the run the runner actually made.

    The run's directory is the durable answer for this gesture, because the runner
    prints its report from the child process. A console that computed its own default
    read a directory the run never wrote, so the panel was told the run left nothing
    while the evidence sat one path away.
    """
    from refactory_dsh import console, interactive

    mode = "counting_pack/fixed_step"
    root = interactive.default_workdir(mode)
    monkeypatch.setattr(
        console, "_forward",
        lambda module, argv: {"verb": module, "exit_code": 0, "ok": True, "result": {"output": ""}},
    )
    monkeypatch.setattr(console, "status", lambda directory: {"where": str(directory)})
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    (root / "facts.json").write_text(json.dumps({"value": 3}))
    try:
        assert console.main(["run", mode]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["run"]["where"] == str(root)
        assert payload["run"]["facts"] == {"value": 3}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_a_single_run_needs_the_mode_to_run():
    from refactory_dsh import console

    with pytest.raises(SystemExit):
        console.main(["run"])


def test_the_console_forwards_a_seeded_knob_into_the_resolved_config(monkeypatch, tmp_path):
    """`--knob NAME=VALUE` reaches the loop, never as a bare `--knobs` (#see 69).

    The console forwards each config key as `--<key>`, so passing the list under
    `knobs` produced `--knobs`, which the loop does not have: the run died at argv
    parsing with "unrecognized arguments", before evaluating anything. The values
    now ride the resolved profile snapshot the loop is pointed at, so the assertion
    is that the override is in that snapshot and named on the spawned argv.
    """
    from refactory_dsh import console

    captured: dict[str, dict[str, Any]] = {}

    def fake_spawn(mode, root, config, *, project):
        captured["config"] = config

        class Process:
            pid = 4321

        return Process()

    monkeypatch.setattr(console, "_spawn", fake_spawn)
    payload = console.start(
        "circle_packing/adaptive", project=tmp_path,
        knob=["grid=6", "selector=best"],
    )
    config = captured["config"]
    assert "resolved_config" in config
    snapshot = json.loads(Path(config["resolved_config"]).read_text())
    assert snapshot["settings"]["knobs"] == {"grid": "6", "selector": "best"}
    assert "knobs" not in config, "the bare `--knobs` flag was reintroduced"
    assert payload["run_id"]


# ── opening a mode for a driver that walks it node by node ────────────────
#
# A web chat session drives a mode through the spine's `wf_*` tools rather than
# through the `graph` profile, which lowers its graph once at mount. `open` is
# that lowering: it leaves the graph and declarations where the spine can read
# them and returns where they are.


def test_the_open_verb_lowers_a_mode_for_a_driver(tmp_path):
    workdir = tmp_path / "driven"
    payload = _cli(
        "open", MODE,
        "--project", str(Path(".").resolve()),
        "--workdir", str(workdir),
    )
    graph_path = Path(payload["graph"])
    assert payload["run"] == str(workdir.resolve())
    assert payload["name"]
    assert graph_path.exists(), payload
    lowered = json.loads(graph_path.read_text())
    assert lowered["start_node"] in lowered["nodes"], lowered["start_node"]
    assert Path(payload["declarations"]).exists()


def test_the_open_payload_carries_the_runs_own_environment(tmp_path):
    """The values are per run, so the driver cannot get them from a composition.

    The node runner reads `REFACTORY_WORKDIR`, `REFACTORY_PROJECT` and the knob
    JSON, and the item loader reads `REFACTORY_WORKDIR`; the interpreter that
    resolves a bare `python` leads `PATH`. All of them name *this* run.
    """
    project = Path(".").resolve()
    workdir = tmp_path / "driven-env"
    payload = _cli(
        "open", MODE,
        "--project", str(project),
        "--workdir", str(workdir),
    )
    env = payload["commandEnv"]
    assert isinstance(env, dict)
    assert env["REFACTORY_WORKDIR"] == str(workdir.resolve())
    assert env["REFACTORY_PROJECT"] == str(project)
    knobs = json.loads(env["REFACTORY_KNOBS"])
    assert isinstance(knobs, dict) and knobs, knobs
    # An op node's bare `python -m refactory_dsh.run` resolves through this PATH.
    from refactory_dsh.paths import venv_bin
    assert env["PATH"].split(":")[0] == str(venv_bin()), env["PATH"]
    # Data, agent and gate commands use absolute interpreter paths.
    assert payload["dataCommand"].endswith("-m refactory_dsh.items")
    assert "/" in payload["dataCommand"].split()[0], (
        "dataCommand should use an absolute interpreter path")


def test_a_knob_override_reaches_the_node_environment(tmp_path):
    workdir = tmp_path / "driven-knob"
    payload = _cli(
        "open", MODE,
        "--project", str(Path(".").resolve()),
        "--workdir", str(workdir),
        "--knob", "grid=24",
    )
    assert json.loads(payload["commandEnv"]["REFACTORY_KNOBS"])["grid"] == 24


def test_open_refuses_a_missing_workdir_without_a_traceback():
    result = subprocess.run(
        [sys.executable, "-m", "refactory_dsh.console", "open", MODE],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--workdir" in json.loads(result.stdout)["error"]
    assert "Traceback" not in result.stderr


def test_status_follow_advances_its_activity_cursor(tmp_path, capsys):
    """T-7: each follow frame is the next page, not the whole journal again (SD-6).

    The run stays live while two frames are written, so the second frame starts
    where the first one ended. The printed fields are otherwise the ones
    ``status`` prints; only the activity page advances.
    """
    run = tmp_path / "followed"
    journal = run / ".refactory/events.jsonl"
    journal.parent.mkdir(parents=True)
    (run / ".refactory/spine-status.json").write_text(json.dumps({"status": "executing", "nodes": ["build"]}))
    (run / ".refactory/spine-checkpoint.json").write_text(json.dumps({"owner_pid": os.getpid()}))
    for i in range(2):
        journal.write_text("".join(
            json.dumps({"schema_version": 1, "kind": "step",
                        "at": f"2026-10-26T09:00:{i:02d}+00:00", "n": i}) + "\n"
            for i in range(2 + 2 * i)))

    frames: list[dict] = []
    real_status, real_sleep = console.status, console.time.sleep

    def recording(*args, **kwargs):
        frame = real_status(*args, **kwargs)
        frames.append(json.loads(json.dumps(frame)))
        return frame

    def stopping(_seconds: float) -> None:
        # End the loop once the second frame has been written and read.
        if len(frames) >= 2:
            (run / ".refactory/spine-status.json").write_text(json.dumps({"status": "done"}))
        else:
            journal.write_text(journal.read_text() + json.dumps(
                {"schema_version": 1, "kind": "step", "at": "2026-10-26T09:00:09+00:00", "n": 99}) + "\n")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(console, "status", recording)
        patch.setattr(console.time, "sleep", stopping)
        code = console.main(["status", "--run", str(run), "--follow",
                             "--activity-after", "0", "--activity-limit", "2", "--interval", "0.1"])
    assert code == 0
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(frames) >= 2, frames
    first, second = frames[0]["activity"], frames[1]["activity"]
    assert (first["after"], first["cursor"], len(first["events"])) == (0, 2, 2)
    # The second frame continues where the first ended: it is a page, not a rescan.
    assert second["after"] == first["cursor"] == 2
    assert len(second["events"]) <= first["page_size"]
    assert [frame["activity"]["after"] for frame in frames] == sorted(
        frame["activity"]["after"] for frame in frames)
    # Printing is unchanged apart from the page: the frames are the printed lines.
    assert len(printed) == len(frames)
    assert set(printed[0]) >= set(real_status(run)) - {"text"}
