"""Tests for F19 strategy state and G22 public run-control check.

Each test fails when the feature it tests is reverted.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from refactory_dsh.control import Control
from refactory_dsh.loop import Loop, LoopConfig
from refactory_dsh.outerloop import OuterLoop, Run

MODE = "circle_packing/adaptive"


def _config(root: Path, **overrides: object) -> LoopConfig:
    base = dict(
        epochs=2, batch_size=2, optimizer="random", seed=5,
        runs_root=root, control_poll_s=0.01,
    )
    base.update(overrides)
    return LoopConfig(**base)  # type: ignore[arg-type]


# ── F19: strategy state ───────────────────────────────────────────────────

def test_strategy_state_dict_default_is_empty():
    """OuterLoop.strategy_state_dict() returns {} by default."""
    loop = Loop(MODE, _config(Path("/tmp/unused")), project=Path("."))
    assert loop.strategy_state_dict() == {}


def test_strategy_state_survives_stop_and_resume(tmp_path):
    """A strategy saves state, the run stops, replay restores it, and the
    resumed strategy makes the same decision.
    """
    # Run 1: run one epoch, which journals strategy_state after close_epoch.
    loop1 = Loop(MODE, _config(tmp_path, epochs=1), project=Path("."))
    # Give it state before running
    loop1._strategy_state = {"temperature": 0.42, "phase": "explore"}
    loop1.run()

    # Verify the journal has strategy_state
    journal_path = tmp_path / "journal.jsonl"
    journal_lines = journal_path.read_text().splitlines()
    has_strategy_state = any(
        json.loads(line).get("event") == "strategy_state"
        for line in journal_lines if line.strip()
    )
    assert has_strategy_state, "strategy_state was not journaled"

    # Run 2: resume — state should be restored
    loop2 = Loop(MODE, _config(tmp_path, resume=True, epochs=2), project=Path("."))
    loop2.init()  # replay happens here
    assert loop2._strategy_state == {"temperature": 0.42, "phase": "explore"}
    assert loop2.strategy_state_dict() == {"temperature": 0.42, "phase": "explore"}


def test_strategy_state_mid_epoch_save_and_resume(tmp_path):
    """A custom strategy() calls run.save_strategy_state() mid-epoch, the run
    is stopped before epoch end, replay restores the mid-epoch state.
    """
    mid_epoch_state = {"step_count": 7, "direction": "up"}

    def _custom_strategy(self, run: Run) -> None:
        """A custom strategy that saves state mid-epoch and stops."""
        self._strategy_state = dict(mid_epoch_state)
        run.save_strategy_state()
        run.stop("done after saving state")

    # Run 1 with the custom strategy override via monkeypatch
    loop1 = Loop(MODE, _config(tmp_path, epochs=5), project=Path("."))
    import types
    loop1.strategy = types.MethodType(_custom_strategy, loop1)
    loop1.run()

    # Verify the journal has strategy_state
    journal_path = tmp_path / "journal.jsonl"
    journal_lines = journal_path.read_text().splitlines()
    has_strategy_state = any(
        json.loads(line).get("event") == "strategy_state"
        for line in journal_lines if line.strip()
    )
    assert has_strategy_state, "strategy_state was not journaled"

    # Resume and check state
    loop2 = Loop(MODE, _config(tmp_path, resume=True, epochs=5), project=Path("."))
    loop2.init()
    assert loop2._strategy_state == mid_epoch_state
    assert loop2.strategy_state_dict() == mid_epoch_state


def test_strategy_state_with_event_key_survives_resume(tmp_path):
    """A strategy state dict containing the key 'event' is valid JSON-serialisable
    data; it must not crash the write path or lose data on the read path.
    """
    state_with_event = {"event": "custom_milestone", "counter": 3}

    def _custom_strategy(self, run: Run) -> None:
        self._strategy_state = dict(state_with_event)
        run.save_strategy_state()
        run.stop("saved")

    loop1 = Loop(MODE, _config(tmp_path, epochs=5), project=Path("."))
    import types
    loop1.strategy = types.MethodType(_custom_strategy, loop1)
    loop1.run()

    # Resume and verify state is fully restored — including the 'event' key
    loop2 = Loop(MODE, _config(tmp_path, resume=True, epochs=5), project=Path("."))
    loop2.init()
    assert loop2._strategy_state == state_with_event
    assert loop2.strategy_state_dict()["event"] == "custom_milestone"
    assert loop2.strategy_state_dict()["counter"] == 3


def test_empty_state_journaled_and_restored(tmp_path):
    """A strategy that returns {} after previously returning non-empty state:
    the empty state is journaled, and replay restores {} (not the stale state).
    """
    loop = Loop(MODE, _config(tmp_path, epochs=1), project=Path("."))
    loop.init()

    # Journal a candidate so replay has something
    loop._journal_candidate(loop.population[0])

    # First: journal non-empty state
    loop._strategy_state = {"old_data": True}
    loop._save_strategy_state()

    # Then: journal empty state
    loop._strategy_state = {}
    loop._save_strategy_state()

    # Resume: must get empty, not the stale non-empty
    loop2 = Loop(MODE, _config(tmp_path, resume=True, epochs=2), project=Path("."))
    loop2.init()
    assert loop2._strategy_state == {}


def test_old_journal_without_strategy_state_replays_cleanly(tmp_path):
    """A journal with no strategy_state events replays without error, state stays {}."""
    loop1 = Loop(MODE, _config(tmp_path, epochs=1), project=Path("."))
    loop1.run()

    # Remove any strategy_state lines from the journal
    journal_path = tmp_path / "journal.jsonl"
    lines = journal_path.read_text().splitlines()
    filtered = [
        line for line in lines
        if line.strip() and json.loads(line).get("event") != "strategy_state"
    ]
    journal_path.write_text("\n".join(filtered) + "\n")

    loop2 = Loop(MODE, _config(tmp_path, resume=True, epochs=2), project=Path("."))
    loop2.init()
    assert loop2._strategy_state == {}


# ── G22: check_control ────────────────────────────────────────────────────

def test_custom_strategy_honours_stop_via_check_control(tmp_path):
    """A custom strategy calls run.check_control() and receives 'stop'."""
    Control(tmp_path).request("stop")
    loop = Loop(MODE, _config(tmp_path), project=Path("."))
    loop.init()
    run = Run(loop=loop)
    result = run.check_control()
    assert result == "stop"


def test_custom_strategy_honours_pause_via_check_control(tmp_path):
    """run.check_control() blocks during pause and has NOT returned before
    the release.  The test sets an Event just before check_control returns
    and asserts it is NOT set while the pause is active — proving the pause
    actually blocks.
    """
    control = Control(tmp_path)
    control.request("pause")

    loop = Loop(MODE, _config(tmp_path), project=Path("."))
    loop.init()
    run = Run(loop=loop)

    check_returned = threading.Event()

    def check() -> None:
        result = run.check_control()
        check_returned.set()
        assert result is None

    thread_check = threading.Thread(target=check, daemon=True)
    thread_check.start()

    # Wait enough time for check_control to have returned if it were not blocking.
    time.sleep(0.3)
    assert not check_returned.is_set(), (
        "check_control returned before resume — the pause did not block"
    )

    # Now release the pause
    control.request("resume")
    check_returned.wait(timeout=5)
    thread_check.join(timeout=5)
    assert check_returned.is_set(), "check_control did not return after resume"


def test_check_control_one_shot_not_repeated(tmp_path):
    """A one-shot population act honoured through check_control is consumed once."""
    loop = Loop(MODE, _config(tmp_path), project=Path("."))
    loop.run()
    victim = loop.leaderboard()[-1].id
    control = Control(tmp_path)

    # Write the control file directly (the search is finished, so request() refuses)
    (tmp_path / "control.json").write_text(
        json.dumps({"action": "reject", "target": victim, "at": 0.0})
    )
    assert control.read() == "reject"

    run = Run(loop=loop)
    result = run.check_control()
    assert result is None  # reject is a population act, not stop
    assert victim in loop.rejected
    assert control.read() is None, "the act stayed in the file to be honored again"

    # Second call should see nothing
    result2 = run.check_control()
    assert result2 is None


def test_old_format_replay_branch_removed(tmp_path):
    """A journal line with the old format (no 'state' key — strategy keys spread
    as top-level record keys) is silently ignored during replay; the strategy
    state remains {}.  Proves the old-format branch was removed.
    """
    # First run to get a valid journal with at least one candidate
    loop1 = Loop(MODE, _config(tmp_path, epochs=1), project=Path("."))
    loop1.run()

    # Append an old-format strategy_state line (no "state" wrapper)
    journal_path = tmp_path / "journal.jsonl"
    old_format_line = json.dumps({"event": "strategy_state", "temperature": 0.5})
    with journal_path.open("a") as f:
        f.write(old_format_line + "\n")

    # Replay: the old-format line should be ignored; strategy state stays {}
    loop2 = Loop(MODE, _config(tmp_path, resume=True, epochs=2), project=Path("."))
    loop2.init()
    assert loop2._strategy_state == {}, (
        "old-format strategy_state was honoured; the branch should have been removed"
    )
