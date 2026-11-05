"""Controlling a running search: stop, pause, resume, from another process.

Words reach a search through the intent ledger; control is a small file the loop
polls at a generation boundary, because a panel or a terminal has to be able to
hold or end a run without killing it and without writing prose.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from refactory_dsh.adopt import best
from refactory_dsh.control import Control, ControlError
from refactory_dsh.loop import Loop, LoopConfig

MODE = "circle_packing/adaptive"


def _config(root: Path, **overrides: object) -> LoopConfig:
    base = dict(epochs=2, batch_size=2, optimizer="random", seed=5, runs_root=root,
                control_poll_s=0.01)
    base.update(overrides)
    return LoopConfig(**base)  # type: ignore[arg-type]


def _progress(root: Path) -> list[dict]:
    path = root / "progress.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_a_request_round_trips_and_absent_means_run(tmp_path):
    control = Control(tmp_path)
    assert control.read() is None
    control.request("pause")
    assert control.read() == "pause"
    control.request("resume")
    assert control.read() == "resume"
    control.clear()
    assert control.read() is None


def test_an_unknown_action_is_refused_by_name(tmp_path):
    with pytest.raises(ControlError) as refused:
        Control(tmp_path).request("explode")
    assert "explode" in str(refused.value)


def test_an_unreadable_request_reads_as_none(tmp_path):
    """A half-written or hand-edited file must not be read as an instruction."""
    (tmp_path / "control.json").write_text("{not json")
    assert Control(tmp_path).read() is None


def test_a_stop_ends_the_run_with_the_reason(tmp_path):
    Control(tmp_path).request("stop")
    report = Loop(MODE, _config(tmp_path), project=Path(".")).run()
    assert "a person asked the run to stop" in str(report["bound"])


def test_a_pause_holds_the_run_until_it_is_resumed(tmp_path):
    control = Control(tmp_path)
    control.request("pause")

    def release() -> None:
        time.sleep(0.1)
        control.request("resume")

    thread = threading.Thread(target=release, daemon=True)
    thread.start()
    report = Loop(MODE, _config(tmp_path), project=Path(".")).run()
    thread.join(timeout=5)
    assert report["evaluations"] > 0, "the run continued after the resume"
    steps = [record["step"] for record in _progress(tmp_path)]
    assert "paused" in steps


def test_a_pause_then_a_stop_ends_the_run(tmp_path):
    control = Control(tmp_path)
    control.request("pause")

    def stop_it() -> None:
        time.sleep(0.1)
        control.request("stop")

    thread = threading.Thread(target=stop_it, daemon=True)
    thread.start()
    report = Loop(MODE, _config(tmp_path), project=Path(".")).run()
    thread.join(timeout=5)
    assert "a person asked the run to stop" in str(report["bound"])


def test_the_console_steer_verb_pauses_resumes_and_stops(tmp_path):
    """The control file is reachable from the verb a person and the panel already use."""
    from refactory_dsh import console

    for flag, expected in (("--pause", "pause"), ("--resume", "resume"), ("--stop", "stop")):
        assert console.main(["steer", "--run", str(tmp_path), flag]) == 0
        assert Control(tmp_path).read() == expected, flag


# ── the population acts: keep, reject, adopt ───────────────────────────────
def test_a_keep_and_a_reject_carry_the_candidate_they_name(tmp_path):
    """The act names a candidate, so a request without one cannot be honored."""
    control = Control(tmp_path)
    with pytest.raises(ControlError, match="names a candidate"):
        control.request("reject")
    with pytest.raises(ControlError, match="does not take a candidate"):
        control.request("pause", target="abc")
    with pytest.raises(ControlError, match="does not take --lineage"):
        control.request("adopt", lineage=True)
    with pytest.raises(ControlError, match="does not take a destination name"):
        control.request("keep", target="abc", as_name="pack")

    assert control.request("reject", target="abc", lineage=True)["lineage"] is True
    request = control.read_request()
    assert request["target"] == "abc" and request["lineage"] is True

    # One request at a time: the file holds one act, so a second one asked for
    # before the loop reaches its boundary would drop the first. The loop clears
    # the file when it honors one, so a request still here is one not yet honored.
    with pytest.raises(ControlError, match="still waiting"):
        control.request("adopt", as_name="pack")
    with pytest.raises(ControlError, match="still waiting"):
        control.request("reject", target="other")
    # Asking for the same act again is not a second act.
    assert control.request("reject", target="abc", lineage=True)["target"] == "abc"
    # Pause and stop are not candidate acts: a stop must not wait its turn.
    assert control.request("stop")["action"] == "stop"

    control.clear()
    assert control.request("adopt", as_name="pack")["as"] == "pack"


def test_a_rejected_candidate_leaves_the_leaderboard(tmp_path):
    """Rejection is a selection fact: the score stays recorded, the choice is gone."""
    loop = Loop(MODE, _config(tmp_path), project=Path("."))
    loop.run()
    board = loop.leaderboard()
    assert len(board) > 1, "the run produced nothing to reject"
    victim = board[-1].id

    loop._honor({"action": "reject", "target": victim})
    remaining = [candidate.id for candidate in loop.leaderboard()]
    assert victim not in remaining
    assert victim in [candidate.id for candidate in loop.population], (
        "the run rewrote its own history; the candidate's score is a fact about the run"
    )
    assert loop.report()["rejected"] == [victim], (
        "the report does not say a person dismissed a candidate"
    )
    assert best(loop.root)["candidate"] != victim, (
        "the adopter would write a winner the person had already dismissed"
    )


def test_a_keep_survives_elite_selection(tmp_path):
    """What keep is for: a good point that a newer batch outscored still gets followed."""
    # batch_size=4 so enough distinct candidates survive; F14 removed ops that
    # the random sampler used to waste attempts on, so fewer proposals are
    # refused and more produce duplicate states at small batch sizes.
    loop = Loop(MODE, _config(tmp_path, batch_size=4), project=Path("."))
    loop.run()
    board = loop.leaderboard()
    assert len(board) > 2
    chosen = board[-1].id

    assert chosen not in [candidate.id for candidate in loop.select_parents(1)]
    loop._honor({"action": "keep", "target": chosen})
    assert chosen in [candidate.id for candidate in loop.select_parents(1)], (
        "a kept candidate was culled by elite selection anyway"
    )
    assert loop.report()["kept"] == [chosen]


def test_a_keep_clears_a_rejection_and_a_lineage_reject_takes_the_branch(tmp_path):
    """One rule for both flags: a candidate is in and protected, or out. Not both."""
    loop = Loop(MODE, _config(tmp_path), project=Path("."))
    loop.run()
    seed = loop.population[0].id
    everyone = [candidate.id for candidate in loop.population]
    assert len(everyone) > 1

    loop._honor({"action": "reject", "target": seed, "lineage": True})
    assert sorted(loop.rejected) == sorted(everyone), (
        "a lineage reject left descendants proposing from a branch that was dropped"
    )
    assert loop.leaderboard() == []

    loop._honor({"action": "keep", "target": seed, "lineage": True})
    assert loop.rejected == []
    assert loop.leaderboard(), "clearing a rejection did not bring the candidates back"


def test_a_rejection_our_journal_recorded_survives_a_resume(tmp_path):
    loop = Loop(MODE, _config(tmp_path), project=Path("."))
    loop.run()
    victim = loop.leaderboard()[-1].id
    loop._honor({"action": "reject", "target": victim})

    resumed = Loop(MODE, _config(tmp_path, resume=True), project=Path("."))
    resumed.run()
    assert victim in resumed.rejected, (
        "a resumed run resurrected a branch the person had dismissed"
    )
    assert victim not in [candidate.id for candidate in resumed.leaderboard()]


def test_adopt_now_writes_a_package_without_ending_the_run(tmp_path):
    """The winner is reachable while the search is still going."""
    loop = Loop(MODE, _config(tmp_path, epochs=1), project=tmp_path)
    loop.run()
    winner = loop.leaderboard()[0]

    outcome = loop._honor({"action": "adopt", "as": "adopted_now"})
    assert "error" not in outcome, outcome
    assert outcome["candidate"] == winner.id
    assert outcome["score"] == winner.score
    assert (tmp_path / ".factory" / "packages" / "adopted_now" / "mode.py").exists()
    assert loop.adoptions, "the run did not record what the person's adopt wrote"

    # The refusal is an event, not the end of the search: adopting never overwrites.
    again = loop._honor({"action": "adopt", "as": "adopted_now"})
    assert "already exists" in str(again.get("error")), again


def test_adopt_now_with_no_name_uses_the_packages_own_name(tmp_path):
    """The panel's Adopt best now names nothing, so the default has to be a name.

    It was `f"{self.package}_adopted"`, where `self.package` is the loaded `Package`
    object: the default became a dataclass repr and the filesystem refused it with
    "File name too long". The GUI's Adopt best now is the caller that names nothing,
    which is why no unit test with an explicit `as` caught it.
    """
    loop = Loop(MODE, _config(tmp_path, epochs=1), project=tmp_path)
    loop.run()

    outcome = loop._honor({"action": "adopt"})
    assert "error" not in outcome, outcome
    assert outcome["as"] == "circle_packing_adopted"
    assert (tmp_path / ".factory" / "packages" / "circle_packing_adopted" / "mode.py").exists()


def test_a_destination_that_is_not_a_package_name_is_refused(tmp_path):
    """One directory, named like a package: not a path, not a repr, not empty."""
    loop = Loop(MODE, _config(tmp_path, epochs=1), project=tmp_path)
    loop.run()

    # An empty `as` is not a name, it is "no name given": that falls back to the
    # default above. The rest are refusals.
    for wrong in ("../../escape", "Not A Name", "with/slash"):
        outcome = loop._honor({"action": "adopt", "as": wrong})
        assert "error" in outcome, wrong
        assert "package name" in outcome["error"], outcome
    assert not (tmp_path / "escape").exists()
    assert not (tmp_path.parent / "escape").exists()


def test_a_one_shot_act_is_not_repeated_at_the_next_boundary(tmp_path):
    """A boundary that adopted the same winner every epoch is a bug, not enthusiasm.

    A finished search refuses to queue a population act it can no longer reach, so
    the file a live boundary would read is written directly here; what this pins is
    that the boundary consumes it once rather than honoring it again.
    """
    loop = Loop(MODE, _config(tmp_path), project=Path("."))
    loop.run()
    victim = loop.leaderboard()[-1].id
    control = Control(tmp_path)

    with pytest.raises(ControlError, match="search is finished"):
        control.request("reject", target=victim)

    (tmp_path / "control.json").write_text(
        json.dumps({"action": "reject", "target": victim, "at": 0.0})
    )
    assert control.read() == "reject"
    assert loop._apply_control() is None
    assert loop.rejected == [victim]
    assert control.read() is None, "the act stayed in the file to be honored again"


def test_the_console_steer_verb_carries_the_population_acts(tmp_path):
    from refactory_dsh import console

    control = Control(tmp_path)
    # Each act is consumed the way the loop consumes it, so the next one is a first
    # act again rather than a second request in a file that holds one.
    assert console.main(["steer", "--run", str(tmp_path), "--reject", "abc"]) == 0
    request = control.read_request()
    assert request["action"] == "reject" and request["target"] == "abc"
    control.clear()

    assert console.main(["steer", "--run", str(tmp_path), "--keep", "abc", "--lineage"]) == 0
    assert control.read_request()["lineage"] is True
    control.clear()

    assert console.main(["steer", "--run", str(tmp_path), "--adopt", "--as", "pack"]) == 0
    assert control.read_request()["as"] == "pack"
    control.clear()

    # A flag that names no act is refused where the person can fix it, rather than
    # written for the loop to interpret.
    assert console.main(["steer", "--run", str(tmp_path), "--lineage"]) != 0
    assert control.read_request() is None

    # One act at a time: a second one would silently drop the first.
    assert console.main(["steer", "--run", str(tmp_path), "--keep", "abc"]) == 0
    assert console.main(["steer", "--run", str(tmp_path), "--reject", "abc"]) != 0
    assert control.read_request()["action"] == "keep", (
        "the second act replaced the first instead of being refused"
    )
    assert console.main(["steer", "--run", str(tmp_path), "--pause"]) == 0, (
        "a pause is not held up behind a pending candidate act"
    )


def test_the_live_leaderboard_marks_what_a_person_did(tmp_path):
    """A reader of a live run sees their own acts in the table they are reading."""
    from refactory_dsh import console

    loop = Loop(MODE, _config(tmp_path), project=Path("."))
    loop.run()
    board = loop.leaderboard()
    loop._honor({"action": "keep", "target": board[0].id})
    loop._honor({"action": "reject", "target": board[-1].id})

    rows = {row["id"]: row for row in console.leaderboard(tmp_path)}
    assert rows[board[0].id]["kept"] is True
    assert rows[board[-1].id]["rejected"] is True
    assert rows[board[1].id]["kept"] is False
