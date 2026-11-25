"""One event vocabulary, two levels, and the same records behind both.

The failure this locks down: a run wrote ``progress.jsonl`` and printed nothing
until it finished, so a search that was working looked like a hang. The meaningful
level is what a watcher gets; the verbose level is everything, for a caller
debugging the search itself.
"""

from __future__ import annotations

import io

from refactory_dsh import events


def test_a_meaningful_event_renders_one_readable_line():
    line = events.render({
        "step": "forward", "epoch": 1, "candidate": "abc123", "index": 2, "of": 4,
        "score": 0.96, "exit_ok": True, "duration_s": 1.25,
    })
    assert line is not None
    assert "abc123" in line and "2/4" in line and "0.960" in line and "1.2s" in line


def test_a_verbose_only_event_is_hidden_by_default_and_shown_when_asked():
    record = {"step": "backward", "epoch": 1, "knobs": 2, "failures": 0}
    assert events.render(record) is None
    shown = events.render(record, verbose=True)
    assert shown is not None and "backward" in shown


def test_every_meaningful_event_renders_at_the_default_level():
    for step in sorted(events.MEANINGFUL):
        assert events.render({"step": step, "epoch": 1}) is not None, step


def test_the_optimizer_line_names_the_rules_that_refused():
    line = events.render({
        "step": "optimizer", "epoch": 2, "proposed": 3, "accepted": 0, "refused": 5,
        "refusal_codes": {"E-shape": 3, "V7": 2},
    })
    assert line is not None
    assert "refused 5" in line and "E-shape 3" in line and "V7 2" in line


def test_a_failed_candidate_says_so_with_its_reason():
    line = events.render({
        "step": "forward", "epoch": 1, "candidate": "deadbeef", "index": 0, "of": 1,
        "score": None, "exit_ok": False, "error": "the graph did not finish",
    })
    assert line is not None
    assert "failed" in line and "did not finish" in line


def test_the_printer_shows_meaningful_events_and_verbose_shows_more():
    records = [
        {"step": "init", "epoch": 0, "optimizer": "random", "epochs": 1, "batch_size": 2, "population": 1},
        {"step": "loss", "epoch": 1, "best": 0.5},
        {"step": "report", "epoch": 1, "best_score": 0.5, "raw_best_score": 0.5, "evaluations": 1, "bound": "epochs"},
    ]
    quiet = io.StringIO()
    for record in records:
        events.Printer(quiet)(record)
    loud = io.StringIO()
    for record in records:
        events.Printer(loud, verbose=True)(record)
    assert quiet.getvalue().count("\n") == 2
    assert loud.getvalue().count("\n") == 3
    assert "loss" in loud.getvalue() and "loss" not in quiet.getvalue()


def test_a_child_output_line_is_only_printed_when_verbose():
    quiet = io.StringIO()
    events.Printer(quiet).output_line("node says something\n")
    loud = io.StringIO()
    events.Printer(loud, verbose=True).output_line("node says something\n")
    assert quiet.getvalue() == ""
    assert "node says something" in loud.getvalue()


def test_scheduler_line_shows_proposed_and_accepted():
    """The scheduler line says what was proposed this round and what the gate decided."""
    line = events.render({
        "step": "scheduler", "epoch": 2, "best": 0.5, "bound": "",
        "proposed": 3, "accepted": 2,
        "gate": {"kept": 1, "rejected": 1},
        "val_best": 0.7, "val_current": 0.6,
    })
    assert line is not None
    assert "proposed 3, accepted 2" in line
    assert "kept 1, rejected 1" in line


def test_scheduler_checkpoint_line_shows_proposed_and_accepted():
    """Under checkpoint selection, proposed/accepted appear alongside applied/not applied."""
    line = events.render({
        "step": "scheduler", "epoch": 1, "best": 0.5, "bound": "",
        "proposed": 5, "accepted": 4,
        "gate": {"kept": 2, "rejected": 0, "selection": "checkpoint"},
        "val_best": 0.8, "val_current": 0.7,
    })
    assert line is not None
    assert "proposed 5, accepted 4" in line
    assert "applied 2, not applied 0" in line


def test_scheduler_line_without_proposed_still_works():
    """Old records without proposed/accepted still render (backward compat)."""
    line = events.render({
        "step": "scheduler", "epoch": 1, "best": 0.5, "bound": "",
        "gate": {"kept": 1, "rejected": 0, "selection": "checkpoint"},
        "val_best": 0.6, "val_current": 0.6,
    })
    assert line is not None
    assert "applied 1, not applied 0" in line
    assert "proposed" not in line


def test_scheduler_no_gate_shows_proposed():
    """Without a gate, proposed/accepted appear alongside best/mean."""
    line = events.render({
        "step": "scheduler", "epoch": 1, "best": 0.5, "mean": 0.4,
        "proposed": 2, "accepted": 2, "improved": True, "bound": "no",
    })
    assert line is not None
    assert "proposed 2, accepted 2" in line
    assert "best 0.500" in line
