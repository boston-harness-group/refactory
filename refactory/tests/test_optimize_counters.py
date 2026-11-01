"""Optimize's progress lines say what they count, and the counts agree (dm-265).

The README's offline search printed ``improved False`` for an epoch whose best rose
from the seed's 0.795 to 0.875, and ``proposals 4 · accepted 4 · refused 1``: the first
count was only the first round of proposals, the others every round's.
"""
from __future__ import annotations

import json
from pathlib import Path

from refactory_dsh import events
from refactory_dsh.loop import Loop, LoopConfig
from refactory_dsh.optim import Optimizer, Proposal

MODE = "circle_packing/adaptive"


def _readme_search(root: Path) -> tuple[dict, list[dict]]:
    """The README's search: random, knob_only, walker, 2 epochs of 4, 8 evaluations, seed 5."""
    config = LoopConfig(optimizer="random", edit_preset="knob_only", evaluator="walker",
                        epochs=2, batch_size=4, max_evaluations=8, seed=5, runs_root=root)
    report = Loop(MODE, config, project=Path(".")).run()
    records = [json.loads(line) for line in (root / "progress.jsonl").read_text().splitlines()]
    return report, records


def test_the_first_epoch_is_compared_with_the_seed(tmp_path):
    report, records = _readme_search(tmp_path)
    first = next(r for r in records if r.get("step") == "scheduler" and r.get("epoch") == 1)
    assert report["seed_score"] < first["best"], "the README's search improves on its seed"
    assert first["improved"] is True
    assert "improved True" in events.render(first)
    loss = next(r for r in records if r.get("step") == "loss" and r.get("epoch") == 1)
    assert loss["best_ever"] == first["best"]


def test_the_seed_alone_is_not_an_improvement(tmp_path):
    """An epoch whose best is the seed's own score did not improve on it."""
    loop = Loop(MODE, LoopConfig(optimizer="random", epochs=1, batch_size=1, seed=5,
                                 runs_root=tmp_path), project=Path("."))
    loop.init()
    loop.population = loop.population[:1]  # the seed, and none of its neighbours
    loop.open_epoch()
    loop.forward()
    assert loop.loss_step()["improved"] is False
    assert loop.loss["best"] == loop._seed_score()


def test_a_seed_that_did_not_run_gives_the_first_epoch_nothing_to_improve_on(tmp_path):
    """A seed scored in an old journal without an outcome is kept but ineligible: its score
    is not the point the search started from."""
    loop = Loop(MODE, LoopConfig(optimizer="random", epochs=1, batch_size=2, seed=5,
                                 runs_root=tmp_path), project=Path("."))
    loop.init()
    seed, other = loop.population[0], loop.population[1]
    seed.evaluated, seed.exit_ok, seed.score = True, False, 0.1
    other.evaluated, other.exit_ok, other.score = True, True, 0.5
    loop.population = [seed, other]
    assert loop._seed_score() is None
    assert loop.loss_step()["improved"] is False


def test_proposals_are_the_accepted_and_the_refused_of_the_epoch(tmp_path):
    _, records = _readme_search(tmp_path)
    lines = [r for r in records if r.get("step") == "optimizer"]
    assert lines, "the search proposed"
    for record in lines:
        assert record["proposed"] == record["accepted"] + record["refused"], record
        assert sum(record["epoch_refusal_codes"].values()) >= record["refused"]
    # The README's second epoch: four accepted after one duplicate was refused.
    second = next(r for r in lines if r["epoch"] == 2)
    assert (second["proposed"], second["accepted"], second["refused"]) == (5, 4, 1)
    assert events.render(second) == (
        "epoch 2 · proposals 5 this epoch: accepted 4 for evaluation, refused 1 (E-duplicate 1)")


class _Repeating(Optimizer):
    """Proposes one edit over and over, so every epoch refuses some of its proposals."""

    name = "repeating"

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        return [Proposal(edit={"op": "set_knob", "name": "grid", "value": 18},
                         parent=parents[0].id, rationale="again") for _ in range(max(1, n))]


def test_each_epochs_refusals_are_counted_with_that_epochs_codes(tmp_path):
    config = LoopConfig(optimizer="random", edit_preset="knob_only", epochs=3, batch_size=2,
                        seed=5, runs_root=tmp_path)
    report = Loop(MODE, config, project=Path("."), optimizer=_Repeating(k=1)).run()
    records = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    lines = [r for r in records if r.get("step") == "optimizer"]
    assert [r["epoch"] for r in lines] == [1, 2, 3]
    for record in lines:
        assert record["proposed"] == record["accepted"] + record["refused"], record
        assert sum(record["epoch_refusal_codes"].values()) == record["refused"], record
    last = lines[-1]
    assert last["refused"] and sum(last["refusal_codes"].values()) > last["refused"], (
        "the run's tally has more than the last epoch's own refusals")
    # The JSON summary keeps the two apart the same way.
    summary = report["epochs_summary"]
    assert [s["refused"] for s in summary] == [r["refused"] for r in lines]
    assert summary[-1]["refused_total"] > summary[-1]["refused"] > 0


def test_the_line_names_the_epochs_codes_not_the_runs():
    line = events.render({"step": "optimizer", "epoch": 3, "proposed": 4, "accepted": 4,
                          "refused": 0, "epoch_refusal_codes": {},
                          "refusal_codes": {"E-duplicate": 2}})
    assert line == "epoch 3 · proposals 4 this epoch: accepted 4 for evaluation, refused 0"


def test_the_json_epoch_summary_counts_what_the_progress_line_counts(tmp_path):
    """`optimize --json` said proposed 5, accepted 12, rejected 1 for the README's epoch 2:
    12 was the population and 1 the run's total. Each count is now the epoch's own."""
    report, records = _readme_search(tmp_path)
    lines = {r["epoch"]: r for r in records if r.get("step") == "optimizer"}
    keys = ("proposed", "accepted", "refused")
    for summary in report["epochs_summary"]:
        assert {k: summary[k] for k in keys} == {k: lines[summary["epoch"]][k] for k in keys}
    second = report["epochs_summary"][1]
    assert {k: second[k] for k in (*keys, "population", "refused_total")} == {
        "proposed": 5, "accepted": 4, "refused": 1, "population": 12, "refused_total": 1}
    assert not {"rejected", "structural_accepted"} & set(second), "the old, misleading keys"
