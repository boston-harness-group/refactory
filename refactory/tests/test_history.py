"""Reading a finished run, and the history an optimizer is shown.

Part of the history issue: an optimizer cannot ask "have I tried this?" unless the
run's own past, and previous runs, are readable objects.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from refactory_dsh import history, nn
from refactory_dsh.loop import Loop, LoopConfig

MODE = "circle_packing/adaptive"


def _config(root: Path) -> LoopConfig:
    return LoopConfig(epochs=1, batch_size=3, optimizer="random", seed=5, runs_root=root)


def _run(root: Path) -> dict:
    return Loop(MODE, _config(root), project=Path(".")).run()


def test_a_finished_run_reads_back(tmp_path):
    root = tmp_path / "run-a"
    report = _run(root)
    archive = history.read_run(root)
    assert archive.mode == MODE
    assert archive.best_score == report["best_score"]
    assert archive.seed_score == report["seed_score"]
    assert archive.bound == report["bound"]
    assert archive.candidates
    assert archive.candidates[0]["score"] == report["best_score"]


def test_a_run_without_a_report_still_reads(tmp_path):
    root = tmp_path / "partial"
    _run(root)
    (root / "report.json").unlink()
    archive = history.read_run(root)
    assert archive.candidates
    assert archive.best_score is None, "a partial run has no reported best"


def test_an_empty_directory_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        history.read_run(tmp_path / "nothing")


def test_a_run_directory_that_is_not_one_says_so(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        history.read_run(empty)


def test_the_summary_is_bounded_and_says_when_it_truncated(tmp_path):
    root = tmp_path / "run-a"
    _run(root)
    archive = history.read_run(root, limit=1)
    summary = archive.summary()
    assert summary["candidates_truncated"] is True
    assert len(archive.candidates) == 1
    assert "knob_attribution" in summary
    assert "rejection_codes" in summary


def test_find_runs_returns_siblings_newest_first_and_excludes_one(tmp_path):
    _run(tmp_path / "older")
    _run(tmp_path / "newer")
    both = history.find_runs(tmp_path)
    assert {a.path.name for a in both} == {"older", "newer"}
    without = history.find_runs(tmp_path, exclude=tmp_path / "newer")
    assert [a.path.name for a in without] == ["older"]


def test_find_runs_ignores_a_directory_that_is_not_a_run(tmp_path):
    (tmp_path / "notes").mkdir()
    _run(tmp_path / "real")
    assert [a.path.name for a in history.find_runs(tmp_path)] == ["real"]


def test_the_loop_tells_the_optimizer_what_it_already_tried(tmp_path):
    loop = Loop(MODE, _config(tmp_path / "run"), project=Path("."))
    loop.init()
    loop.forward()
    loop.loss_step()
    loop.backward()
    ctx = loop._context()  # noqa: SLF001 - the context is the subject
    assert ctx.history, "the optimizer was shown nothing it had tried"
    assert ctx.tried, "the optimizer was not told which states exist"
    assert len(ctx.tried) == len(loop.population)
    assert not ctx.history_truncated
    for row in ctx.history:
        assert row["score"] is not None
        assert "edits" in row


def test_the_history_is_bounded_and_says_so(tmp_path):
    loop = Loop(
        MODE,
        LoopConfig(epochs=1, batch_size=3, optimizer="random", seed=5,
                   runs_root=tmp_path / "run", history_limit=1),
        project=Path("."),
    )
    loop.init()
    loop.forward()
    ctx = loop._context()  # noqa: SLF001
    assert len(ctx.history) == 1
    assert ctx.history_truncated is True


def test_a_previous_run_is_offered_as_a_prior(tmp_path):
    _run(tmp_path / "a-previous-run")
    loop = Loop(MODE, _config(tmp_path / "current"), project=Path("."))
    loop.init()
    ctx = loop._context()  # noqa: SLF001
    assert ctx.prior_runs, "a sibling run was not offered as a prior"
    prior = ctx.prior_runs[0]
    assert prior["mode"] == MODE
    assert prior["best_score"] is not None
    assert prior["candidates"] >= 1


def test_the_prompt_carries_the_history_and_the_priors(tmp_path):
    _run(tmp_path / "a-previous-run")
    loop = Loop(MODE, _config(tmp_path / "current"), project=Path("."))
    loop.init()
    loop.forward()
    loop.loss_step()
    loop.backward()
    prompt = loop._context().render()  # noqa: SLF001
    assert "## What this run has already tried" in prompt
    assert "## Previous runs of this mode" in prompt


def test_no_batch_slot_is_silently_lost(tmp_path):
    """Every candidate the optimizer returns is evaluated or refused, never dropped.

    The steps are driven by hand, so no epoch is opened and only the evaluation
    bound ends the loop. By 32 evaluations the random optimizer has proposed a state
    the search already holds, which is what the optimizer's own refusal is for.
    """
    import json as _json

    loop = Loop(
        MODE,
        LoopConfig(epochs=2, batch_size=4, optimizer="random", seed=5,
                   runs_root=tmp_path / "run", max_evaluations=32),
        project=Path("."),
    )
    loop.init()
    counts = []
    while True:
        before = len(loop.population)
        evaluated = loop.forward()
        loop.loss_step()
        loop.backward()
        candidates, rejections = loop.optimizer_step()
        added = loop.step(candidates)
        counts.append((len(candidates), len(added), len(rejections)))
        if not loop.scheduler(evaluated=evaluated, proposed=len(candidates) + len(rejections),
                              accepted=len(candidates), refused=len(rejections)):
            break
    for proposed, added, refused in counts:
        assert added + refused >= proposed, (
            f"{proposed} proposed, {added} added, {refused} refused: a slot vanished"
        )
    assert not [r for r in loop.rejections if r.attempt == 0], (
        "a candidate reached step() that the optimizer's loop should have refused, "
        "which means the slot was lost one layer too late"
    )
    # and every refusal names a rule: a validator rule, or an edit refusal
    from refactory_dsh.nn import RULES

    report = loop.report()
    for code in report["rejection_codes"]:
        assert code in RULES or code.startswith(("I:", "E-")), (
            f"{code} is not a documented rule, so its count is not actionable"
        )


def test_a_strategy_that_proposes_only_duplicates_makes_no_progress_and_says_so(tmp_path):
    """The failure mode the duplicate refusal exists to prevent."""
    from refactory_dsh.optim import Optimizer, Proposal

    class Stuck(Optimizer):
        name = "stuck"

        def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
            return [
                Proposal(
                    edit={"op": "set_knob", "name": "order", "value": "asc"},
                    parent=parents[0].id,
                )
            ] * n

    loop = Loop(
        MODE,
        LoopConfig(epochs=2, batch_size=3, optimizer="random", seed=5,
                   runs_root=tmp_path / "run"),
        project=Path("."),
        optimizer=Stuck(k=2),
    )
    report = loop.run()
    assert "E-duplicate" in report["rejection_codes"], (
        "a strategy proposing the same state repeatedly must be told why nothing "
        f"happened; codes were {report['rejection_codes']}"
    )


@pytest.mark.parametrize("bound, basis", [
    ("seed scored on train, val and test", "test: the final evaluation"),
    ("stopped before the holdout audit", "holdout"),
    ("held-out part exhausted", "val: the validation gate's score"),
])
def test_a_prior_run_shows_the_optimizer_no_test_or_holdout_information(tmp_path, bound, basis):
    """A previous run's summary states train facts only: a best score measured on any
    other part is replaced by the run's train score, and a bound that names a held-out
    part is left out. The request's "Previous runs" table carries neither."""
    prior = tmp_path / "a-previous-run"
    _run(prior)
    report = json.loads((prior / "report.json").read_text())
    report.update(bound=bound, best_score=0.987654, best_score_basis=basis,
                  best_train_score=0.4321, holdout={"test": 0.876543})
    (prior / "report.json").write_text(json.dumps(report))
    [summary] = history.prior_summaries(tmp_path, exclude=tmp_path / "current")
    assert summary["bound"] == "" and summary["best_score"] == 0.4321
    loop = Loop(MODE, _config(tmp_path / "current"), project=Path("."))
    loop.init()
    prompt = loop._context().render()  # noqa: SLF001
    section = prompt[prompt.index("## Previous runs of this mode"):]
    section = section[:section.index("\n## ", 1)]
    rows = [line for line in section.splitlines() if line.startswith("| ") and "---" not in line]
    cells = [cell for row in rows[1:] for cell in row.strip("|").split("|")[1:]]
    shown = " ".join(cells)
    assert "0.4321" in shown, section
    assert "0.987654" not in shown and "0.876543" not in shown
    assert not re.search(r"(?i)test|holdout|held.out", shown), shown


def test_a_train_bound_and_score_are_kept():
    archive = history.RunArchive(path=Path("p"), bound="epochs", best_score=0.5)
    assert archive.summary()["bound"] == "epochs"
    assert history._train_only("evaluations") == "evaluations"  # noqa: SLF001
    assert history._train_only("the latest contest") == "the latest contest"  # noqa: SLF001
