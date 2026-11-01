"""The running example, end to end.

This is the source of truth for "is the implementation working". It walks the
same example the design was written around, offline and deterministically:

    a brief -> create-mode -> a registered, smoke-scoring package
            -> the outer loop optimizes it -> a leaderboard and a structural win

Every issue in `.work/todo.json` is verified against this file staying green, so a
change that breaks the example fails the gate rather than one narrow unit test.
"""

from __future__ import annotations

import fcntl
import json
import shutil
from pathlib import Path

import pytest

from refactory_dsh import nn, registry
from refactory_dsh.create import CreateConfig, create
from refactory_dsh.loop import Loop, LoopConfig
from tests_support import FakeModel, create_answers, good_files, render

CREATED = "example_pack"
TARGET = "circle_packing/adaptive"


@pytest.fixture(autouse=True)
def _clean_created():
    root = registry.workspace_root(Path(".")) / CREATED
    # Serialize access to the shared workspace package directory so that
    # parallel xdist workers do not race on creation and cleanup.
    lock_path = root.parent / ".test_example_lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = open(lock_path, "w")
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    try:
        shutil.rmtree(root, ignore_errors=True)
        yield
        shutil.rmtree(root, ignore_errors=True)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def _config(root: Path, **overrides: object) -> LoopConfig:
    base = dict(epochs=2, batch_size=8, optimizer="random", seed=5, runs_root=root)
    base.update(overrides)
    return LoopConfig(**base)  # type: ignore[arg-type]


# ── half one: a brief becomes a working package ────────────────────────────
def _create_config(**overrides) -> CreateConfig:
    """A create config with the person's strategy approval recorded (`user` gate).

    `single_shot` because the running example's fixture package has no Task binding;
    naming a Task would be the class-of-work default, and would win over it.
    """
    return CreateConfig(
        verdicts={"gate_strategy": "proceed"}, single_shot=True, **overrides
    )


def test_a_brief_becomes_a_registered_package_that_scores(tmp_path):
    model = FakeModel(create_answers(CREATED, good_files(CREATED)))
    result = create(
        _create_config(name=CREATED, brief="count to a target", workdir=tmp_path / "create"),
        client=model,
    )
    assert result.package == CREATED
    assert result.smoke_score is not None
    assert (result.path / "mode.py").exists()
    assert f"{CREATED}/counter" in registry.render_catalogue(Path("."))


def test_the_created_mode_validates_with_all_its_calls_resolved(tmp_path):
    create(
        _create_config(name=CREATED, brief="count", workdir=tmp_path / "create"),
        client=FakeModel(create_answers(CREATED, good_files(CREATED))),
    )
    _package, mode_class = registry.resolve(Path("."), f"{CREATED}/counter")
    mode = mode_class()
    assert nn.validate_state(
        mode.state_dict(), invariants=mode.invariants(), resolve_calls=True
    ) == []


# ── half two: the loop optimizes it, and the report says what happened ─────
def _optimize(mode_id: str, root: Path, **overrides: object) -> dict:
    return Loop(mode_id, _config(root, **overrides), project=Path(".")).run()


def test_the_loop_improves_the_mode_it_is_given(tmp_path):
    report = _optimize(TARGET, tmp_path / "runs")
    assert report["best_score"] > report["seed_score"], (
        "the running example must show the loop finding an improvement"
    )


def test_the_report_has_the_walkthroughs_shape(tmp_path):
    report = _optimize(TARGET, tmp_path / "runs")
    for key in (
        "mode", "optimizer", "bound", "epochs_completed", "evaluations",
        "seed_score", "best_score", "leaderboard", "lineage", "structural_wins",
        "accepted_op_mix", "structural", "rejection_codes", "cost",
        "proposal_outcomes", "epochs_summary",
    ):
        assert key in report, f"the report lost {key}"
    scores = [row["score"] for row in report["leaderboard"]]
    assert scores == sorted(scores, reverse=True)
    assert report["bound"] in {"epochs", "evaluations"}
    assert report["lineage"], "the best candidate must have a lineage"


def test_the_report_is_markdown_with_its_sections(tmp_path):
    root = tmp_path / "runs"
    _optimize(TARGET, root)
    markdown = (root / "report.md").read_text()
    assert markdown.startswith("# Outer loop")
    for section in ("## Leaderboard", "## Refusals by rule", "## Accepted edits by operation"):
        assert section in markdown


def test_a_structural_candidate_wins_the_running_example(tmp_path):
    """The seed pipeline wastes its budget; the fix is a node, not a parameter."""
    report = _optimize(TARGET, tmp_path / "runs")
    assert report["structural_wins"], "no structural candidate scored"
    assert all(row["exit_ok"] for row in report["structural_wins"])
    assert report["structural"]["structural_yield"] != "n/a"


def test_every_evaluated_candidate_validated_first(tmp_path):
    """The example's central guarantee: nothing invalid ever costs a run.

    The steps are driven by hand, so no epoch is opened and only the evaluation
    bound ends the loop: 12 evaluations already include proposals that validation
    must refuse.
    """
    loop = Loop(TARGET, _config(tmp_path / "runs", max_evaluations=12), project=Path("."))
    loop.init()
    while True:
        evaluated = loop.forward()
        loop.loss_step()
        loop.backward()
        candidates, rejections = loop.optimizer_step()
        loop.step(candidates)
        if not loop.scheduler(evaluated=evaluated, proposed=len(candidates) + len(rejections),
                              accepted=len(candidates), refused=len(rejections)):
            break
    for candidate in loop.population:
        assert nn.validate_state(candidate.state) == [], candidate.id


def test_the_whole_example_runs_offline_and_reproducibly(tmp_path):
    """Same seed, same result: the example is a test, not a demo."""
    first = _optimize(TARGET, tmp_path / "a")
    second = _optimize(TARGET, tmp_path / "b")
    assert first["best_score"] == second["best_score"]
    assert [c["id"] for c in first["leaderboard"]] == [c["id"] for c in second["leaderboard"]]


def test_the_run_leaves_the_artifacts_a_person_needs(tmp_path):
    root = tmp_path / "runs"
    _optimize(TARGET, root)
    for name in ("config.json", "loss.json", "gradient.json", "report.json", "report.md", "state.json"):
        assert (root / name).exists(), f"the run did not leave {name}"
    assert (root / "runs").is_dir(), "per-candidate run directories are missing"


def test_the_created_package_can_be_optimized_too(tmp_path):
    """The full chain: a brief, a package, and then a search over it."""
    create(
        _create_config(name=CREATED, brief="count", workdir=tmp_path / "create"),
        client=FakeModel(create_answers(CREATED, good_files(CREATED))),
    )
    report = _optimize(f"{CREATED}/counter", tmp_path / "runs")
    assert report["best_score"] >= report["seed_score"]
    assert report["leaderboard"]
    assert json.dumps(report, default=str)
