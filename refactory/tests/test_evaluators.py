"""The evaluator seam: the loop's loss, and what makes the two agree.

The audit that produced this file: two of the loop's three extension points had a
base class and a registry, and the third -- how a mode is run to get the score the
loop learns from -- was a string compared against a literal inside `_evaluate`. It
worked for exactly two and neither could be tested against a shared contract.

These assert the seam rather than the two implementations, because the point of a
seam is what can be plugged into it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from refactory_dsh import evaluators
from refactory_dsh.evaluators import Evaluator, RunResult, WalkerEvaluator
from refactory_dsh.loop import Loop, LoopConfig

ROOT = Path(__file__).resolve().parent.parent.parent
MODE = "circle_packing/adaptive"


def _config(tmp_path: Path, **overrides: object) -> LoopConfig:
    # `random` rather than the default: these test how a mode is *run*, and the
    # model-backed proposer would need a client for reasons that have nothing to
    # do with the evaluator.
    base = {
        "epochs": 1,
        "batch_size": 1,
        "seed": 5,
        "optimizer": "random",
        "runs_root": tmp_path / "run",
    }
    base.update(overrides)
    return LoopConfig(**base)  # type: ignore[arg-type]


# ── the registry ───────────────────────────────────────────────────────────
def test_the_two_that_ship_are_registered_by_name():
    assert set(evaluators.names()) >= {"walker", "dsh"}


def test_an_unknown_evaluator_is_refused_loudly(tmp_path):
    """A silent fallback would make the report lie about how candidates were run."""
    with pytest.raises(evaluators.EvaluatorError, match="unknown evaluator"):
        evaluators.create("nonsense", config=_config(tmp_path))


def test_the_loop_refuses_an_unknown_evaluator(tmp_path):
    with pytest.raises(Exception, match="unknown evaluator"):
        Loop(MODE, _config(tmp_path, evaluator="nonsense"), project=ROOT)


def test_each_evaluator_says_whether_it_leaves_this_process():
    """The loop should not have to match on a name to know which one is slow."""
    walker = evaluators.create("walker", config=_config(Path("/tmp")))
    dsh = evaluators.create("dsh", config=_config(Path("/tmp")))
    assert walker.out_of_process is False
    assert dsh.out_of_process is True


# ── the seam actually takes a new implementation ───────────────────────────
def test_a_third_evaluator_needs_no_edit_to_the_loop(tmp_path):
    """The property the abstraction exists for.

    A new evaluator is registered and used without touching `loop.py`. If this
    ever needs a change to the loop, the seam has stopped being a seam.
    """

    class Fixed(Evaluator):
        name = "fixed"
        seen: list[str] = []

        @classmethod
        def from_settings(cls, *, config, model=None, project=None):
            return cls()

        def evaluate(self, mode, state, workdir):
            Fixed.seen.append(mode.name)
            return RunResult(score=0.5, facts={"canned": True}, exit_ok=True)

    evaluators.register("fixed", Fixed)
    try:
        config = _config(tmp_path, evaluator="fixed", epochs=1, batch_size=1)
        loop = Loop(MODE, config, project=ROOT)
        assert isinstance(loop.evaluator, Fixed), "the loop built the wrong evaluator"
        report = loop.run()
        assert Fixed.seen, "the registered evaluator was never called"
        # The loop's own size cost is taken off the number the evaluator returned, so
        # the raw score is what says the number came from the seam.
        assert report["raw_best_score"] == 0.5, (
            "the score did not come from the evaluator, so the loop is not reading "
            f"the seam: {report}"
        )
        assert report["best_score"] == pytest.approx(
            0.5 - report["scoring"]["parsimony"] * report["best"]["nodes"]
        )
    finally:
        evaluators.REGISTRY.pop("fixed", None)


def test_the_loop_does_not_branch_on_the_evaluator_name():
    """The branch was the defect. A seam that is still an `if` is not a seam."""
    import inspect

    source = inspect.getsource(Loop._measure)
    assert "evaluator ==" not in source, (
        "`_evaluate` branches on the evaluator name again; the choice belongs in "
        "the registry"
    )
    assert "evaluator.evaluate(" in source, "`_measure` does not call the seam"


# ── the two must agree ─────────────────────────────────────────────────────
def test_the_walker_scores_a_mode(tmp_path):
    """The fast one, on its own, with no loop around it."""
    from refactory_dsh import registry

    _, mode_class = registry.resolve(ROOT, MODE)
    mode = mode_class()
    walker = WalkerEvaluator(model=None, project=ROOT)
    result = walker.evaluate(mode, mode.state_dict(), tmp_path / "run")
    assert result.exit_ok, result.error
    assert result.score > 0.0
    assert result.duration_s >= 0.0


@pytest.mark.integration
def test_both_evaluators_agree_on_the_same_mode(tmp_path):
    """The equivalence property, asserted on the seam rather than inside the loop.

    Both are asked the same question — run this mode, tell me what it scored — and
    they must give the same answer, or the loop is searching a fiction.
    """
    from refactory_dsh import registry

    _, mode_class = registry.resolve(ROOT, MODE)
    mode = mode_class()
    state = mode.state_dict()

    walker = WalkerEvaluator(model=None, project=ROOT)
    dsh = evaluators.create("dsh", config=_config(tmp_path))
    fast = walker.evaluate(mode, state, tmp_path / "fast")
    shipped = dsh.evaluate(mode, state, tmp_path / "shipped")

    assert shipped.exit_ok, shipped.error
    assert fast.score == shipped.score, (
        f"the two evaluators disagree about the same mode: walker {fast.score}, "
        f"dsh {shipped.score}"
    )


def test_omit_steering_is_public_on_walker_evaluator_and_saved_for_resume(tmp_path, monkeypatch):
    """The outer-loop setting reaches mode.evaluate and the durable inner run config."""
    import json

    from refactory_dsh import registry

    config = _config(tmp_path, omit_steering=True)
    restored = LoopConfig.from_dict(config.to_dict())
    assert restored.omit_steering is True
    _, mode_class = registry.resolve(ROOT, MODE)
    mode = mode_class()
    original = mode.evaluate
    seen = {}

    def observe(workdir, **kwargs):
        seen.update(kwargs)
        return original(workdir, **kwargs)

    monkeypatch.setattr(mode, "evaluate", observe)
    evaluator = WalkerEvaluator.from_settings(config=config, model=None, project=ROOT)
    result = evaluator.evaluate(mode, mode.state_dict(), tmp_path / "inner")
    assert result.exit_ok, result.error
    assert seen["omit_steering"] is True
    saved = json.loads((tmp_path / "inner/.refactory/run-config.json").read_text())
    assert saved["omit_steering"] is True


def test_omit_steering_false_overrides_saved_true_on_reused_workdir(tmp_path, monkeypatch):
    """An explicit false policy replaces a prior true value for a reused candidate."""
    import json
    from refactory_dsh import registry

    _, mode_class = registry.resolve(ROOT, MODE)
    mode = mode_class()
    original = mode.evaluate
    observed = []

    def observe(workdir, **kwargs):
        observed.append(kwargs["omit_steering"])
        return original(workdir, **kwargs)

    monkeypatch.setattr(mode, "evaluate", observe)
    workdir = tmp_path / "reused"
    state = mode.state_dict()
    enabled = WalkerEvaluator.from_settings(
        config=_config(tmp_path, omit_steering=True), model=None, project=ROOT
    )
    disabled = WalkerEvaluator.from_settings(
        config=_config(tmp_path, omit_steering=False), model=None, project=ROOT
    )
    assert enabled.evaluate(mode, state, workdir).exit_ok
    assert json.loads((workdir / ".refactory/run-config.json").read_text())["omit_steering"] is True
    assert disabled.evaluate(mode, state, workdir).exit_ok
    assert observed == [True, False]
    assert json.loads((workdir / ".refactory/run-config.json").read_text())["omit_steering"] is False


def test_persist_steering_option_does_not_erase_malformed_run_config(tmp_path):
    from refactory_dsh.evaluators import _persist_steering_option

    path = tmp_path / ".refactory/run-config.json"
    path.parent.mkdir(parents=True)
    original = "{not json"
    path.write_text(original)
    with pytest.raises(ValueError, match="cannot read run configuration"):
        _persist_steering_option(tmp_path, False)
    assert path.read_text() == original
