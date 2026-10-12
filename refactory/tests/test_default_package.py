"""The meta-modes as a package, and the exceptions that package needs.

Issue #29. Putting `create`, `edit` and `design` in a package rather than leaving
them as three loose entry points is what makes them modes like any other — and it
immediately exposed a latent bug: `CreateMode`'s `survey` node wrote
`survey.json` without declaring it, and the executor skips its write check for a
node that declares no writes, so the graph ran and had never been validated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from refactory_dsh import nn, registry
from refactory_dsh.loop import Loop, LoopConfig

PACKAGE = "default_pack"
MODES = ("create", "edit", "design", "create_fast", "edit_fast", "design_fast", "create_fidelity")


def _mode(identifier: str):
    return registry.resolve(Path("."), identifier)[1]()


# ── it is a package like any other ─────────────────────────────────────────


def test_the_meta_modes_are_discovered_by_name():
    """They are modes, not entry points that happen to have a graph."""
    package = registry.load(PACKAGE, Path("."))
    assert sorted(mode.name for mode in package.modes) == sorted(MODES)


def test_every_meta_mode_validates_as_a_mode():
    """The check a loose entry point never got. This is what found the bug."""
    for name in MODES:
        mode = _mode(f"{PACKAGE}/{name}")
        issues = nn.validate_state(
            mode.state_dict(), invariants=mode.invariants(), resolve_calls=True
        )
        assert issues == [], f"{PACKAGE}/{name}: {[str(issue) for issue in issues]}"


def test_the_survey_node_declares_the_file_the_strategy_node_reads():
    """The specific latent bug, held so it cannot come back.

    `synthesize_strategy` reads `survey.json`; `survey` wrote it without declaring it, and the
    executor skips the write check when a node declares no writes — so nothing
    noticed until the graph was validated.
    """
    graph = _mode(f"{PACKAGE}/create").graph()
    assert "survey.json" in graph.nodes["synthesize_strategy"].reads
    assert "survey.json" in graph.nodes["survey"].writes, (
        "the survey node writes a file it does not declare"
    )


def test_the_package_names_the_operations_it_uses():
    package = registry.load(PACKAGE, Path("."))
    assert package.ops, "the package names no ops, so its nodes could not resolve"
    for module in package.ops:
        __import__(module)


# ── the exceptions the package needs, stated ───────────────────────────────


@pytest.mark.parametrize("name", ["create", "design", "create_fidelity"])
def test_an_unscored_mode_declares_why(name):
    """Unscored on purpose is a declaration, not an inference.

    `is_scored()` is derived from whether `score()` is overridden, so a mode that
    produces no signal looks identical to a mode whose author forgot. The reason
    is what tells the registry, the sandbox and a reader which it is.
    """
    mode = _mode(f"{PACKAGE}/{name}")
    assert not mode.is_scored(), f"{name} was expected to be unscored"
    assert mode.unscored_reason.strip(), f"{name} is unscored and does not say why"


def test_a_scored_mode_needs_no_reason():
    """The exception must not become the rule: a scored mode declares nothing."""
    scored = _mode("circle_packing/adaptive")
    assert scored.is_scored()
    assert scored.unscored_reason == ""


def test_the_loop_refuses_an_unscored_mode_as_a_search_target(tmp_path):
    """You cannot hill-climb on "did it work", so this must fail loudly."""
    with pytest.raises(Exception) as raised:
        Loop(
            f"{PACKAGE}/create",
            LoopConfig(epochs=1, batch_size=2, optimizer="random", seed=1,
                       runs_root=tmp_path / "runs"),
            project=Path("."),
        ).run()
    assert "score()" in str(raised.value), (
        f"the refusal does not say the mode produces no signal: {raised.value}"
    )


def test_the_package_documents_its_own_exceptions():
    """A reader meets the exceptions where the package is defined."""
    import importlib

    doc = importlib.import_module(f"refactory_packages.{PACKAGE}").__doc__ or ""
    for promised in ("unscored", "ops", "model"):
        assert promised in doc.lower(), (
            f"the package docstring does not state the {promised!r} exception"
        )


def test_an_entry_point_and_a_mode_agree(tmp_path):
    """`python -m refactory_dsh.create` and `default_pack/create` are one thing."""
    from refactory_dsh.create import CreateMode

    assert CreateMode.name == "create"
    assert _mode(f"{PACKAGE}/create").__class__ is CreateMode
