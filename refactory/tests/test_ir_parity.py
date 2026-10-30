"""IR structural checks over every mode this checkout ships.

Previously this file compared refactory-dsh's IR against remote-factory's v1
primitives as a live oracle.  With remote-factory deprecated, the checks are:

1. Every shipped mode produces a valid IR via ``nn.validate_state``.
2. ``to_dict()`` round-trips cleanly through ``from_dict()``.
3. The IR field snapshot catches unintentional field additions or removals.
"""

from __future__ import annotations

from typing import Any

import pytest

from refactory_dsh import lowering, registry
from refactory_dsh.ir import Workflow
from refactory_dsh.nn import validate_state

from reference_ir import check_field_snapshot


def _shipped_packages() -> list[str]:
    """Every package installed from `refactory_packages/`, not the workspace."""
    return registry.installed_names(None)


def _modes() -> list[tuple[str, dict[str, Any]]]:
    """Every shipped mode as ``(identifier, state)``, built at collection time."""
    corpus: list[tuple[str, dict[str, Any]]] = []
    for package_name in _shipped_packages():
        package = registry.load(package_name)
        for mode_class in package.modes:
            corpus.append((f"{package_name}/{mode_class.name}", mode_class().state_dict()))
    return corpus


CORPUS = _modes()
IDS = [identifier for identifier, _state in CORPUS]


@pytest.mark.parametrize(("identifier", "state"), CORPUS, ids=IDS)
def test_a_shipped_mode_is_valid(identifier, state):
    issues = validate_state(state)
    assert not issues, f"{identifier} has validation issues: {issues}"


@pytest.mark.parametrize(("identifier", "state"), CORPUS, ids=IDS)
def test_a_shipped_mode_round_trips_through_dict(identifier, state):
    workflow = lowering.to_workflow(state, name=identifier)
    serialized = workflow.to_dict()
    reconstructed = Workflow.from_dict(serialized)
    assert reconstructed.to_dict() == serialized, (
        f"{identifier} does not round-trip through to_dict/from_dict"
    )


def test_ir_field_snapshot():
    """A field added or removed from any IR class without updating the snapshot fails."""
    check_field_snapshot()
