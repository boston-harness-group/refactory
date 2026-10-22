"""Our IR against the reference's, over every mode this checkout ships.

`refactory_dsh.ir` is a copy, and a copy drifts. For every mode of every package
under `refactory_packages/`, one lowering runs twice -- once building our classes,
once building the reference's, with only the IR classes swapped -- and the two
`to_dict()` results must serialize to the same bytes. The reference's NetworkX
validator must also accept the graph our copy serialized.

The reference is a test-time oracle installed from the non-default `reference`
group, so every test here skips with a reason when `factory` is not importable, and
the `toolchain` marker makes `--toolchain-gate` report that skip as unverified
rather than as a pass.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from refactory_dsh import lowering, registry

from reference_ir import (
    check_extension_coverage,
    reference_lowering,
    reference_workflow,
    strip_extension_keys,
)

pytestmark = [pytest.mark.toolchain("factory")]


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


def _serialized(workflow: Any) -> bytes:
    """The shared-vocabulary bytes of a workflow, for byte-parity comparison."""
    return json.dumps(
        strip_extension_keys(workflow.to_dict()), default=str
    ).encode()


#: The shipped modes the reference validator refuses, with the issue it reports.
#:
#: These are limits of the v1 validator or representation gaps, not defects in the
#: lowering: the byte-parity test runs the same lowering against the reference's own
#: classes and shows the two graphs serialize identically.
#:
#: - `sympy_pack/exact` is a read-modify-write loop. `derive` reads and writes
#:   `state.json`, and `_validate_data_dependencies` unions only the writes of a
#:   node's `nx.ancestors`, so a node's own writes never count.
#: - `java_build_pack/build` has a DataNode with the same `subgraph_entry`/
#:   `subgraph_exit` gap as `dataset_probe/dataset`. Additionally, `finish`
#:   reads `build.sh` written inside the subgraph, which the reference validator
#:   cannot trace through the stripped structure.
#: - `dataset_probe/dataset` uses a DataNode with `subgraph_entry`/`subgraph_exit`
#:   fields that exist in v2's IR but not in v1's pinned DataNode (v1 refactored
#:   DataNode to use real edges in commit 48ab38b7). Stripping these extension
#:   fields means the reference validator cannot see the subgraph structure, so it
#:   reports the entry node as unreachable and the DataNode as having no JoinNode.
#: - `loop_mode/loop` uses a gate with a custom forward condition label. The v1
#:   validator requires PROCEED, unconditional, or RELOOP and does not recognize
#:   arbitrary condition labels as forward edges.
#:
#: Our `nn.validate_state` accepts all four and both runtimes execute them. A mode
#: outside this map being refused fails the test, and a mapped mode being
#: accepted fails too, so the exceptions cannot drift unnoticed.
REFERENCE_REFUSALS: dict[str, str | list[str]] = {
    "sympy_pack/exact": "node 'derive' reads {'state.json'} but no predecessor writes them",
    "java_build_pack/build": [
        "node 'read_item' is unreachable from start_node",
        "node 'write_build' is unreachable from start_node",
        "node 'finish' reads {'build.sh'} but no predecessor writes them",
        "DataNode 'items' has no downstream JoinNode — branches have no convergence point",
    ],
    "dataset_probe/dataset": [
        "node 'work' is unreachable from start_node",
        "DataNode 'data' has no downstream JoinNode — branches have no convergence point",
    ],
    "loop_mode/loop": (
        "GateNode 'again' has outgoing edges but no PROCEED, unconditional, "
        "or RELOOP edge — execution will stall"
    ),
}


@pytest.mark.parametrize(("identifier", "state"), CORPUS, ids=IDS)
def test_a_shipped_mode_serializes_byte_identically_to_the_reference(identifier, state):
    ours = lowering.to_workflow(state, name=identifier)
    reference = reference_lowering(state, name=identifier)
    assert _serialized(reference) == _serialized(ours), (
        f"{identifier} serializes differently from the reference IR"
    )


@pytest.mark.parametrize(("identifier", "state"), CORPUS, ids=IDS)
def test_the_reference_validator_accepts_a_shipped_mode(identifier, state):
    ours = lowering.to_workflow(state, name=identifier)
    issues = reference_workflow(ours).validate_graph()
    recorded = REFERENCE_REFUSALS.get(identifier)
    if recorded is None:
        assert issues == [], f"the reference validator refused {identifier}"
        return
    expected = [recorded] if isinstance(recorded, str) else list(recorded)
    assert issues, (
        f"the reference validator now accepts {identifier}; remove it from "
        "REFERENCE_REFUSALS"
    )
    assert issues == expected, (
        f"{identifier} is refused for a different reason than the recorded one"
    )


def test_every_local_only_field_is_declared_as_an_extension():
    """A field added to the IR and not named in EXTENSION_FIELDS fails here.

    This is the parity guard: the reference cannot catch a field it does not
    know about, so every local-only field must be declared as an extension or
    actually exist in the reference.  The reverse is also checked: a field
    declared as an extension that the reference now has is stale.
    """
    check_extension_coverage()
