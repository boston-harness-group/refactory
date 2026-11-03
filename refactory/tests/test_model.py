"""DshModelClient unit tests.

Tests for max_completion_tokens route propagation (G31).
"""

from __future__ import annotations

from pathlib import Path


# ── G31: max_completion_tokens route tests ────────────────────────────────


def test_max_completion_tokens_appears_in_route():
    """When _max_completion_tokens is set, route() includes it."""
    from refactory_dsh.model import DshModelClient

    client = DshModelClient.__new__(DshModelClient)
    client._worker_backend = "dsh"
    client._provider = "anthropic"
    client._model = "claude-opus-4-6"
    client._effort = "medium"
    client._permission_mode = "skip"
    client._max_completion_tokens = "16384"
    route = client.route()
    assert route["max_completion_tokens"] == "16384"


def test_max_completion_tokens_omitted_when_unset():
    """When _max_completion_tokens is empty, route() does not include it."""
    from refactory_dsh.model import DshModelClient

    client = DshModelClient.__new__(DshModelClient)
    client._worker_backend = "dsh"
    client._provider = "anthropic"
    client._model = "claude-opus-4-6"
    client._effort = "medium"
    client._permission_mode = "skip"
    client._max_completion_tokens = ""
    route = client.route()
    assert "max_completion_tokens" not in route


def test_for_node_propagates_max_completion_tokens():
    """for_node propagates max_completion_tokens from node_routes."""
    from refactory_dsh.model import DshModelClient

    client = DshModelClient.__new__(DshModelClient)
    client._worker_backend = "dsh"
    client._provider = "anthropic"
    client._model = "claude-opus-4-6"
    client._effort = ""
    client._permission_mode = "skip"
    client._max_completion_tokens = ""
    client._role_routes = {}
    client._node_routes = {"solve": {"max_completion_tokens": "16384"}}
    client._node_id = ""
    client._role = "worker"
    client._dsh_home = Path(".")
    node_client = client.for_node("solve", role="solver")
    assert node_client.route()["max_completion_tokens"] == "16384"
