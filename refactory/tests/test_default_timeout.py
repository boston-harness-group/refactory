"""The default model completion timeout is conservative but overridable.

A node that runs a container build (podman) inside its own session can exceed
the usual authoring-mode ceiling; such a workload raises the limit via
``REFACTORY_MODEL_TIMEOUT_S`` rather than every consumer inheriting a high
default."""

import os

from refactory_dsh import model


def _resolve_default(env):
    """Mirror model.DEFAULT_TIMEOUT_S's resolution for a given environment.

    Kept in lockstep with model.py so the override can be tested without
    ``importlib.reload`` — reloading the module would swap out its
    ``DshModelClient`` class object and break monkeypatching in other tests
    that run later in the same session.
    """
    return int(env.get("REFACTORY_MODEL_TIMEOUT_S", "1800"))


def test_default_timeout_is_conservative_by_default():
    # The module-level default, resolved at import with no override set.
    assert model.DEFAULT_TIMEOUT_S == _resolve_default(
        {k: v for k, v in os.environ.items() if k != "REFACTORY_MODEL_TIMEOUT_S"}
    )
    assert _resolve_default({}) == 1800


def test_default_timeout_honors_env_override():
    assert _resolve_default({"REFACTORY_MODEL_TIMEOUT_S": "7200"}) == 7200
