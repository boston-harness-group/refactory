"""The default model completion timeout is a constant, not an environment setting.

A workload whose nodes run longer than the authoring modes' ceiling raises it where
the run is configured: a node's own ``timeout_s``, or ``--model-timeout`` for every
agent step. ``REFACTORY_MODEL_TIMEOUT_S`` no longer exists."""

import os
import subprocess
import sys

from refactory_dsh import model


def test_default_timeout_is_conservative():
    assert model.DEFAULT_TIMEOUT_S == 1800


def test_the_removed_environment_override_changes_nothing():
    env = {**os.environ, "REFACTORY_MODEL_TIMEOUT_S": "7200"}
    out = subprocess.run(
        [sys.executable, "-c", "from refactory_dsh import model; print(model.DEFAULT_TIMEOUT_S)"],
        env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "1800"
