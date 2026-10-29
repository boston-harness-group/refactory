"""Real runtime/model tests; --run-integration, optional REFACTORY_SMOKE_PROJECT."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from refactory_dsh import sandbox
from refactory_dsh.model import DshModelClient
from test_sandbox import GOOD, NAME, _stage

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("input_exists", [False, True], ids=["missing-input", "valid-input"])
def test_smoke_enforces_production_agent_inputs(tmp_path, input_exists):
    files = dict(GOOD)
    files["mode.py"] = files["mode.py"].replace(
        "g.add(nn.Node('start', 'op', call=OPS + ':start',"
        " writes=('state.json',), produces=('value',), knobs=('target',)))",
        "g.add(nn.Node('start', 'agent', reads=('input.txt',), writes=('result.txt',),"
        " prompt_template='Write exactly ok followed by a newline to result.txt using a file-writing tool.'))",
    )
    assert files["mode.py"] != GOOD["mode.py"]
    _stage(tmp_path, files)
    workdir = tmp_path / "run"
    workdir.mkdir()
    if input_exists:
        (workdir / "input.txt").write_text("Available input.\n")
    client = DshModelClient(
        project=Path(os.environ.get("REFACTORY_SMOKE_PROJECT", str(tmp_path))),
    )
    result = sandbox.smoke(
        f"{NAME}/counter", workdir, project=tmp_path,
        model=True, model_context=client.context(),
        timeout_s=180 if input_exists else 40,
    )
    if input_exists:
        assert result.ok, result.problem()
        assert result.payload.get("exit_ok") is True, result.payload
        assert result.payload["score"] == 3.0
        assert (workdir / "result.txt").read_text() == "ok\n"
    else:
        # The graph driver may wait for steering after a worker failure. Its
        # persisted failure is authoritative even when smoke reaches its bound.
        status_path = workdir / ".refactory" / "spine-status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        detail = (result.problem() + str(result.payload.get("error", ""))
                  + str(status.get("last_failure", {}).get("error", "")))
        assert not result.timed_out, result.problem()
        assert not result.ok or result.payload.get("exit_ok") is False, result.payload
        assert "required worker input is missing: input.txt" in detail, detail
        assert not (workdir / "result.txt").exists()
