"""`REFACTORY_SPAWN_LOG` counts the processes refactory starts, and nothing is counted without it."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

from refactory_dsh import spawn_count

PROGRAM = ("import subprocess, refactory_dsh\n"
           "subprocess.run(['git', '--version'], capture_output=True)\n"
           "subprocess.run(['true'])\n")


def _run(env: dict[str, str]) -> None:
    subprocess.run([sys.executable, "-c", PROGRAM], check=True, env=env)


def test_a_refactory_process_logs_its_start_and_each_launch(tmp_path):
    log = tmp_path / "spawns.jsonl"
    _run({**os.environ, spawn_count.ENV: str(log)})

    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert [line["kind"] for line in lines] == ["start", "launch", "launch"]
    assert spawn_count.summary(log) == {"launches": 2, "by_program": {"git": 1, "true": 1},
                                        "starts": {"python": 1}}


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_node_processes_it_starts_are_counted_too(tmp_path):
    log = tmp_path / "spawns.jsonl"
    script = ("const cp = require('node:child_process');"
              "cp.spawnSync('true'); cp.execFileSync('true'); cp.spawn('true').on('exit', () => {});")
    program = f"import subprocess, refactory_dsh\nsubprocess.run(['node', '-e', {script!r}], check=True)\n"
    subprocess.run([sys.executable, "-c", program], check=True, env={**os.environ, spawn_count.ENV: str(log)})

    found = spawn_count.summary(log)
    assert found["starts"] == {"python": 1, "node": 1}
    assert found["by_program"] == {"node": 1, "true": 3}


def test_unset_nothing_is_counted_or_changed(tmp_path):
    env = {name: value for name, value in os.environ.items() if name != spawn_count.ENV}
    result = subprocess.run([sys.executable, "-c", "import os, sys, refactory_dsh\n"
                             "print(os.environ.get('NODE_OPTIONS', ''))\n"
                             "print('refactory_dsh.spawn_count' in sys.modules)"],
                            check=True, capture_output=True, text=True, env=env)

    assert result.stdout.splitlines()[-1] == "False"
    assert "spawn_count" not in result.stdout
