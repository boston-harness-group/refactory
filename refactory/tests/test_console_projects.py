"""The GUI's projects endpoint: the console lists the same projects as ``refactory projects`` (#351).

This test fails without the ``list_projects`` function because it would not exist.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from refactory_dsh import projects
from refactory_dsh.console import list_projects
from tests_support import git_repo


def test_gui_projects_list(tmp_path):
    """A registered project appears in the list_projects() result."""
    repo = git_repo(tmp_path / "myrepo")
    projects.register(repo, kind="existing")
    result = list_projects()
    names = [p["name"] for p in result]
    paths = [p["path"] for p in result]
    assert "myrepo" in names or str(repo) in paths
    assert any(p["path"] == str(repo) for p in result)


def _activity_cli(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "refactory_dsh.console", "status", *argv],
                          capture_output=True, text=True)


def test_activity_options_are_accepted_clamped_and_refused(tmp_path):
    """T-4: ``--activity-limit`` clamps at 1000, and 0/-1/non-integer exit 2.

    Validation happens before any run directory is read, so the refusals hold
    even when the run does not exist.
    """
    run = tmp_path / "run"
    (run / ".refactory").mkdir(parents=True)
    (run / ".refactory/events.jsonl").write_text(
        "".join(json.dumps({"schema_version": 1, "kind": "step", "at": "2026-10-26T09:00:00+00:00", "n": i}) + "\n"
                for i in range(3)))
    servable = json.loads(_activity_cli("--run", str(run), "--activity-limit", "5000").stdout)["activity"]
    assert servable["page_size"] == 1000
    assert servable["events"][0]["n"] == 0

    accepted = json.loads(_activity_cli("--run", str(run), "--activity-limit", "1",
                                        "--activity-after", "2").stdout)["activity"]
    assert (accepted["after"], accepted["cursor"], len(accepted["events"])) == (2, 3, 1)

    for refusal in (("--activity-limit", "0"), ("--activity-limit", "-1"), ("--activity-after", "-1")):
        result = _activity_cli("--run", str(run), *refusal)
        assert result.returncode == 2, (refusal, result.stdout)
    refused = json.loads(_activity_cli("--run", str(run), "--activity-after", "-1").stdout)
    assert "--activity-after" in refused["error"] and "nonnegative" in refused["error"]
    refused = json.loads(_activity_cli("--run", str(run), "--activity-limit", "0").stdout)
    assert "--activity-limit" in refused["error"]

    # A non-integer is refused by the parser, as `--after` is (exit 2, no answer).
    assert _activity_cli("--run", str(run), "--activity-limit", "abc").returncode == 2
    # Neither option is required, and both are on the status sub-parser.
    listed = _activity_cli("--help").stdout
    assert "--activity-after" in listed and "--activity-limit" in listed
