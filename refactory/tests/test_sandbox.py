"""Running code we did not write.

The sandbox issue: create-mode imports and runs model-written Python, and nothing
contained it. `check_package` imported it into the driver process, and the smoke run
executed in-process too.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from refactory_dsh import registry, sandbox
from refactory_dsh.paths import venv_bin

NAME = "sandboxed_pkg"

GOOD = {
    "__init__.py": (
        "from refactory_dsh.nn import Package\n"
        "from .mode import Counter\n\n"
        "PACKAGE = Package(name='sandboxed_pkg', modes=(Counter,),"
        " ops=('refactory_packages.sandboxed_pkg.ops',))\n"
    ),
    "ops.py": (
        "def start(ctx):\n"
        "    ctx.write_json('state.json', {'value': 0})\n"
        "    return {'value': 0}\n\n"
        "def finish(ctx):\n"
        "    ctx.write_json('best.json', {'best': ctx.knob('target', 3)})\n"
        "    return {'best': ctx.knob('target', 3)}\n"
    ),
    "mode.py": (
        "from refactory_dsh import nn\n"
        "OPS = 'refactory_packages.sandboxed_pkg.ops'\n\n"
        "class Counter(nn.Mode):\n"
        "    name = 'counter'\n"
        "    @classmethod\n"
        "    def declare(cls):\n"
        "        return {'target': nn.Knob(3, 'int', lo=1, hi=9)}\n"
        "    def build(self):\n"
        "        g = nn.Graph()\n"
        "        g.add(nn.Node('start', 'op', call=OPS + ':start',"
        " writes=('state.json',), produces=('value',), knobs=('target',)))\n"
        "        g.add(nn.Node('finish', 'op', call=OPS + ':finish',"
        " writes=('best.json',), produces=('best',)))\n"
        "        g.connect('start', 'finish')\n"
        "        return g\n"
        "    def score(self, run):\n"
        "        return float(run.facts['best'])\n"
    ),
}


def _stage(tmp_path: Path, files: dict[str, str], name: str = NAME) -> Path:
    root = registry.workspace_root(tmp_path) / name
    root.mkdir(parents=True, exist_ok=True)
    for filename, body in files.items():
        (root / filename).write_text(body.replace("sandboxed_pkg", name))
    registry.install_workspace(tmp_path)
    return root


# ── validation without execution ───────────────────────────────────────────
def test_a_well_formed_package_passes_the_ast_check(tmp_path):
    _stage(tmp_path, GOOD)
    assert sandbox.ast_check(NAME, tmp_path) == []


def test_the_ast_check_catches_a_file_that_does_not_parse(tmp_path):
    files = dict(GOOD)
    files["mode.py"] = "class Counter(nn.Mode:\n    pass\n"
    _stage(tmp_path, files)
    problems = sandbox.ast_check(NAME, tmp_path)
    assert any("does not parse" in problem for problem in problems)


def test_the_ast_check_catches_a_missing_manifest(tmp_path):
    _stage(tmp_path, {"__init__.py": "x = 1\n", "mode.py": "y = 2\n"})
    problems = sandbox.ast_check(NAME, tmp_path)
    assert any("PACKAGE" in problem for problem in problems)


def test_the_ast_check_catches_a_mode_with_no_signal(tmp_path):
    files = dict(GOOD)
    files["mode.py"] = files["mode.py"].replace("    def score(self, run):\n", "    def _score(self, run):\n")
    _stage(tmp_path, files)
    problems = sandbox.ast_check(NAME, tmp_path)
    assert any("no score()" in problem for problem in problems), problems


def test_the_ast_check_refuses_a_package_that_reaches_for_eval(tmp_path):
    files = dict(GOOD)
    files["ops.py"] = files["ops.py"] + "\n\ndef sneaky(ctx):\n    return eval('1')\n"
    _stage(tmp_path, files)
    problems = sandbox.ast_check(NAME, tmp_path)
    assert any("eval()" in problem for problem in problems), problems


def test_the_ast_check_does_not_execute_anything(tmp_path):
    """The whole point: a package that would blow up on import still parses."""
    files = dict(GOOD)
    files["mode.py"] = "raise RuntimeError('would explode on import')\n" + files["mode.py"]
    _stage(tmp_path, files)
    problems = sandbox.ast_check(NAME, tmp_path)
    assert all("explode" not in problem for problem in problems), (
        "the AST check imported the package"
    )


def test_the_ast_check_says_so_when_the_package_is_missing(tmp_path):
    assert "no package" in sandbox.ast_check("nowhere", tmp_path)[0]


# ── the environment ────────────────────────────────────────────────────────
def test_the_environment_is_an_allowlist_with_secrets_dropped():
    scrubbed = sandbox.scrub({
        "PATH": "/usr/bin", "HOME": "/home/x", "DEEPSEEK_API_KEY": "sk-secret",
        "AWS_SECRET_ACCESS_KEY": "shh", "MY_TOKEN": "t", "SOMETHING_RANDOM": "v",
    })
    assert "/usr/bin" in scrubbed["PATH"], "the caller's PATH is kept"
    assert scrubbed["PATH"].startswith(str(venv_bin())), (
        "the project's own python comes first, because a check means the interpreter "
        "that has the project installed"
    )
    assert "DEEPSEEK_API_KEY" not in scrubbed
    assert "AWS_SECRET_ACCESS_KEY" not in scrubbed
    assert "MY_TOKEN" not in scrubbed
    assert "SOMETHING_RANDOM" not in scrubbed, "an allowlist, not a denylist"


def test_a_sandboxed_run_does_not_see_the_real_credentials(tmp_path):
    result = sandbox.run_sandboxed(
        "import json, os; print(json.dumps("
        "{'key': 'DEEPSEEK_API_KEY' in os.environ, 'home': os.environ.get('HOME')}))",
        workdir=tmp_path / "run",
        env={**os.environ, "DEEPSEEK_API_KEY": "sk-should-not-be-visible"},
    )
    assert result.ok, result.stderr
    assert result.payload["key"] is False
    assert result.payload["home"] == str(tmp_path / "run"), (
        "HOME is redirected so a run cannot write into the real home"
    )


# ── execution under limits ─────────────────────────────────────────────────
def test_a_run_that_exceeds_its_time_limit_is_killed(tmp_path):
    result = sandbox.run_sandboxed(
        "import time; time.sleep(30)", workdir=tmp_path / "run", timeout_s=2
    )
    assert not result.ok and result.timed_out
    assert "time limit" in result.problem()


def test_a_failing_run_reports_the_last_line_of_its_error(tmp_path):
    result = sandbox.run_sandboxed(
        "raise SystemExit('the reason')", workdir=tmp_path / "run"
    )
    assert not result.ok
    assert "the reason" in result.problem() or result.returncode != 0


def test_a_sandboxed_run_returns_its_json_payload(tmp_path):
    result = sandbox.run_sandboxed(
        "import json; print(json.dumps({'score': 1.5}))", workdir=tmp_path / "run"
    )
    assert result.ok and result.payload == {"score": 1.5}


def test_the_sandboxed_verification_imports_out_of_process(tmp_path):
    _stage(tmp_path, GOOD)
    result = sandbox.verify_sandboxed(NAME, project=tmp_path)
    assert result.ok, result.stderr
    assert result.payload["problems"] == []


def test_the_sandboxed_verification_reports_a_broken_mode(tmp_path):
    files = dict(GOOD)
    files["mode.py"] = files["mode.py"].replace(":finish", ":nowhere")
    _stage(tmp_path, files)
    result = sandbox.verify_sandboxed(NAME, project=tmp_path)
    problems = result.payload.get("problems") or [result.problem()]
    assert problems and any("nowhere" in str(p) for p in problems)


def test_a_sandboxed_smoke_scores_a_real_mode(tmp_path):
    _stage(tmp_path, GOOD)
    result = sandbox.smoke(f"{NAME}/counter", tmp_path / "run", project=tmp_path)
    assert result.ok, result.stderr
    assert result.payload["score"] == 3.0


# ── promotion ──────────────────────────────────────────────────────────────
def test_a_package_is_generated_until_it_is_promoted(tmp_path):
    _stage(tmp_path, GOOD)
    (registry.workspace_root(tmp_path) / NAME / ".origin.json").write_text(
        json.dumps({"origin": "create-mode", "trust": "generated"})
    )
    assert registry.load(NAME, tmp_path).trust == "generated"
    recorded = sandbox.promote(tmp_path, NAME, note="reviewed by hand")
    assert recorded["trust"] == "trusted"
    assert recorded["note"] == "reviewed by hand"
    assert recorded["promoted_at"]
    assert registry.load(NAME, tmp_path).trust == "trusted"


def test_promoting_something_that_is_not_there_is_refused(tmp_path):
    with pytest.raises(sandbox.SandboxError, match="no workspace package"):
        sandbox.promote(tmp_path, "nowhere")


# ── the command line ───────────────────────────────────────────────────────
def test_the_sandbox_cli_ast_verb(tmp_path, capsys, monkeypatch):
    _stage(tmp_path, GOOD)
    monkeypatch.chdir(tmp_path)
    assert sandbox.main(["--ast", NAME]) == 0
    assert json.loads(capsys.readouterr().out)["problems"] == []


def test_the_sandbox_cli_reports_problems_as_a_failure(tmp_path, capsys, monkeypatch):
    _stage(tmp_path, {"__init__.py": "x = 1\n"})
    monkeypatch.chdir(tmp_path)
    assert sandbox.main(["--ast", NAME]) == 1
    assert json.loads(capsys.readouterr().out)["problems"]


def test_the_sandbox_cli_needs_an_instruction(capsys):
    assert sandbox.main([]) == 2
    assert "usage" in capsys.readouterr().err


def test_smoke_can_supply_a_model_client(tmp_path, monkeypatch):
    """A mode with an agent node is the common case, so smoke can score it.

    Without a model the run raises `agent node … needs a model client`, which is
    "not measured" rather than a score — so the option exists, and what it builds is
    the same client every other path uses.
    """
    from refactory_dsh import sandbox

    seen: dict[str, str] = {}

    def fake_run_sandboxed(code: str, **_kwargs: object):
        seen["code"] = code
        return sandbox.SandboxResult(ok=True, payload={}, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox, "run_sandboxed", fake_run_sandboxed)
    sandbox.smoke("circle_packing/adaptive", tmp_path / "w", project=Path("."), model=True)
    assert "DshModelClient" in seen["code"]
    # The generated client is handed the trusted parent's routing context, so the
    # sandboxed package never reads a credential out of its own environment.
    assert "model=DshModelClient(**" in seen["code"]

    sandbox.smoke("circle_packing/adaptive", tmp_path / "w2", project=Path("."))
    assert "DshModelClient" not in seen["code"]


def test_a_nodes_model_call_cannot_background_its_own_work(monkeypatch):
    """The one-shot completion a node makes is foreground-only (#63).

    The base profile's spawn subagent tool defaults to background-and-continuable. A
    node is one completion whose answer *is* its artifact, so an agent that spawns its
    researchers, says they are "running in the background" and ends its turn has done
    the node's work nowhere: the research stage did exactly that, twice, leaving
    `research/` empty. The graph profile patches the same row for the CEO.
    """
    import subprocess

    import yaml

    from refactory_dsh.model import NODE_PATCH, DshModelClient

    assert NODE_PATCH.exists(), f"{NODE_PATCH} is missing; the flags a node runs with are not reviewable"
    overlay = yaml.safe_load(NODE_PATCH.read_text())
    row = next(entry for entry in overlay if entry.get("id") == "tool-subagent")
    assert row["config"]["enableRunInBackground"] is False
    assert row["config"]["backgroundMode"] == "one-shot"

    seen: dict[str, object] = {}

    def fake_completion(self, cmd, run_dir, stop_at=None, budget_s=None):  # type: ignore[no-untyped-def]
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "answer", "")

    # A node's completion now streams through ``_run_completion`` (Popen, so a
    # timeout can dispose managed tool processes); patching ``subprocess.run``
    # would let the real dsh CLI run.
    monkeypatch.setattr(DshModelClient, "_run_completion", fake_completion)
    DshModelClient().complete("say something")
    cmd = seen["cmd"]
    assert "--patch" in cmd  # type: ignore[operator]
    assert str(NODE_PATCH) in cmd  # type: ignore[operator]
