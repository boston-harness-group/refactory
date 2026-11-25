"""A design run with the fix-round budget through each runtime's real entry point.

The scenario: a small repository published on the fake GitHub; the builder emits
tool.py with VERSION 2 and updates the pinning test so the check passes, but the
QA judge blocks the change each time. With ``fix_rounds=1`` (the default) the
first QA redo spends the one round; the second hits the budget and the run delivers
a draft pull request whose description lists its Open items (a ``qa_finding`` for
the QA judge's blocking finding).

Both runtimes — the walker and the spine — drive the same scripted agents, and the
test checks that each runtime delivers a draft with the right facts and Open items.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import finish, open_items
from refactory_dsh.paths import REPO_ROOT
from tests_support import (DESIGN_DOC_ANSWER, PITFALLS_ANSWER, REVIEW_ANSWERS,
                           VERIFICATION_ANSWER, git, git_repo, graph_profile,
                           model_home_with_a_key, stub)

HERE = Path(__file__).resolve().parent
FAKE_DSH = HERE / "parity_dsh.py"
FAKE_GH = HERE / "fake_gh.py"
OWNER = REPO_ROOT / "packages/workflow/workflow-spine/tests/fixtures/graph-owner.mjs"
NODE = shutil.which("node")
PYTHON = str(REPO_ROOT / ".venv/bin/python")
REFACTORY = str(REPO_ROOT / ".venv/bin/refactory")

pytestmark = [
    pytest.mark.skipif(
        NODE is None or not (REPO_ROOT / "packages/workflow/workflow-spine/node_modules/@deepseek-ai/cordis").exists()
        or not Path(REFACTORY).exists(),
        reason="the spine owner needs Node, the installed workspace (pnpm install) and the venv (uv sync)"),
    pytest.mark.slow,
]

NAME = "fixround_tool"
LOGIN = "octo"
SLUG = f"{LOGIN}/app"
BRIEF = "Change tool.py to VERSION 2."
CHECK = "python -m pytest -q tests"
TARGET_FILES = {
    "tool.py": "VERSION = 1\n",
    "tests/test_pin.py": "import tool\n\n\ndef test_pin():\n    assert tool.VERSION == 1\n",
}

LEAN_RESEARCH = json.dumps([{
    "focus": "where VERSION is set", "slug": "version",
    "finding": "tool.py:1 sets VERSION; tests/test_pin.py:5 pins it to 1",
    "ids": ["FND-version-1"],
    "footprint": {"existing_functions": ["tool.py:1 VERSION"],
                  "new_items": [], "shared_engine": []}}])

QA_REVIEW_FAIL = "BLOCKING: tool.py:1: missing migration script for VERSION bump\n\nVERDICT: FAIL"
QA_REDO = json.dumps({"verdict": "redo", "must_fix": [
    {"source": "qa_director", "where": "tool.py:1", "what": "missing migration script", "why": "ask 1"}
], "follow_ups": [], "dismissed": []})


def _files(files: dict[str, str]) -> str:
    return "\n".join(f"===FILE: {name}===\n{body}" for name, body in files.items()) + "\n"


UPDATED_PIN = "import tool\n\n\ndef test_pin():\n    assert tool.VERSION == 2\n"
GOOD_EMIT_1 = _files({"tool.py": "VERSION = 2\n", "tests/test_pin.py": UPDATED_PIN})
GOOD_EMIT_2 = _files({"tool.py": '"""tool"""\nVERSION = 2\n', "tests/test_pin.py": UPDATED_PIN})

HEALTH_OK = {"text": REVIEW_ANSWERS[0], "repo_questions": True}
CODE_OK = {"text": REVIEW_ANSWERS[1], "repo_questions": True}
QA_FAIL = {"text": QA_REVIEW_FAIL, "repo_questions": True}

SCRIPT: dict[str, list[Any]] = {
    "size_change": [json.dumps([{"size": "scoped", "why": "one line",
                                  "footprint": {"existing_functions": ["tool.py:1 VERSION"],
                                                "new_items": [], "shared_engine": []},
                                  "cross_cutting": []}])],
    "research_director": [LEAN_RESEARCH],
    "pitfalls_researcher": [{"text": PITFALLS_ANSWER,
                             "files": {"pitfalls.md": "# Pitfalls\n\n(scripted)\n"}}],
    "strategy_director": [{"text": "perspectives", "files": {
        "strategy/simplest.md": "### FND-simplest-1 — the simplest thing\n\n# simplest\n\nDetails.\n",
        "strategy/criteria.md": "### FND-criteria-1 — the criteria check\n\n# criteria\n\nDetails.\n"}}],
    "synthesize_strategy": ["one plan"],
    "design_doc": [DESIGN_DOC_ANSWER],
    "apply_corrections": [DESIGN_DOC_ANSWER],
    "emit": [GOOD_EMIT_1, GOOD_EMIT_2],
    "early_asks_check": [{"judge": "early_asks"}, {"judge": "early_asks"}],
    "health_checker": [HEALTH_OK, HEALTH_OK],
    "code_reviewer": [CODE_OK, CODE_OK],
    "qa_director": [QA_FAIL, QA_FAIL],
    "verifier": [VERIFICATION_ANSWER, VERIFICATION_ANSWER],
    "qa_judge": [{"text": QA_REDO, "judge": "intent_audit"},
                 {"text": QA_REDO, "judge": "intent_audit"}],
    "overwatch": [{"judge": "correction_overwatch"}],
}


class World:
    """One runtime's sealed world: its GitHub, the target repo, DSH and the run."""

    def __init__(self, base: Path, gh_shim: Path) -> None:
        self.base = base
        base.mkdir(parents=True)
        github = self.github
        (github / LOGIN).mkdir(parents=True)
        (github / "login").write_text(LOGIN + "\n")
        stub(base / "bin", "gh", f'exec "{sys.executable}" "{FAKE_GH}" "$@"')
        home = base / "dsh-home"
        model_home_with_a_key(home)
        graph_profile(home, REPO_ROOT)
        self.dsh.mkdir()
        (self.dsh / "scenario.json").write_text(json.dumps(SCRIPT))
        (self.dsh / "owner-script.json").write_text(json.dumps({"answers": []}))
        launcher = self.dsh / "dsh"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_DSH}" --scenario "{self.dsh}" '
                            f'--owner "{OWNER}" --node-bin "{NODE}" "$@"\n')
        launcher.chmod(0o755)
        self.env = {key: value for key, value in os.environ.items()
                    if key not in ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN",
                                   "DSH_CLI", "REFACTORY_WORKDIR", "REFACTORY_PROJECT",
                                   "REFACTORY_WORKER_CONTEXT")}
        project = base / "project"
        project.mkdir()
        self.env.update({
            "PATH": os.pathsep.join([str(base / "bin"), str(gh_shim), os.environ.get("PATH", "")]),
            "FAKE_GH_ROOT": str(github), "GH_CONFIG_DIR": str(base / "gh"),
            "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": f"url.file://{github}/.insteadOf",
            "GIT_CONFIG_VALUE_0": "https://github.com/",
            "DSH_HOME": str(home), "REFACTORY_MODEL_DSH_HOME": str(home),
            "REFACTORY_HOME": str(base / "refactory-home"), "REFACTORY_PROJECTS_DIR": str(base / "projects"),
            "REFACTORY_PROJECT": str(project),
            "DSH_CLI": str(launcher), "PYTHONDONTWRITEBYTECODE": "1",
        })
        repo = git_repo(self.app)
        for name, text in TARGET_FILES.items():
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_text(text)
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "tests")
        bare = github / f"{SLUG}.git"
        git(bare.parent, "init", "-q", "--bare", "-b", "main", bare.name)
        (github / f"{SLUG}.json").write_text(json.dumps({"visibility": "private", "default_branch": "main"}))
        git(repo, "push", "-q", str(bare), "main:refs/heads/main")
        git(repo, "remote", "add", "origin", f"https://github.com/{SLUG}.git")

    @property
    def github(self) -> Path:
        return self.base / "github"

    @property
    def dsh(self) -> Path:
        return self.base / "dsh"

    @property
    def app(self) -> Path:
        return self.base / "app"

    @property
    def run(self) -> Path:
        return self.base / "run"

    def call(self, argv: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
        done = subprocess.run(argv, env=self.env, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout)
        with (self.base / "commands.log").open("a") as log:
            log.write(f"$ {' '.join(argv)}\nexit {done.returncode}\n{done.stdout[-4000:]}\n{done.stderr[-8000:]}\n")
        return done

    def nodes(self) -> list[str]:
        return [json.loads(line)["node"] for line in (self.dsh / "calls.jsonl").read_text().splitlines()]


FLAGS = ["--name", NAME, "--check", CHECK, "--omit-steering"]
RUNTIMES = ("walker", "spine")


def _walker(world: World) -> subprocess.CompletedProcess[str]:
    first = world.call([PYTHON, "-m", "refactory_dsh.design", "--brief", BRIEF, "--target", str(world.app),
                        "--workdir", str(world.run), *FLAGS])
    assert first.returncode == 2 and "gate_strategy" in first.stderr, first.stderr[-3000:]
    steer = world.call([PYTHON, "-m", "refactory_dsh.console", "steer", "--run", str(world.run),
                        "--verdict", "gate_strategy=proceed"])
    assert steer.returncode == 0, steer.stderr[-3000:]
    return world.call([PYTHON, "-m", "refactory_dsh.design", "--resume", str(world.run)])


def _spine(world: World) -> subprocess.CompletedProcess[str]:
    (world.dsh / "owner-script.json").write_text(json.dumps({"answers": ["proceed"]}))
    return world.call([REFACTORY, "ceo", BRIEF, "--target", str(world.app), "--workdir", str(world.run),
                       "--headless", *FLAGS])


@pytest.fixture(scope="module")
def worlds(tmp_path_factory: pytest.TempPathFactory, _real_gh_shim: tuple[Path, Path]) -> dict[str, World]:
    root = tmp_path_factory.mktemp("fixround")
    made = {runtime: World(root / runtime, _real_gh_shim[0]) for runtime in RUNTIMES}
    failures = []
    for runtime, world in made.items():
        try:
            done = (_walker if runtime == "walker" else _spine)(world)
            assert done.returncode in (0, 2), done.stderr[-5000:]
        except BaseException as error:  # noqa: BLE001
            log = world.base / "commands.log"
            failures.append(f"{runtime}: {error}\n{log.read_text()[-6000:] if log.exists() else ''}")
    if failures:
        pytest.fail("\n\n".join(failures), pytrace=False)
    return made


def _facts(world: World, runtime: str) -> dict[str, Any]:
    if runtime == "walker":
        return json.loads((world.run / ".refactory/progress.json").read_text())["facts"]
    return json.loads((world.run / "facts.json").read_text())


def _pull(world: World) -> dict[str, Any]:
    pulls = json.loads((world.github / f"{SLUG}.pulls.json").read_text())
    assert len(pulls) >= 1
    return pulls[-1]


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_fix_round_budget_delivers_draft_with_open_items(worlds, runtime):
    """A design run where the QA judge always says redo stops after one fix
    round (fix_rounds=1, the default) and delivers a draft PR with Open items."""
    world = worlds[runtime]
    facts = _facts(world, runtime)
    dd = facts.get("draft_delivery")
    assert dd is not None, f"expected draft_delivery fact; facts keys: {sorted(facts)}"
    assert finish.FIX_ROUND_SPENT in dd["reason"], dd["reason"]
    assert facts["finish_rounds"] >= 1

    nodes = world.nodes()
    assert nodes.count("emit") == 2, f"expected 2 emits; got {nodes.count('emit')} in {nodes}"

    pull = _pull(world)
    assert pull["draft"] is True

    items = open_items.parse(pull["body"])
    assert items, f"expected Open items in the PR body; body starts with: {pull['body'][:300]}"
    finding_items = [item for item in items if item.kind in ("qa_finding", "unfinished_ask")]
    assert finding_items, f"expected qa_finding or unfinished_ask items; got kinds: {[i.kind for i in items]}"

    assert open_items.parse(open_items.render(items)) == items
