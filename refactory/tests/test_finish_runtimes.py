"""A finish (`--refine <PR> --finish`) through each runtime's real entry point.

A fixture repository is published on the fake GitHub with a draft pull request whose
description lists its Open items. Each case then runs through

- **the walker**, `python -m refactory_dsh.design --refine <PR> --finish`, and
- **the spine**, `refactory ceo ... --refine <PR> --finish --headless`, which lowers the
  graph and drives it through `graph-owner.mjs` with nobody at the gates,

each agent answered by node id by the fake dsh (`parity_dsh.py`), as
`test_runtime_equivalence.py` does. No run is given a verdict.

- ``fixed``: two open items, a QA finding and an environment item. The run gets past the
  approval gate only because both are concrete; it pushes onto the pull request's own
  branch, opens no second pull request, never shows the environment item to design_doc or
  the builder, and ticks both items in the description with their evidence.
- ``judgment``: a judgment call is selected too, so the gate waits for a person.
- ``rounds``: one fix round, then a draft, the items left open with why.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import delivery, finish, open_items, worktree
from refactory_dsh.open_items import OpenItem
from refactory_dsh.paths import REPO_ROOT
from tests_support import (DESIGN_DOC_ANSWER, PITFALLS_ANSWER, REVIEW_ANSWERS, VERIFICATION_ANSWER, git,
                           git_repo, graph_profile, model_home_with_a_key, stub)

HERE = Path(__file__).resolve().parent
FAKE_DSH = HERE / "parity_dsh.py"
FAKE_GH = HERE / "fake_gh.py"
OWNER = REPO_ROOT / "packages/workflow/workflow-spine/tests/fixtures/graph-owner.mjs"
NODE = shutil.which("node")
PYTHON = str(REPO_ROOT / ".venv/bin/python")
REFACTORY = str(REPO_ROOT / ".venv/bin/refactory")

pytestmark = pytest.mark.skipif(
    NODE is None or not (REPO_ROOT / "packages/workflow/workflow-spine/node_modules/@deepseek-ai/cordis").exists()
    or not Path(REFACTORY).exists(),
    reason="the spine owner needs Node, the installed workspace (pnpm install) and the venv (uv sync)")

NAME = "finish_tool"
LOGIN = "octo"
SLUG = f"{LOGIN}/app"
URL = f"https://github.com/{SLUG}/pull/1"
BRIEF = "Finish the pull request's open items."
CHECK = "python -m pytest -q tests"
TARGET_FILES = {
    "tool.py": "VERSION = 1\n",
    "tests/test_pin.py": "import tool\n\n\ndef test_pin():\n    assert tool.VERSION == 1\n",
}
QA = OpenItem("qa-1", "qa_finding", "tool.py still sets VERSION to 1; the pull request is for VERSION 2",
              ("tool.py:1", "tests/test_pin.py::test_pin"))
ENV = OpenItem("env-1", "environment", "ENV-ONLY-SUMMARY: the CI runner ran out of disk",
               ("log: No space left on device in tests/test_disk.py::test_disk",))
#: A change failure the builder is not told how to fix: its test fails on the pull request
#: already, so the run's check excuses it as the base's, and it must stay open.
CHANGE = OpenItem("cf-2", "change_failure", "tests/test_two.py::test_two fails",
                  ("tests/test_two.py::test_two",))
JUDGMENT = OpenItem("jc-1", "judgment_call", "whether NOTES.md belongs in this pull request",
                    ("the excuse rule used: out of the brief's scope",))


def _body(*items: OpenItem) -> str:
    return (f"{delivery.BODY_START}\n## Summary\n\nVersion two.\n{delivery.BODY_END}\n\n"
            + open_items.render(list(items)))


QA_PROCEED = json.dumps({"verdict": "proceed", "must_fix": [], "follow_ups": [], "dismissed": [],
                         "finish_items": [{"id": "qa-1", "met": True, "evidence": "tests/test_pin.py::test_pin"}]})
#: A design with no ## Changes: the design review sends it back once, then is out of rewrites.
NO_CHANGES = DESIGN_DOC_ANSWER.replace("## Changes", "## Plan")


def _files(files: dict[str, str]) -> str:
    return "\n".join(f"===FILE: {name}===\n{body}" for name, body in files.items()) + "\n"


#: Every agent node a finish asks, by node id (`parity_dsh.py`). No research, pitfalls or
#: strategy agent: a finish is a light refine.
SCRIPT: dict[str, list[Any]] = {
    "design_doc": [DESIGN_DOC_ANSWER],
    "emit": [_files({"tool.py": "VERSION = 2\n",
                     "tests/test_pin.py": "import tool\n\n\ndef test_pin():\n    assert tool.VERSION == 2\n"})],
    "early_asks_check": [{"judge": "early_asks"}],
    "health_checker": [{"text": REVIEW_ANSWERS[0], "repo_questions": True}],
    "code_reviewer": [{"text": REVIEW_ANSWERS[1], "repo_questions": True}],
    "qa_director": [{"text": REVIEW_ANSWERS[2], "repo_questions": True}],
    "verifier": [VERIFICATION_ANSWER],
    "qa_judge": [{"text": QA_PROCEED, "judge": "intent_audit"}],
    "overwatch": [{"judge": "correction_overwatch"}],
    # Never asked: a finish skips them. An answer is here so a run that wrongly asks one
    # fails on the assertion that names it, not on a missing script.
    "pitfalls_researcher": [PITFALLS_ANSWER],
}


class World:
    """One runtime's sealed world: its GitHub with the pull request, the clone, DSH and the run."""

    def __init__(self, base: Path, gh_shim: Path, *, body: str, script: dict[str, list[Any]] | None = None) -> None:
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
        (self.dsh / "scenario.json").write_text(json.dumps({**SCRIPT, **(script or {})}))
        (self.dsh / "owner-script.json").write_text(json.dumps({"answers": []}))
        launcher = self.dsh / "dsh"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_DSH}" --scenario "{self.dsh}" '
                            f'--owner "{OWNER}" --node-bin "{NODE}" "$@"\n')
        launcher.chmod(0o755)
        self.env = {key: value for key, value in os.environ.items()
                    if key not in ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN",
                                   "DSH_CLI", "REFACTORY_WORKDIR", "REFACTORY_PROJECT",
                                   "REFACTORY_WORKER_CONTEXT")}
        self.env.update({
            "PATH": os.pathsep.join([str(base / "bin"), str(gh_shim), os.environ.get("PATH", "")]),
            "FAKE_GH_ROOT": str(github), "GH_CONFIG_DIR": str(base / "gh"),
            "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": f"url.file://{github}/.insteadOf",
            "GIT_CONFIG_VALUE_0": "https://github.com/",
            "DSH_HOME": str(home), "REFACTORY_MODEL_DSH_HOME": str(home),
            "REFACTORY_HOME": str(base / "refactory-home"), "REFACTORY_PROJECTS_DIR": str(base / "projects"),
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
        # The pull request: a branch with the first run's work on it, open as a draft.
        git(repo, "checkout", "-q", "-b", "version-two")
        (repo / "NOTES.md").write_text("Version two is coming.\n")
        (repo / "tests/test_two.py").write_text("def test_two():\n    assert False\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "version two: notes")
        git(repo, "push", "-q", str(bare), "version-two:refs/heads/version-two")
        git(repo, "checkout", "-q", "main")
        git(repo, "remote", "add", "origin", f"https://github.com/{SLUG}.git")
        (github / f"{SLUG}.pulls.json").write_text(json.dumps([{
            "number": 1, "head": "version-two", "base": "main", "state": "OPEN", "title": "Version two",
            "body": body, "draft": True}]))

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

    def prompts(self, node: str) -> list[str]:
        calls = [json.loads(line) for line in (self.dsh / "calls.jsonl").read_text().splitlines()]
        return [call["prompt"] for call in calls if call["node"] == node]

    def nodes(self) -> list[str]:
        return [json.loads(line)["node"] for line in (self.dsh / "calls.jsonl").read_text().splitlines()]


FLAGS = ["--name", NAME, "--check", CHECK, "--refine", URL, "--finish", "--omit-steering"]


def _walker(world: World, *extra: str) -> subprocess.CompletedProcess[str]:
    return world.call([PYTHON, "-m", "refactory_dsh.design", "--brief", BRIEF, "--target", str(world.app),
                       "--workdir", str(world.run), *FLAGS, *extra])


def _spine(world: World, *extra: str) -> subprocess.CompletedProcess[str]:
    return world.call([REFACTORY, "ceo", BRIEF, "--target", str(world.app), "--workdir", str(world.run),
                       "--headless", *FLAGS, *extra])


def _status(world: World) -> str:
    return json.loads((world.run / ".refactory/spine-status.json").read_text()).get("status", "")


def _finished(runtime: str, world: World, *extra: str) -> None:
    done = (_walker if runtime == "walker" else _spine)(world, *extra)
    assert done.returncode == 0, done.stderr[-5000:]
    assert runtime == "walker" or _status(world) == "done"


def _asked(runtime: str, world: World, *extra: str) -> None:
    """Nobody is there to answer the approval gate: the walker stops at it, the spine is refused there."""
    done = (_walker if runtime == "walker" else _spine)(world, *extra)
    assert done.returncode != 0 and "gate_strategy" in done.stderr, done.stderr[-5000:]
    assert runtime == "walker" or _status(world) == "refused"


#: The early asks judge finds the brief's ask not built on the first attempt.
NOT_BUILT = json.dumps({"items": [{"id": "ask-1", "kind": "ask", "text": "the brief", "built": False,
                                   "missing": "qa-1's test of VERSION 2"}]})

#: One sealed world per runtime and case: what its pull request lists, what its agents
#: answer beyond `SCRIPT`, and how the finish is driven.
#: - ``fixed``: a QA finding the QA judge confirms (resolved), a change failure whose test
#:   still fails (open) and an environment item (resolved).
#: - ``review``: the design review is out of rewrites, so the gate is left to a person.
#: - ``judgment``: --items names a judgment call too, so the gate is left to a person.
#: - ``rounds``: the early asks judge finds the first build lacking (gate_check sends it
#:   back: the round), and the second build breaks a test the self-check runs, which would
#:   send it back again: no second round, a draft.
CASES: dict[str, tuple[str, dict[str, list[Any]], Any, tuple[str, ...]]] = {
    "fixed": (_body(QA, CHANGE, ENV), {}, _finished, ()),
    "review": (_body(QA, ENV), {"design_doc": [NO_CHANGES]}, _asked, ()),
    "judgment": (_body(QA, JUDGMENT, ENV), {}, _asked, ("--items", "qa-1,jc-1")),
    "rounds": (_body(QA, ENV), {
        "emit": [SCRIPT["emit"][0],
                 _files({"tests/test_pin.py": "import tool\n\n\ndef test_pin():\n    assert tool.VERSION == 3\n"})],
        "early_asks_check": [NOT_BUILT, {"judge": "early_asks"}],
    }, _finished, ()),
}
RUNTIMES = ("walker", "spine")


@pytest.fixture(scope="module")
def worlds(tmp_path_factory: pytest.TempPathFactory, _real_gh_shim: tuple[Path, Path]) -> dict[str, World]:
    root = tmp_path_factory.mktemp("finish")
    made = {f"{case}-{runtime}": World(root / f"{case}-{runtime}", _real_gh_shim[0], body=body, script=script)
            for case, (body, script, _drive, _extra) in CASES.items() for runtime in RUNTIMES}
    with ThreadPoolExecutor(max_workers=len(made)) as pool:
        futures = {}
        for name, world in made.items():
            case, runtime = name.rsplit("-", 1)
            _body_, _script, drive, extra = CASES[case]
            futures[name] = pool.submit(drive, runtime, world, *extra)
    failures = []
    for name, future in futures.items():
        try:
            future.result()
        except BaseException as error:  # noqa: BLE001 - each runtime's failure is reported
            log = made[name].base / "commands.log"
            failures.append(f"{name}: {error}\n{log.read_text()[-6000:] if log.exists() else ''}")
    if failures:
        pytest.fail("\n\n".join(failures), pytrace=False)
    return made


def _facts(world: World, runtime: str) -> dict[str, Any]:
    if runtime == "walker":
        return json.loads((world.run / ".refactory/progress.json").read_text())["facts"]
    return json.loads((world.run / "facts.json").read_text())


def _pull(world: World) -> dict[str, Any]:
    [pull] = json.loads((world.github / f"{SLUG}.pulls.json").read_text())
    assert pull["number"] == 1 and pull["head"] == "version-two"
    return pull


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_finish_updates_the_same_pull_request_on_each_runtime(worlds, runtime):
    world = worlds[f"fixed-{runtime}"]
    facts = _facts(world, runtime)
    assert facts["finish_gate"]["skipped"] is True, facts.get("finish_gate")
    assert facts["finish"]["builder"] == ["qa-1", "cf-2"] and facts["finish"]["environment"] == ["env-1"]
    # A light refine: one design, one build, no research, pitfalls or strategy agent.
    nodes = world.nodes()
    assert nodes.count("design_doc") == 1 and nodes.count("emit") == 1, nodes
    assert not {"research_director", "pitfalls_researcher", "strategy_director", "synthesize_strategy"} & set(nodes)
    [design] = world.prompts("design_doc")
    [build] = world.prompts("emit")
    assert QA.summary in design and finish.FIX_PLAN in design
    for prompt in (design, build):
        assert ENV.summary not in prompt and "test_disk" not in prompt
    # The same pull request, at the run's head; no second one.
    head = worktree.recorded(world.run).head
    assert git(world.github / f"{SLUG}.git", "rev-parse", "refs/heads/version-two") == head
    pull = _pull(world)
    items = {item.id: item for item in open_items.parse(pull["body"])}
    assert items["qa-1"].status == "resolved"
    assert items["qa-1"].results == (f"finish run at `{head[:12]}`: the QA judge confirmed it met "
                                     "(tests/test_pin.py::test_pin)",)
    assert items["cf-2"].status == "open" and "attempted, not confirmed" in items["cf-2"].results[-1]
    assert items["env-1"].status == "resolved" and "passed, so the failure did not recur" in items["env-1"].results[-1]
    # Only the boxes of what was resolved, and one result line per selected item, changed.
    before = _body(QA, CHANGE, ENV)
    assert pull["body"].replace("- [x] **qa-1**", "- [ ] **qa-1**").replace("- [x] **env-1**", "- [ ] **env-1**") \
        .split(open_items.START, 1)[1].replace("\n  - result: " + items["qa-1"].results[-1], "") \
        .replace("\n  - result: " + items["cf-2"].results[-1], "") \
        .replace("\n  - result: " + items["env-1"].results[-1], "") == before.split(open_items.START, 1)[1]
    assert pull["body"].count("## Open items") == 1
    assert "- The approval gate was skipped: every item this finish run fixes is concrete" in pull["body"]
    assert json.loads((world.run / delivery.RECORD).read_text())["status"] == "updated"


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_judgment_call_leaves_the_gate_to_a_person_on_each_runtime(worlds, runtime):
    world = worlds[f"judgment-{runtime}"]
    facts = _facts(world, runtime)
    assert facts["finish_gate"]["skipped"] is False
    assert "jc-1 is a judgment_call" in facts["finish_gate"]["reason"]
    assert "emit" not in world.nodes()
    assert _pull(world)["body"] == _body(QA, JUDGMENT, ENV)


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_finish_gets_one_fix_round_on_each_runtime(worlds, runtime):
    """gate_check's send-back for an ask not built is the round; the second build breaks a
    test, which the self-check would otherwise send back once more (its own limits allow
    it): the run is a draft instead, and the builder is asked twice, never three times."""
    world = worlds[f"rounds-{runtime}"]
    facts = _facts(world, runtime)
    nodes = world.nodes()
    assert nodes.count("emit") == 2, nodes
    assert facts["finish_rounds"] == 1 and facts["early_asks_sendback_count"] == 1
    assert finish.ROUND_SPENT in facts["draft_delivery"]["reason"], facts["draft_delivery"]
    pull = _pull(world)
    assert pull["draft"] is True
    head = worktree.recorded(world.run).head
    assert git(world.github / f"{SLUG}.git", "rev-parse", "refs/heads/version-two") == head
    items = {item.id: item for item in open_items.parse(pull["body"])}
    assert items["qa-1"].status == "open" and "the run delivered a draft" in items["qa-1"].results[-1]
    # The check failed on tests/test_pin.py, not on the test the environment item names.
    assert items["env-1"].status == "resolved" and "but not on tests/test_disk.py::test_disk" in \
        items["env-1"].results[-1]


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_design_review_out_of_rewrites_leaves_the_gate_to_a_person_on_each_runtime(worlds, runtime):
    """Every item is concrete, but the design review found no ## Changes twice: the gate is
    not answered, and nobody is there to answer it."""
    world = worlds[f"review-{runtime}"]
    facts = _facts(world, runtime)
    assert facts["finish_gate"]["skipped"] is False and "design review" in facts["finish_gate"]["reason"]
    assert world.nodes().count("design_doc") == 2 and "emit" not in world.nodes()
    from refactory_dsh.steering import Ledger

    assert not [entry for entry in Ledger(world.run).entries() if entry["kind"] == "verdict"]
