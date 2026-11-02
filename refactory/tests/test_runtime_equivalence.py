"""Design mode on the walker and on the spine: one scripted run, two runtimes (dm-158).

The done-test for walker parity (dm-123, #436). One scripted design run goes through each
runtime's real entry point, and what each left behind is compared:

- **the walker** through its main, `python -m refactory_dsh.design`, whose agents call
  `DshModelClient`;
- **the spine** through `refactory ceo`, which lowers the graph, writes the worker specs and
  starts `dsh --profile graph run`. That launch is handed to `graph-owner.mjs`: the real
  graph-startup row and the real spine plugin, mounted as the graph profile mounts them,
  with a scripted person at the user gates instead of a CEO model. Commands run as the
  spine runs them, and every agent node through its worker (`worker_node`).

Both runtimes' agents reach the same fake `dsh` (`parity_dsh.py`, as `DSH_CLI`), which
answers by node id, so the same answers drive both whatever order they visit the nodes in.
The scenario: a small repository published on the fake GitHub; a design run whose approval
gate is answered "proceed" with one correction; a first build that breaks a pinning test,
which the self-check sends back for one fix-up; the review fork and the judges; and the pull
request. The walker, which cannot put a gate to a person, is answered the one way it can be:
it stops at the gate, the person runs `console steer`, and the run resumes. The spine runs
it twice: answered at the gate's question (`card`), and stopped at the gate, steered and
resumed through `refactory graph-control resume` (`resumed`), which is the walker's flow.

What is compared, normalised for paths, times, ids and durations: visits (the order of node
visits and each one's outcome), gate verdicts, facts, the run directory's node-written
files, the prompts each agent received, the route each agent's model session ran on (dsh's
arguments, its patches, the route variables and the worker context), the commits pushed,
and the delivery (the pull request and the gh calls that made it). The visits are read
from each runtime's own record, but `events.jsonl` and the walker's progress record are
not themselves compared yet.

Every difference is either equal or explained by a named parity gap (`GAPS`). A gap is a
precise rewrite of one side that makes the two agree.
`test_the_runtimes_agree_but_for_the_named_gaps` fails on any difference no listed gap
explains, and each open gap has its own strict xfail, which starts passing (and so fails)
once the gap is closed: take it out of `SCENARIO_GAPS` then. `PARITY_REPORT=1` prints every
difference before any gap is applied.

PR #516 (agent workspaces) changes where agents run; this test compares what they emit and
what is committed, never an agent's working directory.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh.design import DesignMode
from refactory_dsh.paths import REPO_ROOT
from refactory_dsh.run_state import is_alive
from tests_support import (DESIGN_DOC_ANSWER, PITFALLS_ANSWER, RESEARCH_ANSWER, REVIEW_ANSWERS,
                           VERIFICATION_ANSWER, git, git_repo, graph_profile, model_home_with_a_key,
                           stub)

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

NAME = "parity_tool"
LOGIN = "octo"
SLUG = f"{LOGIN}/app"
BRIEF = "Change tool.py to VERSION 2."
CHECK = "python -m pytest -q tests"
CORRECTION = "Also give tool.py a NAME constant set to 'tool'."
TARGET_FILES = {
    "tool.py": "VERSION = 1\n",
    "tests/test_pin.py": "import tool\n\n\ndef test_pin():\n    assert tool.VERSION == 1\n",
}
#: How long each reviewer's completion takes.
HOLD_S = 1.5
REVIEWERS = ("health_checker", "code_reviewer", "qa_director", "verifier")
QA_PROCEED = json.dumps({"verdict": "proceed", "must_fix": [], "follow_ups": [], "dismissed": []})
OVERWATCH_PASS = json.dumps({"verdict": "pass", "must_fix": [], "follow_ups": [], "corrections": []})


def _files(files: dict[str, str]) -> str:
    """The builder's answer: `===FILE:` blocks."""
    return "\n".join(f"===FILE: {name}===\n{body}" for name, body in files.items()) + "\n"


#: What each agent node answers, by node id, visit by visit (`parity_dsh.py`).
SCRIPT: dict[str, list[Any]] = {
    "research_director": [RESEARCH_ANSWER],
    "pitfalls_researcher": [{"text": PITFALLS_ANSWER, "files": {"pitfalls.md": "# Pitfalls\n\n(scripted)\n"}}],
    "strategy_director": [{"text": "perspectives", "files": {
        "strategy/simplest.md": "### FND-simplest-1 — the simplest thing\n\n# simplest\n\nDetails.\n",
        "strategy/criteria.md": "### FND-criteria-1 — the criteria check\n\n# criteria\n\nDetails.\n"}}],
    "synthesize_strategy": ["one plan"],
    "design_doc": [DESIGN_DOC_ANSWER],
    "apply_corrections": [DESIGN_DOC_ANSWER.replace("## Changes\n\n- tool.py\n",
                                                    "## Changes\n\n- tool.py, with a NAME constant\n")],
    # The first build breaks the pinning test; the self-check sends it back, and the
    # fix-up updates the test.
    "emit": [_files({"tool.py": "VERSION = 2\nNAME = 'tool'\n"}),
             _files({"tests/test_pin.py": "import tool\n\n\ndef test_pin():\n    assert tool.VERSION == 2\n    assert tool.NAME == 'tool'\n"})],
    "early_asks_check": [{"judge": "early_asks"}],
    # The four reviewers each take a moment, so reviewers run one after another are told
    # apart from reviewers run at once (`test_the_review_branches_run_at_once`).
    "health_checker": [{"text": REVIEW_ANSWERS[0], "repo_questions": True, "hold_s": HOLD_S}],
    "code_reviewer": [{"text": REVIEW_ANSWERS[1], "repo_questions": True, "hold_s": HOLD_S}],
    "qa_director": [{"text": REVIEW_ANSWERS[2], "repo_questions": True, "hold_s": HOLD_S}],
    "verifier": [{"text": VERIFICATION_ANSWER, "hold_s": HOLD_S}],
    "qa_judge": [{"text": QA_PROCEED, "judge": "intent_audit"}],
    "overwatch": [{"judge": "correction_overwatch"}],
}


# ── one runtime's world ───────────────────────────────────────────────────────


@dataclass
class World:
    """One runtime's sealed world: its GitHub, target, DSH home, fake dsh and run."""

    base: Path
    env: dict[str, str]

    @property
    def run(self) -> Path:
        return self.base / "run"

    @property
    def dsh(self) -> Path:
        return self.base / "dsh"

    @property
    def github(self) -> Path:
        return self.base / "github"

    def call(self, argv: list[str], timeout: int = 300) -> subprocess.CompletedProcess[str]:
        """A command a person runs, from the checkout, as `refactory ceo` runs DSH."""
        done = subprocess.run(argv, env=self.env, cwd=REPO_ROOT, capture_output=True, text=True,
                              timeout=timeout)
        with (self.base / "commands.log").open("a") as log:
            log.write(f"$ {' '.join(argv)}\nexit {done.returncode}\n{done.stdout[-4000:]}\n"
                      f"{done.stderr[-8000:]}\n")
        return done


def _world(base: Path, gh_shim: Path, answers: list[Any], *, built: bool = True) -> World:
    """GitHub with `octo/app` on it, its checkout, a DSH home and the fake dsh, under ``base``."""
    base.mkdir(parents=True)
    github = base / "github"
    (github / LOGIN).mkdir(parents=True)
    (github / "login").write_text(LOGIN + "\n")
    stub(base / "bin", "gh", f'exec "{sys.executable}" "{FAKE_GH}" "$@"')
    home = base / "dsh-home"
    model_home_with_a_key(home)
    graph_profile(home, REPO_ROOT)
    dsh = base / "dsh"
    dsh.mkdir()
    scenario = json.loads(json.dumps(SCRIPT))
    if not built:
        scenario["emit"] = [
            _files({"tool.py": "VERSION = 2\n"}),
            _files({"tests/test_pin.py": "import tool\n\n\ndef test_pin():\n    assert tool.VERSION == 2\n"}),
        ]
    (dsh / "scenario.json").write_text(json.dumps(scenario))
    (dsh / "owner-script.json").write_text(json.dumps({"answers": answers}))
    launcher = dsh / "dsh"
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_DSH}" --scenario "{dsh}" '
                        f'--owner "{OWNER}" --node-bin "{NODE}" "$@"\n')
    launcher.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if key not in ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN",
                          "DSH_CLI", "REFACTORY_WORKDIR", "REFACTORY_PROJECT", "REFACTORY_WORKER_CONTEXT")}
    env.update({
        "PATH": os.pathsep.join([str(base / "bin"), str(gh_shim), os.environ.get("PATH", "")]),
        "FAKE_GH_ROOT": str(github), "GH_CONFIG_DIR": str(base / "gh"),
        "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": f"url.file://{github}/.insteadOf",
        "GIT_CONFIG_VALUE_0": "https://github.com/",
        "DSH_HOME": str(home), "REFACTORY_MODEL_DSH_HOME": str(home),
        "REFACTORY_HOME": str(base / "refactory-home"), "REFACTORY_PROJECTS_DIR": str(base / "projects"),
        "DSH_CLI": str(launcher), "PYTHONDONTWRITEBYTECODE": "1",
    })
    repo = git_repo(base / "app")
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
    return World(base, env)


def _steer(world: World) -> None:
    """The person answers the stopped gate: proceed, with the correction."""
    done = world.call([PYTHON, "-m", "refactory_dsh.console", "steer", "--run", str(world.run),
                       "--verdict", "gate_strategy=proceed", "--kind", "correction", "--note", CORRECTION])
    assert done.returncode == 0, done.stderr[-3000:]


def _walk(world: World) -> None:
    """The walker: its main stops at the gate, the person steers, the main resumes."""
    first = world.call([PYTHON, "-m", "refactory_dsh.design", "--name", NAME, "--brief", BRIEF,
                        "--check", CHECK, "--target", str(world.base / "app"), "--workdir", str(world.run),
                        "--omit-steering"])
    assert first.returncode == 2 and "gate_strategy" in first.stderr, first.stderr[-3000:]
    _steer(world)
    resumed = world.call([PYTHON, "-m", "refactory_dsh.design", "--resume", str(world.run)])
    assert resumed.returncode == 0, resumed.stderr[-5000:]


def _ceo(world: World) -> subprocess.CompletedProcess[str]:
    return world.call([REFACTORY, "ceo", BRIEF, "--name", NAME, "--check", CHECK,
                       "--target", str(world.base / "app"), "--workdir", str(world.run),
                       "--headless", "--omit-steering"])


def _spine_status(world: World) -> str:
    return json.loads((world.run / ".refactory/spine-status.json").read_text()).get("status", "")


def _spine_at_the_card(world: World) -> None:
    """The spine: `refactory ceo`, the person answering the gate's question in place."""
    done = _ceo(world)
    assert done.returncode == 0 and _spine_status(world) == "done", done.stderr[-5000:]


def _spine_resumed(world: World) -> None:
    """The spine: nobody can be asked, so the run stops at the gate (`refused`, as a headless
    owner records it); the person steers, and `graph-control resume` goes on."""
    done = _ceo(world)
    assert done.returncode != 0 and _spine_status(world) == "refused", done.stderr[-5000:]
    assert "gate_strategy is waiting for a person" in done.stderr, done.stderr[-3000:]
    _steer(world)
    resumed = world.call([REFACTORY, "graph-control", "resume", "--run", str(world.run), "--json"])
    assert resumed.returncode == 0, resumed.stdout[-2000:] + resumed.stderr[-3000:]
    owner = json.loads(resumed.stdout)["pid"]
    # Bounded poll with increasing intervals, capping at a generous multiple of the
    # expected duration rather than a flat constant (timing robustness under load).
    started = time.monotonic()
    interval = 0.1
    cap = 300  # generous cap; a normal run finishes in ~30 s
    while is_alive(owner):
        elapsed = time.monotonic() - started
        assert elapsed < cap, f"the resumed owner did not finish after {elapsed:.0f} s"
        time.sleep(min(interval, 2.0))
        interval = min(interval * 1.5, 2.0)  # ramp from 100 ms to 2 s
    assert _spine_status(world) == "done", (world.run / ".refactory/spine-status.json").read_text()


# ── what a run left, normalised ───────────────────────────────────────────────

GRAPH = DesignMode().graph()
FORKS = {node.id: tuple(node.targets) for node in GRAPH.nodes.values() if node.kind == "fork"}
JOINS = {node.id for node in GRAPH.nodes.values() if node.kind == "join"}
GATES = {node.id for node in GRAPH.nodes.values() if node.kind == "gate"}

#: Files each runtime keeps for itself, compared through what is built from them (the
#: visits from the events, verdicts and facts from the resume records), or holding only
#: process bookkeeping: worker receipts, delivery markers, locks, homes, lowered graphs.
BOOKKEEPING = (
    ".refactory/events.jsonl", ".refactory/progress.json", ".refactory/spine-checkpoint.json",
    ".refactory/facts.lock", ".refactory/invocation.lock", ".refactory/invocation.json",
    ".refactory/supervisor.lock", ".refactory/supervisor.log", ".refactory/console.json",
    ".refactory/workers/", ".refactory/agent-delivery/", ".refactory/worker-specs/", ".refactory/prompts/",
    ".refactory/prepare-", ".refactory/check-home/", ".refactory/base-check-home/",
    ".refactory/check-copy-home/", ".refactory/validation-history/", ".refactory/graph-control",
    ".refactory/set-aside/", "graphs/", "console.json", "recovery.log",
)


class Normaliser:
    """Rewrites what differs between any two runs of the same scenario: paths, ids, times."""

    def __init__(self, world: World, tmp_root: Path) -> None:
        paths = [(str(world.base / "refactory-home" / "worktrees"), "<WORKTREES>"), (str(world.base), "<BASE>"),
                 (str(REPO_ROOT), "<CHECKOUT>"), (str(tmp_root), "<TMP>")]
        # Longest first: one of these may be a prefix of another.
        self.paths = sorted(paths, key=lambda pair: -len(pair[0]))

    def __call__(self, text: str) -> str:
        for path, name in self.paths:
            text = text.replace(path, name)
        for pattern, name in (
            (r"run-[0-9a-f]{12}", "run-<ID>"), (r"app-[0-9a-f]{8}", "app-<H>"),
            (r"\b[0-9a-f]{64}\b", "<HASH>"), (r"\b[0-9a-f]{40}\b", "<SHA>"), (r"\b[0-9a-f]{32}\b", "<UUID>"),
            (r"\b[0-9a-f]{7,12}\b", "<SHORT>"),
            (r"\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d(\.\d+)?(Z|[+-]\d\d:?\d\d)?", "<TIME>"),
            (r"\b\d+(\.\d+)? ?s\b", "<S>s"), (r'("(?:s|elapsed_s|duration_s|seconds)": )[\d.]+', r"\1<S>"),
        ):
            text = re.sub(pattern, name, text)
        return text


def _canonical(value: Any) -> Any:
    """A value with the JSON inside its strings (and JSON lines) decoded, so it compares as
    data and a difference shows as the field that differs."""
    if isinstance(value, str) and value[:1] in ("[", "{"):
        try:
            return _canonical(json.loads(value))
        except ValueError:
            try:
                return [_canonical(json.loads(line)) for line in value.splitlines() if line.strip()]
            except ValueError:
                return value
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def _text(value: Any) -> str:
    value = _canonical(value)
    return value if isinstance(value, str) else json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False)


def _fork_of(node: str) -> str | None:
    return next((fork for fork, targets in FORKS.items() if node in targets), None)


def _grouped(visits: list[tuple[str, str]]) -> list[str]:
    """Visits as lines; a fork's branches, which run at once, as one sorted line."""
    lines: list[str] = []
    index = 0
    while index < len(visits):
        node, outcome = visits[index]
        fork = _fork_of(node)
        if fork is None:
            lines.append(f"{node}: {outcome}")
            index += 1
            continue
        branches = []
        while index < len(visits) and _fork_of(visits[index][0]) == fork:
            branches.append(f"{visits[index][0]}={visits[index][1]}")
            index += 1
        lines.append(f"{fork} branches: {', '.join(sorted(branches))}")
    return lines


def _walker_visits(run: Path, *, graph: Any = None, gates: set[str] | None = None,
                   forks: dict[str, tuple[str, ...]] | None = None) -> list[tuple[str, str]]:
    """The walker's visits from its events. It keeps no per-visit verdict, so a gate's is
    read from the route it took: the edge to the node it started next."""
    _graph = graph or GRAPH
    _gates = gates if gates is not None else GATES
    _forks = forks if forks is not None else FORKS
    events = [json.loads(line) for line in (run / ".refactory/events.jsonl").read_text().splitlines()
              if line.strip()]
    started = [event for event in events if event.get("kind") == "node_started"]
    ends = [event for event in events if event.get("kind") in ("node_completed", "node_failed")]
    visits: list[tuple[str, str]] = []
    for index, event in enumerate(started):
        node = event["node_id"]
        end = next((item for item in ends if item["node_id"] == node), None)
        if end is None:
            visits.append((node, "unfinished"))
            continue
        ends.remove(end)
        if end["kind"] == "node_failed":
            visits.append((node, "stopped"))
        elif node in _gates:
            following = started[index + 1]["node_id"] if index + 1 < len(started) else ""
            verdicts = [edge.condition for edge in _graph.out_edges(node)
                        if edge.target == following or following in _forks.get(edge.target, ())]
            visits.append((node, verdicts[0] if len(verdicts) == 1 else f"? -> {following}"))
        else:
            visits.append((node, "completed"))
    return visits


def _spine_visits(run: Path, *, forks: dict[str, tuple[str, ...]] | None = None,
                  joins: set[str] | None = None) -> list[tuple[str, str]]:
    """The spine's visits from its own log; fork and join nodes are its control nodes."""
    _forks = forks if forks is not None else FORKS
    _joins = joins if joins is not None else JOINS
    log = json.loads((run / ".refactory/spine-checkpoint.json").read_text())["state"]["log"]
    visits: list[tuple[str, str]] = []
    for line in log:
        if line.startswith("completed "):
            node = line.removeprefix("completed ").strip()
            if node not in _forks and node not in _joins:
                visits.append((node, "completed"))
        elif found := re.match(r"gate (\S+): (\S+) ->", line):
            visits.append((found.group(1), found.group(2)))
        else:
            visits.append((line, "logged"))
    return visits


def _facts(run: Path, runtime: str) -> dict[str, Any]:
    if runtime == "walker":
        return json.loads((run / ".refactory/progress.json").read_text())["facts"]
    return json.loads((run / "facts.json").read_text())


def _verdicts(run: Path, runtime: str) -> dict[str, str]:
    """Each gate's last verdict, from each runtime's own record of it: the walker's
    progress record, the spine's checkpoint."""
    if runtime == "walker":
        return dict(sorted(json.loads((run / ".refactory/progress.json").read_text())["gate_verdicts"].items()))
    return dict(sorted(json.loads((run / ".refactory/spine-checkpoint.json").read_text())
                       ["state"]["gate_verdicts"].items()))


def _artifacts(run: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for path in sorted(run.rglob("*")):
        relative = path.relative_to(run).as_posix()
        if path.is_dir() or relative.startswith(BOOKKEEPING):
            continue
        try:
            found[relative] = path.read_text()
        except UnicodeDecodeError:
            found[relative] = f"<{path.stat().st_size} bytes>"
    return found


STATUS = ".refactory/spine-status.json"


def _status_identity(artifacts: dict[str, str], runtime: str) -> dict[str, str]:
    """The status file with the two fields that name who wrote it -- its runtime and its
    owning process -- checked and set aside: each runtime names itself, by design."""
    if STATUS not in artifacts:
        return artifacts
    status = json.loads(artifacts[STATUS])
    assert status.get("runtime") == runtime and isinstance(status.get("owner_pid"), int), status
    status.update(runtime="<RUNTIME>", owner_pid="<PID>")
    return {**artifacts, STATUS: json.dumps(status)}


CONFIG = ".refactory/run-config.json"


def _config_identity(artifacts: dict[str, str], world: World, runtime: str) -> dict[str, str]:
    """`run-config.json` with the fields that name the runtime that runs it -- the
    runtime, the command that started it, the one that continues it, and on the spine
    the supervisor's owner fragment, which a walked run has no supervisor for -- checked
    and set aside, as the status file's are. Every other field (mode, project, knobs,
    model route, limits, fingerprints, driver) is compared."""
    if CONFIG not in artifacts:
        return artifacts
    config = json.loads(artifacts[CONFIG])
    # The owner_argv names the mode the run was started with; for the card
    # scenario it is ``default_pack/design``, for custom modes it is the
    # mode id the test registered.  Accept whatever the run recorded.
    actual_owner = config.get("owner_argv")
    expected = ({"runtime": "walker", "command": "python -m refactory_dsh.design",
                 "resume_command": f"python -m refactory_dsh.design --resume {world.run}",
                 "owner_argv": None}
                if runtime == "walker" else
                {"runtime": "spine", "command": "refactory ceo",
                 "resume_command": f"refactory graph-control resume --run {world.run}",
                 "owner_argv": actual_owner})
    assert {key: config.get(key) for key in expected} == expected, config
    assert runtime == "spine" or "owner_argv" not in config, config
    config.update({key: f"<{key.upper()}>" for key in expected})
    return {**artifacts, CONFIG: json.dumps(config)}


def _prompts(world: World) -> dict[str, str]:
    calls = [json.loads(line) for line in (world.dsh / "calls.jsonl").read_text().splitlines()]
    return {f"{call['node']}#{call['visit']}": call["prompt"] for call in calls}


def _routes(world: World) -> dict[str, Any]:
    """The route each completion's model session ran on, as dsh was started with it."""
    calls = [json.loads(line) for line in (world.dsh / "calls.jsonl").read_text().splitlines()]
    # The patches in the order dsh was given them: a path's spelling must not reorder them.
    return {f"{call['node']}#{call['visit']}": {**call["route"], "patches": list(call["route"]["patches"].items())}
            for call in calls}


def _receipts(world: World) -> dict[str, list[dict[str, Any]]]:
    """Each node's worker receipts, in the order they started."""
    found: dict[str, list[dict[str, Any]]] = {}
    for path in sorted((world.run / ".refactory/workers").glob("worker-*.json")):
        receipt = json.loads(path.read_text())
        found.setdefault(receipt.get("node_id", ""), []).append(receipt)
    return {node: sorted(items, key=lambda item: item["started"]) for node, items in found.items()}


def _commits(world: World) -> dict[str, str]:
    """What each pull request carries: every commit pushed to its branch, message and patch."""
    bare = world.github / f"{SLUG}.git"
    found: dict[str, str] = {}
    for pull in json.loads((world.github / f"{SLUG}.pulls.json").read_text()):
        shas = git(bare, "log", "--reverse", "--format=%H", f"main..{pull['head']}").split()
        found[f"pull {pull['number']}"] = "\n\n".join(
            git(bare, "show", "--format=%s%n%n%b", "--patch", sha) for sha in shas)
    return found


def _delivery(world: World) -> dict[str, str]:
    calls = world.base / "bin" / "calls.log"
    return {"pull requests": (world.github / f"{SLUG}.pulls.json").read_text(),
            "gh calls": calls.read_text() if calls.exists() else ""}


def observe(world: World, runtime: str, tmp_root: Path, *,
            graph: Any = None) -> dict[str, dict[str, str]]:
    """Everything compared about one run, by aspect and key, as normalised text.

    ``graph`` overrides the module-level GRAPH for scenarios that use a different graph
    (e.g. the complex parity scenario).  When data files for an aspect are missing
    (no ``calls.jsonl`` when the walker uses a scripted model, no ``pulls.json`` when
    a scenario creates no pull requests), that aspect is an empty dict.
    """
    normal = Normaliser(world, tmp_root)
    _graph = graph or GRAPH
    _forks = {n.id: tuple(n.targets) for n in _graph.nodes.values() if n.kind == "fork"} if graph else FORKS
    _joins = {n.id for n in _graph.nodes.values() if n.kind == "join"} if graph else JOINS
    _gates = {n.id for n in _graph.nodes.values() if n.kind == "gate"} if graph else GATES
    if runtime == "walker":
        visits = _walker_visits(world.run, graph=_graph, gates=_gates, forks=_forks)
    else:
        visits = _spine_visits(world.run, forks=_forks, joins=_joins)
    aspects: dict[str, dict[str, Any]] = {
        "visits": {"order and outcome": "\n".join(_grouped(visits))},
        "verdicts": _verdicts(world.run, runtime),
        "facts": _facts(world.run, runtime),
        "artifacts": _config_identity(_status_identity(_artifacts(world.run), runtime), world, runtime),
    }
    # Aspects that depend on the DSH model shim or GitHub infrastructure: gracefully
    # empty when their data files do not exist (e.g. the walker uses a scripted model,
    # or a scenario creates no pull requests).
    try:
        aspects["prompts"] = _prompts(world)
    except (FileNotFoundError, KeyError):
        aspects["prompts"] = {}
    try:
        aspects["routes"] = _routes(world)
    except (FileNotFoundError, KeyError):
        aspects["routes"] = {}
    try:
        aspects["commits"] = _commits(world)
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        aspects["commits"] = {}
    try:
        aspects["delivery"] = _delivery(world)
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        aspects["delivery"] = {}
    return {aspect: {key: normal(_text(value)) for key, value in values.items()}
            for aspect, values in aspects.items()}


# ── differences and the named gaps that explain them ─────────────────────────


@dataclass(frozen=True)
class Difference:
    aspect: str
    key: str
    walker: str | None
    spine: str | None

    def __str__(self) -> str:
        if self.walker is None or self.spine is None:
            side = "spine" if self.walker is None else "walker"
            return f"[{self.aspect}] {self.key}: only on the {side}"
        diff = difflib.unified_diff(self.walker.splitlines(), self.spine.splitlines(), "walker", "spine",
                                    lineterm="", n=1)
        body = "\n".join(f"    {line[:240]}" for line in list(diff)[2:40])
        return f"[{self.aspect}] {self.key}:\n{body}"


def differences(walker: dict[str, dict[str, str]], spine: dict[str, dict[str, str]]) -> list[Difference]:
    found = []
    for aspect in walker:
        for key in sorted(set(walker[aspect]) | set(spine[aspect])):
            left, right = walker[aspect].get(key), spine[aspect].get(key)
            if left != right:
                found.append(Difference(aspect, key, left, right))
    return found


Rewrite = Callable[[str, str, str, "str | None"], "str | None"]


@dataclass(frozen=True)
class Gap:
    """A known parity gap: a rewrite of one side's value that makes the two agree."""

    name: str
    tracker: str
    title: str
    rewrite: Rewrite
    #: Whether the gap still shows in a difference its rewrite leaves alone. A rewrite that
    #: explains only some forms of the gap is a no-op on the other forms, which are open
    #: gaps all the same, not closed ones.
    present: Callable[[Difference], bool] | None = None

    def __str__(self) -> str:
        return f"{self.name} ({self.tracker}): {self.title}"


GAPS: dict[str, Gap] = {}

#: The gaps open in each scenario. A gap that closes makes its xfail pass, which fails:
#: take it out of here then.
SCENARIO_GAPS: dict[str, list[str]] = {
    "card": [],
    "resumed": [],
    "complex": [],
}


def _rewritten(difference: Difference, gaps: list[Gap]) -> tuple[str | None, str | None]:
    walker, spine = difference.walker, difference.spine
    for gap in gaps:
        walker = gap.rewrite(difference.aspect, difference.key, "walker", walker)
        spine = gap.rewrite(difference.aspect, difference.key, "spine", spine)
    return walker, spine


def unexplained(found: list[Difference], gaps: list[Gap]) -> list[Difference]:
    """The differences the gaps do not explain, shown after the gaps' rewrites."""
    left = []
    for difference in found:
        walker, spine = _rewritten(difference, gaps)
        if walker != spine:
            left.append(Difference(difference.aspect, difference.key, walker, spine))
    return left


def shown(gap: Gap, found: list[Difference]) -> list[Difference]:
    """The differences ``gap`` still shows in: its rewrite changes them, or the gap says it is
    present. A gap is closed only then on no difference, whatever else differs on the key."""
    return [difference for difference in found
            if gap.rewrite(difference.aspect, difference.key, "walker", difference.walker) != difference.walker
            or gap.rewrite(difference.aspect, difference.key, "spine", difference.spine) != difference.spine
            or (gap.present is not None and gap.present(difference))]


# ── the runs ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory, _real_gh_shim: tuple[Path, Path]) -> dict[str, Any]:
    """The walker's run and the spine's two, made at once, each in its own sealed world."""
    root = tmp_path_factory.mktemp("parity")
    shim = _real_gh_shim[0]
    card = [{"selected": "proceed", "custom": CORRECTION}]
    worlds = {"walker": _world(root / "walker", shim, []), "card": _world(root / "card", shim, card),
              "resumed": _world(root / "resumed", shim, []),
              "unbuilt-walker": _world(root / "unbuilt-walker", shim, [], built=False),
              "unbuilt-spine": _world(root / "unbuilt-spine", shim, [], built=False)}
    drivers = {"walker": _walk, "card": _spine_at_the_card, "resumed": _spine_resumed,
               "unbuilt-walker": _walk, "unbuilt-spine": _spine_resumed}
    with ThreadPoolExecutor(max_workers=len(worlds)) as pool:
        futures = {name: pool.submit(drivers[name], world) for name, world in worlds.items()}
    failures = []
    for name, future in futures.items():
        try:
            future.result()
        except BaseException as error:  # noqa: BLE001 - every runtime's failure is reported
            log = worlds[name].base / "commands.log"
            failures.append(f"{name}: {error}\n{log.read_text()[-6000:] if log.exists() else ''}")
    if failures:
        pytest.fail("\n\n".join(failures), pytrace=False)
    tmp_root = tmp_path_factory.getbasetemp()
    observed = {name: observe(world, "walker" if name == "walker" else "spine", tmp_root)
                for name, world in worlds.items() if name in ("walker", "card", "resumed")}
    found = {scenario: differences(observed["walker"], observed[scenario])
             for scenario in SCENARIO_GAPS if scenario in observed}
    if os.environ.get("PARITY_REPORT"):
        for scenario, items in found.items():
            print(f"\n==== walker vs spine ({scenario}): {len(items)} difference(s)")
            print("\n".join(str(item) for item in items))
    return {"worlds": worlds, "observed": observed, "found": found}


# ── the tests ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("runtime", ["walker", "card", "resumed"])
def test_each_run_went_through_the_whole_scenario(runs, runtime):
    """Not a vacuous comparison: each run built twice, applied the correction, reviewed,
    and opened its pull request with both changes."""
    observed = runs["observed"][runtime]
    visits = observed["visits"]["order and outcome"].splitlines()
    assert "apply_corrections: completed" in visits
    assert visits.count("emit: completed") == 2, visits
    # The self-check sends back from prepare_check; the full check runs beside the
    # reviews in fork_qa.  At least one reloop from prepare_check must have occurred
    # (the first attempt's self-check finds the pinned test failing).
    assert any("prepare_check:" in line and "reloop" in line for line in visits), visits
    assert "delivery_gate: proceed" in visits, visits
    assert any(line.startswith("fork_qa branches:") for line in visits), visits
    assert "NAME constant" in observed["artifacts"]["design.md"]
    [patch] = observed["commits"].values()
    assert "+VERSION = 2" in patch and "+NAME = 'tool'" in patch and "assert tool.VERSION == 2" in patch
    [pull] = json.loads(observed["delivery"]["pull requests"])
    assert pull["state"] == "OPEN" and not pull["draft"]
    route = json.loads(observed["routes"]["verifier#0"])
    assert route["env"]["REFACTORY_MODEL_ID"] == "fake-model" and route["patches"], route


@pytest.mark.parametrize("runtime", ["walker", "card", "resumed"])
def test_the_builder_is_told_to_run_the_self_checks_tests_on_each_runtime(runs, runtime):
    """The builder runs the tests its change breaks in its own session, so the self-check does
    not send it back to a cold one. Its very first briefing names ``refactory-targeted-tests``,
    which runs the self-check's own selection; the fix-up's adds the time the self-check
    measured. Neither runtime's builder is told to run the whole suite or given a fixed
    timeout."""
    world = runs["worlds"][runtime]
    calls = [json.loads(line) for line in (world.dsh / "calls.jsonl").read_text().splitlines()]
    first, fixup = [call["prompt"] for call in calls if call["node"] == "emit"]
    told = ("Before you stop, run `refactory-targeted-tests` and fix what fails. It runs "
            "exactly what the self-check will run for your change. ")
    assert f"{told}No self-check has timed these tests yet.\n" in first
    assert f"{told}The last self-check's tests took about " in fixup
    for prompt in (first, fixup):
        assert "none yet" not in prompt and "Your targeted test command" not in prompt
        assert "Do not run the whole test suite: the full check runs after you finish" in prompt
        assert "5m" not in prompt and "{{builder_tests}}" not in prompt


@pytest.mark.parametrize("runtime", ["walker", "card", "resumed"])
def test_correction_audit_and_overwatch_accept_only_built_correction(runs, runtime):
    """The real audit feeds the scripted overwatch in each runtime: built passes; unbuilt reloops."""
    worlds = runs["worlds"]
    built = worlds[runtime]
    unbuilt = worlds["unbuilt-walker" if runtime == "walker" else "unbuilt-spine"]
    built_audit = json.loads((built.run / "correction-audit.json").read_text())
    built_verdict = json.loads((built.run / "overwatch-verdict.json").read_text())
    assert built_audit and all(item["status"] == "pass" for item in built_audit), built_audit
    assert built_verdict["verdict"] == "pass", built_verdict
    assert built_verdict["corrections"] and all(item["met"] is True for item in built_verdict["corrections"])
    calls = [json.loads(line) for line in built.dsh.joinpath("calls.jsonl").read_text().splitlines()]
    assert calls and all("What the user has said" not in call["prompt"] for call in calls)
    correction_prompt = next(call["prompt"] for call in calls if call["node"] == "apply_corrections")
    assert CORRECTION in correction_prompt
    first_emit = next(call for call in calls if call["node"] == "emit")
    assert "Read the approved design from `design.md`" in first_emit["prompt"]
    assert "with a NAME constant" in first_emit["available_design"]

    unbuilt_audit = json.loads((unbuilt.run / "correction-audit.json").read_text())
    unbuilt_verdict = json.loads((unbuilt.run / "overwatch-verdict.json").read_text())
    assert unbuilt_audit and any(item["status"] == "fail" for item in unbuilt_audit), unbuilt_audit
    assert unbuilt_verdict["verdict"] == "redo", unbuilt_verdict
    assert unbuilt_verdict["corrections"] and all(item["met"] is False for item in unbuilt_verdict["corrections"])
    assert "reloop" in (unbuilt.run / "reviews/overwatch-gate.txt").read_text()


#: Scenarios that use the ``runs`` fixture (the full card scenario infrastructure).
#: The "complex" scenario has its own standalone test (test_complex_scenario_runtimes_agree).
_RUNS_SCENARIOS = sorted(s for s in SCENARIO_GAPS if s in ("card", "resumed"))


@pytest.mark.parametrize("scenario", _RUNS_SCENARIOS)
def test_the_runtimes_agree_but_for_the_named_gaps(runs, scenario):
    """Every difference between the walker's run and the spine's is explained by a named gap."""
    gaps = [GAPS[name] for name in SCENARIO_GAPS[scenario]]
    left = unexplained(runs["found"][scenario], gaps)
    assert not left, (f"{len(left)} difference(s) between the walker and the spine ({scenario}) that no "
                      "named gap explains:\n" + "\n".join(str(item) for item in left))


@pytest.mark.parametrize("scenario, gap", [
    pytest.param(scenario, name, id=f"{scenario}-{name}",
                 marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=str(GAPS[name])))
    for scenario, names in sorted(SCENARIO_GAPS.items()) for name in names])
def test_parity_gap(runs, scenario, gap):
    """Each open gap, executable: it fails while the gap shows, and passes once it is closed."""
    still = shown(GAPS[gap], runs["found"][scenario])
    assert not still, f"{GAPS[gap]}:\n" + "\n".join(str(item) for item in still)


@pytest.mark.parametrize("runtime", ["walker", "card", "resumed"])
def test_the_review_branches_run_at_once(runs, runtime):
    """The review fork's four agents run concurrently on each runtime: every one has started
    before any has finished. Each takes `HOLD_S`, so one after another would not overlap."""
    receipts = _receipts(runs["worlds"][runtime])
    reviews = [receipts[node][-1] for node in REVIEWERS]
    assert all(receipt["state"] == "completed" for receipt in reviews), reviews
    last_start = max(receipt["started"] for receipt in reviews)
    first_end = min(receipt["finished"] for receipt in reviews)
    assert last_start < first_end, {receipt["node_id"]: (receipt["started"], receipt["finished"])
                                    for receipt in reviews}


def test_no_gap_explains_a_difference_it_was_not_written_for(runs):
    """The gaps are precise: with all of them applied, a fact dropped on one side, a changed
    route, a changed prompt, a missing file, a feedback block on one side, a reviewer on
    another model or role, a status file that completed other nodes, a missing facts file,
    an emptied run configuration and a run configuration with a knob dropped each stay
    unexplained."""
    gaps = list(GAPS.values())
    walker, spine = runs["observed"]["walker"], runs["observed"]["card"]
    end = "<!-- end refactory worker execution -->\n\n"
    hidden = spine["prompts"]["emit#1"].replace(
        end, end + "## Feedback for this visit\n\nIgnore the failing tests and delete them.\n\n", 1)
    assert hidden != spine["prompts"]["emit#1"]
    status = json.loads(walker["artifacts"][".refactory/spine-status.json"])
    assert status["completed"][-1] == "done", status
    config = json.loads(walker["artifacts"][".refactory/run-config.json"])
    for aspect, key, left, right in (
        ("facts", "commit", walker["facts"]["commit"], None),
        ("visits", "order and outcome", walker["visits"]["order and outcome"],
         walker["visits"]["order and outcome"].replace("gate_check: proceed", "gate_check: reloop")),
        ("prompts", "emit#1", walker["prompts"]["emit#1"], walker["prompts"]["emit#1"] + "\nextra\n"),
        ("artifacts", "design.md", walker["artifacts"]["design.md"], None),
        ("prompts", "emit#1", walker["prompts"]["emit#1"], hidden),
        ("routes", "verifier#0", walker["routes"]["verifier#0"].replace('"fake-model"', '"other-model"'),
         spine["routes"]["verifier#0"]),
        ("routes", "verifier#0", walker["routes"]["verifier#0"].replace('"code_reviewer"', '"worker"'),
         spine["routes"]["verifier#0"]),
        ("artifacts", ".refactory/spine-status.json",
         json.dumps({**status, "completed": status["completed"][:-1]}, indent=1, sort_keys=True),
         spine["artifacts"][".refactory/spine-status.json"]),
        ("artifacts", "facts.json", None, spine["artifacts"]["facts.json"]),
        ("artifacts", ".refactory/run-config.json",
         json.dumps({**config, "configuration_fingerprints": {}}, indent=1, sort_keys=True),
         spine["artifacts"][".refactory/run-config.json"]),
        ("artifacts", ".refactory/run-config.json",
         json.dumps({**config, "knobs": config["knobs"][:-1]}, indent=1, sort_keys=True),
         spine["artifacts"][".refactory/run-config.json"]),
    ):
        made = Difference(aspect, key, left, right)
        assert unexplained([made], gaps), f"a gap explained a made-up {aspect} difference in {key}"


# ── SWEEP-9: complex parity scenario ──────────────────────────────────────────


def _complex_graph():
    """Build the complex scenario graph (brief item 1).

    Topology:
    - prepare (op) → agent1 (agent, knob in prompt) → gate_strategy (user gate)
    - gate_strategy → apply_corrections (op) → write_relative (op, writes file, ctx.model=None)
    - write_relative → fn_gate (fn gate, prints stdout, reloops once)
    - fn_gate ←reloop→ agent2 (agent)
    - fn_gate →proceed→ outer_fork
      - branch_main: inner_fork (NESTED FORK inside a fork branch)
        - inner_a (op) | inner_b (op, reads knob)
        - inner_join
        → branch_main_end (op)
      - branch_opt: opt_fail (op, FAILS — optional branch first node)
        → opt_step2 (op)
      - branch_c (op)
    - outer_join → done (op)
    """
    from refactory_dsh.nn import Graph, Node

    graph = Graph()

    # 1. Prepare op.
    graph.add(Node("prepare", "op", call="loop_ops:noop",
                    produces=("name", "brief", "check")))

    # 2. Agent node (uses knob in prompt template).
    graph.add(Node("agent1", "agent",
                    prompt_template="Design with knob={{test_knob}}: {{brief}}",
                    writes=("design.md",), produces=("design",)))
    graph.connect("prepare", "agent1")

    # 3. User gate — stop + resume here.
    graph.add(Node("gate_strategy", "gate", gate_kind="user",
                    reads=("design.md",), max_iterations=5,
                    notes="Approve the strategy"))
    graph.connect("agent1", "gate_strategy")

    # 4. Apply correction op.
    graph.add(Node("apply_corrections", "op",
                    call="loop_ops:parity_apply_correction",
                    reads=("design.md",), writes=("design.md",),
                    produces=("corrections_applied",)))
    graph.connect("gate_strategy", "apply_corrections", condition="proceed")

    # 5. Op that writes a relative path and reads ctx.model (SWEEP-3, SWEEP-5).
    graph.add(Node("write_relative", "op",
                    call="loop_ops:parity_write_relative",
                    writes=("parity_output.txt",),
                    produces=("ctx_model_is_none", "knob_applied")))
    graph.connect("apply_corrections", "write_relative")

    # 6. fn gate that prints to stdout and reloops once (SWEEP-4).
    graph.add(Node("fn_gate", "gate", gate_kind="fn",
                    evaluator="loop_ops:parity_fn_gate",
                    reads=(), max_iterations=3, notes="fn check",
                    produces=("parity_visits",)))
    graph.connect("write_relative", "fn_gate")

    # 7. Agent node after reloop (gets stdout feedback).
    graph.add(Node("agent2", "agent",
                    prompt_template="Continue: {{brief}}",
                    writes=("output.txt",), produces=("result",)))
    graph.connect("fn_gate", "agent2", condition="reloop")
    graph.connect("agent2", "fn_gate")

    # 8. Outer fork with 3 branches including a nested fork and an optional that fails.
    graph.add(Node("outer_fork", "fork",
                    targets=("branch_main", "opt_fail", "branch_c"),
                    optional_targets=("opt_fail",)))
    graph.connect("fn_gate", "outer_fork", condition="proceed")

    # Branch main: contains a NESTED fork (fork inside a fork branch).
    graph.add(Node("branch_main", "fork",
                    targets=("inner_a", "inner_b")))

    # Inner branch A.
    graph.add(Node("inner_a", "op", call="loop_ops:noop",
                    produces=("inner_a_done",)))

    # Inner branch B: reads knob value.
    graph.add(Node("inner_b", "op", call="loop_ops:parity_read_knob",
                    produces=("knob_read",)))

    # Inner join.
    graph.add(Node("inner_join", "join",
                    sources=("inner_a", "inner_b")))
    graph.connect("branch_main", "inner_join")
    graph.connect("inner_a", "inner_join")
    graph.connect("inner_b", "inner_join")

    # After inner fork: branch_main_end.
    graph.add(Node("branch_main_end", "op", call="loop_ops:noop",
                    produces=("main_done",)))
    graph.connect("inner_join", "branch_main_end")

    # Optional branch: first node FAILS (two-node optional branch).
    graph.add(Node("opt_fail", "op", call="loop_ops:parity_nested_fork_fail",
                    produces=("opt_result",)))
    graph.add(Node("opt_step2", "op", call="loop_ops:noop"))
    graph.connect("opt_fail", "opt_step2")

    # Branch C: simple op.
    graph.add(Node("branch_c", "op", call="loop_ops:noop",
                    produces=("c_done",)))

    # Outer join.
    graph.add(Node("outer_join", "join",
                    sources=("branch_main_end", "opt_step2", "branch_c")))
    graph.connect("outer_fork", "outer_join")
    graph.connect("branch_main_end", "outer_join")
    graph.connect("opt_step2", "outer_join")
    graph.connect("branch_c", "outer_join")

    # Done.
    graph.add(Node("done", "op", call="loop_ops:noop"))
    graph.connect("outer_join", "done")

    return graph


# ── Complex scenario: mode package for the spine ─────────────────────────────

#: Scripted answers for the complex scenario's agent nodes.
_COMPLEX_SCRIPT: dict[str, list[Any]] = {
    "agent1": ["scripted design answer"],
    "agent2": ["scripted continue answer"],
}


def _write_complex_mode_package(project: Path) -> None:
    """Write a Mode package for the complex scenario so `refactory ceo --mode` can run it."""
    pkg = project / ".factory" / "packages" / "complex_test"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(
        'from refactory_dsh.nn import Package\n'
        'from .mode import ComplexTestMode\n\n'
        'PACKAGE = Package(\n'
        '    name="complex_test",\n'
        '    description="test mode for complex parity scenario",\n'
        '    modes=(ComplexTestMode,),\n'
        '    ops=("loop_ops",),\n'
        ')\n'
    )
    # The mode module imports _complex_graph from the test file. But since the test
    # file is not importable from the package, we replicate the graph build inline.
    (pkg / "mode.py").write_text(_COMPLEX_MODE_SOURCE)


_COMPLEX_MODE_SOURCE = '''\
"""Mode for the complex parity scenario."""
from refactory_dsh.nn import Graph, Knob, Mode, Node


class ComplexTestMode(Mode):
    name = "complex"
    description = "complex parity test mode"
    unscored_reason = "it is a test mode, judged by its parity comparison"

    @classmethod
    def declare(cls):
        return {
            "test_knob": Knob("", "str", description="test knob for parity"),
            "name": Knob("", "str", description="name"),
            "brief": Knob("", "str", description="brief"),
            "check": Knob("", "str", description="check"),
        }

    def build(self):
        graph = Graph()
        graph.add(Node("prepare", "op", call="loop_ops:noop",
                        produces=("name", "brief", "check")))
        graph.add(Node("agent1", "agent",
                        prompt_template="Design with knob={{test_knob}}: {{brief}}",
                        writes=("design.md",), produces=("design",)))
        graph.connect("prepare", "agent1")
        graph.add(Node("gate_strategy", "gate", gate_kind="user",
                        reads=("design.md",), max_iterations=5,
                        notes="Approve the strategy"))
        graph.connect("agent1", "gate_strategy")
        graph.add(Node("apply_corrections", "op",
                        call="loop_ops:parity_apply_correction",
                        reads=("design.md",), writes=("design.md",),
                        produces=("corrections_applied",)))
        graph.connect("gate_strategy", "apply_corrections", condition="proceed")
        graph.add(Node("write_relative", "op",
                        call="loop_ops:parity_write_relative",
                        writes=("parity_output.txt",),
                        produces=("ctx_model_is_none", "knob_applied")))
        graph.connect("apply_corrections", "write_relative")
        graph.add(Node("fn_gate", "gate", gate_kind="fn",
                        evaluator="loop_ops:parity_fn_gate",
                        reads=(), max_iterations=3, notes="fn check",
                        produces=("parity_visits",)))
        graph.connect("write_relative", "fn_gate")
        graph.add(Node("agent2", "agent",
                        prompt_template="Continue: {{brief}}",
                        writes=("output.txt",), produces=("result",)))
        graph.connect("fn_gate", "agent2", condition="reloop")
        graph.connect("agent2", "fn_gate")
        graph.add(Node("outer_fork", "fork",
                        targets=("branch_main", "opt_fail", "branch_c"),
                        optional_targets=("opt_fail",)))
        graph.connect("fn_gate", "outer_fork", condition="proceed")
        graph.add(Node("branch_main", "fork",
                        targets=("inner_a", "inner_b")))
        graph.add(Node("inner_a", "op", call="loop_ops:noop",
                        produces=("inner_a_done",)))
        graph.add(Node("inner_b", "op", call="loop_ops:parity_read_knob",
                        produces=("knob_read",)))
        graph.add(Node("inner_join", "join",
                        sources=("inner_a", "inner_b")))
        graph.connect("branch_main", "inner_join")
        graph.connect("inner_a", "inner_join")
        graph.connect("inner_b", "inner_join")
        graph.add(Node("branch_main_end", "op", call="loop_ops:noop",
                        produces=("main_done",)))
        graph.connect("inner_join", "branch_main_end")
        graph.add(Node("opt_fail", "op", call="loop_ops:parity_nested_fork_fail",
                        produces=("opt_result",)))
        graph.add(Node("opt_step2", "op", call="loop_ops:noop"))
        graph.connect("opt_fail", "opt_step2")
        graph.add(Node("branch_c", "op", call="loop_ops:noop",
                        produces=("c_done",)))
        graph.add(Node("outer_join", "join",
                        sources=("branch_main_end", "opt_step2", "branch_c")))
        graph.connect("outer_fork", "outer_join")
        graph.connect("branch_main_end", "outer_join")
        graph.connect("opt_step2", "outer_join")
        graph.connect("branch_c", "outer_join")
        graph.add(Node("done", "op", call="loop_ops:noop"))
        graph.connect("outer_join", "done")
        return graph
'''


def _complex_world(base: Path, gh_shim: Path, answers: list[Any]) -> World:
    """A sealed world for the complex scenario: custom mode, custom script."""
    base.mkdir(parents=True)
    github = base / "github"
    (github / LOGIN).mkdir(parents=True)
    (github / "login").write_text(LOGIN + "\n")
    stub(base / "bin", "gh", f'exec "{sys.executable}" "{FAKE_GH}" "$@"')
    home = base / "dsh-home"
    model_home_with_a_key(home)
    graph_profile(home, REPO_ROOT)
    dsh = base / "dsh"
    dsh.mkdir()
    (dsh / "scenario.json").write_text(json.dumps(_COMPLEX_SCRIPT))
    (dsh / "owner-script.json").write_text(json.dumps({"answers": answers}))
    launcher = dsh / "dsh"
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_DSH}" --scenario "{dsh}" '
                        f'--owner "{OWNER}" --node-bin "{NODE}" "$@"\n')
    launcher.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if key not in ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN",
                          "DSH_CLI", "REFACTORY_WORKDIR", "REFACTORY_PROJECT", "REFACTORY_WORKER_CONTEXT")}
    # Add the tests directory to PYTHONPATH so loop_ops is importable by the spine's
    # op runner and the forkserver child (both resolve "loop_ops:..." at runtime).
    # REFACTORY_PROJECT points at the app directory so the registry finds the workspace
    # package (complex_test) when `refactory ceo --mode` resolves it.
    env.update({
        "PATH": os.pathsep.join([str(base / "bin"), str(gh_shim), os.environ.get("PATH", "")]),
        "FAKE_GH_ROOT": str(github), "GH_CONFIG_DIR": str(base / "gh"),
        "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": f"url.file://{github}/.insteadOf",
        "GIT_CONFIG_VALUE_0": "https://github.com/",
        "DSH_HOME": str(home), "REFACTORY_MODEL_DSH_HOME": str(home),
        "REFACTORY_HOME": str(base / "refactory-home"), "REFACTORY_PROJECTS_DIR": str(base / "projects"),
        "DSH_CLI": str(launcher), "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(HERE) + os.pathsep + env.get("PYTHONPATH", ""),
        "REFACTORY_PROJECT": str(base / "app"),
    })
    # The target repo (needed by ceo for worktree creation)
    repo = git_repo(base / "app")
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
    # Write the complex mode package into the project so the registry can find it.
    _write_complex_mode_package(base / "app")
    return World(base, env)


def _complex_ceo(world: World) -> subprocess.CompletedProcess[str]:
    """Run the complex scenario on the spine via `refactory ceo --mode`."""
    return world.call([REFACTORY, "ceo", "complex_test/complex",
                       "--knob", "test_knob=hello",
                       "--knob", f"name={NAME}", "--knob", f"brief={BRIEF}",
                       "--knob", "check=true",
                       "--workdir", str(world.run),
                       "--headless", "--omit-steering"])


def _complex_spine_resumed(world: World) -> None:
    """The spine: headless run stops at gate_strategy, person steers, resume goes on."""
    done = _complex_ceo(world)
    status = _spine_status(world)
    assert done.returncode != 0 and status == "refused", (
        f"expected refused, got rc={done.returncode} status={status}\n"
        f"stderr={done.stderr[-3000:]}")
    assert "gate_strategy is waiting for a person" in done.stderr, done.stderr[-3000:]
    _steer(world)
    resumed = world.call([REFACTORY, "graph-control", "resume", "--run", str(world.run), "--json"])
    assert resumed.returncode == 0, resumed.stdout[-2000:] + resumed.stderr[-3000:]
    owner = json.loads(resumed.stdout)["pid"]
    started = time.monotonic()
    interval = 0.1
    cap = 300
    while is_alive(owner):
        elapsed = time.monotonic() - started
        assert elapsed < cap, f"the resumed owner did not finish after {elapsed:.0f} s"
        time.sleep(min(interval, 2.0))
        interval = min(interval * 1.5, 2.0)
    assert _spine_status(world) == "done", (world.run / ".refactory/spine-status.json").read_text()


def _complex_walk(world: World) -> None:
    """The walker side of the complex scenario: execute → stop → steer → resume."""
    from refactory_dsh.nn.executor import execute
    graph = _complex_graph()
    knobs = {"test_knob": "hello"}

    class _ScriptedModel:
        def __init__(self, wd):
            self.prompts: list[str] = []
            self._wd = wd
        def complete(self, prompt, **kw):
            self.prompts.append(prompt)
            (Path(self._wd) / "design.md").write_text("# Design\n")
            (Path(self._wd) / "output.txt").write_text("output\n")
            return "scripted answer"
        def route(self):
            return {"worker_backend": "dsh"}
        def context(self):
            return {}
        def for_node(self, node_id, role=None):
            return self

    model = _ScriptedModel(world.run)

    # Run 1: walker stops at the user gate.
    result1 = execute(graph, world.run, model=model, knobs=knobs,
                      task={"name": "test", "brief": "parity test", "check": "true"})
    assert result1.gate_stop == "gate_strategy"
    assert result1.exit_ok is False

    # Steer with the real CLI
    _steer(world)

    # Run 2: resume at the gate.
    result2 = execute(graph, world.run, model=model, knobs=knobs,
                      task={"name": "test", "brief": "parity test", "check": "true"},
                      start="gate_strategy", facts=dict(result1.facts))
    assert result2.exit_ok is True, result2.error


def test_complex_scenario_runtimes_agree(tmp_path, _real_gh_shim):
    """Brief ask 2: the complex two-runtime scenario runs on the spine for real.

    The graph exercises: a NESTED fork (fork inside a fork branch), a two-node
    optional branch whose first node fails, an op writing a relative path and
    reading ctx.model, an fn gate printing to stdout and relooping once, an agent
    gate answered with real `refactory steer`, knobs reaching a prompt and an op,
    stop + resume after the first gate round.

    Walker: execute() → steer CLI → execute() resume (the walker's production path).
    Spine: `refactory ceo --mode complex_test/complex` → steer → graph-control resume
    (the same _ceo / _spine_resumed flow the card scenario uses).

    Both sides are observed with observe(), then compared through
    differences()/unexplained() for the aspects both runtimes produce through
    the same code paths: facts, artifacts, and verdicts.  Agent prompts and
    routes are collected by observe() but not compared because the walker uses
    a scripted in-process model while the spine uses the DSH model shim, making
    model-generated content differ by construction.  SCENARIO_GAPS["complex"]
    is empty.
    """
    shim = _real_gh_shim[0]
    complex_graph = _complex_graph()

    # ── Walker side ──────────────────────────────────────────────────
    walker_world = _complex_world(tmp_path / "walker", shim, [])
    _complex_walk(walker_world)

    # ── Spine side ───────────────────────────────────────────────────
    spine_world = _complex_world(tmp_path / "spine", shim, [])
    _complex_spine_resumed(spine_world)

    # ── Compare via observe/differences/unexplained ────────────────
    walker_obs = observe(walker_world, "walker", tmp_path, graph=complex_graph)
    spine_obs = observe(spine_world, "spine", tmp_path, graph=complex_graph)

    # Op-produced facts and op-produced artifacts must agree; agent-produced
    # content (design.md, output.txt, model-set facts) differs because the
    # walker uses a scripted model while the spine uses the DSH shim.
    op_fact_keys = {"ctx_model_is_none", "knob_applied", "parity_visits",
                    "corrections_applied", "knob_read"}
    op_artifact_keys = {"parity_output.txt"}
    walker_compared: dict[str, dict[str, str]] = {
        "facts": {k: v for k, v in walker_obs["facts"].items() if k in op_fact_keys},
        "artifacts": {k: v for k, v in walker_obs["artifacts"].items() if k in op_artifact_keys},
        "verdicts": walker_obs["verdicts"],
    }
    spine_compared: dict[str, dict[str, str]] = {
        "facts": {k: v for k, v in spine_obs["facts"].items() if k in op_fact_keys},
        "artifacts": {k: v for k, v in spine_obs["artifacts"].items() if k in op_artifact_keys},
        "verdicts": spine_obs["verdicts"],
    }
    found = differences(walker_compared, spine_compared)
    gaps = [GAPS[name] for name in SCENARIO_GAPS["complex"] if name in GAPS]
    remaining = unexplained(found, gaps)
    assert not remaining, (
        f"complex scenario: {len(remaining)} unexplained difference(s):\n"
        + "\n".join(str(d) for d in remaining)
    )

    # ── Walker-side verifications ────────────────────────────────────
    walker_facts_raw = _facts(walker_world.run, "walker")
    assert walker_facts_raw.get("ctx_model_is_none") is True
    assert walker_facts_raw.get("knob_applied") == "hello"
    assert walker_facts_raw.get("knob_read") == "hello"
    assert walker_facts_raw.get("parity_visits") == 2
    corrections = walker_facts_raw.get("corrections_applied", [])
    assert len(corrections) == 1
    assert "NAME constant" in corrections[0]
    assert (walker_world.run / "parity_output.txt").exists()
    assert "opt_result" not in walker_facts_raw


def test_halt_inside_fork_branch_both_runtimes(tmp_path, _real_gh_shim):
    """Brief ask 3: a halt inside a fork branch is recorded as halted on both runtimes.

    Two-runtime equivalence test: drives a fork graph with a halt gate through both
    the walker (execute) and the spine (refactory ceo → lowering → graph-owner.mjs).
    Both must record the run as halted.
    """
    from refactory_dsh.nn import Graph, Node
    from refactory_dsh.nn.executor import execute

    # Graph: prepare → fork → halt_branch (fn gate returning halt) | ok_branch (op) → join → done
    graph = Graph()
    graph.add(Node("prepare", "op", call="loop_ops:noop"))
    graph.add(Node("fork1", "fork", targets=("halt_branch", "ok_branch")))
    graph.connect("prepare", "fork1")
    graph.add(Node("halt_branch", "gate", gate_kind="fn",
                    evaluator="loop_ops:always_halt", max_iterations=1))
    graph.add(Node("ok_branch", "op", call="loop_ops:noop"))
    graph.add(Node("join1", "join", sources=("halt_branch", "ok_branch")))
    graph.connect("fork1", "join1")
    graph.connect("halt_branch", "join1", condition="proceed")
    graph.connect("ok_branch", "join1")
    graph.add(Node("done", "op", call="loop_ops:noop"))
    graph.connect("join1", "done")

    # ── Walker side ──────────────────────────────────────────────────
    walker_dir = tmp_path / "walker"
    result = execute(graph, walker_dir)
    assert result.halted is True, (
        f"Walker: expected halted=True, got halted={result.halted}, error={result.error!r}")
    status_file = walker_dir / ".refactory" / "spine-status.json"
    if status_file.exists():
        walker_status = json.loads(status_file.read_text()).get("status", "")
        assert walker_status == "halted", f"Walker status file says {walker_status!r}"

    # ── Spine side ───────────────────────────────────────────────────
    # Write the halt mode as a workspace package so `refactory ceo --mode` can run it.
    shim = _real_gh_shim[0]
    spine_base = tmp_path / "spine"
    spine_base.mkdir(parents=True)
    github = spine_base / "github"
    (github / LOGIN).mkdir(parents=True)
    (github / "login").write_text(LOGIN + "\n")
    stub(spine_base / "bin", "gh", f'exec "{sys.executable}" "{FAKE_GH}" "$@"')
    home = spine_base / "dsh-home"
    model_home_with_a_key(home)
    graph_profile(home, REPO_ROOT)
    dsh = spine_base / "dsh"
    dsh.mkdir()
    (dsh / "scenario.json").write_text("{}")
    (dsh / "owner-script.json").write_text(json.dumps({"answers": []}))
    launcher = dsh / "dsh"
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_DSH}" --scenario "{dsh}" '
                        f'--owner "{OWNER}" --node-bin "{NODE}" "$@"\n')
    launcher.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if key not in ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN",
                          "DSH_CLI", "REFACTORY_WORKDIR", "REFACTORY_PROJECT", "REFACTORY_WORKER_CONTEXT")}
    env.update({
        "PATH": os.pathsep.join([str(spine_base / "bin"), str(shim), os.environ.get("PATH", "")]),
        "FAKE_GH_ROOT": str(github), "GH_CONFIG_DIR": str(spine_base / "gh"),
        "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": f"url.file://{github}/.insteadOf",
        "GIT_CONFIG_VALUE_0": "https://github.com/",
        "DSH_HOME": str(home), "REFACTORY_MODEL_DSH_HOME": str(home),
        "REFACTORY_HOME": str(spine_base / "refactory-home"),
        "REFACTORY_PROJECTS_DIR": str(spine_base / "projects"),
        "DSH_CLI": str(launcher), "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(HERE) + os.pathsep + env.get("PYTHONPATH", ""),
        "REFACTORY_PROJECT": str(spine_base / "app"),
    })
    repo = git_repo(spine_base / "app")
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

    # Write the halt mode package
    pkg = spine_base / "app" / ".factory" / "packages" / "halt_test"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(
        'from refactory_dsh.nn import Package\n'
        'from .mode import HaltTestMode\n\n'
        'PACKAGE = Package(\n'
        '    name="halt_test",\n'
        '    description="test mode for halt-in-fork equivalence",\n'
        '    modes=(HaltTestMode,),\n'
        '    ops=("loop_ops",),\n'
        ')\n'
    )
    (pkg / "mode.py").write_text(
        'from refactory_dsh.nn import Graph, Mode, Node\n\n'
        'class HaltTestMode(Mode):\n'
        '    name = "halt"\n'
        '    description = "halt inside fork branch"\n'
        '    unscored_reason = "test mode"\n\n'
        '    def build(self):\n'
        '        graph = Graph()\n'
        '        graph.add(Node("prepare", "op", call="loop_ops:noop"))\n'
        '        graph.add(Node("fork1", "fork", targets=("halt_branch", "ok_branch")))\n'
        '        graph.connect("prepare", "fork1")\n'
        '        graph.add(Node("halt_branch", "gate", gate_kind="fn",\n'
        '                        evaluator="loop_ops:always_halt", max_iterations=1))\n'
        '        graph.add(Node("ok_branch", "op", call="loop_ops:noop"))\n'
        '        graph.add(Node("join1", "join", sources=("halt_branch", "ok_branch")))\n'
        '        graph.connect("fork1", "join1")\n'
        '        graph.connect("halt_branch", "join1", condition="proceed")\n'
        '        graph.connect("ok_branch", "join1")\n'
        '        graph.add(Node("done", "op", call="loop_ops:noop"))\n'
        '        graph.connect("join1", "done")\n'
        '        return graph\n'
    )

    spine_world = World(spine_base, env)
    done = spine_world.call([REFACTORY, "ceo", "halt_test/halt",
                             "--workdir", str(spine_world.run),
                             "--headless", "--omit-steering"])
    spine_status = _spine_status(spine_world)
    assert spine_status == "halted", (
        f"Spine: expected halted, got {spine_status!r}; rc={done.returncode}\n"
        f"stderr={done.stderr[-3000:]}")

    # Both runtimes agree: the run is halted.


def test_correction_via_steer_subprocess(tmp_path, _real_gh_shim):
    """Brief ask 4 / correction 1: drive `refactory steer` as a subprocess on both runtimes.

    The walker side runs through execute() → subprocess steer → execute() resume.
    The spine side drives `refactory steer` as a subprocess against a spine run
    (the _ceo / _spine_resumed flow).
    Both sides verify the correction reaches apply_corrections exactly once.
    """
    from refactory_dsh.nn.executor import execute
    shim = _real_gh_shim[0]
    graph = _complex_graph()
    knobs = {"test_knob": "hello"}

    # ── Walker side ──────────────────────────────────────────────────
    walker_world = _complex_world(tmp_path / "walker", shim, [])

    class _ScriptedModel:
        def __init__(self, wd):
            self.prompts: list[str] = []
            self._wd = wd
        def complete(self, prompt, **kw):
            self.prompts.append(prompt)
            (Path(self._wd) / "design.md").write_text("# Design\n")
            (Path(self._wd) / "output.txt").write_text("output\n")
            return "scripted answer"
        def route(self):
            return {"worker_backend": "dsh"}
        def context(self):
            return {}
        def for_node(self, node_id, role=None):
            return self

    model = _ScriptedModel(walker_world.run)

    # Run 1: walker stops at user gate.
    result1 = execute(graph, walker_world.run, model=model, knobs=knobs,
                      task={"name": "test", "brief": "parity test", "check": "true"})
    assert result1.gate_stop == "gate_strategy"

    # Steer with real subprocess
    steer_done = subprocess.run(
        [sys.executable, "-m", "refactory_dsh.console", "steer",
         "--run", str(walker_world.run),
         "--verdict", "gate_strategy=proceed",
         "--kind", "correction",
         "--note", "Add a parity constant."],
        capture_output=True, text=True, timeout=30)
    assert steer_done.returncode == 0, steer_done.stderr[-2000:]

    # Run 2: resume at the gate — correction applied exactly once.
    result2 = execute(graph, walker_world.run, model=model, knobs=knobs,
                      task={"name": "test", "brief": "parity test", "check": "true"},
                      start="gate_strategy", facts=dict(result1.facts))
    assert result2.exit_ok is True, result2.error

    # Correction applied exactly once on walker.
    walker_corrections = result2.facts.get("corrections_applied", [])
    assert len(walker_corrections) == 1, (
        f"walker: expected 1 correction, got {len(walker_corrections)}: {walker_corrections}")
    assert "parity constant" in walker_corrections[0]

    # ── Spine side ───────────────────────────────────────────────────
    spine_world = _complex_world(tmp_path / "spine", shim, [])
    _complex_spine_resumed(spine_world)

    spine_facts = json.loads((spine_world.run / "facts.json").read_text())
    spine_corrections = spine_facts.get("corrections_applied", [])
    assert len(spine_corrections) == 1, (
        f"spine: expected 1 correction, got {len(spine_corrections)}: {spine_corrections}")
    assert "NAME constant" in spine_corrections[0]


def test_forkserver_child_imports_test_ops(tmp_path):
    """Correction 2: prove the forkserver child can import test ops.

    Runs a test-module op (loop_ops:noop) in the forkserver child on the default
    walker path. The op is in refactory/tests/loop_ops.py, which is on sys.path
    because conftest.py adds it.  The forkserver child receives the parent's
    sys.path through _op_child_target, so it can import loop_ops without
    relying on parent sys.path changes made after the forkserver starts.

    Fails if the child cannot import the test op module.
    """
    import os as _os
    from refactory_dsh.nn import Graph, Node
    from refactory_dsh.nn.executor import execute

    graph = Graph()
    graph.add(Node("test_op", "op", call="loop_ops:pid_reporting_gate",
                    produces=("gate_pid",)))

    result = execute(graph, tmp_path)
    assert result.exit_ok, result.error
    # The op ran in a forkserver child (different PID from parent).
    gate_pid = result.facts.get("gate_pid")
    assert gate_pid is not None, "test op did not run"
    assert gate_pid != _os.getpid(), (
        f"test op ran in-process (PID {gate_pid} == parent {_os.getpid()}); "
        "expected forkserver child"
    )


# ── Ask 6: data failure scenario equivalence ─────────────────────────────────

#: The failure-scenario Task as a standalone module, written to a temp directory
#: so both in-process (walker) and subprocess (spine → items.py) can import it.
_FAILURE_TASK_SOURCE = '''\
"""A Task for the data failure equivalence scenario."""
import time
from pathlib import Path
from refactory_dsh.task import Task, TaskInstance, VerifyResult

class FailureScenarioTask(Task):
    name = "failure_scenario"

    def instances(self):
        return [
            TaskInstance(id=""),          # no id — exercises the id fallback
            TaskInstance(id="raise_item"),
            TaskInstance(id="timeout_item"),
            TaskInstance(id="setup_fails"),
            TaskInstance(id="last_fails"),
        ]

    def setup(self, instance, workspace):
        if instance.id == "setup_fails":
            raise ValueError("setup failed for setup_fails")

    def verify(self, instance, workspace):
        if instance.id == "raise_item":
            raise ValueError("verify raised")
        if instance.id == "timeout_item":
            time.sleep(30)  # hangs past the per-item deadline
        return VerifyResult(passed=True, score=1.0)
'''

#: Task where every item fails — for the all-items-failed equivalence case.
_ALL_FAIL_TASK_SOURCE = '''\
"""A Task where every item fails, for the all-items-failed equivalence case."""
from refactory_dsh.task import Task, TaskInstance, VerifyResult

class AllFailTask(Task):
    name = "all_fail_equiv"

    def instances(self):
        return [
            TaskInstance(id="fail_a"),
            TaskInstance(id="fail_b"),
        ]

    def verify(self, instance, workspace):
        raise ValueError(f"{instance.id} verify failed")
'''

#: The subgraph op: a Python callable the walker runs in-process and the spine
#: runs as a command.  Reads current_item.json and appends its id to seen.json.
_FAILURE_OP_SOURCE = '''\
"""Op for the data failure equivalence scenario."""
import json
from pathlib import Path

def record(ctx):
    item = ctx.read_json("current_item.json", {}) or {}
    seen_path = Path(ctx.workdir) / "seen.json"
    seen = json.loads(seen_path.read_text()) if seen_path.exists() else {"ids": []}
    seen["ids"].append(item.get("id", ""))
    ctx.write_json("seen.json", seen)
    ctx.set_fact("value", seen)
    if item.get("id") == "last_fails":
        raise RuntimeError("op failed for last_fails")
'''

DATA_DRAIN = REPO_ROOT / "packages/workflow/workflow-spine/tests/fixtures/data-drain.mts"


def _write_failure_package(project: Path) -> None:
    """Write the failure-scenario Task and op as a proper package."""
    pkg = project / ".factory" / "packages" / "failure_scenario"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(
        'from refactory_dsh.nn import Package\n'
        'from .task import FailureScenarioTask\n'
        'from .all_fail_task import AllFailTask\n'
        'PACKAGE = Package(\n'
        '    name="failure_scenario",\n'
        '    description="data failure scenario",\n'
        '    modes=(),\n'
        '    ops=("refactory_packages.failure_scenario.ops",),\n'
        ')\n'
    )
    (pkg / "task.py").write_text(_FAILURE_TASK_SOURCE)
    (pkg / "all_fail_task.py").write_text(_ALL_FAIL_TASK_SOURCE)
    (pkg / "ops.py").write_text(_FAILURE_OP_SOURCE)


def _run_spine(graph_json: Path, workdir: Path, python_bin_dir: str, timeout_ms: int) -> int:
    """Drive the TS spine drain (index.ts) via data-drain.mts."""
    env = {**os.environ, "REFACTORY_WORKDIR": str(workdir)}
    # The lowered graph's node commands use bare `python`, so the venv bin
    # directory must lead PATH — matching what graph-startup does.
    env["PATH"] = f"{python_bin_dir}:{env.get('PATH', '')}"
    result = subprocess.run(
        [str(NODE), "--import", "tsx/esm", str(DATA_DRAIN),
         str(graph_json), str(workdir), f"{python_bin_dir}/python", str(timeout_ms)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120,
        env=env,
    )
    if result.returncode != 0:
        sys.stderr.write(f"spine drain failed:\nstdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}\n")
    return result.returncode


def test_data_failure_scenario_matches(tmp_path):
    """Walker and spine agree on item_scores for a data node with failures.

    Drives the real spine (lowering + the TS drain in index.ts, the way
    test_equivalence.py's data-node test runs DSH).  Items include one without
    an id (exercises the id fallback), one whose verify raises, one whose verify
    hangs past the deadline (error_kind "timeout"), and one (the LAST) whose
    subgraph op fails.  Both runtimes produce the same ids, passed, score and
    error_kind values.  A rerun in the same directory produces the same result
    (no stale entries from the previous item set).  No SCENARIO_GAPS entry.
    """
    from unittest.mock import patch
    from refactory_dsh import lowering, nn
    from refactory_dsh.nn.executor import CURRENT_ITEM, Executor, execute
    from refactory_dsh.task import register_task

    # ── importable Task package ──
    project = tmp_path / "project"
    _write_failure_package(project)
    sys.path.insert(0, str(project / ".factory" / "packages"))
    try:
        # Import so the walker can use the registered name too.
        from failure_scenario.task import FailureScenarioTask  # type: ignore[import-not-found]
        register_task("failure_scenario", FailureScenarioTask)

        task_ref = "refactory_packages.failure_scenario.task:FailureScenarioTask"

        graph = nn.Graph()
        graph.add(nn.Node(
            "data", "data", subgraph_entry="work", subgraph_exit="work",
            task_ref="failure_scenario",
            writes=(CURRENT_ITEM,), produces=("item_scores",),
        ))
        graph.add(nn.Node(
            "work", "op", call="loop_ops:fail_on_last_fails",
            reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
        ))
        graph.add(nn.Node(
            "report", "op", call="loop_ops:write_best",
            writes=("best.json",), produces=("best",),
        ))
        graph.connect("data", "report")

        # ── Walker run (deadline monkey-patched to 1 s) ──
        walker_dir = tmp_path / "walker"
        with patch.object(Executor, "_data_item_deadline", return_value=1.0):
            walker_result = execute(graph, walker_dir)
        assert walker_result.exit_ok, walker_result.error
        walker_scores = walker_result.facts.get("item_scores", [])

        # ── Spine run (real TS drain via data_drain_fixture.mts) ──
        spine_dir = tmp_path / "spine"
        spine_dir.mkdir(parents=True, exist_ok=True)

        # The spine graph uses a command, not a call.  Build a graph whose work
        # node has a command that calls the op through refactory_dsh.run (as
        # the spine runs FnNodes).
        spine_graph = nn.Graph()
        spine_graph.add(nn.Node(
            "data", "data", subgraph_entry="work", subgraph_exit="work",
            task_ref=task_ref,
            writes=(CURRENT_ITEM,), produces=("item_scores",),
        ))
        spine_graph.add(nn.Node(
            "work", "op", call="loop_ops:fail_on_last_fails",
            reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
        ))
        spine_graph.add(nn.Node(
            "report", "op", call="loop_ops:write_best",
            writes=("best.json",), produces=("best",),
        ))
        spine_graph.connect("data", "report")

        state = {"graph": spine_graph.to_state()}
        lowering.write_declarations(state, spine_dir)
        graph_path, workflow = lowering.write_mode_graph(
            state, tmp_path / "graphs", name="failure_scenario",
        )

        python_bin = str(REPO_ROOT / ".venv/bin")
        # The subprocesses (items.py, refactory_dsh.run) need to import
        # the Task package (refactory_packages.failure_scenario.*) and
        # the test ops (loop_ops).  REFACTORY_PROJECT lets the runner's
        # install_workspace_packages find the package; the tests dir on
        # PYTHONPATH makes loop_ops importable the way pytest does.
        extra_path = os.pathsep.join([
            str(project / ".factory" / "packages"),
            str(HERE),
        ])
        env_patch = {
            "REFACTORY_PROJECT": str(project),
            "PYTHONPATH": extra_path,
        }
        old_env = {}
        for key, value in env_patch.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value
        try:
            # 2 s per command: enough for normal begin/end, but timeout_item's
            # verify (30 s sleep) will exceed it, producing error_kind "timeout".
            rc = _run_spine(graph_path, spine_dir, python_bin, 2_000)
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        assert rc == 0, "spine drain failed"
        spine_facts = json.loads((spine_dir / "facts.json").read_text())
        spine_scores = spine_facts.get("item_scores", [])

        # ── Compare ──
        def normalize(scores):
            return sorted(
                [(s["id"], s["passed"], s["score"], s.get("error_kind"))
                 for s in scores],
                key=lambda x: x[0],
            )

        assert normalize(walker_scores) == normalize(spine_scores), (
            f"walker and spine disagree:\nwalker: {walker_scores}\nspine: {spine_scores}"
        )

        # ── Rerun in same directories — no stale entries ──
        with patch.object(Executor, "_data_item_deadline", return_value=1.0):
            walker_result2 = execute(graph, walker_dir)
        assert walker_result2.exit_ok, walker_result2.error
        walker_scores2 = walker_result2.facts.get("item_scores", [])

        for key, value in env_patch.items():
            os.environ[key] = value
        try:
            rc2 = _run_spine(graph_path, spine_dir, python_bin, 2_000)
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        assert rc2 == 0, "spine drain rerun failed"
        spine_facts2 = json.loads((spine_dir / "facts.json").read_text())
        spine_scores2 = spine_facts2.get("item_scores", [])

        assert normalize(walker_scores2) == normalize(spine_scores2), (
            f"rerun mismatch:\nwalker: {walker_scores2}\nspine: {spine_scores2}"
        )
        assert len(spine_scores2) == len(walker_scores2) == 5, (
            f"rerun should have exactly 5 entries (no stale): {len(spine_scores2)}"
        )

        # ── Ask 6: setup_fails item should have an error record ──
        setup_fail_walker = next(
            (s for s in walker_scores if s["id"] == "setup_fails"), None
        )
        setup_fail_spine = next(
            (s for s in spine_scores if s["id"] == "setup_fails"), None
        )
        assert setup_fail_walker is not None, "walker missing setup_fails"
        assert setup_fail_spine is not None, "spine missing setup_fails"
        assert setup_fail_walker["passed"] is False
        assert setup_fail_spine["passed"] is False
        # Both should have an error_kind for the setup failure
        assert setup_fail_walker.get("error_kind") is not None
        assert setup_fail_spine.get("error_kind") is not None

        # ── Ask 6: all-items-failed sub-case ──
        from failure_scenario.all_fail_task import AllFailTask  # type: ignore[import-not-found]
        register_task("all_fail_equiv", AllFailTask)

        all_fail_task_ref = "refactory_packages.failure_scenario.all_fail_task:AllFailTask"

        af_graph = nn.Graph()
        af_graph.add(nn.Node(
            "data", "data", subgraph_entry="work", subgraph_exit="work",
            task_ref="all_fail_equiv",
            writes=(CURRENT_ITEM,), produces=("item_scores",),
        ))
        af_graph.add(nn.Node(
            "work", "op", call="loop_ops:record_item",
            reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
        ))
        af_graph.add(nn.Node(
            "report", "op", call="loop_ops:write_best",
            writes=("best.json",), produces=("best",),
        ))
        af_graph.connect("data", "report")

        # Walker: should fail (all items failed)
        af_walker_dir = tmp_path / "af_walker"
        af_walker_result = execute(af_graph, af_walker_dir)
        assert af_walker_result.exit_ok is False, "walker should fail when all items fail"
        assert "all" in (af_walker_result.error or "").lower()

        # Spine: should fail
        af_spine_dir = tmp_path / "af_spine"
        af_spine_dir.mkdir(parents=True, exist_ok=True)

        af_spine_graph = nn.Graph()
        af_spine_graph.add(nn.Node(
            "data", "data", subgraph_entry="work", subgraph_exit="work",
            task_ref=all_fail_task_ref,
            writes=(CURRENT_ITEM,), produces=("item_scores",),
        ))
        af_spine_graph.add(nn.Node(
            "work", "op", call="loop_ops:record_item",
            reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
        ))
        af_spine_graph.add(nn.Node(
            "report", "op", call="loop_ops:write_best",
            writes=("best.json",), produces=("best",),
        ))
        af_spine_graph.connect("data", "report")

        af_state = {"graph": af_spine_graph.to_state()}
        lowering.write_declarations(af_state, af_spine_dir)
        af_graph_path, _ = lowering.write_mode_graph(
            af_state, tmp_path / "af_graphs", name="all_fail",
        )

        for key, value in env_patch.items():
            os.environ[key] = value
        try:
            af_rc = _run_spine(af_graph_path, af_spine_dir, python_bin, 2_000)
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        # The spine should also fail (non-zero exit)
        assert af_rc != 0, "spine should fail when all items fail"

    finally:
        sys.path.remove(str(project / ".factory" / "packages"))
