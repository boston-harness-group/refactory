"""Helpers and fixtures the delivery tests share (`test_delivery*.py`).

The tests were one file; they are split by theme so CI can run them on
separate workers. Nothing here is collected.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest

import child_patches
import delivery_fakes
from refactory_dsh import cli, console, delivery, nn, projects, worktree
from refactory_dsh.design import DesignMode
from refactory_dsh.projects import projects_dir
from tests_support import DESIGN_DOC_ANSWER, PITFALLS_ANSWER, RESEARCH_ANSWER, VERIFICATION_ANSWER, REVIEW_ANSWERS, FakeModel, git, git_repo, overwatch_judgment, qa_judgment, render, stub

NAME = "delivered_tool"
FAKE_GH = Path(__file__).resolve().parent / "fake_gh.py"
LOGIN = "octo"
#: The running test's fakes for the op children (`child_patches`).
_CHILD: list[child_patches.ChildPatches] = []


@pytest.fixture(autouse=True)
def _child_patches(monkeypatch):
    """Each test's fakes of what an op calls reach the child the op runs in."""
    _CHILD[:] = [child_patches.for_test(monkeypatch)]
    yield
    _CHILD.clear()


def _child() -> child_patches.ChildPatches:
    """The running test's :class:`child_patches.ChildPatches`."""
    return _CHILD[0]


def _unfake(target, name: str) -> None:
    """Put ``target.name`` back as it was before any fake, here and in every op child."""
    _child().setattr(target, name, delivery_fakes.as_real, _module=target.__name__, _name=name)


# ── PR title and body unit tests ───────────────────────────────────────────


class _TitleCtx:
    """Minimal context for _title() and _body() unit tests."""
    def __init__(self, facts=None, files=None, workdir=None):
        self.facts = dict(facts or {})
        self._files = dict(files or {})
        self.workdir = Path(workdir) if workdir else None
    def exists(self, name):
        return name in self._files
    def read_text(self, name):
        return self._files[name]
    def read_json(self, name, default=None):
        import json as _j
        if name not in self._files:
            return default
        try:
            return _j.loads(self._files[name])
        except (ValueError, TypeError):
            return default


def _git_pair(tmp_path):
    """Set up a minimal two-commit git repo; return (base, head, path)."""
    import subprocess
    subprocess.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], capture_output=True, check=True)
    (tmp_path / "f.txt").write_text("a\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], capture_output=True, check=True)
    base = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / "g.txt").write_text("b\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "work"], capture_output=True, check=True)
    head = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    return base, head


# ── fixture-based marker tests ─────────────────────────────────────────────

_FIXTURES = Path(__file__).resolve().parent / "fixtures"




# ── dm-108: delivery recognises its own merge ─────────────────────────────


def _git_init(path):
    """Initialise a git repo with user config at ``path`` and return the path."""
    import subprocess as sp
    path.mkdir(parents=True, exist_ok=True)
    sp.run(["git", "init", str(path)], capture_output=True, check=True)
    sp.run(["git", "-C", str(path), "config", "user.email", "t@t"], capture_output=True, check=True)
    sp.run(["git", "-C", str(path), "config", "user.name", "T"], capture_output=True, check=True)
    return path


def _make_github(base: Path) -> Path:
    """GitHub under ``base``: gh (logged in as `octo`) in ``base/bin``, repositories in ``base/github``."""
    root = base / "github"
    root.mkdir()
    (root / "login").write_text(LOGIN + "\n")
    stub(base / "bin", "gh", f'exec "{sys.executable}" "{FAKE_GH}" "$@"')
    return root


def _use_github(monkeypatch: pytest.MonkeyPatch, base: Path) -> Path:
    """Point gh and git at the GitHub ``_make_github`` made under ``base``."""
    root = base / "github"
    monkeypatch.setenv("PATH", f"{base / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_GH_ROOT", str(root))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", f"url.file://{root}/.insteadOf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "https://github.com/")
    child_patches.for_test(monkeypatch).setattr(delivery, "VERIFY_WAIT_S", 0)
    return root


@pytest.fixture
def github(tmp_path, monkeypatch) -> Path:
    """GitHub, as far as delivery can tell: gh logged in as `octo`, repositories on disk."""
    _make_github(tmp_path)
    return _use_github(monkeypatch, tmp_path)


def _published(github: Path, slug: str, source: Path) -> Path:
    """A repository on GitHub holding ``source``'s main, which ``source`` names as origin."""
    bare = github / f"{slug}.git"
    bare.parent.mkdir(parents=True, exist_ok=True)
    git(bare.parent, "init", "-q", "--bare", "-b", "main", bare.name)
    (github / f"{slug}.json").write_text(json.dumps({"visibility": "private", "default_branch": "main"}))
    # Pushed before origin is added: from git 2.56 a push to a URL that matches a configured
    # remote also updates its tracking refs, and the person's clone here has none.
    git(source, "push", "-q", f"https://github.com/{slug}.git", "main:refs/heads/main")
    git(source, "remote", "add", "origin", f"https://github.com/{slug}.git")
    return bare


def _remote_head(github: Path, slug: str, branch: str) -> str:
    try:
        return git(github / f"{slug}.git", "rev-parse", "-q", "--verify", f"refs/heads/{branch}")
    except subprocess.CalledProcessError:
        return ""


def _pulls(github: Path, slug: str) -> list[dict]:
    path = github / f"{slug}.pulls.json"
    return json.loads(path.read_text()) if path.exists() else []


def _describe(github: Path, slug: str, body: str) -> None:
    """Give the one pull request on ``slug`` the description ``body``, as its author would on GitHub."""
    path = github / f"{slug}.pulls.json"
    [pull] = json.loads(path.read_text())
    path.write_text(json.dumps([{**pull, "body": body}]))


def _calls(tmp_path: Path) -> str:
    log = tmp_path / "bin" / "calls.log"
    return log.read_text() if log.exists() else ""


def _answers(*emits: str) -> list[str]:
    """Every answer a design run needs, in the order its graph asks.

    pick_corrections is a gate (no model answer needed).  When gate_corrections
    is empty (no ledger or no corrections), pick_corrections returns "skip"
    and apply_corrections does not run — no model answer consumed.
    """
    return [RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
            DESIGN_DOC_ANSWER, *emits,
            *REVIEW_ANSWERS[:3], VERIFICATION_ANSWER, REVIEW_ANSWERS[3]]


def _mode(**knobs) -> DesignMode:
    """Design mode with some of its knobs set, as a person sets them for one run."""
    mode = DesignMode()
    state = mode.state_dict()
    for name, value in knobs.items():
        state["knobs"][name]["value"] = value
    mode.load_state_dict(state)
    return mode


def _design(workdir: Path, emit: dict[str, str] | None = None, *, mode: DesignMode | None = None,
            check: str = "python tool.py", **task: str):
    emitted = render(emit or {"tool.py": "print('built')\n"})
    # A refine designs straight from the pull request: no research, pitfalls or
    # strategy answers.
    answers = (_answers(emitted)[4:] if "refine" in task else _answers(emitted))
    model = FakeModel(answers)
    used_mode = mode or DesignMode()
    return used_mode.run(
        workdir, model=model, verdicts={"gate_strategy": "proceed"},
        task={"name": NAME, "brief": "Build a tool\nthat prints built.", "check": check, **task},
    )


def _record(workdir: Path) -> dict:
    return json.loads((workdir / delivery.RECORD).read_text())


# ── draft delivery tests ──────────────────────────────────────────────────

DRAFT_FACT = {
    "reason": "review reloop limit reached after 3 review cycle(s)",
    "unresolved": [
        {"text": "The logging format is inconsistent", "source": "review", "rounds": [1, 2, 3]},
        {"text": "Missing error handling in parser", "source": "overwatch", "rounds": [2]},
    ],
}


# ── b45a: every run with an applied commit ends in a pull request ──────────
#
# The build loop's limit stops route into a draft (b40b, dm-224), each by its own way
# (correction 2): a check still failing at its limit goes straight to the report, a
# review limit at gate_qa goes on through the correction audit and the overwatch, and a
# limit at gate_overwatch or the refusal bound goes to the report. Every run here walks
# the real design graph with a scripted model, and delivers to the fake GitHub.

#: What the QA judge and the overwatch hold the change back for, and the line each is
#: listed as in the draft (`judges.must_fix_line`).
QA_HELD = {"source": "qa", "where": "tool.py:1", "what": "no test shows the tool prints built",
           "why": "the brief's only ask", "severity": "high"}
QA_HELD_LINE = "[qa] tool.py:1: no test shows the tool prints built (why: the brief's only ask) (high)"
OVERWATCH_HELD = {"source": "overwatch", "where": "tool.py:1", "what": "nothing a user runs shows it",
                  "why": "the brief's ask", "severity": "high"}
OVERWATCH_HELD_LINE = "[overwatch] tool.py:1: nothing a user runs shows it (why: the brief's ask) (high)"
#: An emission materialise refuses: an edit to a file the last attempt does not have.
REFUSED = "===EDIT: missing.py===\n<<<<<<< SEARCH\nnot there\n=======\nx\n>>>>>>> REPLACE\n"
BUILT = render({"tool.py": "print('built')\n"})
QA_JUDGE_MARK = "You are the QA judge."
OVERWATCH_JUDGE_MARK = "Write `overwatch-verdict.json`"


def _attempt(emission: str, *, judge: str | None = None, overwatch: str | None = None) -> list[str]:
    """One attempt's answers: the emission, the reviews and the verifier beside the check,
    then the judges' verdicts (the QA judge proceeds and the overwatch passes when not given)."""
    return [emission, *REVIEW_ANSWERS[:3], VERIFICATION_ANSWER,
            *([judge] if judge else []), *([overwatch] if overwatch else [])]


def _limited(run: Path, *attempts: list[str], check: str = "python tool.py",
             **knobs: int) -> tuple[nn.RunResult, FakeModel]:
    """A new project's design run through ``attempts``, with ``knobs`` set."""
    model = FakeModel([RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan", DESIGN_DOC_ANSWER,
                       *(answer for attempt in attempts for answer in attempt)])
    result = _mode(**knobs).run(run, model=model, verdicts={"gate_strategy": "proceed"},
                                task={"name": NAME, "brief": "Build a tool\nthat prints built.",
                                      "check": check})
    return result, model


def _asked(model: FakeModel, mark: str) -> int:
    """How many prompts carried ``mark``."""
    return sum(1 for prompt in model.prompts if mark in prompt)


def _draft_pull(github: Path) -> dict:
    [pull] = _pulls(github, f"{LOGIN}/{NAME}")
    return pull


#: A person's description: prose with no heading above it, and one with no heading at all.
PERSON_PROSE = "Why this matters, in my words, above everything else.\n\n## Context\n\nKept as written.\n"
PERSON_HEADINGLESS = "Just my paragraph.\nA second line of it, and no heading anywhere.\n"


def _section(body: str) -> str:
    """The sections delivery wrote, between its markers."""
    return body[body.index(delivery.BODY_START):body.index(delivery.BODY_END) + len(delivery.BODY_END)]


def _changed_pair(tmp_path, *names: str) -> tuple[str, str]:
    """`_git_pair`'s repository with ``names`` committed on top: (base, head)."""
    base, _ = _git_pair(tmp_path)
    for name in names:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(f"{name}\n")
        git(tmp_path, "add", name)
    git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "work")
    return base, git(tmp_path, "rev-parse", "HEAD")


def _a_body(tmp_path, design: str, *names: str) -> str:
    from types import SimpleNamespace
    base, head = _changed_pair(tmp_path, *names)
    ctx = _TitleCtx(facts={"brief": "b", "check": "pytest", "name": "b45a"}, files={"design.md": design})
    return delivery._body(ctx, SimpleNamespace(path=str(tmp_path), base=base, head=head, branch="b"))


def _repo_pair(tmp_path, base_files: dict[str, str], head_files: dict[str, str],
               removed: tuple[str, ...] = ()) -> tuple[Path, str, str]:
    """A repository whose base commit holds ``base_files`` and whose head writes
    ``head_files`` and removes ``removed``: (repository, base, head)."""
    repo = tmp_path / "r"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    for files in (base_files, head_files):
        if files is head_files:
            for name in removed:
                git(repo, "rm", "-q", name)
        for name, text in files.items():
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_text(text)
        git(repo, "add", "-A")
        git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "c")
        if files is base_files:
            base = git(repo, "rev-parse", "HEAD")
    return repo, base, git(repo, "rev-parse", "HEAD")


def _pair_body(repo: Path, base: str, head: str, design: str) -> str:
    from types import SimpleNamespace
    ctx = _TitleCtx(facts={"brief": "b", "check": "pytest", "name": "t"}, files={"design.md": design})
    return delivery._body(ctx, SimpleNamespace(path=str(repo), base=base, head=head, branch="b"))


def _main_moved(github: Path, slug: str, name: str, tmp_path: Path) -> str:
    """Someone merged ``name`` into main on GitHub; returns main's new tip."""
    other = tmp_path / f"other-{name}"
    git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(other))
    (other / name).write_text(f"{name}\n")
    git(other, "add", name)
    git(other, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", name)
    git(other, "push", "-q", "origin", "main")
    return git(other, "rev-parse", "HEAD")


# ── b45a review: the overall limit, the failing refusal, and what a redo keeps ──
FAILING = render({"tool.py": "print('built')\nraise SystemExit(1)\n"})


def _describing(monkeypatch, tmp_path, body: str, *, title: str = "T", draft: bool = False) -> dict:
    """A pull request as `_describe` reads and edits it, through a stand-in for `_gh`."""
    state = {"body": body, "title": title, "draft": draft, "calls": []}

    def gh(*args, **_):
        state["calls"].append(args)
        if args[:2] == ("pr", "view"):
            return json.dumps({"body": state["body"], "title": state["title"], "isDraft": state["draft"]})
        if args[:2] == ("pr", "edit") and "--body-file" in args:
            state["body"] = Path(args[args.index("--body-file") + 1]).read_text()
        if args[:2] == ("pr", "edit") and "--title" in args:
            state["title"] = args[args.index("--title") + 1]
        if args[:2] == ("pr", "ready"):
            state["draft"] = "--undo" in args
        return ""

    monkeypatch.setattr(delivery, "_gh", gh)
    return state


OURS = f"{delivery.BODY_START}\n## Summary\n\nS.\n{delivery.BODY_END}\n"


def _design_draft(workdir, github, tmp_path, *, mode=None, emit=None, target=None,
                  check="python tool.py", generate="", draft_fact=None, before=(), seen=None):
    """Run a full design with the draft_delivery fact injected into the ops that read it.

    When *target* is None the run creates a new project (same as ``_design``).
    When *target* is a repo path, the run uses that existing repo.
    *before* are steps of `delivery_fakes.deliver_after` delivery runs first (main
    moving on GitHub, say). *seen*, when given, gets ``head``: the run's reviewed commit
    as delivery found it, before delivery's own bookkeeping can move the worktree record.
    """
    fact = draft_fact if draft_fact is not None else DRAFT_FACT
    import refactory_dsh.design_ops as design_ops_mod

    seen_file = Path(workdir).with_name(Path(workdir).name + ".seen-head")
    steps = [["draft", {"fact": fact}],
             *([["seen_head", {"path": str(seen_file)}]] if seen is not None else []), *before]
    # The report the design graph calls: its judges decided, so it is report_judged.
    _child().setattr(design_ops_mod, "report_judged", delivery_fakes.report_with_draft, fact=fact)
    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after, steps=steps)
    task_kw: dict[str, str] = {}
    if target is not None:
        task_kw["target"] = str(target)
    if generate:
        task_kw["generate"] = generate
    model = FakeModel(_answers(render(emit or {"tool.py": "print('built')\n"})))
    result = (mode or DesignMode()).run(
        workdir, model=model, verdicts={"gate_strategy": "proceed"},
        task={"name": NAME, "brief": "Build a tool\nthat prints built.",
              "check": check, **task_kw},
    )
    if seen is not None:
        seen["head"] = seen_file.read_text()
    return result


def _parents(github: Path, slug: str, commit: str) -> list[str]:
    """The parents of ``commit`` in the repository on GitHub."""
    return git(github / f"{slug}.git", "rev-list", "--parents", "-n1", commit).split()[1:]


def _shown(github: Path, slug: str, commit: str, name: str) -> str:
    """The file ``name`` as ``commit`` on GitHub holds it."""
    return git(github / f"{slug}.git", "show", f"{commit}:{name}")


def _draft_items(body: str) -> list[str]:
    """The item lines of the draft section at the top of a pull request's description."""
    found = delivery._DRAFT_SECTION.search(body)
    assert found is not None, body[:200]
    section = found.group(0)
    return [line for line in section.splitlines() if line.startswith("- ")]


def _registry_note(run: Path) -> dict:
    """What `refactory projects` keeps about the run's branch."""
    found = worktree.recorded(run)
    return next(note for entry in projects.load()["projects"] for note in entry.get("runs", [])
                if note.get("branch") == found.branch)


# ── shared design runs ─────────────────────────────────────────────────────
#
# Many tests start from one of two design runs and then do something to what it left:
# a new project whose first pull request is open (`new_run`), or an existing repository
# with a pull request a run opened (`opened_pull`, for refinements). Those runs are the
# same every time, so each is walked once, for real, through the whole design graph,
# and every test gets its own copy of everything it left: the run directory, the
# worktree, the project and its registry, gh's log and GitHub. The copy is restored at
# the path the run was made at -- one stage directory for the module -- because the
# worktrees, records and remotes name absolute paths.
#
# A copy is not the run in one respect: a reviewer's sign-off is bound to the inode and
# ctime of what it saw (`agent_delivery._fingerprint`), and copied files have new ones.
# So a test that needs current sign-offs -- `report` without a draft fact, which calls
# `require_completion` -- walks its own run.


@dataclass
class Staged:
    """A restored design run, and what its test needs to read it."""

    root: Path  # what `tmp_path` is to a test that walks its own run: `bin/calls.log` is here
    github: Path
    run: Path | None = None  # `new_run`: the run's directory
    result: nn.RunResult | None = None  # `new_run`: what the run returned
    repo: Path | None = None  # `opened_pull`: the person's checkout
    url: str = ""  # `opened_pull`: the pull request the first run opened
    pull: dict | None = None  # `opened_pull`: that pull request as GitHub showed it


def _use_home(monkeypatch: pytest.MonkeyPatch, base: Path) -> None:
    """Refactory's home and projects folder under ``base``, as `_isolate_refactory_home` sets them."""
    home = base / "refactory-home"
    monkeypatch.setenv("REFACTORY_HOME", str(home / "home"))
    monkeypatch.setenv("REFACTORY_PROJECTS_DIR", str(home / "projects"))
    monkeypatch.setenv("GH_CONFIG_DIR", str(home / "gh"))
    for name in ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def _build_new(stage: Path, github: Path) -> Staged:
    result = _design(stage / "run")
    assert result.exit_ok, result.error
    return Staged(stage, github, stage / "run", result)


def _build_opened(stage: Path, github: Path) -> Staged:
    repo, url, pull = _opened(github, stage)
    return Staged(stage, github, repo=repo, url=url, pull=pull)


@pytest.fixture(scope="module")
def _stage(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("delivery-stage")


def _golden(stage: Path, factory: pytest.TempPathFactory, kind: str, build) -> tuple[Path, Staged]:
    """Walk one shared run at ``stage`` and keep a pristine copy of everything it left."""
    shutil.rmtree(stage)
    stage.mkdir()
    github = _make_github(stage)
    with pytest.MonkeyPatch.context() as monkeypatch:
        _use_github(monkeypatch, stage)
        _use_home(monkeypatch, stage)
        made = build(stage, github)
    pristine = factory.mktemp(f"delivery-{kind}") / "tree"
    shutil.copytree(stage, pristine, symlinks=True)
    return pristine, made


def _restore(stage: Path, golden: tuple[Path, Staged], monkeypatch: pytest.MonkeyPatch) -> Staged:
    """This test's own copy of a shared run, at the stage, with gh and the home pointed at it."""
    pristine, made = golden
    shutil.rmtree(stage)
    shutil.copytree(pristine, stage, symlinks=True)
    _use_github(monkeypatch, stage)
    _use_home(monkeypatch, stage)
    return copy.deepcopy(made)


@pytest.fixture(scope="module")
def _new_golden(_stage: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Staged]:
    return _golden(_stage, tmp_path_factory, "new", _build_new)


@pytest.fixture(scope="module")
def _opened_golden(_stage: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Staged]:
    return _golden(_stage, tmp_path_factory, "opened", _build_opened)


@pytest.fixture
def new_run(_stage: Path, _new_golden: tuple[Path, Staged], monkeypatch: pytest.MonkeyPatch) -> Staged:
    """A new project's design run, finished: its private repository made, its first pull request open."""
    return _restore(_stage, _new_golden, monkeypatch)


@pytest.fixture
def opened_pull(_stage: Path, _opened_golden: tuple[Path, Staged], monkeypatch: pytest.MonkeyPatch) -> Staged:
    """`acme/app` on GitHub, the person's checkout of it, and a pull request a design run opened there."""
    return _restore(_stage, _opened_golden, monkeypatch)


class _WorkspaceBuilder(FakeModel):
    """A builder that does its work in its workspace: it writes the tool there, runs the
    check there, and stops, answering only that it is done."""

    def __init__(self, answers: list[str]) -> None:
        super().__init__(answers)
        self.ran: list[str] = []

    def complete(self, prompt: str, *, system: str = "", workdir: object = None, cwd: object = None,
                 env: object = None, confine: bool = False) -> str:
        answer = super().complete(prompt, system=system, workdir=workdir, cwd=cwd, env=env, confine=confine)
        if cwd is not None and "Write the artifact `" in prompt:
            clone = Path(str(cwd))
            (clone / "tool.py").write_text("import app\nprint('built')\n")
            (clone / "app.py").write_text("print('app, reworked')\n")
            ran = subprocess.run([sys.executable, "tool.py"], cwd=clone, capture_output=True, text=True,
                                 env={**os.environ, **{k: v for k, v in dict(env or {}).items() if v is not None}})
            self.ran.append(ran.stdout)
            return "Done: the tool is written and runs."
        return answer


# ── refining an open pull request ──────────────────────────────────────────
def _opened(github: Path, tmp_path: Path) -> tuple[Path, str, dict]:
    """An existing repository with a pull request a design run opened."""
    repo = git_repo(tmp_path / "repo")
    _published(github, "acme/app", repo)
    first = _design(tmp_path / "first", {"tool.py": "print('one')\n"}, target=str(repo))
    assert first.exit_ok, first.error
    return repo, first.facts["pull_request"], _pulls(github, "acme/app")[0]


# ── every route that starts a design run says how it is delivered ──────────
def _launched(monkeypatch) -> list[list[str]]:
    """What a `refactory ceo` run would have handed DSH; nothing is launched."""
    from refactory_dsh import workers, interactive, dsh_profile

    class _Run:
        returncode = 0

    seen: list[list[str]] = []
    monkeypatch.setattr(workers, "run_graph_command", lambda argv, **kwargs: seen.append(argv) or _Run())
    monkeypatch.setattr(interactive, "require_build", lambda **kw: None)
    monkeypatch.setattr(dsh_profile, "require", lambda *a, **k: None)
    return seen


def _stacked(github: Path, tmp_path: Path) -> tuple[Path, str]:
    """A repository on GitHub with a pushed branch `stack` one commit ahead of main; main is checked out."""
    repo = git_repo(tmp_path / "repo")
    _published(github, "acme/app", repo)
    git(repo, "checkout", "-q", "-b", "stack")
    (repo / "lib.py").write_text("x = 1\n")
    git(repo, "add", "lib.py")
    git(repo, "commit", "-q", "-m", "the stack's own commit")
    git(repo, "push", "-q", "-u", "origin", "stack")
    git(repo, "checkout", "-q", "main")
    return repo, git(repo, "rev-parse", "stack")


def _move_main_then_deliver(tmp_path, slug, **edit):
    """Between the run's start and delivery, commit ``edit`` (`delivery_fakes.move_main`'s
    files, commands, copy_from) to main, once: the reloop delivers again with main moved."""
    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after, steps=[["move_main", {
        "slug": slug, "clone": str(tmp_path / "other"), "message": "main moves", **edit}]])


def _counting_tool(counter: Path) -> dict[str, str]:
    """A check that passes and counts each run of it, whatever runs it."""
    return {"tool.py": f"open({str(counter)!r}, 'a').write('x')\n"}


def _redo_delivery_after_merge(github, tmp_path, monkeypatch, slug, *, move_main_again):
    """Deliver once with main moved (the merge is pushed, then verification fails), optionally
    move main again, redo delivery. Returns (checks run by the first delivery, checks run by
    the redo, the redo's record, the run directory)."""
    repo = git_repo(tmp_path / "repo")
    (repo / "extra.txt").write_text("placeholder\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    _published(github, slug, repo)
    counter = tmp_path / "check-runs"
    other = tmp_path / "other"

    def move_main(name):
        if not other.exists():
            git(tmp_path, "clone", "-q", f"https://github.com/{slug}.git", str(other))
        (other / name).write_text("new\n")
        git(other, "add", ".")
        git(other, "-c", "user.name=t", "-c", "user.email=t@example.com",
            "commit", "-q", "-m", f"main moves: {name}")
        git(other, "push", "-q", "origin", "main")

    found_file = tmp_path / "reviewed-record.json"
    # The first delivery: main moves, the push lands, verification fails.
    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after, steps=[
        ["seen_record", {"path": str(found_file)}],
        ["move_main", {"slug": slug, "clone": str(other), "files": {"extra.txt": "new\n"},
                       "message": "main moves: extra.txt"}],
        ["touch", {"path": str(github / "stale-view"), "text": "0" * 40}],
        ["patch", {"target": "refactory_dsh.delivery", "name": "VERIFY_ATTEMPTS", "value": 1}]])
    run = tmp_path / "run"
    result = _design(run, _counting_tool(counter), target=str(repo), check="python tool.py")
    assert "Unverified" in str(result.error), result.error
    branch = json.loads((run / "design.json").read_text())["branch"]
    assert _remote_head(github, slug, branch), "the first delivery must have pushed the merge"
    first_runs = len(counter.read_text())

    # The node was stopped before delivery advanced the worktree record to its merge: the
    # run still stands at the reviewed commit, with the merge already on GitHub.
    found = worktree.Worktree(**json.loads(found_file.read_text()))
    git(Path(found.path), "reset", "--hard", found.head)
    worktree._write_json(run / worktree.RECORD, asdict(found))
    (github / "stale-view").unlink(missing_ok=True)
    _unfake(delivery, "deliver_design")
    _child().setattr(delivery, "VERIFY_ATTEMPTS", 5)
    if move_main_again:
        move_main("second.txt")
    delivery.deliver_design(nn.RunContext(run, facts=result.facts, enforce_writes=False))
    return first_runs, len(counter.read_text()) - first_runs, _record(run), run


# Every name above, underscored ones too, for the test files' star import.
__all__ = [name for name in dir() if not name.startswith("__") and name != "pytestmark"]
