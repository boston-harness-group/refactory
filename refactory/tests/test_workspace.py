"""Agent workspaces: where an agent node changes files and runs commands.

The graph field (`nn.graph.workspace_declaration`), its lowering into the IR, the
declarations and the worker spec, and the runtime (`refactory_dsh/workspace.py`, with the
product's clone preparation in `refactory_dsh/clones.py`) on both runtimes: the spine's
worker (`worker_node.execute`) and the walker (`Executor._visit_agent`). The agent is a
stand-in DSH CLI driven through the real `DshModelClient`, so the working directory,
the environment and the sandbox patch are the ones a real node gets.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from refactory_dsh import clones, lowering, openshell_backend, repo_change, workers, workspace, worktree
from refactory_dsh.model import DshModelClient
from refactory_dsh.nn import Graph, GraphError, Node
from refactory_dsh.nn.executor import Executor, RunContext
from tests_support import git, git_repo

ANSWER = "answer.txt"


# ── the graph field ───────────────────────────────────────────────────────────

def test_a_workspace_declaration_is_canonical_agent_only_and_saved_only_when_declared():
    scratch = Node("solve", "agent", prompt_template="solve", writes=(ANSWER,), workspace="scratch")
    assert scratch.workspace == {"kind": "scratch", "collect": "files"}
    clone = Node("emit", "agent", prompt_template="x", writes=("emit.txt",),
                 workspace={"collect": "diff", "kind": "clone"})
    assert clone.workspace == {"kind": "clone", "of": "attempt", "collect": "diff"}
    assert Node.from_state(clone.to_state()) == clone
    plain = Node("plan", "agent", prompt_template="x", workspace="none")
    assert plain.workspace is None and "workspace" not in plain.to_state()
    assert "workspace" not in Node("plan", "agent", prompt_template="x").to_state()
    with pytest.raises(GraphError, match="not an agent but declares a workspace"):
        Node("op", "op", call="m:f", workspace="scratch")
    with pytest.raises(GraphError, match="needs a clone"):
        Node("solve", "agent", prompt_template="x", writes=(ANSWER,), workspace={"kind": "scratch", "collect": "diff"})
    with pytest.raises(GraphError, match="declares none"):
        Node("emit", "agent", prompt_template="x", workspace={"kind": "clone", "collect": "diff"})
    with pytest.raises(GraphError, match="unknown clone workspace 'of'"):
        Node("emit", "agent", prompt_template="x", writes=("e",), workspace={"kind": "clone", "of": "head"})
    with pytest.raises(GraphError, match="unknown workspace kind"):
        Node("emit", "agent", prompt_template="x", workspace="container")


def _state(declared, *, writes=(ANSWER,), check_timeout_s=None) -> dict:
    graph = Graph()
    graph.add(Node("solve", "agent", prompt_template="Solve it.", writes=writes, workspace=declared))
    knobs = {"check_timeout_s": {"value": check_timeout_s}} if check_timeout_s else {}
    return {"graph": graph.to_state(), "knobs": knobs}


def test_lowering_carries_a_declared_workspace_and_leaves_none_out():
    declared = lowering.to_workflow(_state("scratch")).to_dict()["nodes"]["solve"]
    assert declared["workspace"] == {"kind": "scratch", "collect": "files"}
    assert "workspace" not in lowering.to_workflow(_state(None)).to_dict()["nodes"]["solve"]
    assert lowering.node_declarations(_state("scratch"))["solve"]["workspace"] == declared["workspace"]
    assert "workspace" not in lowering.node_declarations(_state(None))["solve"]


def test_a_workspace_on_the_ceo_subagent_route_is_refused_at_lowering(tmp_path):
    with pytest.raises(ValueError, match="only a dispatched worker can give them"):
        workers.lower_workflow(lowering.to_workflow(_state("scratch")), tmp_path, None)
    # A graph without one lowers on that route as it always did.
    plain = lowering.to_workflow(_state(None))
    assert workers.lower_workflow(plain, tmp_path, None) is plain


def _lowered_spec(state: dict, workdir: Path, context: dict) -> tuple[dict, int]:
    lowered = workers.lower_workflow(lowering.to_workflow(state), workdir, context,
                                     lowering.node_declarations(state))
    node = lowered.nodes["solve"]
    spec = Path(shlex.split(node.dispatch_command)[-1])
    return json.loads(spec.read_text()), node.command_timeout_s


def test_the_worker_spec_carries_the_workspace_and_reserves_its_preparation(tmp_path):
    context = DshModelClient(worker_backend="dsh").context()
    plain, plain_s = _lowered_spec(_state(None), tmp_path, context)
    assert "workspace" not in plain
    scratch, scratch_s = _lowered_spec(_state("scratch"), tmp_path, context)
    assert scratch["workspace"] == {"kind": "scratch", "collect": "files"}
    assert scratch_s - plain_s == workspace.SCRATCH_S
    clone = {"kind": "clone", "collect": "diff"}
    cloned, cloned_s = _lowered_spec(_state(clone, check_timeout_s=200), tmp_path, context)
    assert cloned["workspace"] == {"kind": "clone", "of": "attempt", "collect": "diff"}
    from refactory_dsh.invocations import INSTALL_TIMEOUT_S
    assert cloned_s - plain_s == workspace.CLONE_S + 3 * INSTALL_TIMEOUT_S + 2 * 200


# ── a stand-in agent ──────────────────────────────────────────────────────────

def _agent(tmp_path: Path, body: str) -> Path:
    """A stand-in DSH CLI: records where it ran, then does ``body`` (Python) in its cwd."""
    script = tmp_path / "agent-cli"
    seen = tmp_path / "seen.jsonl"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, subprocess, sys\n"
        "prompt = sys.stdin.read()\n"
        "cwd = pathlib.Path.cwd()\n"
        f"with open({str(seen)!r}, 'a') as out:\n"
        "    out.write(json.dumps({'cwd': str(cwd), 'listing': sorted(os.listdir(cwd)),\n"
        "                          'home': os.environ.get('HOME'), 'prompt': prompt}) + '\\n')\n"
        + body + "\n")
    script.chmod(0o755)
    return script


def _seen(tmp_path: Path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "seen.jsonl").read_text().splitlines()]


def _client(script: Path) -> DshModelClient:
    return DshModelClient(worker_backend="dsh", dsh_cli=str(script), attempts=1)


def _walk(graph: Graph, workdir: Path, client, *, facts=None, knobs=None) -> None:
    ctx = RunContext(workdir, knobs=knobs or {}, facts=facts or {}, task={})
    ctx.model = client
    Executor(graph, ctx).run(start=graph.start)


SOLVER = ("(cwd / 'answer.txt').write_text('42\\n')\n"
          "(cwd / 'scratch-notes.txt').write_text('working\\n')\n"
          "print('solved')\n")


@pytest.mark.parametrize("runtime", ["walker", "spine"])
def test_a_scratch_solver_works_in_a_new_empty_folder_and_its_answer_is_collected(tmp_path, runtime, monkeypatch):
    monkeypatch.setattr(workspace, "_sandbox_runner", lambda root: ("landlock", ""))
    workdir = tmp_path / "run"
    workdir.mkdir()
    script = _agent(tmp_path, SOLVER)
    state = _state("scratch")
    for _ in range(2):
        if runtime == "walker":
            _walk(Graph.from_state(state["graph"]), workdir, _client(script))
        else:
            from refactory_dsh import worker_node
            spec, _ = _lowered_spec(state, workdir, _client(script).context())
            worker_node.execute(spec, workdir)
        assert (workdir / ANSWER).read_text() == "42\n"
        assert not (workdir / "scratch-notes.txt").exists(), "only declared writes are collected"
    first, second = _seen(tmp_path)
    root = workdir / workspace.ROOT / "solve"
    for seen in (first, second):
        assert Path(seen["cwd"]).parent == root and Path(seen["cwd"]).name.startswith("scratch-")
        assert seen["listing"] == [], "the agent starts in an empty folder"
        assert seen["home"] == seen["cwd"]
    assert first["cwd"] != second["cwd"], "each dispatch gets a fresh folder"
    assert not Path(first["cwd"]).exists(), "an earlier dispatch's folder is removed"
    # The contract tells the agent where it works, and its tools are confined there.
    assert f"Your workspace is {first['cwd']}" in first["prompt"]
    patches = [json.loads(path.read_text()) for path in (workdir / ".refactory" / "workers").glob("*.patch.yml")]
    policies = [row["config"] for rows in patches for row in rows if row.get("id") == "sandbox-policy"]
    assert sorted(policy["workspaceRoot"] for policy in policies) == sorted([first["cwd"], second["cwd"]])
    assert {policy["mode"] for policy in policies} == {"workspace-write"}
    assert all({"id": "approval", "config": {"policy": "never"}} in rows for rows in patches)
    for rows in patches:
        assert {"id": "bash-sandbox", "config": {
            "timeoutMs": 600_000, "maxTimeoutMs": 600_000}} in rows
        assert {"id": "tool-bash", "config": {
            "enableRunInBackground": False, "promoteOnTimeout": False}} in rows



def test_without_a_working_sandbox_a_workspace_with_a_shell_is_refused_unless_chosen(tmp_path, monkeypatch, capsys):
    """Nothing here can confine the shell: the node is refused, not run with full access and
    the person's DSH home, unless the person opts in; then it runs unconfined and says so,
    and its shell is still given a DSH home of its own."""
    monkeypatch.setattr(workspace, "_sandbox_runner", lambda root: ("", "no runner here"))
    workdir = tmp_path / "run"
    workdir.mkdir()
    graph = Graph.from_state(_state("scratch")["graph"])
    ctx = RunContext(workdir, knobs={}, facts={}, task={})
    ctx.model = _client(_agent(tmp_path, SOLVER))
    refused = Executor(graph, ctx).run(start="solve")
    assert "nothing on this machine can confine it (no runner here)" in str(refused.error)
    assert not (tmp_path / "seen.jsonl").exists(), "the agent never ran"

    monkeypatch.setenv(*workspace.UNCONFINED_OPT_IN)
    _walk(graph, workdir, _client(_agent(tmp_path, SOLVER)))
    rows = json.loads(next((workdir / ".refactory" / "workers").glob("*.patch.yml")).read_text())
    assert {"id": "sandbox-policy", "config": {"mode": "danger-full-access"}} in rows
    [seen] = _seen(tmp_path)
    assert {"id": "shell-env", "config": {"dshHome": str(Path(seen["cwd"]) / ".refactory" / "dsh-home")}} in rows
    dispatch = json.loads((workdir / workspace.ROOT / "solve" / "dispatch.json").read_text())
    assert dispatch["sandbox"] == "not confined: REFACTORY_WORKSPACE_SANDBOX=unconfined"
    assert "without DSH's workspace-write policy" in capsys.readouterr().err


def test_a_native_backend_runs_under_its_own_sandbox_with_its_own_login(tmp_path, monkeypatch):
    """codex and Claude Code bring their own sandbox and find their login under HOME: no DSH
    policy is set for them, nothing is refused, and their process keeps the person's HOME
    and cache directory (checked statically: no live login here). Even where DSH could
    confine a shell, theirs is not DSH's to confine."""
    monkeypatch.setattr(workspace, "_sandbox_runner", lambda root: ("landlock", ""))
    for backend in ("codex", "claude_code"):
        route = {"worker_backend": backend, "permission_mode": "skip"}
        assert workspace.confinement(route)[0] == ""
        assert not workspace._needs_confinement(route)
        assert workers.permission_patch(route, workspace_root=tmp_path) == []
    run, place = _workspace_with(tmp_path)
    native = workspace.agent_env(place, {"worker_backend": "codex"})
    assert "HOME" not in native and "XDG_CACHE_HOME" not in native
    assert native["BASH_ENV"] == place.env["BASH_ENV"] and native["PATH"] == place.env["PATH"]
    dsh = workspace.agent_env(place, {"worker_backend": "dsh"})
    assert dsh["HOME"] == str(place.path / ".refactory" / "home")


def test_a_client_that_cannot_run_in_a_workspace_is_refused(tmp_path):
    class TextOnly:
        def complete(self, prompt, *, system="", workdir=None):
            return "an answer"

    workdir = tmp_path / "run"
    workdir.mkdir()
    ctx = RunContext(workdir, knobs={}, facts={}, task={})
    ctx.model = TextOnly()
    result = Executor(Graph.from_state(_state("scratch")["graph"]), ctx).run(start="solve")
    assert "its complete() takes no cwd" in str(result.error)
    assert not (workdir / ANSWER).exists()


def test_the_run_end_removes_the_workspaces(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "_sandbox_runner", lambda root: ("landlock", ""))
    workdir = tmp_path / "run"
    workdir.mkdir()
    graph = Graph.from_state(_state("scratch")["graph"])
    from refactory_dsh.nn.executor import execute
    execute(graph, workdir, model=_client(_agent(tmp_path, SOLVER)))
    assert (workdir / ANSWER).read_text() == "42\n"
    assert not (workdir / workspace.ROOT).exists()


# ── a clone of the attempt ────────────────────────────────────────────────────

def _run_with_worktree(tmp_path: Path, files: dict[str, str] | None = None) -> tuple[Path, Path, worktree.Worktree]:
    repo = git_repo(tmp_path / "repo")
    for name, text in (files or {}).items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
        git(repo, "add", name)
    if files:
        git(repo, "commit", "-q", "-m", "more")
    run = tmp_path / "run"
    run.mkdir()
    found = worktree.create(repo, "HEAD", "0123456789ab", run)
    return run, repo, found


def _clone_state() -> dict:
    graph = Graph()
    graph.add(Node("emit", "agent", prompt_template="Make the change.", writes=("emit.txt",),
                   workspace={"kind": "clone", "of": "attempt", "collect": "diff"}))
    return {"graph": graph.to_state(), "knobs": {}}


BUILDER = ("(cwd / 'app.py').write_text(\"print('changed')\\n\")\n"
           "(cwd / 'tool.py').write_text(\"import app\\nprint('tool')\\n\")\n"
           "(cwd / 'gone.py').unlink()\n"
           "ran = subprocess.run([sys.executable, 'tool.py'], capture_output=True, text=True)\n"
           f"pathlib.Path({{seen!r}}).with_name('ran.txt').write_text(ran.stdout)\n"
           "print('done')\n")


@pytest.mark.parametrize("runtime", ["walker", "spine"])
def test_a_clone_builder_edits_and_runs_there_and_its_diff_is_what_materialise_commits(
        tmp_path, runtime, monkeypatch):
    monkeypatch.setattr(workspace, "_sandbox_runner", lambda root: ("landlock", ""))
    run, repo, found = _run_with_worktree(tmp_path, {"gone.py": "x = 1\n"})
    script = _agent(tmp_path, BUILDER.replace("{seen!r}", repr(str(tmp_path / "seen.jsonl"))))
    state = _clone_state()
    if runtime == "walker":
        _walk(Graph.from_state(state["graph"]), run, _client(script))
    else:
        from refactory_dsh import worker_node
        lowered = workers.lower_workflow(lowering.to_workflow(state), run, _client(script).context(),
                                         lowering.node_declarations(state))
        spec = json.loads(Path(shlex.split(lowered.nodes["emit"].dispatch_command)[-1]).read_text())
        worker_node.execute(spec, run)
    [seen] = _seen(tmp_path)
    clone = Path(seen["cwd"])
    assert clone == run / workspace.ROOT / "emit" / "repo"
    assert seen["home"] == str(clone / ".refactory" / "home"), "its own HOME, inside what it may write"
    # Its own refs and no hooks or remote: nothing it does reaches the person's repository.
    assert git(clone, "config", "core.hooksPath") == "/dev/null" and git(clone, "remote") == ""
    assert (tmp_path / "ran.txt").read_text() == "changed\ntool\n", "the builder ran its change in the clone"
    assert (Path(found.path) / "app.py").read_text() == "print('app')\n", "the run's worktree is untouched"
    emission = (run / "emit.txt").read_text()
    assert emission == ("===FILE: app.py===\nprint('changed')\n"
                        "===DELETE: gone.py===\n"
                        "===FILE: tool.py===\nimport app\nprint('tool')\n")
    # Materialise commits exactly the workspace's change.
    ctx = RunContext(run, knobs={}, facts={"name": "tool"}, task={})
    ctx.enforce_writes = False
    repo_change.materialise(ctx)
    advanced = worktree.recorded(run)
    changed = git(Path(advanced.path), "diff", "--name-status", "--no-renames", advanced.base, advanced.head)
    assert sorted(changed.splitlines()) == ["A\ttool.py", "D\tgone.py", "M\tapp.py"]
    for name in ("app.py", "tool.py"):
        assert git(Path(advanced.path), "show", f"{advanced.head}:{name}") + "\n" == (clone / name).read_text()
    assert clones.tree_of(clone) == git(clone, "rev-parse", f"{advanced.head}^{{tree}}")


def _workspace_with(tmp_path: Path, files: dict[str, str] | None = None):
    run, repo, found = _run_with_worktree(tmp_path, files)
    place = workspace.prepare(run, "emit", {"kind": "clone", "collect": "diff"}, facts={})
    return run, place


@pytest.mark.parametrize("change, reason", [
    (lambda clone: (clone / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\0\0"), "logo.png is a binary file"),
    (lambda clone: (clone / "data.txt").write_bytes(b"valid utf-8\0with a NUL\n"), "data.txt is a binary file"),
    (lambda clone: (clone / "link.py").symlink_to("app.py"), "link.py is a symlink"),
    (lambda clone: (clone / "app.py").chmod(0o755), "app.py changes its file mode (100644 to 100755)"),
    (lambda clone: ((clone / "run.sh").write_text("echo\n"), (clone / "run.sh").chmod(0o755)), "run.sh is executable"),
    (lambda clone: (clone / "doc.md").write_text("intro\n===FILE: x===\n"), "doc.md has a line starting with ===FILE:"),
], ids=["binary", "nul-bytes", "symlink", "mode-change", "new-executable", "block-header"])
def test_a_change_the_emission_cannot_carry_is_refused_with_the_reason(tmp_path, change, reason):
    run, place = _workspace_with(tmp_path)
    (place.path / "ok.py").write_text("ok = 1\n")
    change(place.path)
    workspace.collect(place, run, ["emit.txt"], final_response="done")
    refusal = workspace.refusal_for(run, "emit.txt", (run / "emit.txt").read_text())
    assert refusal is not None and reason in refusal
    # Materialise sends the reason back rather than applying any of it.
    ctx = RunContext(run, knobs={}, facts={"name": "tool"}, task={})
    ctx.enforce_writes = False
    result = repo_change.materialise(ctx)
    assert reason in result["emission_refused"]
    found = worktree.recorded(run)
    assert found.head == found.base, "nothing of a refused change is committed"


def test_a_refusal_never_outlives_the_emission_it_was_about(tmp_path):
    run, place = _workspace_with(tmp_path)
    (place.path / "logo.png").write_bytes(b"\0binary")
    workspace.collect(place, run, ["emit.txt"])
    (place.path / "logo.png").unlink()
    (place.path / "ok.py").write_text("ok = 1\n")
    workspace.collect(place, run, ["emit.txt"])
    assert workspace.refusal_for(run, "emit.txt", (run / "emit.txt").read_text()) is None
    assert (run / "emit.txt").read_text() == "===FILE: ok.py===\nok = 1\n"


def test_the_environments_and_refactorys_own_folder_are_never_collected(tmp_path):
    run, place = _workspace_with(tmp_path)
    for name in (".venv/lib/x.py", "pkg/node_modules/y/index.js", ".refactory/home/.cache/z"):
        (place.path / name).parent.mkdir(parents=True, exist_ok=True)
        (place.path / name).write_text("made by an install\n")
    (place.path / "app.py").write_text("print('new')\n")
    workspace.collect(place, run, ["emit.txt"])
    assert (run / "emit.txt").read_text() == "===FILE: app.py===\nprint('new')\n"


# ── its own code ──────────────────────────────────────────────────────────────

WHICH = ("import shutil\n"
         "found = shutil.which('whichcode')\n"
         "print(subprocess.run([found], capture_output=True, text=True).stdout.strip() if found else 'none')\n")


def _whichcode(directory: Path, said: str) -> None:
    (directory / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
    (directory / ".venv" / worktree.MARKER).write_text("")
    script = directory / ".venv" / "bin" / "whichcode"
    script.write_text(f"#!/bin/sh\necho {said}\n")
    script.chmod(0o755)


def test_the_workspace_runs_its_own_environment_not_the_run_worktrees(tmp_path, monkeypatch):
    """The run worktree's virtual environment, which its environment overlay and an
    inherited PATH both put first, is not the clone's: the clone's own comes first, and
    a clone with none of its own does not fall back to the worktree's."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # `uv sync` in a clone makes that clone's environment, as a real install does.
    (bin_dir / "uv").write_text("#!/bin/sh\nmkdir -p .venv/bin\n"
                                "printf '#!/bin/sh\\necho workspace\\n' > .venv/bin/whichcode\n"
                                "chmod +x .venv/bin/whichcode\n")
    (bin_dir / "uv").chmod(0o755)
    run, repo, found = _run_with_worktree(tmp_path, {"pyproject.toml": "[project]\nname = 'x'\n"})
    _whichcode(Path(found.path), "worktree")
    monkeypatch.setenv("PATH", os.pathsep.join([str(Path(found.path) / ".venv" / "bin"), str(bin_dir),
                                                os.environ["PATH"]]))
    agent = _agent(tmp_path, WHICH)
    place = workspace.prepare(run, "emit", {"kind": "clone", "collect": "diff"}, facts={})
    assert _client(agent).complete("which", workdir=run, cwd=place.path, env=place.env).strip() == "workspace"
    # A clone with no environment of its own: the worktree's is not on its PATH.
    (place.path / ".venv").rename(place.path / "no-venv")
    bare = clones.agent_environment(worktree.check_environment(place.path, place.path / ".refactory" / "home"),
                                    found.path, place.path / ".refactory" / "home")
    assert _client(agent).complete("which", workdir=run, cwd=place.path, env=bare).strip() == "none"
    # Without a workspace the same agent gets the run worktree's: what the workspace undoes.
    assert _client(agent).complete("which", workdir=run).strip() == "worktree"


# ── prepared once per run ─────────────────────────────────────────────────────

def _logging_tools(tmp_path: Path, monkeypatch) -> Path:
    log = tmp_path / "tools.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "uv").write_text(f"#!/bin/sh\necho \"uv $*\" >> {log}\nmkdir -p .venv\n")
    (bin_dir / "uv").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return log


def _attempt(run: Path, place: workspace.Workspace) -> worktree.Worktree:
    """What materialise does with the workspace's emission: the next attempt."""
    workspace.collect(place, run, ["emit.txt"])
    ctx = RunContext(run, knobs={}, facts={"name": "tool"}, task={})
    ctx.enforce_writes = False
    assert "emission_refused" not in repo_change.materialise(ctx)
    return worktree.recorded(run)


def test_a_clone_is_prepared_once_and_reinstalls_only_when_its_manifests_change(tmp_path, monkeypatch):
    log = _logging_tools(tmp_path, monkeypatch)
    run, repo, found = _run_with_worktree(tmp_path, {"pyproject.toml": "[project]\nname = 'x'\n",
                                                     "uv.lock": "version = 1\n"})
    facts = {"setup": f"echo setup >> {log}", "generate": f"echo generate >> {log}"}
    declared = {"kind": "clone", "collect": "diff"}

    def calls() -> list[str]:
        return [line.split()[0] for line in log.read_text().splitlines()]

    first = workspace.prepare(run, "emit", declared, facts=facts)
    assert (first.synced, first.prepared) == ("cloned", "installs, generate and setup")
    assert calls() == ["uv", "generate", "setup"]
    (first.path / "app.py").write_text("print('attempt 1')\n")
    # A file the agent adds is in the commit, and the clone holds it, untracked.
    (first.path / "added").mkdir()
    (first.path / "added" / "tool.py").write_text("print('new')\n")
    head = _attempt(run, first).head

    second = workspace.prepare(run, "emit", declared, facts=facts)
    assert (second.synced, second.prepared, second.base) == ("verified", "generate", head)
    assert calls() == ["uv", "generate", "setup", "generate"], "unchanged manifests: no install, no setup"
    assert (second.path / "app.py").read_text() == "print('attempt 1')\n"

    (second.path / "uv.lock").write_text("version = 2\n")
    _attempt(run, second)
    third = workspace.prepare(run, "emit", declared, facts=facts)
    assert (third.synced, third.prepared) == ("verified", "installs, generate and setup")
    assert calls()[4:] == ["uv", "generate", "setup"], "a changed lockfile installs again"

    # Nothing materialised since: the clone keeps the agent's work as it left it.
    (third.path / "draft.py").write_text("x = 1\n")
    kept = workspace.prepare(run, "emit", declared, facts=facts)
    assert kept.synced == "kept" and (kept.path / "draft.py").exists()


def test_a_head_the_clone_does_not_hold_resets_the_clone_to_it(tmp_path, monkeypatch):
    log = _logging_tools(tmp_path, monkeypatch)
    facts = {"setup": f"echo setup >> {log}"}
    run, repo, found = _run_with_worktree(tmp_path)
    place = workspace.prepare(run, "emit", {"kind": "clone", "collect": "diff"}, facts=facts)
    (place.path / "app.py").write_text("print('mine')\n")
    # The attempt became something else: a check amended it, say.
    found = worktree.commit_attempt(run, {Path("app.py"): "print('theirs')\n"}, "tool: attempt 1")
    again = workspace.prepare(run, "emit", {"kind": "clone", "collect": "diff"}, facts=facts)
    assert again.synced == "reset" and again.base == found.head
    assert (again.path / "app.py").read_text() == "print('theirs')\n"
    # A reset removed what setup made, so setup runs again though no manifest changed.
    assert log.read_text().splitlines() == ["setup", "setup"]


def test_a_failed_preparation_step_is_reported_to_the_agent_not_fatal(tmp_path):
    run, repo, found = _run_with_worktree(tmp_path)
    place = workspace.prepare(run, "emit", {"kind": "clone", "collect": "diff"},
                              facts={"generate": "echo broken >&2; exit 3"})
    assert place.problems and "the generate command" in place.problems[0]
    from refactory_dsh.agent_delivery import begin
    record = begin(run, "emit", "Make it.", ["emit.txt"], workspace=place.describe())
    assert "Preparing it reported a problem" in record["prompt"]
    assert record["execution"]["workspace"]["problems"] == list(place.problems)


# ── OpenShell: the same workspace, packed, run elsewhere and brought back ─────

def _sandbox_round_trip(place: workspace.Workspace, tmp_path: Path, edit) -> Path:
    """What an OpenShell sandbox does with a workspace: unpack the upload, work, hand it back."""
    tarball = openshell_backend.pack_workspace(place.path, "rfws-test")
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    with tarfile.open(tarball) as archive:
        archive.extractall(sandbox, filter="tar")
    inside = sandbox / place.path.name
    edit(inside)
    (inside / ".factory").mkdir(exist_ok=True)
    (inside / ".factory" / "run.log").write_text("the sandbox's log\n")
    return inside


def _edits(root: Path) -> None:
    (root / "app.py").write_text("print('from the sandbox')\n")
    (root / "new.py").write_text("x = 1\n")
    (root / "gone.py").unlink()


def test_a_workspace_from_an_openshell_sandbox_collects_the_same_diff(tmp_path):
    files = {"gone.py": "y = 2\n"}
    run, place = _workspace_with(tmp_path / "local", files)
    _edits(place.path)
    workspace.collect(place, run, ["emit.txt"])
    local = (run / "emit.txt").read_text()

    run, place = _workspace_with(tmp_path / "remote", files)
    (place.path / ".venv").mkdir()
    (place.path / ".venv" / "kept").write_text("an install's\n")
    downloaded = _sandbox_round_trip(place, tmp_path / "remote", _edits)
    workspace.sync_back(downloaded, place.path)
    workspace.collect(place, run, ["emit.txt"])
    assert (run / "emit.txt").read_text() == local
    assert [line for line in local.splitlines() if line.startswith("===")] == [
        "===FILE: app.py===", "===DELETE: gone.py===", "===FILE: new.py==="]
    assert (place.path / ".venv" / "kept").exists(), "the environment stays as it is here"
    assert not (place.path / ".factory" / "run.log").exists(), "the sandbox's run artifacts are not work"


def test_the_openshell_plan_for_a_workspace_runs_the_agent_inside_it(tmp_path):
    run, place = _workspace_with(tmp_path)
    plan = openshell_backend.compose_plan(place.path, ["sh", "-c", workspace.OPENSHELL_AGENT])
    assert plan.project_dir == f"{openshell_backend.WORKSPACE_ROOT}/repo"
    assert plan.run_command.startswith(f"cd {plan.project_dir} && sh -c ")
    assert "claude -p" in plan.run_command
    tarball = openshell_backend.pack_workspace(place.path, plan.name)
    names = tarfile.open(tarball).getnames()
    assert "repo/app.py" in names
    assert not any(name.startswith("repo/.refactory/openshell") for name in names), \
        "the tarball is never packed into itself"


def test_an_openshell_session_brings_the_workspace_back_to_the_same_collect(tmp_path, monkeypatch):
    """The OpenShell route of `workspace.ask`: the sandbox's lifecycle is the backend's, faked
    here at its download, where the sandbox's copy of the workspace comes back."""
    run, place = _workspace_with(tmp_path)

    def session(workspace_dir, argv, *, context, timeout, dest, receipts):
        assert argv == ["sh", "-c", workspace.OPENSHELL_AGENT] and receipts == run
        assert (Path(workspace_dir) / ".refactory" / "session" / "prompt.md").read_text() == "Make it."
        inside = _sandbox_round_trip(place, tmp_path, _edits_and_reply)
        return 0, inside

    def _edits_and_reply(root: Path) -> None:
        (root / "app.py").write_text("print('from the sandbox')\n")
        (root / ".refactory" / "session" / "response.md").write_text("done in the sandbox\n")

    monkeypatch.setattr(openshell_backend, "run_workspace_session", session)

    class OpenShellClient:
        def route(self):
            return {"worker_backend": "openshell"}

        def context(self):
            return {"attempts": 1, "timeout_s": 60}

    reply = workspace.ask(place, OpenShellClient(), "Make it.", workdir=run)
    assert reply == "done in the sandbox\n"
    workspace.collect(place, run, ["emit.txt"])
    assert (run / "emit.txt").read_text() == "===FILE: app.py===\nprint('from the sandbox')\n"


# ── what the node is told, and what it answers besides its diff ───────────────

def test_a_workspace_nodes_contract_names_its_workspace_and_how_its_work_is_taken(tmp_path):
    from refactory_dsh import agent_delivery, target_guard

    run, place = _workspace_with(tmp_path)
    found = worktree.recorded(run)
    worktree.write_wrapper(found, run)
    target_guard.record(run, Path(found.target), output_only=("emit",), also=(Path(found.path),))
    plain = agent_delivery.begin(run, "emit", "Make it.", ["emit.txt", "verifier-response.md"])
    agent_delivery.invalidate(run, ["emit"], "test")
    told = agent_delivery.begin(run, "emit", "Make it.", ["emit.txt", "verifier-response.md"],
                                workspace=place.describe())
    prompt, execution = told["prompt"], told["execution"]
    assert f"Your workspace is {place.path}: a clone of the run's worktree at its last attempt" in prompt
    assert "the workspace's diff is your answer. The runtime writes emit.txt from it" in prompt
    assert "Your final response is saved as verifier-response.md" in prompt
    assert execution["outputs"] == [] and execution["workspace"]["path"] == str(place.path)
    assert execution["derived"] == {"emit.txt": "the workspace's diff", "verifier-response.md": "your final response"}
    # The run worktree's wrapper and the ban on generators are for a node that works there.
    for line in ("Run repository commands only through", "Do not run generators"):
        assert line in plain["prompt"] and line not in prompt, line
    assert f"never change the target at {found.target}" in prompt


def test_the_builders_answers_to_the_verifier_are_read_from_their_own_file(tmp_path):
    from refactory_dsh import create_ops

    ctx = RunContext(tmp_path, knobs={}, facts={}, task={})
    ctx.enforce_writes = False
    (tmp_path / "emit.txt").write_text("===FILE: a.py===\nx = 'fixed in a block'\n")
    (tmp_path / create_ops.VERIFIER_RESPONSE_FILE).write_text(
        "Done.\n\n- tool.py:3: the parser drops a flag: fixed\n- the docs: rebutted: out of scope README.md:1\n")
    assert create_ops.verifier_rebuttals(ctx) == ["- tool.py:3: the parser drops a flag: fixed",
                                                  "- the docs: rebutted: out of scope README.md:1"]
    # A builder that still writes its own emission answers in its ## Verifier findings section.
    (tmp_path / create_ops.VERIFIER_RESPONSE_FILE).unlink()
    (tmp_path / "emit.txt").write_text("## Verifier findings\n- x: fixed\n===FILE: a.py===\nfixed = 1\n")
    assert create_ops.verifier_rebuttals(ctx) == ["- x: fixed"]


# ── a pending merge ───────────────────────────────────────────────────────────

def test_a_pending_merge_is_resolved_in_the_clone_and_materialise_commits_it(tmp_path):
    """Delivery found main moved and conflicting, and sent the run back to emit to merge it.
    The clone starts the same merge, the builder resolves it there, and the emission holds
    the resolution; materialise starts the merge in the worktree and commits it."""
    run, repo, found = _run_with_worktree(tmp_path)
    worktree.commit_attempt(run, {Path("app.py"): "print('attempt')\n"}, "tool: attempt 1")
    (repo / "app.py").write_text("print('main moved')\n")
    git(repo, "commit", "-qam", "main moved")
    tip = git(repo, "rev-parse", "main")
    (run / ".refactory" / "pending-merge.json").write_text(json.dumps({"target_tip": tip, "conflict_files": ["app.py"]}))

    place = workspace.prepare(run, "emit", {"kind": "clone", "collect": "diff"}, facts={})
    assert place.conflicts == ("app.py",)
    assert "<<<<<<<" in (place.path / "app.py").read_text()
    (place.path / "app.py").write_text("print('attempt and main')\n")
    workspace.collect(place, run, ["emit.txt"])
    assert (run / "emit.txt").read_text() == "===FILE: app.py===\nprint('attempt and main')\n"

    ctx = RunContext(run, knobs={}, facts={"name": "tool"}, task={})
    ctx.enforce_writes = False
    assert "emission_refused" not in repo_change.materialise(ctx)
    merged = worktree.recorded(run)
    assert git(Path(merged.path), "show", f"{merged.head}:app.py") == "print('attempt and main')"
    assert merged.base == tip, "the merge moves the run's base to main's tip"
    again = workspace.prepare(run, "emit", {"kind": "clone", "collect": "diff"}, facts={})
    assert again.synced == "verified" and again.conflicts == ()
    with pytest.raises(subprocess.CalledProcessError):
        git(again.path, "rev-parse", "--verify", "-q", "MERGE_HEAD")


def test_what_the_preparation_made_is_not_the_agents_change(tmp_path):
    """A file setup builds and a tracked file generate rewrites are the run's steps' work:
    the check runs them again on the attempt. They stay out of the emission while the
    agent leaves them as they were made, and are part of it once it changes them."""
    run, repo, found = _run_with_worktree(tmp_path, {"generated.txt": "old\n"})
    place = workspace.prepare(run, "emit", {"kind": "clone", "collect": "diff"},
                              facts={"setup": "echo built > built.txt", "generate": "echo new > generated.txt"})
    assert (place.path / "built.txt").exists() and (place.path / "generated.txt").read_text() == "new\n"
    (place.path / "tool.py").write_text("print(1)\n")
    workspace.collect(place, run, ["emit.txt"])
    assert (run / "emit.txt").read_text() == "===FILE: tool.py===\nprint(1)\n"
    (place.path / "built.txt").write_text("changed by the agent\n")
    workspace.collect(place, run, ["emit.txt"])
    assert "===FILE: built.txt===\nchanged by the agent\n" in (run / "emit.txt").read_text()


def test_a_diff_nodes_other_outputs_are_its_final_response(tmp_path):
    run, place = _workspace_with(tmp_path)
    (place.path / "app.py").write_text("print(2)\n")
    workspace.collect(place, run, ["emit.txt", "verifier-response.md"], final_response="- x: fixed\n")
    assert (run / "verifier-response.md").read_text() == "- x: fixed\n"


def test_a_workspace_in_use_is_not_swept_and_is_once_it_is_free(tmp_path):
    run, repo, found = _run_with_worktree(tmp_path)
    declared = {"kind": "clone", "collect": "diff"}
    with workspace.dispatched(run, "emit", declared, facts={}) as place:
        workspace.sweep(run)
        assert place.path.is_dir(), "the dispatch holding it keeps it"
    workspace.sweep(run)
    assert not (run / workspace.ROOT).exists()


# ── the second review's fixes ─────────────────────────────────────────────────

def test_the_shells_caches_are_inside_the_workspace_and_writable_there(tmp_path, monkeypatch):
    """pnpm's store, uv's cache and the XDG cache are the workspace HOME's: the person's are
    read-only under the sandbox, and pnpm and uv fail there (H1). Under the Landlock launcher
    DSH confines a shell with, where this machine has one, they are written."""
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "persons-uv-cache"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "persons-cache"))
    run, place = _workspace_with(tmp_path)
    home = place.path / ".refactory" / "home"
    assert place.env["pnpm_config_store_dir"] is None
    assert place.env["UV_CACHE_DIR"] == str(home / ".cache" / "uv")
    assert place.env["XDG_CACHE_HOME"] == str(home / ".cache")
    from refactory_dsh.paths import dsh_root
    launcher = workspace._landlock_launcher(Path(dsh_root()).expanduser().resolve())
    if launcher is None or subprocess.run([str(launcher), "--probe"], capture_output=True).returncode:
        pytest.skip("no Landlock launcher on this machine")
    env = {**os.environ, **{k: v for k, v in place.env.items() if v is not None}}
    env.pop("pnpm_config_store_dir", None)
    grants = [str(launcher), "--ro", "/", "--rw", "/dev/null", "--rw", "/tmp", "--rw", str(place.path), "--"]
    command = 'mkdir -p "$UV_CACHE_DIR" "$XDG_CACHE_HOME/x" && touch "$UV_CACHE_DIR/ok" "$XDG_CACHE_HOME/x/ok"'
    confined = subprocess.run([*grants, "bash", "-c", command], env=env, capture_output=True, text=True,
                              cwd=place.path)
    assert confined.returncode == 0, confined.stderr
    assert not (tmp_path / "persons-uv-cache").exists() and not (tmp_path / "persons-cache").exists()


RUN_ONLY = ("REFACTORY_WORKDIR", "REFACTORY_PROJECT", "REFACTORY_RUNTIME_ROOT", "SRF_WORK_DIR", "VENV_BIN")

ENV_PROBE = (
    "shell = subprocess.run(['bash', '-c', 'env'], capture_output=True, text=True).stdout\n"
    "pathlib.Path(os.environ['ENV_PROBE_OUT']).write_text(json.dumps({'process': dict(os.environ), "
    "'shell': dict(line.split('=', 1) for line in shell.splitlines() if '=' in line)}))\n"
    "(cwd / 'answer.txt').write_text('42\\n')\nprint('done')\n")


def test_the_runs_variables_reach_the_dsh_process_but_not_the_shell(tmp_path, monkeypatch):
    """The DSH process keeps the run's own variables (its observer reads them); the shell it
    runs a command in, `bash -c` as DSH's bash tool does, does not see them (H2)."""
    monkeypatch.setattr(workspace, "_sandbox_runner", lambda root: ("landlock", ""))
    for name in RUN_ONLY:
        monkeypatch.setenv(name, f"/run-only/{name}")
    from refactory_dsh.budgets import ResourceLimits, RunBudget
    ledger = RunBudget.open(tmp_path / "budgeted", ResourceLimits(max_elapsed_s=3600)).path
    monkeypatch.setenv("REFACTORY_BUDGET_LEDGER", str(ledger))
    monkeypatch.setenv("ENV_PROBE_OUT", str(tmp_path / "envs.json"))
    workdir = tmp_path / "run"
    workdir.mkdir()
    _walk(Graph.from_state(_state("scratch")["graph"]), workdir, _client(_agent(tmp_path, ENV_PROBE)))
    envs = json.loads((tmp_path / "envs.json").read_text())
    for name in RUN_ONLY:
        assert name in envs["process"], name
        assert name not in envs["shell"], name
    assert envs["process"]["REFACTORY_WORKDIR"] == "/run-only/REFACTORY_WORKDIR"
    assert envs["process"]["REFACTORY_BUDGET_LEDGER"] == str(ledger)
    assert "REFACTORY_BUDGET_LEDGER" not in envs["shell"]
    assert "REFACTORY_WORKER_RECEIPT" in envs["process"]
    assert not [name for name in envs["shell"] if name.startswith(("REFACTORY_", "SRF_"))]
    assert "BASH_ENV" not in envs["shell"] and envs["shell"]["HOME"] == envs["process"]["HOME"]


def test_a_round_that_adds_a_file_is_verified_not_reset(tmp_path):
    run, place = _workspace_with(tmp_path)
    (place.path / "pkg").mkdir()
    (place.path / "pkg" / "new.py").write_text("x = 1\n")
    _attempt(run, place)
    again = workspace.prepare(run, "emit", {"kind": "clone", "collect": "diff"}, facts={})
    assert again.synced == "verified"
    # The added file, unchanged since, is no change; changed, it is one.
    workspace.collect(again, run, ["emit.txt"])
    assert "has no change" in (workspace.refusal_for(run, "emit.txt", (run / "emit.txt").read_text()) or "")
    (again.path / "pkg" / "new.py").write_text("x = 2\n")
    workspace.collect(again, run, ["emit.txt"])
    assert (run / "emit.txt").read_text() == "===FILE: pkg/new.py===\nx = 2\n"


def _conflicting_run(tmp_path: Path):
    """A run whose attempt and main both changed app.py and gen.txt, and whose delivery sent
    it back to merge main: app.py is the run's own conflict, gen.txt a generated one."""
    run, repo, found = _run_with_worktree(tmp_path, {"gen.txt": "generated 0\n"})
    worktree.commit_attempt(run, {Path("app.py"): "print('attempt')\n", Path("gen.txt"): "generated 1\n"},
                            "tool: attempt 1")
    (repo / "app.py").write_text("print('main moved')\n")
    (repo / "gen.txt").write_text("generated 2\n")
    (repo / "main_only.py").write_text("print('main')\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "main moved")
    tip = git(repo, "rev-parse", "main")
    (run / ".refactory" / "pending-merge.json").write_text(
        json.dumps({"target_tip": tip, "conflict_files": ["app.py"]}))
    return run, tip


def _materialise(run: Path) -> dict:
    ctx = RunContext(run, knobs={}, facts={"name": "tool"}, task={})
    ctx.enforce_writes = False
    return repo_change.materialise(ctx)


def test_a_generated_conflict_resolves_to_mains_side_as_materialise_resolves_it(tmp_path):
    run, tip = _conflicting_run(tmp_path)
    place = workspace.prepare(run, "emit", {"kind": "clone", "collect": "diff"}, facts={})
    assert place.conflicts == ("app.py",)
    assert (place.path / "gen.txt").read_text() == "generated 2\n"
    (place.path / "app.py").write_text("print('both')\n")
    workspace.collect(place, run, ["emit.txt"])
    assert (run / "emit.txt").read_text() == "===FILE: app.py===\nprint('both')\n"
    assert "emission_refused" not in _materialise(run)
    merged = worktree.recorded(run)
    assert git(Path(merged.path), "show", f"{merged.head}:gen.txt") == "generated 2"


def test_a_given_up_merge_never_reaches_the_next_emission_as_the_agents_change(tmp_path):
    """Materialise refused the merge (markers left in) and dropped the pending merge: the
    next dispatch puts the clone back at the attempt, so main's files are not emitted, in a
    one-parent commit, as the agent's own."""
    run, tip = _conflicting_run(tmp_path)
    place = workspace.prepare(run, "emit", {"kind": "clone", "collect": "diff"}, facts={})
    workspace.collect(place, run, ["emit.txt"])  # app.py still holds its markers
    assert "conflict markers" in _materialise(run)["emission_refused"]
    assert not (run / ".refactory" / "pending-merge.json").exists()
    again = workspace.prepare(run, "emit", {"kind": "clone", "collect": "diff"}, facts={})
    assert again.synced == "reset"
    assert not (again.path / "main_only.py").exists()
    (again.path / "tool.py").write_text("print(1)\n")
    workspace.collect(again, run, ["emit.txt"])
    assert (run / "emit.txt").read_text() == "===FILE: tool.py===\nprint(1)\n"


def test_a_merge_the_agent_committed_survives_a_refused_round(tmp_path):
    run, tip = _conflicting_run(tmp_path)
    place = workspace.prepare(run, "emit", {"kind": "clone", "collect": "diff"}, facts={})
    (place.path / "app.py").write_text("print('both')\n")
    git(place.path, "add", "-A")
    git(place.path, "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "merged")
    (place.path / "logo.png").write_bytes(b"\0binary")  # refused by collect: the merge stays pending
    workspace.collect(place, run, ["emit.txt"])
    assert workspace.refusal_for(run, "emit.txt", (run / "emit.txt").read_text())
    again = workspace.prepare(run, "emit", {"kind": "clone", "collect": "diff"}, facts={})
    assert again.synced == "kept" and again.conflicts == ("app.py",)
    assert (again.path / "app.py").read_text() == "print('both')\n", "the resolution is kept"
    (again.path / "logo.png").unlink()
    workspace.collect(again, run, ["emit.txt"])
    assert "emission_refused" not in _materialise(run)
    merged = worktree.recorded(run)
    assert git(Path(merged.path), "show", f"{merged.head}:app.py") == "print('both')"
    assert merged.base == tip


def test_a_path_the_emission_cannot_carry_is_refused_by_name(tmp_path):
    assert workspace.EMISSION_PATH == repo_change.EMISSION_PATH
    run, place = _workspace_with(tmp_path)
    (place.path / "my notes.md").write_text("x\n")
    (place.path / "ok.py").write_text("y\n")
    workspace.collect(place, run, ["emit.txt"])
    refusal = workspace.refusal_for(run, "emit.txt", (run / "emit.txt").read_text())
    assert refusal and "'my notes.md' has a whitespace" in refusal


@pytest.mark.parametrize("name, character", [
    ("tool\u202e.py", "U+202E RIGHT-TO-LEFT OVERRIDE"),
    ("tool\x7f.py", "U+007F"),
    ("tool\x85.py", "U+0085"),
], ids=["bidi", "del", "c1"])
def test_a_name_no_emitted_path_may_hold_is_refused_as_an_emitted_blocks_is(tmp_path, name, character):
    """dm-278: a file the builder made in its workspace is refused by the check an emitted
    block's path gets (`emitted_paths.unprintable_refusal`), not an ASCII-only one of its own."""
    assert workspace.unprintable_refusal is repo_change.unprintable_refusal
    run, place = _workspace_with(tmp_path)
    (place.path / name).write_text("x = 1\n")
    (place.path / "ok.py").write_text("y\n")
    workspace.collect(place, run, ["emit.txt"])
    refusal = workspace.refusal_for(run, "emit.txt", (run / "emit.txt").read_text())
    expected = repo_change.unprintable_refusal(name)
    assert character in expected
    assert refusal is not None and expected in refusal and "has a whitespace" not in refusal
    assert name not in (run / "emit.txt").read_text(), "no block names it"


def test_a_file_name_that_is_not_utf8_is_refused_and_goes_back_to_emit(tmp_path):
    """dm-278: git decodes ``tool\\xff.py`` to a lone surrogate, which once crashed collect
    writing emit.txt (UnicodeEncodeError); it is refused by name like any unprintable path."""
    name = os.fsdecode(b"tool\xff.py")
    run, place = _workspace_with(tmp_path)
    (place.path / name).write_text("x = 1\n")
    (place.path / "ok.py").write_text("y\n")
    workspace.collect(place, run, ["emit.txt"], final_response="done")
    refusal = workspace.refusal_for(run, "emit.txt", (run / "emit.txt").read_text())
    assert refusal is not None and "U+DCFF (the byte 0xFF, which is not UTF-8)" in refusal
    assert repo_change.unprintable_refusal(name) in refusal
    # Materialise sends the reason back to emit rather than applying any of it.
    ctx = RunContext(run, knobs={}, facts={"name": "tool"}, task={})
    ctx.enforce_writes = False
    result = repo_change.materialise(ctx)
    assert "the byte 0xFF" in result["emission_refused"]
    found = worktree.recorded(run)
    assert found.head == found.base, "nothing of a refused change is committed"


def test_a_nested_manifest_change_installs_again(tmp_path, monkeypatch):
    log = _logging_tools(tmp_path, monkeypatch)
    run, repo, found = _run_with_worktree(tmp_path, {"pyproject.toml": "[project]\nname = 'x'\n",
                                                     "packages/a/package.json": "{}\n"})
    declared = {"kind": "clone", "collect": "diff"}
    first = workspace.prepare(run, "emit", declared, facts={})
    (first.path / "packages" / "a" / "package.json").write_text('{"dependencies": {"b": "1"}}\n')
    _attempt(run, first)
    again = workspace.prepare(run, "emit", declared, facts={})
    assert (again.synced, again.prepared) == ("verified", "installs, generate and setup")
    assert [line.split()[0] for line in log.read_text().splitlines()] == ["uv", "uv"]


def test_a_file_too_large_to_review_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "MAX_FILE_BYTES", 1000)
    monkeypatch.setattr(workspace, "MAX_TOTAL_BYTES", 1500)
    run, place = _workspace_with(tmp_path)
    (place.path / "big.txt").write_text("x" * 2000)
    workspace.collect(place, run, ["emit.txt"])
    refusal = workspace.refusal_for(run, "emit.txt", (run / "emit.txt").read_text()) or ""
    assert "big.txt is 0 MB, over the 0 MB a collected file may be" in refusal
    (place.path / "big.txt").unlink()
    for name in ("a.txt", "b.txt"):
        (place.path / name).write_text("x" * 900)
    workspace.collect(place, run, ["emit.txt"])
    assert "add up to" in (workspace.refusal_for(run, "emit.txt", (run / "emit.txt").read_text()) or "")


def test_a_refusal_for_another_emission_does_not_refuse_this_one(tmp_path):
    """The record is keyed by the emission's digest: an emit.txt written since (by a builder
    that writes its own, or by hand) is applied, not refused for the old one's reasons."""
    run, place = _workspace_with(tmp_path)
    (place.path / "logo.png").write_bytes(b"\0binary")
    workspace.collect(place, run, ["emit.txt"])
    assert (run / workspace.REFUSAL).exists()
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint('by hand')\n")
    assert "emission_refused" not in _materialise(run)


def test_the_last_normalisation_is_told_to_the_agent(tmp_path):
    run, place = _workspace_with(tmp_path)
    (place.path / "crlf.txt").write_bytes(b"a\r\nb\r\n")
    workspace.collect(place, run, ["emit.txt"])
    again = workspace.prepare(run, "emit", {"kind": "clone", "collect": "diff"}, facts={})
    assert again.normalised == ("crlf.txt",)
    from refactory_dsh.agent_delivery import begin
    told = begin(run, "emit", "Make it.", ["emit.txt"], workspace=again.describe())["prompt"]
    assert "changed the line endings or the blank lines at the ends of crlf.txt" in told
    assert "If no output file is declared" not in told, "its answer is its diff, not a returned result"


def test_tool_caches_and_a_global_ignore_never_decide_the_change(tmp_path, monkeypatch):
    ignores = tmp_path / "global-ignore"
    ignores.write_text("*.new\n")
    config = tmp_path / "gitconfig"
    config.write_text(f"[core]\n\texcludesFile = {ignores}\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    run, place = _workspace_with(tmp_path)
    for name in (".hypothesis/examples/x", ".coverage", ".tox/py/x", "sub/.nox/y"):
        (place.path / name).parent.mkdir(parents=True, exist_ok=True)
        (place.path / name).write_text("tool cache\n")
    (place.path / "feature.new").write_text("mine\n")
    workspace.collect(place, run, ["emit.txt"])
    assert (run / "emit.txt").read_text() == "===FILE: feature.new===\nmine\n"


def test_a_custom_model_without_metering_is_refused_for_its_budget_not_its_workspace(tmp_path):
    from refactory_dsh.budgets import BudgetExhausted, _UnmeteredModel

    class Budget:
        def require_usage_capability(self, metered):
            pass

    with pytest.raises(BudgetExhausted):
        _UnmeteredModel(object(), Budget()).complete("x", workdir=tmp_path, cwd=tmp_path, env={}, confine=True)


def test_a_clone_without_its_own_environment_drops_the_worktrees(tmp_path, monkeypatch):
    """The DSH process would otherwise carry the run worktree's VIRTUAL_ENV, which the
    environment overlay names, into a clone that has none."""
    run, repo, found = _run_with_worktree(tmp_path)
    _whichcode(Path(found.path), "worktree")
    place = workspace.prepare(run, "emit", {"kind": "clone", "collect": "diff"}, facts={})
    assert place.env["VIRTUAL_ENV"] is None
    agent = _agent(tmp_path, "print(os.environ.get('VIRTUAL_ENV', 'none'))\n")
    assert _client(agent).complete("x", workdir=run, cwd=place.path, env=place.env).strip() == "none"
    assert _client(agent).complete("x", workdir=run).strip() == str(Path(found.path) / ".venv")


def test_openshell_workspaces_are_experimental_and_never_sync_a_wrong_folder(tmp_path, monkeypatch):
    run, repo, found = _run_with_worktree(tmp_path)
    route = {"worker_backend": "openshell", "permission_mode": "skip"}
    with pytest.raises(workspace.WorkspaceError, match="experimental"):
        workspace.prepare(run, "emit", {"kind": "clone", "collect": "diff"}, facts={}, route=route)
    monkeypatch.setenv(*workspace.OPENSHELL_OPT_IN)
    place = workspace.prepare(run, "emit", {"kind": "clone", "collect": "diff"}, facts={}, route=route)
    empty = tmp_path / "download"
    empty.mkdir()
    with pytest.raises(workspace.WorkspaceError, match="not a copy of the workspace"):
        workspace.sync_back(empty, place.path)
    assert (place.path / "app.py").exists(), "nothing was deleted"
    # Its packed copy is a repository of its own: the borrowed objects come with it.
    workspace.dissociate(place.path)
    assert not (place.path / ".git" / "objects" / "info" / "alternates").exists()
    assert git(place.path, "cat-file", "-t", "HEAD") == "commit"


@pytest.mark.parametrize("layout", ["copy", "empty"])
def test_an_openshell_download_that_is_not_the_workspace_is_never_synced(tmp_path, monkeypatch, layout):
    """The backend's lifecycle, faked at its seams (no SDK or gateway here): a download that
    holds the workspace's copy comes back; one that does not (an empty or other layout) is
    reported as none, never handed to the sync as a folder with every file deleted."""
    import factory.contained.openshell as reference
    run, place = _workspace_with(tmp_path)
    for name in ("verify_gateway_provider", "extract_tarball", "start_run", "_terminate_remote"):
        monkeypatch.setattr(openshell_backend, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(openshell_backend, "connect", lambda gateway: object())
    monkeypatch.setattr(openshell_backend, "create_sandbox", lambda *args: object())
    monkeypatch.setattr(reference, "build_default_policy", lambda: None)
    monkeypatch.setattr(openshell_backend, "build_upload_argv", lambda *args, **kwargs: ["true"])
    monkeypatch.setattr(openshell_backend, "_poll_run", lambda *args, **kwargs: (0, None))

    def download(plan, dest, *, paths=None):
        if layout == "copy":
            shutil.copytree(place.path, Path(dest) / place.path.name)
        return []

    monkeypatch.setattr(openshell_backend, "download_artifacts", download)
    context = {**DshModelClient(worker_backend="dsh").context(), "attempts": 1}
    code, downloaded = openshell_backend.run_workspace_session(
        place.path, ["true"], context=context, timeout=10, dest=tmp_path / "staging", receipts=run)
    if layout == "copy":
        assert (code, downloaded) == (0, tmp_path / "staging" / place.path.name)
    else:
        assert downloaded is None and code != 0
