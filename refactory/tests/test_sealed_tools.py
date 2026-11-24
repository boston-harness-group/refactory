"""Tools on the steps an edit adds: listed, refused when not sealable, and run sealed.

A step the outer loop adds may list tools from `workspace.SEALED_TOOLS`; anything else is
refused naming that set. It runs in a fresh scratch workspace holding copies of the files
it reads, and each of its commands runs under `seal.runner_command`, which DSH's local
sandbox provider puts in front of the command (its ``runnerCommand``). These tests run
that runner for real (Linux namespaces, or Seatbelt on macOS): what it denies is tested
by a step trying it -- the grader's files and gold answers, another item's directory,
the run directory, the home directory, the network, writes outside its workspace, links
out of it, and the DSH process's environment.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import structure_fuzz as sf
from refactory_dsh import lowering, seal, seal_reaper, workers, workspace
from refactory_dsh.model import DshModelClient
from refactory_dsh.nn import Graph, Node, pipeline_view
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn.executor import Executor, RunContext
from refactory_dsh.optim import base as base_mod

#: CI runs the seal on both platforms (Linux with user namespaces allowed, macOS with
#: Seatbelt): there a seal that cannot run fails these tests instead of skipping them.
REQUIRED = (os.environ.get("REFACTORY_REQUIRE_CODE_SANDBOX") == "1"
            or os.environ.get("GITHUB_ACTIONS") == "true")
ROOT = Path(__file__).resolve().parents[2]


def _sealable() -> None:
    ok, why = seal.available()
    if not ok:
        if REQUIRED:
            pytest.fail(f"the seal must run on this machine: {why}")
        pytest.skip(f"no seal here: {why}")


# ── the core operator ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("tools", [["read"], ["bash", "web_fetch"], ["web_search"], True, "bash",
                                   [], ["bash", "bash"]])
def test_a_tool_outside_the_sealed_set_is_refused_naming_the_set(tools):
    state = sf.cached_applier("fuzz_draft_review").seed_state
    edit = {"op": "insert", "after": "draft", "kind": "agent", "text": "Check it.", "tools": tools}
    with pytest.raises(nn_edits.EditError) as refused:
        nn_edits.apply(edit, state)
    assert f"from {', '.join(workspace.SEALED_TOOLS)}" in str(refused.value)
    # The same inside a step written into another operator.
    branch = {"op": "parallelize", "after": "draft",
              "new": [{"kind": "agent", "text": "a", "tools": tools}, {"kind": "agent", "text": "b"}]}
    with pytest.raises(nn_edits.EditError, match="lists the tools a new step uses"):
        nn_edits.apply(branch, state)


def test_a_listed_tool_makes_a_sealed_step_that_reads_its_files_in_its_workspace():
    state = sf.cached_applier("fuzz_draft_review").seed_state
    after = nn_edits.apply({"op": "insert", "after": "draft", "kind": "agent",
                            "text": "Check it.", "tools": ["bash"]}, state)
    added = [node for nid, node in after["graph"]["nodes"].items()
             if nid not in state["graph"]["nodes"]]
    assert len(added) == 1, "no collect step: its files are copied into its workspace"
    node = added[0]
    assert workspace.is_sealed(node)
    assert node["tools"] == ["bash"] and node.get("tool_policy") is None
    assert node["workspace"]["kind"] == "scratch"
    line = next(line for line in pipeline_view.render(after).splitlines() if node["id"] in line)
    assert "sealed workspace (tools: bash)" in line, line


def test_a_sealed_step_is_a_candidate_with_no_run_switch():
    applier = sf.cached_applier("fuzz_draft_review")
    parent = sf.Candidate.from_state(applier.seed_state)
    edit = {"op": "insert", "after": "draft", "kind": "agent", "text": "Check it.", "tools": ["bash"]}
    if seal.available()[0]:
        out = applier.attempt(parent, edit, set())
        assert isinstance(out, sf.Candidate), out
    # A tools list a person writes outside a scratch workspace is still refused.
    assert base_mod._new_node_tools(
        applier.seed_state,
        {"graph": {"nodes": {"x": {"kind": "agent", "tool_policy": None, "tools": ["bash"]}}}})


#: A stand-in for a systemd-run that expands variables in the command it is given, as
#: GitHub's runners' does: ``${...}`` anywhere in an argument, and ``$NAME`` as a whole
#: argument, each becoming the variable's value or nothing.
EXPANDING_SYSTEMD_RUN = """\
import os, re, sys
args = sys.argv[1:]
args = args[args.index("--") + 1:]
def value(name):
    return os.environ.get(name, "") if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) else ""
def expand(arg):
    whole = re.fullmatch(r"\\$([A-Za-z_][A-Za-z0-9_]*)", arg)
    if whole:
        return value(whole.group(1))
    return re.sub(r"\\$\\{([^}]*)\\}", lambda match: value(match.group(1)), arg)
args = [expand(arg) for arg in args]
os.execv(args[0], args)
"""


@pytest.fixture(params=["scoped", "unscoped", "expanding", "delegated"])
def seal_path(request, monkeypatch, tmp_path_factory):
    """Each Linux path a sealed command can take: in a capped systemd user scope (where
    this machine's user manager makes one), without one (per-process limits only, as on
    a CI runner with no user manager), through a systemd-run that expands ``$NAME``
    in its arguments, and in a cgroup of its own in the run's delegated cgroup (where
    this process's cgroup is delegated: `seal_cgroup`). macOS has one path."""
    if not sys.platform.startswith("linux"):
        if request.param != "scoped":
            pytest.skip("macOS has one path")
        return request.param
    monkeypatch.setattr(seal, "_SCOPE", [])
    if request.param == "unscoped":
        monkeypatch.setattr(seal, "_scope_prefix", lambda: [])
        monkeypatch.setattr(seal.seal_cgroup, "_proc_cgroup", lambda: "")
    elif request.param == "delegated":
        from test_seal_cgroup import _not_delegated

        if _not_delegated():
            pytest.skip(_not_delegated())
        monkeypatch.setattr(seal, "_scope_prefix", lambda: [])
    elif request.param == "expanding":
        fake = tmp_path_factory.mktemp("fake-systemd") / "systemd-run"
        fake.write_text(f"#!{sys.executable}\n" + EXPANDING_SYSTEMD_RUN)
        fake.chmod(0o755)
        monkeypatch.setattr(seal, "_SCOPE", [(str(fake), "--user", "--scope", "--")])
    elif not seal.linux_scope():
        pytest.skip("no systemd user scope here")
    return request.param


# ── the seal itself ───────────────────────────────────────────────────────────

def _layout(tmp_path: Path) -> tuple[Path, workspace.Workspace]:
    """A run directory with two items, the grader's file, and a gold answer outside it, and
    item ``a``'s sealed step prepared as both runtimes prepare it."""
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "gold.json").write_text('{"answer": "GOLD-OUTSIDE"}')
    run = tmp_path / "run"
    (run / "items" / "b").mkdir(parents=True)
    (run / "items" / "b" / "answer.txt").write_text("OTHER-ITEM")
    (run / "grader.json").write_text('{"answer": "GOLD-IN-RUN"}')
    (run / "input.txt").write_text("THE-INPUT")
    place = workspace.prepare(run, "probe", "scratch", route={"worker_backend": "dsh"},
                              tools=("bash",), reads=("input.txt",))
    assert place is not None and place.runner
    return run, place


def _sealed(place: workspace.Workspace, script: str, **options) -> subprocess.CompletedProcess:
    """``script`` run as DSH runs a sealed step's command: the runner, its bwrap-style
    profile arguments, ``--``, then ``bash -c``."""
    argv = [*place.runner, "--ro-bind", "/", "/", "--bind", str(place.path), str(place.path),
            "--", "bash", "-c", script]
    return subprocess.run(argv, cwd=place.path, capture_output=True, text=True, timeout=60,
                          env={**os.environ, "REFACTORY_SECRET_PROBE": "LEAKED-ENV"}, **options)


#: Prints the arguments and environment of the step's parent process (the DSH process,
#: here the test) where the platform lets a process read another's: Linux through
#: /proc, macOS through sysctl(KERN_PROCARGS2).
PARENT_ENV = r"""python3 - "$PPID" <<'EOF'
import ctypes, os, sys
pid = int(sys.argv[1])
for path in (f"/proc/{pid}/environ", f"/proc/{pid}/cmdline", "/proc/1/environ"):
    try:
        print(open(path, "rb").read().decode(errors="replace"))
    except OSError:
        pass
if sys.platform == "darwin":
    libc = ctypes.CDLL(None)
    mib = (ctypes.c_int * 3)(1, 49, pid)
    size = ctypes.c_size_t(1 << 20)
    buf = ctypes.create_string_buffer(size.value)
    if libc.sysctl(mib, 3, buf, ctypes.byref(size), None, 0) == 0:
        print(buf.raw[:size.value].decode(errors="replace"))
EOF"""


def test_a_sealed_command_sees_its_inputs_and_nothing_else(tmp_path, seal_path):
    if seal_path == "expanding":
        pytest.skip("these probes' own ${...} would be expanded; the hostile step covers this path")
    _sealable()
    run, place = _layout(tmp_path)
    home = os.path.expanduser("~")
    # A real file of the person's home: ~/.zshrc where there is one, else the first
    # readable file there (the CI runner's own dotfiles).
    home_files = [os.path.join(home, ".zshrc")] + sorted(
        os.path.join(home, name) for name in os.listdir(home))
    home_file = next((path for path in home_files if os.path.isfile(path)
                      and os.access(path, os.R_OK) and os.path.getsize(path) > 0), "")
    probes = {
        "input": "cat input.txt",
        "grader": f"cat {run / 'grader.json'}",
        "gold": f"cat {tmp_path / 'bench' / 'gold.json'}",
        "other item": f"cat {run / 'items' / 'b' / 'answer.txt'}",
        "other item, relative": "cat ../../../../items/b/answer.txt",
        "run listing": f"ls {run}",
        "home listing": f"ls -A {home}",
        "home file": f"cat {shlex.quote(home_file)}" if home_file else "true",
        "link out": f"ln -s {run / 'grader.json'} out-link && cat out-link",
        "write outside": f"echo x > {run / 'written.txt'} && echo WROTE",
        "network": "exec 3<>/dev/tcp/1.1.1.1/53 && echo CONNECTED",
        "loopback": "exec 3<>/dev/tcp/127.0.0.1/22 && echo CONNECTED",
        "environment": "env",
        "parent environment": PARENT_ENV,
        "every process": "cat /proc/[0-9]*/environ 2>/dev/null | tr '\\0' '\\n'",
        "unmount": f"for m in /usr /etc /tmp /proc {home} {run} /; do umount -l $m 2>/dev/null; "
                   f"umount $m 2>/dev/null; done; cat {run / 'grader.json'}",
        "remount": f"mount -o remount,rw /usr 2>/dev/null; touch /usr/x /etc/x /x 2>/dev/null "
                   "&& echo WROTE; ls /usr/x /etc/x /x 2>/dev/null",
        "terminals": "for t in /dev/tty* /dev/pts/*; do [ -e \"$t\" ] && exec 9<\"$t\" "
                     "&& echo OPENED $t; done 2>/dev/null; true",
        "homebrew data": "for d in /opt/homebrew/etc /opt/homebrew/var /opt/homebrew/share "
                         "/usr/local/etc /usr/local/var /usr/local/share /usr/local/src; "
                         "do ls -A $d 2>/dev/null; done | head -1",
        "system secrets": "cat /etc/shadow /etc/sudoers /etc/hostname /etc/hosts 2>/dev/null | head -1",
        "limits": "ulimit -t; ulimit -f",
        "settings": "python3 -c 'import os; print(os.cpu_count())'",
        "write inside": "echo made > made.txt && cat made.txt",
    }
    out = {name: _sealed(place, script) for name, script in probes.items()}
    assert out["input"].stdout == "THE-INPUT"
    assert out["write inside"].stdout == "made\n"
    for name in ("grader", "gold", "other item", "other item, relative", "link out",
                 *(("home file",) if home_file else ())):
        assert out[name].returncode != 0 and out[name].stdout == "", (name, out[name])
    assert not any(secret in result.stdout + result.stderr for result in out.values()
                   for secret in ("GOLD-OUTSIDE", "GOLD-IN-RUN", "OTHER-ITEM", "LEAKED-ENV"))
    assert out["run listing"].stdout == "" and out["write outside"].stdout == ""
    assert not (run / "written.txt").exists()
    real_home = set(os.listdir(home))
    listed = set(out["home listing"].stdout.split())
    assert not (listed & real_home - {""}), sorted(listed & real_home)
    for name in ("network", "loopback"):
        assert "CONNECTED" not in out[name].stdout, name
    assert "REFACTORY_" not in out["environment"].stdout
    assert f"HOME={place.path.resolve()}" in out["environment"].stdout
    for name in ("parent environment", "every process"):
        assert "LEAKED-ENV" not in out[name].stdout and "REFACTORY_" not in out[name].stdout, name
    assert out["unmount"].stdout == "", out["unmount"]
    assert "WROTE" not in out["remount"].stdout and out["remount"].stdout == ""
    assert "OPENED" not in out["terminals"].stdout, out["terminals"].stdout
    assert out["homebrew data"].stdout == ""
    assert out["system secrets"].stdout == ""
    assert out["limits"].stdout.split() == ["3600", "4194304"], out["limits"]
    assert out["settings"].stdout.strip().isdigit(), out["settings"]


def test_a_sealed_command_gets_no_descriptor_but_its_own(tmp_path, seal_path):
    _sealable()
    _run, place = _layout(tmp_path)
    with open(tmp_path / "bench" / "gold.json", "rb") as gold:
        result = _sealed(place, f"cat /dev/fd/{gold.fileno()} <&- 2>/dev/null; echo; ls -l /dev/fd/",
                         pass_fds=(gold.fileno(),))
    assert "GOLD-OUTSIDE" not in result.stdout, (result.stdout, result.stderr, place.runner[:16])
    assert result.returncode == 0, ("the command ran", result.stderr)


DETACH = {
    "background": "(sleep 3; echo late > late-bg.txt) & true",
    "setsid": "perl -MPOSIX -e 'fork and exit; POSIX::setsid(); sleep 3; "
              "open(F, \">late-setsid.txt\"); print F 1' & true",
}


@pytest.mark.parametrize("how", sorted(DETACH))
def test_a_process_a_command_leaves_ends_with_it(tmp_path, how):
    """Linux: the PID namespace dies with the command. macOS: the reaper ends every
    process in the command's sandbox, a detached one included, and collecting the
    workspace does so again (`seal_reaper`)."""
    _sealable()
    _run, place = _layout(tmp_path)
    _sealed(place, DETACH[how])
    import time
    time.sleep(5)
    assert not list(place.path.glob("late-*.txt")), how


@pytest.mark.skipif(sys.platform != "darwin", reason="the reaper is macOS's PID namespace")
def test_collecting_a_sealed_workspace_ends_what_its_runner_could_not(tmp_path):
    """A runner killed outright cannot reap; collect does, and only this step's sandbox."""
    _sealable()
    from refactory_dsh import seal_reaper
    _run, place = _layout(tmp_path)
    (tmp_path / "other").mkdir()
    _o, other = _layout(tmp_path / "other")
    argv = [*place.runner, "--", "bash", "-c", "sleep 30"]
    others = [*other.runner, "--", "bash", "-c", "sleep 30"]
    mine = subprocess.Popen(argv[4:], cwd=place.path)  # sandbox-exec onward: no reaper
    theirs = subprocess.Popen(others[4:], cwd=other.path)
    try:
        import time
        time.sleep(1)
        assert seal_reaper.sealed_pids(str(place.path)), "the sealed process is found"
        workspace.collect(place, _run, [])
        assert mine.wait(timeout=5) == -9
        assert theirs.poll() is None, "another step's sealed process is left alone"
    finally:
        theirs.kill()
        mine.kill()


def test_a_seal_that_cannot_be_set_up_runs_nothing(tmp_path):
    _sealable()
    _run, place = _layout(tmp_path)
    marker = tmp_path / "ran"
    # No ``--`` before the command: the runner refuses rather than guess what to run.
    argv = [*place.runner, "touch", str(marker)]
    result = subprocess.run(argv, cwd=place.path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 97 and seal.FAILURE in result.stderr
    assert not marker.exists()


def test_dsh_puts_the_runner_in_front_of_each_command(tmp_path):
    """DSH's own local sandbox provider, given the runner as its ``runnerCommand``, wraps a
    ``bash -c`` the way the bash tool asks it to, and the wrapped command is sealed."""
    _sealable()
    if not (ROOT / "node_modules" / ".pnpm").is_dir():
        pytest.skip("DSH is not installed here")
    _run, place = _layout(tmp_path)
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"runner": list(place.runner), "workspace": str(place.path),
                                   "command": "cat ../../../../grader.json; cat input.txt",
                                   "failure": seal.FAILURE.strip()}))
    result = subprocess.run(["node", str(Path(__file__).with_name("sealed_tools_test.mjs")),
                             str(ROOT), str(request), "provider"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["enforcement"] == "full"
    assert observed["argv"][:len(place.runner)] == list(place.runner)
    assert observed["stdout"] == "THE-INPUT" and "GOLD" not in observed["stderr"]


def test_the_patch_mounts_the_runner_and_the_tools_plugin_admits_only_the_listed_tools(tmp_path):
    """The generated overlay replaces DSH's sandbox provider's runner and adds the tools
    plugin; through a real AgentLoop, a tool the step does not list never executes and
    never reaches the provider."""
    if not (ROOT / "node_modules" / ".pnpm").is_dir():
        pytest.skip("DSH is not installed here")
    patch = tmp_path / "patch.yml"
    workers.write_patch(patch, ROOT, {"worker_backend": "dsh", "permission_mode": "skip",
                                      "provider": "x", "model": "y"},
                        workspace_root=tmp_path, tool_home=tmp_path / ".refactory",
                        seal={"tools": ["bash"], "runner": ["/bin/echo", "runner"]})
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"patch": str(patch), "cwd": str(tmp_path)}))
    result = subprocess.run(["node", str(Path(__file__).with_name("sealed_tools_test.mjs")),
                             str(ROOT), str(request), "plugin"],
                            capture_output=True, text=True, timeout=60,
                            env={"PATH": os.environ.get("PATH", "/usr/bin"), "HOME": str(tmp_path)})
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["sandbox"] == {"runnerCommand": ["/bin/echo", "runner"],
                                   "runnerFailureSignatures": [seal.FAILURE.strip()]}
    assert observed["offered"] == ["bash"]
    assert observed["requests"] == 5, observed
    assert observed["hostile_executions"] == 0 and observed["bash_executions"] == 1


def test_a_sealed_step_is_refused_where_it_cannot_run_sealed(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    for route in ({"worker_backend": "claude_code"}, {"worker_backend": "dsh", "permission_mode": "default"}):
        with pytest.raises(workspace.WorkspaceError, match="only the dsh worker backend"):
            workspace.prepare(run, "probe", "scratch", route=route, tools=("bash",))
    with pytest.raises(workspace.WorkspaceError, match="from bash"):
        workspace.prepare(run, "probe", "scratch", route={"worker_backend": "dsh"}, tools=("read",))
    monkeypatch.setattr(seal, "available", lambda: (False, "no namespaces"))
    with pytest.raises(workspace.WorkspaceError, match="no namespaces"):
        workspace.prepare(run, "probe", "scratch", route={"worker_backend": "dsh"}, tools=("bash",))
    # The person's choice to run workspaces unconfined never unseals a sealed step.
    monkeypatch.setenv(*workspace.UNCONFINED_OPT_IN)
    with pytest.raises(workspace.WorkspaceError, match="no namespaces"):
        workspace.prepare(run, "probe", "scratch", route={"worker_backend": "dsh"}, tools=("bash",))


def test_a_read_naming_a_file_outside_the_run_directory_is_refused(tmp_path):
    _sealable()
    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "gold.txt").write_text("GOLD")
    (run / "link.txt").symlink_to(tmp_path / "gold.txt")
    for name in ("../gold.txt", "link.txt"):
        with pytest.raises(workspace.WorkspaceError, match="not a file of the run directory"):
            workspace.prepare(run, "probe", "scratch", route={"worker_backend": "dsh"},
                              tools=("bash",), reads=(name,))


# ── both runtimes ─────────────────────────────────────────────────────────────

#: A stand-in DSH CLI: runs each probe through the ``runnerCommand`` its patch gives DSH's
#: sandbox provider, as DSH's bash tool would, and writes the step's answer from them.
AGENT = """\
import json, pathlib, subprocess, sys
cwd = pathlib.Path.cwd()
last = max(i for i, arg in enumerate(sys.argv) if arg == '--patch')
patch = json.loads(pathlib.Path(sys.argv[last + 1]).read_text())
runner = next(row['config']['runnerCommand'] for row in patch if row.get('id') == 'sandbox')
allow = next(entry['config']['allow'] for row in patch for entry in row.get('insert', [])
             if entry.get('id') == 'refactory-tools')
def bash(script):
    done = subprocess.run([*runner, '--ro-bind', '/', '/', '--', 'bash', '-c', script],
                          cwd=cwd, capture_output=True, text=True, timeout=60)
    return done.stdout.strip() or 'DENIED'
policy = [row['config']['mode'] for row in patch if row.get('id') == 'sandbox-policy']
answer = {'tools': allow, 'policy': policy, 'input': bash('cat input.txt'),
          'grader': bash('cat {grader}'), 'other': bash('cat {other}')}
bash('echo ' + json.dumps(json.dumps(answer)) + ' > answer.txt')
"""


@pytest.mark.parametrize("runtime", ["walker", "spine"])
def test_a_sealed_step_runs_sealed_on_both_runtimes_and_its_output_is_collected(tmp_path, runtime):
    _sealable()
    run = tmp_path / "run"
    (run / "items" / "b").mkdir(parents=True)
    (run / "items" / "b" / "answer.txt").write_text("OTHER-ITEM")
    (run / "grader.json").write_text("GOLD-IN-RUN")
    (run / "input.txt").write_text("THE-INPUT")
    script = tmp_path / "agent-cli"
    script.write_text(f"#!{sys.executable}\n" + AGENT.replace("{grader}", str(run / "grader.json"))
                      .replace("{other}", str(run / "items" / "b" / "answer.txt")))
    script.chmod(0o755)
    client = DshModelClient(worker_backend="dsh", dsh_cli=str(script), attempts=1)
    graph = Graph()
    graph.add(Node("solve", "agent", prompt_template="Solve it.", reads=("input.txt",),
                   writes=("answer.txt",), tools=("bash",), workspace="scratch"))
    if runtime == "walker":
        ctx = RunContext(run, knobs={}, facts={}, task={})
        ctx.model = client
        Executor(graph, ctx).run(start=graph.start)
    else:
        from refactory_dsh import worker_node
        state = {"graph": graph.to_state(), "knobs": {}}
        lowered = workers.lower_workflow(lowering.to_workflow(state), run, client.context(),
                                         lowering.node_declarations(state))
        spec = json.loads(Path(shlex.split(lowered.nodes["solve"].dispatch_command)[-1]).read_text())
        assert spec["tools"] == ["bash"]
        worker_node.execute(spec, run)
    answer = json.loads((run / "answer.txt").read_text())
    # DSH's own sandbox policy for the step is workspace-write, never full access.
    assert answer == {"tools": ["bash"], "policy": ["workspace-write"], "input": "THE-INPUT",
                      "grader": "DENIED", "other": "DENIED"}


def _claude_code_client(tmp_path: Path, **overrides) -> DshModelClient:
    """A client routed as the LiveMath arms are (Claude Code, ``dontAsk``), with a DSH home
    and provider routes of its own: ``vertex`` lists the arm's model."""
    settings = dict(worker_backend="claude_code", permission_mode="dontAsk",
                    model="claude-opus-4-6", effort="medium", attempts=1,
                    dsh_home=str(tmp_path / "dsh-home"),
                    provider_config={"vertex": {"models": [{"id": "claude-opus-4-6"}]}})
    return DshModelClient(**{**settings, **overrides})


@pytest.mark.parametrize("runtime", ["walker", "spine"])
def test_a_sealed_step_of_a_claude_code_run_runs_sealed_on_dsh_with_the_run_s_model(
        tmp_path, runtime):
    """The LiveMath arms' route: a sealed step moves to the dsh backend in skip mode with the
    arm's model and effort, on both runtimes; a step without tools keeps the arm's route."""
    _sealable()
    run = tmp_path / "run"
    (run / "items" / "b").mkdir(parents=True)
    (run / "items" / "b" / "answer.txt").write_text("OTHER-ITEM")
    (run / "grader.json").write_text("GOLD-IN-RUN")
    (run / "input.txt").write_text("THE-INPUT")
    argv = tmp_path / "argv.json"
    script = tmp_path / "agent-cli"
    script.write_text(f"#!{sys.executable}\n" + AGENT.replace("{grader}", str(run / "grader.json"))
                      .replace("{other}", str(run / "items" / "b" / "answer.txt"))
                      + f"pathlib.Path({str(argv)!r}).write_text(json.dumps(sys.argv))\n")
    script.chmod(0o755)
    client = _claude_code_client(tmp_path, dsh_cli=str(script))
    assert client.for_sealed().route() == {
        "worker_backend": "dsh", "provider": "vertex", "model": "claude-opus-4-6",
        "effort": "medium", "permission_mode": "skip"}
    assert workspace.sealed_steps(client) == (True, "")
    graph = Graph()
    graph.add(Node("solve", "agent", prompt_template="Solve it.", reads=("input.txt",),
                   writes=("answer.txt",), tools=("bash",), workspace="scratch"))
    if runtime == "walker":
        ctx = RunContext(run, knobs={}, facts={}, task={})
        ctx.model = client
        result = Executor(graph, ctx).run(start=graph.start)
        assert result.exit_ok, result.error
    else:
        from refactory_dsh import worker_node
        graph.add(Node("plain", "agent", prompt_template="Think.", writes=("plain.txt",),
                       tool_policy="none"))
        state = {"graph": graph.to_state(), "knobs": {}}
        lowered = workers.lower_workflow(lowering.to_workflow(state), run, client.context(),
                                         lowering.node_declarations(state))
        specs = {name: json.loads(Path(shlex.split(lowered.nodes[name].dispatch_command)[-1])
                                  .read_text()) for name in ("solve", "plain")}
        assert specs["plain"]["model_context"]["worker_backend"] == "claude_code"
        assert specs["solve"]["model_context"]["worker_backend"] == "dsh"
        worker_node.execute(specs["solve"], run)
    answer = json.loads((run / "answer.txt").read_text())
    assert answer == {"tools": ["bash"], "policy": ["workspace-write"], "input": "THE-INPUT",
                      "grader": "DENIED", "other": "DENIED"}
    called = json.loads(argv.read_text())
    assert called[called.index("--provider") + 1] == "vertex"
    assert called[called.index("--model") + 1] == "claude-opus-4-6"


def test_without_dsh_a_tool_step_is_never_offered_and_is_refused_on_the_same_answer(tmp_path):
    """No DSH runtime, or no DSH provider for the run's model: the loop's request offers no
    tool step, and its attempts refuse one (`workspace.sealed_steps`, one answer)."""
    _sealable()
    from refactory_dsh.loop import Loop, LoopConfig
    from refactory_dsh.optim import Candidate, Proposal
    from test_pipeline_view import _mode

    mode = _mode("livemath/skill_graph")
    seed = Candidate.from_state(mode.state_dict())
    edit = {"op": "insert", "after": "compare", "kind": "agent", "text": "Check it.",
            "tools": ["bash"], "to": "decide_task"}
    clients = {"": _claude_code_client(tmp_path, dsh_cli="/bin/true"),
               "no DSH runtime": _claude_code_client(tmp_path, dsh_cli="",
                                                     runtime_root=str(tmp_path / "none")),
               "no DSH provider": _claude_code_client(tmp_path, dsh_cli="/bin/true",
                                                      provider_config={})}
    for why, client in clients.items():
        loop = Loop("livemath/skill_graph", LoopConfig(runs_root=tmp_path / "runs" / (why or "ok")),
                    project=Path("."), model=client)
        assert loop._tools_offered() is (not why), why
        out = loop.optimizer.attempt(Proposal(edits=[edit]), seed, mode.invariants(), True, (),
                                     protected=mode.edit_protected())
        if not why:
            assert isinstance(out, Candidate), out
        else:
            assert isinstance(out, list) and out[0].startswith("E-new-node-tools"), out
            assert why in out[0], out


@pytest.mark.parametrize("runtime", ["walker", "spine"])
def test_an_unsealable_tools_list_is_refused_on_both_runtimes(tmp_path, runtime):
    graph = Graph()
    graph.add(Node("solve", "agent", prompt_template="Solve it.", writes=("answer.txt",),
                   tools=("read",), workspace="scratch"))
    client = DshModelClient(worker_backend="dsh", attempts=1)
    if runtime == "walker":
        ctx = RunContext(tmp_path, knobs={}, facts={}, task={})
        ctx.model = client
        result = Executor(graph, ctx).run(start=graph.start)
        assert not result.exit_ok and "cannot run sealed" in str(result.error), result.error
    else:
        state = {"graph": graph.to_state(), "knobs": {}}
        with pytest.raises(ValueError, match="tool restrictions"):
            workers.lower_workflow(lowering.to_workflow(state), tmp_path, client.context(),
                                   lowering.node_declarations(state))


def test_the_contract_points_a_sealed_step_at_its_own_copies():
    from refactory_dsh.agent_delivery import _execution_contract

    described = {"kind": "scratch", "collect": "files", "path": "/w/scratch-1", "sealed": True,
                 "tools": ["bash"], "sandbox": "sealed: its files only, no network"}
    prompt, execution = _execution_contract(Path("/run"), "solve", "Go.", ["input.txt"],
                                            ["answer.txt"], {}, workspace=described)
    assert execution["inputs"] == ["/w/scratch-1/input.txt"]
    assert execution["outputs"] == ["/w/scratch-1/answer.txt"]
    assert "workdir" not in execution and "python" not in execution
    assert "there is no network" in prompt and '"workdir"' not in prompt


@pytest.mark.parametrize("runner", ["fails", "does not seal"])
def test_where_the_seal_cannot_be_made_the_step_is_refused_never_run_unsealed(
        tmp_path, monkeypatch, runner):
    """A machine whose namespaces (or Seatbelt) are unavailable gets a runner that fails;
    one that starts but does not seal runs the probe's command with full access. Either
    way the probe says the seal is unavailable, and a sealed step is refused: there is
    no write-only fallback."""
    fake = tmp_path / "fake-runner"
    body = ("echo 'unshare: unshare failed: Operation not permitted' >&2; exit 1"
            if runner == "fails" else
            'while [ "$1" != "--" ]; do shift; done; shift; exec "$@"')
    fake.write_text(f"#!/bin/sh\n{body}\n")
    fake.chmod(0o755)
    if sys.platform == "darwin":
        monkeypatch.setattr(seal.snippet, "SANDBOX_EXEC", str(fake))
        monkeypatch.setattr(seal, "_MAC_SHIM", 'while [ "$1" != "--" ]; do shift; done; shift; exec "$@"')
    else:
        monkeypatch.setattr(seal.shutil, "which", lambda name: str(fake))
    seal.available.cache_clear()
    try:
        ok, why = seal.available()
        assert not ok, why
        run = tmp_path / "run"
        run.mkdir()
        with pytest.raises(workspace.WorkspaceError, match="the seal cannot run here"):
            workspace.prepare(run, "probe", "scratch", route={"worker_backend": "dsh"},
                              tools=("bash",))
    finally:
        seal.available.cache_clear()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux's allow-list root")
def test_a_sealed_linux_command_sees_only_the_listed_parts_of_etc_and_runs_unprivileged(tmp_path, seal_path):
    _sealable()
    _run, place = _layout(tmp_path)
    result = _sealed(place, "ls -A /etc; echo uid=$(id -u); ulimit -u")
    listed = set(result.stdout.split("uid=")[0].split())
    assert listed <= {path.removeprefix("/etc/") for path in seal.LINUX_ETC}, listed
    assert f"uid={os.getuid()}" in result.stdout
    assert result.stdout.split()[-1] == ("512" if place.whole_step_cap[0] else "128")


def test_a_sealed_step_never_runs_as_root(tmp_path, monkeypatch):
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux maps the caller's uid into the seal")
    monkeypatch.setattr(seal.os, "getuid", lambda: 0)
    with pytest.raises(seal.SealError, match="does not run as root"):
        seal.runner_command(tmp_path)


def test_collect_moves_a_sealed_workspace_out_and_never_copies_through_a_link(tmp_path):
    _sealable()
    run, place = _layout(tmp_path)
    _sealed(place, f"mkdir -p real && echo mine > real/a.txt && ln -s {tmp_path / 'bench'} linked "
                   "&& echo ok > answer.txt")
    with pytest.raises(workspace.WorkspaceError, match="under one"):
        workspace.collect(place, run, ["linked/gold.json"])
    assert place.path.name.endswith(".collected"), "moved off the path its sandbox may write"
    assert not (run / "linked").exists()
    with pytest.raises(workspace.WorkspaceError, match="as a symlink"):
        (place.path / "alias.txt").symlink_to(place.path / "answer.txt")
        workspace.collect(place, run, ["alias.txt"])
    copied = workspace.collect(place, run, ["answer.txt", "real/a.txt"])
    assert copied["copied"] == ["answer.txt", "real/a.txt"]
    assert (run / "answer.txt").read_text() == "ok\n"


# ── a hostile step, end to end ────────────────────────────────────────────────

def _listener(port: int):
    """A TCP listener on loopback at ``port`` (0: any free one), or ``None`` where that
    port is taken (the lab's model proxy holds 8082: it is then the real target)."""
    import socket

    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        sock.close()
        return None
    sock.listen(8)
    return sock


def test_a_hostile_step_is_blocked_at_every_turn_and_its_own_output_is_collected(tmp_path, seal_path):
    """A step that tries every way out: the gold answer beside its input, the person's
    keys and credentials, writing outside, a link out at collection, the model proxy, the
    network and DNS, and becoming root. Each attempt prints a marker only when it
    succeeds, so no secret is ever printed; the step's own output is still collected."""
    _sealable()
    home = Path(os.path.expanduser("~"))
    run = tmp_path / "run"
    item = run / "items" / "a"
    item.mkdir(parents=True)
    (item / "input.txt").write_text("THE-INPUT")
    (item / "expected.txt").write_text("GOLD-BESIDE-INPUT")
    (run / "items" / "b").mkdir()
    (run / "items" / "b" / "expected.txt").write_text("GOLD-OTHER-ITEM")
    credentials = tmp_path / "creds" / "service-account.json"
    credentials.parent.mkdir()
    credentials.write_text('{"private_key": "PLANTED-KEY"}')
    # The person's real files where they exist, and the run's credentials file (the
    # DSH process's GOOGLE_APPLICATION_CREDENTIALS where it has one, else a planted one).
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or str(credentials)
    secrets = {"ssh dir": home / ".ssh", "claude.json": home / ".claude.json",
               "credentials": Path(gac), "planted credentials": credentials,
               "gold beside input": item / "expected.txt",
               "gold of another item": run / "items" / "b" / "expected.txt"}
    place = workspace.prepare(run, "hostile", "scratch", route={"worker_backend": "dsh"},
                              tools=("bash",), reads=("items/a/input.txt",))
    assert place is not None and place.runner
    assert sorted(os.listdir(place.path / "items" / "a")) == ["input.txt"], "only the input is staged"

    proxy = _listener(8082)
    free = _listener(0)
    ports = [8082, free.getsockname()[1]]

    def q(path: Path | str) -> str:
        return shlex.quote(str(path))

    reads = "; ".join(
        f"{{ cat {q(path)} || ls -A {q(path)}; }} >/dev/null 2>&1 && test -n \"$(cat {q(path)} "
        f"2>/dev/null; ls -A {q(path)} 2>/dev/null)\" && echo 'GOT:{name}'"
        for name, path in secrets.items())
    script = "\n".join([
        # The gold beside the input, by its absolute path above, relatively from the
        # staged copy, and by searching for it.
        reads,
        f"cat {q(os.path.relpath(item / 'expected.txt', place.path))} >/dev/null 2>&1 "
        "&& echo 'GOT:gold relative'",
        f"find {q(tmp_path)} / -maxdepth 4 -name expected.txt -print 2>/dev/null | grep -q . "
        "&& echo 'GOT:gold found'",
        f"grep -rl GOLD- {q(tmp_path)} 2>/dev/null | grep -q . && echo 'GOT:gold grep'",
        "cat \"$GOOGLE_APPLICATION_CREDENTIALS\" >/dev/null 2>&1 && echo 'GOT:credentials by env'",
        # Writing outside the workspace.
        f"echo x > {q(run / 'planted.txt')} 2>/dev/null && echo 'WROTE:run dir'",
        f"echo x > {q(home / '.refactory-hostile-probe')} 2>/dev/null && echo 'WROTE:home'",
        "echo x > /tmp/refactory-hostile-probe 2>/dev/null && echo 'WROTE:/tmp'",
        "echo x > ../escaped.txt 2>/dev/null && echo 'WROTE:above the workspace'",
        # Links for the collector to follow out: the answer a link to the gold, an output
        # directory a link to the run directory, and a directory output holding a link.
        f"ln -s {q(item / 'expected.txt')} linked-answer.txt",
        f"ln -s {q(run)} linked-dir",
        f"mkdir -p report && ln -s {q(item / 'expected.txt')} report/gold.txt "
        "&& echo 'MINE' > report/own.txt",
        # The model proxy, any loopback listener, the internet and DNS.
        *(f"(exec 3<>/dev/tcp/127.0.0.1/{port}) 2>/dev/null && echo 'NET:loopback {port}'"
          for port in ports),
        "(exec 3<>/dev/tcp/8.8.8.8/53) 2>/dev/null && echo 'NET:8.8.8.8:53 tcp'",
        "python3 -c \"import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); "
        "s.settimeout(3); s.sendto(b'x', ('8.8.8.8', 53)); print('NET:8.8.8.8:53 udp')\" 2>/dev/null",
        "python3 -c \"import socket; socket.getaddrinfo('example.com', 443); print('NET:dns')\" "
        "2>/dev/null",
        "getent hosts example.com >/dev/null 2>&1 && echo 'NET:dns getent'",
        # Root: as it starts, through setuid, a namespace of its own, sudo and su.
        "[ \"$(id -u)\" = 0 ] && echo 'ROOT:id'",
        "python3 -c 'import os; os.setuid(0); print(\"ROOT:setuid\")' 2>/dev/null",
        "[ \"$(unshare -r id -u 2>/dev/null)\" = 0 ] && echo 'ROOT:unshare'",
        "sudo -n true >/dev/null 2>&1 && echo 'ROOT:sudo'",
        "su -c true root </dev/null >/dev/null 2>&1 && echo 'ROOT:su'",
        # Its own work.
        "echo 'MY-ANSWER' > answer.txt",
        "true",
    ])
    env_extra = {"GOOGLE_APPLICATION_CREDENTIALS": gac}
    argv = [*place.runner, "--ro-bind", "/", "/", "--bind", str(place.path), str(place.path),
            "--", "bash", "-c", script]
    try:
        result = subprocess.run(argv, cwd=place.path, capture_output=True, text=True, timeout=300,
                                env={**os.environ, **env_extra})
    finally:
        for sock in (proxy, free):
            if sock is not None:
                sock.close()
    escaped = [line for line in result.stdout.splitlines()
               if line.split(":", 1)[0] in ("GOT", "WROTE", "NET", "ROOT")]
    assert result.returncode == 0, result.stderr[-2000:]
    assert escaped == [], escaped
    assert not (run / "planted.txt").exists() and not (home / ".refactory-hostile-probe").exists()
    assert not (place.path.parent / "escaped.txt").exists()
    for secret in ("GOLD-BESIDE-INPUT", "GOLD-OTHER-ITEM", "PLANTED-KEY"):
        assert secret not in result.stdout + result.stderr, secret

    # Collection follows no link out, and still takes the step's own output.
    for name in ("linked-answer.txt", "linked-dir/items/a/expected.txt"):
        with pytest.raises(workspace.WorkspaceError, match="symlink, or under one"):
            workspace.collect(place, run, [name])
    copied = workspace.collect(place, run, ["answer.txt", "report"])
    assert copied["copied"] == ["answer.txt", "report"]
    assert (run / "answer.txt").read_text() == "MY-ANSWER\n"
    assert sorted(os.listdir(run / "report")) == ["own.txt"], "the link inside it is left out"
    assert not (run / "linked-answer.txt").exists()


def test_collection_reads_no_link_a_still_running_process_swaps_in(tmp_path):
    """A sealed process that outlives its command (as on macOS before the reaper ends it)
    swaps the answer between a file and a link to the gold answer while it is collected:
    every collection copies the file or refuses the link, never the gold."""
    _sealable()
    run, place = _layout(tmp_path)
    gold = tmp_path / "bench" / "gold.json"
    # The answer is swapped between a file and a link to the gold; the directory output
    # between a real one and a link to the gold's directory.
    swapper = subprocess.Popen(
        [sys.executable, "-c", "import os, sys\nws, gold = sys.argv[1:]\nos.chdir(ws)\n"
         "os.mkdir('dA'); open('dA/x', 'w').write('MINE')\n"
         "os.symlink(os.path.dirname(gold), 'dL')\n"
         "while True:\n"
         "    open('a.tmp', 'w').write('MINE')\n    os.replace('a.tmp', 'answer.txt')\n"
         "    os.symlink(gold, 'l.tmp')\n    os.replace('l.tmp', 'answer.txt')\n"
         "    os.rename('dA', 'dir')\n    os.rename('dir', 'dA')\n"
         "    os.rename('dL', 'dir')\n    os.rename('dir', 'dL')\n",
         str(place.path), str(gold)])
    unsealed = workspace.Workspace(**{**place.__dict__, "runner": (), "tools": ()})
    try:
        outcomes = {"copied": 0, "refused": 0}
        for _ in range(300):
            try:
                workspace.collect(unsealed, run, ["answer.txt", "dir"])
            except workspace.WorkspaceError:
                outcomes["refused"] += 1
                continue
            outcomes["copied"] += 1
            if (run / "answer.txt").exists():
                assert (run / "answer.txt").read_text() in ("", "MINE")
            if (run / "dir").exists():
                for name in os.listdir(run / "dir"):
                    assert "GOLD" not in (run / "dir" / name).read_text()
    finally:
        swapper.kill()
        swapper.wait()
    assert outcomes["copied"] and outcomes["refused"], outcomes


# ── Linux: the namespace, binds, limits and processes ─────────────────────────

LINUX = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux's namespaces")


@LINUX
def test_where_user_namespaces_work_the_seal_runs_in_its_own_pid_namespace(tmp_path):
    """Not a skip where the machine can make namespaces: a runner that lost one of them
    (the PID namespace, say) fails here instead of turning every seal test into a skip."""
    if subprocess.run(["unshare", "-Urnm", "true"], capture_output=True).returncode != 0:
        pytest.skip("this machine makes no user namespaces")
    seal.available.cache_clear()
    try:
        assert seal.available() == (True, "")
    finally:
        seal.available.cache_clear()
    _run, place = _layout(tmp_path)
    result = _sealed(place, "ls /proc | grep -c '^[0-9]*$'; readlink /proc/self/ns/pid; "
                            "readlink /proc/1/exe")
    count, pid_ns, init = result.stdout.split()
    assert int(count) < 10, result.stdout
    assert pid_ns != os.readlink("/proc/self/ns/pid")


@LINUX
def test_each_bind_is_read_only_even_where_its_owner_could_write(tmp_path, monkeypatch):
    """A folder the user owns, bound in like a system folder: readable, never writable.
    (In the home directory: the runner covers /tmp while it builds the new root.)"""
    import tempfile

    _sealable()
    home = Path.home().resolve()
    if home == Path("/tmp") or home.is_relative_to("/tmp"):
        pytest.skip("the home directory is under /tmp, which the runner covers")
    bound = Path(tempfile.mkdtemp(prefix=".seal-bound-", dir=home))
    try:
        (bound / "lib.txt").write_text("LIB")
        monkeypatch.setattr(seal, "LINUX_READS", (*seal.LINUX_READS, str(bound)))
        _run, place = _layout(tmp_path)
        result = _sealed(place, f"cat {bound}/lib.txt; echo x > {bound}/new.txt && echo WROTE; "
                                f"echo x >> {bound}/lib.txt && echo APPENDED")
        assert result.stdout.split() == ["LIB"], result
        assert not (bound / "new.txt").exists() and (bound / "lib.txt").read_text() == "LIB"
    finally:
        shutil.rmtree(bound)


@LINUX
def test_a_sealed_command_is_capped_as_a_whole_or_each_process_tightly(tmp_path):
    _sealable()
    _run, place = _layout(tmp_path)
    argv = [*place.runner, "--", "bash", "-c", "cat /proc/self/cgroup; ulimit -u; ulimit -v; sleep 30"]
    command = subprocess.Popen(argv, cwd=place.path, stdout=subprocess.PIPE, text=True)
    try:
        cgroup = command.stdout.readline().strip().removeprefix("0::")
        limits = [command.stdout.readline().strip(), command.stdout.readline().strip()]
        if place.cgroup:
            # A delegated cgroup caps the step (`seal_cgroup`): the command runs in it.
            assert f"/sys/fs/cgroup{cgroup}" == place.cgroup
            assert Path(f"{place.cgroup}/memory.max").read_text().strip() == str(8 << 30)
            assert Path(f"{place.cgroup}/pids.max").read_text().strip() == "256"
            assert limits == ["512", "unlimited"]
        elif seal.linux_scope():
            assert Path(f"/sys/fs/cgroup{cgroup}/memory.max").read_text().strip() == str(8 << 30)
            assert Path(f"/sys/fs/cgroup{cgroup}/pids.max").read_text().strip() == "256"
            assert limits == ["512", "unlimited"]
        else:
            assert limits == ["128", "8388608"]
    finally:
        command.kill()
        command.wait()
    # Where the person's user manager can make a scope, the seal uses one.
    prefix = seal._scope_prefix()
    if prefix and subprocess.run([*prefix, "true"], capture_output=True).returncode == 0:
        assert seal.linux_scope()


@LINUX
def test_a_sealed_command_has_a_session_keyring_of_its_own(tmp_path):
    if not shutil.which("keyctl"):
        pytest.skip("no keyctl here")
    _sealable()
    _run, place = _layout(tmp_path)
    mine = subprocess.run(["keyctl", "show", "@s"], capture_output=True, text=True).stdout
    theirs = _sealed(place, "keyctl show @s").stdout
    assert len(theirs.splitlines()) == 2, theirs  # a keyring of its own, empty
    # A process that outlives the login that started it (a detached lab run) holds a
    # revoked session keyring: then there is no keyring of ours to compare against, and
    # only the comparison is left out; the sealed step's own keyring is still checked.
    if len(mine.splitlines()) >= 2:
        assert theirs.splitlines()[1].split()[0] != mine.splitlines()[1].split()[0]


def _ticker(place: workspace.Workspace) -> list[str]:
    """The runner argv for a command that writes ``tick`` into the workspace every 0.2 s,
    from a process it detaches."""
    return [*place.runner, "--ro-bind", "/", "/", "--", "bash", "-c",
            "(setsid bash -c 'while true; do date +%s%N > tick; sleep 0.2; done' &); sleep 600"]


def _ticks_stopped(path: Path) -> bool:
    before = path.read_text()
    import time
    time.sleep(1.0)
    return path.read_text() == before


@LINUX
def test_a_runner_dies_with_the_process_that_started_it(tmp_path):
    """DSH killed outright: the runner it started, and everything sealed, die with it."""
    _sealable()
    _run, place = _layout(tmp_path)
    parent = subprocess.Popen([sys.executable, "-c", "import subprocess, sys, time\n"
                               "subprocess.Popen(sys.argv[1:])\n"
                               "time.sleep(600)", *_ticker(place)], cwd=place.path)
    tick = place.path / "tick"
    import time
    for _ in range(100):
        if tick.exists():
            break
        time.sleep(0.1)
    assert tick.exists()
    parent.kill()
    parent.wait()
    time.sleep(0.5)
    assert _ticks_stopped(tick)
    assert not seal_reaper._linux_runner_pids(str(place.path))


@LINUX
def test_collect_ends_every_sealed_process_before_it_reads_the_workspace(tmp_path):
    """A runner still going when its step is collected (DSH alive, the command not ended):
    collection kills it and its namespace, and the workspace stops changing."""
    _sealable()
    run, place = _layout(tmp_path)
    runner = subprocess.Popen(_ticker(place), cwd=place.path)
    sealed_at = str(place.path)
    tick = place.path / "tick"
    import time
    try:
        for _ in range(100):
            if tick.exists():
                break
            time.sleep(0.1)
        assert tick.exists()
        assert not _ticks_stopped(tick), "the writer runs before collection"
        workspace.collect(place, run, ["tick"])
        assert _ticks_stopped(place.path / "tick")
        assert runner.wait(timeout=5) is not None
        assert not seal_reaper._linux_runner_pids(sealed_at)
    finally:
        runner.kill()


def test_a_sealed_workspace_over_its_cap_is_not_collected(tmp_path, monkeypatch):
    _sealable()
    run, place = _layout(tmp_path)
    (place.path / "answer.txt").write_bytes(b"x" * 8192)
    monkeypatch.setattr(workspace, "SEALED_WORKSPACE_MAX_BYTES", 4096)
    with pytest.raises(workspace.WorkspaceError, match="over the"):
        workspace.collect(place, run, ["answer.txt"])
    assert not (run / "answer.txt").exists()


@pytest.mark.parametrize("name", [".", "", "items", "items/a", "missing.txt"])
def test_a_read_that_is_not_a_file_of_the_run_is_refused(tmp_path, name):
    """A sealed step reads files, one by one: never the run directory, a folder (it could
    hold another item's answer) or a file the run lacks, and nothing is left behind."""
    _sealable()
    run = tmp_path / "run"
    (run / "items" / "a").mkdir(parents=True)
    (run / "items" / "a" / "input.txt").write_text("IN")
    (run / "items" / "b").mkdir()
    (run / "items" / "b" / "answer.txt").write_text("OTHER-ITEM")
    with pytest.raises(workspace.WorkspaceError, match="not a file|does not have"):
        workspace.prepare(run, "probe", "scratch", route={"worker_backend": "dsh"},
                          tools=("bash",), reads=(name,))
    copied = [path for path in (run / ".refactory").rglob("*") if path.is_file()]
    assert not any("OTHER-ITEM" in path.read_text(errors="replace") for path in copied)
    place = workspace.prepare(run, "probe2", "scratch", route={"worker_backend": "dsh"},
                              tools=("bash",), reads=("items/a/input.txt",))
    assert _sealed(place, "cat items/a/input.txt").stdout == "IN"


@LINUX
def test_a_failed_scope_probe_is_tried_again_and_warns(monkeypatch, capsys):
    calls = []
    real = subprocess.run

    def flaky(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(argv, 20)
        return subprocess.CompletedProcess(argv, 0, stdout=f"{8 << 30}\n", stderr="")

    monkeypatch.setattr(seal, "_SCOPE", [])
    monkeypatch.setattr(seal, "_WARNED", set())
    monkeypatch.setattr(seal, "_scope_prefix", lambda: ["/usr/bin/env", "--"])
    monkeypatch.setattr(seal.subprocess, "run", flaky)
    assert seal.linux_scope() == ()
    assert "the systemd user scope probe failed" in capsys.readouterr().err
    assert seal.linux_scope() == ("/usr/bin/env", "--") and len(calls) == 2
    assert seal.linux_scope() == ("/usr/bin/env", "--") and len(calls) == 2, "kept once found"
    # A probe that fails again is not said again: once per process.
    monkeypatch.setattr(seal, "_SCOPE", [])
    calls.clear()
    assert seal.linux_scope() == () and capsys.readouterr().err == ""
    monkeypatch.setattr(seal.subprocess, "run", real)


@LINUX
def test_no_user_manager_to_ask_is_not_warned_about_on_every_command(monkeypatch, capsys):
    """Where there is no user manager to probe (a system service), no probe runs and
    `linux_scope` says nothing, however often it is asked: `step_cap` then caps the step
    with a delegated cgroup, or says once per reason that it has no whole-step cap."""
    monkeypatch.setattr(seal, "_SCOPE", [])
    monkeypatch.setattr(seal, "_WARNED", set())
    monkeypatch.setattr(seal, "_scope_prefix", lambda: [])
    for _ in range(3):
        assert seal.linux_scope() == ()
    assert capsys.readouterr().err == ""


@LINUX
def test_each_layer_closes_inherited_descriptors(tmp_path, seal_path):
    """Both layers, each on its own: the runner's start (`seal_reaper.exec_runner`) and the
    command's last step inside the seal (`seal._CLOSE`)."""
    _sealable()
    _run, place = _layout(tmp_path)
    reaper = str(Path(seal_reaper.__file__))
    with open(tmp_path / "bench" / "gold.json", "rb") as gold:
        outer = subprocess.run([sys.executable, "-I", reaper, "--exec", "/bin/sh", "-c",
                                f"cat /dev/fd/{gold.fileno()} 2>/dev/null; true"],
                               capture_output=True, text=True, pass_fds=(gold.fileno(),))
        inner_argv = list(place.runner[place.runner.index("--exec") + 1:])
        if inner_argv[0] == seal_reaper.CGROUP_FLAG:
            inner_argv = inner_argv[2:]
        inner = subprocess.run([*inner_argv, "--", "bash", "-c",
                                f"cat /dev/fd/{gold.fileno()} <&- 2>/dev/null; echo ran"],
                               cwd=place.path, capture_output=True, text=True,
                               pass_fds=(gold.fileno(),))
    assert outer.returncode == 0 and "GOLD" not in outer.stdout, outer
    assert inner.returncode == 0 and inner.stdout.split() == ["ran"], inner


@LINUX
def test_the_linux_runner_passes_no_shell_code_through_its_arguments(tmp_path):
    """Nothing the runner hands systemd-run holds a '$' (its scripts are files), and an
    argument that would is refused rather than passed on."""
    _run, place = _layout(tmp_path)
    assert not any("$" in argument for argument in place.runner)
    with pytest.raises(seal.SealError, match="no '\\$'"):
        seal.linux_arguments("/w/$HOME", seal.LINUX_READS, seal.LINUX_LIMITS)


def test_a_sealed_step_is_judged_on_its_own_route_as_its_dispatch_takes_it(tmp_path):
    """A tool step whose role route (or its own route) names a model no DSH provider lists
    is refused at the attempt, with the reason its dispatch would give (`sealed_step`)."""
    _sealable()
    from refactory_dsh.loop import Loop, LoopConfig
    from refactory_dsh.optim import Candidate, Proposal
    from test_pipeline_view import _mode

    mode = _mode("livemath/skill_graph")
    seed = Candidate.from_state(mode.state_dict())
    client = _claude_code_client(tmp_path, dsh_cli="/bin/true",
                                 role_routes={"solver": {"model": "claude-sonnet-4-6"}})
    loop = Loop("livemath/skill_graph", LoopConfig(runs_root=tmp_path / "runs"),
                project=Path("."), model=client)
    assert loop._tools_offered()  # the run's own model has a DSH provider
    edit = {"op": "insert", "after": "compare", "kind": "agent", "text": "Check it.",
            "tools": ["bash"], "to": "decide_task"}
    refused = loop.optimizer.attempt(Proposal(edits=[edit]), seed, mode.invariants(), True, (),
                                     protected=mode.edit_protected())
    assert isinstance(refused, list) and refused[0].startswith("E-new-node-tools"), refused
    assert "'claude-sonnet-4-6'" in refused[0], refused
    node = {"kind": "agent", "role": "solver", "tools": ["bash"], "workspace": {"kind": "scratch"}}
    ok, why = workspace.sealed_step(client, "probe", node)
    with pytest.raises(workspace.WorkspaceError) as raised:
        workspace.for_sealed_step(client.for_node("probe", role="solver"), "probe", True)
    assert not ok and why == str(raised.value)
    # Its own route to the run's model is taken.
    own = {**edit, "route": {"model": "claude-opus-4-6"}}
    taken = loop.optimizer.attempt(Proposal(edits=[own]), seed, mode.invariants(), True, (),
                                   protected=mode.edit_protected())
    assert isinstance(taken, Candidate), taken


def test_the_sealed_route_takes_the_exact_model_and_an_effort_dsh_offers(tmp_path):
    long = _claude_code_client(tmp_path, dsh_cli="/bin/true", model="claude-opus-4-6[1m]")
    with pytest.raises(ValueError, match="no DSH provider"):
        long.for_sealed()
    offered = {"vertex": {"models": [{"id": "claude-opus-4-6",
                                      "reasoningEfforts": {"low": "low", "medium": "medium"}}]}}
    assert _claude_code_client(tmp_path, dsh_cli="/bin/true",
                               provider_config=offered).for_sealed().route()["effort"] == "medium"
    for effort, config in (("max", offered),
                           ("medium", {"vertex": {"models": [{"id": "claude-opus-4-6",
                                                              "reasoningEfforts": False}]}})):
        client = _claude_code_client(tmp_path, dsh_cli="/bin/true", effort=effort,
                                     provider_config=config)
        with pytest.raises(ValueError, match=f"effort {effort!r}"):
            client.for_sealed()


def test_a_run_offered_no_tool_step_says_why_once_in_its_journal_and_start_line(tmp_path):
    """Without a DSH provider for the arm's model, the loss of tool steps is visible: the
    journal records the reason once and the start summary prints it."""
    _sealable()
    from refactory_dsh import events
    from refactory_dsh.loop import Loop, LoopConfig

    client = _claude_code_client(tmp_path, dsh_cli="/bin/true", provider_config={})
    loop = Loop("livemath/skill_graph", LoopConfig(runs_root=tmp_path / "runs", epochs=1,
                                                    batch_size=1, optimizer="random"),
                project=Path("."), model=client)
    reason = loop._tools_not_offered()
    assert "no DSH provider" in reason, reason
    loop.init()
    journal = [json.loads(line) for line in (loop.root / "journal.jsonl").read_text().splitlines()]
    noted = [r for r in journal if r.get("kind", r.get("event")) == "tools_not_offered"
             or r.get("type") == "tools_not_offered"]
    assert len(noted) == 1 and reason in json.dumps(noted[0]), journal[:3]
    progress = [json.loads(line) for line in (loop.root / "progress.jsonl").read_text().splitlines()]
    start = next(r for r in progress if r.get("step") == "init")
    assert f"no tool steps: {reason}" in events.render(start)
    # A run that permits only knob edits loses nothing, and says nothing.
    knobs = Loop("livemath/skill_graph", LoopConfig(runs_root=tmp_path / "knobs",
                                                     allowed_ops=("set_knob",)),
                 project=Path("."), model=client)
    assert knobs._tools_not_offered() == ""


@LINUX
def test_a_seal_started_without_sbin_on_path_still_runs(tmp_path, monkeypatch):
    """A run whose PATH lacks /usr/sbin and /sbin (a system service's can): the seal still
    finds pivot_root (and unshare), and the command runs sealed."""
    _sealable()
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    _run, place = _layout(tmp_path)
    assert any(arg.endswith("/pivot_root") for arg in place.runner), place.runner
    done = _sealed(place, "cat input.txt")
    assert done.returncode == 0 and done.stdout == "THE-INPUT", done
