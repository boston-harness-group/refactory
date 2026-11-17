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
import subprocess
import sys
from pathlib import Path

import pytest
import structure_fuzz as sf
from refactory_dsh import lowering, seal, workers, workspace
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
    state = sf.Applier(sf.seeds()["fuzz_draft_review"]).seed_state
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
    state = sf.Applier(sf.seeds()["fuzz_draft_review"]).seed_state
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
    applier = sf.Applier(sf.seeds()["fuzz_draft_review"])
    parent = sf.Candidate.from_state(applier.seed_state)
    edit = {"op": "insert", "after": "draft", "kind": "agent", "text": "Check it.", "tools": ["bash"]}
    if seal.available()[0]:
        out = applier.attempt(parent, edit, set())
        assert isinstance(out, sf.Candidate), out
    # A tools list a person writes outside a scratch workspace is still refused.
    assert base_mod._new_node_tools(
        applier.seed_state,
        {"graph": {"nodes": {"x": {"kind": "agent", "tool_policy": None, "tools": ["bash"]}}}})


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


def test_a_sealed_command_sees_its_inputs_and_nothing_else(tmp_path):
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
        "homebrew data": "ls /opt/homebrew/etc /opt/homebrew/var /usr/local/etc /usr/local/var "
                         "2>/dev/null | head -1",
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


def test_a_sealed_command_gets_no_descriptor_but_its_own(tmp_path):
    _sealable()
    _run, place = _layout(tmp_path)
    with open(tmp_path / "bench" / "gold.json", "rb") as gold:
        result = _sealed(place, f"cat /dev/fd/{gold.fileno()} <&- 2>/dev/null; true",
                         pass_fds=(gold.fileno(),))
    assert "GOLD-OUTSIDE" not in result.stdout


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
    assert observed["requests"] == 4, observed
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
answer = {'tools': allow, 'input': bash('cat input.txt'),
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
    assert answer == {"tools": ["bash"], "input": "THE-INPUT", "grader": "DENIED", "other": "DENIED"}


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
