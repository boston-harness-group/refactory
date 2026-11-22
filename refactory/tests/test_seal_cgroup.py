"""A sealed step's whole-step cap through a delegated cgroup (`seal_cgroup`).

Where no systemd user scope lasts as long as the run (a system service), a run launched
with ``-p Delegate=yes`` caps each sealed step through a cgroup of its own, made in the
run's delegated cgroup, which the step's runner joins before the seal starts.

CI has no delegated cgroup, so most tests stand in for cgroupfs with :class:`FakeCgroups`:
a directory tree that keeps the kernel's rules the seal depends on (no controller turned
on while a process sits in the cgroup, no process moved into a cgroup whose controllers
are on for its children, no cgroup removed while a live process is in it). The runner
itself is real: it writes its pid into the fake step cgroup's ``cgroup.procs``. One live
test runs where this process's cgroup really is delegated, and skips with the reason
elsewhere.

Each test names the runtimes it covers in its docstring.
"""

from __future__ import annotations

import errno
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from refactory_dsh import lowering, seal, seal_cgroup, seal_reaper, workers, workspace
from refactory_dsh.model import DshModelClient
from refactory_dsh.nn import Graph, Node
from refactory_dsh.nn.executor import Executor, RunContext

LINUX = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="cgroup v2 is Linux's")
UNIT = "system.slice/run.service"


class FakeCgroups:
    """A cgroup v2 tree under ``tmp_path``: the service ``UNIT``, delegated to this user
    (writable, under a ``system.slice`` it cannot write), holding ``procs``. Every write,
    mkdir and rmdir the seal makes is logged in :attr:`ops`, relative to the root."""

    def __init__(self, tmp_path: Path, *, procs: tuple[str, ...] | None = None,
                 controllers: str = "cpu io memory pids", swap: bool = True,
                 delegated: bool = True, parent_writable: bool = False, delay: float = 0.0):
        self.root = tmp_path / "cgroup"
        self.swap, self.delay = swap, delay
        self.ops: list[tuple[str, str, str]] = []
        self.unmovable: set[str] = set()
        self.forking: set[str] = set()
        self.root.mkdir()
        (self.root / "cgroup.controllers").write_text("cpuset cpu io memory pids\n")
        (self.root / "cgroup.subtree_control").write_text(controllers + "\n")
        self._make(self.root / "system.slice")
        (self.root / "system.slice" / "cgroup.subtree_control").write_text(controllers + "\n")
        self.base = self.root / UNIT
        self._make(self.base)
        (self.base / "cgroup.procs").write_text(
            "\n".join(procs if procs is not None else (str(os.getpid()),)) + "\n")
        if not delegated:
            for entry in self.base.iterdir():
                entry.chmod(0o444)
            self.base.chmod(0o555)
        if not parent_writable:
            (self.root / "system.slice").chmod(0o555)

    def install(self, monkeypatch, groups: str | None = None) -> FakeCgroups:
        monkeypatch.setattr(seal_cgroup, "ROOT", str(self.root))
        monkeypatch.setattr(seal_cgroup, "_proc_cgroup", lambda: groups or f"0::/{UNIT}\n")
        for name in ("_read", "_write", "_mkdir", "_rmdir"):
            monkeypatch.setattr(seal_cgroup, name, getattr(self, name))
        # No user scope here, as on a system service: the delegated cgroup is next.
        monkeypatch.setattr(seal, "linux_scope", lambda: ())
        return self

    def _rel(self, path: Path) -> str:
        return str(Path(path).relative_to(self.root))

    def _make(self, path: Path) -> None:
        os.mkdir(path)
        enabled = (path.parent / "cgroup.subtree_control").read_text().split()
        files = {"cgroup.procs": "", "cgroup.subtree_control": "", "cgroup.kill": "",
                 "cgroup.controllers": " ".join(enabled) + "\n"}
        if "memory" in enabled:
            files.update({"memory.max": "max\n", "memory.events": "oom_kill 0\n"})
            if self.swap:
                files["memory.swap.max"] = "max\n"
        if "pids" in enabled:
            files["pids.max"] = "max\n"
        for name, text in files.items():
            (path / name).write_text(text)

    def procs(self, cgroup: Path) -> list[str]:
        """The live processes listed in ``cgroup`` (a runner that wrote itself in and has
        since exited is gone, as the kernel would have it)."""
        return [pid for pid in (cgroup / "cgroup.procs").read_text().split()
                if seal_reaper._alive(int(pid))]  # noqa: SLF001

    def _enabled(self, cgroup: Path) -> bool:
        return "memory" in (cgroup / "cgroup.subtree_control").read_text().split()

    def _read(self, path: Path) -> str:
        path = Path(path)
        if path.name == "cgroup.events":
            return f"populated {int(bool(self.procs(path.parent)))}\n"
        return path.read_text()

    def _write(self, path: Path, text: str) -> None:
        path = Path(path)
        self.ops.append(("write", self._rel(path), text))
        if not path.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))
        if not os.access(path, os.W_OK):
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(path))
        cgroup = path.parent
        if path.name == "cgroup.procs":
            time.sleep(self.delay)
            pid = text.strip()
            if self._enabled(cgroup):
                raise OSError(errno.EBUSY, os.strerror(errno.EBUSY), str(path))
            if pid in self.unmovable:
                raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(path))
            for listing in self.root.rglob("cgroup.procs"):
                kept = [p for p in listing.read_text().split() if p != pid]
                listing.write_text("\n".join(kept) + ("\n" if kept else ""))
            path.write_text(path.read_text() + pid + "\n")
            if pid in self.forking:
                # A process that forks as it moves: its child is left where it was (the
                # stand-in child is the process itself, so it stays live).
                listing = self.base / "cgroup.procs"
                listing.write_text(listing.read_text() + pid + "\n")
        elif path.name == "cgroup.subtree_control":
            if "+memory" in text.split() and self.procs(cgroup):
                raise OSError(errno.EBUSY, os.strerror(errno.EBUSY), str(path))
            enabled = set(path.read_text().split())
            enabled.update(word[1:] for word in text.split() if word.startswith("+"))
            path.write_text(" ".join(sorted(enabled)) + "\n")
        elif path.name == "cgroup.kill":
            for pid in self.procs(cgroup):
                os.kill(int(pid), signal.SIGKILL)
        else:
            path.write_text(text + "\n")

    def _mkdir(self, path: Path) -> None:
        self.ops.append(("mkdir", self._rel(path), ""))
        self._make(Path(path))

    def _rmdir(self, path: Path) -> None:
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))
        listed = path.joinpath("cgroup.procs").read_text().split()
        if self.procs(path) or any(entry.is_dir() for entry in path.iterdir()):
            self.ops.append(("busy", self._rel(path), " ".join(listed)))
            raise OSError(errno.EBUSY, os.strerror(errno.EBUSY), str(path))
        self.ops.append(("rmdir", self._rel(path), " ".join(listed)))
        shutil.rmtree(path)

    def steps(self) -> list[str]:
        return sorted(entry.name for entry in self.base.iterdir()
                      if entry.name.startswith(seal_cgroup.PREFIX))


def _limits() -> list[tuple[str, str]]:
    return [(name, value) for name, value in seal_reaper.cgroup_limits()]


# ── setting the delegated cgroup up ───────────────────────────────────────────

def test_the_service_moves_into_a_leaf_then_the_controllers_turn_on_then_the_step(tmp_path,
                                                                                monkeypatch):
    """[`seal_cgroup`, shared by both runtimes] In this order: the leaf, every process of
    the unit moved into it, the memory and pids controllers turned on, the step's cgroup
    beside the leaf, and its limits -- the scope's values, from `seal_reaper`."""
    fake = FakeCgroups(tmp_path, procs=(str(os.getpid()), str(os.getppid()))).install(monkeypatch)
    ws = tmp_path / "ws"
    ws.mkdir()
    path, held = seal_cgroup.make(ws)
    step = f"{UNIT}/{seal_cgroup.step_name(ws)}"
    assert path == str(fake.root / step)
    assert fake.ops == [
        ("mkdir", f"{UNIT}/main", ""),
        ("write", f"{UNIT}/main/cgroup.procs", str(os.getpid())),
        ("write", f"{UNIT}/main/cgroup.procs", str(os.getppid())),
        ("write", f"{UNIT}/cgroup.subtree_control", "+memory +pids"),
        ("mkdir", step, ""),
        *(("write", f"{step}/{name}", value) for name, value in _limits())]
    assert _limits() == [("memory.max", str(8 << 30)), ("memory.swap.max", "0"),
                         ("pids.max", "256")]
    assert seal.SCOPE_LIMITS == ("MemoryMax=8589934592", "MemorySwapMax=0", "TasksMax=256")
    assert held == f"memory.max={8 << 30}, memory.swap.max=0, pids.max=256"
    assert fake.procs(fake.base) == [] and len(fake.procs(fake.base / "main")) == 2


def test_setting_up_twice_or_from_a_process_already_in_the_leaf_changes_nothing(tmp_path,
                                                                              monkeypatch):
    """[`seal_cgroup`, both runtimes] A second step, and a later worker of the same unit
    (started in the leaf, so its cgroup is ``<unit>/main``), find the controllers on and
    only make their own step cgroups."""
    fake = FakeCgroups(tmp_path).install(monkeypatch)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    seal_cgroup.make(tmp_path / "a")
    first = len(fake.ops)
    monkeypatch.setattr(seal_cgroup, "_proc_cgroup", lambda: f"0::/{UNIT}/main\n")
    seal_cgroup.make(tmp_path / "b")
    later = fake.ops[first:]
    assert [op[:2] for op in later] == [("mkdir", f"{UNIT}/{seal_cgroup.step_name(tmp_path / 'b')}"),
                                        *(("write", f"{UNIT}/{seal_cgroup.step_name(tmp_path / 'b')}"
                                           f"/{name}") for name, _ in _limits())]
    assert fake.steps() == sorted([seal_cgroup.step_name(tmp_path / "a"),
                                   seal_cgroup.step_name(tmp_path / "b")])


def test_workers_sharing_the_unit_set_it_up_once_under_the_lock(tmp_path, monkeypatch):
    """[`seal_cgroup`, both runtimes: a concurrent data pass, or several workers of one
    unit] Steps prepared at once set the unit up exactly once: the moves and the
    controllers' turning on are never interleaved, and each step gets its cgroup."""
    pids = tuple(str(pid) for pid in (os.getpid(), os.getppid()))
    fake = FakeCgroups(tmp_path, procs=pids, delay=0.05).install(monkeypatch)
    spaces = [tmp_path / f"ws{index}" for index in range(6)]
    errors: list[BaseException] = []

    def make(ws: Path) -> None:
        ws.mkdir()
        try:
            seal_cgroup.make(ws)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=make, args=(ws,)) for ws in spaces]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    setup = [op for op in fake.ops if op[1].startswith(f"{UNIT}/main")
             or op[1] == f"{UNIT}/cgroup.subtree_control"]
    assert setup == [("mkdir", f"{UNIT}/main", ""),
                     *(("write", f"{UNIT}/main/cgroup.procs", pid) for pid in pids),
                     ("write", f"{UNIT}/cgroup.subtree_control", "+memory +pids")], setup
    assert fake.steps() == sorted(seal_cgroup.step_name(ws) for ws in spaces)


@pytest.mark.parametrize("case,why", [
    ("cgroup v1", "this process is in no cgroup v2 hierarchy"),
    ("not delegated", "is not delegated to it (launch the run with -p Delegate=yes)"),
    ("a tree this user owns", "is not delegated to it"),
    ("no memory controller", "has no memory controller"),
    ("a process that cannot move", "cannot move process"),
    ("a process forking as it moves", "cannot turn on the memory and pids controllers"),
    ("no step cgroup", "cannot make"),
])
def test_where_no_delegated_cgroup_can_cap_a_step_it_says_why(tmp_path, monkeypatch, case, why):
    """[`seal_cgroup`, both runtimes] Each way the cap cannot be made raises
    `Unavailable` naming why (EBUSY, EACCES, no controller, no delegation), leaving
    nothing half made behind."""
    fake = FakeCgroups(tmp_path, delegated=case != "not delegated",
                       parent_writable=case == "a tree this user owns",
                       controllers="cpu io pids" if case == "no memory controller"
                       else "cpu io memory pids")
    groups = "1:name=systemd:/system.slice/run.service\n" if case == "cgroup v1" else None
    fake.install(monkeypatch, groups=groups)
    if case == "a process that cannot move":
        fake.unmovable.add(str(os.getpid()))
    elif case == "a process forking as it moves":
        fake.forking.add(str(os.getpid()))
    elif case == "no step cgroup":
        real = fake._mkdir

        def refuse_steps(path):
            if Path(path).name.startswith(seal_cgroup.PREFIX):
                raise PermissionError(errno.EACCES, "Permission denied", str(path))
            real(path)

        monkeypatch.setattr(seal_cgroup, "_mkdir", refuse_steps)
    with pytest.raises(seal_cgroup.Unavailable, match=why.replace("(", r"\(").replace(")", r"\)")):
        seal_cgroup.make(tmp_path)
    assert fake.steps() == []
    if case == "a process forking as it moves":
        assert "Device or resource busy" in str(_unavailable(tmp_path))


def _unavailable(ws: Path) -> seal_cgroup.Unavailable:
    try:
        seal_cgroup.make(ws)
    except seal_cgroup.Unavailable as exc:
        return exc
    raise AssertionError("made")


def test_a_kernel_without_a_swap_account_caps_the_rest_and_says_so(tmp_path, monkeypatch):
    """[`seal_cgroup`] No ``memory.swap.max``: memory and processes are still capped, and
    the record says swap is not."""
    FakeCgroups(tmp_path, swap=False).install(monkeypatch)
    _path, held = seal_cgroup.make(tmp_path)
    assert held == f"memory.max={8 << 30}, memory.swap.max not kept by this kernel, pids.max=256"


def test_a_user_scope_comes_first_and_no_cgroup_is_made(tmp_path, monkeypatch):
    """[`seal.step_cap`, both runtimes] Where the user's manager lasts and makes a scope,
    the step uses it, and the delegated cgroup is left untouched."""
    fake = FakeCgroups(tmp_path).install(monkeypatch)
    scope = ("/usr/bin/systemd-run", "--user", "--scope", "--")
    monkeypatch.setattr(seal, "linux_scope", lambda: scope)
    if not sys.platform.startswith("linux"):
        pytest.skip("macOS has no scope or cgroup")
    cap = seal.step_cap(tmp_path)
    assert cap.capped and cap.scope == scope and cap.cgroup == ""
    assert cap.how.startswith("a systemd user scope (")
    assert fake.ops == []
    monkeypatch.setattr(seal, "linux_scope", lambda: ())
    cap = seal.step_cap(tmp_path)
    assert cap.capped and cap.cgroup and "a delegated cgroup" in cap.how


# ── a sealed step under it ────────────────────────────────────────────────────

def _sealable() -> None:
    from test_sealed_tools import _sealable as sealable

    sealable()


def _prepare(run: Path, node: str = "look") -> workspace.Workspace:
    place = workspace.prepare(run, node, "scratch", route={"worker_backend": "dsh"},
                              tools=("bash",), reads=("input.txt",))
    assert place is not None
    return place


def _run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / "input.txt").write_text("THE-INPUT")
    return run


@LINUX
@pytest.mark.parametrize("case", ["not delegated", "a process forking as it moves"])
def test_a_step_without_the_cap_runs_with_per_process_limits_and_its_record_says_why(
        tmp_path, monkeypatch, case):
    """[the dispatch record both runtimes write, `workspace.prepare`] Where the delegated
    cgroup cannot be made, the step still runs: unscoped, with the tight per-process
    limits, its record saying it had no whole-step cap and why."""
    _sealable()
    fake = FakeCgroups(tmp_path, delegated=case == "a process forking as it moves")
    fake.install(monkeypatch)
    if case != "not delegated":
        fake.forking.add(str(os.getpid()))
    run = _run_dir(tmp_path)
    place = _prepare(run)
    record = json.loads((place.root / "dispatch.json").read_text())
    assert record["whole_step_cap"] is False and "cgroup" not in record, record
    assert "no delegated cgroup either" in record["cap"]
    assert ("Delegate=yes" if case == "not delegated" else "Device or resource busy") in record["cap"]
    assert record["cap"].endswith(f"per-process limits only ({seal.LINUX_LIMITS_UNSCOPED})")
    assert seal_reaper.CGROUP_FLAG not in place.runner
    assert f"{seal.LIMITS} {seal.LINUX_LIMITS_UNSCOPED}" in place.runner
    done = subprocess.run([*place.runner, "--", "bash", "-c", "cat input.txt; echo; ulimit -u"],
                          cwd=place.path, capture_output=True, text=True, timeout=60)
    assert done.stdout.split() == ["THE-INPUT", "128"], done
    workspace.collect(place, run, [])


#: A stand-in for DSH that runs one sealed command through the step's runner, and writes
#: what it saw as the step's answer: the command's output, and the step's dispatch record
#: (beside its workspace, where DSH -- never the sealed command -- can read it). Exits 1
#: after the command when told to fail.
STEP = """\
import json, pathlib, subprocess, sys
cwd = pathlib.Path.cwd()
last = max(i for i, arg in enumerate(sys.argv) if arg == '--patch')
patch = json.loads(pathlib.Path(sys.argv[last + 1]).read_text())
runner = next(row['config']['runnerCommand'] for row in patch if row.get('id') == 'sandbox')
done = subprocess.run([*runner, '--ro-bind', '/', '/', '--', 'bash', '-c',
                       'cat input.txt; echo; ulimit -u'], cwd=cwd, capture_output=True, text=True,
                      timeout=60)
record = json.loads((cwd.parent / 'dispatch.json').read_text())
(cwd / 'answer.txt').write_text(json.dumps({'out': done.stdout.split(), 'record': record}))
sys.exit({fail})
"""


def _step_client(tmp_path: Path, fail: bool = False) -> DshModelClient:
    script = tmp_path / "agent-cli"
    script.write_text(f"#!{sys.executable}\n" + STEP.replace("{fail}", str(int(fail))))
    script.chmod(0o755)
    return DshModelClient(worker_backend="dsh", dsh_cli=str(script), attempts=1)


def _dispatch(runtime: str, run: Path, client: DshModelClient) -> None:
    graph = Graph()
    graph.add(Node("look", "agent", prompt_template="Look.", reads=("input.txt",),
                   writes=("answer.txt",), tools=("bash",), workspace="scratch"))
    if runtime == "walker":
        ctx = RunContext(run, knobs={}, facts={}, task={})
        ctx.model = client
        Executor(graph, ctx).run(start=graph.start)
        return
    from refactory_dsh import worker_node

    state = {"graph": graph.to_state(), "knobs": {}}
    lowered = workers.lower_workflow(lowering.to_workflow(state), run, client.context(),
                                     lowering.node_declarations(state))
    spec = json.loads(Path(shlex.split(lowered.nodes["look"].dispatch_command)[-1]).read_text())
    worker_node.execute(spec, run)


@LINUX
@pytest.mark.parametrize("runtime", ["walker", "spine"])
def test_a_sealed_step_runs_in_its_delegated_cgroup_and_leaves_none_behind(tmp_path, monkeypatch,
                                                                          runtime):
    """[walker: the executor's dispatch; spine: its agent dispatch, `worker_node`] With
    no user scope and a delegated cgroup, the step's record says "a delegated cgroup" with
    its limits and ``whole_step_cap`` true; its runner joined the step's cgroup (its pid
    was there) with the loose per-process bound; and once the step is collected, after
    its PID namespace is reaped, the cgroup is gone."""
    _sealable()
    fake = FakeCgroups(tmp_path).install(monkeypatch)
    order: list[str] = []
    real_reap = seal_reaper.reap
    monkeypatch.setattr(seal_reaper, "reap", lambda ws, *a: order.append("reap") or real_reap(ws, *a))
    real_rmdir = fake._rmdir
    monkeypatch.setattr(seal_cgroup, "_rmdir", lambda path: order.append("rmdir") or real_rmdir(path))
    run = _run_dir(tmp_path)
    _dispatch(runtime, run, _step_client(tmp_path))
    answer = json.loads((run / "answer.txt").read_text())
    assert answer["out"] == ["THE-INPUT", "512"], answer
    record = answer["record"]
    assert record["whole_step_cap"] is True, record
    assert record["cap"].startswith(f"a delegated cgroup (memory.max={8 << 30}, "
                                    "memory.swap.max=0, pids.max=256) at "), record
    assert record["cgroup"] == str(fake.base / seal_cgroup.step_name(record["path"]))
    removed = [op for op in fake.ops if op[0] == "rmdir"]
    assert [op[1] for op in removed] == [str(Path(record["cgroup"]).relative_to(fake.root))]
    assert removed[0][2], "the runner moved itself into the step's cgroup"
    assert order.index("rmdir") > order.index("reap"), order
    assert fake.steps() == [], "no cgroup left behind on a normal finish"


@LINUX
@pytest.mark.parametrize("runtime", ["walker", "spine"])
def test_a_step_that_fails_leaves_no_cgroup_behind(tmp_path, monkeypatch, runtime):
    """[walker, spine] A dispatch that fails after its command ran (nothing collected)
    still has its processes ended and its cgroup removed when the dispatch ends."""
    _sealable()
    fake = FakeCgroups(tmp_path).install(monkeypatch)
    run = _run_dir(tmp_path)
    try:
        _dispatch(runtime, run, _step_client(tmp_path, fail=True))
    except Exception:  # noqa: BLE001 - the spine raises it; the walker records it
        pass
    assert not (run / "answer.txt").exists(), "the step failed before it was collected"
    assert [op[1] for op in fake.ops if op[0] == "write" and op[1].endswith("/memory.max")]
    assert [op[0] for op in fake.ops].count("rmdir") == 1, fake.ops
    assert fake.steps() == []


@LINUX
def test_a_dead_runner_s_cgroup_goes_at_the_next_prepare_and_at_the_run_s_end(tmp_path, monkeypatch):
    """[`workspace.prepare` and the run-end sweep, both runtimes] A step whose runner died
    with its sealed command still running, and was never collected: the next dispatch of
    the node ends that command and removes its cgroup; so does the run's end."""
    _sealable()
    fake = FakeCgroups(tmp_path).install(monkeypatch)
    run = _run_dir(tmp_path)
    first = _prepare(run)
    left = subprocess.Popen([*first.runner, "--", "bash", "-c", "sleep 60"], cwd=first.path)
    try:
        for _ in range(500):
            if fake.procs(Path(first.cgroup)):
                break
            time.sleep(0.01)
        assert fake.procs(Path(first.cgroup)) == [str(left.pid)]
        second = _prepare(run)
        assert second.path != first.path and fake.steps() == [Path(second.cgroup).name]
        assert left.wait(timeout=10) is not None, "the dead dispatch's command was ended"
    finally:
        left.kill()
        left.wait()
    assert ("busy", str(Path(first.cgroup).relative_to(fake.root)), str(left.pid)) not in fake.ops
    workspace.sweep(run)
    assert fake.steps() == []


@LINUX
def test_a_leftover_still_busy_is_ended_through_cgroup_kill_only_after_waiting(tmp_path,
                                                                             monkeypatch):
    """[`seal_cgroup.remove`] A process the reap did not end (it is no runner of this
    workspace) is waited for, then ended through ``cgroup.kill``, then the cgroup goes."""
    fake = FakeCgroups(tmp_path).install(monkeypatch)
    path, _held = seal_cgroup.make(tmp_path)
    stray = subprocess.Popen(["sleep", "60"])
    try:
        fake._write(Path(path) / "cgroup.procs", str(stray.pid))
        started = time.monotonic()
        assert seal_cgroup.remove(path, wait_s=0.4) == ""
        assert time.monotonic() - started >= 0.2
        assert stray.wait(timeout=5) == -signal.SIGKILL
    finally:
        stray.kill()
        stray.wait()
    kinds = [op[0] for op in fake.ops if op[1].startswith(str(Path(path).relative_to(fake.root)))]
    assert kinds.index("busy") < kinds.index("rmdir")
    assert ("write", f"{Path(path).relative_to(fake.root)}/cgroup.kill", "1") in fake.ops
    assert seal_cgroup.remove(str(fake.base / "main")).endswith("so it was left")


@LINUX
def test_a_runner_that_cannot_join_its_cgroup_runs_the_command_uncapped_and_says_so(tmp_path,
                                                                                  monkeypatch):
    """[the runner both runtimes put in front of a sealed command] Its step's cgroup is
    gone before a command starts: the command still runs, sealed, under the tight
    per-process limits, and the step's record says it ran without the whole-step cap."""
    _sealable()
    FakeCgroups(tmp_path).install(monkeypatch)
    run = _run_dir(tmp_path)
    place = _prepare(run)
    assert place.whole_step_cap[0] is True
    shutil.rmtree(place.cgroup)
    done = subprocess.run([*place.runner, "--", "bash", "-c", "cat input.txt; echo; ulimit -u"],
                          cwd=place.path, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and done.stdout.split() == ["THE-INPUT", "128"], done
    workspace.collect(place, run, [])
    record = json.loads((place.root / "dispatch.json").read_text())
    assert record["whole_step_cap"] is False
    assert "cannot move into the step's cgroup" in record["cap"], record


# ── live: a really delegated cgroup ───────────────────────────────────────────

def _not_delegated() -> str:
    """Why this process's cgroup cannot cap a step for real, or "" where it can."""
    if not sys.platform.startswith("linux"):
        return "cgroup v2 is Linux's"
    try:
        base = seal_cgroup.delegated_base()
    except seal_cgroup.Unavailable as exc:
        return f"no delegated cgroup here: {exc}"
    offered = (base / "cgroup.controllers").read_text().split()
    if not all(name in offered for name in seal_cgroup.CONTROLLERS):
        return f"the delegated cgroup {base} lacks the memory or pids controller"
    return ""


def _oom_kills(cgroup: Path) -> int:
    for line in (cgroup / "memory.events").read_text().splitlines():
        if line.startswith("oom_kill "):
            return int(line.split()[1])
    return 0


@pytest.mark.skipif(bool(_not_delegated()), reason=_not_delegated() or "delegated")
def test_live_a_step_cannot_outgrow_its_delegated_cgroup(tmp_path, monkeypatch):
    """[live, where this process's cgroup is delegated (``-p Delegate=yes``); the runner
    and seal both runtimes use] A sealed command allocating past the step's memory.max is
    OOM-killed inside the step, and nothing outside it is: the step's cgroup counts the
    kill, this process's cgroup does not. Its address space is not what stopped it (no
    per-process memory limit under the cap). The real cap holds the scope's values, and
    the step's cgroup is gone once it is collected."""
    from test_sealed_tools import _layout, _sealed

    _sealable()
    monkeypatch.setattr(seal, "linux_scope", lambda: ())
    run, place = _layout(tmp_path)
    cap = Path(place.cgroup)
    assert place.whole_step_cap[0] is True and "a delegated cgroup" in place.whole_step_cap[1]
    assert [(cap / name).read_text().strip() for name, _ in _limits()] == [
        str(8 << 30), "0", "256"]
    workspace.collect(place, run, [])
    assert not cap.exists()

    monkeypatch.setattr(seal_reaper, "STEP_MEMORY_MAX", 256 << 20)
    shutil.rmtree(tmp_path / "run")
    shutil.rmtree(tmp_path / "bench")
    run, place = _layout(tmp_path)
    cap = Path(place.cgroup)
    assert (cap / "memory.max").read_text().strip() == str(256 << 20)
    own = Path(seal_cgroup.ROOT) / seal_cgroup.own_cgroup().lstrip("/")
    before = _oom_kills(own)
    done = _sealed(place, "ulimit -v; python3 -c \"x = b'x' * (600 << 20); print('survived')\"; "
                          "echo exit=$?")
    assert done.stdout.split() == ["unlimited", "exit=137"], done
    assert _oom_kills(cap) >= 1
    assert _oom_kills(own) == before, "nothing outside the step was OOM-killed"
    (place.path / "fork.py").write_text(
        "import os, time\nn = 0\nwhile n < 400:\n    try:\n        pid = os.fork()\n"
        "    except OSError:\n        break\n    if pid == 0:\n        time.sleep(5)\n"
        "        os._exit(0)\n    n += 1\nprint(n)\n")
    forked = _sealed(place, "python3 fork.py")
    assert 100 < int(forked.stdout.split()[-1]) < 256, forked
    workspace.collect(place, run, [])
    assert not cap.exists()


@LINUX
def test_a_planted_dispatch_record_never_releases_another_cgroup(tmp_path, monkeypatch):
    """[`workspace.prepare` and the run-end sweep, both runtimes] A dispatch record that
    names a cgroup other than its own sealed workspace's (written by anything with the run
    directory in reach) is ignored: that cgroup, a live step's, is never ended or removed."""
    _sealable()
    fake = FakeCgroups(tmp_path).install(monkeypatch)
    run = _run_dir(tmp_path)
    other = tmp_path / "other-ws"
    other.mkdir()
    victim, _held = seal_cgroup.make(other)
    place = _prepare(run)
    record = json.loads((place.root / "dispatch.json").read_text())
    record["cgroup"] = victim
    (place.root / "dispatch.json").write_text(json.dumps(record))
    reaped: list[str] = []
    monkeypatch.setattr(seal_reaper, "reap", lambda ws, *a: reaped.append(ws) or [])
    _prepare(run)
    workspace.sweep(run)
    assert Path(victim).is_dir() and str(other) not in reaped
    assert Path(victim).name in fake.steps()


@LINUX
def test_collecting_a_step_removes_its_cgroup_after_its_processes_are_reaped(tmp_path,
                                                                            monkeypatch):
    """[`workspace.collect`, both runtimes] Collection itself -- not only the end of the
    dispatch -- removes the step's cgroup, once the step's runner and PID namespace are
    killed and reaped (a command still running when the step is collected)."""
    _sealable()
    fake = FakeCgroups(tmp_path).install(monkeypatch)
    run = _run_dir(tmp_path)
    place = _prepare(run)
    cgroup = Path(place.cgroup)
    running = subprocess.Popen([*place.runner, "--", "bash", "-c", "sleep 60"], cwd=place.path)
    try:
        for _ in range(500):
            if fake.procs(cgroup):
                break
            time.sleep(0.01)
        assert fake.procs(cgroup) == [str(running.pid)]
        workspace.collect(place, run, [])
        assert not cgroup.exists()
        assert running.wait(timeout=10) == -signal.SIGKILL
    finally:
        running.kill()
        running.wait()
    rel = str(cgroup.relative_to(fake.root))
    assert [op[0] for op in fake.ops if op[1] == rel] == ["mkdir", "rmdir"], \
        "never busy: reaped first"
