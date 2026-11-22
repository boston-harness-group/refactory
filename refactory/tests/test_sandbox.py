"""Running code we did not write.

The sandbox issue: create-mode imports and runs model-written Python, and nothing
contained it. `check_package` imported it into the driver process, and the smoke run
executed in-process too.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
import time
from pathlib import Path

import pytest

from refactory_dsh import registry, sandbox
from refactory_dsh.paths import venv_bin
from tests_support import on_stdin

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
    assert scrubbed["PATH"] == "/usr/bin", (
        "the caller's PATH passes through unchanged; the worker environment "
        "or check environment has already set it to the target's Python"
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
        "import time; time.sleep(30)", workdir=tmp_path / "run", timeout_s=1
    )
    assert not result.ok and result.timed_out
    assert "timed out" in result.problem() or "inactivity" in result.problem()


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


def test_a_check_command_is_bounded_by_its_timeout_not_by_memory(tmp_path):
    """A 2 GB address-space cap failed every Node check that started worker threads, vitest among them (#325)."""
    import resource

    probe = ("import json, resource; print(json.dumps("
             "[resource.getrlimit(resource.RLIMIT_AS)[0], resource.getrlimit(resource.RLIMIT_CPU)[0]]))")
    result = sandbox.run_command(f"{shlex.quote(sys.executable)} -c {shlex.quote(probe)}",
                                 workdir=tmp_path / "check", timeout_s=30)
    assert result.ok, result.stderr
    address_space, cpu = json.loads(result.stdout)
    assert address_space == resource.getrlimit(resource.RLIMIT_AS)[0], "a check sets no address-space cap"
    desired = sandbox.COMMAND_BACKSTOP_S * (os.cpu_count() or 1)
    hard = resource.getrlimit(resource.RLIMIT_CPU)[1]
    if hard != resource.RLIM_INFINITY and desired > hard:
        expected_cpu = hard
    else:
        expected_cpu = desired
    assert cpu == expected_cpu, (
        "a check's CPU limit is the backstop on every core (not the inactivity limit), "
        f"capped at the system hard limit; expected {expected_cpu}, got {cpu}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="this machine has no node")
def test_a_node_check_that_starts_worker_threads_passes(tmp_path):
    """What #325 reported: under the cap, Node aborts reserving a second worker thread's memory.

    Linux enforces an address-space cap; macOS cannot set one, so there this passes either way.
    """
    script = ("let left = 4; for (let i = 0; i < 4; i++) new (require('worker_threads').Worker)"
              "('', {eval: true}).on('exit', code => { if (code) process.exit(code); "
              "if (--left === 0) process.exit(0) })")
    result = sandbox.run_command(f"node -e {shlex.quote(script)}", workdir=tmp_path / "check")
    assert result.ok, result.problem()


def _running(pid: int) -> bool:
    """Whether ``pid`` is a live process: not gone, and not a zombie waiting to be reaped."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    try:
        return not (stat.exists() and stat.read_text().rsplit(")", 1)[-1].split()[0] == "Z")
    except (ProcessLookupError, FileNotFoundError):
        return False  # reaped between exists() and read_text()


def test_a_check_that_runs_out_of_time_stops_everything_it_started(tmp_path):
    """A process a timed-out check left behind ran on: only the shell was stopped."""
    workdir = tmp_path / "check"
    result = sandbox.run_command("sleep 60 & echo $! > sleep.pid; wait", workdir=workdir, timeout_s=1)
    assert result.timed_out
    pid = int((workdir / "sleep.pid").read_text())
    deadline = time.monotonic() + 5
    while _running(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _running(pid), "the check's background process outlived its timeout"


def test_signal_killed_once_retry_succeeds(tmp_path):
    """A command killed by a signal once passes on the retry; no attempt spent."""
    marker = tmp_path / "marker"
    # First run: kill self with SIGABRT.  Second run: the marker exists, so exit 0.
    cmd = (
        f'if [ -f {shlex.quote(str(marker))} ]; then exit 0; fi; '
        f'touch {shlex.quote(str(marker))}; kill -ABRT $$'
    )
    result = sandbox.run_command(cmd, workdir=tmp_path / "w", timeout_s=30)
    assert result.ok
    assert result.returncode == 0
    assert marker.exists()


def test_signal_killed_twice_names_the_signal(tmp_path):
    """A command killed by a signal twice: infrastructure failure naming the signal."""
    cmd = 'kill -ABRT $$'
    result = sandbox.run_command(cmd, workdir=tmp_path / "w", timeout_s=30)
    assert not result.ok
    assert result.returncode < 0
    assert "killed by SIGABRT" in result.stderr


def test_ordinary_failure_not_retried(tmp_path):
    """An ordinary failing check (returncode > 0) is unchanged by the retry logic."""
    marker = tmp_path / "marker"
    cmd = f'echo run >> {shlex.quote(str(marker))}; exit 1'
    result = sandbox.run_command(cmd, workdir=tmp_path / "w", timeout_s=30)
    assert not result.ok
    assert result.returncode == 1
    # Each run appends a line; more than one means the retry logic misfired.
    assert marker.read_text().count("run") == 1


def test_timeout_does_not_retry(tmp_path):
    """A timed-out command keeps today's handling (no retry)."""
    result = sandbox.run_command("sleep 60", workdir=tmp_path / "w", timeout_s=1)
    assert result.timed_out
    assert result.returncode == 124


# ── inactivity-based timeout tests ─────────────────────────────────────────

def test_command_producing_output_survives_past_old_wall_limit(tmp_path):
    """A command printing every 0.2 s for 2 s is not stopped by inactivity=1 s."""
    result = sandbox.run_command(
        'for i in $(seq 1 10); do echo "line_$i"; sleep 0.2; done',
        workdir=tmp_path / "w", timeout_s=1,
    )
    assert result.ok, f"expected ok, got: {result.problem()}"
    assert not result.timed_out, "should not have timed out"
    assert "line_10" in result.stdout, "all lines should be present"


def test_silent_command_stopped_at_inactivity_limit(tmp_path):
    """sleep 10 with inactivity=1 s is stopped after ~1 s, not at the backstop."""
    start = time.monotonic()
    result = sandbox.run_command("sleep 10", workdir=tmp_path / "w", timeout_s=1)
    elapsed = time.monotonic() - start
    assert result.timed_out
    assert result.timeout_kind == "inactivity"
    assert elapsed < 5, f"should be stopped in ~1 s, took {elapsed:.1f} s"


def test_timeout_output_is_kept_in_result(tmp_path):
    """After an inactivity timeout, stdout contains the lines printed before silence."""
    result = sandbox.run_command(
        'echo kept_line_1; echo kept_line_2; sleep 10',
        workdir=tmp_path / "w", timeout_s=1,
    )
    assert result.timed_out
    assert "kept_line_1" in result.stdout, "output must be kept on timeout"
    assert "kept_line_2" in result.stdout, "output must be kept on timeout"


def test_timeout_kind_set_to_inactivity(tmp_path):
    """A silent command's timeout_kind is 'inactivity', not None or 'backstop'."""
    result = sandbox.run_command("sleep 10", workdir=tmp_path / "w", timeout_s=1)
    assert result.timed_out
    assert result.timeout_kind == "inactivity"


def test_backstop_kills_despite_continuous_output(tmp_path, monkeypatch):
    """A command printing every 0.2 s is stopped at the backstop (3 s)."""
    monkeypatch.setattr(sandbox, "COMMAND_BACKSTOP_S", 3)
    start = time.monotonic()
    result = sandbox.run_command(
        'while true; do echo tick; sleep 0.2; done',
        workdir=tmp_path / "w", timeout_s=1,
    )
    elapsed = time.monotonic() - start
    assert result.timed_out
    assert result.timeout_kind == "backstop"
    assert 2.5 < elapsed < 8, f"backstop should fire at ~3 s, took {elapsed:.1f} s"


def test_rlimit_cpu_derives_from_backstop(tmp_path):
    """RLIMIT_CPU is based on COMMAND_BACKSTOP_S, not the caller's timeout_s."""
    probe = ("import json, resource; print(json.dumps("
             "resource.getrlimit(resource.RLIMIT_CPU)[0]))")
    result = sandbox.run_command(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(probe)}",
        workdir=tmp_path / "w", timeout_s=10,
    )
    assert result.ok, result.stderr
    cpu = json.loads(result.stdout)
    import resource as _res
    desired = sandbox.COMMAND_BACKSTOP_S * (os.cpu_count() or 1)
    hard = _res.getrlimit(_res.RLIMIT_CPU)[1]
    if hard != _res.RLIM_INFINITY and desired > hard:
        expected = hard
    else:
        expected = desired
    assert cpu == expected, (
        f"RLIMIT_CPU should be backstop * cpu_count = {desired}, "
        f"capped at hard limit {hard}; got {cpu}"
    )
    # Key assertion: RLIMIT_CPU is NOT the inactivity timeout_s (10) times cpu_count —
    # that would be far smaller than the backstop-based value.
    assert cpu != 10 * (os.cpu_count() or 1), (
        "RLIMIT_CPU must be based on the backstop, not the caller's timeout_s"
    )


def test_problem_includes_timeout_kind_and_output_tail(tmp_path):
    """problem() on a timed-out command says 'inactivity' and includes output."""
    result = sandbox.run_command(
        'echo visible_line; sleep 10',
        workdir=tmp_path / "w", timeout_s=1,
    )
    assert result.timed_out
    problem = result.problem()
    assert "inactivity" in problem
    assert "visible_line" in problem


def test_problem_keeps_both_streams_tails(tmp_path):
    """What a stopped command printed last on stderr is kept beside its stdout."""
    result = sandbox.run_command('echo out_line; echo err_line >&2; sleep 10',
                                 workdir=tmp_path / "w", timeout_s=1)
    assert result.timed_out
    problem = result.problem()
    assert "[stdout] out_line" in problem and "[stderr] err_line" in problem
    report = sandbox.timeout_output(result)
    assert "[stdout] out_line" in report and "[stderr] err_line" in report


_MEMORY_PROBE = """
import json, pathlib, resource, sys
from refactory_dsh import sandbox
scale = 1 if sys.platform == "darwin" else 1024
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
result = sandbox.run_command(sys.argv[2], workdir=pathlib.Path(sys.argv[1]), timeout_s=60)
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
print(json.dumps({"growth": after - before, "ok": result.ok, "bytes": result.output_bytes,
                  "kept": len(result.stdout), "end": result.stdout[-12:]}))
"""


@pytest.mark.parametrize("writer", [
    "sys.stdout.write('y\\n' * 25_000_000)",       # 50 MB of short lines
    "sys.stdout.write('x' * 50_000_000)",           # 50 MB with no newline at all
])
def test_a_large_output_is_kept_within_bounded_memory(tmp_path, writer):
    """50 MB of output keeps the driver's memory growth modest: the output is read in
    chunks into a bounded buffer (head and tail), never as one list of lines or one
    re-copied buffer. Measured in a fresh interpreter, so the peak is this run's own."""
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote('import sys; ' + writer)}"
    env = {**os.environ, "PYTHONPATH": str(Path(sandbox.__file__).resolve().parent.parent)}
    import subprocess
    probe = subprocess.run([sys.executable, "-c", _MEMORY_PROBE, str(tmp_path / "w"), command],
                           capture_output=True, text=True, env=env, timeout=60,
                           check=False)
    assert probe.returncode == 0, probe.stderr
    measured = json.loads(probe.stdout.strip().splitlines()[-1])
    assert measured["ok"] and measured["bytes"] == 50_000_000
    assert measured["kept"] <= sandbox.OUTPUT_KEEP_BYTES + 100, measured
    assert measured["end"].strip("\n")[-1] in "xy", measured
    assert measured["growth"] < 96 * 1024 * 1024, f"driver grew {measured['growth'] / 2**20:.0f} MiB"


def test_output_beyond_the_cap_keeps_its_head_and_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "OUTPUT_KEEP_BYTES", 4000)
    monkeypatch.setattr(sandbox, "OUTPUT_TAIL_BYTES", 1000)
    result = sandbox.run_command(
        'echo FIRST; i=0; while [ $i -lt 2000 ]; do echo "line $i"; i=$((i+1)); done; echo LAST',
        workdir=tmp_path / "w", timeout_s=10)
    assert result.ok
    assert result.stdout.startswith("FIRST\n") and result.stdout.endswith("LAST\n")
    assert "bytes of output not kept" in result.stdout
    assert len(result.stdout) < 4200


def test_a_never_ending_printer_is_stopped_at_the_output_limit(tmp_path, monkeypatch):
    """A logging loop never goes silent; it is stopped at the output limit, not the backstop."""
    monkeypatch.setattr(sandbox, "OUTPUT_LIMIT_BYTES", 2 * 1024 * 1024)
    start = time.monotonic()
    result = sandbox.run_command("while true; do echo spam-spam-spam; done",
                                 workdir=tmp_path / "w", timeout_s=30)
    assert time.monotonic() - start < 5
    assert result.timed_out and result.timeout_kind == "output_limit"
    assert result.output_bytes > 2 * 1024 * 1024
    assert result.stdout.rstrip().endswith("spam-spam-spam")
    assert "output limit" in result.problem() and "[stdout] spam-spam-spam" in result.problem()


def test_carriage_returns_read_as_newlines(tmp_path):
    """As the old ``text=True`` path did (universal newlines)."""
    result = sandbox.run_command(r"printf 'a\r\nb\rc\n'; printf 'e\r\n' >&2",
                                 workdir=tmp_path / "w", timeout_s=5)
    assert result.ok
    assert result.stdout == "a\nb\nc\n" and result.stderr == "e\n"


def test_what_a_command_leaves_running_does_not_survive_it(tmp_path, monkeypatch):
    """A background child that still holds the output is stopped after a short grace
    once the command's own process exits, and the reader threads are done."""
    import threading

    monkeypatch.setattr(sandbox, "EXIT_GRACE_S", 0.5)
    pidfile = tmp_path / "bg.pid"
    start = time.monotonic()
    result = sandbox.run_command(
        f"sh -c 'echo $$ > {pidfile}; exec sleep 20' & echo early",
        workdir=tmp_path / "w", timeout_s=30)
    assert time.monotonic() - start < 4
    assert result.ok and result.stdout == "early\n"
    pid = int(pidfile.read_text())
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(pid, 9)
        pytest.fail("the background sleep outlived its command")
    assert not [t for t in threading.enumerate() if t.name == "sandbox-drain" and t.is_alive()]


def test_refactory_env_writes_python_unbuffered():
    """A check outside any worktree, a fast run's check and the probe use this environment."""
    assert sandbox.refactory_env()["PYTHONUNBUFFERED"] == "1"


def test_a_check_reads_no_input(tmp_path):
    """A check runs unattended, so nothing it runs can read or wait on the launcher's stdin."""
    with on_stdin("y\n"):
        result = sandbox.run_command('read answer; echo "read: $answer"', workdir=tmp_path / "check")
    assert result.ok, result.problem()
    assert result.stdout.strip() == "read:", "a check's stdin is empty, so it cannot wait on a terminal"


@pytest.mark.skipif(sys.platform != "linux", reason="only Linux can set an address-space cap")
def test_generated_code_keeps_its_memory_cap(tmp_path):
    """Lifting the cap on check commands leaves both caps on snippets run through
    `run_sandboxed`, as verification and readiness are. A smoke's caps are below (dm-256)."""
    result = sandbox.run_sandboxed(
        "import json, resource; print(json.dumps({'cap': resource.getrlimit(resource.RLIMIT_AS)[0]}))",
        workdir=tmp_path / "run",
    )
    assert result.ok, result.stderr
    assert result.payload == {"cap": sandbox.DEFAULT_MEMORY_MB * 1024 * 1024}


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


# ── a smoke's memory caps (dm-256) ─────────────────────────────────────────
#
# A smoke with a model starts a DSH worker, which is Node, whose WebAssembly reserves
# far more address space than it uses: under the snippet's 2 GB address-space cap it
# died at load ("WebAssembly.instantiate(): Out of memory"), so create's smoke failed on
# Linux for every mode with an agent node. Such a smoke drops only the address-space
# cap; the heap cap stays, so the mode's own code is still bounded.


def _stage_finishing_with(tmp_path: Path, lines: str) -> None:
    """The counter package, whose `finish` op first runs ``lines``."""
    files = dict(GOOD)
    files["ops.py"] = GOOD["ops.py"].replace("def finish(ctx):\n", "def finish(ctx):\n" + lines)
    _stage(tmp_path, files)


@pytest.mark.skipif(sys.platform != "linux", reason="only Linux enforces these caps")
@pytest.mark.parametrize("model", [False, True])
def test_a_smoke_keeps_the_heap_cap_and_drops_the_address_space_cap_only_for_a_model(tmp_path, model, graph_profile_home):
    import resource

    from refactory_dsh.model import DshModelClient

    # Written beside the run directory, which holds only what the graph declares.
    _stage_finishing_with(tmp_path, (
        "    import json, resource\n"
        "    (ctx.workdir.parent / 'limits.json').write_text(json.dumps({"
        "'as': resource.getrlimit(resource.RLIMIT_AS)[0], 'data': resource.getrlimit(resource.RLIMIT_DATA)[0]}))\n"
    ))
    context = DshModelClient(attempts=1).context() if model else None
    result = sandbox.smoke(f"{NAME}/counter", tmp_path / "run", project=tmp_path, model=model,
                           model_context=context)
    assert result.ok and result.payload.get("exit_ok"), result.stdout + result.stderr
    limits = json.loads((tmp_path / "limits.json").read_text())
    cap = sandbox.DEFAULT_MEMORY_MB * 1024 * 1024
    assert limits == {"as": resource.RLIM_INFINITY if model else cap, "data": cap}


@pytest.mark.skipif(sys.platform != "linux", reason="only Linux enforces these caps")
def test_a_smoke_with_a_model_still_stops_a_heap_bomb(tmp_path, graph_profile_home):
    """The mode's own code in a model smoke cannot take more than the heap cap."""
    from refactory_dsh.model import DshModelClient

    _stage_finishing_with(tmp_path, (
        "    try:\n"
        f"        bytearray({sandbox.DEFAULT_MEMORY_MB + 512} * 1024 * 1024)\n"
        "    except MemoryError:\n"
        "        raise RuntimeError('the heap cap stopped the allocation')\n"
    ))
    result = sandbox.smoke(f"{NAME}/counter", tmp_path / "run", project=tmp_path, model=True,
                           model_context=DshModelClient(attempts=1).context())
    assert result.payload.get("exit_ok") is False, result.stdout + result.stderr
    assert "the heap cap stopped the allocation" in str(result.payload.get("error")), result.payload


def test_a_smoke_with_a_model_starts_its_dsh_worker(tmp_path, graph_profile_home):
    """The real DSH worker, started by the same `sandbox.smoke` call create makes, reaches
    its model endpoint: here a stand-in on this machine that refuses every request, so no
    model is called. Under the address-space cap it died at load and sent nothing.
    Agent nodes own model calls on DSH; operation subprocesses have no model client.
    The endpoint refusal is expected and must survive the graph's error reporting.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from refactory_dsh.model import DshModelClient
    from tests_support import model_home_with_a_key

    requests: list[str] = []

    class Refuse(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            requests.append(self.path)
            self.rfile.read(int(self.headers.get("content-length") or 0))
            self.send_response(400)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"type": "error", "error": {
                "type": "invalid_request_error", "message": "stand-in endpoint"}}).encode())

        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Refuse)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        home = model_home_with_a_key(graph_profile_home, provider="anthropic", model="claude-opus-4-6",
                                     base_url=f"http://127.0.0.1:{server.server_address[1]}")
        files = dict(GOOD)
        files["mode.py"] = files["mode.py"].replace(
            "g.add(nn.Node('start', 'op', call=OPS + ':start',"
            " writes=('state.json',), produces=('value',), knobs=('target',)))",
            "g.add(nn.Node('start', 'agent', writes=('result.txt',),"
            " prompt_template='Write ok to result.txt.'))",
        )
        assert files["mode.py"] != GOOD["mode.py"]
        _stage(tmp_path, files)
        context = DshModelClient(dsh_home=home, attempts=1, timeout_s=120).context()
        result = sandbox.smoke(f"{NAME}/counter", tmp_path / "run", project=tmp_path, model=True,
                               model_context=context, timeout_s=300)
    finally:
        server.shutdown()
    assert "Out of memory" not in result.stdout + result.stderr, result.stdout + result.stderr
    assert result.ok and result.payload.get("exit_ok") is False, result.stdout + result.stderr
    assert "stand-in endpoint" in str(result.payload.get("error")), result.payload
    assert requests, "the worker never reached its model endpoint"


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
    assert "result = evaluate_on_dsh(" in seen["code"]
    assert "mode.evaluate(" not in seen["code"]

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

    def fake_completion(self, cmd, run_dir, stop_at=None, budget_s=None, task=None, target_workdir=None):  # type: ignore[no-untyped-def]
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


# ── run_sandboxed inactivity / backstop tests ──────────────────────────────

def test_run_sandboxed_inactivity_timeout_fires_on_idle(tmp_path):
    """An idle process is stopped after the inactivity limit."""
    result = sandbox.run_sandboxed(
        "import time; time.sleep(30)",
        workdir=tmp_path / "idle",
        timeout_s=1,
    )
    assert result.timed_out
    assert result.timeout_kind == "inactivity"
    assert result.returncode == 124


def test_run_sandboxed_active_item_survives_past_timeout_s(tmp_path):
    """A process producing output survives past timeout_s (inactivity-based)."""
    # Prints every 0.3 s for ~2 s, with a 1 s inactivity limit — should survive.
    code = (
        "import time, json, sys\n"
        "for i in range(7):\n"
        "    print(f'tick {i}', flush=True)\n"
        "    time.sleep(0.3)\n"
        "print(json.dumps({'score': 42}), flush=True)\n"
    )
    result = sandbox.run_sandboxed(code, workdir=tmp_path / "active", timeout_s=1)
    assert result.ok
    assert not result.timed_out
    assert result.timeout_kind is None
    assert result.payload.get("score") == 42


def test_run_sandboxed_backstop_fires_despite_activity(tmp_path, monkeypatch):
    """The backstop fires even when activity keeps resetting the inactivity limit."""
    # Set a very small backstop to avoid waiting
    monkeypatch.setattr(sandbox, "COMMAND_BACKSTOP_S", 2)
    # Print continuously but backstop is 2 s (min(4*1, 2) = 2)
    code = (
        "import time, sys\n"
        "while True:\n"
        "    print('.', flush=True)\n"
        "    time.sleep(0.1)\n"
    )
    result = sandbox.run_sandboxed(code, workdir=tmp_path / "backstop", timeout_s=1)
    assert result.timed_out
    assert result.timeout_kind == "backstop"


def test_run_sandboxed_propagates_inner_timeout_kind(tmp_path):
    """When the payload contains a timeout_kind, it's propagated to SandboxResult."""
    code = (
        "import json\n"
        "print(json.dumps({'exit_ok': False, 'timeout_kind': 'inactivity', 'score': 0}), flush=True)\n"
    )
    result = sandbox.run_sandboxed(code, workdir=tmp_path / "inner", timeout_s=30)
    assert result.timed_out
    assert result.timeout_kind == "inactivity"
