"""Running code we did not write.

The sandbox issue: create-mode imports and runs model-written Python, and nothing
contained it. `check_package` imported it into the driver process, and the smoke run
executed in-process too.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import signal
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
    assert "time limit" in result.problem()


def test_a_sandboxed_run_keeps_its_wall_clock_unless_it_asks_for_inactivity(tmp_path):
    """Only a smoke opts in to the inactivity limit: every other caller (verify, describe,
    readiness) keeps the wall clock it had, so a run that keeps printing is still stopped
    at its limit, reported as the time limit rather than as inactivity."""
    code = "import time\nwhile True:\n    print('.', flush=True)\n    time.sleep(0.1)\n"
    started = time.monotonic()
    result = sandbox.run_sandboxed(code, workdir=tmp_path / "run", timeout_s=1)
    assert time.monotonic() - started < 10
    assert result.timed_out and result.timeout_kind is None
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
        inactivity=True,
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
    result = sandbox.run_sandboxed(code, workdir=tmp_path / "active", timeout_s=1, inactivity=True)
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
    result = sandbox.run_sandboxed(code, workdir=tmp_path / "backstop", timeout_s=1, inactivity=True)
    assert result.timed_out
    assert result.timeout_kind == "backstop"


def test_run_sandboxed_propagates_inner_timeout_kind(tmp_path):
    """When the payload contains a timeout_kind, it's propagated to SandboxResult."""
    code = (
        "import json\n"
        "print(json.dumps({'exit_ok': False, 'timeout_kind': 'inactivity', 'score': 0}), flush=True)\n"
    )
    result = sandbox.run_sandboxed(code, workdir=tmp_path / "inner", timeout_s=30, inactivity=True)
    assert result.timed_out
    assert result.timeout_kind == "inactivity"


# ── a smoke's nested run (the watched sandbox around run_on_dsh) ───────────

#: The inner process a model smoke runs: the real `run_on_dsh`, with DSH replaced by a
#: child that writes CHILD to stdout and its pid beside the run. Everything that would
#: reach a real profile or build is stubbed; the output capture, the inner watcher and
#: the activity mark are the real ones.
_NESTED = '''
import json, sys
from pathlib import Path
from refactory_dsh import dsh_profile, dsh_runtime, lowering, sandbox
dsh_profile.requirement = lambda **kw: None
dsh_runtime.require_build = lambda **kw: None
dsh_runtime.dsh_command = lambda **kw: [sys.executable, "-u", "-c", {child!r}]
lowering.write_mode_graph = lambda *a, **kw: None
lowering.write_declarations = lambda *a, **kw: None
run = dsh_runtime.run_on_dsh({{"knobs": {{}}}}, Path.cwd() / "inner", timeout_s=600, inactivity=True,
                             activity_file=Path.cwd() / sandbox.ACTIVITY_FILE)
print(json.dumps({{"returncode": run.returncode, "ticks": run.stdout.count("tick")}}))
'''


def _nested(child: str, pid_file: Path) -> str:
    pid = f"import os, pathlib; pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
    return _NESTED.format(child=pid + child)


def test_a_smoke_whose_nested_run_keeps_working_outlives_its_inactivity_limit(tmp_path):
    """The nested run's output never reaches the sandbox's pipes (`run_on_dsh` captures
    it), so before the mark the sandbox stopped it `timeout_s` after it began; the mark
    it leaves for each line it sees keeps it running to its end."""
    child = "import time\nfor i in range(12):\n    print('tick', i, flush=True)\n    time.sleep(0.5)\n"
    result = sandbox.run_sandboxed(_nested(child, tmp_path / "dsh.pid"), workdir=tmp_path / "run",
                                   timeout_s=2, inactivity=True, nested_watch=True)
    assert not result.timed_out, result.stderr
    assert result.payload == {"returncode": 0, "ticks": 12}, result.stdout + result.stderr
    assert result.elapsed_s is not None and result.elapsed_s > 4


def test_a_smoke_whose_nested_run_goes_quiet_is_stopped_and_its_run_shut_down(tmp_path):
    """Silence is still stopped at the limit, and the nested run is asked to stop rather
    than killed: it shuts its DSH child down within the sandbox's grace."""
    child = "import time\ntime.sleep(120)\n"
    result = sandbox.run_sandboxed(_nested(child, tmp_path / "dsh.pid"), workdir=tmp_path / "run",
                                   timeout_s=2, inactivity=True, nested_watch=True)
    assert result.timed_out and result.timeout_kind == "inactivity", result.stderr
    assert result.elapsed_s is not None and result.elapsed_s < sandbox.STOP_GRACE_S
    assert "cleanup unconfirmed" not in result.stderr
    pid = int((tmp_path / "dsh.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


# ── a smoke's sample of a Task-bound mode ──────────────────────────────────

_SAMPLED = {
    "__init__.py": (
        "from refactory_dsh.nn import Package\n"
        "from .mode import Knobbed, Plain\n\n"
        "PACKAGE = Package(name='sandboxed_pkg', modes=(Knobbed, Plain),"
        " ops=('refactory_packages.sandboxed_pkg.ops',))\n"
    ),
    "task.py": (
        "from refactory_dsh.task import Task, TaskInstance, VerifyResult\n\n"
        "class Many(Task):\n"
        "    name = 'many'\n"
        "    def instances(self, **kw):\n"
        # Listed highest id first: the first items listed are the ones the train part
        # holds out (`task.partition` holds out the highest ids).
        "        return [TaskInstance(id=f'item_{i:03d}', metadata={}) for i in reversed(range(177))]\n"
        "    def setup(self, instance, workspace, **kw):\n"
        "        pass\n"
        "    def verify(self, instance, workspace, **kw):\n"
        "        return VerifyResult(passed=True, score=1.0)\n"
    ),
    "ops.py": (
        "import json, time\n\n"
        "def work(ctx):\n"
        "    item = ctx.read_json('current_item.json', {}) or {}\n"
        "    started = time.time()\n"
        "    time.sleep(1.0)\n"
        "    ctx.write_json('span.json', {'id': item.get('id'), 'start': started, 'end': time.time()})\n"
        "    return {}\n\n"
        "def finish(ctx):\n"
        "    ctx.write_json('best.json', {'best': 1.0})\n"
        "    return {'best': 1.0}\n"
    ),
    "mode.py": (
        "from refactory_dsh import nn\n"
        "OPS = 'refactory_packages.sandboxed_pkg.ops'\n"
        "TASK = 'refactory_packages.sandboxed_pkg.task:Many'\n\n"
        "def _graph(**data):\n"
        "    g = nn.Graph()\n"
        "    g.add(nn.Node('data', 'data', subgraph_entry='work', subgraph_exit='work',"
        " task_ref=TASK, writes=('current_item.json',), produces=('item_scores',), **data))\n"
        "    g.add(nn.Node('work', 'op', call=OPS + ':work', reads=('current_item.json',),"
        " writes=('span.json',)))\n"
        "    g.add(nn.Node('finish', 'op', call=OPS + ':finish', writes=('best.json',),"
        " produces=('best',)))\n"
        "    g.connect('data', 'finish')\n"
        "    return g\n\n"
        "class Knobbed(nn.Mode):\n"
        "    name = 'knobbed'\n"
        "    @classmethod\n"
        "    def declare(cls):\n"
        "        return {'data_parallelism': nn.Knob(24, 'int', lo=1, hi=64, frozen=True)}\n"
        "    def build(self):\n"
        "        return _graph(parallelism_knob='data_parallelism', knobs=('data_parallelism',))\n"
        "    def score(self, result):\n"
        "        return 1.0\n\n"
        "class Plain(nn.Mode):\n"
        "    name = 'plain'\n"
        "    def build(self):\n"
        "        return _graph()\n"
        "    def score(self, result):\n"
        "        return 1.0\n"
    ),
}


def _most_at_once(spans: list[dict]) -> int:
    """The most spans that were open at one moment."""
    edges = sorted([(span["start"], 1) for span in spans] + [(span["end"], -1) for span in spans],
                   key=lambda edge: (edge[0], edge[1]))
    most = current = 0
    for _at, step in edges:
        current += step
        most = max(most, current)
    return most


@pytest.mark.parametrize(("mode", "knobs", "at_once"), [
    ("knobbed", None, 3),                       # the knob, 24, is no ceiling for 3 items
    ("knobbed", {"data_parallelism": "2"}, 2),  # the mode's own knob caps the sample
    ("plain", None, 3),                         # no knob: the node's parallelism is set
])
def test_a_smoke_of_a_task_bound_mode_runs_a_parallel_sample_of_its_train_part(
        tmp_path, mode, knobs, at_once):
    """The smoke create runs for this mode: 3 of the Task's 177 items, from its train
    part, at once. The mode has no agent node, so create smokes it without a model
    (`create_ops.smoke` passes ``model=bool(agent_nodes)``), on the walker.

    A node whose parallelism is a knob cannot also be given ``parallelism`` (the graph
    refuses both), so the knob is set instead, held to its own value."""
    _stage(tmp_path, _SAMPLED)
    run = tmp_path / "run"
    result = sandbox.smoke(f"{NAME}/{mode}", run, project=tmp_path, model=False, knobs=knobs,
                           smoke_items=3, timeout_s=120)
    assert result.ok and result.payload.get("exit_ok"), result.stdout + result.stderr
    assert result.payload["item_limit"] == 3
    spans = [json.loads(path.read_text()) for path in run.glob("passes/data/*/span.json")]
    # The lowest ids of the train part, which the default split holds out last.
    assert sorted(span["id"] for span in spans) == ["item_000", "item_001", "item_002"]
    assert _most_at_once(spans) == at_once


def _held_out(ids: list[str], holdout: float) -> set[str]:
    """Which of the sampled Task's ids a run with ``holdout`` holds out."""
    from refactory_dsh import task as task_module

    items = [{"id": f"item_{i:03d}"} for i in reversed(range(177))]
    return {item["id"] for item in task_module.partition(items, holdout)[task_module.HOLDOUT]
            if item["id"] in ids}


@pytest.mark.parametrize("holdout", [0.4, 0.9])
def test_a_smoke_samples_the_train_part_of_the_split_its_runs_use(tmp_path, holdout):
    """Under a holdout other than the default the sample is still train: none of it is
    held out by that split."""
    _stage(tmp_path, _SAMPLED)
    run = tmp_path / "run"
    result = sandbox.smoke(f"{NAME}/plain", run, project=tmp_path, smoke_items=3,
                           holdout=holdout, timeout_s=120)
    assert result.ok and result.payload.get("exit_ok"), result.stdout + result.stderr
    ran = sorted(json.loads(path.read_text())["id"]
                 for path in run.glob("passes/data/*/span.json"))
    assert len(ran) == 3 and _held_out(ran, holdout) == set(), ran


def test_a_smoke_samples_only_the_train_ids_of_a_parts_manifest(tmp_path):
    """Under a parts manifest the sample comes from its train ids only, never its val or
    test ids -- here the lowest ids, which a fraction split would have kept in train."""
    _stage(tmp_path, _SAMPLED)
    manifest = tmp_path / "parts.json"
    manifest.write_text(json.dumps({
        "train": [f"item_{i:03d}" for i in range(100, 177)],
        "val": [f"item_{i:03d}" for i in range(0, 50)],
        "test": [f"item_{i:03d}" for i in range(50, 100)],
    }))
    run = tmp_path / "run"
    result = sandbox.smoke(f"{NAME}/plain", run, project=tmp_path, smoke_items=3,
                           parts_manifest=manifest, timeout_s=120)
    assert result.ok and result.payload.get("exit_ok"), result.stdout + result.stderr
    ran = sorted(json.loads(path.read_text())["id"]
                 for path in run.glob("passes/data/*/span.json"))
    assert ran == ["item_100", "item_101", "item_102"]


# ── a smoke's inactivity limit, on the spine: what counts as work ──────────

#: One mode per way a run works without the inner run's own output: an op that prints,
#: an op that rewrites a file and prints nothing, one item running silent nodes in a
#: row, and an op that does nothing at all. Each takes longer than the limit.
_WORKING = {
    "__init__.py": (
        "from refactory_dsh.nn import Package\n"
        "from .mode import Printing, Rewriting, Steps, Streaming, Silent\n\n"
        "PACKAGE = Package(name='sandboxed_pkg', modes=(Printing, Rewriting, Steps, Streaming,"
        " Silent),"
        " ops=('refactory_packages.sandboxed_pkg.ops',))\n"
    ),
    "task.py": (
        "from refactory_dsh.task import Task, TaskInstance, VerifyResult\n\n"
        "class One(Task):\n"
        "    name = 'one'\n"
        "    def instances(self, **kw):\n"
        "        return [TaskInstance(id='item_000', metadata={})]\n"
        "    def setup(self, instance, workspace, **kw):\n"
        "        pass\n"
        "    def verify(self, instance, workspace, **kw):\n"
        "        return VerifyResult(passed=True, score=1.0)\n"
    ),
    "ops.py": (
        "import json, os, time\n"
        "from pathlib import Path\n\n"
        "def printing(ctx):\n"
        "    for i in range(20):\n"
        "        print('tick', i, flush=True)\n"
        "        time.sleep(0.25)\n"
        "    return {'best': 1.0}\n\n"
        "def rewriting(ctx):\n"
        "    for i in range(20):\n"
        "        ctx.write_json('span.json', {'tick': i})\n"
        "        time.sleep(0.25)\n"
        "    return {'best': 1.0}\n\n"
        "def step(ctx):\n"
        "    time.sleep(1.5)\n"
        "    return {}\n\n"
        # What DSH's worker observer does while a model streams (`worker_observer.mjs`):
        # touch the worker's activity file, in the item's pass, and nothing else.
        "def streaming(ctx):\n"
        "    alive = ctx.workdir / '.refactory' / 'workers' / 'worker-test.activity'\n"
        "    alive.parent.mkdir(parents=True, exist_ok=True)\n"
        "    for i in range(20):\n"
        "        alive.touch()\n"
        "        time.sleep(0.25)\n"
        "    return {}\n\n"
        "def silent(ctx):\n"
        "    (ctx.workdir / 'op.pid').write_text(str(os.getpid()))\n"
        "    time.sleep(120)\n"
        "    return {'best': 1.0}\n\n"
        "def finish(ctx):\n"
        "    return {'best': 1.0}\n"
    ),
    "mode.py": (
        "from refactory_dsh import nn\n"
        "OPS = 'refactory_packages.sandboxed_pkg.ops'\n"
        "TASK = 'refactory_packages.sandboxed_pkg.task:One'\n\n"
        "def _one(call, writes=()):\n"
        "    g = nn.Graph()\n"
        "    g.add(nn.Node('work', 'op', call=OPS + ':' + call, writes=writes, produces=('best',)))\n"
        "    return g\n\n"
        "class _Scored(nn.Mode):\n"
        "    def score(self, result):\n"
        "        return 1.0\n\n"
        "class Printing(_Scored):\n"
        "    name = 'printing'\n"
        "    def build(self):\n"
        "        return _one('printing')\n\n"
        "class Rewriting(_Scored):\n"
        "    name = 'rewriting'\n"
        "    def build(self):\n"
        "        return _one('rewriting', writes=('span.json',))\n\n"
        "class Silent(_Scored):\n"
        "    name = 'silent'\n"
        "    def build(self):\n"
        "        return _one('silent')\n\n"
        "def _item(*calls):\n"
        "    g = nn.Graph()\n"
        "    names = [f'n{i}' for i in range(len(calls))]\n"
        "    g.add(nn.Node('data', 'data', subgraph_entry=names[0], subgraph_exit=names[-1],"
        " task_ref=TASK, writes=('current_item.json',), produces=('item_scores',)))\n"
        "    for name, call in zip(names, calls):\n"
        "        g.add(nn.Node(name, 'op', call=OPS + ':' + call, reads=('current_item.json',)))\n"
        "    for first, then in zip(names, names[1:]):\n"
        "        g.connect(first, then)\n"
        "    g.add(nn.Node('finish', 'op', call=OPS + ':finish', produces=('best',)))\n"
        "    g.connect('data', 'finish')\n"
        "    return g\n\n"
        "class Steps(_Scored):\n"
        "    name = 'steps'\n"
        "    def build(self):\n"
        "        return _item('step', 'step', 'step')\n\n"
        "class Streaming(_Scored):\n"
        "    name = 'streaming'\n"
        "    def build(self):\n"
        "        return _item('streaming')\n"
    ),
}

#: How long the run may go without work, in each runtime. A model smoke's own limit is
#: the run's (`run_on_dsh`), a little less than the sandbox's: 6 s gives the run 3 s.
_LIMITS = {"spine": 6, "walker": 3}
#: The run's own limit, either way: every working op above runs longer than this.
_RUN_LIMIT_S = 3


def _smoke(tmp_path: Path, mode: str, runtime: str) -> sandbox.SandboxResult:
    """The smoke create runs, of ``mode``: with a model it goes `sandbox.smoke` ->
    `evaluate_on_dsh` -> `run_on_dsh`, on the graph profile's spine (`graph-owner.mjs`,
    which spawns each node command in a process group of its own, as DSH does); without
    one, the walker runs in the sandbox's own process."""
    from refactory_dsh.model import DshModelClient
    from tests_support import graph_owner_launcher

    _stage(tmp_path, _WORKING)
    if runtime == "walker":
        return sandbox.smoke(f"{NAME}/{mode}", tmp_path / "run", project=tmp_path, model=False,
                             smoke_items=3, timeout_s=_LIMITS[runtime])
    launcher = graph_owner_launcher(tmp_path / "dsh")
    if launcher is None:
        pytest.skip("the spine owner needs Node and the installed workspace (pnpm install)")
    context = DshModelClient(attempts=1, dsh_cli=str(launcher)).context()
    return sandbox.smoke(f"{NAME}/{mode}", tmp_path / "run", project=tmp_path, model=True,
                         model_context=context, smoke_items=3, timeout_s=_LIMITS[runtime])


@pytest.mark.parametrize("runtime", ["spine", "walker"])
@pytest.mark.parametrize("mode", [
    "printing",   # an op that prints for 5 s and writes nothing
    "rewriting",  # an op that rewrites a declared file for 5 s and prints nothing
    "steps",      # one item, three silent 1.5 s nodes in a row
    "streaming",  # an op in an item whose worker's activity file moves for 5 s
])
def test_a_smoke_that_keeps_working_outlives_its_inactivity_limit(
        tmp_path, graph_profile_home, mode, runtime):
    """Work inside a node or an item counts: none of these is stopped, though each takes
    longer than the run allows without work."""
    result = _smoke(tmp_path, mode, runtime)
    assert not result.timed_out, result.stdout + result.stderr
    assert result.ok and result.payload.get("exit_ok"), result.stdout + result.stderr
    assert result.elapsed_s is not None and result.elapsed_s > _RUN_LIMIT_S + 1


@pytest.mark.parametrize("runtime", ["spine", "walker"])
def test_a_smoke_whose_op_goes_quiet_is_stopped_and_leaves_no_op_running(
        tmp_path, graph_profile_home, runtime):
    """An op that does nothing is stopped at the limit, and the stop ends the op's own
    process too -- on the spine, though the op runs in a process group of its own:
    nothing the run started is left running without a parent."""
    result = _smoke(tmp_path, "silent", runtime)
    pid = int((tmp_path / "run" / "op.pid").read_text())
    try:
        assert result.timed_out and result.timeout_kind == "inactivity", result.stdout + result.stderr
        assert result.elapsed_s is not None and result.elapsed_s < _LIMITS[runtime] + 15
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        try:  # only the op this test started, should the assertion above fail
            os.kill(pid, 9)
        except ProcessLookupError:
            pass


def test_a_look_at_a_run_costs_the_same_however_many_files_it_holds(tmp_path, monkeypatch):
    """The signs of work are a fixed set of files per item and worker (`activity.signs`):
    a checkout of thousands of files in the run directory adds nothing to a look."""
    from refactory_dsh import activity

    run = tmp_path / "run"
    (run / ".refactory" / "workers").mkdir(parents=True)
    (run / ".refactory" / "workers" / "worker-a.activity").write_text("")
    (run / "passes" / "data" / "item_0" / ".refactory").mkdir(parents=True)

    def calls() -> int:
        counted = [0]
        real_stat, real_scandir = os.stat, os.scandir

        def stat(*a, **kw):
            counted[0] += 1
            return real_stat(*a, **kw)

        def scandir(*a, **kw):
            counted[0] += 1
            return real_scandir(*a, **kw)

        monkeypatch.setattr(os, "stat", stat)
        monkeypatch.setattr(os, "scandir", scandir)
        try:
            activity.signs(run)
        finally:
            monkeypatch.setattr(os, "stat", real_stat)
            monkeypatch.setattr(os, "scandir", real_scandir)
        return counted[0]

    empty = calls()
    for folder in range(20):
        for name in range(100):
            path = run / "checkout" / f"d{folder}" / f"f{name}.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
            (run / "passes" / "data" / "item_0" / f"f{folder}_{name}.txt").write_text("")
    assert calls() == empty


def test_a_watched_sandbox_never_counts_the_runs_budget_ledger(tmp_path):
    """The sandbox's own look (a smoke without a model) leaves the run's budget ledger
    out, wherever it lives; the same file under another name is a sign of work."""
    run = tmp_path / "run"
    workers = run / ".refactory" / "workers"
    workers.mkdir(parents=True)
    ledger, other = workers / "ledger.json", workers / "worker-a.json"
    env = {"REFACTORY_BUDGET_LEDGER": str(ledger)}
    before = sandbox._signs_of_work(run, env, nested_watch=False)
    ledger.write_text("spent")
    assert sandbox._signs_of_work(run, env, nested_watch=False) == before
    other.write_text("working")
    assert sandbox._signs_of_work(run, env, nested_watch=False) != before


@pytest.mark.parametrize("model", [True, False])
def test_a_run_is_looked_at_in_one_place(tmp_path, monkeypatch, model):
    """A model smoke's own run (`run_on_dsh`) looks at the run's signs of work and marks
    what it sees; the sandbox around it then reads only that mark. Without a model the
    sandbox is the only watcher, and looks at the signs itself."""
    _stage(tmp_path, GOOD)
    seen: dict[str, object] = {}
    monkeypatch.setattr(sandbox, "run_sandboxed",
                        lambda code, **kw: seen.update(kw) or sandbox.SandboxResult(ok=True))
    sandbox.smoke(f"{NAME}/counter", tmp_path / "run", project=tmp_path, model=model,
                  model_context={})
    assert seen["nested_watch"] is model

    run = tmp_path / "run"
    (run / ".refactory" / "workers").mkdir(parents=True, exist_ok=True)
    before = sandbox._signs_of_work(run, {}, nested_watch=True)
    (run / ".refactory" / "workers" / "worker-a.activity").write_text("")
    assert sandbox._signs_of_work(run, {}, nested_watch=True) == before
    (run / sandbox.ACTIVITY_FILE).write_text("seen")
    assert sandbox._signs_of_work(run, {}, nested_watch=True) != before


@pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_CPU is POSIX")
def test_a_smoke_that_hits_its_cpu_limit_says_so_and_leaves_nothing_running(tmp_path):
    """However a watched smoke's process ends -- here its CPU limit, which ends it with
    SIGXCPU, so it cannot shut its own DSH run down -- the sandbox ends what it left
    running (the DSH child, whose group the run recorded) and says why it ended."""
    child = "import time\ntime.sleep(120)\n"
    # Once the stand-in DSH child is running, the smoke's own process spins.
    burn = ("import pathlib, threading, time\n"
            "def _burn():\n"
            f"    while not pathlib.Path({str(tmp_path / 'dsh.pid')!r}).exists():\n"
            "        time.sleep(0.05)\n"
            "    while True:\n"
            "        pass\n"
            "threading.Thread(target=_burn, daemon=True).start()\n")
    # As a model smoke runs it: the nested run's directory is the sandbox's own.
    nested = _nested(child, tmp_path / "dsh.pid").replace('Path.cwd() / "inner"', "Path.cwd()")
    result = sandbox.run_sandboxed(burn + nested, workdir=tmp_path / "run", timeout_s=60,
                                   inactivity=True, nested_watch=True, cpu_s=3)
    pid = int((tmp_path / "dsh.pid").read_text())
    try:
        assert result.returncode == -signal.SIGXCPU, (result.returncode, result.stderr)
        assert "the smoke hit its CPU limit (3 s of CPU time per process)" in result.stderr
        assert result.problem() == "the smoke hit its CPU limit (3 s of CPU time per process)"
        deadline = time.monotonic() + 5  # an ended process is gone once it is reaped
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        try:  # only the stand-in this test started, should an assertion above fail
            os.kill(pid, 9)
        except ProcessLookupError:
            pass


@pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_CPU is POSIX")
def test_a_sandboxed_process_reaches_its_soft_cpu_limit_before_its_hard_one():
    """The hard CPU limit sits a second above the soft one, so Linux ends a process at
    its CPU limit with SIGXCPU, which says what happened, rather than with the SIGKILL
    it sends when the two are equal."""
    import subprocess

    probe = "import resource; print(*resource.getrlimit(resource.RLIMIT_CPU))"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         check=True, preexec_fn=sandbox._limits(0, 7, 10))
    assert out.stdout.split() == ["7", "8"]


def test_a_smoke_is_given_a_cpu_limit_it_cannot_reach_before_its_backstop(tmp_path, monkeypatch):
    """A smoke may work for hours, up to its backstop: its CPU limit is that backstop on
    every core, as a command's is, so only a runaway reaches it first."""
    _stage(tmp_path, GOOD)
    seen: dict[str, object] = {}
    monkeypatch.setattr(sandbox, "run_sandboxed",
                        lambda code, **kw: seen.update(kw) or sandbox.SandboxResult(ok=True))
    sandbox.smoke(f"{NAME}/counter", tmp_path / "run", project=tmp_path, model=True,
                  model_context={}, timeout_s=3600)
    backstop = sandbox.smoke_backstop_s(3600)
    assert seen["backstop_s"] == backstop
    assert seen["cpu_s"] == math.ceil(backstop) * (os.cpu_count() or 1)
    assert seen["cpu_s"] >= backstop > sandbox.DEFAULT_CPU_S


def test_a_watchers_look_interval_follows_its_limit():
    """A one-hour limit is checked every 10 s, not twice a second; a short limit in a
    twentieth of itself, never more often than the floor."""
    from refactory_dsh import activity

    assert activity.look_interval(3600, sandbox.WATCH_INTERVAL_S) == 10.0
    assert activity.look_interval(60, sandbox.WATCH_INTERVAL_S) == 3.0
    assert activity.look_interval(3, sandbox.WATCH_INTERVAL_S) == sandbox.WATCH_INTERVAL_S


# ── a smoke's sample cuts only a smoke's run ───────────────────────────────

from refactory_dsh.task import Task as _Task  # noqa: E402
from refactory_dsh.task import TaskInstance as _TaskInstance  # noqa: E402
from refactory_dsh.task import VerifyResult as _VerifyResult  # noqa: E402


class _Listed(_Task):
    IDS: tuple[str, ...] = ()

    def instances(self, **kw):
        return [_TaskInstance(id=item, metadata={}) for item in self.IDS]

    def setup(self, instance, workspace, **kw):
        pass

    def verify(self, instance, workspace, **kw):
        return _VerifyResult(passed=True, score=1.0)


class OverlapA(_Listed):
    """Shares ``a2`` with `OverlapB`, whose lowest id it is."""
    name = "overlap_a"
    IDS = ("a1", "a2", "z9")


class OverlapB(_Listed):
    name = "overlap_b"
    IDS = ("b1", "a2")


def _two_task_nodes():
    from refactory_dsh import nn

    class TwoTasks(nn.Mode):
        name = "two_tasks"

        def build(self):
            graph = nn.Graph()
            for node_id, task in (("first", "OverlapA"), ("second", "OverlapB")):
                graph.add(nn.Node(node_id, "data", subgraph_entry=f"{node_id}_work",
                                  subgraph_exit=f"{node_id}_work", task_ref=f"test_sandbox:{task}",
                                  writes=("current_item.json",), produces=("item_scores",),
                                  limit=2))
                graph.add(nn.Node(f"{node_id}_work", "op", call="loop_ops:noop"))
            graph.connect("first", "second")
            return graph

        def score(self, result):
            return 1.0

    return TwoTasks()


def test_a_task_bound_node_outside_a_smoke_runs_its_whole_part(tmp_path):
    """Outside a smoke a Task-bound node iterates its whole selected part, its own
    ``limit`` included, as before smokes were sampled: only a smoke cuts it."""
    from refactory_dsh.nn import Node
    from refactory_dsh.nn.executor import load_items

    mode = _two_task_nodes()
    nodes = mode.build().nodes
    assert [item["id"] for item in load_items(nodes["first"], tmp_path)] == ["a1", "a2", "z9"]
    assert isinstance(nodes["first"], Node) and nodes["first"].limit == 2


def test_each_task_bound_node_of_a_smoke_runs_only_its_own_sample(tmp_path):
    """Two Task-bound nodes whose ids overlap: the run's part names both samples, yet
    each node runs only its own -- `first` would otherwise also run `a2`, `second`'s."""
    from refactory_dsh.nn import Node
    from refactory_dsh.nn.executor import load_items

    mode = _two_task_nodes()
    sample = sandbox.smoke_sample(mode, tmp_path, 1, holdout=0.0)
    assert sample["items"] == ["a1", "a2"]
    graph_nodes = {node_id: Node.from_state({**entry, "id": node_id})
                   for node_id, entry in mode.state_dict()["graph"]["nodes"].items()}
    assert [item["id"] for item in load_items(graph_nodes["first"], tmp_path)] == ["a1"]
    assert [item["id"] for item in load_items(graph_nodes["second"], tmp_path)] == ["a2"]


def test_a_smoke_of_id_less_items_runs_the_items_its_sample_names(tmp_path):
    """Items with no id are named by their position among all the node's items, by the
    sample, the run's part and the loader alike: a sample of 12 id-less items names
    "0", "1" and "10" (ids sort as text), and those three run -- not the item at
    position 2 of the selection, which is item 10."""
    from refactory_dsh import nn
    from refactory_dsh.nn import Node
    from refactory_dsh.nn.executor import load_items

    class Unnamed(nn.Mode):
        name = "unnamed"

        def build(self):
            graph = nn.Graph()
            graph.add(nn.Node("data", "data", subgraph_entry="work", subgraph_exit="work",
                              task_ref="test_sandbox:OverlapA", writes=("current_item.json",),
                              produces=("item_scores",),
                              inline_items=tuple({"n": n} for n in range(12))))
            graph.add(nn.Node("work", "op", call="loop_ops:noop"))
            return graph

        def score(self, result):
            return 1.0

    mode = Unnamed()
    sample = sandbox.smoke_sample(mode, tmp_path, 3, holdout=0.0)
    assert sample["per_node"] == {"data": ["0", "1", "10"]}
    node = Node.from_state({**mode.state_dict()["graph"]["nodes"]["data"], "id": "data"})
    assert [item["n"] for item in load_items(node, tmp_path)] == [0, 1, 10]


def test_a_task_bound_node_with_no_train_item_is_refused_before_the_run(tmp_path):
    """A node that would run nothing is refused, naming it, rather than run empty."""
    from refactory_dsh import nn

    class Empty(nn.Mode):
        name = "empty"

        def build(self):
            graph = nn.Graph()
            graph.add(nn.Node("data", "data", subgraph_entry="work", subgraph_exit="work",
                              task_ref="test_sandbox:Nothing", writes=("current_item.json",),
                              produces=("item_scores",)))
            graph.add(nn.Node("work", "op", call="loop_ops:noop"))
            return graph

        def score(self, result):
            return 1.0

    with pytest.raises(sandbox.SampleRefused, match="data node data has no train item to run"):
        sandbox.smoke_sample(Empty(), tmp_path, 3)


class Nothing(_Listed):
    name = "nothing"
    IDS = ()


@pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_CORE is POSIX")
def test_a_sandboxed_process_leaves_no_core_dump():
    """SIGXCPU ends a process with a core dump by default: the sandbox allows none."""
    import subprocess

    probe = "import resource; print(*resource.getrlimit(resource.RLIMIT_CORE))"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         check=True, preexec_fn=sandbox._limits(0, 7, 10))
    assert out.stdout.split() == ["0", "0"]


@pytest.mark.parametrize("mode", ["create", "create_fast", "edit", "edit_fast"])
def test_the_smoke_node_of_each_create_mode_consumes_the_split_knobs(mode):
    """The split knobs are the smoke node's: a node consumes only the knobs it lists, and
    a declared knob no node consumes fails validation (V7)."""
    from refactory_dsh import registry
    from refactory_dsh.nn.validate import validate_state

    package = registry.load("default_pack")
    mode_class = next(cls for cls in package.modes if cls.name == mode)
    graph = mode_class().build()
    consumers = graph.knob_consumers()
    for name in ("profile", "holdout", "parts_manifest", "smoke_items"):
        assert "smoke" in consumers.get(name, ()), (mode, name)
    issues = [issue for issue in validate_state(mode_class().state_dict(), resolve_calls=False)
              if getattr(issue, "code", "") == "V7" and getattr(issue, "knob", "") in
              ("profile", "holdout", "parts_manifest")]
    assert not issues, issues


@pytest.mark.parametrize("given", [None, 5000.0], ids=["default", "explicit"])
def test_a_smokes_backstop_never_passes_the_runs_budget(tmp_path, monkeypatch, given):
    """With a run budget active, the backstop is held to what the budget has left --
    a caller's explicit backstop (the empty-input run's share of a smoke) included."""
    from refactory_dsh.budgets import ResourceLimits, RunBudget

    _stage(tmp_path, GOOD)
    budget = RunBudget.create(tmp_path / "ledger", ResourceLimits(max_elapsed_s=60))
    seen: dict[str, object] = {}
    monkeypatch.setattr(sandbox, "run_sandboxed",
                        lambda code, **kw: seen.update(kw) or sandbox.SandboxResult(ok=True))
    extra = {} if given is None else {"backstop_s": given}
    sandbox.smoke(f"{NAME}/counter", tmp_path / "run", project=tmp_path, model=False,
                  model_context={"budget_ledger": str(budget.path)}, timeout_s=3600, **extra)
    assert 0 < float(seen["backstop_s"]) <= 60
    assert float(seen["timeout_s"]) <= 60
