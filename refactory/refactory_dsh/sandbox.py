"""Running code we did not write.

create-mode writes Python and then we run it. Nothing sandboxed it: `check_package`
imported the generated package into the driver process to validate it, and the smoke
run executed it in-process too. Hand-written modes are trusted; a generated package
is not, and that distinction only appeared when create-mode existed.

Two bars, because they catch different things:

- **Validation without execution.** A generated package is parsed, not imported. An
  AST check guarantees it is syntactically valid, declares a `PACKAGE`, and declares
  its modes and operations -- which is what most broken generations actually are --
  and it guarantees it without running a line of the model's code.
- **Execution under limits.** The smoke run is a real execution, so it happens in a
  subprocess with resource limits, a scrubbed environment and a working directory
  it cannot escape.

The old system sandboxed the candidate program (`srf/ops/common/sandbox.py`, under
firejail or gvisor). This is the same idea for a different artifact.
"""

from __future__ import annotations

import ast
import collections
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from refactory_dsh import registry
from refactory_dsh.paths import project_root, project_for_nodes

#: Ceilings for one sandboxed run. A shell command (`run_command`) is bounded by its
#: inactivity limit (seconds of no stdout/stderr) rather than by the memory and CPU
#: ceilings, which hold for code snippets (#325).
#:
#: Measured silent gaps (PYTHONUNBUFFERED=1, 2026-11-21):
#:   ./install.sh           1.8 s max gap   (18 s total)
#:   pnpm run docs:generate 1.5 s max gap   (4 s total)
#:   pnpm test (vitest)    35.0 s max gap   (79 s total)
#:   pytest -q -n 16       ~60 s max gap   (~1170 s total, estimated from prior runs)
#:
#: 3× the largest gap (60 s) = 180 s, which is today's default.
DEFAULT_TIMEOUT_S = 180
#: Inactivity timeout for smoke passes; stops when no progress for this many seconds.
DEFAULT_MODEL_SMOKE_TIMEOUT_S = 3600
DEFAULT_MEMORY_MB = 2048
DEFAULT_CPU_S = 120
DEFAULT_FILE_MB = 16384

#: The absolute wall-clock backstop for any command, matching the model session backstop
#: (``model.DEFAULT_BACKSTOP_S``).  A command that keeps producing output is not stopped
#: before this.  Overridable with ``REFACTORY_COMMAND_BACKSTOP_S``.
COMMAND_BACKSTOP_S = int(os.environ.get("REFACTORY_COMMAND_BACKSTOP_S", "14400"))

#: What `run_command` keeps of each output stream: all of it up to this many bytes, and
#: beyond it the first part and the last ``OUTPUT_TAIL_BYTES`` with a note of what was
#: left out.  This repository's own full pytest logs measure 0.46-0.64 MB (2026-11-22),
#: so 8 MiB keeps a log more than ten times that size whole.
OUTPUT_KEEP_BYTES = 8 * 1024 * 1024
#: Of a stream longer than ``OUTPUT_KEEP_BYTES``, how much of its end is kept.
OUTPUT_TAIL_BYTES = 1024 * 1024
#: A command whose stdout and stderr together pass this many bytes is stopped, reported
#: as ``output_limit`` with its output kept, as an inactivity or backstop stop is: a
#: logging loop keeps the inactivity limit from firing and would otherwise run to the
#: backstop.  400 times the largest real pytest log above.
OUTPUT_LIMIT_BYTES = 256 * 1024 * 1024
#: How long `run_command` waits, after the command's own process exits, for what it
#: started in the background to close its output, before that is stopped.
EXIT_GRACE_S = 2.0
#: What can stop a command before it exits (`SandboxResult.timeout_kind`).
STOP_KINDS = ("inactivity", "backstop", "output_limit")

#: Where a run nested in a watched sandbox (`run_sandboxed` with ``inactivity``) marks
#: that it is working, relative to the sandbox's directory: the output of its DSH child,
#: which the nested run keeps in its own process (`dsh_runtime.run_on_dsh`), and its
#: own shutdown once it stops for inactivity.
ACTIVITY_FILE = ".refactory/sandbox-activity"
#: How often a watched run's signs of work are looked at (`activity.signs`).
WATCH_INTERVAL_S = 0.5
#: How long a watched sandbox waits, once it has asked its run to stop, before killing
#: it: the run's own shutdown of its DSH child (`owned_process.SHUTDOWN_GRACE_S`) with
#: room to report.
STOP_GRACE_S = 30

#: Shell exit codes for signals that may be the change's own fault (SIGABRT, SIGBUS,
#: SIGSEGV).  Retried once in run_command; if they persist, the base comparison decides
#: whether the crash is the change's fault or infrastructure.
_AMBIGUOUS_SHELL_EXITS = frozenset({134, 135, 139})


def is_signal_crash(result: SandboxResult) -> tuple[bool, str]:
    """Whether a sandbox result is a signal crash that survived run_command's retry.

    Returns (True, signal_label) when the return code is a direct signal
    (rc < 0) or a shell-wrapped SIGABRT/SIGBUS/SIGSEGV (exit 134/135/139).
    run_command already retried once for both cases; a positive return code
    in _AMBIGUOUS_SHELL_EXITS means the retry also crashed.
    """
    rc = result.returncode
    if rc is None:
        return False, ""
    if rc < 0:
        sig_num = -rc
        try:
            sig_name = signal.Signals(sig_num).name
        except (ValueError, KeyError):
            sig_name = f"signal {sig_num}"
        return True, f"killed by {sig_name}"
    if rc in _AMBIGUOUS_SHELL_EXITS:
        sig_num = rc - 128
        try:
            sig_name = signal.Signals(sig_num).name
        except (ValueError, KeyError):
            sig_name = f"signal {sig_num}"
        return True, f"killed by {sig_name}"
    return False, ""

#: Anything whose name looks like a credential is removed from the environment.
#: A generated package has no reason to read a key, and every reason not to.
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")

#: The file the EXIT-trap epilogue writes its final cwd into when it leaves the
#: worktree root.  Parsed from there and stripped from the child's output.
_CWD_SENTINEL_FILE = ".refactory-cwd-sentinel"

#: The environment a sandboxed run is allowed to see. An allowlist, not a denylist:
#: a name nobody thought of is a name that is not passed.
_ALLOWED_ENV = (
    "PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "PYTHONPATH", "VIRTUAL_ENV",
    "REFACTORY_WORKDIR", "REFACTORY_KNOBS", "DSH_WORKFLOW_NODE", "SYSTEMROOT",
    "REFACTORY_BUDGET_LEDGER", "REFACTORY_PYTHON_BIN",
)


class SandboxError(RuntimeError):
    """A sandboxed run could not be carried out."""


@dataclass
class SandboxResult:
    """What a sandboxed run produced."""

    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    timeout_s: int | None = None
    elapsed_s: float | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    prep_failure: dict[str, Any] | None = None
    cwd_outside: str | None = None
    change_caused_prep: dict[str, Any] | None = None
    #: When ``timed_out`` is True, what stopped the command: ``'inactivity'`` (no output
    #: for the inactivity limit), ``'backstop'`` (the absolute wall-clock cap),
    #: ``'output_limit'`` (more output than ``OUTPUT_LIMIT_BYTES``), or ``None`` (a
    #: budget-exhaustion or legacy timeout).
    timeout_kind: str | None = None
    #: Bytes the command wrote to stdout and stderr together, when `run_command` ran it.
    output_bytes: int | None = None

    def stop_reason(self) -> str:
        """What stopped a command before it exited, in words; empty when nothing did."""
        if not self.timed_out or self.timeout_kind not in STOP_KINDS:
            return ""
        if self.timeout_kind == "inactivity":
            return f"inactivity, {self.timeout_s}s silent"
        if self.timeout_kind == "output_limit":
            return f"output limit, {self.output_bytes or 0} bytes of output"
        elapsed = f" {self.elapsed_s:.0f}s elapsed" if self.elapsed_s is not None else ""
        return f"backstop,{elapsed}"

    def problem(self) -> str:
        if self.timed_out:
            stopped = self.stop_reason()
            if stopped:
                tail = output_tail(self.stdout, self.stderr, 5)
                suffix = f" — last output:\n{tail}" if tail else ""
                verb = "stopped" if self.timeout_kind == "output_limit" else "timed out"
                return f"{verb} ({stopped}){suffix}"
            limit = f" of {self.timeout_s}s" if self.timeout_s is not None else ""
            return f"the run exceeded its time limit{limit}; partial artifacts remain in the workdir"
        if self.returncode != 0:
            tail = self.stderr.strip().splitlines()
            return tail[-1] if tail else f"exit {self.returncode}"
        return ""


def output_tail(stdout: str | None, stderr: str | None, n: int) -> str:
    """The last *n* non-empty lines of each stream, each line labelled with its stream.

    For the messages a stopped command leaves: what it printed last can be on either
    stream (pytest reports on stdout, a build tool on stderr), so both are kept.
    """
    if n <= 0:
        return ""
    out: list[str] = []
    for label, text in (("stdout", stdout), ("stderr", stderr)):
        lines = [line for line in (text or "").splitlines() if line.strip()]
        out.extend(f"[{label}] {line}" for line in lines[-n:])
    return "\n".join(out)


def timeout_output(result: SandboxResult, n: int = 200) -> str:
    """The text of ``timeout-output.txt`` for a command stopped before it exited.

    Why it was stopped, then the last *n* lines of each stream, labelled (`output_tail`).
    """
    elapsed = f"{result.elapsed_s:.0f}" if result.elapsed_s is not None else "unknown"
    limit = result.timeout_s if result.timeout_s is not None else "unknown"
    header = (f"Stopped: {result.stop_reason() or 'timeout'} "
              f"(elapsed {elapsed}s, inactivity limit {limit}s)")
    return header + "\n\n" + output_tail(result.stdout, result.stderr, n) + "\n"


class _Output:
    """One output stream of a command, kept within ``keep`` bytes.

    Whole up to ``keep`` bytes; beyond that, the first ``keep - tail`` bytes and the
    last ``tail`` bytes, with a note of how much was left out between them.  Chunks are
    appended as they arrive and never re-copied, so reading is linear in the output.
    """

    def __init__(self, keep: int, tail: int) -> None:
        self.tail_cap = max(0, min(tail, keep))
        self.head_cap = max(0, keep - self.tail_cap)
        self.head = bytearray()
        self.tail: collections.deque[bytes] = collections.deque()
        self.tail_len = 0
        self.total = 0
        self.lock = threading.Lock()

    def add(self, chunk: bytes) -> None:
        with self.lock:
            self.total += len(chunk)
            room = self.head_cap - len(self.head)
            if room > 0:
                self.head += chunk[:room]
                chunk = chunk[room:]
            if not chunk:
                return
            self.tail.append(chunk)
            self.tail_len += len(chunk)
            # Drop whole chunks from the front while what remains still covers the cap.
            while self.tail and self.tail_len - len(self.tail[0]) >= self.tail_cap:
                self.tail_len -= len(self.tail.popleft())

    def text(self) -> str:
        """The kept output, decoded, with newlines normalised as ``text=True`` did."""
        with self.lock:
            tail = b"".join(self.tail)
            if self.tail_len > self.tail_cap:
                tail = tail[self.tail_len - self.tail_cap:]
            dropped = self.total - len(self.head) - len(tail)
            parts = [bytes(self.head)]
        if dropped > 0:
            parts.append(f"\n[... {dropped} bytes of output not kept ...]\n".encode())
        parts.append(tail)
        decoded = b"".join(parts).decode("utf-8", errors="replace")
        return decoded.replace("\r\n", "\n").replace("\r", "\n")


def _drain(pipe: Any, output: _Output, activity: list[float]) -> None:
    """Read *pipe* to its end in chunks into *output*, noting when output last arrived."""
    try:
        fd = pipe.fileno()
    except (OSError, ValueError):
        return
    while True:
        try:
            chunk = os.read(fd, 65536)
        except (OSError, ValueError):
            return
        if not chunk:
            return
        activity[0] = time.monotonic()
        output.add(chunk)


# ── validation without execution ───────────────────────────────────────────
def missing_init(root: Path) -> str:
    """Why a package root has no `__init__.py`, and what it holds instead.

    A package emitted with project-relative paths nests inside itself, and the
    failure then reads as a missing file four directories away from the mistake. The
    message therefore names the root it looked in, what that root holds, and the
    convention the answer was supposed to follow.
    """
    found = sorted(entry.name for entry in root.iterdir())[:8]
    return (
        f"__init__.py is missing from the package root {root}; it holds "
        f"{', '.join(found) if found else 'nothing'}. Paths in an emit answer are "
        "relative to the package root, so `__init__.py` means exactly `__init__.py`."
    )


def ast_check(name: str, project: Path | None = None) -> list[str]:
    """Everything wrong with a package that can be seen without importing it."""
    project = Path(project or project_for_nodes())
    root = registry.where(name, project)
    problems: list[str] = []
    if root is None:
        return [f"no package {name!r} in either root"]
    files = sorted(root.glob("*.py"))
    if not (root / "__init__.py").exists():
        return [missing_init(root)]

    parsed: dict[str, ast.Module] = {}
    for path in files:
        try:
            parsed[path.name] = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            problems.append(f"{path.name} does not parse: line {exc.lineno}: {exc.msg}")
    if problems:
        return problems

    init = parsed["__init__.py"]
    packages = [
        node for node in init.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", "") == "PACKAGE" for t in node.targets)
    ]
    if not packages:
        problems.append("__init__.py does not assign PACKAGE at module level")
    else:
        text = ast.unparse(packages[0].value)
        if "Package(" not in text:
            problems.append("PACKAGE is not built by Package(...)")
        if f"name=" not in text.replace(" ", ""):
            problems.append("PACKAGE does not declare a name")
        if "modes=" not in text.replace(" ", ""):
            problems.append("PACKAGE declares no modes")

    mode_classes = {
        cls.name: cls
        for module in parsed.values()
        for cls in module.body
        if isinstance(cls, ast.ClassDef)
    }
    if not mode_classes:
        problems.append("no class is declared anywhere in the package")
    for class_name, cls in sorted(mode_classes.items()):
        methods = {
            node.name for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        bases = {ast.unparse(base) for base in cls.bases}
        declared = any("Mode" in base for base in bases)
        if declared and "build" not in methods:
            problems.append(f"class {class_name} looks like a Mode but declares no build()")
        if declared and "score" not in methods:
            problems.append(
                f"class {class_name} looks like a Mode but declares no score(), so it "
                "produces no signal"
            )

    for module in parsed.values():
        for node in ast.walk(module):
            if isinstance(node, ast.Call):
                called = ast.unparse(node.func)
                if called in ("eval", "exec", "compile", "__import__"):
                    problems.append(
                        f"the package calls {called}(); a generated package is not run "
                        "with that reach"
                    )
    return sorted(set(problems))


# ── execution under limits ─────────────────────────────────────────────────
def is_secret(name: str) -> bool:
    """Whether an environment name looks like a credential, which a sandboxed run never sees."""
    return any(marker in name.upper() for marker in _SECRET_MARKERS)


def scrub(env: dict[str, str] | None = None) -> dict[str, str]:
    """The environment a sandboxed run sees: an allowlist, with secrets dropped.

    The caller's PATH passes through the allowlist unchanged; the worker
    environment or check environment has already set it to the target's Python.
    """
    from refactory_dsh.paths import project_root

    source = dict(os.environ if env is None else env)
    out: dict[str, str] = {}
    for name in _ALLOWED_ENV:
        if name not in source:
            continue
        if is_secret(name):
            continue
        out[name] = source[name]
    return out


def _limits(memory_mb: int, cpu_s: int, file_mb: int, *,
            address_space: bool = True):  # pragma: no cover - posix only
    """A preexec that caps what the child can consume.

    ``memory_mb`` caps the heap (RLIMIT_DATA) and, unless ``address_space`` is false,
    the whole address space (RLIMIT_AS). Node reserves address space it never uses, so
    a child that starts Node keeps the heap cap and drops only the address-space one.

    ``cpu_s`` is the soft CPU limit, and the hard limit is one second above it: Linux
    sends SIGKILL, not SIGXCPU, when a process reaches a hard limit equal to its soft
    one, and a CPU limit ended by SIGKILL cannot be told from any other kill
    (`_ended_by`). macOS sends SIGXCPU either way.
    """
    import resource

    def apply() -> None:
        # Cap RLIMIT_CPU at the system's hard limit — a non-root process cannot
        # raise its hard limit, so setrlimit would silently fail.
        soft, ceiling = cpu_s, cpu_s + 1
        try:
            _soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
            if hard != resource.RLIM_INFINITY and ceiling > hard:
                ceiling = hard
                soft = min(cpu_s, max(hard - 1, 1))
        except (ValueError, OSError):
            pass
        caps = (
            (resource.RLIMIT_CPU, (soft, ceiling)),
            # SIGXCPU (and SIGXFSZ) end a process with a core dump by default; a sandboxed
            # child leaves none.
            (resource.RLIMIT_CORE, (0, 0)),
            (resource.RLIMIT_FSIZE, (file_mb * 1024 * 1024,) * 2),
        )
        for which, value in caps:
            try:
                resource.setrlimit(which, value)
            except (ValueError, OSError):
                pass
        for attribute, value in (
            *((("RLIMIT_AS", memory_mb * 1024 * 1024),) if address_space else ()),
            ("RLIMIT_DATA", memory_mb * 1024 * 1024),
        ):
            which = getattr(resource, attribute, None)
            if which is None or memory_mb <= 0:
                continue
            try:
                resource.setrlimit(which, (value, value))
            except (ValueError, OSError):
                pass
        try:
            os.setsid()
        except OSError:
            pass

    return apply


def _apply_forwarded(child_env: dict[str, str], forwarded: dict[str, str] | None) -> None:
    """Add the environment a declared tool needs, after containment is set.

    Applied last because it exists to *undo* one effect of containment: ``HOME`` is
    rewritten to the working directory, and a per-user toolchain manager -- elan,
    rustup, pyenv, nvm -- finds its installation through ``HOME``. Only a package's
    own declaration, read by the driver, reaches here.

    The secret filter still applies. A declaration is our own code, but the rule
    "no secret enters a sandbox" is not worth making an exception to for a tool,
    and a tool that needed a credential to *run* would be a different design.
    """
    for name, value in (forwarded or {}).items():
        if is_secret(name):
            continue
        child_env[name] = value


def _progress_at(workdir: Path) -> dict[str, Any]:
    """Read existing runtime progress without inventing a completion count."""
    progress: dict[str, Any] = {}
    for key, relative in (("runtime", ".refactory/progress.json"), ("item", "current_item.json")):
        try:
            value = json.loads((workdir / relative).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict):
            continue
        if key == "item":
            progress["current_item_id"] = value.get("id")
        else:
            progress["next_node"] = value.get("node")
            progress["recorded_steps"] = value.get("steps")
    return progress


def refactory_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """``scrub(env)`` with refactory's own interpreter first on PATH.

    For a check of an artifact built for refactory outside any worktree (create mode, a
    fast run with no target): its ``python`` must import ``refactory_dsh`` however the
    launching shell's PATH reads, and `scrub` does not add it. Its Python writes
    unbuffered (``PYTHONUNBUFFERED``), as a worktree check's does
    (`worktree.check_environment`): output held in a pipe's buffer looks like silence to
    `run_command`'s inactivity limit.
    """
    from refactory_dsh.paths import venv_bin

    out = scrub(env)
    bindir = str(venv_bin())
    rest = [entry for entry in out.get("PATH", "").split(os.pathsep) if entry and entry != bindir]
    out["PATH"] = os.pathsep.join([bindir, *rest])
    out["VIRTUAL_ENV"] = str(Path(bindir).parent)
    out["PYTHONUNBUFFERED"] = "1"
    return out


def run_sandboxed(
    code: str,
    *,
    workdir: Path,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    memory_mb: int = DEFAULT_MEMORY_MB,
    cpu_s: int = DEFAULT_CPU_S,
    file_mb: int = DEFAULT_FILE_MB,
    env: dict[str, str] | None = None,
    forwarded: dict[str, str] | None = None,
    address_space: bool = True,
    inactivity: bool = False,
    backstop_s: float | None = None,
    nested_watch: bool = False,
) -> SandboxResult:
    """Run a snippet in a subprocess with limits, from the workspace, scrubbed.

    A subprocess rather than a thread or a same-process guard, because the thing
    being contained is *code*: only a separate process can be killed, capped and
    given its own working directory. ``address_space=False`` keeps the heap cap and
    drops the address-space cap (`_limits`).

    ``timeout_s`` is a wall clock unless ``inactivity`` is set (a smoke's choice,
    `smoke`): the run is then stopped only after ``timeout_s`` seconds with no sign of
    work -- its own output, or the run's signs of work in ``workdir`` (`activity.signs`)
    -- or at ``backstop_s`` of wall time (`_run_watched`). ``nested_watch`` says the
    snippet runs a watched run of its own (a model smoke's `run_on_dsh`), which looks at
    those signs and marks what it sees in ``ACTIVITY_FILE``: the sandbox then reads only
    that mark, so the run is looked at in one place.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    child_env = scrub(env)
    child_env["HOME"] = str(workdir)
    child_env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    _apply_forwarded(child_env, forwarded)
    kwargs: dict[str, Any] = {}
    if sys.platform != "win32":
        kwargs["preexec_fn"] = _limits(memory_mb, cpu_s, file_mb, address_space=address_space)
    if inactivity:
        return _run_watched(
            [sys.executable, "-c", code], workdir=workdir, env=child_env, timeout_s=timeout_s,
            backstop_s=backstop_s if backstop_s is not None else min(4 * timeout_s, COMMAND_BACKSTOP_S),
            nested_watch=nested_watch, cpu_s=cpu_s,
            **kwargs,
        )
    started = time.monotonic()
    try:
        if child_env.get("REFACTORY_BUDGET_LEDGER"):
            from refactory_dsh.owned_process import run_owned

            result = run_owned(
                [sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, cwd=str(workdir), env=child_env, timeout=timeout_s,
                label="budgeted validation", **kwargs,
            )
        else:
            result = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True,
                cwd=str(workdir), env=child_env, timeout=timeout_s, **kwargs,
            )
    except subprocess.TimeoutExpired as exc:
        return SandboxResult(
            ok=False,
            timed_out=True,
            returncode=124,
            timeout_s=timeout_s,
            elapsed_s=round(time.monotonic() - started, 3),
            progress=_progress_at(workdir),
            stdout=(exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
            stderr=f"timed out after {timeout_s}s",
        )
    payload: dict[str, Any] = {}
    for line in reversed(result.stdout.strip().splitlines()):
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            payload = loaded
            break
    return SandboxResult(
        ok=result.returncode == 0,
        timeout_s=timeout_s,
        elapsed_s=round(time.monotonic() - started, 3),
        payload=payload,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def _signs_of_work(workdir: Path, env: dict[str, str], *, nested_watch: bool) -> Any:
    """What `_run_watched` compares: the mark a nested watched run leaves in
    ``ACTIVITY_FILE``, or, with none, the run's own signs of work (`activity.signs`);
    never the run's budget ledger."""
    return _signs_watch(workdir, env, nested_watch=nested_watch)()


def _signs_watch(workdir: Path, env: dict[str, str], *, nested_watch: bool) -> Any:
    """A look at `_signs_of_work`, with what it reads resolved once."""
    from refactory_dsh import activity

    if nested_watch:
        mark = workdir / ACTIVITY_FILE

        def _mark() -> int | None:
            try:
                return mark.stat().st_mtime_ns
            except OSError:
                return None

        return _mark
    return activity.Watch(workdir, ignore=(env.get("REFACTORY_BUDGET_LEDGER"),)).look


def mark_activity(path: Path) -> None:
    """Leave the mark `_run_watched` reads as work: the time, in ``path``.

    For a run nested inside a watched sandbox (a smoke's DSH run,
    `dsh_runtime.run_on_dsh`), whose output its own process keeps.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(time.time_ns()))
    except OSError:
        pass


def _stop_watched(proc: subprocess.Popen, workdir: Path) -> bool:
    """Ask a watched run to stop and give it ``STOP_GRACE_S`` to do so; then kill its group.

    Only the direct child is asked, as `owned_process.shutdown` asks: a smoke's process
    shuts its own DSH run down in that window. Whatever is left in its process group is
    killed afterwards either way. Returns whether the child had to be killed.
    """
    try:
        proc.terminate()
    except ProcessLookupError:
        pass
    forced = False
    try:
        proc.wait(timeout=STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        forced = True
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    elif forced:
        proc.kill()
    _end_left_behind(proc, workdir, own_group=False)
    if forced:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    return forced


def _end_left_behind(proc: subprocess.Popen, workdir: Path, *, own_group: bool = True) -> None:
    """End what a watched run left running once its process is gone, however it went:
    what is left in its own process group, and the groups its nested run recorded --
    the DSH child and each node command (`activity.end_recorded`), which a process that
    died (a CPU limit, a kill) could not end itself."""
    from refactory_dsh import activity

    if own_group and os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    activity.end_recorded(activity.watch_dir(workdir))


def _ended_by(returncode: int | None, cpu_s: int | None) -> str:
    """Why a watched run's process ended, when a signal ended it, for the person."""
    if returncode is None or returncode >= 0:
        return ""
    number = -returncode
    if number == getattr(signal, "SIGXCPU", None):
        limit = f" ({cpu_s} s of CPU time per process)" if cpu_s else ""
        return f"the smoke hit its CPU limit{limit}"
    try:
        name = signal.Signals(number).name
    except ValueError:
        name = f"signal {number}"
    return f"the smoke's process was ended by {name}"


def _run_watched(
    argv: list[str],
    *,
    workdir: Path,
    env: dict[str, str],
    timeout_s: int,
    backstop_s: float,
    nested_watch: bool = False,
    cpu_s: int | None = None,
    **kwargs: Any,
) -> SandboxResult:
    """`run_sandboxed` with an inactivity limit and a wall-clock backstop.

    Work is any output, or a change in the run's signs of work (`_signs_of_work`). A run
    stopped for either limit is asked to stop and given ``STOP_GRACE_S`` before it is
    killed (`_stop_watched`); a kill says that its cleanup is unconfirmed, as
    `owned_process.shutdown` does. A run that stopped itself for one of these limits
    reports it in its payload's ``timeout_kind``, which is returned as the result's. A
    run whose process ended any other way -- its CPU limit, a signal, a crash -- has
    what it left running ended the same way (`_end_left_behind`), and a signal says
    which (`_ended_by`).
    """
    from refactory_dsh import activity as activity_module
    from refactory_dsh.owned_process import _termination_as_exit

    started = time.monotonic()
    out = _Output(OUTPUT_KEEP_BYTES, OUTPUT_TAIL_BYTES)
    err = _Output(OUTPUT_KEEP_BYTES, OUTPUT_TAIL_BYTES)
    activity = [started]
    with _termination_as_exit():
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(workdir), env=env,
            start_new_session=os.name == "posix", **kwargs,
        )
        readers = [threading.Thread(target=_drain, args=(pipe, sink, activity), daemon=True,
                                    name="sandbox-drain")
                   for pipe, sink in ((proc.stdout, out), (proc.stderr, err))]
        for reader in readers:
            reader.start()
        look = _signs_watch(workdir, env, nested_watch=nested_watch)
        interval = activity_module.look_interval(timeout_s, WATCH_INTERVAL_S)
        seen = look()
        look_at = started + WATCH_INTERVAL_S
        stop: str | None = None
        try:
            while proc.poll() is None:
                now = time.monotonic()
                if now >= look_at:
                    current = look()
                    if current != seen:
                        seen, activity[0] = current, now
                    look_at = now + interval
                if now - activity[0] >= timeout_s:
                    stop = "inactivity"
                elif now - started >= backstop_s:
                    stop = "backstop"
                if stop is not None:
                    break
                time.sleep(0.05)
        except BaseException:
            _stop_watched(proc, workdir)
            for reader in readers:
                reader.join(timeout=5)
            raise
        if stop is not None:
            forced = _stop_watched(proc, workdir)
            for reader in readers:
                reader.join(timeout=5)
            note = f"timed out ({stop}) after {timeout_s}s"
            if forced:
                note += (f"; the run did not stop within {STOP_GRACE_S}s of being asked; "
                         "forced exit; child cleanup unconfirmed")
            return SandboxResult(
                ok=False, timed_out=True, returncode=124, timeout_s=timeout_s, timeout_kind=stop,
                elapsed_s=round(time.monotonic() - started, 3), progress=_progress_at(workdir),
                stdout=out.text(), stderr="\n".join(part for part in (err.text().rstrip(), note) if part),
            )
        _end_left_behind(proc, workdir)
        grace_end = time.monotonic() + EXIT_GRACE_S
        for reader in readers:
            reader.join(timeout=max(0.0, grace_end - time.monotonic()))
    stdout, stderr = out.text(), err.text()
    ended = _ended_by(proc.returncode, cpu_s)
    if ended:
        stderr = "\n".join(part for part in (stderr.rstrip(), ended) if part)
    payload: dict[str, Any] = {}
    for line in reversed(stdout.strip().splitlines()):
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            payload = loaded
            break
    inner = payload.get("timeout_kind")
    return SandboxResult(
        ok=proc.returncode == 0,
        timed_out=inner in STOP_KINDS,
        timeout_kind=inner if inner in STOP_KINDS else None,
        timeout_s=timeout_s,
        elapsed_s=round(time.monotonic() - started, 3),
        payload=payload,
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def run_command(
    command: str,
    *,
    workdir: Path,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    memory_mb: int = 0,
    cpu_s: int | None = None,
    file_mb: int = DEFAULT_FILE_MB,
    env: dict[str, str] | None = None,
    budget: Any = None,
    home: Path | None = None,
    exact_env: bool = False,
    stdin: Any = None,
    worktree_root: Path | None = None,
) -> SandboxResult:
    """Run a shell command in the sandbox, bounded by inactivity rather than memory.

    A command is what a brief declares as its check ("run the tests", "build it"),
    and it is arbitrary, so it gets the same scrubbed environment and file limit as
    anything else we did not write. It is stopped when it has written nothing to stdout
    or stderr for ``timeout_s`` seconds, at the ``COMMAND_BACKSTOP_S`` wall clock however
    much it writes, or once its output passes ``OUTPUT_LIMIT_BYTES``; the result says
    which (``timeout_kind``) and keeps its output (`_Output`). It has no memory cap
    unless ``memory_mb`` sets one, because V8 reserves far more address space than it
    uses, so Node fails under any practical address-space limit once it starts worker
    threads, as vitest does (#325).
    Its CPU limit, unless ``cpu_s`` sets one, is the backstop on every core, so a test
    runner's worker threads are not stopped before it. When it is stopped, everything
    the command started is stopped with it, not only its shell; so is what it left
    running in the background that still holds its output ``EXIT_GRACE_S`` after it
    exits. ``exact_env``
    runs it with ``env`` as given instead -- a design run's worktree builds its own,
    without refactory's interpreter or variables (`worktree.check_environment`) -- and
    ``home`` moves HOME out of ``workdir``, so tool caches stay out of what is checked.
    ``stdin`` is the command's standard input, empty unless given: a check runs
    unattended, so nothing it runs can read or wait on the launcher's stdin.
    ``worktree_root``, when given, enables cwd-escape detection: an EXIT trap compares
    the command's final working directory with the worktree root (both realpath'd, prefix
    check) and records the directory in ``SandboxResult.cwd_outside`` when it has left.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    child_env = dict(env or {}) if exact_env else scrub(env)
    child_env["HOME"] = str(home or workdir)
    kwargs: dict[str, Any] = {}
    if sys.platform != "win32":
        # RLIMIT_CPU is based on the backstop, not on the inactivity limit, so a
        # long-running active command is not killed by SIGXCPU (PIT-001).
        cpu = cpu_s if cpu_s is not None else math.ceil(COMMAND_BACKSTOP_S) * (os.cpu_count() or 1)
        kwargs["preexec_fn"] = _limits(memory_mb, cpu, file_mb)
    # When the caller wants cwd-escape detection, wrap the command in a shell that
    # installs an EXIT trap: the trap writes the final $PWD to a sentinel file so
    # the sentinel never reaches stdout or stderr.  The command sits on its own
    # line, so comments, `exit` and multi-line commands all work, and the trap
    # fires after `exit` too (#351 I3).
    sentinel_path: Path | None = None
    if worktree_root is not None:
        sentinel_path = workdir / _CWD_SENTINEL_FILE
        sentinel_path.unlink(missing_ok=True)
        wt_real = str(Path(worktree_root).resolve())
        wrapped = (
            f"__refactory_wt_root='{wt_real}'\n"
            f"__refactory_sentinel='{sentinel_path}'\n"
            "trap '__refactory_final=\"$(cd -P . 2>/dev/null && pwd -P)\"; "
            "case \"$__refactory_final\" in \"$__refactory_wt_root\"/*|\"$__refactory_wt_root\") ;; "
            "*) printf \"%s\" \"$__refactory_final\" > \"$__refactory_sentinel\" ;; "
            "esac' EXIT\n"
            f"{command}\n"
        )
    else:
        wrapped = command
    # text=False: reader threads take raw bytes from the pipes (`_drain`), decoded at the end.
    options = dict(shell=True, text=False, cwd=str(workdir), env=child_env,
                   stdin=subprocess.DEVNULL if stdin is None else stdin, **kwargs)

    def _stop_session(proc: subprocess.Popen[bytes], readers: list[threading.Thread]) -> None:
        """Stop everything the command started, then let the readers drain to the end.

        This session was created by this call and contains only this command's shell and
        tools. No descendant discovery or name matching.
        """
        try:
            os.killpg(proc.pid, signal.SIGTERM) if os.name == "posix" else proc.terminate()
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        try:
            # Whatever ignored SIGTERM, or outlived its shell, still holds the pipes.
            os.killpg(proc.pid, signal.SIGKILL) if os.name == "posix" else proc.kill()
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait(timeout=2)
        for reader in readers:
            reader.join(timeout=5)

    def _once() -> tuple[int | None, str, str, str | None, int]:
        """Run the command once: (returncode, stdout, stderr, stop kind, output bytes).

        It is stopped when it has been silent for ``timeout_s``, at the backstop, or once
        its output passes ``OUTPUT_LIMIT_BYTES``; the stop kind says which, with the
        returncode None. A budget stop raises. Each call has its own output and reader
        threads, and joins them before it returns, so a retry shares nothing with it.
        """
        backstop, limit = COMMAND_BACKSTOP_S, OUTPUT_LIMIT_BYTES
        out = _Output(OUTPUT_KEEP_BYTES, OUTPUT_TAIL_BYTES)
        err = _Output(OUTPUT_KEEP_BYTES, OUTPUT_TAIL_BYTES)
        began = time.monotonic()
        proc = subprocess.Popen(wrapped, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                start_new_session=os.name == "posix", **options)
        # activity[0]: when output last arrived (a float assignment, atomic under the GIL).
        activity = [began]
        readers = [threading.Thread(target=_drain, args=(pipe, sink, activity), daemon=True,
                                    name="sandbox-drain")
                   for pipe, sink in ((proc.stdout, out), (proc.stderr, err))]
        for reader in readers:
            reader.start()
        stop: str | None = None
        try:
            while proc.poll() is None:
                if budget is not None:
                    budget.checkpoint()
                now = time.monotonic()
                if out.total + err.total > limit:
                    stop = "output_limit"
                elif now - activity[0] >= timeout_s:
                    stop = "inactivity"
                elif now - began >= backstop:
                    stop = "backstop"
                if stop is not None:
                    break
                time.sleep(0.05)
        except BaseException:
            _stop_session(proc, readers)
            raise
        if stop is not None:
            _stop_session(proc, readers)
            return None, out.text(), err.text(), stop, out.total + err.total
        # The command's own process exited. What it started in the background may still
        # hold the pipes: it gets a short grace to finish, then is stopped as at a timeout,
        # so nothing the command started outlives it.
        grace_end = time.monotonic() + EXIT_GRACE_S
        for reader in readers:
            reader.join(timeout=max(0.0, grace_end - time.monotonic()))
        if any(reader.is_alive() for reader in readers):
            _stop_session(proc, readers)
        return proc.returncode, out.text(), err.text(), None, out.total + err.total

    started = time.monotonic()

    def _stopped(stdout: str, stderr: str, kind: str, total: int) -> SandboxResult:
        if sentinel_path is not None:
            sentinel_path.unlink(missing_ok=True)
        return SandboxResult(
            ok=False, timed_out=True, returncode=124, timeout_s=timeout_s,
            elapsed_s=round(time.monotonic() - started, 3), progress=_progress_at(workdir),
            stdout=stdout, stderr=stderr or f"timed out after {timeout_s}s",
            timeout_kind=kind, output_bytes=total,
        )

    if budget is not None:
        budget.checkpoint()
    returncode, stdout, stderr, kind, total = _once()
    if kind is not None:
        return _stopped(stdout, stderr, kind, total)
    assert returncode is not None
    if returncode < 0:
        # Killed by a signal — retry once; the first death is transient.
        returncode, stdout, stderr, kind, total = _once()
        if kind is not None:
            return _stopped(stdout, stderr, kind, total)
        assert returncode is not None
        if returncode < 0:
            sig_num = -returncode
            try:
                sig_name = signal.Signals(sig_num).name
            except (ValueError, KeyError):
                sig_name = f"signal {sig_num}"
            sig_label = f"killed by {sig_name}"
            if sentinel_path is not None:
                sentinel_path.unlink(missing_ok=True)
            return SandboxResult(
                ok=False,
                timeout_s=timeout_s,
                elapsed_s=round(time.monotonic() - started, 3),
                returncode=returncode,
                stdout=stdout,
                stderr=f"{sig_label}\n{stderr}" if stderr else sig_label,
                output_bytes=total,
            )
    elif returncode in _AMBIGUOUS_SHELL_EXITS:
        # Shell-wrapped SIGABRT/SIGBUS/SIGSEGV — retry once (#377, extended to
        # 128+N shells).  If the retry still crashes, the result falls through to
        # the caller's base comparison, which decides whether the crash is the
        # change's fault or infrastructure.
        returncode, stdout, stderr, kind, total = _once()
        if kind is not None:
            return _stopped(stdout, stderr, kind, total)
        assert returncode is not None
    cwd_outside: str | None = None
    if sentinel_path is not None:
        try:
            cwd_outside = sentinel_path.read_text().strip() or None
        except OSError:
            pass
        finally:
            sentinel_path.unlink(missing_ok=True)
    return SandboxResult(
        ok=returncode == 0,
        timeout_s=timeout_s,
        elapsed_s=round(time.monotonic() - started, 3),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        cwd_outside=cwd_outside,
        output_bytes=total,
    )


def model_readiness(
    workdir: Path, *, model_context: dict[str, str] | None = None, require_ready: bool = True,
) -> dict[str, Any]:
    """Check the model path after the same scrub/HOME rewrite as real smoke.

    No package is imported and no model is called. Presence is not proof that the
    provider will accept the key; the later real smoke remains mandatory.
    Diagnostic callers may set require_ready=False to receive a known local
    refusal as data. Authoring keeps the default that raises before research.
    """
    from refactory_dsh.model import DshModelClient

    context = model_context if model_context is not None else DshModelClient().context()
    code = (
        "import json\nfrom pathlib import Path\n"
        "from refactory_dsh.model import DshModelClient\n"
        f"print(json.dumps(DshModelClient(**{context!r}).readiness(workdir=Path.cwd(), require_ready={require_ready!r})))\n"
    )
    result = run_sandboxed(code, workdir=workdir, timeout_s=40)
    if not result.ok or not (
        result.payload.get("ready") is True
        or (result.payload.get("ready") is None and result.payload.get("status") == "diagnostic-unavailable")
        or (not require_ready and result.payload.get("status") in ("missing-credential", "invalid-configuration"))
    ):
        raise SandboxError(result.problem() or "sandbox model configuration is not ready; run refactory doctor")
    return result.payload


def smoke_backstop_s(timeout_s: float) -> float:
    """The wall-clock backstop of a smoke whose inactivity limit is ``timeout_s``.

    The backstop a command gets (four times its limit, at most ``COMMAND_BACKSTOP_S``),
    plus ``STOP_GRACE_S``, so a smoke with a model reaches its own run's backstop, stops
    and reports first.
    """
    return min(4 * timeout_s, COMMAND_BACKSTOP_S) + STOP_GRACE_S


def smoke_wall_s(timeout_s: float) -> float:
    """The longest one smoke can take: its backstop, then the stop's grace and kill.

    What a node that runs smokes covers per run (`lowering._smoke_envelope_s`), so the
    node's own timeout does not fire before the smoke's stop is reported.
    """
    return smoke_backstop_s(timeout_s) + STOP_GRACE_S + 5


class SampleRefused(ValueError):
    """A smoke's sample cannot be drawn: a Task-bound node has no train item to run."""


def _some(ids: Any, shown: int = 10) -> str:
    listed = sorted(str(item) for item in ids)
    more = f", and {len(listed) - shown} more" if len(listed) > shown else ""
    return ", ".join(listed[:shown]) + more


def smoke_sample(mode: Any, workdir: Path, items: int, *, holdout: float | None = None,
                 parts_manifest: Path | str | None = None) -> dict[str, Any]:
    """Cut ``mode`` down to a smoke's sample: ``items`` items of each Task's train part.

    The train part is the one a real run of the mode uses: under ``parts_manifest``
    (a ``parts.json`` with train/val/test id lists) its ``train`` ids, otherwise the
    ``holdout`` fraction's split (`task.partition`; by default the outer loop's own,
    ``task.DEFAULT_HOLDOUT``). Of that part the sample takes the ``items`` lowest ids --
    the split holds out the highest ids, so these stay train under any larger holdout
    too while the part has room. The run directory names them: all nodes' samples as
    its ``parts.json`` train part (`task.write_subset`), so no held-out item can be run,
    and each node's own in ``SMOKE_SAMPLE``, which only a smoke writes, so each node
    runs only its sample (`nn.executor.load_items`). A part with fewer items runs them
    all. Each such node runs its sample at once. A
    node whose parallelism is a knob (``parallelism_knob``, or a ``parallelism`` field
    binding) gets that knob set, held to the knob's own value as a ceiling, since a node
    may not carry both; any other node gets ``parallelism``, held to the value it
    declares when that is above one. Nothing changes for ``items`` 0 or a mode with no
    Task-bound data node.

    Runs in the smoke's sandboxed process, before the run. What it sets is state and the
    run directory's part, which the walker and DSH read the same way
    (`nn.executor.load_items`, `Node.effective_parallelism`).
    A node that would run nothing is `SampleRefused`, naming why: under a manifest, none
    of its train ids is one of the node's items; otherwise the node has no train item. A
    manifest some of whose train ids the node lacks is sampled from the rest, with a
    warning naming the missing ids.
    Returns ``{"item_limit": items, "nodes": [...], "items": [...], "per_node": {...},
    "warnings": [...]}`` for the nodes cut, or ``{}``.
    """
    from refactory_dsh import task as task_module
    from refactory_dsh.nn.edits import configure
    from refactory_dsh.nn.executor import SMOKE_SAMPLE, full_items
    from refactory_dsh.nn.graph import MAX_PARALLELISM, Node

    if items <= 0:
        return {}
    fraction = task_module.DEFAULT_HOLDOUT if holdout is None else float(holdout)
    manifest_train: set[str] | None = None
    if parts_manifest:
        manifest = json.loads(Path(parts_manifest).read_text())
        manifest_train = set(task_module.manifest_part_ids(manifest, task_module.TRAIN) or [])
    state = mode.state_dict()
    nodes = (state.get("graph") or {}).get("nodes") or {}
    knob_widths: dict[str, int] = {}
    sampled: list[str] = []
    chosen: list[str] = []
    per_node: dict[str, list[str]] = {}
    warnings: list[str] = []
    # The part, before any manifest or sample of the smoke's own is written, so a Task
    # whose items depend on it sees it (`full_items`).
    (Path(workdir) / task_module.MANIFEST).unlink(missing_ok=True)
    (Path(workdir) / SMOKE_SAMPLE).unlink(missing_ok=True)
    task_module.write_subset(workdir, task_module.TRAIN, fraction)
    for node_id, entry in nodes.items():
        node = Node.from_state({**entry, "id": node_id})
        if node.kind != "data" or not node.task_ref:
            continue
        every = full_items(node, Path(workdir))
        keyed = [(task_module._item_key(item, index), item) for index, item in enumerate(every)]  # noqa: SLF001
        if manifest_train is not None:
            train = [key for key, _item in keyed if key in manifest_train]
            missing = manifest_train - {key for key, _item in keyed}
            if not train:
                raise SampleRefused(
                    f"data node {node_id}: none of the parts manifest's {len(manifest_train)} "
                    f"train ids is one of its {len(keyed)} items ({_some(manifest_train)}); "
                    "the manifest is not this Task's")
            if missing:
                warnings.append(
                    f"data node {node_id}: {len(missing)} of the parts manifest's "
                    f"{len(manifest_train)} train ids are not among its items ({_some(missing)})")
        else:
            held = {id(item) for item in task_module.partition(every, fraction)[task_module.HOLDOUT]}
            train = [key for key, item in keyed if id(item) not in held]
            if not train:
                raise SampleRefused(f"data node {node_id} has no train item to run "
                                    f"({len(keyed)} items, holdout {fraction})")
        per_node[node_id] = sorted(train)[:items]
        chosen.extend(key for key in per_node[node_id] if key not in chosen)
        knob = node.parallelism_knob or node.bound_knob("parallelism")
        if knob:
            entry_of_knob = (state.get("knobs") or {}).get(knob, {})
            ceiling = int(entry_of_knob.get("value") or 0)
            width = min(items, ceiling) if ceiling > 0 else items
            floor = (entry_of_knob.get("spec") or {}).get("lo")
            # Never below the knob's own floor, which `configure` refuses: a floor above
            # the sample leaves the width there, and the run still has only ``items``
            # items to run at once.
            width = max(width, int(floor)) if floor is not None else width
            knob_widths[knob] = min(knob_widths.get(knob, width), width)
        else:
            declared = node.parallelism if node.parallelism > 1 else MAX_PARALLELISM
            entry["parallelism"] = min(items, declared)
        sampled.append(node_id)
    if not sampled:
        task_module._subset_file(Path(workdir)).unlink(missing_ok=True)  # noqa: SLF001
        return {}
    for knob, width in knob_widths.items():
        state = configure(state, knob, max(1, min(width, MAX_PARALLELISM)))
    mode.load_state_dict(state)
    # The part names the union, so the run's selection holds every node's sample; each
    # node then runs only its own (`nn.executor.load_items`).
    task_module.make_manifest(Path(workdir), train_ids=chosen)
    sample_file = Path(workdir) / SMOKE_SAMPLE
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    sample_file.write_text(json.dumps(per_node, indent=2))
    return {"item_limit": items, "nodes": sampled, "items": chosen, "per_node": per_node,
            "warnings": warnings}


def smoke(
    identifier: str,
    workdir: Path,
    *,
    project: Path | None = None,
    model: bool = False,
    knobs: dict[str, Any] | None = None,
    model_context: dict[str, str] | None = None,
    smoke_items: int = 0,
    holdout: float | None = None,
    parts_manifest: Path | str | None = None,
    **kwargs: Any,
) -> SandboxResult:
    """Run one mode once, in a sandbox, and report what it scored.

    Model-backed smoke runs use the production DSH graph executor, including its
    worker input checks. Deterministic smoke runs keep the model-free walker.

    The package's declared tools have their environment forwarded, so a mode whose
    checker is a per-user toolchain finds it here exactly as it does unsandboxed.
    Without that, this path and the walker disagree about the same mode on the same
    machine -- which is how the Lean mode came to score 1.0 one way and 0.0 the
    other.

    ``model`` supplies a model client. Almost every mode worth shipping has an
    `agent` node, and a smoke without a model cannot score one at all -- it would
    report "not measured" for the common case, which makes the check vacuous exactly
    where it matters. The client reaches a model the same way every other path does
    (`DshModelClient`, which runs the one-shot DSH profile from the checkout), so the
    contained process receives only the trusted model context. DSH reads the managed
    credential store itself; it does not materialize a key into the environment.
    This local same-user store is not a filesystem secret-isolation boundary.

    ``knobs`` overrides the mode's declared defaults for this run. Without it a smoke
    exercises the *default* path only, so a mode whose new input path is broken --
    the one the brief asked for -- scores on the fixture and is accepted.

    ``smoke_items`` cuts each data node bound to a Task down to at most that many items
    of the Task's train part, run at once (`smoke_sample`), with a model or without;
    never a held-out item. The train part is the one the mode's real runs use: the
    ``holdout`` fraction's, or the train ids of ``parts_manifest``; by default the outer
    loop's default split. 0 runs every item.

    ``timeout_s`` is an inactivity limit here, not a wall clock: a smoke that is still
    working is not stopped before the backstop (`smoke_backstop_s`). Work is output, or
    the run's signs of work (`activity.signs`): the run's state, every item's pass,
    every worker's activity, and each node's output and declared writes. A run with a
    model gets a slightly shorter limit of its own, so it stops itself first and says
    why; it is the one that looks at those signs, and it marks what it sees in
    ``ACTIVITY_FILE`` for this sandbox to see.

    Its CPU limit, unless ``cpu_s`` sets one, is the backstop on every core, as a
    command's is (`run_command`, PIT-001), and every process of the smoke inherits it.
    The inactivity limit and the backstop are what stop a runaway smoke; the CPU limit
    no longer ends a working one before the backstop, and only catches a run that keeps
    every core busy for the whole of it.
    """
    project = Path(project or project_for_nodes())
    client = (
        "from refactory_dsh.model import DshModelClient\n"
        "from refactory_dsh.dsh_runtime import evaluate_on_dsh\n"
        "import os\n"
        f"os.environ['REFACTORY_PROJECT'] = {str(project.resolve())!r}\n"
        if model
        else ""
    )
    from refactory_dsh.model import DshModelClient

    # Bind the trusted parent's home/route before HOME becomes the smoke directory.
    # The generated package still receives no credential environment variables.
    context = model_context if model_context is not None else DshModelClient().context()
    call = f"model=DshModelClient(**{context!r})" if model else ""
    evaluation = (
        f"evaluate_on_dsh(mode, mode.state_dict(), Path({str(Path(workdir))!r}), "
        f"timeout_s=smoke_timeout_s, inactivity=True, "
        f"activity_file=Path({str(Path(workdir) / ACTIVITY_FILE)!r}), {call})"
        if model else f"mode.evaluate(Path({str(Path(workdir))!r}))"
    )
    # The walker's own node writes mark the run as working (`activity.mark`); a run with
    # a model names its watch directory to its node commands itself.
    watch = (
        ""
        if model else
        "import os\nfrom refactory_dsh import activity\n"
        f"os.environ[activity.WATCH_ENV] = str(activity.watch_dir(Path({str(Path(workdir))!r})))\n"
    )
    code = (
        "import json\n"
        "from pathlib import Path\n"
        "from refactory_dsh import registry\n"
        "from refactory_dsh.nn.edits import configure\n"
        f"{client}"
        f"package, mode_class = registry.resolve(Path({str(project)!r}), {identifier!r})\n"
        "mode = mode_class()\n"
        f"overrides = {dict(knobs or {})!r}\n"
        "if overrides:\n"
        "    state = mode.state_dict()\n"
        "    for name, value in overrides.items():\n"
        "        state = configure(state, name, value)\n"
        "    mode.load_state_dict(state)\n"
        "from refactory_dsh.sandbox import SampleRefused, smoke_sample\n"
        "try:\n"
        f"    sample = smoke_sample(mode, Path({str(Path(workdir))!r}), {int(smoke_items)!r}, "
        f"holdout={holdout!r}, parts_manifest={str(parts_manifest) if parts_manifest else None!r})\n"
        # Refused before the run: said as the result, so the caller stops rather than
        # repairing code that has nothing wrong with it.
        "except SampleRefused as refused:\n"
        "    print(json.dumps({'exit_ok': False, 'error': str(refused), 'sample_refused': True}))\n"
        "    raise SystemExit(0)\n"
        f"{watch}"
        f"result = {evaluation}\n"
        # The knob values the run used, so a comparison can say what differed (dm-263).
        # Reported after the run, so a mode that cannot report them still passes.
        "try:\n"
        "    knobs = {name: entry.get('value') for name, entry in"
        " (mode.state_dict().get('knobs') or {}).items() if isinstance(entry, dict)}\n"
        "except Exception:\n"
        "    knobs = None\n"
        "print(json.dumps({'exit_ok': result.exit_ok, 'error': result.error,"
        " 'score': result.score, 'iterations': result.iterations, 'knobs': knobs,"
        " 'timeout_kind': getattr(result, 'timeout_kind', None),"
        " 'item_limit': sample.get('item_limit'), 'sample': sample}, default=str))\n"
    )
    from refactory_dsh import toolchain

    forwarded = toolchain.tool_env(identifier.split("/", 1)[0], project)
    kwargs.setdefault("timeout_s", DEFAULT_MODEL_SMOKE_TIMEOUT_S if model else DEFAULT_TIMEOUT_S)
    if model:
        # A model call starts a DSH worker, which is Node, and Node's WebAssembly
        # (fetch's HTTP parser among it) reserves far more address space than it uses.
        # Under the snippet's address-space cap that fails with
        # "WebAssembly.instantiate(): Out of memory" on Linux (dm-256), so a smoke with a
        # model drops only that cap. The heap cap stays, so the mode's own code is
        # still bounded; a smoke without a model keeps both.
        kwargs.setdefault("address_space", False)
    if context.get("budget_ledger"):
        from refactory_dsh.budgets import RunBudget

        budget = RunBudget.from_path(context["budget_ledger"])
        budget.checkpoint()
        remaining = budget.remaining_s()
        if remaining is not None:
            kwargs["timeout_s"] = min(kwargs["timeout_s"], remaining)
            kwargs.setdefault("backstop_s", min(smoke_backstop_s(kwargs["timeout_s"]), remaining))
        forwarded["REFACTORY_BUDGET_LEDGER"] = str(budget.path)
    if model:
        # Leave time for the inner runtime to shut down before sandbox termination.
        timeout = kwargs["timeout_s"]
        code = f"smoke_timeout_s={max(1, int(timeout - min(5, timeout / 2)))!r}\n" + code
    kwargs.setdefault("backstop_s", smoke_backstop_s(kwargs["timeout_s"]))
    # The CPU limit every process of the smoke inherits -- its own, the DSH child's, each
    # node command's -- is the backstop on every core, as a command's is (`run_command`,
    # PIT-001): a run that is still working reaches its backstop before it could use that
    # much, so the limit no longer ends one early, as `DEFAULT_CPU_S` did once a smoke
    # could run for hours. What is past it is runaway.
    kwargs.setdefault("cpu_s", math.ceil(kwargs["backstop_s"]) * (os.cpu_count() or 1))
    return run_sandboxed(
        code, workdir=Path(workdir), env=dict(os.environ), forwarded=forwarded,
        inactivity=True, nested_watch=model, **kwargs,
    )


def verify_sandboxed(
    name: str,
    *,
    project: Path | None = None,
    shared_project: Path | None = None,
    requires_data_node: bool = False,
    requires_task_ref: str = "",
    requires_task_refs: tuple[str, ...] = (),
    requires_input: tuple[str, ...] = (),
    **kwargs: Any,
) -> SandboxResult:
    """Everything wrong with a package, checked without importing it here.

    The check needs the package loaded, and loading **is** execution. So the whole
    check runs in a subprocess: the package name, its ops modules, every mode's
    construction, its score and its knobs, and every validator rule. This is the
    only place a generated package is loaded, and it is loaded somewhere that
    cannot reach the driver's filesystem, environment, or imports.

    ``shared_project``, when given alongside ``project``, puts the shared
    workspace's packages on the import path behind the staging root so that a
    staged package can import from other workspace packages (e.g.
    ``from refactory_packages.base_ws import ops``).

    Answers in one call what would otherwise be three: there is no read-only way
    to ask whether model-written code is a usable mode.
    """
    project = Path(project or project_for_nodes())
    required_tasks = tuple(dict.fromkeys(
        ref for ref in (*requires_task_refs, requires_task_ref) if ref
    ))
    _shared_str = str(shared_project) if shared_project else ""
    code = (
        "import json\n"
        "import importlib\n"
        "from pathlib import Path\n"
        "from refactory_dsh import registry\n"
        "from refactory_dsh.nn import validate_state\n"
        # When a shared project is given alongside a staging project, install
        # the shared workspace first and the staging workspace at
        # position 0 so the staged package wins for its own name while every
        # other workspace package still resolves.
        f"_shared = {_shared_str!r}\n"
        "if _shared:\n"
        f"    registry.install_workspace(Path(_shared), project_override=Path({str(project)!r}))\n"
        "problems = []\n"
        f"requires_data_node = {bool(requires_data_node)!r}\n"
        f"required_tasks = {required_tasks!r}\n"
        f"required_inputs = {tuple(requires_input)!r}\n"
        f"package = registry.load({name!r}, Path({str(project)!r}))\n"
        f"expected = {name!r}\n"
        # Concatenation, not a nested f-string: `{name!r}` inside one would insert
        # the quotes too and break the enclosing literal.
        "if package.name != expected:\n"
        "    problems.append('PACKAGE.name is ' + repr(package.name) + ', not ' + repr(expected))\n"
        "if not package.modes:\n"
        "    problems.append('PACKAGE.modes is empty; a package with no mode cannot be run')\n"
        "for ops_module in package.ops:\n"
        "    try:\n"
        "        importlib.import_module(ops_module)\n"
        "    except Exception as exc:\n"
        "        problems.append(f'the declared ops module {ops_module!r} does not import:"
        " {type(exc).__name__}: {exc}')\n"
        "for mode_class in package.modes:\n"
        "    identifier = f'{package.name}/{mode_class.name}'\n"
        "    try:\n"
        "        mode = mode_class()\n"
        "    except Exception as exc:\n"
        "        problems.append(f'{identifier}: cannot be constructed: {exc}')\n"
        "        continue\n"
        "    if not mode.is_scored() and not getattr(mode, 'unscored_reason', ''):\n"
        "        problems.append(f'{identifier} does not override score(), so it produces no signal')\n"
        "    if not mode.parameters():\n"
        "        problems.append(f'{identifier} declares no knobs, so there is nothing to optimize')\n"
        "    for issue in validate_state(mode.state_dict(), invariants=mode.invariants(),"
        " resolve_calls=True, **mode.grader_scope()):\n"
        "        problems.append(f'{identifier}: {issue}')\n"
        # A creation can be asked for a mode that iterates data. The brief reaching
        # the prompt is a request; this is the check, and it runs where the package is
        # loaded because that is the only place its graph exists.
        "    if requires_data_node and not any(\n"
        "        node.kind == 'data' and node.subgraph_entry and node.subgraph_exit\n"
        "        for node in mode.graph().nodes.values()\n"
        "    ):\n"
        "        problems.append(f'{identifier} declares no DataNode that iterates a "
        "subgraph, which this creation asked for')\n"
        # A Task binding is a declaration too: the mode has to be bound to the Task
        # the brief named, or the run produced something the Task does not describe.
        "    for required_task in required_tasks:\n"
        "        if not any(node.kind == 'data' and node.task_ref == required_task\n"
        "                   for node in mode.graph().nodes.values()):\n"
        "            problems.append(f'{identifier} declares no DataNode bound to the Task "
        "{required_task!r}, which this creation or revision requires')\n"
        # The brief's inputs are a declaration too, and the one the pipeline could not
        # check before: a mode can ignore the thing it was asked to operate on and
        # still be structurally perfect. What is checkable is whether the mode has a
        # way to receive it — a knob, or a read named after it — because those are the
        # two ways a caller hands anything to a mode.
        "    surface = [str(knob).lower() for knob in mode.parameters()] + [\n"
        "        str(read).lower()\n"
        "        for node in mode.graph().nodes.values()\n"
        "        for read in node.reads\n"
        "    ]\n"
        "    for wanted in required_inputs:\n"
        "        slug = str(wanted).strip().lower()\n"
        "        if slug and not any(slug in entry or entry in slug for entry in surface):\n"
        "            problems.append(\n"
        "                f'{identifier} has no way to receive {wanted!r}, which this '\n"
        "                'creation asked for: declare a knob named after it, or read a '\n"
        "                'file under that name from the run directory, rather than '\n"
        "                'hardcoding a fixture'\n"
        "            )\n"
        "print(json.dumps({'problems': problems}))\n"
    )
    return run_sandboxed(code, workdir=project, env=dict(os.environ), **kwargs)


def describe_package(
    name: str, *, project: Path | None = None, include_contracts: bool = False, **kwargs: Any,
) -> SandboxResult:
    """A package's manifest and mode ids, read without importing it here.

    Reading ``PACKAGE`` means importing the module that declares it, and for a
    generated package that is the model's code. So the read happens where the
    check happens. A caller that only wants to *name* what a package contains — to
    report what was registered, or to choose a mode to smoke — should not have to
    run the package in the driver to find out. ``include_contracts`` also constructs
    the modes here and reports their data-node Task/runtime-source bindings.
    """
    project = Path(project or project_for_nodes())
    code = (
        "import json\n"
        "from pathlib import Path\n"
        "from refactory_dsh import registry\n"
        f"package = registry.load({name!r}, Path({str(project)!r}))\n"
        "payload = {'describe': package.describe(), 'ids': package.ids()}\n"
        f"if {include_contracts!r}:\n"
        "    payload['mode_contracts'] = {\n"
        "        f'{package.name}/{mode.name}': [\n"
        "            {'node': node.id, 'task_ref': node.task_ref,\n"
        "             'source_path': node.source_path, 'source_format': node.source_format}\n"
        "            for node in mode().graph().nodes.values() if node.kind == 'data'\n"
        "        ] for mode in package.modes\n"
        "    }\n"
        "print(json.dumps(payload))\n"
    )
    return run_sandboxed(code, workdir=project, env=dict(os.environ), **kwargs)


def promote(project: Path, name: str, *, note: str = "") -> dict[str, Any]:
    """Mark a package trusted, explicitly, and record who did it.

    Nothing acts on the tier until someone decides it may: a package that was
    generated stays generated until a person says otherwise.
    """
    project = Path(project)
    root = registry.workspace_root(project) / name
    if not (root / "__init__.py").exists():
        raise SandboxError(f"no workspace package {name!r}")
    origin = root / registry.ORIGIN
    recorded = json.loads(origin.read_text()) if origin.exists() else {}
    import datetime

    recorded["trust"] = "trusted"
    recorded.setdefault("origin", "promoted")
    recorded["promoted_at"] = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat(timespec="seconds")
    if note:
        recorded["note"] = note
    origin.write_text(json.dumps(recorded, indent=2))
    return recorded


def usage() -> str:
    """The one usage line, printed both for ``--help`` and for misuse."""
    return (
        "usage: python -m refactory_dsh.sandbox "
        "--ast <package> | --verify <package> | --smoke <package>/<mode>"
    )


def main(argv: list[str] | None = None) -> int:
    """``python -m refactory_dsh.sandbox --ast <pkg> | --smoke <pkg>/<mode>``.

    Runnable so the smoke check can itself be run under a different containment
    (firejail, gvisor, a container) without changing anything above it.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    project = project_for_nodes()
    workdir = Path(os.environ.get("REFACTORY_WORKDIR") or project)
    if argv[:1] in (["-h"], ["--help"]):
        # Asking for help is not a failure: usage goes to stdout and the exit
        # status is 0, so a person or a doc checker can read it. A bare
        # invocation with no instruction is still misuse: usage on stderr, 2.
        print(usage())
        return 0
    try:
        if argv[:1] == ["--ast"] and len(argv) == 2:
            problems = ast_check(argv[1], project)
            print(json.dumps({"package": argv[1], "problems": problems}))
            return 1 if problems else 0
        if argv[:1] == ["--verify"] and len(argv) == 2:
            result = verify_sandboxed(argv[1], project=project)
            problems = list(result.payload.get("problems") or [])
            if not result.ok and not problems:
                problems = [result.problem() or "the check could not be run"]
            print(json.dumps({"package": argv[1], "problems": problems}))
            return 1 if problems else 0
        if argv[:1] == ["--smoke"] and len(argv) == 2:
            result = smoke(argv[1], workdir, project=project)
            print(json.dumps({"ok": result.ok, "problem": result.problem(), **result.payload}))
            return 0 if result.ok else 1
    except SandboxError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    print(usage(), file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover - the entry point
    raise SystemExit(main())
