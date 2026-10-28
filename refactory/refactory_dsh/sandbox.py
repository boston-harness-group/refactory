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
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from refactory_dsh import registry
from refactory_dsh.paths import project_root, project_for_nodes

#: Ceilings for one sandboxed run. A shell command (`run_command`) is bounded by its
#: time rather than by the memory and CPU ceilings, which hold for code snippets (#325).
DEFAULT_TIMEOUT_S = 180
#: Whole-mode smoke budget when model calls are required; independently configurable.
DEFAULT_MODEL_SMOKE_TIMEOUT_S = 3600
DEFAULT_MEMORY_MB = 2048
DEFAULT_CPU_S = 120
DEFAULT_FILE_MB = 16384

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

    def problem(self) -> str:
        if self.timed_out:
            limit = f" of {self.timeout_s}s" if self.timeout_s is not None else ""
            return f"the run exceeded its time limit{limit}; partial artifacts remain in the workdir"
        if self.returncode != 0:
            tail = self.stderr.strip().splitlines()
            return tail[-1] if tail else f"exit {self.returncode}"
        return ""


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
    """
    import resource

    def apply() -> None:
        caps = (
            (resource.RLIMIT_CPU, (cpu_s, cpu_s)),
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
    launching shell's PATH reads, and `scrub` does not add it.
    """
    from refactory_dsh.paths import venv_bin

    out = scrub(env)
    bindir = str(venv_bin())
    rest = [entry for entry in out.get("PATH", "").split(os.pathsep) if entry and entry != bindir]
    out["PATH"] = os.pathsep.join([bindir, *rest])
    out["VIRTUAL_ENV"] = str(Path(bindir).parent)
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
) -> SandboxResult:
    """Run a snippet in a subprocess with limits, from the workspace, scrubbed.

    A subprocess rather than a thread or a same-process guard, because the thing
    being contained is *code*: only a separate process can be killed, capped and
    given its own working directory. ``address_space=False`` keeps the heap cap and
    drops the address-space cap (`_limits`).
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
    """Run a shell command in the sandbox, bounded by time rather than memory.

    A command is what a brief declares as its check ("run the tests", "build it"),
    and it is arbitrary, so it gets the same scrubbed environment and file limit as
    anything else we did not write. What bounds it is ``timeout_s``. It has no memory cap
    unless ``memory_mb`` sets one, because V8 reserves far more address space than it
    uses, so Node fails under any practical address-space limit once it starts worker
    threads, as vitest does (#325).
    Its CPU limit, unless ``cpu_s`` sets one, is that time on every core, so a test
    runner's worker threads are not stopped before the timeout. When the time runs out,
    everything the command started is stopped with it, not only its shell. ``exact_env``
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
        cpu = cpu_s if cpu_s is not None else math.ceil(timeout_s) * (os.cpu_count() or 1)
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
    options = dict(shell=True, text=True, cwd=str(workdir), env=child_env,
                   stdin=subprocess.DEVNULL if stdin is None else stdin, **kwargs)

    def _once(deadline: float) -> subprocess.CompletedProcess[str]:
        """Run the command once, bounded by *deadline*; raise on timeout or budget."""
        proc = subprocess.Popen(wrapped, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                start_new_session=os.name == "posix", **options)
        try:
            while True:
                if budget is not None:
                    budget.checkpoint()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout_s)
                try:
                    # A budget is checked between short waits; without one, wait out the time.
                    stdout, stderr = proc.communicate(
                        timeout=remaining if budget is None else min(remaining, 0.1))
                    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            # This session was created by this call and contains only this
            # check's shell/tools. No descendant discovery or name matching.
            try:
                os.killpg(proc.pid, signal.SIGTERM) if os.name == "posix" else proc.terminate()
            except ProcessLookupError:
                pass
            try:
                proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL) if os.name == "posix" else proc.kill()
                except ProcessLookupError:
                    pass
                proc.communicate(timeout=2)
            raise

    started = time.monotonic()
    try:
        if budget is not None:
            budget.checkpoint()
        result = _once(time.monotonic() + timeout_s)
        if result.returncode < 0:
            # Killed by a signal — retry once; the first death is transient.
            result = _once(time.monotonic() + timeout_s)
            if result.returncode < 0:
                sig_num = -result.returncode
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
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=f"{sig_label}\n{result.stderr}" if result.stderr else sig_label,
                )
        elif result.returncode in _AMBIGUOUS_SHELL_EXITS:
            # Shell-wrapped SIGABRT/SIGBUS/SIGSEGV — retry once (#377, extended to
            # 128+N shells).  If the retry still crashes, the result falls through to
            # the caller's base comparison, which decides whether the crash is the
            # change's fault or infrastructure.
            result = _once(time.monotonic() + timeout_s)
    except subprocess.TimeoutExpired:
        if sentinel_path is not None:
            sentinel_path.unlink(missing_ok=True)
        return SandboxResult(
            ok=False, timed_out=True, returncode=124, timeout_s=timeout_s,
            elapsed_s=round(time.monotonic() - started, 3), progress=_progress_at(workdir),
            stderr=f"timed out after {timeout_s}s",
        )
    cwd_outside: str | None = None
    if sentinel_path is not None:
        try:
            cwd_outside = sentinel_path.read_text().strip() or None
        except OSError:
            pass
        finally:
            sentinel_path.unlink(missing_ok=True)
    return SandboxResult(
        ok=result.returncode == 0,
        timeout_s=timeout_s,
        elapsed_s=round(time.monotonic() - started, 3),
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        cwd_outside=cwd_outside,
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


def smoke(
    identifier: str,
    workdir: Path,
    *,
    project: Path | None = None,
    model: bool = False,
    knobs: dict[str, Any] | None = None,
    model_context: dict[str, str] | None = None,
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
        f"timeout_s=smoke_timeout_s, {call})"
        if model else f"mode.evaluate(Path({str(Path(workdir))!r}))"
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
        f"result = {evaluation}\n"
        # The knob values the run used, so a comparison can say what differed (dm-263).
        # Reported after the run, so a mode that cannot report them still passes.
        "try:\n"
        "    knobs = {name: entry.get('value') for name, entry in"
        " (mode.state_dict().get('knobs') or {}).items() if isinstance(entry, dict)}\n"
        "except Exception:\n"
        "    knobs = None\n"
        "print(json.dumps({'exit_ok': result.exit_ok, 'error': result.error,"
        " 'score': result.score, 'iterations': result.iterations, 'knobs': knobs}, default=str))\n"
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
        forwarded["REFACTORY_BUDGET_LEDGER"] = str(budget.path)
    if model:
        # Leave time for the inner runtime to shut down before sandbox termination.
        timeout = kwargs["timeout_s"]
        code = f"smoke_timeout_s={max(1, int(timeout - min(5, timeout / 2)))!r}\n" + code
    return run_sandboxed(
        code, workdir=Path(workdir), env=dict(os.environ), forwarded=forwarded, **kwargs
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
        # the shared workspace first (appended) and the staging workspace at
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
        " resolve_calls=True):\n"
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
