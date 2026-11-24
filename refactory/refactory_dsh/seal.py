"""The sealed workspace: where the tools of a step an edit adds run.

A step the outer loop adds may list tools (:data:`SEALED_TOOLS`). It works in a fresh
``scratch`` workspace holding copies of the files it reads, and every command its shell
runs goes through :func:`runner_command`, which DSH's local sandbox provider puts in
front of the command (its ``runnerCommand``). What the command can then see:

- its workspace, read and write, at the same path;
- the operating system's own programs and libraries, read only;
- nothing else: no other item's directory, no run directory, no project, no home
  directory, no grader file or gold answer wherever the run keeps them, no network;
- limits on its CPU time, file size and, on Linux, processes and memory (:data:`LIMITS`;
  a cap on the whole step where one can be made, :func:`step_cap`: a systemd user scope,
  :func:`linux_scope`, or else a cgroup delegated to the run, `seal_cgroup`);
- on Linux, a fresh session keyring, and death with the DSH process that started it
  (`seal_reaper.exec_runner`); /usr/local only as far as its programs and libraries
  (:data:`LOCAL_KEPT`);
- an environment of its own (`PATH`, its workspace as HOME and TMPDIR, the locale) and
  no inherited descriptor: nothing of the DSH process's, so no credential or run
  variable; no other process's arguments or environment either;
- where the item's Task declares one (`task.Task.tool_environment`), its interpreter
  first on that PATH as ``python3`` and ``python``, any further commands it names beside
  them, and the interpreter's and those commands' directories read only (:func:`tool_path`):
  never one that holds the run, the home directory or refactory's own; and, where it asks
  for one (Linux only), a ``/tmp`` of the step's own, writable and kept across its commands
  (a folder beside the workspace; LibreOffice puts its pipe there).

It is the code step's sandbox (`snippet`) with a writable workspace and a shell:

- **Linux**: user, network, mount and PID namespaces (``unshare -Urnmp``). A new root
  is built from an allow-list (:data:`LINUX_READS` and the :data:`LINUX_ETC` files, each
  bound alone and read only; four ``/dev`` nodes; a fresh ``/proc``; the
  workspace read and write at its own path) and pivoted to; the old root is detached
  and the new one is read only, so the workspace is the one place it writes.
  It never runs as root (:class:`SealError`). The command then runs in a nested user
  namespace as the caller's own uid, not root, and can make no user namespace of its
  own, so every mount is locked: nothing can be unmounted or remounted to reveal what
  is underneath. The network namespace has only
  a down loopback, and the PID namespace shows only the step's own processes.
- **macOS**: Seatbelt, deny by default (:func:`mac_profile`): reads of the system's
  programs and libraries (not Homebrew's or ``/usr/local``'s data), reads and writes of
  the workspace, starting processes from what it may read, no other process's
  information, only named system settings, and nothing on the network. macOS has no
  PID namespace, so the runner ends every process left in the command's sandbox when
  the command ends (`seal_reaper`), and collecting the workspace does so again.

DSH's in-process tools (read, write, edit, glob, grep, web) run outside any runner, so a
sealed step may list only tools whose effects run through it: the shell. A runner that
cannot set the seal up prints ``refactory-seal:`` and exits 97, and the command does not
run (`FAILURE`). There is no fallback: where the seal cannot run (:func:`available`) a
new step is not offered tools.

A step on the ``claude_code`` worker backend may instead list Claude Code's own tools
from :data:`CLAUDE_CODE_TOOLS` (``Read`` and ``Bash``), as a harness that runs Claude
Code with ``--tools Read,Bash`` does. Claude Code runs them in its own process, so the
adapter confines them with a ``PreToolUse`` hook: each Bash command is rewritten to run
under the same runner (the runner, ``--``, ``/bin/bash -c`` and the command), and a Read
of any path whose real path is outside the workspace is refused. Any other tool is
refused. The workspace and its runner are made here and in `workspace` as for a DSH
step; only who puts the runner in front of a command differs.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from refactory_dsh import seal_cgroup, seal_reaper, snippet

#: The tools a step an edit adds may list (`Node.tools`): those whose every effect runs
#: through the seal. DSH's file and web tools run in its own process, outside any
#: runner, so they are not among them.
SEALED_TOOLS = ("bash",)
#: The Claude Code tools a sealed step on the ``claude_code`` backend may list instead:
#: Bash runs each command through the runner, and Read reads inside the workspace only,
#: both enforced by the adapter's ``PreToolUse`` hook (the module docstring).
CLAUDE_CODE_TOOLS = ("Read", "Bash")

#: What the runner prints when it could not seal a command, as DSH's provider matches it.
FAILURE = "refactory-seal: "

#: What a sealed command's environment holds: nothing of the DSH process's (no
#: credential, no run variable), only these, with HOME and TMPDIR its workspace.
PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

#: A sealed command's limits, as ``ulimit`` takes them: CPU seconds and the largest file
#: (KiB). macOS counts processes per user, not per sandbox, and does not enforce an
#: address-space limit, so it gets only these.
#: (Defined in `seal_reaper`, which a runner whose scope's bus went away uses to fall back
#: to the unscoped limits, and which runs with nothing but the standard library.)
LIMITS = seal_reaper.LIMITS
#: Linux, where a step's commands run in a systemd user scope (:func:`linux_scope`) or a
#: delegated cgroup (`seal_cgroup`): either caps the memory and processes of the command
#: and everything it starts together (:data:`SCOPE_LIMITS`, `seal_reaper.cgroup_limits`:
#: the same values), so the processes (counted in the command's own user namespace) need
#: only a loose per-process bound.
LINUX_LIMITS = seal_reaper.LINUX_LIMITS
SCOPE_LIMITS = seal_reaper.scope_properties()
#: Linux without a whole-step cap: each process gets a tight one (128 processes, 8 GiB of
#: address space each).
LINUX_LIMITS_UNSCOPED = seal_reaper.LINUX_LIMITS_UNSCOPED

#: macOS: closes every descriptor above stderr, then checks that none is left (a command
#: that would start with one does not start), then runs its arguments. Linux runs the
#: same step from a file (``seal_close.sh``), since nothing on macOS rewrites arguments.
_CLOSE = ('for fd in /dev/fd/*; do n="${fd##*/}"; '
          '[ "$n" -gt 2 ] 2>/dev/null && eval "exec $n>&-" 2>/dev/null; done; '
          'for fd in /dev/fd/*; do n="${fd##*/}"; '
          'if [ "$n" -gt 2 ] 2>/dev/null && [ -e "$fd" ]; then '
          'echo "refactory-seal: descriptor $n is still open" >&2; exit 97; fi; done; '
          'exec "$@"')

#: macOS: sets the limits, then runs the command (after DSH's profile arguments and
#: ``--``) in the workspace with that environment (``$cmd_path`` its PATH), through
#: :data:`_CLOSE`.
_EXEC = (f'ulimit {LIMITS} || {{ echo "refactory-seal: cannot set limits" >&2; exit 97; }}; '
         'while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do shift; done; '
         '[ "$1" = "--" ] || { echo "refactory-seal: no command" >&2; exit 97; }; shift; '
         'cd "$ws" || exit 97; '
         f'exec /bin/bash -c \'{_CLOSE}\' refactory-seal '
         '/usr/bin/env -i HOME="$ws" TMPDIR="$ws" PATH="$cmd_path" '
         f'LANG={snippet.LOCALE} LC_ALL={snippet.LOCALE} "$@"')

#: What a sealed Linux command may read besides its workspace, each bound read only and
#: alone (never what is mounted inside it): the programs and libraries, and from /etc
#: only what they need to start (the loader's cache, users and groups by name, time
#: zones, certificates, the alternatives programs link through).
LINUX_READS = ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32")
LINUX_ETC = ("/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d", "/etc/passwd",
             "/etc/group", "/etc/nsswitch.conf", "/etc/localtime", "/etc/ssl", "/etc/alternatives",
             "/etc/ca-certificates")
#: What of ``/usr/local`` a sealed Linux command may read (its programs and libraries,
#: CUDA's included); everything else there is covered (``seal_linux.sh``).
LOCAL_KEPT = ("bin", "sbin", "lib", "lib64", "libexec", "include", "cuda", "cuda-*")

#: The Linux runner's two steps, run from files (never inline code: systemd-run expands
#: ``$NAME`` in the arguments it is given): building the new root and starting the
#: command (``seal_linux.sh``), and the command's last step, which closes and checks its
#: descriptors (``seal_close.sh``).
LINUX_SCRIPT = Path(__file__).with_name("seal_linux.sh")
CLOSE_SCRIPT = Path(__file__).with_name("seal_close.sh")


#: Where a system program the seal runs is looked for when this process's PATH lacks it:
#: a system service's PATH may leave out the sbin folders (``pivot_root`` lives there).
SYSTEM_DIRS = ("/usr/sbin", "/sbin", "/usr/bin", "/bin")


def system_tool(name: str) -> str:
    """The path of the system program ``name`` the seal runs: found on PATH, else in
    :data:`SYSTEM_DIRS`, else the bare name (which then fails, naming it)."""
    found = shutil.which(name)
    if found:
        return found
    for folder in SYSTEM_DIRS:
        candidate = os.path.join(folder, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return name


def linux_arguments(ws: str, reads: tuple[str, ...], limits: str,
                    path: str = PATH, tools: tuple[str, ...] = (), tmp: str = "") -> list[str]:
    """What follows ``bash`` in the Linux runner: the script and its arguments, none of
    which holds a ``$`` (`seal_linux.sh` says what each is); ``path`` the command's PATH,
    ``tools`` the tool environment's folders (:func:`tool_path`) not already in ``reads``,
    ``tmp`` the folder bound writable at ``/tmp`` (`ToolPath.tmp`), or none."""
    tools = tuple(d for d in tools if not any(_inside(d, root) for root in reads))
    arguments = [str(LINUX_SCRIPT), "refactory-seal", ws, str(os.getuid()), str(os.getgid()),
                 f"{LIMITS} {limits}", str(CLOSE_SCRIPT), ":".join(reads), ":".join(LOCAL_KEPT),
                 path, snippet.LOCALE, system_tool("pivot_root"), ":".join(tools), tmp]
    expanded = [argument for argument in arguments if "$" in argument or "\n" in argument]
    if expanded or any(":" in path for path in (*reads, *tools, tmp)):
        raise SealError(f"a sealed runner's arguments hold no '$', newline or ':' in a path: {expanded}")
    return arguments


#: What a sealed macOS command may read besides its workspace: the system's programs and
#: libraries, the command-line tools, Homebrew's programs and libraries (never its or
#: ``/usr/local``'s ``etc``, ``var`` or anything else a person or service keeps there),
#: and the files the shell itself opens.
_TOOLCHAIN = ("bin", "sbin", "lib", "libexec", "opt", "Cellar", "Frameworks", "include")
MAC_READS = ("/usr/bin", "/usr/sbin", "/usr/lib", "/usr/libexec", "/usr/share", "/bin", "/sbin",
             "/System", "/Library/Developer/CommandLineTools",
             *(f"{root}/{part}" for root in ("/opt/homebrew", "/usr/local") for part in _TOOLCHAIN),
             "/private/etc", "/private/var/select")
#: The devices a sealed macOS command may open; ``/dev/fd`` is its own descriptors.
MAC_DEVICES = ("/dev/null", "/dev/zero", "/dev/random", "/dev/urandom")
#: The system settings a sealed macOS command may read (`sysctlbyname`): the machine's
#: shape, never another process's arguments and environment (``kern.procargs2``). The
#: ``sysctl`` command itself also reads each setting's format through an unnamed entry,
#: which stays denied, so it fails where a program asking by name does not.
MAC_SYSCTLS = ('(sysctl-name-prefix "hw.") (sysctl-name-prefix "kern.os") '
               '(sysctl-name-prefix "machdep.cpu.") (sysctl-name "kern.hostname") '
               '(sysctl-name "kern.version") (sysctl-name "kern.argmax") '
               '(sysctl-name "kern.maxfilesperproc") (sysctl-name "kern.usrstack64") '
               '(sysctl-name "kern.secure_kernel") (sysctl-name "kern.boottime") '
               '(sysctl-name "kern.ngroups") (sysctl-name "sysctl.proc_cputype")')

#: The shim that runs the command inside the profile sandbox-exec applies (macOS).
#: Arguments: the workspace, the command's PATH, then whatever DSH appends.
_MAC_SHIM = 'ws="$1"; cmd_path="$2"; shift 2; ' + _EXEC


class SealError(RuntimeError):
    """The seal cannot be set up for this workspace; the message says why."""


def _check(workspace: Path, readable: tuple[str, ...]) -> str:
    path = os.path.realpath(workspace)
    if not os.path.isdir(path) or path == "/":
        raise SealError(f"a sealed workspace is an existing directory, not {path!r}")
    for root in readable:
        if path == root or path.startswith(root.rstrip("/") + "/"):
            raise SealError(f"the workspace {path} is inside {root}, which every sealed step may read")
    return path


def mac_profile(workspace: Path, extra: tuple[str, ...] = ()) -> str:
    """The deny-by-default Seatbelt profile a sealed macOS command runs under; ``extra``
    are more directories it may read and run programs from (a Task's interpreter's,
    :func:`tool_path`)."""
    reads = [path for path in (*MAC_READS, *extra) if os.path.exists(path)]
    ws = _check(workspace, tuple(reads))
    above = sorted({parent for path in (*reads, ws) for parent in snippet._ancestors(path)})
    readable = " ".join(f"(subpath {snippet._sbpl(path)})" for path in reads)
    devices = " ".join(f"(literal {snippet._sbpl(path)})" for path in MAC_DEVICES)
    metadata = " ".join(f"(literal {snippet._sbpl(path)})" for path in above)
    return ("(version 1)\n"
            "(deny default)\n"
            "(allow process-fork)\n"
            f"(allow process-exec (subpath {snippet._sbpl(ws)}) {readable})\n"
            # Another process's arguments and environment (KERN_PROCARGS2, the DSH
            # process's credentials among them) are read through process-info and
            # sysctl-read, which the default does not deny: both are denied outright.
            "(deny process-info*)\n"
            "(allow process-info* (target self))\n"
            "(allow signal (target same-sandbox))\n"
            "(deny sysctl-read)\n"
            f"(allow sysctl-read {MAC_SYSCTLS})\n"
            f'(allow file-read* (literal "/") (literal "/dev") {devices} '
            f'(regex #"^/dev/fd(/|$)") {readable} (subpath {snippet._sbpl(ws)}))\n'
            + (f"(allow file-read-metadata {metadata})\n" if metadata else "")
            + f'(allow file-write* (literal "/dev/null") (regex #"^/dev/fd/") '
              f'(subpath {snippet._sbpl(ws)}))\n')


@dataclass(frozen=True)
class StepCap:
    """How a sealed step's commands are capped as a whole (`step_cap`): ``capped``, and
    what the cap is or why there is none (its dispatch record's ``whole_step_cap`` and
    ``cap``); the systemd user scope prefix, or the delegated cgroup, that makes it."""

    capped: bool
    how: str
    scope: tuple[str, ...] = ()
    cgroup: str = ""


@dataclass(frozen=True)
class ToolPath:
    """A Task's tool environment as a sealed command gets it (:func:`tool_path`): ``bin``,
    first on its PATH, holds ``python3``, ``python`` and the Task's further commands;
    ``reads`` are the interpreter's directories, the commands' folders and ``bin``, each
    bound read only; ``tmp``, when set, is the step's own folder bound writable at
    ``/tmp`` (Linux only)."""

    bin: str
    reads: tuple[str, ...]
    python: str
    tmp: str = ""


#: The names a Task's interpreter answers to on a sealed command's PATH.
PYTHON_NAMES = ("python3", "python")


def distinct_paths(paths) -> list[str]:
    """Absolute and resolved forms of ``paths``, existing ones only, in order, without
    one that another of them contains."""
    out: list[str] = []
    for path in paths:
        for form in (os.path.abspath(path), os.path.realpath(path)):
            if os.path.exists(form) and form not in out:
                out.append(form)
    return [p for p in out
            if not any(o != p and (p + "/").startswith(o.rstrip("/") + "/") for o in out)]


@functools.lru_cache(maxsize=None)
def interpreter_dirs(python: str) -> tuple[str, ...]:
    """The directories ``python`` runs from: the prefixes it reports itself (``sys.prefix``,
    ``base_prefix``, ``exec_prefix``, ``base_exec_prefix``) and the folder its executable
    resolves to. Never a folder guessed from where ``python`` is named (a link to it may
    sit anywhere); none at all when it cannot report."""
    import json

    try:
        proc = subprocess.run(
            [python, "-I", "-c", "import json, sys; print(json.dumps([sys.prefix, "
             "sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix]))"],
            capture_output=True, text=True, timeout=60)
        found = json.loads(proc.stdout) if proc.returncode == 0 else []
    except (OSError, subprocess.SubprocessError, ValueError):
        return ()
    if not found:
        return ()
    return tuple(distinct_paths([*found, os.path.dirname(os.path.realpath(python))]))


def _inside(path: str, root: str) -> bool:
    path, root = os.path.realpath(path), os.path.realpath(root)
    return path == root or path.startswith(root.rstrip("/") + "/")


def refused_binding(bound, *, held=(), within=()) -> str | None:
    """Why binding the folders ``bound`` into a confined root would show what it must
    not: one of them holds a path of ``held`` (or is it), or lies inside one of
    ``within``; ``None`` when none does. Paths compare resolved. The model-code sandbox
    (`refactory_packages.spreadsheetbench.sandbox`) and the seal's tool environment
    (:func:`tool_path`) both refuse on it."""
    for folder in bound:
        for secret in held:
            if os.path.lexists(secret) and _inside(secret, folder):
                return f"{secret}: it is inside {folder}"
        for root in within:
            if os.path.lexists(root) and _inside(folder, root):
                return f"{folder}: it is inside {root}"
    return None


def tool_path(python: str, bin_dir: Path, *, workspace: Path, runs=(), hidden=(),
              commands=None, reads=(), tmp: Path | None = None) -> ToolPath:
    """The Task interpreter ``python`` for a sealed command: ``bin_dir`` (made afresh, beside
    the workspace and never inside it) holds ``python3`` and ``python``, each a two-line
    script that runs ``python`` by its own path (a link would lose a virtual environment),
    and the folders it reports it runs from (`interpreter_dirs`) are bound read only.

    Refused (`SealError`) when one of those folders holds, or lies inside, any of: the
    run's roots ``runs`` (the run every item's pass is in, and the outer loop's root above
    it), refactory's own directory, or a Task's ``hidden`` paths (its dataset, the secrets
    `ToolEnvironment.hidden` names); or when one holds the home directory. The workspace is
    inside the run, so it is covered. So the seal shows no item's pass, grader file or
    dataset through it.

    ``commands`` (name -> script text, `task.ToolEnvironment.commands`) are written into
    ``bin_dir`` beside them, and ``reads`` (the folders they run from) are bound read only,
    held to the same refusals. ``tmp`` (`task.ToolEnvironment.private_tmp`), made afresh
    beside the workspace, is bound writable at ``/tmp``: the step's own, kept across its
    commands; refused on macOS, whose seal cannot give a command a /tmp of its own."""
    from refactory_dsh.paths import refactory_home

    python = os.path.abspath(python)
    if not os.path.isfile(python) or not os.access(python, os.X_OK):
        raise SealError(f"the Task's interpreter {python} is not an executable file")
    dirs = interpreter_dirs(python)
    if not dirs:
        raise SealError(f"the Task's interpreter {python} does not report where it runs from")
    secret = [*map(str, runs), str(refactory_home()), *map(str, hidden)]
    why = refused_binding(dirs, held=[*secret, str(Path.home())],
                          within=secret)
    if why:
        raise SealError(f"the Task's interpreter {python} runs from a folder a sealed step may "
                        f"not read ({why})")
    reads = tuple(str(r) for r in reads)
    for folder in reads:
        if not os.path.isabs(folder) or not os.path.isdir(folder):
            raise SealError(f"the Task's tool environment reads {folder!r}, which is not a folder")
    extra = tuple(distinct_paths(reads))
    why = refused_binding(extra, held=[*secret, str(Path.home())], within=secret)
    if why:
        raise SealError(f"the Task's tool environment reads a folder a sealed step may not "
                        f"read ({why})")
    commands = dict(commands or {})
    for name in commands:
        if not name or name in PYTHON_NAMES or "/" in name or name.startswith("."):
            raise SealError(f"the Task's tool environment cannot name a command {name!r}")
    if tmp is not None:
        if sys.platform == "darwin":
            raise SealError("a sealed command gets a /tmp of its own on Linux only")
        if _inside(str(tmp), str(workspace)) or _inside(str(workspace), str(tmp)) \
                or _inside(str(tmp), str(bin_dir)) or _inside(str(bin_dir), str(tmp)):
            raise SealError(f"the step's /tmp folder {tmp} is not beside the workspace")
    bin_dir = Path(bin_dir)
    if _inside(str(bin_dir), str(workspace)) or _inside(str(workspace), str(bin_dir)):
        raise SealError(f"the interpreter's folder {bin_dir} is not beside the workspace")
    for path in (python, os.path.realpath(python), str(bin_dir), *dirs, *extra,
                 *((str(tmp),) if tmp is not None else ())):
        if any(ch in path for ch in ":$\n\"'\\"):
            raise SealError(f"a sealed command's PATH cannot name {path!r}")
    # Run by the name it was given where that name is inside its own folders (a virtual
    # environment's bin/python, which only that name makes one), else by what it resolves to:
    # a name elsewhere (a link beside other files) is not in the seal.
    named = any(python.startswith(folder.rstrip("/") + "/") for folder in dirs)
    run_as = python if named else os.path.realpath(python)
    shutil.rmtree(bin_dir, ignore_errors=True)
    bin_dir.mkdir(parents=True)  # raises when an earlier one could not be removed
    for name in PYTHON_NAMES:
        script = bin_dir / name
        script.write_text(f'#!/bin/sh\nexec "{run_as}" "$@"\n')
        script.chmod(0o555)
    for name, text in commands.items():
        script = bin_dir / name
        script.write_text(text)
        script.chmod(0o555)
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
        Path(tmp).mkdir(mode=0o700, parents=True)  # raises when an earlier one is still there
    return ToolPath(str(bin_dir), tuple(distinct_paths([str(bin_dir), *dirs, *extra])), run_as,
                    tmp=str(tmp) if tmp is not None else "")


def tool_path_from_environment(environment, bin_dir: Path, *,
                               workspace: Path, runs=(),
                               tmp_dir: Path | None = None,
                               workdir: Path | None = None) -> ToolPath:
    """Unpack a `task.ToolEnvironment` and call `tool_path`: the shared path for sealed
    agent steps (`workspace._tool_path`) and sealed script steps (`snippet.sealed_execute`)."""
    python = environment.python()
    hidden = list(environment.hidden(workdir)) if environment.hidden is not None else []
    commands = dict(environment.commands()) if environment.commands is not None else {}
    reads = list(environment.reads()) if environment.reads is not None else []
    return tool_path(python, bin_dir, workspace=workspace, runs=runs, hidden=hidden,
                     commands=commands, reads=reads,
                     tmp=(tmp_dir if environment.private_tmp else None))


def runner_command(workspace: Path, cap: StepCap | None = None,
                   tools: ToolPath | None = None) -> list[str]:
    """The runner argv DSH's local sandbox provider puts in front of each command of a
    sealed step (``runnerCommand``), under the step's whole-step ``cap`` (`step_cap`).
    Without one (a probe, `run`) a command gets a user scope where one can be made, and
    per-process limits only otherwise. ``tools`` (:func:`tool_path`): the Task's
    interpreter first on the command's PATH, its directories read only."""
    extra = tools.reads if tools is not None else ()
    path = f"{tools.bin}:{PATH}" if tools is not None else PATH
    tmp = tools.tmp if tools is not None else ""
    if sys.platform == "darwin":
        if tmp:
            raise SealError("a sealed command gets a /tmp of its own on Linux only")
        ws = _check(workspace, (*MAC_READS, *extra))
        # The reaper runs sandbox-exec, then ends whatever the command left behind.
        reaper = str(Path(__file__).with_name("seal_reaper.py"))
        return [sys.executable, "-I", reaper, ws, snippet.SANDBOX_EXEC, "-p",
                mac_profile(workspace, extra), "/bin/sh", "-c", _MAC_SHIM, "refactory-seal", ws,
                path]
    if not sys.platform.startswith("linux"):
        raise SealError(f"no sealed workspace on {sys.platform}")
    ws = _check(workspace, LINUX_READS + LINUX_ETC + extra)
    if os.getuid() == 0:
        # Root here is root inside, which reads what only root may (/etc/shadow and the
        # like): a sealed step runs as an ordinary user, or not at all.
        raise SealError("a sealed step does not run as root")
    scope, cgroup = (cap.scope, cap.cgroup) if cap is not None else (linux_scope(), "")
    if cgroup and (not cgroup.startswith("/") or "\n" in cgroup):
        raise SealError(f"a step's cgroup is an absolute path on one line, not {cgroup!r}")
    # Started through the reaper's --exec: a fresh session keyring, and killed with DSH;
    # with --cgroup, moved into the step's delegated cgroup first.
    reaper = str(Path(__file__).with_name("seal_reaper.py"))
    return [sys.executable, "-I", reaper, "--exec",
            *([seal_reaper.CGROUP_FLAG, cgroup] if cgroup else []), *scope,
            system_tool("unshare"), "-Urnmp", "--fork", "--kill-child",
            "--mount-proc", "--", snippet.BASH,
            *linux_arguments(ws, LINUX_READS + LINUX_ETC,
                             LINUX_LIMITS if scope or cgroup else LINUX_LIMITS_UNSCOPED, path,
                             extra, tmp)]


def user_manager_kept(groups: str | None = None,
                      linger_root: str = "/var/lib/systemd/linger") -> bool:
    """Whether this user's systemd manager runs as long as this process does: this process
    is in one of the user's sessions or in the manager itself (its cgroup is under
    ``user-<uid>.slice``), or the user lingers. Otherwise the manager stops a few seconds
    after the user's last session ends, and with it every scope it made -- a sealed
    command's included, killed mid-command, and the bus a new scope is made through gone
    -- so a sealed command runs unscoped instead (`linux_scope`)."""
    import pwd

    if groups is None:
        try:
            with open("/proc/self/cgroup", encoding="utf-8") as handle:
                groups = handle.read()
        except OSError:
            groups = ""
    if f"/user-{os.getuid()}.slice/" in groups:
        return True
    try:
        name = pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        return False
    return os.path.exists(os.path.join(linger_root, name))


def _no_scope_reason() -> str:
    """Why this machine gives a sealed command no systemd user scope, in plain words, or
    "" where it may (`_scope_prefix`)."""
    if not shutil.which("systemd-run"):
        return "no systemd-run here"
    if not os.path.isdir(f"/run/user/{os.getuid()}"):
        return "no systemd user manager is running for this user"
    if not user_manager_kept():
        return ("this run is outside the user's sessions and the user does not linger, so "
                "systemd's user manager may stop (and take a scope's commands with it) at "
                "any time")
    return ""


#: The reasons for no whole-step cap this process has warned of, each once (`step_cap`).
_WARNED: set[str] = set()


def step_cap(workspace: Path) -> StepCap:
    """The cap on the whole of what the sealed step in ``workspace`` starts (memory, swap,
    processes), in this order: a systemd user scope where the user's manager runs as long
    as this process (`linux_scope`); else a cgroup of its own in the cgroup systemd
    delegated to this run (`seal_cgroup.make`; a launch with ``-p Delegate=yes``); else
    none, with per-process limits only (and on macOS). Says which, or why none, for each
    sealed step's dispatch record (`workspace.Workspace.describe`) and the run's report.
    A delegated cgroup made here is the caller's to remove (`workspace.release_cgroup`)."""
    if sys.platform == "darwin":
        return StepCap(False, "macOS gives a sealed command per-process limits only")
    scope = linux_scope()
    if scope:
        return StepCap(True, "a systemd user scope (" + ", ".join(SCOPE_LIMITS) + ")", scope=scope)
    no_scope = _no_scope_reason() or "the systemd user scope probe failed"
    try:
        cgroup, held = seal_cgroup.make(workspace)
    except seal_cgroup.Unavailable as exc:
        if str(exc) not in _WARNED:
            _WARNED.add(str(exc))
            print(f"refactory: warning: a sealed step runs without a whole-step cap here "
                  f"({exc}), under per-process limits only ({LINUX_LIMITS_UNSCOPED})",
                  file=sys.stderr)
        return StepCap(False, f"{no_scope}; no delegated cgroup either: {exc}; per-process "
                              f"limits only ({LINUX_LIMITS_UNSCOPED})")
    return StepCap(True, f"a delegated cgroup ({held}) at {cgroup}", cgroup=cgroup)


def _scope_prefix() -> list[str]:
    systemd_run = shutil.which("systemd-run")
    runtime = f"/run/user/{os.getuid()}"
    if _no_scope_reason():
        return []
    # Where systemd-run can be told not to expand $NAME in the command, it is told so;
    # the runner's arguments hold no '$' either way (`linux_arguments`).
    try:
        usage = subprocess.run([systemd_run, "--help"], capture_output=True, text=True,
                               timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    expand = ["--expand-environment=no"] if "--expand-environment" in usage else []
    return ["/usr/bin/env", f"XDG_RUNTIME_DIR={runtime}",
            f"DBUS_SESSION_BUS_ADDRESS=unix:path={runtime}/bus", systemd_run, "--user",
            "--scope", "--quiet", "--collect", *expand,
            *(f"--property={p}" for p in SCOPE_LIMITS), "--"]


#: The scope prefix once a probe found one (`linux_scope`); a failed probe is not kept.
_SCOPE: list[tuple[str, ...]] = []
#: The `_WARNED` key of a failed user scope probe (`linux_scope`).
_PROBE_FAILED = "the systemd user scope probe failed"



def _scope_bus_exists(prefix: tuple[str, ...]) -> bool:
    """Whether the user bus a kept scope prefix connects through is still there."""
    bus = seal_reaper.scope_bus(list(prefix))
    return bus is None or os.path.exists(bus)


def linux_scope() -> tuple[str, ...]:
    """What puts a sealed Linux command in a systemd user scope capped at
    :data:`SCOPE_LIMITS`, or ``()`` where there is no user manager to make one, or none
    that runs as long as this process (`user_manager_kept`) (the step then gets a
    delegated cgroup where there is one, else :data:`LINUX_LIMITS_UNSCOPED`: `step_cap`). A probe
    that finds its own memory cap is kept for the process; a failed one is tried again
    next time, so one slow probe does not drop every later step to the fallback. A kept
    prefix whose user bus is gone is dropped and probed again. A command whose scope bus
    goes away after its runner was made runs unscoped instead (`seal_reaper.exec_runner`)."""
    if _SCOPE:
        if _scope_bus_exists(_SCOPE[0]):
            return _SCOPE[0]
        # systemd's user manager went away (its runtime directory goes with the person's
        # last session): probed again, so later commands run unscoped rather than fail.
        _SCOPE.clear()
    prefix = _scope_prefix()
    found: tuple[str, ...] = ()
    if prefix:
        try:
            probe = subprocess.run(
                [*prefix, "/bin/sh", "-c",
                 'cat "/sys/fs/cgroup$(sed -n "s/^0:://p" /proc/self/cgroup)/memory.max"'],
                capture_output=True, text=True, timeout=20)
            if probe.returncode == 0 and probe.stdout.strip() == str(seal_reaper.STEP_MEMORY_MAX):
                found = tuple(prefix)
        except (OSError, subprocess.SubprocessError):
            pass
    if found:
        _SCOPE.append(found)
    elif prefix and _PROBE_FAILED not in _WARNED:
        # Said once, and only when a user manager was there to ask and its probe failed:
        # where there is none, `step_cap` caps the step another way (a delegated cgroup)
        # or says, once per reason, that it has no whole-step cap.
        _WARNED.add(_PROBE_FAILED)
        print("refactory: warning: the systemd user scope probe failed here; a sealed step's "
              "command runs without a user scope (each step's dispatch record says what caps "
              "it instead)", file=sys.stderr)
    return found


def run(workspace: Path, command: list[str], *, timeout_s: float = 60) -> subprocess.CompletedProcess:
    """``command`` run sealed in ``workspace``, as DSH runs a sealed step's command: the
    runner, a stand-in for DSH's profile arguments, ``--``, then the command."""
    argv = [*runner_command(workspace), "--ro-bind", "/", "/", "--", *command]
    return subprocess.run(argv, cwd=os.path.realpath(workspace), capture_output=True,
                          text=True, timeout=timeout_s)


@functools.lru_cache(maxsize=1)
def available() -> tuple[bool, str]:
    """Whether a sealed command runs here and is sealed, and why not: one probe per
    process. The probe writes and reads its workspace and must not read a file beside it."""
    try:
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "outside.txt").write_text("outside")
            (Path(root) / "ws").mkdir()
            result = run(Path(root) / "ws", ["/bin/sh", "-c", "echo sealed > probe && cat probe "
                                             "&& cat ../outside.txt 2>/dev/null; true"])
    except (SealError, OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if result.returncode != 0 or result.stdout.strip() != "sealed":
        return False, (result.stderr.strip() or f"exit {result.returncode}: "
                       f"{result.stdout.strip()[:100]!r}")[:300]
    return True, ""
