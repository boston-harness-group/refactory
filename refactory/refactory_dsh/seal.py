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
  variable; no other process's arguments or environment either.

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
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from refactory_dsh import seal_cgroup, seal_reaper, snippet

#: The tools a step an edit adds may list (`Node.tools`): those whose every effect runs
#: through the seal. DSH's file and web tools run in its own process, outside any
#: runner, so they are not among them.
SEALED_TOOLS = ("bash",)

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
#: ``--``) in the workspace with that environment, through :data:`_CLOSE`.
_EXEC = (f'ulimit {LIMITS} || {{ echo "refactory-seal: cannot set limits" >&2; exit 97; }}; '
         'while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do shift; done; '
         '[ "$1" = "--" ] || { echo "refactory-seal: no command" >&2; exit 97; }; shift; '
         'cd "$ws" || exit 97; '
         f'exec /bin/bash -c \'{_CLOSE}\' refactory-seal '
         f'/usr/bin/env -i HOME="$ws" TMPDIR="$ws" PATH="{PATH}" '
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


def linux_arguments(ws: str, reads: tuple[str, ...], limits: str) -> list[str]:
    """What follows ``bash`` in the Linux runner: the script and its arguments, none of
    which holds a ``$`` (`seal_linux.sh` says what each is)."""
    arguments = [str(LINUX_SCRIPT), "refactory-seal", ws, str(os.getuid()), str(os.getgid()),
                 f"{LIMITS} {limits}", str(CLOSE_SCRIPT), ":".join(reads), ":".join(LOCAL_KEPT),
                 PATH, snippet.LOCALE]
    expanded = [argument for argument in arguments if "$" in argument or "\n" in argument]
    if expanded or any(":" in path for path in reads):
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
#: Arguments: the workspace, then whatever DSH appends.
_MAC_SHIM = 'ws="$1"; shift; ' + _EXEC


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


def mac_profile(workspace: Path) -> str:
    """The deny-by-default Seatbelt profile a sealed macOS command runs under."""
    reads = [path for path in MAC_READS if os.path.exists(path)]
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


def runner_command(workspace: Path, cap: StepCap | None = None) -> list[str]:
    """The runner argv DSH's local sandbox provider puts in front of each command of a
    sealed step (``runnerCommand``), under the step's whole-step ``cap`` (`step_cap`).
    Without one (a probe, `run`) a command gets a user scope where one can be made, and
    per-process limits only otherwise."""
    if sys.platform == "darwin":
        ws = _check(workspace, MAC_READS)
        # The reaper runs sandbox-exec, then ends whatever the command left behind.
        reaper = str(Path(__file__).with_name("seal_reaper.py"))
        return [sys.executable, "-I", reaper, ws, snippet.SANDBOX_EXEC, "-p", mac_profile(workspace),
                "/bin/sh", "-c", _MAC_SHIM, "refactory-seal", ws]
    if not sys.platform.startswith("linux"):
        raise SealError(f"no sealed workspace on {sys.platform}")
    ws = _check(workspace, LINUX_READS + LINUX_ETC)
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
            shutil.which("unshare") or "unshare", "-Urnmp", "--fork", "--kill-child",
            "--mount-proc", "--", snippet.BASH,
            *linux_arguments(ws, LINUX_READS + LINUX_ETC,
                             LINUX_LIMITS if scope or cgroup else LINUX_LIMITS_UNSCOPED)]


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
    else:
        print("refactory: warning: a sealed step's command runs without a capped systemd user "
              "scope here", file=sys.stderr)
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
