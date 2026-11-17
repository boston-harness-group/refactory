"""The sealed workspace: where the tools of a step an edit adds run.

A step the outer loop adds may list tools (:data:`SEALED_TOOLS`). It works in a fresh
``scratch`` workspace holding copies of the files it reads, and every command its shell
runs goes through :func:`runner_command`, which DSH's local sandbox provider puts in
front of the command (its ``runnerCommand``). What the command can then see:

- its workspace, read and write, at the same path;
- the operating system's own programs and libraries, read only;
- nothing else: no other item's directory, no run directory, no project, no home
  directory, no grader file or gold answer wherever the run keeps them, no network;
- an environment of its own (`PATH`, its workspace as HOME and TMPDIR, the locale) and
  no inherited descriptor: nothing of the DSH process's, so no credential or run
  variable; no other process's arguments or environment either.

It is the code step's sandbox (`snippet`) with a writable workspace and a shell:

- **Linux**: user, network, mount and PID namespaces (``unshare -Urnmp``). A new root
  is built from an allow-list (:data:`LINUX_READS` read only, four ``/dev`` nodes, a
  fresh ``/proc`` and ``/tmp``, the workspace read and write at its own path) and
  pivoted to; the old root is detached. The command then runs in a nested user
  namespace as the caller's own uid, not root, so every mount is locked: nothing can be
  unmounted or remounted to reveal what is underneath. The network namespace has only
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
from pathlib import Path

from refactory_dsh import snippet

#: The tools a step an edit adds may list (`Node.tools`): those whose every effect runs
#: through the seal. DSH's file and web tools run in its own process, outside any
#: runner, so they are not among them.
SEALED_TOOLS = ("bash",)

#: What the runner prints when it could not seal a command, as DSH's provider matches it.
FAILURE = "refactory-seal: "

#: What a sealed command's environment holds: nothing of the DSH process's (no
#: credential, no run variable), only these, with HOME and TMPDIR its workspace.
PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

#: Closes every descriptor above stderr, then runs the command (after DSH's profile
#: arguments and ``--``) in the workspace with that environment; ``$run`` is what runs
#: it (Linux: the nested, unprivileged user namespace).
_EXEC = ('for fd in /dev/fd/*; do n="${fd##*/}"; '
         '[ "$n" -gt 2 ] 2>/dev/null && eval "exec $n>&-" 2>/dev/null; done; '
         'while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do shift; done; '
         '[ "$1" = "--" ] || { echo "refactory-seal: no command" >&2; exit 97; }; shift; '
         'cd "$ws" || exit 97; '
         f'exec $run /usr/bin/env -i HOME="$ws" TMPDIR="$ws" PATH="{PATH}" '
         f'LANG={snippet.LOCALE} LC_ALL={snippet.LOCALE} "$@"')

#: What a sealed Linux command may read besides its workspace, each bound read only.
LINUX_READS = ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32", "/etc")

#: The shell that builds the new root inside the new namespaces (Linux), where it is
#: root of its own user namespace. Arguments: the workspace, the caller's uid and gid,
#: then whatever DSH appends. Any step that fails stops it: nothing runs half sealed.
_SHELL = (
    'ws="$1"; uid="$2"; gid="$3"; shift 3; '
    'fail() { echo "refactory-seal: $1" >&2; exit 97; }; '
    'exec 3<"$ws" || fail "cannot open the workspace"; '
    'mount -n -t tmpfs -o mode=0755,size=1m seal /tmp || fail "cannot mount /tmp"; '
    'r=/tmp/root; mkdir "$r" && mount -n -t tmpfs -o mode=0755,size=16m seal "$r" '
    '|| fail "cannot make the new root"; '
    f'for d in {" ".join(LINUX_READS)}; do '
    'if [ -L "$d" ]; then cp -P "$d" "$r$d" || fail "cannot link $d"; '
    'elif [ -d "$d" ]; then mkdir -p "$r$d" && mount -n --rbind "$d" "$r$d" '
    '&& mount -n -o remount,bind,ro "$r$d" || fail "cannot bind $d"; fi; done; '
    'mkdir -p "$r/dev" "$r/proc" "$r/tmp" "$r$ws" || fail "cannot make the mount points"; '
    'for n in null zero random urandom; do touch "$r/dev/$n" '
    '&& mount -n --bind "/dev/$n" "$r/dev/$n" || fail "cannot bind /dev/$n"; done; '
    'ln -s /proc/self/fd "$r/dev/fd" && ln -s /proc/self/fd/0 "$r/dev/stdin" '
    '&& ln -s /proc/self/fd/1 "$r/dev/stdout" && ln -s /proc/self/fd/2 "$r/dev/stderr" '
    '|| fail "cannot link /dev/fd"; '
    'mount -n -t proc proc "$r/proc" || fail "cannot mount /proc"; '
    'mount -n -t tmpfs -o mode=1777,size=64m seal "$r/tmp" || fail "cannot mount /tmp"; '
    'mount -n --bind /proc/self/fd/3 "$r$ws" || fail "cannot bind the workspace"; '
    'exec 3<&-; '
    'cd "$r" && mkdir .old && pivot_root . .old || fail "cannot pivot to the new root"; '
    'cd / && umount -n -l /.old && rmdir /.old || fail "cannot detach the old root"; '
    'run="/usr/bin/unshare --user --mount --map-user=$uid --map-group=$gid --"; '
    + _EXEC
)

#: What a sealed macOS command may read besides its workspace: the system's programs and
#: libraries, the command-line tools, Homebrew's, and the files the shell itself opens.
MAC_READS = ("/usr", "/bin", "/sbin", "/System", "/Library/Developer/CommandLineTools",
             "/opt/homebrew", "/private/etc", "/private/var/select")
#: Inside those, what holds a person's or a service's data rather than programs.
MAC_DATA = ("/usr/local/var", "/usr/local/etc", "/opt/homebrew/var", "/opt/homebrew/etc")
#: The devices a sealed macOS command may open; ``/dev/fd`` is its own descriptors.
MAC_DEVICES = ("/dev/null", "/dev/zero", "/dev/random", "/dev/urandom")
#: The system settings a sealed macOS command may read (`sysctl`): the machine's shape,
#: never another process's arguments and environment (``kern.procargs2``).
MAC_SYSCTLS = ('(sysctl-name-prefix "hw.") (sysctl-name-prefix "kern.os") '
               '(sysctl-name-prefix "machdep.cpu.") (sysctl-name "kern.hostname") '
               '(sysctl-name "kern.version") (sysctl-name "kern.argmax") '
               '(sysctl-name "kern.maxfilesperproc") (sysctl-name "kern.usrstack64") '
               '(sysctl-name "kern.secure_kernel") (sysctl-name "kern.boottime") '
               '(sysctl-name "kern.ngroups") (sysctl-name "sysctl.proc_cputype")')

#: The shim that runs the command inside the profile sandbox-exec applies (macOS).
#: Arguments: the workspace, then whatever DSH appends.
_MAC_SHIM = 'ws="$1"; shift; run=; ' + _EXEC


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
    data = " ".join(f"(subpath {snippet._sbpl(path)})" for path in MAC_DATA)
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
            f"(deny file-read* (with no-report) {data})\n"
            + (f"(allow file-read-metadata {metadata})\n" if metadata else "")
            + f'(allow file-write* (literal "/dev/null") (regex #"^/dev/fd/") '
              f'(subpath {snippet._sbpl(ws)}))\n')


def runner_command(workspace: Path) -> list[str]:
    """The runner argv DSH's local sandbox provider puts in front of each command of a
    sealed step (``runnerCommand``)."""
    if sys.platform == "darwin":
        ws = _check(workspace, MAC_READS)
        # The reaper runs sandbox-exec, then ends whatever the command left behind.
        reaper = str(Path(__file__).with_name("seal_reaper.py"))
        return [sys.executable, "-I", reaper, ws, snippet.SANDBOX_EXEC, "-p", mac_profile(workspace),
                "/bin/sh", "-c", _MAC_SHIM, "refactory-seal", ws]
    if not sys.platform.startswith("linux"):
        raise SealError(f"no sealed workspace on {sys.platform}")
    ws = _check(workspace, LINUX_READS)
    return [shutil.which("unshare") or "unshare", "-Urnmp", "--fork", "--kill-child",
            "--mount-proc", "--", snippet.BASH, "-c", _SHELL, "refactory-seal", ws,
            str(os.getuid()), str(os.getgid())]


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
