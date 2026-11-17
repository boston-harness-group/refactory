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
  a capped systemd user scope where one can be made, :func:`linux_scope`);
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

#: A sealed command's limits, as ``ulimit`` takes them: CPU seconds and the largest file
#: (KiB). macOS counts processes per user, not per sandbox, and does not enforce an
#: address-space limit, so it gets only these.
LIMITS = "-t 3600 -f 4194304"
#: Linux, where each command runs in a systemd user scope (:func:`linux_scope`): the
#: scope caps the memory and processes of the command and everything it starts
#: together (:data:`SCOPE_LIMITS`), so the processes (counted in the command's own user
#: namespace) need only a loose per-process bound.
LINUX_LIMITS = "-u 512"
SCOPE_LIMITS = ("MemoryMax=8G", "MemorySwapMax=0", "TasksMax=256")
#: Linux without a user scope: no cap on the whole, so each process gets a tight one
#: (128 processes, 8 GiB of address space each).
LINUX_LIMITS_UNSCOPED = "-u 128 -v 8388608"

#: Closes every descriptor above stderr, then checks that none is left (a command that
#: would start with one does not start), then runs its arguments. The last step before
#: the command, inside every namespace the command runs in.
_CLOSE = ('for fd in /dev/fd/*; do n="${fd##*/}"; '
          '[ "$n" -gt 2 ] 2>/dev/null && eval "exec $n>&-" 2>/dev/null; done; '
          'for fd in /dev/fd/*; do n="${fd##*/}"; '
          'if [ "$n" -gt 2 ] 2>/dev/null && [ -e "$fd" ]; then '
          'echo "refactory-seal: descriptor $n is still open" >&2; exit 97; fi; done; '
          'exec "$@"')

#: Sets the limits, then runs the command (after DSH's profile arguments and ``--``) in
#: the workspace with that environment, through :data:`_CLOSE`; ``$run`` is what runs
#: it (Linux: the nested, unprivileged user namespace).
_EXEC = (f'ulimit {LIMITS} $limits || {{ echo "refactory-seal: cannot set limits" >&2; exit 97; }}; '
         'while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do shift; done; '
         '[ "$1" = "--" ] || { echo "refactory-seal: no command" >&2; exit 97; }; shift; '
         'cd "$ws" || exit 97; '
         f'exec $run /bin/bash -c \'{_CLOSE}\' refactory-seal '
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
#: CUDA's included); everything else there is covered (`linux_shell`).
LOCAL_KEPT = ("bin", "sbin", "lib", "lib64", "libexec", "include", "cuda", "cuda-*")

def linux_shell(reads: tuple[str, ...], limits: str) -> str:
    """The shell that builds the new root inside the new namespaces (Linux), where it is
    root of its own user namespace, from ``reads``, with ``limits``. Arguments: the
    workspace, the caller's uid and gid, then whatever DSH appends. Any step that fails
    stops it: nothing runs half sealed."""
    return (
        'ws="$1"; uid="$2"; gid="$3"; shift 3; '
        'fail() { echo "refactory-seal: $1" >&2; exit 97; }; '
        'exec 3<"$ws" || fail "cannot open the workspace"; '
        # One more user namespace (the unprivileged one the command runs in) and none after
        # it: the command cannot become root of a namespace of its own.
        'echo 1 > /proc/sys/user/max_user_namespaces || fail "cannot limit user namespaces"; '
        'mount -n -t tmpfs -o mode=0755,size=1m seal /tmp || fail "cannot mount /tmp"; '
        'r=/tmp/root; mkdir "$r" && mount -n -t tmpfs -o mode=0755,size=16m seal "$r" '
        '|| fail "cannot make the new root"; '
        # Each bound alone (--bind, not --rbind), so nothing mounted inside it comes along
        # with flags of its own, then made read only.
        'mkdir -p "$r/etc" || fail "cannot make /etc"; '
        f'for d in {" ".join(reads)}; do '
        'if [ -L "$d" ]; then cp -P "$d" "$r$d" || fail "cannot link $d"; '
        'elif [ -e "$d" ]; then { if [ -d "$d" ]; then mkdir -p "$r$d"; else touch "$r$d"; fi; } '
        '&& mount -n --bind "$d" "$r$d" '
        '&& mount -n -o remount,bind,ro "$r$d" || fail "cannot bind $d"; fi; done; '
        # /usr/local keeps only its programs and libraries (:data:`LOCAL_KEPT`), as on macOS:
        # its etc, share, src, var and the rest are each covered by an empty, read-only mount.
        'for e in "$r"/usr/local/* "$r"/usr/local/.[!.]*; do '
        '{ [ -e "$e" ] && [ ! -L "$e" ]; } || continue; case "${e##*/}" in '
        f'{"|".join(LOCAL_KEPT)}) continue;; esac; '
        'if [ -d "$e" ]; then mount -n -t tmpfs -o ro,size=4k,mode=0555 seal "$e"; '
        'else mount -n --bind /dev/null "$e" && mount -n -o remount,bind,ro "$e"; fi '
        '|| fail "cannot cover $e"; done; '
        'mkdir -p "$r/dev" "$r/proc" "$r/tmp" || fail "cannot make the mount points"; '
        'for n in null zero random urandom; do touch "$r/dev/$n" '
        '&& mount -n --bind "/dev/$n" "$r/dev/$n" || fail "cannot bind /dev/$n"; done; '
        'ln -s /proc/self/fd "$r/dev/fd" && ln -s /proc/self/fd/0 "$r/dev/stdin" '
        '&& ln -s /proc/self/fd/1 "$r/dev/stdout" && ln -s /proc/self/fd/2 "$r/dev/stderr" '
        '|| fail "cannot link /dev/fd"; '
        'mount -n -t proc proc "$r/proc" || fail "cannot mount /proc"; '
        'mkdir -p "$r$ws" && mount -n --bind /proc/self/fd/3 "$r$ws" || fail "cannot bind the workspace"; '
        'exec 3<&-; '
        'cd "$r" && mkdir .old && pivot_root . .old || fail "cannot pivot to the new root"; '
        'cd / && umount -n -l /.old && rmdir /.old || fail "cannot detach the old root"; '
        # The new root itself (/, /tmp, the folders above the workspace) is read only, as
        # macOS's is: the workspace is the one place a sealed command writes.
        'mount -n -o remount,bind,ro / || fail "cannot make the new root read only"; '
        'run="/usr/bin/unshare --user --mount --map-user=$uid --map-group=$gid --"; '
        f'limits="{limits}"; '
        + _EXEC
    )

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
_MAC_SHIM = 'ws="$1"; shift; run=; limits=; ' + _EXEC


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
    ws = _check(workspace, LINUX_READS + LINUX_ETC)
    if os.getuid() == 0:
        # Root here is root inside, which reads what only root may (/etc/shadow and the
        # like): a sealed step runs as an ordinary user, or not at all.
        raise SealError("a sealed step does not run as root")
    scope = linux_scope()
    # Started through the reaper's --exec: a fresh session keyring, and killed with DSH.
    reaper = str(Path(__file__).with_name("seal_reaper.py"))
    return [sys.executable, "-I", reaper, "--exec", *scope,
            shutil.which("unshare") or "unshare", "-Urnmp", "--fork", "--kill-child",
            "--mount-proc", "--", snippet.BASH, "-c",
            linux_shell(LINUX_READS + LINUX_ETC, LINUX_LIMITS if scope else LINUX_LIMITS_UNSCOPED),
            "refactory-seal", ws, str(os.getuid()), str(os.getgid())]


def _scope_prefix() -> list[str]:
    systemd_run = shutil.which("systemd-run")
    runtime = f"/run/user/{os.getuid()}"
    if not systemd_run or not os.path.isdir(runtime):
        return []
    return ["/usr/bin/env", f"XDG_RUNTIME_DIR={runtime}",
            f"DBUS_SESSION_BUS_ADDRESS=unix:path={runtime}/bus", systemd_run, "--user",
            "--scope", "--quiet", "--collect", *(f"--property={p}" for p in SCOPE_LIMITS), "--"]


#: The scope prefix once a probe found one (`linux_scope`); a failed probe is not kept.
_SCOPE: list[tuple[str, ...]] = []


def linux_scope() -> tuple[str, ...]:
    """What puts a sealed Linux command in a systemd user scope capped at
    :data:`SCOPE_LIMITS`, or ``()`` where there is no user manager to make one (the
    command then gets :data:`LINUX_LIMITS_UNSCOPED`, with a warning each time). A probe
    that finds its own memory cap is kept for the process; a failed one is tried again
    next time, so one slow probe does not drop every later step to the fallback."""
    if _SCOPE:
        return _SCOPE[0]
    prefix = _scope_prefix()
    found: tuple[str, ...] = ()
    if prefix:
        try:
            probe = subprocess.run(
                [*prefix, "/bin/sh", "-c",
                 'cat "/sys/fs/cgroup$(sed -n "s/^0:://p" /proc/self/cgroup)/memory.max"'],
                capture_output=True, text=True, timeout=20)
            if probe.returncode == 0 and probe.stdout.strip() == str(8 << 30):
                found = tuple(prefix)
        except (OSError, subprocess.SubprocessError):
            pass
    if found:
        _SCOPE.append(found)
    else:
        print("refactory: warning: a sealed step's command runs without a capped systemd user "
              f"scope here, under per-process limits only ({LINUX_LIMITS_UNSCOPED})",
              file=sys.stderr)
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
