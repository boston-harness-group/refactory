"""Sandboxed builder probe: run named tests in an isolated throwaway clone.

The builder can test its candidate emission before making its final answer,
without touching the target, the run's worktree or the shared ``.git``.
Every isolation guarantee from dm-96 and #368 is preserved.

The probe is a client of the agent-workspace machinery (`clones`, `workspace`): its
clone is made by `clones.clone_shared` and its environment loses the run worktree's
tools as a workspace's does (`workspace.strip_run_paths`). A builder that works in a
workspace (design mode's) tests there directly and needs no probe; the design probe
(`run_design_probe`) still uses it.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from refactory_dsh import sandbox, worktree
from refactory_dsh.worktree import (
    BRANCH_PREFIX, NO_HOOKS, COMMITTER, Worktree,
    _json, _write_json, _inside,
)
from refactory_dsh.check_digest import (
    build_probe_command, digest_check_output, format_digest,
)


@dataclass
class ProbeResult:
    status: str  # "pass", "fail", "timed_out", "refused", "error"
    digest: str = ""
    refusal: str = ""
    raw_tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "digest": self.digest,
                "refusal": self.refusal, "raw_tail": self.raw_tail}


# ── emission matching ─────────────────────────────────────────────────────────

def _emission_matches_last(workdir: Path, emission_content: str) -> bool:
    """Whether the emission matches the last materialised one (the applied_to case).

    Uses the same three-part condition as _commit_emission:
    - head differs from base;
    - the digests are equal;
    - head^ equals attempt_on.
    """
    from refactory_dsh.repo_change import EMISSION
    found = worktree.recorded(workdir)
    if found is None:
        return False
    if found.head == found.base:
        return False
    last = _json(Path(workdir) / EMISSION)
    if not last:
        return False
    digest = hashlib.sha256(emission_content.encode()).hexdigest()
    if last.get("digest") != digest:
        return False
    try:
        parent = worktree.git(Path(found.path), "rev-parse", f"{found.head}^").strip()
    except (ValueError, subprocess.SubprocessError):
        return False
    return parent == last.get("attempt_on")


# ── apply emission ────────────────────────────────────────────────────────────

def _apply_emission_to_clone(clone_root: Path, files: dict[Path, str], deletes: tuple[Path, ...]) -> None:
    """Write the emission's files into a clone, with .git and symlink guards."""
    for relative in (*files, *deletes):
        if any(part.lower() == ".git" for part in relative.parts):
            raise worktree.AttemptRefused(
                f"{relative} is inside git's own state, not a file of the artifact; leave it out"
            )
        _inside(clone_root, relative)
    for relative in deletes:
        target = clone_root / relative
        if target.is_file():
            target.unlink()
    for relative, content in files.items():
        destination = _inside(clone_root, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)


# ── PATH stripping ────────────────────────────────────────────────────────────

def _strip_worktree_path(env: dict[str, str], run_worktree_path: str) -> dict[str, str]:
    """Strip the run worktree's .venv/bin and node_modules/.bin from PATH (`workspace.strip_run_paths`)."""
    from refactory_dsh.workspace import strip_run_paths

    return strip_run_paths(env, run_worktree_path)


# ── warm probe ────────────────────────────────────────────────────────────────

_warm_enabled: bool = False


def _warm_key(head: str, target: str, setup: str, generate: str) -> str:
    """Cache key for a warm probe clone: changes when the attempt identity changes."""
    return hashlib.sha256(
        (head + "\0" + target + "\0" + setup + "\0" + generate).encode()
    ).hexdigest()


class CloneNotRemoved(OSError):
    """A warm clone is still on disk after its removal; the message names it and why."""


def sweep_warm_clones(workdir: Path) -> None:
    """Remove all warm probe clones for this run.

    Called by materialise (``refactory/refactory_dsh/repo_change.py``, the step
    right after emit) once the builder's turn is over, and at the run's end: by
    the walker (``refactory/refactory_dsh/nn/executor.py``, through the
    ``refactory.run_end`` entry point), by a spine run's owner
    (``refactory/refactory_dsh/interactive.py``) and by the DSH evaluator
    (``refactory/refactory_dsh/dsh_runtime.py``). Cold probes are cleaned by
    ``sweep_dead_probes``; warm clones are identified by having a ``cache.key``
    file.

    A clone whose lock another probe holds is in use and is left. One that cannot
    be removed raises :class:`CloneNotRemoved` naming it, after every other clone
    was tried, so the caller's warning says which clone is left on disk.
    """
    refactory_dir = Path(workdir) / ".refactory"
    if not refactory_dir.is_dir():
        return
    left: list[str] = []
    for entry in sorted(refactory_dir.iterdir()):
        if not entry.name.startswith("probe-") or not entry.is_dir():
            continue
        if not (entry / "cache.key").is_file():
            continue
        try:
            _try_evict_clone(entry, refactory_dir)
        except CloneNotRemoved as exc:
            left.append(str(exc))
    if left:
        raise CloneNotRemoved("; ".join(left))


def _remove_tree(path: Path) -> list[str]:
    """Remove ``path`` as far as it can be removed; what could not be, as messages."""
    failures: list[str] = []

    def record(_function: Any, _name: str, error: Any) -> None:
        failures.append(str(error[1] if isinstance(error, tuple) else error))

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=record)
    else:  # pragma: no cover - Python 3.11 names the handler onerror
        shutil.rmtree(path, onerror=record)
    return failures


def _try_evict_clone(probe_dir: Path, refactory_dir: Path) -> bool:
    """Evict a warm clone: take its lock without blocking, then remove it.

    Returns True when the clone is gone, False when another probe holds its lock
    (in use, so it is left). Raises :class:`CloneNotRemoved` naming the clone when
    it is still on disk afterwards.
    """
    try:
        fd = open(probe_dir / "probe.lock", "w")
    except OSError as exc:
        if not probe_dir.exists():
            return True  # another sweep removed it first
        raise CloneNotRemoved(f"could not remove warm clone {probe_dir}: {exc}") from exc
    with fd:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        except OSError as exc:
            raise CloneNotRemoved(f"could not remove warm clone {probe_dir}: {exc}") from exc
        failures = _remove_tree(probe_dir)
        if failures:
            raise CloneNotRemoved(f"could not remove warm clone {probe_dir}: {failures[0]}")
        probe_id = probe_dir.name.removeprefix("probe-")
        (refactory_dir / "slot" / f"run-probe-{probe_id}.json").unlink(missing_ok=True)
    return True


def _do_cold_clone(probe_dir: Path, clone_dir: Path, found: Worktree, remaining: float) -> None:
    """Create a cold clone of the target for probing, at the run's attempt.

    The agent-workspace clone (`clones.clone_shared`): the target's objects shared, its
    own refs, no remote, no hooks.
    """
    from refactory_dsh.clones import clone_shared

    probe_dir.mkdir(parents=True, exist_ok=True)
    clone_shared(Path(found.target), clone_dir, found.head, timeout=max(1.0, min(60, remaining)))


def _stage_emission(clone_dir: Path, clone_base: str, emission: str | None,
                    workdir: Path, found: Worktree) -> str:
    """Apply the emission to the clone, stage and commit.  Returns clone_head."""
    clone_head = clone_base
    if emission is not None:
        emission_path = workdir / emission
        if emission_path.is_file():
            from refactory_dsh.repo_change import apply_emission
            emission_content = emission_path.read_text()
            files_dict, deletes, _rev, _repeated = apply_emission(
                emission_content, found, workdir)
            if files_dict or deletes:
                _apply_emission_to_clone(clone_dir, files_dict, deletes)
                all_paths = [str(p) for p in files_dict]
                if all_paths:
                    worktree.git(clone_dir, "add", "-f", "--", *all_paths)
                for d in deletes:
                    dp = clone_dir / d
                    if dp.exists():
                        worktree.git(clone_dir, "rm", "-q", "-f", "--",
                                     str(d))
                worktree.git(
                    clone_dir, *NO_HOOKS, "-c", "commit.gpgsign=false",
                    "commit", "-q", "--no-verify", "--allow-empty",
                    "-m", "probe candidate",
                    env={**os.environ, **COMMITTER},
                )
                clone_head = worktree.git(
                    clone_dir, "rev-parse", "HEAD").strip()
    return clone_head


# ── sweep ─────────────────────────────────────────────────────────────────────

def sweep_dead_probes(workdir: Path) -> None:
    """Kill orphaned probe processes and clean up their directories.

    Only sweeps probes whose owner.pid is dead.  A concurrent probe whose owner
    is still alive is left alone.
    """
    refactory_dir = Path(workdir) / ".refactory"
    if not refactory_dir.is_dir():
        return
    for entry in sorted(refactory_dir.iterdir()):
        if not entry.name.startswith("probe-") or not entry.is_dir():
            continue
        # Skip warm clones — cleaned up by sweep_warm_clones in materialise
        if (entry / "cache.key").is_file():
            continue
        owner_file = entry / "owner.pid"
        if not owner_file.is_file():
            shutil.rmtree(entry, ignore_errors=True)
            continue
        try:
            owner_pid = int(owner_file.read_text().strip())
        except (ValueError, OSError):
            shutil.rmtree(entry, ignore_errors=True)
            continue
        if _pid_alive(owner_pid):
            continue
        # Owner is dead — kill any test process group, then clean up
        pgid_file = entry / "pgid"
        if pgid_file.is_file():
            try:
                pgid = int(pgid_file.read_text().strip())
                os.killpg(pgid, signal.SIGKILL)
            except (ValueError, OSError, ProcessLookupError):
                pass
        shutil.rmtree(entry, ignore_errors=True)

    # Sweep leaked slot/run-probe-*.json records
    slot_dir = refactory_dir / "slot"
    if slot_dir.is_dir():
        for slot_file in sorted(slot_dir.iterdir()):
            if not slot_file.name.startswith("run-probe-"):
                continue
            try:
                record = json.loads(slot_file.read_text())
                pid = record.get("owner_pid")
                if pid and not _pid_alive(int(pid)):
                    slot_file.unlink(missing_ok=True)
            except (json.JSONDecodeError, ValueError, OSError):
                slot_file.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    """Whether a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


# ── prepared clone ────────────────────────────────────────────────────────────

@contextmanager
def prepared_clone(
    workdir: Path,
    *,
    emission: str | None = None,
    deadline: float,
    test_ids: list[str] | None = None,
    warm: bool = False,
) -> Iterator[tuple[Path, dict[str, str], Worktree]]:
    """Context manager: create an isolated throwaway clone, prepare it, yield it, clean up.

    Yields (clone_root, env, clone_found) where env has PATH, HOME etc set
    for running commands in the clone.

    When *warm* is True the clone is kept on disk after the context exits and
    reused by subsequent probes whose cache key matches.  Concurrent probes
    are serialised by ``fcntl.flock`` on the clone directory.  Warm clones
    are cleaned up by ``sweep_warm_clones``, called from materialise and at
    the run's end.
    """
    workdir = Path(workdir)
    lock_fd = None
    warm_hit = False
    probe_id = uuid.uuid4().hex[:12]
    probe_dir = workdir / ".refactory" / f"probe-{probe_id}"
    clone_dir = probe_dir / "repo"
    home_dir = probe_dir / "home"
    dsh_home_dir = probe_dir / "dsh_home"
    found = None

    if not warm:
        # Cold path: record owner PID for sweep_dead_probes
        probe_dir.mkdir(parents=True, exist_ok=True)
        (probe_dir / "owner.pid").write_text(str(os.getpid()))

    try:
        # Find the run worktree
        found = worktree.recorded(workdir)
        if found is None:
            raise ValueError("no worktree recorded for this run; probe requires a worktree")
        wt_path = Path(found.path)
        if not wt_path.is_dir():
            raise ValueError(f"run worktree {wt_path} does not exist")
        head = worktree.git(wt_path, "rev-parse", "HEAD").strip()
        if head != found.head:
            raise ValueError(
                f"worktree HEAD {head[:12]} differs from recorded {found.head[:12]}; "
                "refusing probe to avoid stale state"
            )

        # Handle repeated-emission case
        if emission is not None:
            emission_path = workdir / emission
            if emission_path.is_file():
                emission_content = emission_path.read_text()
                if _emission_matches_last(workdir, emission_content):
                    raise _RepeatedEmission(
                        "this emission has already been checked; "
                        "see check-record.json for the result"
                    )

        start = time.monotonic()

        if warm:
            # ── warm path: find or create a reusable clone ───────────
            from refactory_dsh.run import _read_facts as _rf
            facts_for_key = _rf(workdir)
            cache_key = _warm_key(
                found.head, found.target,
                str(facts_for_key.get("setup", "")),
                str(facts_for_key.get("generate", "")),
            )
            refactory_dir = workdir / ".refactory"
            refactory_dir.mkdir(parents=True, exist_ok=True)

            # Global creation lock — serialises clone lookup and creation
            creation_lock_path = refactory_dir / "probe-warm.lock"
            creation_lock_fd = open(creation_lock_path, "w")
            fcntl.flock(creation_lock_fd, fcntl.LOCK_EX)
            try:
                # Scan for matching or stale warm clones
                for entry in sorted(refactory_dir.iterdir()):
                    if not entry.name.startswith("probe-") or not entry.is_dir():
                        continue
                    key_file = entry / "cache.key"
                    if not key_file.is_file():
                        continue
                    if key_file.read_text().strip() == cache_key:
                        probe_dir = entry
                        probe_id = entry.name.removeprefix("probe-")
                        clone_dir = probe_dir / "repo"
                        home_dir = probe_dir / "home"
                        dsh_home_dir = probe_dir / "dsh_home"
                        warm_hit = True
                    else:
                        # Stale eviction — lock-safe (fix 6). One it cannot
                        # remove is left for the next sweep, which names it.
                        try:
                            _try_evict_clone(entry, refactory_dir)
                        except CloneNotRemoved:
                            pass

                if not warm_hit:
                    # Cold clone under the creation lock
                    _do_cold_clone(probe_dir, clone_dir, found, deadline)
                    clone_base = worktree.git(clone_dir, "rev-parse", "HEAD").strip()
                    # Write clone_base FIRST, cache.key LAST (fix 3A)
                    (probe_dir / "clone_base").write_text(clone_base)
                    (probe_dir / "cache.key").write_text(cache_key)
                    # Slot record for the new warm clone
                    slot_dir = workdir / ".refactory" / "slot"
                    slot_dir.mkdir(parents=True, exist_ok=True)
                    _write_json(slot_dir / f"run-probe-{probe_id}.json", {
                        "target": found.target, "path": str(clone_dir),
                        "branch": f"{BRANCH_PREFIX}probe-{probe_id}",
                        "base": clone_base, "head": clone_base,
                        "run": str(workdir), "owner_pid": os.getpid(),
                    })

                target_probe_dir = probe_dir
            finally:
                # Release the creation lock BEFORE per-clone lock (fix 6C)
                fcntl.flock(creation_lock_fd, fcntl.LOCK_UN)
                creation_lock_fd.close()

            # Acquire per-clone lock outside the creation lock.
            # If the directory was removed between creation-lock release and
            # here (TOCTOU), fall through to the cold-clone path below.
            try:
                lock_fd = open(target_probe_dir / "probe.lock", "w")
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            except FileNotFoundError:
                lock_fd = None

            # TOCTOU guard: clone may have been evicted between lock release
            # and per-clone lock acquisition (fix 6C)
            if lock_fd is None or not (target_probe_dir / "cache.key").is_file():
                # Clone was evicted — create a fresh cold clone
                if lock_fd is not None:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        lock_fd.close()
                    except OSError:
                        pass
                    lock_fd = None
                probe_id = uuid.uuid4().hex[:12]
                probe_dir = workdir / ".refactory" / f"probe-{probe_id}"
                clone_dir = probe_dir / "repo"
                home_dir = probe_dir / "home"
                dsh_home_dir = probe_dir / "dsh_home"
                _do_cold_clone(probe_dir, clone_dir, found,
                               max(0, deadline - (time.monotonic() - start)))
                clone_base = worktree.git(clone_dir, "rev-parse", "HEAD").strip()
                (probe_dir / "clone_base").write_text(clone_base)
                (probe_dir / "cache.key").write_text(cache_key)
                slot_dir = workdir / ".refactory" / "slot"
                slot_dir.mkdir(parents=True, exist_ok=True)
                _write_json(slot_dir / f"run-probe-{probe_id}.json", {
                    "target": found.target, "path": str(clone_dir),
                    "branch": f"{BRANCH_PREFIX}probe-{probe_id}",
                    "base": clone_base, "head": clone_base,
                    "run": str(workdir), "owner_pid": os.getpid(),
                })
                lock_fd = open(probe_dir / "probe.lock", "w")
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                warm_hit = False

            # Warm-hit operations wrapped in try/except for cold fallback (fix 3B)
            try:
                if warm_hit:
                    clone_base = (probe_dir / "clone_base").read_text().strip()
                    worktree.git(clone_dir, "reset", "--hard", clone_base)
                    worktree.git(clone_dir, "clean", "-fdx",
                                 "--exclude=.venv", "--exclude=node_modules")
                    # Fix 1: delete install records so ensure_environment re-runs
                    for stale_record in dsh_home_dir.glob(
                            "worktrees/*/run-probe-*.json"):
                        try:
                            stale_record.unlink()
                        except OSError:
                            pass
                else:
                    clone_base = worktree.git(
                        clone_dir, "rev-parse", "HEAD").strip()

                clone_head = _stage_emission(
                    clone_dir, clone_base, emission, workdir, found)
            except worktree.AttemptRefused:
                # The builder's emission is refused before it touches the clone
                # (apply_emission and the .git and symlink guards run first), so
                # the clone is not broken: keep it warm for the next probe.
                raise
            except Exception:
                # Evict the broken clone and fall back to a cold clone (fix 3B)
                if lock_fd is not None:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        lock_fd.close()
                    except OSError:
                        pass
                    lock_fd = None
                # If eviction fails (clone locked by another process, or not
                # removable), a second clone with the same cache.key is
                # created; the orphan is cleaned by the next sweep.
                try:
                    _try_evict_clone(probe_dir, refactory_dir)
                except CloneNotRemoved:
                    pass
                probe_id = uuid.uuid4().hex[:12]
                probe_dir = workdir / ".refactory" / f"probe-{probe_id}"
                clone_dir = probe_dir / "repo"
                home_dir = probe_dir / "home"
                dsh_home_dir = probe_dir / "dsh_home"
                _do_cold_clone(probe_dir, clone_dir, found,
                               max(0, deadline - (time.monotonic() - start)))
                clone_base = worktree.git(
                    clone_dir, "rev-parse", "HEAD").strip()
                (probe_dir / "clone_base").write_text(clone_base)
                (probe_dir / "cache.key").write_text(cache_key)
                slot_dir = workdir / ".refactory" / "slot"
                slot_dir.mkdir(parents=True, exist_ok=True)
                _write_json(slot_dir / f"run-probe-{probe_id}.json", {
                    "target": found.target, "path": str(clone_dir),
                    "branch": f"{BRANCH_PREFIX}probe-{probe_id}",
                    "base": clone_base, "head": clone_base,
                    "run": str(workdir), "owner_pid": os.getpid(),
                })
                lock_fd = open(probe_dir / "probe.lock", "w")
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                warm_hit = False
                clone_head = _stage_emission(
                    clone_dir, clone_base, emission, workdir, found)

        else:
            # ── cold path (warm=False) ───────────────────────────────
            _do_cold_clone(probe_dir, clone_dir, found, deadline)
            clone_base = worktree.git(clone_dir, "rev-parse", "HEAD").strip()
            clone_head = _stage_emission(
                clone_dir, clone_base, emission, workdir, found)

        # Create synthetic worktree record
        probe_branch = f"{BRANCH_PREFIX}probe-{probe_id}"
        clone_found = Worktree(
            target=found.target,
            path=str(clone_dir),
            branch=probe_branch,
            base=clone_base,
            head=clone_head,
            run=str(workdir),
        )
        if not warm:
            slot_dir = workdir / ".refactory" / "slot"
            slot_dir.mkdir(parents=True, exist_ok=True)
            record_path = slot_dir / f"run-probe-{probe_id}.json"
            from dataclasses import asdict
            _write_json(record_path, {**asdict(clone_found), "owner_pid": os.getpid()})

        # Prepare environment using the factored helper
        home_dir.mkdir(parents=True, exist_ok=True)
        dsh_home_dir.mkdir(parents=True, exist_ok=True)

        elapsed = time.monotonic() - start
        remaining = deadline - elapsed
        if remaining <= 0:
            raise TimeoutError("probe deadline exceeded during clone")

        from refactory_dsh.repo_change import prepare_check_environment
        from refactory_dsh.run import _read_facts
        facts = _read_facts(workdir)

        # Point REFACTORY_HOME inside the probe directory so that
        # ensure_environment writes its slot/run-probe-*.json record
        # there instead of the shared ~/.refactory.  Restored in finally.
        saved_refactory_home = os.environ.get("REFACTORY_HOME")
        os.environ["REFACTORY_HOME"] = str(dsh_home_dir)
        try:
            clone_found, env, problems, prep_fail = prepare_check_environment(
                clone_found, facts,
                amend=False, commit_lockfiles=False,
                run=workdir, home=home_dir, timeout=remaining,
            )
        finally:
            if saved_refactory_home is not None:
                os.environ["REFACTORY_HOME"] = saved_refactory_home
            else:
                os.environ.pop("REFACTORY_HOME", None)
        if problems:
            raise ValueError(f"probe environment preparation failed: {'; '.join(problems)}")

        # Strip run worktree paths and set probe-specific env vars
        env = _strip_worktree_path(env, found.path)
        env["DSH_HOME"] = str(dsh_home_dir)
        env["REFACTORY_HOME"] = str(dsh_home_dir)
        env["LEFTHOOK"] = "0"
        env["HUSKY"] = "0"

        yield clone_dir, env, clone_found

    except _RepeatedEmission as e:
        raise
    finally:
        if warm:
            # Release the per-clone flock but keep the clone on disk
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                except OSError:
                    pass
        else:
            # Cold path: clean up everything
            if probe_dir.is_dir():
                shutil.rmtree(probe_dir, ignore_errors=True)
            slot_path = workdir / ".refactory" / "slot" / f"run-probe-{probe_id}.json"
            slot_path.unlink(missing_ok=True)
            try:
                from refactory_dsh.paths import refactory_home as _rh
                _shared_slot = worktree.slot(Path(found.target)) if found is not None else None
                if _shared_slot is not None:
                    _leaked = _shared_slot / f"run-probe-{probe_id}.json"
                    _leaked.unlink(missing_ok=True)
            except Exception:
                pass


class _RepeatedEmission(Exception):
    """The emission matches the last materialised one."""


class _FactsProxy:
    """Minimal proxy to pass facts to prepare_commands (which reads ctx.facts)."""
    def __init__(self, facts: dict[str, Any]) -> None:
        self.facts = facts

    def knob(self, name: str, default: Any = None) -> Any:
        return default


# ── run_probe ─────────────────────────────────────────────────────────────────

def run_probe(
    workdir: Path,
    emission_path: str,
    test_ids: list[str],
    timeout: float = 90,
) -> ProbeResult:
    """Run named tests on the candidate emission in an isolated clone.

    Returns a ProbeResult with a digest of the output (not raw stdout).
    """
    workdir = Path(workdir)

    # Check for repeated emission first — if the emission already went through
    # the full check, there is no point building a targeted command or cloning.
    emission_file = workdir / emission_path
    if emission_file.is_file():
        if _emission_matches_last(workdir, emission_file.read_text()):
            return ProbeResult(
                status="refused",
                refusal="this emission has already been checked; "
                        "see check-record.json for the result",
            )

    # Build the targeted command
    facts_path = workdir / "facts.json"
    if facts_path.is_file():
        facts = json.loads(facts_path.read_text())
    else:
        facts = {}
    check_command = str(facts.get("check") or "")
    if not check_command:
        return ProbeResult(status="error", refusal="no check command found in facts")

    from refactory_dsh import worktree as _wt
    _found_wt = _wt.recorded(workdir)
    _probe_root = Path(_found_wt.path) if _found_wt else None
    command, refusal = build_probe_command(check_command, test_ids, root=_probe_root)
    if refusal:
        return ProbeResult(status="refused", refusal=refusal)

    deadline = timeout
    try:
        with prepared_clone(workdir, emission=emission_path, deadline=deadline, test_ids=test_ids, warm=_warm_enabled) as (clone_root, env, clone_found):
            remaining = max(10, deadline - 30)  # reserve some time for cleanup
            # Record process group
            probe_dir = clone_root.parent
            result = sandbox.run_command(
                command, workdir=clone_root, timeout_s=remaining,
                env=env, home=Path(env.get("HOME", str(probe_dir / "home"))),
                exact_env=True,
            )
            output = "\n".join(filter(None, [result.stdout, result.stderr]))
            if result.timed_out:
                return ProbeResult(
                    status="timed_out",
                    digest="the probe timed out before the tests finished",
                    raw_tail=output[-500:] if output else "",
                )
            digest = digest_check_output(output)
            formatted = format_digest(digest)
            return ProbeResult(
                status="pass" if result.ok else "fail",
                digest=formatted[:2000],
                raw_tail=output[-500:] if output else "",
            )
    except _RepeatedEmission as e:
        return ProbeResult(status="refused", refusal=str(e))
    except worktree.AttemptRefused as e:
        return ProbeResult(status="refused", refusal=str(e))
    except TimeoutError as e:
        return ProbeResult(status="timed_out", digest=str(e))
    except Exception as e:
        return ProbeResult(status="error", refusal=f"{type(e).__name__}: {e}")


# ── design probe ─────────────────────────────────────────────────────────────

def run_design_probe(
    workdir: Path,
    *,
    command: str,
    timeout: float = 30,
) -> tuple[ProbeResult, str]:
    """Run a command inside a throwaway clone for design-phase probing.

    Like run_probe but with no emission: the clone is at the run's base.
    Records the probe result to probes.jsonl so claims can cite a verifiable id.
    """
    workdir = Path(workdir)
    probe_id = uuid.uuid4().hex[:12]
    try:
        with prepared_clone(workdir, emission=None, deadline=timeout) as (clone_root, env, clone_found):
            remaining = max(5, timeout - 10)
            probe_dir = clone_root.parent
            result = sandbox.run_command(
                command, workdir=clone_root, timeout_s=int(remaining),
                env=env, home=Path(env.get("HOME", str(probe_dir / "home"))),
                exact_env=True,
            )
            output = "\n".join(filter(None, [result.stdout, result.stderr]))
            if result.timed_out:
                status = "timed_out"
                digest_text = "the probe timed out"
            else:
                status = "pass" if result.ok else "fail"
                digest_text = format_digest(digest_check_output(output))[:2000]
            probe_result = ProbeResult(
                status=status,
                digest=digest_text,
                raw_tail=output[-500:] if output else "",
            )
    except TimeoutError as e:
        status = "timed_out"
        probe_result = ProbeResult(status=status, digest=str(e))
    except Exception as e:
        status = "error"
        probe_result = ProbeResult(status=status, refusal=f"{type(e).__name__}: {e}")
    # Record the probe result to probes.jsonl (outside the context manager,
    # so the probe directory is cleaned up first, but the record persists).
    probes_path = workdir / ".refactory" / "probes.jsonl"
    probes_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"id": probe_id, "command": command, "status": probe_result.status}
    with probes_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return probe_result, probe_id


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sandboxed builder probe")
    parser.prog = "builder_probe"
    parser.add_argument("--workdir", required=True, help="Run directory (absolute)")
    parser.add_argument("--timeout", type=float, default=90, help="Timeout in seconds")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test", nargs="+", dest="test_ids", help="Test IDs to run")
    group.add_argument("--command", dest="command", help="Shell command to run in the clone")

    parser.add_argument("--emission", default=None, help="Emission file name (required with --test)")

    args = parser.parse_args()

    if args.command is not None:
        if args.emission is not None:
            parser.error("--emission cannot be used with --command")
        probe_result, probe_id = run_design_probe(
            Path(args.workdir),
            command=args.command,
            timeout=args.timeout,
        )
        output = {**probe_result.to_dict(), "id": probe_id}
        print(json.dumps(output))
    else:
        _warm_enabled = True
        emission = args.emission if args.emission is not None else "emit.txt"
        result = run_probe(
            Path(args.workdir),
            emission,
            args.test_ids,
            timeout=args.timeout,
        )
        print(json.dumps(result.to_dict()))
