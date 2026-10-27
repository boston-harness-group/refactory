"""The steps a mode uses to change a repository: where it works, what lands, how it is checked.

A mode that writes code into a repository -- design and design_fast, or one you write --
attaches these rather than wiring its own (#351):

- `knobs`, `knob_names` and `produces` are what the mode's prepare node declares, so a run
  names its `target`, `base`, `worktree`, `setup` and `generate`.
- `locate`, called from the mode's own prepare op, gives the run a git worktree of its
  target on its own branch, from a pinned base -- a new project when there is no target --
  or, with `worktree` off, the artifact directory. Its `emit_scope` fact is the text the
  mode's emit prompt includes (``{{emit_scope}}``): where to write, and in which blocks.
- `materialise_node` applies ``emit.txt`` as the run's next attempt. An emission it cannot
  apply is recorded as ``emission_refused``, which `refused` reads for the mode's check, so
  the reason goes back to emit instead of ending the run.
- `run_check` runs a command where the run works, in the worktree's own environment. The
  node that calls it declares `CHECK_WRITES` and `CHECK_PRODUCES` too: an install that
  writes a lockfile amends the attempt, so the check moves what materialise reported.
- `report` settles the run's place, writes the mode's report with it, and only then records
  the run as succeeded, for the mode's report op.

The agent node that writes ``emit.txt`` is the mode's own. Name it ``emit``, so the
target guard holds it to its declared output (`OUTPUT_ONLY`, #319).

Nothing here reaches outside the run: the work stays on the run's branch. Delivering it as
a pull request is design's (`delivery`), behind an approval gate whose question names the
delivery, because that question is the person's consent to it (#350).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

from refactory_dsh import projects, sandbox, target_guard, worktree
from refactory_dsh.nn import Knob, Node
from refactory_dsh.paths import project_for_nodes
from refactory_dsh.worktree import AttemptRefused, _alive, _json
from refactory_dsh.emitted_paths import HIDDEN_FORMAT, _unprintable, unprintable_refusal  # noqa: F401

#: Where the steps' own ops live, for the nodes that call them.
OPS = "refactory_dsh.repo_change"

#: Project-owned software artifacts are separate from discoverable mode packages.
ARTIFACTS = Path(".factory") / "artifacts"

#: Which run owns each artifact name. Kept beside the artifacts rather than in them,
#: because materialise replaces an artifact directory whole.
OWNERS = Path(".factory") / "artifact-owners"

#: Spine states in which a run no longer writes its artifact. A failed run is among
#: them so a retry under the same name can start; if the failed run resumes, it is
#: refused before it writes or checks the artifact. `refused` is not: the spine writes
#: it when a gate question could not be put, and the gate is still due.
FINISHED = frozenset({"done", "halted", "failed"})

#: The file emit writes its answer into. Materialise reads it; the mode declarations
#: name it in emit's ``writes``; agent_delivery moves it aside before a re-dispatch.
EMIT_FILE = "emit.txt"

#: The agent node whose whole answer is `emit.txt`. Materialise writes the artifact
#: from it, so a change it makes to the target directly bypasses materialise, the
#: check and the reviews (#319); the agent-delivery bracket refuses that.
OUTPUT_ONLY = ("emit",)

#: The path portion of an emission header: any character except whitespace,
#: '=', backtick, single quote, and double quote.  Structural safety
#: (absolute, .., backslash, control chars) is validated separately by
#: _artifact_paths after parsing.
EMISSION_PATH = r"[^\s=`'\"]+"

#: The blocks `emit.txt` is written in: a whole file, edits to an existing one, or its
#: removal. Edits are what keep a small change to a long file small (#330 I4).
BLOCK = re.compile(rf"^===(FILE|EDIT|DELETE):\s*({EMISSION_PATH})\s*===\s*$", re.MULTILINE)

#: The lines that frame one edit in an `===EDIT:` block: text copied from the file, and
#: what replaces it.
SEARCH, DIVIDER, REPLACE = "<<<<<<< SEARCH", "=======", ">>>>>>> REPLACE"

#: What the read tool leaves where it cut text short. Copied into a file, it cuts the
#: file short too: one such file passed every gate because nothing ran it (#330 I4).
TRUNCATED = re.compile(r"\.\.\. \(line truncated to \d+ chars\)$"
                       r"|^\(Output capped\. Showing lines \d+-\d+\. Use offset=\d+ to continue\.\)$"
                       r"|^\(Showing lines \d+-\d+ of \d+\. Use offset=\d+ to continue\.\)$",
                       re.MULTILINE)

#: One emission's edits: for each file, the lines to find and the lines put in their place.
Edits = dict[Path, list[tuple[str, str]]]

#: Which emission the run's last attempt applied, on top of which commit and to which
#: commit's files, so the same emission given again applies to those files rather than
#: to its own result.
EMISSION = Path(".refactory") / "emission.json"

#: What a run's facts say in place of a worktree, so a prompt that names one never shows
#: a raw placeholder when the run has none.
NO_WORKTREE = "(this run has no worktree: `worktree` is off)"

#: What a node that calls `run_check` writes and returns besides its own: an install that
#: writes a lockfile amends the attempt (`worktree.ensure_environment`), and a generate
#: command amends its tracked-file changes too, so the check rewrites ``changes.diff``
#: and moves the ``commit`` and ``files`` facts to the amended one. The base comparison
#: (`base_check`) writes ``test-guardedness.json`` and names the failures it excused as
#: ``inherited_failures``, empty when it excused none.
from refactory_dsh.invocations import CHECK_WRITES, CHECK_PRODUCES  # noqa: E402


# ── what a mode declares ──────────────────────────────────────────────────────
def knobs() -> dict[str, Knob]:
    """The knobs that say where a run works.

    All frozen: they say what is being built, so no search may move them.
    """
    return {
        "target": Knob("", "str", frozen=True,
                       description="the git repository to change; empty starts a new project "
                                   "in the projects folder"),
        "base": Knob("", "str", frozen=True,
                     description="the commit, branch or tag the run starts from (default: the "
                                 "target's HEAD, or a new project's main)"),
        "worktree": Knob(True, "bool", frozen=True,
                         description="work in a worktree of the target on the run's own branch; "
                                     "off writes the artifact under .factory/artifacts instead"),
        "setup": Knob("", "str", frozen=True,
                      description="a command run in the worktree before each check, after its "
                                  "dependencies are installed, such as a build"),
        "generate": Knob("", "str", frozen=True,
                         description="a command run in the worktree after dependencies are installed, "
                                     "before setup and each check; whatever it changes in tracked files "
                                     "is amended into the attempt. A generator that itself needs a build "
                                     "includes that build in its own command"),
        "include_uncommitted": Knob(False, "bool", frozen=True,
                                    description="start from the target's uncommitted changes instead "
                                                "of only committed HEAD; copies modified, staged, "
                                                "untracked and deleted files into the worktree and "
                                                "commits them as the run's base"),
    }


def knob_names() -> tuple[str, ...]:
    """The knobs `knobs` declares, for the prepare node's ``knobs``."""
    return tuple(knobs())


def produces() -> tuple[str, ...]:
    """The facts `locate` returns, for the prepare node's ``produces``."""
    return ("target", "worktree", "branch", "base", "source", "check_root", "emit_scope", "setup", "generate", "probe_instruction",
            "design_probe_instruction")


def materialise_node(**fields: Any) -> Node:
    """The op that applies ``emit.txt`` as the run's next attempt (`materialise`)."""
    return Node(**{
        "id": "materialise", "kind": "op", "call": f"{OPS}:materialise",
        "reads": (EMIT_FILE,), "writes": ("changes.diff", "changed-functions.json"),
        "produces": ("artifact", "files", "commit", "change", "emission_refused",
                     "changed_functions", "consecutive_refusals"),
        "notes": "commit the emitted files on the run's branch, or write the artifact directory",
        **fields,
    })


# ── where the run works ───────────────────────────────────────────────────────
def _artifact_name(name: str) -> str:
    if (not name or name in {".", ".."} or name != name.strip()
            or any(char in name for char in "/\\:")
            or any(ord(char) < 32 for char in name)):
        raise ValueError("artifact name must be one directory name, without separators, drives or '..'")
    return name


def _project_dir(project: Path, *parts: Path) -> None:
    """Refuse a project path that crosses a symlink or is not a directory."""
    for path in parts:
        if path.is_symlink() or not path.resolve().is_relative_to(project):
            raise ValueError(f"artifact path must not cross a symlink: {path}")
        if path.exists() and not path.is_dir():
            raise ValueError(f"artifact path is not a directory: {path}")


def _artifact_root(name: str) -> Path:
    """Use the configured project, never the lowered operation's working directory."""
    project = project_for_nodes().expanduser().resolve()
    root = project / ARTIFACTS / _artifact_name(name)
    _project_dir(project, project / ".factory", project / ARTIFACTS, root)
    return root


def _still_open(run: Path, claimant: Any = None) -> str | None:
    """What ``run`` is doing while it may still write its artifact; None once it cannot.

    A run that reported (``design.json``) or whose directory is gone is over. A run the
    spine drives says so in its status, unless the process the spine checkpoint names
    as its owner has exited (a terminal CEO that finished, a GUI that was restarted). A
    walked run is over once its progress records an error, or once ``claimant``, the
    walking process that claimed the name, has exited (Ctrl-C records no error). A run
    that left neither record never got far enough to matter.
    """
    if not run.is_dir() or (run / "design.json").exists():
        return None
    spine = _json(run / ".refactory" / "spine-status.json")
    # A walked run writes the status file too, but no checkpoint names its owner: its
    # liveness is the claimant's, or else the walking process its status file names,
    # read from its progress below.
    from refactory_dsh.run_state import owner_pid, walked

    if spine and not walked(run):
        status = str(spine.get("status") or "")
        owner = _json(run / ".refactory" / "spine-checkpoint.json").get("owner_pid")
        if status in FINISHED or not _alive(owner):
            return None
        return f"status {status!r} since {spine.get('at')}"
    progress = _json(run / ".refactory" / "progress.json")
    walker = claimant if claimant is not None else owner_pid(run)
    if progress and not progress.get("error") and _alive(walker):
        return f"walking, last at {progress.get('node')!r}"
    return None


def _claim(name: str, run: Path) -> None:
    """Own the artifact ``name`` for ``run``, or refuse while another open run owns it.

    Two runs with one name share `.factory/artifacts/<name>`, so one run's materialise
    rewrote the other's artifact mid-review (#330 K4). Claimed at prepare, and again at
    materialise: a run that resumes after another took its name must find out before it
    overwrites that run's work.
    """
    project = project_for_nodes().expanduser().resolve()
    _project_dir(project, project / ".factory", project / OWNERS)
    path = project / OWNERS / f"{_artifact_name(name)}.json"
    run = Path(run).resolve()
    claim = _json(path)
    owner = str(claim.get("run") or "")
    if owner == str(run) and claim.get("pid") == os.getpid():
        return
    if owner and owner != str(run):
        busy = _still_open(Path(owner), claim.get("pid"))
        if busy is not None:
            raise ValueError(
                f"artifact {name!r} belongs to another open run, {owner} ({busy}): two runs "
                f"with one name overwrite each other's {ARTIFACTS / name}. Choose another name, "
                f"or finish or halt that run first; if it was abandoned, delete {path}."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{run.name}.tmp")
    # The claiming process is the walker itself on the walker, and a short-lived node
    # command on the spine, which `_still_open` reads from its checkpoint instead.
    pending.write_text(json.dumps({"run": str(run), "pid": os.getpid(),
                                   "claimed": datetime.now(timezone.utc).isoformat()}) + "\n")
    pending.replace(path)


def _require_owner(name: str, run: Path) -> None:
    """Refuse when another run has claimed ``name`` since this one wrote the artifact.

    Stricter than :func:`_claim`, which lets a run take a name over from a finished one:
    a step that checks or reports the artifact must not pass off a finished run's files
    as this run's own. A name nobody has claimed is left alone.
    """
    project = project_for_nodes().expanduser().resolve()
    path = project / OWNERS / f"{_artifact_name(name)}.json"
    owner = str(_json(path).get("run") or "")
    if owner and owner != str(Path(run).resolve()):
        raise ValueError(
            f"artifact {name!r} was claimed by another run, {owner}, after this run wrote it, "
            f"so {ARTIFACTS / name} may hold that run's files. Redo emit or materialise to "
            "write this run's artifact again before checking or reporting it."
        )


def _artifact_paths(names: list[str]) -> list[Path]:
    """The emitted paths, refused unless each names a distinct file inside the artifact."""
    checked: list[Path] = []
    for relative in names:
        path = Path(relative)
        if refusal := unprintable_refusal(relative):
            raise AttemptRefused(refusal)
        if (path.is_absolute() or PureWindowsPath(relative).drive
                or "\\" in relative or ".." in path.parts or not path.parts
                or relative.endswith("/")):
            raise AttemptRefused(f"{relative!r} is not a file inside the artifact directory; name it by its "
                                 "path from the artifact's root")
        if path in checked:
            raise AttemptRefused(f"{relative!r} duplicates an artifact file; write it once")
        checked.append(path)
    for path in checked:
        if any(parent in checked for parent in path.parents):
            raise AttemptRefused(f"{str(path)!r} conflicts with an artifact file used as its directory; "
                                 "rename one of them")
    return checked


def _run_id(workdir: Path) -> str:
    """This invocation's id, so the run's branch names the run; random for a walked run."""
    found = str(_json(Path(workdir) / worktree.INVOCATION).get("id") or "")
    return found[:12] if re.fullmatch(r"[0-9a-f]{12,}", found) else uuid.uuid4().hex[:12]


def _open_worktree(ctx: Any, name: str, task: dict[str, Any],
                   output_only: tuple[str, ...] = ()) -> dict[str, Any]:
    """Give the run its own worktree of its target, and say what that means (#351).

    The target is the repository ``target`` names -- relative to the project the run
    belongs to, not to the run directory a node starts in -- or a new project in the
    projects folder when it names none. The run branches from ``base`` -- the target's
    HEAD, or a new project's ``main`` -- pinned to a commit here, and never works in the
    target itself: what the person has there, committed or not, stays theirs.
    """
    launcher = project_for_nodes().expanduser().resolve()
    named = str(task.get("target") or ctx.knob("target", "") or "").strip()
    given = str(task.get("base") or ctx.knob("base", "") or "").strip()
    if not named and given and not (projects.projects_dir() / name).exists():
        raise ValueError(f"base {given!r} names a commit, but this run starts a new project, which has only "
                         "its empty main; pass target=<repository> with it, or leave base empty")
    target = worktree.repository_root(launcher / Path(named).expanduser()) if named else projects.new_project(name)
    rev = given or ("HEAD" if named else "main")
    # Pinned before anything else is touched, so a base that names nothing is refused
    # before an earlier run's worktree is pruned or this one's is made. A redo keeps
    # the base it pinned the first time, even if that branch has moved or gone since.
    run_id = _run_id(ctx.workdir)
    earlier = worktree.recorded(ctx.workdir)
    if earlier is not None and earlier.branch == worktree.BRANCH_PREFIX + run_id:
        base = earlier.base
    else:
        base = worktree.resolve_base(target, rev)
    for note in worktree.prune(target):
        # A worktree kept is one the person may want to look at; one removed is routine.
        if note.startswith("kept "):
            target_guard.warn(note)
        else:
            print(f"refactory: {note}", file=sys.stderr)
    found = worktree.create(target, base, run_id, ctx.workdir)
    # When opted in, copy the person's uncommitted changes into the worktree and
    # commit them as the run's own base, so the attempts' diffs show only the run's
    # change.  The person's checkout is never modified.
    # On a redo of prepare, the branch already has attempt commits; do not re-copy,
    # which would add a second base commit on top of existing attempts and re-pin
    # the base over them.
    uncommitted_entries: list[dict[str, str]] = []
    if ctx.knob("include_uncommitted", False) not in (False, 0, "0", "false"):
        # Guard on the branch having no attempt commits, not on the record
        # existing.  A first prepare on a clean tree writes no record (no
        # entries), so checking "record exists" would let a redo after the
        # person edits files copy them over existing attempt commits.
        has_attempts = found.head != found.base
        if not has_attempts:
            uncommitted_entries = worktree.copy_uncommitted(target, Path(found.path), ctx.workdir)
            # Re-read the worktree record so `found.head` and `found.base` reflect the
            # new commit.
            found = worktree.recorded(ctx.workdir) or found
            if uncommitted_entries:
                print(f"refactory: {len(uncommitted_entries)} uncommitted file(s) copied into the "
                      f"worktree and committed as the run's base", file=sys.stderr)
        else:
            existing_record = worktree.uncommitted_record(found)
            if existing_record:
                uncommitted_entries = existing_record
            print(f"refactory: skipping uncommitted-changes copy on redo "
                  f"(branch has attempt commits)", file=sys.stderr)
    worktree.write_wrapper(found, ctx.workdir)
    # Every step of a worktree run is held to its own worktree (#319, #347).
    # In a worktree run, target changes fail the step; launcher-checkout
    # changes are warnings. Another session or person may be working there.
    warn_dirs: tuple[Path, ...] = ()
    if launcher != target and (launcher / ".git").exists():
        warn_dirs = (launcher,)
    baseline = target_guard.record(ctx.workdir, target,
                                   output_only=(*OUTPUT_ONLY, *output_only),
                                   also=(Path(found.path),),
                                   warn_also=warn_dirs)
    if baseline["changes"] and not uncommitted_entries:
        target_guard.warn(f"{len(baseline['changes'])} uncommitted change(s) in {target} are not part of "
                          f"this run, which starts from commit {found.base[:12]}: "
                          f"{target_guard._named(sorted(baseline['changes']))}")
    elif baseline["changes"] is None:
        target_guard.warn(f"the target {target} has more than {target_guard.LIMIT} uncommitted paths, "
                          "so this run cannot tell whether emit writes to it directly")
    if not named:
        entry = next((project for project in projects.load()["projects"]
                      if project.get("path") == str(target)), {})
        others = [note["branch"] for note in entry.get("runs", []) if note.get("branch") != found.branch]
        if others and not given:
            target_guard.warn(f"{target} has earlier runs, whose work is on their branches, not on main: "
                              f"{target_guard._named(others)}. This run starts from main; pass "
                              "base=<branch> to build on one of them instead.")
    projects.record_run(target, run=ctx.workdir, branch=found.branch, base=found.base, status="running")
    return {"target": str(target), "worktree": found.path, "branch": found.branch, "base": found.base,
            "source": found.path, "check_root": f"the root of the run's worktree, {found.path}",
            "emit_scope": (
                f"Paths are relative to the root of this run's worktree, {found.path}: a git worktree of "
                f"{target} on branch {found.branch}, from the base commit {found.base[:12]}. Each attempt "
                "builds on the last: the worktree holds the base and every earlier attempt, as committed "
                f"(`git -C {found.path} diff {found.base[:12]}` shows the change so far), so read the files "
                "there and change only what this attempt needs. For each file, write one ===FILE: or "
                "===DELETE: block, or one or more ===EDIT: blocks; each block starts with a line of "
                "exactly one of:\n"
                "===FILE: <relative path>=== followed by the whole file, for a new file or one you "
                "rewrite entirely;\n"
                "===EDIT: <relative path>=== followed by one or more edits to an existing file, each "
                f"written as\n{SEARCH}\n<text copied exactly from the file as it is now>\n{DIVIDER}\n"
                f"<the text that replaces it>\n{REPLACE}\nwhere the copied text appears in the file "
                "exactly once;\n"
                "===DELETE: <relative path>=== on its own, to remove a file.\n"
                "Prefer ===EDIT: for a file that exists: retyping a long file risks changing lines you "
                "did not mean to. Never copy the read tool's line numbers, or a note such as "
                "'(line truncated to 2000 chars)', into a block."
            )}


def wants_worktree(ctx: Any) -> bool:
    """Whether the run's ``worktree`` knob asks for a worktree, as it does unless turned off."""
    return ctx.knob("worktree", True) not in (False, 0, "0", "false")


def locate(ctx: Any, name: str, task: dict[str, Any], *, in_worktree: bool | None = None,
           needs_worktree: tuple[str, ...] = (), output_only: tuple[str, ...] = ()) -> dict[str, Any]:
    """Where the run works, as facts for the mode's prepare op to return.

    ``name`` is the run's artifact, and a new project's name. In a worktree unless
    ``in_worktree`` -- by default the ``worktree`` knob -- says otherwise (`_open_worktree`).
    Without one, the run writes the artifact directory, which it claims, and watches the
    project instead; ``target``, ``base``, ``setup`` and ``generate`` are refused then, with
    ``needs_worktree``, the mode's own knobs that only a worktree gives a meaning to.

    ``output_only`` names further agent nodes -- besides ``emit`` -- whose only
    output is their declared file.  The agent-delivery bracket refuses one that
    changed the run's worktree while it ran; in a worktree run, target
    changes are failures and launcher-checkout changes are warnings.
    """
    _artifact_name(name)
    if in_worktree is None:
        in_worktree = wants_worktree(ctx)
    if in_worktree:
        located = _open_worktree(ctx, name, task, output_only=output_only)
    else:
        given = [knob for knob in ("target", "base", "setup", "generate", *needs_worktree)
                 if str(task.get(knob) or ctx.knob(knob, "") or "").strip()]
        if given:
            raise ValueError(f"{', '.join(given)} need the run's worktree, which is off; turn it on, or leave "
                             "them empty to write the artifact under .factory/artifacts")
        # A run without a worktree must not inherit one an earlier run left in this
        # directory; that worktree's own record (`worktree.slot`) still tracks it.
        (Path(ctx.workdir) / worktree.RECORD).unlink(missing_ok=True)
        _claim(name, ctx.workdir)
        # The baseline every later step compares with (#330 I1), and the bracket that
        # holds `emit` to its declared output (#319).
        project = project_for_nodes().expanduser()
        baseline = target_guard.record(ctx.workdir, project,
                                       output_only=(*OUTPUT_ONLY, *output_only))
        if baseline["changes"] is None:
            target_guard.warn(f"the target {baseline['root']} has more than {target_guard.LIMIT} "
                              "files to watch, so this run cannot tell whether it moves or "
                              "whether emit writes to it directly")
        located = {"target": str(project), "worktree": NO_WORKTREE, "branch": NO_WORKTREE,
                   "base": NO_WORKTREE, "source": str(project), "check_root": "the artifact's own directory",
                   "emit_scope": "Paths are relative to the artifact's own directory, which holds "
                                 "only the files you write here. Write each file as a block "
                                 "introduced by a line of exactly ===FILE: <relative path>=== "
                                 "followed by its full contents."}
    # Resolved once, as create resolves its own inputs: the check reads the fact.
    setup = str(task.get("setup") or ctx.knob("setup", "") or "").strip()
    generate = str(task.get("generate") or ctx.knob("generate", "") or "").strip()
    # Probe instruction: available only when there is a worktree
    if in_worktree and located.get("worktree") and located["worktree"] != NO_WORKTREE:
        python = sys.executable
        abs_workdir = str(Path(ctx.workdir).resolve())
        probe_instruction = (
            "\n## Builder probe\n\n"
            "You can test your candidate before making your final emission. "
            "Write emit.txt first, then run the probe to see whether specific "
            "failing tests pass with your change.\n\n"
            "**Do not apply or test your change directly in the target or the "
            "run's worktree.** The probe runs in a throwaway clone that is "
            "cleaned up when the builder's turn ends (materialise) or at the "
            "run's end. It is not an attempt.\n\n"
            "To run the probe:\n"
            "```bash\n"
            f"{python} -m refactory_dsh.builder_probe "
            f"--workdir {abs_workdir} "
            "--emission emit.txt "
            "--test <test_id> [<test_id> ...] "
            "--timeout 90\n"
            "```\n\n"
            "- `--test` takes one or more test identifiers from the digest "
            "(e.g. `tests/test_foo.py::test_bar` for pytest, "
            "`src/foo.test.ts > suite > test` for vitest).\n"
            "- The output is a JSON object with `status`, `digest` and `refusal` fields.\n"
            "- The default timeout is 90 seconds. If the probe times out, retry with "
            "`--timeout <higher>` and pass a matching `timeoutMs` to the bash tool.\n"
            "- A cold-cache probe may time out on the first run; retry.\n"
        )
    else:
        probe_instruction = ""

    # Design probe instruction: for the design writer (design_doc)
    if in_worktree and located.get("worktree") and located["worktree"] != NO_WORKTREE:
        design_probe_instruction = (
            "\n## Design probe\n\n"
            "You can probe the current codebase to verify a mechanism your design "
            "relies on. The probe runs in a throwaway clone at the run's base — "
            "not your proposed changes — with hooks disabled and an isolated HOME.\n\n"
            "To run a probe:\n"
            "```bash\n"
            f"{python} -m refactory_dsh.builder_probe "
            f"--workdir {abs_workdir} "
            '--command "<shell command>" '
            "--timeout 30\n"
            "```\n\n"
            "Examples:\n"
            '- Named test: `--command "python -m pytest tests/test_foo.py::test_bar -x"`\n'
            '- Mode lowering: `--command "python -c \\"from refactory_dsh import lowering; '
            "from refactory_dsh.design import DesignMode; print(lowering.to_workflow(DesignMode().state_dict(), name='design'))\\\"\"` \n"
            '- Python snippet: `--command "python -c \\"import ast; print(ast.dump(...))\\\""`\n\n'
            "- `--command` takes a single shell string (double-quote it).\n"
            "- The output is a JSON object with `id`, `status`, `digest` and `raw_tail` fields.\n"
            "- The `id` is the probe's verifiable identifier. Cite it next to the claim "
            "it supports as `(probe <id>: <status>)` on the claim line itself.\n"
            "- The default timeout is 30 seconds; pass `--timeout <higher>` for slow "
            "operations, and pass a matching `timeoutMs` to the bash tool.\n"
            "- Installs (dependencies) are bounded separately by INSTALL_TIMEOUT_S (30 min), "
            "not by the probe's deadline.\n"
            "- A cold-cache probe may time out on the first run; retry.\n"
            "- The probe's filesystem inside the clone is not restricted; "
            "target_guard refuses and restores a write to the target.\n"
            "- The CLI exits 0 for successful dispatch; argparse errors exit 2.\n"
        )
    else:
        design_probe_instruction = ""

    return {**located, "setup": setup, "generate": generate,
            "probe_instruction": probe_instruction,
            "design_probe_instruction": design_probe_instruction}


# ── what lands: emit.txt, applied ─────────────────────────────────────────────
def _emitted(answer: str, *, edits: bool) -> tuple[dict[Path, str], Edits, tuple[Path, ...]]:
    """emit.txt as whole files, edits to existing files and deletions; refused when malformed.

    ``edits`` says whether this run takes edits and deletions at all: only a worktree has
    a last attempt for them to apply to.
    """
    kinds = "===FILE:, ===EDIT: or ===DELETE:" if edits else "===FILE:"
    headers = [line for line in answer.splitlines() if line.startswith(("===FILE:", "===EDIT:", "===DELETE:"))]
    bad_lines = [line for line in headers if BLOCK.fullmatch(line) is None]
    if bad_lines:
        raise AttemptRefused(
            f"emit.txt has a malformed block header: {bad_lines[0]!r}; "
            f"a header must be exactly {kinds} <relative path>=== "
            "where the path contains no whitespace, '=', backtick, or quotes"
        )
    matches = list(BLOCK.finditer(answer))
    if not matches:
        raise AttemptRefused(f"emit.txt holds no {kinds} blocks; write each file's change as one, as the "
                             "prompt describes, with no other text")
    names = [match.group(2) for match in matches]
    # Several ===EDIT: blocks for the same file are merged; a file named by an
    # ===EDIT: and a ===FILE: or ===DELETE:, or by two ===FILE: blocks, is refused.
    kinds_by_name: dict[str, set[str]] = {}
    for match in matches:
        kinds_by_name.setdefault(match.group(2), set()).add(match.group(1))
    for name, used in kinds_by_name.items():
        if used == {"EDIT"}:
            continue  # several ===EDIT: blocks for one file are applied in order
        if names.count(name) > 1:
            blocks = " and a ".join(f"==={k}:" for k in sorted(used))
            raise AttemptRefused(
                f"emit.txt names {name} in more than one block ({blocks}); "
                "a file may appear in several ===EDIT: blocks, but not in a mix with ===FILE: or ===DELETE:")
    unique_names = list(dict.fromkeys(names))
    paths = {name: path for name, path in zip(unique_names, _artifact_paths(unique_names))}
    files: dict[Path, str] = {}
    changed: Edits = {}
    deletes: list[Path] = []
    for index, match in enumerate(matches):
        path = paths[match.group(2)]
        kind = match.group(1)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(answer)
        body = answer[match.end():end].strip("\n")
        if kind != "FILE" and not edits:
            raise AttemptRefused(f"{path}: this run has no worktree, so its artifact directory holds only "
                                 "whole files; write every file with ===FILE:")
        if kind == "FILE":
            refuse_truncated(path, body)
            files[path] = body + "\n"
        elif kind == "EDIT":
            changed.setdefault(path, []).extend(_pairs(path, body))
        elif body.strip():
            raise AttemptRefused(f"the ===DELETE: block for {path} has text after it; a deletion takes none, "
                                 "so leave the line after the header empty")
        else:
            deletes.append(path)
    return files, changed, tuple(deletes)


def _pairs(path: Path, body: str) -> list[tuple[str, str]]:
    """The edits of one ``===EDIT:`` block, as (lines to find, lines to put in their place)."""
    lines = body.split("\n")
    pairs: list[tuple[str, str]] = []
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        if lines[index] != SEARCH:
            raise AttemptRefused(f"the edit to {path} has text outside its {SEARCH} ... {REPLACE} pairs: "
                                 f"{lines[index][:80]!r}; put every line inside a pair")
        try:
            end = lines.index(REPLACE, index + 1)
        except ValueError:
            raise AttemptRefused(f"an edit to {path} is missing its {REPLACE} line; close every {SEARCH} "
                                 f"with {DIVIDER} and {REPLACE}") from None
        dividers = [at for at in range(index + 1, end) if lines[at] == DIVIDER]
        if len(dividers) != 1:
            raise AttemptRefused(
                f"an edit to {path} has {len(dividers)} {DIVIDER} lines, so where its search text ends is "
                + ("missing; put one between the search and the replacement lines" if not dividers else
                   f"unclear; leave a line of exactly {DIVIDER} out of the edit, searching for the lines next "
                   "to it, or write the file whole with ===FILE:"))
        divider = dividers[0]
        search = "".join(line + "\n" for line in lines[index + 1:divider])
        replacement = "".join(line + "\n" for line in lines[divider + 1:end])
        if not search.strip():
            raise AttemptRefused(f"an edit to {path} has nothing to search for; to add a file, write it with "
                                 "===FILE:, and to add lines, search for the lines next to them")
        refuse_truncated(path, replacement)
        pairs.append((search, replacement))
        index = end + 1
    if not pairs:
        raise AttemptRefused(f"the ===EDIT: block for {path} holds no {SEARCH} ... {REPLACE} edits; give it "
                             "at least one, or leave the file out")
    return pairs


def _occurrences(text: str, search: str) -> list[int]:
    """Every position where ``search`` occurs in ``text``, regardless of line boundaries."""
    found: list[int] = []
    at = text.find(search)
    while at != -1:
        found.append(at)
        at = text.find(search, at + 1)
    return found


def _edited(path: Path, text: str, pairs: list[tuple[str, str]], shown: str) -> tuple[str, int]:
    """``text`` with each edit applied in turn, and how many were already in it.

    Each edit is refused unless its search text is there exactly once. Search text
    is matched as exact text anywhere in the file, like the dsh edit tool's old_string.
    :func:`_pairs` ends a search's last line with a newline; when the text with that
    newline is not in the file, it is matched without it, so a search may stop inside
    a line -- a sentence in a paragraph kept on one line.
    The read tool shows a file with ``\r`` stripped and every line ended, so a file
    whose lines all end in ``\r\n``, or whose last line has no newline, is matched as
    shown and written back as it was. An edit whose result is already where it would go
    -- a model re-sending an earlier attempt's edit -- is skipped rather than applied
    twice. ``shown`` is the command that prints ``text``, so a refusal names where to
    copy from.
    """
    crlf = "\r\n" in text and text.count("\r\n") == text.count("\n")
    work = text.replace("\r\n", "\n") if crlf else text
    open_end = not work.endswith("\n")
    work += "\n" if open_end else ""
    skipped = 0
    for search, replacement in pairs:
        starts = _occurrences(work, search)
        within_line = False
        if not starts and search.endswith("\n") and not _already_applied(work, search, replacement, starts):
            inline = _occurrences(work, search[:-1])
            if inline:
                search, starts, within_line = search[:-1], inline, True
                replacement = replacement.removesuffix("\n")
        if _already_applied(work, search, replacement, starts, within_line=within_line):
            skipped += 1
            continue
        if len(starts) != 1:
            first = next((line.strip() for line in search.splitlines() if line.strip()), "")[:80]
            raise AttemptRefused(
                f"an edit to {path} searches for text beginning {first!r}, which "
                + (f"is not in the file the edit applies to; copy it exactly from `{shown}`"
                   if not starts else f"appears {len(starts)} times; include more text around it until it appears once"))
        work = work[:starts[0]] + replacement + work[starts[0] + len(search):]
    if open_end and work.endswith("\n"):
        work = work[:-1]
    return (work.replace("\n", "\r\n") if crlf else work), skipped


def _already_applied(text: str, search: str, replacement: str, starts: list[int], *,
                     within_line: bool = False) -> bool:
    """Whether an edit's replacement already stands where it would go, as after a re-sent edit.

    Either the search lines are gone and the replacement is there once, or the search
    lines are there once inside a copy of the replacement (an insertion sent again).
    An edit that removes lines is never taken as applied: nothing shows it was. A
    search matched ``within_line`` looks for its replacement inside lines too.
    """
    if not replacement.strip() or len(starts) > 1:
        return False
    placed = (_occurrences if within_line else _line_starts)(text, replacement)
    if not starts:
        return len(placed) == 1
    return any(at <= starts[0] and starts[0] + len(search) <= at + len(replacement) for at in placed)


def _line_starts(text: str, lines: str) -> list[int]:
    """Where ``lines`` occur in ``text`` starting at a line's start."""
    found: list[int] = []
    at = text.find(lines)
    while at != -1:
        if at == 0 or text[at - 1] == "\n":
            found.append(at)
        at = text.find(lines, at + 1)
    return found


def refuse_truncated(path: Path, text: str) -> None:
    """Refuse text copied from a read that the read tool cut short, before it is written anywhere."""
    match = TRUNCATED.search(text)
    if match:
        line = text.count("\n", 0, match.start()) + 1
        raise AttemptRefused(
            f"{path} would contain, on line {line}, the read tool's note that it cut text short: "
            f"{match.group(0)!r}. Text copied from that read is incomplete, so the file would be too. Write "
            "only text that was shown in full: copy a file's text from the file itself, or, writing "
            "===EDIT: blocks, leave out the lines you could not read whole.")


def _count_refusal(ctx: Any, refused: Exception) -> dict[str, Any]:
    """Increment the consecutive-refusal counter and raise ``BuildLoopStop`` at the bound.

    Called from both ``repo_change.materialise`` and ``create_ops.materialise``
    (the shared helper the brief asks for).  A refusal still spends no attempt
    (#330 E5); the bound is separate.
    """
    count = int(ctx.facts.get("consecutive_refusals", 0)) + 1
    ctx.set_fact("consecutive_refusals", count)
    bound = ctx.knob("refusal_attempts", 3)
    if count >= bound:
        from refactory_dsh.run import BuildLoopStop

        raise BuildLoopStop(
            f"emissions refused {count} consecutive times: {refused}"
        )
    return {"emission_refused": str(refused)}


def materialise(ctx: Any, *, source: Path | None = None, files: tuple[str, ...] = ()) -> dict[str, Any]:
    """Make the emitted files the run's artifact: a commit in its worktree, when it has one.

    Each attempt is one commit on the run's branch (`worktree.commit_attempt`), building
    on the last: whole files, edits to files as the last attempt committed them, and
    deletions. ``changes.diff`` holds the branch's diff from its base, which is what the
    reviewers judge. A run without a worktree writes the artifact directory, as it always
    has: from the emitted files alone, or, given ``source``, from its ``files`` with the
    emitted files over them (design_fast's `source_project`). An emission that cannot be
    applied -- a malformed block, an edit whose text is not in its file, text copied from a
    truncated read, or an emission that changes nothing because every file and edit is
    already in the last attempt -- does not end the run: it is recorded as
    ``emission_refused``, and the check sends the reason back to emit. So does an emission
    derived from an agent's workspace whose change the blocks could not carry (a binary
    file, a symlink, a mode change), which the workspace's collect step refused
    (`workspace.refusal_for`).
    """
    # Sweep dead probes and warm clones before proceeding. A warm clone left on
    # disk is named in a warning; it does not stop the attempt.
    from refactory_dsh.builder_probe import CloneNotRemoved, sweep_dead_probes, sweep_warm_clones
    sweep_dead_probes(ctx.workdir)
    try:
        sweep_warm_clones(ctx.workdir)
    except CloneNotRemoved as exc:
        import logging
        logging.getLogger(__name__).warning("materialise: %s", exc)
    try:
        found = worktree.recorded(ctx.workdir)
        # Cleared only once set, so a run whose graph predates the fact still materialises.
        cleared = {"emission_refused": ""} if ctx.facts.get("emission_refused") else {}
        # Like emission_refused: reset only when set, so a run lowered before the
        # refusal count existed still materialises.
        reset_refusals = {"consecutive_refusals": 0} if ctx.facts.get("consecutive_refusals") else {}
        from refactory_dsh.workspace import refusal_for
        if ctx.exists(EMIT_FILE):
            collected = refusal_for(ctx.workdir, EMIT_FILE, ctx.read_text(EMIT_FILE))
            if collected is not None:
                raise AttemptRefused(collected)
        if found is None:
            change = (f"The artifact directory is a copy of {source} with this run's files applied."
                      if source is not None else "The artifact directory holds only the files this run wrote.")
            return {**_materialise_directory(ctx, source, files), "commit": NO_WORKTREE, **cleared,
                    **reset_refusals, "change": change}
        return {**_commit_emission(ctx), **cleared, **reset_refusals}
    except AttemptRefused as refused:
        return _count_refusal(ctx, refused)


def apply_emission(answer: str, found: worktree.Worktree, workdir: Path, *,
                   from_worktree: bool = False) -> tuple[dict[Path, str], tuple[Path, ...], str, list[str]]:
    """Parse emit.txt and resolve edits against the run worktree, read-only.

    Returns (files, deletes, rev, repeated) where files maps relative paths to content,
    deletes lists paths to remove, rev is the revision the emission was
    applied against, and repeated lists paths whose edits were already applied.
    The .git refusal and _inside symlink refusal are applied.

    ``from_worktree`` reads each edited file from the worktree instead of the
    commit. Materialise passes it while a merge is in progress, when the worktree
    holds main's changes for cleanly merged files and the markers for conflicted
    ones; reading the commit would silently revert main's change.
    """
    files_raw, changed, deletes = _emitted(answer, edits=True)
    path = Path(found.path)
    digest_val = hashlib.sha256(answer.encode()).hexdigest()
    rev = found.head
    last = _json(Path(workdir) / EMISSION)
    if _is_repeated_emission(found, digest_val, last):
        rev = str(last["applied_to"])
        deletes = tuple(name for name in deletes if worktree.committed_text(found, name) is not None)

    repeated: list[str] = []
    for relative, pairs in changed.items():
        if from_worktree:
            current_path = path / relative
            current = current_path.read_text() if current_path.exists() else None
        else:
            current = worktree.committed_text(found, relative, rev)
        if current is None:
            raise AttemptRefused(f"{relative} is not a file of the last attempt, so it cannot be edited; "
                                 "write it whole with ===FILE:")
        files_raw[relative], skipped = _edited(relative, current, pairs, f"git -C {found.path} show {rev[:12]}:{relative}")
        if skipped:
            repeated.append(str(relative))

    # Apply .git and _inside guards
    for relative in (*files_raw, *deletes):
        if any(part.lower() == ".git" for part in relative.parts):
            raise AttemptRefused(f"{relative} is inside git's own state, not a file of the artifact; leave it out")
        worktree._inside(path, relative)

    return files_raw, deletes, rev, repeated


def _is_repeated_emission(found: worktree.Worktree, digest: str, last: dict[str, Any]) -> bool:
    """Whether the emission matches the last materialised one."""
    if found.head == found.base:
        return False
    if not last or last.get("digest") != digest:
        return False
    try:
        parent = worktree.git(Path(found.path), "rev-parse", f"{found.head}^").strip()
    except (ValueError, OSError):
        return False
    return parent == last.get("attempt_on")


def _commit_emission(ctx: Any) -> dict[str, Any]:
    """Apply emit.txt to the run's worktree as its next attempt; one a prune removed is added back."""
    found = worktree.require_current(ctx.workdir)
    answer = ctx.read_text(EMIT_FILE)
    path = Path(found.path)
    digest = hashlib.sha256(answer.encode()).hexdigest()
    # The paths the emission names, before anything is applied.
    files, changed, _ = _emitted(answer, edits=True)
    # ── Start a pending merge if materialise needs to commit one ──
    # When the delivery gate recorded a pending merge (pending-merge.json) and
    # MERGE_HEAD is not present, start the merge now.  The merge exists only
    # inside this function: on ANY failure, the except/finally blocks abort it.
    pending_merge_path = Path(ctx.workdir) / ".refactory" / "pending-merge.json"
    pending_merge_data = None
    if pending_merge_path.exists():
        try:
            pending_merge_data = json.loads(pending_merge_path.read_text())
        except (OSError, ValueError):
            pass
    merge_in_progress = False
    try:
        worktree.git(path, "rev-parse", "--verify", "-q", "MERGE_HEAD")
        merge_in_progress = True
    except ValueError:
        pass
    if not merge_in_progress and pending_merge_data:
        target_tip = str(pending_merge_data.get("target_tip", ""))
        pm_conflict_files = pending_merge_data.get("conflict_files", [])
        if target_tip:
            merge_env = {**os.environ, **worktree.COMMITTER}
            try:
                worktree.git(path, *worktree.NO_HOOKS, "-c", "commit.gpgsign=false",
                             "merge", "--no-commit", "--no-verify", target_tip,
                             env=merge_env)
            except ValueError:
                pass  # expected — conflicts
            # Check if merge is actually in progress now.
            try:
                worktree.git(path, "rev-parse", "--verify", "-q", "MERGE_HEAD")
                merge_in_progress = True
            except ValueError:
                pass  # clean fast-forward — no MERGE_HEAD
            # Resolve generated-file conflicts to theirs (main's side).
            if merge_in_progress:
                try:
                    _pm_conflict_output = worktree.git(path, "diff", "--name-only", "--diff-filter=U")
                except ValueError:
                    _pm_conflict_output = ""
                _pm_unmerged = [f for f in _pm_conflict_output.strip().splitlines() if f.strip()]
                _pm_authored_set = set(pm_conflict_files)
                for _pm_cfile in _pm_unmerged:
                    if _pm_cfile not in _pm_authored_set:
                        try:
                            worktree.git(path, *worktree.NO_HOOKS, "checkout", "--theirs", "--", _pm_cfile)
                            worktree.git(path, *worktree.NO_HOOKS, "add", "--", _pm_cfile)
                        except ValueError:
                            pass
    # During a merge-in-progress, edits must apply to the worktree text for
    # EVERY path, not only unmerged ones.  A cleanly-merged file contains main's
    # changes in the worktree but not in the committed text; reading from the
    # commit would silently revert main's change.
    unmerged_paths: set[str] = set()
    if merge_in_progress:
        try:
            unmerged_output = worktree.git(path, "diff", "--name-only", "--diff-filter=U")
            unmerged_paths = {f.strip() for f in unmerged_output.splitlines() if f.strip()}
        except ValueError:
            pass
        # Before writing anything, ensure every unmerged path appears in the emission
        # (as either ===FILE: or ===EDIT:).  A partial emission would stage stray files
        # and leave unmerged entries to be committed with markers.
        emission_paths = {
            (relative.as_posix() if hasattr(relative, 'as_posix') else str(relative))
            for relative in (*files.keys(), *changed.keys())
        }
        missing_unmerged = sorted(unmerged_paths - emission_paths)
        if missing_unmerged:
            # Abort the merge so the worktree is never left mid-merge.
            try:
                worktree.git(path, "merge", "--abort")
            except ValueError:
                pass
            (Path(ctx.workdir) / ".refactory" / "pending-merge.json").unlink(missing_ok=True)
            raise AttemptRefused(
                f"merge in progress but the emission does not resolve all conflicted files; "
                f"missing: {', '.join(missing_unmerged)}. Every conflicted file must appear "
                f"in the emission as ===FILE: or ===EDIT:."
            )

    try:
        files, deletes, rev, repeated = apply_emission(
            answer, found, ctx.workdir, from_worktree=merge_in_progress)
    except BaseException:
        # Abort any merge so the worktree is never left mid-merge on failure.
        if merge_in_progress:
            try:
                worktree.git(path, "rev-parse", "--verify", "-q", "MERGE_HEAD")
                worktree.git(path, "merge", "--abort")
            except ValueError:
                pass
            (Path(ctx.workdir) / ".refactory" / "pending-merge.json").unlink(missing_ok=True)
        raise
    if repeated:
        target_guard.warn(f"emit re-sent edits the last attempt already holds, so they were skipped rather than "
                          f"applied twice: {target_guard._named(repeated)}")
    # ── Conflict-marker detection: refuse emissions that leave markers ──
    _CONFLICT_MARKER = re.compile(r'^<{7}(?! SEARCH).*$|^>{7}(?! REPLACE).*$', re.MULTILINE)
    marker_files = []
    for relative, content in files.items():
        if _CONFLICT_MARKER.search(content):
            marker_files.append(str(relative))
    if marker_files:
        # Abort any merge so the worktree is never left mid-merge.
        if merge_in_progress:
            try:
                worktree.git(path, "merge", "--abort")
            except ValueError:
                pass
            (Path(ctx.workdir) / ".refactory" / "pending-merge.json").unlink(missing_ok=True)
        raise AttemptRefused(
            f"the emission leaves git conflict markers in: {', '.join(marker_files)}; "
            "resolve the conflicts and remove the <<<<<<< / >>>>>>> lines"
        )
    # No drift warning: the run's base is pinned, so a moving checkout changes nothing it sees.
    attempt = len(worktree.git(path, "rev-list", f"{found.base}..HEAD").split()) + 1
    # An emission after a self-check fix-up (`design_ops._self_check_fixup`) finishes the
    # attempt the self-check sent back, so it is amended into it, and the branch keeps one
    # commit an attempt.
    fixup = bool(int(ctx.facts.get("self_check_fixup_count") or 0))
    try:
        advanced = worktree.commit_attempt(ctx.workdir, files, f"{ctx.facts['name']}: attempt {attempt}",
                                           deletes, amending=fixup)
    except BaseException:
        # On any commit failure during a merge, abort so the worktree is clean.
        if merge_in_progress:
            try:
                worktree.git(path, "rev-parse", "--verify", "-q", "MERGE_HEAD")
                worktree.git(path, "merge", "--abort")
            except ValueError:
                pass
            (Path(ctx.workdir) / ".refactory" / "pending-merge.json").unlink(missing_ok=True)
        raise
    # Clean up pending-merge record after a successful merge commit.
    pending_merge_path.unlink(missing_ok=True)
    worktree._write_json(Path(ctx.workdir) / EMISSION, {"digest": digest, "attempt_on": found.head,
                                                         "applied_to": rev})
    _warn_if_shared_changed(advanced, "since the last check")
    diff_text = worktree.diff(advanced)
    ctx.write_text("changes.diff", diff_text)
    from refactory_dsh.changed_functions import compute_and_write as _cf_write
    cf_formatted = _cf_write(diff_text, Path(advanced.path), ctx.workdir, advanced.base)
    wrapper = Path(ctx.workdir).resolve() / worktree.WRAPPER
    return {"artifact": advanced.path, "files": _branch_files(advanced),
            "commit": advanced.head,
            "changed_functions": cf_formatted,
            "change": (
                f"The artifact is a git worktree of {advanced.target} on branch {advanced.branch}. "
                f"The change under review is `git -C {advanced.path} diff {advanced.base[:12]}`, also "
                f"saved as changes.diff in the run directory. Run commands with `{wrapper} <command>`, "
                "which switches to the worktree directory and runs with a clean, check-equivalent environment (env -i)."
            )}


def _branch_files(found: worktree.Worktree) -> list[str]:
    """The files the run's branch adds or changes from its base, which is what it delivers."""
    changed = worktree.git(Path(found.path), "diff", "--name-only", "--diff-filter=d", found.base, found.head)
    return sorted(changed.splitlines())


def _materialise_directory(ctx: Any, source: Path | None, files: tuple[str, ...]) -> dict[str, Any]:
    """Write the emitted files into the artifact directory, replacing an earlier attempt's.

    Given ``source``, its ``files`` are copied in first; the artifact may not replace it.
    Always writes ``changed-functions.json`` (empty list when no diff exists).
    """
    checked, _, _ = _emitted(ctx.read_text(EMIT_FILE), edits=False)
    root = _artifact_root(str(ctx.facts["name"]))
    _claim(str(ctx.facts["name"]), ctx.workdir)
    target_guard.warn_if_moved(ctx.workdir)
    if source is not None and (root == source or root in source.parents):
        raise ValueError("output artifact may not replace the source project")
    # Validate and finish all writes privately before moving an earlier artifact.
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".design-", dir=root.parent))
    preserve_backup = False
    previous = temporary / "previous"
    try:
        stage = temporary / "artifact"
        stage.mkdir()
        for relative in files if source is not None else ():
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, target)
        for relative, content in checked.items():
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        replaced = root.exists()
        if replaced:
            root.rename(previous)
        try:
            stage.rename(root)
        except BaseException:
            if replaced:
                try:
                    previous.rename(root)
                except OSError as error:
                    preserve_backup = True
                    raise RuntimeError(f"artifact replacement failed; previous artifact retained at {previous}") from error
            raise
    finally:
        if not preserve_backup and not (previous.exists() and not root.exists()):
            shutil.rmtree(temporary)
    # Write changed-functions.json as [] for directory-mode (no worktree, no diff).
    import json as _json
    (ctx.workdir / "changed-functions.json").write_text(_json.dumps([]) + "\n")
    return {"artifact": str(root), "files": sorted(set(files) | {str(path) for path in checked}),
            "changed_functions": "(none)"}


# ── how it is checked, and reported ───────────────────────────────────────────
def _warn_if_shared_changed(found: worktree.Worktree, when: str) -> None:
    """Report a change to the git configuration or hooks the worktree shares with its target."""
    if worktree.shared_changed(found):
        target_guard.warn(f"the git configuration or hooks the run's worktree shares with {found.target} "
                          f"changed {when}; check them before your next commit there")


def prepare_commands(ctx: Any, found: worktree.Worktree, env: dict[str, str], home: Path, timeout: float,
                     budget: Any, *, amend: bool = True) -> tuple[worktree.Worktree, list[str], dict[str, Any] | None]:
    """Run the run's generate command, then its setup command, in a prepared worktree.

    Generated files are inputs to what setup builds and checks, so generate comes first.
    With ``amend`` (the attempt's worktree), what generate rewrites in tracked files is
    amended into the attempt; without it (the base worktree base_check runs), the
    rewrite stays local. Returns the worktree, at its new head when amended, the
    problems that stop the check, and the prep-failure dict of the first failing step.
    """
    problems: list[str] = []
    prep_fail: dict[str, Any] | None = None
    generate = str(ctx.facts.get("generate") or "").strip()
    if generate:
        prepared = sandbox.run_command(generate, workdir=Path(found.path), timeout_s=timeout,
                                       budget=budget, env=env, home=home, exact_env=True)
        if not prepared.ok:
            detail_gen = "\n".join((prepared.problem(), prepared.stdout or "", prepared.stderr or "")).strip()
            crashed_gen, sig_label_gen = sandbox.is_signal_crash(prepared)
            if crashed_gen and sig_label_gen:
                problems.append(f"the generate command `{generate}` {sig_label_gen}: {detail_gen}")
            else:
                problems.append(f"the generate command `{generate}` failed: {detail_gen}")
            gen_rc = prepared.returncode
            if crashed_gen and gen_rc is not None and gen_rc > 0:
                gen_rc = -(gen_rc - 128)
            prep_fail = {"step": generate, "returncode": gen_rc, "timed_out": prepared.timed_out}
        elif amend:
            edits = worktree.tracked_edits(found)
            if edits:
                found = worktree.amend(found, ctx.workdir, edits, "files written by the generate command")
    setup = str(ctx.facts.get("setup") or "").strip()
    if not problems and setup:
        prepared = sandbox.run_command(setup, workdir=Path(found.path), timeout_s=timeout,
                                       budget=budget, env=env, home=home, exact_env=True)
        if not prepared.ok:
            detail_setup = "\n".join((prepared.problem(), prepared.stdout or "", prepared.stderr or "")).strip()
            crashed_setup, sig_label_setup = sandbox.is_signal_crash(prepared)
            if crashed_setup and sig_label_setup:
                problems.append(f"the setup command `{setup}` {sig_label_setup}: {detail_setup}")
            else:
                problems.append(f"the setup command `{setup}` failed: {detail_setup}")
            setup_rc = prepared.returncode
            if crashed_setup and setup_rc is not None and setup_rc > 0:
                setup_rc = -(setup_rc - 128)
            prep_fail = {"step": setup, "returncode": setup_rc, "timed_out": prepared.timed_out}
    return found, problems, prep_fail


def prepare_check_environment(
    found: worktree.Worktree,
    facts: dict[str, Any],
    *,
    amend: bool,
    commit_lockfiles: bool,
    run: Path,
    home: Path,
    timeout: float,
    budget: Any = None,
    record: Path | None = None,
) -> tuple[worktree.Worktree, dict[str, str], list[str], dict[str, Any] | None]:
    """Install dependencies, build check environment, run generate/setup.

    Factored from _check_in_worktree so the check, the base check, the probe and an
    agent's clone workspace (`clones.prepare`) can all call it.  *amend* and
    *commit_lockfiles* have no default — callers must pass them explicitly. *record*
    is where the install record is kept (`worktree.ensure_environment`).

    Returns (found, env, problems, prep_fail).
    """
    amended, problems, install_fail = worktree.ensure_environment(
        found, run, budget, commit_lockfiles=commit_lockfiles, record=record,
    )
    if amended.head != found.head:
        found = amended
    env = worktree.check_environment(Path(found.path), home)
    prep_fail = install_fail
    if not problems:
        # Build a minimal context proxy for prepare_commands
        _ctx = _PrepCtx(facts, workdir=run)
        found, problems, cmd_fail = prepare_commands(
            _ctx, found, env, home, timeout, budget, amend=amend,
        )
        if cmd_fail:
            prep_fail = cmd_fail
    return found, env, problems, prep_fail


class _PrepCtx:
    """Minimal context proxy for prepare_commands when called from prepare_check_environment."""
    def __init__(self, facts: dict[str, Any], workdir: Path | None = None) -> None:
        self.facts = facts
        self.workdir = workdir

    def knob(self, name: str, default: Any = None) -> Any:
        return default


def _check_home(ctx: Any) -> Path:
    home = Path(ctx.workdir).resolve() / worktree.CHECK_HOME
    home.mkdir(parents=True, exist_ok=True)
    return home


def _prepare_in_worktree(ctx: Any, found: worktree.Worktree, timeout: float
                         ) -> tuple[worktree.Worktree, dict[str, str], Path, list[str], dict[str, Any] | None]:
    """Prepare the environment the check needs in the run's worktree.

    An install that writes a lockfile amends the attempt (`worktree.ensure_environment`),
    and a generate command amends its tracked-file changes too, so the commit fact and
    the reviewers' diff follow them to their new head.
    """
    home = _check_home(ctx)
    head_before = found.head
    found, env, problems, prep_fail = prepare_check_environment(
        found, ctx.facts if hasattr(ctx, 'facts') else {},
        amend=True, commit_lockfiles=True,
        run=ctx.workdir, home=home, timeout=timeout, budget=getattr(ctx, "budget", None),
    )
    worktree.write_wrapper(found, ctx.workdir)
    if found.head != head_before:
        ctx.set_fact("commit", found.head)
        ctx.set_fact("files", _branch_files(found))
        diff_text = worktree.diff(found)
        ctx.write_text("changes.diff", diff_text)
        from refactory_dsh.changed_functions import compute_and_write as _cf_write
        cf_formatted = _cf_write(diff_text, Path(found.path), ctx.workdir, found.base)
        ctx.set_fact("changed_functions", cf_formatted)
    return found, env, home, problems, prep_fail


def _restored_attempt(ctx: Any) -> worktree.Worktree:
    """The run's worktree at the committed attempt, with what a check or reviewer changed put back."""
    found, restored = worktree.restore(ctx.workdir)
    if found.head != str(ctx.facts.get("commit")):
        raise ValueError("the worktree is not at the attempt materialise committed; materialise again before checking")
    if restored:
        target_guard.warn(f"tracked files in the run's worktree had changed since materialise and were put "
                          f"back, so the check runs the committed attempt: {target_guard._named(restored)}")
    return found


def prepare_attempt(ctx: Any, timeout: float) -> list[str]:
    """Bring the attempt to the commit its check and its reviewers both see, before either starts.

    The environment is installed, then the ``generate`` command runs (its tracked-file
    changes amended into the attempt) and the ``setup`` command builds, as `run_check`
    does first; afterwards the commit no longer moves, so a check run with
    ``prepared=True`` can run beside the reviews. Returns the problems that stopped the
    preparation, empty when it worked or when the run has no worktree.
    """
    if worktree.recorded(ctx.workdir) is None:
        return []
    found, _env, _home, problems, _fail = _prepare_in_worktree(ctx, _restored_attempt(ctx), timeout)
    _warn_if_shared_changed(found, "while preparing for the check")
    return problems


#: The run's check copy: a detached worktree of the run's own at the attempt's commit, in
#: the target's worktree slot beside the run's worktree and its base worktree.
CHECK_COPY_PREFIX = "check-"


def _check_copy_path(target: Path, run_id: str) -> Path:
    return worktree.slot(target) / f"{CHECK_COPY_PREFIX}{run_id}"


def _check_copy(ctx: Any, found: worktree.Worktree, timeout: float
                ) -> tuple[worktree.Worktree, dict[str, str], Path, list[str]]:
    """The run's check copy at the attempt's commit, with its environment prepared.

    It is kept across attempts -- moved to the new commit with what the last check left
    in tracked and untracked files put back, its installed environment kept -- and its
    hooks and HOME are its own, as the base worktree's are. It goes when the run's base worktree
    does (`remove_check_copy`). Returns the copy, its environment and home, and the
    problems that stopped its preparation.
    """
    target = Path(found.target)
    run_id = Path(found.path).name.removeprefix("run-")
    path = _check_copy_path(target, run_id)
    if path.exists():
        try:
            worktree.git(path, "checkout", "-q", "--force", "--detach", found.head)
            worktree.git(path, "clean", "-fdq")
        except ValueError:
            remove_check_copy(target, run_id)
    if not path.exists():
        worktree.git(target, *worktree.NO_HOOKS, "worktree", "add", "-q", "--detach",
                     str(path), found.head)
    worktree._isolate_hooks(target, path)
    copy = worktree.Worktree(path=str(path), branch=f"{worktree.BRANCH_PREFIX}check-{run_id}",
                             base=found.base, head=found.head, target=found.target, run=found.run)
    from refactory_dsh.invocations import CHECK_COPY_HOME

    home = Path(ctx.workdir).resolve() / CHECK_COPY_HOME
    home.mkdir(parents=True, exist_ok=True)
    copy, env, problems, _fail = prepare_check_environment(
        copy, ctx.facts, amend=False, commit_lockfiles=False, run=ctx.workdir, home=home,
        timeout=timeout, budget=getattr(ctx, "budget", None))
    return copy, env, home, problems


def remove_check_copy(target: Path, run_id: str) -> None:
    """Remove the run's check copy, when the run ends or the copy cannot be reused."""
    path = _check_copy_path(target, run_id)
    if path.exists():
        try:
            worktree.git(target, "worktree", "remove", "--force", str(path))
        except ValueError:
            shutil.rmtree(path, ignore_errors=True)


def _check_in_worktree(ctx: Any, found: worktree.Worktree, command: str, timeout: float,
                       *, triage_prep: bool = True) -> Any:
    """Run the check in the run's worktree, after preparing the environment it needs."""
    budget = getattr(ctx, "budget", None)
    found, env, home, problems, prep_fail = _prepare_in_worktree(ctx, found, timeout)
    if problems:
        if triage_prep:
            from refactory_dsh import base_check as _bc
            triage = _bc.compare_prep_failure(
                ctx, prep_fail, problems, found, env, home, timeout,
            )
            if triage is not None:
                return sandbox.SandboxResult(
                    ok=False,
                    returncode=triage["returncode"],
                    elapsed_s=0.0,
                    stderr=triage["error_text"],
                    change_caused_prep=triage,
                )
        return sandbox.SandboxResult(ok=False, returncode=1, elapsed_s=0.0, prep_failure=prep_fail,
                                     stderr=(
            "the worktree's environment could not be prepared:\n" + "\n".join(problems)))
    result = sandbox.run_command(command, workdir=Path(found.path), timeout_s=timeout,
                                budget=budget, env=env, home=home, exact_env=True,
                                worktree_root=Path(found.path))
    if result.cwd_outside:
        target_guard.warn(f"the check's command ended in {result.cwd_outside}, outside "
                          f"the run's worktree {found.path}")
    return result


def refused(ctx: Any) -> str:
    """Why `materialise` could not apply the last ``emit.txt``; empty when it did."""
    return str(ctx.facts.get("emission_refused") or "")


def run_check(ctx: Any, command: str, timeout: float,
              *, triage_prep: bool = True, prepared: bool = False,
              beside_reviews: bool = False) -> Any:
    """Run ``command`` where the run works, and return its `sandbox.SandboxResult`.

    In the run's worktree it runs on the committed attempt -- what a check or reviewer
    changed is put back first -- in the worktree's own environment: its dependencies
    installed when its manifests call for them, then the ``generate`` fact's command (whose
    tracked-file changes are amended into the attempt), then the ``setup`` fact's command, and
    nothing of refactory's own on the path (`worktree.check_environment`). Afterwards the
    worktree is put back again, so the reviews and the next attempt see what edits apply
    to. Without a worktree it runs in the artifact directory the run claimed.

    With ``prepared``, `prepare_attempt` has already done all of that for this commit:
    the command runs in the run's worktree as it is, without preparing again.

    With ``beside_reviews``, the reviewers are reading and running the delivered files
    in the run's worktree while the command runs, so it runs in the run's check copy
    instead (`_check_copy`): neither can see, or delete, what the other leaves, and the
    run's worktree is not touched.
    """
    found = worktree.recorded(ctx.workdir)
    if found is None:
        root = _artifact_root(str(ctx.facts["name"]))
        if Path(str(ctx.facts["artifact"])) != root:
            raise ValueError("artifact location is stale or ambiguous; materialise again before checking")
        _require_owner(str(ctx.facts["name"]), ctx.workdir)
        target_guard.warn_if_moved(ctx.workdir)
        # An artifact built for refactory: its check imports refactory_dsh.
        return sandbox.run_command(command, workdir=root, timeout_s=timeout, budget=getattr(ctx, "budget", None),
                                   env=sandbox.refactory_env(), exact_env=True)
    if beside_reviews:
        if found.head != str(ctx.facts.get("commit")):
            raise ValueError("the worktree is not at the attempt its check was prepared for; "
                             "prepare the check again")
        copy, env, home, problems = _check_copy(ctx, found, timeout)
        if problems:
            return sandbox.SandboxResult(ok=False, returncode=1, elapsed_s=0.0, stderr=(
                "the check copy's environment could not be prepared:\n" + "\n".join(problems)))
        result = sandbox.run_command(command, workdir=Path(copy.path), timeout_s=timeout,
                                     budget=getattr(ctx, "budget", None), env=env, home=home,
                                     exact_env=True, worktree_root=Path(copy.path))
        if result.cwd_outside:
            target_guard.warn(f"the check's command ended in {result.cwd_outside}, outside "
                              f"its check copy {copy.path}")
        _warn_if_shared_changed(found, "while running the check beside the reviews")
        return result
    if prepared:
        found = worktree.recorded(ctx.workdir)
        if found is None or found.head != str(ctx.facts.get("commit")):
            raise ValueError("the worktree is not at the attempt its check was prepared for; "
                             "prepare the check again")
        home = _check_home(ctx)
        result = sandbox.run_command(command, workdir=Path(found.path), timeout_s=timeout,
                                     budget=getattr(ctx, "budget", None),
                                     env=worktree.check_environment(Path(found.path), home),
                                     home=home, exact_env=True, worktree_root=Path(found.path))
        if result.cwd_outside:
            target_guard.warn(f"the check's command ended in {result.cwd_outside}, outside "
                              f"the run's worktree {found.path}")
    else:
        found = _restored_attempt(ctx)
        result = _check_in_worktree(ctx, found, command, timeout, triage_prep=triage_prep)
    # _check_in_worktree already warns about cwd escape; do not warn twice.
    _warn_if_shared_changed(found, "while preparing for and running the check")
    # The reviews and the next attempt read the worktree, and edits apply to the
    # committed attempt: a check that reformats a file must not leave them apart.
    _, changed = worktree.restore(ctx.workdir)
    if changed:
        target_guard.warn(f"the check changed tracked files in the run's worktree; they were put back, so "
                          f"the reviews and the next attempt see the committed attempt: "
                          f"{target_guard._named(changed)}")
    return result


def report(ctx: Any, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Settle the run's place, write the mode's report ``name`` from ``payload``, and return it.

    Call it from the mode's report op, after the mode's own completion checks. A run in a
    worktree is refused if the worktree moved, warned about anything not on its branch, and
    given its branch, base and diff stat in the report; only once the report is written is it
    recorded as succeeded, which keeps its worktree and branch. A run without one is refused
    if another run claimed its artifact directory.
    """
    found = worktree.recorded(ctx.workdir)
    if found is None:
        _require_owner(str(ctx.facts["name"]), ctx.workdir)
        target_guard.warn_if_moved(ctx.workdir)
        ctx.write_json(name, payload)
        return payload
    found = worktree.require_current(ctx.workdir)
    edited = worktree.tracked_edits(found)
    if edited:
        target_guard.warn(f"tracked files in the run's worktree differ from its last attempt and are not "
                          f"part of its branch {found.branch}: {target_guard._named(edited)}")
    _warn_if_shared_changed(found, "during the reviews")
    # Record uncommitted-changes metadata in the report for the PR's Verification.
    uncommitted = worktree.uncommitted_record(found)
    if uncommitted:
        payload = {**payload, "uncommitted_base": True,
                   "uncommitted_files": uncommitted}
        verification = payload.get("verification_notes", [])
        verification = list(verification) if isinstance(verification, list) else []
        verification.insert(0, "The base included uncommitted changes from the working tree")
        payload["verification_notes"] = verification
    payload = {**payload, "target": found.target, "branch": found.branch, "base": found.base, "head": found.head,
               "diff_stat": worktree.git(Path(found.path), "diff", "--stat", found.base, "HEAD").strip()}
    ctx.write_json(name, payload)
    # Record outcome only for runs that will NOT deliver (design_fast).
    # For runs with a delivery plan, delivery._delivered records "succeeded"
    # after a successful push, and _kept guards against recording it when
    # the merge check fails, the merge is abandoned, or there are conflicts.
    # Recording it here prematurely would persist through a failed delivery.
    if not ctx.facts.get("delivery_plan"):
        worktree.record_outcome(found, "succeeded")
    projects.record_run(Path(found.target), run=ctx.workdir, branch=found.branch, base=found.base,
                        status="succeeded")
    # Clean up the base worktree (created by base_check for the base comparison gate)
    from refactory_dsh import base_check
    run_id = Path(found.path).name.removeprefix("run-")
    base_check.remove_base_worktree(Path(found.target), run_id)
    remove_check_copy(Path(found.target), run_id)
    return payload
