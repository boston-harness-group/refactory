"""``refactory-targeted-tests``: the builder runs the tests the self-check will run.

A builder that stops without running the tests its change breaks is sent back by the
self-check (`design_ops.prepare_check`) into a new, cold session. This command lets it find
those failures in the session it is in. It is put on the PATH of a clone workspace
(`clones.prepare` calls `install`), and does three things:

1. It reads the workspace's current change the way the run will take it: the emission
   (`workspace.derive_emission`), which is the diff from the clone's base that
   materialise commits and the self-check then sees.
2. It selects tests with the self-check's own selection (`design_ops.self_check_selection`,
   which `design_ops._fixup_self_check` calls too), not a selector of its own.
3. It runs that selection with the check command's own options kept (`-n 4` and so on),
   as `design_ops.targeted_command` builds it, and prints what it selected and the result.

With ``--failed`` it reruns only the tests that failed in its last run in this workspace
(recorded in ``LAST_RUN``, from the run's own ``FAILED``/``ERROR`` lines), so a builder can
iterate on a failure without rerunning what passed; with no record it runs the full
selection. The builder runs the plain command once before it stops.

It runs the whole selection in one command. The self-check cuts the selection into chunks
only to bound its time, so a builder may run more than the self-check reaches, never less.
It sets no limit of its own.

The command's inputs are what the run knew when the dispatch began, written to the node's
workspace root (`CONTEXT`): the check command, the last check's record and the earlier
self-checks. Nothing here writes to the run.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

#: The command's name on the builder's PATH.
COMMAND = "refactory-targeted-tests"

#: The node's workspace root, under which the command and its inputs are written.
BIN = "bin"
CONTEXT = "targeted-tests.json"

#: What the last run in this workspace failed on, next to `CONTEXT`.
LAST_RUN = "targeted-tests-last.json"

#: A failing test as pytest's short summary names it: ``FAILED tests/x.py::test_y - why``.
_FAILED = re.compile(r"^(?:FAILED|ERROR) (\S+)", re.MULTILINE)

_BLOCK = re.compile(r"^===(?:FILE|DELETE): (.*)===$", re.MULTILINE)


def install(workdir: Path, root: Path, prepared: dict[str, Any], facts: dict[str, Any]) -> None:
    """Put the command on the PATH of the clone that ``prepared`` describes.

    ``prepared`` is what `clones.prepare` returns; its ``env`` gains the directory the
    command is in. The inputs are written afresh on every dispatch, so the command always
    reflects the run's last check and self-checks.
    """
    from refactory_dsh.invocations import SELF_CHECK_RECORD

    root = Path(root)
    context = {
        "check": str(facts.get("check") or ""),
        "path": str(prepared["path"]), "base": str(prepared["base"]),
        "conflicts": list(prepared.get("conflicts") or ()),
        "deletes": bool(prepared.get("deletes", True)),
        "skip": dict(prepared.get("skip") or {}),
        "generate_owned": dict(prepared.get("generate_owned") or {}),
        "prev_record": _json(Path(workdir) / "check-record.json"),
        "earlier": _lines(Path(workdir) / SELF_CHECK_RECORD),
    }
    (root / CONTEXT).write_text(json.dumps(context, default=str))
    directory = root / BIN
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / COMMAND
    # refactory's own interpreter, isolated (-I) so the workspace's directory and variables
    # cannot shadow refactory_dsh, and no bytecode written into refactory's source.
    script.write_text(
        "#!/bin/sh\n"
        "# Written by refactory (targeted_tests.py): runs the tests the self-check will run.\n"
        f"PYTHONDONTWRITEBYTECODE=1 exec {shlex.quote(sys.executable)} -I -m "
        f"refactory_dsh.targeted_tests {shlex.quote(str(root / CONTEXT))} \"$@\"\n")
    script.chmod(0o755)
    env = prepared.setdefault("env", {})
    ambient = env.get("PATH") or ""
    env["PATH"] = f"{directory}:{ambient}" if ambient else str(directory)


def selection(context: dict[str, Any]) -> tuple[list[str], Any]:
    """The files the workspace's change touches, and the tests the self-check selects for them."""
    from refactory_dsh import design_ops, workspace
    from refactory_dsh.test_selection import repository_files

    root = Path(context["path"])
    emission, _refused, _approximate = workspace.derive_emission(
        root, context["base"], conflicts=tuple(context.get("conflicts") or ()),
        deletes=bool(context.get("deletes", True)), skip=dict(context.get("skip") or {}),
        generate_owned=dict(context.get("generate_owned") or {}))
    touched = [match.group(1) for match in _BLOCK.finditer(emission)]
    # The self-check sees the attempt committed: a file the builder has only created counts.
    untracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"],
        capture_output=True, timeout=60, check=False).stdout.decode(errors="replace").split("\0")
    files = sorted({name for name in [*repository_files(root), *untracked]
                    if name and (root / name).is_file() and not workspace._excluded(Path(name))})
    chosen = design_ops.self_check_selection(
        root, context["check"], touched, context.get("prev_record") or {},
        list(context.get("earlier") or ()), files=files)
    return touched, chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=COMMAND, description="Run the tests the self-check will run for your change.")
    parser.add_argument("context", help="the node's workspace inputs (written by refactory)")
    parser.add_argument("--failed", action="store_true",
                        help="rerun only the tests that failed in the last run here "
                             "(the full selection when none is recorded)")
    parser.add_argument("--list", action="store_true",
                        help="print the selection and the command as JSON; run nothing")
    args = parser.parse_args(argv)
    context = json.loads(Path(args.context).read_text())
    check = str(context.get("check") or "")
    if not check:
        print(f"{COMMAND}: this run has no check command, so there is no self-check to mirror.")
        return 2
    from refactory_dsh import design_ops

    touched, chosen = selection(context)
    root = Path(context["path"])
    record = Path(args.context).with_name(LAST_RUN)
    tests = list(chosen.tests)
    rerun = _last_failures(record) if args.failed else []
    if rerun:
        tests = rerun
    line, refusal = (design_ops.targeted_command(check, tests, root) if tests else ("", None))
    if args.list:
        print(json.dumps({"touched": touched, "selection": tests, "command": line,
                          "refusal": refusal, "rerun_failed": bool(rerun)}))
        return 0
    if args.failed and not rerun:
        print(f"{COMMAND}: no failures are recorded from a last run here; running the full "
              "selection.")
    print(f"{COMMAND}: {len(touched)} changed file(s) in your workspace; the self-check will "
          f"run {len(chosen.tests)} test file(s) or id(s) for them"
          + (f"; rerunning the {len(rerun)} that failed last time:" if rerun else ":"))
    for name in tests[:40]:
        print(f"  {name}")
    if len(tests) > 40:
        print(f"  ... and {len(tests) - 40} more")
    if not chosen.tests:
        print(f"{COMMAND}: nothing is selected: no test the self-check runs depends on this change.")
        return 0
    if refusal or not line:
        print(f"{COMMAND}: the check command cannot be targeted ({refusal}); the full check "
              "runs after you finish.")
        return 2
    print(f"{COMMAND}: running: {line}", flush=True)
    code, failed = _run_and_collect(line, root)
    _remember(record, failed)
    if code:
        print(f"{COMMAND}: FAILED (exit {code}): these are what the self-check would send you "
              "back for. Fix them and run this again.", flush=True)
    else:
        print(f"{COMMAND}: PASSED: every test the self-check will run for this change passes.",
              flush=True)
    return code


def _run_and_collect(line: str, root: Path) -> tuple[int, list[str]]:
    """Run ``line`` with its output passed through as it arrives; the exit code, and the
    tests it reported failing."""
    proc = subprocess.Popen(line, shell=True, cwd=root, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace")
    seen: list[str] = []
    assert proc.stdout is not None
    for text in proc.stdout:
        print(text, end="", flush=True)
        seen.append(text)
    code = proc.wait()
    return code, list(dict.fromkeys(_FAILED.findall("".join(seen))))


def _remember(path: Path, failed: list[str]) -> None:
    """Record what the last run failed on (empty when it passed)."""
    try:
        path.write_text(json.dumps({"failed": failed}))
    except OSError:
        pass


def _last_failures(path: Path) -> list[str]:
    """The tests the last run failed on; empty when there is no record or it passed."""
    failed = _json(path).get("failed")
    return [str(name) for name in failed] if isinstance(failed, list) else []


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _lines(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    try:
        text = path.read_text()
    except OSError:
        return entries
    for line in text.splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


if __name__ == "__main__":
    sys.exit(main())
