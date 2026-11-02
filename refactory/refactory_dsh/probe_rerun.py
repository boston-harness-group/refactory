"""Rerun the design verifier's probes with deterministic code, and check its PROBED claims.

The verifier saves every probe as its own file under ``reviews/probes/``
(``<requirement-id>-<n>.py`` or ``.sh``) and names that file on each PROBED line of
``reviews/verification.md`` (`verification._HUNK_REVERT_METHOD`). A replay of the verifier
on a known-buggy commit showed why the file matters: where the bugs were, it wrote
"PROBED via <the builder's test>" and so inherited the builder's wrong assumptions. Here
"probed" is checked mechanically:

- each probe file runs against the attempt's commit in a scratch checkout -- never the
  builder's workspace or the committed tree -- with `PROBE_TIMEOUT_S` per probe and
  `PROBES_TOTAL_S` for all of them; the probes past the total are recorded as not run
  (advisory);
- ``reviews/probe-results.md`` (and ``.json``) records each probe's exit code, pass/fail
  and the last lines of its output;
- a PROBED line naming no probe file, a missing one, or one that only invokes a test the
  repository had before the change is UNVERIFIED (`design_ops._check_unverified_coverage`
  gives it the coverage rule's one QA-director re-run, then lists it as advisory);
- a probe that fails is a defect finding (kind defect, severity high, breaks its
  requirement) that `design_ops.gate_qa` sends back.

Design mode runs this as the ``rerun_probes`` op before ``synthesize_qa``, which both
runtimes run through the same callable; ``refactory replay-verifier`` runs `run_probes`
after the replayed verifier.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

#: Where the verifier writes its probes, in its workspace and (copied) in the run directory.
PROBES = "reviews/probes"
RESULTS_MD = "reviews/probe-results.md"
RESULTS_JSON = "reviews/probe-results.json"
#: Each probe's deadline, and the deadline for all of them together.
PROBE_TIMEOUT_S = 120
PROBES_TOTAL_S = 600
#: How long preparing the scratch checkout's environment may take per command.
PREPARE_TIMEOUT_S = 600
#: How many trailing output lines the table keeps.
TAIL_LINES = 5

PROBE_NAME = re.compile(r"^[A-Za-z0-9_.-]+\.(?:py|sh)$")
_PROBE_REF = re.compile(r"reviews/probes/([A-Za-z0-9_.-]+\.(?:py|sh))\b")
_REQUIREMENT = re.compile(r"^#{2,6}\s*Requirement:\s*(.+?)\s*$", re.I)
#: A PROBED claim: the uppercase word, not "READ, not probed".
_PROBED = re.compile(r"\bPROBED\b")
#: A test file and optional test id a probe hands to pytest, or a test module it imports.
_TEST_ID = re.compile(r"([\w./-]*test[\w./-]*\.py)(?:::(\w+))?")
_FROM_IMPORT = re.compile(r"^[ \t]*from[ \t]+([\w.]+)[ \t]+import[ \t]+(?:\(([\w\s,]+)\)|([\w \t,]+))", re.M)
#: A check of the probe's own: an assertion, an expected raise, or an explicit exit.
_OWN_CHECK = {".py": re.compile(r"\bassert\b|pytest\.raises|\braise\b|sys\.exit\("),
              ".sh": re.compile(r"\bexit\b|\[\s|\btest\s|python3?\s+(?:-c|-\s*<<)|<<")}


def requirement_of(name: str) -> str:
    """The requirement id a probe file is named for: ``R1-2.py`` → ``R1``."""
    stem = Path(name).stem
    return re.sub(r"-\d+$", "", stem)


def header_of(text: str) -> str:
    """The probe's one-line header comment (requirement, failure point, expected outcome)."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#!"):
            continue
        return stripped.lstrip("#").strip() if stripped.startswith("#") else ""
    return ""


def probed_claims(report: str) -> list[dict[str, str]]:
    """Every PROBED line of the verifier's report: its requirement and the probe file it names."""
    claims: list[dict[str, str]] = []
    requirement = ""
    for line in report.splitlines():
        heading = _REQUIREMENT.match(line.strip())
        if heading:
            requirement = heading[1]
            continue
        if not _PROBED.search(line):
            continue
        named = _PROBE_REF.search(line)
        claims.append({"requirement": requirement or (requirement_of(named[1]) if named else "?"),
                       "file": named[1] if named else "", "line": line.strip()[:300]})
    return claims


def _base_files(repo: Path, base: str) -> list[str]:
    try:
        out = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", base],
                             capture_output=True, text=True, timeout=60, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    return out.splitlines()


def _base_text(repo: Path, base: str, name: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "show", f"{base}:{name}"],
                              capture_output=True, text=True, timeout=60, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return ""


def _existing(files: list[str], relative: str) -> str | None:
    """The base's file that ``relative`` names, from the repository root or any subdirectory."""
    relative = relative.lstrip("./")
    for name in files:
        if name == relative or name.endswith("/" + relative):
            return name
    return None


def existing_tests_invoked(text: str, repo: Path, base: str,
                           files: list[str] | None = None) -> list[str]:
    """The tests the repository had at ``base`` that the probe runs or imports."""
    files = _base_files(repo, base) if files is None else files
    found: list[str] = []
    for match in _TEST_ID.finditer(text):
        name = _existing(files, match[1])
        if name is None:
            continue
        if match[2] is None or re.search(rf"^\s*(?:async\s+)?def\s+{match[2]}\b",
                                         _base_text(repo, base, name), re.M):
            found.append(name + (f"::{match[2]}" if match[2] else ""))
    for match in _FROM_IMPORT.finditer(text):
        names = match[2] if match[2] is not None else match[3]
        tests = [part.strip() for part in names.split(",") if part.strip().startswith("test")]
        name = _existing(files, match[1].replace(".", "/") + ".py") if tests else None
        if name is None:
            continue
        source = _base_text(repo, base, name)
        found += [f"{name}::{test}" for test in tests
                  if re.search(rf"^\s*(?:async\s+)?def\s+{test}\b", source, re.M)]
    return sorted(set(found))


def only_existing_tests(name: str, text: str, repo: Path, base: str,
                        files: list[str] | None = None) -> list[str]:
    """The pre-existing tests a probe merely invokes, or ``[]`` when it has a check of its own."""
    invoked = existing_tests_invoked(text, repo, base, files)
    if not invoked:
        return []
    body = "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))
    own = _OWN_CHECK[Path(name).suffix].search(body)
    return [] if own else invoked


def _command(probe: Path, text: str, env: dict[str, str]) -> list[str]:
    if probe.suffix == ".sh":
        return ["bash", str(probe)]
    python = (shutil.which("python", path=env.get("PATH")) or shutil.which("python3", path=env.get("PATH"))
              or sys.executable)
    if re.search(r"^\s*(?:async\s+)?def\s+test_", text, re.M):
        return [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(probe)]
    return [python, str(probe)]


def _run_one(command: list[str], cwd: Path, env: dict[str, str], timeout: float) -> tuple[int | None, str]:
    """Run ``command`` in its own process group; ``(exit code or None on timeout, output)``."""
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        out, _ = process.communicate(timeout=timeout)
        return process.returncode, out.decode(errors="replace")
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
        out, _ = process.communicate()
        return None, out.decode(errors="replace")


def _tail(output: str) -> str:
    return "\n".join(output.rstrip().splitlines()[-TAIL_LINES:])


def run_probes(probes: Path, checkout: Path, report: str, *, base: str, env: dict[str, str] | None = None,
               reset: Callable[[Path], None] | None = None, per_probe_s: float = PROBE_TIMEOUT_S,
               total_s: float = PROBES_TOTAL_S) -> dict[str, Any]:
    """Run each probe file in ``probes`` in the scratch ``checkout``; check ``report``'s claims.

    ``checkout`` is a clone at the attempt's commit, made for this; ``reset`` puts it back
    to that commit before each probe. ``base`` is the commit the change started from, which
    decides what a pre-existing test is. Returns the results `write_results` records.
    """
    env = dict(os.environ if env is None else env)
    env["PYTHONPATH"] = os.pathsep.join([str(checkout), *filter(None, [env.get("PYTHONPATH")])])
    files = _base_files(checkout, base)
    names = sorted(path.name for path in probes.iterdir()
                   if path.is_file() and PROBE_NAME.match(path.name)) if probes.is_dir() else []
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    for name in names:
        text = (probes / name).read_text(errors="replace")
        entry: dict[str, Any] = {"file": name, "requirement": requirement_of(name),
                                 "header": header_of(text), "exit_code": None, "tail": "",
                                 "existing_tests": only_existing_tests(name, text, checkout, base, files)}
        left = total_s - (time.monotonic() - started)
        if left <= 0:
            entry["status"] = "unrun"
            entry["tail"] = f"not run: the {int(total_s)} s total for all probes was spent"
            results.append(entry)
            continue
        if reset is not None:
            reset(checkout)
        # The probe runs from inside the checkout, at the path the verifier gave it.
        target = checkout / PROBES / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(probes / name, target)
        code, output = _run_one(_command(target, text, env), checkout, env, min(per_probe_s, left))
        entry["exit_code"] = code
        entry["tail"] = _tail(output)
        entry["status"] = ("timed_out" if code is None else "pass" if code == 0 else "fail")
        results.append(entry)
    if reset is not None:
        reset(checkout)
    by_file = {entry["file"]: entry for entry in results}
    unverified: list[dict[str, str]] = []
    for claim in probed_claims(report):
        entry = by_file.get(claim["file"]) if claim["file"] else None
        if not claim["file"]:
            why = "its PROBED line names no probe file under reviews/probes/"
        elif entry is None:
            why = f"its probe file reviews/probes/{claim['file']} does not exist"
        elif entry["existing_tests"]:
            why = (f"its probe file {claim['file']} only invokes tests the repository already had: "
                   + ", ".join(entry["existing_tests"]))
        else:
            continue
        unverified.append({"id": claim["requirement"], "requirement": f"{claim['line']} ({why})",
                           "why": why})
    return {"probes": results, "unverified": unverified, "per_probe_s": per_probe_s,
            "total_s": total_s}


def defects(results: dict[str, Any]) -> list[dict[str, str]]:
    """Each failing probe as a must-fix item the QA gate sends back (`judges.blocks`)."""
    items = []
    for entry in results.get("probes", []):
        if entry.get("status") not in ("fail", "timed_out") or entry.get("existing_tests"):
            continue
        how = (f"timed out after {int(results.get('per_probe_s', PROBE_TIMEOUT_S))} s"
               if entry["status"] == "timed_out" else f"exits {entry['exit_code']}")
        last = entry.get("tail", "").splitlines()[-1:] or [""]
        items.append({"source": "probe rerun", "where": f"{PROBES}/{entry['file']}",
                      "what": f"the verifier's probe for {entry['requirement']} {how} on this "
                              f"attempt: {last[0]}".rstrip(": "),
                      "why": entry.get("header") or "a probe of the requirement fails",
                      "kind": "defect", "severity": "high", "breaks": "ask"})
    return items


def render(results: dict[str, Any]) -> str:
    """``reviews/probe-results.md``: the table, then the unverified claims and the failures."""
    lines = ["# Probe results", "",
             f"Each probe file the verifier wrote, rerun on this attempt's commit in a scratch "
             f"checkout ({int(results.get('per_probe_s', PROBE_TIMEOUT_S))} s per probe, "
             f"{int(results.get('total_s', PROBES_TOTAL_S))} s for all).", "",
             "| file | requirement | exit code | result | last lines of output |",
             "| --- | --- | --- | --- | --- |"]
    for entry in results.get("probes", []):
        result = {"pass": "pass", "fail": "FAIL", "timed_out": "FAIL (timed out)",
                  "unrun": "not run (advisory)"}[entry["status"]]
        if entry.get("existing_tests"):
            result += " — only invokes existing tests: UNVERIFIED"
        tail = entry.get("tail", "").replace("|", "\\|").replace("\n", "<br>")
        code = "" if entry.get("exit_code") is None else str(entry["exit_code"])
        lines.append(f"| {entry['file']} | {entry['requirement']} | {code} | {result} | {tail} |")
    if not results.get("probes"):
        lines.append("| (no probe files) | | | | |")
    if results.get("unverified"):
        lines += ["", "## UNVERIFIED claims", ""]
        lines += [f"- UNVERIFIED {item['id']}: {item['requirement']}"
                  for item in results["unverified"]]
    found = defects(results)
    if found:
        lines += ["", "## Failing probes (defects, blocking)", ""]
        lines += [f"- {item['where']}: {item['what']} ({item['why']})" for item in found]
    unrun = [entry["file"] for entry in results.get("probes", []) if entry["status"] == "unrun"]
    if unrun:
        lines += ["", "## Not run (advisory)", "", *(f"- {name}" for name in unrun)]
    return "\n".join(lines) + "\n"


def write_results(out: Path, results: dict[str, Any]) -> str:
    """Write ``reviews/probe-results.md`` and ``.json`` under ``out``; return the markdown."""
    text = render(results)
    (out / RESULTS_MD).parent.mkdir(parents=True, exist_ok=True)
    (out / RESULTS_MD).write_text(text)
    (out / RESULTS_JSON).write_text(json.dumps(results, indent=2, sort_keys=True))
    return text


def read_results(ctx: Any) -> dict[str, Any]:
    """This round's probe results, or ``{}`` when none were recorded."""
    if not ctx.exists(RESULTS_JSON):
        return {}
    try:
        found = json.loads(ctx.read_text(RESULTS_JSON))
    except ValueError:
        return {}
    return found if isinstance(found, dict) else {}


def rerun_probes(ctx: Any) -> dict[str, Any]:
    """Op before ``synthesize_qa``: rerun the verifier's probe files on the attempt.

    The probes come from the verifier's workspace clone (``reviews/probes/`` there; a
    directory is not a declared write, because an agent node fails on a declared output it
    leaves unchanged) and are copied to the run directory, replacing the last round's. They
    run in this op's own clone at the attempt (`clones._prepare`), with that clone's
    prepared environment, reset before each probe.
    """
    from refactory_dsh import clones, worktree
    from refactory_dsh.workspace import ROOT, _slug

    workdir = Path(ctx.workdir)
    probes = workdir / PROBES
    shutil.rmtree(probes, ignore_errors=True)
    source = workdir / ROOT / _slug("verifier") / "repo" / PROBES
    if source.is_dir():
        probes.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if path.is_file() and not path.is_symlink() and PROBE_NAME.match(path.name):
                shutil.copyfile(path, probes / path.name)
    report = ctx.read_text("reviews/verification.md") if ctx.exists("reviews/verification.md") else ""
    found = worktree.recorded(workdir)
    if found is None:
        results: dict[str, Any] = {"probes": [], "unverified": [], "per_probe_s": PROBE_TIMEOUT_S,
                                   "total_s": PROBES_TOTAL_S,
                                   "note": "the run has no worktree, so there is no attempt to rerun probes on"}
    else:
        prepared = clones._prepare(workdir, workdir / ROOT / "probe_rerun", "attempt",
                                   facts=dict(ctx.facts), command_timeout_s=PREPARE_TIMEOUT_S)
        env = dict(os.environ)
        for key, value in (prepared.get("env") or {}).items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = str(value)
        results = run_probes(probes, Path(prepared["path"]), report, base=found.base, env=env,
                             reset=clones._clean)
    text = write_results(workdir, results)
    return {"probe_results": text}
