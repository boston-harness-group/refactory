"""The verifier's probes are files, rerun by deterministic code (`probe_rerun`).

A PROBED claim without a probe file of its own is UNVERIFIED (the coverage rule's
re-run, then advisory); a probe file that fails is a blocking defect; one that passes
verifies its requirement. The rerun happens in a scratch checkout of the attempt, never
the run's worktree or the verifier's workspace, within a total cap.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from refactory_dsh import clones, design_ops, judges, nn, probe_rerun, worktree
from refactory_dsh.design import DesignMode
from refactory_dsh.verification import DESIGN_VERIFIER_PROMPT
from refactory_dsh.workspace import ROOT


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def _repo(path: Path) -> tuple[str, str]:
    """A target with a base commit (calc and its test) and an attempt that changes calc."""
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@t")
    _git(path, "config", "user.name", "t")
    # The base's double is wrong (5 for 2) and the attempt fixes it: a probe that calls it
    # tells the two apart by behaviour, not by reading the source.
    (path / ".gitignore").write_text("lib/\n")
    (path / "calc.py").write_text("def double(x):\n    return x + x + 1\n")
    (path / "tests").mkdir()
    (path / "tests" / "test_calc.py").write_text(
        "from calc import double\n\n\ndef test_double():\n    assert double(2) == 4\n")
    # A package of its own in a subdirectory project, as refactory's is: importable only
    # with that project's root on the import path, never from the checkout's root.
    (path / "proj" / "mypkg").mkdir(parents=True)
    (path / "proj" / "pyproject.toml").write_text('[project]\nname = "mypkg"\nversion = "0"\n')
    (path / "proj" / "mypkg" / "__init__.py").write_text("def triple(x):\n    return 3 * x\n")
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", "base")
    base = _git(path, "rev-parse", "HEAD")
    (path / "calc.py").write_text("def double(x):\n    return 2 * x  # attempt\n")
    _git(path, "commit", "-qam", "attempt")
    return base, _git(path, "rev-parse", "HEAD")


def _setup(tmp_path: Path, monkeypatch, report: str, probes: dict[str, str],
           judgment: str = '{"verdict": "proceed", "must_fix": [], "follow_ups": []}',
           prepared: list | None = None) -> nn.RunContext:
    """A design run directory after the verifier: its report, its probe files in its
    workspace clone, the three reviews and the QA judge's verdict. `clones._prepare` makes
    a plain clone (at the base or the attempt) and records each call in ``prepared``."""
    target = tmp_path / "target"
    base, head = _repo(target)
    run = tmp_path / "run"
    run.mkdir()
    (run / worktree.RECORD).parent.mkdir(parents=True, exist_ok=True)
    (run / worktree.RECORD).write_text(json.dumps({
        "path": str(target), "branch": "b", "base": base, "head": head,
        "target": str(target), "run": str(run)}))
    verifier = run / ROOT / "verifier" / "repo"
    (verifier / probe_rerun.PROBES).mkdir(parents=True)
    for name, text in probes.items():
        (verifier / probe_rerun.PROBES / name).write_text(text)

    def prepare(workdir, root, of, *, facts, command_timeout_s, budget=None):
        repo = Path(root) / "repo"
        if prepared is not None:
            prepared.append(of)
        if not repo.exists():
            clones.clone_shared(target, repo, base if of == "base" else head)
        return {"path": str(repo), "env": {}}

    monkeypatch.setattr(clones, "_prepare", prepare)
    ctx = nn.RunContext(run, facts={})
    ctx.write_text("reviews/verification.md", report)
    for name, _ in design_ops.DESIGN_QA_REVIEWS:
        ctx.write_text(name, "ran it; fine\n")
    ctx.write_text(judges.QA_JUDGE, judgment)
    return ctx


def _run(tmp_path: Path, monkeypatch, report: str, probes: dict[str, str],
         judgment: str = '{"verdict": "proceed", "must_fix": [], "follow_ups": []}',
         prepared: list | None = None) -> nn.RunContext:
    """`_setup`, then `rerun_probes`."""
    ctx = _setup(tmp_path, monkeypatch, report, probes, judgment, prepared)
    probe_rerun.rerun_probes(ctx)
    return ctx


REPORT = "#### Requirement: R1\n- Failure points: PROBED {}\n\nVERDICT: PASS\n"
OWN_PROBE = ("# R1: double at the last element; expected 2*x\nimport inspect\nfrom calc import double\n"
             "assert double(3) == 6\nassert '# attempt' in inspect.getsource(double)\n")


def test_a_probed_claim_without_a_probe_file_is_unverified(tmp_path, monkeypatch):
    """The replay's shape: "PROBED via test_x" and no file → UNVERIFIED → the coverage
    rule's QA-director re-run, naming the requirement."""
    ctx = _run(tmp_path, monkeypatch, REPORT.format("via tests/test_calc.py::test_double"), {})
    results = probe_rerun.read_results(ctx)
    assert [item["id"] for item in results["unverified"]] == ["R1"]
    assert "names no probe file" in results["unverified"][0]["why"]
    verdict = design_ops.gate_qa(ctx)
    assert verdict == "coverage_rerun"
    assert "R1" in ctx.facts["qa_coverage_rerun_ids"]
    assert "UNVERIFIED R1" in ctx.read_text(probe_rerun.RESULTS_MD)


def test_a_probe_runs_in_the_sandbox_with_its_own_environment(tmp_path, monkeypatch):
    """A probe is model-authored code, so it runs the way base_check runs reverted code.

    `sandbox.run_command` bounds it (CPU and file-size limits, the process group stopped
    at the deadline, no stdin) and watches for it leaving its checkout; the probe keeps
    the environment and HOME the rerun built for it.
    """
    from refactory_dsh import sandbox

    seen = {}
    real = sandbox.run_command

    def spy(command, **kwargs):
        seen.update(kwargs, command=command)
        return real(command, **kwargs)

    monkeypatch.setattr(sandbox, "run_command", spy)
    code, output = probe_rerun._run_one(
        ["sh", "-c", "echo probed"], tmp_path, {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}, 30)
    assert (code, output.strip()) == (0, "probed")
    assert seen["exact_env"] is True and seen["env"]["HOME"] == str(tmp_path)
    assert seen["worktree_root"] == tmp_path and seen["workdir"] == tmp_path


def test_a_probe_that_leaves_its_checkout_says_so(tmp_path):
    """The sandbox's cwd watch reaches the probe's output, where the reviewers see it."""
    from refactory_dsh import sandbox

    result = sandbox.SandboxResult(ok=True, returncode=0, stdout="ran\n", cwd_outside="/elsewhere")
    original = sandbox.run_command
    try:
        sandbox.run_command = lambda *a, **k: result
        code, output = probe_rerun._run_one(["true"], tmp_path, {}, 5)
    finally:
        sandbox.run_command = original
    assert code == 0 and "left its checkout for /elsewhere" in output


def test_a_probe_timeout_is_reported_as_timed_out(tmp_path):
    code, _ = probe_rerun._run_one(["sh", "-c", "sleep 30"], tmp_path,
                                   {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}, 1)
    assert code is None


def test_ignored_files_do_not_survive_from_one_probe_to_the_next(tmp_path):
    """One probe must not change the next one's result through a gitignored path.

    `clones._clean` keeps ignored files (build caches); between probes that let a file
    under a gitignored directory such as reviews/ leak. Installed environments stay.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / ".gitignore").write_text("/scratch/\n.venv/\n")
    (repo / "a.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")
    probes = tmp_path / "probes"
    probes.mkdir()
    (probes / "R1-1.sh").write_text("#!/bin/sh\n# R1: leaves state\nmkdir -p scratch .venv\n"
                                    "echo leak > scratch/leak.txt\necho env > .venv/kept.txt\n"
                                    "[ -f a.py ]\n")
    (probes / "R2-1.sh").write_text("#!/bin/sh\n# R2: must start clean\n[ ! -f scratch/leak.txt ]\n")

    results = probe_rerun.run_probes(probes, repo, "", base=base,
                                     env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
                                     reset=probe_rerun.reset_checkout)
    status = {entry["file"]: entry["status"] for entry in results["probes"]}
    assert status == {"R1-1.sh": "pass", "R2-1.sh": "pass"}, results["probes"]
    assert (repo / ".venv" / "kept.txt").exists(), "an installed environment must be kept"


def test_the_graph_and_the_cli_reset_probe_checkouts_with_ignored_files(monkeypatch):
    """Both callers of run_probes use the reset that drops ignored files."""
    import inspect

    from refactory_dsh import cli

    assert "reset=reset_checkout" in inspect.getsource(probe_rerun.rerun_probes)
    assert "reset=probe_rerun.reset_checkout" in inspect.getsource(cli)


def test_a_probe_that_checks_nothing_is_unverified_not_a_pass(tmp_path, monkeypatch):
    """`print("ok")` exits 0. Without a floor, "every probe passed" says nothing.

    A probe is model-authored, so a pass only counts when the file contains a check of
    its own.
    """
    probe = '# R1: nothing\nprint("ok")\n'
    ctx = _run(tmp_path, monkeypatch, REPORT.format("reviews/probes/R1-1.py"), {"R1-1.py": probe})
    results = probe_rerun.read_results(ctx)
    assert results["probes"][0]["status"] == "no_check", results["probes"][0]
    assert [item["id"] for item in results["unverified"]] == ["R1"]
    assert "without checking anything" in results["unverified"][0]["why"]


def test_a_probe_whose_run_collected_nothing_is_unverified(tmp_path, monkeypatch):
    """A pytest probe whose only test is skipped exits 0 and checks nothing."""
    probe = ("# R1: skipped\nimport pytest\n\n\n"
             "@pytest.mark.skip(reason='x')\ndef test_it():\n    assert False\n")
    ctx = _run(tmp_path, monkeypatch, REPORT.format("reviews/probes/R1-1.py"), {"R1-1.py": probe})
    results = probe_rerun.read_results(ctx)
    assert results["probes"][0]["status"] == "no_check", results["probes"][0]
    assert [item["id"] for item in results["unverified"]] == ["R1"]


def test_an_exit_status_is_not_a_check_of_the_probes_own(tmp_path, monkeypatch):
    """`sys.exit(pytest(<existing test>))` is the obvious way to write the one thing a
    probe may not be, and it used to satisfy the own-check rule."""
    probe = ("# R1: reuse by exit status\nimport subprocess, sys\n"
             "sys.exit(subprocess.run([sys.executable, '-m', 'pytest', '-q',\n"
             "    'tests/test_calc.py::test_double']).returncode)\n")
    ctx = _run(tmp_path, monkeypatch, REPORT.format("reviews/probes/R1-1.py"), {"R1-1.py": probe})
    results = probe_rerun.read_results(ctx)
    assert results["probes"][0]["existing_tests"] == ["tests/test_calc.py::test_double"]
    assert [item["id"] for item in results["unverified"]] == ["R1"]


def test_a_bare_exit_127_does_not_excuse_a_failing_probe(tmp_path, monkeypatch):
    """Exit 127 alone is not evidence of a missing command.

    A probe whose last line is `exit 127` would otherwise mark itself "environment",
    which `defects` skips — a blocking check any probe could opt out of.
    """
    probe = ("#!/bin/sh\n# R1: self-exempting\n"
             "python3 -c 'import calc; assert calc.double(2) == 99' || exit 127\n")
    ctx = _run(tmp_path, monkeypatch, REPORT.format("reviews/probes/R1-1.sh"), {"R1-1.sh": probe})
    results = probe_rerun.read_results(ctx)
    entry = results["probes"][0]
    assert entry["status"] == "fail", entry
    assert "environment" not in entry, entry
    assert [item["where"] for item in probe_rerun.defects(results)] == ["reviews/probes/R1-1.sh"]


def test_a_named_missing_command_is_still_an_environment_failure(tmp_path, monkeypatch):
    """The downgrade still applies when the output names what was not found."""
    probe = "#!/bin/sh\n# R1: missing tool\nthis-command-does-not-exist --version\n"
    ctx = _run(tmp_path, monkeypatch, REPORT.format("reviews/probes/R1-1.sh"), {"R1-1.sh": probe})
    entry = probe_rerun.read_results(ctx)["probes"][0]
    assert entry["status"] == "environment", entry
    assert "this-command-does-not-exist" in entry["environment"]


def test_a_probe_that_only_runs_an_existing_test_is_unverified(tmp_path, monkeypatch):
    probe = "#!/bin/sh\n# R1: reuse\npytest -q tests/test_calc.py::test_double\n"
    ctx = _run(tmp_path, monkeypatch, REPORT.format("reviews/probes/R1-1.sh"), {"R1-1.sh": probe})
    results = probe_rerun.read_results(ctx)
    assert results["probes"][0]["existing_tests"] == ["tests/test_calc.py::test_double"]
    assert "only invokes tests the repository already had" in results["unverified"][0]["why"]
    assert design_ops.gate_qa(ctx) == "coverage_rerun"


def test_a_probe_that_imports_and_calls_an_existing_test_is_unverified(tmp_path, monkeypatch):
    probe = "# R1: reuse\nfrom tests.test_calc import test_double\ntest_double()\n"
    ctx = _run(tmp_path, monkeypatch, REPORT.format("reviews/probes/R1-1.py"), {"R1-1.py": probe})
    assert [item["id"] for item in probe_rerun.read_results(ctx)["unverified"]] == ["R1"]


def test_a_script_pytest_cannot_run_as_a_test_is_unverified_not_a_defect(tmp_path, monkeypatch):
    """A .py probe with a test_* name in it is run under pytest; when pytest finds
    nothing it can run (a method, a helper with arguments) it exits 5 or 1 with a missing
    fixture. That says the probe checked nothing, not that the change is wrong."""
    method = ("# R1: a checker class\nfrom calc import double\n\n\nclass Checker:\n"
              "    def __init__(self):\n        self.n = 2\n\n"
              "    def test_case(self):\n        assert double(self.n) == 4\n\n\n"
              "Checker().test_case()\n")
    helper = ("# R2: a helper\nfrom calc import double\n\n\n"
              "def test_value(v):\n    assert double(v) == 2 * v\n\n\ntest_value(3)\n")
    report = REPORT.format("reviews/probes/R1-1.py") + "\n" + REPORT.format(
        "reviews/probes/R2-1.py").replace("R1", "R2")
    ctx = _run(tmp_path, monkeypatch, report, {"R1-1.py": method, "R2-1.py": helper})
    results = probe_rerun.read_results(ctx)
    statuses = {entry["file"]: entry["status"] for entry in results["probes"]}
    assert statuses == {"R1-1.py": "no_check", "R2-1.py": "no_check"}, results["probes"]
    assert probe_rerun.defects(results) == []


def test_a_pytest_probe_whose_assertion_fails_is_still_a_defect(tmp_path, monkeypatch):
    """The pytest exceptions above must not excuse a real failure."""
    probe = ("# R1: double under pytest\nfrom calc import double\n\n\n"
             "def test_double():\n    assert double(3) == 7\n")
    ctx = _run(tmp_path, monkeypatch, REPORT.format("reviews/probes/R1-1.py"), {"R1-1.py": probe})
    results = probe_rerun.read_results(ctx)
    assert results["probes"][0]["status"] == "fail", results["probes"][0]
    assert [item["where"] for item in probe_rerun.defects(results)] == ["reviews/probes/R1-1.py"]


def test_a_pytest_probe_that_cannot_import_the_change_is_still_a_defect(tmp_path, monkeypatch):
    """pytest exits 2 when collection fails -- here because the function the brief asked
    for was never added. That is the probe's verdict on the change, not a probe pytest
    could not run, so it must still block."""
    probe = ("# R1: triple, which the brief asks for\nfrom calc import triple\n\n\n"
             "def test_triple():\n    assert triple(2) == 6\n")
    ctx = _run(tmp_path, monkeypatch, REPORT.format("reviews/probes/R1-1.py"), {"R1-1.py": probe})
    results = probe_rerun.read_results(ctx)
    assert (results["probes"][0]["status"], results["probes"][0]["exit_code"]) == ("fail", 2)
    assert [item["where"] for item in probe_rerun.defects(results)] == ["reviews/probes/R1-1.py"]


def test_a_timed_out_probe_is_a_blocking_defect():
    results = {"per_probe_s": 120, "probes": [
        {"file": "R1-1.sh", "requirement": "R1", "status": "timed_out", "exit_code": None,
         "tail": "still going", "existing_tests": [], "header": "R1: hangs"}]}
    [item] = probe_rerun.defects(results)
    assert "timed out after 120 s" in item["what"] and judges.blocks(item)


def test_a_failing_probe_file_is_a_blocking_defect(tmp_path, monkeypatch):
    """The judge proceeded; the probe fails on the attempt; the gate sends it back."""
    probe = "# R1: double(3) at the end; expected 7\nfrom calc import double\nassert double(3) == 7\n"
    ctx = _run(tmp_path, monkeypatch, REPORT.format("reviews/probes/R1-1.py"), {"R1-1.py": probe})
    [item] = probe_rerun.defects(probe_rerun.read_results(ctx))
    assert (item["kind"], item["severity"], item["breaks"]) == ("defect", "high", "ask")
    assert judges.blocks(item)
    verdict = design_ops.gate_qa(ctx)
    assert verdict == "reloop"
    assert "reviews/probes/R1-1.py" in verdict.reason and "exits 1" in verdict.reason
    assert "| R1-1.py | R1 | 1 | FAIL |" in ctx.read_text(probe_rerun.RESULTS_MD)
    assert "probe-results.md" in design_ops.synthesize_qa(ctx)["qa"]


def test_a_passing_probe_file_verifies_its_requirement(tmp_path, monkeypatch):
    ctx = _run(tmp_path, monkeypatch, REPORT.format("reviews/probes/R1-1.py"),
               {"R1-1.py": OWN_PROBE})
    results = probe_rerun.read_results(ctx)
    assert results["probes"][0]["status"] == "pass" and results["unverified"] == []
    assert results["probes"][0]["header"].startswith("R1: double at the last element")
    assert design_ops.gate_qa(ctx) == "proceed"


def test_the_rerun_uses_a_scratch_checkout_and_leaves_the_workspaces_untouched(tmp_path, monkeypatch):
    """The probe sees the attempt's commit and may write anywhere in its checkout; the
    run's worktree and the verifier's workspace are as they were."""
    probe = ("# R1: the attempt's code; expected 2 * x\nimport pathlib\n"
             "assert '2 * x' in pathlib.Path('calc.py').read_text()\n"
             "pathlib.Path('calc.py').write_text('broken')\npathlib.Path('junk.txt').write_text('x')\n")
    ctx = _run(tmp_path, monkeypatch, REPORT.format("reviews/probes/R1-1.py"),
               {"R1-1.py": probe, "R1-2.py": OWN_PROBE})
    assert [p["status"] for p in probe_rerun.read_results(ctx)["probes"]] == ["pass", "pass"]
    target = tmp_path / "target"
    assert _git(target, "status", "--porcelain") == ""
    verifier = ctx.workdir / ROOT / "verifier" / "repo"
    assert sorted(p.name for p in verifier.rglob("*") if p.is_file()) == ["R1-1.py", "R1-2.py"]
    scratch = ctx.workdir / ROOT / "probe_rerun" / "repo"
    assert "2 * x" in (scratch / "calc.py").read_text() and not (scratch / "junk.txt").exists()


def test_the_total_cap_leaves_the_remaining_probes_unrun_and_advisory(tmp_path, monkeypatch):
    # Thread-local clock: each thread (head-side, base-side, main) gets its own counter
    # so the parallel base-revert rerun does not consume the head thread's tick budget.
    import threading
    _local = threading.local()

    def _clock():
        if not hasattr(_local, "n"):
            _local.n = 0
        _local.n += 1
        return 10_000.0 if _local.n > 2 else 0.0

    monkeypatch.setattr(probe_rerun.time, "monotonic", _clock)
    failing = "# R2: expected 7\nfrom calc import double\nassert double(3) == 7\n"
    ctx = _run(tmp_path, monkeypatch, REPORT.format("reviews/probes/R1-1.py"),
               {"R1-1.py": OWN_PROBE, "R2-1.py": failing})
    results = probe_rerun.read_results(ctx)
    assert [p["status"] for p in results["probes"]] == ["pass", "unrun"]
    assert probe_rerun.defects(results) == []
    text = ctx.read_text(probe_rerun.RESULTS_MD)
    assert "not run (advisory)" in text and f"{probe_rerun.PROBES_TOTAL_S} s" in text
    assert design_ops.gate_qa(ctx) == "proceed"


def test_the_verifier_is_told_to_save_each_probe_as_a_file():
    for text in ("reviews/probes/", "<requirement-id>-<n>.py", "<requirement-id>-<n>.sh",
                 "the requirement, the failure point and the expected outcome",
                 "must not just call an\nexisting repository test",
                 "Each PROBED line in reviews/verification.md names its\nprobe file"):
        assert text in DESIGN_VERIFIER_PROMPT, text


def test_the_rerun_runs_before_the_synthesis_on_both_runtimes():
    graph = DesignMode().graph()
    node = graph.nodes["rerun_probes"]
    assert node.kind == "op" and node.call == "refactory_dsh.probe_rerun:rerun_probes"
    edges = {(edge.source, edge.target, edge.condition) for edge in graph.edges}
    assert ("gate_check", "rerun_probes", "proceed") in edges
    assert ("rerun_probes", "synthesize_qa", None) in edges
    assert ("gate_check", "synthesize_qa", "proceed") not in edges


def test_replay_verifier_reruns_the_probes_and_prints_the_table(tmp_path, monkeypatch, capsys):
    import refactory_dsh.model as model_mod
    from refactory_dsh import cli

    work = tmp_path / "work"
    _repo(work)
    run_dir = tmp_path / "run"
    (run_dir / ".refactory").mkdir(parents=True)
    (run_dir / "brief.json").write_text('{"brief": "double"}')
    (run_dir / ".refactory" / "worktree.json").write_text(json.dumps({"path": str(work)}))

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def context(self):
            return {"worker_backend": "dsh", "provider": "p", "model": "m", "effort": "",
                    "routes_resolved": True}

        def complete(self, prompt, *, cwd, **kwargs):
            (Path(cwd) / probe_rerun.PROBES).mkdir(parents=True)
            (Path(cwd) / probe_rerun.PROBES / "R1-1.py").write_text(OWN_PROBE)
            return REPORT.format("reviews/probes/R1-1.py")

    monkeypatch.setattr(model_mod, "DshModelClient", FakeClient)
    assert cli.main(["replay-verifier", "--run-dir", str(run_dir), "--provider", "p",
                     "--model", "m"]) == 0
    out = capsys.readouterr().out
    assert "# Probe results" in out and "| R1-1.py | R1 | 0 | pass |" in out
    assert (run_dir / probe_rerun.RESULTS_MD).is_file()
    assert not (work / "reviews").exists() and (run_dir / probe_rerun.PROBES / "R1-1.py").is_file()


PACKAGE_REPORT = "#### Requirement: R1\n- Failure points: PROBED reviews/probes/R1-1.py\n\nVERDICT: PASS\n"


def test_a_probe_importing_the_projects_own_package_runs_and_passes(tmp_path, monkeypatch):
    """The replay's 14 false defects: the package lives in a subdirectory project, so the
    checkout's root alone on the import path gave ModuleNotFoundError."""
    probe = "# R1: triple(2); expected 6\nfrom mypkg import triple\nassert triple(2) == 6\n"
    ctx = _run(tmp_path, monkeypatch, PACKAGE_REPORT, {"R1-1.py": probe})
    [entry] = probe_rerun.read_results(ctx)["probes"]
    assert entry["status"] == "pass", entry["tail"]
    # mypkg is unchanged, so the probe passes on the base too: an advisory finding,
    # never a re-run, so the gate proceeds.
    assert entry.get("discriminating") is False
    assert design_ops.gate_qa(ctx) == "proceed"


def test_a_probe_of_the_package_that_asserts_false_is_a_defect(tmp_path, monkeypatch):
    probe = "# R1: triple(2); expected 7\nfrom mypkg import triple\nassert triple(2) == 7\n"
    ctx = _run(tmp_path, monkeypatch, PACKAGE_REPORT, {"R1-1.py": probe})
    [entry] = probe_rerun.read_results(ctx)["probes"]
    assert entry["status"] == "fail" and "AssertionError" in entry["tail"]
    assert len(probe_rerun.defects(probe_rerun.read_results(ctx))) == 1
    assert design_ops.gate_qa(ctx) == "reloop"


def test_the_projects_package_not_importable_is_could_not_run_not_a_defect(tmp_path, monkeypatch):
    """ModuleNotFoundError for the project's own package: the probe never reached the change.
    Listed for the person; not a defect, and the change is not sent back."""
    probe = "# R1: triple\nraise ModuleNotFoundError(\"No module named 'mypkg'\")\n"
    ctx = _run(tmp_path, monkeypatch, PACKAGE_REPORT, {"R1-1.py": probe})
    results = probe_rerun.read_results(ctx)
    [entry] = results["probes"]
    assert entry["status"] == "environment" and "mypkg" in entry["environment"]
    assert probe_rerun.defects(results) == [] and results["unverified"] == []
    table = ctx.read_text(probe_rerun.RESULTS_MD)
    assert "could not run (environment)" in table
    assert "## Could not run (environment, advisory)" in table and "Failing probes" not in table
    assert design_ops.gate_qa(ctx) == "proceed"


def test_a_missing_module_the_change_was_to_add_stays_a_defect(tmp_path, monkeypatch):
    """Only the project's own, existing package is the environment: a submodule or a module
    the checkout does not have is the probe's verdict on the change."""
    for missing in ("mypkg.added", "nothere"):
        probe = f"# R1: x\nimport {missing}\n"
        ctx = _run(tmp_path / missing, monkeypatch, PACKAGE_REPORT, {"R1-1.py": probe})
        [entry] = probe_rerun.read_results(ctx)["probes"]
        assert entry["status"] == "fail", missing
        assert len(probe_rerun.defects(probe_rerun.read_results(ctx))) == 1


def test_a_missing_interpreter_is_could_not_run(tmp_path):
    out = tmp_path / "probes"
    out.mkdir()
    (out / "R1-1.sh").write_text("#!/bin/sh\n# R1: x\nno-such-tool-xyz --version\n")
    target = tmp_path / "target"
    base, _ = _repo(target)
    results = probe_rerun.run_probes(out, target, PACKAGE_REPORT.replace(".py", ".sh"), base=base)
    [entry] = results["probes"]
    assert entry["status"] == "environment" and "no-such-tool-xyz" in entry["environment"]
    assert probe_rerun.defects(results) == []


def test_a_probe_importing_refactory_dsh_runs_from_the_scratch_checkout(tmp_path):
    """This repository: a scratch clone has no environment of its own; the probe runs with
    the check's interpreter (found in the source checkout) and imports refactory_dsh from
    the clone's sources, not an installed copy."""
    repo = Path(__file__).resolve().parents[2]
    checkout = tmp_path / "repo"
    clones.clone_shared(repo, checkout, "HEAD")
    out = tmp_path / "probes"
    out.mkdir()
    (out / "R1-1.py").write_text(
        "# R1: refactory_dsh imports from the checkout\nimport sys\nimport refactory_dsh.probe_rerun as m\n"
        f"assert m.__file__.startswith({str(checkout)!r}), m.__file__\n")
    check = "pnpm test && .venv/bin/python -m pytest -q refactory/tests"
    env = {key: value for key, value in __import__("os").environ.items() if key != "PYTHONPATH"}
    env["PATH"] = "/usr/bin:/bin"
    results = probe_rerun.run_probes(out, checkout, PACKAGE_REPORT, base="HEAD", env=env,
                                     check=check, roots=[repo])
    [entry] = results["probes"]
    assert entry["status"] == "pass", entry["tail"]


# ── base-revert discrimination ──

#: Calls the changed function: 6 on the attempt, 7 on the base, so it fails there.
CALLING_PROBE = "# R1: double(3) is 6\nfrom calc import double\nassert double(3) == 6\n"
#: Carries its own copy of the logic: passes with or without the change.
COPYING_PROBE = "# R1: a copy of double works\ndef my_double(x):\n    return 2 * x\n\nassert my_double(3) == 6\n"
#: Declares preserved behaviour, and checks code the change does not touch.
PRESERVED_PROBE = "# R1: non-regression: triple is unchanged\nfrom mypkg import triple\nassert triple(2) == 6\n"


def _report(*claims: tuple[str, str]) -> str:
    """A verifier report: one PROBED line per (requirement, probe file)."""
    out: dict[str, list[str]] = {}
    for requirement, name in claims:
        out.setdefault(requirement, []).append(f"- Failure points: PROBED reviews/probes/{name}")
    return "".join(f"#### Requirement: {req}\n" + "\n".join(lines) + "\n\n" for req, lines in out.items()
                   ) + "VERDICT: PASS\n"


def _findings(ctx: nn.RunContext) -> dict[str, dict]:
    return {item["id"]: item for item in probe_rerun.read_results(ctx).get("base_revert", [])}


def test_a_requirement_is_verified_when_one_of_its_probes_discriminates(tmp_path, monkeypatch):
    """R1 has a probe that calls the changed function (fails on the base) beside a helper
    that carries its own copy (passes on both): verified, the helper only shown in the
    table. R2 has only the copy: advisory "unverified: probes pass on the base too"."""
    report = _report(("R1", "R1-1.py"), ("R1", "R1-2.py"), ("R2", "R2-1.py"))
    ctx = _run(tmp_path, monkeypatch, report, {
        "R1-1.py": CALLING_PROBE, "R1-2.py": COPYING_PROBE,
        "R2-1.py": COPYING_PROBE.replace("# R1", "# R2")})
    results = probe_rerun.read_results(ctx)
    by_file = {entry["file"]: entry for entry in results["probes"]}
    assert all(entry["status"] == "pass" for entry in by_file.values()), results["probes"]
    assert (by_file["R1-1.py"]["discriminating"], by_file["R1-1.py"]["base_exit_code"]) == (True, 1)
    assert (by_file["R1-2.py"]["discriminating"], by_file["R1-2.py"]["base_exit_code"]) == (False, 0)
    assert by_file["R2-1.py"]["discriminating"] is False
    # Per requirement, not per probe: R1 is verified, R2 is not.
    assert list(_findings(ctx)) == ["R2"]
    assert _findings(ctx)["R2"]["kind"] == "not_discriminating"
    assert _findings(ctx)["R2"]["finding"].startswith("unverified: probes pass on the base too: R2")
    # Never the coverage rule's `unverified`, which would re-run the QA director.
    assert results["unverified"] == []
    table = ctx.read_text(probe_rerun.RESULTS_MD)
    assert "| R1-1.py | R1 | 0 | pass |" in table and "| ✓ |" in table and "| ✗ |" in table
    assert "## Not verified against the base (advisory)" in table
    assert "R1 (" not in table.split("## Not verified against the base")[1]
    assert isinstance(results["base_rerun_wall_s"], (int, float))
    assert "Base rerun wall-clock:" in table


def test_a_requirement_verified_by_one_discriminating_probe_has_no_finding(tmp_path, monkeypatch):
    ctx = _run(tmp_path, monkeypatch, _report(("R1", "R1-1.py"), ("R1", "R1-2.py")),
               {"R1-1.py": CALLING_PROBE, "R1-2.py": COPYING_PROBE})
    results = probe_rerun.read_results(ctx)
    assert results["base_revert"] == [] and results["unverified"] == []
    assert "Not verified against the base" not in ctx.read_text(probe_rerun.RESULTS_MD)
    assert design_ops.gate_qa(ctx) == "proceed"
    assert "probes pass on the base too" not in ctx.read_text("reviews/qa-gate.txt")


def test_a_non_discriminating_requirement_goes_to_the_judge_and_the_follow_ups_once(tmp_path,
                                                                                    monkeypatch):
    """No re-run: the first gate_qa proceeds, with the finding in the synthesis the judge
    reads and in the follow-ups; the coverage rule's re-run is never started for it."""
    ctx = _run(tmp_path, monkeypatch, _report(("R1", "R1-1.py")), {"R1-1.py": COPYING_PROBE})
    assert "unverified: probes pass on the base too: R1" in design_ops.synthesize_qa(ctx)["qa"]
    ctx.write_text(judges.QA_JUDGE, '{"verdict": "proceed", "must_fix": [], "follow_ups": []}')
    assert design_ops.gate_qa(ctx) == "proceed"
    assert "qa_coverage_rerun_done" not in ctx.facts
    [line] = [item for item in json.loads(ctx.read_text("reviews/follow-ups.json"))["qa"]
              if "base too" in item]
    assert line.startswith("unverified: probes pass on the base too: R1 (R1-1.py pass with")
    assert "follow-up: unverified: probes pass on the base too: R1" in ctx.read_text("reviews/qa-gate.txt")


def test_a_requirement_with_only_non_regression_probes_is_reported_as_such(tmp_path, monkeypatch):
    """A probe may declare preserved behaviour, which passes on the base by design; a
    requirement with nothing else is not silently verified."""
    ctx = _run(tmp_path, monkeypatch, _report(("R1", "R1-1.py")), {"R1-1.py": PRESERVED_PROBE})
    [entry] = probe_rerun.read_results(ctx)["probes"]
    assert (entry["status"], entry["discriminating"]) == ("pass", "non_regression")
    finding = _findings(ctx)["R1"]
    assert finding["kind"] == "non_regression_only"
    assert finding["finding"].startswith("verified by non-regression probes only: R1")
    assert "verified by non-regression probes only: R1" in design_ops.synthesize_qa(ctx)["qa"]
    ctx.write_text(judges.QA_JUDGE, '{"verdict": "proceed", "must_fix": [], "follow_ups": []}')
    assert design_ops.gate_qa(ctx) == "proceed"
    assert any(item.startswith("verified by non-regression probes only: R1")
               for item in json.loads(ctx.read_text("reviews/follow-ups.json"))["qa"])


def test_a_probe_the_base_cannot_run_is_base_unavailable_not_discriminating(tmp_path, monkeypatch):
    """An environment failure on the base says nothing about the change."""
    probe = ("# R1: double\nfrom calc import double\nif double(3) == 7:\n"
             "    raise ModuleNotFoundError(\"No module named 'mypkg'\")\nassert double(3) == 6\n")
    ctx = _run(tmp_path, monkeypatch, _report(("R1", "R1-1.py")), {"R1-1.py": probe})
    results = probe_rerun.read_results(ctx)
    [entry] = results["probes"]
    assert results["base_run"]["probes"][0]["status"] == "environment"
    assert entry["discriminating"] == "base_unavailable" and "mypkg" in entry["base_unavailable"]
    finding = _findings(ctx)["R1"]
    assert finding["kind"] == "base_unavailable"
    assert finding["finding"].startswith("base unavailable: R1 could not be checked against the base")
    assert "| base unavailable |" in ctx.read_text(probe_rerun.RESULTS_MD)


def _entry(status: str, code: int | None, header: str = "R1: x") -> dict:
    return {"file": "R1-1.py", "requirement": "R1", "status": status, "exit_code": code,
            "header": header, "existing_tests": [], "tail": ""}


@pytest.mark.parametrize("base_status, base_code", [("timed_out", None), ("fail", 1), ("fail", -9)])
def test_a_probe_that_times_out_or_crashes_on_the_base_discriminates(base_status, base_code):
    results = {"probes": [_entry("pass", 0)], "unverified": []}
    probe_rerun._base_revert_discriminate(results, {"probes": [_entry(base_status, base_code)]}, 0.5)
    assert results["probes"][0]["discriminating"] is True
    assert results["base_revert"] == [] and results["base_rerun_wall_s"] == 0.5


def test_a_base_that_could_not_be_prepared_leaves_every_probe_base_unavailable():
    results = {"probes": [_entry("pass", 0)], "unverified": []}
    probe_rerun._base_revert_discriminate(
        results, {"probes": [], "note": "the base could not be prepared: no clone"}, 0.0)
    assert results["probes"][0]["base_unavailable"] == "the base could not be prepared: no clone"
    assert [item["kind"] for item in results["base_revert"]] == ["base_unavailable"]


def _base_check_worktree(ctx: nn.RunContext) -> Path:
    """The base check's worktree, as `base_check._ensure_base_worktree` leaves it, with
    something the base's setup built in an ignored directory."""
    from refactory_dsh import base_check

    found = worktree.recorded(ctx.workdir)
    path = base_check._base_worktree_path(Path(found.target), Path(found.path).name.removeprefix("run-"))
    path.parent.mkdir(parents=True, exist_ok=True)
    _git(Path(found.target), "worktree", "add", "-q", "--detach", str(path), found.base)
    (path / "lib").mkdir()
    (path / "lib" / "built.js").write_text("built by setup\n")
    return path


def test_the_base_checks_worktree_is_reused_and_never_reset(tmp_path, monkeypatch):
    """With the base check's ``base-<run id>`` worktree at the base, the probes run there:
    nothing is prepared for the base, what its setup built survives, the probe copies are
    gone afterwards, and the base comparison that reuses the worktree still finds it
    prepared."""
    import sys

    from refactory_dsh import base_check

    prepared: list[str] = []
    ctx = _setup(tmp_path, monkeypatch, _report(("R1", "R1-1.py")), {"R1-1.py": CALLING_PROBE},
                 prepared=prepared)
    # The run's worktree has its own directory under the slot, as a real run's has.
    found = worktree.recorded(ctx.workdir)
    run_wt = worktree.slot(Path(found.target)) / "run-abc"
    run_wt.parent.mkdir(parents=True, exist_ok=True)
    _git(Path(found.target), "worktree", "add", "-q", "--detach", str(run_wt), found.head)
    record = json.loads((ctx.workdir / worktree.RECORD).read_text())
    (ctx.workdir / worktree.RECORD).write_text(json.dumps({**record, "path": str(run_wt)}))
    base_wt = _base_check_worktree(ctx)
    check = f"{sys.executable} -m pytest -q tests/test_calc.py"
    ctx.facts["check"] = check
    key = base_check.cache_key(found.base, "", "", check)
    base_check._write_targeted(ctx.workdir / base_check.TARGETED_CACHE_FILE, key, {}, prepared=True)

    probe_rerun.rerun_probes(ctx)

    results = probe_rerun.read_results(ctx)
    assert results["base_run"]["checkout"] == "the base check's worktree"
    assert prepared == ["attempt"]
    assert results["probes"][0]["discriminating"] is True
    assert (base_wt / "lib" / "built.js").read_text() == "built by setup\n"
    assert not (base_wt / "reviews").exists()
    assert _git(base_wt, "rev-parse", "HEAD") == found.base
    assert _git(base_wt, "status", "--porcelain", "--ignored") == "!! lib/"

    def not_again(ctx):
        raise AssertionError("the base was prepared again")

    monkeypatch.setattr(base_check, "_ensure_base_worktree", not_again)
    result, source = base_check.targeted_base_failures(
        ctx, check, ["tests/test_calc.py::test_double"], prepare_timeout=60, timeout=120)
    assert source == "a run of 1 test(s) on the base", result.environment_failure
    assert result.failures == {"tests/test_calc.py::test_double"}


def test_a_base_check_worktree_at_another_commit_is_not_reused(tmp_path, monkeypatch):
    prepared: list[str] = []
    ctx = _setup(tmp_path, monkeypatch, _report(("R1", "R1-1.py")), {"R1-1.py": CALLING_PROBE},
                 prepared=prepared)
    base_wt = _base_check_worktree(ctx)
    _git(base_wt, "checkout", "-q", "--detach", worktree.recorded(ctx.workdir).head)
    probe_rerun.rerun_probes(ctx)
    assert sorted(prepared) == ["attempt", "base"]
    assert probe_rerun.read_results(ctx)["base_run"]["checkout"] == "the probes' base clone"
    assert (base_wt / "lib" / "built.js").exists()


def test_the_fallback_base_clone_is_prepared_once_per_run(tmp_path, monkeypatch):
    """Without the base check's worktree, the probes' own base clone is prepared on the
    first pass only; what was built in it survives the probes and the next pass."""
    prepared: list[str] = []
    ctx = _run(tmp_path, monkeypatch, _report(("R1", "R1-1.py")), {"R1-1.py": CALLING_PROBE},
               prepared=prepared)
    clone = ctx.workdir / ROOT / probe_rerun.BASE_ROOT / "repo"
    (clone / "lib").mkdir()
    (clone / "lib" / "built.js").write_text("built by setup\n")
    probe_rerun.rerun_probes(ctx)
    assert prepared.count("base") == 1 and prepared.count("attempt") == 2
    assert (clone / "lib" / "built.js").exists()
    assert probe_rerun.read_results(ctx)["probes"][0]["discriminating"] is True


@pytest.mark.parametrize("shared", [False, True])
def test_the_base_side_runs_with_its_own_home(tmp_path, monkeypatch, shared):
    """A base-side probe writes into its own HOME, never this process's."""
    process_home = tmp_path / "process-home"
    process_home.mkdir()
    monkeypatch.setenv("HOME", str(process_home))
    probe = ("# R1: double\nimport os, pathlib\nfrom calc import double\n"
             "side = 'base' if double(3) == 7 else 'head'\n"
             "pathlib.Path(os.environ['HOME'], 'mark-' + side).write_text(os.getcwd())\n"
             "assert double(3) == 6\n")
    ctx = _setup(tmp_path, monkeypatch, _report(("R1", "R1-1.py")), {"R1-1.py": probe})
    if shared:
        _base_check_worktree(ctx)
    probe_rerun.rerun_probes(ctx)
    assert not (process_home / "mark-base").exists()
    marks = list((ctx.workdir / ROOT / probe_rerun.BASE_ROOT).rglob("mark-base"))
    assert len(marks) == 1 and marks[0].parent.name == "home"


def test_the_head_and_base_sides_run_at_the_same_time(tmp_path, monkeypatch):
    """Both preparations and both reruns are under way before either finishes: each
    waits for the other at a barrier, which a serial order never reaches."""
    import threading

    barrier = {"prepare": threading.Barrier(2, timeout=20), "run": threading.Barrier(2, timeout=20)}
    events: list[str] = []
    ctx = _setup(tmp_path, monkeypatch, _report(("R1", "R1-1.py")), {"R1-1.py": CALLING_PROBE})
    real_prepare, real_run = clones._prepare, probe_rerun.run_probes

    def prepare(workdir, root, of, **kwargs):
        events.append(f"prepare {of} start")
        barrier["prepare"].wait()
        out = real_prepare(workdir, root, of, **kwargs)
        events.append(f"prepare {of} end")
        return out

    def run(probes, checkout, report, **kwargs):
        side = "base" if report == "" else "head"
        events.append(f"run {side} start")
        barrier["run"].wait()
        out = real_run(probes, checkout, report, **kwargs)
        events.append(f"run {side} end")
        return out

    monkeypatch.setattr(clones, "_prepare", prepare)
    monkeypatch.setattr(probe_rerun, "run_probes", run)
    probe_rerun.rerun_probes(ctx)
    for stage in ("prepare", "run"):
        starts = [i for i, e in enumerate(events) if e.startswith(stage) and e.endswith("start")]
        ends = [i for i, e in enumerate(events) if e.startswith(stage) and e.endswith("end")]
        assert len(starts) == 2 and max(starts) < min(ends), events
    assert probe_rerun.read_results(ctx)["probes"][0]["discriminating"] is True


def test_the_verifier_is_told_never_to_reimplement_the_changed_functions():
    assert "never re-implement or copy their logic into the probe" in DESIGN_VERIFIER_PROMPT
    assert "by importing them from the\nworkspace" in DESIGN_VERIFIER_PROMPT


def test_the_rerun_and_discrimination_run_on_the_spine(tmp_path, monkeypatch):
    """The spine runs the op as its lowered command, `python -m refactory_dsh.run
    refactory_dsh.probe_rerun:rerun_probes`, in a real run directory with the lowered
    declarations: once with the probes' own base clone, once in the base check's worktree."""
    import os
    import sys

    from refactory_dsh import lowering
    from refactory_dsh.run import NODE_ENV, WORKDIR_ENV

    target = tmp_path / "target"
    base, head = _repo(target)
    run_wt = worktree.slot(target) / "run-spine"
    run_wt.parent.mkdir(parents=True, exist_ok=True)
    _git(target, "worktree", "add", "-q", "--detach", str(run_wt), head)
    run = tmp_path / "run"
    (run / worktree.RECORD).parent.mkdir(parents=True)
    (run / worktree.RECORD).write_text(json.dumps({
        "path": str(run_wt), "branch": "b", "base": base, "head": head,
        "target": str(target), "run": str(run)}))
    probes = run / ROOT / "verifier" / "repo" / probe_rerun.PROBES
    probes.mkdir(parents=True)
    (probes / "R1-1.py").write_text(CALLING_PROBE)
    (probes / "R1-2.py").write_text(COPYING_PROBE)
    (probes / "R2-1.py").write_text(COPYING_PROBE.replace("# R1", "# R2"))
    (run / "reviews").mkdir()
    (run / "reviews/verification.md").write_text(
        _report(("R1", "R1-1.py"), ("R1", "R1-2.py"), ("R2", "R2-1.py")))
    (run / "facts.json").write_text("{}")
    state = DesignMode().state_dict()
    lowering.write_declarations(state, run)
    command = lowering.to_workflow(state).nodes["rerun_probes"].command
    assert command == "python -m refactory_dsh.run refactory_dsh.probe_rerun:rerun_probes"
    env = {**os.environ, WORKDIR_ENV: str(run),
           NODE_ENV: json.dumps({"id": "rerun_probes", "type": "FnNode"})}

    def spine() -> dict:
        proc = subprocess.run([sys.executable, *command.split()[1:]], cwd=run, env=env,
                              capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
        return json.loads((run / probe_rerun.RESULTS_JSON).read_text())

    for checkout in ("the probes' base clone", "the base check's worktree"):
        if checkout == "the base check's worktree":
            base_wt = worktree.slot(target) / "base-spine"
            _git(target, "worktree", "add", "-q", "--detach", str(base_wt), base)
            (base_wt / "lib").mkdir()
            (base_wt / "lib" / "built.js").write_text("built by setup\n")
        results = spine()
        assert results["base_run"]["checkout"] == checkout
        by_file = {entry["file"]: entry["discriminating"] for entry in results["probes"]}
        assert by_file == {"R1-1.py": True, "R1-2.py": False, "R2-1.py": False}
        assert [(item["id"], item["kind"]) for item in results["base_revert"]] == [
            ("R2", "not_discriminating")]
        assert results["unverified"] == []
    assert (base_wt / "lib" / "built.js").exists() and not (base_wt / "reviews").exists()


def _failing(file: str, requirement: str, header: str, tail: str) -> dict:
    return {"file": file, "requirement": requirement, "status": "fail", "exit_code": 1,
            "existing_tests": [], "header": header, "tail": tail}


#: A recorded run's first QA round: the judge proceeded, and the one probe that failed
#: tests an ask that, the judge found, cannot be met without breaking another ask.
REPLAY_PROBE = _failing("r1-1.py", "r1", "R1: a wrapped line never starts a finding",
                        "AssertionError: expected no findings, got 1")
REPLAY_ASK = "every form accepted today is still accepted (keep the existing tests passing)"
#: A brief whose two asks conflict: the output must stay byte-identical, and gain a field.
CLI_PROBE = _failing("r2-1.py", "r2", "R2: `tool list --json` output is unchanged",
                     "AssertionError: output differs: new key 'owner'")
CLI_ASK = "add an owner field to the output"
#: A probe that fails because the change is wrong, with no conflict behind it.
DEFECT_PROBE = _failing("r3-1.py", "r3", "R3: parse('') returns []",
                        "TypeError: 'NoneType' object is not iterable")


def _qa_round(tmp_path, probe: dict, dismissal: dict | None = None,
              must_fix: list | None = None) -> nn.RunContext:
    """A QA round after the reviews: ``probe`` failed on rerun and the judge proceeded
    (or, with ``must_fix``, asked for a redo), dismissing what ``dismissal`` names."""
    ctx = nn.RunContext(tmp_path, facts={})
    for name, _ in design_ops.DESIGN_QA_REVIEWS:
        ctx.write_text(name, "ran it; fine\n")
    ctx.write_text(probe_rerun.RESULTS_JSON,
                   json.dumps({"per_probe_s": 120, "probes": [probe], "unverified": []}))
    ctx.write_text(judges.QA_JUDGE, json.dumps({
        "verdict": "redo" if must_fix else "proceed", "must_fix": must_fix or [],
        "follow_ups": [], "dismissed": [dismissal] if dismissal else []}))
    return ctx


def _dismissal(failing: dict, **fields: str) -> dict:
    return {"source": "reviews/probe-results.md", "finding": f"{failing['file']} fails",
            "why": "meeting it breaks the other ask",
            "probe": f"reviews/probes/{failing['file']}", **fields}


def _conflict_line(ctx: nn.RunContext) -> str:
    [line] = [line for line in ctx.read_text("reviews/qa-gate.txt").splitlines()
              if "brief conflict" in line]
    return line


def test_the_recorded_rounds_probe_dismissed_as_a_brief_conflict_goes_to_the_person(tmp_path):
    ctx = _qa_round(tmp_path, REPLAY_PROBE, _dismissal(REPLAY_PROBE, conflicts_with=REPLAY_ASK))
    assert design_ops.gate_qa(ctx) == "proceed"
    line = _conflict_line(ctx)
    assert line.startswith("follow-up: brief conflict for the person to settle: "
                           "reviews/probes/r1-1.py (the verifier's probe for r1 exits 1")
    assert f'conflicts with the ask "{REPLAY_ASK}"' in line
    assert "meeting it breaks the other ask" in line


def test_any_two_conflicting_asks_go_to_the_person_the_same_way(tmp_path):
    ctx = _qa_round(tmp_path, CLI_PROBE, _dismissal(CLI_PROBE, probe="r2-1.py",
                                                    conflicts_with=CLI_ASK))
    assert design_ops.gate_qa(ctx) == "proceed"
    line = _conflict_line(ctx)
    assert "reviews/probes/r2-1.py" in line and f'conflicts with the ask "{CLI_ASK}"' in line


def test_a_failing_probe_without_a_dismissal_still_sends_the_change_back(tmp_path):
    for probe in (REPLAY_PROBE, CLI_PROBE, DEFECT_PROBE):
        verdict = design_ops.gate_qa(_qa_round(tmp_path / probe["file"], probe))
        assert verdict == "reloop" and f"reviews/probes/{probe['file']}" in verdict.reason


def test_a_probe_dismissal_that_names_no_conflicting_ask_still_sends_it_back(tmp_path):
    """A real defect: the judge set the probe aside but named no ask it conflicts with
    (whether a claimed conflict is real is the judge's call; the gate checks the field)."""
    for index, blank in enumerate(({}, {"conflicts_with": ""}, {"conflicts_with": "  "})):
        ctx = _qa_round(tmp_path / str(index), DEFECT_PROBE, _dismissal(DEFECT_PROBE, **blank))
        verdict = design_ops.gate_qa(ctx)
        assert verdict == "reloop" and "reviews/probes/r3-1.py" in verdict.reason
        assert "brief conflict" not in verdict.reason


def test_a_conflict_dismissal_that_names_no_probe_file_still_sends_it_back(tmp_path):
    dismissal = _dismissal(DEFECT_PROBE, probe="", conflicts_with=CLI_ASK)
    verdict = design_ops.gate_qa(_qa_round(tmp_path, DEFECT_PROBE, dismissal))
    assert verdict == "reloop" and "reviews/probes/r3-1.py" in verdict.reason


def test_a_conflict_dismissal_of_one_probe_leaves_the_others_blocking(tmp_path):
    ctx = _qa_round(tmp_path, DEFECT_PROBE, _dismissal(CLI_PROBE, conflicts_with=CLI_ASK))
    verdict = design_ops.gate_qa(ctx)
    assert verdict == "reloop" and "reviews/probes/r3-1.py" in verdict.reason


ANOTHER_MUST_FIX = [{"source": "reviews/code-review.md", "where": "app.py:3", "what": "crashes",
                     "kind": "defect", "severity": "high", "breaks": "ask"}]


def test_a_redo_for_another_item_leaves_the_conflicting_probe_out_of_its_feedback(tmp_path):
    ctx = _qa_round(tmp_path, CLI_PROBE, _dismissal(CLI_PROBE, conflicts_with=CLI_ASK),
                    must_fix=ANOTHER_MUST_FIX)
    verdict = design_ops.gate_qa(ctx)
    assert verdict == "reloop" and "crashes" in verdict.reason
    assert "reviews/probes/r2-1.py" not in verdict.reason


def test_at_the_review_limit_the_draft_still_lists_the_brief_conflict(tmp_path, monkeypatch):
    """The judge dismisses one probe as a brief conflict and redoes for another item, at
    the review limit: the change goes on as a draft, and the conflict is in its follow-ups."""
    from refactory_dsh import create_ops
    from refactory_dsh.run import BuildLoopStop

    def at_the_limit(ctx, problems, *, kind):
        raise BuildLoopStop("review reloop limit reached after 1 review cycle(s)")

    def draft(ctx, stop, *, check_failing=None, restore=True, into="draft_delivery"):
        fact = {"reason": str(stop)}
        ctx.set_fact(into, fact)
        return fact

    monkeypatch.setattr(create_ops, "repair", at_the_limit)
    monkeypatch.setattr(design_ops, "_draft_for_stop", draft)
    ctx = _qa_round(tmp_path, CLI_PROBE, _dismissal(CLI_PROBE, conflicts_with=CLI_ASK),
                    must_fix=ANOTHER_MUST_FIX)
    assert design_ops.gate_qa(ctx) == "proceed"
    assert ctx.read_text("reviews/qa-gate.txt").startswith("draft: review reloop limit")
    [line] = [item for item in json.loads(ctx.read_text("reviews/follow-ups.json"))["qa"]
              if "brief conflict" in item]
    assert "reviews/probes/r2-1.py" in line and f'conflicts with the ask "{CLI_ASK}"' in line


def test_the_qa_judge_is_told_how_to_dismiss_a_failing_probe_once():
    from refactory_dsh.design import QA_JUDGE_PROMPT

    assert QA_JUDGE_PROMPT.count("`conflicts_with`") == 1
    assert '"probe": "", "conflicts_with": ""' in QA_JUDGE_PROMPT


def _gate_on(runtime: str, ctx: nn.RunContext) -> str:
    """``gate_qa``'s verdict on ``runtime``: the walker calls it in-process; the spine runs
    the gate's lowered command (`python -m refactory_dsh.run ...:gate_qa`) and routes on the
    last stdout line."""
    if runtime == "walker":
        return str(design_ops.gate_qa(ctx))
    import os
    import sys

    from refactory_dsh import lowering
    from refactory_dsh.run import NODE_ENV, WORKDIR_ENV

    state = DesignMode().state_dict()
    lowering.write_declarations(state, Path(ctx.workdir))
    command = lowering.to_workflow(state).nodes["gate_qa"].evaluator_command
    assert command.split()[:3] == ["python", "-m", "refactory_dsh.run"]
    env = {**os.environ, WORKDIR_ENV: str(ctx.workdir),
           NODE_ENV: json.dumps({"id": "gate_qa", "type": "GateNode"})}
    proc = subprocess.run([sys.executable, *command.split()[1:]], cwd=ctx.workdir, env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return [line for line in proc.stdout.splitlines() if line.strip()][-1]


@pytest.mark.parametrize("runtime", ["walker", "spine"])
def test_brief_conflict_dismissal_routes_the_same_on_both_runtimes(tmp_path, runtime):
    honoured = _qa_round(tmp_path / "a", CLI_PROBE, _dismissal(CLI_PROBE, conflicts_with=CLI_ASK))
    assert _gate_on(runtime, honoured) == "proceed"
    assert "brief conflict for the person to settle" in honoured.read_text("reviews/qa-gate.txt")
    defect = _qa_round(tmp_path / "b", DEFECT_PROBE, _dismissal(DEFECT_PROBE))
    assert _gate_on(runtime, defect) == "reloop"


# ── a whole design run: the finding reaches the pull request with no extra agent call ──

COPY_REPORT = ("#### Requirement: R1\n- Failure points: PROBED reviews/probes/R1-1.py\n\n"
               "No blocking findings.\n\nVERDICT: PASS\n")
TOOL_COPY_PROBE = ("# R1: build prints built\ndef build():\n    return 'built'\n\n"
                   "assert build() == 'built'\n")


def _design_run_with_verifier(tmp_path: Path, verifier_answer: str):
    """A scripted design run (`test_follow_ups`'s) whose verifier, when it answers
    ``COPY_REPORT``, saves a probe that carries its own copy of the logic."""
    import sys

    from tests_support import (DESIGN_DOC_ANSWER, PITFALLS_ANSWER, RESEARCH_ANSWER, FakeModel,
                               git_repo, render)

    class Model(FakeModel):
        def complete(self, prompt, *, system="", workdir=None, cwd=None, env=None, confine=False):
            answer = super().complete(prompt, system=system, workdir=workdir, cwd=cwd, env=env,
                                      confine=confine)
            if answer.startswith(COPY_REPORT.split("\n\n")[0]) and cwd is not None:
                probes = Path(cwd) / probe_rerun.PROBES
                probes.mkdir(parents=True, exist_ok=True)
                (probes / "R1-1.py").write_text(TOOL_COPY_PROBE)
            return answer

    repo = git_repo(tmp_path / "repo")
    model = Model([RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan", DESIGN_DOC_ANSWER,
                   render({"tool.py": "def build():\n    print('built')\n"}),
                   "health: ok", "code review: ok", "qa: ok", verifier_answer])
    mode = DesignMode()
    state = mode.state_dict()
    state["knobs"]["pr"]["value"] = False
    mode.load_state_dict(state)
    run = mode.run(tmp_path / "run", model=model, verdicts={"gate_strategy": "proceed"},
                   task={"name": "nd_tool", "brief": "a tool", "check": f"{sys.executable} tool.py",
                         "target": str(repo)})
    return run, model


def test_a_non_discriminating_requirement_proceeds_to_delivery_with_no_extra_agent_call(tmp_path):
    """The run's only issue is a requirement whose one probe passes on the base too: it is
    delivered, the finding is in the pull request's follow-ups, and no agent is asked
    anything more than in the same run whose verifier saved no probe."""
    from tests_support import VERIFICATION_ANSWER

    plain, plain_model = _design_run_with_verifier(tmp_path / "plain", VERIFICATION_ANSWER)
    assert plain.exit_ok, plain.error
    run, model = _design_run_with_verifier(tmp_path / "copy", COPY_REPORT)
    assert run.exit_ok, run.error
    results = json.loads((tmp_path / "copy/run" / probe_rerun.RESULTS_JSON).read_text())
    assert [(entry["file"], entry["status"], entry["discriminating"])
            for entry in results["probes"]] == [("R1-1.py", "pass", False)]
    assert [item["kind"] for item in results["base_revert"]] == ["not_discriminating"]
    assert len(model.prompts) == len(plain_model.prompts)
    assert not any("coverage re-run" in prompt.lower() for prompt in model.prompts)
    assert (tmp_path / "copy/run/reviews/qa-gate.txt").read_text().startswith("proceed:")
    pull_request = (tmp_path / "copy/run/pull-request.md").read_text()
    follow_ups = pull_request.split("## Follow-ups\n\n", 1)[1]
    assert ("- unverified: probes pass on the base too: R1 (R1-1.py pass with the change "
            "reverted, so they do not show the change works) (qa)") in follow_ups
