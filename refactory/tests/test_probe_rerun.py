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
    (path / "calc.py").write_text("def double(x):\n    return x + x\n")
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


def _run(tmp_path: Path, monkeypatch, report: str, probes: dict[str, str],
         judgment: str = '{"verdict": "proceed", "must_fix": [], "follow_ups": []}') -> nn.RunContext:
    """A design run directory after the verifier: its report, its probe files in its
    workspace clone, the three reviews and the QA judge's verdict; then `rerun_probes`."""
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
        if not repo.exists():
            clones.clone_shared(target, repo, head)
        return {"path": str(repo), "env": {}}

    monkeypatch.setattr(clones, "_prepare", prepare)
    ctx = nn.RunContext(run, facts={})
    ctx.write_text("reviews/verification.md", report)
    for name, _ in design_ops.DESIGN_QA_REVIEWS:
        ctx.write_text(name, "ran it; fine\n")
    ctx.write_text(judges.QA_JUDGE, judgment)
    probe_rerun.rerun_probes(ctx)
    return ctx


REPORT = "#### Requirement: R1\n- Failure points: PROBED {}\n\nVERDICT: PASS\n"
OWN_PROBE = ("# R1: double at the last element; expected 2*x\nfrom calc import double\n"
             "assert double(3) == 6\n")


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
    clock = iter([0.0, 0.0, 10_000.0])
    monkeypatch.setattr(probe_rerun.time, "monotonic", lambda: next(clock, 10_000.0))
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
