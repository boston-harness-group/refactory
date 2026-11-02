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
