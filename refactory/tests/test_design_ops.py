"""Tests for the steer → pick_gate_corrections → apply_corrections pipeline.

Each test that pins a fix fails without its corresponding change:
- T1: one-call verdict+correction through ``_steer`` is collected.
- T3b: two-call form (verdict first, note second) still works.
- T4: process-scoped, CEO-source, info-kind and another-gate corrections are
  not collected.
- T5: the card's command (no ``--kind``) auto-derives constraint via
  ``NOTE_KINDS`` and is collected.
- Bridge: bridge ``answer`` with feedback is collected (passes on main).
- Empty note: an empty ``--note`` in one call records neither verdict nor note.
- Two corrections: two correction notes before resume are both collected.
- Multi-line: a correction containing a ``## `` heading survives whole.
- Escaped boundary: a ``\\## [`` line reaches apply_corrections unescaped.
"""
from __future__ import annotations

from pathlib import Path

import pytest


# ── helpers ──────────────────────────────────────────────────────────────


class _Ctx:
    """Minimal stand-in for the gate evaluator's ctx argument."""

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir
        self.facts: dict[str, str] = {}

    def set_fact(self, name: str, value: str) -> None:
        self.facts[name] = value


def _setup_run(tmp_path: Path) -> Path:
    """Create a minimal run directory with seeded ledger and gate visit."""
    run = tmp_path / "run"
    run.mkdir()
    # Create the files gate_strategy reads so the review hash is stable.
    (run / "design.md").write_text("# placeholder\n")
    (run / "delivery.md").write_text("delivery placeholder\n")
    from refactory_dsh.steering import Ledger
    ledger = Ledger.seed(run, "Build a tool.")
    ledger.visit("gate_strategy", ["design.md", "delivery.md"])
    return run


# ── T1: single steer call with verdict + correction ─────────────────────


def test_single_steer_call_correction_collected(tmp_path):
    """One _steer call with verdict and correction collects the correction.

    Fails without the reorder in _steer: the note lands before the verdict,
    so pick_gate_corrections never sees it after the proceed entry.
    """
    from refactory_dsh.console import _steer
    from refactory_dsh.design_ops import pick_gate_corrections

    run = _setup_run(tmp_path)
    _steer(run, verdicts=["gate_strategy=proceed"], kind="correction", note="X")

    ctx = _Ctx(run)
    result = pick_gate_corrections(ctx)
    assert result == "proceed"
    assert "X" in ctx.facts["gate_corrections"]
    assert "1" in ctx.facts["design_corrections_applied"]


# ── T3b: two-call form ──────────────────────────────────────────────────


def test_two_call_correction_collected(tmp_path):
    """A verdict in one call, then a correction in a later call, is collected.

    Passes on main: this pins the existing two-call behaviour.
    """
    from refactory_dsh.console import _steer
    from refactory_dsh.design_ops import pick_gate_corrections

    run = _setup_run(tmp_path)
    # First call: verdict only.
    _steer(run, verdicts=["gate_strategy=proceed"])
    # Second call: correction only.
    _steer(run, kind="constraint", note="use websockets")

    ctx = _Ctx(run)
    result = pick_gate_corrections(ctx)
    assert result == "proceed"
    assert "use websockets" in ctx.facts["gate_corrections"]
    assert "1" in ctx.facts["design_corrections_applied"]


# ── T4: negative cases ──────────────────────────────────────────────────


def test_non_gate_corrections_not_collected(tmp_path):
    """Process-scoped, CEO-source, info-kind, and another-gate corrections
    are not collected.

    Sub-cases (a), (b), (c) pass on main (existing filtering).
    Sub-case (d) — another gate — catches a regression and fails on main.
    """
    from refactory_dsh.design_ops import pick_gate_corrections
    from refactory_dsh.steering import Ledger

    # (a) Process-scoped constraint after proceed verdict.
    run_a = tmp_path / "run_a"
    run_a.mkdir()
    (run_a / "design.md").write_text("# placeholder\n")
    (run_a / "delivery.md").write_text("delivery\n")
    ledger_a = Ledger.seed(run_a, "Build a tool.")
    ledger_a.visit("gate_strategy", ["design.md", "delivery.md"])
    ledger_a.verdict("gate_strategy", "proceed", approval=True)
    ledger_a.note("constraint", "process thing", scope="process")
    ctx_a = _Ctx(run_a)
    assert pick_gate_corrections(ctx_a) == "skip"
    assert "0" in ctx_a.facts["design_corrections_applied"]

    # (b) Info note after proceed verdict.
    run_b = tmp_path / "run_b"
    run_b.mkdir()
    (run_b / "design.md").write_text("# placeholder\n")
    (run_b / "delivery.md").write_text("delivery\n")
    ledger_b = Ledger.seed(run_b, "Build a tool.")
    ledger_b.visit("gate_strategy", ["design.md", "delivery.md"])
    ledger_b.verdict("gate_strategy", "proceed", approval=True)
    ledger_b.note("info", "just info")
    ctx_b = _Ctx(run_b)
    assert pick_gate_corrections(ctx_b) == "skip"
    assert "0" in ctx_b.facts["design_corrections_applied"]

    # (c) Correction after a non-gate_strategy verdict.
    run_c = tmp_path / "run_c"
    run_c.mkdir()
    (run_c / "design.md").write_text("# placeholder\n")
    (run_c / "delivery.md").write_text("delivery\n")
    ledger_c = Ledger.seed(run_c, "Build a tool.")
    ledger_c.visit("gate_strategy", ["design.md", "delivery.md"])
    ledger_c.verdict("gate_design_review", "proceed", approval=True)
    ledger_c.note("correction", "fix the review")
    ctx_c = _Ctx(run_c)
    assert pick_gate_corrections(ctx_c) == "skip"
    assert "0" in ctx_c.facts["design_corrections_applied"]

    # (d) Correction written before a proceed verdict (earlier unrelated call).
    run_d = tmp_path / "run_d"
    run_d.mkdir()
    (run_d / "design.md").write_text("# placeholder\n")
    (run_d / "delivery.md").write_text("delivery\n")
    ledger_d = Ledger.seed(run_d, "Build a tool.")
    ledger_d.visit("gate_strategy", ["design.md", "delivery.md"])
    ledger_d.note("correction", "stale correction")
    ledger_d.verdict("gate_strategy", "proceed", approval=True)
    ctx_d = _Ctx(run_d)
    assert pick_gate_corrections(ctx_d) == "skip"
    assert "0" in ctx_d.facts["design_corrections_applied"]


# ── T5: card command (no --kind) auto-derives kind ───────────────────────


def test_card_command_note_collected_without_explicit_kind(tmp_path):
    """The card's command omits --kind; NOTE_KINDS maps proceed→constraint,
    which pick_gate_corrections collects.

    Fails without the reorder: the auto-derived constraint note lands before
    the verdict.
    """
    from refactory_dsh.console import _steer
    from refactory_dsh.design_ops import pick_gate_corrections

    run = _setup_run(tmp_path)
    # kind="info" is the default when --kind is not given.
    _steer(run, verdicts=["gate_strategy=proceed"], kind="info",
           note="use websockets")

    ctx = _Ctx(run)
    result = pick_gate_corrections(ctx)
    assert result == "proceed"
    assert "use websockets" in ctx.facts["gate_corrections"]
    assert "1" in ctx.facts["design_corrections_applied"]


# ── Bridge answer test ───────────────────────────────────────────────────


def test_bridge_answer_correction_collected(tmp_path):
    """The bridge's ``answer`` action writes verdict then note; pick collects it.

    Passes on main — the bridge already writes verdict before note.
    """
    from refactory_dsh.design_ops import pick_gate_corrections
    from refactory_dsh.steering import Ledger

    run = tmp_path / "run"
    run.mkdir()
    (run / "design.md").write_text("# placeholder\n")
    (run / "delivery.md").write_text("delivery\n")
    ledger = Ledger.seed(run, "Build a tool.")
    review = ledger.visit("gate_strategy", ["design.md", "delivery.md"])
    # The bridge writes verdict then note, as steering.py main() does.
    ledger.verdict("gate_strategy", "proceed", review=review, answered=True)
    ledger.note("constraint", "use websockets")

    ctx = _Ctx(run)
    result = pick_gate_corrections(ctx)
    assert result == "proceed"
    assert "use websockets" in ctx.facts["gate_corrections"]


# ── Empty note guard ─────────────────────────────────────────────────────


def test_empty_note_records_nothing(tmp_path):
    """An empty --note with a verdict records neither the verdict nor the note.

    Fails without the empty-note guard: with the reorder, the verdict would be
    written first and then Ledger.note would raise on the empty body, leaving a
    verdict without its paired note.
    """
    from refactory_dsh.console import _steer
    from refactory_dsh.steering import Ledger, SteeringError

    run = _setup_run(tmp_path)
    ledger = Ledger(run)
    entries_before = len(ledger.entries())

    with pytest.raises(SteeringError):
        _steer(run, verdicts=["gate_strategy=proceed"], kind="correction",
               note="   ")

    # Nothing was written.
    entries_after = len(ledger.entries())
    assert entries_after == entries_before


# ── Two corrections before resume ────────────────────────────────────────


def test_two_corrections_both_collected(tmp_path):
    """Two correction notes given between proceed and resume are both collected.

    Fails without the collect-all change: the old collector only grabs the
    first entry after the verdict.
    """
    from refactory_dsh.design_ops import pick_gate_corrections
    from refactory_dsh.steering import Ledger

    run = tmp_path / "run"
    run.mkdir()
    (run / "design.md").write_text("# placeholder\n")
    (run / "delivery.md").write_text("delivery\n")
    ledger = Ledger.seed(run, "Build a tool.")
    ledger.visit("gate_strategy", ["design.md", "delivery.md"])
    ledger.verdict("gate_strategy", "proceed", approval=True)
    ledger.note("correction", "first fix")
    ledger.note("correction", "second fix")

    ctx = _Ctx(run)
    result = pick_gate_corrections(ctx)
    assert result == "proceed"
    assert "first fix" in ctx.facts["gate_corrections"]
    assert "second fix" in ctx.facts["gate_corrections"]
    assert "2" in ctx.facts["design_corrections_applied"]


# ── Multi-line correction with heading ───────────────────────────────────


def test_multiline_correction_survives_heading(tmp_path):
    """A correction whose body contains a ``## `` heading is not cut.

    Fails without parse_entries: the raw ``^## `` split treats the heading
    inside the note as a new entry boundary.
    """
    from refactory_dsh.design_ops import pick_gate_corrections
    from refactory_dsh.steering import Ledger

    run = tmp_path / "run"
    run.mkdir()
    (run / "design.md").write_text("# placeholder\n")
    (run / "delivery.md").write_text("delivery\n")
    ledger = Ledger.seed(run, "Build a tool.")
    ledger.visit("gate_strategy", ["design.md", "delivery.md"])
    ledger.verdict("gate_strategy", "proceed", approval=True)
    ledger.note("correction", "line one\n## Heading inside note\nline three")

    ctx = _Ctx(run)
    result = pick_gate_corrections(ctx)
    assert result == "proceed"
    assert "line one" in ctx.facts["gate_corrections"]
    assert "## Heading inside note" in ctx.facts["gate_corrections"]
    assert "line three" in ctx.facts["gate_corrections"]


# ── Escaped boundary line ────────────────────────────────────────────────


def test_escaped_boundary_unescaped(tmp_path):
    r"""An escaped ``\## [`` line in a correction reaches gate_corrections unescaped.

    Fails without parse_entries: the raw split returns the escaped text verbatim.
    """
    from refactory_dsh.design_ops import pick_gate_corrections
    from refactory_dsh.steering import Ledger

    run = tmp_path / "run"
    run.mkdir()
    (run / "design.md").write_text("# placeholder\n")
    (run / "delivery.md").write_text("delivery\n")
    ledger = Ledger.seed(run, "Build a tool.")
    ledger.visit("gate_strategy", ["design.md", "delivery.md"])
    ledger.verdict("gate_strategy", "proceed", approval=True)
    ledger.note("correction", "fix this\n## [looks like a boundary] but is not\nand more")

    ctx = _Ctx(run)
    result = pick_gate_corrections(ctx)
    assert result == "proceed"
    # The escaped line should be unescaped by parse_entries.
    assert "## [looks like a boundary] but is not" in ctx.facts["gate_corrections"]


# ── CEO-source correction not collected ──────────────────────────────────


def test_ceo_source_correction_not_collected(tmp_path):
    """A CEO-source correction is not a gate correction.

    Passes on main — tests the parse_entries-based collector's source filter.
    """
    from refactory_dsh.design_ops import pick_gate_corrections
    from refactory_dsh.steering import Ledger

    run = tmp_path / "run"
    run.mkdir()
    (run / "design.md").write_text("# placeholder\n")
    (run / "delivery.md").write_text("delivery\n")
    ledger = Ledger.seed(run, "Build a tool.")
    ledger.visit("gate_strategy", ["design.md", "delivery.md"])
    ledger.verdict("gate_strategy", "proceed", approval=True)
    ledger.note("correction", "ceo wants this", source="ceo")

    ctx = _Ctx(run)
    assert pick_gate_corrections(ctx) == "skip"
    assert "0" in ctx.facts["design_corrections_applied"]


# ── Fact count on proceed path (DSH-level) ───────────────────────────────


def test_count_fact_survives_on_proceed(tmp_path):
    """pick_gate_corrections sets '1 correction(s)' and apply_corrections
    does not overwrite it (produces entry removed).

    Fails without removing produces from apply_corrections: the executor
    overwrites the count with the full design.md text.
    """
    from refactory_dsh.console import _steer
    from refactory_dsh.design_ops import pick_gate_corrections

    run = _setup_run(tmp_path)
    _steer(run, verdicts=["gate_strategy=proceed"], kind="correction",
           note="fix the API")

    ctx = _Ctx(run)
    result = pick_gate_corrections(ctx)
    assert result == "proceed"
    assert ctx.facts["design_corrections_applied"] == "1 correction(s)"


def test_apply_corrections_updates_every_part_a_correction_touches():
    """The builder reads the whole design, so a correction must not leave ## Changes or
    ## Docs stating what it overrides (a lab run's corrected design kept a stale Changes
    entry). The prompt asks for every touched part -- the light template's sections and
    any optional one the design has -- keeps the untouched sections and the structure,
    and records the corrections in their own section."""
    from refactory_dsh.design import DesignMode

    prompt = " ".join(DesignMode().build().nodes["apply_corrections"].prompt_template.split())
    assert "Do not change the ## Docs section" not in prompt
    assert "## Summary, ## Changes and ## Tests, and any optional section" in prompt
    assert "(such as ## Rule and sites or ## Docs)" in prompt
    assert "add a `## Corrections at approval` section" in prompt
    assert "Leave no part stating what a correction overrides" in prompt
    assert "keep the document's section headings and structure" in prompt


# ── dm-226: a correction given after the gate is applied, never dropped ──────


def _approved(tmp_path: Path, *corrections: str) -> tuple[Path, "_Ctx"]:
    """A run whose design gate_strategy approved (the answer used, as the gate uses it),
    with ``corrections`` given before the gate went on, and pick_corrections run once."""
    from refactory_dsh.design_ops import pick_gate_corrections
    from refactory_dsh.steering import Ledger

    run = _setup_run(tmp_path)
    ledger = Ledger(run)
    entry = ledger.verdict("gate_strategy", "proceed")
    for text in corrections:
        ledger.note("correction", text)
    ledger.take_verdict("gate_strategy", entry["scope"], ["proceed", "reloop"])
    ctx = _Ctx(run)
    pick_gate_corrections(ctx)
    return run, ctx


def test_a_correction_after_the_gate_went_on_is_picked_and_applied_on_the_next_visit(tmp_path):
    """dm-226: the gate used its answer (verdict-use) and pick_corrections ran; a correction
    given then is picked the next time a send-back passes pick_corrections, beside the ones
    already applied, and only once."""
    from refactory_dsh.design_ops import (CORRECTIONS_PICKED, pick_gate_corrections,
                                          unapplied_corrections)
    from refactory_dsh.steering import Ledger

    run, ctx = _approved(tmp_path, "use websockets")
    assert ctx.facts["gate_corrections"] == "- use websockets"
    assert unapplied_corrections(run, ctx.facts) == []
    (run / "design.md").write_text("# corrected once\n")
    Ledger(run).note("correction", "log every reconnect")
    assert unapplied_corrections(run, ctx.facts) == ["log every reconnect"]

    assert pick_gate_corrections(ctx) == "proceed"
    assert ctx.facts["gate_corrections"] == "- use websockets\n\n- log every reconnect"
    assert ctx.facts["design_corrections_applied"] == "2 correction(s)"
    # The backup keeps the design as it was approved, before any correction.
    assert (run / "design.md.pre_corrections").read_text() == "# placeholder\n"
    assert unapplied_corrections(run, ctx.facts) == []
    # Nothing new: the next send-back skips the agent and keeps every correction.
    assert pick_gate_corrections(ctx) == "skip"
    assert ctx.facts["gate_corrections"] == "- use websockets\n\n- log every reconnect"
    assert CORRECTIONS_PICKED in ctx.facts


def test_a_correction_after_the_answer_was_used_but_before_the_pick_is_picked(tmp_path):
    """The gate's answer used (verdict-use) used to close the window, so a correction given
    between it and pick_corrections was dropped."""
    from refactory_dsh.design_ops import pick_gate_corrections
    from refactory_dsh.steering import Ledger

    run = _setup_run(tmp_path)
    ledger = Ledger(run)
    entry = ledger.verdict("gate_strategy", "proceed")
    ledger.take_verdict("gate_strategy", entry["scope"], ["proceed", "reloop"])
    ledger.note("correction", "late but before the pick")
    ctx = _Ctx(run)
    assert pick_gate_corrections(ctx) == "proceed"
    assert ctx.facts["gate_corrections"] == "- late but before the pick"


def test_a_design_approved_again_has_every_correction_applied_again(tmp_path):
    """A reloop at gate_strategy rewrites design.md: the next approval applies its
    corrections from scratch, not only the new ones."""
    from refactory_dsh.design_ops import pick_gate_corrections
    from refactory_dsh.steering import Ledger

    run, ctx = _approved(tmp_path, "use websockets")
    assert pick_gate_corrections(ctx) == "skip"
    # Reapproved with no new correction: the rewritten design still needs the old one.
    Ledger(run).verdict("gate_strategy", "proceed", approval=True)
    (run / "design.md").write_text("# redone\n")
    assert pick_gate_corrections(ctx) == "proceed"
    assert ctx.facts["gate_corrections"] == "- use websockets"
    assert (run / "design.md.pre_corrections").read_text() == "# redone\n"
    Ledger(run).note("correction", "and log reconnects")
    assert pick_gate_corrections(ctx) == "proceed"
    assert ctx.facts["gate_corrections"] == "- use websockets\n\n- and log reconnects"


@pytest.mark.parametrize("where, applies", [({"status": "executing", "nodes": ["qa_director"]}, True),
                                            ({"status": "executing", "nodes": ["deliver"]}, False),
                                            ({"status": "executing", "nodes": [],
                                              "running_nodes": [{"id": "deliver", "kind": "operation"}]},
                                             False),
                                            ({"status": "done"}, False)])
def test_steer_says_what_becomes_of_a_correction_given_after_the_gate(tmp_path, where, applies):
    """Steer no longer answers a bare "Note recorded" for a correction the gate's pick has
    passed: it says the correction is applied before the builder's next round, or that a
    run past its final review will not build it."""
    import json as _json

    from refactory_dsh import readable
    from refactory_dsh.console import _steer

    run, ctx = _approved(tmp_path, "use websockets")
    (run / "facts.json").write_text(_json.dumps(ctx.facts))
    (run / ".refactory").mkdir(exist_ok=True)
    (run / ".refactory/spine-status.json").write_text(_json.dumps(where))
    payload = _steer(run, kind="correction", note="log every reconnect")
    payload["recorded_note"], payload["recorded_kind"] = "log every reconnect", "correction"
    text = readable.format_steer(payload)
    assert payload["late_correction"]["applied"] is applies
    assert payload["late_correction"]["pending"] == ["log every reconnect"]
    assert ("applied the next time the builder is sent back" in text) is applies
    assert ("will not build this correction" in text) is (not applies)
    # Before the gate's pick, and for a note that is not a correction, it says nothing more.
    assert "late_correction" not in _steer(run, kind="info", note="fyi")
    (run / "facts.json").write_text("{}")
    assert "late_correction" not in _steer(run, kind="correction", note="another")


#: The send-backs to the builder: each goes through pick_corrections (dm-226).
SEND_BACKS = ("prepare_check", "gate_self_check", "gate_check", "gate_qa", "gate_overwatch",
              "delivery_gate")
LATE = "name the reconnect counter reconnects"


def test_every_send_back_to_the_builder_goes_through_pick_corrections_on_both_runtimes():
    from refactory_dsh import lowering
    from refactory_dsh.design import DesignMode

    for edges in (DesignMode().build().edges,
                  lowering.to_workflow(DesignMode().state_dict()).edges):
        found = {(edge.source, edge.target, edge.condition) for edge in edges}
        assert {(gate, "pick_corrections", "reloop") for gate in SEND_BACKS} <= found
        assert {(s, c) for s, t, c in found if t == "emit"} == {
            ("pick_corrections", "skip"), ("apply_corrections", None)}


@pytest.mark.usefixtures("fake_gh")
def test_on_the_walker_a_correction_given_during_the_reviews_is_built_before_delivery(
        tmp_path, monkeypatch):
    """dm-226 on the walker: the person corrects the approved design while the first
    attempt is built. The overwatch's gate does not pass the attempt built without it;
    the send-back goes through pick_corrections, apply_corrections writes it into
    design.md, the builder builds it, and the pull request lists it."""
    import test_self_check as T
    from refactory_dsh.steering import Ledger
    from tests_support import (DESIGN_DOC_ANSWER, REVIEW_ANSWERS, VERIFICATION_ANSWER,
                               FakeModel, render)

    T._Chunks(monkeypatch, fails=set())
    run = tmp_path / "run"
    run.mkdir()
    Ledger.seed(run, "a tool")
    Ledger(run).verdict("gate_strategy", "proceed", approval=True)
    corrected = DESIGN_DOC_ANSWER + f'\n## Corrections at approval\n\n- "{LATE}" — § Changes\n'

    class Steered(FakeModel):
        steered = False

        def complete(self, prompt, *, system="", workdir=None, **workspace):
            if T.EMIT in prompt and not self.steered:
                self.steered = True
                Ledger(run).note("correction", LATE)
            return super().complete(prompt, system=system, workdir=workdir, **workspace)

    second = render({"tool.py": "VERSION = 1\nNAME = 'tool'\nreconnects = 0\n"})
    first = render({"tool.py": "VERSION = 1\nNAME = 'tool'\n"})
    model = Steered([*T._answers(first), corrected, second, *REVIEW_ANSWERS[:3],
                     VERIFICATION_ANSWER, REVIEW_ANSWERS[3]])
    result = T._design(run, model, T._target(tmp_path / "repo", T.PINNED))
    assert result.exit_ok, result.error
    applied = [p for p in model.prompts if "corrections when approving the design, or since" in p]
    assert len(applied) == 1 and LATE in applied[0]
    assert sum(T.EMIT in p for p in model.prompts) == 2
    assert sum("You are the overwatch." in p for p in model.prompts) == 2
    assert LATE in result.facts["gate_corrections"]
    assert LATE in (run / "design.md").read_text()
    assert f"- {LATE}" in (run / "pull-request.md").read_text()


@pytest.mark.usefixtures("fake_gh")
def test_on_the_spine_steer_then_the_overwatch_gate_then_pick_corrections_apply_it(
        tmp_path, monkeypatch):
    """dm-226 on the spine, node by node and through the CLI: pick_corrections has run;
    `refactory steer --kind correction` says the correction is applied before the
    builder's next round; gate_overwatch sends the attempt back; pick_corrections, the
    send-back's next node, picks it for apply_corrections."""
    import json as _json
    import subprocess

    import test_self_check as T
    from refactory_dsh.design_ops import CORRECTIONS_PICKED, __name__ as ops
    from refactory_dsh.paths import REPO_ROOT
    from refactory_dsh.steering import Ledger

    run = T._spine_run(T._target(tmp_path / "repo", T.PINNED), tmp_path, monkeypatch,
                       "abcdef700070")
    ledger = Ledger.seed(run, "a tool")
    ledger.verdict("gate_strategy", "proceed", approval=True)
    (run / "design.md").write_text("# placeholder\n")
    code, out, err = T._node(run, f"{ops}:pick_gate_corrections", "pick_corrections", "gate",
                             monkeypatch)
    assert code == 0 and T._label(out) == "skip", err
    assert CORRECTIONS_PICKED in T._facts(run)
    (run / ".refactory").mkdir(exist_ok=True)
    (run / ".refactory/spine-status.json").write_text(_json.dumps(
        {"status": "executing", "nodes": ["qa_director"]}))

    said = subprocess.run([str(REPO_ROOT / ".venv/bin/refactory"), "steer", "--run", str(run),
                           "--note", LATE, "--kind", "correction"],
                          capture_output=True, text=True, timeout=300)
    assert said.returncode == 0, said.stderr
    assert "Note recorded (correction)." in said.stdout
    assert "applied the next time the builder is sent back" in said.stdout

    code, out, err = T._node(run, f"{ops}:gate_overwatch", "gate_overwatch", "gate", monkeypatch)
    assert code == 0 and T._label(out) == "reloop", (out, err)
    assert LATE in out and LATE in T._facts(run)["overwatch_review"]
    code, out, err = T._node(run, f"{ops}:pick_gate_corrections", "pick_corrections", "gate",
                             monkeypatch)
    assert code == 0 and T._label(out) == "proceed", err
    assert T._facts(run)["gate_corrections"] == f"- {LATE}"


# ── a send-back reopens exactly its cycle, on both runtimes (dm-226 review) ──


def _nested():
    """``S → H{proceed: H2, skip: B, reloop: S}``, ``H2{a: A, b: B}``, ``A → B → X{reloop: H,
    proceed: E}``: a send-back from X onto the gate H, which routes through another gate."""
    from refactory_dsh import nn

    graph = nn.Graph()
    for node in ("S", "A", "B", "E"):
        graph.add(nn.Node(node, "op", call="m:f"))
    for gate in ("H", "H2", "X"):
        graph.add(nn.Node(gate, "gate", evaluator="m:g", max_iterations=3))
    for source, target, condition in [("S", "H", None), ("H", "H2", "proceed"), ("H", "B", "skip"),
                                      ("H", "S", "reloop"), ("H2", "A", "a"), ("H2", "B", "b"),
                                      ("A", "B", None), ("B", "X", None), ("X", "H", "reloop"),
                                      ("X", "E", "proceed")]:
        graph.connect(source, target, condition)
    return graph


def _reopened(graph, completed: list[str], gate: str, target: str) -> list[str]:
    from refactory_dsh.nn.record import RunRecord

    record = RunRecord(completed=list(completed))
    record.gate(graph, gate, "reloop", target=target, retry=True, reason="why")
    return [node for node in completed if node not in record.completed]


def test_on_the_walker_a_send_back_reopens_the_cycle_through_nested_gates_and_nothing_else():
    """A, behind the head gate's route through H2, is reopened; S, which the head gate's
    own retry reaches, and E, outside the cycle, are not."""
    graph = _nested()
    done = ["S", "H", "H2", "A", "B", "E"]
    assert sorted(_reopened(graph, done, "X", "H")) == ["A", "B", "H", "H2"]


def test_on_the_walker_design_send_backs_reopen_the_whole_round_and_only_it():
    from refactory_dsh.design import DesignMode

    graph = DesignMode().build()
    before = ["gate_strategy", "pick_corrections", "apply_corrections", "emit", "materialise",
              "prepare_check", "fork_qa", "code_reviewer", "check", "join_qa", "gate_check",
              "synthesize_qa", "qa_judge", "gate_qa", "audit_corrections", "overwatch",
              "gate_overwatch", "report", "deliver"]
    after_check = _reopened(graph, before, "gate_check", "pick_corrections")
    assert {"fork_qa", "code_reviewer", "check", "join_qa", "emit"} <= set(after_check)
    assert not {"gate_strategy", "report", "deliver", "synthesize_qa"} & set(after_check)
    after_delivery = _reopened(graph, before, "delivery_gate", "pick_corrections")
    assert {"report", "deliver", "emit", "gate_overwatch"} <= set(after_delivery)
    assert "gate_strategy" not in after_delivery


def test_a_gates_visit_feedback_is_spent_when_it_decides_on_the_walker():
    from refactory_dsh.nn.record import RunRecord

    graph = _nested()
    record = RunRecord(completed=["S", "H", "H2", "A", "B"])
    record.gate(graph, "X", "reloop", target="H", retry=True, reason="sent back")
    assert record.feedback == {"H": "sent back"}
    record.gate(graph, "H", "skip", target="B", retry=False)
    assert record.feedback == {}


def test_a_pending_verdict_after_the_approval_neither_unapplies_nor_drops_corrections(tmp_path):
    """A steered gate_strategy answer given mid-build, never used by the gate, is not a new
    approval: proceed does not re-apply the corrections (or overwrite the backup), and a
    stray reloop does not drop the ones given after it."""
    from refactory_dsh.design_ops import pick_gate_corrections, unapplied_corrections
    from refactory_dsh.steering import Ledger

    run, ctx = _approved(tmp_path, "use websockets")
    (run / "design.md").write_text("# corrected\n")
    Ledger(run).verdict("gate_strategy", "proceed")
    assert unapplied_corrections(run, ctx.facts) == []
    assert pick_gate_corrections(ctx) == "skip"
    assert (run / "design.md.pre_corrections").read_text() == "# placeholder\n"
    Ledger(run).verdict("gate_strategy", "reloop")
    Ledger(run).note("correction", "log every reconnect")
    assert unapplied_corrections(run, ctx.facts) == ["log every reconnect"]
    assert pick_gate_corrections(ctx) == "proceed"
    assert ctx.facts["gate_corrections"] == "- use websockets\n\n- log every reconnect"


def test_the_same_correction_given_again_is_applied_again(tmp_path):
    from refactory_dsh.design_ops import pick_gate_corrections
    from refactory_dsh.steering import Ledger

    run, ctx = _approved(tmp_path, "use websockets")
    Ledger(run).note("correction", "use websockets")
    assert pick_gate_corrections(ctx) == "proceed"
    assert ctx.facts["gate_corrections"] == "- use websockets\n\n- use websockets"


@pytest.mark.usefixtures("fake_gh")
def test_steer_on_a_finished_walker_run_says_it_will_not_build_the_correction(tmp_path, monkeypatch):
    """The walker records where it stands as the spine does, so steer answers alike."""
    import test_self_check as T
    from refactory_dsh import readable
    from refactory_dsh.console import _steer
    from refactory_dsh.steering import Ledger
    from tests_support import FakeModel, render

    T._Chunks(monkeypatch, fails=set())
    run = tmp_path / "run"
    run.mkdir()
    Ledger.seed(run, "a tool")
    Ledger(run).verdict("gate_strategy", "proceed", approval=True)
    model = FakeModel(T._answers(render({"tool.py": "VERSION = 1\nNAME = 'tool'\n"})))
    assert T._design(run, model, T._target(tmp_path / "repo", T.PINNED)).exit_ok
    payload = _steer(run, kind="correction", note=LATE)
    payload["recorded_note"], payload["recorded_kind"] = LATE, "correction"
    assert payload["late_correction"]["applied"] is False
    assert "will not build this correction" in readable.format_steer(payload)
