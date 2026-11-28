"""Replay real proposer refusals and request sections against the current code.

Each fixture in ``fixtures/proposer_replay/`` was harvested from a real
outer-loop run.  The replay test checks that every entry still produces the
same result (refusal messages, codes, section text) as it did when harvested.

Regenerating expectations
-------------------------
Set ``PROPOSER_REPLAY_REGEN=1`` to overwrite every fixture's expected results
with the current code's output, then commit the updated fixtures.  This is for
intentional product changes only.

``PROPOSER_REPLAY_REGEN=1 PYTHONPATH=$PWD python -m pytest tests/test_proposer_replay.py -x``
"""
from __future__ import annotations

import atexit
import difflib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from proposer_replay import (
    _mode_protection,
    _resolve_mode,
    render_sections,
    replay_attempt,
)

HERE = Path(__file__).resolve().parent
FIXTURE_DIR = HERE / "fixtures" / "proposer_replay"
PROJECT = HERE.parent

REGEN = os.environ.get("PROPOSER_REPLAY_REGEN", "") == "1"


def _fixture_files() -> list[Path]:
    if not FIXTURE_DIR.exists():
        return []
    return sorted(FIXTURE_DIR.glob("*.json"))


def _load_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _unified_diff(a: str, b: str, label: str) -> str:
    return "\n".join(difflib.unified_diff(
        a.splitlines(), b.splitlines(),
        fromfile=f"expected/{label}", tofile=f"actual/{label}",
        lineterm="",
    ))


# ── parametrize over fixture files ──────────────────────────────────────


def _refusal_ids() -> list[tuple[str, int]]:
    ids: list[tuple[str, int]] = []
    for path in _fixture_files():
        fixture = _load_fixture(path)
        for idx in range(len(fixture.get("refusals", []))):
            ids.append((path.stem, idx))
    return ids


def _section_ids() -> list[tuple[str, int]]:
    ids: list[tuple[str, int]] = []
    for path in _fixture_files():
        fixture = _load_fixture(path)
        for idx in range(len(fixture.get("sections", []))):
            ids.append((path.stem, idx))
    return ids


@pytest.fixture(scope="module")
def fixtures() -> dict[str, dict[str, Any]]:
    return {p.stem: _load_fixture(p) for p in _fixture_files()}


# ── refusal replay ──────────────────────────────────────────────────────


@pytest.mark.parametrize("fixture_name,entry_idx", _refusal_ids(),
                         ids=[f"{n}[{i}]" for n, i in _refusal_ids()])
def test_refusal_replay(fixtures: dict, fixture_name: str, entry_idx: int) -> None:
    fixture = fixtures[fixture_name]
    entry = fixture["refusals"][entry_idx]
    mode_id = fixture["mode"]
    frozen = fixture.get("frozen", [])
    states = fixture["states"]

    parent_state = states[entry["parent_state_ref"]]
    seed_state = states[fixture["seed_state_digest"]]
    edit = entry["edit"]
    expected = entry["expected"]

    mode_cls = _resolve_mode(mode_id, PROJECT)
    assert mode_cls is not None, f"mode {mode_id!r} not available"

    protected, graders, invariants = _mode_protection(mode_cls, seed_state)

    actual = replay_attempt(
        edit, parent_state, seed_state, protected, graders, invariants,
        frozen=frozen,
    )

    if REGEN:
        entry["expected"] = actual
        _mark_dirty(fixture_name, fixture)
        return

    if actual != expected:
        msg_parts = [
            f"Refusal replay mismatch: {fixture_name}[{entry_idx}]",
            f"Edit: {json.dumps(edit.get('op', edit.get('edits', '?')), default=str)[:200]}",
        ]
        if actual.get("accepted") != expected.get("accepted"):
            msg_parts.append(
                f"Accepted: expected={expected.get('accepted')}, "
                f"actual={actual.get('accepted')}")
        if not actual.get("accepted") and not expected.get("accepted"):
            exp_reasons = "\n".join(expected.get("reasons", []))
            act_reasons = "\n".join(actual.get("reasons", []))
            if exp_reasons != act_reasons:
                msg_parts.append(_unified_diff(exp_reasons, act_reasons, "reasons"))
        if actual.get("accepted") and expected.get("accepted"):
            if actual.get("candidate_digest") != expected.get("candidate_digest"):
                msg_parts.append(
                    f"Digest: expected={expected.get('candidate_digest')}, "
                    f"actual={actual.get('candidate_digest')}")
        pytest.fail("\n".join(msg_parts))


# ── section replay ──────────────────────────────────────────────────────


@pytest.mark.parametrize("fixture_name,entry_idx", _section_ids(),
                         ids=[f"{n}[{i}]" for n, i in _section_ids()])
def test_section_replay(fixtures: dict, fixture_name: str, entry_idx: int) -> None:
    fixture = fixtures[fixture_name]
    entry = fixture["sections"][entry_idx]
    mode_id = fixture["mode"]
    edit_preset = fixture.get("edit_preset", "broad")
    states = fixture["states"]

    state = states[entry["state_ref"]]
    expected_sections = entry["sections"]

    mode_cls = _resolve_mode(mode_id, PROJECT)
    assert mode_cls is not None, f"mode {mode_id!r} not available"

    seed_state = states[fixture["seed_state_digest"]]
    try:
        protected, graders, _ = _mode_protection(mode_cls, seed_state)
    except Exception:
        protected, graders = (), ()

    actual_sections = render_sections(
        state, mode_cls, edit_preset, protected, graders,
    )

    if REGEN:
        entry["sections"] = actual_sections
        _mark_dirty(fixture_name, fixture)
        return

    for key in sorted(set(expected_sections) | set(actual_sections)):
        exp = expected_sections.get(key, "")
        act = actual_sections.get(key, "")
        if exp != act:
            diff = _unified_diff(exp, act, f"{fixture_name}[{entry_idx}].{key}")
            pytest.fail(
                f"Section mismatch: {fixture_name}[{entry_idx}].{key}\n{diff}")


# ── sabotage checks ────────────────────────────────────────────────────


def _find_refused_by_code(
    code: str, fixtures_map: dict[str, dict[str, Any]],
) -> tuple[dict, dict, str] | None:
    """Return (fixture, entry, fixture_name) for the first entry refused with *code*."""
    for name, fixture in sorted(fixtures_map.items()):
        for entry in fixture.get("refusals", []):
            exp = entry["expected"]
            if not exp["accepted"] and code in exp.get("codes", []):
                return fixture, entry, name
    return None


def _replay_one(fixture: dict, entry: dict) -> dict[str, Any]:
    mode_cls = _resolve_mode(fixture["mode"], PROJECT)
    assert mode_cls is not None
    states = fixture["states"]
    seed_state = states[fixture["seed_state_digest"]]
    parent_state = states[entry["parent_state_ref"]]
    protected, graders, invariants = _mode_protection(mode_cls, seed_state)
    return replay_attempt(
        entry["edit"], parent_state, seed_state, protected, graders, invariants,
        frozen=fixture.get("frozen", []),
    )


def test_sabotage_refusal_message(fixtures: dict) -> None:
    """Monkeypatching a refusal message in product code makes the replay detect it."""
    import refactory_dsh.optim.base as base_mod

    hit = _find_refused_by_code("E-too-many-edits", fixtures)
    if hit is None:
        pytest.skip("no E-too-many-edits refusal in fixtures")
    fixture, entry, name = hit

    original_rule = base_mod.COUNTING_RULE
    try:
        base_mod.COUNTING_RULE = original_rule.replace("changes", "SABOTAGED")
        actual = _replay_one(fixture, entry)
    finally:
        base_mod.COUNTING_RULE = original_rule

    assert actual != entry["expected"], (
        "replay must detect a changed refusal message in product code")
    assert not actual["accepted"]


def test_sabotage_loosen_check(fixtures: dict) -> None:
    """Monkeypatching a limit so a recorded refusal is accepted breaks the replay."""
    import refactory_dsh.optim.base as base_mod
    import proposer_replay as replay_mod

    hit = _find_refused_by_code("E-too-many-edits", fixtures)
    if hit is None:
        pytest.skip("no E-too-many-edits refusal in fixtures")
    fixture, entry, name = hit

    orig_base = base_mod.MAX_EDITS_PER_PROPOSAL
    orig_replay = replay_mod.MAX_EDITS_PER_PROPOSAL
    orig_cls = base_mod.Optimizer.max_edits
    try:
        base_mod.MAX_EDITS_PER_PROPOSAL = 9999
        replay_mod.MAX_EDITS_PER_PROPOSAL = 9999
        base_mod.Optimizer.max_edits = 9999
        actual = _replay_one(fixture, entry)
    finally:
        base_mod.MAX_EDITS_PER_PROPOSAL = orig_base
        replay_mod.MAX_EDITS_PER_PROPOSAL = orig_replay
        base_mod.Optimizer.max_edits = orig_cls

    assert actual != entry["expected"], (
        "replay must detect a loosened check that flips a refusal to acceptance")


def test_sabotage_section_builder(fixtures: dict) -> None:
    """Monkeypatching a section builder line in product code breaks the replay."""
    import refactory_dsh.optim.base as base_mod

    fixture = None
    sec_entry = None
    for f in sorted(fixtures.values(), key=lambda x: x["mode"]):
        for s in f.get("sections", []):
            if "pipeline_view" in s.get("sections", {}):
                fixture = f
                sec_entry = s
                break
        if fixture is not None:
            break
    if fixture is None:
        pytest.skip("no section with pipeline_view in fixtures")

    states = fixture["states"]
    state = states[sec_entry["state_ref"]]
    seed_state = states[fixture["seed_state_digest"]]
    mode_cls = _resolve_mode(fixture["mode"], PROJECT)
    assert mode_cls is not None
    try:
        protected, graders, _ = _mode_protection(mode_cls, seed_state)
    except Exception:
        protected, graders = (), ()

    original_lead = base_mod.PIPELINE_VIEW_LEAD
    try:
        base_mod.PIPELINE_VIEW_LEAD = "SABOTAGED LEAD LINE"
        actual = render_sections(state, mode_cls,
                                 fixture.get("edit_preset", "broad"),
                                 protected, graders)
    finally:
        base_mod.PIPELINE_VIEW_LEAD = original_lead

    expected_pv = sec_entry["sections"]["pipeline_view"]
    assert actual["pipeline_view"] != expected_pv, (
        "replay must detect a changed section builder line in product code")


def test_pure_move_does_not_break(fixtures: dict) -> None:
    """A pure move (helper to another module with re-export) does not break."""
    import importlib
    mod = importlib.import_module("refactory_dsh.optim.base")
    for name in ("Candidate", "Proposal", "candidate_id", "proposal_refusal",
                 "Optimizer", "composition_section", "operator_lines",
                 "pipeline_view_lines"):
        assert hasattr(mod, name), f"{name} missing from refactory_dsh.optim.base"

    fixture = next(iter(sorted(fixtures.values(), key=lambda x: x["mode"])))
    if not fixture["refusals"]:
        pytest.skip("no refusals")

    actual = _replay_one(fixture, fixture["refusals"][0])
    assert actual == fixture["refusals"][0]["expected"]


# ── regen helper ────────────────────────────────────────────────────────


_dirty: dict[str, dict[str, Any]] = {}


def _mark_dirty(name: str, fixture: dict[str, Any]) -> None:
    _dirty[name] = fixture


def _flush_regen() -> None:
    for name, fixture in _dirty.items():
        path = FIXTURE_DIR / f"{name}.json"
        path.write_text(json.dumps(fixture, indent=2, sort_keys=True,
                                   default=str, ensure_ascii=False) + "\n")


atexit.register(_flush_regen)
