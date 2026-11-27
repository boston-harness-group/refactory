"""Harvest and replay real proposer refusals and request sections.

Harvester
---------
``python tests/proposer_replay.py harvest <run_dir>... --out <dir>``

Reads each run directory (journal, rejections, config, optimizer trace),
reconstructs every rejected proposal's parent state, and replays it through
the current code.  One JSON fixture file per run is written under ``<dir>``.

The fixture stores:

- **refusal entries**: each with the parent state, the edit, the recorded
  reasons (reference only), and the *current* result (refused + codes + full
  message, or accepted + a digest of the new state).
- **section entries**: for each distinct parent state, the pipeline view,
  composition section, and operator lines rendered from it.

Regenerating expectations
-------------------------
Set ``PROPOSER_REPLAY_REGEN=1`` and run the replay test to overwrite every
fixture's expected results with the current code's output.  This is for
intentional product changes, not for making a failing test pass silently.

``PYTHONPATH=$PWD python -m pytest tests/test_proposer_replay.py``
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn.mode import Mode
from refactory_dsh.optim import Candidate, Optimizer, Proposal
from refactory_dsh.optim.base import (
    MAX_EDITS_PER_PROPOSAL,
    composition_section,
    operator_lines,
    pipeline_view_lines,
    proposal_refusal,
)

HERE = Path(__file__).resolve().parent
FIXTURE_DIR = HERE / "fixtures" / "proposer_replay"


# ── helpers ──────────────────────────────────────────────────────────────


def _state_digest(state: dict[str, Any]) -> str:
    blob = json.dumps(state, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _sanitize_text(text: str) -> str:
    """Remove absolute paths, hostnames from a string."""
    import re
    text = re.sub(r"/home/\S+?/experiments/\S*", "<run_path>", text)
    text = re.sub(r"/home/\S+", "<home>", text)
    text = re.sub(r"/tmp/\S+", "<tmp>", text)
    return text


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, list):
        return [_sanitize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_value(v) for k, v in value.items()}
    return value


# ── mode resolution ─────────────────────────────────────────────────────


def _resolve_mode(mode_id: str, project: Path) -> type[Mode] | None:
    """Resolve a mode class by its ``package/name`` identifier, or None."""
    try:
        from refactory_dsh.registry import resolve
        _pkg, mode_cls = resolve(project, mode_id)
        return mode_cls
    except Exception:
        return None


def _mode_protection(mode_cls: type[Mode], state: dict[str, Any]
                     ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[Any, ...]]:
    """Return (protected, graders, invariants) for a mode class and state."""
    mode = mode_cls(state=state)
    return (
        tuple(mode.edit_protected()),
        tuple(mode.grader_nodes()),
        mode.invariants(),
    )


# ── attempt replay ──────────────────────────────────────────────────────


def replay_attempt(
    edit: dict[str, Any],
    parent_state: dict[str, Any],
    seed_state: dict[str, Any],
    protected: tuple[str, ...],
    graders: tuple[str, ...],
    invariants: tuple[Any, ...],
    frozen: Sequence[str] = (),
    seen: Sequence[str] = (),
) -> dict[str, Any]:
    """Replay one edit through the real attempt path, return the result."""
    proposal = Proposal(edit=edit)
    refused = proposal_refusal(proposal, max_edits=MAX_EDITS_PER_PROPOSAL,
                               parent_state=parent_state)
    if refused:
        return {"accepted": False, "reasons": list(refused),
                "codes": sorted({_reason_code(r) for r in refused})}

    optimizer = Optimizer()
    parent = Candidate.from_state(parent_state)
    outcome = optimizer.attempt(
        proposal, parent, invariants, True, frozen, sorted(seen),
        seed_state=seed_state, protected=protected, graders=graders,
    )
    if isinstance(outcome, Candidate):
        return {"accepted": True, "candidate_digest": _state_digest(outcome.state),
                "candidate_id": outcome.id}
    return {"accepted": False, "reasons": list(outcome),
            "codes": sorted({_reason_code(r) for r in outcome})}


def _reason_code(reason: str) -> str:
    from refactory_dsh.optim.base import reason_code
    return reason_code(reason)


# ── section rendering ───────────────────────────────────────────────────


def render_sections(
    state: dict[str, Any],
    mode_cls: type[Mode],
    edit_preset: str = "broad",
    protected: Sequence[str] = (),
    graders: Sequence[str] = (),
) -> dict[str, str]:
    """Render the graph-dependent request sections from a state."""
    mode = mode_cls(state=state)
    described = mode.describe()
    node_states = [dict(ns) for ns in
                   (state.get("graph") or {}).get("nodes", {}).values()]
    edges = described["edges"]
    knobs_map = described["knobs"]
    knobs_list = [{"name": name, **spec} for name, spec in knobs_map.items()]
    catalogue = nn_edits.catalogue(edit_preset)

    sections: dict[str, str] = {}

    pv = pipeline_view_lines(node_states, edges, knobs_list,
                             list(protected), list(graders))
    sections["pipeline_view"] = "\n".join(pv)

    comp = composition_section(catalogue, node_states, edges,
                               protected=list(protected),
                               knob_values={k.get("name"): k.get("value")
                                            for k in knobs_list})
    sections["composition"] = "\n".join(comp)

    ops = operator_lines(catalogue, node_states, edges,
                         protected=list(protected),
                         knob_values={k.get("name"): k.get("value")
                                      for k in knobs_list})
    sections["operator_lines"] = "\n".join(ops)

    return sections


# ── journal parsing ─────────────────────────────────────────────────────


def _parse_journal(journal_path: Path
                   ) -> tuple[dict[str, Any] | None,
                              dict[str, dict[str, Any]],
                              dict[int, str]]:
    """Parse a journal.

    Returns (seed_state, candidate_states_by_id, parent_by_epoch).
    """
    seed_state: dict[str, Any] | None = None
    candidates: dict[str, dict[str, Any]] = {}
    parent_by_epoch: dict[int, str] = {}

    with journal_path.open() as f:
        for line in f:
            entry = json.loads(line)
            event = entry.get("event")
            if event == "original_seed":
                seed_state = entry["state"]
            elif event == "candidate" and "state" in entry:
                candidates[entry["id"]] = entry["state"]
            elif event == "proposal" and "parent" in entry and "epoch" in entry:
                parent_by_epoch.setdefault(entry["epoch"], entry["parent"])

    return seed_state, candidates, parent_by_epoch


def _parse_rejections(path: Path) -> list[dict[str, Any]]:
    """Parse rejections.jsonl, return list of rejection dicts."""
    entries: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def _parse_journal_rejections(journal_path: Path) -> list[dict[str, Any]]:
    """Parse rejection events from the journal (they carry epoch info)."""
    entries: list[dict[str, Any]] = []
    with journal_path.open() as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("event") == "rejection":
                entries.append(entry)
    return entries


# ── harvesting ──────────────────────────────────────────────────────────


def harvest_run(
    run_dir: Path,
    project: Path,
) -> dict[str, Any] | None:
    """Harvest one run directory into a fixture dict, or None on skip."""
    config_path = run_dir / "config.json"
    journal_path = run_dir / "journal.jsonl"
    rejections_path = run_dir / "rejections.jsonl"

    if not config_path.exists() or not journal_path.exists():
        print(f"  SKIP {run_dir}: missing config.json or journal.jsonl",
              file=sys.stderr)
        return None

    config = json.loads(config_path.read_text())
    mode_id = config["mode"]
    edit_preset = config.get("edit_preset", "broad")
    frozen = list(config.get("frozen", []))

    mode_cls = _resolve_mode(mode_id, project)
    if mode_cls is None:
        print(f"  SKIP {run_dir}: mode {mode_id!r} not available",
              file=sys.stderr)
        return None

    seed_state, candidates, parent_by_epoch = _parse_journal(journal_path)
    if seed_state is None:
        print(f"  SKIP {run_dir}: no original_seed in journal",
              file=sys.stderr)
        return None

    # Build the set of existing candidate ids (for the "seen" check)
    all_seen = set(candidates.keys())

    # Deduplicated state pool: digest → state dict.
    states: dict[str, dict[str, Any]] = {}

    def _store(state: dict[str, Any]) -> str:
        digest = _state_digest(state)
        if digest not in states:
            states[digest] = state
        return digest

    seed_digest = _store(seed_state)

    # ── refusal entries ─────────────────────────────────────────────
    refusal_entries: list[dict[str, Any]] = []
    seen_pairs: set[str] = set()

    journal_rejections = _parse_journal_rejections(journal_path)
    if not rejections_path.exists():
        print(f"  NOTE {run_dir}: no rejections.jsonl, using journal only",
              file=sys.stderr)
        raw_rejections = journal_rejections
    else:
        raw_rejections = _parse_rejections(rejections_path)

    skipped: list[str] = []
    for idx, rej in enumerate(raw_rejections):
        edit = rej.get("edit", {})
        epoch = rej.get("epoch")

        # Find parent: from journal rejection epoch, or from rejections.jsonl
        # which doesn't carry epoch — match by edit against journal rejections
        if epoch is None:
            for jr in journal_rejections:
                if jr.get("edit") == edit and jr.get("reasons") == rej.get("reasons"):
                    epoch = jr.get("epoch")
                    break
        if epoch is None:
            epoch = min(parent_by_epoch.keys()) if parent_by_epoch else 1

        parent_id = parent_by_epoch.get(epoch)
        if parent_id is None:
            skipped.append(f"rejection {idx}: no parent found for epoch {epoch}")
            continue

        parent_state = candidates.get(parent_id)
        if parent_state is None:
            skipped.append(f"rejection {idx}: parent {parent_id} state not in journal")
            continue

        parent_digest = _store(parent_state)

        # Dedup by (parent_state_digest, edit)
        pair_key = parent_digest + ":" + json.dumps(
            edit, sort_keys=True, default=str)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        try:
            protected, graders, invariants = _mode_protection(mode_cls, seed_state)
        except Exception as exc:
            skipped.append(f"rejection {idx}: cannot build mode from seed: {exc}")
            continue

        try:
            result = replay_attempt(
                edit, parent_state, seed_state, protected, graders, invariants,
                frozen=frozen, seen=sorted(all_seen - {parent_id}),
            )
        except Exception as exc:
            skipped.append(f"rejection {idx}: replay error: {type(exc).__name__}: {exc}")
            continue

        entry: dict[str, Any] = {
            "parent_state_ref": parent_digest,
            "edit": edit,
            "recorded_reasons": _sanitize_value(rej.get("reasons", [])),
            "recorded_codes": _sanitize_value(rej.get("codes", [])),
            "expected": result,
        }
        refusal_entries.append(entry)

    # ── section entries ─────────────────────────────────────────────
    section_entries: list[dict[str, Any]] = []
    seen_section_digests: set[str] = set()

    parent_states_for_sections: list[tuple[str, dict[str, Any]]] = []
    for epoch in sorted(parent_by_epoch.keys()):
        pid = parent_by_epoch[epoch]
        pstate = candidates.get(pid)
        if pstate is None:
            continue
        digest = _store(pstate)
        if digest in seen_section_digests:
            continue
        seen_section_digests.add(digest)
        parent_states_for_sections.append((digest, pstate))

    if seed_digest not in seen_section_digests:
        seen_section_digests.add(seed_digest)
        parent_states_for_sections.insert(0, (seed_digest, seed_state))

    try:
        protected_for_sections, graders_for_sections, _ = _mode_protection(
            mode_cls, seed_state)
    except Exception:
        protected_for_sections = ()
        graders_for_sections = ()

    for digest, pstate in parent_states_for_sections:
        try:
            sections = render_sections(
                pstate, mode_cls, edit_preset,
                protected_for_sections, graders_for_sections,
            )
        except Exception as exc:
            skipped.append(f"section for state {digest[:8]}: "
                           f"{type(exc).__name__}: {exc}")
            continue

        section_entries.append({
            "state_ref": digest,
            "sections": sections,
        })

    fixture = {
        "mode": mode_id,
        "edit_preset": edit_preset,
        "frozen": frozen,
        "seed_state_digest": seed_digest,
        "states": states,
        "refusals": refusal_entries,
        "sections": section_entries,
        "skipped": skipped,
    }
    return fixture


def _find_run_dirs(path: Path) -> list[Path]:
    """Find actual run directories (containing config.json + journal.jsonl)."""
    if (path / "config.json").exists() and (path / "journal.jsonl").exists():
        return [path]
    runs: list[Path] = []
    for child in sorted(path.iterdir()):
        if child.is_dir():
            runs.extend(_find_run_dirs(child))
    return runs


def _fixture_name(run_dir: Path) -> str:
    """A short, filesystem-safe fixture name from a run directory path."""
    parts = run_dir.parts
    # Take the last 2-3 meaningful parts
    meaningful = [p for p in parts[-4:] if p not in ("run", "runs")]
    name = "_".join(meaningful[-3:])
    # Clean up
    name = name.replace("/", "_").replace(" ", "_")
    return name


def harvest(run_dirs: list[str], out_dir: str) -> None:
    """Harvest multiple run directories into fixture files."""
    project = Path(__file__).resolve().parent.parent
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for run_path_str in run_dirs:
        run_path = Path(run_path_str)
        actual_dirs = _find_run_dirs(run_path)
        if not actual_dirs:
            print(f"SKIP {run_path}: no run directories found", file=sys.stderr)
            continue

        for run_dir in actual_dirs:
            print(f"Harvesting {run_dir} ...", file=sys.stderr)
            fixture = harvest_run(run_dir, project)
            if fixture is None:
                continue

            name = _fixture_name(run_dir)
            dest = out / f"{name}.json"

            fixture_text = json.dumps(fixture, indent=2, sort_keys=True,
                                      default=str, ensure_ascii=False)
            dest.write_text(fixture_text + "\n")

            n_refusals = len(fixture["refusals"])
            n_sections = len(fixture["sections"])
            n_skipped = len(fixture["skipped"])
            print(f"  -> {dest.name}: {n_refusals} refusals, "
                  f"{n_sections} sections, {n_skipped} skipped",
                  file=sys.stderr)
            if fixture["skipped"]:
                for reason in fixture["skipped"]:
                    print(f"     skipped: {reason}", file=sys.stderr)


# ── CLI ─────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Harvest proposer replay fixtures from run directories")
    sub = parser.add_subparsers(dest="command")

    harvest_cmd = sub.add_parser("harvest",
                                 help="Harvest fixtures from run directories")
    harvest_cmd.add_argument("run_dirs", nargs="+",
                             help="Run directories to harvest")
    harvest_cmd.add_argument("--out", required=True,
                             help="Output directory for fixtures")

    args = parser.parse_args()
    if args.command == "harvest":
        harvest(args.run_dirs, args.out)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
