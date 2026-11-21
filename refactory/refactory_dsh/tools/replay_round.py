"""Ask one proposal round of a stopped run again, on a copy, with the evidence as the code
checked out now renders it.

    python -m refactory_dsh.tools.replay_round <run dir> <copy dir> [--epoch N]
        [--project DIR] [--grep WORD,WORD]

For checking a change to what the optimizer reads (`optim.evidence`, the analysts', the
proposer's or the memory pass's prompts) against a round a real run made: the same train
records, the same graph, the same settings and model route, asked again. It:

1. copies ``<run dir>`` to ``<copy dir>`` (which must not exist; a link is copied as what it
   points to, so nothing in the copy leads back into the run); the run is only read;
2. in the copy, cuts the journal before the epoch's proposal round -- the first event of
   ``ROUND_EVENTS`` after its ``epoch_open`` -- and moves the round's analyst cache
   (``analyst_findings/``), the optimizer trace and the resolved-profile snapshot aside
   (``*.before-replay``), so nothing is answered from the old round and a resume with
   newer code is not refused for its changed context;
3. resumes the run's loop on the copy (its ``config.json``, its recorded model route) and
   makes the round's proposal step once (`Loop.optimizer_step`) with the memory pass and
   the installing of proposals left out: the analysts, their merges and the proposer are
   asked; nothing is validated, smoke-tested, evaluated or rolled out. An epoch whose round
   read a memory note reads the one the journal kept (the memory event is before the cut);
4. writes ``<copy dir>/replay-round.md``: each analyst and merge call's findings and the
   proposal's diagnosis, before (the run's own trace) and after, and with ``--grep`` how
   many findings name each word, before and after.

Model calls go through the run's own route (``config.json``'s resolved settings), as a
resumed ``refactory optimize`` would make them.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

#: The events a proposal round journals, the first of which after the epoch opens is where
#: the copy's journal is cut: the analysis, the integrity judge, refusals, the smoke test,
#: the candidates it made and the step that closed it.
ROUND_EVENTS = ("analyst_findings", "integrity_policy", "integrity_verdict", "rejection",
                "smoke", "candidate", "proposal", "step", "epoch_end", "bound")
#: What is moved aside in the copy (suffix ``.before-replay``).
SET_ASIDE = ("analyst_findings", "optimizer_trace.jsonl", "resolved-profile.json")
REPORT = "replay-round.md"


def cut_journal(journal: Path, epoch: int) -> int:
    """Cut ``journal`` before epoch ``epoch``'s proposal round; the lines kept."""
    lines = journal.read_text().splitlines()
    opened = None
    for index, line in enumerate(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            break
        if event.get("event") == "epoch_open" and int(event.get("epoch") or 0) == epoch:
            opened = index
        elif opened is not None and event.get("event") in ROUND_EVENTS:
            journal.write_text("".join(f"{kept}\n" for kept in lines[:index]))
            return index
    raise SystemExit(f"replay_round: no proposal round of epoch {epoch} in {journal}")


def prepare(run_dir: Path, copy_dir: Path, epoch: int) -> int:
    """Copy the run and cut the copy back to the start of ``epoch``'s round."""
    run_dir, copy_dir = run_dir.resolve(), copy_dir.resolve()
    if copy_dir.exists():
        raise SystemExit(f"replay_round: {copy_dir} exists; give a new folder")
    if copy_dir == run_dir or run_dir in copy_dir.parents:
        raise SystemExit("replay_round: the copy must be outside the run")
    # Links are copied as what they point to, so nothing in the copy leads back into the
    # run (a link left dangling is skipped).
    shutil.copytree(run_dir, copy_dir, symlinks=False, ignore_dangling_symlinks=True)
    for name in SET_ASIDE:
        path = copy_dir / name
        if path.exists():
            path.rename(copy_dir / f"{name}.before-replay")
    return cut_journal(copy_dir / "journal.jsonl", epoch)


def build_loop(copy_dir: Path, project: Path, client: Any = None) -> Any:
    """The run's loop, resumed on the copy, with its recorded model route (or ``client``)."""
    from refactory_dsh import strategies
    from refactory_dsh.loop import LoopConfig
    from refactory_dsh.model import DshModelClient

    saved = json.loads((copy_dir / "config.json").read_text())
    config = LoopConfig.from_dict({**saved, "runs_root": str(copy_dir), "resume": True,
                                   "print_events": False})
    if client is None:
        snapshot = config.resolved_profile or {}
        settings = snapshot.get("settings") or saved
        client = DshModelClient(
            routes_resolved=True,
            routing_provenance=(snapshot.get("worker_configuration") or {}).get("provenance"),
            **{key: settings[key] for key in (
                "worker_backend", "provider", "model", "effort", "permission_mode",
                "setting_sources", "role_routes", "node_routes", "provider_config")
               if key in settings},
            **({"claude_executable": ce} if (ce := settings.get("claude_executable")) else {}))
    return strategies.create(config.strategy, saved["mode"], config,
                             strategy_options=config.strategy_options, project=project,
                             model=client)


def ask_round(loop: Any) -> None:
    """The resumed loop's proposal step, once: the memory pass and the installing of its
    proposals left out (`Loop.optimizer_step`)."""
    from refactory_dsh.outerloop import Run

    loop.init()
    loop._memory_pass = lambda: None  # noqa: SLF001 - the round's note is the journal's
    loop.install = lambda run, proposals, parents: []
    loop.optimizer_step(Run(_loop=loop))


def _calls(path: Path, epoch: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            call = json.loads(line)
        except json.JSONDecodeError:
            continue
        if call.get("epoch") == epoch:
            out.append(call)
    return out


def _said(call: dict[str, Any]) -> list[str]:
    """What a call concluded: its findings (each finding and change), or a proposal's
    diagnosis."""
    if call.get("kind") == "proposal":
        return [str(call.get("diagnosis") or "")]
    return [" ".join(str(f.get(key) or "") for key in ("finding", "change", "preserve")).strip()
            for f in call.get("findings") or () if isinstance(f, dict)]


def report(copy_dir: Path, epoch: int, words: Sequence[str]) -> str:
    """The round before and after, as ``replay-round.md`` says it."""
    sides = {"before": _calls(copy_dir / "optimizer_trace.jsonl.before-replay", epoch),
             "after": _calls(copy_dir / "optimizer_trace.jsonl", epoch)}
    lines = [f"# Replayed round: epoch {epoch}", ""]
    if words:
        lines += ["| word | findings naming it before | after |", "|---|---|---|"]
        for word in words:
            counts = [sum(word.lower() in said.lower() for call in sides[side]
                          for said in _said(call)) for side in ("before", "after")]
            lines.append(f"| {word} | {counts[0]} | {counts[1]} |")
        lines.append("")
    for side, calls in sides.items():
        lines += [f"## {side.capitalize()}", ""]
        for call in calls:
            kind = call.get("kind")
            if not str(kind).startswith(("analyst", "proposal")):
                continue
            where = ", ".join(f"{k} {call[k]}" for k in ("group", "level", "batch") if k in call)
            lines.append(f"### {kind}{f' ({where})' if where else ''}"
                         + (f" -- items {', '.join(map(str, call['items']))}"
                            if call.get("items") else ""))
            lines += [f"- {said}" for said in _said(call)] or ["- (nothing)"]
            lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, client: Any = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m refactory_dsh.tools.replay_round",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path, help="the stopped run (only read)")
    parser.add_argument("copy_dir", type=Path, help="a new folder for the copy")
    parser.add_argument("--epoch", type=int, default=1, help="the round's epoch (1)")
    parser.add_argument("--project", type=Path, default=Path("."),
                        help="the project the run was made in (the current folder)")
    parser.add_argument("--grep", default="",
                        help="comma-separated words to count in the findings, before and after")
    args = parser.parse_args(argv)
    kept = prepare(args.run_dir, args.copy_dir, args.epoch)
    print(f"copied to {args.copy_dir}; journal cut to its first {kept} events", file=sys.stderr)
    ask_round(build_loop(args.copy_dir.resolve(), args.project.resolve(), client))
    words = [word.strip() for word in args.grep.split(",") if word.strip()]
    (args.copy_dir / REPORT).write_text(report(args.copy_dir, args.epoch, words) + "\n")
    print(f"the round's calls: {args.copy_dir / 'optimizer_trace.jsonl'}; "
          f"before and after: {args.copy_dir / REPORT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
