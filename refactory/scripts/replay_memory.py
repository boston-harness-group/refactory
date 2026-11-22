"""Re-render a run's recorded memory requests with the current record prompt, offline.

    uv run python refactory/scripts/replay_memory.py RUN_DIR [--out DIR] [--epoch N]
    uv run python refactory/scripts/replay_memory.py RUN_DIR --epoch N --reply REPLY.json
    uv run python refactory/scripts/replay_memory.py RUN_DIR --epoch N --journal \
        [--reply E=REPLY.json ...] [--swap-proposal]

Reads ``RUN_DIR/optimizer_trace.jsonl`` (read only) and, for every memory call it
recorded -- the single call or a split pass's synthesis (kind ``memory``) and its
comparison calls (kind ``memory_compare``) -- writes the request the current code would
send in its place: ``OUT/epoch_<N>/<call>.system.txt`` and ``<call>.prompt.txt``, plus
``OUT/manifest.json`` saying what each came from and what could not be rebuilt. No model
is called; the files are for replaying through a model by hand.

A request recorded before the record (the optimizer-coach prompt) is converted section
by section: its title, the previous-note section (dropped: the record carries earlier
observations itself), the regressions heading and the answer format change; the
objective, both pipelines, what changed, "What was tried this epoch" and the
comparison with every trace are kept byte for byte, as the current code renders them
from the same inputs. A request recorded with the current prompt is written as it is.

A split pass's synthesis is rebuilt from the comparison replies the trace recorded,
which the old comparison prompt wrote: replay the comparison requests first and pass
their replies (``--compare-replies DIR``, one ``part<k>.json`` per part) for a synthesis
made from the new ones. The record's facts (the train items each change fixed and broke,
by id, and the items that keep failing) are not part of any request -- the harness adds
them after the answer -- and are not rebuilt here: a recorded request holds only the
counts. ``--reply`` reads one model reply to a re-rendered note request as the record
would (`memory.note_from`, the request's own items and change ids): what is kept, and
what is stripped and why.

``--journal`` rebuilds the record's facts from ``RUN_DIR/journal.jsonl`` as the loop had
them at each memory pass up to epoch N: the run's mode and ``config.json`` make a `Loop`
whose working directory is a temporary one (nothing is written into the run), its
journal replayed up to that pass's ``optimizer_memory`` event (`Loop._replay`: the
``train_records``/``train_records_epoch``, ``proposal``, ``strategy_state`` val_gate,
``rejection`` and ``integrity_verdict`` events), and the loop's own `_tried_in_round` and
`Loop._record_facts` read from it, over every round so far. Each pass's observations
come from the reply given for it (``--reply E=FILE``, read as the loop reads one,
`memory.note_from`, pinned only to the changes that pass compared), or none (the facts
alone; a pass that failed in the run and gets no reply is left out of the passes, as
the loop leaves it). The record of epoch N, rendered as the
readers see it (`memory.render_record`), goes to ``OUT/epoch_N/record.json`` (and the
whole record, nothing left out, to ``record.full.json``). The mode's package must be
importable (``--project``).

``--swap-proposal`` (with ``--journal``) takes epoch N's last recorded proposal request
(kind ``proposal``) and writes three: ``proposal.old`` as recorded, ``proposal.record``
with its "## Optimizer memory" section replaced by the record as the request shows it
(`memory.request_lines` of the note the proposal read, its text the rendered record),
and ``proposal.none`` with that section removed, heading and all; each as
``.prompt.txt`` and ``.system.txt``. The section is found by the request's own headings
and must be there exactly once.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from refactory_dsh.optim import memory

#: How a recorded request names itself: the old title (and its comparison calls'), and
#: the current one.
OLD_TITLE = "# Writing the optimizer's memory -- epoch "
OLD_COMPARE_TITLE = "# Comparing two rollouts for the optimizer's memory -- epoch "
NEW_COMPARE_TITLE = "# Comparing two rollouts for the run's record -- epoch "
OLD_REGRESSIONS = "### Regressions (right -> wrong) -- HIGHEST PRIORITY"
NEW_REGRESSIONS = "### Regressions (right -> wrong)"
OLD_SECTION = "\n## Previous optimizer memory\n"
COMPARISON = "## Longitudinal comparison ("
ANSWER = "\n\n## Answer\n"
OLD_SYNTHESIS_LINE = "observations for you. They are your evidence from the comparison."
NEW_SYNTHESIS_LINE = "observations. They are your evidence from the comparison."


def _answer_tail(prompt: str, answer: str) -> str:
    at = prompt.rfind(ANSWER)
    if at == -1:
        raise ValueError("the recorded request has no '## Answer' section")
    return prompt[:at] + ANSWER + answer


def _headings(prompt: str) -> str:
    return "\n".join(NEW_REGRESSIONS if line == OLD_REGRESSIONS else line
                     for line in prompt.split("\n"))


def convert_note(prompt: str) -> str:
    """A recorded single or synthesis request, in the current form (`memory.build_prompt`,
    `MemoryRequest.synthesis_prompt`)."""
    if prompt.startswith(memory.REQUEST_TITLE):
        return prompt
    if not prompt.startswith(OLD_TITLE):
        raise ValueError("not a memory request this script knows")
    text = memory.REQUEST_TITLE + " -- epoch " + prompt[len(OLD_TITLE):]
    start = text.find(OLD_SECTION)
    if start != -1:
        end = text.find(COMPARISON, start)
        if end == -1:
            raise ValueError("the previous-note section is not followed by the comparison")
        text = text[:start + 1] + text[end:]
    text = text.replace(OLD_SYNTHESIS_LINE, NEW_SYNTHESIS_LINE, 1)
    return _answer_tail(_headings(text), memory.answer_text())


def convert_compare(prompt: str) -> str:
    """A recorded comparison request, in the current form (`MemoryRequest.compare_prompt`)."""
    if prompt.startswith(NEW_COMPARE_TITLE):
        return prompt
    if not prompt.startswith(OLD_COMPARE_TITLE):
        raise ValueError("not a comparison request this script knows")
    text = NEW_COMPARE_TITLE + prompt[len(OLD_COMPARE_TITLE):]
    return _answer_tail(_headings(text), memory.compare_answer_text())


def synthesis_from(prompt: str, observations: list[dict[str, Any]]) -> str:
    """A converted synthesis request with its parts' observations replaced by new ones
    (``observations``: one ``{"part", "items", "observations"}`` per part), rendered as
    `MemoryRequest.synthesis_prompt` renders them (without its cut to the room)."""
    start = prompt.find("\n### Part 1 of ")
    end = prompt.rfind(ANSWER)
    if start == -1 or end == -1:
        raise ValueError("the synthesis request has no parts' observations to replace")
    found = [memory.fit_observations(list(r.get("observations") or ()))[0] for r in observations]
    parts = [f"### Part {r.get('part')} of {len(observations)}: items "
             f"{', '.join(str(i) for i in r.get('items') or ())}\n"
             + ("\n".join(memory._observation_line(o) for o in kept) or "(no observation)")
             for r, kept in zip(observations, found)]
    return prompt[:start + 1] + "\n\n".join(parts) + prompt[end:]


def _calls(run: Path, epoch: int | None) -> list[dict[str, Any]]:
    """Each memory call's last recorded attempt, by (epoch, kind, part)."""
    trace = run / "optimizer_trace.jsonl"
    if not trace.exists():
        raise SystemExit(f"no optimizer_trace.jsonl in {run}")
    last: dict[tuple[Any, ...], dict[str, Any]] = {}
    for line in trace.read_text().splitlines():
        try:
            call = json.loads(line)
        except json.JSONDecodeError:
            continue
        if call.get("kind") not in ("memory", "memory_compare") or not call.get("prompt"):
            continue
        if epoch is not None and call.get("epoch") != epoch:
            continue
        last[(call.get("epoch"), call.get("kind"), call.get("part") or 0)] = call
    return [last[key] for key in sorted(last, key=lambda k: (k[0] or 0, k[1] != "memory_compare",
                                                             k[2]))]


def items_and_changes(prompt: str) -> tuple[list[str], list[str]]:
    """The compared items a re-rendered note request shows (its traced items, or a
    synthesis's parts' items, and its stable successes) and the change ids its "What was
    tried this epoch" names."""
    items = re.findall(r"^#### Task (\S+?)(?::| --|$)", prompt, re.M)
    for line in re.findall(r"^### Part \d+ of \d+: items (.*)$", prompt, re.M):
        items += [i.strip() for i in line.split(",") if i.strip()]
    tried = prompt.split("## What was tried this epoch", 1)[-1].split(COMPARISON, 1)[0]
    changes = re.findall(r"^- `([^`]+)` \(round ", tried, re.M)
    return list(dict.fromkeys(items)), list(dict.fromkeys(changes))


def replay(run: Path, out: Path, epoch: int | None = None,
           compare_replies: Path | None = None) -> dict[str, Any]:
    """Write every re-rendered request under ``out`` and return the manifest."""
    manifest: dict[str, Any] = {"run": str(run), "calls": [], "not_rebuilt": [
        "the record's facts (fixed/broken item ids, keeps_failing): the harness adds them "
        "after the answer from the run's journal; a recorded request holds only counts"]}
    for call in _calls(run, epoch):
        name = (f"compare_part{call.get('part')}" if call["kind"] == "memory_compare"
                else "synthesis" if call.get("parts") else "note")
        folder = out / f"epoch_{call.get('epoch')}"
        folder.mkdir(parents=True, exist_ok=True)
        entry: dict[str, Any] = {"epoch": call.get("epoch"), "kind": call["kind"], "name": name}
        try:
            if call["kind"] == "memory_compare":
                system, prompt = memory.COMPARE_SYSTEM, convert_compare(call["prompt"])
            else:
                system = memory.SYNTHESIS_SYSTEM if call.get("parts") else memory.MEMORY_SYSTEM
                prompt = convert_note(call["prompt"])
                if call.get("parts"):
                    replies = (compare_replies / f"epoch_{call.get('epoch')}"
                               if compare_replies else None)
                    if replies is not None and replies.is_dir():
                        prompt = synthesis_from(prompt, _new_observations(run, call, replies))
                        entry["observations"] = f"from {replies}"
                    else:
                        entry["observations"] = ("the recorded comparison replies (written "
                                                 "under the old comparison prompt)")
        except ValueError as error:
            entry["error"] = str(error)
            manifest["calls"].append(entry)
            continue
        (folder / f"{name}.system.txt").write_text(system)
        (folder / f"{name}.prompt.txt").write_text(prompt)
        entry.update({"system": str(folder / f"{name}.system.txt"),
                      "prompt": str(folder / f"{name}.prompt.txt"),
                      "converted": prompt != call["prompt"] or system != call.get("system"),
                      "chars": len(system) + len(prompt)})
        manifest["calls"].append(entry)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _new_observations(run: Path, synthesis: dict[str, Any], replies: Path) -> list[dict[str, Any]]:
    """Each part's observations from a reply file (``part<k>.json``) in ``replies``, read
    as the loop reads a comparison reply (`memory.parse_observations`)."""
    from refactory_dsh.optim.random_optimizer import reply_object

    parts = [c for c in _calls(run, synthesis.get("epoch")) if c["kind"] == "memory_compare"]
    out = []
    for call in parts:
        path = replies / f"part{call.get('part')}.json"
        if not path.exists():
            raise ValueError(f"no reply for comparison part {call.get('part')}: {path}")
        found = memory.parse_observations(reply_object(path.read_text()))
        if found is None:
            raise ValueError(f"{path} has no observations list")
        out.append({"part": call.get("part"), "items": call.get("items") or [],
                    "observations": found})
    return out


def check_reply(prompt: str, reply: str, epoch: int | None) -> dict[str, Any]:
    """What the record keeps of one reply to a re-rendered note request, and what it
    strips and why (`memory.note_from`, with the request's items and change ids)."""
    from refactory_dsh.optim.random_optimizer import reply_object

    items, changes = items_and_changes(prompt)
    facts = {"changes": [{"id": c} for c in changes], "keeps_failing": []}
    note = memory.note_from(reply_object(reply), epoch=epoch, facts=facts, items=items)
    if note is None:
        return {"error": 'the reply has no "observations" list'}
    return {"kept": note["record"]["observations"], "stripped": note["stripped"],
            "unpinned": note["unpinned"]}


# ── the record's facts from the journal, as the loop had them ───────────────


def _events(run: Path) -> list[dict[str, Any]]:
    """The run's journal, read as `Loop._replay` reads it (a half-written last line ends it)."""
    journal = run / "journal.jsonl"
    if not journal.exists():
        raise SystemExit(f"no journal.jsonl in {run}")
    out = []
    for line in journal.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return out


def _passes(events: list[dict[str, Any]], epoch: int) -> list[dict[str, Any]]:
    """Each memory pass the journal holds up to ``epoch`` (``optimizer_memory``), in order."""
    return [e for e in events if e.get("event") == "optimizer_memory"
            and int(e.get("epoch") or 0) <= epoch]


class NoModel:
    """The replay's model client: every optimizer and strategy can be built with it, and
    any call raises, so the replay makes no model call."""

    def complete(self, *_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("replay_memory makes no model calls")


def _loop_at(run: Path, events: list[dict[str, Any]], at: dict[str, Any], project: Path,
             scratch: Path) -> Any:
    """The run's `Loop`, its journal replayed up to (not including) the pass ``at``, in a
    working directory of its own under ``scratch``: what the loop held when it made that
    pass."""
    from refactory_dsh.loop import Loop, LoopConfig

    saved = json.loads((run / "config.json").read_text())
    mode = str(saved.get("mode") or next((e.get("mode") for e in events
                                          if e.get("event") == "start"), "") or "")
    if not mode:
        raise SystemExit(f"{run}/config.json names no mode")
    where = scratch / f"epoch_{at.get('epoch')}"
    where.mkdir(parents=True, exist_ok=True)
    journal = where / "journal.jsonl"
    journal.write_text("".join(json.dumps(e) + "\n" for e in events[:events.index(at)]))
    config = LoopConfig.from_dict(saved)
    config.runs_root = where
    # A model proposer (autoresearch) needs a client to be built; this one refuses every
    # call, so the replay cannot make one.
    loop = Loop(mode, config, project=project, model=NoModel())
    loop._replay(journal)
    return loop


def facts_at(loop: Any, at: dict[str, Any]) -> dict[str, Any]:
    """The record's facts the loop handed the pass ``at``, by the loop's own
    `Loop._record_facts`: every round before its epoch, the gate's word as it stood, and,
    without a gate, the train leader then (the incumbent when a not-kept change was
    compared), its ancestors and every epoch's leader."""
    epoch = int(at["epoch"])
    tried, refused = loop._tried_in_round(epoch - 1)
    return loop._record_facts(epoch, tried, refused,
                              current=at.get("prev") if at.get("rejected") else at.get("curr"))


def compared_changes(loop: Any, at: dict[str, Any]) -> list[str]:
    """The changes the pass ``at`` compared (`Loop._make_memory_pass`): those between its
    two pipelines (`Loop._lineage_edits`) and its not-kept change; observations may be
    pinned to these only."""
    by_id = {c.id: c for c in loop.population}
    prev, curr = by_id.get(str(at.get("prev"))), by_id.get(str(at.get("curr")))
    if prev is None or curr is None:
        return []
    lineage = loop._lineage_edits(prev, curr)
    out = [str(step.get("id")) for step in [*(lineage.get("prev") or ()),
                                            *(lineage.get("curr") or ())]]
    if at.get("rejected"):
        out.append(str(at["rejected"].get("child")))
    return out


def _compared(at: dict[str, Any]) -> dict[str, Any]:
    """The pass ``at`` as the record lists it (`MemoryRequest.compared`)."""
    same = at.get("prev") == at.get("curr")
    curr = {"rejected": at["rejected"]} if at.get("rejected") else {}
    return {"epoch": at.get("epoch"), "prev": at.get("prev"), "curr": at.get("curr"),
            "compared": memory._what(curr, same), "items": int(at.get("pairs") or 0)}


def _items(at: dict[str, Any]) -> list[str]:
    """The items the pass ``at`` compared: its sample, less what either side lacked."""
    missing = {str(i) for i in at.get("missing") or ()}
    return [str(i) for i in at.get("sample") or () if str(i) not in missing]


def record_from_journal(run: Path, epoch: int, replies: dict[int, str], project: Path,
                        scratch: Path) -> dict[str, Any]:
    """The record the loop would hold after epoch ``epoch``'s pass: every pass up to it,
    each with the facts the loop had then and the observations of the reply given for it
    (``replies``: epoch -> reply text; none, the facts alone), as `memory.note_from` reads
    an answer, each carrying the record before it."""
    from refactory_dsh.optim.random_optimizer import reply_object

    events = _events(run)
    passes = _passes(events, epoch)
    if not any(int(p.get("epoch") or 0) == epoch for p in passes):
        raise SystemExit(f"the journal holds no memory pass for epoch {epoch}")
    record: dict[str, Any] | None = None
    for at in passes:
        loop = _loop_at(run, events, at, project, scratch)
        text = replies.get(int(at["epoch"]))
        # A pass that wrote no observations in the run, and gets no reply here, is what
        # the loop made of it: the facts, without listing it among the passes.
        failed = text is None and not at.get("updated")
        payload = reply_object(text) if text is not None else {"observations": []}
        note = memory.note_from(payload, epoch=int(at["epoch"]), facts=facts_at(loop, at),
                                earlier=record, items=_items(at),
                                compared=None if failed else _compared(at),
                                pinnable=compared_changes(loop, at))
        if note is None:
            raise SystemExit(f'the reply for epoch {at["epoch"]} has no "observations" list')
        record = note["record"]
    return record or {}


# ── the proposal request with the record, and without memory ────────────────

MEMORY_HEADING = "## Optimizer memory"


def memory_section(prompt: str) -> tuple[int, int]:
    """Where a request's "## Optimizer memory" section is: from its heading line to the
    next line of the request that starts with "## " (a fenced block's lines are the
    quoted text's, not the request's). Raises ``ValueError`` unless the heading is there
    exactly once and a heading follows it."""
    lines = prompt.split("\n")
    offsets, at = [], 0
    for line in lines:
        offsets.append(at)
        at += len(line) + 1
    fence = None
    heads = []
    for index, line in enumerate(lines):
        if fence is not None:
            if line == fence:
                fence = None
            continue
        if re.fullmatch(r"`{3,}\S*", line):
            fence = re.match(r"`+", line).group(0)
            continue
        if line.startswith("## "):
            heads.append(index)
    found = [i for i in heads if lines[i] == MEMORY_HEADING]
    if len(found) != 1:
        raise ValueError(f"the request has {len(found)} '{MEMORY_HEADING}' sections, not one")
    after = [i for i in heads if i > found[0]]
    if not after:
        raise ValueError(f"no section of the request follows '{MEMORY_HEADING}'")
    return offsets[found[0]], offsets[after[0]]


def swap_proposal(run: Path, epoch: int, record_text: str, out: Path) -> dict[str, int]:
    """Epoch ``epoch``'s last recorded proposal request three ways (`swap-proposal`), written
    under ``out``; returns each variant's characters (system prompt and request)."""
    calls = [c for c in _trace(run) if c.get("kind") == "proposal" and c.get("epoch") == epoch
             and c.get("prompt")]
    if not calls:
        raise SystemExit(f"no proposal request recorded for epoch {epoch}")
    call = calls[-1]
    prompt, system = str(call["prompt"]), str(call.get("system") or "")
    start, end = memory_section(prompt)
    passes = [p for p in _passes(_events(run), epoch)]
    if not passes:
        raise SystemExit(f"no memory pass before epoch {epoch}'s proposal")
    shown = {**dict(passes[-1].get("memory") or {}), "text": record_text}
    section = prompt[start:end]
    tail = section[len(section.rstrip("\n")):]
    variants = {"old": prompt,
                "record": prompt[:start] + "\n".join(memory.request_lines(shown)) + tail
                + prompt[end:],
                "none": prompt[:start] + prompt[end:]}
    out.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name, text in variants.items():
        (out / f"proposal.{name}.prompt.txt").write_text(text)
        (out / f"proposal.{name}.system.txt").write_text(system)
        sizes[name] = len(system) + len(text)
    return sizes


def _trace(run: Path) -> list[dict[str, Any]]:
    out = []
    for line in (run / "optimizer_trace.jsonl").read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _replies(given: list[str], epoch: int | None) -> dict[int, str]:
    """``--reply`` values: ``E=FILE`` for epoch E's pass, a bare ``FILE`` for ``--epoch``'s."""
    out = {}
    for value in given:
        name, sep, path = value.partition("=")
        if sep and name.isdigit():
            out[int(name)] = Path(path).read_text()
        elif epoch is None:
            raise SystemExit(f"--reply {value}: name its epoch (E=FILE) or give --epoch")
        else:
            out[epoch] = Path(value).read_text()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path, help="a run directory holding optimizer_trace.jsonl")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write (default: ./memory-replay-<run name>)")
    parser.add_argument("--epoch", type=int, default=None, help="only this epoch's pass")
    parser.add_argument("--compare-replies", type=Path, default=None,
                        help="a folder of epoch_<N>/part<k>.json replies to the comparison "
                             "requests, for the synthesis")
    parser.add_argument("--reply", action="append", default=[],
                        help="a model reply to a pass's note (or synthesis) request: E=FILE "
                             "for epoch E's pass, or FILE for --epoch's; without --journal, "
                             "print what the record keeps and strips of --epoch's")
    parser.add_argument("--journal", action="store_true",
                        help="rebuild epoch --epoch's record from the run's journal.jsonl "
                             "(facts) and the replies given (observations)")
    parser.add_argument("--swap-proposal", action="store_true",
                        help="with --journal: write epoch --epoch's proposal request as "
                             "recorded, with the record, and without memory")
    parser.add_argument("--project", type=Path, default=Path("."),
                        help="the project whose packages hold the run's mode (default: .)")
    args = parser.parse_args(argv)
    out = args.out or Path(f"memory-replay-{args.run.resolve().name}")
    manifest = replay(args.run, out, args.epoch, args.compare_replies)
    if args.journal or args.swap_proposal:
        if args.epoch is None or not args.journal:
            parser.error("--journal and --swap-proposal need --epoch, and --swap-proposal "
                         "needs --journal")
        import tempfile

        with tempfile.TemporaryDirectory() as scratch:
            record = record_from_journal(args.run, args.epoch,
                                         _replies(args.reply, args.epoch), args.project,
                                         Path(scratch))
        folder = out / f"epoch_{args.epoch}"
        folder.mkdir(parents=True, exist_ok=True)
        text = memory.render_record(record)
        (folder / "record.json").write_text(text)
        (folder / "record.full.json").write_text(json.dumps(record, indent=1))
        print(f"record: {folder / 'record.json'} ({len(text)} chars)")
        if args.swap_proposal:
            try:
                sizes = swap_proposal(args.run, args.epoch, text, folder)
            except ValueError as error:
                raise SystemExit(f"--swap-proposal: {error}")
            for name, chars in sizes.items():
                print(f"proposal.{name}: {folder / f'proposal.{name}.prompt.txt'} ({chars} chars)")
        return 0
    if args.reply:
        if args.epoch is None:
            parser.error("--reply needs --epoch")
        note = next((c for c in manifest["calls"] if c["kind"] == "memory" and "prompt" in c), None)
        if note is None:
            parser.error(f"no note request for epoch {args.epoch}")
        reply = _replies(args.reply, args.epoch).get(args.epoch)
        if reply is None:
            parser.error(f"no reply for epoch {args.epoch}")
        print(json.dumps(check_reply(Path(note["prompt"]).read_text(), reply, args.epoch),
                         indent=2))
        return 0
    for call in manifest["calls"]:
        print(f"epoch {call['epoch']} {call['name']}: "
              + (call.get("error") or f"{call['prompt']} ({call['chars']} chars)"))
    print(f"manifest: {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
