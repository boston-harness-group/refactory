"""Re-render a run's recorded memory requests with the current record prompt, offline.

    uv run python refactory/scripts/replay_memory.py RUN_DIR [--out DIR] [--epoch N]
    uv run python refactory/scripts/replay_memory.py RUN_DIR --epoch N --reply REPLY.json

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
    return {"kept": note["record"]["observations"], "stripped": note["stripped"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path, help="a run directory holding optimizer_trace.jsonl")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write (default: ./memory-replay-<run name>)")
    parser.add_argument("--epoch", type=int, default=None, help="only this epoch's pass")
    parser.add_argument("--compare-replies", type=Path, default=None,
                        help="a folder of epoch_<N>/part<k>.json replies to the comparison "
                             "requests, for the synthesis")
    parser.add_argument("--reply", type=Path, default=None,
                        help="a model reply to the epoch's note (or synthesis) request: "
                             "print what the record keeps and strips")
    args = parser.parse_args(argv)
    out = args.out or Path(f"memory-replay-{args.run.resolve().name}")
    manifest = replay(args.run, out, args.epoch, args.compare_replies)
    if args.reply is not None:
        if args.epoch is None:
            parser.error("--reply needs --epoch")
        note = next((c for c in manifest["calls"] if c["kind"] == "memory" and "prompt" in c), None)
        if note is None:
            parser.error(f"no note request for epoch {args.epoch}")
        print(json.dumps(check_reply(Path(note["prompt"]).read_text(), args.reply.read_text(),
                                     args.epoch), indent=2))
        return 0
    for call in manifest["calls"]:
        print(f"epoch {call['epoch']} {call['name']}: "
              + (call.get("error") or f"{call['prompt']} ({call['chars']} chars)"))
    print(f"manifest: {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
