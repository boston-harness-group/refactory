"""The optimizer's requests do not repeat themselves, and stay in order.

Every call that reads items -- the proposal, the analysts, their merges, memory and its
comparison, the step review -- is built here from fixtures (the S4 skill_graph and S1
solver start states, a nested fork, a synthetic 10-round history) and held to:

- repeated long lines under 5% of its characters;
- no item's reply (the same text) shown twice in one request;
- the pipeline described once;
- no heading from an item's text outside a fence (it would read as a section);
- the proposal's sections in the agreed order.

Size is reported, never capped (`refactory_dsh.tools.request_sizes`): a request may grow
when it says more; it must not say the same thing twice.
"""

from __future__ import annotations

import collections
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import registry, strategies
from refactory_dsh.loop import LoopConfig
from refactory_dsh.optim import Candidate, analysts, base, evidence, feedback, memory
from refactory_dsh.tools.request_sizes import measure, own_headings, repeated_share, table

#: What the request tests fail on (the one table of them): repetition and disorder, never
#: size.
REPEATED_SHARE = 0.05   # repeated long lines: a reply or instruction said twice is waste
LONG_REPLY = 200        # a reply this long shown twice in one request is a repeat


def _reply(node: str, item: str, n: int = 6) -> str:
    return "\n".join([f"## Option analysis by {node}",
                      *(f"{node} note {item}.{i}: this option is checked in turn against the "
                        f"hypotheses of item {item}" for i in range(n)),
                      "# Summary", f"{node} sums up item {item} with care and some detail here"])


_CALLS: list[int] = []


def _model_class():
    from test_livemath_blocks import _NodeScriptedModel

    class Varied(_NodeScriptedModel):
        def for_node(self, node_id: str, role: str = "") -> "Varied":
            copy = Varied(self.node_answers, self.default)
            copy._current_node = node_id
            return copy

        def complete(self, prompt: str, **_: Any) -> str:
            _CALLS.append(1)  # every call its own reply, across rollouts
            body = _reply(self._current_node, str(len(_CALLS)))
            return body + ("\n<answer>A</answer>" if self._current_node in ("decide", "solve")
                           else "")
    return Varied


def _rollout(mode_name: str, tmp: Path, run: int) -> list[dict[str, Any]]:
    _package, mode = registry.resolve(Path("."), mode_name)
    result = mode().run(tmp / f"{mode_name.replace('/', '_')}{run}", model=_model_class()({}))
    assert result.exit_ok, result.error
    return [r for r in result.item_records if r.get("exchanges")]


def _records(mode_name: str, tmp: Path, runs: int = 3) -> list[dict[str, Any]]:
    """Several rollouts of the fixture items, each item an item of its own."""
    return [{**r, "id": f"{r['id']}#{run}"} for run in range(runs)
            for r in _rollout(mode_name, tmp, run)]


@pytest.fixture
def requests(tmp_path, monkeypatch) -> dict[str, list[tuple[str, list[dict[str, Any]]]]]:
    """Every call kind's requests, each with the records it shows."""
    from test_evidence import _nested_record
    from test_livemath_blocks import _patch_task

    _patch_task(monkeypatch)
    built: dict[str, list[tuple[str, list[dict[str, Any]]]]] = {}
    shapes = {"S4": ("livemath/skill_graph", _records("livemath/skill_graph", tmp_path)),
              "S1": ("livemath/solver", _records("livemath/solver", tmp_path)),
              "nested": ("livemath/skill_graph", [_nested_record(i) for i in range(6)])}
    for name, (mode_name, records) in shapes.items():
        loop = strategies.create("default", mode_name, LoopConfig(
            epochs=1, batch_size=1, optimizer="random", seed=1,
            runs_root=tmp_path / name / "run", edit_preset="broad"), project=Path("."))
        seed = Candidate.from_state(loop.mode.state_dict())
        seed.evaluated, seed.score = True, 0.5
        loop.population.append(seed)
        loop._train_item_records[seed.id] = records
        ctx = replace(loop._context(), item_records=records,
                      analysis={"findings": [{"source": "failure", "finding": "F",
                                              "items": [records[0]["id"]]}],
                                "batches": [], "group_size": 8}, **_history(records))
        built.setdefault("proposal", []).append((ctx.render(), records))
        pipeline = analysts.pipeline_text(ctx)
        # One rollout of each item per batch (a batch's items are distinct items).
        distinct = list({str(r["id"]).split("#")[0]: r for r in records}.values())
        fails = [r for r in distinct if not r.get("passed")][:8] or distinct[:8]
        passes = [r for r in distinct if r.get("passed")][:8] or distinct[:8]
        built.setdefault("analyst_failure", []).append(
            (analysts.analyst_batch(ctx, "failure", 1, 1, fails, pipeline)[0], fails))
        built.setdefault("analyst_success", []).append(
            (analysts.analyst_batch(ctx, "success", 1, 1, passes, pipeline)[0], passes))
        finding = {"source": "failure", "finding": "F", "items": [records[0]["id"]]}
        built.setdefault("analyst_merge", []).append(
            (analysts.merge_prompt(ctx, "failure", [[finding], [finding]], 1, pipeline), []))
        if name == "nested":
            before = [_nested_record(i) for i in range(6)]
            after = [{**_nested_record(i + 100), "id": f"n{i}"} for i in range(6)]
        else:  # two rollouts of the same items
            before = _rollout(mode_name, tmp_path / name, 10)
            after = _rollout(mode_name, tmp_path / name, 11)
        entries, _missing = memory.pairs(before, after, [r["id"] for r in after[:6]])
        compared = [*before[:6], *after[:6]]
        side = {"id": seed.id, "pipeline": pipeline, "rollout": "train"}
        request = memory.MemoryRequest(epoch=3, mode_name=mode_name, prev=side, curr=side,
                                       changes=[], previous_note="", entries=entries)
        built.setdefault("memory", []).append((request()[0], compared))
        built.setdefault("memory_compare", []).append(
            (request.compare_prompt(entries, 1, 1)[0], compared))
        step = next(e["node"] for e in records[0]["exchanges"])
        samples = [{"item": r["id"], "input": r["exchanges"][0]["prompt"],
                    "output": r["exchanges"][0]["response"]} for r in records[:3]]
        built.setdefault("step_review", []).append(
            (feedback.review_prompt({"step": step, "candidate": seed.id, "kind": "insert",
                                     "samples": samples}, mode_name), records[:3]))
    return built


def _history(records: list[dict[str, Any]], rounds: int = 10) -> dict[str, Any]:
    """A synthetic ``rounds`` rounds (ten by default): two attempts a round, each with an edit and what it built,
    the last round's children with their changed items, the reasoning of every round."""
    rows, reasoning = [], []
    for epoch in range(1, rounds + 1):
        for k in range(2):
            cid = f"c{epoch}{k}"
            edit = {"op": "set_knob", "name": "decide_task",
                    "patch": [{"kind": "append", "content": f"Round {epoch}.{k}: check twice."}]}
            rows.append({"id": cid, "parent": f"c{epoch - 1}0" if epoch > 1 else "seed",
                         "score": 0.4 + epoch / 100, "structural": False, "exit_ok": True,
                         "edits": [edit], "own": 1, "epoch": epoch,
                         "built": [f"~ decide agent  told decide_task \"Round {epoch}.{k}\""],
                         "gate": "rejected"})
            reasoning.append({"id": cid, "epoch": epoch, "edit": [edit],
                              "flow_change": f"Round {epoch}.{k}: decide checks twice now, "
                                             "reading each option against the notes again",
                              "wiring": [f"~ decide agent  told decide_task \"Round {epoch}.{k}"
                                         f" check twice\"  -> answer.txt (line {i})"
                                         for i in range(3)],
                              "rationale": f"Round {epoch}.{k}: " + "because " * 60,
                              "prediction": "more items pass", "gate": "rejected",
                              "score": 0.4, "fixed": 1, "broken": 2, "items": len(records)})
    pairs = [{"child": f"c{rounds}0", "parent": f"c{rounds - 1}0", "edit": rows[-2]["edits"],
              "gate": "rejected", "flow_change": reasoning[-2]["flow_change"],
              "wiring": reasoning[-2]["wiring"],
              "child_records": records, "parent_records": records, "steps": []}]
    return {"history": rows, "reasoning": reasoning, "edit_pairs": pairs}


KINDS = ["proposal", "analyst_failure", "analyst_success", "analyst_merge", "memory",
         "memory_compare", "step_review"]


@pytest.mark.parametrize("kind", KINDS)
def test_no_request_repeats_itself_or_lets_an_item_pose_as_a_section(kind, requests):
    for text, records in requests[kind]:
        lines = [" ".join(line.split()) for line in text.splitlines()]
        top = [(n, line[:110]) for line, n in collections.Counter(
            line for line in lines if len(line) >= 60).most_common(3) if n > 1]
        assert repeated_share(text) < REPEATED_SHARE, (kind, f"{repeated_share(text):.1%}", top)
        headings = own_headings(text)
        planted = {line.strip() for r in records for e in r.get("exchanges") or ()
                   for line in str(e.get("response") or "").splitlines()
                   if line.startswith("#")}
        assert not planted & set(headings), (kind, sorted(planted & set(headings))[:3])
        for record in records:
            for exchange in record.get("exchanges") or ():
                body = evidence.escape_headings(str(exchange.get("response") or "").strip())
                if len(body) >= LONG_REPLY:
                    assert text.count(body) <= 1, (kind, record["id"], exchange.get("node"))


def repeated_blocks(text: str, lines: int = 3, chars: int = 120) -> list[str]:
    """Each run of ``lines`` consecutive non-blank lines (``chars``+ characters together)
    that appears more than once: a block said twice, however small its share of the
    request (the line share above can miss one)."""
    from refactory_dsh.tools.request_sizes import POINTERS

    # Content lines only: the renderer's own pointers and short labels are structure.
    rows = [" ".join(line.split()) for line in text.splitlines()
            if len(line.strip()) >= 40 and not line.startswith(("```", *POINTERS))]
    seen: dict[str, int] = {}
    for i in range(len(rows) - lines + 1):
        block = "\n".join(rows[i:i + lines])
        if len(block) >= chars:
            seen[block] = seen.get(block, 0) + 1
    return [block for block, n in seen.items() if n > 1]


#: The section holding each knob's exact text: knobs are separate edit targets a patch
#: quotes exactly, so a paragraph two knobs share is shown in each (it is two texts).
KNOB_TEXTS = "## Each step's settings"


def _without_knob_texts(text: str) -> str:
    """``text`` without the settings section (fenced knob texts inside it included)."""
    out, skipping, fence = [], False, None
    for line in text.splitlines():
        found = re.match(r"^(`{3,})", line)
        if found and (fence is None or found.group(1) == fence):
            fence = None if fence else found.group(1)
        elif fence is None and line.startswith("## "):
            skipping = line.startswith(KNOB_TEXTS)
        if not skipping:
            out.append(line)
    return "\n".join(out)


@pytest.mark.parametrize("kind", KINDS)
def test_no_block_of_lines_is_said_twice(kind, requests):
    for text, _records in requests[kind]:
        repeats = repeated_blocks(_without_knob_texts(text))
        assert repeats == [], (kind, repeats[0][:300])


def test_the_proposal_describes_the_pipeline_once_and_keeps_the_order(requests):
    from test_request_order import ORDER, _sections

    for text, _records in requests["proposal"]:
        assert text.count(base.PIPELINE_VIEW_LEAD) == 1 and "\nNodes:\n" not in text
        heads = _sections(text)
        at = [next((i for i, h in enumerate(heads) if h.startswith(h0)), None) for h0 in ORDER]
        present = [i for i in at if i is not None]
        assert present == sorted(present), [h for h in ORDER]


def test_the_sizes_are_reported(requests, capsys):
    rows = {}
    for kind in KINDS:
        sizes = [measure(text) for text, _ in requests[kind]]
        rows[kind] = {"calls": len(sizes), "chars_max": max(m["chars"] for m in sizes),
                      "chars_mean": round(sum(m["chars"] for m in sizes) / len(sizes)),
                      "repeated_mean": sum(m["repeated"] for m in sizes) / len(sizes),
                      "repeated_max": max(m["repeated"] for m in sizes),
                      "largest_sections": max(sizes, key=lambda m: m["chars"])["largest"]}
    print(table(rows))  # reported, not asserted
    assert set(rows) == set(KINDS)


def test_the_history_grows_by_a_line_per_attempt_and_its_size_is_reported(tmp_path,
                                                                          monkeypatch, capsys):
    """Ten synthetic rounds: the proposal's history sections, by round. Older attempts are
    one line each (their full reasoning summarised by memory), so each round adds about the
    same; only the latest round's reasoning is shown in full."""
    from test_livemath_blocks import _patch_task
    from test_request_order import _sections

    _patch_task(monkeypatch)
    records = _records("livemath/skill_graph", tmp_path, runs=1)
    loop = strategies.create("default", "livemath/skill_graph", LoopConfig(
        epochs=1, batch_size=1, optimizer="random", seed=1, runs_root=tmp_path / "h" / "run",
        edit_preset="broad"), project=Path("."))
    seed = Candidate.from_state(loop.mode.state_dict())
    seed.evaluated, seed.score = True, 0.5
    loop.population.append(seed)
    loop._train_item_records[seed.id] = records
    history = ("Recent results", "What this run has already tried", "Your earlier reasoning",
               "What the last edit changed")
    sizes, text = [], ""
    for rounds in range(1, 11):
        text = replace(loop._context(), item_records=records,
                       **_history(records, rounds)).render()
        parts = text.split("\n## ")
        sizes.append(sum(len(part) for part in parts if part.startswith(history)))
        assert sum(1 for part in parts if part.startswith(history)) >= 3, "each one measured"
    growth = [b - a for a, b in zip(sizes, sizes[1:])]
    print("| round | " + " | ".join(map(str, range(1, 11))) + " |")
    print("| history chars | " + " | ".join(f"{n:,}" for n in sizes) + " |")  # reported
    assert max(growth[3:]) <= 2 * max(1, min(growth[3:])), growth  # linear, not compounding
    assert "Round 10.0: because" in text and "Round 3.0: because" not in text
    assert _sections(text)  # the request still parses into its sections


#: A claim about where a new step's output goes.
WIRING_CLAIM = re.compile(r"(?i)\b(?:its|their|a new step's|the new step's) outputs? "
                          r"(?:goes|go|reaches|reach|is put|are put|is placed) (?:in|to|the)|"
                          r"next model step's prompt|goes into the prompt")
def _system_written(text: str) -> str:
    """``text`` without its fenced blocks (item outputs, knob texts). The fixture's
    model-written fields -- findings, flow changes, predictions, rationales, the memory
    note, replies -- are sentinels that make no claim (``_history``, the scripted model),
    so whatever the scan finds outside fences is the system's own text."""
    kept, fence = [], None
    for line in text.splitlines():
        found = re.match(r"^(`{3,})", line)
        if found and (fence is None or found.group(1) == fence):
            fence = None if fence else found.group(1)
            continue
        if fence is None:
            kept.append(line)
    return "\n".join(kept)


@pytest.mark.parametrize("kind", KINDS)
def test_every_wiring_claim_the_system_writes_is_the_one_operators_say(kind, requests):
    """Where a new step's output goes is said by `core_ops.REACHES` alone: any such claim
    in system-written text is that phrase (a hand-written restatement would drift)."""
    from refactory_dsh.nn import core_ops

    reaches = re.compile(re.escape(core_ops.REACHES).replace(re.escape("{where}"), r"[^-]*?"))
    for text, _records in requests[kind]:
        system = reaches.sub("", _system_written(text))
        claims = [m.group(0) for m in WIRING_CLAIM.finditer(system)]
        assert claims == [], (kind, claims[:3])


def test_the_size_report_says_what_it_could_not_read(tmp_path, capsys):
    from refactory_dsh.tools import request_sizes

    trace = tmp_path / "optimizer_trace.jsonl"
    trace.write_text('{"kind": "proposal", "prompt": "hello"}\nnot json\n')
    assert request_sizes.main([str(trace), str(tmp_path / "missing")]) == 0
    out = capsys.readouterr().out
    assert "| proposal | 1 |" in out
    assert "2 line(s) or file(s) could not be read" in out and f"{trace}:2" in out


#: Wording that says items are shown by how much fits: items are picked by a rule
#: (`base._shown_items`) and shown whole, so no request says size chose them.
SIZE_CHOSEN = re.compile(r"(?i)as many (?:\w+ ){0,3}as (?:fit|fits|the budget|room)\b|"
                         r"\bas (?:fit|fits)\b|while they fit|within (?:the|its|their) budget")


def _selected_requests() -> list[str]:
    """Proposal requests whose items are not all shown: picked by analyst findings, by
    reader findings, and with none."""
    from test_request_order import FINDINGS, _ctx, _many

    reader = [{"group": 1, "items": [], "failure_patterns": [
        {"pattern": "A", "step": "draft", "items": ["f2"], "evidence": ["q"]}],
        "success_patterns": []}]
    return [_ctx(_many(), findings=FINDINGS).render(),
            _ctx(_many(), reader_findings=reader).render(), _ctx(_many()).render()]


@pytest.mark.parametrize("kind", KINDS)
def test_no_request_says_items_were_chosen_by_how_much_fits(kind, requests):
    texts = [text for text, _records in requests[kind]]
    if kind == "proposal":
        texts += _selected_requests()
        assert all("Some items in full" in t for t in texts[-3:]), "setup: items picked"
    for text in texts:
        said = [m.group(0) for m in SIZE_CHOSEN.finditer(_system_written(text))]
        assert said == [], (kind, said[:3])


def test_the_operators_lead_is_its_constant_and_makes_no_wiring_claim(requests):
    """The lead is pinned (a reworded restatement would change it and fail here); the
    claim scan above is the net for everything else."""
    assert not WIRING_CLAIM.search(base.STRUCTURAL_LEAD)
    assert not re.search(r"(?i)\bsees\b|reach", base.STRUCTURAL_LEAD.split("line below says")[1])
    leads = [text.split("### The operations you may use\n", 1)[1].split("\n### ", 1)[0]
             for text, _records in requests["proposal"]]
    assert leads and all(lead.count(base.STRUCTURAL_LEAD) == 1 for lead in leads), leads[:1]


def _view_block(text: str) -> str:
    """The pipeline view a request shows: its lead and the fenced lines after it."""
    start = text.index(base.PIPELINE_VIEW_LEAD)
    fence = text.index("```text\n", start)
    return text[start:text.index("\n```", fence + 8)]


@pytest.mark.parametrize("kind", ["analyst_failure", "analyst_success", "analyst_merge",
                                  "memory", "memory_compare"])
def test_every_call_that_shows_the_pipeline_shows_the_proposer_s_view_once(kind, requests):
    """Analysts, their merges and memory read the pipeline in the proposer's words: the
    same view, drawn once, and no node table or node JSON."""
    for (text, _records), (proposal, _r) in zip(requests[kind], requests["proposal"]):
        assert text.count(base.PIPELINE_VIEW_LEAD) == 1, kind
        if kind.startswith("analyst"):
            assert _view_block(text) == _view_block(proposal), kind
        assert "Nodes:" not in text and '"field_bindings"' not in text, kind


def _settings_once(text: str) -> str:
    """``text`` with its settings section said once: each distinct line of it kept, so
    two knobs' exact texts sharing a sentence (each an edit target in its own right) are
    one source, and anything else saying it again is a repeat."""
    out, inside, seen, fence = [], False, set(), None
    for line in text.splitlines():
        found = re.match(r"^(`{3,})", line)
        if found and (fence is None or found.group(1) == fence):
            fence = None if fence else found.group(1)
        elif fence is None and line.startswith("## "):
            inside = line.startswith(KNOB_TEXTS)
        key = " ".join(line.split())
        if inside and key:
            if key in seen:
                continue
            seen.add(key)
        out.append(line)
    return "\n".join(out)


@pytest.mark.parametrize("kind", KINDS)
def test_no_long_sentence_is_said_twice_even_inside_fences(kind, requests):
    """The guard below reads outside fences; this one reads inside them too, which is where
    a step's instructions repeat: once in "Each step's settings" and again inside an item's
    prompt (indented under the model client's wrapper). Every call kind that shows items
    renders them with its settings said once (`base.settled_for`), so none does."""
    from refactory_dsh.tools.request_sizes import repeated_sentences

    for text, _records in requests[kind]:
        repeats = repeated_sentences(_settings_once(text), fenced=True)
        assert repeats == [], (kind, repeats[:3])


@pytest.mark.parametrize("kind", KINDS)
def test_no_sentence_of_100_characters_is_said_twice(kind, requests):
    """Any sentence of `request_sizes.REPEAT_SENTENCE`+ characters said twice fails, however
    small its share of the request: an operation described in a table row and again in a
    list, a rule given in two sections. The steps' own knob texts (exact edit targets in
    the settings) and fenced text (items, knob texts) are not the request's sentences."""
    from refactory_dsh.tools.request_sizes import repeated_sentences

    for text, _records in requests[kind]:
        repeats = repeated_sentences(_without_knob_texts(text))
        assert repeats == [], (kind, repeats[:3])
