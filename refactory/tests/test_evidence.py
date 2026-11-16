"""One item's evidence, rendered once for every call that reads items (`optim.evidence`).

A step's prompt re-embeds the task's input and the earlier steps' replies, and the same
instructions recur in every item: shown as each step's prompt plus its reply, an item grows
with the square of its steps and a batch repeats each instruction per item (61% of an S4
analyst batch was repeated text). Rendered here, the input appears once per item, a step
says by name what it took from what is already shown, its own instructions appear once per
request, and nothing is lost: every reply and every distinct instruction text is there
exactly once.
"""

from __future__ import annotations

import collections
import re
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh.optim import analysts, evidence


def _repeated_share(text: str) -> float:
    """The share of ``text`` in content lines (40+ characters) that appear more than once;
    the renderer's own short pointers (``Sees:``, ``... same as item``) are not content."""
    lines = [" ".join(line.split()) for line in text.splitlines()
             if not line.startswith(("Sees: ", f"{evidence.REST}: same as"))]
    counts = collections.Counter(line for line in lines if len(line) >= 40)
    return sum(len(line) * (n - 1) for line, n in counts.items() if n > 1) / max(1, len(text))


@pytest.fixture
def s4_records(monkeypatch, tmp_path) -> list[dict[str, Any]]:
    """Real livemath/skill_graph passes over the fixture items (compare, then precision and
    hypotheses in a fork, then decide), each call answered with a reply of its own."""
    from test_livemath_blocks import _NodeScriptedModel, _patch_task

    from refactory_packages.livemath.skill_graph import SkillGraph

    _patch_task(monkeypatch)

    class Varied(_NodeScriptedModel):
        count = 0

        def for_node(self, node_id: str, role: str = "") -> "Varied":
            copy = Varied(self.node_answers, self.default)
            copy._current_node = node_id
            return copy

        def complete(self, prompt: str, **_: Any) -> str:
            Varied.count += 1
            node = self._current_node
            body = "\n".join(f"{node} note {Varied.count}.{i}: this option is checked in turn "
                             "against the hypotheses" for i in range(6))
            return body + ("\n<answer>A</answer>" if node == "decide" else "")

    records = []
    for run in range(3):
        result = SkillGraph().run(tmp_path / f"pass{run}", model=Varied({}))
        assert result.exit_ok, result.error
        records += [{**r, "id": f"{r['id']}#{run}"} for r in result.item_records
                    if r.get("exchanges")]
    assert len(records) >= 3
    return records


def _batch(records) -> str:
    return "\n\n".join(evidence.dedup([analysts.item_text(r) for r in records]))


def _naive(records) -> str:
    return "\n\n".join("\n".join(f"{e['prompt']}\n{e['response']}" for e in r["exchanges"])
                       for r in records)


def _no_loss(records, text: str) -> None:
    """Every reply once; every distinct instruction (a step's rest of its prompt) once."""
    blocks = dict(zip([str(r["id"]) for r in records],
                      re.split(r"(?m)^(?=### Item )", text)[1:]))
    for record in records:
        block = blocks[str(record["id"])]
        for exchange in record["exchanges"]:
            body = evidence._demoted(str(exchange["response"]).strip())  # as fenced
            assert block.count(body) >= 1 and (len(body) < 200 or text.count(body) == 1), (
                record["id"], exchange["node"])
    instructions = set()
    for record in records:
        seen = evidence._Seen()
        meta = record.get("metadata") or {}
        key = next((k for k in analysts.TRAJECTORY_INPUT_KEYS if meta.get(k)), None)
        if key:
            seen.add(meta[key], "the input")
            for choice in analysts._choice_lines(meta.get("choices")):
                seen.add(choice, "the input", 1)
        for exchange in record["exchanges"]:
            _sources, rest = seen.split(exchange["prompt"])
            if rest:
                instructions.add(rest)
            seen.add(exchange["response"], f"{exchange['node']}'s reply")
    for rest in instructions:
        assert text.count(evidence._demoted(rest)) == 1, rest[:200]


def test_s4_items_repeat_under_5_percent_and_lose_nothing(s4_records):
    text = _batch(s4_records)
    assert _repeated_share(_naive(s4_records)) > 0.3  # setup: shown whole, they repeat
    assert _repeated_share(text) < 0.05, _repeated_share(text)
    assert len(text) < len(_naive(s4_records)) / 2
    _no_loss(s4_records, text)
    # The input once per item; each step says what it took by name.
    assert "Sees: the input, compare's reply" in text
    assert re.search(r"Rest of its prompt: same as item \S+, step decide\.", text)


def _nested_record(item: int) -> dict[str, Any]:
    """A nested fork: plan, then (left, then inner (deep_a, deep_b)) beside right, then
    merge; each prompt carries the input and what came before it, with its instructions."""
    question = f"What is the value of expression number {item} under the rules?"
    replies = {node: "\n".join(f"{node} finding {item}.{i}: the reasoning holds at this step"
                               for i in range(4))
               for node in ("plan", "left", "deep_a", "deep_b", "right", "merge")}
    feeds = {"plan": [], "left": ["plan"], "deep_a": ["plan", "left"],
             "deep_b": ["plan", "left"], "right": ["plan"],
             "merge": ["left", "deep_a", "deep_b", "right"]}
    exchanges = []
    for node, before in feeds.items():
        prompt = "\n".join([f"## Question\n{question}",
                            *(f"## {b} notes\n{replies[b]}" for b in before),
                            f"## Your task\nYou are the {node} step: do the {node} work fully."])
        exchanges.append({"node": node, "prompt": prompt, "response": replies[node]})
    return {"id": f"n{item}", "passed": False, "score": 0.0, "label": "7", "prediction": "5",
            "fail_reason": "wrong value", "metadata": {"question": question},
            "exchanges": exchanges}


def test_a_nested_fork_s_items_repeat_under_5_percent_and_lose_nothing():
    records = [_nested_record(i) for i in range(6)]
    text = _batch(records)
    assert _repeated_share(_naive(records)) > 0.3
    assert _repeated_share(text) < 0.05, _repeated_share(text)
    _no_loss(records, text)
    assert "Sees: the input, plan's reply, left's reply, deep_a's reply, deep_b's reply, " \
        "right's reply" not in text  # merge does not see plan: said as what it took only
    assert "Sees: left's reply, deep_a's reply, deep_b's reply, right's reply" in text \
        or "Sees: the input, left's reply, deep_a's reply, deep_b's reply, right's reply" in text


def test_the_proposer_s_items_are_the_same_renderer_without_step_prompts():
    from refactory_dsh.optim import base

    record = _nested_record(1)
    block = base._trajectory(record)
    assert block.splitlines()[0].startswith("### Item n1 — answered wrong")
    assert "Rest of its prompt" not in block and "Sees:" not in block
    for exchange in record["exchanges"]:
        assert exchange["response"] in block
    assert block.count("What is the value of expression number 1") == 1


def test_memory_s_comparison_says_a_repeated_instruction_once():
    from refactory_dsh.optim import memory

    prev, curr = _nested_record(1), {**_nested_record(1), "passed": True, "score": 1.0}
    entries = [{"id": "n1", "category": "improved", "prev": prev, "curr": curr}]
    texts = {("n1", side): analysts.item_text(e) for side, e in (("prev", prev), ("curr", curr))}
    deduped = memory._deduped(entries, texts, same=True)
    joined = deduped[("n1", "prev")] + deduped[("n1", "curr")]
    assert "Rest of its prompt: same as item n1" in deduped[("n1", "curr")]
    for exchange in prev["exchanges"]:
        rest = re.search(r"## Your task\n(.*)", exchange["prompt"]).group(1)
        assert joined.count(rest) == 1


def test_a_step_s_own_short_line_is_never_taken_for_the_input():
    """An input line as short as "A", "---" or "1." is not matched in a step's prompt (its
    own instruction line would vanish into "[the input]"); a choice line is, however
    short."""
    record = {"id": "s1", "passed": False, "score": 0.0,
              "metadata": {"question": "Which holds?\nA\n---\n1.\nThe long question line here",
                           "choices": ["3"]},
              "exchanges": [{"node": "solve",
                             "prompt": "Which holds?\nThe long question line here\nA. 3\n"
                                       "Answer in this form:\nA\n---\n1.\nOne line each.",
                             "response": "fine"}]}
    text = analysts.item_text(record)
    rest = text.split("Rest of its prompt:", 1)[1]
    for line in ("A", "---", "1."):
        assert f"\n{line}\n" in rest, line
    assert "A. 3" not in rest and "The long question line here" not in rest


def test_a_long_line_said_as_where_it_was_shown_is_marked_never_dropped():
    """Two items whose step instructions share a long line but differ in another: the
    second keeps its own line and marks the shared one as where it was shown -- nothing
    vanishes without a mark."""
    shared = "Compare every option against each hypothesis before choosing any of them."

    def item(i: int, own: str) -> str:
        return (f"### Item i{i} — answered wrong (score 0.0)\n#### Step `solve` (1 of 1)\n"
                f"{evidence.REST}:\n```\n{shared}\n{own}\n```\nReply:\n```\nr{i}\n```")

    first, second = evidence.dedup([item(1, "Own line of the first item's step, long enough."),
                                    item(2, "A different own line for the second item here.")])
    assert shared in first and shared not in second
    assert "[a line of item i1, step solve's instructions, as shown there]" in second
    assert "A different own line for the second item here." in second
