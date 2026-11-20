"""One item's evidence, rendered once for every call that reads items.

The proposer, the analysts and memory's comparison all read train items. A step's prompt
re-embeds what came before it -- the task's input, earlier steps' replies -- and the same
instruction text recurs in every item, so an item shown as each step's prompt plus its
reply grows with the square of its steps, and a batch repeats each instruction once per
item. Here an item is shown with:

- the task's input once;
- for each step, what its prompt took from what is already shown (the input, an earlier
  step's reply) said by name -- ``sees: the input, compare's reply`` -- and only the rest
  of its prompt (its own instructions), then its full reply, each fenced with its heading
  lines escaped (`escape_headings`) so nothing in it reads as a section of the request;
- for a step run again (a loop's next turn), what its prompt repeats from that step's
  previous run -- the conversation so far as its opening, its last reply quoted whole --
  said by name (``sees: solve's prompt (step 3)``), so only what is new since that run is
  shown: an item whose step reruns with its whole history would otherwise grow with the
  square of its runs (an item of 30 turns was 1.2M characters);
- and, across the items of one request, an instruction text shown in an earlier item said
  as "same as item X" (`dedup`), so each distinct instruction appears once.

Nothing is lost: every reply and every distinct instruction text appears once in the
request (``tests/test_evidence.py`` checks it). Generic: it compares lines of text, with no
knowledge of the task or the pipeline.
"""

from __future__ import annotations

import re
from typing import Any, Collection, Iterable, Mapping, Sequence

#: A reply's line shorter than this (after stripping) is never matched as repeated text:
#: a bare heading or label is not worth a reference. The task's input matches whole.
MIN_LINE = 12
#: What a step's prompt repeats from that step's previous run (`_Seen.split`: an opening,
#: or a text quoted whole) is said by name only from this many characters: a shorter one
#: is left to the line matching, as a marker would save little and say less.
REPEAT_MIN = 200
#: What heads the part of a step's prompt that is its own, which `dedup` looks for.
REST = "Rest of its prompt"
#: What heads the part of a step's prompt that its settings do not hold (`render_item`
#: with ``settled``): a reloop's reason, say.
ALSO = "Also in its prompt (not in its settings)"


def settled_lines(texts: Iterable[Any]) -> frozenset[str]:
    """The lines of ``texts`` (each step's settings, shown elsewhere in the request),
    normalised as `render_item` compares a step's prompt with them."""
    return frozenset(_norm(line) for text in texts if isinstance(text, str)
                     for line in text.splitlines() if line.strip())
#: How an item rendered here reads, said once for every call that shows items.
TRACE_FORM = ("its input once, then each step in the order the steps ran -- what its prompt "
              "took from the input, an earlier step's reply or, for a step run again, what "
              "its previous run's prompt or reply already showed, said by name (`Sees:`), any "
              "part of its prompt the request does not show elsewhere (`" + "Also in its "
              "prompt (not in its settings)" + "`: a reason a judge sent it back with, say; "
              "its own instructions are its settings, under the pipeline's \"Each step's "
              "settings\"), and its reply -- the expected answer, the prediction, the score "
              "and the grader's feedback")
_WHOLE = 10 ** 9


def _fmt_score(value: Any) -> str:
    from .base import fmt_score

    return fmt_score(value)


def _norm(line: str) -> str:
    return " ".join(line.split())


def _fence(text: str) -> str:
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


#: A line Markdown reads as a heading (`#` to `######` then a space or the line's end,
#: indented at most three spaces).
_HEADING = re.compile(r"(?m)^([ \t]{0,3})(#{1,6})(?=[ \t]|$)")


def escape_headings(text: str) -> str:
    """``text`` with each heading line's ``#`` run escaped (``\\##``), so a step's own
    "## Answer", quoted as evidence, can never read as one of the request's headings at
    any level. The words are unchanged; only the marker is escaped, as Markdown escapes
    it. Evidence only: a text the optimizer may copy back is quoted by `exact_block`."""
    return _HEADING.sub(lambda m: f"{m.group(1)}\\{m.group(2)}", text)


def exact_block(text: Any) -> list[str]:
    """``text`` byte-exact in a code fence longer than any run of backticks in it, so no
    line of it can close the fence and nothing in it is read as Markdown (a heading, a
    setext underline, a fence): for a text the optimizer may copy into a prompt -- a knob
    text, a finding's proposed change, the memory note -- which must not gain escapes."""
    body = "" if text is None else str(text)
    fence = _fence(body)
    return [fence, body, fence]


def quoted(text: Any) -> str:
    """A model's text embedded in one of the request's own lines: a one-line text as it
    is (mid-line, nothing in it reads as a heading or a fence); a text of several lines
    on lines of its own, byte-exact in a fence (`exact_block`)."""
    body = "" if text is None else str(text)
    if "\n" not in body:
        return body
    return "\n" + "\n".join(exact_block(body)) + "\n"


def fenced(text: str) -> list[str]:
    """``text`` in a code fence longer than any run of backticks in it, its heading lines
    escaped (`escape_headings`): a quoted step output, trace or prompt, whose own headings
    cannot read as the request's at any level."""
    body = escape_headings(text)
    fence = _fence(body)
    return [fence, body, fence]


def clip(text: Any, limit: int, *, item: Any = None, step: str = "",
         omissions: list[dict[str, Any]] | None = None,
         mark: str = "[… {chars:,} chars omitted]") -> str:
    """``text`` whole when it fits ``limit``, else its head and tail around ``mark``;
    each cut added to ``omissions`` (item, step, chars)."""
    text = ("" if text is None else str(text)).strip()
    if len(text) <= limit:
        return text
    keep = max(0, limit // 2)
    chars = len(text) - 2 * keep
    if omissions is not None:
        omissions.append({"item": item, "step": step, "chars": chars})
    return f"{text[:keep]}\n{mark.format(chars=chars)}\n{text[len(text) - keep:]}"


class _Seen:
    """The lines already shown for one item, each with where it was shown, and each step's
    latest run as shown (its prompt and its reply), which that step's next run may repeat
    as its opening."""

    def __init__(self) -> None:
        self.where: dict[str, str] = {}
        self.runs: dict[str, list[tuple[str, list[str]]]] = {}

    def add(self, text: Any, label: str, shortest: int = MIN_LINE) -> None:
        for line in str(text or "").splitlines():
            key = _norm(line)
            if key and len(key) >= shortest:
                self.where.setdefault(key, label)

    def ran(self, node: str, texts: Iterable[tuple[str, Any]]) -> None:
        """Step ``node``'s run as shown: its texts (``(label, text)``: its prompt, its
        reply), which the step's next run may open with or quote (`split` with
        ``node``)."""
        self.runs[node] = [(label, str(text).splitlines()) for label, text in texts if text]

    def _repeat(self, lines: Sequence[str], node: str | None) -> tuple[str, int] | None:
        """The text of step ``node``'s previous run that ``lines`` open with (its prompt
        or its reply: the longer opening) and how many lines it covers; ``None`` when no
        opening of at least `REPEAT_MIN` characters is repeated. Lines are compared as
        `split` compares them, whitespace aside."""
        best: tuple[str, int] | None = None
        chars = 0
        for label, earlier in self.runs.get(node, []) if node is not None else []:
            count = 0
            for line, before in zip(lines, earlier):
                if line != before and _norm(line) != _norm(before):
                    break
                count += 1
            # Blank lines closing the opening are left to the rest.
            while count and not lines[count - 1].strip():
                count -= 1
            size = sum(map(len, lines[:count])) + count
            if size >= REPEAT_MIN and size > chars:
                best, chars = (label, count), size
        return best

    def _quoted(self, lines: Sequence[str], node: str | None) -> dict[int, tuple[str, int, str]]:
        """Where ``lines`` quote a text of step ``node``'s previous run whole (its reply,
        say, under a turn label: ``- Assistant: <its first line>``, indented or not), of
        at least `REPEAT_MIN` characters: each start line -> (label, line count, the
        label's text before the quote on that line)."""
        texts = []
        for label, earlier in self.runs.get(node, []) if node is not None else []:
            first, last = 0, len(earlier)
            while first < last and not earlier[first].strip():
                first += 1
            while last > first and not earlier[last - 1].strip():
                last -= 1
            body = earlier[first:last]
            if body and sum(map(len, body)) + len(body) >= REPEAT_MIN:
                texts.append((label, body, _norm(body[0])))
        found: dict[int, tuple[str, int, str]] = {}
        if not texts:
            return found
        normed = [_norm(line) for line in lines]
        start = 0
        while start < len(lines):
            for label, body, head in texts:
                if (start + len(body) > len(lines) or not head
                        or not normed[start].endswith(head)):
                    continue
                if all(normed[start + offset] == _norm(want)
                       for offset, want in enumerate(body[1:], 1)):
                    lead = normed[start][:len(normed[start]) - len(head)].strip()
                    found[start] = (label, len(body), lead)
                    start += len(body) - 1
                    break
            start += 1
        return found

    def split(self, prompt: Any, node: str | None = None) -> tuple[list[str], str]:
        """What ``prompt`` took from what is shown (each source once, in order) and the
        rest of it, its runs of shown lines said by name. With ``node``, what the prompt
        repeats from that step's previous run is said by that run's name: the opening it
        repeats (the conversation so far, `_repeat`), and any text of that run it quotes
        whole (its last reply under a turn label, `_quoted`), so a step run again shows
        only what is new since its previous run."""
        sources: list[str] = []
        rest: list[str] = []
        last = None
        lines = str(prompt or "").splitlines()
        repeat = self._repeat(lines, node)
        if repeat is not None:
            label, count = repeat
            sources.append(label)
            rest.append(f"[{label}]")
            last = label
            lines = lines[count:]
        quoted = self._quoted(lines, node)
        index = 0
        while index < len(lines):
            if index in quoted:
                label, count, lead = quoted[index]
                if label not in sources:
                    sources.append(label)
                rest.append(f"{lead} [{label}]" if lead else f"[{label}]")
                last = None if lead else label
                index += count
                continue
            line = lines[index]
            index += 1
            label = self.where.get(_norm(line))
            if label is None:
                rest.append(line)
                last = None
                continue
            if label not in sources:
                sources.append(label)
            if label != last:
                rest.append(f"[{label}]")
                last = label
        text = "\n".join(rest).strip()
        # A prompt that is only what was shown has no instructions of its own.
        only_refs = all(re.fullmatch(r"\[[^\]]+\]", line.strip()) or not line.strip()
                        for line in text.splitlines())
        return sources, "" if only_refs else text


def prompt_lines(seen: "_Seen", prompt: Any, settled: Collection[str] | None,
                 cut: Any, node: str | None = None) -> list[str]:
    """What a step's prompt took from what is shown, by name (``Sees:``), then the rest of
    it: whole (`REST`), or, with ``settled``, only what the request does not show
    elsewhere (`ALSO`); ``cut`` clips that text. With ``node`` (the step), an opening
    repeated from that step's previous run is said by that run's name (`_Seen.split`)."""
    sources, rest = seen.split(prompt, node)
    lines = [f"Sees: {', '.join(sources)}"] if sources else []
    if settled is not None:
        rest = "\n".join(line for line in rest.splitlines()
                          if line.strip() and _norm(line) not in settled
                          and not re.fullmatch(r"\[[^\]]+\]", line.strip()))
        if rest:
            lines += [f"{ALSO}:", *fenced(cut(rest))]
    elif rest:
        lines += [f"{REST}:", *fenced(cut(rest))]
    return lines


def item_history(outcomes: Sequence[tuple[str, bool]]) -> str:
    """An item's history in a run, from its train outcomes ``(graph id, solved)`` in
    scoring order: ``history: solved k/n in g graphs``, with ``, flaky`` when the
    outcome flipped at least twice in that order or one graph both solved and failed it.
    Empty with fewer than two outcomes (the one shown is all there is)."""
    if len(outcomes) < 2:
        return ""
    solved = sum(1 for _graph, passed in outcomes if passed)
    graphs = len({graph for graph, _passed in outcomes})
    flips = sum(1 for (_a, x), (_b, y) in zip(outcomes, outcomes[1:]) if x != y)
    mixed = {graph for graph, passed in outcomes if passed} & {
        graph for graph, passed in outcomes if not passed}
    return (f"history: solved {solved}/{len(outcomes)} in {graphs} "
            f"graph{'' if graphs == 1 else 's'}" + (", flaky" if flips >= 2 or mixed else ""))


def render_item(record: Mapping[str, Any], *, input_keys: Sequence[str],
                outcome: str, expected: str, metadata: str, choices: Sequence[str],
                prompts: bool = True, prompt_chars: int = _WHOLE, reply_chars: int = _WHOLE,
                omissions: list[dict[str, Any]] | None = None,
                heading: str = "###", missing: Sequence[str] = (), label: str = "",
                show_input: bool = True, show_expected: bool = True,
                settled: Collection[str] | None = None, expected_label: str = "",
                errors_said: Mapping[str, str] | None = None) -> str:
    """One item: its verdict, what was expected and what it gave, the grader's reason, the
    task's input once, then each step -- with ``prompts``, what it saw by name and the rest
    of its prompt (a step run again: what it repeats from its previous run said by that
    run's name, `_Seen.split`) -- and its reply, every text fenced.
    ``prompt_chars``/``reply_chars`` clip (head and tail, marked, each cut in
    ``omissions``). ``label`` names this rendering
    of the item (a rollout) in its heading, so a "same as" reference says which one;
    without ``show_input``/``show_expected`` the input and the expected answer are left
    out (another rendering of the same item shows them), the input still read as what a
    step's prompt took ("Sees: the input"). With ``settled`` (the lines of the texts the
    request shows elsewhere, each step's settings: `settled_lines`), each step says what it
    saw by name and only the part of its prompt that is in neither (a judge's reason on a
    re-run, say), as `ALSO`; its instructions are left to where they are shown. With
    ``expected_label`` (the label of the expected answer's choice), the expected answer is
    only that label when the item's choices are shown here, since its choice line says
    the rest. ``errors_said`` maps an error text the request lists elsewhere to where:
    the item's error is said as that. A record's ``item_history`` (`item_history`: the loop
    sets it on a failing item, and on a passing one that failed before) ends its heading."""
    meta = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    exchanges = [e for e in record.get("exchanges") or () if isinstance(e, Mapping)]
    item = record.get("id")
    key = next((k for k in input_keys if meta.get(k)), None)
    if expected_label and choices and show_input and key is not None:
        expected = expected_label

    def cut(value: Any, step: str, limit: int) -> str:
        return clip(value, limit, item=item, step=step, omissions=omissions)

    # The item's history in the run (``item_history``: the loop sets it on a failing item
    # and on a flaky pass, never on a pass that never failed).
    history = str(record.get("item_history") or "")
    lines = [f"{heading} Item {item}{f' ({label})' if label else ''} — {outcome} "
             f"(score {_fmt_score(record.get('score'))})"
             + (f" — {metadata}" if metadata else "") + (f" — {history}" if history else "")]
    said = [part for part in (
        f"Expected: {cut(expected, 'expected', reply_chars)}"
        if expected and show_expected else "",
        f"Given: {cut(record.get('prediction'), 'prediction', reply_chars)}"
        if record.get("prediction") is not None else "") if part]
    if said:
        lines.append("; ".join(said))
    if (record.get("fail_reason") and not record.get("passed")
            and str(record["fail_reason"]).strip() != str(record.get("error_text") or "").strip()):
        lines.append(f"Grader: {cut(record['fail_reason'], 'grader', reply_chars)}")
    lines += list(missing)
    if record.get("error_text"):
        said = (errors_said or {}).get(str(record["error_text"]))
        lines.append(f"Error: {said}" if said else
                     f"Error: {cut(record['error_text'], 'error', reply_chars)}")
    seen = _Seen()
    if key is not None:
        if show_input:
            text = cut(meta[key], "input", reply_chars)
            if choices:
                text += "\n\nChoices:\n" + "\n".join(cut(c, "choices", reply_chars)
                                                       for c in choices)
            lines += [f"Input ({key}):", *fenced(text)]
        # A choice line is matched however short ("A. 3"); any other line of the input
        # only from `MIN_LINE` characters, so a step's own short line ("A", "---", "1.")
        # is never taken for the input.
        seen.add(meta[key], "the input")
        for choice in choices:
            seen.add(choice, "the input", 1)
    elif exchanges and not prompts and settled is None:
        lines += ["Input (the first step's prompt):",
                  *fenced(cut(exchanges[0].get("prompt"), "input", prompt_chars))]
    for number, exchange in enumerate(exchanges, 1):
        node = str(exchange.get("node") or "agent")
        lines.append(f"{heading}# Step `{node}` ({number} of {len(exchanges)})")
        shown = prompts or settled is not None
        if shown:
            lines += prompt_lines(seen, exchange.get("prompt"), settled,
                                  lambda text, node=node: cut(text, node, prompt_chars), node)
        lines += ["Reply:", *fenced(cut(exchange.get("response"), node, reply_chars))]
        seen.add(exchange.get("response"), f"{node}'s reply")
        # What this run showed, which the step's next run may repeat as its opening.
        seen.ran(node, [*([(f"{node}'s prompt (step {number})", exchange.get("prompt"))]
                          if shown else []),
                        (f"{node}'s reply (step {number})", exchange.get("response"))])
    if not exchanges:
        lines.append("(no step recorded a model call for this item)")
    return "\n".join(lines)


_REST_BLOCK = re.compile(rf"(?ms)^({re.escape(REST)}|{re.escape(ALSO)}):\n(`{{3,}})\n(.*?)\n\2$")
_ITEM = re.compile(r"(?m)^#+ Item (.+?) — ")
_STEP = re.compile(r"(?m)^#+ Step `([^`]+)`")


#: A line of an instruction this long, shown in an earlier one of the same request, is
#: said as where it was shown instead of again.
DEDUP_LINE = 40


def dedup(blocks: Sequence[str]) -> list[str]:
    """The items of one request, each text of a step's prompt it shows (`REST`, or `ALSO`:
    a reason a judge gave, the same in two rollouts of an item) that an earlier item
    showed said as "same as item X, step Y" instead of again; and in one that differs,
    each long line an earlier one showed said as where it was."""
    first: dict[str, str] = {}
    lines_at: dict[str, str] = {}
    out = []
    for block in blocks:
        item = (_ITEM.search(block) or [None, "?"])[1]

        def swap(match: re.Match[str], block: str = block, item: str = item) -> str:
            name, fence, text = match.group(1), match.group(2), match.group(3)
            body = _norm(text)
            steps = _STEP.findall(block[:match.start()])
            where = f"item {item}, step {steps[-1] if steps else '?'}"
            if body in first:
                said = f"{name}: same as {first[body]}."
                # Only when saying so is shorter than the text itself.
                return said if len(said) < len(match.group(0)) else match.group(0)
            first[body] = where
            kept, last = [], None
            for line in text.splitlines():
                key = _norm(line)
                seen = lines_at.get(key) if len(key) >= DEDUP_LINE else None
                if seen is None:
                    kept.append(line)
                    last = None
                    if len(key) >= DEDUP_LINE:
                        lines_at[key] = where
                elif seen != last:
                    kept.append(f"[a line of {seen}'s instructions, as shown there]")
                    last = seen
            return f"{name}:\n{fence}\n" + "\n".join(kept) + f"\n{fence}"

        out.append(_REST_BLOCK.sub(swap, block))
    return out


def dedup_texts(texts: Sequence[str], labels: Sequence[str]) -> list[str]:
    """Each of ``texts`` (labelled), its long lines an earlier one showed said as where."""
    lines_at: dict[str, str] = {}
    out = []
    for text, label in zip(texts, labels):
        kept, last = [], None
        for line in str(text or "").splitlines():
            key = _norm(line)
            seen = lines_at.get(key) if len(key) >= DEDUP_LINE else None
            if seen is None:
                kept.append(line)
                last = None
                if len(key) >= DEDUP_LINE:
                    lines_at[key] = label
            elif seen != last:
                kept.append(f"[as in {seen}]")
                last = seen
        out.append("\n".join(kept))
    return out
