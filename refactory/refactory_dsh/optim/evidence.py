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
- for a step run again (a loop's next turn), what its prompt repeats exactly from the run
  of that step that fed it -- the conversation so far, its last reply quoted -- said in
  place as those lines (``[same as solve's prompt (step 3), lines 1-40]``), every other
  line shown, so the prompt reads back exactly and only what is new is shown: an item whose
  step reruns with its whole history would otherwise grow with the square of its runs (an
  item of 30 turns was 1.2M characters);
- and, across the items of one request, an instruction text shown in an earlier item said
  as "same as item X" (`dedup`), so each distinct instruction appears once.

Nothing is lost: every reply and every distinct instruction text appears once in the
request (``tests/test_evidence.py`` checks it). Generic: it compares lines of text, with no
knowledge of the task or the pipeline.
"""

from __future__ import annotations

import functools
import json
import re
from typing import Any, Collection, Iterable, Mapping, Sequence

#: A reply's line shorter than this (after stripping) is never matched as repeated text:
#: a bare heading or label is not worth a reference. The task's input matches whole.
MIN_LINE = 12
#: What a step's prompt repeats exactly from the run of that step that fed it
#: (`_Seen.split`) is said as that text's lines only from this many characters: a
#: shorter repeat is left in place, as a marker would save little and say less.
REPEAT_MIN = 200
#: A run's reply quoted whole in a later run's prompt shows the later run came after it
#: on the same path (it was fed it) only from this many characters (`_Seen._earlier`).
FED_MIN = 40
#: The earlier runs of a step searched for the one that fed a run (`_Seen._earlier`).
RUNS_SEARCHED = 64
#: A line found more often than this in an earlier text never anchors a repeat
#: (`_blocks`): a common line (a fence, a closing brace) would anchor everywhere.
ANCHOR_REPEATS = 16
#: How long a line must be to anchor a repeat, and how much of its end keys it
#: (`_blocks`); shorter lines join a repeat only beside one that anchors it.
ANCHOR = 16
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
              "took from the input, an earlier step's reply or, for a step run again, the "
              "run of it that fed it, said by name (`Sees:`; what a run repeats exactly of "
              "that run, in place as those lines: `[same as ...]`), any "
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


@functools.lru_cache(maxsize=4096)
def _norm_lines(text: str) -> tuple[str, ...]:
    """Each line of ``text`` in its normal form (`_norm`), kept for the texts an item's
    renderings read again and again (a request is rendered several times to fit)."""
    return tuple(_norm(line) for line in text.splitlines())


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
    return list(_fenced(text))


@functools.lru_cache(maxsize=1024)
def _fenced(text: str) -> tuple[str, str, str]:
    body = escape_headings(text)
    fence = _fence(body)
    return fence, body, fence


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
    runs as shown (their prompts and replies), which a later run of the same step may
    repeat."""

    def __init__(self) -> None:
        self.where: dict[str, str] = {}
        # Per step, each run as shown: its texts (prompt if shown, then reply), each
        # ``(label, text)``, and its reply.
        self.runs: dict[str, list[tuple[tuple[tuple[str, str], ...], str]]] = {}

    def add(self, text: Any, label: str, shortest: int = MIN_LINE) -> None:
        for key in _norm_lines(str(text or "")):
            if key and len(key) >= shortest:
                self.where.setdefault(key, label)

    def ran(self, node: str, reply: tuple[str, Any],
            prompt: tuple[str, Any] | None = None) -> None:
        """A run of step ``node`` as shown: its reply and, when shown, its prompt (each
        ``(label, text)``), which a later run of the step may repeat (`split` with
        ``node``)."""
        texts = [(label, str(text or "")) for label, text in
                 ([prompt] if prompt is not None else []) + [reply]]
        self.runs.setdefault(node, []).append(
            (tuple((label, text) for label, text in texts if text), texts[-1][1]))

    def _earlier(self, prompt: str, node: str | None) -> tuple[tuple[str, str], ...]:
        """The texts of the run of step ``node`` this prompt certainly came after on the
        same path: the latest earlier run whose reply (at least `FED_MIN` characters) the
        prompt quotes whole, as `_blocks` matches it -- the run fed it. Same-named runs
        on parallel paths feed each other nothing, so none of theirs is named. Empty
        when no run fed it."""
        if node is None:
            return ()
        for texts, reply in reversed(self.runs.get(node, [])[-RUNS_SEARCHED:]):
            probe = _probe(reply)
            if probe is not None and probe[0] in prompt and _fed(prompt, reply):
                return texts
        return ()

    def split(self, prompt: Any, node: str | None = None) -> tuple[list[str], str]:
        """What ``prompt`` took from what is shown (each source once, in order) and the
        rest of it, its runs of shown lines said by name. With ``node``, what it repeats
        exactly from the run of that step that fed it (`_earlier`) -- the conversation so
        far, its last reply quoted -- is said as that text and those of its lines,
        ``[same as solve's prompt (step 3), lines 1-40]`` (`_same`), every other line in
        place, so the prompt can be read back exactly."""
        sources: list[str] = []
        rest: list[str] = []
        last = None
        prompt = str(prompt or "")
        lines = _lines(prompt)
        marks = {start: (end, label, said) for start, end, label, said
                 in _repeats(prompt, self._earlier(prompt, node))}
        index = 0
        while index < len(lines):
            if index in marks:
                end, label, said = marks[index]
                if label not in sources:
                    sources.append(label)
                rest.append(said)
                last = None
                index = end
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
        # A prompt that is only names for what was shown has no instructions of its own;
        # a repeat said in place (`SAME`) is kept, as it says which lines it stands for.
        only_refs = all((re.fullmatch(r"\[[^\]]+\]", line.strip()) and SAME not in line)
                        or not line.strip() for line in text.splitlines())
        return sources, "" if only_refs else text


@functools.lru_cache(maxsize=4096)
def _lines(text: str) -> tuple[str, ...]:
    """``text``'s lines, kept for the texts an item's renderings split again and again."""
    return tuple(text.splitlines())


@functools.lru_cache(maxsize=8192)
def _probe(reply: str) -> tuple[str, int, int] | None:
    """A reply that could show a later run was fed it (at least `FED_MIN` characters):
    its longest line, stripped (a quick test for a quote), and its first and last lines
    that are not blank; ``None`` for a shorter one."""
    lines = _lines(reply)
    body = [k for k, line in enumerate(lines) if line.strip()]
    if not body or sum(len(lines[k]) for k in body) < FED_MIN:
        return None
    return max((lines[k] for k in body), key=len).strip(), body[0], body[-1]


@functools.lru_cache(maxsize=8192)
def _fed(prompt: str, reply: str) -> bool:
    """Whether ``prompt`` quotes ``reply`` whole (`_quotes_whole`; ``reply`` has a
    `_probe`): the run that wrote it fed the run that got the prompt."""
    probe = _probe(reply)
    return probe is not None and _quotes_whole(_lines(prompt), _lines(reply), probe)


@functools.lru_cache(maxsize=4096)
def _repeats(prompt: str, earlier: tuple[tuple[str, str], ...]
             ) -> tuple[tuple[int, int, str, str], ...]:
    """What ``prompt`` repeats exactly of the ``earlier`` texts (``(label, text)``, in
    turn), each ``(start line, end line, label, the line saying it, `_same`)``."""
    lines = _lines(prompt)
    covered = [False] * len(lines)
    out = []
    for label, text in earlier:
        before = _lines(text)
        for c0, c1, e0, e1, lead, indent in _blocks(lines, covered, before, REPEAT_MIN):
            out.append((c0, c1, label, _same(label, e0, e1, len(before), lead, indent)))
    return tuple(out)


def _quotes_whole(lines: Sequence[str], reply: Sequence[str],
                  probe: tuple[str, int, int]) -> bool:
    """Whether ``lines`` quote ``reply`` whole -- every line from its first to its last
    that is not blank (``probe``'s), in order -- as `_blocks` matches a quote: each line
    after the first the same prefix + the reply's line, the first that prefix or a label
    + the reply's."""
    _longest, first, last = probe
    head = reply[first]
    # The prefix is read off the next line that is not blank, if there is one.
    later = next((k for k in range(first + 1, last + 1) if reply[k]), None)
    for i, line in enumerate(lines):
        if i + last - first >= len(lines) or not line.endswith(head):
            continue
        if later is None:
            return True
        quoted = lines[i + later - first]
        if not quoted.endswith(reply[later]):
            continue
        indent = quoted[:len(quoted) - len(reply[later])]
        if all(lines[i + k - first] == indent + reply[k] for k in range(first + 1, last + 1)):
            return True
    return False


#: How a repeated text is said in place (`_same`); `prompt_lines` keeps these lines.
SAME = "[same as "


def _same(label: str, first: int, end: int, total: int, lead: str, indent: str) -> str:
    """The line that stands for lines ``first``..``end`` (0-based, end exclusive) of a text
    of ``total`` lines labelled ``label``: the whole text, or those lines; with ``lead``,
    the text before the first of them on its line; with ``indent``, the prefix every
    further line has."""
    span = ("" if first == 0 and end == total else
            f", line {first + 1}" if end - first == 1 else f", lines {first + 1}-{end}")
    prefixed = (f", each line after the first prefixed {json.dumps(indent)}" if indent
                else "")
    return f"{lead}{SAME}{label}{span}{prefixed}]"


def _blocks(lines: Sequence[str], covered: list[bool], earlier: Sequence[str],
            least: int) -> list[tuple[int, int, int, int, str, str]]:
    """The runs of ``lines`` that repeat runs of ``earlier`` exactly, each at least
    ``least`` characters, none over a line already ``covered`` (which it marks): each
    ``(start, end, earlier start, earlier end, lead, indent)``. A repeat may be quoted:
    every line after its first is ``indent`` + the earlier line, its first ``lead`` + the
    earlier line (a turn label, ``- Assistant: ``); otherwise both are empty. Found from
    lines of `ANCHOR` or more characters (keyed by their last `ANCHOR`, so a quoted line
    is found whatever its prefix) whose key occurs at most `ANCHOR_REPEATS` times in
    ``earlier``, then extended both ways."""
    at: dict[str, list[int]] = {}
    for position, line in enumerate(earlier):
        if len(line) >= ANCHOR and len(line.strip()) >= 8:
            held = at.setdefault(line[-ANCHOR:], [])
            if len(held) <= ANCHOR_REPEATS:
                held.append(position)
    found = []

    def forward(start: int, at_earlier: int, indent: str) -> int:
        """How many lines from ``start`` are ``indent`` + ``earlier`` from ``at_earlier``."""
        end = 0
        while (start + end < len(lines) and at_earlier + end < len(earlier)
               and not covered[start + end]
               and lines[start + end] == indent + earlier[at_earlier + end]):
            end += 1
        return end

    i = 0
    while i < len(lines):
        best = None
        candidates = (at.get(lines[i][-ANCHOR:], ())
                      if not covered[i] and len(lines[i]) >= ANCHOR else ())
        for j in candidates if len(candidates) <= ANCHOR_REPEATS else ():
            if not lines[i].endswith(earlier[j]):
                continue
            head = lines[i][:len(lines[i]) - len(earlier[j])]
            forms = []
            # Every line prefixed alike, reaching back over earlier lines prefixed alike,
            # then perhaps one line under a label.
            end = forward(i, j, head)
            back = 0
            while (i - back > 0 and j - back > 0 and not covered[i - back - 1]
                   and lines[i - back - 1] == head + earlier[j - back - 1]):
                back += 1
            start, first, lead = i - back, j - back, head
            if (first > 0 and start > 0 and not covered[start - 1] and earlier[first - 1]
                    and lines[start - 1].endswith(earlier[first - 1])):
                start, first = start - 1, first - 1
                lead = lines[start][:len(lines[start]) - len(earlier[first])]
            forms.append((start, i + end, first, j + end, lead, head))
            # This line under a label, the lines after it prefixed alike (the prefix read
            # off the next line that is not blank).
            step = next((k for k in range(1, min(len(lines) - i, len(earlier) - j))
                         if earlier[j + k]), None)
            if step is not None and lines[i + step].endswith(earlier[j + step]):
                indent = lines[i + step][:len(lines[i + step]) - len(earlier[j + step])]
                end = forward(i + 1, j + 1, indent)
                forms.append((i, i + 1 + end, j, j + 1 + end, head, indent))
            for form in forms:
                size = sum(len(line) + 1 for line in lines[form[0]:form[1]])
                if best is None or size > best[0]:
                    best = (size, *form)
        if best is not None and best[0] >= least:
            _size, start, end, first, stop, lead, indent = best
            for k in range(start, end):
                covered[k] = True
            found.append((start, end, first, stop, lead, indent))
            i = end
        else:
            i += 1
    return found


def prompt_lines(seen: "_Seen", prompt: Any, settled: Collection[str] | None,
                 cut: Any, node: str | None = None) -> list[str]:
    """What a step's prompt took from what is shown, by name (``Sees:``), then the rest of
    it: whole (`REST`), or, with ``settled``, only what the request does not show
    elsewhere (`ALSO`); ``cut`` clips that text. With ``node`` (the step), what it repeats
    exactly from the run of the step that fed it is said in place as those lines
    (`_Seen.split`)."""
    sources, rest = seen.split(prompt, node)
    lines = [f"Sees: {', '.join(sources)}"] if sources else []
    if settled is not None:
        # A name for a shown line goes to ``Sees:``; a repeat (`SAME`) stays in place,
        # where it stands for its lines.
        rest = "\n".join(line for line in rest.splitlines()
                          if line.strip() and _norm(line) not in settled
                          and (SAME in line
                               or not re.fullmatch(r"\[[^\]]+\]", line.strip())))
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
    of its prompt (a step run again: what it repeats exactly from the run of it that fed
    it said in place as those lines, `_Seen.split`) -- and its reply, every text fenced.
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
        # What this run showed, which a later run of the step may repeat.
        seen.ran(node, (f"{node}'s reply (step {number})", exchange.get("response")),
                 (f"{node}'s prompt (step {number})", exchange.get("prompt")) if shown
                 else None)
    if not exchanges:
        lines.append("(no step recorded a model call for this item)")
    return "\n".join(lines)


_REST_BLOCK = re.compile(rf"(?ms)^({re.escape(REST)}|{re.escape(ALSO)}):\n(`{{3,}})\n(.*?)\n\2$")
_ITEM = re.compile(r"(?m)^#+ Item (.+?) — ")
_STEP = re.compile(r"(?m)^#+ Step `([^`]+)`(?: \((\d+) of (\d+))?")


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
            if SAME in text:
                # It names this item's own earlier runs: in another item it would name
                # that item's, so it is never said as another's, nor any line of it.
                return match.group(0)
            body = _norm(text)
            steps = _STEP.findall(block[:match.start()])
            step, number, total = steps[-1] if steps else ("?", "", "")
            # A step that ran more than once in the item is named by its run.
            again = sum(1 for other, _n, _t in _STEP.findall(block) if other == step) > 1
            where = (f"item {item}, step {step}"
                     + (f" ({number} of {total})" if again and number else ""))
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


def clear_caches() -> None:
    """Forget what renderings kept to read again (`_lines`, `_norm_lines`, `_fenced`,
    `_probe`, `_fed`, `_repeats`): a request clears them when it is made
    (`base.ProposalContext.render`), so no text outlives the round that showed it."""
    for cached in (_lines, _norm_lines, _fenced, _probe, _fed, _repeats):
        cached.cache_clear()
