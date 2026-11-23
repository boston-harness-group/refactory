"""One item's evidence, rendered once for every call that reads items.

The proposer, the analysts and memory's comparison all read train items. A step's prompt
re-embeds what came before it -- the task's input, earlier steps' replies -- and the same
instruction text recurs in every item, so an item shown as each step's prompt plus its
reply grows with the square of its steps, and a batch repeats each instruction once per
item. Here an item is shown with:

- the task's input once;
- what the harness did apart from what a model wrote: the steps the harness completed as
  code (its op and code-gate visits) named once with their counts (`harness_line`), and
  under each model step's first run how it runs (`ran_as`: a model's reply, the tools its graph gave it and
  its real tool calls where the harness records them), from the record and the graph alone
  -- never from reading a reply -- so a tool call or a grader a model wrote into its own
  reply reads as that model's claim (`CLAIMS`);
- for each step, what its prompt took from what is already shown (the input, an earlier
  step's reply) said by name -- ``sees: the input, compare's reply`` -- and only the rest
  of its prompt (its own instructions), then its full reply, each fenced with its heading
  lines escaped (`escape_headings`) so nothing in it reads as a section of the request;
- for a step run again (a loop's next turn), what its prompt repeats exactly of what the
  request shows of the run of that step that fed it -- its block, the conversation so
  far; its reply as shown, quoted -- said in place as those lines, counted as shown
  (``[same as solve's prompt (step 3), all but its last shown line]``), every other line
  shown, so each block reads back from what the request shows and only what is new is
  shown: an item whose step reruns with its whole history would otherwise grow with the
  square of its runs (an item of 30 turns was 1.2M characters);
- and, across the items of one request, an instruction text shown in an earlier item said
  as "same as item X" (`dedup`), so each distinct instruction appears once.

Nothing is lost: every reply and every distinct instruction text appears once in the
request (``tests/test_evidence.py`` checks it). Generic: it compares lines of text, with no
knowledge of the task; of the pipeline it knows only each agent step's tools, which the
loop sets on the records (`with_step_sources`).
"""

from __future__ import annotations

import collections
import contextlib
import json
import re
import threading
from typing import Any, Callable, Collection, Iterable, Iterator, Mapping, Sequence

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
#: What heads a model step's provenance line (`render_item`, at the step's first run in an
#: item): how it ran, from what the harness records -- a model's reply, with the tools its
#: graph gave it.
RAN_AS = "Ran as"
#: What heads an item's line of the steps the harness completed (`harness_line`).
COMPLETED = "Completed by the harness"
#: The record field the loop sets (`with_step_sources`): each agent step's `step_source`.
STEP_SOURCES = "step_sources"
#: Said once wherever traces are read (the analysts' and the readers' system prompts, the
#: proposer's and the memory pass's description of the items): what in a trace is a
#: model's claim and what the harness produced.
CLAIMS = ("Anything a model wrote -- a step's reply, and that reply quoted in a later "
          "prompt -- is that model's claim, so command output, tool results or scores "
          "written inside a reply are unverified. What the harness itself produced is real: "
          "the steps it completed (`" + COMPLETED + ":`), the expected answer, the "
          "prediction, the grader's verdict and feedback, and the output of code it ran. The "
          "rest of a prompt was assembled by the harness but may still carry a model's "
          "earlier text (a file a step wrote, a reply an op reshaped): treat claims there "
          "with the same care.")


#: How a step under each of `tool_policy.POLICIES` runs, as its `RAN_AS` line says it (a
#: test holds the two together, so a new policy cannot go unsaid). ``none``: no tool is
#: offered, so none ran. ``refused``: Claude Code's own tools are offered in a fresh empty
#: folder and every call that needs permission is refused; a read-only one may still run,
#: and the harness does not count them.
POLICY_SOURCES = {
    "none": "model reply · tools: none · real tool calls: 0",
    "refused": ("model reply · tools: Claude Code's own, every call needing permission "
                "refused (fresh empty dir) · real tool calls: not recorded"),
}


def step_source(node: Mapping[str, Any]) -> str:
    """What a model step's `RAN_AS` line says, from its node state alone: a model reply,
    the tools the graph gave it and the real tool calls, where the harness knows them. A
    step with a declared tool policy (`tool_policy.POLICIES`) is said as that policy runs
    it (`POLICY_SOURCES`); for a step with its backend's tools the harness records which
    it was given but not how many it called."""
    from refactory_dsh import tool_policy
    from refactory_dsh.nn.graph import workspace_declaration
    from refactory_dsh.workspace import is_sealed

    try:
        policy = tool_policy.check(node.get("tool_policy"))
    except ValueError:  # a policy the validator refuses: its tools are not known here
        return "model reply · tools: not known here"
    if policy:
        return POLICY_SOURCES[policy]
    tools = [str(tool) for tool in node.get("tools") or ()]
    try:
        sealed, own = is_sealed(dict(node)), workspace_declaration(node.get("workspace"))
    except ValueError:  # a declaration the validator refuses: said as the run dir's
        sealed, own = False, None
    where = "sealed workspace" if sealed else "own workspace" if own is not None else "run dir"
    given = ", ".join(tools) if tools else "the backend's own"
    return f"model reply · tools: {given} ({where}) · real tool calls: not recorded"


def step_sources(nodes: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None
                 ) -> dict[str, str]:
    """Each agent step of a graph (node states, by id or as a list) with its `step_source`."""
    states = nodes.values() if isinstance(nodes, Mapping) else (nodes or ())
    return {str(node.get("id")): step_source(node) for node in states
            if isinstance(node, Mapping) and node.get("kind") == "agent"}


def with_step_sources(records: Iterable[Mapping[str, Any]],
                      nodes: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None
                      ) -> list[dict[str, Any]]:
    """``records`` (one graph's train records) each with its graph's `step_sources`
    (``STEP_SOURCES``), which `render_item` says under each step; no graph, unchanged."""
    sources = step_sources(nodes) if nodes else {}
    return [{**record, STEP_SOURCES: sources} if sources else dict(record)
            for record in records]


def ran_as(record: Mapping[str, Any], node: str) -> str:
    """A model step's `RAN_AS` line in ``record``: its `step_source` when the loop set the
    graph's (`with_step_sources`); else only what every recorded exchange is, a model's
    reply (the record keeps one exchange per model call), its tools not known here."""
    sources = record.get(STEP_SOURCES)
    said = sources.get(node) if isinstance(sources, Mapping) else None
    return f"{RAN_AS}: {said or 'model reply · tools: not known here'}"


def harness_line(record: Mapping[str, Any]) -> str:
    """The steps the harness completed on the item -- its op and code-gate visits, the
    record's ``steps``, each kept once the visit succeeded -- each named once in the order
    it first completed, with how often (``xN``) and ``(gate)`` for a code gate; empty
    without any. Their order among the model steps is not recorded, so they are said apart
    from them."""
    runs: dict[str, list[Any]] = {}
    for visit in record.get("steps") or ():
        if isinstance(visit, Mapping) and visit.get("node"):
            held = runs.setdefault(str(visit["node"]), [visit.get("kind"), 0])
            held[1] += 1
    if not runs:
        return ""
    return f"{COMPLETED}: " + ", ".join(
        f"{node}{' (gate)' if kind == 'gate' else ''}{f' x{count}' if count > 1 else ''}"
        for node, (kind, count) in runs.items())


#: How an item rendered here reads, said once for every call that shows items.
TRACE_FORM = ("its input once, the steps the harness completed as code, each named once with "
              "how often (`" + COMPLETED + ":`; their order among the model steps is not "
              "recorded), then each model step in the order the steps ran -- at its first "
              "run in the request, how it runs (`" + RAN_AS + ":` a model's reply, the tools "
              "its graph gave it and its real tool calls where the harness records them), "
              "what its prompt "
              "took from the input, an earlier step's reply or, for a step run again, the "
              "run of it that fed it, said by name (`Sees:`; what a run repeats exactly of "
              "what is shown of that run, in place as those lines, counted as shown: "
              "`[same as ...]`), any "
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


def _norm_lines(text: str) -> tuple[str, ...]:
    """Each line of ``text`` in its normal form (`_norm`), kept while `keeping` lasts (an
    item is rendered again and again to fit a request)."""
    return _kept("norm", text, lambda: tuple(_norm(line) for line in text.splitlines()),
                 lambda found: sum(map(len, found)) + KEPT_OVERHEAD * len(found))


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
    return list(_kept("fenced", text, lambda: _fenced(text),
                      lambda found: len(found[1]) + len(text) + 3 * KEPT_OVERHEAD))


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
        # Per step, each run: its prompt and reply as written (what shows a later run was
        # fed it), the reply's `_probe`, and its texts as shown (`_Source`), which a later
        # run may repeat.
        self.runs: dict[str, list[tuple[str, str, Any, tuple[_Source, ...]]]] = {}

    def add(self, text: Any, label: str, shortest: int = MIN_LINE) -> None:
        for key in _norm_lines(str(text or "")):
            if key and len(key) >= shortest:
                self.where.setdefault(key, label)

    def ran(self, node: str, reply: tuple[str, Any, str | None, tuple[int, int] | None],
            prompt: tuple[str, Any, tuple[tuple[int, int, Any], ...] | None,
                          tuple[int, int] | None] | None = None) -> None:
        """A run of step ``node`` as shown: its reply ``(label, text, the text shown of it
        unclipped (stripped), or None when not shown, what of it is shown complete when
        clipped (`_complete`), or None)`` and, when its prompt was shown, ``(label, text,
        the lines of the prompt each line its block shows stands for (`view`), what of the
        block is shown complete when clipped, or None)``. A later run of the step may
        repeat what is shown (`split` with ``node``)."""
        sources: list[_Source] = []
        if prompt is not None and prompt[2]:
            sources.append(("prompt", prompt[0], str(prompt[1] or ""), tuple(prompt[2]),
                            prompt[3]))
        if reply[2]:
            sources.append(("reply", reply[0], str(reply[2]), reply[3]))
        written = str(reply[1] or "")
        self.runs.setdefault(node, []).append(
            (str(prompt[1] or "") if prompt is not None else "", written, _probe(written),
             tuple(sources)))

    def _earlier(self, prompt: str, node: str | None) -> tuple[_Source, ...]:
        """What is shown of the run of step ``node`` this prompt certainly came after on
        the same path: the latest earlier run whose reply (at least `FED_MIN` characters)
        the prompt quotes whole, as `_blocks` matches it -- the run fed it. Same-named runs
        on parallel paths feed each other nothing, so none of theirs is named. When several
        runs wrote that reply, the one whose prompt this prompt continues furthest
        (`_shared`) is: the run on its path; none, when that ties."""
        if node is None:
            return ()
        runs = self.runs.get(node, [])[-RUNS_SEARCHED:]
        for at in range(len(runs) - 1, -1, -1):
            _prompt, reply, probe, sources = runs[at]
            if probe is None or probe[0] not in prompt or not _fed(prompt, reply, probe):
                continue
            alike = [k for k, (_p, other, _probe, _s) in enumerate(runs) if other == reply]
            if len(alike) == 1:
                return sources
            shared = {k: _shared(prompt, runs[k][0]) for k in alike}
            most = max(shared.values())
            chosen = [k for k, count in shared.items() if count == most]
            return runs[chosen[0]][3] if most and len(chosen) == 1 else ()
        return ()

    def view(self, prompt: Any, node: str | None = None,
             settled: Collection[str] | None = None
             ) -> tuple[list[str], list[tuple[str, int, int, Any]]]:
        """What ``prompt`` took from what is shown (each source once, in order), and the
        lines its block shows, each with the lines of the prompt it stands for (``start``,
        ``end``): its own lines, runs of shown lines said by name, and what it repeats
        exactly of what the run of that step that fed it showed (`_earlier`), said in place
        as those lines (`_repeats`; such a line also links to what it names, `_pieces`).
        With ``settled`` (`ALSO`), blank lines, the lines its settings hold and the names of
        shown lines are not shown; a repeat always is."""
        sources: list[str] = []
        entries: list[tuple[str, int, int, Any]] = []
        last = None
        prompt = str(prompt or "")
        lines = prompt.splitlines()
        marks = {start: (end, label, said, link) for start, end, label, said, link
                 in _repeats(prompt, self._earlier(prompt, node))}
        index = 0
        while index < len(lines):
            if index in marks:
                end, label, said, link = marks[index]
                if label not in sources:
                    sources.append(label)
                entries.append((said, index, end, link))
                last = None
                index = end
                continue
            line = lines[index]
            label = self.where.get(_norm(line))
            if label is None:
                entries.append((line, index, index + 1, None))
                last = None
            else:
                if label not in sources:
                    sources.append(label)
                if label != last:
                    entries.append((f"[{label}]", index, index + 1, None))
                    last = label
                else:
                    text, start, _end, _link = entries[-1]
                    entries[-1] = (text, start, index + 1, None)
            index += 1
        if settled is not None:
            entries = [entry for entry in entries if entry[0].strip() and (
                SAME in entry[0] or (_norm(entry[0]) not in settled
                                     and not re.fullmatch(r"\[[^\]]+\]", entry[0].strip())))]
        while entries and not entries[0][0].strip():
            entries.pop(0)
        while entries and not entries[-1][0].strip():
            entries.pop()
        # A prompt that is only names for what was shown has no instructions of its own; a
        # repeat said in place (`SAME`) is kept, as it says which lines it stands for.
        if all(re.fullmatch(r"\[[^\]]+\]", text.strip()) and SAME not in text
               for text, _s, _e, _l in entries):
            entries = []
        return sources, entries

    def split(self, prompt: Any, node: str | None = None) -> tuple[list[str], str]:
        """`view` with every line shown (`REST`): its sources, and its block's text."""
        sources, entries = self.view(prompt, node)
        return sources, "\n".join(text for text, _s, _e, _l in entries)


#: A text a later run may repeat, as the request shows it: ``("prompt", label, the prompt,
#: each line its block shows as ``(start, end, link)`` -- the lines of the prompt it stands
#: for, and, for a repeat of an earlier prompt, ``(that source, the line of it start
#: is)`` -- visible)`` or ``("reply", label, the reply stripped, visible)``; ``visible`` is
#: None when it is shown unclipped, else how many of its lines it shows complete at its
#: head and at its tail (`_complete`).
_Source = tuple


def _complete(text: str, clipped: str) -> tuple[int, int]:
    """How many of ``text``'s lines its ``clipped`` showing (head and tail around a marked
    count, `clip`) shows complete at its head, and at its tail."""
    lines, shown = text.splitlines(), clipped.splitlines()
    head = 0
    while head < min(len(lines), len(shown) - 1) and shown[head] == lines[head]:
        head += 1
    tail = 0
    while (tail < min(len(lines) - head, len(shown) - head - 1)
           and shown[-1 - tail] == lines[-1 - tail]):
        tail += 1
    return head, tail


def _probe(reply: str) -> tuple[str, int, int] | None:
    """A reply that could show a later run was fed it (at least `FED_MIN` characters):
    its longest line, stripped (a quick test for a quote), and its first and last lines
    that are not blank; ``None`` for a shorter one. Worked out once a run (`_Seen.ran`)."""
    lines = reply.splitlines()
    body = [k for k, line in enumerate(lines) if line.strip()]
    if not body or sum(len(lines[k]) for k in body) < FED_MIN:
        return None
    return max((lines[k] for k in body), key=len).strip(), body[0], body[-1]


def _shared(prompt: str, earlier: str) -> int:
    """The longest run of lines ``prompt`` repeats exactly of the prompt ``earlier``
    (`_blocks`): the conversation it continues, which a repeated turn cannot lengthen."""

    def make() -> int:
        lines = prompt.splitlines()
        return max((c1 - c0 for c0, c1, *_rest in _blocks(
            lines, [False] * len(lines), earlier.splitlines(), REPEAT_MIN, quoted=False)),
            default=0)

    return _kept("shared", (prompt, earlier), make, lambda _count: 1)


def _fed(prompt: str, reply: str, probe: tuple[str, int, int]) -> bool:
    """Whether ``prompt`` quotes ``reply`` (its `_probe`) whole (`_quotes_whole`): the run
    that wrote it fed the run that got the prompt."""
    return _kept("fed", (prompt, reply),
                 lambda: _quotes_whole(prompt.splitlines(), reply.splitlines(), probe),
                 lambda _found: 1)


def _matches(prompt: str, earlier: str) -> tuple[tuple[int, int, int, int, str, str], ...]:
    """The runs of lines ``prompt`` repeats exactly, as they are, of the prompt ``earlier``
    (`_blocks`), kept while `keeping` lasts."""

    def make() -> tuple[tuple[int, int, int, int, str, str], ...]:
        lines = prompt.splitlines()
        return tuple(_blocks(lines, [False] * len(lines), earlier.splitlines(), REPEAT_MIN,
                             quoted=False))

    return _kept("matches", (prompt, earlier), make,
                 lambda found: 4 * KEPT_OVERHEAD * len(found))


def _pieces(source: _Source, first: int, end: int
            ) -> list[tuple[int, int, str, str, _Source, int]]:
    """How lines ``first``..``end`` of ``source``'s prompt are said as lines a block shows:
    runs of whole lines its block shows, said as those (`_shown_span`); where a line of
    its block is itself a repeat of an earlier prompt and the lines fall within it only
    in part, as that earlier prompt's shown lines, recursively. Each ``(start, end, the
    line that says it, its label, the source it names, the line of that source start
    is)``, in ``source``'s lines; a line said by none is left out (shown in place)."""
    _kind, label, _text, entries, visible = source
    out: list[tuple[int, int, str, str, _Source, int]] = []
    run: list[int] = []

    def flush() -> None:
        if run:
            said = _shown_span(label, run[0], run[-1] + 1, len(entries), visible)
            if said is not None:
                start = entries[run[0]][0]
                out.append((start, entries[run[-1]][1], said, label, source, start))
            run.clear()

    for k, (start, stop, link) in enumerate(entries):
        if stop <= first or start >= end:
            continue
        if first <= start and stop <= end:
            if run and run[-1] != k - 1:
                flush()
            run.append(k)
            continue
        flush()
        if link is not None:
            earlier, at = link
            low, high = max(first, start), min(end, stop)
            for y0, y1, said, name, named, local in _pieces(
                    earlier, low - start + at, high - start + at):
                out.append((y0 - at + start, y1 - at + start, said, name, named, local))
    flush()
    return out


def _repeats(prompt: str, sources: tuple[_Source, ...]
             ) -> list[tuple[int, int, str, str, Any]]:
    """What ``prompt`` repeats exactly of what ``sources`` show, in turn: each ``(start
    line, end line, label, the line that says it, link)``. A prompt's repeat is said as
    lines a block shows (`_pieces`: its own block's, or, within one of its own repeats, the
    earlier block's), each of at least `REPEAT_MIN` characters, linked to what it names; a
    reply's as lines of the reply as shown (`_same`), unlinked."""
    lines = prompt.splitlines()
    covered = [False] * len(lines)
    out: list[tuple[int, int, str, str, Any]] = []
    for source in sources:
        if source[0] == "prompt":
            for c0, _c1, e0, e1, _lead, _indent in _matches(prompt, source[2]):
                for y0, y1, said, name, named, local in _pieces(source, e0, e1):
                    a, b = c0 + y0 - e0, c0 + y1 - e0
                    if (not any(covered[a:b])
                            and sum(len(line) + 1 for line in lines[a:b]) >= REPEAT_MIN):
                        covered[a:b] = [True] * (b - a)
                        out.append((a, b, name, said, (named, local)))
        else:
            _kind, label, text, visible = source
            before = text.splitlines()
            for c0, c1, e0, e1, lead, indent in _blocks(lines, covered, before, REPEAT_MIN):
                # A reply shown clipped is said whole (as shown, what it left out marked
                # there) or by lines its head shows complete.
                if visible is None or (e0 == 0 and e1 == len(before)) or e1 <= visible[0]:
                    out.append((c0, c1, label, _same(label, e0, e1, len(before), lead,
                                                     indent), None))
                else:
                    covered[c0:c1] = [False] * (c1 - c0)
    return sorted(out, key=lambda mark: mark[0])


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


#: How a repeated text is said in place (`_same`, `_shown_span`); `view` keeps these lines.
SAME = "[same as "


def _same(label: str, first: int, end: int, total: int, lead: str, indent: str) -> str:
    """The line that stands for lines ``first``..``end`` (0-based, end exclusive) of a reply
    as shown, of ``total`` lines, labelled ``label``: the whole reply, or those lines;
    with ``lead``, the text before the first of them on its line; with ``indent``, the
    prefix every further line has (said only when there is one)."""
    span = ("" if first == 0 and end == total else
            f", line {first + 1}" if end - first == 1 else f", lines {first + 1}-{end}")
    prefixed = (f", each line after the first prefixed {json.dumps(indent)}"
                if indent and end - first > 1 else "")
    return f"{lead}{SAME}{label}{span}{prefixed}]"


def _shown_span(label: str, first: int, end: int, total: int,
                visible: tuple[int, int] | None = None) -> str | None:
    """The line that stands for lines ``first``..``end`` (0-based, end exclusive) of the
    ``total`` lines the block of the prompt labelled ``label`` shows, counted as shown.
    A block shown clipped (``visible``: its lines shown complete at its head and its
    tail) is said whole (as shown, what it left out marked there) or counted from an end
    only over lines it shows complete; ``None`` when that cannot say the span."""

    def lines(count: int) -> str:
        return "shown line" if count == 1 else f"{count} shown lines"

    head, tail = visible if visible is not None else (total, total)
    if first == 0 and end == total:
        span = ""
    elif first == 0:
        span = f", all but its last {lines(total - end)}" if total - end <= tail else None
    elif end == total:
        span = f", all but its first {lines(first)}" if first <= head else None
    elif visible is None or end <= head:
        span = (f", its shown line {first + 1}" if end - first == 1 else
                f", its shown lines {first + 1}-{end}")
    elif first <= head and total - end <= tail:
        span = f", all but its first {lines(first)} and its last {lines(total - end)}"
    else:
        span = None
    return None if span is None else f"{SAME}{label}{span}]"


def _blocks(lines: Sequence[str], covered: list[bool], earlier: Sequence[str],
            least: int, quoted: bool = True) -> list[tuple[int, int, int, int, str, str]]:
    """The runs of ``lines`` that repeat runs of ``earlier`` exactly, each at least
    ``least`` characters, none over a line already ``covered`` (which it marks): each
    ``(start, end, earlier start, earlier end, lead, indent)``. A repeat may be quoted:
    every line after its first is ``indent`` + the earlier line, its first ``lead`` + the
    earlier line (a turn label, ``- Assistant: ``); otherwise both are empty. Found from
    lines of `ANCHOR` or more characters (keyed by their last `ANCHOR`, so a quoted line
    is found whatever its prefix) whose key occurs at most `ANCHOR_REPEATS` times in
    ``earlier``, then extended both ways. Without ``quoted``, only lines repeated as they
    are."""
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
            if not lines[i].endswith(earlier[j]) or (not quoted and lines[i] != earlier[j]):
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
            if (quoted and first > 0 and start > 0 and not covered[start - 1]
                    and earlier[first - 1] and lines[start - 1].endswith(earlier[first - 1])):
                start, first = start - 1, first - 1
                lead = lines[start][:len(lines[start]) - len(earlier[first])]
            forms.append((start, i + end, first, j + end, lead, head))
            # This line under a label, the lines after it prefixed alike (the prefix read
            # off the next line that is not blank).
            step = next((k for k in range(1, min(len(lines) - i, len(earlier) - j))
                         if earlier[j + k]), None)
            if quoted and step is not None and lines[i + step].endswith(earlier[j + step]):
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
                 cut: Any, node: str | None = None
                 ) -> tuple[list[str], tuple[tuple[int, int, Any], ...], tuple[int, int] | None]:
    """What a step's prompt took from what is shown, by name (``Sees:``), then the rest of
    it: whole (`REST`), or, with ``settled``, only what the request does not show
    elsewhere (`ALSO`); ``cut`` clips that text. With ``node`` (the step), what it repeats
    exactly of what the run of the step that fed it showed is said in place as those lines
    (`_Seen.view`). Also the lines of the prompt each shown line stands for, which a later
    run may repeat, and, when the block is shown clipped, how many of its lines it shows
    complete at its head and its tail (`_complete`; else None)."""
    sources, entries = seen.view(prompt, node, settled)
    lines = [f"Sees: {', '.join(sources)}"] if sources else []
    text = "\n".join(line for line, _start, _end, _link in entries)
    if not text:
        return lines, (), None
    clipped = cut(text)
    whole = clipped == text.strip()
    lines += [f"{REST if settled is None else ALSO}:", *fenced(text if whole else clipped)]
    return (lines, tuple((start, end, link) for _line, start, end, link in entries),
            None if whole else _complete(text.strip(), clipped))


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
    sets it on a failing item, and on a passing one that failed before) ends its heading.
    What the harness did is said apart from what a model wrote: the steps the harness
    completed (`harness_line`), and under each model step's first run how it runs
    (`ran_as`)."""
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
    # What the Task declares the item's solver was given beyond its input, which no model
    # call of the record carries (``item_context``, `Task.item_context`), each in full.
    context = record.get("item_context") if isinstance(record.get("item_context"), Mapping) else {}
    for name, value in context.items():
        if show_input and value not in (None, ""):
            lines += [f"Given to the solver ({name}):", *fenced(cut(value, "input", reply_chars))]
            seen.add(value, f"the item's {name}")
    completed = harness_line(record)
    if completed:
        lines.append(completed)
    said_how: set[str] = set()
    for number, exchange in enumerate(exchanges, 1):
        node = str(exchange.get("node") or "agent")
        lines.append(f"{heading}# Step `{node}` ({number} of {len(exchanges)})")
        if node not in said_how:  # how a step runs, once: at its first run in the item
            said_how.add(node)
            lines.append(ran_as(record, node))
        shown = prompts or settled is not None
        ranges, visible = None, None
        if shown:
            block, ranges, visible = prompt_lines(
                seen, exchange.get("prompt"), settled,
                lambda text, node=node: cut(text, node, prompt_chars), node)
            lines += block
        reply = cut(exchange.get("response"), node, reply_chars)
        lines += ["Reply:", *fenced(reply)]
        seen.add(exchange.get("response"), f"{node}'s reply")
        # What this run showed, which a later run of the step may repeat.
        whole = str(exchange.get("response") or "").strip()
        seen.ran(node, (f"{node}'s reply (step {number})", exchange.get("response"), whole,
                        None if reply == whole else _complete(whole, reply)),
                 (f"{node}'s prompt (step {number})", exchange.get("prompt"), ranges, visible)
                 if shown else None)
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
    """The items of one request, each step's `RAN_AS` line an earlier item said left out
    (`said_once`), and each text of a step's prompt it shows (`REST`, or `ALSO`:
    a reason a judge gave, the same in two rollouts of an item) that an earlier item
    showed said as "same as item X, step Y" instead of again; and in one that differs,
    each long line an earlier one showed said as where it was. An item that says a repeat
    in place (`SAME`) is left as it is: a later run's repeat counts the lines its blocks
    show, which this would change; its texts are still where a later item's are said to
    be."""
    first: dict[str, str] = {}
    lines_at: dict[str, str] = {}
    how: set[tuple[str, str]] = set()
    out = []
    for block in blocks:
        item = (_ITEM.search(block) or [None, "?"])[1]
        kept_whole = SAME in block

        def swap(match: re.Match[str], block: str = block, item: str = item,
                 kept_whole: bool = kept_whole) -> str:
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
            if body in first and not kept_whole:
                said = f"{name}: same as {first[body]}."
                # Only when saying so is shorter than the text itself.
                return said if len(said) < len(match.group(0)) else match.group(0)
            first.setdefault(body, where)
            kept, last = [], None
            for line in text.splitlines():
                key = _norm(line)
                seen = lines_at.get(key) if len(key) >= DEDUP_LINE else None
                if seen is None or kept_whole:
                    kept.append(line)
                    last = None
                    if len(key) >= DEDUP_LINE:
                        lines_at.setdefault(key, where)
                elif seen != last:
                    kept.append(f"[a line of {seen}'s instructions, as shown there]")
                    last = seen
            if kept_whole:
                return match.group(0)
            return f"{name}:\n{fence}\n" + "\n".join(kept) + f"\n{fence}"

        out.append(said_once(_REST_BLOCK.sub(swap, block), how))
    return out


def said_once(block: str, how: set[tuple[str, str]]) -> str:
    """``block`` without a step's `RAN_AS` line an earlier item of the request already said
    for that step (``how``, added to): how a step runs is said once a request (the
    analysts' and memory's through `dedup`, the proposer's items in turn). The line
    only ever follows a step's heading, which no fenced text can hold (`escape_headings`)."""
    lines = block.split("\n")
    kept: list[str] = []
    for at, line in enumerate(lines):
        step = _STEP.match(lines[at - 1]) if at and line.startswith(f"{RAN_AS}: ") else None
        if step is not None:
            if (step[1], line) in how:
                continue
            how.add((step[1], line))
        kept.append(line)
    return "\n".join(kept)


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


#: About the most bytes what renderings keep to read again may hold at once (`keeping`):
#: each entry counted as its characters plus `KEPT_OVERHEAD` a string it holds; tens of
#: megabytes at most while one pass lasts, none after it.
KEPT_CHARS = 64_000_000
#: What a kept string costs beyond its characters (a Python string's header), in bytes.
KEPT_OVERHEAD = 56


class _Kept:
    """What renderings read again (a text's lines in normal form, a fenced text, a repeat
    found), bounded by about the bytes it holds (characters, and `KEPT_OVERHEAD` a
    string): once full, nothing more is added."""

    def __init__(self, chars: int) -> None:
        self.chars = chars
        self.used = 0
        self.held: collections.OrderedDict[Any, tuple[Any, int]] = collections.OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: Any, make: Callable[[], Any], size: Callable[[Any], int]) -> Any:
        with self.lock:
            if key in self.held:
                self.held.move_to_end(key)
                return self.held[key][0]
        value = make()
        cost = size(value) + 2 * KEPT_OVERHEAD
        with self.lock:
            # Added only while there is room: a pass reads its texts again in the same
            # order, so evicting the oldest would miss every time; this keeps a part.
            if key not in self.held and self.used + cost <= self.chars:
                self.held[key] = (value, cost)
                self.used += cost
        return value


_KEEP: dict[str, Any] = {"store": None, "depth": 0}
_KEEP_LOCK = threading.Lock()


@contextlib.contextmanager
def keeping() -> Iterator[None]:
    """While it lasts, what renderings read again is kept (at most `KEPT_CHARS`), so the
    renderings of one request, one analysts' round or one memory pass -- the same items
    rendered again to fit -- do the work once; when the last one ends it is dropped, so
    no text outlives the pass that showed it. Outside it nothing is kept."""
    with _KEEP_LOCK:
        if not _KEEP["depth"]:
            _KEEP["store"] = _Kept(KEPT_CHARS)
        _KEEP["depth"] += 1
    try:
        yield
    finally:
        with _KEEP_LOCK:
            _KEEP["depth"] -= 1
            if not _KEEP["depth"]:
                _KEEP["store"] = None


def _kept(kind: str, key: Any, make: Callable[[], Any], size: Callable[[Any], int]) -> Any:
    """``make()``, kept under ``(kind, key)`` while `keeping` lasts (``size``: the
    characters it holds)."""
    store = _KEEP["store"]
    return make() if store is None else store.get((kind, key), make, size)
