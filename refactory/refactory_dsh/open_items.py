"""A pull request's Open items: one format, written by a run and read by a finish.

A design run gets one fix round and always delivers a pull request. What it leaves is
listed in the description in its own marker block, outside delivery's markers, and a
finish run (`--refine <PR> --finish`, `finish.py`) reads it, works on it, and writes back
only what it decided::

    <!-- refactory:open-items -->
    ## Open items

    - [ ] **qa-1** (qa_finding): the empty-input case prints a traceback
      - evidence: tests/test_cli.py::test_empty_input
      - evidence: cli.py:42
    - [x] **env-1** (environment): the lab's runner timed out
      - evidence: log `Timeout after 600 s`
      - result: finish run at `0123456789ab`: re-checked, the check passed
    <!-- /refactory:open-items -->

One item line each: a task-list box (``[x]`` when resolved), the id in bold, the kind in
brackets, and the summary. Under it, indented, ``- evidence: <text>`` lines are its
evidence (test ids, ``file:line``, a log excerpt, the excuse rule a run used) and
``- result: <text>`` lines are what a finish found. Any other line under it is a note:
kept, never evidence.

:func:`render` and :func:`parse` round-trip. :func:`update` edits a description in place:
inside the block it changes only the box of an item it has a result for and adds that
result's line under it; every other line, in the block or out of it, stays byte for byte,
including comments, malformed items, logs and a kind it does not know. A description with
no block but an ``## Open items`` heading is read and updated the same way, up to the next
heading. An unknown kind is kept as written, and read as ``judgment_call`` only to decide
whether a finish works on it unasked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: The block's markers and its heading.
START = "<!-- refactory:open-items -->"
END = "<!-- /refactory:open-items -->"
HEADING = "## Open items"

#: What an item can be.
KINDS = ("change_failure", "unfinished_ask", "qa_finding", "judgment_call", "environment")
#: The kinds a finish can fix without a person deciding anything first.
CONCRETE = ("change_failure", "unfinished_ask", "qa_finding")
#: The two states of an item.
STATUSES = ("open", "resolved")

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ITEM = re.compile(r"^(?P<lead>[-*+]\s+\[)(?P<box>[ xX])(?P<rest>\]\s+\*\*(?P<id>[^*\s]+)\*\*\s*"
                   r"\((?P<kind>[A-Za-z_]+)\)\s*:\s*(?P<summary>.*?)\s*)$")
_UNDER = re.compile(r"^(?:\s{2,}|\t)")
_LABELLED = re.compile(r"^\s+[-*+]\s+(?P<label>evidence|result):\s*(?P<text>\S.*?)\s*$", re.I)
_SECTION = re.compile(r"^##\s+Open items\s*$", re.I)
#: What ends a legacy section: the next heading of the same or a higher level, or a marker.
_LEGACY_END = re.compile(r"^(?:#{1,2}\s|<!--\s*/?(?:delivery|refactory):)")


def _flat(text: str) -> str:
    return " ".join(str(text).split())


@dataclass(frozen=True)
class OpenItem:
    """One thing a pull request leaves open, or one a later run resolved.

    ``kind`` is as written; `effective_kind` is what a finish acts on.
    """

    id: str
    kind: str
    summary: str
    evidence: tuple[str, ...] = field(default_factory=tuple)
    status: str = "open"
    results: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not _ID.match(self.id):
            raise ValueError(f"an open item's id is letters, digits, '.', '_' and '-', not {self.id!r}")
        if not re.fullmatch(r"[A-Za-z_]+", self.kind):
            raise ValueError(f"an open item's kind is a word, not {self.kind!r}")
        if self.status not in STATUSES:
            raise ValueError(f"an open item's status is open or resolved, not {self.status!r}")
        summary = _flat(self.summary)
        if not summary:
            raise ValueError(f"the open item {self.id} has no summary")
        object.__setattr__(self, "summary", summary)
        for name in ("evidence", "results"):
            object.__setattr__(self, name, tuple(e for e in (_flat(x) for x in getattr(self, name)) if e))

    @property
    def effective_kind(self) -> str:
        """The kind a finish acts on: one of `KINDS`, an unknown one read as ``judgment_call``."""
        kind = self.kind.lower()
        return kind if kind in KINDS else "judgment_call"

    @property
    def concrete(self) -> bool:
        """Whether a finish may fix it without asking: a concrete kind, with evidence."""
        return self.effective_kind in CONCRETE and bool(self.evidence)

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "kind": self.kind, "summary": self.summary,
                "evidence": list(self.evidence), "status": self.status, "results": list(self.results)}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "OpenItem":
        return cls(id=str(data["id"]), kind=str(data["kind"]), summary=str(data["summary"]),
                   evidence=tuple(str(e) for e in data.get("evidence") or ()),  # type: ignore[union-attr]
                   status=str(data.get("status") or "open"),
                   results=tuple(str(e) for e in data.get("results") or ()))  # type: ignore[union-attr]


def _item_lines(item: OpenItem) -> list[str]:
    box = "x" if item.status == "resolved" else " "
    return [f"- [{box}] **{item.id}** ({item.kind}): {item.summary}",
            *(f"  - evidence: {entry}" for entry in item.evidence),
            *(f"  - result: {entry}" for entry in item.results)]


def render(items: list[OpenItem] | tuple[OpenItem, ...]) -> str:
    """The Open items block for ``items``, markers included, ending with a newline."""
    lines = [START, HEADING, ""]
    for item in items:
        lines += _item_lines(item)
    if not items:
        lines.append("(none)")
    return "\n".join([*lines, END]) + "\n"


def _region(lines: list[str]) -> tuple[int, int] | None:
    """The first Open items region: (its first line inside, the line after it).

    The marker block when there is one; else an ``## Open items`` heading up to the next
    heading or marker. None when there is neither.
    """
    for index, line in enumerate(lines):
        if line.strip() == START:
            end = next((at for at in range(index + 1, len(lines)) if lines[at].strip() == END), None)
            if end is not None:
                return index + 1, end
    for index, line in enumerate(lines):
        if _SECTION.match(line):
            end = index + 1
            while end < len(lines) and not _LEGACY_END.match(lines[end]):
                end += 1
            return index + 1, end
    return None


def _items_in(lines: list[str], start: int, end: int) -> list[tuple[OpenItem, int, int]]:
    """Each item in ``lines[start:end]``: the item, its line, and the line after its last."""
    found: list[tuple[OpenItem, int, int]] = []
    index = start
    while index < end:
        matched = _ITEM.match(lines[index])
        if not matched:
            index += 1
            continue
        last = index + 1
        while last < end and lines[last].strip() and _UNDER.match(lines[last]):
            last += 1
        evidence, results, fence = [], [], False
        for line in lines[index + 1:last]:
            if line.strip().startswith(("```", "~~~")):
                fence = not fence
                continue
            labelled = None if fence else _LABELLED.match(line)
            if labelled:
                (evidence if labelled["label"].lower() == "evidence" else results).append(labelled["text"])
        try:
            item = OpenItem(id=matched["id"], kind=matched["kind"], summary=matched["summary"],
                            evidence=tuple(evidence), results=tuple(results),
                            status="resolved" if matched["box"] in "xX" else "open")
        except ValueError:
            item = None
        if item is not None and all(other.id != item.id for other, _, _ in found):
            found.append((item, index, last))
        index = last
    return found


def parse(body: str) -> list[OpenItem]:
    """The items of a pull request's first Open items region; empty when it has none.

    A line that is not an item, or not an item's evidence or result, is a person's and is
    left out; two items with one id keep the first.
    """
    lines = str(body or "").replace("\r\n", "\n").split("\n")
    region = _region(lines)
    return [] if region is None else [item for item, _, _ in _items_in(lines, *region)]


def update(body: str, outcome: dict[str, tuple[str, str]],
           missing: list[OpenItem] | tuple[OpenItem, ...] = ()) -> str:
    """``body`` with each result in ``outcome`` (id -> (status, result)) applied in place.

    An item it finds gets a ``- result:`` line after its last line, and its box ticked when
    the result resolves it; an open result leaves its box as it is. Nothing else changes.
    An item of ``missing`` that has a result but is no longer in the region is added at the
    region's end; with no region at all, a new block holding them goes at the end.
    """
    text = str(body or "")
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(newline)
    region = _region(lines)
    if region is None:
        added = [OpenItem(**{**item.to_dict(), "status": outcome[item.id][0],  # type: ignore[arg-type]
                             "results": (*item.results, outcome[item.id][1])})
                 for item in missing if item.id in outcome]
        if not added:
            return text
        block = render(added).rstrip("\n").replace("\n", newline)
        return text.rstrip(newline) + (newline * 2 if text.strip() else "") + block + newline
    start, end = region
    found = _items_in(lines, start, end)
    edits: list[tuple[int, int, list[str]]] = []
    for item, at, last in found:
        if item.id not in outcome:
            continue
        status, said = outcome[item.id]
        line = lines[at]
        if status == "resolved":
            matched = _ITEM.match(line)
            assert matched is not None
            line = matched["lead"] + "x" + matched["rest"]
        edits.append((at, last, [line, *lines[at + 1:last], f"  - result: {_flat(said)}"]))
    known = {item.id for item, _, _ in found}
    tail = [line for item in missing if item.id in outcome and item.id not in known
            for line in _item_lines(OpenItem(**{**item.to_dict(), "status": outcome[item.id][0],  # type: ignore[arg-type]
                                                 "results": (*item.results, outcome[item.id][1])}))]
    for at, last, replacement in sorted(edits, reverse=True):
        lines[at:last] = replacement
    if tail:
        shift = sum(len(r) - (last - at) for at, last, r in edits)
        insert = end + shift
        while insert > start and not lines[insert - 1].strip():
            insert -= 1
        lines[insert:insert] = tail
    return newline.join(lines)


def without(body: str) -> str:
    """``body`` with its Open items taken out: every marker block, and a legacy section."""
    lines = str(body or "").split("\n")
    kept: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.strip() == START:
            end = next((at for at in range(index + 1, len(lines)) if lines[at].strip() == END), None)
            if end is not None:
                index = end + 1
                continue
        if _SECTION.match(line):
            index += 1
            while index < len(lines) and not _LEGACY_END.match(lines[index]):
                index += 1
            continue
        kept.append(line)
        index += 1
    return "\n".join(kept)


def select(items: list[OpenItem], wanted: str = "") -> list[OpenItem]:
    """The items a finish works on, in the description's order.

    ``wanted`` is a comma list of ids; empty means every open item whose kind is not
    (or reads as) ``judgment_call``. An id that is not there, or is resolved already, is
    refused by name.
    """
    names = [name.strip() for name in str(wanted or "").split(",") if name.strip()]
    if not names:
        return [item for item in items if item.status == "open" and item.effective_kind != "judgment_call"]
    known = {item.id: item for item in items}
    missing = [name for name in names if name not in known]
    if missing:
        raise ValueError(f"the pull request has no open item {', '.join(missing)}; its items are "
                         f"{', '.join(known) or '(none)'}")
    done = [name for name in names if known[name].status == "resolved"]
    if done:
        raise ValueError(f"{', '.join(done)} {'is' if len(done) == 1 else 'are'} resolved already")
    return [item for item in items if item.id in names]
