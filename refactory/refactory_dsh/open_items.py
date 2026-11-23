"""A pull request's Open items: one format, written by a run and read by a finish.

A design run gets one fix round and always delivers a pull request. What it leaves is
listed in the description under ``## Open items``, and a finish run (`--refine <PR>
--finish`, `finish.py`) reads that list, works on it, and writes it back. The format is
GitHub markdown a person reads and edits, and stable to parse::

    ## Open items

    - [ ] **qa-1** (qa_finding): the empty-input case prints a traceback
      - tests/test_cli.py::test_empty_input
      - cli.py:42
    - [x] **env-1** (environment): the lab's runner timed out
      - log: `Timeout after 600 s`

One item line each: a task-list box (``[x]`` when resolved), the id in bold, the kind in
brackets, and the summary. Its evidence follows as an indented list, one entry per
line: test ids, ``file:line``, a log excerpt, or the excuse rule a run used.

:func:`render` and :func:`parse` round-trip. :func:`parse` also takes what a person does
to the text: a ticked box (``[x]`` or ``[X]``) is resolved, a comment written between
items or after the section is skipped, an indented line under an item is one more
evidence entry, and an unknown kind is read as ``judgment_call``, the kind a finish
never works on unasked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: The section's heading.
HEADING = "## Open items"

#: What an item can be.
KINDS = ("change_failure", "unfinished_ask", "qa_finding", "judgment_call", "environment")
#: The kinds a finish can fix without a person deciding anything first.
CONCRETE = ("change_failure", "unfinished_ask", "qa_finding")
#: The two states of an item.
STATUSES = ("open", "resolved")

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ITEM = re.compile(r"^[-*+]\s+\[(?P<box>[ xX])\]\s+\*\*(?P<id>[^*\s]+)\*\*\s*"
                   r"\((?P<kind>[A-Za-z_]+)\)\s*:?\s*(?P<summary>.*?)\s*$")
_EVIDENCE = re.compile(r"^(?:\s{2,}|\t)\s*(?:[-*+]\s+)?(?P<text>\S.*?)\s*$")
_SECTION = re.compile(r"^##\s+Open items\s*$", re.I)
#: What ends the section: the next heading of the same or a higher level, or a delivery marker.
_END = re.compile(r"^(?:#{1,2}\s|<!--\s*/?delivery:)")
_COMMENT = re.compile(r"<!--.*?-->", re.S)


def _flat(text: str) -> str:
    return " ".join(str(text).split())


@dataclass(frozen=True)
class OpenItem:
    """One thing a pull request leaves open, or one a later run resolved."""

    id: str
    kind: str
    summary: str
    evidence: tuple[str, ...] = field(default_factory=tuple)
    status: str = "open"

    def __post_init__(self) -> None:
        if not _ID.match(self.id):
            raise ValueError(f"an open item's id is letters, digits, '.', '_' and '-', not {self.id!r}")
        if self.kind not in KINDS:
            raise ValueError(f"an open item's kind is one of {', '.join(KINDS)}, not {self.kind!r}")
        if self.status not in STATUSES:
            raise ValueError(f"an open item's status is open or resolved, not {self.status!r}")
        summary = _flat(self.summary)
        if not summary:
            raise ValueError(f"the open item {self.id} has no summary")
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "evidence", tuple(e for e in (_flat(x) for x in self.evidence) if e))

    @property
    def concrete(self) -> bool:
        """Whether a finish may fix it without asking: a concrete kind, with evidence."""
        return self.kind in CONCRETE and bool(self.evidence)

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "kind": self.kind, "summary": self.summary,
                "evidence": list(self.evidence), "status": self.status}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "OpenItem":
        return cls(id=str(data["id"]), kind=str(data["kind"]), summary=str(data["summary"]),
                   evidence=tuple(str(e) for e in data.get("evidence") or ()),  # type: ignore[union-attr]
                   status=str(data.get("status") or "open"))


def render(items: list[OpenItem] | tuple[OpenItem, ...]) -> str:
    """The ``## Open items`` section for ``items``, ending with a newline."""
    lines = [HEADING, ""]
    for item in items:
        box = "x" if item.status == "resolved" else " "
        lines.append(f"- [{box}] **{item.id}** ({item.kind}): {item.summary}")
        lines += [f"  - {entry}" for entry in item.evidence]
    if not items:
        lines.append("(none)")
    return "\n".join(lines) + "\n"


def _bounds(lines: list[str]) -> list[tuple[int, int]]:
    """Each ``## Open items`` section in ``lines``: (its heading's index, the index after it)."""
    found, index = [], 0
    while index < len(lines):
        if _SECTION.match(lines[index]):
            end = index + 1
            while end < len(lines) and not _END.match(lines[end]):
                end += 1
            found.append((index, end))
            index = end
        else:
            index += 1
    return found


def parse(body: str) -> list[OpenItem]:
    """The items of the first ``## Open items`` section in a pull request's description.

    Empty when there is none. A line that is not an item, or an item's evidence, is a
    person's comment and is skipped; so is an HTML comment. Two items with one id keep
    the first.
    """
    lines = str(body or "").replace("\r\n", "\n").split("\n")
    bounds = _bounds(lines)
    if not bounds:
        return []
    start, end = bounds[0]
    items: list[OpenItem] = []
    current: dict[str, object] | None = None

    def close() -> None:
        if current is None:
            return
        try:
            item = OpenItem(**current)  # type: ignore[arg-type]
        except ValueError:
            return
        if all(other.id != item.id for other in items):
            items.append(item)

    for raw in lines[start + 1:end]:
        line = _COMMENT.sub("", raw).rstrip()
        if raw.strip() and not line.strip():
            continue  # an HTML comment of its own line
        matched = _ITEM.match(line)
        if matched:
            close()
            kind = matched["kind"].lower()
            current = {"id": matched["id"], "kind": kind if kind in KINDS else "judgment_call",
                       "summary": matched["summary"], "evidence": (),
                       "status": "resolved" if matched["box"] in "xX" else "open"}
            continue
        if not line.strip():
            continue
        entry = _EVIDENCE.match(line)
        if entry and current is not None:
            current["evidence"] = (*current["evidence"], entry["text"])  # type: ignore[misc]
        else:
            # Not indented: a person's comment between items ends the one above it.
            close()
            current = None
    close()
    return items


def replace(body: str, items: list[OpenItem] | tuple[OpenItem, ...]) -> str:
    """``body`` with its Open items section(s) replaced by one listing ``items``.

    The section takes the first one's place; any later one is removed. A body without one
    gets it at its end. Everything else in ``body`` stays as it was.
    """
    text = str(body or "")
    lines = text.split("\n")
    bounds = _bounds(lines)
    section = render(items).rstrip("\n").split("\n")
    if not bounds:
        return text.rstrip("\n") + ("\n\n" if text.strip() else "") + "\n".join(section) + "\n"
    kept: list[str] = []
    cursor = 0
    for number, (start, end) in enumerate(bounds):
        kept += lines[cursor:start]
        if number == 0:
            kept += [*section, ""]
        cursor = end
    kept += lines[cursor:]
    return "\n".join(kept)


def without(body: str) -> str:
    """``body`` with every Open items section taken out."""
    lines = str(body or "").split("\n")
    kept, cursor = [], 0
    for start, end in _bounds(lines):
        kept += lines[cursor:start]
        cursor = end
    kept += lines[cursor:]
    return "\n".join(kept)


def select(items: list[OpenItem], wanted: str = "") -> list[OpenItem]:
    """The items a finish works on, in the description's order.

    ``wanted`` is a comma list of ids; empty means every open item that is not a
    ``judgment_call``. An id that is not there, or is resolved already, is refused by name.
    """
    names = [name.strip() for name in str(wanted or "").split(",") if name.strip()]
    if not names:
        return [item for item in items if item.status == "open" and item.kind != "judgment_call"]
    known = {item.id: item for item in items}
    missing = [name for name in names if name not in known]
    if missing:
        raise ValueError(f"the pull request has no open item {', '.join(missing)}; its items are "
                         f"{', '.join(known) or '(none)'}")
    done = [name for name in names if known[name].status == "resolved"]
    if done:
        raise ValueError(f"{', '.join(done)} {'is' if len(done) == 1 else 'are'} resolved already")
    return [item for item in items if item.id in names]
