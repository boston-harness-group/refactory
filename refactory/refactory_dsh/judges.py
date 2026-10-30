"""Judge nodes: a model call that writes a structured verdict, and the reader its gate uses.

A judge reads prose -- the reviews, the evidence map, the correction audit -- and
answers in JSON. The gate after it reads that JSON with ``json.loads`` and never scans
prose for markers, so a review that quotes a finding's format, or a line such as
"-> 1 finding", cannot decide anything. The gate is an ordinary Python node: the walker
calls it in-process and the spine runs it as ``python -m refactory_dsh.run``, so both
runtimes take the same decision from the same file, and a scripted model drives it in
tests.

Design mode has two judges (``refactory_dsh.design``): ``qa_judge`` after the review
fork, and ``overwatch`` before delivery. Their gates are ``design_ops.gate_qa`` and
``design_ops.gate_overwatch``.

What holds a change back is remote-factory design-v2's rule: an ask or approved
correction that was not built, the check failing, or a critical or high defect in product
code. Each must-fix item carries its ``kind`` (``ask``, ``check``, ``defect``, ``test``,
``docs`` or ``style``), its ``severity`` and what it ``breaks``. An item the judge itself
labels ``test``, ``docs`` or ``style`` is a follow-up whatever its severity, and a
``defect`` that is neither critical nor high and breaks nothing is one too, whatever the
verdict says (`blocks`). A correction the person gave and the overwatch finds unmet always
sends the change back.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

#: What the QA judge writes, and the verdicts it may give.
QA_JUDGE = "qa-judge.json"
QA_VERDICTS = ("proceed", "redo")

#: What the overwatch writes in design mode, and the verdicts it may give.
OVERWATCH_VERDICT = "overwatch-verdict.json"
OVERWATCH_VERDICTS = ("pass", "redo")

#: How a judge says a tool or credential it needs is missing: the run stops for a
#: person instead of sending the builder a defect it cannot fix.
UNAVAILABLE = "UNAVAILABLE:"

#: A line of a judge's file that starts with the marker: when the file holds no readable
#: verdict, the run stops, as create's overwatch gate stops on one. A JSON string cannot
#: hold a raw line break, so no line of a JSON verdict starts with it; only prose around
#: or instead of the JSON can.
UNAVAILABLE_LINE = re.compile(r"(?im)^\s*UNAVAILABLE:.*$")

#: The severities a must-fix item may carry, and those that block on their own.
SEVERITIES = ("critical", "high", "medium", "low")
BLOCKING_SEVERITIES = ("critical", "high")

#: What a must-fix item may break; all but ``none`` block a product defect whatever its
#: severity.
BREAKS = ("ask", "correction", "health", "none")

#: What a must-fix item is. ``ask`` (an ask or approved correction not built) and ``check``
#: (the check failing) always block; ``defect`` (a product-code defect) blocks when it is
#: critical or high; the rest are never sent back, whatever their severity.
KINDS = ("ask", "check", "defect", "test", "docs", "style")
BLOCKING_KINDS = ("ask", "check")
FOLLOW_UP_KINDS = ("test", "docs", "style")


class Unreadable(ValueError):
    """The judge's file is missing, is not one JSON object, or is not a verdict."""


def blocks(item: dict[str, str]) -> bool:
    """Whether a must-fix item holds the change back.

    Never when the judge labels it ``test``, ``docs`` or ``style`` (a weak assertion or a
    missing edge-case test is a follow-up, even when it touches a correction). Always when
    it is an unbuilt ``ask`` or a failing ``check``. Otherwise it is a product defect and
    blocks when it is critical or high, or breaks an ask, a correction the person gave, or
    the health check. A missing ``kind`` is a defect."""
    kind = item.get("kind") or "defect"
    if kind in FOLLOW_UP_KINDS:
        return False
    if kind in BLOCKING_KINDS:
        return True
    return item.get("severity") in BLOCKING_SEVERITIES or item.get("breaks") not in ("none", "")


@dataclass(frozen=True)
class Judgment:
    """A judge's verdict, as its gate reads it."""

    verdict: str
    #: Each item: ``source``, ``where``, ``what``, ``why``, ``kind``, ``severity``, ``breaks``.
    must_fix: tuple[dict[str, str], ...] = ()
    follow_ups: tuple[str, ...] = ()
    #: Findings the judge set aside: ``source``, ``finding``, ``why``.
    dismissed: tuple[dict[str, str], ...] = ()
    #: The overwatch's account of each gate correction: ``id``, ``met``, ``evidence``.
    corrections: tuple[dict[str, Any], ...] = field(default=())

    @property
    def blocking(self) -> list[dict[str, str]]:
        """The must-fix items that hold the change back (`blocks`)."""
        return [item for item in self.must_fix if blocks(item)]

    @property
    def unmet(self) -> list[dict[str, Any]]:
        """The corrections the judge says are not met."""
        return [item for item in self.corrections if item.get("met") is False]

    @property
    def unjudged(self) -> list[dict[str, Any]]:
        """The corrections the judge lists without saying whether they are met."""
        return [item for item in self.corrections if item.get("met") is None]


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True)


def _record(value: Any, keys: tuple[str, ...], main: str) -> dict[str, str]:
    """One list entry as a dict of strings; a bare string is its ``main`` field."""
    if isinstance(value, dict):
        return {key: _text(value.get(key)) for key in keys}
    return {key: (_text(value) if key == main else "") for key in keys}


def _must_fix(value: Any) -> dict[str, str]:
    """A must-fix item. A severity it does not give, or one it misspells, is high, and
    a kind it does not give, or one it misspells, is a defect: a missing label never lets a
    defect through. What it breaks defaults to nothing."""
    item = _record(value, ("source", "where", "what", "why", "kind", "severity", "breaks"),
                   "what")
    kind = item["kind"].lower()
    item["kind"] = kind if kind in KINDS else "defect"
    severity = item["severity"].lower()
    item["severity"] = severity if severity in SEVERITIES else "high"
    breaks = item["breaks"].lower()
    item["breaks"] = breaks if breaks in BREAKS else ("none" if not breaks else breaks)
    return item


def _follow_up(value: Any) -> str:
    if isinstance(value, dict):
        parts = [_text(value.get(key)) for key in ("what", "where", "why", "finding")]
        return " — ".join(part for part in parts if part) or _text(value)
    return _text(value)


def _met(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "yes", "met"):
        return True
    if isinstance(value, str) and value.strip().lower() in ("false", "no", "unmet", "not met"):
        return False
    return None


def _objects(text: str) -> list[Any]:
    """Every top-level JSON object in ``text``, bare, fenced or among prose, in order.

    A brace that does not start a JSON object (prose such as ``{x}``) is skipped, and an
    object's own nested objects are part of it, not further objects.
    """
    decoder = json.JSONDecoder()
    found: list[Any] = []
    index = text.find("{")
    while index >= 0:
        try:
            value, end = decoder.raw_decode(text, index)
        except ValueError:
            index = text.find("{", index + 1)
            continue
        found.append(value)
        index = text.find("{", end)
    return found


def parse(text: str, verdicts: tuple[str, ...], *, name: str = "the judge's file") -> Judgment:
    """Read a judge's JSON verdict, or raise :class:`Unreadable` saying why.

    The file's only JSON content must be one JSON object: on its own, in a code fence,
    or among prose. Two or more objects are unreadable, because which is the verdict
    would be a guess. A ``redo`` must name what to fix (a ``must_fix`` item, or for the
    overwatch a correction it found unmet or could not judge): a redo with nothing in it
    gives the builder nothing to do.
    """
    if not text.strip():
        raise Unreadable(f"{name} is empty")
    objects = _objects(text)
    if not objects:
        raise Unreadable(f"{name} holds no JSON object")
    if len(objects) > 1:
        raise Unreadable(f"{name} holds {len(objects)} JSON objects; it must hold exactly one")
    data = objects[0]
    verdict = data.get("verdict")
    if not isinstance(verdict, str) or verdict.strip().lower() not in verdicts:
        raise Unreadable(f"{name} has verdict {verdict!r}; expected one of {', '.join(verdicts)}")
    lists = {}
    for key in ("must_fix", "follow_ups", "dismissed", "corrections"):
        value = data.get(key, [])
        if value is None:
            value = []
        if not isinstance(value, list):
            raise Unreadable(f"{name}: {key} is not a list")
        lists[key] = value
    corrections = []
    for item in lists["corrections"]:
        if not isinstance(item, dict):
            raise Unreadable(f"{name}: each corrections entry must be an object")
        corrections.append({"id": _text(item.get("id")), "met": _met(item.get("met")),
                            "evidence": _text(item.get("evidence"))})
    judgment = Judgment(
        verdict=verdict.strip().lower(),
        must_fix=tuple(_must_fix(item) for item in lists["must_fix"]),
        follow_ups=tuple(text for text in (_follow_up(item) for item in lists["follow_ups"]) if text),
        dismissed=tuple(_record(item, ("source", "finding", "why"), "finding")
                        for item in lists["dismissed"]),
        corrections=tuple(corrections),
    )
    if (judgment.verdict == "redo" and not judgment.must_fix and not judgment.unmet
            and not judgment.unjudged):
        raise Unreadable(f"{name} says redo but names nothing to fix")
    return judgment


def read(ctx: Any, name: str, verdicts: tuple[str, ...]) -> Judgment:
    """The judgment in the run directory's ``name``; :class:`Unreadable` when there is none."""
    return parse(text_of(ctx, name), verdicts, name=name)


def text_of(ctx: Any, name: str) -> str:
    """A judge's file as text; :class:`Unreadable` when it is missing or not text."""
    if not ctx.exists(name):
        raise Unreadable(f"{name} was not written")
    try:
        return ctx.read_text(name)
    except (OSError, UnicodeDecodeError) as exc:
        raise Unreadable(f"{name} could not be read ({exc})") from None


def excerpt(ctx: Any, name: str, limit: int = 300) -> str:
    """The start of a judge's file on one line, for a note that says it was unreadable."""
    try:
        text = " ".join(ctx.read_text(name).split()) if ctx.exists(name) else ""
    except (OSError, UnicodeDecodeError):
        return ""
    return text if len(text) <= limit else text[:limit] + "…"


def unavailable(judgment: Judgment) -> str:
    """The first must-fix item saying a dependency is unavailable, or ``""``."""
    for item in judgment.must_fix:
        if item.get("what", "").startswith(UNAVAILABLE):
            return item["what"]
    return ""


def must_fix_line(item: dict[str, str]) -> str:
    """One must-fix item as a line of feedback: where, what, and why it matters."""
    head = f"[{item['source']}] " if item.get("source") else ""
    where = f"{item['where']}: " if item.get("where") else ""
    label = [item.get("kind", "") if item.get("kind") not in ("", "defect", None) else "",
             item.get("severity", "")]
    if item.get("breaks") not in ("", "none", None):
        label.append(f"breaks {item['breaks']}")
    tag = f" ({', '.join(part for part in label if part)})" if any(label) else ""
    why = f" (why: {item['why']})" if item.get("why") else ""
    return f"{head}{where}{item.get('what', '')}{why}{tag}".strip()


def correction_line(item: dict[str, Any]) -> str:
    """One correction as the overwatch judged it, with its evidence."""
    met = {True: "met", False: "NOT met", None: "not judged"}[item.get("met")]
    evidence = f" — {item['evidence']}" if item.get("evidence") else ""
    return f"{item.get('id') or '(no id)'}: {met}{evidence}"


def feedback(judgment: Judgment, judge: str, *, passed: bool = False) -> str:
    """What the builder is told when ``judge`` sends the change back.

    The must-fix items, each with where, why and severity; every correction the judge
    found unmet or could not judge, with its evidence; then the follow-ups, which the
    builder may fix if it is cheap and the pull request lists otherwise. ``passed`` is a
    pass the gate turned into a redo because it leaves a correction unmet.
    """
    lines = [f"The {judge} passed the change but found corrections the person gave not met. "
             "Fix each of these:" if passed else
             f"The {judge} sent the change back. Fix each of these:"]
    lines += [f"- {must_fix_line(item)}" for item in (judgment.must_fix if not passed else ())]
    lines += [f"- correction {correction_line(item)}" for item in judgment.unmet]
    lines += [f"- correction {correction_line(item)}" for item in
              (judgment.unjudged if not passed else ())]
    leftovers = [*judgment.follow_ups,
                 *(must_fix_line(item) for item in (judgment.must_fix if passed else ()))]
    if leftovers:
        lines.append("Follow-ups (fix them only if it is cheap; otherwise the pull request "
                     "lists them):")
        lines += [f"- {item}" for item in leftovers]
    return "\n".join(lines)
