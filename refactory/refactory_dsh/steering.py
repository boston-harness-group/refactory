"""Steering a run while *using* it: the intent ledger, and the transform that adds it.

Two ways to change a run, and they are different acts on different things:

- **Structure** is the outer loop's job. It searches a mode's graph and knobs, and a
  result is adopted as a new mode (see :mod:`refactory_dsh.adopt`).
- **Steering** is the person's, while using a mode. You add context, correct it, or
  forbid something. **The graph does not change.** What changes is what the agents
  know, which is what steering means during use.

**What was here before, and why it is back.** The previous system had a *User Intent
Ledger* at ``.factory/strategy/user-intent.md``: seeded once by an ``init_user_intent``
node, append-only, described as "the user's original idea **and feedback**", read by
research, strategy, builder review, QA and overwatch, and checked by a gate that work
must be "traceable to user intent". It was thrown away with the rest of that system,
and ``default_pack/create``'s intent became a static class attribute.

It was also already broken, in a way worth not repeating. ``2026-09-12-create-v2-run.md``
records BUG-05: ``REFACTORY_INTENT`` never arrived, so the ledger held the literal
default ``"Build something useful."`` -- and the fidelity gate, having nothing to check
against, checked nothing. The user's real words survived only because the CEO quoted
them by hand into subagent briefs. So :meth:`Ledger.seed` refuses a default rather than
accepting one, and :func:`fidelity` checks the precondition rather than pretending to
check the output.

**The layer is inert when unused.** :func:`annotate` returns the workflow it was given
when the ledger is empty, so a run nobody speaks to is the run it would have been.
Without that, the interactive and headless paths could disagree about the same mode and
there would be no way to tell which was wrong.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator, Mapping

#: Where the ledger lives inside a run directory.
LEDGER = "intent-ledger.md"

#: The kinds an entry can be. A constraint is not information: it has to be
#: *respected*, not merely known, which is why it is a separate kind rather than a
#: note with a different tone.
KINDS: tuple[str, ...] = ("info", "correction", "constraint")

#: The scopes a note can have. A product note is an auditable requirement when its
#: kind is correction or constraint. A process note is about the run itself and is
#: never an auditable requirement because the audit cannot cite itself as evidence.
SCOPES: tuple[str, ...] = ("product", "process")

#: Who said a note: the person (``user-relayed``) or the CEO on its own
#: (``ceo``). Only user-relayed entries are auditable requirements; CEO
#: entries are context the agent sees but the audit does not check against.
SOURCES: tuple[str, ...] = ("user-relayed", "ceo")

#: Bookkeeping entry kinds written by the gate protocol. Stored in the
#: ledger for routing, but filtered out of rendered prompts.
BOOKKEEPING_KINDS: frozenset[str] = frozenset({"gate-visit", "verdict", "verdict-use"})

#: The entry kind a gate verdict is recorded under. Separate from the three the
#: person writes, because a verdict is a decision the run routed on, not something
#: the person said about it, and a resumed run has to tell them apart.
VERDICT_KIND = "verdict"

#: How a piece of work names the ledger entry it serves. A line, so it is readable
#: in the artifact and findable without parsing the prose around it.
CITATION = "Serves:"

#: The placeholder the previous system fell back to. Refused rather than accepted:
#: accepting it is what made the fidelity check vacuous, and a caller that reintroduces
#: it should fail here rather than produce a run nobody can account for.
PLACEHOLDER = "Build something useful."


class SteeringError(ValueError):
    """The steering layer cannot be built or trusted. Always names what is wrong."""


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _escape_body(text: str) -> str:
    """Escape entry-boundary markers so one write is always one entry.

    Splits on ``splitlines(keepends=True)`` — the same split ``parse_entries``
    uses — so the writer and parser agree on every line boundary Python
    recognises.  A line whose content (after stripping the line ending) starts
    with zero-or-more backslashes followed by ``## [`` gains one leading
    backslash (byte-stuffing).
    """
    if not text:
        return text
    parts = text.splitlines(True)
    out: list[str] = []
    for part in parts:
        content = part
        # Strip exactly the terminator splitlines(keepends=True) left.
        for ending in ("\r\n", "\r", "\n", "\x0b", "\x0c", "\x1c", "\x1d",
                       "\x1e", "\x85", "\u2028", "\u2029"):
            if content.endswith(ending):
                content = content[: -len(ending)]
                break
        if content.lstrip("\\").startswith("## ["):
            out.append("\\" + part)
        else:
            out.append(part)
    return "".join(out)


def _unescape_line(line: str) -> str:
    """Remove one level of byte-stuffing from a body line.

    Called inside ``parse_entries`` on each body line before appending, so all
    consumers see the original text.
    """
    if line.startswith("\\") and line.lstrip("\\").startswith("## ["):
        return line[1:]
    return line


class Ledger:
    """The run's record of what was asked for, and everything said since.

    Append-only, timestamped, and written into the run directory so it travels with
    the run rather than with the session that started it. A ledger kept in the session
    would be lost on resume. Historical answers remain audit evidence; only an
    eligible pending answer or explicit upfront approval can authorize routing.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.path = self.root / LEDGER

    # ── writing ────────────────────────────────────────────────────────────
    @classmethod
    def seed(cls, root: Path, request: str) -> "Ledger":
        """Open a ledger with the request that started the run.

        Refuses a blank request and refuses the legacy placeholder. Both are the
        BUG-05 failure: a ledger holding a default is one a fidelity check cannot
        check, and it looks identical to a ledger that was filled in.
        """
        text = (request or "").strip()
        if not text:
            raise SteeringError(
                "a run that can be steered needs the request it is for; seeding with "
                "an empty intent is what makes the fidelity gate vacuous"
            )
        if text == PLACEHOLDER:
            raise SteeringError(
                f"{PLACEHOLDER!r} is the placeholder the previous system defaulted to. "
                "Pass what was actually asked for: a ledger holding a default is "
                "indistinguishable from one that was filled in, which is the bug this "
                "refuses to reintroduce."
            )
        ledger = cls(root)
        if ledger.path.exists() and ledger.path.stat().st_size > 0:
            raise SteeringError(
                f"{ledger.path} already exists; a ledger is seeded once and appended "
                "to thereafter, so seeding it again would put a second origin in it"
            )
        ledger.root.mkdir(parents=True, exist_ok=True)
        ledger.path.write_text(f"# User Intent Ledger\n\n## [{_now()}] Initial Idea\n{_escape_body(text)}\n")
        return ledger

    def note(self, kind: str, text: str, *, scope: str = "product",
             source: str = "user-relayed") -> dict[str, Any]:
        """Append one thing said about this run. Returns the entry as it was recorded.

        ``scope`` is ``'product'`` (default) or ``'process'``. A process note is
        about the run's own behaviour (e.g. "add a row to the audit table") and is
        never an auditable requirement, because the audit cannot cite itself as
        evidence for a directive about its own behaviour.

        ``source`` is ``'user-relayed'`` (default) or ``'ceo'``. CEO notes render
        under their own heading and are never auditable requirements; only the
        person's words (``user-relayed``) are.
        """
        if kind not in KINDS:
            raise SteeringError(f"unknown note kind {kind!r}; known: {list(KINDS)}")
        if scope not in SCOPES:
            raise SteeringError(f"unknown note scope {scope!r}; known: {list(SCOPES)}")
        if source not in SOURCES:
            raise SteeringError(f"unknown note source {source!r}; known: {list(SOURCES)}")
        body = (text or "").strip()
        if not body:
            raise SteeringError("an empty note records nothing and dilutes the ledger")
        entry = {"at": _now(), "kind": kind, "text": body, "scope": scope, "source": source}
        self.root.mkdir(parents=True, exist_ok=True)
        scope_suffix = " (process)" if scope == "process" else ""
        source_suffix = " (ceo)" if source == "ceo" else ""
        with self.path.open("a") as handle:
            handle.write(f"\n## [{entry['at']}] {kind}{scope_suffix}{source_suffix}\n{_escape_body(body)}\n")
        return entry

    def verdict(
        self, gate: str, value: str, *, approval: bool = False,
        review: str | None = None, answered: bool = False,
    ) -> dict[str, Any]:
        """Record a one-use answer for the current review, or explicit upfront approval.

        Interactive answers are already used. A public steer answer is pending for
        the exact declared input files, then consumed once. Only ``approval=True``
        is standing authorization, as explicitly supplied by ``ceo --approve``.
        """
        gate, value = (gate or "").strip(), (value or "").strip()
        if not gate or not value or len(gate.splitlines()) != 1 or len(value.splitlines()) != 1:
            raise SteeringError("a verdict needs a single-line gate and nonempty value")
        if approval and answered:
            raise SteeringError("an interactive answer is not standing approval")
        if review is None and not approval:
            reads = self.gate_reads(gate)
            if reads is None:
                raise SteeringError(
                    f"cannot scope an answer for gate {gate!r}: no unique user-gate declaration; "
                    "open or run the mode to that gate before steering, or use ceo --approve "
                    "for explicit upfront authorization"
                )
            review = self.review(gate, reads)
        entry = {
            "at": _now(), "kind": VERDICT_KIND, "id": uuid.uuid4().hex,
            "gate": gate, "verdict": value,
            "scope": "run" if approval else review,
            "source": "upfront" if approval else "interactive" if answered else "steer",
            "consumed": "yes" if answered else "no",
        }
        self._decision_entry(entry)
        return entry

    def _decision_entry(self, entry: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        body = "".join(f"{key}: {value}\n" for key, value in entry.items() if key not in ("at", "kind"))
        with self.path.open("a") as handle:
            handle.write(f"\n## [{entry['at']}] {entry['kind']}\n{_escape_body(body)}")

    @staticmethod
    def _fields(entry: dict[str, Any]) -> dict[str, str]:
        return {key.strip(): value.strip() for line in entry["text"].splitlines()
                for key, separator, value in [line.partition(":")] if separator}

    def verdicts(self) -> dict[str, str]:
        """Historical latest answers for display, never authorization to replay them."""
        found: dict[str, str] = {}
        for entry in self.entries():
            fields = self._fields(entry)
            if entry["kind"] == VERDICT_KIND and fields.get("gate") and fields.get("verdict"):
                found[fields["gate"]] = fields["verdict"]
        return found

    def gate_reads(self, gate: str) -> list[str] | None:
        """Resolve steering from the actual latest visit, or a unique lowered gate."""
        # A real visit is authoritative even when an old lowered graph remains.
        # Walker runs need this declaration because Mode.run does not lower files.
        for entry in reversed(self.entries()):
            fields = self._fields(entry)
            if entry["kind"] == "gate-visit" and fields.get("gate") == gate:
                return json.loads(fields["reads"])
        matches = []
        for path in sorted((self.root / "graphs").glob("*.graph.json")):
            graph = json.loads(path.read_text())
            node = graph.get("nodes", {}).get(gate)
            if isinstance(node, dict) and node.get("evaluator_type") == "user":
                matches.append(list(node.get("reads") or []))
        return matches[0] if len(matches) == 1 else None

    def visit(self, gate: str, reads: list[str] | tuple[str, ...]) -> str:
        """Record the current declared review inputs for public steering to discover.

        In name order, as :meth:`review` fingerprints them and the lowered graph keeps
        them, so a walked gate and a driven one record the same visit."""
        review = self.review(gate, reads)
        self._decision_entry({"at": _now(), "kind": "gate-visit", "gate": gate,
                              "review": review, "reads": json.dumps(sorted(reads))})
        return review

    def review(self, gate: str, reads: list[str] | tuple[str, ...]) -> str:
        """Fingerprint the declared review inputs, including missing files."""
        files: dict[str, Any] = {}
        for rel in sorted(reads):
            path = self.root / rel
            if path.is_dir():
                files[rel] = {
                    str(child.relative_to(path)): hashlib.sha256(child.read_bytes()).hexdigest()
                    for child in sorted(path.rglob("*")) if child.is_file()
                }
            else:
                files[rel] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        body = json.dumps({"gate": gate, "files": files}, sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()

    def take_verdict(self, gate: str, review: str, labels: list[str]) -> str:
        """Consume the newest eligible answer; old/used/changed answers never replay.

        Legacy records lack an explicit scope and cannot prove current approval.
        They remain visible in the ledger, but require the person to answer again.
        A mismatching pending answer expires so reverting a file cannot revive it.

        Verdicts are lowercased. ``halt`` is always accepted at a user gate even
        when it is not in ``labels``, because every user gate offers halt.
        """
        entries = self.entries()
        used = {self._fields(e).get("id") for e in entries if e["kind"] == "verdict-use"}
        latest = next((self._fields(e) for e in reversed(entries)
                       if e["kind"] == VERDICT_KIND and self._fields(e).get("gate") == gate), {})
        value = (latest.get("verdict") or "").lower()
        if latest.get("scope") == "run" and latest.get("source") == "upfront":
            if value not in [l.lower() for l in labels] and value != "halt":
                raise SteeringError(f"upfront verdict {value!r} is not accepted by gate {gate!r}")
            return value
        decision = latest.get("id")
        if not decision or decision in used or latest.get("consumed") == "yes":
            return ""
        valid = latest.get("scope") == review and (
            value in [l.lower() for l in labels] or value == "halt"
        )
        self._decision_entry({"at": _now(), "kind": "verdict-use", "id": decision,
                              "gate": gate, "result": "used" if valid else "expired"})
        self._mark_consumed(decision)
        return value if valid else ""

    def _mark_consumed(self, verdict_id: str) -> None:
        """Update the original verdict entry's consumed field from 'no' to 'yes'."""
        if not self.path.is_file():
            return
        text = self.path.read_text()
        lines = text.splitlines(keepends=True)
        found_id = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == f"id: {verdict_id}":
                found_id = True
            elif found_id and stripped == "consumed: no":
                lines[i] = line.replace("consumed: no", "consumed: yes")
                break
            elif found_id and stripped.startswith("## ["):
                break  # Moved past the entry without finding consumed: no
        if found_id:
            self.path.write_text("".join(lines))

    # ── reading ────────────────────────────────────────────────────────────
    def exists(self) -> bool:
        return self.path.exists() and self.path.stat().st_size > 0

    def entries(self) -> list[dict[str, Any]]:
        """Every entry, parsed back out: timestamps and kinds, not just the text."""
        if not self.exists():
            return []
        return parse_entries(self.path.read_text())

    def constraints(self) -> list[str]:
        return [entry["text"] for entry in self.entries() if entry["kind"] == "constraint"]

    def render(self, *, entries: list[dict[str, Any]] | None = None) -> str:
        """The prompt block, from current entries or one atomic dispatch snapshot."""
        return render_entries(self.entries() if entries is None else entries)


def parse_entries(text: str) -> list[dict[str, Any]]:
    """Parse one ledger snapshot without reading a changing file again."""
    found: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("## ["):
            if current is not None:
                current["text"] = "\n".join(body).strip()
                found.append(current)
            stamp, _, kind = line[4:].partition("] ")
            kind = kind.strip()
            # Parse suffixes right-to-left: writing order is
            # {kind}{scope_suffix}{source_suffix}, e.g. "correction (process) (ceo)",
            # so strip the rightmost suffix (ceo) first, then (process).
            source = "user-relayed"
            if kind.endswith(" (ceo)"):
                kind = kind[:-len(" (ceo)")].strip()
                source = "ceo"
            scope = "product"
            if kind.endswith(" (process)"):
                kind = kind[:-len(" (process)")].strip()
                scope = "process"
            current = {"at": stamp, "kind": kind, "text": "", "scope": scope, "source": source}
            body = []
        elif current is not None:
            body.append(_unescape_line(line))
    if current is not None:
        current["text"] = "\n".join(body).strip()
        found.append(current)
    return found


def render_entries(entries: list[dict[str, Any]]) -> str:
    """Render supplied entries; empty input leaves an unsteered prompt unchanged.

    Bookkeeping kinds (gate-visit, verdict, verdict-use) are stored in the
    ledger for routing but filtered out of rendered prompts (E12).

    Entries are partitioned by source (C5): user-relayed entries appear under
    "What the user has said" and are auditable requirements; CEO entries appear
    under "CEO observations" with an explicit non-auditable disclaimer.
    Legacy entries without a source annotation default to user-relayed.
    """
    if not entries:
        return ""
    renderable = [e for e in entries if e.get("kind") not in BOOKKEEPING_KINDS]
    if not renderable:
        return ""
    user_entries = [e for e in renderable if e.get("source", "user-relayed") != "ceo"]
    ceo_entries = [e for e in renderable if e.get("source", "user-relayed") == "ceo"]
    sections: list[str] = []
    if user_entries:
        lines = [
            "## What the user has said", "",
            "This is the run's intent ledger: what was asked for, and everything said",
            "since. Entries marked `constraint` must be respected, not merely known.", "",
        ]
        for entry in user_entries:
            lines.append(f"- **[{entry['at']}] [{entry['kind']}]** {entry['text']}")
        sections.append("\n".join(lines))
    if ceo_entries:
        lines = [
            "## CEO observations", "",
            "These are the CEO's own observations. They are context for the agent,",
            "not auditable requirements.", "",
        ]
        for entry in ceo_entries:
            lines.append(f"- **[{entry['at']}] [{entry['kind']}]** {entry['text']}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _is_gate_node(n: Any) -> bool:
    """True for gate nodes in both pre-lowered workflow and IR.

    Pre-lowered: ``kind == "gate"``.
    IR (GateNode from ir.py ~124-136): class name ``GateNode``, no ``kind``.
    """
    return getattr(n, 'kind', None) == 'gate' or type(n).__name__ == 'GateNode'


def _is_user_gate(n: Any) -> bool:
    """True for user gates across both representations.

    Pre-lowered: ``gate_kind == "user"``.
    IR: ``evaluator_type == "user"`` (no ``gate_kind``).
    """
    return _is_gate_node(n) and (
        getattr(n, 'gate_kind', None) == 'user'
        or getattr(n, 'evaluator_type', None) == 'user'
    )


def _is_agent_node(node: Any) -> bool:
    """A node that makes a model call, on either side of the lowering.

    The transform runs on the lowered IR (``AgentNode``) and on the walker's own
    graph (``nn.Node`` with ``kind == "agent"``), so it duck-types instead of
    importing either: the two runtimes must carry the same ledger, and a mode's
    author must not be the thing that decides whether they do.
    """
    return node.__class__.__name__ == "AgentNode" or getattr(node, "kind", "") == "agent"


STEERING_START = "<!-- refactory steering context -->"
STEERING_END = "<!-- end refactory steering context -->"


#: Where the ledger being folded will live once its run is installed, while a new lowering
#: is staged (`installing_to`); unset otherwise, when the ledger read is the run's own.
_INSTALLED_ROOT: ContextVar[Path | None] = ContextVar("steering_installed_root", default=None)


@contextmanager
def installing_to(root: Path) -> Iterator[None]:
    """Fold, inside this block, as for the run directory ``root`` (dm-229).

    `invocations.prepare` stages a new lowering in a temporary directory and folds the
    ledger from there, then moves the ledger into ``root`` and deletes the stage. The
    entries are still read from the stage; the prompts cite the ledger where it will be.
    """
    token = _INSTALLED_ROOT.set(Path(root))
    try:
        yield
    finally:
        _INSTALLED_ROOT.reset(token)


def _citation_block(ledger: Ledger) -> str:
    """Keep live steering instructions separate from the node's output format."""
    installed = _INSTALLED_ROOT.get()
    path = Ledger(installed).path if installed is not None else ledger.path
    return (
        "\n\nRead the live ledger at `" + str(path) + "` before acting. "
        "Respect its constraints while preserving the node's required output format. "
        "Do not append steering citations or other prose to code, JSON, or another "
        "declared artifact. The runtime records supplied ledger context and artifact "
        "hashes separately; that receipt does not certify semantic compliance."
    )


def prompt_with_ledger(prompt: str, ledger: Ledger, *, entries: list[dict[str, Any]] | None = None) -> str:
    """Replace only this layer's own block with current context at dispatch."""
    start = prompt.find(STEERING_START)
    end = prompt.find(STEERING_END, start)
    if start >= 0 and end >= start:
        prompt = prompt[:start].rstrip() + prompt[end + len(STEERING_END):]
    block = ledger.render(entries=entries)
    if not block:
        return prompt
    return f"{prompt}\n\n{STEERING_START}\n{block}{_citation_block(ledger)}\n{STEERING_END}"


def annotate(workflow: Any, ledger: Ledger) -> Any:
    """Fold the ledger into every agent node. A no-op when the ledger is empty.

    Every agent node, not one the author remembered to wire: a mode declares *what* it
    does, and the steering layer is not its business, so a mode author cannot be the
    thing that makes the layer work. Applied to the lowered workflow rather than to the
    mode, which is what keeps the mode Pythonic and unchanged.
    """
    block = ledger.render()
    if not block:
        return workflow
    for node in workflow.nodes.values():
        if not _is_agent_node(node):
            continue
        existing = getattr(node, "prompt_template", "") or ""
        node.prompt_template = prompt_with_ledger(existing, ledger)
    return workflow


def design_probe_steps(workflow: Any) -> dict[str, str]:
    """Find the design-writing node from graph declarations.

    Pattern: an agent writer whose outputs include a file F, a gate with a
    'reloop' conditioned edge back to the writer, another node that reads F and
    leads into that gate (the check that verifies the writer's claims, including
    the probe ids it cites), and a downstream user gate that also reads F.

    Works on both the pre-lowered workflow (``build()``) and the IR
    (``to_workflow()``), using ``_is_gate_node`` / ``_is_user_gate``.

    Returns a dict mapping node_id -> 'writer'.
    """
    nodes = workflow.nodes  # dict[str, Node]
    edges = workflow.edges  # list of edge objects

    def _edge_condition(edge: Any) -> str:
        """Normalise edge condition to a plain string across both representations."""
        cond = getattr(edge, 'condition', None)
        if cond is None:
            return ""
        # IR edges may store VerdictType enum values.
        return cond.value if hasattr(cond, 'value') else str(cond)

    # Build edge index: from -> list[(to, condition)]
    outgoing: dict[str, list[tuple[str, str]]] = {}
    for edge in edges:
        src = getattr(edge, 'source', None)
        dst = getattr(edge, 'target', None)
        if src is None or dst is None:
            continue
        cond = _edge_condition(edge)
        outgoing.setdefault(src, []).append((dst, cond))

    result: dict[str, str] = {}

    for nid, node in nodes.items():
        if not _is_agent_node(node):
            continue
        writes = set(getattr(node, 'writes', ()) or ())
        if not writes:
            continue
        user_reads: set[str] = set()
        for ug_node in nodes.values():
            if _is_user_gate(ug_node):
                user_reads |= set(getattr(ug_node, 'reads', ()) or ())
        shown = writes & user_reads
        if not shown:
            continue

        for gate_id, gate_node in nodes.items():
            if not _is_gate_node(gate_node) or gate_id == nid:
                continue
            if not any(dst == nid and cond == 'reloop' for dst, cond in outgoing.get(gate_id, [])):
                continue
            # A check between the writer and its gate reads what the writer wrote.
            checked = any(
                check_id not in (nid, gate_id)
                and shown & set(getattr(check_node, 'reads', ()) or ())
                and any(dst == gate_id for dst, _ in outgoing.get(check_id, []))
                for check_id, check_node in nodes.items()
            )
            if checked:
                result[nid] = 'writer'

    return result


def for_run(workflow: Any, workdir: Path, *, omit_steering: bool | None = None) -> Any:
    """Fold the run directory's ledger into a workflow about to run.

    Called by **both** runtimes.  A mode lowered for DSH and one executed by the
    walker have to carry the same ledger; annotating in one path only is a
    difference between the runtimes that nothing would report.
    """
    from refactory_dsh.run_preparation import saved_run_config
    omit = saved_run_config(workdir).get("omit_steering", False) if omit_steering is None else omit_steering
    if not omit:
        workflow = annotate(workflow, Ledger(workdir))
    # Inject the design probe fact reference into the structurally identified writer.
    for nid in design_probe_steps(workflow):
        node = workflow.nodes[nid]
        existing = getattr(node, "prompt_template", "") or ""
        if "{{design_probe_instruction}}" not in existing:
            node.prompt_template = existing + "\n\n{{design_probe_instruction}}"
    return workflow


def citations(text: str) -> list[str]:
    """The ledger entries a piece of work says it serves, in the order it names them."""
    found: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped.lower().startswith(CITATION.lower()):
            continue
        rest = stripped[len(CITATION):].strip()
        for stamp in rest.replace(",", " ").split():
            stamp = stamp.strip("`[]()")
            if stamp and stamp not in found:
                found.append(stamp)
    return found


def chain(ledger: Ledger, outputs: Mapping[str, str], *,
          deliveries: Mapping[str, Any] | None = None,
          completed_nodes: frozenset[str] | set[str] | None = None) -> list[str]:
    """Problems with the chain from the ledger to the work, mechanically.

    Current runtime receipts establish delivered context and artifact identity.
    Legacy outputs can still cite entries inline. Neither establishes that the
    model read, understood, or satisfied the intent; that remains a judgement.

    ``completed_nodes`` is the set of node ids the spine records as completed
    (from ``spine-status.json``).  When supplied, a node that is not in the
    set has not run and is skipped — even if it ran and wrote nothing, a
    completed node is still judged.  When absent, the legacy heuristic (no
    delivery and empty output text) is used.
    """
    known = {entry["at"] for entry in ledger.entries()}
    problems: list[str] = []
    for node_id in sorted(outputs):
        delivery = (deliveries or {}).get(node_id)
        # E9: Decide "has not run" from the spine's record of completed nodes
        # when available.  A node that ran and wrote nothing must still be
        # judged; only a node that has not run yet is skipped.
        if completed_nodes is not None:
            if node_id not in completed_nodes:
                continue
        else:
            # Legacy fallback: no delivery and no output files → unrun.
            if delivery is None and not outputs[node_id].strip():
                continue
        if delivery is not None:
            # E9: A pending delivery has not completed yet — skip it.
            if delivery.get("status") == "pending":
                continue
            if delivery.get("status") != "delivered":
                problems.append(f"node {node_id!r} has no successful artifact delivery receipt")
                continue
            if delivery.get("missing_now"):
                problems.append(f"node {node_id!r} delivered outputs that are now missing: "
                                + ", ".join(delivery["missing_now"]))
            cited = [entry["at"] for entry in delivery.get("entries", [])]
        else:
            # Existing runs can still establish their historical inline chain.
            cited = citations(outputs[node_id])
        if not cited:
            problems.append(
                f"node {node_id!r} produced work that cites no ledger entry, so it is "
                "not traceable to what was asked for"
            )
            continue
        for stamp in cited:
            if stamp not in known:
                problems.append(
                    f"node {node_id!r} cites {stamp!r}, which is not in the ledger, so "
                    "the citation dangles"
                )
    return problems


def outputs_in(run_root: Path) -> dict[str, str]:
    """What each agent node wrote, read back so the citation chain can be checked.

    Read from the declarations the lowering left beside the run, so it works for a
    run either runtime executed and does not need the state. A file the node never
    wrote is an empty output, which the chain reports as a node that cited nothing.
    """
    from refactory_dsh import lowering

    from refactory_dsh.agent_delivery import receipts_in

    declared = lowering.read_declarations(Path(run_root))
    # Direct Mode.evaluate() need not lower a graph. Its runtime receipts carry
    # the same declarations without making steering depend on a second lowering.
    for node_id, receipt in receipts_in(Path(run_root)).items():
        declared.setdefault(node_id, {"kind": "agent", "writes": receipt.get("writes", [])})
    outputs: dict[str, str] = {}
    for node_id, entry in declared.items():
        if not isinstance(entry, dict) or entry.get("kind") != "agent":
            continue
        text = ""
        for name in entry.get("writes") or ():
            path = Path(run_root) / str(name)
            if path.is_file():
                text += "\n" + path.read_text(errors="replace")
        outputs[str(node_id)] = text
    return outputs


def fidelity(ledger: Ledger, outputs: Mapping[str, str] | None = None, *,
             deliveries: Mapping[str, Any] | None = None,
             completed_nodes: frozenset[str] | set[str] | None = None) -> list[str]:
    """What is wrong with a ledger a run is about to be judged against.

    Says which half it is, because the halves differ in cost and in what they can
    conclude:

    - **The precondition**, always. A ledger that is missing, empty, or holding the
      placeholder. Deterministic, free. BUG-05 was this half being broken while the
      gate reported a pass.
    - **The chain**, when ``outputs`` is given. Every agent output must have a delivery receipt for known
      ledger entries, or legacy inline citations (``chain``). Deterministic, free.
    - **The judgement** -- whether the work *satisfies* the intent -- is **not here**.
      It needs a model to read the work against the ledger, and checking citations
      does not substitute for it. That limit is in `architecture.md` §12.
    """
    problems: list[str] = []
    if not ledger.exists():
        return [f"no intent ledger at {ledger.path}: the run has nothing to be judged against"]
    entries = ledger.entries()
    if not entries:
        return ["the intent ledger has no entries"]
    origin = entries[0]["text"].strip()
    if not origin:
        problems.append("the ledger's origin is empty")
    if origin == PLACEHOLDER:
        problems.append(f"the ledger's origin is the placeholder {PLACEHOLDER!r}")
    if outputs is not None:
        problems.extend(chain(ledger, outputs, deliveries=deliveries,
                              completed_nodes=completed_nodes))
    return problems


def interactive(
    state: dict[str, Any],
    workdir: Path,
    *,
    request: str = "",
    name: str = "",
    description: str = "",
) -> tuple[Any, Ledger]:
    """Lower a mode for a person to *use*, with the steering layer attached.

    The one entry point, and a named function over the IR rather than a branch inside
    :func:`~refactory_dsh.lowering.to_workflow`, so that "does the steering layer
    change the result" is a question one test can ask, and so a mode is not lowered
    differently depending on who is going to run it.
    """
    from refactory_dsh import lowering

    workflow = lowering.to_workflow(state, name=name, description=description)
    ledger = Ledger(workdir)
    if request:
        ledger = Ledger.seed(workdir, request)
    return annotate(workflow, ledger), ledger


#: Maps a verdict label to the note kind ``console._steer`` should use when the
#: person gives ``--verdict`` and ``--note`` together with no explicit ``--kind``.
NOTE_KINDS: dict[str, str] = {"reloop": "correction", "proceed": "constraint"}


def strip_steering(text: str) -> str:
    """Remove the STEERING_START/STEERING_END block from a prompt string.

    Strips trailing whitespace before the marker, as
    :func:`prompt_with_ledger` does.
    """
    start = text.find(STEERING_START)
    if start < 0:
        return text
    end = text.find(STEERING_END, start)
    if end < 0:
        return text
    return text[:start].rstrip() + text[end + len(STEERING_END):]


__all__ = [
    "CITATION",
    "KINDS",
    "LEDGER",
    "NOTE_KINDS",
    "PLACEHOLDER",
    "BOOKKEEPING_KINDS",
    "SCOPES",
    "SOURCES",
    "STEERING_START",
    "STEERING_END",
    "VERDICT_KIND",
    "Ledger",
    "SteeringError",
    "annotate",
    "chain",
    "citations",
    "fidelity",
    "for_run",
    "interactive",
    "outputs_in",
    "strip_steering",
]


def main() -> int:
    """Private JSON bridge used by both terminal and GUI workflow user gates."""
    try:
        request = json.load(sys.stdin)
        ledger = Ledger(Path.cwd())
        gate = str(request["gate"])
        review = ledger.review(gate, request["reads"])
        labels = request["labels"]
        if request["action"] == "begin":
            review = ledger.visit(gate, request["reads"])
            result = {"review": review, "verdict": ledger.take_verdict(gate, review, labels)}
        elif request["action"] == "poll":
            if review != request["review"]:
                raise SteeringError(f"gate {gate!r} inputs changed while awaiting the answer")
            result = {"review": review, "verdict": ledger.take_verdict(gate, review, labels)}
        elif request["action"] == "answer":
            if review != request["review"]:
                raise SteeringError(f"gate {gate!r} review inputs changed while awaiting the answer; ask again")
            verdict = request["verdict"].lower()
            if verdict not in [l.lower() for l in labels] and verdict != "halt":
                raise SteeringError(f"gate {gate!r} does not accept {verdict!r}")
            result = ledger.verdict(gate, verdict, review=review, answered=True)
            # Record feedback as steering in the intent ledger.
            feedback = str(request.get("feedback") or "").strip()
            if feedback:
                kind = "correction" if verdict == "reloop" else "constraint" if verdict == "proceed" else "info"
                ledger.note(kind, feedback)
        else:
            raise SteeringError("unknown user-gate decision action")
        print(json.dumps(result))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
