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
from pathlib import Path
from typing import Any, Mapping

#: Where the ledger lives inside a run directory.
LEDGER = "intent-ledger.md"

#: The kinds an entry can be. A constraint is not information: it has to be
#: *respected*, not merely known, which is why it is a separate kind rather than a
#: note with a different tone.
KINDS: tuple[str, ...] = ("info", "correction", "constraint")

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
        ledger.path.write_text(f"# User Intent Ledger\n\n## [{_now()}] Initial Idea\n{text}\n")
        return ledger

    def note(self, kind: str, text: str) -> dict[str, Any]:
        """Append one thing the user said. Returns the entry as it was recorded."""
        if kind not in KINDS:
            raise SteeringError(f"unknown note kind {kind!r}; known: {list(KINDS)}")
        body = (text or "").strip()
        if not body:
            raise SteeringError("an empty note records nothing and dilutes the ledger")
        entry = {"at": _now(), "kind": kind, "text": body}
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(f"\n## [{entry['at']}] {kind}\n{body}\n")
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
        if not gate or not value or "\n" in gate or "\n" in value:
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
            handle.write(f"\n## [{entry['at']}] {entry['kind']}\n{body}")

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
        """Record the current declared review inputs for public steering to discover."""
        review = self.review(gate, reads)
        self._decision_entry({"at": _now(), "kind": "gate-visit", "gate": gate,
                              "review": review, "reads": json.dumps(list(reads))})
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
        """
        entries = self.entries()
        used = {self._fields(e).get("id") for e in entries if e["kind"] == "verdict-use"}
        latest = next((self._fields(e) for e in reversed(entries)
                       if e["kind"] == VERDICT_KIND and self._fields(e).get("gate") == gate), {})
        value = latest.get("verdict", "")
        if latest.get("scope") == "run" and latest.get("source") == "upfront":
            if value not in labels:
                raise SteeringError(f"upfront verdict {value!r} is not accepted by gate {gate!r}")
            return value
        decision = latest.get("id")
        if not decision or decision in used or latest.get("consumed") == "yes":
            return ""
        valid = latest.get("scope") == review and value in labels
        self._decision_entry({"at": _now(), "kind": "verdict-use", "id": decision,
                              "gate": gate, "result": "used" if valid else "expired"})
        return value if valid else ""

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
            current = {"at": stamp, "kind": kind.strip(), "text": ""}
            body = []
        elif current is not None:
            body.append(line)
    if current is not None:
        current["text"] = "\n".join(body).strip()
        found.append(current)
    return found


def render_entries(entries: list[dict[str, Any]]) -> str:
    """Render supplied entries; empty input leaves an unsteered prompt unchanged."""
    if not entries:
        return ""
    lines = [
        "## What the user has said", "",
        "This is the run's intent ledger: what was asked for, and everything said",
        "since. Entries marked `constraint` must be respected, not merely known.", "",
    ]
    for entry in entries:
        lines.append(f"- **[{entry['at']}] [{entry['kind']}]** {entry['text']}")
    return "\n".join(lines)


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


def authored_graph_state(state: dict[str, Any]) -> dict[str, Any]:
    """Compare candidate source instructions separately from live ledger context.

    Runtime steering is supplied again at dispatch. Its own appended block is
    not an authored graph edit; every other node field remains authoritative.
    """
    import copy

    result = copy.deepcopy(state)
    for node in result.get("nodes", {}).values():
        if node.get("kind") != "agent":
            continue
        prompt = str(node.get("prompt_template") or "")
        start = prompt.find(STEERING_START)
        end = prompt.find(STEERING_END, start)
        if start >= 0 and end >= start:
            prompt = prompt[:start] + prompt[end + len(STEERING_END):]
        node["prompt_template"] = prompt.rstrip()
    return result


def _citation_block(ledger: Ledger) -> str:
    """Keep live steering instructions separate from the node's output format."""
    return (
        "\n\nRead the live ledger at `" + str(ledger.path) + "` before acting. "
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


def for_run(workflow: Any, workdir: Path) -> Any:
    """Fold the run directory's ledger into a workflow about to run.

    Called by **both** runtimes. A mode lowered for DSH and one executed by the
    walker have to carry the same ledger; annotating in one path only is a
    difference between the runtimes that nothing would report.
    """
    return annotate(workflow, Ledger(workdir))


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


def chain(ledger: Ledger, outputs: Mapping[str, str], *, deliveries: Mapping[str, Any] | None = None) -> list[str]:
    """Problems with the chain from the ledger to the work, mechanically.

    Current runtime receipts establish delivered context and artifact identity.
    Legacy outputs can still cite entries inline. Neither establishes that the
    model read, understood, or satisfied the intent; that remains a judgement.
    """
    known = {entry["at"] for entry in ledger.entries()}
    problems: list[str] = []
    for node_id in sorted(outputs):
        delivery = (deliveries or {}).get(node_id)
        if delivery is not None:
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


def fidelity(ledger: Ledger, outputs: Mapping[str, str] | None = None, *, deliveries: Mapping[str, Any] | None = None) -> list[str]:
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
        problems.extend(chain(ledger, outputs, deliveries=deliveries))
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


__all__ = [
    "CITATION",
    "KINDS",
    "LEDGER",
    "PLACEHOLDER",
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
]


def main() -> int:
    """Private JSON bridge used by both terminal and GUI workflow user gates."""
    from refactory_dsh.run import _read_facts
    if _read_facts(Path.cwd()).get("create_fidelity"):
        from refactory_dsh.create_fidelity_gates import human_main
        return human_main()
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
            verdict = request["verdict"]
            if verdict not in labels:
                raise SteeringError(f"gate {gate!r} does not accept {verdict!r}")
            result = ledger.verdict(gate, verdict, review=review, answered=True)
        else:
            raise SteeringError("unknown user-gate decision action")
        print(json.dumps(result))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
