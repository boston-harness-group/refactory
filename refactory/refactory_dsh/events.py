"""What a run says while it runs, in one vocabulary.

A run writes a durable record (``progress.jsonl``) and a person watching needs a
line now and then, not that file at the end. Both come from the same record: this
module owns the event names, which of them a person wants by default, and how one
is rendered. Nothing else invents a progress format, so the terminal, the web
panel and the file cannot disagree about what happened.

Two levels, and the difference is deliberate:

- **meaningful** — the events a person watching a search wants: the run's shape,
  each candidate's score, what the optimizer kept and refused, each epoch's best,
  the final report, the held-out audit, and what the run did with each control act
  a person asked for. Printed by default.
- **verbose** — everything else a run records, rendered as the raw event. Printed
  only when a caller asks, because a stream nobody can read is not streaming.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable, TextIO

#: The events a watcher gets by default. Everything else is verbose-only.
MEANINGFUL = frozenset({
    "init",
    "forward",
    "rescore",
    "optimizer",
    "scheduler",
    "report",
    "holdout",
    "control",
})

#: Record keys the printer owns; a verbose line shows the event's own fields only.
_ENVELOPE = ("step", "epoch", "generation", "evaluations", "time")


def _score(value: Any) -> str:
    """A score as a person reads it, or `-` when the candidate was not scored."""
    if isinstance(value, bool) or value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return str(value)


def render(record: dict[str, Any], *, verbose: bool = False) -> str | None:
    """One line for one event, or ``None`` when the default level hides it.

    ``verbose`` renders every event: the meaningful ones as they always read, the
    rest as the event's own fields, so a caller debugging a run sees what was
    recorded rather than a second, lossier format.
    """
    step = str(record.get("step", ""))
    epoch = record.get("epoch", 0)
    if step == "init":
        return (
            f"run · optimizer {record.get('optimizer', '?')} · "
            f"{record.get('epochs', '?')} epoch(s) x {record.get('batch_size', '?')} · "
            f"population {record.get('population', '?')}"
            + (" · resumed" if record.get("resumed") else "")
            + (f" · warning: {record['warning']}" if record.get("warning") else "")
        )
    if step == "forward":
        where = f"{record.get('index', '?')}/{record.get('of', '?')}"
        line = (
            f"epoch {epoch} · candidate {where} {record.get('candidate', '?')} · "
            f"score {_score(record.get('score'))}"
        )
        if record.get("exit_ok") is False:
            detail = str(record.get("error") or "").strip()
            return line + " · failed" + (f": {detail[:160]}" if detail else "")
        if record.get("duration_s") is not None:
            line += f" · {float(record['duration_s']):.1f}s"
        return line
    if step == "rescore":
        # Fresh evidence: the candidate the optimizer edits, rolled out again on this
        # epoch's train sample. Shown because it costs a whole train evaluation.
        if record.get("failed"):
            return (f"epoch {epoch} · fresh evidence: re-scoring {record.get('candidate', '?')} "
                    f"failed ({record.get('records')}/{record.get('expected')} items"
                    + (f": {str(record.get('error'))[:120]}" if record.get("error") else "")
                    + "); its previous records stay")
        line = (f"epoch {epoch} · fresh evidence: re-scored {record.get('candidate', '?')} on "
                f"this epoch's train sample · score {_score(record.get('score'))}")
        if record.get("duration_s") is not None:
            line += f" · {float(record['duration_s']):.1f}s"
        return line
    if step == "optimizer":
        # Each count says what it counts: this epoch's proposals, each either accepted
        # for evaluation or refused, so the first is the sum of the other two (dm-265).
        line = (
            f"epoch {epoch} · proposals {record.get('proposed', 0)} this epoch: "
            f"accepted {record.get('accepted', 0)} for evaluation, "
            f"refused {record.get('refused', 0)}"
        )
        epoch_codes = record.get("epoch_refusal_codes")
        codes = record.get("refusal_codes")
        if isinstance(epoch_codes, dict):
            if epoch_codes:
                line += " (" + ", ".join(f"{code} {count}" for code, count
                                         in sorted(epoch_codes.items())) + ")"
        elif isinstance(codes, dict) and codes:
            # A record from before the epoch's own codes were kept: the run's tally.
            shown = ", ".join(f"{code} {count}" for code, count in sorted(codes.items()))
            line += f" (run so far: {shown})"
        return line
    if step == "scheduler":
        gate = record.get("gate")
        if isinstance(gate, dict):
            # With the validation gate: what it decided this epoch and its val scores;
            # the train scores are labelled as such, and train "improved" is not shown.
            return (
                f"epoch {record.get('epoch', epoch)} done · validation gate kept "
                f"{gate.get('kept', 0)}, rejected {gate.get('rejected', 0)} · val best "
                f"{_score(record.get('val_best'))} (current {_score(record.get('val_current'))}) · "
                f"train best {_score(record.get('best'))} · bound {record.get('bound') or 'no'}"
            )
        return (
            f"epoch {record.get('epoch', epoch)} done · best {_score(record.get('best'))} · "
            f"mean {_score(record.get('mean'))} · improved {record.get('improved', '?')} · "
            f"bound {record.get('bound') or 'no'}"
        )
    if step == "report":
        return (
            f"done · best {_score(record.get('best_score'))} "
            f"(raw {_score(record.get('raw_best_score'))}) · "
            f"{record.get('evaluations', '?')} evaluations · bound {record.get('bound') or 'no'}"
        )
    if step == "holdout":
        return (
            f"holdout · train {_score(record.get('train'))} · "
            f"holdout {_score(record.get('holdout'))} · gap {_score(record.get('gap'))} · "
            f"overfit {record.get('overfit', '?')}"
        )
    if step == "paused":
        return "paused · waiting for a resume or a stop"
    if step == "control":
        return _control_line(record)
    if step == "proposal_failed":
        # Shown without --verbose: a round that never reached the model must not
        # read as one that proposed nothing.
        detail = " ".join(str(record.get("detail") or "").split())[:200]
        return (
            f"epoch {epoch} · proposal failed · {record.get('optimizer', '?')} · "
            f"{record.get('error', '?')}" + (f": {detail}" if detail else "")
        )
    if not verbose:
        return None
    fields = {key: value for key, value in record.items() if key not in _ENVELOPE}
    detail = json.dumps(fields, default=str) if fields else ""
    return f"epoch {epoch} · {step}" + (f" · {detail}" if detail else "")


def _control_line(record: dict[str, Any]) -> str:
    """One act a person asked for, and what it did.

    A control event is the run saying back what it did with the request, which is the
    only confirmation the person gets: the act is honored in another process and this
    stream is where they read that it landed and what it changed.
    """
    action = str(record.get("action") or "?")
    if action == "stop":
        return "control · stop asked; the run ends at this boundary"
    if action == "adopt":
        line = f"control · adopt {record.get('as', '?')}"
        if record.get("error"):
            return line + f" · refused: {record['error']}"
        return (
            line + f" · candidate {record.get('candidate', '?')} "
            f"scored {_score(record.get('score'))} · wrote {record.get('package_root', '?')}"
        )
    if action in ("keep", "reject"):
        line = f"control · {action} {record.get('target', '?')}"
        if record.get("lineage"):
            line += " and its descendants"
        applied = record.get("applied")
        if isinstance(applied, list):
            line += f" · {len(applied)} candidate(s) changed"
        unknown = record.get("unknown")
        if isinstance(unknown, list) and unknown:
            line += f" · not in the population: {', '.join(str(i) for i in unknown)}"
        return line
    return f"control · {action}"


class Printer:
    """Render a run's events to a stream, at one of the two levels."""

    def __init__(self, stream: TextIO | None = None, *, verbose: bool = False) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.verbose = verbose

    def __call__(self, record: dict[str, Any]) -> None:
        """Print one event, unless the selected level hides it."""
        line = render(record, verbose=self.verbose)
        if line is not None:
            print(line, file=self.stream, flush=True)

    def output_line(self, line: str) -> None:
        """Print one line of a child's own output; only the verbose level shows it."""
        if self.verbose and line.strip():
            print(line.rstrip("\n"), file=self.stream, flush=True)


#: The shape a caller supplies when it wants events somewhere other than a stream.
Sink = Callable[[dict[str, Any]], None]
