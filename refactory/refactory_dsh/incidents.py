"""The harness's own failures: `HarnessIncident` and what tells one apart in text.

A module of its own with no imports, so every layer can raise or subclass the one class
(`model` re-exports it; `snippet.SandboxUnavailable` subclasses it).
"""
from __future__ import annotations

from typing import Any


#: How a `HarnessIncident` raised in another process's node is told apart in an item
#: record's text (`items.error_kind`): its message starts with this.
HARNESS_INCIDENT_TEXT = "harness incident: "


class HarnessIncident(RuntimeError):
    """The harness, not the candidate, kept an item from an outcome: its grader or
    sandbox failed (SpreadsheetBench's LibreOffice recalculation, its model-code
    sandbox; any Task's ``setup`` or ``verify``, or an op, may raise it). Its record's
    ``error_kind`` is ``harness_incident`` (`items.error_kind`): the loop runs the item
    again and, still one, leaves it out of the score (`loop.Loop._settle_incidents`).
    One raised in another process reaches the record as text, so its message starts with
    `HARNESS_INCIDENT_TEXT`, after its class's name (`HARNESS_INCIDENT_CLASSES`). A model
    that failed -- a wrong answer, an output limit, a session still writing at the
    backstop (`HarnessBackstop`) -- is never one."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        HARNESS_INCIDENT_CLASSES.add(cls.__name__)


#: The names of `HarnessIncident` and every subclass defined in this process: an error
#: that crossed a process as text is one only when its first line names one of them as
#: its type (`items.error_kind`), so a message that merely starts with
#: `HARNESS_INCIDENT_TEXT` -- a model's reply quoted in a ValueError, a code node's own
#: exception -- is not.
HARNESS_INCIDENT_CLASSES: set[str] = {"HarnessIncident"}


class SealFailed(HarnessIncident):
    """A sealed step's command could not start in its seal: its runner failed before the
    command ran (systemd's user scope bus gone, a session keyring or the sandbox refused;
    `seal_reaper.record_incident`). The machine's failure, so its item is a harness
    incident, never a step whose commands quietly found nothing (`workspace.collect`)."""
