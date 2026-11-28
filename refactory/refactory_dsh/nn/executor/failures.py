"""Errors and failure classification helpers for the executor."""

from __future__ import annotations

from ..graph import Edge


class _BranchFailed(Exception):
    """A branch error string wrapped so it flows through the per-item try/except."""


class _ItemDeadlineError(Exception):
    """An item's lifecycle (setup + branch + verify) exceeded its deadline."""

    def __init__(self, node_id: str, item_id: str, deadline_s: float):
        self.node_id = node_id
        self.item_id = item_id
        self.deadline_s = deadline_s
        super().__init__(
            f"item '{item_id}' of '{node_id}' timed out after {deadline_s:.1f}s"
        )


class ExecutionError(RuntimeError):
    """A run could not be carried out."""


class OpOutcomeUnknown(ExecutionError):
    """A child exited after dispatch without reporting its side effects."""


class OpTimeoutError(RuntimeError):
    """An op or fn gate exceeded its per-node timeout."""

    def __init__(self, node_id: str, timeout_s: float):
        self.node_id = node_id
        self.timeout_s = timeout_s
        super().__init__(f"op '{node_id}' timed out after {timeout_s:.1f}s")


class GateStop(Exception):
    """The walker stopped at a user gate that needs a person's answer."""

    def __init__(self, gate: str, reads: list[str], edges: list[Edge]):
        self.gate = gate
        self.reads = reads
        self.edges = edges
        super().__init__(f"gate {gate!r} is waiting for a person")


class HaltStop(Exception):
    """A halt verdict ended the run."""

    def __init__(self, gate: str, reason: str = "verdict"):
        self.gate = gate
        self.reason = reason
        super().__init__(f"gate {gate!r} halted the run")

    def detail(self) -> str:
        """What the run's status says of this halt: which gate, and the reason it gave
        (a halt verdict's own reason, or the cap that refused a retry), as the spine
        keeps it. A plain ``halt`` with no reason says only which gate."""
        return f"{self}: {self.reason}" if self.reason and self.reason != "verdict" else str(self)


def route_verdict(verdict: str, edges: list[Edge], gate_kind: str) -> tuple[Edge | None, str]:
    """Route a verdict to an edge, matching the spine's ``Spine.verdict()``.

    Returns ``(edge, reason)``. When the edge is ``None``, the run is halted
    with the given reason (``"verdict"``).

    - Match labels case-insensitively.
    - ``proceed`` with no proceed edge falls through to the unconditional edge.
    - ``halt`` follows a halt edge if there is one; otherwise returns
      ``(None, "verdict")`` — terminal halt.
    - An unknown verdict raises :class:`ExecutionError`.
    """
    lower = verdict.lower()
    # Direct match (case-insensitive)
    for edge in edges:
        if (edge.condition or "").lower() == lower:
            return edge, ""
    # proceed falls through to unconditional
    if lower == "proceed":
        for edge in edges:
            if not edge.condition:
                return edge, ""
    # halt with no halt edge is terminal
    if lower == "halt":
        return None, "verdict"
    raise ExecutionError(
        f"verdict {verdict!r} is not one of {sorted(e.condition or '' for e in edges)}"
    )
