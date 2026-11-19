"""A proposal request shaped like S4's round 2 (S4J-s42, main 4e53948481): the edited
candidate's 12 train items, each three steps with long replies (every item shown whole
under "Trajectories"), and its child, which added a branch (two new steps) and moved 6
items -- 3 broken, 3 fixed -- each a five-step trace whose replies carry their own
Markdown headings at the request's levels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: The length of each step's reply (the S4 replies were 2-6k characters).
REPLY = 3000
ITEMS = 12
FLIPPED = {"i00": "broken", "i01": "broken", "i02": "broken",
           "i03": "fixed", "i04": "fixed", "i05": "fixed"}
#: With ``graded``: two items whose score moved without a flip (both wrong, graded).
GRADED = {"i07": ("score rose", 0.2, 0.6), "i09": ("score dropped", 0.6, 0.2)}


def reply(node: str, item: str, side: str, size: int | None = None) -> str:
    """A reply of about ``size`` characters (``REPLY``) with headings at every level the
    request uses, then distinct prose."""
    size = REPLY if size is None else size
    head = [f"## Theorem-Level Precision ({node}, {item}, {side})",
            f"#### Analysis of the case {item} by {node}",
            f"##### Step 1 of {node} on {item}"]
    body, k = [], 0
    while len("\n".join(head + body)) < size:
        body.append(f"{side} {node} {item} line {k}: the quantity is computed again and "
                    f"checked against the bound stated in the problem ({k * 7 % 13}).")
        k += 1
    return "\n".join(head + body)


def _record(item: str, passed: bool, steps: list[tuple[str, str]], prediction: str
            ) -> dict[str, Any]:
    return {"id": item, "passed": passed, "score": float(passed), "prediction": prediction,
            "label": "42", "fail_reason": None if passed else "the answer is not 42",
            "metadata": {"question": f"Problem {item}: find the value. " + "Context. " * 40},
            "exchanges": [{"node": node, "prompt": f"Solve {item}.", "response": text}
                          for node, text in steps]}


def parent_records(size: int | None = None, side: str = "parent",
                   graded: bool = False) -> list[dict[str, Any]]:
    """The parent's records; ``side`` names the run (another run of the same pipeline,
    a re-score, says other words); with ``graded``, `GRADED`'s items scored in between."""
    out = []
    for n in range(ITEMS):
        item = f"i{n:02d}"
        passed = FLIPPED.get(item) == "broken" or (item not in FLIPPED and n % 2 == 0)
        out.append(_record(item, passed, [(node, reply(node, item, side, size))
                                          for node in ("solve", "check", "final")],
                           "42" if passed else "41"))
        if graded and item in GRADED:
            out[-1]["score"] = GRADED[item][1]
    return out


def child_records(size: int | None = None, graded: bool = False) -> list[dict[str, Any]]:
    """The child: solve unchanged, a new branch (alt, merge), check and final changed on
    the flipped items (and, with ``graded``, `GRADED`'s); every other item the same as
    the parent's."""
    parents = {r["id"]: r for r in parent_records(size, graded=graded)}
    out = []
    for item, record in parents.items():
        moved = item in FLIPPED or (graded and item in GRADED)
        if not moved:
            out.append(record)
            continue
        passed = FLIPPED.get(item) == "fixed"
        steps = [("solve", record["exchanges"][0]["response"]),
                 *((node, reply(node, item, "child", size))
                   for node in ("alt", "merge", "check", "final"))]
        out.append(_record(item, passed, steps, "42" if passed else "43"))
        if item in GRADED and graded:
            out[-1]["score"] = GRADED[item][2]
    return out


def context(size: int | None = None, parent_shown: bool = True, graded: bool = False,
            **fields: Any):
    """The proposal context: livemath/solver's pipeline, the parent's records as the
    edited candidate's items (with ``parent_shown`` off, a re-score of it with other
    replies, so no trace of the pair is under "Trajectories"), the child as the last
    edit's pair. ``size``: each reply's length; ``graded``: all four kinds of change."""
    from refactory_dsh import registry
    from refactory_dsh.nn import edits as nn_edits
    from refactory_dsh.optim import ProposalContext
    from refactory_dsh.optim.memory import state_view

    _package, mode = registry.resolve(Path("."), "livemath/solver")
    view = state_view(mode().state_dict())
    pair = {"child": "c2", "parent": "c1", "gate": "rejected",
            "edit": [{"op": "branch", "step": "solve", "into": ["alt", "merge"]}],
            "parent_records": parent_records(size, graded=graded),
            "child_records": child_records(size, graded),
            "steps": ["alt", "merge"]}
    return ProposalContext(mode_name="m", nodes=view.nodes, edges=view.edges,
                           node_states=view.node_states, knobs=view.knobs,
                           catalogue=nn_edits.catalogue("broad"),
                           item_records=parent_records(size, "parent" if parent_shown
                                                       else "rescored", graded),
                           edit_pairs=[pair], **fields)
