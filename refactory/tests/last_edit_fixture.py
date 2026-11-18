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


def reply(node: str, item: str, side: str) -> str:
    """A reply with headings at every level the request uses, then distinct prose."""
    head = [f"## Theorem-Level Precision ({node}, {item}, {side})",
            f"#### Analysis of the case {item} by {node}",
            f"##### Step 1 of {node} on {item}"]
    body, k = [], 0
    while len("\n".join(head + body)) < REPLY:
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


def parent_records() -> list[dict[str, Any]]:
    out = []
    for n in range(ITEMS):
        item = f"i{n:02d}"
        passed = FLIPPED.get(item) == "broken" or (item not in FLIPPED and n % 2 == 0)
        out.append(_record(item, passed, [(node, reply(node, item, "parent"))
                                          for node in ("solve", "check", "final")],
                           "42" if passed else "41"))
    return out


def child_records() -> list[dict[str, Any]]:
    """The child: solve unchanged, a new branch (alt, merge), check and final changed on
    the flipped items; every other item the same as the parent's."""
    parents = {r["id"]: r for r in parent_records()}
    out = []
    for item, record in parents.items():
        if item not in FLIPPED:
            out.append(record)
            continue
        passed = FLIPPED[item] == "fixed"
        steps = [("solve", record["exchanges"][0]["response"]),
                 *((node, reply(node, item, "child")) for node in ("alt", "merge", "check",
                                                                   "final"))]
        out.append(_record(item, passed, steps, "42" if passed else "43"))
    return out


def context(**fields: Any):
    """The proposal context: livemath/solver's pipeline, the parent's records as the
    edited candidate's items, the child as the last edit's pair."""
    from refactory_dsh import registry
    from refactory_dsh.nn import edits as nn_edits
    from refactory_dsh.optim import ProposalContext
    from refactory_dsh.optim.memory import state_view

    _package, mode = registry.resolve(Path("."), "livemath/solver")
    view = state_view(mode().state_dict())
    pair = {"child": "c2", "parent": "c1", "gate": "rejected",
            "edit": [{"op": "branch", "step": "solve", "into": ["alt", "merge"]}],
            "parent_records": parent_records(), "child_records": child_records(),
            "steps": ["alt", "merge"]}
    return ProposalContext(mode_name="m", nodes=view.nodes, edges=view.edges,
                           node_states=view.node_states, knobs=view.knobs,
                           catalogue=nn_edits.catalogue("broad"),
                           item_records=parent_records(), edit_pairs=[pair], **fields)
