"""Finishing a pull request: a truncated refine of its Open items (`--refine <PR> --finish`).

A design run gets one fix round and always delivers; what it leaves is the pull
request's ``## Open items`` (`open_items`). A finish is what a person would do after
review: read those items, fix the ones named, and push onto the same pull request.

- **Which items.** `prepare` reads the description (`select`): the ids the
  ``finish_items`` knob names, or every open item that is not a ``judgment_call``.
- **Light path.** It is a refine, so `route_refine_design` skips research and strategy;
  design_doc is shown exactly the selected items (`context_block`). ``environment``
  items are never shown to it, nor to the builder: the run's own check re-evaluates
  them (`results`).
- **The approval gate.** When every selected item the builder gets is concrete (a
  change failure, unfinished ask or QA finding, with an ``evidence:`` line) and the
  design review passed, `after_design_review` answers ``gate_strategy`` ``proceed`` for
  its next visit only: a one-use answer scoped to the design and delivery plan the gate
  reads, as a steered answer is, so both runtimes route the gate the same way. The reason
  is in that ledger entry, the ``finish_gate`` fact and a line of the pull request. A
  ``judgment_call`` among the items, or a design review out of rewrites, leaves the gate
  to the person.
- **One fix round.** `prepare` refuses round knobs above `ROUND_KNOBS`, and every gate
  that sends a change back to the builder (prepare_check's self-check, gate_check's
  combined and early-asks send-backs, gate_qa's and gate_overwatch's, delivery_gate's
  conflict) asks `allows_round` first and counts what it sends (`sent_back`, the
  ``finish_rounds`` fact): once one round is spent, the next is a draft instead. A
  refused emission's retry is not a round: nothing was applied.
- **Same pull request out.** Delivery pushes onto the pull request's branch as any refine
  does, and adds each selected item's result under it in the description's Open items,
  ticking only what the run confirmed (`results`, `open_items.update`).
"""

from __future__ import annotations

import json
import re
from typing import Any

from refactory_dsh import open_items
from refactory_dsh.open_items import OpenItem

#: The fix rounds a finish run gets.
ROUNDS = 1

#: The round knobs a finish run sets (`run_flags`), and the most `prepare` accepts:
#: ``check_attempts=2`` lets gate_check send back once (`design_ops._combined_sendback_bound`),
#: ``check_attempt_cap=0`` turns off the convergence extension past it, and
#: ``self_check_fixups=0`` makes a failing self-check a send-back that counts as the round
#: (`design_ops.prepare_check`), not a fix-up that spends nothing.
ROUND_KNOBS = {"check_attempts": 2, "check_attempt_cap": 0, "self_check_fixups": 0}

#: The fact counting a finish run's send-backs to the builder, every gate together.
ROUNDS_FACT = "finish_rounds"

#: The reason a gate goes to a draft instead of sending the change back.
ROUND_SPENT = "finish run: its one fix round is spent"

_TEST_ID = re.compile(r"[\w./-]+\.py::[\w\[\]./:-]+|[\w./-]+\.(?:test|spec)\.[jt]sx?")


def _on(value: Any) -> bool:
    return value not in (False, None, 0, "0", "", "false", "False")


def fact(ctx: Any) -> dict[str, Any] | None:
    """The run's ``finish`` fact, when this run is a finish."""
    found = ctx.facts.get("finish")
    return found if isinstance(found, dict) and found.get("selected") else None


def rounds_spent(facts: dict[str, Any]) -> int:
    """Send-backs to the builder a finish run has made."""
    try:
        return int(facts.get(ROUNDS_FACT) or 0)
    except (TypeError, ValueError):
        return 0


def allows_round(ctx: Any) -> bool:
    """Whether a gate may send the change back to the builder: always, except on a finish
    run that has spent its round. A gate that may not goes on as a draft instead."""
    return fact(ctx) is None or rounds_spent(ctx.facts) < ROUNDS


def sent_back(ctx: Any, verdict: Any) -> Any:
    """``verdict``, counted as a finish run's round when it sends the change back."""
    if fact(ctx) is not None and str(verdict) == "reloop":
        ctx.set_fact(ROUNDS_FACT, rounds_spent(ctx.facts) + 1)
    return verdict


# ── prepare ─────────────────────────────────────────────────────────────────
def prepare(ctx: Any, pull: dict[str, Any] | None) -> dict[str, Any]:
    """The ``finish`` fact `prepare` returns: ``""`` unless this run is a finish.

    Reads the pull request's Open items, selects what to work on, and tells the ledger
    which items the builder gets (never an ``environment`` item). Refuses a finish
    without a refine, round knobs above `ROUND_KNOBS`, and a selection with nothing for
    the builder.
    """
    wanted = str(ctx.knob("finish_items", "") or "").strip()
    if not _on(ctx.knob("finish", False)):
        if wanted:
            raise ValueError("finish_items goes with finish: they are the Open items a finish works on")
        return {"finish": "", "finish_items_check": ""}
    if not isinstance(pull, dict):
        raise ValueError("finish goes with refine: it finishes an open pull request's Open items; pass "
                         "refine=<pull request>")
    if not _on(ctx.knob("pr", True)):
        raise ValueError("finish updates the pull request it finishes, so it takes no pr=0 (--no-pr)")
    over = [f"{name}={ctx.knob(name, limit)} (at most {limit})" for name, limit in ROUND_KNOBS.items()
            if int(ctx.knob(name, limit)) > limit]
    if over:
        raise ValueError("a finish run gets one fix round, so it takes no larger round knobs: "
                         + ", ".join(over))
    items = open_items.parse(str(pull.get("body") or ""))
    if not items:
        raise ValueError(f"the pull request {pull.get('url')} has no Open items block with an item in it, "
                         "so there is nothing to finish")
    selected = open_items.select(items, wanted)
    if not selected:
        raise ValueError(f"the pull request {pull.get('url')} has no open item a finish works on unasked "
                         "(judgment calls need --items naming them)")
    builder = [item for item in selected if item.effective_kind != "environment"]
    if not builder:
        raise ValueError(f"{', '.join(item.id for item in selected)} "
                         f"{'is an environment item' if len(selected) == 1 else 'are environment items'}, "
                         "which the run's check re-evaluates and the builder never gets; select an item "
                         "to fix with them, or re-run the check")
    from refactory_dsh.steering import Ledger

    note = (f"This run finishes the pull request {pull.get('url')}: fix exactly these of its open items, "
            "and nothing else:\n" + "\n".join(_line(item) for item in builder))
    ledger = Ledger(ctx.workdir)
    if note not in {entry["text"] for entry in ledger.entries()}:
        ledger.note("info", note)
    return {"finish": {
        "pull_request": str(pull.get("url") or ""),
        "selected": [item.to_dict() for item in selected],
        "builder": [item.id for item in builder],
        "environment": [item.id for item in selected if item.effective_kind == "environment"],
    }, "finish_items_check": judge_block(builder)}


#: What the QA judge is asked on a finish (`design.QA_JUDGE_PROMPT`): a QA finding or an
#: unfinished ask is resolved only when the judge confirms it (`results`).
JUDGE_ASK = (
    "\n## The open items this finish run fixes\n"
    "Also put `\"finish_items\"` in that JSON object: one entry per item below, "
    "`{\"id\": \"<its id>\", \"met\": true | false, \"evidence\": \"<the test id or path:line that "
    "shows it>\"}`. `met` is true only when you confirmed in the code, and with a test, that "
    "the item is fixed.\n")


def judge_block(items: list[OpenItem]) -> str:
    """The QA judge's ask about the items, for its prompt's ``finish_items_check``."""
    return JUDGE_ASK + "\n".join(_line(item) for item in items) + "\n"


def _line(item: OpenItem) -> str:
    evidence = "; ".join(item.evidence) or "(no evidence given)"
    return f"- **{item.id}** ({item.kind}): {item.summary} — evidence: {evidence}"


def _selected(found: dict[str, Any]) -> list[OpenItem]:
    return [OpenItem.from_dict(entry) for entry in found.get("selected") or []]


def builder_items(found: dict[str, Any]) -> list[OpenItem]:
    """The selected items the builder gets: never an ``environment`` item."""
    return [item for item in _selected(found) if item.effective_kind != "environment"]


#: What design_doc is told on a finish, above the items.
FIX_PLAN = ("This is a finish run: a truncated refine that fixes exactly the open items below, "
            "each with the evidence that it is open. Write a short fix plan: for each item, the "
            "file and function to change and the test that proves it fixed. Change nothing else.")


def context_block(found: dict[str, Any]) -> str:
    """The items design_doc plans for, after the pull request in its ``refine_context``."""
    return ("### Open items this finish run fixes\n" + FIX_PLAN + "\n\n"
            + "\n".join(_line(item) for item in builder_items(found)) + "\n\n")


def shown_description(body: str) -> str:
    """A finish's pull request description as design_doc sees it: without its Open items,
    which `context_block` gives selected, and without a draft's unresolved findings."""
    from refactory_dsh.delivery import _without_draft_section

    return open_items.without(_without_draft_section(str(body or ""))).strip()


# ── the approval gate ───────────────────────────────────────────────────────
GATE = "gate_strategy"


def decide_gate(ctx: Any, found: dict[str, Any]) -> dict[str, Any]:
    """Whether the approval gate may be skipped: the ``finish_gate`` fact `route_refine_design`
    sets. Eligible when every item the builder gets is concrete; the skip itself waits for
    the design review (`after_design_review`)."""
    items = builder_items(found)
    vague = [item for item in items if not item.concrete]
    if vague:
        why = "; ".join(f"{item.id} is a {item.kind}"
                        + ("" if item.effective_kind not in open_items.CONCRETE else " with no evidence")
                        for item in vague)
        return {"skipped": False, "eligible": False, "reason": f"the approval gate is asked: {why}"}
    return {"skipped": False, "eligible": True,
            "reason": "the approval gate is asked unless the design review passes: every item is concrete"}


#: What the approval gate reads (`design.py`'s gate_strategy), which its one-use answer is scoped to.
GATE_READS = ("design.md", "delivery.md")


def after_design_review(ctx: Any, *, passed: bool) -> None:
    """Answer the approval gate's next visit ``proceed`` when the finish is eligible and the
    design review passed; otherwise leave the gate to the person, saying why.

    The answer is one-use: scoped to the design and delivery plan the gate is about to
    read (`Ledger.review`), it is taken at that visit and never again, on either runtime.
    """
    found, gate = fact(ctx), ctx.facts.get("finish_gate")
    if found is None or not isinstance(gate, dict) or not gate.get("eligible"):
        return
    items = builder_items(found)
    if not passed:
        ctx.set_fact("finish_gate", {**gate, "skipped": False,
                                     "reason": "the approval gate is asked: the design review left "
                                               "problems it could not have rewritten, for the person "
                                               "to decide"})
        return
    from refactory_dsh.steering import Ledger

    reason = ("the approval gate was skipped: every item this finish run fixes is concrete, with "
              f"evidence ({', '.join(f'{item.id}: {item.kind}' for item in items)}), and the design "
              "review passed")
    ledger = Ledger(ctx.workdir)
    ledger.verdict(GATE, "proceed", review=ledger.review(GATE, GATE_READS), reason=reason)
    ctx.set_fact("finish_gate", {**gate, "skipped": True, "reason": reason})


# ── results ─────────────────────────────────────────────────────────────────
def _check(ctx: Any) -> tuple[str, str]:
    """The run's last check: its status (pass, skipped, a failure's kind such as code_failure, or
    "" when it never ran) and its output."""
    from refactory_dsh.invocations import EVIDENCE

    try:
        data = json.loads(ctx.read_text(EVIDENCE)) if ctx.exists(EVIDENCE) else {}
    except ValueError:
        data = {}
    validation = data.get("validation") if isinstance(data, dict) else None
    if not isinstance(validation, dict):
        return "", ""
    return (str(validation.get("status") or ""),
            "\n".join(str(validation.get(key) or "") for key in ("stdout", "stderr")))


#: What the check's output carries when part of it was not kept (`sandbox._Output`).
_TRUNCATED = re.compile(r"\[\.\.\. \d+ bytes of output not kept \.\.\.\]")


def _confirmed(ctx: Any) -> dict[str, str]:
    """The items the run's QA judge confirmed met, with its evidence (`JUDGE_ASK`)."""
    from refactory_dsh import judges

    try:
        objects = judges._objects(judges.text_of(ctx, judges.QA_JUDGE))  # noqa: SLF001
    except judges.Unreadable:
        return {}
    said = objects[0].get("finish_items") if len(objects) == 1 and isinstance(objects[0], dict) else None
    found: dict[str, str] = {}
    for entry in said if isinstance(said, list) else []:
        if (isinstance(entry, dict) and judges._met(entry.get("met")) is True  # noqa: SLF001
                and str(entry.get("evidence") or "").strip()):
            found[str(entry.get("id") or "")] = " ".join(str(entry["evidence"]).split())
    return found


def _named(item: OpenItem) -> list[str]:
    return sorted({match for entry in item.evidence for match in _TEST_ID.findall(entry)})


def results(ctx: Any, found: dict[str, Any], *,
            draft: dict[str, Any] | None) -> dict[str, tuple[str, str]]:
    """Each selected item's result: (open or resolved, the evidence for it), decided per item.

    - ``change_failure``: resolved only when the run's check passed and every test the
      item names passed in it: none of them is a failure the base has too, excused.
    - ``qa_finding``, ``unfinished_ask`` (and a judgment call a person selected): resolved
      only when the run delivers ready for review and its QA judge confirmed the item met,
      with evidence. Otherwise it was attempted, not confirmed.
    - ``environment``: resolved when the run's own check does not show its failure again:
      the check passed, or it failed on the code, its output whole, and on none of the
      tests the item names.
    """
    commit = str(ctx.facts.get("commit") or "")
    at = f"finish run at `{commit[:12]}`" if commit else "finish run"
    status, output = _check(ctx)
    command = str(ctx.facts.get("check") or "")
    inherited = {str(name) for name in ctx.facts.get("inherited_failures") or []}
    confirmed = _confirmed(ctx)
    found_results: dict[str, tuple[str, str]] = {}
    for item in _selected(found):
        kind, named = item.effective_kind, _named(item)
        if kind == "change_failure":
            failing = [line for line in output.splitlines() if re.search(r"FAIL|ERROR", line)]
            excused = [name for name in named
                       if name in inherited or any(name in line for line in failing)]
            if status == "pass" and named and not excused:
                found_results[item.id] = ("resolved", f"{at}: {', '.join(named)} passed in `{command}`")
            else:
                why = ("it names no test" if not named
                       else f"the check did not pass ({status or 'it did not run'})" if status != "pass"
                       else f"{', '.join(excused)} still failed, excused as a failure the base has too")
                found_results[item.id] = ("open", f"{at}: attempted, not confirmed: {why}")
            continue
        if kind != "environment":
            if draft:
                why = " ".join(str(draft.get("reason") or "the run is a draft").split())
                found_results[item.id] = ("open", f"{at}: attempted, not confirmed: the run delivered a "
                                                  f"draft ({why})")
            elif item.id in confirmed:
                found_results[item.id] = ("resolved", f"{at}: the QA judge confirmed it met "
                                                      f"({confirmed[item.id]})")
            else:
                found_results[item.id] = ("open", f"{at}: attempted, not confirmed: the QA judge did not "
                                                  "confirm it met")
            continue
        if status == "pass":
            found_results[item.id] = ("resolved", f"{at}: re-checked, `{command}` passed, so "
                                                  "the failure did not recur")
        elif status == "code_failure" and named and _TRUNCATED.search(output):
            found_results[item.id] = ("open", f"{at}: re-checked, `{command}` failed and its output was "
                                              "cut, so whether the failure recurred is not known")
        elif status == "code_failure" and named and not any(name in output for name in named):
            found_results[item.id] = ("resolved", f"{at}: re-checked, `{command}` failed, but "
                                                  f"not on {', '.join(named)}: the failure did not recur")
        elif status == "code_failure" and named:
            again = [name for name in named if name in output]
            found_results[item.id] = ("open", f"{at}: re-checked, the failure recurred ({', '.join(again)})")
        elif status == "code_failure":
            found_results[item.id] = ("open", f"{at}: re-checked, `{command}` failed and the item names no "
                                              "test, so whether its failure recurred is not known")
        elif status in ("", "skipped"):
            found_results[item.id] = ("open", f"{at}: not re-checked, the check "
                                              f"{'was skipped' if status == 'skipped' else 'did not run'}")
        else:
            # A timeout, a missing tool or an unreadable result says nothing about which
            # tests fail: the item stays open.
            found_results[item.id] = ("open", f"{at}: re-checked, but the check did not "
                                              f"finish cleanly ({status})")
    return found_results


def update_body(body: str, found: dict[str, Any], outcome: dict[str, tuple[str, str]]) -> str:
    """The pull request's description with each result in ``outcome`` added in place
    (`open_items.update`): the items as they are now, a box a person ticked meanwhile
    included, and every other line as it was."""
    return open_items.update(body, outcome, _selected(found))


def pr_section(ctx: Any, found: dict[str, Any], *, draft: dict[str, Any] | None) -> str:
    """The pull request text's ``## Finish run`` section: the gate's decision and each item's result."""
    gate = ctx.facts.get("finish_gate")
    lines = []
    if isinstance(gate, dict) and gate.get("reason"):
        lines.append(f"- {str(gate['reason'])[0].upper()}{str(gate['reason'])[1:]}.")
    outcome = results(ctx, found, draft=draft)
    for item in _selected(found):
        state, why = outcome[item.id]
        role = "re-checked, not given to the builder" if item.effective_kind == "environment" else "worked on"
        lines.append(f"- **{item.id}** ({item.kind}, {role}): {state} — {why}")
    return "## Finish run\n\n" + "\n".join(lines)
