"""The core structural operators: one line each, the data wired by the system.

Part of the edit layer (`nn.edits`): each operator here is a macro registered in its
catalogue, checked by a precondition and expanded into the seven primitives, so it
reaches every outer loop through the one attempt path (`Optimizer.attempt`) and is held
to every rule a hand-written edit is. They say nothing about what a step is for:

- ``insert``: a new step after or before an existing one;
- ``remove``: a step out of the pipeline, a fork branch included; on a gate, ``keep``
  names the verdict whose path stays (so it also un-routes);
- ``parallelize``: existing consecutive steps, copies of a step, or new steps, run at
  once between a fork and a join;
- ``serialize``: a fork's branches one after another, in their declared order;
- ``branch``: a new step as another (optionally optional) branch of an existing fork;
- ``rewire``: what an existing step sees (the task's own input, other steps' outputs),
  and with ``text`` an agent's instructions too, in place (a fork's branch stays one);
- ``loop`` (one-line form): a judge after a step that can send it back with a reason;
- ``route``: a judge after a step choosing among labelled next steps that rejoin;
- ``split`` (one-line form): a step replaced by two new ones, or by one (replace);
- ``merge``: two consecutive steps as one;
- ``fallback``: a step that answers instead when a per-item step genuinely fails (never
  for time or a harness fault), writing that step's outputs;
- ``map``: a new step run once per element of a per-item list, its results combined.

**What the proposer writes** is the kind of step, where, and its text (a prompt, a
callable or code). Every text has a default (the neighbour's, or a neutral line), so a
strategy with no model can apply any operator.

**What the system wires** (`_seen`, `_step_nodes`, `_place`):

- a new step's inputs: by default what its neighbour sees plus the neighbour's outputs
  (``sees`` names others: ``"task"`` for the task's own input, `operators.task_inputs`,
  or a step id for that step's outputs). Facts reach a model step as ``{{slots}}``
  in its own prompt knob; files through a ``collect`` step in front of it
  (`operators.collect`), so a missing file is said, not a crash. A code step gets the
  same files and facts in ``inp``;
- its output: one file named after its id and one fact of that name, placed as `REACHES`
  says (``to`` names another place), guarded when
  the step may not run;
- names: fresh ids, files and facts, the same for the same edit on the same parent;
- the fork's targets and the join's sources, edges, the gate's verdict edges, the
  loop's iteration limit (a knob of its own);
- every text it creates goes into a knob of its own (``<id>__task``, ``<id>__code``),
  for later ``set_knob``.

Nothing here repairs a proposal: an operator either expands into primitives or is
refused with the reason, and the expanded state is validated like any other. Steps
downstream of a change keep the file and fact names they read, unless the operator is
about changing that (``rewire``, ``remove``).
"""

from __future__ import annotations

import re
from contextvars import ContextVar
from typing import Sequence



#: The core operators, in the order the request lists them.
CORE_OPS = ("insert", "remove", "parallelize", "serialize", "branch", "rewire", "loop",
            "route", "split", "merge", "fallback", "map")


#: The kinds of step an operator adds: a model call, a plain op (a callable), or code.
STEP_KINDS = ("agent", "op", "code")


#: Judges of ``loop`` and ``route`` besides a ``module:function`` callable.
JUDGE_KINDS = ("agent", "code")


#: The most steps one ``parallelize`` or ``route`` adds or runs at once.
MAX_BRANCHES = 8


#: The most labels a ``route`` chooses among.
MAX_ROUTES = 6



#: Neutral texts, so an operator applies with none given.
NEUTRAL_STEP = "Work on the task with what you are given, and write your result."


NEUTRAL_CHECK = "Check the work above against the task: is it complete and correct?"


NEUTRAL_ROUTE = "Choose which way the work should go next."


NEUTRAL_REVIEW = "Review the work above against the task and write the final result."


NEUTRAL_ELEMENT = "Work on the element below, with what you are given, and write your result."


NEUTRAL_COMBINE = "Combine the results for each element above into one result for the task."


#: A map run's own prompt, before the input that holds its element and what to do with it.
MAP_ELEMENT_PROMPT = "Do what the input below says, for the one element it gives."


#: The same for a run with tools, which reads its element's file itself.
MAP_ELEMENT_TOOLS_PROMPT = "Do what the file {file} says, for the one element it gives."


#: A map's elements by default, and at most (one branch each, `MAX_BRANCHES`).
MAP_MAX = (4, 2, MAX_BRANCHES)


#: What a judge agent is told about its answer's first line: the word alone, with an
#: example (``{example}``, the last of its labels). A judge that reviews first and never
#: writes the word is asked again for it (`node_retry.settle_verdict`); this keeps that rare.
VERDICT_RULE = ("Your answer's first line is one word alone: {labels}. Why goes on the lines "
                "after it. Nothing comes before that word: no heading, no preamble, no "
                "\"Verdict:\". For example, a first line \"{example}\", then why, in a "
                "sentence or two, on the next.")


#: The rules judges were built with before, still matched in the prompts of states that
#: carry them (`system_appended`).
EARLIER_VERDICT_RULES = ("Start your answer with one word, {labels}, alone or followed by "
                         "why; nothing before it.",)




def verdict_rule(labels: Sequence[str]) -> str:
    """`VERDICT_RULE` for a judge answering one of ``labels``."""
    return VERDICT_RULE.format(labels=" or ".join(labels), example=list(labels)[-1])




def _rule_pattern(rule: str) -> str:
    """A rule as a pattern that matches it at a text's end, whatever its placeholders hold
    (each on one line)."""
    parts = re.split(r"\{(\w+)\}", rule)
    return "".join(re.escape(part) if index % 2 == 0 else rf"(?P<{part}>.+?)"
                   for index, part in enumerate(parts)) + r"\s*$"




def system_appended(text: str) -> tuple[str, str]:
    """``text`` split into the step's own words and what the system appended to them (the
    verdict rule a judge's prompt ends with, `VERDICT_RULE`, or an earlier one,
    `EARLIER_VERDICT_RULES`); ``(text, "")`` when nothing was. Our own constants are
    matched, never the model's words."""
    for rule in (VERDICT_RULE, *EARLIER_VERDICT_RULES):
        found = re.search(_rule_pattern(rule), text or "")
        if found:
            return text[:found.start()].rstrip(), text[found.start():].strip()
    return text, ""




#: The loop's default iteration limit and its knob's bounds.
LOOP_ITERATIONS = (2, 1, 5)


#: What ``sees`` calls the task's own input (`operators.task_inputs`).
TASK = "task"


#: The heading a ``collect`` step's fact gets in a prompt.
FILES_HEADING = "Files from earlier steps"




class _Refused(Exception):
    """A precondition that does not hold, with its reason."""




# ── reading a state ───────────────────────────────────────────────────────


def _edits():
    from refactory_dsh.nn import edits

    return edits




#: The new steps an operator's expansion wired by default (no 'sees' or 'reads' given
#: for them): only these are given the item's input files by default (`_with_item_files`).
_DEFAULT_WIRED: ContextVar[set[str] | None] = ContextVar("default_wired", default=None)




def _default_wired(node_id: str) -> None:
    found = _DEFAULT_WIRED.get()
    if found is not None:
        found.add(node_id)




#: How a refusal ends that names tools as the other way to write several files (a
#: take-over's, a fallback's): said only where the run offers a tool step
#: (`refusal_without_tools`).
TOOLS_HINTS = (", or with 'tools'", ", or 'tools'")

