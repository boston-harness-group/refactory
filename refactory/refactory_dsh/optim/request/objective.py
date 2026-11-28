"""Objective, score, time, answer instructions, table helpers."""


from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from ..types import COUNTING_RULE


def time_lines(facts: Mapping[str, Any] | None) -> list[str]:
    """"Time per item", in numbers, so time is reasoned about from them: how long the
    edited candidate's train items took each (median, 90th percentile, longest: from each
    item's own record). Nothing when it is not known. A model step has no time limit to
    state (a session ends on inactivity), and an item that timed out is said once, in
    "Item results" (`HARNESS_TIMEOUT`), not here."""
    facts = dict(facts or {})
    median = facts.get("per_item_median_s")
    if median is None:
        return []
    return ["## Time per item",
            f"The candidate you edit took {median} s per item at the median on its train "
            f"evaluation ({facts.get('per_item_p90_s')} s at the 90th percentile, "
            f"{facts.get('per_item_max_s')} s the longest; each item's own time).", ""]


def _cost_note(parsimony: float | None) -> list[str]:
    """With a cost term in the objective, what the cost column of the results says."""
    if not parsimony:
        return []
    return ["The cost per item column is each candidate's measured model cost (USD) over its "
            "train items, so a change's cost can be weighed against the size cost the score "
            "carries; - where it was not measured."]


def fmt_score(value: Any) -> str:
    """A score as every request prints it: 4 significant figures (``0.5714``); anything
    that is not a number as it is (``None`` as "n/a")."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a" if value is None else str(value)
    return f"{float(value):.4g}"


def _cost_cell(value: Any) -> str:
    return f"{float(value):.4g}" if isinstance(value, (int, float)) else "-"


#: What a run's score does not count unless the task's own score measures it.
_UNSCORED = "inference cost, latency, and the number of steps or model calls"


#: How the validation gate decides (``ValidationGate.judge_round``), said beside the
#: objective in every request, since the analyst, merge and memory requests do not carry
#: the proposal request's gate note.
_GATE_RULE = ("The validation gate decides on the task's score on the validation items: "
              "it keeps a candidate only when that score beats the current candidate's{none}; "
              "a train score{train} only breaks a tie between candidates that beat the "
              "current one by the same validation score; a tie with the current candidate "
              "is rejected.")


#: How ``selection`` ``checkpoint`` decides (``Loop._apply_round``), said in its place.
_CHECKPOINT_RULE = ("Each round's change is applied unless its train run fails (of several, "
                    "the one with the highest train score{train}); the validation score of "
                    "each applied pipeline{none} only picks the winner, the best of them, and "
                    "when the run stops.")


def _gate_rule(gated: bool | str, *, none: str, train: str) -> str:
    """The gate's rule for ``gated`` (a selection, or True for the strict gate)."""
    rule = _CHECKPOINT_RULE if gated == "checkpoint" else _GATE_RULE
    return rule.format(none=none, train=train)


def objective_text(parsimony: float | None, *, item_mean: bool = False,
                   gated: bool | str = False, cost_line: bool = True) -> str:
    """What the score measures, said from the run's own settings (``LoopConfig.parsimony``,
    whether the score is the items' mean and whether a validation gate decides), so the
    optimizer does not trade task score for a cost the run does not count. Empty when
    ``parsimony`` is not known (``None``). Domain-free: it names no task, and it says
    that cost is outside the score only as far as the loop knows: a task's own score
    may measure it.

    With parsimony 0 the score is the task's own score and nothing else. Above 0 the
    train score carries ``parsimony`` per node of the graph, as `Loop` computes it. Under
    the validation gate (either way) the gate's own comparison is the task's validation
    score, which carries no size cost; a train score only breaks a tie between
    candidates that beat the current one by the same validation score, and a tie with
    the current one is rejected (``ValidationGate.judge_round``). ``gated`` is the run's
    selection (``ProposalContext.selection``; True is the strict gate): under
    ``checkpoint`` the rule is that every change is applied unless its train run fails."""
    if parsimony is None:
        return ""
    task = "the task's own score" + (" (the mean of the items' scores)" if item_mean else "")
    unless = "unless the task's own score measures them"
    if not parsimony:
        text = (f"The score is {task} and nothing else. This run adds no cost term to it "
                f"(parsimony is 0): {_UNSCORED} are not part of the score {unless}. A "
                "change is judged only by the task score it reaches.")
        return text + (" " + _gate_rule(gated, none="", train="") if gated else "")
    size = (f"a size cost of {parsimony:g} per node of the pipeline's graph (score = raw "
            f"score - {parsimony:g} x nodes)")
    # Said by the compute policy instead when one follows (``cost_line=False``).
    rest = (" Apart from that size cost, inference cost, latency and the number of model "
            f"calls are not part of the score {unless}." if cost_line else "")
    if gated:
        return (f"The train scores shown are {task} minus {size}. "
                + _gate_rule(gated,
                             none=(" (no size cost)" if gated == "checkpoint"
                                   else ", and that score carries no size cost"),
                             train=", size cost included" + ("" if gated == "checkpoint" else ","))
                + rest)
    return f"The score is {task} minus {size}.{rest}"


def objective_lines(parsimony: float | None, *, item_mean: bool = False,
                    gated: bool | str = False) -> list[str]:
    """"What the score measures" (`objective_with_policy`) as request lines; none when not
    known."""
    text = objective_with_policy(parsimony, item_mean=item_mean, gated=gated)
    return ["## What the score measures", *text.split("\n"), ""] if text else []


def objective_with_policy(parsimony: float | None, *, item_mean: bool = False,
                          gated: bool | str = False) -> str:
    """What the score measures (`objective_text`) and, on its own line, the run's compute
    policy (`compute_policy`): what every optimizer call that recommends or weighs a
    change is told (the proposer, the analysts and their merges, the memory pass), from
    this one place. Empty when the scoring settings are not known."""
    text = objective_text(parsimony, item_mean=item_mean, gated=gated, cost_line=False)
    policy = compute_policy(parsimony, item_mean=item_mean, gated=gated)
    return f"{text}\n{policy}" if text and policy else text


def compute_policy(parsimony: float | None, *, item_mean: bool = False,
                   gated: bool | str = False) -> str:
    """The run's compute policy in one sentence, from its scoring settings: with
    parsimony 0, cost is not a reason for or against a change (extra steps and model
    calls per item are not penalised, so a change that adds them is weighed like any
    other); with a size cost, that cost (in the train scores, under a validation gate,
    whose own score carries none) is the only one counted. Empty when the settings are
    not known."""
    if parsimony is None:
        return ""
    if parsimony:
        where = " in the train scores" if gated else ""
        return (f"Compute policy: apart from the size cost{where}, inference cost, latency "
                "and the number of model calls are no reason for or against a change in "
                "this run, unless the task's own score measures them.")
    # What the score text does not say already (it says what the score is).
    return ("Compute policy: cost is not a reason for or against a change in this run"
            + ("; per-item cost is reported separately." if item_mean else "."))


def answer_instructions(catalogue: Sequence[dict[str, Any]], *, parsimony: float | None = None,
                        items: int = 0, code_nodes: bool = False) -> str:
    """The answer format and its rules, with examples for what this run permits.

    ``items`` is the number of train items when the raw score is known to be their
    mean (0 otherwise): only then does the size cost line compare a node with one item.
    """
    structural = structural_example(catalogue)
    ops = {op.get("op") for op in catalogue}
    kinds = ["a prompt or setting"] + ([] if structural is None else [
        *(["a new agent step"] if ops & {"insert", "insert_after", "insert_before", "add_node"}
          else []),
        *(["a code step"] if code_nodes else []),
        *(["a gate or loop"] if "loop" in ops else []),
        *(["parallel steps"] if ops & {"parallel", "parallelize"} else []),
        *(["removing, rewiring or merging steps"] if ops & {"remove", "rewire", "merge"}
          else [])])
    lines = [
        "Write one JSON object and nothing else:",
        "",
        '{"diagnosis": "<what the evidence shows is limiting the score>",',
        ' "considered": "<optional, short: the kinds of change you weighed for this '
        f'diagnosis ({", ".join(kinds)}) and why you chose yours>",',
        ' "proposals": [{"edit": {"op": "<operation>", ...},',
        '                "rationale": "<why, citing the evidence>",',
        *(['                "addresses": "<structural: the failure it targets, citing the '
           'steps and items in the evidence by id>",',
           '                "flow_change": "<structural: which step sees what, or what runs '
           'beside or after what, once it is made>",'] if structural is not None else []),
        '                "prediction": "<expected effect: which steps\' outputs or which '
        'items change, and how>",',
        '                "confidence": 0.0}]}',
        "",
        "Rules:",
        '- A parameter move is {"op": "set_knob", "name": "<parameter>", "value": <value>} or, '
        'for a text parameter, the same with "patch" in place of "value" (its form is in the '
        '`set_knob` row above).',
    ]
    if any(op.get("op") == "set_knob" for op in catalogue):
        # SkillOpt's patch mode edits its skill only by local append/insert/replace/delete
        # patches; a whole new value stays available for what a patch cannot express.
        lines += ["- To change a text parameter such as a prompt, patch it, in the form "
                  "the `set_knob` row above gives.",
                  "  A whole new `value` is for a setting that is not text, a new step's own",
                  "  prompt, or a prompt that has to be replaced outright."]
    if structural is not None and structural.get("op") == "insert":
        lines += [
            "- A structural move is one line: a structural operator above (or a written-out "
            "operation, with the fields it lists). For example:",
            "  " + json.dumps(structural),
            "  The new step gets its own `<id>__task` knob holding its text, so a later round "
            "can patch or rewrite it with `set_knob`.",
            "- A structural proposal also says what it `addresses` (the failure, citing steps "
            "and items in the evidence by id), its `flow_change` (which step sees what, or "
            "what runs beside or after what) and a `prediction` that can be checked (which "
            "steps' outputs or which items change, and how). Its edit line stays one line; "
            "the thought goes here. A proposal without them is still judged, and marked.",
        ]
    elif structural is not None:
        lines += [
            "- A structural move is one of the structural operations above, with the fields it "
            "lists. A `node` is a node object written from the node schema (or a block, or a "
            "copy of a node's JSON, with a new id). For example:",
            "  " + json.dumps(structural),
            "  A new agent node gets its own `<new step id>__system_prompt` knob holding its "
            "system prompt and `<new step id>__task` knob holding its `prompt_template`, so a "
            "later round can patch or rewrite either with `set_knob`.",
        ]
    else:
        lines.append("- This run permits parameter moves only.")
    if parsimony:
        size = f"- Size cost: {parsimony:g} per node (score = raw score - {parsimony:g} x nodes)"
        if items:
            # Said only when the raw score is the mean of the items' scores.
            size += (f"; the raw score is the mean over the {items} train items, so one item "
                     f"is worth 1/{items} = {1 / items:.3g}")
        else:
            size += "; compare it with the size of the score differences between candidates"
        lines.append(size + ".")
    lines += [
        "- Ground each change in the evidence: a setting, a prompt or a structural change, whose",
        "  effect you can state -- the failure it addresses (the step where it starts), how it",
        "  changes what a step sees or how the work is done, and a prediction that can be",
        "  checked. Choose the kind of change by the cause you diagnosed; neither kind is the",
        "  safer default.",
        "- Your prediction is recorded and later checked against the outcome, so say what",
        "  you expect and how you could be wrong.",
        "- If the evidence justifies no change, return an empty proposal list and say why.",
    ]
    return "\n".join(lines) + "\n"


# ANSWER_INSTRUCTIONS is defined after cross-module imports, below.


def bundle_examples(catalogue: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Bundles for the example, neither kind first by privilege: two steps' prompt knobs,
    then -- when the run also permits a structural op that takes a node and
    ``set_knob`` -- a structural edit that adds a node and the new node's own prompt
    knob."""
    patch = [{"kind": "replace", "target": "<exact text in it>", "content": "<new text>"}]
    out = [[{"op": "set_knob", "name": "<step A's prompt knob>", "patch": patch},
            {"op": "set_knob", "name": "<step B's prompt knob>", "patch": patch}]]
    structural = structural_example(catalogue)
    ops = {op.get("op") for op in catalogue}
    if structural is not None and "node" in structural and "set_knob" in ops:
        # The node leaves its system prompt to the knob the bundle sets.
        node = {k: v for k, v in structural["node"].items() if k != "system_prompt_template"}
        out.append([{**structural, "node": node},
                    {"op": "set_knob", "name": "<new step id>__system_prompt",
                     "value": "<system prompt text>"}])
    elif structural is not None and "set_knob" in ops:
        # A new step, and how a later step is told to use what it makes.
        out.append([structural, {"op": "set_knob", "name": "<a later step's prompt knob>",
                                 "patch": [{"kind": "append",
                                            "content": "<how to use the new step's output>"}]}])
    return out


def bundle_instructions(max_edits: int, catalogue: Sequence[dict[str, Any]] = (),
                        proposals: int = 0) -> str:
    """How a proposal bundles edits, said only when the run allows more than one.

    When the request asks for a single proposal (``proposals == 1``) only one
    candidate is evaluated, so an idea put in a separate proposal is never run, and
    fixes that only work together would be split across rounds and each judged
    alone: the advice is then to cover every step of the diagnosed failure path in
    the one proposal. Otherwise separate ideas go in separate proposals."""
    shown = [{"edits": bundle, "rationale": "<why these changes only work together>",
              "prediction": "<expected effect on the score>", "confidence": 0.0}
             for bundle in bundle_examples(catalogue)]
    # Never more proposals in one example than the request asks for: past that, each
    # bundle is an example of its own, one answer each.
    per = proposals if proposals > 0 else len(shown)
    examples = [json.dumps({"proposals": shown[i:i + per]}) for i in range(0, len(shown), per)]
    example = "\n\nor:\n\n".join(examples)
    if proposals == 1:
        scope = f"""\
Only one proposal is asked for, so only one candidate is evaluated: an idea left for a
separate proposal is not run this round. When your diagnosis traces a failure through
several steps, make this one proposal's edits cover every step it names on that
failure path (up to {max_edits} changes), so the fix is evaluated as a whole; changes that
only work together each fail when tried alone."""
    else:
        scope = """\
Bundle only what belongs to one idea: separate ideas go in separate proposals, so each
is judged on its own."""
    return f"""\
Bundling: when one fix needs coordinated changes -- several steps' prompts that have
to agree, or a new node and the knob that feeds it -- one proposal may carry up to
{max_edits} changes as an ordered list of edits under "edits" instead of "edit". {COUNTING_RULE}
They are applied in order to the same parent, each to the state the one before it left,
and the result is evaluated and judged once as one candidate; if any edit is refused,
the whole proposal is, and the refusal names the edit.

{scope}

Example{"s" if len(examples) > 1 else ""}:

{example}"""


def _rules_text() -> str:
    from refactory_dsh.nn.validate import rules_text

    return rules_text()


#: Trace keys the request leaves out: the leaderboard is "Recent results" already,
#: and measurements carry per-trial worker receipts and run-directory paths -- noise to
#: an optimizer, which reads their aggregates (``measurement_summary``) instead.
TRACE_OMITTED_KEYS = ("leaderboard", "measurements", "workers", "workdir", "trials")


#: The longest key or text a trace row shows; a text knob's value used as a key
#: (``knob_attribution``) is shown by its head and length.
TRACE_TEXT_CHARS = 160


#: The trace rows about what running a pipeline cost (seconds, inner model calls). Left
#: out of the request when the loop adds no cost term (parsimony 0) and the score is the
#: items' mean (so no cost is in it): the run's gradient.json, journal and report keep
#: them.
UNSCORED_TRACE_KEYS = ("cost",)


def _attribution_key(value: str, knob: str, current: Mapping[str, Any],
                     seed: Mapping[str, Any], first: Any = None) -> str:
    """A knob value as ``knob_attribution`` names it: a long text (`SAID_ONCE_CHARS`+) by
    which one it is -- the current one (whole under "Each step's settings"), the seed's,
    or the one candidate ``first`` was first evaluated with -- with its length; a short
    one as it is. Unique for the knob's values; `_trace_rows` never merges two."""
    if len(value) < SAID_ONCE_CHARS:
        return value
    if isinstance(current.get(knob), str) and value == current[knob]:
        return f"the current text ({len(value)} chars)"
    if isinstance(seed.get(knob), str) and value == seed[knob]:
        return f"the seed's text ({len(value)} chars)"
    if first:
        return f"the text {first} was first scored with ({len(value)} chars)"
    return f"a text of {len(value)} chars, #{hashlib.sha1(value.encode()).hexdigest()[:8]}"


def _trace_rows(traces: Sequence[dict[str, Any]],
                omit: Sequence[str] = (), *, current: Mapping[str, Any] | None = None,
                seed: Mapping[str, Any] | None = None) -> list[tuple[str, Any]]:
    """The traces as the request shows them: no receipts, no paths, short texts, and
    none of the top-level rows ``omit`` names. A text value in ``knob_attribution`` is
    named by which it is (`_attribution_key`: ``current`` and ``seed`` are the knobs'
    current and starting values), never repeated."""

    def short(text: str) -> str:
        return text if len(text) <= TRACE_TEXT_CHARS else (
            f"{text[:TRACE_TEXT_CHARS - 40]}… ({len(text)} characters)")

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {short(str(k)): clean(v) for k, v in value.items()
                    if k not in TRACE_OMITTED_KEYS}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, str):
            return short(value)
        return value

    def named(attribution: Any) -> Any:
        if not isinstance(attribution, dict):
            return attribution
        out = {}
        for knob, entry in attribution.items():
            if isinstance(entry, dict) and isinstance(entry.get("tried"), dict):
                tried: dict[str, Any] = {}
                for value, row in entry["tried"].items():
                    key = _attribution_key(str(value), str(knob), current or {}, seed or {},
                                           row.get("first") if isinstance(row, dict) else None)
                    if key in tried:  # never one row for two values
                        key += f" #{hashlib.sha1(str(value).encode()).hexdigest()[:8]}"
                    # Which candidate it came from is in the key when it names a text.
                    tried[key] = ({k: v for k, v in row.items() if k != "first"}
                                  if isinstance(row, dict) else row)
                entry = {**entry, "tried": tried}
            out[knob] = entry
        return out

    return [(key, clean(named(value) if key == "knob_attribution" else value))
            for key, value in _flatten(traces)
            if key not in TRACE_OMITTED_KEYS and key not in omit]


def _flatten(rows: Sequence[dict[str, Any]]) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for row in rows:
        for key, value in row.items():
            out.append((key, value))
    return out


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "(none)"
    head = "| " + " | ".join(headers) + " |"
    rule = "|" + "|".join("---" for _ in headers) + "|"
    body = [
        "| " + " | ".join(_cell(value) for value in row) + " |" for row in rows
    ]
    return "\n".join([head, rule, *body])


def _table_filled(headers: list[str], rows: list[list[Any]]) -> str:
    """`_table` without the columns that are empty in every row (nothing to read there)."""
    keep = [i for i in range(len(headers))
            if any(row[i] not in (None, "", []) for row in rows)]
    return _table([headers[i] for i in keep], [[row[i] for i in keep] for row in rows])


def _cell(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text.replace("|", "\\|").replace("\n", " ")


# --- cross-module imports (resolved at definition time, used at call time) ---
from .composition import structural_example  # noqa: E402
from .edits import SAID_ONCE_CHARS  # noqa: E402

#: The answer format for a run that permits every operation, as the request shows it.
ANSWER_INSTRUCTIONS = answer_instructions(  # noqa: E402
    [{"op": "insert_after", "kind": "topology", "fields": ["target", "node"]}])
