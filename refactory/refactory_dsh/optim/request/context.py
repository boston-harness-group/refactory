"""ProposalContext and the request rendering entry point."""


from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

#: How the edited candidate's train trajectories reach the optimizer
#: (``ProposalContext``): every train item, every proposal, at full detail -- its input,
#: every step's output in the order the pass ran them, the expected answer and the
#: prediction -- so a proposal can name the failure pattern and see what already works.
#: One call's evidence budget: when every item fits it in full, they all go to the
#: proposal call inline; when not, readers go through them in groups that each fit it,
#: and the proposal call gets their findings, a table of every item and some in full.
TRAJECTORY_BUDGET = 120000


#: The whole proposal request's room in characters, tied to the optimizer model's context
#: as ``analysts.ANALYST_ROOM`` is (same size, same reasoning: Opus 4.6 reads 200k
#: tokens, Claude Code's own system prompt and the reserved reply take 30-40k, and traces
#: run 2.5-3.5 characters a token). Only the items' texts give way to it, whatever their
#: sizes (`ProposalContext.render`): "Some items in full" and the blocks of "What the last
#: edit changed" share what the rest of the request leaves, items whole while they fit
#: and in rank order, the rest named; an item or block that alone passes its share has
#: its longest texts cut to fit (marked), or is named. Nothing else is cut for it. A
#: request the provider still refuses as too long is asked again once at half its length
#: (``AutoresearchOptimizer.propose``), which these then always make shorter.
PROPOSAL_ROOM = 400000


#: How many reader calls run at once.
READER_CONCURRENCY = 4


#: A reader's findings as the proposal call shows them: at most this many patterns
#: per reader, each pattern's text and quotes clipped, at most this many quotes and
#: item ids per pattern, and the whole findings block within a share of the budget.
READER_PATTERNS = 12


READER_PATTERN_CHARS = 400


READER_QUOTES = 3


READER_QUOTE_CHARS = 240


READER_ITEM_IDS = 40


READER_FINDINGS_SHARE = 0.4


#: How the edited graph's nodes are shown in full (``ProposalContext.node_states``):
#: each long text field keeps its head and tail within this many characters, and the
#: whole section (each node's state plus the generated node schemas) fits this cap.
NODE_SPEC_TEXT_CHARS = 1200


NODE_SPEC_TEXT_FLOOR = 200


NODE_SPEC_SECTION_CHARS = 30000


#: A prompt line's tail counts as a piece of a reason or a reply (`_settled`) only after a
#: label of at most this many characters ("Sent back because:"), and only when the tail
#: is at least `TAIL_MIN` characters or `TAIL_MIN_WORDS` words: a framing line's last
#: word ("- User: ## Question") is not a piece of a reply that happens to use it.
TAIL_LABEL_MAX = 24


TAIL_MIN = 20


TAIL_MIN_WORDS = 3


#: The shortest piece of an item's data (`_data_lines`) that marks a prompt line as
#: carrying it; a shorter one ("A", "yes") would match framing by chance.
ITEM_DATA_MIN = 8


#: A heading of at most this many words is a bare label (`_bare_label`).
BARE_HEADING_WORDS = 3


def _bare_label(line: str) -> bool:
    """Whether a line is only a label: one word ("Independent") or a short heading
    ("## Question", "## Option Comparison Notes"). A model's free text using one says
    nothing that a framing line holding the same words carries from the item."""
    words = line.split()
    if len(words) == 1:
        return True
    heading = line.lstrip("#")
    return heading != line and len(heading.split()) <= BARE_HEADING_WORDS


def _data_lines(value: Any, *, free: bool = False) -> set[str]:
    """Every line of ``value``'s strings (nested in lists and dicts too), normalised, of at
    least `ITEM_DATA_MIN` characters: an item's input, a step's reply or a code gate's
    reason, which a prompt line holding one carries from that item. With ``free`` (a
    model's reply text), a bare label (`_bare_label`) is left out; a structured field (a
    gate's reason, a string inside a JSON reply) counts at any length from
    `ITEM_DATA_MIN`, one word or a heading included. A string that is JSON counts by its
    own strings too, so a reason extracted from one is matched in the prompt quoting it."""
    from ..evidence import _norm

    out: set[str] = set()
    stack: list[tuple[Any, bool]] = [(value, free)]
    while stack:
        current, loose = stack.pop()
        if isinstance(current, Mapping):
            stack += [(v, loose) for v in current.values()]
        elif isinstance(current, (list, tuple)):
            stack += [(v, loose) for v in current]
        elif isinstance(current, str):
            out |= {n for line in current.splitlines()
                    if len(n := _norm(line)) >= ITEM_DATA_MIN
                    and not (loose and _bare_label(n))}
            text = current.strip()
            if text[:1] in ("{", "[", '"'):
                try:
                    stack.append((json.loads(text), False))
                except json.JSONDecodeError:
                    pass
    return out


def _holds(line: str, datum: str) -> bool:
    """Whether ``datum`` is in ``line`` as whole words: a reply's heading "## Option C" is
    not in the op's heading "## Option Comparison Notes"."""
    start = line.find(datum)
    while start != -1:
        end = start + len(datum)
        if (start == 0 or not line[start - 1].isalnum()) and \
                (end == len(line) or not line[end].isalnum()):
            return True
        start = line.find(datum, start + 1)
    return False


def _whole_texts(value: Any) -> set[str]:
    """Each string of ``value`` (nested, and inside a JSON string) whole, normalised: a
    reason or a reply, which a prompt line holding a piece of it carries."""
    from ..evidence import _norm

    out: set[str] = set()
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            stack += list(current.values())
        elif isinstance(current, (list, tuple)):
            stack += list(current)
        elif isinstance(current, str) and current.strip():
            out.add(_norm(current))
            if current.strip()[:1] in ("{", "[", '"'):
                try:
                    stack.append(json.loads(current))
                except json.JSONDecodeError:
                    pass
    return out


def settled_for(knobs: Iterable[Mapping[str, Any]], node_states: Iterable[Mapping[str, Any]],
                records: Iterable[Mapping[str, Any]]) -> frozenset[str]:
    """What an item's step does not repeat of its prompt (`ITEM_FORM`), in every request that
    shows items (the proposer's, the analysts', memory's): ``knobs`` and ``node_states``
    are the settings and nodes the request shows, ``records`` its items. It is text that
    comes from no item: the lines of every setting's text ("Each step's
    settings"), of the nodes' own prompt templates (each node in full, in the
    reference), and the framing the step's ops put around its inputs (the model
    client's wrapper, their headings). Framing is a line in every one of the step's
    prompts, over two or more items, that holds nothing from any item's own data
    (its input, a step's reply, or a code gate's reason; as whole words), nor a piece of
    a code gate's reason or of a reply (a reason spread over several prompt lines).

    Never a line just because it recurs: a judge's reason that is not in every one
    of the step's prompts, or a choice text (part of an item's input) said the same
    way in two items, is each item's own evidence and is shown."""
    from ..evidence import _norm, settled_lines

    texts = [k.get("value") for k in knobs]
    texts += [n.get(field) for n in node_states
              for field in ("prompt_template", "system_prompt_template")]
    records = list(records)
    every: dict[str, set[str]] = {}
    items: dict[str, set[str]] = {}
    data: set[str] = set()
    # Reasons and replies whole: a prompt line may carry a piece of one (a gate's
    # reason is stored as one line, and reaches a prompt over several). Never an op's
    # output, which is often the framed prompt itself.
    said: set[str] = set()
    for record in records:
        # Of the record's steps, only a code gate's reason is item data: an op's
        # output is often the framed prompt itself, and a visit's node id is no item's.
        reasons = [step.get("reason") for step in record.get("steps") or ()
                   if isinstance(step, Mapping) and step.get("kind") == "gate"]
        data |= _data_lines(record.get("metadata")) | _data_lines(reasons)
        said |= _whole_texts(reasons)
        for exchange in record.get("exchanges") or ():
            node = str(exchange.get("node"))
            lines = {_norm(line) for line in str(exchange.get("prompt") or "").splitlines()
                     if line.strip()}
            every[node] = lines if node not in every else every[node] & lines
            items.setdefault(node, set()).add(str(record.get("id")).split("#")[0])
            data |= _data_lines(exchange.get("response"), free=True)
            said |= _whole_texts(exchange.get("response"))

    def carried(line: str) -> bool:
        if any(_holds(line, datum) for datum in data):
            return True
        # The line inside a reason or a reply, or its tail after a short label
        # ("Feedback: ..."): a piece of one. The bounds keep a framing line's last word
        # ("- User: ## Question") from matching a reply that happens to use it.
        if line in said:
            return True  # a whole reason or reply, however short ("## Reloop")
        if _bare_label(line):
            return False  # a heading or one word is no piece of a reason
        words = line.split(" ")
        for start in range(len(words)):
            label, tail = " ".join(words[:start]), " ".join(words[start:])
            if len(label) > TAIL_LABEL_MAX:
                return False
            if len(tail) < ITEM_DATA_MIN or (start and len(tail) < TAIL_MIN
                                            and len(words) - start < TAIL_MIN_WORDS):
                return False
            if any(tail in text for text in said):
                return True
        return False

    framing = {line for node, lines in every.items() if len(items[node]) > 1
               for line in lines if not carried(line)}
    return settled_lines(texts) | framing


@dataclass
class ProposalContext:
    """The gradient: everything the optimizer is allowed to know.

    It carries the surface it may write to, the catalogue of edits it may use, the
    blocks it may insert, the run's numbers, and the reasons recent proposals were
    refused. Nothing here reaches ``score()`` or ``invariants()``.
    """

    epoch: int = 0
    generation: int = 0
    mode_name: str = ""
    best_score: float | None = None
    #: The mean score of every candidate scored so far (``Loop.loss_step``), over
    #: ``mean_over`` of them; not one batch's mean.
    mean_score: float | None = None
    mean_over: int = 0
    #: The run's intent ledger, rendered, read fresh before each generation. A note a
    #: person adds while the search runs reaches the next proposal, so a search is
    #: steerable and not only watchable.
    steering: str = ""
    knobs: list[dict[str, Any]] = field(default_factory=list)
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    leaderboard: list[dict[str, Any]] = field(default_factory=list)
    traces: list[dict[str, Any]] = field(default_factory=list)
    catalogue: list[dict[str, Any]] = field(default_factory=list)
    insertable: list[dict[str, Any]] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    #: Every evaluated candidate this run, bounded. An optimizer that can only see
    #: the top few cannot ask "have I tried this?", and the base class cannot refuse
    #: a duplicate without knowing what already exists.
    history: list[dict[str, Any]] = field(default_factory=list)
    #: The state ids that already exist, so a proposal recreating one can be refused
    #: instead of silently consuming a batch slot.
    tried: list[str] = field(default_factory=list)
    #: How many parents the loop would have offered before the population was handed
    #: over. An optimizer that has no parent policy of its own selects this many and
    #: behaves as it did; one that has a policy ignores it and chooses from the pool.
    #: Without it, widening the pool silently changes every existing optimizer.
    elite: int = 3
    history_truncated: bool = False
    #: Summaries of previous runs, so a strategy can carry a prior across runs.
    prior_runs: list[dict[str, Any]] = field(default_factory=list)
    #: The mode author's facts about the mode (`nn.Mode.guidance`), shown under "About this mode".
    guidance: str = ""
    preferred_edits: list[str] = field(default_factory=list)
    #: Knobs the operator has frozen for this run, on top of the ones the mode
    #: itself declares frozen. Freezing is a property of the run, not of the mode,
    #: so it cannot be edited into the definition by a candidate.
    frozen: list[str] = field(default_factory=list)
    #: The mode's protected nodes (``Mode.protected_nodes``): no edit may remove,
    #: rewire or replace them, and one that does is refused with ``E-protected``.
    protected: list[str] = field(default_factory=list)
    #: The mode's grader nodes (``Mode.grader_nodes``), which the pipeline view reads as
    #: it does in the loop's records.
    graders: list[str] = field(default_factory=list)
    #: Facts about the edited graph, worked out once by the loop (`graph_facts`): what
    #: ``"sees": ["task"]`` gives, what a fork's new branch can see, and which removals,
    #: replacements and merges an attempt refuses, with why.
    graph_facts: list[str] = field(default_factory=list)
    #: What the last ``render`` left out of an item's texts (each: item, step, chars), for
    #: the run's own record; the request says only "[... N chars omitted]".
    omissions: list[dict[str, Any]] = field(default_factory=list, repr=False, compare=False)
    preset: str = "broad"
    #: What the outer loop decided for *this* iteration, as data.
    #:
    #: An outer loop that only logs its decision has not made one: the proposal
    #: cannot see it, so the log is the only place it exists and the search is the
    #: default loop with extra bookkeeping. A strategy that branches puts the
    #: branch here, and a proposer that can act on it does.
    strategy: dict[str, Any] = field(default_factory=dict)
    #: Per-item records for the parent/current candidate the optimizer is editing (F6).
    item_records: list[dict[str, Any]] = field(default_factory=list)
    #: Alias for the parent candidate's train records.
    parent_item_records: list[dict[str, Any]] = field(default_factory=list)
    #: Why the pipeline shown is not the parent's, when it is not (empty when it is).
    graph_note: str = ""
    #: The edited graph's nodes in node-state form, as ``add_node`` and
    #: ``insert_after`` take a node, so a model can copy one and change it.
    node_states: list[dict[str, Any]] = field(default_factory=list)
    node_spec_text_chars: int = NODE_SPEC_TEXT_CHARS
    node_spec_section_chars: int = NODE_SPEC_SECTION_CHARS
    #: One call's evidence budget (``TRAJECTORY_BUDGET``).
    evidence_budget: int = TRAJECTORY_BUDGET
    #: The whole request's room in characters (``PROPOSAL_ROOM``; the proposer takes its
    #: system prompt and closing line off it). ``render`` keeps within it by showing
    #: fewer items whole; 0 is no room (every picked item shown, whatever their size).
    request_room: int = PROPOSAL_ROOM
    #: The most characters the items "Trajectories" shows whole take together when not
    #: every item fits (`_fit_shown`); 0 is no limit. ``render`` sets it from
    #: ``request_room`` when the request would pass that, never otherwise.
    shown_items_room: int = 0
    #: What the last ``render`` measured of its sections (``items``: the characters the
    #: items shown whole take), for ``render`` to fit them.
    fit_sizes: dict[str, int] = field(default_factory=dict, repr=False, compare=False)
    #: What one ``render`` works out once and reads again while it renders (its copies
    #: share it); emptied when it starts and when it ends.
    rendering: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    #: The latest children of the edited candidate with their parents' train records
    #: (``Loop._edit_pairs``), for "What the last edit changed".
    edit_pairs: list[dict[str, Any]] = field(default_factory=list)
    #: The optimizer's earlier reasoning beside outcomes (``Loop._reasoning_entries``).
    reasoning: list[dict[str, Any]] = field(default_factory=list)
    #: What readers found in the items, group by group, when they did not fit one
    #: call (``AutoresearchOptimizer``); ``None`` when no reader went through them.
    reader_findings: list[dict[str, Any]] | None = None
    #: The groups readers could not read, with why.
    reader_failures: list[dict[str, Any]] = field(default_factory=list)
    #: The proposal attempt within this generation (``Loop.propose`` and
    #: ``Optimizer.propose_valid`` count from 1).
    attempt: int = 0
    #: How many edits one proposal may bundle (``Proposal.edits``;
    #: ``MAX_EDITS_PER_PROPOSAL`` in a run). 1 asks for single edits only, and the
    #: request does not mention bundling.
    max_edits: int = 1
    #: The run's size cost per node (``LoopConfig.parsimony``), stated in the answer
    #: rules with what it means in score units; ``None`` says nothing about it.
    parsimony: float | None = None
    #: Whether the score is the mean of the items' scores (``Loop`` checks every train
    #: evaluation); only then does the size cost line compare a node with one item.
    score_is_item_mean: bool = False
    #: How many proposals this request asks for (``AutoresearchOptimizer`` sets it to
    #: its ``n``); 0 when not said. With 1, only one candidate is evaluated, so the
    #: bundling advice asks for one proposal covering the whole failure path instead
    #: of separate proposals that would never run.
    proposals: int = 0
    #: How many validation items the gate's score is the mean of (``Loop``: the val
    #: part's size, under ``val_gate`` when the score is the items' mean); 0 when not
    #: known. The gate note then says what one validation item is worth.
    val_items: int = 0
    #: The seed's knob values (``Loop``: the population's root candidate), so the
    #: history can show a ``set_knob`` as a diff against the value it replaced; empty
    #: when not known (the first change to a text knob then shows its length and head).
    seed_knob_values: dict[str, Any] = field(default_factory=dict)
    #: Whether this run's edits may add code nodes (``LoopConfig.allow_code_nodes`` and a
    #: machine that can sandbox them, `refactory_dsh.snippet.available`); only then does
    #: the request describe them.
    code_nodes: bool = False
    #: The candidate whose pipeline "The pipeline you may edit" describes, and so the
    #: one a proposal that names no parent is applied to (``Loop._edited_id``); ``None``
    #: when not known.
    parent: str | None = None
    #: The outer loop's name (``Loop.name``), said beside its decision; empty for none.
    strategy_name: str = ""
    #: The edit levels this request asks for, one proposal each (``EDIT_LEVELS``;
    #: ``LoopConfig.portfolio``, or one level a strategy decided); empty asks for none
    #: in particular.
    edit_levels: list[str] = field(default_factory=list)
    #: The validation gate's score of each candidate it judged, by id, under the
    #: default loop's ``val_gate``; empty otherwise. For a parent policy to rank by
    #: what the gate ranks by (``HillClimbSelector``). Never rendered: the model's
    #: request shows gate decisions, not val scores.
    val_scores: dict[str, float] = field(default_factory=dict)
    #: Each judged candidate's per-item validation scores, by candidate id, under the
    #: default loop's ``val_gate`` and for a parent policy that asks for them
    #: (``ParentSelector.uses_item_scores``); empty otherwise, and always empty in the
    #: context a proposer is handed. Each inner map is
    #: ``Loop.part_item_scores(id, "val")``: item keys are a per-run HMAC of the item id
    #: (`loop.opaque_item_key`), the same keys ``Run.part_item_scores`` gives a
    #: strategy, so a parent policy can line candidates up item by item (a per-item
    #: Pareto frontier) without learning an item id. Validation only, never test, and
    #: never rendered.
    val_item_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    #: How the validation gate selects (``Loop.selection_mode``): ``single``, ``pool``,
    #: or empty without the gate. Under ``pool`` the gate note says what the pool is.
    selection: str = ""
    #: Under ``pool``, whether a population parent picker (``archive``, ``evolutionary``,
    #: ``islands``, ``map_elites``, ``pareto``) chooses the parent from every judged candidate
    #: rather than from the pool; the gate note says which (``_gate_note``).
    population_parents: bool = False
    #: What is there before each node of the edited graph, on every path that reaches
    #: it (`available_before`): ``{node: {"facts": [...], "files": [...], "made_by":
    #: {name: [step ids]}, "task_inputs": [...]}}``. Rendered under "The pipeline you may
    #: edit", so a prompt's ``{{name}}`` names a fact that exists there (rule V15) rather
    #: than a file name, made by the step meant; empty says nothing.
    available: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Wiring of the edited graph that runs but is likely not what was meant
    #: (`nn.validate.wiring_warnings`), said beside its edges; never a refusal.
    wiring_warnings: list[str] = field(default_factory=list)
    #: The machine's code-node sandbox answer (`snippet.available`) this request was
    #: written with: ``code_nodes`` offers by it and the refusal of a code node
    #: (``E-code-nodes``) refuses by it, so the two agree within a request; ``None``
    #: when not taken (the run does not allow code nodes). Never rendered.
    sandbox: tuple[bool, str] | None = None
    #: Whether a person answers gates while this run's candidates run. An outer loop's
    #: candidates run headless, so the default leaves the person-answered gate kind out
    #: of the node schemas (`node_schema.render_schema`).
    person_gates: bool = False
    #: Every evaluated candidate of the run with what its own edits did, its train
    #: fixed/broken counts against its parent and the gate's word (``Loop._ledger``,
    #: ``LoopConfig.run_ledger``), for "Every change so far"; empty says nothing.
    ledger: list[dict[str, Any]] = field(default_factory=list)
    #: The loop's memory note (`optim.memory`; ``LoopConfig.lessons``): ``{"text",
    #: "epoch", ...}`` and, when the epoch's pass failed, ``not_updated``; for
    #: "Optimizer memory" here and in every analyst and merge call. Empty says nothing.
    memory: dict[str, Any] = field(default_factory=dict)
    #: The steps the last round's candidates added, with their train evidence
    #: (``feedback.step_evidence``; ``LoopConfig.step_review``) and, once a model
    #: optimizer has reviewed them, each step's ``review`` or ``review_error``, for "Did
    #: the added steps contribute?".
    added_steps: list[dict[str, Any]] = field(default_factory=list)
    #: Whether "How nodes compose" says what makes a new step pay off
    #: (``LoopConfig.structure_guidance``; ``feedback.STRUCTURE_GUIDANCE``).
    structure_guidance: bool = False
    #: The analyst stage's settings (``LoopConfig.analysts``; `optim.analysts.settings`):
    #: ``{"group_size", "merge_batch_size", "max_batches"}``; empty is off. A model
    #: proposer then runs failure and success analysts over ``item_records`` and merges
    #: their findings before it writes the proposal. Never rendered itself.
    analysts: dict[str, Any] = field(default_factory=dict)
    #: The round's analysis (`AutoresearchOptimizer._analyse`): the batches, the merges
    #: and the ranked ``findings``, for "What the analysts found, ranked"; ``None`` when
    #: no analyst ran.
    analysis: dict[str, Any] | None = None
    #: The integrity judge's verdicts on this run's proposals (``Loop._integrity``,
    #: ``LoopConfig.integrity_judge``), for "Integrity review of your proposals"; empty
    #: (the judge off) says nothing.
    integrity: list[dict[str, Any]] = field(default_factory=list)
    #: The latest refused proposals as written (``{"edit", "reasons"}``, at most
    #: ``REFUSED_SHOWN``), shown under "Why your last proposals were refused" so a retry
    #: fixes the proposal it wrote rather than writing one from scratch.
    refused: list[dict[str, Any]] = field(default_factory=list)
    #: What slim repairs dropped or changed of recent proposals they turned into
    #: candidates (``{"candidate", "dropped"}``; `Rejection.dropped`), one line each
    #: (`repair_loss_lines`), so the proposer knows that text did not land.
    repair_losses: list[dict[str, Any]] = field(default_factory=list)
    #: Whether a new step's tools can run sealed on this machine (`workspace.sealed_available`);
    #: only then do the compositions show a tool step.
    new_node_tools: bool = False
    #: What a new agent step may choose for its own route, field by field
    #: (`workers.StepRoutes.offered`); empty offers no route, and the request says none.
    step_routes: dict[str, list[str]] = field(default_factory=dict)
    #: Time, in numbers, for "Time per item" (`time_lines`): ``per_item_median_s``/
    #: ``per_item_p90_s``/``per_item_max_s`` (the edited candidate's items' own times,
    #: `loop.item_time_spread`); empty says nothing.
    time_facts: dict[str, Any] = field(default_factory=dict)
    #: Proposals already made earlier in this round (each: ``{"edits", "rationale",
    #: "parent"}``), so the model does not duplicate them. Shown as facts, not
    #: instructions.
    prior_proposals: list[dict[str, Any]] = field(default_factory=list)

    def _settled(self) -> frozenset[str]:
        """The lines an item's step does not repeat (`settled_for`) over this request's
        items, the edited candidate's children included; within one ``render``, worked
        out once (``fit_sizes`` aside, nothing it reads changes while it renders)."""
        held = self.rendering.get("settled")
        if held is None:
            held = settled_for(self.knobs, self.node_states,
                               [*self.item_records,
                                *(r for pair in self.edit_pairs or ()
                                  for side in ("parent_records", "child_records")
                                  for r in pair.get(side) or ())])
            if self.rendering.get("on"):
                self.rendering["settled"] = held
        return held

    def _edit_change(self) -> list[str]:
        """"What the last edit changed" (`_edit_change_lines`), naming a reply
        "Trajectories" shows whole by where it is."""
        from .edits import _edit_change_lines, knob_homes
        from .items import TRAJECTORIES_HEADING, trajectories_whole
        shown: dict[str, str] = {}
        if self.item_records:
            whole = trajectories_whole(
                self.item_records, budget=self.evidence_budget, findings=self.reader_findings,
                analysed=(None if self.analysis is None
                          else list(self.analysis.get("findings") or ())),
                shown_room=self.shown_items_room, settled=self._settled())
            for record in whole:
                for exchange in record.get("exchanges") or ():
                    reply = str(exchange.get("response") or "").strip()
                    if reply:
                        shown.setdefault(reply, f"under \"{TRAJECTORIES_HEADING[3:]}\", item "
                                                f"{record.get('id')}, step `{exchange.get('node')}`")
        return _edit_change_lines(self.edit_pairs, homes=knob_homes(self.knobs), shown=shown)

    def _leaderboard_lines(self) -> list[str]:
        """The leaderboard: id, parent, score, structural, gate and cost. A candidate's edits
        are in "What this run has already tried", one line each, so they are not repeated
        here; only a candidate that list leaves out (it was truncated) has its edits here."""
        from .edits import _gate_column, compact_edits
        from .objective import _cost_cell, _table, fmt_score
        listed = {c.get("id") for c in self.history}
        missing = [c for c in self.leaderboard if c.get("id") not in listed]
        gate = _gate_column(self.leaderboard)
        return [_table(
            ["id", "parent", "score", "structural", *(["edits"] if missing else []), *gate,
             *(["cost per item"] if self.parsimony else [])],
            [[c["id"], c.get("parent"), fmt_score(c.get("score")), c.get("structural"),
              *([compact_edits(c.get("edits"), self.seed_knob_values)
                 if c.get("id") not in listed else "(below)"] if missing else []),
              *([c.get("gate", "")] if gate else []),
              *([_cost_cell(c.get("cost_per_item"))] if self.parsimony else [])]
             for c in self.leaderboard])]

    def _tried_lines(self) -> list[str]:
        """What this run has tried and how each attempt did: the recent results, the
        traces' numbers, the last edit's children and the reasoning behind recent
        candidates (each with what its edit built), every candidate so far, the run-wide
        feedback, earlier runs and the latest refusals.

        Each long text the optimizer wrote is whole in one place: a knob's current text in
        "Each step's settings" (its exact patch target), the last edit's children's edits
        and predictions under "What the last edit changed", the other recent candidates'
        edits under "Your earlier reasoning". The attempts list, the evidence table and the
        other sections name it and point there."""
        from ..refusals import refused_lines, repair_loss_lines
        from .edits import _attempt_lines, _current_line, _gate_column, _gate_note, _reasoning_lines, knob_homes, reasoning_shown
        from .objective import UNSCORED_TRACE_KEYS, _cost_note, _table, _trace_rows, fmt_score
        below = [str(p.get("child")) for p in self.edit_pairs or ()]
        elsewhere = {
            **{str(e.get("id")): "Your earlier reasoning"
               for e in reasoning_shown(self.reasoning or [])},
            **{child: "What the last edit changed" for child in below}}
        homes = knob_homes(self.knobs)
        lines = [
            "## Recent results",
            *_gate_note(self.leaderboard, self.history, val_items=self.val_items,
                        selection=self.selection, population=self.population_parents),
            *_cost_note(self.parsimony),
            *self._leaderboard_lines(),
            "",
            "## What this run has already tried",
            (
                f"{len(self.history)} candidate(s), best "
                f"{fmt_score(max((c.get('score') or 0.0) for c in self.history))}"
                if self.history and not _gate_column(self.history)
                else (f"{len(self.history)} candidate(s); "
                      f"{_current_line(self.history, self.selection)}")
                if self.history
                else "(nothing evaluated yet)"
            )
            + (
                " — older candidates omitted, so do not assume the list is complete"
                if self.history_truncated
                else ""
            )
            + (". One line each, oldest first: the edits it added to its parent, what they "
               "built (its first pipeline line, `+` new or `~` changed), its train score "
               "beside its parent's, and the gate's decision." if self.history else ""),
            *_attempt_lines(self.history, self.seed_knob_values, below, elsewhere, homes),
            "",
            "## Evidence from the traces",
            _table(["what", "value"], _evidence_rows(_trace_rows(
                self.traces, omit=UNSCORED_TRACE_KEYS
                if self.parsimony == 0 and self.score_is_item_mean else (),
                current={k.get("name"): k.get("value") for k in self.knobs},
                seed=self.seed_knob_values),
                [] if self.history_truncated else self.history)),
        ]
        if self.edit_pairs:
            lines += ["", *self._edit_change()]
        if self.reasoning:
            predicted = [str(p.get("child")) for p in self.edit_pairs or () if p.get("prediction")]
            lines += ["", *_reasoning_lines(
                self.reasoning, below, predicted, homes,
                reviewed={str(e.get("candidate")) for e in self.added_steps or ()})]
        lines += _feedback_sections(self)
        if self.prior_runs:
            lines += [
                "",
                "## Previous runs of this mode",
                "What earlier searches already settled, so it does not have to be "
                "rediscovered:",
                _table(
                    ["path", "optimizer", "seed", "best", "bound", "best edits"],
                    [
                        [r.get("path"), r.get("optimizer"), fmt_score(r.get("seed_score")),
                         fmt_score(r.get("best_score")), r.get("bound"),
                         json.dumps(r.get("best_edits"), default=str)]
                        for r in self.prior_runs
                    ],
                ),
            ]
        if self.rejections:
            lines += [
                "",
                "## Why your last proposals were refused",
                "A proposal that breaks one of these is not evaluated at all. It costs "
                "nothing but the answer, so fix the stated reason rather than guessing.",
                _table(["reasons"], [[r] for r in self.rejections[-12:]]),
                *refused_lines(self.refused, homes),
                *repair_loss_lines(self.repair_losses),
            ]
        elif self.repair_losses:
            lines += ["", "## Your repaired proposals", *repair_loss_lines(self.repair_losses)]
        return lines

    def render(self) -> str:
        """The request, as text, for a model optimizer: the score, the pipeline (once, as
        the view), what this run has tried, what the analysts found, the edits it may
        make, the items, then the reference, and the answer's form last. Fills
        ``omissions`` with any text it had to leave out of an item.

        Within ``request_room``, whatever the items' sizes: a request that would pass it
        is made again with the items "Some items in full" shows (``shown_items_room``,
        `_fit_shown`) given what the rest of the request leaves. Items stay whole while
        they fit, in rank order, the rest named; the first, when it alone passes the room,
        has its longest texts cut to fit, or is named. Each cut is marked in place and in
        ``omissions``. Nothing else gives way: the rest of the request is never cut.

        What one render reads again is worked out once (``rendering``)."""
        from .. import evidence

        self.rendering.clear()
        self.rendering["on"] = True
        try:
            with evidence.keeping():
                return self._render_fitted()
        finally:
            self.rendering.clear()

    def _render_fitted(self) -> str:
        """`render`'s work: the request, fitted to ``request_room`` by showing fewer items
        whole (``shown_items_room``); the rest of the request is never cut."""
        from .items import _ordered_items
        text = self._render_request()
        if (not self.request_room or self.shown_items_room
                or len(text) <= self.request_room):
            return text
        items = _ordered_items(self.item_records or ())
        if not items or not self.fit_sizes.get("items"):
            return text  # nothing of the items' texts can give way
        # What the items shown whole may take is what the rest of the request leaves; the
        # rest changes a little with what is shown (a pointer to an item shown whole), so
        # the room is made again from the last render, a few times.
        room = self.request_room - (len(text) - self.fit_sizes.get("items", 0))
        for _ in range(5):
            text = replace(self, shown_items_room=max(1, room))._render_request()
            over = len(text) - self.request_room
            if over <= 0:
                break
            room -= over
        return text

    def _render_request(self) -> str:
        """`render`'s request at the context's ``shown_items_room``."""
        from .composition import REFERENCE_HEADING, REFERENCE_POINTER, _blocks_section, _fit_state, _item_count, composition_section, operation_lines, pipeline_view_lines
        from .edits import _available_section
        from .items import _item_evidence, _trajectories
        from .objective import _rules_text, _table, answer_instructions, bundle_instructions, fmt_score, objective_lines, time_lines
        self.omissions.clear()
        self.fit_sizes.clear()
        sections = [
            f"# Update request — generation {self.generation}, epoch {self.epoch}",
            "",
            f"Mode: `{self.mode_name}`. "
            + (f"Best score so far: {fmt_score(self.best_score)}. " if not self.selection else
               # Under checkpoint the current candidate's: which one is best on val only
               # picks the winner, and is never said.
               f"Train score of the current candidate, the one being trained: "
               f"{fmt_score(self.best_score)}. " if self.selection == "checkpoint" else
               f"Train score of the candidate the validation gate ranks first: "
               f"{fmt_score(self.best_score)}. ")
            + (f"Mean over the {self.mean_over} candidate(s) scored so far: "
               f"{fmt_score(self.mean_score)}."
             if self.mean_over else f"Mean score so far: {fmt_score(self.mean_score)}."),
            "",
            *objective_lines(self.parsimony, item_mean=self.score_is_item_mean,
                             gated=self.selection),
            *time_lines(self.time_facts),
            *([self.steering, ""] if self.steering.strip() else []),
            *_strategy_section(self.strategy, self.strategy_name),
            # 1-2: the score (above), then the pipeline once, in the view's lines.
            "## The pipeline you may edit",
            _levers_line(self.catalogue),
            *([f"Note: {self.graph_note}"] if self.graph_note else []),
            *([f"Protected nodes: {', '.join(f'`{n}`' for n in self.protected)}. An edit may "
               "not remove them, change their declarations or out-edges, move a knob they "
               "read, or let a run finish without them; a new node may go in front of one."]
              if self.protected else []),
            *pipeline_view_lines(self.node_states, self.edges, self.knobs, self.protected,
                                 self.graders),
            *(["Wiring warnings (the graph runs, but likely not as meant; not refusals):",
                *(f"- {w}" for w in self.wiring_warnings)] if self.wiring_warnings else []),
            REFERENCE_POINTER,
            "",
            # 3: what this run has tried, and how each attempt did.
            *self._tried_lines(),
        ]
        # 4: what the analysts found, failures and successes.
        if self.analysis is not None:
            from ..analysts import section_lines
            sections += ["", *section_lines(self.analysis)]
        # 5: the steps' settings, what the author says, the blocks it may insert.
        sections += [
            "",
            *_settings_lines(self.knobs),
            "## About this mode",
            self.guidance or "(the mode's author offered no guidance)",
            (
                "Operations the author suggests trying first: "
                + ", ".join(f"`{name}`" for name in self.preferred_edits)
                if self.preferred_edits
                else ""
            ),
            "",
            *(["`score_effect: none` on a block means inserting it cannot change the "
               "score. It is available and it may be worth having, but it is not a way "
               "to improve the number.", ""] if self.insertable else []),
            *_blocks_section(self.insertable, self.catalogue),
        ]
        # 6: the items themselves, chosen rather than clipped.
        if self.item_records:
            sections += ["", *_item_evidence(self.item_records)]
            sections += ["", *_trajectories(
                self.item_records, budget=self.evidence_budget,
                findings=self.reader_findings, failures=self.reader_failures,
                analysed=(None if self.analysis is None
                          else list(self.analysis.get("findings") or ())),
                omissions=self.omissions, settled=self._settled(),
                shown_room=self.shown_items_room, sizes=self.fit_sizes)]
        if self.prior_proposals:
            sections += ["", *_prior_proposals_lines(self.prior_proposals)]
        # 7: the reference, last and once: the operations, how nodes compose, what each
        # node has before it runs, each node in full with the schemas, the rules.
        sections += [
            "",
            REFERENCE_HEADING,
            "Everything an edit is written from, each said once: the operations, how nodes "
            "compose, what each node has before it runs, each node in full with the node "
            "schema, and the rules the validator checks (the pipeline view above is the "
            "graph's steps and edges).",
            "",
            "### The operations you may use",
            *operation_lines(self.catalogue, self.node_states, self.edges,
                             protected=self.protected,
                             knob_values={k.get("name"): k.get("value") for k in self.knobs},
                             step_routes=self.step_routes, tools=self.new_node_tools,
                             code_nodes=self.code_nodes),
            "",
            *composition_section(self.catalogue, self.node_states, self.edges,
                                 code_nodes=self.code_nodes,
                                 text_knobs=[k.get("name") for k in self.knobs
                                             if k.get("kind") == "str"],
                                 structure_guidance=self.structure_guidance,
                                 tools=self.new_node_tools, protected=self.protected,
                                 knob_values={k.get("name"): k.get("value")
                                              for k in self.knobs},
                                 frozen=[k.get("name") for k in self.knobs if k.get("frozen")]),
            # Only when the view could not be drawn are the nodes and edges listed here.
            *([] if _fit_state(self.node_states, self.edges, self.knobs) is not None else [
                "The pipeline view could not be drawn; the graph as it declares itself:",
                "Nodes:",
                _table(["id", "kind", "call", "knobs", "notes"],
                       [[n["id"], n["kind"], n.get("call", ""), n.get("knobs", []),
                         n.get("notes", "")] for n in self.nodes]),
                "Edges:",
                _table(["source", "target", "condition"],
                       [[e["source"], e["target"], e.get("condition")] for e in self.edges])]),
            *_available_section(self.available, code_nodes=self.code_nodes),
            "",
            *(["### What this graph allows",
               "Worked out on the pipeline you edit, by an attempt of each edit: what a new "
               "step's `sees` can name, and which removals, replacements and merges are "
               "refused, with why; one that could not be worked out says so. An edit not "
               "listed for a step was accepted.", *self.graph_facts, ""] if self.graph_facts else []),
            *_node_reference(self),
            "### The rules the validator checks",
            "Every proposal is checked against these rules before anything runs; they are "
            "the validator's own, generated from it. One that breaks a rule comes back with "
            "the rule's code and the reason, so it can be fixed in a later answer, and it "
            "costs nothing but the answer.",
            _rules_text(),
        ]
        sections += ["", "## Answer", answer_instructions(
            self.catalogue, parsimony=self.parsimony,
            items=_item_count(self.item_records) if self.score_is_item_mean else 0,
            code_nodes=self.code_nodes)]
        if self.max_edits > 1:
            sections += ["", bundle_instructions(self.max_edits, self.catalogue,
                                                 proposals=self.proposals)]
        if self.edit_levels:
            sections += ["", *levels_instructions(self.edit_levels, code_nodes=self.code_nodes)]
        return "\n".join(sections)


def _levers_line(catalogue: Sequence[dict[str, Any]]) -> str:
    """What the request lets the optimizer change, said before anything else about the
    pipeline: its structure and its steps' settings, as equal levers when the run permits
    structural operations; the settings only when it does not."""
    from .composition import structural_ops
    if not structural_ops(catalogue):
        return ("This run permits changes to the steps' settings only (their prompts and "
                "parameters, under \"Each step's settings\"); the structure below stays as it "
                "is.")
    return ("You may change the pipeline's structure -- its steps and how they connect, "
            "with the operations in the reference at the end -- and each step's settings -- "
            "its prompts and parameters, under \"Each step's settings\". Both are levers of "
            "the same standing: read where the pipeline fails first, then choose by the cause "
            "the evidence shows.")


def _exact_in_table(value: Any) -> bool:
    """Whether a table cell shows ``value`` as it is: a cell puts a text on one line and
    escapes ``|`` (`_cell`), so a text with either is shown whole below the table."""
    return not (isinstance(value, str) and ("\n" in value or "|" in value))


def _fence(text: str) -> str:
    """A code fence longer than any run of backticks in ``text``."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def _prior_proposals_lines(prior: Sequence[dict[str, Any]]) -> list[str]:
    """Facts about proposals already made earlier in this round."""
    from .edits import compact_edits
    lines = [
        "## Proposals already made in this round",
        f"{len(prior)} proposal(s) earlier in this round, each from a separate call. "
        "Do not duplicate them.",
    ]
    for index, entry in enumerate(prior, 1):
        edits = entry.get("edits") or []
        if not edits and entry.get("edit"):
            edits = [entry["edit"]]
        rationale = str(entry.get("rationale") or "")
        parent = str(entry.get("parent") or "")
        edit_summary = compact_edits(edits, {})
        lines.append(f"- Slot {index}: parent {parent}, edit {edit_summary}"
                     + (f" — {rationale}" if rationale else ""))
    return lines


def _settings_lines(knobs: Sequence[dict[str, Any]], text_chars: int | None = None,
                    exact: bool = True) -> list[str]:
    """"Each step's settings": every knob in a table; a text the table cannot show as it
    is (a line break or a ``|``) is shown there by reference and whole below it, fenced,
    so a patch can quote its target text exactly. ``text_chars`` clips each such text
    (head and tail, marked), for an analyst call that must fit a smaller room. Without
    ``exact`` (a reader that never patches: the analysts), a long line an earlier knob's
    text showed is said as where it was (`evidence.dedup_texts`), so the steps' shared
    paragraphs are read once."""
    from .items import _clip_middle
    from .objective import _table
    # Each text clipped once, here (a clipped one says how much was left out), whether
    # the table shows it or it is shown after the table.
    shown = {str(k["name"]): (_clip_middle(k["value"], text_chars)
                              if text_chars is not None and isinstance(k.get("value"), str)
                              else k.get("value")) for k in knobs}
    whole = [k for k in knobs if not _exact_in_table(shown[str(k["name"])])]
    lines = ["## Each step's settings",
             ("The prompts and parameters the steps read, which `set_knob` moves (a new step "
              "brings its own). A patch quotes its target text exactly: a value in the table "
              "is exact as shown"
              + ("; a text with line breaks or `|` is shown whole after the table." if whole
                 else ".")) if exact else
             ("The prompts and parameters the steps read, which `set_knob` moves; a text with "
              "line breaks or `|` is shown after the table, a line an earlier text showed "
              "said as where it is."),
             _table(["name", "value", "kind", "values", "range", "frozen", "reachable",
                     "description"],
                    [[k["name"], shown[str(k["name"])]
                      if _exact_in_table(shown[str(k["name"])])
                      else f"(shown {'whole ' if shown[str(k['name'])] == k['value'] else ''}"
                           f"below, {len(k['value'])} characters)",
                      k["kind"], k.get("values"), k.get("range"),
                      k["frozen"], k.get("reachable", True), k.get("description", "")]
                     for k in knobs])]
    texts = [shown[str(knob["name"])] for knob in whole]
    if not exact:
        from ..evidence import dedup_texts

        texts = dedup_texts(texts, [f"`{knob['name']}`" for knob in whole])
    from ..evidence import exact_block

    # Byte-exact on every path (`evidence.exact_block`): a fence no line of the text can
    # close, so its own headings read as none and nothing copied from it gains an escape.
    for knob, text in zip(whole, texts):
        lines += [f"`{knob['name']}`:", *exact_block(text)]
    return [*lines, ""]


def _node_reference(ctx: "ProposalContext") -> list[str]:
    """Each node of the edited graph in full and the node schemas (`_node_specs`), under
    their own heading; nothing when the request carries no node states."""
    from .edits import _node_specs
    specs = _node_specs(ctx.node_states, text_chars=ctx.node_spec_text_chars,
                        section_chars=ctx.node_spec_section_chars,
                        knob_values={k.get("name"): k.get("value") for k in ctx.knobs},
                        person_gates=ctx.person_gates, refer=True,
                        catalogue=ctx.catalogue)
    return [*specs, ""] if specs else []


def _count(value: Any) -> int:
    """A count from a trace, 0 when it is not a whole number."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def structural_text(stats: Mapping[str, Any]) -> str:
    """The run's structural changes so far (``Loop._structural_stats``) in plain words:
    how many candidates made one, how many did not, how many structural proposals the
    validator refused, and the operations used -- and, when none was tried, that none
    has been (a fact about this run, not a judgement of structure)."""
    structural = _count(stats.get("structural_candidates"))
    other = _count(stats.get("knob_candidates"))
    refused = _count(stats.get("structural_refusals"))
    if structural:
        text = (f"{structural} candidate(s) so far made a structural change and {other} "
                "did not")
    elif refused:
        text = ("No structural change has been evaluated in this run yet; every candidate "
                f"so far ({other}) keeps the starting pipeline's structure")
    else:
        text = ("No structural change has been tried in this run yet; every candidate so "
                f"far ({other}) keeps the starting pipeline's structure")
    if refused:
        text += (f". {refused} structural proposal(s) were refused by the validator before "
                 "running")
    ops = stats.get("accepted_ops")
    if isinstance(ops, Mapping) and ops:
        text += ". Operations in the candidates so far: " + ", ".join(
            f"`{op}` x{_count(n)}" for op, n in ops.items())
    return text + "."


def _evidence_rows(rows: Sequence[tuple[str, Any]],
                   history: Sequence[Mapping[str, Any]] = ()) -> list[list[str]]:
    """"Evidence from the traces" rows: what the run has done with structure first, in
    plain words (`structural_text`), then every other trace row as data. Before the
    loop's first gradient (a strategy's first request, the traces still empty) the
    structure row is counted from ``history``, every evaluated candidate, when it is
    whole (the caller passes none when it is truncated)."""
    out = [["structural changes", structural_text(value)] for key, value in rows
           if key == "structural" and isinstance(value, Mapping)]
    if not out and history:
        flags = [bool(c.get("structural")) for c in history if isinstance(c, Mapping)]
        if flags:
            out = [["structural changes", structural_text(
                {"structural_candidates": sum(flags),
                 "knob_candidates": len(flags) - sum(flags)})]]
    return out + [[str(key), json.dumps(value, default=str)] for key, value in rows
                  if not (key == "structural" and isinstance(value, Mapping))]


def _feedback_sections(ctx: "ProposalContext") -> list[str]:
    """The run-wide feedback sections (`optim.feedback`): every change so far, whether
    the last round's added steps contributed, the loop's memory note (`optim.memory`),
    and the integrity judge's verdicts on its proposals (`optim.integrity`); each only
    when its field holds something."""
    from .edits import reasoning_shown
    from .. import feedback, integrity, memory

    out: list[str] = []
    above = [str(e.get("id")) for e in reasoning_shown(ctx.reasoning or [])]
    for lines in (feedback.ledger_lines(ctx.ledger, in_full=above, selection=ctx.selection),
                  feedback.step_review_lines(ctx.added_steps),
                  memory.request_lines(ctx.memory),
                  integrity.feedback_lines(ctx.integrity)):
        if lines:
            out += ["", *lines]
    return out


#: The edit levels a portfolio round asks for, one proposal each (``edit_level``).
EDIT_LEVELS = ("prompt_knob", "pipeline")


_LEVEL_TEXT = {
    "prompt_knob": "a prompt/knob-level change: only `set_knob` edits (patch or rewrite a "
                   "prompt, move a setting) on the current pipeline",
    "pipeline": "a pipeline-level change: at least one structural operation (add, remove, "
                "rewire, gate, loop or parallelise steps{code}), with any knob moves it needs",
}


def edit_level(steps: Sequence[Any]) -> str:
    """``prompt_knob`` when every edit is a ``set_knob``, else ``pipeline``."""
    ops = [step.get("op") if isinstance(step, dict) else None for step in steps]
    return "prompt_knob" if ops and all(op == "set_knob" for op in ops) else "pipeline"


#: What a pipeline-level ask adds when the run allows code steps (``code_nodes``).
_CODE_STEP_TEXT = (", or add a code step that checks or extracts something "
                   "deterministically, at no model call")


def _level_text(level: str, code_nodes: bool) -> str:
    return _LEVEL_TEXT.get(level, level).replace("{code}", _CODE_STEP_TEXT if code_nodes else "")


def levels_instructions(levels: Sequence[str], *, code_nodes: bool = False) -> list[str]:
    """What the request asks for when it names edit levels; the pipeline level names code
    steps among its moves when the run allows them (``code_nodes``)."""
    if len(levels) == 1:
        return ["## The kind of change this round",
                f"Propose exactly one update: {_level_text(levels[0], code_nodes)}."]
    return ["## The kind of change this round",
            f"Propose exactly {len(levels)} updates, one per level, each your best of its "
            "kind:",
            *[f"- {_level_text(level, code_nodes)}" for level in levels],
            "The search evaluates every one of them on validation and keeps what wins; do "
            "not choose between them, and do not leave one out."]


#: Keys of ``ProposalContext.strategy`` that are not a decision of the outer loop: the
#: run's resolved search contract rides in the same dict and is not rendered here.
_NOT_DECISIONS = ("search_contract",)


#: Decision keys said first, as sentences, in this order.
_DECISION_LEAD = ("purpose", "why")


def _strategy_section(decision: Mapping[str, Any] | None, strategy: str = "") -> list[str]:
    """What the outer loop decided for this round, as data: its purpose and reason
    first, then every other key it published. Empty when it decided nothing (the
    default loop), so a request without a decision is unchanged."""
    from .edits import _bounded
    shown = {key: value for key, value in (decision or {}).items()
             if key not in _NOT_DECISIONS and value not in (None, "", [], {})}
    if not shown:
        return []
    who = f"The search strategy (`{strategy}`)" if strategy else "The search strategy"
    lines = ["## What this round is for",
             f"{who} decided the following for this round. It decides where the search "
             "goes; you decide what to write. Take it into account, and say in your "
             "rationale if the evidence argues against it."]
    for key in (*[k for k in _DECISION_LEAD if k in shown],
                *[k for k in shown if k not in _DECISION_LEAD]):
        value = shown[key]
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        lines.append(f"- {key}: {_bounded(text, 600)}")
    return [*lines, ""]
