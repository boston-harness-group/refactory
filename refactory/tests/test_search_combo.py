"""Where to search and what to write, combined freely (Python walker loop).

A strategy or a parent picker decides where the search goes; the proposer (random or
the model) decides what to write; the validation gate decides what is kept. These run
every registered strategy with both proposers and every parent picker with the model
proposer, end to end on a toy mode with a scripted model (no paid calls), and assert
what each combination does to the search, not that a branch ran:

- every proposal is applied to the candidate the model's request described;
- a picker's choice is the candidate the request describes and the child's parent;
- with ``map_elites`` a structurally new candidate holds its own shape group and is
  edited again later;
- the strategy's decision is in the request text, and the reflective edit-kind
  decision follows the evidence;
- the SkillOpt-matched default (one lineage, model proposer, ``val_gate``) produces
  the leaderboard it produced before this change.
"""
from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path
from typing import Any, Callable

import pytest

from refactory_dsh import nn, snippet, strategies
from refactory_dsh import task as task_module
from refactory_dsh.loop import Loop, LoopConfig, LoopError
from refactory_dsh import optim
from refactory_dsh.optim import Candidate

# ── the toy task and mode ──────────────────────────────────────────────────

TASK_REF = "search_combo_task"
TRAIN = [f"t{i}" for i in range(1, 7)]
VAL = [f"v{i}" for i in range(1, 5)]
TEST = [f"x{i}" for i in range(1, 4)]
SEED_PROMPT = "You answer items."


class ComboTask(task_module.Task):
    """An item passes when the solve step's answer says ``OK:<item id>``. The solve step
    answers so when its system prompt holds the item's class, ``fix:c<number>``: ``t2``,
    ``v2`` and ``x2`` share ``c2``, so a fix learned on train items moves val too."""

    name = TASK_REF

    def instances(self, *, part: str | None = None, seed: int | None = None):  # type: ignore[no-untyped-def]
        return [task_module.TaskInstance(id=item) for item in TRAIN + VAL + TEST]

    def verify(self, instance, workspace, *, part=None, seed=None):  # type: ignore[no-untyped-def]
        answer = workspace / "answer.txt"
        text = answer.read_text() if answer.exists() else ""
        passed = f"OK:{instance.id}" in text
        return task_module.VerifyResult(passed=passed, score=1.0 if passed else 0.0)


task_module.register_task(TASK_REF, ComboTask)


class Combo(nn.Mode):
    """A data pass: ``prep`` (op) then ``solve`` (agent, writes answer.txt)."""

    name = "combo"

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node("data", "data", subgraph_entry="prep", subgraph_exit="solve",
                          task_ref=TASK_REF, writes=("current_item.json",),
                          produces=("item_scores",)))
        graph.add(nn.Node("prep", "op", call="combo_ops:prep", reads=("current_item.json",),
                          produces=("question",)))
        graph.add(nn.Node("solve", "agent", prompt_template="Item {{question}}",
                          system_prompt_template=SEED_PROMPT, writes=("answer.txt",),
                          tool_policy="none"))
        graph.add(nn.Node("finish", "op", call="combo_ops:finish", produces=("done",)))
        graph.connect("prep", "solve")
        graph.connect("data", "finish")
        return graph

    def score(self, run: nn.RunResult) -> float:
        scores = [entry["score"] for entry in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0


@contextlib.contextmanager
def _combo_mode():
    from test_loop import _patch_mode

    with _patch_mode(Combo):
        yield


# ── the scripted model ─────────────────────────────────────────────────────

AGENT = {"op": "insert_after", "target": "solve",
         "node": {"id": "extra_check", "kind": "agent", "writes": ["extra_check.txt"],
                  "prompt_template": "Check item {{question}}",
                  "system_prompt_template": "You check answers."}}
PARALLEL = {"op": "parallel", "target": "solve", "branches": [
    {"id": "side_a", "kind": "agent", "writes": ["side_a.txt"],
     "prompt_template": "First angle on {{question}}"},
    {"id": "side_b", "kind": "agent", "writes": ["side_b.txt"],
     "prompt_template": "Second angle on {{question}}"}]}
CODE = {"op": "insert_after", "target": "solve",
        "node": {"id": "digest", "kind": "op", "call": snippet.RUN, "inputs": ["question"],
                 "produces": ["digest_len"],
                 "code": ("def main(inp):\n"
                          "    return {'facts': {'digest_len': len(inp['facts']['question'])}}\n")}}
#: The knob each structural edit adds, which a later edit of that child moves.
NEW_KNOB = {"extra_check": "extra_check__system_prompt", "side_a": "side_a__system_prompt",
            "digest": "digest__code"}


def _section(text: str, heading: str) -> str:
    """One ``## `` section of a request, without its heading."""
    if f"## {heading}" not in text:
        return ""
    return text.split(f"## {heading}", 1)[1].split("\n## ", 1)[0]


class Request:
    """One proposal request, parsed into the parts these tests read."""

    def __init__(self, text: str, index: int) -> None:
        self.text = text
        self.index = index
        self.pipeline = _section(text, "The pipeline you may edit")
        self.knobs = _section(text, "Parameters you may move")
        self.decision = _section(text, "What this round is for")
        wanted = re.search(r"Propose at most (\d+) update", text)
        self.wanted = int(wanted.group(1)) if wanted else 1
        picked = re.search(r"chose candidate (\w+)", self.decision)
        self.picked = picked.group(1) if picked else None
        kinds = _section(text, "The kind of change this round")
        self.levels = [level for level, phrase in (("prompt_knob", "prompt/knob-level change"),
                                                   ("pipeline", "pipeline-level change"))
                       if phrase in kinds]

    def has_node(self, node_id: str) -> bool:
        return f"| {node_id} |" in self.pipeline


Script = Callable[["ComboModel", Request], list[dict[str, Any]]]


def fixer(model: "ComboModel", request: Request) -> list[dict[str, Any]]:
    """Each proposal teaches ``solve`` one more train item, by a patch that appends to
    the prompt the request showed: a lineage gains one item per generation only when
    every edit lands on the candidate the request described."""
    return [model.fix() for _ in range(request.wanted)]


def portfolio(model: "ComboModel", request: Request) -> list[dict[str, Any]]:
    """One proposal per level the request asks for: a fix for prompt/knob, a new agent
    step (plus a fix) for pipeline."""
    out = []
    for level in request.levels or ["prompt_knob"] * request.wanted:
        if level == "prompt_knob":
            out.append({**model.fix(), "level": level})
        else:
            new = {**AGENT, "node": {**AGENT["node"], "id": f"extra_{request.index}",
                                     "writes": [f"extra_{request.index}.txt"]}}
            out.append({"edits": [new, model.fix()["edit"]], "level": level,
                        "rationale": "a new step"})
    return out


class ComboModel:
    """The run's model: agent steps answer by their system prompt; the optimizer's
    proposal requests are answered by ``script``. Every request is kept."""

    def __init__(self, script: Script = fixer) -> None:
        self.script = script
        self.requests: list[Request] = []
        self.replies: list[list[dict[str, Any]]] = []
        self.fixes = 0
        self.node = ""

    # The client protocol the loop, the walker and the budget use.
    def for_node(self, node_id: str, role: str | None = None) -> "ComboModel":
        view = object.__new__(ComboModel)
        view.__dict__ = self.__dict__.copy()
        view.node = node_id
        view._root = getattr(self, "_root", self)
        return view

    def with_tool_policy(self, policy: str) -> "ComboModel":
        return self

    def with_wall(self, wall: Any) -> "ComboModel":
        return self

    def with_budget(self, budget: Any) -> "ComboModel":
        return self

    def context(self) -> dict[str, Any]:
        return {}

    def route(self) -> dict[str, Any]:
        return {"worker_backend": "dsh"}

    def fix(self) -> dict[str, Any]:
        root = getattr(self, "_root", self)
        item = f"c{root.fixes % len(TRAIN) + 1}"
        root.fixes += 1
        return {"edit": {"op": "set_knob", "name": "solve__system_prompt",
                         "patch": [{"kind": "append", "content": f" fix:{item}"}]},
                "rationale": f"teach {item}", "prediction": "one more item"}

    def complete(self, prompt: str, *, system: str = "", **_: Any) -> str:
        root = getattr(self, "_root", self)
        if prompt.startswith("# Update request"):
            request = Request(prompt, len(root.requests))
            root.requests.append(request)
            proposals = root.script(self, request)
            root.replies.append(proposals)
            return json.dumps({"diagnosis": "scripted", "proposals": proposals})
        if prompt.startswith("# Reading trajectories"):
            return json.dumps({"failure_patterns": [], "success_patterns": []})
        item = re.search(r"Item ([a-z])(\d+)", prompt)
        if item and f"fix:c{item.group(2)}" in (system or ""):
            return f"OK:{item.group(1)}{item.group(2)}"
        return "NO"


# ── helpers ────────────────────────────────────────────────────────────────


def _parts(root: Path) -> Path:
    path = root / "parts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"train": TRAIN, "val": VAL, "test": TEST}))
    return path


def _config(root: Path, **overrides: Any) -> LoopConfig:
    base: dict[str, Any] = dict(epochs=3, batch_size=2, optimizer="autoresearch", seed=11,
                                runs_root=root / "run", holdout=0.0, max_evaluations=60,
                                parsimony=0.01)
    base.update(overrides)
    if base.get("val_gate"):
        base.setdefault("parts_manifest", str(_parts(root)))
    return LoopConfig(**base)


def _run(name: str, root: Path, model: ComboModel, **overrides: Any) -> Loop:
    with _combo_mode():
        config = _config(root, strategy=name, **overrides)
        loop = strategies.create(name, "fake/combo", config, project=Path("."), model=model)
        loop.run()
    return loop


def _children(loop: Loop) -> list[Candidate]:
    """Candidates the search proposed (not the seed, not the starting sweep)."""
    sweep = {cid for cid, record in loop.proposals.items()
             if record.get("origin") == "seed_population"}
    return [c for c in loop.population if c.parent and c.id not in sweep]


def _fixed(candidate: Candidate) -> set[str]:
    value = str(((candidate.state.get("knobs") or {}).get("solve__system_prompt") or {})
                .get("value") or "")
    return set(re.findall(r"fix:(c\d)", value))


def _described_prompt(request: Request) -> set[str]:
    """The fix tokens of the solve prompt the request showed (the knob table)."""
    row = next((line for line in request.knobs.splitlines()
                if line.startswith("| solve__system_prompt |")), "")
    return set(re.findall(r"fix:(c\d)", row))


def _gated(loop: Loop) -> set[str]:
    return set(loop._gate_accepted) | set(loop._gate_rejected)


GATED_STRATEGIES = [n for n in strategies.names()
                    if n not in ("best_of_n", "autoscientists", "reflective")]
UNGATED_STRATEGIES = ["best_of_n", "autoscientists", "reflective"]


# ── 1. every strategy x {random, autoresearch} ─────────────────────────────


@pytest.mark.parametrize("name", strategies.names())
def test_every_strategy_runs_with_the_model_proposer_and_its_edits_score(tmp_path, name):
    """The model's edits are applied to the candidate the request showed, evaluated,
    and (under the gate) judged by it: the best candidate carries more than one of the
    model's fixes, so a later request edited an earlier model edit."""
    model = ComboModel()
    gated = name in GATED_STRATEGIES
    loop = _run(name, tmp_path, model, val_gate=gated, epochs=4)
    assert loop.optimizer.proposal_failures == [], loop.optimizer.proposal_failures
    assert model.requests, f"{name} never asked the model"
    children = [c for c in _children(loop) if c.evaluated]
    assert children, f"{name} evaluated nothing the model proposed"
    for child in children:
        parent = next(c for c in loop.population if c.id == child.parent)
        # Applied to the parent it names, and that parent is a pipeline a request showed.
        assert _fixed(parent) <= _fixed(child), (child.id, parent.id)
    if gated:
        assert {c.id for c in children} & _gated(loop), "the gate judged no model child"
    best = max(children, key=lambda c: (c.score or 0.0))
    seed = loop.population[0]
    assert (best.score or 0.0) > (seed.score if seed.score is not None else 0.0) or not seed.evaluated
    if name not in ("best_of_n", "autoscientists"):  # single rounds: depth one by design
        assert max(len(_fixed(c)) for c in children) >= 2, (
            f"{name}: no model edit was ever edited again")


@pytest.mark.parametrize("name", strategies.names())
def test_every_strategy_runs_with_the_random_proposer(tmp_path, name):
    """The random sampler under every strategy: children are made, evaluated and
    (where the gate runs) judged; no strategy asks a model it does not need."""
    model = ComboModel()
    gated = name in GATED_STRATEGIES
    loop = _run(name, tmp_path, model, optimizer="random", val_gate=gated, epochs=3)
    assert loop.optimizer.proposal_failures == []
    assert model.requests == [], "the random proposer sent a proposal request"
    children = [c for c in loop.population if c.parent and c.evaluated]
    assert children, f"{name} evaluated no child"
    for child in children:
        assert any(c.id == child.parent for c in loop.population)
    if gated:
        assert {c.id for c in children} & _gated(loop), "the gate judged no child"


@pytest.mark.parametrize("name", strategies.names())
def test_every_strategy_runs_with_the_portfolio(tmp_path, name):
    """``portfolio: prompt+pipeline`` under every strategy: every round that asks for
    levels gets one candidate per level asked, and both kinds are evaluated."""
    model = ComboModel(portfolio)
    gated = name in GATED_STRATEGIES
    loop = _run(name, tmp_path, model, val_gate=gated, epochs=3, batch_size=1,
                portfolio="prompt+pipeline")
    assert loop.optimizer.proposal_failures == []
    asked = [r for r in model.requests if r.levels]
    if name in ("best_of_n", "autoscientists"):
        return  # single sampling rounds of their own; the portfolio is the loop's step
    assert asked, f"{name}: no request asked for edit levels"
    levels = {loop.proposals[c.id].get("level") for c in _children(loop) if c.evaluated}
    assert {"prompt_knob", "pipeline"} <= levels, levels
    report = json.loads((loop.root / "report.json").read_text())["portfolio"]
    assert report["by_level"]["pipeline"]["evaluated"] >= 1


# ── the portfolio ──────────────────────────────────────────────────────────


def test_portfolio_evaluates_both_levels_under_the_gate_and_counts_the_extra(tmp_path):
    model = ComboModel(portfolio)
    loop = _run("default", tmp_path, model, val_gate=True, batch_size=1, epochs=3,
                portfolio="prompt+pipeline")
    for request in model.requests:
        assert request.levels == ["prompt_knob", "pipeline"] and request.wanted == 2
        assert "Propose exactly 2 updates, one per level" in request.text
    children = [c for c in _children(loop) if c.evaluated]
    by_level = {"prompt_knob": [], "pipeline": []}
    for child in children:
        by_level[loop.proposals[child.id]["level"]].append(child)
    assert by_level["pipeline"] and all(c.is_structural() for c in by_level["pipeline"])
    assert by_level["prompt_knob"] and not any(
        optim.pipeline_shape(c.state)["agents"] > 1 for c in by_level["prompt_knob"]
        if not c.is_structural())
    assert {c.id for c in children} <= _gated(loop), "a portfolio candidate skipped the gate"
    report = json.loads((loop.root / "report.json").read_text())
    rounds = report["portfolio"]["rounds"]
    assert rounds == len(model.requests)
    # Evaluations, not candidates: a train run plus each val (or test) occasion, for
    # every candidate beyond the batch size (1) of its round.
    def cost(cid: str) -> int:
        return 1 + sum((loop._occasion_counters.get(cid) or {}).values())

    per_round = report["portfolio"]["per_round"]
    extra = sum(cost(c["id"]) for entry in per_round for c in entry["candidates"][1:]
                if c["evaluated"])
    assert report["portfolio"]["extra_evaluations"] == extra >= 2 * (rounds - 1)
    for level, entry in report["portfolio"]["by_level"].items():
        assert entry["evaluations"] >= 2 * entry["evaluated"], (level, entry)
    # The pipeline candidate (one node more, so 0.01 lower) beat its parent on val but
    # lost to the prompt sibling accepted first: shadowed, and the report says so.
    shadowed = [c for entry in per_round for c in entry["candidates"] if c["shadowed"]]
    assert shadowed and all(c["level"] == "pipeline" and c["val"] > c["parent_val"]
                            for c in shadowed)
    text = (loop.root / "report.md").read_text()
    assert "## Portfolio" in text and "| shadowed |" in text


def test_the_portfolio_report_lists_candidates_in_the_order_the_gate_judged_them(
        tmp_path, monkeypatch):
    """Shadowing is read off the report's order, so that order must be the gate's: the
    candidates ``ValidationGate.judge_round`` is handed are recorded and compared round
    by round."""
    from refactory_dsh.outerloop import ValidationGate

    judged: list[str] = []
    original = ValidationGate.judge_round

    def recording(self, run, candidates, *args, **kwargs):  # type: ignore[no-untyped-def]
        judged.extend(candidate.id for candidate in candidates)
        return original(self, run, candidates, *args, **kwargs)

    monkeypatch.setattr(ValidationGate, "judge_round", recording)
    loop = _run("default", tmp_path, ComboModel(portfolio), val_gate=True, batch_size=1,
                epochs=3, portfolio="prompt+pipeline")
    report = json.loads((loop.root / "report.json").read_text())["portfolio"]
    listed = [c["id"] for entry in report["per_round"] for c in entry["candidates"]
              if c["gate"] in ("kept", "current", "rejected")]
    assert listed and listed == [cid for cid in judged if cid in set(listed)]
    for entry in report["per_round"]:
        levels = [c["level"] for c in entry["candidates"]]
        assert levels == ["prompt_knob", "pipeline"], levels  # the order proposed


def test_portfolio_refuses_a_second_candidate_of_one_level_and_asks_for_the_missing_one(tmp_path):
    def script(model: ComboModel, request: Request) -> list[dict[str, Any]]:
        if request.levels == ["pipeline"]:
            return portfolio(model, request)
        return [model.fix(), model.fix()]  # two prompt changes, no pipeline change

    model = ComboModel(script)
    loop = _run("default", tmp_path, model, batch_size=1, epochs=2, portfolio="prompt+pipeline")
    assert any(r.reasons and r.reasons[0].startswith("E-level") for r in loop.rejections)
    retry = [r for r in model.requests if r.levels == ["pipeline"]]
    assert retry and "Propose exactly one update: a pipeline-level change" in retry[0].text
    assert any(loop.proposals[c.id].get("level") == "pipeline" for c in _children(loop))


def test_a_strategy_may_ask_the_portfolio_for_one_level(tmp_path):
    """reflective decides the edit kind; under the portfolio that round asks for that
    level only, and evaluates one candidate."""
    model = ComboModel(portfolio)
    loop = _run("reflective", tmp_path, model, epochs=4, batch_size=1,
                portfolio="prompt+pipeline")
    single = [r for r in model.requests if len(r.levels) == 1]
    assert single, [r.levels for r in model.requests]
    # The settings-only merge round asks for that level alone: its candidates are not
    # refused as a second prompt/knob change until the attempts run out.
    assert loop.merges >= 1
    # A merge edit sets a knob to the other parent's value (the scripted model patches).
    assert not [r for r in loop.rejections if r.reasons[0].startswith("E-level")
                and isinstance(r.edit, dict) and "value" in r.edit]
    for request in single:
        kind = re.search(r"edit_kind: (\w+)", request.decision).group(1)
        assert request.levels == [kind] and request.wanted == 1


def test_portfolio_needs_the_model_proposer(tmp_path):
    with _combo_mode(), pytest.raises(LoopError, match="asks no model"):
        Loop("fake/combo", _config(tmp_path, optimizer="random", portfolio="prompt+pipeline"),
             project=Path("."), model=ComboModel())


def test_the_request_shows_one_example_per_composition():
    from refactory_dsh.nn import edits as nn_edits
    from refactory_dsh.optim.base import composition_section

    state = nn_edits.apply({"op": "insert_after", "target": "solve", "node": {
        "id": "post", "kind": "op", "call": "combo_ops:finish", "produces": ["post_done"]}},
        Combo().state_dict())  # solve then needs a plain out-edge for the examples
    nodes = [dict(n, id=i) for i, n in state["graph"]["nodes"].items()]
    text = "\n".join(composition_section(nn_edits.catalogue(), nodes, state["graph"]["edges"],
                                         code_nodes=True))
    examples = [json.loads(line.strip()) for line in text.splitlines() if line.startswith('  {"')]
    ops = [e.get("op") or e["edits"][0]["op"] for e in examples]
    assert ops.count("parallel") == 1 and "insert_after" in ops and "loop" in ops
    assert len(examples) == 4 and "none is the preferred shape" in text


# ── 2. every parent picker x autoresearch ──────────────────────────────────


@pytest.mark.parametrize("picker", ["archive", "evolutionary", "islands", "hill_climb",
                                    "map_elites"])
def test_every_picker_hands_its_choice_to_the_model_and_the_edit_lands_there(tmp_path, picker):
    """The request describes the picked candidate (its prompt in the knob table), says
    who picked it, and every child of that request has it as parent."""
    model = ComboModel()
    loop = _run("default", tmp_path, model, parent_picker=picker, epochs=4)
    assert loop.optimizer.name == f"autoresearch+{picker}"
    assert loop.optimizer.proposal_failures == []
    by_id = {c.id: c for c in loop.population}
    assert model.requests
    for request in model.requests:
        assert request.picked in by_id, request.decision
        assert f"{picker} chose candidate" in request.decision
        # The pipeline shown is the picked one's: its prompt, fix tokens and all.
        assert _described_prompt(request) == _fixed(by_id[request.picked])
    for child in _children(loop):
        record = loop.proposals[child.id]
        assert child.parent in {r.picked for r in model.requests}
        assert _fixed(by_id[child.parent]) <= _fixed(child), record
    assert max(len(_fixed(c)) for c in _children(loop)) >= 2


@pytest.mark.parametrize("picker", ["map_elites", ""])
def test_an_ungated_run_scores_its_last_proposals(tmp_path, picker):
    """Each epoch scores the batch the one before proposed, then proposes the next.
    Without the validation gate the last epoch's proposals were never scored, so a run
    of N epochs scored N-1 rounds; now they are scored before the run reports (as the
    gate does), with no extra proposal round, and charged like any evaluation."""
    model = ComboModel()
    loop = _run("default", tmp_path, model, parent_picker=picker, epochs=3)
    assert loop.bound == "epochs" and len(loop.epochs) == 3
    last = loop.population[-1]  # the last epoch's proposal
    assert last in _children(loop) and last.evaluated and last.exit_ok
    assert loop._unevaluated() == []
    assert loop.evaluations == len(loop.population)
    assert loop.rollouts == sum(len(loop._train_item_records[c.id]) for c in loop.population)


@pytest.mark.parametrize("name", strategies.names())
def test_no_strategy_leaves_its_last_round_unscored_without_the_gate(tmp_path, name):
    """Every strategy, ungated: a search that ends on its epoch limit scores what its
    last epoch proposed (strategies that drive their own epochs -- reflective,
    adaptive_meta, the curricula -- used to leave it), and each evaluation is counted."""
    model = ComboModel()
    loop = _run(name, tmp_path, model, epochs=4)
    assert loop.optimizer.proposal_failures == []
    if loop.epochs:
        assert loop.epochs[-1].accepted, f"{name}: the last epoch proposed nothing to score"
    assert all(c.evaluated for c in _children(loop)), f"{name} left a proposal unscored"
    unscored = set(loop._unevaluated())
    if name in ("best_of_n", "autoscientists"):
        # One round of samples against the seed, no epochs: the seed and the starting
        # sweep it does not evaluate stay unscored, as before; nothing it proposed does.
        sweep = {cid for cid, record in loop.proposals.items()
                 if record.get("origin") == "seed_population"}
        assert unscored <= sweep | {loop.population[0].id}
    else:
        assert unscored == set(), f"{name} left its last round unscored"
    assert loop.evaluations == len(loop.population) - len(unscored)
    assert loop.rollouts == sum(len(loop._train_item_records[c.id])
                                for c in loop.population if c.evaluated)


@pytest.mark.parametrize("name", GATED_STRATEGIES)
def test_no_strategy_leaves_its_last_round_ungated(tmp_path, name):
    """Every strategy the gate runs with: the last epoch's proposals are scored and
    judged by the gate before the run reports, as the default loop's are."""
    model = ComboModel()
    loop = _run(name, tmp_path, model, val_gate=True, epochs=4)
    assert loop.epochs[-1].accepted, f"{name}: the last epoch proposed nothing to score"
    assert loop._unevaluated() == [], f"{name} left its last round unscored"
    assert {c.id for c in loop.population} == _gated(loop), f"{name} left a candidate ungated"


def test_a_strategy_stopped_by_another_bound_is_not_finished(tmp_path):
    """adaptive_meta drives its own epochs and stops with "epochs" whatever ended them;
    the loop's recorded bound decides, so a run the evaluation cap ended skips the
    post-training evaluation instead of spending it."""
    model = ComboModel()
    loop = _run("adaptive_meta", tmp_path, model, val_gate=True, epochs=6, max_evaluations=5)
    assert len(loop.epochs) < 6 and loop.evaluations >= 5
    final = loop._strategy_state.get("final_evaluation") or {}
    assert (final.get("final_eval") or {}).get("status") == "skipped", final


@pytest.mark.parametrize("gated", [False, True])
def test_a_finished_curriculum_scores_its_last_stage(tmp_path, gated):
    """A curriculum that runs out of stages before the epoch limit ends normally too
    (``curriculum_complete``): its last stage's batch is scored (and, under the gate,
    judged)."""
    model = ComboModel()
    loop = _run("staged_curriculum", tmp_path, model, epochs=20, val_gate=gated)
    assert loop.bound == "curriculum_complete" and len(loop.epochs) == 8
    assert loop.epochs[-1].accepted
    assert loop._unevaluated() == []
    if gated:
        assert {c.id for c in loop.population} == _gated(loop)
    else:
        assert loop.evaluations == len(loop.population)


def test_a_population_picker_runs_under_the_validation_gate_from_the_pool(tmp_path):
    """No longer refused: under the gate a population picker defaults to the pool
    selection and picks, every step, a candidate the gate judged and kept in the pool."""
    model = ComboModel()
    loop = _run("default", tmp_path, model, parent_picker="map_elites", val_gate=True,
                batch_size=1, epochs=3)
    assert loop.selection_mode() == "pool"
    judged = set(loop._gate_accepted) | set(loop._gate_rejected)
    picked = [r.picked for r in model.requests if r.picked]
    assert picked and set(picked) <= judged


def test_hill_climb_picker_edits_the_gates_current_candidate(tmp_path):
    """Train and val disagree: ``fix:c5`` passes a train item and no val item, so its
    child beats the seed on train and the gate rejects it (equal on val). Nothing is
    kept, so the next step edits the gate's current candidate, the seed."""
    def script(model: ComboModel, request: Request) -> list[dict[str, Any]]:
        if request.index == 0:
            return [{"edit": {"op": "set_knob", "name": "solve__system_prompt",
                              "patch": [{"kind": "append", "content": " fix:c5"}]},
                     "rationale": "train only"}]
        return fixer(model, request)

    model = ComboModel(script)
    loop = _run("default", tmp_path, model, parent_picker="hill_climb", val_gate=True,
                batch_size=1, epochs=3)
    seed = loop.population[0]
    train_only = next(c for c in loop.population if _fixed(c) == {"c5"})
    assert (train_only.score or 0) > (seed.score or 0), "the toy does not disagree"
    assert train_only.id in loop._gate_rejected and not loop.kept
    assert model.requests[1].picked == seed.id, "edited the train leader, not the gate's"
    assert all(c.parent == seed.id for c in _children(loop)[:2])


def test_hill_climb_ranks_kept_candidates_by_validation_under_the_gate(tmp_path):
    """A kept candidate with the better train score but the worse val score is not the
    one hill_climb edits under the gate."""
    from refactory_dsh.optim.base import ProposalContext

    replies = [
        {"edit": {"op": "set_knob", "name": "solve__system_prompt",  # train 3/6, val 1/4
                  "value": SEED_PROMPT + " fix:c1 fix:c5 fix:c6"}, "rationale": "a"},
        {"edit": {"op": "set_knob", "name": "solve__system_prompt",  # train 2/6, val 2/4
                  "value": SEED_PROMPT + " fix:c1 fix:c2"}, "rationale": "b"},
    ]

    def script(model: ComboModel, request: Request) -> list[dict[str, Any]]:
        return [replies[request.index]] if request.index < len(replies) else []

    loop = _run("default", tmp_path, ComboModel(script), parent_picker="hill_climb",
                val_gate=True, batch_size=1, epochs=3)
    first = next(c for c in loop.population if _fixed(c) == {"c1", "c5", "c6"})
    second = next(c for c in loop.population if _fixed(c) == {"c1", "c2"})
    gate = loop._gate()
    assert first.id in loop._gate_accepted and gate.current_id == second.id
    assert (first.score or 0) > (second.score or 0), "the kept one must lead on train"
    loop.kept.append(first.id)
    parents = loop._gated_parents()
    assert {c.id for c in parents} == {first.id, second.id}
    ctx = ProposalContext(val_scores=loop._val_scores())
    assert loop.optimizer.picker.select(ctx, 1, parents)[0].id == second.id
    assert loop.optimizer.picker.select(ProposalContext(), 1, parents)[0].id == first.id


def test_a_strategy_named_parent_is_credited_to_the_strategy_not_the_picker(tmp_path):
    model = ComboModel()
    loop = _run("tree_search", tmp_path, model, parent_picker="map_elites", epochs=3)
    decided = [r for r in model.requests if "action:" in r.decision]
    assert decided
    for request in decided:
        assert "map_elites chose" not in request.decision
        assert "not used this round: the strategy chose candidate" in request.decision
        named = re.search(r"- parent: (\w+)", request.decision).group(1)
        assert _described_prompt(request) == _fixed(next(
            c for c in loop.population if c.id == named))


# ── 3. map_elites keeps a structural child in its own shape group ──────────


def _structural_script(structural: list[dict[str, Any]]) -> Script:
    def script(model: ComboModel, request: Request) -> list[dict[str, Any]]:
        for node_id, knob in NEW_KNOB.items():
            if request.has_node(node_id):
                # A structural child: edit its own new step.
                value = ("def main(inp):\n    return {'facts': {'digest_len': 2}}\n"
                         if knob.endswith("__code") else f"Edited again, request {request.index}.")
                return [{"edit": {"op": "set_knob", "name": knob, "value": value},
                         "rationale": f"edit {node_id} again"}]
        if request.index == 0:
            return [{"edit": edit, "rationale": "a new shape"} for edit in structural]
        return fixer(model, request)
    return script


def _structures() -> list[dict[str, Any]]:
    found = [AGENT, PARALLEL]
    if snippet.available()[0]:
        found.append(CODE)
    return found


def test_pipeline_shape_reads_agents_gates_forks_and_code():
    from refactory_dsh.nn import edits as nn_edits

    seed = Combo().state_dict()
    assert optim.pipeline_shape(seed) == {"nodes": 4, "agents": 1, "gate": False,
                                          "parallel": False, "code": False}
    assert optim.pipeline_shape(nn_edits.apply(AGENT, seed))["agents"] == 2
    forked = optim.pipeline_shape(nn_edits.apply(PARALLEL, seed))
    assert forked["parallel"] and forked["agents"] == 3
    coded = dict(seed, graph={**seed["graph"], "nodes": {
        **seed["graph"]["nodes"], "c": {"id": "c", "kind": "op", "call": snippet.RUN}}})
    assert optim.pipeline_shape(coded)["code"]
    gated = dict(seed, graph={**seed["graph"], "edges": [
        *seed["graph"]["edges"], {"source": "solve", "target": "prep", "condition": "retry"}]})
    assert optim.pipeline_shape(gated)["gate"]


def test_map_elites_separates_a_parallel_fork_from_the_same_agents_in_sequence():
    """Same agent count, same size band: the fork is the only shape difference, and it
    is a niche of its own."""
    from refactory_dsh.nn import edits as nn_edits
    from refactory_dsh.optim.policies import MapElitesSelector

    seed = Combo().state_dict()
    def agent(node_id: str) -> dict[str, Any]:
        return {"id": node_id, "kind": "agent", "writes": [f"{node_id}.txt"],
                "prompt_template": "Look at {{question}}"}
    sequence = seed
    for edit in ({"op": "insert_after", "target": "solve", "node": agent("one")},
                 {"op": "insert_after", "target": "one", "node": agent("two")},
                 {"op": "insert_after", "target": "two", "node": {
                     "id": "tail", "kind": "op", "call": "combo_ops:finish",
                     "produces": ["tail_done"]}}):
        sequence = nn_edits.apply(edit, sequence)
    forked = nn_edits.apply(PARALLEL, seed)
    a, b = optim.pipeline_shape(sequence), optim.pipeline_shape(forked)
    assert a["agents"] == b["agents"] == 3 and a["parallel"] != b["parallel"]
    strong, weak = _scored(sequence, 0.9), _scored(forked, 0.1)
    picker = MapElitesSelector(seed=0)
    differ = [i for i, (x, y) in enumerate(zip(picker.features(strong), picker.features(weak)))
              if x != y]
    assert differ == [3], "only the parallel axis may tell them apart"
    picker.add(strong)
    picker.add(weak)
    assert len(picker.cells) == 2 and weak in picker.cells.values()


def test_map_elites_keeps_each_structural_child_in_its_own_group_and_edits_it_again(tmp_path):
    structural = _structures()
    model = ComboModel(_structural_script(structural))
    loop = _run("default", tmp_path, model, parent_picker="map_elites", epochs=10,
                batch_size=len(structural), seed=3)
    assert loop.optimizer.proposal_failures == []
    picker = loop.optimizer.picker
    by_id = {c.id: c for c in loop.population}
    best = max((c for c in loop.population if c.evaluated), key=lambda c: c.score or 0.0)
    for edit in structural:
        node_id = edit["node"]["id"] if "node" in edit else edit["branches"][0]["id"]
        child = next(c for c in loop.population if c.own_edits() == [edit])
        assert child.evaluated, node_id
        assert (child.score or 0.0) < (best.score or 0.0), (
            f"{node_id}: the structural child is the leader, so this proves nothing")
        # It (or a better candidate of its shape) holds its own cell.
        cell = picker.features(child)
        holder = picker.cells.get(cell)
        assert holder is not None and optim.pipeline_shape(holder.state) == optim.pipeline_shape(child.state)
        assert sum(1 for key in picker.cells if key == cell) == 1
        # A later request described it (its new step and knob), and the edit it got
        # was evaluated as its child.
        asked = [r for r in model.requests if r.picked == child.id]
        assert asked, f"{node_id}: map_elites never picked the structural child again"
        assert all(r.has_node(node_id) and NEW_KNOB[node_id] in r.knobs for r in asked)
        grandchildren = [c for c in loop.population if c.parent == child.id and c.evaluated]
        assert grandchildren, f"{node_id}: its edit was never evaluated"
        assert any(e.get("name") == NEW_KNOB[node_id]
                   for g in grandchildren for e in g.own_edits())
    shapes = {tuple(sorted(optim.pipeline_shape(c.state).items())) for c in picker.cells.values()}
    assert len(shapes) == len(structural) + 1, "one group per shape, the seed's included"


# ── 4. the strategy decision reaches the model ─────────────────────────────


@pytest.mark.parametrize("name,expected", [
    ("staged_curriculum", "stage 1 of 4 (initial)"),
    ("adaptive_meta", None),
    ("tree_search", "action:"),
    ("research_then_optimize", "research:"),
    ("reflective", "branch:"),
    ("best_of_n", "independent variants of the seed"),
    ("autoscientists", "independent lines of attack"),
])
def test_the_strategy_decision_is_in_the_request_text(tmp_path, name, expected):
    model = ComboModel()
    loop = _run(name, tmp_path, model, val_gate=name in GATED_STRATEGIES, epochs=4)
    decided = [r for r in model.requests if r.decision]
    assert decided, f"{name}'s decision never reached a request"
    if expected is None:  # adaptive_meta decides "no meta" while it progresses: still data
        assert all("meta:" in r.decision for r in decided)
        return
    assert any(expected in r.decision for r in decided), [r.decision for r in decided]
    assert f"The search strategy (`{name}`)" in decided[0].decision


def test_the_default_loop_request_has_no_decision_section(tmp_path):
    model = ComboModel()
    _run("default", tmp_path, model, val_gate=True)
    assert model.requests and not any("## What this round is for" in r.text
                                      for r in model.requests)


def test_a_cli_run_of_the_default_loop_shows_no_decision_section(tmp_path):
    """A command-line run carries the resolved search contract in the same dict as a
    strategy's decision; it is not a decision, so the default request stays as it was."""
    model = ComboModel()
    _run("default", tmp_path, model, val_gate=True,
         resolved_profile={"contract": {"objective": "Maximize the score"}})
    assert model.requests and not any("## What this round is for" in r.text
                                      for r in model.requests)


def test_tree_search_debugs_a_failed_candidate(tmp_path):
    """The action follows the tree: a failed leaf is expanded next, as ``debug``."""
    broken = {"op": "insert_after", "target": "solve",
              "node": {"id": "broken", "kind": "op", "call": "combo_ops:explode"}}

    def script(model: ComboModel, request: Request) -> list[dict[str, Any]]:
        if request.index == 0:
            return [{"edit": broken, "rationale": "breaks"}, model.fix()]
        return fixer(model, request)

    model = ComboModel(script)
    loop = _run("tree_search", tmp_path, model, epochs=3, allowed_ops=())
    failed = [c for c in loop.population if c.evaluated and not c.exit_ok]
    assert failed, "the broken candidate did not fail, so this proves nothing"
    debug = [r for r in model.requests if "action: debug" in r.decision]
    assert debug and debug[0].has_node("broken"), [r.decision for r in model.requests]
    assert any(c.parent == failed[0].id for c in loop.population), "debug expanded nothing"


# ── a failed round names the parent the strategy or picker chose ───────────


def _failed_events(loop: Loop) -> list[dict[str, Any]]:
    lines = (loop.root / "progress.jsonl").read_text().splitlines()
    return [e for e in map(json.loads, lines) if e.get("step") == "proposal_failed"]


def test_a_failed_round_names_the_parent_the_strategy_chose(tmp_path):
    """tree_search names a failed leaf (not the leader) as the node to debug; when the
    model then fails, the proposal_failed event names that node."""
    broken = {"op": "insert_after", "target": "solve",
              "node": {"id": "broken", "kind": "op", "call": "combo_ops:explode"}}

    def script(model: ComboModel, request: Request) -> list[dict[str, Any]]:
        if request.index == 0:
            return [{"edit": broken, "rationale": "breaks"}, model.fix()]
        raise RuntimeError("model down")

    loop = _run("tree_search", tmp_path, ComboModel(script), epochs=3)
    failed = next(c for c in loop.population if c.evaluated and not c.exit_ok)
    leader = loop.select_parents(1)[0]
    assert failed.id != leader.id
    events = _failed_events(loop)
    assert events and events[0]["parent"] == failed.id, events


def test_a_failed_round_names_the_parent_the_picker_chose(tmp_path):
    def script(model: ComboModel, request: Request) -> list[dict[str, Any]]:
        if request.index < 3:
            return fixer(model, request)
        raise RuntimeError("model down")

    model = ComboModel(script)
    loop = _run("default", tmp_path, model, parent_picker="map_elites", epochs=5)
    events = _failed_events(loop)
    assert events, "the model never failed"
    failing = model.requests[3:]  # one event per failing request, in order
    assert len(events) == len(failing)
    for event, request in zip(events, failing):
        assert request.picked and event["parent"] == request.picked


class _NamesParentInPropose(Loop):
    """A strategy that names the parent only in ``strategy_decision``, inside
    ``propose`` -- after the loop chose its own -- and never touches the loop's."""

    name = "names_parent"

    def epoch_step(self, run=None):  # type: ignore[no-untyped-def]
        self.strategy_decision = {}
        return super().epoch_step(run)

    def propose(self, run, parents, n):  # type: ignore[no-untyped-def]
        others = [c for c in parents if c.id != self._editing_parent]
        if others and not self.strategy_decision:
            self.strategy_decision = {"parent": others[-1].id, "purpose": "not the leader"}
            self.named.append((others[-1].id, self._editing_parent))
        return super().propose(run, parents, n)


def test_a_failed_round_names_the_parent_in_the_strategy_decision(tmp_path):
    def script(model: ComboModel, request: Request) -> list[dict[str, Any]]:
        if request.index == 0:
            return fixer(model, request)
        raise RuntimeError("model down")

    with _combo_mode():
        loop = _NamesParentInPropose("fake/combo", _config(tmp_path, epochs=3),
                                     project=Path("."), model=ComboModel(script))
        loop.named = []
        loop.run()
    events = _failed_events(loop)
    failing = loop.named[1:]  # the first named round's request was answered
    assert events and len(events) == len(failing)
    for event, (named, own) in zip(events, failing):
        assert named != own and event["parent"] == named


# ── 5. the reflective edit-kind decision follows the evidence ──────────────


def _scored(state: dict[str, Any], score: float, parent: str | None = None,
            edits: list[dict[str, Any]] | None = None) -> Candidate:
    candidate = Candidate.from_state(state, parent=parent, edits=edits or [])
    candidate.evaluated, candidate.score = True, score
    return candidate


def _states() -> tuple[dict[str, Any], dict[str, Any]]:
    from refactory_dsh.nn import edits as nn_edits

    seed = Combo().state_dict()
    return seed, nn_edits.apply(AGENT, seed)


def _prompt(state: dict[str, Any], text: str) -> dict[str, Any]:
    from refactory_dsh.nn import edits as nn_edits

    return nn_edits.apply({"op": "set_knob", "name": "solve__system_prompt", "value": text}, state)


def test_edit_kind_is_pipeline_when_winners_and_losers_differ_in_shape():
    seed, grown = _states()
    population = [_scored(_prompt(grown, f"w{i}"), 0.9 - i * 0.01) for i in range(3)]
    population += [_scored(_prompt(seed, f"l{i}"), 0.2 - i * 0.01) for i in range(3)]
    decision = strategies.edit_kind_decision(population)
    assert decision["edit_kind"] == "pipeline"
    assert "1 agent step" in decision["why"] and "2 agent step" in decision["why"]
    # Advice, not a requirement: code steps are named beside the other pipeline moves.
    assert "add a code step that checks or extracts" in decision["purpose"]


def test_edit_kind_is_prompt_knob_when_one_shape_spans_the_range():
    seed, grown = _states()
    population = [_scored(_prompt(seed, "a"), 0.9), _scored(_prompt(grown, "b"), 0.5),
                  _scored(_prompt(seed, "c"), 0.1)]
    assert strategies.edit_kind_decision(population)["edit_kind"] == "prompt_knob"


def test_edit_kind_turns_to_structure_only_when_settings_stop_paying():
    seed, _ = _states()
    root = _scored(seed, 0.5)
    knob = [{"op": "set_knob", "name": "solve__system_prompt", "value": "x"}]
    improving = [root] + [_scored(_prompt(seed, f"p{i}"), 0.6 + i * 0.1, root.id, knob)
                          for i in range(3)]
    assert strategies.edit_kind_decision(improving)["edit_kind"] == "prompt_knob"
    stalled = [root] + [_scored(_prompt(seed, f"q{i}"), 0.4, root.id, knob) for i in range(3)]
    decision = strategies.edit_kind_decision(stalled)
    assert decision["edit_kind"] == "pipeline" and "did not beat their parents" in decision["why"]
    assert strategies.edit_kind_decision([root]) == {}, "one candidate is no evidence"


def test_reflective_tells_the_model_a_pipeline_change_when_the_evidence_says_so(tmp_path):
    """End to end: once a structural child wins, the next mutate request says
    'pipeline-level change' with the shapes as the reason."""
    def script(model: ComboModel, request: Request) -> list[dict[str, Any]]:
        if request.index == 0:
            # A structural child that also fixes three items, so its shape wins.
            return [{"edits": [AGENT, {"op": "set_knob", "name": "solve__system_prompt",
                                       "value": SEED_PROMPT + " fix:c1 fix:c2 fix:c3"}],
                     "rationale": "grow"}]
        return [{"edit": {"op": "set_knob", "name": "solve__system_prompt",
                          "value": f"{SEED_PROMPT} try {request.index}"}, "rationale": "x"}]

    model = ComboModel(script)
    _run("reflective", tmp_path, model, epochs=4, batch_size=1)
    kinds = [r.decision for r in model.requests if "edit_kind:" in r.decision]
    assert any("edit_kind: pipeline" in d and "2 agent step" in d for d in kinds), kinds


def test_reflective_merges_the_two_best_parents(tmp_path):
    from refactory_dsh.outerloop import Run

    with _combo_mode():
        loop = strategies.create("reflective", "fake/combo", _config(tmp_path),
                                 project=Path("."), model=ComboModel())
    seed = Combo().state_dict()
    low = _scored(seed, 0.1)
    a = _scored(_prompt(seed, "a"), 0.9)
    b = _scored(_prompt(seed, "b"), 0.8)
    loop.population = [low, a, b]
    loop.strategy_decision = {"branch": "merge"}
    proposals = loop.propose(Run(_loop=loop), [low, a, b], 1)
    assert proposals and proposals[0].parent == a.id
    assert proposals[0].edit == {"op": "set_knob", "name": "solve__system_prompt", "value": "b"}


# ── 6. the default path: the edit lands on the described candidate ─────────


def test_without_the_gate_a_model_edit_lands_on_the_candidate_it_was_written_for(tmp_path):
    """The request shows the leader; a proposal that names no parent went to the
    first evaluated candidate (the seed), so the lineage never got past one edit."""
    model = ComboModel()
    loop = _run("default", tmp_path, model, epochs=4, batch_size=1)
    by_id = {c.id: c for c in loop.population}
    children = _children(loop)
    assert any(c.parent != loop.population[0].id for c in children), (
        "every model edit was applied to the seed")
    for request, child in zip(model.requests, children):
        assert _described_prompt(request) == _fixed(by_id[child.parent])
    assert max(len(_fixed(c)) for c in children) >= 3


# ── 7. regression guard: the SkillOpt-matched default ──────────────────────

#: The leaderboard (id, score) of the SkillOpt-matched default on this toy for seed
#: 42, recorded on the base this change started from (origin/structural-child-knobs).
DEFAULT_LEADERBOARD: list[tuple[str, float]] = [
    ("fc7149763faa", 0.626667), ("292e9e280e7c", 0.46), ("5b64391fa673", 0.293333),
    ("91b1706604b4", 0.126667), ("b33d1d5ff726", -0.04)]


def test_the_skillopt_matched_default_leaderboard_is_unchanged(tmp_path):
    model = ComboModel()
    loop = _run("default", tmp_path, model, val_gate=True, batch_size=1, seed=42, epochs=4)
    board = [(c.id, round(c.score or 0.0, 6)) for c in loop.leaderboard()]
    print("LEADERBOARD", json.dumps(board))
    assert board == DEFAULT_LEADERBOARD
