"""The loop: eight steps, a bound, and a report that says what happened.

Phase 6 gate of ``docs/plans/pythonic-modes.md``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from refactory_dsh import nn, optim
from refactory_dsh.evaluators import Evaluator, RunResult
from refactory_dsh.loop import Loop, LoopConfig, LoopError
from refactory_dsh.optim import Candidate, Optimizer, Proposal

MODE = "circle_packing/adaptive"


def _config(root: Path, **overrides: object) -> LoopConfig:
    base = dict(epochs=2, batch_size=4, optimizer="random", seed=5, runs_root=root)
    base.update(overrides)
    return LoopConfig(**base)  # type: ignore[arg-type]


class ScriptedOptimizer(Optimizer):
    """Proposes one fixed edit per batch, so a test can assert what it did."""

    name = "scripted"

    def __init__(self, edit: dict, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._edit = edit

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        return [
            Proposal(edit=self._edit, parent=parents[0].id, rationale="scripted")
            for _ in range(max(1, n))
        ]


@pytest.fixture(scope="module")
def _default_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    """One run of the default configuration, for the tests that only read it.

    The run is deterministic (seed 5), and a dozen tests each started an identical
    one to read a different part of it, about 1.6 s apiece.
    """
    root = tmp_path_factory.mktemp("default-loop-run")
    report = Loop(MODE, _config(root), project=Path(".")).run()
    return root, report


@pytest.fixture
def default_run(_default_run: tuple[Path, dict]) -> tuple[Path, dict]:
    """The shared default run's directory, and a copy of its report to read."""
    root, report = _default_run
    return root, copy.deepcopy(report)


def _structural_edit(mode: nn.Mode) -> dict:
    block = {node.id: node for node in mode.insertable()}["restart_on_stall"]
    return {"op": "insert_after", "target": "accept", "node": block.to_state()}


def test_init_builds_a_starting_batch(tmp_path):
    loop = Loop(MODE, _config(tmp_path), project=Path("."))
    loop.init()
    assert len(loop.population) >= 2
    assert loop.population[0].parent is None
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "state.json").exists()


def _patch_mode(mode_class: type[nn.Mode]):
    """Point the registry at a local mode class for the duration of a test."""
    import contextlib

    import refactory_dsh.registry as registry

    class FakePackage:
        name = "fake"
        description = ""

        def mode(self, name: str) -> type[nn.Mode]:
            return mode_class

    @contextlib.contextmanager
    def patched():  # type: ignore[no-untyped-def]
        original = registry.resolve
        registry.resolve = lambda project, identifier: (FakePackage(), mode_class)  # type: ignore[assignment]
        try:
            yield
        finally:
            registry.resolve = original  # type: ignore[assignment]

    return patched()


class _BoundedInt(nn.Mode):
    """A mode whose only knob is a bounded integer, so the gradient can enumerate it."""

    name = "bounded"

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {"step": nn.Knob(2, "int", lo=1, hi=4, description="a bounded counter")}

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node("only", "op", call="loop_ops:write_number",
                          knobs=("step",), writes=("number.json",), produces=("value",)))
        return graph

    def score(self, run: nn.RunResult) -> float:
        return float(run.facts["value"])


class _OpenInt(nn.Mode):
    """A mode whose only knob is an integer with no declared bounds."""

    name = "open"

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {"step": nn.Knob(2, "int", description="an open counter")}

    def build(self) -> nn.Graph:
        return _BoundedInt().build()

    def score(self, run: nn.RunResult) -> float:
        return float(run.facts["value"])


class _Building(nn.Mode):
    """A constructible mode, so the loop can get as far as judging it."""

    name = "building"

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node("only", "op", call="loop_ops:write_best",
                          writes=("best.json",), produces=("best",)))
        return graph


class Unscored(_Building):
    pass  # keeps Mode.score, which means "unscored"


class Unrunnable(_Building):
    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node("start", "op", call="loop_ops:write_best",
                          writes=("best.json",), produces=("best",)))
        graph.add(nn.Node("orphan", "op", call="loop_ops:write_best",
                          writes=("other.json",), produces=("other",)))
        return graph

    def score(self, run: nn.RunResult) -> float:
        return float(run.facts.get("best", 0.0))


def test_init_refuses_a_mode_that_does_not_score(tmp_path):
    with _patch_mode(Unscored):
        with pytest.raises(LoopError, match="does not override score"):
            Loop(MODE, _config(tmp_path), project=Path(".")).init()


def test_init_refuses_a_mode_whose_seed_state_is_not_runnable(tmp_path):
    with _patch_mode(Unrunnable):
        with pytest.raises(LoopError, match="not runnable"):
            Loop(MODE, _config(tmp_path), project=Path(".")).init()


def test_two_epochs_complete_and_report_the_bound(default_run):
    root, report = default_run
    assert report["bound"] == "epochs"
    assert report["epochs_completed"] == 2
    assert report["evaluations"] >= 4
    assert (root / "report.json").exists()
    assert (root / "gradient.json").exists()
    markdown = (root / "report.md").read_text()
    assert markdown.startswith("# Outer loop"), "the report must be markdown, not JSON"
    assert "## Leaderboard" in markdown
    assert "## Refusals by rule" in markdown


def test_the_report_names_what_the_run_was(default_run):
    _root, report = default_run
    assert report["mode"] == MODE
    assert report["optimizer"] == "random"
    assert report["batch_size"] == 4
    assert report["seed_score"] is not None
    assert report["best_score"] >= report["seed_score"]


def test_the_leaderboard_is_ordered_and_complete(default_run):
    _root, report = default_run
    scores = [row["score"] for row in report["leaderboard"]]
    assert scores == sorted(scores, reverse=True)
    assert len(report["leaderboard"]) + len(report["failed_candidates"]) == report["evaluations"]
    assert all(row["exit_ok"] and row["score"] is not None for row in report["leaderboard"])
    assert all(not row["exit_ok"] and row["score"] is None for row in report["failed_candidates"])


def test_every_evaluated_candidate_validated_first(tmp_path):
    """An invalid candidate never reaches a run, so it never costs budget.

    The steps are driven by hand, so no epoch is opened and only the evaluation
    bound ends the loop: 12 evaluations already include proposals that validation
    must refuse (without it, two invalid candidates join the population).
    """
    loop = Loop(MODE, _config(tmp_path, max_evaluations=12), project=Path("."))
    loop.init()
    while True:
        evaluated = loop.forward()
        loop.loss_step()
        loop.backward()
        candidates, rejections = loop.optimizer_step()
        loop.step(candidates)
        if not loop.scheduler(evaluated=evaluated, proposed=len(candidates) + len(rejections),
                              accepted=len(candidates), refused=len(rejections)):
            break
    for candidate in loop.population:
        assert nn.validate_state(candidate.state) == [], candidate.id


def test_refusals_are_grouped_by_rule(default_run):
    _root, report = default_run
    codes = report["rejection_codes"]
    assert codes, "the random sampler should have been refused at least once"
    for code in codes:
        assert code.startswith(("V", "E-", "I:")), f"uncoded refusal: {code}"


def test_a_structural_candidate_is_proposed_and_evaluated(tmp_path):
    from refactory_dsh import registry

    _package, mode_class = registry.resolve(Path("."), MODE)
    edit = _structural_edit(mode_class())
    loop = Loop(
        MODE,
        _config(tmp_path),
        project=Path("."),
        optimizer=ScriptedOptimizer(edit, k=1),
    )
    report = loop.run()
    assert report["structural"]["structural_candidates"] >= 1
    structural = [row for row in report["leaderboard"] if row["structural"] and row["exit_ok"]]
    assert structural, "no structural candidate ran"


def test_a_structural_edit_beats_its_parent(tmp_path):
    """The seed pipeline wastes its budget; the fix is a node, and it shows."""
    from refactory_dsh import registry

    _package, mode_class = registry.resolve(Path("."), MODE)
    loop = Loop(
        MODE,
        _config(tmp_path),
        project=Path("."),
        optimizer=ScriptedOptimizer(_structural_edit(mode_class()), k=1),
    )
    report = loop.run()
    assert report["best_score"] > report["seed_score"]
    winners = [c for c in report["structural_wins"] if c["score"] == report["best_score"]]
    assert winners, "the structural candidate did not reach the top of the leaderboard"
    assert all(c["exit_ok"] for c in report["structural_wins"]), (
        "a structural candidate that never ran is not a win"
    )


def test_proposal_outcomes_pair_a_proposal_with_what_happened(default_run):
    _root, report = default_run
    outcomes = report["proposal_outcomes"]
    assert outcomes
    for row in outcomes:
        assert "delta" in row
        if row["parent_score"] is not None and row["score"] is not None:
            assert row["delta"] == pytest.approx(row["score"] - row["parent_score"])


def test_the_report_counts_model_calls_made_inside_the_graphs(default_run):
    """An inserted agent node costs model calls, and the report says how many."""
    _root, report = default_run
    assert report["cost"]["inner_model_calls"] == 0, (
        "the deterministic seed pipeline makes no model calls"
    )


def test_the_gradient_carries_knob_attribution_and_reasons(tmp_path):
    loop = Loop(MODE, _config(tmp_path), project=Path("."))
    loop.init()
    loop.forward()
    loop.loss_step()
    gradient = loop.backward()
    attribution = gradient["knob_attribution"]
    assert "order" in attribution
    assert sum(v["n"] for v in attribution["order"]["tried"].values()) >= 1
    assert "asc" in attribution["order"]["untried"], (
        "the gradient must say which values nobody has tried"
    )
    assert gradient["cost"]["n"] >= 1
    assert "rejection_codes" in gradient
    assert gradient["structural"]["accepted_ops"]


def test_the_report_names_candidates_that_were_never_evaluated(default_run):
    """A batch proposed for an epoch that did not run must be named, not implied.

    The pipeline proposes the next epoch's batch at the end of an epoch, so a run
    stopped by the epoch bound holds candidates that were installed and never run.
    The run reported `evaluations: 3` for a 2-epoch configuration without saying
    which candidates that left out, and a model-driven round whose output was
    dropped looked like a round that happened.
    """
    root, report = default_run  # the default configuration runs two epochs
    unevaluated = report["unevaluated"]
    assert unevaluated, "the final epoch's batch was dropped silently"
    assert len(unevaluated) == report["population"] - report["evaluations"]
    assert all(candidate["id"] not in unevaluated for candidate in report["leaderboard"])
    markdown = (root / "report.md").read_text()
    assert "never evaluated" in markdown, "the report.md does not say so"
    assert all(f"`{candidate}`" in markdown for candidate in unevaluated)


def test_the_scheduler_reports_the_evaluation_bound(tmp_path):
    loop = Loop(MODE, _config(tmp_path, max_evaluations=2), project=Path("."))
    report = loop.run()
    assert report["bound"] == "evaluations"
    assert report["evaluations"] == 2


def test_the_epoch_summary_counts_what_the_epoch_did(default_run):
    _root, report = default_run
    summary = report["epochs_summary"]
    assert len(summary) == 2
    assert summary[0]["evaluated"] >= 1
    assert summary[0]["proposed"] >= 0


def test_a_run_leaves_the_analysis_of_a_model_optimizer(tmp_path):
    """A reasoning optimizer's diagnosis is written where a person can read it."""

    class Diagnosis(Optimizer):
        name = "diagnosis"

        def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
            self.last_diagnosis = "the generator repeats itself"
            return [
                Proposal(
                    edit={"op": "set_knob", "name": "order", "value": "asc"},
                    parent=parents[0].id,
                )
            ]

    loop = Loop(
        MODE, _config(tmp_path), project=Path("."), optimizer=Diagnosis(k=1)
    )
    loop.run()
    written = sorted((tmp_path / "analysis").glob("*.md"))
    assert written
    assert "the generator repeats itself" in written[0].read_text()


def test_the_starting_population_sweeps_the_parameters(tmp_path):
    """The first epoch must produce a gradient with contrast, not just candidates."""
    loop = Loop(MODE, _config(tmp_path), project=Path("."))
    loop.init()
    swept = [
        candidate
        for candidate in loop.population
        if not candidate.is_structural() and candidate.edits
    ]
    assert swept, "the starting population only moved structure"
    moved = {edit["name"] for candidate in swept for edit in candidate.edits}
    assert {"order", "grid", "selector"} & moved


def test_a_bounded_numeric_knob_reports_the_values_nobody_ran(tmp_path):
    """`step` declares 1..4; the gradient must say which of those never ran.

    The enum case is covered above. A numeric knob with both bounds declared has
    just as finite a set, and reporting `untried: []` there tells an optimizer
    that a value it never tried has already been ruled out.
    """
    with _patch_mode(_BoundedInt):
        # batch_size=2 so the sweep cannot cover all 4 values in one batch,
        # even when set_knob is picked every time (F14 removed set_node_field
        # and replace_node, making set_knob more likely).
        loop = Loop(MODE, _config(tmp_path, batch_size=2), project=Path("."))
        loop.init()
        loop.forward()
        loop.loss_step()
        attribution = loop.backward()["knob_attribution"]
    tried = attribution["step"]["tried"]
    untried = attribution["step"]["untried"]
    assert untried, "the sweep cannot have run all four values in one batch"
    assert set(untried) <= {1, 2, 3, 4}
    assert all(str(value) not in tried for value in untried)
    assert {int(value) for value in tried} | set(untried) == {1, 2, 3, 4}


def test_an_unbounded_numeric_knob_reports_no_untried_values(tmp_path):
    """There is no enumerable set to read, so the gradient must not invent one."""
    with _patch_mode(_OpenInt):
        loop = Loop(MODE, _config(tmp_path), project=Path("."))
        loop.init()
        loop.forward()
        loop.loss_step()
        attribution = loop.backward()["knob_attribution"]
    assert attribution["step"]["untried"] == []


def test_the_gradient_reports_which_values_are_untried(tmp_path):
    loop = Loop(MODE, _config(tmp_path), project=Path("."))
    loop.init()
    loop.forward()
    loop.loss_step()
    attribution = loop.backward()["knob_attribution"]
    untried = attribution["grid"]["untried"]
    assert untried, "every grid value was swept, so the assertion is not exercising this"
    assert all(str(value) not in attribution["grid"]["tried"] for value in untried)


def test_a_knob_move_selected_by_the_optimizer_improves_the_run(tmp_path):
    loop = Loop(
        MODE,
        _config(tmp_path),
        project=Path("."),
        optimizer=ScriptedOptimizer({"op": "set_knob", "name": "order", "value": "asc"}, k=1),
    )
    report = loop.run()
    assert report["best_score"] > report["seed_score"]


def test_rejections_are_persisted_as_a_stream(tmp_path):
    loop = Loop(MODE, _config(tmp_path, batch_size=6), project=Path("."))
    loop.run()
    path = tmp_path / "rejections.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            assert row["reasons"]
            assert row["codes"]


def test_a_proposal_record_is_never_overwritten_by_a_later_candidate(default_run):
    """Every recorded proposal names a parent that is not itself."""
    _root, report = default_run
    for row in report["proposal_outcomes"]:
        assert row["parent"] != row["id"], (
            f"{row['id']} was recorded as its own parent, so a later candidate "
            "overwrote an earlier record"
        )


def test_the_best_candidates_edits_match_its_lineage(default_run):
    _root, report = default_run
    lineage = report["lineage"]
    if lineage:
        best = lineage[-1]
        recorded = next(
            (r for r in report["proposal_outcomes"] if r["id"] == best["id"]), None
        )
        if recorded is not None:
            assert recorded["edits"] == best["edits"][-len(recorded["edits"]):] or (
                recorded["edits"] == best["edits"]
            ), "the proposal record disagrees with the lineage it belongs to"


class _Agentic(nn.Mode):
    """A scored mode whose graph calls a model, so a ledger has somewhere to land."""

    name = "agentic"

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {"step": nn.Knob(2, "int", lo=1, hi=4, description="a bounded counter")}

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node("draft", "agent", prompt_template="draft a line",
                          writes=("draft.txt",), produces=("draft",)))
        graph.add(nn.Node("only", "op", call="loop_ops:write_number",
                          knobs=("step",), reads=("draft.txt",),
                          writes=("number.json",), produces=("value",)))
        graph.connect("draft", "only")
        return graph

    def score(self, run: nn.RunResult) -> float:
        return float(run.facts["value"])


def test_a_search_carries_its_ledger_to_the_mode_it_is_searching(tmp_path):
    """`optimize --intent …` and `steer --note …` reach the mode's own agent nodes.

    The loop seeds the run's ledger at the run root and evaluates each candidate in
    `runs/<id>/`, and the steering transform folds in the ledger of the directory it
    is handed. Nothing read across that gap, so the intent a person gave, and every
    note they added while the search was going, was written, reported by `steer`, and
    seen by no node. Found by running it: the prompts a scripted model received
    contained neither the ledger nor the intent.
    """
    from tests_support import FakeModel

    model = FakeModel(["a line"] * 50)
    with _patch_mode(_Agentic):
        loop = Loop(
            "agentic/only",
            _config(tmp_path, epochs=1, batch_size=1, intent="the number must be four"),
            project=Path("."),
            model=model,
        )
        loop.init()
        loop.run()

    prompts = "\n".join(model.prompts)
    assert "What the user has said" in prompts, prompts[:400]
    assert "the number must be four" in prompts
    # The candidate keeps a copy, so the directory says what was true when it ran.
    assert list((tmp_path / "runs").glob("*/intent-ledger.md")), sorted(
        str(path) for path in (tmp_path / "runs").iterdir()
    )


def test_a_note_steers_the_candidates_that_run_after_it(tmp_path):
    """Steering a search is the same act as steering a run: append, and the next
    candidate sees it. The note lands in the ledger between two evaluations."""
    from refactory_dsh.steering import Ledger
    from tests_support import FakeModel

    seen: list[str] = []
    model = FakeModel([""] * 50)

    class Watching(FakeModel):  # type: ignore[misc]
        def complete(self, prompt: str, *, system: str = "") -> str:
            seen.append(prompt)
            return super().complete(prompt, system=system)

    with _patch_mode(_Agentic):
        loop = Loop(
            "agentic/only",
            _config(tmp_path, epochs=1, batch_size=2, intent="the number must be two"),
            project=Path("."),
            model=Watching([""] * 50),
        )
        loop.init()
        first = loop.population[:1]
        loop._evaluate(first[0])
        Ledger(tmp_path).note("constraint", "never print the number")
        loop._evaluate(loop.population[1])

    assert "never print the number" not in seen[0], "the note reached a candidate that had already run"
    assert "never print the number" in seen[1], seen[1][:400]


def test_the_search_can_be_seeded_with_the_knobs_a_mode_needs(tmp_path):
    """A mode whose inputs are frozen knobs cannot be searched without them (#see 69).

    Frozen means the *search* may not move a knob. It does not mean a caller may not
    say what the run operates on — and without that, every candidate of an input-taking
    mode runs with the declared defaults, scores the same thing, and the run reports a
    search that found nothing.
    """
    class NeedsInput(nn.Mode):
        name = "needs_input"

        @classmethod
        def declare(cls):
            return {
                "source": nn.Knob("", "str", frozen=True, description="what to read"),
                "step": nn.Knob(2, "int", lo=1, hi=4, description="a counter"),
            }

        def build(self):
            graph = nn.Graph()
            graph.add(nn.Node("only", "op", call="loop_ops:write_number",
                              knobs=("step", "source"), writes=("number.json",),
                              produces=("value",)))
            return graph

        def score(self, run):
            return float(run.facts["value"])

    with _patch_mode(NeedsInput):
        loop = Loop(
            "needs_input/only",
            _config(tmp_path, epochs=1, batch_size=1, knobs={"step": "3"}),
            project=Path("."),
        )
        loop.init()
    assert loop.mode.values()["step"] == 3, (
        "the seeded knob value did not reach the loop's own mode"
    )
    assert loop.population[0].state["knobs"]["step"]["value"] == 3, (
        "the seeded knob value did not reach the seed candidate"
    )


def test_a_knob_a_person_sets_from_the_command_line_is_parsed_by_kind():
    """`--knob target=7` is text, and an int knob has to receive the integer 7."""
    from refactory_dsh import nn as nn_module

    state = nn_module.Mode().state_dict() if False else {
        "version": 1,
        "graph": {"start": "", "nodes": {}, "edges": []},
        "knobs": {
            "count": {"value": 0, "spec": {"kind": "int", "default": 0, "frozen": False,
                                           "lo": 0, "hi": 10, "enum": [], "description": ""}},
            "flag": {"value": False, "spec": {"kind": "bool", "default": False,
                                              "frozen": True, "enum": [], "description": ""}},
            "label": {"value": "", "spec": {"kind": "str", "default": "", "frozen": False,
                                            "enum": [], "description": ""}},
        },
        "ports": [],
        "memory": [],
    }
    from refactory_dsh.nn import edits as edits_module

    out = edits_module.configure(state, "count", "7")
    assert out["knobs"]["count"]["value"] == 7
    out = edits_module.configure(out, "flag", "1")
    assert out["knobs"]["flag"]["value"] is True, "a frozen knob is settable by the caller"
    out = edits_module.configure(out, "label", "hello")
    assert out["knobs"]["label"]["value"] == "hello"
    with pytest.raises(edits_module.EditError, match="expected a number"):
        edits_module.configure(state, "count", "seven")


def test_a_search_builds_a_model_client_for_the_modes_agent_nodes(tmp_path, monkeypatch):
    """Every optimizer needs one, because the mode's own agent nodes do.

    The client was built only for `autoresearch`, so under `random` every candidate of
    an agent mode failed with `agent node 'propose' needs a model client` and the run
    reported a search that found nothing.
    """
    from refactory_dsh import model as model_module
    from refactory_dsh.loop import main as loop_main

    built: list[tuple[str, str]] = []

    def fake_client(*, model: str = "", provider: str = "", **kwargs: object) -> object:
        built.append((model, provider))
        return object()  # the evaluator only checks that a client exists

    monkeypatch.setattr(model_module, "DshModelClient", fake_client)
    loop_main([
        "circle_packing/adaptive", "--epochs", "1", "--batch-size", "1",
        "--optimizer", "random", "--seed", "1",
        "--runs-root", str(tmp_path / "run"), "--project", ".",
    ])
    assert built, "the search built no model client, so an agent mode could not run"


# ── the size cost the search compares on ───────────────────────────────────


class _Fixed(Evaluator):
    """Scores every candidate the same, so only its size can decide a ranking."""

    name = "fixed_for_parsimony"

    @classmethod
    def from_settings(cls, *, config, model=None, project=None):  # type: ignore[no-untyped-def]
        return cls()

    def evaluate(self, mode, state, workdir):  # type: ignore[no-untyped-def]
        return RunResult(score=0.5, facts={}, exit_ok=True)


def _fixed_search(tmp_path, **overrides):
    """A real mode, a size-adding scripted edit, and an evaluator with one score.

    Two epochs, because the run proposes the next batch at the end of an epoch: the
    added node is installed then and evaluated in the second one.
    """
    from refactory_dsh import evaluators, registry

    _package, mode_class = registry.resolve(Path("."), MODE)
    evaluators.register(_Fixed.name, _Fixed)
    try:
        loop = Loop(
            MODE,
            _config(tmp_path, epochs=2, batch_size=1, evaluator=_Fixed.name, **overrides),
            project=Path("."),
            optimizer=ScriptedOptimizer(_structural_edit(mode_class()), k=1),
        )
        return loop.run()
    finally:
        evaluators.REGISTRY.pop(_Fixed.name, None)


def test_the_search_compares_the_raw_score_with_the_size_cost_taken_off(tmp_path):
    """A candidate that costs more nodes must beat a smaller one on the raw score.

    Every candidate here scores 0.5 from the evaluator, so the only difference
    between the seed and the candidate that adds a node is that node. Ranking on the
    raw score would keep the larger graph; ranking on the costed score keeps the
    smaller one, and the report carries both numbers so a reader can see which.
    """
    report = _fixed_search(tmp_path)
    seed = next(row for row in report["leaderboard"] if row["parent"] is None)
    child = next(row for row in report["leaderboard"] if row["structural"])

    assert seed["raw_score"] == pytest.approx(0.5)
    assert child["raw_score"] == pytest.approx(0.5), (
        "the structural candidate must score the same as the seed, or this is not "
        "testing the size cost"
    )
    assert child["nodes"] == seed["nodes"] + 1, "the edit did not add a node"
    assert child["score"] == pytest.approx(0.5 - 0.01 * child["nodes"])
    assert seed["score"] == pytest.approx(0.5 - 0.01 * seed["nodes"])
    assert report["best"]["id"] != child["id"], (
        "the larger graph won on a score the size cost should have taken off"
    )
    assert report["best_score"] == pytest.approx(seed["score"])
    assert report["raw_best_score"] == pytest.approx(0.5)
    assert report["scoring"]["compared"] == "penalized"
    assert report["scoring"]["parsimony"] == 0.01


def test_a_zero_size_cost_leaves_the_raw_score_alone(tmp_path):
    """The coefficient is a person's setting, and zero switches the cost off."""
    report = _fixed_search(tmp_path, parsimony=0.0)
    assert report["best_score"] == pytest.approx(0.5)
    assert report["raw_best_score"] == pytest.approx(0.5)
    assert report["scoring"]["parsimony"] == 0.0


def test_every_recorded_score_carries_the_raw_one_beside_it(default_run):
    """The journal is what a resumed run and an adopter read, so both live in it."""
    root, _report = default_run
    events = [
        json.loads(line)
        for line in (root / "journal.jsonl").read_text().splitlines()
        if line.strip()
    ]
    scored = [event for event in events if event.get("event") == "score"]
    assert scored, "the run recorded no score"
    for event in scored:
        if not event["exit_ok"]:
            assert event["raw_score"] is None and event["score"] is None, event
            assert event["error"], event
            continue
        assert event["raw_score"] is not None, event
        assert event["score"] == pytest.approx(
            event["raw_score"] - event["parsimony"] * event["nodes"]
        )


def test_the_run_prints_meaningful_events_while_it_goes(tmp_path, capsys):
    """A watched search says where it is; the report is not the first thing seen."""
    report = Loop(MODE, _config(tmp_path, epochs=1, batch_size=2, print_events=True),
                  project=Path(".")).run()
    err = capsys.readouterr().err
    assert "run · optimizer random" in err
    assert "epoch 1 · candidate" in err
    assert "done · best" in err
    assert "loss" not in err, "the firehose is only for --verbose"
    assert report["best_score"] is not None


def test_a_verbose_run_prints_more_than_the_default(tmp_path, capsys):
    """Verbose is a superset: the events the meaningful level summarizes, shown."""
    Loop(
        MODE,
        _config(tmp_path, epochs=1, batch_size=2, print_events=True, verbose=True),
        project=Path("."),
    ).run()
    err = capsys.readouterr().err
    assert "loss" in err and "backward" in err


def test_a_library_run_prints_nothing(tmp_path, capsys):
    """No caller asked for events, so none are printed; the file is still written."""
    Loop(MODE, _config(tmp_path, epochs=1, batch_size=2), project=Path(".")).run()
    assert capsys.readouterr().err == ""
    assert (tmp_path / "progress.jsonl").exists()


def test_the_proposer_reads_the_run_ledger_live(tmp_path):
    """A note added while the search runs reaches the next proposal.

    Steering a search is not only watching it: the thing that proposes edits has to
    see what the person said since the run started.
    """
    from refactory_dsh.steering import Ledger

    loop = Loop(MODE, _config(tmp_path, intent="keep the step small"), project=Path("."))
    loop.init()
    assert "keep the step small" in loop._context().render()
    Ledger(tmp_path).note("constraint", "never add a node")
    live = loop._context().render()
    assert "never add a node" in live
    assert "must be respected" in live


def test_no_ledger_means_no_steering_section(tmp_path):
    """A run nobody steered reads exactly as it did before the ledger existed."""
    loop = Loop(MODE, _config(tmp_path), project=Path("."))
    loop.init()
    assert "What the user has said" not in loop._context().render()


@pytest.mark.parametrize("owner", [True, False])
def test_user_pause_stops_only_the_owned_resource_clock(tmp_path, monkeypatch, owner):
    from refactory_dsh.budgets import BudgetExhausted, ResourceLimits, RunBudget
    from refactory_dsh.control import Control

    now = [100.0]
    monkeypatch.setattr("refactory_dsh.budgets.time.time", lambda: now[0])
    loop = Loop(MODE, _config(tmp_path), project=Path("."))
    loop.budget = RunBudget.create(tmp_path, ResourceLimits(max_elapsed_s=5, max_tokens=10))
    loop._owns_resource_budget = owner
    loop.budget.record_usage("before-pause", 2)
    now[0] += 1
    requests = iter([{"action": "pause"}, {"action": "resume"}])
    monkeypatch.setattr(Control, "read_request", lambda _control: next(requests))

    def wait(_seconds):
        assert (loop.budget.snapshot()["started_at"] is None) is owner
        now[0] += 10

    monkeypatch.setattr("refactory_dsh.loop.time.sleep", wait)
    loop._apply_control()
    usage = loop.budget.snapshot()
    assert usage["tokens"] == 2
    assert usage["elapsed_s"] == (1 if owner else 11)
    assert usage["started_at"] is not None
    if owner:
        loop.budget.checkpoint()
    else:
        with pytest.raises(BudgetExhausted):
            loop.budget.checkpoint()
