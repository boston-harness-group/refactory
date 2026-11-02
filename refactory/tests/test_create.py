"""create-mode: a brief in, a validated and registered package out.

Phase 7 gate of ``docs/plans/pythonic-modes.md``.
"""

from __future__ import annotations

import json
import tempfile
import shutil
from pathlib import Path

import pytest

from refactory_dsh import create as create_module
from refactory_dsh import registry
from refactory_dsh.create import CreateConfig, CreateError, CreateMode
from refactory_dsh import create_ops, sandbox
from refactory_dsh.create_ops import parse_files
from refactory_dsh.paths import REPO_ROOT
from test_fast_modes import PLAN, QA_PASS
from tests_support import (
    FakeModel,
    RESEARCH_ANSWER,
    REVIEW_ANSWERS,
    baked_data_files,
    create_answers,
    create_script,
    good_data_files,
    good_files,
    render,
    strategy_answer,
    unfed_data_files,
)

NAME = "created_pkg"

#: The Task the per-item fixture binds to: the one `dataset_probe` ships, so the
#: binding is to a real class of work rather than to a string in a prompt.
TASK = "refactory_packages.dataset_probe.tasks:CountingTask"


def _config(**overrides) -> CreateConfig:
    """A create config standing in for the person's answers.

    `gate_strategy` is a `user` gate, carried over from create-v2: the strategy is
    a decision the person owns. A walker run stops at it; a test that wants a
    package passes the approval it is standing in for. The default is `single_shot`
    because the package these fixtures emit has no Task binding; a test that names a
    Task overrides it, and naming a Task wins.
    """
    return CreateConfig(
        verdicts={"gate_strategy": "proceed"}, single_shot=True, **overrides
    )



@pytest.fixture(autouse=True)
def _clean_package():
    """A created package is real: leave the workspace as it was found.

    Uses a file lock so xdist workers do not step on each other's packages.
    """
    import fcntl
    root = registry.workspace_root(Path(".")) / NAME
    lock_path = root.parent / f".{NAME}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = open(lock_path, "w")
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    shutil.rmtree(root, ignore_errors=True)
    try:
        yield
    finally:
        shutil.rmtree(root, ignore_errors=True)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


# ── T2: @-path accepted by parse_files ─────────────────────────────────────
def test_parse_files_keeps_at_path():
    files = parse_files("===FILE: packages/@types/index.d.ts===\ncontent\n")
    assert "packages/@types/index.d.ts" in files
    assert files["packages/@types/index.d.ts"] == "content\n"


# ── T6: malformed header refused by parse_files ───────────────────────────
@pytest.mark.parametrize("answer, fragment", [
    ("===FILE: has space.txt===\ncontent\n", "has space.txt"),
    ("===FILE:===\ncontent\n", "===FILE:==="),
])
def test_parse_files_refuses_malformed_header(answer, fragment):
    import re as _re
    with pytest.raises(ValueError, match=_re.escape(fragment)):
        parse_files(answer)


# ── T7: create-mode loop-back on malformed header (full graph) ────────────
def test_create_mode_loops_back_on_malformed_header(tmp_path):
    """A malformed header loops back to emit through the full create graph.

    First emission has a space in the path (malformed); second is valid.
    The run succeeds, emit runs twice, and the retry prompt quotes the bad line.
    """
    bad_emission = "===FILE: has space.txt===\ncontent\n"
    model = FakeModel(create_script(NAME, [bad_emission, render(good_files(NAME))]))
    result = create_module.create(
        _config(name=NAME, brief="count to a target", workdir=tmp_path / "run"),
        client=model,
    )
    assert result.package == NAME
    emit_prompts = [p for p in model.prompts if "Write the package" in p]
    assert len(emit_prompts) == 2, "emit must run twice"
    assert "has space.txt" in emit_prompts[-1], "retry prompt must quote the bad line"


def test_create_fast_mode_loops_back_on_malformed_header(tmp_path, monkeypatch):
    """A malformed header loops back to emit through the full create_fast graph.

    First emission has a space in the path (malformed); second is valid.
    The run succeeds, emit runs twice, and the retry prompt quotes the bad line.
    """
    from refactory_dsh.fast import CreateFastMode

    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    name = "fast_counter"
    bad_emission = "===FILE: has space.txt===\ncontent\n"
    good_emission = render(good_files(name))
    mode = CreateFastMode()
    state = mode.state_dict()
    state["knobs"]["single_shot"]["value"] = True
    mode.load_state_dict(state)
    client = FakeModel([PLAN, bad_emission, good_emission, QA_PASS])
    run = mode.run(tmp_path / "run", model=client,
                   task={"name": name, "brief": "a reusable counter"})
    assert run.exit_ok, run.error
    emit_prompts = [p for p in client.prompts if "emit the package" in p]
    assert len(emit_prompts) == 2, "emit must run twice"
    assert "has space.txt" in emit_prompts[-1], "retry prompt must quote the bad line"


# ── T8: EDIT and DELETE headers refused by parse_files (whole files only) ──
@pytest.mark.parametrize("header", [
    "===EDIT: src/main.py===",
    "===DELETE: src/main.py===",
])
def test_parse_files_refuses_edit_header_with_whole_files_message(header):
    """EDIT and DELETE headers are refused by parse_files, quoting the line
    and saying this mode takes whole files only."""
    import re as _re
    answer = f"{header}\nsome body\n"
    with pytest.raises(ValueError, match=_re.escape(header)) as exc_info:
        parse_files(answer)
    assert "whole files only" in str(exc_info.value)
    assert "===FILE: <path>===" in str(exc_info.value)


# ── T9b: backtick- and quote-wrapped paths refused by parse_files ──────────
@pytest.mark.parametrize("answer, fragment", [
    ("===FILE: `src/a.py`===\ncontent\n", "`src/a.py`"),
    ("===FILE: 'src/a.py'===\ncontent\n", "'src/a.py'"),
    ('===FILE: "src/a.py"===\ncontent\n', '"src/a.py"'),
])
def test_parse_files_refuses_backtick_and_quote_wrapped_paths(answer, fragment):
    import re as _re
    with pytest.raises(ValueError, match=_re.escape(fragment)):
        parse_files(answer)


def test_the_command_line_carries_the_persons_approval():
    """`--verdict gate=value` is how a headless create is told the person's answer.

    Without it the run stops at `gate_strategy`; the flag is the same spelling
    `console steer --verdict` uses, so one answer can be stated up front or recorded
    on a paused run.
    """
    from refactory_dsh.create import parse_verdicts

    assert parse_verdicts(["gate_strategy=proceed"]) == {"gate_strategy": "proceed"}
    assert parse_verdicts(["a=1", "b=reloop"]) == {"a": "1", "b": "reloop"}
    assert parse_verdicts([]) == {}
    for bad in ("proceed", "gate_strategy=", "=proceed", ""):
        with pytest.raises(ValueError, match="--verdict takes"):
            parse_verdicts([bad])


def test_parse_files_splits_marked_blocks():
    answer = (
        "===FILE: __init__.py===\nvalue = 1\n"
        "===FILE: ops.py===\nx = 2\n"
    )
    files = parse_files(answer)
    assert sorted(files) == ["__init__.py", "ops.py"]
    assert files["__init__.py"] == "value = 1\n"
    assert files["ops.py"] == "x = 2\n"


def test_parse_files_returns_nothing_for_a_bare_answer():
    assert parse_files("here is some prose") == {}


def test_create_mode_is_itself_a_mode():
    mode = CreateMode()
    graph = mode.graph()
    assert sorted(graph.nodes) == [
        "code_reviewer", "design", "emit", "fork_qa", "gate_overwatch", "gate_qa",
        "gate_research", "gate_strategy", "health_checker", "join_qa", "materialise",
        "overwatch", "prepare", "qa_director", "record_findings", "register", "report",
        "research_director", "review", "smoke", "strategy_director", "survey",
        "synthesize_qa", "synthesize_strategy", "validate",
    ], "the graph is create-v2's phases, not v1's"
    assert {node.kind for node in graph.nodes.values()} == {
        "op", "agent", "gate", "fork", "join",
    }
    assert mode.is_scored() is False, "creating is judged by the package, not by a number"


# prepare() checks the model credential, here a placeholder in the test's own home (dm-257).
@pytest.mark.usefixtures("configured_model_home")
def test_a_brief_alone_names_the_package(tmp_path):
    """"Make me a thing that counts teapots" is a complete request.

    The name is how the result is found later, so it has to exist -- but refusing to
    start over a missing name, when the brief already says what is wanted, is the
    wrong thing to be strict about. The brief's own words become the name.
    """
    from refactory_dsh import create_ops
    from refactory_dsh.nn import RunContext

    ctx = RunContext(
        tmp_path,
        knobs={"brief": "count the teapots in a tray", "single_shot": 1},
        facts={},
        task={},
    )
    assert create_ops.prepare(ctx)["name"] == "count_the_teapots"


# prepare() checks the model credential, here a placeholder in the test's own home (dm-257).
@pytest.mark.usefixtures("configured_model_home")
def test_prepare_reads_the_knobs_a_run_on_dsh_carries(tmp_path):
    """The walker passes name/brief in the task; DSH has no task, so they are knobs."""
    from refactory_dsh import create_ops
    from refactory_dsh.nn import RunContext

    ctx = RunContext(
        tmp_path,
        knobs={"name": "knob_pack", "brief": "from the knobs", "single_shot": 1},
        facts={},
        task={},
    )
    published = create_ops.prepare(ctx)
    assert published["name"] == "knob_pack"
    assert published["brief"] == "from the knobs"


# prepare() checks the model credential, here a placeholder in the test's own home (dm-257).
@pytest.mark.usefixtures("configured_model_home")
def test_prepare_reads_the_mode_to_revise_from_the_knobs(tmp_path):
    """`--mode edit`'s subject travels as a knob, because a DSH run has no task."""
    from refactory_dsh import create_ops
    from refactory_dsh.nn import RunContext

    ctx = RunContext(
        tmp_path,
        knobs={"name": "revised", "brief": "fewer steps", "revision_of": "counting_pack/fixed_step"},
        facts={},
        task={},
    )
    assert create_ops.prepare(ctx)["revision_of"] == "counting_pack/fixed_step"


def test_prepare_refuses_a_run_with_nothing_to_build(tmp_path):
    from refactory_dsh import create_ops
    from refactory_dsh.nn import RunContext

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        create_ops.prepare(RunContext(tmp_path, knobs={}, facts={}, task={}))


def test_a_good_answer_creates_registers_and_smoke_scores(tmp_path):
    model = FakeModel(create_answers(NAME, good_files(NAME)))
    result = create_module.create(
        _config(name=NAME, brief="count to a target", workdir=tmp_path / "run"),
        client=model,
    )
    assert result.package == NAME
    assert result.mode == f"{NAME}/counter"
    assert result.smoke_score and result.smoke_score > 0
    assert f"{NAME}/counter" in registry.render_catalogue(Path("."))
    assert (result.path / "mode.py").exists()
    assert result.path.resolve().is_relative_to(
        registry.workspace_root(Path(".")).resolve()
    ), "a generated package must not be written into the source tree"


def test_the_created_mode_runs_and_scores(tmp_path):
    model = FakeModel(create_answers(NAME, good_files(NAME)))
    create_module.create(
        _config(name=NAME, brief="count", workdir=tmp_path / "run"), client=model
    )
    _package, mode_class = registry.resolve(Path("."), f"{NAME}/counter")
    workdir = tmp_path / "smoke"
    result = mode_class().evaluate(workdir)
    assert result.exit_ok, result.error
    assert result.score == 3.0
    assert result.facts["value"] == 3


def test_a_broken_package_is_refused_and_then_regenerated(tmp_path):
    broken = {
        "__init__.py": "PACKAGE = object()\n",
        "ops.py": "def nothing():\n    return None\n",
        "mode.py": "X = 1\n",
    }
    model = FakeModel(create_script(NAME, [render(broken), render(good_files(NAME))]))
    result = create_module.create(
        _config(name=NAME, brief="count", attempts=3, workdir=tmp_path / "run"),
        client=model,
    )
    assert result.smoke_score and result.smoke_score > 0
    # The last agent call is overwatch now, so the retry is the last *emit* prompt.
    emit_prompts = [prompt for prompt in model.prompts if "Write the package" in prompt]
    retry_prompt = emit_prompts[-1]
    assert len(emit_prompts) == 2, "the run must have retried the emit once"
    assert "What went wrong last time" in retry_prompt
    assert "PACKAGE" in retry_prompt, (
        "the retry must say what was wrong; the AST check reports the manifest"
    )


def test_a_package_that_never_works_fails_loudly(tmp_path):
    broken = {"__init__.py": "PACKAGE = object()\n"}
    model = FakeModel([
        *create_script(NAME, [render(broken)] * 3)[:4],
        *[render(broken)] * 3,
    ])
    with pytest.raises(CreateError, match="does not work after"):
        create_module.create(
            _config(name=NAME, brief="count", attempts=2, workdir=tmp_path / "run"),
            client=model,
        )


def test_the_gate_refuses_a_mode_with_no_signal(tmp_path):
    """A mode with no score() and no knobs is not a deliverable."""
    hollow = dict(good_files(NAME))
    hollow["mode.py"] = (
        "from refactory_dsh import nn\n"
        "OPS = 'refactory_packages.created_pkg.ops'\n\n"
        "class Counter(nn.Mode):\n"
        "    name = 'counter'\n"
        "    def build(self):\n"
        "        graph = nn.Graph()\n"
        "        graph.add(nn.Node('finish', 'op', call=OPS + ':finish',"
        " writes=('best.json',), produces=('best',)))\n"
        "        return graph\n"
    )
    model = FakeModel(create_script(NAME, [render(hollow), render(hollow), render(good_files(NAME))]))
    result = create_module.create(
        _config(name=NAME, brief="count", attempts=3, workdir=tmp_path / "run"),
        client=model,
    )
    assert result.smoke_score and result.smoke_score > 0
    combined = "\n".join(model.prompts)
    assert "score()" in combined, "the retry must say the mode produces no signal"
    assert "knobs" in combined, "the retry must say there is nothing to optimize"


def test_extends_hands_the_model_the_reference_source(tmp_path):
    model = FakeModel(create_answers(NAME, good_files(NAME)))
    create_module.create(
        _config(
            name=NAME, brief="count", extends="circle_packing/adaptive",
            workdir=tmp_path / "run",
        ),
        client=model,
    )
    combined = "\n".join(model.prompts)
    assert "circle_packing" in combined
    assert "greedy_pack" in combined, "the reference source should be included"


# ── the phases create-v2 had and this did not ──────────────────────────────
def _tmpdir() -> Path:
    import tempfile

    return Path(tempfile.mkdtemp(prefix="create-phase-"))


def test_the_survey_names_what_already_exists():
    """It makes "does this already exist?" a question the system can answer."""
    from refactory_dsh.run import run_node

    from refactory_dsh.lowering import DECLARATIONS

    workdir = _tmpdir()
    # The runner enforces the declarations it is given, and outside a graph the
    # caller supplies them.
    (workdir / DECLARATIONS).parent.mkdir(parents=True, exist_ok=True)
    (workdir / DECLARATIONS).write_text(json.dumps({
        "survey": {"reads": [], "writes": [], "produces": ["survey", "existing"]}
    }))
    facts = run_node("refactory_dsh.create_ops:survey", workdir=workdir, node_id="survey")
    assert "circle_packing/adaptive" in facts["survey"]
    assert "annealing_pack/anneal" in facts["survey"]
    assert "circle_packing" in facts["existing"]


def test_the_strategy_prompt_carries_the_survey_and_the_schema():
    model = FakeModel(create_answers(NAME, good_files(NAME)))
    create_module.create(
        _config(name=NAME, brief="count", workdir=_tmpdir()), client=model
    )
    strategy_prompt = model.prompts[2]
    assert "What already exists" in strategy_prompt
    assert "circle_packing/adaptive" in strategy_prompt
    assert '"considered"' in strategy_prompt, "the schema must be in the prompt"
    assert "evaluator" in strategy_prompt, "a gate declares an evaluator, not a call"


def test_the_design_prompt_carries_the_chosen_approach():
    model = FakeModel(create_answers(NAME, good_files(NAME)))
    create_module.create(
        _config(name=NAME, brief="count", workdir=_tmpdir()), client=model
    )
    design_prompt = model.prompts[3]
    assert "The approach that was chosen" in design_prompt
    assert "count up one at a time" in design_prompt


def _coherent() -> dict:
    return json.loads(strategy_answer(NAME))


def _review(strategy: dict) -> list[str]:
    from refactory_dsh.create_ops import _review as review

    return review(json.dumps(strategy))


def test_a_coherent_design_passes_review():
    assert _review(_coherent()) == []


def test_review_insists_on_a_real_choice():
    strategy = _coherent()
    strategy["considered"] = [{"approach": "the only one", "why_not": ""}]
    assert any("at least two" in p for p in _review(strategy))


def test_review_requires_each_explanatory_field():
    for field in ("chosen", "why", "score"):
        strategy = _coherent()
        strategy[field] = ""
        assert any(field in p for p in _review(strategy)), field


def test_review_refuses_a_closest_existing_that_is_not_installed():
    strategy = _coherent()
    strategy["closest_existing"] = "nowhere/mode"
    assert any("not an installed mode" in p for p in _review(strategy))


def test_review_accepts_none_as_an_answer_about_what_is_closest():
    strategy = _coherent()
    strategy["closest_existing"] = "none"
    assert _review(strategy) == []


def test_review_refuses_a_design_with_no_knobs():
    strategy = _coherent()
    strategy["knobs"] = []
    assert any("no knobs" in p for p in _review(strategy))


def test_review_refuses_a_declared_graph_that_does_not_hold_together():
    """The same validator the loop uses, applied before any code exists."""
    strategy = _coherent()
    strategy["nodes"].append({"id": "orphan", "kind": "op", "call": "x:y",
                              "reads": [], "writes": [], "produces": [], "knobs": []})
    problems = _review(strategy)
    assert any("V2" in p or "not reachable" in p for p in problems), problems


def test_review_refuses_a_gate_that_is_not_a_gate():
    strategy = _coherent()
    for node in strategy["nodes"]:
        if node["id"] == "check":
            node.pop("evaluator")
            node["call"] = "x:y"
    assert any("evaluator" in p for p in _review(strategy))


def test_review_refuses_an_answer_that_is_not_json():
    from refactory_dsh.create_ops import _review as review

    assert review("prose, with no object at all")[0].startswith(
        "the strategy answer is not a JSON object"
    )


def test_the_gate_reloops_then_fails_loudly(tmp_path):
    """A design that never holds together is refused, with its reasons."""
    bad = _coherent()
    bad["considered"] = []
    model = FakeModel([RESEARCH_ANSWER, json.dumps(bad), json.dumps(bad), json.dumps(bad),
                       "a design", render(good_files(NAME))])
    with pytest.raises(CreateError, match="design is still not coherent"):
        create_module.create(
            _config(name=NAME, brief="count", attempts=1, workdir=tmp_path / "run"),
            client=model,
        )


def test_a_bad_design_is_retried_with_its_reasons(tmp_path):
    bad = _coherent()
    bad["considered"] = [{"approach": "one", "why_not": ""}]
    model = FakeModel([
        *create_script(NAME, [render(good_files(NAME))])[:2],
        json.dumps(bad), strategy_answer(NAME), "a design", render(good_files(NAME)),
        *REVIEW_ANSWERS,
    ])
    result = create_module.create(
        _config(name=NAME, brief="count", workdir=tmp_path / "run"), client=model
    )
    assert result.smoke_score and result.smoke_score > 0
    assert "at least two" in model.prompts[3], (
        "the strategy retry must be told what was wrong with the design"
    )


# ── a generated package is never loaded in this process ────────────────────
#
# Issue #23: create-mode writes Python, and deciding whether it is usable means
# loading it — which means executing model-written code. The rule is that the
# *driver* never does. These hold it, because the failure is silent: an import
# that succeeds looks exactly like a package that was never imported.


def _valid_package(name: str) -> dict[str, str]:
    """A minimal package that passes every check, so a failure is about loading."""
    ops = f"refactory_packages.{name}.ops"
    return {
        "__init__.py": (
            "from refactory_dsh.nn import Package\n"
            "from .mode import Counter\n\n"
            f"PACKAGE = Package(name={name!r}, modes=(Counter,), ops=({ops!r},))\n"
        ),
        "ops.py": (
            "def start(ctx):\n"
            "    ctx.write_json('state.json', {'value': 0})\n"
            "    return {'value': 0}\n\n"
            "def finish(ctx):\n"
            "    ctx.write_json('best.json', {'best': ctx.knob('target', 3)})\n"
            "    return {'best': ctx.knob('target', 3)}\n"
        ),
        "mode.py": (
            "from refactory_dsh import nn\n"
            f"OPS = {ops!r}\n\n"
            "class Counter(nn.Mode):\n"
            "    name = 'counter'\n"
            "    @classmethod\n"
            "    def declare(cls):\n"
            "        return {'target': nn.Knob(3, 'int', lo=1, hi=9)}\n"
            "    def build(self):\n"
            "        g = nn.Graph()\n"
            "        g.add(nn.Node('start', 'op', call=OPS + ':start',"
            " writes=('state.json',), produces=('value',), knobs=('target',)))\n"
            "        g.add(nn.Node('finish', 'op', call=OPS + ':finish',"
            " writes=('best.json',), produces=('best',)))\n"
            "        g.connect('start', 'finish')\n"
            "        return g\n"
            "    def score(self, run):\n"
            "        return float(run.facts['best'])\n"
        ),
    }


def _stage(tmp_path: Path, name: str, files: dict[str, str]) -> None:
    from refactory_dsh import registry

    root = registry.workspace_root(tmp_path) / name
    root.mkdir(parents=True, exist_ok=True)
    for filename, body in files.items():
        (root / filename).write_text(body)
    registry.install_workspace(tmp_path)


def _record_package_imports(monkeypatch, name: str) -> list[str]:
    """Every import of this package, from anywhere in this process."""
    import importlib
    from refactory_dsh import registry

    module_name = f"{registry.ROOT_PACKAGE}.{name}"
    imported: list[str] = []
    real = importlib.import_module

    def recording(target: str, *args, **kwargs):
        if target == module_name or target.startswith(module_name + "."):
            imported.append(target)
        return real(target, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", recording)
    return imported


def test_check_package_never_imports_the_package_it_checks(tmp_path, monkeypatch):
    """The whole point of #23: the check must not run model code in the driver."""
    _stage(tmp_path, "spied_pkg", _valid_package("spied_pkg"))
    monkeypatch.chdir(tmp_path)
    imported = _record_package_imports(monkeypatch, "spied_pkg")
    problems = create_ops.check_package("spied_pkg")
    assert problems == [], problems
    assert imported == [], (
        f"check_package imported the generated package in this process: {imported}"
    )


def test_reading_a_package_manifest_does_not_import_it(tmp_path, monkeypatch):
    """`register` and `smoke` need the mode ids, not the module in this process."""
    _stage(tmp_path, "manifest_pkg", _valid_package("manifest_pkg"))
    imported = _record_package_imports(monkeypatch, "manifest_pkg")
    described = sandbox.describe_package("manifest_pkg", project=tmp_path)
    assert described.ok, described.stderr
    assert described.payload["ids"], "the manifest names no mode"
    assert imported == [], f"describe_package imported it here: {imported}"


def test_a_creation_asked_for_a_data_node_is_checked_for_one():
    """`--data-node` was prompt text; the gate now checks what it asked for.

    A run asked for a mode that iterates data could emit one that never iterates and
    validate cleanly, so the run looked successful and the mode did nothing the brief
    asked for.
    """
    import shutil as _shutil

    from refactory_dsh import registry
    from refactory_dsh.create_ops import check_package
    from tests_support import good_files

    root = registry.workspace_root(Path(".")) / "checked_pkg"
    _shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    for filename, body in good_files("checked_pkg").items():
        (root / filename).write_text(body)
    try:
        registry.install_workspace(Path.cwd())
        assert check_package("checked_pkg") == [], "the fixture package should be clean"
        problems = check_package("checked_pkg", requires_data_node=True)
        assert any("DataNode" in problem for problem in problems), (
            f"a data-node creation was not told its mode does not iterate: {problems}"
        )
        bound = check_package("checked_pkg", requires_task_ref="probe:Task")
        assert any("probe:Task" in problem for problem in bound), (
            f"a Task-bound creation was not told its mode is unbound: {bound}"
        )
    finally:
        _shutil.rmtree(root, ignore_errors=True)


# ── a harness is for a class of work ───────────────────────────────────────
#
# A Task names a class of work, and naming one is the data-node directive: the
# produced package binds a `data` node to that Task, and the executor runs its
# subgraph once per item. The single-shot harness is the special case, and it has
# to be asked for by name.


def test_naming_only_a_task_yields_a_data_node_harness_bound_to_it(tmp_path):
    """A Task alone is the data-node directive: no second flag is asked for."""
    model = FakeModel(create_answers(NAME, good_data_files(NAME, TASK)))
    run_dir = tmp_path / "run"
    result = create_module.create(
        _config(name=NAME, brief="process every item a Task lists", task_ref=TASK,
                workdir=run_dir),
        client=model,
    )
    assert result.package == NAME
    # The Task's own verdicts: CountingTask scores alpha 1.0 and beta 0.5.
    assert result.smoke_score == 0.75

    problems = create_ops.check_package(
        NAME, requires_data_node=True, requires_task_ref=TASK
    )
    assert problems == [], problems

    _package, mode_class = registry.resolve(Path("."), f"{NAME}/chore")
    bound = [
        node for node in mode_class().build().nodes.values()
        if node.kind == "data" and node.task_ref == TASK
    ]
    assert bound, "naming a Task did not bind a data node to it"
    assert bound[0].subgraph_entry and bound[0].subgraph_exit

    recorded = json.loads((run_dir / "create.json").read_text())
    assert recorded["task_ref"] == TASK, "the run's own facts must name the Task"
    assert "once per item" in recorded["class_of_work"]
    assert recorded["data_node"] == "yes"
    assert create_ops.baked_item_problems(result.path) == [], (
        "the harness must take its items from the Task at run time, not from the "
        "items the authoring run seeded"
    )


def test_a_harness_that_bakes_its_items_in_is_refused(tmp_path, monkeypatch):
    """`inline_items` is read before the Task, so it closes the harness's input port.

    A package whose items are baked in is the hardcoded-sample failure one level up:
    it runs the items the authoring run seeded and never the ones the class of work
    supplies.
    """
    _stage(tmp_path, NAME, baked_data_files(NAME, TASK))
    monkeypatch.chdir(tmp_path)
    problems = create_ops.check_package(NAME)
    assert any("inline_items" in problem for problem in problems), problems
    assert any("run time" in problem for problem in problems)


def test_a_data_node_that_names_no_source_is_refused(tmp_path, monkeypatch):
    """A data node with no Task and no source iterates nothing, so it is not a harness."""
    _stage(tmp_path, NAME, unfed_data_files(NAME))
    monkeypatch.chdir(tmp_path)
    problems = create_ops.check_package(NAME)
    assert any("no way to be fed at run time" in problem for problem in problems), problems


def test_prepare_refuses_a_creation_that_names_no_task_and_is_not_single(tmp_path):
    """The refusal is before the first model call: `prepare` is the first node."""
    from refactory_dsh import create_ops
    from refactory_dsh.nn import RunContext

    ctx = RunContext(
        tmp_path, knobs={"name": "nowhere_pack", "brief": "count"}, facts={}, task={}
    )
    with pytest.raises(ValueError, match="--single") as raised:
        create_ops.prepare(ctx)
    assert "--task" in str(raised.value)
    assert "class of work is the default" in str(raised.value)


# prepare() checks the model credential, here a placeholder in the test's own home (dm-257).
@pytest.mark.usefixtures("configured_model_home")
def test_an_edit_is_not_a_creation_and_needs_no_directive(tmp_path):
    """`edit` walks the same graph with a mode to revise, which is not a creation."""
    from refactory_dsh import create_ops
    from refactory_dsh.nn import RunContext

    ctx = RunContext(tmp_path, knobs={"name": "revised", "brief": "fewer steps"}, facts={})
    ctx.task = {"name": "revised", "brief": "fewer steps", "mode": "counting_pack/fixed_step"}
    facts = create_ops.prepare(ctx)
    assert facts["revision_of"] == "counting_pack/fixed_step"


def test_neither_a_task_nor_single_is_refused_before_any_model_call(tmp_path):
    """The default is the class of work; silence is refused, not quietly defaulted."""
    model = FakeModel(create_answers(NAME, good_files(NAME)))
    with pytest.raises(CreateError, match="--single") as raised:
        create_module.create(
            CreateConfig(
                name=NAME, brief="count", workdir=tmp_path / "run",
                verdicts={"gate_strategy": "proceed"},
            ),
            client=model,
        )
    message = str(raised.value)
    assert "--task" in message and "class of work is the default" in message
    assert model.prompts == [], "the refusal must happen before any model call"


def test_single_still_produces_the_single_shot_package(tmp_path):
    """`--single` is the old product, asked for by name: no Task, no data node."""
    model = FakeModel(create_answers(NAME, good_files(NAME)))
    result = create_module.create(
        _config(name=NAME, brief="count", workdir=tmp_path / "run"), client=model
    )
    assert result.smoke_score and result.smoke_score > 0
    assert create_ops.check_package(NAME) == []
    _package, mode_class = registry.resolve(Path("."), f"{NAME}/counter")
    assert not [n for n in mode_class().build().nodes.values() if n.kind == "data"]


def test_the_command_line_carries_single_and_task(monkeypatch, capsys):
    """The two shell spellings reach the config, and one of them is required."""
    seen: list[CreateConfig] = []

    class Result:
        def describe(self) -> dict:
            return {}

    monkeypatch.setattr("refactory_dsh.model.DshModelClient", lambda **kwargs: None)
    monkeypatch.setattr(
        create_module, "create",
        lambda config, client=None, **kwargs: (seen.append(config), Result())[1],
    )
    assert create_module.main(["--name", "one_pack", "--brief", "count", "--single"]) == 0
    assert seen[0].single_shot is True and seen[0].task_ref == ""
    assert create_module.main(
        ["--name", "one_pack", "--brief", "count", "--task", TASK]
    ) == 0
    assert seen[1].task_ref == TASK
    capsys.readouterr()


def test_the_command_line_refuses_a_creation_with_neither(monkeypatch, capsys):
    """Without `--task` or `--single`, the shell command exits with the reason."""
    monkeypatch.setattr("refactory_dsh.model.DshModelClient", lambda **kwargs: None)
    assert create_module.main(["--name", "nowhere_pack", "--brief", "count"]) == 2
    reported = capsys.readouterr().err
    assert "--task" in reported and "--single" in reported


# ── the path convention (#65) ──────────────────────────────────────────────
#
# An emit answer is written relative to the package's own root, and a model that
# reads "relative path" as relative to the checkout writes
# `refactory/refactory_packages/<name>/mode.py`. Joined onto the package root that
# nested the package inside itself, and the run then failed with "__init__.py is
# missing" four directories away from the mistake — with the guard passing, because
# the nested path really is inside the root.


@pytest.mark.parametrize("prefix", [
    "{name}/",
    "refactory/refactory_packages/{name}/",
    "refactory_packages/{name}/",
    ".factory/packages/{name}/",
])
def test_a_leading_package_prefix_is_stripped(tmp_path, prefix):
    from refactory_dsh.nn import RunContext

    workdir = tmp_path / "run"
    workdir.mkdir()
    stem = prefix.format(name=NAME)
    (workdir / "emit.txt").write_text(
        f"===FILE: {stem}__init__.py===\nfrom refactory_dsh.nn import Package\n"
        f"===FILE: {stem}ops.py===\n"
    )
    ctx = RunContext(workdir, facts={"name": NAME}, task={})
    ctx._declared = ("staging/manifest.json",)
    ctx._declared_facts = ("package_path", "files")
    produced = create_ops.materialise(ctx)
    assert sorted(produced["files"]) == ["__init__.py", "ops.py"], (
        "the prefix restates the package root, so it must not become a directory"
    )
    root = Path(produced["package_path"])
    assert (root / "__init__.py").exists()
    assert not (root / "refactory").exists()


def test_an_emit_path_that_cannot_be_package_relative_is_refused():
    for bad in ("/etc/passwd", "../outside.py", ""):
        with pytest.raises(ValueError, match="relative"):
            create_ops._package_relative(bad, NAME)
    # A subdirectory inside the package is legitimate and stays one.
    assert create_ops._package_relative("modes/counter.py", NAME) == "modes/counter.py"


@pytest.mark.parametrize("char, named", [
    ("\x7f", "U+007F (a control character)"), ("\x9b", "U+009B (a control character)"),
    ("\u202e", "U+202E RIGHT-TO-LEFT OVERRIDE"), ("\u2066", "U+2066 LEFT-TO-RIGHT ISOLATE"),
    ("\u200b", "U+200B ZERO WIDTH SPACE"), ("\ufeff", "U+FEFF ZERO WIDTH NO-BREAK SPACE"),
], ids=["DEL", "C1-control", "bidi-override", "bidi-isolate", "zero-width-space", "byte-order-mark"])
def test_an_emit_path_holding_a_control_or_format_character_is_refused_by_name(char, named):
    """dm-208: create mode's path check refuses what `_artifact_paths` refuses."""
    import re as _re
    with pytest.raises(ValueError, match=_re.escape(named)):
        create_ops._package_relative(f"modes/count{char}er.py", NAME)
    # A joiner Persian names are written with is not one of them.
    persian = "docs/\u0645\u06cc\u200c\u062e\u0648\u0627\u0647\u0645.md"
    assert create_ops._package_relative(persian, NAME) == persian


@pytest.mark.parametrize("bad_path, said", [
    ("mode\u202e.py", "U+202E RIGHT-TO-LEFT OVERRIDE"),
    ("../outside.py", "does not resolve to a file inside the package root"),
    ("/etc/outside.py", "which is not a path inside the package"),
], ids=["bidi", "dotdot", "absolute"])
def test_create_fast_sends_a_path_it_cannot_place_back_to_emit(tmp_path, monkeypatch, bad_path, said):
    """Review of #519: a path `_package_relative` refuses goes back to emit with the reason,
    as a malformed header does, instead of ending the run; nothing of it is written."""
    from refactory_dsh.fast import CreateFastMode

    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    name = "fast_counter"
    mode = CreateFastMode()
    state = mode.state_dict()
    state["knobs"]["single_shot"]["value"] = True
    mode.load_state_dict(state)
    bad_emission = render(good_files(name)) + f"===FILE: {bad_path}===\ncontent\n"
    client = FakeModel([PLAN, bad_emission, render(good_files(name)), QA_PASS])
    run = mode.run(tmp_path / "run", model=client, task={"name": name, "brief": "a reusable counter"})
    assert run.exit_ok, run.error
    emit_prompts = [p for p in client.prompts if "emit the package" in p]
    assert len(emit_prompts) == 2, "emit must run twice"
    assert said in emit_prompts[-1], "the retry prompt must say why the path was refused"
    assert not list(tmp_path.rglob("outside.py")) and not list(tmp_path.rglob("mode\u202e.py"))


def test_the_missing_manifest_reason_names_the_root_and_what_is_there(tmp_path):
    """The retry prompt has to say something the model can act on.

    "…is missing; a package needs one" is true and useless when the mistake is a
    path four directories long.
    """
    from refactory_dsh import sandbox

    nested = tmp_path / "pkg" / "refactory" / "refactory_packages" / "pkg"
    nested.mkdir(parents=True)
    (nested / "mode.py").write_text("X = 1\n")
    message = sandbox.missing_init(tmp_path / "pkg")
    assert str(tmp_path / "pkg") in message
    assert "refactory" in message, "the message must say what the root holds instead"
    assert "relative to the package root" in message


def test_each_package_gets_its_own_run_directory(monkeypatch, capsys):
    """Two creations must not share one workdir (#64).

    `--workdir` defaulted to the fixed `.factory/create`, so a second run wrote its
    `brief.json`, research and strategy documents over the first one's — and a
    cancelled run's leftovers were what the next run read.
    """
    seen: list[Path] = []

    class Result:
        def describe(self) -> dict:
            return {}

    monkeypatch.setattr("refactory_dsh.model.DshModelClient", lambda **kwargs: None)
    monkeypatch.setattr(
        create_module, "create",
        lambda config, client=None, **kwargs: (seen.append(Path(config.workdir)), Result())[1],
    )
    for name in ("one_pack", "two_pack"):
        assert create_module.main(["--name", name, "--brief", "count"]) == 0
    assert seen[0] != seen[1], "two packages shared one run directory"
    assert seen[0].parts[-2:] == ("create", "one_pack")
    assert seen[1].parts[-2:] == ("create", "two_pack")
    capsys.readouterr()


# ── the research stage (#63) ───────────────────────────────────────────────
#
# The stage that grounds the design in facts produced nothing, twice, in independent
# runs: `research/` was empty and `research-plan.json` held a prose status message
# about subagents "running in the background". `writes=("research-plan.json",)` is
# satisfied by any bytes, so nothing noticed.


def _research_ctx(workdir: Path, plan: str) -> object:
    from refactory_dsh.nn import RunContext

    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "research-plan.json").write_text(plan)
    ctx = RunContext(workdir, knobs={"research_attempts": 2}, facts={}, task={})
    ctx._declared = ("research/", "research-review.txt")
    ctx._declared_facts = ("findings_recorded",)
    return ctx


def test_the_research_gate_refuses_prose_in_the_plan(tmp_path):
    ctx = _research_ctx(tmp_path, "research: the subagents are running in the background")
    verdict = create_ops.gate_research(ctx)
    assert verdict == "reloop"
    assert "running in the background" in verdict.reason
    review = (tmp_path / "research-review.txt").read_text()
    assert "not a JSON array" in review
    assert "running in the background" in review, (
        "the retry must be shown what it wrote, or it writes the same thing again"
    )
    assert "running in the background" in ctx.facts["research_review"]


def test_a_plan_whose_findings_were_never_written_is_refused(tmp_path):
    plan = json.dumps([
        {"focus": "one", "slug": "one", "finding": "answered"},
        {"focus": "two", "slug": "two", "finding": "answered"},
        {"focus": "three", "slug": "three", "finding": "answered"},
    ])
    ctx = _research_ctx(tmp_path, plan)
    # `record_findings` is what puts them on disk; without it the gate refuses.
    assert create_ops.gate_research(ctx) == "reloop"
    assert "research/one.md is missing or empty" in (tmp_path / "research-review.txt").read_text()
    create_ops.record_findings(ctx)
    assert create_ops.gate_research(ctx) == "proceed"
    assert (tmp_path / "research/one.md").read_text().strip().endswith("answered")


def test_a_finding_a_subagent_already_wrote_is_left_alone(tmp_path):
    plan = json.dumps([
        {"focus": "one", "slug": "one", "finding": "the short version"},
    ])
    ctx = _research_ctx(tmp_path, plan)
    (tmp_path / "research").mkdir()
    (tmp_path / "research/one.md").write_text("the long form a subagent wrote\n")
    produced = create_ops.record_findings(ctx)
    assert produced["findings_recorded"] == []
    assert "long form" in (tmp_path / "research/one.md").read_text()


def test_a_plan_that_stops_being_a_plan_fails_loudly(tmp_path):
    ctx = _research_ctx(tmp_path, "still prose")
    assert create_ops.gate_research(ctx) == "reloop"
    with pytest.raises(ValueError, match="still not usable after 2 attempt"):
        create_ops.gate_research(ctx)


def test_the_research_prompt_states_the_shape_and_the_retry(tmp_path):
    """The model has to be told the array it owes, and what was wrong last time."""
    prompt = CreateMode().build().nodes["research_director"].prompt_template
    assert "research-plan.json" in prompt
    assert '"slug"' in prompt and '"finding"' in prompt
    assert "{{research_review}}" in prompt


def test_a_created_packages_artifacts_stay_in_the_run_directory(tmp_path):
    """Every agent node's completion runs in the run directory, not the checkout.

    The model client spawns an agent with tools; run from the checkout, the reviews it
    writes land in the repository, the run's own `reviews/` stays empty, and the QA
    gate reads nothing. The walker tells the client where the node lives.
    """
    model = FakeModel(create_answers(NAME, good_files(NAME)))
    workdir = tmp_path / "run"
    create_module.create(_config(name=NAME, brief="count", workdir=workdir), client=model)
    assert model.workdirs, "no completion was told where to run"
    assert all(Path(str(where)).resolve() == workdir.resolve() for where in model.workdirs), (
        f"a completion ran outside the run directory: {set(map(str, model.workdirs))}"
    )
    for artifact in ("reviews/health-check.md", "reviews/code-review.md", "reviews/qa.md"):
        assert (workdir / artifact).exists(), f"{artifact} stayed out of the run directory"
        assert not (REPO_ROOT / artifact).exists(), (
            f"{artifact} was written into the checkout instead of the run directory"
        )


# ── the brief's inputs are checked (#67) ───────────────────────────────────
#
# `validate` proved a package was structurally a mode, not that it did what was
# asked. A mode whose whole subject was a Python package took no package at all — it
# built a fixture hardcoded in `ops.py` — and passed every check. The brief's inputs
# are now declarable, and the check is whether the mode has a *way to receive* each
# one: a knob named after it, or a read under that name.


def test_a_mode_that_cannot_receive_the_briefs_input_is_refused(tmp_path, monkeypatch):
    """The counter takes `target` and reads `state.json`, so it cannot take a package."""
    _stage(tmp_path, NAME, _valid_package(NAME))
    monkeypatch.chdir(tmp_path)
    problems = create_ops.check_package(NAME, requires_input=("package",))
    assert any("no way to receive 'package'" in problem for problem in problems), problems
    assert any("declare a knob named after it" in problem for problem in problems)


def test_a_mode_that_takes_the_input_passes(tmp_path, monkeypatch):
    """A knob named after the input is the caller's handle on it."""
    files = _valid_package(NAME)
    files["mode.py"] = files["mode.py"].replace(
        "        return {'target': nn.Knob(3, 'int', lo=1, hi=9)}\n",
        "        return {'package_path': nn.Knob('', 'str'),\n"
        "                'target': nn.Knob(3, 'int', lo=1, hi=9)}\n",
    ).replace(
        "knobs=('target',)))",
        "knobs=('target', 'package_path')))",
    )
    _stage(tmp_path, NAME, files)
    monkeypatch.chdir(tmp_path)
    assert create_ops.check_package(NAME, requires_input=("package",)) == []


def test_declared_inputs_refuses_a_name_too_short_to_check():
    assert create_ops.declared_inputs("package, source_tree") == ("package", "source_tree")
    assert create_ops.declared_inputs("") == ()
    with pytest.raises(ValueError, match="too short to check"):
        create_ops.declared_inputs("a")


# prepare() checks the model credential, here a placeholder in the test's own home (dm-257).
@pytest.mark.usefixtures("configured_model_home")
def test_the_design_and_emit_prompts_are_told_what_the_mode_must_take(tmp_path):
    """The rule reaches the two prompts that decide the interface and the code."""
    graph = CreateMode().build()
    for node_id in ("design", "emit"):
        prompt = graph.nodes[node_id].prompt_template
        assert "{{requires_input}}" in prompt, node_id
    from refactory_dsh.nn import RunContext

    ctx = RunContext(
        tmp_path,
        knobs={"name": NAME, "brief": "count", "requires_input": "package", "single_shot": 1},
        facts={},
        task={},
    )
    published = create_ops.prepare(ctx)
    assert "takes these as input: `package`" in published["requires_input"]
    assert "hardcodes a fixture" in published["requires_input"]


# ── the input path and the artifact are checked (#69) ──────────────────────
#
# `validate` and smoke ran the *default* knob path, so a revision whose new input
# path emitted a `.dist-info` holding only `RECORD` — not a wheel — scored 1.0 on the
# built-in fixture and was accepted. Nothing supplied the input the brief asked for,
# and nothing looked at what came out.


def test_a_wheel_that_is_not_a_wheel_is_refused(tmp_path):
    """A file that says it is a wheel has to be one: METADATA and WHEEL beside RECORD."""
    import zipfile

    root = tmp_path / "smoke"
    (root / "out").mkdir(parents=True)
    good = root / "out" / "good-1.0-py3-none-any.whl"
    with zipfile.ZipFile(good, "w") as archive:
        archive.writestr("good/__init__.py", "")
        for member in ("RECORD", "METADATA", "WHEEL"):
            archive.writestr(f"good-1.0.dist-info/{member}", "")
    assert create_ops.artifact_problems(root) == []

    bad = root / "out" / "bad-1.0-py3-none-any.whl"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("bad/__init__.py", "")
        archive.writestr("bad-1.0.dist-info/RECORD", "")
    problems = create_ops.artifact_problems(root)
    assert len(problems) == 2, problems
    assert all("bad-1.0-py3-none-any.whl" in problem for problem in problems)
    assert any("METADATA" in problem for problem in problems)
    assert any("WHEEL" in problem for problem in problems)


def test_a_file_that_is_not_an_archive_at_all_is_refused(tmp_path):
    root = tmp_path / "smoke"
    root.mkdir()
    (root / "broken.whl").write_bytes(b"this is not a zip")
    problems = create_ops.artifact_problems(root)
    assert len(problems) == 1 and "not a readable archive" in problems[0]


def test_the_smoke_runs_the_knobs_a_sample_names(tmp_path, monkeypatch):
    """A sample is how the brief's inputs are exercised at all (#69)."""
    from refactory_dsh import sandbox as sandbox_module
    from refactory_dsh.nn import RunContext

    seen: dict[str, object] = {}

    def fake_smoke(identifier, workdir, *, project=None, model=False, knobs=None, **kwargs):
        seen["knobs"] = knobs
        return sandbox_module.SandboxResult(ok=True, payload={"score": 0.5, "iterations": 3})

    monkeypatch.setattr(sandbox_module, "smoke", fake_smoke)
    monkeypatch.setattr(sandbox_module, "describe_package", lambda name, **kwargs: (
        sandbox_module.SandboxResult(ok=True, payload={"ids": [f"{name}/counter"]})
    ))
    monkeypatch.setattr("refactory_dsh.create_ops._agent_nodes", lambda name: [])

    workdir = tmp_path / "smoke"
    workdir.mkdir()
    ctx = RunContext(workdir, knobs={}, facts={"name": NAME}, task={})
    ctx.knobs["sample_knobs"] = json.dumps(["target=7"])
    ctx._declared = ("smoke.json",)
    ctx._declared_facts = ("smoke_score", "smoke_ok")
    produced = create_ops.smoke(ctx)
    assert produced["smoke_score"] == 0.5
    assert seen["knobs"] == {"target": "7"}, "the smoke ran the default path instead"


def test_a_sample_must_be_name_value():
    with pytest.raises(ValueError, match="--sample takes NAME=VALUE"):
        create_ops._sample_knobs(["target"])


# ── what the hard run found in the pipeline ────────────────────────────────
#
# Two defects the sanity check surfaced, both in the pipeline rather than in the
# mode it produced. The overwatch pass refused to state PASS and named them.


def test_a_bolded_blocking_finding_is_not_invisible_to_the_qa_gate():
    """`**BLOCKING:** x` is a blocking finding, and `startswith` did not see it.

    A real run merged three reviews into a synthesis that said "0 blocking
    finding(s)" and let the gate say `proceed`, while the same body held four bolded
    blockers. The overwatch pass caught it; the counter should not have needed it to.
    """
    from refactory_dsh.nn import RunContext

    for marker in ("BLOCKING: the test is deleted", "**BLOCKING:** the test is deleted",
                   "- **BLOCKING** x", "  * BLOCKING: x"):
        workdir = Path(tempfile.mkdtemp())
        (workdir / "reviews").mkdir()
        (workdir / "reviews/qa.md").write_text(f"# QA\n\n{marker}\n")
        (workdir / "reviews/health-check.md").write_text("# Health\nok\n")
        (workdir / "reviews/code-review.md").write_text("# Code\nok\n")
        ctx = RunContext(workdir, knobs={}, facts={}, task={})
        ctx._declared = ("reviews/qa-synthesized.md",)
        ctx._declared_facts = ("qa", "qa_blocking", "qa_missing")
        produced = create_ops.synthesize_qa(ctx)
        assert produced["qa_blocking"] == 1, f"{marker!r} counted as {produced['qa_blocking']}"
        assert "1 blocking finding(s)" in produced["qa"]
        ctx._declared = ("reviews/qa-gate.txt", "reviews/finding-sources.json")
        assert create_ops.gate_qa(ctx) == "reloop"


def test_a_mode_that_raises_is_not_reported_as_a_measured_zero(monkeypatch, tmp_path):
    """The sandbox exiting 0 while the mode failed is "not measured", not "0.0"."""
    from refactory_dsh import sandbox as sandbox_module
    from refactory_dsh.nn import RunContext

    monkeypatch.setattr(
        sandbox_module, "describe_package",
        lambda name, **kwargs: sandbox_module.SandboxResult(ok=True, payload={"ids": [f"{name}/counter"]}),
    )
    monkeypatch.setattr("refactory_dsh.create_ops._agent_nodes", lambda name: ["draft"])
    monkeypatch.setattr(
        sandbox_module, "smoke",
        lambda *args, **kwargs: sandbox_module.SandboxResult(
            ok=True, payload={"exit_ok": False, "error": "the mode raised", "score": 0.0},
        ),
    )
    workdir = tmp_path / "smoke"
    workdir.mkdir()
    ctx = RunContext(workdir, knobs={}, facts={"name": NAME}, task={})
    ctx._declared = ("smoke.json",)
    ctx._declared_facts = ("smoke_score", "smoke_ok")
    with pytest.raises(ValueError, match="did not finish a run with a model"):
        create_ops.smoke(ctx)


def test_a_review_that_reloops_the_build_says_why_into_the_rebuild(tmp_path):
    """A gate that sends the build back has to hand its reasons to the next attempt.

    The hard sanity run burned two full review rounds (~an hour) refusing the same
    findings, because `gate_qa` and `gate_overwatch` wrote their reasons to a file for
    a reader and set no fact the emit prompt reads: the rebuild re-emitted the same
    design. `review_design` and `validate` already feed theirs back. The verdict
    carries the same reason, which the runner prints for the CEO driving the run.
    """
    from refactory_dsh.nn import RunContext

    graph = CreateMode().build()
    prompt = graph.nodes["emit"].prompt_template
    assert "{{qa_gate}}" in prompt and "{{overwatch_review}}" in prompt
    assert "qa_gate" in graph.nodes["gate_qa"].produces
    assert "overwatch_review" in graph.nodes["gate_overwatch"].produces

    (tmp_path / "reviews").mkdir()
    (tmp_path / "reviews/qa-synthesized.md").write_text(
        "# QA synthesis\n\n- **BLOCKING:** adding pytest.ini makes it score 1.0\n"
    )
    ctx = RunContext(tmp_path, knobs={}, facts={}, task={})
    for name in create_ops.QA_REVIEWS:
        ctx.write_text(name, "BLOCKING: adding pytest.ini makes it score 1.0")
    create_ops.synthesize_qa(ctx)
    ctx._declared = ("reviews/qa-gate.txt", "validation.txt", "reviews/finding-sources.json")
    ctx._declared_facts = ("qa_gate", "validation")
    verdict = create_ops.gate_qa(ctx)
    assert verdict == "reloop"
    assert "adding pytest.ini" in ctx.facts["qa_gate"]
    assert "adding pytest.ini" in verdict.reason and not verdict.reason.startswith("reloop")

    (tmp_path / "reviews/overwatch.md").write_text(
        "Wrote the audit.\n\nVERDICT: FAIL\n\nThe scorer can be gamed by deleting the test.\n"
    )
    ctx._declared = ("reviews/overwatch-gate.txt", "validation.txt")
    ctx._declared_facts = ("overwatch_review", "validation")
    verdict = create_ops.gate_overwatch(ctx)
    assert verdict == "reloop"
    assert "deleting the test" in ctx.facts["overwatch_review"]
    assert "deleting the test" in verdict.reason


def test_the_design_review_and_the_package_gate_say_why_they_send_work_back(tmp_path):
    """Each reloop carries the reason its gate wrote, for the CEO as for the retry."""
    from refactory_dsh.nn import RunContext

    ctx = RunContext(tmp_path, knobs={}, facts={"name": "no_such_package_here"}, task={})
    ctx.write_text("strategy.json", "a plan in prose, not JSON")
    verdict = create_ops.review_design(ctx)
    assert verdict == "reloop" and verdict.reason and verdict.reason == ctx.facts["design_review"]

    ctx.write_json("brief.json", {"task_refs": [], "data_node": False})
    verdict = create_ops.validate(ctx)
    assert verdict == "reloop" and verdict.reason and verdict.reason == ctx.facts["validation"]
