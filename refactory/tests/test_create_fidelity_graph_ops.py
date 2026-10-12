"""Independent graph contract and real native portable operation boundaries."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from refactory_dsh.create_fidelity import CreateFidelityMode
from refactory_dsh import create_fidelity_ops as ops
from refactory_dsh.create_fidelity_prompts import reference, load_prompt
from refactory_dsh import create_fidelity_prompts as prompts
from refactory_dsh.nn import Graph, Executor, Node, RunContext, execute, validate_state

FIXTURE = Path(__file__).parent / "fixtures/create_fidelity/v1/reference.json"
PINNED_TEMPLATE_SHA256 = "50d8a754f707e915a0b13bd8a235d201b1930939d3b6801958dde9242e8a672e"


def test_exact_source_topology_roles_gates_and_original_artifact_contracts():
    source = json.loads(FIXTURE.read_text())
    mode = CreateFidelityMode()
    graph = mode.graph()
    assert graph.start == source["graph"]["start_node"]
    assert len(graph.nodes) == source["node_count"] == 29
    assert len(graph.edges) == source["edge_count"] == 33
    assert set(graph.nodes) == set(source["graph"]["nodes"])
    assert {edge.key() for edge in graph.edges} == {(e["source"], e["target"], e["condition"] or "") for e in source["graph"]["edges"]}
    for identifier, original in source["graph"]["nodes"].items():
        node = graph.nodes[identifier]
        assert set(original["reads"]) <= set(node.reads)
        assert set(original["writes"]) <= set(node.writes)
        assert node.blocking == original["blocking"]
        if original.get("timeout"):
            assert node.timeout_s == original["timeout"]
        if original.get("role"):
            assert node.role == original["role"]
        if original.get("evaluator_type") in ("agent", "user"):
            assert node.gate_kind == original["evaluator_type"]
            assert original["gate_prompt"] in node.prompt_template
        if identifier.endswith("_director"):
            assert node.kind == "op"
            assert node.call.endswith(":" + identifier)
            assert node.timeout_s == 3600
    assert graph.nodes["fork_qa"].targets == graph.nodes["join_qa"].sources
    assert not validate_state(mode.state_dict(), invariants=mode.invariants())


def test_native_human_question_preserves_source_review_criteria():
    from refactory_dsh.lowering import to_workflow
    mode = CreateFidelityMode()
    gate = to_workflow(mode.state_dict()).nodes["gate_strategy"]
    assert gate.evaluator_type == "user"
    assert gate.gate_prompt == mode.graph().nodes["gate_strategy"].prompt_template
    assert "INTENT FIDELITY CHECK" in gate.gate_prompt
    assert "Graph Topology" in gate.gate_prompt


def test_production_prompts_preserve_full_source_and_native_mapping():
    frozen = json.loads(FIXTURE.read_text())
    shipping = reference()
    assert shipping["source_commit"] == frozen["source_commit"]
    assert shipping["source_graph_sha256"] == frozen["graph_sha256"]
    for original_path, original_text in frozen["role_prompts"].items():
        role = Path(original_path).stem
        assert original_text in load_prompt(role)
    for role in ("builder", "health_checker", "code_reviewer", "researcher", "strategist"):
        prompt = load_prompt(role, role if role not in ("researcher", "strategist") else "research_director")
        assert "No score override or Task" not in prompt  # supplied by real native API at render time
        assert "artifact-only local authoring run" in prompt
        assert "runtime knobs" in prompt


def test_canonical_bootstrap_template_matches_independent_source_pin_and_rejects_tampering(tmp_path, monkeypatch):
    supplied = prompts.factory_template()
    assert supplied["source_commit"] == "3ab6f8f7ca792c874f2654574d7f7fd6dd18b896"
    assert supplied["source_path"] == "templates/factory_config.md"
    assert hashlib.sha256(supplied["text"].encode()).hexdigest() == PINNED_TEMPLATE_SHA256
    assert supplied["source_sha256"] == PINNED_TEMPLATE_SHA256
    altered = dict(supplied, text=supplied["text"] + "\nChanged canonical input.\n")
    # A resource cannot bless its own changed bytes by changing its recorded hash.
    altered["source_sha256"] = hashlib.sha256(altered["text"].encode()).hexdigest()
    resource = tmp_path / "altered-template.json"
    resource.write_text(json.dumps(altered))
    monkeypatch.setattr(prompts, "_FACTORY_TEMPLATE_RESOURCE", resource)
    with pytest.raises(ValueError, match="pinned source"):
        prompts.factory_template()


def test_substantive_input_delivery_preserves_template_looking_data_bytes(tmp_path):
    ctx = RunContext(tmp_path, facts={"name": "authored_name"})
    raw = "# Complete child evidence\n\nCode example keeps literal {{name}} and {{native_api}} as input data.\n"
    ctx.write_text("report.md", raw)
    node = Node("synthesis", "agent", prompt_template="Author {{name}} from actual evidence", reads=("report.md",), writes=("strategy.md",))
    prompt = ops.agent_prompt(ctx, node)
    assert prompt.startswith("Author authored_name from actual evidence")
    assert raw in prompt
    assert "literal authored_name" not in prompt


def test_unsupported_variant_rejected_before_any_dispatch(tmp_path):
    class NoCalls:
        def complete(self, *args, **kwargs):
            pytest.fail("unsupported variant must fail before model I/O")
    for variant in ("plugin", "update-existing", "provider-swap"):
        workdir = tmp_path / variant
        result = execute(CreateFidelityMode().graph(), workdir,
                         knobs={"name": "portable_test", "brief": "Produce a reusable local mode", "variant": variant}, model=NoCalls())
        assert not result.exit_ok
        assert "unsupported" in result.error
        assert not (workdir / ".factory/create-request.json").exists()


class BootstrapIO:
    """Only external model responses are scripted; real graph/ops/delivery run."""
    def __init__(self):
        self.calls = []
        self.template_input = None
    def complete(self, prompt, *, system="", workdir=None):
        contract = json.loads(prompt.split("<!-- refactory worker execution -->", 1)[1].split("<!-- end refactory worker execution -->", 1)[0].split("\n", 2)[2])
        node = contract["node"]
        self.calls.append(node)
        output = Path(contract["outputs"][0])
        if node == "create_factory_md":
            # Fake only the external author. Actual graph, input publication,
            # file copy/fill, declaration checks and delivery receipts execute.
            template = output.parent / prompts.FACTORY_TEMPLATE_PATH
            provenance = json.loads((output.parent / prompts.FACTORY_TEMPLATE_SOURCE_PATH).read_text())
            assert str(template) in contract["inputs"]
            assert str(output.parent / prompts.FACTORY_TEMPLATE_SOURCE_PATH) in contract["inputs"]
            assert hashlib.sha256(template.read_bytes()).hexdigest() == PINNED_TEMPLATE_SHA256
            assert provenance["source_sha256"] == provenance["template_sha256"] == PINNED_TEMPLATE_SHA256
            assert template.read_text() in prompt
            self.template_input = template.read_bytes()
            shutil.copyfile(template, output)
            filled = output.read_text().replace("TODO: Describe the project goal here.", "Author the reusable requested package.")
            filled = filled.replace("- src/**/*.py\n- tests/**/*.py", "- .factory/packages/portable_test/**")
            filled = filled.replace("python eval/score.py", "python -m refactory_dsh.create_fidelity_ops validate portable_test --project .")
            filled = filled.replace("\n0.8\n", "\nAll requested native artifact and CLI acceptance checks must pass; intentionally unscored, no optimizer score.\n")
            filled = filled.replace("\nmain\n", "\nNot applicable: local artifact-only creation; no branch, commit or publication.\n")
            filled = filled.replace("## Smoke Test\n", "## Smoke Test\n\n```bash\npython -m refactory_dsh.registry list --project . --json\n```\n\n")
            output.write_text(filled)
        elif node == "graph_explorer":
            output.write_text("# Graph context\n\nThe project main.py module imports json and defines main. The module is a direct CLI entry point. This conclusion is grounded in graph.json and the inspected source.\n")
        else:
            pytest.fail("unexpected bootstrap model node " + node)
        return "Delivered the requested artifact."


def test_bootstrap_crosses_real_branch_study_and_delivery_boundaries(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.py").write_text("import json\ndef main():\n    return json.dumps({'ready': True})\n")
    run = tmp_path / "run"
    original = CreateFidelityMode().graph()
    identifiers = {"init_user_intent", "gate_has_factory", "discover", "gate_factory_md_exists", "create_factory_md", "factory_init", "graph_update", "study", "graph_explorer", "concat_study"}
    graph = Graph.from_nodes([n for n in original.nodes.values() if n.id in identifiers], [e for e in original.edges if e.source in identifiers and e.target in identifiers], original.start)
    model = BootstrapIO()
    ctx = RunContext(run, knobs={"name": "portable_test", "brief": "Read a runtime JSON file and produce a reusable output", "project_path": str(source)}, model=model)
    executor = Executor(graph, ctx)
    result = executor.run()
    assert result.exit_ok, result.error
    assert result.facts["create_fidelity"] is True
    assert model.calls == ["create_factory_md", "graph_explorer"]
    assert hashlib.sha256(model.template_input).hexdigest() == PINNED_TEMPLATE_SHA256
    assert [(step.node, step.verdict) for step in executor.steps if step.kind == "gate"] == [("gate_has_factory", "halt"), ("gate_factory_md_exists", "halt")]
    knowledge = json.loads((run / "graph.json").read_text())
    assert any(n["id"] == "main:main" for n in knowledge["nodes"])
    assert any(e["source"] == "main" and e["target"] == "json" for e in knowledge["edges"])
    assert "direct CLI entry point" in (run / ".factory/strategy/study-combined.md").read_text()
    assert (run / ".refactory/agent-delivery").is_dir()
    assert (run / prompts.FACTORY_TEMPLATE_PATH).read_bytes() == model.template_input
    config = json.loads((run / ".factory/config.json").read_text())
    assert config["bootstrap_template_input"]["source_sha256"] == PINNED_TEMPLATE_SHA256
    assert config["source_sha256"] == hashlib.sha256((run / "factory.md").read_bytes()).hexdigest()
    assert {"Goal", "Scope", "Guards", "Command", "Threshold", "Smoke Test"} <= set(config["sections"])
    assert "intentionally unscored" in config["sections"]["Threshold"]
    assert "Not applicable" in config["sections"]["Target Branch"]
    assert "python eval/score.py" not in (run / "factory.md").read_text()
    receipts = [json.loads(f.read_text()) for f in (run / ".refactory/agent-delivery").glob("*/receipt.json")]
    receipt = next(r for r in receipts if r["node"] == "create_factory_md")
    assert receipt["status"] == "delivered"
    fingerprints = receipt["execution"]["input_fingerprints"]
    assert fingerprints[prompts.FACTORY_TEMPLATE_PATH] == PINNED_TEMPLATE_SHA256
    assert fingerprints[prompts.FACTORY_TEMPLATE_SOURCE_PATH] == hashlib.sha256((run / prompts.FACTORY_TEMPLATE_SOURCE_PATH).read_bytes()).hexdigest()
    assert not (source / "factory.md").exists()
    assert not (source / ".factory").exists()


@pytest.mark.parametrize("changed", ["template", "line-endings", "provenance"])
def test_bootstrap_refuses_changed_canonical_input_before_config_initialization(tmp_path, changed):
    source = tmp_path / "source"
    source.mkdir()
    ctx = RunContext(tmp_path / "run", knobs={"name": "portable_test", "brief": "Produce an unscored reusable native artifact", "project_path": str(source)})
    ctx.facts.update(ops.init_user_intent(ctx))
    ctx.write_text("factory.md", "# Factory\n\n## Goal\nProduce the requested local artifact.\n")
    if changed == "template":
        ctx.write_text(prompts.FACTORY_TEMPLATE_PATH, ctx.read_text(prompts.FACTORY_TEMPLATE_PATH) + "\nChanged source template.\n")
    elif changed == "line-endings":
        path = ctx.path(prompts.FACTORY_TEMPLATE_PATH)
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    else:
        record = ctx.read_json(prompts.FACTORY_TEMPLATE_SOURCE_PATH)
        record["source_commit"] = "untrusted revision"
        ctx.write_json(prompts.FACTORY_TEMPLATE_SOURCE_PATH, record)
    with pytest.raises(ValueError, match="template input or provenance changed"):
        ops.factory_init(ctx)
    assert not ctx.exists(".factory/config.json")


def test_factory_initialization_records_exact_filled_file_bytes(tmp_path):
    ctx = RunContext(tmp_path)
    content = b"# Factory\r\n\r\n## Goal\r\nProduce the requested local artifact.\r\n"
    ctx.path("factory.md").write_bytes(content)
    ops.factory_init(ctx)
    assert ctx.read_json(".factory/config.json")["source_sha256"] == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize("has_research_config", [False, True])
def test_discovery_preserves_actual_conditional_bootstrap_inputs(tmp_path, has_research_config):
    source = tmp_path / "source"
    (source / ".factory/strategy").mkdir(parents=True)
    eval_spec = '{"checks": ["Inspect the actual native artifact"]}\n'
    (source / ".factory/eval_spec.json").write_text(eval_spec)
    strategy = "# Existing strategy\n\n" + ("## Research Configuration\n\nMetric supplied by this project.\n" if has_research_config else "No research configuration.\n")
    (source / ".factory/strategy/current.md").write_text(strategy)
    ctx = RunContext(tmp_path / "run", facts={"project_path": str(source)})
    ops.discover(ctx)
    context = ctx.read_json(".factory/eval_profile.json")["bootstrap_context"]
    assert context["eval_spec"]["applicable"] is True
    assert context["eval_spec"]["content"] == eval_spec
    assert context["eval_spec"]["source_sha256"] == hashlib.sha256(eval_spec.encode()).hexdigest()
    assert context["research_configuration"]["exists"] is True
    assert context["research_configuration"]["applicable"] is has_research_config
    if has_research_config:
        assert context["research_configuration"]["content"] == strategy
        assert context["research_configuration"]["source_sha256"] == hashlib.sha256(strategy.encode()).hexdigest()
    else:
        assert "content" not in context["research_configuration"]


def make_package(project: Path):
    root = project / ".factory/packages/portable_probe"
    root.mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").write_text("from refactory_dsh import nn\nfrom .mode import Probe\nPACKAGE=nn.Package(name='portable_probe',description='Read a runtime document',modes=(Probe,),ops=('refactory_packages.portable_probe.ops',))\n")
    (root / "mode.py").write_text("""from refactory_dsh import nn
class Probe(nn.Mode):
    name='read'
    description='Copy runtime JSON through the native invocation'
    unscored_reason='Produces a document with no optimization metric'
    @classmethod
    def declare(cls):
        return {'document_path': nn.Knob('', 'str', description='Runtime JSON file')}
    def build(self):
        g=nn.Graph()
        g.add(nn.Node('read','op',call='refactory_packages.portable_probe.ops:read',knobs=('document_path',),writes=('result.json',)))
        return g
    def invariants(self):
        return (nn.artifact_produced('result.json',by='read'),)
""")
    (root / "ops.py").write_text("""import json
from pathlib import Path
def read(ctx):
    data=json.loads(Path(ctx.knob('document_path')).read_text())
    if not isinstance(data, dict): raise ValueError('input must be an object')
    ctx.write_json('result.json',data)
""")
    (root / "README.md").write_text("# Portable probe\n\nSupply document_path at runtime; each native invocation writes result.json under its own output directory.\n")
    return root


def test_real_native_admission_export_and_two_distinct_invocations(tmp_path):
    project = tmp_path / "artifact-project"
    root = make_package(project)
    report = ops.admit_native_package(project, "portable_probe", requires_input=("document_path",))
    assert report["ok"], report
    assert report["modes"][0]["id"] == "portable_probe/read"
    assert report["modes"][0]["unscored_reason"]
    assert (root / "SKILL.md").is_file()
    assert "document_path" in (root / "SKILL.md").read_text()
    before = ops.package_digest(root)
    outputs = []
    for i in (1, 2):
        source = tmp_path / f"input-{i}.json"
        source.write_text(json.dumps({"value": i}))
        workdir = tmp_path / f"invocation-{i}"
        result = ops.run_native_mode(project, "portable_probe/read", workdir, {"document_path": str(source)})
        assert result["exit_ok"], result
        assert result["raw_evidence"]["returncode"] == 0
        outputs.append(json.loads((workdir / "result.json").read_text()))
    assert outputs == [{"value": 1}, {"value": 2}]
    assert ops.package_digest(root) == before
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]")
    failed_run = tmp_path / "invalid-invocation"
    result = ops.run_native_mode(project, "portable_probe/read", failed_run, {"document_path": str(invalid)})
    assert not result["exit_ok"]
    assert "input must be an object" in result["error"]
    assert not (failed_run / "result.json").exists()


def test_pytest_cache_does_not_invalidate_admission_but_authored_files_do(tmp_path):
    root = make_package(tmp_path)
    (root / "test_smoke.py").write_text("def test_smoke():\n    assert True\n")
    report = ops.admit_native_package(tmp_path, "portable_probe")
    assert report["ok"], report
    before = ops.package_digest(root)
    result = subprocess.run([sys.executable, "-m", "pytest", "test_smoke.py", "-q"], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (root / ".pytest_cache/v/cache/nodeids").is_file()
    assert ops.package_digest(root) == before == report["package_digest"]
    for file in ("README.md", "ops.py", "test_smoke.py", ".pytest_cache/authored_test.py", "test_cases.json"):
        path = root / file
        original = path.read_bytes() if path.exists() else None
        path.write_bytes((original or b"") + b"\nReal authored change\n")
        assert ops.package_digest(root) != before, file
        if original is None:
            path.unlink()
        else:
            path.write_bytes(original)


def typed_package(project):
    root = make_package(project)
    (root / "mode.py").write_text("""from refactory_dsh import nn
class Probe(nn.Mode):
    name='read'
    description='Runtime parameters preserve their declared types'
    unscored_reason='Produces an invocation artifact'
    @classmethod
    def declare(cls):
        return {'count':nn.Knob(1,'int',lo=0,hi=8,frozen=True),
                'ratio':nn.Knob(1.0,'float'), 'flag':nn.Knob(False,'bool'),
                'choice':nn.Knob(1,'enum',enum=(1,2)),
                'label':nn.Knob('','str')}
    def build(self):
        g=nn.Graph()
        g.add(nn.Node('read','op',call='refactory_packages.portable_probe.ops:read',
                      knobs=('count','ratio','flag','choice','label'),writes=('result.json',)))
        return g
""")
    (root / "ops.py").write_text("def read(ctx):\n    ctx.write_json('result.json', {name:ctx.knob(name) for name in ('count','ratio','flag','choice','label')})\n")
    return root


def test_native_cli_reuses_declared_kind_parsing_and_accepts_frozen_runtime_inputs(tmp_path):
    typed_package(tmp_path)
    run = tmp_path / "typed-invocation"
    command = [sys.executable, "-m", "refactory_dsh.create_fidelity_ops", "run", "portable_probe/read", "--project", str(tmp_path), "--workdir", str(run)]
    for value in ("count=3", "ratio=1.25", "flag=true", "choice=2", "label=001"):
        command.extend(("--knob", value))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads((run / "result.json").read_text()) == {"count": 3, "ratio": 1.25, "flag": True, "choice": 2, "label": "001"}


@pytest.mark.parametrize("entry", ["count=9", "count=3.5", "ratio=bad", "flag=maybe", "choice=7", "unknown=3"])
def test_native_runtime_input_parser_rejects_invalid_names_types_bounds_and_enums(tmp_path, entry):
    typed_package(tmp_path)
    run = tmp_path / "invalid-invocation"
    name, value = entry.split("=", 1)
    report = ops.run_native_mode(tmp_path, "portable_probe/read", run, {name: value})
    assert not report["exit_ok"]
    assert name in report["error"]
    assert not (run / "result.json").exists()


def test_malformed_approval_metadata_becomes_failed_precheck_report(tmp_path):
    make_package(tmp_path)
    assert ops.admit_native_package(tmp_path, "portable_probe")["ok"]
    corrupted = tmp_path / ".refactory/create-fidelity-gates/corrupt/verdict.json"
    corrupted.parent.mkdir(parents=True)
    corrupted.write_text("{bad")
    ctx = RunContext(tmp_path, facts={"name": "portable_probe", "artifact_project": str(tmp_path),
                     **{stage + "_verdict": "proceed" for stage in ("gate_build", "gate_qa", "gate_overwatch", "gate_doc_freshness")}})
    assert ops.gate_precheck(ctx) == "halt"
    report = json.loads((tmp_path / ".factory/reviews/precheck.json").read_text())
    assert report["outcome"] == "halt" and report["ok"] is False
    assert ctx.facts["create_fidelity_precheck_ok"] is False
    assert "human approval evidence unreadable" in ctx.facts["create_fidelity_precheck_error"]


def test_native_validator_refuses_unfeedable_input_broken_calls_and_stale_precheck(tmp_path):
    root = make_package(tmp_path)
    required = ops.validate_native_package(tmp_path, "portable_probe", requires_input=("missing_input",))
    assert not required["ok"]
    assert any("missing runtime input" in p for p in required["problems"])
    admitted = ops.admit_native_package(tmp_path, "portable_probe")
    assert admitted["ok"]
    (root / "ops.py").write_text((root / "ops.py").read_text() + "\n# Package modified after admission\n")
    ctx = RunContext(tmp_path, facts={"name": "portable_probe", "artifact_project": str(tmp_path), **{g + "_verdict": "proceed" for g in ("gate_build", "gate_qa", "gate_overwatch", "gate_doc_freshness")}})
    assert ops.gate_precheck(ctx) == "halt"
    assert ctx.facts["create_fidelity_precheck_ok"] is False
    assert "stale" in ctx.facts["create_fidelity_precheck_error"]
    (root / "mode.py").write_text((root / "mode.py").read_text().replace("ops:read", "ops:missing"))
    broken = ops.validate_native_package(tmp_path, "portable_probe")
    assert not broken["ok"]
    assert any("missing" in p for p in broken["problems"])


@pytest.mark.parametrize("marker", ["[FAIL] ", "Status: NOT_VERIFIED — ", "Severity: HIGH — "])
def test_synthesis_keeps_full_reports_and_promotes_independent_duplicate_issues(tmp_path, marker):
    ctx = RunContext(tmp_path)
    reports = []
    issue = marker + "The native CLI is missing the required runtime document_path option and violates the original acceptance contract."
    for i, task_id in enumerate(("custom-one", "custom-two", "workflow-validate", "cli-integration")):
        body = f"# {task_id}\n\nActual command: validation check {i}\nActual output: specific original evidence {i}.\n" + (issue if i < 2 else "Native graph and discovery checks passed with command evidence.")
        output = f".factory/reviews/testers/{task_id}.md"
        ctx.write_text(output, body)
        reports.append({"task_id": task_id, "status": "completed", "body": body, "output": output, "output_digest": hashlib.sha256(body.encode()).hexdigest()})
    ctx.write_json(".factory/qa_director-findings.json", {"status": "completed", "reports": reports})
    ctx.write_text(".factory/qa_director-findings.md", "\n\n".join(report["body"] for report in reports))
    for review in ("health-check.md", "code-review.md"):
        ctx.write_text(".factory/reviews/" + review, "# Review\n\nActual evidence: inspected runtime input surface and command output, identifying the required missing option.\n")
    ops.synthesize_qa(ctx)
    output = ctx.read_text(".factory/reviews/qa-synthesized.md")
    assert "HIGH (custom-one, custom-two)" in output
    assert all(report["body"] in output for report in reports)
    ctx.write_text(reports[0]["output"], "modified stale report")
    with pytest.raises(ValueError, match="stale"):
        ops.synthesize_qa(ctx)


@pytest.mark.parametrize("statement", [
    "All invalid-input probes correctly failed with useful stderr and preserved existing output bytes.",
    "No blocking issues were found; every required acceptance criterion passed with concrete command evidence.",
    "Status: PASS — malformed inputs failed exactly as required and valid outputs stayed unchanged.",
    "```text\nStatus: FAIL — this is captured expected negative-test output, not an adverse QA finding.\n```",
])
def test_synthesis_does_not_promote_successful_or_ambiguous_repeated_evidence(tmp_path, statement):
    ctx = RunContext(tmp_path)
    reports = []
    for task_id in ("custom-one", "custom-two", "workflow-validate", "cli-integration"):
        body = f"# {task_id}\n\n{statement}\n"
        path = f".factory/reviews/testers/{task_id}.md"
        ctx.write_text(path, body)
        reports.append({"task_id": task_id, "status": "completed", "body": body, "output": path, "output_digest": hashlib.sha256(body.encode()).hexdigest()})
    ctx.write_json(".factory/qa_director-findings.json", {"status": "completed", "reports": reports})
    ctx.write_text(".factory/qa_director-findings.md", "\n\n".join(report["body"] for report in reports))
    for review in ("health-check.md", "code-review.md"):
        ctx.write_text(".factory/reviews/" + review, "# Review\n\nEvery original user acceptance requirement passed with complete actual command/output evidence.\n")
    ops.synthesize_qa(ctx)
    output = ctx.read_text(".factory/reviews/qa-synthesized.md")
    confidence = output.split("## Cross-tester finding confidence", 1)[1].split("## Health checker", 1)[0]
    assert "- HIGH" not in confidence
    assert all(report["body"] in output for report in reports)
    if not statement.startswith("```"):
        assert "REPEATED EVIDENCE" in confidence
