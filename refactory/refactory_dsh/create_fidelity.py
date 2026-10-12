"""The source Create graph with reviewed refactory-dsh native adapters.

This opt-in candidate has its own graph identity. Public Create, Design, Fast
and saved graph states are unaffected. Source topology is deliberately explicit.
"""

from __future__ import annotations

from refactory_dsh.create_fidelity_prompts import (
    FACTORY_TEMPLATE_PATH, FACTORY_TEMPLATE_SOURCE_PATH,
    load_prompt, node_prompt, NATIVE_INSTRUCTIONS,
)
from refactory_dsh.nn import Graph, Knob, Mode, Node, Port, artifact_produced

OPS = "refactory_dsh.create_fidelity_ops"
DIRECTORS = "refactory_dsh.create_fidelity_directors"
S = ".factory/strategy/"
R = ".factory/reviews/"


class CreateFidelityMode(Mode):
    name = "create_fidelity"
    description = "source-faithful Create with native portable package authoring"
    unscored_reason = "creation is judged by native admission and verified user acceptance evidence"

    @classmethod
    def declare(cls):
        return {
            "name": Knob("", "str", description="name of the reusable native package"),
            "brief": Knob("", "str", description="complete original authoring request"),
            "variant": Knob("portable", "str", description="portable is supported; unsupported variants fail before dispatch"),
            "output_folder": Knob("", "str", description="package output folder; defaults to this run's native package workspace"),
            "project_path": Knob("", "str", description="read-only source project to study; defaults to the invocation project"),
            "requires_input": Knob("", "str", description="comma-separated runtime input parameter names required by the brief"),
            "child_attempts": Knob(2, "int", lo=1, hi=5, description="quality review attempts for each director child"),
            "director_parallelism": Knob(7, "int", lo=1, hi=7, description="maximum concurrently dispatched DSH director children"),
        }

    def build(self):
        g = Graph()
        g.add(Node("init_user_intent", "op", call=f"{OPS}:init_user_intent",
                   knobs=("name", "brief", "variant", "output_folder", "project_path", "requires_input"),
                   writes=(S + "user-intent.md", ".factory/create-request.json", R + "repair-feedback.md", "factory.md", ".factory/config.json",
                           FACTORY_TEMPLATE_PATH, FACTORY_TEMPLATE_SOURCE_PATH),
                   produces=("create_fidelity", "name", "brief", "variant", "output_folder",
                             "project_path", "artifact_project", "requires_input", "native_api")))
        g.add(Node("gate_has_factory", "gate", evaluator=f"{OPS}:gate_has_factory"))
        g.add(Node("discover", "op", call=f"{OPS}:discover", writes=(".factory/eval_profile.json",)))
        g.add(Node("gate_factory_md_exists", "gate", evaluator=f"{OPS}:gate_factory_md_exists"))
        g.add(Node("create_factory_md", "agent", role="ceo",
                   prompt_template=load_prompt("ceo", "create_factory_md"),
                   reads=(".factory/eval_profile.json", S + "user-intent.md",
                          FACTORY_TEMPLATE_PATH, FACTORY_TEMPLATE_SOURCE_PATH), writes=("factory.md",)))
        g.add(Node("factory_init", "op", call=f"{OPS}:factory_init",
                   reads=("factory.md", FACTORY_TEMPLATE_PATH, FACTORY_TEMPLATE_SOURCE_PATH), writes=(".factory/config.json",)))
        g.add(Node("graph_update", "op", call=f"{OPS}:graph_update", writes=("graph.json",)))
        g.add(Node("study", "op", call=f"{OPS}:study", reads=("graph.json",), writes=(S + "observations.md",)))
        g.add(Node("graph_explorer", "agent", role="researcher",
                   prompt_template=load_prompt("researcher", "graph_explorer"),
                   reads=(S + "observations.md", "graph.json"), writes=(S + "graph-context.md",)))
        g.add(Node("concat_study", "op", call=f"{OPS}:concat_study",
                   reads=(S + "graph-context.md", S + "observations.md"), writes=(S + "study-combined.md",)))
        for stage, plan, inputs, reports in (
            ("research_director", S + "research-plan.json", (S + "study-combined.md", S + "user-intent.md"), S + "research/"),
            ("strategy_director", S + "strategy-plan.json", (S + "research-plan.json", S + "study-combined.md", S + "user-intent.md", ".factory/research_director-findings.json", ".factory/research_director-findings.md"), S + "perspectives/"),
            ("qa_director", R + "qa-plan.json", (R + "builder-latest.md", S + "current.md", S + "user-intent.md", R + "repair-feedback.md"), R + "testers/"),
        ):
            g.add(Node(stage, "op", call=f"{DIRECTORS}:{stage}", role="ceo", timeout_s=3600,
                       knobs=("child_attempts", "director_parallelism"), reads=inputs,
                       writes=(plan, reports, ".factory/directors/", f".factory/{stage}-findings.json",
                               f".factory/{stage}-findings.md", f".factory/{stage}-quality.md")))
        g.add(Node("synthesize_strategy", "agent", role="strategist",
                   prompt_template=load_prompt("strategist", "synthesize_strategy"),
                   reads=(S + "strategy-plan.json", S + "user-intent.md", ".factory/strategy_director-findings.json", ".factory/strategy_director-findings.md"),
                   writes=(S + "current.md",)))
        g.add(Node("gate_strategy", "gate", gate_kind="user",
                   reads=(S + "current.md", S + "user-intent.md"), max_iterations=4,
                   prompt_template=node_prompt("gate_strategy") + "\n\n" + NATIVE_INSTRUCTIONS,
                   notes="Review the actual current strategy and intent; approval must bind its digest."))
        g.add(Node("archivist_plan", "agent", role="archivist", blocking=False,
                   prompt_template=load_prompt("archivist", "archivist_plan"),
                   reads=(S + "current.md", S + "user-intent.md"), writes=(".factory/archive/create-plan.md",)))
        g.add(Node("builder", "agent", role="builder", timeout_s=1800,
                   prompt_template=load_prompt("builder", "builder"),
                   reads=(S + "current.md", S + "user-intent.md", ".factory/create-request.json", R + "repair-feedback.md"),
                   writes=(R + "builder-latest.md", ".factory/packages/", R + "native-admission.json")))
        for stage, inputs in (
            ("gate_build", (R + "builder-latest.md", S + "user-intent.md")),
            ("gate_qa", (R + "qa-synthesized.md", S + "user-intent.md")),
            ("gate_overwatch", (R + "overwatch-latest.md", S + "user-intent.md")),
            ("gate_doc_freshness", (R + "qa-synthesized.md",)),
        ):
            g.add(Node(stage, "gate", gate_kind="agent", role="ceo", reads=inputs + (R + "repair-feedback.md",),
                       writes=(R + "repair-feedback.md",),
                       prompt_template=load_prompt("ceo", stage), max_iterations=3 if stage == "gate_qa" else 4,
                       notes="Preserve source CEO gate ownership and strict current evidence verdicts."))
        for stage, output in (("health_checker", "health-check.md"), ("code_reviewer", "code-review.md")):
            g.add(Node(stage, "agent", role=stage, prompt_template=load_prompt(stage, stage),
                       reads=(R + "builder-latest.md", S + "current.md", S + "user-intent.md", R + "repair-feedback.md"), writes=(R + output,)))
        g.add(Node("fork_qa", "fork", targets=("health_checker", "code_reviewer", "qa_director")))
        g.add(Node("join_qa", "join", sources=("health_checker", "code_reviewer", "qa_director"),
                   reads=(R + "code-review.md", R + "health-check.md", R + "qa-plan.json")))
        g.add(Node("synthesize_qa", "op", call=f"{OPS}:synthesize_qa",
                   reads=(R + "code-review.md", R + "health-check.md", R + "qa-plan.json", ".factory/qa_director-findings.json", ".factory/qa_director-findings.md", R + "repair-feedback.md"),
                   writes=(R + "qa-synthesized.md",)))
        g.add(Node("overwatch", "agent", role="ceo", timeout_s=1800,
                   prompt_template=load_prompt("ceo", "overwatch"),
                   reads=(R + "builder-latest.md", R + "code-review.md", R + "health-check.md", R + "qa-synthesized.md", S + "current.md", S + "user-intent.md", R + "repair-feedback.md"),
                   writes=(R + "overwatch-latest.md",)))
        g.add(Node("gate_precheck", "gate", evaluator=f"{OPS}:gate_precheck",
                   reads=(R + "qa-synthesized.md", R + "overwatch-latest.md", S + "current.md"),
                   writes=(R + "precheck.json",), produces=("create_fidelity_precheck_ok", "create_fidelity_precheck_error")))
        g.add(Node("archivist_build", "agent", role="archivist", blocking=False,
                   prompt_template=load_prompt("archivist", "archivist_build") + "\nRead .factory/reviews/precheck.json; document the actual final outcome and unresolved failures.",
                   reads=(R + "qa-synthesized.md", R + "precheck.json", R + "overwatch-latest.md"), writes=(".factory/archive/create-build.md",)))
        edges = (
            ("gate_strategy", "archivist_plan", "proceed"), ("archivist_plan", "builder", None),
            ("builder", "gate_build", None), ("gate_build", "fork_qa", "proceed"), ("gate_build", "builder", "reloop"),
            ("fork_qa", "join_qa", None), ("gate_qa", "builder", "reloop"),
            ("gate_doc_freshness", "gate_precheck", "proceed"), ("gate_doc_freshness", "builder", "reloop"),
            ("gate_precheck", "archivist_build", "proceed"), ("gate_precheck", "archivist_build", "halt"),
            ("graph_update", "study", None), ("study", "graph_explorer", None), ("graph_explorer", "concat_study", None),
            ("gate_has_factory", "graph_update", "proceed"), ("gate_has_factory", "discover", "halt"),
            ("discover", "gate_factory_md_exists", None), ("gate_factory_md_exists", "factory_init", "proceed"),
            ("gate_factory_md_exists", "create_factory_md", "halt"), ("create_factory_md", "factory_init", None),
            ("factory_init", "graph_update", None), ("init_user_intent", "gate_has_factory", None),
            ("concat_study", "research_director", None), ("research_director", "strategy_director", None),
            ("strategy_director", "synthesize_strategy", None), ("synthesize_strategy", "gate_strategy", None),
            ("gate_strategy", "strategy_director", "reloop"), ("join_qa", "synthesize_qa", None),
            ("synthesize_qa", "gate_qa", None), ("gate_qa", "overwatch", "proceed"), ("overwatch", "gate_overwatch", None),
            ("gate_overwatch", "gate_doc_freshness", "proceed"), ("gate_overwatch", "builder", "reloop"),
        )
        for source, target, condition in edges:
            g.connect(source, target, condition)
        return g

    def ports(self):
        return (Port("archive", ".factory/archive/create-build.md", "text/markdown"),)

    def invariants(self):
        return (artifact_produced(".factory/strategy/current.md", by="synthesize_strategy"),
                artifact_produced(".factory/reviews/qa-synthesized.md", by="synthesize_qa"),
                artifact_produced(".factory/reviews/overwatch-latest.md", by="overwatch"))
