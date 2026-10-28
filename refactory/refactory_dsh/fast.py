"""Small authoring graphs with sequential coding and focused review.

These are ordinary modes. Copy a state to customize research, QA or repair;
the runtime does not silently inject the full authoring graph.
"""

from __future__ import annotations

from dataclasses import replace

from refactory_dsh import delivery, repo_change
from refactory_dsh.create_ops import MARKER_RULE, SEVERITY_RULE
from refactory_dsh.create import CreateMode
from refactory_dsh.design import DesignMode
from refactory_dsh.edit import EditMode
from refactory_dsh.nn import Graph, Knob, Node

OPS = "refactory_dsh.fast_ops"

STRATEGY_PROMPT = (
    "Current stage: write one concise implementation plan for the coding agent.\n"
    "Brief: {{brief}}\n"
    "Use the supplied context to identify the files to change, the requested behavior "
    "and constraints to preserve, and the checks that establish completion. "
    "If a source detail is missing, read only the relevant named project/reference file. "
    "Do not inspect framework internals, worker receipts or prior acceptance history. "
    "Do not delegate, call other agents, research alternatives, implement files or run "
    "checks. Write implementation-plan.txt in at most 150 words, then return a short "
    "completion message. One coder and one focused QA agent follow sequentially; "
    "the runtime owns checks and any bounded repair.\n"
)


def _strategy_node(graph: Graph, *, artifact: bool) -> None:
    context = (
        "It is built in {{source}}; read only what the change needs there.\n"
        "Required runtime check, run from {{check_root}}: {{check}}\n"
        if artifact else
        "Plan the minimum reusable graph, declared knobs, score and required artifact "
        "invariants; avoid alternative architectures.\n"
        "Package name: {{name}}\nClass of work: {{class_of_work}}\n"
        "Inputs: {{requires_input}}\nAPI: {{api}}\nReference source: {{reference}}\n"
    )
    graph.add(Node(
        "strategy", "agent", role="strategist", resource_role="work",
        prompt_template=STRATEGY_PROMPT + context,
        reads=("brief.json", "source-manifest.json" if artifact else "survey.json"),
        writes=("implementation-plan.txt",), produces=("implementation_plan",),
        notes="one concise strategy, followed by one coder and one QA agent; no parallel workers",
    ))

QA_PROMPT = (
    "Perform one focused QA review of the requested change. Brief: {{brief}}\n"
    "Artifact/package path: {{output_path}}\n"
    "Emitted or changed files to review: {{review_files}}\n"
    "Required check command and its runtime result: {{focused_check}}\n"
    "Read those files and relevant existing checks; inspect other project source only "
    "when needed to understand a dependency or finding. Use the supplied runtime result "
    "as evidence for the check already executed. Run any additional checks the brief "
    "requires that this result does not cover. Check the requested behavior "
    "and obvious regressions; do not delegate or launch research, other reviewers, an optimizer or "
    "a full model evaluation. Honor checks explicitly requested in the brief, and state "
    "any that could not run. Do not modify the implementation during QA. Do not inspect "
    "framework internals, delivery receipts or prior acceptance histories to discover "
    "this stage's output contract; it is specified below.\n"
    "Write focused-qa.json with exactly this shape: "
    '{"status":"pass" or "fail", "checks":[{"name":"...", "command":"... or source inspection", '
    '"status":"pass" or "fail" or "not_run", "evidence":"actual output or specific source finding"}], '
    '"findings":["unresolved defects"], "not_run":["checks outside this focused scope"]}. '
    "At least one substantive check is required. A required check not run is a failure. "
    "Structural admission, runtime validation and this review are different evidence. "
    "After writing focused-qa.json, return a short completion message. The runtime "
    "reads that file, applies the repair bound and advances the graph; do not run "
    "the gate, generate extra reports or repeat completed checks for handoff."
)

#: What design_fast's QA is told about the change it reviews: in a worktree, the branch's
#: diff, and the command that runs a check in the worktree's own environment.
ARTIFACT_QA = ("\nThe change: {{change}}\n"
               "test-guardedness.json in the run directory, when there is one, says which new "
               "tests fail without the change.\n")


def _qa_nodes(graph: Graph, *, artifact: bool, design: bool = False) -> None:
    graph.add(Node(
        "focused_check", "op", call=f"{OPS}:focused_check", role="validation" if artifact else "ceo",
        resource_role="final_validation",
        knobs=("check", "check_timeout_s", "check_attempt_cap"),
        reads=("emit.txt",) + (() if artifact else ("authoring-evidence.json", "smoke.json")),
        writes=("focused-check.json", "focused-source.json", "authoring-evidence.json", "check-record.json")
               + (("check.txt", *repo_change.CHECK_WRITES) if artifact else ()),
        produces=("focused_check", "output_path", "review_files", "runtime_evidence",
                  "base_unavailable",
                  "check_failures_history",
                  "check_blocker_history",
                  "convergence_extras_used",
                  "convergence_granted")
                 + (repo_change.CHECK_PRODUCES if artifact else ()),
        notes="run the explicitly requested bounded check; no implicit full evaluation",
    ))
    from refactory_dsh.authoring_evidence import REVIEW_CONTEXT, AUDIT_INSTRUCTIONS

    # Build focused_qa prompt: add {{review_instructions}} and repo_questions schema
    # only for design modes (design=True). create_fast/edit_fast never produce the fact.
    qa_prompt = QA_PROMPT
    if design:
        # Insert repo_questions into JSON schema description
        qa_prompt = qa_prompt.replace(
            '"not_run":["checks outside this focused scope"]}',
            '"not_run":["checks outside this focused scope"], '
            '"repo_questions":[{"function":"<exactly as listed under Changed functions, '
            'e.g. pkg/mod.py::Class.method>", "duplicates":"...", '
            '"callers":"...", "shared_state":"..."}]}',
        )
    focused_qa_prompt = (
        qa_prompt + "\nEach entry in `findings` is a BLOCKING finding. " + SEVERITY_RULE + " " + MARKER_RULE
        + " A check of something non-core that fails ends its name with `[non-core]`.\n"
        + (ARTIFACT_QA if artifact else "")
        + ("{{review_instructions}}\nChanged functions:\n{{changed_functions}}\n\n"
           if design else "")
        + REVIEW_CONTEXT + AUDIT_INSTRUCTIONS
    )

    # Not test-guardedness.json: only a worktree run writes it, and a declared input must
    # exist before the worker is dispatched, so a worktree-off run would stop here (dm-152).
    # ARTIFACT_QA points to it when it is there.
    focused_qa_reads = (
        ("brief.json", "focused-check.json", "authoring-evidence.json")
        + (("changed-functions.json",) if design else ())
    )

    graph.add(Node(
        "focused_qa", "agent", role="code_reviewer",
        prompt_template=focused_qa_prompt,
        resource_role="final_validation",
        reads=focused_qa_reads,
        writes=("focused-qa.json", "intent-audit.json"),
        produces=("focused_qa",), notes="one light focused QA stage; no reviewer fan-out",
    ))

    gate_qa_reads = (
        ("focused-qa.json", "focused-check.json", "focused-source.json",
         "authoring-evidence.json", "intent-audit.json")
        + (("changed-functions.json", "reviews/verification.md", "emit.txt", "verifier-response.md")
           if design else ())
    )
    gate_qa_produces = ("repair_feedback", "repairs_used", "check_repairs_used",
                        "review_repairs_used", "qa_passed")
    gate_qa_writes = (
        ("fast-report.json", "repair.txt",
         "site-revert-results.json", "correction-audit.json",
         "reviews/finding-sources.json", "reviews/follow-ups.json", "reviews/follow-ups.md")
        + (("reviews/verification-unresolved.md",) if design else ())
    )

    graph.add(Node(
        "gate_qa", "gate", evaluator=f"{OPS}:gate_qa",
        resource_role="final_validation",
        knobs=("repair_attempts", "check_attempt_cap"), reads=gate_qa_reads,
        writes=gate_qa_writes,
        produces=gate_qa_produces + ("check_failures_history", "check_blocker_history",
                                     "convergence_extras_used", "convergence_granted"),
        max_iterations=13 if design else 12,
        notes="repair only within the shared bound; unresolved findings are failure",
    ))

    if design:
        from refactory_dsh.verification import VERIFIER_PROMPT, VERIFIER_TIMEOUT_S
        graph.add(Node(
            "verifier", "agent",
            prompt_template=VERIFIER_PROMPT,
            timeout_s=VERIFIER_TIMEOUT_S,
            reads=("brief.json", "test-guardedness.json", "changes.diff"),
            writes=("reviews/verification.md",),
            produces=("verification",),
            role="code_reviewer",
            resource_role="final_validation",
            notes="design-blind verification: apply the evidence log's methods "
                  "to the brief, diff and repo",
        ))
        graph.add(Node("fork_qa", "fork",
                       targets=("focused_qa", "verifier"),
                       optional_targets=("verifier",),
                       notes="focused QA and verifier in parallel"))
        graph.add(Node("join_qa", "join",
                       sources=("focused_qa",),
                       notes="wait for focused QA (verifier is optional)"))
        graph.add(Node(
            "gate_check", "gate",
            evaluator=f"{OPS}:gate_check",
            resource_role="final_validation",
            reads=("focused-check.json", "check.txt", "check-record.json"),
            writes=("fast-report.json", "repair.txt"),
            produces=("repair_feedback", "repairs_used", "check_repairs_used",
                      "convergence_extras_used", "convergence_granted"),
            knobs=("repair_attempts", "check_attempt_cap"),
            max_iterations=12,
            notes="route check failures and guardedness blocks; verifier handled in gate_qa",
        ))
        graph.connect("focused_check", "gate_check")
        graph.connect("gate_check", "fork_qa", condition="proceed")
        graph.connect("gate_check", "emit", condition="reloop")
        graph.connect("fork_qa", "join_qa")
        graph.connect("verifier", "join_qa")
        graph.connect("join_qa", "gate_qa")
        graph.connect("gate_qa", "emit", condition="reloop")
    else:
        graph.connect("focused_check", "focused_qa")
        graph.connect("focused_qa", "gate_qa")
        graph.connect("gate_qa", "emit", condition="reloop")


class CreateFastMode(CreateMode):
    name = "create_fast"
    description = "create a small package with one sequential strategist, coder and QA agent"
    plan_first = True

    @classmethod
    def declare(cls) -> dict[str, Knob]:
        knobs = super().declare()
        for name in ("research_attempts", "design_attempts", "attempts", "review_attempts"):
            knobs.pop(name, None)
        knobs.update({
            "repair_attempts": Knob(1, "int", lo=0, hi=3, description="total repairs across admission, runtime validation and focused QA"),
            "check": Knob("", "str", description="explicit focused check command in the emitted package; empty means not requested"),
            "check_timeout_s": Knob(60, "int", lo=1, hi=600, description="seconds allowed for the explicit focused check"),
            "check_attempt_cap": Knob(8, "int", lo=0, description="hard cap on total check attempts including convergence grants; 0 or a value at or below repair_attempts disables the extension"),
            "refusal_attempts": Knob(3, "int", lo=1, hi=10, frozen=True, description="consecutive refused emissions before the run stops"),
        })
        return knobs

    def build(self) -> Graph:
        full = super().build()
        graph = Graph()
        graph.add(replace(full.nodes["prepare"], call=f"{OPS}:prepare_package",
                          resource_role="work",
                          writes=(*full.nodes["prepare"].writes, "repair.txt", "validation.txt"),
                          produces=(*full.nodes["prepare"].produces, "repair_feedback", "repairs_used")))
        graph.add(replace(full.nodes["survey"], call=f"{OPS}:survey_package", resource_role="work",
                          notes="record only the explicitly requested reference; fast mode does not research other packages"))
        if self.plan_first:
            _strategy_node(graph, artifact=False)
        graph.add(Node(
            "emit", "agent", role="builder", resource_role="work",
            prompt_template=(
                "Current stage: emit the package into emit.txt.\n"
                + self.intent + "\nPackage name: {{name}}\nBrief: {{brief}}\n"
                + ("Implementation plan: {{implementation_plan}}\n" if self.plan_first else "")
                +
                "Class of work: {{class_of_work}}\nInputs: {{requires_input}}\n"
                "API: {{api}}\nSelected reference: {{survey}}\nReference source: {{reference}}\n"
                "Last structural validation: {{validation}}\nRepair findings: {{repair_feedback}}\n"
                "Implement the smallest complete reusable package that satisfies the brief. "
                "Use the supplied API and reference source directly. Inspect reference files "
                "only to resolve a concrete missing dependency or detail; do not search for "
                "other packages, read framework internals, delivery receipts or prior "
                "acceptance histories. Preserve Task bindings, feedable inputs and required "
                "artifact invariants. Do not delegate or launch research or review subagents. A focused QA stage "
                "and canonical runtime validation follow. Emit the whole package as file "
                "blocks introduced by ===FILE: <relative path>===, starting with __init__.py "
                "containing PACKAGE = nn.Package(...). Paths are relative to the package root, "
                "never prefixed with the package name. Write those blocks to emit.txt. "
                "Write nothing but emit.txt: do not apply or test your change anywhere — "
                "the target, the run's worktree or a copy — because materialise and the "
                "check do that after you answer and send you the result. Then return a "
                "short completion message. Materialization, registration, structural "
                "admission, the requested check and canonical smoke belong to later runtime "
                "nodes; do not invoke them or locate their implementation during emission."
            ), reads=("brief.json", "survey.json", "repair.txt", "validation.txt")
                     + (("implementation-plan.txt",) if self.plan_first else ()),
            writes=(repo_change.EMIT_FILE,), produces=("emit",), notes="implement directly from the brief and current source",
        ))
        graph.add(replace(full.nodes["materialise"], resource_role="work"))
        graph.add(replace(full.nodes["validate"], evaluator=f"{OPS}:validate_package",
                          resource_role="work",
                          writes=("validation.txt", "repair.txt", "fast-report.json"),
                          knobs=("repair_attempts",), max_iterations=4,
                          produces=("validation", "repair_feedback", "repairs_used",
                                    "check_repairs_used",
                                    "check_failures_history",
                                    "check_blocker_history",
                                    "convergence_extras_used",
                                    "convergence_granted")))
        _qa_nodes(graph, artifact=False, design=False)
        graph.add(replace(full.nodes["register"], resource_role="work"))
        graph.add(replace(full.nodes["smoke"], evaluator="refactory_dsh.create_ops:gate_smoke",
                          knobs=("allow_regression", "sample_knobs", "smoke_timeout_s", "repair_attempts", "check_attempt_cap"),
                          writes=(*full.nodes["smoke"].writes, "repair.txt", "fast-report.json"),
                          produces=(*full.nodes["smoke"].produces, "repair_feedback", "repairs_used",
                                    "check_repairs_used", "review_repairs_used",
                                    "check_failures_history",
                                    "check_blocker_history",
                                    "convergence_extras_used",
                                    "convergence_granted"),
                          resource_role="final_validation", max_iterations=4))
        graph.add(replace(full.nodes["report"], call=f"{OPS}:report_package",
                          resource_role="final_validation",
                          writes=("create.json", "fast-report.json")))
        if self.plan_first:
            graph.connect("survey", "strategy")
            graph.connect("strategy", "emit")
        else:
            graph.connect("survey", "emit")
        for source, target in (("prepare", "survey"),
                               ("emit", "materialise"), ("materialise", "validate"),
                               ("register", "smoke")):
            graph.connect(source, target)
        graph.connect("validate", "emit", condition="reloop")
        graph.connect("validate", "register", condition="proceed")
        graph.connect("smoke", "focused_check", condition="proceed")
        graph.connect("smoke", "emit", condition="reloop")
        graph.connect("gate_qa", "report", condition="proceed")
        return graph


class EditFastMode(CreateFastMode):
    name = "edit_fast"
    description = "make a small package revision directly with one coder, focused QA and regression validation"
    intent = EditMode.intent
    plan_first = False


class DesignFastMode(DesignMode):
    name = "design_fast"
    description = "build a small artifact with one sequential strategist, coder and QA agent"

    @classmethod
    def declare(cls) -> dict[str, Knob]:
        knobs = super().declare()
        knobs.pop("check_attempts")
        knobs.pop("review_attempts")
        knobs.pop("design_review_attempts")
        knobs.pop("judge_rereads")  # design's judges; design_fast has a focused QA instead
        knobs.pop("self_check_fixups")
        knobs.pop("self_check_timeout_s")
        knobs.pop("refine_design")
        # design_fast's builder has no timeout retry: its verifier also has none,
        # its runs are shorter and cheaper than design's, and leaving it alone keeps
        # the candidate id stable.
        knobs.pop("builder_timeout_s")
        # It works where design works (`repo_change`), but delivers nothing: a pull request
        # is opened only behind an approval gate that names it (#350), and fast mode has
        # none. Its work stays on the run's branch.
        for name in delivery.knobs():
            knobs.pop(name)
        knobs["repair_attempts"] = Knob(1, "int", lo=0, hi=3, description=(
            "total repairs, for an emission materialise could not apply, a failed check or focused QA"))
        knobs["check_attempt_cap"] = Knob(8, "int", lo=0, description=(
            "hard cap on total check attempts including convergence grants; "
            "0 or a value at or below repair_attempts disables the extension"))
        knobs["refusal_attempts"] = Knob(3, "int", lo=1, hi=10, frozen=True, description=(
            "consecutive refused emissions before the run stops"))
        knobs["source_project"] = Knob("", "str", description=(
            "an existing directory copied into the artifact directory before the change is applied, "
            "instead of a worktree; it takes no target, base or setup"))
        return knobs

    def build(self) -> Graph:
        graph = Graph()
        graph.add(Node(
            "prepare", "op", call=f"{OPS}:prepare_artifact", resource_role="work",
            knobs=("name", "brief", "check", "source_project", *repo_change.knob_names()),
            writes=("brief.json", "source-manifest.json", "repair.txt", "research-review.txt", "check.txt",
                    "reviews/design-review-feedback.txt",
                    "check-record.json", "feedback-initial.json"),
            produces=("name", "brief", "check", "validation", "research_review", "research_scope",
                      "research_scope_reason", "qa_gate",
                      "overwatch_review", "design_review_feedback", "design_review_findings",
                      "deliverable_kind", "review_instructions", "verification_methods",
                      "source_project", "source_files", "repair_feedback", "repairs_used",
                      *repo_change.produces()),
            notes="read the artifact brief, and say where the run works",
        ))
        _strategy_node(graph, artifact=True)
        from refactory_dsh.design import BUILDER_WORKSPACE, WORKSPACE_BRIEFING
        from refactory_dsh.create_ops import VERIFIER_RESPONSE_FILE
        graph.add(Node(
            "emit", "agent", role="builder", resource_role="work",
            prompt_template=(
                "Current stage: make the change in your workspace.\n"
                "Build artifact {{name}}. Brief: {{brief}}\n"
                "Implementation plan: {{implementation_plan}}\n"
                "Required check: {{check}}\n"
                "Repair feedback: {{repair_feedback}}\n"
                "Implement the smallest complete change. Read the relevant project files to make "
                "the change. Do not read framework internals, worker receipts, graph configuration "
                "or prior acceptance histories. Do not delegate or launch research, reviewer agents "
                "or an optimizer.\n"
                + WORKSPACE_BRIEFING
                + "The runtime runs the requested check and one focused QA review next. Preserve all "
                "requested behavior. Do not invoke downstream graph operations yourself. When the "
                "repair feedback names verifier findings, answer each in your final message: the "
                "finding, then fixed, or rebutted: <reason> <file:line>. The final message is saved "
                f"as {VERIFIER_RESPONSE_FILE}.\n"
            ), reads=("brief.json", "source-manifest.json", "implementation-plan.txt", "repair.txt", "check-record.json"),
            writes=(repo_change.EMIT_FILE, VERIFIER_RESPONSE_FILE), produces=("emit",),
            feedback_keys=("repair_feedback",),
            workspace=dict(BUILDER_WORKSPACE),
            notes="make the change in a clone of the last attempt; its diff is the emission",
        ))
        graph.add(repo_change.materialise_node(
            call=f"{OPS}:materialise_artifact", resource_role="work",
            reads=("emit.txt", "source-manifest.json"),
            writes=("changes.diff", "changed-functions.json"),
            produces=("artifact", "files", "commit", "change", "emission_refused",
                      "changed_functions", "consecutive_refusals"),
            knobs=("refusal_attempts",),
            notes="commit the emitted files on the run's branch, or write the artifact directory: "
                  "a copy of source_project, when there is one",
        ))
        graph.add(Node(
            "gate_emit", "gate", evaluator=f"{OPS}:gate_emit", resource_role="work",
            knobs=("repair_attempts",), writes=("fast-report.json", "repair.txt"),
            produces=("repair_feedback", "repairs_used"), max_iterations=4,
            notes="send an emission materialise could not apply back to emit, within the repair bound",
        ))
        _qa_nodes(graph, artifact=True, design=True)
        graph.add(Node(
            "report", "op", call=f"{OPS}:report_artifact", resource_role="final_validation",
            reads=("focused-qa.json", "focused-check.json"), writes=("design.json", "fast-report.json"),
            produces=("designed",), notes="report only observed checks and the actual repair limit",
        ))
        graph.connect("prepare", "strategy")
        graph.connect("strategy", "emit")
        graph.connect("emit", "materialise")
        graph.connect("materialise", "gate_emit")
        graph.connect("gate_emit", "focused_check", condition="proceed")
        graph.connect("gate_emit", "emit", condition="reloop")
        graph.connect("gate_qa", "report", condition="proceed")
        return graph
