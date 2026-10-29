"""create-mode: a mode whose output is a whole package.

Creating a mode is the same shape as optimizing one, one level up — design,
emit, validate, and regenerate with the reasons when validation refuses. The
identical validate/reject/regenerate loop is what makes the outer loop safe, so
it is the same loop here, run by the same executor over the same kind of graph.

    prepare -> research -> accepted strategy/design -> emit -> materialise
    -> validate -> register -> canonical smoke -> health/code/adversarial QA
    -> Overwatch intent audit -> report

Admission, smoke and review failures return to emit within one attempt bound.
Missing validation dependencies and uncertain execution stop with diagnostics.

The gate is deterministic. A model's package is imported, its graph is built,
every call it names is resolved, and its score is checked to be real before the
package is registered at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from refactory_dsh import registry
from refactory_dsh.paths import project_root, project_for_nodes
from refactory_dsh.create_ops import (
    CREATION_DIRECTIVE, MARKER_RULE, OVERWATCH_VERDICT_RULE, RESEARCH_SCHEMA, SEVERITY_RULE, STRATEGY_SCHEMA,
)
from refactory_dsh.model import DEFAULT_TIMEOUT_S
from refactory_dsh.sandbox import DEFAULT_MODEL_SMOKE_TIMEOUT_S
from refactory_dsh.nn import Graph, Knob, Mode, Node, RunResult

OPS = "refactory_dsh.create_ops"


class CreateMode(Mode):
    """Emit a package of modes from a brief.

    Unscored on purpose: its run is judged by whether the package it produced
    imports, validates and smoke-scores, which the graph itself checks.

    ``intent`` is what the run is *for*, and it is a class attribute because a
    revision is this same graph with a different brief: survey, choose, review,
    design, emit, validate. Subclassing to change the intent is cheaper and truer
    than a second copy of the graph.
    """

    name = "create"
    description = "design, emit and validate a whole package from a brief"
    unscored_reason = "it is judged by whether the package it produced imports, validates and smoke-scores"
    intent = "Make a new package of modes."

    @classmethod
    def declare(cls) -> dict[str, Knob]:
        return {
            "attempts": Knob(
                3, "int", lo=1, hi=8,
                description="how many times the emit step may be retried with the "
                            "validator's reasons",
            ),
            "review_attempts": Knob(
                3, "int", lo=1, hi=5,
                description="how many times the review gates may reloop before "
                            "stopping; defaults to the same as check_attempts",
            ),
            "refusal_attempts": Knob(
                3, "int", lo=1, hi=10, frozen=True,
                description="consecutive refused emissions before the run stops",
            ),
            "research_attempts": Knob(
                3, "int", lo=1, hi=8,
                description="how many times the research step may be retried when its "
                            "plan is not a plan or its findings were never written",
            ),
            "design_attempts": Knob(
                2, "int", lo=1, hi=5,
                description="how many times the strategy step may be retried when the "
                            "intended graph does not hold together",
            ),
            # create-v2's two creation directives: a Task class the produced mode
            # must satisfy, and whether the package iterates data per item.
            "task_ref": Knob(
                "", "str",
                description="a Task class the produced mode must satisfy, as "
                            "`module:Class` or a registered name (empty for none)",
            ),
            "single_shot": Knob(
                False, "bool",
                description="build the harness for one run of one thing, with no Task "
                            "binding; the explicit alternative to naming a Task, and "
                            "one of the two is required",
            ),
            "smoke_timeout_s": Knob(
                DEFAULT_MODEL_SMOKE_TIMEOUT_S, "int", lo=1, hi=86400,
                description="wall-clock seconds allowed for each full mode validation "
                            "pass, including its model calls; timeout fails creation",
            ),
            "check_attempt_cap": Knob(
                8, "int", lo=0,
                description="hard cap on total check attempts including convergence grants; "
                            "0 or a value at or below the regular attempts disables the extension",
            ),
            "sample_knobs": Knob(
                "", "str",
                description="knob values a validation run should use, as a JSON array "
                            "of \"NAME=VALUE\" strings; without them the smoke runs the "
                            "declared defaults, which for an input-taking mode is the "
                            "built-in fixture",
            ),
            # What the brief says the mode operates on. `validate` cannot read the
            # brief, so this is how "the mode must take a Python package" becomes
            # something checked instead of something assumed.
            "requires_input": Knob(
                "", "str",
                description="the inputs the produced mode must accept, comma-separated "
                            "(e.g. `package` or `source_tree`); each must be received "
                            "as a knob named after it or read from the run directory",
            ),
            "data_node": Knob(
                False, "bool",
                description="build the package as a DataNode that runs its subgraph "
                            "once per data item; naming a Task implies this, so the "
                            "knob is only for a caller that wants to be explicit",
            ),
            # What to build, and what to call it. The walker passes these in the run's
            # task; a run on DSH has no task to pass, so they travel as knobs and the
            # prepare op reads whichever it was given. One *thing* either way: the
            # brief is what the run is for, and naming it is how the package is found.
            "name": Knob(
                "", "str",
                description="the package to create (empty takes the name from the brief)",
            ),
            "brief": Knob(
                "", "str", required=True,
                description="what the package should do, in the asker's words",
            ),
            # `--mode edit`'s subject: the mode being revised. The walker passes it in
            # the run's task; a run on DSH carries it here, and without it an "edit"
            # is a creation that never compares itself to anything.
            "revision_of": Knob(
                "", "str",
                description="for an edit: the <package>/<mode> being revised",
            ),
            # The escape hatch from the smoke step's regression refusal. A person may
            # intend to trade score for something the score does not measure, and that
            # has to be sayable before the run rather than discovered as a failure.
            "allow_regression": Knob(
                False, "bool",
                description="for an edit: skip original-mode execution and score comparison; "
                            "still validate the revised mode (default false)",
            ),
        }

    def build(self) -> Graph:
        graph = Graph()
        graph.add(Node(
            "prepare", "op", call=f"{OPS}:prepare_create",
            # What to build (`name`, `brief`) and the creation directives
            # (`task_ref`, `single_shot`, `data_node`) are read here and published as
            # facts. The walker passes the first two in the run's task; on DSH they
            # are knobs. Naming no Task and not asking for `single_shot` is refused
            # here, before the first model call.
            knobs=("name", "brief", "task_ref", "single_shot", "data_node",
                   "revision_of", "requires_input", "smoke_timeout_s"),
            writes=("brief.json", "research-review.txt", "strategy-review.txt", "validation.txt",
                    "staging/source-snapshot.json"),
            produces=("name", "brief", "extends", "revision_of", "intent", "validation",
                      "design_review", "api", "reference", "task_ref", "class_of_work",
                      "data_node", "requires_input", "research_review", "qa_gate",
                      "overwatch_review", "package_root", "source_hash",
                      "source_existed", "dest_hash", "dest_existed"),
            notes="read the brief, set up per-run staging, and publish to the prompts",
        ))
        graph.add(Node(
            "survey", "op", call=f"{OPS}:survey",
            writes=("survey.json",),
            produces=("survey", "existing"),
            notes="enumerate what already exists, before anything is invented",
        ))

        # ── create-v2: the research director ───────────────────────────────
        # v2 replaced v1's fixed three-researcher fork with a director that chooses
        # the directions itself. Ours spawns one subagent per direction, which is
        # this runtime's way of fanning out, and the plan it leaves behind is what
        # the strategy director reads — the fan-out is not a side effect.
        graph.add(Node(
            "research_director", "agent",
            prompt_template=(
                "{{intent}}\n\n"
                "Package name: {{name}}\nBrief: {{brief}}\n\n"
                "## What already exists\n{{survey}}\n\n"
                "You are the research director. Decide what has to be learned before "
                "this package can be designed: between three and five questions, each "
                "one answerable from this checkout, from the reference package, or from "
                "the API below. Spawn one subagent per question and have each write its "
                "findings to `research/<slug>.md` under the run directory, then write "
                "`research-plan.json` yourself.\n\n"
                "## What was wrong with the last plan\n{{research_review}}\n\n"
                "## The plan's shape\n" + RESEARCH_SCHEMA + "\n"
                "## API\n{{api}}\n\n## Reference package\n{{reference}}\n"
            ),
            reads=("brief.json", "survey.json", "research-review.txt"),
            writes=("research-plan.json",),
            produces=("research",),
            notes="choose the research directions and have a subagent answer each one",
        ))
        # The plan says what was learned; these two make sure it is a plan at all, and
        # that the findings it names exist before the strategy stage reads them. A
        # node's `writes` contract is satisfied by any bytes, so the stage that was
        # supposed to ground the design could narrate its intent and pass.
        graph.add(Node(
            "record_findings", "op", call=f"{OPS}:record_findings",
            reads=("research-plan.json",),
            writes=("research/",),
            produces=("findings_recorded",),
            notes="write each direction's finding where the reviews will look for it",
        ))
        graph.add(Node(
            "gate_research", "gate", evaluator=f"{OPS}:gate_research",
            reads=("research-plan.json",),
            writes=("research-review.txt",),
            produces=("research_review", "research_attempts_used"),
            knobs=("research_attempts",),
            max_iterations=8,
            notes="proceed only when the plan is a plan and its findings exist",
        ))
        graph.add(Node(
            "strategy_director", "agent",
            prompt_template=(
                "{{intent}}\n\n"
                "Package name: {{name}}\nBrief: {{brief}}\n\n"
                "## What already exists\n{{survey}}\n\n"
                "## What the research found\n{{research}}\n\n"
                "You are the strategy director. From the research, choose between two "
                "and five strategy perspectives on how this package could work — "
                "including one that argues for the simplest thing that satisfies the "
                "brief, and one that says what the acceptance criteria must be. Spawn "
                "one subagent per perspective and have each write `strategy/<slug>.md` "
                "under the run directory, with each finding as a heading "
                "`### FND-<slug>-<n> \u2014 <the finding>`, numbered from 1 within the "
                "file. Then write `strategy-plan.json` as an array of "
                "`{\"perspective\", \"slug\", \"intent_items\", \"ids\"}` objects, "
                "where `intent_items` lists the asks from the brief that perspective "
                "covers and `ids` lists the `FND-<slug>-<n>` ids from that "
                "perspective's file. Every ask must be covered by at least one "
                "perspective; name the uncovered ones rather than leaving them out.\n\n"
                "## Reference package\n{{reference}}\n"
            ),
            reads=("brief.json", "survey.json", "research-plan.json"),
            writes=("strategy-plan.json", "strategy/"),
            produces=("strategy_plan",),
            notes="choose the strategy perspectives and their intent coverage",
        ))
        graph.add(Node(
            "synthesize_strategy", "agent",
            prompt_template=(
                "{{intent}}\n\n"
                "Package name: {{name}}\nBrief: {{brief}}\n\n"
                "## What already exists\n{{survey}}\n\n"
                "## What the research found\n{{research}}\n\n"
                "## The perspectives\n{{strategy_plan}}\n\n"
                "## What went wrong last time\n{{design_review}}\n\n"
                "Compile the perspectives into one strategy. Every ask in the brief must "
                "appear in it unless the user explicitly agreed to exclude it; naming "
                "an ask out of scope does not supply that agreement. Preserve user-approved "
                "non-goals and later explicit user steering. An unmet ask remains "
                "unresolved. Then decide "
                "the approach: consider at least two, say why the one you chose wins, "
                "say what it will not do, name the existing mode it is closest to, and "
                "describe the graph you intend to write.\n\n"
                "The research detail files (`research/*.md`) and strategy detail "
                "files (`strategy/*.md`) are available as reads. Each carries finding "
                "ids (`FND-<slug>-<n>`). Include a `findings` field in the strategy "
                "mapping every finding id to the node or knob that handles it, or "
                "`[set-aside: reason]`.\n\n" + STRATEGY_SCHEMA
            ),
            reads=("brief.json", "survey.json", "research-plan.json",
                   "strategy-plan.json", "strategy-review.txt",
                   "research/", "strategy/"),
            writes=("strategy.json",),
            produces=("strategy",),
            notes="compile the perspectives into one intent-faithful strategy",
        ))
        graph.add(Node(
            "review", "gate", evaluator=f"{OPS}:review_design",
            reads=("strategy.json", "research/", "strategy/"),
            writes=("strategy-review.txt",),
            produces=("design_review", "design_attempts_used",
                      "finding_coverage"),
            knobs=("design_attempts",),
            max_iterations=8,
            notes="check the intended graph before any code exists",
        ))
        # create-v2 puts a person here (`gate_strategy`), and it is the one v2 gate
        # this port keeps as-is: the strategy is a decision the person owns, and a
        # run that can approve its own strategy is not reviewed at all. A walker run
        # stops at it; a headless DSH run stops and prints the steer and resume commands.
        graph.add(Node(
            "gate_strategy", "gate", gate_kind="user",
            reads=("strategy.json",),
            max_iterations=4,
            notes=(
                "Approve the strategy before any code is written? The chosen approach "
                "is in strategy.json in the run directory, and the package is written "
                "from it. Answer proceed to build it, or reloop to have the strategy "
                "director choose another approach."
            ),
        ))
        graph.add(Node(
            "design", "agent",
            prompt_template=(
                "{{intent}}\n\n"
                "Package name: {{name}}\nBrief: {{brief}}\n\n"
                "## The class of work this harness is for\n{{class_of_work}}\n\n"
                "Read the API below and write the design for the package: every "
                "operation, every node, every knob, and what score() reads. Be concrete "
                "and complete — the next step writes the code from your design alone. "
                "For each required deliverable, name its meaningful content, producer, "
                "run-time check, and whether score() measures it. Protect required "
                "ungraded producers with Mode.invariants(), using artifact_produced "
                "where appropriate. A placeholder or same-named fallback is not the "
                "requested output. Preserve mathematical reasoning in the delivered "
                "document, not only in transient programs or execution metadata. "
                "Design for the whole class of work, not for one sample: when a Task is "
                "bound, the harness reads its input from that Task's item or accepts it "
                "as a declared input. The `data` node is the harness's input port, so "
                "its items come from the Task it binds or from a `source_path` with a "
                "`source_format` at run time, never from a list baked into the graph.\n\n"
                "## The approach that was chosen\n{{strategy}}\n\n"
                "## What this mode must take as input\n{{requires_input}}\n"
                "## API\n{{api}}\n\n"
                "## Reference package\n{{reference}}\n"
            ),
            reads=("brief.json", "strategy.json"),
            writes=("design.md",),
            produces=("design",),
            notes="plan the package before any code is written",
        ))
        graph.add(Node(
            "emit", "agent",
            prompt_template=(
                "Write the package `{{name}}` as complete, runnable Python.\n\n"
                "Brief: {{brief}}\n\n"
                "## The class of work this harness is for\n{{class_of_work}}\n\n"
                "A harness is for a class of work, not for one instance of it. When a "
                "Task is bound, the harness reads its input from that Task's item — the "
                "`data` node writes `current_item.json` before each pass — or accepts "
                "it as one of the declared inputs below, and it must not hardcode a "
                "single sample. The `data` node is the input port and stays feedable at "
                "run time: declare `task_ref`, or `source_path` with a `source_format` "
                "of `directory`, `jsonl` or `csv`, and never `inline_items`. Any item "
                "of the class may arrive, with whatever path or fields it names, so "
                "assume nothing about a particular one.\n\n"
                "Keep each required ungraded deliverable's producer in invariants(); "
                "do not let successful numeric grading conceal a missing explanation. "
                "If the required content is absent, fail or retry instead of publishing "
                "a success-shaped fallback. Treat the Task input's documented JSON "
                "values and strings as data; do not reject ordinary quoted prose just "
                "because it resembles Python syntax. Content checks should enforce "
                "the promised structure and substantive content. Identify unresolved "
                "fields in their actual template context; punctuation, braces, or "
                "ordinary words alone are not evidence of a placeholder. Keep guards "
                "small and specific to a stated contract violation.\n\n"
                "Design: Read the file `design.md` in the run directory. It contains the package design with all operations, nodes, knobs, and score definitions.\n\n"
                "## What this mode must take as input\n{{requires_input}}\n\n"
                "## API\n{{api}}\n\n"
                "## Reference package\n{{reference}}\n\n"
                "## What went wrong last time\n{{validation}}\n\n"
                "## What the reviews refused\n{{qa_gate}}\n{{overwatch_review}}\n\n"
                "Write every file into emit.txt as a block introduced by a line of exactly:\n"
                "===FILE: <relative path>===\n"
                "followed by the file's full contents. Every path is relative to the "
                "**package's own root**: the manifest is exactly `__init__.py`, not "
                "`{{name}}/__init__.py`, not `refactory/refactory_packages/{{name}}/"
                "__init__.py`, and not `.factory/packages/{{name}}/__init__.py`. The "
                "package is written where it belongs; a path that repeats its "
                "directory buries the package inside itself. Emit `__init__.py` (with "
                "`PACKAGE = nn.Package(...)`), the ops module, and any module the "
                "ops or the mode need. Use the declared ops module path in every "
                "`call` and `evaluator`. When a Task is bound, the mode must accept "
                "that Task's per-item prompt as its input and produce what its "
                "`verify()` scores, and the graph's start node is a `data` node whose "
                "`subgraph_entry` and `subgraph_exit` delimit the per-item subgraph. "
                "No prose outside the blocks. Do not create the files themselves: the "
                "materialise step writes them from emit.txt. Then return a short "
                "completion message.\n"
            ),
            reads=("brief.json", "design.md", "validation.txt"),
            writes=("emit.txt",),
            produces=("emit",),
            notes="write the package's source files",
        ))
        graph.add(Node(
            "materialise", "op", call=f"{OPS}:materialise",
            reads=("emit.txt",),
            writes=("staging/manifest.json",),
            produces=("package_path", "files", "emission_refused", "consecutive_refusals"),
            knobs=("refusal_attempts",),
            notes="write the staged package to disk",
        ))
        # create-v2's `gate_build`: the built thing is checked before it is reviewed.
        graph.add(Node(
            "validate", "gate", evaluator=f"{OPS}:validate",
            reads=("emit.txt", "brief.json"),
            writes=("validation.txt",),
            produces=("validation", "attempts_used"),
            knobs=("attempts",),
            max_iterations=12,
            notes="import the package and check it is a usable mode",
        ))

        # ── create-v2: the three reviews, in parallel ──────────────────────
        # v2 fans out to a health check, a code review and an adversarial QA
        # director, and waits for all three before synthesising. The fork reaches
        # its branches through `targets` and the join waits on `sources`, so one
        # review finishing early cannot let the run past the barrier.
        graph.add(Node("fork_qa", "fork",
                       targets=("health_checker", "code_reviewer", "qa_director"),
                       notes="start the three reviews"))
        graph.add(Node(
            "health_checker", "agent",
            prompt_template=(
                "Brief: {{brief}}\n\n## The design\nRead the file `design.md` in the run directory. It contains the package design to check the implementation against.\n\n"
                "You are the health checker. Run the created package the way a user "
                "would: import it, list its modes, check its declared tools, and run "
                "bounded interface and artifact checks. Canonical smoke has already run. "
                "Use its authoritative receipt and outputs. A targeted extra model check must "
                "name a concrete unmet criterion and its bounded scope. Record every "
                "command and its exact output in `reviews/health-check.md`. If anything "
                "fails or is missing, write BLOCKING: <path>: <description> where <path> is the file the builder must change to fix the problem. A defect in the code is reported against the code file, even when the requirement it misses comes from design.md or the brief. A problem in a run document (design.md, strategy.json, or another file an earlier step wrote) is reported with that document's path. "
                "If `staging/package-diff.json` exists, use it to focus checks on what changed from the source package. "
                "A check you did not run is not a check that passed — say which ones you "
                "skipped."
                + " " + SEVERITY_RULE + " " + MARKER_RULE
            ),
            reads=("brief.json", "design.md", "staging/package-diff.json"),
            writes=("reviews/health-check.md",),
            produces=("health_check",),
            notes="run the package and record what happened, command by command",
        ))
        graph.add(Node(
            "code_reviewer", "agent",
            prompt_template=(
                "## The design\nRead the file `design.md` in the run directory. It contains the package design.\n\n## The emitted source\nRead the file `emit.txt` in the run directory. It contains the emitted package source files. Review these against the design.\n\n"
                "You are the code reviewer. Read the emitted source against the design "
                "and the API: are every op, node, knob and fact declared and consumed, "
                "does `score()` read a fact the graph produces, and would the package "
                "import? Check that required outputs not measured by score() have "
                "explicit invariants and real content checks. Attempt the typed removal "
                "or bypass of their producers with normal validation; it must refuse. "
                "Check that admission and content guards also accept legitimate data "
                "and notation supported by the contract, not only reject bad cases. "
                "If `staging/package-diff.json` exists, review the diff between the source and the staged package. "
                "Record findings in `reviews/code-review.md`. A defect that "
                "would stop the package working is written as BLOCKING: <path>: <description> where <path> is the file the builder must change to fix the problem. A defect in the code is reported against the code file, even when the requirement it misses comes from design.md or the brief. A problem in a run document (design.md, strategy.json, or another file an earlier step wrote) is reported with that document's path. Style is not blocking."
                + " " + SEVERITY_RULE + " " + MARKER_RULE
            ),
            reads=("design.md", "emit.txt", "staging/package-diff.json"),
            writes=("reviews/code-review.md",),
            produces=("code_review",),
            notes="review the emitted source against the design",
        ))
        graph.add(Node(
            "qa_director", "agent",
            prompt_template=(
                "Brief: {{brief}}\n\n## The strategy (the acceptance criteria are in "
                "it)\nRead the file `strategy.json` in the run directory. It contains the strategy with acceptance criteria to test the package against.\n\n"
                "You are the QA director. Design adversarial checks against this "
                "package — the cases a friendly reviewer would not try — and run them. "
                "Full model/Task validation has already run in canonical smoke. Inspect its "
                "receipt and delivered outputs; do not repeat the full feed by default. "
                "Missing semantic checks, required output "
                "contracts, or other prerequisites remain blocking. "
                "Read the actual delivered files against the brief: mathematical "
                "explanations need the formula or state/transition, why it counts the "
                "requested objects, and verification. Method names, hashes, paths and "
                "successful execution alone do not establish that. Through the public "
                "package interface, check acceptance of ordinary quoted prose, Unicode "
                "and mathematical notation, including nested LaTeX braces, where the "
                "contract supports them. Check malformed inputs and unresolved template "
                "fields too, recording each acceptance or rejection and its reason. "
                "Inspect what score() "
                "does not grade, and mark any untested requirement as unverified. "
                "If `staging/package-diff.json` exists, use the diff to identify what changed from the source. "
                "Write `reviews/qa.md` with each check and its exact result. Every "
                "acceptance criterion in the strategy must have a check. Anything that "
                "breaks an acceptance criterion is written as BLOCKING: <path>: <description> where <path> is the file the builder must change to fix the problem. A defect in the code is reported against the code file, even when the requirement it misses comes from design.md or the brief. A problem in a run document (design.md, strategy.json, or another file an earlier step wrote) is reported with that document's path."
                + " " + SEVERITY_RULE + " " + MARKER_RULE
            ),
            reads=("brief.json", "strategy.json", "staging/package-diff.json"),
            writes=("reviews/qa.md",),
            produces=("qa_review",),
            notes="attack the package against its acceptance criteria",
        ))
        graph.add(Node("join_qa", "join",
                       sources=("health_checker", "code_reviewer", "qa_director"),
                       notes="wait for all three reviews"))
        graph.add(Node(
            "synthesize_qa", "op", call=f"{OPS}:synthesize_qa",
            reads=("reviews/health-check.md", "reviews/code-review.md", "reviews/qa.md"),
            writes=("reviews/qa-synthesized.md",),
            produces=("qa", "qa_blocking", "qa_missing"),
            notes="merge the three reviews; an absent review is not a passed one",
        ))
        graph.add(Node(
            "gate_qa", "gate", evaluator=f"{OPS}:gate_qa",
            reads=("reviews/qa-synthesized.md", "reviews/health-check.md",
                   "reviews/code-review.md", "reviews/qa.md"),
            writes=("reviews/qa-gate.txt", "reviews/finding-sources.json",
                    "reviews/follow-ups.json", "reviews/follow-ups.md"),
            produces=("qa_gate",),
            max_iterations=4,
            notes="proceed when every review ran and none found a core defect; "
                  "non-core findings become follow-ups",
        ))

        # ── create-v2: the overwatch pass ──────────────────────────────────
        graph.add(Node(
            "overwatch", "agent",
            prompt_template=(
                "Brief: {{brief}}\n\n## The strategy\nRead the file `strategy.json` in the run directory. It contains the strategy and acceptance criteria.\n\n"
                "## The merged reviews\nRead the file `reviews/qa-synthesized.md` in the run directory. It contains the merged reviews from all three reviewers.\n\n"
                "## What the last audit was refused for\n{{overwatch_review}}\n\n"
                "You are the overwatch. This is not another review: audit the reviews "
                "themselves. Take each ask in the brief and each acceptance criterion, "
                "and say what evidence proves it — a command and its output — or that "
                "nothing does. Reproduce bounded spot checks on what the reviews claim "
                "passed: import the package, read back its knobs, and inspect actual "
                "artifacts. Audit actual final validation receipts and the accepted design "
                "against the brief and latest intent ledger. Required behavior without "
                "adequate evidence is failed or unverified with a precise reason. "
                "Any blocking finding is written as BLOCKING: <path>: <description> where <path> is the file the builder must change to fix the problem. A defect in the code is reported against the code file, even when the requirement it misses comes from design.md or the brief. A problem in a run document (design.md, strategy.json, or another file an earlier step wrote) is reported with that document's path. "
                + SEVERITY_RULE + " " + MARKER_RULE + " " + OVERWATCH_VERDICT_RULE + " "
                "Write `reviews/overwatch.md` ending with exactly `VERDICT: PASS` or "
                "`VERDICT: FAIL`. A missing required check cannot pass."
            ),
            reads=("brief.json", "strategy.json", "reviews/qa-synthesized.md", "design.md"),
            writes=("reviews/overwatch.md",),
            produces=("overwatch",),
            notes="audit the evidence behind the reviews, not the work again",
        ))
        graph.add(Node(
            "gate_overwatch", "gate", evaluator=f"{OPS}:gate_overwatch",
            reads=("reviews/overwatch.md",),
            writes=("reviews/overwatch-gate.txt", "reviews/follow-ups.json",
                    "reviews/follow-ups.md"),
            produces=("overwatch_review",),
            max_iterations=4,
            notes="proceed on a stated PASS verdict with no core finding; "
                  "non-core findings become follow-ups",
        ))

        graph.add(Node(
            "register", "op", call=f"{OPS}:register",
            writes=("registered.json", "staging/package-diff.json"),
            produces=("registered", "package_path"),
            notes="confirm the registry lists the new package",
        ))
        graph.add(Node(
            "smoke", "gate", evaluator=f"{OPS}:gate_smoke", role="validation",
            resource_role="final_validation", max_iterations=12,
            knobs=("allow_regression", "sample_knobs", "smoke_timeout_s", "attempts", "check_attempt_cap"),
            writes=("smoke.json", "revision.json", "authoring-evidence.json", "validation.txt"),
            produces=("smoke_score", "smoke_ok", "smoke_model", "smoke_reason",
                      "before_score", "after_score", "regressed", "revision_note",
                      "runtime_evidence", "validation", "attempts_used",
                      "check_attempts_used",
                      "convergence_extras_used",
                      "convergence_granted"),
            notes="run the new mode once, and compare a revision with the mode it "
                  "revises; a mode that cannot score is not finished",
        ))
        graph.add(Node(
            "report", "op", call=f"{OPS}:write_result",
            writes=("create.json",),
            produces=("created",),
            notes="summarise what was created",
        ))

        graph.connect("prepare", "survey")
        graph.connect("survey", "research_director")
        graph.connect("research_director", "record_findings")
        graph.connect("record_findings", "gate_research")
        graph.connect("gate_research", "research_director", condition="reloop")
        graph.connect("gate_research", "strategy_director", condition="proceed")
        graph.connect("strategy_director", "synthesize_strategy")
        graph.connect("synthesize_strategy", "review")
        graph.connect("review", "synthesize_strategy", condition="reloop")
        graph.connect("review", "gate_strategy", condition="proceed")
        graph.connect("gate_strategy", "strategy_director", condition="reloop")
        graph.connect("gate_strategy", "design", condition="proceed")
        graph.connect("design", "emit")
        graph.connect("emit", "materialise")
        graph.connect("materialise", "validate")
        graph.connect("validate", "emit", condition="reloop")
        graph.connect("validate", "register", condition="proceed")
        graph.connect("fork_qa", "join_qa")
        graph.connect("join_qa", "synthesize_qa")
        graph.connect("synthesize_qa", "gate_qa")
        graph.connect("gate_qa", "emit", condition="reloop")
        graph.connect("gate_qa", "overwatch", condition="proceed")
        graph.connect("overwatch", "gate_overwatch")
        graph.connect("gate_overwatch", "emit", condition="reloop")
        graph.connect("gate_overwatch", "overwatch", condition="audit")
        graph.connect("gate_overwatch", "report", condition="proceed")
        graph.connect("register", "smoke")
        graph.connect("smoke", "fork_qa", condition="proceed")
        graph.connect("smoke", "emit", condition="reloop")
        from refactory_dsh.authoring_evidence import wire_reviews
        wire_reviews(graph)
        return graph


@dataclass
class CreateConfig:
    """What to create, and how hard to try."""

    name: str
    brief: str
    extends: str = ""
    attempts: int = 3
    workdir: Path = Path(".factory/create")
    max_steps: int = 200
    seed: int = 0
    #: Wall-clock ceiling for one model completion. A create-v2 node writes and runs
    #: code, so the ceiling exists to catch a hung subprocess rather than to bound the
    #: work; raise it for a slower route or a larger package.
    model_timeout_s: int = DEFAULT_TIMEOUT_S
    #: Wall-clock seconds for each full mode validation pass, separate from one completion.
    smoke_timeout_s: int = DEFAULT_MODEL_SMOKE_TIMEOUT_S
    #: A Task class the produced mode must satisfy (create-v2's `--task`).
    task_ref: str = ""
    #: The inputs the brief says the mode takes, comma-separated. Checked against the
    #: produced mode's knobs and reads, so a mode that hardcodes a fixture instead of
    #: accepting the input is refused rather than shipped.
    requires_input: str = ""
    #: Knob values the validation run uses, as `NAME=VALUE` strings. They are how the
    #: brief's inputs are exercised: without one the smoke runs the defaults.
    samples: list[str] = field(default_factory=list)
    #: Whether the produced graph is a DataNode iterating a subgraph per item. A
    #: `task_ref` implies it; this is only for a caller that wants to be explicit.
    data_node: bool = False
    #: The explicit special case: one run of one thing, with no Task binding. Without
    #: it or a `task_ref`, `create` refuses before its first model call.
    single_shot: bool = False
    #: Verdicts for gates a person answers. `gate_strategy` is one, so a walker
    #: run stops at it; a headless DSH run stops and prints the steer and resume commands.
    verdicts: dict[str, str] = field(default_factory=dict)
    fast: bool = False
    repair_attempts: int = 1
    check: str = ""
    check_timeout_s: int = 60
    resource_limits: dict[str, Any] | None = None
    #: The run's knobs as its flags gave them (`run_flags.knobs`), applied last, as
    #: `refactory ceo` applies its own: the same flags configure the same run.
    knobs: list[str] = field(default_factory=list)


@dataclass
class CreateResult:
    """What the run produced."""

    package: str
    mode: str
    path: Path
    files: list[str] = field(default_factory=list)
    smoke_score: float | None = None
    iterations: int = 0
    duration_s: float = 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "package": self.package,
            "mode": self.mode,
            "path": str(self.path),
            "files": self.files,
            "smoke_score": self.smoke_score,
            "iterations": self.iterations,
            "duration_s": self.duration_s,
        }


def parse_verdicts(pairs: list[str]) -> dict[str, str]:
    """The gate verdicts a headless authoring run was given, as ``gate=value``.

    ``create``, ``edit`` and ``design`` each contain a gate a person answers
    (``gate_strategy``), and a headless run must be *given* that answer rather than
    invent one: a creation that approved its own plan is not a reviewed creation. The
    spelling matches ``console steer --verdict``, so the same answer can be recorded
    on a paused run and stated up front on a new one.
    """
    verdicts: dict[str, str] = {}
    for pair in pairs:
        gate, separator, value = str(pair).partition("=")
        gate, value = gate.strip(), value.strip()
        if not separator or not gate or not value:
            raise ValueError(f"--verdict takes <gate>=<value>, not {pair!r}")
        verdicts[gate] = value
    return verdicts


def create(config: CreateConfig, client: Any = None, *, enable_recovery: bool = False) -> CreateResult:
    """Run create-mode once. Returns what was created, or raises why it was not."""
    # Before the mode is loaded and long before the first model call: a creation that
    # says neither what class of work it is for nor that it is the single-shot special
    # case has not said what it is building.
    if not config.task_ref and not config.single_shot:
        raise CreateError(CREATION_DIRECTIVE)
    if not 1 <= config.smoke_timeout_s <= 86400:
        raise CreateError("smoke_timeout_s must be between 1 and 86400 seconds")
    from refactory_dsh.fast import CreateFastMode

    mode = CreateFastMode() if config.fast else CreateMode()
    state = mode.state_dict()
    if config.fast:
        state["knobs"]["repair_attempts"]["value"] = int(config.repair_attempts)
        state["knobs"]["check"]["value"] = config.check
        state["knobs"]["check_timeout_s"]["value"] = int(config.check_timeout_s)
    else:
        state["knobs"]["attempts"]["value"] = int(config.attempts)
    state["knobs"]["smoke_timeout_s"]["value"] = int(config.smoke_timeout_s)
    state["knobs"]["task_ref"]["value"] = config.task_ref
    state["knobs"]["single_shot"]["value"] = bool(config.single_shot)
    state["knobs"]["data_node"]["value"] = bool(config.data_node)
    state["knobs"]["requires_input"]["value"] = config.requires_input
    state["knobs"]["sample_knobs"]["value"] = json.dumps(list(config.samples))
    from refactory_dsh.interactive import with_knobs

    try:
        state = with_knobs(state, list(config.knobs))
    except ValueError as exc:
        raise CreateError(str(exc)) from exc
    mode.load_state_dict(state)

    workdir = Path(config.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    mode_name = "create_fast" if config.fast else "create"
    from refactory_dsh.run_ownership import ownership
    from refactory_dsh.nn.executor import GateStop, HaltStop

    with ownership(workdir):
        # Run preparation.
        from refactory_dsh.run_preparation import prepare_run, walker_context
        from refactory_dsh import steering
        from refactory_dsh.paths import project_root

        prepare_run(
            state, workdir,
            invocation_name=f"default_pack-{mode_name}",
            mode_id=f"default_pack/{mode_name}",
            intent=config.brief,
            transform=lambda workflow, stage_dir=None: steering.for_run(workflow, stage_dir or workdir),
            runtime="walker",
            command=f"python -m refactory_dsh.create",
            resume_command=f"python -m refactory_dsh.create --resume {workdir}",
            fast=config.fast,
            worker_backend=client.route().get("worker_backend", "") if client and callable(getattr(client, "route", None)) else "",
            project=project_root(),
            knobs=list(config.knobs),
            model_context=walker_context(client),
            omit_steering=getattr(config, 'omit_steering', False),
        )

        # Persist up-front verdicts as run-scoped ledger entries.
        ledger = steering.Ledger(workdir)
        for gate, value in config.verdicts.items():
            ledger.verdict(gate, value, approval=True)

        run = mode.run(
            workdir,
            model=client,
            task={
                "name": config.name,
                "brief": config.brief,
                "extends": config.extends,
                "task_ref": config.task_ref,
                "single_shot": config.single_shot,
                "data_node": config.data_node,
                "requires_input": config.requires_input,
            },
            max_steps=config.max_steps,
            resource_limits=config.resource_limits,
            enable_recovery=enable_recovery,
        )
    if run.gate_stop:
        return RunResult(exit_ok=False, error=run.error, gate_stop=run.gate_stop, workdir=workdir)
    if run.halted:
        return RunResult(exit_ok=False, error=run.error, halted=True, workdir=workdir)
    if not run.exit_ok:
        raise CreateError(f"creating {config.name!r} failed: {run.error}")

    name = str(run.facts.get("name") or config.name)
    package = registry.load(name, project_for_nodes())
    # The report node owns create.json and derives its contract from the emitted
    # package. Preparation facts describe the request, not necessarily the result.
    return CreateResult(
        package=name,
        mode=package.ids()[0],
        path=registry.where(name, project_for_nodes()) or registry.workspace_root(project_for_nodes()) / name,
        files=list(run.facts.get("files") or []),
        smoke_score=run.facts.get("smoke_score"),
        iterations=run.iterations,
        duration_s=run.duration_s,
    )


class CreateError(RuntimeError):
    """create-mode could not produce a working package."""


def result_from_run(run: RunResult) -> dict[str, Any]:
    """The facts of a create-mode run, for a caller that drove the executor itself."""
    return dict(run.facts)


def main(argv: list[str] | None = None) -> int:
    """``python -m refactory_dsh.create --name <n> --brief <text> [--extends <mode>]``.

    The walker's main. Its run flags are `refactory ceo`'s (`run_flags`, dm-134).
    """
    import argparse
    import sys

    from refactory_dsh import run_flags
    from refactory_dsh.run_preparation import ACKNOWLEDGE_HELP, RESUME_HELP

    parser = argparse.ArgumentParser(
        prog="python -m refactory_dsh.create",
        description="Create a package of modes from a brief.",
    )
    parser.add_argument("--brief", required=False, default="", help="what the package should do")
    parser.add_argument("--extends", default="", help="a <package>/<mode> to use as a reference")
    parser.add_argument("--fast", action="store_true", help="use the inspectable create_fast graph with focused QA")
    run_flags.add_arguments(parser)
    # The two creation directives, reachable from the shell. One of them is required:
    # a Task names the class of work, and naming one implies the data node.
    parser.add_argument(
        "--data-node", action="store_true", default=False, dest="data_node",
        help="build the package as a DataNode that runs its subgraph once per item; "
             "naming a Task implies this, so it is only for a caller that wants to be "
             "explicit",
    )
    parser.add_argument(
        "--input", action="append", default=[], metavar="NAME", dest="inputs",
        help="an input the produced mode must accept, named the way a knob or a file "
             "would be (repeatable). The mode must receive it as a knob or read a file "
             "under that name, or validation refuses it",
    )
    parser.add_argument(
        "--sample", action="append", default=[], metavar="NAME=VALUE", dest="samples",
        help="a knob value the validation run should use (repeatable). Give it when the "
             "brief's inputs are real, so the smoke exercises the input path instead of "
             "the built-in fixture",
    )
    parser.add_argument(
        "--task", default=None, metavar="TASK_REF", dest="task_ref",
        help="a Task class (module:Class or a registered name) naming the class of work "
             "the produced harness is for; the harness gets a `data` node bound to that "
             "Task and runs its subgraph once per item",
    )
    parser.add_argument(
        "--single", action="store_true", default=False, dest="single_shot",
        help="build a harness for one run of one thing, with no Task binding — the "
             "explicit alternative to --task; one of the two is required",
    )
    parser.add_argument(
        "--resume", default="", metavar="RUN_DIR", help=RESUME_HELP,
    )
    parser.add_argument("--acknowledge-failure", action="store_true", help=ACKNOWLEDGE_HELP)
    parser.add_argument(
        "--smoke-timeout", type=int, default=DEFAULT_MODEL_SMOKE_TIMEOUT_S, dest="smoke_timeout_s",
        help="seconds for each full mode validation pass (default: 3600); independent of --model-timeout",
    )
    args = parser.parse_args(argv)
    if args.acknowledge_failure and not args.resume:
        parser.error("--acknowledge-failure goes with --resume: it acknowledges the boundary a run stopped at")
    if args.resume:
        problem = run_flags.resume_refusal(parser, args)
        if problem:
            parser.error(problem)
    else:
        if not args.name:
            parser.error("--name is required when not resuming")
        if not args.brief:
            parser.error("--brief is required when not resuming")
        mode_id = "default_pack/create_fast" if args.fast else "default_pack/create"
        problem = run_flags.refusal(args, mode_id, runtime=run_flags.WALKER,
                                    label="create --fast" if args.fast else "create")
        if problem:
            parser.error(problem)

    from refactory_dsh.model import DshModelClient, ModelChoiceError, require_model_choice
    from refactory_dsh.paths import project_root

    try:
        # Every agent node calls a model, so the route is settled before anything runs (dm-258).
        client = require_model_choice(
            DshModelClient(project=None) if args.resume
            else DshModelClient(project=project_root(), **run_flags.model_options(args)),
            interactive=sys.stdin.isatty())
    except ModelChoiceError as exc:
        print(f"refactory: {exc}", file=sys.stderr)
        return 1
    if args.resume:
        workdir = Path(args.resume)
        from refactory_dsh.fast import CreateFastMode
        from refactory_dsh.run_preparation import resume_walked

        try:
            mode_class, run = resume_walked(workdir, "create", (CreateMode, CreateFastMode), model=client,
                                            max_steps=200, acknowledge_failure=args.acknowledge_failure,
                                            resource_limits=run_flags.resource_limits(args))
        except Exception as exc:
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
            return 2
        if run.gate_stop:
            from refactory_dsh.run_preparation import print_gate_stop
            print_gate_stop(run.gate_stop, workdir, graph=mode_class().build())
            return 2
        if run.halted:
            print(json.dumps({"halted": True, "error": run.error}), file=sys.stderr)
            return 2
        name = str(run.facts.get("name") or args.name)
        print(json.dumps({
            "resumed": str(workdir),
            "name": name,
            "exit_ok": run.exit_ok,
            "error": run.error,
            "facts": {key: value for key, value in run.facts.items() if key in
                      ("name", "files", "package_path", "smoke_score", "validation")},
        }, indent=2, default=str))
        return 0 if run.exit_ok else 2

    workdir = Path(args.workdir or f".factory/create/{args.name}")
    try:
        result = create(
            CreateConfig(
                name=args.name,
                brief=args.brief,
                extends=args.extends,
                workdir=workdir,
                data_node=bool(args.data_node),
                task_ref=str(args.task_ref or ""),
                single_shot=bool(args.single_shot),
                requires_input=", ".join(args.inputs),
                samples=list(args.samples),
                model_timeout_s=int(args.model_timeout_s or DEFAULT_TIMEOUT_S),
                smoke_timeout_s=int(args.smoke_timeout_s),
                verdicts=parse_verdicts(args.verdict),
                fast=args.fast,
                resource_limits=run_flags.resource_limits(args),
                # The attempts, the check and every other knob the flags set travel as
                # the knobs `refactory ceo` gives the spine.
                knobs=run_flags.knobs(args, mode_id, brief=args.brief, name=args.name),
            ),
            client=client,
            enable_recovery=True,
        )
    except (CreateError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    from refactory_dsh.nn.mode import RunResult
    if isinstance(result, RunResult):
        if result.gate_stop:
            from refactory_dsh.run_preparation import print_gate_stop
            from refactory_dsh.fast import CreateFastMode
            print_gate_stop(result.gate_stop, workdir, graph=(CreateFastMode if args.fast else CreateMode)().build())
            return 2
        if result.halted:
            print(json.dumps({"halted": True, "error": result.error}), file=sys.stderr)
            return 2
    print(json.dumps(result.describe(), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - the entry point
    raise SystemExit(main())
