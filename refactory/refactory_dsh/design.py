"""design-v2: a brief in, a checked software artifact out.

The third of the three meta-modes. create emits a package of modes; optimize searches
one; design emits software. The shape is the same as create's -- design, emit, check,
regenerate with the reasons -- and only the artifact and its gate differ, which is
what makes it a mode rather than a second program.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from refactory_dsh import delivery, design_contract, judges, pitfalls, repo_change
from refactory_dsh.create_ops import API, MARKER_RULE, RESEARCH_SCHEMA, SEVERITY_RULE
from refactory_dsh.invocations import CHECK_PREP, CHECK_VERDICT, SELF_CHECK_RECORD
from refactory_dsh.model import DEFAULT_TIMEOUT_S
from refactory_dsh.nn import Graph, Knob, Mode, Node, RunResult
from refactory_dsh.design_ops import (CORRECTIONS_PICKED, EARLY_ASKS_JUDGE_MARK,
                                      EARLY_ASKS_VERDICT, UPDATED_TESTS)

OPS = "refactory_dsh.design_ops"

#: The timeout recovery rule (`Node.retry`): a session that times out is run once more,
#: fresh, with twice its `timeout_s`. Remote-factory's CEO did this by judgment; here it
#: is the graph's rule, on both runtimes. Both the verifier and the builder declare it.
#: The verifier uses `outputs_only`; the builder uses its workspace, which provides the
#: same isolation: each attempt gets a fresh workspace via `workspace.dispatched(fresh=True)`.
TIMEOUT_RETRY = {"on": ["timeout"], "times": 1, "timeout_factor": 2}

#: The builder session limit before its retry: the model default (1800 s), and the
#: `builder_timeout_s` knob overrides it for a run.
BUILDER_TIMEOUT_S = 1800

#: Design mode's verifier runs beside the check, in the review fork.
BESIDE_CHECK_VERIFIER_NOTE = (
    "\n\n## The check runs beside this review\n"
    "The check `{{check}}` runs at the same time, on the same commit, and is judged on "
    "its own. Do not run the whole check yourself.\n"
)

#: What design mode's builder checks before it answers (approved 2026-10-26): every extra
#: build round the QA judge caused was a gate correction with no test that exercises it,
#: or a fix whose test still passes with the fix reverted. Design mode only. The builder
#: works in a clone of the attempt (`BUILDER_WORKSPACE`), so it reverts and reruns there.
EMIT_CHECKLIST = (
    "\n## Before you stop\n"
    "- For each correction made at the design gate and each ask in the brief, find the test "
    "that proves it and run it in your workspace.\n"
    "- Then revert that change in the workspace and run the same test again: it must fail. "
    "If it still passes, the test does not prove the change: fix the test, then restore "
    "the change and run the test once more.\n"
    "- A correction counts as met only if a test fails without it. The reviewers check this.\n"
    "- Avoid a test that calls the helper under test instead of driving the real path the "
    "brief names, test inputs nothing uses (such as a verdict list nothing consumes), and "
    "assertions on results the test built by hand.\n"
)

#: Where design mode's builder works (`workspace.py`): a prepared clone of the last
#: attempt, whose diff the runtime writes into emit.txt. design_fast's builder too.
BUILDER_WORKSPACE = {"kind": "clone", "of": "attempt", "collect": "diff"}

#: How design mode's builder is told to work in its workspace, which tests to run there,
#: and when to stop. It runs the tests its change touches and the brief's own example, not
#: the full suite (approved 2026-10-27): in the live trial (dm-149) the builder's full-suite
#: runs took 11.6 of its 27 minutes, and the check runs the full suite after it anyway (in
#: design, the full check runs beside the reviews in ``fork_qa``). ``{{builder_tests}}`` is the
#: ready-to-paste command for those tests and its measured time (``design_ops.builder_tests``).
WORKSPACE_BRIEFING = (
    "## Your workspace\n"
    "Your workspace is a prepared clone of the last attempt, with its dependencies "
    "installed and its setup and generate commands run; its path is in the execution "
    "block above. Make the change there. `git diff HEAD` there shows what you changed "
    "since the last attempt; the run's base commit is {{base}}.\n"
    "Run there, as often as you need, the tests your change touches: the test files you "
    "changed or added, and the tests that import the modules you changed. Run the brief's "
    "own example or CLI path there too. They run there as they run in the check.\n"
    "Your targeted test command: {{builder_tests}}\n"
    "Run this targeted command (and any single test you add). Do not run the whole test "
    "suite: the full check runs after you finish, beside the reviews.\n"
    "If you wrap a command in `timeout`, give it about twice the time stated above; DSH "
    "workspace shell commands are killed after at most 10 minutes.\n"
    "When you are done, stop: the workspace's diff is your answer. The runtime turns it "
    "into the next attempt, which the check and the reviews then judge. Return a short "
    "completion message.\n"
)

#: How design's builder names the tests it updated (dm-270): the last line of its completion
#: message, which `design_ops.note_updated_tests` reads from the emit receipt's final
#: response. Design only: nothing reads it in design_fast, whose final message is its
#: verifier response.
UPDATED_TESTS_COMPLETION = (
    "When you updated an existing test because the brief or a correction changes what it "
    "pins, end the message with one line: `Updated tests: <test id> (follows "
    "<the ask or correction>), ...`.\n"
)

_ACCEPTANCE_BASELINE = (
    "Acceptance baseline: the original brief, any approved design, and explicit user "
    "steering. Preserve user-approved non-goals. A README limitation, proposed "
    "exclusion, coordinator summary or reviewer note cannot waive a requested "
    "behavior without explicit user agreement to that change. Report conflicts "
    "instead of silently narrowing scope. Reviews must mark observed noncompliance "
    "as BLOCKING even when the limitation is documented.\n\n"
)

#: The acceptance baseline as a judge reads it: a judge writes no markers, so
#: noncompliance is a defect to judge rather than a line to mark.
_JUDGE_BASELINE = _ACCEPTANCE_BASELINE.replace(
    "Reviews must mark observed noncompliance as BLOCKING even when the limitation is "
    "documented.",
    "Observed noncompliance is a defect even when the limitation is documented.",
)
assert _JUDGE_BASELINE != _ACCEPTANCE_BASELINE

#: What holds a change back, for both judges: remote-factory design-v2's rule. The
#: reviews report; a judge decides on substance, so a review's wording or markers decide
#: nothing, and its gate applies the same rule to what the judge names (`judges.blocks`).
_JUDGE_RULE = (
    "Send the change back (`redo`, with a must_fix item) only for: an ask in the brief or a "
    "correction the person gave that was not built (`kind` `ask`); the check failing, "
    "including a failed health check, where the artifact does not run as a user runs it "
    "(`kind` `check`); or a critical or high defect in product code (`kind` `defect`: "
    "wrong behaviour, data loss, a crash or a security problem, or a regression, with the "
    "test or path:line that shows it).\n"
    "These are always follow-ups, never must_fix: weak or tautological test assertions and "
    "missing edge-case tests, docs wording, style, dead code, minor refactors, and any low or "
    "medium finding. Put them in `follow_ups`. If you list one in must_fix anyway, label it "
    "with its `kind` (`test`, `docs` or `style`, or a `defect` of `medium` or `low` "
    "severity): the gate moves it to the follow-ups and the pull request lists it. A weak "
    "test is a `test` item even when it covers a correction: the correction is built.\n"
    "Give each must_fix item its `kind` (`ask`, `check`, `defect`, `test`, `docs` or "
    "`style`), its `severity` (`critical`, `high`, `medium` or `low`) and what it `breaks` "
    "(`ask`, `correction`, `health` or `none`). Answer `redo` only when at least one item "
    "holds the change back by this rule.\n\n"
)

_JUDGE_UNAVAILABLE = (
    "If a tool or credential needed to judge an ask is unavailable, answer `redo` with "
    "one must_fix item whose `what` starts with `" + judges.UNAVAILABLE + "` and names "
    "the dependency and the exact action that unblocks it: the run then stops for a "
    "person instead of sending the builder a defect it cannot fix."
)

_MUST_FIX_SHAPE = (
    '{"source": "<the review it came from>", "where": "<path:line>", "what": "<the defect>", '
    '"why": "<the ask, correction or behaviour it breaks>", '
    '"kind": "ask" | "check" | "defect" | "test" | "docs" | "style", '
    '"severity": "critical" | "high" | "medium" | "low", '
    '"breaks": "ask" | "correction" | "health" | "none"}'
)

#: Design mode's QA judge: reads the merged reviews and decides proceed or redo.
QA_JUDGE_PROMPT = (
    "Artifact name: {{name}}\nBrief: {{brief}}\n"
    "Artifact directory (absolute): {{artifact}}\n{{change}}\n"
    "The check: {{check}}\n\n"
    "## Corrections the person gave when approving the design\n{{gate_corrections}}\n\n"
    "## What sent this attempt back\n{{review_feedback}}\n\n"
    "## What to read\n"
    "`reviews/qa-synthesized.md` in the run directory holds, in full, this attempt's "
    "health check, code review and QA director's attack, and the design-blind verifier's "
    "report when it gave one, then the check's result, the diff stat and the design's "
    "Changes section. `design.md` holds the approved design, whole, and `brief.json` the "
    "brief. Read the code at the artifact path wherever a finding needs it.\n\n"
    + _JUDGE_BASELINE
    + "You are the QA judge. The reviewers report; you decide whether this change goes "
    "on to the overwatch or back to the builder. Judge each finding on its substance and "
    "against the code, never by its wording or its markers: a line that quotes a "
    "finding's format, an example or a review rule is not a finding, and neither is a "
    "reviewer's tally of its own findings. The check has already passed on this attempt "
    "and stays binding; do not run it again.\n\n"
    "A test that was updated to pass after failing \u2014 one whose assertions were "
    "removed, weakened, or made trivially true \u2014 is a must-fix unless the update "
    "legitimately follows an ask or correction that retires the pinned behaviour. "
    "The updated-tests section of qa-synthesized.md lists which tests were modified "
    "after failing.\n\n"
    + _JUDGE_RULE
    + "A finding that is wrong (it does not hold against the code) is dismissed, with the "
    "reason. Two reviewers reporting the same defect is one must_fix item.\n\n"
    "Write `" + judges.QA_JUDGE + "` in the run directory holding one JSON object and "
    "nothing else that is JSON:\n"
    '{"verdict": "proceed" | "redo", "must_fix": [' + _MUST_FIX_SHAPE + '], '
    '"follow_ups": ["<a real finding that does not hold the change back>"], '
    '"dismissed": [{"source": "<the review>", "finding": "<what it said>", '
    '"why": "<why it does not hold, or is not a finding>"}]}\n'
    "`redo` names at least one must_fix item that holds the change back; `proceed` names "
    "none. " + _JUDGE_UNAVAILABLE + "\n\n"
    "The evidence map below is the overwatch's starting point once you proceed: when your "
    "verdict is `redo`, the builder goes again and the map is made next round, so an "
    "empty one (`{\"requirements\": []}`) will do.\n"
    "{{qa_judge_reread}}"
)

#: Design mode's early asks judge (b48a): beside the check in ``fork_qa``, it reads the
#: diff against the brief's asks and the approved corrections and says which are built.
#: Presence, not quality: the QA judge's bar is unchanged. Its not-built items are read
#: by ``gate_check`` and included in the combined send-back.
EARLY_ASKS_PROMPT = (
    "Artifact directory (absolute): {{artifact}}\n{{change}}\n\n"
    "Brief: {{brief}}\n\n"
    "## Corrections the person gave when approving the design\n{{gate_corrections}}\n\n"
    "## What to read\n"
    "`changes.diff` in the run directory holds this attempt's whole diff from its base. Read "
    "the code at the artifact path wherever the diff needs it. Do not run the tests: they "
    "are running beside you.\n\n"
    + EARLY_ASKS_JUDGE_MARK + " The builder has just finished an attempt. For each ask in the "
    "brief and each correction above (none when that section is empty), decide whether the "
    "attempt builds it: whether the diff holds code that does what it asks, on the path the "
    "brief names. A function that is defined but never called, a setting nothing reads, or "
    "a fallback the ask removes but the diff keeps does not build an ask. Judge presence, "
    "not quality: whether code that is there is right is the reviewers' question. Take the "
    "brief's asks as a reviewer would, one item for each distinct thing it asks for.\n\n"
    "Write `" + EARLY_ASKS_VERDICT + "` in the run directory holding one JSON object and "
    "nothing else that is JSON:\n"
    '{"items": [{"id": "ask-1", "kind": "ask" | "correction", "text": "<the ask or '
    'correction, in a few words>", "built": true | false, "evidence": "<path:line in the '
    'diff that builds it>", "missing": "<what is not built>"}]}\n'
    "List every ask and every correction. A built item cites the path:line that builds it; "
    "an item that is not built says what is missing.\n"
    "{{early_asks_reread}}"
)

#: What the code reviewer and the QA director read about the tests the builder updated
#: (dm-270): the lines its completion messages gave (`design_ops.note_updated_tests`).
UPDATED_TESTS_NOTE = (
    "## Tests the builder says it updated\n{{updated_tests}}\n"
    "A test weakened to pass after it failed (assertions removed, weakened or made "
    "trivially true) is BLOCKING unless the update follows an ask or correction that "
    "retires the behaviour it pinned.\n\n"
)

#: Design mode's overwatch: verifies acceptance with the correction audit as evidence.
OVERWATCH_PROMPT = (
    "Artifact directory (absolute): {{artifact}}\n"
    "Read and run the delivered files at this exact path; do not infer it "
    "from your working directory. {{change}}\n\n"
    "Brief: {{brief}}\nThe check it must pass: {{check}}\n\n"
    "## Corrections the person gave when approving the design\n{{gate_corrections}}\n\n"
    "## The correction audit\n{{correction_audit}}\n\n"
    "## What to read\n"
    "In the run directory: `design.md` (the approved design), `reviews/qa-synthesized.md` "
    "(the merged reviews, the check's result and the diff stat), `changes.diff`, and, when "
    "they are there, `correction-audit.json` (the audit above, per clause and site), "
    "`intent-audit.json` (the QA judge's map of every requirement to its evidence) and "
    "`qa-judge.json` (the QA judge's verdict and follow-ups).\n\n"
    + _JUDGE_BASELINE
    + "You are the overwatch. Verify acceptance rather than repeating the reviews: take "
    "each ask in the brief and each correction above and say what proves it (a test "
    "that fails without the change, or a command and its output) or that nothing does. "
    "Run your own spot checks on what the reviews and the map claim. Do not promote a "
    "syntax check, file-presence check, self-report or label-string check into semantic "
    "evidence. If required validation is unavailable, keep that requirement unverified "
    "and say what would unblock it.\n\n"
    "The correction audit is evidence, not a verdict. For each correction it took the "
    "sites the map names, checked each is in the diff, and reverted it to see whether the "
    "named tests fail. A site whose code the change deletes is removed. A site the change "
    "leaves as the base has it is unchanged, and the audit counts it as not met for you to "
    "decide: a correction that dropped a planned change is met by it, and a correction "
    "the change never applied is not. A site no named test catches, or that is not where "
    "the map says, is a question for you, not a defect by itself: decide from the code and "
    "the tests whether the correction is met. Decide every correction yourself, with "
    "evidence. A correction you find not met (`\"met\": false`) sends the change back "
    "whatever your verdict.\n\n"
    + _JUDGE_RULE
    + "Write `" + judges.OVERWATCH_VERDICT + "` in the run directory holding one JSON "
    "object and nothing else that is JSON:\n"
    '{"verdict": "pass" | "redo", "must_fix": [' + _MUST_FIX_SHAPE.replace(
        "<the review it came from>", "overwatch") + '], '
    '"follow_ups": ["<a real finding that does not hold the change back>"], '
    '"corrections": [{"id": "<its requirement id in authoring-evidence.json, such as '
    'ledger-5>", "met": true | false, "evidence": "<test id or path:line>"}]}\n'
    "List every correction and constraint the person gave in `corrections`. `redo` names "
    "at least one must_fix item that holds the change back, or a correction that is not "
    "met. " + _JUDGE_UNAVAILABLE + "\n"
    "{{overwatch_reread}}"
)


def prepare_check_iterations(knobs: dict[str, Any]) -> int:
    """prepare_check's iteration limit: one more than the most reloops its knobs allow.

    It reloops for a fix-up, an early stop or a refused emission. Attempts reach at most
    ``check_attempt_cap`` (when it is above ``check_attempts``), the run's fix-ups are at
    most ``self_check_fixups x check_attempts``, and each applied emission comes after at
    most ``refusal_attempts - 1`` refusals, so the reloops are at most
    ``(attempts + fix-ups + 1) x refusals + fix-ups + attempts``: (8 + 6 + 1) x 2 + 6 + 8
    = 44 with the defaults. Past that, b39b's stop has already ended the run.
    """
    attempts = int(knobs.get("check_attempts", 3))
    reach = max(attempts, int(knobs.get("check_attempt_cap", 8)))
    fixups = int(knobs.get("self_check_fixups", 2)) * attempts
    refusals = max(int(knobs.get("refusal_attempts", 3)) - 1, 0)
    return (reach + fixups + 1) * refusals + fixups + reach + 1



def builder_timeout(values: dict[str, Any]) -> int:
    """The builder's session limit, from the knob a person may set."""
    return int(values.get("builder_timeout_s") or BUILDER_TIMEOUT_S)


class DesignMode(Mode):
    """Design, emit and check a software artifact.

    Unscored on purpose, like create-mode: its run is judged by whether the artifact
    passes the check the brief declared.
    """

    name = "design"
    description = "design, emit and check a software artifact from a brief"
    unscored_reason = "it is judged by whether the artifact passes the check its brief declared"

    @classmethod
    def declare(cls) -> dict[str, Knob]:
        return {
            "check_attempts": Knob(
                3, "int", lo=1, hi=5,
                description="how many times the emit step may be retried when the "
                            "artifact does not pass its check",
            ),
            "review_attempts": Knob(
                3, "int", lo=1, hi=5,
                description="how many times the review gates may reloop before "
                            "stopping; defaults to the same as check_attempts",
            ),
            "check_timeout_s": Knob(
                180, "int", lo=10, hi=1800,
                description="how long the brief's check may run",
            ),
            "check_attempt_cap": Knob(
                8, "int", lo=0,
                description="hard cap on total check attempts including convergence grants; "
                            "0 or a value at or below check_attempts disables the extension",
            ),
            "design_review_attempts": Knob(
                1, "int", lo=0, hi=3,
                description="how many times a design whose ## Changes section is missing "
                            "or empty is sent back for a rewrite before it goes to the "
                            "approval gate as it is; every other finding is advisory and "
                            "never sends it back",
            ),
            # Both mains pass these as knobs (`run_flags`, dm-134); a caller of the
            # walker's API may still give them in the run's task.
            "name": Knob("", "str", description="the artifact to design (empty takes the name from the brief)"),
            "brief": Knob("", "str", required=True, description="what to build, in the asker's words"),
            "check": Knob("", "str", description="the command that decides whether the artifact works"),
            # Where the run works (#351), as any mode that changes a repository declares it
            # (`repo_change`), and how its work is delivered once its reviews pass (#350).
            "refusal_attempts": Knob(
                3, "int", lo=1, hi=10, frozen=True,
                description="consecutive refused emissions before the run stops",
            ),
            "judge_rereads": Knob(
                1, "int", lo=0, hi=3, frozen=True,
                description="how many times a round a judge whose verdict could not be read "
                            "is asked again, with no builder round, before its gate goes on "
                            "without it",
            ),
            "self_check_fixups": Knob(
                2, "int", lo=0, hi=5,
                description="how many times the self-check may send an attempt back to emit, "
                            "spending nothing, before the early stop spends it; a run gets "
                            "this many times check_attempts",
            ),
            "self_check_timeout_s": Knob(
                300, "int", lo=10, hi=600,
                description="how long the self-check's tests may run before the reviews; "
                            "never more than a third of check_timeout_s",
            ),
            "builder_timeout_s": Knob(
                1800, "int", lo=60, hi=7200, frozen=True,
                description="how long the builder's model session may run before "
                            "timeout; its retry doubles this",
            ),
            "refine_design": Knob(
                "light", "str", values=("light", "full"), frozen=True,
                description="on a refine: light skips research and strategy; "
                            "full walks the whole design phase",
            ),
            **repo_change.knobs(),
            **delivery.knobs(),
        }

    #: The gates whose iteration limits follow the knobs' values.
    _LIMITS = {"prepare_check": prepare_check_iterations}

    def graph(self) -> Graph:
        """The graph, with prepare_check's limit following the knobs."""
        graph = super().graph()
        for name, limit in self._LIMITS.items():
            if name in graph.nodes:
                graph.nodes[name].max_iterations = limit(self.values())
        builder = graph.nodes.get("emit")
        if builder is not None and builder.retry is not None:
            builder.timeout_s = builder_timeout(self.values())
        return graph

    def state_dict(self) -> dict[str, Any]:
        """The state, with those limits following the knobs' values, so a lowered run (the
        spine) gets the limits the walker does."""
        state = super().state_dict()
        for name, limit in self._LIMITS.items():
            node = state["graph"].get("nodes", {}).get(name)
            if node is not None:
                node["max_iterations"] = limit(self.values())
        node = state["graph"].get("nodes", {}).get("emit")
        if node is not None and node.get("retry") is not None:
            node["timeout_s"] = builder_timeout(self.values())
        return state

    def build(self) -> Graph:
        graph = Graph()
        graph.add(Node(
            "prepare", "op", call=f"{OPS}:prepare_design",
            knobs=("name", "brief", "check", *repo_change.knob_names(), *delivery.knobs()),
            writes=("brief.json", "research-review.txt", "check.txt", "delivery.md",
                    "reviews/design-review-feedback.txt", "pitfalls-review.txt",
                    "check-record.json", "feedback-initial.json"),
            produces=("name", "brief", "check", "validation", "research_review", "research_scope",
                      "research_scope_reason", "qa_gate",
                      "overwatch_review", "design_review_feedback", "design_review_findings",
                      "deliverable_kind", "review_instructions", "verification_methods",
                      *repo_change.produces(), "delivery", "delivery_plan",
                      "has_pitfalls_step", "pitfalls", "pitfalls_review", "builder_tests",
                      "claims_check_feedback", "design_review_status", "refine_context",
                      "gate_strategy_feedback", "gate_revision_instruction"),
            notes="read the brief and the command that decides whether it is done",
        ))

        # ── design-v2: route_refine_design gate ──
        graph.add(Node(
            "route_refine_design", "gate",
            evaluator=f"{OPS}:route_refine_design",
            knobs=("refine_design",),
            writes=("research-plan.json", "pitfalls.json", "pitfalls.md",
                    "strategy-plan.json", "strategy.json",
                    "research-review.txt", "pitfalls-review.txt",
                    "research/", "strategy/"),
            produces=("research", "findings_recorded", "research_review",
                      "research_attempts_used", "research_advisory",
                      "pitfalls", "pitfalls_review", "pitfalls_attempts_used",
                      "pitfalls_advisory", "strategy_plan", "strategy", "refine_context"),
            max_iterations=2,
            notes="on a light refine, skip research and strategy and go straight to design_doc",
        ))

        # ── design-v2's head: research, perspectives, then the design itself ──
        graph.add(Node(
            "research_director", "agent",
            prompt_template=(
                "Artifact name: {{name}}\nBrief: {{brief}}\n"
                "It will be checked by running, from {{check_root}}:\n"
                "    {{check}}\n\n"
                "## Research scope\n{{research_scope}}\n\n"
                "You are the research director. Decide what has to be learned before "
                "this artifact can be designed: between three and five questions for "
                "a full design, or one focused question for a scoped change (two or "
                "three only when it spans unrelated parts of the code). Each must be "
                "answerable by reading {{source}} — including the source code of tools "
                "and libraries the change depends on, not only the files it changes — "
                "or by running something there. Spawn "
                "one subagent per question and have each write its findings to "
                "`research/<slug>.md` under the run directory, then write "
                "`research-plan.json` yourself.\n\n"
                "## Footprint\n"
                "Record a `footprint` object on every direction, with exactly these "
                "three keys, each a list. A direction that "
                "touches no code records an empty footprint: "
                '`{"existing_functions": [], "new_items": [], "shared_engine": []}`.\n\n'
                "- `existing_functions`: each entry is a string `\"path:line name\"` "
                "for an existing function the direction's change modifies: one function "
                "per entry, the line its definition starts on, and its path as the "
                "target's checkout shows it. Test "
                "files, test functions and docs are neither touched functions nor "
                "added items.\n"
                "- `new_items`: each entry is a string naming what is added and the "
                "path it goes in — a module, class, graph node, edge, node field, "
                "record or data shape, knob, entry point, wiring helper, or any new "
                "non-test source file. Give the file:line where it would be added "
                "or wired.\n"
                "- `shared_engine`: a list of `\"path:line\"` citations of the engine "
                "code the change modifies — code that every run of the target passes "
                "through, such as its executor, scheduler, worker dispatch, or "
                "compilation or lowering step. A function is not an engine because "
                "several gates or modes call it. Empty when the change does not touch "
                "one.\n"
                "- Wording in the brief about the change's size (\"small\", "
                "\"a one-line fix\", \"a scoped change\", \"a full change\") is a "
                "claim to check against the code, never a footprint.\n\n"
                "## What was wrong with the last plan\n{{research_review}}\n\n"
                "## The plan's shape\n" + RESEARCH_SCHEMA
            ),
            reads=("brief.json", "research-review.txt"),
            writes=("research-plan.json",),
            produces=("research",),
            notes="choose the research directions and have a subagent answer each one",
        ))
        # The same pair the create graph carries, for the same reason: the plan's
        # `writes` contract is satisfied by any bytes, so the stage that grounds the
        # design could narrate its intent and pass.
        graph.add(Node(
            "record_findings", "op", call="refactory_dsh.create_ops:record_findings",
            reads=("research-plan.json",),
            writes=("research/",),
            produces=("findings_recorded",),
            notes="write each direction's finding where the reviews will look for it",
        ))
        # Design mode's own research gate: it never fails the run (create's still does).
        graph.add(Node(
            "gate_research", "gate", evaluator=f"{OPS}:gate_design_research",
            reads=("research-plan.json",),
            writes=("research-review.txt",),
            produces=("research_review", "research_attempts_used", "research_advisory"),
            max_iterations=8,
            notes="reloop once only when there is no parseable plan; otherwise proceed, "
                  "with what is off listed on the approval card",
        ))
        graph.add(Node(
            "pitfalls_researcher", "agent",
            prompt_template=(
                "Artifact name: {{name}}\nBrief: {{brief}}\n"
                "Repository: {{source}}\n\n"
                "## Research scope\n{{research_scope}}\n\n"
                "## What the research found\n{{research}}\n\n"
                "## What was wrong with the last attempt\n{{pitfalls_review}}\n\n"
                "You are the pitfalls researcher. Before any strategy or design exists, "
                "identify the risks a correct design must account for. Read the repository "
                "and the brief, and produce a risk-ranked list of pitfalls. For a scoped "
                "change, list at most five: only the risks a correct design of this "
                "change must account for.\n\n"
                "For each pitfall, provide:\n"
                "- **id**: a short identifier in the format PIT-001, PIT-002, etc.\n"
                "- **risk**: high, medium, or low.\n"
                "- **category**: exactly one of:\n"
                "  - `pinning_tests` — tests that pin current behaviour. Name the file, "
                "the line, and the exact assertion text. Say whether the change must keep "
                "it or deliberately change it.\n"
                "  - `repo_conventions` — the repository's own rules and conventions, "
                "taken from its code, docs and agent instructions (CLAUDE.md, AGENTS.md, "
                "CONTRIBUTING, or similar) with citations. Never use a built-in list.\n"
                "  - `shared_state` — other writers and readers of the files, environment "
                "variables, HOME, locks, and processes the change touches.\n"
                "  - `callers` — callers of the code being changed, and the user paths "
                "each serves.\n"
                "  - `imitated_steps` — the existing step the change imitates, and the "
                "settings it reads.\n"
                "  - `scope_boundaries` — modes and paths that must not change.\n"
                "- **evidence**: one of these forms:\n"
                "  - `file:line` \u2014 cite a file and line number from the repository (for code that exists).\n"
                "  - `absent: <path or glob>` \u2014 when the pitfall is about something missing (no tests, no config file). A glob like `tests/test_*.py` is accepted. The gate verifies the path or glob does NOT match anything in the worktree.\n"
                "  - `brief.json` or `research/<slug>.md` \u2014 cite a run artifact when evidence comes from the brief or research, not from repository code.\n"
                "- **prescription**: what a correct design must do about it.\n\n"
                "Sort by risk (high first). Write `pitfalls.json` as a JSON array of "
                "these objects. Then write `pitfalls.md` as a human-readable rendering "
                "of the same list. Do not read strategy.json or design.md — this step "
                "is independent of them."
            ),
            reads=("brief.json", "research-plan.json"),
            writes=("pitfalls.json", "pitfalls.md"),
            produces=("pitfalls",),
            notes="identify risk-ranked pitfalls before strategy, independent of any design",
        ))
        graph.add(Node(
            "gate_pitfalls", "gate", evaluator=f"{OPS}:gate_pitfalls",
            reads=("pitfalls.json",),
            writes=("pitfalls-review.txt",),
            produces=("pitfalls_review", "pitfalls_attempts_used", "pitfalls_advisory"),
            max_iterations=8,
            notes="reloop once only when there are no parseable pitfalls; otherwise proceed, "
                  "with what is off listed on the approval card",
        ))
        # A scoped change skips the perspectives step: both runtimes route on this gate's
        # labels (`scoped`, or `proceed`).
        graph.add(Node(
            "route_strategy", "gate", evaluator=f"{OPS}:route_strategy",
            writes=("strategy-plan.json", "strategy/"),
            produces=("strategy_plan",),
            max_iterations=8,
            notes="a scoped change is planned straight from the research; a full design weighs perspectives",
        ))
        graph.add(Node(
            "strategy_director", "agent",
            prompt_template=(
                "Artifact name: {{name}}\nBrief: {{brief}}\n"
                "It is built in {{source}}; read what is there before choosing.\n\n"
                "## Pitfalls\n{{pitfalls}}\n\n"
                "## What the research found\n{{research}}\n\n"
                "## Research scope\n{{research_scope}}\n\n"
                "You are the strategy director. From the research, choose between two "
                "and five perspectives on how this artifact could be built — including "
                "the simplest thing that could pass the check, and one that says what "
                "the check will not catch. Spawn one subagent per perspective and have "
                "it write `strategy/<slug>.md` under the run directory, with each "
                "finding as a heading `### FND-<slug>-<n> \u2014 <the finding>`, numbered "
                "from 1 within the file. Then write "
                "`strategy-plan.json` as an array of `{\"perspective\", \"slug\", "
                "\"intent_items\", \"ids\"}` objects, where `intent_items` lists the "
                "asks from the brief that perspective covers and `ids` lists the "
                "`FND-<slug>-<n>` ids from that perspective's file. Every ask must "
                "be covered."
            ),
            reads=("brief.json", "research-plan.json", "pitfalls.json"),
            writes=("strategy-plan.json", "strategy/"),
            produces=("strategy_plan",),
            notes="choose the perspectives and record what each one covers",
        ))
        graph.add(Node(
            "synthesize_strategy", "agent",
            prompt_template=(
                "Artifact name: {{name}}\nBrief: {{brief}}\nIt is built in {{source}}.\n"
                "The check, run from {{check_root}}:\n    {{check}}\n\n"
                "## Pitfalls\n{{pitfalls}}\n\n"
                "## What the research found\n{{research}}\n\n"
                "## The perspectives\n{{strategy_plan}}\n\n"
                + _ACCEPTANCE_BASELINE
                + "Compile the perspectives into one plan: what will be built, which files "
                "it needs, how the check will pass, and what is deliberately out of "
                "scope with the user's agreement. Every remaining ask in the brief must "
                "appear in the plan; an unmet ask remains unresolved. "
                "For each ask distinguish what the supplied check establishes from "
                "what still needs independent evidence. Plan executable semantic checks "
                "for the gaps; a successful command or an existing output file does not "
                "establish that the output means what the user requested. For exchange "
                "formats, plan acceptance by an established consuming application or "
                "independent format reader, not just a parser the artifact author writes."
                "\n\nThe research detail files (`research/*.md`) and strategy detail "
                "files (`strategy/*.md`) are available as reads. Each carries finding "
                "ids (`FND-<slug>-<n>`). Use the findings that change what will be built, "
                "citing their ids; the rest stay in those files for the design to read."
            ),
            reads=("brief.json", "research-plan.json", "strategy-plan.json", "pitfalls.json",
                   "research/", "strategy/"),
            writes=("strategy.json",),
            produces=("strategy",),
            notes="compile the perspectives into one plan",
        ))
        # design-v2's `design_doc`: the design is a document of its own, written
        # before the code and approved before anything is built.
        graph.add(Node(
            "design_doc", "agent",
            prompt_template=(
                "Artifact name: {{name}}\nBrief: {{brief}}\nIt is built in {{source}}.\n"
                "The check, run from {{check_root}}:\n    {{check}}\n\n"
                "## The plan\n{{strategy}}\n\n"
                # Empty except on a light refine: the pull request it follows up.
                "{{refine_context}}"
                "## Pitfalls\n{{pitfalls}}\n\n"
                "## What the design checks found\n{{design_review_feedback}}\n\n"
                "\n## Strategy gate feedback\n{{gate_strategy_feedback}}\n\n"
                "{{gate_revision_instruction}}\n\n"
                + _ACCEPTANCE_BASELINE
                + "Write the design document for this artifact: every file it will "
                "add or change, what each one does, the interfaces between them, and how the "
                "check exercises them. Be concrete — the next step writes the code from "
                "this document and the brief. Specify accepted input syntax and unambiguous "
                "rejection behavior: unsupported counts, signs, units, or other values "
                "must not silently become different valid requests. For visual output, "
                "plan a rendered inspection as well as format/geometry checks, including "
                "labels, margins, clipping, and agreement with the requested dimensions.\n\n"
                "Make exactly one decision for each design choice — do not present "
                "alternatives, do not end a choice with 'a simpler alternative…' or leave "
                "it open. A choice left open here becomes an unreviewed decision in the "
                "code. Cite every claim about existing code with its file path and line "
                "number or range (e.g. `path/to/file.py:42` or `path/to/file.py:42-58`). "
                "When claiming what the brief says, quote the brief's words verbatim. "
                "The document holds only the final result — no 'Wait—', 'Actually', "
                "'Let me reconsider' or other working-out.\n\n"
                "For claims about existing code, use the subject-at-citation-quote form: "
                "`name()` at `path/to/file.py:42`: `def name(self):`. The subject "
                "(the backtick span before `at`) names the function, attribute or class "
                "the claim is about; the quote (the backtick span after `:`) is the line "
                "content at that position. This form is mechanically checked, and a claim "
                "the check cannot confirm is shown to the person who approves the design. "
                "Bare citations without a quoted line (for files "
                "being added, or site markers) are fine as-is.\n\n"
                + design_contract.prompt_contract()
                + "\n"
                "Research and perspective findings \u2014 read research/*.md and "
                "strategy/*.md directly, and cite a finding id (FND-<slug>-<n>) where it "
                "changes a decision. Do not list findings that change nothing; the "
                "document is for building the change, not for accounting for the "
                "research. The person who approves the design reads it: keep it short.\n"
                "\n\n## Claims check feedback\n{{claims_check_feedback}}"
            ),
            reads=("brief.json", "strategy.json", "reviews/design-review-feedback.txt",
                   "pitfalls.json", "research/", "strategy/"),
            writes=("design.md",),
            produces=("design",),
            notes="write the design document the code is written from",
        ))
        # An op, not a gate: it never sends the design back (one design pass). What it
        # cannot confirm is listed first on the approval gate's card.
        graph.add(Node(
            "check_design_claims", "op",
            call="refactory_dsh.design_claims:check_design_claims",
            reads=("design.md",),
            produces=("claims_check", "claims_check_feedback"),
            notes="deterministic check: cited paths exist, quoted lines match within a "
                  "window, named identifiers are present; resolves drift, lists the rest",
        ))
        graph.add(Node(
            "gate_design_review", "gate",
            evaluator=f"{OPS}:gate_design_review",
            reads=("design.md", "brief.json", "pitfalls.json", "research/", "strategy/"),
            writes=("reviews/design-review-feedback.txt",),
            produces=("design_review_feedback", "design_review_attempts_used",
                      "design_review_rewrites", "design_review_findings", "pitfall_coverage",
                      "finding_coverage", "design_review_status", "design_card",
                      "design_review_advisory", "design_review_blocking",
                      "gate_strategy_feedback", "reloop_scope", "reloop_scope_reason",
                      "gate_revision_instruction"),
            knobs=("design_review_attempts",),
            max_iterations=6,
            notes="reloop to design_doc only for a missing or empty ## Changes; list "
                  "everything else on the approval card; never fail the run",
        ))
        # ── classify + route the person's reloop at gate_strategy ──────────
        # classify_reloop: an LLM judge that reads the person's feedback and
        # classifies it as `document` (wording/site/test fix) or `approach`
        # (the strategy itself needs changing).
        graph.add(Node(
            "classify_reloop", "agent",
            tool_policy="none",
            prompt_template=(
                "The person reviewed the design and sent it back with feedback.\n"
                "Read their feedback above.\n\n"
                "Classify whether they want:\n"
                "- **document**: the feedback is about wording, formatting, structure, "
                "missing details, a site, or a test in design.md — the technical "
                "approach itself is fine.\n"
                "- **approach**: the feedback rejects or changes the strategy, "
                "architecture, algorithm choice, or component boundaries.\n\n"
                "When in doubt, choose approach.\n\n"
                "Write exactly one JSON object with two fields:\n"
                "```json\n"
                '{"scope": "document" | "approach", '
                '"reason": "one-line reason"}\n'
                "```"
            ),
            writes=("reloop-scope.json",),
            notes="classify the person's strategy-gate feedback as document or approach",
        ))
        # scope_reloop: a fn-gate that reads the classifier's JSON and routes
        # to synthesize_strategy (approach) or design_doc (document).
        graph.add(Node(
            "scope_reloop", "gate",
            evaluator=f"{OPS}:scope_reloop",
            reads=("reloop-scope.json",),
            produces=("reloop_scope", "reloop_scope_reason", "gate_strategy_feedback",
                      "gate_revision_instruction"),
            max_iterations=4,
            notes="route to synthesize_strategy (approach) or design_doc (document)",
        ))
        # design-v2 puts the person's approval on the design document, after the plan
        # exists and before any code does.
        # The approval covers delivery too, which is the only step that reaches outside
        # the run: delivery.md says which pull request it opens, or which repository it
        # creates, and the approval is scoped to that file as it is to design.md.
        graph.add(Node(
            "gate_strategy", "gate", gate_kind="user",
            reads=("design.md", "delivery.md"),
            max_iterations=4,
            # `{{delivery}}` is filled from the run's facts wherever the question is put
            # (the spine's question card, the panel, the CEO's view), so the card itself
            # names the repository a run would create, and who can see it. The card is
            # plain text, as the question card shows it (`design_ops.design_card`, dm-163).
            notes=(
                "{{design_card}}\n\n"
                "Approve the design before the artifact is written? It is in design.md "
                "in the run directory. Then, {{delivery}}; delivery.md there says more. "
                "Answer proceed to build and deliver it that way, or reloop to have the "
                "plan redone."
            ),
        ))
        graph.add(Node(
            "emit", "agent",
            prompt_template=(
                "Write the artifact `{{name}}` so that it is complete and runnable.\n\n"
                "Brief: {{brief}}\n"
                "It will be checked by running: {{check}}\n\n"
                "## The design\nRead the approved design from `design.md` in the run directory.\n\n"
                "## What went wrong last time\n{{validation}}\n\n"
                "## What the reviews refused\n{{qa_gate}}\n{{overwatch_review}}\n\n"
                + _ACCEPTANCE_BASELINE
                + WORKSPACE_BRIEFING
                + UPDATED_TESTS_COMPLETION
                + EMIT_CHECKLIST
            ),
            reads=("brief.json", "design.md", "check.txt", "check-record.json"),
            writes=(repo_change.EMIT_FILE,),
            produces=("emit",),
            feedback_keys=("validation", "qa_gate", "overwatch_review"),
            workspace=dict(BUILDER_WORKSPACE),
            # Behaviour change: emit gains timeout_s and retry, which changes the
            # candidate id; runs in flight at this commit cannot be resumed.  The
            # spine reservation for the builder roughly doubles (1800 + 3600 plus
            # overhead and preparation).
            timeout_s=builder_timeout(self.values()),
            retry=TIMEOUT_RETRY,
            knobs=("builder_timeout_s",),
            notes="make the change in a clone of the last attempt; its diff is the emission",
        ))
        graph.add(repo_change.materialise_node(
            call=f"{OPS}:design_materialise",
            # check.txt: at the refusal bound, the check's receipt is put back for the draft
            # (`design_ops._draft_for_stop`). updated-tests.json: what the builder says it
            # updated (`design_ops.note_updated_tests`).
            writes=("changes.diff", "changed-functions.json", "check.txt", UPDATED_TESTS),
            produces=("artifact", "files", "commit", "change", "emission_refused",
                      "changed_functions", "consecutive_refusals", "draft_delivery",
                      "updated_tests", "empty_emission_fingerprint"),
            knobs=("refusal_attempts",),
        ))
        # ── the check, beside the reviews ──
        # prepare_check settles the commit the check and the reviewers both see; the
        # check runs as a branch of the review fork and records its decision; gate_check
        # routes on it once the reviews are done, because a node inside a fork may not
        # send the run outside it.
        check_produces = ("validation", "check_attempts_used", "review_attempts_used",
                          "attempts_used", "runtime_evidence", "base_unavailable",
                          "check_failures_history", "check_blocker_history",
                          "convergence_extras_used", "convergence_granted", "inherited_failures",
                          "empty_emission_fingerprint")
        check_writes = ("check.txt", "validation.txt", "authoring-evidence.json",
                        "check-record.json", "test-guardedness.json", CHECK_VERDICT)
        check_knobs = ("check_attempts", "check_timeout_s", "check_attempt_cap")
        defaults = {name: knob.default for name, knob in self.declare().items()}
        self_check_produces = (*check_produces, *repo_change.CHECK_PRODUCES,
                               "review_feedback", "review_baseline",
                               "self_check_fixup_count", "total_self_check_fixups_used",
                               "builder_tests", "draft_delivery", "early_asks_rereads",
                               "early_asks_reread")
        graph.add(Node(
            "prepare_check", "gate", evaluator=f"{OPS}:prepare_check", role="validation",
            resource_role="final_validation",
            reads=("brief.json",),
            writes=(*check_writes, *repo_change.CHECK_WRITES, CHECK_PREP, SELF_CHECK_RECORD),
            produces=self_check_produces,
            knobs=(*check_knobs, "self_check_fixups", "self_check_timeout_s"),
            # From the knobs, so the build loop's stop always comes first: 45 with the
            # defaults (`prepare_check_iterations`; `DesignMode.graph` follows the knobs).
            max_iterations=prepare_check_iterations(defaults),
            notes="settle the commit the check and the reviews see; every attempt's "
                  "self-check runs the tests the change touches here (first or fix-up): "
                  "a failure goes back to emit as a fix-up, spending no attempt and running "
                  "no review; the full check runs beside the reviews in fork_qa; a limit "
                  "stop with an applied commit and current check evidence goes to the "
                  "report as a draft",
        ))
        # ── the early asks judge, beside the check and the reviews in fork_qa ──
        graph.add(Node(
            "early_asks_check", "agent",
            prompt_template=EARLY_ASKS_PROMPT,
            reads=("brief.json", "changes.diff"),
            writes=(EARLY_ASKS_VERDICT,),
            resource_role="final_validation",
            notes="judge whether the attempt builds each ask and correction",
        ))
        graph.add(Node(
            "check", "op", call=f"{OPS}:check_beside_reviews", role="validation",
            resource_role="final_validation",
            reads=("brief.json", CHECK_PREP),
            writes=check_writes, produces=check_produces,
            knobs=check_knobs, max_iterations=8,
            notes="run the brief's check, in a sandbox, beside the reviews",
        ))
        graph.add(Node(
            "gate_check", "gate", evaluator=f"{OPS}:gate_check", role="validation",
            resource_role="final_validation",
            reads=(CHECK_PREP, CHECK_VERDICT, EARLY_ASKS_VERDICT,
                   "reviews/health-check.md", "reviews/code-review.md",
                   "reviews/qa.md", "reviews/verification.md"),
            # It runs the whole check itself when no decision stands for this attempt.
            writes=(*check_writes, *repo_change.CHECK_WRITES),
            produces=(*check_produces, *repo_change.CHECK_PRODUCES, "draft_delivery",
                      "combined_sendback_count", "early_asks_sendback_count"),
            knobs=check_knobs, max_iterations=8,
            notes="compose one send-back from check failures and review findings without "
                  "spending an attempt; the combined send-back names every failing test "
                  "and every review finding of the round; a failing check skips the QA "
                  "judge and overwatch by graph topology (gate_check → pick_corrections "
                  "bypasses synthesize_qa → qa_judge → overwatch); at its limit, deliver "
                  "the attempt as a draft marked check failing; when the check and all "
                  "reviews passed, proceed",
        ))

        # ── design-v2's three reviews, in parallel, and the overwatch pass ──
        from refactory_dsh.verification import DESIGN_VERIFIER_TIMEOUT_S, VERIFIER_PROMPT
        graph.add(Node(
            "verifier", "agent",
            retry=TIMEOUT_RETRY, outputs_only=True,
            prompt_template=VERIFIER_PROMPT + BESIDE_CHECK_VERIFIER_NOTE,
            timeout_s=DESIGN_VERIFIER_TIMEOUT_S,
            # Not test-guardedness.json: the check beside it writes that file.
            reads=("brief.json", "changes.diff"),
            writes=("reviews/verification.md",),
            produces=("verification",),
            role="code_reviewer",
            resource_role="final_validation",
            notes="design-blind verification: apply the evidence log's methods "
                  "to the brief, diff and repo",
        ))
        graph.add(Node("fork_qa", "fork",
                       targets=("health_checker", "code_reviewer", "qa_director", "verifier",
                                "check", "early_asks_check"),
                       optional_targets=("verifier", "early_asks_check"),
                       notes="start the three reviews, the verifier, the check and the "
                             "early asks judge"))
        graph.add(Node(
            "health_checker", "agent",
            prompt_template=(
                "Artifact directory (absolute): {{artifact}}\n"
                "Read and run the delivered files at this exact path; do not infer it "
                "from your working directory. {{change}}\n\n"
                "Brief: {{brief}}\nThe check it must pass: {{check}}\n\n"
                "## The design\nRead the approved design from `design.md` in the run directory.\n\n"
                + _ACCEPTANCE_BASELINE
                + "You are the health checker. Run the artifact the way a user would: run "
                "targeted checks for uncovered requirements. The check itself runs beside "
                "this review, so do not run it again. Record every command and "
                "its exact output in `reviews/health-check.md`. "
                "Anything that fails or is "
                "missing is written as BLOCKING: <path>: <description> where <path> is the file the builder must change to fix the problem. A defect in the code is reported against the code file, even when the requirement it misses comes from design.md or the brief. A problem in a run document (design.md, strategy.json, or another file an earlier step wrote) is reported with that document's path. A "
                "check you did not run is not a check that passed.\n"
                + SEVERITY_RULE + " " + MARKER_RULE + "\n"
                "{{review_instructions}}\n"
                "Changed functions:\n{{changed_functions}}\n\n"
            ),
            reads=("brief.json", "design.md", "changed-functions.json"),
            writes=("reviews/health-check.md",),
            produces=("health_check",),
            notes="run the artifact and record what happened, command by command",
        ))
        graph.add(Node(
            "code_reviewer", "agent",
            prompt_template=(
                "## Pitfalls\n{{pitfalls}}\n\n"
                "Check the built change against every high-risk pitfall. A high pitfall "
                "the code contradicts is BLOCKING.\n\n"
                "Artifact directory (absolute): {{artifact}}\n"
                "Read and run the delivered files at this exact path; do not infer it "
                "from your working directory. {{change}}\n\n"
                "Brief: {{brief}}\n\n"
                "## The design\nRead the approved design from `design.md` in the run directory.\n\n"
                "## The emitted source\nRead what was emitted from `emit.txt` in the run directory. "
                "In a worktree it may edit or delete files rather than write them whole, so judge "
                "the result: the files at the artifact path, and the diff named above.\n\n"
                + _ACCEPTANCE_BASELINE
                + UPDATED_TESTS_NOTE
                + "You are the code reviewer. Read the delivered files against the design "
                "and the brief: is every file the design names present, does it do what "
                "the design says, and would it run? Trace input values through parsing "
                "to the delivered artifact: count/role binding, signs, units, and "
                "unsupported input must preserve the request or explicitly reject it. "
                "A matching number elsewhere in prose is not evidence for the requested "
                "field. Record findings in "
                "`reviews/code-review.md`. A defect that would stop the artifact working "
                "is written as BLOCKING: <path>: <description> where <path> is the file the builder must change to fix the problem. A defect in the code is reported against the code file, even when the requirement it misses comes from design.md or the brief. A problem in a run document (design.md, strategy.json, or another file an earlier step wrote) is reported with that document's path.\n"
                + SEVERITY_RULE + " " + MARKER_RULE + "\n"
                "{{review_instructions}}\n"
                "Changed functions:\n{{changed_functions}}\n\n"
                "When a false claim about design.md is found, report it against the code file "
                "that relies on the claim, not against design.md, because a blocking finding against design.md "
                "is downgraded by scope_filter_findings.\n"
            ),
            reads=("brief.json", "design.md", "emit.txt", "pitfalls.json",
                   "changed-functions.json"),
            writes=("reviews/code-review.md",),
            produces=("code_review",),
            notes="review the emitted files against the design",
        ))
        graph.add(Node(
            "qa_director", "agent",
            prompt_template=(
                "Artifact directory (absolute): {{artifact}}\n"
                "Read and run the delivered files at this exact path; do not infer it "
                "from your working directory. {{change}}\n\n"
                "Brief: {{brief}}\nThe check it must pass: {{check}}\n\n"
                "## The approved design\nRead the approved design from `design.md` in the run directory.\n\n"
                "## The plan\nRead the synthesis strategy from `strategy.json` in the run directory.\n\n"
                + _ACCEPTANCE_BASELINE
                + UPDATED_TESTS_NOTE
                + "You are the QA director. Design adversarial checks on the artifact — "
                "the cases a friendly reviewer would not try — and run them. Vary one "
                "semantic property at a time and inspect the actual output, not just "
                "the exit code. Include conflicting nearby numbers, unsupported counts, "
                "and alternative sign/unit encodings when the interface parses them. "
                "Accepted output must match the request; unsupported input must fail "
                "clearly without silently substituting a valid case. Write "
                "`reviews/qa.md` with each check and its exact result. Anything that "
                "breaks an ask in the brief is written as BLOCKING: <path>: <description> where <path> is the file the builder must change to fix the problem. A defect in the code is reported against the code file, even when the requirement it misses comes from design.md or the brief. A problem in a run document (design.md, strategy.json, or another file an earlier step wrote) is reported with that document's path.\n"
                + SEVERITY_RULE + " " + MARKER_RULE + "\n"
                "{{review_instructions}}\n"
                "Changed functions:\n{{changed_functions}}\n\n"
            ),
            reads=("brief.json", "design.md", "strategy.json",
                   "changed-functions.json"),
            writes=("reviews/qa.md",),
            produces=("qa_review",),
            notes="attack the artifact against the brief",
        ))
        graph.add(Node("join_qa", "join",
                       sources=("check", "health_checker", "code_reviewer", "qa_director",
                                "early_asks_check", "verifier"),
                       notes="wait for the check, all three reviews, the verifier "
                             "and the early asks judge"))
        # ── the reviews report, and two judges decide (refactory_dsh.judges) ──
        graph.add(Node(
            "synthesize_qa", "op", call=f"{OPS}:synthesize_qa",
            reads=("reviews/health-check.md", "reviews/code-review.md", "reviews/qa.md",
                   "reviews/verification.md", "check-record.json", "design.md"),
            writes=("reviews/qa-synthesized.md",),
            # It starts each round's judging: the last round's verdicts are set aside and
            # the judges' rereads start again.
            produces=("qa", "qa_missing", "qa_judge_rereads", "qa_judge_reread",
                      "overwatch_rereads", "overwatch_reread", "correction_audit"),
            notes="put this attempt's reviews, its check and its scope in one document, "
                  "unread; an absent review is not a passed one",
        ))
        graph.add(Node(
            "qa_judge", "agent",
            prompt_template=QA_JUDGE_PROMPT,
            # Not reviews/verification.md: the verifier is optional, and a declared input
            # must exist on the spine. The synthesis carries its report when it gave one.
            reads=("brief.json", "design.md", "reviews/qa-synthesized.md"),
            writes=(judges.QA_JUDGE,),
            resource_role="final_validation",
            notes="decide from the reviews whether the change goes on or back to the builder",
        ))
        graph.add(Node(
            "gate_qa", "gate", evaluator=f"{OPS}:gate_qa",
            reads=(judges.QA_JUDGE, "reviews/health-check.md", "reviews/code-review.md",
                   "reviews/qa.md", "changed-functions.json"),
            writes=("reviews/qa-gate.txt", "reviews/follow-ups.json", "reviews/follow-ups.md"),
            produces=("qa_gate", "qa_judge_rereads", "qa_judge_reread", "qa_draft",
                      "qa_unresolved"),
            knobs=("judge_rereads",),
            max_iterations=5,
            notes="route on the QA judge's verdict: a redo for what holds the change back goes "
                  "to the builder, or, at the review limit, on as a draft; an unreadable "
                  "verdict is read again once; follow-ups go to the pull request",
        ))
        # The correction audit measures the QA judge's evidence map before the overwatch
        # reads it; it routes nothing.
        graph.add(Node(
            "audit_corrections", "op", call=f"{OPS}:audit_corrections",
            reads=("authoring-evidence.json", "intent-audit.json", "design.md"),
            writes=("correction-audit.json", "site-revert-results.json"),
            produces=("correction_audit",),
            resource_role="final_validation",
            notes="measure each correction's sites in the diff and their tests, as evidence "
                  "for the overwatch",
        ))
        graph.add(Node(
            "overwatch", "agent",
            prompt_template=OVERWATCH_PROMPT,
            # Not qa-judge.json, intent-audit.json or correction-audit.json: a declared input
            # must exist on the spine, and a QA judge or a correction audit a person skipped
            # leaves none this round (synthesize_qa sets the last round's aside). The prompt
            # reads them when they are there.
            reads=("brief.json", "design.md", "reviews/qa-synthesized.md",
                   "changes.diff", "test-guardedness.json"),
            writes=(judges.OVERWATCH_VERDICT,),
            produces=("overwatch",),
            notes="verify acceptance, with the correction audit as evidence, and judge",
        ))
        graph.add(Node(
            "gate_overwatch", "gate", evaluator=f"{OPS}:gate_overwatch",
            reads=(judges.OVERWATCH_VERDICT, "correction-audit.json"),
            writes=("reviews/overwatch-gate.txt", "reviews/follow-ups.json", "reviews/follow-ups.md"),
            produces=("overwatch_review", "overwatch_rereads", "overwatch_reread", "draft_delivery",
                      "overwatch_unresolved"),
            knobs=("judge_rereads",),
            max_iterations=4,
            notes="route on the overwatch's verdict: an unmet correction or a redo for what "
                  "holds the change back goes to the builder, or, at the review limit, to the "
                  "report as a draft; a pass to the report; an unreadable verdict is read "
                  "again once; follow-ups go to the pull request",
        ))
        graph.add(Node(
            "report", "op", call=f"{OPS}:report_judged",
            writes=("design.json",),
            produces=("designed",),
            notes="summarise what was designed",
        ))
        # After the report, so only a run whose reviews passed and whose sign-offs held
        # reaches anything outside it (#350), or a draft: a limit stop with an applied
        # commit and current check evidence (`design_ops._draft_for_stop`).
        graph.add(Node(
            "pick_corrections", "gate", evaluator=f"{OPS}:pick_gate_corrections",
            produces=("gate_corrections", "design_corrections_applied", CORRECTIONS_PICKED),
            max_iterations=2,
            notes="deterministically extract the person's corrections to the approved design "
                  "from the ledger, after the approval and on every send-back to the builder; "
                  "skip the agent when none is new",
        ))
        graph.add(Node(
            "apply_corrections", "agent",
            prompt_template=(
                "Read `design.md` from the run directory.\n\n"
                "The person gave the following corrections when approving the design, or "
                "since:\n\n"
                "{{gate_corrections}}\n\n"
                "Update `design.md` so that every part of it the corrections touch says what "
                "they say: the heading (H1), ## Summary, ## Changes and ## Tests, and any "
                "optional section the design has (such as ## Rule and sites or ## Docs) "
                "whose wording, example text or file list a correction replaces or adds "
                "to. Leave no part stating what a correction overrides, because the "
                "builder reads the whole document. Keep the sections the corrections do "
                "not touch as they are, and keep the document's section headings and "
                "structure. Then add a `## Corrections at approval` section, or bring the "
                "one the design has up to date: one bullet "
                "per correction, quoting it and naming the section(s) of the design that "
                "now carry it. Write the updated `design.md`. "
                "Return a short message listing what was changed."
            ),
            reads=("design.md",),
            writes=("design.md",),
            notes="revise design.md with the person's corrections before the builder runs",
        ))
        graph.add(Node(
            "deliver", "op", call="refactory_dsh.delivery:deliver_design",
            resource_role="final_validation",
            knobs=("check_timeout_s",),
            reads=("design.json", "delivery.md", "reviews/follow-ups.json"),
            writes=("delivery.json", "pull-request.md", "design.json"),
            produces=("delivery_result", "pull_request", "delivery_conflict_files",
                      "delivery_target_tip", "delivered_commit"),
            notes="push the run's branch and open its pull request, as the approval gate said",
        ))
        graph.add(Node(
            "delivery_gate", "gate",
            evaluator="refactory_dsh.delivery:evaluate_delivery_conflicts",
            max_iterations=8,
            knobs=("check_attempts",),
            produces=("validation", "delivery_conflict_reloops"),
            notes="on authored-file conflict reloop to the builder; otherwise proceed to done",
        ))
        graph.add(Node(
            "done", "op", call="refactory_dsh.design_ops:noop",
            notes="terminal node after delivery succeeds",
        ))

        graph.connect("prepare", "route_refine_design")
        graph.connect("route_refine_design", "research_director", condition="proceed")
        graph.connect("route_refine_design", "design_doc", condition="light")
        graph.connect("research_director", "record_findings")
        graph.connect("record_findings", "gate_research")
        graph.connect("gate_research", "research_director", condition="reloop")
        graph.add(Node(
            "classify_scale", "op", call=f"{OPS}:classify_scale",
            reads=("research-plan.json",),
            produces=("research_scope", "research_scope_reason"),
            notes="classify the change's scale from the research footprint",
        ))
        graph.connect("gate_research", "classify_scale", condition="proceed")
        graph.connect("classify_scale", "pitfalls_researcher")
        graph.connect("pitfalls_researcher", "gate_pitfalls")
        graph.connect("gate_pitfalls", "pitfalls_researcher", condition="reloop")
        graph.connect("gate_pitfalls", "route_strategy", condition="proceed")
        graph.connect("route_strategy", "strategy_director", condition="proceed")
        graph.connect("route_strategy", "synthesize_strategy", condition="scoped")
        graph.connect("strategy_director", "synthesize_strategy")
        graph.connect("synthesize_strategy", "design_doc")
        graph.connect("design_doc", "check_design_claims")
        graph.connect("check_design_claims", "gate_design_review")
        graph.connect("gate_design_review", "design_doc", condition="reloop")
        graph.connect("gate_design_review", "gate_strategy", condition="proceed")
        graph.connect("gate_strategy", "classify_reloop", condition="reloop")
        graph.connect("classify_reloop", "scope_reloop")
        graph.connect("scope_reloop", "synthesize_strategy", condition="approach")
        graph.connect("scope_reloop", "design_doc", condition="document")
        graph.connect("gate_strategy", "pick_corrections", condition="proceed")
        # Every send-back to the builder goes through pick_corrections, so a correction the
        # person gives after the approval is applied to design.md before the builder's next
        # round (dm-226); with nothing new it skips straight to emit.
        graph.connect("pick_corrections", "apply_corrections", condition="proceed")
        graph.connect("pick_corrections", "emit", condition="skip")
        graph.connect("apply_corrections", "emit")
        graph.connect("emit", "materialise")
        graph.connect("materialise", "prepare_check")
        graph.connect("prepare_check", "pick_corrections", condition="reloop")
        graph.connect("prepare_check", "fork_qa", condition="proceed")
        graph.connect("prepare_check", "report", condition="draft")
        graph.connect("fork_qa", "join_qa")
        # No explicit edges from early_asks_check or verifier to join_qa:
        # the fork→join drains all targets (forwardWalk adds fork targets),
        # and join_qa.sources includes both for uncompleteBetween on reloop.
        # An explicit edge would cause advance() to add join_qa to the cursor
        # when one branch completes, before the other branches finish.
        graph.connect("join_qa", "gate_check")
        graph.connect("gate_check", "pick_corrections", condition="reloop")
        graph.connect("gate_check", "synthesize_qa", condition="proceed")
        graph.connect("gate_check", "report", condition="draft")
        graph.connect("synthesize_qa", "qa_judge")
        graph.connect("qa_judge", "gate_qa")
        graph.connect("gate_qa", "pick_corrections", condition="reloop")
        # A judge-only reread: the QA judge's verdict could not be read, so it is asked
        # again, with no builder round (`design_ops._read_judge`; `judge_rereads` caps it).
        graph.connect("gate_qa", "qa_judge", condition="reread")
        graph.connect("gate_qa", "audit_corrections", condition="proceed")
        graph.connect("audit_corrections", "overwatch")
        graph.connect("overwatch", "gate_overwatch")
        # No `audit` edge back to the overwatch: the evidence map's shape is evidence the
        # overwatch reads (`design_ops.audit_corrections`), not a free round of its own.
        # The `reread` edge is the judge-only reread of a verdict that could not be read.
        graph.connect("gate_overwatch", "pick_corrections", condition="reloop")
        graph.connect("gate_overwatch", "overwatch", condition="reread")
        graph.connect("gate_overwatch", "report", condition="proceed")
        graph.connect("report", "deliver")
        graph.connect("deliver", "delivery_gate")
        graph.connect("delivery_gate", "pick_corrections", condition="reloop")
        graph.connect("delivery_gate", "done", condition="proceed")
        from refactory_dsh.authoring_evidence import wire_reviews
        wire_reviews(graph, check_beside=True, audit_node="qa_judge", judge_nodes=("overwatch",))
        return graph

    def invariants(self) -> tuple[Any, ...]:
        return ()


def main(argv: list[str] | None = None) -> int:
    """``python -m refactory_dsh.design --name <n> --brief <text> --check <cmd>``.

    The walker's main. Its run flags are `refactory ceo`'s (`run_flags`, dm-134): the same
    spelling and the same knobs, so a run started either way is configured the same.
    """
    import argparse
    import json
    import sys

    from refactory_dsh import run_flags
    from refactory_dsh.run_preparation import ACKNOWLEDGE_HELP, RESUME_HELP

    parser = argparse.ArgumentParser(
        prog="python -m refactory_dsh.design",
        description="Design, emit and check a software artifact from a brief.",
    )
    parser.add_argument("--brief", required=False, default="", help="what to build")
    parser.add_argument("--fast", action="store_true", help="use the inspectable design_fast graph with focused QA")
    run_flags.add_arguments(parser)
    parser.add_argument("--resume", default="", metavar="RUN_DIR", help=RESUME_HELP)
    parser.add_argument("--acknowledge-failure", action="store_true", help=ACKNOWLEDGE_HELP)
    args = parser.parse_args(argv)
    if args.acknowledge_failure and not args.resume:
        parser.error("--acknowledge-failure goes with --resume: it acknowledges the boundary a run stopped at")

    from refactory_dsh.model import DshModelClient, ModelChoiceError, require_model_choice
    from refactory_dsh.paths import project_root

    if args.resume:
        problem = run_flags.resume_refusal(parser, args)
        if problem:
            parser.error(problem)
        workdir = Path(args.resume)
        try:
            client = require_model_choice(DshModelClient(project=None), interactive=sys.stdin.isatty())
        except ModelChoiceError as exc:  # dm-258
            print(f"refactory: {exc}", file=sys.stderr)
            return 1
        from refactory_dsh.fast import DesignFastMode
        from refactory_dsh.run_preparation import resume_walked

        try:
            mode_class, run = resume_walked(workdir, "design", (DesignMode, DesignFastMode), model=client,
                                            acknowledge_failure=args.acknowledge_failure,
                                            resource_limits=run_flags.resource_limits(args))
        except Exception as exc:
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
            return 2
        if run.gate_stop:
            from refactory_dsh.run_preparation import print_gate_stop
            print_gate_stop(run.gate_stop, workdir, graph=mode_class().build())
            return 2
        print(json.dumps({
            "resumed": str(workdir), "exit_ok": run.exit_ok, "error": run.error,
            "facts": {key: value for key, value in run.facts.items() if key in
                      ("name", "files", "artifact", "check", "validation")},
        }, indent=2, default=str))
        if not run.exit_ok:
            try:
                node = mode_class().graph().nodes.get(str(run.error or "").split(":", 1)[0])
                if node is not None and node.kind == "gate" and node.gate_kind == "user":
                    from refactory_dsh.design_ops import design_review_status
                    drs = design_review_status(run.facts or {})
                    if drs:
                        print(drs, file=sys.stderr)
            except Exception:
                pass
        return 0 if run.exit_ok else 2

    if not args.name:
        parser.error("--name is required when not resuming")
    if not args.brief:
        parser.error("--brief is required when not resuming")
    mode_id = "default_pack/design_fast" if args.fast else "default_pack/design"
    problem = run_flags.refusal(args, mode_id, runtime=run_flags.WALKER, label="--fast" if args.fast else "design")
    if problem:
        parser.error(problem)

    from refactory_dsh.create import parse_verdicts
    from refactory_dsh.fast import DesignFastMode
    from refactory_dsh.interactive import with_knobs

    mode = DesignFastMode() if args.fast else DesignMode()
    # The run's settings travel as the knobs `refactory ceo` gives the spine, not as a task.
    knobs = run_flags.knobs(args, mode_id, brief=args.brief, name=args.name)
    try:
        mode.load_state_dict(with_knobs(mode.state_dict(), knobs))
        verdicts = parse_verdicts(args.verdict)
        resource_limits = run_flags.resource_limits(args)
    except ValueError as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2
    workdir = Path(args.workdir or f".factory/design/{args.name}")
    mode_name = mode_id.split("/", 1)[1]

    try:
        client = require_model_choice(DshModelClient(project=project_root(), **run_flags.model_options(args)),
                                      interactive=sys.stdin.isatty())
    except ModelChoiceError as exc:  # dm-258
        print(f"refactory: {exc}", file=sys.stderr)
        return 1
    from refactory_dsh.run_ownership import ownership

    try:
        with ownership(workdir):
            # Run preparation.
            from refactory_dsh.run_preparation import prepare_run, walker_context
            from refactory_dsh import steering

            prepare_run(
                mode.state_dict(), workdir,
                invocation_name=f"default_pack-{mode_name}",
                mode_id=mode_id,
                intent=args.brief,
                transform=lambda workflow, stage_dir=None: steering.for_run(
                    workflow, stage_dir or workdir, omit_steering=getattr(args, "omit_steering", False)),
                runtime="walker",
                command="python -m refactory_dsh.design",
                resume_command=f"python -m refactory_dsh.design --resume {workdir}",
                fast=args.fast,
                worker_backend=client.route().get("worker_backend", "") if callable(getattr(client, "route", None)) else "",
                project=project_root(),
                knobs=knobs,
                model_context=walker_context(client),
                omit_steering=getattr(args, 'omit_steering', False),
            )

            # Persist up-front verdicts as run-scoped ledger entries.
            ledger = steering.Ledger(workdir)
            for gate, value in verdicts.items():
                ledger.verdict(gate, value, approval=True)

            if not args.fast:
                # A gate answered up front shows nobody delivery.md, so the plan is said here (#350).
                from refactory_dsh import delivery

                values = mode.values()
                knob_values = {name: values[name] for name in ("name", "brief", "target", "base", "refine", "worktree",
                                                               "pr", "repo_owner", "repo_visibility")}
                (delivery.announce if "gate_strategy" in verdicts else delivery.forget)(knob_values, workdir)

            result = mode.run(workdir, model=client, task={}, resource_limits=resource_limits, enable_recovery=True)
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2
    if result.gate_stop:
        from refactory_dsh.run_preparation import print_gate_stop
        print_gate_stop(result.gate_stop, workdir, graph=mode.build())
        return 2
    if result.halted:
        print(json.dumps({"halted": True, "error": result.error}), file=sys.stderr)
        return 2
    if not result.exit_ok:
        print(json.dumps({"error": result.error}), file=sys.stderr)
        # dm-127: show design review status when stopped at a user gate.
        try:
            node = mode.graph().nodes.get(str(result.error or "").split(":", 1)[0])
            if node is not None and node.kind == "gate" and node.gate_kind == "user":
                from refactory_dsh.design_ops import design_review_status
                drs = design_review_status(result.facts or {})
                if drs:
                    print(drs, file=sys.stderr)
        except Exception:
            pass
        return 2
    print(json.dumps({
        "artifact": str(result.facts.get("artifact")),
        "branch": result.facts.get("branch"),
        "pull_request": result.facts.get("pull_request"),
        "delivery": result.facts.get("delivery_result"),
        "files": result.facts.get("files", []),
        "check": result.facts.get("check"),
        "iterations": result.iterations,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - the entry point
    raise SystemExit(main())
