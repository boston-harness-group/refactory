"""Independent verification step: design-blind, running in parallel with the reviewers.

The verifier is given the brief, the diff and the repository — never design.md.
It applies the evidence log's design-blind methods and reports each finding with a
runnable counterexample or a cited file:line. For a change to an existing repository
the code reviewers already apply M2, M3, M11 and M12 to every changed function
(`REPO_CHANGE_REVIEW_INSTRUCTIONS`), so there the verifier applies only the ones no
reviewer does (M1, M2, M13, M15) within its bound (`DESIGN_VERIFIER_TIMEOUT_S` in
design mode, `VERIFIER_TIMEOUT_S` in design_fast); for a new artifact, and in fix_pack,
which has no reviewers, it applies all of them. In the review fork it is an optional
branch: a verifier that fails, or in design mode runs out of time twice (its declared
retry, `design.TIMEOUT_RETRY`), is missing, and its findings never get a builder round
of their own.

See docs/hardening.md §15 for why the verifier is design-blind.
"""

from __future__ import annotations

from typing import Any

from refactory_dsh.create_ops import MARKER_RULE

#: The most wall time the review fork's verifier session may take, on either runtime.
#: It runs beside the three reviews, which take about eight minutes, so a verifier
#: inside this bound adds little or nothing to the round. Its prompt asks for less,
#: because a session stopped at the wall leaves nothing behind.
VERIFIER_TIMEOUT_S = 600

#: Design mode's own verifier bound, which a medium change fits. Measured from the worker
#: receipts of the lab's design runs. Before any bound (2026-10-21 and 22), 22 of the 23
#: verifier sessions on changes of 500 diff lines or more finished within 1,200 s: median
#: 740 s, longest 1,467 s. Under the 600 s bound, 45 of the 88 sessions since timed out (45
#: of 73 on changes of 100 lines or more; 7 of 8 such sessions on 2026-10-27). A session
#: that still runs out is retried once with twice the limit (`design.TIMEOUT_RETRY`), so
#: the verifier is missing only after 3,600 s in all.
#: design_fast keeps `VERIFIER_TIMEOUT_S`, and no retry.
DESIGN_VERIFIER_TIMEOUT_S = 1200

_METHODS_HEADER = """
Apply each of these design-blind verification methods to the brief, the diff
and the repository. Do not read design.md or strategy.json.
"""

_REVIEW_FORK_NOTE = """The code reviewers already check callers, shared state and the diff's
own claims; do not repeat that work. Aim to finish within about seven minutes, and
write each finding to reviews/verification.md as you find it.
"""

M1_METHOD = """M1  Adversarial inputs — find inputs the change accepts that would produce
    wrong output or violate the brief's constraints."""

M2_METHOD = """M2  Trace a real case — pick a concrete scenario the brief describes, trace it
    through the diff line by line, and report where it goes wrong or is not handled."""

M3_METHOD = """M3  Where else does this pattern appear — search the repository for every call
    site, re-export, or copy of the code the diff changes, and report any that
    the diff does not update."""

M11_METHOD = """M11 Verify claims against code — for every claim the diff's comments or
    docstrings make, confirm the code matches."""

M12_METHOD = """M12 Shared state — identify every file, environment variable, lock or process
    the diff's code reads or writes, find every other reader/writer in the
    repository, and report conflicts."""

M13_METHOD = """M13 Boundary values — for every numeric or string parameter the diff parses,
    test zero, one, maximum, empty, and type-boundary values against the code."""

M15_METHOD = """M15 Weakened tests — for every test the diff changes, determine whether the
    assertion is now weaker than before; a test that would pass when the code
    is wrong is blocking."""

#: For a change to an existing repository: the methods no reviewer applies.
VERIFICATION_METHODS = (
    _METHODS_HEADER + _REVIEW_FORK_NOTE + "\n"
    + M1_METHOD + "\n"
    + M2_METHOD + "\n"
    + M13_METHOD + "\n"
    + M15_METHOD + "\n"
)

#: For a new artifact, whose reviewers do not apply M3, M11 or M12, and for fix_pack.
FULL_VERIFICATION_METHODS = (
    _METHODS_HEADER + "\n"
    + M1_METHOD + "\n"
    + M2_METHOD + "\n"
    + M3_METHOD + "\n"
    + M11_METHOD + "\n"
    + M12_METHOD + "\n"
    + M13_METHOD + "\n"
    + M15_METHOD + "\n"
)


def verification_methods(deliverable_kind: str) -> str:
    """The methods a review-fork verifier applies, by what the run delivers."""
    return VERIFICATION_METHODS if deliverable_kind == "repository_change" else FULL_VERIFICATION_METHODS


_HUNK_REVERT_METHOD = """
## Hunk-revert verification

You have a workspace clone of the attempt.  Use it as a scratch copy for the revert
checks below.  Never modify the builder's workspace or the committed tree — only
this clone.

### Requirements and corrections

The requirements from brief.json and the person's gate corrections are listed below.
For each, identify the implementing hunk(s) from the diff and check them.

{{requirements_and_corrections}}

### Method — for each behaviour-changing requirement or correction above

1. From the diff (changes.diff), identify the hunk(s) that implement it.
2. In the workspace clone, revert the hunk with `git apply --reverse` on a
   patch containing only the overlapping hunks, or manually undo the lines.
3. Run the specific test(s) you judge should catch the revert — not the whole
   suite.  Record the command and its output.
4. Restore the file: `git checkout HEAD -- <file>`.
5. Report:
   - The hunk (file:line range).
   - The test command you ran.
   - Its output (truncated to the relevant part).
   - **CAUGHT** when the test failed after the revert (good).
   - **NOT CAUGHT** when the test still passed after the revert (bad — blocking).

Requirements that are docs-only or test-only (no behaviour change) are reported as
**NOT APPLICABLE** — they need no revert check.

### Time budget
Your time budget is """ + str(DESIGN_VERIFIER_TIMEOUT_S) + """ s (~20 min), with one
retry at 2× (""" + str(DESIGN_VERIFIER_TIMEOUT_S * 2) + """ s) for a total of
""" + str(DESIGN_VERIFIER_TIMEOUT_S * 3) + """ s.  If you run out of time before
checking every requirement, report the unchecked ones as **UNCHECKED** (advisory) —
they do not block.

### Probes
After the hunk-revert checks, write and run 2–3 probes for the brief's edge cases
(on both runtimes when the brief names both).

### Output format in reviews/verification.md

For each requirement, write a section:

#### Requirement: <short id or summary>
- Hunk: <file:line-range>
- Test: <command>
- Output: <relevant excerpt>
- Result: CAUGHT | NOT CAUGHT | NOT APPLICABLE | UNCHECKED

Then the probe results, then the existing verification methods below.
"""

_HUNK_REVERT_JUDGING = """
### Judging
- A hunk implementing a requirement or correction that no test catches when reverted:
  BLOCKING: <path:line>: <requirement> is not caught by any test when its hunk is
  reverted (tested with <test command>)
  This is a finding of kind `check` with severity `high` that blocks.
- A probe that fails is a normal defect finding:
  BLOCKING: <path:line>: <description>
"""


def _verifier_prompt(methods: str) -> str:
    return (
        "Artifact directory (absolute): {{artifact}}\n"
        "Read and run the delivered files at this exact path; do not infer it "
        "from your working directory. {{change}}\n\n"
        "Brief: {{brief}}\n\n"
        + methods +
        "\nFor each finding, report it as:\n"
        "BLOCKING: <file:line>: <description with a runnable counterexample or "
        "a cited file:line chain>\n"
        + MARKER_RULE + "\n\n"
        "design.md is in the run directory. Do NOT read it, or strategy.json, "
        "or any review files. You verify the change against the brief and the "
        "repository only.\n\n"
        "End with a line that is exactly VERDICT: PASS when there are no "
        "blocking findings, or VERDICT: FAIL when there are any."
    )


def _design_verifier_prompt(methods: str) -> str:
    """The design-mode verifier prompt with the hunk-revert method."""
    return (
        "Artifact directory (absolute): {{artifact}}\n"
        "Read and run the delivered files at this exact path; do not infer it "
        "from your working directory. {{change}}\n\n"
        "Brief: {{brief}}\n\n"
        + _HUNK_REVERT_METHOD
        + _HUNK_REVERT_JUDGING
        + methods +
        "\nFor each finding, report it as:\n"
        "BLOCKING: <file:line>: <description with a runnable counterexample or "
        "a cited file:line chain>\n"
        + MARKER_RULE + "\n\n"
        "design.md is in the run directory. Do NOT read it, or strategy.json, "
        "or any review files. You verify the change against the brief and the "
        "repository only.\n\n"
        "End with a line that is exactly VERDICT: PASS when there are no "
        "blocking findings, or VERDICT: FAIL when there are any."
    )


#: The review fork's verifier without hunk-revert (for design_fast, which has no
#: workspace clone, no retry, and a shorter timeout).
VERIFIER_PROMPT = _verifier_prompt("{{verification_methods}}")
#: Design mode's verifier with the hunk-revert method.  Requires a workspace clone,
#: ``DESIGN_VERIFIER_TIMEOUT_S`` and a retry; ``design.py`` uses this one.
DESIGN_VERIFIER_PROMPT = _design_verifier_prompt("{{verification_methods}}")
#: fix_pack's inline verifier, with no reviewers beside it and no prepare fact.
FULL_VERIFIER_PROMPT = _verifier_prompt(FULL_VERIFICATION_METHODS)


def parse_verification_findings(ctx: Any) -> list[str]:
    """Parse BLOCKING findings and VERDICT from the verifier output."""
    from refactory_dsh.create_ops import PASS_MARK, blocking_findings

    text = ctx.read_text("reviews/verification.md") if ctx.exists(
        "reviews/verification.md") else ""
    if not text.strip():
        return ["reviews/verification.md is empty or missing"]
    findings = list(blocking_findings(text))
    verdict_line = text.strip().splitlines()[-1].strip()
    if verdict_line != PASS_MARK:
        findings.append(
            f"the verifier did not state {PASS_MARK!r}; "
            f"its last line is: {verdict_line!r}"
        )
    return findings


def gate_verification_generic(ctx: Any) -> str:
    """Advisory verification gate for modes without dedicated review budgets.

    Always proceeds: writes unresolved findings when present but never
    blocks or spends attempt counters.  Used by fix_pack and any future
    mode that uses repo_change but has no review stage.
    """
    findings = parse_verification_findings(ctx)
    if not findings:
        ctx.write_text("reviews/verification-gate.txt", "proceed: no blocking finding\n")
        return "proceed"
    feedback = "\n".join(findings)
    ctx.write_text("reviews/verification-gate.txt", feedback + "\n")
    ctx.write_text("reviews/verification-unresolved.md", feedback + "\n")
    return "proceed"


def wire_verification(graph: Any, *, check_node: str, proceed_target: str,
                      evaluator: str = "") -> None:
    """Add verifier and gate_verification nodes for custom wiring (fix_pack).

    Full-mode and fast-mode graphs add the verifier directly to the review fork;
    only the custom form (``check_node`` and ``proceed_target`` required) remains::

      check_node →(proceed) verifier → gate_verification →(proceed) proceed_target
                                                            →(reloop) emit
    """
    from refactory_dsh.nn import Node

    OPS_GENERIC = "refactory_dsh.verification"

    src = check_node
    dst = proceed_target
    gate_eval = evaluator or f"{OPS_GENERIC}:gate_verification_generic"
    src_condition = "proceed"

    # Inline on the main path, a verifier that ran out of time would fail the run, so
    # this one keeps the route's envelope rather than the review fork's bound.
    graph.add(Node(
        "verifier", "agent",
        prompt_template=FULL_VERIFIER_PROMPT,
        reads=("brief.json", "test-guardedness.json", "changes.diff"),
        writes=("reviews/verification.md",),
        produces=("verification",),
        role="code_reviewer",
        resource_role="final_validation",
        notes="design-blind verification: apply the evidence log's methods to the brief, diff and repo",
    ))

    graph.add(Node(
        "gate_verification", "gate",
        evaluator=gate_eval,
        resource_role="final_validation",
        reads=("reviews/verification.md",),
        writes=("reviews/verification-gate.txt", "reviews/verification-unresolved.md"),
        produces=("verification_gate",),
        max_iterations=2,
        notes="advisory verification gate; always proceeds",
    ))

    # Remove the existing edge — raise when it does not exist.
    old_edges = [e for e in graph.edges
                 if e.source == src and e.target == dst
                 and e.condition == src_condition]
    if not old_edges:
        raise ValueError(
            f"wire_verification: no edge {src!r} ->({src_condition!r}) {dst!r} "
            f"to replace; existing edges: "
            + ", ".join(f"{e.source}->{e.target}({e.condition})" for e in graph.edges)
        )
    graph.edges = [e for e in graph.edges
                   if not (e.source == src and e.target == dst
                           and e.condition == src_condition)]

    graph.connect(src, "verifier", condition=src_condition)
    graph.connect("verifier", "gate_verification")
    graph.connect("gate_verification", dst, condition="proceed")
    graph.connect("gate_verification", "emit", condition="reloop")
