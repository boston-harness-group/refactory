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
from refactory_dsh.probe_rerun import PROBE_TIMEOUT_S, PROBES_TOTAL_S

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

M15_METHOD = """M15 Weakened tests — run `git diff <base> -- <test files>` (the base is the
    commit the diff starts from) and list every existing test function the diff
    modifies, by file::name. Compare each one with its base version and list each
    loosened assertion: an assert removed or relaxed (== to in, exact to approx,
    a narrower raises), an expected value changed to match the new behaviour, a
    test renamed or reworded to accept different behaviour, or a skip/xfail added.
    Report a loosened assertion that would let the test pass when the code is
    wrong; it does not block, because an expected value updated to match a
    behaviour the brief asked for looks the same and is the required edit. You
    may not conclude "no weakened tests" without listing the modified test
    functions you compared (write "none modified" when the diff changes no
    existing test function)."""

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

### Edge-case sweep
After the hunk-revert checks, sweep the edge cases. For each requirement in the brief,
probe every class below that applies, and say which classes do not apply and why:

- Failure points: inject each failure at every distinct point in the diff where it
  can occur (e.g. setup, each step/op inside the changed path, verify, the end/close
  of a pass), not only the first point that works.
- Position: the first, a middle, and the LAST element of a sequence or loop; make
  the LAST element fail at each of the failure points above.
- All / none / exactly-one: e.g. every item fails, no item fails, exactly one fails.
- Timeouts and hangs: each place a deadline applies; check the deadline actually
  stops the work and that the timed-out work is not retried.
- Interaction with control flow: a gate, pause, halt, stop or resume occurring
  inside the changed path.
- Concurrency: two branches or items at once, where the change runs in forks or
  threads.
- Both runtimes: whenever the brief names both the walker and the spine, or the
  change touches shared runtime code, run the same probe on the walker AND on the
  spine (through lowering + the TS drain, where the repo has a harness for it) and
  compare the results; a difference is a defect. A runtime you could not run the
  probe on is UNCHECKED for that runtime.

A probe is new code you write and run. Running the builder's existing tests does
not count as a probe (you may run them in addition). PROBED means executed, with its
command and output shown; a check made only by reading code is reported as "READ, not
probed".

Save every probe as its own file under `reviews/probes/` in your workspace, named
`<requirement-id>-<n>.py` (a Python script, or a test file with `test_` functions, run
with pytest) or `<requirement-id>-<n>.sh`, run from the repository root. Its first line
is a comment stating the requirement, the failure point and the expected outcome. First
delete any `reviews/probes/` an earlier attempt left. A probe file must not just call an
existing repository test (pytest on an existing test id, or importing and calling a test
function): it makes its own check. Each PROBED line in reviews/verification.md names its
probe file (`reviews/probes/<file>`). After you finish, the runtime reruns every probe
file on this attempt in a fresh checkout (""" + str(PROBE_TIMEOUT_S) + """ s each,
""" + str(PROBES_TOTAL_S) + """ s in all): a probe that fails is a blocking defect, and a
PROBED line whose file is missing or only calls an existing test is UNVERIFIED.

Each probe states the exact command you ran and its output. If a probe fails
for an environment reason (import path, missing test helper such as loop_ops), fix
the environment (e.g. set PYTHONPATH to the repo's test helpers) and rerun the same
probe; never substitute a weaker probe. A failing probe is a
defect finding. Stay within the time budget above; list every class you did not
probe as **UNCHECKED** with the requirement it belongs to.

### Output format in reviews/verification.md

For each requirement, write a section:

#### Requirement: <short id or summary>
- Hunk: <file:line-range>
- Test: <command>
- Output: <relevant excerpt>
- Result: CAUGHT | NOT CAUGHT | NOT APPLICABLE | UNCHECKED

Then the edge-case sweep: per requirement, each class as PROBED (probe file
`reviews/probes/<file>`, command, output, result), "READ, not probed", NOT APPLICABLE (why) or UNCHECKED. Then the existing verification methods below.
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
