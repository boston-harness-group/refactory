"""The steps mode's prompts: V2's instance template split at its "Recommended Workflow".

The swe mode sends one session V2's whole instance template (`swe_port.INSTANCE_TEMPLATE`,
upstream's with the swe substitutions), whose "## Recommended Workflow" lists six stages in
prose: 1 Inspect, 2 Plan, 3 Implement, 4 Execute, 5 Verify, 6 Submit
(spreadsheet.yaml:27-45). The steps mode runs four sealed sessions in a row, one per stage
group, and each is sent that same template with the workflow cut to its own stages and a
short section saying what it reads and writes:

- ``inspect``: stage 1 Inspect (:28-32); reads ``input.xlsx``, writes ``inspect_notes.md``;
- ``plan``: stage 2 Plan (:34); also reads ``inspect_notes.md``, writes ``plan.md``;
- ``implement``: stages 3 Implement and 4 Execute (:36-39); also reads ``plan.md``, writes
  ``output.xlsx``;
- ``verify``: stages 5 Verify and 6 Submit (:41-45); also reads ``output.xlsx``, writes
  ``verify_report.md`` and may rewrite ``output.xlsx``.

Every step's prompt is, byte for byte, the swe prompt with its workflow section's body
replaced by the step's stages (`WORKFLOW`, slices of that body which, in order, make it
whole) followed by the step's section (`STEP_SECTIONS`, the only new text). So each step
keeps the shared parts unchanged: "Important" (:11-13), "Tools" (:15-20, with the swe
substitutions: `Bash`, ``view_xlsx`` a command, submit = ending the turn), "Environment"
(:22-25), the skill section when a skill is set (`swe_port.skill_section`: every step reads
the one ``SKILL.md``), "Task instructions" and "Input and Output" (:51-58), with the item's
instruction and paths filled from prepare's facts (``{{instruction}}``,
``{{spreadsheet_path}}``, ``{{output_path}}``, ``{{skill_section}}``). The system prompt
appended to Claude Code's is V2's system template (:4-6) for every step.

Each text is the step's own open knob (``<step>__task``, ``<step>__system_prompt``), so the
loop edits one step's instructions without touching the others. With the defaults the
pipeline is a starting point that differs from upstream (four sessions, each with its own
turn cap, instead of one): its baseline is to be measured, not assumed equal to swe's.
"""
from __future__ import annotations

import re

from . import swe_port

#: The four steps, in order.
STEPS = ("inspect", "plan", "implement", "verify")
#: The files the steps hand on, in the session's working directory.
INSPECT_NOTES = "inspect_notes.md"
PLAN = "plan.md"
VERIFY_REPORT = "verify_report.md"

#: The workflow section's heading in the swe prompt (spreadsheet.yaml:27).
WORKFLOW_HEADING = "## Recommended Workflow\n"
#: Where the workflow's body ends: the skill section's slot before ``</task_context>``.
WORKFLOW_END = "{{skill_section}}</task_context>"
#: Where each step's stages start in the workflow's body (the first step's at its start).
STAGE_STARTS = {"inspect": "1. **Inspect**", "plan": "2. **Plan**",
                "implement": "3. **Implement**", "verify": "5. **Verify**"}


def _cut(template: str) -> tuple[str, str, str]:
    """``template`` as (everything to the workflow's body, the body, the rest)."""
    if template.count(WORKFLOW_HEADING) != 1 or template.count(WORKFLOW_END) != 1:
        raise ValueError("the template has no single workflow section ending at the skill slot")
    start = template.index(WORKFLOW_HEADING) + len(WORKFLOW_HEADING)
    end = template.index(WORKFLOW_END)
    return template[:start], template[start:end], template[end:]


HEAD, WORKFLOW_BODY, TAIL = _cut(swe_port.INSTANCE_TEMPLATE)


def _stages(body: str) -> dict[str, str]:
    """``body`` cut where each step's first stage starts: each step's stages, verbatim."""
    starts = []
    for step in STEPS:
        marker = STAGE_STARTS[step]
        if body.count(marker) != 1:
            raise ValueError(f"the workflow holds {marker!r} {body.count(marker)} times")
        starts.append(body.index(marker))
    if starts[0] != 0 or starts != sorted(starts):
        raise ValueError("the workflow's stages are not in the steps' order")
    ends = [*starts[1:], len(body)]
    return {step: body[a:b] for step, a, b in zip(STEPS, starts, ends)}


#: Each step's stages: slices of the swe prompt's workflow body, in order its whole.
WORKFLOW = _stages(WORKFLOW_BODY)

#: Each step's own section after its stages: what it reads, writes and leaves to the next
#: step. The only text not from upstream's template.
STEP_SECTIONS = {
    "inspect": (
        "## This Step\n"
        "This is step 1 of 4 (Inspect). Later steps plan, implement and verify the edit, "
        "each in a new session that starts from the files named here. Inspect the input and "
        f"write your notes to `{INSPECT_NOTES}` in the working directory: the workbook's "
        "structure, the sheets and cells relevant to the instruction, and the conventions you "
        "see (formulas, formats, layout). Do not write {{output_path}}: a later step does. "
        "When the notes are written, end your turn with a short final reply.\n\n"),
    "plan": (
        "## This Step\n"
        f"This is step 2 of 4 (Plan). The Inspect step's notes are in `{INSPECT_NOTES}`: read "
        "them first (you may also view {{spreadsheet_path}}). Write your plan to "
        f"`{PLAN}` in the working directory: which cells to change and how. Do not write "
        "{{output_path}}: the next step implements your plan. When the plan is written, end "
        "your turn with a short final reply.\n\n"),
    "implement": (
        "## This Step\n"
        f"This is step 3 of 4 (Implement and Execute). Read `{INSPECT_NOTES}` (the Inspect "
        f"step's notes) and `{PLAN}` (the Plan step's plan), then write {{{{output_path}}}} "
        "from {{spreadsheet_path}} as planned. The next step verifies it. When "
        "{{output_path}} is written, end your turn with a short final reply.\n\n"),
    "verify": (
        "## This Step\n"
        "This is step 4 of 4 (Verify and Submit). {{output_path}} was written by the "
        f"Implement step from `{PLAN}` (the Plan step's plan) and `{INSPECT_NOTES}` (the "
        "Inspect step's notes). Check it against the instruction and {{spreadsheet_path}}, "
        "recalculating with LibreOffice as needed, and correct {{output_path}} where it is "
        "wrong: it is graded as you leave it. Write what you checked and found to "
        f"`{VERIFY_REPORT}` in the working directory, then submit.\n\n"),
}


def step_task(step: str) -> str:
    """The step's prompt template (its ``<step>__task`` knob's default): the swe prompt with
    the workflow cut to the step's stages and the step's section after them."""
    return HEAD + WORKFLOW[step] + STEP_SECTIONS[step] + TAIL


#: Each step's default prompt template.
TASKS = {step: step_task(step) for step in STEPS}


def render(step: str, instruction: str, skill: str = "", template: str | None = None) -> str:
    """The prompt one item's ``step`` session is sent with the default (or ``template``), as
    the step renders it (`nn.executor.render`: each ``{{name}}`` once)."""
    values = swe_port.prompt_facts(instruction, skill)
    return re.sub(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}",
                  lambda m: values.get(m.group(1), m.group(0)),
                  TASKS[step] if template is None else template)
