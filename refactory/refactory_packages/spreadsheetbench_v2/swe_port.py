"""SpreadsheetBench 2's own agent setup (its SWE-agent config), ported to a sealed Claude Code step.

Copied from SpreadsheetBench 2's repository (``SWE-agent/``, MIT License, Copyright (c)
2024 John Yang, Carlos E. Jimenez, Alexander Wettig, Shunyu Yao, Karthik Narasimhan, Ofir
Press): ``config/spreadsheet.yaml`` (sha256 ``749fed4e…5209``), ``tools/view_xlsx/bin/view_xlsx``
(``0a920569…8f33``, shipped byte for byte as ``swe_tools/view_xlsx``), ``tools/submit`` and
``spreadsheet.Dockerfile`` (``8b77ecac…7913``). ``tests/fixtures/
spreadsheetbench_v2_swe_upstream.json`` holds those texts as ``yaml.safe_load`` reads them,
and the tests rebuild this module's prompt from them.

Upstream, one SWE-agent run per task: the system template (spreadsheet.yaml:4-6), then the
instance template (:8-60) filled with the task's instruction, input and output paths; the
model calls ``bash``, ``view_xlsx`` and ``submit`` as functions (:97-105), at most 50
model calls (:111), each command cut at 60 s (:99), in a Docker image with Python 3.11,
pandas, openpyxl, numpy, matplotlib, xlsxwriter and LibreOffice (spreadsheet.Dockerfile).
The task's output file is graded.

Here, one sealed Claude Code session (`swe_mode.Swe`) with ``Read`` and ``Bash`` in a
workspace holding ``input.xlsx``; ``view_xlsx``, ``soffice`` and ``libreoffice`` are on its
PATH (`tasks.SpreadsheetBenchV2SweTask`); it edits the workbook and writes ``output.xlsx``,
which V2's grade grades.

**Deviations from upstream**, each needed by this runtime (the prompt's are
`SUBSTITUTIONS`, applied to upstream's instance template in this order):

1. *Tools* (spreadsheet.yaml:16-20, :97-105): Claude Code's ``Bash`` (upstream ``bash``) and
   ``Read`` (upstream has none: the agent mode's tool pair, which this runtime's sealed
   Claude Code step offers) are its tools; ``view_xlsx`` is a command on the PATH run
   through ``Bash`` (upstream: a function the model calls; same script, same arguments);
   there is no ``submit``: the session ends when the agent ends its turn, so "submit"
   means that (:20, :45). The tool list's count stays "four" (upstream lists three under
   "four"; here ``Bash``, ``Read``, ``view_xlsx`` and ``submit``).
2. *Paths* (:30-32, :57-58): ``{{spreadsheet_path}}`` is ``input.xlsx`` and
   ``{{output_path}}`` is ``output.xlsx``, in the session's working directory (upstream:
   the task's paths in its container).
3. *System prompt* (:4-6): upstream's system template, appended to Claude Code's own
   system prompt (the step's ``system_prompt_template``, as a route's appended system
   prompt); upstream it is the whole system message.
4. *Call cap* (:111 ``per_instance_call_limit: 50``): Claude Code's turn cap of 50
   (``--max-turns``, the route's ``max_turns``). Reaching it ends the session; whatever
   ``output.xlsx`` it left is graded (upstream: SWE-agent's autosubmission on exit).
5. *No per-command 60 s timeout* (:99-100): the sealed step's own limits apply (each
   command's CPU time, the inactivity limit), as in the agent mode.
6. *Skill* (none upstream): the ``skill`` knob, written to ``SKILL.md``; when it is
   non-empty a "## Skill" section before ``</task_context>`` tells the agent to read it
   (`skill_section`). Empty (the default), the prompt is upstream's with the
   substitutions above and nothing else, and ``SKILL.md`` is an empty file (the sealed
   step reads every file it declares).
7. *Environment* (spreadsheet.Dockerfile): ``python3`` is the pinned interpreter
   (Python 3.12 with openpyxl, pandas and numpy; no matplotlib, xlsxwriter or flake8;
   upstream: Python 3.11.10 with those too), LibreOffice is the grader's own (`recalc.soffice`)
   as ``soffice`` and ``libreoffice``, with the session's own writable ``/tmp`` (its pipe
   lives there) and its profile in the session's HOME, the workspace; ``import uno`` is not
   on ``python3``'s path (upstream adds LibreOffice's program folder to ``PYTHONPATH``).
   No network.
8. *One tool call per response* (:17) is the prompt's instruction only: Claude Code may
   still make several calls in one response.
"""
from __future__ import annotations

from pathlib import Path

#: The input and the output as the prompt names them, in the session's working directory.
INPUT_PATH = "input.xlsx"
OUTPUT_PATH = "output.xlsx"
#: The skill file in the workspace (`skill_section`).
SKILL_PATH = "SKILL.md"
#: Upstream's call cap (spreadsheet.yaml:111 ``per_instance_call_limit``), Claude Code's
#: turn cap here.
MAX_TURNS = 50
#: The tools the session is offered (deviation 1).
TOOLS = ("Read", "Bash")

#: spreadsheet.yaml:4-6, the system template, verbatim.
SYSTEM_TEMPLATE = (
    "You are a helpful Spreadsheet Automation Engineer that interacts with a computer shell "
    "to solve data tasks.\n"
    "You operate in a REPL (Read-Eval-Print Loop) environment where you must issue exactly "
    "ONE command at a time."
)

#: spreadsheet.yaml:8-60, the instance template as ``yaml.safe_load`` reads it, verbatim.
UPSTREAM_INSTANCE_TEMPLATE = (
    "<task_context>\n"
    "\n"
    "## Important\n"
    "- When completing spreadsheet tasks, strictly avoid altering any cells that already "
    "contain values unless explicitly instructed. Modify only the cells that are required "
    "for the task.\n"
    "- You need to complete the instructions and ensure that the original formatting is "
    "preserved as much as possible.\n"
    "\n"
    "## Tools\n"
    "You are provided with four tools: `bash`, `view_xlsx` and `submit`. You must use these "
    "tools to complete the target task.\n"
    "**Important: You can only call ONE tool at a time per response.**\n"
    "- `bash`: run shell commands (e.g., file operations, calling Python scripts with "
    "`python3`)\n"
    "- `view_xlsx`: inspect `.xlsx` files (list sheets, view sheet contents, see original "
    "formulas and values)\n"
    "- `submit`: finalize and submit your solution after verification\n"
    "\n"
    "## Environment\n"
    "You can complete the corresponding spreadsheet tasks using the following Python "
    "environment:\n"
    "`Pandas`, `openpyxl`, `numpy` and `LibreOffice`\n"
    "You can use these environments to manipulate the spreadsheet data.\n"
    "\n"
    "## Recommended Workflow\n"
    "1. **Inspect**: Use `view_xlsx` to inspect the data. This will allow you to understand "
    "the structure and contents of the spreadsheet.\n"
    "   - You must choose mode `list` or `content` to inspect the data.\n"
    "   - List all sheets: `view_xlsx {{spreadsheet_path}} list`\n"
    "   - View specific sheet: `view_xlsx {{spreadsheet_path}} content \"Sheet1\"`\n"
    "   - View row range: `view_xlsx {{spreadsheet_path}} content \"Sheet1\" 1 50`\n"
    "\n"
    "2. **Plan**: Decide how to manipulate the data using `openpyxl` or `LibreOffice`.\n"
    "\n"
    "3. **Implement**: Create a python script (e.g., `process_data.py`) that performs the "
    "edit.\n"
    "   - You can use `bash` tool to write Python code directly, e.g., cat > /tmp/script.py "
    "<< 'EOF' ... EOF), which can then be used to verify syntax correctness.\n"
    "\n"
    "4. **Execute**: Run the python script using `python3 /tmp/script.py`\n"
    "\n"
    "5. **Verify**: Check the output file exists, ensure the answer is correct.\n"
    "   - Do NOT combine verify and submit commands (e.g., `python3 script.py && submit`). "
    "Run them separately to ensure verification is complete before submission.\n"
    "   - Ensure the existing structure, layout, and formatting of the model are preserved "
    "throughout the process.\n"
    "\n"
    "6. **Submit**: When verification is successful, run `submit`.\n"
    "\n"
    "</task_context>\n"
    "\n"
    "<instructions>\n"
    "\n"
    "## Task instructions\n"
    "You need to process a spreadsheet file based on specific instructions.\n"
    "**Instruction:** {{instruction}}\n"
    "\n"
    "## Input and Output\n"
    "You are provided with an input spreadsheet file and you need to generate an output "
    "spreadsheet file based on the output_path.\n"
    "**Input File:** {{spreadsheet_path}}\n"
    "**Output Path:** {{output_path}}\n"
    "\n"
    "</instructions>"
)

#: The prompt's substitutions, ``(upstream text, this runtime's, why)``, each found once in
#: upstream's instance template and applied in order (the module docstring's deviations 1
#: and 6; deviation 2 is the placeholders' values, `prompt_facts`).
SUBSTITUTIONS: tuple[tuple[str, str, str], ...] = (
    ("You are provided with four tools: `bash`, `view_xlsx` and `submit`.",
     "You are provided with four tools: `Bash`, `Read`, `view_xlsx` and `submit`.",
     "Claude Code's Bash and Read; submit is ending the turn (spreadsheet.yaml:16)"),
    ("- `bash`: run shell commands",
     "- `Bash`: run shell commands",
     "Claude Code's Bash (spreadsheet.yaml:18)"),
    ("formulas and values)\n",
     "formulas and values); a command on your PATH: run it with `Bash`\n"
     "- `Read`: read a file in the working directory\n",
     "view_xlsx is a command, not a function; Read is offered (spreadsheet.yaml:19)"),
    ("- `submit`: finalize and submit your solution after verification\n",
     "- `submit`: finalize and submit your solution after verification; to submit, end your "
     "turn with a short final reply (there is no `submit` command)\n",
     "no submit tool: the session ends with the agent's turn (spreadsheet.yaml:20)"),
    ("You can use `bash` tool",
     "You can use `Bash` tool",
     "Claude Code's Bash (spreadsheet.yaml:37)"),
    ("6. **Submit**: When verification is successful, run `submit`.\n",
     "6. **Submit**: When verification is successful, submit: end your turn with a short "
     "final reply.\n",
     "no submit tool (spreadsheet.yaml:45)"),
    ("\n</task_context>",
     "\n{{skill_section}}</task_context>",
     "the skill's section, empty without a skill (none upstream; spreadsheet.yaml:47)"),
)


def ported(template: str) -> str:
    """``template`` with `SUBSTITUTIONS` applied, each required to occur exactly once."""
    for old, new, _why in SUBSTITUTIONS:
        if template.count(old) != 1:
            raise ValueError(f"upstream's template holds {old!r} {template.count(old)} times")
        template = template.replace(old, new)
    return template


#: The solve step's prompt: upstream's instance template with `SUBSTITUTIONS`; its
#: ``{{...}}`` slots are filled from the item's facts (`prompt_facts`).
INSTANCE_TEMPLATE = ported(UPSTREAM_INSTANCE_TEMPLATE)


def skill_section(skill: str) -> str:
    """The prompt's skill section: none for an empty skill, so the prompt is upstream's."""
    if not str(skill or "").strip():
        return ""
    return (f"## Skill\nBefore you start, read `{SKILL_PATH}` in the working directory: it holds "
            "guidance for these spreadsheet tasks.\n\n")


def prompt_facts(instruction: str, skill: str) -> dict[str, str]:
    """The values of `INSTANCE_TEMPLATE`'s slots for one item."""
    return {"instruction": str(instruction), "spreadsheet_path": INPUT_PATH,
            "output_path": OUTPUT_PATH, "skill_section": skill_section(skill)}


def render(instruction: str, skill: str = "") -> str:
    """The prompt one item's session is sent, as the step renders `INSTANCE_TEMPLATE`
    (`nn.executor.render`: each ``{{name}}`` once, never what a value holds)."""
    import re

    values = prompt_facts(instruction, skill)
    return re.sub(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}",
                  lambda m: values.get(m.group(1), m.group(0)), INSTANCE_TEMPLATE)


#: tools/view_xlsx/bin/view_xlsx, byte for byte (its ``#!/usr/bin/env python3`` runs the
#: pinned interpreter, first on the sealed PATH).
VIEW_XLSX = (Path(__file__).with_name("swe_tools") / "view_xlsx").read_text(encoding="utf-8")
