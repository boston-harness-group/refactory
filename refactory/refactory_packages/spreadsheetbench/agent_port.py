"""SkillOpt's SpreadsheetBench agent harness (``claude_code_exec``), ported line for line.

Copied from microsoft/SkillOpt (MIT License, Copyright (c) 2026 Microsoft Corporation),
the commit the solver's port copies (`skillopt_port`, ``ec9d021``; these files are the
same at the public ``4fa8396``): ``skillopt/envs/spreadsheetbench/codegen_agent.py`` (the
workspace and its files) and ``skillopt/model/codex_harness.py`` with
``skillopt/model/backend_config.py`` (how the ``claude`` CLI is started). Each piece names
the lines it copies. SkillOpt runs this harness when its target backend is
``claude_code_exec`` and its SpreadsheetBench mode is ``single`` (adapter.py:70-76 refuses
any other mode); the paper's "Claude Code" harness (arXiv 2605.23904, §4).

One call per item (codegen_agent.py:466-504): the workspace holds ``input.xlsx``,
``.agents/skills/skillopt-target/SKILL.md``, ``task.md`` and ``run_solution.py``; Claude
Code works there with ``Read`` and ``Bash`` and writes ``solution.py``; the item's code is
that file, else the first fenced block of the final message; it is then run and graded
as the chat path's code is (rollout.py:757-863). No gold feedback reaches the target:
``run_single`` takes none (codegen_agent.py:443-454).

What is not SkillOpt's is said where it is decided: the workspace is sealed (`seal`,
`workspace`; SkillOpt's is a folder of the run, its tools unconfined), and ``python``
there is the pinned interpreter model code runs under (`interpreter`; SkillOpt's is
whatever the agent's shell finds).
"""
from __future__ import annotations

from typing import Any

from . import skillopt_port as port

# ── the workspace files ─────────────────────────────────────────────────────

#: Where SkillOpt puts the skill in the workspace (codex_harness.py:111-114).
SKILL_PATH = ".agents/skills/skillopt-target/SKILL.md"
#: The task file (codex_harness.py:101, ``task_filename``).
TASK_PATH = "task.md"
#: The validation driver (codegen_agent.py:354).
DRIVER_PATH = "run_solution.py"
#: The input's name in the workspace (codegen_agent.py:355).
INPUT_PATH = "input.xlsx"
#: What the agent writes, and SkillOpt reads back (codegen_agent.py:491-492).
SOLUTION_PATH = "solution.py"


# codex_harness.py:48-73 ``render_skill_md``
def render_skill_md(
    skill_content: str,
    *,
    name: str = "skillopt-target",
    description: str = "Dynamic ReflACT skill for the current benchmark task.",
    preamble: str = "",
) -> str:
    body = skill_content.strip() or "No additional dynamic guidance was provided for this task."
    chunks = [
        "---",
        f'name: "{name}"',
        f'description: "{description}"',
        "---",
        "",
        "# ReflACT Target Skill",
        "",
    ]
    if preamble.strip():
        chunks.append(preamble.strip())
        chunks.append("")
    chunks.extend([
        "## Dynamic Guidance",
        "",
        body,
        "",
    ])
    return "\n".join(chunks)


# codegen_agent.py:214-225 ``_build_codex_skill``
def build_codex_skill(skill_content: Any) -> str:
    """SKILL.md for ``skill_content``."""
    return render_skill_md(
        str(skill_content or ""),
        description="Dynamic ReflACT skill for solving the current SpreadsheetBench task.",
        preamble=(
            "Use this skill when solving the current SpreadsheetBench task in this workspace.\n"
            "Write a single self-contained Python solution to `solution.py`.\n"
            "The solution must operate on the provided `INPUT_PATH` and `OUTPUT_PATH` variables.\n"
            "You may inspect `input.xlsx` and run `python run_solution.py` to validate locally,\n"
            "but do not hardcode values from the preview or from one specific workbook."
        ),
    )


# codegen_agent.py:228-256 ``_build_codex_task``, with the diagnostic branches off, as
# ``build_user`` has them (the trainer never passes ``diagnostic_mode``).
def build_codex_task(instruction: str, input_xlsx: str, instruction_type: str = "",
                     answer_position: str = "") -> str:
    """task.md: the chat path's first user message, then the harness's own section."""
    prompt = port.build_user(instruction, input_xlsx, instruction_type, answer_position)
    return f"{prompt}\n\n{CODEX_TASK_SECTION}"


#: codegen_agent.py:248-256: the harness's own section of task.md, after the task's prompt.
CODEX_TASK_SECTION = (
    "## Codex Harness Task\n"
    "- Read `.agents/skills/skillopt-target/SKILL.md` before writing code; do not call a Skill tool.\n"
    "- Read and optionally inspect `input.xlsx` in this workspace.\n"
    "- Write the final Python solution to `solution.py`.\n"
    "- The script should use the provided `INPUT_PATH` and `OUTPUT_PATH` variables.\n"
    "- If you want to validate locally, run `python run_solution.py`.\n"
    "- Do not return a code fence as the primary artifact; the source of truth is `solution.py`.\n"
)


# codegen_agent.py:259-316 ``_build_codex_driver``
RUN_SOLUTION = (
    "import os\n"
    "import pathlib\n"
    "import re\n"
    "import shutil\n"
    "import subprocess\n"
    "import sys\n"
    "import tempfile\n\n"
    'INPUT_PATH = "input.xlsx"\n'
    'OUTPUT_PATH = "output.xlsx"\n'
    "code = pathlib.Path('solution.py').read_text(encoding='utf-8')\n"
    "code = re.sub(r'^\\s*(INPUT_PATH|OUTPUT_PATH)\\s*=\\s*.+$', '', code, flags=re.MULTILINE)\n"
    "# Write patched code to a temporary file and run it in a clean subprocess\n"
    "# with a scrubbed environment. This avoids in-process exec/compile but is\n"
    "# not a filesystem, process, or network sandbox.\n"
    "_work_dir = str(pathlib.Path.cwd())\n"
    "_temp_dir = tempfile.mkdtemp(prefix='skillopt-generated-')\n"
    "try:\n"
    "    _patched = pathlib.Path(_temp_dir) / 'runner.py'\n"
    "    _safe_env = {\n"
    "        'PATH': os.environ.get('PATH') or os.defpath,\n"
    "        'HOME': _work_dir,\n"
    "        'TMPDIR': _temp_dir,\n"
    "    }\n"
    "    for _key in (\n"
    "        'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV',\n"
    "        'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH',\n"
    "        'LANG', 'LANGUAGE', 'LC_ALL', 'LC_CTYPE',\n"
    "        'PYTHONIOENCODING', 'PYTHONUTF8',\n"
    "        'SYSTEMDRIVE', 'PATHEXT', 'COMSPEC',\n"
    "    ):\n"
    "        if os.environ.get(_key):\n"
    "            _safe_env[_key] = os.environ[_key]\n"
    "    if os.name == 'nt':\n"
    "        _safe_env['SYSTEMROOT'] = (\n"
    "            os.environ.get('SYSTEMROOT')\n"
    "            or os.environ.get('SystemRoot')\n"
    "            or os.environ.get('WINDIR', '')\n"
    "        )\n"
    "        _safe_env['USERPROFILE'] = _work_dir\n"
    "        _safe_env['TEMP'] = _temp_dir\n"
    "        _safe_env['TMP'] = _temp_dir\n"
    "        _safe_env['APPDATA'] = _temp_dir\n"
    "        _safe_env['LOCALAPPDATA'] = _temp_dir\n"
    "    _safe_env = {k: v for k, v in _safe_env.items() if v}\n"
    "    _patched.write_text(\n"
    "        f'INPUT_PATH = {INPUT_PATH!r}\\nOUTPUT_PATH = {OUTPUT_PATH!r}\\n' + code,\n"
    "        encoding='utf-8',\n"
    "    )\n"
    "    _res = subprocess.run([sys.executable, str(_patched)], capture_output=True, text=True, env=_safe_env)\n"
    "finally:\n"
    "    shutil.rmtree(_temp_dir, ignore_errors=True)\n"
    "if _res.returncode != 0:\n"
    "    print(_res.stdout, end='')\n"
    "    print(_res.stderr, end='')\n"
    "    sys.exit(2)\n"
)


# codegen_agent.py:344-349, the prompt ``_prepare_codex_workspace`` returns
AGENT_PROMPT = (
    "Read `.agents/skills/skillopt-target/SKILL.md` directly; do not call a Skill tool.\n"
    "Read `task.md`, inspect `input.xlsx` if useful, and write the final solution to `solution.py`.\n"
    "You may run `python run_solution.py` to validate the script locally.\n"
    "In your final response, briefly confirm whether `solution.py` was written and summarize the approach."
)


# ── how the CLI is started: codex_harness.py ────────────────────────────────

# codex_harness.py:729-742 ``_normalize_target_exec_prompt``
def normalize_target_exec_prompt(prompt: str) -> str:
    """Avoid wording that makes Claude Code call an unregistered Skill tool."""
    text = prompt or ""
    replacements = {
        "Use the `skillopt-target` skill available in this workspace.": (
            "Read `.agents/skills/skillopt-target/SKILL.md` directly; do not call a Skill tool."
        ),
        "- Use the local `skillopt-target` skill before writing code.": (
            "- Read `.agents/skills/skillopt-target/SKILL.md` before writing code; do not call a Skill tool."
        ),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


# codex_harness.py:701-715 ``_exec_prompt``
def exec_prompt(prompt: str, *, allow_file_edits: bool = False) -> str:
    """What the CLI is given after ``--``: SkillOpt's fixed lead, then the prompt."""
    edit_instruction = (
        "You may modify files in the workspace when the task asks you to create an artifact. "
        if allow_file_edits
        else "Do not modify files. "
    )
    return (
        "Use the workspace files to solve the task. Read task.md and the skill at "
        ".agents/skills/skillopt-target/SKILL.md before answering. "
        "If ATTACHMENTS.md exists, read it and inspect the listed local files. "
        "Do not call a Skill tool; the ReflACT guidance is a local markdown file. "
        f"Do not ask for permission. {edit_instruction}"
        "Return only the final answer text, keeping any required <answer>...</answer> tags exactly.\n\n"
        f"{normalize_target_exec_prompt(prompt)}"
    )


#: The solve step's whole prompt: ``_run_exec_backend`` passes ``allow_file_edits=True``
#: (codegen_agent.py:361-376) to ``run_claude_code_exec``, whose CLI path sends
#: ``_exec_prompt`` of the workspace prompt (codex_harness.py:928).
SOLVE_PROMPT = exec_prompt(AGENT_PROMPT, allow_file_edits=True)

#: backend_config.py:66-69, :132-136 and configs/_base_/default.yaml:21-25: the defaults
#: ``get_claude_code_exec_config`` returns (effort ``medium``; the trainer passes the
#: config's ``claude_code_exec_effort``, trainer.py:787-793), and one retry on an empty
#: response (``EXEC_EMPTY_RESPONSE_RETRIES``).
EFFORT = "medium"
MAX_THINKING_TOKENS = 16384
EMPTY_RESPONSE_RETRIES = 1
#: codex_harness.py:897: ``Read,Bash`` when the caller names no tools, as
#: ``_run_exec_backend`` does not.
TOOLS = ("Read", "Bash")
#: codex_harness.py:904: the permission mode when the caller names none.
PERMISSION_MODE = "bypassPermissions"


# codex_harness.py:884-928 ``_run_claude_code_cli_exec``: the argv, for the parity check
def cli_argv(*, path: str, work_dir: str, model: str, prompt: str, effort: str = EFFORT,
             max_thinking_tokens: int = MAX_THINKING_TOKENS, profile: str = "") -> list[str]:
    """The ``claude`` command line SkillOpt runs for one attempt (``cwd=work_dir``), with
    ``allowed_tools`` and ``permission_mode`` unset, ``data_dirs`` and ``images`` empty,
    and ``allow_file_edits=True``, as ``_run_exec_backend`` calls it."""
    tools = ",".join(TOOLS)
    cmd = [
        path,
        "-p",
        "--output-format",
        "text",
        "--permission-mode",
        PERMISSION_MODE,
        "--add-dir",
        work_dir,
        "--tools",
        tools,
        "--allowedTools",
        tools,
    ]
    if profile:
        cmd.extend(["--settings", '{"env":{"CLAUDE_CODE_USE_BEDROCK":"0"}}'])
        cmd.extend(["--append-system-prompt", f"Profile: {profile}"])
    if model:
        cmd.extend(["--model", model])
    if effort:
        cmd.extend(["--effort", effort])
    if max_thinking_tokens > 0:
        cmd.extend(["--max-thinking-tokens", str(max_thinking_tokens)])
    cmd.extend(["--", exec_prompt(prompt, allow_file_edits=True)])
    return cmd


# ── after the call: codegen_agent.py:491-504 ────────────────────────────────

def item_code(solution: str | None, final_message: str) -> str:
    """codegen_agent.py:491-496: ``solution.py`` when the agent left one, else the first
    fenced block of its final message (the whole message when it has none)."""
    if solution is not None:
        return solution
    return port.extract_code(final_message)


def conversation(final_message: str) -> list[dict[str, Any]]:
    """codegen_agent.py:501: what SkillOpt saves as the item's conversation: the final
    message alone. Its analysts see this, the verification block appended
    (rollout.py:846-858), and the chat path's prompts (rollout.py:700-718), never the
    agent's tool calls (the CLI path's raw output is its final text, so
    ``format_claude_trace_steps`` finds no steps, codex_harness.py:526-600) nor its
    ``solution.py``."""
    return [{"role": "assistant", "content": final_message}]
