# SpreadsheetBench (`spreadsheetbench/solver`, `spreadsheetbench/agent`, `spreadsheetbench_v2/agent`, `spreadsheetbench_v2/swe`, `spreadsheetbench_v2/steps`)

SkillOpt's SpreadsheetBench target agent, ported so the outer loop can be measured
against SkillOpt on the same items, prompts, code runner and grader. Source:
microsoft/SkillOpt at `ec9d021` (MIT), `skillopt/envs/spreadsheetbench/`; every ported
function names the lines it copies (`refactory_packages/spreadsheetbench/skillopt_port.py`).

Two modes, both SkillOpt's: `spreadsheetbench/solver` is its default harness (the
multi-turn codegen loop, below), and `spreadsheetbench/agent` its Claude Code agent
harness ([below](#the-agent-harness-spreadsheetbenchagent)). They share the data, the
Task, the code runner and the grader. `spreadsheetbench_v2/agent` runs that agent harness
on a second dataset, SpreadsheetBench 2, with V2's own task text and grader
([below](#spreadsheetbench-2-spreadsheetbench_v2agent)); `spreadsheetbench_v2/swe` runs V2's
own agent setup on it ([below](#v2s-own-agent-setup-spreadsheetbench_v2swe)), and
`spreadsheetbench_v2/steps` that setup with its workflow split into graph steps
([below](#v2s-workflow-as-steps-spreadsheetbench_v2steps)).

## The data

SpreadsheetBench Verified 400, Hugging Face `KAKA22/SpreadsheetBench` at revision
`ab0b742b0fc95b946f212d80ac7771b5531272e4`, archive `spreadsheetbench_verified_400.tar.gz`
(sha256 `10ef893d…c949`). The data is not shipped. Set `SB_DATA` to the extracted
`spreadsheetbench_verified_400` directory; its `dataset.json` must match the checksum in
`manifest.json`, which also holds SkillOpt's id split (80 train, 40 val, 280 test).
`SB_MANIFEST` names another split in the same format (a smoke split).

Two test items are known by name:

- **42930 can never pass.** The release pairs `1_42930_init.xlsx` with
  `1_43930_golden.xlsx`, so no test case is found: the item scores 0 with
  `no-test-cases` and no model call, as in SkillOpt. A test pass scores at most 279/280.
- **56225 passes only with a written value.** The release marks it
  `exclude: "ignore, initial file already passes verification"`, but the input does not
  pass: its O3 is 0, the gold's 1.8019. Nor can LibreOffice reproduce the gold's SUMIFS,
  so a formula answer fails under `recalc` too. SkillOpt ignores the field, and so does
  this port.

## The graph

```
data ─▶ prepare ─▶ has_case ─▶ frame ─▶ solve ─▶ run_turn ─▶ gold_check ─▶ next_turn ─▶ turn_gate
                     │           ▲                                                         │
                     │           └──────────────────────── reloop ─────────────────────────┤
                     └── skip ──▶ finalize ◀─────────────────── proceed ────────────────────┘
```

- `prepare` records the item's start and builds SkillOpt's system file plus `## Skill`,
  the first user message (instruction, type, answer position, a 5-row preview of every
  sheet) and the history.
- `frame` flattens the history the way SkillOpt's Claude backend does: one `claude -p`
  prompt per turn, `History:` then `- User:`/`- Assistant:` lines (stripped, continuation
  lines indented two spaces), then `Answer the latest user request.`
  (`refactory_packages.skillopt.claude_backend.claude_backend_messages`).
- `solve` is one model call whose reply is the node's output. It runs as SkillOpt's
  `claude -p` did: Claude Code's own system prompt with SkillOpt's text appended, and
  Claude Code's default tools offered and refused by `dontAsk` (`tool_policy="refused"`,
  [architecture](architecture.md)), in a fresh empty directory, with no settings loaded.
  Its route must be the `claude_code` worker with permission mode `dontAsk`, and its
  Claude Code must report exactly 2.1.299, the lab's install that SkillOpt's runs used (SkillOpt pins none): `refactory optimize` and
  `refactory run` refuse another before they start (the SDK's bundled 2.1.281
  included), so name it with `--claude-executable`.
- `run_turn` waits 3 s, takes the reply's first fenced block (the whole reply when there
  is none, so prose comes back as a SyntaxError, not as "no code"), and runs it with
  SkillOpt's runner. An error is the next user message. A solve call that ended without
  an answer is not a reply, and only a call that answered with nothing gets SkillOpt's
  "No Python code block" message:
  - all 5 sessions crashed (a worker or product failure): the item ends there with no
    answer (score 0, `fail_reason` `no answer: solve crashed in all 5 sessions`, the
    failure under `missing_steps`), as a crashed step does in any mode;
  - it ran out of output (only possible with `max_completion_tokens` set; not retried):
    the same, `no answer: solve reached its output limit without answering`;
  - the worker was refused for its setup (a key, a model name, a Claude Code worker that
    is not logged in): not retried, the item fails with that error (`error_text`
    `solve: CompletionFailed: ...`, on both runtimes); a run whose every item fails so
    fails ("all N items failed").

  Either way the prediction an earlier turn wrote is not graded, as in SkillOpt: a call
  that fails after its retries raises out of `run_multi`, and `process_one_codegen`
  returns `llm-call-failed` before grading (rollout.py:763-767). SkillOpt retries a
  logged-out call as it does any other; here it is not retried, since the next session
  is refused the same way.
- `gold_check` is the train-only gold feedback node (below); `next_turn` ends the item on
  a clean run, unless that node says the output is wrong.
- `turn_gate` loops while turns (30) remain and more than 10 s remain before the harness
  backstop.
- `finalize` fails an empty last code block, runs the last code once more when there is
  no prediction, and records an item past the backstop as the model's own failure
  (`backstop`).

The item's input workbook is its item file (`Task.item_files`): `setup` copies it into
each pass as `item/input.xlsx`, the data node declares it, and `prepare`, `run_turn` and
`finalize` list it in their reads and open it through `ctx.item_file("input")`. A step
the outer loop adds can read it the same way; a sealed tool step it adds in the pass gets
a copy in its workspace by default (unless its `sees` names other inputs), and its commands run the pinned interpreter
as `python3` (`SpreadsheetBenchTask.tool_environment`). Model code still gets it under the input's own name in its sandbox, as
SkillOpt's did.

The grader (`recalc.evaluate`) is SkillOpt's cell comparison on cached values, then, when
the prediction's answer cells hold formulas, the same comparison on a LibreOffice-
recalculated copy; the item passes if either matches. `SB_GRADER=cached` is SkillOpt's
grader alone. LibreOffice is `SB_SOFFICE`, else `soffice` on the PATH, else the macOS app.

## Train-only gold feedback

In SkillOpt's main training rollout (and no other rollout), a clean run is graded and, if
wrong, the model is told which cells are wrong, without their expected values. In
refactory this is a declared library mechanism, `refactory_dsh.feedback`:

- the Task says what an attempt is told (`Task.train_feedback`);
- the outer loop asks for it per evaluation (`Run.evaluate(..., gold_feedback=True)`),
  which it refuses for val and test, and records it in the run's `subset.json`;
- one protected op node calling `refactory_dsh.feedback:train_feedback`, with no reads or
  writes and one fact, asks the Task only on such a train evaluation; elsewhere its fact
  is empty. Rule V17 lets that fact, and nothing else grader-derived, through.

Both arms ask for it, so both get the same information: the SkillOpt strategy on its
train rollout, and the default loop on its own train evaluations (each candidate's train
rollout and the parent's re-score) whenever the Task offers it (`Task.train_feedback`).
Neither asks on val or test, nor on the rollouts that only compare skills (SkillOpt's
slow update, the default loop's memory pass).

**The train score is the first attempt's.** An item told it was wrong scores 0 however
its later turns end; its record keeps the conversation, the feedback included (all but
feedback on a last turn, which no call saw), says
`first attempt wrong (right after gold feedback)` (or `still wrong`) as its fail
reason, and keeps its grade after the feedback under `gold_feedback`. So the turns the
answer's feedback buys reach the analysts but never lift the score that drives either
arm's search. SkillOpt scores the attempt after the feedback; with it the starting skill
passed 75 or 76 of the 76 clean train items, leaving no failures to learn from. "First
attempt" means the first clean run the grader judged: a turn whose code failed to run,
then a right answer, still passes.

## Where model code runs

A pinned interpreter with SkillOpt's versions (Python 3.12, openpyxl 3.1.5, pandas 2.3.3,
numpy 2.5.4 and pandas' own dependencies), built once with
`python -m refactory_packages.spreadsheetbench.interpreter` or named by `SB_PYTHON`. The
code runs with no network, and only its output and temporary directories writable.

- Linux: an allow-list. The code's root holds only the system directories (`/usr`,
  `/etc` and the `/bin`, `/lib` links), the interpreter's own directories, a few device
  files, a copy of the item's input and an empty working directory (read-only), and its
  output and temporary directories. The dataset and any copy of it, the run, the outer
  loop's root, the home directory, `/tmp`, model caches and shared drives are simply not
  there. Built in user, mount, network and PID namespaces and entered with `pivot_root`,
  then a nested user and mount namespace, so the code cannot undo it.
- macOS: Seatbelt with a hide-list: the dataset and entries named like the release beside
  it, the run, the outer loop's root, the Hugging Face caches (`$XDG_CACHE_HOME/huggingface`,
  `HF_HOME`, `HF_DATASETS_CACHE`), `/new_data` and the home directory's secrets.

Where neither can run, code is not run.

## Differences from SkillOpt as shipped

1. **No item budget and no per-call wall** (SkillOpt: 600 s per item, 120 s per call).
   A solve call ends on inactivity only (`REFACTORY_MODEL_INACTIVITY_S`: nothing
   streamed and no command running); the one limit is the harness backstop
   (`REFACTORY_MODEL_BACKSTOP_S`, 4 h), for each call and for each item's turn loop. An
   item past it is the model's own failure (`error_kind` `backstop`), scored 0 like a
   wrong answer and never rerun.
   The lab's SkillOpt wrapper is to run with the same backstop as its item limit.
2. **Item start**: SkillOpt's clock starts when its worker picks the item up; here when
   the pass's first step runs.
3. **Tools**: refused, as SkillOpt's were. SkillOpt sends no tool schemas of its own,
   but its CLI calls carried Claude Code's default tools under `dontAsk`, which refused
   any write, non-read-only command or read outside the call's empty directory (60% of
   so-s43's target calls had a refusal, nearly all Bash). What needs no permission (a
   read-only command, Read in that directory, ToolSearch, a subagent under the same
   rules) runs in both; here the directory is empty, where SkillOpt's held its
   system-prompt file. A write to Claude Code's memory folder succeeds here for the
   session and is removed with it; in SkillOpt's runs the home was read-only, so it
   failed. With no tools at all, as the port first ran, the model wrote tool calls and
   invented their output as text, at a rate that varied by run: 99 of 304 train solve
   replies in the clean OL-s43 run, 84 of 273 in SO-s42, 5 of 43 in the val start-skill
   pass.
4. **Interpreter**: pandas installed in the pinned environment, not added through
   `PYTHONPATH`; `PYTHONPATH` and `PYTHONHOME` are not passed to model code.
   `INPUT_PATH` names a copy of the input and the code runs in an empty working
   directory (SkillOpt's inherits the trainer's), so paths in tracebacks differ.
5. **Sandbox**: model code cannot read the golden file, which in SkillOpt's runs sat
   readable beside the input (the lab's sandbox hid secrets only). Stricter, so it can
   only lower a score that came from reading the answer.
6. **Grader**: `recalc` by default. Its scores compare only with a SkillOpt run graded
   the same way (the lab's wrapper with `SB_GRADER=recalc`), never with SkillOpt's
   cached-grader numbers. A recalculation LibreOffice cannot make (it fails, or runs past
   its 180 s wall) is told apart by recalculating a known-good tiny workbook with the same
   LibreOffice (`recalc.soffice_works`, trusted for 5 minutes): if that works, the
   prediction's own file is at fault and keeps its cached, failing grade
   (`recalc_failed`); if not, the grader is broken, a harness incident (`error_kind`
   `harness_incident`), not a wrong answer: the outer loop runs the item again and, still one, leaves it out of the score
   (outer-loop.md, "A harness incident is rerun"). So is a sandbox setup that fails.
   An answer cell holding an array formula (openpyxl's `ArrayFormula`) counts as a
   formula and is recalculated; the lab's `lo_grader.py` counts only strings starting
   with `=`, so it graded an array formula on its cached value (none, in a workbook
   openpyxl wrote) and failed it even when right. Excel computes array formulas, so this
   departs from `lo_grader.py`. Formulas are never repaired: a function saved without
   the `_xlfn.` prefix Excel needs (AGGREGATE, say) fails, as it would in Excel.
7. **Analysts' view**: a feedback message after the last model call (a loop that ended on
   turns or the backstop) is not in the conversation the analysts see; `n_turns` counts
   model calls; each message is shown stripped, as the History framing sends it.
8. **Formula note**: SkillOpt's verification report ends, when cells read as text
   starting with `=`, with "WARNING: n cells contain Excel formulas (openpyxl cannot
   evaluate them):" and the cells. The cells it lists hold text starting with `=`, not
   formulas (the report reads cached values, where a real formula reads as its value),
   so the line is wrong under either grader; only "openpyxl cannot evaluate" holds, and
   only under `cached`. A proposer cited it to argue for writing plain values, so under
   `recalc` the line says "NOTE: n cells hold text that starts with '=' (stored as text,
   not as formulas):", with "at least 10" (and "at least n more") when the scan's cap of
   10 cells is hit. `SB_GRADER=cached` keeps SkillOpt's line byte for byte, for parity
   with SkillOpt, not because it is true there. (`mode.py` lists this as deviation 7.)
   The report reaches the analysts and the proposer, not the solver: the gold feedback
   keeps only the wrong cells.

## Parity checks

`python -m refactory_packages.spreadsheetbench.parity` runs the free checks against a real
SkillOpt run directory and the release data, with no model call: `prompts` (§6.1),
`executor` (§6.3, optionally against SkillOpt's own `run_generated_code`), `grader` (§6.4,
cached or recalc against the lab's rescore files), `feedback` (§6.5), and `startskill`,
which compares the port's val passes with SkillOpt's; the reference numbers are an input.
On so-s42 (2026-11-17): 3,014 of 3,088 target calls equal in system and prompt (the other
74 are later turns of rollouts that saved no conversation); 399 of 399 items' first message
and preview equal; 279 of 279 test-baseline code files agree with SkillOpt's executor; every
graded prediction of all 31 passes regrades to SkillOpt's `hard`; 5 of 5 recoverable train
feedback messages equal.

## The agent harness (`spreadsheetbench/agent`)

SkillOpt's target backend `claude_code_exec` with SpreadsheetBench mode `single`
(`adapter.py:70-76` refuses any other mode with it): the paper's "Claude Code" harness
(arXiv 2605.23904, §4, Table 1). Ported from `codegen_agent.py` (`_build_codex_skill`,
`_build_codex_task`, `_build_codex_driver`, `_prepare_codex_workspace`, `run_single`'s exec
branch) and `codex_harness.py` (`render_skill_md`, `prepare_workspace`, `_exec_prompt`,
`run_claude_code_exec` and its CLI path) into `agent_port.py`; the graph is `agent_mode.py`,
its ops `agent_ops.py`. SkillOpt's public history does not hold `ec9d021`; these files are
the ones at `4fa8396` (`codegen_agent.py` unchanged since July, `run_target_exec` at
`codex_harness.py:1977`).

```
data ─▶ prepare ─▶ has_case ─▶ solve ─▶ finalize
                     └──────── skip ──────▲
```

- `prepare` writes SkillOpt's workspace into the pass: `input.xlsx` (the item file),
  `.agents/skills/skillopt-target/SKILL.md` (the skill under SkillOpt's front matter and
  preamble), `task.md` (the solver's first user message, then SkillOpt's "Codex Harness
  Task" section) and `run_solution.py` (SkillOpt's driver); and, for the analysts, what
  `process_one_codegen` saves before the call.
- `solve` is one Claude Code session in a sealed copy of those four files
  (`tools=("Read", "Bash")`, a `scratch` workspace; [outer-loop.md](outer-loop.md), "New
  agents run without tools, or sealed"). It may read the files, run commands (each one
  sealed: the workspace only, no network, the pinned interpreter as `python` and
  `python3`) and writes `solution.py`. Its reply, the final message, is its first output.
- `finalize` takes `solution.py` when the session left one, else the reply's first fenced
  block (`codegen_agent.py:491-496`), runs it once on a copy of the input with SkillOpt's
  runner in the code sandbox, and records the outcome; the Task grades it as the solver's.

What SkillOpt starts, and what this mode sends (`tests/test_spreadsheetbench_agent.py`
checks each against SkillOpt's own code):

| | SkillOpt (`_run_claude_code_cli_exec`, `codex_harness.py:884-955`) | `spreadsheetbench/agent` |
|---|---|---|
| prompt | `_exec_prompt(prompt, allow_file_edits=True)` after `--` | the same bytes (frozen knob `solve__prompt_template`) |
| system prompt | Claude Code's own, nothing appended | the same (frozen knob `solve__system_prompt`, empty) |
| tools | `--tools Read,Bash --allowedTools Read,Bash` | `tools`/`allowedTools` `Read`, `Bash` |
| permission mode | `--permission-mode bypassPermissions` | route `skip` (`bypassPermissions`); another is refused |
| folder | `cwd` the workspace, `--add-dir` it | the sealed workspace, also its added directory |
| model, effort | `--model <target>`, `--effort medium` | the route's (`claude-opus-4-6`, `medium`) |
| thinking | `--max-thinking-tokens 16384` (on 2.1.299 with opus-4-6, adaptive thinking is sent anyway) | frozen knob `max_thinking_tokens`, bound to the solve step's route on both runtimes (the same request) |
| Claude Code | whatever `claude` is on the machine (SkillOpt pins none; the lab's is 2.1.299) | pinned to 2.1.299, the lab's install, as the solver's |

SkillOpt's base config sets `claude_code_exec_use_sdk: auto`, which tries the Python Agent
SDK first when `claude-agent-sdk` is installed, and its SDK path differs: it appends a
system-prompt text and asks for structured output whose `final_response` keeps
`<answer>` tags (`codex_harness.py:801-881`), and its trace steps reach the analysts. This
mode ports the CLI path, the paper's "through the claude CLI"; which path a given SkillOpt
run took depends on that run's environment, not its code.

**No gold feedback.** SkillOpt's single path takes none (`run_single` has no gold
argument); this graph has no feedback node, so neither arm's agent is told anything
about its answer, and a train score is the attempt's.

**What the analysts see.** SkillOpt saves, for an exec rollout, `final_message or raw` as
the whole conversation (`codegen_agent.py:501`: the final message, or the CLI log after an
empty answer), then the verification block (`rollout.py:846-858`), or for code that failed
to run `## Execution (case N)` and its first 500 characters of error (`rollout.py:799-812`),
and as the target's prompts the chat path's system and user prompts and the preview
(`rollout.py:700-721`). Its CLI path's raw output is the final text, so no trace steps
reach them (`codex_harness.py:526-600`), nor the agent's `solution.py`. Each arm here gets
the same:

- the SkillOpt strategy renders this mode's records as SkillOpt does
  (`skillopt_view.agent_rollout_result`, chosen by the mode: `MODE_VIEWS`): the
  instruction, the chat prompts with the preview, the reply, and the verification or
  error block (the record's `prediction`);
- the default loop shows each item's instruction, the preview as the item's context
  (`SpreadsheetBenchAgentTask.item_context`, in full after the input), the solve step's
  prompt and reply, and the same verification or error block as what the item gave. It
  does not show the chat path's prompts, which this agent never sees, nor the tool calls
  or `solution.py`, which SkillOpt's analysts do not see either.

### Differences from SkillOpt's harness

1. **Sealed workspace.** SkillOpt's workspace is a folder of its run
   (`<out>/predictions/<id>/codex_single`) and its tools are not confined: `Bash` under
   `bypassPermissions` reaches the dataset (golden files included), the other items'
   outputs and the network. Here the agent sees its four files and the pinned interpreter's
   folders, never a folder holding what SB hides from model code (`sandbox.hidden_paths`,
   the dataset included: `SpreadsheetBenchTask.tool_environment`'s `hidden`). Stricter,
   so it can only lower a score that came from reading outside the workspace.
2. **Interpreter.** `python` in the workspace is the pinned interpreter model code runs
   under, so `run_solution.py` validates with the interpreter that grades. SkillOpt's
   driver runs whatever `python` the agent's shell finds (`sys.executable` of that
   process), while its grader runs the trainer's own (`executor.py:92-126`).
3. **No item budget.** SkillOpt gives the call `min(600, remaining)` seconds and its
   batch runner scores the item a timeout 600 s after it started, grading included
   (`codegen_agent.py:480-490`, `rollout.py:983-1011`). Here the session ends on inactivity
   and the harness backstop, as in the solver.
4. **No settings.** No user or project settings, CLAUDE.md or MCP servers are loaded;
   SkillOpt's `claude -p` loads the operator's.
5. **Retry.** A crashed or empty session gets one more, in a fresh workspace, after the
   adapter has asked an empty last turn once, in the session, for its answer. SkillOpt
   runs `claude -p` once more in the same folder with a note asking for an `<answer>` tag
   (`codex_harness.py:718-726`, `EXEC_EMPTY_RESPONSE_RETRIES=1`).
6. **A failed session's file.** A session that fails leaves no output: its `solution.py`
   is not graded. SkillOpt grades a `solution.py` that a timed-out or failed call left.
7. **Bash's folder.** Each command starts in the workspace, so a `cd` does not carry to
   the next command; Claude Code's shell keeps it in SkillOpt's.
8. **Read.** On Linux a Read is given a copy of the workspace file, taken after the check
   (outer-loop.md); the content is the same. A Read outside the workspace is refused,
   except of this session's own tool-results and temporary folders in Claude Code's
   folders (where it saves a large tool result for the model to Read back): never another
   session's (an earlier attempt's, or a workspace whose folder name collides), and the
   session's own go with it. SkillOpt's agent may read anything.
9. **Transport.** The Agent SDK's streamed session, not `claude -p --output-format text`.
   On Claude Code 2.1.299 against a stub API the two requests differ slightly: the SDK
   session's system prompt has two lines `-p`'s does not and lacks one it has, the Bash
   tool's text gives a 1800 s (not 600 s) default for background commands, and thinking is
   displayed as `updates` (not `omitted`). Both send adaptive thinking for
   `claude-opus-4-6` despite `--max-thinking-tokens 16384`.
10. **Grader.** `recalc` by default, as in the solver (difference 6 above).
11. **No prediction beside the workspace.** SkillOpt's workspace sits inside the item's
    output folder, so an agent can write `../<case>_pred.xlsx`, and grading then skips
    running its code (`rollout.py:796-799`; workspace at `codegen_agent.py:332-333`). The
    seal shows the agent nothing outside its workspace.

### Problems in SkillOpt's exec path

What its code does, for reading its numbers (evidence only). The port stays faithful to
each, except where a difference above says otherwise, until they are measured:

1. **Conflicting output instructions.** `task.md` ends its first section with "Return only
   a ```python``` code block." (`codegen_agent.py:164`) and its harness section with "Do not
   return a code fence as the primary artifact; the source of truth is `solution.py`."
   (`:255`); the CLI lead asks to "Return only the final answer text, keeping any required
   <answer>...</answer> tags" (`codex_harness.py:713`), and an empty answer's retry asks for
   an `<answer>` tag (`:725`).
2. **The analysts see the wrong prompt and only the final message.** Their "Target System
   Prompt" and "Target User Prompt" are the chat path's (`rollout.py:705-719`), which this
   agent never sees, and their conversation is `final_message or raw` alone
   (`codegen_agent.py:501`): no tool calls (the CLI's raw output is its final text, so
   `format_claude_trace_steps` finds none, `codex_harness.py:526-600`) and no code.
3. **Grading time is counted against the agent.** The call gets `min(600, remaining)`
   seconds (`codegen_agent.py:480-490`), and the batch scores the item a timeout 600 s after
   it started, grading included, whatever `solution.py` holds (`rollout.py:983-1011`).
4. **A raw-log fallback.** Without `solution.py` the code is
   `extract_code(final_message or raw)` (`codegen_agent.py:496`): after an empty answer,
   `raw` is the CLI log (`===== CLAUDE CLI ATTEMPT 1 =====` ...,
   `codex_harness.py:1007-1028`), which is then run as code.
5. **An unconfined workspace.** It is a folder of the run, `<out>/predictions/<id>/codex_single`
   (`codegen_agent.py:332-333`), and `Bash` runs under `bypassPermissions`
   (`codex_harness.py:904`) with no sandbox: the dataset, its golden files, other items'
   outputs and the network are reachable.
6. **Two sessions per attempt.** Under `claude_code_exec_use_sdk: auto` an SDK failure or an
   empty structured answer falls through to the CLI in the same attempt and the same folder
   (`codex_harness.py:975-1028`), and the one retry reruns in that folder again.

7. **A prediction the agent can write itself.** The workspace is inside the item's output
   folder (`codegen_agent.py:332-333`), and grading runs the code only when
   `<case>_pred.xlsx` does not exist (`rollout.py:796-799`): an agent that writes
   `../1_pred.xlsx` is graded on that file, its code never run.

Two more: `run_solution.py` runs the code at module level after its own two path lines
(`codegen_agent.py:305-309`) while the grader runs it indented inside `try:`
(`executor.py:12-23`), so a triple-quoted string spanning lines gains four spaces per line
at grading only; and the driver runs whatever `python` the agent's shell finds while the
grader runs the trainer's own (difference 2 above).

## SpreadsheetBench 2 (`spreadsheetbench_v2/agent`)

A second dataset for comparing the SkillOpt strategy with the default outer loop: both
arms run the same graph, so the optimizer is the only difference. SpreadsheetBench 2
(arXiv 2606.29955; Hugging Face `KAKA22/SpreadsheetBench-v2`, archive
`spreadsheetbench-v2-09-10.zip`, sha256 `52699c97…5b3e`) is business workflows on large
multi-sheet workbooks: Debugging (repair planted errors), Financial_Model (complete a
model) and Template (fill a schedule). **SkillOpt never ran it.** What is whose:

| | SkillOpt's (as in `spreadsheetbench/agent`) | SpreadsheetBench 2's |
|---|---|---|
| harness | one sealed Claude Code session, `Read` and `Bash`, the skill as `.agents/skills/skillopt-target/SKILL.md`, `run_solution.py`, `solution.py` run on the input in the code sandbox; prompt, settings, retry and starting skill | — (V2 runs SWE-agent with `bash`, `view_xlsx`, `submit` and a 50-call cap; not used) |
| task text | the "Codex Harness Task" section of `task.md` | the `<instructions>` block of V2's instance template (`SWE-agent/config/spreadsheet.yaml:49-60`): instruction, input file, output path (`input.xlsx`, `output.xlsx`) |
| data, split | — | the release, prepared and split by workbook (below) |
| grade | — | `open_spreadsheet.py` on the output, then `evaluation/evaluation.py` (`evaluation_port.py`, line for line) |

V2's tool list, workflow and "Important" notes (`spreadsheet.yaml:4-47`) describe its
SWE-agent harness and are not sent. As in `spreadsheetbench/agent` there is no wall-clock
or turn cap (inactivity only, and the harness backstop), and the analysts of both arms see
the same: the instruction, the input's preview (the default loop's item context; SkillOpt's
"Spreadsheet Preview"), the final message, and the verification or execution-error block.
SkillOpt has no V2 environment, so the SkillOpt strategy uses its generic analyst prompts.
`data_parallelism` is 8 (SB's agent mode: 24), since each item's grade runs LibreOffice and
loads a large workbook.

**No gold feedback.** One session has no later turn to tell, and SB's agent mode takes none
either (SkillOpt's single path), so the Task offers no `train_feedback`.

### Where the data is hidden

The agent's sealed workspace holds its own files only; the prepared folder (every golden,
every sibling input, `tasks.json`) is on both hide-lists: the sealed steps' tool
environment (`tasks.tool_hidden`; an interpreter running from inside it is refused) and
model code's sandbox (`agent_ops.finalize`). On Linux both sandboxes are allow-lists, so
the hide-lists are a second guard there; on macOS (Seatbelt) the hide-list is the
protection.

### Prepared data

The grader reads cached values, and many shipped files have none or stale ones: 7
Debugging-05 tasks cannot pass as shipped, and the Financial_Model DigiMark inputs do not
open in openpyxl until LibreOffice rewrites them. V2's README asks for every input and
golden to be recalculated with its `open_spreadsheet.py` before any run. So, once:

```
python -m refactory_packages.spreadsheetbench_v2.prepare prepare <release>/spreadsheetbench-v2 <out>
python -m refactory_packages.spreadsheetbench_v2.prepare check [--grades] <out>
```

`prepare` checks every source file against `manifest.json`, recalculates each input and
each golden (once) as V2's script does — LibreOffice, iterative calculation on (100 steps,
0.0001), `calculateAll`, stored as "Calc MS Excel 2007 XML" (`open_spreadsheet.py`,
`lo_open_spreadsheet.py`; LibreOffice found as the recalc grader finds it, `SB_SOFFICE`) —
one file per LibreOffice, resumably (a file is skipped only while its checksum still
matches), and writes `tasks.json` and `prepared.json` (the manifest's sha256, LibreOffice's
version, every file's sha256).

**Opaque names.** A Debugging source id names its planted error (`01` Double Counting, `05`
Incorrect Average, `10` Unit Mismatch, ...), and an item's id reaches the agent's folder
and environment and both arms' analysts. So each item's id is a seeded hash of its source
(`sb2-<12 hex>`), its input is `items/<id>/input.xlsx` and each golden
`goldens/<hash>.xlsx`; source ids, groups and file names stay in `manifest.json` and the
prepared `tasks.json`, grader side. The split and the drops are computed from the source
ids.

`check --grades` then grades, one task at a time, each golden against itself and each input
with V2's grader, and records the result in `prepared.json`. **The Task requires it** and
leaves out every task whose golden fails against itself, with the reason (also listed in
`manifest.json`'s `excluded` once the lab's full check is committed); a grader failure
stops the check rather than dropping a task. Set `SB2_DATA` to `<out>`: the Task refuses a
directory that is not prepared or checked, was prepared for another manifest, or has any
file (or `tasks.json`) whose checksum changed.

### Drops and the split

277 of the release's 321 tasks remain (Debugging 80, Financial_Model 100, Template 97);
`manifest.json` lists every dropped id with its reason:

- the 24 **Visualization** tasks: V2 grades them by exporting charts through Excel or WPS
  COM on Windows and asking a VLM checklist judge, not by file comparison;
- the 10 Debugging **Embedded Hardcode** tasks: V2 grades them by formula text, so only the
  golden's exact formulas pass (owner's decision: impossible);
- the 10 Debugging **Inconsistent Color Coding** tasks: V2 grades font colours against a
  convention the instruction never states (owner's decision: unspecified).

Goldens are shared within a Debugging workbook (10 tasks) and a Financial_Model project
(5), and each sibling input holds the others' fixes, so the split is by group, never by
task: seed 20261123, `random.Random(seed).sample` of the sorted groups (Debugging 2 train
and 1 val, Financial_Model 5 and 3), Template groups as the audit chose them (train
`16_re_direct_cap`, `14_re_dcf`, `03_consolidation`; val `01_bond_accounting`,
`02_cash_sweep`); the rest is test.

| part | Debugging | Financial_Model | Template | total |
|---|---|---|---|---|
| train | 16 (`08`, `10`) | 25 (`03`, `06`, `13`, `17`, `18`) | 25 | 66 |
| val | 8 (`03`) | 15 (`01`, `12`, `19`) | 15 | 38 |
| test | 56 | 60 | 57 | 173 |

### The grade

V2's pipeline as its README runs it: the output is recalculated by `open_spreadsheet.py`,
then `evaluation.py` compares it with the prepared golden. Its rules, ported line for line:
a number within 1% (relative; absolute against 0), `None` equal to 0 and to `""`, dates as
whole Excel days (so within 1% too), formatting ignored, a cell holding an Excel error
compared by its formula (so a real `#DIV/0!` or `#N/A` cell never equals `"N/A"` text:
V2's not-meaningful rule applies only between texts such as `"N/A"` and `"-"`, and a formula
error only matches the same formula), and an item
correct when every changed cell matches and at least 99.8% of the unchanged ones do, after
rounding to four places (1,955 of 1,959 passes). A task whose golden changes no answer cell
fails whatever the output (V2's modification accuracy is 0.0 then). The parity fixture
(`tests/fixtures/spreadsheetbench_v2_grader_parity.json`) holds V2's own results on 30
cases and 28 value pairs.

How it runs differs, never what it decides:

1. **One workbook at a time.** V2 holds the input, golden and output open twice each at
   once; a Debugging-10 workbook then needs more than 3 GB, and V2's `except Exception`
   scores the `MemoryError` 0 with an empty message. Here each workbook is read and closed
   in turn, and the formula view only when an answer cell holds an Excel error.
2. **No silent 0.** A prepared input or golden that cannot be read, or memory running out,
   is a harness incident (`GraderFailed`) with its message: run again, then left out of the
   score. An output the model's code wrote that cannot be opened is the model's failure,
   `output-unreadable: <why>` (V2 also scores it 0). An output LibreOffice, running, cannot
   open is graded as written when LibreOffice works on a known-good workbook (V2's script
   leaves such a file in place). When LibreOffice itself fails on it -- crashing, killed
   (out of memory), failing to calculate or store, or past its process guard -- a fresh
   check on a known-good workbook runs: LibreOffice unhealthy, it is a harness incident;
   healthy, the output is recalculated once more (retries in one process run one at a time,
   but other items' first attempts still run beside it, so a retry can still run out of
   memory next to other work), and failing again scores 0
   (`recalc-failed: LibreOffice could not recalculate the output ...`), so an output that
   hangs LibreOffice every time cannot turn into an incident left out of the score.
3. **Recalculation.** One file per private LibreOffice profile and pipe, a copy opened
   under a plain name, the source never changed (V2: one service on port 2002, the file
   replaced). The helper streams nothing to measure inactivity by, so a 900 s guard on
   LibreOffice's own process keeps a hung one from holding an item; reaching it is a harness
   incident, not the model's failure.

### Not verified here

The full prepare of all 277 tasks (554 files) and `check --grades` on them are meant for
the lab; locally, the recalculation ran through LibreOffice 7.4 on single files (the
DigiMark input among them, which then opens) and on the tests' circular reference.

## V2's own agent setup (`spreadsheetbench_v2/swe`)

SpreadsheetBench 2's leaderboard runs its own SWE-agent config
(`SWE-agent/config/spreadsheet.yaml`), not SkillOpt's harness; this mode ports that setup to
one sealed Claude Code session per item, on the same data, split, Task items and grade as
`spreadsheetbench_v2/agent`. The session has Claude Code's `Bash` alone in a workspace
holding `input.xlsx`; it edits the workbook itself and writes `output.xlsx`, which V2's
grade grades (no `solution.py`, nothing run afterwards). No `output.xlsx` fails the item
(`output-not-found: the agent left no output.xlsx`).

- **Prompt**: V2's instance template, with only the substitutions this runtime needs
  (`swe_port.SUBSTITUTIONS`): `Bash` for `bash`, `view_xlsx` named as a command run with
  `Bash`, "submit" meaning ending the turn (no `submit` tool), and the paths `input.xlsx`
  and `output.xlsx`. The "Important", "Environment" and "Recommended Workflow" sections are
  V2's. V2's system template is appended to Claude Code's system prompt. Both are frozen
  knobs.
- **Tools on the PATH**: V2's `view_xlsx` (byte for byte, `swe_tools/view_xlsx`), and
  LibreOffice as `soffice` and `libreoffice` -- the grader's own (`SB_SOFFICE`), its
  install folder and `/etc/fonts` bound read only. LibreOffice puts its pipe in `/tmp`
  only, so these sessions get a `/tmp` of their own, writable and kept across their commands
  and removed with the dispatch (`ToolEnvironment.private_tmp`, Linux only: on macOS the
  step is refused); its profile is in the workspace, the command's HOME. The launcher caps
  LibreOffice's thread pool and malloc arenas (`MAX_CONCURRENCY=4`, `MALLOC_ARENA_MAX=2`):
  on a many-core machine its xlsx export otherwise fails under a sealed command's
  per-process limits.
- **V2's limits**, each a frozen knob bound to the solve step's route: the 50-call cap is
  Claude Code's `--max-turns 50` (`max_turns`); the 60 s command timeout is Claude Code's
  default and longest Bash timeout (`bash_timeout_ms`, `BASH_DEFAULT_TIMEOUT_MS` and
  `BASH_MAX_TIMEOUT_MS`); the 10,000-character observation is the most of a command's
  output shown inline (`bash_output_chars`, Claude Code's `bashOutputMaxChars` setting).
  The last two were read from Claude Code 2.1.299's code, not run against it.
- **Turn cap reached**: the session ends with a fixed note as its answer and the
  `output.xlsx` it left is graded. The adapter records the session's facts beside its workspace,
  out of the session's reach (Claude Code's turn count, the cap, whether it was reached);
  a file the session writes in its workspace under the same name is ignored. The item's
  `final.json` records `n_turns` and `turn_limit`, and its verdict ends with
  `(the session stopped at its turn cap: 50 turns, cap 50)`, which the analysts and reports
  see. When the adapter left no facts (a stand-in model), `final.json` says
  `session_facts: missing` and the verdict says the turn count is unknown.
- **Tool calls**: each item's folder keeps `tool_calls.json`, the session's tool calls in
  order (tool, the command as the model wrote it, output cut to 2,000 characters, whether it
  was an error) with Claude Code's turn count, to compare runs step by step with V2's
  SWE-agent trajectories. Neither the agent nor either arm's analysts see it (no record
  field, view or item context reads it), so both arms stay on the same information.
- **Skill**: the `skill` knob, written to `SKILL.md`, is the text the SkillOpt strategy and
  the default loop optimize. It is empty by default (V2 has none); empty, the prompt does
  not mention it (`SKILL.md` is then an empty file in the workspace).
- **Analysts**: as in `spreadsheetbench_v2/agent` -- the input's preview as the item's
  context; SkillOpt's view shows the final message, then V2's mismatch block or the fail
  reason, with V2's system template and the prompt sent as the target's prompts. No gold
  feedback reaches the agent.

**Deviations from V2**, each with its `spreadsheet.yaml` line in `swe_port.py`'s docstring:
Claude Code's `Bash` in place of SWE-agent's function calls, and `view_xlsx` a command whose
argument docs are not sent as a schema (its `--help` prints them); no `submit` tool (the kept
line "Do NOT combine verify and submit commands" then means: verify before ending the
turn); the paths, and the input a writable copy (V2 mounts it read only; the grade never
reads this copy); the system template appended rather than the whole system message; the
turn cap as `--max-turns` (a Claude Code turn is one model response, which may make several
tool calls; a SWE-agent call makes one); a timed-out command is reported in Claude Code's
words, and V2's stop after 3 consecutive timeouts has no equivalent; long output is cut to a
preview with the whole output saved to a file (V2: head and tail with a warning), and V2's
no-output warning is not ported; the skill section, only when a skill is set; the
environment: the pinned Python 3.12 with openpyxl, pandas and numpy (V2's image: Python
3.11 with matplotlib, xlsxwriter and flake8 too), LibreOffice's `uno` not on `python3`'s
path, no network; "one tool call per response" is the prompt's instruction only; a crashed
session is run once more and an empty last turn asked once for its answer (V2 does neither).

**Not verified here.** No live model run. LibreOffice inside the seal ran locally
(LibreOffice 26.8, `test_libreoffice_runs_inside_the_seal`); CI has no LibreOffice and
skips that test, so the stand-in checks only that `soffice` is reachable. When a run
directory is itself under `/tmp`, the step's own `/tmp` shows the empty folders the seal
mounts the workspace on (they go with the dispatch).

## V2's workflow as steps (`spreadsheetbench_v2/steps`)

`spreadsheetbench_v2/swe` sends one session V2's whole workflow in prose ("Recommended
Workflow": 1 Inspect, 2 Plan, 3 Implement, 4 Execute, 5 Verify, 6 Submit), so the outer
loop can only edit text. This mode lifts that workflow into the graph, so the loop has
structure to optimize:

```
prepare -> has_case -> inspect -> plan -> implement -> verify -> finalize
```

Each step is a sealed Claude Code session set up exactly as swe's solve step -- the same
Task items, split and grade, `Bash` alone with `view_xlsx` and LibreOffice on the PATH, a
`/tmp` of its own, the same retry -- on swe's code paths (`swe_ops.lay_workspace`,
`swe_ops.session_facts`, `SpreadsheetBenchV2SweTask`'s tool environment). Steps hand on
files through the pass, each declared:

| step | upstream stages (`spreadsheet.yaml`) | reads (besides `input.xlsx`, `SKILL.md`) | writes |
|---|---|---|---|
| `inspect` | 1 Inspect (:28-32) | -- | `inspect_notes.md` |
| `plan` | 2 Plan (:34) | `inspect_notes.md` | `plan.md` |
| `implement` | 3 Implement, 4 Execute (:36-39) | `inspect_notes.md`, `plan.md` | `output.xlsx` |
| `verify` | 5 Verify, 6 Submit (:41-45) | `inspect_notes.md`, `plan.md`, `output.xlsx` | `verify_report.md`, `output.xlsx` |

Inspect and plan do not declare `output.xlsx`, so one they leave is never handed on.
Verify may correct `output.xlsx` (swe's "verify, then submit"); the `output.xlsx` in the
pass after the last step is graded as swe's is, and none fails the item
(`output-not-found`).

- **Prompts**: each step's prompt is swe's (V2's instance template with swe's
  substitutions) with the "Recommended Workflow" body cut to the step's own stages, byte for
  byte, followed by a short "## This Step" section naming what it reads and writes
  (`steps_port.STEP_SECTIONS`, the only text not V2's). The steps' stages in order are V2's
  whole workflow. Everything else goes to every step unchanged: "Important", "Tools",
  "Environment", "Task instructions" and "Input and Output", with the item's instruction
  and paths filled from prepare's facts (`{{instruction}}`, `{{spreadsheet_path}}`,
  `{{output_path}}`). The appended system prompt is V2's system template for every step.
- **Knobs**: each step's prompt and system prompt are its own open knobs,
  `<step>__task` and `<step>__system_prompt` (eight texts), so the loop edits one step
  without touching the others. `skill` is one `SKILL.md` every step reads; when it is
  non-empty every prompt names it (`{{skill_section}}`). Frozen, as swe's: `max_turns`
  (50, bound to every step's route), `bash_timeout_ms` (60 s) and `bash_output_chars`
  (10,000).
- **Cost**: each step has its own 50-turn cap, so an item may use up to 200 turns where
  swe's one session uses 50. The item's `final.json` records each step's turns
  (`step_turns`, Claude Code's own counts) and their total (`n_turns`), the steps stopped at
  their cap (`turn_limits`; the verdict then names each one), the steps with no session
  facts (`session_facts_missing`) and the steps that left no reply (`steps_without_reply`).
  `tool_calls.json` keeps every step's calls, in step order. Each step's session facts
  are its own declared write, `<step>/.refactory-session.json`
  (`workspace.is_session_log`).
- **A hand-off a step did not write**: a read that only sealed Claude Code steps write, as
  a file they may leave out, is optional for a sealed Claude Code reader
  (`workspace.optional_reads`, on both runtimes): a step stopped at its cap before it wrote
  its notes leaves the next step to run without them, and verify runs without an
  `output.xlsx` implement did not write (and may write one). Never silently: each absent
  read is recorded per step (`workspace.absent_reads`), in `final.json` as
  `missing_inputs` (`{"plan": ["inspect_notes.md"]}`) and in the verdict the analysts see
  (`(the plan step ran without inspect_notes.md: no earlier step wrote it)`). Every other
  read is still required. A step whose reply has no text is listed in
  `steps_without_reply`.
- **Analysts**: as swe's -- the input's preview as the item's context. SkillOpt's view
  shows each step's final message headed by its step, in order, then V2's verdict; the
  target's prompts are each step's prompt as sent and its system prompt, headed by step.
- **No added structure**: no loops, judges or routes; only `data` and `finalize` are
  protected. What else helps is the loop's to find.

**A different starting point.** This is not V2's setup: four sessions with their own turn
caps and hand-off files replace one. Its baseline score and cost against
`spreadsheetbench_v2/swe` are to be measured, not assumed.

**Not verified here.** No live model run; the steps ran with scripted agents (walker and
spine) and the real Claude Code adapter with a faked Agent SDK.
