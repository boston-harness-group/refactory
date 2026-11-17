# SpreadsheetBench (`spreadsheetbench/solver`)

SkillOpt's SpreadsheetBench target agent, ported so the outer loop can be measured
against SkillOpt on the same items, prompts, code runner and grader. Source:
microsoft/SkillOpt at `ec9d021` (MIT), `skillopt/envs/spreadsheetbench/`; every ported
function names the lines it copies (`refactory_packages/spreadsheetbench/skillopt_port.py`).

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
- **56225 passes if the input comes back unchanged.** The release marks it
  `exclude: "ignore, initial file already passes verification"`; SkillOpt ignores the
  field, and so does this port.

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
- `solve` is one model call with no tools; its reply is the node's output.
- `run_turn` waits 3 s, takes the reply's first fenced block (the whole reply when there
  is none, so prose comes back as a SyntaxError, not as "no code"), and runs it with
  SkillOpt's runner. An error is the next user message.
- `gold_check` is the train-only gold feedback node (below); `next_turn` ends the item on
  a clean run, unless that node says the output is wrong.
- `turn_gate` loops while turns (30) and the item budget (more than 10 s) remain.
- `finalize` fails an empty last code block, runs the last code once more when there is
  no prediction, and scores an item past its budget 0.

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

The SkillOpt strategy asks for it on its train rollout only.

## Where model code runs

A pinned interpreter with SkillOpt's versions (Python 3.12, openpyxl 3.1.5, pandas 2.3.3,
numpy 2.5.4 and pandas' own dependencies), built once with
`python -m refactory_packages.spreadsheetbench.interpreter` or named by `SB_PYTHON`. The
code runs in a fresh directory holding a copy of the input, with no network, the dataset,
the run and the home directory's secrets hidden, and nothing writable but its output
directory and its temporary directory: Linux user, mount and network namespaces, macOS
Seatbelt. Where neither can run, code is not run.

## Differences from SkillOpt as shipped

1. **Item budget**: `SB_ITEM_BUDGET_S`, 3600 s by default (SkillOpt: 600 s). One setting
   for the port and for the lab's SkillOpt wrapper, so every compared approach gets the
   same budget. The loop stops when 10 s or less are left and an item past the budget
   scores 0, as in SkillOpt. **Late calls are not killed**: a call that starts inside the
   budget runs to its end, so wall time and cost can exceed the budget though the score
   follows SkillOpt's rule.
2. **Per-call cap**: none beyond the budget (the lab's `SB_LLM_CALL_CAP=task`; SkillOpt
   ships 120 s).
3. **Retries**: a crashed call is retried 3 times (4 sessions); SkillOpt's wrapper makes
   5 attempts.
4. **Tools**: none on the solve call. SkillOpt sends no tool schemas, but its CLI calls
   carried Claude Code's default tools, which `dontAsk` refused (74% of so-s42's target
   calls tried one).
5. **Interpreter**: pandas installed in the pinned environment, not added through
   `PYTHONPATH`; `PYTHONPATH` and `PYTHONHOME` are not passed to model code.
6. **Sandbox**: model code cannot read the golden file, which in SkillOpt's runs sat
   readable beside the input. Stricter, so it can only lower a score that came from
   reading the answer.
7. **Grader**: `recalc` by default.
8. **Analysts' view**: a feedback message after the last model call (a loop that ended on
   turns or budget) is not in the conversation the analysts see.

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
