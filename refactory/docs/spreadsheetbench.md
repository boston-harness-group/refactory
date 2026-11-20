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
- `solve` is one model call with no tools; its reply is the node's output.
- `run_turn` waits 3 s, takes the reply's first fenced block (the whole reply when there
  is none, so prose comes back as a SyntaxError, not as "no code"), and runs it with
  SkillOpt's runner. An error is the next user message. A solve call that ended without
  an answer (all 5 sessions crashed, or it ran out of output) is not a reply: the item
  ends there with no answer (score 0, `fail_reason` `no answer: solve crashed in all 5
  sessions`, the failure under `missing_steps`), as a crashed step does in any mode. Only
  a call that answered with nothing gets SkillOpt's "No Python code block" message.
- `gold_check` is the train-only gold feedback node (below); `next_turn` ends the item on
  a clean run, unless that node says the output is wrong.
- `turn_gate` loops while turns (30) remain and more than 10 s remain before the harness
  backstop.
- `finalize` fails an empty last code block, runs the last code once more when there is
  no prediction, and records an item past the backstop as the model's own failure
  (`backstop`).

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
3. **Tools**: none on the solve call. SkillOpt sends no tool schemas, but its CLI calls
   carried Claude Code's default tools, which `dontAsk` refused (74% of so-s42's target
   calls tried one).
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
