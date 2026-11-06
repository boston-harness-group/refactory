# 2. Optimize it

The loop searches a mode's architecture **and** its parameters. This tutorial runs
it, reads what it found, and reads the diff that explains the result.

Prerequisite: nothing. `circle_packing/adaptive` ships.

## Run it

```bash
uv run refactory optimize circle_packing/adaptive \
  --epochs 2 --batch-size 4 --optimizer random --seed 5 --runs-root /tmp/tour/run
```

```
{
  "mode": "circle_packing/adaptive",
  "optimizer": "random",
  "evaluator": "walker",
  "bound": "epochs",
  "evaluations": 8,
  "seed_score": 0.795,
  "best_score": 0.875,
  "raw_best_score": 0.955,
  "scoring": {"compared": "penalized", "parsimony": 0.01, ...},
  "holdout": {"status": "not_applicable", "train": 0.955, "holdout": null, "overfit": null,
              "note": "the mode has no data node, so it iterates no items and there is nothing to hold out"},
  "structural_wins": 0,
  "rejection_codes": {"E-shape": 1, "I:fact_produced": 4, "V10": 2, "V2": 3, ...},
  "report": "/tmp/tour/run/report.md"
}
```

Three things to notice:

- **A bound is always named.** `"bound": "epochs"` — a run never ends for an
  unstated reason. The other bound is `max_evaluations`.
- **The score improved**: seed `0.795` → best `0.875`, in 8 inner runs. Those are the numbers the search compares: the raw score with `0.01` taken off for each node of the candidate's graph. The unpenalised pair is beside them (`0.875` → `0.955`), so a harness that won on merit can be told from one that won on size. Here every candidate has 8 nodes, and the `duplicate` the search tried next has 9 — which is why it scores `0.865` raw `0.955` and loses to its parent. `structural_wins` counts only structural candidates that beat their parent, so it is `0` here.
- **Nothing is held out when there is nothing to hold out.** `circle_packing/adaptive` has no data node and iterates no items, so the audit records `holdout: null` and says why instead of inventing a number. A mode bound to a Task is audited on the items it was not tuned on.
- **Refusals cost nothing.** `rejection_codes` lists every proposal the validator
  rejected *before* it could be evaluated.

Execution failures remain in `failed_candidates`, with their errors, but cannot
win or be adopted. If every evaluated candidate fails, the report has no winner,
`exit_ok: false`, and the CLI exits nonzero. The holdout status distinguishes a
measurement from a failed or unavailable audit and from a mode with no split.
Adoption requires a completed audit for the current winner; `--allow-overfit`
acknowledges a measured gap but cannot override a missing or failed audit.
An explicit live Adopt request audits its current winner before writing the package;
a later winner receives its own audit. Legacy score records without an explicit
successful exit remain visible in history but are ineligible until re-evaluated.

Resuming a completed search through `LoopConfig(resume=True)` honors its existing
epoch and evaluation bounds. It can consume a queued selection change and audit
the resulting winner without spending another proposal epoch.

## The steps, in order

The loop is named after a training loop, and running it with `--verbose`-style
inspection shows the same eight steps a PyTorch loop has:

| step | what it did |
|---|---|
| `init` | seeded the population — the seed, plus one neighbour per movable knob |
| `forward` | ran the batch; the only step that spends budget |
| `loss` | turned the batch into one number |
| `backward` | read the traces into the gradient the optimizer receives |
| `optimizer` | proposed the next batch, validating before anything ran |
| `step` | installed the offspring that were genuinely new points |
| `scheduler` | closed the epoch and recorded what bound the run |
| `report` | wrote the leaderboard, the gradient, and the markdown |

## Read the leaderboard

```bash
uv run refactory leaderboard --run /tmp/tour/run
```

Each row carries the candidate, its score, and a **diff against its parent**:

```json
[
  {"id": "4caa88db3dbc", "parent": "b6f0d3c9e601", "score": 0.875, "raw_score": 0.955,
   "nodes": 8, "size_cost": 0.08, "structural": false,
   "edits": [{"op": "set_knob", "name": "grid", "value": 18}],
   "diff": {"knobs": {"grid": {"before": 10, "after": 18}}, ...}},
  {"id": "6061dbf4589f", "parent": "4caa88db3dbc", "score": 0.865, "raw_score": 0.955,
   "nodes": 9, "size_cost": 0.09, "structural": true,
   "edits": [{"op": "set_knob", "name": "grid", "value": 18}, {"op": "duplicate", "target": "accept"}]}
]
```

This is the point of a diff on a leaderboard: without it the table says which
number is bigger and nothing about why.

## Read one candidate in full

```bash
uv run refactory candidate --run /tmp/tour/run 6061dbf4589f
```

The `rendered` field is the human-readable form of the same diff, and the row also
carries what the run measured (`score`, `exit_ok`, `duration_s`).

## Read the report

```bash
cat /tmp/tour/run/report.md
```

```
# Outer loop — circle_packing/adaptive

Optimizer `random`, 2 epoch(s) of 2, batch 4, 8 inner run(s). Stopped because **epochs** bound the run.

Seed score **0.795** → best **0.875** (raw 0.875 → 0.955). The compared score is the raw one minus 0.01 per node of the candidate's graph.
```

The markdown carries a leaderboard, the structural candidates that scored, and a
section for why proposals were refused — because a reader needs the refusals as
much as the wins.

## Watch a run that has not finished

A run is a directory of files, so you can read it *while* it runs:

```bash
uv run refactory status --run /tmp/tour/run --after 0
```

`--after` takes a line count, not a timestamp: a reader polling wants the events it
has not seen, and a clock cannot tell it which those are.

`progress.jsonl` has one line **per candidate evaluated**, not per epoch — an epoch
can be an hour.

## Then watch it on DSH

The walker is the debug runtime. The same mode on the runtime that ships:

```bash
uv run python -m pytest refactory/tests/test_equivalence.py -q --run-integration
```

```
6 passed
```

Tutorial [4](04-ship-it-on-dsh.md) walks through that in detail.

## What to take away

- A run is a **directory**, readable without a service — before, during, and after.
- The report says what happened *and* what was refused, and by which rule.
- Every candidate's state is in the journal, and its diff is one command away.

Next: [author with a model](03-author-with-a-model.md).
