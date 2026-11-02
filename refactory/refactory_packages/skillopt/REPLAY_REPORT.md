# SkillOpt Offline Replay Report

## s42 Replay (seed 42)

**Recording:** `~/skillopt/runs/skillopt-train-20261023-e10-s42`
**Status:** `mismatch` at epoch 1, merge stage
**Calls matched:** 58 / 1280

### Breakdown

| Stage | Calls | Status |
|-------|-------|--------|
| Solver rollout (epoch 1, 35 items) | 35 | ✓ matched |
| Val baseline + epoch 1 eval (17 items) | 17 | ✓ matched |
| Analyst failure (4 batches) | 4 | ✓ matched |
| Analyst success (2 batches) | 2 | ✓ matched |
| Merge failure | 1 | ✗ mismatch |
| **Total matched** | **58** | |

### First Divergence

- **Epoch:** 1
- **Step:** 1
- **Stage:** merge (aggregate)
- **Field:** patch ordering in the merge prompt

**Cause:** A ran analyst batches concurrently via `ThreadPoolExecutor`;
patches arrived in completion order. B runs analyst batches sequentially;
patches are in batch-index order. The merge prompt serializes all patches
with `json.dumps`, so the ordering difference changes the prompt content.

Additionally, A's `_normalise_patches` (trainer.py:157-186) injects
`source_type` and `support_count` fields into each edit before serialization.
B's current port does not add these fields, contributing to the prompt diff.

**B value (first patch reasoning):**
> The overwhelming pattern (5/8) is dismissing meta-options. The agent needs
> explicit guidance to treat 'a stronger result can be proven' as a serious
> candidate...

**A value (nearest recorded, first patch reasoning):**
> The agent failed to properly handle 'meta-options' — choices that assert
> another option is correct but a stronger result holds...

### Matched Stages Detail

All 35 solver rollout calls for epoch 1 matched by content: B's livemath
prompt construction (mode.py `SYSTEM_PROMPT` + ops.py `format_prompt`)
produces the same prompts as A after the transport wrapping is normalized.

All 6 analyst calls matched: B's analyst prompt construction (system prompt
from the LiveMath-specific analyst template + user prompt with skill, budget,
and trajectory text including target system/user prompts) matched A's
recorded analyst calls.

The 17 validation calls (baseline evaluation of initial skill) also matched.

### Porting Bugs Identified and Fixed

1. **Missing `source_type` / `support_count` on edits** — A's
   `_normalise_patches` (skillopt/engine/trainer.py:157-186, MIT) adds
   `source_type` and `support_count` to each edit dict via
   `item.setdefault()` before merging. B's trainer.py only set
   `result["source_type"]` on the parent result dict, not on individual
   edits within `patch["edits"]`. **Fixed** in trainer.py: now propagates
   `source_type` and `support_count` to each edit dict after extraction,
   matching A's `_normalise_patches` behavior.

### Behavioral Differences (Not Porting Bugs)

2. **Sequential vs concurrent analyst execution** — A runs analyst
   minibatches concurrently with `ThreadPoolExecutor(max_workers=workers)`.
   B runs them sequentially. The resulting patch order differs, which
   changes the merge prompt. This is a behavioral difference, not a
   porting bug — B's sequential execution is correct but produces patches
   in a different order.
