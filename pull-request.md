# F14 part 1: the optimizer changes a node only through knobs

## Summary

Remove `set_node_field` and `replace_node` from the optimizer's edit catalogue. Introduce typed knob-to-field bindings (`field_bindings` on `Node`) that generalise the existing `parallelism_knob` pattern, and auto-declare a system-prompt knob on every agent node so instructions are optimizable out of the box. Bindable fields use an explicit allowlist enforced by V13 on both runtimes.

## Changes

- **nn/graph.py** — Add `field_bindings: tuple[tuple[str, str], ...]` to `Node`; add `BINDABLE_FIELDS` allowlist (`system_prompt_template`, `prompt_template`, `max_iterations`, `parallelism`); sparse serialization; round-trip via `to_state()`/`from_state()`.
- **nn/edits.py** — Delete `_replace_node`, `_set_node_field`, and their `_primitive()` registrations; delete `random_edit()` branches for both ops. Add data-slot preservation check in `_set_knob` and `_text_patch`: if a knob is bound to a template field, every `{{slot}}` from the original template must appear in the new value.
- **nn/validate.py** — Add rule V13: field name must be in `BINDABLE_FIELDS`; `parallelism` restricted to data nodes; knob must exist in declared knobs. Refuses `role`, `notes`, `tools`, `tool_policy`, and everything else not in the allowlist.
- **nn/mode.py** — `_auto_system_prompt_knobs()`: after `build()`, every agent node gets an open `str` knob `{node_id}__system_prompt` bound to `system_prompt_template`, with the original template text as default. Skipped on name collision with user-declared knobs.
- **nn/executor.py** — Resolve `system_prompt_template`, `prompt_template`, and `max_iterations` from `field_bindings` before rendering. Add `_effective_max_iterations()` helper used by gate retry and halt paths.
- **lowering.py** — Resolve field bindings for agent nodes (system_prompt_template → system_prompt, prompt_template, max_iterations) and gate nodes (max_iterations) before emitting IR. Strip `field_bindings` from lowered output.
- **optim/base.py** — Remove `replace_node` and `set_node_field` from `is_structural()`.
- **loop.py** — Refuse replay of journal entries containing `set_node_field` or `replace_node` with a clear `LoopError`.
- **livemath/mode.py** — Freeze the auto-declared `solve__system_prompt` knob explicitly; add it to the solve node's `knobs` and `field_bindings`. The `skill` knob remains the optimizable text.
- **docs** — Update architecture.md, tutorial.html, 01-write-a-mode.md, documentation.json: remove the two ops from counts, lists, and preset tables.

## Tests

- `test_nn.py` — `test_field_bindings_roundtrip`, `test_field_bindings_sparse_serialization`, `test_v13_unbindable_field`, `test_v13_unbindable_tools`, `test_v13_unbindable_tool_policy`, `test_v13_unbindable_notes`, `test_v13_unbindable_limit`, `test_v13_parallelism_only_data_nodes`, `test_v13_valid_binding`, `test_v13_nonexistent_knob`, `test_auto_system_prompt_knob_static_template`, `test_auto_system_prompt_knob_slotted_template`, `test_auto_system_prompt_knob_empty_template`, `test_auto_knob_collision_detection`, `test_lowering_resolves_field_bindings`, `test_lowering_strips_field_bindings`, `test_executor_resolves_field_bindings`, `test_field_binding_precedence`, `test_data_slot_preservation_prompt_template`, `test_data_slot_preservation_system_prompt_template`, `test_frozen_knob_field_binding_refused`, `test_configure_frozen_field_binding_succeeds`, `test_text_patch_on_system_prompt_knob`, `test_is_structural_after_removal`, `test_two_runtime_system_prompt_knob`, `test_livemath_knob_names_unchanged` (Python and TS runtimes as noted in each).
- `test_tool_policy.py` — Rewritten to use `_tool_policy_escalated` directly or `set_knob`; `test_add_node_with_tools_triggers_escalation` (edit-path guard via `add_node`).
- `test_item_ownership.py` — Rewritten to use direct state mutation; `test_add_node_with_exam_field_triggers_guard` (edit-path guard via `add_node`).
- `test_loop_holdout.py` — Rewritten to use direct state mutation and `add_node`.
- `test_loop.py` — `test_replay_refuses_removed_ops`.

## Verification

All tests run on both runtimes where applicable. Lowering snapshots regenerated. `refactory-targeted-tests` passes. LiveMath knob set pinned including frozen `solve__system_prompt`.

## Behaviour changes

- **System prompt now optimizable by default** on every agent node. An open `str` knob `{node_id}__system_prompt` is auto-declared with the original template text as default. Templates with `{{...}}` slots get the slot-preservation check. **To turn it off:** freeze the auto-knob in `Mode.declare()`.
- **Bindable fields are an explicit allowlist:** `system_prompt_template`, `prompt_template`, `max_iterations`, `parallelism` (data nodes only). Everything else refused by V13.
- **Affected strategies/presets:** `broad` and `structure` presets included `set_node_field` and `replace_node`; they no longer do. `conservative` and `knob_only` never included them.
- **Existing journals** containing `set_node_field` or `replace_node` are refused on resume.

## Follow-ups

- F14 part 2: restrict `add_node` and insert/wrap macros to declared library blocks (after a block library exists for arm C).
- Arm C itself.
