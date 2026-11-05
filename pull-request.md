# F14 part 1: the optimizer changes a node only through knobs

## Summary

Remove `set_node_field` and `replace_node` from the edit catalogue. Introduce typed knob-to-field bindings (`field_bindings` on `Node`) with an explicit allowlist (`system_prompt_template`, `prompt_template`, `max_iterations`, `parallelism`). Auto-declare an open `str` knob on every agent node's `system_prompt_template`, making it optimizable by default. Frozen knobs stay frozen on every path.

## Changes

### Removals
- `set_node_field` and `replace_node` deleted from `nn/edits.py` (functions, `_primitive()` registrations, `random_edit()` branches).
- Removed from `is_structural()` in `optim/base.py`.
- Replay refusal in `loop.py`: journals containing these ops raise `LoopError` with a clear message.

### Field bindings (`nn/graph.py`)
- `Node.field_bindings: tuple[tuple[str, str], ...]` — pairs of `(field_name, knob_name)`.
- `BINDABLE_FIELDS` allowlist: `system_prompt_template`, `prompt_template`, `max_iterations`, `parallelism`.
- Sparse serialization (omitted when empty); round-trip via `to_state()`/`from_state()`.

### Validation (`nn/validate.py`)
- `V13`: field must be in `BINDABLE_FIELDS`; `parallelism` only on data nodes; knob must exist and be consumed by the node. Refuses `role`, `notes`, `tools`, `tool_policy`, `limit`, `id`, `kind`, `call`, and every other field not in the allowlist.

### Auto system-prompt knob (`nn/mode.py`)
- Every agent node gets `{node_id}__system_prompt` — an open `str` knob defaulting to the node's current `system_prompt_template` (or `""` for empty). Includes nodes with `{{...}}` template slots.
- Authors freeze it to opt out. LiveMath freezes `solve__system_prompt` explicitly; the `skill` knob remains the optimizable text.

### Slot preservation (`nn/edits.py`)
- `set_knob` and `text_patch` on a knob bound to `prompt_template` or `system_prompt_template` enforce that every `{{slot}}` in the original template appears in the new value (E-knob).

### Lowering (`lowering.py`)
- Field bindings resolved to concrete values; `field_bindings` stripped from the IR (TS spine never sees binding metadata).

### Executor (`nn/executor.py`)
- Python walker resolves `system_prompt_template` from field bindings before template rendering.

### LiveMath (`livemath/mode.py`)
- `solve__system_prompt` frozen in mode declaration. Same knob names, same prompts, F20 seed-42 replay matches.

### Docs
- `architecture.md`: operation count updated, removed ops listed, field-bindings paragraph added.
- `tutorials/01-write-a-mode.md`: `set_node_field` example replaced with `remove_node`.
- `tutorial.html`: preset table updated.
- `documentation.json`: embedded text updated.

## Tests

| Test | Runtime | What it proves |
|------|---------|---------------|
| `test_field_bindings_roundtrip` | Python | Serialization lossless |
| `test_field_bindings_sparse_serialization` | Python | Empty bindings omitted from state |
| `test_v13_unbindable_field` | Python | Fields outside allowlist refused |
| `test_v13_unbindable_tools` | Python | `tools` refused |
| `test_v13_unbindable_tool_policy` | Python | `tool_policy` refused |
| `test_v13_unbindable_notes` | Python | `notes` refused |
| `test_v13_unbindable_limit` | Python | `limit` refused |
| `test_v13_parallelism_only_data_nodes` | Python | `parallelism` restricted to data nodes |
| `test_v13_valid_binding` | Python | Valid bindings pass |
| `test_v13_nonexistent_knob` | Python | Missing knob refused |
| `test_auto_system_prompt_knob_static_template` | Python | Auto-knob created for static template |
| `test_auto_system_prompt_knob_slotted_template` | Python | Auto-knob created for slotted template; slot check enforced |
| `test_auto_system_prompt_knob_empty_template` | Python | Auto-knob created with default `""` |
| `test_auto_knob_collision_detection` | Python | No auto-knob when name collides |
| `test_lowering_resolves_field_bindings` | TS (via lowering) | Lowered IR has resolved value |
| `test_lowering_strips_field_bindings` | TS | No binding metadata in IR |
| `test_executor_resolves_field_bindings` | Python | Walker uses knob value |
| `test_field_binding_precedence` | Both | Binding overrides direct field |
| `test_data_slot_preservation_prompt_template` | Python | Missing slot refused |
| `test_data_slot_preservation_system_prompt_template` | Python | Missing slot refused |
| `test_frozen_knob_field_binding_refused` | Python | Frozen knob edit refused |
| `test_configure_frozen_field_binding_succeeds` | Python | Person can set frozen knob |
| `test_text_patch_on_system_prompt_knob` | Python | text_patch works on auto-knob; frozen refused |
| `test_is_structural_after_removal` | Python | set_knob non-structural; add_node structural |
| `test_two_runtime_system_prompt_knob` | Both | Same rendered prompt and records on walker and spine |
| `test_replay_refuses_removed_ops` | Python | Journal with removed ops raises LoopError |
| `test_livemath_knob_names_unchanged` | Python | Pinned knob set including frozen auto-knob |
| `test_catalogue_excludes_removed_free_form_ops` | Python | replace_node/set_node_field not in catalogue |
| `test_add_node_with_tools_triggers_escalation` | Python | E-tool-policy-escalation via add_node edit |
| `test_add_node_with_exam_field_triggers_guard` | Python | E-exam/E-items via add_node edit |

## Verification

- `pnpm test` (TS): all passing.
- `.venv/bin/python -m pytest -q -n 4 refactory/tests`: all passing.
- F20 seed-42 replay: matched.
- LiveMath knob names pinned including frozen `solve__system_prompt`.

## Behaviour changes

- **System prompt now optimizable by default** on every agent node. An open `str` knob `{node_id}__system_prompt` is auto-declared with the original template text as default. For slotted templates, slot preservation is enforced.
- **To turn it off:** freeze the auto-knob in `Mode.declare()`.
- **Bindable fields are an explicit allowlist:** `system_prompt_template`, `prompt_template`, `max_iterations`, `parallelism` (data nodes only). Everything else refused by V13.
- **Affected strategies/presets:** `broad` and `structure` presets included `set_node_field` and `replace_node`; they no longer do. `conservative` and `knob_only` never included them.
- **Existing journals** with `set_node_field` or `replace_node` are refused on resume.

## Follow-ups

- **F14 part 2:** restrict `add_node` and insert/wrap macros to declared library blocks (after a block library exists for arm C).
- **Arm C** itself.
- **Full `parallelism_knob` → `field_bindings` unification:** remove `parallelism_knob` shorthand from `Node` (deferred to avoid breaking existing state dicts).
