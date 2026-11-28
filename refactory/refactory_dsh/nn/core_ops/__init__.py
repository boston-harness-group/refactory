"""The core structural operators, split into focused modules.

Every name the original ``core_ops`` module exposed is re-exported here,
so ``from refactory_dsh.nn.core_ops import X`` and ``core_ops.X`` still work.
"""
# ruff: noqa: F401
from __future__ import annotations

from ._common import (
    CORE_OPS,
    EARLIER_VERDICT_RULES,
    FILES_HEADING,
    JUDGE_KINDS,
    LOOP_ITERATIONS,
    MAP_ELEMENT_PROMPT,
    MAP_ELEMENT_TOOLS_PROMPT,
    MAP_MAX,
    MAX_BRANCHES,
    MAX_ROUTES,
    NEUTRAL_CHECK,
    NEUTRAL_COMBINE,
    NEUTRAL_ELEMENT,
    NEUTRAL_REVIEW,
    NEUTRAL_ROUTE,
    NEUTRAL_STEP,
    STEP_KINDS,
    TASK,
    TOOLS_HINTS,
    VERDICT_RULE,
    _DEFAULT_WIRED,
    _Refused,
    _default_wired,
    _edits,
    _rule_pattern,
    system_appended,
    verdict_rule,
)

from .state import (
    _available,
    _consumed,
    _data_ends,
    _downstream,
    _forks_naming,
    _fresh,
    _grader_side,
    _is_code,
    _join_of,
    _kind_of,
    _knob_values,
    _outputs,
    _require_step,
    _sealed,
    _sole_out,
    _taken_names,
    _task_inputs,
    _task_knob,
    _text_of,
    _uses_tools,
)

from .visibility import (
    SEES_NAMES_SHOWN,
    _PLACEMENT_SHOWN,
    _heading,
    _missing_edge_hint,
    _no_task,
    _not_on_every_path_hint,
    _placement_anchors,
    _producers_of,
    _readable,
    _seen,
    _strip_unseen,
    _with_sections,
)

from .build import (
    _Steps,
    _add_chain,
    _add_detached,
    _carried_knobs,
    _chain,
    _copy_spec,
    _default_code,
    _drop_orphans,
    _frozen_settings,
    _orphaned_knobs,
    _output_renames,
    _renamed,
    _respec,
    _shared_knobs,
    _simulate,
    _spec,
    _step_nodes,
)

from .placement import (
    _dead_in,
    _filled_text_knobs,
    _fills,
    _new_facts,
    _place,
    _place_all,
    _readers_from,
    _reslotted,
    dead_knob_reason,
    dead_text_knobs,
)

from .checks import (
    _head,
    _inputs_helper,
    _losses,
    _owned,
    _plain_step,
    _slots_in_field,
    _strip_steps,
    _upstream,
)

from .insert import (
    _by_declaration,
    _code_text,
    _expand_insert,
    _insert_default,
    _insert_spec,
    _marked,
    _names_itself,
    _neighbour_view,
    _parse_takes_over,
    _pre_insert,
    _read_renames,
    _renamed_refs,
    _renamed_slots,
    _roles,
    _runs_first_on_every_path,
    _select_reads,
    _successors,
    _take_over_plan,
    _unscored_take_over,
    _where,
    _writer_group,
)

from .remove import (
    CODE_HINT_NOTE,
    _bypass,
    _expand_remove,
    _fold_forks,
    _pre_remove,
    _replace_hint,
)

from .parallelize import (
    _chain_of,
    _clone_agent,
    _expand_parallelize,
    _fork_join_ids,
    _pre_parallelize,
    _tools_beside,
)

from .serialize import (
    _expand_serialize,
    _pre_serialize,
    _serialize_plan,
    _serialize_steps,
)

from .branch import (
    _branch_spec,
    _expand_branch,
    _pre_branch,
)

from .rewire import (
    OWN_TEXT_HINT,
    _expand_rewire,
    _own_text_hint,
    _own_text_slots,
    _pre_rewire,
    _rewire_seen,
)

from .loop import (
    _agents_after,
    _back_to,
    _default_gate_code,
    _expand_loop_line,
    _forward_flow,
    _gate_state,
    _iterations_knob,
    _judge,
    _judge_spec,
    _loop_cap,
    _loop_cycle_valid,
    _map_runs,
    _pass_of,
    _pre_loop_line,
    _reach,
    _scope_start,
    is_one_line_loop,
)

from .route import (
    _LABEL,
    _expand_route,
    _pre_route,
    _route_next,
    _route_take_over,
    _routes,
)

from .map_op import (
    COMBINE_OPS,
    NAMING_FIELDS,
    _combine_spec,
    _expand_map,
    _map_list,
    _map_max,
    _map_part,
    _map_removed,
    _maps,
    _pre_map,
    _pre_remove_map,
    _remove_map,
    _unplaced,
)

from .fallback import (
    _at_rejoin,
    _expand_fallback,
    _fallback_owner,
    _fallback_parts,
    _fallback_spec,
    _pre_fallback,
    _unfallback,
)

from .split_op import (
    _expand_split_line,
    _pre_split_line,
    _split_specs,
    _verdict_read,
    is_one_line_split,
)

from .merge import (
    _expand_merge,
    _merge_kind,
    _merge_text,
    _merge_tools,
    _pre_merge,
)

from .catalog import (
    FALLBACK_TOOLS,
    HINT_KNOBS,
    INSERT_TOOLS,
    JUDGE_TOOLS,
    MAP_TOOLS,
    MERGE_TOOLS,
    OPERATORS,
    REACHES,
    ROUTE_FIELD,
    ROUTE_JUDGE_TOOLS,
    SEALED_CODE,
    SEALED_JUDGE_CODE,
    STEP_FIELDS,
    STEP_SEALED,
    STEP_TOOLS,
    _NODE_FIELDS,
    _graded_step,
    _guarded,
    _guarded_expand,
    _needed,
    _open_knob,
    _outputs_read,
    _pointed,
    _reader_hint,
    _resolved_texts,
    _settle_optional,
    _taken_away,
    _tools_refusal,
    _unknown_fields,
    _with_item_files,
    _without_text,
    default_item_inputs,
    example,
    item_files_before,
    item_input_notes,
    lost_inputs,
    parallel_join_id,
    reader_knobs,
    refusal_without_tools,
    register,
    route_field_line,
    sample,
    select_call,
    unread_by_bundle,
    unread_new_steps,
    wiring_notes,
    without_tools,
)

__all__ = ['CODE_HINT_NOTE', 'COMBINE_OPS', 'CORE_OPS', 'EARLIER_VERDICT_RULES', 'FALLBACK_TOOLS', 'FILES_HEADING', 'HINT_KNOBS', 'INSERT_TOOLS', 'JUDGE_KINDS', 'JUDGE_TOOLS', 'LOOP_ITERATIONS', 'MAP_ELEMENT_PROMPT', 'MAP_ELEMENT_TOOLS_PROMPT', 'MAP_MAX', 'MAP_TOOLS', 'MAX_BRANCHES', 'MAX_ROUTES', 'MERGE_TOOLS', 'NAMING_FIELDS', 'NEUTRAL_CHECK', 'NEUTRAL_COMBINE', 'NEUTRAL_ELEMENT', 'NEUTRAL_REVIEW', 'NEUTRAL_ROUTE', 'NEUTRAL_STEP', 'OPERATORS', 'OWN_TEXT_HINT', 'REACHES', 'ROUTE_FIELD', 'ROUTE_JUDGE_TOOLS', 'SEALED_CODE', 'SEALED_JUDGE_CODE', 'SEES_NAMES_SHOWN', 'STEP_FIELDS', 'STEP_KINDS', 'STEP_SEALED', 'STEP_TOOLS', 'TASK', 'TOOLS_HINTS', 'VERDICT_RULE', 'dead_knob_reason', 'dead_text_knobs', 'default_item_inputs', 'example', 'is_one_line_loop', 'is_one_line_split', 'item_files_before', 'item_input_notes', 'lost_inputs', 'parallel_join_id', 'reader_knobs', 'refusal_without_tools', 'register', 'route_field_line', 'sample', 'select_call', 'system_appended', 'unread_by_bundle', 'unread_new_steps', 'verdict_rule', 'wiring_notes', 'without_tools']
