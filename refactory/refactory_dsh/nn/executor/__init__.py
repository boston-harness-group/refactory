"""Execute a built graph in this process.

One node per branch at a time, with independent fork branches scheduled concurrently
and every control-flow branch decided by a gate's verdict. That model is why the
validator can make promises a free-for-all executor could not: fan-out is forbidden on a non-gate node, every
cycle passes through a gate, and a gate carries the iteration bound that stops
it.

The executor also holds the dataflow model to its word. A node declares what it
writes; an op that writes anything else is refused, because the validator's
"every read is produced by some node" promise is only true if the declarations
are.
"""

# Re-export every name the old module exposed so existing imports keep working.

from .context import (
    ModelClient as ModelClient,
    RunContext as RunContext,
    Step as Step,
    Verdict as Verdict,
    _SENTINEL as _SENTINEL,
    _eval_condition as _eval_condition,
    _render_conditionals as _render_conditionals,
    by_declaration as by_declaration,
    declared_files as declared_files,
    fills_facts as fills_facts,
    names_matter as names_matter,
    render as render,
)
from .failures import (
    ExecutionError as ExecutionError,
    GateStop as GateStop,
    HaltStop as HaltStop,
    OpOutcomeUnknown as OpOutcomeUnknown,
    OpTimeoutError as OpTimeoutError,
    _BranchFailed as _BranchFailed,
    _ItemDeadlineError as _ItemDeadlineError,
    route_verdict as route_verdict,
)
from .items import (
    CURRENT_ITEM as CURRENT_ITEM,
    ITEMS_OVERRIDE as ITEMS_OVERRIDE,
    SMOKE_SAMPLE as SMOKE_SAMPLE,
    declared_items as declared_items,
    full_items as full_items,
    load_items as load_items,
    smoke_sample_ids as smoke_sample_ids,
    task_items as task_items,
)
from .op_child import (
    _MarkedCapture as _MarkedCapture,
    _RUN_WARNING_RE as _RUN_WARNING_RE,
    _ask_model as _ask_model,
    _emit_run_warnings as _emit_run_warnings,
    _ensure_forkserver_preload as _ensure_forkserver_preload,
    _finish_op_child as _finish_op_child,
    _forkserver_ctx as _forkserver_ctx,
    _kill_op_child as _kill_op_child,
    _main_file_lock as _main_file_lock,
    _op_child_target as _op_child_target,
)
from .walker import (
    PROGRESS as PROGRESS,
    Executor as Executor,
    _run_end_cleanup as _run_end_cleanup,
    execute as execute,
)


def __getattr__(name: str):
    if name == "_forkserver_modules":
        from . import op_child
        return op_child._forkserver_modules
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
