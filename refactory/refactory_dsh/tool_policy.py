"""The tool policies an agent step may declare (``tool_policy``).

Both make a step whose model can change nothing: its reply is its one output, it reads
only what its prompt inlines, and asking it again repeats no work. They differ in what
the model is shown:

``none``
    No tools at all: the request carries no tool definitions (DSH's no-tools plugin,
    Claude Code's ``tools: []``).
``refused``
    Claude Code's default tools are offered, and Claude Code's own ``dontAsk``
    permission mode refuses every call that needs permission (a command, a write, a read
    outside its directory) with its usual "Permission to use ... has been denied"
    result -- what ``claude -p --permission-mode dontAsk`` does, as SkillOpt's harness
    ran it. The session works in a fresh empty directory, also its one added directory
    (``--add-dir``), so a tool that needs no permission (Read or Glob there, ToolSearch,
    a subagent under the same rules) finds nothing. Only the ``claude_code`` backend,
    with ``permission_mode`` ``dontAsk``, and never with user settings, whose allow rules
    could let a tool run.
"""
from __future__ import annotations

from typing import Any

#: Every accepted ``tool_policy`` value; a step without one keeps its backend's tools.
POLICIES = ("none", "refused")

#: The worker backends that can run each policy.
BACKENDS = {"none": ("dsh", "claude_code"), "refused": ("claude_code",)}

#: The permission mode ``refused`` runs under: Claude Code's own refusal.
REFUSED_PERMISSION_MODE = "dontAsk"
#: The route permission modes that give it: ``dontAsk``, or ``default``, which leaves the
#: mode to the Claude Code adapter, whose own default is ``dontAsk`` (the adapter refuses
#: a refused session under any other).
REFUSED_ROUTE_MODES = (REFUSED_PERMISSION_MODE, "default")


def no_effect(policy: Any) -> bool:
    """Whether ``policy`` makes a step whose model can change nothing (`POLICIES`)."""
    return policy in POLICIES


def check(policy: Any) -> str:
    """``policy`` as given ('' for none declared), or ValueError for an unknown one."""
    if policy in (None, ""):
        return ""
    if policy not in POLICIES:
        raise ValueError(f"invalid tool_policy {policy!r}; accepted: {', '.join(POLICIES)}")
    return policy


def route_refusal(policy: Any, route: dict[str, Any]) -> str | None:
    """Why a step with ``policy`` cannot run on ``route``, or None when it can."""
    if not no_effect(policy):
        return None
    backend = route.get("worker_backend", "dsh")
    if backend not in BACKENDS[policy]:
        return (f"the {backend!r} worker backend cannot enforce tool_policy {policy!r}, "
                f"which runs only on {' or '.join(BACKENDS[policy])}")
    if policy == "refused":
        mode = route.get("permission_mode") or "skip"
        if mode not in REFUSED_ROUTE_MODES:
            return (f"tool_policy 'refused' needs permission_mode {REFUSED_PERMISSION_MODE!r}, "
                    f"whose refusal it is; the route has {mode!r}")
        if "user" in str(route.get("setting_sources") or "").split(","):
            return ("tool_policy 'refused' cannot load user settings (setting_sources has "
                    "'user'): their allow rules could let a tool run")
    return None
