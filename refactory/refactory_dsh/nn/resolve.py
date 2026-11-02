"""Runtime-owned callable resolution.

A mode names a step as ``"module.path:function"`` and the runtime imports it. The
library and the engine therefore never name a package: the only thing that knows
where an operation lives is the mode that asked for it.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable


class ResolutionError(ImportError):
    """A ``module:function`` target does not resolve to a callable."""


def resolve(target: str) -> Callable[..., Any]:
    """Import ``"module.path:function"`` and return the callable it names.

    Both ``mod:fn`` and ``mod.fn`` are accepted. The reason is reported with the
    target and the underlying import error, because a mode whose op does not
    resolve must be fixable from the message alone.
    """
    if not target or not isinstance(target, str):
        raise ResolutionError(f"empty callable target: {target!r}")
    if ":" in target:
        module_name, _, attribute = target.partition(":")
    else:
        module_name, _, attribute = target.rpartition(".")
    if not module_name or not attribute:
        raise ResolutionError(
            f"{target!r} is not a 'module.path:function' target"
        )
    # An op runs in a forkserver child, which does not carry the caller's
    # __main__: the child would have to re-import the caller's script, and that
    # re-runs its module-level code once per op visit -- including a module-level
    # execute(), which nests runs on one run directory. So a one-file script
    # cannot name its own op; put it in a module the child can import.
    if module_name == "__main__":
        raise ResolutionError(
            f"{target!r} names the caller's __main__, which an op child cannot "
            "import: move the function into an importable module (for example "
            "my_ops.py beside the script) and name it 'my_ops:" + attribute + "'"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ResolutionError(f"cannot import {module_name!r} for {target!r}: {exc}") from exc
    obj: Any = module
    for part in attribute.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise ResolutionError(
                f"{module_name!r} has no attribute {attribute!r} ({target!r})"
            ) from exc
    if not callable(obj):
        raise ResolutionError(f"{target!r} resolved to {type(obj).__name__}, not a callable")
    return obj


def catalogue(module_name: str) -> list[str]:
    """The ``module:function`` targets a module offers.

    Only callables defined in the module itself are listed, so an op module that
    imports helpers does not advertise them. This is how a mode, or the generator
    that writes one, finds the operations a package already has.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ResolutionError(f"cannot import {module_name!r}: {exc}") from exc
    names = []
    for name, obj in vars(module).items():
        if name.startswith("_") or not callable(obj):
            continue
        if getattr(obj, "__module__", "") != module_name:
            continue
        names.append(f"{module_name}:{name}")
    return sorted(names)
