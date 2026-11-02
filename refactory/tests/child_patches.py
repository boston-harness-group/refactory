"""Test fakes that reach the child process each op and fn gate runs in.

The walker runs every op and fn gate in a child forked from its forkserver, as the spine
runs each in a process of its own, so a ``monkeypatch.setattr`` made in the test's
process never reaches the op it is meant for. A test that fakes something an op calls
names the fake here instead, with :meth:`ChildPatches.setattr`: a constant, or a
module-level function of a test module. The child imports that module by name, as it
imports the test ops (``loop_ops``): the forkserver starts with the test process's
``sys.path``, which `conftest` gives the tests directory.

The fake is installed in the test's process as ``monkeypatch.setattr`` installs it, and
recorded in the environment the child is handed. Each op and fn gate of the run is then
dispatched, through the real forkserver path, as :func:`run`, which installs the
recorded fakes in the child before running the node's own callable. A fake's keyword
``params`` (JSON) are bound to it; :func:`real` is what the attribute was before any
fake, in either process. State a fake keeps across dispatches lives in files.
"""
from __future__ import annotations

import functools
import importlib
import json
import os
import threading
import types
from typing import Any

from refactory_dsh.nn.resolve import resolve

#: The environment variable the recorded fakes and the nodes' own callables travel in.
ENV = "REFACTORY_TEST_CHILD_PATCHES"
#: What each patched attribute was before the running test's first fake of it, by
#: ``(target, attribute)``; in an op child, before the fakes it installed.
_REAL: dict[tuple[str, str], Any] = {}
_LOCK = threading.Lock()


def _key(target: Any) -> str:
    """How the child finds ``target``: a module by its name, a class as ``module:Class``."""
    if isinstance(target, types.ModuleType):
        return target.__name__
    return f"{target.__module__}:{target.__qualname__}"


def _target(key: str) -> Any:
    return resolve(key) if ":" in key else importlib.import_module(key)


def real(target: Any, name: str) -> Any:
    """What ``target.name`` was before any fake replaced it."""
    return _REAL.get((_key(target), name), getattr(target, name))


def _value(kind: str, value: Any, params: dict[str, Any]) -> Any:
    if kind == "constant":
        return value
    if kind == "module":
        return importlib.import_module(value)
    function = resolve(value)
    return functools.partial(function, **params) if params else function


def _install(patches: list[list[Any]]) -> None:
    for key, name, kind, value, params in patches:
        target = _target(key)
        _REAL.setdefault((key, name), getattr(target, name))
        setattr(target, name, _value(kind, value, params))


def run(ctx: Any) -> Any:
    """The child's op: install the recorded fakes, then run the node's own callable."""
    spec = json.loads(os.environ[ENV])
    _install(spec["patches"])
    return resolve(spec["calls"][ctx.node_id])(ctx)


def for_test(monkeypatch: Any) -> "ChildPatches":
    """The test's one :class:`ChildPatches`, undone with its ``monkeypatch``."""
    found = getattr(monkeypatch, "_child_patches", None)
    if found is None:
        found = ChildPatches(monkeypatch)
        monkeypatch._child_patches = found  # noqa: SLF001 - one per test
    return found


class ChildPatches:
    """``monkeypatch.setattr`` for attributes an op or fn gate reaches."""

    def __init__(self, monkeypatch: Any) -> None:
        self.monkeypatch = monkeypatch
        self.patches: list[list[Any]] = []
        self.calls: dict[str, str] = {}
        self.routed = False

    def setattr(self, target: Any, name: str, value: Any, **params: Any) -> None:
        """Replace ``target.name`` (a module's or a class's) in this process and in every op
        child of the test. A method's fake takes no ``params``: they would not bind ``self``.

        ``value`` is a module-level function of an importable module (bound to
        ``params``), an importable module, or a JSON constant.
        """
        if isinstance(value, types.ModuleType):
            kind, recorded = "module", value.__name__
        elif callable(value):
            ref = f"{value.__module__}:{value.__qualname__}"
            if "<locals>" in ref or resolve(ref) is not value:
                raise ValueError(f"{ref} cannot reach the op's child: make it a module-level function")
            kind, recorded = "function", ref
        else:
            json.dumps(value)
            kind, recorded = "constant", value
        key = _key(target)
        if not any(patch[:2] == [key, name] for patch in self.patches):
            # What the attribute is now, a plain monkeypatch of this test's included, is
            # what this test's fakes call through to; it is put back with the test.
            self.monkeypatch.setitem(_REAL, (key, name), getattr(target, name))
        self.monkeypatch.setattr(target, name, _value(kind, recorded, params))
        self.patches = [patch for patch in self.patches if patch[:2] != [key, name]]
        self.patches.append([key, name, kind, recorded, params])
        self._publish()
        if not self.routed:
            self._route()

    def _publish(self) -> None:
        with _LOCK:
            self.monkeypatch.setenv(ENV, json.dumps({"patches": self.patches, "calls": self.calls}))

    def _route(self) -> None:
        """Dispatch each op and fn gate as :func:`run`, through the walker's own path."""
        from refactory_dsh.nn.executor import Executor

        dispatch = Executor._run_op_timed
        patches = self

        def routed(executor: Any, callable_target: str, node: Any, timeout: float) -> Any:
            if patches.calls.get(node.id) != callable_target:
                patches.calls[node.id] = callable_target
                patches._publish()
            return dispatch(executor, f"{__name__}:run", node, timeout)

        self.monkeypatch.setattr(Executor, "_run_op_timed", routed)
        self.routed = True
