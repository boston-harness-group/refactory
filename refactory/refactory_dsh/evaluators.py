"""How a mode is run to get the score the loop learns from.

The outer loop is a training loop, so it needs a loss. Two of its extension points
are seams with a base class and a registry -- :class:`~refactory_dsh.optim.Optimizer`
proposes and :class:`~refactory_dsh.outerloop.OuterLoop` drives -- and this is the
third. It was a string compared against a literal inside ``Loop._evaluate``:

    evaluator: str = "walker"
    if self.config.evaluator == "dsh":
        self._evaluate_on_dsh(candidate, mode, workdir)
        return

That is enough for exactly two evaluators and enough for neither to be tested
against a shared contract. An evaluator answers one question -- *given a mode and a
state, run it and tell me what it scored* -- and answering it in one place is what
lets a third one be added without touching the loop.

**Both return the same result.** :class:`~refactory_dsh.nn.mode.RunResult` already
carried the fields a candidate needs (`score`, `facts`, `exit_ok`, `error`,
`duration_s`, `model_calls`), so this module adds no result type and the loop's
candidate assignment is written once instead of once per branch.

**The two that ship are not the same kind of thing.** ``walker`` runs the graph
in-process: it is the fast one, and the speed is what makes a search over dozens of
candidates practical. ``dsh`` lowers the state to IR and runs it on the runtime that
ships, which costs a process boot per candidate. They must agree on the score, and
:func:`~refactory_dsh.evaluators.create` is where the choice between them lives;
``tests/test_equivalence.py`` is what keeps them honest.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from refactory_dsh.memory import MemoryStore
from refactory_dsh.nn.mode import RunResult

#: Kinds a mode can be run in. A search wants the fast one; a person using a mode
#: wants the one whose result they will act on.
DEFAULT_MAX_STEPS = 1000


def _print_line(line: str) -> None:
    """One line of a candidate's own output, printed without doubling its newline."""
    print(line.rstrip("\n"), flush=True)


def _persist_steering_option(workdir: Path, omit_steering: bool | None) -> None:
    """Persist an explicit prompt policy for both runtimes and actual inner-run resume."""
    if omit_steering is None:
        return
    import json

    path = Path(workdir) / ".refactory" / "run-config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            config = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read run configuration at {path}: {error}") from error
        if not isinstance(config, dict):
            raise ValueError(f"run configuration at {path} must be a JSON object")
    else:
        config = {}
    config["omit_steering"] = bool(omit_steering)
    path.write_text(json.dumps(config, indent=2))


class EvaluatorError(ValueError):
    """An evaluator cannot be built. Always names what is wrong."""


class Evaluator:
    """One way to run a mode, and the score the loop reads off the result.

    A subclass supplies ``name``, ``evaluate`` and ``from_settings``. It is built
    from the loop's own settings rather than reached for globals, so an evaluator
    is testable without a loop.
    """

    #: Selected by ``LoopConfig.evaluator``.
    name: str = ""
    #: ``True`` for an evaluator that runs a graph somewhere other than this
    #: process, so a caller can tell the slow one from the fast one without
    #: matching on the name.
    out_of_process: bool = False
    budget: Any = None

    @classmethod
    def from_settings(
        cls, *, config: Any, model: Any = None, project: Path | None = None
    ) -> "Evaluator":
        """Build this evaluator from a loop's configuration."""
        raise NotImplementedError

    def evaluate(self, mode: Any, state: dict[str, Any], workdir: Path) -> RunResult:
        """Run ``mode`` once in ``workdir`` and return what it produced.

        ``state`` is passed alongside the instance because the out-of-process
        evaluator has to lower it to IR; the in-process one ignores it.
        """
        raise NotImplementedError


class WalkerEvaluator(Evaluator):
    """Run the graph in this process. The fast one, and the default."""

    name = "walker"

    def __init__(
        self,
        *,
        model: Any = None,
        project: Path | None = None,
        memory_global_root: Path | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        omit_steering: bool = False,
    ) -> None:
        self.model = model
        self.project = Path(project or Path("."))
        self.memory_global_root = memory_global_root
        self.max_steps = max_steps
        self.omit_steering = bool(omit_steering)

    @classmethod
    def from_settings(
        cls, *, config: Any, model: Any = None, project: Path | None = None
    ) -> "WalkerEvaluator":
        return cls(
            model=model,
            project=project,
            memory_global_root=config.memory_global_root,
            max_steps=config.max_steps,
            omit_steering=getattr(config, "omit_steering", False),
        )

    def evaluate(self, mode: Any, state: dict[str, Any], workdir: Path) -> RunResult:
        _persist_steering_option(workdir, self.omit_steering)
        # One store per inner run, and its `run`-retention namespaces are deleted
        # when it ends. The roots already keep candidates apart; forgetting is what
        # makes the policy true of the *same* candidate evaluated twice. A resumed
        # run re-evaluates candidates, and a run-scoped memory left on disk would be
        # read the second time, so the resumed run would not be the run it resumed.
        store = MemoryStore(
            mode.memory_declarations(),
            run_root=workdir,
            project_root=self.project / ".factory",
            global_root=self.memory_global_root,
        )
        started = time.time()
        try:
            result = mode.evaluate(
                workdir, model=self.model, memory=store, max_steps=self.max_steps,
                **({"shared_budget": self.budget} if self.budget is not None else {}),
                enable_recovery=True, omit_steering=self.omit_steering,
            )
        finally:
            # Also on a crash or a stop: a namespace that outlives the run it was
            # scoped to is the bug, however the run ended.
            store.forget_run_scoped()
        result.duration_s = round(time.time() - started, 4)
        return result


class DshEvaluator(Evaluator):
    """Lower the state to IR and run it on DSH. The runtime that ships."""

    name = "dsh"
    out_of_process = True

    def __init__(
        self, *, timeout_s: int | None = None, on_output: Callable[[str], None] | None = None,
        model: Any = None, omit_steering: bool = False
    ) -> None:
        from refactory_dsh.dsh_runtime import DEFAULT_TIMEOUT_S

        self.timeout_s = DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
        self.model = model
        self.omit_steering = bool(omit_steering)
        #: Where a candidate's own run output goes while it runs. Only a verbose
        #: run passes one: the meaningful per-candidate line is the loop's own event.
        self.on_output = on_output

    @classmethod
    def from_settings(
        cls, *, config: Any, model: Any = None, project: Path | None = None
    ) -> "DshEvaluator":
        watching = bool(getattr(config, "print_events", False))
        verbose = bool(getattr(config, "verbose", False))
        return cls(
            timeout_s=config.dsh_timeout_s,
            on_output=_print_line if watching and verbose else None,
            model=model,
            omit_steering=getattr(config, "omit_steering", False),
        )

    def evaluate(self, mode: Any, state: dict[str, Any], workdir: Path) -> RunResult:
        _persist_steering_option(workdir, self.omit_steering)
        from refactory_dsh.dsh_runtime import evaluate_on_dsh

        model = self.model
        if self.budget is not None and model is None:
            from refactory_dsh.model import DshModelClient

            model = DshModelClient().with_budget(self.budget)

        return evaluate_on_dsh(
            mode, state, workdir, timeout_s=self.timeout_s, on_output=self.on_output,
            **({"model": model} if model is not None else {}),
        )


#: The evaluators the loop can select by name.
REGISTRY: dict[str, type[Evaluator]] = {
    WalkerEvaluator.name: WalkerEvaluator,
    DshEvaluator.name: DshEvaluator,
}


def names() -> list[str]:
    return sorted(REGISTRY)


def register(name: str, cls: type[Evaluator]) -> None:
    """Add an evaluator without editing the library."""
    if not issubclass(cls, Evaluator):
        raise TypeError(f"{cls.__name__} is not an Evaluator")
    REGISTRY[name] = cls


def create(
    name: str, *, config: Any, model: Any = None, project: Path | None = None
) -> Evaluator:
    """Build an evaluator by name, failing loud on an unknown one.

    Falling back silently would invalidate every result of the run, because the
    report would not say how the candidates were run.
    """
    if name not in REGISTRY:
        raise EvaluatorError(f"unknown evaluator {name!r}; known: {names()}")
    return REGISTRY[name].from_settings(config=config, model=model, project=project)


__all__ = [
    "DEFAULT_MAX_STEPS",
    "DshEvaluator",
    "Evaluator",
    "EvaluatorError",
    "REGISTRY",
    "RunResult",
    "WalkerEvaluator",
    "create",
    "names",
    "register",
]
