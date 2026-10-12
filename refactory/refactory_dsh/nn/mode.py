"""A mode: a Python class whose architecture and parameters are serializable.

The class plays the role ``nn.Module`` plays: it declares the parameters, builds
the structure, and scores a run. The state dict plays the role ``state_dict``
plays, with one deliberate difference — it carries the architecture as well as
the parameters, because the outer loop searches architecture too. The validator
in :mod:`refactory_dsh.nn.validate` is what replaces the structural guarantee
``load_state_dict(strict=True)`` gets for free in PyTorch.

Authority after genesis is the state, not the class. ``build()`` is how an author
constructs the starting architecture; from then on the optimizer edits that
architecture as data, and an edit is applied to the state dict rather than by
re-running ``build()``. A knob is therefore consumed by a node at run time, never
read by ``build()`` to choose structure; the validator refuses a knob no node
consumes, so the two mechanisms cannot be confused.

``score()`` and ``invariants()`` are methods. They are never part of the state,
so no proposal can reach them: protecting the score is structural, not defensive.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping

from .declarations import Memory, Port
from .graph import Graph, GraphError, Node
from .knob import Knob, KnobError
from .validate import Invariant

#: Bumped when the state layout changes incompatibly.
STATE_VERSION = 1

_TOP_LEVEL = frozenset({"version", "graph", "knobs", "ports", "memory"})
_KNOB_ENTRY = frozenset({"value", "spec"})


class StateError(ValueError):
    """A state dict is not a valid state for this mode."""


@dataclass
class RunResult:
    """What one inner run produced: the facts, the score, and how the run went."""

    facts: dict[str, Any] = field(default_factory=dict)
    iterations: int = 0
    duration_s: float = 0.0
    exit_ok: bool = True
    error: str | None = None
    workdir: Path | None = None
    score: float = 0.0
    #: How many model calls the run made, so an inserted agent node's cost is a
    #: reported number rather than something a reader has to infer from wall clock.
    model_calls: int = 0


class _Values:
    """Attribute access to a mode's current parameter values."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_values", values)

    def __getattr__(self, name: str) -> Any:
        values = object.__getattribute__(self, "_values")
        if name not in values:
            known = ", ".join(sorted(values)) or "none"
            raise AttributeError(f"no knob {name!r}; declared knobs: {known}")
        return values[name]

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)


class Mode:
    """Base class for a mode. Subclasses declare knobs, build, score."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""

    #: Why this mode is deliberately unscored, or empty when it is scored.
    #:
    #: A mode with no ``score()`` produces no signal, so the outer loop cannot
    #: search it and a created package containing one is refused. That is the right
    #: default. But some modes are *about* producing something rather than about
    #: maximising a number — a mode that writes a package is judged by whether the
    #: package works — and those are unscored on purpose. Saying so here turns an
    #: inference ("it happens not to override score") into a declaration the
    #: registry, the sandbox and a reader can all check.
    unscored_reason: ClassVar[str] = ""

    def __init__(self, state: dict[str, Any] | None = None) -> None:
        if state is None:
            self._knobs = self._declared_entries()
            self._ports = [port.to_state() for port in self.ports()]
            self._memory = [declared.to_state() for declared in self.memory()]
            self._graph_state = self.build().to_state()
        else:
            self._knobs: dict[str, dict[str, Any]] = {}
            self._ports: list[dict[str, Any]] = []
            self._memory: list[dict[str, Any]] = []
            self._graph_state: dict[str, Any] = {}
            self.load_state_dict(state)

    # ── the author's surface ───────────────────────────────────────────────
    @classmethod
    def declare(cls) -> dict[str, Knob]:
        """The parameters this mode exposes. Override to declare knobs."""
        return {}

    def build(self) -> Graph:
        """Construct the starting architecture. Pure: no I/O, no randomness."""
        raise NotImplementedError(f"{type(self).__name__} does not implement build()")

    def score(self, run: RunResult) -> float:
        """Turn one run into the number the outer loop maximizes.

        The default means "unscored": a mode that does not override this produces
        no signal, and the loop reports that rather than searching a constant.
        """
        return 0.0

    def invariants(self) -> tuple[Invariant, ...]:
        """Mode-declared rules the validator checks on every candidate."""
        return ()

    def edit_policy(self) -> str:
        """Which edit preset the outer loop may use on this mode."""
        return "broad"

    def insertable(self) -> tuple[Node, ...]:
        """Nodes a structural edit may insert into this mode.

        Declaring these is what makes random structural search possible at all:
        without a candidate block the optimizer has nothing legal to add, and
        inventing one would mean inserting an op that does not exist.
        """
        return ()

    def guidance(self) -> str:
        """Free text for the optimizer, for what the surface cannot say.

        A knob's description says what a knob *is*; this says what is worth doing.
        The traces eventually show that structure rarely helps on a pipeline and
        that one knob is the lever, but only after a batch has been spent finding
        out. The author already knows.
        """
        return ""

    def preferred_edits(self) -> tuple[str, ...]:
        """Edit operations worth trying first, if the author has an opinion.

        A prior, not a restriction: an optimizer is free to use anything in the
        catalogue, and a sampler uses these to weight its choice.
        """
        return ()

    def memory(self) -> tuple[Memory, ...]:
        """Namespaces this mode may remember, each with a retention policy.

        Declared rather than implicit: a cross-run prior needs a named place to
        live and a stated life, or it becomes a private convention somebody
        invented. ``run`` is scoped to one inner run, ``project`` to the project,
        ``global`` across projects.
        """
        return ()

    def ports(self) -> tuple[Port, ...]:
        """What this mode publishes, so a reader knows what a run produced.

        A port names an artifact and its media type. Without one, a run's outputs
        are inferred from file writes, which is not the same as saying what the run
        produced -- and it is what a GUI and a downstream mode read.
        """
        return ()

    # ── parameters ─────────────────────────────────────────────────────────
    def _declared_entries(self) -> dict[str, dict[str, Any]]:
        return {
            name: {"value": knob.default, "spec": knob.spec()}
            for name, knob in self.declare().items()
        }

    @property
    def k(self) -> _Values:
        """Current parameter values, by name: ``self.k.grid``."""
        return _Values(self.values())

    def values(self) -> dict[str, Any]:
        return {name: entry["value"] for name, entry in self._knobs.items()}

    def parameters(self) -> dict[str, Knob]:
        """The write surface: what an optimizer is allowed to move."""
        return {name: Knob.from_spec(entry["spec"]) for name, entry in self._knobs.items()}

    def knob(self, name: str) -> Knob:
        try:
            return Knob.from_spec(self._knobs[name]["spec"])
        except KeyError as exc:
            raise KnobError(f"no knob {name!r} on {type(self).__name__}") from exc

    def knob_names(self) -> list[str]:
        return sorted(self._knobs)

    def is_scored(self) -> bool:
        """Whether this mode overrides :meth:`score`."""
        return type(self).score is not Mode.score

    # ── structure ──────────────────────────────────────────────────────────
    def graph(self) -> Graph:
        """The architecture as a runnable graph."""
        return Graph.from_state(self._graph_state)

    # ── running ────────────────────────────────────────────────────────────
    def run(
        self,
        workdir: Path,
        *,
        model: Any = None,
        task: Mapping[str, Any] | None = None,
        memory: Any = None,
        **kwargs: Any,
    ) -> RunResult:
        """Execute the built graph once: this is the forward pass."""
        from .executor import execute

        return execute(
            self.graph(), Path(workdir), knobs=self.values(),
            model=model, task=task, memory=memory, **kwargs,
        )

    @classmethod
    def resume(
        cls, workdir: Path, *, model: Any = None, max_steps: int = 2000,
        resource_limits: Any = None,
    ) -> "RunResult":
        """Continue a run where it failed, from the record it left in its directory.

        An authoring run's artifacts persist per node, so the expensive part of a
        failed run is the nodes that already ran. This reads
        ``.refactory/progress.json`` — the node that failed, the facts, the knobs and
        the task — and starts there. A missing record is refused by name rather than
        silently replaying the whole graph.
        """
        import json

        from .executor import PROGRESS, execute

        path = Path(workdir) / PROGRESS
        if not path.exists():
            raise ValueError(
                f"no run to resume: {path} does not exist, so there is no record of "
                "where the run got to"
            )
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not readable JSON: {exc}") from exc
        node = str(record.get("node") or "").strip()
        if record.get("outcome") in ("running", "unknown"):
            raise ValueError(
                f"node {node!r} has an unknown side-effect outcome; inspect its artifacts "
                "before an explicit restart; automatic replay refused"
            )
        branches = record.get("branches") or {}
        if any(branch.get("status") not in ("queued", "not-started") for branch in branches.values()):
            raise ValueError(
                "partial fork recovery needs per-branch outcome review; automatic replay "
                "would repeat completed or unknown side effects and is refused"
            )
        if not node:
            raise ValueError(f"{path} names no node to resume at")
        if record.get("error"):
            # The recorded node failed; resume *at* it, because its work is what is
            # missing. A record with no error means the run finished or was stopped
            # between nodes, and the recorded node is the one still due.
            pass
        mode = cls()
        state = mode.state_dict()
        for name, value in (record.get("knobs") or {}).items():
            if name in state.get("knobs", {}):
                state["knobs"][name]["value"] = value
        mode.load_state_dict(state)
        from refactory_dsh.steering import for_run

        graph = for_run(mode.graph(), Path(workdir))
        expected_graph = record.get("graph")
        current_graph = graph.to_state()
        if record.get("facts", {}).get("create_fidelity") and expected_graph is not None:
            from refactory_dsh.steering import authored_graph_state
            expected_graph = authored_graph_state(expected_graph)
            current_graph = authored_graph_state(current_graph)
        if expected_graph is not None and expected_graph != current_graph:
            raise ValueError("graph or instructions changed; dependent results are invalid; start a new run")
        source_manifest = Path(workdir) / "source-manifest.json"
        if source_manifest.exists():
            source = json.loads(source_manifest.read_text())
            if source.get("source_project"):
                from refactory_dsh.fast_ops import _source_manifest

                if _source_manifest(Path(source["source_project"])) != source.get("files"):
                    raise ValueError("source project changed; dependent results are invalid; start a new run")
        from refactory_dsh.budgets import LEDGER, ResourceLimits, RunBudget

        budget = RunBudget.inherited(model)
        # Direct Python/legacy CLI authoring resumes use their recorded routes,
        # just as interactive/console recovery does. Later project edits are not
        # permission to change the identity of completed work.
        if record.get("model_context"):
            from refactory_dsh.model import DshModelClient
            model = DshModelClient(**record["model_context"])
            if budget is not None:
                model = model.with_budget(budget)
        owns_budget = budget is None and (Path(workdir) / LEDGER).exists()
        if owns_budget:
            budget = RunBudget.from_path(Path(workdir) / LEDGER)
            limits = resource_limits if isinstance(resource_limits, ResourceLimits) else ResourceLimits.from_dict(resource_limits) if resource_limits is not None else None
            budget.resume(limits)
            if callable(getattr(model, "with_budget", None)):
                model = model.with_budget(budget)
            resource_limits = budget.limits
        try:
            return execute(
                mode.graph(), Path(workdir), knobs=mode.values(), model=model,
                facts=dict(record.get("facts") or {}), task=dict(record.get("task") or {}),
                start=node, max_steps=max_steps, resource_limits=resource_limits,
            )
        finally:
            if owns_budget and budget is not None:
                budget.pause()

    def evaluate(
        self,
        workdir: Path,
        *,
        model: Any = None,
        task: Mapping[str, Any] | None = None,
        memory: Any = None,
        **kwargs: Any,
    ) -> RunResult:
        """Run once and put :meth:`score` on the result.

        A run that did not complete is left at score 0 with its error recorded, so
        "ran and scored badly" and "never ran" stay distinguishable. A ``score()``
        that raises is a failed run too, rather than a crash of the loop.
        """
        result = self.run(workdir, model=model, task=task, memory=memory, **kwargs)
        if not result.exit_ok:
            return result
        try:
            result.score = float(self.score(result))
        except Exception as exc:
            result.exit_ok = False
            result.error = f"score() raised {type(exc).__name__}: {exc}"
        return result

    # ── state ──────────────────────────────────────────────────────────────
    def port_declarations(self) -> tuple[Port, ...]:
        """The ports as they stand, from the state."""
        return tuple(Port.from_state(entry) for entry in self._ports)

    def memory_declarations(self) -> tuple[Memory, ...]:
        """The memory namespaces as they stand, from the state."""
        return tuple(Memory.from_state(entry) for entry in self._memory)

    def state_dict(self) -> dict[str, Any]:
        """The whole state: architecture, parameters, and the declared surface.

        ``ports`` and ``memory`` travel with it because a run has to act on them --
        a port says what to publish, a memory says what may be kept and for how
        long -- and because the validator checks the ports against the graph.
        """
        return {
            "version": STATE_VERSION,
            "graph": copy.deepcopy(self._graph_state),
            "knobs": copy.deepcopy(self._knobs),
            "ports": copy.deepcopy(self._ports),
            "memory": copy.deepcopy(self._memory),
        }

    def load_state_dict(self, state: Mapping[str, Any], *, strict: bool = True) -> None:
        """Adopt ``state``. With ``strict``, anything unexpected is an error.

        Strictness is the detector for "the optimizer wrote something it should
        not have": an unknown node field, an unknown knob, an unknown entry key
        and an out-of-bounds value all fail here rather than quietly running.
        """
        if not isinstance(state, Mapping):
            raise StateError(f"a state dict must be a mapping, got {type(state).__name__}")
        unknown = sorted(set(state) - _TOP_LEVEL)
        if unknown and strict:
            raise StateError(f"unknown state dict field(s): {unknown}")
        version = state.get("version")
        if version != STATE_VERSION:
            raise StateError(
                f"state version {version!r} is not {STATE_VERSION}; "
                "this state was written by a different layout"
            )
        if "graph" not in state:
            raise StateError("a state dict needs a 'graph'")

        try:
            graph = Graph.from_state(state["graph"])
        except GraphError as exc:
            raise StateError(f"the graph is not readable: {exc}") from exc

        knobs: dict[str, dict[str, Any]] = {}
        for name, entry in (state.get("knobs") or {}).items():
            if not isinstance(entry, Mapping) or "spec" not in entry:
                raise StateError(f"knob {name!r} needs a 'value' and a 'spec'")
            extra = sorted(set(entry) - _KNOB_ENTRY)
            if extra and strict:
                raise StateError(f"knob {name!r} has unknown field(s): {extra}")
            knob = Knob.from_spec(dict(entry["spec"]))
            try:
                value = knob.coerce(entry.get("value", knob.default))
            except KnobError as exc:
                raise StateError(f"knob {name!r}: {exc}") from exc
            knobs[name] = {"value": value, "spec": knob.spec()}

        self._ports = [Port.from_state(entry).to_state() for entry in state.get("ports") or ()]
        self._memory = [
            Memory.from_state(entry).to_state() for entry in state.get("memory") or ()
        ]
        self._graph_state = graph.to_state()
        self._knobs = knobs

    def clone(self) -> "Mode":
        """A fresh instance holding a copy of this state."""
        return type(self)(state=copy.deepcopy(self.state_dict()))

    def describe(self) -> dict[str, Any]:
        """A summary for the optimizer's prompt: surface, structure, score."""
        graph = self.graph()
        knobs: dict[str, dict[str, Any]] = {}
        for name, entry in sorted(self._knobs.items()):
            knob = Knob.from_spec(entry["spec"])
            low, high = knob.range()
            knobs[name] = {
                "value": entry["value"],
                "default": knob.default,
                "kind": knob.kind,
                # Named separately: an enum has members, a str knob may declare
                # values, and a numeric knob is a range. One overloaded "bounds"
                # field made an optimizer author guess which they were looking at.
                "values": list(knob.accepted()),
                "range": [low, high],
                "frozen": knob.frozen,
                "description": knob.description,
            }
        return {
            "mode": type(self).__name__,
            "knobs": knobs,
            "nodes": [
                {
                    "id": node.id,
                    "kind": node.kind,
                    "role": node.role,
                    "call": node.call or node.evaluator,
                    "knobs": list(node.knobs),
                    "notes": node.notes,
                    # Who answers a gate: `fn` runs its evaluator, `user` is put to a
                    # person, `agent` to the running agent. Named for a gate only --
                    # every other kind defaults to `fn` and it would say nothing.
                    **({"gate_kind": node.gate_kind} if node.kind == "gate" else {}),
                }
                for node in (graph.nodes[nid] for nid in sorted(graph.nodes))
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "condition": edge.condition,
                }
                for edge in sorted(graph.edges, key=lambda e: e.key())
            ],
            "ports": list(self._ports),
            "memory": list(self._memory),
            "guidance": self.guidance(),
            "preferred_edits": list(self.preferred_edits()),
            "scored": self.is_scored(),
        }
