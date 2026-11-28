"""RunContext, Verdict, rendering, and op-callable decorators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from refactory_dsh import activity

from .failures import ExecutionError

# Sentinel for SWEEP-6 refilled feedback cleanup.
_SENTINEL = object()


class ModelClient(Protocol):
    """The one thing an agent node needs: text in, text out."""

    def complete(self, prompt: str, *, system: str = "") -> str: ...


@dataclass
class Step:
    """One visited node."""

    node: str
    kind: str
    duration_s: float
    verdict: str | None = None
    error: str | None = None
    #: Why a gate gave its verdict, when it said: the same reason the spine reports
    #: under ``ran``, so a headless run explains a reloop as a driven one does.
    reason: str | None = None
    #: A gate's ``proceed`` taken because it ran out of reloops (``on_exhausted:
    #: continue``), not because the check passed.
    exhausted: bool = False


class RunContext:
    """What a node's callable sees: a work directory, the knobs, and the facts.

    Facts are the run's own numbers, and they are what ``score()`` reads. A write
    is checked against the node's declaration so the graph's dataflow statements
    stay true.
    """

    def __init__(
        self,
        workdir: Path,
        *,
        knobs: dict[str, Any] | None = None,
        facts: dict[str, Any] | None = None,
        verdicts: dict[str, str] | None = None,
        model: ModelClient | None = None,
        task: dict[str, Any] | None = None,
        memory: Any = None,
        enforce_writes: bool = True,
    ) -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.knobs: dict[str, Any] = dict(knobs or {})
        self.facts: dict[str, Any] = dict(facts or {})
        #: Verdicts for gates a person or an agent answers, by node id. Empty in a
        #: headless run, which is why such a gate stops rather than deciding.
        self.verdicts: dict[str, str] = dict(verdicts or {})
        self.model = model
        from refactory_dsh.budgets import RunBudget

        self.budget = RunBudget.inherited(model)
        self.task: dict[str, Any] = dict(task or {})
        #: The mode's declared memory, for an op that has something to remember.
        self.memory = memory
        self.enforce_writes = enforce_writes
        self.iteration = 0
        self.node_id = ""
        self._declared: tuple[str, ...] = ()
        self._declared_reads: tuple[str, ...] = ()
        self._declared_facts: tuple[str, ...] = ()
        #: The name each built file is read under (`graph.Node.renames` ``reads``).
        self.read_renames: dict[str, str] = {}
        self._produced: set[str] = set()
        self.writes_made: list[str] = []
        #: A code or verdict node's declaration (`refactory_dsh.snippet.spec`): its code,
        #: inputs and out-edge labels. ``None`` for every other node.
        self.snippet: dict[str, Any] | None = None

    # ── knobs ──────────────────────────────────────────────────────────────
    def knob(self, name: str, default: Any = None) -> Any:
        return self.knobs.get(name, default)

    # ── files ──────────────────────────────────────────────────────────────
    def path(self, name: str) -> Path:
        """A file inside the work directory. Escaping it is refused."""
        candidate = Path(name)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ExecutionError(f"{name!r} is not a relative path inside the work directory")
        return self.workdir / candidate

    def _note_write(self, name: str) -> None:
        if not self.enforce_writes or not self._declared:
            return
        for declared in self._declared:
            if name == declared or name.startswith(declared.rstrip("/") + "/"):
                self.writes_made.append(name)
                return
        raise ExecutionError(
            f"node {self.node_id!r} wrote {name!r}, which it does not declare; "
            f"declared writes: {list(self._declared)}"
        )

    def _writable(self, name: str) -> Path:
        """The path a write to *name* goes to, cut loose from any run-level input
        a data pass reads through (items.detach_read_through): a pass must not
        change the run's files, whatever the node declares."""
        from refactory_dsh.items import detach_read_through

        target = self.path(name)
        detach_read_through(self.workdir, name)
        return target

    def write_text(self, name: str, text: str) -> Path:
        self._note_write(name)
        target = self._writable(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        # A watched run counts a node's writes as work (`activity`); others do nothing.
        activity.mark()
        return target

    def append_text(self, name: str, text: str) -> Path:
        self._note_write(name)
        target = self._writable(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a") as handle:
            handle.write(text)
        activity.mark()
        return target

    def read_text(self, name: str) -> str:
        self._check_item_read(name)
        return self.path(name).read_text()

    def write_json(self, name: str, payload: Any) -> Path:
        import json

        return self.write_text(name, json.dumps(payload, indent=2, default=str))

    def read_json(self, name: str, default: Any = None) -> Any:
        import json

        self._check_item_read(name)
        target = self.path(name)
        if not target.exists():
            return default
        try:
            return json.loads(target.read_text())
        except json.JSONDecodeError:
            return default

    def exists(self, name: str) -> bool:
        self._check_item_read(name)
        return self.path(name).exists()

    def _pass_item_files(self) -> dict[str, str]:
        """``{path: role}`` for the item files the Task of this item's pass lays in
        (`Task.item_files`); empty outside a pass. Read from the pass's marker."""
        import json

        from refactory_dsh.items import PASS_MARKER
        from refactory_dsh.task import TaskError, task_item_files

        marker = self.workdir / PASS_MARKER
        try:
            ref = str(json.loads(marker.read_text()).get("task") or "")
        except (OSError, ValueError, AttributeError):
            return {}
        if not ref:
            return {}
        try:
            return {path: role for role, path in task_item_files(ref).items()}
        except (TaskError, ImportError) as exc:
            raise ExecutionError(f"node {self.node_id!r}: the Task's item files cannot be "
                                 f"read: {exc}") from exc

    def _check_item_read(self, name: str) -> None:
        """Refuse a read, through this context's read helpers, of an item file the node
        does not declare in its ``reads`` (as `item_file` does). ``path`` gives a path,
        not a read, and is not checked: an op reads an item file through `item_file`."""
        files = self._pass_item_files()
        normal = Path(name).as_posix()
        if normal in files and normal not in self._declared_reads:
            raise ExecutionError(f"node {self.node_id!r} reads item file {normal!r} "
                                 f"({files[normal]}), which it does not declare; declared "
                                 f"reads: {list(self._declared_reads)}")

    def item_file(self, role: str) -> Path | None:
        """The file the item's Task laid into this pass under ``role`` (`Task.item_files`),
        or ``None`` when this item has none. Refused (`ExecutionError`) outside an item's
        pass, for a role the Task does not declare, and when this node does not list the
        file in its ``reads``: what a step takes in is what it declares, so a read around
        the declaration fails loudly instead of passing unseen. Both runtimes hand an op
        this context (`run.build_op_context`)."""
        import json

        from refactory_dsh.items import PASS_MARKER
        from refactory_dsh.task import TaskError, task_item_files

        marker = self.workdir / PASS_MARKER
        try:
            ref = str(json.loads(marker.read_text()).get("task") or "")
        except (OSError, ValueError, AttributeError):
            ref = ""
        if not ref:
            raise ExecutionError(f"node {self.node_id!r} asks for item file {role!r} outside "
                                 "an item's pass (no Task laid one in)")
        try:
            files = task_item_files(ref)
        except (TaskError, ImportError) as exc:
            raise ExecutionError(f"node {self.node_id!r}: the Task's item files cannot be "
                                 f"read: {exc}") from exc
        if role not in files:
            raise ExecutionError(f"node {self.node_id!r} asks for item file {role!r}, which "
                                 f"the Task does not lay in; it lays in: "
                                 f"{', '.join(sorted(files)) or 'none'}")
        name = files[role]
        if name not in self._declared_reads:
            raise ExecutionError(f"node {self.node_id!r} reads item file {name!r} ({role}), "
                                 f"which it does not declare; declared reads: "
                                 f"{list(self._declared_reads)}")
        target = self.path(name)
        return target if target.is_file() else None

    # ── facts ──────────────────────────────────────────────────────────────
    def set_fact(self, name: str, value: Any) -> None:
        """Set a run fact. A node may only set facts it declares in ``produces``."""
        self.facts[name] = value
        self._produced.add(name)

    def _internal_fact(self, name: str, value: Any) -> None:
        """A fact the run itself records, not one a node claims to produce."""
        self.facts[name] = value

    @property
    def declared_reads(self) -> tuple[str, ...]:
        """The current node's declared reads, for gate discovery."""
        return self._declared_reads

    def _check_produces(self) -> None:
        if not self.enforce_writes:
            return
        extra = sorted(self._produced - set(self._declared_facts))
        if extra:
            raise ExecutionError(
                f"node {self.node_id!r} produced the fact(s) {extra}, which it does "
                f"not declare; declared facts: {list(self._declared_facts)}"
            )

    # ── pass isolation ─────────────────────────────────────────────────────
    def pass_scope(self, pass_dir: Path, produces: tuple[str, ...]):
        """Context manager that isolates one data-node item pass.

        On enter: saves *workdir* and *facts*, sets ``workdir`` to *pass_dir*
        and ``facts`` to a deep copy: a pass that edits a fact in place (appends
        to a list) changes its own copy, as the spine's pass does with its own
        ``facts.json``.

        On exit: merges only keys in *produces* plus ``"item_scores"`` that the
        pass added or changed into the saved (run-level) facts, then restores
        both ``workdir`` and ``facts``.
        """
        import copy
        from contextlib import contextmanager

        @contextmanager
        def _scope():
            saved_workdir = self.workdir
            saved_facts = self.facts
            self.workdir = pass_dir
            self.facts = copy.deepcopy(saved_facts)
            try:
                yield
            finally:
                # Merge only declared produces + item_scores. The pass's copy is
                # discarded on exit, so its values can be handed over as they are.
                allowed = set(produces) | {"item_scores"}
                for key in allowed:
                    if key in self.facts:
                        if key not in saved_facts or saved_facts[key] != self.facts[key]:
                            saved_facts[key] = self.facts[key]
                self.workdir = saved_workdir
                self.facts = saved_facts

        return _scope()


class Verdict(str):
    """A gate's verdict label, carrying why the gate gave it.

    It *is* the label, so routing, comparisons and every caller that reads a
    gate's return as a string are unchanged. The node runner prints ``reason``
    before the label, which is how the spine hands a reloop's cause to the node
    the run goes back to and to the CEO driving it. A reason that opens with its
    own label (``"reloop: …"``, the form gate facts are written in) drops it, so
    a reader is not told the verdict twice.
    """

    reason: str

    def __new__(cls, label: str, reason: str = "") -> Verdict:
        verdict = super().__new__(cls, label)
        verdict.reason = reason.removeprefix(f"{label}: ")
        return verdict


def render(template: str, ctx: RunContext) -> str:
    """Fill ``{{name}}`` and evaluate ``{% if/elif/else/endif %}`` blocks.

    Deliberately not :meth:`str.format`: a prompt is prose full of braces, and a
    single stray one must not be able to make an agent node unrunnable.
    """
    import re

    values = {**ctx.knobs, **ctx.facts}

    def replace(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        if key not in values:
            return match.group(0)
        return str(values[key])

    template = _render_conditionals(template, values)
    return re.sub(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}", replace, template)


def fills_facts(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Mark an op that fills each ``{{name}}`` in the ``str`` knobs it reads (its node's
    ``knobs``) from the run's facts and knobs, as an agent's prompt is filled
    (:func:`render`). The validator reads the mark: a slot in such a knob is one of the
    op's inputs (V15, V17), and a slot in a knob nothing fills is refused (V20)."""
    fn.fills_facts = True  # type: ignore[attr-defined]
    return fn


def by_declaration(fn: Callable[..., Any] | None = None, *,
                   files: Sequence[str] | None = None) -> Any:
    """Mark an op or gate callable that opens the files its node declares and writes and
    sets its declared outputs, under the declared names (``ctx.declared_reads``, its
    declared writes and facts), never a name of its own choosing. An edit may then rename
    that node's files and outputs (``insert``'s 'takes_over', `core_ops._take_over_plan`);
    a callable without the mark is taken to open or return them by name. (A fact it reads
    is held to its ``inputs`` either way.)

    ``files`` names the files it was built to read, one per role, when it finds each
    role's file by `declared_files` (so an edit's check resolves them as its run does)."""
    def mark(target: Callable[..., Any]) -> Callable[..., Any]:
        target.by_declaration = True  # type: ignore[attr-defined]
        if files is not None:
            target.declared_files = tuple(files)  # type: ignore[attr-defined]
        return target
    return mark(fn) if fn is not None else mark


def declared_files(reads: Sequence[str], files: Sequence[str],
                   renames: Mapping[str, str] | None = None) -> list[str | None]:
    """Each role's file among a node's declared ``reads``, for a callable built to read
    ``files`` (one per role, `by_declaration`): the name the node reads that built file
    under (``renames``, its `graph.Node.renames` ``reads``; else the built name), or None
    when the node does not read it (an edit took out the step that wrote it). Never by
    position or count. No reads at all (a context with no declaration): the built names."""
    reads = list(reads)
    if not reads:
        return list(files)
    named = [(renames or {}).get(name, name) for name in files]
    return [name if name in reads else None for name in named]


def names_matter(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Mark an op or gate callable whose behaviour depends on the names of the files it
    reads, not only on what they hold (``select`` matches a label against them): an edit
    that renames one of its reads would change what it does, so ``insert``'s 'takes_over'
    refuses to (`core_ops._take_over_plan`)."""
    fn.names_matter = True  # type: ignore[attr-defined]
    return fn


def _eval_condition(expr: str, values: dict[str, Any]) -> bool:
    """Evaluate a simple conditional expression against *values*.

    Supports ``name == 'literal'``, ``name != 'literal'``, and bare ``name``
    (truthy check).  Anything else evaluates to ``False`` so the template
    degrades to the ``{% else %}`` branch rather than crashing a run.
    """
    import re

    expr = expr.strip()

    # A literal ends at the quote that opened it, so either quote may appear inside
    # the other: "don't explain", 'say "hi"'.
    m = re.fullmatch(
        r"""([A-Za-z_]\w*)\s*(==|!=)\s*(?:'([^']*)'|"([^"]*)")""", expr,
    )
    if m:
        var, op = m.group(1), m.group(2)
        lit = m.group(3) if m.group(3) is not None else m.group(4)
        val = str(values.get(var, ""))
        return (val == lit) if op == "==" else (val != lit)

    m = re.fullmatch(r"([A-Za-z_]\w*)", expr)
    if m:
        return bool(values.get(m.group(1)))

    return False


def _render_conditionals(template: str, values: dict[str, Any]) -> str:
    """Process ``{% if %}`` / ``{% elif %}`` / ``{% else %}`` / ``{% endif %}``."""
    import re

    # DOTALL: a condition may span lines, and its opening tag must be recognized
    # whenever its else/endif are, or raw syntax and every branch reach the model.
    pattern = re.compile(r"\{%[-\s]*(if|elif|else|endif)\b\s*(.*?)[-\s]*%\}", re.DOTALL)

    result: list[str] = []
    # Each frame: (emitting: is the current branch active, settled: has any
    # branch in this if/elif/else chain already been taken).
    stack: list[list[bool]] = []
    pos = 0

    def _emitting() -> bool:
        return all(frame[0] for frame in stack)

    for m in pattern.finditer(template):
        tag, expr = m.group(1), m.group(2).strip()

        if _emitting():
            result.append(template[pos:m.start()])
        pos = m.end()

        if tag == "if":
            cond = _eval_condition(expr, values)
            stack.append([cond, cond])
        elif tag == "elif" and stack:
            settled = stack[-1][1]
            cond = not settled and _eval_condition(expr, values)
            stack[-1] = [cond, settled or cond]
        elif tag == "else" and stack:
            stack[-1] = [not stack[-1][1], True]
        elif tag == "endif" and stack:
            stack.pop()

    if _emitting():
        result.append(template[pos:])
    return "".join(result)
