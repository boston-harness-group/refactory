"""The Task protocol: what a dataset mode's items are, and how one is judged.

A dataset mode iterates its subgraph once per item. Where the items come from, and
whether a pass was any good, is the Task's business rather than the graph's — the
same split remote-factory's `factory/task.py` makes, so a mode bound to a benchmark
or a dataset carries that binding as a declaration instead of as instructions in a
prompt.

A Task is three hooks, all optional:

- ``instances()`` — the items. The default is one unnamed instance, which is what a
  mode that only needs to run once wants.
- ``setup(instance, workspace)`` — prepare the workspace before the pass.
- ``verify(instance, workspace)`` — judge the pass, or return ``None`` to say the
  Task has no opinion and the mode's own score stands.

There is deliberately no hook that writes the prompt. An item is data — it reaches the
pass as ``current_item.json`` — and what the agent for an item is asked is the mode's
own business, stated in its agent node's prompt template. Two places that both decide
the ask is one place too many.

Resolution is by ``module:Class`` or by a name registered here, and a ref that does
not resolve is refused by name: a mode that names a Task nobody can load must not
silently run its items unverified.

A run directory may also say which *part* of the items it evaluates. The outer loop
writes that choice beside the run, and the loader both runtimes use reads it, so a
search is examined on the items it was tuned against and the winner can be examined
on the items no candidate saw. The choice is a file rather than a knob: a search
that could set it would take its own exam.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence


class TaskError(ValueError):
    """A Task reference does not resolve, or a Task misbehaves."""


@dataclass(frozen=True)
class TaskInstance:
    """One item a Task says the mode has to process."""

    id: str
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolEnvironment:
    """What the commands of a Task's sealed tool steps run with (`Task.tool_environment`):
    an interpreter put first on their PATH as ``python3`` and ``python``."""

    #: The facts the optimizer's request states about it, in plain words: what
    #: ``python3`` is and what it holds (``"python3 (and python) = Python 3.12 with
    #: openpyxl 3.1.5"``). Never advice.
    summary: str
    #: Returns the interpreter's path, checked as the Task checks it wherever else it
    #: runs; raises when there is none or it is not the one declared.
    python: Callable[[], str]
    #: Given the item's pass, the grader-side paths (the dataset, its answers, secrets)
    #: that no folder of the environment may hold or lie inside, beside the run's roots and
    #: refactory's own directory, which are always refused (`seal.tool_path`). It must list
    #: the Task's grader-side data whenever the interpreter could sit near it: the seal
    #: binds the folders the interpreter reports running from, so an interpreter installed
    #: inside the dataset (``<x>/dataset/python``) would show the dataset unless it is
    #: named here. None: nothing more.
    hidden: Callable[[Path], Sequence[str]] | None = None


@dataclass(frozen=True)
class VerifyResult:
    """What one pass was worth."""

    passed: bool
    score: float
    details: str = ""
    prediction: str = ""


class Task:
    """The three hooks a dataset binding implements."""

    name = ""
    #: Item fields only the grader may see: the answer, and anything that states it.
    #: ``verify`` gets the whole item (its instance), but the per-item file the
    #: graph's nodes read (``current_item.json``) never carries these, on either
    #: runtime (`node_item`). A node an optimizer adds can read that file, so a Task
    #: whose items hold their own answer has to say so here. Default: none.
    grader_fields: tuple[str, ...] = ()
    #: Facts or files ``verify`` (or a scorer the Task runs) leaves where the graph's
    #: nodes could find them: grader-derived, so no node outside the grader side may
    #: consume them (V17). Default: none.
    grader_outputs: tuple[str, ...] = ()
    #: The files ``verify`` reads in an item's workspace: what is scored. A step that takes
    #: over another's outputs reaches the score by writing one of these when nothing after
    #: it reads them (`nn.core_ops`). Default: none declared, so a take-over of any name
    #: is taken (the Task may score it).
    graded_files: tuple[str, ...] = ()
    #: The files ``setup`` lays into each item's pass, by role: ``{role: path}``, each
    #: path relative to the pass and the same for every item (``item/input.xlsx``: keep
    #: the extension a reader needs). The data node declares each in its ``writes`` (rule
    #: V12), so any step of the pass, a sealed one included, can list it in ``reads`` and
    #: is given it; an op opens one through ``ctx.item_file(role)``, which refuses a file
    #: the op does not declare. Only what the item gives its solver belongs here, never
    #: its answer: a path the grader reads or writes is refused (`checked_item_files`),
    #: and ``setup`` must lay each as a regular file (`check_laid_item_files`). An item
    #: without one (it has no such input) leaves it out. Default: none.
    item_files: Mapping[str, str] = MappingProxyType({})
    #: What the commands of a sealed tool step in an item's pass run with
    #: (`ToolEnvironment`): its interpreter first on their PATH, its directories readable
    #: inside the seal, read only (`seal.tool_path`). Both runtimes read it from the class
    #: of the pass's Task (`pass_tool_environment`). Default: none, so such a command runs
    #: the machine's own programs only.
    tool_environment: ToolEnvironment | None = None
    #: What the integrity judge (`optim.integrity`, ``LoopConfig.integrity_judge``) is to
    #: treat as legitimate for this task: ``{category: why, in plain words}``, e.g.
    #: ``{"answer_leak": "the task is to recall facts about these items"}``. A category
    #: named here is off for the run (not asked about) unless the run's own setting
    #: names it, and the judge is told the reason. Default: none.
    integrity_allowances: Mapping[str, str] = MappingProxyType({})
    #: The task's policy per integrity category (``off``, ``warn`` or ``veto``), where
    #: it differs from the run's default; a run's own setting for a category wins.
    #: Default: none.
    integrity_policy: Mapping[str, str] = MappingProxyType({})
    #: What the optimizer's analysts are shown in each item's header, as SkillOpt's
    #: analysts see each item's task type: the names `item_meta` returns, a category of
    #: the item, never its answer. A name also in ``grader_fields`` is never shown.
    #: Default: none.
    item_metadata: tuple[str, ...] = ()

    def item_meta(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        """The values ``item_metadata`` names for one item, from its metadata: by default
        the item's own fields of those names (a Task may derive them instead)."""
        return {name: metadata[name] for name in self.item_metadata
                if metadata.get(name) not in (None, "", [], {})}

    def instances(
        self, *, part: str | None = None, seed: int | None = None,
    ) -> list[TaskInstance]:
        """The items to process. One unnamed instance unless the Task says otherwise.

        ``part`` and ``seed`` are optional: a Task that depends on neither
        ignores them (backward compatible).  A Task whose items vary by seed
        overrides this and uses them.
        """
        return [TaskInstance(id="default")]

    def setup(
        self, instance: TaskInstance, workspace: Path,
        *, part: str | None = None, seed: int | None = None,
    ) -> None:
        """Prepare the workspace for one item. Nothing to do by default."""

    def verify(
        self, instance: TaskInstance, workspace: Path,
        *, part: str | None = None, seed: int | None = None,
    ) -> VerifyResult | None:
        """Judge one pass, or ``None`` when the Task has no opinion."""
        return None

    def train_feedback(
        self, instance: TaskInstance, workspace: Path,
        *, part: str | None = None, seed: int | None = None,
    ) -> Any:
        """What the item's attempt so far is told, judged against its answer: a JSON value,
        or ``None`` for nothing. Called mid-pass, and only on a train evaluation the outer
        loop asked to give gold feedback (`gold_feedback_on`), by the one grader-side op
        that may hand grader-derived data to the pass (`refactory_dsh.feedback`). Never
        called on val or test. Nothing by default.

        A value saying the attempt is wrong must be a mapping whose ``passed`` is
        ``False``: an item told that once scores 0 on this evaluation, however its later
        turns end (`feedback.SCORING`). Any other value is not counted."""
        return None

    def reply_correct(self, reply: str, metadata: Mapping[str, Any]) -> bool | None:
        """Whether ``reply``, taken as the item's answer, is what this Task would grade
        correct (``metadata``: the item's record metadata); ``None`` when a reply cannot be
        read as an answer. The optimizer reads a step's replies with it -- was a step's
        first reply already right before a judge sent it back? -- and never edits by it.
        Default: no reading."""
        return None


#: The Tasks a bare name can resolve to.
TASKS: dict[str, type[Task]] = {}


def register_task(name: str, cls: type[Task]) -> None:
    """Make a Task resolvable by name, and return nothing."""
    TASKS[name] = cls


def load_task(ref: str, project: Path | None = None) -> Task:
    """Resolve a Task reference to an instance.

    ``module:Class`` imports it; a bare name is looked up in :data:`TASKS`. Anything
    else is refused with the ref in the message, because a mode bound to a Task that
    cannot be loaded would run its items with nothing judging them.
    """
    ref = (ref or "").strip()
    if not ref:
        raise TaskError("an empty Task reference names no Task")
    if ":" in ref:
        module_name, _, attribute = ref.partition(":")
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise TaskError(f"cannot import {module_name!r} for Task {ref!r}: {exc}") from exc
        cls = getattr(module, attribute, None)
        if cls is None:
            raise TaskError(f"{module_name!r} has no attribute {attribute!r} (Task {ref!r})")
        if not isinstance(cls, type) or not issubclass(cls, Task):
            raise TaskError(f"{ref!r} is not a Task subclass")
        return cls()
    if ref not in TASKS:
        known = ", ".join(sorted(TASKS)) or "none"
        raise TaskError(f"no Task named {ref!r} is registered; registered: {known}")
    return TASKS[ref]()


def grader_outputs(ref: str) -> tuple[str, ...]:
    """A Task's ``grader_outputs``, read from its class without building it (a Task may
    load its items when built). ``()`` when the reference names no Task."""
    ref = (ref or "").strip()
    if ":" in ref:
        module_name, _, attribute = ref.partition(":")
        cls = getattr(importlib.import_module(module_name), attribute, None)
    else:
        cls = TASKS.get(ref)
    if not isinstance(cls, type) or not issubclass(cls, Task):
        return ()
    return tuple(str(name) for name in (getattr(cls, "grader_outputs", ()) or ()))


#: What an item file's role may be called: a word.
_ROLE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _task_class(ref: str) -> type[Task] | None:
    """The Task class a reference names, without building it (a Task may load its items
    when built); ``None`` when it names none. An import that fails raises."""
    ref = (ref or "").strip()
    if ":" in ref:
        module_name, _, attribute = ref.partition(":")
        cls = getattr(importlib.import_module(module_name), attribute, None)
    else:
        cls = TASKS.get(ref)
    return cls if isinstance(cls, type) and issubclass(cls, Task) else None


def _folded(text: str) -> str:
    """``text`` as names are compared: NFKC, without invisible format characters, and
    case-folded, so ``Answer.txt`` or a full-width form names ``answer.txt``."""
    kept = "".join(c for c in unicodedata.normalize("NFKC", text)
                   if unicodedata.category(c) != "Cf")
    return kept.casefold()


def checked_item_files(task: Task | type[Task]) -> dict[str, str]:
    """``task``'s item files (`Task.item_files`) as ``{role: path}``, each checked: a role
    is a word; a path is relative, normalised and inside the pass (no ``..``, no ``.``
    part, not the folder of another item file), outside the runtime's own records
    (``.refactory/``, the pass's read-through and grading files, ``current_item.json``),
    and none of the files the grader reads or writes (``graded_files``,
    ``grader_outputs``). Anything else raises `TaskError`: an item file is what the item
    gives its solver, never what it is graded on."""
    from refactory_dsh.items import FACTS, GRADER_FILES, PASS_INFRA
    from refactory_dsh.nn.executor import CURRENT_ITEM

    # One declaration for the class and every instance, read the one way: the validator
    # and the view read the class, the runtimes an instance.
    cls = task if isinstance(task, type) else type(task)
    name = getattr(cls, "name", "") or cls.__name__
    if not isinstance(task, type) and any(
            key in vars(task) for key in ("item_files", "graded_files", "grader_outputs")):
        raise TaskError(f"Task {name!r} sets item_files, graded_files or grader_outputs on an "
                        "instance; they are class attributes, read the same everywhere")
    declared = getattr(cls, "item_files", None) or {}
    if not isinstance(declared, Mapping):
        raise TaskError(f"Task {name!r} item_files is {{role: path}}, not {declared!r}")
    grader = {_folded(str(n)) for n in (*(getattr(cls, "graded_files", ()) or ()),
                                        *(getattr(cls, "grader_outputs", ()) or ()))}
    runtime = {_folded(n) for n in (CURRENT_ITEM, FACTS, *GRADER_FILES, *PASS_INFRA)}
    out: dict[str, str] = {}
    for role, path in declared.items():
        if not isinstance(role, str) or not _ROLE.fullmatch(role):
            raise TaskError(f"Task {name!r} item_files: role {role!r} is not a word")
        if not isinstance(path, str) or not path.strip():
            raise TaskError(f"Task {name!r} item_files[{role!r}] is not a path")
        pure = PurePosixPath(path)
        if (pure.is_absolute() or str(pure) != path or "\\" in path
                or any(part in ("", ".", "..") for part in pure.parts)):
            raise TaskError(f"Task {name!r} item_files[{role!r}] = {path!r} is not a normalised "
                            "path inside the item's pass")
        if any(unicodedata.category(c) in ("Cc", "Cf") for c in path):
            raise TaskError(f"Task {name!r} item_files[{role!r}] = {path!r} holds a control or "
                            "invisible character")
        folded = _folded(path)
        if _folded(pure.parts[0]) == ".refactory" or folded in runtime:
            raise TaskError(f"Task {name!r} item_files[{role!r}] = {path!r} is one of the "
                            "runtime's own files")
        if folded in grader or _folded(pure.name) in grader:
            raise TaskError(f"Task {name!r} item_files[{role!r}] = {path!r} is a file the grader "
                            "reads or writes (graded_files, grader_outputs): an item file is "
                            "what the item gives its solver, never what it is graded on")
        out[role] = path
    paths = sorted(_folded(p) for p in out.values())
    for index, path in enumerate(paths):
        if path in paths[index + 1:] or any(other.startswith(path + "/") for other in paths):
            raise TaskError(f"Task {name!r} item_files: {path!r} is named twice or is the "
                            "folder of another item file")
    return out


def task_item_files(ref: str) -> dict[str, str]:
    """The item files (`checked_item_files`) of the Task ``ref`` names, read from its class;
    ``{}`` when it names none. A declaration that fails the check raises `TaskError`."""
    cls = _task_class(ref)
    return checked_item_files(cls) if cls is not None else {}


def task_tool_environment(ref: str) -> ToolEnvironment | None:
    """The `Task.tool_environment` of the Task ``ref`` names, read from its class; ``None``
    when it names none or declares none. A declaration of anything else raises
    `TaskError`."""
    cls = _task_class(ref)
    declared = getattr(cls, "tool_environment", None) if cls is not None else None
    if declared is None:
        return None
    if not isinstance(declared, ToolEnvironment) or not callable(declared.python):
        raise TaskError(f"Task {ref!r} tool_environment is a ToolEnvironment, not {declared!r}")
    return declared


def pass_tool_environment(workdir: Path) -> tuple[str, ToolEnvironment] | None:
    """``(task ref, its tool environment)`` for the item's pass ``workdir`` is, read from
    the pass's marker (`items.PASS_MARKER`) as both runtimes write it; ``None`` outside a
    pass or when its Task declares none."""
    from refactory_dsh.items import PASS_MARKER

    try:
        ref = str(json.loads((Path(workdir) / PASS_MARKER).read_text()).get("task") or "")
    except (OSError, ValueError, AttributeError):
        return None
    environment = task_tool_environment(ref) if ref else None
    return (ref, environment) if environment is not None else None


def check_laid_item_files(task: Task | None, workspace: Path) -> None:
    """After ``setup``: each item file ``task`` declares that is there is a regular file of
    the pass, reached through no link, so a step that reads it reads what the Task laid
    in and nothing it points at. Both runtimes call this after every ``setup``; anything
    else raises `TaskError`, which fails the item."""
    import os
    import stat

    if task is None:
        return
    root = Path(workspace)
    for role, path in checked_item_files(task).items():
        target = root / path
        if not os.path.lexists(target):
            continue
        current = root
        for part in PurePosixPath(path).parts:
            current = current / part
            mode = os.lstat(current).st_mode
            if stat.S_ISLNK(mode) or (current != target and not stat.S_ISDIR(mode)):
                raise TaskError(f"item file {path!r} ({role}) is reached through a link: "
                                "setup must lay it into the pass as a file of its own")
        found = os.lstat(target)
        if not stat.S_ISREG(found.st_mode):
            raise TaskError(f"item file {path!r} ({role}) is not a regular file")
        if found.st_nlink > 1:
            # A hard link shares its content with a file outside the pass (the answer's,
            # say): a copy of its own is what a step may read.
            raise TaskError(f"item file {path!r} ({role}) is a hard link to another file: "
                            "setup must lay it into the pass as a file of its own")


def graded_files(state: Mapping[str, Any]) -> tuple[str, ...] | None:
    """The files the state's Tasks score (``Task.graded_files``), or ``None`` when one of
    its data nodes' Tasks declares none (any file a pass leaves may then be scored)."""
    nodes = ((state.get("graph") or {}).get("nodes") or {}).values()
    refs = sorted({str(n.get("task_ref")) for n in nodes
                   if n.get("kind") == "data" and n.get("task_ref")})
    out: list[str] = []
    for ref in refs:
        module_name, _, attribute = ref.partition(":")
        try:
            cls = (getattr(importlib.import_module(module_name), attribute, None) if attribute
                   else TASKS.get(ref))
        except ImportError:
            return None
        declared = tuple(getattr(cls, "graded_files", ()) or ()) if isinstance(cls, type) else ()
        if not declared:
            return None
        out += [str(name) for name in declared]
    return tuple(out) if refs else None


def grader_secrets(state: dict[str, Any], *, part: str | None = None) -> set[str]:
    """The values only the grader may see, for content search: for each data node's
    Task, each item's grader-only field values (strings of 8+ characters, and the strings
    inside a mapping or a list) that no node-visible field of the same item also holds.
    ``part`` picks the items (``Task.instances(part=...)``); a Task that cannot be loaded
    contributes nothing. The outer loop refuses code that contains one (``E-secret``)."""
    from refactory_dsh.nn.graph import Graph

    def strings(value: Any) -> set[str]:
        if isinstance(value, str):
            return {value} if len(value) >= 8 else set()
        if isinstance(value, dict):
            return set().union(*(strings(v) for v in value.values())) if value else set()
        if isinstance(value, (list, tuple)):
            return set().union(*(strings(v) for v in value)) if value else set()
        return set()

    secrets: set[str] = set()
    graph = Graph.from_state(state.get("graph") or {})
    refs = sorted({n.task_ref for n in graph.nodes.values() if n.kind == "data" and n.task_ref})
    for ref in refs:
        try:
            task = load_task(ref)
            instances = task.instances(part=part) if part else task.instances()
        except Exception:  # noqa: BLE001 - an unloadable Task is another check's to report
            continue
        hidden = set(getattr(task, "grader_fields", ()) or ())
        for instance in instances:
            visible = strings({k: v for k, v in instance.metadata.items() if k not in hidden})
            for name in hidden:
                secrets |= strings(instance.metadata.get(name)) - visible
    return secrets


def offers_train_feedback(state: dict[str, Any]) -> bool:
    """Whether a data node's Task of ``state`` says what a train attempt is told
    (it overrides `Task.train_feedback`). The default loop then asks for gold feedback
    on its train evaluations, as SkillOpt's train rollouts do. A Task that cannot be
    loaded raises: the run cannot say what its passes are told."""
    from refactory_dsh.nn.graph import Graph

    graph = Graph.from_state(state.get("graph") or {})
    refs = sorted({n.task_ref for n in graph.nodes.values() if n.kind == "data" and n.task_ref})
    return any(type(load_task(ref)).train_feedback is not Task.train_feedback for ref in refs)


def integrity_declaration(state: dict[str, Any]) -> dict[str, dict[str, str]]:
    """What the graph's Tasks declare for the integrity judge: ``{"policy": {...},
    "allowances": {...}}``, from each data node's Task (``Task.integrity_policy``,
    ``Task.integrity_allowances``). Where two Tasks disagree on a category the more
    lenient policy holds (off, then warn, then veto), and an allowance of either beats
    any policy (`optim.integrity.resolve_policy`); allowances are joined, each reason
    kept. A Task that cannot be loaded declares nothing and is warned about; an unknown
    category or policy word raises ``TaskError``."""
    import warnings

    from refactory_dsh.nn.graph import Graph
    from refactory_dsh.optim.integrity import CATEGORIES, check_policy

    order = {"off": 0, "warn": 1, "veto": 2}
    policy: dict[str, str] = {}
    allowances: dict[str, str] = {}
    graph = Graph.from_state(state.get("graph") or {})
    refs = sorted({n.task_ref for n in graph.nodes.values() if n.kind == "data" and n.task_ref})
    for ref in refs:
        try:
            task = load_task(ref)
        except Exception as exc:  # noqa: BLE001 - an unloadable Task is another check's to report
            warnings.warn(f"integrity judge: Task {ref!r} could not be loaded, so its "
                          f"declaration is not read ({type(exc).__name__}: {exc})",
                          RuntimeWarning, stacklevel=2)
            continue
        try:
            declared = check_policy(dict(getattr(task, "integrity_policy", None) or {}),
                                    f"Task {ref!r} integrity_policy")
        except ValueError as exc:
            raise TaskError(str(exc)) from exc
        for name, level in declared.items():
            if name not in policy or order[level] < order[policy[name]]:
                policy[name] = level
        unknown = set(dict(getattr(task, "integrity_allowances", None) or {})) - set(CATEGORIES)
        if unknown:
            raise TaskError(f"Task {ref!r} integrity_allowances: unknown categories "
                            f"{', '.join(sorted(unknown))}; known: {', '.join(CATEGORIES)}")
        for name, text in dict(getattr(task, "integrity_allowances", None) or {}).items():
            if str(text).strip():
                allowances[str(name)] = (f"{allowances[name]}; {text}" if name in allowances
                                         else str(text))
    return {"policy": policy, "allowances": allowances}


def item_metadata_reader(state: dict[str, Any]) -> Any:
    """How a train record's declared metadata is read (``Task.item_metadata``,
    ``Task.item_meta``) for the graph's Tasks: a function of a record returning
    ``{name: value}``, never a name any Task lists in ``grader_fields``; ``None`` when no
    Task declares any. A Task that cannot be loaded declares nothing."""
    from refactory_dsh.nn.graph import Graph

    try:
        graph = Graph.from_state(state.get("graph") or {})
    except Exception:  # noqa: BLE001 - a graph that cannot be built declares nothing
        return None
    refs = sorted({n.task_ref for n in graph.nodes.values() if n.kind == "data" and n.task_ref})
    tasks: list[Task] = []
    hidden: set[str] = {"id", "path"}
    for ref in refs:
        try:
            task = load_task(ref)
        except Exception:  # noqa: BLE001 - an unloadable Task is another check's to report
            continue
        hidden |= {str(name) for name in getattr(task, "grader_fields", ()) or ()}
        if getattr(task, "item_metadata", ()):
            tasks.append(task)
    if not tasks:
        return None

    def read(record: dict[str, Any]) -> dict[str, Any]:
        meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        out: dict[str, Any] = {}
        for task in tasks:
            try:
                found = task.item_meta(meta)
            except Exception:  # noqa: BLE001 - a header is a nicety, never a failure
                continue
            for name, value in (found or {}).items():
                if (str(name) not in hidden and str(name) in task.item_metadata
                        and value not in (None, "", [], {})):
                    out.setdefault(str(name), value)
        return out

    return read


def reply_judge(state: dict[str, Any]) -> Any:
    """How a step's reply is read as an item's answer for the graph's Tasks
    (`Task.reply_correct`): a function ``(reply, metadata) -> bool | None``, the first
    Task with a reading deciding; ``None`` when none has one. A graph that cannot be
    built or a Task that cannot be loaded raises: the run could not evaluate it either.
    A reading that raises raises too; its caller says so (`optim.edit_effect`)."""
    from refactory_dsh.nn.graph import Graph

    graph = Graph.from_state(state.get("graph") or {})
    tasks = []
    for ref in sorted({n.task_ref for n in graph.nodes.values() if n.kind == "data" and n.task_ref}):
        task = load_task(ref)
        if type(task).reply_correct is not Task.reply_correct:
            tasks.append(task)
    if not tasks:
        return None

    def judge(reply: str, metadata: Mapping[str, Any]) -> bool | None:
        for task in tasks:
            verdict = task.reply_correct(reply, metadata)
            if verdict is not None:
                return bool(verdict)
        return None

    return judge


def node_item(item: dict[str, Any], task: Task | None) -> dict[str, Any]:
    """The item as the graph's nodes may read it: without the Task's grader-only fields.

    Both runtimes write ``current_item.json`` through here; the grader keeps the whole
    item (``instance_of``).
    """
    hidden = set(getattr(task, "grader_fields", ()) or ())
    if not hidden:
        return item
    return {key: value for key, value in item.items() if key not in hidden}


def instance_of(item: dict[str, Any]) -> TaskInstance:
    """The Task instance one loaded item describes."""
    metadata = {
        key: value for key, value in item.items() if key not in ("id", "path")
    }
    return TaskInstance(
        id=str(item.get("id") or ""),
        path=str(item["path"]) if item.get("path") else None,
        metadata=metadata,
    )


# ── the part of the items a run directory evaluates ────────────────────────

#: The file in a run directory that names the part of the items it evaluates.
SUBSET = "subset.json"

#: The parts an item selection can name.
TRAIN = "train"
HOLDOUT = "holdout"
VAL = "val"
TEST = "test"

#: The holdout fraction used when a run directory names a part but no fraction.
DEFAULT_HOLDOUT = 0.3

#: An explicit id-list manifest for three-way partitioning.
MANIFEST = "parts.json"


def _item_key(item: dict[str, Any], index: int) -> str:
    """The id an item is partitioned by, falling back to its position."""
    return str(item.get("id") or index)


def partition(
    items: Sequence[dict[str, Any]], holdout: float
) -> dict[str, list[dict[str, Any]]]:
    """Split items into a train part and a holdout part, by sorted instance id.

    The partition is deterministic, so both runtimes and the audit agree about which
    items are which. The ids are sorted, the last ``holdout`` fraction of them is
    held out, and at least one item stays in train whenever there are two. Fewer than
    two items cannot be split at all, and the holdout part is then empty rather than
    an error: one instance is a fact about the Task, not a reason to fail a run.
    """
    fraction = min(max(float(holdout), 0.0), 1.0)
    order = sorted(
        range(len(items)), key=lambda index: (_item_key(items[index], index), index)
    )
    if len(order) < 2 or fraction <= 0.0:
        return {TRAIN: list(items), HOLDOUT: []}
    count = min(len(order) - 1, max(1, int(round(len(order) * fraction))))
    held = set(order[-count:])
    return {
        TRAIN: [item for index, item in enumerate(items) if index not in held],
        HOLDOUT: [item for index, item in enumerate(items) if index in held],
    }


def counts(items: Sequence[dict[str, Any]], holdout: float) -> dict[str, int]:
    """How many items each part holds, without materialising either part."""
    parts = partition(items, holdout)
    return {TRAIN: len(parts[TRAIN]), HOLDOUT: len(parts[HOLDOUT])}


def _subset_file(workdir: Path) -> Path:
    """Where a run directory records the part of the items it evaluates."""
    return Path(workdir) / SUBSET


def make_manifest(
    workdir: Path,
    *,
    train_ids: list[str],
    val_ids: list[str] | None = None,
    test_ids: list[str] | None = None,
) -> Path:
    """Write an explicit id-list manifest for three-way partitioning."""
    manifest: dict[str, list[str]] = {"train": list(train_ids)}
    if val_ids is not None:
        manifest["val"] = list(val_ids)
    if test_ids is not None:
        manifest["test"] = list(test_ids)
    path = Path(workdir) / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return path


def split_to_manifest(
    workdir: Path,
    items: Sequence[dict[str, Any]],
    *,
    holdout: float = DEFAULT_HOLDOUT,
    val_fraction: float = 0.0,
) -> Path:
    """Partition items by fractions and write the resulting id lists as a manifest."""
    parts = partition(items, holdout)
    train_items = parts[TRAIN]
    test_items = parts[HOLDOUT]
    train_ids = [str(item.get("id") or i) for i, item in enumerate(items) if item in train_items]
    test_ids = [str(item.get("id") or i) for i, item in enumerate(items) if item in test_items]
    val_ids: list[str] | None = None
    if val_fraction > 0 and len(train_ids) > 1:
        split_point = max(1, int(round(len(train_ids) * (1 - val_fraction))))
        val_ids = train_ids[split_point:]
        train_ids = train_ids[:split_point]
    return make_manifest(workdir, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids or None)


def manifest_part_ids(manifest: Mapping[str, Any], part: str) -> list[str] | None:
    """The ids a parts manifest gives ``part``, or ``None`` when it gives it none.

    The holdout audit's part is the manifest's ``val`` part (empty when it has
    none): those items are not train, so the search never scored them, and they
    are not test, which nothing but a strategy's own final evaluation may touch.
    Splitting off a fraction instead would draw the holdout from every item the
    Task has, test items included.
    """
    if part in manifest:
        return [str(x) for x in manifest[part]]
    if part == HOLDOUT:
        return [str(x) for x in manifest.get(VAL, [])]
    return None


def write_subset(workdir: Path, part: str, holdout: float, *, seed: int | None = None,
                 gold_feedback: bool = False) -> Path:
    """Record which part of a Task's items a run directory evaluates.

    The outer loop writes this before it evaluates a candidate and the item loader
    reads it, so the part travels with the run directory rather than with the search.
    ``gold_feedback`` marks a train evaluation whose passes may be told how their
    attempt compares with the answer (`Task.train_feedback`); it is refused for any
    other part, so val and test passes never are.
    """
    if gold_feedback and part != TRAIN:
        raise TaskError(f"gold feedback is for train evaluations only, not {part!r}")
    path = _subset_file(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {"part": str(part), "holdout": float(holdout)}
    if seed is not None:
        data["seed"] = seed
    if gold_feedback:
        data[GOLD_FEEDBACK] = True
    path.write_text(json.dumps(data, indent=2))
    return path


#: The ``subset.json`` key that turns gold feedback on for a train evaluation.
GOLD_FEEDBACK = "gold_feedback"


def gold_feedback_on(workdir: Path) -> bool:
    """Whether a run directory's passes get gold feedback: only when its ``subset.json``
    names the train part and turns it on. No file, any other part, or any other value
    is off."""
    record = read_subset(workdir)
    return record.get("part") == TRAIN and record.get(GOLD_FEEDBACK) is True


def read_subset(workdir: Path) -> dict[str, Any]:
    """The part a run directory names, or ``{}`` when it names none."""
    path = _subset_file(workdir)
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def subset_part_seed(workdir: Path) -> tuple[str | None, int | None]:
    """The part and seed ``subset.json`` names, each ``None`` when it names none."""
    record = read_subset(workdir)
    part = record.get("part") or None
    seed = record.get("seed")
    return (str(part) if part else None), (int(seed) if seed is not None else None)


def part_seed_kwargs(
    method: Any, *, part: str | None, seed: int | None,
) -> dict[str, Any]:
    """The ``part``/``seed`` keywords a Task hook accepts, decided by its signature.

    A Task written before ``part`` and ``seed`` existed takes neither, so they are
    passed only to a hook that names them or takes ``**kwargs``. Deciding from the
    signature rather than by calling and catching ``TypeError`` means a TypeError
    raised inside the hook is that item's failure, never a silent second call.
    """
    try:
        params = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return {"part": part, "seed": seed}
    accepted = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return {
        name: value
        for name, value in (("part", part), ("seed", seed))
        if name in params and params[name].kind in accepted
    }


def select_subset(
    items: Sequence[dict[str, Any]], workdir: Path
) -> list[dict[str, Any]]:
    """The part of ``items`` a run directory names, or all of them when it names none.

    Called by the one item loader both runtimes use, so the walker and DSH iterate
    the same items of the same Task. When a ``parts.json`` manifest exists, item
    ids are looked up there. When ``seed`` is present in subset.json, the returned
    items are shuffled within the part using ``random.Random(seed)``.
    """
    import random as _random

    record = read_subset(workdir)
    part = str(record.get("part") or "")

    # Try manifest first
    manifest_path = Path(workdir) / MANIFEST
    if manifest_path.exists() and part:
        try:
            manifest = json.loads(manifest_path.read_text())
            ids = manifest_part_ids(manifest, part) if isinstance(manifest, dict) else None
            if ids is not None:
                allowed_ids = set(ids)
                selected = [
                    item for i, item in enumerate(items)
                    if str(item.get("id") or i) in allowed_ids
                ]
                seed = record.get("seed")
                if seed is not None:
                    selected = list(selected)
                    _random.Random(seed).shuffle(selected)
                return selected
        except (OSError, json.JSONDecodeError):
            pass

    if part not in (TRAIN, HOLDOUT, VAL, TEST):
        return list(items)
    if part in (VAL, TEST):
        raise TaskError(
            f"part {part!r} requires a parts manifest (parts.json); "
            f"no manifest was found in {workdir}"
        )
    effective = part
    try:
        fraction = float(record.get("holdout", DEFAULT_HOLDOUT))
    except (TypeError, ValueError):
        fraction = DEFAULT_HOLDOUT
    selected = partition(items, fraction)[effective]
    seed = record.get("seed")
    if seed is not None:
        selected = list(selected)
        _random.Random(seed).shuffle(selected)
    return selected
