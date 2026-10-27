"""The operations create-mode's own graph calls.

Creating a package is a mode like any other: it has a graph, it runs through the
same executor, and its gate refuses a package that will not run. The difference
is only what it produces — a directory of Python that becomes a new package.
"""

from __future__ import annotations

import importlib
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from refactory_dsh import registry
from refactory_dsh.run import BuildLoopStop
from refactory_dsh.paths import project_root, project_for_nodes
from refactory_dsh.nn import Package, PackageError, Verdict, validate_state
from refactory_dsh.repo_change import refuse_truncated, EMISSION_PATH, unprintable_refusal

#: Where a create/edit run stages its package inside the run directory, so two
#: parallel runs of the same package name never collide.  A git worktree is not
#: used because ``.factory/`` is gitignored: a worktree of the project would
#: contain no ``.factory/packages/`` and nothing written there becomes a commit.
#: A staging copy inside the run directory isolates the run and records its own
#: content hash so ``register()`` can detect whether the shared package changed
#: since ``prepare``.
STAGING_DIR = Path(".factory-staging")

STAGING = "staging"

#: The marker the emit agent uses to separate files. JSON would need every
#: newline escaped, which is where a generated source file usually dies.
FILE_MARKER = re.compile(rf"^===FILE:\s*({EMISSION_PATH})\s*===\s*$", re.MULTILINE)

REQUIRED_FILES = ("__init__.py",)

#: The node field that would bake a data node's items into the graph. The loader reads
#: it before the Task or the source, so setting it closes the harness's input port.
BAKED_ITEMS_KEY = "inline_items"

#: Why a creation with no Task and no `single_shot` is refused rather than defaulted.
#:
#: A harness is for a class of work, not for one instance of it. A Task is how a class
#: of work is named, and naming one is what binds the produced graph's `data` node to
#: the items the run iterates, so `create` requires the caller to say which it is
#: instead of quietly producing a single-shot harness or inventing a Task.
CREATION_DIRECTIVE = (
    "create-mode needs to know what it is building. Name the class of work with a Task "
    "— `--task <module:Class>` on the command line, or the `task_ref` knob — and the "
    "produced harness binds a `data` node to that Task and runs its subgraph once per "
    "item the Task lists. Or say `--single` (the `single_shot` knob) for a harness that "
    "runs once, for one input, with no Task binding. The class of work is the default "
    "because a harness built for one hardcoded example is the special case, not the "
    "product, and only naming a Task makes the per-item binding real."
)

API = """\
You are writing a package for the refactory mode library. The API is:

    from refactory_dsh import nn

    class MyMode(nn.Mode):
        name = "shortname"                 # the mode's name inside the package
        description = "one line"

        @classmethod
        def declare(cls):                  # the parameters the outer loop may move
            return {"rounds": nn.Knob(3, "int", lo=1, hi=8, description="..."),
                    "order": nn.Knob("desc", "enum", enum=("desc", "asc")),
                    "seed": nn.Knob(7, "int", frozen=True, description="...")}

        def build(self):                   # pure: no I/O, no randomness, no clock
            graph = nn.Graph()
            graph.add(nn.Node("first", "op", call="refactory_packages.my_pkg.ops:first",
                              reads=(), writes=("a.json",), produces=("a",),
                              knobs=("rounds",), notes="what it does"))
            graph.add(nn.Node("check", "gate", evaluator="refactory_packages.my_pkg.ops:check",
                              reads=("a.json",), produces=("ok",), max_iterations=8))
            graph.add(nn.Node("done", "op", call="refactory_packages.my_pkg.ops:done",
                              reads=("a.json",), writes=("best.json",), produces=("best",)))
            graph.connect("first", "check")
            graph.connect("check", "first", condition="retry")
            graph.connect("check", "done", condition="proceed")
            return graph

        def score(self, run):              # read run.facts here, never a knob
            return float(run.facts["best"])

        def invariants(self):              # what the validator must keep true
            return (nn.fact_produced("best"),)

        def insertable(self):              # blocks a structural edit may add
            return (nn.Node("observe", "op", call="refactory_packages.my_pkg.ops:observe",
                            reads=("a.json",), writes=("observations.json",),
                            produces=("spread",)),)

    PACKAGE = nn.Package(name="my_pkg", description="...", modes=(MyMode,),
                         ops=("refactory_packages.my_pkg.ops",))

For required deliverables that score() does not measure, protect their actual
producer in invariants(), e.g. nn.artifact_produced("solution.md", by="explain").
This refuses removing that producer even when another node can write a fallback
file with the same name. It is a structural constraint, not proof of semantic
quality: run checks must still reject absent or placeholder content, and reviews
must inspect the actual delivered output. A numeric answer or execution log is
not a mathematical derivation. Invariants are code outside the optimizer's state;
notes and optimization intent alone are not enforceable constraints.

An operation is a plain function of the run context:

    def first(ctx):
        value = int(ctx.knob("rounds", 3))      # read a declared knob
        ctx.write_json("a.json", {"value": value})   # write a file you declared
        return {"a": value}                     # return the facts you declared

The context offers `ctx.knob(name, default)`, `ctx.read_json(name, default)`,
`ctx.write_json(name, payload)`, `ctx.write_text`, `ctx.append_text`,
`ctx.read_text`, `ctx.exists`, `ctx.set_fact(name, value)`, `ctx.facts`,
`ctx.task` (what the run was started with), `ctx.workdir`, `ctx.iteration` and
`ctx.model`.

A gate is an operation that returns its verdict: one of its out-edges' conditions.
When it sends the run back, return the verdict as `nn.Verdict(label, reason)`, so
whoever is driving the run, and the person watching, learn why. A node that needs
the reason on its next pass reads it from a fact or file the gate writes:

    def check(ctx):
        value = ctx.read_json("a.json", {})["value"]
        if value >= 3:
            return "proceed"
        return nn.Verdict("retry", f"value {value} is below 3")

A mode whose agent changes code in a repository takes `refactory_dsh.repo_change`'s
steps rather than writing into the repository itself: declare `**repo_change.knobs()`;
list `*repo_change.knob_names()` and `*repo_change.produces()` on the prepare node and
return `repo_change.locate(ctx, name, ctx.task)`'s facts from its operation; name the
agent node that writes `emit.txt` `emit` and put `{{emit_scope}}` in its prompt; add `repo_change.materialise_node()`; in the check gate, send
`repo_change.refused(ctx)` back to `emit` and run the check with
`repo_change.run_check(ctx, command, timeout)`, declaring `*repo_change.CHECK_WRITES` and
`*repo_change.CHECK_PRODUCES` on that gate; call `repo_change.report(ctx, name, payload)`,
which writes the report file `name`, from the report operation; add
`"refactory_dsh.repo_change"` to `PACKAGE.ops`; and call
`verification.wire_verification(graph, check_node=..., proceed_target=...)` after
the check and report edges are connected, to add the design-blind verifier between
the check gate and the proceed target.  Full-mode and fast-mode graphs add the
verifier directly to the review fork; only this custom form is used by packages.
`refactory_packages.fix_pack` is a complete example.

Generated packages live under `.factory/packages/<name>/` and are imported as
`refactory_packages.<name>`. Use that qualified namespace for `call`, `evaluator`
and `PACKAGE.ops`; the package name itself remains the bare name, e.g. `my_pkg`.

Rules the validator enforces on every candidate, so a violation is a refusal:
- every node is reachable from the start node, and the graph has a terminal
- a non-gate node has at most one out-edge, and it carries no verdict
- a gate has at least two out-edges, each with a ``condition``, and every cycle
  passes through a gate
- every file a node reads is written by some node (declare `reads`/`writes`)
- every fact a node returns is declared in its `produces`
- every knob a node lists in `knobs` is declared by the mode, and every declared
  knob is consumed by at least one node
- every `call`/`evaluator` is a real importable ``module:function``
- score() is a method and is never in the state, so nothing can move it
"""


def _slug(text: str, *, words: int = 3) -> str:
    """A package name from what the run is for.

    A name is how the result is found later, so a run given only an idea still needs
    one. Three words, lowercased, `_`-joined -- which is the shape a package name
    already has to be, so the same text that asks for something also names it.
    """
    kept = [part for part in re.findall(r"[a-z0-9]+", text.lower()) if part]
    slug = "_".join(kept[:words]) or "created_pack"
    if slug[0].isdigit():
        slug = f"pkg_{slug}"
    return slug[:40]


def _mode_contracts(name: str) -> dict[str, list[dict[str, Any]]]:
    """Read actual data bindings where loading generated code is already contained."""
    from refactory_dsh import sandbox

    result = sandbox.describe_package(name, project=project_for_nodes(), include_contracts=True)
    if not result.ok:
        raise ValueError(f"cannot read the mode's Task bindings: {result.problem()}")
    return result.payload["mode_contracts"]


def _required_bindings(ctx: Any, revision: str) -> tuple[list[str], bool]:
    """Explicit directives win; otherwise a revision preserves the source contract."""
    explicit = str(ctx.knob("task_ref", "") or "").strip()
    requires_data = bool(ctx.knob("data_node", 0)) or bool(explicit)
    if explicit or bool(ctx.knob("single_shot", 0)) or not revision:
        return ([explicit] if explicit else []), requires_data
    package, _mode = registry.split(revision)
    contracts = _mode_contracts(package)
    if revision not in contracts:
        raise ValueError(f"the source package has no mode {revision!r}")
    nodes = contracts[revision]
    refs = sorted({node["task_ref"] for node in nodes if node.get("task_ref")})
    return refs, requires_data or bool(nodes)


def _binding_metadata(task_refs: list[str], data_node: bool) -> dict[str, Any]:
    """Describe the actual or requested bindings without calling data input single-shot."""
    refs = sorted(set(task_refs))
    if refs:
        class_of_work = (
            "a harness for the class of work named by the Task " + refs[0]
            if len(refs) == 1 else "a harness for the classes of work named by the Tasks " + ", ".join(refs)
        ) + (
            ": a `data` node bound to each Task runs its subgraph once per item, "
            "folding each `verify` verdict into the run's facts; runtime sources may "
            "supply other items of that class"
        )
    elif data_node:
        class_of_work = "a harness with data nodes fed from runtime sources, with no Task binding"
    else:
        class_of_work = "a harness for one run of one thing, with no Task binding"
    return {"task_ref": refs[0] if len(refs) == 1 else "", "task_refs": refs,
            "class_of_work": class_of_work, "data_node": "yes" if data_node else "no"}


def prepare(ctx: Any) -> dict[str, Any]:
    """Read the brief the run was given and publish it to the prompts."""
    task = dict(getattr(ctx, "task", {}) or {})
    # The walker passes name/brief in the run's task; a run on DSH carries them as
    # knobs, because DSH's task is the CEO's prompt. Both are the same fact.
    name = str(task.get("name") or ctx.knob("name", "") or "").strip()
    brief = str(task.get("brief") or ctx.knob("brief", "") or "").strip()
    from refactory_dsh.brief_guard import require_brief
    require_brief(brief, source="brief")
    if not name and brief:
        # "an idea" is enough to start: the name is how the result is found later, and
        # a slug of the brief is a better answer than refusing to begin.
        name = _slug(brief)
    if not name:
        raise ValueError(
            "create-mode needs what to build: pass {'name': ..., 'brief': ...} in the "
            "task (the walker) or the `name`/`brief` knobs (a run on DSH)"
        )
    extends = str(task.get("extends") or "").strip()
    revision = str(task.get("mode") or ctx.knob("revision_of", "") or "").strip()
    intent = str(task.get("intent") or "Make a new package of modes.").strip()
    # The creation directives: a Task class the produced mode must satisfy, whether
    # the caller asked for the single-shot special case, and whether the produced
    # graph is a DataNode that runs a subgraph once per item. A Task names a class of
    # work, so naming one implies the data node rather than asking for it twice.
    task_ref = str(ctx.knob("task_ref", "") or "").strip()
    single_shot = bool(ctx.knob("single_shot", 0))
    # A revision is not a creation: `edit` walks this same graph with a mode to
    # revise, and the creation directive is about what a *new* harness is for.
    if not revision and not task_ref and not single_shot:
        raise ValueError(CREATION_DIRECTIVE)
    # The inputs the brief says the mode must take. Published as a fact so the design
    # and emit prompts state them, and read by `validate` so a mode that hardcodes a
    # fixture instead is refused rather than shipped.
    inputs = declared_inputs(ctx.knob("requires_input", ""))
    from refactory_dsh.sandbox import DEFAULT_MODEL_SMOKE_TIMEOUT_S
    smoke_timeout_s = int(ctx.knob("smoke_timeout_s", DEFAULT_MODEL_SMOKE_TIMEOUT_S))
    if not 1 <= smoke_timeout_s <= 86400:
        raise ValueError("smoke_timeout_s must be between 1 and 86400 seconds")
    # Fail before research, using the exact downstream HOME rewrite and scrub.
    # Other ModelClient implementations own their own readiness contract.
    from refactory_dsh.model import DshModelClient
    from refactory_dsh.sandbox import model_readiness

    client = getattr(ctx, "model", None)
    if client is None or isinstance(client, DshModelClient):
        readiness = model_readiness(
            ctx.workdir / ".model-readiness",
            model_context=client.context() if isinstance(client, DshModelClient) else None,
        )
        if readiness.get("warning"):
            import sys

            print(f"refactory: warning: {readiness['warning']}", file=sys.stderr)
    task_refs, data_node = _required_bindings(ctx, revision)
    metadata = _binding_metadata(task_refs, data_node)
    task_ref = metadata["task_ref"]
    reference = _revision_source(revision) if revision else _reference(extends)
    # Feedback files are declared worker inputs even on the first pass. Initialize
    # them truthfully; only their later gates can record actual validation.
    for feedback in ("research-review.txt", "strategy-review.txt", "validation.txt"):
        ctx.write_text(feedback, "not run: initial authoring pass; no prior feedback\n")
    ctx.write_json("brief.json", {
        "name": name, "brief": brief, "extends": extends, "mode": revision,
        "task_ref": task_ref, "task_refs": task_refs,
        "single_shot": single_shot, "data_node": data_node,
        "requires_input": list(inputs), "smoke_timeout_s": smoke_timeout_s,
    })
    return {
        "name": name,
        "brief": brief,
        "extends": extends,
        "revision_of": revision,
        "intent": intent,
        "validation": "(no validation has run yet)",
        "design_review": "(no design has been reviewed yet)",
        "research_review": "(no research review has run yet)",
        **UNREFUSED,
        "api": API,
        "reference": reference,
        "task_ref": task_ref,
        "class_of_work": metadata["class_of_work"],
        "data_node": metadata["data_node"],
        "requires_input": (
            "The brief says this mode takes these as input: " + ", ".join(f"`{name}`" for name in inputs)
            + ". It must receive each one as a declared knob named after it, or by "
            "reading a file under that name from the run directory. A mode that "
            "hardcodes a fixture instead of accepting the input is refused by "
            "validation."
        ) if inputs else "(the brief named no inputs this mode must accept)",
    }


def _extract_json_array(text: str) -> list[Any]:
    """Extract the first JSON array from *text*, tolerating fences and prose.

    Returns the array's items as written (objects keep every key; a caller decides
    what a non-object item means) or an empty list when the text contains no
    parseable array.  Both ``_plan`` and ``classify_scale`` use this
    so there is exactly one tolerant-array extractor.
    """
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        return []
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return list(payload)


def _plan(text: str) -> list[dict[str, Any]]:
    """The plan's entries, or an empty list when the answer is not a plan.

    Tolerant about the fences and prose around the array — a model that writes
    "here is the plan:" before it has still written a plan — and intolerant about
    everything else, because the alternative is a downstream stage that runs on a
    stage which did nothing.
    """
    entries: list[dict[str, Any]] = []
    for item in _extract_json_array(text):
        if not isinstance(item, dict):
            continue
        entries.append({
            key: str(item.get(key) or "").strip() for key in ("focus", "slug", "finding")
        } | {"ids": item.get("ids") or []})
    return entries


def _excerpt(text: str, limit: int = 200) -> str:
    """The start of an answer, on one line, for a retry prompt to read."""
    flat = " ".join(str(text).split())
    return f"{flat[:limit]}{'…' if len(flat) > limit else ''}" if flat else "(empty)"


def record_findings(ctx: Any) -> dict[str, Any]:
    """Put every direction's answer on disk, so the evidence exists before it is used.

    The research director's subagents write `research/<slug>.md` themselves; this fills
    in a direction that has no note from the finding the plan carries. It is what makes
    the plan's `finding` field load-bearing rather than decorative, and it is why a
    walker run — where the agent's answer *is* the plan, and no subagent exists — leaves
    the same artifacts a DSH run does.
    """
    recorded: list[str] = []
    for entry in _plan(ctx.read_text("research-plan.json")):
        slug = entry["slug"]
        if not slug or not RESEARCH_SLUG.fullmatch(slug):
            continue
        path = f"research/{slug}.md"
        if ctx.exists(path) and ctx.read_text(path).strip():
            continue
        fnd_id = f"FND-{slug}-1"
        ctx.write_text(path, f"### {fnd_id} \u2014 {entry['finding']}\n\n# {entry['focus']}\n\n{entry['finding']}\n")
        recorded.append(path)
    return {"findings_recorded": recorded}


def research_problems(ctx: Any, raw: str, research_scope: str, *,
                      count: str = "required") -> list[str]:
    """What is wrong with a research plan, one line each; [] when nothing is.

    Create's `gate_research` refuses a plan on these; design mode's
    `design_ops.gate_design_research` lists them as advice (``count="suggested"``) and
    proceeds.
    """
    problems: list[str] = []
    entries = _plan(raw)
    if not entries:
        problems.append(
            "research-plan.json is not a JSON array of {focus, slug, finding} objects; "
            f"it holds: {_excerpt(raw)}"
        )
    else:
        if research_scope == "scoped":
            if not 1 <= len(entries) <= 3:
                problems.append(
                    f"the plan has {len(entries)} direction(s); one to three are {count} for a scoped change"
                )
        else:
            if not 3 <= len(entries) <= 5:
                problems.append(
                    f"the plan has {len(entries)} direction(s); three to five are {count}"
                )
        seen: set[str] = set()
        for index, entry in enumerate(entries):
            label = entry["slug"] or f"entry {index + 1}"
            for field in ("focus", "slug", "finding"):
                if not entry[field]:
                    problems.append(f"{label}: `{field}` is empty")
            slug = entry["slug"]
            if slug and not RESEARCH_SLUG.fullmatch(slug):
                problems.append(
                    f"{label}: `slug` must be lower-case letters, digits and dashes"
                )
            if slug and slug in seen:
                problems.append(f"{label}: `slug` is used twice")
            seen.add(slug)
            path = f"research/{slug}.md"
            if slug and RESEARCH_SLUG.fullmatch(slug) and not (
                ctx.exists(path) and ctx.read_text(path).strip()
            ):
                problems.append(f"{label}: {path} is missing or empty")
            elif slug and RESEARCH_SLUG.fullmatch(slug) and ctx.exists(path):
                content = ctx.read_text(path)
                from refactory_dsh.findings_coverage import definitions
                defs = definitions(content)
                if content.strip() and not defs:
                    problems.append(
                        f"{label}: {path} has no FND-<slug>-<n> finding ids"
                    )
                else:
                    for fid, fslug, _fn, _line_text in defs:
                        if fslug != slug:
                            problems.append(
                                f"{label}: {path} has id {fid} with slug "
                                f"'{fslug}' but the file's slug is '{slug}'"
                            )
                    seen_fids: set[str] = set()
                    for fid, _fslug, _fn, _line_text in defs:
                        if fid in seen_fids:
                            problems.append(
                                f"{label}: {path} has duplicate id {fid}"
                            )
                        seen_fids.add(fid)
    return problems


def gate_research(ctx: Any) -> str:
    """Proceed only when the plan is a plan and its findings are on disk.

    The stage exists to ground the design in facts, and a node that narrates its
    intent instead of producing the array satisfies ``writes=("research-plan.json",)``
    with any bytes at all: two real runs wrote a prose status message about subagents
    "running in the background", left ``research/`` empty, and the design proceeded on
    unverified assumptions. The reasons go to `research-review.txt` and back into the
    next attempt's prompt, exactly as the design review's do.
    """
    raw = ctx.read_text("research-plan.json")
    allowed = int(ctx.knob("research_attempts", 3))
    # Read research_scope from the run's facts, NOT from knobs.
    research_scope = ctx.facts.get("research_scope", "full")
    problems = research_problems(ctx, raw, research_scope)
    used = int(ctx.facts.get("research_attempts_used", 0)) + 1
    report = "\n".join(f"- {problem}" for problem in problems)
    ctx.write_text("research-review.txt", report or "ok\n")
    ctx.set_fact(
        "research_review", report or "(every direction is answered and written down)"
    )
    if not problems:
        return "proceed"
    if used >= allowed:
        raise ValueError(
            f"the research is still not usable after {used} attempt(s):\n{report}"
        )
    ctx.set_fact("research_attempts_used", used)
    return Verdict("reloop", report)


def _revision_source(identifier: str) -> str:
    """The source of the mode being revised, so a revision starts from it."""
    try:
        package, mode_class = registry.resolve(project_for_nodes(), identifier)
    except PackageError as exc:
        raise ValueError(f"cannot revise {identifier!r}: {exc}") from exc
    root = registry.where(package.name, project_for_nodes())
    if root is None:
        raise ValueError(f"cannot find the source of {identifier!r}")
    blocks = [
        f"# --- {path.name} ---\n{path.read_text()}"
        for path in sorted(root.glob("*.py"))
    ]
    ours = root / "modes" / f"{mode_class.name}.py"
    if ours.exists():
        blocks.append(f"# --- modes/{ours.name} (the mode being revised) ---\n{ours.read_text()}")
    return (
        f"This is a revision of `{identifier}`. Keep everything the brief does not ask "
        "to change, and say in the design what you changed and why.\n\n"
        + "\n\n".join(blocks)
    )


def _reference(extends: str) -> str:
    """The source of an existing package, when the new one extends it."""
    if not extends:
        return "(no reference package was requested; design from the brief)"
    package_name = extends.split("/", 1)[0]
    root = registry.where(package_name, project_for_nodes()) or registry.packages_root() / package_name
    if not root.is_dir():
        return f"(the requested reference package {package_name!r} does not exist)"
    blocks = [
        f"# --- {path.relative_to(root)} ---\n{path.read_text()}"
        for path in sorted(root.rglob("*.py"))
    ]
    return "\n\n".join(blocks)


STRATEGY_SCHEMA = """\
Write one JSON object and nothing else:

{"considered": [{"approach": "...", "why_not": "..."}],
 "chosen": "<the approach>",
 "why": "<why it wins>",
 "will_not": ["<something deliberately out of scope>"],
 "closest_existing": "<package>/<mode>, or 'none'",
 "knobs": [{"name": "...", "kind": "int|float|bool|str|enum", "default": ...,
            "lo": null, "hi": null, "enum": [], "description": "..."}],
 "nodes": [{"id": "...", "kind": "op|agent|gate|fork|join|data",
            "reads": [], "writes": [], "produces": [], "knobs": [],
            "notes": "..."}],
 "edges": [{"source": "...", "target": "...", "condition": null}],
 "start": "<node id>",
 "score": "<which produced fact score() reads, and why that is the right number>",
 "findings": {"FND-<slug>-<n>": "<node or knob that handles it, or [set-aside: reason]>"}}

Rules:
- At least two approaches in `considered`. One approach is not a choice.
- `nodes` and `edges` describe the graph you intend to write, so it can be checked
  before any code exists. The node fields above are common fields, not a closed
  list. Add the fields required by each node's kind below, at the node's top level;
  putting them only in `notes` does not declare them. List each declared knob in
  `knobs` on the nodes that consume it.
- An `op` requires `call: "<module>:<function>"`.
- An `agent` requires a nonempty `prompt_template` containing its actual worker
  instructions. A `call` or a module path to a prompt constant is not a substitute
  for `prompt_template`. It may also declare `role` and `tools`.
- A function `gate` requires `evaluator: "<module>:<function>"`. A person-answered
  or agent-answered gate instead declares `gate_kind: "user"` or `"agent"` and no
  evaluator. Gates need at least two outgoing edges with distinct, nonempty
  `condition` verdicts, no unconditional edge, and a bounded `max_iterations`.
- A `data` node requires `subgraph_entry` and `subgraph_exit`, naming the first
  and last nodes of the per-item subgraph. Bind the requested `task_ref`, or use a
  runtime `source_path` and `source_format`; do not bake instances into the graph.
  It writes `current_item.json` before each pass; declare that write when consumed.
  Declare `item_scores` in `produces` when score() aggregates Task verdicts.
  The subgraph is entered through these fields, NOT through an ordinary edge.
  The data node has at most one outgoing edge, to the continuation AFTER all items;
  the per-item exit does not need an edge back to the data node or to that
  continuation. For example, `items` with subgraph_entry=`solve` and
  subgraph_exit=`publish` has per-item edges solve -> execute -> explain -> publish
  and one outer edge items -> finish. Do not add items -> solve or publish -> finish.
  When the requested harness is Task-bound, its start node is the data node.
- A `fork` declares `targets` naming its branch entries. A `join` declares `sources`
  naming the branch exits it waits for. These fields connect the parallel branches;
  do not replace them with extra ordinary edges from the fork or to the join.
- Omit kind-specific fields that do not apply to that node. All node ids named by
  edges, subgraph boundaries, targets and sources must exist in `nodes`.
- Every read must be written by some node, and every declared knob consumed.
- `closest_existing` must name a mode from the survey, or be 'none' if none is close.
"""


RESEARCH_SCHEMA = """\
Write one JSON array and nothing else, one object per direction:

[{"focus": "<the question this direction answers>",
  "slug": "<a file-name-safe name for it>",
  "finding": "<what the answer is, in one paragraph>",
  "ids": ["FND-<slug>-1", "FND-<slug>-2"]}]

Rules:
- Between three and five directions for a full design, or one to three for a scoped change. The research scope is a fact set by the prepare step.
- `slug` is lower-case letters, digits and dashes only, and `research/<slug>.md` is
  where that direction's long form belongs.
- `finding` states what the answer *is*. A direction nobody could answer says so in
  the finding ("not answered: <why>") rather than being dropped or invented.
- Each finding in `research/<slug>.md` is a heading `### FND-<slug>-<n> \u2014 <the finding>`,
  numbered from 1 within the file, and `ids` lists them.
- A plan that says subagents are still working is not a plan: by the time you answer,
  the work is either done or it is not.
"""

#: What the emit and review prompts show for the review gates' verdicts before any
#: gate has refused anything. A fact no gate has produced yet would otherwise reach
#: the worker as a raw ``{{qa_gate}}``, which reads as text to act on, not as "none".
#: Fast modes say instead that they run no such gate (``fast_ops.FAST_UNREFUSED``).
UNREFUSED = {
    "qa_gate": "(no review has refused anything yet)",
    "overwatch_review": "(no audit has refused anything yet)",
}

#: A slug a research file can be named by.
RESEARCH_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def survey(ctx: Any) -> dict[str, Any]:
    """What already exists, before anything is invented.

    Cheap and deterministic, and it makes "does this already exist?" a question the
    system can answer instead of a question nobody asked. Without it, `--extends` is
    the only reuse channel and it is manual.
    """
    installed: list[dict[str, Any]] = []
    for name in registry.discover_names(project_for_nodes()):
        try:
            package = registry.load(name, project_for_nodes())
        except PackageError as exc:
            installed.append({"package": name, "problem": str(exc)})
            continue
        entry: dict[str, Any] = {
            "package": name,
            "description": package.description,
            "requires": list(package.requires),
            "modes": [],
        }
        for mode_class in package.modes:
            try:
                mode = mode_class()
            except Exception as exc:
                entry["modes"].append({"name": mode_class.name, "problem": str(exc)})
                continue
            entry["modes"].append({
                "id": f"{name}/{mode_class.name}",
                "description": getattr(mode_class, "description", ""),
                "knobs": sorted(mode.parameters()),
                "nodes": sorted(mode.graph().nodes),
            })
        installed.append(entry)

    ctx.write_json("survey.json", {"packages": installed})
    lines = ["Existing packages and what they offer:", ""]
    for entry in installed:
        lines.append(f"- {entry['package']}: {entry.get('description', '')}")
        for mode in entry.get("modes", ()):
            if "problem" in mode:
                lines.append(f"    {mode['name']} (unusable: {mode['problem']})")
                continue
            lines.append(
                f"    {mode['id']}: knobs {mode['knobs']}; nodes {mode['nodes']}"
            )
    return {"survey": "\n".join(lines), "existing": [e["package"] for e in installed]}


def review_design(ctx: Any) -> str:
    """Gate: check the intended graph before any code is written.

    Catching a bad design here is far cheaper than catching bad code at `emit`, and
    it needs no new machinery: it is the same validator the loop uses, applied to
    the design's declared nodes and knobs rather than to a built graph. Calls are not
    resolved, because the package that will define them does not exist yet.
    """
    raw = ctx.read_text("strategy.json")
    attempts_allowed = int(ctx.knob("design_attempts", 2))
    problems = _review(raw)
    from refactory_dsh import findings_coverage
    problems.extend(findings_coverage.gate_problems(ctx, document='strategy.json'))
    used = int(ctx.facts.get("design_attempts_used", 0)) + 1
    report = "\n".join(f"- {problem}" for problem in problems)
    ctx.write_text("strategy-review.txt", report or "ok\n")
    ctx.set_fact("design_review", report or "(the intended design is coherent)")
    if not problems:
        return "proceed"
    if used >= attempts_allowed:
        raise ValueError(
            f"the design is still not coherent after {used} attempt(s):\n{report}"
        )
    ctx.set_fact("design_attempts_used", used)
    return Verdict("reloop", report)


def _review(raw: str) -> list[str]:
    from refactory_dsh.nn import Graph, Issue, Knob, validate

    try:
        payload = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError) as exc:
        return [f"the strategy answer is not a JSON object: {exc}"]

    problems: list[str] = []
    considered = payload.get("considered") or []
    if len(considered) < 2:
        problems.append(
            f"only {len(considered)} approach was considered; a choice needs at least two"
        )
    for field in ("chosen", "why", "score"):
        if not str(payload.get(field) or "").strip():
            problems.append(f"the strategy says nothing about {field!r}")

    existing = set(ctx_existing())
    closest = str(payload.get("closest_existing") or "none")
    if closest != "none" and closest not in existing:
        problems.append(
            f"closest_existing names {closest!r}, which is not an installed mode; "
            f"known: {sorted(existing)}"
        )

    knobs: dict[str, Knob] = {}
    for entry in payload.get("knobs") or ():
        try:
            knobs[str(entry["name"])] = Knob.from_spec({
                "default": entry.get("default"),
                "kind": entry.get("kind", "float"),
                "lo": entry.get("lo"),
                "hi": entry.get("hi"),
                "enum": entry.get("enum") or [],
                "frozen": entry.get("frozen", False),
                "description": entry.get("description", ""),
            })
        except Exception as exc:
            problems.append(f"knob {entry.get('name')!r} is not valid: {exc}")
    if not knobs:
        problems.append("the strategy declares no knobs, so there is nothing to optimize")

    try:
        graph = Graph.from_state({
            "start": str(payload.get("start") or ""),
            "nodes": {str(n["id"]): n for n in payload.get("nodes") or ()},
            "edges": list(payload.get("edges") or ()),
        })
    except Exception as exc:
        return problems + [f"the declared graph does not build: {exc}"]

    for issue in validate(
        graph, {name: knob.default for name, knob in knobs.items()},
        declared_knobs=knobs, resolve_calls=False,
    ):
        problems.append(str(issue))
    return problems


def ctx_existing() -> list[str]:
    """The installed mode ids, for a design to be compared against."""
    ids: list[str] = []
    for name in registry.discover_names(project_for_nodes()):
        try:
            package = registry.load(name, project_for_nodes())
        except PackageError:
            continue
        ids.extend(package.ids())
    return ids


def _package_relative(relative: str, name: str) -> str:
    """One emit path, as a path inside the package's own directory.

    The prompt asks for a path relative to the package root, and a model that reads
    "relative" as "relative to the checkout" answers with
    `refactory/refactory_packages/<name>/mode.py`. Joined onto the package root, that
    nests the package inside itself and the mistake surfaces four directories away as
    "__init__.py is missing". The prefix is stripped rather than accepted as a nesting,
    and a path that cannot be made package-relative is refused by name instead of
    written somewhere nobody looks.
    """
    from pathlib import PurePosixPath

    if refusal := unprintable_refusal(str(relative)):
        raise ValueError(refusal)
    raw = str(relative).replace("\\", "/").strip()
    parts = [part for part in PurePosixPath(raw).parts if part not in ("", ".")]
    if raw.startswith("/") or not parts:
        raise ValueError(
            f"the emit answer names {relative!r}, which is not a path inside the "
            f"package; write it relative to the package root, e.g. `mode.py`"
        )
    # Longest first, so `refactory/refactory_packages/<name>/x.py` loses the whole
    # prefix and `<name>/x.py` only its own.
    for prefix in (
        ("refactory", "refactory_packages", name),
        (".factory", "packages", name),
        ("refactory_packages", name),
        ("refactory", "refactory_packages"),
        (".factory", "packages"),
        (name,),
    ):
        if tuple(parts[: len(prefix)]) == prefix:
            parts = parts[len(prefix):]
            break
    if not parts or ".." in parts:
        raise ValueError(
            f"the emit answer names {relative!r}, which does not resolve to a file "
            f"inside the package root; write paths relative to it, e.g. `mode.py`"
        )
    return str(PurePosixPath(*parts))


def materialise(ctx: Any) -> dict[str, Any]:
    """Turn the emit agent's answer into a package directory."""
    answer = ctx.read_text("emit.txt")
    try:
        files = parse_files(answer)
    except ValueError as exc:
        if isinstance(exc, BuildLoopStop):
            raise  # not a refusal — re-raise the stop
        from refactory_dsh.repo_change import _count_refusal
        return _count_refusal(ctx, exc)
    if not files:
        raise ValueError(
            "the answer contained no '===FILE: <name>===' blocks, so there is "
            "nothing to write"
        )
    name = str(ctx.facts["name"])
    # Every path is placed before anything is removed or written, and a path that cannot be
    # placed goes back to emit like a malformed header rather than ending the run.
    try:
        placed = {relative: _package_relative(relative, name) for relative in files}
    except ValueError as exc:
        from refactory_dsh.repo_change import _count_refusal
        return _count_refusal(ctx, exc)
    # When the run has a staging directory (from prepare_create), write there
    # instead of the shared .factory/packages/<name>/.
    staging_root = ctx.facts.get("package_root")
    if staging_root:
        root = Path(staging_root)
    else:
        root = registry.workspace_root(project_for_nodes()) / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    written: list[str] = []
    for relative, content in files.items():
        inside = placed[relative]
        refuse_truncated(Path(inside), content)
        target = root / inside
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"{relative!r} escapes the package directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        written.append(inside)
    # What the writer records about where this came from, which outranks anything a
    # generated manifest claims about itself.
    (root / ".origin.json").write_text(
        json.dumps(
            {
                "origin": "create-mode",
                "ref": "",
                "trust": "generated",
                "at": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(timespec="seconds"),
            },
            indent=2,
        )
    )
    # When staging, make the staged package importable for validation.
    staging_root = ctx.facts.get("package_root")
    if staging_root:
        staging_base = Path(staging_root).parent.parent.parent  # up from .factory/packages/<name>
        registry.install_workspace(project_for_nodes(), project_override=staging_base)
    else:
        registry.install_workspace(project_for_nodes())
    ctx.write_json(f"{STAGING}/manifest.json", {"package": name, "files": sorted(written)})
    cleared = {"emission_refused": ""} if ctx.facts.get("emission_refused") else {}
    reset_refusals = {"consecutive_refusals": 0} if ctx.facts.get("consecutive_refusals") else {}
    return {"package_path": str(root), "files": sorted(written), **cleared, **reset_refusals}


def parse_files(answer: str) -> dict[str, str]:
    """Split an answer into ``path -> source`` blocks."""
    for line in answer.splitlines():
        if line.startswith(("===EDIT:", "===DELETE:")):
            raise ValueError(
                f"this mode takes whole files only (===FILE: <path>===); "
                f"got: {line!r}"
            )
        if line.startswith("===FILE:") and FILE_MARKER.fullmatch(line) is None:
            raise ValueError(
                f"malformed emission header: {line!r}; "
                "a path may contain any characters except whitespace, '=', "
                "backtick, or quotes"
            )
    files: dict[str, str] = {}
    matches = list(FILE_MARKER.finditer(answer))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(answer)
        body = answer[match.end() : end].strip("\n")
        files[match.group(1)] = body + "\n"
    return files


def validate(ctx: Any) -> str:
    """Gate: import the staged package and say whether it is a usable mode.

    Every reason is written out and fed back into the next emit prompt, because a
    model that is told exactly which line broke can fix it in one attempt.
    """
    refused = str(ctx.facts.get("emission_refused") or "")
    if refused:
        detail = f"materialise could not apply emit.txt, so nothing was checked: {refused}"
        ctx.write_text("validation.txt", detail + "\n")
        ctx.set_fact("validation", detail)
        return Verdict("reloop", detail)
    name = str(ctx.facts["name"])
    attempts = int(ctx.knob("attempts", 3))
    brief = ctx.read_json("brief.json", {}) or {}
    if "task_refs" in brief:
        task_refs, requires_data = list(brief["task_refs"]), bool(brief.get("data_node"))
    elif brief.get("task_ref") or brief.get("single_shot"):
        # Preserve an explicit directive already saved by an older prepare node.
        task_refs = [str(brief["task_ref"])] if brief.get("task_ref") else []
        requires_data = bool(brief.get("data_node")) or bool(task_refs)
    else:
        # Saved authoring runs before this metadata existed still inherit their source.
        revision = str(ctx.facts.get("revision_of") or brief.get("mode") or "")
        task_refs, requires_data = _required_bindings(ctx, revision)
        requires_data = requires_data or bool(brief.get("data_node"))
    # When the package is staged (prepare_create), the project for validation is
    # the staging base, not the shared checkout.
    staging_root = ctx.facts.get("package_root")
    staging_project = Path(staging_root).parent.parent.parent if staging_root else None
    try:
        problems = check_package(
            name,
            # The same implication `prepare` publishes as a fact: naming a Task is
            # what asks for a data node, so the gate checks for one either way.
            requires_data_node=requires_data,
            requires_task_refs=tuple(task_refs),
            requires_input=declared_inputs(ctx.knob("requires_input", "")),
            project_override=staging_project,
        )
    except Exception as exc:
        problems = [f"{type(exc).__name__}: {exc}"]

    used = int(ctx.facts.get("attempts_used", 0)) + 1
    if not problems:
        ctx.write_text("validation.txt", "ok\n")
        ctx.set_fact("validation", "(the previous attempt validated)")
        return "proceed"
    report = "\n".join(f"- {problem}" for problem in problems)
    ctx.write_text("validation.txt", report + "\n")
    ctx.set_fact("validation", report)
    if used >= attempts:
        raise ValueError(
            f"the package {name!r} still does not work after {used} attempt(s):\n{report}"
        )
    ctx.set_fact("attempts_used", used)
    return Verdict("reloop", report)


def _evict(module_name: str) -> None:
    """Drop a package and its submodules from the import cache.

    A regenerated package must not be judged by the module object an earlier
    attempt left behind: the whole point of the loop is that the second attempt is
    a different program.
    """
    for cached in [
        name
        for name in sys.modules
        if name == module_name or name.startswith(module_name + ".")
    ]:
        del sys.modules[cached]


def declared_inputs(raw: Any) -> tuple[str, ...]:
    """The inputs a brief declared, from the comma-separated knob a run carries.

    A name, not a sentence: the check is whether the produced mode has a *way to
    receive* it — a knob named after it, or a read under that name — so the name has
    to be something a knob or a file could be called. Anything shorter than three
    characters is refused rather than accepted, because `"a" in "package_path"` is not
    a check.
    """
    names = tuple(
        part.strip() for part in str(raw or "").replace(";", ",").split(",") if part.strip()
    )
    for name in names:
        if len(name) < 3:
            raise ValueError(
                f"required input {name!r} is too short to check; name it the way a knob "
                "or a file would be named, e.g. `package` or `source_tree`"
            )
    return names


def baked_item_problems(root: Path) -> list[str]:
    """Every place a package bakes item data into its graph instead of reading it.

    A data node is the harness's input port. Its items come from the Task it binds or
    from the `source_path`/`source_format` it names, loaded at run time by
    :func:`refactory_dsh.nn.executor.load_items`; `inline_items` is read first by that
    loader, so a package that sets it never reads the run's items at all. That is the
    same failure as hardcoding one sample, one level up. The check reads the source
    because the driver must not import model-written code, and it names the file and
    line so the emit step can act on it.
    """
    import ast

    problems: list[str] = []
    seen: set[tuple[str, int]] = set()
    for path in sorted(Path(root).rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except (SyntaxError, OSError):
            # `ast_check` already reports a file that does not parse.
            continue
        for node in ast.walk(tree):
            baked = (
                isinstance(node, ast.keyword) and node.arg == BAKED_ITEMS_KEY
            ) or (
                isinstance(node, ast.Dict)
                and any(
                    isinstance(key, ast.Constant) and key.value == BAKED_ITEMS_KEY
                    for key in node.keys
                )
            )
            if not baked:
                continue
            where = (path.name, int(getattr(node, "lineno", 0)))
            if where in seen:
                continue
            seen.add(where)
            problems.append(
                f"{where[0]}:{where[1]} bakes a fixed item list into the graph with "
                f"`{BAKED_ITEMS_KEY}`; a data node takes the items of the class of work "
                "from the Task it binds or from the `source_path` it names, loaded at "
                "run time, so the harness can be fed any item and not only the ones the "
                "authoring run seeded"
            )
    return problems


def _empty_literal(value: Any) -> bool:
    """Whether a source expression is a literal that names nothing.

    A keyword the emit step writes as `task_ref=""` or `source_path=""` names no
    source; an expression it computes at import time is left alone, because the
    driver cannot evaluate it and the graph-level checks still run.
    """
    import ast

    return isinstance(value, ast.Constant) and not str(value.value or "").strip()


def _names_a_run_time_source(fields: dict[str, Any]) -> bool:
    """Whether a data node's own fields feed it from a Task or a source."""
    task_ref = fields.get("task_ref")
    if task_ref is not None and not _empty_literal(task_ref):
        return True
    source_path = fields.get("source_path")
    source_format = fields.get("source_format")
    if source_path is None or source_format is None:
        return False
    return not _empty_literal(source_path) and not _empty_literal(source_format)


def unfeedable_data_node_problems(root: Path) -> list[str]:
    """Every data node that names no way to be fed at run time.

    A data node is the harness's input port, and the executor feeds it from the Task
    it binds or from the `source_path` and `source_format` it names. A node that
    declares neither iterates nothing however much the class of work holds, so the
    port is closed and the harness is not a harness. The check reads the source
    because the driver must not import model-written code, and it judges only a
    `Node(...)` call whose kind is the literal `"data"`: a node built by a helper is
    left to the graph-level checks.
    """
    import ast

    problems: list[str] = []
    for path in sorted(Path(root).rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if ast.unparse(node.func).rsplit(".", 1)[-1] != "Node":
                continue
            kind = ""
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                kind = str(node.args[1].value)
            for keyword in node.keywords:
                if keyword.arg == "kind" and isinstance(keyword.value, ast.Constant):
                    kind = str(keyword.value.value)
            if kind != "data":
                continue
            fields = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
            if _names_a_run_time_source(fields):
                continue
            line = int(getattr(node, "lineno", 0))
            problems.append(
                f"{path.name}:{line} declares a `data` node that names no way to be fed "
                "at run time; it takes the items of the class of work from `task_ref`, "
                "or from `source_path` with a `source_format` of `directory`, `jsonl` "
                "or `csv`, so the harness can be fed from outside the package"
            )
    return problems


def check_package(
    name: str,
    *,
    requires_data_node: bool = False,
    requires_task_ref: str = "",
    requires_task_refs: tuple[str, ...] = (),
    requires_input: tuple[str, ...] = (),
    project_override: Path | None = None,
) -> list[str]:
    """Everything wrong with a staged package, as a list of reasons.

    A generated package is **never imported here**. It is parsed, and then every
    semantic check — the name, the ops modules, each mode's construction, its score
    and its knobs, every validator rule — runs in a subprocess, because loading a
    package is executing it and the model's code has no business running in the
    driver. See :func:`refactory_dsh.sandbox.verify_sandboxed`.

    The two checks are both run rather than the first that fires: the retry prompt
    is more useful the more of the truth it carries, and parsing and semantics
    catch different things.

    ``project_override``, when given, is the directory whose ``.factory/packages/``
    holds the staged package (e.g. the per-run staging base).  Without it, the
    shared project checkout is used.
    """
    from refactory_dsh import sandbox

    shared_project = project_for_nodes() if project_override else None
    project = project_override or project_for_nodes()
    registry.install_workspace(project)
    if shared_project:
        registry.install_workspace(shared_project)
    root = registry.workspace_root(project) / name
    if not root.is_dir():
        return [f"no directory at {root}"]
    problems: list[str] = []
    for required in REQUIRED_FILES:
        if not (root / required).exists():
            problems.append(
                sandbox.missing_init(root)
                if required == "__init__.py"
                else f"{required} is missing from the package root {root}; a package needs one"
            )
    if problems:
        return problems

    problems.extend(sandbox.ast_check(name, project))
    if any("does not parse" in problem for problem in problems):
        # Nothing further can be learned from a file that is not valid Python.
        return problems
    # Read before the graph is loaded: a harness whose items are baked in, or whose
    # input port names nothing, is refused for the same reason a hardcoded sample is,
    # and the emit step can fix the source.
    problems.extend(baked_item_problems(root))
    problems.extend(unfeedable_data_node_problems(root))

    checked = sandbox.verify_sandboxed(
        name,
        project=project,
        shared_project=shared_project,
        requires_data_node=requires_data_node,
        requires_task_ref=requires_task_ref,
        requires_task_refs=requires_task_refs,
        requires_input=requires_input,
    )
    problems.extend(checked.payload.get("problems") or [])
    if not checked.ok and not problems:
        return [checked.problem() or "the structural check could not be run"]
    return problems


def register(ctx: Any) -> dict[str, Any]:
    """Confirm the registry now sees the package under its own name.

    When the run staged the package in its own directory (``prepare_create``), copy it
    atomically to the shared ``.factory/packages/<name>/``.  If the shared package's
    state differs from what ``prepare`` recorded — including when it now exists and did
    not before — refuse and name the other run (read from ``.origin.json``).

    Reloop safety: when the shared package's ``.origin.json`` names this run, the
    conflict check is skipped — this run already registered it on a previous iteration,
    and re-registering the updated staging copy is the intended behaviour.
    """
    name = str(ctx.facts["name"])
    staging_root = ctx.facts.get("package_root")
    if staging_root:
        import tempfile
        shared = registry.workspace_root(project_for_nodes()) / name
        # Detect whether the shared package changed since prepare.
        source_hash = ctx.facts.get("source_hash", "")
        source_existed = ctx.facts.get("source_existed", source_hash != "")
        own_run_id = _invocation_run_id(ctx.workdir)
        # When editing from a different source package, verify the source
        # hasn't changed since prepare recorded its hash.
        _revision_of_fact = str(ctx.facts.get("revision_of") or "")
        _source_pkg = _revision_of_fact.split("/")[0] if _revision_of_fact else ""
        if _source_pkg and _source_pkg != name:
            source_loc = registry.workspace_root(project_for_nodes()) / _source_pkg
            if source_existed:
                if source_loc.exists():
                    current_source = _dir_hash(source_loc)
                    if current_source != source_hash:
                        other_run = _origin_run_id(source_loc)
                        other_info = f" (registered by run {other_run})" if other_run else ""
                        raise ValueError(
                            f"the source package {_source_pkg!r} changed since this run's "
                            f"prepare (hash {source_hash[:12]} -> {current_source[:12]})"
                            f"{other_info}; another run registered it first. This run's "
                            f"staged package is kept at {staging_root} for inspection."
                        )
                else:
                    raise ValueError(
                        f"the source package {_source_pkg!r} existed when this run's "
                        f"prepare ran, but it is gone now; another run deleted it. "
                        f"This run's staged package is kept at {staging_root} "
                        f"for inspection."
                    )
            # For the destination check below: use the destination's recorded
            # state from prepare, not the source's.  An edit whose destination
            # already existed at prepare must not be falsely refused.
            dest_hash = ctx.facts.get("dest_hash", "")
            dest_existed = ctx.facts.get("dest_existed", False)
            source_hash = dest_hash
            source_existed = dest_existed
        if shared.exists():
            current = _dir_hash(shared)
            # A reloop: this run already registered the package on a previous
            # iteration.  The origin stamp written by _stamp_origin_run_id
            # identifies the registering run, so a match means the shared
            # package is ours and safe to overwrite.
            is_reloop = _origin_run_id(shared) == own_run_id
            if not is_reloop:
                if not source_existed:
                    # Package did not exist at prepare but now does: another run created it.
                    other_run = _origin_run_id(shared)
                    other_info = f" (registered by run {other_run})" if other_run else ""
                    raise ValueError(
                        f"the shared package {name!r} did not exist when this run's prepare "
                        f"ran, but it exists now{other_info}; another run registered it first. "
                        f"This run's staged package is kept at {staging_root} for inspection."
                    )
                if current != source_hash:
                    other_run = _origin_run_id(shared)
                    other_info = f" (registered by run {other_run})" if other_run else ""
                    raise ValueError(
                        f"the shared package {name!r} changed since this run's prepare "
                        f"(hash {source_hash[:12]} -> {current[:12]}){other_info}; another run "
                        f"registered it first. This run's staged package is kept at "
                        f"{staging_root} for inspection."
                    )
        elif source_existed:
            raise ValueError(
                f"the shared package {name!r} existed when this run's prepare ran, "
                f"but it is gone now; another run deleted it. "
                f"This run's staged package is kept at {staging_root} for inspection."
            )
        # Record the source-versus-staged diff for reviewers before copying.
        _record_staging_diff(ctx, Path(staging_root))
        # Write this run's id into the staged .origin.json before copying, and that its
        # smoke has not passed yet: the listing marks it until it does (dm-264).
        _stamp_origin_run_id(Path(staging_root), ctx.workdir)
        registry.record_smoke(Path(staging_root), registry.SMOKE_PENDING)
        # Atomic swap: write to a sibling temp dir, rename the old aside, rename
        # the new in, then delete the old.
        shared.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=shared.parent))
        old_aside: Path | None = None
        try:
            dest = temp / name
            shutil.copytree(staging_root, dest)
            if shared.exists():
                old_aside = shared.with_name(f".{name}-old-{os.getpid()}")
                shared.rename(old_aside)
            dest.rename(shared)
        except BaseException:
            # Roll back: if we moved the old aside, put it back.
            if old_aside is not None and old_aside.exists() and not shared.exists():
                old_aside.rename(shared)
            raise
        finally:
            if old_aside is not None and old_aside.exists():
                shutil.rmtree(old_aside, ignore_errors=True)
            if temp.exists():
                shutil.rmtree(temp, ignore_errors=True)
    elif (registry.workspace_root(project_for_nodes()) / name).is_dir():
        # Written in place by `materialise`: it is listed from now on, unverified.
        registry.record_smoke(registry.workspace_root(project_for_nodes()) / name,
                              registry.SMOKE_PENDING)
    registry.install_workspace(project_for_nodes())
    importlib.invalidate_caches()
    if name not in registry.discover_names(project_for_nodes()):
        raise ValueError(f"the registry does not list a package named {name!r}")
    # Read the manifest where it is safe to load it. Reporting what was registered
    # must not run the package in the driver's process to find out.
    from refactory_dsh import sandbox

    described = sandbox.describe_package(name, project=project_for_nodes())
    if not described.ok:
        raise ValueError(
            f"the package is registered but not readable: {described.problem()}"
        )
    ctx.write_json("registered.json", described.payload["describe"])
    result: dict[str, Any] = {"registered": ", ".join(described.payload["ids"])}
    if staging_root:
        # Point package_path to the shared location so downstream steps
        # (smoke / _revision) write provenance into the installed package,
        # not the per-run staging copy that register already consumed.
        result["package_path"] = str(registry.workspace_root(project_for_nodes()) / name)
    return result


def _record_staging_diff(ctx: Any, staged_root: Path) -> None:
    """Record the diff between the source package and the staged result for reviewers.

    ``prepare_create`` saves a snapshot of the source package's files in
    ``staging/source-snapshot.json``.  This function compares that snapshot with
    the staged package (after ``materialise`` wrote into it) and writes a
    structured diff to ``staging/package-diff.json`` so reviewers can see exactly
    what the run changed.  For a new create every file is ``added``; for an edit
    the diff shows modifications, additions and deletions.
    """
    import difflib

    source_snapshot: dict[str, str | None] = ctx.read_json(
        f"{STAGING}/source-snapshot.json", {}
    )
    staged_files: dict[str, str | None] = {}
    for path in sorted(staged_root.rglob("*")):
        if path.is_file():
            rel = str(path.relative_to(staged_root))
            try:
                staged_files[rel] = path.read_text()
            except (UnicodeDecodeError, OSError):
                staged_files[rel] = None

    all_paths = sorted(set(source_snapshot) | set(staged_files))
    if not all_paths:
        return

    changes: list[dict[str, str]] = []
    diff_lines: list[str] = []
    for rel in all_paths:
        old_content = source_snapshot.get(rel)
        new_content = staged_files.get(rel)
        in_source = rel in source_snapshot
        in_staged = rel in staged_files

        if not in_source:
            changes.append({"path": rel, "kind": "added"})
            diff_lines.append(f"--- /dev/null")
            diff_lines.append(f"+++ staged/{rel}")
            if new_content is not None:
                for line in new_content.splitlines():
                    diff_lines.append(f"+{line}")
            diff_lines.append("")
        elif not in_staged:
            changes.append({"path": rel, "kind": "deleted"})
            diff_lines.append(f"--- source/{rel}")
            diff_lines.append(f"+++ /dev/null")
            if old_content is not None:
                for line in old_content.splitlines():
                    diff_lines.append(f"-{line}")
            diff_lines.append("")
        elif old_content != new_content:
            changes.append({"path": rel, "kind": "modified"})
            if old_content is not None and new_content is not None:
                diff = difflib.unified_diff(
                    old_content.splitlines(keepends=True),
                    new_content.splitlines(keepends=True),
                    fromfile=f"source/{rel}",
                    tofile=f"staged/{rel}",
                )
                diff_lines.extend(line.rstrip("\n") for line in diff)
            else:
                diff_lines.append(f"--- source/{rel} (binary)")
                diff_lines.append(f"+++ staged/{rel} (binary)")
                diff_lines.append("Binary files differ")
            diff_lines.append("")

    ctx.write_json(f"{STAGING}/package-diff.json", {
        "source_file_count": len(source_snapshot),
        "staged_file_count": len(staged_files),
        "changes": changes,
        "diff": "\n".join(diff_lines),
    })


def _origin_run_id(shared: Path) -> str:
    """Read the registering run's id from the package's ``.origin.json``, or empty."""
    origin_path = shared / ".origin.json"
    if origin_path.exists():
        try:
            data = json.loads(origin_path.read_text())
            return str(data.get("run_id", ""))
        except (json.JSONDecodeError, OSError):
            pass
    return ""


def _invocation_run_id(workdir: Path) -> str:
    """The run's invocation id, or the workdir basename as fallback.

    Uses the same source as ``repo_change._run_id``: the ``invocation.json``
    file the spine writes into the run directory before the first node runs.
    """
    from refactory_dsh import worktree as _wt
    inv_path = Path(workdir) / _wt.INVOCATION
    if inv_path.exists():
        try:
            data = json.loads(inv_path.read_text())
            found = str(data.get("id") or "")
            if re.fullmatch(r"[0-9a-f]{12,}", found):
                return found[:12]
        except (json.JSONDecodeError, OSError):
            pass
    return Path(workdir).name


def _stamp_origin_run_id(staging_root: Path, workdir: Path) -> None:
    """Write the registering run's id into the staged package's ``.origin.json``."""
    origin_path = staging_root / ".origin.json"
    run_id = _invocation_run_id(workdir)
    try:
        data = json.loads(origin_path.read_text()) if origin_path.exists() else {}
    except (json.JSONDecodeError, OSError):
        data = {}
    data["run_id"] = run_id
    origin_path.write_text(json.dumps(data, indent=2))


def _dir_hash(root: Path) -> str:
    """A content hash of every file under ``root``, for change detection.

    Skips ``__pycache__`` directories and ``.pyc`` bytecode files, as the
    repository's other content hashes do (``authoring_evidence.py`` and
    ``registry.py``).  An import of the shared package between prepare and
    register must not change the hash.
    """
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            h.update(str(path.relative_to(root)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def prepare_create(ctx: Any) -> dict[str, Any]:
    """Read the brief and set up a per-run staging directory for the package.

    This replaces ``prepare()`` for create/edit modes that need isolation.  The
    package is written into a staging directory inside the run directory rather
    than the shared ``.factory/packages/<name>/``, so two parallel runs of the
    same package name never collide.  For edit mode, the source package is copied
    into the staging directory and its content hash recorded so ``register()``
    can detect mid-run changes.

    Why staging rather than a git worktree: ``.factory/`` is in ``.gitignore``
    in this repository, so a git worktree of the project contains no
    ``.factory/packages/``, and nothing written there can become a commit.
    """
    # First, run the original prepare to get all the standard facts.
    prepared = prepare(ctx)
    name = str(prepared["name"])

    # Set up the staging directory.  Resolve to an absolute path so that
    # downstream subprocess calls (verify_sandboxed) can find the package when
    # their cwd is set to the staging project directory.
    staging_base = Path(ctx.workdir).resolve() / STAGING_DIR
    pkg_root = staging_base / ".factory" / "packages" / name
    pkg_root.mkdir(parents=True, exist_ok=True)

    # For edit mode: if a source package exists, copy it into staging.
    # Record both the hash and whether the package existed, so register can
    # refuse when a new package appears (source_hash="" was previously skipped).
    source_hash = ""
    source_existed = False
    # When revision_of is set (edit mode), use the package component of
    # revision_of (the part before the first '/') to find the source.
    revision_of = str(prepared.get("revision_of") or "")
    if revision_of:
        source_pkg_name = revision_of.split("/")[0]
    else:
        source_pkg_name = name
    shared = registry.workspace_root(project_for_nodes()) / source_pkg_name

    # Snapshot the source package's files so register() can record a diff for
    # reviewers.  For a new create the snapshot is empty; for edit it captures
    # every file in the shared package before the run touches it.
    source_files: dict[str, str | None] = {}
    if shared.is_dir():
        # Copy existing package as the starting point for an edit.
        shutil.copytree(shared, pkg_root, dirs_exist_ok=True)
        source_hash = _dir_hash(shared)
        source_existed = True
        for _src_path in sorted(shared.rglob("*")):
            if _src_path.is_file():
                if "__pycache__" in _src_path.parts or _src_path.suffix == ".pyc":
                    continue
                _rel = str(_src_path.relative_to(shared))
                try:
                    source_files[_rel] = _src_path.read_text()
                except (UnicodeDecodeError, OSError):
                    source_files[_rel] = None  # binary or unreadable
    ctx.write_json(f"{STAGING}/source-snapshot.json", source_files)

    # When revision_of differs from name (editing into a new destination),
    # record the destination's state separately so register() can check each
    # against its own record.
    dest_hash = ""
    dest_existed = False
    if revision_of and source_pkg_name != name:
        dest_shared = registry.workspace_root(project_for_nodes()) / name
        if dest_shared.is_dir():
            dest_hash = _dir_hash(dest_shared)
            dest_existed = True

    return {
        **prepared,
        "package_root": str(pkg_root),
        "source_hash": source_hash,
        "source_existed": source_existed,
        "dest_hash": dest_hash,
        "dest_existed": dest_existed,
    }


#: A wheel's mandatory members, beside `RECORD`, for it to be a wheel at all.
WHEEL_REQUIRED = ("METADATA", "WHEEL")

#: How deep to look for the artifacts a run produced.
ARTIFACT_DEPTH = 6


def artifact_problems(root: Path) -> list[str]:
    """What is wrong with the archives a run produced, judged by their own format.

    No domain knowledge and no declaration needed: a file that says it is a wheel has
    to *be* one. This is what the other half of #69 needed — the revision's new input
    path emitted a `.dist-info` holding only `RECORD`, which is not a wheel, and the
    default fixture path emitted all three, so nothing noticed.

    A `.whl` or `.zip` that cannot be opened is a problem; a wheel missing `METADATA`
    or `WHEEL` beside its `RECORD` is a problem. Both name the file so the emit step
    can act on them.
    """
    import zipfile

    problems: list[str] = []
    if not Path(root).is_dir():
        return problems
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() not in (".whl", ".zip") or not path.is_file():
            continue
        if len(path.relative_to(root).parts) > ARTIFACT_DEPTH:
            continue
        try:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
        except (zipfile.BadZipFile, OSError) as exc:
            problems.append(f"{path.relative_to(root)} is not a readable archive: {exc}")
            continue
        if path.suffix.lower() != ".whl":
            continue
        for required in WHEEL_REQUIRED:
            if not any(
                name.endswith(f".dist-info/{required}") for name in names
            ):
                problems.append(
                    f"{path.relative_to(root)} is not a valid wheel: it has no "
                    f"{required} beside its RECORD. A wheel needs the metadata that "
                    "names the distribution and the version of the wheel format."
                )
    return problems


def _sample_knobs(raw: Any) -> dict[str, str]:
    """The knob values a validation run should use, as `NAME=VALUE` pairs.

    A person states them when the brief's inputs are real: without them the smoke runs
    the default path, and the path the brief asked for goes untested. The value travels
    as a JSON array of strings in a knob (so a path containing `=`, a comma or a space
    survives), or as a list when a caller passes it directly.
    """
    items: Any = raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            items = []
        else:
            try:
                items = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"the sample knobs are not a JSON array of NAME=VALUE strings: {exc}"
                ) from exc
    pairs: dict[str, str] = {}
    for item in list(items or []):
        name, separator, value = str(item).partition("=")
        if not separator or not name.strip() or not value.strip():
            raise ValueError(f"--sample takes NAME=VALUE, not {item!r}")
        pairs[name.strip()] = value.strip()
    return pairs


def _model_context(ctx: Any) -> dict[str, str]:
    from refactory_dsh.model import DshModelClient

    client = getattr(ctx, "model", None)
    return (client if isinstance(client, DshModelClient) else DshModelClient()).context()


def _smoke_timeout_record(result: Any, identifier: str, workdir: Path, *, model: bool) -> dict[str, Any]:
    """Describe an incomplete pass without claiming a score or model request occurred."""
    return {
        "mode": identifier, "ok": False, "score": None, "timed_out": True,
        "returncode": result.returncode, "timeout_s": result.timeout_s,
        "elapsed_s": result.elapsed_s, "progress": result.progress,
        "workdir": str(workdir), "sandboxed": True,
        "model": "requested" if model else "none", "error": result.problem(),
    }


def smoke(ctx: Any) -> dict[str, Any]:
    """One canonical execution of the revision, followed by optional comparison."""
    from refactory_dsh import sandbox
    from refactory_dsh.authoring_evidence import failure_kind
    import uuid

    name = str(ctx.facts["name"])
    revision = str(ctx.facts.get("revision_of") or "").strip()
    # Distinct directories preserve outputs from earlier repair attempts.
    workdir = ctx.workdir / "smoke" / uuid.uuid4().hex
    timeout_s = int(ctx.knob("smoke_timeout_s", sandbox.DEFAULT_MODEL_SMOKE_TIMEOUT_S))
    if not 1 <= timeout_s <= 86400:
        raise ValueError("smoke_timeout_s must be between 1 and 86400 seconds")
    described = sandbox.describe_package(name, project=project_for_nodes())
    if not described.ok:
        raise ValueError(f"the package could not be read: {described.problem()}")
    identifier = described.payload["ids"][0]
    samples = _sample_knobs(ctx.knob("sample_knobs", ""))
    agent_nodes = _agent_nodes(name)
    context = _model_context(ctx)
    budget = getattr(ctx, "budget", None)
    if budget is not None:
        budget.checkpoint()
        context["budget_ledger"] = str(budget.path)
    result = sandbox.smoke(
        identifier, workdir, project=project_for_nodes(), model=bool(agent_nodes),
        knobs=samples or None, model_context=context, timeout_s=timeout_s,
    )
    error = result.problem() if not result.ok else str(result.payload.get("error") or "")
    ok = result.ok and result.payload.get("exit_ok") is not False
    payload = {
        "mode": identifier, "ok": ok, "score": result.payload.get("score") if ok else None,
        "iterations": result.payload.get("iterations"), "workdir": str(workdir.resolve()),
        "sandboxed": True, "model": "used" if agent_nodes else "none",
        "timeout_s": timeout_s, "elapsed_s": result.elapsed_s,
        "timed_out": result.timed_out, "returncode": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr, "error": error,
        "sample_knobs": samples,
        "status": "pass" if ok else failure_kind(error, timed_out=result.timed_out, returncode=result.returncode),
    }
    ctx.write_json("smoke.json", payload)
    if budget is not None:
        budget.checkpoint()
    if not ok:
        qualifier = " with a model" if agent_nodes else ""
        raise ValueError(f"the new mode did not finish a run{qualifier}: {error or 'it failed without saying why'}")
    _check_artifacts(workdir, samples)
    score = float(result.payload.get("score") or 0.0)
    revised = _revision(ctx, revision, score, not_comparable=(
        f"{identifier} calls a model ({', '.join(agent_nodes)}), and one sample of a model call "
        "is not a score to refuse a revision on" if agent_nodes else ""),
        revised_knobs=result.payload.get("knobs"))
    return {"smoke_score": score, "smoke_ok": True,
            **({"smoke_model": "used"} if agent_nodes else {}), **revised}


def repair(ctx: Any, findings: list[str], *, kind: str = "check",
           early_stop: bool = False) -> str:
    """Share the existing full-mode attempt allowance across all build failures.

    ``kind`` is ``'check'`` for a failed runtime check and ``'review'`` for a
    review-driven reloop.  Each kind has its own counter and limit; both
    increment ``attempts_used`` against the overall cap.
    """
    feedback = "\n".join(findings)
    used = int(ctx.facts.get("attempts_used", 0)) + 1
    ctx.write_text("validation.txt", feedback + "\n")
    ctx.set_fact("validation", feedback)
    ctx.set_fact("attempts_used", used)

    if kind == "check":
        kind_used = int(ctx.facts.get("check_attempts_used", 0)) + 1
        ctx.set_fact("check_attempts_used", kind_used)
        kind_limit = int(ctx.knob("check_attempts", ctx.knob("attempts", 3)))
        if kind_used >= kind_limit:
            if not early_stop:
                # R6: rolling convergence — grant another check attempt each time
                # the failure set is a non-empty strict subset of the previous
                # one, with no new failures and the runner completed fully.
                # Repeats at every attempt past the regular limit, up to
                # check_attempt_cap total check attempts.  Replaces the old
                # one-shot convergence_extra_attempts knob.
                from refactory_dsh import base_check
                history = list(ctx.facts.get("check_failures_history", []))
                extras_used = int(ctx.facts.get("convergence_extras_used", 0))
                cap = int(ctx.knob("check_attempt_cap", 8))
                # 0 or a value at or below the regular attempts disables extension.
                granted = False
                if cap > kind_limit and kind_used < cap and len(history) >= 2:
                    current = set(history[-1])
                    previous = set(history[-2])
                    # R6 correction 3: judge from the recorded check output,
                    # not the feedback text.  The check_blocker_history fact
                    # records the excusal_blocker result computed from the full
                    # check output at each attempt.
                    blocker_history = list(ctx.facts.get("check_blocker_history", []))
                    if blocker_history:
                        blocker = blocker_history[-1]
                    else:
                        # Fallback: compute from feedback (first attempt).
                        check_command = str(ctx.facts.get("check") or "").strip()
                        blocker = base_check.excusal_blocker(
                            check_command, feedback, current
                        ) if current else "empty failure set"
                    all_prior = set().union(*(set(h) for h in history[:-1]))
                    if (current
                            and current < previous
                            and current <= all_prior
                            and blocker is None):
                        ctx.set_fact("convergence_extras_used", extras_used + 1)
                        ctx.set_fact("convergence_granted", True)
                        granted = True
                if not granted:
                    if cap > kind_limit and kind_used >= cap:
                        raise BuildLoopStop(
                            f"check attempt cap ({cap}) reached after {kind_used} "
                            f"check attempt(s): {feedback}"
                        )
                    raise BuildLoopStop(
                        f"check attempts exhausted after {kind_used} check failure(s): {feedback}"
                    )
            else:
                # Early stop: do not grant convergence.
                raise BuildLoopStop(
                    f"check attempts exhausted after {kind_used} check failure(s): {feedback}"
                )
    else:
        kind_used = int(ctx.facts.get("review_attempts_used", 0)) + 1
        ctx.set_fact("review_attempts_used", kind_used)
        kind_limit = int(ctx.knob("review_attempts", ctx.knob("check_attempts", ctx.knob("attempts", 3))))
        if kind_used >= kind_limit:
            raise BuildLoopStop(
                f"review reloop limit reached after {kind_used} review cycle(s): {feedback}"
            )

    overall_limit = int(ctx.knob("attempts", ctx.knob("check_attempts", 3)))
    # R6: apply the granted extra attempt to the overall cap too.
    extras_granted = int(ctx.facts.get("convergence_extras_used", 0))
    if used >= overall_limit + extras_granted:
        raise BuildLoopStop(f"authoring unresolved after {used} attempt(s): {feedback}")
    return Verdict("reloop", feedback)


def gate_smoke(ctx: Any) -> str:
    from refactory_dsh import authoring_evidence as evidence

    # Never let a previous successful smoke survive a failed new attempt: not in the
    # facts, and not on the package, which the listing marks until its smoke passes.
    # `.origin.json` is in the hash a concurrent run's register compares, so it is
    # written only when its record changes: once after registering, on a pass.
    ctx.set_fact("smoke_ok", False)
    package = Path(str(ctx.facts.get("package_path") or ""))
    registered = bool(ctx.facts.get("package_path")) and (package / registry.ORIGIN).is_file()
    if registered and registry.recorded_smoke(package) != registry.SMOKE_PENDING:
        registry.record_smoke(package, registry.SMOKE_PENDING)
    for name in ("smoke.json", "revision.json"):
        if ctx.exists(name):
            path = ctx.workdir / ".refactory" / "validation-history" / "prior"
            path.mkdir(parents=True, exist_ok=True)
            import uuid
            (path / (uuid.uuid4().hex + "-" + name)).write_text(ctx.read_text(name))
            (ctx.workdir / name).unlink()
    try:
        produced = smoke(ctx)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        receipt = ctx.read_json("smoke.json", {})
        kind = receipt.get("status") if receipt.get("status") not in (None, "pass") else evidence.failure_kind(detail)
        if "revision source" in detail and "did not finish" in detail:
            kind = "source_unavailable"
        evidence.record(ctx, {"status": kind, "detail": detail,
                              "smoke": receipt, "revision": ctx.read_json("revision.json", None)})
        evidence.stop_unrepairable(kind, detail)
        from refactory_dsh.fast_ops import _repair
        return (_repair if "repair_attempts" in ctx.knobs else repair)(ctx, [detail])
    for key, value in produced.items():
        ctx.set_fact(key, value)
    if registered and registry.recorded_smoke(package) != registry.SMOKE_PASSED:
        registry.record_smoke(package, registry.SMOKE_PASSED)
    evidence.record(ctx, {"status": "pass", "smoke": ctx.read_json("smoke.json", {}),
                          "revision": ctx.read_json("revision.json", None)})
    return "proceed"


def _changed_defaults(state: dict[str, Any], revised_knobs: Any,
                      samples: dict[str, str]) -> list[str]:
    """The knobs the revision runs with another value than its source does (dm-263).

    Both smokes run on their own defaults, except the sample knobs both are given, so a
    default the revision changed moves its score without the change being better: an
    edit that raised `budget` from 3 to 8 scored 0.875 → 1.08. Each as ``budget 3 → 8``.
    Only the knobs both modes have: a knob the revision adds is part of its change.
    """
    if not isinstance(revised_knobs, dict):
        return []
    source = {name: entry.get("value") for name, entry in (state.get("knobs") or {}).items()
              if isinstance(entry, dict)}
    # Compared as JSON, the form the revision's values came back in.
    source = json.loads(json.dumps(source, default=str))

    def shown(value: Any) -> str:
        return value if isinstance(value, str) else json.dumps(value, default=str)

    return [f"{name} {shown(source[name])} → {shown(revised_knobs[name])}"
            for name in sorted(set(source) & set(revised_knobs) - set(samples))
            if source[name] != revised_knobs[name]]


def _revision(
    ctx: Any, revision: str, after: float | None, *, not_comparable: str = "",
    revised_knobs: Any = None,
) -> dict[str, Any]:
    """Score the mode a revision came from, and refuse a revision that scores lower.

    The comparison is only meaningful between two modes a single smoke run can be
    repeated on: one sample of a mode that calls a model is not a score to refuse on,
    and a source mode with no ``score()`` has no number to compare against. Both cases
    are recorded with the reason, because "not compared" and "compared and fine" must
    not read alike.

    The provenance is written next to the revision rather than kept in the run: what a
    package came from, at which state, and what it scored are facts about the package
    that outlive the run directory. ``revision_of_state`` is the source's state dict at
    the moment of the revision, so "revised from X" means a state and not a name that
    may have moved since.
    """
    if not revision:
        return {}
    from refactory_dsh import registry, sandbox

    project = project_for_nodes()
    try:
        package, mode_class = registry.resolve(project, revision)
    except Exception as exc:
        raise ValueError(f"cannot revise {revision!r}: {exc}") from exc
    source = mode_class()
    state = source.state_dict()
    from refactory_dsh.optim.base import candidate_id

    task = ctx.task if isinstance(getattr(ctx, "task", None), dict) else {}
    original_state = task.get("revision_of_state")
    if original_state is not None and state != original_state:
        raise ValueError(
            f"the revision source {revision} changed state during authoring; "
            "start a new revision against its current state"
        )
    allowed = bool(ctx.knob("allow_regression", 0))
    reason = ("comparison intentionally skipped: allow_regression is enabled; "
              "the original mode was not executed and non-regression was not measured"
              if allowed else not_comparable)
    if not reason and getattr(mode_class, "unscored_reason", ""):
        reason = f"{revision} is unscored, so there is no score to compare against"
    if not reason and [n for n in source.build().nodes.values() if n.kind == "agent"]:
        reason = f"{revision} calls a model, and one sample of it is not a score"
    samples = _sample_knobs(ctx.knob("sample_knobs", ""))
    missing_samples = sorted(set(samples) - set(state["knobs"]))
    if not reason and missing_samples:
        reason = (
            f"{revision} does not accept the sample knobs {', '.join(missing_samples)}, "
            "so the source cannot be measured on the revision's inputs"
        )

    before: float | None = None
    if not reason:
        scored = sandbox.smoke(
            revision, ctx.workdir / "smoke-source", project=project,
            knobs=samples or None,
            model_context=_model_context(ctx),
            timeout_s=int(ctx.knob("smoke_timeout_s", sandbox.DEFAULT_MODEL_SMOKE_TIMEOUT_S)),
        )
        if scored.timed_out:
            # Keep the new mode's completed smoke receipt intact. The source pass
            # is incomplete, so neither non-regression nor provenance is established.
            ctx.write_json("revision.json", {
                **_smoke_timeout_record(scored, revision, ctx.workdir / "smoke-source", model=False),
                "phase": "source-comparison", "revision_of": revision,
                "before_score": None, "after_score": after, "regressed": None,
            })
            raise ValueError(f"the revision source {revision} did not finish a run: {scored.problem()}")
        if not scored.ok or scored.payload.get("exit_ok") is False:
            error = scored.problem() if not scored.ok else (
                scored.payload.get("error") or "it failed without saying why"
            )
            ctx.write_json("revision.json", {
                "phase": "source-comparison", "revision_of": revision,
                "before_score": None, "after_score": after, "regressed": None,
                "exit_ok": False, "returncode": scored.returncode, "error": error,
                "workdir": str(ctx.workdir / "smoke-source"), "sandboxed": True,
                "sample_knobs": samples,
            })
            raise ValueError(f"the revision source {revision} did not finish a run: {error}")
        before = float(scored.payload.get("score") or 0.0)

    regressed = None if allowed else bool(before is not None and after is not None and after < before)
    # Said beside the comparison, not acted on: whether to re-score on equal settings is
    # a separate decision.
    changed = _changed_defaults(state, revised_knobs, samples)
    recorded = {
        "revision_of": revision,
        "revision_of_state": state,
        "revision_of_digest": candidate_id(state),
        "before_score": before,
        "after_score": after,
        "regressed": regressed,
        "allow_regression": allowed,
        "changed_defaults": changed,
        "brief": ctx.facts.get("brief", ""),
        "note": reason,
    }
    ctx.write_json("revision.json", recorded)
    # Where the package is, when the run has got that far. A node invoked without one
    # (a test, or a run that stopped earlier) must not write provenance into whatever
    # directory it happens to be standing in.
    package = str(ctx.facts.get("package_path") or "")
    if package:
        path = Path(package) / ".origin.json"
        current = json.loads(path.read_text()) if path.exists() else {}
        current.update({key: value for key, value in recorded.items() if key != "note"})
        if not (regressed and not allowed):
            # The smoke passed and the revision stands: recorded in this same write, so
            # the file changes once after registering (`gate_smoke`).
            current[registry.SMOKE] = registry.SMOKE_PASSED
        path.write_text(json.dumps(current, indent=2, default=str))
    if regressed and not allowed:
        raise ValueError(
            f"the revision scores {after} where {revision} scores {before}, so it is not "
            "an improvement; pass `--knob allow_regression=1` if that is intended"
        )
    return {
        "before_score": before,
        "after_score": after,
        "regressed": regressed,
        "revision_note": (
            reason or f"compared with {revision}: {before} → {after}"
            + (f" (not like for like: {', '.join(changed)})" if changed else "")
        ),
    }


def _check_artifacts(workdir: Path, samples: dict[str, str]) -> None:
    """Refuse a run whose archives are not the format they claim to be.

    The score is not enough: a mode can score above zero while emitting a file that is
    not a wheel. A sample run is the one that produces a real artifact, so this is
    where the check belongs.
    """
    problems = artifact_problems(workdir)
    if not problems:
        return
    with_samples = f" (run with {samples})" if samples else ""
    detail = "\n".join(f"- {problem}" for problem in problems)
    raise ValueError(
        f"the run produced an artifact that is not what it claims to be{with_samples}:\n{detail}"
    )


def _could_not_reach_a_model(result: Any) -> bool:
    """Whether a smoke failed because there was no model, rather than because of the mode.

    The two are different facts: a mode that cannot reach a model is unmeasured, and a
    mode that ran and failed is a defect the emit step has to fix.
    """
    problem = result.problem().lower()
    return any(
        marker in problem
        for marker in ("needs a model client", "model completion failed", "no model", "credentials", "api key")
    )


def _agent_nodes(name: str) -> list[str]:
    """The agent nodes of a package's first mode, or an empty list.

    Read from the package rather than from a prompt: "does this mode call a model" is
    a property of its graph.
    """
    from refactory_dsh import registry

    try:
        package = registry.load(name, project_for_nodes())
        mode = package.modes[0]()
        return [node.id for node in mode.build().nodes.values() if node.kind == "agent"]
    except Exception:
        # A package that will not build is the validate gate's finding, not this
        # one's; saying nothing here keeps the two failures distinguishable.
        return []


def write_result(ctx: Any) -> dict[str, Any]:
    """The run's own summary, written where a person can read it."""
    from refactory_dsh.authoring_evidence import require_completion

    require_completion(ctx)
    contracts = _mode_contracts(str(ctx.facts["name"]))
    nodes = [node for mode_nodes in contracts.values() for node in mode_nodes]
    metadata = _binding_metadata([node["task_ref"] for node in nodes if node.get("task_ref")], bool(nodes))
    payload = {
        "package": ctx.facts.get("name"),
        "extends": ctx.facts.get("extends"),
        "files": ctx.facts.get("files", []),
        "registered": ctx.facts.get("registered"),
        "smoke_score": ctx.facts.get("smoke_score"),
        # What kind of harness this run produced: the Task it is bound to, or that it
        # is the single-shot special case. A reader of the run's own result must not
        # have to infer that from the package.
        **metadata,
        "mode_contracts": contracts,
        "validation_evidence": ctx.read_json("authoring-evidence.json", None),
        "intent_audit": ctx.read_json("intent-audit.json", None),
    }
    if ctx.facts.get("revision_of"):
        payload.update({
            "revision_of": ctx.facts.get("revision_of"),
            "before_score": ctx.facts.get("before_score"),
            "after_score": ctx.facts.get("after_score"),
            "regressed": ctx.facts.get("regressed"),
            "revision_note": ctx.facts.get("revision_note"),
        })
    ctx.write_json("create.json", payload)
    return {"created": json.dumps(payload, default=str)}


# ── the v2 review phases ───────────────────────────────────────────────────
#
# create-v2 runs three reviews in parallel (a health check, a code review, and an
# adversarial QA director), merges them into one synthesis, and then audits the
# synthesis with an overwatch pass. Our fork/join does the parallel part; these
# ops do the merge and the two gates, so the three verdicts a reader cares about
# (did every review run, is anything blocking, did overwatch pass) are decided by
# the runtime rather than by an agent's summary of itself.

#: The reviews the fork produces, in the order the synthesis reads them.
QA_REVIEWS = (
    "reviews/health-check.md",
    "reviews/code-review.md",
    "reviews/qa.md",
)

#: How a review marks a finding it will not pass with.
BLOCKING_MARK = "BLOCKING"


# ── finding-scope filter ──────────────────────────────────────────────────
#
# A post-build gate (one that reloops to ``emit``) may block only on findings
# about files the builder can change.  A finding citing a run document owned
# by an earlier step is downgraded to a non-blocking note at the gate layer.
#
# Classification uses only the structured form ``<path>: <rest>`` at the
# start of the finding text (after ``blocking_findings`` has stripped the
# ``BLOCKING:`` prefix).  Free-form findings stay blocking.


def _run_doc_owners(workdir: Path) -> dict[str, str]:
    """Map run-document names to their owning node ids.

    Derived from the graph's node ``writes`` declarations in
    ``.refactory/declarations.json``.  A file no node writes is not a run
    document.  Only exact file names are indexed, not directory writes.
    """
    from refactory_dsh.lowering import read_declarations

    decls = read_declarations(workdir)
    owners: dict[str, str] = {}
    for node_id, entry in decls.items():
        if not isinstance(entry, dict):
            continue
        for written in entry.get("writes", ()):
            if isinstance(written, str) and not written.endswith("/"):
                if written not in owners:
                    owners[written] = node_id
    return owners


def classify_finding(
    finding_text: str, run_doc_owners: dict[str, str],
) -> tuple[bool, str | None]:
    """Classify a single finding by its structured target path.

    Only the structured form ``<path>: <rest>`` (where *path* matches a
    node's ``writes`` declaration exactly) names its target.  When that
    path is a run document, the finding becomes non-blocking.  Every other
    finding — free-form text, or a path not in the owners map — stays
    blocking.

    Returns ``(is_blocking, owning_step_or_None)``.
    """
    m = re.match(r'^([\w][\w/._-]*\.\w+):\s', finding_text)
    if not m:
        # Free-form finding, no structured path → stays blocking.
        return (True, None)
    target = m.group(1)
    if target in run_doc_owners:
        return (False, run_doc_owners[target])
    return (True, None)


def scope_filter_findings(
    findings: list[str], workdir: Path,
) -> tuple[list[str], list[str]]:
    """Separate blocking findings from non-blocking notes about run documents.

    Returns ``(blocking, notes)`` where *blocking* are findings that remain
    blocking and *notes* are human-readable strings for downgraded findings.
    """
    owners = _run_doc_owners(workdir)
    if not owners:
        return (list(findings), [])
    blocking: list[str] = []
    notes: list[str] = []
    for finding in findings:
        is_blocking, owning_step = classify_finding(finding, owners)
        if is_blocking:
            blocking.append(finding)
        else:
            target_m = re.match(r'^([\w][\w/._-]*\.\w+):\s', finding)
            target_name = target_m.group(1) if target_m else "?"
            notes.append(
                f"Non-blocking note ({owning_step} owns {target_name}): {finding}"
            )
    return (blocking, notes)


def marks(line: str, marker: str) -> bool:
    """Whether a line states the marker, however a markdown author decorated it.

    A reviewer that writes `**BLOCKING:** the test is deleted` has marked a blocking
    finding, and `line.startswith("BLOCKING")` does not see it. That is not
    hypothetical: a real run merged three reviews into a synthesis that said "0
    blocking finding(s)" and let the QA gate say `proceed` while the same body held
    four bolded blockers — and it was the overwatch pass, refusing to state PASS,
    that caught it.
    """
    cleaned = line.strip().lstrip("*-+>#\t ").lstrip("*_` ")
    return cleaned.upper().startswith(marker.upper())

#: How the overwatch pass states its own verdict.
PASS_MARK = "VERDICT: PASS"


def blocking_findings(text: str) -> list[str]:
    """Current marked findings, without empty verdicts or appended old reports."""
    findings: list[str] = []
    historical = False
    fence: tuple[str, int] | None = None
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        fence_line = re.match(r"^(`{3,}|~{3,})(.*)$", stripped)
        if fence is not None:
            if (fence_line and fence_line[1][0] == fence[0]
                    and len(fence_line[1]) >= fence[1] and not fence_line[2].strip()):
                fence = None
            continue
        if fence_line:
            fence = (fence_line[1][0], len(fence_line[1]))
            continue
        heading = re.match(r"^#{1,6}\s+(.*)$", stripped)
        if heading:
            title = heading[1].strip(" *_`")
            if re.match(r"^(?:current|latest) (?:review|findings|verdict|result|revision)\b", title, re.I):
                historical = False
            elif re.match(r"^(?:previous (?:review|run|attack)|superseded|historical (?:review|findings)|archived review)\b", title, re.I) or re.search(r"\(superseded\)", title, re.I):
                historical = True
        if historical:
            continue
        if stripped.startswith(">"):
            continue
        # `BLOCKING` with no separator after the span is a prose mention
        # (`BLOCKING` finding), not a marker; `BLOCKING`: the CLI crashes is one.
        # A span that quotes more than the marker is refused where the line's
        # start is read, below.
        without_mentions = re.sub(r"`BLOCKING`(?![*_]*\s*[:—–-])", "", stripped, flags=re.I)
        if not re.search(r"BLOCKING", without_mentions, re.I):
            continue
        cleaned = stripped.lstrip("*-+>#\t ")
        cleaned = re.sub(r"^\d+[.)]\s+", "", cleaned).lstrip("*_ ")
        # A line that opens with a code span marks a finding only when the span
        # holds the marker itself (`BLOCKING`, `BLOCKING:`). A span that quotes
        # more, as a review quoting an example does (`- **BLOCKING** x`), is a
        # quotation however its text begins.
        if cleaned.startswith("`"):
            if not re.match(r"`BLOCKING(?:[*_]*\s*[:—–-])?`", cleaned, re.I):
                continue
            cleaned = cleaned[1:]
        # A numbered finding is still a finding, including when its marker is
        # a heading: "### BLOCKING 1: ..." is not an empty section title.
        cleaned = re.sub(
            r"^(BLOCKING[*_`]*)\s+\d+\s*[:.)—–-]\s*", r"\1: ", cleaned, flags=re.I,
        )
        # Retain existing decorated/no-colon finding syntax. A bare BLOCKING
        # heading or a 'Blocking findings' section title carries no finding.
        marked = re.match(r"^BLOCKING(?:(?:[*_`]*\s*[:—–-][*_`]*\s*)|(?:[*_`]+\s+)|\s+)(.*)$", cleaned, re.I)
        if marked is None:
            continue
        # Bare-space/decoration-only match (no separator) requires uppercase BLOCKING.
        # "Blocking finding: `x`" and "**Blocking** finding: `x`" are prose,
        # not findings; "BLOCKING foo" and "**BLOCKING** foo" still are.
        gap = cleaned[8:marked.start(1)]
        if not re.search(r'[:—–\-]', gap) and cleaned[:8] != 'BLOCKING':
            continue
        content = marked[1].strip()
        if heading and not re.match(r"^BLOCKING[*_`]*\s*[:—–-]", cleaned, re.I):
            continue
        verdict_text = re.sub(r"[*_`]", "", content)
        # A numbered/code-formatted title such as "## 11. `BLOCKING:` findings"
        # names a section. Keep actual findings written as headings, however.
        if heading and re.fullmatch(r"(?:findings|issues|defects|items|blockers)[.:]?", verdict_text, re.I):
            continue
        # Reviewers also label their verdict: "BLOCKING defects: none.".
        # Strip only the category, then apply the same narrow negative verdict
        # grammar; "defects: none, except ..." must remain a finding.
        verdict_text = re.sub(
            r"^(?:findings|issues|defects|items|blockers)\s*:\s*", "", verdict_text, flags=re.I,
        )
        # 'none.' is a verdict; the following explanation does not turn it into
        # a defect. 'none, except ...' remains actionable instead of being hidden.
        verdict = re.split(r"[.!](?:\s|$)", verdict_text, maxsplit=1)[0].strip()
        if not content or re.fullmatch(
            r"(?:none|0|no(?: current)?(?: blocking)?(?: findings| issues| defects| items| blockers)?)"
            r"(?:\s*\(current revision\)|\s+(?:in|for)\s+(?:(?:the\s+)?"
            r"(?:(?:current|latest|revised|emitted|reviewed)\s+)+|this\s+)"
            r"(?:artifact|revision|version|package|implementation|code))?", verdict, re.I,
        ):
            continue
        continuation = []
        for following in lines[index + 1:]:
            if not following.strip() or following.lstrip().startswith(("#", "```", "~~~")) or marks(following, BLOCKING_MARK):
                break
            continuation.append(following.strip())
        findings.append("\n".join([content, *continuation]))
    return findings


# ── core and non-core findings ────────────────────────────────────────────
#
# A build is held back only by a defect in its core: an ask in the brief, an
# accepted criterion, a correction or constraint the person gave, or existing
# behaviour it breaks. Every other blocking finding is real work for later, and
# the run delivers with it listed as a follow-up instead of spending a round.

#: What every post-build reviewer is told about classifying its blocking findings.
SEVERITY_RULE = (
    "End the first line of each BLOCKING finding with `[core]` when it breaks an ask in "
    "the brief, an acceptance criterion, a correction or constraint the person gave, or "
    "existing behaviour (a regression: name the test or file:line that shows it), and "
    "with `[non-core]` otherwise: extra tests, docs, edge cases beyond the asks, "
    "refactors and style. A finding with no tag counts as core. Non-core findings do "
    "not hold the change back; its pull request lists them as follow-ups."
)

#: What every reviewer is told about where and how to write the blocking marker
#: (dm-198).  Held by test_every_reviewer_carries_marker_rule.
MARKER_RULE = (
    "Write the blocking marker only at the start of a finding line, in its "
    "plain form: BLOCKING: <path>: <description>. Never wrap it in backticks, "
    "a code span, or quotation marks while explaining a check or describing "
    "the parser. When discussing the marker itself, call it \"the blocking "
    "marker\". Write every real finding as a plain line outside any fenced "
    "code block, because fenced blocks are not read as findings. When you show "
    "the format as an example rather than report a finding, put the example "
    "inside a fenced code block."
)

#: What the overwatch pass is told about its verdict, beside SEVERITY_RULE.
OVERWATCH_VERDICT_RULE = (
    "State `VERDICT: FAIL` only for a core defect or an ask with no evidence; when every "
    "finding left is non-core, state `VERDICT: PASS`."
)

_CORE_TAG = re.compile(r"\[\s*core\b[^\]]*\]", re.I)
_NON_CORE_END = re.compile(r"\[\s*non[\s_-]?core\s*\][\s*_`.]*$", re.I)


def is_non_core(finding: str) -> bool:
    """Whether a finding's first line ends with the non-core tag, and nothing in it is
    tagged core. A tag anywhere else, an untagged finding and a doubly tagged one are
    core, so a missing or misplaced tag never lets a core defect through."""
    lines = finding.strip().splitlines()
    return (bool(lines) and bool(_NON_CORE_END.search(lines[0].strip()))
            and not _CORE_TAG.search(finding))


def split_core(findings: list[str]) -> tuple[list[str], list[str]]:
    """``(core, non_core)``, each in the order given."""
    return ([f for f in findings if not is_non_core(f)],
            [f for f in findings if is_non_core(f)])


def non_core_ride_along(non_core: list[str]) -> str:
    """The part of a repair's feedback that carries the round's non-core findings."""
    if not non_core:
        return ""
    return ("Non-core findings (fix them if it is cheap; otherwise leave them, and the "
            "pull request lists them as follow-ups):\n"
            + "\n".join(f"- {finding}" for finding in non_core))


#: The run's non-core findings at the point each gate let the build through.
FOLLOW_UPS = "reviews/follow-ups.json"


def record_follow_ups(ctx: Any, gate: str, findings: list[str]) -> None:
    """Replace ``gate``'s follow-ups with the findings it proceeded with.

    Called only when the gate proceeds, so a finding a later round fixed is not
    listed, and ``reviews/follow-ups.md`` always says what the delivered change leaves.
    """
    recorded = ctx.read_json(FOLLOW_UPS, {}) if ctx.exists(FOLLOW_UPS) else {}
    if not isinstance(recorded, dict):
        recorded = {}
    recorded[gate] = list(findings)
    ctx.write_json(FOLLOW_UPS, recorded)
    lines = [f"- {item}" for item in follow_ups(ctx)]
    ctx.write_text("reviews/follow-ups.md",
                   "# Follow-ups (non-core findings)\n\n"
                   + ("\n".join(lines) if lines else "None.") + "\n")


def follow_ups(ctx: Any) -> list[str]:
    """Every recorded follow-up, each with the gate that let it through."""
    recorded = ctx.read_json(FOLLOW_UPS, {}) if ctx.exists(FOLLOW_UPS) else {}
    if not isinstance(recorded, dict):
        return []
    return [f"{finding} ({source})" for source, items in recorded.items()
            if isinstance(items, list) for finding in items]


def _review_snapshot(ctx: Any) -> dict[str, Any]:
    reviews: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in QA_REVIEWS:
        text = ctx.read_text(name) if ctx.exists(name) else ""
        if not text.strip():
            missing.append(name)
            continue
        reviews.append({"path": name, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "findings": blocking_findings(text)})
    return {"version": 1, "reviews": reviews, "missing": missing}


def synthesize_qa(ctx: Any) -> dict[str, Any]:
    """Preserve the full reports and publish the current findings used by the gate."""
    snapshot = _review_snapshot(ctx)
    blocking = sum(len(review["findings"]) for review in snapshot["reviews"])
    missing = snapshot["missing"]
    sections = [f"## {review['path']}\n\n{ctx.read_text(review['path']).strip()}"
                for review in snapshot["reviews"]]
    body = (
        "# QA synthesis\n\n"
        f"{len(sections)} review(s) merged, {blocking} blocking finding(s), "
        f"{len(missing)} review(s) missing.\n"
        + ("\n## WARNING: NO REVIEW FROM " + ", ".join(missing) + "\n" if missing else "")
        + "\n" + "\n\n".join(sections) + "\n"
    )
    # ── verifier status section ──
    if ctx.exists("reviews/verification.md"):
        v_text = ctx.read_text("reviews/verification.md").strip()
        if v_text:
            v_findings = list(blocking_findings(v_text))
            body += (f"\n## Verification\n\nVerification: present "
                     f"({len(v_findings)} finding(s)).\n\n{v_text}\n")
        else:
            body += "\n## Verification\n\nVerification: missing (branch failed or empty).\n"
    else:
        body += "\n## Verification\n\nVerification: not run.\n"
    manifest = {"version": 1, "digests": {review["path"]: review["sha256"] for review in snapshot["reviews"]},
                "missing": missing}
    body += "\n<!-- refactory-qa-snapshot " + json.dumps(manifest, sort_keys=True) + " -->\n"
    ctx.write_text("reviews/qa-synthesized.md", body)
    return {"qa": body, "qa_blocking": blocking, "qa_missing": len(missing)}


def _legacy_reviews_match(ctx: Any, text: str, snapshot: dict[str, Any]) -> bool:
    """Compare complete legacy report sections, never a matching substring."""
    headings = list(re.finditer(r"^## (reviews/(?:health-check|code-review|qa)\.md)\s*$", text, re.M))
    names = [heading[1] for heading in headings]
    expected = [review["path"] for review in snapshot["reviews"]]
    if names != expected or len(names) != len(set(names)):
        return False
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        if text[heading.end():end].strip() != ctx.read_text(heading[1]).strip():
            return False
    return True


def gate_qa(ctx: Any) -> str:
    """Route from current findings, checking the synthesis's review snapshots."""
    text = ctx.read_text("reviews/qa-synthesized.md")
    snapshot = _review_snapshot(ctx)
    manifest = {"version": 1, "digests": {review["path"]: review["sha256"] for review in snapshot["reviews"]},
                "missing": snapshot["missing"]}
    recorded = re.search(r"<!-- refactory-qa-snapshot (.+) -->\s*$", text)
    if recorded is not None:
        try:
            fresh = json.loads(recorded[1]) == manifest
        except ValueError:
            fresh = False
    else:
        # Historical synthesis already embeds each report verbatim. Accept that
        # representation only while every present report still matches its text.
        fresh = _legacy_reviews_match(ctx, text, snapshot)
    from refactory_dsh import authoring_evidence as evidence

    for review in snapshot["reviews"]:
        detail = ctx.read_text(review["path"])
        unavailable = re.search(r"(?im)^\s*UNAVAILABLE:.*$", detail)
        if unavailable:
            evidence.stop_unrepairable("unavailable", unavailable[0])
    scope_notes: list[str] = []
    non_core: list[str] = []
    stale = evidence.current_problems(ctx, ("health_checker", "code_reviewer", "qa_director"))
    if stale:
        reason = "reloop: " + "; ".join(stale)
    elif not fresh:
        reason = "reloop: review reports changed after synthesis; run fresh reviews and synthesis"
    elif snapshot["missing"]:
        reason = "reloop: a review did not run: " + ", ".join(snapshot["missing"])
    else:
        # Collect raw findings per review, then apply scope filter.
        raw_findings = [finding for review in snapshot["reviews"]
                        for finding in review["findings"]]
        filtered, scope_notes = scope_filter_findings(raw_findings, ctx.workdir)
        # Rebuild with review-path prefixes for the still-blocking findings.
        blocking_indexed: list[str] = []
        idx = 0
        for review in snapshot["reviews"]:
            for finding in review["findings"]:
                if finding in filtered:
                    blocking_indexed.append(f"{review['path']}: {finding}")
                    filtered.remove(finding)  # consume first match
                idx += 1
        # Only a core defect sends the build back; non-core findings ride along with it.
        core, non_core = split_core(blocking_indexed)
        reason = "reloop: " + "\n".join(core) if core else ""
        if reason and non_core:
            reason += "\n" + non_core_ride_along(non_core)
        if scope_notes:
            note_text = "\n".join(scope_notes)
            if reason:
                reason += "\n" + note_text
    # ── per-changed-function entry validation (design mode only) ──
    deliverable_kind_value = str(ctx.facts.get("deliverable_kind", ""))
    advisory = ""
    if not reason and deliverable_kind_value == "repository_change":
        from refactory_dsh.authoring_evidence import (
            advisory_repo_questions, check_review_questions)
        from refactory_dsh.changed_functions import review_functions
        import json as _json

        changed_fns: list[str] = []
        if ctx.exists("changed-functions.json"):
            try:
                changed_fns = review_functions(_json.loads(ctx.read_text("changed-functions.json")))
            except ValueError:
                pass

        if changed_fns:
            rq_problems: list[str] = []
            _review_file_to_node = {
                "reviews/health-check.md": "health_checker",
                "reviews/code-review.md": "code_reviewer",
                "reviews/qa.md": "qa_director",
            }
            for review in snapshot["reviews"]:
                reviewer_name = _review_file_to_node.get(
                    review["path"],
                    review["path"].rsplit("/", 1)[-1].replace(".md", "").replace("-", "_"),
                )
                review_text_content = ctx.read_text(review["path"])
                rq_problems.extend(
                    check_review_questions(review_text_content, changed_fns, reviewer_name))
            if rq_problems:
                advisory = advisory_repo_questions(rq_problems)

    # ── verifier advisory handling ──
    from refactory_dsh.verification import parse_verification_findings
    verifier_raw = parse_verification_findings(ctx) if ctx.exists("reviews/verification.md") else []
    verifier_filtered = [f for f in verifier_raw
                         if f != "reviews/verification.md is empty or missing"]
    verifier_text = "\n".join(verifier_filtered) if verifier_filtered else ""
    verifier_rebuttals: list[str] = []

    # Case 1: QA blocking + verifier findings → combine
    if reason and verifier_text:
        reason += "\n[verifier] " + verifier_text + "\n\n" + VERIFIER_RESPONSE

    if reason:
        ctx.write_text("reviews/qa-gate.txt", reason + "\n")
        ctx.set_fact("qa_gate", reason)
        _write_finding_sources_full(ctx, snapshot, verifier_filtered)
        return repair(ctx, [reason], kind='review') if ctx.exists(evidence.EVIDENCE) else Verdict("reloop", reason)

    # Case 2: Only verifier findings, QA passed. The verifier never gets a builder
    # round of its own: Overwatch reads what it found and sends the build back only
    # for a finding that breaks an ask, and the pull request lists the rest.
    verifier_rebuttals = _parse_verifier_rebuttals_full(ctx) if verifier_filtered else []
    record_verification_unresolved(ctx, verifier_filtered, verifier_rebuttals)

    # Case 3: No core defect — proceed, with any non-core findings as follow-ups
    record_follow_ups(ctx, "qa", non_core)
    found = (f"no core defect; {len(non_core)} non-core finding(s) are follow-ups"
             if non_core else "nothing blocking")
    # Before the advisory: delivery reads the repository-question advisory to the end.
    notes = "".join(f"{note}\n" for note in (
        *scope_notes, *(f"follow-up: {finding}" for finding in non_core), advisory) if note)
    ctx.write_text("reviews/qa-gate.txt",
                   ("proceed: no core finding\n" if non_core else "proceed: no blocking finding\n")
                   + notes)
    ctx.set_fact("qa_gate", f"(the reviews found {found})" + (f"\n{advisory}" if advisory else ""))
    ctx.write_text("validation.txt", "(reviews passed)\n")
    ctx.set_fact("validation", "(reviews passed)")
    _write_finding_sources_full(ctx, snapshot, verifier_filtered,
                                rebuttals=verifier_rebuttals, proceeding=True)
    return "proceed"


def _finding_description(finding: str) -> str:
    """Strip the optional file:line: prefix from a verifier finding for matching."""
    m = re.match(r'\S+:\d+:\s*', finding)
    return finding[m.end():] if m else finding


def _parse_verifier_rebuttals_full(ctx: Any) -> list[str]:
    """The builder's rebuttals of the verifier's findings (`verifier_rebuttals`)."""
    return verifier_rebuttals(ctx)


_FULL_REVIEW_FILE_TO_NODE = {
    "reviews/health-check.md": "health_checker",
    "reviews/code-review.md": "code_reviewer",
    "reviews/qa.md": "qa_director",
}


def _write_finding_sources_full(ctx: Any, snapshot: dict[str, Any],
                                verifier_findings: list[str],
                                rebuttals: list[str] | None = None,
                                proceeding: bool = False) -> None:
    """Write reviews/finding-sources.json on every gate_qa visit (full mode)."""
    existing: list[dict[str, Any]] = []
    if ctx.exists("reviews/finding-sources.json"):
        try:
            existing = json.loads(ctx.read_text("reviews/finding-sources.json"))
        except (json.JSONDecodeError, ValueError):
            existing = []

    round_num = max((e.get("last_round", 0) for e in existing), default=0) + 1
    rebuttals = rebuttals or []

    current: dict[tuple[str, str], str] = {}
    for review in snapshot["reviews"]:
        source = _FULL_REVIEW_FILE_TO_NODE.get(
            review["path"],
            review["path"].rsplit("/", 1)[-1].replace(".md", "").replace("-", "_"),
        )
        for finding in review["findings"]:
            current[(source, finding)] = review["path"]

    for finding in verifier_findings:
        current[("verifier", finding)] = "reviews/verification.md"

    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for entry in existing:
        key = (entry["source"], entry["finding"])
        seen.add(key)
        entry = dict(entry)
        if key in current:
            entry["last_round"] = round_num
            if entry["source"] == "verifier" and any(_finding_description(entry["finding"]).lower() in r.lower() for r in rebuttals):
                entry["fate"] = "rebutted"
            else:
                entry["fate"] = "unresolved"
        else:
            entry["fate"] = "fixed"
        entries.append(entry)

    for key, review_path in current.items():
        if key not in seen:
            source, finding = key
            fate = "unresolved"
            if source == "verifier" and any(_finding_description(finding).lower() in r.lower() for r in rebuttals):
                fate = "rebutted"
            entries.append({
                "source": source, "review_path": review_path,
                "finding": finding, "first_round": round_num,
                "last_round": round_num, "fate": fate,
            })

    ctx.write_text("reviews/finding-sources.json",
                   json.dumps(classify_finding_sources(entries, round_num, proceeding),
                              indent=2) + "\n")


def classify_finding_sources(entries: list[dict[str, Any]], round_num: int,
                             proceeding: bool) -> list[dict[str, Any]]:
    """Give each finding record its class, and its fate when the gate let the build through.

    A non-core finding still open in the round the gate proceeded on is a follow-up.
    """
    for entry in entries:
        entry["class"] = "non-core" if is_non_core(entry["finding"]) else "core"
        if (proceeding and entry["class"] == "non-core" and entry["last_round"] == round_num
                and entry["fate"] == "unresolved"):
            entry["fate"] = "follow-up"
    return entries


#: Where a builder answers the verifier's findings: its own output file, since a builder
#: that works in a workspace no longer writes emit.txt (`workspace.collect` derives it).
VERIFIER_RESPONSE_FILE = "verifier-response.md"

#: What a builder sent back for a core defect is told about the verifier's findings
#: that ride along with it, when it writes its own emission (create, edit, fast-create):
#: its answers go in emit.txt.
VERIFIER_RESPONSE = ("For each verifier finding, respond in emit.txt under "
                     "## Verifier findings with the finding, then: fixed, or "
                     "rebutted: <reason> <file:line>")

#: The same, for a builder that works in a workspace (design_fast's), whose emit.txt is
#: derived from its diff: its answers are its final message, saved as their own file.
VERIFIER_RESPONSE_WORKSPACE = ("For each verifier finding, answer in your final message, which is saved as "
                               f"{VERIFIER_RESPONSE_FILE}: the finding, then fixed, or "
                               "rebutted: <reason> <file:line>")


def verifier_rebuttals(ctx: Any) -> list[str]:
    """The builder's answers to the verifier: its ``fixed`` and ``rebutted:`` lines.

    Read from `VERIFIER_RESPONSE_FILE`, the builder's final message: the lines of its
    ``## Verifier findings`` section when it has one, else every line. From that section
    of emit.txt for a builder that still writes its emission itself.
    """
    if ctx.exists(VERIFIER_RESPONSE_FILE):
        lines = ctx.read_text(VERIFIER_RESPONSE_FILE).splitlines()
        sectioned = any(line.strip().startswith("## Verifier findings") for line in lines)
    elif ctx.exists("emit.txt"):
        lines = ctx.read_text("emit.txt").splitlines()
        sectioned = True
    else:
        return []
    in_section = not sectioned
    rebuttals: list[str] = []
    for line in lines:
        stripped = line.strip()
        if sectioned and stripped.startswith("## Verifier findings"):
            in_section = True
            continue
        if in_section:
            if sectioned and (stripped.startswith("## ") or stripped.startswith("===FILE:")):
                break
            if "rebutted:" in stripped.lower() or "fixed" in stripped.lower():
                rebuttals.append(stripped)
    return rebuttals


def record_verification_unresolved(ctx: Any, findings: list[str], rebuttals: list[str]) -> None:
    """Write this round's verifier findings, each with the builder's rebuttal if any.

    Overwatch reads the file and the pull request lists it. A round with no finding
    empties it, so an earlier round's findings are never reported as current.
    """
    lines = []
    for finding in findings:
        desc = _finding_description(finding)
        reb = next((r for r in rebuttals if desc.lower() in r.lower()), None)
        lines.append(finding + (f" — {reb}" if reb else ""))
    if lines or ctx.exists("reviews/verification-unresolved.md"):
        ctx.write_text("reviews/verification-unresolved.md",
                       "\n".join(lines) + ("\n" if lines else ""))


def gate_overwatch(ctx: Any) -> str:
    """Proceed when the overwatch pass stated PASS, raised no core finding, and the
    evidence, intent audit and correction audit all hold."""
    text = ctx.read_text("reviews/overwatch.md")
    from refactory_dsh import authoring_evidence as evidence

    unavailable = re.search(r"(?im)^\s*UNAVAILABLE:.*$", text)
    if unavailable:
        evidence.stop_unrepairable("unavailable", unavailable[0])

    # Compute four separate problem categories so the reason names each one.
    evidence_problems = evidence.current_problems(ctx, ("overwatch",))
    audit_problems_list = evidence.audit_problems(ctx)
    correction_problems = evidence.correction_audit_problems(ctx)
    review_findings_raw = blocking_findings(text)
    review_findings, ow_scope_notes = scope_filter_findings(review_findings_raw, ctx.workdir)
    # Non-core findings never block; a FAIL verdict always does, because the overwatch
    # is told to state PASS when only non-core findings remain.
    review_findings, ow_non_core = split_core(review_findings)
    verdict = text.strip().splitlines()[-1].strip() if text.strip() else ""
    passed = verdict == PASS_MARK

    if (passed and not evidence_problems and not audit_problems_list
            and not review_findings and not correction_problems):
        record_follow_ups(ctx, "overwatch", ow_non_core)
        not_measured = evidence.correction_audit_advisories(ctx)
        ctx.write_text("reviews/overwatch-gate.txt", "proceed: overwatch passed\n"
                       + "".join(f"follow-up: {finding}\n" for finding in ow_non_core)
                       + (evidence.advisory_correction_audit(not_measured) + "\n"
                          if not_measured else ""))
        ctx.set_fact("overwatch_review", "(overwatch passed)")
        ctx.write_text("validation.txt", "(overwatch passed)\n")
        ctx.set_fact("validation", "(overwatch passed)")
        return "proceed"

    # ── Correction-audit path: route to repair, not back to overwatch ────
    if (passed and not review_findings
            and not evidence_problems and not audit_problems_list
            and correction_problems):
        reason = ("correction audit: " + "; ".join(correction_problems)
                  + "\n" + _excerpt(text, 2000))
        ctx.write_text("reviews/overwatch-gate.txt", reason + "\n")
        ctx.set_fact("overwatch_review", reason)
        ctx.write_text("validation.txt", reason + "\n")
        ctx.set_fact("validation", reason)
        return repair(ctx, [reason], kind='review')

    # ── Audit-only path: code is not at fault ────────────────────────────
    # When VERDICT: PASS, no blocking findings and no evidence problems,
    # but audit_problems exist, route back to overwatch without spending
    # an attempt.  gate_overwatch's max_iterations bounds this.
    if (passed and not review_findings
            and not evidence_problems and audit_problems_list):
        reason = ("intent audit problems: "
                  + "; ".join(audit_problems_list)
                  + "\n" + _excerpt(text, 2000))
        ctx.write_text("reviews/overwatch-gate.txt", reason + "\n")
        ctx.set_fact("overwatch_review", reason)
        ctx.write_text("validation.txt", reason + "\n")
        ctx.set_fact("validation", reason)
        return Verdict("audit", reason)

    # ── Reloop path: at least one non-audit problem ──────────────────────
    parts: list[str] = []
    if evidence_problems:
        parts.append("evidence problems: " + "; ".join(evidence_problems))
    if audit_problems_list:
        parts.append("intent audit problems: " + "; ".join(audit_problems_list))
    if correction_problems:
        parts.append("correction audit: " + "; ".join(correction_problems))
    if review_findings:
        parts.append("blocking review findings: " + "; ".join(review_findings))
    if not passed:
        parts.append(
            f"overwatch verdict is missing or not {PASS_MARK!r}"
            f" (last line: {verdict!r})"
        )
    if ow_scope_notes:
        parts.append("non-blocking notes: " + "; ".join(ow_scope_notes))
    reason = ("reloop: " + "; ".join(parts) + "\n"
              + (non_core_ride_along(ow_non_core) + "\n" if ow_non_core else "")
              + _excerpt(text, 2000))
    ctx.write_text("reviews/overwatch-gate.txt", reason + "\n")
    ctx.set_fact("overwatch_review", reason)
    ctx.write_text("validation.txt", reason + "\n")
    ctx.set_fact("validation", reason)
    return repair(ctx, [reason], kind='review') if ctx.exists(evidence.EVIDENCE) else Verdict("reloop", reason)
