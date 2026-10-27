"""Fixtures shared by the tests that need a generated package.

Kept in one module so `test_create.py` and `test_example.py` exercise the same
package rather than two copies that drift apart. The DSH profile helpers
``_profile`` and ``_link_bundle`` are shared by test_dsh_profile.py,
test_cli.py and test_interactive.py.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from refactory_dsh import dsh_profile


def _profile(home: Path, name: str, bundles: list[str]) -> Path:
    """Create a minimal DSH profile directory with a manifest."""
    directory = home / "profiles" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "package.json").write_text(
        json.dumps({
            "name": f"dsh-profile-{name}",
            "dependencies": {
                bundle: "link:/checkout"
                for bundle in bundles
                if bundle in {entry.bundle for entry in dsh_profile.LAYERS.values()}
            },
            "dsh": {"profile": {"bundles": bundles}},
        })
    )
    return directory


def _link_bundle(home: Path, name: str, bundle: str, target: Path) -> None:
    """Create the node_modules symlink that ``dsh plugin add`` creates.

    The target is the directory the symlink resolves to — either this
    checkout's ``packages/bundle/graph`` or a different directory that
    simulates another checkout.
    """
    link = home / "profiles" / name / "node_modules" / bundle
    link.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)

#: A placeholder stored as the configured key in a fake model home. Never a real
#: credential, and it must never reach any output.
PLACEHOLDER_KEY = "placeholder-not-a-real-key"

#: The run model a sealed test's DSH home holds (dm-258), as a person who chose one in
#: Settings → Models has it. A fake: no test reaches a provider with it.
TEST_MODEL_CHOICE = {"provider": "fakeprov", "model": "fake-model"}


def store_model_choice(home: Path, provider: str = TEST_MODEL_CHOICE["provider"],
                       model: str = TEST_MODEL_CHOICE["model"]) -> Path:
    """Write the one model choice where Settings → Models keeps it, as that page writes it.

    Plain text rather than `dsh_profile.model_choice_write`, so the tests of that writer
    do not depend on it for their own setup.
    """
    path = home / "profiles" / dsh_profile.GUI_PROFILE / "cordis.patch.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "- id: agent-default-model\n"
        "  name: \"@deepseek-ai/dsh-agent-default-model\"\n"
        "  config:\n"
        f"    provider: {provider}\n"
        f"    model: {model}\n"
    )
    return path


def model_home_with_a_key(home: Path, *, provider: str = "fakeprov", model: str = "fake-model",
                          base_url: str = "http://127.0.0.1:9") -> Path:
    """A DSH home whose headless profile routes two providers: one with a stored key, one without.

    The profile is the shape install writes for `headless` (DSH's own bundles, no
    refactory layer), with the routes in its patch, which is where the pinned harness
    keeps what Settings -> Models saves. The store is the managed credential document.
    ``provider``/``model`` (by default `fakeprov/fake-model`) is the route the person
    chose (the stored choice, dm-258) and the headless profile's own default, at
    ``base_url``, and has its key; `otherprov` has none.
    """
    store_model_choice(home, provider, model)
    profile = home / "profiles" / "headless"
    profile.mkdir(parents=True, exist_ok=True)
    (profile / "package.json").write_text(json.dumps({
        "name": "dsh-profile-headless", "private": True, "dependencies": {},
        "dsh": {"profile": {"bundles": ["@deepseek-ai/dsh-base", "@deepseek-ai/dsh-headless"]}},
    }))
    (profile / "cordis.yml").write_text("[]\n")
    (profile / "cordis.patch.yml").write_text(
        "- id: llm-pi-ai\n"
        "  name: '@deepseek-ai/dsh-llm-pi-ai'\n"
        "  config:\n"
        "    providers:\n"
        f"      {provider}:\n"
        f"        baseURL: {base_url}\n"
        "        apiKeyEnv: REFACTORY_TEST_KEY_SET\n"
        "      otherprov:\n"
        "        baseURL: http://127.0.0.1:9\n"
        "        apiKeyEnv: REFACTORY_TEST_KEY_UNSET\n"
        "- id: agent-default-model\n"
        "  name: '@deepseek-ai/dsh-agent-default-model'\n"
        "  config:\n"
        f"    provider: {provider}\n"
        f"    model: {model}\n"
    )
    store = home / ".credentials.yaml"
    store.write_text(f"version: 1\nrefs:\n  REFACTORY_TEST_KEY_SET: {PLACEHOLDER_KEY}\n")
    store.chmod(0o600)
    return home


def graph_profile(home: Path, checkout: Path) -> Path:
    """The `graph` profile install.sh leaves: DSH's base bundle and ``checkout``'s graph bundle."""
    directory = _profile(home, dsh_profile.GRAPH_PROFILE, ["@deepseek-ai/dsh-base", dsh_profile.GRAPH_BUNDLE])
    _link_bundle(home, dsh_profile.GRAPH_PROFILE, dsh_profile.GRAPH_BUNDLE, checkout / "packages" / "bundle" / "graph")
    return directory


_INIT = '''\
"""A generated package."""
from refactory_dsh.nn import Package

from .mode import Counter

PACKAGE = Package(
    name="{name}",
    description="count to a target",
    modes=(Counter,),
    ops=("refactory_packages.{name}.ops",),
)
'''

_OP_HEAD = '''\
"""Operations for the counter mode."""
'''

_OPS = '''

def start(ctx):
    target = int(ctx.knob("target", 3))
    ctx.write_json("state.json", {"value": 0, "target": target})
    return {"value": 0}


def step(ctx):
    state = ctx.read_json("state.json", {"value": 0, "target": 3})
    state["value"] += 1
    ctx.write_json("state.json", state)
    return {"value": state["value"]}


def check(ctx):
    state = ctx.read_json("state.json", {"value": 0, "target": 3})
    done = state["value"] >= state["target"]
    ctx.set_fact("done", done)
    return "proceed" if done else "retry"


def finish(ctx):
    state = ctx.read_json("state.json", {"value": 0, "target": 3})
    ctx.write_json("best.json", {"best": state["value"]})
    return {"best": state["value"]}
'''

_MODE_HEAD = '''\
"""The counter mode."""
from refactory_dsh import nn

OPS = "refactory_packages.{name}.ops"


class Counter(nn.Mode):
'''

_MODE_TAIL = '''\
    name = "counter"
    description = "count to a target"

    @classmethod
    def declare(cls):
        return {"target": nn.Knob(3, "int", lo=1, hi=9, description="where to stop")}

    def build(self):
        graph = nn.Graph()
        graph.add(nn.Node("start", "op", call=OPS + ":start",
                          writes=("state.json",), produces=("value",), knobs=("target",)))
        graph.add(nn.Node("step", "op", call=OPS + ":step",
                          reads=("state.json",), writes=("state.json",), produces=("value",)))
        graph.add(nn.Node("check", "gate", evaluator=OPS + ":check",
                          reads=("state.json",), produces=("done",), max_iterations=20))
        graph.add(nn.Node("finish", "op", call=OPS + ":finish",
                          reads=("state.json",), writes=("best.json",), produces=("best",)))
        graph.connect("start", "step")
        graph.connect("step", "check")
        graph.connect("check", "step", condition="retry")
        graph.connect("check", "finish", condition="proceed")
        return graph

    def score(self, run):
        return float(run.facts["best"])

    def invariants(self):
        return (nn.fact_produced("best"),)
'''


def good_files(name: str) -> dict[str, str]:
    """A working generated package under the given name."""
    return {
        "__init__.py": _INIT.format(name=name),
        "ops.py": _OP_HEAD + _OPS,
        "mode.py": _MODE_HEAD.format(name=name) + _MODE_TAIL,
    }


_DATA_INIT = '''\
"""A generated package whose one mode iterates a Task's items."""
from refactory_dsh.nn import Package

from .mode import Chore

PACKAGE = Package(
    name="{name}",
    description="process every item the Task lists",
    modes=(Chore,),
    ops=("refactory_packages.{name}.ops",),
)
'''

_DATA_OPS = '''\
"""Operations for the per-item harness."""


def record(ctx):
    item = ctx.read_json("current_item.json", {}) or {}
    seen = ctx.read_json("seen.json", {"ids": []}) or {"ids": []}
    seen.setdefault("ids", []).append(str(item.get("id")))
    seen["min_items"] = int(ctx.knob("min_items", 1) or 1)
    ctx.write_json("seen.json", seen)
    return {"value": len(seen["ids"])}


def done(ctx):
    ctx.write_json("done.json", {"finished": True})
    return {"finished": True}
'''

_DATA_MODE = '''\
"""The per-item harness mode."""
from refactory_dsh import nn

OPS = "refactory_packages.{name}.ops"
TASK = "{task_ref}"


class Chore(nn.Mode):
    name = "chore"
    description = "iterate a Task's items and score the passes it verified"

    @classmethod
    def declare(cls):
        return {{"min_items": nn.Knob(1, "int", lo=1, hi=10)}}

    def build(self):
        graph = nn.Graph()
        graph.add(nn.Node("data", "data", subgraph_entry="work", subgraph_exit="work",
                          task_ref=TASK, writes=("current_item.json",),
                          produces=("item_scores",)))
        graph.add(nn.Node("work", "op", call=OPS + ":record", knobs=("min_items",),
                          reads=("current_item.json",), writes=("seen.json",),
                          produces=("value",)))
        graph.add(nn.Node("finish", "op", call=OPS + ":done", writes=("done.json",),
                          produces=("finished",)))
        graph.connect("data", "finish")
        return graph

    def score(self, run):
        scores = [entry["score"] for entry in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0
'''


def good_data_files(name: str, task_ref: str) -> dict[str, str]:
    """A working generated package whose one mode is a data node bound to `task_ref`.

    The Task is the one `dataset_probe` ships, whose `verify` reads what the pass
    wrote: the package is a harness for that class of work, not for one sample.
    """
    return {
        "__init__.py": _DATA_INIT.format(name=name),
        "ops.py": _DATA_OPS,
        "mode.py": _DATA_MODE.format(name=name, task_ref=task_ref),
    }


def baked_data_files(name: str, task_ref: str) -> dict[str, str]:
    """The same package with the items baked into the graph instead of fed at run time.

    `inline_items` is read before the Task, so the harness stops taking the items the
    class of work supplies and iterates only the ones the authoring run seeded.
    """
    files = good_data_files(name, task_ref)
    files["mode.py"] = files["mode.py"].replace(
        'task_ref=TASK, writes=("current_item.json",),',
        'task_ref=TASK, inline_items=[{"id": "alpha"}, {"id": "beta"}],\n'
        '                          writes=("current_item.json",),',
    )
    return files


def unfed_data_files(name: str) -> dict[str, str]:
    """A data node that names no Task and no source: the input port is shut."""
    files = good_data_files(name, "unused:Task")
    files["mode.py"] = files["mode.py"].replace("task_ref=TASK, ", "")
    return files


def strategy_answer(name: str) -> str:
    """A coherent strategy answer, in the shape create-mode's review gate checks."""
    import json

    ops = f"refactory_packages.{name}.ops"
    return json.dumps({
        "considered": [
            {"approach": "count up one at a time", "why_not": ""},
            {"approach": "compute the answer directly", "why_not": "nothing to iterate"},
        ],
        "chosen": "count up one at a time",
        "why": "it is the simplest thing that has a state to watch",
        "will_not": ["threads", "a second package"],
        "closest_existing": "circle_packing/adaptive",
        "knobs": [
            {"name": "target", "kind": "int", "default": 3, "lo": 1, "hi": 9,
             "description": "where to stop"},
        ],
        "nodes": [
            {"id": "start", "kind": "op", "call": f"{ops}:start",
             "writes": ["state.json"], "produces": ["value"], "knobs": ["target"]},
            {"id": "step", "kind": "op", "call": f"{ops}:step",
             "reads": ["state.json"], "writes": ["state.json"], "produces": ["value"]},
            {"id": "check", "kind": "gate", "evaluator": f"{ops}:check",
             "reads": ["state.json"], "produces": ["done"], "max_iterations": 20},
            {"id": "finish", "kind": "op", "call": f"{ops}:finish",
             "reads": ["state.json"], "writes": ["best.json"], "produces": ["best"]},
        ],
        "edges": [
            {"source": "start", "target": "step", "condition": None},
            {"source": "step", "target": "check", "condition": None},
            {"source": "check", "target": "step", "condition": "retry"},
            {"source": "check", "target": "finish", "condition": "proceed"},
        ],
        "start": "start",
        "score": "the fact 'best', produced by finish",
        "findings": {
            "FND-api-surface-1": "start node handles the API surface",
            "FND-existing-modes-1": "step node handles existing modes",
            "FND-acceptance-1": "check node handles acceptance criteria",
            "FND-simplest-1": "step node handles simplest perspective",
            "FND-criteria-1": "check node handles criteria perspective",
        },
    })


#: The answers create-v2's non-emit agents give, in the order the graph asks.
#:
#: The research answer is the plan array the gate checks, not a sentence about having
#: researched: a prose answer is exactly the failure the gate exists to catch, so a
#: fixture that fed one would be testing the bug.
#: A valid pitfalls answer for the pitfalls_researcher agent node.
PITFALLS_ANSWER = json.dumps([
    {
        "id": "PIT-001",
        "risk": "high",
        "category": "pinning_tests",
        "evidence": "tests/test_example.py:10",
        "prescription": "keep the assertion or deliberately change it with a reason",
    },
])

RESEARCH_ANSWER = json.dumps([
    {
        "focus": "which API surface the package builds against",
        "slug": "api-surface",
        "finding": "the mode API in the checkout: nn.Mode, nn.Graph, ctx.knob/read_json/write_json",
        "ids": ["FND-api-surface-1"],
    },
    {
        "focus": "what already exists that this resembles",
        "slug": "existing-modes",
        "finding": "circle_packing/adaptive is the closest existing mode and is not this",
        "ids": ["FND-existing-modes-1"],
    },
    {
        "focus": "the acceptance criteria the brief asks for",
        "slug": "acceptance",
        "finding": "the mode must count to a target and score how close it landed",
        "ids": ["FND-acceptance-1"],
    },
])

#: A research answer with footprints that satisfy the scoped rule.
#: Every direction carries a footprint.  Directions that touch no code record
#: empty footprints.  The aggregate: 1 existing function, no new items, no
#: shared engine → scoped.
SCOPED_RESEARCH_ANSWER = json.dumps([
    {
        "focus": "which API surface the package builds against",
        "slug": "api-surface",
        "finding": "the mode API in the checkout: nn.Mode, nn.Graph, ctx.knob/read_json/write_json",
        "ids": ["FND-api-surface-1"],
        "footprint": {
            "existing_functions": ["app.py:1 app.main"],
            "new_items": [],
            "shared_engine": [],
        },
    },
    {
        "focus": "what already exists that this resembles",
        "slug": "existing-modes",
        "finding": "circle_packing/adaptive is the closest existing mode and is not this",
        "ids": ["FND-existing-modes-1"],
        "footprint": {"existing_functions": [], "new_items": [], "shared_engine": []},
    },
    {
        "focus": "the acceptance criteria the brief asks for",
        "slug": "acceptance",
        "finding": "the mode must count to a target and score how close it landed",
        "ids": ["FND-acceptance-1"],
        "footprint": {"existing_functions": [], "new_items": [], "shared_engine": []},
    },
])
PERSPECTIVES_ANSWER = "perspectives: the simplest thing that works, and the criteria"
DESIGN_ANSWER = "a design"
#: The three reviews and the overwatch pass. Overwatch states its verdict because
#: the gate reads that line rather than trusting the review's own summary.
REVIEW_ANSWERS = [
    "health: the package imports, registers and smoke-scores",
    "code review: the ops, nodes and knobs agree with the design",
    "qa: every acceptance criterion has a check and they hold",
    "VERDICT: PASS",
]

#: The verifier's answer: a passing verification with no blocking findings.
VERIFICATION_ANSWER = "No blocking findings.\n\nVERDICT: PASS"

#: A contract-compliant design.md answer for design-mode's design_doc agent.
DESIGN_DOC_ANSWER = (
    "# Do the thing\n\n"
    "## Summary\n\nFix it properly.\n\n"
    "## Failure trace\n\nN/A — this is a new feature.\n\n"
    "## Ask mapping\n\n"
    "| Ask (quoted from brief) | Design section | Test |\n"
    "|---|---|---|\n"
    '| "a tool" | § Changes | test_tool.py |\n\n'
    "## Rule and sites\n\n"
    "- `tool.py:1` — **[changed]**\n\n"
    "## Docs\n\n"
    "- docs/models.md\n\n"
    "## Changes\n\n"
    "- tool.py\n\n"
    "## Tests\n\n"
    "- `test_tool.py::test_build` — tool.py prints what it built\n\n"
    "## Research and perspective findings\n\n"
    "- FND-api-surface-1: addressed at `tool.py:1`\n"
    "- FND-existing-modes-1: addressed at § Changes\n"
    "- FND-acceptance-1: addressed at `tool.py:1`\n"
    "- FND-simplest-1: addressed at `tool.py:1`\n"
    "- FND-criteria-1: addressed at § Changes\n\n"
    "## Anti-patterns to avoid\n\n"
    "- PIT-001: addressed at `tool.py:1`\n"
)


def create_script(name: str, emits: list[str]) -> list[str]:
    """The whole answer script, with the given emit answers in the order used.

    A reloop back to `emit` consumes the next one, so a test that expects a retry
    passes two; a run whose validation never passes is scripted without the reviews,
    because it never reaches them.
    """
    return [
        RESEARCH_ANSWER,
        PERSPECTIVES_ANSWER,
        strategy_answer(name),
        DESIGN_ANSWER,
        *emits,
        *REVIEW_ANSWERS,
    ]


def create_answers(name: str, files: dict[str, str]) -> list[str]:
    """Every answer create-v2's agents need, in the order the graph asks for them.

    The three reviews and the overwatch pass answer as text; the executor writes
    each to the file the node declares, and the QA synthesis and its gate read them
    back. Overwatch must state its verdict line, because the gate reads it.
    """
    return create_script(name, [render(files)])


def render(files: dict[str, str]) -> str:
    """The emit agent's answer format: `===FILE: name===` blocks."""
    return "\n".join(f"===FILE: {name}===\n{body}" for name, body in files.items()) + "\n"


#: What a review beside a check answers when its script has no review for that attempt
#: (`FakeModel.complete`), keyed by a line of each reviewer's prompt. Only a prompt that
#: says the check runs beside it is answered this way.
BESIDE_CHECK = "## The check runs beside this review"
BESIDE_CHECK_REVIEWS = (
    ("You are the health checker.", "health: ok"),
    ("You are the code reviewer.", "code review: ok"),
    ("You are the QA director.", "qa: ok"),
    ("You verify the change against the brief and the repository only.", "No blocking findings.\n\nVERDICT: PASS"),
)


def _is_emission(answer: str) -> bool:
    return any(marker in answer for marker in ("===FILE:", "===EDIT:", "===DELETE:"))


def apply_in_workspace(answer: str, workspace) -> bool:
    """What a scripted builder does in its workspace: the emission's changes, made there.

    A builder that works in a workspace (design mode's) changes files rather than
    writing blocks, and the runtime derives the blocks from its diff. A test scripts it
    with the blocks it used to emit: each whole file is written as given (text copied
    from a truncated read included, as a builder would write it), each edit is applied to
    the workspace's file as it is, each deletion made. A block a builder could not have
    made in its workspace -- malformed, an edit whose text is not there, a path outside
    it -- changes nothing, as a builder that made no change would. Returns whether
    anything changed.
    """
    from pathlib import Path

    from refactory_dsh import repo_change
    from refactory_dsh.worktree import AttemptRefused

    root = Path(str(workspace))
    matches = list(repo_change.BLOCK.finditer(answer))
    changed = False
    for index, match in enumerate(matches):
        kind, relative = match.group(1), Path(match.group(2))
        if relative.is_absolute() or ".." in relative.parts:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(answer)
        body = answer[match.end():end].strip("\n")
        target = root / relative
        if kind == "DELETE":
            if target.is_file():
                target.unlink()
                changed = True
            continue
        if kind == "FILE":
            text = body + "\n"
        else:
            if not target.is_file():
                continue
            try:
                text, _ = repo_change._edited(relative, target.read_text(),
                                              repo_change._pairs(relative, body), str(relative))
            except AttemptRefused:
                continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or target.read_text() != text:
            target.write_text(text)
            changed = True
    return changed


#: Design mode's two judges (`refactory_dsh.judges`), keyed by a line of each prompt.
QA_JUDGE_MARK = "You are the QA judge."
OVERWATCH_JUDGE_MARK = "Write `overwatch-verdict.json`"
#: The early asks judge's prompt marker (`design_ops.EARLY_ASKS_JUDGE_MARK`).
EARLY_ASKS_JUDGE_MARK = "You are the early asks judge."

#: What a judge answers when the script has no verdict for it: the QA judge proceeds and
#: the overwatch passes, with nothing listed.
QA_PROCEED = json.dumps({"verdict": "proceed", "must_fix": [], "follow_ups": [], "dismissed": []})
OVERWATCH_PASS = json.dumps({"verdict": "pass", "must_fix": [], "follow_ups": [], "corrections": []})


def qa_judgment(verdict: str = "proceed", must_fix: list | None = None,
                follow_ups: list | None = None, dismissed: list | None = None) -> str:
    """A scripted QA judge's answer: the JSON it writes to qa-judge.json."""
    return json.dumps({"verdict": verdict, "must_fix": must_fix or [],
                       "follow_ups": follow_ups or [], "dismissed": dismissed or []})


def overwatch_judgment(verdict: str = "pass", must_fix: list | None = None,
                       follow_ups: list | None = None, corrections: list | None = None) -> str:
    """A scripted design overwatch's answer: the JSON it writes to overwatch-verdict.json."""
    return json.dumps({"verdict": verdict, "must_fix": must_fix or [],
                       "follow_ups": follow_ups or [], "corrections": corrections or []})


def early_asks_judgment(*items: dict, prompt: str = "") -> str:
    """A scripted early asks judge's answer: the JSON it writes to early-asks.json.

    With no ``items``, every ask is built: one ask, and one correction when the prompt
    gives the person's corrections, each citing a line of the diff.
    """
    if not items:
        built = [{"id": "ask-1", "kind": "ask", "text": "the brief", "built": True,
                  "evidence": "changes.diff:1"}]
        corrections = prompt.split("## Corrections the person gave when approving the design\n",
                                   1)[-1].split("\n\n## What to read", 1)[0] if prompt else ""
        if corrections.strip():
            built.append({"id": "correction-1", "kind": "correction", "text": "the corrections",
                          "built": True, "evidence": "changes.diff:1"})
        items = tuple(built)
    return json.dumps({"items": list(items)})


def judge_finds_everything_built(ctx) -> dict:
    """Stands in for the early asks judge's agent node on a walk of the spine's node
    processes: writes the verdict a judge that found everything built writes."""
    ctx.write_text("early-asks.json", early_asks_judgment())
    return {}


def not_built(text: str, kind: str = "ask", missing: str = "") -> dict:
    """One item the scripted early asks judge found not built."""
    return {"id": f"{kind}-9", "kind": kind, "text": text, "built": False, "evidence": "",
            "missing": missing or f"nothing in the diff does {text!r}"}


class Scripted(str):
    """An answer the next node to ask gets as written, a judge included: how a test
    scripts a judge's unreadable file."""


def _judgment_of(answer: str) -> dict:
    try:
        data = json.loads(answer)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _judge_answer(prompt: str, answers: list[str]) -> str | None:
    """A judge's answer when the script has none for it, or None to take the script's next.

    The QA judge takes the next answer when it is a QA verdict (`qa_judgment`) and
    otherwise proceeds. The overwatch takes the next answer when it is an overwatch
    verdict (`overwatch_judgment`), and passes in place of a create-style overwatch answer
    that ends in `VERDICT: PASS`, which it consumes, so a script written before the judges
    still reads as the run it describes. Any other answer, and a `Scripted` one, is taken
    as written: an unreadable verdict is what the gate then sees.
    """
    following = answers[0] if answers else ""
    if isinstance(following, Scripted):
        return None
    judgment = _judgment_of(following)
    if QA_JUDGE_MARK in prompt:
        is_qa = judgment.get("verdict") in ("proceed", "redo") and "corrections" not in judgment
        return None if is_qa else QA_PROCEED
    if OVERWATCH_JUDGE_MARK in prompt:
        if judgment.get("verdict") in ("pass", "redo") and "dismissed" not in judgment:
            return None
        lines = [line.strip() for line in following.splitlines() if line.strip()]
        if lines and lines[-1] == "VERDICT: PASS":
            answers.pop(0)
            return OVERWATCH_PASS
    return None


class FakeModel:
    """Answers with the given texts, in order.

    Design mode reviews an attempt beside its check, so an attempt whose check fails is
    reviewed too, though `gate_check` sends it back without reading the reviews. A script
    written for the check before the reviews has its next emission queued at that point:
    a reviewer asking then gets a passing review instead, and the script is not consumed.

    Design mode's QA judge and overwatch answer in JSON. A script with no verdict for a
    judge gets the default one (`_judge_answer`): scripts written before the judges, and
    tests about something else, need not script them. The early asks judge, which runs
    beside every attempt's first self-check, answers from ``early_asks`` in order, a
    scripted verdict for each call, and finds everything built once that is empty.
    """

    def __init__(self, answers: list[str], early_asks: list[str] | None = None) -> None:
        self.answers = list(answers)
        self.early_asks = list(early_asks or [])
        self.prompts: list[str] = []
        #: The directory each completion was told to run in; see the walker's `_ask_model`.
        self.workdirs: list[object] = []
        #: Where each completion's agent worked: its workspace (`workspace.py`), or None
        #: when it worked in the run directory.
        self.cwds: list[object] = []
        self.envs: list[object] = []

    def complete(self, prompt: str, *, system: str = "", workdir: object = None, cwd: object = None,
                 env: object = None, confine: bool = False) -> str:
        self.prompts.append(prompt)
        self.workdirs.append(workdir)
        self.cwds.append(cwd)
        self.envs.append(env)
        answer = self._answer(prompt, workdir)
        # A builder that works in a workspace makes its scripted emission's changes there;
        # the runtime derives emit.txt from them (`apply_in_workspace`).
        if cwd is not None and _is_emission(answer):
            apply_in_workspace(answer, cwd)
        return answer

    def _answer(self, prompt: str, workdir: object) -> str:
        if EARLY_ASKS_JUDGE_MARK in prompt:
            queue = getattr(self, "early_asks", None)
            return queue.pop(0) if queue else early_asks_judgment(prompt=prompt)
        beside = (next((text for marker, text in BESIDE_CHECK_REVIEWS if marker in prompt), None)
                  if BESIDE_CHECK in prompt else None)
        if beside is not None and self.answers and _is_emission(self.answers[0]):
            return beside
        judged = _judge_answer(prompt, self.answers)
        answer = judged if judged is not None else (self.answers.pop(0) if self.answers else "")
        # Scripted pitfalls fixtures: agent_delivery's capture_final writes the
        # model's text answer to writes[0] (pitfalls.json), but pitfalls.md
        # (writes[1]) must also exist for delivery to succeed.  Write it as a
        # side-effect, matching the intent-audit fixture below.
        # Scripted strategy fixtures: write strategy/*.md files for V6 compliance.
        if answer and "strategy director" in prompt.lower() and workdir is not None:
            from pathlib import Path

            root = Path(workdir)
            for slug, text in [("simplest", "the simplest thing"),
                               ("criteria", "the criteria check")]:
                sfile = root / "strategy" / f"{slug}.md"
                sfile.parent.mkdir(parents=True, exist_ok=True)
                sfile.write_text(
                    f"### FND-{slug}-1 — {text}\n\n# {slug}\n\nDetails.\n"
                )
        # Scripted research/extra.md: opt-in, used only by the dropped-finding test.
        if (answer and "research director" in prompt.lower()
                and workdir is not None
                and getattr(self, '_write_extra_research', False)):
            from pathlib import Path

            root = Path(workdir)
            extra = root / "research" / "extra.md"
            extra.parent.mkdir(parents=True, exist_ok=True)
            extra.write_text(
                "### FND-extra-1 — extra finding text\n\n"
                "# Extra\n\nSomething important.\n"
            )
        if answer and "pitfalls researcher" in prompt and workdir is not None:
            from pathlib import Path

            root = Path(workdir)
            root.mkdir(parents=True, exist_ok=True)
            (root / "pitfalls.md").write_text(
                "# Pitfalls\n\n(scripted fixture)\n"
            )
        # Scripted review fixtures explicitly deliver the new evidence-map artifact.
        # Adversarial coverage/waiver tests supply their own maps instead.
        if answer and "Write intent-audit.json as" in prompt and workdir is not None:
            from pathlib import Path

            root = Path(workdir)
            baseline = json.loads((root / "authoring-evidence.json").read_text())
            audit_rows = []
            for item in baseline["requirements"]:
                row = {"id": item["id"], "requirement": item.get("text", item.get("path", "")),
                       "status": "pass",
                       "evidence": [{"path": "authoring-evidence.json",
                                     "detail": "scripted fixture inspected the executed result"}]}
                if item.get("kind") in ("correction", "constraint"):
                    text = item.get("text", "")
                    # Keyed by requirement id, or by the correction's own text.
                    by = getattr(self, '_correction_sites', {})
                    sites = by.get(item["id"]) or by.get(text, [])
                    clause = {"clause": text,
                              "evidence": [{"path": "authoring-evidence.json",
                                            "detail": "full constraint verified"}],
                              "sites": sites}
                    # A test whose correction is about the design itself can give
                    # its site-less clauses an evidence form, e.g. {"kind": "design",
                    # "heading": "Summary"}.
                    if not sites:
                        clause.update(getattr(self, "_siteless_clause", {}))
                    row["clauses"] = [clause]
                audit_rows.append(row)
            (root / "intent-audit.json").write_text(json.dumps({"requirements": audit_rows}))
        # Scripted repo_questions: when the run directory has changed-functions.json,
        # auto-generate a valid repo_questions record for reviewer answers only.
        # Only inject into prompts that reference "Changed functions:" (reviewer prompts),
        # never into emit, verifier, strategy or other non-reviewer answers.
        if answer and workdir is not None and "Changed functions:" in prompt:
            from pathlib import Path

            root = Path(workdir)
            cf_path = root / "changed-functions.json"
            if cf_path.is_file():
                try:
                    cf_data = json.loads(cf_path.read_text())
                    if isinstance(cf_data, list) and cf_data:
                        # Build a valid record from the computed list
                        rq_entries = []
                        for entry in cf_data:
                            if isinstance(entry, dict) and "function" in entry:
                                fn = entry["function"]
                                file_path = entry.get("file", "unknown.py")
                                line = entry.get("line", 1)
                                rq_entries.append({
                                    "function": fn,
                                    "duplicates": f"absent: searched the repository for {fn}",
                                    "callers": f"{file_path}:{line} is called from main",
                                    "shared_state": f"absent: searched {fn} for shared state",
                                })
                        if rq_entries:
                            rq_json = json.dumps(rq_entries)
                            # For JSON answers (focused-qa.json): inject repo_questions key.
                            # Check this BEFORE the markdown path: a focused_qa answer is
                            # valid JSON that does not contain "repo_questions" as text, so
                            # the markdown guard would match first and then skip because the
                            # prompt mentions focused-qa.json, leaving the elif unreachable.
                            if ("Write focused-qa.json" in prompt or "focused-qa.json" in prompt) and answer.strip().startswith("{"):
                                try:
                                    qa_obj = json.loads(answer)
                                    if isinstance(qa_obj, dict) and "repo_questions" not in qa_obj:
                                        qa_obj["repo_questions"] = rq_entries
                                        answer = json.dumps(qa_obj)
                                except (json.JSONDecodeError, ValueError):
                                    pass
                            # For markdown answers (reviews/*.md): append if not already present
                            elif "## Repository questions" not in answer and "repo_questions" not in answer:
                                answer = answer.rstrip()
                                # Insert before VERDICT line if present
                                if "\nVERDICT:" in answer:
                                    parts = answer.rsplit("\nVERDICT:", 1)
                                    answer = (
                                        parts[0] + "\n\n"
                                        "## Repository questions\n\n"
                                        "```json\n" + rq_json + "\n```\n\n"
                                        "VERDICT:" + parts[1]
                                    )
                                else:
                                    answer += (
                                        "\n\n## Repository questions\n\n"
                                        "```json\n" + rq_json + "\n```\n"
                                    )
                except (json.JSONDecodeError, OSError):
                    pass
        return answer


class StaleEmitModel(FakeModel):
    """The redo's emit worker: an emit.txt already in its run reads to it as the work done."""

    def complete(self, prompt: str, *, system: str = "", workdir: object = None, cwd: object = None,
                 env: object = None, confine: bool = False) -> str:
        from pathlib import Path

        answer = super().complete(prompt, system=system, workdir=workdir, cwd=cwd, env=env, confine=confine)
        if "Write the artifact `" in prompt and (Path(str(workdir)) / "emit.txt").exists():
            return "emit.txt already contains the correct content."
        return answer


def git(root, *args: str, env: dict[str, str] | None = None) -> str:
    """What a git command prints in ``root``; a failing command fails the test."""
    import subprocess

    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                          text=True, env=env).stdout.strip()


def git_repo(root):
    """A person's checkout: a repository on `main` whose one commit holds `app.py`."""
    root.mkdir(parents=True)
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "test")
    (root / "app.py").write_text("print('app')\n")
    git(root, "add", "app.py")
    git(root, "commit", "-q", "-m", "base")
    return root


def stub(bin_dir, name: str, body: str) -> None:
    """A stand-in for a tool on the PATH, which logs each call to `calls.log` beside it."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / name
    script.write_text(f"#!/bin/sh\necho \"{name} $*\" >> \"{bin_dir}/calls.log\"\n{body}\n")
    script.chmod(0o755)


@contextmanager
def on_stdin(text: str) -> Iterator[None]:
    """Put ``text`` on this process's stdin, so a child that inherited it could read it."""
    read, write = os.pipe()
    os.write(write, text.encode())
    os.close(write)
    saved = os.dup(0)
    os.dup2(read, 0)
    os.close(read)
    try:
        yield
    finally:
        os.dup2(saved, 0)
        os.close(saved)
