# 1. Write a mode

A mode is a Python class. This tutorial builds the smallest one that is worth
optimizing, runs it, and then breaks it on purpose so you can see the validator
refuse it.

Everything here is offline and takes a few seconds.

## The shape

A mode declares three things:

- **parameters** in `declare()` — what an optimizer may move;
- **structure** in `build()` — the graph, built once, pure;
- **a score** in `score(run)` — how a run becomes the number being maximized.

The graph is *data*. `build()` produces it once; after that the optimizer edits the
state, never the class ([why](../architecture.md#2-the-pytorch-mapping)).

## Write it

A mode needs a package to live in. Put it in the **workspace root**,
`.factory/packages/counter/` — the writable half of the registry, and the same
place `create`-mode writes its output. The shipped packages live beside it in
`refactory_packages/`, which is read-only.

So this goes at `.factory/packages/counter/__init__.py`:

```python
"""A counter package: one mode, and the ops it calls."""
from refactory_dsh.nn import Package

from .mode import Counter

PACKAGE = Package(
    name="counter",
    description="count up to a target, and score the count reached",
    modes=(Counter,),
    ops=("refactory_packages.counter.ops",),
)
```

`.factory/packages/counter/mode.py`:

```python
"""The counter mode."""
from refactory_dsh.nn import Graph, Knob, Mode, Node


class Counter(Mode):
    name = "counter"
    description = "count up to a target"

    @classmethod
    def declare(cls):
        return {
            "target": Knob(3, "int", lo=1, hi=9, description="where to stop counting"),
            "step": Knob(1, "int", lo=1, hi=3, description="how much each step adds"),
        }

    def build(self):
        graph = Graph()
        graph.add(Node(
            "count", "op", call="refactory_packages.counter.ops:count",
            knobs=("target", "step"), writes=("count.json",), produces=("value",),
            notes="add `step` until the count reaches `target`",
        ))
        return graph

    def score(self, run):
        # What the run fact says, never a knob: a score that reads a knob is a
        # score about the input rather than about the run.
        return float(run.facts["value"])
```

`.factory/packages/counter/ops.py`:

```python
"""The operation the counter's node calls."""
from typing import Any


def count(ctx: Any) -> dict[str, Any]:
    target = int(ctx.knob("target", 3))
    step = int(ctx.knob("step", 1))
    value = 0
    while value < target:
        value += step
    ctx.write_json("count.json", {"value": value})
    return {"value": value}
```

Three rules the code above obeys, each of which the validator enforces if you
break it:

- the node lists `knobs=("target", "step")` — a declared knob no node consumes is
  refused (`V7`);
- the node declares `writes=("count.json",)` — a read of a file nothing writes is
  refused (`V6`);
- `score()` reads `run.facts`, never `ctx.knob(...)`.

## Register and run it

The registry scans `refactory_packages/`, so the package is found by name:

```bash
uv run refactory package list
```

```
counter/  count up to a target, and score the count reached
  counter/counter                        count up to a target
```

```bash
uv run python -c "
from pathlib import Path
from refactory_dsh import registry
package, mode_class = registry.resolve(Path('.'), 'counter/counter')
mode = mode_class()
print('state keys:', sorted(mode.state_dict()))
print('score     :', mode.evaluate(Path('/tmp/counter')).score)
"
```

```
state keys: ['graph', 'knobs', 'memory', 'ports', 'version']
score     : 3.0
```

Note the state keys: the architecture (`graph`) travels with the parameters
(`knobs`). That is the divergence from PyTorch this whole system rests on.

## Validate it before running it

Validation is what makes a search cheap: an invalid candidate is refused before it
costs a run.

```bash
uv run python -c "
from pathlib import Path
from refactory_dsh import nn, registry
mode = registry.resolve(Path('.'), 'counter/counter')[1]()
print(nn.validate_state(mode.state_dict(), invariants=mode.invariants()))
"
```

```
[]
```

An empty list means runnable and scoreable. Now break it deliberately — point the
node at a file nothing produces:

```bash
uv run python -c "
from pathlib import Path
from refactory_dsh import nn, registry
from refactory_dsh.nn import edits
mode = registry.resolve(Path('.'), 'counter/counter')[1]()
state = mode.state_dict()
broken = edits.apply({'op': 'remove_node', 'id': 'count'}, state)
for issue in nn.validate_state(broken):
    print(issue)
"
```

```
V6 [count]: reads 'nowhere.json', which no node on a path from 'count' to a terminal produces
```

That message is what an optimizer is shown *instead of* a failed run. It is why
the running example reports refusals as `V6: 4` rather than a batch of zeroes, and
why none of them cost evaluation budget.

The catalogue of edits is discoverable, so you never guess a field name:

```bash
uv run python -c "
from refactory_dsh.nn import edits
for op in edits.catalogue():
    print(op['op'], op['fields'])
"
```

## If your mode changes a repository

A mode whose agent writes code should not write into the person's checkout, and
should not wire its own git handling either. `refactory_dsh.repo_change` holds the
steps design and design_fast use, and any mode can attach them:

| Step | What it gives the run |
|---|---|
| `repo_change.knobs()`, `knob_names()`, `produces()` | the `target`, `base`, `worktree`, `setup` and `generate` knobs, and the facts prepare returns, for the prepare node to declare; when a check loads built output (e.g. `lib/*.js` from a TypeScript build, or a compiled Python extension), the `setup` knob must be set to the build command, or the run stops naming the unbuilt packages |
| `repo_change.locate(ctx, name, task, *, output_only=())` | called from your prepare op: a git worktree of the target on the run's own branch, from a pinned base (a new project when there is no target), or the artifact directory with `worktree` off; its `emit_scope` fact tells your agent where to write and in which blocks |
| `repo_change.materialise_node()` | applies `emit.txt` as the run's next attempt, one commit each; an emission it cannot apply is recorded, not fatal |
| `repo_change.refused(ctx)` | in your check gate: why the last emission could not be applied, to send back to the agent |
| `repo_change.run_check(ctx, command, timeout)` | runs a command where the run works, in the worktree's own environment; the node that calls it also declares `repo_change.CHECK_WRITES` and `CHECK_PRODUCES`, because an install that writes a lockfile amends the attempt, and a generate command amends its tracked-file changes too |
| `repo_change.report(ctx, name, payload)` | in your report op: settles the run's place, writes your report file `name` with its branch, base and diff stat added, then records the run as succeeded, which keeps its worktree and branch |

[`fix_pack/make_pass`](../../refactory_packages/fix_pack/mode.py) is a complete
mode built this way: it changes a repository until a failing check passes. Its
graph:

```python
# In refactory_packages/fix_pack/mode.py, where OPS = "refactory_packages.fix_pack.ops".
def build(self) -> nn.Graph:
    graph = nn.Graph()
    graph.add(nn.Node(
        "prepare", "op", call=f"{OPS}:prepare",
        knobs=("name", "brief", "check", *repo_change.knob_names()),
        writes=("feedback.txt",),
        produces=("name", "brief", "check", "feedback", *repo_change.produces()),
        notes="read the check, and give the run its worktree",
    ))
    graph.add(nn.Node(
        # Named `emit`, so the target guard holds the agent to emit.txt (`repo_change.OUTPUT_ONLY`).
        "emit", "agent",
        prompt_template=(
            "Make this check pass: {{check}}\n"
            "It runs from {{check_root}}.\n"
            "What is wanted: {{brief}}\n\n"
            "## What the last check said\n{{feedback}}\n\n"
            "{{emit_scope}}\n\n"
            "Write those blocks into emit.txt, with no prose outside them. Do not create or "
            "change the files themselves: the materialise step applies emit.txt, and tells the "
            "next attempt if it could not. Then return a short completion message.\n"
        ),
        reads=("feedback.txt",),
        writes=("emit.txt",),
        produces=("emit",),
        notes="write the fix as blocks in emit.txt",
    ))
    graph.add(repo_change.materialise_node())
    graph.add(nn.Node(
        "check", "gate", evaluator=f"{OPS}:check",
        knobs=("attempts", "check_timeout_s"),
        # What `repo_change.run_check` may move: an install's lockfile amends the attempt.
        writes=("feedback.txt", *repo_change.CHECK_WRITES),
        produces=("feedback", "attempts_used", *repo_change.CHECK_PRODUCES),
        max_iterations=5,
        notes="run the check where the run works; reloop with what it said until it passes",
    ))
    graph.add(nn.Node(
        "report", "op", call=f"{OPS}:report",
        writes=("result.json",),
        produces=("result",),
        notes="say which branch holds the fix",
    ))
    graph.connect("prepare", "emit")
    graph.connect("emit", "materialise")
    graph.connect("materialise", "check")
    graph.connect("check", "emit", condition="reloop")
    graph.connect("check", "report", condition="proceed")
    return graph
```

Its [ops](../../refactory_packages/fix_pack/ops.py) are as short: `prepare` returns
`repo_change.locate(ctx, name, task, *, output_only=())`'s facts, `check` sends `repo_change.refused(ctx)`
back to `emit` and otherwise runs `repo_change.run_check`, and `report` calls
`repo_change.report(ctx, "result.json", ...)`.

Three rules come with the steps:

- name the agent that writes `emit.txt` `emit`, so the target guard holds it to
  that file and a direct write to the checkout fails the node;
- list `refactory_dsh.repo_change` in your package's `ops`, so a resumed run pins
  the steps it calls as well as your own;
- the steps reach nothing outside the run: the work stays on the run's branch,
  which the report names. Design delivers its branch as a pull request with steps
  of its own (`delivery`), behind an approval gate a person answers, because that
  question is their consent to it.

A mode that changes code without these steps gets no worktree: the engine does
not add one for it.

## What to take away

- A mode is data from `build()` onward. The class is a *seed*, not the truth.
- Declare what a node reads, writes, produces, and consumes. The declarations are
  what the validator checks against, and a node that writes something it did not
  declare is an error rather than a surprise.
- `score()` reads facts. If it read a knob, optimizing would be circular.

Next: [optimize it](02-optimize-it.md).
