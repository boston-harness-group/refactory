# Datasets and Tasks

A mode normally walks its graph once. A mode with a **`data` node** walks a subgraph
once per *item* — one benchmark instance, one row of a file, one directory — and an
item can carry its own prompt and its own verdict. That is how a benchmark or a
dataset becomes a mode instead of a script around one.

## A harness is for a class of work

What this system produces is a harness for a **class** of work — "write the build config for any Java buildroot", not "write the config for this buildroot". A mode that runs once is the special case rather than the default, because a mode that was tuned until one instance passed makes no claim about the next instance.

The `data` node is the harness's **input port**. What it is pointed at at run time — a bound Task's `instances()`, a directory, a JSONL or CSV file — is material rather than meaning, and a harness with its items written into the graph is a harness for exactly those items. `create` binds a Task by name and leaves the port open, and the validate gate refuses a package whose items are baked into the graph instead of read from the node at run time.

A class-of-work harness scores **per item**: its `score()` aggregates the verdicts the Task's `verify` returned for each pass, and the run reports the individual verdicts under the `item_scores` fact alongside that aggregate. That is what makes the score a statement about the class rather than about the one run that produced it.

## The node

```python
graph.add(nn.Node(
    "data", "data",
    subgraph_entry="work",     # the first node of the per-item subgraph
    subgraph_exit="work",      # the last one, included in every pass
    task_ref="my_pack.tasks:MyTask",   # where the items come from (optional)
    writes=("current_item.json",),
    produces=("item_scores",),
))
graph.connect("data", "finish")        # where the run goes after every item
```

The subgraph is reached **through the data node**, not by an edge: an edge into it
would run it once outside the iteration. A pass ends once its exit has run, on both
runtimes, so the exit has no out-edge (the validator's V12); an edit that adds a node
after the exit (`insert_after`, `duplicate`) makes that node the new exit, and one that
adds a node before the entry makes it the new entry. Each pass writes the item to
`current_item.json` in the run directory, so an op or an agent node reads the item it
is working on rather than being told it in a prompt.

Items come from, in order of precedence: `inline_items` declared on the node; the
`source_path` read as `source_format` (`directory` — sorted subdirectories; `jsonl` —
one object per line; `csv` — a header row); or the bound Task's `instances()`. The
node's `limit` caps what is used. A source that is named and missing is refused by
name, because an empty selection and an unbound source must not look alike.

## Feeding it at run time

The graph declares the source the mode was built with, and a run may name another. That is what makes a harness usable on items it has never seen without editing the mode:

```bash
uv run refactory ceo java_build_pack/build --source /path/to/my/trees
```

The runner writes the choice into the run directory as `items.json`, and `load_items` reads it on both runtimes, so the walker and DSH iterate the same items. `--source-format` says how to read it (`directory` by default, or `jsonl`/`csv`), and an entry keyed `*` applies to every data node of the run while an entry keyed by node id wins for that node.

An item in a `directory` source is a subdirectory, and it may describe itself in an `item.json` beside its content — those fields reach the pass the way a Task's metadata does, which is how a harness learns an item's entry point. The directory's own name stays the id, so a described item cannot rename itself out of the selection it was listed in.

## The Task

A Task is three optional hooks, resolved by `module:Class` or by a name registered with
`register_task`. A reference that does not resolve is refused rather than run with
nothing judging the items.

```python
from refactory_dsh.task import Task, TaskInstance, VerifyResult

class CountingTask(Task):
    name = "counting"

    def instances(self):                      # the items
        return [TaskInstance(id="alpha"), TaskInstance(id="beta")]

    def setup(self, instance, workspace):     # before a pass
        ...

    def verify(self, instance, workspace):    # after a pass; None means "no opinion"
        return VerifyResult(passed=True, score=1.0)
```

There is no hook that writes the prompt. An item is data — it reaches the pass as `current_item.json` — and what the agent for an item is asked is the mode's own business, stated in its agent node's prompt template. Two places that both decide the ask is one place too many.

An item that carries its own answer says which fields are the answer's. `grader_fields`
on the Task (default none) names them: `current_item.json`, which every node of the pass
can read (an optimizer's new nodes included), never holds them, on either runtime, and
`verify` still gets the whole item. `revealed_by` maps a grader-only field to the knob
that shows it to the nodes in a run that turns the knob on; a mode freezes such a knob,
so a search cannot. LiveMath's Task hides `correct_choice`, `label`, `label_text`,
`theorem` and `sketch`, and its frozen `use_theorem` and `use_sketch` reveal the last two.

The verdicts are recorded as `item_scores` when the data node declares that fact, so
`score()` reads what the Task decided rather than a number the graph guessed.

## A worked example

A package with one data node bound to a two-item Task, whose subgraph is one op that
appends the current item's id, scores the mean of the Task's verdicts:

```bash
uv run refactory package smoke dataset_probe/dataset
```

```json
{"mode": "dataset_probe/dataset", "package": "dataset_probe", "exit_ok": true,
 "error": null, "score": 0.75, "iterations": 4}
```

`0.75` is the mean of `1.0` for `alpha` and `0.5` for `beta`: two passes, each with
its own item on disk, each judged by the Task.

A two-item Task proves the shape but not the claim. `java_build_pack` is the example that does the work the product is named for: its Task lists three small Java source trees, its one agent node writes `build.sh` for whichever item is on disk, and the Task runs that script and compares what the program printed with the output recorded beside the sources. The mode's score is the share of trees it built, and pointing the same harness at a fourth tree it has never seen is what the binding buys:

```bash
uv run refactory ceo java_build_pack/build --intent "build each Java tree the Task lists"
```

## Creating one

`create` makes you say which of the two things you are building, before it spends anything on a model:

```bash
# a harness for a class of work: the Task supplies the items and judges each pass
uv run refactory ceo "a harness that counts the items of any counting instance" \
  --mode create --name my_pack --task my_pack.tasks:CountingTask

# the special case: a harness for one run of one thing
uv run refactory ceo "a harness that counts to ten" --mode create --name my_once --single
```

Naming a Task is the whole directive: the produced package gets a `data` node bound to that Task, and the validate gate checks the package really has one rather than taking the prompt's word for it. A creation that names neither a Task nor `--single` is refused with both options named, because a harness built for one hardcoded example is the special case and must be asked for as one.

The validate gate refuses three shapes, each by name: a package that declares no data node when a Task was named, one whose data node is bound to a different Task, and one that closes its input port — items written into the graph as `inline_items`, or a data node that names neither a Task nor a source to read at run time. Asking is not the same as getting.

## On the runtimes

Both runtimes iterate the node the same way and produce the same facts. The walker
owns the whole loop and runs it in-process. DSH's spine walks the subgraph the way it
walks the rest of the graph, so the item loop is split across the loader's three
commands — `list` (how many items), `begin` (materialise one and run the Task's
`setup`), and `end` (run the Task's `verify` and record the score) — and each pass's
deterministic nodes run in-process while its agent nodes are handed to the subagents
the graph names. The loader is one implementation
(`refactory_dsh.nn.executor.load_items`), so the items a mode selects do not depend on
which runtime is running it. The composition supplies the command
(`dataCommand`, set by the `graph` bundle to `python -m refactory_dsh.items`); a graph
with a data node and no loader is refused at load rather than handed to the model,
because a model cannot enter a subgraph the cursor never enters.

`refactory/tests/test_equivalence.py` runs one data-driven mode on both runtimes and
compares the facts.

## Failed items and the all-fail rule

Each item's lifecycle (setup → subgraph → verify) is wrapped in a try/except. A
failed item is recorded as:

```json
{"id": "…", "passed": false, "score": 0.0, "error_text": "…", "error_kind": "…"}
```

- **`error_kind`** is the exception class name (e.g. `"ValueError"`,
  `"RuntimeError"`), or `"timeout"` when the per-item deadline is exceeded. On the
  walker, `error_kind` is the class name of the actual exception raised during setup,
  verify or the subgraph walk (a setup `ValueError` records `"ValueError"`; a branch
  failure wrapped in `RuntimeError` records `"RuntimeError"`). On the spine, `error_kind`
  is `"RuntimeError"` for subgraph and verify failures recorded through `end --failed`
  (because the wrapping at `items.py` uses `RuntimeError`), except when the original
  record was written by `command_begin`'s setup handler (which preserves the original
  class).
- **`error_text`** describes the failure, truncated to 4096 characters. On the walker,
  it is the exception message. On the spine, it is the failing command's stderr tail
  (bounded to 4096 chars), passed to `end --failed`. The two texts describe the same
  failure but are not word-for-word identical.

A data node fails only when **every** item it processed fails (the all-fail rule).
The count is against items, not records, so a 3-item node with 1 failure and 2
passes succeeds. On the walker, the check compares the number of failed items
against the total item count. On the spine, the check uses run-local pass state
(processed and failed counts under `.refactory/pass_state/<node_id>.json`) so it
works whether or not the node declares `item_scores` in its `produces`.

A per-item deadline covers only setup and verify (and item loading), matching the
spine's per-command `runDataCommand` deadline. The deadline does not depend on
`enable_recovery`.
