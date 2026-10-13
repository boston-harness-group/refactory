# Benchmark Cheating Prevention

When building scored benchmark modes for the refactory-dsh outer loop, the agent
under evaluation has full tool access to its workspace. If gold answers are
reachable from the workspace, the agent will find and copy them instead of
solving the problem. This guide covers the known leak vectors and how to close
them.

## 1. Never put gold answers in `TaskInstance.metadata`

The data node serializes the full `metadata` dict to `current_item.json` in the
workspace. Anything in metadata is visible to the agent.

**Bad:**

```python
TaskInstance(
    id=item["id"],
    metadata={
        "question": item["question"],
        "solution": item["solution"],   # leaked to workspace
    },
)
```

**Good:**

```python
TaskInstance(
    id=item["id"],
    metadata={
        "question": item["question"],
        "fen": item["fen"],
        # solutions are NOT in metadata
    },
)
```

Look up solutions in `verify()` by reading the source data file directly:

```python
def verify(self, instance, workspace):
    solutions = self._load_solutions()  # reads from data file
    gold = solutions[instance.id]["answer"]
    # ...compare predicted vs gold...
```

## 2. Audit `setup()` for answer leaks

Even if metadata is clean, a custom `setup()` method might explicitly write
solutions to workspace files. Only write the problem statement.

```python
def setup(self, instance, workspace):
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "current_item.json").write_text(json.dumps({
        "id": instance.id,
        "question": instance.metadata["question"],
        # DO NOT include solution fields here
    }))
```

## 3. Don't expose gold answers as facts

Calling `ctx.set_fact("gold_answer", ...)` in an op may expose the answer
through the runtime's fact system. Only set facts the agent should see. Scoring
belongs in `verify()`, not in facts.

## 4. Be aware of path exposure in the worker prompt

The DSH worker prompt includes `refactory_source` and `runtime_root` paths. A
capable model could navigate from these paths to the package's `data/` directory
and read the answer file. Mitigations:

- Use puzzle IDs that don't directly map to answer file structure
- For high-stakes benchmarks, consider serving answers from an API
- Monitor for suspicious file reads in agent transcripts

## 5. Validate with a spot-check

Before trusting any 90%+ score, manually compare a few model predictions against
gold answers. If they match exactly — especially with unusual formatting — the
model is likely reading the answer.

```bash
# Quick spot-check: compare latest answer to gold
cat /tmp/run-dir/answer.txt
python3 -c "import json; items=json.load(open('data/items.json')); print(items[0]['solution'])"
```

## 6. Red-flag suspicious accuracy

If a baseline model scores vastly above published benchmarks for the same task,
assume cheating until proven otherwise. Investigate before investing compute in
optimization runs that start from an inflated baseline.

## Checklist for new benchmark modes

- [ ] `TaskInstance.metadata` contains NO solution fields
- [ ] `setup()` writes NO solution data to the workspace
- [ ] No op sets gold answers as facts
- [ ] `verify()` loads solutions from source data, not from metadata or workspace
- [ ] Spot-check confirms the model gets some items genuinely wrong
- [ ] Baseline accuracy is plausible for the model and task difficulty
