"""Seed graphs, ops and a Task for the structure-edit invariant harness (`structure_fuzz`).

Small synthetic modes, each built to cover one graph shape the edit layer has to keep
wired correctly: a fork and its join (and a fork nested in a branch), the three gate
kinds with their targets, a gate loop, a data pass with a body, ops that read, write and
produce, and a graph with several terminals. LiveMath's skill graph is the real seed.

Every op here is one generic callable, `step`: it writes each file its node declares and
sets each fact its node declares, with a value named after the node. That makes any copy
of a node runnable whatever reads, writes and produces the copy declares, so a run that
fails is a wiring failure, never an op that did not expect its new position. `proceed`
is the one fn-gate evaluator. Both are resolved by name on both runtimes (this directory
is on the path of the walker and of the spine's node commands).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from refactory_dsh import nn
from refactory_dsh.task import Task, TaskInstance, VerifyResult

MODULE = "structure_fuzz_graphs"
STEP = f"{MODULE}:step"
PROCEED = f"{MODULE}:proceed"
TASK_REF = f"{MODULE}:FuzzTask"


# ── the ops ───────────────────────────────────────────────────────────────


def step(ctx: Any) -> dict[str, Any]:
    """Write every declared file and set every declared fact, deterministically."""
    node = ctx.node_id
    for name in ctx._declared:  # noqa: SLF001 - the run sets the node's declarations
        ctx.write_text(name, f"{name} written by {node}\n")
    return {fact: f"{fact} from {node}" for fact in ctx._declared_facts}  # noqa: SLF001


def proceed(ctx: Any) -> str:
    """A fn gate that always answers ``proceed``."""
    return "proceed"


class FuzzTask(Task):
    """Two items; a pass scores 1 when it left an ``answer.txt``."""

    name = "structure_fuzz"

    def instances(self, *, part: str | None = None, seed: int | None = None) -> list[TaskInstance]:
        return [TaskInstance(id="one"), TaskInstance(id="two")]

    def verify(self, instance: TaskInstance, workspace: Path) -> VerifyResult:
        passed = (Path(workspace) / "answer.txt").exists()
        return VerifyResult(passed=passed, score=1.0 if passed else 0.0)


# ── node helpers ──────────────────────────────────────────────────────────


def op(node_id: str, *, reads: tuple[str, ...] = (), writes: tuple[str, ...] = (),
       produces: tuple[str, ...] = (), knobs: tuple[str, ...] = ()) -> nn.Node:
    return nn.Node(node_id, "op", call=STEP, reads=reads, writes=writes, produces=produces,
                   knobs=knobs)


def agent(node_id: str, prompt: str, *, writes: str, reads: tuple[str, ...] = (),
          produces: tuple[str, ...] = (), system: str = "") -> nn.Node:
    return nn.Node(node_id, "agent", prompt_template=prompt,
                   system_prompt_template=system or f"You are the {node_id} step.",
                   writes=(writes,), reads=reads, produces=produces, tool_policy="none",
                   role="solver", timeout_s=60)


def gate(node_id: str, kind: str = "fn", *, reads: tuple[str, ...] = (),
         max_iterations: int = 1) -> nn.Node:
    return nn.Node(node_id, "gate", gate_kind=kind, evaluator=PROCEED if kind == "fn" else "",
                   reads=reads, max_iterations=max_iterations,
                   notes=f"does {node_id} pass?")


def _graph(nodes: list[nn.Node], edges: list[tuple[str, str] | tuple[str, str, str]]) -> nn.Graph:
    graph = nn.Graph()
    for node in nodes:
        graph.add(node)
    for edge in edges:
        graph.connect(*edge)
    return graph


# ── the seed modes ────────────────────────────────────────────────────────


class ForkJoin(nn.Mode):
    """A fork of an agent branch and a two-op branch, joined, then a deciding agent."""

    name = "fuzz_fork_join"

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {"style": nn.Knob("plain", "str", description="tone of the deciding step")}

    def build(self) -> nn.Graph:
        return _graph([
            op("setup", writes=("brief.txt",), produces=("brief",)),
            nn.Node("split", "fork", targets=("left", "right_a")),
            agent("left", "Read the brief: {{brief}}", writes="left.txt", reads=("brief.txt",)),
            op("right_a", reads=("brief.txt",), writes=("right_a.txt",), produces=("right_fact",)),
            op("right_b", reads=("right_a.txt",), writes=("right_b.txt",)),
            nn.Node("merge", "join", sources=("left", "right_b")),
            nn.Node("decide", "agent", prompt_template="Decide in a {{style}} way: {{right_fact}}",
                    system_prompt_template="You decide.", writes=("answer.txt",),
                    reads=("left.txt", "right_b.txt"), tool_policy="none", role="solver",
                    knobs=("style",)),
            op("done", reads=("answer.txt",), produces=("finished",)),
        ], [("setup", "split"), ("split", "merge"), ("left", "merge"), ("right_a", "right_b"),
            ("right_b", "merge"), ("merge", "decide"), ("decide", "done")])


class NestedFork(nn.Mode):
    """A fork whose second branch holds a fork of its own."""

    name = "fuzz_nested_fork"

    def build(self) -> nn.Graph:
        return _graph([
            op("begin", writes=("seed.txt",), produces=("topic",)),
            nn.Node("outer", "fork", targets=("solo", "inner")),
            agent("solo", "Work alone on {{topic}}", writes="solo.txt"),
            nn.Node("inner", "fork", targets=("deep_a", "deep_b")),
            op("deep_a", writes=("deep_a.txt",), produces=("deep_a_fact",)),
            agent("deep_b", "Go deep on {{topic}}", writes="deep_b.txt"),
            nn.Node("inner_join", "join", sources=("deep_a", "deep_b")),
            op("after_inner", reads=("deep_a.txt", "deep_b.txt"), writes=("inner.txt",)),
            nn.Node("outer_join", "join", sources=("solo", "after_inner")),
            op("end", reads=("solo.txt", "inner.txt"), writes=("answer.txt",),
               produces=("finished",)),
        ], [("begin", "outer"), ("outer", "outer_join"), ("solo", "outer_join"),
            ("inner", "inner_join"), ("deep_a", "inner_join"), ("deep_b", "inner_join"),
            ("inner_join", "after_inner"), ("after_inner", "outer_join"), ("outer_join", "end")])


class Gates(nn.Mode):
    """An fn gate, a user gate and an agent gate, each choosing between two targets."""

    name = "fuzz_gates"

    def build(self) -> nn.Graph:
        return _graph([
            op("draft", writes=("draft.txt",), produces=("draft",)),
            gate("check", "fn", reads=("draft.txt",)),
            agent("polish", "Polish: {{draft}}", writes="polished.txt", reads=("draft.txt",)),
            op("rough", writes=("polished.txt",)),
            gate("approve", "user", reads=("polished.txt",)),
            op("ship", reads=("polished.txt",), writes=("answer.txt",), produces=("shipped",)),
            agent("redo", "Redo the draft {{draft}}", writes="redo.txt"),
            gate("triage", "agent"),
            op("archive", reads=("answer.txt",), produces=("finished",)),
            op("drop", produces=("dropped",)),
        ], [("draft", "check"), ("check", "polish", "proceed"), ("check", "rough", "reject"),
            ("polish", "approve"), ("rough", "approve"),
            ("approve", "ship", "proceed"), ("approve", "redo", "reject"),
            ("ship", "triage"), ("triage", "archive", "proceed"), ("triage", "drop", "reject"),
            ("redo", "drop")])


class GateLoop(nn.Mode):
    """An agent step retried by an fn gate's reloop edge."""

    name = "fuzz_gate_loop"

    def build(self) -> nn.Graph:
        return _graph([
            op("prepare", writes=("task.txt",), produces=("task",)),
            agent("work", "Do the task: {{task}}", writes="work.txt", reads=("task.txt",)),
            gate("review", "fn", reads=("work.txt",), max_iterations=3),
            op("finish", reads=("work.txt",), writes=("answer.txt",), produces=("finished",)),
        ], [("prepare", "work"), ("work", "review"), ("review", "work", "reloop"),
            ("review", "finish", "proceed")])


class DataPass(nn.Mode):
    """A data pass over a Task's items: prepare, solve, check, then one run-level finish."""

    name = "fuzz_data_pass"

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {"data_parallelism": nn.Knob(2, "int", lo=1, hi=4, frozen=True,
                                            description="items at once")}

    def build(self) -> nn.Graph:
        return _graph([
            nn.Node("items", "data", subgraph_entry="read_item", subgraph_exit="check",
                    task_ref=TASK_REF, writes=("current_item.json",),
                    produces=("item_scores",), parallelism_knob="data_parallelism",
                    knobs=("data_parallelism",)),
            op("read_item", reads=("current_item.json",), produces=("question",)),
            agent("solve", "Solve: {{question}}", writes="answer.txt"),
            op("check", reads=("answer.txt",), produces=("checked",)),
            op("report", writes=("result.json",), produces=("finished",)),
        ], [("read_item", "solve"), ("solve", "check"), ("items", "report")])

    def invariants(self) -> tuple[nn.Invariant, ...]:
        return (nn.fact_produced("item_scores"),)

    def protected_nodes(self) -> tuple[str, ...]:
        return ("items", "check")


class Dataflow(nn.Mode):
    """A pipeline of ops whose files and facts feed an agent and a final op."""

    name = "fuzz_dataflow"

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {"depth": nn.Knob(2, "int", lo=1, hi=5, description="how deep to go")}

    def build(self) -> nn.Graph:
        return _graph([
            op("load", writes=("raw.txt",), produces=("source",)),
            op("clean", reads=("raw.txt",), writes=("clean.txt",), produces=("cleaned",),
               knobs=("depth",)),
            agent("summarise", "Summarise {{source}} after {{cleaned}}", writes="summary.txt",
                  reads=("clean.txt",), produces=("summary",)),
            op("emit", reads=("summary.txt", "raw.txt"), writes=("answer.txt",),
               produces=("finished",)),
        ], [("load", "clean"), ("clean", "summarise"), ("summarise", "emit")])


class MultiTerminal(nn.Mode):
    """An fn gate that ends the run at one of three terminals."""

    name = "fuzz_multi_terminal"

    def build(self) -> nn.Graph:
        return _graph([
            agent("classify", "Classify the request", writes="label.txt", produces=("label",)),
            gate("route", "fn", reads=("label.txt",)),
            op("accept", reads=("label.txt",), writes=("answer.txt",), produces=("accepted",)),
            agent("escalate", "Escalate: {{label}}", writes="escalation.txt"),
            op("reject", produces=("rejected",)),
        ], [("classify", "route"), ("route", "accept", "proceed"),
            ("route", "escalate", "escalate"), ("route", "reject", "reject")])


SYNTHETIC = (ForkJoin, NestedFork, Gates, GateLoop, DataPass, Dataflow, MultiTerminal)
