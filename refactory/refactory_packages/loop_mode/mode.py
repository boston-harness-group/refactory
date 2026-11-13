"""The optimizer loop as a mode: eight steps, and an edge back to the first.

The loop was a Python class whose control flow was `if` statements. Here it is a
graph — `init → forward → loss → backward → optimizer → step → scheduler`, then a
gate that either loops back to `forward` or leaves for `report` — so the same
outer loop that edits a mode's architecture can now edit the loop's.

`score()` reads the best score the inner run reached. That is what closes the
tower: this mode is scorable, so a run of it produces the signal a further loop
would need to search the loop itself (how many epochs, how large a batch, which
optimizer), and nothing about the outer loop has to change for that to work.
"""

from __future__ import annotations

from refactory_dsh.nn import Graph, Knob, Mode, Node, Port, RunResult

#: The loop's run directory inside this mode's working directory, and the files
#: the run leaves there. Named once, because a node's declared write and a port's
#: path are the same fact stated in two places the validator compares.
RUN = "loop"
JOURNAL = f"{RUN}/journal.jsonl"
REPORT_MD = f"{RUN}/report.md"
REPORT_JSON = f"{RUN}/report.json"
MANIFEST = f"{RUN}/manifest.json"
STATE = f"{RUN}/state.json"
CONFIG = f"{RUN}/config.json"
PENDING = f"{RUN}/pending.json"
RUNS = f"{RUN}/runs/"
PROGRESS = f"{RUN}/progress.jsonl"

#: The steps, in the order the loop runs them. A reader of this tuple is reading
#: the loop's control flow.
STEPS = ("init", "forward", "loss", "backward", "optimizer", "step", "scheduler")


def _op(step: str) -> str:
    return f"refactory_packages.loop_mode.ops:{step}"


class LoopAsMode(Mode):
    """The outer loop, as a graph the same executor runs any other mode through."""

    name = "loop"
    description = "The optimizer loop's eight steps as a runnable, searchable graph."

    @classmethod
    def declare(cls) -> dict[str, Knob]:
        # `target` says what to optimize; the rest are the loop's own settings,
        # which is why they are knobs here: an outer loop is free to move them.
        return {
            "target": Knob(
                "circle_packing/adaptive", "str", frozen=True,
                description="the <package>/<mode> this run optimizes; frozen, because "
                            "moving it changes what is measured rather than how well",
            ),
            "epochs": Knob(2, "int", lo=1, hi=16, description="how many epochs to run"),
            "batch_size": Knob(4, "int", lo=1, hi=16, description="candidates per batch"),
            "project": Knob(
                ".", "str", frozen=True,
                description="the project the target is resolved against; frozen, because "
                            "a different project is a different run rather than a knob",
            ),
            "optimizer": Knob(
                "random", "enum", enum=("random", "autoresearch"),
                description="which optimizer proposes the next batch",
            ),
            "elite": Knob(3, "int", lo=1, hi=8, description="parents each batch is proposed against"),
            "attempts": Knob(3, "int", lo=1, hi=8, description="retries per batch after a refusal"),
            "max_evaluations": Knob(
                64, "int", lo=1, hi=4096, description="hard ceiling on inner runs",
            ),
            "seed": Knob(
                0, "int", frozen=True,
                description="the run's randomness; frozen, so a score measures the "
                            "structure rather than the draw",
            ),
            "integrity_judge": Knob(
                "off", "str", frozen=True,
                description="the loop's integrity judge (LoopConfig.integrity_judge: off, "
                            "on, flag, or a JSON object); frozen, because it guards the "
                            "search rather than being a setting to optimize",
            ),
        }

    def build(self) -> Graph:
        graph = Graph()
        settings = ("epochs", "batch_size", "optimizer", "elite", "attempts",
                    "max_evaluations", "seed", "integrity_judge")
        graph.add(Node(
            "init", "op", call=_op("init"),
            knobs=("target", "project") + settings,
            writes=(CONFIG, JOURNAL, STATE), produces=("population",),
            notes="seed the population; the only step that reads the settings",
        ))
        graph.add(Node(
            "forward", "op", call=_op("forward"),
            knobs=("target", "project"),
            reads=(CONFIG, JOURNAL), writes=(JOURNAL, STATE, PROGRESS, RUNS),
            produces=("evaluated", "best"),
            notes="run the batch; the only step that spends evaluation budget",
        ))
        graph.add(Node(
            "loss", "op", call=_op("loss"),
            knobs=("target", "project"),
            reads=(JOURNAL,), writes=(JOURNAL, STATE), produces=("best", "improved"),
            notes="turn the batch into one number",
        ))
        graph.add(Node(
            "backward", "op", call=_op("backward"),
            knobs=("target", "project"),
            reads=(JOURNAL,), writes=(JOURNAL, STATE), produces=("knobs", "failures"),
            notes="read the traces into the gradient the proposal step receives",
        ))
        graph.add(Node(
            "optimizer", "op", call=_op("optimizer"),
            knobs=("target", "project"),
            reads=(CONFIG, JOURNAL), writes=(PENDING, JOURNAL, STATE),
            produces=("proposed", "accepted", "refused"),
            notes="propose the next batch, validated before anything is run",
        ))
        graph.add(Node(
            "step", "op", call=_op("step"),
            knobs=("target", "project"),
            reads=(JOURNAL, PENDING), writes=(JOURNAL, STATE),
            produces=("offered", "kept"),
            notes="install the offspring that are genuinely new points",
        ))
        graph.add(Node(
            "scheduler", "op", call=_op("scheduler"),
            knobs=("target", "project"),
            reads=(JOURNAL,), writes=("scheduler.json", JOURNAL, STATE),
            produces=("another", "bound", "best"),
            notes="close the epoch and say what bound the run",
        ))
        graph.add(Node(
            "again", "gate", evaluator=_op("again"),
            knobs=("target", "project"), reads=("scheduler.json",),
            produces=("verdict",), max_iterations=64,
            notes="loop back to forward, or leave for the report",
        ))
        graph.add(Node(
            "report", "op", call=_op("report"),
            knobs=("target", "project"),
            reads=(CONFIG, JOURNAL), writes=(REPORT_MD, REPORT_JSON, MANIFEST),
            produces=("best", "evaluations", "bound"),
            notes="write the report, the visibility surface",
        ))
        for earlier, later in zip(STEPS, STEPS[1:]):
            graph.connect(earlier, later)
        graph.connect("scheduler", "again")
        graph.connect("again", "forward", condition="again")
        graph.connect("again", "report", condition="done")
        return graph

    def score(self, run: RunResult) -> float:
        """The best score the inner run reached: the loop's own signal.

        A run that never reached a score is not the same as one that scored zero,
        so an absent best is reported as no signal rather than as a zero the outer
        loop would read as a real measurement.
        """
        best = run.facts.get("best")
        if best is None:
            raise ValueError(
                "the loop run produced no best score, so it is not searchable: "
                f"facts were {sorted(run.facts)}"
            )
        return float(best)

    def ports(self) -> tuple[Port, ...]:
        return (
            Port("report", REPORT_MD, "text/markdown", description="the loop's report"),
            Port("journal", JOURNAL, "application/x-ndjson",
                 description="every event the run recorded"),
        )

    def preferred_edits(self) -> tuple[str, ...]:
        # The setting whose value the traces most often settle is the batch size;
        # structure is rarely the lever inside a loop this short.
        return ("set_knob", "insert_after")

    def guidance(self) -> str:
        return (
            "This mode IS the optimizer loop, so its knobs are the loop's settings. "
            "Moving `target` changes what is measured rather than how well, so it is "
            "frozen. The steps are fixed by the loop's own semantics: an inserted node "
            "that cannot change the best score is advisory and is marked as such."
        )

    def insertable(self) -> tuple[Node, ...]:
        return (
            Node(
                "note", "op", call="refactory_packages.loop_mode.ops:report",
                knobs=("target", "project"), reads=(REPORT_JSON,), produces=(), score_effect="none",
                notes="re-read the run's report; observable, but it cannot move the "
                      "score, and saying so keeps a search from spending a batch on it",
            ),
        )
