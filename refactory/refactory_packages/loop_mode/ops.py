"""The eight steps of the optimizer loop, as operations a node can call.

Each step is one node of the loop's own graph, and each node may run in its own
process — so every step rehydrates the run from the journal it writes and puts
back what the next step reads. That is the honest cost of the loop being data:
what was one object's fields becomes the run directory.

`init` reads the settings and starts the run; every other step reads the
configuration the run wrote, so a step can never disagree with the one before it
about what the run was asked to do.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from refactory_dsh.loop import PENDING, Loop, LoopConfig
from refactory_dsh.outerloop import Run

#: The name the configuration is stored under, as the run directory records it.
CONFIG = "config.json"

#: Where the gate reads the scheduler's verdict from.
SCHEDULER = "scheduler.json"


#: Where the loop keeps its run, inside the mode's own working directory. Fixed
#: rather than a knob: a knob here would move the artifacts the port declarations
#: name, so the declaration would stop being true the moment anyone moved it.
RUN = "loop"


def _root(ctx: Any) -> Path:
    root = ctx.path(RUN)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _project(ctx: Any) -> Path:
    """The project the registry resolves the target against: this run's own cwd."""
    return Path(ctx.knob("project", "."))


def _target(ctx: Any) -> str:
    target = str(ctx.knob("target", "") or "")
    if not target:
        raise ValueError(
            "the loop mode needs a target: set the `target` knob to the mode to "
            "optimize, as <package>/<name>"
        )
    return target


def _settings(ctx: Any) -> dict[str, Any]:
    """The loop settings this run was started with, from the mode's own knobs."""
    return {
        "epochs": int(ctx.knob("epochs", 2)),
        "batch_size": int(ctx.knob("batch_size", 4)),
        "optimizer": str(ctx.knob("optimizer", "random")),
        "elite": int(ctx.knob("elite", 3)),
        "attempts": int(ctx.knob("attempts", 3)),
        "max_evaluations": int(ctx.knob("max_evaluations", 64)),
        "seed": int(ctx.knob("seed", 0)),
        "integrity_judge": _integrity_setting(ctx.knob("integrity_judge", "off")),
    }


def _integrity_setting(value: Any) -> Any:
    """The judge's setting as the knob holds it: a bare word, or a JSON object as text,
    checked (`optim.integrity.parse_setting`); a malformed value is refused naming the
    knob."""
    from refactory_dsh.optim.integrity import parse_setting

    text = str(value if value is not None else "off").strip() or "off"
    try:
        setting = json.loads(text) if text.startswith("{") else text
        parse_setting(setting)
    except ValueError as exc:  # json.JSONDecodeError included
        raise ValueError(f"the loop mode's integrity_judge knob is malformed ({text[:200]!r}): "
                         f"{exc}") from exc
    return setting


def _resumed(ctx: Any) -> Loop:
    """The run, rebuilt from the journal the previous steps wrote.

    A step does not carry the loop in memory — the graph runs nodes one at a time
    and may run each in its own process — so it reads the run back. The journal is
    already the run's durable state, so nothing new is invented here.
    """
    root = _root(ctx)
    stored = json.loads((root / CONFIG).read_text())
    # Resume, always: a step carries on from the journal the previous step wrote,
    # and `resume=False` means "start over" — which here would delete the run
    # between two nodes of its own loop.
    stored["resume"] = True
    loop = Loop(_target(ctx), LoopConfig.from_dict(stored), project=_project(ctx), model=getattr(ctx, "model", None))
    loop.bind_budget(getattr(ctx, "budget", None))
    # `init` is also the replay entry point: with `resume` set it rebuilds the
    # population and the counters from the journal and returns, which is exactly
    # what a step needs and is why every step goes through it.
    loop.init()
    return loop


def _best(loop: Loop) -> float | None:
    scores = [
        candidate.score
        for candidate in loop.population
        if candidate.evaluated and candidate.exit_ok and candidate.score is not None
    ]
    return max(scores) if scores else None


# ── the steps ──────────────────────────────────────────────────────────────


def init(ctx: Any) -> dict[str, Any]:
    """Seed the population, and refuse to search a target that cannot be run.

    The one step that reads the settings: it writes them into the run directory,
    and every later step reads them from there.
    """
    config = LoopConfig(runs_root=_root(ctx), **_settings(ctx))
    loop = Loop(_target(ctx), config, project=_project(ctx), model=getattr(ctx, "model", None))
    loop.bind_budget(getattr(ctx, "budget", None))
    loop.init()
    return {"population": len(loop.population)}


def forward(ctx: Any) -> dict[str, Any]:
    """Evaluate the batch: the inner runs, and the only step that spends budget."""
    loop = _resumed(ctx)
    loop.open_epoch()
    evaluated = loop.forward()
    return {"evaluated": evaluated, "best": _best(loop)}


def loss(ctx: Any) -> dict[str, Any]:
    """Turn the batch into one number, and the facts around it."""
    loop = _resumed(ctx)
    measured = loop.loss_step()
    return {"best": measured.get("best"), "improved": bool(measured.get("improved"))}


def backward(ctx: Any) -> dict[str, Any]:
    """Read the traces and compute the gradient the proposal step receives."""
    loop = _resumed(ctx)
    gradient = loop.backward()
    return {
        "knobs": len(gradient.get("knob_attribution") or {}),
        "failures": len(gradient.get("run_failures") or []),
    }


def optimizer(ctx: Any) -> dict[str, Any]:
    """Propose the next batch, and write it for the step that installs it.

    Validation already happened in here, so a refused proposal cost no budget and
    its reason is in the run's rejection log by the time this returns.
    """
    loop = _resumed(ctx)
    candidates, rejections = loop.optimizer_step()
    loop.write_pending(candidates)
    return {
        "proposed": len(candidates) + len(rejections),
        "accepted": len(candidates),
        "refused": len(rejections),
    }


def step(ctx: Any) -> dict[str, Any]:
    """Install the offspring the proposal step found, and keep what survives."""
    loop = _resumed(ctx)
    offered = loop.read_pending()
    run = Run(_loop=loop)
    kept = loop.step([candidate for candidate in offered if loop.accepts(run, candidate)], run)
    for candidate in kept:
        loop.update(run, candidate)
    return {"offered": len(offered), "kept": len(kept)}


def scheduler(ctx: Any) -> dict[str, Any]:
    """Close the epoch and record what bounded the run.

    The verdict itself is left in a file for the gate to read, because the gate is
    a separate node and may be a separate process.
    """
    loop = _resumed(ctx)
    # This epoch's counts, as the optimizer step reported them (its `proposed`,
    # `accepted` and `refused` facts), so the summary agrees with the progress line.
    another = loop.scheduler(
        evaluated=sum(1 for candidate in loop.population if candidate.evaluated),
        proposed=int(ctx.facts.get("proposed") or 0),
        accepted=int(ctx.facts.get("accepted") or 0),
        refused=int(ctx.facts.get("refused") or 0),
    )
    loop.close_epoch()
    ctx.write_json(SCHEDULER, {"another": another, "bound": loop.bound, "epoch": loop.epoch})
    return {"another": another, "bound": loop.bound, "best": _best(loop)}


def report(ctx: Any) -> dict[str, Any]:
    """Score the last epoch's proposals, then write the run's report — the visibility
    surface — and publish the score.

    A run that ended on its epoch bound scores the batch its last epoch proposed
    before it reports (``Loop.score_last_batch``), as the loop does when it runs
    itself, so the graph spends and scores what the loop does.
    """
    loop = _resumed(ctx)
    loop.score_last_batch()
    # The run's own report, as ``Loop.run`` writes it: with the integrity judge on, the
    # winner's combined change from the seed is judged here.
    loop.audit_on_report = True
    payload = loop.report()
    return {
        "best": payload.get("best_score"),
        "evaluations": payload.get("evaluations"),
        "bound": payload.get("bound"),
    }


def again(ctx: Any) -> str:
    """The loop's one decision: another epoch, or the report."""
    verdict = ctx.read_json(SCHEDULER, {"another": False})
    return "again" if verdict.get("another") else "done"


def pending_exists(ctx: Any) -> bool:
    """Whether a proposal is waiting to be installed. For a reader, not the run."""
    return ctx.path(PENDING).exists()
