"""A walker main for any registered mode, as the parity tests drive it.

``python -m refactory_dsh.design`` is the walker's main for design mode only; a workspace
mode (the complex parity scenario's) has none. This one does what that main does for any
``<package>/<mode>``: prepare the run as `refactory ceo` prepares it (`prepare_run`, the
same knobs, intent, model route and ``run-config.json``), walk it with `DshModelClient`,
and continue a stopped run from its own directory under the same checks
(`recovery.prepare_walked`). It exits 2 at a gate waiting for a person, as the design
main does.

    python -m mode_walker --mode <package>/<mode> --workdir RUN [--knob k=v ...]
    python -m mode_walker --resume --workdir RUN
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

COMMAND = "python -m mode_walker"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=COMMAND)
    parser.add_argument("--mode", default="")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--knob", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    from refactory_dsh import recovery, registry, steering
    from refactory_dsh.interactive import with_knobs
    from refactory_dsh.model import DshModelClient
    from refactory_dsh.paths import project_root
    from refactory_dsh.run_ownership import ownership
    from refactory_dsh.run_preparation import (prepare_run, print_gate_stop, saved_run_config,
                                               walker_context)

    workdir = Path(args.workdir).expanduser().resolve()
    project = project_root()
    if args.resume:
        mode_id = str(saved_run_config(workdir).get("mode") or "")
        mode_class = registry.resolve(project, mode_id)[1]
        with ownership(workdir):
            recovery.prepare_walked(workdir, mode_id)
            result = mode_class.resume(workdir, model=DshModelClient(project=None), enable_recovery=True)
    else:
        mode_id = args.mode
        mode_class = registry.resolve(project, mode_id)[1]
        mode = mode_class()
        mode.load_state_dict(with_knobs(mode.state_dict(), list(args.knob)))
        client = DshModelClient(project=project)
        workdir.mkdir(parents=True, exist_ok=True)
        with ownership(workdir):
            prepare_run(
                mode.state_dict(), workdir,
                invocation_name=mode_id.replace("/", "-"), mode_id=mode_id, intent=f"Run {mode_id}.",
                transform=lambda workflow, stage_dir=None: steering.for_run(
                    workflow, stage_dir or workdir, omit_steering=True),
                runtime="walker", command=COMMAND,
                resume_command=f"{COMMAND} --resume --workdir {workdir}",
                fast=mode_id.endswith("_fast"),
                worker_backend=client.route().get("worker_backend", ""),
                project=project, knobs=list(args.knob), model_context=walker_context(client),
                omit_steering=True,
            )
            result = mode.run(workdir, model=client, task={}, enable_recovery=True)
    if result.gate_stop:
        print_gate_stop(result.gate_stop, workdir, graph=mode_class().build())
        return 2
    if result.halted or not result.exit_ok:
        print(json.dumps({"halted": result.halted, "error": result.error}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
