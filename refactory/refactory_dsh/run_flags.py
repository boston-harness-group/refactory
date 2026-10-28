"""One run's flags, for every command that starts one (dm-134, #436 gap 22).

`refactory ceo`, which runs on the spine (through `refactory_dsh.interactive`), and the
walker's mains, `python -m refactory_dsh.design | create | edit`, take a run's flags from
here: one spelling, one parse, one meaning on both runtimes.

- **Knobs.** Everything a flag says about the run that a mode declares as a knob -- the
  brief, the name, the check, where the run works, how it is delivered, how many attempts
  it gets -- becomes the same ``NAME=VALUE`` list on both (:func:`knobs`), applied as
  ``--knob`` is. A knob the mode does not declare is refused by name, as ``--knob`` is.
- **The model route** (:func:`model_options`) and **the resource limits**
  (:func:`resource_limits`) are read the same way for both.
- **Refusals** (:func:`refusal`). Every flag either reaches the runtime that runs it or
  is refused with a message naming what to use instead. None is accepted and dropped.

Two spellings are aliases, and say what they map to: ``--verdict`` is ``--approve``, and
``--attempts N`` is the mode's attempts knob (:data:`ATTEMPTS`).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SPINE = "spine"
WALKER = "walker"

#: The authoring modes that work in a worktree of their target (#351).
DESIGN_MODES = ("default_pack/design", "default_pack/design_fast")
#: The flags that say how a design run is delivered (#350). design_fast, which has no
#: approval gate, takes none of them.
DELIVERY_FLAGS: tuple[str, ...] = ("--no-pr", "--refine", "--review-comments")
#: The knob ``--attempts N`` sets on each authoring mode.
ATTEMPTS = {
    "default_pack/design": "check_attempts", "default_pack/create": "attempts", "default_pack/edit": "attempts",
    "default_pack/design_fast": "repair_attempts", "default_pack/create_fast": "repair_attempts",
    "default_pack/edit_fast": "repair_attempts",
}
#: What a resume may still be told: its resource limits, and how it is run.
RESUME_TAKES = frozenset({"resume", "acknowledge_failure", "max_elapsed_s", "max_tokens", "validation_reserve_s",
                          "validation_reserve_tokens", "warn_tokens", "headless", "help", "omit_steering"})
#: The message the spine's resume refuses changes with (`interactive.run`).
RESUME_RETAINS = ("resume retains saved inputs, knobs, instructions and routes; changes require "
                  "explicit invalidation")


def slug(text: str, *, words: int = 3) -> str:
    """A name for what was asked for, when the asker did not give one."""
    kept = [part for part in re.findall(r"[a-z0-9]+", text.lower()) if part]
    name = "_".join(kept[:words]) or "new_mode"
    return f"pkg_{name}" if name[0].isdigit() else name[:40]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The flags every run takes, spelled and parsed the same on every command."""
    from refactory_dsh import repo_change

    parser.add_argument("--name", default="", help="the package or artifact name (default: from what you said)")
    parser.add_argument("--check", default="",
                        help="required for design; optional focused check for create_fast/edit_fast")
    parser.add_argument(
        "--knob", action="append", default=[], metavar="NAME=VALUE",
        help="set one of the mode's knobs for this run, e.g. allow_regression=1 (repeatable)",
    )
    parser.add_argument("--workdir", default="", help="the run directory")
    parser.add_argument(
        "--attempts", type=int, default=None, metavar="N",
        help="an alias: --knob check_attempts=N for design, attempts=N for create and edit, "
             "repair_attempts=N for their _fast variants",
    )
    parser.add_argument("--repair-attempts", type=int, default=None,
                        help="an alias: --knob repair_attempts=N, the _fast modes' repair limit (0..3)")
    parser.add_argument("--source-project", default="",
                        help="design_fast only: copy this directory into the artifact directory and apply "
                             "the change there, instead of working in a worktree")
    parser.add_argument("--target", default="", metavar="REPO",
                        help="design and design_fast: the git repository to change; the run works in a "
                             "worktree of it on its own branch (default: a new project in ~/refactory-projects)")
    parser.add_argument("--base", default="",
                        help="design and design_fast: the commit, branch or tag to start from")
    parser.add_argument("--no-worktree", action="store_true",
                        help="design and design_fast: write the artifact under .factory/artifacts instead "
                             "of a worktree")
    parser.add_argument("--setup", default="",
                        help="design and design_fast: a command run in the worktree before each check, "
                             "such as a build")
    parser.add_argument("--generate", default="",
                        help="design and design_fast: " + repo_change.knobs()["generate"].description)
    parser.add_argument("--no-pr", action="store_true",
                        help="design only: keep the work on the run's local branch instead of opening a pull "
                             "request once the reviews pass")
    parser.add_argument("--refine", default="", metavar="PR",
                        help="design only: follow up an open pull request (its URL or owner/repo#number); the "
                             "run starts from its branch and updates it")
    parser.add_argument("--review-comments", action="store_true",
                        help="design only, with --refine: add the pull request's reviews and comments to the "
                             "run's steering")
    parser.add_argument("--refine-design", default=None, choices=["light", "full"],
                        help="design only, with --refine: light (default) skips research and strategy; "
                             "full walks the whole design phase")
    parser.add_argument("--worker-backend", default="",
                        help="the backend every agent node runs on: dsh (default), codex or claude_code")
    parser.add_argument("--effort", default="", help="the workers' reasoning effort")
    parser.add_argument("--permission-mode", default="",
                        help="worker permissions: skip (new-run default), default (native policy), or a "
                             "backend-specific mode")
    parser.add_argument("--role-routes", type=json.loads, default=None, metavar="JSON",
                        help="per-role worker routes, as a JSON object")
    parser.add_argument("--node-routes", type=json.loads, default=None, metavar="JSON",
                        help="per-node worker routes, as a JSON object")
    parser.add_argument(
        "--model-timeout", type=int, default=None, dest="model_timeout_s",
        help="seconds one model completion may take before it is called hung "
             "(default: 1800); sets the per-completion limit for every agent step",
    )
    parser.add_argument("--max-elapsed-s", type=float, default=None,
                        help="shared elapsed seconds, including nested work and validation")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="shared tokens; requires provider usage telemetry")
    parser.add_argument("--validation-reserve-s", type=float, default=None,
                        help="elapsed seconds retained for required final validation")
    parser.add_argument("--validation-reserve-tokens", type=int, default=None,
                        help="tokens retained for required final validation")
    parser.add_argument("--warn-tokens", type=int, default=None,
                        help="per-step token warning threshold; emits a run_warning once per step "
                             "(does not stop the run)")
    parser.add_argument("--provider", default="",
                        help="provider route for this run's model (default: the stored choice; see "
                             "`refactory model set`)")
    parser.add_argument("--model", default="",
                        help="model id for this run, e.g. deepseek-v4-pro (default: the stored choice; see "
                             "`refactory model set`)")
    parser.add_argument(
        "--approve", "--verdict", dest="verdict", action="append", default=[], metavar="GATE=VALUE",
        help="answer a gate up front, e.g. gate_strategy=proceed, for a run with nobody sitting at "
             "it (repeatable; --verdict is an alias of --approve)",
    )
    parser.add_argument(
        "--intent", default="",
        help="run a mode: what this run is for, in your words; authored runs take the "
             "brief itself as the intent",
    )
    parser.add_argument(
        "--source", default="", metavar="PATH",
        help="run a mode: read the items from here instead of the source the mode "
             "declares, so a harness built for a class of work is pointed at items it "
             "has not seen",
    )
    parser.add_argument(
        "--source-format", default="", choices=["", "directory", "jsonl", "csv"],
        help="run a mode: how to read --source (default: a directory of one item per "
             "subdirectory)",
    )
    parser.add_argument(
        "--timeout", type=int, default=0,
        help="seconds the spine's run may take, waiting at a gate included; 0 waits for you. "
             "The walker never waits at a gate: bound it with --max-elapsed-s",
    )
    parser.add_argument(
        "--headless", action="store_true", default=False,
        help="run without a conversation: one task, one report, then exit, for scripts "
             "and CI; without it a spine run is live and a line you type between turns "
             "reaches the CEO. The walker always runs this way",
    )
    parser.add_argument(
        "--omit-steering", action="store_true", default=False,
        help="omit the steering ledger from agent prompts (benchmark runs); gate "
             "routing, correction audits and overwatch enforcement still apply",
    )


def _located(args: argparse.Namespace) -> list[str]:
    """The flags that place the run in a worktree of its target, or deliver it."""
    return [flag for flag, given in (("--target", args.target), ("--base", args.base),
                                     ("--no-worktree", args.no_worktree), ("--setup", args.setup),
                                     ("--generate", args.generate),
                                     ("--no-pr", args.no_pr), ("--refine", args.refine),
                                     ("--review-comments", args.review_comments)) if given]


def refusal(args: argparse.Namespace, mode_id: str | None, *, runtime: str, label: str = "") -> str | None:
    """Why this run's flags cannot be honoured as given, or ``None``.

    ``mode_id`` is the authoring mode the run starts, or ``None`` for a mode run by name
    (`refactory ceo <package>/<mode>`), whose knobs the mode itself checks. ``label`` is
    how the message names the mode (`--mode create`, `--fast`). The same flags give the
    same message on either runtime, except where a runtime cannot honour one.
    """
    located = _located(args)
    in_worktree = [flag for flag in located if flag not in ("--no-worktree", "--no-pr")]
    if args.no_worktree and in_worktree:
        return (f"refactory: --no-worktree cannot take {', '.join(in_worktree)}: without a worktree the "
                "artifact is written under .factory/artifacts")
    if args.refine and args.base:
        return "refactory: --refine starts from the pull request's head, so it takes no --base"
    if args.review_comments and not args.refine:
        return "refactory: --review-comments goes with --refine: they are the pull request's comments"
    if args.refine_design and not args.refine:
        return "refactory: --refine-design goes with --refine: it says how a refine designs its change"
    timeout = args.model_timeout_s
    if timeout is not None and not (1 <= timeout <= 7200):
        return f"refactory: --model-timeout must be between 1 and 7200 seconds, got {timeout}"
    if runtime == WALKER and args.timeout:
        return ("refactory: the walker does not wait at a gate (it stops there, for `refactory steer` to "
                "answer), so it takes no --timeout; bound the run with --max-elapsed-s instead")
    label = label or f"--mode {str(mode_id or '').split('/', 1)[-1]}"
    if args.attempts is not None:
        if mode_id not in ATTEMPTS:
            return (f"refactory: --attempts is an authoring mode's attempts knob, and {mode_id or 'this mode'} "
                    "names none; set the knob you mean with --knob NAME=VALUE")
        if ATTEMPTS[mode_id] == "repair_attempts" and args.repair_attempts is not None:
            return f"refactory: --attempts is --repair-attempts on {label}; give one of them"
    if mode_id is None:
        return None
    for flag, given in (("--intent", args.intent), ("--source", args.source),
                        ("--source-format", args.source_format)):
        if given:
            return (f"refactory: {flag} is for running a mode (`refactory ceo <package>/<mode> {flag} ...`); "
                    "an authoring run takes its brief as its intent and reads no items")
    if located and mode_id not in DESIGN_MODES:
        return (f"refactory: {label} does not take {', '.join(located)}: only --mode design and "
                "design_fast work in a worktree of their target")
    if mode_id == "default_pack/design_fast":
        delivered = [flag for flag in located if flag in DELIVERY_FLAGS]
        if delivered:
            return (f"refactory: {label} does not take {', '.join(delivered)}: it has no approval "
                    "gate, so it opens no pull request, and its work stays on the run's branch")
        placed = [flag for flag in located if flag in ("--target", "--base", "--setup", "--generate")]
        if args.source_project and placed:
            return (f"refactory: --source-project cannot take {', '.join(placed)}: it copies a directory "
                    "into the artifact, and they place the run in a worktree of a repository instead")
    return None


def resume_refusal(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str | None:
    """Why a resume cannot take these flags, or ``None``. A resumed run keeps the inputs,
    knobs, instructions and route it was saved with, as the spine's does
    (`interactive.run`), so any other flag given is refused rather than ignored; only its
    resource limits may change."""
    given = [action.option_strings[0] for action in parser._actions  # noqa: SLF001
             if action.option_strings and action.dest not in RESUME_TAKES
             and getattr(args, action.dest, action.default) != action.default]
    if given:
        return f"refactory: {RESUME_RETAINS} (given: {', '.join(given)})"
    return None


def knobs(args: argparse.Namespace, mode_id: str | None, *, brief: str | None = None, name: str = "",
          revision: str = "") -> list[str]:
    """The run's knobs, in one order, as ``NAME=VALUE``: what the flags say, then ``--knob``.

    Last, so a person's own ``--knob`` wins over the value derived from a flag.
    """
    found: list[str] = []
    if brief is not None:
        found.append(f"brief={brief}")
    if name:
        found.append(f"name={name}")
    if getattr(args, "task_ref", None):
        found.append(f"task_ref={args.task_ref}")
    if getattr(args, "single_shot", False):
        found.append("single_shot=1")
    if args.check:
        found.append(f"check={args.check}")
    if args.repair_attempts is not None:
        found.append(f"repair_attempts={args.repair_attempts}")
    if args.attempts is not None and mode_id in ATTEMPTS:
        found.append(f"{ATTEMPTS[mode_id]}={args.attempts}")
    if args.source_project:
        found.append(f"source_project={args.source_project}")
    if mode_id is None or mode_id in DESIGN_MODES:
        found.extend(worktree_knobs(args, deliver=mode_id != "default_pack/design_fast"))
    if revision:
        found.append(f"revision_of={revision}")
    found.extend(str(pair) for pair in args.knob)
    return found


def worktree_knobs(args: argparse.Namespace, *, deliver: bool = True) -> list[str]:
    """`--target`, `--base`, `--no-worktree`, `--setup`, `--generate` and, with ``deliver``, delivery's, as knobs."""
    found = []
    if args.target:
        # Absolute, because the run resolves it from its own directory, not this shell's.
        found.append(f"target={Path(args.target).expanduser().resolve()}")
    if args.base:
        found.append(f"base={args.base}")
    if args.no_worktree:
        found.append("worktree=0")
    if args.setup:
        found.append(f"setup={args.setup}")
    if args.generate:
        found.append(f"generate={args.generate}")
    if not deliver:
        return found
    if args.no_pr:
        found.append("pr=0")
    if args.refine:
        found.append(f"refine={args.refine}")
    if args.review_comments:
        found.append("review_comments=1")
    if args.refine_design:
        found.append(f"refine_design={args.refine_design}")
    return found


def model_options(args: argparse.Namespace) -> dict[str, Any]:
    """The model route's settings, as `DshModelClient` takes them: only those given."""
    options: dict[str, Any] = {key: getattr(args, key) for key in (
        "provider", "model", "worker_backend", "effort", "permission_mode") if getattr(args, key)}
    for key in ("role_routes", "node_routes"):
        if getattr(args, key) is not None:
            options[key] = getattr(args, key)
    if args.model_timeout_s is not None:
        options["timeout_s"] = int(args.model_timeout_s)
    return options


def resource_limits(args: argparse.Namespace) -> dict[str, Any] | None:
    """The run's shared resource limits, or ``None`` when none is set."""
    from dataclasses import asdict

    from refactory_dsh.budgets import ResourceLimits

    limits = ResourceLimits(
        max_elapsed_s=args.max_elapsed_s, max_tokens=args.max_tokens,
        validation_reserve_s=args.validation_reserve_s or 0,
        validation_reserve_tokens=args.validation_reserve_tokens or 0, warn_tokens=args.warn_tokens,
    )
    return asdict(limits) if limits.configured else None


def interactive_options(args: argparse.Namespace) -> list[str]:
    """The route, limit and session flags as `refactory_dsh.interactive` spells them; an
    option nobody gave is left out, so the interactive defaults are the ones that apply."""
    forwarded: list[str] = []
    if args.model_timeout_s is not None:
        forwarded += ["--model-timeout", str(args.model_timeout_s)]
    for key in ("worker_backend", "effort", "permission_mode", "max_elapsed_s", "max_tokens",
                "validation_reserve_s", "validation_reserve_tokens", "warn_tokens"):
        value = getattr(args, key, None)
        if value is not None and value != "":
            forwarded += ["--" + key.replace("_", "-"), str(value)]
    for key in ("role_routes", "node_routes"):
        value = getattr(args, key, None)
        if value is not None:
            forwarded += ["--" + key.replace("_", "-"), json.dumps(value)]
    if args.headless:
        forwarded += ["--headless"]
    if getattr(args, "omit_steering", False):
        forwarded += ["--omit-steering"]
    return forwarded
