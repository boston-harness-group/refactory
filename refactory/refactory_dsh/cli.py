"""The one command: `refactory`.

    refactory ceo "an idea" --mode design     describe it; the CEO designs and builds it
    refactory ceo "an idea" --mode create     build a new package of modes
    refactory ceo <package>/<mode>            use a mode, with you answering any gate
    refactory optimize <package>/<mode>       search a mode's architecture and knobs
    refactory list                            what this checkout can do
    refactory status|leaderboard|best|adopt … read and keep a run
    refactory package list|smoke|add|…        manage packages of modes
    refactory ui                              the GUI
    refactory doctor                          is this machine ready

Every verb is a thin dispatcher onto the module that already owns it — `console`,
`registry`, `loop`, `interactive`, `create`, `design`, `edit` — so there is one
implementation and this file only decides how a person reaches it. The upstream
project's shape is the model: one `ceo` command whose `--mode` picks what is built,
because "describe what you want" is the use case, and everything else is plumbing.

    uv run refactory ceo "a mode that drafts a haiku and checks the syllables" --mode create
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from refactory_dsh.paths import REPO_ROOT, node_bin, venv_bin

#: The authoring modes, by the verb a person names them with.
AUTHORING: dict[str, str] = {
    "create_fast": "default_pack/create_fast",
    "edit_fast": "default_pack/edit_fast",
    "design_fast": "default_pack/design_fast",
    "create": "default_pack/create",
    "design": "default_pack/design",
    "edit": "default_pack/edit",
}

#: Verbs the console already answers, so `refactory <verb>` is `console <verb>`.
READ_VERBS: tuple[str, ...] = (
    "help", "docs", "workers", "profiles", "profile-save", "profile-resolve", "comparison",
    "runtime", "status", "graph-control", "runs", "artifacts", "leaderboard", "candidate", "best", "adopt", "steer", "stop",
)

#: Package verbs the registry already answers.
PACKAGE_VERBS: tuple[str, ...] = (
    "list", "show", "verify", "smoke", "tools", "add", "import", "remove", "enable", "disable",
)


def slug(text: str, *, words: int = 3) -> str:
    """A name for what was asked for, when the asker did not give one."""
    kept = [part for part in re.findall(r"[a-z0-9]+", text.lower()) if part]
    name = "_".join(kept[:words]) or "new_mode"
    return f"pkg_{name}" if name[0].isdigit() else name[:40]


def _forward(module: str, argv: list[str]) -> int:
    """Run one of the modules that owns a verb, in this process."""
    import importlib

    owner = importlib.import_module(module)
    return int(owner.main(argv))


def _launcher() -> Path:
    """The GUI launcher this checkout ships."""
    return REPO_ROOT / "refactory-dsh"


def _run_argv(identifier: str, args: argparse.Namespace) -> list[str]:
    """One run's arguments, as `refactory_dsh.interactive` spells them.

    `ceo` is the only way to use a mode, so the run options live on this parser and
    are forwarded only when the target actually names a mode. An option nobody gave
    is left out, so the interactive defaults are the ones that apply.
    """
    forwarded = [identifier]
    if args.intent:
        forwarded += ["--intent", str(args.intent)]
    if args.workdir:
        forwarded += ["--workdir", str(args.workdir)]
    if args.source:
        forwarded += ["--source", str(args.source)]
    if args.source_format:
        forwarded += ["--source-format", str(args.source_format)]
    for pair in args.knob:
        forwarded += ["--knob", pair]
    for pair in args.verdict:
        forwarded += ["--approve", pair]
    if args.timeout:
        forwarded += ["--timeout", str(args.timeout)]
    if args.provider:
        forwarded += ["--provider", str(args.provider)]
    if args.model:
        forwarded += ["--model", str(args.model)]
    for key in ("worker_backend", "effort", "permission_mode", "max_elapsed_s", "max_tokens", "validation_reserve_s", "validation_reserve_tokens"):
        value = getattr(args, key, None)
        if value is not None and value != "":
            forwarded += ["--" + key.replace("_", "-"), str(value)]
    if args.headless:
        forwarded += ["--headless"]
    return forwarded


def cmd_ceo(args: argparse.Namespace) -> int:
    """Describe what you want: an authoring mode, or a mode to run.

    A positional that names a mode (`pkg/mode`) *runs* that mode; anything else is
    what you are asking for, and `--mode` says what to do about it. That is the whole
    contract, and it is why the default is `design`: the headline case is "I have an
    idea", not "I have a mode".
    """
    target = str(args.what)
    mode = str(args.mode or "")
    if "/" in target and not mode:
        return _forward("refactory_dsh.interactive", _run_argv(target, args))

    if mode == "" or mode == "design":
        identifier, authoring = AUTHORING["design"], "design"
    elif mode in AUTHORING:
        identifier, authoring = AUTHORING[mode], mode
    elif "/" in mode:
        return _forward("refactory_dsh.interactive", _run_argv(mode, args))
    else:
        print(
            f"refactory: --mode {mode!r} is not one of "
            f"{', '.join(sorted(AUTHORING))} or <package>/<mode>",
            file=sys.stderr,
        )
        return 2

    authoring = authoring.removesuffix("_fast")
    check = str(args.check or "")
    if authoring == "design" and not check:
        print(
            "refactory: design needs --check \"<command that proves it works>\", because a "
            "design nobody can run is not a finished one",
            file=sys.stderr,
        )
        return 2

    if authoring == "edit" and not args.revise:
        print(
            "refactory: --mode edit needs --revise <package>/<mode> — the mode to revise; "
            "without it an \"edit\" is a creation that cannot report what changed",
            file=sys.stderr,
        )
        return 2

    knobs = [f"brief={target}", f"name={args.name or slug(target)}"]
    if authoring == "create" and not args.revise:
        # The mode refuses this at its own first node, which is a profile boot later; a
        # directive nobody gave is worth saying in milliseconds instead.
        from refactory_dsh.create_ops import CREATION_DIRECTIVE

        given = {pair.split("=", 1)[0] for pair in args.knob if "=" in pair}
        if not args.task_ref and not args.single_shot and not ({"task_ref", "single_shot"} & given):
            print(f"refactory: {CREATION_DIRECTIVE}", file=sys.stderr)
            return 2
        if args.task_ref:
            knobs.append(f"task_ref={args.task_ref}")
        if args.single_shot:
            knobs.append("single_shot=1")
    if check:
        knobs.append(f"check={check}")
    if getattr(args, "repair_attempts", None) is not None:
        knobs.append(f"repair_attempts={args.repair_attempts}")
    if getattr(args, "source_project", ""):
        knobs.append(f"source_project={args.source_project}")
    if args.revise:
        knobs.append(f"revision_of={args.revise}")
    # Last, so a person's own setting wins over the one derived from the flags above.
    knobs.extend(str(pair) for pair in args.knob)
    forwarded = [identifier, "--intent", target]
    if args.timeout:
        forwarded += ["--timeout", str(args.timeout)]
    for pair in knobs:
        forwarded += ["--knob", pair]
    if args.provider:
        forwarded += ["--provider", str(args.provider)]
    if args.model:
        forwarded += ["--model", str(args.model)]
    if args.workdir:
        forwarded += ["--workdir", str(args.workdir)]
    # The person answers `gate_strategy` at the terminal. `--approve` is for a run
    # with nobody sitting there, and it is an explicit act rather than a default.
    for pair in args.verdict:
        forwarded += ["--approve", pair]
    for key in ("worker_backend", "effort", "permission_mode", "max_elapsed_s", "max_tokens", "validation_reserve_s", "validation_reserve_tokens"):
        value = getattr(args, key, None)
        if value is not None and value != "":
            forwarded += ["--" + key.replace("_", "-"), str(value)]
    if args.headless:
        forwarded += ["--headless"]
    print(f"refactory: {authoring} — {target}", file=sys.stderr)
    return _forward("refactory_dsh.interactive", forwarded)


def cmd_list(args: argparse.Namespace) -> int:
    """What this checkout can do.

    Readable by default -- a person typed this -- and JSON behind `--json`, which is
    what a script or the CEO gets from the same console.
    """
    from refactory_dsh import registry

    if args.json:
        if args.what == "packages":
            return _forward("refactory_dsh.registry", ["list", "--json"])
        return _forward("refactory_dsh.console", ["modes"])
    print(registry.render_catalogue(REPO_ROOT), end="")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    """The GUI, launched from this checkout (the panel and the tool both live here)."""
    launcher = _launcher()
    if not launcher.exists():
        print(f"refactory: no launcher at {launcher}", file=sys.stderr)
        return 2
    forwarded = [str(launcher)]
    if args.port:
        forwarded += ["--port", str(args.port)]
    if args.no_open:
        forwarded.append("--no-open")
    return subprocess.call(forwarded, cwd=str(REPO_ROOT))


def cmd_doctor(args: argparse.Namespace) -> int:
    """Whether this machine can run a mode, and what to do about anything missing."""
    checks: list[tuple[str, bool | None, str]] = []
    interpreter = Path(sys.executable)
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks.append(("python >= 3.12", sys.version_info >= (3, 12), version))
    checks.append(("uv", shutil.which("uv") is not None, shutil.which("uv") or "install from https://docs.astral.sh/uv"))
    checks.append(("node >= 22.19", _node_ok(), node_bin()))
    for module, why in (("refactory_dsh", "this package"),):
        try:
            __import__(module)
            checks.append((f"import {module}", True, why))
        except Exception as exc:  # the whole point is to report it, not to raise
            checks.append((f"import {module}", False, f"{exc}"))
    try:
        from refactory_dsh import dsh_profile

        # One row per target. `graph` is what a mode runs on; `refactory-web` is
        # what the GUI boots over the harness's own `web` template, whose rows are
        # never written into the harness profile. A target may carry more than one
        # layer, so the row aggregates every gap it still has. The attached Martty
        # target is shown only once the operator has that profile: a profile that
        # does not exist has not chosen a re:factory path and must not read as a
        # broken install.
        gaps = dsh_profile.readiness_gaps()
        for entry in dsh_profile.TARGETS:
            if not entry.owned and not (
                dsh_profile.profile_dir(entry.profile) / "package.json"
            ).exists():
                continue
            problems = [reason for name, _bundle, reason in gaps if name == entry.profile]
            checks.append((
                f"DSH profile {entry.profile!r}",
                not problems,
                " | ".join(problems)
                or f"{dsh_profile.profile_dir(entry.profile)} — {entry.why}",
            ))
        # A harness-owned profile an older install mutated. The migration is
        # stated, not performed: it is the operator's machine state, so the row
        # fails with the exact command and the operator decides when to run it.
        for profile, bundles in dsh_profile.stale_mutations():
            checks.append((
                f"DSH profile {profile!r} (harness-owned)",
                False,
                "carries re:factory layers from the old install; migrate with "
                f"`{dsh_profile.migration_command(profile, bundles)}`",
            ))
    except Exception as exc:
        checks.append(("DSH profiles", False, str(exc)))
    try:
        from refactory_dsh import harness_check

        outcome = harness_check.check()
        if outcome.serving is None:
            checks.append((
                "harness pin vs serving dsh", True,
                f"pinned {outcome.pinned.version or 'unknown version'}; no separate dsh on "
                f"PATH, so the pinned launcher is the serving harness",
            ))
        else:
            checks.append((
                "harness pin vs serving dsh", outcome.ok,
                outcome.reason or (
                    f"pinned and serving both report {outcome.pinned.version} "
                    f"({outcome.serving.launcher})"
                ),
            ))
    except Exception as exc:
        checks.append(("harness pin vs serving dsh", False, str(exc)))
    try:
        from refactory_dsh import registry

        count = len(registry.installed_names(REPO_ROOT))
        checks.append(("packages of modes", count > 0, f"{count} package(s) under {REPO_ROOT}"))
    except Exception as exc:
        checks.append(("packages of modes", False, str(exc)))

    try:
        from tempfile import TemporaryDirectory
        from refactory_dsh.model import DshModelClient
        from refactory_dsh.sandbox import model_readiness

        client = DshModelClient(provider=getattr(args, "provider", ""), model=getattr(args, "model", ""))
        with TemporaryDirectory(prefix="refactory-readiness-") as work:
            readiness = model_readiness(Path(work), model_context=client.context())
        unverified = readiness.get("status") == "diagnostic-unavailable"
        checks.append(("sandbox model configuration", None if unverified else True,
                       f"{readiness['provider']}/{readiness['model']}: {readiness['status']}; authentication not tested"
                       + (f"; {readiness['warning']}" if readiness.get("warning") else "")))
    except Exception as exc:
        checks.append(("sandbox model configuration", False, str(exc)))

    width = max(len(name) for name, _ok, _detail in checks)
    for name, ok, detail in checks:
        marker = "ok  " if ok is True else "MISS" if ok is False else "WARN"
        print(f"  {marker}  {name:<{width}}  {detail}")
    bad = [name for name, ok, _detail in checks if ok is False]
    if bad:
        print(f"\nrefactory doctor: {len(bad)} problem(s): {', '.join(bad)}")
        return 1
    if any(ok is None for _name, ok, _detail in checks):
        print("\nrefactory doctor: local prerequisites ready; model configuration unverified (authentication not tested)")
    else:
        print("\nrefactory doctor: local configuration ready (authentication not tested) — try `refactory list`, or `refactory ceo \"an idea\" --mode design`")
    return 0


def _smoke_without_a_model(identifier: str) -> str:
    """Why `package smoke` cannot check this mode, or the empty string.

    `smoke` runs a mode with no model client, which is the right default: a check
    that quietly made paid calls would be a nasty surprise. A mode whose graph has
    an `agent` node therefore cannot be smoked at all, and the honest answer is to
    say so and name the command that does supply a model, rather than to fail deep
    inside the executor with "agent node 'draft' needs a model client".
    """
    try:
        from refactory_dsh import registry

        mode = registry.resolve(REPO_ROOT, identifier)[1]()
        graph = mode.build()
    except Exception:
        return ""
    agents = [node.id for node in graph.nodes.values() if node.kind == "agent"]
    if not agents:
        return ""
    return (
        f"refactory: {identifier} calls a model ({len(agents)} agent node(s): "
        f"{', '.join(agents)}), and `package smoke` supplies none by default.\n"
        f"           add --model to score it, or use `refactory ceo {identifier}` to run it on DSH."
    )


def _node_ok() -> bool:
    """Whether the Node this repository needs is runnable."""
    try:
        out = subprocess.run([str(node_bin()), "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    match = re.search(r"v(\d+)\.(\d+)", out.stdout)
    if not match:
        return False
    major, minor = int(match.group(1)), int(match.group(2))
    return major > 22 or (major == 22 and minor >= 19)


def build_parser() -> argparse.ArgumentParser:
    """The command line, with one verb per question a person has."""
    parser = argparse.ArgumentParser(
        prog="refactory",
        description="Create a mode, use it, optimize it, use it again.",
        epilog=(
            "Start with `refactory doctor`, then `refactory list`, then "
            "`refactory ceo \"an idea\" --mode design`."
        ),
    )
    sub = parser.add_subparsers(dest="verb", required=True)

    ceo = sub.add_parser(
        "ceo",
        help="describe what you want: the CEO designs or builds it (or run the mode you name)",
        description=(
            "Give it an idea and a mode, or name a mode to run. An authoring mode runs on "
            "DSH with your terminal attached, so the plan-approval gate reaches you."
        ),
    )
    ceo.add_argument("what", help="what you want, or <package>/<mode> to run")
    ceo.add_argument(
        "--mode", default="",
        help="design (the default), create, edit, their _fast variants, or <package>/<mode>",
    )
    ceo.add_argument("--name", default="", help="the package or artifact name (default: from what you said)")
    ceo.add_argument("--check", default="", help="required for design; optional focused check for create_fast/edit_fast")
    ceo.add_argument("--revise", default="", metavar="PKG/MODE", help="edit only: the mode to revise")
    ceo.add_argument(
        "--task", default="", metavar="TASK_REF", dest="task_ref",
        help="create only: the class of work, as a Task the produced harness binds a data "
             "node to (module:Class, or a registered name); the harness then runs its "
             "subgraph once per item the Task lists",
    )
    ceo.add_argument(
        "--single", action="store_true", default=False, dest="single_shot",
        help="create only: build a harness for one run of one thing, with no Task "
             "binding; one of --task or --single is required",
    )
    ceo.add_argument(
        "--knob", action="append", default=[], metavar="NAME=VALUE",
        help="set one of the authoring mode's knobs for this run, e.g. allow_regression=1",
    )
    ceo.add_argument("--workdir", default="", help="the run directory")
    ceo.add_argument("--repair-attempts", type=int, default=None)
    ceo.add_argument("--source-project", default="")
    for key in ("worker_backend", "effort", "permission_mode"):
        ceo.add_argument("--" + key.replace("_", "-"), default="")
    for key in ("max_elapsed_s", "max_tokens", "validation_reserve_s", "validation_reserve_tokens"):
        ceo.add_argument("--" + key.replace("_", "-"), type=int if "tokens" in key else float, default=None)
    ceo.add_argument(
        "--provider", default="",
        help="provider route for this run's model (default: the settings default)",
    )
    ceo.add_argument(
        "--model", default="",
        help="model id for this run, e.g. deepseek-v4-pro (default: the settings default)",
    )
    ceo.add_argument(
        "--approve", dest="verdict", action="append", default=[], metavar="GATE=VALUE",
        help="answer a gate up front, e.g. gate_strategy=proceed, for a run with nobody "
             "sitting at it; a gate a person would be asked at is theirs, so run without "
             "this and answer it, or drive the mode node by node with wf_open",
    )
    ceo.add_argument(
        "--intent", default="",
        help="run a mode: what this run is for, in your words; authored runs take the "
             "brief itself as the intent",
    )
    ceo.add_argument(
        "--source", default="", metavar="PATH",
        help="run a mode: read the items from here instead of the source the mode "
             "declares, so a harness built for a class of work is pointed at items it "
             "has not seen",
    )
    ceo.add_argument(
        "--source-format", default="", choices=["", "directory", "jsonl", "csv"],
        help="run a mode: how to read --source (default: a directory of one item per "
             "subdirectory)",
    )
    ceo.add_argument(
        "--timeout", type=int, default=0,
        help="run a mode: seconds to wait at a gate; 0 waits for you",
    )
    ceo.add_argument(
        "--headless", action="store_true", default=False,
        help="run without a conversation: one task, one report, then exit, for scripts "
             "and CI; without it the run is live and a line you type between turns "
             "reaches the CEO",
    )

    optimize = sub.add_parser("optimize", help="search a mode's architecture and knobs")
    optimize.add_argument("mode", help="the mode, as <package>/<name>")
    from refactory_dsh.profiles import add_arguments
    add_arguments(optimize)
    optimize.add_argument("--runs-root", default=None)
    optimize.add_argument("--resume", action="store_true")
    optimize.add_argument(
        "--verbose", action="store_true",
        help="print every event, and each candidate's own run output, not only the "
             "meaningful ones",
    )

    listing = sub.add_parser("list", help="what this checkout can do")
    listing.add_argument("what", nargs="?", default="modes", choices=["modes", "packages"])
    listing.add_argument("--json", action="store_true", help="the console's JSON instead of the listing")

    for verb in READ_VERBS:
        # The console owns these; `refactory status --help` shows the console's help.
        sub.add_parser(verb, help=f"`refactory {verb}` — the console's own verb, arguments pass through")

    package = sub.add_parser("package", help="manage packages of modes")
    package.add_argument(
        "what", nargs="?", choices=PACKAGE_VERBS,
        help="the registry verb; its own arguments pass through",
    )

    ui = sub.add_parser("ui", help="the Web GUI (the tool, the panel, the conversation)")
    ui.add_argument("--port", type=int, default=0, help="default: the launcher's own")
    ui.add_argument("--no-open", action="store_true", help="do not open a browser")

    doctor = sub.add_parser("doctor", help="check local prerequisites and sandbox credential configuration (no model request)")
    doctor.add_argument("--provider", default="", help="provider route to check (default: DSH settings)")
    doctor.add_argument("--model", default="", help="model selection to report (default: DSH settings)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch one verb onto the module that owns it."""
    argv = list(sys.argv[1:] if argv is None else argv)

    # The run-reading verbs and the package verbs take the console's and the
    # registry's own arguments verbatim -- including flags -- so they are forwarded
    # before argparse, which would otherwise try to claim `--run` for itself.
    if argv and argv[0] in READ_VERBS:
        return _forward("refactory_dsh.console", argv)
    if argv and argv[0] == "package" and len(argv) >= 2:
        if argv[1] == "smoke" and len(argv) >= 3 and "--model" not in argv:
            problem = _smoke_without_a_model(argv[2])
            if problem:
                print(problem, file=sys.stderr)
                return 2
        return _forward("refactory_dsh.registry", argv[1:])

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verb == "ceo":
        return cmd_ceo(args)
    if args.verb == "optimize":
        forwarded = [args.mode]
        for flag, value in vars(args).items():
            if flag in ("verb", "mode") or value is None:
                continue
            name = f"--{flag.replace('_', '-')}"
            # A switch forwards as itself, and only when it is on; a value-taking
            # option forwards its value.
            if isinstance(value, bool):
                if value:
                    forwarded.append(name)
            elif isinstance(value, list):
                for element in value:
                    forwarded.extend([name, str(element)])
            elif isinstance(value, dict):
                forwarded.extend([name, json.dumps(value)])
            else:
                forwarded.extend(["--preset" if flag == "edit_preset" else name, str(value)])
        return _forward("refactory_dsh.loop", forwarded)
    if args.verb == "list":
        return cmd_list(args)
    if args.verb == "ui":
        return cmd_ui(args)
    if args.verb == "doctor":
        return cmd_doctor(args)
    if args.verb == "package":  # handled before argparse; kept for completeness
        return _forward("refactory_dsh.registry", [])
    parser.error(f"unknown verb {args.verb!r}")  # pragma: no cover - argparse rejects first
    return 2


if __name__ == "__main__":  # pragma: no cover - the entry point
    raise SystemExit(main())
