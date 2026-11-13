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
    refactory model set|show                  the one model every run uses

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
# One spelling of a run's flags for every command that starts one (dm-134); kept here by
# name for the callers that read them from this module.
from refactory_dsh.run_flags import DELIVERY_FLAGS, slug  # noqa: F401

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
    from refactory_dsh import run_flags

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
    return forwarded + run_flags.interactive_options(args)


def _worktree_knobs(args: argparse.Namespace, *, deliver: bool = True) -> list[str]:
    """`--target`, `--base`, `--no-worktree`, `--setup`, `--generate` and, with ``deliver``, delivery's, as knobs."""
    from refactory_dsh.run_flags import worktree_knobs

    return worktree_knobs(args, deliver=deliver)


def _names_a_mode(text: str) -> bool:
    """Whether `text` is a well-formed `<package>/<mode>` identifier.

    The shape decides, not the registry. A brief that mentions a path has a slash too,
    and a mistyped mode name should be refused by the runner as an unknown package,
    not taken as a one-word brief.
    """
    from refactory_dsh.nn.package import PackageError, split

    try:
        split(text)
    except PackageError:
        return False
    return True


def cmd_ceo(args: argparse.Namespace) -> int:
    """Describe what you want: an authoring mode, or a mode to run.

    A positional that is a mode identifier (`pkg/mode`) *runs* that mode; anything
    else, a brief that mentions a path included, is what you are asking for, and
    `--mode` says what to do about it. That is the whole contract, and it is why the
    default is `design`: the headline case is "I have an idea", not "I have a mode".
    """
    from refactory_dsh import run_flags

    target = str(args.what)
    mode = str(args.mode or "")
    runs_a_mode = "/" in mode if mode else _names_a_mode(target)
    identifier, authoring = "", ""
    if not runs_a_mode:
        if mode == "" or mode == "design":
            identifier, authoring = AUTHORING["design"], "design"
        elif mode in AUTHORING:
            identifier, authoring = AUTHORING[mode], mode
        else:
            print(
                f"refactory: --mode {mode!r} is not one of "
                f"{', '.join(sorted(AUTHORING))} or <package>/<mode>",
                file=sys.stderr,
            )
            return 2
    # A flag a run would ignore is refused, not dropped: the person meant it. The walker's
    # mains refuse the same flags with the same messages (`run_flags`, dm-134).
    problem = run_flags.refusal(args, identifier or None, runtime=run_flags.SPINE,
                                label=f"--mode {mode or 'design'}")
    if problem:
        print(problem, file=sys.stderr)
        return 2
    if runs_a_mode:
        # A named mode takes them as its knobs; one that does not declare them refuses them by name.
        args.knob = run_flags.knobs(args, None, name=args.name, revision=args.revise)
        return _forward("refactory_dsh.interactive", _run_argv(mode or target, args))

    from refactory_dsh.brief_guard import require_brief
    try:
        require_brief(target, source="what")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
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

    if authoring == "create" and not args.revise:
        # The mode refuses this at its own first node, which is a profile boot later; a
        # directive nobody gave is worth saying in milliseconds instead.
        from refactory_dsh.create_ops import CREATION_DIRECTIVE

        given = {pair.split("=", 1)[0] for pair in args.knob if "=" in pair}
        if not args.task_ref and not args.single_shot and not ({"task_ref", "single_shot"} & given):
            print(f"refactory: {CREATION_DIRECTIVE}", file=sys.stderr)
            return 2
    knobs = run_flags.knobs(args, identifier, brief=target, name=args.name or slug(target), revision=args.revise)
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
    forwarded += run_flags.interactive_options(args)
    first = target.split('\n', 1)[0]
    short = first[:80] + '…' if len(first) > 80 else first
    print(f"refactory: {authoring} — {short}", file=sys.stderr)
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
    # What pyproject.toml declares (`requires-python`), so doctor and the installer agree.
    checks.append(("python >= 3.11", sys.version_info >= (3, 11), version))
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

        # One row per target. `graph` is what a mode runs on; `refactory` is
        # what the GUI boots over the harness's own `web` template, whose rows are
        # never written into the harness profile. A target may carry more than one
        # layer, so the row aggregates every gap it still has. The attached Martty
        # target is shown only once the operator has that profile: a profile that
        # does not exist has not chosen a re:factory path and must not read as a
        # broken install.
        session_home = dsh_profile.dsh_home()
        worker_home = dsh_profile.model_home()
        checks.append(("DSH installation/session home", True, str(session_home)))
        checks.append(("DSH model home", True, str(worker_home)))
        if worker_home != session_home:
            problem = dsh_profile.requirement(home=worker_home)
            checks.append(("DSH model profile 'graph'", problem is None,
                           problem or str(dsh_profile.profile_dir(home=worker_home))))
        gaps = dsh_profile.readiness_gaps()
        for entry in dsh_profile.TARGETS:
            if not entry.owned and not (
                dsh_profile.profile_dir(entry.profile) / "package.json"
            ).exists():
                continue
            problems = [reason for name, _bundle, reason in gaps if name == entry.profile]
            # Show which checkout the graph bundle comes from on the ok row too.
            ok_detail = f"{dsh_profile.profile_dir(entry.profile)} — {entry.why}"
            if entry.owned and not problems:
                for key in entry.layers:
                    layer = dsh_profile.LAYERS[key]
                    resolved = dsh_profile.bundle_checkout_root(entry.profile, layer.bundle)
                    if resolved is not None:
                        ok_detail += f" (bundle: {resolved})"
                        break
            checks.append((
                f"DSH profile {entry.profile!r}",
                not problems,
                " | ".join(problems) or ok_detail,
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
    if getattr(args, "run", None):
        try:
            run_path = Path(args.run).expanduser().resolve()
            rc_path = run_path / ".refactory" / "run-config.json"
            if not rc_path.exists():
                checks.append(("run driver", False, f"no run-config.json at {rc_path}"))
            else:
                rc_data = json.loads(rc_path.read_text())
                drv = rc_data.get("driver", {})
                checks.append(("run driver commit", True, drv.get("commit", "unknown")[:12]))
                checks.append(("run driver profile", True, drv.get("profile", "graph")))
                pin_co = drv.get("pin_checkout")
                checks.append(("run pinned", True if pin_co else None,
                                str(pin_co or "no pin — driver has not moved")))
                if pin_co and not Path(pin_co).is_dir():
                    checks.append(("pin checkout exists", False, f"{pin_co} is missing"))
                elif pin_co:
                    checks.append(("pin checkout exists", True, str(pin_co)))
        except Exception as exc:
            checks.append(("run driver", False, str(exc)))
    try:
        from refactory_dsh import registry

        count = len(registry.installed_names(REPO_ROOT))
        checks.append(("packages of modes", count > 0, f"{count} package(s) under {REPO_ROOT}"))
    except Exception as exc:
        checks.append(("packages of modes", False, str(exc)))

    # The Claude Code a claude_code worker runs: the one the route (a flag, the project
    # file) or REFACTORY_CLAUDE_EXECUTABLE chooses, else the one the Agent SDK bundles.
    try:
        from refactory_dsh import worker_config, workers
        from refactory_dsh.paths import dsh_root, project_root

        effective = worker_config.effective(project_root())
        chosen = effective["settings"].get("claude_executable", "")
        origin = effective["provenance"]["fields"].get("claude_executable", "")
        if not chosen and os.environ.get(workers.CLAUDE_EXECUTABLE_ENV):
            chosen, origin = workers.claude_executable_choice({}), workers.CLAUDE_EXECUTABLE_ENV
        if chosen:
            runs = os.path.isfile(chosen) and os.access(chosen, os.X_OK)
            found = workers.claude_code_version(chosen) if runs else ""
            checks.append(("claude_code executable", bool(found), f"{chosen} (from {origin}): " + (
                f"Claude Code {found}" if found else
                "reports no version" if runs else "missing or not executable")))
        elif (bundled := workers.bundled_claude_code(dsh_root())) is None:
            checks.append(("claude_code executable", None,
                           f"{workers.PACKAGES['claude_code']} is not installed (only claude_code workers need it)"))
        else:
            checks.append(("claude_code executable", True if bundled["path"] else None,
                           (f"{bundled['path'] or 'no platform binary found'} (bundled with "
                            f"{workers.CLAUDE_AGENT_SDK} {bundled['sdk_version']}): Claude Code "
                            f"{bundled['version'] or 'unknown'}")))
    except Exception as exc:
        checks.append(("claude_code executable", False, str(exc)))

    # The run model (dm-258): the route a run here would get — a flag, the environment,
    # the project file or the one stored choice, resolved exactly as a run resolves it
    # (`refactory runtime` shows the same). Its provider's key is the one checked, never
    # the headless profile's default.
    route: dict[str, str] = {}
    try:
        from refactory_dsh import dsh_profile
        from refactory_dsh.console import model_route
        from refactory_dsh.model import half_route_refusal
        from refactory_dsh.paths import project_root

        resolved = model_route(project_root(), provider=str(getattr(args, "provider", "") or ""),
                               model=str(getattr(args, "model", "") or ""))
        route = {"provider": resolved["provider"], "model": resolved["model"]}
        if resolved["worker_backend"] != "dsh":
            checks.append(("run model", None, f"{resolved['worker_backend']} workers use their own model settings"))
            route = {}
        elif route["provider"] and route["model"]:
            checks.append(("run model", True, f"{route['provider']}/{route['model']} ({resolved['source']['provider']})"))
        elif half_route_refusal(route["provider"], route["model"]):
            checks.append(("run model", False, half_route_refusal(route["provider"], route["model"])))
            route = {}
        else:
            keyed, refusal = dsh_profile.first_run_plan()
            if refusal or not keyed:
                checks.append(("run model", False, refusal))
                route = {}
            else:
                route = keyed[0]
                checks.append(("run model", None,
                               f"none chosen yet; the first run will use {route['provider']}/{route['model']}, "
                               "the only provider with a key (`refactory model set` chooses another)"))
    except Exception as exc:
        checks.append(("run model", False, str(exc)))
        route = {}

    if route.get("provider") and route.get("model"):
        try:
            from tempfile import TemporaryDirectory
            from refactory_dsh.model import DshModelClient
            from refactory_dsh.sandbox import model_readiness

            client = DshModelClient(provider=route["provider"], model=route["model"])
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
    # Every other flag is a run's, spelled as the walker's mains spell it (dm-134).
    from refactory_dsh import run_flags
    run_flags.add_arguments(ceo)

    optimize = sub.add_parser("optimize", help="search a mode's architecture and knobs")
    optimize.add_argument("mode", help="the mode, as <package>/<name>")
    # A short list; every other setting is a profile's (`refactory profile-save`).
    from refactory_dsh.profiles import add_cli_arguments
    add_cli_arguments(optimize)

    listing = sub.add_parser("list", help="what this checkout can do")
    listing.add_argument("what", nargs="?", default="modes", choices=["modes", "packages"])
    listing.add_argument("--json", action="store_true", help="the console's JSON instead of the listing")

    for verb in READ_VERBS:
        # The console owns these; `refactory status --help` shows the console's help.
        sub.add_parser(verb, help=f"`refactory {verb}` — the console's own verb, arguments pass through")

    projects = sub.add_parser("projects", help="list the projects design runs work on; `projects prune` cleans up")
    projects.add_argument("what", nargs="?", choices=("list", "prune"), help="its own arguments pass through")

    package = sub.add_parser("package", help="manage packages of modes")
    package.add_argument(
        "what", nargs="?", choices=PACKAGE_VERBS,
        help="the registry verb; its own arguments pass through",
    )

    # ── the run model ──
    model_group = sub.add_parser(
        "model", help="the model every run uses: one choice, shared with Settings → Models",
        description="Every run uses one model choice, stored where Settings → Models writes it. "
                    "--provider/--model on a command override it for that run only.",
    )
    model_sub = model_group.add_subparsers(dest="model_action", required=True)
    model_set = model_sub.add_parser("set", help="store the model every run uses")
    model_set.add_argument("--provider", required=True, help="the provider route, e.g. deepseek-official")
    model_set.add_argument("--model", required=True, help="the model id, e.g. deepseek-v4-pro")
    model_sub.add_parser("show", help="print the stored model choice as JSON")

    ui = sub.add_parser("ui", help="the Web GUI (the tool, the panel, the conversation)")
    ui.add_argument("--port", type=int, default=0, help="default: the launcher's own")
    ui.add_argument("--no-open", action="store_true", help="do not open a browser")

    doctor = sub.add_parser("doctor", help="check local prerequisites and sandbox credential configuration (no model request)")
    doctor.add_argument("--provider", default="", help="provider route to check (default: the stored model choice)")
    doctor.add_argument("--model", default="", help="model to check (default: the stored model choice)")
    doctor.add_argument("--run", type=str, default=None, metavar="DIR",
                        help="check a specific run's driver pin state")

    # ── pin management ──
    pin_group = sub.add_parser("pin", help="manage driver pins")
    pin_sub = pin_group.add_subparsers(dest="pin_action")
    pin_sub.add_parser("list", help="list all pins with commit, creation time and driver checkout")
    pin_sub.add_parser("prune", help="remove pins no run refers to")

    # ── replay verifier ──
    replay = sub.add_parser(
        "replay-verifier",
        help="replay the verifier on an existing run directory and a commit",
        description="Replay the verifier on an existing run directory. "
                    "When --commit is given, a temporary worktree of that commit "
                    "is checked out, the verifier runs against it, and the "
                    "worktree is removed.",
    )
    replay.add_argument("--run-dir", required=True, help="path to the existing run directory")
    replay.add_argument("--commit", default="", help="commit hash to check out as a temporary worktree")
    replay.add_argument("--provider", default="",
                        help="the verifier's provider route (default: as `refactory ceo` resolves it)")
    replay.add_argument("--model", default="", help="the verifier's model id")
    replay.add_argument("--effort", default="", help="the verifier's reasoning effort")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch one verb onto the module that owns it."""
    argv = list(sys.argv[1:] if argv is None else argv)

    # The run-reading verbs and the package verbs take the console's and the
    # registry's own arguments verbatim -- including flags -- so they are forwarded
    # before argparse, which would otherwise try to claim `--run` for itself.
    READABLE_VERBS = {'status', 'steer', 'graph-control'}
    if argv and argv[0] in READ_VERBS:
        verb = argv[0]
        if verb in READABLE_VERBS and '--json' in argv:
            forward_argv = [arg for arg in argv if arg != '--json']
            return _forward('refactory_dsh.console', forward_argv)
        if verb in READABLE_VERBS and '--help' not in argv:
            import contextlib
            import io as _io
            buf = _io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = _forward('refactory_dsh.console', argv)
            raw = buf.getvalue()
            if code != 0:
                try:
                    error_payload = json.loads(raw)
                    error_msg = error_payload.get('error', '') if isinstance(error_payload, dict) else ''
                except (json.JSONDecodeError, AttributeError, TypeError):
                    error_msg = ''
                if error_msg:
                    print(f'refactory: {error_msg}', file=sys.stderr)
                else:
                    print(f'refactory: {raw}', file=sys.stderr, end='')
                return code
            try:
                payload = json.loads(raw)
                text = payload.get('text', '') if isinstance(payload, dict) else ''
            except (json.JSONDecodeError, AttributeError, TypeError):
                text = ''
            if text:
                print(text)
            else:
                print(raw, end='')
            return code
        return _forward('refactory_dsh.console', argv)
    if argv and argv[0] == "projects":
        return _forward("refactory_dsh.projects", argv[1:])
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
            else:
                forwarded.extend(["--preset" if flag == "edit_preset" else name, str(value)])
        return _forward("refactory_dsh.loop", forwarded)
    if args.verb == "list":
        return cmd_list(args)
    if args.verb == "ui":
        return cmd_ui(args)
    if args.verb == "doctor":
        return cmd_doctor(args)
    if args.verb == "model":
        return _cmd_model(args)
    if args.verb == "pin":
        return _cmd_pin(args)
    if args.verb == "replay-verifier":
        return _cmd_replay_verifier(args)
    if args.verb == "package":  # handled before argparse; kept for completeness
        return _forward("refactory_dsh.registry", [])
    parser.error(f"unknown verb {args.verb!r}")  # pragma: no cover - argparse rejects first
    return 2


def _cmd_model(args: argparse.Namespace) -> int:
    """Store or show the one model choice (dm-258)."""
    from refactory_dsh import dsh_profile

    path = dsh_profile.model_choice_path()
    if args.model_action == "set":
        try:
            choice = dsh_profile.model_choice_write(args.provider, args.model)
        except (ValueError, OSError) as exc:
            print(f"refactory: {exc}", file=sys.stderr)
            return 1
        print(f"refactory: runs now use {choice['provider']}/{choice['model']} (stored in {path})")
        return 0
    choice = dsh_profile.stored_model_choice()
    if not choice["provider"]:
        print(f"refactory: {dsh_profile.NO_MODEL_CHOICE_MESSAGE}", file=sys.stderr)
        return 1
    print(json.dumps({**choice, "stored_in": str(path)}, indent=2))
    return 0


def _cmd_pin(args: argparse.Namespace) -> int:
    """Manage driver pins: list and prune."""
    from refactory_dsh import pin

    action = getattr(args, "pin_action", None)
    if action == "list":
        pins = pin.list_pins()
        if not pins:
            print("no pins")
            return 0
        for entry in pins:
            commit = entry.get("commit", "?")[:12]
            created = entry.get("created", "?")
            driver = entry.get("driver_checkout", "?")
            runs = entry.get("runs", [])
            print(f"  {commit}  created {created}  driver {driver}  runs {len(runs)}")
        return 0
    if action == "prune":
        pruned = pin.prune_pins()
        if pruned:
            for sha in pruned:
                print(f"  pruned {sha[:12]}")
        else:
            print("nothing to prune")
        return 0
    print("usage: refactory pin {list,prune}", file=sys.stderr)
    return 2


def _read_facts(run_dir: Path) -> dict:
    """The run's facts (its check command among them), or ``{}`` when none are readable."""
    try:
        found = json.loads((run_dir / "facts.json").read_text())
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def _replay_probe_results(run_dir: Path, repo_root: Path, report: str, commit: str | None) -> str:
    """Rerun the probe files the replayed verifier wrote, in a scratch clone at the commit.

    The probes move from the verifier's checkout to ``run_dir/reviews/probes``; the table
    goes to ``run_dir/reviews/probe-results.md`` and is returned.
    """
    import shutil
    import tempfile

    from refactory_dsh import clones, probe_rerun, worktree

    written = repo_root / probe_rerun.PROBES
    probes = run_dir / probe_rerun.PROBES
    shutil.rmtree(probes, ignore_errors=True)
    if written.is_dir():
        shutil.copytree(written, probes)
        shutil.rmtree(written, ignore_errors=True)
        if not any(written.parent.iterdir()):
            written.parent.rmdir()
    try:
        rev = commit or worktree.git(repo_root, "rev-parse", "HEAD").strip()
        found = worktree.recorded(run_dir)
        base = found.base if found is not None else f"{rev}~1"
        with tempfile.TemporaryDirectory(prefix="refactory-probe-rerun-") as scratch:
            checkout = Path(scratch) / "repo"
            clones.clone_shared(repo_root, checkout, rev)
            # The check's interpreter: the clone has no environment of its own, so it is
            # looked up in the verifier's checkout and then the repository's main checkout.
            facts = _read_facts(run_dir)
            roots = [repo_root]
            try:
                common = worktree.git(repo_root, "rev-parse", "--path-format=absolute",
                                      "--git-common-dir").strip()
                roots.append(Path(common).parent)
            except (ValueError, OSError):
                pass
            results = probe_rerun.run_probes(probes, checkout, report, base=base,
                                             reset=probe_rerun.reset_checkout,
                                             check=str(facts.get("check") or ""), roots=roots)
    except (ValueError, OSError) as exc:  # WorkspaceError is a ValueError
        return f"# Probe results\n\nThe probes could not be rerun: {exc}\n"
    return probe_rerun.write_results(run_dir, results)


def _replay_verifier_checkout(run_dir: Path, commit: str | None) -> tuple[Path, Path]:
    """A throwaway worktree for a replayed verifier: ``(checkout, repository to remove it from)``.

    Always a new temporary worktree, never the run's own: the verifier reverts hunks with
    ``git apply --reverse`` and restores files with ``git checkout HEAD -- <file>``, which
    in the run's worktree would discard a builder's uncommitted work and could leave hunks
    reverted if it died midway -- as the graph's verifier gets a clone for. The commit is
    ``--commit``, else the head the run recorded. Raises ``ValueError`` when there is no
    repository or no commit to check out.
    """
    import subprocess
    import tempfile

    source: Path | None = None
    head: str | None = None
    record = run_dir / ".refactory" / "worktree.json"
    if record.is_file():
        try:
            data = json.loads(record.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"{record} cannot be read: {exc}") from exc
        if data.get("path") and Path(data["path"]).is_dir():
            source = Path(data["path"])
        head = data.get("head") or None
    if source is None:
        found = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True,
                               text=True, timeout=10, cwd=str(Path.cwd()))
        if found.returncode != 0:
            raise ValueError("no git repository: run from the target, or replay a run "
                             "that recorded its worktree")
        source = Path(found.stdout.strip())
    if not commit and not head:
        # No head recorded: the commit the run's worktree is at now.
        current = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                                 capture_output=True, text=True, timeout=10)
        head = current.stdout.strip() if current.returncode == 0 else None
    target = commit or head
    if not target:
        raise ValueError("no commit to check out: give --commit")
    checkout = Path(tempfile.mkdtemp(prefix="refactory-replay-verifier-")) / "worktree"
    subprocess.run(["git", "-C", str(source), "worktree", "add", "--detach", str(checkout), target],
                   check=True, capture_output=True, timeout=60)
    return checkout, source


def _cmd_replay_verifier(args: argparse.Namespace) -> int:
    """Replay the verifier on an existing run directory and a commit.

    The verifier runs in a temporary worktree of ``--commit`` (else the run's recorded
    head), with the same prompt and workspace setup as the graph's verifier node; the
    report goes to stdout and the worktree is removed. The run's own worktree is never
    touched (`_replay_verifier_checkout`).
    """
    import subprocess
    import tempfile

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"refactory: run directory not found: {run_dir}", file=sys.stderr)
        return 1

    brief_path = run_dir / "brief.json"
    if not brief_path.is_file():
        print(f"refactory: brief.json not found in {run_dir}", file=sys.stderr)
        return 1

    commit = args.commit
    temp_worktree = None
    git_source = None

    try:
        try:
            temp_worktree, git_source = _replay_verifier_checkout(run_dir, commit)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            print(f"refactory: {exc}", file=sys.stderr)
            return 1
        repo_root = temp_worktree

        # ── read brief and build context for the verifier prompt ──
        brief_data = json.loads(brief_path.read_text())
        brief_text = brief_data.get("brief", "")

        # Read the diff
        diff_path = run_dir / "changes.diff"
        diff_text = diff_path.read_text() if diff_path.is_file() else ""

        # Read corrections from the intent ledger
        corrections = ""
        ledger_path = run_dir / "intent-ledger.md"
        if ledger_path.is_file():
            from refactory_dsh.design_ops import gate_corrections_due
            _approval, due = gate_corrections_due(run_dir)
            if due:
                corrections = "\n\n".join(f"- {text}" for _, text in due)

        # Build the requirements_and_corrections fact
        parts: list[str] = []
        if brief_text:
            parts.append(f"## Requirements from the brief\n{brief_text}")
        if corrections:
            parts.append(f"## Gate corrections\n{corrections}")
        req_and_corr = "\n\n".join(parts) if parts else "(no requirements)"

        # Build the verification methods fact
        from refactory_dsh.verification import (
            DESIGN_VERIFIER_PROMPT, DESIGN_VERIFIER_TIMEOUT_S,
            verification_methods,
        )
        methods = verification_methods("repository_change")

        # Fill the template variables in the prompt
        import re as _re

        values = {
            "artifact": str(repo_root),
            "change": f"The diff is in changes.diff ({len(diff_text)} bytes).",
            "brief": brief_text,
            "requirements_and_corrections": req_and_corr,
            "verification_methods": methods,
        }

        def _fill(match: _re.Match[str]) -> str:
            key = match.group(1).strip()
            return str(values[key]) if key in values else match.group(0)

        prompt = _re.sub(
            r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}", _fill, DESIGN_VERIFIER_PROMPT,
        )

        # ── invoke the verifier's model session ──
        from refactory_dsh.model import DshModelClient
        from refactory_dsh.nn import RunContext

        # The route as a design run settles it: flags over the environment, the project
        # file and the stored choice, then completed or refused (`require_route`).
        from refactory_dsh.interactive import require_route
        from refactory_dsh.model import ModelChoiceError

        route = {key: value for key, value in (
            ("provider", args.provider), ("model", args.model), ("effort", args.effort)) if value}
        try:
            context = require_route(DshModelClient(
                timeout_s=DESIGN_VERIFIER_TIMEOUT_S, project=repo_root, **route,
            ).context())
        except ModelChoiceError as exc:
            print(f"refactory: {exc}", file=sys.stderr)
            return 1
        client = DshModelClient(**context)

        # Create a RunContext with the run directory as the work directory
        ctx = RunContext(run_dir)

        # Copy the reads the verifier expects into the cwd if not present
        for read_file in ("brief.json", "changes.diff"):
            src = run_dir / read_file
            dst = repo_root / read_file
            if src.is_file() and not dst.exists():
                import shutil as _shutil
                _shutil.copy2(str(src), str(dst))

        print(f"Replaying verifier on {run_dir} (worktree: {repo_root})...",
              file=sys.stderr)
        print(f"Timeout: {DESIGN_VERIFIER_TIMEOUT_S}s", file=sys.stderr)

        output = client.complete(prompt, workdir=run_dir, cwd=repo_root)

        # Write reviews/verification.md into the run directory
        report_path = run_dir / "reviews" / "verification.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(output)

        # Print the report to stdout
        print(output)

        # ── rerun the verifier's probe files, as design mode's rerun_probes op does ──
        print(_replay_probe_results(run_dir, repo_root, output, commit))
        return 0

    finally:
        if temp_worktree is not None and git_source is not None:
            # Removed from the repository it was added to, which is not necessarily
            # the one the command was started in.
            try:
                subprocess.run(["git", "-C", str(git_source), "worktree", "remove", "--force",
                                str(temp_worktree)], capture_output=True, timeout=30)
            except (OSError, subprocess.SubprocessError):
                pass
            import shutil as _shutil

            _shutil.rmtree(temp_worktree.parent, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover - the entry point
    raise SystemExit(main())
