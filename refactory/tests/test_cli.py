"""The one command: `refactory`.

These hold the entry point's own behaviour — what each verb reaches, and the two
refusals that exist to stop a person being confused by a low-level error.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import contextlib
import io as _io

from refactory_dsh import cli, dsh_profile, nn
from refactory_dsh import console as _console_mod
from refactory_dsh.brief_guard import require_brief, check_required_knobs
from refactory_dsh.nn.knob import Knob, KnobError
from refactory_dsh.nn.validate import validate_state, Issue
from refactory_dsh.readable import format_status
from refactory_dsh.paths import REPO_ROOT
from tests_support import PLACEHOLDER_KEY, _link_bundle, _profile, model_home_with_a_key


def _run(*argv: str, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    """The console script, as a person runs it."""
    return subprocess.run(
        [str(REPO_ROOT / ".venv" / "bin" / "refactory"), *argv],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout,
    )


def test_doctor_reports_every_prerequisite():
    """One command that answers "is this machine ready", and says what is missing.

    The verdict is a fact about the machine, not the code: a developer's DSH home
    can hold a stale attached profile or a different serving `dsh`, and doctor
    reports those rather than swallowing them. So this pins the prerequisite rows
    and that a verdict is stated -- CI, on a clean machine, checks the ready path.
    """
    result = _run("doctor")
    assert result.returncode in (0, 1), result.stdout + result.stderr
    for wanted in (
        "python", "uv", "node", "DSH profile 'graph'",
        "packages of modes", "sandbox model configuration",
    ):
        assert wanted in result.stdout, f"doctor does not check {wanted!r}"
    verdict = result.stdout.splitlines()[-1]
    assert verdict.startswith("refactory doctor: "), verdict
    assert "ready" in verdict or "problem(s)" in verdict, verdict


def test_list_is_readable_by_a_person_and_json_to_a_script():
    """The default is for a human; `--json` is the console's own answer."""
    readable = _run("list")
    assert readable.returncode == 0
    assert "circle_packing/" in readable.stdout
    assert not readable.stdout.lstrip().startswith("["), "the listing is a JSON dump again"

    machine = _run("list", "modes", "--json")
    assert machine.returncode == 0
    entries = json.loads(machine.stdout)
    assert any(entry["id"] == "circle_packing/adaptive" for entry in entries)
    assert all("action" in entry for entry in entries)


def test_smoke_refuses_a_mode_that_calls_a_model_naming_the_command_that_works():
    """`package smoke` runs without a model on purpose, so an agent mode cannot be
    smoked — and the failure says which command does supply one, instead of failing
    inside the executor with "agent node 'draft' needs a model client"."""
    problem = cli._smoke_without_a_model("default_pack/create")
    assert "calls a model" in problem
    assert "refactory ceo default_pack/create" in problem
    assert cli._smoke_without_a_model("circle_packing/adaptive") == ""
    # An identifier nobody can resolve is the registry's error to give, not ours.
    assert cli._smoke_without_a_model("not/a_mode") == ""


def test_status_shows_a_run_that_stopped_short_of_its_end(tmp_path):
    """The spine records an early end as halted, naming what the run never reached."""
    run_dir = tmp_path / "incomplete_run"
    ref = run_dir / ".refactory"
    ref.mkdir(parents=True)
    (ref / "spine-status.json").write_text(json.dumps({
        "status": "halted", "nodes": [], "completed": ["emit", "check"], "running_nodes": [],
        "error": "never reached: deliver",
    }))
    (ref / "spine-checkpoint.json").write_text(json.dumps({"owner_pid": os.getpid()}))
    (run_dir / "facts.json").write_text(json.dumps({"name": "test_halt"}))

    payload = _console_mod.status(run_dir)
    assert payload["state"] == "halted" and payload["finished"] is True
    assert "never reached: deliver" in format_status(payload)


def test_the_read_verbs_pass_their_flags_through():
    """`refactory status --run X` is the console's own command, flags and all.

    Argparse cannot express "everything after the verb belongs to somebody else", so
    these are forwarded before it runs; a regression here shows up as
    `unrecognized arguments: --run`.
    """
    result = _run("status", "--run", "/tmp/definitely-not-a-run")
    assert "unrecognized arguments" not in result.stderr
    # The readable path prints the error as text, not JSON.
    assert result.returncode != 0
    assert result.stderr, "a missing run must be reported, not accepted"


def test_a_name_is_derived_from_what_was_asked_for():
    """`ceo "an idea"` with no `--name` still names the result."""
    assert cli.slug("Count the teapots in a tray") == "count_the_teapots"
    assert cli.slug("2 fast ideas") == "pkg_2_fast_ideas"
    assert cli.slug("!!!") == "new_mode"
    assert len(cli.slug("a " * 40)) <= 40


def test_edit_refuses_to_run_without_the_mode_it_revises():
    """`--mode edit` without `--revise` is a creation, not a revision.

    A revision is judged by what changed, so a run that never said what it started
    from cannot be one. The CLI refuses rather than quietly creating a mode.
    """
    result = _run("ceo", "fewer steps please", "--mode", "edit", "--name", "x")
    assert result.returncode == 2
    assert "--revise <package>/<mode>" in result.stderr


def test_the_authoring_modes_are_the_ones_the_catalogue_offers():
    """`--mode create|design|edit` names real modes, with real authoring actions."""
    from refactory_dsh import console

    actions = {entry["id"]: entry["action"] for entry in console.catalogue(project=REPO_ROOT)}
    for verb, identifier in cli.AUTHORING.items():
        assert identifier in actions, f"{verb} points at {identifier}, which the catalogue does not list"
        assert actions[identifier] == verb, (
            f"{identifier}'s action is {actions[identifier]!r}, so `--mode {verb}` would not run it"
        )


@pytest.mark.parametrize("argv", [["--help"], ["ceo", "--help"], ["optimize", "--help"], ["package", "--help"], ["graph-control", "--help"]])
def test_help_is_available_for_every_verb_a_person_starts_with(argv):
    result = _run(*argv)
    assert result.returncode == 0
    assert "usage:" in result.stdout


#: An option argparse prints at the start of its own line, as `  --flag METAVAR`.
_OPTION = re.compile(r"^\s+(--[a-z][a-z0-9-]*)", re.MULTILINE)

#: The verbs whose options the README tabulates, so every one of their flags is owed a row.
_TABULATED = ("ceo", "optimize", "list", "ui", "doctor")


def test_the_command_reference_matches_the_command_line():
    """The reference's command tables are the parser's own verbs, options and defaults.

    Both directions matter: a reader told about a verb or flag that does not exist
    gets an exit 2, and a flag the parser accepts but the reference omits is a
    feature nobody finds. The verbs are read back from `--help` rather than from a
    list here, so a new verb cannot be added without the reference learning about it.
    The tables live in `refactory/docs/cli.md`; the README links there.
    """
    readme = (REPO_ROOT / "README.md").read_text()
    assert "refactory/docs/cli.md" in readme, "the README does not point at the command reference"
    reference = (REPO_ROOT / "refactory" / "docs" / "cli.md").read_text()

    verbs = re.search(r"\{([a-z,-]+)\}", _run("--help").stdout)
    assert verbs, "`refactory --help` no longer lists its verbs, so this test proves nothing"
    verb_list = verbs.group(1).split(",")

    missing = [verb for verb in verb_list if f"refactory {verb}" not in reference]
    assert missing == [], f"the command reference documents no command for: {missing}"

    declared: dict[str, set[str]] = {}
    # One console-script process per verb, as a person runs it; none depends on
    # another, so they run side by side.
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        helps = dict(zip(verb_list, pool.map(lambda verb: _run(verb, "--help"), verb_list)))
    for verb in verb_list:
        helped = helps[verb]
        assert helped.returncode == 0, f"the reference documents `refactory {verb}`, which has no help: {helped.stderr}"
        declared[verb] = set(_OPTION.findall(helped.stdout))
    named = set(re.findall(r"--[a-z][a-z0-9-]*", reference))

    # Walker commands (python -m refactory_dsh.*) are documented in their own
    # section and are not `refactory` CLI verbs, so exclude their flags.
    walker_match = re.search(r"(?m)^## Walker.*?(?=^## |\Z)", reference, re.DOTALL)
    walker_flags = set(re.findall(r"--[a-z][a-z0-9-]*", walker_match.group())) if walker_match else set()

    # argparse writes `-h, --help` on one line, which `_OPTION` skips; every verb
    # still accepts it, so the reference naming it is correct.
    invented = sorted(named - set().union(*declared.values()) - {"--help"} - walker_flags)
    assert invented == [], f"the command reference invents options no verb accepts: {invented}"

    undocumented = {
        verb: sorted(declared[verb] - named)
        for verb in _TABULATED if declared[verb] - named
    }
    assert undocumented == {}, f"the command reference omits options the parser accepts: {undocumented}"


def test_ceo_passes_a_persons_knob_to_the_authoring_run(monkeypatch):
    """`--knob allow_regression=1` is how a person says a lower score is intended.

    The refusal it answers lives in the run's smoke step, so before this flag there
    was no way to say "yes, trade the score" from the command line at all.
    """
    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "_forward", lambda module, argv: seen.append(argv) or 0)
    code = cli.main([
        "ceo", "make it smaller", "--mode", "edit",
        "--revise", "counting_pack/fixed_step", "--knob", "allow_regression=1",
    ])
    assert code == 0
    forwarded = seen[0]
    assert forwarded[0] == "default_pack/edit"
    assert "--knob" in forwarded and "allow_regression=1" in forwarded
    assert "revision_of=counting_pack/fixed_step" in forwarded


def test_the_model_a_run_names_is_passed_to_the_runtime(monkeypatch):
    """`--model`/`--provider` reach the graph profile, which is what runs the CEO.

    The flags are only useful if they arrive: the profile takes them for the agent it
    creates, and the mode's own agent nodes run through that same agent.
    """
    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "_forward", lambda module, argv: seen.append(argv) or 0)
    assert cli.main([
        "ceo", "counting_pack/fixed_step", "--intent", "count",
        "--model", "deepseek-v4-pro", "--provider", "deepseek-official",
    ]) == 0
    forwarded = seen[0]
    assert forwarded[0] == "counting_pack/fixed_step", "a named mode must run, not be authored"
    assert "--intent" in forwarded and "count" in forwarded
    assert "--model" in forwarded and "deepseek-v4-pro" in forwarded
    assert "--provider" in forwarded and "deepseek-official" in forwarded

    seen.clear()
    assert cli.main([
        "ceo", "an idea", "--mode", "create", "--single",
        "--model", "deepseek-v4-pro", "--provider", "deepseek-official",
    ]) == 0
    authored = seen[0]
    assert authored[0] == "default_pack/create"
    assert "--model" in authored and "deepseek-v4-pro" in authored
    assert "--provider" in authored and "deepseek-official" in authored


#: Briefs that contain a slash without being a `<package>/<mode>` identifier.
_BRIEFS_WITH_A_SLASH = (
    "Add a --version flag to refactory/refactory_dsh/cli.py",
    "make ./scripts/run.sh fail fast",
    "compare the v1/v2 outputs",
)


@pytest.mark.parametrize("brief", _BRIEFS_WITH_A_SLASH)
def test_ceo_designs_a_brief_that_mentions_a_path(monkeypatch, brief):
    """A slash in what someone asked for does not make it a mode name.

    Any `/` used to mean "run this mode", so a brief that mentioned a path failed with
    `PackageError: package 'Add a --version flag to … (refactory' must be lower-case
    letters…`, about a package nobody named, unless `--mode design` was spelled out
    for what is already the default.
    """
    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "_forward", lambda module, argv: seen.append(argv) or 0)
    assert cli.main(["ceo", brief, "--check", "pytest -q"]) == 0
    forwarded = seen[0]
    assert forwarded[0] == "default_pack/design", "a brief must be designed, not run as a mode"
    assert forwarded[forwarded.index("--intent") + 1] == brief


def test_ceo_refuses_a_brief_that_mentions_a_path_as_design_would(monkeypatch, capsys):
    """Without `--check`, such a brief gets design's own refusal, not a package error."""
    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "_forward", lambda module, argv: seen.append(argv) or 0)
    assert cli.main(["ceo", _BRIEFS_WITH_A_SLASH[0]]) == 2
    assert seen == [], "a refused design must not reach the runtime"
    assert "design needs --check" in capsys.readouterr().err


def test_ceo_gives_a_brief_that_mentions_a_path_its_target_once(monkeypatch, tmp_path):
    """`--target` on such a brief is design's knob, not also a named mode's."""
    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "_forward", lambda module, argv: seen.append(argv) or 0)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["ceo", _BRIEFS_WITH_A_SLASH[0], "--check", "pytest -q", "--target", "."]) == 0
    assert seen[0][0] == "default_pack/design"
    assert seen[0].count(f"target={tmp_path.resolve()}") == 1


@pytest.mark.parametrize("identifier", ["counting_pack/fixed_step", "countng_pack/fixed_step"])
def test_ceo_runs_a_well_formed_mode_name(monkeypatch, identifier):
    """`<package>/<mode>` still runs, including one this checkout does not have.

    The identifier's shape decides, not the registry: a mistyped mode name reaches the
    runner and is refused there as an unknown package, rather than being taken as a
    one-word brief and refused for a missing `--check`.
    """
    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "_forward", lambda module, argv: seen.append(argv) or 0)
    assert cli.main(["ceo", identifier]) == 0
    assert seen[0][0] == identifier, "a named mode must run, not be authored"


def test_ceo_refuses_a_creation_that_names_no_class_of_work(monkeypatch, capsys):
    """Neither `--task` nor `--single` is a decision nobody made.

    The mode refuses this at its own first node, which is a profile boot and a model
    later; the command line says it in milliseconds instead, and forwards nothing.
    """
    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "_forward", lambda module, argv: seen.append(argv) or 0)
    assert cli.main(["ceo", "an idea", "--mode", "create", "--name", "x"]) == 2
    assert seen == [], "a refused creation must not reach the runtime"
    assert "class of work" in capsys.readouterr().err


def test_ceo_passes_the_class_of_work_as_a_knob(monkeypatch):
    """`--task` says what class of work the produced harness is for."""
    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "_forward", lambda module, argv: seen.append(argv) or 0)
    assert cli.main([
        "ceo", "an idea", "--mode", "create", "--name", "x",
        "--task", "refactory_packages.dataset_probe.tasks:CountingTask",
    ]) == 0
    forwarded = seen[0]
    assert forwarded[0] == "default_pack/create"
    assert "task_ref=refactory_packages.dataset_probe.tasks:CountingTask" in forwarded


def test_ceo_single_reaches_the_mode_as_the_knob_it_reads(monkeypatch):
    """The special case travels as `single_shot`, which is the knob the mode reads."""
    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "_forward", lambda module, argv: seen.append(argv) or 0)
    assert cli.main(["ceo", "count to ten", "--mode", "create", "--single"]) == 0
    assert "single_shot=1" in seen[0]


def test_doctor_reports_a_checkout_mismatch_as_a_failing_row(
    tmp_path, monkeypatch, capsys
):
    """Doctor shows MISS for a profile whose symlink points to another checkout."""
    from argparse import Namespace

    from refactory_dsh.cli import cmd_doctor

    home = tmp_path / "home"
    # Create all owned targets with their bundles declared.
    for entry in dsh_profile.TARGETS:
        if entry.owned:
            bundles = [dsh_profile.LAYERS[key].bundle for key in entry.layers]
            _profile(home, entry.profile, bundles)
    # Link graph to another checkout.
    other = tmp_path / "other-checkout" / "packages" / "bundle" / "graph"
    _link_bundle(home, "graph", dsh_profile.GRAPH_BUNDLE, other)
    monkeypatch.setenv("DSH_HOME", str(home))
    monkeypatch.delenv("REFACTORY_MODEL_DSH_HOME", raising=False)
    monkeypatch.setattr(
        "refactory_dsh.sandbox.model_readiness",
        lambda *a, **kw: {"provider": "test", "model": "test", "status": "configured"},
    )
    cmd_doctor(Namespace(provider="", model=""))
    output = capsys.readouterr().out
    graph_line = next(line for line in output.splitlines() if "DSH profile 'graph'" in line)
    assert "MISS" in graph_line
    assert "other-checkout" in output


def test_doctor_ok_row_names_the_checkout_the_bundle_comes_from(
    tmp_path, monkeypatch, capsys
):
    """Doctor shows which checkout the graph bundle comes from on the ok row."""
    from argparse import Namespace

    from refactory_dsh.cli import cmd_doctor

    home = tmp_path / "home"
    bundle_target = tmp_path / "this-checkout" / "packages" / "bundle" / "graph"
    for entry in dsh_profile.TARGETS:
        if entry.owned:
            bundles = [dsh_profile.LAYERS[key].bundle for key in entry.layers]
            _profile(home, entry.profile, bundles)
    _link_bundle(home, "graph", dsh_profile.GRAPH_BUNDLE, bundle_target)
    monkeypatch.setenv("DSH_HOME", str(home))
    monkeypatch.delenv("REFACTORY_MODEL_DSH_HOME", raising=False)
    monkeypatch.setattr(
        "refactory_dsh.sandbox.model_readiness",
        lambda *a, **kw: {"provider": "test", "model": "test", "status": "configured"},
    )
    # Patch checkout_mismatch to return None so the profile is "ok"
    monkeypatch.setattr(dsh_profile, "checkout_mismatch", lambda *a, **kw: None)
    cmd_doctor(Namespace(provider="", model=""))
    output = capsys.readouterr().out
    graph_line = next(line for line in output.splitlines() if "DSH profile 'graph'" in line)
    assert "ok" in graph_line
    assert str(bundle_target) in output


def test_doctor_confirms_a_configured_key_and_names_a_missing_one(tmp_path, monkeypatch, capsys):
    """dm-257: doctor (and create/edit's credential check) could never confirm a key.

    The readiness probe imported a settings package the pinned harness (0.2.0-rc.2) does
    not ship, so every install reported "diagnostic-unavailable". It now reads the
    routes from the headless profile's entries and asks the managed store whether the
    referenced key is set, never for its value. The home and its store are fakes.
    """
    from argparse import Namespace

    from refactory_dsh.cli import cmd_doctor

    home = model_home_with_a_key(tmp_path / "dsh")
    monkeypatch.setenv("DSH_HOME", str(home))
    monkeypatch.delenv("REFACTORY_MODEL_DSH_HOME", raising=False)
    monkeypatch.delenv("DSH_CLI", raising=False)

    def row(output: str) -> str:
        return next(line for line in output.splitlines() if "sandbox model configuration" in line)

    cmd_doctor(Namespace(provider="", model="", run=None))
    configured = capsys.readouterr().out
    assert row(configured).split()[0] == "ok", row(configured)
    assert "fakeprov/fake-model: configured" in row(configured)

    cmd_doctor(Namespace(provider="otherprov", model="other-model", run=None))
    missing = capsys.readouterr().out
    assert row(missing).split()[0] == "MISS", row(missing)
    assert "REFACTORY_TEST_KEY_UNSET is not configured for otherprov" in row(missing)

    # DeepSeek's own route is a different plugin, whose key reference has a default.
    cmd_doctor(Namespace(provider="deepseek-official", model="deepseek-flash", run=None))
    deepseek = capsys.readouterr().out
    assert row(deepseek).split()[0] == "MISS", row(deepseek)
    assert "DEEPSEEK_API_KEY is not configured for deepseek-official" in row(deepseek)
    assert PLACEHOLDER_KEY not in configured + missing + deepseek


def test_doctor_asks_for_the_python_the_project_declares(monkeypatch, capsys):
    """dm-266: doctor wanted Python 3.12 while pyproject.toml declared 3.11, so a 3.11 install
    that `uv sync` accepted failed doctor. Both now name the version the code needs."""
    import tomllib
    from argparse import Namespace

    from refactory_dsh.cli import cmd_doctor

    declared = {
        tomllib.loads((root / "pyproject.toml").read_text())["project"]["requires-python"]
        for root in (REPO_ROOT, REPO_ROOT / "refactory")
    }
    assert declared == {">=3.11"}, declared
    monkeypatch.setattr(
        "refactory_dsh.sandbox.model_readiness",
        lambda *a, **kw: {"provider": "test", "model": "test", "status": "configured"},
    )
    cmd_doctor(Namespace(provider="", model="", run=None))
    rows = [line.split(None, 1)[1] for line in capsys.readouterr().out.splitlines()
            if line.strip().startswith(("ok", "MISS", "WARN"))]
    assert any(row.startswith("python >= 3.11 ") for row in rows), rows


def test_doctor_names_the_claude_code_the_claude_code_worker_runs(tmp_path, monkeypatch, capsys):
    """The bundled Claude Code and its version, a chosen one and where it came from, a WARN
    when the optional adapter is absent and a MISS when a chosen one cannot run."""
    from argparse import Namespace

    from refactory_dsh import workers
    from refactory_dsh.cli import cmd_doctor

    monkeypatch.setattr(
        "refactory_dsh.sandbox.model_readiness",
        lambda *a, **kw: {"provider": "test", "model": "test", "status": "configured"},
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))

    def row() -> str:
        cmd_doctor(Namespace(provider="", model="", run=None))
        [line] = [line.strip() for line in capsys.readouterr().out.splitlines()
                  if "claude_code executable" in line and not line.startswith("refactory doctor")]
        return line

    monkeypatch.setattr(workers, "bundled_claude_code", lambda root: {
        "path": "/sdk/claude", "version": "2.1.263", "sdk_version": "0.3.263"})
    bundled = row()
    assert bundled.split()[0] == "ok"
    assert "/sdk/claude (bundled with @anthropic-ai/claude-agent-sdk 0.3.263): Claude Code 2.1.263" in bundled

    monkeypatch.setattr(workers, "bundled_claude_code", lambda root: None)
    absent = row()
    assert absent.split()[0] == "WARN" and "is not installed" in absent

    chosen = tmp_path / "claude"
    chosen.write_text('#!/bin/sh\necho "2.1.300 (Claude Code)"\n')
    chosen.chmod(0o755)
    monkeypatch.setenv("REFACTORY_CLAUDE_EXECUTABLE", str(chosen))
    found = row()
    assert found.split()[0] == "ok"
    assert f"{chosen} (from REFACTORY_CLAUDE_EXECUTABLE): Claude Code 2.1.300" in found

    monkeypatch.setenv("REFACTORY_CLAUDE_EXECUTABLE", str(tmp_path / "gone"))
    assert row().split()[0] == "MISS"


def test_require_build_passes_when_spine_lib_exists(tmp_path):
    """A checkout with a built spine is not refused."""
    from refactory_dsh.paths import require_build, SPINE_LIB

    lib = tmp_path / SPINE_LIB
    lib.parent.mkdir(parents=True)
    lib.write_text("built")
    require_build(root=tmp_path)  # should not raise


def test_require_build_refuses_when_spine_lib_is_missing(tmp_path):
    """A checkout with no build refuses at once, naming pnpm run build."""
    from refactory_dsh.paths import require_build

    with pytest.raises(RuntimeError, match="pnpm run build"):
        require_build(root=tmp_path)


def test_require_build_skips_check_when_running_from_source(tmp_path):
    """Running from source (SOURCE_CLI_ENTRY exists) does not need the lib build."""
    from refactory_dsh.paths import require_build, SOURCE_CLI_ENTRY

    source = tmp_path / SOURCE_CLI_ENTRY
    source.parent.mkdir(parents=True)
    source.write_text("source entry")
    # No SPINE_LIB exists, but no error — tsx compiles on the fly
    require_build(root=tmp_path)  # should not raise


def test_graph_controls_reach_public_console_admission(tmp_path):
    result = _run("graph-control", "preview", "--run", str(tmp_path), "--node", "check")
    assert result.returncode == 2
    # With readable output, the error goes to stderr with the refactory: prefix.
    assert "durable graph checkpoint" in result.stderr
    assert "invalid choice" not in result.stderr and "unrecognized arguments" not in result.stderr


def test_status_shows_run_warnings(tmp_path):
    """Run warnings from events.jsonl appear in the readable status output."""
    run_dir = tmp_path / "warn_run"
    run_dir.mkdir()
    ref = run_dir / ".refactory"
    ref.mkdir()
    (ref / "spine-status.json").write_text(json.dumps({
        "status": "executing",
        "nodes": ["emit"],
        "completed": ["survey"],
        "running_nodes": [{"id": "emit", "kind": "agent"}],
    }))
    (ref / "spine-checkpoint.json").write_text(json.dumps({"owner_pid": 99999}))
    (run_dir / "facts.json").write_text(json.dumps({"name": "warn_test"}))
    # Write a run_warning event.
    (ref / "events.jsonl").write_text(
        json.dumps({
            "kind": "run_warning", "node_id": "emit",
            "state": "warning", "summary": "worktree file changed: README.md",
            "schema_version": 1, "run_id": str(run_dir),
        }) + "\n"
    )

    readable = _run("status", "--run", str(run_dir))
    assert readable.returncode == 0
    assert "Warning: worktree file changed: README.md" in readable.stdout


def test_status_shows_steer_then_resume_at_gate(tmp_path):
    """At a user gate, status shows the steer and resume commands, not resume alone."""
    run_dir = tmp_path / "gate_run"
    run_dir.mkdir()
    ref = run_dir / ".refactory"
    ref.mkdir()
    (ref / "spine-status.json").write_text(json.dumps({
        "status": "waiting_user",
        "nodes": ["gate_strategy"],
        "running_nodes": [{"id": "gate_strategy", "kind": "gate"}],
        "completed": ["survey"],
    }))
    (ref / "spine-checkpoint.json").write_text(json.dumps({"owner_pid": os.getpid()}))
    (run_dir / "facts.json").write_text(json.dumps({"name": "gate_test"}))
    graphs = run_dir / "graphs"
    graphs.mkdir()
    (graphs / "test.graph.json").write_text(json.dumps({
        "nodes": {
            "gate_strategy": {
                "_type": "GateNode",
                "evaluator_type": "user",
                "gate_prompt": "Approve the strategy?",
            },
        },
        "edges": [
            {"source": "gate_strategy", "condition": "proceed"},
            {"source": "gate_strategy", "condition": "reloop"},
        ],
    }))

    readable = _run("status", "--run", str(run_dir))
    assert readable.returncode == 0
    output = readable.stdout
    assert "refactory steer --run" in output
    assert "gate_strategy=<verdict>" in output
    assert "refactory graph-control resume --run" in output
    # The steer command must appear before the resume command.
    steer_pos = output.index("refactory steer --run")
    resume_pos = output.index("refactory graph-control resume --run")
    assert steer_pos < resume_pos


def test_status_is_readable_by_default_and_json_behind_flag(tmp_path):
    """status prints readable text by default; --json gives the console's JSON unchanged."""
    run_dir = tmp_path / "my_run"
    run_dir.mkdir()
    ref = run_dir / ".refactory"
    ref.mkdir()
    (ref / "spine-status.json").write_text(json.dumps({
        "status": "executing",
        "nodes": ["emit"],
        "completed": ["survey", "strategy"],
        "running_nodes": [{"id": "emit", "kind": "agent"}],
    }))
    (ref / "spine-checkpoint.json").write_text(json.dumps({"owner_pid": 99999}))
    workers = ref / "workers"
    workers.mkdir()
    (workers / "001.json").write_text(json.dumps({"node_id": "survey", "usage": {"tokens": 5000}}))
    (run_dir / "facts.json").write_text(json.dumps({"name": "my_design"}))

    readable = _run("status", "--run", str(run_dir))
    assert readable.returncode == 0
    # Should not be valid JSON (it's readable text).
    try:
        json.loads(readable.stdout)
        is_json = True
    except json.JSONDecodeError:
        is_json = False
    assert not is_json, "readable output should not be valid JSON"
    assert "Run " in readable.stdout
    assert "Steps done" in readable.stdout
    assert "emit (agent)" in readable.stdout

    machine = _run("status", "--run", str(run_dir), "--json")
    assert machine.returncode == 0
    payload = json.loads(machine.stdout)
    assert "run_id" in payload
    assert "state" in payload
    assert payload["authoring"]["completed"] == ["survey", "strategy"]

    # The CLI's --json must match the console's own JSON.
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        _console_mod.main(["status", "--run", str(run_dir)])
    console_payload = json.loads(buf.getvalue())
    # Both payloads come from the same static run directory; no fields change.
    assert payload == console_payload


def test_steer_is_readable_by_default_and_json_behind_flag(tmp_path):
    """steer prints readable text by default; --json gives the console's JSON unchanged."""
    run_dir = tmp_path / "steer_run"
    run_dir.mkdir()
    ref = run_dir / ".refactory"
    ref.mkdir()
    (ref / "spine-status.json").write_text(json.dumps({
        "status": "waiting_user", "nodes": ["gate_strategy"],
        "completed": ["survey"],
    }))
    (ref / "spine-checkpoint.json").write_text(json.dumps({"owner_pid": 99999}))
    (run_dir / "facts.json").write_text(json.dumps({"name": "steer_test"}))
    # Seed a minimal ledger.
    ledger_path = run_dir / "intent-ledger.md"
    ledger_path.write_text("# Intent Ledger\n\n## seed\nold entry\n")
    # Create a graph so steer can derive gates.
    graphs = run_dir / "graphs"
    graphs.mkdir()
    (graphs / "test.graph.json").write_text(json.dumps({
        "nodes": {
            "gate_strategy": {
                "_type": "GateNode",
                "evaluator_type": "user",
                "gate_prompt": "Approve?",
            },
        },
        "edges": [
            {"source": "gate_strategy", "condition": "proceed"},
            {"source": "gate_strategy", "condition": "reloop"},
        ],
    }))

    readable = _run("steer", "--run", str(run_dir),
                     "--verdict", "gate_strategy=proceed", "--note", "test note")
    assert readable.returncode == 0
    try:
        json.loads(readable.stdout)
        is_json = True
    except json.JSONDecodeError:
        is_json = False
    assert not is_json, "readable steer output should not be valid JSON"
    assert "Recorded verdict gate_strategy=proceed" in readable.stdout
    assert "Note recorded" in readable.stdout
    assert "graph-control resume" in readable.stdout

    machine = _run("steer", "--run", str(run_dir), "--json")
    assert machine.returncode == 0
    payload = json.loads(machine.stdout)
    assert "entries" in payload

    # The CLI's --json must match the console's own JSON.
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        _console_mod.main(["steer", "--run", str(run_dir)])
    console_payload = json.loads(buf.getvalue())
    # Both payloads come from the same static run directory; no fields change.
    assert payload == console_payload


def test_steer_shows_note_kind(tmp_path):
    """A note's kind appears in the readable steer output."""
    run_dir = tmp_path / "kind_run"
    run_dir.mkdir()
    ref = run_dir / ".refactory"
    ref.mkdir()
    (ref / "spine-status.json").write_text(json.dumps({
        "status": "executing", "nodes": ["emit"],
    }))
    (ref / "spine-checkpoint.json").write_text(json.dumps({"owner_pid": 99999}))
    (run_dir / "facts.json").write_text(json.dumps({"name": "kind_test"}))
    (run_dir / "intent-ledger.md").write_text("# Intent Ledger\n")

    result = _run("steer", "--run", str(run_dir),
                   "--note", "keep it small", "--kind", "constraint")
    assert result.returncode == 0
    assert "Note recorded" in result.stdout
    assert "constraint" in result.stdout


def test_steer_shows_verdict_scope(tmp_path):
    """A verdict's scope appears in the readable steer output."""
    run_dir = tmp_path / "scope_run"
    run_dir.mkdir()
    ref = run_dir / ".refactory"
    ref.mkdir()
    (ref / "spine-status.json").write_text(json.dumps({
        "status": "waiting_user", "nodes": ["approve"],
        "completed": [],
    }))
    (ref / "spine-checkpoint.json").write_text(json.dumps({"owner_pid": 99999}))
    (run_dir / "facts.json").write_text(json.dumps({"name": "scope_test"}))
    (run_dir / "intent-ledger.md").write_text("# Intent Ledger\n")
    graphs = run_dir / "graphs"
    graphs.mkdir()
    (graphs / "test.graph.json").write_text(json.dumps({
        "nodes": {
            "approve": {
                "_type": "GateNode",
                "evaluator_type": "user",
                "gate_prompt": "Approve the draft?",
            },
        },
        "edges": [
            {"source": "approve", "condition": "proceed"},
            {"source": "approve", "condition": "reloop"},
        ],
    }))

    result = _run("steer", "--run", str(run_dir),
                   "--verdict", "approve=proceed")
    assert result.returncode == 0
    assert "Recorded verdict approve=proceed" in result.stdout
    assert "scoped to" in result.stdout


def test_graph_control_is_readable_by_default_and_json_behind_flag(tmp_path):
    """graph-control prints readable text by default; --json gives JSON unchanged."""
    run_dir = tmp_path / "gc_run"
    run_dir.mkdir()
    ref = run_dir / ".refactory"
    ref.mkdir()
    (ref / "spine-checkpoint.json").write_text(json.dumps({"owner_pid": 99999}))
    (ref / "spine-status.json").write_text(json.dumps({"status": "executing", "nodes": ["emit"]}))

    readable = _run("graph-control", "pause", "--run", str(run_dir))
    assert readable.returncode == 0
    try:
        json.loads(readable.stdout)
        is_json = True
    except json.JSONDecodeError:
        is_json = False
    assert not is_json, "readable graph-control output should not be valid JSON"
    assert "Queued pause" in readable.stdout

    machine = _run("graph-control", "pause", "--run", str(run_dir), "--json")
    assert machine.returncode == 0
    payload = json.loads(machine.stdout)
    assert payload["status"] == "queued"

    # The CLI's --json output must match the console's own JSON for the same call.
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        _console_mod.main(["graph-control", "pause", "--run", str(run_dir)])
    console_payload = json.loads(buf.getvalue())
    # Compare every key except those that identify the individual request (id, text).
    per_request_keys = {"id", "text"}
    for key in set(payload) | set(console_payload):
        if key in per_request_keys:
            continue
        assert payload.get(key) == console_payload.get(key), f"--json and console disagree on {key}"


# ── empty-brief refusal ───────────────────────────────────────────────────


def test_ceo_refuses_an_empty_brief(monkeypatch, capsys):
    """``refactory ceo ""`` with an authoring mode must exit 2 and forward nothing."""
    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "_forward", lambda module, argv: seen.append(argv) or 0)
    code = cli.main(["ceo", "", "--mode", "design", "--name", "x", "--check", "true"])
    assert code == 2, "an empty brief must be refused"
    assert seen == [], "nothing must be forwarded when the brief is empty"
    captured = capsys.readouterr()
    assert "empty or whitespace-only" in captured.err
    assert "what" in captured.err


def test_ceo_refuses_a_whitespace_only_brief(monkeypatch, capsys):
    """``refactory ceo "   "`` with an authoring mode must exit 2 and forward nothing."""
    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "_forward", lambda module, argv: seen.append(argv) or 0)
    code = cli.main(["ceo", "   ", "--mode", "design", "--name", "x", "--check", "true"])
    assert code == 2, "a whitespace-only brief must be refused"
    assert seen == [], "nothing must be forwarded when the brief is whitespace"
    captured = capsys.readouterr()
    assert "empty or whitespace-only" in captured.err


def test_require_brief_unit():
    """The shared guard function raises on None, empty and whitespace; passes otherwise."""
    require_brief("valid idea")  # must not raise
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        require_brief("")
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        require_brief("   ")
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        require_brief(None)
    with pytest.raises(ValueError, match="what"):
        require_brief("", source="what")


def test_knob_required_flag():
    """The ``required`` flag on Knob: construction, round-trip and refusal on non-str."""
    k = Knob("", "str", required=True)
    assert k.required is True
    with pytest.raises(KnobError, match="only a 'str' knob may be required"):
        Knob(0, "int", required=True)
    spec = k.spec()
    assert spec["required"] is True
    rebuilt = Knob.from_spec(spec)
    assert rebuilt.required is True
    assert Knob("", "str", required=False).spec()["required"] is False


def test_v9_refuses_empty_required_knob():
    """V9 validation catches a non-default empty required knob."""
    from refactory_dsh.nn.graph import Graph, Node

    graph = Graph()
    graph.add(Node("work", "op", call="refactory_dsh.create_ops:prepare", knobs=("brief",)))
    state = {
        "version": 1,
        "graph": graph.to_state(),
        "knobs": {
            "brief": {
                "value": " ",
                "spec": Knob("", "str", required=True, description="idea").spec(),
            },
        },
    }
    issues = validate_state(state, resolve_calls=False)
    v9_required = [i for i in issues if i.code == "V9" and "required knob 'brief'" in i.message]
    assert len(v9_required) == 1, f"expected one V9 required-knob issue, got {[str(i) for i in issues]}"

    # A non-empty value produces no such issue.
    state["knobs"]["brief"]["value"] = "a real idea"
    issues2 = validate_state(state, resolve_calls=False)
    v9_required2 = [i for i in issues2 if i.code == "V9" and "required knob 'brief'" in i.message]
    assert v9_required2 == []


def test_v9_allows_default_empty_required_knob():
    """V9 does not flag a required knob whose value equals its default placeholder."""
    from refactory_dsh.nn.graph import Graph, Node

    graph = Graph()
    graph.add(Node("work", "op", call="refactory_dsh.create_ops:prepare", knobs=("brief",)))
    state = {
        "version": 1,
        "graph": graph.to_state(),
        "knobs": {
            "brief": {
                "value": "",
                "spec": Knob("", "str", required=True, description="idea").spec(),
            },
        },
    }
    issues = validate_state(state, resolve_calls=False)
    v9_required = [i for i in issues if i.code == "V9" and "required knob 'brief'" in i.message]
    assert v9_required == [], (
        "V9 must not flag a required knob whose value equals its default; "
        f"got {[str(i) for i in v9_required]}"
    )


def test_mode_run_refuses_empty_required_knob(tmp_path):
    """Mode.run refuses an empty required knob before the workdir is created."""
    from refactory_dsh.nn.graph import Graph, Node

    class StubMode(nn.Mode):
        name = "stub"

        @classmethod
        def declare(cls):
            return {"brief": Knob("", "str", required=True, description="idea")}

        def build(self):
            graph = Graph()
            graph.add(Node("work", "op", call="refactory_dsh.create_ops:prepare", knobs=("brief",)))
            return graph

    mode = StubMode()
    state = mode.state_dict()
    state["knobs"]["brief"]["value"] = "   "
    mode.load_state_dict(state)
    run_dir = tmp_path / "should-not-exist"
    with pytest.raises(ValueError, match="required knob 'brief'"):
        mode.run(run_dir)
    assert not run_dir.exists(), "run directory must not be created when a required knob is empty"


def test_a_required_knob_left_at_its_default_is_refused_unless_the_task_gives_it(tmp_path, monkeypatch):
    """A required knob's default is a placeholder: a run with no value for it anywhere is
    refused, and one whose task carries it (as the walker passes a brief) is not."""
    from refactory_dsh.nn import executor
    from refactory_dsh.nn.graph import Graph, Node

    class StubMode(nn.Mode):
        name = "stub"

        @classmethod
        def declare(cls):
            return {"brief": Knob("", "str", required=True, description="idea")}

        def build(self):
            graph = Graph()
            graph.add(Node("work", "op", call="refactory_dsh.create_ops:prepare", knobs=("brief",)))
            return graph

    ran = []
    monkeypatch.setattr(executor, "execute", lambda *args, **kwargs: ran.append(kwargs.get("task")) or "ran")
    run_dir = tmp_path / "never"
    with pytest.raises(ValueError, match="required knob 'brief'"):
        StubMode().run(run_dir)
    with pytest.raises(ValueError, match="required knob 'brief'"):
        StubMode().run(run_dir, task={"brief": ""})
    assert not run_dir.exists() and ran == []
    assert StubMode().run(run_dir, task={"brief": "a real idea"}) == "ran"


def test_interactive_prepare_refuses_empty_required_knob(tmp_path, monkeypatch):
    """interactive.prepare refuses an empty required knob before lowering."""
    from refactory_dsh import interactive, registry
    from refactory_dsh.nn.graph import Graph, Node

    class StubMode(nn.Mode):
        name = "stub"

        @classmethod
        def declare(cls):
            return {"brief": Knob("", "str", required=True, description="idea")}

        def build(self):
            graph = Graph()
            graph.add(Node("work", "op", call="refactory_dsh.create_ops:prepare", knobs=("brief",)))
            return graph

    monkeypatch.setattr(registry, "resolve", lambda project, identifier: ("stub_pack", StubMode))

    workdir = tmp_path / "run-test"
    # A value given blank is refused either way; no value at all only when headless,
    # because a session a person opens gets the idea in the conversation.
    for knobs, headless in ((["brief=   "], False), (["brief="], False), ([], True)):
        with pytest.raises(ValueError, match="required knob 'brief'"):
            interactive.prepare("stub_pack/stub", workdir=workdir, knobs=knobs, headless=headless)
        assert not workdir.exists()


def test_prepare_ops_refuse_an_empty_brief_with_a_name():
    """create_ops.prepare and design_ops.prepare_design refuse an empty brief when a name is given."""
    from unittest.mock import MagicMock

    from refactory_dsh import create_ops, design_ops

    ctx = MagicMock()
    ctx.task = {"name": "x", "brief": ""}
    ctx.knob = lambda name, default="": default
    ctx.workdir = Path(".")

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        create_ops.prepare(ctx)

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        design_ops.prepare_design(ctx)


def test_cli_docs_mention_empty_brief_refusal():
    """The CLI reference documents the empty-brief refusal."""
    cli_md = (REPO_ROOT / "refactory" / "docs" / "cli.md").read_text()
    assert "whitespace" in cli_md
