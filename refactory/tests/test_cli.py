"""The one command: `refactory`.

These hold the entry point's own behaviour — what each verb reaches, and the two
refusals that exist to stop a person being confused by a low-level error.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from refactory_dsh import cli
from refactory_dsh.paths import REPO_ROOT


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


def test_the_read_verbs_pass_their_flags_through():
    """`refactory status --run X` is the console's own command, flags and all.

    Argparse cannot express "everything after the verb belongs to somebody else", so
    these are forwarded before it runs; a regression here shows up as
    `unrecognized arguments: --run`.
    """
    result = _run("status", "--run", "/tmp/definitely-not-a-run")
    assert "unrecognized arguments" not in result.stderr
    payload = json.loads(result.stdout[result.stdout.index("{"):])
    assert payload["error"], "a missing run must be reported, not accepted"


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


def test_fidelity_candidate_forwards_portable_brief_without_task_or_single(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_forward", lambda module, args: calls.append((module, args)) or 0)
    assert cli.main(["ceo", "A reusable deterministic package", "--mode", "create_fidelity",
                     "--name", "portable_example", "--model", "gpt-6.1-sol", "--effort", "high"]) == 0
    assert calls[0][0] == "refactory_dsh.interactive"
    assert calls[0][1][0] == "default_pack/create_fidelity"
    assert "brief=A reusable deterministic package" in calls[0][1]
    assert "name=portable_example" in calls[0][1]
    assert "single_shot=1" not in calls[0][1]


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
    for verb in verb_list:
        helped = _run(verb, "--help")
        assert helped.returncode == 0, f"the reference documents `refactory {verb}`, which has no help: {helped.stderr}"
        declared[verb] = set(_OPTION.findall(helped.stdout))
    named = set(re.findall(r"--[a-z][a-z0-9-]*", reference))

    # argparse writes `-h, --help` on one line, which `_OPTION` skips; every verb
    # still accepts it, so the reference naming it is correct.
    invented = sorted(named - set().union(*declared.values()) - {"--help"})
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


def test_graph_controls_reach_public_console_admission(tmp_path):
    result = _run("graph-control", "preview", "--run", str(tmp_path), "--node", "check")
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert "durable graph checkpoint" in payload["error"]
    assert "invalid choice" not in result.stderr and "unrecognized arguments" not in result.stderr
