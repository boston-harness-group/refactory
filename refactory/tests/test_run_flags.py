"""One flag builder for `refactory ceo` and the walker's mains (dm-134, #436 gap 22).

Each run flag is given to both real entry points, `refactory ceo` (which forwards to
`refactory_dsh.interactive`, the spine's launcher) and `python -m refactory_dsh.design`
(the walker's main), and what each would run with is captured where it is handed on: the
knobs, the model route, the resource limits and the gates answered up front must be the
same. A flag one runtime cannot honour is refused there, with a message naming what to use
instead, and a resume refuses every flag that would change the run it continues.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import cli, create, design, edit, interactive, run_flags
from refactory_dsh.nn import RunResult

#: A design run answered up front says how it will deliver, which asks gh for its login.
pytestmark = pytest.mark.usefixtures("fake_gh")

BRIEF = "Change tool.py to VERSION 2."
BASE = ["--name", "tool", "--check", "true"]

#: Each flag, as one case: what both commands are given (the walker's `--fast` is the
#: spine's `--mode design_fast`).
CASES: dict[str, tuple[bool, list[str]]] = {
    "name and check": (False, []),
    "--knob": (False, ["--knob", "check_timeout_s=60", "--knob", "self_check_fixups=1"]),
    "--attempts": (False, ["--attempts", "2"]),
    "--workdir": (False, ["--workdir", "the-run"]),
    "--target, --base, --setup, --generate": (False, ["--target", ".", "--base", "v1", "--setup", "make",
                                                      "--generate", "make gen"]),
    "--no-worktree": (False, ["--no-worktree"]),
    "--no-pr": (False, ["--no-pr"]),
    "--refine": (False, ["--refine", "octo/app#3", "--review-comments", "--refine-design", "full"]),
    "the fast modes' --attempts": (True, ["--attempts", "2"]),
    "--repair-attempts": (True, ["--repair-attempts", "2"]),
    "--source-project": (True, ["--source-project", "src"]),
    "--worker-backend, --effort, --permission-mode": (False, ["--worker-backend", "codex", "--effort", "high",
                                                              "--permission-mode", "default"]),
    "--role-routes, --node-routes": (False, ["--role-routes", '{"builder": {"model": "m2"}}',
                                             "--node-routes", '{"emit": {"model": "m3"}}']),
    "--model-timeout": (False, ["--model-timeout", "90"]),
    "elapsed limits": (False, ["--max-elapsed-s", "100", "--validation-reserve-s", "10"]),
    "token limits": (False, ["--max-tokens", "1000", "--validation-reserve-tokens", "10", "--warn-tokens", "50"]),
    "--provider, --model": (False, ["--provider", "prov", "--model", "mod"]),
    "--approve": (False, ["--approve", "gate_strategy=proceed"]),
    "--verdict, the alias": (False, ["--verdict", "gate_strategy=proceed"]),
    "--headless": (False, ["--headless"]),
    "--omit-steering": (False, ["--omit-steering"]),
}
#: The flags one runtime cannot honour, and the runtime that refuses them.
REFUSED = {"--timeout": "walker", "--intent": "both", "--source": "both", "--source-format": "both"}


def _shared_options() -> set[str]:
    parser = argparse.ArgumentParser()
    run_flags.add_arguments(parser)
    return {option for action in parser._actions for option in action.option_strings  # noqa: SLF001
            if option.startswith("--") and option != "--help"}


def _spine(monkeypatch, fast: bool, flags: list[str]) -> dict[str, Any]:
    """`refactory ceo`, forwarded to the spine's launcher, which is stopped where it would run."""
    seen: dict[str, Any] = {}

    def run(identifier, **kwargs):
        seen.update(kwargs, identifier=identifier)
        return 0

    monkeypatch.setattr(interactive, "run", run)
    mode = ["--mode", "design_fast"] if fast else []
    assert cli.main(["ceo", BRIEF, *mode, *BASE, *flags]) == 0
    return seen


def _walker(monkeypatch, tmp_path: Path, fast: bool, flags: list[str]) -> dict[str, Any]:
    """`python -m refactory_dsh.design`, stopped where it would walk the graph."""
    seen: dict[str, Any] = {}

    class Client:
        def __init__(self, **kwargs):
            seen["route"] = {key: value for key, value in kwargs.items() if key != "project"}

        def context(self):
            return {"route": "recorded"}

        def route(self):
            return {}

    def prepare_run(state, workdir, **kwargs):
        seen["prepared"] = kwargs
        Path(workdir).mkdir(parents=True, exist_ok=True)
        return Path(workdir) / "graphs"

    def walk(self, workdir, **kwargs):
        seen["walked"] = {**kwargs, "knobs": self.values()}
        return RunResult(exit_ok=True, facts={})

    real_parse = create.parse_verdicts

    def parse_verdicts(pairs):
        seen["approve"] = list(pairs)
        return real_parse(pairs)

    monkeypatch.setattr("refactory_dsh.model.DshModelClient", Client)
    monkeypatch.setattr("refactory_dsh.run_preparation.prepare_run", prepare_run)
    monkeypatch.setattr("refactory_dsh.create.parse_verdicts", parse_verdicts)
    from refactory_dsh.fast import DesignFastMode
    monkeypatch.setattr(design.DesignMode, "run", walk)
    monkeypatch.setattr(DesignFastMode, "run", walk)
    monkeypatch.chdir(tmp_path)
    assert design.main(["--brief", BRIEF, *(["--fast"] if fast else []), *BASE, *flags]) == 0
    return seen


def _route(spine: dict[str, Any]) -> dict[str, Any]:
    """The route the spine's launcher builds its model client from, as the walker's is given."""
    found = {key: spine[key] for key in ("provider", "model", "worker_backend", "effort", "permission_mode")
             if spine.get(key)}
    found.update({key: spine[key] for key in ("role_routes", "node_routes") if spine.get(key) is not None})
    if spine.get("model_timeout_s") is not None:
        found["timeout_s"] = spine["model_timeout_s"]
    return found


def test_every_run_flag_is_compared_or_refused():
    """No flag of the shared builder is left out of this file: a new one must be given a case."""
    covered = {token for _fast, flags in CASES.values() for token in flags if token.startswith("--")}
    assert _shared_options() == covered | set(REFUSED) | {"--name", "--check"}


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_flag_configures_the_same_run_on_both_runtimes(case, monkeypatch, tmp_path):
    fast, flags = CASES[case]
    monkeypatch.chdir(tmp_path)  # both resolve `--target .` from here
    spine = _spine(monkeypatch, fast, flags)
    walker = _walker(monkeypatch, tmp_path, fast, flags)
    assert spine["identifier"] == walker["prepared"]["mode_id"]
    assert spine["knobs"] == walker["prepared"]["knobs"], "the same knobs, in the same order"
    assert _route(spine) == walker["route"]
    assert spine["resource_limits"] == walker["walked"]["resource_limits"]
    assert spine["approve"] == walker["approve"]
    assert walker["walked"]["task"] == {}, "the walker's settings travel as knobs, as the spine's do"
    assert spine["headless"] is ("--headless" in flags)
    # The knobs reach the walked mode: each one the flags set is its value there.
    values = walker["walked"]["knobs"]
    for pair in walker["prepared"]["knobs"]:
        key, _, value = pair.partition("=")
        assert str(values[key]).lower() in (value.lower(), {"0": "false", "1": "true"}.get(value, value)), pair


def test_the_aliases_say_what_they_map_to(monkeypatch, tmp_path):
    """`--verdict` is `--approve`, and `--attempts` is each mode's attempts knob."""
    help_text = {action.dest: action.help for action in _parser()._actions}  # noqa: SLF001
    assert "--verdict is an alias of --approve" in help_text["verdict"]
    assert "check_attempts=N for design" in help_text["attempts"]
    assert "repair_attempts=N for their _fast variants" in help_text["attempts"]
    assert run_flags.ATTEMPTS["default_pack/design"] == "check_attempts"
    assert run_flags.ATTEMPTS["default_pack/create"] == run_flags.ATTEMPTS["default_pack/edit"] == "attempts"
    walker = _walker(monkeypatch, tmp_path, False, ["--attempts", "2"])
    assert "check_attempts=2" in walker["prepared"]["knobs"] and walker["walked"]["knobs"]["check_attempts"] == 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    run_flags.add_arguments(parser)
    return parser


def test_the_walker_refuses_a_gate_timeout_and_names_the_limit(monkeypatch, tmp_path, capsys):
    """The walker never waits at a gate, so `--timeout` is refused, not ignored; the spine
    waits, and takes it."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as refused:
        design.main(["--brief", BRIEF, *BASE, "--timeout", "5"])
    assert refused.value.code == 2
    assert ("the walker does not wait at a gate (it stops there, for `refactory steer` to answer), so it "
            "takes no --timeout; bound the run with --max-elapsed-s instead") in capsys.readouterr().err
    assert _spine(monkeypatch, False, ["--timeout", "5"])["timeout_s"] == 5


@pytest.mark.parametrize("flag, value", [("--intent", "why"), ("--source", "items"),
                                         ("--source-format", "jsonl")])
def test_a_mode_runs_flags_are_refused_by_both_authoring_commands(flag, value, monkeypatch, tmp_path, capsys):
    """An authoring run takes its brief as its intent and reads no items: refused on both,
    with the same message, where `refactory ceo` used to drop them."""
    message = (f"refactory: {flag} is for running a mode (`refactory ceo <package>/<mode> {flag} ...`); "
               "an authoring run takes its brief as its intent and reads no items")
    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "_forward", lambda module, argv: seen.append(argv) or 0)
    assert cli.main(["ceo", BRIEF, *BASE, flag, value]) == 2
    assert message in capsys.readouterr().err and not seen
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        design.main(["--brief", BRIEF, *BASE, flag, value])
    assert message in capsys.readouterr().err


def test_a_knob_a_mode_does_not_declare_is_refused_on_both(monkeypatch, tmp_path, capsys):
    """`--repair-attempts` is the _fast modes' knob: design refuses it by name on both
    runtimes, which apply the same knobs the same way, where the walker used to drop it."""
    spine = _spine(monkeypatch, False, ["--repair-attempts", "2"])
    with pytest.raises(ValueError, match="repair_attempts") as on_the_spine:
        interactive.with_knobs(design.DesignMode().state_dict(), spine["knobs"])
    monkeypatch.chdir(tmp_path)
    assert design.main(["--brief", BRIEF, *BASE, "--repair-attempts", "2"]) == 2
    assert str(on_the_spine.value) in capsys.readouterr().err


def test_a_named_mode_takes_the_authoring_flags_as_knobs(monkeypatch):
    """`refactory ceo <package>/<mode>` used to drop `--name`, `--check` and `--attempts`;
    they reach the mode as knobs now, and `--attempts`, which names no knob of an arbitrary
    mode, is refused by name."""
    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "_forward", lambda module, argv: seen.append(argv) or 0)
    assert cli.main(["ceo", "default_pack/design", "--name", "tool", "--check", "true"]) == 0
    assert {"name=tool", "check=true"} <= set(seen[0])
    assert cli.main(["ceo", "default_pack/design", "--attempts", "2"]) == 2
    assert len(seen) == 1


def test_the_located_flags_are_refused_the_same_on_create(monkeypatch, tmp_path, capsys):
    """Only design and design_fast work in a worktree of their target, on either command."""
    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "_forward", lambda module, argv: seen.append(argv) or 0)
    assert cli.main(["ceo", BRIEF, "--mode", "create", "--single", "--target", "."]) == 2
    spine = capsys.readouterr().err
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        create.main(["--name", "tool", "--brief", BRIEF, "--single", "--target", "."])
    walker = capsys.readouterr().err
    tail = "does not take --target: only --mode design and design_fast work in a worktree of their target"
    assert f"refactory: --mode create {tail}" in spine and f"refactory: create {tail}" in walker


@pytest.mark.parametrize("main", [design.main, create.main, edit.main], ids=["design", "create", "edit"])
def test_each_walker_main_takes_every_run_flag_and_a_resume_refuses_them(main, tmp_path, capsys):
    """Every walker main parses every shared flag, and a resume refuses each one given,
    by name, as the spine's resume refuses changes: only the resource limits may change."""
    values = {"--knob": "a=1", "--attempts": "2", "--repair-attempts": "1", "--role-routes": "{}",
              "--node-routes": "{}", "--model-timeout": "60", "--max-elapsed-s": "100",
              "--max-tokens": "1000", "--validation-reserve-s": "1", "--validation-reserve-tokens": "1",
              "--warn-tokens": "5", "--refine-design": "full", "--source-format": "jsonl", "--timeout": "5"}
    switches = {"--no-worktree", "--no-pr", "--review-comments", "--headless", "--omit-steering"}
    argv = ["--resume", str(tmp_path / "run")]
    for option in sorted(_shared_options() - {"--verdict"}):
        argv += [option] if option in switches else [option, values.get(option, "x")]
    with pytest.raises(SystemExit) as refused:
        main(argv)
    assert refused.value.code == 2
    err = capsys.readouterr().err
    assert run_flags.RESUME_RETAINS in err
    given = err.split("(given: ", 1)[1].split(")", 1)[0].split(", ")
    kept = {"--max-elapsed-s", "--max-tokens", "--validation-reserve-s", "--validation-reserve-tokens",
            "--warn-tokens", "--headless", "--omit-steering"}
    assert set(given) == _shared_options() - {"--verdict"} - kept


@pytest.mark.parametrize("verb, spine_flags, walker_flags", [
    ("create", ["--mode", "create", "--single"], ["--single"]),
    ("edit", ["--mode", "edit", "--revise", "circle_packing/adaptive"], ["--mode", "circle_packing/adaptive"]),
])
def test_create_and_edit_take_the_same_knobs(verb, spine_flags, walker_flags, monkeypatch, tmp_path):
    """The walker's create and edit mains configure the run `refactory ceo` configures:
    the brief, the name, the creation directive, the mode to revise, the attempts alias and
    every `--knob`, in the same order."""
    from refactory_dsh.create import CreateMode
    from refactory_dsh.edit import EditMode

    flags = ["--name", "tool", "--attempts", "2", "--knob", "smoke_timeout_s=60"]
    monkeypatch.chdir(tmp_path)
    spine = _spine_argv(monkeypatch, ["ceo", BRIEF, *spine_flags, *flags])
    seen: dict = {}
    monkeypatch.setattr("refactory_dsh.run_preparation.prepare_run",
                        lambda state, workdir, **kwargs: seen.update(kwargs) or Path(workdir))
    for mode in (CreateMode, EditMode):
        monkeypatch.setattr(mode, "run", lambda self, workdir, **kwargs: RunResult(
            exit_ok=False, error="gate_strategy is waiting", gate_stop="gate_strategy"))
    _patch_client(monkeypatch)
    main = create.main if verb == "create" else edit.main
    assert main(["--brief", BRIEF, *walker_flags, *flags]) == 2
    assert seen["knobs"] == spine["knobs"]


def _spine_argv(monkeypatch, argv: list[str]) -> dict:
    seen: dict = {}
    monkeypatch.setattr(interactive, "run", lambda identifier, **kwargs: seen.update(kwargs) or 0)
    assert cli.main(argv) == 0
    return seen


def _patch_client(monkeypatch) -> None:
    class Client:
        def __init__(self, **kwargs):
            pass

        def context(self):
            return {}

        def route(self):
            return {}

    monkeypatch.setattr("refactory_dsh.model.DshModelClient", Client)


def test_a_persons_own_knob_wins_over_the_one_a_flag_derives(monkeypatch, tmp_path):
    """`--knob` comes last on both runtimes, so `--knob check_attempts=5` beats `--attempts 2`."""
    monkeypatch.chdir(tmp_path)
    flags = ["--attempts", "2", "--knob", "check_attempts=5"]
    spine = _spine(monkeypatch, False, flags)
    walker = _walker(monkeypatch, tmp_path, False, flags)
    assert spine["knobs"][-2:] == walker["prepared"]["knobs"][-2:] == ["check_attempts=2", "check_attempts=5"]
    assert walker["walked"]["knobs"]["check_attempts"] == 5
