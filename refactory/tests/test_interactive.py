"""Running one mode with a person in the loop (level 3).

The outer loop searches a mode; this runs one for a person to *use*. What these
hold: the intent ledger is folded into the graph before the run (seeding it
afterwards would produce a run nobody steered), the command names the graph the
profile loads, and a run that stopped at a gate is a failure naming the gate
rather than a score of 0.
"""

from __future__ import annotations

import json
from pathlib import Path

from refactory_dsh import dsh_runtime, interactive, registry, steering

ROOT = Path(__file__).resolve().parent.parent.parent
#: A mode with agent nodes, so "the ledger reaches the agents" has something to reach.
WITH_AGENTS = "default_pack/create"
#: A mode with a gate a person answers and a deterministic gate.
WITH_A_GATE = "review_pack/review"


def test_prepare_seeds_the_ledger_then_folds_it_into_the_graph(tmp_path):
    """Seeded **before** the lowering: the lowering is what reaches the agents."""
    name, graphs = interactive.prepare(
        WITH_AGENTS, workdir=tmp_path / "run", intent="write a counter mode", project=ROOT
    )
    assert (tmp_path / "run" / "intent-ledger.md").exists()

    graph = json.loads((graphs / f"{name}.graph.json").read_text())
    specs = [json.loads(path.read_text()) for path in (tmp_path / 'run/.refactory/worker-specs').glob('*.json')]
    prompts = ' '.join(spec['prompt'] for spec in specs)
    assert "write a counter mode" in prompts, (
        "the graph the profile loads does not carry the intent, so nobody steered it"
    )


def test_prepare_leaves_the_mode_unchanged(tmp_path):
    """D15's property for the interactive path: the mode is the author's."""
    source = ROOT / "refactory" / "refactory_packages" / "review_pack"
    before = {
        path: path.read_bytes() for path in sorted(source.rglob("*.py"))
    }
    interactive.prepare(WITH_A_GATE, workdir=tmp_path / "run", intent="review this", project=ROOT)
    after = {path: path.read_bytes() for path in sorted(source.rglob("*.py"))}
    assert after == before, "lowering for a run rewrote the mode it lowered"


def test_the_command_names_the_graph_the_profile_loads(tmp_path):
    name, graphs = interactive.prepare(
        WITH_A_GATE, workdir=tmp_path / "run", intent="review this", project=ROOT
    )
    argv = interactive.command(WITH_A_GATE, workdir=tmp_path / "run", graphs=graphs, name=name)
    assert argv[argv.index("--mode") + 1] == name
    assert argv[argv.index("--graphs") + 1] == str(graphs)
    assert (graphs / f"{name}.graph.json").exists()


def test_the_command_hands_the_knobs_the_node_runner_can_read(tmp_path):
    """The flag is the knob JSON itself, not a JSON string containing it.

    The node runner accepts `--knobs` only when the value parses to an object; a
    double-encoded string leaves every node on its op's fallback default, so the
    run under DSH is not the candidate the loop scored on the walker.
    """
    name, graphs = interactive.prepare(
        WITH_A_GATE, workdir=tmp_path / "run", intent="review this", project=ROOT
    )
    argv = interactive.command(WITH_A_GATE, workdir=tmp_path / "run", graphs=graphs, name=name)
    knobs = json.loads(argv[argv.index("--knobs") + 1])
    assert isinstance(knobs, dict), f"--knobs must carry the values, got {knobs!r}"
    expected = registry.resolve(ROOT, WITH_A_GATE)[1]().state_dict()["knobs"]
    assert knobs == {key: entry["value"] for key, entry in expected.items()}


def test_a_run_that_refused_a_gate_is_a_failure_naming_it(tmp_path, monkeypatch):
    """The refusal must not read as a run that merely scored 0."""
    mode = registry.resolve(ROOT, WITH_A_GATE)[1]()
    work = tmp_path / "run"
    (work / ".refactory").mkdir(parents=True)
    (work / ".refactory" / "spine-status.json").write_text(
        json.dumps({"status": "refused", "error": "gate approve is answered by a person"})
    )

    def _fake_run(state, workdir, **kwargs):
        return dsh_runtime.DshRun(facts={"topic": "x"}, workdir=Path(workdir), returncode=0)

    monkeypatch.setattr(dsh_runtime, "run_on_dsh", _fake_run)
    result = dsh_runtime.evaluate_on_dsh(mode, mode.state_dict(), work)
    assert result.exit_ok is False
    assert "gate approve" in (result.error or ""), result.error
    assert "did not finish" in (result.error or "")


def test_a_run_the_spine_finished_is_scored(tmp_path, monkeypatch):
    """The other side: a `done` status scores normally."""
    mode = registry.resolve(ROOT, WITH_A_GATE)[1]()
    work = tmp_path / "run"
    (work / ".refactory").mkdir(parents=True)
    (work / ".refactory" / "spine-status.json").write_text(json.dumps({"status": "done"}))

    def _fake_run(state, workdir, **kwargs):
        return dsh_runtime.DshRun(
            facts={"published_chars": 42}, workdir=Path(workdir), returncode=0
        )

    monkeypatch.setattr(dsh_runtime, "run_on_dsh", _fake_run)
    result = dsh_runtime.evaluate_on_dsh(mode, mode.state_dict(), work)
    assert result.exit_ok is True, result.error
    assert result.score == 42.0


def test_the_work_directory_reaches_dsh_as_an_absolute_path(tmp_path):
    """The default root is relative, and DSH refuses a relative session cwd.

    `interactive <mode>` with no `--workdir` used to stop before the graph started
    with "session header cwd must be an absolute path", so the one command a person
    runs to *use* a mode did not work out of the box.
    """
    name, graphs = interactive.prepare(WITH_A_GATE, workdir=Path(".factory/probe-relative"))
    argv = interactive.command(WITH_A_GATE, workdir=Path(".factory/probe-relative"), graphs=graphs, name=name)
    workdir = Path(argv[argv.index("--workdir") + 1])
    assert workdir.is_absolute(), f"DSH was handed the relative workdir {workdir}"
    assert Path(argv[argv.index("--graphs") + 1]).is_absolute()


def test_headless_is_exactly_todays_argv_and_the_default_chats(tmp_path):
    """`--headless` promises a script the argv it got before.

    So it must *leave the chat flag out*, not set it to false, and the live run's
    argv must be today's plus that one flag: nothing else may drift between them.
    """
    name, graphs = interactive.prepare(
        WITH_A_GATE, workdir=tmp_path / "run", intent="review this", project=ROOT
    )
    live = interactive.command(WITH_A_GATE, workdir=tmp_path / "run", graphs=graphs, name=name)
    headless = interactive.command(
        WITH_A_GATE, workdir=tmp_path / "run", graphs=graphs, name=name, headless=True
    )
    assert "--chat" in live, "the default run is not a live session"
    assert "--chat" not in headless, "a headless run asked DSH for a conversation"
    assert live == [*headless, "--chat"], "the only difference must be the chat flag"


def test_main_forwards_headless_to_the_run(monkeypatch):
    """The switch reaches the run, and its absence means live."""
    seen: list[dict] = []
    monkeypatch.setattr(interactive, "run", lambda *a, **k: seen.append(k) or 0)
    assert interactive.main([WITH_A_GATE, "--headless"]) == 0
    assert seen[-1]["headless"] is True
    assert interactive.main([WITH_A_GATE]) == 0
    assert seen[-1]["headless"] is False


def test_the_graph_argv_names_the_route_the_context_resolved(monkeypatch, tmp_path):
    """The CEO's own argv must carry the route the resolved context selected.

    The fresh-run path resolved a project file's provider into ``model_context``
    but built the graph argv from the untouched CLI flags, so the CEO ran the
    runtime's default provider while every node dispatch honored the project's
    selection: a project routed to a litellm gateway silently billed the
    runtime default's account.
    """
    from refactory_dsh import model as model_module
    from refactory_dsh import workers

    class RoutedClient:
        def __init__(self, **kwargs):
            self._given = kwargs

        def context(self):
            return {**self._resolved, "budget_ledger": ""}

        #: What a project file's ``refactory.workers.json`` would resolve to.
        _resolved = {
            "routes_resolved": True, "routing_provenance": {},
            "dsh_home": str(tmp_path), "provider": "gateway", "model": "gateway-model",
            "timeout_s": 1800, "attempts": 3, "backoff_s": 1.0,
            "runtime_root": str(ROOT), "dsh_cli": "", "python_bin": str(ROOT / ".venv/bin"),
            "worker_backend": "dsh", "effort": "", "permission_mode": "skip",
            "role_routes": {}, "node_routes": {}, "provider_config": {},
        }

    monkeypatch.setattr(model_module, "DshModelClient", RoutedClient)
    seen: list[list[str]] = []

    class _Run:
        returncode = 0

    def _capture(argv, **kwargs):
        seen.append(argv)
        return _Run()

    monkeypatch.setattr(workers, "run_graph_command", _capture)
    interactive.run(WITH_A_GATE, workdir=tmp_path / "run", project=ROOT, headless=True)
    argv = seen[-1]
    assert argv[argv.index("--provider") + 1] == "gateway"
    assert argv[argv.index("--model") + 1] == "gateway-model"
