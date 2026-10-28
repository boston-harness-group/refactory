"""One model choice for every run, at the boundary that matters: what each DSH process gets (dm-258).

A terminal run used to read the `graph` and `headless` profiles' own default, so with a
DeepSeek key stored it sent a person's code to DeepSeek although they had chosen another
provider in the GUI. These tests point ``DSH_CLI`` at a fake dsh that records the argv and
environment of every DSH process a run starts and answers like a scripted model, and check
that every one of them is handed the stored choice, that no entry point starts one without
a choice, and the visible first-run choice that replaces a bare refusal.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from refactory_dsh import cli, console, dsh_profile, interactive, workers
from refactory_dsh.model import DshModelClient, ModelChoiceError, require_model_choice
from tests_support import PLACEHOLDER_KEY, TEST_MODEL_CHOICE, store_model_choice

MESSAGE = dsh_profile.NO_MODEL_CHOICE_MESSAGE
MODE = "choicepack/ask"

_PACKAGE_INIT = '''\
"""Two agent nodes in a row: the smallest graph whose run starts a CEO and two workers."""
from refactory_dsh.nn import Package

from .mode import Ask

PACKAGE = Package(name="choicepack", description="ask a model twice", modes=(Ask,), ops=())
'''

_PACKAGE_MODE = '''\
"""Draft a line, then polish it."""
from refactory_dsh import nn


class Ask(nn.Mode):
    name = "ask"
    description = "draft a line, then polish it"

    @classmethod
    def declare(cls):
        return {"lines": nn.Knob(1, "int", lo=1, hi=3, description="how many lines to draft")}

    def build(self):
        graph = nn.Graph()
        graph.add(nn.Node("draft", "agent", prompt_template="Write {{lines}} line(s) of verse.",
                          knobs=("lines",), writes=("draft.md",), produces=("draft",)))
        graph.add(nn.Node("polish", "agent", prompt_template="Polish the verse in draft.md.",
                          reads=("draft.md",), writes=("final.md",), produces=("final",)))
        graph.connect("draft", "polish")
        return graph

    def score(self, run):
        return float(bool(run.facts.get("final")))
'''

#: The fake dsh. As the graph runner (`--profile graph`) it drains the lowered graph the
#: way an automatic run does, running each agent node's dispatch command; as a worker
#: (`--profile headless`) it answers like a scripted model and, as the observer does,
#: writes the route it was started on into the worker's receipt. Every invocation is
#: recorded, argv and model environment, before anything else.
_FAKE_DSH = r'''
import json, os, subprocess, sys
from pathlib import Path

argv = sys.argv[1:]


def option(name):
    return argv[argv.index(name) + 1] if name in argv else None


record = {"argv": argv, "profile": option("--profile"), "provider": option("--provider"),
          "model": option("--model"),
          "env": {key: value for key, value in os.environ.items()
                  if key.startswith(("REFACTORY_MODEL", "DSH_HOME"))}}
with open(os.environ["FAKE_DSH_LOG"], "a") as log:
    log.write(json.dumps(record) + "\n")

if record["profile"] == "headless":
    sys.stdin.read()
    receipt = os.environ.get("REFACTORY_WORKER_RECEIPT")
    if receipt:
        data = json.loads(Path(receipt).read_text())
        data["actual_route"] = {"worker_backend": "dsh", "provider": record["provider"] or "",
                                "model": record["model"] or ""}
        Path(receipt).write_text(json.dumps(data, indent=2) + "\n")
    print("A scripted line of verse.")
    sys.exit(0)

graphs, name, workdir = Path(option("--graphs")), option("--mode"), Path(option("--workdir"))
graph = json.loads((graphs / (name + ".graph.json")).read_text())
following = {}
for edge in graph["edges"]:
    following.setdefault(edge["source"], []).append(edge["target"])
env = {**os.environ, "REFACTORY_KNOBS": option("--knobs") or "{}", "REFACTORY_WORKDIR": str(workdir)}
queue, seen = [graph["start_node"]], set()
while queue:  # each node once, in graph order: enough to dispatch every worker
    node = queue.pop(0)
    if node in seen:
        continue
    seen.add(node)
    command = graph["nodes"][node].get("dispatch_command")
    if command:
        if subprocess.run(command, shell=True, cwd=workdir, env=env).returncode:
            sys.exit(1)
    queue += following.get(node, [])
(workdir / ".refactory").mkdir(exist_ok=True)
(workdir / ".refactory" / "spine-status.json").write_text(json.dumps({"status": "done"}))
print(json.dumps({"ok": True, "status": "done"}))
'''


class FakeDsh:
    def __init__(self, root: Path) -> None:
        self.log = root / "calls.jsonl"
        script = root / "fake_dsh.py"
        script.write_text(_FAKE_DSH)
        self.cli = root / "dsh"
        self.cli.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        self.cli.chmod(0o755)

    def calls(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines() if line.strip()]

    def routes(self) -> list[tuple[str | None, str | None, str | None]]:
        return [(call["profile"], call["provider"], call["model"]) for call in self.calls()]


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "project"
    package = root / ".factory" / "packages" / "choicepack"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(_PACKAGE_INIT)
    (package / "mode.py").write_text(_PACKAGE_MODE)
    monkeypatch.setenv("REFACTORY_PROJECT", str(root))
    return root


@pytest.fixture
def fake_dsh(tmp_path, monkeypatch, project, graph_profile_home) -> FakeDsh:
    """Every DSH process goes to the fake; the build check is not what is under test."""
    fake = FakeDsh(tmp_path)
    monkeypatch.setenv("DSH_CLI", str(fake.cli))
    monkeypatch.setenv("FAKE_DSH_LOG", str(fake.log))
    for name in ("REFACTORY_WORKER_CONTEXT", "REFACTORY_MODEL_PROVIDER", "REFACTORY_MODEL_ID",
                 "REFACTORY_WORKER_BACKEND"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(interactive, "require_build", lambda **kwargs: None)
    monkeypatch.setattr("refactory_dsh.paths.require_build", lambda *args, **kwargs: None)
    monkeypatch.setattr("refactory_dsh.dsh_runtime.require_build", lambda *args, **kwargs: None)
    return fake


def keys(home: Path, keyed: dict[str, str], unkeyed: tuple[str, ...] = (), *, deepseek: bool = False,
         harness_default: str = "") -> None:
    """The headless profile install writes, routing these providers, with their keys stored.

    ``keyed`` maps a provider route to its default model (the first its route lists);
    ``deepseek`` stores DeepSeek's own key; ``harness_default`` sets the profile's own
    `agent-default-model`, the default the bug used. Values are placeholders.
    """
    profile = home / "profiles" / "headless"
    profile.mkdir(parents=True, exist_ok=True)
    (profile / "package.json").write_text(json.dumps({
        "name": "dsh-profile-headless", "private": True, "dependencies": {},
        "dsh": {"profile": {"bundles": ["@deepseek-ai/dsh-base", "@deepseek-ai/dsh-headless"]}},
    }))
    (profile / "cordis.yml").write_text("[]\n")
    rows = ["- id: llm-pi-ai", "  name: '@deepseek-ai/dsh-llm-pi-ai'", "  config:", "    providers:"]
    for name, model in keyed.items():
        rows += [f"      {name}:", "        baseURL: http://127.0.0.1:9",
                 f"        apiKeyEnv: KEY_{name.upper().replace('-', '_')}",
                 "        models:", f"          - id: {model}"]
    for name in unkeyed:
        rows += [f"      {name}:", "        baseURL: http://127.0.0.1:9",
                 f"        apiKeyEnv: KEY_{name.upper().replace('-', '_')}"]
    if harness_default:
        provider, model = harness_default.split("/")
        rows += ["- id: agent-default-model", "  name: '@deepseek-ai/dsh-agent-default-model'",
                 "  config:", f"    provider: {provider}", f"    model: {model}"]
    (profile / "cordis.patch.yml").write_text("\n".join(rows) + "\n")
    refs = {f"KEY_{name.upper().replace('-', '_')}": PLACEHOLDER_KEY for name in keyed}
    if deepseek:
        refs["DEEPSEEK_API_KEY"] = PLACEHOLDER_KEY
    store = home / ".credentials.yaml"
    store.write_text("version: 1\nrefs:\n" + "".join(f"  {ref}: {value}\n" for ref, value in refs.items()))
    store.chmod(0o600)


def _workers(run: Path) -> list[dict]:
    receipts = [json.loads(path.read_text()) for path in (run / ".refactory" / "workers").glob("*.json")]
    return [receipt for receipt in receipts if receipt.get("role") != "coordinator"]


# ── correction 4: the argv every DSH process gets ─────────────────────────────


def test_a_terminal_run_hands_the_stored_choice_to_the_ceo_and_every_worker(tmp_path, fake_dsh, capsys):
    home = Path(os.environ["DSH_HOME"])
    store_model_choice(home, "chosen-prov", "chosen-model")
    # The trap the bug fell into: DeepSeek as the headless profile's own default.
    keys(home, {}, harness_default="deepseek-official/deepseek-flash")
    run = tmp_path / "run"

    assert cli.main(["ceo", MODE, "--workdir", str(run), "--headless"]) == 0, capsys.readouterr().err

    assert fake_dsh.routes() == [("graph", "chosen-prov", "chosen-model"),
                                 ("headless", "chosen-prov", "chosen-model"),
                                 ("headless", "chosen-prov", "chosen-model")]
    observed = [(receipt["node_id"], receipt["actual_route"]["provider"], receipt["actual_route"]["model"])
                for receipt in _workers(run)]
    assert sorted(observed) == [("draft", "chosen-prov", "chosen-model"), ("polish", "chosen-prov", "chosen-model")]
    saved = json.loads((run / ".refactory" / "run-config.json").read_text())["model_context"]
    assert (saved["provider"], saved["model"]) == ("chosen-prov", "chosen-model")


def test_a_deepseek_key_never_routes_a_run_that_chose_another_provider(tmp_path, fake_dsh, capsys):
    home = Path(os.environ["DSH_HOME"])
    store_model_choice(home, "chosen-prov", "chosen-model")
    keys(home, {"chosen-prov": "chosen-model"}, deepseek=True, harness_default="deepseek-official/deepseek-flash")
    run = tmp_path / "run"

    assert cli.main(["ceo", MODE, "--workdir", str(run), "--headless"]) == 0, capsys.readouterr().err

    calls = fake_dsh.calls()
    assert len(calls) == 3
    named = [(call["provider"], call["model"], call["env"].get("REFACTORY_MODEL_PROVIDER", ""),
              call["env"].get("REFACTORY_MODEL_ID", "")) for call in calls]
    assert not [route for route in named if "deepseek" in json.dumps(route)], named
    assert {(call["provider"], call["model"]) for call in calls} == {("chosen-prov", "chosen-model")}


def test_flags_override_the_choice_and_route_maps_still_apply(tmp_path, fake_dsh, capsys):
    run = tmp_path / "run"
    code = interactive.main([MODE, "--workdir", str(run), "--headless",
                             "--provider", "flag-prov", "--model", "flag-model",
                             "--node-routes", json.dumps({"polish": {"model": "polish-model"}}),
                             "--role-routes", json.dumps({"ceo": {"effort": "high"}})])
    assert code == 0, capsys.readouterr().err
    assert fake_dsh.routes()[0] == ("graph", "flag-prov", "flag-model")
    observed = {receipt["node_id"]: (receipt["actual_route"]["provider"], receipt["actual_route"]["model"])
                for receipt in _workers(run)}
    assert observed == {"draft": ("flag-prov", "flag-model"), "polish": ("flag-prov", "polish-model")}
    efforts = {receipt["node_id"]: receipt["requested_route"]["effort"] for receipt in _workers(run)}
    assert efforts == {"draft": "high", "polish": "high"}


_NO_CHOICE_ENTRY_POINTS = {
    "ceo": ["ceo", MODE, "--headless"],
    "create": ["ceo", "a counter that counts", "--mode", "create", "--single", "--name", "counted", "--headless"],
    "edit": ["ceo", "make it terser", "--mode", "edit", "--revise", MODE, "--name", "terser", "--headless"],
    "optimize": ["optimize", MODE, "--evaluator", "dsh", "--max-evaluations", "1"],
    "package smoke": ["package", "smoke", MODE, "--model"],
}


@pytest.mark.parametrize("entry", sorted(_NO_CHOICE_ENTRY_POINTS))
def test_with_no_choice_every_entry_point_refuses_before_any_dsh_process(entry, tmp_path, fake_dsh, no_model_choice,
                                                                      capsys, monkeypatch):
    keys(no_model_choice, {}, ("fakeprov",), harness_default="deepseek-official/deepseek-flash")
    monkeypatch.chdir(tmp_path)
    argv = list(_NO_CHOICE_ENTRY_POINTS[entry])
    if entry == "optimize":
        argv += ["--runs-root", str(tmp_path / "search")]
    assert cli.main(argv) == 1
    assert f"refactory: {MESSAGE}" in capsys.readouterr().err
    assert fake_dsh.calls() == []
    assert dsh_profile.stored_model_choice() == {"provider": "", "model": ""}


def test_with_no_choice_a_bare_console_open_refuses_before_anything_is_written(tmp_path, fake_dsh, no_model_choice,
                                                                               capsys):
    """A `console open` that names no model. The GUI's `wf_open` always names the session's."""
    keys(no_model_choice, {}, ("fakeprov",))
    run = tmp_path / "gui-run"
    assert console.main(["open", MODE, "--workdir", str(run)]) == 1
    assert json.loads(capsys.readouterr().out) == {"error": MESSAGE}
    assert not run.exists()
    assert fake_dsh.calls() == []


def test_a_gui_open_runs_its_workers_on_the_session_model_and_is_not_refused(tmp_path, fake_dsh, no_model_choice,
                                                                             capsys):
    """`wf_open` passes the GUI session's model as `--provider`/`--model`: that is the run's route."""
    run = tmp_path / "gui-run"
    assert console.main(["open", MODE, "--workdir", str(run), "--provider", "session-prov",
                         "--model", "session-model"]) == 0, capsys.readouterr().out
    specs = [json.loads(path.read_text()) for path in (run / ".refactory" / "worker-specs").glob("*.json")]
    assert {(spec["model_context"]["provider"], spec["model_context"]["model"]) for spec in specs} == {
        ("session-prov", "session-model")}
    assert dsh_profile.stored_model_choice() == {"provider": "", "model": ""}


def test_a_console_open_lowers_every_worker_onto_the_stored_choice(tmp_path, fake_dsh, capsys):
    store_model_choice(Path(os.environ["DSH_HOME"]), "chosen-prov", "chosen-model")
    run = tmp_path / "gui-run"
    assert console.main(["open", MODE, "--workdir", str(run)]) == 0, capsys.readouterr().out
    saved = json.loads((run / ".refactory" / "run-config.json").read_text())["model_context"]
    assert (saved["provider"], saved["model"]) == ("chosen-prov", "chosen-model")
    specs = [json.loads(path.read_text()) for path in (run / ".refactory" / "worker-specs").glob("*.json")]
    assert sorted((spec["node_id"], spec["model_context"]["provider"], spec["model_context"]["model"])
                  for spec in specs) == [("draft", "chosen-prov", "chosen-model"),
                                         ("polish", "chosen-prov", "chosen-model")]


@pytest.mark.parametrize("evaluator", ["walker", "dsh"])
def test_both_evaluators_of_an_optimize_run_get_the_same_route_as_the_spine(evaluator, tmp_path, fake_dsh, capsys,
                                                                              monkeypatch):
    store_model_choice(Path(os.environ["DSH_HOME"]), "chosen-prov", "chosen-model")
    monkeypatch.chdir(tmp_path)
    cli.main(["optimize", MODE, "--evaluator", evaluator, "--optimizer", "random", "--preset", "knob_only",
              "--epochs", "1", "--batch-size", "1", "--max-evaluations", "1", "--seed", "1",
              "--runs-root", str(tmp_path / "search")])
    routes = fake_dsh.routes()
    assert routes, capsys.readouterr().err
    assert {(provider, model) for _profile, provider, model in routes} == {("chosen-prov", "chosen-model")}
    if evaluator == "dsh":
        assert "graph" in [profile for profile, _provider, _model in routes]


# ── correction 7: a visible first-run choice instead of a bare refusal ─────────


def test_one_keyed_provider_is_used_saved_and_said_once(tmp_path, fake_dsh, no_model_choice, capsys):
    keys(no_model_choice, {"fakeprov": "fake-model"}, ("otherprov",), harness_default="deepseek-official/deepseek-flash")
    line = "Using fakeprov/fake-model, the only provider with a key; change it with `refactory model set`"

    assert cli.main(["ceo", MODE, "--workdir", str(tmp_path / "first"), "--headless"]) == 0
    assert line in capsys.readouterr().err
    assert dsh_profile.stored_model_choice() == {"provider": "fakeprov", "model": "fake-model"}
    assert set(fake_dsh.routes()) == {("graph", "fakeprov", "fake-model"), ("headless", "fakeprov", "fake-model")}

    assert cli.main(["ceo", MODE, "--workdir", str(tmp_path / "second"), "--headless"]) == 0
    assert "Using " not in capsys.readouterr().err
    assert len(fake_dsh.calls()) == 6
    assert {(provider, model) for _profile, provider, model in fake_dsh.routes()} == {("fakeprov", "fake-model")}


def test_two_keyed_providers_refuse_a_headless_run_and_list_them(tmp_path, fake_dsh, no_model_choice, capsys):
    keys(no_model_choice, {"fakeprov": "fake-model"}, deepseek=True)
    assert cli.main(["ceo", MODE, "--workdir", str(tmp_path / "run"), "--headless"]) == 1
    said = capsys.readouterr().err
    assert "several providers have a key: deepseek-official, fakeprov" in said
    assert "refactory model set --provider <provider> --model <model>" in said
    assert fake_dsh.calls() == []
    assert dsh_profile.stored_model_choice() == {"provider": "", "model": ""}


class _Terminal:
    """A stdin a person is typing at: a terminal, with one answer ready."""

    def __init__(self, answer: str) -> None:
        self.answer = answer

    def isatty(self) -> bool:
        return True

    def readline(self) -> str:
        return self.answer + "\n"


def test_two_keyed_providers_at_a_terminal_ask_once_and_keep_the_answer(tmp_path, fake_dsh, no_model_choice, capsys,
                                                                         monkeypatch):
    keys(no_model_choice, {"fakeprov": "fake-model"}, deepseek=True)
    monkeypatch.setattr(sys, "stdin", _Terminal("fakeprov"))
    assert interactive.main([MODE, "--workdir", str(tmp_path / "run")]) == 0
    said = capsys.readouterr()
    assert "deepseek-official" in said.err and "fakeprov (fake-model)" in said.err
    assert "Using fakeprov/fake-model; change it with `refactory model set`" in said.err
    assert dsh_profile.stored_model_choice() == {"provider": "fakeprov", "model": "fake-model"}
    assert {(provider, model) for _profile, provider, model in fake_dsh.routes()} == {("fakeprov", "fake-model")}
    assert fake_dsh.routes()[0][0] == "graph" and "--chat" in fake_dsh.calls()[0]["argv"]


def test_no_keyed_provider_refuses_with_the_instruction(tmp_path, fake_dsh, no_model_choice, capsys):
    keys(no_model_choice, {}, ("fakeprov", "otherprov"), harness_default="deepseek-official/deepseek-flash")
    assert cli.main(["ceo", MODE, "--workdir", str(tmp_path / "run"), "--headless"]) == 1
    assert capsys.readouterr().err.strip().endswith(MESSAGE)
    assert fake_dsh.calls() == []


def test_a_stored_choice_is_never_replaced_by_a_keyed_provider(tmp_path, fake_dsh, capsys):
    home = Path(os.environ["DSH_HOME"])
    stored = store_model_choice(home, "chosen-prov", "chosen-model")
    before = stored.read_bytes()
    keys(home, {"fakeprov": "fake-model"})
    assert cli.main(["ceo", MODE, "--workdir", str(tmp_path / "run"), "--headless"]) == 0
    assert "Using " not in capsys.readouterr().err
    assert stored.read_bytes() == before
    assert {(provider, model) for _profile, provider, model in fake_dsh.routes()} == {("chosen-prov", "chosen-model")}


# ── migration: a GUI choice is the stored choice, in place ─────────────────────

_GUI_PATCH = '''\
# Your profile's own patch, layered over its bundles (entry overrides,
# overrides, disables, and insert lists; `!!js` expressions allowed).
- id: ui-settings-general
  name: "@deepseek-ai/dsh-client-ui-settings-general"
  config:
    theme: dark
- id: steer-fields
  name: "@deepseek-ai/dsh-graph"
  config: !!js |
    ({ fields: ["intent", "verdict"] })
- id: llm-pi-ai
  name: "@deepseek-ai/dsh-llm-pi-ai"
  config:
    providers:
      anthropic:
        apiKeyEnv: ANTHROPIC_API_KEY
- id: agent-default-model
  name: "@deepseek-ai/dsh-agent-default-model"
  config:
    provider: anthropic
    model: claude-opus-4-6
'''


def test_a_choice_made_in_the_gui_is_used_by_a_terminal_run_as_it_is(tmp_path, fake_dsh, capsys):
    path = dsh_profile.model_choice_path()
    path.write_text(_GUI_PATCH)
    assert cli.main(["ceo", MODE, "--workdir", str(tmp_path / "run"), "--headless"]) == 0
    assert {(provider, model) for _profile, provider, model in fake_dsh.routes()} == {("anthropic", "claude-opus-4-6")}
    assert path.read_text() == _GUI_PATCH
    assert "Using " not in capsys.readouterr().err


# ── correction 1: `refactory model set` edits its own entry and nothing else ───


def test_model_set_keeps_every_other_byte_and_reads_back(capsys):
    path = dsh_profile.model_choice_path()
    path.write_text(_GUI_PATCH)
    assert cli.main(["model", "set", "--provider", "deepseek-official", "--model", "@cf/deepseek:v4"]) == 0
    after = path.read_text()
    assert after == _GUI_PATCH.replace("provider: anthropic\n    model: claude-opus-4-6",
                                       'provider: deepseek-official\n    model: "@cf/deepseek:v4"')
    assert dsh_profile.stored_model_choice() == {"provider": "deepseek-official", "model": "@cf/deepseek:v4"}
    capsys.readouterr()
    assert cli.main(["model", "show"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert (shown["provider"], shown["model"], shown["stored_in"]) == ("deepseek-official", "@cf/deepseek:v4", str(path))


def test_model_set_adds_its_entry_to_a_patch_without_one_and_to_a_fresh_profile(capsys):
    path = dsh_profile.model_choice_path()
    path.write_text(_GUI_PATCH.split("- id: agent-default-model")[0])
    assert cli.main(["model", "set", "--provider", "p1", "--model", "m1"]) == 0
    assert path.read_text().startswith(_GUI_PATCH.split("- id: agent-default-model")[0])
    assert dsh_profile.stored_model_choice() == {"provider": "p1", "model": "m1"}
    path.unlink()
    assert cli.main(["model", "set", "--provider", "p2", "--model", "m2"]) == 0
    assert dsh_profile.stored_model_choice() == {"provider": "p2", "model": "m2"}
    assert path.stat().st_mode & 0o777 == 0o600


def test_model_set_writes_by_rename_and_a_failed_write_leaves_the_file(monkeypatch, capsys):
    path = dsh_profile.model_choice_path()
    path.write_text(_GUI_PATCH)
    inode = path.stat().st_ino
    assert cli.main(["model", "set", "--provider", "p1", "--model", "m1"]) == 0
    assert path.stat().st_ino != inode
    before = path.read_bytes()

    def refuse(source, target):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", refuse)
    assert cli.main(["model", "set", "--provider", "p2", "--model", "m2"]) == 1
    assert "disk full" in capsys.readouterr().err
    assert path.read_bytes() == before
    assert [entry.name for entry in path.parent.iterdir()] == [path.name]


def test_model_set_leaves_a_file_it_cannot_read_alone(capsys):
    path = dsh_profile.model_choice_path()
    path.write_text("theme: dark\n")
    assert cli.main(["model", "set", "--provider", "p1", "--model", "m1"]) == 1
    assert "left alone" in capsys.readouterr().err
    assert path.read_text() == "theme: dark\n"


def test_model_show_without_a_choice_says_how_to_choose(no_model_choice, capsys):
    assert cli.main(["model", "show"]) == 1
    assert MESSAGE in capsys.readouterr().err


# ── correction 3: an ordinary exception, so library code never exits ──────────


def test_the_refusal_is_an_ordinary_exception_raised_before_any_dsh_process(tmp_path, fake_dsh, no_model_choice):
    assert not issubclass(ModelChoiceError, SystemExit) and issubclass(ModelChoiceError, Exception)
    client = DshModelClient(routes_resolved=True, worker_backend="dsh", dsh_cli=str(fake_dsh.cli), attempts=1)
    with pytest.raises(ModelChoiceError, match="No model chosen"):
        client.complete("hello", workdir=tmp_path)
    with pytest.raises(ModelChoiceError, match="names provider half-prov but no model"):
        require_model_choice(DshModelClient(routes_resolved=True, provider="half-prov"))
    assert fake_dsh.calls() == []


# ── correction 5: doctor checks the chosen provider, never the profile's ───────


def _doctor(capsys, provider: str = "", model: str = "") -> tuple[int, dict[str, str]]:
    """Doctor's exit code and its two model rows, each as ``MARK  name  detail``."""
    from argparse import Namespace

    code = cli.cmd_doctor(Namespace(provider=provider, model=model, run=None))
    rows = {}
    for line in capsys.readouterr().out.splitlines():
        stripped = line.strip()
        for name in ("run model", "sandbox model configuration"):
            if stripped[4:].strip().startswith(name):
                rows[name] = stripped
    return code, rows


def test_doctor_checks_the_stored_choice_not_the_headless_default(monkeypatch, capsys):
    monkeypatch.delenv("DSH_CLI", raising=False)
    home = Path(os.environ["DSH_HOME"])
    keys(home, {"fakeprov": "fake-model"}, ("otherprov",), harness_default="fakeprov/fake-model")
    store_model_choice(home, "otherprov", "other-model")
    _code, rows = _doctor(capsys)
    assert rows["run model"].startswith("ok") and "otherprov/other-model (stored choice)" in rows["run model"]
    assert rows["sandbox model configuration"].startswith("MISS")
    assert "KEY_OTHERPROV is not configured for otherprov" in rows["sandbox model configuration"]


def test_doctor_without_a_choice_says_what_the_first_run_will_do(monkeypatch, capsys, no_model_choice):
    monkeypatch.delenv("DSH_CLI", raising=False)
    keys(no_model_choice, {"fakeprov": "fake-model"}, ("otherprov",), harness_default="otherprov/other-model")
    _code, rows = _doctor(capsys)
    assert rows["run model"].startswith("WARN")
    assert "the first run will use fakeprov/fake-model" in rows["run model"]
    assert rows["sandbox model configuration"].startswith("ok")
    assert "fakeprov/fake-model: configured" in rows["sandbox model configuration"]
    assert dsh_profile.stored_model_choice() == {"provider": "", "model": ""}

    keys(no_model_choice, {}, ("fakeprov",), harness_default="fakeprov/fake-model")
    code, rows = _doctor(capsys)
    assert code == 1 and rows["run model"].startswith("MISS") and MESSAGE in rows["run model"]
    assert "sandbox model configuration" not in rows


# ── correction 6: `refactory runtime` says where the route came from ───────────


def _runtime(capsys, *args: str) -> dict:
    assert console.main(["runtime", *args]) == 0
    return json.loads(capsys.readouterr().out)["model_route"]


def test_runtime_names_the_source_of_the_route(project, capsys, monkeypatch, no_model_choice):
    for name in ("REFACTORY_MODEL_PROVIDER", "REFACTORY_MODEL_ID", "REFACTORY_WORKER_CONTEXT"):
        monkeypatch.delenv(name, raising=False)
    project_flag = ["--project", str(project)]
    route = _runtime(capsys, *project_flag)
    assert (route["provider"], route["model"], route["source"]) == ("", "", {"provider": "none", "model": "none"})

    store_model_choice(no_model_choice, "chosen-prov", "chosen-model")
    route = _runtime(capsys, *project_flag)
    assert (route["provider"], route["source"]["provider"]) == ("chosen-prov", "stored choice")
    assert route["stored_choice"] == {"provider": "chosen-prov", "model": "chosen-model"}

    route = _runtime(capsys, *project_flag, "--provider", "flag-prov", "--model", "flag-model")
    assert (route["provider"], route["model"]) == ("flag-prov", "flag-model")
    assert route["source"] == {"provider": "flag", "model": "flag"}

    (project / "refactory.workers.json").write_text(json.dumps({
        "model": "project-model",
        "node_routes": {"polish": {"model": "node-model"}},
        "role_routes": {"reviewer": {"provider": "role-prov", "model": "role-model"}},
    }))
    route = _runtime(capsys, *project_flag)
    assert route["source"] == {"provider": "stored choice", "model": "project file"}
    assert route["overrides"] == [
        {"applies_to": "node polish", "source": "node route", "model": "node-model"},
        {"applies_to": "role reviewer", "source": "role route", "provider": "role-prov", "model": "role-model"},
    ]


# ── the OpenShell backend: out of the one choice's scope, never silent ─────────


def test_an_openshell_run_takes_no_route_from_the_choice_and_says_whose_model_it_uses(tmp_path, fake_dsh, project,
                                                                                     capsys, monkeypatch):
    """Its graph runs in the sandbox, whose only inference is the gateway's verified claude-code provider."""
    from refactory_dsh import openshell_backend

    store_model_choice(Path(os.environ["DSH_HOME"]), "chosen-prov", "chosen-model")
    (project / "refactory.workers.json").write_text(json.dumps({"worker_backend": "openshell"}))
    handed: list[tuple[list[str], dict]] = []

    class _Done:
        returncode = 0

    monkeypatch.setattr(openshell_backend, "run_in_openshell",
                        lambda argv, *, context, **kwargs: handed.append((argv, context)) or _Done())
    run = tmp_path / "run"
    assert interactive.main([MODE, "--workdir", str(run), "--headless"]) == 0
    assert interactive.OPENSHELL_MODEL_NOTE in capsys.readouterr().err
    [(argv, context)] = handed
    assert "--provider" not in argv and "--model" not in argv
    assert context["worker_backend"] == "openshell" and not context["provider"]
    assert "chosen-prov" not in json.dumps(handed)
    assert fake_dsh.calls() == []
    # The backend takes no provider of its own either, so nothing can name one.
    with pytest.raises(ValueError, match="does not support provider"):
        workers.validate_route({"worker_backend": "openshell", "provider": "chosen-prov"})


def test_an_openshell_run_is_not_refused_or_asked_for_a_choice(tmp_path, fake_dsh, project, no_model_choice,
                                                               monkeypatch, capsys):
    from refactory_dsh import openshell_backend

    keys(no_model_choice, {"fakeprov": "fake-model"})
    (project / "refactory.workers.json").write_text(json.dumps({"worker_backend": "openshell"}))

    class _Done:
        returncode = 0

    monkeypatch.setattr(openshell_backend, "run_in_openshell", lambda argv, **kwargs: _Done())
    assert interactive.main([MODE, "--workdir", str(tmp_path / "run"), "--headless"]) == 0
    assert "Using " not in capsys.readouterr().err
    assert dsh_profile.stored_model_choice() == {"provider": "", "model": ""}


# ── the paths below the entry points ──────────────────────────────────────────


@pytest.mark.parametrize("module, argv", [
    ("create", ["--name", "counted", "--brief", "a counter"]),
    ("edit", ["--mode", MODE, "--name", "terser", "--brief", "make it terser"]),
    ("design", ["--name", "tool", "--brief", "a tool", "--check", "true"]),
])
def test_with_no_choice_the_walker_authoring_commands_refuse(module, argv, tmp_path, fake_dsh, no_model_choice,
                                                             capsys, monkeypatch):
    import importlib

    keys(no_model_choice, {}, ("fakeprov",))
    monkeypatch.chdir(tmp_path)
    assert importlib.import_module(f"refactory_dsh.{module}").main(argv) == 1
    assert f"refactory: {MESSAGE}" in capsys.readouterr().err
    assert fake_dsh.calls() == []


def test_a_headless_run_of_deterministic_nodes_needs_no_choice(tmp_path, fake_dsh, no_model_choice, capsys):
    """The README's first steps call no model, so they are not asked for one."""
    keys(no_model_choice, {"fakeprov": "fake-model"})
    assert cli.main(["ceo", "counting_pack/fixed_step", "--workdir", str(tmp_path / "run"), "--headless"]) == 0
    assert fake_dsh.routes() == [("graph", None, None)]
    assert "Using " not in capsys.readouterr().err
    assert dsh_profile.stored_model_choice() == {"provider": "", "model": ""}


def test_a_dsh_evaluator_without_a_client_runs_its_ceo_on_the_stored_choice(tmp_path, fake_dsh):
    from refactory_dsh import registry
    from refactory_dsh.evaluators import DshEvaluator

    store_model_choice(Path(os.environ["DSH_HOME"]), "chosen-prov", "chosen-model")
    mode = registry.resolve(Path(os.environ["REFACTORY_PROJECT"]), MODE)[1]()
    DshEvaluator(timeout_s=60).evaluate(mode, mode.state_dict(), tmp_path / "candidate")
    assert fake_dsh.routes()[0] == ("graph", "chosen-prov", "chosen-model")


def test_a_route_map_back_to_dsh_starts_from_the_stored_choice():
    client = DshModelClient(routes_resolved=True, worker_backend="codex", model="codex-model",
                            node_routes={"draft": {"worker_backend": "dsh"}})
    route = client.for_node("draft").route()
    assert (route["worker_backend"], route["provider"], route["model"]) == (
        "dsh", TEST_MODEL_CHOICE["provider"], TEST_MODEL_CHOICE["model"])


def test_the_readiness_probe_never_reports_a_profile_default(tmp_path, monkeypatch, no_model_choice):
    """With no route named, the probe says so; it does not read the headless profile's default."""
    monkeypatch.delenv("DSH_CLI", raising=False)
    keys(no_model_choice, {"fakeprov": "fake-model"}, harness_default="fakeprov/fake-model")
    probed = DshModelClient(routes_resolved=True).readiness(workdir=tmp_path, require_ready=False)
    assert (probed["status"], probed["reason"]) == ("diagnostic-unavailable", "no-model-chosen")
    assert "fakeprov" not in json.dumps(probed)


def test_the_first_run_choice_returns_a_stored_choice_untouched(no_model_choice):
    stored = store_model_choice(no_model_choice, "chosen-prov", "chosen-model")
    before = stored.read_bytes()
    keys(no_model_choice, {"fakeprov": "fake-model"})
    assert dsh_profile.first_run_choice() == {"provider": "chosen-prov", "model": "chosen-model"}
    assert stored.read_bytes() == before


# ── review of e9e4e8edf9: half routes, error text, read-back, guards ─────────

HALF = "No model chosen: the run names provider otherprov but no model"


@pytest.mark.parametrize("named_by", ["flag", "environment", "project file"])
def test_a_provider_named_without_a_model_is_refused_unless_it_is_the_stored_one(named_by, tmp_path, fake_dsh, project,
                                                                                 capsys, monkeypatch):
    """`--provider X` alone never borrows the stored model of another provider."""
    store_model_choice(Path(os.environ["DSH_HOME"]), "chosen-prov", "chosen-model")
    argv = ["ceo", MODE, "--workdir", str(tmp_path / "run"), "--headless"]
    if named_by == "flag":
        argv += ["--provider", "otherprov"]
    elif named_by == "environment":
        monkeypatch.setenv("REFACTORY_MODEL_PROVIDER", "otherprov")
    else:
        (project / "refactory.workers.json").write_text(json.dumps({"provider": "otherprov"}))
    assert cli.main(argv) == 1
    assert f"refactory: {HALF}" in capsys.readouterr().err
    assert fake_dsh.calls() == []


def test_the_stored_provider_named_alone_keeps_its_model_and_a_model_alone_keeps_the_provider(tmp_path, fake_dsh,
                                                                                              capsys):
    store_model_choice(Path(os.environ["DSH_HOME"]), "chosen-prov", "chosen-model")
    assert cli.main(["ceo", MODE, "--workdir", str(tmp_path / "one"), "--headless", "--provider", "chosen-prov"]) == 0
    assert {(provider, model) for _profile, provider, model in fake_dsh.routes()} == {("chosen-prov", "chosen-model")}
    assert cli.main(["ceo", MODE, "--workdir", str(tmp_path / "two"), "--headless", "--model", "other-model"]) == 0
    assert {(provider, model) for _profile, provider, model in fake_dsh.routes()[3:]} == {("chosen-prov", "other-model")}
    capsys.readouterr()


def test_doctor_and_runtime_report_a_half_route_as_the_run_does(project, capsys, monkeypatch):
    monkeypatch.delenv("DSH_CLI", raising=False)
    store_model_choice(Path(os.environ["DSH_HOME"]), "chosen-prov", "chosen-model")
    code, rows = _doctor(capsys, provider="otherprov")
    assert code == 1 and rows["run model"].startswith("MISS") and HALF in rows["run model"]
    assert "sandbox model configuration" not in rows
    assert console.main(["runtime", "--project", str(project), "--provider", "otherprov"]) == 0
    reported = json.loads(capsys.readouterr().out)
    assert (reported["model_route"]["provider"], reported["model_route"]["model"]) == ("otherprov", "")
    assert reported["nested"]["status"] == "no-model-chosen" and reported["nested"]["message"].startswith(HALF)


def test_a_codex_run_starts_its_local_ceo_on_the_stored_choice(tmp_path, fake_dsh, capsys):
    """The workers keep Codex's own settings; the CEO is a DSH agent and gets the choice."""
    store_model_choice(Path(os.environ["DSH_HOME"]), "chosen-prov", "chosen-model")
    # The fake answers in text, not in Codex's files, so the run itself may stop early.
    interactive.main([MODE, "--workdir", str(tmp_path / "run"), "--headless", "--worker-backend", "codex"])
    routes = fake_dsh.routes()
    assert routes and routes[0] == ("graph", "chosen-prov", "chosen-model"), capsys.readouterr().err
    assert all(route == ("headless", None, None) for route in routes[1:])


def test_model_set_never_prints_the_line_a_yaml_error_is_on(capsys):
    path = dsh_profile.model_choice_path()
    broken = ("- id: llm-pi-ai\n  config:\n    headers:\n"
              "      Authorization: Bearer sk-SECRETVALUE: oops\n")
    path.write_text(broken)
    assert cli.main(["model", "set", "--provider", "p1", "--model", "m1"]) == 1
    said = capsys.readouterr()
    assert "SECRETVALUE" not in said.err + said.out and "Bearer" not in said.err + said.out
    assert "at line 4, column" in said.err and "left alone" in said.err
    assert path.read_text() == broken


def test_model_set_refuses_a_file_it_cannot_edit_in_place_without_crashing(capsys):
    path = dsh_profile.model_choice_path()
    indented = "  - id: ui\n    config:\n      theme: dark\n"
    path.write_text(indented)
    assert cli.main(["model", "set", "--provider", "p1", "--model", "m1"]) == 1
    assert "could not be edited in place" in capsys.readouterr().err
    assert path.read_text() == indented


def test_model_set_refuses_an_edit_that_would_change_another_entry(monkeypatch, capsys):
    """The read-back guard: whatever the edit did, another entry must read back unchanged."""
    path = dsh_profile.model_choice_path()
    path.write_text(_GUI_PATCH)
    edit = dsh_profile._with_model_choice
    monkeypatch.setattr(dsh_profile, "_with_model_choice",
                        lambda text, provider, model: edit(text, provider, model).replace("theme: dark", "theme: light"))
    assert cli.main(["model", "set", "--provider", "p1", "--model", "m1"]) == 1
    assert "could not be edited in place" in capsys.readouterr().err
    assert path.read_text() == _GUI_PATCH


@pytest.mark.parametrize("entry", [
    "- id: agent-default-model\n  disabled: true\n  config:\n    provider: a\n    model: b\n",
    "- id: agent-default-model\n  config:\n    provider: a\n",
])
def test_a_disabled_or_half_written_entry_is_no_choice(entry, capsys):
    dsh_profile.model_choice_path().write_text(entry)
    assert dsh_profile.stored_model_choice() == {"provider": "", "model": ""}
    assert cli.main(["model", "show"]) == 1
    assert MESSAGE in capsys.readouterr().err


def test_a_first_run_choice_that_cannot_be_stored_is_a_refusal(tmp_path, fake_dsh, no_model_choice, capsys):
    keys(no_model_choice, {"fakeprov": "fake-model"})
    path = dsh_profile.model_choice_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("  - id: ui\n    config:\n      theme: dark\n")
    assert cli.main(["ceo", MODE, "--workdir", str(tmp_path / "run"), "--headless"]) == 1
    said = capsys.readouterr().err
    assert "fakeprov/fake-model could not be stored" in said and "could not be edited in place" in said
    assert fake_dsh.calls() == []


# ── `refactory runtime` names the credential source it would use ──────────────

_CREDENTIAL_SOURCES = (
    "REFACTORY_MODEL_DSH_HOME", "CLAUDE_CODE_USE_VERTEX", "ANTHROPIC_BASE_URL",
    "OPENAI_BASE_URL", "REFACTORY_MODEL_BASE_URL", "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY", "DEEPSEEK_API_KEY",
)


def _nested(capsys, project: Path, monkeypatch) -> dict:
    for name in _CREDENTIAL_SOURCES:
        monkeypatch.delenv(name, raising=False)
    assert console.main(["runtime", "--project", str(project)]) == 0
    return json.loads(capsys.readouterr().out)["nested"]


def test_runtime_names_the_credential_store_it_would_use(project, capsys, monkeypatch):
    """`authenticated: false` on a machine that runs models reads as a missing key: say
    where the client finds its credential, by path, and never a value."""
    store = Path(os.environ["DSH_HOME"]) / ".credentials.yaml"
    store.unlink(missing_ok=True)

    nested = _nested(capsys, project, monkeypatch)
    assert nested["authenticated"] is False
    assert nested["authenticated_source"] is None
    assert str(store) in nested["authenticated_checked"]
    assert "missing" in nested["message"].lower() or "unverified" in nested["message"].lower()

    store.write_text("{}")
    nested = _nested(capsys, project, monkeypatch)
    # An empty store is a source, not a credential: the probe read it and refused.
    assert nested["authenticated"] is False
    assert nested["authenticated_source"].endswith(str(store))
    assert "invalid" in nested["message"].lower()


def test_runtime_names_a_vertex_or_base_url_route_without_printing_it(project, capsys, monkeypatch):
    store = Path(os.environ["DSH_HOME"]) / ".credentials.yaml"
    store.unlink(missing_ok=True)

    nested = _nested(capsys, project, monkeypatch)
    assert nested["authenticated"] is False

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8082")
    assert console.main(["runtime", "--project", str(project)]) == 0
    nested = json.loads(capsys.readouterr().out)["nested"]
    assert nested["authenticated"] is True
    assert nested["authenticated_source"] == "base-url proxy (ANTHROPIC_BASE_URL)"
    assert "127.0.0.1" not in json.dumps(nested), "a URL is not a credential, but keep it out anyway"

    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    assert console.main(["runtime", "--project", str(project)]) == 0
    nested = json.loads(capsys.readouterr().out)["nested"]
    assert nested["authenticated_source"] == "Vertex (CLAUDE_CODE_USE_VERTEX)"


def test_credential_source_names_the_store_dotenv_routes_and_scrubbed_keys(
        tmp_path, monkeypatch):
    """Each source the client would rely on, by name or path; an exported key on its own
    is named as scrubbed, so someone with only ANTHROPIC_API_KEY set sees why."""
    from refactory_dsh.model import credential_source

    for name in _CREDENTIAL_SOURCES:
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "dsh"
    home.mkdir()
    monkeypatch.setenv("DSH_HOME", str(home))

    assert credential_source()["configured"] is False
    assert str(home / ".credentials.yaml") in credential_source()["checked"]

    (home / ".env").write_text("")
    assert credential_source()["source"] == f"dotenv {home / '.env'}"

    (home / ".env").unlink()
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("")
    assert credential_source(workdir=project)["source"] == f"dotenv {project / '.env'}"

    (project / ".env").unlink()
    # A worker's own environment reaches DSH, so an exported key runs the top-level
    # worker; only the scrubbed nested paths lose it, which the source says.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    found = credential_source()
    assert found["configured"] is True
    assert found["source"] == "environment (ANTHROPIC_API_KEY; scrubbed for nested workers)"

    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8082")
    assert credential_source()["source"] == "base-url proxy (ANTHROPIC_BASE_URL)"
