"""Profiles bind real settings, measurements and immutable validation contracts."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from refactory_dsh import console, profiles
from refactory_dsh.adopt import _score_by_id
from refactory_dsh.loop import Loop, LoopConfig, LoopError
from refactory_dsh.nn.mode import RunResult
from refactory_dsh.optim import Proposal
from refactory_dsh.outerloop import Run
from test_loop import _BoundedInt, _patch_mode
from test_loop_holdout import _run, Probe


@pytest.mark.usefixtures("no_model_choice")
@pytest.mark.parametrize("backend", ["dsh", "codex", "claude_code"])
def test_runtime_default_placeholders_never_become_resolved_routes(monkeypatch, tmp_path, backend):
    from refactory_dsh.model import DshModelClient
    monkeypatch.setattr(DshModelClient, "readiness", lambda *args, **kwargs: {
        "status": "diagnostic-unavailable", "provider": "(runtime default)", "model": "(runtime default)",
    })
    identity = profiles.runtime_identity({**profiles.defaults(), "worker_backend": backend}, tmp_path, uses_models=True)
    assert identity["status"] == "runtime_defaults_unresolved"
    assert identity["route"]["model"] == ""
    assert identity["route"]["provider"] == ""


def test_runtime_identity_preserves_explicit_route_and_never_reads_a_profile_default(monkeypatch, tmp_path, no_model_choice):
    """dm-258: the route a probe reports is the profile default; the identity never takes it."""
    from refactory_dsh import dsh_profile
    from refactory_dsh.model import DshModelClient
    from tests_support import store_model_choice
    monkeypatch.setattr(DshModelClient, "readiness", lambda *args, **kwargs: {
        "status": "configured", "provider": "configured-provider", "model": "configured-model",
    })
    identity = profiles.runtime_identity(profiles.defaults(), tmp_path, uses_models=True)
    assert identity["status"] == "runtime_defaults_unresolved"
    assert (identity["route"]["provider"], identity["route"]["model"]) == ("", "")
    # With a choice stored, the unset route is that choice, still not the probe's.
    store_model_choice(no_model_choice, "chosen-provider", "chosen-model")
    assert dsh_profile.stored_model_choice()["provider"] == "chosen-provider"
    identity = profiles.runtime_identity(profiles.defaults(), tmp_path, uses_models=True)
    assert identity["status"] == "configured_unverified"
    assert (identity["route"]["provider"], identity["route"]["model"]) == ("chosen-provider", "chosen-model")
    monkeypatch.setattr(DshModelClient, "readiness", lambda *args, **kwargs: {
        "status": "diagnostic-unavailable", "provider": "(runtime default)", "model": "(runtime default)",
    })
    identity = profiles.runtime_identity({**profiles.defaults(), "worker_backend": "codex", "model": "explicit-model"}, tmp_path, uses_models=True)
    assert identity["status"] == "configured_unverified"
    assert identity["route"]["model"] == "explicit-model"


def test_versions_are_immutable_and_console_resolves_exactly(tmp_path, monkeypatch):
    first = profiles.save(tmp_path, "bounded", {"settings": {"optimizer": "random", "repeats": 2}, "contract": {"objective": "Original"}})
    second = profiles.save(tmp_path, "bounded", {"settings": {"optimizer": "random", "repeats": 3}, "contract": {"objective": "Explicit change"}})
    assert first["version"] == 1 and second["version"] == 2
    assert profiles.load(tmp_path, "bounded", 1)["contract"]["objective"] == "Original"
    expected = profiles.resolve(tmp_path, "circle_packing/adaptive", name="bounded", version=1)
    spawned = []
    monkeypatch.setattr(console, "_spawn", lambda mode, root, config, **kwargs: spawned.append(config) or SimpleNamespace(pid=1))
    receipt = console.start("circle_packing/adaptive", project=tmp_path, run_id="gui", profile="bounded", profile_version=1)
    assert json.loads(Path(spawned[0]["resolved_config"]).read_text()) == expected == receipt["resolved_profile"]


@pytest.mark.parametrize("settings, expected", [({}, "skip"), ({"permission_mode": ""}, "skip"),
                                               ({"permission_mode": "default"}, "default")])
def test_new_profiles_persist_the_effective_permission_policy(tmp_path, settings, expected):
    normalized = profiles.normalize(settings)
    assert normalized["permission_mode"] == expected
    saved = profiles.save(tmp_path, "permission-policy", {"settings": settings})
    assert saved["settings"]["permission_mode"] == expected
    assert profiles.load(tmp_path, "permission-policy")["settings"]["permission_mode"] == expected


@pytest.mark.parametrize("legacy_policy", ["", None])
def test_legacy_profile_policy_keeps_native_defaults_without_rewriting(tmp_path, legacy_policy):
    saved = profiles.save(tmp_path, "historical", {"settings": {"optimizer": "random", "permission_mode": "default"}})
    # Reproduce the earlier saved representation, including profiles predating
    # the optional field. This is historical input, not a new recipe request.
    if legacy_policy is None:
        saved["settings"].pop("permission_mode")
    else:
        saved["settings"]["permission_mode"] = legacy_policy
    path = tmp_path / profiles.PROFILE_ROOT / "historical/1.json"
    path.write_text(json.dumps(saved, indent=2))
    before = path.read_bytes()
    assert profiles.load(tmp_path, "historical", 1)["settings"]["permission_mode"] == "default"
    assert profiles.list_profiles(tmp_path)[0]["settings"]["permission_mode"] == "default"
    assert profiles.resolve(tmp_path, "circle_packing/adaptive", name="historical", version=1)["settings"]["permission_mode"] == "default"
    updated = profiles.save(tmp_path, "historical", {"settings": {"optimizer": "random"}})
    assert updated["version"] == 2 and updated["settings"]["permission_mode"] == "skip"
    assert path.read_bytes() == before


def test_permission_policy_changes_evaluation_identity(tmp_path, monkeypatch):
    from refactory_dsh.model import DshModelClient
    monkeypatch.setattr(DshModelClient, "readiness", lambda *args, **kwargs: {
        "status": "configured", "provider": "explicit-provider", "model": "explicit-model",
    })
    settings = {"optimizer": "autoresearch", "provider": "explicit-provider", "model": "explicit-model"}
    legacy = profiles.resolve(tmp_path, "circle_packing/adaptive", overrides={**settings, "permission_mode": "default"})
    current = profiles.resolve(tmp_path, "circle_packing/adaptive", overrides=settings)
    assert legacy["evaluation_context"]["routing"]["route"]["permission_mode"] == "default"
    assert current["evaluation_context"]["routing"]["route"]["permission_mode"] == "skip"
    assert legacy["context_hash"] != current["context_hash"]


@pytest.mark.parametrize("legacy_policy", ["default", ""])
@pytest.mark.parametrize("route_kind", ["role_routes", "node_routes"])
def test_historical_backend_switch_keeps_policy_and_cache_identity(tmp_path, monkeypatch, legacy_policy, route_kind):
    from refactory_dsh.model import DshModelClient
    monkeypatch.delenv("REFACTORY_WORKER_CONTEXT", raising=False)
    monkeypatch.setattr(DshModelClient, "readiness", lambda *args, **kwargs: {
        "status": "configured", "provider": "explicit-provider", "model": "explicit-model",
    })
    saved = profiles.save(tmp_path, "routed", {"settings": {
        "optimizer": "autoresearch", "provider": "explicit-provider", "model": "explicit-model",
        "permission_mode": "default", route_kind: {"builder": {"worker_backend": "codex"}},
    }})
    saved["settings"]["permission_mode"] = legacy_policy
    path = tmp_path / profiles.PROFILE_ROOT / "routed/1.json"
    path.write_text(json.dumps(saved))
    before = path.read_bytes()
    snapshot = profiles.resolve(tmp_path, "circle_packing/adaptive", name="routed")
    settings = snapshot["settings"]
    client = DshModelClient(provider=settings["provider"], model=settings["model"],
                            permission_mode=settings["permission_mode"], **{route_kind: settings[route_kind]})
    assert client.for_node("builder", "builder").route()["permission_mode"] == "default"
    refreshed = profiles.refresh(tmp_path, "circle_packing/adaptive", settings, snapshot, model_client=client)
    assert refreshed["context_hash"] == snapshot["context_hash"]
    changed = profiles.resolve(tmp_path, "circle_packing/adaptive", name="routed", overrides={
        route_kind: {"builder": {"worker_backend": "codex", "permission_mode": "skip"}},
    })
    assert changed["context_hash"] != snapshot["context_hash"]
    assert path.read_bytes() == before


def test_legacy_blank_snapshot_refuses_reuse_under_new_policy_without_rewriting(tmp_path, monkeypatch):
    from refactory_dsh.model import DshModelClient
    readiness = lambda *args, **kwargs: {
        "status": "configured", "provider": "explicit-provider", "model": "explicit-model",
    }
    monkeypatch.setattr(DshModelClient, "readiness", readiness)
    # The historical implicit route was restricted, while the saved settings
    # contained an empty policy. The client deliberately has no context() method
    # so that the actual old representation is retained by this fixture.
    legacy_route = {"worker_backend": "dsh", "provider": "explicit-provider", "model": "explicit-model",
                    "effort": "", "permission_mode": "default"}
    legacy_client = SimpleNamespace(route=lambda: dict(legacy_route), readiness=readiness)
    config = LoopConfig(epochs=1, batch_size=1, optimizer="autoresearch", permission_mode="", runs_root=tmp_path)
    first = Loop("circle_packing/adaptive", config, project=tmp_path, model=legacy_client)
    first.init()
    first.forward()  # Deterministic graph; no model request or proposal.
    path = tmp_path / profiles.SNAPSHOT
    before = path.read_bytes()
    assert json.loads(before)["settings"]["permission_mode"] == ""
    config.resume = True
    with pytest.raises(LoopError, match="evaluation context changed"):
        Loop("circle_packing/adaptive", config, project=tmp_path,
             model=DshModelClient(provider="explicit-provider", model="explicit-model")).init()
    assert path.read_bytes() == before


@pytest.mark.parametrize("settings", [{"repeats": 0}, {"repeats": 11}, {"holdout": 1}, {"max_tokens": -1}, {"unknown": 1}, {"allowed_ops": ["remove_node"], "edit_preset": "knob_only"}])
def test_invalid_contract_refused(settings, tmp_path):
    with pytest.raises(profiles.ProfileError):
        profiles.save(tmp_path, "bad", {"settings": settings})


def test_repeats_measure_training_and_holdout_without_leaking_audit(tmp_path):
    loop, report = _run(tmp_path, repeats=3)
    assert report["evaluations"] == 1 and report["measurements"] == 6
    assert report["best"]["measurements"]["count"] == 3
    assert report["best"]["measurements"]["stdev"] == 0
    assert report["holdout"]["measurements"]["count"] == 3
    assert report["holdout"]["holdout"] == 0
    assert "holdout" not in vars(loop._context())


def test_failed_trial_never_becomes_a_winner(tmp_path):
    with _patch_mode(_BoundedInt):
        loop = Loop("fake/bounded", LoopConfig(epochs=1, batch_size=1, optimizer="random", repeats=3, runs_root=tmp_path), project=Path("."))
        values = iter([RunResult(score=2), RunResult(score=0, exit_ok=False, error="control failure")])
        loop.evaluator = SimpleNamespace(evaluate=lambda *args: next(values))
        result = loop.run()
    candidate = result["failed_candidates"][0]
    assert candidate["score"] is None and candidate["raw_score"] is None
    assert candidate["measurements"]["count"] == 2
    assert candidate["measurements"]["successful"] == 1
    assert candidate["measurements"]["tokens"] is None
    assert result["best"] is None and result["exit_ok"] is False


def test_required_artifact_is_enforced_and_reported(tmp_path):
    profiles.save(tmp_path, "required", {"settings": {"optimizer": "random", "epochs": 1, "batch_size": 1}, "contract": {"required_artifacts": ["required.txt"]}})
    snapshot = profiles.resolve(tmp_path, "circle_packing/adaptive", name="required")
    config = LoopConfig.from_dict({**snapshot["settings"], "runs_root": tmp_path / "run", "resolved_profile": snapshot})
    report = Loop("circle_packing/adaptive", config, project=tmp_path).run()
    assert report["best"] is None
    assert report["failed_candidates"][0]["measurements"]["trials"][0]["required_contract"]["missing_artifacts"] == ["required.txt"]


@pytest.mark.parametrize("score", [None, float("nan"), float("inf"), "bad"])
def test_adoption_rejects_unmeasured_scores_even_with_success_flag(score):
    assert _score_by_id([{"event": "score", "id": "broken", "exit_ok": True, "score": score}]) == {}


def test_preset_is_a_runtime_boundary(tmp_path):
    loop = Loop("circle_packing/adaptive", LoopConfig(optimizer="random", edit_preset="knob_only", runs_root=tmp_path), project=Path("."))
    loop.init()
    result = Run(loop).attempt(Proposal(edit={"op": "remove_node", "id": "accept"}), loop.population[0])
    assert isinstance(result, list) and result[0].startswith("E-policy")


def test_frozen_knob_cannot_be_removed_by_another_edit(tmp_path):
    loop = Loop("circle_packing/adaptive", LoopConfig(optimizer="random", frozen=("acceptance_mode",), runs_root=tmp_path), project=Path("."))
    loop.init()
    result = Run(loop).attempt(Proposal(edit={"op": "remove_knob", "name": "acceptance_mode"}), loop.population[0])
    assert isinstance(result, list) and result[0].startswith("E-frozen")


def test_old_configs_remain_readable():
    old = LoopConfig.from_dict({"epochs": 1, "runs_root": "/tmp/old", "future_field": True})
    assert old.repeats == 1 and old.worker_backend == "dsh"


def test_repeat_usage_includes_winner_validation(tmp_path):
    with _patch_mode(Probe):
        loop = Loop("fake/probe", LoopConfig(epochs=1, batch_size=1, optimizer="random", repeats=2,
                    max_tokens=9, validation_reserve_tokens=2, runs_root=tmp_path), project=Path("."))
        def evaluate(mode, state, workdir):
            loop.budget.record_usage(str(workdir), 2)
            from refactory_dsh import task as _task
            subset = _task.read_subset(workdir)
            part = subset.get("part", "train")
            expected, _ = loop._expected_item_ids(state, part)
            scores = [{"id": eid, "passed": True, "score": 1.0} for eid in expected]
            return RunResult(score=1, facts={"item_scores": scores} if scores else {})
        loop.evaluator = SimpleNamespace(evaluate=evaluate)
        report = loop.run()
    assert report["exit_ok"] is True
    assert report["resources"]["tokens"] == 8
    assert report["measurements"] == 4
    assert report["holdout"]["measurements"]["tokens"] == 4


def test_exhausted_trial_stays_incomplete_and_unselectable(tmp_path):
    with _patch_mode(_BoundedInt):
        loop = Loop("fake/bounded", LoopConfig(epochs=1, batch_size=1, optimizer="random", repeats=3,
                    max_tokens=7, validation_reserve_tokens=2, runs_root=tmp_path), project=Path("."))
        def evaluate(mode, state, workdir):
            loop.budget.record_usage(str(workdir), 3)
            return RunResult(score=10)
        loop.evaluator = SimpleNamespace(evaluate=evaluate)
        report = loop.run()
    assert report["best"] is None and report["exit_ok"] is False
    assert report["resources"]["tokens"] == 6
    assert report["failed_candidates"][0]["measurements"]["count"] == 2


def test_resume_refreshes_context_instead_of_trusting_old_snapshot(tmp_path):
    config = LoopConfig(epochs=1, batch_size=1, optimizer="random", runs_root=tmp_path)
    Loop("circle_packing/adaptive", config, project=Path(".")).run()
    config.resume = True
    config.model = "changed-model"
    with pytest.raises(LoopError, match="evaluation context changed"):
        Loop("circle_packing/adaptive", config, project=Path(".")).init()


def test_resume_bound_extension_preserves_original_snapshot(tmp_path):
    config = LoopConfig(epochs=1, batch_size=1, optimizer="random", runs_root=tmp_path)
    Loop("circle_packing/adaptive", config, project=Path(".")).run()
    before = (tmp_path / profiles.SNAPSHOT).read_bytes()
    config.resume = True
    config.epochs = 2
    Loop("circle_packing/adaptive", config, project=Path(".")).init()
    assert (tmp_path / profiles.SNAPSHOT).read_bytes() == before
    assert 'resume_configuration' in (tmp_path / "journal.jsonl").read_text()


def test_proposer_uses_shared_budget_and_owned_directory(tmp_path, monkeypatch):
    import copy
    class Model:
        def with_budget(self, budget):
            client = copy.copy(self)
            client._budget = budget
            return client
        def complete(self, prompt, *, system="", workdir=None):
            assert workdir == tmp_path / "proposer"
            self._budget.record_usage("proposal", 3)
            return '{"proposals": []}'
    monkeypatch.setattr(profiles, "runtime_identity", lambda *args, **kwargs: {"status": "deterministic-test-client"})
    with _patch_mode(_BoundedInt):
        loop = Loop("fake/bounded", LoopConfig(epochs=1, batch_size=1, optimizer="autoresearch", max_tokens=10, runs_root=tmp_path), project=Path("."), model=Model())
        report = loop.run()
    assert report["resources"]["tokens"] == 3
    assert loop.optimizer._client._budget is loop.budget


def test_fresh_search_refuses_an_old_paused_resource_ledger(tmp_path):
    config = LoopConfig(epochs=1, batch_size=1, optimizer="random", runs_root=tmp_path)
    Loop("circle_packing/adaptive", config, project=Path(".")).run()
    with pytest.raises(LoopError, match="use --resume"):
        Loop("circle_packing/adaptive", config, project=Path(".")).run()


def test_explicit_resource_extension_keeps_previous_usage(tmp_path):
    config = LoopConfig(epochs=1, batch_size=1, optimizer="random", max_tokens=5, runs_root=tmp_path)
    first = Loop("circle_packing/adaptive", config, project=Path("."))
    first.run()
    first.budget.record_usage("previous", 4)
    config.resume, config.max_tokens = True, 10
    result = Loop("circle_packing/adaptive", config, project=Path(".")).run()
    assert result["resources"]["tokens"] == 4
    assert result["resources"]["limits"]["max_tokens"] == 10


@pytest.mark.parametrize("changed", ["model", "dsh_home", "node_routes", "permission_mode"])
def test_supplied_client_context_is_recorded_and_changed_context_refuses_reuse(tmp_path, monkeypatch, changed):
    from refactory_dsh.model import DshModelClient
    monkeypatch.setattr(DshModelClient, "readiness", lambda *args, **kwargs: {
        "status": "configured", "provider": "runtime-provider", "model": "runtime-default",
    })
    profiles.save(tmp_path, "fixed", {"settings": {"optimizer": "autoresearch", "batch_size": 1},
                                    "contract": {"objective": "Keep the declared requirement", "required_artifacts": []}})
    snapshot = profiles.resolve(tmp_path, "circle_packing/adaptive", name="fixed")
    config = LoopConfig.from_dict({**snapshot["settings"], "runs_root": tmp_path / "run", "resolved_profile": snapshot})
    client_args = {"provider": "selected-provider", "model": "selected-model", "effort": "high",
                   "dsh_home": tmp_path / "selected-home", "node_routes": {"optimizer": {"model": "proposer-model"}},
                   "permission_mode": "default"}
    first = Loop("circle_packing/adaptive", config, project=tmp_path, model=DshModelClient(**client_args))
    first.init()
    first.forward()  # The shipped circle graph is deterministic; no worker is dispatched.
    recorded = first.config.resolved_profile
    assert recorded["settings"]["model"] == "selected-model"
    assert recorded["settings"]["node_routes"] == client_args["node_routes"]
    assert recorded["settings"]["permission_mode"] == "default"
    assert recorded["evaluation_context"]["routing"]["route"]["model"] == "selected-model"
    assert recorded["evaluation_context"]["routing"]["client_context"]["dsh_home"] == str(client_args["dsh_home"])
    assert recorded["contract"]["objective"] == "Keep the declared requirement"
    assert recorded["profile"] == {"name": "fixed", "version": 1}
    before = (config.runs_root / profiles.SNAPSHOT).read_bytes()
    config.resume = True
    changes = {"model": "another-model", "dsh_home": tmp_path / "another-home",
               "node_routes": {"optimizer": {"model": "another-proposer"}}, "permission_mode": "skip"}
    client_args[changed] = changes[changed]
    with pytest.raises(LoopError, match="evaluation context changed"):
        Loop("circle_packing/adaptive", config, project=tmp_path, model=DshModelClient(**client_args)).init()
    assert (config.runs_root / profiles.SNAPSHOT).read_bytes() == before
