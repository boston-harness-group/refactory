"""Versioned search recipes and immutable, resolved run snapshots.

Profiles choose existing search mechanisms. The mode still owns score(), Tasks,
ports and invariants; no profile can replace their implementation.
"""
from __future__ import annotations

import hashlib
import inspect
import importlib
import json
import math
import re
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
PROFILE_ROOT = Path(".factory/profiles")
SNAPSHOT = "resolved-profile.json"
SELECTION_RULE = "max_valid_mean_score_minus_parsimony"
RUNTIME_FIELDS = {"runs_root", "resume", "progress", "print_events", "verbose",
                  "control_poll_s", "memory_global_root", "resolved_profile"}


class ProfileError(ValueError):
    """A profile cannot be resolved without changing its declared contract."""


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def defaults() -> dict[str, Any]:
    from refactory_dsh.loop import LoopConfig
    return {k: v for k, v in LoopConfig().to_dict().items() if k not in RUNTIME_FIELDS}


def normalize(settings: dict[str, Any]) -> dict[str, Any]:
    from refactory_dsh import optim, strategies
    from refactory_dsh.nn.edits import PRESETS, catalogue

    settings = dict(settings)
    if "preset" in settings:
        settings["edit_preset"] = settings.pop("preset")
    if "knob" in settings:
        knobs = dict(settings.get("knobs") or {})
        for pair in settings.pop("knob") or ():
            name, separator, value = str(pair).partition("=")
            if not separator or not name.strip():
                raise ProfileError("knob must be NAME=VALUE")
            knobs[name.strip()] = value.strip()
        settings["knobs"] = knobs
    result = defaults()
    unknown = set(settings) - set(result)
    if unknown:
        raise ProfileError(f"unknown search settings: {', '.join(sorted(unknown))}")
    result.update({key: value for key, value in settings.items() if value is not None})
    # A new recipe records its effective policy, rather than depending on a
    # future worker default. Explicit native defaults remain a distinct choice.
    result["permission_mode"] = result["permission_mode"] or "skip"
    for key in ("epochs", "batch_size", "elite", "max_evaluations", "attempts", "max_steps", "dsh_timeout_s", "history_limit"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ProfileError(f"{key} must be a positive integer")
    if not isinstance(result["repeats"], int) or not 1 <= result["repeats"] <= 10:
        raise ProfileError("repeats must be an integer from 1 to 10")
    for key in ("holdout", "parsimony", "max_elapsed_s", "max_tokens", "validation_reserve_s", "validation_reserve_tokens"):
        value = result[key]
        if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0):
            raise ProfileError(f"{key} must be a finite nonnegative number")
    if not 0 <= result["holdout"] < 1:
        raise ProfileError("holdout must be at least 0 and less than 1")
    from refactory_dsh.budgets import ResourceLimits
    try:
        ResourceLimits.from_dict(result)
    except ValueError as error:
        raise ProfileError(str(error)) from error
    for key, choices in (("optimizer", optim.names()), ("strategy", strategies.names()),
                         ("evaluator", ("walker", "dsh")), ("edit_preset", PRESETS),
                         ("worker_backend", ("dsh", "codex", "claude_code"))):
        if result[key] not in choices:
            raise ProfileError(f"unknown {key}: {result[key]}")
    for key in ("knobs", "role_routes", "node_routes"):
        if not isinstance(result[key], dict):
            raise ProfileError(f"{key} must be an object")
    from refactory_dsh.worker_config import validate_provider_config
    try:
        validate_provider_config(result.get("provider_config") or {}, origin="profile")
    except ValueError as error:
        raise ProfileError(str(error)) from error
    from refactory_dsh.workers import resolve_route, validate_route
    route = {key: result[key] for key in ("worker_backend", "provider", "model", "effort")}
    route["permission_mode"] = result["permission_mode"]
    try:
        validate_route(route)
        for role in result["role_routes"]:
            resolve_route(route, role=role, role_routes=result["role_routes"])
        for node in result["node_routes"]:
            resolve_route(route, node=node, node_routes=result["node_routes"])
    except ValueError as error:
        raise ProfileError(str(error)) from error
    for key in ("frozen", "allowed_ops"):
        if not isinstance(result[key], (list, tuple)) or any(not isinstance(v, str) for v in result[key]):
            raise ProfileError(f"{key} must be an array of names")
        result[key] = list(result[key])
    if set(result["allowed_ops"]) - {op["op"] for op in catalogue(result["edit_preset"])}:
        raise ProfileError("allowed_ops must belong to the selected edit preset")
    return result


def _name(name: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}", name):
        raise ProfileError("profile name must use letters, numbers, dots, underscores or hyphens")
    return name


def _read_profile(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    if result.get("schema_version", 1) != SCHEMA_VERSION:
        raise ProfileError("unsupported profile schema_version")
    # Earlier writers saved an empty policy to mean the native restricted
    # default. Resolve that historical meaning in memory without rewriting the
    # immutable version or broadening its next launch to the new default.
    if not result["settings"].get("permission_mode"):
        result["settings"]["permission_mode"] = "default"
    return result


def list_profiles(project: Path) -> list[dict[str, Any]]:
    return [_read_profile(path) for path in sorted((project / PROFILE_ROOT).glob("*/*.json"))]


def save(project: Path, name: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Save a new immutable version. Editing a recipe always creates a version."""
    root = project / PROFILE_ROOT / _name(name)
    root.mkdir(parents=True, exist_ok=True)
    versions = [int(p.stem) for p in root.glob("*.json") if p.stem.isdigit()]
    contract = dict(spec.get("contract") or {})
    unknown = set(contract) - {"objective", "required_artifacts", "selection_rule"}
    if unknown:
        raise ProfileError(f"unsupported contract fields: {', '.join(sorted(unknown))}; the mode owns scoring and invariants")
    if contract.get("selection_rule", SELECTION_RULE) != SELECTION_RULE:
        raise ProfileError(f"supported selection_rule is {SELECTION_RULE}; latency and cost are reported, not ranked")
    required = contract.get("required_artifacts") or []
    if not isinstance(required, list) or any(not isinstance(p, str) or Path(p).is_absolute() or ".." in Path(p).parts for p in required):
        raise ProfileError("required_artifacts must be relative paths within the candidate directory")
    from refactory_dsh.worker_config import resolve as resolve_workers
    supplied = spec.get("settings") or {}
    routing = resolve_workers(project, supplied)
    profile = {"schema_version": SCHEMA_VERSION, "name": name, "version": max(versions, default=0) + 1,
               "settings": normalize({**supplied, **routing["settings"]}),
               "contract": {"objective": str(contract.get("objective") or "Maximize the mode's immutable score"),
                            "required_artifacts": required, "selection_rule": SELECTION_RULE}}
    path = root / f"{profile['version']}.json"
    with path.open("x") as handle:
        json.dump(profile, handle, indent=2)
    return profile


def load(project: Path, name: str, version: int | None = None) -> dict[str, Any]:
    root = project / PROFILE_ROOT / _name(name)
    versions = [int(p.stem) for p in root.glob("*.json") if p.stem.isdigit()]
    if not versions or version is not None and version not in versions:
        raise ProfileError(f"profile {name!r} version {version or 'latest'} does not exist")
    return _read_profile(root / f"{version or max(versions)}.json")


def resolve(project: Path, mode_id: str, *, name: str = "", version: int | None = None,
            overrides: dict[str, Any] | None = None, validate_settings: bool = True,
            recipe: dict[str, Any] | None = None, model_client: Any = None) -> dict[str, Any]:
    from refactory_dsh import registry
    from refactory_dsh.nn import edits
    profile = recipe if recipe is not None else load(project, name, version) if name else {"settings": {}, "contract": {}}
    from refactory_dsh.worker_config import FIELDS, resolve as resolve_workers
    chosen = {**profile["settings"], **(overrides or {})}
    routing = None
    if recipe is None and validate_settings:
        routing = resolve_workers(project, {key: value for key, value in (overrides or {}).items() if key in FIELDS},
                                  profile={key: value for key, value in profile["settings"].items() if key in FIELDS})
        chosen.update(routing["settings"])
    settings = normalize(chosen) if validate_settings else {**defaults(), **chosen}
    _package, klass = registry.resolve(project, mode_id)
    mode = klass()
    state = mode.state_dict()
    for knob, value in settings["knobs"].items():
        state = edits.configure(state, knob, value)
    mode.load_state_dict(state)
    source = Path(inspect.getfile(klass))
    inputs = {}
    implementations = {}
    nodes = (state.get("graph") or {}).get("nodes") or []
    for node in nodes.values() if isinstance(nodes, dict) else nodes:
        if not isinstance(node, dict):
            continue
        for key in ("call", "task_ref", "evaluator"):
            ref = node.get(key)
            if not isinstance(ref, str) or not ref:
                continue
            try:
                if key == "task_ref":
                    from refactory_dsh.task import load_task
                    task = load_task(ref, project)
                    implementations[ref] = hashlib.sha256(Path(inspect.getfile(type(task))).read_bytes()).hexdigest()
                    inputs[f"task:{ref}"] = digest(task.instances())
                elif ":" in ref:
                    module = importlib.import_module(ref.split(":", 1)[0])
                    implementations[ref] = hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
            except (ValueError, TypeError, OSError, ImportError):
                implementations[ref] = "unavailable"
        path = node.get("source_path") if isinstance(node, dict) else None
        if path:
            absolute = Path(path) if Path(path).is_absolute() else project / path
            inputs[str(path)] = hashlib.sha256(absolute.read_bytes()).hexdigest() if absolute.is_file() else digest({str(p.relative_to(absolute)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(absolute.rglob("*")) if p.is_file()}) if absolute.is_dir() else "unavailable"
    contract = {"version": profile.get("version", 1),
                "objective": (profile.get("contract") or {}).get("objective") or settings["intent"] or "Maximize the mode's immutable score",
                "required_artifacts": sorted(set([p.path for p in mode.port_declarations()] + (profile.get("contract") or {}).get("required_artifacts", []))),
                "invariants": [{"name": i.name, "description": i.description} for i in mode.invariants()],
                "selection_rule": SELECTION_RULE,
                "fixed": "Task definitions, scoring implementation, item selection and mode invariants",
                "allowed_ops": settings["allowed_ops"] or [op["op"] for op in edits.catalogue(settings["edit_preset"])],
                "frozen": sorted(set(settings["frozen"]) | {k for k, v in state.get("knobs", {}).items() if (v.get("spec") or {}).get("frozen")}),
                "training": {"holdout": settings["holdout"], "proposal_access": "training only"},
                "termination": {k: settings[k] for k in ("epochs", "max_evaluations", "max_elapsed_s", "max_tokens", "repeats")}}
    # Bounds can be explicitly extended on resume without reusing evidence under
    # another scoring/input/model contract. They still live in the exact snapshot.
    evaluation_fields = {k: v for k, v in settings.items() if k not in (
        "epochs", "batch_size", "elite", "max_evaluations", "max_elapsed_s", "max_tokens",
        "validation_reserve_s", "validation_reserve_tokens", "attempts", "history_limit", "prior_runs")}
    route_context = runtime_identity(settings, project, model_client=model_client, uses_models=settings["optimizer"] == "autoresearch" or any(n.get("kind") == "agent" for n in (nodes.values() if isinstance(nodes, dict) else nodes) if isinstance(n, dict)))
    context = {"mode": mode_id, "seed_state": digest(state), "implementation": hashlib.sha256(source.read_bytes()).hexdigest(), "dependencies": implementations, "routing": route_context,
               "inputs": inputs, "contract": digest({k: v for k, v in contract.items() if k != "termination"}),
               "settings": digest(evaluation_fields)}
    return {"schema_version": SCHEMA_VERSION, "profile": {"name": name, "version": profile.get("version")} if name else None,
            "mode": mode_id, "settings": settings, "contract": contract,
            "worker_configuration": routing,
            "evaluation_context": context, "context_hash": digest(context)}


def refresh(project: Path, mode_id: str, settings: dict[str, Any], previous: dict[str, Any], *, model_client: Any = None) -> dict[str, Any]:
    """Recheck source/input identity without relying on a still-existing recipe."""
    recipe = {"settings": {}, "contract": previous.get("contract") or {},
              "version": (previous.get("contract") or {}).get("version", 1)}
    result = resolve(project, mode_id, overrides=settings, recipe=recipe, validate_settings=False, model_client=model_client)
    result["profile"] = previous.get("profile")
    result["worker_configuration"] = previous.get("worker_configuration")
    return result


def runtime_identity(settings: dict[str, Any], project: Path, *, uses_models: bool, model_client: Any = None) -> dict[str, Any]:
    """Read nonsecret effective configuration; this is not authentication evidence."""
    if not uses_models:
        return {"status": "no_model_required_by_current_graph_or_optimizer"}
    from refactory_dsh.model import DshModelClient
    model = model_client if callable(getattr(model_client, "route", None)) and callable(getattr(model_client, "readiness", None)) else DshModelClient(routes_resolved=True, **{key: settings[key] for key in
        ("worker_backend", "provider", "model", "effort", "permission_mode", "role_routes", "node_routes")
        })
    readiness = model.readiness(workdir=project, require_ready=False)
    route = model.route()
    for key in ("provider", "model"):
        value = readiness.get(key)
        if (not route.get(key) and readiness.get("status") != "diagnostic-unavailable"
                and isinstance(value, str) and value.strip() and value != "(runtime default)"):
            route[key] = value
    resolved = bool(route.get("model")) and (route["worker_backend"] != "dsh" or bool(route.get("provider")))
    context = model.context() if callable(getattr(model, "context", None)) else {}
    # Runtime/credential-store location can change a default route or executable.
    # Keep this nonsecret context, but never bind evidence to a per-run ledger path.
    context = {key: context[key] for key in ("dsh_home", "runtime_root", "dsh_cli", "python_bin", "timeout_s", "attempts", "backoff_s") if key in context}
    return {"status": "configured_unverified" if resolved else "runtime_defaults_unresolved", "route": route,
            "client_context": context}


def add_arguments(parser: Any) -> None:
    """All launch surfaces parse this same optional configuration vocabulary."""
    for key, value in defaults().items():
        flag = "--" + ("preset" if key == "edit_preset" else key.replace("_", "-"))
        # provider_config travels in the project file and saved profile
        # snapshots, not as a command-line object.
        if key in ("knobs", "frozen", "allowed_ops", "role_routes", "node_routes", "provider_config"):
            continue
        kind = float if key in ("holdout", "parsimony", "max_elapsed_s", "validation_reserve_s") else int if isinstance(value, int) or key == "max_tokens" else str
        parser.add_argument(flag, dest=key, type=kind, default=None)
    parser.add_argument("--knob", action="append", default=None, metavar="NAME=VALUE")
    parser.add_argument("--frozen", action="append", default=None, metavar="KNOB")
    parser.add_argument("--allowed-ops", action="append", default=None, metavar="OP")
    for key in ("role_routes", "node_routes"):
        parser.add_argument("--" + key.replace("_", "-"), type=json.loads, default=None)
    parser.add_argument("--profile", default="")
    parser.add_argument("--profile-version", type=int, default=None)
