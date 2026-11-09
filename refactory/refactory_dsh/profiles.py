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


#: The list-valued command-line options that add to a mapping setting, one
#: ``NAME=VALUE`` pair each, on top of what a saved profile already holds.
PAIR_OPTIONS = {"knob": "knobs", "strategy_option": "strategy_options"}


def cli_overrides(args: Any) -> dict[str, Any]:
    """The settings a command line gave explicitly, to override a profile's.

    Every launch surface reads its parsed arguments through this, so none of them
    drops a setting a profile holds or adds one the command line left out."""
    known = defaults()
    return {key: value for key, value in vars(args).items()
            if (key in known or key in PAIR_OPTIONS) and value is not None}


def _type_problem(key: str, value: Any, hint: Any) -> str:
    """Why ``value`` cannot be the setting ``key`` of type ``hint``; empty when it can."""
    import types
    import typing

    origin, members = typing.get_origin(hint), typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        problems = [_type_problem(key, value, member) for member in members]
        return "" if any(not problem for problem in problems) else " or ".join(p for p in problems if p)
    if hint is type(None):
        return "" if value is None else "null"
    if hint is bool:
        # Switches arrive as 0 or 1 from the command line.
        return "" if value in (True, False, 0, 1) and not isinstance(value, float) else "true or false"
    if hint is int:
        return "" if isinstance(value, int) and not isinstance(value, bool) else "an integer"
    if hint is float:
        return "" if isinstance(value, (int, float)) and not isinstance(value, bool) else "a number"
    if hint is str:
        return "" if isinstance(value, str) else "a string"
    if origin is dict or hint is dict:
        return "" if isinstance(value, dict) else "an object"
    if origin is tuple or hint is tuple:
        return "" if isinstance(value, (list, tuple)) else "an array"
    return ""


def _check_types(settings: dict[str, Any]) -> None:
    """Refuse a setting whose value is not of its ``LoopConfig`` field's type, so a
    string seed or a ``true`` repeat count is refused rather than carried into a run.

    Values are checked, never rewritten: a switch given as 1 stays 1, because the
    settings digest of a run started that way must not change on resume."""
    import dataclasses
    import typing

    from refactory_dsh.loop import LoopConfig

    hints = typing.get_type_hints(LoopConfig)
    for field in dataclasses.fields(LoopConfig):
        if field.name not in settings or field.name in RUNTIME_FIELDS:
            continue
        problem = _type_problem(field.name, settings[field.name], hints[field.name])
        if problem:
            raise ProfileError(f"{field.name} must be {problem}, not {settings[field.name]!r}")


def normalize(settings: dict[str, Any]) -> dict[str, Any]:
    from refactory_dsh import optim, strategies
    from refactory_dsh.nn.edits import PRESETS, catalogue, removed

    settings = dict(settings)
    if "preset" in settings:
        settings["edit_preset"] = settings.pop("preset")
    for option, target in PAIR_OPTIONS.items():
        # `--knob NAME=VALUE` and `--strategy-option KEY=VALUE` add to the mapping a
        # profile holds instead of replacing it.
        if option not in settings:
            continue
        pairs = settings.pop(option) or ()
        merged = dict(settings.get(target) or {})
        for pair in pairs:
            name, separator, value = str(pair).partition("=")
            if not separator or not name.strip():
                raise ProfileError(f"{option} must be NAME=VALUE, got {pair!r}")
            if option == "strategy_option":
                from refactory_dsh.loop import _coerce_strategy_value
                merged[name.strip()] = _coerce_strategy_value(value.strip())
            else:
                merged[name.strip()] = value.strip()
        settings[target] = merged
    result = defaults()
    unknown = set(settings) - set(result)
    if unknown:
        raise ProfileError(f"unknown search settings: {', '.join(sorted(unknown))}")
    result.update({key: value for key, value in settings.items() if value is not None})
    _check_types(result)
    # A new recipe records its effective policy, rather than depending on a
    # future worker default. Explicit native defaults remain a distinct choice.
    result["permission_mode"] = result["permission_mode"] or "skip"
    for key in ("epochs", "batch_size", "elite", "max_evaluations", "attempts", "max_steps", "dsh_timeout_s", "history_limit", "max_concurrent_evaluations"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ProfileError(f"{key} must be a positive integer")
    if not isinstance(result["repeats"], int) or not 1 <= result["repeats"] <= 10:
        raise ProfileError("repeats must be an integer from 1 to 10")
    for key in ("holdout", "parsimony", "max_elapsed_s", "max_tokens", "validation_reserve_s", "validation_reserve_tokens", "max_rollouts"):
        value = result.get(key)
        if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0):
            raise ProfileError(f"{key} must be a finite nonnegative number")
    # val_gate: a switch; the command line passes it as 0 or 1.
    if result.get("val_gate") not in (True, False, 0, 1):
        raise ProfileError("val_gate must be true or false (1 or 0 on the command line)")
    result["val_gate"] = bool(result.get("val_gate"))
    # The default loop's early stop (needs val_gate): a count of stale epochs, 0 off,
    # and a switch the command line passes as 0 or 1.
    patience = result.get("early_stop_patience")
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 0:
        raise ProfileError("early_stop_patience must be a nonnegative integer (0 is off)")
    if result.get("early_stop_at_full_val") not in (True, False, 0, 1):
        raise ProfileError("early_stop_at_full_val must be true or false (1 or 0 on the command line)")
    result["early_stop_at_full_val"] = bool(result.get("early_stop_at_full_val"))
    # allow_new_node_tools: a switch; the command line passes it as 0 or 1.
    if result.get("allow_new_node_tools") not in (True, False, 0, 1, None):
        raise ProfileError("allow_new_node_tools must be true or false (1 or 0 on the command line)")
    result["allow_new_node_tools"] = bool(result.get("allow_new_node_tools"))
    # allow_code_nodes: a switch, on unless turned off; the command line passes 0 or 1.
    if result.get("allow_code_nodes") not in (True, False, 0, 1, None):
        raise ProfileError("allow_code_nodes must be true or false (1 or 0 on the command line)")
    result["allow_code_nodes"] = result.get("allow_code_nodes") not in (False, 0)
    # fresh_evidence: a switch; the command line passes it as 0 or 1.
    if result.get("fresh_evidence") not in (True, False, 0, 1):
        raise ProfileError("fresh_evidence must be true or false (1 or 0 on the command line)")
    result["fresh_evidence"] = bool(result.get("fresh_evidence"))
    max_edits = result.get("max_edits_per_proposal")
    if isinstance(max_edits, bool) or not isinstance(max_edits, int) or max_edits < 1:
        raise ProfileError("max_edits_per_proposal must be a positive integer (1 is single edits)")
    minibatch = result.get("train_minibatch")
    if isinstance(minibatch, bool) or not isinstance(minibatch, int) or minibatch < 0:
        raise ProfileError("train_minibatch must be a nonnegative integer (0 is the whole train part)")
    if result["val_gate"]:
        from refactory_dsh.loop import val_gate_refusal
        refusal = val_gate_refusal(str(result.get("optimizer")), str(result.get("strategy")),
                                   str(result.get("parent_picker") or ""))
        if refusal:
            raise ProfileError(refusal)
    # max_items_in_flight: None or positive int
    mif = result.get("max_items_in_flight")
    if mif is not None:
        if isinstance(mif, bool) or not isinstance(mif, int) or mif < 1:
            raise ProfileError("max_items_in_flight must be a positive integer or None")
    if not 0 <= result["holdout"] < 1:
        raise ProfileError("holdout must be at least 0 and less than 1")
    from refactory_dsh.budgets import ResourceLimits
    try:
        ResourceLimits.from_dict(result)
    except ValueError as error:
        raise ProfileError(str(error)) from error
    # A package-contributed strategy (``Package.strategies``) is looked up only
    # when the name is not one of the library's, so the common case imports nothing.
    strategy_choices = (strategies.names() if result["strategy"] in strategies.STRATEGIES
                        else strategies.available())
    for key, choices in (("optimizer", optim.names()), ("strategy", strategy_choices),
                         ("evaluator", ("walker", "dsh")), ("edit_preset", PRESETS),
                         ("worker_backend", ("dsh", "codex", "claude_code"))):
        if result[key] not in choices:
            raise ProfileError(f"unknown {key}: {result[key]}")
    picker = result.get("parent_picker") or ""
    if picker:
        if picker not in optim.picker_names():
            raise ProfileError(f"unknown parent_picker: {picker}; known: {', '.join(optim.picker_names())}")
        if result["optimizer"] in optim.picker_names():
            raise ProfileError(
                f"the {result['optimizer']} optimizer picks parents itself; pair the "
                f"{picker} parent picker with a proposer (autoresearch or random)")
    result["parent_picker"] = picker
    from refactory_dsh.loop import PORTFOLIOS
    portfolio = result.get("portfolio") or ""
    if portfolio not in PORTFOLIOS:
        raise ProfileError(f"unknown portfolio: {portfolio}; known: {', '.join(p for p in PORTFOLIOS if p)}")
    if portfolio and result["optimizer"] != "autoresearch":
        raise ProfileError(f"portfolio {portfolio} asks the model proposer for one candidate per "
                           "edit level; use optimizer autoresearch")
    result["portfolio"] = portfolio
    for key in ("knobs", "role_routes", "node_routes", "strategy_options"):
        if not isinstance(result[key], dict):
            raise ProfileError(f"{key} must be an object")
    try:
        strategies.check_options(result["strategy"], result["strategy_options"])
    except ValueError as error:
        raise ProfileError(str(error)) from error
    from refactory_dsh.worker_config import validate_provider_config
    try:
        validate_provider_config(result.get("provider_config") or {}, origin="profile")
    except ValueError as error:
        raise ProfileError(str(error)) from error
    from refactory_dsh.workers import resolve_route, validate_route
    route = {key: result[key] for key in ("worker_backend", "provider", "model", "effort")}
    route["permission_mode"] = result["permission_mode"]
    if result.get("setting_sources"):
        route["setting_sources"] = result["setting_sources"]
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
    for op in result["allowed_ops"]:
        if removed(op) is not None:
            raise ProfileError(f"allowed_ops names {op!r}, which no longer exists: {removed(op)}")
    outside = set(result["allowed_ops"]) - {op["op"] for op in catalogue(result["edit_preset"])}
    if outside:
        # The selected strategy's own required ops are permitted with any preset,
        # as ``Loop.permitted_ops`` adds them (#493 G34).
        outside -= set(getattr(strategies.resolve(result["strategy"]), "required_ops", ()))
    if outside:
        raise ProfileError("allowed_ops must belong to the selected edit preset "
                           "or the selected strategy's required ops")
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
    if not isinstance(spec, dict):
        raise ProfileError("a profile spec is an object with settings and contract")
    unknown = set(spec) - {"settings", "contract"}
    if unknown:
        # A misspelled section would otherwise save a profile of defaults.
        raise ProfileError(f"unknown profile fields: {', '.join(sorted(unknown))}; "
                           "a profile has settings and contract")
    if not isinstance(spec.get("settings") or {}, dict) or not isinstance(spec.get("contract") or {}, dict):
        raise ProfileError("a profile's settings and contract are objects")
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
    profile = {"schema_version": SCHEMA_VERSION, "name": name, "version": 0,
               "settings": normalize({**supplied, **routing["settings"]}),
               "contract": {"objective": str(contract.get("objective") or "Maximize the mode's immutable score"),
                            "required_artifacts": required, "selection_rule": SELECTION_RULE}}
    # Only a spec that validated creates the profile's directory, and a version is
    # never overwritten: a concurrent save that took the number takes the next one.
    root.mkdir(parents=True, exist_ok=True)
    while True:
        versions = [int(p.stem) for p in root.glob("*.json") if p.stem.isdigit()]
        profile["version"] = max(versions, default=0) + 1
        try:
            with (root / f"{profile['version']}.json").open("x") as handle:
                json.dump(profile, handle, indent=2)
        except FileExistsError:
            continue
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
    from refactory_dsh import node_retry, registry
    from refactory_dsh.nn import edits
    profile = recipe if recipe is not None else load(project, name, version) if name else {"settings": {}, "contract": {}}
    from refactory_dsh.worker_config import FIELDS, resolve as resolve_workers
    overrides = dict(overrides or {})
    if (name and overrides.get("strategy") not in (None, profile["settings"].get("strategy"))
            and "strategy_options" not in overrides and profile["settings"].get("strategy_options")):
        # The profile's strategy options belong to the profile's strategy: a launch that
        # names another strategy starts from none, and the snapshot records the drop.
        overrides["strategy_options"] = {}
    chosen = {**profile["settings"], **overrides}
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
        "validation_reserve_s", "validation_reserve_tokens", "attempts", "history_limit", "prior_runs",
        "max_concurrent_evaluations", "max_items_in_flight", "val_gate",
        "early_stop_patience", "early_stop_at_full_val", "train_minibatch",
        "fresh_evidence", "max_edits_per_proposal")}
    # A run that sets no settings sources digests as it did before the field existed,
    # so a run started then still resumes; one that sets them binds its evidence to them.
    if not evaluation_fields.get("setting_sources"):
        evaluation_fields.pop("setting_sources", None)
    # Likewise a run that leaves code nodes on (the default) digests as one started
    # before the switch existed, so it still resumes.
    if evaluation_fields.get("allow_code_nodes", True) is True:
        evaluation_fields.pop("allow_code_nodes", None)
    # And a run with no parent picker (the default) as one started before the setting.
    if not evaluation_fields.get("parent_picker"):
        evaluation_fields.pop("parent_picker", None)
    if not evaluation_fields.get("portfolio"):
        evaluation_fields.pop("portfolio", None)
    route_context = runtime_identity(settings, project, model_client=model_client, uses_models=settings["optimizer"] == "autoresearch" or any(n.get("kind") == "agent" for n in (nodes.values() if isinstance(nodes, dict) else nodes) if isinstance(n, dict)))
    context = {"mode": mode_id, "seed_state": digest(state), "implementation": hashlib.sha256(source.read_bytes()).hexdigest(), "dependencies": implementations, "routing": route_context,
               "inputs": inputs, "contract": digest({k: v for k, v in contract.items() if k != "termination"}),
               "settings": digest(evaluation_fields),
               # How the runtime treats a crashed agent step: evidence from before a change
               # to it is refused on resume rather than mixed in (`node_retry.SEMANTICS`).
               "runtime_semantics": node_retry.SEMANTICS}
    return {"schema_version": SCHEMA_VERSION, "profile": {"name": name, "version": profile.get("version")} if name else None,
            # What the launch changed on top of the saved profile, as given.
            "overrides": dict(overrides or {}) if name else None,
            "mode": mode_id, "settings": settings, "contract": contract,
            "worker_configuration": routing,
            "evaluation_context": context, "context_hash": digest(context)}


def refresh(project: Path, mode_id: str, settings: dict[str, Any], previous: dict[str, Any], *,
            model_client: Any = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Recheck source/input identity without relying on a still-existing recipe.

    ``overrides`` are the flags given on resume; they join the recorded overrides."""
    recipe = {"settings": {}, "contract": previous.get("contract") or {},
              "version": (previous.get("contract") or {}).get("version", 1)}
    result = resolve(project, mode_id, overrides=settings, recipe=recipe, validate_settings=False, model_client=model_client)
    result["profile"] = previous.get("profile")
    result["overrides"] = ({**(previous.get("overrides") or {}), **overrides}
                           if overrides else previous.get("overrides"))
    result["worker_configuration"] = previous.get("worker_configuration")
    return result


def runtime_identity(settings: dict[str, Any], project: Path, *, uses_models: bool, model_client: Any = None) -> dict[str, Any]:
    """Read nonsecret effective configuration; this is not authentication evidence."""
    if not uses_models:
        return {"status": "no_model_required_by_current_graph_or_optimizer"}
    from refactory_dsh.model import DshModelClient
    model = model_client if callable(getattr(model_client, "route", None)) and callable(getattr(model_client, "readiness", None)) else DshModelClient(routes_resolved=True, **{key: settings[key] for key in
        ("worker_backend", "provider", "model", "effort", "permission_mode", "role_routes", "node_routes")
        }, setting_sources=settings.get("setting_sources") or "")
    # The route is the one the client holds: a flag, a route map or the one stored
    # choice. A provider or model a readiness probe reports is never taken into it —
    # that was the profile default, the source a run must not use (dm-258).
    route = model.route()
    resolved = bool(route.get("model")) and (route["worker_backend"] != "dsh" or bool(route.get("provider")))
    context = model.context() if callable(getattr(model, "context", None)) else {}
    # Runtime/credential-store location can change a default route or executable.
    # Keep this nonsecret context, but never bind evidence to a per-run ledger path.
    context = {key: context[key] for key in ("dsh_home", "runtime_root", "dsh_cli", "python_bin", "timeout_s", "attempts", "backoff_s",
                                     "retry_window_s") if key in context}
    return {"status": "configured_unverified" if resolved else "runtime_defaults_unresolved", "route": route,
            "client_context": context}


#: ``--help`` text for the settings that have it.
SETTING_HELP = {
    "parent_picker": "pair a parent picker (archive, evolutionary, islands, hill_climb, "
                     "map_elites) with --optimizer: the picker chooses the candidate each "
                     "step edits, the optimizer writes the edits (default: none)",
    "portfolio": "prompt+pipeline: each round asks the model proposer for one "
                 "prompt/knob-level and one pipeline-level candidate and evaluates both, "
                 "gated in that order (default: off)",
}


def add_arguments(parser: Any) -> None:
    """All launch surfaces parse this same optional configuration vocabulary."""
    for key, value in defaults().items():
        flag = "--" + ("preset" if key == "edit_preset" else key.replace("_", "-"))
        # provider_config travels in the project file and saved profile
        # snapshots, not as a command-line object.
        if key in ("knobs", "frozen", "allowed_ops", "role_routes", "node_routes", "provider_config", "strategy_options"):
            continue
        kind = float if key in ("holdout", "parsimony", "max_elapsed_s", "validation_reserve_s") else int if isinstance(value, int) or key in ("max_tokens", "max_rollouts", "max_items_in_flight") else str
        parser.add_argument(flag, dest=key, type=kind, default=None, help=SETTING_HELP.get(key))
    parser.add_argument("--knob", action="append", default=None, metavar="NAME=VALUE")
    parser.add_argument("--frozen", action="append", default=None, metavar="KNOB")
    parser.add_argument("--allowed-ops", action="append", default=None, metavar="OP")
    parser.add_argument("--strategy-option", action="append", default=None, metavar="KEY=VALUE",
                        help="strategy-specific option as KEY=VALUE; may be repeated, and adds to "
                             "a profile's strategy_options")
    for key in ("role_routes", "node_routes"):
        parser.add_argument("--" + key.replace("_", "-"), type=json.loads, default=None)
    parser.add_argument("--profile", default="")
    parser.add_argument("--profile-version", type=int, default=None)
