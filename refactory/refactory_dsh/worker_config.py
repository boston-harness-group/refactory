"""Project defaults over the existing worker route schema, resolved once per run."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from refactory_dsh.workers import resolve_route, validate_route

FILENAME = "refactory.workers.json"
ROUTE_FIELDS = ("worker_backend", "provider", "model", "effort", "permission_mode",
                 "openshell_gateway", "openshell_image")
FIELDS = (*ROUTE_FIELDS, "role_routes", "node_routes", "provider_config")


def validate_provider_config(config: Any, *, origin: str = FILENAME) -> dict[str, Any]:
    """Validate a nonsecret provider-profile map: provider route name to profile.

    A profile is what a nested DSH runtime's ``llm-pi-ai`` row accepts: endpoint,
    protocol and model catalog, with credentials named by ``apiKeyEnv`` reference
    and never carried as values. Whole-shape validation stays with the nested
    runtime, which fails loud on a malformed row; this rejects what cannot be a
    route map at all and anything that looks like a pasted credential.
    """
    if not isinstance(config, dict) or any(not isinstance(k, str) or not k for k in config):
        raise ValueError(f"{origin}: provider_config must be an object keyed by nonempty provider names")
    for name, profile in config.items():
        if not isinstance(profile, dict):
            raise ValueError(f"{origin}: provider_config.{name} must be an object")
        if "apiKey" in profile or "api_key" in profile:
            raise ValueError(
                f"{origin}: provider_config.{name} carries a credential value; "
                "name the native-store reference with apiKeyEnv instead")
    return config


def validate(settings: dict[str, Any]) -> None:
    """Validate partial defaults and each override without dispatching a worker."""
    unknown = set(settings) - set(FIELDS)
    if unknown:
        raise ValueError(f"{FILENAME}: unknown routing fields {sorted(unknown)}; credentials belong in native stores")
    base = validate_route({key: settings[key] for key in ROUTE_FIELDS if key in settings})
    for field in ("role_routes", "node_routes"):
        routes = settings.get(field, {})
        if not isinstance(routes, dict) or any(not isinstance(k, str) or not k for k in routes):
            raise ValueError(f"{FILENAME}: {field} must be an object with nonempty role/node names")
        for name, override in routes.items():
            try:
                resolve_route(base, node=name, node_routes={name: override})
            except ValueError as error:
                raise ValueError(f"{FILENAME}: {field}.{name}: {error}") from error
    if "provider_config" in settings:
        validate_provider_config(settings["provider_config"])


def resolve(project: Path, overrides: dict[str, Any] | None = None, *,
            inherited: dict[str, Any] | None = None, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return nonsecret effective settings and per-field provenance.

    Inherited resolved context wins over project/environment defaults. Explicit
    overrides are only the supplied fields; route maps replace their lower layer
    as a whole, so an explicit empty object can clear project overrides.
    """
    project = Path(project).expanduser().resolve()
    if inherited is None:
        inherited = json.loads(os.environ.get("REFACTORY_WORKER_CONTEXT") or "{}")
    if not isinstance(inherited, dict):
        raise ValueError("REFACTORY_WORKER_CONTEXT must contain an object")
    route: dict[str, Any] = {"worker_backend": "dsh", "provider": "", "model": "",
                            "effort": "", "permission_mode": "skip"}
    maps: dict[str, Any] = {"role_routes": {}, "node_routes": {}, "provider_config": {}}
    sources = {key: "default" for key in FIELDS}
    layers: list[tuple[str, dict[str, Any]]] = []
    file_info: dict[str, Any] = {"path": str(project / FILENAME), "loaded": False}
    if inherited:
        layers.append(("inherited-context", {key: inherited[key] for key in FIELDS if key in inherited}))
    else:
        env = {key: os.environ[name] for key, name in (
            ("worker_backend", "REFACTORY_WORKER_BACKEND"), ("provider", "REFACTORY_MODEL_PROVIDER"),
            ("model", "REFACTORY_MODEL_ID"), ("effort", "REFACTORY_MODEL_EFFORT")) if os.environ.get(name)}
        if env.get("worker_backend", "dsh") != "dsh":
            env = {"worker_backend": env["worker_backend"]}
        layers.append(("environment", env))
        path = project / FILENAME
        if path.exists():
            raw = path.read_bytes()
            try:
                settings = json.loads(raw)
                if not isinstance(settings, dict):
                    raise ValueError("expected a JSON object")
                validate(settings)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}: invalid worker configuration: {error}") from error
            file_info.update(loaded=True, sha256=hashlib.sha256(raw).hexdigest())
            layers.append(("project-file", settings))
    for source, supplied in (("selected-profile", profile), ("explicit-launch", overrides)):
        layers.append((source, {
            key: value for key, value in (supplied or {}).items()
            if key in FIELDS and value is not None
            and not (key in ("worker_backend", "permission_mode") and value == "")
        }))
    for source, values in layers:
        base = {key: value for key, value in values.items() if key in ROUTE_FIELDS}
        if base.get("worker_backend", route["worker_backend"]) != route["worker_backend"]:
            for key in ("provider", "model", "effort", "permission_mode"):
                sources[key] = source + ":backend-switch"
        route = resolve_route(route, node="run", node_routes={"run": base})
        for key in FIELDS:
            if key in values:
                sources[key] = source
                if key in maps:
                    maps[key] = values[key]
    settings = {**route, **maps}
    validate(settings)
    return {"settings": settings, "provenance": {"project": str(project), "file": file_info,
             "fields": sources, "precedence": "node > role > run; explicit launch/profile > project > environment > defaults; inherited/saved contexts bypass project files"}}


def effective(project: Path) -> dict[str, Any]:
    """Read-only route preview used by terminal, panel and CEO discovery."""
    return resolve(project)
