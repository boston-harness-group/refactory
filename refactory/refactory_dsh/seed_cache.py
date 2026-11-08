"""Reuse a seed candidate's part results from an earlier run (``--seed-results``).

Every optimizer run starts by evaluating its seed, and on a slow task that is a
large share of the run. When several runs share a seed, a task model and a task,
the seed's per-item results from one of them can stand in for the others' first
evaluation of it:

    python -m refactory_dsh.seed_cache export RUN_DIR seed-results.json
    refactory optimize <pkg>/<mode> --seed-results seed-results.json ...

A cached result is used only when all of these hold:

* the candidate is the cache's candidate (its id is a hash of its state);
* the part's item set is exactly the cached one;
* the run's *score context* equals the cache's. That is everything that decides
  what an item scores, and nothing that only shapes the search: the mode, its
  implementation and dependency hashes, the task's inputs, the task-model route
  and provider configuration, the evaluator, the item timeout, the model client's
  inactivity limit, and a hash of the mode's package files (which covers files a
  task runs out of process, such as a grader). Epochs, strategy, budgets and the
  proposer's route are not in it.

A mismatched context stops the run instead of quietly evaluating, so a cache
that was meant to apply never silently doesn't. So does a run whose seed is not
the cache's candidate, or whose whole part holds other items than the cache
recorded (``check_applies``). Each cached part is served at most once per run; a
later occasion of the same part is a fresh measurement and runs for real. Cached
results don't count toward ``max_rollouts``.

With ``repeats`` above 1 every trial is exported and served: the score context
holds ``repeats``, so a cache serves only a run measuring the same number of trials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

FORMAT = 1
#: Written to every run root, so a later export doesn't have to reconstruct it.
CONTEXT_FILE = "score-context.json"
#: Parts a cache may serve. The holdout audit always runs for real.
PARTS = ("train", "val", "test")


class SeedCacheError(ValueError):
    """A seed-results file can't be used for this run."""


def package_files_hash(mode_class: type) -> str:
    """A hash of every file in the package that defines ``mode_class``."""
    module = sys.modules.get(mode_class.__module__)
    source = getattr(module, "__file__", None)
    if not source:
        return ""
    source_path = Path(source).resolve()
    if not mode_class.__module__.startswith("refactory_packages."):
        # Not a package mode (a test's, say): its own file is all there is to hash.
        return hashlib.sha256(source_path.read_bytes()).hexdigest()
    # The package is the top directory under refactory_packages, not the module's
    # own directory, when the mode lives in a subpackage.
    root = source_path.parent
    while root.parent.name != "refactory_packages" and (root.parent / "__init__.py").exists():
        root = root.parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode() + b"\0")
    return digest.hexdigest()


def score_context(config: Any, mode_class: type, *, inactivity_s: int | None = None) -> dict[str, Any]:
    """What decides an item's score in a run with ``config``.

    ``inactivity_s`` defaults to the model client's limit in this process.
    """
    from refactory_dsh import model

    profile = config.resolved_profile or {}
    evaluation = profile.get("evaluation_context") or {}
    routing = evaluation.get("routing") or {}
    return {
        "format": FORMAT,
        "mode": evaluation.get("mode"),
        "implementation": evaluation.get("implementation"),
        "dependencies": evaluation.get("dependencies"),
        "inputs": evaluation.get("inputs"),
        "route": routing.get("route"),
        "client_timeout_s": (routing.get("client_context") or {}).get("timeout_s"),
        "provider_config": config.provider_config,
        "node_routes": config.node_routes,
        "evaluator": config.evaluator,
        "dsh_timeout_s": config.dsh_timeout_s,
        "repeats": config.repeats,
        "model_inactivity_s": model.DEFAULT_INACTIVITY_S if inactivity_s is None else inactivity_s,
        "package_files": package_files_hash(mode_class),
    }


def context_diff(ours: dict[str, Any], theirs: dict[str, Any]) -> list[str]:
    """The keys on which two score contexts differ."""
    return sorted(key for key in set(ours) | set(theirs) if ours.get(key) != theirs.get(key))


def check_applies(path: str | Path, cache: dict[str, Any], candidate: str,
                  part_items: dict[str, set[str]], *, train_sample: bool = False) -> None:
    """Refuse a cache that can never serve this run: another seed candidate, or a
    part whose whole item set differs from the one recorded.

    ``part_items`` holds each part's ids in this run. ``train_sample`` says the run
    scores train minibatches, when the cached train items are one such sample: then
    they need only lie inside this run's train part, and a different sample is
    measured for real (``Loop._cached_measure`` journals the miss).
    """
    differing = []
    if cache.get("candidate") != candidate:
        differing.append(f"candidate (cached {cache.get('candidate')}, this run's seed {candidate})")
    for part, entry in sorted((cache.get("parts") or {}).items()):
        ours = part_items.get(part)
        cached = set(entry.get("items") or ())
        if not ours or cached == ours or (part == "train" and train_sample and cached < ours):
            continue
        differing.append(f"items of {part} ({len(cached)} cached, {len(ours)} in this run, "
                         f"{len(cached ^ ours)} differ)")
    if differing:
        raise SeedCacheError(
            f"{path} cannot serve this run: {'; '.join(differing)}. Drop --seed-results "
            "or export it again from a run with this seed and these parts")


def load(path: str | Path, context: dict[str, Any]) -> dict[str, Any]:
    """Read a seed-results file, and check it was made under ``context``."""
    try:
        cache = json.loads(Path(path).expanduser().read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SeedCacheError(f"cannot read seed results {path}: {exc}") from exc
    if cache.get("format") != FORMAT:
        raise SeedCacheError(f"{path} is seed-results format {cache.get('format')!r}, not {FORMAT}")
    differing = context_diff(context, cache.get("score_context") or {})
    if differing:
        raise SeedCacheError(
            f"{path} was recorded under a different score context ({', '.join(differing)}); "
            "its results would not be comparable. Drop --seed-results or export it again "
            "from a run made under this one's conditions")
    return cache


# ── export ──────────────────────────────────────────────────────────────────


def _journal(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    for line in (run_dir / "journal.jsonl").read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            break  # a run killed mid-write leaves a torn last line
    return records


def _trials(records: list[dict[str, Any]], measurements: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    """Every trial of one complete measurement, with its own item records; None when
    the measurement was incomplete. A record names its trial in ``repeat``."""
    measurements = measurements or {}
    trials = measurements.get("trials") or [{"trial": 1, "score": None}]
    if measurements and not measurements.get("complete", True):
        return None
    by_trial: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_trial.setdefault(int(record.get("repeat") or 1), []).append(
            {k: v for k, v in record.items() if k != "repeat"})
    result = []
    for index, trial in enumerate(trials, start=1):
        number = int(trial.get("trial") or index)
        if number not in by_trial or trial.get("exit_ok") is False:
            return None
        result.append({"score": trial.get("score"), "duration_s": trial.get("duration_s") or 0.0,
                       "model_calls": trial.get("model_calls") or 0, "workdir": trial.get("workdir"),
                       "item_records": by_trial[number]})
    return result


def export(run_dir: Path, *, project: Path, inactivity_s: int | None = None) -> dict[str, Any]:
    """The seed's first complete result on each part of a finished or running run."""
    from refactory_dsh import registry
    from refactory_dsh.loop import LoopConfig
    from refactory_dsh.optim.base import Candidate

    run_dir = run_dir.expanduser().resolve()
    journal = _journal(run_dir)
    seed_state = next((r["state"] for r in journal if r.get("event") == "original_seed"), None)
    if seed_state is None:
        raise SeedCacheError(f"{run_dir}/journal.jsonl has no original_seed event")
    cid = Candidate.from_state(seed_state).id

    stored = run_dir / CONTEXT_FILE
    if stored.exists():
        context = json.loads(stored.read_text())
        if inactivity_s is not None and context.get("model_inactivity_s") != inactivity_s:
            raise SeedCacheError(
                f"{stored} records model_inactivity_s={context.get('model_inactivity_s')}, "
                f"not {inactivity_s}")
    else:
        # A run from before score contexts were recorded: rebuild it from the run's
        # configuration. Its inactivity limit was only ever in the launch
        # environment, so the caller has to state it.
        if inactivity_s is None:
            raise SeedCacheError(
                f"{run_dir} predates {CONTEXT_FILE}; pass --inactivity-s with the "
                "REFACTORY_MODEL_INACTIVITY_S it was launched with (600 if unset)")
        cfg = json.loads((run_dir / "config.json").read_text())
        mode_id = cfg.pop("mode")
        _, mode_class = registry.resolve(project, mode_id)
        context = score_context(LoopConfig.from_dict(cfg), mode_class, inactivity_s=inactivity_s)

    parts: dict[str, Any] = {}
    train = next((r for r in journal if r.get("event") == "train_records" and r.get("id") == cid), None)
    score = next((r for r in journal if r.get("event") == "score" and r.get("id") == cid), None)
    trials = (_trials(train["records"], score.get("measurements"))
              if train and score and score.get("exit_ok") and train.get("records") else None)
    if trials:
        parts["train"] = {
            "items": sorted({str(r["id"]) for trial in trials for r in trial["item_records"]}),
            "workdir": str(run_dir / "runs" / cid),
            "result": {"score": score.get("raw_score"), "exit_ok": True, "error": None,
                       "duration_s": score.get("duration_s") or 0.0,
                       "model_calls": score.get("model_calls") or 0,
                       "facts": score.get("facts") or {}},
            "trials": trials,
        }
    for part in ("val", "test"):
        result = next((r.get("result") or {} for r in journal
                       if r.get("event") == "part_result" and r.get("id") == cid and r.get("part") == part), None)
        occasion = run_dir / "runs" / cid / "occasions" / f"{part}_0"
        records_path = occasion / "item_records.json"
        if not result or not result.get("exit_ok") or not records_path.exists():
            continue
        measured = occasion / "measurements.json"
        trials = _trials(json.loads(records_path.read_text()),
                         json.loads(measured.read_text()) if measured.exists() else None)
        if not trials:
            continue
        parts[part] = {
            "items": sorted({str(r["id"]) for trial in trials for r in trial["item_records"]}),
            "workdir": str(occasion),
            "result": {"score": result.get("score"), "exit_ok": True, "error": None,
                       "duration_s": result.get("duration_s") or 0.0, "model_calls": 0,
                       "facts": {}},
            "trials": trials,
        }
    if not parts:
        raise SeedCacheError(f"{run_dir} has no complete seed result to export")
    return {"format": FORMAT, "candidate": cid, "source": str(run_dir),
            "score_context": context, "parts": parts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m refactory_dsh.seed_cache")
    sub = parser.add_subparsers(dest="verb", required=True)
    out = sub.add_parser("export", help="write a run's seed results for --seed-results")
    out.add_argument("run_dir", type=Path)
    out.add_argument("out", type=Path)
    out.add_argument("--project", type=Path, default=Path(os.environ.get("REFACTORY_PROJECT", ".")))
    out.add_argument("--inactivity-s", type=int, default=None,
                     help="the REFACTORY_MODEL_INACTIVITY_S the run was launched with; "
                          "needed only for runs without " + CONTEXT_FILE)
    args = parser.parse_args(argv)
    try:
        cache = export(args.run_dir, project=args.project.resolve(), inactivity_s=args.inactivity_s)
    except (SeedCacheError, OSError) as exc:
        print(f"seed_cache: {exc}", file=sys.stderr)
        return 1
    args.out.write_text(json.dumps(cache, indent=2, default=str))
    summary = {part: {"items": len(entry["items"]), "score": entry["result"]["score"]}
               for part, entry in cache["parts"].items()}
    print(json.dumps({"candidate": cache["candidate"], "parts": summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
