"""Explicit live driver. Defaults are inspection only, never paid model execution."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .contract import FIXTURES, digest, reference
from .evidence import authoring_score, capability_score, hash_tree, normalize, route_score, write_report
from .isolation import probe, profile, runtime_probe
from .operator_evidence import OperatorEvidenceContext, evaluator_provenance
from . import approval

LANES = ("remote-reference", "dsh-baseline", "dsh-candidate")
ADAPTER_INSTRUCTIONS = {
    "dsh": "Native mapping: deliver .factory/packages/booking_conflicts_bench with PACKAGE and one Mode. Declare a string knob bookings_path; the operation receives it from RunContext. Write conflicts.json under ctx.workdir. Use standard native discovery/validation. No Task/scoring contract is required for the portable candidate; baseline --single is its native invocation mapping.",
    "remote": "Native mapping: deliver .factory/workflows/booking_conflicts_bench.py with meta/workflow(), CLI and README. Runtime parameter bookings_path is the field in invocation-local inputs.json; declare inputs.json in Workflow.runtime_inputs. Discovery uses the authoring project; execution uses a distinct invocation project_path containing inputs.json. Resolve packaged booking_conflicts.py from the workflow module's __file__; never bake a benchmark input path into the workflow.",
}


def authoring_brief(flavor: str) -> str:
    """Give every lane the same verification ownership, plus native spellings."""
    return ((FIXTURES / "brief.md").read_text() + "\n" +
            (FIXTURES / "authoring-boundary.md").read_text() +
            "\n## Native adapter\n" + ADAPTER_INSTRUCTIONS[flavor] +
            "\nArtifact-only trial: do not publish, open PRs, or modify framework source. "
            "Follow the full native Create workflow and pause for real human strategy approval.\n")


def read(path: Path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def source_identity(source: Path, revision: str | None) -> dict:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True, stderr=subprocess.DEVNULL).strip()
        patch = subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=source)
        untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=source, text=True).splitlines()
    except subprocess.CalledProcessError:
        head, patch, untracked = None, b"", []
    # A git archive has no observed HEAD. Its claimed revision stays separate;
    # actual source bytes are frozen so a changed archive cannot masquerade as it.
    if head is None:
        source_files = [p for p in source.rglob("*") if p.is_file() and not p.is_symlink()
                        and not {".venv", "node_modules", "__pycache__", ".git", ".factory"} & set(p.parts)]
    else:
        tracked = subprocess.check_output(["git", "ls-files"], cwd=source, text=True).splitlines()
        source_files = [source / p for p in [*tracked, *untracked] if (source / p).is_file() and not (source / p).is_symlink()]
    tree_digest = hashlib.sha256(json.dumps({str(p.relative_to(source)): digest(p) for p in sorted(source_files)}, sort_keys=True).encode()).hexdigest()
    return {"source_root": str(source), "commit": head, "declared_revision": revision or head,
            "source_tree_sha256": tree_digest,
            "patch_sha256": hashlib.sha256(patch).hexdigest(),
            "untracked_sha256": {p: digest(source / p) for p in untracked if (source / p).is_file()},
            "dependencies": {p: digest(source / p) for p in ("uv.lock", "pnpm-lock.yaml", "pyproject.toml") if (source / p).is_file()}}


def static_graph(source: Path, python: Path, mode: str) -> dict:
    """Read a trusted frozen source graph without running it or loading a model."""
    if mode == "create-v2":
        code = "from factory.workflow.contributed.create_v2.workflow import workflow; graph=workflow().to_dict()"
        module = source
    else:
        code = ("from refactory_dsh import registry; from pathlib import Path; "
                f"graph=registry.load('default_pack',Path.cwd()).mode({mode!r})().graph().to_state()")
        module = source / "refactory"
    env = {**os.environ, "PYTHONPATH": str(module), "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        cp = subprocess.run([str(python), "-c", "import json; " + code + "; print(json.dumps(graph))"],
                            cwd=source, env=env, capture_output=True, text=True, timeout=30)
        return {"status": "passed" if cp.returncode == 0 else "unavailable",
                "origin": "static-source-inspection", "graph": json.loads(cp.stdout) if cp.returncode == 0 else {},
                "stderr": cp.stderr, "returncode": cp.returncode}
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return {"status": "unavailable", "origin": "static-source-inspection", "reason": str(exc), "graph": {}}


def runtime_identity(root: Path, dependency_cache: dict | None = None) -> dict:
    """Bind the installed worker runtime separately from the Python mode source."""
    root = root.resolve()
    modules = root / "node_modules"
    store = (modules / ".pnpm").resolve() if (modules / ".pnpm").exists() else modules.resolve()
    cache = dependency_cache if dependency_cache is not None else {}
    if str(store) not in cache:
        files = hash_tree(store)
        links = {str(p.relative_to(store)): str(p.readlink()) for p in sorted(store.rglob("*")) if p.is_symlink()}
        cache[str(store)] = {"root": str(store), "file_count": len(files),
            "sha256": hashlib.sha256(json.dumps({"files": files, "links": links}, sort_keys=True).encode()).hexdigest()}
    module_roots = [modules, *root.glob("packages/*/*/node_modules")]
    bindings = {str(p.relative_to(root)): str(p.resolve()) for base in module_roots
                for p in base.rglob("*") if p.is_symlink()}
    launcher = modules / ".bin/dsh"
    package = modules / "@deepseek-ai/dsh/package.json"
    return {"root": str(root), "source": source_identity(root, None),
            "dependencies": cache[str(store)], "module_bindings": bindings,
            "dsh_package": {"path": str(package.resolve()), "version": read(package, {}).get("version"),
                            "sha256": digest(package) if package.is_file() else None},
            "launcher": {"path": str(launcher), "sha256": digest(launcher) if launcher.is_file() else None}}


def prepare(args) -> dict:
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "trial.json").exists():
        raise ValueError("Trial already exists; create a new root for a changed contract/configuration")
    baseline = read(FIXTURES / "baseline.json")
    sources = {"remote-reference": args.reference.resolve(), "dsh-baseline": args.baseline.resolve(), "dsh-candidate": args.candidate.resolve()}
    projects = {lane: root / lane / "project" for lane in LANES}
    for project in projects.values():
        project.mkdir(parents=True, exist_ok=False)
    operator = root / "operator"
    operator.mkdir()
    (operator / "isolation-canary.txt").write_text("operator evidence\n")
    package = Path(__file__).resolve().parents[2]
    identities = {}
    dependency_cache = {}
    for lane, project in projects.items():
        flavor = "dsh" if lane.startswith("dsh") else "remote"
        (project / "BRIEF.md").write_text(authoring_brief(flavor))
        inp = project / "inputs"
        inp.mkdir()
        for name in ("overlap", "touching", "invalid-duplicate"):
            shutil.copyfile(FIXTURES / (name + ".json"), inp / (name + ".json"))
        revision = (baseline["reference_commit"] if flavor == "remote" else
                    baseline["dsh_baseline_commit"] if lane == "dsh-baseline" else
                    getattr(args, "candidate_revision", None))
        identities[lane] = source_identity(sources[lane], revision)
        runtime_root = (getattr(args, "baseline_runtime_root", None) if lane == "dsh-baseline" else
                        getattr(args, "candidate_runtime_root", None) if lane == "dsh-candidate" else None) or args.runtime_root
        runtime_root = runtime_root.resolve()
        runtime_pin = runtime_identity(runtime_root, dependency_cache) if flavor == "dsh" else None
        # An archived candidate contains another copy of the independent checker
        # and controlled fixture solutions. Hide test/benchmark sources in every
        # snapshot, not just the harness checkout's FIXTURES directory.
        source_tests = [path for src in sources.values() for path in
                        (src / "tests", src / "refactory/tests", src / "refactory/benchmarks")]
        hidden = [operator, FIXTURES, *source_tests, *getattr(args, "hidden", [])]
        peers = [p for other, p in projects.items() if other != lane]
        runtime_home = project / "runtime/dsh"
        profile_dir = runtime_home / "profiles/headless"
        profile_dir.mkdir(parents=True)
        profile_source = args.dsh_home.expanduser().resolve() / "profiles/headless"
        for filename in ("package.json", "cordis.patch.yml", "pnpm-workspace.yaml"):
            if (profile_source / filename).is_file():
                shutil.copyfile(profile_source / filename, profile_dir / filename)
        auth_source = args.dsh_home.expanduser().resolve() / ".credentials.yaml"
        if auth_source.is_file():
            # Native credential store remains in place; only a read-only reference
            # is shared. No credential value enters config, prompts or reports.
            (runtime_home / ".credentials.yaml").symlink_to(auth_source)
        (project / "runtime/claude").mkdir()
        (project / "tmp").mkdir()
        adapter_root = project / "runtime/benchmark"
        adapter_package = adapter_root / "benchmarks/create_fidelity"
        adapter_package.mkdir(parents=True)
        shutil.copyfile(package / "benchmarks/__init__.py", adapter_root / "benchmarks/__init__.py")
        for module in (package / "benchmarks/create_fidelity").glob("*.py"):
            shutil.copyfile(module, adapter_package / module.name)
        python = args.dsh_python if flavor == "dsh" else args.remote_python
        mode_name = "create_fidelity" if lane == "dsh-candidate" else "create" if flavor == "dsh" else "create-v2"
        write(operator / "graphs" / f"{lane}.json", static_graph(sources[lane], python, mode_name))
        python_roots = json.loads(subprocess.check_output([str(python), "-c", "import json,sys; print(json.dumps([sys.prefix,sys.base_prefix]))"], text=True))
        dependencies = [runtime_root / "node_modules", *map(Path, python_roots)]
        if runtime_pin:
            dependencies.append(Path(runtime_pin["dependencies"]["root"]))
        node = shutil.which("node")
        if node:
            dependencies.append(Path(node).resolve())
        claude = shutil.which("claude")
        if flavor == "remote" and claude:
            dependencies.append(Path(claude).resolve())
        readonly = [*sources.values(), package, adapter_root, *dependencies, auth_source]
        allowed_reads = [sources[lane], adapter_root, *dependencies, auth_source,
                         root / lane / "config.json", root / lane / "isolation.sb"]
        policy = root / lane / "isolation.sb"
        policy.write_text(profile(project, peers, readonly, hidden, allowed_reads=allowed_reads))
        config = {"schema_version": 1, "lane": lane, "project": str(project), "source": str(sources[lane]),
                  "identity": identities[lane], "mode": mode_name,
                  "authoring_context_version": 2, "authoring_brief_sha256": digest(project / "BRIEF.md"),
                  "runtime_identity": runtime_pin,
                  "python": str(args.dsh_python if flavor == "dsh" else args.remote_python),
                  "adapter_root": str(adapter_root), "adapter_sha256": hash_tree(adapter_root),
                  "policy": str(policy), "inputs": str(inp),
                  "total_timeout_s": args.total_timeout, "shutdown_grace_s": 15,
                  "model_context": {"runtime_root": str(runtime_root),
                      "dsh_home": str(runtime_home),
                      "python_bin": str(args.dsh_python.parent), "provider": args.provider,
                      "model": args.model, "effort": "high", "timeout_s": 3600,
                      "attempts": 1, "worker_backend": "dsh"},
                  "allowances": {"research_director": 3600, "strategy_director": 3600,
                      "qa_director": 3600, "builder": 1800, "overwatch": 1800},
                  "route_policy": f"DSH directors and children require served {args.model}/high; remote defaults disclosed",
                  "runtime_state_isolation": {"status": "pending-probe", "dsh_home": str(runtime_home),
                      "claude_config_dir": str(project / "runtime/claude"),
                      "private_tmp": str(project / "tmp"), "shared_auth_reference": str(auth_source)},
                  "forbidden_controls": [str(operator / "isolation-canary.txt"), *[str(p / "isolation-canary.txt") for p in peers],
                      *[str(p) for src in sources.values() for p in
                        (src / "refactory/tests/fixtures/create_fidelity/v1/acceptance.py",
                         src / "refactory/tests/test_create_fidelity_native_lane.py") if p.is_file()]]}
        write(root / lane / "config.json", config)
    trial = {"schema_version": 1, "benchmark_version": 1, "trial_id": root.name,
             "shared_brief_sha256": digest(FIXTURES / "brief.md"),
             "authoring_context_version": 2,
             "authoring_boundary_sha256": digest(FIXTURES / "authoring-boundary.md"),
             "checker_sha256": digest(FIXTURES / "acceptance.py"),
             "reference_sha256": digest(FIXTURES / "reference.json"), "identities": identities,
             "model": args.model, "effort": "high", "route_authorization": args.route_authorization,
             "full_mode_allowance_s": args.total_timeout,
             "calibration": "Initial declared 12-hour ceiling; no claim of calibrated completion time"}
    write(root / "trial.json", trial)
    return trial


def preflight(root: Path, receipt_root: Path | None) -> dict:
    projects = [root / lane / "project" for lane in LANES]
    result = {"isolation": {}, "runtime_state_isolation": {},
              "route": {"status": "unavailable", "reason": "No served route probe supplied"}}
    for lane, project in zip(LANES, projects):
        config = read(root / lane / "config.json")
        result["isolation"][lane] = probe(project, [p for p in projects if p != project], Path(config["policy"]), config["python"])
        runtime = config.get("runtime_state_isolation", {})
        result["runtime_state_isolation"][lane] = runtime_probe(project, Path(config["policy"]), config["python"],
            [Path(p) for p in config.get("forbidden_controls", [])]) if runtime.get("dsh_home") else {
                "status": "unavailable", "reason": "No per-lane runtime session/cache isolation configuration"}
    if receipt_root is not None:
        workers = [read(p) for p in sorted(receipt_root.rglob(".refactory/workers/*.json"))]
        trial = read(root / "trial.json", {})
        route = route_score(workers, model=trial.get("model", "gpt-6.1-sol"))
        completed = read(receipt_root / "probe.json", {})
        if completed.get("status") != "completed":
            route.update(status="unavailable", reason=completed.get("reason", "Probe did not complete"))
        result["route"] = route
        result["route"]["raw_receipts"] = [str(p) for p in sorted(receipt_root.rglob(".refactory/workers/*.json"))]
        result["route"]["probe"] = str(receipt_root / "probe.json")
    result["status"] = "passed" if result["route"]["status"] == "passed" and all(x["status"] == "passed" for group in ("isolation", "runtime_state_isolation") for x in result[group].values()) else "unavailable"
    write(root / "operator/preflight.json", result)
    return result


def execute_command(argv: list[str], *, cwd: Path, env: dict, output: Path,
                    timeout: float, grace: float = 15) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    record = {"argv": argv, "cwd": str(cwd), "timeout_s": timeout, "shutdown_grace_s": grace,
              "effective_deadline_layer": "paired-harness-wall", "started_unix": time.time()}
    with (output / "stdout.log").open("wb") as stdout, (output / "stderr.log").open("wb") as stderr:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            record["returncode"] = proc.wait(timeout=timeout)
            record["status"] = "completed" if proc.returncode == 0 else "failed"
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            record["status"] = "timed_out" if isinstance(exc, subprocess.TimeoutExpired) else "interrupted"
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=grace)
                record["cleanup"] = "owned process group terminated"
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=grace)
                record["cleanup"] = "forced owned-group kill; runtime cleanup unconfirmed"
    record["elapsed_s"] = time.time() - record["started_unix"]
    write(output / "command.json", record)
    return record


def run_lane(root: Path, lane: str, operation: str) -> dict:
    config_path = root / lane / "config.json"
    config = read(config_path)
    source, project = Path(config["source"]), Path(config["project"])
    current = source_identity(source, config["identity"]["declared_revision"])
    if current != config["identity"]:
        raise ValueError(f"{lane} source changed after freeze; use a new candidate trial")
    if config.get("adapter_sha256") and hash_tree(Path(config["adapter_root"])) != config["adapter_sha256"]:
        raise ValueError("Frozen benchmark adapter changed after preparation")
    runtime_pin = config.get("runtime_identity")
    if runtime_pin and runtime_identity(Path(runtime_pin["root"])) != runtime_pin:
        raise ValueError("Worker runtime or installed dependencies changed after preparation")
    env = dict(os.environ)
    # Always execute the hashed adapter copy; a candidate repository may carry
    # its own older benchmarks package, which must not shadow this controller.
    env["PYTHONPATH"] = os.pathsep.join([config["adapter_root"], str(source / "refactory" if lane.startswith("dsh") else source)])
    env["PATH"] = str(Path(config["python"]).parent) + os.pathsep + env.get("PATH", "")
    env["REFACTORY_PROJECT"] = str(project)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    runtime = config["runtime_state_isolation"]
    env["DSH_HOME"] = runtime["dsh_home"]
    env["CLAUDE_CONFIG_DIR"] = runtime["claude_config_dir"]
    env["TMPDIR"] = runtime["private_tmp"]
    policy = Path(config["policy"])
    if operation == "native":
        # Native computation needs no model/network access. Apply this at the
        # lane owner so every descendant inherits it without nested sandboxing.
        native_policy = policy.with_name("native-isolation.sb")
        native_policy.write_text(policy.read_text() + "\n(deny network*)\n")
        policy = native_policy
    argv = ["/usr/bin/sandbox-exec", "-f", str(policy), config["python"], "-m", "benchmarks.create_fidelity.lane", operation, str(config_path)]
    result = execute_command(argv, cwd=project, env=env, output=project / "raw" / operation,
                           timeout=config["total_timeout_s"] if operation == "author" else 300,
                           grace=config["shutdown_grace_s"])
    if runtime_pin:
        stable = (runtime_identity(Path(runtime_pin["root"])) == runtime_pin
                  and source_identity(source, config["identity"]["declared_revision"]) == config["identity"])
        write(project / "raw" / operation / "runtime-integrity.json", {
            "status": "passed" if stable else "failed", "runtime_identity": runtime_pin,
            "boundary": "Source, launcher, module bindings and installed dependency bytes checked before and after the operation; authoring sandbox denies their writes",
        })
        if not stable:
            result.update(status="failed", integrity_error="Source or installed runtime changed during operation")
    return result


def run_pair(root: Path, lanes: list[str]) -> list[dict]:
    receipt = read(root / "operator/preflight.json", {})
    if receipt.get("status") != "passed":
        blocked = [{"lane": lane, "status": "unavailable", "reason": "Required route/isolation preflight did not pass"} for lane in lanes]
        write(root / "operator/launch-blocked.json", blocked)
        return blocked
    # No model run is reachable without the explicit live command and preflight.
    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        return list(pool.map(lambda lane: run_lane(root, lane, "author"), lanes))


def collect(root: Path) -> dict:
    lanes = []
    configs = {lane: read(root / lane / "config.json") for lane in LANES}
    lane_roots = tuple(Path(config["project"]) for config in configs.values())
    operator_root = (root / "operator").resolve()
    if not operator_root.is_relative_to(root.resolve()) or any(
            operator_root.is_relative_to(path.resolve()) or path.resolve().is_relative_to(operator_root)
            for path in lane_roots):
        raise ValueError("Operator review context must be outside every lane-writable project")
    context_path = root / "operator/review-context.json"
    if context_path.is_symlink() or not context_path.resolve().is_relative_to((root / "operator").resolve()):
        raise ValueError("Operator review context must not be a link to lane-authored evidence")
    review_context = read(context_path)
    evaluation = {**evaluator_provenance(),
                  "evaluated_unix": time.time(),
                  "context_sha256": digest(context_path) if review_context is not None else None,
                  "boundary": "Post-run operator evaluation is versioned separately from frozen authoring source and adapter identities"}
    for lane in LANES:
        config = configs[lane]
        project = Path(config["project"])
        operator_evidence = None if review_context is None else OperatorEvidenceContext(
            root=root / "operator/reviews", trial_id=root.name, lane_id=lane,
            reviewer_id=review_context["reviewer_id"], lane_roots=lane_roots,
            trusted_user_references=review_context.get("trusted_user_references", {}).get(lane, {}))
        raw_file = project / ".refactory/events.jsonl"
        raw = [json.loads(line) for line in raw_file.read_text().splitlines() if line.strip()] if raw_file.is_file() else read(project / "raw/remote-events.json", {}).get("events", [])
        normalized = normalize(raw, trial_id=root.name, source=str(raw_file if raw_file.is_file() else project / "raw/remote-events.json"))
        write(root / "operator" / f"{lane}-normalized.json", normalized)
        outcome = read(project / "author-result.json", {})
        workers = [read(p) for p in project.rglob(".refactory/workers/*.json")]
        directors = {p.name.removesuffix("-findings.json"): read(p) for p in (project / ".factory").glob("*-findings.json")}
        native = read(project / "native-result.json", {})
        checker = read(root / "operator" / f"{lane}-checker.json")
        launch = read(project / "raw/launch.json", {})
        static = read(root / "operator/graphs" / f"{lane}.json", {})
        runtime_integrity = read(project / "raw/author/runtime-integrity.json", {})
        fidelity = authoring_score(normalized, outcome, director_manifests=directors, workers=workers, lane=lane, evidence_root=project,
                                   expected_model=config["model_context"]["model"], operator_evidence=operator_evidence)
        if lane.startswith("dsh"):
            fidelity.append({"requirement_id": "F.runtime", "status": runtime_integrity.get("status", "unavailable"),
                             "reason": "Separate frozen worker runtime/dependency identity, checked before and after authoring",
                             "evidence": ["raw/author/runtime-integrity.json"] if runtime_integrity else []})
        lanes.append({"lane": lane, "identity": config["identity"], "graph": launch.get("graph", static.get("graph", {})),
            "graph_origin": "native-launch" if launch else "static-source-inspection",
            "evidence_root": str(project), "preflight": read(root / "operator/preflight.json", {}),
            "evaluation": evaluation,
            "source_graph_path": str(root / "operator/graphs" / f"{lane}.json"),
            "capability": capability_score(checker, native, evidence_root=project, operator_evidence=operator_evidence),
            "fidelity": fidelity})
    report = write_report(root / "operator/report", lanes)
    report["evaluation"] = evaluation
    write(root / "operator/report/report.json", report)
    write(root / "operator/report/evaluator.json", evaluation)
    return report


def check_artifact(root: Path, lane: str) -> dict:
    project = root / lane / "project"
    scripts = [p for p in project.rglob("booking_conflicts.py") if "native-runs" not in p.parts]
    if len(scripts) != 1:
        result = {"status": "unavailable", "reason": "Expected exactly one authored booking_conflicts.py", "total": 0}
    else:
        output = root / "operator" / f"{lane}-checker.json"
        cp = subprocess.run([sys.executable, str(FIXTURES / "acceptance.py"), str(scripts[0]), "--report", str(output)], capture_output=True, text=True, timeout=600)
        result = read(output, {"status": "failed", "reason": cp.stderr, "total": 0})
        result["checker_sha256"] = digest(FIXTURES / "acceptance.py")
    write(root / "operator" / f"{lane}-checker.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("root", type=Path)
    for name in ("reference", "baseline", "candidate", "dsh-python", "remote-python", "runtime-root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--provider", default="openai")
    p.add_argument("--baseline-runtime-root", type=Path, help="Frozen baseline worker runtime; defaults to --runtime-root")
    p.add_argument("--candidate-runtime-root", type=Path, help="Frozen candidate worker runtime; defaults to --runtime-root")
    p.add_argument("--candidate-revision", help="Declared candidate revision for a source archive; Git checkouts default to their observed HEAD")
    p.add_argument("--model", choices=["gpt-6.1-sol", "gpt-6-sol"], default="gpt-6.1-sol")
    p.add_argument("--route-authorization", default="Original Sol 6.1/high requirement", help="Record the explicit authorization when selecting a fallback")
    p.add_argument("--dsh-home", type=Path, default=Path.home() / ".dsh", help="Read-only native profile/credential-reference source; session state stays private")
    p.add_argument("--hidden", type=Path, action="append", default=[], help="Historical/other evidence roots forbidden to authoring workers (repeatable)")
    p.add_argument("--total-timeout", type=int, default=43200)
    p = sub.add_parser("preflight")
    p.add_argument("root", type=Path)
    p.add_argument("--route-receipts", type=Path)
    p = sub.add_parser("live", help="Explicitly launch paid full Create workflows after preflight")
    p.add_argument("root", type=Path)
    p.add_argument("--lanes", nargs="+", choices=LANES, default=list(LANES[:2]))
    p = sub.add_parser("approve", help="Deliver an explicit human decision on an already produced strategy")
    p.add_argument("root", type=Path)
    p.add_argument("lane", choices=LANES)
    p.add_argument("--strategy-digest", required=True)
    p.add_argument("--verdict", choices=["proceed", "reloop", "halt"], required=True)
    p.add_argument("--approver", required=True)
    p.add_argument("--feedback", default="")
    for command in ("native", "check", "report", "probe"):
        p = sub.add_parser(command)
        p.add_argument("root", type=Path)
        if command != "report":
            p.add_argument("lane", choices=LANES)
    args = parser.parse_args()
    if args.command == "prepare": result = prepare(args)
    elif args.command == "preflight": result = preflight(args.root.resolve(), args.route_receipts)
    elif args.command == "live": result = run_pair(args.root.resolve(), args.lanes)
    elif args.command == "approve": result = approval.deliver(args.root.resolve() / args.lane / "project", digest=args.strategy_digest, verdict=args.verdict, approver=args.approver, feedback=args.feedback)
    elif args.command == "native": result = run_lane(args.root.resolve(), args.lane, "native")
    elif args.command == "probe": result = run_lane(args.root.resolve(), args.lane, "probe")
    elif args.command == "check": result = check_artifact(args.root.resolve(), args.lane)
    else: result = collect(args.root.resolve())
    print(json.dumps(result, indent=2))
    successful = (isinstance(result, dict) and result.get("status") in {"passed", "completed"}
                  or isinstance(result, list) and bool(result)
                  and all(item.get("status") in {"passed", "completed"} for item in result))
    return 0 if args.command in {"prepare", "approve"} or successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
