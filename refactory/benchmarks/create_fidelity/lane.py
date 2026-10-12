"""Thin subprocess adapters over each framework's native Python entry points.

This module never writes a generated package. It only launches native authoring,
discovers its output, and feeds runtime inputs to the delivered mode.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .evidence import hash_tree
from . import approval

NAME = "booking_conflicts_bench"


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def dsh_author(config: dict, project: Path) -> dict:
    from refactory_dsh import registry
    from refactory_dsh.model import DshModelClient
    os.environ["REFACTORY_PROJECT"] = str(project)
    identifier = config["mode"]
    mode = registry.load("default_pack", project).mode(identifier)()
    state = mode.state_dict()
    knobs = {"name": NAME, "brief": (project / "BRIEF.md").read_text(), "requires_input": "bookings_path"}
    if identifier == "create":
        knobs.update(single_shot=True)
    for key, value in knobs.items():
        state["knobs"][key]["value"] = value
    mode.load_state_dict(state)
    client = DshModelClient(**config["model_context"])
    graph = mode.graph().to_state()
    save(project / "raw" / "launch.json", {"mode": identifier, "graph": graph,
         "source": str(sys.modules[mode.__class__.__module__].__file__),
         "route": client.route(), "entrypoint": "registry.load/Mode.run"})
    # No standing verdicts. Candidate pauses until its native human gate receives
    # a digest-bound answer. Human approval is an explicit later operation.
    result = mode.run(project, model=client, task=knobs,
                      resource_limits={"max_elapsed_s": config["total_timeout_s"]})
    while not result.exit_ok:
        progress_path = project / ".refactory/progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
        if progress.get("node") != "gate_strategy" or "approval" not in str(result.error).lower() and "verdict" not in str(result.error).lower():
            break
        inputs = ("strategy.json", "brief.json") if identifier == "create" else approval.DEFAULT_INPUTS
        pending = approval.request(project, "Review the strategy and original intent; explicitly approve or request revision.", inputs=inputs)
        answer = None
        while answer is None:
            time.sleep(0.25)
            answer = approval.take(project, pending)
        if answer["verdict"] == "halt":
            return {"exit_ok": False, "error": "Human halted strategy", "approval": answer}
        from refactory_dsh.steering import Ledger
        ledger = Ledger(project)
        if answer["feedback"]:
            ledger.note("correction", answer["feedback"])
        reads = tuple(mode.graph().nodes["gate_strategy"].reads)
        ledger.verdict("gate_strategy", answer["verdict"], review=ledger.visit("gate_strategy", reads))
        if identifier == "create":
            # The pinned baseline's Mode.resume compares newly annotated live
            # ledger prompts against the pre-answer graph and refuses them. Its
            # documented Mode.run(start=..., facts=...) recovery API preserves
            # the native gate, fresh ledger consumption and cumulative budget.
            # Record this adapter mapping; do not patch the frozen baseline.
            save(project / "raw" / f"resume-{time.time_ns()}.json", {
                "entrypoint": "Mode.run(start=checkpoint.node, facts=checkpoint.facts)",
                "reason": "Pinned baseline Mode.resume rejects new human ledger context",
                "node": progress["node"], "strategy_digest": answer["strategy_digest"],
                "checkpoint_sha256": hashlib.sha256(progress_path.read_bytes()).hexdigest(),
            })
            result = mode.run(project, model=client, task=progress.get("task", knobs),
                              start=progress["node"], facts=progress.get("facts", {}),
                              resource_limits={"max_elapsed_s": config["total_timeout_s"]})
        else:
            # Candidate exercises the certified checkpoint recovery API itself.
            result = type(mode).resume(project, model=client)
    return {"exit_ok": result.exit_ok, "error": result.error,
            "duration_s": result.duration_s, "facts": result.facts}


def dsh_probe(config: dict, project: Path) -> dict:
    from refactory_dsh.model import DshModelClient
    directory = project / "route-probe"
    directory.mkdir(exist_ok=True)
    client = DshModelClient(**{**config["model_context"], "timeout_s": 60, "attempts": 1}).for_node("route-probe", role="ceo")
    result = {"requested": client.route(), "kind": "isolated-route-preflight-not-authoring", "probe_timeout_s": 60}
    try:
        result.update(status="completed", response=client.complete("Respond with exactly ROUTE_OK. Do not call tools or modify files.", workdir=directory))
    except Exception as error:
        result.update(status="unavailable", reason=str(error))
    save(directory / "probe.json", result)
    return result


def remote_author(config: dict, project: Path) -> dict:
    from factory.workflow.registry import WorkflowRegistry
    from factory.workflow.primitives import DEFAULT_AGENT_POOL
    from factory.workflow.executor import WorkflowExecutor
    from factory.agents.runner import invoke_agent

    workflow = WorkflowRegistry.get_workflow("create-v2", project)
    if workflow is None:
        raise ValueError("Pinned native registry cannot discover create-v2")
    os.environ["FACTORY_IDEA"] = (project / "BRIEF.md").read_text()
    save(project / "raw" / "launch.json", {"mode": "create-v2", "graph": workflow.to_dict(),
         "source": str(sys.modules[WorkflowExecutor.__module__].__file__),
         "entrypoint": "WorkflowRegistry.get_workflow/WorkflowExecutor.execute",
         "route": "native role defaults; actual served configuration unverified"})
    calls = []

    async def observed_agent(role, task, project_path, **kwargs):
        index = len(calls)
        record = {"role": role, "requested": kwargs, "task": task}
        calls.append(record)
        save(project / "raw" / f"remote-worker-{index}.json", record)
        output, code = await invoke_agent(role, task, project_path, **kwargs)
        record.update(output=output, returncode=code, actual_route=None)
        save(project / "raw" / f"remote-worker-{index}.json", record)
        return output, code

    def human(prompt: str) -> str:
        pending = approval.request(project, prompt)
        answer = None
        while answer is None:
            time.sleep(0.25)
            answer = approval.take(project, pending)
        if answer["verdict"] == "reloop":
            if not answer["feedback"].strip():
                raise ValueError("A strategy revision needs the human's feedback")
            with (project / ".factory/strategy/user-intent.md").open("a") as stream:
                stream.write("\n## Human strategy feedback\n" + answer["feedback"] + "\n")
            return "reloop strategy_director " + answer["feedback"]
        return answer["verdict"]

    executor = WorkflowExecutor(workflow, project, agent_pool=DEFAULT_AGENT_POOL,
                                agent_fn=observed_agent, input_fn=human, auto_approve=False)
    result = asyncio.run(executor.execute())
    save(project / "raw" / "remote-events.json", {"events": result.events})
    return {"success": result.success, "halted": result.halted,
            "error": result.halt_reason, "nodes_executed": result.nodes_executed,
            "duration_ms": result.duration_ms}


def _offline_argv(code: str) -> tuple[list[str] | None, dict]:
    """A model-free OS boundary inherited by Python and shell descendants."""
    if sys.platform != "darwin":
        return None, {"status": "unavailable", "reason": "No OS network-denial boundary is implemented on this platform"}
    probe_code = """
import ctypes,json,os
library=ctypes.CDLL('/usr/lib/libSystem.B.dylib',use_errno=True)
check=library.sandbox_check
check.restype=ctypes.c_int
checks={operation:check(os.getpid(),operation.encode(),0) for operation in ('network-outbound','network-inbound')}
print(json.dumps(checks))
"""
    try:
        import ctypes
        library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        check = library.sandbox_check
        check.restype = ctypes.c_int
        observed = {operation: check(os.getpid(), operation.encode(), 0) for operation in ("network-outbound", "network-inbound")}
        child = subprocess.run([sys.executable, "-c", probe_code], capture_output=True, text=True, timeout=10)
        inherited = json.loads(child.stdout) if child.returncode == 0 else {}
    except (OSError, ValueError, AttributeError, subprocess.TimeoutExpired) as error:
        return None, {"status": "unavailable", "reason": f"Kernel network policy probe unavailable: {error}"}
    boundary = {"kind": "Kernel-verified inherited Seatbelt network denial", "parent_checks": observed, "child_checks": inherited}
    if any(observed.get(operation) != 1 or inherited.get(operation) != 1 for operation in ("network-outbound", "network-inbound")):
        return None, {**boundary, "status": "unavailable", "reason": "Native lane and child do not both have kernel-enforced network denial"}
    return [sys.executable, "-c", code], {**boundary, "status": "passed"}


def _native_child(project: Path, code: str) -> dict:
    # These guards precede generated imports. The OS boundary is mandatory;
    # API/audit guards alone cannot constrain a generated shell descendant.
    guard = """
import importlib.util, json, socket, sys
def deny_model(*args, **kwargs):
    raise RuntimeError('Native benchmark execution must not invoke a model')
def offline_audit(event, args):
    if (event == 'socket.__new__' and args[1] in (socket.AF_INET,socket.AF_INET6)) or (event == 'socket.connect' and args[0].family in (socket.AF_INET,socket.AF_INET6)) or event == 'socket.getaddrinfo':
        raise RuntimeError('Native benchmark execution must remain offline')
sys.addaudithook(offline_audit)
if importlib.util.find_spec('refactory_dsh') is not None:
    from refactory_dsh.model import DshModelClient
    DshModelClient.complete = deny_model
    DshModelClient._complete = deny_model
if importlib.util.find_spec('factory') is not None:
    import factory.agents.runner as agent_runner
    agent_runner.invoke_agent = deny_model
"""
    argv, boundary = _offline_argv(guard + "\n" + code)
    if argv is None:
        return {"status": "unavailable", "native_boundary": boundary, "error": boundary["reason"]}
    allowed = {key: os.environ[key] for key in ("PATH", "PYTHONPATH", "LANG", "LC_ALL", "TMPDIR") if key in os.environ}
    allowed.update(HOME=str(project), REFACTORY_PROJECT=str(project), PYTHONDONTWRITEBYTECODE="1")
    try:
        result = subprocess.run(argv, cwd=project, env=allowed, capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "unavailable", "native_boundary": {**boundary, "status": "unavailable"}, "error": str(error)}
    raw = {"argv": argv, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "timeout_s": 180}
    if result.returncode != 0:
        unavailable = "sandbox_apply" in result.stderr or "sandbox-exec:" in result.stderr
        return {"status": "unavailable" if unavailable else "failed", "native_boundary": {**boundary, "status": "unavailable" if unavailable else "passed"},
                "error": result.stderr.strip(), "raw_evidence": raw}
    payload = None
    for line in reversed(result.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if payload is None:
        return {"status": "failed", "native_boundary": boundary, "error": "Native subprocess supplied no inspectable result", "raw_evidence": raw}
    return {**payload, "native_boundary": boundary, "raw_evidence": raw}


def _artifact_fingerprint(root: Path) -> dict:
    if any(path.is_symlink() for path in root.rglob("*")) or root.is_symlink():
        raise ValueError("Generated native artifact contains a symlink")
    return hash_tree(root)


def _native_verify(project: Path, inputs: Path, *, remote: bool) -> dict:
    project, inputs = project.resolve(), inputs.resolve()
    # Locate the promised native artifact without importing generated code.
    artifact = project / ".factory/workflows" if remote else project / ".factory/packages" / NAME
    if not artifact.resolve().is_relative_to(project):
        raise ValueError("Generated native artifact escapes its authoring project")
    required = artifact / (NAME + ".py" if remote else "__init__.py")
    if not required.is_file():
        return {"discovered": False, "status": "unavailable", "reason": "Authoring did not deliver the promised native artifact"}
    frozen = _artifact_fingerprint(artifact)
    prefix = f"from pathlib import Path\nproject=Path({str(project)!r})\nname={NAME!r}\n"
    if remote:
        inspection = prefix + """
from factory.workflow.registry import WorkflowRegistry
from factory.workflow.validation import validate_workflow
entries=WorkflowRegistry.discover(project)
if name not in entries: raise ValueError('Generated workflow is not discoverable')
workflow=WorkflowRegistry.get_workflow(name,project)
state=workflow.to_dict()
if any(n.get('_type') in ('AgentNode','LLMNode') or n.get('evaluator_type') == 'agent' for n in state['nodes'].values()):
    raise ValueError('Generated computation requires a model')
issues=validate_workflow(workflow)
print(json.dumps({'discovered':True,'inspected':state,'validated':not issues,'issues':[str(i) for i in issues]}))
"""
    else:
        inspection = prefix + """
from refactory_dsh import registry
from refactory_dsh.nn.validate import validate
if name not in registry.discover_names(project): raise ValueError('Generated package is not discoverable')
package=registry.load(name,project)
ids=package.ids()
if len(ids)!=1: raise ValueError('Expected one generated reusable mode')
mode=package.mode(ids[0].split('/')[-1])()
if any(n.kind == 'agent' or n.gate_kind == 'agent' for n in mode.graph().nodes.values()): raise ValueError('Generated computation requires a model')
if 'bookings_path' not in mode.parameters(): raise ValueError('Missing runtime bookings_path input')
issues=validate(mode.graph(),mode.values(),declared_knobs=mode.parameters(),invariants=mode.invariants(),ports=mode.ports())
print(json.dumps({'discovered':True,'inspected':mode.describe(),'validated':not issues,'issues':[str(i) for i in issues],'mode_id':ids[0]}))
"""
    result = _native_child(project, inspection)
    after = _artifact_fingerprint(artifact)
    result.update(artifact=str(artifact), package_before_import=frozen, package_after_inspection=after, invocations=[])
    if after != frozen:
        result.update(status="failed", validated=False, error="Generated package changed during import/native inspection", package_unchanged=False)
        return result
    result["package_unchanged"] = True
    if result.get("status") in ("failed", "unavailable") or not result.get("validated"):
        return result
    for case in ("overlap", "touching", "invalid-duplicate"):
        run_dir = project / "native-runs" / case
        run_dir.mkdir(parents=True, exist_ok=False)
        input_file = str((inputs / (case + ".json")).resolve())
        if remote:
            save(run_dir / "inputs.json", {"bookings_path": input_file})
            execution = prefix + f"run_dir=Path({str(run_dir)!r})\n" + """
import asyncio
from factory.workflow.registry import WorkflowRegistry
from factory.workflow.executor import WorkflowExecutor
async def no_model(*args,**kwargs): raise RuntimeError('Native benchmark execution must not invoke a model')
workflow=WorkflowRegistry.get_workflow(name,project)
run=asyncio.run(WorkflowExecutor(workflow,run_dir,agent_fn=no_model,input_fn=lambda _: (_ for _ in ()).throw(EOFError())).execute())
print(json.dumps({'exit_ok':run.success,'error':run.halt_reason}))
"""
        else:
            execution = prefix + f"identifier={result['mode_id']!r}\nrun_dir=Path({str(run_dir)!r})\ninput_file={input_file!r}\n" + """
from refactory_dsh import registry
_, mode_class=registry.resolve(project,identifier)
mode=mode_class()
state=mode.state_dict()
state['knobs']['bookings_path']['value']=input_file
mode.load_state_dict(state)
run=mode.run(run_dir,model=None)
print(json.dumps({'exit_ok':run.exit_ok,'error':run.error}))
"""
        before = _artifact_fingerprint(artifact)
        invocation = _native_child(project, execution)
        after = _artifact_fingerprint(artifact)
        out = run_dir / "conflicts.json"
        try:
            value = json.loads(out.read_text()) if out.is_file() else None
        except (OSError, ValueError):
            value = None
        invocation.update(input=case, run_directory=str(run_dir), output=value, output_exists=out.exists(),
                          package_before=before, package_after=after)
        if before != frozen or after != frozen:
            invocation.update(exit_ok=False, status="failed", error="Generated package changed during native invocation")
            result.update(status="failed", package_unchanged=False)
        result["invocations"].append(invocation)
        if invocation.get("status") == "unavailable":
            result.update(status="unavailable", error=invocation.get("error"))
            break
    result["package_after_native"] = _artifact_fingerprint(artifact)
    return result


def dsh_native(project: Path, inputs: Path) -> dict:
    return _native_verify(project, inputs, remote=False)


def remote_native(project: Path, inputs: Path) -> dict:
    return _native_verify(project, inputs, remote=True)


def _native_complete(result: dict) -> bool:
    if (result.get("status") in ("failed", "unavailable") or not result.get("discovered")
            or not result.get("inspected") or not result.get("validated")
            or result.get("native_boundary", {}).get("status") != "passed"
            or result.get("package_unchanged") is not True):
        return False
    frozen = result.get("package_before_import")
    if not frozen or result.get("package_after_inspection") != frozen or result.get("package_after_native") != frozen:
        return False
    invocations = result.get("invocations", [])
    by_input = {invocation.get("input"): invocation for invocation in invocations}
    if len(invocations) != 3 or set(by_input) != {"overlap", "touching", "invalid-duplicate"}:
        return False
    if any(invocation.get("native_boundary", {}).get("status") != "passed"
           or invocation.get("package_before") != frozen or invocation.get("package_after") != frozen for invocation in invocations):
        return False
    for name, expected in (("overlap", {"conflicts": [["a", "b"]]}), ("touching", {"conflicts": []})):
        if by_input[name].get("exit_ok") is not True or by_input[name].get("output") != expected:
            return False
    invalid = by_input["invalid-duplicate"]
    return invalid.get("exit_ok") is False and invalid.get("output_exists") is False and bool(invalid.get("error"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["author", "native", "probe"])
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    project = Path(config["project"])
    try:
        if args.operation == "probe":
            result = dsh_probe(config, project)
            save(project / "probe-result.json", result)
            return 0 if result.get("status") == "completed" else 1
        if args.operation == "author":
            result = (dsh_author if config["lane"].startswith("dsh") else remote_author)(config, project)
        else:
            result = (dsh_native if config["lane"].startswith("dsh") else remote_native)(project, Path(config["inputs"]))
        save(project / f"{args.operation}-result.json", result)
        return (0 if _native_complete(result) else 1) if args.operation == "native" else (0 if result.get("exit_ok", result.get("success", False)) else 1)
    except Exception as exc:
        save(project / f"{args.operation}-result.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
