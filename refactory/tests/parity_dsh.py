"""A stand-in for `dsh --profile headless`, scripted by node id (dm-158).

`test_runtime_equivalence.py` points both runtimes at it through `DSH_CLI`: the walker's
`DshModelClient` and the spine's worker (`worker_node`) each start it once per model
completion, exactly as they start the real DSH. It knows which node is asking from the
worker receipt the client opens for every completion (`REFACTORY_WORKER_RECEIPT`), and
answers with that node's next scripted answer, so the same script drives either runtime
in whatever order it visits the nodes.

    parity_dsh.py --scenario DIR --owner OWNER --node-bin NODE [dsh's own arguments]

A launch of the graph profile (``--profile graph run ...``, what `refactory ceo` starts) is
handed to OWNER (`graph-owner.mjs`), the spine mounted as that profile mounts it, with
``DIR/owner-script.json`` as the person's answers. Any other launch is one completion.

``DIR/scenario.json`` maps a node id to its answers, the n-th call of a node getting the
n-th answer (and the last one again after that). An answer is either text, which is the
completion, or an object:

- ``text``: the completion. A node that works in a workspace collected as a diff
  (design's ``emit``, #516) makes an emission's ``===FILE:``/``===EDIT:``/``===DELETE:``
  changes in that workspace, as `tests_support.FakeModel` does
  (`tests_support.apply_in_workspace`), and the runtime derives its output from them;
- ``files``: ``{path: text}`` written relative to the completion's working directory
  first, as an agent with tools writes beside its answer;
- ``judge``: a judgment built from what the run holds — ``early_asks`` (every ask built,
  and every correction the prompt gives) or ``intent_audit`` (every requirement of
  `authoring-evidence.json` passed, written to `intent-audit.json`);
- ``hold_s``: take that long to answer;
- ``repo_questions``: append the repository-questions record a reviewer gives for the
  functions in `changed-functions.json`.

Every completion is appended to ``DIR/calls.jsonl``: the node, its visit, the prompt it was
given on stdin, its working directory, and the route it was started on: dsh's own
arguments with each ``--patch`` file's content, the route variables, the worker context
and the receipt's role and requested route. The test compares the prompts and the routes;
the working directory is kept for reading a failure, never compared (agent workspaces,
#516, move it).
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from pathlib import Path

#: The early asks judge's corrections section (`design_ops` renders it from the facts).
CORRECTIONS = "## Corrections the person gave when approving the design\n"


def _receipt() -> dict:
    receipt = os.environ.get("REFACTORY_WORKER_RECEIPT", "")
    if not receipt:
        return {}
    try:
        found = json.loads(Path(receipt).read_text())
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


#: The variables a completion's route reaches dsh through.
ROUTE_ENV = ("REFACTORY_WORKER_BACKEND", "REFACTORY_MODEL_PROVIDER", "REFACTORY_MODEL_ID",
             "REFACTORY_MODEL_EFFORT")


def _route(argv: list[str], receipt: dict) -> dict:
    """What this completion runs on: its arguments, patches, route and worker context."""
    patches = {}
    for index, part in enumerate(argv[:-1]):
        if part == "--patch":
            try:
                patches[argv[index + 1]] = Path(argv[index + 1]).read_text()
            except OSError as error:
                patches[argv[index + 1]] = f"<unreadable: {error}>"
    try:
        context = json.loads(os.environ.get("REFACTORY_WORKER_CONTEXT") or "{}")
    except ValueError:
        context = {"unreadable": os.environ.get("REFACTORY_WORKER_CONTEXT")}
    return {"argv": argv, "patches": patches, "env": {name: os.environ.get(name) for name in ROUTE_ENV},
            "context": context, "role": receipt.get("role"), "requested_route": receipt.get("requested_route")}


def _early_asks(prompt: str) -> str:
    items = [{"id": "ask-1", "kind": "ask", "text": "the brief", "built": True,
              "evidence": "changes.diff:1"}]
    given = prompt.split(CORRECTIONS, 1)[-1].split("\n\n## ", 1)[0] if CORRECTIONS in prompt else ""
    if given.strip() and not given.strip().startswith("("):
        items.append({"id": "correction-1", "kind": "correction", "text": "the corrections",
                      "built": True, "evidence": "changes.diff:1"})
    return json.dumps({"items": items})


def _run_dir(cwd: Path) -> Path:
    """Where the run's own files are: the completion's directory, else the node's run."""
    if (cwd / "facts.json").exists():
        return cwd
    return Path(os.environ.get("REFACTORY_WORKDIR") or cwd)


def _intent_audit(cwd: Path) -> None:
    baseline = json.loads((_run_dir(cwd) / "authoring-evidence.json").read_text())
    rows = []
    for item in baseline["requirements"]:
        row = {"id": item["id"], "requirement": item.get("text", item.get("path", "")),
               "status": "pass",
               "evidence": [{"path": "authoring-evidence.json",
                             "detail": "scripted judge inspected the executed result"}]}
        if item.get("kind") in ("correction", "constraint"):
            row["clauses"] = [{"clause": item.get("text", ""), "sites": [
                {"site": "tool.py:2", "tests": ["tests/test_pin.py::test_pin"],
                 "scope": "connected"}],
                "evidence": [{"path": "tool.py", "detail": "the correction changes the built module"}]}]
        rows.append(row)
    (cwd / "intent-audit.json").write_text(json.dumps({"requirements": rows}))



def _correction_overwatch(cwd: Path) -> None:
    run = _run_dir(cwd)
    baseline = json.loads((run / "authoring-evidence.json").read_text())
    audit = json.loads((run / "correction-audit.json").read_text())
    by_id = {item.get("id"): item for item in audit if isinstance(item, dict)}
    corrections = []
    for requirement in baseline.get("requirements", []):
        if requirement.get("kind") not in ("correction", "constraint"):
            continue
        item = by_id.get(requirement["id"], {})
        met = item.get("status") in ("pass", "advisory")
        corrections.append({"id": requirement["id"], "met": met,
                            "evidence": f"correction-audit.json reports {item.get('status', 'missing')}"})
    verdict = "pass" if all(item["met"] for item in corrections) else "redo"
    (cwd / "overwatch-verdict.json").write_text(
        json.dumps({"verdict": verdict, "must_fix": [], "follow_ups": [],
                    "corrections": corrections}))

def _repo_questions(cwd: Path, answer: str) -> str:
    found = _run_dir(cwd) / "changed-functions.json"
    try:
        functions = json.loads(found.read_text())
    except (OSError, ValueError):
        return answer
    entries = [{"function": entry["function"],
                "duplicates": f"absent: searched the repository for {entry['function']}",
                "callers": f"{entry.get('file', 'unknown.py')}:{entry.get('line', 1)} is called from main",
                "shared_state": f"absent: searched {entry['function']} for shared state"}
               for entry in functions if isinstance(entry, dict) and "function" in entry]
    if not entries:
        return answer
    record = "## Repository questions\n\n```json\n" + json.dumps(entries) + "\n```\n"
    head, verdict, tail = answer.rstrip().rpartition("\nVERDICT:")
    if verdict:
        return f"{head}\n\n{record}\nVERDICT:{tail}\n"
    return answer.rstrip() + "\n\n" + record


def _contract(prompt: str) -> dict:
    """The execution contract the runtime put at the top of the prompt (`agent_delivery`),
    which names the node's workspace as the agent reads it; ``{}`` without one."""
    from refactory_dsh.agent_delivery import CONTRACT_END, CONTRACT_START

    if CONTRACT_START not in prompt:
        return {}
    block = prompt.split(CONTRACT_START, 1)[1].split(CONTRACT_END, 1)[0]
    start = block.rfind("\n{\n")
    try:
        found = json.loads(block[start + 1:]) if start >= 0 else {}
    except ValueError:
        return {}
    return found if isinstance(found, dict) else {}


def _is_emission(text: str) -> bool:
    return any(marker in text for marker in ("===FILE:", "===EDIT:", "===DELETE:"))


def _option(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def main(argv: list[str]) -> int:
    scenario = Path(_option(argv, "--scenario"))
    rest = argv[argv.index("--node-bin") + 2:]
    profile = _option(rest, "--profile") if "--profile" in rest else ""
    if profile.startswith("graph"):
        after = rest[rest.index("--profile") + 2:]
        with open(scenario / "launches.jsonl", "a") as out:
            out.write(json.dumps({"argv": rest}) + "\n")
        node_bin = _option(argv, "--node-bin")
        os.execv(node_bin, [node_bin, _option(argv, "--owner"), "--script", str(scenario / "owner-script.json"),
                        "--", *after])
    prompt = sys.stdin.read()
    cwd = Path.cwd()
    receipt = _receipt()
    node = str(receipt.get("node_id") or "")
    route = _route(rest, receipt)
    script = json.loads((scenario / "scenario.json").read_text())
    log = scenario / "calls.jsonl"
    with open(scenario / "calls.lock", "w") as lock:
        # Fork branches ask at once: the count and the record are one step.
        fcntl.flock(lock, fcntl.LOCK_EX)
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        visit = sum(1 for call in calls if call["node"] == node)
        record = {"node": node, "visit": visit, "cwd": str(cwd), "prompt": prompt, "route": route}
        if node == "emit":
            receipt_path = Path(os.environ.get("REFACTORY_WORKER_RECEIPT", ""))
            run_dir = receipt_path.parents[2] if len(receipt_path.parents) > 2 else _run_dir(cwd)
            design = run_dir / "design.md"
            record["available_design"] = design.read_text() if design.exists() else None
        with open(log, "a") as out:
            out.write(json.dumps(record) + "\n")
    answers = script.get(node)
    if not answers:
        print(f"parity_dsh: no scripted answer for node {node!r}", file=sys.stderr)
        return 3
    answer = answers[min(visit, len(answers) - 1)]
    if isinstance(answer, str):
        answer = {"text": answer}
    for name, text in (answer.get("files") or {}).items():
        path = cwd / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    text = answer.get("text", "")
    workspace = _contract(prompt).get("workspace") or {}
    if workspace.get("collect") == "diff" and _is_emission(text):
        from tests_support import apply_in_workspace

        # The agent runs in the workspace its contract names: a runtime that starts it
        # anywhere else fails here rather than having its changes made for it.
        if cwd.resolve() != Path(workspace["path"]).resolve():
            print(f"parity_dsh: node {node!r} ran in {cwd}, not in its workspace {workspace['path']}",
                  file=sys.stderr)
            return 4
        apply_in_workspace(text, Path(workspace["path"]))
    judge = answer.get("judge")
    if judge == "early_asks":
        text = _early_asks(prompt)
    elif judge == "intent_audit":
        _intent_audit(cwd)
    elif judge == "correction_overwatch":
        _correction_overwatch(cwd)
    if answer.get("hold_s"):
        # A completion that takes a while, so concurrent sessions overlap visibly.
        time.sleep(float(answer["hold_s"]))
    if answer.get("repo_questions"):
        text = _repo_questions(cwd, text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
