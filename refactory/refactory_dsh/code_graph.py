"""Code knowledge graph: build and query for design mode's research agents.

graphify (PyPI ``graphifyy``) extracts a tree-sitter-based knowledge graph of a
codebase.  This module wraps it for design mode:

- ``build_code_graph`` is the deterministic op that starts the build as a detached
  process and returns a prompt paragraph for the agents.
- ``explain_node`` and ``callers_of`` query graph.json directly (no graphify binary).
- ``cli_graph`` dispatches ``refactory graph explain|callers`` from the console.

No module-level ``import graphify`` or ``from graphify import`` anywhere: graphify is
optional, and the only reference is the subprocess command assembled at runtime.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

CODE_EXTENSIONS = frozenset({
    ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".go", ".rs", ".java", ".rb",
    ".c", ".h", ".cpp", ".cs", ".kt", ".swift", ".php", ".lua", ".scala", ".sh",
})

_DEFAULT_TIMEOUT_S = 300
_DEFAULT_MAX_CODE_FILES = 20000


# ── atomic status ──────────────────────────────────────────────────────────────

def _write_status(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def _read_status(graph_dir: Path) -> dict[str, Any]:
    status_path = graph_dir / "status.json"
    if not status_path.exists():
        return {"status": "absent"}
    try:
        return json.loads(status_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"status": "failed: unreadable status.json"}


# ── query functions ────────────────────────────────────────────────────────────

def _load_graph(graph_dir: Path) -> tuple[list[dict], list[dict]]:
    graph_path = graph_dir / "graphify-out" / "graph.json"
    data = json.loads(graph_path.read_text())
    return data.get("nodes", []), data.get("links", [])


def _match_name(label: str, query: str) -> bool:
    bare = label.rstrip("()")
    return bare == query or label == query


def _line_number(source_location: str) -> int:
    if source_location and source_location.startswith("L"):
        try:
            return int(source_location[1:])
        except ValueError:
            pass
    return 0


def _format_location(source_file: str, source_location: str) -> str:
    line = _line_number(source_location)
    return f"{source_file}:{line}" if line else source_file


def explain_node(graph_dir: Path, name: str) -> str:
    nodes, links = _load_graph(graph_dir)
    node_by_id: dict[str, dict] = {n["id"]: n for n in nodes}
    matched = [n for n in nodes if _match_name(n.get("label", ""), name)]
    if not matched:
        return ""

    parts: list[str] = []
    for node in matched:
        nid = node["id"]
        loc = _format_location(node.get("source_file", ""), node.get("source_location", ""))
        parts.append(f"{node.get('label', '')}  {loc}")

        callers: list[str] = []
        callees: list[str] = []
        importers: list[str] = []
        contained: list[str] = []

        for link in links:
            rel = link.get("relation", "")
            if rel == "references":
                continue
            if rel == "calls" and link["target"] == nid:
                src_node = node_by_id.get(link["source"], {})
                callers.append(f"  {_format_location(link.get('source_file', ''), link.get('source_location', ''))} {src_node.get('label', link['source'])}")
            elif rel == "calls" and link["source"] == nid:
                tgt_node = node_by_id.get(link["target"], {})
                callees.append(f"  {_format_location(link.get('source_file', ''), link.get('source_location', ''))} {tgt_node.get('label', link['target'])}")
            elif rel == "imports" and link["target"] == nid:
                src_node = node_by_id.get(link["source"], {})
                importers.append(f"  {_format_location(link.get('source_file', ''), link.get('source_location', ''))} {src_node.get('label', link['source'])}")
            elif rel == "contains" and link["source"] == nid:
                tgt_node = node_by_id.get(link["target"], {})
                contained.append(f"  {_format_location(tgt_node.get('source_file', ''), tgt_node.get('source_location', ''))} {tgt_node.get('label', link['target'])}")

        if callers:
            parts.append("  Callers:")
            parts.extend(callers)
        if callees:
            parts.append("  Callees:")
            parts.extend(callees)
        if importers:
            parts.append("  Importers:")
            parts.extend(importers)
        if contained:
            parts.append("  Contains:")
            parts.extend(contained)
        parts.append("")

    return "\n".join(parts).rstrip()


def callers_of(graph_dir: Path, name: str) -> str | None:
    """Return callers as text, empty string if none, or None if the node is not found."""
    nodes, links = _load_graph(graph_dir)
    node_by_id = {n["id"]: n for n in nodes}
    matched = [n for n in nodes if _match_name(n.get("label", ""), name)]
    if not matched:
        return None

    parts: list[str] = []
    for node in matched:
        nid = node["id"]
        if len(matched) > 1:
            parts.append(f"{node.get('label', '')}  ({node.get('source_file', '')})")
        for link in links:
            if link.get("relation") == "calls" and link["target"] == nid:
                src_node = node_by_id.get(link["source"], {})
                parts.append(f"{_format_location(link.get('source_file', ''), link.get('source_location', ''))} {src_node.get('label', link['source'])}")
    return "\n".join(parts).rstrip()


# ── CLI handler ────────────────────────────────────────────────────────────────

def cli_graph(args: Any) -> int:
    graph_path = getattr(args, "graph", None) or ""
    if not graph_path:
        print("Code graph not available (no --graph given). Use grep/find instead.", file=sys.stderr)
        return 1

    graph_dir = Path(graph_path)
    if graph_dir.name == "graph.json":
        graph_dir = graph_dir.parent.parent
    elif (graph_dir / "code-graph").is_dir():
        graph_dir = graph_dir / "code-graph"

    status = _read_status(graph_dir)
    st = status.get("status", "absent")

    if st == "building":
        started = status.get("started", 0)
        timeout = _DEFAULT_TIMEOUT_S
        if started and (time.time() - started) > timeout + 60:
            st = "failed: stale build (likely OOM-killed)"

    if st != "ready":
        reason = st if st.startswith(("failed", "skipped")) else st
        print(f"Code graph not available ({reason}). Use grep/find instead.", file=sys.stderr)
        return 1

    name = args.name
    command = args.graph_command

    if command == "explain":
        result = explain_node(graph_dir, name)
    elif command == "callers":
        result = callers_of(graph_dir, name)
    else:
        print(f"Unknown graph command: {command}", file=sys.stderr)
        return 1

    if result is None:
        print(f"No node named '{name}' found in the graph.")
    elif result:
        print(result)
    else:
        print(f"No callers found for '{name}'.")
    return 0


# ── build op ───────────────────────────────────────────────────────────────────

def _find_graphify() -> str | None:
    override = os.environ.get("_REFACTORY_GRAPHIFY_BIN_OVERRIDE")
    if override is not None:
        return override or None
    candidate = Path(sys.executable).parent / "graphify"
    if candidate.is_file():
        return str(candidate.resolve())
    found = shutil.which("graphify")
    return found


def _find_refactory_cmd() -> str | None:
    candidate = Path(sys.executable).parent / "refactory"
    if candidate.is_file():
        return str(candidate.resolve())
    return None


def _count_code_files(source: str) -> int:
    try:
        result = subprocess.run(
            ["git", "ls-files"], cwd=source, capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return 0
        count = 0
        for line in result.stdout.splitlines():
            if line and Path(line).suffix in CODE_EXTENSIONS:
                count += 1
        return count
    except (subprocess.TimeoutExpired, OSError):
        return 0


def _find_reusable_cache(run_dir: Path, target_key: str) -> Path | None:
    parent = run_dir.parent
    if not parent.is_dir():
        return None
    for sibling in parent.iterdir():
        if sibling == run_dir or not sibling.is_dir():
            continue
        cg = sibling / "code-graph"
        if not cg.is_dir():
            continue
        st = _read_status(cg)
        if st.get("status") != "ready":
            continue
        if st.get("target_key") == target_key:
            return cg / "graphify-out"
    return None


def _target_key(ctx: Any) -> str:
    target = ctx.facts.get("target", "")
    if target:
        resolved = str(Path(target).resolve())
        return resolved
    source = ctx.facts.get("source", "")
    if source:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-common-dir"], cwd=source,
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                common = result.stdout.strip()
                return str(Path(source, common).resolve())
        except (subprocess.TimeoutExpired, OSError):
            pass
    return ""


def _make_build_script(
    *,
    graphify_bin: str,
    source: str,
    graph_dir: str,
    timeout_s: int,
    target_key: str,
) -> str:
    return f'''\
import json
import os
import signal
import subprocess
import sys
import threading
import time

GRAPHIFY = {graphify_bin!r}
SOURCE = {source!r}
GRAPH_DIR = {graph_dir!r}
TIMEOUT_S = {timeout_s}
TARGET_KEY = {target_key!r}
GRAPHIFY_OUT = os.path.join(GRAPH_DIR, "graphify-out")
STATUS_FILE = os.path.join(GRAPH_DIR, "status.json")


def _write_status(data):
    tmp = STATUS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, STATUS_FILE)


def _timeout_handler():
    _write_status({{"status": "failed: timeout"}})
    try:
        os.killpg(os.getpid(), signal.SIGTERM)
    except OSError:
        os._exit(1)


timer = threading.Timer(TIMEOUT_S, _timeout_handler)
timer.daemon = True
timer.start()

try:
    env = {{**os.environ, "GRAPHIFY_OUT": GRAPHIFY_OUT}}
    result = subprocess.run(
        [GRAPHIFY, "extract", SOURCE, "--code-only", "--update", "--out", GRAPH_DIR],
        cwd=GRAPH_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _write_status({{"status": "failed: graphify exit " + str(result.returncode),
                       "stderr": result.stderr[-500:]}})
        timer.cancel()
        sys.exit(1)

    graph_path = os.path.join(GRAPHIFY_OUT, "graph.json")
    with open(graph_path) as f:
        data = json.load(f)
    n_nodes = len(data.get("nodes", []))
    n_edges = len(data.get("links", []))

    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=SOURCE,
            capture_output=True, text=True, timeout=5)
        commit = commit_result.stdout.strip() if commit_result.returncode == 0 else ""
    except Exception:
        commit = ""
    try:
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=SOURCE,
            capture_output=True, text=True, timeout=10)
        dirty = bool(dirty_result.stdout.strip()) if dirty_result.returncode == 0 else False
    except Exception:
        dirty = False

    _write_status({{
        "status": "ready",
        "nodes": n_nodes,
        "edges": n_edges,
        "commit": commit,
        "dirty": dirty,
        "target_key": TARGET_KEY,
    }})
except Exception as exc:
    _write_status({{"status": "failed: " + str(exc)[:200]}})
finally:
    timer.cancel()
'''


_PARAGRAPH_TEMPLATE = (
    "## Code knowledge graph\n"
    "A pre-built code graph answers callers and neighbours in about 2 s. For each function "
    "you investigate, run `{refactory_cmd} graph callers NAME --graph {graph_dir}` "
    "or `{refactory_cmd} graph explain NAME --graph {graph_dir}` first, and fall "
    "back to grep when it says the graph is not available. A graph answer is a lead, not "
    "evidence: always confirm by reading the cited code."
)


def build_code_graph(ctx: Any) -> dict[str, Any]:
    """Deterministic op: start the code graph build as a detached process."""
    knobs = ctx.knobs if hasattr(ctx, "knobs") else {}
    code_graph_on = knobs.get("code_graph", True)
    if isinstance(code_graph_on, dict):
        code_graph_on = code_graph_on.get("value", True)
    if not code_graph_on:
        return {"code_graph_paragraph": ""}

    source = str(ctx.facts.get("source", ""))
    if not source:
        return {"code_graph_paragraph": ""}

    run_dir = ctx.workdir if hasattr(ctx, "workdir") else Path(ctx.path(".")).resolve()
    graph_dir = run_dir / "code-graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    refactory_cmd = _find_refactory_cmd()
    if not refactory_cmd:
        _write_status(graph_dir / "status.json",
                      {"status": "skipped: refactory command not found"})
        return {"code_graph_paragraph": ""}

    graphify_bin = _find_graphify()
    if not graphify_bin:
        _write_status(graph_dir / "status.json",
                      {"status": "skipped: graphify not installed"})
        return {"code_graph_paragraph": ""}

    timeout_s = knobs.get("code_graph_timeout_s", _DEFAULT_TIMEOUT_S)
    if isinstance(timeout_s, dict):
        timeout_s = timeout_s.get("value", _DEFAULT_TIMEOUT_S)
    max_files = knobs.get("code_graph_max_files", _DEFAULT_MAX_CODE_FILES)
    if isinstance(max_files, dict):
        max_files = max_files.get("value", _DEFAULT_MAX_CODE_FILES)

    n_code_files = _count_code_files(source)
    if n_code_files > max_files:
        _write_status(graph_dir / "status.json",
                      {"status": f"skipped: {n_code_files} code files exceeds limit ({max_files})"})
        return {"code_graph_paragraph": ""}

    tkey = _target_key(ctx)

    reusable = _find_reusable_cache(run_dir, tkey) if tkey else None
    graphify_out = graph_dir / "graphify-out"
    if reusable and reusable.is_dir():
        if graphify_out.exists():
            shutil.rmtree(graphify_out)
        shutil.copytree(reusable, graphify_out)

    _write_status(graph_dir / "status.json",
                  {"status": "building", "started": time.time()})

    script = _make_build_script(
        graphify_bin=graphify_bin,
        source=source,
        graph_dir=str(graph_dir),
        timeout_s=int(timeout_s),
        target_key=tkey,
    )
    script_path = graph_dir / "_build.py"
    script_path.write_text(script)

    log_path = graph_dir / "build.log"
    with open(log_path, "w") as log_file:
        subprocess.Popen(
            [sys.executable, str(script_path)],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    paragraph = _PARAGRAPH_TEMPLATE.format(
        refactory_cmd=refactory_cmd,
        graph_dir=str(graph_dir),
    )
    return {"code_graph_paragraph": paragraph}
