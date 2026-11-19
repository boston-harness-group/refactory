"""Code knowledge graph: build and query for design mode's research agents.

graphify (PyPI ``graphifyy``) extracts a tree-sitter-based knowledge graph of a
codebase.  This module wraps it for design mode:

- ``build_code_graph`` is the deterministic op that starts the build as a detached
  process and returns a prompt paragraph for the agents.
- ``explain_node`` and ``callers_of`` query graph.json directly (no graphify binary).
- ``query_graph`` and ``path_between`` find the code a brief names and the chain of
  code edges between two nodes, for the explorer step (``await_code_graph`` →
  ``explore_code`` → ``record_graph_context``).
- ``cli_graph`` dispatches ``refactory graph explain|callers|query|path`` from the console.

No module-level ``import graphify`` or ``from graphify import`` anywhere: graphify is
optional, and the only reference is the subprocess command assembled at runtime.
"""
from __future__ import annotations

import json
import os
import re
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

#: Where a run keeps its graph: run state beside the workers' homes, not an artifact of
#: the run. graphify's cache and timestamps would otherwise read as results.
GRAPH_SUBDIR = Path(".refactory") / "code-graph"

_DEFAULT_TIMEOUT_S = 300
_DEFAULT_MAX_CODE_FILES = 20000

#: The persistent settings: the project's refactory.workers.json may carry a
#: ``"code_graph"`` object with these keys; each environment variable sets the same
#: thing. ``--no-graph`` and ``--knob`` win over both.
SETTINGS_KEY = "code_graph"
SETTINGS: tuple[tuple[str, str, str, type], ...] = (
    ("enabled", "code_graph", "REFACTORY_CODE_GRAPH", bool),
    ("timeout_s", "code_graph_timeout_s", "REFACTORY_CODE_GRAPH_TIMEOUT_S", int),
    ("max_files", "code_graph_max_files", "REFACTORY_CODE_GRAPH_MAX_FILES", int),
)


def validate_settings(section: Any, *, origin: str = "refactory.workers.json") -> dict[str, Any]:
    """The project file's ``code_graph`` object, checked (`worker_config` owns the rule)."""
    from refactory_dsh.worker_config import validate_code_graph

    return validate_code_graph(section, origin=origin)


def setting_knobs(project: Path) -> list[str]:
    """The person's persistent code-graph settings as ``NAME=VALUE`` knobs.

    The project file's ``code_graph`` object wins over the environment, as it does for
    the worker route (`worker_config`). Only design declares these knobs, so only a
    design run is given them.
    """
    from refactory_dsh.worker_config import FILENAME

    section: dict[str, Any] = {}
    path = project / FILENAME
    if path.is_file():
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError(f"{path}: {error}") from error
        if isinstance(data, dict) and SETTINGS_KEY in data:
            section = validate_settings(data[SETTINGS_KEY], origin=str(path))
    found = []
    for key, knob, env, kind in SETTINGS:
        if key in section:
            value = section[key]
            found.append(f"{knob}={int(value) if kind is bool else value}")
        elif os.environ.get(env, "").strip():
            found.append(f"{knob}={os.environ[env].strip()}")
    return found


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


def explain_node(graph_dir: Path, name: str, stats: dict[str, int] | None = None) -> str | None:
    """The matching nodes with their connections, or None if no node has the name.

    Nodes with no callers, callees, importers or contents (type nodes such as ``Any``
    repeat once per file) are counted, not printed one empty entry each.
    """
    nodes, links = _load_graph(graph_dir)
    node_by_id: dict[str, dict] = {n["id"]: n for n in nodes}
    matched = [n for n in nodes if _match_name(n.get("label", ""), name)]
    if stats is not None:
        stats["hits"] = len(matched)
    if not matched:
        return None

    parts: list[str] = []
    bare: list[str] = []
    for node in matched:
        nid = node["id"]
        loc = _format_location(node.get("source_file", ""), node.get("source_location", ""))

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

        if not (callers or callees or importers or contained):
            bare.append(loc)
            continue
        parts.append(f"{node.get('label', '')}  {loc}")
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

    shown = len(matched) - len(bare)
    if len(matched) > 1 or bare:
        summary = f"{len(matched)} node(s) named '{name}'; {shown} with connections shown"
        if bare:
            listed = ", ".join(bare[:5]) + (f" and {len(bare) - 5} more" if len(bare) > 5 else "")
            summary += f"; {len(bare)} with none ({listed})"
        parts.insert(0, summary + "\n")
    return "\n".join(parts).rstrip()


def callers_of(graph_dir: Path, name: str, stats: dict[str, int] | None = None) -> str | None:
    """Return callers as text, empty string if none, or None if the node is not found.

    ``stats["hits"]`` is set to the number of callers found.
    """
    nodes, links = _load_graph(graph_dir)
    node_by_id = {n["id"]: n for n in nodes}
    matched = [n for n in nodes if _match_name(n.get("label", ""), name)]
    if stats is not None:
        stats["hits"] = 0
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
                if stats is not None:
                    stats["hits"] += 1
    return "\n".join(parts).rstrip()


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A dead child of a still-running walker stays a zombie until it is reaped.
    try:
        stat = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return True
    return bool(stat) and not stat.startswith("Z")


def _building_state(status: dict[str, Any]) -> str:
    """``building``, or why a build that says so has in fact ended without a status."""
    pid = status.get("pid")
    if isinstance(pid, int) and pid > 0 and not _process_alive(pid):
        return "failed: the build process ended without writing a status (killed, e.g. out of memory)"
    started = status.get("started", 0)
    timeout = status.get("timeout_s", _DEFAULT_TIMEOUT_S)
    if started and (time.time() - started) > timeout + 60:
        return "failed: stale build (still building after its timeout)"
    return "building"


# ── CLI handler ────────────────────────────────────────────────────────────────

QUERY_LOG = "queries.jsonl"


def _log_query(graph_dir: Path, kind: str, query: str, *, hits: int, started: float,
               status: str) -> None:
    """One line per `refactory graph` call, so a run shows whether its agents used the graph.

    Only into a graph directory the run made: a query against no graph has nowhere to log.
    Never fails the query.
    """
    if not graph_dir.is_dir():
        return
    entry = {"at": time.time(), "node": os.environ.get("REFACTORY_NODE_ID") or None,
             "kind": kind, "query": query, "hits": hits, "status": status,
             "ms": round((time.monotonic() - started) * 1000)}
    try:
        with open(graph_dir / QUERY_LOG, "a", encoding="utf-8") as log:
            log.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def _node_windows(run_dir: Path) -> list[tuple[float, float, str]]:
    """When each node ran, from the run's events: node and worker start/end on either runtime."""
    from datetime import datetime

    events = run_dir / ".refactory" / "events.jsonl"
    open_at: dict[str, float] = {}
    windows: list[tuple[float, float, str]] = []
    try:
        lines = events.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            event = json.loads(line)
            node = event.get("node_id")
            at = datetime.fromisoformat(str(event["at"]).replace("Z", "+00:00")).timestamp()
        except (ValueError, KeyError, TypeError):
            continue
        if not node:
            continue
        kind, state = event.get("kind"), event.get("state")
        if kind == "node_started" or (kind == "worker" and state == "starting"):
            open_at.setdefault(node, at)
        elif kind in ("node_completed", "node_failed") or (
                kind == "worker" and state in ("completed", "failed", "timeout", "cancelled",
                                               "budget_exhausted")):
            if node in open_at:
                windows.append((open_at.pop(node), at, node))
    now = time.time()
    windows.extend((start, now, node) for node, start in open_at.items())
    return windows


def usage(run_dir: Path) -> dict[str, Any] | None:
    """The run's graph use for its report, or None when the run built no graph.

    A query counts under the node its worker named (``REFACTORY_NODE_ID``, set by the
    model client on both runtimes). One without a name (a route that sets none, such as
    the spine's CEO subagents) counts under the single node that was running when it was
    made, under ``ambiguous`` when several were, and under ``unattributed`` when none
    was: never a guess between nodes.
    """
    graph_dir = run_dir / GRAPH_SUBDIR
    if not graph_dir.is_dir():
        return None
    status = str(_read_status(graph_dir).get("status", "absent")).split(":", 1)[0]
    entries = []
    try:
        for line in (graph_dir / QUERY_LOG).read_text(encoding="utf-8").splitlines():
            try:
                entries.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        pass
    windows = _node_windows(run_dir) if entries else []
    by_node: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    unanswered = 0
    for entry in entries:
        node = entry.get("node")
        if not node:
            at = entry.get("at", 0)
            inside = {w[2] for w in windows if w[0] <= at <= w[1]}
            node = inside.pop() if len(inside) == 1 else ("ambiguous" if inside else "unattributed")
        by_node[node] = by_node.get(node, 0) + 1
        kind = str(entry.get("kind"))
        by_kind[kind] = by_kind.get(kind, 0) + 1
        if entry.get("status") != "ready":
            unanswered += 1
    return {"status": status, "queries": len(entries), "by_node": dict(sorted(by_node.items())),
            "by_kind": dict(sorted(by_kind.items())), "not_available": unanswered}


def cli_graph(args: Any) -> int:
    graph_path = getattr(args, "graph", None) or ""
    if not graph_path:
        print("Code graph not available (no --graph given). Use grep/find instead.", file=sys.stderr)
        return 1

    graph_dir = Path(graph_path)
    if graph_dir.name == "graph.json":
        graph_dir = graph_dir.parent.parent
    elif (graph_dir / GRAPH_SUBDIR).is_dir():
        graph_dir = graph_dir / GRAPH_SUBDIR

    status = _read_status(graph_dir)
    st = status.get("status", "absent")

    if st == "building":
        st = _building_state(status)

    command = args.graph_command
    if command == "path":
        name = f"{args.start} -> {args.end}"
    elif command == "query":
        name = " ".join(args.terms)
    else:
        name = args.name
    started = time.monotonic()

    if st != "ready":
        reason = st if st.startswith(("failed", "skipped")) else st
        print(f"Code graph not available ({reason}). Use grep/find instead.", file=sys.stderr)
        _log_query(graph_dir, command, name, hits=0, started=started, status=st)
        return 1

    stats: dict[str, int] = {}
    if command == "explain":
        result = explain_node(graph_dir, name, stats)
    elif command == "callers":
        result = callers_of(graph_dir, name, stats)
    elif command == "query":
        result = query_graph(graph_dir, name, getattr(args, "depth", 1), stats)
    elif command == "path":
        result = path_between(graph_dir, args.start, args.end, stats)
    else:
        print(f"Unknown graph command: {command}", file=sys.stderr)
        return 1
    _log_query(graph_dir, command, name, hits=stats.get("hits", 0), started=started, status=st)

    if result is None:
        print(f"No code node matches '{name}' in the graph." if command in ("query", "path")
              else f"No node named '{name}' found in the graph.")
    elif result:
        print(result)
    elif command == "path":
        print(f"No path of code edges within {_PATH_MAX_HOPS} hops: {name}.")
    else:
        print(f"No callers found for '{name}'.")
    return 0


# ── build op ───────────────────────────────────────────────────────────────────

def _find_graphify() -> str | None:
    override = os.environ.get("_REFACTORY_GRAPHIFY_BIN_OVERRIDE")
    if override is not None:
        return override or None
    # The driver's own venv first, then PATH, then where `uv tool install` (install.sh)
    # puts it: the driver runs as `.venv/bin/python` with neither bin dir on PATH.
    candidate = Path(sys.executable).parent / "graphify"
    if candidate.is_file():
        return str(candidate.resolve())
    found = shutil.which("graphify")
    if found:
        return found
    for bin_dir in _uv_tool_bin_dirs():
        candidate = bin_dir / "graphify"
        if candidate.is_file():
            return str(candidate)
    return None


def _uv_tool_bin_dirs() -> list[Path]:
    """Where uv puts tool executables: UV_TOOL_BIN_DIR, XDG_BIN_HOME, else ~/.local/bin."""
    dirs = [os.environ.get("UV_TOOL_BIN_DIR"), os.environ.get("XDG_BIN_HOME")]
    if os.environ.get("XDG_DATA_HOME"):
        dirs.append(str(Path(os.environ["XDG_DATA_HOME"]).parent / "bin"))
    dirs.append(str(Path.home() / ".local" / "bin"))
    return [Path(d) for d in dirs if d]


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
        cg = sibling / GRAPH_SUBDIR
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
    reused: bool = False,
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
REUSED = {reused!r}
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


# A background build: never compete with the run's own work for the CPU.
try:
    os.nice(10)
except OSError:
    pass

# The pid lets the query command tell a build that died (e.g. killed for memory)
# from one still running, instead of waiting out the timeout.
_write_status({{"status": "building", "started": time.time(), "timeout_s": TIMEOUT_S,
               "pid": os.getpid()}})

timer = threading.Timer(TIMEOUT_S, _timeout_handler)
timer.daemon = True
timer.start()

def _extract():
    env = {{**os.environ, "GRAPHIFY_OUT": GRAPHIFY_OUT}}
    return subprocess.run(
        [GRAPHIFY, "extract", SOURCE, "--code-only", "--update", "--out", GRAPH_DIR],
        cwd=GRAPH_DIR,
        env=env,
        capture_output=True,
        text=True,
    )


try:
    result = _extract()
    if result.returncode > 0 and REUSED:
        # graphify refuses an update of a graph reused from another run of this repository
        # when the update comes out smaller (some sources' extraction was incomplete),
        # which a build from nothing does not compare against. Build this checkout afresh.
        import shutil
        shutil.rmtree(GRAPHIFY_OUT, ignore_errors=True)
        result = _extract()
    if result.returncode != 0:
        if result.returncode < 0:
            why = "graphify killed by signal " + str(-result.returncode)
            if -result.returncode == signal.SIGKILL:
                why += " (likely out of memory)"
        else:
            why = "graphify exit " + str(result.returncode)
        _write_status({{"status": "failed: " + why, "stderr": result.stderr[-500:]}})
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
    graph_dir = run_dir / GRAPH_SUBDIR
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

    started = time.time()
    _write_status(graph_dir / "status.json",
                  {"status": "building", "started": started, "timeout_s": int(timeout_s)})

    script = _make_build_script(
        graphify_bin=graphify_bin,
        source=source,
        graph_dir=str(graph_dir),
        timeout_s=int(timeout_s),
        target_key=tkey,
        reused=bool(reusable and reusable.is_dir()),
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


# ── query and path ─────────────────────────────────────────────────────────────

#: The relations a query or path walks: code structure. ``references`` (type mentions,
#: which route every path through nodes such as ``Any``), ``uses`` and ``rationale_for``
#: (comments) are left out.
STRUCTURAL_RELATIONS = frozenset({
    "calls", "indirect_call", "imports", "imports_from", "contains", "method",
    "inherits", "extends", "re_exports", "declares", "defines", "depends_on",
})

_QUERY_SEEDS = 12
_QUERY_TEST_SEEDS = 3
_QUERY_NEIGHBOURS = 40
_PATH_MAX_HOPS = 10
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "into", "when", "what", "which",
    "add", "use", "make", "new", "its", "not", "are", "has", "have", "should", "must",
})


def _is_code_node(node: dict) -> bool:
    return bool(node.get("source_file")) and node.get("file_type") not in ("rationale", "concept")


def _is_test(node: dict) -> bool:
    path = str(node.get("source_file", "")).lower()
    return any(part in path for part in ("test", "spec"))


def _terms(text: str) -> list[str]:
    seen: list[str] = []
    for word in re.split(r"[^A-Za-z0-9_]+", text.lower()):
        for part in [word, *word.split("_")] if "_" in word else [word]:
            if len(part) >= 3 and part not in _STOPWORDS and part not in seen:
                seen.append(part)
    return seen


def _node_line(node: dict) -> str:
    return f"{node.get('label', node.get('id', ''))}  " + _format_location(
        node.get("source_file", ""), node.get("source_location", ""))


def _structural(links: list[dict]) -> list[dict]:
    return [link for link in links if link.get("relation") in STRUCTURAL_RELATIONS]


def query_graph(graph_dir: Path, text: str, depth: int = 1,
                stats: dict[str, int] | None = None) -> str | None:
    """The code nodes whose names or files best match *text*'s terms, and their neighbours
    over code-structure edges up to *depth* hops; None when nothing matches."""
    nodes, links = _load_graph(graph_dir)
    terms = _terms(text)
    if stats is not None:
        stats["hits"] = 0
    if not terms:
        return None
    by_id = {n["id"]: n for n in nodes}
    scored: list[tuple[int, str]] = []
    for node in nodes:
        if not _is_code_node(node):
            continue
        name = str(node.get("norm_label") or node.get("label", "")).lower()
        name_parts = set(_terms(name))
        path = str(node.get("source_file", "")).lower()
        score = sum(3 if term in name_parts else 2 if term in name else 1 if term in path else 0
                    for term in terms)
        if score:
            scored.append((score, node["id"]))
    if not scored:
        return None
    structural = _structural(links)
    degree: dict[str, int] = {}
    for link in structural:
        degree[link["source"]] = degree.get(link["source"], 0) + 1
        degree[link["target"]] = degree.get(link["target"], 0) + 1
    scored.sort(key=lambda item: (-item[0], -degree.get(item[1], 0), item[1]))
    # Test names repeat the words of what they test, so they would crowd out the code:
    # the code first, then a few tests.
    code = [nid for _, nid in scored if not _is_test(by_id[nid])]
    tests = [nid for _, nid in scored if _is_test(by_id[nid])]
    code = code[:_QUERY_SEEDS - min(len(tests), _QUERY_TEST_SEEDS)]
    seeds = code + tests[:_QUERY_SEEDS - len(code)]
    if stats is not None:
        stats["hits"] = len(scored)

    adjacent: dict[str, list[tuple[str, str, str]]] = {}
    for link in structural:
        source, target, rel = link["source"], link["target"], link["relation"]
        adjacent.setdefault(source, []).append((target, rel, "->"))
        adjacent.setdefault(target, []).append((source, rel, "<-"))

    parts = [f"{len(scored)} code node(s) match {terms}; the {len(seeds)} best:"]
    parts.extend(f"  {_node_line(by_id[nid])}  ({degree.get(nid, 0)} edges)" for nid in seeds)
    seen = set(seeds)
    frontier = list(seeds)
    shown = 0
    neighbour_lines: list[str] = []
    # An edge between two matches is shown once, from the better ranked.
    paired: set[frozenset[str]] = set()
    for hop in range(1, max(1, min(depth, 3)) + 1):
        following: list[str] = []
        for nid in frontier:
            for other, rel, arrow in adjacent.get(nid, []):
                if hop == 1 and other in seeds and other != nid:
                    pair = frozenset((nid, other))
                    if pair not in paired and shown < _QUERY_NEIGHBOURS:
                        paired.add(pair)
                        neighbour_lines.append(
                            f"  [1] {by_id[nid].get('label', nid)} {arrow} {rel} {arrow} "
                            f"{_node_line(by_id[other])}")
                        shown += 1
                    continue
                if other in seen or other not in by_id or not _is_code_node(by_id[other]):
                    continue
                seen.add(other)
                following.append(other)
                if shown < _QUERY_NEIGHBOURS:
                    neighbour_lines.append(
                        f"  [{hop}] {by_id[nid].get('label', nid)} {arrow} {rel} {arrow} "
                        f"{_node_line(by_id[other])}")
                    shown += 1
        frontier = following
    if neighbour_lines:
        total = len(seen) - len(seeds) + len(paired)
        parts.append(f"\nNeighbours over code edges (depth {depth}; {shown} of {total} shown):")
        parts.extend(neighbour_lines)
    return "\n".join(parts)


def _resolve(nodes: list[dict], name: str) -> list[str]:
    exact = [n["id"] for n in nodes if _is_code_node(n) and _match_name(n.get("label", ""), name)]
    if exact:
        return exact
    lowered = name.lower().rstrip("()")
    return [n["id"] for n in nodes
            if _is_code_node(n) and str(n.get("norm_label", "")).lower() == lowered]


def path_between(graph_dir: Path, start: str, end: str,
                 stats: dict[str, int] | None = None) -> str | None:
    """The shortest chain of code-structure edges between two named nodes, either
    direction; None when a name matches no code node, "" when no chain is found."""
    from collections import deque

    nodes, links = _load_graph(graph_dir)
    by_id = {n["id"]: n for n in nodes}
    starts, ends = _resolve(nodes, start), set(_resolve(nodes, end))
    # A path through a test file (one that imports both) says nothing about the code:
    # tests are only ends, never steps.
    through_tests = any(_is_test(by_id[nid]) for nid in (*starts, *ends))
    if stats is not None:
        stats["hits"] = 0
    if not starts or not ends:
        return None
    adjacent: dict[str, list[tuple[str, dict, bool]]] = {}
    for link in _structural(links):
        adjacent.setdefault(link["source"], []).append((link["target"], link, True))
        adjacent.setdefault(link["target"], []).append((link["source"], link, False))
    previous: dict[str, tuple[str, dict, bool] | None] = {nid: None for nid in starts}
    queue = deque((nid, 0) for nid in starts)
    found = ""
    while queue:
        nid, hops = queue.popleft()
        if nid in ends:
            found = nid
            break
        if hops >= _PATH_MAX_HOPS:
            continue
        for other, link, forward in adjacent.get(nid, []):
            if other in previous or other not in by_id or not _is_code_node(by_id[other]):
                continue
            if other not in ends and not through_tests and _is_test(by_id[other]):
                continue
            previous[other] = (nid, link, forward)
            queue.append((other, hops + 1))
    if not found:
        return ""
    chain: list[str] = []
    nid = found
    while previous[nid] is not None:
        before, link, forward = previous[nid]  # type: ignore[misc]
        where = _format_location(link.get("source_file", ""), link.get("source_location", ""))
        arrow = f"--{link['relation']}-->" if forward else f"<--{link['relation']}--"
        chain.append(f"  {_node_line(by_id[nid])}")
        chain.append(f"    {arrow}  ({where})")
        nid = before
    chain.append(f"  {_node_line(by_id[nid])}")
    chain.reverse()
    if stats is not None:
        stats["hits"] = (len(chain) - 1) // 2
    return f"{(len(chain) - 1) // 2} hop(s):\n" + "\n".join(chain)


# ── the explorer step: await_code_graph → explore_code → record_graph_context ─────

#: The explorer's answer, in the run directory: what research, strategy, design and the
#: builder are given as the code the change touches.
GRAPH_CONTEXT = "graph-context.md"
#: The marker a runtime writes when the explorer did not finish (an optional branch).
EXPLORER_MISSING = "reviews/explore_code-missing.txt"
EXPLORER_NODE = "explore_code"
#: The explorer prompt's first line (`design.EXPLORER_PROMPT`).
EXPLORER_MARK = "You are the code explorer."
#: How much of the explorer's file the later prompts carry; the rest stays in the file.
GRAPH_CONTEXT_MAX_CHARS = 6000
_DEFAULT_WAIT_S = 120

_USAGE_TEMPLATE = (
    "`{cmd} graph query WORDS... --depth 2 --graph {dir}` finds the code nodes matching "
    "the words, with their neighbours; `{cmd} graph explain NAME --graph {dir}` lists a "
    "node's callers, callees, importers and contents; `{cmd} graph path A B --graph {dir}` "
    "shows the chain of code edges between two nodes; `{cmd} graph callers NAME --graph "
    "{dir}` lists only the callers."
)

_GRAPH_HINT = (
    "run `{cmd} graph query|explain|path|callers ... --graph {dir}` (about a second each). "
    "A graph answer is a lead, not evidence: confirm it in the code."
)
#: After a code context: for what it does not cover.
_FOLLOWUP_TEMPLATE = (
    "## Code graph follow-ups\nFor code the code context above does not cover, " + _GRAPH_HINT)
#: A ready graph with no code context (the explorer failed or wrote nothing).
_GRAPH_ONLY_TEMPLATE = "## Code graph\nTo find code and its callers, " + _GRAPH_HINT


def _knob(ctx: Any, name: str, default: Any) -> Any:
    knobs = ctx.knobs if hasattr(ctx, "knobs") else {}
    value = knobs.get(name, default)
    return value.get("value", default) if isinstance(value, dict) else value


def _run_dir(ctx: Any) -> Path:
    return ctx.workdir if hasattr(ctx, "workdir") else Path(ctx.path(".")).resolve()


def _graph_state(graph_dir: Path) -> tuple[str, dict[str, Any]]:
    status = _read_status(graph_dir)
    state = str(status.get("status", "absent"))
    return (_building_state(status) if state == "building" else state), status


def await_code_graph(ctx: Any) -> dict[str, Any]:
    """Deterministic op: wait, at most ``code_graph_wait_s``, for the background build, and
    say whether the explorer has a graph. Never fails: no graph is a state, not an error.

    It also removes an earlier visit's code context and missing marker, so
    `record_graph_context` reads only what this visit's explorer wrote.
    """
    ctx.path(GRAPH_CONTEXT).unlink(missing_ok=True)
    ctx.path(EXPLORER_MISSING).unlink(missing_ok=True)
    if not _knob(ctx, "code_graph", True):
        return {"code_graph_state": "off (code_graph=0)", "code_graph_usage": ""}
    graph_dir = _run_dir(ctx) / GRAPH_SUBDIR
    wait_s = int(_knob(ctx, "code_graph_wait_s", _DEFAULT_WAIT_S))
    deadline = time.monotonic() + wait_s
    state, status = _graph_state(graph_dir)
    while state == "building" and time.monotonic() < deadline:
        time.sleep(1)
        state, status = _graph_state(graph_dir)
    if state == "building":
        return {"code_graph_state": f"still building after {wait_s} s", "code_graph_usage": ""}
    if state != "ready":
        return {"code_graph_state": state, "code_graph_usage": ""}
    cmd = _find_refactory_cmd() or "refactory"
    return {
        "code_graph_state": f"ready: {status.get('nodes', '?')} nodes, "
                            f"{status.get('edges', '?')} edges",
        "code_graph_usage": _USAGE_TEMPLATE.format(cmd=cmd, dir=graph_dir),
    }


def _explorer_queries(graph_dir: Path) -> int:
    """The answered queries the explorer made, counted as `usage` counts them: under the
    node its worker named, or, for a query with no name (a route that sets none, such as
    the spine's CEO subagents), when the explorer was the one node running."""
    try:
        lines = (graph_dir / QUERY_LOG).read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    entries = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except ValueError:
            continue
    windows = (_node_windows(graph_dir.parent.parent)
               if any(not entry.get("node") for entry in entries) else [])
    count = 0
    for entry in entries:
        if entry.get("status") != "ready":
            continue
        node = entry.get("node")
        if not node:
            at = entry.get("at", 0)
            inside = {w[2] for w in windows if w[0] <= at <= w[1]}
            node = inside.pop() if len(inside) == 1 else None
        if node == EXPLORER_NODE:
            count += 1
    return count


def record_graph_context(ctx: Any) -> dict[str, Any]:
    """Deterministic op: the explorer's code context, headed by what it was made from: the
    code graph (with the explorer's query count), or file exploration and why. The file
    keeps all of it; the prompts' fact is cut at ``GRAPH_CONTEXT_MAX_CHARS`` and points at
    the file. Never fails; with no file the later prompts carry nothing.

    It also settles the on-demand query paragraph for the steps after it: one that points
    past the code context when the graph is ready (or offers the graph alone when there is
    no context), none when the graph failed or is off, and the build's own paragraph while
    the build is still running.
    """
    graph_dir = _run_dir(ctx) / GRAPH_SUBDIR
    state, _ = _graph_state(graph_dir) if graph_dir.is_dir() else ("absent", {})

    body = ""
    if ctx.exists(GRAPH_CONTEXT):
        try:
            body = ctx.read_text(GRAPH_CONTEXT).strip()
        except (OSError, UnicodeDecodeError):
            body = ""

    paragraph = str(ctx.facts.get("code_graph_paragraph", ""))
    if state == "ready":
        template = _FOLLOWUP_TEMPLATE if body else _GRAPH_ONLY_TEMPLATE
        paragraph = template.format(cmd=_find_refactory_cmd() or "refactory", dir=graph_dir)
    elif state != "building":
        paragraph = ""

    if not body:
        why = "the explorer step wrote nothing"
        if ctx.exists(EXPLORER_MISSING):
            why = "the explorer step did not finish: " + ctx.read_text(EXPLORER_MISSING).strip()[:200]
        # The file stays, saying why, so the steps that declare it read the reason.
        ctx.write_text(GRAPH_CONTEXT, f"No code context: {why}.\n")
        return {"graph_context": "", "graph_context_source": f"none: {why}",
                "code_graph_paragraph": paragraph}

    if body.startswith("Made from "):
        # Recorded already (a resumed visit): the heading is this op's to write, and the
        # headings below it are nested already.
        body = body.split("\n\n", 1)[1].strip() if "\n\n" in body else ""
    else:
        # The prompts carry it under their own `## Code context` heading: its headings nest.
        body = re.sub(r"(?m)^(#{1,5}) ", r"#\1 ", body)
    queries = _explorer_queries(graph_dir)
    asked = str(ctx.facts.get("code_graph_state", state))
    if queries:
        source = f"the code graph ({asked}; {queries} explorer queries)"
    elif asked.startswith("ready"):
        source = "file exploration: the code graph was ready, but the explorer ran no queries"
    else:
        source = f"file exploration: no code graph ({asked})"
    full = f"Made from {source}.\n\n{body}\n"
    ctx.write_text(GRAPH_CONTEXT, full)
    text = full
    if len(body) > GRAPH_CONTEXT_MAX_CHARS:
        text = (f"Made from {source}.\n\n"
                + body[:GRAPH_CONTEXT_MAX_CHARS].rsplit("\n", 1)[0]
                + f"\n\n(cut at {GRAPH_CONTEXT_MAX_CHARS} characters; all of it is in "
                f"{GRAPH_CONTEXT})\n")
    return {"graph_context": text, "graph_context_source": source,
            "code_graph_paragraph": paragraph}
