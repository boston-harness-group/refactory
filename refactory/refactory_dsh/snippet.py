"""Code nodes: an op or a gate whose Python an optimizer writes, run sandboxed.

A node of the graph runs a ``module:function`` an author wrote, and until now an
optimizer could only reuse one an existing node already named (rule V8). A code node
is one generic callable whose body is data -- the node's ``code`` field, normally
bound to the node's own ``<id>__code`` knob -- so an edit can add a step that
computes something and a later edit can rewrite it, like a prompt.

Three callables, each an ordinary node callable both runtimes invoke the same way
(the spine through ``python -m refactory_dsh.run``, the walker in its forkserver
child), so the two cannot disagree about what a code node does:

- :data:`RUN` (``op``): ``main(inp)`` returns ``{"facts": {...}, "files": {...}}``;
- :data:`GATE` (``gate``): ``main(inp)`` returns ``{"verdict": label, "reason": text}``,
  the label one of the gate's out-edge conditions;
- :data:`VERDICT` (``gate``, no code): the verdict is the first word of a declared
  upstream output -- the gate's first ``required_reads`` file, else its first
  ``inputs`` fact -- written by an agent or a code node; the rest is the reason.

``inp`` is ``{"files": {read name: text}, "facts": {input name: value}}``: the
node's declared reads that exist, and its declared ``inputs`` facts that are set.
The snippet never sees the context, the run directory, any other fact, the knobs or
the environment. A trusted wrapper (this module) writes its outputs through the
context, so the declared writes and facts are still enforced.

The snippet runs in :func:`execute`: a user, network and mount namespace
(``unshare -Urnm``), empty tmpfs over every place a run or a person keeps files,
the system Python in isolated mode with no site packages, JSON on stdin and stdout,
resource limits and a wall clock. Nothing falls back to running unsandboxed: where
the sandbox is not available (:func:`available`) a code node fails, and the request
does not offer code nodes.

What the static check refuses (rule V18) is in :func:`static_issues`; it never
executes the code.
"""

from __future__ import annotations

import ast
import functools
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

#: The code op, the code gate and the verdict gate, as ``call``/``evaluator`` names them.
RUN = "refactory_dsh.snippet:run"
GATE = "refactory_dsh.snippet:gate"
VERDICT = "refactory_dsh.snippet:verdict"
#: The callables that run a node's ``code``.
CODE_CALLS = (RUN, GATE)

#: The most bytes a node's code may hold (it travels in the declarations and the knobs).
MAX_CODE_BYTES = 4096
#: The modules a snippet may import: pure, standard-library text and number tools.
ALLOWED_IMPORTS = ("re", "json", "math", "collections", "itertools", "statistics",
                   "fractions", "unicodedata", "string")
#: The allowlist as every text that states it writes it (the rules, the request).
ALLOWED_IMPORTS_TEXT = ", ".join(ALLOWED_IMPORTS)
#: Names a snippet may not use at all: every way out of pure computation on ``inp``.
FORBIDDEN_NAMES = ("open", "__import__", "exec", "eval", "compile", "getattr", "setattr",
                   "delattr", "globals", "locals", "vars", "input", "breakpoint", "help",
                   "exit", "quit", "memoryview", "__builtins__", "__loader__", "__spec__")

#: The wall clock of one snippet, in seconds (well under any node's command timeout).
WALL_S = 10
#: Its CPU seconds, heap/address space and largest file.
CPU_S = 10
MEMORY_MB = 512
FILE_MB = 1
#: The most output a snippet may return, in bytes of JSON.
MAX_OUTPUT_BYTES = 1_000_000

#: The interpreter a snippet runs under: the system one, which lives outside every
#: directory the sandbox hides (a project's virtualenv does not).
PYTHON = "/usr/bin/python3"

#: Directories hidden behind an empty tmpfs: where people, runs and temporary files
#: live. The run directory and the project are added when they are elsewhere.
HIDDEN_ROOTS = ("/home", "/root", "/tmp", "/var/tmp", "/dev/shm", "/run/user", "/mnt",
                "/media", "/srv", "/opt", "/workspace", "/new_data", "/data")

#: The exit code the sandbox's shell uses when it could not hide a directory.
_MOUNT_FAILED = 97

#: The shell that sets the sandbox up, inside the new namespaces: hide each directory,
#: then run the driver from the (empty) /tmp. Paths arrive as arguments, never in text.
_SHELL = (
    'py="$1"; drv="$2"; shift 2; '
    'for d in "$@"; do mount -t tmpfs -o size=8m,mode=0755 tmpfs "$d" || exit 97; done; '
    'cd /tmp || exit 97; '
    'exec "$py" -I -S -c "$drv"'
)

#: The driver that runs inside: read code and input, call ``main``, write the result.
#: What the snippet prints goes to stderr, so it cannot forge the result.
_DRIVER = """\
import sys, json
out = sys.stdout
sys.stdout = sys.stderr
try:
    data = json.loads(sys.stdin.read())
    space = {"__name__": "snippet"}
    exec(compile(data["code"], "<snippet>", "exec"), space)
    text = json.dumps({"ok": True, "result": space["main"](data["inp"])})
except BaseException as exc:
    text = json.dumps({"ok": False, "error": (type(exc).__name__ + ": " + str(exc))[:2000]})
out.write(text)
out.flush()
"""


class SnippetError(RuntimeError):
    """A code node could not run, or returned something its node cannot use."""


# ── the static check (V18) ────────────────────────────────────────────────


def static_issues(code: Any) -> list[str]:
    """Everything wrong with ``code`` that can be seen without running it (rule V18).

    It parses; it defines a module-level ``main``; it is at most
    :data:`MAX_CODE_BYTES`; it imports only :data:`ALLOWED_IMPORTS`; it uses none of
    :data:`FORBIDDEN_NAMES` and no dunder name or attribute (``x.__class__``).
    """
    if not isinstance(code, str) or not code.strip():
        return ["has no code: a code node's code defines main(inp)"]
    size = len(code.encode())
    if size > MAX_CODE_BYTES:
        return [f"code is {size} bytes; the most a code node holds is {MAX_CODE_BYTES}"]
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"code does not parse: line {exc.lineno}: {exc.msg}"]
    problems: list[str] = []
    if not any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body):
        problems.append("code defines no module-level main(inp)")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"imports {alias.name!r}; allowed: {', '.join(ALLOWED_IMPORTS)}")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                problems.append(f"imports from {node.module!r}; allowed: {', '.join(ALLOWED_IMPORTS)}")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or (node.id.startswith("__") and node.id.endswith("__")):
                problems.append(f"uses {node.id!r}, which a code node may not")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            problems.append(f"reads the attribute {node.attr!r}; a code node uses no dunder attributes")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append("declares global or nonlocal names")
        elif isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith)):
            problems.append("uses async code")
    return sorted(set(problems))


# ── the sandbox ───────────────────────────────────────────────────────────


def _environment() -> dict[str, str]:
    """What the snippet's process sees: the language settings `sandbox.scrub` keeps,
    a fixed PATH and a HOME inside the empty /tmp. Never a knob, a node manifest or a
    credential."""
    from refactory_dsh.sandbox import scrub

    kept = {name: value for name, value in scrub().items() if name in ("LANG", "LC_ALL")}
    return {**kept, "PATH": "/usr/bin:/bin", "HOME": "/tmp"}


def _hidden(extra: tuple[str, ...] = ()) -> list[str]:
    """The directories to hide: :data:`HIDDEN_ROOTS` that exist, and each extra path
    (the run directory, the project, the current directory) that is not inside one."""
    roots = [root for root in HIDDEN_ROOTS if os.path.isdir(root)]
    for path in extra:
        if not path:
            continue
        resolved = os.path.realpath(path)
        if not os.path.isdir(resolved) or resolved == "/":
            continue
        if not any(resolved == root or resolved.startswith(root.rstrip("/") + "/") for root in roots):
            roots.append(resolved)
    # A directory inside another one is hidden with it; mounting it too would fail
    # once its parent is covered.
    roots = sorted(set(roots))
    return [root for root in roots
            if not any(root != other and root.startswith(other.rstrip("/") + "/") for other in roots)]


def _command(extra: tuple[str, ...]) -> list[str]:
    return [shutil.which("unshare") or "unshare", "-Urnm", "--", "/bin/sh", "-c", _SHELL, "sh",
            PYTHON, _DRIVER, *_hidden(extra)]


def _sandboxed(code: str, inp: Any, *, timeout_s: float, extra: tuple[str, ...]) -> Any:
    from refactory_dsh.sandbox import _limits

    try:
        payload = json.dumps({"code": code, "inp": inp})
    except (TypeError, ValueError) as exc:
        raise SnippetError(f"the code node's input is not JSON: {exc}") from exc
    try:
        proc = subprocess.run(
            _command(extra), input=payload, capture_output=True, text=True,
            timeout=timeout_s, env=_environment(), cwd="/",
            preexec_fn=_limits(MEMORY_MB, CPU_S, FILE_MB),
        )
    except subprocess.TimeoutExpired as exc:
        raise SnippetError(f"the code ran over its {timeout_s:g} s limit") from exc
    except OSError as exc:
        raise SnippetError(f"the sandbox could not start: {exc}") from exc
    if proc.returncode == _MOUNT_FAILED:
        raise SnippetError("the sandbox could not hide the file system: "
                           + (proc.stderr.strip().splitlines() or ["mount failed"])[-1])
    if len(proc.stdout) > MAX_OUTPUT_BYTES:
        raise SnippetError(f"the code returned more than {MAX_OUTPUT_BYTES} bytes")
    try:
        reply = json.loads(proc.stdout)
    except ValueError:
        tail = (proc.stderr.strip().splitlines() or [f"exit {proc.returncode}"])[-1]
        raise SnippetError(f"the code did not finish (exit {proc.returncode}): {tail[:500]}") from None
    if not isinstance(reply, dict) or not reply.get("ok"):
        raise SnippetError(f"the code raised {str((reply or {}).get('error', 'an error'))[:2000]}")
    return reply.get("result")


@functools.lru_cache(maxsize=1)
def _probe() -> tuple[bool, str]:
    if not sys.platform.startswith("linux"):
        return False, f"code nodes need Linux namespaces; this is {sys.platform}"
    if shutil.which("unshare") is None:
        return False, "the unshare command is not installed"
    if not os.path.exists(PYTHON):
        return False, f"{PYTHON} does not exist"
    try:
        result = _sandboxed("def main(inp):\n    return inp['facts']['x'] + 1\n",
                            {"files": {}, "facts": {"x": 1}}, timeout_s=WALL_S, extra=())
    except SnippetError as exc:
        return False, str(exc)
    if result != 2:
        return False, f"the sandbox probe returned {result!r}, not 2"
    return True, ""


def available() -> tuple[bool, str]:
    """Whether code nodes can run here, and why not: a probe that runs a tiny snippet
    in the real sandbox, once per process."""
    return _probe()


def execute(code: str, inp: Any, *, timeout_s: float = WALL_S,
            hide: tuple[str, ...] = ()) -> Any:
    """``main(inp)`` of ``code``, run in the sandbox; :class:`SnippetError` when it
    cannot run there, fails, or runs over its limits. ``hide`` names directories to
    hide besides :data:`HIDDEN_ROOTS` (the run directory, the project)."""
    ok, why = available()
    if not ok:
        raise SnippetError(f"code nodes cannot run here: {why}")
    issues = static_issues(code)
    if issues:
        raise SnippetError("the code is refused (rule V18): " + "; ".join(issues))
    return _sandboxed(code, inp, timeout_s=timeout_s, extra=hide)


# ── what a node declares ──────────────────────────────────────────────────


def is_code_node(node: Any) -> bool:
    """Whether a node (a ``Node`` or its state) runs a node's code."""
    get = node.get if isinstance(node, Mapping) else (lambda name, default="": getattr(node, name, default))
    if get("kind", "") == "op":
        return get("call", "") == RUN
    if get("kind", "") == "gate":
        return get("evaluator", "") == GATE
    return False


def spec(node: Any, labels: Any, knobs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """What a code or verdict node's callable needs, beyond the context: its code (the
    bound knob's value when the run carries one), the knob it is bound to, its declared
    ``inputs`` and ``required_reads``, and -- for a gate -- its out-edge labels. Built
    from the graph on both runtimes: the walker hands it to its child, the lowering
    writes it into the declarations the spine's runner reads."""
    return {
        "code": str(node.bound("code", knobs or {}) or ""),
        "code_knob": node.bound_knob("code"),
        "inputs": list(node.inputs),
        "required_reads": list(node.required_reads),
        "labels": sorted({str(label) for label in labels if label}),
    }


def _spec(ctx: Any) -> dict[str, Any]:
    found = getattr(ctx, "snippet", None)
    if not isinstance(found, dict):
        raise SnippetError(f"node {ctx.node_id!r} runs a code-node callable but the runtime "
                           "gave it no code-node declaration")
    return found


def _code(ctx: Any, found: dict[str, Any]) -> str:
    """The node's code: its knob's value in this run when it is bound to one, else the
    declared code."""
    knob = found.get("code_knob") or ""
    value = ctx.knobs.get(knob) if knob else None
    return str(value) if isinstance(value, str) and value else str(found.get("code") or "")


def _input(ctx: Any, found: dict[str, Any]) -> dict[str, Any]:
    files = {}
    for name in ctx.declared_reads:
        if ctx.exists(name):
            try:
                files[name] = ctx.read_text(name)
            except (UnicodeDecodeError, IsADirectoryError):
                continue
    facts = {name: ctx.facts[name] for name in found.get("inputs") or () if name in ctx.facts}
    return {"files": files, "facts": facts}


def _hide(ctx: Any) -> tuple[str, ...]:
    return (str(ctx.workdir), os.environ.get("REFACTORY_PROJECT", ""), os.getcwd())


def _label(found: dict[str, Any], label: Any, node_id: str) -> str:
    labels = list(found.get("labels") or ())
    wanted = str(label or "").strip()
    for known in labels:
        if known.lower() == wanted.lower():
            return known
    raise SnippetError(f"gate {node_id!r} answered {wanted!r}, which is not one of its "
                       f"out-edge conditions {labels}")


def run(ctx: Any) -> dict[str, Any]:
    """The code op: run the node's code on its declared inputs; write the files it
    returns (each a declared write) and return the facts it returns (each declared)."""
    from refactory_dsh.nn import ExecutionError

    found = _spec(ctx)
    result = execute(_code(ctx, found), _input(ctx, found), hide=_hide(ctx))
    if result is None:
        result = {}
    if not isinstance(result, dict) or set(result) - {"facts", "files"}:
        raise SnippetError(f"code op {ctx.node_id!r} returned {str(result)[:200]!r}; main(inp) "
                           "returns {'facts': {...}, 'files': {...}}")
    facts = result.get("facts") or {}
    files = result.get("files") or {}
    if not isinstance(facts, dict) or not isinstance(files, dict):
        raise SnippetError(f"code op {ctx.node_id!r}: 'facts' and 'files' are objects")
    writes = set(ctx._declared)  # noqa: SLF001 - the node's declared writes
    for name, text in files.items():
        if name not in writes:
            raise ExecutionError(f"code op {ctx.node_id!r} returned the file {name!r}, which it "
                                 f"does not declare; declared writes: {sorted(writes)}")
        if not isinstance(text, str):
            raise SnippetError(f"code op {ctx.node_id!r}: file {name!r} is not text")
        ctx.write_text(name, text)
    declared = set(ctx._declared_facts)  # noqa: SLF001
    extra = sorted(set(facts) - declared)
    if extra:
        raise ExecutionError(f"code op {ctx.node_id!r} returned the fact(s) {extra}, which it "
                             f"does not declare; declared facts: {sorted(declared)}")
    return dict(facts)


def gate(ctx: Any) -> Any:
    """The code gate: run the node's code; its ``verdict`` (one of the gate's out-edge
    conditions) routes, its ``reason`` goes to the node a reloop sends control back to."""
    from refactory_dsh.nn import Verdict

    found = _spec(ctx)
    result = execute(_code(ctx, found), _input(ctx, found), hide=_hide(ctx))
    if isinstance(result, str):
        result = {"verdict": result}
    if not isinstance(result, dict) or "verdict" not in result:
        raise SnippetError(f"code gate {ctx.node_id!r} returned {str(result)[:200]!r}; main(inp) "
                           "returns {'verdict': <label>, 'reason': <text>}")
    return Verdict(_label(found, result.get("verdict"), ctx.node_id),
                   reason=str(result.get("reason") or ""))


#: A word of a verdict line: what a label can be.
_WORD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_-]*")


def parse_verdict(text: str, labels: list[str]) -> tuple[str, str] | None:
    """``(label, reason)`` from a verdict text: the first label (any case) among the
    words of its first non-empty line; the reason is what follows it on that line and
    every later line. ``None`` when that line names no label."""
    lines = str(text or "").strip().splitlines()
    if not lines:
        return None
    first, rest = lines[0], lines[1:]
    lower = {label.lower(): label for label in labels}
    for match in _WORD.finditer(first):
        if match.group(0).lower() in lower:
            tail = first[match.end():].lstrip(" \t:-—*.)]")
            reason = "\n".join([tail, *rest]).strip()
            return lower[match.group(0).lower()], reason
    return None


def verdict(ctx: Any) -> Any:
    """The verdict gate: route on a declared upstream output, run no code.

    The source is the gate's first ``required_reads`` file, else its first ``inputs``
    fact -- whatever an agent or a code node upstream wrote there. Its first line names
    the verdict (one of the gate's out-edge conditions); the rest is the reason, which
    a reloop hands to the step it sends control back to.
    """
    from refactory_dsh.nn import Verdict

    found = _spec(ctx)
    labels = list(found.get("labels") or ())
    reads = list(found.get("required_reads") or ())
    inputs = list(found.get("inputs") or ())
    if reads:
        source = f"file {reads[0]!r}"
        text = ctx.read_text(reads[0])
    elif inputs:
        source = f"fact {inputs[0]!r}"
        if inputs[0] not in ctx.facts:
            raise SnippetError(f"verdict gate {ctx.node_id!r}: its input fact {inputs[0]!r} is not set")
        text = str(ctx.facts[inputs[0]])
    else:
        raise SnippetError(f"verdict gate {ctx.node_id!r} declares no required_reads and no "
                           "inputs, so it has nothing to read its verdict from")
    parsed = parse_verdict(text, labels)
    if parsed is None:
        head = (text.strip().splitlines() or [""])[0][:200]
        raise SnippetError(f"verdict gate {ctx.node_id!r}: the first line of {source} ({head!r}) "
                           f"names none of its out-edge conditions {labels}")
    label, reason = parsed
    return Verdict(label, reason=reason)


# ── the shared grader-secret scan (E-secret) ──────────────────────────────


def secret_in(text: str, secrets: Any) -> str | None:
    """The first grader secret ``text`` contains, or ``None``."""
    for secret in sorted(secrets or (), key=lambda s: (-len(s), s)):
        if secret and secret in text:
            return secret
    return None
