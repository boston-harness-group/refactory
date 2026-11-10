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
  ``inputs`` fact -- written by an agent or a code node, or else the word alone on its
  last line (:func:`parse_verdict`); the rest is the reason.

``inp`` is ``{"files": {read name: text}, "facts": {input name: value}}``: the
node's declared reads that exist, and its declared ``inputs`` facts that are set.
The snippet never sees the context, the run directory, any other fact, the knobs or
the environment. A trusted wrapper (this module) writes its outputs through the
context, so the declared writes and facts are still enforced.

The snippet runs in :func:`execute`, and the sandbox is the security boundary: user,
network, mount and PID namespaces (``unshare -Urnmp --fork --kill-child
--mount-proc``) under ``no_new_privs``; a fresh ``/proc`` showing only the sandbox's
own processes, with ``/proc/sys`` hidden; empty tmpfs over every place a run, a
person or a service keeps files or sockets (``/home``, ``/tmp``, ``/run``,
``/var/lib`` ...); the system Python in isolated mode with no site packages; JSON on
stdin and stdout; CPU, memory, file-size and process-count limits that fail closed;
a wall clock, after which the whole process group is killed (and the PID namespace
with it). Nothing falls back to running unsandboxed: where the sandbox is not
available (:func:`available`) a code node fails, and the request does not offer code
nodes.

The static check (rule V18, :func:`static_issues`) is defence in depth: it never
executes the code, and it refuses what pure computation on ``inp`` never needs.
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
#: Attribute names a snippet may never use even where they exist: string formatting
#: reaches attributes by name (``"{0.__class__}".format(x)``), which the dunder ban
#: cannot see.
DENIED_ATTRIBUTES = ("format", "format_map", "Formatter", "Template", "vformat",
                     "get_field", "convert_field")
#: The allowlist as every text that states it writes it (the rules, the request).
ALLOWED_IMPORTS_TEXT = ", ".join(ALLOWED_IMPORTS)
#: Names a snippet may not use at all: every way out of pure computation on ``inp``.
FORBIDDEN_NAMES = ("open", "__import__", "exec", "eval", "compile", "getattr", "setattr",
                   "delattr", "globals", "locals", "vars", "input", "breakpoint", "help",
                   "exit", "quit", "memoryview", "__builtins__", "__loader__", "__spec__")

#: The wall clock of one snippet, in seconds (well under any node's command timeout).
WALL_S = 10
#: Its CPU seconds, heap/address space, largest file and most processes.
CPU_S = 10
MEMORY_MB = 512
FILE_MB = 1
NPROC = 32
#: The most output a snippet may return, in bytes of JSON.
MAX_OUTPUT_BYTES = 1_000_000

#: The interpreter a snippet runs under: the system one, which lives outside every
#: directory the sandbox hides (a project's virtualenv does not).
PYTHON = "/usr/bin/python3"
#: The shell that sets the sandbox up inside the namespaces.
BASH = "/bin/bash"

#: Directories hidden behind an empty tmpfs: where people, runs and temporary files
#: live. The run directory and the project are added when they are elsewhere.
HIDDEN_ROOTS = ("/home", "/root", "/tmp", "/var/tmp", "/dev/shm", "/run", "/var/run",
                "/var/lib", "/mnt", "/media", "/srv", "/opt", "/workspace", "/new_data",
                "/data", "/proc/sys")

#: The exit codes the sandbox's shell uses when it could not hide a directory, and
#: when it could not cap the process count.
_MOUNT_FAILED = 97
_NPROC_FAILED = 96

#: The shell that sets the sandbox up, inside the new namespaces (where it is PID 1,
#: under a fresh /proc): cap the process count (here, where it counts this namespace's
#: processes, not the user's on the host), hide each directory, then run the driver
#: from the (empty) /tmp. Paths arrive as arguments, never in text.
_SHELL = (
    'py="$1"; drv="$2"; n="$3"; shift 3; '
    'ulimit -u "$n" || exit 96; '
    'for d in "$@"; do mount -n -t tmpfs -o size=8m,mode=0755 tmpfs "$d" || exit 97; done; '
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
        elif isinstance(node, ast.Attribute) and node.attr not in allowed_attributes():
            problems.append(f"uses the attribute {node.attr!r}, which is not among the methods "
                            "of str, bytes, list, dict, set, tuple, int, float and the allowed "
                            "modules' public names (string formatting is refused)")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append("declares global or nonlocal names")
        elif isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith)):
            problems.append("uses async code")
    return sorted(set(problems))


@functools.lru_cache(maxsize=1)
def allowed_attributes() -> frozenset:
    """The attribute names a snippet may use: the public methods of the builtin value
    types and of what the allowed modules hand out (match objects, patterns, counters,
    fractions ...), and the allowed modules' public names, less
    :data:`DENIED_ATTRIBUTES`. An allowlist, so a name nobody thought of is refused."""
    import collections
    import fractions
    import importlib
    import re as re_module
    import types

    sources: list[Any] = [str, bytes, list, dict, set, frozenset, tuple, int, float, complex,
                          bool, collections.Counter, collections.deque,
                          collections.defaultdict, collections.OrderedDict,
                          fractions.Fraction, type(re_module.compile("")),
                          type(re_module.match("", ""))]
    sources += [importlib.import_module(name) for name in ALLOWED_IMPORTS]
    # A module an allowed module happens to import (`re.enum`, `json.decoder`,
    # `collections.abc`, `statistics.random` ...) is a way to reach modules the import
    # allowlist refuses, so a name whose value is a module is never allowed.
    names = {name for source in sources for name in dir(source) if not name.startswith("_")}
    modules = {name for source in sources for name in dir(source)
               if isinstance(getattr(source, name, None), types.ModuleType)}
    return frozenset(names - modules - set(DENIED_ATTRIBUTES))


# ── the sandbox ───────────────────────────────────────────────────────────


def _environment() -> dict[str, str]:
    """What the snippet's process sees: the language settings `sandbox.scrub` keeps,
    a fixed PATH and a HOME inside the empty /tmp. Never a knob, a node manifest or a
    credential."""
    from refactory_dsh.sandbox import scrub

    kept = {name: value for name, value in scrub().items() if name in ("LANG", "LC_ALL")}
    return {**kept, "PATH": "/usr/bin:/bin", "HOME": "/tmp"}


def _hidden(extra: tuple[str, ...] = ()) -> list[str]:
    """The directories to hide: :data:`HIDDEN_ROOTS` that exist (a link, such as
    ``/var/run``, by what it points to), and each extra path (the run directory, the
    project, the current directory) that is not inside one."""
    roots = [os.path.realpath(root) for root in HIDDEN_ROOTS if os.path.isdir(root)]
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
    # /proc/sys is hidden after the fresh /proc is mounted, so it goes last.
    hidden = [path for path in _hidden(extra) if path != "/proc/sys"] + ["/proc/sys"]
    # bash, not sh: `ulimit -u` is bash's (dash spells the process cap differently).
    return [shutil.which("unshare") or "unshare", "-Urnmp", "--fork", "--kill-child",
            "--mount-proc", "--", BASH, "-c", _SHELL, "bash", PYTHON, _DRIVER, str(NPROC),
            *hidden]


def _confine() -> None:  # pragma: no cover - runs in the forked child
    """The child's preexec: its own session (so a timeout kills the whole group),
    ``no_new_privs``, and the CPU, memory and file-size caps. Unlike `sandbox._limits`
    it raises when a cap cannot be set: this sandbox fails closed."""
    import ctypes
    import resource

    os.setsid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")
    for which, value in ((resource.RLIMIT_CPU, CPU_S), (resource.RLIMIT_FSIZE, FILE_MB << 20),
                         (resource.RLIMIT_AS, MEMORY_MB << 20),
                         (resource.RLIMIT_DATA, MEMORY_MB << 20)):
        resource.setrlimit(which, (value, value))


def _kill_group(proc: subprocess.Popen) -> None:
    import signal

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _sandboxed(code: str, inp: Any, *, timeout_s: float, extra: tuple[str, ...]) -> Any:
    try:
        payload = json.dumps({"code": code, "inp": inp})
    except (TypeError, ValueError) as exc:
        raise SnippetError(f"the code node's input is not JSON: {exc}") from exc
    try:
        proc = subprocess.Popen(
            _command(extra), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, env=_environment(), cwd="/",
            preexec_fn=_confine,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SnippetError(f"the sandbox could not start: {exc}") from exc
    try:
        stdout, stderr = proc.communicate(payload, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        _kill_group(proc)
        proc.communicate()
        raise SnippetError(f"the code ran over its {timeout_s:g} s limit") from exc
    finally:
        # Whatever the snippet left running in its group goes with it.
        _kill_group(proc)
    proc = subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
    if proc.returncode in (_MOUNT_FAILED, _NPROC_FAILED):
        what = "hide the file system" if proc.returncode == _MOUNT_FAILED else "cap its processes"
        raise SnippetError(f"the sandbox could not {what}: "
                           + (proc.stderr.strip().splitlines() or ["failed"])[-1])
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
    for needed in (PYTHON, BASH):
        if not os.path.exists(needed):
            return False, f"{needed} does not exist"
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


#: The verdict that ends the run with no edge for it, as the router takes it
#: (`nn.executor.route_verdict`, the spine's ``Spine.verdict``): a code or verdict gate
#: may answer it too. (``proceed`` with no proceed edge falls through to an
#: unconditional edge, which a gate never has (V5), so it is not offered.)
HALT = "halt"


def answers(found: dict[str, Any]) -> list[str]:
    """What a code or verdict gate may answer: its out-edge conditions, and ``halt``."""
    labels = list(found.get("labels") or ())
    return labels + ([HALT] if HALT not in {label.lower() for label in labels} else [])


def _label(found: dict[str, Any], label: Any, node_id: str) -> str:
    known = answers(found)
    wanted = str(label or "").strip()
    for option in known:
        if option.lower() == wanted.lower():
            return option
    raise SnippetError(f"gate {node_id!r} answered {wanted!r}, which is not one of its "
                       f"out-edge conditions or halt {known}")


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


#: What may come before a verdict's first word: markdown and quoting.
_LEAD = " \t*#>-_`'\"“”‘’:([{"
#: What ends the first word and may come after it before the reason.
_WORD = re.compile(r"[^\s:;,.!?*`'\"“”‘’)\]}]+")
#: What may close a last line's verdict word: markdown and quoting (``**proceed**``).
_CLOSE = " \t*_`'\"“”‘’)]}"
#: What may follow a last line's verdict word, alone: trailing punctuation.
_TRAIL = _CLOSE + ".,;:!?"
#: What separates a last line's verdict word from its reason (``reloop: show B``).
_SEP = ":—–-."
#: What a reason starts after, once its verdict word is read.
_REASON_LEAD = " \t:;,.!?*`'\"“”‘’)]}-—–"


def is_answer_word(label: Any) -> bool:
    """Whether a gate label can be a code or verdict gate's answer: one word, as
    :func:`parse_verdict` reads a first word or a last line's word (no space, no leading
    markdown or quotes, no trailing punctuation). V5 refuses any other label on such a
    gate's out-edges."""
    text = str(label or "")
    return bool(text) and text[0] not in _LEAD and _WORD.fullmatch(text) is not None


def _lead_word(line: str) -> tuple[str, str]:
    """``(word, rest)``: a text's first word after leading markdown or quotes, and what
    follows it; ``("", text)`` when it starts with no word."""
    body = line.strip().lstrip(_LEAD)
    match = _WORD.match(body)
    return (match.group(0), body[match.end():]) if match else ("", body)


def parse_verdict(text: str, labels: list[str]) -> tuple[str, str] | None:
    """``(label, reason)`` from a verdict text, matched against ``labels`` in any case.

    1. Its first word -- after leading markdown or quotes (``**``, ``#``, ``-``, ``"``,
       ``:``) -- when that is a label; the reason is the rest of the text.
    2. Otherwise its last non-empty line, when that (after the same leading markdown or
       quotes) is a label alone, with at most closing markdown and trailing punctuation
       (``proceed.``, ``**proceed**``), or a label then a separator (``:``, ``—``,
       ``-``, ``.``) and more (``> reloop — show B``): the standard "reason first,
       verdict last" review. The reason is the text before that line, then what follows
       the separator.

    ``None`` otherwise -- nothing between the first word and the last line is read:
    "Do not proceed" names no verdict, rather than proceed, and neither does a last line
    "reloop the check if unsure"."""
    lower = {label.lower(): label for label in labels}
    body = str(text or "").strip()
    word, rest = _lead_word(body)
    label = lower.get(word.lower())
    if label is not None:
        return label, rest.lstrip(_REASON_LEAD).strip()
    lines = body.splitlines()
    if not lines:
        return None
    word, rest = _lead_word(lines[-1])
    label = lower.get(word.lower())
    if label is None:
        return None
    rest = rest.lstrip(_CLOSE)
    if rest.strip(_TRAIL):
        if rest[0] not in _SEP:
            return None
        rest = rest.lstrip(_REASON_LEAD).strip()
    else:
        rest = ""
    reason = "\n".join(part for part in ("\n".join(lines[:-1]).strip(), rest) if part)
    return label, reason


def verdict(ctx: Any) -> Any:
    """The verdict gate: route on a declared upstream output, run no code.

    The source is the gate's first ``required_reads`` file, else its first ``inputs``
    fact -- whatever an agent or a code node upstream wrote there. Its first word, or
    else the word alone on its last line (:func:`parse_verdict`), names the verdict (one
    of the gate's out-edge conditions, or halt); the rest is the reason, which a reloop
    hands to the step it sends control back to.
    """
    from refactory_dsh.nn import Verdict

    found = _spec(ctx)
    labels = answers(found)
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
        lines = text.strip().splitlines() or [""]
        raise SnippetError(f"verdict gate {ctx.node_id!r}: neither the first word of {source} "
                           f"({_lead_word(lines[0])[0]!r}, in its first line {lines[0][:200]!r}) "
                           f"nor its last line ({lines[-1][-200:]!r}) is one of its answers "
                           f"{labels} (its out-edge conditions, or halt); the writer starts the "
                           "text with the answer, or puts it alone on the last line")
    label, reason = parsed
    return Verdict(label, reason=reason)


# ── the shared grader-secret scan (E-secret) ──────────────────────────────


def _normal(text: str) -> str:
    return " ".join(str(text).casefold().split())


def secret_in(text: str, secrets: Any) -> str | None:
    """The first grader secret ``text`` contains, compared case- and whitespace-blind,
    or ``None``. Best effort: it catches a value pasted into code, not one rebuilt from
    pieces; the control that holds is that grader fields never reach a node's inputs
    (V17 and the stripped item file)."""
    haystack = _normal(text)
    for secret in sorted(secrets or (), key=lambda s: (-len(s), s)):
        needle = _normal(secret)
        if needle and needle in haystack:
            return secret
    return None
