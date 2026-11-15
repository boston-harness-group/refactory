"""Count the processes refactory starts, when asked to: a measurement, never a behaviour.

A process launch costs under a millisecond on Linux and 30-50 ms on a Mac whose
endpoint-security agent inspects every exec, so the number of launches on a path is
what decides whether it runs at the same speed on both. :data:`ENV` names a file; when
it is set, every refactory Python process appends one line there when it starts and one
for each process it launches, and Node processes it starts do the same through
``spawn_count.cjs`` (put on ``NODE_OPTIONS``). Unset, nothing here runs.

    REFACTORY_SPAWN_LOG=/tmp/spawns.jsonl uv run pytest refactory/tests/test_x.py
    uv run python -m refactory_dsh.spawn_count /tmp/spawns.jsonl
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

#: The file launches are appended to; nothing is counted when it is unset.
ENV = "REFACTORY_SPAWN_LOG"

#: The audit events that start a process (Python's own `subprocess`, `os.posix_spawn`,
#: the `os.exec*`/`os.spawn*` family and `os.system`).
EVENTS = frozenset({"subprocess.Popen", "os.posix_spawn", "os.exec", "os.spawn", "os.system"})

NODE_PRELOAD = Path(__file__).with_name("spawn_count.cjs")

_installed: list[str] = []


def _program(event: str, args: tuple) -> tuple[str, list[str]]:
    """The program a launch runs and the first few of its arguments."""
    if event == "os.system":
        command = str(args[0])
        return command.split()[0] if command.split() else command, [command[:200]]
    argv = args[1] if len(args) > 1 else None
    if isinstance(argv, (str, bytes)):
        argv = [argv]
    argv = [os.fsdecode(part) for part in (argv or [])]
    program = os.fsdecode(args[0]) if args and args[0] is not None else (argv[0] if argv else "?")
    return program, [part[:120] for part in argv[:6]]


def _callers(depth: int = 4) -> list[str]:
    """The innermost refactory frames that asked for the launch, as ``module:function:line``."""
    found: list[str] = []
    frame = sys._getframe(2)
    while frame is not None and len(found) < depth:
        name = frame.f_globals.get("__name__", "")
        if name.startswith(("refactory", "tests", "test_")) and name != __name__:
            found.append(f"{name}:{frame.f_code.co_name}:{frame.f_lineno}")
        frame = frame.f_back
    return found


def _write(path: str, line: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
    except OSError:
        pass


def install() -> None:
    """Start counting in this process, and in the Node processes it starts, when :data:`ENV` is set."""
    path = os.environ.get(ENV)
    if not path or _installed:
        return
    _installed.append(path)
    quoted = str(NODE_PRELOAD).replace("\\", "\\\\").replace('"', '\\"')
    preload = f'--require="{quoted}"'
    if preload not in os.environ.get("NODE_OPTIONS", ""):
        os.environ["NODE_OPTIONS"] = f"{os.environ.get('NODE_OPTIONS', '')} {preload}".strip()
    pid = os.getpid()
    _write(path, {"pid": pid, "kind": "start", "runtime": "python", "argv": sys.argv[:4],
                  "test": os.environ.get("PYTEST_CURRENT_TEST")})

    def hook(event: str, args: tuple) -> None:
        if event not in EVENTS:
            return
        try:
            program, argv = _program(event, args)
            _write(path, {"pid": os.getpid(), "kind": "launch", "runtime": "python",
                          "program": program, "argv": argv,
                          "callers": _callers(), "test": os.environ.get("PYTEST_CURRENT_TEST")})
        except Exception:  # noqa: BLE001 -- a measurement never fails the launch it measures
            pass

    sys.addaudithook(hook)


def summary(path: Path) -> dict:
    """What a log holds: launches by program, and process starts by runtime."""
    launches: Counter[str] = Counter()
    starts: Counter[str] = Counter()
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            line = json.loads(raw)
        except ValueError:
            continue
        if line.get("kind") == "start":
            starts[line.get("runtime", "?")] += 1
        elif line.get("kind") == "launch":
            launches[os.path.basename(str(line.get("program", "?")))] += 1
    return {"launches": sum(launches.values()), "by_program": dict(launches.most_common()),
            "starts": dict(starts)}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m refactory_dsh.spawn_count LOG", file=sys.stderr)
        return 2
    print(json.dumps(summary(Path(args[0])), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
