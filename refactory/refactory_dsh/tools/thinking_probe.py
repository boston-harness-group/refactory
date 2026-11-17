"""Probe what a long silent-thinking session shows as activity, with a real model.

A node's model session ends on inactivity only (`model.DEFAULT_INACTIVITY_S`): the
client watches the worker receipt and the session's activity file (`model.activity_path`),
which the worker observer touches on every streamed frame. This runs one no-tools
completion whose prompt forces long thinking (no test items) and logs, with timestamps:

* ``cli``: a bare ``claude -p --output-format stream-json --include-partial-messages``
  session -- every stream event, so a reader sees whether thinking streams deltas or
  stays silent until its block ends;
* ``client``: the same prompt through refactory's own node session
  (`DshModelClient.for_node(...).with_tool_policy("none")`, the route given) -- every
  write of the receipt and of the activity file, i.e. each inactivity-clock reset.

Each arm ends with a summary: the longest gap between events (``cli``) or between clock
resets (``client``). A gap near the inactivity limit means a working session would be
stopped. It makes real model calls: run it where the run's model access is (the lab),
not on a laptop login::

    python -m refactory_dsh.tools.thinking_probe --log probe.jsonl \\
        --model claude-opus-4-6 --backend claude_code --effort high
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

#: A question with a short answer that takes long, careful reasoning to reach.
PROMPT = (
    "Think very carefully and at length before answering; check every step twice.\n\n"
    "Count the lattice points (x, y, z) with integer coordinates and x^2 + y^2 + z^2 <= 400 "
    "for which x + y + z is divisible by 7 and x*y*z is not divisible by 3. Do not write "
    "code; reason it out. Answer with the number only, inside <answer></answer>."
)


class Log:
    """Timestamped JSONL records, also echoed to stderr."""

    def __init__(self, path: Path):
        self.path, self.started = path, time.monotonic()
        self.file = path.open("a", encoding="utf-8")
        self.lock = threading.Lock()

    def write(self, arm: str, event: str, **detail: Any) -> float:
        elapsed = round(time.monotonic() - self.started, 3)
        row = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "elapsed_s": elapsed, "arm": arm,
               "event": event, **detail}
        with self.lock:
            self.file.write(json.dumps(row) + "\n")
            self.file.flush()
            print(json.dumps(row)[:300], file=sys.stderr)
        return elapsed


def _gaps(times: list[float]) -> float:
    return max((later - earlier for earlier, later in zip(times, times[1:])), default=0.0)


def probe_cli(log: Log, args: argparse.Namespace) -> None:
    """A bare Claude CLI session: every stream event, timestamped."""
    cmd = [args.claude, "-p", "--output-format", "stream-json", "--include-partial-messages",
           "--verbose", "--tools", ""]
    if args.model:
        cmd += ["--model", args.model]
    env = dict(os.environ)
    if args.max_thinking_tokens:
        env["MAX_THINKING_TOKENS"] = str(args.max_thinking_tokens)
    start = log.write("cli", "start", cmd=cmd)
    times, kinds = [start], {}
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(PROMPT)
    proc.stdin.close()
    for line in proc.stdout:
        try:
            message = json.loads(line)
        except ValueError:
            continue
        kind = message.get("type", "?")
        inner = message.get("event") or {}
        if kind == "stream_event":
            kind = f"stream_event:{inner.get('type')}"
            delta = (inner.get("delta") or {}).get("type")
            block = (inner.get("content_block") or {}).get("type")
            kind += f":{delta or block}" if (delta or block) else ""
        elif kind == "system":
            kind = f"system:{message.get('subtype')}"
        kinds[kind] = kinds.get(kind, 0) + 1
        times.append(log.write("cli", kind))
    proc.wait()
    end = log.write("cli", "exit", returncode=proc.returncode)
    times.append(end)
    log.write("cli", "summary", events=kinds, total_s=round(end - start, 1),
              longest_gap_s=round(_gaps(times), 1),
              thinking_deltas=sum(v for k, v in kinds.items() if "thinking_delta" in k))


def probe_client(log: Log, args: argparse.Namespace) -> None:
    """Refactory's own node session: every inactivity-clock reset, timestamped."""
    from refactory_dsh.model import DshModelClient, activity_path

    route: dict[str, Any] = {"worker_backend": args.backend, "model": args.model or "",
                             "effort": args.effort or "", "attempts": 1}
    if args.backend == "dsh":
        route["provider"] = args.provider
    # Inactivity off for the probe: it measures the gaps, it does not act on them.
    client = DshModelClient(**route, inactivity_s=0).for_node("probe", role="solver")
    client = client.with_tool_policy("none")
    run = Path(tempfile.mkdtemp(prefix="thinking-probe-"))
    stop = threading.Event()
    resets: list[float] = []

    def watch() -> None:
        seen: dict[Path, float] = {}
        while not stop.is_set():
            for receipt in (run / ".refactory" / "workers").glob("*.json"):
                for path in (receipt, activity_path(receipt)):
                    try:
                        mtime = path.stat().st_mtime
                    except OSError:
                        continue
                    if seen.get(path) != mtime:
                        seen[path] = mtime
                        resets.append(log.write("client", "clock_reset", file=path.name,
                                                mtime=mtime))
            time.sleep(0.25)

    start = log.write("client", "start", route=route, run=str(run))
    resets.append(start)
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    error = ""
    try:
        answer = client.complete(PROMPT, workdir=run)
        log.write("client", "answer", text=answer[-500:])
    except Exception as exc:  # noqa: BLE001 - the probe records any failure
        error = f"{type(exc).__name__}: {exc}"
        log.write("client", "error", error=error[:2000])
    stop.set()
    watcher.join()
    end = log.write("client", "end")
    resets.append(end)
    log.write("client", "summary", total_s=round(end - start, 1), resets=len(resets) - 2,
              longest_gap_s=round(_gaps(sorted(resets)), 1), error=error[:300])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--log", type=Path, required=True, help="JSONL file the events go to")
    parser.add_argument("--arm", choices=("cli", "client", "both"), default="both")
    parser.add_argument("--claude", default=os.environ.get("FACTORY_CLAUDE_BIN", "claude"),
                        help="the Claude CLI for the cli arm")
    parser.add_argument("--backend", default="claude_code", choices=("claude_code", "dsh"),
                        help="the client arm's worker backend")
    parser.add_argument("--provider", default="", help="the client arm's provider (dsh)")
    parser.add_argument("--model", default="")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--max-thinking-tokens", type=int, default=32000,
                        help="MAX_THINKING_TOKENS for the cli arm")
    args = parser.parse_args(argv)
    log = Log(args.log)
    if args.arm in ("cli", "both"):
        probe_cli(log, args)
    if args.arm in ("client", "both"):
        probe_client(log, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
