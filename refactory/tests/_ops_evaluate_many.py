"""Minimal test op for item-gate tests.

Provides a process-safe in-flight counter via fcntl-locked shared files.
Every ``sleep_op`` call increments the counter on entry, records the peak,
sleeps, and decrements in ``finally``.  The peak is written to ``$PEAK_FILE``.

The counter file (``$COUNTER_FILE``) stores "current peak" and is protected
by fcntl.flock so it works across forkserver child processes.
"""
import fcntl
import os
import time
from pathlib import Path


def _update_counter(counter_file, delta):
    """Atomically update the shared counter; return (new_current, new_peak)."""
    with open(counter_file, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            parts = f.read().strip().split()
            current = int(parts[0]) if len(parts) > 0 else 0
            peak = int(parts[1]) if len(parts) > 1 else 0
            current += delta
            peak = max(peak, current)
            f.seek(0)
            f.truncate()
            f.write(f"{current} {peak}")
            f.flush()
            return current, peak
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def sleep_op(ctx):
    """Sleep for `delay` seconds, tracking in-flight count via counter."""
    item = ctx.read_json("current_item.json")
    delay = float(item.get("delay", 0))
    item_id = str(item.get("id", "?"))

    counter_file = os.environ.get("COUNTER_FILE", "")
    if counter_file:
        _update_counter(counter_file, +1)
    try:
        time.sleep(delay)
    finally:
        if counter_file:
            _, peak = _update_counter(counter_file, -1)
            peak_file = os.environ.get("PEAK_FILE", "")
            if peak_file:
                Path(peak_file).write_text(str(peak))

    return {"counted": item_id}
