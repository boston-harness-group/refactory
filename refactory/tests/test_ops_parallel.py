"""Test ops for data_parallel tests. Run inside forkserver child processes."""
import json
import os
import time
from pathlib import Path


def barrier_op(ctx):
    """Each item signals ready, then waits until K items are ready."""
    item = ctx.read_json("current_item.json")
    item_id = str(item.get("id", "?"))
    barrier_dir = Path(os.environ["BARRIER_DIR"])
    k = int(os.environ.get("BARRIER_K", "3"))

    # Signal ready.
    (barrier_dir / f"ready_{item_id}").write_text(str(time.monotonic()))

    # Wait for all K items to be ready.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        ready = list(barrier_dir.glob("ready_*"))
        if len(ready) >= k:
            break
        time.sleep(0.02)
    else:
        raise TimeoutError(f"barrier timeout: only {len(list(barrier_dir.glob('ready_*')))} of {k} ready")

    (barrier_dir / f"done_{item_id}").write_text(str(time.monotonic()))
    return {"counted": item_id}


def sleep_op(ctx):
    """Sleep for `delay` seconds (from the item)."""
    item = ctx.read_json("current_item.json")
    delay = float(item.get("delay", 0))
    barrier_dir_str = os.environ.get("BARRIER_DIR", "")
    item_id = str(item.get("id", "?"))

    if barrier_dir_str:
        barrier_dir = Path(barrier_dir_str)
        barrier_dir.mkdir(parents=True, exist_ok=True)
        (barrier_dir / f"start_{item_id}").write_text(str(time.time()))

    time.sleep(delay)

    if barrier_dir_str:
        (barrier_dir / f"end_{item_id}").write_text(str(time.time()))

    return {"counted": item_id}


def noop_op(ctx):
    """No-op: just record that this item was visited."""
    item = ctx.read_json("current_item.json")
    item_id = str(item.get("id", "?"))
    visited_dir_str = os.environ.get("VISITED_DIR", "")
    if visited_dir_str:
        visited_dir = Path(visited_dir_str)
        visited_dir.mkdir(parents=True, exist_ok=True)
        (visited_dir / item_id).write_text("1")
    return {"counted": item_id}


def long_sleep_op(ctx):
    """Sleep for a very long time (for deadline testing)."""
    item = ctx.read_json("current_item.json")
    delay = float(item.get("delay", 0))
    time.sleep(delay)
    return {"counted": item.get("id")}


def maybe_fail_op(ctx):
    """Fail if the item's 'fail' field is truthy; succeed otherwise."""
    item = ctx.read_json("current_item.json")
    item_id = str(item.get("id", "?"))
    if item.get("fail"):
        raise RuntimeError(f"item {item_id} forced to fail")
    visited_dir_str = os.environ.get("VISITED_DIR", "")
    if visited_dir_str:
        visited_dir = Path(visited_dir_str)
        visited_dir.mkdir(parents=True, exist_ok=True)
        (visited_dir / item_id).write_text("1")
    return {"counted": item_id}
