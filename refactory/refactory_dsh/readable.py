"""Render the console's JSON payloads as readable text for terminal users.

One rendering, not two: these functions produce the ``text`` field the console
adds to every status, steer and graph-control payload, and both the CLI and
tool-refactory print that text instead of keeping a second formatter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def format_status(payload: dict[str, Any]) -> str:
    """Readable status block for a design or optimizer run."""
    lines: list[str] = []
    authoring = payload.get('authoring') if isinstance(payload.get('authoring'), dict) else None

    if authoring is not None:
        name = authoring.get('name') or payload.get('mode') or ''
        lines.append(f"Run {payload.get('run_id', '')}  {name}")

        # Position: running nodes with kinds, or wait reason, or idle.
        activity = payload.get('activity') or {}
        running_nodes = activity.get('running_nodes') or []
        state = payload.get('state', '')
        if state == 'waiting_user' and activity.get('wait_reason'):
            lines.append(f"Position: {activity['wait_reason']}")
        elif running_nodes:
            parts = []
            for entry in running_nodes:
                if isinstance(entry, dict):
                    node_id = entry.get('id', '')
                    kind = entry.get('kind', '')
                    parts.append(f"{node_id} ({kind})" if kind else str(node_id))
                else:
                    parts.append(str(entry))
            lines.append(f"Position: {', '.join(parts)}")
        else:
            lines.append("Position: idle")

        completed = authoring.get('completed') or []
        lines.append(f"Steps done: {len(completed)}")

        step_tokens = authoring.get('step_tokens')
        if isinstance(step_tokens, dict) and step_tokens:
            entries = [f"{k}: {v:,}" for k, v in step_tokens.items()
                       if isinstance(v, (int, float))]
            if entries:
                lines.append(f"Tokens per step: {', '.join(entries)}")

        total_tokens = authoring.get('total_tokens')
        if isinstance(total_tokens, (int, float)):
            lines.append(f"Total tokens: {total_tokens:,}")

        owner_alive = payload.get('owner_alive')
        if owner_alive is True:
            lines.append("Owner: alive")
        elif owner_alive is False:
            lines.append("Owner: not alive")

        lines.append(f"State: {state}")

        retried = activity.get('node_retries')
        if isinstance(retried, dict) and retried:
            lines.append("Retried after a timeout: " + ", ".join(
                f"{node} ({count}x)" for node, count in sorted(retried.items())))

        error = authoring.get('error')
        if error:
            lines.append(f"Error: {error}")

        control = (activity.get('control') or {}) if isinstance(activity.get('control'), dict) else {}
        # dm-101: A stale failed control is cleared only when a later control on
        # the same run succeeded (the receipt file is overwritten with status
        # 'applied').  Both the "Last failed control" line and the matching
        # run_warning are suppressed once a later success overwrites the receipt.
        control_succeeded = control.get('status') == 'applied'
        if control.get('status') == 'failed':
            action = control.get('action', '')
            message = control.get('message', '')
            lines.append(f"Last failed control: {action} — {message}")

        warnings = [
            e.get('summary', '') for e in (activity.get('events') or [])
            if isinstance(e, dict) and e.get('kind') == 'run_warning' and e.get('summary')
            and not (control_succeeded and e.get('summary', '').startswith('graph control'))
        ]
        for w in dict.fromkeys(warnings):
            lines.append(f"Warning: {w}")

        drs = activity.get('design_review_status')
        if isinstance(drs, str) and drs.strip():
            lines.append("")
            for drs_line in drs.splitlines():
                lines.append(drs_line)

        if state in ('waiting_user', 'paused', 'died'):
            root = payload.get('root', payload.get('run_id', ''))
            if state == 'waiting_user':
                gate = next(
                    (n for n in running_nodes
                     if isinstance(n, dict) and n.get('kind') == 'gate'),
                    None,
                )
                if gate:
                    gate_id = gate.get('id', '')
                    lines.append(f"Next: refactory steer --run {root} --verdict {gate_id}=<verdict>")
                    lines.append(f"      refactory graph-control resume --run {root}")
                else:
                    lines.append(f"Next: refactory graph-control resume --run {root}")
            else:
                lines.append(f"Next: refactory graph-control resume --run {root}")
    else:
        # Optimizer run.
        lines.append(f"Run {payload.get('run_id', '')}  {payload.get('mode', '')}")
        best = payload.get('best_score')
        score_text = f"best score {best}" if isinstance(best, (int, float)) else "no score yet"
        lines.append(f"State: {payload.get('state', '')}; {score_text}.")
        counters = payload.get('counters') or {}
        epoch = counters.get('epoch', 0)
        generation = counters.get('generation', 0)
        evaluations = counters.get('evaluations', 0)
        lines.append(f"Epoch {epoch}, generation {generation}, {evaluations} evaluation(s).")

    return '\n'.join(lines)


def format_steer(payload: dict[str, Any]) -> str:
    """Readable steer output showing only what this call recorded."""
    lines: list[str] = []
    recorded_verdicts = payload.get('recorded_verdicts') or []
    recorded_note = payload.get('recorded_note')
    has_verdict = bool(recorded_verdicts)
    has_note = recorded_note is not None

    if has_verdict:
        scopes = payload.get('recorded_scopes') or {}
        for pair in recorded_verdicts:
            scope = scopes.get(pair, '')
            if scope:
                lines.append(f"Recorded verdict {pair}, scoped to {scope}.")
            else:
                lines.append(f"Recorded verdict {pair}.")
        root = payload.get('run', '')
        # Read the resume command from run-config.json when available.
        resume_cmd = f"refactory graph-control resume --run {root}"
        config_path = Path(root) / '.refactory/run-config.json' if root else None
        if config_path and config_path.exists():
            try:
                import json as _json
                config = _json.loads(config_path.read_text())
                resume_cmd = config.get('resume_command', resume_cmd)
            except (OSError, ValueError):
                pass
        lines.append("Continue the run:")
        lines.append(f"  {resume_cmd}")

    if has_note:
        kind = payload.get('recorded_kind', 'info')
        lines.append(f"Note recorded ({kind}).")
        late = payload.get('late_correction')
        if isinstance(late, dict) and late.get('message'):
            lines.append(str(late['message']))

    if not has_verdict and not has_note:
        entries = payload.get('entries') or []
        count = len(entries)
        word = 'entry' if count == 1 else 'entries'
        lines.append(f"Ledger: {count} {word}.")

    return '\n'.join(lines)


def format_graph_control(payload: dict[str, Any]) -> str:
    """Readable graph-control output: what was queued or applied, and why if it failed."""
    lines: list[str] = []
    status = payload.get('status', '')
    action = payload.get('action', '')
    run_id = payload.get('run_id', '')

    if status == 'queued':
        lines.append(f"Queued {action} for {run_id}.")
        if payload.get('resume_pending'):
            lines.append(
                'Resume is pending; a new owner will start when the current one exits.'
            )
    elif status == 'preview':
        nodes = payload.get('invalidated') or []
        count = len(nodes)
        names = ', '.join(str(n) for n in nodes)
        lines.append(f"Preview ({run_id}): {count} node(s) would be invalidated: {names}.")
    elif status == 'deferred':
        nodes = payload.get('invalidated') or []
        count = len(nodes)
        names = ', '.join(str(n) for n in nodes)
        lines.append(f"Preview (deferred, {run_id}): {count} node(s) would be invalidated: {names}.")
    elif status == 'failed':
        message = payload.get('message', '')
        lines.append(f"{action} failed ({run_id}): {message}.")

    last_failed = payload.get('last_failed_receipt')
    if isinstance(last_failed, dict):
        fa = last_failed.get('action', '')
        fm = last_failed.get('message', '')
        lines.append(f"Last failed control ({run_id}): {fa} — {fm}.")

    if status == 'acknowledged':
        message = payload.get('message', '')
        files = payload.get('files', [])
        lines.append(f"Acknowledged ({run_id}): {message}")
        if files:
            lines.append(f"Set aside {len(files)} file(s):")
            for f in files:
                lines.append(f"  {f.get('original_path', '?')}")
    elif status == 'withdrawn':
        control_id = payload.get('control_id', '')
        lines.append(f"Withdrawn control {control_id} ({run_id}).")
    elif action == 'amend-command':
        field = payload.get('field', '')
        old_val = payload.get('old_value', '')
        new_val = payload.get('new_value', '')
        message = payload.get('message', '')
        lines.append(message or f"Corrected {field}: {old_val!r} \u2192 {new_val!r}.")
    elif status == 'signalled':
        lines.append(payload.get('message', f"Sent SIGTERM to lingering owner."))

    # Name an owner this call started -- a resume of a run whose owner is gone, or a
    # rerun of a failed run -- so it reads apart from a control queued for a live one.
    owner = payload['owner'] if isinstance(payload.get('owner'), dict) else payload
    if isinstance(owner.get('pid'), int):
        lines.append(f"Started a new owner for {run_id} (pid {owner['pid']}).")

    return '\n'.join(lines)
