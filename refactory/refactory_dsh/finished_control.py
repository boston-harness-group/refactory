"""Finish controls without inventing another optimizer generation.

Legacy queued selections are replayed through the loop's existing selection
owner before adoption. This recovers a person's choice, not a new measurement.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def recover_selection(root: Path, *, project: Path | None = None) -> dict[str, Any] | None:
    """Apply a legacy finished-run Keep/Reject through the existing journal owner.

    No init, strategy, evaluation, proposal, or holdout call occurs. A changed
    winner must still pass adoption's ordinary current-winner audit guard.
    """
    from refactory_dsh.control import Control, ControlError, TARGETED, search_finished
    from refactory_dsh.loop import Loop, LoopConfig

    root = Path(root)
    control = Control(root)
    request = control.read_request()
    if not search_finished(root) or not (root / 'report.json').exists() or request is None or request['action'] not in TARGETED:
        return None
    saved = _read(root / 'config.json')
    mode = str(saved.get('mode') or _read(root / 'report.json').get('mode') or '')
    config = LoopConfig.from_dict(saved)
    config.runs_root = root
    loop = Loop(mode, config, project=project)
    loop._replay(root / 'journal.jsonl')
    target = str(request.get('target') or '')
    if target not in {candidate.id for candidate in loop.population}:
        raise ControlError(f"legacy pending {request['action']} names unknown candidate {target!r}; "
                           "the request was preserved; inspect the run's leaderboard and control.json")
    outcome = loop._honor({**request, 'recovered': True})
    # Keep the completed report's selection view current without rerunning its
    # measurements. Its old audit is deliberately retained and may now mismatch.
    report = _read(root / 'report.json')
    board = loop.leaderboard()
    winner = board[0] if board else None
    audit = report.get('holdout') or {}
    audited = winner is not None and audit.get('winner') == winner.id and audit.get('status') in ('measured', 'not_applicable')
    report.update(kept=loop.kept, rejected=loop.rejected,
                  leaderboard=[loop._row(candidate, detail=True) for candidate in board],
                  best=loop._row(winner, detail=True) if winner else None,
                  **(loop.best_score_fields(winner) if winner else {"best_score": None}),
                  raw_best_score=winner.raw_score if winner else None,
                  lineage=loop._lineage(winner),
                  structural_wins=loop._structural_wins(board),
                  exit_ok=audited,
                  error=None if audited else 'Recovered selection has no matching successful winner audit; the steering tutorial explains re-auditing without new proposals.')
    report['selection_recovery'] = outcome
    _write(root / 'report.json', report)
    rendered = loop._render_report(report)
    if not audited:
        rendered = '> Selection changed after the search. The recorded held-out audit below is historical and does not certify the current winner; re-audit before adoption.\n\n' + rendered
    (root / 'report.md').write_text(rendered)
    if control.read_request() == request:
        control.clear()
    return outcome


def _record(root: Path, receipt: dict[str, Any]) -> None:
    with (root / 'journal.jsonl').open('a') as handle:
        handle.write(json.dumps({'event': 'control', **receipt}) + '\n')
    for name in ('state.json', 'report.json'):
        path = root / name
        value = _read(path)
        value['adoptions'] = [*(value.get('adoptions') or []), receipt]
        _write(path, value)


def adopt_finished(root: Path, request: dict[str, Any], *, project: Path | None = None) -> dict[str, Any]:
    """Execute an explicit finished-run adoption now and retain its outcome."""
    from refactory_dsh.adopt import adopt, _journal_entries
    from refactory_dsh.control import Control, ControlError, request_identity

    control = Control(root)
    pending = control.read_request()
    mode = next((str(row['mode']) for row in _journal_entries(root)
                 if row.get('event') == 'start' and row.get('mode')), '')
    name = str(request.get('as') or (pending or {}).get('as') or f"{mode.split('/', 1)[0]}_adopted")
    if pending and pending['action'] == 'adopt' and pending.get('as') and pending['as'] != name:
        raise ControlError(f"a legacy adoption for {pending['as']!r} is pending; repeat adoption with that name to resolve it first")
    receipt = {**request, 'as': name, 'at': round(time.time(), 3), 'finished': True}
    receipt['request'] = request_identity(pending if pending and pending['action'] == 'adopt' else request)
    try:
        recovered = recover_selection(root, project=project)
        if recovered is not None:
            receipt['recovered_control'] = recovered
        written = adopt(root, as_name=name, project=project)
    except Exception as error:
        receipt.update(status='refused', error=f'{type(error).__name__}: {error}')
        _record(root, receipt)
        raise ControlError(receipt['error']) from error
    receipt.update(written, status='applied')
    _record(root, receipt)
    # Never erase a request another caller wrote while adoption was in progress.
    remaining = control.read_request()
    if remaining is not None and remaining == pending:
        control.clear()
    return receipt


def latest_receipt(root: Path) -> dict[str, Any] | None:
    """Expose a queued request or the last applied/refused control on reads."""
    from refactory_dsh.control import Control, request_identity, search_finished

    pending = Control(root).read_request()
    last = None
    journal = root / 'journal.jsonl'
    if journal.exists():
        for line in journal.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get('event') == 'control':
                last = {key: value for key, value in row.items() if key != 'event'}
                last.setdefault('status', 'refused' if last.get('error') else 'applied')
    if pending is not None:
        finished = search_finished(root)
        if finished and last is not None and last.get('request') == request_identity(pending) and last.get('status') == 'refused':
            return last
        return {**pending, 'status': 'queued', 'finished': finished,
                'message': 'Legacy request awaits recovery by adoption; no optimizer is running.' if finished
                else 'Waiting for the next generation boundary.'}
    return last
