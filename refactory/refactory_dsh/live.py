"""Durable activity shared by terminal and native GUI readers.

Readers never execute or restart work. Cursors count complete journal records;
an interrupted final write is not acknowledged until its boundary is repaired.

The journal is the only record of activity. `events.index.jsonl` and its
`events.index.provenance.json` anchor beside it are derived from it and can be
deleted at any time: they hold byte offsets and one timestamp per record, so a
reader can seek to a page and report the journal's size without parsing it. Every
indexed answer has to equal what a full scan of the journal answers, and a
disposable index never gets to change one.

The index has two layers because a page needs two different lookups. The overview
line (`.refactory/events.index.overview.json`) is the newest counted record's `i`
and `at`, so `total` and `last_event_at` cost one small file read. The lines of
`events.index.jsonl` answer "where does record `i` start", and only a page asks
that: they are read from the record's own line onwards, never from the file's
start, so a page costs what it serves and not what the history behind it holds.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from typing import Any

EVENTS = Path('.refactory/events.jsonl')

#: One line per complete journal record, in journal order, grouped by strides.
INDEX = Path('.refactory/events.index.jsonl')

#: Which journal the index covers, and how many index lines stand behind it.
PROVENANCE = Path('.refactory/events.index.provenance.json')

#: The newest counted record: `total` and `last_event_at` without touching the index.
OVERVIEW = Path('.refactory/events.index.overview.json')

#: Records one index stride carries, and the fixed width of a stride line, so the
#: stride covering a record can be found without parsing anything. A page bigger
#: than a stride reads whole strides, which is still `O(page_size)`.
STRIDE = 64

#: A stride line is `{"s":<first record>,"g":[<start>…],"l":[<last_at>…]}`, one
#: newline, with 9 digits per offset and 5 bytes (`"202"`) per timestamp, so every
#: line is the same width and stride `k` starts at byte `k * _STRIDE_LINE`.
_STRIDE_LINE = (len(b'{"s":') + 10 + len(b',"g":[') + STRIDE * 9 + (STRIDE - 1)
                + len(b'],"l":[') + STRIDE * 5 + (STRIDE - 1) + len(b']}\n'))

#: Records one activity page carries by default, and the most it may carry.
PAGE_DEFAULT, PAGE_MAX = 200, 1000

#: Distinguishes two temp files one writer publishes in the same process, so a
#: PID alone cannot collide and every temp is private to one publish.
_PUBLISH_SERIAL = count()

#: A gap record is what a malformed complete line becomes; it has no `at`.
_GAP_SUMMARY = 'An incomplete or malformed activity record could not be recovered.'


def read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def emit(root: Path, kind: str, **fields: Any) -> dict[str, Any]:
    root = Path(root).resolve()
    invocation = read(root / '.refactory/invocation.json')
    event = {**fields, 'schema_version': 1, 'run_id': str(root),
             'invocation': invocation.get('id'), 'at': datetime.now(timezone.utc).isoformat(), 'kind': kind}
    path = root / EVENTS
    path.parent.mkdir(parents=True, exist_ok=True)
    # Repair only the boundary after an interrupted write. The damaged bytes
    # remain as a visible journal gap; later events must still be readable.
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        size = os.fstat(fd).st_size
        separator = b'\n' if size and os.pread(fd, 1, size - 1) != b'\n' else b''
        os.write(fd, separator + (json.dumps(event, ensure_ascii=False) + '\n').encode())
    finally:
        os.close(fd)
    return event


def _record(root: Path, line: bytes) -> dict[str, Any]:
    """One complete journal line as a record; a malformed line becomes a gap.

    A gap keeps its slot: it was written, so a reader that did not see it would
    have a cursor that disagrees with the journal's own line count.
    """
    try:
        record = json.loads(line)
    except ValueError:
        record = None
    if not isinstance(record, dict):
        return {'schema_version': 1, 'run_id': str(Path(root).resolve()), 'kind': 'journal_gap',
                'state': 'unavailable', 'summary': _GAP_SUMMARY}
    return record


def _timestamp(record: dict[str, Any]) -> str | None:
    at = record.get('at')
    return at if isinstance(at, str) else None


@dataclass(frozen=True)
class _Coverage:
    """What the index stands behind: the newest record, and the bytes it read.

    `total` and `last_at` are the newest counted record's index and `at`, which is
    what today's `all_events[-1].get('at')` answers. `size` is the journal's byte
    size when the records were derived, so the suffix after it is all a later call
    has to parse. `anchor` is the first record's digest, kept so a replacement
    journal at the same path is caught even when the new one is already longer
    than the index's own end. `lines` is how many index lines stand behind this
    coverage — a count, not a position in the file, because a concurrent builder
    may already have appended the next stride.
    """
    total: int
    last_at: str | None
    size: int
    anchor: str
    first_end: int
    lines: int


@dataclass(frozen=True)
class _Line:
    """One record's place in the journal: where its line starts, and its own `at`."""
    start: int
    at: str | None


def _link(path: Path) -> tuple[int, int] | None:
    """A file's identity, or None where the filesystem has no stable inode."""
    try:
        status = path.stat()
    except FileNotFoundError:
        return None
    return (status.st_dev, status.st_ino) if status.st_ino else None


def _anchor(journal: Path, end: int) -> str | None:
    """The digest of the journal's first complete record, read in one bounded go."""
    try:
        with journal.open('rb') as handle:
            head = handle.read(end)
    except OSError:
        return None
    return hashlib.sha256(head).hexdigest() if len(head) == end else None


def _stride(rows: list[tuple[int, int]], ats: list[str | None]) -> bytes:
    """One stride's fixed-width index line.

    Every value is written at a fixed width, so the stride holding record `j` is
    arithmetic and the page's own line is a seek rather than a walk from record 0.
    A missing `at` is the three bytes `nul` in the same slot a timestamp's first
    three bytes take, which is all the index ever needs to say which record is
    newest.
    """
    starts = b','.join(f'{start:9d}'.encode() for _, start in rows)
    last = b','.join(b'"' + ('nul' if at is None else at[:3]).encode() + b'"' for at in ats)
    # A final short stride is padded out, so every line occupies the same bytes
    # and the line a later append adds starts where the seek expects it.
    return (b'{"s":' + f'{rows[0][0]:<10d}'.encode() + b',"g":[' + starts + b'],"l":[' + last
            + b']}').ljust(_STRIDE_LINE - 1, b' ') + b'\n'


def _unstride(line: bytes) -> list[_Line] | None:
    """A stride line's records, or None when the line is not one this version wrote.

    The line stands behind exactly the records it names, so a short tail stride is
    a valid last line and not a torn one; a line of the wrong width, one that does
    not parse, or one whose offsets do not advance is not trusted.
    """
    # The final stride carries only the records left, so its line is shorter, and
    # the anchor's `index_lines` is what says how many lines to expect.
    if not line.endswith(b'\n') or len(line) > _STRIDE_LINE:
        return None
    try:
        row = json.loads(line)
    except ValueError:
        return None
    if not isinstance(row, dict) or not isinstance(row.get('g'), list) or not isinstance(row.get('l'), list):
        return None
    starts, values = row['g'], row['l']
    if len(starts) != len(values) or not 0 < len(starts) <= STRIDE:
        return None
    if not all(isinstance(start, int) and start >= 0 for start in starts):
        return None
    if any(later <= earlier for earlier, later in zip(starts, starts[1:])):
        return None
    return [_Line(start, None if value.startswith('nul') else value.strip())
            for start, value in zip(starts, values)]


def _indexed(root: Path, journal: Path, size: int) -> _Coverage | None:
    """What provenance plus the overview line claim, or None when they may not.

    Only two small files are read here, never the index: the overview line holds
    `total` and `last_event_at`, and provenance says how many lines stand behind
    it, so a warm call answers both without walking the history.
    """
    try:
        provenance = json.loads((root / PROVENANCE).read_text())
        overview = json.loads((root / OVERVIEW).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(provenance, dict) or not isinstance(overview, dict):
        return None
    link = _link(journal)
    if (provenance.get('schema_version') != 1 or provenance.get('journal') != str(EVENTS)
            or provenance.get('run_id') != str(Path(root).resolve())):
        return None
    if link and (provenance.get('st_dev'), provenance.get('st_ino')) != link:
        return None
    total, last_at = overview.get('total'), overview.get('last_at')
    recorded, lines = provenance.get('size_at_write'), provenance.get('index_lines')
    first_end, anchor = provenance.get('first_record_end'), provenance.get('first_record_sha256')
    if not (isinstance(total, int) and total >= 0 and (last_at is None or isinstance(last_at, str))
            and isinstance(recorded, int) and isinstance(lines, int) and isinstance(first_end, int)
            and isinstance(anchor, str) and overview.get('index_lines') == lines):
        return None
    # `total` is only sound when every record it counts has the line that places
    # it, so the two counts have to agree — which is also what catches an index
    # that was cut short while its anchor still described a longer one.
    if lines != -(-total // STRIDE):
        return None
    # `recorded` is the bytes the index claims to cover; a journal truncated below
    # it is covered by nothing.
    if recorded > size or not 0 < first_end <= size:
        return None
    return _Coverage(total, last_at, recorded, anchor, first_end, lines)


def _trusted(root: Path, journal: Path, size: int) -> _Coverage | None:
    """The index, when it may claim to describe this journal.

    Every check here exists because a new invocation moves `events.jsonl` away
    and lets a different journal take its path (`invocations.prepare`). An index
    that survived that would answer the archived journal's offsets for the new
    one's records, so anything less than all the checks means "index nothing" and
    the next write rebuilds from record 0.
    """
    coverage = _indexed(root, journal, size)
    if coverage is None:
        return None
    # The index must hold exactly the lines the anchor claims: one short cannot
    # place a page, and a torn line past them is a failed write this reader cuts
    # away rather than leaving for a later append to sit behind.
    try:
        status = (root / INDEX).stat()
    except OSError:
        return None
    if status.st_size != coverage.lines * _STRIDE_LINE:
        return None
    # The first record is the only journal bytes a trusted prefix needs to read:
    # if they still match, the rest of the prefix is the bytes it was derived from.
    if _anchor(journal, coverage.first_end) != coverage.anchor:
        return None
    return coverage


def _rebuild(root: Path, journal: Path, size: int) -> _Coverage:
    """Index the whole journal once, write the derived files, and return them.

    This is the answer every reader falls back to, so it has to produce exactly
    what a full scan produces — a rebuild is never a way to answer differently.
    """
    rows: list[tuple[int, int]] = []
    ats: list[str | None] = []
    anchor = ''
    first_end = total = 0
    last_at: str | None = None
    lines: list[bytes] = []
    with journal.open('rb') as handle:
        while line := handle.readline():
            # An unterminated final line is a write still in flight; it is not
            # counted and not served until a later append completes it.
            if not line.endswith(b'\n'):
                break
            if not line.strip():
                continue
            rows.append((total, handle.tell() - len(line)))
            ats.append(_timestamp(_record(root, line)))
            # `ats` is emptied at every stride boundary, so the newest record's
            # `at` is kept here and not read back off the last stride written.
            last_at = ats[-1]
            anchor = anchor or hashlib.sha256(line).hexdigest()
            first_end = first_end or handle.tell()
            # The counted prefix ends at the last counted line, not at the file's
            # end: an unterminated tail is a write in flight now, but a later
            # append terminates those same bytes and makes them a counted
            # `journal_gap`. Recording the file's end would skip that gap, so the
            # suffix has to be re-read from before it.
            size = handle.tell()
            total += 1
            if len(rows) == STRIDE:
                lines.append(_stride(rows, ats))
                rows, ats = [], []
    if rows:
        lines.append(_stride(rows, ats))
    coverage = _Coverage(total, last_at, size, anchor, first_end, len(lines))
    _write(root, journal, coverage, lines)
    return coverage


def _write(root: Path, journal: Path, coverage: _Coverage, fresh: list[bytes]) -> None:
    """Publish the derived files; the journal itself is never written here.

    The index append is an in-place truncate-then-write through a descriptor opened
    without O_APPEND: lines this coverage did not produce (a torn tail, another
    journal's lines) are cut away, then the new complete lines are written at that
    offset. There is deliberately no O_APPEND here — an append cannot cut a torn
    tail. What keeps two racing builders consistent is that a record's byte offset
    in the journal never changes once written (DEC-001 forbids in-place journal
    rewrites), so any builder's coverage of the same journal prefix yields the same
    stride lines at the same offsets. The overview and provenance are each written
    to a temporary file and renamed, so a reader racing a builder sees a whole
    number of lines and an anchor that is either the previous version or this one.

    Each temporary is per-writer (PID + a per-process counter), so no two
    publishes share an inode: every concurrent reader is also a writer of these
    derived files, and a fixed temp name would let two builders truncate the same
    file mid-flight and publish a torn one. The reader re-parses and rejects a
    torn pair, so a collision is not silent corruption — it is a forced full
    O(N) rebuild, the whole scan this index exists to avoid. A run directory a
    reader may not write to is not an error: the index is an optimisation, so the
    answer already computed is still the answer.
    """
    index = root / INDEX
    link = _link(journal)
    anchor = {'schema_version': 1, 'journal': str(EVENTS), 'run_id': str(Path(root).resolve()),
              'st_dev': link[0] if link else None, 'st_ino': link[1] if link else None,
              'size_at_write': coverage.size, 'first_record_sha256': coverage.anchor,
              'first_record_end': coverage.first_end, 'index_lines': coverage.lines}
    overview = {'schema_version': 1, 'index_lines': coverage.lines,
                'total': coverage.total, 'last_at': coverage.last_at}
    try:
        kept = coverage.lines - len(fresh)
        try:
            fd = os.open(index, os.O_WRONLY | os.O_CREAT, 0o600)
            os.ftruncate(fd, kept * _STRIDE_LINE)
            os.lseek(fd, kept * _STRIDE_LINE, os.SEEK_SET)
            os.write(fd, b''.join(fresh))
        finally:
            os.close(fd)
        for name, value in ((OVERVIEW, overview), (PROVENANCE, anchor)):
            _publish(root, name, value)
    except OSError:
        # Best-effort by contract: the answer already computed stands, and the next
        # reader rebuilds. A lost temp-file race is benign (the winner publishes the
        # same content); a real error (permissions, disk full) leaves every later
        # reader on the slow path rather than wrong.
        return


def _publish(root: Path, name: Path, value: dict[str, Any]) -> None:
    """Write one derived JSON file atomically, through a temp private to this writer.

    The name carries the PID and a per-process serial, so no two publishes — in
    this process or another — ever share an inode, and nothing here truncates
    another writer's in-flight bytes. There is deliberately no ``O_EXCL``: a temp
    left by a crashed writer (or a recycled PID) would then make every later
    publish fail with ``FileExistsError``, wedging the index for the run forever.
    A unique name needs no exclusivity — the file is this writer's alone, and the
    ``finally`` removes it whatever happens, including a lost ``os.replace``.
    """
    payload = json.dumps(value, sort_keys=True) + '\n'
    temp = root / f'{name.name}.{os.getpid()}.{next(_PUBLISH_SERIAL)}.tmp'
    try:
        temp.write_text(payload)
        os.replace(temp, root / name)
    finally:
        temp.unlink(missing_ok=True)


def _extend(root: Path, journal: Path, size: int) -> _Coverage:
    """Index only the journal suffix a trusted index has not covered yet.

    A journal whose identity and size are unchanged costs no journal read at all:
    the overview line already answers both the total and the newest `at`.

    `size == coverage.size` is therefore tested before any read. That is also what
    keeps a poll cheap while a write is in flight: the walk below stops at a line
    with no newline and re-publishes the same coverage, so a journal that has
    trailing unterminated bytes leaves `coverage.size` at the counted prefix and
    never equals `size` — without the early return below, every poll tick would
    re-read that suffix and rewrite two identical derived files.
    """
    coverage = _trusted(root, journal, size)
    if coverage is None:
        return _rebuild(root, journal, size)
    if size == coverage.size:
        return coverage
    total, last_at = coverage.total, coverage.last_at
    counted_end = coverage.size
    measured = size          # the journal's own size, before `size` is the walk's cursor
    rows: list[tuple[int, int]] = []
    ats: list[str | None] = []
    fresh: list[bytes] = []
    with journal.open('rb') as handle:
        if handle.seek(0, os.SEEK_END) != measured:
            # The journal moved after it was measured; re-index what is there now.
            return _rebuild(root, journal, handle.tell())
        handle.seek(coverage.size)
        while line := handle.readline():
            # `f"` counts nothing here and is not acknowledged, so the counted
            # prefix stays where it is; a later append terminates it and it
            # becomes a counted `journal_gap` the next pass has to read.
            if not line.endswith(b'\n'):
                break
            if not line.strip():
                continue
            rows.append((total, handle.tell() - len(line)))
            ats.append(_timestamp(_record(root, line)))
            # The suffix can span several strides, so the newest `at` is kept as
            # the journal is walked; `ats` is emptied at every stride boundary.
            last_at = ats[-1]
            size = handle.tell()
            total += 1
            if len(rows) == STRIDE:
                fresh.append(_stride(rows, ats))
                rows, ats = [], []
    if rows:
        fresh.append(_stride(rows, ats))
    if size == counted_end and not fresh:
        # Nothing but unterminated trailing bytes since the last count: the walk
        # consumed no record, so this coverage *is* the answer. Returning it
        # without publishing is what stops a poll tick from rewriting identical
        # derived files for as long as the writer stays mid-record.
        return coverage
    # A suffix that held only blank lines counts nothing and keeps the old `at`.
    newest = last_at
    extended = _Coverage(total, newest, size, coverage.anchor, coverage.first_end,
                         coverage.lines + len(fresh))
    _write(root, journal, extended, fresh)
    return extended

def _lines(root: Path, coverage: _Coverage, first: int, want: int) -> list[_Line] | None:
    """The index lines that place records `first` … `first + want`, and no others.

    This is the seek that makes a page cost its page: the stride is arithmetic,
    and nothing before `first`'s stride is read or parsed. The stride's records
    before `first` are dropped here, so the caller starts at `after`'s own line
    and never re-serves a record it was told to skip.
    """
    if first >= coverage.total:
        return []
    stride = first // STRIDE
    last = min(coverage.lines, -(-(first + want) // STRIDE))
    out: list[_Line] = []
    try:
        with (root / INDEX).open('rb') as handle:
            handle.seek(stride * _STRIDE_LINE)
            for _ in range(last - stride):
                row = _unstride(handle.read(_STRIDE_LINE))
                if row is None:
                    return None
                out.extend(row)
    except OSError:
        return None
    # Anything short of the records it has to place cannot answer the page.
    return out[first % STRIDE:] if len(out) >= min(want, coverage.total - first) else None


def summary(root: Path) -> dict[str, Any]:
    """The journal's own count and newest `at`, with no records parsed.

    `last_event_at` is the newest counted record's `at` when it is a string and
    `None` when it is not — a `journal_gap`, an object without `at`, or no journal
    at all. That is exactly what `events()` has always answered for the same
    journal (`all_events[-1].get('at')`), and no reader may substitute an earlier
    record's timestamp for it.
    """
    root = Path(root).resolve()
    journal = root / EVENTS
    try:
        size = journal.stat().st_size
    except FileNotFoundError:
        return {'total': 0, 'last_event_at': None}
    coverage = _extend(root, journal, size)
    return {'total': coverage.total, 'last_event_at': coverage.last_at}


def page(root: Path, *, after: int = 0, limit: int = PAGE_DEFAULT) -> dict[str, Any]:
    """The page, the total, and the newest record's `at`, in one pass.

    `total` and `last_event_at` come from the overview line, and the page seeks to
    the stride that holds `after`, so a reader that polls pays for its page and
    the suffix appended since — never for the history behind it.

    `after` past the journal's end is not caller error: it answers an empty page,
    and the cursor is clamped back to `total` so a returned cursor never points
    beyond the records that exist (R-1.3 leaves this open; a caller trusting
    `cursor <= total` would otherwise be surprised by 5000 on a 1,100-record
    journal). `has_more` stays `false` either way.
    """
    if after < 0:
        raise ValueError('event cursor must be nonnegative')
    root = Path(root).resolve()
    journal = root / EVENTS
    try:
        size = journal.stat().st_size
    except FileNotFoundError:
        return {'events': [], 'after': after, 'cursor': 0, 'total': 0, 'page_size': limit,
                'has_more': False, 'last_event_at': None}
    coverage = _extend(root, journal, size)
    cursor_end = min(after, coverage.total)
    events: list[dict[str, Any]] = []
    if after < coverage.total:
        # A missing or unreadable index line is answered by the journal, never by
        # a guess: the rebuild writes the index again for the next reader.
        #
        # Expected performance cliff, bounded by the write window: while a writer
        # is mid-stride the index can be torn between stride boundaries, so
        # `_lines` rejects a short read and this rebuilds from the journal — a
        # whole-journal scan. It happens only for a page whose stride is the one
        # being appended to, only until that stride completes, and only while a
        # writer holds the journal open mid-record; the next poll sees a whole
        # stride and seeks again. It is self-healing by design (the index is
        # derived and never authoritative, R-6.5), so a reader polling a live
        # journal pays one full scan per stride boundary it happens to land on,
        # not one per poll.
        lines = _lines(root, coverage, after, limit)
        if lines is None:
            coverage = _rebuild(root, journal, size)
            lines = _lines(root, coverage, after, limit) or []
        if not lines:      # a journal that moved while it was read serves no page
            return {'events': [], 'after': after, 'cursor': cursor_end, 'total': coverage.total,
                    'page_size': limit, 'has_more': after < coverage.total,
                    'last_event_at': coverage.last_at}
        with journal.open('rb') as handle:
            handle.seek(lines[0].start)
            while len(events) < limit and (line := handle.readline()):
                # A trailing line without its newline is a write in flight, and
                # the index has no line for it, so the page ends before it.
                if line.endswith(b'\n') and line.strip():
                    events.append(_record(root, line))
    cursor = min(cursor_end + len(events), coverage.total)
    return {'events': events, 'after': after, 'cursor': cursor, 'total': coverage.total,
            'page_size': limit, 'has_more': cursor < coverage.total,
            'last_event_at': coverage.last_at}


def events(root: Path, after: int = 0) -> tuple[list[dict[str, Any]], int]:
    """Every record from ``after`` on, and how many the journal holds.

    This is the full scan every indexed answer has to agree with. A trailing line
    with no newline is a write still in flight, so it is neither served nor
    counted; a malformed *complete* line keeps its slot as a `journal_gap`.
    """
    if after < 0:
        raise ValueError('event cursor must be nonnegative')
    try:
        lines = (root / EVENTS).read_bytes().splitlines(keepends=True)
    except FileNotFoundError:
        return [], 0
    records = []
    for line in lines:
        if not line.endswith(b'\n'):
            break
        if not line.strip():
            continue
        records.append(_record(root, line))
    return records[after:], len(records)


def events_page(root: Path, *, after: int = 0, limit: int = PAGE_DEFAULT) -> tuple[list[dict[str, Any]], int]:
    """Records ``[after:after+limit]`` and the journal's complete-record count."""
    result = page(root, after=after, limit=limit)
    return result['events'], result['total']


def overview(root: Path) -> dict[str, Any]:
    """The journal's summary facts: the count, and the newest record's `at`."""
    result = summary(root)
    return {'total': result['total'], 'last_event_at': result['last_event_at']}


def graph_snapshot(root: Path) -> dict[str, Any] | None:
    """Read topology from this run's checkpoint, without exposing commands/prompts.

    Checkpoints may be absent or from an older runtime. Never infer connections
    from event order or import/execute a mode to reconstruct missing topology.
    """
    checkpoint = read(root / '.refactory/spine-checkpoint.json')
    graph = checkpoint.get('graph')
    if not isinstance(graph, dict) or not isinstance(graph.get('nodes'), dict):
        return None
    nodes = [{'id': key, 'kind': str(node.get('_type', 'Node'))}
             for key, node in graph['nodes'].items() if isinstance(node, dict)]
    ids = {node['id'] for node in nodes}
    edges = []
    for edge in graph.get('edges', []) if isinstance(graph.get('edges'), list) else []:
        if not isinstance(edge, dict):
            continue
        source, target = edge.get('source'), edge.get('target')
        if not isinstance(source, str) or not isinstance(target, str) or source not in ids or target not in ids:
            continue
        edges.append({'source': source, 'target': target,
                      'label': str(edge.get('condition') or '')})
    # The spine declares these connections on nodes, rather than in `edges`.
    def connect(source: Any, target: Any, label: str) -> None:
        if isinstance(source, str) and isinstance(target, str) and source in ids and target in ids:
            if not any(edge['source'] == source and edge['target'] == target for edge in edges):
                edges.append({'source': source, 'target': target, 'label': label})
    for key, node in graph['nodes'].items():
        if not isinstance(node, dict):
            continue
        if node.get('_type') == 'ForkNode' and isinstance(node.get('targets'), list):
            for target in node['targets']:
                connect(key, target, 'branch')
        if node.get('_type') == 'JoinNode' and isinstance(node.get('sources'), list):
            for source in node['sources']:
                connect(source, key, 'join')
        if node.get('_type') == 'DataNode':
            connect(key, node.get('subgraph_entry'), 'each item')
            connect(node.get('subgraph_exit'), key, 'next item')
    successors = checkpoint.get('successors')
    if isinstance(successors, dict):
        for source, target in successors.items():
            connect(source, target, '')
    skipped = checkpoint.get('skipped', [])
    return {'nodes': nodes, 'edges': edges,
            'skipped': [node for node in skipped if isinstance(node, str) and node in ids]
            if isinstance(skipped, list) else []} if nodes else None


def _usage(usage: Any) -> Any:
    """A receipt's usage with its call log replaced by how many calls it holds.

    The log repeats every earlier call in each new record, so a worker that makes
    a thousand requests used to make the journal quadratic. The count is the fact
    a reader wants; a receipt that already carries a count passes through.
    """
    if not isinstance(usage, dict) or not isinstance(usage.get('calls'), list):
        return usage
    return {**usage, 'calls': len(usage['calls'])}


def _receipt(worker: dict[str, Any]) -> dict[str, Any]:
    return {**worker, 'usage': _usage(worker.get('usage'))}


def snapshot(root: Path, *, after: int = 0, limit: int = PAGE_DEFAULT) -> dict[str, Any]:
    root = Path(root).resolve()
    spine = read(root / '.refactory/spine-status.json')
    invocation = read(root / '.refactory/invocation.json')
    # One pass over the journal for the page and for both summary facts; two
    # passes are what made every status call read the whole history twice.
    history = page(root, after=after, limit=limit)
    workers = [_receipt(read(p)) for p in sorted((root / '.refactory/workers').glob('*.json'))]
    deliveries = [read(p) for p in sorted((root / '.refactory/agent-delivery').glob('*/receipt.json'))]
    # A delivery context does not prove the coordinator actually dispatched it.
    supplied = [{'worker_id': row.get('id'), 'node_id': row.get('node'),
                 'state': row.get('status'), 'started': row.get('started'),
                 'context_revision': len(row.get('contexts', [])),
                 'receipt': str(Path('.refactory/agent-delivery') / str(row.get('id')) / 'receipt.json')}
                for row in deliveries if row]
    nodes = spine.get('nodes', [])
    completed = spine.get('completed', [])
    wait = spine.get('wait_reason')
    if not wait:
        wait = ('stopped at graph boundary' if spine.get('status') == 'halted' else
                'paused by user' if spine.get('status') == 'paused' else
                'failure requires attention' if spine.get('status') in ('failed', 'refused') else
                'graph boundary; worker dispatch unconfirmed' if nodes else None)
    # dm-95: override wait_reason when a node failure is awaiting retry
    last_failure = spine.get('last_failure')
    if last_failure and isinstance(last_failure, dict) and spine.get('status') == 'ready':
        wait = f'step {last_failure.get("node", "unknown")} failed; awaiting retry'
    return {'schema_version': 1, 'run_id': str(root), 'invocation': invocation.get('id'),
            'events': history['events'], 'cursor': history['cursor'], 'after': history['after'],
            'total': history['total'], 'page_size': history['page_size'], 'has_more': history['has_more'],
            'last_event_at': history['last_event_at'],
            'started_at': invocation.get('started'), 'nodes': nodes, 'completed': completed,
            'running_nodes': spine.get('running_nodes'),
            'graph': graph_snapshot(root),
            'wait_reason': wait,
            **({'last_failure': last_failure} if last_failure and isinstance(last_failure, dict) else {}),
            'workers': workers, 'deliveries': supplied,
            'resources': read(root / '.refactory/resource-usage.json') or None,
            'control': read(root / '.refactory/graph-control-receipt.json') or None,
            'outputs': read(root / 'facts.json')}
