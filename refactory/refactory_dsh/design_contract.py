"""The design-document contract: what design.md must contain.

Defined once; read by design_doc, gate_design_review, delivery and the audit.
No split_clauses: the overwatch audit enumerates each constraint's clauses
itself, and audit_problems checks them.

Site markers are accepted in every unambiguous form: bare ``[changed]``,
bold ``**[changed]**``, backtick-wrapped `` `[changed]` ``, or with a
colon-description ``[changed: description]``.  The canonical form shown in
prompts and docstrings is ``[changed]`` / ``[left: <reason>]`` (bare, no
backticks).  ``[new]`` is treated as equivalent to ``changed``.

The three consumers are:
1. PR body "Changes" — ``delivery.py`` ``describe_changes()``, from the ``## Changes``
   bullets and the ``[changed]`` sites (``build_what_changed()``)
2. Design review gate marker check — ``design_ops.py`` ``gate_design_review()``
   via ``unmarked_sites()``
3. Run warnings — ``delivery.py`` ``target_guard.warn()`` on mismatches
"""
from __future__ import annotations

import dataclasses
import re


TITLE = "Title"
SUMMARY = "Summary"
FAILURE_TRACE = "Failure trace"
ASK_MAPPING = "Ask mapping"
RULE_AND_SITES = "Rule and sites"
DOCS = "Docs"
CHANGES = "Changes"
TESTS = "Tests"
GUARDEDNESS_EXEMPTIONS = "Test guardedness exemptions"
SITE_REVERT_EXEMPTIONS = "Site revert exemptions"
ANTI_PATTERNS = "Anti-patterns to avoid"
#: Added by apply_corrections when the person corrected the design at the approval gate.
CORRECTIONS = "Corrections at approval"
#: What design_doc must write (the light template). Every other section is optional,
#: and each consumer of one treats its absence as a design without it.
REQUIRED_SECTIONS = (SUMMARY, CHANGES, TESTS)
#: The sections a design needs to be read as a contract design (`parse`), so its
#: summary, its changes and any rule sites it lists are taken from their sections.
CONTRACT_SECTIONS = (SUMMARY, CHANGES)
#: The section the build cannot proceed without: the files to change. A design missing
#: it is the only one sent back for a rewrite; every other gap is listed for the person
#: at the approval gate.
BUILD_SECTIONS = (CHANGES,)


def missing_sections(text: str, sections: tuple[str, ...] = BUILD_SECTIONS) -> list[str]:
    """The *sections* whose ``## heading`` is absent, or present with nothing under it.

    The heading must match exactly (case aside): ``### Changes`` or ``## Changes made``
    is not the ``## Changes`` section.
    """
    lines = text.splitlines()
    return [section for section in sections
            if not any(line for line in _section_lines(lines, section))]


def has_non_empty_section(text: str, section: str) -> bool:
    """Check if the design text has the named ``## section`` with non-empty content."""
    lower = text.lower()
    heading = f"## {section.lower()}"
    pos = lower.find(heading)
    if pos < 0:
        return False
    after = pos + len(heading)
    next_heading = lower.find("\n## ", after)
    content = text[after:next_heading].strip() if next_heading >= 0 else text[after:].strip()
    return bool(content)


@dataclasses.dataclass
class DesignContract:
    title: str
    summary: str
    rule_sites: list[dict[str, str]]
    docs: list[dict[str, str]]
    changes: list[str]
    has_contract: bool


def _strip_prefix(heading: str) -> str:
    """Strip a leading ``Design:`` prefix (case-insensitive)."""
    m = re.match(r'^[Dd]esign:\s*', heading)
    return heading[m.end():].strip() if m else heading.strip()


def _strip_run_name(text: str, run_name: str) -> str:
    """Strip a leading run name and its separator from *text*.

    ``"b16b_graph_ends — Make sinks() agree ..."`` with run_name ``b16b_graph_ends``
    gives ``"Make sinks() agree ..."``.  The separator may be `` — ``, `` - ``,
    `` – ``, or ``: ``.
    """
    if not run_name:
        return text
    collapsed = re.sub(r'\s+', ' ', text).strip()
    name_collapsed = re.sub(r'\s+', ' ', run_name).strip()
    if not collapsed.lower().startswith(name_collapsed.lower()):
        return text
    rest = collapsed[len(name_collapsed):]
    rest = re.sub(r'^[\s]*[—–:\-][\s]*', '', rest, count=1)
    return rest.strip()


def _names_match(a: str, b: str) -> bool:
    """Case-insensitive, whitespace-collapsed equality."""
    return re.sub(r'\s+', ' ', a).strip().lower() == re.sub(r'\s+', ' ', b).strip().lower()


def _section_lines(lines: list[str], heading: str) -> list[str]:
    """Return the body lines between ``## heading`` and the next ``##``."""
    inside = False
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower() == f"## {heading.lower()}":
            inside = True
            continue
        if inside and stripped.startswith("## "):
            break
        if inside:
            result.append(stripped)
    return result


def _bullets(lines: list[str]) -> list[str]:
    """Return lines starting with ``- `` or ``* ``."""
    result: list[str] = []
    for line in lines:
        if line.startswith("- "):
            result.append(line[2:].strip())
        elif line.startswith("* "):
            result.append(line[2:].strip())
    return result


def _h1(lines: list[str]) -> str:
    """The first H1 heading, stripped."""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return ""


# A backticked marker is in marker position only when what follows is the end of
# the entry or optional spaces then a separator character.
_MARKER_SEP = re.compile(r'\s*(?:[—–:;]|- |\.\s*$)|\s*$')


def _find_marker(body: str, bare_pattern: str, backtick_pattern: str) -> re.Match | None:
    """Find the first match, preferring bare/bold over backtick-wrapped.

    Pass 1: try *bare_pattern*, skipping matches inside inline code spans
    (odd preceding backtick count), matches right after a quote character,
    and matches inside an unclosed ``[left`` construction.
    Pass 2: if no bare/bold match, try *backtick_pattern* with a separator
    guard — the match is accepted only when what follows matches
    ``_MARKER_SEP`` (end-of-entry, or optional spaces then ``—``, ``–``,
    ``:``, ``- ``, ``;``, or a final ``.``).
    """
    for m in re.finditer(bare_pattern, body, re.IGNORECASE):
        if body[:m.start()].count('`') % 2 != 0:
            continue
        # (a) Quote guard: skip if immediately preceded by a quote character.
        if m.start() > 0 and body[m.start() - 1] in ('"', "'"):
            continue
        # (b) Open-bracket guard: skip if inside an unclosed [left ...].
        prefix = body[:m.start()]
        last_left = prefix.lower().rfind('[left')
        if last_left >= 0 and ']' not in prefix[last_left + 1:]:
            continue
        return m
    for m in re.finditer(backtick_pattern, body, re.IGNORECASE):
        if _MARKER_SEP.match(body[m.end():]):
            return m
    return None


def parse(text: str, run_name: str = "") -> DesignContract:
    """Parse a design.md into its contract parts."""
    lines = text.splitlines()
    lower_lines = [line.strip().lower() for line in lines]

    # Detect contract: the contract's section headings present (Summary and Changes;
    # Rule and sites and Docs are optional and read when present).
    has_contract = all(
        f"## {sec.lower()}" in lower_lines
        for sec in CONTRACT_SECTIONS
    )

    raw_heading = _h1(lines)
    heading = _strip_prefix(raw_heading)
    heading = _strip_run_name(heading, run_name)

    if has_contract:
        # Contract design.
        title = heading if heading and not _names_match(heading, run_name) else ""

        summary_lines = _section_lines(lines, SUMMARY)
        summary = " ".join(line for line in summary_lines if line)

        rule_lines = _section_lines(lines, RULE_AND_SITES)
        rule_sites: list[dict[str, str]] = []
        for line in rule_lines:
            if not (line.startswith("- ") or line.startswith("* ")):
                continue
            body = (line[2:] if line.startswith("- ") or line.startswith("* ") else line).strip()
            status = "changed"
            reason = ""
            desc = ""
            m_changed = _find_marker(
                body,
                r'\*{0,2}\[(?:changed|new)(?::\s*([^]]*))?\]\*{0,2}',
                r'`\*{0,2}\[(?:changed|new)(?::\s*([^]]*))?\]\*{0,2}`',
            )
            m_left_reason = _find_marker(
                body,
                r'\*{0,2}\[left:\s*(.*?)\]\*{0,2}',
                r'`\*{0,2}\[left:\s*(.*?)\]\*{0,2}`',
            )
            m_left = _find_marker(
                body,
                r'\*{0,2}\[left\]\*{0,2}',
                r'`\*{0,2}\[left\]\*{0,2}`',
            )
            if m_changed:
                status = "changed"
                path_text = body[:m_changed.start()].strip().rstrip("—–-:").strip()
                # Colon-description inside the marker takes priority.
                colon_desc = m_changed.group(1)
                if colon_desc is not None:
                    desc = colon_desc.strip()
                else:
                    desc = body[m_changed.end():].strip()
                    desc = re.sub(r'^[—–\-:\s]+', '', desc).strip()
                desc = desc.rstrip('.')
            elif m_left_reason:
                status = "left"
                reason = m_left_reason.group(1).strip()
                path_text = body[:m_left_reason.start()].strip().rstrip("—–-:").strip()
            elif m_left:
                status = "left"
                reason = ""
                path_text = body[:m_left.start()].strip().rstrip("—–-:").strip()
            else:
                # No marker: the design has not said whether this change covers it.
                status = "unmarked"
                path_text = body
            rule_sites.append({"path": path_text, "status": status, "reason": reason, "description": desc})

        docs_lines = _section_lines(lines, DOCS)
        docs = [{"path": b, "update": ""} for b in _bullets(docs_lines)]

        changes_lines = _section_lines(lines, CHANGES)
        changes = _bullets(changes_lines)

        return DesignContract(
            title=title, summary=summary, rule_sites=rule_sites,
            docs=docs, changes=changes, has_contract=True,
        )
    else:
        # Pre-contract design.
        title = heading
        if _names_match(title, run_name) or not title:
            # Fall back to the brief's first sentence — the caller handles that.
            title = ""

        # Summary: first paragraph after H1, skipping sub-headings.
        summary = ""
        past_heading = False
        para: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not past_heading:
                if stripped.startswith("# "):
                    past_heading = True
                continue
            if stripped.startswith("#"):
                if not para:
                    continue  # skip sub-heading before prose
                break  # hit next heading after collecting prose
            if not stripped and para:
                break
            if stripped.startswith("---") and para:
                break
            if stripped:
                para.append(stripped)
        summary = " ".join(para)

        return DesignContract(
            title=title, summary=summary, rule_sites=[], docs=[],
            changes=[], has_contract=False,
        )


def left_without_reason(parsed: DesignContract) -> list[str]:
    """Sites marked ``[left]`` with no reason."""
    return [site["path"] for site in parsed.rule_sites
            if site["status"] == "left" and not site["reason"]]


def unmarked_sites(parsed: DesignContract) -> list[str]:
    """Sites listed without a ``[changed]`` or ``[left: <reason>]`` marker."""
    return [site["path"] for site in parsed.rule_sites if site["status"] == "unmarked"]


def _extract_file_path(site_path: str) -> str:
    """Extract the file path from a site's ``path`` text.

    Takes the first backticked token, or the text before the first " — ",
    and strips its ``:line-range`` suffix.  For example::

        `delivery.py:989-1022` — `_body()` renders the PR body
        →  delivery.py

        `nn/knob.py:49-81` (`Knob` dataclass)
        →  nn/knob.py
    """
    text = site_path.strip()
    m = re.match(r'`([^`]+)`', text)
    if m:
        path = m.group(1).strip()
    else:
        idx = text.find(' \u2014 ')
        path = text[:idx].strip() if idx >= 0 else text
    return re.sub(r':\d+(-\d+)?$', '', path)


def _site_matches_diff(site_path: str, diff_files: list[str]) -> str | None:
    """Return the diff file that matches *site_path*, or ``None``.

    Extracts the file path from the first backticked token or text before
    " \u2014 ", strips ``:line-range`` suffixes, and matches against *diff_files*.
    Prefers an exact path match.  Accepts a basename/suffix match only when
    exactly one diff file matches.
    """
    cleaned = _extract_file_path(site_path)
    if not cleaned:
        return None
    for df in diff_files:
        if df == cleaned:
            return df
    suffix_matches = [df for df in diff_files if df.endswith('/' + cleaned)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return None


def build_what_changed(
    parsed: DesignContract, diff_files: list[str],
) -> tuple[list[str], list[str]]:
    """Build the "What changed" list from ``[changed]`` sites cross-checked
    against the diff.

    Returns ``(items, mismatches)``:

    - ``items``: one line per changed file for the PR body.  Multiple
      ``[changed]`` sites in the same file are merged into one item with
      their descriptions joined by ``; ``.
    - ``mismatches``: file paths of sites marked ``[changed]`` whose file
      is not in the diff.

    For a pre-contract design (``not parsed.has_contract``), returns the
    diff file list as items with no mismatches.
    """
    if not parsed.has_contract:
        return (list(diff_files), [])

    changed_sites = [s for s in parsed.rule_sites if s['status'] == 'changed']

    file_descriptions: dict[str, list[str]] = {}
    mismatches: list[str] = []

    for site in changed_sites:
        diff_file = _site_matches_diff(site['path'], diff_files)
        if diff_file is not None:
            descs = file_descriptions.setdefault(diff_file, [])
            desc = site.get('description', '')
            if desc:
                descs.append(desc)
        else:
            extracted = _extract_file_path(site['path'])
            mismatches.append(extracted if extracted else site['path'])

    items: list[str] = []
    for diff_file in diff_files:
        if diff_file in file_descriptions:
            descs = file_descriptions[diff_file]
            if descs:
                items.append(f"{diff_file} \u2014 {'; '.join(descs)}")
            else:
                items.append(diff_file)
        else:
            items.append(diff_file)

    return (items, mismatches)


def changed_file(path: str, diff_files: list[str]) -> str | None:
    """The file of ``diff_files`` that ``path`` names, or None (`_site_matches_diff`)."""
    return _site_matches_diff(f"`{path}`", diff_files)


#: A token that names a file by its path: a directory and an extension.
_PATH = re.compile(r"^[^\s`]+/[^\s`]*\.\w+$")


def _named_files(bullet: str, tracked: set[str]) -> list[str]:
    """The files a ``## Changes`` bullet names: its leading path and its backticked tokens
    that are paths (`_PATH`) or files the repository tracks. ``Core: the tokenizer…`` and
    ``tool --list`` name none."""
    tokens = [_extract_file_path(bullet), *re.findall(r"`([^`]+)`", bullet)]
    names: list[str] = []
    for token in tokens:
        token = re.sub(r":\d+(-\d+)?$", "", token.strip())
        if token and (_PATH.match(token) or token in tracked) and token not in names:
            names.append(token)
    return names


def describe_changes(
    parsed: DesignContract, diff_files: list[str], tracked: set[str] | frozenset[str] = frozenset(),
    deleted: list[str] | tuple[str, ...] = (),
) -> tuple[list[str], list[str]]:
    """The pull request's "Changes" (dm-191): the design's ``## Changes`` bullets, one per
    behaviour, as written (each cut to 300 characters), then each file in the diff that no
    kept bullet names, as `build_what_changed` gives it (its ``[changed]`` sites'
    descriptions, or its name).

    A bullet is left out only when it names files (`_named_files`; ``tracked`` is the
    repository's files) and the branch neither changes nor deletes (``deleted``) any of
    them: what it describes is not in the branch. A bullet that names no file -- "Core: the tokenizer…" or "Behaviour
    change: ``tool --list`` prints…" -- is a behaviour, and is kept.

    Returns ``(items, mismatches)``: ``mismatches`` names the files a left-out bullet or a
    ``[changed]`` site names that the diff does not change.
    """
    by_sites, mismatches = build_what_changed(parsed, diff_files)
    kept: list[str] = []
    described: set[str] = set()
    for bullet in parsed.changes:
        names = _named_files(bullet, set(tracked) | set(diff_files) | set(deleted))
        touched = [*diff_files, *deleted]
        changed = [match for match in (changed_file(name, touched) for name in names) if match]
        if names and not changed:
            mismatches.extend(names)
            continue
        described.update(changed)
        text = " ".join(bullet.split())
        kept.append(text if len(text) <= 300 else text[:299].rstrip() + "…")
    return ([*kept, *(item for path, item in zip(diff_files, by_sites) if path not in described)],
            list(dict.fromkeys(mismatches)))


#: The sections a design may write for the pull request itself (dm-191), each carried into
#: its description under the same heading.
PR_SECTIONS = ("Core", "Behaviour changes", "Behavior changes")


def pr_sections(text: str) -> list[tuple[str, str]]:
    """The design's ``## Core`` and ``## Behaviour changes`` sections, as written, for the
    pull request: ``(heading, text)`` for each the design has with something under it."""
    lines = text.splitlines()
    found: list[tuple[str, str]] = []
    for heading in PR_SECTIONS:
        body = "\n".join(_section_lines(lines, heading)).strip()
        if body:
            found.append((heading, body))
    return found


def test_names(text: str) -> list[str]:
    """The tests a design's ``## Tests`` names: each bullet's first backticked id."""
    names: list[str] = []
    for bullet in _bullets(_section_lines(text.splitlines(), TESTS)):
        named = re.match(r"`([^`]+)`", bullet)
        if named:
            names.append(named.group(1))
    return names


def title_is_run_name(text: str, run_name: str) -> bool:
    """True when the H1 heading, after stripping the Design: prefix, equals the run name or starts with it and leaves nothing after the separator."""
    if not run_name:
        return False
    raw = _strip_prefix(_h1(text.splitlines()))
    return bool(raw) and not _strip_run_name(raw, run_name)


def _levenshtein(a: str, b: str) -> int:
    """Levenshtein distance for short strings (≤ 12 chars each)."""
    if len(a) > 12 or len(b) > 12:
        return 999
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if a[i - 1] == b[j - 1] else 1))
            prev = tmp
    return dp[n]


#: Words that appear in brackets in ordinary prose or markdown and are not
#: marker attempts, even though some are within edit-distance 2 of ``left``.
_NOT_MARKER_WORDS = frozenset({
    'see', 'note', 'todo', 'test', 'list', 'set', 'ref', 'fix', 'let',
    'add', 'get', 'put', 'run', 'use', 'try', 'end', 'key', 'map', 'var',
    'log', 'old', 'int', 'str', 'tag', 'type', 'link', 'text', 'file',
    'changes', 'change', 'leaves', 'lift', 'rest', 'feat', 'best',
})


def find_near_miss_markers(site_text: str) -> list[str]:
    """For an unmarked site line, find bracket expressions that look like misspelled markers.

    Returns suggestions like ``"possible misspelled marker [chnaged] (did you
    mean [changed]?)"``.  Ignores link text ``[text](url)`` and common
    non-marker words.
    """
    suggestions: list[str] = []
    # Strip link text: [text](url)
    cleaned = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', site_text)
    for m in re.finditer(r'\[(\w+)(?::[^\]]*)?\]', cleaned):
        word = m.group(1).lower()
        if word in ('changed', 'left', 'new'):
            continue  # recognised marker keyword
        if word in _NOT_MARKER_WORDS:
            continue
        if len(word) > 15:
            continue
        for target in ('changed', 'left'):
            if _levenshtein(word, target) <= 2:
                suggestions.append(
                    f"possible misspelled marker [{m.group(1)}] "
                    f"(did you mean [{target}]?)"
                )
                break
    return suggestions


def prompt_contract() -> str:
    """The text the design_doc prompt appends: the light template.

    Three sections are required; the rest are optional, and every consumer of an
    optional section treats its absence as a design without it.
    """
    return (
        "Your design.md must contain, in this order:\n\n"
        "1. An H1 heading (`#`) that says what the change does. Never use "
        "the run name or artifact name as the title.\n\n"
        "2. `## Summary` — the problem and the change, in two to four sentences.\n\n"
        "3. `## Changes` — one bullet per file the change adds or modifies: "
        "`path` — what changes. This is the one section the build cannot "
        "proceed without.\n\n"
        "4. `## Tests` — one bullet per test: its repo-relative id (for example "
        "`tests/test_foo.py::test_bar`) and what it proves.\n\n"
        "When the person corrects the design at the approval gate, the step that "
        "applies the corrections adds `## Corrections at approval`; do not write it "
        "yourself.\n\n"
        "Add any of these optional sections only where it helps the build; leave out "
        "the ones that would say nothing:\n"
        "- `## Failure trace` — for a failure the brief reports, a file:line chain "
        "from the symptom to the root cause.\n"
        "- `## Ask mapping` — a table `| Ask (quoted from brief) | Design section | "
        "Test |`, one row per ask, quoting the brief verbatim.\n"
        "- `## Rule and sites` — the rule the change makes or corrects, then each place "
        "in the code that applies or restates it, as a bullet citing file and line, "
        "marked [changed] or [left: <reason>].\n"
        "- `## Test guardedness exemptions` — when a new test passes on the base (for "
        "example one that pins existing behaviour), its file path and why it cannot "
        "fail without the change. A new test that passes on the base and is not listed "
        "here is rejected.\n"
        "- `## Site revert exemptions` — a changed site that a mechanical "
        "revert-and-test cannot catch (logging, a side channel no test observes), as "
        "`path:line` and the reason.\n"
        "- `## Anti-patterns to avoid` — for a high-risk pitfall, the site or "
        "§ section that handles it, or `[set-aside: <reason>]`.\n"
        "- `## Docs` — the docs that describe the changed behaviour, and the update "
        "each needs.\n"
        "- `## Core` and `## Behaviour changes` — when the brief asks the pull request for "
        "them: the core of the change, and each behaviour a user will see change. Delivery "
        "carries both into the pull request's description as written.\n"
        "- `## Runtime coverage` — which runtimes run the changed code, and how each "
        "is covered.\n"
    )
