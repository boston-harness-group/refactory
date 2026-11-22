"""Tests for finding coverage: research and perspective findings must be
addressed or set aside in the design.

Each test fails without the corresponding production change.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import nn
from refactory_dsh.nn import Verdict
from test_pitfalls import _contract
from tests_support import (
    DESIGN_DOC_ANSWER,
    FakeModel,
    PITFALLS_ANSWER,
    RESEARCH_ANSWER,
    REVIEW_ANSWERS,
    VERIFICATION_ANSWER,
    render,
    strategy_answer,
)


# ── gate integration tests (1–4) ──────────────────────────────────────────

#: A design run plans its delivery by asking gh for its login, so it asks the fake (dm-271).
pytestmark = pytest.mark.usefixtures("fake_gh")


def test_gate_records_an_uncited_research_finding_without_listing_it(tmp_path):
    """1: A finding in a research file, missing from the design, fails the gate."""
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({
        'source': str(tmp_path),
        'design_review_attempts_used': 0,
        'has_pitfalls_step': False,
    })
    ctx._declared_reads = ('reviews/design-review.md', 'design.md', 'brief.json',
                            'research/')
    (tmp_path / 'reviews').mkdir(parents=True, exist_ok=True)
    ctx.write_text('reviews/design-review.md',
                   'All confirmed.\n\nVERDICT: PASS\n')
    (tmp_path / 'research').mkdir(exist_ok=True)
    (tmp_path / 'research' / 'foo.md').write_text(
        '### FND-foo-1 \u2014 lowering op-name strings\n\n# Foo\n\nDetails.\n')
    ctx.write_text('design.md', _contract(
        body='## Research and perspective findings\n\n(empty)'))
    ctx.write_json('brief.json', {'name': 'x', 'brief': 'do a thing'})
    import refactory_dsh.design_ops as ops
    verdict = ops.gate_design_review(ctx)
    # A design cites only the findings that change a decision: an uncited one is
    # recorded for the reviewer, never listed at the gate or sent back.
    assert verdict == 'proceed'
    assert 'FND-foo-1: MISSING' in ctx.facts['finding_coverage']
    assert 'FND-foo-1' not in ctx.facts['design_review_advisory']
    assert 'FND-foo-1' not in ctx.facts['design_review_findings']


def test_gate_records_an_uncited_perspective_finding_without_listing_it(tmp_path):
    """2: A finding in a perspective file, missing from the design, fails."""
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({
        'source': str(tmp_path),
        'design_review_attempts_used': 0,
        'has_pitfalls_step': False,
    })
    ctx._declared_reads = ('reviews/design-review.md', 'design.md', 'brief.json',
                            'strategy/')
    (tmp_path / 'reviews').mkdir(parents=True, exist_ok=True)
    ctx.write_text('reviews/design-review.md',
                   'All confirmed.\n\nVERDICT: PASS\n')
    (tmp_path / 'strategy').mkdir(exist_ok=True)
    (tmp_path / 'strategy' / 'bar.md').write_text(
        '### FND-bar-1 \u2014 refactory_packages must be walked\n\n'
        '# Bar\n\nDetails.\n')
    ctx.write_text('design.md', _contract(
        body='## Research and perspective findings\n\n(empty)'))
    ctx.write_json('brief.json', {'name': 'x', 'brief': 'do a thing'})
    import refactory_dsh.design_ops as ops
    verdict = ops.gate_design_review(ctx)
    assert verdict == 'proceed'
    assert 'FND-bar-1: MISSING' in ctx.facts['finding_coverage']
    assert 'FND-bar-1' not in ctx.facts['design_review_advisory']


def test_gate_passes_when_finding_set_aside(tmp_path):
    """3: A finding set aside with a reason passes."""
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({
        'source': str(tmp_path),
        'design_review_attempts_used': 0,
        'has_pitfalls_step': False,
    })
    ctx._declared_reads = ('reviews/design-review.md', 'design.md', 'brief.json',
                            'research/')
    (tmp_path / 'reviews').mkdir(parents=True, exist_ok=True)
    ctx.write_text('reviews/design-review.md',
                   'All confirmed.\n\nVERDICT: PASS\n')
    (tmp_path / 'research').mkdir(exist_ok=True)
    (tmp_path / 'research' / 'foo.md').write_text(
        '### FND-foo-1 \u2014 some finding\n\n# Foo\n\nDetails.\n')
    ctx.write_text('design.md', _contract(
        body='## Research and perspective findings\n\n'
             '- FND-foo-1: [set-aside: not relevant to this change]'))
    ctx.write_json('brief.json', {'name': 'x', 'brief': 'do a thing'})
    import refactory_dsh.design_ops as ops
    result = ops.gate_design_review(ctx)
    assert result == 'proceed'
    assert 'set aside' in ctx.facts.get('finding_coverage', '')


def test_dropped_finding_still_reaches_coverage(tmp_path):
    """4: A finding dropped by the summary still blocks the design."""
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({
        'source': str(tmp_path),
        'design_review_attempts_used': 0,
        'has_pitfalls_step': False,
    })
    ctx._declared_reads = ('reviews/design-review.md', 'design.md', 'brief.json',
                            'research/')
    (tmp_path / 'reviews').mkdir(parents=True, exist_ok=True)
    ctx.write_text('reviews/design-review.md',
                   'All confirmed.\n\nVERDICT: PASS\n')
    (tmp_path / 'research').mkdir(exist_ok=True)
    (tmp_path / 'research' / 'foo.md').write_text(
        '### FND-foo-1 \u2014 finding A\n\n# Foo\n\nDetails.\n')
    (tmp_path / 'research' / 'bar.md').write_text(
        '### FND-bar-1 \u2014 finding B\n\n# Bar\n\nDetails.\n')
    # The design addresses foo but not bar (summary dropped bar)
    ctx.write_text('design.md', _contract(
        body='## Research and perspective findings\n\n'
             '- FND-foo-1: addressed at `file.py:1`'))
    ctx.write_json('brief.json', {'name': 'x', 'brief': 'do a thing'})
    import refactory_dsh.design_ops as ops
    verdict = ops.gate_design_review(ctx)
    assert verdict == 'proceed'
    assert 'FND-bar-1: MISSING' in ctx.facts['finding_coverage']
    assert 'FND-foo-1: addressed' in ctx.facts['finding_coverage']


# ── unit tests: finding_coverage (5–10) ────────────────────────────────────

def test_bare_id_without_site_is_not_addressed():
    """5: A bare id with no site or heading is missing."""
    from refactory_dsh.findings_coverage import finding_coverage
    design = _contract(
        body='## Research and perspective findings\n\n- FND-foo-1\n')
    findings = [{'id': 'FND-foo-1', 'source_file': 'research/foo.md',
                 'text': 'a finding'}]
    _, _, missing = finding_coverage(design, findings)
    assert 'FND-foo-1' in missing


def test_id_at_site_is_addressed():
    """6: An id with a valid site reference is addressed."""
    from refactory_dsh.findings_coverage import finding_coverage
    design = _contract(
        body='## Research and perspective findings\n\n'
             '- FND-foo-1: addressed at `file.py:1`\n')
    findings = [{'id': 'FND-foo-1', 'source_file': 'research/foo.md',
                 'text': 'a finding'}]
    addressed, _, _ = finding_coverage(design, findings)
    assert 'FND-foo-1' in addressed


def test_id_at_heading_is_addressed():
    """7: An id with a § heading reference is addressed."""
    from refactory_dsh.findings_coverage import finding_coverage
    design = _contract(
        body='## Research and perspective findings\n\n'
             '- FND-foo-1: § Changes\n')
    findings = [{'id': 'FND-foo-1', 'source_file': 'research/foo.md',
                 'text': 'a finding'}]
    addressed, _, _ = finding_coverage(design, findings)
    assert 'FND-foo-1' in addressed


def test_id_at_a_place_in_a_section_is_addressed():
    """A § reference may say where in the section, and may name a subsection."""
    from refactory_dsh.findings_coverage import finding_coverage
    design = _contract(
        body='### Sentence insertion\n\nInsert it.\n\n'
             '## Research and perspective findings\n\n'
             '- FND-foo-1: § Changes item 3\n'
             '- FND-foo-2: § Sentence insertion, relative link path\n')
    findings = [{'id': f'FND-foo-{n}', 'source_file': 'research/foo.md',
                 'text': 'a finding'} for n in (1, 2)]
    addressed, _, _ = finding_coverage(design, findings)
    assert addressed == ['FND-foo-1', 'FND-foo-2']


def test_finding_coverage_set_aside():
    """8: A set-aside marker is recognised."""
    from refactory_dsh.findings_coverage import finding_coverage
    design = _contract(
        body='## Research and perspective findings\n\n'
             '- FND-foo-1: [set-aside: reason]\n')
    findings = [{'id': 'FND-foo-1', 'source_file': 'research/foo.md',
                 'text': 'a finding'}]
    _, set_aside, _ = finding_coverage(design, findings)
    assert 'FND-foo-1' in set_aside


def test_finding_coverage_missing():
    """9: A finding not in the section is missing."""
    from refactory_dsh.findings_coverage import finding_coverage
    design = _contract(
        body='## Research and perspective findings\n\n(nothing here)\n')
    findings = [{'id': 'FND-foo-1', 'source_file': 'research/foo.md',
                 'text': 'a finding'}]
    _, _, missing = finding_coverage(design, findings)
    assert 'FND-foo-1' in missing


def test_full_id_equality_matching():
    """10: FND-foo-10 does not satisfy FND-foo-1."""
    from refactory_dsh.findings_coverage import finding_coverage
    design = _contract(
        body='## Research and perspective findings\n\n'
             '- FND-foo-10: addressed at `file.py:1`\n')
    findings = [
        {'id': 'FND-foo-1', 'source_file': 'research/foo.md', 'text': 'first'},
        {'id': 'FND-foo-10', 'source_file': 'research/foo.md', 'text': 'tenth'},
    ]
    addressed, _, missing = finding_coverage(design, findings)
    assert 'FND-foo-1' in missing
    assert 'FND-foo-10' in addressed


# ── unit tests: collect_findings and definitions (11–17) ───────────────────

def test_collect_findings_dashed_slugs(tmp_path):
    """11: Slugs with dashes (api-surface, adversarial-gaps, step-2) parse."""
    from refactory_dsh.findings_coverage import collect_findings
    (tmp_path / 'research').mkdir()
    (tmp_path / 'research' / 'api-surface.md').write_text(
        '### FND-api-surface-1 \u2014 first finding\n')
    (tmp_path / 'strategy').mkdir()
    (tmp_path / 'strategy' / 'adversarial-gaps.md').write_text(
        '### FND-adversarial-gaps-1 \u2014 a gap\n')
    (tmp_path / 'research' / 'step-2.md').write_text(
        '### FND-step-2-3 \u2014 third finding of step-2\n')
    findings, blocking = collect_findings(tmp_path, ['research/', 'strategy/'])
    ids = {f['id']: f for f in findings}
    assert 'FND-api-surface-1' in ids
    assert ids['FND-api-surface-1']['slug'] == 'api-surface'
    assert 'FND-adversarial-gaps-1' in ids
    assert ids['FND-adversarial-gaps-1']['slug'] == 'adversarial-gaps'
    assert 'FND-step-2-3' in ids
    assert ids['FND-step-2-3']['slug'] == 'step-2'
    assert not blocking


def test_gate_research_rejects_files_with_no_ids(tmp_path):
    """12: gate_research reloops when a research file has no FND ids."""
    from refactory_dsh import create_ops
    workdir = tmp_path / 'run'
    workdir.mkdir()
    entries = [{'focus': 'question', 'slug': 'foo', 'finding': 'an answer'}]
    (workdir / 'research-plan.json').write_text(json.dumps(entries))
    (workdir / 'research').mkdir()
    (workdir / 'research' / 'foo.md').write_text(
        '# Question\n\nAn answer with no ids.\n')
    facts: dict[str, Any] = {
        'research_scope': 'scoped', 'research_attempts_used': 0}
    ctx = type('Ctx', (), {
        'workdir': workdir,
        'knob': lambda self, name, default='': (
            {'research_attempts': 3}.get(name, default)),
        'read_text': lambda self, name: (workdir / name).read_text()
            if (workdir / name).exists() else '',
        'exists': lambda self, name: (workdir / name).exists(),
        'write_text': lambda self, name, text: (workdir / name).write_text(text),
        'set_fact': lambda self, name, value: facts.update({name: value}),
        'facts': facts,
    })()
    verdict = create_ops.gate_research(ctx)
    assert isinstance(verdict, Verdict)
    assert 'no FND' in verdict.reason or 'finding ids' in verdict.reason


def test_file_with_no_ids_becomes_path_finding(tmp_path):
    """13: A strategy file with no ids becomes a finding whose id is its path."""
    from refactory_dsh.findings_coverage import collect_findings
    (tmp_path / 'strategy').mkdir()
    (tmp_path / 'strategy' / 'bar.md').write_text(
        '# A strategy file\n\nNo ids here.\n')
    findings, blocking = collect_findings(tmp_path, ['strategy/'])
    assert len(findings) == 1
    assert findings[0]['id'] == 'strategy/bar.md'


def test_slug_mismatch_is_blocking(tmp_path):
    """14: An id whose slug does not match its file is blocking."""
    from refactory_dsh.findings_coverage import collect_findings
    (tmp_path / 'research').mkdir()
    (tmp_path / 'research' / 'foo.md').write_text(
        '### FND-bar-1 \u2014 something\n')
    findings, blocking = collect_findings(tmp_path, ['research/'])
    assert any('slug' in p and 'bar' in p for p in blocking)


def test_duplicate_id_is_blocking(tmp_path):
    """15: Duplicate ids are blocking."""
    from refactory_dsh.findings_coverage import collect_findings
    (tmp_path / 'research').mkdir()
    (tmp_path / 'research' / 'foo.md').write_text(
        '### FND-foo-1 \u2014 first\n### FND-foo-1 \u2014 second\n')
    findings, blocking = collect_findings(tmp_path, ['research/'])
    assert any('duplicate' in p.lower() and 'FND-foo-1' in p for p in blocking)


def test_definition_vs_reference(tmp_path):
    """16: A mid-line reference is not a definition (no slug check)."""
    from refactory_dsh.findings_coverage import collect_findings
    (tmp_path / 'strategy').mkdir()
    (tmp_path / 'strategy' / 'adversarial-gaps.md').write_text(
        '### FND-adversarial-gaps-1 \u2014 a gap\n\n'
        'see FND-api-surface-1 from the research\n')
    findings, blocking = collect_findings(tmp_path, ['strategy/'])
    ids = {f['id'] for f in findings}
    assert 'FND-adversarial-gaps-1' in ids
    # No slug mismatch for the mid-line reference
    assert not any('api-surface' in p and 'slug' in p for p in blocking)


def test_definitions_with_markdown_decoration():
    """17: A bulleted bold definition is parsed."""
    from refactory_dsh.findings_coverage import definitions
    defs = definitions('- **FND-api-1** \u2014 bulleted finding\n')
    assert len(defs) == 1
    assert defs[0][0] == 'FND-api-1'


# ── scripted run tests (18–20) ────────────────────────────────────────────

def test_scripted_design_run_dropped_finding(tmp_path):
    """18: A finding dropped by the summary still reaches the coverage check, which
    records it for the reviewer instead of sending the design back."""
    from refactory_dsh.design import DesignMode

    answers = [
        RESEARCH_ANSWER,
        PITFALLS_ANSWER,
        "perspectives",
        "one plan",
        DESIGN_DOC_ANSWER,      # does not mention FND-extra-1
        render({"tool.py": "print('built')\n"}),
        *REVIEW_ANSWERS[:3],
        VERIFICATION_ANSWER,
        REVIEW_ANSWERS[3],
    ]
    model = FakeModel(answers)
    model._write_extra_research = True

    result = DesignMode().run(
        tmp_path / "run",
        model=model,
        verdicts={"gate_strategy": "proceed"},
        task={"name": "my_tool", "brief": "a tool", "check": "python tool.py"},
    )
    assert result.exit_ok, result.error

    # One design round: the uncovered finding did not send the design back.
    design_doc_prompts = [
        p for p in model.prompts if 'What the design checks found' in p
    ]
    assert len(design_doc_prompts) == 1
    assert 'FND-extra-1: MISSING' in result.facts['finding_coverage']
    assert 'FND-extra-1' not in result.facts['design_review_advisory']
    assert result.facts['design_review_status'].startswith('Design review: PASS')


def test_scripted_create_run_finding_coverage(tmp_path, monkeypatch):
    """19: A create run whose strategy drops a finding raises CreateError,
    and strategy-review.txt names the missing finding."""
    project = tmp_path / "project"  # a create run refuses refactory's checkout (C1)
    project.mkdir()
    monkeypatch.setenv("REFACTORY_PROJECT", str(project))
    from refactory_dsh.create import CreateConfig, CreateError, create

    def bad_strategy(name: str) -> str:
        ops = f"refactory_packages.{name}.ops"
        return json.dumps({
            "considered": [
                {"approach": "count up", "why_not": ""},
                {"approach": "compute", "why_not": "too complex"},
            ],
            "chosen": "count up",
            "why": "simplest",
            "will_not": ["threads"],
            "closest_existing": "circle_packing/adaptive",
            "knobs": [{"name": "target", "kind": "int", "default": 3,
                        "lo": 1, "hi": 9, "description": "where to stop"}],
            "nodes": [
                {"id": "start", "kind": "op", "call": f"{ops}:start",
                 "writes": ["state.json"], "produces": ["value"],
                 "knobs": ["target"]},
                {"id": "step", "kind": "op", "call": f"{ops}:step",
                 "reads": ["state.json"], "writes": ["state.json"],
                 "produces": ["value"]},
                {"id": "check", "kind": "gate", "evaluator": f"{ops}:check",
                 "reads": ["state.json"], "produces": ["done"],
                 "max_iterations": 20},
                {"id": "finish", "kind": "op", "call": f"{ops}:finish",
                 "reads": ["state.json"], "writes": ["best.json"],
                 "produces": ["best"]},
            ],
            "edges": [
                {"source": "start", "target": "step"},
                {"source": "step", "target": "check"},
                {"source": "check", "target": "step", "condition": "retry"},
                {"source": "check", "target": "finish", "condition": "proceed"},
            ],
            "start": "start",
            "score": "best",
            "findings": {
                # Deliberately drop FND-api-surface-1
                "FND-existing-modes-1": "step node handles existing modes",
                "FND-acceptance-1": "check node handles acceptance",
                "FND-simplest-1": "step node handles simplest",
                "FND-criteria-1": "check node handles criteria",
            },
        })

    name = "test_pkg"
    answers = [
        RESEARCH_ANSWER,
        "perspectives",
        bad_strategy(name),   # first attempt — drops FND-api-surface-1
        bad_strategy(name),   # second attempt — still drops it
    ]
    model = FakeModel(answers)

    with pytest.raises(CreateError):
        create(
            CreateConfig(
                name=name,
                brief="count to a target and score how close",
                single_shot=True,
                workdir=tmp_path / "create",
            ),
            client=model,
        )

    review_path = tmp_path / "create" / "strategy-review.txt"
    assert review_path.exists()
    review_text = review_path.read_text()
    assert 'FND-api-surface-1' in review_text


def test_dsh_run_node_reports_missing_finding(tmp_path):
    """20: gate_design_review reports a missing finding on DSH."""
    from refactory_dsh.design import DesignMode
    from refactory_dsh import lowering

    state = DesignMode().state_dict()
    d = tmp_path / 'run'
    d.mkdir()
    lowering.write_declarations(state, d)

    (d / 'research').mkdir()
    (d / 'research' / 'foo.md').write_text(
        '### FND-foo-1 \u2014 important finding\n\n# Foo\n\nDetails.\n')

    (d / 'reviews').mkdir()
    (d / 'reviews' / 'design-review.md').write_text(
        'All confirmed.\n\nVERDICT: PASS\n')

    (d / 'design.md').write_text(_contract(
        body='## Research and perspective findings\n\n(nothing)\n'))
    (d / 'brief.json').write_text(json.dumps(
        {'name': 'x', 'brief': 'do a thing'}))

    (d / '.refactory').mkdir(exist_ok=True)
    facts_data = {
        'source': str(tmp_path),
        'design_review_attempts_used': 0,
    }
    (d / '.refactory' / 'facts.json').write_text(json.dumps(facts_data))

    from refactory_dsh.run import run_node
    result = run_node(
        'refactory_dsh.design_ops:gate_design_review',
        workdir=d,
        node_id='gate_design_review',
    )

    assert result == 'proceed'
    feedback = (d / 'reviews' / 'design-review-feedback.txt').read_text()
    assert feedback.startswith('nothing blocks the build')
    assert 'FND-foo-1' not in feedback


# ── code spans (dm-146) and entries under a heading (dm-219) ───────────────

def _finding(n: int = 1) -> dict:
    return {'id': f'FND-foo-{n}', 'source_file': 'research/foo.md', 'text': 'a finding'}


def test_a_double_backtick_span_in_the_rule_does_not_shift_the_sites_after_it():
    """dm-146: a run of N backticks closes only on a run of N, so a double-backtick span
    holding a single backtick leaves every later site path where it is."""
    from refactory_dsh import coverage_helpers
    from refactory_dsh.findings_coverage import finding_coverage
    design = _contract(
        body='## Research and perspective findings\n\n'
             '- FND-foo-1: the ``===FILE: `x` ===`` header is parsed at `other.py:2`\n'
    ).replace('- `file.py:1` — **[changed]**\n',
              'Rule: a header reads ``===FILE: `path` ===``, the path in backticks.\n'
              '- `file.py:1` — **[changed]**\n- `other.py:2` — **[changed]**\n')
    assert coverage_helpers.site_paths(design) == [
        '===FILE: `path` ===', 'file.py:1', 'other.py:2']
    assert finding_coverage(design, [_finding()]) == (['FND-foo-1'], [], [])


def test_a_double_backtick_span_does_not_shift_citations_in_a_design_without_a_rule_section():
    """The same for a design that names its sites where it uses them."""
    from refactory_dsh import coverage_helpers
    design = _contract(body='Rule: ``a `b` c`` stays one span; the fix is at `file.py:3`.\n')
    design = design.replace('## Rule and sites\n\n- `file.py:1` — **[changed]**\n\n', '')
    assert coverage_helpers.site_paths(design) == ['file.py:3']


def test_code_spans_are_read_the_commonmark_way():
    from refactory_dsh.coverage_helpers import code_spans
    assert code_spans('``a `b` c`` then `d`') == ['a `b` c', 'd']
    assert code_spans('`` ` `` and ` e `') == ['`', 'e']
    # A run nothing closes is literal; an escaped backtick opens nothing.
    assert code_spans('a ``` b `c`') == ['c']
    assert code_spans(r'\`not a span` but `this`') == ['but']
    # A blank line ends any span still open.
    assert code_spans('a `open\n\nclosed `x` b') == ['x']


def test_a_finding_answered_under_its_own_heading_is_addressed():
    """dm-219: an entry written as a heading reads on through the block under it, up to the
    next heading of its level or the next line naming another finding."""
    from refactory_dsh.findings_coverage import finding_coverage
    design = _contract(
        body='## Research and perspective findings\n\n'
             '### FND-foo-1 (the cache key)\n\n'
             'The base is read once, so the key holds:\n\n- at `file.py:1`\n\n'
             '### FND-foo-2\n\nNothing cited here.\n\n'
             '### Notes\n\nThe key is at `file.py:1`.\n\n'
             '### FND-foo-3\n\n- FND-foo-4 is answered at `file.py:1`\n')
    addressed, _, missing = finding_coverage(design, [_finding(n) for n in (1, 2, 3, 4)])
    assert addressed == ['FND-foo-1', 'FND-foo-4']
    assert missing == ['FND-foo-2', 'FND-foo-3']


def test_a_cross_reference_inside_a_headings_block_does_not_end_the_entry():
    """Review of #519: a line that mentions another finding stays in the block; only a line
    that starts with one (or a heading) begins the next entry."""
    from refactory_dsh.findings_coverage import finding_coverage
    design = _contract(
        body='## Research and perspective findings\n\n'
             '### FND-foo-1\n\nSee also FND-foo-2 for context.\n\nAddressed at `file.py:1`.\n\n'
             '### FND-foo-2\n\nNothing.\n')
    addressed, _, missing = finding_coverage(design, [_finding(1), _finding(2)])
    assert addressed == ['FND-foo-1'] and missing == ['FND-foo-2']


def test_a_hash_line_in_a_code_fence_does_not_end_a_headings_entry():
    from refactory_dsh.findings_coverage import finding_coverage
    design = _contract(
        body='## Research and perspective findings\n\n'
             '### FND-foo-1\n\n```python\n# the key is read once\nkey = read()\n```\n\n'
             'Addressed at `file.py:1`.\n')
    assert finding_coverage(design, [_finding()]) == (['FND-foo-1'], [], [])
