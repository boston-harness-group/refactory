"""Live public schemas and bounded, installation-owned documentation retrieval."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

DATA = Path(__file__).with_name('data')
SCHEMA_VERSION = 1
MAX_READ = 12000


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def parser_operations(parser: argparse.ArgumentParser) -> dict[str, Any]:
    """Use the invocation parser itself, including aliases, types and defaults."""
    sub = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    summaries = {action.dest: action.help for action in sub._choices_actions}
    result = {}
    for name, command in sorted(sub.choices.items()):
        parameters = {}
        position = 0
        for action in command._actions:
            # A hidden flag is no parameter: it only refuses with what to use instead.
            if action.dest == 'help' or action.help == argparse.SUPPRESS:
                continue
            kind = ('boolean' if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction))
                    else 'integer' if action.type is int else 'number' if action.type is float
                    else 'object' if action.type is json.loads else 'string')
            parameters[action.dest] = {
                'type': 'array' if isinstance(action, argparse._AppendAction) else kind,
                **({'items': {'type': kind}} if isinstance(action, argparse._AppendAction) else {}),
                'flags': action.option_strings, 'positional': not action.option_strings,
                **({'position': position} if not action.option_strings else {}),
                'required': bool(action.required), 'default': action.default,
                'description': action.help or '', 'nargs': action.nargs,
                **({'enum': list(action.choices)} if action.choices is not None else {}),
            }
            if not action.option_strings:
                position += 1
        result[name] = {'description': summaries.get(name, ''), 'parameters': parameters}
    return result


def operations() -> dict[str, Any]:
    from refactory_dsh.console import _parser
    from refactory_dsh.cli import build_parser, READ_VERBS
    result = parser_operations(_parser())
    # Forwarding aliases use the console's full schema, never the empty CLI stub.
    for name, entry in parser_operations(build_parser()).items():
        if name not in READ_VERBS:
            result['cli:' + name] = entry
    from refactory_dsh.profiles import defaults
    from refactory_dsh.workers import BACKENDS, capabilities
    from refactory_dsh.worker_config import FIELDS, FILENAME
    result['workers']['configuration_schema'] = {'filename': FILENAME, 'fields': list(FIELDS), 'backends': {name: capabilities(name) for name in BACKENDS}}
    for name in ('start', 'profile-resolve', 'cli:optimize'):
        result[name]['effective_defaults'] = defaults()
    return result


def coverage() -> dict[str, Any]:
    return json.loads((DATA / 'coverage.json').read_text())


def bundle() -> dict[str, Any]:
    path = DATA / 'documentation.json'
    if not path.is_file():
        raise ValueError('Shipped documentation is missing; run pnpm run docs:generate and rebuild the installation')
    value = json.loads(path.read_text())
    if value.get('schema_version') != SCHEMA_VERSION or digest(value['pages']) != value.get('documentation_id'):
        raise ValueError('Shipped documentation index is corrupt or incompatible; rebuild the installation')
    return value


def identity() -> dict[str, Any]:
    data = bundle()
    return {key: data[key] for key in ('schema_version', 'runtime_version', 'build_id', 'documentation_id')}


def doc_index() -> list[dict[str, Any]]:
    return [{'page': key, 'title': page['title'], 'sections': page['sections'], 'sha256': page['sha256']}
            for key, page in bundle()['pages'].items()]


def docs(*, page: str = '', query: str = '', offset: int = 0, limit: int = 6000,
         section: str = '') -> dict[str, Any]:
    """Read exact indexed IDs only; no filesystem path or arbitrary file reads."""
    data = bundle()
    if not isinstance(offset, int) or offset < 0 or not isinstance(limit, int) or not 1 <= limit <= MAX_READ:
        raise ValueError(f'docs needs offset >= 0 and limit between 1 and {MAX_READ} characters')
    result = {key: data[key] for key in ('schema_version', 'runtime_version', 'build_id', 'documentation_id')}
    if page:
        if page not in data['pages']:
            raise ValueError('Unknown shipped documentation page; use docs without page or search query for exact page IDs')
        entry = data['pages'][page]
        text = entry['text']
        if section:
            matches = [item for item in entry['sections'] if item['id'] == section or item['title'] == section]
            if len(matches) != 1:
                raise ValueError('Unknown or ambiguous section; use an exact section ID from the documentation index')
            selected = matches[0]
            text = text[selected['start']:selected['end']]
        end = min(len(text), offset + limit)
        return {**result, 'page': page, 'section': section, 'sha256': entry['sha256'],
                'offset': offset, 'text': text[offset:end], 'next_offset': end if end < len(text) else None,
                'total_characters': len(text), 'links': entry['links']}
    if query:
        terms = re.findall(r'[\w-]+', query.lower())
        if not terms:
            raise ValueError('docs search needs a word')
        hits = []
        for key, entry in data['pages'].items():
            for item in entry['sections']:
                text = entry['text'][item['start']:item['end']]
                lower = text.lower()
                score = sum(lower.count(term) for term in terms)
                if score and all(term in lower for term in terms):
                    position = min(lower.find(term) for term in terms)
                    hits.append({'page': key, 'section': item['id'], 'title': item['title'],
                                 'offset': max(0, position - 80), 'score': score,
                                 'excerpt': text[max(0, position - 80):position + 260]})
        hits.sort(key=lambda hit: (-hit['score'], hit['page'], hit['section']))
        # The limit bounds encoded search results as well as individual page text.
        shown = []
        for hit in hits:
            if len(json.dumps([*shown, hit], ensure_ascii=False)) > limit:
                break
            shown.append(hit)
            if len(shown) == 12:
                break
        return {**result, 'query': query, 'matches': shown, 'total_matches': len(hits)}
    return {**result, 'pages': doc_index()}


def mode_facts(entry: dict[str, Any]) -> dict[str, Any]:
    return {key: entry.get(key) for key in ('id', 'description', 'scored', 'parameters', 'nodes', 'edges', 'ports', 'guidance', 'gates')}


def mode_help(entry: dict[str, Any], authored: dict[str, Any] | None = None) -> dict[str, Any]:
    authored = authored if authored is not None else coverage().get('modes', {}).get(entry['id'], {})
    warnings = []
    if not authored.get('purpose') or not authored.get('examples'):
        warnings.append(f"{entry['id']}: missing usage guidance/examples; CEO can inspect its schema but cannot reliably choose inputs. Add a modes entry in data/coverage.json and document a worked example.")
    if authored.get('reviewed_schema') != digest(mode_facts(entry)):
        warnings.append(f"{entry['id']}: guidance has not been reviewed for this live schema; review examples and update reviewed_schema to {digest(mode_facts(entry))}.")
    return {**entry, 'usage': authored, 'coverage_warnings': warnings}


def help_contract(project: Path, topic: str = '') -> dict[str, Any]:
    from refactory_dsh.console import catalogue, level_two
    from refactory_dsh.worker_config import effective
    from refactory_dsh.workers import BACKENDS, capabilities
    authored = coverage()
    modes = catalogue(project=project)
    ops = operations()
    for name, entry in ops.items():
        entry.update(authored.get('operations', {}).get(name, {}))
        entry['schema_id'] = digest(operations_schema(entry))
    if topic:
        if topic in ops:
            return {**identity(), 'operation': topic, **ops[topic]}
        selected = next((mode for mode in modes if mode['id'] == topic), None)
        if selected is not None:
            return {**identity(), 'mode': selected}
        if topic == 'workers':
            return {**identity(), 'configuration': effective(project)}
        raise ValueError('Unknown help topic; use help without topic for live operation and mode IDs')
    return {**identity(), 'operating_guide': bundle()['operating_guide'],
            'operations': {key: {field: value.get(field) for field in ('purpose', 'interfaces', 'doc')}
                           for key, value in ops.items()},
            'modes': [{key: mode.get(key) for key in ('id', 'description', 'scored', 'action', 'interactive', 'coverage_warnings')}
                      for mode in modes],
            'strategies': level_two(), 'worker_capabilities': {backend: capabilities(backend) for backend in BACKENDS},
            'documentation': {'index': 'docs', 'search': {'action': 'docs', 'query': 'worker routing'},
                              'read': {'action': 'docs', 'page': 'refactory/docs/ceo-guide.md'}},
            'next': 'Read help with topic=<operation or mode ID> for full arguments/examples and docs for bounded current-build pages.'}


def operations_schema(entry: dict[str, Any]) -> dict[str, Any]:
    return {key: entry[key] for key in ('description', 'parameters', 'effective_defaults', 'configuration_schema') if key in entry}
