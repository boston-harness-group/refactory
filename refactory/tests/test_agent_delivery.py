from __future__ import annotations

import json
from pathlib import Path
import sys

from refactory_dsh import agent_delivery, nn
from refactory_dsh.model import DshModelClient
from refactory_dsh.steering import Ledger


def test_contract_keeps_task_steering_and_exact_paths_without_exposing_settings(tmp_path):
    Ledger.seed(tmp_path, 'Preserve the public API and run the requested check.')
    task = 'Write result.json after checking input.py.\n'
    context = {'python_bin': str(Path(sys.executable).parent), 'runtime_root': '/selected/runtime',
               'api_key': 'never-in-a-prompt', 'permission_mode': 'skip'}
    first = agent_delivery.begin(tmp_path, 'check', task, ['result.json'], reads=['input.py'], model_context=context)
    prompt = first['prompt']
    assert task in prompt and 'Preserve the public API' in prompt
    assert 'never-in-a-prompt' not in prompt and 'permission_mode' not in prompt
    assert first['execution'] == {
        'node': 'check', 'workdir': str(tmp_path), 'inputs': [str(tmp_path / 'input.py')],
        'input_fingerprints': {'input.py': None},
        'outputs': [str(tmp_path / 'result.json')], 'python': str(Path(sys.executable).parent / 'python'),
        'refactory_source': str(Path(agent_delivery.__file__).parent), 'runtime_root': '/selected/runtime',
    }
    repeated = agent_delivery.begin(tmp_path, 'check', task, ['result.json'], reads=['input.py'], model_context=context, reuse=True)
    assert repeated == first
    assert (tmp_path / first['contexts'][0]['prompt']).read_text() == prompt
    assert json.loads((tmp_path / first['receipt']).read_text())['execution'] == first['execution']


def test_brief_final_response_does_not_replace_delivered_artifact(tmp_path):
    artifact = '{"answer":42}\n'
    first = agent_delivery.begin(tmp_path, 'build', 'Write result.json.', ['result.json'])
    (tmp_path / 'result.json').write_text(artifact)
    receipt = agent_delivery.finish(tmp_path, 'build', final_response='Done.', capture_final=True)
    assert receipt['status'] == 'delivered' and receipt['captured_final'] is None
    assert (tmp_path / 'result.json').read_text() == artifact
    assert (tmp_path / receipt['final_response']).read_text() == 'Done.'
    assert receipt['execution'] == first['execution']


def test_python_and_native_worker_dispatch_share_execution_contract(tmp_path, monkeypatch):
    from refactory_dsh.worker_node import execute

    seen = []
    def complete(self, prompt, *, system='', workdir=None):
        seen.append(prompt)
        (workdir / 'answer.txt').write_text('42')
        return 'Done.'
    monkeypatch.setattr(DshModelClient, 'complete', complete)
    client = DshModelClient(python_bin=Path(sys.executable).parent)

    class OneNode(nn.Mode):
        name = 'contract'
        def build(self):
            graph = nn.Graph()
            graph.add(nn.Node('build', 'agent', prompt_template='Read input.txt; write answer.txt.',
                              reads=('input.txt',), writes=('answer.txt',)))
            return graph

    contexts = []
    for backend in ('python', 'native'):
        workdir = tmp_path / backend
        workdir.mkdir()
        (workdir / 'input.txt').write_text('42')
        if backend == 'python':
            result = OneNode().run(workdir, model=client)
            assert result.exit_ok, result.error
        else:
            execute({'schema_version': 1, 'node_id': 'build', 'reads': ['input.txt'],
                     'writes': ['answer.txt'], 'produces': [], 'prompt': 'Read input.txt; write answer.txt.',
                     'model_context': client.context()}, workdir)
        receipt = agent_delivery.receipts_in(workdir)['build']
        context = receipt['execution']
        assert context['workdir'] == str(workdir)
        assert context['inputs'] == [str(workdir / 'input.txt')]
        assert context['outputs'] == [str(workdir / 'answer.txt')]
        assert seen[-1].endswith('Read input.txt; write answer.txt.')
        contexts.append({key: value for key, value in context.items() if key not in ('workdir', 'inputs', 'outputs')})
    assert contexts[0] == contexts[1]
