"""Owner-only SIGTERM must unwind a real Mode.run fork before exiting."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


@pytest.mark.skipif(os.name != "posix", reason="POSIX SIGTERM/process sessions")
def test_mode_owner_sigterm_cancels_fork_and_settles_detached_runtime_tools(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    tool = tmp_path / "owned_tool.py"
    tool.write_text("""import os,signal,sys,time
from pathlib import Path
tag=sys.argv[1]
def stop(*args):
    Path(tag+'.tool-stopped').write_text('owned tool settled')
    raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
Path(tag+'.tool-ready').write_text(str(os.getpid()))
time.sleep(30)
""")
    runtime = tmp_path / "external_runtime.py"
    runtime.write_text(f"#!{sys.executable}\n" + """import json,os,signal,subprocess,sys,time
from pathlib import Path
receipt=Path(os.environ['REFACTORY_WORKER_RECEIPT'])
record=json.loads(receipt.read_text())
record['actual_route']={**record['requested_route'],'source':'controlled-external-process'}
receipt.write_text(json.dumps(record))
tag=record['node_id']
""" + f"tool_script={str(tool)!r}\n" + """
managed=subprocess.Popen([sys.executable,tool_script,tag],start_new_session=True)
def stop(*args):
    managed.terminate()
    managed.wait(timeout=3)
    if not Path(tag+'.tool-stopped').exists(): raise RuntimeError('owned tool did not settle')
    Path(tag+'.runtime-stopped').write_text('runtime and owned tool settled')
    raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
deadline=time.monotonic()+15
while not Path(tag+'.tool-ready').exists():
    if time.monotonic()>deadline: raise RuntimeError('tool startup failed')
    time.sleep(.01)
Path(tag+'.runtime-ready').write_text(json.dumps({'runtime_pid':os.getpid(),'tool_pid':managed.pid}))
while time.monotonic()<deadline and not Path('fixture-release').exists(): time.sleep(.02)
stop()
""")
    runtime.chmod(0o700)
    driver = tmp_path / "mode_owner.py"
    driver.write_text("""import json,sys
from pathlib import Path
from refactory_dsh.model import DshModelClient
from refactory_dsh.nn import Graph,Node,Mode
class ForkMode(Mode):
    def build(self):
        graph=Graph()
        graph.add(Node('fork','fork',targets=('left','right')))
        for name in ('left','right'):
            graph.add(Node(name,'agent',prompt_template='controlled external task',writes=(name+'.md',),timeout_s=60))
        graph.add(Node('join','join',sources=('left','right')))
        graph.connect('fork','join')
        graph.connect('left','join')
        graph.connect('right','join')
        return graph
root=Path(sys.argv[1])
client=DshModelClient(dsh_cli=sys.argv[2],provider='controlled',model='controlled',effort='high',attempts=1)
try:
    ForkMode().run(root,model=client)
except SystemExit as error:
    workers=[json.loads(path.read_text()) for path in (root/'.refactory/workers').glob('*.json')]
    (root/'owner-unwound.json').write_text(json.dumps({'exit_code':error.code,'states':[w['state'] for w in workers]}))
    raise
""")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]), "PYTHONDONTWRITEBYTECODE": "1"}
    owner = subprocess.Popen([sys.executable, "-B", str(driver), str(run), str(runtime)], cwd=run,
                             env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 8
        while len(list(run.glob("*.runtime-ready"))) < 2:
            if owner.poll() is not None or time.monotonic() > deadline:
                stdout, stderr = owner.communicate(timeout=2)
                pytest.fail(f"fork did not dispatch two runtimes: {owner.returncode} {stdout} {stderr}")
            time.sleep(.02)
        # Only the Python owner receives TERM. Its detached runtimes own their
        # detached tools; their actual cancellation protocol must settle both.
        owner.terminate()
        stdout, stderr = owner.communicate(timeout=8)
        assert owner.returncode == 143, (stdout, stderr)
        unwind = json.loads((run / "owner-unwound.json").read_text())
        assert unwind == {"exit_code": 143, "states": ["cancelled", "cancelled"]}
        assert len(list(run.glob("*.runtime-stopped"))) == 2
        assert len(list(run.glob("*.tool-stopped"))) == 2
        progress = json.loads((run / ".refactory/progress.json").read_text())
        assert progress["node"] == "fork"
        assert {branch["status"] for branch in progress["branches"].values()} == {"cancelled"}
    finally:
        (run / "fixture-release").write_text("release controlled external I/O")
        if owner.poll() is None:
            owner.terminate()
            owner.communicate(timeout=8)
        deadline = time.monotonic() + 4
        while len(list(run.glob("*.runtime-stopped"))) < len(list(run.glob("*.runtime-ready"))) and time.monotonic() < deadline:
            time.sleep(.02)
