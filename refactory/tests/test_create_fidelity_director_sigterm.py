"""External SIGTERM crosses a real director's detached subprocess ownership."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from test_create_fidelity_directors import plan_for
from refactory_dsh.create_fidelity_directors import RESEARCH


@pytest.mark.skipif(os.name != "posix", reason="POSIX SIGTERM/process sessions")
def test_external_sigterm_unwinds_director_owner_and_settles_runtime_owned_tools(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for name in RESEARCH.reads:
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Concrete project context and original booking requirements.\n")
    (run / "controlled-plan.json").write_text(json.dumps(plan_for(RESEARCH, 3)))
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
tag=record['node_id'].split(':')[2]
if tag=='plan':
    print(Path('controlled-plan.json').read_text())
    raise SystemExit(0)
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
    driver = tmp_path / "director_owner.py"
    driver.write_text("""import json,sys
from pathlib import Path
from refactory_dsh.model import DshModelClient
from refactory_dsh.nn import Graph,Node
from refactory_dsh.nn.executor import Executor,RunContext
root=Path(sys.argv[1])
client=DshModelClient(dsh_cli=sys.argv[2],provider='controlled',model='controlled',effort='high',attempts=1)
graph=Graph()
graph.add(Node('research_director','op',call='refactory_dsh.create_fidelity_directors:research_director',writes=('.factory/',),timeout_s=60))
try:
    Executor(graph,RunContext(root,model=client)).run()
except SystemExit as error:
    workers=[json.loads(path.read_text()) for path in (root/'.refactory/workers').glob('*.json')]
    (root/'owner-unwound.json').write_text(json.dumps({'exit_code':error.code,'states':[w['state'] for w in workers]}))
    raise
""")
    source = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1"}
    owner = subprocess.Popen([sys.executable, "-B", str(driver), str(run), str(runtime)], cwd=run,
                             env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 8
        while len(list(run.glob("child-*.runtime-ready"))) < 3:
            if owner.poll() is not None or time.monotonic() > deadline:
                stdout, stderr = owner.communicate(timeout=2)
                pytest.fail(f"director did not dispatch three real runtimes: {owner.returncode} {stdout} {stderr}")
            time.sleep(.02)
        # The target is only the owner. Runtimes and tools have separate sessions;
        # their shutdown must come from the owner's actual cancellation protocol.
        owner.terminate()
        stdout, stderr = owner.communicate(timeout=8)
        assert owner.returncode == 143, (stdout, stderr)
        unwind = json.loads((run / "owner-unwound.json").read_text())
        assert unwind["exit_code"] == 143
        assert unwind["states"].count("cancelled") == 3
        assert unwind["states"].count("completed") == 1  # the actual planner
        assert len(list(run.glob("child-*.runtime-stopped"))) == 3
        assert len(list(run.glob("child-*.tool-stopped"))) == 3
        findings = json.loads((run / ".factory/research_director-findings.json").read_text())
        assert findings["status"] == "failed"
    finally:
        # Fixture runtimes own and settle their own tools even on an assertion
        # failure; cleanup never discovers or signals arbitrary host PIDs.
        (run / "fixture-release").write_text("release controlled external I/O")
        if owner.poll() is None:
            owner.terminate()
            owner.communicate(timeout=8)
