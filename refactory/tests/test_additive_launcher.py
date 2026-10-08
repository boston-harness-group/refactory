"""Launch Refactory over operator-owned web profiles without replacing their UI."""
from __future__ import annotations

import http.cookiejar
import json
import os
import re
import selectors
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

from refactory_dsh.paths import REPO_ROOT
from refactory_dsh.owned_process import shutdown


@pytest.fixture
def additive_home(tmp_path, monkeypatch):
    import shutil
    missing = [tool for tool in ("node", "pnpm") if not shutil.which(tool)]
    if not (REPO_ROOT / "node_modules/.bin/dsh").exists():
        missing.append("pinned harness")
    if missing:
        reason = "additive launch coverage needs " + ", ".join(missing) + "; run ./install.sh"
        if os.environ.get("REFACTORY_REQUIRE_MIGRATION_TESTS") == "1":
            pytest.fail(reason)
        pytest.skip(reason)
    home = tmp_path / "harness"
    user_home = tmp_path / "user"
    (user_home / ".dsh").mkdir(parents=True)
    (user_home / ".dsh/settings.yaml").write_text("agent-default-model: {}\n")
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setenv("DSH_HOME", str(home))
    monkeypatch.delenv("DSH_ROOT", raising=False)
    monkeypatch.delenv("REFACTORY_MODEL_DSH_HOME", raising=False)
    return home, user_home


def rpc(opener, base, method, args):
    payload = {"type": "client-request", "rpcId": method, "method": method,
               "payload": {"args": args}}
    request = urllib.request.Request(base + "/api/" + method,
                                    data=json.dumps(payload).encode(),
                                    headers={"Content-Type": "application/json"})
    result = json.loads(opener.open(request, timeout=20).read())["result"]
    assert result["ok"], result
    return result["value"]


def test_additive_composes_web_and_refactory_without_copying_web(additive_home):
    home, user_home = additive_home
    profile = "web"
    initialize = [str(REPO_ROOT / "node_modules/.bin/dsh"), "--profile", profile]
    subprocess.run([*initialize, "--dump-config"],
                   check=True, capture_output=True, text=True)
    directory = home / "profiles" / profile
    manifest = directory / "package.json"
    data = json.loads(manifest.read_text())
    data["operatorMetadata"] = {"keep": True}
    manifest.write_text(json.dumps(data, indent=2) + "\n")
    patch = directory / "cordis.patch.yml"
    patch.write_text("""- id: agent-preset-registry
  config:
    default: operator
- insert:
    - id: preset-operator
      name: '@deepseek-ai/dsh-agent-preset'
      config:
        id: operator
        name: Operator
        plugins: []
""")
    before = manifest.read_bytes(), patch.read_bytes()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = [str(REPO_ROOT / "refactory-dsh"), "--additive",
               "--port", str(port), "--host", "127.0.0.1", "--no-open"]
    process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True)
    log = ""
    try:
        deadline = time.monotonic() + 60
        with selectors.DefaultSelector() as ready:
            ready.register(process.stdout, selectors.EVENT_READ)
            token = None
            while time.monotonic() < deadline and token is None:
                if ready.select(timeout=max(0, deadline - time.monotonic())):
                    line = process.stdout.readline()
                    log += line
                    match = re.search(r"[?&]token=([A-Za-z0-9_-]{43})", line)
                    if match:
                        token = match[1]
                    if not line and process.poll() is not None:
                        break
        assert token, re.sub(r"token=[A-Za-z0-9_-]+", "token=[redacted]", log)
        base = f"http://127.0.0.1:{port}"
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        page = opener.open(base + "/?token=" + token, timeout=20).read().decode()
        presets = rpc(opener, base, "agentPresets/list", {})["presets"]
        assert {"standard", "ptc", "minimal", "cordis", "ceo"} <= {row["id"] for row in presets}
        assert [row["id"] for row in presets if row.get("isDefault")] == ["ceo"]
        assert all(not row.get("broken") for row in presets)
        assert "@deepseek-ai/dsh-client-ui-refactory" in page
        assert "@deepseek-ai/dsh-client-ui-brand-refactory" not in page
        assert "@deepseek-ai/dsh-client-ui-brand-official" in page
        assert rpc(opener, base, "refactoryConsole/project", {}) == str(REPO_ROOT)
        from refactory_dsh.console import catalogue
        modes = rpc(opener, base, "refactoryConsole/modes", {})
        assert {row["id"] for row in modes} == {row["id"] for row in catalogue(project=REPO_ROOT)}
        created = rpc(opener, base, "session/create", {"request": {"cwd": str(user_home)}})
        assert created["agentPreset"] == "ceo"
        ceo = rpc(opener, base, "session/create", {"request": {"cwd": str(user_home), "agentPreset": "ceo"}})
        assert ceo["agentPreset"] == "ceo"
        assert (home / "profiles/graph/package.json").exists(), "runtime profile must be provisioned"
        assert not (home / "profiles/refactory").exists(), "additive launch should not create the standalone GUI"
        additive = home / "profiles/refactory-web"
        composition = json.loads((additive / "package.json").read_text())
        assert composition["dsh"]["profile"]["bundles"] == [
            "@deepseek-ai/dsh-base", "@deepseek-ai/dsh-web-app",
            "@deepseek-ai/dsh-refactory", "@deepseek-ai/dsh-refactory-web",
        ]
        assert composition["dsh"]["profile"]["refactoryAdditive"] is True
        assert "operatorMetadata" not in composition
        assert "operator" not in {row["id"] for row in presets}, "the web profile must not be cloned"
        assert "file://" not in (additive / "cordis.patch.yml").read_text()
        additive_before = (additive / "cordis.patch.yml").read_bytes()
        # Repeated provisioning keeps operator changes in the additive profile.
        (additive / "operator-note.txt").write_text("keep me")
        again = subprocess.run(["node", str(REPO_ROOT / "refactory/refactory_dsh/additive.mjs"),
                                str(REPO_ROOT)], check=True, capture_output=True, text=True)
        assert again.stdout == ""
        assert (additive / "cordis.patch.yml").read_bytes() == additive_before
        assert (additive / "operator-note.txt").read_text() == "keep me"
        assert manifest.read_bytes() == before[0]
        assert patch.read_bytes() == before[1]
        assert (user_home / ".dsh/settings.yaml").read_text() == "agent-default-model: {}\n"
    finally:
        shutdown(process)


def test_additive_refuses_a_non_web_profile(additive_home):
    home, _ = additive_home
    target = home / "profiles/refactory-web"
    target.mkdir(parents=True)
    manifest = target / "package.json"
    manifest.write_text(json.dumps({"dsh": {"profile": {"bundles": ["@deepseek-ai/dsh-base"]}}}))
    before = manifest.read_bytes()
    result = subprocess.run(["node", str(REPO_ROOT / "refactory/refactory_dsh/additive.mjs"),
                             str(REPO_ROOT)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "must be a web-based profile" in result.stderr
    assert manifest.read_bytes() == before
