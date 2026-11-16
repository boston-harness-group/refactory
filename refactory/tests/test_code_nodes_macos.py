"""Code steps on macOS: the Seatbelt sandbox (`snippet._seatbelt_profile`) holds them in.

Each escape a model-written code step might try is attempted against the real sandbox on
this Mac and must fail: the network, files outside its scratch directory (a fake secret
planted in a temporary home, the run directory, the person's other files), writing
anywhere else, starting or signalling another process, a fork bomb, a memory blow-up and
running past the wall clock. Where `sandbox-exec` or the profile cannot run, code steps
are refused, never run unsandboxed. These run only on macOS; Linux's namespace sandbox has
its own tests in test_code_nodes.py.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from refactory_dsh import snippet

def test_code_steps_run_on_this_mac_where_ci_requires_them():
    """The macOS CI job sets REFACTORY_REQUIRE_CODE_SANDBOX, so a sandbox that stopped
    loading fails the job instead of skipping every test below."""
    if sys.platform != "darwin" or not os.environ.get("REFACTORY_REQUIRE_CODE_SANDBOX"):
        pytest.skip("only where the macOS CI job requires code steps")
    assert snippet.available() == (True, "")


needs_seatbelt = pytest.mark.skipif(
    sys.platform != "darwin" or not snippet.available()[0],
    reason="the Seatbelt sandbox is macOS's" if sys.platform != "darwin"
    else f"no code-node sandbox here: {snippet.available()[1]}")


def _run(code: str, facts: dict | None = None, *, timeout_s: float = 10, extra: tuple = ()):
    return snippet._sandboxed(code, {"files": {}, "facts": facts or {}}, timeout_s=timeout_s,
                              extra=extra)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A temporary home with fake secrets where real ones live, as the person's HOME."""
    home = tmp_path / "home"
    for relative, text in ((".ssh/id_rsa", "FAKE-SSH-KEY"), (".aws/credentials", "FAKE-AWS"),
                           ("Library/Keychains/login.keychain-db", "FAKE-KEYCHAIN"),
                           ("code/other-repo/.env", "FAKE-TOKEN")):
        (home / relative).parent.mkdir(parents=True, exist_ok=True)
        (home / relative).write_text(text)
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.mark.parametrize("relative", [".ssh/id_rsa", ".aws/credentials",
                                      "Library/Keychains/login.keychain-db",
                                      "code/other-repo/.env"])
@needs_seatbelt
def test_a_planted_secret_in_the_home_cannot_be_read_listed_or_even_seen(home, relative):
    secret = str(home / relative)
    for code in ("def main(inp):\n    return open(inp['facts']['p']).read()\n",
                 "import os\ndef main(inp):\n    return os.listdir(os.path.dirname(inp['facts']['p']))\n",
                 "import os\ndef main(inp):\n    return os.stat(inp['facts']['p']).st_size\n"):
        with pytest.raises(snippet.SnippetError, match="PermissionError"):
            _run(code, {"p": secret})


@needs_seatbelt
def test_the_code_sees_neither_the_persons_home_nor_their_environment(home, monkeypatch):
    monkeypatch.setenv("SOME_API_KEY", "k")
    seen = _run("import os\ndef main(inp):\n    return {'home': os.environ['HOME'], "
                "'env': sorted(os.environ), 'cwd': os.getcwd()}\n")
    assert seen["env"] == ["HOME", "PATH", "TMPDIR"]
    assert seen["home"] == seen["cwd"] and str(home) not in seen["home"]


@pytest.mark.parametrize("code", [
    "import socket\ndef main(inp):\n    socket.create_connection(('1.1.1.1', 80), timeout=3)\n",
    "import socket\ndef main(inp):\n    return socket.gethostbyname('example.com')\n",
    "import socket\ndef main(inp):\n    s = socket.socket(socket.AF_UNIX)\n    s.connect('/var/run/mDNSResponder')\n",
], ids=["tcp", "dns", "unix socket"])
@needs_seatbelt
def test_no_network_of_any_kind(code):
    with pytest.raises(snippet.SnippetError, match="the code raised (PermissionError|gaierror)"):
        _run(code)


@needs_seatbelt
def test_it_writes_only_its_own_scratch_directory_which_goes_with_it(tmp_path):
    outside = tmp_path / "escape.txt"
    with pytest.raises(snippet.SnippetError, match="PermissionError"):
        _run("def main(inp):\n    open(inp['facts']['p'], 'w').write('x')\n", {"p": str(outside)})
    assert not outside.exists()
    for code in ("def main(inp):\n    open('/tmp/x', 'w').write('x')\n",
                 "def main(inp):\n    open('/etc/x', 'w').write('x')\n"):
        with pytest.raises(snippet.SnippetError, match="PermissionError"):
            _run(code)
    scratch = _run("import os\ndef main(inp):\n    open('made.txt', 'w').write('x')\n"
                   "    return [os.getcwd(), os.listdir('.')]\n")
    assert scratch[1] == ["made.txt"] and not os.path.exists(scratch[0])


@needs_seatbelt
def test_the_run_directory_and_the_project_are_not_readable(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "answers.json").write_text("the answer")
    with pytest.raises(snippet.SnippetError, match="PermissionError"):
        _run("def main(inp):\n    return open(inp['facts']['p']).read()\n",
             {"p": str(run / "answers.json")}, extra=(str(run),))


@pytest.mark.parametrize("code", [
    "import os\ndef main(inp):\n    n = 0\n    while True:\n        os.fork()\n        n += 1\n",
    "import subprocess\ndef main(inp):\n    subprocess.Popen(['/bin/sleep', '60'])\n",
    "import os, signal\ndef main(inp):\n    os.kill(os.getppid(), signal.SIGTERM)\n",
    "import os\ndef main(inp):\n    os.kill(1, 0)\n",
], ids=["fork bomb", "exec", "signal the parent", "probe another process"])
@needs_seatbelt
def test_no_other_process_is_started_signalled_or_looked_at(code):
    with pytest.raises(snippet.SnippetError, match="PermissionError"):
        _run(code)


@needs_seatbelt
def test_a_memory_blow_up_is_killed_at_the_cap():
    with pytest.raises(snippet.SnippetError, match=f"over its {snippet.MEMORY_MB} MB memory limit"):
        _run("def main(inp):\n    b = b'x' * (4 * 1024 ** 3)\n    return len(b)\n")


@needs_seatbelt
def test_a_file_over_the_size_cap_is_refused():
    with pytest.raises(snippet.SnippetError, match="File too large"):
        _run("def main(inp):\n    open('big', 'wb').write(b'x' * (snippet_mb << 20))\n"
             .replace("snippet_mb", str(snippet.FILE_MB + 1)))


@needs_seatbelt
@pytest.mark.parametrize("code", [
    "def main(inp):\n    for i in range(20):\n        open(f'f{i}', 'wb').write(b'x' * 1000000)\n"
    "    while True:\n        pass\n",
    "def main(inp):\n    i = 0\n    while True:\n        open(f'e{i}', 'w').close()\n        i += 1\n",
], ids=["bytes", "entries"])
def test_the_scratch_directory_is_held_to_linux_s_tmpfs_size(code):
    with pytest.raises(snippet.SnippetError, match="filled its scratch directory"):
        _run(code)


@needs_seatbelt
def test_a_link_in_the_scratch_directory_does_not_reach_outside_it(home):
    with pytest.raises(snippet.SnippetError, match="PermissionError"):
        _run("import os\ndef main(inp):\n    os.symlink(inp['facts']['p'], 'key')\n"
             "    return open('key').read()\n", {"p": str(home / ".ssh/id_rsa")})


@needs_seatbelt
def test_code_past_its_wall_clock_is_stopped_with_nothing_left_running():
    pids = []
    original = subprocess.Popen

    def recording(*args, **kwargs):
        proc = original(*args, **kwargs)
        pids.append(proc.pid)
        return proc

    try:
        subprocess.Popen = recording  # type: ignore[misc]
        with pytest.raises(snippet.SnippetError, match="over its 2 s limit"):
            _run("def main(inp):\n    while True:\n        pass\n", timeout_s=2)
    finally:
        subprocess.Popen = original  # type: ignore[misc]
    assert len(pids) == 1, "one process per code step"
    with pytest.raises(ProcessLookupError):
        os.kill(pids[0], 0)


# ── failing closed ─────────────────────────────────────────────────────────


@needs_seatbelt
def test_without_sandbox_exec_code_steps_are_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(snippet, "SANDBOX_EXEC", str(tmp_path / "no-sandbox-exec"))
    ok, why = snippet._check_sandbox()
    assert not ok and "no-sandbox-exec" in why


@needs_seatbelt
def test_a_profile_that_does_not_load_never_runs_the_code_unsandboxed(monkeypatch, tmp_path):
    marker = tmp_path / "ran.txt"
    monkeypatch.setattr(snippet, "_seatbelt_profile", lambda *args: "(version 1)\n(this is not sbpl")
    with pytest.raises(snippet.SnippetError):
        _run("def main(inp):\n    open(inp['facts']['p'], 'w').write('x')\n", {"p": str(marker)})
    assert not marker.exists()
    ok, _why = snippet._check_sandbox()
    assert not ok


@needs_seatbelt
def test_an_interpreter_installed_around_the_home_directory_is_refused(monkeypatch):
    snippet._mac_interpreter.cache_clear()
    monkeypatch.setattr(snippet.sys, "base_prefix", "/")
    try:
        with pytest.raises(snippet.SnippetError, match="holds the home directory"):
            snippet._mac_interpreter()
    finally:
        snippet._mac_interpreter.cache_clear()


@needs_seatbelt
def test_a_path_a_profile_literal_cannot_hold_is_refused():
    for path in ('/tmp/a"b', "/tmp/a\\b", "relative", "/tmp/new\nline"):
        with pytest.raises(snippet.SnippetError, match="cannot name the path"):
            snippet._sbpl(path)


@needs_seatbelt
def test_a_directory_to_hide_inside_what_the_code_may_read_is_refused():
    _python, prefix = snippet._mac_interpreter()
    with pytest.raises(snippet.SnippetError, match="cannot be hidden"):
        _run("def main(inp):\n    return 1\n", extra=(os.path.join(prefix, "lib"),))
