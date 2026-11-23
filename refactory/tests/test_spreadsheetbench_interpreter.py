"""Building the pinned interpreter (`interpreter.provision`) with a stub ``uv``: a tree
whose packages do not import is built again, and builds that run at once each use a
directory of their own."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from refactory_packages.spreadsheetbench import interpreter

pytestmark = pytest.mark.skipif(os.name == "nt", reason="the stub uv and python are shell scripts")


def _reply(imports: bool) -> str:
    """What the stub python prints for `interpreter.check`'s probe."""
    imported = {name: interpreter.PINS[name] if imports else "AttributeError: no __version__"
                for name in interpreter.NAMED}
    return json.dumps({"python": interpreter.PYTHON + ".1", "imported": imported, **interpreter.PINS})


def _stub_uv(tmp_path: Path) -> Path:
    """A ``uv`` whose ``venv`` writes a python that answers the probe, importing its
    packages only once ``pip install`` has written them (a marker); slow enough that
    builds started together overlap, and failing, as uv does, when its environment was
    removed or written by another build meanwhile."""
    bin_dir = tmp_path / "stub-bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(f"""#!/bin/sh
case "$1" in
venv) dir="$5"; mkdir -p "$dir/bin" "$dir/lib"
      cat > "$dir/bin/python" <<'PY'
#!/bin/sh
here=$(dirname "$0")
if [ -f "$here/../lib/pandas/__init__.py" ]; then echo '{_reply(True)}'; else echo '{_reply(False)}'; fi
PY
      chmod +x "$dir/bin/python" ;;
pip) sleep 0.5; [ -x "$5" ] || {{ echo "no interpreter at $5" >&2; exit 2; }}
     mkdir "$(dirname "$5")/../lib/pandas" && : > "$(dirname "$5")/../lib/pandas/__init__.py" ;;
esac
""")
    uv.chmod(0o755)
    return bin_dir


@pytest.fixture
def stub_env(tmp_path, monkeypatch):
    bin_dir = _stub_uv(tmp_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    interpreter.check.cache_clear()
    yield interpreter.default_home()
    interpreter.check.cache_clear()


def test_a_tree_whose_packages_do_not_import_is_built_again(stub_env):
    """The tree a cut-short move leaves: the interpreter and the metadata, not pandas's
    files. It is refused by `check` and replaced, not handed to model code."""
    home = stub_env
    (home / "bin").mkdir(parents=True)
    (home / "bin" / "python").write_text(f"#!/bin/sh\necho '{_reply(False)}'\n")
    (home / "bin" / "python").chmod(0o755)
    with pytest.raises(interpreter.InterpreterError, match="pandas imports as AttributeError"):
        interpreter.check(str(home / "bin" / "python"))
    assert interpreter.provision() == home / "bin" / "python"
    assert (home / "lib" / "pandas" / "__init__.py").is_file()
    assert sorted(path.name for path in home.parent.iterdir() if not path.name.startswith(".")) \
        == [home.name], "nothing of the broken tree or the build is left beside it"


def test_builds_started_at_once_each_use_a_directory_of_their_own(stub_env):
    """Six processes provision at once, as test workers do: each gets the one built
    interpreter, none fails on another's build directory."""
    code = ("from refactory_packages.spreadsheetbench import interpreter as i; "
            "print(i.provision())")
    procs = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) for _ in range(6)]
    results = [proc.communicate(timeout=120) + (proc.returncode,) for proc in procs]
    assert all(code == 0 for _out, _err, code in results), [err for _out, err, _code in results]
    assert {out.strip() for out, _err, _code in results} == {str(stub_env / "bin" / "python")}
    assert sorted(path.name for path in stub_env.parent.iterdir() if not path.name.startswith(".")) \
        == [stub_env.name]


def _tree(home: Path, python_script: str) -> None:
    (home / "bin").mkdir(parents=True)
    (home / "bin" / "python").write_text(python_script)
    (home / "bin" / "python").chmod(0o755)
    (home / "lib" / "pandas").mkdir(parents=True)
    (home / "lib" / "pandas" / "__init__.py").write_text("")


@pytest.mark.parametrize("why", ["times out", "cannot start"])
def test_a_probe_that_says_nothing_replaces_nothing(stub_env, monkeypatch, why):
    """On a loaded machine the probe of a sound tree can time out, and another process
    may be running code from that tree: the error is raised, and nothing is moved,
    built or removed."""
    home = stub_env
    _tree(home, "#!/bin/sh\nsleep 5\n")
    if why == "cannot start":
        (home / "bin" / "python").chmod(0o644)
    monkeypatch.setattr(interpreter, "PROBE_TIMEOUT_S", 0.5)
    with pytest.raises(interpreter.InterpreterError) as raised:
        interpreter.provision()
    assert not isinstance(raised.value, interpreter.InterpreterMismatch)
    assert (home / "lib" / "pandas" / "__init__.py").is_file()
    assert sorted(path.name for path in home.parent.iterdir() if not path.name.startswith(".")) \
        == [home.name], "no build was started and nothing was set aside"


def test_a_killed_build_s_directory_is_removed_by_the_next_build(stub_env):
    """A build killed before its cleanup (SIGKILL, out of memory, a CI timeout) leaves its
    directory: the next build on the host removes it when its pid is gone, and leaves
    one whose pid runs, or that another host made."""
    home = stub_env
    home.parent.mkdir(parents=True)
    gone = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"],
                          capture_output=True, text=True, check=True).stdout.strip()
    host = interpreter._host()
    dead = [home.with_name(f"{home.name}.partial-{host}-{gone}-aaaa1111"),
            home.with_name(f"{home.name}.stale-{host}-{gone}-bbbb2222")]
    alive = [home.with_name(f"{home.name}.partial-{host}-{os.getpid()}-cccc3333"),
             home.with_name(f"{home.name}.partial-otherhost-{gone}-dddd4444")]
    for path in dead + alive:
        (path / "lib").mkdir(parents=True)
    assert interpreter.provision() == home / "bin" / "python"
    assert [path.exists() for path in dead] == [False, False]
    assert [path.exists() for path in alive] == [True, True]
