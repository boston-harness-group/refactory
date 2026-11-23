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
