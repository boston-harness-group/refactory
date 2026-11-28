"""The split_module script reproduces the loop package byte-for-byte from the
original monolithic loop.py, and rejects unmapped names."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

MAPPING = Path(__file__).resolve().parent.parent / "refactory_dsh" / "tools" / "splits" / "loop.json"
COMMITTED = Path(__file__).resolve().parent.parent / "refactory_dsh" / "loop"
BASE_REV = "15914e84a709ec796e8844614cb57aa8af5512e3"  # the commit loop.py was split from (pinned: main no longer has loop.py)
HAND_WRITTEN = {"__main__.py"}


def _original_source() -> str:
    return subprocess.check_output(
        ["git", "show", f"{BASE_REV}:refactory/refactory_dsh/loop.py"],
        text=True,
    )


def test_split_reproduces_committed_modules(tmp_path):
    source_file = tmp_path / "loop.py"
    source_file.write_text(_original_source())
    out = tmp_path / "loop_pkg"
    out.mkdir()

    from refactory_dsh.tools.split_module import split_module

    split_module(source_file, MAPPING, out)

    generated = {f.name for f in out.glob("*.py")}
    committed = {f.name for f in COMMITTED.glob("*.py") if f.name not in HAND_WRITTEN}
    assert generated == committed, f"file set mismatch: {generated ^ committed}"

    for name in sorted(generated):
        expected = (COMMITTED / name).read_text()
        actual = (out / name).read_text()
        assert actual == expected, f"{name} differs"


def test_unmapped_method_fails(tmp_path):
    import ast

    source = _original_source()
    tree = ast.parse(source)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Loop":
            insert_line = node.end_lineno
            break
    lines = source.splitlines()
    lines.insert(insert_line, "    def _brand_new_method(self): ...")
    source_file = tmp_path / "loop.py"
    source_file.write_text("\n".join(lines))
    out = tmp_path / "loop_pkg"
    out.mkdir()

    from refactory_dsh.tools.split_module import split_module

    with pytest.raises(SystemExit, match="unmapped"):
        split_module(source_file, MAPPING, out)
