"""The split-module script reproduces committed modules byte-for-byte."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MAPPING = REPO / "refactory" / "tools" / "splits" / "optim_base.json"
SCRIPT = REPO / "refactory" / "tools" / "split_module.py"


@pytest.fixture()
def split_dir(tmp_path: Path) -> Path:
    mapping = json.loads(MAPPING.read_text())
    base_commit = mapping["base_commit"]
    source_rel = mapping["source"]

    work = tmp_path / "split"
    work.mkdir()

    source_dir = work / Path(source_rel).parent
    source_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "show", f"{base_commit}:{source_rel}"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0, result.stderr
    (work / source_rel).write_text(result.stdout)

    tools_dir = work / SCRIPT.parent.relative_to(REPO)
    tools_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPT, tools_dir / SCRIPT.name)
    splits_dir = work / MAPPING.parent.relative_to(REPO)
    splits_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MAPPING, splits_dir / MAPPING.name)

    script_copy = tools_dir / SCRIPT.name
    mapping_copy = splits_dir / MAPPING.name
    result = subprocess.run(
        ["python3", "-I", str(script_copy.relative_to(work)),
         str((work / source_rel).relative_to(work)),
         str(mapping_copy.relative_to(work))],
        capture_output=True, text=True, cwd=str(work),
    )
    assert result.returncode == 0, f"split_module failed:\n{result.stderr}"
    return work


def test_split_reproduces_committed_modules(split_dir: Path) -> None:
    mapping = json.loads(MAPPING.read_text())
    modules = mapping["modules"]
    source_rel = mapping["source"]

    all_paths = list(modules.values())
    for path_str in modules.values():
        init = Path(path_str).parent / "__init__.py"
        if init != Path(source_rel).parent / "__init__.py":
            if str(init) not in all_paths:
                all_paths.append(str(init))

    for rel_path in sorted(set(all_paths)):
        committed = REPO / rel_path
        generated = split_dir / rel_path
        assert generated.exists(), f"not generated: {rel_path}"
        assert committed.exists(), f"not committed: {rel_path}"
        expected = committed.read_text()
        actual = generated.read_text()
        if expected != actual:
            diff = subprocess.run(
                ["diff", "-u", str(committed), str(generated)],
                capture_output=True, text=True,
            )
            pytest.fail(f"{rel_path} differs from committed:\n{diff.stdout[:2000]}")


def test_unmapped_name_fails(tmp_path: Path) -> None:
    mapping = json.loads(MAPPING.read_text())
    base_commit = mapping["base_commit"]
    source_rel = mapping["source"]

    work = tmp_path / "unmapped"
    work.mkdir()
    source_dir = work / Path(source_rel).parent
    source_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["git", "show", f"{base_commit}:{source_rel}"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    (work / source_rel).write_text(result.stdout)

    trimmed = dict(mapping)
    trimmed["names"] = {k: v for k, v in mapping["names"].items() if k != "Optimizer"}
    mapping_path = work / "mapping.json"
    mapping_path.write_text(json.dumps(trimmed))

    run = subprocess.run(
        ["python3", "-I", str(SCRIPT),
         str(work / source_rel), str(mapping_path)],
        capture_output=True, text=True, cwd=str(work),
    )
    assert run.returncode != 0, "should fail on unmapped name"
    assert "unmapped" in run.stderr.lower()
