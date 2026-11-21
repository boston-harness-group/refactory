"""The checkout's Python pin matches the Python CI tests on.

Run, check and base worktrees run `uv sync`, which takes `.python-version`; without the
pin uv picked its newest managed Python (3.14), where main's tests differ from CI's
3.12, and a design run's builder was blamed for failures that were already on main.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_the_pin_matches_ci():
    pin = (ROOT / ".python-version").read_text().strip()
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    versions = set(re.findall(r'python-version:\s*"?([0-9.]+)"?', ci))
    assert versions, "ci.yml names no python-version"
    assert versions == {pin}, f".python-version {pin} vs ci.yml {sorted(versions)}"
