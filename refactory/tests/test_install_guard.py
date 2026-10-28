"""The install guard: DSH profiles are machine state, and a live run owns them.

A machine has one set of profiles per DSH home, so installing a second clone repoints
them; a run live on the first checkout must not have its spine replaced underneath it.
These tests use temporary checkouts and a temporary DSH_HOME — the real ``~/.dsh`` is
never touched.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from refactory_dsh import dsh_profile, install_guard, pin, run_ownership


def _checkout(root: Path, layer: str = "dsh-graph") -> Path:
    checkout = root / "checkout"
    (checkout / "packages" / "bundle" / dsh_profile.bundle_directory(layer)).mkdir(
        parents=True, exist_ok=True)
    return checkout


def _profile(home: Path, name: str, checkout: Path, layer: str = "dsh-graph") -> None:
    link = (home / "profiles" / name / "node_modules" / "@deepseek-ai" / layer)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(checkout / "packages" / "bundle" / dsh_profile.bundle_directory(layer))


@pytest.fixture
def world(tmp_path, monkeypatch):
    home = tmp_path / "dsh"
    home.mkdir()
    monkeypatch.setenv("DSH_HOME", str(home))
    monkeypatch.setenv("REFACTORY_HOME", str(tmp_path / "refactory"))
    monkeypatch.delenv(install_guard.FORCE_ENV, raising=False)
    return home, tmp_path


def test_a_profile_pointing_at_a_checkout_with_a_live_run_is_refused(world):
    home, tmp = world
    other, mine = _checkout(tmp / "other"), _checkout(tmp / "mine")
    _profile(home, "graph", other)
    run = tmp / "run"
    run.mkdir()
    pin.register_run_on_driver(other, run)

    with run_ownership.ownership(run):
        code, lines = install_guard.check(mine, ("graph",))

    assert code == 1, "a live run's profile is not repointed"
    said = lines[0]
    assert str(other) in said and str(run) in said
    assert "DSH_HOME=" in said and install_guard.FORCE_ENV in said


def test_without_a_live_run_it_repoints_and_says_so(world):
    home, tmp = world
    other, mine = _checkout(tmp / "other"), _checkout(tmp / "mine")
    _profile(home, "graph", other)

    code, lines = install_guard.check(mine, ("graph",))

    assert (code, len(lines)) == (0, 1)
    said = lines[0]
    assert "repointing" in said and str(other) in said and str(mine) in said


def test_a_profile_at_this_checkout_says_nothing(world):
    home, tmp = world
    mine = _checkout(tmp / "mine")
    _profile(home, "graph", mine)

    assert install_guard.check(mine, ("graph",)) == (0, [])


def test_a_missing_profile_says_nothing(world):
    _home, tmp = world
    mine = _checkout(tmp / "mine")
    assert install_guard.check(mine, ("graph",)) == (0, [])


def test_force_relinks_despite_a_live_run(world, monkeypatch):
    home, tmp = world
    other, mine = _checkout(tmp / "other"), _checkout(tmp / "mine")
    _profile(home, "graph", other)
    run = tmp / "run"
    run.mkdir()
    pin.register_run_on_driver(other, run)
    monkeypatch.setenv(install_guard.FORCE_ENV, "1")

    with run_ownership.ownership(run):
        code, lines = install_guard.check(mine, ("graph",))

    assert code == 0 and "repointing" in lines[0]


def test_install_sh_runs_the_guard_before_it_writes_a_profile():
    """The module is only a guard if the installer calls it, and calls it before the rows
    that relink a profile."""
    installer = (Path(__file__).resolve().parents[2] / "install.sh").read_text()
    guard = installer.index("refactory_dsh.install_guard")
    rows = installer.index("while IFS=$'\\t' read -r kind profile value")
    assert guard < rows, "the guard must run before any profile row is executed"
    assert "|| exit 1" in installer[guard:rows]


def test_a_plan_with_nothing_to_write_is_silent(world):
    """install.sh passes exactly the plan's profiles, so no profiles means the install
    writes nothing and the guard must not refuse it."""
    _home, tmp = world
    mine = _checkout(tmp / "mine")
    assert install_guard.main([str(mine)]) == 0


def test_force_accepts_no_as_false(world, monkeypatch):
    """REFACTORY_INSTALL_FORCE_RELINK=0 is not consent."""
    home, tmp = world
    other, mine = _checkout(tmp / "other"), _checkout(tmp / "mine")
    _profile(home, "graph", other)
    run = tmp / "run"
    run.mkdir()
    pin.register_run_on_driver(other, run)
    monkeypatch.setenv(install_guard.FORCE_ENV, "0")

    with run_ownership.ownership(run):
        code, _lines = install_guard.check(mine, ("graph",))
    assert code == 1
