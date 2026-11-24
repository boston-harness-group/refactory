"""`--shard I/N`: the shards together run every collected test exactly once, packed by
CI seconds; a file too long for one worker is spread function by function; and the plan
does not depend on which parametrizations a runner happens to collect.

`scripts/check-shards.py` checks the real collection (fast-checks job) and what the
shards really ran (the job after them); these hold the packing rules on made-up suites."""

from __future__ import annotations

from types import SimpleNamespace

import conftest


def _suite() -> list[str]:
    tests = [f"refactory/tests/test_long.py::test_{i}" for i in range(8)]
    tests += [f"refactory/tests/test_long.py::test_p[{p}]" for p in ("a", "b", "c")]
    tests += [f"refactory/tests/test_mid.py::TestA::test_{i}" for i in range(3)]
    tests += [f"refactory/tests/test_small_{i}.py::test_x" for i in range(6)]
    tests += ["refactory/tests/test_new.py::test_x", "refactory/tests/test_new.py::test_y"]
    return tests


#: test_long's functions weigh 40..180 s (test_7 is new: no time yet), test_p 150 s.
KNOWN = {
    "test_long.py": 1000.0,
    **{f"test_long.py::test_{i}": 40.0 + 20.0 * i for i in range(7)},
    "test_long.py::test_p": 150.0,
    "test_mid.py": 300.0,
    **{f"test_small_{i}.py": 10.0 * (i + 1) for i in range(6)},
}


def _loads(owner: dict[str, int], weight: dict[str, float], shards: int) -> list[float]:
    loads = [0.0] * shards
    for test, shard in owner.items():
        loads[shard] += weight[test]
    return loads


def test_every_test_lands_in_exactly_one_shard_whatever_the_count():
    tests = _suite()
    for shards in (1, 2, 3, 4, 7, 40):
        owner, weight = conftest._shard_plan(tests, shards, KNOWN)
        assert sorted(owner) == sorted(tests)
        assert set(owner.values()) <= set(range(shards))
        assert set(weight) == set(tests)


def test_each_function_of_a_long_file_weighs_its_own_seconds():
    _, weight = conftest._shard_plan(_suite(), 4, KNOWN)
    for i in range(7):
        assert weight[f"refactory/tests/test_long.py::test_{i}"] == 40.0 + 20.0 * i
    # a new function: the median of the file's listed functions (40..160 and 150): 110 s
    assert weight["refactory/tests/test_long.py::test_7"] == 110.0
    # a parametrized function's seconds are shared by its cases
    assert weight["refactory/tests/test_long.py::test_p[a]"] == 50.0


def test_a_file_without_a_time_runs_and_stays_whole():
    owner, weight = conftest._shard_plan(_suite(), 4, KNOWN)
    new = [t for t in owner if "test_new.py" in t]
    assert len({owner[t] for t in new}) == 1
    # the median of the eight timed files (sorted, upper middle): 50 s
    assert sum(weight[t] for t in new) == 50.0


def test_a_long_file_is_spread_but_a_function_keeps_its_parametrizations_together():
    owner, _ = conftest._shard_plan(_suite(), 4, KNOWN)
    long = [t for t in owner if "test_long.py" in t]
    assert len({owner[t] for t in long}) >= 3  # the 300 s file may fill one alone
    assert len({owner[t] for t in long if "test_p[" in t}) == 1


def test_an_unlisted_file_never_leaves_its_shard_and_weighs_its_seconds():
    owner, weight = conftest._shard_plan(_suite(), 4, KNOWN)
    mid = [t for t in owner if "test_mid.py" in t]
    assert len({owner[t] for t in mid}) == 1
    assert sum(weight[t] for t in mid) == 300.0
    assert weight["refactory/tests/test_small_5.py::test_x"] == 60.0


def test_the_shards_come_out_even_where_the_pieces_allow():
    owner, weight = conftest._shard_plan(_suite(), 4, KNOWN)
    loads = _loads(owner, weight, 4)
    assert max(loads) - min(loads) <= 60.0


def test_a_parametrization_only_some_runners_collect_moves_nothing():
    """A runner whose sandbox probe failed collects fewer cases of a long file's
    function; every test both runners collect must still land on the same shard, or
    one test runs twice and another never runs (#752 review, H1)."""
    full = _suite() + [f"refactory/tests/test_long.py::test_p[{p}-code]" for p in "abcdefgh"]
    full += [f"refactory/tests/test_long.py::test_{i}[x-code]" for i in (2, 5)]
    less = _suite()
    for shards in (2, 3, 4, 5):
        with_code, _ = conftest._shard_plan(full, shards, KNOWN)
        without, _ = conftest._shard_plan(less, shards, KNOWN)
        assert {t: with_code[t] for t in less} == without
        by_function: dict[str, set[int]] = {}
        for test, shard in with_code.items():
            by_function.setdefault(conftest._function_key(test), set()).add(shard)
        assert all(len(found) == 1 for key, found in by_function.items()
                   if key.startswith("test_long.py::"))


def test_no_durations_file_still_runs_everything_once():
    tests = _suite()
    owner, _ = conftest._shard_plan(tests, 3, {})
    assert sorted(owner) == sorted(tests)


def test_longest_scope_goes_first_and_a_scope_keeps_its_order():
    tests = ["a.py::test_1", "a.py::test_2", "b.py::K::test_1", "b.py::K::test_2", "c.py::t"]
    weight = {"a.py::test_1": 1.0, "a.py::test_2": 1.0, "b.py::K::test_1": 5.0,
              "b.py::K::test_2": 0.0, "c.py::t": 3.0}
    assert conftest._longest_scope_first(tests, weight) == [
        "b.py::K::test_1", "b.py::K::test_2", "c.py::t", "a.py::test_1", "a.py::test_2"]


def _config(shard: str | None, workers: int | None, dist: str | None = None):
    options = {"--shard": shard, "numprocesses": workers, "dist": dist}
    option = SimpleNamespace(dist=dist, loadscopereorder=True)
    return SimpleNamespace(option=option, getoption=lambda name, default=None: options[name])


def test_under_shard_xdist_keeps_the_longest_first_order():
    config = _config("2/4", 4)
    conftest._xdist_options(config)
    assert (config.option.dist, config.option.loadscopereorder) == ("loadscope", False)


def test_without_shard_xdist_reorders_as_usual():
    config = _config(None, 4)
    conftest._xdist_options(config)
    assert (config.option.dist, config.option.loadscopereorder) == ("loadscope", True)
