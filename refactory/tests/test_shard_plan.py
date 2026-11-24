"""`--shard I/N`: the shards together run every collected test exactly once, packed by
CI seconds, and a file too long for one worker is spread test by test.

`scripts/check-shards.py` (fast-checks job) checks the real collection; these hold the
packing rules on small made-up suites."""

from __future__ import annotations

import conftest


def _suite() -> list[str]:
    tests = [f"refactory/tests/test_long.py::test_{i}" for i in range(8)]
    tests += [f"refactory/tests/test_mid.py::TestA::test_{i}" for i in range(3)]
    tests += [f"refactory/tests/test_small_{i}.py::test_x" for i in range(6)]
    tests += ["refactory/tests/test_new.py::test_x", "refactory/tests/test_new.py::test_y"]
    return tests


KNOWN = {
    "test_long.py": 800.0,
    **{f"test_long.py::test_{i}": 100.0 for i in range(7)},  # test_7 is new: no time yet
    "test_mid.py": 300.0,
    **{f"test_small_{i}.py": 10.0 * (i + 1) for i in range(6)},
}


def test_every_test_lands_in_exactly_one_shard_whatever_the_count():
    tests = _suite()
    for shards in (1, 2, 3, 4, 7, 40):
        owner, weight = conftest._shard_plan(tests, shards, KNOWN)
        assert sorted(owner) == sorted(tests)
        assert set(owner.values()) <= set(range(shards))
        assert set(weight) == set(tests)


def test_a_file_without_a_time_runs_and_stays_whole():
    owner, weight = conftest._shard_plan(_suite(), 4, KNOWN)
    new = [t for t in owner if "test_new.py" in t]
    assert len({owner[t] for t in new}) == 1
    # the median file of the eight timed ones (sorted list, upper middle): 50 s
    assert sum(weight[t] for t in new) == 50.0


def test_a_file_listed_test_by_test_is_spread_and_its_new_test_gets_the_median():
    owner, weight = conftest._shard_plan(_suite(), 4, KNOWN)
    long = [t for t in owner if "test_long.py" in t]
    assert len({owner[t] for t in long}) == 3  # the 300 s file fills the fourth
    assert weight["refactory/tests/test_long.py::test_7"] == 100.0


def test_an_unlisted_file_never_leaves_its_shard():
    owner, _ = conftest._shard_plan(_suite(), 4, KNOWN)
    mid = [t for t in owner if "test_mid.py" in t]
    assert len({owner[t] for t in mid}) == 1


def test_the_shards_come_out_even_where_the_pieces_allow():
    owner, weight = conftest._shard_plan(_suite(), 4, KNOWN)
    loads = [0.0] * 4
    for test, shard in owner.items():
        loads[shard] += weight[test]
    assert max(loads) - min(loads) <= 100.0  # no more than the largest piece left over


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
