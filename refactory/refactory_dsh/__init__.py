import os as _os

if _os.environ.get("REFACTORY_SPAWN_LOG"):
    # A measurement of how many processes refactory starts (`spawn_count`); unset, nothing runs.
    from refactory_dsh import spawn_count as _spawn_count

    _spawn_count.install()
