"""Optimizers: the ``torch.optim`` of the harness world.

    from refactory_dsh.optim import AutoresearchOptimizer, RandomOptimizer

An optimizer sees a :class:`ProposalContext` — the parameter surface, the edit
catalogue, the leaderboard, the traces, and the reasons recent proposals were
refused — and returns typed edits. The base class runs the validating loop around
it, so no optimizer can spend evaluation budget on a candidate that cannot run.
"""

from __future__ import annotations

from pathlib import Path

from .base import (
    Candidate,
    Optimizer,
    Proposal,
    ProposalContext,
    Rejection,
    candidate_id,
)
from .random_optimizer import AutoresearchOptimizer, RandomOptimizer, extract_json
from .policies import (
    ArchiveSelector,
    EvolutionarySelector,
    HillClimbSelector,
    IslandsSelector,
    MapElitesSelector,
    ParentSelector,
    ParetoSelector,
    pipeline_shape,
)
from .compose import PickedParent

#: The library's own optimizers, by name: filled only by :func:`register` (below, for
#: the library's own). A package contributes more through ``Package.optimizers``
#: (:func:`package_optimizers`, the other route), selectable by name without joining
#: this table; a package name that clashes with one here is refused.
REGISTRY: dict[str, type[Optimizer]] = {}


def register(cls: type[Optimizer], name: str = "") -> type[Optimizer]:
    """Make ``cls`` selectable by ``name`` (its class-level ``name`` when not given),
    as the ``optimizer`` setting, or as the ``parent_picker`` setting when it is a
    :class:`ParentSelector`. Refuses a name already taken by another class: a silent
    replacement would change what every run naming it does."""
    if not (isinstance(cls, type) and issubclass(cls, Optimizer)):
        raise TypeError(f"{cls!r} is not an Optimizer")
    key = name or str(getattr(cls, "name", "") or "")
    if not key:
        raise ValueError(f"{cls.__name__} needs a class-level 'name' to be registered")
    held = REGISTRY.get(key)
    if held is not None and held is not cls:
        raise ValueError(f"optimizer name {key!r} is already registered to {held.__name__}")
    REGISTRY[key] = cls
    return cls


for _cls in (RandomOptimizer, AutoresearchOptimizer, ArchiveSelector, EvolutionarySelector,
             IslandsSelector, HillClimbSelector, MapElitesSelector, ParetoSelector):
    register(_cls)
del _cls


def package_optimizers(project: Path | None = None) -> dict[str, type[Optimizer]]:
    """Optimizers the discoverable packages contribute (``Package.optimizers``), by
    name, found as package strategies are (``registry.contributed``)."""
    from refactory_dsh import registry

    return registry.contributed("optimizers", project)  # type: ignore[return-value]


def names() -> list[str]:
    """The library's own optimizers; :func:`available` adds the packages'."""
    return sorted(REGISTRY)


def available(project: Path | None = None) -> list[str]:
    """Every selectable optimizer: the library's own and the packages'."""
    return sorted(set(REGISTRY) | set(package_optimizers(project)))


def resolve(name: str, project: Path | None = None) -> type[Optimizer]:
    """The class selected by ``name``: the library's own first, then the packages'
    (looked up only when the name is not the library's, so the common case imports
    no package). Fails loud on an unknown name."""
    if name in REGISTRY:
        return REGISTRY[name]
    from refactory_dsh import registry

    found, refused, refused_names = registry.survey("optimizers", project)
    if name in found:
        return found[name][0]  # type: ignore[return-value]
    if name in refused_names:
        # The package that lists it was left out: the run is refused with why.
        raise registry.PackageError(
            f"optimizer {name!r} is not available: {refused[refused_names[name]]}")
    known = sorted(set(REGISTRY) | set(found))
    raise ValueError(f"unknown optimizer {name!r}; known: {known}")


def is_picker(name: str, project: Path | None = None) -> bool:
    """Whether ``name`` chooses parents (a :class:`ParentSelector`), and so can be the
    ``parent_picker`` paired with a proposer; ``False`` for an unknown name."""
    from refactory_dsh.nn.package import PackageError

    try:
        return issubclass(resolve(name, project), ParentSelector)
    except PackageError:
        raise
    except ValueError:
        return False


def is_population(name: str, project: Path | None = None) -> bool:
    """Whether ``name`` is a parent picker that breeds from a population of parents
    (``ParentSelector.population``: an archive, a tournament, islands, niches): under
    the validation gate it is handed every judged candidate, and the default
    ``selection`` is ``pool``. ``False`` for an empty or unknown name."""
    if not name:
        return False
    try:
        cls = resolve(name, project)
    except ValueError:
        return False
    return issubclass(cls, ParentSelector) and bool(getattr(cls, "population", False))


def picker_names(project: Path | None = None) -> list[str]:
    """The optimizers that choose parents, and so can be paired with a proposer
    (the ``parent_picker`` setting): the library's and the packages'."""
    found = dict(package_optimizers(project))
    found.update(REGISTRY)
    return sorted(name for name, cls in found.items() if issubclass(cls, ParentSelector))


def create(name: str, project: Path | None = None, **kwargs: object) -> Optimizer:
    """Build an optimizer by name (:func:`resolve`), failing loud on an unknown one.

    Falling back silently would invalidate every result of the run, because the
    report would not say which optimizer produced them.
    """
    return resolve(name, project)(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "REGISTRY",
    "available",
    "is_picker",
    "is_population",
    "package_optimizers",
    "register",
    "resolve",
    "AutoresearchOptimizer",
    "Candidate",
    "Optimizer",
    "PickedParent",
    "Proposal",
    "ParentSelector",
    "ProposalContext",
    "RandomOptimizer",
    "Rejection",
    "candidate_id",
    "create",
    "extract_json",
    "names",
    "picker_names",
    "pipeline_shape",
]
