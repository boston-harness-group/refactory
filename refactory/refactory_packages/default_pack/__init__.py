"""The default package: the modes a checkout starts with, whose output is a mode or an artifact.

These are modes like any other — they have a graph, they are walked by the same
executor, and they are discovered by the same registry. What makes them a package
rather than a library is that they belong together and are listed together: the
question "what can this checkout create?" is answered by one entry.

They are also where the mode contract is *deliberately* stretched, and the
stretches are stated here rather than special-cased in the library:

- **They are unscored.** ``score()`` is not overridden, on purpose: a creation is
  judged by whether the package it produced imports, validates and smoke-scores —
  which its own graph checks — not by a number. The outer loop therefore refuses
  them as a search target (``Loop`` requires a scored mode), which is correct: you
  cannot hill-climb on "did it work".
- **Their ops live in the library.** ``create_ops`` and ``design_ops`` ship with
  ``refactory_dsh`` rather than under this package, because a mode that writes a
  package needs the validator and the registry, and those are the library's. A
  package may name any importable ops module; this one names two of the library's.
- **They need a model.** No knob makes them run without one, so a run with no model
  client fails loudly at the first agent node rather than producing nothing.
"""

from __future__ import annotations

from refactory_dsh.create import CreateMode
from refactory_dsh.create_fidelity import CreateFidelityMode
from refactory_dsh.design import DesignMode
from refactory_dsh.edit import EditMode
from refactory_dsh.fast import CreateFastMode, EditFastMode, DesignFastMode
from refactory_dsh.nn import Package

PACKAGE = Package(
    name="default_pack",
    description="Create a package, revise one, or design an artifact — from a brief.",
    modes=(CreateMode, EditMode, DesignMode, CreateFastMode, EditFastMode, DesignFastMode, CreateFidelityMode),
    ops=("refactory_dsh.create_ops", "refactory_dsh.design_ops", "refactory_dsh.fast_ops",
         "refactory_dsh.create_fidelity_ops", "refactory_dsh.create_fidelity_directors",
         "refactory_dsh.create_fidelity_gates"),
)
