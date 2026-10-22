"""Adapter for importing modules from the v1 (remote-factory) subtree.

Adds the subtree root to sys.path so ``import factory.workflow.primitives``
resolves to the v1 codebase at ``v1/``.  The path is inserted
only once and only when this module is first imported.

v1's ``factory/__init__.py`` requires ``structlog``, which v2 already depends
on.  No other third-party dependencies are triggered by importing the IR or
the OpenShell confinement layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

_V1_ROOT = Path(__file__).resolve().parents[2] / "v1"

if str(_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(_V1_ROOT))
