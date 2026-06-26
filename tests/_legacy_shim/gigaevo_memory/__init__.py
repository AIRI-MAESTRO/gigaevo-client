"""Test-only vendored copy of the ``gigaevo_memory`` legacy import shim.

The package was renamed to ``gigaevo_client`` in 0.3.0. The shim ships
as a separate PyPI distribution (``gigaevo-memory``) but is not yet
published at >= 0.3.1, so we can't pull it from PyPI as a dev-dep.
Several tests still cover the rename contract (``from gigaevo_memory
import ...``); to keep them runnable in CI we vendor a minimal copy
here and put this directory on ``pythonpath`` via ``pyproject.toml``.

Keep this file in sync with ``gigaevo-memory-internal/client/python-meta/
src/gigaevo_memory/__init__.py``. Once that distribution publishes
0.3.1+ to PyPI, this vendor copy can be deleted and ``gigaevo-memory``
moved back into the ``dev`` extra.
"""

from __future__ import annotations

import importlib
import sys
import warnings as _warnings

_warnings.warn(
    "The 'gigaevo_memory' package was renamed to 'gigaevo_client' in 0.3.0. "
    "Update your imports — 'gigaevo_memory' will be removed in a future "
    "release.",
    DeprecationWarning,
    stacklevel=2,
)

from gigaevo_client import __all__ as __all__  # noqa: F401, E402
from gigaevo_client import __version__ as __version__  # noqa: F401, E402

_SUBMODULES = (
    "_base",
    "_compat",
    "agent_skills",
    "agents",
    "cache",
    "chains",
    "client",
    "config",
    "embeddings",
    "exceptions",
    "memory_cards",
    "models",
    "platform",
    "platform_client",
    "search_types",
    "suite",
    "watcher",
)
for _name in _SUBMODULES:
    sys.modules[f"gigaevo_memory.{_name}"] = importlib.import_module(
        f"gigaevo_client.{_name}"
    )
del _name


def __getattr__(name: str):
    """Forward any top-level attribute lookup to the new package."""
    import gigaevo_client as _new
    try:
        return getattr(_new, name)
    except AttributeError as exc:
        raise AttributeError(
            f"module 'gigaevo_memory' has no attribute {name!r} "
            f"(also not in renamed 'gigaevo_client')"
        ) from exc
