"""GigaEvo Python client library.

The package was renamed from ``gigaevo_memory`` to ``gigaevo_client``
in 0.3.0. The old import path remains available as a thin shim that
emits a :class:`DeprecationWarning` once per process — see
``gigaevo_memory/__init__.py``."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _version

try:
    # Single source of truth lives in ``pyproject.toml`` (``version = ...``).
    # Read it from the installed distribution metadata so the literal is
    # never duplicated here.
    __version__ = _version("gigaevo-client")
except PackageNotFoundError:  # pragma: no cover - only when running uninstalled
    __version__ = "0.0.0"

__all__ = [
    "GigaEvoConfig",
    "GigaEvoClient",
    "MemoryClient",
    "PlatformClient",
    "AsyncPlatformClient",
    "PlatformMemoryClient",
    "GigaEvoSuite",
    "SearchType",
    "CachePolicy",
    "EntityRef",
    "EntityResponse",
    "ChainRecord",
    "BulkSaveResult",
    "ProbeResult",
    "VersionInfo",
    "VersionDetail",
    "DiffResponse",
    "FacetsResponse",
    "AgentSpec",
    "AgentSkillSpec",
    "CapabilityHit",
    "CareChainMetadata",
    "ContextFileRef",
    "EvolutionMeta",
    "LineageResponse",
    "LineageVersion",
    "DifferentialChannelView",
    "VersionScore",
    "DuplicateMember",
    "DuplicatePair",
    "DuplicatesResponse",
    "MemoryCardExplanation",
    "MemoryCardSpec",
    "DatasetMemoryCardSpec",
    "SearchHitData",
    "ChainDict",
    "StepDict",
    "AgentDict",
    "AgentSkillDict",
    "MemoryCardDict",
    "EmbeddingProvider",
    "SentenceTransformersProvider",
    "HuggingFaceProvider",
    "OpenAIProvider",
    "MemoryApiProvider",
    "get_default_provider",
    "set_default_provider",
    "MemoryError",
    "NotFoundError",
    "ConflictError",
    "ConnectionError",
    "ValidationError",
    "probe",
]


def __getattr__(name: str):
    if name == "GigaEvoConfig":
        from .config import GigaEvoConfig

        return GigaEvoConfig
    if name in {"GigaEvoClient", "MemoryClient"}:
        # Both names resolve to the same class. ``MemoryClient`` is the
        # legacy alias preserved for callers that haven't migrated yet.
        from .client import GigaEvoClient

        return GigaEvoClient
    if name == "PlatformMemoryClient":
        from .platform_client import PlatformMemoryClient

        return PlatformMemoryClient
    if name == "PlatformClient":
        from .platform import PlatformClient

        return PlatformClient
    if name == "AsyncPlatformClient":
        from .platform import AsyncPlatformClient

        return AsyncPlatformClient
    if name == "probe":
        from .client import probe

        return probe
    if name == "GigaEvoSuite":
        from .suite import GigaEvoSuite

        return GigaEvoSuite
    if name == "SearchType":
        from .search_types import SearchType

        return SearchType
    if name == "CachePolicy":
        from .cache import CachePolicy

        return CachePolicy
    if name in {
        "EmbeddingProvider",
        "SentenceTransformersProvider",
        "HuggingFaceProvider",
        "OpenAIProvider",
        "MemoryApiProvider",
        "get_default_provider",
        "set_default_provider",
    }:
        from . import embeddings as _embeddings

        return getattr(_embeddings, name)
    if name in {
        "MemoryError",
        "NotFoundError",
        "ConflictError",
        "ConnectionError",
        "ValidationError",
    }:
        from . import exceptions as _exceptions

        return getattr(_exceptions, name)
    if name in {
        "AgentDict",
        "AgentSkillDict",
        "AgentSkillSpec",
        "AgentSpec",
        "CapabilityHit",
        "BulkSaveResult",
        "CareChainMetadata",
        "ChainRecord",
        "ChainDict",
        "ContextFileRef",
        "DiffResponse",
        "DifferentialChannelView",
        "DuplicateMember",
        "DuplicatePair",
        "DuplicatesResponse",
        "EvolutionMeta",
        "EntityRef",
        "LineageResponse",
        "LineageVersion",
        "VersionScore",
        "EntityResponse",
        "FacetsResponse",
        "MemoryCardDict",
        "DatasetMemoryCardSpec",
        "MemoryCardExplanation",
        "MemoryCardSpec",
        "ProbeResult",
        "SearchHitData",
        "StepDict",
        "VersionDetail",
        "VersionInfo",
    }:
        from . import models as _models

        return getattr(_models, name)
    raise AttributeError(f"module 'gigaevo_client' has no attribute {name!r}")
