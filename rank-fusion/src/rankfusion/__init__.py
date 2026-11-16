"""Exact rational weighted rank fusion over paginated directory APIs.

Submodules that pull in asyncio machinery are exposed lazily via PEP 562 so
``python -m rankfusion.cli`` / ``-m rankfusion.service`` do not re-execute a
submodule during package initialization.
"""
from .request import build_request  # noqa: F401
from .types import (  # noqa: F401
    ComponentScore,
    FusionRequest,
    FusionResult,
    RankedItem,
    SourceSpec,
    SourceStats,
    StopBoundary,
)

__all__ = [
    "FusionEngine",
    "FusionHTTPServer",
    "FusionRequest",
    "FusionResult",
    "ComponentScore",
    "RankedItem",
    "SourceSpec",
    "SourceStats",
    "StopBoundary",
    "build_request",
    "run_fusion",
]


def __getattr__(name: str):  # PEP 562 lazy attributes
    if name in ("FusionEngine", "FusionHTTPServer", "run_fusion"):
        if name == "FusionEngine":
            from .engine import FusionEngine

            return FusionEngine
        from .service import FusionHTTPServer, run_fusion

        return {"FusionHTTPServer": FusionHTTPServer, "run_fusion": run_fusion}[name]
    raise AttributeError(name)
