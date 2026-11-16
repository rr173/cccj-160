"""Local HTTP directory simulator for rank-fusion development and drills.

Submodules are imported lazily so that ``python -m datasim.server`` does not
re-execute the server module through the package initializer.
"""
from .catalog import CatalogSource, FaultRule, load_catalog_file  # noqa: F401

__all__ = ["CatalogSource", "FaultRule", "SimulatorServer", "load_catalog_file", "start_simulator"]


def __getattr__(name: str):  # PEP 562 lazy attributes
    if name in ("SimulatorServer", "start_simulator"):
        from . import server

        return getattr(server, name)
    raise AttributeError(name)
