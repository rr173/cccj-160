"""Validation and construction of fusion requests from plain JSON data."""
from __future__ import annotations

from typing import Any

from .errors import RequestValidationError
from .rational import parse_weight
from .types import FusionRequest, SourceSpec

_DEFAULTS = {
    "max_concurrency": 4,
    "max_attempts": 5,
    "request_timeout": 10.0,
    "backoff_base": 0.05,
}


def build_request(data: dict[str, Any]) -> FusionRequest:
    if not isinstance(data, dict):
        raise RequestValidationError("request body must be a JSON object")

    snapshot = data.get("catalog_snapshot")
    if not isinstance(snapshot, str) or not snapshot:
        raise RequestValidationError("catalog_snapshot must be a non-empty string")

    k = data.get("k")
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise RequestValidationError("k must be a non-negative integer")

    raw_sources = data.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise RequestValidationError("sources must be a non-empty list")

    weights = data.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise RequestValidationError("weights must be an object mapping source_id to positive rationals")

    source_ids: list[str] = []
    specs: list[SourceSpec] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_sources):
        if not isinstance(raw, dict):
            raise RequestValidationError(f"sources[{index}] must be an object")
        source_id = raw.get("source_id")
        base_url = raw.get("base_url")
        if not isinstance(source_id, str) or not source_id:
            raise RequestValidationError(f"sources[{index}].source_id must be a non-empty string")
        if source_id in seen:
            raise RequestValidationError(f"duplicate source_id {source_id!r}")
        if not isinstance(base_url, str) or not base_url.startswith("http://"):
            raise RequestValidationError(
                f"sources[{index}].base_url must be an http:// URL"
            )
        if source_id not in weights:
            raise RequestValidationError(f"missing positive weight for source {source_id!r}")
        weight = parse_weight(weights[source_id], source_id=source_id)
        seen.add(source_id)
        source_ids.append(source_id)
        specs.append(SourceSpec(source_id=source_id, base_url=base_url.rstrip("/"), weight=weight))

    # Every weight must correspond to a declared source.
    extra = set(weights) - set(source_ids)
    if extra:
        raise RequestValidationError(f"weights given for unknown source_ids: {sorted(extra)}")

    options = {}
    for name, expected_type in (
        ("max_concurrency", int),
        ("max_attempts", int),
        ("request_timeout", (int, float)),
        ("backoff_base", (int, float)),
    ):
        if name in data:
            value = data[name]
            if isinstance(value, bool) or not isinstance(value, expected_type) or value <= 0:
                raise RequestValidationError(f"{name} must be a positive number")
            options[name] = value
    options = {**_DEFAULTS, **options}

    return FusionRequest(
        source_ids=tuple(source_ids),
        sources=tuple(specs),
        k=k,
        catalog_snapshot=snapshot,
        **options,
    )
