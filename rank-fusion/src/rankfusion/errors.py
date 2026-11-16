"""Typed errors raised by the fusion engine and the HTTP client.

Every error carries a machine readable ``code`` so that the fusion service
can distinguish *incomplete* runs (the honest answer when a directory
misbehaves) from malformed user requests.
"""
from __future__ import annotations

from typing import Any, Optional


class FusionError(Exception):
    """Base class for all fusion errors."""

    code = "fusion_error"
    http_status = 500

    def __init__(self, message: str, *, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class RequestValidationError(FusionError):
    """The fusion request itself is malformed."""

    code = "request_validation_error"
    http_status = 400


class SnapshotMismatchError(FusionError):
    """A directory responded for a different snapshot than the fixed one."""

    code = "snapshot_mismatch"
    http_status = 409


class HTTPProtocolError(FusionError):
    """A response could not be parsed or violated the response schema."""

    code = "http_protocol_error"
    http_status = 502


class UpstreamError(FusionError):
    """A directory returned a non-retryable or persistently failing status."""

    code = "upstream_error"
    http_status = 502


class BoundViolationError(FusionError):
    """An observable upper-bound / ordering contract was violated."""

    code = "bound_violation"
    http_status = 502


class ScoreInconsistencyError(FusionError):
    """A point lookup disagreed with a sorted-access score."""

    code = "score_inconsistency"
    http_status = 502


class IncompleteFusion(FusionError):
    """The run could not be completed; partial candidates must not be sold
    as the final top-K."""

    code = "incomplete"
    http_status = 502
