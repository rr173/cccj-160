"""Request / result data structures for the fusion API."""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Optional

from .errors import RequestValidationError
from .rational import rational_str


@dataclass(frozen=True)
class SourceSpec:
    """One directory participating in the fusion."""

    source_id: str
    base_url: str
    weight: Fraction


@dataclass(frozen=True)
class FusionRequest:
    """A fully validated fusion request bound to one fixed snapshot."""

    source_ids: tuple[str, ...]
    sources: tuple[SourceSpec, ...]
    k: int
    catalog_snapshot: str
    max_concurrency: int = 4
    max_attempts: int = 5
    request_timeout: float = 10.0
    backoff_base: float = 0.05

    def source(self, source_id: str) -> SourceSpec:
        for spec in self.sources:
            if spec.source_id == source_id:
                return spec
        raise RequestValidationError(f"unknown source_id {source_id!r}")


@dataclass
class ComponentScore:
    """One source's contribution for an item."""

    source_id: str
    weight: Fraction
    score: Fraction
    weighted: Fraction
    known: bool  # True = explicitly observed (page or point); False = absent/unseen

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "weight": rational_str(self.weight),
            "score": rational_str(self.score),
            "weighted_score": rational_str(self.weighted),
            "known": self.known,
        }


@dataclass
class RankedItem:
    item_id: str
    total_score: Fraction
    components: list[ComponentScore] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "total_score": rational_str(self.total_score),
            "components": [c.to_dict() for c in self.components],
        }


@dataclass
class SourceStats:
    """Per-source request accounting kept for the trace and the audit trail."""

    source_id: str
    pages_read: int = 0
    point_lookups: int = 0
    empty_pages: int = 0
    http_attempts: int = 0
    retries: int = 0
    exhausted: bool = False
    last_cursor: Optional[str] = None
    # Highest promised upper bound currently relevant for unread pages.
    current_upper_bound: Optional[Fraction] = None
    # Weighted form of current_upper_bound, cached for convenience.
    current_weighted_bound: Optional[Fraction] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "pages_read": self.pages_read,
            "point_lookups": self.point_lookups,
            "empty_pages": self.empty_pages,
            "http_attempts": self.http_attempts,
            "retries": self.retries,
            "exhausted": self.exhausted,
            "last_cursor": self.last_cursor,
            "current_upper_bound": (
                rational_str(self.current_upper_bound)
                if self.current_upper_bound is not None
                else None
            ),
            "current_weighted_bound": (
                rational_str(self.current_weighted_bound)
                if self.current_weighted_bound is not None
                else None
            ),
        }


@dataclass
class StopBoundary:
    """The exact boundary that justifies termination."""

    reason: str
    kth_item_id: Optional[str]
    kth_total: Optional[Fraction]
    # Best possible total for any item still completely unseen.
    unseen_potential: Fraction
    # Per-source weighted bounds behind the unseen potential.
    unseen_potential_breakdown: list[dict[str, Any]]
    # Unresolved seen candidates that were blocking termination (empty on stop).
    blocking_seen: list[dict[str, Any]]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "kth_item_id": self.kth_item_id,
            "kth_total": rational_str(self.kth_total) if self.kth_total is not None else None,
            "unseen_potential": rational_str(self.unseen_potential),
            "unseen_potential_breakdown": self.unseen_potential_breakdown,
            "blocking_seen_candidates": self.blocking_seen,
            "notes": self.notes,
        }


@dataclass
class FusionResult:
    status: str  # "complete" or "incomplete"
    catalog_snapshot: str
    k: int
    items: list[RankedItem]
    source_stats: list[SourceStats]
    stop_boundary: StopBoundary
    # Number of distinct candidates observed on at least one page. This is a
    # lower bound on the union while sources remain non-exhausted; the
    # directory API exposes no total-count endpoint, so the exact union size
    # is only knowable when every source reached its terminal cursor.
    observed_candidate_count: int
    union_fully_enumerated: bool
    universe_size: Optional[int] = None
    incomplete_reason: Optional[str] = None
    incomplete_details: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "status": self.status,
            "catalog_snapshot": self.catalog_snapshot,
            "requested_k": self.k,
            "observed_candidate_count": self.observed_candidate_count,
            "union_fully_enumerated": self.union_fully_enumerated,
            "universe_size": self.universe_size,
            "items": [item.to_dict() for item in self.items],
            "source_stats": [s.to_dict() for s in self.source_stats],
            "stop_boundary": self.stop_boundary.to_dict(),
        }
        if self.status != "complete":
            out["incomplete_reason"] = self.incomplete_reason
            out["incomplete_details"] = self.incomplete_details
        return out
