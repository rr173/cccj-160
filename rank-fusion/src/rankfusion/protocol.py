"""Parsing and local validation of directory HTTP response bodies.

These functions enforce the *observable* part of the interface contract:
snapshot/source binding, score domain, intra-page ordering and response
shape. Cross-page boundary checks live in the engine because they need
state from previous pages.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Optional

from .errors import HTTPProtocolError, SnapshotMismatchError
from .rational import parse_score


@dataclass(frozen=True)
class PageRecord:
    item_id: str
    score: Fraction


@dataclass(frozen=True)
class PageResponse:
    snapshot: str
    source_id: str
    cursor: Optional[str]
    records: tuple[PageRecord, ...]
    next_cursor: Optional[str]
    upper_bound: Optional[Fraction]


@dataclass(frozen=True)
class ItemResponse:
    snapshot: str
    source_id: str
    item_id: str
    present: bool
    score: Optional[Fraction]


def _require_mapping(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise HTTPProtocolError("response body must be a JSON object")
    return body


def _check_binding(body: dict[str, Any], *, expected_snapshot: str, expected_source: str) -> None:
    snapshot = body.get("snapshot")
    if not isinstance(snapshot, str):
        raise HTTPProtocolError("response missing string 'snapshot'")
    if snapshot != expected_snapshot:
        raise SnapshotMismatchError(
            f"snapshot mismatch: request bound to {expected_snapshot!r}, "
            f"directory answered {snapshot!r}",
            details={"expected": expected_snapshot, "received": snapshot},
        )
    source_id = body.get("source_id")
    if source_id != expected_source:
        raise HTTPProtocolError(
            f"source_id mismatch: expected {expected_source!r}, got {source_id!r}"
        )


def parse_page_response(
    body: Any, *, expected_snapshot: str, expected_source: str
) -> PageResponse:
    data = _require_mapping(body)
    _check_binding(data, expected_snapshot=expected_snapshot, expected_source=expected_source)

    raw_records = data.get("records")
    if not isinstance(raw_records, list):
        raise HTTPProtocolError("page response missing list 'records'")

    records: list[PageRecord] = []
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            raise HTTPProtocolError(f"records[{index}] must be an object")
        item_id = raw.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise HTTPProtocolError(f"records[{index}] has invalid item_id")
        try:
            score = parse_score(raw.get("score"), field=f"records[{index}].score")
        except ValueError as exc:
            raise HTTPProtocolError(str(exc)) from exc
        records.append(PageRecord(item_id=item_id, score=score))

    # Contract: within a page, scores are non-increasing and ties break by
    # ascending item_id (Python string comparison == Unicode code point order).
    for left, right in zip(records, records[1:]):
        if left.score < right.score:
            raise HTTPProtocolError(
                f"page ordering violation: {left.item_id}({left.score}) precedes "
                f"{right.item_id}({right.score}) with lower score"
            )
        if left.score == right.score and left.item_id > right.item_id:
            raise HTTPProtocolError(
                f"page ordering violation: tied items {left.item_id!r} and "
                f"{right.item_id!r} are not in ascending item_id order"
            )

    next_cursor = data.get("next_cursor")
    if next_cursor is not None and not isinstance(next_cursor, str):
        raise HTTPProtocolError("next_cursor must be a string or null")

    raw_bound = data.get("upper_bound")
    if raw_bound is not None:
        if not isinstance(raw_bound, str):
            raise HTTPProtocolError("upper_bound must be an exact rational string or null")
        try:
            upper_bound = parse_score(raw_bound, field="upper_bound")
        except ValueError as exc:
            raise HTTPProtocolError(str(exc)) from exc
    else:
        upper_bound = None

    cursor = data.get("cursor")
    if cursor is not None and not isinstance(cursor, str):
        raise HTTPProtocolError("cursor echo must be a string or null")

    return PageResponse(
        snapshot=data["snapshot"],
        source_id=data["source_id"],
        cursor=cursor,
        records=tuple(records),
        next_cursor=next_cursor,
        upper_bound=upper_bound,
    )


def parse_item_response(
    body: Any, *, expected_snapshot: str, expected_source: str, expected_item: str
) -> ItemResponse:
    data = _require_mapping(body)
    _check_binding(data, expected_snapshot=expected_snapshot, expected_source=expected_source)

    item_id = data.get("item_id")
    if item_id != expected_item:
        raise HTTPProtocolError(
            f"item_id mismatch: queried {expected_item!r}, response says {item_id!r}"
        )
    status = data.get("status")
    if status == "present":
        try:
            score = parse_score(data.get("score"), field="score")
        except ValueError as exc:
            raise HTTPProtocolError(str(exc)) from exc
        return ItemResponse(data["snapshot"], data["source_id"], item_id, True, score)
    if status == "absent":
        # Explicit absent carrying snapshot identity: the item scores 0 here.
        if data.get("score") is not None:
            raise HTTPProtocolError("absent item response must not carry a score")
        return ItemResponse(data["snapshot"], data["source_id"], item_id, False, None)
    raise HTTPProtocolError(f"item response has unknown status {status!r}")
