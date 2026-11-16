"""In-memory catalog model for the local directory simulator.

A catalog is a *static complete* directory for one fixed snapshot: every
(item_id, score) pair is known to the simulator. From it the simulator builds
a deterministic page plan honoring the public contract:

  * records descend by score with item_id ascending on ties;
  * every page promises an ``upper_bound`` for the *following* page;
  * empty pages may appear in the middle (and may keep a next_cursor);
  * bounds never increase; only the terminal cursor proves exhaustion.

Fault rules can inject latency, 429/503 responses or deliberately broken
bounds so the fusion engine's contract enforcement can be exercised.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

from rankfusion.errors import RequestValidationError
from rankfusion.rational import parse_score


def encode_cursor(offset: int, slot: int = 0) -> str:
    """Opaque cursor.

    ``offset`` is the record offset; ``slot`` disambiguates empty pages that
    consume no records and therefore share an offset with the following page.
    The cursor is meaningful only within the fixed snapshot, as required.
    """
    raw = json.dumps({"off": offset, "slot": slot}, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> tuple[int, int]:
    try:
        padding = "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode(cursor + padding)
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - any decode failure is a bad cursor
        raise RequestValidationError(f"malformed cursor {cursor!r}") from exc
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("off"), int)
        or not isinstance(payload.get("slot"), int)
    ):
        raise RequestValidationError(f"malformed cursor {cursor!r}")
    offset, slot = payload["off"], payload["slot"]
    if offset < 0 or slot < 0:
        raise RequestValidationError(f"malformed cursor {cursor!r}")
    return offset, slot


@dataclass(frozen=True)
class FaultRule:
    """One fault-injection rule, matched against request context.

    Match fields (all optional; an empty rule matches nothing):
      op:        "page" or "item"
      nth:       1-based count of this source's requests of that op, or
                 request sequence overall when op is omitted with nth
      item_id:   only for item requests
      contains:  substring match on item_id
      attempt:   match only a specific transport attempt (default: every attempt)
    Action fields:
      latency, status (429/503/...), bad_bound (fraction string override),
      bad_absent (force item lookup to report absent)
    """

    op: Optional[str] = None
    nth: Optional[int] = None
    item_id: Optional[str] = None
    contains: Optional[str] = None
    attempt: Optional[int] = None
    latency: Optional[float] = None
    status: Optional[int] = None
    bad_bound: Optional[str] = None
    bad_absent: bool = False


@dataclass
class PagePlan:
    offset: int
    slot: int
    records: tuple[tuple[str, Fraction], ...]
    next_offset: Optional[int]  # None => terminal page, next_cursor is null
    next_slot: Optional[int]
    upper_bound: Optional[Fraction]  # honest bound for the page starting at next_offset


@dataclass
class CatalogSource:
    source_id: str
    snapshot: str
    # Full sorted materialization: list of (item_id, score), contract order.
    ordered: list[tuple[str, Fraction]]
    page_size: int
    empty_page_slots: set[int] = field(default_factory=set)
    faults: list[FaultRule] = field(default_factory=list)
    # Runtime counters, keyed by (op, nth-request-of-op).
    op_counters: dict[str, int] = field(default_factory=dict)
    plans: list[PagePlan] = field(default_factory=list)
    # Fast lookup for /item.
    item_index: dict[str, Fraction] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CatalogSource":
        source_id = config["source_id"]
        snapshot = config["snapshot"]
        page_size = int(config.get("page_size", 5))
        if page_size < 1:
            raise ValueError(f"source {source_id}: page_size must be >= 1")
        entries: list[tuple[str, Fraction]] = []
        for entry in config.get("entries", []):
            item_id = entry["item_id"]
            if not isinstance(item_id, str) or not item_id:
                raise ValueError(f"source {source_id}: invalid item_id {entry!r}")
            score = parse_score(entry.get("score"), field=f"{source_id}/{item_id}.score")
            entries.append((item_id, score))
        if any(item_id in {e[0] for e in entries[:i]} for i, (item_id, _) in enumerate(entries)):
            raise ValueError(f"source {source_id}: duplicate item_id in catalog config")
        # Contract order: score desc, item_id code-point asc.
        entries.sort(key=lambda pair: (-pair[1], pair[0]))

        empty_slots = {int(x) for x in config.get("empty_page_slots", [])}
        faults = [FaultRule(**rule) for rule in config.get("faults", [])]
        source = cls(
            source_id=source_id,
            snapshot=snapshot,
            ordered=entries,
            page_size=page_size,
            empty_page_slots=empty_slots,
            faults=faults,
            item_index=dict(entries),
        )
        source._build_plans()
        return source

    def _build_plans(self) -> None:
        """Lay records out into pages, reserving empty slots where requested."""
        plans: list[PagePlan] = []
        cursor = 0
        slot_index = 0
        total = len(self.ordered)
        while cursor < total:
            if slot_index in self.empty_page_slots:
                # Empty page: no records consumed, continuation promised.
                plans.append(
                    PagePlan(
                        offset=cursor,
                        slot=slot_index,
                        records=(),
                        next_offset=cursor,  # next page starts at same offset
                        next_slot=slot_index + 1,
                        upper_bound=self.ordered[cursor][1],
                    )
                )
                slot_index += 1
                continue
            chunk = self.ordered[cursor : cursor + self.page_size]
            next_cursor = cursor + len(chunk)
            if next_cursor < total:
                bound = self.ordered[next_cursor][1]
                terminal_offset, terminal_slot = next_cursor, slot_index + 1
            else:
                bound = None
                terminal_offset, terminal_slot = None, None
            plans.append(
                PagePlan(
                    offset=cursor,
                    slot=slot_index,
                    records=tuple(chunk),
                    next_offset=terminal_offset,
                    next_slot=terminal_slot,
                    upper_bound=bound,
                )
            )
            cursor = next_cursor
            slot_index += 1
        # A fully empty catalog has exactly one terminal, empty page.
        if not plans:
            plans.append(
                PagePlan(
                    offset=0, slot=0, records=(),
                    next_offset=None, next_slot=None, upper_bound=None,
                )
            )
        self.plans = plans

    # ------------------------------------------------------------------ #
    # Request handling.
    # ------------------------------------------------------------------ #
    def plan_at(self, cursor: Optional[str]) -> PagePlan:
        if cursor is None:
            return self.plans[0]
        offset, slot = decode_cursor(cursor)
        for plan in self.plans:
            if plan.offset == offset and plan.slot == slot:
                return plan
        raise RequestValidationError(
            f"unknown/expired cursor for source {self.source_id}: {cursor!r}"
        )

    def is_terminal_offset(self, offset: int) -> bool:
        return offset >= len(self.ordered)

    def match_faults(
        self, *, op: str, item_id: Optional[str], attempt: int
    ) -> list[FaultRule]:
        nth = self.op_counters.get(op, 0)
        matched: list[FaultRule] = []
        for rule in self.faults:
            if rule.op is not None and rule.op != op:
                continue
            if rule.nth is not None and rule.nth != nth:
                continue
            if rule.item_id is not None and rule.item_id != item_id:
                continue
            if rule.contains is not None and (not item_id or rule.contains not in item_id):
                continue
            if rule.attempt is not None and rule.attempt != attempt:
                continue
            matched.append(rule)
        return matched

    def tick(self, op: str) -> int:
        self.op_counters[op] = self.op_counters.get(op, 0) + 1
        return self.op_counters[op]


def load_catalog_file(path: str | Path) -> tuple[str, list[CatalogSource]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    snapshot = data["catalog_snapshot"]
    sources: list[CatalogSource] = []
    snapshots: set[str] = set()
    for raw in data["sources"]:
        if raw["snapshot"] != snapshot:
            raise ValueError(
                f"catalog file {path}: source {raw['source_id']} snapshot "
                f"{raw['snapshot']!r} != shared snapshot {snapshot!r}"
            )
        source = CatalogSource.from_config(raw)
        if source.source_id in {s.source_id for s in sources}:
            raise ValueError(f"duplicate source_id {source.source_id} in catalog file")
        sources.append(source)
        snapshots.add(source.snapshot)
    if not sources:
        raise ValueError("catalog file must define at least one source")
    return snapshot, sources
