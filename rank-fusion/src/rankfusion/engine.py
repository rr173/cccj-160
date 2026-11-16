"""Exact weighted rank-fusion engine.

The engine never converts anything to float. All scores, weights, bounds and
totals are :class:`fractions.Fraction` values, so ties and near-ties are
decided by exact arithmetic and the item_id Unicode code-point ordering.

Stopping strategy (NRA-style with an exact certificate)
-------------------------------------------------------
For every seen item ``i`` we maintain:

  lower(i) = sum over sources where the component is known
             of w_s * score_s(i)        (unknown components assumed 0)
  upper(i) = lower(i) + sum over sources where it is still unknown
             of w_s * bound_s
  H0       = sum over non-exhausted sources of w_s * bound_s

where ``bound_s`` is the last non-increasing upper bound promised by source
``s`` for its unread pages (0 once the terminal cursor proved exhaustion).
``H0`` is the best total any *completely unseen* item could still achieve.

Termination is only certified when:

  * at least K items have every component known (exact totals), and
  * every unresolved seen item is strictly behind the K-th exact item under
    the full ordering key ``(-total, item_id)`` *for every possible
    completion*, and
  * every unseen item is strictly behind it, i.e. ``H0 < T`` **or**
    ``H0 == T`` is impossible to certify (an unseen equal-score item with a
    smaller id could still win the tie-break) — so equality always keeps
    paging until exhaustion proves no such id exists.

Faults (transport failure, snapshot mismatch, observable bound/ordering
violation, point-vs-page score inconsistency) abort the run and return an
``incomplete`` result; partial candidates are never presented as final.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Optional

from .client import AsyncDirectoryClient
from .errors import BoundViolationError, FusionError
from .protocol import ItemResponse, PageResponse
from .rational import rational_str
from .types import (
    ComponentScore,
    FusionRequest,
    FusionResult,
    RankedItem,
    SourceStats,
    SourceSpec,
    StopBoundary,
)


def order_key(total: Fraction, item_id: str) -> tuple[Fraction, str]:
    """Sort key: higher total first, then item_id code-point ascending."""
    return (-total, item_id)


@dataclass
class _SourceRun:
    spec: SourceSpec
    stats: SourceStats
    exhausted: bool = False
    cursor: Optional[str] = None
    # Bound for unread pages. Before the first page the directory has made no
    # promise, so the conservative bound is the score-domain maximum 1.
    bound: Fraction = Fraction(1)
    started: bool = False
    inflight_page: bool = False
    seen_item_ids: set[str] = field(default_factory=set)
    # item_id -> exact Fraction observed on a page (sorted access)
    page_scores: dict[str, Fraction] = field(default_factory=dict)


@dataclass
class _Job:
    kind: str  # "page" | "item"
    source_id: str
    item_id: Optional[str] = None


@dataclass
class _ItemState:
    item_id: str
    # source_id -> Fraction for components known exactly (present score or absent=0)
    known: dict[str, Fraction] = field(default_factory=dict)
    # source_id -> Fraction observed via sorted access, pending confirmation
    page_only: dict[str, Fraction] = field(default_factory=dict)


class FusionEngine:
    def __init__(self, request: FusionRequest) -> None:
        self.request = request
        self.runs: dict[str, _SourceRun] = {
            spec.source_id: _SourceRun(spec=spec, stats=SourceStats(source_id=spec.source_id))
            for spec in request.sources
        }
        self.items: dict[str, _ItemState] = {}
        self.trace: list[dict[str, Any]] = []
        self._seq = 0
        self._inflight: set[tuple[str, Optional[str]]] = set()
        self._stop_info: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    # Entry point.
    # ------------------------------------------------------------------ #
    async def run(self) -> FusionResult:
        if self.request.k == 0:
            # Contract: no pagination may be read for K=0.
            return self._empty_result(reason="k_zero", notes="K=0 returns an empty ranking without reading any page")

        client = AsyncDirectoryClient(
            max_concurrency=self.request.max_concurrency,
            max_attempts=self.request.max_attempts,
            request_timeout=self.request.request_timeout,
            backoff_base=self.request.backoff_base,
            trace=self._emit_trace,
        )

        try:
            while True:
                stop = self._check_stop()
                if stop is not None:
                    self._stop_info = stop
                    return self._build_complete(stop)
                self._emit_stop_deferred()
                jobs = self._plan_jobs()
                if not jobs:
                    if all(run.exhausted for run in self.runs.values()):
                        # The terminal cursor of every source proves the union is
                        # fully enumerated. Any still-unknown component is an
                        # explicit absence (score 0) and page-only observations
                        # are final; complete the ranking without more requests.
                        return self._finalize_after_exhaustion()
                    # No progress possible and no certificate: fail honestly.
                    raise FusionError("no schedulable jobs but stop certificate is not available")
                await self._execute_batch(jobs, client)
        except FusionError as exc:
            return self._build_incomplete(exc)

    def _finalize_after_exhaustion(self) -> FusionResult:
        """Terminal cursors everywhere: turn page-only scores into known
        components and missing components into explicit zero absences."""
        for run in self.runs.values():
            for item_id, score in run.page_scores.items():
                st = self._state(item_id)
                st.known.setdefault(run.spec.source_id, score)
                st.page_only.pop(run.spec.source_id, None)
            for st in self.items.values():
                st.known.setdefault(run.spec.source_id, Fraction(0))
        # Now every seen item is resolved; H0 is 0 because every bound is 0.
        stop = self._check_stop()
        if stop is None and self.items:
            # K exceeds the universe: keep everything in exact order.
            exact = [(st, self._lower(st)) for st in self.items.values()]
            exact.sort(key=lambda pair: order_key(pair[1], pair[0].item_id))
            kth_state, kth_total = exact[-1]
            h0, breakdown = self._unseen_potential()
            stop = {
                "kth_item_id": kth_state.item_id,
                "kth_total": kth_total,
                "unseen_potential": h0,
                "breakdown": breakdown,
            }
        if stop is None:
            return self._empty_result(
                reason="all_sources_empty",
                notes="every source returned its terminal cursor with no records",
            )
        self._stop_info = stop
        return self._build_complete(stop)

    # ------------------------------------------------------------------ #
    # Item math.
    # ------------------------------------------------------------------ #
    def _state(self, item_id: str) -> _ItemState:
        if item_id not in self.items:
            self.items[item_id] = _ItemState(item_id=item_id)
        return self.items[item_id]

    def _lower(self, st: _ItemState) -> Fraction:
        total = Fraction(0)
        for source_id, score in st.known.items():
            total += self.runs[source_id].spec.weight * score
        return total

    def _range(self, st: _ItemState) -> tuple[Fraction, Fraction]:
        lower = Fraction(0)
        upper = Fraction(0)
        for run in self.runs.values():
            if run.spec.source_id in st.known:
                score = st.known[run.spec.source_id]
                lower += run.spec.weight * score
                upper += run.spec.weight * score
            else:
                upper += run.spec.weight * run.bound
        return lower, upper

    def _is_resolved(self, st: _ItemState) -> bool:
        return set(st.known) == set(self.runs)

    def _unseen_potential(self) -> tuple[Fraction, list[dict[str, Any]]]:
        breakdown: list[dict[str, Any]] = []
        total = Fraction(0)
        for run in self.runs.values():
            contribution = run.spec.weight * run.bound
            total += contribution
            breakdown.append(
                {
                    "source_id": run.spec.source_id,
                    "weight": rational_str(run.spec.weight),
                    "bound": rational_str(run.bound),
                    "weighted_bound": rational_str(contribution),
                    "exhausted": run.exhausted,
                }
            )
        return total, breakdown

    # ------------------------------------------------------------------ #
    # Stop certificate.
    # ------------------------------------------------------------------ #
    def _check_stop(self) -> Optional[dict[str, Any]]:
        k = self.request.k
        resolved = [(st, self._lower(st)) for st in self.items.values() if self._is_resolved(st)]
        if len(resolved) < k:
            return None
        resolved.sort(key=lambda pair: order_key(pair[1], pair[0].item_id))
        kth_state, kth_total = resolved[k - 1]
        kth_key = order_key(kth_total, kth_state.item_id)

        # Unresolved seen candidates: certifiable only if their *best possible*
        # key still ranks strictly behind the K-th exact item.
        blocking: list[dict[str, Any]] = []
        for st in self.items.values():
            if self._is_resolved(st):
                continue
            lower, upper = self._range(st)
            best_key = order_key(upper, st.item_id)
            # best_key < kth_key means even the most favorable completion loses.
            if not (best_key < kth_key):
                blocking.append(
                    {
                        "item_id": st.item_id,
                        "lower_bound": rational_str(lower),
                        "upper_bound": rational_str(upper),
                        "unknown_sources": [
                            sid for sid in self.runs if sid not in st.known
                        ],
                    }
                )
        if blocking:
            return None

        h0, breakdown = self._unseen_potential()
        # Equality cannot certify: an unseen item with the same total and a
        # smaller item_id would win the tie-break. Exhaustion drives H0 to 0.
        if h0 >= kth_total:
            return None

        return {
            "kth_item_id": kth_state.item_id,
            "kth_total": kth_total,
            "unseen_potential": h0,
            "breakdown": breakdown,
        }

    # ------------------------------------------------------------------ #
    # Scheduling.
    # ------------------------------------------------------------------ #
    def _plan_jobs(self) -> list[_Job]:
        capacity = self.request.max_concurrency
        jobs: list[_Job] = []

        # 1) Random access first: confirm every component of seen items that
        #    is still unknown. Point lookups never double-count (the inflight
        #    set and the known-map make each (item, source) pair fire once).
        for item_id, st in self.items.items():
            for source_id, run in self.runs.items():
                if source_id in st.known:
                    continue
                token = ("item", f"{item_id}\x1f{source_id}")
                if token in self._inflight:
                    continue
                jobs.append(_Job("item", source_id, item_id))
                if len(jobs) >= capacity:
                    return jobs

        # 2) Sorted access: at most one new page per source per batch, and only
        #    for sources whose remaining bound can still change the decision.
        for run in self.runs.values():
            if len(jobs) >= capacity:
                break
            if run.exhausted or run.inflight_page:
                continue
            if self._source_is_relevant(run):
                token = ("page", run.spec.source_id)
                if token not in self._inflight:
                    jobs.append(_Job("page", run.spec.source_id))
        return jobs

    def _source_is_relevant(self, run: _SourceRun) -> bool:
        k = self.request.k
        resolved_count = sum(1 for st in self.items.values() if self._is_resolved(st))
        if resolved_count < k:
            # Still need exact material; any non-exhausted source can hold it.
            return True
        stop = self._check_stop()
        if stop is not None:
            return False
        # The source is relevant while its weighted bound feeds any blocker:
        # either an unresolved seen item's upper range or the unseen H0.
        if run.bound <= 0:
            return False
        kth_total = self._kth_total()
        if kth_total is None:
            return True
        kth_id = self._kth_id()
        assert kth_id is not None
        h0, _ = self._unseen_potential()
        if h0 >= kth_total:
            return True
        for st in self.items.values():
            if self._is_resolved(st) or run.spec.source_id in st.known:
                continue
            _, upper = self._range(st)
            if order_key(upper, st.item_id) >= order_key(kth_total, kth_id):
                return True
        return False

    def _kth_total(self) -> Optional[Fraction]:
        resolved = [(st, self._lower(st)) for st in self.items.values() if self._is_resolved(st)]
        if len(resolved) < self.request.k:
            return None
        resolved.sort(key=lambda pair: order_key(pair[1], pair[0].item_id))
        return resolved[self.request.k - 1][1]

    def _kth_id(self) -> Optional[str]:
        resolved = [(st, self._lower(st)) for st in self.items.values() if self._is_resolved(st)]
        if len(resolved) < self.request.k:
            return None
        resolved.sort(key=lambda pair: order_key(pair[1], pair[0].item_id))
        return resolved[self.request.k - 1][0].item_id

    # ------------------------------------------------------------------ #
    # Batch execution.
    # ------------------------------------------------------------------ #
    async def _execute_batch(self, jobs: list[_Job], client: AsyncDirectoryClient) -> None:
        coros = []
        for job in jobs:
            if job.kind == "page":
                token = ("page", job.source_id)
                self._inflight.add(token)
                self.runs[job.source_id].inflight_page = True
                coros.append(self._run_page(job, client))
            else:
                token = ("item", f"{job.item_id}\x1f{job.source_id}")
                self._inflight.add(token)
                coros.append(self._run_item(job, client))

        results = await asyncio.gather(*coros, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                # First fatal error aborts; the incomplete result is built from it.
                if isinstance(result, FusionError):
                    raise result
                raise FusionError(f"unexpected engine failure: {result!r}") from result

    async def _run_page(self, job: _Job, client: AsyncDirectoryClient) -> None:
        run = self.runs[job.source_id]
        try:
            page, attempts, retries = await client.get_page(
                base_url=run.spec.base_url,
                source_id=job.source_id,
                snapshot=self.request.catalog_snapshot,
                cursor=run.cursor,
            )
            run.stats.http_attempts += attempts
            run.stats.retries += retries
            self._apply_page(run, page)
            self._emit_trace(
                {
                    "event": "page_applied",
                    "seq": self._next_seq(),
                    "source_id": job.source_id,
                    "cursor": run.cursor,
                    "records": [
                        {"item_id": r.item_id, "score": rational_str(r.score)}
                        for r in page.records
                    ],
                    "next_cursor": page.next_cursor,
                    "upper_bound": (
                        rational_str(page.upper_bound) if page.upper_bound is not None else None
                    ),
                    "attempts": attempts,
                    "retries": retries,
                }
            )
        except FusionError as exc:
            self._charge_failed_attempt(exc, run.stats)
            raise
        finally:
            self._inflight.discard(("page", job.source_id))
            run.inflight_page = False

    @staticmethod
    def _charge_failed_attempt(exc: FusionError, stats: SourceStats) -> None:
        # The client attaches the attempts it actually spent before giving up.
        attempts = exc.details.get("attempts")
        retries = exc.details.get("retries", 0)
        if isinstance(attempts, int):
            stats.http_attempts += attempts
            stats.retries += retries

    def _apply_page(self, run: _SourceRun, page: PageResponse) -> None:
        run.stats.pages_read += 1
        if not page.records:
            run.stats.empty_pages += 1

        # ---- Cross-page contract checks (observable trust boundary) ----
        if run.started:
            prev_bound = run.bound
            if page.upper_bound is not None and page.upper_bound > prev_bound:
                raise BoundViolationError(
                    f"source {run.spec.source_id}: upper_bound increased from "
                    f"{rational_str(prev_bound)} to {rational_str(page.upper_bound)}"
                )
            if page.records and page.records[0].score > prev_bound:
                raise BoundViolationError(
                    f"source {run.spec.source_id}: first record "
                    f"{page.records[0].item_id} scores {page.records[0].score} "
                    f"which exceeds the promised bound {rational_str(prev_bound)}"
                )
        else:
            # First page: an advertised bound may not exceed the legal domain.
            if page.upper_bound is not None and page.upper_bound > 1:
                raise BoundViolationError(
                    f"source {run.spec.source_id}: first upper_bound "
                    f"{page.upper_bound} exceeds score domain [0, 1]"
                )
            run.started = True

        # Cursor chain must advance; an empty page keeps the same offset but
        # must promise continuation, and only a null next_cursor terminates.
        if page.next_cursor is None:
            run.exhausted = True
            run.bound = Fraction(0)
        else:
            if page.upper_bound is None:
                raise BoundViolationError(
                    f"source {run.spec.source_id}: non-terminal page missing upper_bound"
                )
            if not run.exhausted and not page.records and page.next_cursor == run.cursor:
                # An empty page may legitimately keep the same offset; it must
                # however promise continuation, which is guaranteed by the
                # non-null upper_bound validated above.
                pass
            if not run.exhausted and page.records and page.next_cursor == run.cursor:
                raise BoundViolationError(
                    f"source {run.spec.source_id}: non-empty page did not advance cursor"
                )
            run.bound = page.upper_bound
        run.cursor = page.next_cursor
        run.stats.exhausted = run.exhausted
        run.stats.last_cursor = run.cursor
        run.stats.current_upper_bound = run.bound
        run.stats.current_weighted_bound = run.spec.weight * run.bound

        # ---- Materialize records (no double counting across pages) ----
        for record in page.records:
            if record.item_id in run.page_scores:
                raise BoundViolationError(
                    f"source {run.spec.source_id}: item {record.item_id} "
                    f"returned on two pages"
                )
            run.page_scores[record.item_id] = record.score
            run.seen_item_ids.add(record.item_id)
            st = self._state(record.item_id)
            st.page_only[run.spec.source_id] = record.score

    async def _run_item(self, job: _Job, client: AsyncDirectoryClient) -> None:
        run = self.runs[job.source_id]
        assert job.item_id is not None
        try:
            response, attempts, retries = await client.get_item(
                base_url=run.spec.base_url,
                source_id=job.source_id,
                snapshot=self.request.catalog_snapshot,
                item_id=job.item_id,
            )
            run.stats.http_attempts += attempts
            run.stats.retries += retries
            run.stats.point_lookups += 1
            self._apply_item(run, job.item_id, response)
            self._emit_trace(
                {
                    "event": "item_applied",
                    "seq": self._next_seq(),
                    "source_id": job.source_id,
                    "item_id": job.item_id,
                    "present": response.present,
                    "score": rational_str(response.score) if response.score is not None else None,
                    "attempts": attempts,
                    "retries": retries,
                }
            )
        except FusionError as exc:
            self._charge_failed_attempt(exc, run.stats)
            raise
        finally:
            self._inflight.discard(("item", f"{job.item_id}\x1f{job.source_id}"))

    def _apply_item(self, run: _SourceRun, item_id: str, response: ItemResponse) -> None:
        score = response.score if response.present else Fraction(0)
        # Consistency check against sorted access: point lookup must confirm
        # the exact same rational score for an item already seen on a page.
        if item_id in run.page_scores and run.page_scores[item_id] != score:
            from .errors import ScoreInconsistencyError

            raise ScoreInconsistencyError(
                f"source {run.spec.source_id} item {item_id}: page score "
                f"{rational_str(run.page_scores[item_id])} disagrees with point "
                f"lookup score {rational_str(score)}"
            )
        st = self._state(item_id)
        # Retries return the same logical response; assigning again is
        # idempotent and can never add the component twice.
        st.known[run.spec.source_id] = score
        st.page_only.pop(run.spec.source_id, None)

    # ------------------------------------------------------------------ #
    # Result construction.
    # ------------------------------------------------------------------ #
    def _build_complete(self, stop: dict[str, Any]) -> FusionResult:
        exact: list[tuple[_ItemState, Fraction]] = [
            (st, self._lower(st)) for st in self.items.values() if self._is_resolved(st)
        ]
        exact.sort(key=lambda pair: order_key(pair[1], pair[0].item_id))
        chosen = exact[: self.request.k]

        ranked: list[RankedItem] = []
        for st, total in chosen:
            components = [
                ComponentScore(
                    source_id=run.spec.source_id,
                    weight=run.spec.weight,
                    score=st.known[run.spec.source_id],
                    weighted=run.spec.weight * st.known[run.spec.source_id],
                    known=True,
                )
                for run in self.runs.values()
            ]
            ranked.append(RankedItem(item_id=st.item_id, total_score=total, components=components))

        boundary = StopBoundary(
            reason="certified_top_k",
            kth_item_id=stop["kth_item_id"],
            kth_total=stop["kth_total"],
            unseen_potential=stop["unseen_potential"],
            unseen_potential_breakdown=stop["breakdown"],
            blocking_seen=[],
            notes=(
                "All ranked items have every source component confirmed. Every "
                "unresolved/unseen candidate is strictly behind the K-th item "
                "for every possible completion; equal unseen potential would "
                "have forced continued paging."
            ),
        )
        return FusionResult(
            status="complete",
            catalog_snapshot=self.request.catalog_snapshot,
            k=self.request.k,
            items=ranked,
            source_stats=[run.stats for run in self.runs.values()],
            stop_boundary=boundary,
            observed_candidate_count=len(self.items),
            union_fully_enumerated=all(run.exhausted for run in self.runs.values()),
            universe_size=(
                len(self.items)
                if all(run.exhausted for run in self.runs.values())
                else None
            ),
            trace=list(self.trace),
        )

    def _empty_result(self, *, reason: str, notes: str) -> FusionResult:
        fully_enumerated = all(run.exhausted for run in self.runs.values())
        boundary = StopBoundary(
            reason=reason,
            kth_item_id=None,
            kth_total=None,
            unseen_potential=Fraction(0),
            unseen_potential_breakdown=[],
            blocking_seen=[],
            notes=notes,
        )
        return FusionResult(
            status="complete",
            catalog_snapshot=self.request.catalog_snapshot,
            k=0,
            items=[],
            source_stats=[run.stats for run in self.runs.values()],
            stop_boundary=boundary,
            observed_candidate_count=len(self.items),
            union_fully_enumerated=fully_enumerated,
            universe_size=len(self.items) if fully_enumerated else None,
            trace=list(self.trace),
        )

    def _build_incomplete(self, exc: FusionError) -> FusionResult:
        # Snapshot of the current boundary state: useful for the audit trail,
        # explicitly NOT a ranking.
        h0, breakdown = self._unseen_potential()
        unresolved = [
            {
                "item_id": st.item_id,
                "known_components": sorted(st.known),
                "page_only_components": sorted(st.page_only),
            }
            for st in self.items.values()
            if not self._is_resolved(st)
        ]
        boundary = StopBoundary(
            reason="aborted_incomplete",
            kth_item_id=None,
            kth_total=None,
            unseen_potential=h0,
            unseen_potential_breakdown=breakdown,
            blocking_seen=unresolved,
            notes=(
                "The run aborted before a stop certificate existed. The items "
                "field is empty by policy: partial candidates must never be "
                "served as the final top-K."
            ),
        )
        return FusionResult(
            status="incomplete",
            catalog_snapshot=self.request.catalog_snapshot,
            k=self.request.k,
            items=[],
            source_stats=[run.stats for run in self.runs.values()],
            stop_boundary=boundary,
            observed_candidate_count=len(self.items),
            union_fully_enumerated=all(run.exhausted for run in self.runs.values()),
            universe_size=None,
            incomplete_reason=exc.code,
            incomplete_details=exc.to_dict(),
            trace=list(self.trace),
        )

    def _emit_stop_deferred(self) -> None:
        """Record why no certificate exists at this round (audit trail)."""
        k = self.request.k
        resolved = [(st, self._lower(st)) for st in self.items.values() if self._is_resolved(st)]
        reasons: list[str] = []
        if len(resolved) < k:
            reasons.append("fewer_than_k_resolved_items")
        else:
            resolved.sort(key=lambda pair: order_key(pair[1], pair[0].item_id))
            kth_state, kth_total = resolved[k - 1]
            kth_key = order_key(kth_total, kth_state.item_id)
            blocking = 0
            for st in self.items.values():
                if self._is_resolved(st):
                    continue
                _, upper = self._range(st)
                if not (order_key(upper, st.item_id) < kth_key):
                    blocking += 1
            if blocking:
                reasons.append(f"{blocking}_unresolved_seen_candidate(s)")
            h0, _ = self._unseen_potential()
            if h0 >= kth_total:
                reasons.append(
                    "unseen_potential_ge_kth"
                    + ("(equal: smaller-id unseen item could win tie-break)" if h0 == kth_total else "")
                )
        self._emit_trace(
            {
                "event": "stop_deferred",
                "reasons": reasons,
                "resolved_count": len(resolved),
                "unseen_potential": rational_str(self._unseen_potential()[0]),
                "bounds": {
                    sid: rational_str(run.bound) for sid, run in self.runs.items()
                },
            }
        )

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _emit_trace(self, event: dict[str, Any]) -> None:
        if "seq" not in event:
            event["seq"] = self._next_seq()
        self.trace.append(event)
