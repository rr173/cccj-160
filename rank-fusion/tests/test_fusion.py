"""Test harness for the exact rank-fusion system.

The expected answers always come from ``oracle.load_expected`` (a fully
independent re-implementation over the static catalogs), never from the
product's boundary code. The simulator is started in-process on an ephemeral
port.
"""
from __future__ import annotations

import asyncio
import json
import sys
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from datasim.catalog import CatalogSource, load_catalog_file  # noqa: E402
from datasim.server import SimulatorServer  # noqa: E402
from oracle import float_ordering_differs, load_expected  # noqa: E402
from rankfusion.engine import FusionEngine  # noqa: E402
from rankfusion.errors import (  # noqa: E402
    BoundViolationError,
    SnapshotMismatchError,
)
from rankfusion.request import build_request  # noqa: E402
from rankfusion.service import FusionHTTPServer, run_fusion  # noqa: E402

SCEN = ROOT / "drills" / "scenarios"


def _load_request(name: str, base_url: str) -> dict:
    text = (SCEN / name).read_text(encoding="utf-8").replace("${BASE_URL}", base_url)
    return json.loads(text)


def _servers_for(catalog_name: str) -> SimulatorServer:
    _, sources = load_catalog_file(SCEN / catalog_name)
    return SimulatorServer(sources)


class AsyncTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.servers: list[SimulatorServer] = []

    async def asyncTearDown(self) -> None:
        for server in self.servers:
            server.close()

    async def start(self, catalog_name: str) -> tuple[SimulatorServer, str]:
        server = _servers_for(catalog_name)
        await server.start()
        self.servers.append(server)
        return server, server.base_url

    async def fuse(self, catalog_name: str, request_name: str) -> dict:
        _, base_url = await self.start(catalog_name)
        data = _load_request(request_name, base_url)
        engine = FusionEngine(build_request(data))
        result = await engine.run()
        self.last_engine = engine
        return result.to_dict()

    def assert_matches_oracle(self, result: dict, catalog_name: str, request: dict) -> None:
        self.assertEqual(result["status"], "complete", msg=json.dumps(result, indent=2))
        expected = load_expected(
            SCEN / catalog_name, request["weights"], request["k"]
        )
        if result["union_fully_enumerated"]:
            self.assertEqual(result["universe_size"], expected["universe_size"])
            self.assertEqual(
                result["observed_candidate_count"], expected["universe_size"]
            )
        else:
            # Early certified stop: observed set is a subset of the true union.
            self.assertIsNone(result["universe_size"])
            self.assertLessEqual(
                result["observed_candidate_count"], expected["universe_size"]
            )
        actual = [(i["item_id"], i["total_score"]) for i in result["items"]]
        wanted = [(i["item_id"], i["total"]) for i in expected["top_k"]]
        self.assertEqual(actual, wanted)
        # Every reported component must match the static catalog exactly.
        for item in result["items"]:
            row = next(r for r in expected["order"] if r["item_id"] == item["item_id"])
            for comp in item["components"]:
                self.assertEqual(
                    comp["score"], row["components"][comp["source_id"]]
                )
                w = Fraction(request["weights"][comp["source_id"]])
                self.assertEqual(
                    Fraction(comp["weighted_score"]),
                    w * Fraction(comp["score"]),
                )
                self.assertTrue(comp["known"])
        return expected


class TestRationalGuard(unittest.TestCase):
    def test_float_is_rejected(self) -> None:
        from rankfusion.rational import parse_rational, parse_score

        with self.assertRaises(ValueError):
            parse_rational(0.1)
        with self.assertRaises(ValueError):
            parse_score(0.5)

    def test_exact_strings(self) -> None:
        from rankfusion.rational import parse_rational

        self.assertEqual(parse_rational("3/5"), Fraction(3, 5))
        self.assertEqual(parse_rational("0.25"), Fraction(1, 4))
        self.assertEqual(parse_rational("1"), Fraction(1))
        with self.assertRaises(ValueError):
            parse_rational("1/0")
        with self.assertRaises(ValueError):
            parse_rational("nonsense")

    def test_score_domain(self) -> None:
        from rankfusion.rational import parse_score

        self.assertEqual(parse_score("0"), Fraction(0))
        self.assertEqual(parse_score("1"), Fraction(1))
        with self.assertRaises(ValueError):
            parse_score("101/100")
        with self.assertRaises(ValueError):
            parse_score("-1/100")


class TestMainExample(AsyncTestCase):
    async def test_c_beats_both_single_source_leaders(self) -> None:
        request = _load_request("main.request.json", "http://unused")
        result = await self.fuse("main.json", "main.request.json")
        expected = self.assert_matches_oracle(result, "main.json", request)
        # The headline assertion: c with exact total 6/5, not a or b.
        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["item_id"], "c")
        self.assertEqual(result["items"][0]["total_score"], "6/5")
        comps = {c["source_id"]: c for c in result["items"][0]["components"]}
        self.assertEqual(comps["src-a"]["score"], "3/5")
        self.assertEqual(comps["src-b"]["score"], "3/5")

        # Early termination certificate: both sources still have unread pages.
        stats = {s["source_id"]: s for s in result["source_stats"]}
        self.assertFalse(stats["src-a"]["exhausted"])
        self.assertFalse(stats["src-b"]["exhausted"])
        self.assertGreater(stats["src-a"]["pages_read"], 0)
        self.assertGreater(stats["src-b"]["pages_read"], 0)
        boundary = result["stop_boundary"]
        self.assertEqual(boundary["reason"], "certified_top_k")
        self.assertLess(
            Fraction(boundary["unseen_potential"]),
            Fraction(boundary["kth_total"]),
        )
        self.assertEqual(boundary["blocking_seen_candidates"], [])
        # At least one unread page must exist behind each source's cursor.
        self.assertIsNotNone(stats["src-a"]["last_cursor"])
        self.assertIsNotNone(stats["src-b"]["last_cursor"])
        self.assertIn(expected["top_k"][0]["item_id"], {"c"})


class TestTailTieDoesNotStopEarly(AsyncTestCase):
    async def test_equal_unseen_potential_forces_continuation(self) -> None:
        request = _load_request("tail_tie.request.json", "http://unused")
        result = await self.fuse("tail_tie.json", "tail_tie.request.json")
        self.assert_matches_oracle(result, "tail_tie.json", request)

        # The trace must show at least one round where H0 == the K-th total
        # and the engine explicitly refused to stop.
        deferred = [e for e in self.last_engine.trace if e.get("event") == "stop_deferred"]
        equality_rounds = [
            e for e in deferred
            if any("unseen_potential_ge_kth(equal" in r for r in e["reasons"])
        ]
        self.assertTrue(
            equality_rounds,
            msg="engine never observed H0 == kth_total; scenario lost its edge",
        )
        # And in every such round the recorded unseen_potential really was
        # exactly the K-th total (exact rational equality, no float involved).
        kth_total = Fraction(result["items"][0]["total_score"])
        for event in equality_rounds:
            self.assertEqual(Fraction(event["unseen_potential"]), kth_total)
        # Continuation read further pages; a source with tail data was paged
        # beyond its first page.
        total_pages = sum(s["pages_read"] for s in result["source_stats"])
        self.assertGreater(total_pages, 2)
        # Final winner under exact ordering (a1 < a2 code-point order).
        self.assertEqual(result["items"][0]["item_id"], "a1")
        self.assertEqual(result["items"][0]["total_score"], "1/2")


class TestPagingAndAbsence(AsyncTestCase):
    async def test_empty_pages_different_sizes_and_absent_lookups(self) -> None:
        request = _load_request("paging.request.json", "http://unused")
        result = await self.fuse("paging.json", "paging.request.json")
        self.assert_matches_oracle(result, "paging.json", request)
        stats = {s["source_id"]: s for s in result["source_stats"]}
        # Source p1 declares two empty page slots; the engine must survive them.
        self.assertGreaterEqual(stats["p1"]["empty_pages"], 1)
        # Items absent from a source required explicit point lookups (score 0).
        self.assertGreater(stats["p1"]["point_lookups"] + stats["p2"]["point_lookups"], 0)
        absent_ok = all(
            comp["score"] == "0"
            for item in result["items"]
            for comp in item["components"]
            if comp["score"] == "0"
        )
        self.assertTrue(absent_ok)


class TestSingleSourceAndAllEmpty(AsyncTestCase):
    async def test_single_real_source_mirror_empty(self) -> None:
        request = _load_request("single_source.request.json", "http://unused")
        result = await self.fuse("single_source.json", "single_source.request.json")
        expected = self.assert_matches_oracle(result, "single_source.json", request)
        # K exceeds the universe: every union item is returned, exact order.
        self.assertEqual(len(result["items"]), expected["universe_size"])
        stats = {s["source_id"]: s for s in result["source_stats"]}
        self.assertTrue(stats["empty-mirror"]["exhausted"])
        # Zero-score item still ranked by id tie-break, not dropped.
        ids = [i["item_id"] for i in result["items"]]
        self.assertIn("x4", ids)

    async def test_all_empty_catalog(self) -> None:
        request = _load_request("all_empty.request.json", "http://unused")
        result = await self.fuse("all_empty.json", "all_empty.request.json")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["items"], [])
        self.assertEqual(result["observed_candidate_count"], 0)
        self.assertTrue(result["union_fully_enumerated"])
        self.assertEqual(result["stop_boundary"]["reason"], "all_sources_empty")


class TestNearTieFloatConflation(AsyncTestCase):
    async def test_exact_arithmetic_separates_float_equal_totals(self) -> None:
        request = _load_request("near_tie.request.json", "http://unused")
        # Sanity: float arithmetic really does conflate this material.
        self.assertTrue(
            float_ordering_differs(SCEN / "near_tie.json", request["weights"], request["k"])
        )
        result = await self.fuse("near_tie.json", "near_tie.request.json")
        expected = self.assert_matches_oracle(result, "near_tie.json", request)
        # The two float-conflated totals remain distinct under exact arithmetic.
        ids = [i["item_id"] for i in expected["order"]]
        self.assertIn("aa-near", ids)
        self.assertIn("bb-far", ids)
        actual_rows = {i["item_id"]: i["total_score"] for i in result["items"]}
        self.assertNotEqual(
            actual_rows.get("aa-near"), actual_rows.get("bb-far")
        )
        self.assertEqual(actual_rows["aa-near"], "1/10")
        self.assertEqual(
            actual_rows["bb-far"], "10000000000000001/100000000000000000"
        )
        # And the strict exact ordering is bb-far before aa-near (its rational
        # is 1/10 + 1e-16), whereas float64 would tie them and flip on id only.
        result_ids = [i["item_id"] for i in result["items"]]
        if {"aa-near", "bb-far"} <= set(result_ids):
            self.assertLess(result_ids.index("bb-far"), result_ids.index("aa-near"))


class TestRetryAndOutOfOrder(AsyncTestCase):
    async def test_429_503_retried_without_double_counting(self) -> None:
        request = _load_request("retry_429.request.json", "http://unused")
        result = await self.fuse("retry_429.json", "retry_429.request.json")
        self.assert_matches_oracle(result, "retry_429.json", request)
        stats = {s["source_id"]: s for s in result["source_stats"]}
        # Retries happened.
        self.assertGreater(stats["src-a"]["retries"] + stats["src-b"]["retries"], 0)
        # But no score was double counted: each component appears once.
        for item in result["items"]:
            sources = [c["source_id"] for c in item["components"]]
            self.assertEqual(len(sources), len(set(sources)))
        # Trace shows the injected 429/503 followed by a successful retry.
        statuses = [
            e.get("status")
            for e in self.last_engine.trace
            if e.get("event") == "http_request"
        ]
        self.assertIn(429, statuses)
        self.assertIn(503, statuses)

    async def test_concurrent_responses_arrive_unordered_but_state_stays_coherent(self) -> None:
        # Different latency per source interleaves completions; correctness
        # must not depend on response order.
        _, sources = load_catalog_file(SCEN / "main.json")
        for source in sources:
            if source.source_id == "src-a":
                from datasim.catalog import FaultRule

                source.faults.append(FaultRule(op="page", nth=1, latency=0.2))
        server = SimulatorServer(sources)
        await server.start()
        self.servers.append(server)
        data = _load_request("main.request.json", server.base_url)
        data["max_concurrency"] = 2
        result = (await FusionEngine(build_request(data)).run()).to_dict()
        request = _load_request("main.request.json", "http://unused")
        self.assert_matches_oracle(result, "main.json", request)
        self.assertEqual(result["items"][0]["item_id"], "c")


class TestFaultsAbortIncomplete(AsyncTestCase):
    async def test_wrong_snapshot_fails_and_never_mixes(self) -> None:
        _, base_url = await self.start("main.json")
        data = _load_request("main.request.json", base_url)
        data["catalog_snapshot"] = "snap-does-not-exist"
        result = (await FusionEngine(build_request(data)).run()).to_dict()
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["incomplete_reason"], "snapshot_mismatch")
        self.assertEqual(result["items"], [])

    async def test_increasing_bound_is_a_contract_violation(self) -> None:
        result = await self.fuse("bad_bound.json", "bad_bound.request.json")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["incomplete_reason"], "bound_violation")
        # Partial work must not be presented as a ranking.
        self.assertEqual(result["items"], [])
        self.assertIn("aborted_incomplete", result["stop_boundary"]["reason"])

    async def test_persistent_503_exhausts_attempts(self) -> None:
        from datasim.catalog import FaultRule

        _, sources = load_catalog_file(SCEN / "main.json")
        sources[0].faults.append(FaultRule(op="page", status=503))
        server = SimulatorServer(sources)
        await server.start()
        self.servers.append(server)
        data = _load_request("main.request.json", server.base_url)
        data["max_attempts"] = 3
        result = (await FusionEngine(build_request(data)).run()).to_dict()
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["incomplete_reason"], "upstream_error")
        stats = {s["source_id"]: s for s in result["source_stats"]}
        self.assertEqual(stats["src-a"]["http_attempts"], 3)

    async def test_record_exceeding_promised_bound_violates_contract(self) -> None:
        from datasim.catalog import CatalogSource, FaultRule

        # K=3 with close scores forces the engine to page past the first page.
        # The first page dishonestly promises everything later is <= 1/100,
        # but the second page returns a 1/2 record: an observable overshoot.
        lying = CatalogSource.from_config(
            {
                "source_id": "lie",
                "snapshot": "snap-overshoot",
                "page_size": 1,
                "faults": [{"op": "page", "nth": 1, "bad_bound": "1/100"}],
                "entries": [
                    {"item_id": "i1", "score": "9/10"},
                    {"item_id": "i2", "score": "1/2"},
                    {"item_id": "i3", "score": "1/2"},
                ],
            }
        )
        honest = CatalogSource.from_config(
            {
                "source_id": "ok",
                "snapshot": "snap-overshoot",
                "page_size": 1,
                "entries": [
                    {"item_id": "i1", "score": "8/10"},
                    {"item_id": "i2", "score": "7/10"},
                    {"item_id": "i3", "score": "6/10"},
                ],
            }
        )
        server = SimulatorServer([lying, honest])
        await server.start()
        self.servers.append(server)
        data = {
            "catalog_snapshot": "snap-overshoot",
            "k": 3,
            "weights": {"lie": "1", "ok": "1"},
            "sources": [
                {"source_id": "lie", "base_url": server.base_url},
                {"source_id": "ok", "base_url": server.base_url},
            ],
            "max_concurrency": 1,
        }
        result = (await FusionEngine(build_request(data)).run()).to_dict()
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["incomplete_reason"], "bound_violation")
        self.assertEqual(result["items"], [])


class TestKZero(AsyncTestCase):
    async def test_k_zero_reads_no_pages(self) -> None:
        result = await self.fuse("main.json", "kzero.request.json")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["items"], [])
        self.assertTrue(all(s["pages_read"] == 0 for s in result["source_stats"]))
        self.assertTrue(all(s["point_lookups"] == 0 for s in result["source_stats"]))


class TestPointLookupAbsent(AsyncTestCase):
    async def test_absent_lookup_carries_snapshot_and_scores_zero(self) -> None:
        # An item present in one source but missing in the other must produce
        # an explicit snapshot-bound absent (score 0), not a missing component.
        result = await self.fuse("single_source.json", "single_source.request.json")
        self.assertEqual(result["status"], "complete")
        x1 = next(i for i in result["items"] if i["item_id"] == "x1")
        comps = {c["source_id"]: c for c in x1["components"]}
        # x1 exists only in "only"; the empty mirror reports explicit absent.
        self.assertEqual(comps["empty-mirror"]["score"], "0")
        self.assertEqual(comps["empty-mirror"]["weighted_score"], "0")
        self.assertTrue(comps["empty-mirror"]["known"])
        # The weighted total only reflects the real source component.
        self.assertEqual(x1["total_score"], str(Fraction(2, 3) * Fraction(3, 4)))

    async def test_forced_absent_fault_is_honored_as_zero(self) -> None:
        # The simulator can force an item lookup to report absent. For an item
        # never observed on that source's pages, the snapshot-bound absent is
        # accepted honestly as a zero component.
        from datasim.catalog import FaultRule

        _, sources = load_catalog_file(SCEN / "single_source.json")
        mirror = next(s for s in sources if s.source_id == "empty-mirror")
        # x1 lives only in "only"; the empty mirror's lookup is forced absent.
        mirror.faults.append(FaultRule(op="item", item_id="x1", bad_absent=True))
        server = SimulatorServer(sources)
        await server.start()
        self.servers.append(server)
        data = _load_request("single_source.request.json", server.base_url)
        result = (await FusionEngine(build_request(data)).run()).to_dict()
        self.assertEqual(result["status"], "complete")
        x1 = next(i for i in result["items"] if i["item_id"] == "x1")
        comps = {c["source_id"]: c for c in x1["components"]}
        self.assertEqual(comps["empty-mirror"]["score"], "0")
        self.assertTrue(comps["empty-mirror"]["known"])

    async def test_absent_contradicting_seen_page_score_is_rejected(self) -> None:
        # Forcing an absent for an item the same source already returned on a
        # page is a score inconsistency, not a legitimate zero.
        from datasim.catalog import FaultRule

        _, sources = load_catalog_file(SCEN / "single_source.json")
        only = next(s for s in sources if s.source_id == "only")
        only.faults.append(FaultRule(op="item", item_id="x1", bad_absent=True))
        server = SimulatorServer(sources)
        await server.start()
        self.servers.append(server)
        data = _load_request("single_source.request.json", server.base_url)
        result = (await FusionEngine(build_request(data)).run()).to_dict()
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["incomplete_reason"], "score_inconsistency")


class TestPointLookupInconsistency(AsyncTestCase):
    async def test_point_vs_page_disagreement_is_detected(self) -> None:
        # A point lookup for an item already seen on a page lies about its
        # score: the engine's consistency check must catch the disagreement.
        _, sources = load_catalog_file(SCEN / "main.json")
        source = next(s for s in sources if s.source_id == "src-a")
        original_index = source.item_index.copy()

        server = SimulatorServer(sources)
        original_route = server._route

        async def route(path, query, headers):  # type: ignore[no-untyped-def]
            sid = query.get("source_id", "")
            src = server.sources[sid]
            if query.get("snapshot", "") != src.snapshot:
                return 409, {"error": "snapshot mismatch"}
            if path == "/item" and sid == "src-a" and query.get("item_id") == "c":
                return 200, {
                    "snapshot": source.snapshot,
                    "source_id": source.source_id,
                    "item_id": "c",
                    "status": "present",
                    "score": "1/7",  # conflicts with the page score 3/5
                }
            if path == "/item":
                item_id = query.get("item_id", "")
                base = {
                    "snapshot": src.snapshot,
                    "source_id": sid,
                    "item_id": item_id,
                }
                if item_id in original_index:
                    base.update(status="present", score=str(original_index[item_id]))
                else:
                    base.update(status="absent", score=None)
                return 200, base
            return await original_route(path, query, headers)

        await server.start()
        self.servers.append(server)
        server._route = route  # type: ignore[method-assign]
        data = _load_request("main.request.json", server.base_url)
        result = (await FusionEngine(build_request(data)).run()).to_dict()
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["incomplete_reason"], "score_inconsistency")


class TestHTTPService(AsyncTestCase):
    async def _post(self, base_url: str, payload: bytes) -> tuple[int, dict]:
        port = int(base_url.rsplit(":", 1)[1])
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            writer.write(
                (
                    "POST /fuse HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                    f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
                ).encode()
                + payload
            )
            await writer.drain()
            blob = await reader.read()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
        header, _, body = blob.partition(b"\r\n\r\n")
        status = int(header.split(b"\r\n", 1)[0].split()[1])
        return status, json.loads(body.decode("utf-8"))

    async def test_fuse_endpoint_end_to_end(self) -> None:
        _, dir_url = await self.start("main.json")
        service = FusionHTTPServer()
        await service.start()
        self.addCleanup(service.close)
        data = _load_request("main.request.json", dir_url)
        status, payload = await self._post(service.base_url, json.dumps(data).encode("utf-8"))
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(payload["items"][0]["item_id"], "c")
        self.assertEqual(payload["items"][0]["total_score"], "6/5")

    async def test_service_rejects_bad_request(self) -> None:
        service = FusionHTTPServer()
        await service.start()
        self.addCleanup(service.close)
        status, payload = await self._post(service.base_url, b'{"k": -1}')
        self.assertEqual(status, 400)
        self.assertEqual(payload["code"], "request_validation_error")


class TestRequestValidation(unittest.TestCase):
    def test_weights_must_be_positive(self) -> None:
        with self.assertRaises(Exception):
            build_request(
                {
                    "catalog_snapshot": "s",
                    "k": 1,
                    "weights": {"a": "0"},
                    "sources": [{"source_id": "a", "base_url": "http://x"}],
                }
            )

    def test_float_weight_rejected(self) -> None:
        with self.assertRaises(Exception):
            build_request(
                {
                    "catalog_snapshot": "s",
                    "k": 1,
                    "weights": {"a": 0.5},
                    "sources": [{"source_id": "a", "base_url": "http://x"}],
                }
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
