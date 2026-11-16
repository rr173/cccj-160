#!/usr/bin/env python3
"""Reproducible end-to-end drill runner.

For every scenario it:
  1. starts the local HTTP directory simulator on an ephemeral port,
  2. runs the fusion engine over real HTTP requests,
  3. independently recomputes the expected answer from the static catalog
     (the test oracle never imports product boundary code),
  4. writes the full result AND the complete request trace under drills/out.

Fault scenarios are expected to finish ``incomplete``; ranking scenarios must
match the oracle exactly. The script exits non-zero if any expectation breaks.

Run::

    python3 -I scripts/run_drills.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from datasim.catalog import load_catalog_file  # noqa: E402
from datasim.server import SimulatorServer  # noqa: E402
from oracle import float_ordering_differs, load_expected  # noqa: E402
from rankfusion.engine import FusionEngine  # noqa: E402
from rankfusion.request import build_request  # noqa: E402

SCEN = ROOT / "drills" / "scenarios"
OUT = ROOT / "drills" / "out"

# scenario catalog -> (request template, expected outcome kind)
SCENARIOS: dict[str, tuple[str, str]] = {
    "main.json": ("main.request.json", "ranking"),
    "tail_tie.json": ("tail_tie.request.json", "ranking"),
    "single_source.json": ("single_source.request.json", "ranking"),
    "all_empty.json": ("all_empty.request.json", "ranking"),
    "paging.json": ("paging.request.json", "ranking"),
    "near_tie.json": ("near_tie.request.json", "ranking"),
    "retry_429.json": ("retry_429.request.json", "ranking"),
    "bad_bound.json": ("bad_bound.request.json", "incomplete:bound_violation"),
}


def _load_request(name: str, base_url: str) -> dict:
    return json.loads((SCEN / name).read_text(encoding="utf-8").replace("${BASE_URL}", base_url))


async def run_one(catalog: str, request_file: str, expectation: str) -> dict:
    _, sources = load_catalog_file(SCEN / catalog)
    server = SimulatorServer(sources)
    await server.start()
    try:
        data = _load_request(request_file, server.base_url)
        engine = FusionEngine(build_request(data))
        result = await engine.run()
        payload = result.to_dict()
        trace = engine.trace

        check: dict = {"expectation": expectation, "ok": True, "problems": []}
        if expectation == "ranking":
            if payload["status"] != "complete":
                check["ok"] = False
                check["problems"].append(
                    f"expected complete, got {payload['status']} "
                    f"{payload.get('incomplete_reason')}"
                )
            else:
                expected = load_expected(SCEN / catalog, data["weights"], data["k"])
                actual = [(i["item_id"], i["total_score"]) for i in payload["items"]]
                wanted = [(i["item_id"], i["total"]) for i in expected["top_k"]]
                if actual != wanted:
                    check["ok"] = False
                    check["problems"].append(f"ranking mismatch: {actual} != {wanted}")
                # Exact component verification.
                for item in payload["items"]:
                    row = next(r for r in expected["order"] if r["item_id"] == item["item_id"])
                    for comp in item["components"]:
                        if comp["score"] != row["components"][comp["source_id"]]:
                            check["ok"] = False
                            check["problems"].append(
                                f"component mismatch for {item['item_id']}/{comp['source_id']}"
                            )
        else:
            kind, code = expectation.split(":", 1)
            if payload["status"] != "incomplete" or payload.get("incomplete_reason") != code:
                check["ok"] = False
                check["problems"].append(
                    f"expected incomplete/{code}, got {payload['status']}/"
                    f"{payload.get('incomplete_reason')}"
                )
            if payload["items"]:
                check["ok"] = False
                check["problems"].append("incomplete result must not carry ranked items")

        # K=0 style guard demonstrated through the dedicated request.
        record = {
            "catalog": catalog,
            "request": request_file,
            "snapshot": data["catalog_snapshot"],
            "k": data["k"],
            "weights": data["weights"],
            "result": payload,
            "check": check,
        }
        stem = catalog.replace(".json", "")
        (OUT / f"{stem}.result.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (OUT / f"{stem}.trace.json").write_text(
            json.dumps(
                {
                    "catalog": catalog,
                    "snapshot": data["catalog_snapshot"],
                    "event_count": len(trace),
                    "trace": trace,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return record
    finally:
        server.close()


async def run_k_zero_drill() -> dict:
    _, sources = load_catalog_file(SCEN / "main.json")
    server = SimulatorServer(sources)
    await server.start()
    try:
        data = _load_request("kzero.request.json", server.base_url)
        engine = FusionEngine(build_request(data))
        result = await engine.run()
        payload = result.to_dict()
        check = {
            "expectation": "k_zero_no_io",
            "ok": payload["status"] == "complete"
            and payload["items"] == []
            and all(s["pages_read"] == 0 and s["point_lookups"] == 0 for s in payload["source_stats"]),
            "problems": [] if payload["items"] == [] else ["K=0 returned items"],
        }
        record = {"catalog": "main.json", "request": "kzero.request.json",
                  "result": payload, "check": check}
        (OUT / "kzero.result.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (OUT / "kzero.trace.json").write_text(
            json.dumps({"trace": engine.trace, "event_count": len(engine.trace)}, indent=2),
            encoding="utf-8",
        )
        return record
    finally:
        server.close()


async def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    # Float-conflation guard proof, recorded once for the near-tie material.
    near_req = _load_request("near_tie.request.json", "http://unused")
    float_flips = float_ordering_differs(
        SCEN / "near_tie.json", near_req["weights"], near_req["k"]
    )
    (OUT / "float_conflation_proof.json").write_text(
        json.dumps(
            {
                "note": "float64 ordering differs from the exact Fraction ordering",
                "float_ordering_differs_from_exact": float_flips,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for catalog, (request_file, expectation) in SCENARIOS.items():
        record = await run_one(catalog, request_file, expectation)
        summary.append(
            {
                "scenario": catalog,
                "status": record["result"]["status"],
                "ok": record["check"]["ok"],
                "problems": record["check"]["problems"],
            }
        )
    kzero = await run_k_zero_drill()
    summary.append({"scenario": "kzero", "status": kzero["result"]["status"],
                    "ok": kzero["check"]["ok"], "problems": kzero["check"]["problems"]})

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    width = max(len(row["scenario"]) for row in summary)
    all_ok = True
    for row in summary:
        mark = "PASS" if row["ok"] else "FAIL"
        all_ok &= row["ok"]
        print(f"[{mark}] {row['scenario'].ljust(width)}  status={row['status']}")
        for problem in row["problems"]:
            print(f"        - {problem}")
    print(f"\nfloat64 conflation proven on near_tie material: {float_flips}")
    print(f"artifacts written under {OUT}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
