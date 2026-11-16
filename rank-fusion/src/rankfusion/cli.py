"""Command line entry point for a single fusion run.

Usage::

    python -m rankfusion.cli --request drills/scenarios/main_request.json \\
        --trace-out drills/out/main_trace.json --out drills/out/main_result.json

The request JSON follows the fusion API schema. The tool exits with code 0
even when the result is ``incomplete`` (that is a valid, honest response);
malformed requests exit with code 2.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional

from .engine import FusionEngine
from .errors import RequestValidationError
from .request import build_request


async def _run(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.request).read_text(encoding="utf-8"))
    try:
        fusion_request = build_request(data)
    except RequestValidationError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    engine = FusionEngine(fusion_request)
    result = await engine.run()
    payload = result.to_dict()

    if args.out:
        Path(args.out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.trace_out:
        Path(args.trace_out).write_text(
            json.dumps(
                {
                    "catalog_snapshot": fusion_request.catalog_snapshot,
                    "k": fusion_request.k,
                    "trace": result.trace,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Exact weighted rank fusion")
    parser.add_argument("--request", required=True, help="fusion request JSON file")
    parser.add_argument("--out", help="write the full result JSON here")
    parser.add_argument("--trace-out", help="write the full request trace here")
    args = parser.parse_args(argv)
    raise SystemExit(asyncio.run(_run(args)))
