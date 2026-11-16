"""Local HTTP directory simulator.

Serves the two contract endpoints for every source defined in one static
catalog file::

    GET /page?source_id=...&snapshot=...&cursor=...
    GET /item?source_id=...&snapshot=...&item_id=...

All responses are bound to the requested snapshot. A wrong snapshot is a hard
HTTP 409 so the fusion engine can never mix snapshots into one ranking.

Run standalone with::

    python -m datasim.server --catalog drills/scenarios/main.json --port 8080

Tests use :func:`start_simulator` to spin up an in-process server on an
ephemeral port.
"""
from __future__ import annotations

import asyncio
import json
import urllib.parse
from pathlib import Path
from typing import Any, Optional

from rankfusion.errors import RequestValidationError
from rankfusion.rational import rational_str

from .catalog import CatalogSource, FaultRule, encode_cursor, load_catalog_file


class SimulatorServer:
    def __init__(self, sources: list[CatalogSource], *, host: str = "127.0.0.1") -> None:
        self.host = host
        self.sources: dict[str, CatalogSource] = {s.source_id: s for s in sources}
        self.snapshots = {s.snapshot for s in sources}
        self._server: Optional[asyncio.base_events.Server] = None

    @property
    def port(self) -> int:
        assert self._server is not None and self._server.sockets
        return self._server.sockets[0].getsockname()[1]

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle_connection, self.host, 0)

    async def serve_forever(self) -> None:
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    def close(self) -> None:
        if self._server is not None:
            self._server.close()

    # ------------------------------------------------------------------ #
    # Connection / request plumbing.
    # ------------------------------------------------------------------ #
    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request_line = await reader.readline()
            if not request_line:
                return
            parts = request_line.decode("ascii").split()
            if len(parts) < 2 or parts[0] != "GET":
                await self._send(writer, 405, {"error": "only GET is supported"})
                return
            target = parts[1]
            headers: dict[str, str] = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                if b":" in line:
                    name, value = line.decode("iso-8859-1").split(":", 1)
                    headers[name.strip().lower()] = value.strip()
            parsed = urllib.parse.urlparse(target)
            query = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
            if parsed.path == "/healthz":
                await self._send(writer, 200, {"status": "ok"})
                return
            try:
                status, payload = await self._route(parsed.path, query, headers)
            except RequestValidationError as exc:
                status, payload = 400, {"error": exc.message, "code": exc.code}
            await self._send(writer, status, payload)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (OSError, ConnectionError):
                pass

    async def _route(
        self, path: str, query: dict[str, str], headers: dict[str, str]
    ) -> tuple[int, dict[str, Any]]:
        source_id = query.get("source_id", "")
        source = self.sources.get(source_id)
        if source is None:
            return 404, {"error": f"unknown source_id {source_id!r}"}
        snapshot = query.get("snapshot", "")
        if snapshot != source.snapshot:
            # Snapshot identity is enforced before any data leaves the server.
            return 409, {
                "error": "snapshot mismatch",
                "code": "snapshot_mismatch",
                "expected_snapshot": source.snapshot,
                "received_snapshot": snapshot,
            }
        attempt = int(headers.get("x-retry-attempt", "1"))

        if path == "/page":
            return await self._handle_page(source, query, attempt)
        if path == "/item":
            return await self._handle_item(source, query, attempt)
        return 404, {"error": f"unknown path {path!r}"}

    async def _handle_page(
        self, source: CatalogSource, query: dict[str, str], attempt: int
    ) -> tuple[int, dict[str, Any]]:
        nth = source.tick("page")
        cursor_raw = query.get("cursor") or None
        try:
            plan = source.plan_at(cursor_raw)
        except RequestValidationError as exc:
            return 400, {"error": exc.message, "code": exc.code}

        faults = source.match_faults(op="page", item_id=None, attempt=attempt)
        await self._apply_latency(faults)
        fatal = self._fatal_status(faults)
        if fatal is not None:
            return fatal, {
                "error": f"injected fault status {fatal}",
                "code": "injected_fault",
                "nth": nth,
            }

        bound_override = next((f.bad_bound for f in faults if f.bad_bound is not None), None)
        if bound_override is not None:
            upper_bound: Optional[str] = bound_override
        else:
            upper_bound = (
                rational_str(plan.upper_bound) if plan.upper_bound is not None else None
            )

        body = {
            "snapshot": source.snapshot,
            "source_id": source.source_id,
            "cursor": cursor_raw,
            "records": [
                {"item_id": item_id, "score": rational_str(score)}
                for item_id, score in plan.records
            ],
            "next_cursor": (
                encode_cursor(plan.next_offset, plan.next_slot)
                if plan.next_offset is not None
                else None
            ),
            "upper_bound": upper_bound,
        }
        return 200, body

    async def _handle_item(
        self, source: CatalogSource, query: dict[str, str], attempt: int
    ) -> tuple[int, dict[str, Any]]:
        nth = source.tick("item")
        item_id = query.get("item_id", "")
        if not item_id:
            raise RequestValidationError("item_id query parameter is required")
        faults = source.match_faults(op="item", item_id=item_id, attempt=attempt)
        await self._apply_latency(faults)
        fatal = self._fatal_status(faults)
        if fatal is not None:
            return fatal, {
                "error": f"injected fault status {fatal}",
                "code": "injected_fault",
                "nth": nth,
            }
        force_absent = any(f.bad_absent for f in faults)
        base = {
            "snapshot": source.snapshot,
            "source_id": source.source_id,
            "item_id": item_id,
        }
        if not force_absent and item_id in source.item_index:
            base.update(status="present", score=rational_str(source.item_index[item_id]))
        else:
            # Explicit absent with snapshot identity -> engine treats score as 0.
            base.update(status="absent", score=None)
        return 200, base

    @staticmethod
    async def _apply_latency(faults: list[FaultRule]) -> None:
        delays = [f.latency for f in faults if f.latency is not None]
        if delays:
            await asyncio.sleep(max(delays))

    @staticmethod
    def _fatal_status(faults: list[FaultRule]) -> Optional[int]:
        statuses = [f.status for f in faults if f.status is not None]
        return statuses[-1] if statuses else None

    async def _send(
        self, writer: asyncio.StreamWriter, status: int, payload: dict[str, Any]
    ) -> None:
        reason = {200: "OK", 400: "BAD REQUEST", 404: "NOT FOUND", 405: "METHOD NOT ALLOWED",
                  409: "CONFLICT", 429: "TOO MANY REQUESTS", 500: "INTERNAL ERROR",
                  502: "BAD GATEWAY", 503: "SERVICE UNAVAILABLE", 504: "GATEWAY TIMEOUT"}.get(
            status, "STATUS")
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        writer.write(
            (
                f"HTTP/1.1 {status} {reason}\r\n"
                "Content-Type: application/json; charset=utf-8\r\n"
                f"Content-Length: {len(body)}\r\n"
                "Connection: close\r\n"
                "\r\n"
            ).encode("ascii")
            + body
        )
        await writer.drain()


async def start_simulator(
    catalog_path: str | Path, *, host: str = "127.0.0.1"
) -> SimulatorServer:
    _, sources = load_catalog_file(catalog_path)
    server = SimulatorServer(sources, host=host)
    await server.start()
    return server


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Local HTTP directory simulator")
    parser.add_argument("--catalog", required=True, help="path to static catalog JSON")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)

    snapshot, sources = load_catalog_file(Path(args.catalog))
    server = SimulatorServer(sources, host=args.host)

    async def _run() -> None:
        server._server = await asyncio.start_server(  # noqa: SLF001 - CLI wiring
            server._handle_connection, args.host, args.port
        )
        actual_port = server._server.sockets[0].getsockname()[1]
        print(
            json.dumps(
                {
                    "event": "ready",
                    "base_url": f"http://{args.host}:{actual_port}",
                    "snapshot": snapshot,
                    "sources": [s.source_id for s in sources],
                }
            ),
            flush=True,
        )
        async with server._server:
            await server._server.serve_forever()

    asyncio.run(_run())
