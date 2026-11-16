"""HTTP wrapper exposing the fusion engine.

Endpoints
---------
``POST /fuse``
    Body: a fusion request JSON. Returns 200 with ``status`` either
    ``complete`` (certified exact ranking) or ``incomplete`` (the honest
    answer after a directory fault). Malformed requests return 400.
``GET /healthz``
    Liveness probe.

Both completion states use HTTP 200 because the request itself succeeded;
the ``status`` field distinguishes the certified result from the explicit
incompleteness declaration.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from .engine import FusionEngine
from .errors import FusionError, RequestValidationError
from .request import build_request
from .types import FusionResult


async def run_fusion(data: dict[str, Any]) -> FusionResult:
    """Validate a request and run the engine."""
    request = build_request(data)
    engine = FusionEngine(request)
    return await engine.run()


class FusionHTTPServer:
    def __init__(self, *, host: str = "127.0.0.1") -> None:
        self.host = host
        self._server: Optional[asyncio.base_events.Server] = None

    @property
    def port(self) -> int:
        assert self._server is not None and self._server.sockets
        return self._server.sockets[0].getsockname()[1]

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.host, 0)

    async def serve_forever(self) -> None:
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    def close(self) -> None:
        if self._server is not None:
            self._server.close()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await reader.readline()
            if not request_line:
                return
            parts = request_line.decode("ascii").split()
            if len(parts) < 2:
                await self._send(writer, 400, {"error": "malformed request line"})
                return
            method, target = parts[0], parts[1]
            headers: dict[str, str] = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                if b":" in line:
                    name, value = line.decode("iso-8859-1").split(":", 1)
                    headers[name.strip().lower()] = value.strip()

            path = target.split("?", 1)[0]
            if method == "GET" and path == "/healthz":
                await self._send(writer, 200, {"status": "ok"})
                return
            if method != "POST" or path != "/fuse":
                await self._send(writer, 404, {"error": "use POST /fuse"})
                return

            length = int(headers.get("content-length", "0"))
            if length <= 0:
                raise RequestValidationError("missing request body")
            raw = await reader.readexactly(length)
            try:
                data = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RequestValidationError(f"invalid JSON body: {exc}") from exc

            result = await run_fusion(data)
            await self._send(writer, 200, result.to_dict())
        except RequestValidationError as exc:
            await self._send(writer, exc.http_status, exc.to_dict())
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        except FusionError as exc:  # defensive: engine normally returns incomplete
            await self._send(writer, exc.http_status, exc.to_dict())
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (OSError, ConnectionError):
                pass

    async def _send(self, writer: asyncio.StreamWriter, status: int, payload: dict[str, Any]) -> None:
        reason = {200: "OK", 400: "BAD REQUEST", 404: "NOT FOUND", 500: "INTERNAL ERROR",
                  502: "BAD GATEWAY"}.get(status, "STATUS")
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


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Exact rank-fusion HTTP service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args(argv)

    async def _run() -> None:
        service = FusionHTTPServer(host=args.host)
        service._server = await asyncio.start_server(  # noqa: SLF001 - CLI wiring
            service._handle, args.host, args.port
        )
        print(
            json.dumps(
                {"event": "ready", "base_url": f"http://{args.host}:{args.port}", "endpoint": "/fuse"}
            ),
            flush=True,
        )
        async with service._server:
            await service._server.serve_forever()

    asyncio.run(_run())
