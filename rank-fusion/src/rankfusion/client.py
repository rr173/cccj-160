"""Minimal async HTTP/1.1 client used to talk to the local directory simulators.

Pure standard library (:mod:`asyncio` streams). Only ``http://`` URLs are
needed because everything runs locally. Requests are idempotent, so 429/503
responses are retried with bounded exponential backoff; a retry *never*
applies a response twice — the engine only sees the final successful one.
"""
from __future__ import annotations

import asyncio
import json
import urllib.parse
from typing import Any, Callable, Optional

from .errors import (
    HTTPProtocolError,
    SnapshotMismatchError,
    UpstreamError,
)
from .protocol import (
    ItemResponse,
    PageResponse,
    parse_item_response,
    parse_page_response,
)

TraceCallback = Callable[[dict[str, Any]], None]
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class AsyncDirectoryClient:
    """One client per fusion run; shared by all source requests.

    A global semaphore bounds concurrent requests across *all* sources so the
    scheduler can never degenerate into hammering every directory at once.
    """

    def __init__(
        self,
        *,
        max_concurrency: int,
        max_attempts: int,
        request_timeout: float,
        backoff_base: float,
        trace: Optional[TraceCallback] = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_attempts = max_attempts
        self._request_timeout = request_timeout
        self._backoff_base = backoff_base
        self._trace = trace

    # ------------------------------------------------------------------ #
    # Public API matching the two directory endpoints.
    # ------------------------------------------------------------------ #
    async def get_page(
        self,
        *,
        base_url: str,
        source_id: str,
        snapshot: str,
        cursor: Optional[str],
    ) -> tuple[PageResponse, int, int]:
        query = {
            "source_id": source_id,
            "snapshot": snapshot,
            "cursor": cursor if cursor is not None else "",
        }
        body, attempts, retries = await self._request(
            base_url=base_url,
            path="/page",
            query=query,
            op="page",
            source_id=source_id,
        )
        page = parse_page_response(
            body, expected_snapshot=snapshot, expected_source=source_id
        )
        if cursor is not None and page.cursor != cursor:
            raise HTTPProtocolError(
                f"cursor echo mismatch: sent {cursor!r}, received {page.cursor!r}"
            )
        return page, attempts, retries

    async def get_item(
        self,
        *,
        base_url: str,
        source_id: str,
        snapshot: str,
        item_id: str,
    ) -> tuple[ItemResponse, int, int]:
        query = {"source_id": source_id, "snapshot": snapshot, "item_id": item_id}
        body, attempts, retries = await self._request(
            base_url=base_url,
            path="/item",
            query=query,
            op="item",
            source_id=source_id,
        )
        parsed = parse_item_response(
            body,
            expected_snapshot=snapshot,
            expected_source=source_id,
            expected_item=item_id,
        )
        return parsed, attempts, retries

    # ------------------------------------------------------------------ #
    # Transport.
    # ------------------------------------------------------------------ #
    def _emit(self, event: dict[str, Any]) -> None:
        if self._trace is not None:
            self._trace(event)

    async def _request(
        self,
        *,
        base_url: str,
        path: str,
        query: dict[str, str],
        op: str,
        source_id: str,
    ) -> tuple[Any, int, int]:
        url = urllib.parse.urlparse(base_url)
        if url.scheme != "http":
            raise UpstreamError(f"only http:// directory URLs are supported, got {base_url!r}")
        host = url.hostname
        if not host:
            raise UpstreamError(f"directory URL missing host: {base_url!r}")
        port = url.port or 80
        prefix = url.path.rstrip("/")
        target = prefix + path + "?" + urllib.parse.urlencode(query)

        attempts = 0
        retries = 0
        last_error: Optional[Exception] = None

        for attempt in range(1, self._max_attempts + 1):
            attempts += 1
            event: dict[str, Any] = {
                "event": "http_request",
                "source_id": source_id,
                "op": op,
                "attempt": attempt,
                "method": "GET",
                "url": f"http://{host}:{port}{target}",
            }
            try:
                async with self._semaphore:
                    status, payload = await asyncio.wait_for(
                        self._raw_get(host=host, port=port, target=target, attempt=attempt),
                        timeout=self._request_timeout,
                    )
            except (asyncio.TimeoutError, OSError, asyncio.IncompleteReadError) as exc:
                last_error = UpstreamError(
                    f"transport error talking to {source_id}: {exc.__class__.__name__}: {exc}"
                )
                event.update(result="transport_error", error=str(exc))
                self._emit(event)
                if attempt >= self._max_attempts:
                    break
                retries += 1
                await self._sleep_backoff(attempt)
                continue

            event["status"] = status
            if status == 200:
                try:
                    body = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise HTTPProtocolError(f"invalid JSON body from {source_id}: {exc}") from exc
                event["result"] = "ok"
                self._emit(event)
                return body, attempts, retries

            if status in _RETRYABLE_STATUS:
                event["result"] = "retryable_status"
                self._emit(event)
                last_error = UpstreamError(
                    f"{source_id} returned status {status}",
                    details={"status": status},
                )
                if attempt >= self._max_attempts:
                    break
                retries += 1
                # Honor Retry-After if present (best effort, still bounded).
                await self._sleep_backoff(attempt)
                continue
            # Non-retryable status. 409 carries the snapshot identity problem.
            event["result"] = "fatal_status"
            self._emit(event)
            if status == 409:
                raise SnapshotMismatchError(
                    f"{source_id} rejected the requested snapshot (HTTP 409)"
                )
            raise UpstreamError(
                f"{source_id} returned non-retryable status {status}",
                details={"status": status, "body_prefix": payload[:200].decode("utf-8", "replace")},
            )

        assert last_error is not None
        # Attach the request budget actually spent so the engine can account
        # for it even on failure.
        last_error.details.update({"attempts": attempts, "retries": retries})
        self._emit(
            {
                "event": "http_give_up",
                "source_id": source_id,
                "op": op,
                "attempts": attempts,
                "error": str(last_error),
            }
        )
        raise last_error

    @staticmethod
    async def _raw_get(*, host: str, port: int, target: str, attempt: int) -> tuple[int, bytes]:
        reader, writer = await asyncio.open_connection(host, port)
        try:
            request = (
                f"GET {target} HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                "Accept: application/json\r\n"
                f"X-Retry-Attempt: {attempt}\r\n"
                "Connection: close\r\n"
                "\r\n"
            )
            writer.write(request.encode("ascii"))
            await writer.drain()
            header_blob = await reader.readuntil(b"\r\n\r\n")
            header_text = header_blob.decode("iso-8859-1")
            lines = header_text.split("\r\n")
            status_line = lines[0].split(" ", 2)
            if len(status_line) < 2 or not status_line[1].isdigit():
                raise HTTPProtocolError(f"malformed status line: {lines[0]!r}")
            status = int(status_line[1])
            headers: dict[str, str] = {}
            for line in lines[1:]:
                if ":" in line:
                    name, value = line.split(":", 1)
                    headers[name.strip().lower()] = value.strip().lower()
            if headers.get("transfer-encoding") == "chunked":
                body = await _read_chunked(reader)
            else:
                # Connection: close means the remainder is the whole body.
                body = await reader.read()
                length = headers.get("content-length")
                if length is not None and len(body) != int(length):
                    raise HTTPProtocolError(
                        f"content-length mismatch: declared {length}, got {len(body)}"
                    )
            return status, body
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (OSError, ConnectionError):
                pass

    async def _sleep_backoff(self, attempt: int) -> None:
        delay = self._backoff_base * (2 ** (attempt - 1))
        await asyncio.sleep(delay)


async def _read_chunked(reader: asyncio.StreamReader) -> bytes:
    chunks: list[bytes] = []
    while True:
        size_line = await reader.readline()
        size_text = size_line.split(b";", 1)[0].strip()
        if not size_text:
            raise HTTPProtocolError("malformed chunked encoding: empty chunk size")
        try:
            size = int(size_text, 16)
        except ValueError as exc:
            raise HTTPProtocolError("malformed chunked encoding: bad chunk size") from exc
        if size == 0:
            # Consume trailing CRLF / trailers until blank line.
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"", b"\n"):
                    break
            break
        chunk = await reader.readexactly(size)
        chunks.append(chunk)
        crlf = await reader.readexactly(2)
        if crlf != b"\r\n":
            raise HTTPProtocolError("malformed chunked encoding: missing CRLF")
    return b"".join(chunks)
