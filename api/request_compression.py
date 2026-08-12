from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import zstandard


MAX_DECOMPRESSED_REQUEST_BYTES = 64 * 1024 * 1024
MAX_COMPRESSED_REQUEST_BYTES = 16 * 1024 * 1024


def _header_value(headers: list[tuple[bytes, bytes]], name: bytes) -> str:
    name = name.lower()
    for key, value in headers:
        if key.lower() == name:
            return value.decode("latin-1").strip().lower()
    return ""


def _without_headers(headers: list[tuple[bytes, bytes]], *names: bytes) -> list[tuple[bytes, bytes]]:
    ignored = {name.lower() for name in names}
    return [(key, value) for key, value in headers if key.lower() not in ignored]


class ZstdRequestMiddleware:
    """Decode Codex's optional zstd HTTP request bodies before FastAPI parses JSON."""

    def __init__(
        self,
        app: Callable[[dict[str, Any], Callable[..., Awaitable[dict[str, Any]]], Callable[..., Awaitable[None]]], Awaitable[None]],
        *,
        max_decompressed_bytes: int = MAX_DECOMPRESSED_REQUEST_BYTES,
    ) -> None:
        self.app = app
        self.max_decompressed_bytes = max(1, int(max_decompressed_bytes))

    async def __call__(self, scope: dict[str, Any], receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = list(scope.get("headers") or [])
        encoding = _header_value(headers, b"content-encoding")
        if not encoding:
            await self.app(scope, receive, send)
            return
        encodings = {part.strip() for part in encoding.split(",") if part.strip()}
        if encodings != {"zstd"}:
            await self.app(scope, receive, send)
            return

        chunks: list[bytes] = []
        compressed_size = 0
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                return
            if message.get("type") != "http.request":
                continue
            chunk = message.get("body") or b""
            if chunk:
                compressed_size += len(chunk)
                if compressed_size > MAX_COMPRESSED_REQUEST_BYTES:
                    await self._send_error(send, 413, "Compressed request body is too large")
                    return
                chunks.append(bytes(chunk))
            if not message.get("more_body", False):
                break

        compressed = b"".join(chunks)
        try:
            body = zstandard.ZstdDecompressor().decompress(
                compressed,
                max_output_size=self.max_decompressed_bytes,
            )
        except (zstandard.ZstdError, ValueError, MemoryError) as exc:
            await self._send_error(send, 400, "Invalid zstd request body")
            return

        rewritten_scope = dict(scope)
        rewritten_scope["headers"] = _without_headers(
            headers,
            b"content-encoding",
            b"content-length",
            b"transfer-encoding",
        ) + [(b"content-length", str(len(body)).encode("ascii"))]
        delivered = False

        async def replay_receive() -> dict[str, Any]:
            nonlocal delivered
            if delivered:
                return {"type": "http.request", "body": b"", "more_body": False}
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(rewritten_scope, replay_receive, send)

    @staticmethod
    async def _send_error(send, status_code: int, message: str) -> None:
        payload = json.dumps(
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": "invalid_request_compression",
                }
            },
            ensure_ascii=False,
        ).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode("ascii")),
            ],
        })
        await send({"type": "http.response.body", "body": payload})
