from __future__ import annotations

import json
import os
import unittest

import zstandard
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.request_compression import ZstdRequestMiddleware
from services.protocol.openai_v1_models import list_codex_models


class CodexRequestCompressionTests(unittest.TestCase):
    @staticmethod
    def _app() -> FastAPI:
        app = FastAPI()
        app.add_middleware(ZstdRequestMiddleware)

        @app.post("/echo")
        async def echo(payload: dict[str, object]):
            return payload

        return app

    def test_zstd_body_is_decoded_before_fastapi_validation(self) -> None:
        payload = {"model": "gpt-5.6-luna", "input": [{"role": "user", "content": "hi"}]}
        encoded = zstandard.ZstdCompressor(level=3).compress(
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
        )

        with TestClient(self._app()) as client:
            response = client.post(
                "/echo",
                content=encoded,
                headers={
                    "content-type": "application/json",
                    "content-encoding": "zstd",
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), payload)

    def test_invalid_zstd_body_is_rejected_without_a_json_traceback(self) -> None:
        with TestClient(self._app()) as client:
            response = client.post(
                "/echo",
                content=b"not-zstd",
                headers={
                    "content-type": "application/json",
                    "content-encoding": "zstd",
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "invalid_request_compression")

    def test_compressed_body_size_is_bounded_before_decompression(self) -> None:
        payload = os.urandom(16 * 1024 * 1024 + 1)
        encoded = zstandard.ZstdCompressor(level=1).compress(payload)
        with TestClient(self._app()) as client:
            response = client.post(
                "/echo",
                content=encoded,
                headers={"content-type": "application/json", "content-encoding": "zstd"},
            )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "invalid_request_compression")

    def test_codex_models_wrapper_uses_models_key(self) -> None:
        self.assertEqual(list_codex_models(), {"models": []})
