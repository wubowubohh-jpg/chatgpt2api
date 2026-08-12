from __future__ import annotations

import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.ai import create_router
from services.protocol import openai_v1_response


class CodexHttpContractTests(unittest.TestCase):
    @staticmethod
    def _app() -> FastAPI:
        app = FastAPI()
        app.include_router(create_router())
        return app

    def test_codex_models_query_uses_models_shape_and_etag(self) -> None:
        with mock.patch("api.ai.require_identity", return_value={"id": "test", "role": "admin"}):
            with TestClient(self._app()) as client:
                response = client.get(
                    "/v1/models?client_version=1.2.3",
                    headers={"authorization": "Bearer test"},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"models": []})
        self.assertEqual(response.headers.get("etag"), '"chatgpt2api-codex-models-v1"')

    def test_normal_models_query_keeps_openai_data_shape(self) -> None:
        catalog = {"object": "list", "data": [{"id": "gpt-5.6-luna", "object": "model"}]}
        with (
            mock.patch("api.ai.require_identity", return_value={"id": "test", "role": "admin"}),
            mock.patch("api.ai.openai_v1_models.list_models", return_value=catalog),
        ):
            with TestClient(self._app()) as client:
                response = client.get("/v1/models", headers={"authorization": "Bearer test"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), catalog)

    def test_codex_root_models_alias_matches_base_url_without_v1(self) -> None:
        with mock.patch("api.ai.require_identity", return_value={"id": "test", "role": "admin"}):
            with TestClient(self._app()) as client:
                response = client.get(
                    "/models?client_version=1.2.3",
                    headers={"authorization": "Bearer test"},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"models": []})

    def test_responses_headers_match_codex_presence_semantics(self) -> None:
        emitted: dict[str, str] = {}

        def response_events(payload):
            response_id = str(payload["_response_id"])
            emitted["response_id"] = response_id
            yield openai_v1_response.response_created(response_id, "gpt-5.6-luna", 1)
            yield openai_v1_response.response_completed(
                response_id,
                "gpt-5.6-luna",
                1,
                [],
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                end_turn=True,
            )

        with (
            mock.patch("api.ai.require_identity", return_value={"id": "test", "role": "admin"}),
            mock.patch("api.ai.filter_or_log", new=mock.AsyncMock()),
            mock.patch("api.ai.openai_v1_response.handle", side_effect=response_events),
            mock.patch("services.log_service.LoggedCall.log"),
        ):
            with TestClient(self._app()) as client:
                response = client.post(
                    "/v1/responses",
                    headers={"authorization": "Bearer test"},
                    json={"model": "gpt-5.6-luna", "stream": True, "input": "hi"},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("openai-model"), "gpt-5.6-luna")
        self.assertEqual(response.headers.get("x-request-id"), emitted["response_id"])
        turn_state = response.headers.get("x-codex-turn-state") or ""
        self.assertTrue(turn_state.startswith("turn_"))
        self.assertNotEqual(turn_state, emitted["response_id"])
        self.assertEqual(response.headers.get("x-models-etag"), '"chatgpt2api-codex-models-v1"')
        self.assertNotIn("x-reasoning-included", response.headers)

    def test_compact_echoes_turn_state_and_returns_compaction_item(self) -> None:
        with (
            mock.patch("api.ai.require_identity", return_value={"id": "test", "role": "admin"}),
            mock.patch("api.ai.filter_or_log", new=mock.AsyncMock()),
            mock.patch("services.log_service.LoggedCall.log"),
            mock.patch.object(openai_v1_response, "_upstream_compaction_summary", return_value=""),
        ):
            with TestClient(self._app()) as client:
                response = client.post(
                    "/v1/responses/compact",
                    headers={
                        "authorization": "Bearer test",
                        "x-codex-turn-state": "sticky-turn-state",
                    },
                    json={
                        "model": "gpt-5.6-luna",
                        "input": [{"type": "message", "role": "user", "content": "continue"}],
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("x-codex-turn-state"), "sticky-turn-state")
        self.assertEqual(response.json()["output"][0]["type"], "compaction")

    def test_codex_root_compact_alias_is_available(self) -> None:
        with (
            mock.patch("api.ai.require_identity", return_value={"id": "test", "role": "admin"}),
            mock.patch("api.ai.filter_or_log", new=mock.AsyncMock()),
            mock.patch("services.log_service.LoggedCall.log"),
            mock.patch.object(openai_v1_response, "_upstream_compaction_summary", return_value="summary"),
        ):
            with TestClient(self._app()) as client:
                response = client.post(
                    "/responses/compact",
                    headers={"authorization": "Bearer test"},
                    json={"model": "gpt-5.6-luna", "input": "continue"},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["output"][0]["type"], "compaction")

    def test_memories_endpoint_returns_codex_output_shape(self) -> None:
        with (
            mock.patch("api.ai.require_identity", return_value={"id": "test", "role": "admin"}),
            mock.patch("api.ai.filter_or_log", new=mock.AsyncMock()),
            mock.patch("services.log_service.LoggedCall.log"),
        ):
            with TestClient(self._app()) as client:
                response = client.post(
                    "/v1/memories/trace_summarize",
                    headers={"authorization": "Bearer test"},
                    json={
                        "model": "gpt-5.6-luna",
                        "traces": [{
                            "id": "trace-1",
                            "metadata": {"source_path": "rollout.jsonl"},
                            "items": [{"type": "message", "role": "user", "content": "inspect"}],
                        }],
                    },
                )

        self.assertEqual(response.status_code, 200)
        output = response.json()["output"]
        self.assertEqual(len(output), 1)
        self.assertIn("trace_summary", output[0])
        self.assertIn("memory_summary", output[0])
        self.assertIn("trace-1", output[0]["memory_summary"])

    def test_memories_endpoint_rejects_oversized_trace_batch(self) -> None:
        with (
            mock.patch("api.ai.require_identity", return_value={"id": "test", "role": "admin"}),
            mock.patch("api.ai.filter_or_log", new=mock.AsyncMock()),
            mock.patch("services.log_service.LoggedCall.log"),
        ):
            with TestClient(self._app()) as client:
                response = client.post(
                    "/v1/memories/trace_summarize",
                    headers={"authorization": "Bearer test"},
                    json={"traces": [{"items": [{"content": "x" * 600000}]}]},
                )
        self.assertEqual(response.status_code, 413)


if __name__ == "__main__":
    unittest.main()
