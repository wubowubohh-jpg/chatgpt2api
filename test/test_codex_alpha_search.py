from __future__ import annotations

import json
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import ai as ai_module
from services.protocol import codex_search


class CodexAlphaSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        codex_search.clear_search_sessions()

    def test_request_prompt_preserves_commands_and_reference_context(self) -> None:
        body = {
            "commands": {
                "open": [{"ref_id": "turn0search0", "lineno": 12}],
                "find": [{"ref_id": "turn0search0", "pattern": "install"}],
                "response_length": "short",
            },
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Open the first result"}],
                },
            ],
            "settings": {"external_web_access": True},
        }

        prompt = codex_search.request_prompt(body)

        self.assertIn('"open":[{"ref_id":"turn0search0","lineno":12}]', prompt)
        self.assertIn('"find":[{"ref_id":"turn0search0","pattern":"install"}]', prompt)
        self.assertIn("Open the first result", prompt)
        self.assertIn('"external_web_access":true', prompt)

    def test_empty_search_query_is_a_noop(self) -> None:
        body = {
            "commands": {"search_query": [{"q": ""}]},
            "input": "this context must not turn an accidental call into a search",
        }

        self.assertEqual(codex_search.request_prompt(body), "")

        with mock.patch("services.protocol.codex_search.openai_search.handle") as search:
            response = codex_search.handle(body)

        search.assert_not_called()
        self.assertEqual(response, {"encrypted_output": None, "output": "", "results": []})

    def test_search_result_uses_codex_response_shape_and_reference_ids(self) -> None:
        result = {
            "answer": "Current answer.",
            "sources": [
                {"title": "Primary", "url": "https://example.com/a", "snippet": "A source"},
                {"title": "Duplicate", "url": "https://example.com/a", "snippet": "ignored"},
                {"title": "Secondary", "url": "https://example.com/b", "snippet": "B source"},
            ],
        }

        response = codex_search.response_from_result(result)

        self.assertIsNone(response["encrypted_output"])
        self.assertIn("[turn0search0] Primary", response["output"])
        self.assertIn("[turn0search1] Secondary", response["output"])
        self.assertEqual(
            [item["ref_id"] for item in response["results"]],
            ["turn0search0", "turn0search1"],
        )

    def test_handler_reuses_chatgpt_web_search_not_codex_upstream(self) -> None:
        body = {
            "id": "search-session",
            "model": "gpt-5.6-luna",
            "commands": {"search_query": [{"q": "OpenAI news"}]},
            "max_output_tokens": 1000,
        }
        result = {
            "answer": "News answer.",
            "sources": [{"title": "OpenAI", "url": "https://openai.com/news", "snippet": "News"}],
            "_account_email": "account@example.com",
        }

        with mock.patch("services.protocol.codex_search.openai_search.handle", return_value=result) as search:
            response = codex_search.handle(body)

        search.assert_called_once()
        self.assertIn("OpenAI news", search.call_args.args[0]["prompt"])
        self.assertEqual(response["encrypted_output"], None)
        self.assertEqual(response["_account_email"], "account@example.com")

    def test_restricted_search_modes_and_filters_fail_before_upstream(self) -> None:
        constrained = [
            {"settings": {"external_web_access": False}},
            {"settings": {"external_web_access": "cached"}},
            {"settings": {"external_web_access": "indexed"}},
            {"settings": {"filters": {"allowed_domains": ["openai.com"]}}},
            {"settings": {"user_location": {"type": "approximate", "country": "US"}}},
            {"settings": {"image_settings": {"max_results": 3}}},
            {"commands": {"search_query": [{"q": "news", "domains": ["openai.com"]}]}},
            {"commands": {"image_query": [{"q": "OpenAI logo"}]}},
        ]

        with mock.patch("services.protocol.codex_search.openai_search.handle") as search:
            for extra in constrained:
                with self.subTest(extra=extra), self.assertRaisesRegex(Exception, "cannot honor"):
                    codex_search.handle({
                        "commands": {"search_query": [{"q": "current news"}]},
                        **extra,
                    })

        search.assert_not_called()

    def test_live_search_context_size_is_preserved(self) -> None:
        body = {
            "commands": {"search_query": [{"q": "current news"}]},
            "settings": {
                "external_web_access": "live",
                "allowed_callers": ["direct"],
                "search_context_size": "high",
            },
        }

        prompt = codex_search.request_prompt(body)

        self.assertIn("Search context size requested by Codex: high", prompt)
        self.assertIn('"external_web_access":"live"', prompt)

    def test_session_reference_is_resolved_for_a_follow_up_open(self) -> None:
        first_result = {
            "answer": "First result.",
            "sources": [{"title": "Docs", "url": "https://example.com/docs", "snippet": ""}],
        }
        second_result = {
            "answer": "Opened result.",
            "sources": [{"title": "Install", "url": "https://example.com/install", "snippet": ""}],
        }
        with mock.patch(
            "services.protocol.codex_search.openai_search.handle",
            side_effect=[first_result, second_result],
        ) as search:
            first = codex_search.handle({
                "id": "session-ref-test",
                "commands": {"search_query": [{"q": "example docs"}]},
            })
            second = codex_search.handle({
                "id": "session-ref-test",
                "commands": {"open": [{"ref_id": "turn0search0"}]},
            })

        self.assertEqual(first["results"][0]["ref_id"], "turn0search0")
        self.assertIn('"ref_id":"https://example.com/docs"', search.call_args_list[1].args[0]["prompt"])
        self.assertEqual(second["results"][0]["ref_id"], "turn1search0")

    def test_unknown_reference_and_unsupported_commands_are_explicit(self) -> None:
        with mock.patch("services.protocol.codex_search.openai_search.handle") as search:
            unknown = codex_search.handle({
                "id": "missing-ref-test",
                "commands": {"find": [{"ref_id": "turn0search0", "pattern": "install"}]},
            })
            unsupported = codex_search.handle({
                "id": "unsupported-test",
                "commands": {"screenshot": [{"ref_id": "https://example.com", "pageno": 0}]},
            })

        search.assert_not_called()
        self.assertIn("Unknown search reference", unknown["output"])
        self.assertIn("does not support the screenshot operation", unsupported["output"])

    def test_search_references_are_isolated_by_api_identity(self) -> None:
        search_result = {
            "answer": "Private result.",
            "sources": [{"title": "Docs", "url": "https://example.com/private", "snippet": ""}],
        }
        with mock.patch(
            "services.protocol.codex_search.openai_search.handle",
            return_value=search_result,
        ) as search:
            first = codex_search.handle({
                "id": "shared-request-id",
                "commands": {"search_query": [{"q": "private docs"}]},
            }, owner_id="key-a")
            isolated = codex_search.handle({
                "id": "shared-request-id",
                "commands": {"open": [{"ref_id": "turn0search0"}]},
            }, owner_id="key-b")

        self.assertEqual(first["results"][0]["ref_id"], "turn0search0")
        self.assertIn("Unknown search reference", isolated["output"])
        self.assertEqual(search.call_count, 1)

    def test_v1_alpha_search_route_accepts_codex_wire_request(self) -> None:
        app = FastAPI()
        app.include_router(ai_module.create_router())
        client = TestClient(app)
        expected = {"encrypted_output": None, "output": "Search result", "results": []}

        for path in ("/alpha/search", "/v1/alpha/search"):
            with self.subTest(path=path):
                with (
                    mock.patch.object(ai_module, "require_identity", return_value={"id": "test", "name": "Test", "role": "user"}),
                    mock.patch.object(ai_module, "check_request", return_value=None),
                    mock.patch.object(ai_module.LoggedCall, "log", return_value=None),
                    mock.patch("services.protocol.codex_search.handle", return_value=expected) as handle,
                ):
                    response = client.post(
                        path,
                        headers={"Authorization": "Bearer test"},
                        json={
                            "id": "search-session",
                            "model": "gpt-5.6-luna",
                            "input": "Find current news",
                            "commands": {"search_query": [{"q": "current news"}]},
                            "settings": {"allowed_callers": ["direct"]},
                            "max_output_tokens": 2000,
                        },
                    )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), expected)
                payload = handle.call_args.args[0]
                self.assertEqual(payload["model"], "gpt-5.6-luna")
                self.assertEqual(payload["commands"]["search_query"][0]["q"], "current news")

    def test_codex_exec_search_and_response_continuation_end_to_end(self) -> None:
        app = FastAPI()
        app.include_router(ai_module.create_router())
        client = TestClient(app)
        input_items = [
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [
                    {
                        "type": "namespace",
                        "name": "functions",
                        "tools": [
                            {
                                "type": "custom",
                                "name": "exec",
                                "description": "Run JavaScript with nested tools.",
                            },
                        ],
                    },
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Search the web for current OpenAI news"}],
            },
        ]
        controller_outputs = iter([
            json.dumps({
                "action": "tool",
                "name": "exec",
                "input": (
                    "const result = await tools.web__run({"
                    "search_query:[{q:'current OpenAI news'}]}); text(result);"
                ),
            }),
            json.dumps({"action": "final", "text": "OpenAI published an update.", "complete": True}),
        ])

        def fake_controller_stream(_backend, _request):
            yield next(controller_outputs)

        def response_events(response):
            return [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: {")
            ]

        common_patches = (
            mock.patch.object(ai_module, "require_identity", return_value={"id": "test", "name": "Test", "role": "user"}),
            mock.patch.object(ai_module, "check_request", return_value=None),
            mock.patch.object(ai_module.LoggedCall, "log", return_value=None),
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch(
                "services.protocol.openai_v1_response.stream_text_deltas",
                side_effect=fake_controller_stream,
            ),
            mock.patch(
                "services.protocol.codex_search.openai_search.handle",
                return_value={
                    "answer": "OpenAI published an update.",
                    "sources": [
                        {"title": "OpenAI News", "url": "https://openai.com/news", "snippet": "Update"},
                    ],
                },
            ),
        )
        with common_patches[0], common_patches[1], common_patches[2], common_patches[3], common_patches[4], common_patches[5]:
            first_response = client.post(
                "/v1/responses",
                headers={"Authorization": "Bearer test"},
                json={"model": "gpt-5.6-luna", "stream": True, "input": input_items},
            )
            first_events = response_events(first_response)
            tool_call = next(
                event["item"]
                for event in first_events
                if event["type"] == "response.output_item.done"
                and event["item"]["type"] == "custom_tool_call"
            )

            search_response = client.post(
                "/alpha/search",
                headers={"Authorization": "Bearer test"},
                json={
                    "id": "search-e2e",
                    "model": "gpt-5.6-luna",
                    "commands": {"search_query": [{"q": "current OpenAI news"}]},
                },
            )
            continuation = [
                *input_items,
                tool_call,
                {
                    "type": "custom_tool_call_output",
                    "call_id": tool_call["call_id"],
                    "output": search_response.json()["output"],
                },
            ]
            final_response = client.post(
                "/v1/responses",
                headers={"Authorization": "Bearer test"},
                json={"model": "gpt-5.6-luna", "stream": True, "input": continuation},
            )
            final_events = response_events(final_response)

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(tool_call["name"], "exec")
        self.assertIn("tools.web__run", tool_call["input"])
        self.assertEqual(search_response.status_code, 200)
        self.assertIn("[turn0search0]", search_response.json()["output"])
        completed = next(event["response"] for event in final_events if event["type"] == "response.completed")
        self.assertEqual(completed["output"][0]["content"][0]["text"], "OpenAI published an update.")


if __name__ == "__main__":
    unittest.main()
