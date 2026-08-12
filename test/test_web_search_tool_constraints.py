from __future__ import annotations

import json
import unittest
from unittest import mock

from services.protocol import openai_v1_response, web_search_tool
from utils.helper import responses_sse_stream


class WebSearchToolConstraintTests(unittest.TestCase):
    def test_live_text_search_matches_current_codex_wire_shape(self) -> None:
        body = {
            "tools": [{
                "type": "web_search",
                "external_web_access": True,
                "indexed_web_access": False,
                "search_content_types": ["text"],
            }]
        }

        self.assertTrue(web_search_tool.has_web_search_tool(body))

    def test_optional_cached_search_is_omitted_instead_of_becoming_live_search(self) -> None:
        body = {
            "tools": [{
                "type": "web_search",
                "external_web_access": False,
            }]
        }

        self.assertFalse(web_search_tool.has_web_search_tool(body))

    def test_optional_cached_search_still_validates_its_wire_shape(self) -> None:
        body = {
            "tools": [{
                "type": "web_search",
                "external_web_access": False,
                "search_context_size": "oversized",
            }]
        }

        with self.assertRaises(web_search_tool.WebSearchConstraintError) as raised:
            web_search_tool.has_web_search_tool(body)

        self.assertEqual(
            raised.exception.to_openai_error()["error"]["param"],
            "tools[0].search_context_size",
        )

    def test_required_additional_function_does_not_force_cached_search(self) -> None:
        body = {
            "tool_choice": "required",
            "input": [{
                "type": "additional_tools",
                "role": "developer",
                "tools": [
                    {
                        "type": "web_search",
                        "external_web_access": False,
                    },
                    {
                        "type": "function",
                        "name": "read_file",
                        "parameters": {"type": "object"},
                    },
                ],
            }],
        }

        self.assertFalse(web_search_tool.has_web_search_tool(body))

    def test_required_only_cached_additional_search_is_rejected(self) -> None:
        body = {
            "tool_choice": "required",
            "input": [{
                "type": "additional_tools",
                "role": "developer",
                "tools": [{
                    "type": "web_search",
                    "external_web_access": False,
                }],
            }],
        }

        with self.assertRaises(web_search_tool.WebSearchConstraintError) as raised:
            web_search_tool.has_web_search_tool(body)

        self.assertEqual(
            raised.exception.to_openai_error()["error"]["param"],
            "input[0].tools[0].external_web_access",
        )

    def test_forced_cached_search_is_rejected_instead_of_becoming_live_search(self) -> None:
        body = {
            "tool_choice": {"type": "web_search"},
            "tools": [{
                "type": "web_search",
                "external_web_access": False,
            }],
        }

        with self.assertRaises(web_search_tool.WebSearchConstraintError) as raised:
            web_search_tool.has_web_search_tool(body)
        self.assertEqual(raised.exception.status_code, 400)
        error = raised.exception.to_openai_error()["error"]
        self.assertEqual(error["code"], "invalid_prompt")
        self.assertEqual(error["param"], "tools[0].external_web_access")
        self.assertIn("cached-only", error["message"])

    def test_nested_allowed_domains_are_rejected_before_search(self) -> None:
        body = {
            "input": [{
                "type": "additional_tools",
                "role": "developer",
                "tools": [{
                    "type": "namespace",
                    "name": "functions",
                    "tools": [{
                        "type": "web_search",
                        "filters": {"allowed_domains": ["openai.com"]},
                    }],
                }],
            }]
        }

        with self.assertRaises(web_search_tool.WebSearchConstraintError) as raised:
            web_search_tool.has_web_search_tool(body)

        error = raised.exception.to_openai_error()["error"]
        self.assertEqual(
            error["param"],
            "input[0].tools[0].tools[0].filters.allowed_domains",
        )
        self.assertIn("allowlist", error["message"])

    def test_unavailable_codex_constraints_fail_explicitly(self) -> None:
        cases = (
            (
                {"indexed_web_access": True},
                "tools[0].indexed_web_access",
                "indexed-only",
            ),
            (
                {"user_location": {"type": "approximate", "country": "US"}},
                "tools[0].user_location",
                "per-request search location",
            ),
            (
                {"search_content_types": ["text", "image"]},
                "tools[0].search_content_types",
                "only text results",
            ),
        )

        for fields, expected_param, expected_message in cases:
            with self.subTest(fields=fields):
                body = {"tools": [{"type": "web_search", **fields}]}
                with self.assertRaises(web_search_tool.WebSearchConstraintError) as raised:
                    web_search_tool.has_web_search_tool(body)
                error = raised.exception.to_openai_error()["error"]
                self.assertEqual(error["param"], expected_param)
                self.assertIn(expected_message, error["message"])

    def test_context_size_is_validated_and_forwarded_as_search_instruction(self) -> None:
        body = {
            "tools": [{
                "type": "web_search",
                "external_web_access": True,
                "search_context_size": "high",
            }]
        }
        self.assertTrue(web_search_tool.has_web_search_tool(body))

        backend = mock.Mock()
        backend.search.return_value = {"answer": "result", "sources": []}
        with (
            mock.patch.object(web_search_tool.account_service, "get_text_access_token", return_value="token"),
            mock.patch.object(web_search_tool.account_service, "mark_text_used") as mark_used,
            mock.patch.object(web_search_tool, "OpenAIBackendAPI", return_value=backend),
        ):
            result = web_search_tool.run_web_search("latest news")

        prompt = backend.search.call_args.args[0]
        self.assertTrue(prompt.startswith("latest news\n\n"))
        self.assertIn("search_context_size=high", prompt)
        self.assertEqual(result["answer"], "result")
        mark_used.assert_called_once_with("token")

    def test_unknown_constraint_is_not_silently_ignored(self) -> None:
        body = {
            "tools": [{
                "type": "web_search",
                "future_restriction": {"enabled": True},
            }]
        }

        with self.assertRaises(web_search_tool.WebSearchConstraintError) as raised:
            web_search_tool.has_web_search_tool(body)

        self.assertEqual(
            raised.exception.to_openai_error()["error"]["param"],
            "tools[0].future_restriction",
        )

    def test_responses_stream_returns_fatal_failure_without_calling_search(self) -> None:
        body = {
            "model": "gpt-5.6-luna",
            "stream": True,
            "input": "latest news",
            "tools": [{
                "type": "web_search",
                "external_web_access": False,
            }],
            "tool_choice": {"type": "web_search"},
        }

        with mock.patch("services.protocol.openai_v1_response.run_web_search") as search:
            wire = list(responses_sse_stream(openai_v1_response.handle(body)))

        search.assert_not_called()
        events = [
            json.loads(line[6:])
            for line in wire
            if line.startswith("data: {")
        ]
        failed = next(event for event in events if event["type"] == "response.failed")
        self.assertEqual(failed["response"]["error"]["code"], "invalid_prompt")
        self.assertIn("external_web_access", failed["response"]["error"]["message"])


if __name__ == "__main__":
    unittest.main()
