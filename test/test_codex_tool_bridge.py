from __future__ import annotations

import json
import unittest
from unittest import mock

from services.config import config
from services.openai_backend_api import OpenAIBackendAPI
from services.protocol import codex_conversation_session, codex_tool_bridge, openai_v1_response
from services.protocol.conversation import iter_conversation_payloads
from utils.helper import UpstreamHTTPError, responses_sse_stream


EXEC_TOOL = {
    "type": "custom",
    "name": "exec",
    "description": "Run JavaScript that can call tools.shell_command.",
    "format": {
        "type": "grammar",
        "syntax": "lark",
        "definition": "start: /.+/",
    },
}


def codex_body(*items, tools=None):
    return {
        "model": "gpt-5.6-luna",
        "stream": True,
        "input": list(items),
        **({"tools": tools} if tools is not None else {}),
    }


class CodexToolBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        codex_conversation_session.clear_controller_sessions()
        self.old_cache_settings = config.data.get("chat_completion_cache")
        config.data["chat_completion_cache"] = {
            "enabled": True,
            "ttl_seconds": 60,
            "max_entries": 32,
            "dedupe_inflight": True,
            "stream_cache": True,
            "normalize_messages": True,
            "drop_adjacent_duplicates": True,
            "drop_assistant_history": False,
        }

    def tearDown(self) -> None:
        codex_conversation_session.clear_controller_sessions()
        if self.old_cache_settings is None:
            config.data.pop("chat_completion_cache", None)
        else:
            config.data["chat_completion_cache"] = self.old_cache_settings

    def test_extracts_luna_lite_default_and_named_namespace_tools(self) -> None:
        body = codex_body({
            "type": "additional_tools",
            "role": "developer",
            "tools": [
                {"type": "namespace", "name": "functions", "description": "Default", "tools": [
                    EXEC_TOOL,
                    {"type": "function", "name": "wait", "description": "Wait", "parameters": {"type": "object"}},
                ]},
                {"type": "namespace", "name": "collaboration", "description": "Agents", "tools": [
                    {"type": "function", "name": "send_message", "parameters": {"type": "object"}},
                ]},
                {"type": "tool_search", "execution": "client", "parameters": {"type": "object"}},
            ],
        })

        tools = codex_tool_bridge.response_client_tools(body)

        self.assertEqual(
            [(tool["kind"], tool["namespace"], tool["name"]) for tool in tools],
            [
                ("custom", None, "exec"),
                ("function", None, "wait"),
                ("function", "collaboration", "send_message"),
                ("tool_search", None, "tool_search"),
            ],
        )
        self.assertEqual(tools[0]["format"]["syntax"], "lark")

    def test_extracts_non_lite_top_level_tools(self) -> None:
        body = codex_body("hello", tools=[
            EXEC_TOOL,
            {"type": "function", "name": "wait", "parameters": {"type": "object"}},
        ])

        tools = codex_tool_bridge.response_client_tools(body)

        self.assertEqual([(tool["kind"], tool["name"]) for tool in tools], [("custom", "exec"), ("function", "wait")])

    def test_controller_prompt_only_examples_available_tools(self) -> None:
        body = codex_body({
            "type": "additional_tools",
            "role": "developer",
            "tools": [{
                "type": "function",
                "name": "shell_command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            }],
        })
        tools = codex_tool_bridge.response_client_tools(body)
        prompt = codex_tool_bridge.controller_prompt(tools)

        self.assertIn('"name":"shell_command"', prompt)
        self.assertNotIn("Custom exec tool:", prompt)

    def test_custom_exec_alias_is_coerced_to_shell_command(self) -> None:
        body = codex_body({
            "type": "additional_tools",
            "role": "developer",
            "tools": [{
                "type": "function",
                "name": "shell_command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            }],
        })
        tools = codex_tool_bridge.response_client_tools(body)
        action = codex_tool_bridge.parse_controller_action(
            '{"action":"tool","name":"exec","input":"const r = await tools.shell_command({command: \\"Get-Content README.md -TotalCount 20\\"}); text(r);"}',
            tools,
        )

        self.assertIsNotNone(action)
        self.assertEqual(action["kind"], "function")
        self.assertEqual(action["name"], "shell_command")
        self.assertEqual(json.loads(action["input"]), {"command": "Get-Content README.md -TotalCount 20"})

    def test_repeated_developer_and_environment_messages_are_preserved(self) -> None:
        environment = "<environment_context><cwd>C:\\project</cwd></environment_context>"
        body = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [EXEC_TOOL]},
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": environment}]},
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": environment}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "read the current project"}]},
        )

        _model, messages = openai_v1_response.text_response_parts(body)

        transcript = "\n".join(message["content"] for message in messages if message["role"] == "user")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(transcript.count("<environment_context>"), 2)
        self.assertEqual(sum(message["content"].endswith(environment) for message in messages), 2)
        self.assertIn(
            "MUST select a local inspection executor action",
            "\n".join(message["content"] for message in messages),
        )

        backend = OpenAIBackendAPI()
        try:
            payload = backend._conversation_payload(messages, "gpt-5.6-luna", "Asia/Shanghai")
        finally:
            backend.close()
        payload_text = "\n".join(item["content"]["parts"][0] for item in payload["messages"])
        self.assertEqual(payload_text.count("<environment_context>"), 2)
        self.assertEqual(sum(item["content"]["parts"][0].endswith(environment) for item in payload["messages"]), 2)

    def test_large_exec_catalog_is_bounded_without_compacting_instruction_records(self) -> None:
        repeated_instruction = "<environment_context><cwd>C:\\project</cwd></environment_context>\n" + ("rule\n" * 5000)
        large_exec = {
            **EXEC_TOOL,
            "description": "\n\n".join([
                "Run raw JavaScript through the V8 controller.",
                "### `shell_command`\nRuns PowerShell commands.\n" + ("shell details " * 900),
                "### `apply_patch`\nEdits files.\n" + ("patch details " * 900),
                "### `rare_remote_tool`\nRare remote operation.\n" + ("remote details " * 4000),
            ]),
        }
        body = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [large_exec]},
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": repeated_instruction}]},
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": repeated_instruction}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "read the current project"}]},
        )

        tools = codex_tool_bridge.response_client_tools(body)
        messages = codex_tool_bridge.controller_messages(body, tools, force_tool=True)
        all_text = "\n".join(message["content"] for message in messages)

        for item_index in (1, 2):
            reconstructed = "".join(
                message["content"].split("\n", 1)[1]
                for message in messages
                if message["content"].startswith(f"CODEX_INPUT_RECORD index={item_index} ")
            )
            self.assertEqual(reconstructed, repeated_instruction)
        self.assertIn("shell_command", all_text)
        self.assertIn("rare_remote_tool", all_text)
        self.assertIn("ALL_TOOLS", all_text)
        self.assertLessEqual(
            max(len(message["content"].encode("utf-8")) for message in messages),
            codex_tool_bridge.CONTROLLER_RECORD_MAX_BYTES + 256,
        )
        legacy_messages = [
            {"role": "system", "content": codex_tool_bridge.controller_prompt(tools, force_tool=True)},
            {
                "role": "user",
                "content": "CODEX_REQUEST_TRANSCRIPT_DATA\n" + json.dumps(
                    {"request_transcript": body},
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]
        backend = OpenAIBackendAPI()
        try:
            current_payload = backend._conversation_payload(messages, "gpt-5.6-luna", "Asia/Shanghai")
            legacy_payload = backend._conversation_payload(legacy_messages, "gpt-5.6-luna", "Asia/Shanghai")
        finally:
            backend.close()
        wire_size = lambda payload: len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        self.assertLess(wire_size(current_payload), wire_size(legacy_payload) * 0.65)

    def test_namespace_function_action_emits_codex_wire_item(self) -> None:
        body = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [
                {"type": "namespace", "name": "collaboration", "description": "Agents", "tools": [
                    {
                        "type": "function",
                        "name": "send_message",
                        "parameters": {
                            "type": "object",
                            "properties": {"target": {"type": "string"}, "message": {"type": "string"}},
                            "required": ["target", "message"],
                        },
                    },
                ]},
            ]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "send status"}]},
        )
        controller_output = json.dumps({
            "action": "tool",
            "namespace": "collaboration",
            "name": "send_message",
            "arguments": {"target": "agent-1", "message": "status"},
        })

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_response.stream_text_deltas", return_value=iter([controller_output])),
        ):
            events = list(openai_v1_response.handle(body))

        item = next(event["item"] for event in events if event["type"] == "response.output_item.done")
        self.assertEqual(item["type"], "function_call")
        self.assertEqual(item["namespace"], "collaboration")
        self.assertEqual(item["name"], "send_message")
        self.assertEqual(json.loads(item["arguments"]), {"target": "agent-1", "message": "status"})
        self.assertEqual(events[-1]["type"], "response.completed")

    def test_controller_prose_cannot_claim_a_function_ran(self) -> None:
        body = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [
                {"type": "namespace", "name": "collaboration", "description": "Agents", "tools": [
                    {
                        "type": "function",
                        "name": "send_message",
                        "parameters": {
                            "type": "object",
                            "properties": {"target": {"type": "string"}, "message": {"type": "string"}},
                            "required": ["target", "message"],
                        },
                    },
                ]},
            ]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "send status to agent-1"}]},
        )

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch(
                "services.protocol.openai_v1_response.stream_text_deltas",
                side_effect=lambda _backend, _request: iter(["Status sent."]),
            ) as stream,
        ):
            chunks = list(responses_sse_stream(openai_v1_response.handle(body)))

        decoded = [
            json.loads(chunk[5:].strip())
            for chunk in chunks
            if chunk.startswith("data:") and "[DONE]" not in chunk
        ]
        self.assertEqual(stream.call_count, 2)
        self.assertEqual(decoded[-1]["type"], "response.failed")
        self.assertFalse(any(item["type"] == "response.completed" for item in decoded))

    def test_function_argument_types_and_empty_final_are_rejected(self) -> None:
        body = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [
                {
                    "type": "function",
                    "name": "send_message",
                    "parameters": {
                        "type": "object",
                        "properties": {"target": {"type": "string"}, "message": {"type": "string"}},
                        "required": ["target", "message"],
                        "additionalProperties": False,
                    },
                },
            ]},
        )
        tools = codex_tool_bridge.response_client_tools(body)

        self.assertIsNone(codex_tool_bridge.parse_controller_action('{"action":"final","text":""}', tools))
        self.assertIsNone(codex_tool_bridge.parse_controller_action(
            '{"action":"tool","name":"send_message","arguments":{"target":123,"message":false}}',
            tools,
        ))

    def test_deferred_tool_cannot_be_called_before_tool_search(self) -> None:
        body = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [
                {
                    "type": "function",
                    "name": "deferred_lookup",
                    "defer_loading": True,
                    "parameters": {"type": "object"},
                },
                {"type": "tool_search", "execution": "client", "parameters": {"type": "object"}},
            ]},
        )
        tools = codex_tool_bridge.response_client_tools(body)

        self.assertTrue(next(tool for tool in tools if tool["name"] == "deferred_lookup")["defer_loading"])
        self.assertIsNone(codex_tool_bridge.parse_controller_action(
            '{"action":"tool","name":"deferred_lookup","arguments":{}}',
            tools,
        ))

    def test_tool_search_output_reveals_dynamic_namespace_tool(self) -> None:
        body = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [
                {
                    "type": "tool_search",
                    "execution": "client",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            ]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "create a calendar event"}]},
            {
                "type": "tool_search_call",
                "call_id": "search_1",
                "execution": "client",
                "arguments": {"query": "calendar create"},
            },
            {
                "type": "tool_search_output",
                "call_id": "search_1",
                "status": "completed",
                "execution": "client",
                "tools": [
                    {
                        "type": "namespace",
                        "name": "calendar",
                        "description": "Calendar tools",
                        "tools": [
                            {
                                "type": "function",
                                "name": "create",
                                "defer_loading": True,
                                "parameters": {
                                    "type": "object",
                                    "properties": {"title": {"type": "string"}},
                                    "required": ["title"],
                                },
                            },
                        ],
                    },
                ],
            },
        )
        tools = codex_tool_bridge.response_client_tools(body)
        revealed = next(tool for tool in tools if tool["namespace"] == "calendar" and tool["name"] == "create")
        self.assertFalse(revealed["defer_loading"])

        controller_output = '{"action":"tool","namespace":"calendar","name":"create","arguments":{"title":"Review"}}'
        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_response.stream_text_deltas", return_value=iter([controller_output])),
        ):
            events = list(openai_v1_response.handle(body))

        item = next(event["item"] for event in events if event["type"] == "response.output_item.done")
        self.assertEqual(item["type"], "function_call")
        self.assertEqual(item["namespace"], "calendar")
        self.assertEqual(item["name"], "create")
        self.assertEqual(json.loads(item["arguments"]), {"title": "Review"})

        continuation = {**body, "input": [
            *body["input"],
            item,
            {"type": "function_call_output", "call_id": item["call_id"], "output": "created"},
        ]}
        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch(
                "services.protocol.openai_v1_response.stream_text_deltas",
                return_value=iter(['{"action":"final","text":"Event created"}']),
            ),
        ):
            final_events = list(openai_v1_response.handle(continuation))

        final_output = final_events[-1]["response"]["output"][0]
        self.assertEqual(final_output["type"], "message")
        self.assertEqual(final_output["content"][0]["text"], "Event created")

    def test_local_refusal_repairs_then_bootstraps_exec(self) -> None:
        environment = "<environment_context><cwd>C:\\project</cwd></environment_context>"
        body = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [
                {"type": "namespace", "name": "functions", "description": "", "tools": [EXEC_TOOL]},
            ]},
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": environment}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "read the current project"}]},
        )
        outputs = iter([
            "I cannot access your local filesystem.",
            "I still cannot inspect the project.",
        ])

        def fake_stream(_backend, _request):
            yield next(outputs)

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_response.stream_text_deltas", side_effect=fake_stream) as stream,
        ):
            events = list(openai_v1_response.handle(body))

        self.assertEqual(stream.call_count, 2)
        item = next(event["item"] for event in events if event["type"] == "response.output_item.done")
        self.assertEqual(item["type"], "custom_tool_call")
        self.assertEqual(item["name"], "exec")
        self.assertIn("tools.shell_command", item["input"])
        self.assertIn("Get-ChildItem -Force", item["input"])

    def test_local_repair_plan_is_replaced_with_exec(self) -> None:
        """A prose plan cannot terminate a turn that requires local execution."""
        environment = "<environment_context><cwd>C:\\project</cwd></environment_context>"
        body = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [
                {"type": "namespace", "name": "functions", "description": "", "tools": [EXEC_TOOL]},
            ]},
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": environment}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "read the current project"}]},
        )
        outputs = iter([
            '{"action":"final","text":"I will analyze the project first."}',
            '{"action":"final","text":"I will inspect the files and continue."}',
        ])

        def fake_stream(_backend, _request):
            yield next(outputs)

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_response.stream_text_deltas", side_effect=fake_stream) as stream,
        ):
            events = list(openai_v1_response.handle(body))

        self.assertEqual(stream.call_count, 2)
        item = next(event["item"] for event in events if event["type"] == "response.output_item.done")
        self.assertEqual(item["type"], "custom_tool_call")
        self.assertEqual(item["name"], "exec")
        self.assertIn("Get-ChildItem -Force", item["input"])

    def test_plain_shell_text_is_rejected_as_exec_javascript(self) -> None:
        tools = codex_tool_bridge.response_client_tools(codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [EXEC_TOOL]},
        ))

        action = codex_tool_bridge.parse_controller_action(
            '{"action":"tool","name":"exec","input":"Get-ChildItem -Force"}',
            tools,
        )

        self.assertIsNone(action)

    def test_legacy_marker_cannot_call_a_deferred_function(self) -> None:
        tools = codex_tool_bridge.response_client_tools(codex_body(
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [
                    {
                        "type": "function",
                        "name": "deferred_lookup",
                        "defer_loading": True,
                        "parameters": {"type": "object"},
                    },
                ],
            },
        ))
        marker = (
            '<codex_tool_call name="deferred_lookup">'
            '<![CDATA[{"query":"calendar"}]]>'
            '</codex_tool_call>'
        )

        calls, visible = openai_v1_response.parse_client_tool_calls(marker, tools)

        self.assertEqual(calls, [])
        self.assertEqual(visible, marker)

    def test_legacy_tool_search_marker_requires_valid_json_arguments(self) -> None:
        tools = codex_tool_bridge.response_client_tools(codex_body(
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [
                    {
                        "type": "tool_search",
                        "execution": "client",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                ],
            },
        ))
        marker = (
            '<codex_tool_call name="tool_search">'
            '<![CDATA[calendar tools]]>'
            '</codex_tool_call>'
        )

        calls, visible = openai_v1_response.parse_client_tool_calls(marker, tools)

        self.assertEqual(calls, [])
        self.assertEqual(visible, marker)

    def test_legacy_exec_marker_still_accepts_raw_javascript(self) -> None:
        tools = codex_tool_bridge.response_client_tools(codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [EXEC_TOOL]},
        ))
        marker = (
            '<codex_tool_call name="exec"><![CDATA['
            'const result = await tools.shell_command({command: "Get-ChildItem"}); text(result);'
            ']]></codex_tool_call>'
        )

        calls, visible = openai_v1_response.parse_client_tool_calls(marker, tools)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["kind"], "custom")
        self.assertEqual(calls[0]["name"], "exec")
        self.assertIn("tools.shell_command", calls[0]["input"])
        self.assertEqual(visible, "")

    def test_local_project_request_rejects_unrelated_tool_action(self) -> None:
        environment = "<environment_context><cwd>C:\\project</cwd></environment_context>"
        body = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [
                EXEC_TOOL,
                {
                    "type": "function",
                    "name": "request_user_input",
                    "parameters": {
                        "type": "object",
                        "properties": {"question": {"type": "string"}},
                        "required": ["question"],
                    },
                },
            ]},
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": environment}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "read the current project"}]},
        )
        unrelated = '{"action":"tool","name":"request_user_input","arguments":{"question":"upload the project"}}'

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch(
                "services.protocol.openai_v1_response.stream_text_deltas",
                side_effect=lambda _backend, _request: iter([unrelated]),
            ) as stream,
        ):
            events = list(openai_v1_response.handle(body))

        self.assertEqual(stream.call_count, 2)
        item = next(event["item"] for event in events if event["type"] == "response.output_item.done")
        self.assertEqual(item["type"], "custom_tool_call")
        self.assertEqual(item["name"], "exec")

    def test_named_namespace_exec_is_not_treated_as_local_bootstrap(self) -> None:
        environment = "<environment_context><cwd>C:\\project</cwd></environment_context>"
        body = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [
                {
                    "type": "namespace",
                    "name": "remote",
                    "tools": [{"type": "custom", "name": "exec", "description": "Remote executor"}],
                },
            ]},
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": environment}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "read the current project"}]},
        )
        tools = codex_tool_bridge.response_client_tools(body)
        action = codex_tool_bridge.parse_controller_action(
            '{"action":"tool","namespace":"remote","name":"exec","input":"text(\'remote\')"}',
            tools,
        )

        self.assertFalse(codex_tool_bridge.requires_local_tool(body, tools))
        self.assertFalse(codex_tool_bridge.is_local_executor_action(action))
        self.assertIsNone(codex_tool_bridge.bootstrap_local_action(tools))

    def test_tool_call_and_structured_output_are_paired_by_call_id(self) -> None:
        body = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [EXEC_TOOL]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "inspect the project"}]},
            {"type": "custom_tool_call", "name": "exec", "call_id": "call_1", "input": "text('listed')"},
            {"type": "custom_tool_call_output", "call_id": "call_1", "output": [
                {"type": "input_text", "text": "README.md"},
                {"type": "input_image", "image_url": "data:image/png;base64,abc"},
            ]},
        )

        _model, messages = openai_v1_response.text_response_parts(body)

        transcript = "\n".join(message["content"] for message in messages)
        self.assertIn('"type": "custom_tool_call"', transcript)
        self.assertIn('"type": "custom_tool_call_output"', transcript)
        self.assertEqual(transcript.count('"call_id": "call_1"'), 2)
        self.assertIn('"name": "exec"', transcript)
        self.assertIn('"type": "input_image"', transcript)
        self.assertNotIn("MUST select one tool action", messages[0]["content"])

    def test_continuation_sends_full_tool_records_to_conversation_backend(self) -> None:
        body = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [EXEC_TOOL]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "inspect the project"}]},
            {"type": "custom_tool_call", "name": "exec", "call_id": "call_1", "input": "text('listed')"},
            {"type": "custom_tool_call_output", "call_id": "call_1", "output": "README.md"},
        )
        captured = []

        def fake_stream(_backend, request):
            captured.append(request)
            yield '{"action":"final","text":"README.md is present"}'

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_response.stream_text_deltas", side_effect=fake_stream),
        ):
            list(openai_v1_response.handle(body))

        self.assertEqual(len(captured), 1)
        request_text = "\n".join(str(message["content"]) for message in captured[0].messages)
        self.assertIn('"type": "custom_tool_call"', request_text)
        self.assertIn('"type": "custom_tool_call_output"', request_text)
        self.assertIn("call_1", request_text)
        self.assertIn("README.md", request_text)

    def test_codex_thread_continuation_sends_only_new_tool_result(self) -> None:
        environment = "<environment_context><cwd>C:\\project</cwd></environment_context>"
        first = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [EXEC_TOOL]},
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": environment}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "read the current project"}]},
        )
        first.update({
            "prompt_cache_key": "shared-session",
            "_request_identity_key_id": "admin",
            "client_metadata": {"session_id": "session-1", "thread_id": "thread-1", "turn_id": "turn-1"},
        })
        captured = []
        cursors_before_stream = []

        def fake_stream(_backend, request):
            captured.append(request)
            cursors_before_stream.append((request.conversation_id, request.parent_message_id, request.access_token))
            request.conversation_id = "conv-1"
            request.parent_message_id = f"node-{len(captured)}"
            request.access_token = "token-1"
            if len(captured) == 1:
                yield json.dumps({
                    "action": "tool",
                    "name": "exec",
                    "input": "const r = await tools.shell_command({command: 'Get-ChildItem'}); text(r);",
                })
            else:
                yield '{"action":"final","text":"README.md is present"}'

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_response.stream_text_deltas", side_effect=fake_stream),
            mock.patch(
                "services.protocol.codex_conversation_session.account_service.resolve_access_token",
                side_effect=lambda token: token,
            ),
            mock.patch(
                "services.protocol.codex_conversation_session.account_service.get_account",
                return_value={"status": "normal"},
            ),
        ):
            first_events = list(openai_v1_response.handle(first))
            call_item = next(
                event["item"]
                for event in first_events
                if event["type"] == "response.output_item.done" and event["item"]["type"] == "custom_tool_call"
            )
            second = {
                **first,
                "model": "gpt-5-6-luna",
                "input": [
                    *first["input"],
                    call_item,
                    {"type": "custom_tool_call_output", "call_id": call_item["call_id"], "output": "README.md"},
                ],
            }
            second["client_metadata"] = {**first["client_metadata"], "turn_id": "turn-1"}
            second_events = list(openai_v1_response.handle(second))

        self.assertEqual(len(captured), 2)
        continuation = captured[1]
        self.assertEqual(cursors_before_stream[1], ("conv-1", "node-1", "token-1"))
        continuation_text = "\n".join(message["content"] for message in continuation.messages)
        self.assertIn("custom_tool_call_output", continuation_text)
        self.assertIn("README.md", continuation_text)
        self.assertNotIn(environment, continuation_text)
        self.assertNotIn("read the current project", continuation_text)
        self.assertIn("force_local_tool=false", continuation_text)
        final_output = second_events[-1]["response"]["output"][0]
        self.assertEqual(final_output["content"][0]["text"], "README.md is present")

    def test_plain_text_after_tool_result_is_accepted_as_final(self) -> None:
        body = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [EXEC_TOOL]},
            {"type": "message", "role": "user", "content": "inspect the project"},
            {"type": "custom_tool_call", "name": "exec", "call_id": "call-1", "input": "text('ok')"},
            {"type": "custom_tool_call_output", "call_id": "call-1", "output": "README.md"},
        )

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch(
                "services.protocol.openai_v1_response.stream_text_deltas",
                return_value=iter(["README.md is present"]),
            ) as stream,
        ):
            events = list(openai_v1_response.handle(body))

        self.assertEqual(stream.call_count, 1)
        self.assertEqual(events[-1]["response"]["output"][0]["content"][0]["text"], "README.md is present")

    def test_conversation_cursor_is_captured_and_reused_in_payload(self) -> None:
        payloads = [
            json.dumps({
                "conversation_id": "conv-1",
                "message": {
                    "id": "assistant-node-1",
                    "author": {"role": "assistant"},
                    "channel": "final",
                    "recipient": "all",
                    "content": {"content_type": "text", "parts": ["done"]},
                },
            }),
            "[DONE]",
        ]

        events = list(iter_conversation_payloads(iter(payloads)))
        self.assertEqual(events[-1]["conversation_id"], "conv-1")
        self.assertEqual(events[-1]["parent_message_id"], "assistant-node-1")

        backend = OpenAIBackendAPI()
        try:
            payload = backend._conversation_payload(
                [{"role": "user", "content": "next"}],
                "gpt-5.6-luna",
                "Asia/Shanghai",
                conversation_id="conv-1",
                parent_message_id="assistant-node-1",
            )
        finally:
            backend.close()
        self.assertEqual(payload["conversation_id"], "conv-1")
        self.assertEqual(payload["parent_message_id"], "assistant-node-1")

    def test_tool_search_action_emits_client_search_item(self) -> None:
        body = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [
                {
                    "type": "tool_search",
                    "execution": "client",
                    "description": "Find deferred tools",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            ]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "find a calendar tool"}]},
        )
        controller_output = '{"action":"tool","name":"tool_search","arguments":{"query":"calendar"}}'

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_response.stream_text_deltas", return_value=iter([controller_output])),
        ):
            events = list(openai_v1_response.handle(body))

        item = next(event["item"] for event in events if event["type"] == "response.output_item.done")
        self.assertEqual(item["type"], "tool_search_call")
        self.assertEqual(item["execution"], "client")
        self.assertEqual(item["arguments"], {"query": "calendar"})

    def test_final_controller_json_is_unwrapped_for_user(self) -> None:
        body = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [EXEC_TOOL]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "what is 2+2?"}]},
        )

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch(
                "services.protocol.openai_v1_response.stream_text_deltas",
                return_value=iter(['{"action":"final","text":"4"}']),
            ),
        ):
            events = list(openai_v1_response.handle(body))

        completed = events[-1]["response"]
        self.assertEqual(completed["output"][0]["content"][0]["text"], "4")

    def test_tool_workflows_bypass_text_cache(self) -> None:
        body = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [EXEC_TOOL]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "what is 2+2?"}]},
        )
        calls = 0

        def fake_stream(_backend, _request):
            nonlocal calls
            calls += 1
            yield '{"action":"final","text":"4"}'

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_response.stream_text_deltas", side_effect=fake_stream),
        ):
            list(openai_v1_response.handle(body))
            list(openai_v1_response.handle(body))

        self.assertEqual(calls, 2)

    def test_responses_stream_converts_generator_failure_to_failed_terminal(self) -> None:
        def broken_stream():
            yield openai_v1_response.response_created("resp_test", "gpt-5.6-luna", 1)
            raise RuntimeError("upstream 422")

        payloads = [chunk for chunk in responses_sse_stream(broken_stream()) if chunk.startswith("data:")]
        decoded = [json.loads(chunk[5:].strip()) for chunk in payloads if "[DONE]" not in chunk]

        self.assertEqual(decoded[0]["type"], "response.created")
        self.assertEqual(decoded[-1]["type"], "response.failed")
        self.assertEqual(decoded[-1]["response"]["id"], "resp_test")
        self.assertIn("upstream 422", decoded[-1]["response"]["error"]["message"])
        self.assertFalse(any(item["type"] == "response.completed" for item in decoded))

    def test_responses_stream_marks_upstream_413_as_non_retryable_context_error(self) -> None:
        def oversized_stream():
            yield openai_v1_response.response_created("resp_large", "gpt-5.6-luna", 1)
            raise UpstreamHTTPError("/backend-api/conversation", 413, "")

        payloads = [chunk for chunk in responses_sse_stream(oversized_stream()) if chunk.startswith("data:")]
        decoded = [json.loads(chunk[5:].strip()) for chunk in payloads if "[DONE]" not in chunk]
        error = decoded[-1]["response"]["error"]

        self.assertEqual(decoded[-1]["type"], "response.failed")
        self.assertEqual(error["type"], "invalid_request_error")
        self.assertEqual(error["code"], "context_length_exceeded")


if __name__ == "__main__":
    unittest.main()
