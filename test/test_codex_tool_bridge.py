from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from api.ai import create_router, merge_responses_request_identity
from services.config import config
from services.openai_backend_api import OpenAIBackendAPI
from services.log_service import LoggedCall
from services.protocol import (
    codex_conversation_session,
    codex_tool_bridge,
    codex_tool_grammar,
    openai_v1_response,
    web_search_tool,
)
from services.protocol.conversation import ConversationRequest, ImageOutput, iter_conversation_payloads
from utils.helper import UpstreamHTTPError, current_stream_cancellation, responses_sse_stream


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

PNG_1X1_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP8z8BQDwAFgwJ/luzl4wAAAABJRU5ErkJggg=="


def codex_body(*items, tools=None):
    return {
        "model": "gpt-5.6-luna",
        "stream": True,
        "input": list(items),
        **({"tools": tools} if tools is not None else {}),
    }


V1_AGENT_ID = "00000000-0000-4000-8000-000000000001"


def multi_agent_v1_body(*names: str):
    item_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "text": {"type": "string"},
                "image_url": {"type": "string"},
                "audio_url": {"type": "string"},
                "path": {"type": "string"},
                "name": {"type": "string"},
            },
            "additionalProperties": False,
        },
    }
    schemas = {
        "spawn_agent": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "items": item_schema,
                "agent_type": {"type": "string"},
                "fork_context": {"type": "boolean"},
                "model": {"type": "string"},
                "reasoning_effort": {"type": "string"},
                "service_tier": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "send_input": {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "message": {"type": "string"},
                "items": item_schema,
                "interrupt": {"type": "boolean"},
            },
            "required": ["target"],
            "additionalProperties": False,
        },
        "resume_agent": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
            "additionalProperties": False,
        },
        "wait_agent": {
            "type": "object",
            "properties": {
                "targets": {"type": "array", "items": {"type": "string"}},
                "timeout_ms": {"type": "number"},
            },
            "required": ["targets"],
            "additionalProperties": False,
        },
        "close_agent": {
            "type": "object",
            "properties": {"target": {"type": "string"}},
            "required": ["target"],
            "additionalProperties": False,
        },
    }
    return codex_body({
        "type": "additional_tools",
        "role": "developer",
        "tools": [{
            "type": "namespace",
            "name": "multi_agent_v1",
            "description": "Tools for spawning and managing sub-agents.",
            "tools": [
                {"type": "function", "name": name, "parameters": schemas[name]}
                for name in names
            ],
        }],
    })


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

    def test_codex_identity_headers_are_folded_into_response_payload(self) -> None:
        request = Request({
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [
                (b"session-id", b"session-header"),
                (b"thread-id", b"thread-header"),
                (b"x-codex-turn-metadata", b'{"turn_id":"turn-header","window_id":"window-header"}'),
            ],
        })
        payload = {"client_metadata": {"thread_id": "thread-body"}}

        merge_responses_request_identity(payload, request)

        self.assertEqual(payload["prompt_cache_key"], "session-header")
        self.assertEqual(payload["client_metadata"]["session_id"], "session-header")
        self.assertEqual(payload["client_metadata"]["thread_id"], "thread-body")
        self.assertEqual(payload["client_metadata"]["turn_id"], "turn-header")
        self.assertEqual(payload["client_metadata"]["window_id"], "window-header")

    def test_codex_parent_and_subagent_headers_are_preserved(self) -> None:
        request = Request({
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [
                (b"x-codex-parent-thread-id", b"parent-thread"),
                (b"x-openai-subagent", b"subagent-v2"),
            ],
        })
        payload: dict[str, object] = {}

        merge_responses_request_identity(payload, request)

        self.assertEqual(payload["client_metadata"]["parent_thread_id"], "parent-thread")
        self.assertEqual(payload["client_metadata"]["subagent_header"], "subagent-v2")

    def test_responses_websocket_returns_upgrade_required_for_http_fallback(self) -> None:
        app = FastAPI()
        app.include_router(create_router())

        with TestClient(app) as client:
            with self.assertRaises(Exception) as context:
                with client.websocket_connect("/v1/responses"):
                    pass

        self.assertEqual(getattr(context.exception, "status_code", None), 426)

    def test_response_created_precedes_text_backend_initialization(self) -> None:
        body = {"model": "gpt-5.6-luna", "stream": True, "input": "hello"}

        with mock.patch(
            "services.protocol.openai_v1_response.text_backend",
            side_effect=RuntimeError("slow backend setup"),
        ) as backend:
            events = openai_v1_response.handle(body)
            first = next(events)
            self.assertEqual(first["type"], "response.created")
            backend.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "slow backend setup"):
                next(events)

    def test_logged_responses_stream_is_not_prefetched_before_http_return(self) -> None:
        advanced = False

        def source():
            nonlocal advanced
            advanced = True
            yield openai_v1_response.response_created("resp_lazy", "gpt-5.6-luna", 1)

        call = LoggedCall(
            {"id": "test", "name": "test", "role": "admin"},
            "/v1/responses",
            "gpt-5.6-luna",
            "Responses",
        )
        response = asyncio.run(call.run(lambda _payload: source(), {"_response_id": "resp_lazy"}, sse="responses"))

        self.assertFalse(advanced)
        self.assertEqual(response.headers.get("x-request-id"), "resp_lazy")

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

    def test_code_mode_wait_example_is_not_an_agent_wait(self) -> None:
        body = codex_body("hello", tools=[
            EXEC_TOOL,
            {
                "type": "function",
                "name": "wait",
                "parameters": {
                    "type": "object",
                    "properties": {"cell_id": {"type": "string"}},
                    "required": ["cell_id"],
                    "additionalProperties": False,
                },
            },
        ])

        prompt = codex_tool_bridge.controller_prompt(
            codex_tool_bridge.response_client_tools(body)
        )

        self.assertIn('"name":"wait","namespace":null,"arguments":{"cell_id":', prompt)
        self.assertIn("top-level `wait` function is only for a yielded exec cell", prompt)
        self.assertIn("multi_agent_v1__wait_agent({targets:[child_id]})", prompt)

    def test_controller_messages_include_actual_codex_tool_schemas(self) -> None:
        body = codex_body(
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [
                    {
                        "type": "namespace",
                        "name": "functions",
                        "description": "Built-in tools",
                        "tools": [
                            {
                                "type": "function",
                                "name": "shell_command",
                                "description": "Run a local command",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"command": {"type": "string"}},
                                    "required": ["command"],
                                },
                            },
                            {
                                "type": "function",
                                "name": "apply_patch",
                                "description": "Edit a file",
                                "parameters": {"type": "object"},
                            },
                        ],
                    },
                ],
            },
            {"type": "message", "role": "user", "content": "inspect the project"},
        )
        tools = codex_tool_bridge.response_client_tools(body)
        messages = codex_tool_bridge.controller_messages(body, tools)
        transcript = "\n".join(str(message["content"]) for message in messages)

        self.assertIn("TOOL_DEFINITION_RECORD", transcript)
        self.assertIn('name=shell_command', transcript)
        self.assertIn('name=apply_patch', transcript)
        self.assertIn('"required":["command"]', transcript)
        self.assertIn('"name":"shell_command"', transcript)
        self.assertIn('"complete":true', transcript)
        self.assertIn("Tools have priority over prose", transcript)
        self.assertIn("Select the next tool before writing prose", transcript)

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

    def test_parallel_controller_action_requires_opt_in_and_distinct_calls(self) -> None:
        body = codex_body({
            "type": "additional_tools",
            "role": "developer",
            "tools": [
                {
                    "type": "function",
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
                {
                    "type": "function",
                    "name": "list_dir",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            ],
        })
        tools = codex_tool_bridge.response_client_tools(body)
        payload = json.dumps({
            "action": "tools",
            "calls": [
                {"name": "read_file", "arguments": {"path": "README.md"}},
                {"name": "list_dir", "arguments": {"path": "services"}},
            ],
        })

        self.assertIsNone(codex_tool_bridge.parse_controller_action(payload, tools))
        action = codex_tool_bridge.parse_controller_action(payload, tools, allow_parallel=True)

        self.assertEqual(action["action"], "tools")
        self.assertEqual([call["name"] for call in action["calls"]], ["read_file", "list_dir"])
        duplicate_payload = json.dumps({
            "action": "tools",
            "calls": [
                {"name": "read_file", "arguments": {"path": "README.md"}},
                {"name": "read_file", "arguments": {"path": "README.md"}},
            ],
        })
        self.assertIsNone(codex_tool_bridge.parse_controller_action(
            duplicate_payload,
            tools,
            allow_parallel=True,
        ))

    def test_parallel_tool_response_emits_two_paired_calls(self) -> None:
        body = codex_body(
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [
                    {
                        "type": "function",
                        "name": "read_file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                    {
                        "type": "function",
                        "name": "list_dir",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                ],
            },
            {"type": "message", "role": "user", "content": "inspect both independently"},
        )
        body["parallel_tool_calls"] = True
        controller_output = json.dumps({
            "action": "tools",
            "calls": [
                {"name": "read_file", "arguments": {"path": "README.md"}},
                {"name": "list_dir", "arguments": {"path": "services"}},
            ],
        })

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch(
                "services.protocol.openai_v1_response.stream_text_deltas",
                side_effect=lambda _backend, _request: iter([controller_output]),
            ) as stream,
        ):
            events = list(openai_v1_response.handle(body))

        self.assertEqual(stream.call_count, 1)
        added = [
            event for event in events
            if event.get("type") == "response.output_item.added"
        ]
        done = [
            event for event in events
            if event.get("type") == "response.output_item.done"
        ]
        self.assertEqual([event["output_index"] for event in added], [0, 1])
        self.assertEqual([event["output_index"] for event in done], [0, 1])
        self.assertEqual([event["item"]["name"] for event in done], ["read_file", "list_dir"])
        call_ids = [event["item"]["call_id"] for event in done]
        self.assertEqual(len(set(call_ids)), 2)
        created = next(event["response"] for event in events if event["type"] == "response.created")
        completed = next(event["response"] for event in events if event["type"] == "response.completed")
        self.assertTrue(created["parallel_tool_calls"])
        self.assertTrue(completed["parallel_tool_calls"])
        self.assertFalse(completed["end_turn"])

    def test_custom_exec_rejects_unescaped_windows_path(self) -> None:
        body = codex_body({"type": "additional_tools", "role": "developer", "tools": [EXEC_TOOL]})
        tools = codex_tool_bridge.response_client_tools(body)
        malformed_input = 'const r = await tools.shell_command({command: "Get-Content api\\app.py"}); text(r);'
        valid_input = 'const r = await tools.shell_command({command: "Get-Content api\\\\app.py"}); text(r);'

        self.assertIsNone(codex_tool_bridge.parse_controller_action(
            json.dumps({"action": "tool", "name": "exec", "input": malformed_input}),
            tools,
        ))
        self.assertIsNotNone(codex_tool_bridge.parse_controller_action(
            json.dumps({"action": "tool", "name": "exec", "input": valid_input}),
            tools,
        ))

    def test_codex_apply_patch_lark_grammar_is_enforced(self) -> None:
        grammar = r'''
start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" LF
end_patch: "*** End Patch" LF?
hunk: add_hunk
add_hunk: "*** Add File: " filename LF add_line+
filename: /(.+)/
add_line: "+" /(.*)/ LF
%import common.LF
'''
        tool = {
            "kind": "custom",
            "name": "apply_patch",
            "format": {"type": "grammar", "syntax": "lark", "definition": grammar},
        }
        valid = "*** Begin Patch\n*** Add File: x.txt\n+\n+ok\n*** End Patch\n"

        self.assertEqual(codex_tool_grammar.validate_custom_tool_input(tool, valid), (True, ""))
        valid_action = codex_tool_bridge.parse_controller_action(
            json.dumps({"action": "tool", "name": "apply_patch", "input": valid}),
            [tool],
        )
        invalid_action = codex_tool_bridge.parse_controller_action(
            json.dumps({"action": "tool", "name": "apply_patch", "input": "not a patch"}),
            [tool],
        )
        self.assertIsNotNone(valid_action)
        self.assertIsNone(invalid_action)

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
            "MUST select a tool action that advances the task",
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
                "### `rare_remote_tool`\nRare remote operation.\n"
                + ("remote details " * 4000)
                + "LATE_SCHEMA_MARKER required_marker",
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
        self.assertIn("LATE_SCHEMA_MARKER", all_text)
        self.assertIn("required_marker", all_text)
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
        self.assertLess(wire_size(current_payload), wire_size(legacy_payload))

    def test_encrypted_agent_message_is_retained_as_opaque_context(self) -> None:
        body = codex_body({
            "type": "agent_message",
            "author": "/root",
            "recipient": "/root/worker",
            "content": [
                {"type": "input_text", "text": "Message Type: NEW_TASK\nPayload:\n"},
                {"type": "encrypted_content", "encrypted_content": "opaque"},
            ],
        })

        history = openai_v1_response.messages_from_input(body["input"])
        self.assertIn("OPAQUE_ENCRYPTED_CONTENT_BEGIN", history[0]["content"])
        self.assertIn("opaque", history[0]["content"])

    def test_exec_pragma_matches_codex_parser_contract(self) -> None:
        valid = "// @exec: {\"yield_time_ms\": 10000, \"max_output_tokens\": 1000}\ntext(\"ok\");"
        self.assertEqual(codex_tool_grammar.validate_exec_source(valid), (True, ""))
        invalid = [
            "// @exec: {\"yield_time_ms\": 10000} trailing\ntext(\"ok\");",
            "// @exec: {\"unknown\": 1}\ntext(\"ok\");",
            "// @exec: {\"yield_time_ms\": -1}\ntext(\"ok\");",
            "// @exec: {\"yield_time_ms\": 10000}",
        ]
        for source in invalid:
            with self.subTest(source=source):
                self.assertFalse(codex_tool_grammar.validate_exec_source(source)[0])

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
        self.assertEqual(item["encrypted_function_args"], [])
        self.assertEqual(events[-1]["type"], "response.completed")

    def test_workspace_turn_can_delegate_before_running_a_local_executor(self) -> None:
        spawn_tool = {
            "type": "namespace",
            "name": "collaboration",
            "tools": [{
                "type": "function",
                "name": "spawn_agent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_name": {"type": "string"},
                        "message": {"type": "string", "encrypted": True},
                    },
                    "required": ["task_name", "message"],
                },
            }],
        }
        body = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [EXEC_TOOL, spawn_tool]},
            {"type": "message", "role": "developer", "content": "<environment_context><cwd>C:\\repo</cwd></environment_context>"},
            {"type": "message", "role": "user", "content": "inspect the current project"},
        )
        controller_output = json.dumps({
            "action": "tool",
            "namespace": "collaboration",
            "name": "spawn_agent",
            "arguments": {"task_name": "audit", "message": "inspect the API"},
        })

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_response.stream_text_deltas", return_value=iter([controller_output])),
        ):
            events = list(openai_v1_response.handle(body))

        item = next(event["item"] for event in events if event["type"] == "response.output_item.done")
        self.assertEqual(item["type"], "function_call")
        self.assertEqual(item["name"], "spawn_agent")
        self.assertEqual(item["encrypted_function_args"], [])

    def test_multi_agent_v2_runtime_argument_contracts_are_enforced(self) -> None:
        tools = codex_tool_bridge.response_client_tools(codex_body({
            "type": "additional_tools",
            "role": "developer",
            "tools": [{
                "type": "namespace",
                "name": "collaboration",
                "tools": [
                    {
                        "type": "function",
                        "name": "spawn_agent",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task_name": {"type": "string"},
                                "message": {"type": "string", "encrypted": True},
                                "fork_turns": {"type": "string"},
                                "fork_context": {"type": "boolean"},
                            },
                            "required": ["task_name", "message"],
                            "additionalProperties": False,
                        },
                    },
                    {
                        "type": "function",
                        "name": "send_message",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "target": {"type": "string"},
                                "message": {"type": "string", "encrypted": True},
                            },
                            "required": ["target", "message"],
                            "additionalProperties": False,
                        },
                    },
                    {
                        "type": "function",
                        "name": "wait_agent",
                        "parameters": {
                            "type": "object",
                            "properties": {"timeout_ms": {"type": "integer"}},
                            "additionalProperties": False,
                        },
                    },
                ],
            }],
        }))

        def parse(name: str, arguments: dict[str, object]):
            return codex_tool_bridge.parse_controller_action(json.dumps({
                "action": "tool",
                "namespace": "collaboration",
                "name": name,
                "arguments": arguments,
            }), tools)

        self.assertIsNotNone(parse("spawn_agent", {
            "task_name": "api_audit",
            "message": "inspect the API",
            "fork_turns": "3",
        }))
        self.assertIsNotNone(parse("wait_agent", {"timeout_ms": 1}))
        invalid = [
            ("spawn_agent", {"task_name": "BadName", "message": "inspect"}),
            ("spawn_agent", {"task_name": "root", "message": "inspect"}),
            ("spawn_agent", {"task_name": "audit", "message": ""}),
            ("spawn_agent", {"task_name": "audit", "message": "inspect", "fork_turns": "0"}),
            ("spawn_agent", {"task_name": "audit", "message": "inspect", "fork_context": False}),
            ("send_message", {"target": "", "message": "status"}),
            ("send_message", {"target": "/root/audit", "message": "   "}),
            ("wait_agent", {"timeout_ms": 3_600_001}),
        ]
        for name, arguments in invalid:
            with self.subTest(name=name, arguments=arguments):
                self.assertIsNone(parse(name, arguments))

    def test_v2_agent_new_task_overrides_forked_parent_user_task(self) -> None:
        parent = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "parent task"}],
        }
        child = {
            "type": "agent_message",
            "author": "/root",
            "recipient": "/root/worker",
            "content": [{
                "type": "input_text",
                "text": (
                    "Message Type: NEW_TASK\nTask name: /root/worker\n"
                    "Sender: /root\nPayload:\ninspect services/config.py"
                ),
            }],
        }

        self.assertEqual(codex_tool_bridge.latest_user_text([parent, child]), "inspect services/config.py")
        anchor = "\n".join(
            message["content"]
            for message in codex_tool_bridge.controller_task_anchor_messages([parent, child])
        )
        self.assertIn("inspect services/config.py", anchor)
        self.assertNotIn("parent task", anchor)

    def test_v2_agent_final_answer_is_external_user_context(self) -> None:
        item = {
            "type": "agent_message",
            "author": "/root/worker",
            "recipient": "/root",
            "content": [{
                "type": "input_text",
                "text": (
                    "Message Type: FINAL_ANSWER\nTask name: /root\n"
                    "Sender: /root/worker\nPayload:\nchild verified the parser"
                ),
            }],
        }

        history = codex_tool_bridge.response_item_history_message(item)

        self.assertEqual(history["role"], "user")
        self.assertIn("message_type=FINAL_ANSWER", history["content"])
        self.assertIn("child verified the parser", history["content"])

    def test_encrypted_agent_message_does_not_override_visible_task(self) -> None:
        items = [
            {"type": "message", "role": "user", "content": "visible task"},
            {
                "type": "agent_message",
                "content": [
                    {"type": "input_text", "text": "Message Type: NEW_TASK\nTask name: /root/w\nSender: /root\nPayload:\n"},
                    {"type": "encrypted_content", "encrypted_content": "opaque"},
                ],
            },
        ]

        self.assertEqual(codex_tool_bridge.latest_user_text(items), "visible task")
        history = codex_tool_bridge.response_item_history_message(items[-1])
        self.assertEqual(history["role"], "user")
        self.assertIn("OPAQUE_ENCRYPTED_CONTENT_BEGIN", history["content"])
        self.assertIn("opaque", history["content"])

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

    def test_multi_agent_v1_prompt_uses_runtime_valid_examples(self) -> None:
        tools = codex_tool_bridge.response_client_tools(multi_agent_v1_body(
            "spawn_agent",
            "send_input",
            "resume_agent",
            "wait_agent",
            "close_agent",
        ))

        prompt = codex_tool_bridge.controller_prompt(tools)

        self.assertIn(
            '"namespace":"multi_agent_v1","arguments":{"message":"Delegate one focused task."}',
            prompt,
        )
        self.assertIn(
            f'"namespace":"multi_agent_v1","arguments":{{"target":"{V1_AGENT_ID}","message":"Report your current status."}}',
            prompt,
        )
        self.assertIn("provide exactly one non-empty `message`", prompt)
        self.assertNotIn('"namespace":"multi_agent_v1","arguments":{}', prompt)

    def test_multi_agent_v1_spawn_requires_exactly_one_nonempty_input(self) -> None:
        tools = codex_tool_bridge.response_client_tools(multi_agent_v1_body("spawn_agent"))

        def parse(arguments):
            return codex_tool_bridge.parse_controller_action(json.dumps({
                "action": "tool",
                "namespace": "multi_agent_v1",
                "name": "spawn_agent",
                "arguments": arguments,
            }), tools)

        self.assertIsNotNone(parse({"message": "inspect the API"}))
        self.assertIsNotNone(parse({
            "items": [{"type": "mention", "name": "docs", "path": "plugin://docs@personal"}],
        }))
        invalid = [
            {},
            {"message": "   "},
            {"items": []},
            {"message": "inspect", "items": [{"type": "text", "text": "inspect"}]},
            {"items": [{"type": "unknown", "text": "inspect"}]},
            {"items": [{"type": "mention", "name": "docs"}]},
            {"message": "inspect", "fork_context": True, "agent_type": "explorer"},
            {"message": "inspect", "reasoning_effort": ""},
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                self.assertIsNone(parse(arguments))

    def test_multi_agent_v1_send_input_requires_uuid_and_one_input(self) -> None:
        tools = codex_tool_bridge.response_client_tools(multi_agent_v1_body("send_input"))

        def parse(arguments):
            return codex_tool_bridge.parse_controller_action(json.dumps({
                "action": "tool",
                "namespace": "multi_agent_v1",
                "name": "send_input",
                "arguments": arguments,
            }), tools)

        self.assertIsNotNone(parse({"target": V1_AGENT_ID, "message": "continue"}))
        self.assertIsNotNone(parse({
            "target": V1_AGENT_ID,
            "items": [{"type": "text", "text": "continue"}],
            "interrupt": True,
        }))
        invalid = [
            {"target": "agent-1", "message": "continue"},
            {"target": V1_AGENT_ID},
            {"target": V1_AGENT_ID, "message": ""},
            {"target": V1_AGENT_ID, "items": []},
            {
                "target": V1_AGENT_ID,
                "message": "continue",
                "items": [{"type": "text", "text": "continue"}],
            },
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                self.assertIsNone(parse(arguments))

    def test_multi_agent_v1_agent_id_and_wait_contracts(self) -> None:
        cases = [
            ("resume_agent", {"id": V1_AGENT_ID}, {"id": "not-a-uuid"}),
            ("close_agent", {"target": V1_AGENT_ID}, {"target": "/root/worker"}),
            (
                "wait_agent",
                {"targets": [V1_AGENT_ID], "timeout_ms": 30000},
                {"targets": []},
            ),
            (
                "wait_agent",
                {"targets": [V1_AGENT_ID]},
                {"targets": ["not-a-uuid"]},
            ),
            (
                "wait_agent",
                {"targets": [V1_AGENT_ID]},
                {"targets": [V1_AGENT_ID], "timeout_ms": 0},
            ),
            (
                "wait_agent",
                {"targets": [V1_AGENT_ID]},
                {"targets": [V1_AGENT_ID], "timeout_ms": 1.5},
            ),
        ]
        for name, valid, invalid in cases:
            with self.subTest(name=name, invalid=invalid):
                tools = codex_tool_bridge.response_client_tools(multi_agent_v1_body(name))

                def parse(arguments):
                    return codex_tool_bridge.parse_controller_action(json.dumps({
                        "action": "tool",
                        "namespace": "multi_agent_v1",
                        "name": name,
                        "arguments": arguments,
                    }), tools)

                self.assertIsNotNone(parse(valid))
                self.assertIsNone(parse(invalid))

        # V1 clamps values above its one-hour wait limit in the Rust handler;
        # the proxy must preserve that request instead of rejecting it.
        tools = codex_tool_bridge.response_client_tools(multi_agent_v1_body("wait_agent"))
        self.assertIsNotNone(codex_tool_bridge.parse_controller_action(json.dumps({
            "action": "tool",
            "namespace": "multi_agent_v1",
            "name": "wait_agent",
            "arguments": {"targets": [V1_AGENT_ID], "timeout_ms": 3_600_001},
        }), tools))

    def test_code_mode_v1_nested_agent_calls_require_runtime_valid_arguments(self) -> None:
        tool = {
            **EXEC_TOOL,
            "description": (
                "Run JavaScript.\n\n"
                "### `multi_agent_v1__spawn_agent`\n"
                "exec tool declaration for V1 agents."
            ),
        }
        tools = codex_tool_bridge.response_client_tools(codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [tool]},
        ))

        def parse(source: str):
            return codex_tool_bridge.parse_controller_action(json.dumps({
                "action": "tool",
                "name": "exec",
                "input": source,
            }), tools)

        valid = [
            'const child = await tools.multi_agent_v1__spawn_agent({message: "inspect"}); text(child);',
            (
                'const child = await tools.multi_agent_v1__spawn_agent({message: "inspect"}); '
                'text(child); const result = await tools.multi_agent_v1__wait_agent('
                '{targets: [child.agent_id], timeout_ms: 30000}); '
                'text(result);'
            ),
            (
                'const target = "00000000-0000-4000-8000-000000000001"; '
                'const result = await tools.multi_agent_v1__send_input({target, message: "continue"}); '
                'text(result);'
            ),
            (
                'const result = await tools.multi_agent_v1__spawn_agent({items: '
                '[{type: "text", text: "inspect"}]}); text(result);'
            ),
        ]
        for source in valid:
            with self.subTest(valid=source):
                self.assertIsNotNone(parse(source))

        invalid = [
            'const r = await tools.multi_agent_v1__spawn_agent(); text(r);',
            'const r = await tools.multi_agent_v1__spawn_agent({}); text(r);',
            'const r = await tools.multi_agent_v1__spawn_agent({message: ""}); text(r);',
            (
                'const r = await tools.multi_agent_v1__spawn_agent({message: "inspect", '
                'items: [{type: "text", text: "inspect"}]}); text(r);'
            ),
            'const r = await tools.multi_agent_v1__wait_agent(); text(r);',
            'const r = await tools.multi_agent_v1__wait_agent({targets: []}); text(r);',
            (
                'const r = await tools.multi_agent_v1__wait_agent({targets: '
                '["not-a-uuid"]}); text(r);'
            ),
            (
                'const child = await tools.multi_agent_v1__spawn_agent({message: "inspect"}); '
                'const r = await tools.multi_agent_v1__wait_agent({targets: [child.id]}); text(r);'
            ),
            (
                'const r = await tools.multi_agent_v1__wait_agent('
                '{targets: [child.agent_id]}); text(r);'
            ),
            (
                'const target = "not-a-uuid"; const r = await '
                'tools.multi_agent_v1__send_input({target, message: "continue"}); text(r);'
            ),
            'const r = await tools.multi_agent_v1__resume_agent({id: ""}); text(r);',
            'const r = await tools.multi_agent_v1__close_agent({target: "bad"}); text(r);',
        ]
        for source in invalid:
            with self.subTest(invalid=source):
                self.assertIsNone(parse(source))

    def test_code_mode_v1_nested_validation_does_not_capture_v2_direct_tools(self) -> None:
        tools = codex_tool_bridge.response_client_tools(codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [EXEC_TOOL]},
        ))
        action = codex_tool_bridge.parse_controller_action(json.dumps({
            "action": "tool",
            "name": "exec",
            "input": "const value = await tools.collaboration__spawn_agent({}); text(value);",
        }), tools)
        self.assertIsNotNone(action)

    def test_nested_agent_spawn_is_fingerprinted_and_forms_mutation_barrier(self) -> None:
        source = (
            'const child = await tools.multi_agent_v1__spawn_agent({message: "inspect"}); '
            'text(child);'
        )
        call = {
            "type": "custom_tool_call",
            "name": "exec",
            "call_id": "spawn-call",
            "input": source,
        }
        output = {
            "type": "custom_tool_call_output",
            "call_id": "spawn-call",
            "output": "child",
        }
        action = {"action": "tool", "kind": "custom", "name": "exec", "input": source}
        self.assertTrue(codex_tool_bridge.action_repeats_completed_tool(
            action,
            [call, output],
        ))
        wait_source = (
            'await tools.multi_agent_v1__wait_agent({targets: '
            '["00000000-0000-4000-8000-000000000001"]});'
        )
        wait_call = {**call, "call_id": "wait-call", "input": wait_source}
        wait_output = {**output, "call_id": "wait-call"}
        self.assertFalse(codex_tool_bridge.action_repeats_completed_tool(
            {"action": "tool", "kind": "custom", "name": "exec", "input": wait_source},
            [wait_call, wait_output],
        ))

    def test_mixed_code_mode_agent_and_shell_calls_share_one_fingerprint(self) -> None:
        source = (
            'const child = await tools.multi_agent_v1__spawn_agent({message: "inspect"}); '
            'const listing = await tools.shell_command({command: "Get-ChildItem -Force"}); '
            'text({child, listing});'
        )
        call = {
            "type": "custom_tool_call",
            "name": "exec",
            "call_id": "mixed-call",
            "input": source,
        }
        output = {
            "type": "custom_tool_call_output",
            "call_id": "mixed-call",
            "output": "done",
        }
        action = {"action": "tool", "kind": "custom", "name": "exec", "input": source}
        self.assertTrue(codex_tool_bridge.action_repeats_completed_tool(action, [call, output]))
        changed = source.replace("Get-ChildItem -Force", "Get-Location")
        self.assertFalse(codex_tool_bridge.action_repeats_completed_tool(
            {**action, "input": changed}, [call, output]
        ))

    def test_exec_prompt_explains_v1_code_mode_and_v2_direct_agent_contracts(self) -> None:
        tools = codex_tool_bridge.response_client_tools(codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [EXEC_TOOL]},
        ))
        prompt = codex_tool_bridge.controller_prompt(tools)
        self.assertIn("tools.multi_agent_v1__spawn_agent", prompt)
        self.assertIn("returns `{agent_id, nickname}`", prompt)
        self.assertIn("targets: [spawned.agent_id]", prompt)
        self.assertIn("do not persist across separate exec calls", prompt)
        self.assertIn("literal agent_id UUID", prompt)
        self.assertIn("Never call a V1 agent function with no argument", prompt)
        self.assertIn("V2 `collaboration` tools are direct tools", prompt)

    def test_invalid_v1_agent_id_reference_is_repairable(self) -> None:
        bad = json.dumps({
            "action": "tool",
            "name": "exec",
            "input": (
                'const child = await tools.multi_agent_v1__spawn_agent({message: "inspect"}); '
                'const result = await tools.multi_agent_v1__wait_agent({targets: [child.id]}); '
                'text(result);'
            ),
        })
        good = json.dumps({
            "action": "tool",
            "name": "exec",
            "input": (
                'const child = await tools.multi_agent_v1__spawn_agent({message: "inspect"}); '
                'text(child); const result = await tools.multi_agent_v1__wait_agent('
                '{targets: [child.agent_id]}); text(result);'
            ),
        })
        self.assertTrue(codex_tool_bridge.has_invalid_v1_agent_id_reference(bad))
        self.assertFalse(codex_tool_bridge.has_invalid_v1_agent_id_reference(good))

    def test_agent_targets_sent_to_code_mode_wait_are_bridged_to_v1_exec(self) -> None:
        tools = codex_tool_bridge.response_client_tools(codex_body(
            "hello",
            tools=[
                EXEC_TOOL,
                {
                    "type": "function",
                    "name": "wait",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cell_id": {"type": "string"},
                            "yield_time_ms": {"type": "number"},
                        },
                        "required": ["cell_id"],
                        "additionalProperties": False,
                    },
                },
            ],
        ))

        action = codex_tool_bridge.parse_controller_action(json.dumps({
            "action": "tool",
            "name": "wait",
            "arguments": {"targets": [V1_AGENT_ID], "timeout_ms": 30000},
        }), tools)

        self.assertIsNotNone(action)
        self.assertEqual(action["name"], "exec")
        self.assertIn("tools.multi_agent_v1__wait_agent", action["input"])
        self.assertIn(f'"targets":["{V1_AGENT_ID}"]', action["input"])
        self.assertNotIn("cell_id", action["input"])

    def test_code_mode_wait_bridge_rejects_ambiguous_or_invalid_targets(self) -> None:
        tools = codex_tool_bridge.response_client_tools(codex_body(
            "hello",
            tools=[
                EXEC_TOOL,
                {
                    "type": "function",
                    "name": "wait",
                    "parameters": {
                        "type": "object",
                        "properties": {"cell_id": {"type": "string"}},
                        "required": ["cell_id"],
                        "additionalProperties": False,
                    },
                },
            ],
        ))
        invalid_arguments = [
            {"targets": []},
            {"targets": ["not-an-agent-id"]},
            {"targets": [V1_AGENT_ID], "unknown": True},
            {"targets": [V1_AGENT_ID], "timeout_ms": 0},
        ]
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                self.assertIsNone(codex_tool_bridge.parse_controller_action(json.dumps({
                    "action": "tool",
                    "name": "wait",
                    "arguments": arguments,
                }), tools))

    def test_final_completion_marker_is_explicit(self) -> None:
        tools = codex_tool_bridge.response_client_tools(codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [EXEC_TOOL]},
        ))

        incomplete = codex_tool_bridge.parse_controller_action(
            '{"action":"final","text":"I inspected one file"}',
            tools,
        )
        complete = codex_tool_bridge.parse_controller_action(
            '{"action":"final","text":"The task is complete","complete":true}',
            tools,
        )

        self.assertFalse(codex_tool_bridge.final_action_is_complete(incomplete))
        self.assertTrue(codex_tool_bridge.final_action_is_complete(complete))

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
                return_value=iter(['{"action":"final","text":"Event created","complete":true}']),
            ),
        ):
            final_events = list(openai_v1_response.handle(continuation))

        final_output = final_events[-1]["response"]["output"][0]
        self.assertEqual(final_output["type"], "message")
        self.assertEqual(final_output["content"][0]["text"], "Event created")

    def test_delta_only_tool_search_output_reveals_tool_and_output_schema(self) -> None:
        output_schema = {
            "type": "object",
            "properties": {"agent_id": {"type": "string"}},
            "required": ["agent_id"],
        }
        body = codex_body({
            "type": "tool_search_output",
            "call_id": "search_from_retained_history",
            "status": "completed",
            "execution": "client",
            "tools": [{
                "type": "namespace",
                "name": "multi_agent_v1",
                "tools": [{
                    "type": "function",
                    "name": "spawn_agent",
                    "defer_loading": True,
                    "parameters": {"type": "object"},
                    "output_schema": output_schema,
                }],
            }],
        })

        tools = codex_tool_bridge.response_client_tools(body)

        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["namespace"], "multi_agent_v1")
        self.assertFalse(tools[0]["defer_loading"])
        self.assertEqual(tools[0]["output_schema"], output_schema)

    def test_local_refusal_repairs_then_fails_without_fabricating_exec(self) -> None:
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
            with self.assertRaisesRegex(RuntimeError, "refusing to fabricate"):
                list(openai_v1_response.handle(body))

        self.assertEqual(stream.call_count, 2)

    def test_local_repair_plan_fails_without_inventing_exec(self) -> None:
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
            with self.assertRaisesRegex(RuntimeError, "refusing to fabricate"):
                list(openai_v1_response.handle(body))

        self.assertEqual(stream.call_count, 2)

    def test_followup_analysis_after_local_tool_call_still_requires_executor(self) -> None:
        environment = "<environment_context><cwd>C:\\project</cwd></environment_context>"
        body = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [EXEC_TOOL]},
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": environment}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "read the current project"}]},
            {"type": "custom_tool_call", "name": "exec", "call_id": "call-1", "input": "text('README.md')"},
            {"type": "custom_tool_call_output", "call_id": "call-1", "output": "README.md"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "深入分析"}]},
        )
        tools = codex_tool_bridge.response_client_tools(body)

        self.assertFalse(codex_tool_bridge.current_turn_has_tool_output(body["input"]))
        self.assertTrue(codex_tool_bridge.requires_local_tool(body, tools))

    def test_followup_analysis_after_local_prompt_still_requires_executor(self) -> None:
        environment = "<environment_context><cwd>C:\\project</cwd></environment_context>"
        body = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [EXEC_TOOL]},
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": environment}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "了解当前项目"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "深入分析"}]},
        )
        tools = codex_tool_bridge.response_client_tools(body)

        self.assertTrue(codex_tool_bridge.requires_local_tool(body, tools))

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
            with self.assertRaisesRegex(RuntimeError, "refusing to fabricate"):
                list(openai_v1_response.handle(body))

        self.assertEqual(stream.call_count, 2)

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
                {"type": "input_image", "image_url": f"data:image/png;base64,{PNG_1X1_B64}"},
            ]},
        )

        _model, messages = openai_v1_response.text_response_parts(body)

        transcript = "\n".join(
            str(message["content"])
            for message in messages
            if isinstance(message.get("content"), str)
        )
        media_messages = [
            message
            for message in messages
            if isinstance(message.get("content"), list)
        ]
        self.assertIn('"type": "custom_tool_call"', transcript)
        self.assertIn('"type": "custom_tool_call_output"', transcript)
        self.assertEqual(transcript.count('"call_id": "call_1"'), 2)
        self.assertIn('"name": "exec"', transcript)
        self.assertIn('"forwarded_to_upstream": true', transcript)
        self.assertNotIn(PNG_1X1_B64, transcript)
        self.assertEqual(len(media_messages), 1)
        image_part = next(part for part in media_messages[0]["content"] if part["type"] == "image")
        self.assertEqual(image_part["data"], base64.b64decode(PNG_1X1_B64))
        self.assertNotIn("MUST select one tool action", messages[0]["content"])

        backend = OpenAIBackendAPI(access_token="test-token")
        try:
            with mock.patch.object(backend, "_upload_image", return_value={
                "file_id": "file-image-1",
                "width": 1,
                "height": 1,
                "file_size": len(image_part["data"]),
                "mime_type": "image/png",
                "file_name": "image_1.png",
            }) as upload:
                payload = backend._conversation_payload(messages, "gpt-5.6-luna", "Asia/Shanghai")
        finally:
            backend.close()
        upload.assert_called_once()
        multimodal = next(
            message for message in payload["messages"]
            if message["content"]["content_type"] == "multimodal_text"
        )
        pointer = next(
            part for part in multimodal["content"]["parts"]
            if isinstance(part, dict) and part.get("content_type") == "image_asset_pointer"
        )
        self.assertEqual(pointer["asset_pointer"], "file-service://file-image-1")

    def test_controller_forwards_user_image_without_base64_transcript(self) -> None:
        body = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [EXEC_TOOL]},
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "inspect this screenshot"},
                {"type": "input_image", "image_url": f"data:image/png;base64,{PNG_1X1_B64}"},
            ]},
        )

        _model, messages = openai_v1_response.text_response_parts(body)

        transcript = "\n".join(
            str(message["content"])
            for message in messages
            if isinstance(message.get("content"), str)
        )
        self.assertIn("inspect this screenshot", transcript)
        self.assertNotIn(PNG_1X1_B64, transcript)
        self.assertTrue(any(
            isinstance(message.get("content"), list)
            and any(part.get("type") == "image" for part in message["content"] if isinstance(part, dict))
            for message in messages
        ))

    def test_input_audio_fails_explicitly_without_calling_upstream(self) -> None:
        body = codex_body({
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_audio",
                "input_audio": {"data": "AAAA", "format": "wav"},
            }],
        })

        with mock.patch("services.protocol.openai_v1_response.text_backend") as backend:
            chunks = list(responses_sse_stream(openai_v1_response.handle(body)))

        backend.assert_not_called()
        payloads = [
            json.loads(chunk[len("data:"):].strip())
            for chunk in chunks
            if chunk.startswith("data:") and "[DONE]" not in chunk
        ]
        failed = next(payload for payload in payloads if payload.get("type") == "response.failed")
        self.assertEqual(failed["response"]["error"]["code"], "invalid_prompt")
        self.assertIn("input_audio", failed["response"]["error"]["message"])

    def test_non_message_responses_state_items_are_lossless(self) -> None:
        items = [
            {
                "type": "agent_message",
                "id": "agent-msg-1",
                "author": "worker",
                "recipient": "parent",
                "content": [{"type": "input_text", "text": "done"}],
            },
            {
                "type": "reasoning",
                "id": "reasoning-1",
                "summary": [{"type": "summary_text", "text": "checked"}],
                "encrypted_content": "opaque-reasoning-state",
            },
            {
                "type": "context_compaction",
                "id": "compact-1",
                "encrypted_content": "opaque-compaction-state",
            },
        ]

        for item in items:
            history = codex_tool_bridge.response_item_history_message(item)
            self.assertIsNotNone(history)
            self.assertIn(item["type"], history["content"])
            self.assertIn(item.get("encrypted_content", item.get("id")), history["content"])

    def test_controller_transcript_preserves_agent_and_reasoning_state(self) -> None:
        body = codex_body(
            {
                "type": "agent_message",
                "id": "agent-msg-1",
                "author": "worker-1",
                "recipient": "parent",
                "content": [{"type": "input_text", "text": "inspection complete"}],
            },
            {
                "type": "reasoning",
                "id": "reasoning-1",
                "summary": [{"type": "summary_text", "text": "checked services"}],
                "content": [{"type": "reasoning_text", "text": "opaque"}],
                "encrypted_content": "encrypted-state-1",
            },
            {"type": "message", "role": "user", "content": "continue the task"},
        )

        transcript = "\n".join(
            str(message["content"])
            for message in codex_tool_bridge.controller_transcript_messages(body)
        )

        self.assertIn("CODEX_AGENT_MESSAGE_RECORD", transcript)
        self.assertIn("author=worker-1", transcript)
        self.assertIn("recipient=parent", transcript)
        self.assertIn("inspection complete", transcript)
        self.assertIn('"encrypted_content":"encrypted-state-1"', transcript)
        self.assertIn('"summary":[{"type":"summary_text","text":"checked services"}]', transcript)

    def test_large_controller_context_is_preloaded_without_dropping_environment_records(self) -> None:
        environment = (
            "<environment_context><cwd>C:\\project</cwd></environment_context>\n"
            + ("preserve this system record\n" * 1200)
        )
        body = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [EXEC_TOOL]},
            {"type": "message", "role": "developer", "content": environment},
            {"type": "message", "role": "developer", "content": environment},
            {"type": "message", "role": "user", "content": "read the current project"},
        )
        captured_messages: list[list[dict[str, object]]] = []
        cursors: list[tuple[str, str]] = []

        def fake_stream(_backend, request):
            messages = [dict(message) for message in request.messages]
            captured_messages.append(messages)
            cursors.append((request.conversation_id, request.parent_message_id))
            self.assertLess(
                openai_v1_response._controller_messages_wire_estimate(messages),
                64 * 1024,
            )
            request.conversation_id = "conv-preload"
            request.parent_message_id = f"node-{len(captured_messages)}"
            request.access_token = "token-preload"
            if any("CONTROLLER_CONTEXT_PRELOAD_COMPLETE" in str(item.get("content")) for item in messages):
                yield (
                    '{"action":"tool","name":"exec","input":'
                    '"const r = await tools.shell_command({command: \\"Get-ChildItem -Force\\"}); text(r);"}'
                )
            else:
                marker = next(
                    str(item.get("content") or "")
                    for item in messages
                    if "CONTROLLER_CONTEXT_PRELOAD batch=" in str(item.get("content") or "")
                )
                batch_index = int(marker.split("batch=", 1)[1].split("/", 1)[0])
                yield json.dumps({"context_preload_ack": batch_index})

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_response.stream_text_deltas", side_effect=fake_stream),
        ):
            events = list(openai_v1_response.handle(body))

        self.assertGreater(len(captured_messages), 1)
        self.assertEqual(cursors[0], ("", ""))
        self.assertTrue(all(cursor[0] == "conv-preload" for cursor in cursors[1:]))
        transported = "".join(
            str(message.get("content") or "")
            for batch in captured_messages
            for message in batch
        )
        self.assertEqual(transported.count("<environment_context>"), 2)
        self.assertEqual(events[-1]["type"], "response.completed")
        call_item = next(
            event["item"]
            for event in events
            if event["type"] == "response.output_item.done"
            and event["item"].get("type") == "custom_tool_call"
        )
        self.assertIn("Get-ChildItem -Force", call_item["input"])

    def test_large_plain_responses_context_is_preloaded_losslessly(self) -> None:
        environment = (
            "<environment_context><cwd>C:\\project</cwd></environment_context>\n"
            + ("preserve this ordinary system record\n" * 2200)
        )
        body = {
            "model": "gpt-5-6-luna",
            "stream": True,
            "tool_choice": "none",
            "input": "read the current project",
        }
        messages = [
            {"role": "system", "content": environment},
            {"role": "system", "content": environment},
            {"role": "user", "content": "read the current project"},
        ]
        captured_messages: list[list[dict[str, object]]] = []
        cursor_snapshots: list[tuple[str, str]] = []

        def fake_stream(_backend, request):
            captured_messages.append([dict(message) for message in request.messages or []])
            cursor_snapshots.append((request.conversation_id, request.parent_message_id))
            request.conversation_id = "conv-plain-preload"
            request.parent_message_id = f"node-{len(captured_messages)}"
            if any(
                "TEXT_CONTEXT_PRELOAD_COMPLETE" in str(message.get("content") or "")
                for message in request.messages or []
            ):
                yield "FINAL_PLAIN_TEXT"
            else:
                yield "INTERMEDIATE_MUST_NOT_LEAK"

        with mock.patch(
            "services.protocol.openai_v1_response.stream_text_deltas",
            side_effect=fake_stream,
        ):
            events = list(openai_v1_response.stream_text_response(object(), body, messages))

        self.assertGreater(len(captured_messages), 1)
        self.assertEqual(cursor_snapshots[0], ("", ""))
        self.assertTrue(all(cursor[0] == "conv-plain-preload" for cursor in cursor_snapshots[1:]))
        for batch in captured_messages:
            self.assertLess(
                openai_v1_response._controller_messages_wire_estimate(batch),
                64 * 1024,
            )
        transported = "".join(
            str(message.get("content") or "")
            for batch in captured_messages
            for message in batch
            if "TEXT_CONTEXT_PRELOAD" not in str(message.get("content") or "")
        )
        self.assertEqual(transported.count("<environment_context>"), 2)
        self.assertEqual(transported.count("preserve this ordinary system record"), 4400)
        self.assertNotIn("INTERMEDIATE_MUST_NOT_LEAK", json.dumps(events, ensure_ascii=False))
        self.assertIn("FINAL_PLAIN_TEXT", json.dumps(events, ensure_ascii=False))
        self.assertEqual(events[-1]["type"], "response.completed")

    def test_context_preload_rejects_tool_action_before_final_batch(self) -> None:
        request = ConversationRequest(
            model="gpt-5-6-luna",
            messages=[{"role": "system", "content": "oversized context " * 20_000}],
        )
        captured: list[ConversationRequest] = []

        def fake_stream(_backend, current_request):
            captured.append(current_request)
            current_request.conversation_id = "conv-preload-invalid"
            current_request.parent_message_id = f"node-{len(captured)}"
            current_request.access_token = "token-preload-invalid"
            yield '{"action":"tool","name":"exec","input":"text(\\"must-not-run\\")"}'

        with mock.patch(
            "services.protocol.openai_v1_response.stream_text_deltas",
            side_effect=fake_stream,
        ):
            with self.assertRaisesRegex(RuntimeError, "non-ack action"):
                openai_v1_response._stream_controller_request(
                    object(), request, attempt="invalid_preload"
                )

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].conversation_id, "conv-preload-invalid")
        self.assertNotIn("must-not-run", "".join(
            str(message.get("content") or "")
            for message in captured[0].messages
        ))

    def test_saturated_controller_cursor_is_replaced_by_bounded_checkpoint(self) -> None:
        first = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [EXEC_TOOL]},
            {"type": "message", "role": "user", "content": "inspect the project"},
        )
        first["model"] = "gpt-5-6-luna"
        first.update({
            "_request_identity_key_id": "admin",
            "client_metadata": {
                "session_id": "session-budget",
                "thread_id": "thread-budget",
                "turn_id": "turn-1",
                "window_id": "window-budget",
            },
        })
        captured: list[ConversationRequest] = []
        cursor_snapshots: list[tuple[str, str]] = []

        def fake_stream(_backend, current_request):
            captured.append(current_request)
            cursor_snapshots.append((current_request.conversation_id, current_request.parent_message_id))
            current_request.conversation_id = "conv-budget"
            current_request.parent_message_id = f"node-{len(captured)}"
            current_request.access_token = "token-budget"
            if len(captured) == 1:
                yield json.dumps({
                    "action": "tool",
                    "name": "exec",
                    "input": "const r = await tools.shell_command({command: 'Get-ChildItem'}); text(r);",
                })
            else:
                yield '{"action":"final","text":"the project is inspected","complete":true}'

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
            mock.patch.object(
                openai_v1_response,
                "CONTROLLER_SESSION_TARGET_WIRE_BYTES",
                1,
            ),
        ):
            first_events = list(openai_v1_response.handle(first))
            call_item = next(
                event["item"]
                for event in first_events
                if event.get("type") == "response.output_item.done"
                and event["item"].get("type") == "custom_tool_call"
            )
            second = {
                **first,
                "previous_response_id": first_events[-1]["response"]["id"],
                "client_metadata": {**first["client_metadata"], "turn_id": "turn-2"},
                "input": [
                    *first["input"],
                    call_item,
                    {
                        "type": "custom_tool_call_output",
                        "call_id": call_item["call_id"],
                        "output": "README.md",
                    },
                ],
            }
            second_events = list(openai_v1_response.handle(second))

        self.assertEqual(len(captured), 2)
        self.assertEqual(cursor_snapshots[0], ("", ""))
        self.assertEqual(cursor_snapshots[1], ("", ""))
        checkpoint_text = "\n".join(
            str(message.get("content") or "")
            for message in captured[1].messages
        )
        self.assertIn("CONTROLLER_CONTEXT_COMPACTION", checkpoint_text)
        self.assertNotIn("conv-budget", checkpoint_text)
        self.assertEqual(second_events[-1]["type"], "response.completed")

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
            yield '{"action":"final","text":"README.md is present","complete":true}'

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
                yield '{"action":"final","text":"README.md is present","complete":true}'

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
        self.assertIn("CONTROLLER_TASK_CONTRACT", continuation_text)
        self.assertIn("CODEX_TASK_ANCHOR active_user_request", continuation_text)
        self.assertIn("read the current project", continuation_text)
        self.assertIn("force_progress_tool=false", continuation_text)
        final_output = second_events[-1]["response"]["output"][0]
        self.assertEqual(final_output["content"][0]["text"], "README.md is present")

    def test_delta_input_with_previous_response_keeps_task_context(self) -> None:
        first = codex_body(
            {"type": "message", "role": "user", "content": "inspect the current project"},
        )
        first.update({
            "_request_identity_key_id": "admin",
            "client_metadata": {"session_id": "session-delta", "thread_id": "thread-delta", "turn_id": "turn-1"},
        })
        tool_call = {
            "type": "custom_tool_call",
            "name": "exec",
            "call_id": "call-delta",
            "input": "const r = await tools.shell_command({command: 'Get-ChildItem'}); text(r);",
        }
        tool_output = {
            "type": "custom_tool_call_output",
            "call_id": "call-delta",
            "output": "README.md",
        }
        tools = codex_tool_bridge.response_client_tools({**first, "tools": [EXEC_TOOL]})
        store = codex_conversation_session._SessionStore()
        with (
            mock.patch.object(
                codex_conversation_session.account_service,
                "resolve_access_token",
                side_effect=lambda token: token,
            ),
            mock.patch.object(
                codex_conversation_session.account_service,
                "get_account",
                return_value={"status": "normal"},
            ),
        ):
            initial_plan = store.prepare(first, tools, [{"role": "system", "content": "full"}], force_tool=False)
            self.assertTrue(store.commit(
                initial_plan,
                first,
                tools,
                [tool_call],
                conversation_id="conv-delta",
                parent_message_id="node-delta",
                access_token="token-delta",
                response_id="resp-delta",
                usage={},
            ))
            continuation = {
                **first,
                "previous_response_id": "resp-delta",
                "client_metadata": {**first["client_metadata"], "turn_id": "turn-2"},
                "input": [tool_call, tool_output],
            }
            plan = store.prepare(
                continuation,
                tools,
                [{"role": "system", "content": "full"}],
                force_tool=False,
            )

        self.assertTrue(plan.continued)
        self.assertEqual(plan.conversation_id, "conv-delta")
        self.assertEqual(plan.parent_message_id, "node-delta")
        self.assertEqual(plan.delta_input_items, 1)
        continuation_text = "\n".join(str(message["content"]) for message in plan.messages)
        self.assertIn("inspect the current project", continuation_text)
        self.assertIn("custom_tool_call_output", continuation_text)

    def test_switching_model_and_instructions_keeps_codex_thread_cursor(self) -> None:
        first = codex_body({
            "type": "message",
            "role": "user",
            "content": "continue the repository task",
        })
        first.update({
            "model": "auto",
            "instructions": "first model instructions",
            "_request_identity_key_id": "admin",
            "client_metadata": {
                "session_id": "session-switch",
                "thread_id": "thread-switch",
                "turn_id": "turn-1",
            },
        })
        tools = codex_tool_bridge.response_client_tools({**first, "tools": [EXEC_TOOL]})
        store = codex_conversation_session._SessionStore()
        with (
            mock.patch.object(
                codex_conversation_session.account_service,
                "resolve_access_token",
                side_effect=lambda token: token,
            ),
            mock.patch.object(
                codex_conversation_session.account_service,
                "get_account",
                return_value={"status": "normal"},
            ),
        ):
            initial = store.prepare(first, tools, [{"role": "system", "content": "full"}], force_tool=False)
            self.assertTrue(store.commit(
                initial,
                first,
                tools,
                [{"type": "custom_tool_call", "name": "exec", "call_id": "call-switch", "input": "text('ok')"}],
                conversation_id="conv-switch",
                parent_message_id="node-switch",
                access_token="token-switch",
                response_id="resp-switch",
                usage={},
            ))
            second = {
                **first,
                "model": "gpt-5-6-luna",
                "instructions": "new model instructions",
                "previous_response_id": "resp-switch",
                "client_metadata": {**first["client_metadata"], "turn_id": "turn-2"},
                "input": [
                    *first["input"],
                    {"type": "custom_tool_call", "name": "exec", "call_id": "call-switch", "input": "text('ok')"},
                    {"type": "custom_tool_call_output", "call_id": "call-switch", "output": "ok"},
                ],
            }
            plan = store.prepare(second, tools, [{"role": "system", "content": "full"}], force_tool=False)

        self.assertTrue(plan.continued)
        self.assertEqual(plan.conversation_id, "conv-switch")
        self.assertEqual(plan.parent_message_id, "node-switch")

    def test_new_compaction_checkpoint_resets_retained_web_cursor(self) -> None:
        store = codex_conversation_session._SessionStore()
        first = codex_body(
            {"type": "message", "role": "user", "content": "inspect the project"},
        )
        first.update({
            "_request_identity_key_id": "admin",
            "client_metadata": {
                "session_id": "session-compact",
                "thread_id": "thread-compact",
                "turn_id": "turn-1",
            },
        })
        tools: list[dict[str, object]] = []
        with (
            mock.patch.object(
                codex_conversation_session.account_service,
                "resolve_access_token",
                side_effect=lambda token: token,
            ),
            mock.patch.object(
                codex_conversation_session.account_service,
                "get_account",
                return_value={"status": "normal"},
            ),
        ):
            initial = store.prepare(first, tools, [{"role": "system", "content": "full"}], force_tool=False)
            self.assertTrue(store.commit(
                initial,
                first,
                tools,
                [{"type": "message", "role": "assistant", "content": "old answer"}],
                conversation_id="conv-before-compact",
                parent_message_id="node-before-compact",
                access_token="token-compact",
                response_id="resp-before-compact",
                usage={},
            ))
            compacted = {
                **first,
                "previous_response_id": "resp-before-compact",
                "client_metadata": {**first["client_metadata"], "turn_id": "turn-2"},
                "input": [
                    {"type": "message", "role": "user", "content": "inspect the project"},
                    {"type": "compaction", "id": "cmp-new", "encrypted_content": "summary"},
                ],
            }
            plan = store.prepare(
                compacted,
                tools,
                [{"role": "system", "content": "compacted replay"}],
                force_tool=False,
            )

        self.assertFalse(plan.continued)
        self.assertEqual(plan.conversation_id, "")
        self.assertEqual(plan.parent_message_id, "")
        self.assertEqual(plan.canonical_input_items, compacted["input"])

    def test_local_compaction_window_replaces_cursor_without_typed_item(self) -> None:
        store = codex_conversation_session._SessionStore()
        first = codex_body(
            {"type": "message", "role": "user", "content": "inspect the project"},
        )
        first.update({
            "_request_identity_key_id": "admin",
            "client_metadata": {
                "session_id": "session-local-compact",
                "thread_id": "thread-local-compact",
                "turn_id": "turn-1",
                "window_id": "thread-local-compact:0",
            },
        })
        tools: list[dict[str, object]] = []
        with (
            mock.patch.object(
                codex_conversation_session.account_service,
                "resolve_access_token",
                side_effect=lambda token: token,
            ),
            mock.patch.object(
                codex_conversation_session.account_service,
                "get_account",
                return_value={"status": "normal"},
            ),
        ):
            initial = store.prepare(first, tools, [{"role": "system", "content": "full"}], force_tool=False)
            self.assertTrue(store.commit(
                initial,
                first,
                tools,
                [{"type": "message", "role": "assistant", "content": "old answer"}],
                conversation_id="conv-local-compact",
                parent_message_id="node-local-compact",
                access_token="token-local-compact",
                response_id="resp-local-compact",
                usage={},
            ))
            replacement = {
                **first,
                "previous_response_id": "resp-local-compact",
                "input": [{
                    "type": "message",
                    "role": "user",
                    "content": "<summary_prefix>\nproject summary",
                }],
                "client_metadata": {
                    **first["client_metadata"],
                    "turn_id": "turn-2",
                    "window_id": "thread-local-compact:1",
                },
            }
            plan = store.prepare(
                replacement,
                tools,
                [{"role": "system", "content": "compacted replay"}],
                force_tool=False,
            )

        self.assertFalse(plan.continued)
        self.assertEqual(plan.conversation_id, "")
        self.assertEqual(plan.parent_message_id, "")
        self.assertEqual(plan.canonical_input_items, replacement["input"])

    def test_task_contract_keeps_analysis_request_valid_without_code_change(self) -> None:
        body = codex_body(
            {"type": "message", "role": "user", "content": "详细了解当前项目，给出优化方案"},
        )
        tools = codex_tool_bridge.response_client_tools({
            **body,
            "tools": [{"type": "custom", "name": "exec", "description": "Run local commands"}],
        })
        messages = codex_tool_bridge.controller_messages({**body, "input": body["input"]}, tools)
        transcript = "\n".join(str(message["content"]) for message in messages)

        self.assertIn("task_kind=analysis_or_planning", transcript)
        self.assertIn("A response claiming that no coding task", transcript)
        self.assertIn("code modification is not required", transcript.lower())
        self.assertTrue(codex_tool_bridge.is_task_evasion(
            "No coding task or requested change was included in the provided context."
        ))

    def test_broad_project_task_does_not_inject_an_arbitrary_third_command(self) -> None:
        body = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [EXEC_TOOL]},
            {
                "type": "message",
                "role": "user",
                "content": "Provide a detailed analysis of the current project and an optimization plan.",
            },
            {"type": "custom_tool_call", "name": "exec", "call_id": "scan-1", "input": "const r = await tools.shell_command({command: \"rg --files\"}); text(r);"},
            {"type": "custom_tool_call_output", "call_id": "scan-1", "output": "README.md\nmain.py"},
            {"type": "custom_tool_call", "name": "exec", "call_id": "read-1", "input": "const r = await tools.shell_command({command: \"Get-Content README.md\"}); text(r);"},
            {"type": "custom_tool_call_output", "call_id": "read-1", "output": "project overview"},
        )
        outputs = iter(['{"action":"final","text":"Evidence-based review","complete":true}'])

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch(
                "services.protocol.openai_v1_response.stream_text_deltas",
                side_effect=lambda _backend, _request: iter([next(outputs)]),
            ) as stream,
        ):
            events = list(openai_v1_response.handle(body))

        self.assertEqual(stream.call_count, 1)
        item = next(event["item"] for event in events if event["type"] == "response.output_item.done")
        self.assertEqual(item["type"], "message")
        self.assertEqual(item["content"][0]["text"], "Evidence-based review")

    def test_plain_text_after_tool_result_is_repaired_before_final(self) -> None:
        body = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [EXEC_TOOL]},
            {"type": "message", "role": "user", "content": "inspect the project"},
            {"type": "custom_tool_call", "name": "exec", "call_id": "call-1", "input": "text('ok')"},
            {"type": "custom_tool_call_output", "call_id": "call-1", "output": "README.md"},
        )

        outputs = iter([
            "README.md is present",
            '{"action":"final","text":"README.md is present","complete":true}',
        ])

        def fake_stream(_backend, _request):
            yield next(outputs)

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch(
                "services.protocol.openai_v1_response.stream_text_deltas",
                side_effect=fake_stream,
            ) as stream,
        ):
            events = list(openai_v1_response.handle(body))

        self.assertEqual(stream.call_count, 2)
        self.assertEqual(events[-1]["response"]["output"][0]["content"][0]["text"], "README.md is present")

    def test_task_evasion_is_repaired_with_next_tool_instead_of_bootstrap_repeat(self) -> None:
        first = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [EXEC_TOOL]},
            {"type": "message", "role": "user", "content": "详细了解当前项目，给出优化方案"},
        )
        first.update({
            "prompt_cache_key": "task-contract-session",
            "_request_identity_key_id": "admin",
            "client_metadata": {"session_id": "session-contract", "thread_id": "thread-contract", "turn_id": "turn-1"},
        })
        first_outputs = iter([
            '{"action":"tool","name":"exec","input":"const r = await tools.shell_command({command: \'Get-ChildItem -Force\'}); text(r);"}',
        ])
        captured = []

        def first_stream(_backend, request):
            captured.append(request)
            request.conversation_id = "conv-contract"
            request.parent_message_id = "node-1"
            request.access_token = "token-contract"
            yield next(first_outputs)

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_response.stream_text_deltas", side_effect=first_stream),
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
        continuation = {
            **first,
            "input": [
                *first["input"],
                call_item,
                {"type": "custom_tool_call_output", "call_id": call_item["call_id"], "output": "README.md"},
            ],
        }
        continuation["client_metadata"] = {**first["client_metadata"], "turn_id": "turn-2"}
        continuation_outputs = iter([
            "No coding task or requested change was included in the provided context.",
            '{"action":"tool","name":"exec","input":"const r = await tools.shell_command({command: \'Get-Content README.md -TotalCount 20\'}); text(r);"}',
        ])

        def continuation_stream(_backend, request):
            captured.append(request)
            request.conversation_id = "conv-contract"
            request.parent_message_id = "node-2"
            request.access_token = "token-contract"
            yield next(continuation_outputs)

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_response.stream_text_deltas", side_effect=continuation_stream),
            mock.patch(
                "services.protocol.codex_conversation_session.account_service.resolve_access_token",
                side_effect=lambda token: token,
            ),
            mock.patch(
                "services.protocol.codex_conversation_session.account_service.get_account",
                return_value={"status": "normal"},
            ),
        ):
            events = list(openai_v1_response.handle(continuation))

        self.assertEqual(len(captured), 3)
        repair_text = "\n".join(str(message["content"]) for message in captured[2].messages)
        self.assertIn("TOOL_DEFINITION_RECORD", repair_text)
        self.assertIn("CONTROLLER_TASK_CONTRACT", repair_text)
        self.assertIn("CONTROLLER_REPAIR_RECORD", repair_text)
        item = next(event["item"] for event in events if event["type"] == "response.output_item.done")
        self.assertEqual(item["type"], "custom_tool_call")
        self.assertIn("Get-Content README.md", item["input"])
        self.assertNotIn("Get-ChildItem -Force", item["input"])

    def test_two_invalid_followup_outputs_fail_without_generic_tool(self) -> None:
        body = codex_body(
            {"type": "additional_tools", "role": "developer", "tools": [EXEC_TOOL]},
            {"type": "message", "role": "user", "content": "inspect the current project"},
            {"type": "custom_tool_call", "name": "exec", "call_id": "call-1", "input": "text('listed')"},
            {"type": "custom_tool_call_output", "call_id": "call-1", "output": "README.md"},
        )
        outputs = iter([
            "No coding task or requested change was included in the provided context.",
            "No coding task or requested change was included in the provided context.",
        ])

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_response.stream_text_deltas", side_effect=lambda _backend, _request: iter([next(outputs)])) as stream,
        ):
            with self.assertRaisesRegex(RuntimeError, "refusing to fabricate"):
                list(openai_v1_response.handle(body))

        self.assertEqual(stream.call_count, 2)

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

    def test_reasoning_and_unknown_response_items_are_retained_in_plain_history(self) -> None:
        messages = openai_v1_response.messages_from_input([
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "checked the request"}]},
            {"type": "compaction", "encrypted_content": "opaque"},
            {"type": "message", "role": "user", "content": "continue"},
        ])

        self.assertEqual(messages[0]["role"], "assistant")
        self.assertIn("REASONING_SUMMARY_RECORD", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertIn('"type":"compaction"', messages[1]["content"])
        self.assertEqual(messages[-1], {"role": "user", "content": "continue"})

    def test_mcp_tool_output_is_seen_as_completed_history(self) -> None:
        call = {"type": "function_call", "name": "lookup", "call_id": "mcp-call", "arguments": "{}"}
        output = {
            "type": "mcp_tool_call_output",
            "call_id": "mcp-call",
            "output": {"content": [{"type": "text", "text": "result"}]},
        }

        self.assertTrue(codex_tool_bridge.current_turn_has_tool_output([
            {"type": "message", "role": "user", "content": "lookup"},
            call,
            output,
        ]))
        self.assertTrue(codex_tool_bridge.action_repeats_completed_tool(
            {"action": "tool", "kind": "function", "name": "lookup", "arguments": {}},
            [call, output],
        ))
        history = openai_v1_response.messages_from_input([call, output])
        self.assertIn("EXTERNAL_TOOL_RESULT_RECORD", history[-1]["content"])

    def test_completed_custom_tool_input_is_rejected_within_current_turn(self) -> None:
        tool_input = 'const r = await tools.shell_command({command: "Get-ChildItem"}); text(r);'
        call = {
            "type": "custom_tool_call",
            "name": "exec",
            "call_id": "custom-call",
            "input": tool_input,
        }
        output = {
            "type": "custom_tool_call_output",
            "call_id": "custom-call",
            "output": "README.md",
        }

        self.assertTrue(codex_tool_bridge.action_repeats_completed_tool(
            {"action": "tool", "kind": "custom", "name": "exec", "input": tool_input},
            [
                {"type": "message", "role": "user", "content": "inspect"},
                call,
                output,
            ],
        ))

    def test_equivalent_local_shell_wrapper_is_rejected_across_tool_kinds(self) -> None:
        command = "Get-Content README.md -TotalCount 200"
        custom_input = (
            'const result = await tools.shell_command({command: '
            '"Get-Content   README.md   -TotalCount 200"}); text(result);'
        )
        call = {
            "type": "custom_tool_call",
            "name": "exec",
            "call_id": "custom-shell-call",
            "input": custom_input,
        }
        output = {
            "type": "custom_tool_call_output",
            "call_id": "custom-shell-call",
            "output": "README.md",
        }

        self.assertTrue(codex_tool_bridge.action_repeats_completed_tool(
            {
                "action": "tool",
                "kind": "function",
                "name": "shell_command",
                "arguments": json.dumps({"command": command}),
            },
            [
                {"type": "message", "role": "user", "content": "inspect"},
                call,
                output,
            ],
        ))

    def test_equivalent_exec_command_with_reordered_fields_is_rejected(self) -> None:
        completed_input = (
            'const command = "Get-ChildItem -Force"; '
            'const r = await tools.shell_command({workdir: "C:/repo", command}); text(r);'
        )
        repeated_input = (
            'const r = await tools.shell_command({command: "Get-ChildItem -Force", '
            'workdir: "C:\\\\repo"}); text(r);'
        )
        call = {
            "type": "custom_tool_call",
            "name": "exec",
            "call_id": "reordered-call",
            "input": completed_input,
        }
        output = {
            "type": "custom_tool_call_output",
            "call_id": "reordered-call",
            "output": "files",
        }

        self.assertTrue(codex_tool_bridge.action_repeats_completed_tool(
            {"action": "tool", "kind": "custom", "name": "exec", "input": repeated_input},
            [{"type": "message", "role": "user", "content": "inspect"}, call, output],
        ))

    def test_read_is_allowed_again_after_mutating_tool(self) -> None:
        read_input = (
            'const r = await tools.shell_command({command: "Get-Content README.md"}); text(r);'
        )
        items = [
            {"type": "message", "role": "user", "content": "edit and verify"},
            {"type": "custom_tool_call", "name": "exec", "call_id": "read-1", "input": read_input},
            {"type": "custom_tool_call_output", "call_id": "read-1", "output": "before"},
            {
                "type": "custom_tool_call",
                "name": "apply_patch",
                "call_id": "patch-1",
                "input": "*** Begin Patch\\n*** End Patch",
            },
            {"type": "custom_tool_call_output", "call_id": "patch-1", "output": "Done"},
        ]

        self.assertFalse(codex_tool_bridge.action_repeats_completed_tool(
            {"action": "tool", "kind": "custom", "name": "exec", "input": read_input},
            items,
        ))

    def test_same_tool_input_is_allowed_for_a_new_user_turn(self) -> None:
        tool_input = 'const r = await tools.shell_command({command: "Get-ChildItem"}); text(r);'
        call = {
            "type": "custom_tool_call",
            "name": "exec",
            "call_id": "old-call",
            "input": tool_input,
        }
        output = {
            "type": "custom_tool_call_output",
            "call_id": "old-call",
            "output": "README.md",
        }

        self.assertFalse(codex_tool_bridge.action_repeats_completed_tool(
            {"action": "tool", "kind": "custom", "name": "exec", "input": tool_input},
            [
                {"type": "message", "role": "user", "content": "inspect"},
                call,
                output,
                {"type": "message", "role": "user", "content": "inspect again"},
            ],
            seed_items=[call],
        ))

    def test_wait_and_list_agent_polling_are_repeatable_in_same_turn(self) -> None:
        for name in ("wait_agent", "list_agents"):
            call = {
                "type": "function_call",
                "namespace": "collaboration",
                "name": name,
                "call_id": f"{name}-call",
                "arguments": "{}",
            }
            output = {
                "type": "function_call_output",
                "call_id": f"{name}-call",
                "output": "no activity",
            }
            self.assertFalse(codex_tool_bridge.action_repeats_completed_tool(
                {
                    "action": "tool",
                    "kind": "function",
                    "namespace": "collaboration",
                    "name": name,
                    "input": "{}",
                },
                [
                    {"type": "message", "role": "user", "content": "delegate"},
                    call,
                    output,
                ],
            ))

    def test_responses_lite_nested_web_search_tool_is_detected(self) -> None:
        body = codex_body({
            "type": "additional_tools",
            "role": "developer",
            "tools": [{"type": "namespace", "name": "functions", "tools": [
                {"type": "web_search_preview", "search_context_size": "high"},
            ]}],
        })

        self.assertTrue(web_search_tool.has_web_search_tool(body))

    def test_nested_web_search_does_not_hide_other_client_tools(self) -> None:
        body = codex_body({
            "type": "additional_tools",
            "role": "developer",
            "tools": [{"type": "namespace", "name": "functions", "tools": [
                {"type": "function", "name": "shell_command", "parameters": {"type": "object"}},
                {"type": "web_search_preview"},
            ]}],
        })

        self.assertTrue(web_search_tool.has_web_search_tool(body))
        self.assertTrue(web_search_tool.has_unsupported_tools(body, web_search_tool.WEB_SEARCH_TOOL_TYPES))

    def test_mixed_hosted_search_can_continue_with_a_client_tool(self) -> None:
        body = codex_body(
            {"type": "message", "role": "user", "content": "check the current release, then inspect the project"},
            tools=[EXEC_TOOL, {"type": "web_search_preview"}],
        )
        outputs = iter([
            '{"action":"tool","name":"web_search","arguments":{"query":"current release"}}',
            '{"action":"tool","name":"exec","input":"const r = await tools.shell_command({command: \'Get-ChildItem -Force\'}); text(r);"}',
        ])
        search_result = {
            "answer": "Version 2 is current.",
            "sources": [{"title": "Release", "url": "https://example.test/release"}],
        }

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch(
                "services.protocol.openai_v1_response.stream_text_deltas",
                side_effect=lambda _backend, _request: iter([next(outputs)]),
            ) as stream,
            mock.patch(
                "services.protocol.openai_v1_response.run_web_search",
                return_value=search_result,
            ) as search,
        ):
            events = list(openai_v1_response.handle(body))

        self.assertEqual(stream.call_count, 2)
        search.assert_called_once_with("current release")
        done_items = [
            event["item"]
            for event in events
            if event["type"] == "response.output_item.done"
        ]
        self.assertEqual([item["type"] for item in done_items], ["web_search_call", "custom_tool_call"])
        self.assertFalse(events[-1]["response"]["end_turn"])

    def test_mixed_hosted_search_can_finish_in_the_same_response(self) -> None:
        body = codex_body(
            {"type": "message", "role": "user", "content": "look up the current release"},
            tools=[EXEC_TOOL, {"type": "web_search"}],
        )
        outputs = iter([
            '{"action":"tool","name":"web_search","arguments":{"query":"current release"}}',
            '{"action":"final","text":"Version 2 is current.","complete":true}',
        ])

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch(
                "services.protocol.openai_v1_response.stream_text_deltas",
                side_effect=lambda _backend, _request: iter([next(outputs)]),
            ),
            mock.patch(
                "services.protocol.openai_v1_response.run_web_search",
                return_value={"answer": "Version 2 is current.", "sources": []},
            ),
        ):
            events = list(openai_v1_response.handle(body))

        completed = events[-1]["response"]
        self.assertEqual([item["type"] for item in completed["output"]], ["web_search_call", "message"])
        self.assertEqual(completed["output"][1]["content"][0]["text"], "Version 2 is current.")
        self.assertTrue(completed["end_turn"])

    def test_tool_choice_none_disables_hosted_web_search(self) -> None:
        body = codex_body(
            {"type": "message", "role": "user", "content": "hello"},
            tools=[{"type": "web_search"}],
        )
        body["tool_choice"] = "none"

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch(
                "services.protocol.openai_v1_response.stream_text_deltas",
                return_value=iter(["hello"]),
            ),
            mock.patch("services.protocol.openai_v1_response.run_web_search") as search,
        ):
            events = list(openai_v1_response.handle(body))

        search.assert_not_called()
        self.assertEqual(events[-1]["response"]["output"][0]["type"], "message")

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

    def test_responses_stream_emits_json_heartbeat_while_source_blocks(self) -> None:
        def delayed_stream():
            yield openai_v1_response.response_created("resp_heartbeat", "gpt-5.6-luna", 1)
            time.sleep(0.12)
            yield openai_v1_response.response_completed(
                "resp_heartbeat",
                "gpt-5.6-luna",
                1,
                [],
            )

        with mock.patch.dict(
            "os.environ",
            {"CHATGPT2API_RESPONSES_SSE_HEARTBEAT_SECONDS": "0.05"},
        ):
            payloads = [
                chunk
                for chunk in responses_sse_stream(delayed_stream())
                if chunk.startswith("data:")
            ]
        decoded = [json.loads(chunk[5:].strip()) for chunk in payloads if "[DONE]" not in chunk]

        self.assertTrue(any(item.get("type") == "response.in_progress" for item in decoded))
        self.assertEqual(decoded[-1]["type"], "response.completed")

    def test_responses_stream_disconnect_cancels_blocking_source(self) -> None:
        registered = threading.Event()
        released = threading.Event()

        def blocking_stream():
            yield openai_v1_response.response_created("resp_cancel", "gpt-5.6-luna", 1)
            cancellation = current_stream_cancellation()
            self.assertIsNotNone(cancellation)
            registration = cancellation.register(released.set)
            registered.set()
            released.wait(timeout=2)
            cancellation.unregister(registration)

        with mock.patch.dict(
            "os.environ",
            {"CHATGPT2API_RESPONSES_SSE_HEARTBEAT_SECONDS": "0.05"},
        ):
            stream = responses_sse_stream(blocking_stream())
            self.assertEqual(next(stream), ": stream-open\n\n")
            self.assertIn("response.created", next(stream))
            self.assertTrue(registered.wait(timeout=1))
            self.assertIn("response.in_progress", next(stream))
            stream.close()

        self.assertTrue(released.wait(timeout=1))
        self.assertFalse(any(
            thread.name == "responses-sse-source" and thread.is_alive()
            for thread in threading.enumerate()
        ))

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

    def test_responses_stream_marks_deterministic_4xx_as_non_retryable(self) -> None:
        for status in (400, 401, 403, 404, 422):
            with self.subTest(status=status):
                def failed_stream(status=status):
                    yield openai_v1_response.response_created("resp_4xx", "gpt-5.6-luna", 1)
                    raise UpstreamHTTPError("/backend-api/conversation", status, "")

                payloads = [
                    chunk
                    for chunk in responses_sse_stream(failed_stream())
                    if chunk.startswith("data:")
                ]
                decoded = [
                    json.loads(chunk[5:].strip())
                    for chunk in payloads
                    if "[DONE]" not in chunk
                ]
                error = decoded[-1]["response"]["error"]

                self.assertEqual(decoded[-1]["type"], "response.failed")
                self.assertEqual(error["type"], "invalid_request_error")
                self.assertEqual(error["code"], "invalid_prompt")

    def test_responses_stream_preserves_upstream_context_error_code(self) -> None:
        def failed_stream():
            yield openai_v1_response.response_created("resp_context", "gpt-5.6-luna", 1)
            raise UpstreamHTTPError(
                "/backend-api/conversation",
                400,
                {"error": {
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded",
                    "message": "context window exceeded",
                }},
            )

        payloads = [chunk for chunk in responses_sse_stream(failed_stream()) if chunk.startswith("data:")]
        decoded = [json.loads(chunk[5:].strip()) for chunk in payloads if "[DONE]" not in chunk]
        error = decoded[-1]["response"]["error"]

        self.assertEqual(decoded[-1]["type"], "response.failed")
        self.assertEqual(error["type"], "invalid_request_error")
        self.assertEqual(error["code"], "context_length_exceeded")
        self.assertEqual(error["message"], "context window exceeded")

    def test_responses_stream_preserves_upstream_quota_and_rate_limit_codes(self) -> None:
        cases = [
            (403, "insufficient_quota", "insufficient quota"),
            (429, "rate_limit_exceeded", "try again in 7 seconds"),
        ]
        for status, code, message in cases:
            with self.subTest(status=status):
                def failed_stream(status=status, code=code, message=message):
                    yield openai_v1_response.response_created("resp_error", "gpt-5.6-luna", 1)
                    raise UpstreamHTTPError(
                        "/backend-api/conversation",
                        status,
                        {"error": {"code": code, "message": message}},
                    )

                payloads = [chunk for chunk in responses_sse_stream(failed_stream()) if chunk.startswith("data:")]
                decoded = [json.loads(chunk[5:].strip()) for chunk in payloads if "[DONE]" not in chunk]
                error = decoded[-1]["response"]["error"]
                self.assertEqual(error["code"], code)
                self.assertEqual(error["message"], message)

    def test_image_message_response_adds_item_before_text_delta(self) -> None:
        events = list(openai_v1_response.stream_image_response(
            [ImageOutput(kind="message", model="gpt-image-2", index=1, total=1, text="blocked")],
            "draw",
            "gpt-image-2",
        ))

        self.assertEqual(
            [event["type"] for event in events],
            [
                "response.created",
                "response.output_item.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.output_item.done",
                "response.completed",
            ],
        )
        item_id = events[1]["item"]["id"]
        self.assertEqual(events[2]["item_id"], item_id)
        self.assertEqual(events[4]["item"]["id"], item_id)

    def test_compact_endpoint_returns_codex_compaction_item_without_upstream(self) -> None:
        body = codex_body(
            {"type": "message", "role": "developer", "content": "<environment_context><cwd>C:\\project</cwd></environment_context>"},
            {"type": "message", "role": "user", "content": "inspect the project"},
            {"type": "function_call", "name": "shell_command", "call_id": "call-1", "arguments": "{\"command\":\"Get-ChildItem\"}"},
            {"type": "function_call_output", "call_id": "call-1", "output": "README.md"},
        )
        with mock.patch.object(openai_v1_response, "_upstream_compaction_summary", return_value=""):
            result = openai_v1_response.compact(body)

        self.assertEqual(len(result["output"]), 1)
        item = result["output"][0]
        self.assertEqual(item["type"], "compaction")
        self.assertIn("environment_context", item["encrypted_content"])
        self.assertIn("README.md", item["encrypted_content"])

    def test_compaction_trigger_returns_exactly_one_compaction_sse_item(self) -> None:
        body = codex_body(
            {"type": "message", "role": "user", "content": "long task context"},
            {"type": "compaction_trigger"},
        )
        body["_response_id"] = "resp_compaction_v2"
        with mock.patch.object(
            openai_v1_response,
            "_upstream_compaction_summary",
            return_value="compacted task state",
        ):
            events = list(openai_v1_response.handle(body))

        completed_items = [
            event["item"]
            for event in events
            if event["type"] == "response.output_item.done"
        ]
        self.assertEqual(len(completed_items), 1)
        self.assertEqual(completed_items[0]["type"], "compaction")
        self.assertIn("compacted task state", completed_items[0]["encrypted_content"])
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(events[-1]["response"]["output"], completed_items)
        self.assertNotIn("compaction_trigger", completed_items[0]["encrypted_content"])


if __name__ == "__main__":
    unittest.main()
