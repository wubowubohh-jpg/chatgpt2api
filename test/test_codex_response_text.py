from __future__ import annotations

import unittest
from unittest import mock

from services.protocol import (
    codex_conversation_session,
    codex_response_text,
    codex_tool_bridge,
    openai_v1_response,
)
from services.protocol.chat_completion_cache import cache_key


def _body(*, instructions: str = "base instructions", text: object = None) -> dict[str, object]:
    body: dict[str, object] = {
        "model": "gpt-5.6-luna",
        "instructions": instructions,
        "input": [{"type": "message", "role": "user", "content": "return the result"}],
        "prompt_cache_key": "text-control-session",
        "_request_identity_key_id": "admin",
        "client_metadata": {
            "session_id": "session-text-controls",
            "thread_id": "thread-text-controls",
            "turn_id": "turn-1",
        },
    }
    if text is not None:
        body["text"] = text
    return body


def _commit_initial(
    store: codex_conversation_session._SessionStore,
    body: dict[str, object],
) -> codex_conversation_session.ContinuationPlan:
    plan = store.prepare(
        body,
        [],
        [{"role": "system", "content": "controller protocol"}],
        force_tool=False,
    )
    committed = store.commit(
        plan,
        body,
        [],
        [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "old response"}],
        }],
        conversation_id="conversation-text",
        parent_message_id="node-text",
        access_token="token-text",
        response_id="resp-text",
        usage={},
    )
    if not committed:
        raise AssertionError("initial session was not committed")
    return plan


class CodexResponseTextTests(unittest.TestCase):
    def test_normalizes_codex_text_wire_shape(self) -> None:
        body = _body(text={
            "verbosity": "HIGH",
            "format": {
                "type": "json_schema",
                "name": "answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        })

        controls = codex_response_text.normalized_text_controls(body)

        self.assertEqual(controls["verbosity"], "high")
        self.assertEqual(controls["format"]["type"], "json_schema")
        self.assertEqual(controls["format"]["name"], "answer")
        self.assertTrue(controls["format"]["strict"])

    def test_initial_plan_passes_text_controls_to_controller(self) -> None:
        body = _body(text={
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "strict": True,
                "schema": {"type": "object"},
                "name": "result",
            },
        })
        store = codex_conversation_session._SessionStore()
        full_messages = [{"role": "system", "content": "controller protocol"}]

        plan = store.prepare(
            body,
            [],
            full_messages,
            force_tool=False,
        )
        transcript = "\n".join(str(message["content"]) for message in plan.messages)
        replay_transcript = "\n".join(str(message["content"]) for message in full_messages)

        self.assertIn("CODEX_RESPONSE_TEXT_CONTROLS", transcript)
        self.assertIn('"verbosity":"low"', transcript)
        self.assertIn('"type":"json_schema"', transcript)
        self.assertIn("MUST be one JSON document", transcript)
        self.assertIn("strict=true", transcript)
        self.assertIn("CODEX_RESPONSE_TEXT_CONTROLS", replay_transcript)

    def test_request_signature_includes_text_controls(self) -> None:
        low = _body(text={"verbosity": "low"})
        high = _body(text={"verbosity": "high"})

        self.assertNotEqual(
            codex_conversation_session._request_signature(low),
            codex_conversation_session._request_signature(high),
        )

    def test_control_only_update_reuses_cursor_and_replaces_instructions(self) -> None:
        initial = _body(text={"verbosity": "low"})
        store = codex_conversation_session._SessionStore()
        _commit_initial(store, initial)
        updated = {
            **initial,
            "instructions": "replacement instructions",
            "text": {"verbosity": "high"},
            "previous_response_id": "resp-text",
            "client_metadata": {
                **initial["client_metadata"],
                "turn_id": "turn-2",
            },
        }

        with (
            mock.patch.object(
                codex_conversation_session.account_service,
                "resolve_access_token",
                return_value="token-text",
            ),
            mock.patch.object(
                codex_conversation_session.account_service,
                "get_account",
                return_value={"status": "normal"},
            ),
        ):
            plan = store.prepare(
                updated,
                [],
                [{"role": "system", "content": "full replay must not be used"}],
                force_tool=False,
            )

        transcript = "\n".join(str(message["content"]) for message in plan.messages)
        self.assertTrue(plan.continued)
        self.assertFalse(plan.replayed)
        self.assertEqual(plan.delta_input_items, 0)
        self.assertEqual(plan.conversation_id, "conversation-text")
        self.assertEqual(plan.parent_message_id, "node-text")
        self.assertIn("CODEX_TOP_LEVEL_INSTRUCTIONS_UPDATE", transcript)
        self.assertIn("replacement instructions", transcript)
        self.assertIn('"verbosity":"high"', transcript)
        self.assertNotIn("full replay must not be used", transcript)

    def test_removing_text_controls_sends_explicit_clear_update(self) -> None:
        initial = _body(text={"verbosity": "medium"})
        store = codex_conversation_session._SessionStore()
        _commit_initial(store, initial)
        updated = {
            key: value
            for key, value in initial.items()
            if key != "text"
        }
        updated["previous_response_id"] = "resp-text"
        updated["client_metadata"] = {
            **initial["client_metadata"],
            "turn_id": "turn-2",
        }

        with (
            mock.patch.object(
                codex_conversation_session.account_service,
                "resolve_access_token",
                return_value="token-text",
            ),
            mock.patch.object(
                codex_conversation_session.account_service,
                "get_account",
                return_value={"status": "normal"},
            ),
        ):
            plan = store.prepare(
                updated,
                [],
                [{"role": "system", "content": "full replay"}],
                force_tool=False,
            )

        transcript = "\n".join(str(message["content"]) for message in plan.messages)
        self.assertTrue(plan.continued)
        self.assertIn("CODEX_RESPONSE_TEXT_CONTROLS_UPDATE\nnull", transcript)

    def test_continuation_repeats_active_text_controls_for_self_contained_repair(self) -> None:
        initial = _body(text={"verbosity": "medium"})
        store = codex_conversation_session._SessionStore()
        _commit_initial(store, initial)
        updated = {
            **initial,
            "previous_response_id": "resp-text",
            "input": [
                *initial["input"],
                {"type": "message", "role": "user", "content": "continue"},
            ],
            "client_metadata": {
                **initial["client_metadata"],
                "turn_id": "turn-2",
            },
        }

        with (
            mock.patch.object(
                codex_conversation_session.account_service,
                "resolve_access_token",
                return_value="token-text",
            ),
            mock.patch.object(
                codex_conversation_session.account_service,
                "get_account",
                return_value={"status": "normal"},
            ),
        ):
            plan = store.prepare(
                updated,
                [],
                [{"role": "system", "content": "full replay"}],
                force_tool=False,
            )

        transcript = "\n".join(str(message["content"]) for message in plan.messages)
        self.assertTrue(plan.continued)
        self.assertEqual(plan.delta_input_items, 1)
        self.assertIn("CODEX_RESPONSE_TEXT_CONTROLS", transcript)
        self.assertIn('"verbosity":"medium"', transcript)

    def test_json_schema_validation_accepts_valid_json_and_rejects_drift(self) -> None:
        body = _body(text={
            "format": {
                "type": "json_schema",
                "strict": True,
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "minLength": 1},
                        "count": {"type": "integer", "minimum": 0},
                    },
                    "required": ["answer", "count"],
                    "additionalProperties": False,
                },
            },
        })

        self.assertEqual(
            codex_response_text.validate_response_text(body, '{"answer":"ok","count":1}'),
            (True, ""),
        )
        valid, error = codex_response_text.validate_response_text(
            body,
            '{"answer":"ok","count":1,"extra":true}',
        )
        self.assertFalse(valid)
        self.assertIn("additional properties", error)
        valid, error = codex_response_text.validate_response_text(body, "```json\n{}\n```")
        self.assertFalse(valid)
        self.assertIn("not valid JSON", error)

    def test_json_schema_validation_enforces_advanced_constraints(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "values": {
                    "type": "array",
                    "contains": {"const": "required"},
                },
                "step": {"type": "number", "multipleOf": 0.5},
                "primary": {"type": "string"},
                "secondary": {"type": "string"},
            },
            "dependentRequired": {"primary": ["secondary"]},
            "required": ["values", "step"],
        }

        valid, _ = codex_response_text.validate_json_value(
            {"values": ["other"], "step": 0.3, "primary": "x"},
            schema,
        )
        self.assertFalse(valid)
        self.assertFalse(codex_response_text.validate_json_value(
            {"values": ["required"], "step": 0.3},
            schema,
        )[0])
        self.assertFalse(codex_response_text.validate_json_value(
            {"values": ["required"], "step": 1.0, "primary": "x"},
            schema,
        )[0])
        self.assertTrue(codex_response_text.validate_json_value(
            {"values": ["required"], "step": 1.0, "primary": "x", "secondary": "y"},
            schema,
        )[0])

    def test_cache_key_separates_different_response_schemas(self) -> None:
        first = _body(text={
            "format": {
                "type": "json_schema",
                "schema": {"type": "object", "required": ["first"]},
            },
        })
        second = _body(text={
            "format": {
                "type": "json_schema",
                "schema": {"type": "object", "required": ["second"]},
            },
        })
        messages = [{"role": "user", "content": "return the result"}]

        self.assertNotEqual(
            cache_key(first, messages, stream=True),
            cache_key(second, messages, stream=True),
        )

    def test_direct_response_repairs_invalid_json_before_emitting_text(self) -> None:
        body = {
            **_body(text={
                "format": {
                    "type": "json_schema",
                    "strict": True,
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            }),
            "stream": True,
        }
        outputs = iter(["plain text", '{"answer":"ok"}'])

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch(
                "services.protocol.openai_v1_response.stream_text_deltas",
                side_effect=lambda _backend, _request: iter([next(outputs)]),
            ) as stream,
        ):
            events = list(openai_v1_response.handle(body))

        deltas = [
            event["delta"]
            for event in events
            if event["type"] == "response.output_text.delta"
        ]
        self.assertEqual(stream.call_count, 2)
        self.assertEqual(deltas, ['{"answer":"ok"}'])
        self.assertEqual(events[-1]["type"], "response.completed")

    def test_direct_response_fails_after_bounded_schema_repairs(self) -> None:
        body = {
            **_body(text={
                "format": {
                    "type": "json_schema",
                    "strict": True,
                    "schema": {"type": "object"},
                },
            }),
            "stream": True,
        }

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch(
                "services.protocol.openai_v1_response.stream_text_deltas",
                side_effect=lambda _backend, _request: iter(["not json"]),
            ) as stream,
        ):
            with self.assertRaises(openai_v1_response.ResponseTextValidationError):
                list(openai_v1_response.handle(body))

        self.assertEqual(stream.call_count, 3)

    def test_tool_controller_repairs_schema_invalid_final_action(self) -> None:
        exec_tool = {
            "type": "custom",
            "name": "exec",
            "description": "Run JavaScript.",
            "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"},
        }
        body = {
            **_body(text={
                "format": {
                    "type": "json_schema",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            }),
            "stream": True,
            "input": [
                {"type": "additional_tools", "role": "developer", "tools": [exec_tool]},
                {"type": "message", "role": "user", "content": "return structured output"},
            ],
        }
        outputs = iter([
            '{"action":"final","text":"plain","complete":true}',
            '{"action":"final","text":"{\\\"answer\\\":\\\"ok\\\"}","complete":true}',
        ])

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch(
                "services.protocol.openai_v1_response.stream_text_deltas",
                side_effect=lambda _backend, _request: iter([next(outputs)]),
            ) as stream,
        ):
            events = list(openai_v1_response.handle(body))

        self.assertEqual(stream.call_count, 2)
        final = events[-1]["response"]["output"][0]["content"][0]["text"]
        self.assertEqual(final, '{"answer":"ok"}')
        self.assertTrue(codex_tool_bridge.final_action_is_complete({
            "action": "final",
            "text": final,
            "complete": True,
        }))


if __name__ == "__main__":
    unittest.main()
