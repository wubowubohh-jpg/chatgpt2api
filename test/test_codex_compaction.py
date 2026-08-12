from __future__ import annotations

import json
import unittest

from services.protocol import openai_v1_response
from services.protocol import codex_tool_bridge


class CodexCompactionSourceTests(unittest.TestCase):
    def test_compaction_retains_encrypted_agent_message_as_opaque_context(self) -> None:
        body = {
            "model": "gpt-5.6-luna",
            "input": [
                {
                    "type": "agent_message",
                    "content": [
                        {"type": "input_text", "text": "Message Type: FINAL_ANSWER"},
                        {"type": "encrypted_content", "encrypted_content": "opaque"},
                    ],
                },
                {"type": "compaction_trigger"},
            ],
        }

        events = list(openai_v1_response.response_events(body))
        compacted = next(
            event["item"]["encrypted_content"]
            for event in events
            if event.get("type") == "response.output_item.done"
        )
        self.assertIn("opaque", compacted)

    def test_large_prefix_keeps_complete_recent_task_tool_pair_and_trigger(self) -> None:
        early_instructions = "EARLY_INSTRUCTIONS_ANCHOR\n" + ("old-system-context " * 900)
        old_user = "EARLY_USER_ANCHOR\n" + ("old-user-context " * 700)
        latest_user = (
            "LATEST_USER_TASK_BEGIN\n"
            + ("latest-task-detail " * 300)
            + "\nLATEST_USER_TASK_END"
        )
        latest_output = (
            "LATEST_TOOL_RESULT_BEGIN\n"
            + ("latest-tool-output " * 300)
            + "\nLATEST_TOOL_RESULT_END"
        )
        body = {
            "model": "gpt-5.6-luna",
            "instructions": early_instructions,
            "input": [
                {"type": "message", "role": "user", "content": old_user},
                {"type": "message", "role": "assistant", "content": "old reply"},
                {"type": "message", "role": "user", "content": latest_user},
                {
                    "type": "function_call",
                    "name": "shell_command",
                    "call_id": "call-latest",
                    "arguments": '{"command":"Get-Content important.txt"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-latest",
                    "output": latest_output,
                },
                {"type": "compaction_trigger"},
            ],
        }

        source = openai_v1_response._compaction_source(body)

        self.assertLessEqual(
            len(source.encode("utf-8")),
            openai_v1_response.COMPACTION_SOURCE_MAX_BYTES,
        )
        self.assertIn("EARLY_INSTRUCTIONS_ANCHOR", source)
        self.assertIn("compacted context", source)
        self.assertIn(latest_user, source)
        self.assertIn('Get-Content important.txt', source)
        self.assertIn(json.dumps(latest_output, ensure_ascii=False)[1:-1], source)
        self.assertIn("type=compaction_trigger", source)
        self.assertLess(source.index("LATEST_USER_TASK_BEGIN"), source.index("LATEST_TOOL_RESULT_BEGIN"))
        self.assertLess(source.index("LATEST_TOOL_RESULT_END"), source.index("type=compaction_trigger"))

    def test_oversized_required_records_keep_both_ends_within_utf8_budget(self) -> None:
        latest_user = "USER_BEGIN\n" + ("任务" * 10_000) + "\nUSER_END"
        latest_output = "TOOL_BEGIN\n" + ("结果" * 10_000) + "\nTOOL_END"
        body = {
            "input": [
                {"type": "message", "role": "user", "content": latest_user},
                {
                    "type": "function_call",
                    "name": "shell_command",
                    "call_id": "call-large",
                    "arguments": '{"command":"Get-ChildItem"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-large",
                    "output": latest_output,
                },
                {"type": "compaction_trigger"},
            ],
        }

        source = openai_v1_response._compaction_source(body)

        self.assertLessEqual(
            len(source.encode("utf-8")),
            openai_v1_response.COMPACTION_SOURCE_MAX_BYTES,
        )
        self.assertIn("USER_BEGIN", source)
        self.assertIn("USER_END", source)
        self.assertIn("TOOL_BEGIN", source)
        self.assertIn("TOOL_END", source)
        self.assertIn("record middle omitted for compaction budget", source)
        self.assertIn("type=compaction_trigger", source)


if __name__ == "__main__":
    unittest.main()
