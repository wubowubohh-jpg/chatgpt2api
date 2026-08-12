from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import replace
from typing import Any, Iterable, Iterator

from fastapi import HTTPException

from services.protocol.chat_completion_cache import cache_key, chat_completion_cache, normalize_text_messages
from services.protocol import codex_conversation_session, codex_response_text, codex_tool_bridge
from services.protocol.conversation import (
    ConversationRequest,
    ImageOutput,
    count_message_image_tokens,
    count_message_text_tokens,
    count_text_tokens,
    encode_images,
    normalize_messages,
    stream_image_outputs_with_pool,
    stream_text_deltas,
    text_backend,
)
from services.protocol.web_search_tool import (
    WEB_SEARCH_TOOL_TYPES,
    has_unsupported_tools,
    has_web_search_tool,
    normalized_sources,
    run_web_search,
    search_query_from_messages,
    text_with_url_citations,
)
from utils.helper import extract_image_from_message_content, extract_response_prompt, has_response_image_generation_tool
from utils.image_tokens import (
    count_image_content_tokens,
    count_image_output_items_tokens,
    image_usage,
    token_usage,
)
from utils.log import logger

TOOL_UNAVAILABLE_SYSTEM_MESSAGE = (
    "This compatibility backend cannot execute local tools, shell commands, non-search tools, "
    "or file operations. Do not claim to have run tools or inspected external resources. "
    "If a user asks you to use a tool, say that tool execution is unavailable through this backend."
)

CODEX_TOOL_CALL_RE = re.compile(
    r"(?is)<(?P<tag>codex_tool_call|custom_tool_call|tool_call)\b(?P<attrs>[^>]*)>(?P<body>.*?)</(?P=tag)>"
)
TOOL_NAME_RE = re.compile(r"(?is)\b(?:name|tool_name)\s*=\s*['\"]([^'\"]+)['\"]")
TOOL_INPUT_RE = re.compile(
    r"(?is)<(?:input|custom_tool_input|arguments)\b[^>]*>(.*?)</(?:input|custom_tool_input|arguments)>"
)

RESPONSE_CONTENT_PART_TYPES = {"text", "input_text", "output_text", "image_url", "input_image", "image"}

# ChatGPT Web rejects a large single /backend-api/conversation payload before
# the model context window is reached. Codex prompts are unusually wide because
# they carry the complete tool inventory. Large controller transcripts are
# therefore preloaded over several turns on the same upstream conversation.
CONTROLLER_REQUEST_TARGET_WIRE_BYTES = 56 * 1024
CONTROLLER_TRANSPORT_RECORD_BYTES = 6 * 1024
CONTROLLER_TRANSPORT_BASE_OVERHEAD_BYTES = 4 * 1024
CONTROLLER_TRANSPORT_MESSAGE_OVERHEAD_BYTES = 512
# The Web conversation endpoint accumulates every preload and continuation
# message. Keep a little headroom below the observed Free-account ceiling so a
# later turn can still be sent without relying on a 413 recovery.
CONTROLLER_SESSION_TARGET_WIRE_BYTES = 224 * 1024
CONTROLLER_HOSTED_SEARCH_RESULT_BYTES = 32 * 1024
CONTROLLER_MAX_HOSTED_SEARCHES = 3

HOSTED_WEB_SEARCH_TOOL = {
    "kind": "hosted_web_search",
    "name": "web_search",
    "namespace": None,
    "description": (
        "Search the current web when up-to-date external information is needed. "
        "This tool is executed by the compatibility server; never ask the Codex client to run it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A focused web search query.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


def normalize_thinking_effort(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"", "none", "auto"}:
        return ""
    if normalized in {"low", "medium", "high", "standard", "max"}:
        return normalized
    if normalized in {"xhigh", "extended"}:
        return "extended"
    return ""


def thinking_effort_from_body(body: dict[str, Any]) -> str:
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        return normalize_thinking_effort(reasoning.get("effort"))
    if "thinking_effort" in body:
        return normalize_thinking_effort(body.get("thinking_effort"))
    if "reasoning_effort" in body:
        return normalize_thinking_effort(body.get("reasoning_effort"))
    return ""


def is_text_response_request(body: dict[str, Any]) -> bool:
    return not has_response_image_generation_tool(body)


def has_unsupported_response_tools(body: dict[str, Any]) -> bool:
    return has_unsupported_tools(body, {"image_generation", *WEB_SEARCH_TOOL_TYPES})


def response_client_tools(body: dict[str, Any]) -> list[dict[str, Any]]:
    if str(body.get("tool_choice") or "").strip().lower() == "none":
        return []
    return codex_tool_bridge.response_client_tools(body)


def hosted_web_search_enabled(body: dict[str, Any]) -> bool:
    return (
        str(body.get("tool_choice") or "").strip().lower() != "none"
        and has_web_search_tool(body)
    )


def response_controller_tools(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every action available to the controller, including server-hosted tools."""
    tools = response_client_tools(body)
    if hosted_web_search_enabled(body) and not any(
        tool.get("name") == "web_search" and not tool.get("namespace")
        for tool in tools
    ):
        tools.append(dict(HOSTED_WEB_SEARCH_TOOL))
    return tools


def client_tool_prompt(tools: list[dict[str, Any]]) -> str:
    return codex_tool_bridge.controller_prompt(tools)


def parse_client_tool_calls(text: str, tools: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """Parse legacy XML markers through the same validator as JSON actions."""
    calls: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    for match in CODEX_TOOL_CALL_RE.finditer(str(text or "")):
        name_match = TOOL_NAME_RE.search(match.group("attrs") or "")
        name = name_match.group(1).strip() if name_match else ""
        if not name:
            continue
        body = match.group("body") or ""
        input_match = TOOL_INPUT_RE.search(body)
        raw_input = input_match.group(1) if input_match else body
        raw_input = raw_input.strip()
        if raw_input.startswith("<![CDATA[") and raw_input.endswith("]]>"):
            raw_input = raw_input[9:-3]
        raw_input = raw_input.strip()
        if not raw_input:
            continue
        legacy_payload = json.dumps(
            {"action": "tool", "name": name, "input": raw_input},
            ensure_ascii=False,
        )
        action = codex_tool_bridge.parse_controller_action(legacy_payload, tools)
        if action is None:
            continue
        calls.append(action)
        spans.append(match.span())
    visible = str(text or "")
    for start, end in reversed(spans):
        visible = visible[:start] + visible[end:]
    return calls, visible.strip()


def response_image_tool(body: dict[str, Any]) -> dict[str, object]:
    for tool in body.get("tools") or []:
        if isinstance(tool, dict) and tool.get("type") == "image_generation":
            return tool
    return {}


def extract_response_image(input_value: object) -> tuple[bytes, str] | None:
    if isinstance(input_value, dict):
        if str(input_value.get("type") or "").strip() == "input_image":
            images = extract_image_from_message_content([input_value])
            return images[0] if images else None
        images = extract_image_from_message_content(input_value.get("content"))
        return images[0] if images else None
    if not isinstance(input_value, list):
        return None
    for item in reversed(input_value):
        if isinstance(item, dict):
            if str(item.get("type") or "").strip() == "input_image":
                images = extract_image_from_message_content([item])
                if images:
                    return images[0]
            images = extract_image_from_message_content(item.get("content"))
            if images:
                return images[0]
    return None


def _input_image_parts(input_value: object) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    if isinstance(input_value, dict):
        content = input_value.get("content")
        if isinstance(content, list):
            parts.extend(item for item in content if isinstance(item, dict))
        return parts
    if not isinstance(input_value, list):
        return parts
    if all(isinstance(item, dict) and item.get("type") for item in input_value):
        return [item for item in input_value if isinstance(item, dict)]
    for item in input_value:
        if isinstance(item, dict):
            content = item.get("content")
            if isinstance(content, list):
                parts.extend(part for part in content if isinstance(part, dict))
    return parts


def _is_response_content_part(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    part_type = str(value.get("type") or "").strip()
    return part_type in RESPONSE_CONTENT_PART_TYPES or ("image_url" in value and part_type != "message")


def _message_content_from_response_item(item: dict[str, Any]) -> object:
    content = item.get("content")
    if isinstance(content, list):
        return [dict(part) if isinstance(part, dict) else part for part in content]
    if isinstance(content, str):
        return content
    return extract_response_prompt([item]) or content or ""


def _append_response_message(
    messages: list[dict[str, Any]],
    role: object,
    content: object,
    phase: object = None,
) -> None:
    normalized_phase = str(phase or "").strip()
    if isinstance(content, str):
        if content.strip():
            message = {"role": str(role or "user"), "content": content.strip()}
            if normalized_phase:
                message["phase"] = normalized_phase
            messages.append(message)
        return
    if isinstance(content, list) and content:
        message = {"role": str(role or "user"), "content": content}
        if normalized_phase:
            message["phase"] = normalized_phase
        messages.append(message)


def messages_from_input(input_value: object, instructions: object = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    tool_calls_by_id: dict[str, dict[str, Any]] = {}
    system_text = str(instructions or "").strip()
    if system_text:
        messages.append({"role": "system", "content": system_text})
    if isinstance(input_value, str):
        if input_value.strip():
            messages.append({"role": "user", "content": input_value.strip()})
        return messages
    if isinstance(input_value, dict):
        if _is_response_content_part(input_value):
            _append_response_message(messages, "user", [dict(input_value)])
            return messages
        history_message = codex_tool_bridge.response_item_history_message(input_value)
        if history_message:
            messages.append(history_message)
            return messages
        _append_response_message(
            messages,
            input_value.get("role") or "user",
            _message_content_from_response_item(input_value),
            input_value.get("phase"),
        )
        return messages
    if isinstance(input_value, list):
        if all(_is_response_content_part(item) for item in input_value):
            _append_response_message(messages, "user", [dict(item) for item in input_value if isinstance(item, dict)])
            return messages
        pending_parts: list[dict[str, Any]] = []
        for item in input_value:
            if _is_response_content_part(item):
                pending_parts.append(dict(item))
                continue
            if pending_parts:
                _append_response_message(messages, "user", pending_parts)
                pending_parts = []
            if not isinstance(item, dict):
                continue
            tool_message = codex_tool_bridge.tool_history_message(item, tool_calls_by_id)
            if tool_message:
                messages.append(tool_message)
                continue
            history_message = codex_tool_bridge.response_item_history_message(item)
            if history_message:
                messages.append(history_message)
                continue
            _append_response_message(
                messages,
                item.get("role") or "user",
                _message_content_from_response_item(item),
                item.get("phase"),
            )
        if pending_parts:
            _append_response_message(messages, "user", pending_parts)
    return messages


def text_output_item(
    text: str,
    item_id: str | None = None,
    status: str = "completed",
    annotations: list[dict[str, Any]] | None = None,
    phase: str | None = None,
) -> dict[str, Any]:
    # Codex uses the phase when forking full history: only final_answer
    # assistant messages are retained in a child rollout.
    resolved_phase = phase or ("final_answer" if status == "completed" else "commentary")
    return {
        "id": item_id or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "status": status,
        "role": "assistant",
        "phase": resolved_phase,
        "content": [{"type": "output_text", "text": text, "annotations": annotations or []}],
    }


def web_search_call_item(
    query: str,
    item_id: str | None = None,
    status: str = "completed",
    sources: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    action: dict[str, Any] = {
        "type": "search",
        "query": query,
        "queries": [query],
    }
    if sources:
        action["sources"] = [
            {"type": "url", "url": source["url"]}
            for source in sources
            if source.get("url")
        ]
    return {
        "id": item_id or f"ws_{uuid.uuid4().hex}",
        "type": "web_search_call",
        "status": status,
        "action": action,
    }


def image_output_items(prompt: str, data: list[dict[str, Any]], item_id: str | None = None) -> list[dict[str, Any]]:
    output = []
    for item in data:
        b64_json = str(item.get("b64_json") or "").strip()
        if b64_json:
            output.append({
                "id": item_id or f"ig_{len(output) + 1}",
                "type": "image_generation_call",
                "status": "completed",
                "result": b64_json,
                "revised_prompt": str(item.get("revised_prompt") or prompt).strip() or prompt,
            })
    return output


def response_created(
    response_id: str,
    model: str,
    created: int,
    *,
    parallel_tool_calls: bool = False,
) -> dict[str, Any]:
    return {
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "status": "in_progress",
            "error": None,
            "incomplete_details": None,
            "model": model,
            "output": [],
            "parallel_tool_calls": parallel_tool_calls,
        },
    }


def response_completed(
    response_id: str,
    model: str,
    created: int,
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None = None,
    end_turn: bool | None = None,
    parallel_tool_calls: bool = False,
) -> dict[str, Any]:
    response = {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "model": model,
            "output": output,
            "parallel_tool_calls": parallel_tool_calls,
        },
    }
    if usage:
        response["response"]["usage"] = usage
    if end_turn is not None:
        response["response"]["end_turn"] = end_turn
    return response


def _with_response_id(
    events: Iterable[dict[str, Any]],
    response_id: object,
) -> Iterator[dict[str, Any]]:
    """Rewrite cached response envelopes to the identity of the current HTTP request."""
    current_id = str(response_id or "").strip()
    item_ids: dict[str, str] = {}

    def item_id(value: object) -> object:
        original = str(value or "").strip()
        if not current_id or not original:
            return value
        prefix = original.partition("_")[0] or "item"
        return item_ids.setdefault(
            original,
            f"{prefix}_{uuid.uuid5(uuid.NAMESPACE_URL, current_id + chr(0) + original).hex}",
        )

    def output_item(value: object) -> object:
        if not isinstance(value, dict):
            return value
        rewritten_item = dict(value)
        if rewritten_item.get("id"):
            rewritten_item["id"] = item_id(rewritten_item["id"])
        return rewritten_item

    for event in events:
        if not current_id:
            yield event
            continue
        rewritten = dict(event)
        if rewritten.get("item_id"):
            rewritten["item_id"] = item_id(rewritten["item_id"])
        if isinstance(rewritten.get("item"), dict):
            rewritten["item"] = output_item(rewritten["item"])
        if isinstance(event.get("response"), dict):
            response = dict(event["response"])
            response["id"] = current_id
            if isinstance(response.get("output"), list):
                response["output"] = [output_item(item) for item in response["output"]]
            rewritten["response"] = response
        yield rewritten


def text_response_parts(body: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    model = str(body.get("model") or "auto").strip() or "auto"
    client_tools = response_client_tools(body)
    if client_tools:
        force_tool = (
            codex_tool_bridge.requires_local_tool(body, client_tools)
            and not codex_tool_bridge.current_turn_has_tool_output(body.get("input"))
        )
        return model, codex_tool_bridge.controller_messages(body, client_tools, force_tool=force_tool)
    messages = normalize_text_messages(
        normalize_messages(messages_from_input(body.get("input"), body.get("instructions")))
    )
    text_controls = codex_response_text.controller_text_control_messages(body)
    if text_controls:
        messages = [*text_controls, *messages]
    if has_unsupported_response_tools(body):
        messages.insert(0, {"role": "system", "content": TOOL_UNAVAILABLE_SYSTEM_MESSAGE})
    return model, messages


def _text_output_events(text: str, output_index: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    item_id = f"msg_{uuid.uuid4().hex}"
    item = text_output_item(text, item_id, "completed")
    events: list[dict[str, Any]] = [
        {"type": "response.output_item.added", "output_index": output_index, "item": text_output_item("", item_id, "in_progress")},
    ]
    if text:
        events.append({"type": "response.output_text.delta", "item_id": item_id, "output_index": output_index, "content_index": 0, "delta": text})
    events.extend([
        {"type": "response.output_text.done", "item_id": item_id, "output_index": output_index, "content_index": 0, "text": text},
        {"type": "response.output_item.done", "output_index": output_index, "item": item},
    ])
    return events, item


def _client_tool_events(calls: list[dict[str, Any]], output_index: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for index, call in enumerate(calls, start=output_index):
        call_id = f"call_{uuid.uuid4().hex}"
        kind = call["kind"]
        custom = kind == "custom"
        tool_search = kind == "tool_search"
        item_id = f"{'tsc' if tool_search else 'ctc' if custom else 'fc'}_{uuid.uuid4().hex}"
        item_type = "tool_search_call" if tool_search else "custom_tool_call" if custom else "function_call"
        item = {
            "id": item_id,
            "type": item_type,
            "status": "in_progress",
            "call_id": call_id,
        }
        namespace = codex_tool_bridge.normalize_namespace(call.get("namespace"))
        if tool_search:
            item["execution"] = "client"
            item["arguments"] = json.loads(call["input"])
        else:
            item["name"] = call["name"]
            item["input" if custom else "arguments"] = ""
            if namespace:
                item["namespace"] = namespace
            if not custom and isinstance(call.get("encrypted_function_args"), list):
                item["encrypted_function_args"] = call["encrypted_function_args"]
        events.append({"type": "response.output_item.added", "output_index": index, "item": item})
        if not tool_search:
            event_prefix = "response.custom_tool_call_input" if custom else "response.function_call_arguments"
            events.append({
                "type": f"{event_prefix}.delta",
                "item_id": item_id,
                "call_id": call_id,
                "output_index": index,
                "delta": call["input"],
            })
            events.append({
                "type": f"{event_prefix}.done",
                "item_id": item_id,
                "call_id": call_id,
                "output_index": index,
                "input" if custom else "arguments": call["input"],
            })
        completed_item = dict(item)
        completed_item["status"] = "completed"
        if not tool_search:
            completed_item["input" if custom else "arguments"] = call["input"]
        events.append({"type": "response.output_item.done", "output_index": index, "item": completed_item})
        items.append(completed_item)
    return events, items


def _is_hosted_web_search_action(action: dict[str, Any] | None) -> bool:
    return bool(
        action
        and action.get("action") == "tool"
        and action.get("kind") == "hosted_web_search"
        and action.get("name") == "web_search"
    )


def _hosted_web_search_query(action: dict[str, Any]) -> str:
    try:
        arguments = json.loads(str(action.get("input") or "{}"))
    except json.JSONDecodeError:
        return ""
    return str(arguments.get("query") or "").strip() if isinstance(arguments, dict) else ""


def _hosted_web_search_events(
    query: str,
    result: dict[str, Any],
    output_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    item_id = f"ws_{uuid.uuid4().hex}"
    in_progress = web_search_call_item(query, item_id, "in_progress")
    completed = web_search_call_item(query, item_id, "completed", normalized_sources(result))
    return [
        {"type": "response.output_item.added", "output_index": output_index, "item": in_progress},
        {"type": "response.web_search_call.in_progress", "output_index": output_index, "item_id": item_id},
        {"type": "response.web_search_call.searching", "output_index": output_index, "item_id": item_id},
        {"type": "response.web_search_call.completed", "output_index": output_index, "item_id": item_id},
        {"type": "response.output_item.done", "output_index": output_index, "item": completed},
    ], completed


def _controller_hosted_search_messages(
    body: dict[str, Any],
    tools: list[dict[str, Any]],
    query: str,
    result: dict[str, Any],
    *,
    include_bootstrap: bool,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if include_bootstrap:
        messages.append({"role": "system", "content": codex_tool_bridge.controller_prompt(tools)})
        messages.extend(codex_tool_bridge.controller_tool_messages(tools))
    result_record = json.dumps(
        {
            "query": query,
            "answer": str(result.get("answer") or ""),
            "sources": normalized_sources(result),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    result_record = codex_tool_bridge._truncate_utf8(
        result_record,
        CONTROLLER_HOSTED_SEARCH_RESULT_BYTES,
        "\n[web search result truncated for controller transport]",
    )
    messages.append({
        "role": "user",
        "content": (
            "HOSTED_WEB_SEARCH_RESULT\n"
            "The compatibility server executed this hosted tool. Use the result below, then return "
            "exactly one next controller JSON action. Do not ask the Codex client to execute web_search.\n"
            + result_record
        ),
    })
    canonical_input = body.get("input")
    messages.extend(codex_tool_bridge.controller_task_contract_messages(canonical_input))
    messages.extend(codex_tool_bridge.controller_task_anchor_messages(canonical_input))
    messages.extend(codex_tool_bridge.controller_parallel_tool_state_messages(
        bool(body.get("parallel_tool_calls"))
    ))
    messages.extend(codex_tool_bridge.controller_turn_state_messages(False))
    return messages


def _log_controller_request_shape(
    messages: list[dict[str, Any]],
    *,
    tool_count: int,
    attempt: str,
) -> None:
    contents = [message.get("content") or "" for message in messages]
    encoded_sizes = [
        len(content.encode("utf-8"))
        if isinstance(content, str)
        else len(json.dumps(
            content,
            ensure_ascii=True,
            separators=(",", ":"),
            default=_controller_json_default,
        ).encode("utf-8"))
        for content in contents
    ]
    logger.debug({
        "event": "codex_controller_request_shape",
        "attempt": attempt,
        "message_count": len(messages),
        "tool_count": tool_count,
        "content_bytes": sum(encoded_sizes),
        "max_message_bytes": max(encoded_sizes, default=0),
        "max_message_chars": max(encoded_sizes, default=0),
    })


def _controller_json_default(value: object) -> object:
    if isinstance(value, (bytes, bytearray)):
        return {"binary_bytes": len(value)}
    raise TypeError(f"unsupported controller message value: {type(value).__name__}")


def _controller_messages_wire_estimate(messages: list[dict[str, Any]]) -> int:
    """Conservatively estimate the JSON body after ChatGPT Web conversion."""
    encoded = json.dumps(
        messages,
        ensure_ascii=True,
        separators=(",", ":"),
        default=_controller_json_default,
    ).encode("utf-8")
    return (
        len(encoded)
        + CONTROLLER_TRANSPORT_BASE_OVERHEAD_BYTES
        + len(messages) * CONTROLLER_TRANSPORT_MESSAGE_OVERHEAD_BYTES
    )


def _split_controller_transport_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    if _controller_messages_wire_estimate([message]) <= CONTROLLER_REQUEST_TARGET_WIRE_BYTES:
        return [dict(message)]
    content_value = message.get("content") or ""
    if not isinstance(content_value, str):
        # Binary image inputs are uploaded before the Web conversation request,
        # so splitting or stringifying them would destroy the media attachment.
        return [dict(message)]
    content = content_value
    header, separator, value = content.partition("\n")
    if not separator:
        header = "CONTROLLER_TRANSPORT_RECORD"
        value = content
    chunks = codex_tool_bridge._utf8_chunks(value, CONTROLLER_TRANSPORT_RECORD_BYTES)
    total = len(chunks)
    return [
        {
            **message,
            "content": f"{header} transport_segment={index}/{total}\n{chunk}",
        }
        for index, chunk in enumerate(chunks, start=1)
    ]


def _controller_message_batches(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    expanded: list[dict[str, Any]] = []
    for message in messages:
        expanded.extend(_split_controller_transport_message(message))
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in expanded:
        candidate = [*current, message]
        if current and _controller_messages_wire_estimate(candidate) > CONTROLLER_REQUEST_TARGET_WIRE_BYTES:
            batches.append(current)
            current = [message]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def _controller_cumulative_wire_estimate(messages: list[dict[str, Any]]) -> int:
    """Estimate all requests needed to load messages into one Web cursor."""
    batches = _controller_message_batches(messages)
    total = 0
    for index, batch in enumerate(batches, start=1):
        total += _controller_messages_wire_estimate([
            *batch,
            _controller_transport_marker(index, len(batches), final=index == len(batches)),
        ])
    return total


def _controller_transport_marker(index: int, total: int, *, final: bool) -> dict[str, str]:
    if final:
        content = (
            f"CONTROLLER_CONTEXT_PRELOAD_COMPLETE batch={index}/{total}\n"
            "All preceding records are now loaded in this same upstream conversation. "
            "Earlier assistant replies between transport batches were acknowledgements only: "
            "they were never sent to Codex, never executed a tool, and never completed the task. "
            "Now obey the latest CONTROLLER_TASK_CONTRACT and CONTROLLER_TURN_STATE and return "
            "exactly one valid controller JSON action."
        )
    else:
        content = (
            f"CONTROLLER_CONTEXT_PRELOAD batch={index}/{total}\n"
            "This is transport-only context for a later Codex action. Do not choose a tool, answer "
            "the user, or mark the task complete yet. Preserve every record exactly in conversation "
            f"memory and reply only with {{\"context_preload_ack\":{index}}}."
        )
    return {"role": "system", "content": content}


def _controller_compacted_messages(
    body: dict[str, Any],
    tools: list[dict[str, Any]],
    *,
    force_tool: bool,
) -> list[dict[str, Any]]:
    """Build one bounded replacement conversation when the Web cursor is full.

    This is deliberately an explicit context checkpoint. It does not perform
    duplicate-system/environment cleanup; the source formatter preserves the
    active task, required tool pair, opaque Codex state, and an early context
    excerpt within the bounded replacement record.
    """
    source = _compaction_source(body)
    if not source:
        source = "No prior Codex context was supplied."
    messages: list[dict[str, Any]] = [{
        "role": "system",
        "content": codex_tool_bridge.controller_prompt(tools),
    }]
    messages.extend(codex_tool_bridge.controller_tool_messages(tools))
    messages.append({
        "role": "user",
        "content": (
            "CONTROLLER_CONTEXT_COMPACTION\n"
            "A new upstream Web conversation replaces the previous cursor. The record below is "
            "the bounded context checkpoint. Preserve its facts and continue the active task; "
            "do not claim completion unless the task contract is satisfied.\n"
            + source
        ),
    })
    messages.extend(codex_tool_bridge.controller_task_contract_messages(body.get("input")))
    messages.extend(codex_tool_bridge.controller_task_anchor_messages(body.get("input")))
    messages.extend(codex_tool_bridge.controller_parallel_tool_state_messages(
        bool(body.get("parallel_tool_calls")),
    ))
    messages.extend(codex_tool_bridge.controller_turn_state_messages(force_tool))
    return messages


def _controller_preload_ack(text: str, expected_index: int) -> bool:
    try:
        value = json.loads(str(text or "").strip())
    except (TypeError, ValueError):
        return False
    return (
        isinstance(value, dict)
        and set(value) == {"context_preload_ack"}
        and value.get("context_preload_ack") == expected_index
    )


def _is_controller_payload_too_large_error(error: Exception) -> bool:
    return int(getattr(error, "status_code", 0) or 0) == 413


def _stream_controller_request(backend, request: ConversationRequest, *, attempt: str) -> str:
    """Run one logical controller request, preloading oversized context losslessly."""
    original_messages = list(request.messages or [])
    batches = _controller_message_batches(original_messages)
    if len(batches) <= 1:
        request.controller_wire_bytes += _controller_messages_wire_estimate(original_messages)
        return "".join(stream_text_deltas(backend, request))

    logger.info({
        "event": "codex_controller_context_preload",
        "attempt": attempt,
        "batch_count": len(batches),
        "original_wire_bytes_estimate": _controller_messages_wire_estimate(original_messages),
    })
    final_text = ""
    sent_wire_bytes = 0
    try:
        for index, batch in enumerate(batches, start=1):
            final = index == len(batches)
            request.messages = [*batch, _controller_transport_marker(index, len(batches), final=final)]
            sent_wire_bytes += _controller_messages_wire_estimate(request.messages)
            _log_controller_request_shape(
                request.messages,
                tool_count=0,
                attempt=f"{attempt}_preload_{index}_of_{len(batches)}",
            )
            output = "".join(stream_text_deltas(backend, request))
            if not final:
                if not _controller_preload_ack(output, index):
                    raise RuntimeError(
                        "controller context preload returned a non-ack action; refusing to execute it"
                    )
            else:
                final_text = output
    finally:
        request.messages = original_messages
        request.controller_wire_bytes += sent_wire_bytes
    return final_text


def _split_plain_transport_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Split one ordinary text message without rewriting its content.

    The controller splitter adds a record header to each segment so the tool
    controller can identify transport records.  Ordinary Responses requests
    have no controller grammar, so that header would become user-visible
    prompt text and, more importantly, duplicate an ``<environment_context>``
    line for every segment.  Keep the original bytes in-order instead.
    """
    if _controller_messages_wire_estimate([message]) <= CONTROLLER_REQUEST_TARGET_WIRE_BYTES:
        return [dict(message)]
    content = message.get("content")
    if not isinstance(content, str):
        # Images and other structured content cannot be divided safely.  The
        # caller will surface a deterministic context error for this case
        # instead of silently changing the request.
        return [dict(message)]
    chunks = codex_tool_bridge._utf8_chunks(content, CONTROLLER_TRANSPORT_RECORD_BYTES)
    return [{**message, "content": chunk} for chunk in chunks]


def _plain_message_batches(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Build bounded batches for a no-tools Responses request.

    Message order and each text message's content are retained verbatim.  The
    only added records are transport markers, which are hidden from the
    client and prevent an intermediate model reply from ending the request.
    """
    expanded: list[dict[str, Any]] = []
    for message in messages:
        expanded.extend(_split_plain_transport_message(message))
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in expanded:
        candidate = [*current, message]
        if current and _controller_messages_wire_estimate(candidate) > CONTROLLER_REQUEST_TARGET_WIRE_BYTES:
            batches.append(current)
            current = [message]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def _plain_transport_marker(index: int, total: int, *, final: bool) -> dict[str, str]:
    if final:
        content = (
            f"TEXT_CONTEXT_PRELOAD_COMPLETE batch={index}/{total}\n"
            "All preceding records were transport-only context. Preserve their "
            "order and meaning, then answer the latest user request."
        )
    else:
        content = (
            f"TEXT_CONTEXT_PRELOAD batch={index}/{total}\n"
            "This is transport-only context for a later request. Do not answer "
            "yet; preserve every preceding record and wait for the final batch."
        )
    return {"role": "system", "content": content}


def _stream_plain_text_request(backend, request: ConversationRequest, *, attempt: str) -> Iterator[str]:
    """Stream an ordinary no-tools request, preloading oversized context.

    This path deliberately does not compact, deduplicate, or remove repeated
    system/environment records.  Small requests retain the original one-call
    streaming behavior.  For larger requests, intermediate upstream replies
    are consumed and discarded; only the final batch is exposed as output.
    """
    original_messages = list(request.messages or [])
    batches = _plain_message_batches(original_messages)
    if len(batches) <= 1:
        yield from stream_text_deltas(backend, request)
        return

    logger.info({
        "event": "codex_plain_context_preload",
        "attempt": attempt,
        "batch_count": len(batches),
        "original_wire_bytes_estimate": _controller_messages_wire_estimate(original_messages),
    })
    try:
        for index, batch in enumerate(batches, start=1):
            final = index == len(batches)
            request.messages = [*batch, _plain_transport_marker(index, len(batches), final=final)]
            _log_controller_request_shape(
                request.messages,
                tool_count=0,
                attempt=f"{attempt}_preload_{index}_of_{len(batches)}",
            )
            if final:
                yield from stream_text_deltas(backend, request)
            else:
                # Do not expose or cache an intermediate answer.  No client
                # tools exist in this branch, so an unexpected model reply is
                # harmless; the next batch continues from the same cursor.
                for _delta in stream_text_deltas(backend, request):
                    pass
    finally:
        request.messages = original_messages


def _log_controller_output_shape(text: str, *, attempt: str) -> None:
    """Expose a bounded controller preview only when explicitly debugging upstream output."""
    if os.getenv("CHATGPT2API_DEBUG_CONTROLLER_OUTPUT") != "1":
        return
    value = str(text or "")
    # The explicit environment flag is already the opt-in boundary. Emit at
    # info so deployments that keep the default log-level filter can actually
    # capture this bounded diagnostic when troubleshooting controller output.
    logger.info({
        "event": "codex_controller_output_shape",
        "attempt": attempt,
        "text_bytes": len(value.encode("utf-8")),
        "preview": value[:800],
    })


def _log_controller_parse_shape(
    text: str,
    action: dict[str, Any] | None,
    tools: list[dict[str, Any]],
    *,
    force_tool: bool,
    attempt: str,
) -> None:
    if os.getenv("CHATGPT2API_DEBUG_CONTROLLER_OUTPUT") != "1":
        return
    logger.info({
        "event": "codex_controller_parse_shape",
        "attempt": attempt,
        "force_tool": force_tool,
        "parsed": bool(action),
        "parsed_name": str((action or {}).get("name") or ""),
        "parsed_kind": str((action or {}).get("kind") or ""),
        "parsed_complete": bool((action or {}).get("complete")) if (action or {}).get("action") == "final" else None,
        "tool_names": [
            f"{tool.get('namespace') or ''}.{tool.get('name') or ''}:{tool.get('kind') or ''}"
            for tool in tools
        ],
        "text_bytes": len(str(text or '').encode('utf-8')),
    })


def _plain_controller_final(text: str) -> dict[str, Any] | None:
    source = str(text or "").strip()
    if (
        not source
        or codex_tool_bridge.is_access_refusal(source)
        or source.startswith(("{", "[", "<codex_tool_call", "<custom_tool_call", "<tool_call"))
    ):
        return None
    return {"action": "final", "text": source, "complete": False}


class ResponseTextValidationError(RuntimeError):
    def to_openai_error(self) -> dict[str, Any]:
        return {
            "error": {
                "message": str(self),
                "type": "invalid_response_error",
                "code": "invalid_prompt",
            },
        }


def _final_text_validation_error(
    body: dict[str, Any],
    action: dict[str, Any] | None,
) -> str:
    if not action or action.get("action") != "final":
        return ""
    valid, error = codex_response_text.validate_response_text(
        body,
        str(action.get("text") or ""),
    )
    return "" if valid else error


def _is_stale_controller_cursor_error(error: Exception) -> bool:
    status_code = int(getattr(error, "status_code", 0) or 0)
    if status_code not in {400, 404, 409, 422}:
        return False
    text = str(error).lower()
    return any(marker in text for marker in ("conversation", "parent", "cursor", "not found", "invalid"))


def _replay_controller_output(
    output: list[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    for index, original in enumerate(output):
        item = dict(original)
        item["status"] = "in_progress"
        yield {"type": "response.output_item.added", "output_index": index, "item": item}
        if item.get("type") == "message":
            text = "".join(
                str(part.get("text") or "")
                for part in item.get("content") or []
                if isinstance(part, dict) and part.get("type") in {"output_text", "text"}
            )
            if text:
                yield {
                    "type": "response.output_text.delta",
                    "item_id": item.get("id"),
                    "output_index": index,
                    "content_index": 0,
                    "delta": text,
                }
                yield {
                    "type": "response.output_text.done",
                    "item_id": item.get("id"),
                    "output_index": index,
                    "content_index": 0,
                    "text": text,
                }
        item["status"] = "completed"
        yield {"type": "response.output_item.done", "output_index": index, "item": item}


def stream_text_response(backend, body: dict[str, Any], messages: list[dict[str, Any]] | None = None) -> Iterator[dict[str, Any]]:
    model = str(body.get("model") or "auto").strip() or "auto"
    messages = messages if messages is not None else messages_from_input(body.get("input"), body.get("instructions"))
    thinking_effort = thinking_effort_from_body(body)
    response_id = str(body.get("_response_id") or f"resp_{uuid.uuid4().hex}")
    item_id = f"msg_{uuid.uuid4().hex}"
    created = int(time.time())
    full_text = ""
    parallel_tool_calls = bool(body.get("parallel_tool_calls"))
    yield response_created(
        response_id,
        model,
        created,
        parallel_tool_calls=parallel_tool_calls,
    )
    if backend is None:
        # CallLog reads the first item before constructing StreamingResponse so
        # it can expose Codex headers. Delay account selection/token refresh
        # until after response.created; the SSE heartbeat then covers any slow
        # backend initialization instead of leaving the client with no bytes.
        backend = text_backend(model)
    client_tools = response_client_tools(body)
    if client_tools:
        controller_tools = response_controller_tools(body)
        turn_has_tool_output = codex_tool_bridge.current_turn_has_tool_output(body.get("input"))
        completion_required = turn_has_tool_output
        force_tool = (
            codex_tool_bridge.requires_local_tool(body, client_tools)
            and not turn_has_tool_output
        )
        full_controller_messages = codex_tool_bridge.controller_messages(
            body, controller_tools, force_tool=force_tool,
        )
        pending_events: list[dict[str, Any]] = []
        output: list[dict[str, Any]] = []
        usage: dict[str, Any] = {}
        with codex_conversation_session.controller_session_lock(body):
            plan = codex_conversation_session.prepare_controller_request(
                body,
                controller_tools,
                full_controller_messages,
                force_tool=force_tool,
            )
            logger.debug({
                "event": "codex_controller_session_plan",
                "continued": plan.continued,
                "replayed": plan.replayed,
                "delta_input_items": plan.delta_input_items,
                "conversation_reused": bool(plan.conversation_id),
                "canonical_input_items": len(plan.canonical_input_items),
            })
            if plan.replayed:
                pending_events.extend(_replay_controller_output(plan.output_items))
                output = plan.output_items
                usage = plan.usage
            else:
                controller_messages = plan.messages
                _log_controller_request_shape(
                    controller_messages,
                    tool_count=len(controller_tools),
                    attempt="continuation" if plan.continued else "initial",
                )
                previous_wire_bytes = plan.upstream_wire_bytes
                request = ConversationRequest(
                    model=model,
                    messages=controller_messages,
                    thinking_effort=thinking_effort,
                    conversation_id=plan.conversation_id,
                    parent_message_id=plan.parent_message_id,
                    access_token=plan.access_token,
                    controller_wire_bytes=plan.upstream_wire_bytes,
                )
                if (
                    plan.continued
                    and plan.upstream_wire_bytes
                    and plan.upstream_wire_bytes + _controller_messages_wire_estimate(controller_messages)
                    > CONTROLLER_SESSION_TARGET_WIRE_BYTES
                ):
                    # Replace the saturated Web cursor with a bounded Codex
                    # checkpoint before sampling. This avoids a deterministic
                    # 413 on the next append and also handles local compaction
                    # when the client did not send a typed compaction item.
                    codex_conversation_session.invalidate_controller_session(plan)
                    compacted_messages = _controller_compacted_messages(
                        body,
                        controller_tools,
                        force_tool=force_tool,
                    )
                    plan = replace(
                        plan,
                        messages=compacted_messages,
                        conversation_id="",
                        parent_message_id="",
                        continued=False,
                        upstream_wire_bytes=0,
                    )
                    controller_messages = compacted_messages
                    request = ConversationRequest(
                        model=model,
                        messages=controller_messages,
                        thinking_effort=thinking_effort,
                        access_token=plan.access_token,
                        controller_wire_bytes=0,
                    )
                    logger.info({
                        "event": "codex_controller_context_checkpoint",
                        "reason": "upstream_session_budget",
                        "previous_wire_bytes": previous_wire_bytes,
                    })
                request_parent_before = request.parent_message_id
                try:
                    full_text = _stream_controller_request(
                        backend,
                        request,
                        attempt="continuation" if plan.continued else "initial",
                    )
                    _log_controller_output_shape(full_text, attempt="initial" if not plan.continued else "continuation")
                except Exception as exc:
                    reset_continuation = plan.continued and (
                        _is_stale_controller_cursor_error(exc)
                        or _is_controller_payload_too_large_error(exc)
                    )
                    if not reset_continuation:
                        raise
                    codex_conversation_session.invalidate_controller_session(plan)
                    plan = codex_conversation_session.ContinuationPlan(
                        key=plan.key,
                        messages=full_controller_messages,
                        access_token=plan.access_token,
                    )
                    controller_messages = plan.messages
                    _log_controller_request_shape(
                        controller_messages,
                        tool_count=len(controller_tools),
                        attempt="continuation_reset",
                    )
                    request = ConversationRequest(
                        model=model,
                        messages=controller_messages,
                        thinking_effort=thinking_effort,
                        access_token=plan.access_token,
                    )
                    request_parent_before = ""
                    full_text = ""
                    full_text = _stream_controller_request(
                        backend,
                        request,
                        attempt="continuation_reset",
                    )
                    _log_controller_output_shape(full_text, attempt="continuation_reset")
                action = codex_tool_bridge.parse_controller_action(
                    full_text,
                    controller_tools,
                    allow_parallel=parallel_tool_calls,
                )
                _log_controller_parse_shape(
                    full_text,
                    action,
                    controller_tools,
                    force_tool=force_tool,
                    attempt="continuation" if plan.continued else "initial",
                )
                legacy_calls, _legacy_visible_text = parse_client_tool_calls(full_text, controller_tools)
                if action is None and legacy_calls:
                    action = (
                        {"action": "tools", "calls": legacy_calls}
                        if parallel_tool_calls and len(legacy_calls) > 1
                        else {"action": "tool", **legacy_calls[0]}
                    )
                if action is None and turn_has_tool_output:
                    action = _plain_controller_final(full_text)
                validation_input = getattr(plan, "canonical_input_items", None) or body.get("input")
                prior_output_items = getattr(plan, "prior_output_items", None)
                duplicate_action = codex_tool_bridge.action_repeats_completed_tool(
                    action,
                    validation_input,
                    seed_items=prior_output_items,
                )
                structured_text_error = _final_text_validation_error(body, action)
                rejected_action = (
                    action is not None
                    and (
                        (
                            action.get("action") == "final"
                            and (
                                codex_tool_bridge.is_access_refusal(str(action.get("text") or ""))
                                or codex_tool_bridge.is_task_evasion(str(action.get("text") or ""))
                            )
                        )
                        or (
                            completion_required
                            and action.get("action") == "final"
                            and not codex_tool_bridge.final_action_has_sufficient_evidence(
                                action,
                                validation_input,
                                prior_output_items,
                            )
                        )
                        or (force_tool and not codex_tool_bridge.is_progress_tool_action(action))
                        or duplicate_action
                        or bool(structured_text_error)
                    )
                )
                invalid_output = action is None
                if rejected_action or invalid_output:
                    repaired_text = ""
                    repair_context = full_text
                    if action is None and codex_tool_bridge.has_invalid_v1_agent_id_reference(full_text):
                        repair_context = (
                            "INVALID_V1_AGENT_ID_REFERENCE\n"
                            "Codex V1 spawn_agent returns `{agent_id, nickname}`, never `id`. "
                            "Use `spawned.agent_id` only in the same exec cell after declaring "
                            "`spawned`, or use the literal agent_id UUID from the prior tool output "
                            "in a new exec cell. Emit `text(spawned)` before dependent calls so the "
                            "UUID survives a failed wait.\n"
                            + full_text
                        )
                    elif action is None and codex_tool_bridge.has_unescaped_windows_exec_input(full_text):
                        repair_context = (
                            "INVALID_WINDOWS_PATH_ESCAPE\n"
                            "The exec JavaScript contained a single backslash in a Windows path. "
                            "Use forward slashes or escape each backslash as \\\\ before calling the tool.\n"
                            + full_text
                        )
                    elif duplicate_action:
                        repair_context = (
                            "DUPLICATE_COMPLETED_TOOL_ACTION\n"
                            "The proposed action already has a tool result in the current task. "
                            "Choose a different next action or return complete:true only if the task is actually finished.\n"
                            + full_text
                        )
                    elif structured_text_error:
                        repair_context = (
                            "INVALID_RESPONSE_TEXT_FORMAT\n"
                            f"The final text violates the requested Responses text.format schema: {structured_text_error}\n"
                            "Return a corrected final action whose text is exactly one schema-valid JSON document.\n"
                            + full_text
                        )
                    elif (
                        action is not None
                        and action.get("action") == "final"
                        and not codex_tool_bridge.final_action_has_sufficient_evidence(
                            action,
                            validation_input,
                            prior_output_items,
                        )
                    ):
                        repair_context = (
                            "INSUFFICIENT_TASK_EVIDENCE\n"
                            "The active task is not complete yet. Select a new, non-duplicate local tool action "
                            "that gathers the next required evidence; do not return a final action.\n"
                            + full_text
                        )
                    if request.conversation_id and request.parent_message_id and plan.continued:
                        # Do not append a repair to the poisoned Web conversation. The
                        # invalid assistant answer is already part of that cursor and
                        # can make the model repeat the same answer. Rebuild a small,
                        # self-contained controller request from the continuation plan;
                        # it retains the task/result delta without replaying the 100K
                        # Codex history or the invalid upstream node.
                        repair_messages = [{
                            "role": "system",
                            "content": codex_tool_bridge.controller_prompt(controller_tools),
                        }]
                        repair_messages.extend(codex_tool_bridge.controller_tool_messages(controller_tools))
                        repair_messages.extend(plan.messages)
                        repair_messages.extend(
                            codex_tool_bridge.controller_task_contract_messages(
                                getattr(plan, "canonical_input_items", None) or body.get("input")
                            )
                        )
                        repair_messages.extend(
                            codex_tool_bridge.controller_task_anchor_messages(
                                getattr(plan, "canonical_input_items", None) or body.get("input")
                            )
                        )
                        repair_messages.extend(
                            codex_tool_bridge.controller_parallel_tool_state_messages(
                                parallel_tool_calls
                            )
                        )
                        repair_messages.extend(codex_tool_bridge.controller_turn_state_messages(force_tool))
                        repair_messages.extend(codex_tool_bridge.controller_repair_messages(repair_context))
                        repair_conversation_id = ""
                        repair_parent_message_id = ""
                    elif request.conversation_id and request.parent_message_id:
                        # The initial request already contains the complete Codex
                        # transcript. Keep its Web cursor for a format/path repair so
                        # we do not duplicate a potentially large first payload.
                        repair_messages = codex_tool_bridge.controller_repair_messages(repair_context)
                        repair_messages.extend(
                            codex_tool_bridge.controller_task_contract_messages(
                                getattr(plan, "canonical_input_items", None) or body.get("input")
                            )
                        )
                        repair_messages.extend(
                            codex_tool_bridge.controller_task_anchor_messages(
                                getattr(plan, "canonical_input_items", None) or body.get("input")
                            )
                        )
                        repair_conversation_id = request.conversation_id
                        repair_parent_message_id = request.parent_message_id
                    else:
                        repair_messages = codex_tool_bridge.controller_messages(
                            body,
                            controller_tools,
                            force_tool=force_tool,
                            invalid_output=repair_context,
                        )
                        repair_conversation_id = ""
                        repair_parent_message_id = ""
                    _log_controller_request_shape(
                        repair_messages,
                        tool_count=len(controller_tools),
                        attempt="repair_continuation" if repair_conversation_id else "repair",
                    )
                    repair_request = ConversationRequest(
                        model=model,
                        messages=repair_messages,
                        thinking_effort=thinking_effort,
                        conversation_id=repair_conversation_id,
                        parent_message_id=repair_parent_message_id,
                        access_token=request.access_token,
                        controller_wire_bytes=request.controller_wire_bytes,
                    )
                    repair_parent_before = repair_request.parent_message_id
                    try:
                        repaired_text = _stream_controller_request(
                            backend,
                            repair_request,
                            attempt="repair_continuation" if repair_conversation_id else "repair",
                        )
                    except Exception as exc:
                        reset_repair = bool(repair_conversation_id) and (
                            _is_stale_controller_cursor_error(exc)
                            or _is_controller_payload_too_large_error(exc)
                        )
                        if not reset_repair:
                            raise
                        # The first Web conversation may have expired between the
                        # controller response and its repair. Rebuild the repair
                        # request without carrying a stale cursor.
                        codex_conversation_session.invalidate_controller_session(plan)
                        repair_messages = codex_tool_bridge.controller_messages(
                            body,
                            controller_tools,
                            force_tool=force_tool,
                            invalid_output=repair_context,
                        )
                        repair_conversation_id = ""
                        repair_parent_message_id = ""
                        repaired_text = ""
                        plan = codex_conversation_session.ContinuationPlan(
                            key=plan.key,
                            messages=repair_messages,
                            access_token=request.access_token or plan.access_token,
                        )
                        repair_request = ConversationRequest(
                            model=model,
                            messages=repair_messages,
                            thinking_effort=thinking_effort,
                            access_token=request.access_token or plan.access_token,
                            controller_wire_bytes=0,
                        )
                        repair_parent_before = ""
                        _log_controller_request_shape(
                            repair_messages,
                            tool_count=len(controller_tools),
                            attempt="repair_reset",
                        )
                        repaired_text = _stream_controller_request(
                            backend,
                            repair_request,
                            attempt="repair_reset",
                        )
                    _log_controller_output_shape(
                        repaired_text,
                        attempt="repair_continuation" if repair_conversation_id else "repair",
                    )
                    repaired_action = codex_tool_bridge.parse_controller_action(
                        repaired_text,
                        controller_tools,
                        allow_parallel=parallel_tool_calls,
                    )
                    _log_controller_parse_shape(
                        repaired_text,
                        repaired_action,
                        controller_tools,
                        force_tool=force_tool,
                        attempt="repair_continuation" if repair_conversation_id else "repair",
                    )
                    repaired_legacy_calls, _repaired_visible_text = parse_client_tool_calls(repaired_text, controller_tools)
                    if repaired_action is None and repaired_legacy_calls:
                        repaired_action = (
                            {"action": "tools", "calls": repaired_legacy_calls}
                            if parallel_tool_calls and len(repaired_legacy_calls) > 1
                            else {"action": "tool", **repaired_legacy_calls[0]}
                        )
                    if repaired_action is None and turn_has_tool_output:
                        repaired_action = _plain_controller_final(repaired_text)
                    repaired_refusal = (
                        repaired_action is not None
                        and repaired_action.get("action") == "final"
                        and (
                            codex_tool_bridge.is_access_refusal(str(repaired_action.get("text") or ""))
                            or codex_tool_bridge.is_task_evasion(str(repaired_action.get("text") or ""))
                        )
                    )
                    repaired_force_tool_rejection = (
                        force_tool
                        and repaired_action is not None
                        and not codex_tool_bridge.is_progress_tool_action(repaired_action)
                    )
                    repaired_duplicate_action = codex_tool_bridge.action_repeats_completed_tool(
                        repaired_action,
                        validation_input,
                        seed_items=prior_output_items,
                    )
                    repaired_completion_rejection = (
                        completion_required
                        and repaired_action is not None
                        and repaired_action.get("action") == "final"
                        and not codex_tool_bridge.final_action_has_sufficient_evidence(
                            repaired_action,
                            validation_input,
                            prior_output_items,
                        )
                    )
                    repaired_structured_text_error = _final_text_validation_error(body, repaired_action)
                    if (
                        repaired_action is not None
                        and not repaired_refusal
                        and not repaired_force_tool_rejection
                        and not repaired_duplicate_action
                        and not repaired_completion_rejection
                        and not repaired_structured_text_error
                    ):
                        action = repaired_action
                        full_text = repaired_text
                        request = repair_request
                        request_parent_before = repair_parent_before
                    elif repaired_structured_text_error:
                        raise ResponseTextValidationError(
                            "Codex controller returned invalid structured output after repair: "
                            + repaired_structured_text_error
                        )
                    elif force_tool or completion_required:
                        raise RuntimeError(
                            "Codex tool controller could not produce a valid progress action after repair; "
                            "refusing to fabricate an unrelated local command"
                        )
                    else:
                        raise RuntimeError("Codex tool controller could not produce a valid action")
                if action is None:
                    raise RuntimeError("Codex tool controller returned no action")
                if force_tool and not codex_tool_bridge.is_progress_tool_action(action):
                    raise RuntimeError(
                        "Codex request requires a progress tool, but the controller returned a terminal action"
                    )
                searched_queries: set[str] = set()
                last_search_text = ""
                while _is_hosted_web_search_action(action):
                    query = _hosted_web_search_query(action)
                    normalized_query = " ".join(query.lower().split())
                    if (
                        not query
                        or normalized_query in searched_queries
                        or len(searched_queries) >= CONTROLLER_MAX_HOSTED_SEARCHES
                    ):
                        action = {
                            "action": "final",
                            "text": last_search_text or "The hosted web search produced no usable result.",
                            "complete": True,
                        }
                        break

                    searched_queries.add(normalized_query)
                    search_result = run_web_search(query)
                    search_events, search_item = _hosted_web_search_events(
                        query,
                        search_result,
                        len(output),
                    )
                    pending_events.extend(search_events)
                    output.append(search_item)
                    last_search_text, _search_annotations = text_with_url_citations(search_result)

                    request.messages = _controller_hosted_search_messages(
                        body,
                        controller_tools,
                        query,
                        search_result,
                        include_bootstrap=not bool(request.conversation_id and request.parent_message_id),
                    )
                    _log_controller_request_shape(
                        request.messages,
                        tool_count=len(controller_tools),
                        attempt=f"hosted_web_search_{len(searched_queries)}",
                    )
                    followup_text = _stream_controller_request(
                        backend,
                        request,
                        attempt=f"hosted_web_search_{len(searched_queries)}",
                    )
                    full_text = "\n".join(part for part in (full_text, followup_text) if part)
                    followup_action = codex_tool_bridge.parse_controller_action(
                        followup_text,
                        controller_tools,
                        allow_parallel=parallel_tool_calls,
                    )
                    if followup_action is None:
                        plain_final = _plain_controller_final(followup_text)
                        if plain_final is not None:
                            plain_final["complete"] = True
                            followup_action = plain_final

                    followup_structured_text_error = _final_text_validation_error(body, followup_action)
                    followup_invalid = (
                        followup_action is None
                        or codex_tool_bridge.action_repeats_completed_tool(
                            followup_action,
                            validation_input,
                            seed_items=prior_output_items,
                        )
                        or (
                            followup_action.get("action") == "final"
                            and (
                                codex_tool_bridge.is_access_refusal(str(followup_action.get("text") or ""))
                                or codex_tool_bridge.is_task_evasion(str(followup_action.get("text") or ""))
                                or not codex_tool_bridge.final_action_has_sufficient_evidence(
                                    followup_action,
                                    validation_input,
                                    prior_output_items,
                                )
                            )
                        )
                        or bool(followup_structured_text_error)
                    )
                    if followup_invalid:
                        followup_repair_context = followup_text
                        if followup_structured_text_error:
                            followup_repair_context = (
                                "INVALID_RESPONSE_TEXT_FORMAT\n"
                                f"The final text violates the requested Responses text.format schema: {followup_structured_text_error}\n"
                                "Return a corrected final action whose text is exactly one schema-valid JSON document.\n"
                                + followup_text
                            )
                        request.messages = [
                            *_controller_hosted_search_messages(
                                body,
                                controller_tools,
                                query,
                                search_result,
                                include_bootstrap=not bool(request.conversation_id and request.parent_message_id),
                            ),
                            *codex_tool_bridge.controller_repair_messages(followup_repair_context),
                        ]
                        repaired_followup_text = _stream_controller_request(
                            backend,
                            request,
                            attempt=f"hosted_web_search_{len(searched_queries)}_repair",
                        )
                        full_text = "\n".join(
                            part for part in (full_text, repaired_followup_text) if part
                        )
                        repaired_followup_action = codex_tool_bridge.parse_controller_action(
                            repaired_followup_text,
                            controller_tools,
                            allow_parallel=parallel_tool_calls,
                        )
                        repaired_followup_structured_text_error = _final_text_validation_error(
                            body,
                            repaired_followup_action,
                        )
                        repaired_followup_invalid = (
                            repaired_followup_action is None
                            or codex_tool_bridge.action_repeats_completed_tool(
                                repaired_followup_action,
                                validation_input,
                                seed_items=prior_output_items,
                            )
                            or (
                                repaired_followup_action.get("action") == "final"
                                and (
                                    codex_tool_bridge.is_access_refusal(
                                        str(repaired_followup_action.get("text") or "")
                                    )
                                    or codex_tool_bridge.is_task_evasion(
                                        str(repaired_followup_action.get("text") or "")
                                    )
                                    or not codex_tool_bridge.final_action_has_sufficient_evidence(
                                        repaired_followup_action,
                                        validation_input,
                                        prior_output_items,
                                    )
                                )
                            )
                            or bool(repaired_followup_structured_text_error)
                        )
                        if repaired_followup_structured_text_error:
                            raise ResponseTextValidationError(
                                "Codex controller returned invalid structured output after search repair: "
                                + repaired_followup_structured_text_error
                            )
                        followup_action = None if repaired_followup_invalid else repaired_followup_action

                    if followup_action is None:
                        followup_action = {
                            "action": "final",
                            "text": last_search_text or "The hosted web search produced no usable result.",
                            "complete": True,
                        }
                    action = followup_action

                final_text_error = _final_text_validation_error(body, action)
                if final_text_error:
                    raise ResponseTextValidationError(
                        "Codex controller returned invalid structured output: " + final_text_error
                    )
                visible_text = str(action.get("text") or "") if action.get("action") == "final" else ""
                if visible_text or action.get("action") == "final":
                    text_events, text_item = _text_output_events(visible_text, len(output))
                    pending_events.extend(text_events)
                    output.append(text_item)
                if action.get("action") == "tool":
                    tool_events, tool_items = _client_tool_events([action], len(output))
                    pending_events.extend(tool_events)
                    output.extend(tool_items)
                elif action.get("action") == "tools":
                    calls = [
                        call for call in action.get("calls", [])
                        if isinstance(call, dict)
                    ]
                    tool_events, tool_items = _client_tool_events(calls, len(output))
                    pending_events.extend(tool_events)
                    output.extend(tool_items)
                usage = token_usage(
                    input_text_tokens=count_message_text_tokens(full_controller_messages, model),
                    input_image_tokens=count_message_image_tokens(full_controller_messages, model),
                    output_text_tokens=count_text_tokens(full_text, model),
                )
                committed = codex_conversation_session.commit_controller_response(
                    plan,
                    body,
                    controller_tools,
                    output,
                    conversation_id=request.conversation_id,
                    parent_message_id=(
                        request.parent_message_id
                        if request.parent_message_id != request_parent_before
                        else ""
                    ),
                    access_token=request.access_token,
                    response_id=response_id,
                    usage=usage,
                    upstream_wire_bytes=request.controller_wire_bytes,
                )
                if not committed:
                    # Do not leave a stale parent cursor after an incomplete SSE
                    # stream. The next request will deliberately rebuild the Web
                    # conversation instead of branching from an unknown node.
                    codex_conversation_session.invalidate_controller_session(plan)
                    logger.warning({
                        "event": "codex_controller_session_not_committed",
                        "continued": plan.continued,
                        "has_conversation_id": bool(request.conversation_id),
                        "has_parent_message_id": bool(request.parent_message_id),
                    })
        yield from pending_events
        yield response_completed(
            response_id,
            model,
            created,
            output,
            usage,
            end_turn=not any(item.get("type") in codex_tool_bridge.TOOL_CALL_TYPES for item in output),
            parallel_tool_calls=parallel_tool_calls,
        )
        return
    controls = codex_response_text.normalized_text_controls(body)
    structured_format = controls.get("format") if isinstance(controls, dict) else None
    if isinstance(structured_format, dict):
        request_messages = messages
        for attempt in range(3):
            request = ConversationRequest(
                model=model,
                messages=request_messages,
                thinking_effort=thinking_effort,
            )
            candidate = "".join(_stream_plain_text_request(
                backend,
                request,
                attempt=f"structured_{attempt + 1}",
            ))
            valid, validation_error = codex_response_text.validate_response_text(body, candidate)
            if valid:
                full_text = candidate
                messages = request_messages
                break
            if attempt == 2:
                raise ResponseTextValidationError(
                    "Model returned invalid structured output after two repairs: "
                    + validation_error
                )
            request_messages = [
                *messages,
                {
                    "role": "system",
                    "content": (
                        "STRUCTURED_OUTPUT_REPAIR\n"
                        f"The prior answer violated text.format: {validation_error}\n"
                        "Return only one JSON document that validates against the requested schema."
                    ),
                },
                {
                    "role": "assistant",
                    "content": codex_tool_bridge._truncate_utf8(
                        candidate,
                        24 * 1024,
                        "\n[invalid output truncated]",
                    ),
                },
                {
                    "role": "user",
                    "content": "Correct the structured output now. Return JSON only.",
                },
            ]
        text_events, item = _text_output_events(full_text, 0)
        yield from text_events
        usage = token_usage(
            input_text_tokens=count_message_text_tokens(messages, model),
            input_image_tokens=count_message_image_tokens(messages, model),
            output_text_tokens=count_text_tokens(full_text, model),
        )
        yield response_completed(
            response_id,
            model,
            created,
            [item],
            usage,
            end_turn=True,
            parallel_tool_calls=parallel_tool_calls,
        )
        return

    request = ConversationRequest(model=model, messages=messages, thinking_effort=thinking_effort)
    yield {"type": "response.output_item.added", "output_index": 0, "item": text_output_item("", item_id, "in_progress")}
    for delta in _stream_plain_text_request(backend, request, attempt="plain"):
        full_text += delta
        yield {"type": "response.output_text.delta", "item_id": item_id, "output_index": 0, "content_index": 0, "delta": delta}
    yield {"type": "response.output_text.done", "item_id": item_id, "output_index": 0, "content_index": 0, "text": full_text}
    item = text_output_item(full_text, item_id, "completed")
    yield {"type": "response.output_item.done", "output_index": 0, "item": item}
    usage = token_usage(
        input_text_tokens=count_message_text_tokens(messages, model),
        input_image_tokens=count_message_image_tokens(messages, model),
        output_text_tokens=count_text_tokens(full_text, model),
    )
    yield response_completed(
        response_id,
        model,
        created,
        [item],
        usage,
        end_turn=True,
        parallel_tool_calls=parallel_tool_calls,
    )


def stream_web_search_response(body: dict[str, Any], messages: list[dict[str, Any]] | None = None) -> Iterator[dict[str, Any]]:
    model = str(body.get("model") or "auto").strip() or "auto"
    messages = messages if messages is not None else messages_from_input(body.get("input"), body.get("instructions"))
    query = search_query_from_messages(messages) or extract_response_prompt(body.get("input"))
    if not query:
        raise HTTPException(status_code=400, detail={"error": "input text is required for web_search"})

    response_id = str(body.get("_response_id") or f"resp_{uuid.uuid4().hex}")
    search_id = f"ws_{uuid.uuid4().hex}"
    item_id = f"msg_{uuid.uuid4().hex}"
    created = int(time.time())
    yield response_created(response_id, model, created)

    searching_item = web_search_call_item(query, search_id, "in_progress")
    yield {"type": "response.output_item.added", "output_index": 0, "item": searching_item}
    yield {"type": "response.web_search_call.in_progress", "output_index": 0, "item_id": search_id}
    yield {"type": "response.web_search_call.searching", "output_index": 0, "item_id": search_id}
    result = run_web_search(query)
    search_item = web_search_call_item(query, search_id, "completed", normalized_sources(result))
    yield {"type": "response.web_search_call.completed", "output_index": 0, "item_id": search_id}
    yield {"type": "response.output_item.done", "output_index": 0, "item": search_item}

    text, annotations = text_with_url_citations(result)
    message_item = text_output_item("", item_id, "in_progress", annotations)
    yield {"type": "response.output_item.added", "output_index": 1, "item": message_item}
    if text:
        yield {"type": "response.output_text.delta", "item_id": item_id, "output_index": 1, "content_index": 0, "delta": text}
    yield {"type": "response.output_text.done", "item_id": item_id, "output_index": 1, "content_index": 0, "text": text}
    message_item = text_output_item(text, item_id, "completed", annotations)
    yield {"type": "response.output_item.done", "output_index": 1, "item": message_item}
    usage = token_usage(
        input_text_tokens=count_message_text_tokens(messages, model),
        input_image_tokens=count_message_image_tokens(messages, model),
        output_text_tokens=count_text_tokens(text, model),
    )
    yield response_completed(response_id, model, created, [search_item, message_item], usage, end_turn=True)


def stream_image_response(
    image_outputs: Iterable[ImageOutput],
    prompt: str,
    model: str,
    input_image_tokens: int = 0,
    size: object = None,
    quality: str = "auto",
    response_id: str = "",
) -> Iterator[dict[str, Any]]:
    response_id = str(response_id or f"resp_{uuid.uuid4().hex}")
    created = int(time.time())
    yield response_created(response_id, model, created)
    for output in image_outputs:
        if output.kind == "message":
            text = output.text
            item_id = f"msg_{uuid.uuid4().hex}"
            yield {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": text_output_item("", item_id, "in_progress"),
            }
            usage = token_usage(
                input_text_tokens=count_text_tokens(prompt, model),
                input_image_tokens=input_image_tokens,
                output_text_tokens=count_text_tokens(text, model),
            )
            yield {"type": "response.output_text.delta", "item_id": item_id, "output_index": 0, "content_index": 0, "delta": text}
            yield {"type": "response.output_text.done", "item_id": item_id, "output_index": 0, "content_index": 0, "text": text}
            item = text_output_item(text, item_id, "completed")
            yield {"type": "response.output_item.done", "output_index": 0, "item": item}
            yield response_completed(response_id, model, created, [item], usage, end_turn=True)
            return
        if output.kind != "result":
            continue
        items = image_output_items(prompt, output.data)
        if items:
            usage = image_usage(
                input_text_tokens=count_text_tokens(prompt, model),
                input_image_tokens=input_image_tokens,
                output_tokens=count_image_output_items_tokens(output.data, size, quality),
            )
            for output_index, item in enumerate(items):
                in_progress_item = dict(item)
                in_progress_item["status"] = "in_progress"
                in_progress_item.pop("result", None)
                yield {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": in_progress_item,
                }
                yield {"type": "response.output_item.done", "output_index": output_index, "item": item}
            yield response_completed(response_id, model, created, items, usage, end_turn=True)
            return
    raise RuntimeError("image generation failed")


def collect_response(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    completed = {}
    for event in events:
        if event.get("type") == "response.completed":
            completed = event.get("response") if isinstance(event.get("response"), dict) else {}
    if not completed:
        raise RuntimeError("response generation failed")
    return completed


COMPACTION_SOURCE_MAX_BYTES = 24 * 1024
COMPACTION_ITEM_MAX_BYTES = 4 * 1024
COMPACTION_OUTPUT_MAX_BYTES = 20 * 1024
COMPACTION_EARLY_CONTEXT_MAX_BYTES = 4 * 1024
COMPACTION_OMISSION_NOTICE = "older compacted context omitted; recent records retained"


def _utf8_size(value: str) -> int:
    return len(value.encode("utf-8"))


def _truncate_compaction_middle(value: str, max_bytes: int, notice: str) -> str:
    if max_bytes <= 0:
        return ""
    if _utf8_size(value) <= max_bytes:
        return value
    marker = f"\n[{notice}]\n"
    marker_bytes = _utf8_size(marker)
    if marker_bytes >= max_bytes:
        return codex_tool_bridge._truncate_utf8(value, max_bytes)
    content_budget = max_bytes - marker_bytes
    head_budget = (content_budget * 2) // 3
    tail_budget = content_budget - head_budget
    encoded = value.encode("utf-8")
    head = encoded[:head_budget].decode("utf-8", errors="ignore")
    tail = encoded[-tail_budget:].decode("utf-8", errors="ignore") if tail_budget else ""
    return head + marker + tail


def _compaction_join(records: list[str]) -> str:
    return "\n\n".join(record for record in records if record)


def _fit_required_compaction_records(records: list[str], max_bytes: int) -> list[str]:
    """Fit priority records fairly, preserving both ends only when they cannot all fit."""
    if not records:
        return []
    if _utf8_size(_compaction_join(records)) <= max_bytes:
        return records

    separator_bytes = 2 * (len(records) - 1)
    remaining = max(0, max_bytes - separator_bytes)
    sizes = [_utf8_size(record) for record in records]
    allocations = [0] * len(records)
    pending = set(range(len(records)))
    while pending:
        share = remaining // len(pending)
        fitting = [index for index in pending if sizes[index] <= share]
        if not fitting:
            for offset, index in enumerate(sorted(pending)):
                allocations[index] = share + (1 if offset < remaining % len(pending) else 0)
            break
        for index in fitting:
            allocations[index] = sizes[index]
            remaining -= sizes[index]
            pending.remove(index)

    return [
        _truncate_compaction_middle(record, allocations[index], "record middle omitted for compaction budget")
        for index, record in enumerate(records)
        if allocations[index] > 0
    ]


def _early_compaction_excerpt(records: list[str], max_bytes: int) -> str:
    if not records or max_bytes <= 0:
        return ""
    label = "EARLY_CONTEXT_EXCERPT\n"
    if _utf8_size(label) >= max_bytes:
        return ""
    excerpt = _truncate_compaction_middle(
        _compaction_join(records),
        max_bytes - _utf8_size(label),
        COMPACTION_OMISSION_NOTICE,
    )
    return label + excerpt


def _compaction_item_text(
    item: object,
    item_index: int,
    calls_by_id: dict[str, dict[str, Any]],
) -> str:
    if not isinstance(item, dict):
        return f"INPUT[{item_index}] role=user type=text\n{item}"
    item_type = str(item.get("type") or "message").strip().lower()
    if item_type == "compaction_trigger":
        return (
            f"INPUT[{item_index}] role=system type=compaction_trigger\n"
            "REMOTE_COMPACTION_V2_TRIGGER"
        )
    tool_message = codex_tool_bridge.tool_history_message(item, calls_by_id)
    if tool_message:
        role = str(tool_message.get("role") or "user")
        content = str(tool_message.get("content") or "")
        return f"INPUT[{item_index}] role={role} type={item_type}\n{content}"
    history_message = codex_tool_bridge.response_item_history_message(item)
    if history_message:
        role = str(history_message.get("role") or "assistant")
        content = str(history_message.get("content") or "")
        return f"INPUT[{item_index}] role={role} type={item_type}\n{content}"
    if item_type == "message" or "content" in item:
        text = "\n".join(
            part_text
            for _part_type, part_text in codex_tool_bridge._response_text_parts(item.get("content"))
            if part_text.strip()
        )
        return (
            f"INPUT[{item_index}] role={item.get('role') or 'user'} type={item_type}\n{text}"
            if text
            else f"INPUT[{item_index}] role={item.get('role') or 'user'} type={item_type}"
        )
    return (
        f"INPUT[{item_index}] role={item.get('role') or 'user'} type={item_type}\n"
        + json.dumps(item, ensure_ascii=False, separators=(",", ":"))
    )


def _compaction_source(body: dict[str, Any]) -> str:
    early_records: list[str] = []
    instructions = str(body.get("instructions") or "").strip()
    if instructions:
        early_records.append("TOP_LEVEL_INSTRUCTIONS role=system\n" + instructions)
    input_value = body.get("input")
    items = input_value if isinstance(input_value, list) else [input_value]
    calls_by_id: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    call_positions: dict[str, int] = {}
    latest_call_position: int | None = None
    latest_output_position: int | None = None
    latest_user_position: int | None = None
    latest_trigger_position: int | None = None
    for item_index, item in enumerate(items):
        if item is None:
            continue
        item_type = (
            str(item.get("type") or "message").strip().lower()
            if isinstance(item, dict)
            else "text"
        )
        record = _compaction_item_text(item, item_index, calls_by_id)
        position = len(records)
        call_id = str(item.get("call_id") or "").strip() if isinstance(item, dict) else ""
        records.append({"text": record, "type": item_type, "call_id": call_id})
        if not isinstance(item, dict):
            latest_user_position = position
        elif item_type == "message" and str(item.get("role") or "user").strip().lower() == "user":
            latest_user_position = position
        elif item_type == "agent_message":
            latest_user_position = position
        if item_type in codex_tool_bridge.TOOL_CALL_TYPES:
            latest_call_position = position
            if call_id:
                call_positions[call_id] = position
        elif item_type in codex_tool_bridge.TOOL_OUTPUT_TYPES:
            latest_output_position = position
        elif item_type == "compaction_trigger":
            latest_trigger_position = position

    if not records and not early_records:
        return ""

    required_positions = {
        position
        for position in (latest_user_position, latest_call_position, latest_output_position, latest_trigger_position)
        if position is not None
    }
    if latest_output_position is not None:
        output_call_id = str(records[latest_output_position].get("call_id") or "")
        paired_call_position = call_positions.get(output_call_id)
        if paired_call_position is not None:
            required_positions.add(paired_call_position)
    if not required_positions and records:
        required_positions.add(len(records) - 1)

    ordered_required = [
        str(records[position]["text"])
        for position in sorted(required_positions)
    ]
    fitted_required = _fit_required_compaction_records(
        ordered_required,
        COMPACTION_SOURCE_MAX_BYTES,
    )
    required_size = _utf8_size(_compaction_join(fitted_required))

    earliest_required = min(required_positions, default=len(records))
    prefix_records = early_records + [
        str(record["text"])
        for position, record in enumerate(records)
        if position < earliest_required
    ]
    early_budget = min(
        COMPACTION_EARLY_CONTEXT_MAX_BYTES,
        max(0, COMPACTION_SOURCE_MAX_BYTES - required_size - (2 if fitted_required else 0)),
    )
    early_excerpt = _early_compaction_excerpt(prefix_records, early_budget)

    selected_positions = set(required_positions)
    selected_optional: dict[int, str] = {}
    base_records = ([early_excerpt] if early_excerpt else []) + fitted_required
    remaining = COMPACTION_SOURCE_MAX_BYTES - _utf8_size(_compaction_join(base_records))
    for position in range(len(records) - 1, earliest_required - 1, -1):
        if position in selected_positions or remaining <= 2:
            continue
        available = remaining - 2
        candidate = str(records[position]["text"])
        candidate = _truncate_compaction_middle(
            candidate,
            min(COMPACTION_ITEM_MAX_BYTES, available),
            "record middle omitted for compaction budget",
        )
        if not candidate:
            continue
        selected_optional[position] = candidate
        remaining -= _utf8_size(candidate) + 2

    recent_records: list[str] = []
    required_by_position = dict(zip(sorted(required_positions), fitted_required))
    for position in sorted(required_positions | set(selected_optional)):
        if position in required_by_position:
            recent_records.append(required_by_position[position])
        else:
            recent_records.append(selected_optional[position])
    source = _compaction_join(([early_excerpt] if early_excerpt else []) + recent_records)
    return codex_tool_bridge._truncate_utf8(source, COMPACTION_SOURCE_MAX_BYTES)


def _upstream_compaction_summary(body: dict[str, Any], source: str) -> str:
    if not source:
        return ""
    model = str(body.get("model") or "auto").strip() or "auto"
    backend = None
    try:
        backend = text_backend(model)
        request = ConversationRequest(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize this Codex execution transcript for a later model. "
                        "Preserve the active user request, exact Windows paths, files, commands, "
                        "tool results, errors, and unfinished steps. Do not invent completion. "
                        "Return only the compact summary."
                    ),
                },
                {"role": "user", "content": source},
            ],
            thinking_effort=thinking_effort_from_body(body),
        )
        summary = "".join(stream_text_deltas(backend, request)).strip()
        if summary:
            return codex_tool_bridge._truncate_utf8(summary, COMPACTION_OUTPUT_MAX_BYTES)
    except Exception as exc:  # noqa: BLE001 - local compaction remains available
        logger.warning({
            "event": "codex_compaction_upstream_failed",
            "error_type": type(exc).__name__,
        })
    finally:
        if backend is not None:
            backend.close()
    return ""


def compact(body: dict[str, Any]) -> dict[str, Any]:
    """Return a Responses compaction item for Codex's long-history endpoint."""
    source = _compaction_source(body)
    summary = _upstream_compaction_summary(body, source)
    if not summary:
        summary = source or "No prior Codex context was supplied."
    summary = codex_tool_bridge._truncate_utf8(summary, COMPACTION_OUTPUT_MAX_BYTES)
    return {
        "output": [{
            "type": "compaction",
            "encrypted_content": "CODEX_COMPACTION_SUMMARY\n" + summary,
        }],
    }


def _responses_request_kind(body: dict[str, Any]) -> str:
    metadata = body.get("client_metadata")
    if not isinstance(metadata, dict):
        return ""
    value = metadata.get("request_kind")
    if value:
        return str(value).strip().lower()
    nested = metadata.get("x-codex-turn-metadata")
    if isinstance(nested, str):
        try:
            parsed = json.loads(nested)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            return str(parsed.get("request_kind") or "").strip().lower()
    return ""


def is_local_compaction_request(body: dict[str, Any]) -> bool:
    """Codex local compaction is a normal Responses request with a full history."""
    return _responses_request_kind(body) == "compaction" and not has_compaction_trigger(body)


def stream_local_compaction_response(body: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Summarize a bounded transcript for Codex's local-compaction reducer."""
    response_id = str(body.get("_response_id") or f"resp_{uuid.uuid4().hex}")
    model = str(body.get("model") or "auto").strip() or "auto"
    created = int(time.time())
    source = _compaction_source(body)
    summary = _upstream_compaction_summary(body, source)
    if not summary:
        summary = source or "No prior Codex context was supplied."
    summary = codex_tool_bridge._truncate_utf8(summary, COMPACTION_OUTPUT_MAX_BYTES)
    yield response_created(response_id, model, created)
    events, item = _text_output_events(summary, 0)
    yield from events
    usage = token_usage(
        input_text_tokens=count_text_tokens(source, model),
        output_text_tokens=count_text_tokens(summary, model),
    )
    yield response_completed(response_id, model, created, [item], usage, end_turn=True)


def has_compaction_trigger(body: dict[str, Any]) -> bool:
    input_value = body.get("input")
    return isinstance(input_value, list) and any(
        isinstance(item, dict)
        and str(item.get("type") or "").strip().lower() == "compaction_trigger"
        for item in input_value
    )


def stream_compaction_trigger_response(body: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Serve Codex remote-compaction V2 on the normal Responses stream."""
    response_id = str(body.get("_response_id") or f"resp_{uuid.uuid4().hex}")
    model = str(body.get("model") or "auto").strip() or "auto"
    created = int(time.time())
    result = compact(body)
    compacted = result["output"][0]
    item = {
        "id": f"cmp_{uuid.uuid4().hex}",
        "type": "compaction",
        "encrypted_content": compacted["encrypted_content"],
    }
    source = _compaction_source(body)
    usage = token_usage(
        input_text_tokens=count_text_tokens(source, model),
        output_text_tokens=count_text_tokens(str(item["encrypted_content"]), model),
    )
    yield response_created(response_id, model, created)
    yield {"type": "response.output_item.added", "output_index": 0, "item": item}
    yield {"type": "response.output_item.done", "output_index": 0, "item": item}
    yield response_completed(
        response_id,
        model,
        created,
        [item],
        usage,
        end_turn=True,
    )


def response_events(body: dict[str, Any]) -> Iterator[dict[str, Any]]:
    # Validate opaque/media inputs before every dispatch path, including remote
    # compaction.  Compaction cannot recover encrypted inter-agent messages.
    codex_tool_bridge.ensure_supported_media(body.get("input"))
    if has_compaction_trigger(body):
        yield from stream_compaction_trigger_response(body)
        return
    if is_local_compaction_request(body):
        yield from stream_local_compaction_response(body)
        return
    if is_text_response_request(body):
        model, messages = text_response_parts(body)
        if hosted_web_search_enabled(body) and not has_unsupported_response_tools(body):
            yield from stream_web_search_response(body, messages)
            return
        if response_client_tools(body):
            yield from stream_text_response(None, body, messages)
            return
        key = cache_key(body, messages, stream=bool(body.get("stream")))
        yield from _with_response_id(
            chat_completion_cache.get_or_compute_stream(
                key,
                lambda: stream_text_response(None, body, messages),
            ),
            body.get("_response_id"),
        )
        return

    prompt = extract_response_prompt(body.get("input"))
    if not prompt:
        raise HTTPException(status_code=400, detail={"error": "input text is required"})
    model = str(body.get("model") or "gpt-image-2").strip() or "gpt-image-2"
    image_info = extract_response_image(body.get("input"))
    if image_info:
        image_data, mime_type = image_info
        images = encode_images([(image_data, "image.png", mime_type)])
    else:
        images = None
    input_image_tokens = count_image_content_tokens(_input_image_parts(body.get("input")), model)
    tool = response_image_tool(body)
    image_outputs = stream_image_outputs_with_pool(ConversationRequest(
        prompt=prompt,
        model=model,
        size=tool.get("size"),
        quality=str(tool.get("quality") or "auto"),
        response_format="b64_json",
        images=images,
    ))
    yield from stream_image_response(
        image_outputs,
        prompt,
        model,
        input_image_tokens,
        tool.get("size"),
        str(tool.get("quality") or "auto"),
        str(body.get("_response_id") or ""),
    )


def handle(body: dict[str, Any]) -> dict[str, Any] | Iterator[dict[str, Any]]:
    events = response_events(body)
    if body.get("stream"):
        return events
    return collect_response(events)
