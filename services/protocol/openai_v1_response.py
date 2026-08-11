from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any, Iterable, Iterator

from fastapi import HTTPException

from services.protocol.chat_completion_cache import cache_key, chat_completion_cache, normalize_text_messages
from services.protocol import codex_conversation_session, codex_tool_bridge
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


def _append_response_message(messages: list[dict[str, Any]], role: object, content: object) -> None:
    if isinstance(content, str):
        if content.strip():
            messages.append({"role": str(role or "user"), "content": content.strip()})
        return
    if isinstance(content, list) and content:
        messages.append({"role": str(role or "user"), "content": content})


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
        _append_response_message(
            messages,
            input_value.get("role") or "user",
            _message_content_from_response_item(input_value),
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
            _append_response_message(
                messages,
                item.get("role") or "user",
                _message_content_from_response_item(item),
            )
        if pending_parts:
            _append_response_message(messages, "user", pending_parts)
    return messages


def text_output_item(
    text: str,
    item_id: str | None = None,
    status: str = "completed",
    annotations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": item_id or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "status": status,
        "role": "assistant",
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


def response_created(response_id: str, model: str, created: int) -> dict[str, Any]:
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
            "parallel_tool_calls": False,
        },
    }


def response_completed(
    response_id: str,
    model: str,
    created: int,
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None = None,
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
            "parallel_tool_calls": False,
        },
    }
    if usage:
        response["response"]["usage"] = usage
    return response


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


def _log_controller_request_shape(
    messages: list[dict[str, Any]],
    *,
    tool_count: int,
    attempt: str,
) -> None:
    contents = [str(message.get("content") or "") for message in messages]
    encoded_sizes = [len(content.encode("utf-8")) for content in contents]
    logger.debug({
        "event": "codex_controller_request_shape",
        "attempt": attempt,
        "message_count": len(messages),
        "tool_count": tool_count,
        "content_bytes": sum(encoded_sizes),
        "max_message_bytes": max(encoded_sizes, default=0),
        "max_message_chars": max((len(content) for content in contents), default=0),
    })


def _log_controller_output_shape(text: str, *, attempt: str) -> None:
    """Expose a bounded controller preview only when explicitly debugging upstream output."""
    if os.getenv("CHATGPT2API_DEBUG_CONTROLLER_OUTPUT") != "1":
        return
    value = str(text or "")
    logger.debug({
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
    logger.debug({
        "event": "codex_controller_parse_shape",
        "attempt": attempt,
        "force_tool": force_tool,
        "parsed": bool(action),
        "parsed_name": str((action or {}).get("name") or ""),
        "parsed_kind": str((action or {}).get("kind") or ""),
        "tool_names": [
            f"{tool.get('namespace') or ''}.{tool.get('name') or ''}:{tool.get('kind') or ''}"
            for tool in tools
        ],
        "text_bytes": len(str(text or '').encode('utf-8')),
    })


def _plain_controller_final(text: str) -> dict[str, str] | None:
    source = str(text or "").strip()
    if (
        not source
        or codex_tool_bridge.is_access_refusal(source)
        or source.startswith(("{", "[", "<codex_tool_call", "<custom_tool_call", "<tool_call"))
    ):
        return None
    return {"action": "final", "text": source}


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
    response_id = f"resp_{uuid.uuid4().hex}"
    item_id = f"msg_{uuid.uuid4().hex}"
    created = int(time.time())
    full_text = ""
    yield response_created(response_id, model, created)
    client_tools = response_client_tools(body)
    if client_tools:
        force_tool = (
            codex_tool_bridge.requires_local_tool(body, client_tools)
            and not codex_tool_bridge.current_turn_has_tool_output(body.get("input"))
        )
        full_controller_messages = codex_tool_bridge.controller_messages(
            body, client_tools, force_tool=force_tool,
        )
        pending_events: list[dict[str, Any]] = []
        output: list[dict[str, Any]] = []
        usage: dict[str, Any] = {}
        with codex_conversation_session.controller_session_lock(body):
            plan = codex_conversation_session.prepare_controller_request(
                body,
                client_tools,
                full_controller_messages,
                force_tool=force_tool,
            )
            if plan.replayed:
                pending_events.extend(_replay_controller_output(plan.output_items))
                output = plan.output_items
                usage = plan.usage
            else:
                controller_messages = plan.messages
                _log_controller_request_shape(
                    controller_messages,
                    tool_count=len(client_tools),
                    attempt="continuation" if plan.continued else "initial",
                )
                request = ConversationRequest(
                    model=model,
                    messages=controller_messages,
                    thinking_effort=thinking_effort,
                    conversation_id=plan.conversation_id,
                    parent_message_id=plan.parent_message_id,
                    access_token=plan.access_token,
                )
                request_parent_before = request.parent_message_id
                try:
                    for delta in stream_text_deltas(backend, request):
                        full_text += delta
                    _log_controller_output_shape(full_text, attempt="initial" if not plan.continued else "continuation")
                except Exception as exc:
                    if not plan.continued or not _is_stale_controller_cursor_error(exc):
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
                        tool_count=len(client_tools),
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
                    for delta in stream_text_deltas(backend, request):
                        full_text += delta
                    _log_controller_output_shape(full_text, attempt="continuation_reset")
                action = codex_tool_bridge.parse_controller_action(full_text, client_tools)
                _log_controller_parse_shape(
                    full_text,
                    action,
                    client_tools,
                    force_tool=force_tool,
                    attempt="continuation" if plan.continued else "initial",
                )
                legacy_calls, _legacy_visible_text = parse_client_tool_calls(full_text, client_tools)
                if action is None and legacy_calls:
                    action = {"action": "tool", **legacy_calls[0]}
                if action is None and codex_tool_bridge.current_turn_has_tool_output(body.get("input")):
                    action = _plain_controller_final(full_text)
                rejected_action = (
                    action is not None
                    and (
                        (action.get("action") == "final" and codex_tool_bridge.is_access_refusal(str(action.get("text") or "")))
                        or (force_tool and not codex_tool_bridge.is_local_executor_action(action))
                    )
                )
                invalid_output = action is None
                if rejected_action or invalid_output:
                    repaired_text = ""
                    if request.conversation_id and request.parent_message_id:
                        repair_messages = codex_tool_bridge.controller_repair_messages(full_text)
                        repair_conversation_id = request.conversation_id
                        repair_parent_message_id = request.parent_message_id
                    else:
                        repair_messages = codex_tool_bridge.controller_messages(
                            body,
                            client_tools,
                            force_tool=force_tool,
                            invalid_output=full_text,
                        )
                        repair_conversation_id = ""
                        repair_parent_message_id = ""
                    _log_controller_request_shape(
                        repair_messages,
                        tool_count=len(client_tools),
                        attempt="repair_continuation" if repair_conversation_id else "repair",
                    )
                    repair_request = ConversationRequest(
                        model=model,
                        messages=repair_messages,
                        thinking_effort=thinking_effort,
                        conversation_id=repair_conversation_id,
                        parent_message_id=repair_parent_message_id,
                        access_token=request.access_token,
                    )
                    repair_parent_before = repair_request.parent_message_id
                    try:
                        for delta in stream_text_deltas(backend, repair_request):
                            repaired_text += delta
                    except Exception as exc:
                        if not repair_conversation_id or not _is_stale_controller_cursor_error(exc):
                            raise
                        # The first Web conversation may have expired between the
                        # controller response and its repair. Rebuild the repair
                        # request without carrying a stale cursor.
                        codex_conversation_session.invalidate_controller_session(plan)
                        repair_messages = codex_tool_bridge.controller_messages(
                            body,
                            client_tools,
                            force_tool=force_tool,
                            invalid_output=full_text,
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
                        )
                        repair_parent_before = ""
                        _log_controller_request_shape(
                            repair_messages,
                            tool_count=len(client_tools),
                            attempt="repair_reset",
                        )
                        for delta in stream_text_deltas(backend, repair_request):
                            repaired_text += delta
                    _log_controller_output_shape(
                        repaired_text,
                        attempt="repair_continuation" if repair_conversation_id else "repair",
                    )
                    repaired_action = codex_tool_bridge.parse_controller_action(repaired_text, client_tools)
                    _log_controller_parse_shape(
                        repaired_text,
                        repaired_action,
                        client_tools,
                        force_tool=force_tool,
                        attempt="repair_continuation" if repair_conversation_id else "repair",
                    )
                    repaired_legacy_calls, _repaired_visible_text = parse_client_tool_calls(repaired_text, client_tools)
                    if repaired_action is None and repaired_legacy_calls:
                        repaired_action = {"action": "tool", **repaired_legacy_calls[0]}
                    if repaired_action is None and codex_tool_bridge.current_turn_has_tool_output(body.get("input")):
                        repaired_action = _plain_controller_final(repaired_text)
                    repaired_refusal = (
                        repaired_action is not None
                        and repaired_action.get("action") == "final"
                        and codex_tool_bridge.is_access_refusal(str(repaired_action.get("text") or ""))
                    )
                    if repaired_action is not None and not repaired_refusal:
                        action = repaired_action
                        full_text = repaired_text
                        request = repair_request
                        request_parent_before = repair_parent_before
                    elif force_tool:
                        action = codex_tool_bridge.bootstrap_local_action(client_tools)
                        request = repair_request
                        request_parent_before = repair_parent_before
                    else:
                        raise RuntimeError("Codex tool controller could not produce a valid action")
                if action is None:
                    raise RuntimeError("Codex tool controller returned no action")
                if force_tool and not codex_tool_bridge.is_local_executor_action(action):
                    action = codex_tool_bridge.bootstrap_local_action(client_tools)
                    if action is None:
                        raise RuntimeError("Codex request requires a local tool, but no local executor is available")
                visible_text = str(action.get("text") or "") if action.get("action") == "final" else ""
                if visible_text or action.get("action") == "final":
                    text_events, text_item = _text_output_events(visible_text, 0)
                    pending_events.extend(text_events)
                    output.append(text_item)
                if action.get("action") == "tool":
                    tool_events, tool_items = _client_tool_events([action], len(output))
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
                    client_tools,
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
        yield response_completed(response_id, model, created, output, usage)
        return
    request = ConversationRequest(model=model, messages=messages, thinking_effort=thinking_effort)
    yield {"type": "response.output_item.added", "output_index": 0, "item": text_output_item("", item_id, "in_progress")}
    for delta in stream_text_deltas(backend, request):
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
    yield response_completed(response_id, model, created, [item], usage)


def stream_web_search_response(body: dict[str, Any], messages: list[dict[str, Any]] | None = None) -> Iterator[dict[str, Any]]:
    model = str(body.get("model") or "auto").strip() or "auto"
    messages = messages if messages is not None else messages_from_input(body.get("input"), body.get("instructions"))
    query = search_query_from_messages(messages) or extract_response_prompt(body.get("input"))
    if not query:
        raise HTTPException(status_code=400, detail={"error": "input text is required for web_search"})

    response_id = f"resp_{uuid.uuid4().hex}"
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
    yield response_completed(response_id, model, created, [search_item, message_item], usage)


def stream_image_response(
    image_outputs: Iterable[ImageOutput],
    prompt: str,
    model: str,
    input_image_tokens: int = 0,
    size: object = None,
    quality: str = "auto",
) -> Iterator[dict[str, Any]]:
    response_id = f"resp_{uuid.uuid4().hex}"
    created = int(time.time())
    yield response_created(response_id, model, created)
    for output in image_outputs:
        if output.kind == "message":
            text = output.text
            item = text_output_item(text)
            usage = token_usage(
                input_text_tokens=count_text_tokens(prompt, model),
                input_image_tokens=input_image_tokens,
                output_text_tokens=count_text_tokens(text, model),
            )
            yield {"type": "response.output_text.delta", "item_id": item["id"], "output_index": 0, "content_index": 0, "delta": text}
            yield {"type": "response.output_text.done", "item_id": item["id"], "output_index": 0, "content_index": 0, "text": text}
            yield {"type": "response.output_item.done", "output_index": 0, "item": item}
            yield response_completed(response_id, model, created, [item], usage)
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
                yield {"type": "response.output_item.done", "output_index": output_index, "item": item}
            yield response_completed(response_id, model, created, items, usage)
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


def response_events(body: dict[str, Any]) -> Iterator[dict[str, Any]]:
    if is_text_response_request(body):
        model, messages = text_response_parts(body)
        if has_web_search_tool(body) and not has_unsupported_response_tools(body):
            yield from stream_web_search_response(body, messages)
            return
        if response_client_tools(body):
            yield from stream_text_response(text_backend(model), body, messages)
            return
        key = cache_key(body, messages, stream=bool(body.get("stream")))
        yield from chat_completion_cache.get_or_compute_stream(
            key,
            lambda: stream_text_response(text_backend(model), body, messages),
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
    yield from stream_image_response(image_outputs, prompt, model, input_image_tokens, tool.get("size"), str(tool.get("quality") or "auto"))


def handle(body: dict[str, Any]) -> dict[str, Any] | Iterator[dict[str, Any]]:
    events = response_events(body)
    if body.get("stream"):
        return events
    return collect_response(events)
