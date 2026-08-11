from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Iterator

from services.account_service import account_service
from services.model_service import normalize_model_identifier
from services.protocol import codex_tool_bridge


SESSION_TTL_SECONDS = 2 * 60 * 60
SESSION_MAX_ENTRIES = 512


@dataclass(frozen=True)
class ContinuationPlan:
    key: str = ""
    generation: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    conversation_id: str = ""
    parent_message_id: str = ""
    access_token: str = ""
    continued: bool = False
    delta_input_items: int = 0
    replayed: bool = False
    output_items: list[dict[str, Any]] = field(default_factory=list)
    response_id: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    canonical_input_items: list[Any] = field(default_factory=list)
    prior_output_items: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _Session:
    generation: int
    model: str
    instructions_signature: str
    tools: list[dict[str, Any]]
    input_items: list[Any]
    output_items: list[dict[str, Any]]
    conversation_id: str
    parent_message_id: str
    access_token: str
    response_id: str
    usage: dict[str, Any]
    session_id: str
    turn_id: str
    last_request_signature: str
    updated_at: float = field(default_factory=time.monotonic)


def _json_signature(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _model_signature(value: object) -> str:
    return normalize_model_identifier(str(value or "auto").strip() or "auto")


def _client_metadata(body: dict[str, Any]) -> dict[str, Any]:
    value = body.get("client_metadata")
    return value if isinstance(value, dict) else {}


def _previous_response_id(body: dict[str, Any]) -> str:
    return str(body.get("previous_response_id") or "").strip()


def _request_signature(body: dict[str, Any]) -> str:
    """Identify a retried Responses request without including gateway metadata."""
    return _json_signature({
        "model": body.get("model"),
        "instructions": body.get("instructions"),
        "input": body.get("input"),
        "tools": body.get("tools"),
        "tool_choice": body.get("tool_choice"),
        "previous_response_id": _previous_response_id(body),
    })


def _session_key(body: dict[str, Any]) -> str:
    metadata = _client_metadata(body)
    thread_id = str(metadata.get("thread_id") or "").strip()
    prompt_cache_key = str(body.get("prompt_cache_key") or "").strip()
    session_seed = f"thread:{thread_id}" if thread_id else f"cache:{prompt_cache_key}"
    if session_seed == "cache:":
        return ""
    identity = str(body.get("_request_identity_key_id") or "anonymous").strip() or "anonymous"
    return hashlib.sha256(f"{identity}\0{session_seed}".encode("utf-8")).hexdigest()


def _tool_key(tool: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(tool.get("namespace") or ""),
        str(tool.get("name") or ""),
        str(tool.get("kind") or ""),
    )


def _changed_tools(previous: list[dict[str, Any]], current: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previous_by_key = {_tool_key(tool): _json_signature(tool) for tool in previous}
    return [
        tool
        for tool in current
        if previous_by_key.get(_tool_key(tool)) != _json_signature(tool)
    ]


def _same_output_item(current: object, expected: object) -> bool:
    if current == expected:
        return True
    if not isinstance(current, dict) or not isinstance(expected, dict):
        return False
    if str(current.get("type") or "") != str(expected.get("type") or ""):
        return False
    current_id = str(current.get("id") or "")
    expected_id = str(expected.get("id") or "")
    if current_id and expected_id and current_id == expected_id:
        return True
    current_call_id = str(current.get("call_id") or "")
    expected_call_id = str(expected.get("call_id") or "")
    if current_call_id and expected_call_id and current_call_id == expected_call_id:
        return True
    if str(current.get("type") or "") == "message":
        return (
            str(current.get("role") or "") == "assistant"
            and str(expected.get("role") or "") == "assistant"
            and current.get("content") == expected.get("content")
        )
    return False


def _same_input_item(current: object, expected: object) -> bool:
    if current == expected:
        return True
    if not isinstance(current, dict) or not isinstance(expected, dict):
        return False
    current_type = str(current.get("type") or "message")
    expected_type = str(expected.get("type") or "message")
    if current_type != expected_type:
        return False
    if current_type in codex_tool_bridge.TOOL_CALL_TYPES | codex_tool_bridge.TOOL_OUTPUT_TYPES:
        return _same_output_item(current, expected)
    return False


def _prefix_length(current: list[Any], expected: list[Any]) -> int | None:
    if len(current) < len(expected):
        return None
    if all(_same_input_item(current[index], item) for index, item in enumerate(expected)):
        return len(expected)
    return None


def _longest_history_overlap(history: list[Any], current: list[Any]) -> int:
    """Return the number of current items already present at history's tail."""
    maximum = min(len(history), len(current))
    for size in range(maximum, 0, -1):
        if all(
            _same_input_item(history[len(history) - size + index], current[index])
            for index in range(size)
        ):
            return size
    return 0


def _tool_call_seed(items: list[Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("call_id") or ""): item
        for item in items
        if isinstance(item, dict)
        and str(item.get("type") or "") in codex_tool_bridge.TOOL_CALL_TYPES
        and str(item.get("call_id") or "")
    }


def _drop_replayed_output_items(
    suffix: list[Any],
    output_items: list[dict[str, Any]],
) -> tuple[list[Any], list[dict[str, Any]]]:
    offset = 0
    dropped: list[dict[str, Any]] = []
    for expected in output_items:
        if offset >= len(suffix) or not _same_output_item(suffix[offset], expected):
            break
        if isinstance(expected, dict):
            dropped.append(expected)
        offset += 1
    return suffix[offset:], dropped


class _SessionStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._sessions: OrderedDict[str, _Session] = OrderedDict()
        self._key_locks: dict[str, RLock] = {}
        self._response_keys: dict[str, str] = {}

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()
            self._key_locks.clear()
            self._response_keys.clear()

    def _lookup_key(self, body: dict[str, Any]) -> str:
        previous_response_id = _previous_response_id(body)
        with self._lock:
            response_key = self._response_keys.get(previous_response_id, "")
        return response_key or _session_key(body)

    @contextmanager
    def session_lock(self, body: dict[str, Any]) -> Iterator[None]:
        key = self._lookup_key(body)
        if not key:
            yield
            return
        with self._lock:
            lock = self._key_locks.setdefault(key, RLock())
        with lock:
            yield

    def prepare(
        self,
        body: dict[str, Any],
        tools: list[dict[str, Any]],
        full_messages: list[dict[str, Any]],
        *,
        force_tool: bool,
    ) -> ContinuationPlan:
        key = self._lookup_key(body)
        input_value = body.get("input")
        if not key or not isinstance(input_value, list):
            return ContinuationPlan(
                key=key,
                messages=full_messages,
                canonical_input_items=copy.deepcopy(input_value) if isinstance(input_value, list) else [],
            )

        now = time.monotonic()
        with self._lock:
            self._expire_locked(now)
            session = self._sessions.get(key)
            if session is None:
                return ContinuationPlan(
                    key=key,
                    messages=full_messages,
                    canonical_input_items=copy.deepcopy(input_value),
                )
            metadata = _client_metadata(body)
            session_id = str(metadata.get("session_id") or "").strip()
            turn_id = str(metadata.get("turn_id") or "").strip()
            identity_mismatch = (
                session.model != _model_signature(body.get("model"))
                or session.instructions_signature != _json_signature(body.get("instructions"))
                or (session.session_id and session_id and session.session_id != session_id)
            )
            if identity_mismatch:
                self._sessions.pop(key, None)
                return ContinuationPlan(
                    key=key,
                    messages=full_messages,
                    canonical_input_items=copy.deepcopy(input_value),
                )

            request_signature = _request_signature(body)
            if session.output_items and session.last_request_signature == request_signature:
                return ContinuationPlan(
                    key=key,
                    generation=session.generation,
                    continued=True,
                    replayed=True,
                    output_items=copy.deepcopy(session.output_items),
                    response_id=session.response_id,
                    usage=copy.deepcopy(session.usage),
                    canonical_input_items=copy.deepcopy(session.input_items),
                    prior_output_items=copy.deepcopy(session.output_items),
                )

            previous_response_id = _previous_response_id(body)
            prefix_length = _prefix_length(input_value, session.input_items)
            delta_allowed = (
                bool(previous_response_id and previous_response_id == session.response_id)
                or bool(session.session_id and session_id and session.session_id == session_id)
            )
            if prefix_length is None and not delta_allowed:
                self._sessions.pop(key, None)
                return ContinuationPlan(
                    key=key,
                    messages=full_messages,
                    canonical_input_items=copy.deepcopy(input_value),
                )
            if prefix_length is not None:
                suffix = copy.deepcopy(input_value[prefix_length:])
                canonical_input_items = copy.deepcopy(input_value)
            elif previous_response_id and previous_response_id == session.response_id:
                # Codex may send only the new tool call/result items and link them
                # to the prior response instead of replaying the entire input.
                overlap = _longest_history_overlap(session.input_items, input_value)
                suffix = copy.deepcopy(input_value[overlap:])
                canonical_input_items = copy.deepcopy(session.input_items) + copy.deepcopy(input_value[overlap:])
            elif session.session_id and session_id and session.session_id == session_id:
                # Some Codex builds omit previous_response_id but keep a stable
                # client session. Treat the request as a delta rather than
                # dropping the task context when its input is shorter.
                overlap = _longest_history_overlap(session.input_items, input_value)
                suffix = copy.deepcopy(input_value[overlap:])
                canonical_input_items = copy.deepcopy(session.input_items) + copy.deepcopy(input_value[overlap:])
            else:
                self._sessions.pop(key, None)
                return ContinuationPlan(
                    key=key,
                    messages=full_messages,
                    canonical_input_items=copy.deepcopy(input_value),
                )

            suffix, replayed_items = _drop_replayed_output_items(suffix, session.output_items)
            if not suffix:
                if (not turn_id or turn_id == session.turn_id) and session.output_items:
                    return ContinuationPlan(
                        key=key,
                        generation=session.generation,
                        continued=True,
                        replayed=True,
                        output_items=copy.deepcopy(session.output_items),
                        response_id=session.response_id,
                        usage=copy.deepcopy(session.usage),
                        canonical_input_items=canonical_input_items,
                        prior_output_items=copy.deepcopy(session.output_items),
                    )
                self._sessions.pop(key, None)
                return ContinuationPlan(
                    key=key,
                    messages=full_messages,
                    canonical_input_items=canonical_input_items,
                )

            previous_tool_keys = {_tool_key(tool) for tool in session.tools}
            current_tool_keys = {_tool_key(tool) for tool in tools}
            if not previous_tool_keys.issubset(current_tool_keys):
                self._sessions.pop(key, None)
                return ContinuationPlan(key=key, messages=full_messages)
            changed_tools = _changed_tools(session.tools, tools)
            messages = codex_tool_bridge.controller_tool_messages(changed_tools)
            messages.extend(codex_tool_bridge.controller_task_contract_messages(canonical_input_items))
            messages.extend(codex_tool_bridge.controller_task_anchor_messages(canonical_input_items))
            seed_calls = _tool_call_seed(session.output_items)
            seed_calls.update(_tool_call_seed(replayed_items))
            messages.extend(codex_tool_bridge.controller_transcript_messages({"input": suffix}, seed_calls=seed_calls))
            messages.extend(codex_tool_bridge.controller_turn_state_messages(force_tool))
            if not messages:
                return ContinuationPlan(key=key, messages=full_messages)

            access_token = account_service.resolve_access_token(session.access_token)
            if not access_token or account_service.get_account(access_token) is None:
                self._sessions.pop(key, None)
                return ContinuationPlan(key=key, messages=full_messages)
            session.updated_at = now
            self._sessions.move_to_end(key)
            return ContinuationPlan(
                key=key,
                generation=session.generation,
                messages=messages,
                conversation_id=session.conversation_id,
                parent_message_id=session.parent_message_id,
                access_token=access_token,
                continued=True,
                delta_input_items=len(suffix),
                canonical_input_items=canonical_input_items,
                prior_output_items=copy.deepcopy(session.output_items),
            )

    def commit(
        self,
        plan: ContinuationPlan,
        body: dict[str, Any],
        tools: list[dict[str, Any]],
        output_items: list[dict[str, Any]],
        *,
        conversation_id: str,
        parent_message_id: str,
        access_token: str,
        response_id: str,
        usage: dict[str, Any],
    ) -> bool:
        input_value = body.get("input")
        if not isinstance(input_value, list) or not conversation_id or not parent_message_id or not access_token:
            return False
        key = plan.key or _session_key(body) or (f"response:{response_id}" if response_id else "")
        if not key:
            return False
        canonical_input_items = (
            copy.deepcopy(plan.canonical_input_items)
            if plan.canonical_input_items
            else copy.deepcopy(input_value)
        )
        now = time.monotonic()
        metadata = _client_metadata(body)
        with self._lock:
            self._expire_locked(now)
            current = self._sessions.get(key)
            if plan.continued and (current is None or current.generation != plan.generation):
                return False
            generation = (current.generation + 1) if current is not None else 1
            self._sessions[key] = _Session(
                generation=generation,
                model=_model_signature(body.get("model")),
                instructions_signature=_json_signature(body.get("instructions")),
                tools=copy.deepcopy(tools),
                input_items=canonical_input_items,
                output_items=copy.deepcopy(output_items),
                conversation_id=conversation_id,
                parent_message_id=parent_message_id,
                access_token=access_token,
                response_id=response_id,
                usage=copy.deepcopy(usage),
                session_id=str(metadata.get("session_id") or "").strip(),
                turn_id=str(metadata.get("turn_id") or "").strip(),
                last_request_signature=_request_signature(body),
                updated_at=now,
            )
            self._sessions.move_to_end(key)
            if response_id:
                self._response_keys[response_id] = key
            while len(self._sessions) > SESSION_MAX_ENTRIES:
                removed_key, _removed = self._sessions.popitem(last=False)
                self._response_keys = {
                    response: mapped_key
                    for response, mapped_key in self._response_keys.items()
                    if mapped_key != removed_key
                }
            return True

    def invalidate(self, plan: ContinuationPlan) -> None:
        if not plan.key:
            return
        with self._lock:
            current = self._sessions.get(plan.key)
            if current is not None and (
                not plan.continued or current.generation == plan.generation
            ):
                self._sessions.pop(plan.key, None)
                self._response_keys = {
                    response: mapped_key
                    for response, mapped_key in self._response_keys.items()
                    if mapped_key != plan.key
                }

    def _expire_locked(self, now: float) -> None:
        expired = [
            key
            for key, session in self._sessions.items()
            if now - session.updated_at > SESSION_TTL_SECONDS
        ]
        for key in expired:
            self._sessions.pop(key, None)


_sessions = _SessionStore()


def prepare_controller_request(
    body: dict[str, Any],
    tools: list[dict[str, Any]],
    full_messages: list[dict[str, Any]],
    *,
    force_tool: bool,
) -> ContinuationPlan:
    return _sessions.prepare(body, tools, full_messages, force_tool=force_tool)


def commit_controller_response(
    plan: ContinuationPlan,
    body: dict[str, Any],
    tools: list[dict[str, Any]],
    output_items: list[dict[str, Any]],
    *,
    conversation_id: str,
    parent_message_id: str,
    access_token: str,
    response_id: str,
    usage: dict[str, Any],
) -> bool:
    return _sessions.commit(
        plan,
        body,
        tools,
        output_items,
        conversation_id=conversation_id,
        parent_message_id=parent_message_id,
        access_token=access_token,
        response_id=response_id,
        usage=usage,
    )


def invalidate_controller_session(plan: ContinuationPlan) -> None:
    _sessions.invalidate(plan)


def clear_controller_sessions() -> None:
    _sessions.clear()


def controller_session_lock(body: dict[str, Any]):
    return _sessions.session_lock(body)
