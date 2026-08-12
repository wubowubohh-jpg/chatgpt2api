from __future__ import annotations

import copy
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any
from urllib.parse import urlparse

from services.protocol import openai_search
from services.protocol.web_search_tool import WebSearchConstraintError, clean_search_text
from utils.helper import extract_response_prompt


SEARCH_COMMAND_KEYS = (
    "search_query",
    "image_query",
    "open",
    "click",
    "find",
    "screenshot",
    "finance",
    "weather",
    "sports",
    "time",
)
REFERENCE_COMMAND_KEYS = ("open", "find")
UNSUPPORTED_COMMAND_KEYS = ("click", "screenshot")
SEARCH_SETTING_FIELDS = {
    "user_location",
    "search_context_size",
    "filters",
    "image_settings",
    "allowed_callers",
    "external_web_access",
}
SEARCH_CONTEXT_SIZES = {"low", "medium", "high"}
SEARCH_ALLOWED_CALLERS = {"direct", "shell", "code_interpreter"}
SEARCH_SESSION_TTL_SECONDS = 30 * 60
SEARCH_SESSION_MAX_ENTRIES = 256


@dataclass
class _SearchSession:
    references: dict[str, str] = field(default_factory=dict)
    next_turn: int = 0
    updated_at: float = field(default_factory=time.monotonic)


class _SearchSessionStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: OrderedDict[str, _SearchSession] = OrderedDict()

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    def begin(self, session_id: str, commands: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
        resolved = copy.deepcopy(commands)
        now = time.monotonic()
        with self._lock:
            self._expire_locked(now)
            session = self._sessions.get(session_id) if session_id else None
            references = session.references if session is not None else {}
            for key in REFERENCE_COMMAND_KEYS:
                for operation in resolved.get(key) or []:
                    ref_id = str(operation.get("ref_id") or "").strip()
                    if not ref_id or _is_url(ref_id):
                        continue
                    url = references.get(ref_id)
                    if not url:
                        return resolved, "", f"Unknown search reference {ref_id!r}. Run search_query first or pass a URL."
                    operation["ref_id"] = url

            if not session_id:
                return resolved, "turn0", ""
            if session is None:
                session = _SearchSession(updated_at=now)
                self._sessions[session_id] = session
            turn_prefix = f"turn{session.next_turn}"
            session.next_turn += 1
            session.updated_at = now
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > SEARCH_SESSION_MAX_ENTRIES:
                self._sessions.popitem(last=False)
            return resolved, turn_prefix, ""

    def register(self, session_id: str, results: list[dict[str, Any]]) -> None:
        if not session_id:
            return
        now = time.monotonic()
        with self._lock:
            self._expire_locked(now)
            session = self._sessions.get(session_id)
            if session is None:
                session = _SearchSession(updated_at=now)
                self._sessions[session_id] = session
            for result in results:
                ref_id = str(result.get("ref_id") or "").strip()
                url = str(result.get("url") or "").strip()
                if ref_id and _is_url(url):
                    session.references[ref_id] = url
            session.updated_at = now
            self._sessions.move_to_end(session_id)

    def _expire_locked(self, now: float) -> None:
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.updated_at > SEARCH_SESSION_TTL_SECONDS
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)


_search_sessions = _SearchSessionStore()


def clear_search_sessions() -> None:
    _search_sessions.clear()


def _is_url(value: str) -> bool:
    parsed = urlparse(str(value or "").strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _meaningful_commands(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    commands: dict[str, Any] = {}
    for key in SEARCH_COMMAND_KEYS:
        items = value.get(key)
        if not isinstance(items, list):
            continue
        kept = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if key in {"search_query", "image_query"} and not str(item.get("q") or "").strip():
                continue
            kept.append(item)
        if kept:
            commands[key] = kept
    response_length = str(value.get("response_length") or "").strip().lower()
    if response_length in {"short", "medium", "long"}:
        commands["response_length"] = response_length
    return commands


def _constraint_error(param: str, reason: str) -> WebSearchConstraintError:
    return WebSearchConstraintError(
        f"The ChatGPT Web compatibility search backend cannot honor {param}: {reason}.",
        param,
    )


def _validate_search_constraints(body: dict[str, Any]) -> str:
    commands = body.get("commands")
    if isinstance(commands, dict):
        image_queries = commands.get("image_query")
        if isinstance(image_queries, list) and image_queries:
            raise _constraint_error(
                "commands.image_query",
                "the available upstream returns text and URL sources, not Codex image search results",
            )
        search_queries = commands.get("search_query")
        if isinstance(search_queries, list):
            for index, query in enumerate(search_queries):
                if not isinstance(query, dict) or "domains" not in query:
                    continue
                domains = query.get("domains")
                if domains in (None, []):
                    continue
                raise _constraint_error(
                    f"commands.search_query[{index}].domains",
                    "the available upstream cannot enforce a source-domain allowlist before retrieval",
                )

    settings = body.get("settings")
    if settings is None:
        return ""
    if not isinstance(settings, dict):
        raise _constraint_error("settings", "the value must be an object")
    unknown_fields = sorted(set(settings) - SEARCH_SETTING_FIELDS)
    if unknown_fields:
        raise _constraint_error(
            f"settings.{unknown_fields[0]}",
            "this field is not supported and would otherwise be ignored",
        )

    external_access = settings.get("external_web_access")
    if external_access not in (None, True, "live"):
        if external_access not in (False, "cached", "indexed"):
            raise _constraint_error(
                "settings.external_web_access",
                "the value must be true, false, live, cached, or indexed",
            )
        raise _constraint_error(
            "settings.external_web_access",
            f"{external_access!r} requires a restricted retrieval mode, but the available upstream always performs live web search",
        )

    allowed_callers = settings.get("allowed_callers")
    if allowed_callers is not None:
        if not isinstance(allowed_callers, list) or any(
            not isinstance(caller, str) or caller not in SEARCH_ALLOWED_CALLERS
            for caller in allowed_callers
        ):
            raise _constraint_error(
                "settings.allowed_callers",
                "the value must contain only direct, shell, or code_interpreter",
            )

    context_size = settings.get("search_context_size")
    if context_size is not None and context_size not in SEARCH_CONTEXT_SIZES:
        raise _constraint_error(
            "settings.search_context_size",
            "the value must be low, medium, or high",
        )

    for field, reason in (
        ("filters", "domain allowlists and blocklists cannot be enforced before retrieval"),
        ("user_location", "the available upstream does not accept a per-request search location"),
        ("image_settings", "the available upstream does not expose Codex image search results"),
    ):
        if settings.get(field) not in (None, {}, []):
            raise _constraint_error(f"settings.{field}", reason)
    return str(context_size or "")


def request_prompt(body: dict[str, Any]) -> str:
    """Build the ChatGPT Web search prompt for Codex's alpha/search request."""
    context_size = _validate_search_constraints(body)
    raw_commands = body.get("commands")
    commands = _meaningful_commands(raw_commands)
    if isinstance(raw_commands, dict) and not any(key in commands for key in SEARCH_COMMAND_KEYS):
        return ""

    input_value = body.get("input")
    context = extract_response_prompt(input_value)
    if not commands:
        return context

    parts = [
        "Execute the following Codex web search commands. Return the requested result with source URLs.",
        "Commands JSON:\n" + json.dumps(commands, ensure_ascii=False, separators=(",", ":")),
    ]
    if context:
        parts.append(
            "Recent request context (use it only to resolve references such as turn0search0):\n" + context
        )
    if context_size:
        parts.append(
            "Search context size requested by Codex: "
            f"{context_size}. Keep the result breadth and detail consistent with that level."
        )
    if isinstance(input_value, (dict, list)):
        parts.append(
            "Recent request context JSON:\n"
            + json.dumps(input_value, ensure_ascii=False, separators=(",", ":"))
        )
    settings = body.get("settings")
    if isinstance(settings, dict) and settings:
        parts.append("Search settings:\n" + json.dumps(settings, ensure_ascii=False, separators=(",", ":")))
    return "\n\n".join(parts)


def _normalized_sources(result: dict[str, Any]) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    raw_sources = result.get("sources")
    if not isinstance(raw_sources, list):
        return sources
    for raw in raw_sources:
        if not isinstance(raw, dict):
            continue
        url = str(raw.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        sources.append({
            "title": str(raw.get("title") or "").strip(),
            "url": url,
            "snippet": str(raw.get("snippet") or "").strip(),
        })
    return sources


def response_from_result(
    result: dict[str, Any],
    max_output_tokens: object = None,
    ref_prefix: str = "turn0",
) -> dict[str, Any]:
    answer = clean_search_text(str(result.get("answer") or "")).strip()
    sources = _normalized_sources(result)
    lines = [answer] if answer else []
    if sources:
        lines.extend(["", "Sources:"])
        for index, source in enumerate(sources):
            title = source["title"] or source["url"]
            lines.append(f"[{ref_prefix}search{index}] {title}")
            lines.append(source["url"])
            if source["snippet"]:
                lines.append(source["snippet"])
    output = "\n".join(lines).strip()

    try:
        token_limit = int(max_output_tokens) if max_output_tokens is not None else 0
    except (TypeError, ValueError):
        token_limit = 0
    if token_limit > 0:
        char_limit = token_limit * 4
        if len(output) > char_limit:
            output = output[:char_limit].rstrip() + "\n[output truncated]"

    results = [
        {
            "type": "text_result",
            "ref_id": f"{ref_prefix}search{index}",
            "url": source["url"],
            "title": source["title"],
            "snippet": source["snippet"],
        }
        for index, source in enumerate(sources)
    ]
    return {
        "encrypted_output": None,
        "output": output,
        "results": results,
    }


def handle(body: dict[str, Any], owner_id: str = "") -> dict[str, Any]:
    raw_commands = body.get("commands")
    commands = _meaningful_commands(raw_commands)
    if isinstance(raw_commands, dict) and not any(key in commands for key in SEARCH_COMMAND_KEYS):
        return {"encrypted_output": None, "output": "", "results": []}
    for key in UNSUPPORTED_COMMAND_KEYS:
        if commands.get(key):
            return {
                "encrypted_output": None,
                "output": f"The compatibility search backend does not support the {key} operation.",
                "results": [],
            }
    request_id = str(body.get("id") or "").strip()
    owner = str(owner_id or "").strip()
    session_id = f"{owner}\x1f{request_id}" if owner and request_id else request_id
    resolved_commands, ref_prefix, reference_error = _search_sessions.begin(session_id, commands)
    if reference_error:
        return {"encrypted_output": None, "output": reference_error, "results": []}
    resolved_body = dict(body)
    if isinstance(body.get("commands"), dict):
        resolved_body["commands"] = resolved_commands
    prompt = request_prompt(resolved_body)
    if not prompt:
        return {"encrypted_output": None, "output": "", "results": []}
    result = openai_search.handle({"prompt": prompt})
    response = response_from_result(result, body.get("max_output_tokens"), ref_prefix=ref_prefix or "turn0")
    _search_sessions.register(session_id, response["results"])
    if result.get("_account_email"):
        response["_account_email"] = result["_account_email"]
    return response
