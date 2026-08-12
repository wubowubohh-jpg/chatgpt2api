from __future__ import annotations

import re
from contextvars import ContextVar
from typing import Any

from fastapi import HTTPException

from services.account_service import account_service
from services.openai_backend_api import OpenAIBackendAPI

WEB_SEARCH_TOOL_TYPES = {"web_search", "web_search_preview", "web_search_preview_2025_03_11"}
SEARCH_CHAT_MODEL_PREFIXES = (
    "gpt-4o-search-preview",
    "gpt-4o-mini-search-preview",
    "gpt-5-search-api",
)

_WEB_SEARCH_TOOL_FIELDS = {
    "type",
    "external_web_access",
    "indexed_web_access",
    "filters",
    "user_location",
    "search_context_size",
    "search_content_types",
}
_WEB_SEARCH_LOCATION_FIELDS = {"type", "country", "region", "city", "timezone"}
_WEB_SEARCH_CONTEXT_SIZES = {"low", "medium", "high"}
_request_search_context_size: ContextVar[str | None] = ContextVar(
    "request_search_context_size",
    default=None,
)


class WebSearchConstraintError(HTTPException):
    """A deterministic Codex web-search constraint this backend cannot honor."""

    def __init__(self, message: str, param: str) -> None:
        self.message = message
        self.param = param
        self._openai_error = {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                # Codex treats invalid_prompt as fatal instead of retrying the
                # same request after a deterministic capability mismatch.
                "code": "invalid_prompt",
            }
        }
        super().__init__(status_code=400, detail=self._openai_error["error"])

    def __str__(self) -> str:
        return self.message

    def to_openai_error(self) -> dict[str, Any]:
        return self._openai_error


def _tool_type(tool: object) -> str:
    return str(tool.get("type") or "").strip() if isinstance(tool, dict) else ""


def _iter_web_search_tools(tools: object, path: str):
    if not isinstance(tools, list):
        return
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            continue
        tool_path = f"{path}[{index}]"
        tool_type = _tool_type(tool)
        if tool_type in WEB_SEARCH_TOOL_TYPES:
            yield tool, tool_path
        elif tool_type == "namespace":
            yield from _iter_web_search_tools(tool.get("tools"), f"{tool_path}.tools")


def _constraint_error(param: str, reason: str) -> WebSearchConstraintError:
    return WebSearchConstraintError(
        f"The ChatGPT Web compatibility search backend cannot honor {param}: {reason}.",
        param,
    )


def _require_optional_bool(tool: dict[str, Any], field: str, path: str) -> bool | None:
    if field not in tool:
        return None
    value = tool.get(field)
    if not isinstance(value, bool):
        raise _constraint_error(f"{path}.{field}", "the value must be a boolean")
    return value


def _validate_web_search_tool(
    tool: dict[str, Any],
    path: str,
    *,
    allow_unavailable_cached: bool = False,
) -> tuple[str | None, bool]:
    unknown_fields = sorted(set(tool) - _WEB_SEARCH_TOOL_FIELDS)
    if unknown_fields:
        field = unknown_fields[0]
        raise _constraint_error(
            f"{path}.{field}",
            "this web_search field is not supported and would otherwise be ignored",
        )

    external_web_access = _require_optional_bool(tool, "external_web_access", path)
    cached_search_unavailable = external_web_access is False
    if external_web_access is False:
        if not allow_unavailable_cached:
            raise _constraint_error(
                f"{path}.external_web_access",
                "false requests cached-only search, but the available upstream always performs live web search",
            )

    indexed_web_access = _require_optional_bool(tool, "indexed_web_access", path)
    if indexed_web_access is True:
        raise _constraint_error(
            f"{path}.indexed_web_access",
            "indexed-only live access is not exposed by the available upstream",
        )

    if "filters" in tool and tool.get("filters") is not None:
        filters = tool.get("filters")
        if not isinstance(filters, dict):
            raise _constraint_error(f"{path}.filters", "the value must be an object")
        unknown_filters = sorted(set(filters) - {"allowed_domains"})
        if unknown_filters:
            field = unknown_filters[0]
            raise _constraint_error(
                f"{path}.filters.{field}",
                "this filter is not supported and would otherwise be ignored",
            )
        if "allowed_domains" in filters:
            domains = filters.get("allowed_domains")
            if not isinstance(domains, list) or any(
                not isinstance(domain, str) or not domain.strip()
                for domain in domains
            ):
                raise _constraint_error(
                    f"{path}.filters.allowed_domains",
                    "the value must be a list of non-empty domain names",
                )
            raise _constraint_error(
                f"{path}.filters.allowed_domains",
                "the available upstream cannot enforce a source-domain allowlist before retrieval",
            )

    if "user_location" in tool and tool.get("user_location") is not None:
        location = tool.get("user_location")
        if not isinstance(location, dict):
            raise _constraint_error(f"{path}.user_location", "the value must be an object")
        unknown_location = sorted(set(location) - _WEB_SEARCH_LOCATION_FIELDS)
        if unknown_location:
            field = unknown_location[0]
            raise _constraint_error(
                f"{path}.user_location.{field}",
                "this location field is not supported and would otherwise be ignored",
            )
        location_type = location.get("type", "approximate")
        if location_type != "approximate":
            raise _constraint_error(
                f"{path}.user_location.type",
                "Codex only defines the approximate location type",
            )
        for field in ("country", "region", "city", "timezone"):
            value = location.get(field)
            if field in location and value is not None and not isinstance(value, str):
                raise _constraint_error(
                    f"{path}.user_location.{field}",
                    "the value must be a string",
                )
        raise _constraint_error(
            f"{path}.user_location",
            "the available upstream does not accept a per-request search location",
        )

    if "search_context_size" in tool and tool.get("search_context_size") is not None:
        context_size = tool.get("search_context_size")
        if context_size not in _WEB_SEARCH_CONTEXT_SIZES:
            raise _constraint_error(
                f"{path}.search_context_size",
                "the value must be low, medium, or high",
            )
    else:
        context_size = None

    if "search_content_types" in tool and tool.get("search_content_types") is not None:
        content_types = tool.get("search_content_types")
        if not isinstance(content_types, list) or not content_types or any(
            not isinstance(content_type, str) or not content_type.strip()
            for content_type in content_types
        ):
            raise _constraint_error(
                f"{path}.search_content_types",
                "the value must be a non-empty list of content-type names",
            )
        unsupported = sorted({value.strip() for value in content_types} - {"text"})
        if unsupported:
            raise _constraint_error(
                f"{path}.search_content_types",
                f"only text results are supported, not {', '.join(unsupported)}",
            )
    if cached_search_unavailable:
        # Codex includes cached search in ordinary `tool_choice:auto`
        # requests even when the user is not asking to search. This backend
        # has no cached index, so keep the optional capability unavailable
        # after validating its complete wire shape instead of failing the
        # unrelated turn or silently escalating it to live internet access.
        return context_size, False
    return context_size, True


def _merge_context_size(current: str | None, value: str | None, path: str) -> str | None:
    if current and value and current != value:
        raise _constraint_error(
            f"{path}.search_context_size",
            "multiple web_search tools requested conflicting context sizes",
        )
    return current or value


def _validate_chat_web_search_options(options: dict[str, Any]) -> str | None:
    unknown_fields = sorted(set(options) - {"search_context_size", "user_location"})
    if unknown_fields:
        field = unknown_fields[0]
        raise _constraint_error(
            f"web_search_options.{field}",
            "this web search option is not supported and would otherwise be ignored",
        )
    if options.get("user_location") is not None:
        raise _constraint_error(
            "web_search_options.user_location",
            "the available upstream does not accept a per-request search location",
        )
    context_size = options.get("search_context_size")
    if context_size is not None and context_size not in _WEB_SEARCH_CONTEXT_SIZES:
        raise _constraint_error(
            "web_search_options.search_context_size",
            "the value must be low, medium, or high",
        )
    return context_size


def _iter_tool_types(tools: object):
    if not isinstance(tools, list):
        return
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = _tool_type(tool)
        if tool_type == "namespace":
            yield from _iter_tool_types(tool.get("tools"))
        elif tool_type:
            yield tool_type


def has_web_search_tool(body: dict[str, Any]) -> bool:
    found = False
    context_size: str | None = None
    _request_search_context_size.set(None)
    tool_lists: list[object] = [body.get("tools")]
    definitions = list(_iter_web_search_tools(body.get("tools"), "tools"))
    # Responses Lite carries dynamic tools in an `additional_tools` input item
    # instead of the top-level `tools` field.
    input_value = body.get("input")
    if isinstance(input_value, list):
        for input_index, item in enumerate(input_value):
            if isinstance(item, dict) and _tool_type(item) == "additional_tools":
                tool_lists.append(item.get("tools"))
                definitions.extend(_iter_web_search_tools(
                    item.get("tools"),
                    f"input[{input_index}].tools",
                ))

    tool_choice = body.get("tool_choice")
    choice_type = _tool_type(tool_choice)
    if not choice_type and isinstance(tool_choice, str):
        choice_type = tool_choice.strip()
    explicitly_selected = choice_type in WEB_SEARCH_TOOL_TYPES
    required_only_search = (
        choice_type == "required"
        and bool(definitions)
        and not any(
            tool_type not in WEB_SEARCH_TOOL_TYPES
            for tools in tool_lists
            for tool_type in _iter_tool_types(tools)
        )
    )
    force_search = explicitly_selected or required_only_search

    for tool, path in definitions:
        value, executable = _validate_web_search_tool(
            tool,
            path,
            allow_unavailable_cached=not force_search,
        )
        if not executable:
            continue
        context_size = _merge_context_size(
            context_size,
            value,
            path,
        )
        found = True
    if found:
        _request_search_context_size.set(context_size)
        return True
    if explicitly_selected:
        choice_context_size, _executable = _validate_web_search_tool(
            tool_choice,
            "tool_choice",
        )
        _request_search_context_size.set(
            choice_context_size
        )
        return True
    return False


def is_web_search_chat_request(body: dict[str, Any]) -> bool:
    model = str(body.get("model") or "").strip()
    has_tool = has_web_search_tool(body)
    options = body.get("web_search_options")
    if isinstance(options, dict):
        option_context_size = _validate_chat_web_search_options(options)
        if not has_tool:
            _request_search_context_size.set(option_context_size)
    return has_tool or isinstance(options, dict) or any(
        model == prefix or model.startswith(f"{prefix}-")
        for prefix in SEARCH_CHAT_MODEL_PREFIXES
    )


def has_unsupported_tools(body: dict[str, Any], allowed_types: set[str]) -> bool:
    tool_lists: list[object] = [body.get("tools")]
    input_value = body.get("input")
    if isinstance(input_value, list):
        tool_lists.extend(
            item.get("tools")
            for item in input_value
            if isinstance(item, dict) and _tool_type(item) == "additional_tools"
        )
    return any(
        tool_type not in allowed_types
        for tools in tool_lists
        for tool_type in _iter_tool_types(tools)
    )


def message_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = str(item.get("text") or item.get("input_text") or "").strip()
            else:
                text = ""
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return ""


def search_query_from_messages(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if str(message.get("role") or "").strip().lower() != "user":
            continue
        text = message_text(message.get("content"))
        if text:
            return text
    return ""


def _readable_annotation_part(parts: list[str]) -> str:
    for part in parts:
        value = part.strip()
        lower = value.lower()
        if value and not (
            lower.startswith(("turn", "source", "sources"))
            or re.fullmatch(r"\d+", value)
        ):
            return value
    return ""


def clean_search_text(text: str) -> str:
    def replace_annotation(match: re.Match[str]) -> str:
        parts = [part.strip() for part in match.group(1).split("\ue202")]
        kind = (parts[0] if parts else "").lower()
        data = parts[1:]
        if kind == "url":
            label = data[0] if data else ""
            url = data[1] if len(data) > 1 else ""
            if label and url.startswith(("http://", "https://")):
                return f"{label} ({url})"
            return label or url
        if kind == "cite":
            return _readable_annotation_part(data)
        return _readable_annotation_part(data)

    text = re.sub(r"\ue200([^\ue201]*)\ue201", replace_annotation, text)
    text = re.sub(r"\ue200[^\ue201]*$", "", text)
    return re.sub(r"\s+([.,;:!?])", r"\1", text).strip()


def normalized_sources(result: dict[str, Any]) -> list[dict[str, str]]:
    sources = result.get("sources")
    if not isinstance(sources, list):
        return []
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in sources:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        output.append({"title": title, "url": url, "snippet": snippet})
    return output


def text_with_url_citations(result: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    text = clean_search_text(str(result.get("answer") or ""))
    annotations: list[dict[str, Any]] = []
    sources = normalized_sources(result)
    if sources:
        text = text.rstrip()
        if text:
            text += "\n\n"
        text += "Sources:\n"
        for index, source in enumerate(sources, start=1):
            title = source["title"] or source["url"]
            line_prefix = f"{index}. {title}"
            text += line_prefix
            if source["url"]:
                if source["title"]:
                    text += " - "
                start = len(text)
                text += source["url"]
                annotations.append({
                    "type": "url_citation",
                    "start_index": start,
                    "end_index": len(text),
                    "url": source["url"],
                    "title": source["title"] or source["url"],
                })
            text += "\n"
    return text.strip(), annotations


def run_web_search(query: str) -> dict[str, Any]:
    token = account_service.get_text_access_token()
    context_size = _request_search_context_size.get()
    prompt = query
    if context_size:
        prompt = (
            f"{query}\n\n"
            f"Codex search_context_size={context_size}. "
            "Use that requested amount of relevant source context while keeping source URLs."
        )
    result = OpenAIBackendAPI(token).search(prompt)
    account_service.mark_text_used(token)
    return result
