from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from lark import Lark
from lark.exceptions import GrammarError, UnexpectedInput


MAX_GRAMMAR_BYTES = 256 * 1024
MAX_CUSTOM_INPUT_BYTES = 2 * 1024 * 1024
MAX_JS_SAFE_INTEGER = 9_007_199_254_740_991
EXEC_PRAGMA_PREFIX = "// @exec:"


def validate_exec_source(value: str) -> tuple[bool, str]:
    """Validate the first-line exec pragma exactly as Codex's Rust parser does."""
    source = str(value or "")
    if not source.strip():
        return False, "exec expects non-empty JavaScript source"
    first_line, separator, rest = source.partition("\n")
    if not first_line.lstrip().startswith(EXEC_PRAGMA_PREFIX):
        return True, ""
    if not separator or not rest.strip():
        return False, "exec pragma must be followed by JavaScript source on subsequent lines"
    directive = first_line.lstrip()[len(EXEC_PRAGMA_PREFIX):].strip()
    if not directive:
        return False, "exec pragma must be a JSON object"
    try:
        parsed = json.loads(directive)
    except json.JSONDecodeError as exc:
        return False, f"exec pragma must be valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return False, "exec pragma must be a JSON object"
    unknown = sorted(set(parsed) - {"yield_time_ms", "max_output_tokens"})
    if unknown:
        return False, f"exec pragma only supports yield_time_ms and max_output_tokens; got {unknown[0]}"
    for key in ("yield_time_ms", "max_output_tokens"):
        if key not in parsed:
            continue
        number = parsed[key]
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            return False, f"exec pragma field {key} must be a non-negative safe integer"
        if number > MAX_JS_SAFE_INTEGER:
            return False, f"exec pragma field {key} must be a non-negative safe integer"
    return True, ""


def _python_lark_definition(definition: str) -> str:
    """Normalize the zero-width line token used by Codex's apply_patch grammar."""
    return definition.replace("/(.+)/", r"/[^\r\n]+/").replace(
        "/(.*)/",
        r"/[^\r\n]+/?",
    )


@lru_cache(maxsize=128)
def _compile_lark(definition: str) -> Lark:
    normalized = _python_lark_definition(definition)
    try:
        return Lark(normalized, parser="lalr", lexer="contextual", start="start")
    except GrammarError:
        return Lark(normalized, parser="earley", lexer="dynamic", start="start")


def validate_custom_tool_input(tool: dict[str, Any], value: str) -> tuple[bool, str]:
    tool_format = tool.get("format")
    if not isinstance(tool_format, dict):
        return True, ""
    format_type = str(tool_format.get("type") or "").strip().lower()
    if format_type != "grammar":
        return False, f"unsupported custom tool format type: {format_type or '<empty>'}"
    syntax = str(tool_format.get("syntax") or "").strip().lower()
    if syntax != "lark":
        return False, f"unsupported custom tool grammar syntax: {syntax or '<empty>'}"
    definition = str(tool_format.get("definition") or "")
    if not definition.strip():
        return False, "custom tool Lark grammar definition is empty"
    if len(definition.encode("utf-8")) > MAX_GRAMMAR_BYTES:
        return False, "custom tool Lark grammar exceeds the 256 KiB safety limit"
    if len(value.encode("utf-8")) > MAX_CUSTOM_INPUT_BYTES:
        return False, "custom tool input exceeds the 2 MiB validation limit"
    if str(tool.get("name") or "").strip() == "exec":
        valid, error = validate_exec_source(value)
        if not valid:
            return False, error
    try:
        parser = _compile_lark(definition)
    except (GrammarError, ValueError) as exc:
        return False, f"invalid custom tool Lark grammar: {exc}"
    try:
        parser.parse(value)
    except UnexpectedInput as exc:
        return False, f"custom tool input does not match its Lark grammar at offset {exc.pos_in_stream}"
    return True, ""
