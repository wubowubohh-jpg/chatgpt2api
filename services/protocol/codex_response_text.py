from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any

from jsonschema import FormatChecker
from jsonschema import exceptions as jsonschema_exceptions
from jsonschema import validators
from referencing import Registry
from referencing.exceptions import NoSuchResource


_VERBOSITY_GUIDANCE = {
    "low": "Keep the final answer concise and include only essential information.",
    "medium": "Use a balanced amount of detail in the final answer.",
    "high": "Give a thorough final answer with the detail needed to support the result.",
}


def _reject_remote_schema(uri: str) -> object:
    raise NoSuchResource(ref=uri)


def normalized_text_controls(body: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the Codex Responses text controls in their canonical wire shape."""
    value = body.get("text")
    if not isinstance(value, Mapping):
        return None

    controls: dict[str, Any] = {}
    verbosity = str(value.get("verbosity") or "").strip().lower()
    if verbosity in _VERBOSITY_GUIDANCE:
        controls["verbosity"] = verbosity

    format_value = value.get("format")
    if isinstance(format_value, Mapping):
        format_type = str(format_value.get("type") or "").strip().lower()
        if format_type == "json_schema" and "schema" in format_value:
            controls["format"] = {
                "type": "json_schema",
                "strict": format_value.get("strict") is True,
                "schema": format_value.get("schema"),
                "name": str(format_value.get("name") or "codex_output_schema"),
            }

    return controls or None


def text_controls_signature(body: Mapping[str, Any]) -> str:
    return json.dumps(
        normalized_text_controls(body),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def controller_text_control_messages(
    body: Mapping[str, Any],
    *,
    include_clear: bool = False,
) -> list[dict[str, str]]:
    """Translate Responses text controls into instructions for the Web controller."""
    controls = normalized_text_controls(body)
    if controls is None:
        if not include_clear:
            return []
        return [{
            "role": "system",
            "content": (
                "CODEX_RESPONSE_TEXT_CONTROLS_UPDATE\n"
                "null\n"
                "These controls replace the previous response text controls. There is no current "
                "verbosity or structured-output constraint."
            ),
        }]

    guidance: list[str] = [
        "CODEX_RESPONSE_TEXT_CONTROLS",
        json.dumps(controls, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        "These controls apply to the final response text (or a final controller action's text field), "
        "not to tool arguments or intermediate tool results.",
    ]
    verbosity = controls.get("verbosity")
    if isinstance(verbosity, str):
        guidance.append(_VERBOSITY_GUIDANCE[verbosity])

    format_value = controls.get("format")
    if isinstance(format_value, dict):
        guidance.extend([
            "For the final controller action, the value of its text field MUST be one JSON document "
            "that validates against format.schema.",
            "Return no Markdown fence, preamble, commentary, or trailing text inside that text value.",
        ])
        if format_value.get("strict"):
            guidance.append(
                "strict=true: do not finish until the JSON value conforms to every schema constraint."
            )
    return [{"role": "system", "content": "\n".join(guidance)}]


def _resolve_local_ref(root: object, reference: str) -> object | None:
    if reference == "#":
        return root
    if not reference.startswith("#/"):
        return None
    current = root
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _matches_type(value: object, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
        ) or (
            isinstance(value, float)
            and math.isfinite(value)
        )
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def _schema_error(value: object, schema: object, root: object, path: str = "$") -> str | None:
    if not isinstance(schema, (Mapping, bool)):
        return f"{path} has an invalid schema"
    try:
        validator_type = validators.validator_for(schema)
        validator_type.check_schema(schema)
        registry = Registry(retrieve=_reject_remote_schema)
        validator = validator_type(
            schema,
            format_checker=FormatChecker(),
            registry=registry,
        )
        error = next(iter(validator.iter_errors(value)), None)
    except jsonschema_exceptions.SchemaError as exc:
        return f"{path} has an invalid schema: {exc.message}"
    except Exception as exc:  # referencing errors include unresolved remote refs
        return f"{path} schema validation failed: {type(exc).__name__}: {exc}"
    if error is None:
        return None
    location = path
    for part in error.absolute_path:
        location += f"[{part}]" if isinstance(part, int) else f".{part}"
    message = str(error.message or "schema constraint failed")
    message = message[:1].lower() + message[1:]
    return f"{location} {message}"


def validate_response_text(body: Mapping[str, Any], text: str) -> tuple[bool, str]:
    """Validate final text against Codex's JSON-schema text format, when present."""
    controls = normalized_text_controls(body)
    format_value = controls.get("format") if isinstance(controls, dict) else None
    if not isinstance(format_value, dict):
        return True, ""
    try:
        value = json.loads(text)
    except (TypeError, ValueError) as exc:
        return False, f"final text is not valid JSON: {exc}"
    schema = format_value.get("schema")
    error = _schema_error(value, schema, schema)
    return (error is None, error or "")


def validate_json_value(value: object, schema: object) -> tuple[bool, str]:
    error = _schema_error(value, schema, schema)
    return (error is None, error or "")
