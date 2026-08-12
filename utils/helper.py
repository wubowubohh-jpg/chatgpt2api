import base64
import hashlib
import json
import mimetypes
import os
import queue
import re
import threading
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

from curl_cffi import requests
from fastapi import HTTPException
from services.proxy_service import proxy_settings
from utils.log import logger

BASE_IMAGE_MODELS = {"gpt-image-2", "codex-gpt-image-2"}
IMAGE_MODEL_PLAN_TYPES = ("plus", "team", "pro")
CODEX_IMAGE_MODEL = "codex-gpt-image-2"
PREFIXED_CODEX_IMAGE_MODELS = {
    f"{plan_type}-{CODEX_IMAGE_MODEL}"
    for plan_type in IMAGE_MODEL_PLAN_TYPES
}
IMAGE_MODELS = BASE_IMAGE_MODELS | PREFIXED_CODEX_IMAGE_MODELS
PUBLIC_IMAGE_MODELS = BASE_IMAGE_MODELS | PREFIXED_CODEX_IMAGE_MODELS
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

SUPPORTED_JSON_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}
MAX_JSON_IMAGE_BYTES = 10 * 1024 * 1024
MAX_JSON_EDIT_IMAGES = 10
DATA_URL_IMAGE_RE = re.compile(r"^data:(?P<mime>[-+./\w]+);base64,(?P<data>.*)$", re.DOTALL)
REMOTE_IMAGE_TIMEOUT_SECONDS = 20


def _image_extension(mime_type: str) -> str:
    image_type = mime_type.split("/", 1)[1].split(";", 1)[0].lower() if "/" in mime_type else "png"
    return "jpg" if image_type == "jpeg" else image_type or "png"


def _decode_json_image_string(value: str, index: int, filename: str | None = None, mime_type: str | None = None) -> tuple[bytes, str, str]:
    text = value.strip()
    if not text:
        raise HTTPException(status_code=400, detail={"error": "image file is empty"})
    match = DATA_URL_IMAGE_RE.match(text)
    if match:
        resolved_mime = (match.group("mime") or "image/png").lower()
        encoded = match.group("data")
    else:
        if text.startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail={"error": "remote image URLs are not supported"})
        resolved_mime = (mime_type or "image/png").lower()
        encoded = text
    if resolved_mime == "image/jpg":
        resolved_mime = "image/jpeg"
    if resolved_mime not in SUPPORTED_JSON_IMAGE_MIME_TYPES:
        raise HTTPException(status_code=400, detail={"error": "unsupported image mime type"})
    try:
        image_data = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"error": "invalid base64 image data"}) from exc
    if not image_data:
        raise HTTPException(status_code=400, detail={"error": "image file is empty"})
    if len(image_data) > MAX_JSON_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail={"error": "image file is too large"})
    return image_data, filename or f"image_{index}.{_image_extension(resolved_mime)}", resolved_mime


def _extract_json_image_value(item: object) -> tuple[str, str | None, str | None]:
    if isinstance(item, str):
        return item, None, None
    if not isinstance(item, dict):
        raise HTTPException(status_code=400, detail={"error": "image entry must be a base64 string or object"})
    filename = str(item.get("filename") or item.get("file_name") or "").strip() or None
    mime_type = str(item.get("mime_type") or item.get("mimeType") or "").strip() or None
    value = item.get("b64_json") or item.get("base64")
    if not value:
        image_url = item.get("image_url") or item.get("url")
        if isinstance(image_url, dict):
            filename = filename or str(image_url.get("filename") or image_url.get("file_name") or "").strip() or None
            mime_type = mime_type or str(image_url.get("mime_type") or image_url.get("mimeType") or "").strip() or None
            value = image_url.get("url") or image_url.get("image_url")
        else:
            value = image_url
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(status_code=400, detail={"error": "image entry must include image data"})
    return value, filename, mime_type


def normalize_json_edit_images(image: object = None, images: object = None) -> list[tuple[bytes, str, str]]:
    raw_images = images if images is not None else image
    if raw_images is None:
        raise HTTPException(status_code=400, detail={"error": "image file is required"})
    entries = raw_images if isinstance(raw_images, list) else [raw_images]
    if not entries:
        raise HTTPException(status_code=400, detail={"error": "image file is required"})
    if len(entries) > MAX_JSON_EDIT_IMAGES:
        raise HTTPException(status_code=400, detail={"error": f"images supports up to {MAX_JSON_EDIT_IMAGES} items"})
    normalized = []
    for index, item in enumerate(entries, start=1):
        value, filename, mime_type = _extract_json_image_value(item)
        normalized.append(_decode_json_image_string(value, index, filename, mime_type))
    return normalized


def new_uuid() -> str:
    return str(uuid.uuid4())


def split_image_model(model: object) -> tuple[str | None, str | None]:
    normalized = str(model or "").strip().lower()
    if not normalized:
        return None, None
    if normalized in BASE_IMAGE_MODELS:
        return None, normalized
    for plan_type in IMAGE_MODEL_PLAN_TYPES:
        prefix = f"{plan_type}-"
        if normalized.startswith(prefix):
            base_model = normalized[len(prefix):]
            if base_model == CODEX_IMAGE_MODEL:
                return plan_type, base_model
    return None, None


def is_supported_image_model(model: object) -> bool:
    _, base_model = split_image_model(model)
    return base_model is not None


def is_codex_image_model(model: object) -> bool:
    _, base_model = split_image_model(model)
    return base_model == CODEX_IMAGE_MODEL


def is_image_chat_request(body: dict[str, object]) -> bool:
    model = str(body.get("model") or "").strip()
    modalities = body.get("modalities")
    if is_supported_image_model(model):
        return True
    return isinstance(modalities, list) and "image" in {str(item or "").strip().lower() for item in modalities}


_UPSTREAM_BODY_LOG_LIMIT = 500


class UpstreamHTTPError(RuntimeError):
    """Raised when an upstream HTTP call returns a non-2xx status.

    Carries structured fields (status_code, body, retry_after) so callers can
    branch on status code instead of string-matching on str(exc). The full
    body is preserved on the instance; the formatted message truncates it
    to keep log lines reasonable.
    """

    def __init__(
        self,
        context: str,
        status_code: int,
        body: Any,
        retry_after: int | None = None,
    ) -> None:
        self.context = context
        self.status_code = status_code
        self.body = body
        self.retry_after = retry_after
        if isinstance(body, (dict, list)):
            try:
                body_str = json.dumps(body, ensure_ascii=False)
            except (TypeError, ValueError):
                body_str = repr(body)
        else:
            body_str = str(body)
        if len(body_str) > _UPSTREAM_BODY_LOG_LIMIT:
            body_str = body_str[:_UPSTREAM_BODY_LOG_LIMIT] + "…[truncated]"
        super().__init__(f"{context} failed: status={status_code}, body={body_str}")

    def to_openai_error(self) -> dict[str, Any]:
        """Expose structured upstream errors to OpenAI/Responses clients.

        Codex classifies ``response.failed`` by ``error.code``.  Collapsing
        every upstream failure into ``upstream_error`` makes fatal context and
        quota errors look retryable and can trigger repeated identical turns.
        """
        candidate: Any = self.body
        if isinstance(candidate, str):
            try:
                decoded = json.loads(candidate)
            except (TypeError, ValueError):
                decoded = None
            if decoded is not None:
                candidate = decoded
        if isinstance(candidate, dict) and isinstance(candidate.get("error"), dict):
            candidate = candidate["error"]
        if not isinstance(candidate, dict):
            candidate = {}

        status = int(self.status_code or 0)
        code = str(candidate.get("code") or "").strip()
        error_type = str(candidate.get("type") or "").strip()
        message = str(candidate.get("message") or candidate.get("detail") or "").strip()
        if not code:
            code = {
                400: "invalid_request_error",
                401: "authentication_error",
                403: "permission_denied",
                404: "not_found",
                413: "context_length_exceeded",
                429: "rate_limit_exceeded",
            }.get(status, "server_error" if status >= 500 else "upstream_error")
        if not error_type:
            if code == "context_length_exceeded":
                error_type = "invalid_request_error"
            elif code == "insufficient_quota":
                error_type = "insufficient_quota"
            elif code == "rate_limit_exceeded":
                error_type = "rate_limit_error"
            elif status >= 500:
                error_type = "server_error"
            elif status == 401:
                error_type = "authentication_error"
            elif status == 403:
                error_type = "permission_error"
            else:
                error_type = "invalid_request_error"
        if not message:
            message = str(self) or "upstream request failed"
        if self.retry_after is not None and code == "rate_limit_exceeded":
            if "try again" not in message.lower():
                message = f"{message.rstrip('.')} Try again in {self.retry_after} seconds."
        return {
            "error": {
                "message": message,
                "type": error_type,
                "param": candidate.get("param"),
                "code": code,
            }
        }


def ensure_ok(response: requests.Response, context: str) -> None:
    if 200 <= response.status_code < 300:
        return
    body: Any = response.text
    try:
        body = response.json()
    except Exception:
        pass
    retry_after_header = response.headers.get("Retry-After") if hasattr(response, "headers") else None
    retry_after: int | None = None
    if retry_after_header is not None:
        ra_str = str(retry_after_header).strip()
        if ra_str.isdigit():
            retry_after = int(ra_str)
    raise UpstreamHTTPError(context, response.status_code, body, retry_after=retry_after)


def sse_json_stream(items) -> Iterator[str]:
    yield ": stream-open\n\n"
    try:
        for item in items:
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
    except Exception as exc:
        logger.warning({
            "event": "sse_stream_error",
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        })
        error = exc.to_openai_error() if hasattr(exc, "to_openai_error") else {
            "error": {"message": str(exc), "type": exc.__class__.__name__}
        }
        yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


class StreamCancellation:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: dict[int, Callable[[], object]] = {}
        self._next_id = 1

    def register(self, callback: Callable[[], object]) -> int:
        with self._lock:
            if self._event.is_set():
                registration = 0
            else:
                registration = self._next_id
                self._next_id += 1
                self._callbacks[registration] = callback
        if registration == 0:
            try:
                callback()
            except Exception:
                pass
        return registration

    def unregister(self, registration: int) -> None:
        if registration:
            with self._lock:
                self._callbacks.pop(registration, None)

    def cancel(self) -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
            callbacks = list(self._callbacks.values())
            self._callbacks.clear()
        for callback in callbacks:
            try:
                callback()
            except Exception:
                pass


_CURRENT_STREAM_CANCELLATION: ContextVar[StreamCancellation | None] = ContextVar(
    "chatgpt2api_stream_cancellation",
    default=None,
)


def current_stream_cancellation() -> StreamCancellation | None:
    return _CURRENT_STREAM_CANCELLATION.get()


def responses_sse_stream(items) -> Iterator[str]:
    yield ": stream-open\n\n"
    response: dict[str, Any] = {}
    terminal = False

    # Codex wraps every SSE read in an idle timeout.  A comment-only heartbeat
    # is consumed by the SSE framing layer and does not reset that timer, so
    # run the potentially blocking compatibility generator in a worker and
    # emit a parser-visible Responses event while it waits on upstream work.
    try:
        heartbeat_seconds = float(os.getenv("CHATGPT2API_RESPONSES_SSE_HEARTBEAT_SECONDS", "5"))
    except (TypeError, ValueError):
        heartbeat_seconds = 5.0
    heartbeat_seconds = min(max(heartbeat_seconds, 0.05), 60.0)
    events: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=8)
    producer_stop = threading.Event()
    cancellation = StreamCancellation()

    def put_event(kind: str, value: object) -> bool:
        while not producer_stop.is_set():
            try:
                events.put((kind, value), timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        cancellation_token = _CURRENT_STREAM_CANCELLATION.set(cancellation)
        try:
            for value in items:
                if not put_event("item", value):
                    return
        except Exception as exc:  # noqa: BLE001 - surfaced by the consumer below
            put_event("error", exc)
        finally:
            put_event("done", None)
            _CURRENT_STREAM_CANCELLATION.reset(cancellation_token)

    producer = threading.Thread(
        target=produce,
        name="responses-sse-source",
        daemon=True,
    )
    producer.start()
    try:
        while True:
            try:
                kind, value = events.get(timeout=heartbeat_seconds)
            except queue.Empty:
                if not terminal:
                    heartbeat = {
                        "type": "response.in_progress",
                    }
                    yield f"data: {json.dumps(heartbeat, ensure_ascii=False)}\n\n"
                continue
            if kind == "done":
                break
            if kind == "error":
                if isinstance(value, Exception):
                    raise value
                raise RuntimeError(str(value))
            item = value
            if isinstance(item, dict):
                event_type = str(item.get("type") or "")
                if event_type == "response.created" and isinstance(item.get("response"), dict):
                    response = dict(item["response"])
                if event_type in {"response.completed", "response.failed", "response.incomplete"}:
                    terminal = True
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            if terminal:
                break
    except Exception as exc:
        logger.warning({
            "event": "responses_sse_stream_error",
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        })
        if not terminal:
            public_error = exc.to_openai_error() if hasattr(exc, "to_openai_error") else {}
            error_value = public_error.get("error") if isinstance(public_error, dict) else {}
            if not isinstance(error_value, dict):
                error_value = {}
            message = str(error_value.get("message") or str(exc) or "response generation failed")
            code = str(error_value.get("code") or "upstream_error")
            error_type = str(error_value.get("type") or "server_error")
            if isinstance(exc, UpstreamHTTPError) and exc.status_code == 413:
                code = "context_length_exceeded"
                error_type = "invalid_request_error"
                message = (
                    "The Codex compatibility request is larger than the ChatGPT Web conversation "
                    "endpoint accepts. Start a new turn or reduce non-instruction tool output."
                )
            elif isinstance(exc, UpstreamHTTPError) and exc.status_code in {400, 401, 403, 404, 422}:
                codex_fatal_codes = {
                    "context_length_exceeded",
                    "insufficient_quota",
                    "usage_not_included",
                    "cyber_policy",
                    "invalid_prompt",
                    "bio_policy",
                }
                if code not in codex_fatal_codes:
                    # Codex treats unknown response.failed codes as retryable.
                    # These status codes are deterministic at this gateway, so
                    # expose the fatal code Codex recognizes and keep the real
                    # upstream message for diagnosis.
                    code = "invalid_prompt"
                    error_type = "invalid_request_error"
            failed_response = {
                "id": str(response.get("id") or f"resp_{uuid.uuid4().hex}"),
                "object": "response",
                "created_at": int(response.get("created_at") or time.time()),
                "status": "failed",
                "error": {
                    "type": error_type,
                    "code": code,
                    "message": message,
                },
                "incomplete_details": None,
                "model": response.get("model"),
                "output": [],
            }
            event = {"type": "response.failed", "response": failed_response}
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            terminal = True
    finally:
        producer_stop.set()
        cancellation.cancel()
        producer.join(timeout=1.0)
        if not producer.is_alive():
            close = getattr(items, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
    if not terminal:
        failed_response = {
            "id": str(response.get("id") or f"resp_{uuid.uuid4().hex}"),
            "object": "response",
            "created_at": int(response.get("created_at") or time.time()),
            "status": "failed",
            "error": {
                "type": "server_error",
                "code": "stream_ended_without_terminal_event",
                "message": "response stream ended without a terminal event",
            },
            "incomplete_details": None,
            "model": response.get("model"),
            "output": [],
        }
        event = {"type": "response.failed", "response": failed_response}
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def anthropic_sse_stream(items) -> Iterator[str]:
    try:
        for item in items:
            event = str(item.get("type") or "message_delta") if isinstance(item, dict) else "message_delta"
            yield f"event: {event}\n"
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
    except Exception as exc:
        logger.warning({
            "event": "anthropic_sse_stream_error",
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        })
        error = {"type": "error", "error": {"type": exc.__class__.__name__, "message": str(exc)}}
        yield "event: error\n"
        yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"


def iter_sse_payloads(response: requests.Response) -> Iterator[str]:
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, bytes) else str(raw_line)
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload:
            yield payload


def save_images_from_text(text: str, prefix: str) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matches = re.findall(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", text or "")
    saved_paths: list[Path] = []
    timestamp = int(time.time() * 1000)
    for index, data_url in enumerate(matches, start=1):
        header, encoded = data_url.split(",", 1)
        image_type = header.split(";")[0].removeprefix("data:image/").strip() or "png"
        extension = "jpg" if image_type == "jpeg" else image_type
        output_path = OUTPUT_DIR / f"{prefix}_{timestamp}_{index}.{extension}"
        output_path.write_bytes(base64.b64decode(encoded))
        saved_paths.append(output_path)
    return saved_paths


def anonymize_token(token: object) -> str:
    value = str(token or "").strip()
    if not value:
        return "token:empty"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"token:{digest}"


def extract_response_prompt(input_value: object) -> str:
    if isinstance(input_value, str):
        return input_value.strip()
    if isinstance(input_value, dict):
        role = str(input_value.get("role") or "").strip().lower()
        if role and role != "user":
            return ""
        return extract_prompt_from_message_content(input_value.get("content"))
    if not isinstance(input_value, list):
        return ""
    prompt_parts: list[str] = []
    for item in input_value:
        if isinstance(item, dict) and str(item.get("type") or "").strip() == "input_text":
            text = str(item.get("text") or "").strip()
            if text:
                prompt_parts.append(text)
            continue
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role and role != "user":
            continue
        prompt = extract_prompt_from_message_content(item.get("content"))
        if prompt:
            prompt_parts.append(prompt)
    return "\n".join(prompt_parts).strip()


def has_response_image_generation_tool(body: dict[str, object]) -> bool:
    tools = body.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict) and str(tool.get("type") or "").strip() == "image_generation":
                return True
    tool_choice = body.get("tool_choice")
    return isinstance(tool_choice, dict) and str(tool_choice.get("type") or "").strip() == "image_generation"


def extract_prompt_from_message_content(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip()
        if item_type == "text":
            text = str(item.get("text") or "").strip()
            if text:
                parts.append(text)
        elif item_type == "input_text":
            text = str(item.get("text") or item.get("input_text") or "").strip()
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _message_image_url(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("url") or value.get("image_url") or "").strip()
    return str(value or "").strip()


def _decode_message_image_url(value: object) -> tuple[bytes, str] | None:
    source = _message_image_url(value)
    if source.startswith("data:"):
        header, _, data = source.partition(",")
        mime = header.split(";")[0].removeprefix("data:") or "image/png"
        return base64.b64decode(data), mime
    if not source.startswith(("http://", "https://")):
        return None
    parsed = urlparse(source)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None

    try:
        response = requests.get(
            source,
            headers={"Accept": "image/*,*/*;q=0.8", "User-Agent": "chatgpt2api vision fetcher"},
            timeout=REMOTE_IMAGE_TIMEOUT_SECONDS,
            allow_redirects=True,
            **proxy_settings.build_session_kwargs(),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"error": f"image_url fetch failed: {exc}"}) from exc
    if not 200 <= response.status_code < 300:
        raise HTTPException(status_code=400, detail={"error": f"image_url fetch failed: HTTP {response.status_code}"})
    content_length = str(response.headers.get("content-length") or "").strip()
    if content_length.isdigit() and int(content_length) > MAX_JSON_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail={"error": "image_url exceeds 10MB limit"})
    image_data = response.content
    if not image_data:
        raise HTTPException(status_code=400, detail={"error": "image_url returned empty content"})
    if len(image_data) > MAX_JSON_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail={"error": "image_url exceeds 10MB limit"})
    mime = str(response.headers.get("content-type") or "image/png").split(";", 1)[0].lower()
    guessed_mime = mimetypes.guess_type(parsed.path)[0] or ""
    if mime and not mime.startswith("image/") and mime not in {"application/octet-stream", "binary/octet-stream"}:
        raise HTTPException(status_code=400, detail={"error": "image_url must point to an image"})
    if not mime.startswith("image/") and guessed_mime.startswith("image/"):
        mime = guessed_mime
    if not mime.startswith("image/"):
        mime = "image/png"
    return image_data, mime


def _decode_message_image_object(item: dict[str, object]) -> tuple[bytes, str] | None:
    data = item.get("data")
    if isinstance(data, (bytes, bytearray)):
        return bytes(data), str(item.get("mime") or item.get("mime_type") or "image/png")
    for key in ("image_url", "url"):
        image = _decode_message_image_url(item.get(key))
        if image:
            return image
    value = item.get("b64_json") or item.get("base64")
    if isinstance(value, str) and value.strip():
        image_data, _, mime = _decode_json_image_string(
            value,
            1,
            mime_type=str(item.get("mime") or item.get("mime_type") or item.get("mimeType") or "image/png"),
        )
        return image_data, mime
    source = item.get("source")
    if isinstance(source, dict) and str(source.get("type") or "") == "base64":
        encoded = str(source.get("data") or "")
        mime = str(source.get("media_type") or source.get("mime_type") or "image/png")
        image_data, _, resolved_mime = _decode_json_image_string(encoded, 1, mime_type=mime)
        return image_data, resolved_mime
    return None


def extract_image_from_message_content(content: object) -> list[tuple[bytes, str]]:
    if not isinstance(content, list):
        return []
    images = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip()
        if item_type == "image_url":
            image = _decode_message_image_url(item.get("image_url") or item.get("url") or item)
            if image:
                images.append(image)
        elif item_type in {"input_image", "image"}:
            image = _decode_message_image_object(item)
            if image:
                images.append(image)
    return images


def extract_chat_image(body: dict[str, object]) -> list[tuple[bytes, str]]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return []
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "").strip().lower() != "user":
            continue
        images = extract_image_from_message_content(message.get("content"))
        if images:
            return images
    return []


def extract_chat_prompt(body: dict[str, object]) -> str:
    direct_prompt = str(body.get("prompt") or "").strip()
    if direct_prompt:
        return direct_prompt
    messages = body.get("messages")
    if not isinstance(messages, list):
        return ""
    prompt_parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "").strip().lower() != "user":
            continue
        prompt = extract_prompt_from_message_content(message.get("content"))
        if prompt:
            prompt_parts.append(prompt)
    return "\n".join(prompt_parts).strip()


def parse_image_count(raw_value: object) -> int:
    try:
        value = int(raw_value or 1)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail={"error": "n must be an integer"}) from exc
    if value < 1 or value > 4:
        raise HTTPException(status_code=400, detail={"error": "n must be between 1 and 4"})
    return value


def build_chat_image_markdown_content(image_result: dict[str, object]) -> str:
    image_items = image_result.get("data") if isinstance(image_result.get("data"), list) else []
    markdown_images: list[str] = []
    for index, item in enumerate(image_items, start=1):
        if not isinstance(item, dict):
            continue
        b64_json = str(item.get("b64_json") or "").strip()
        if b64_json:
            markdown_images.append(f"![image_{index}](data:image/png;base64,{b64_json})")
    return "\n\n".join(markdown_images) if markdown_images else "Image generation completed."
