from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Header, HTTPException, Request, WebSocket
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from api.image_inputs import parse_image_edit_request, read_image_sources
from api.support import require_identity, resolve_image_base_url
from services.content_filter import check_request, request_shape, request_text
from services.editable_file_task_service import editable_file_task_service
from services.log_service import LoggedCall
from services.protocol import (
    anthropic_v1_messages,
    codex_search,
    openai_v1_chat_complete,
    openai_v1_image_edit,
    openai_v1_image_generations,
    openai_v1_models,
    openai_v1_response,
    codex_memories,
    codex_files,
    openai_search,
)


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str = "gpt-image-2"
    n: int = Field(default=1, ge=1, le=4)
    size: str | None = None
    quality: str = "auto"
    response_format: str = "b64_json"
    history_disabled: bool = True
    stream: bool | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    prompt: str | None = None
    n: int | None = None
    stream: bool | None = None
    modalities: list[str] | None = None
    messages: list[dict[str, object]] | None = None


class ResponseCreateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    input: object | None = None
    tools: list[dict[str, object]] | None = None
    tool_choice: object | None = None
    stream: bool | None = None


class AnthropicMessageRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    messages: list[dict[str, object]] | None = None
    system: object | None = None
    stream: bool | None = None


class SearchRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


class CodexSearchRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str = ""
    model: str = "auto"
    reasoning: object | None = None
    input: object | None = None
    commands: dict[str, object] | None = None
    settings: dict[str, object] | None = None
    max_output_tokens: int | None = Field(default=None, ge=0)


class EditableFileTaskRequest(BaseModel):
    prompt: str = ""
    base64_images: list[str] = Field(default_factory=list)
    client_task_id: str | None = None


def merge_responses_request_identity(payload: dict[str, object], request: Request) -> None:
    """Keep Codex session identity when a proxy sends it only as HTTP headers."""
    metadata_value = payload.get("client_metadata")
    metadata = dict(metadata_value) if isinstance(metadata_value, dict) else {}

    nested_values = [metadata.get("x-codex-turn-metadata"), request.headers.get("x-codex-turn-metadata")]
    for nested in nested_values:
        if not isinstance(nested, str):
            continue
        try:
            nested_value = json.loads(nested)
        except (TypeError, ValueError):
            nested_value = None
        if isinstance(nested_value, dict):
            for key, value in nested_value.items():
                if key not in metadata and isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)

    header_aliases = {
        "session-id": "session_id",
        "thread-id": "thread_id",
        "x-client-request-id": "thread_id",
        "x-codex-installation-id": "installation_id",
        "x-codex-routing-hint": "routing_hint",
        "x-codex-window-id": "window_id",
        "x-codex-turn-id": "turn_id",
        "x-codex-turn-state": "turn_state",
        "x-codex-parent-thread-id": "parent_thread_id",
        "x-openai-subagent": "subagent_header",
    }
    for header_name, metadata_name in header_aliases.items():
        value = str(request.headers.get(header_name) or "").strip()
        if value and not str(metadata.get(metadata_name) or "").strip():
            metadata[metadata_name] = value

    session_id = str(metadata.get("session_id") or "").strip()
    if session_id and not str(payload.get("prompt_cache_key") or "").strip():
        payload["prompt_cache_key"] = session_id
    if metadata:
        payload["client_metadata"] = metadata


async def filter_or_log(call: LoggedCall, text: str) -> None:
    try:
        await run_in_threadpool(check_request, text)
    except HTTPException as exc:
        call.log("调用失败", status="failed", error=str(exc.detail))
        raise


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/models")
    @router.get("/v1/models")
    async def list_models(
            request: Request,
            authorization: str | None = Header(default=None),
    ):
        require_identity(authorization)
        try:
            # Codex appends client_version and expects ModelsResponse, whose
            # wire key is ``models``. Keep the normal OpenAI ``data`` catalog
            # for all other clients.
            if request.query_params.get("client_version") is not None:
                payload = await run_in_threadpool(openai_v1_models.list_codex_models)
                return JSONResponse(
                    content=payload,
                    headers={"etag": '"chatgpt2api-codex-models-v1"'},
                )
            return await run_in_threadpool(openai_v1_models.list_models)
        except Exception as exc:
            raise HTTPException(status_code=502, detail={"error": str(exc)}) from exc

    @router.post("/v1/images/generations")
    async def generate_images(
            body: ImageGenerationRequest,
            request: Request,
            authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        payload = body.model_dump(mode="python")
        payload["base_url"] = resolve_image_base_url(request)
        call = LoggedCall(identity, "/v1/images/generations", body.model, "文生图", request_text=body.prompt)
        await filter_or_log(call, body.prompt)
        return await call.run(openai_v1_image_generations.handle, payload)

    @router.post("/v1/images/edits")
    async def edit_images(
            request: Request,
            authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        payload, image_sources, mask_sources = await parse_image_edit_request(request)
        prompt = str(payload["prompt"])
        model = str(payload["model"])
        call = LoggedCall(identity, "/v1/images/edits", model, "图生图", request_text=prompt)
        await filter_or_log(call, prompt)
        payload["images"] = await read_image_sources(image_sources)
        if mask_sources:
            payload["mask"] = await read_image_sources(mask_sources)
        payload["base_url"] = resolve_image_base_url(request)
        return await call.run(openai_v1_image_edit.handle, payload)

    @router.post("/v1/chat/completions")
    async def create_chat_completion(body: ChatCompletionRequest, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        payload = body.model_dump(mode="python")
        model = str(payload.get("model") or "auto")
        request_preview = request_text(payload.get("prompt"), payload.get("messages"))
        call = LoggedCall(
            identity,
            "/v1/chat/completions",
            model,
            "文本生成",
            request_text=request_preview,
            request_shape=request_shape(payload.get("messages")),
        )
        await filter_or_log(call, request_preview)
        return await call.run(openai_v1_chat_complete.handle, payload)

    @router.post("/responses")
    @router.post("/v1/responses")
    async def create_response(
            body: ResponseCreateRequest,
            request: Request,
            authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        payload = body.model_dump(mode="python")
        merge_responses_request_identity(payload, request)
        payload["_response_id"] = f"resp_{uuid.uuid4().hex}"
        # Keep controller-session state isolated when clients reuse a prompt cache key.
        payload["_request_identity_key_id"] = str(identity.get("id") or "anonymous")
        model = str(payload.get("model") or "auto")
        request_preview = request_text(payload.get("input"), payload.get("instructions"))
        call = LoggedCall(
            identity,
            "/v1/responses",
            model,
            "Responses",
            request_text=request_preview,
            request_shape=request_shape(payload.get("input")),
        )
        await filter_or_log(call, request_preview)
        return await call.run(openai_v1_response.handle, payload, sse="responses")

    @router.post("/responses/compact")
    @router.post("/v1/responses/compact")
    async def compact_response(
            body: ResponseCreateRequest,
            request: Request,
            authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        payload = body.model_dump(mode="python")
        merge_responses_request_identity(payload, request)
        payload["_request_identity_key_id"] = str(identity.get("id") or "anonymous")
        model = str(payload.get("model") or "auto")
        request_preview = request_text(payload.get("input"), payload.get("instructions"))
        call = LoggedCall(
            identity,
            "/v1/responses/compact",
            model,
            "Responses compact",
            request_text=request_preview,
            request_shape=request_shape(payload.get("input")),
        )
        await filter_or_log(call, request_preview)
        result = await call.run(openai_v1_response.compact, payload)
        metadata = payload.get("client_metadata") if isinstance(payload.get("client_metadata"), dict) else {}
        turn_state = str(metadata.get("turn_state") or "").strip()
        if isinstance(result, dict) and turn_state:
            return JSONResponse(content=result, headers={"x-codex-turn-state": turn_state})
        return result

    @router.post("/memories/trace_summarize")
    @router.post("/v1/memories/trace_summarize")
    async def summarize_memories(
            body: codex_memories.MemorySummarizeRequest,
            authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        payload = body.model_dump(mode="python")
        model = str(payload.get("model") or "auto")
        request_preview = request_text(payload.get("traces"), payload.get("reasoning"))
        call = LoggedCall(
            identity,
            "/v1/memories/trace_summarize",
            model,
            "Codex memory summarize",
            request_text=request_preview,
        )
        await filter_or_log(call, request_preview)
        return await call.run(codex_memories.handle, payload)

    @router.post("/files")
    @router.post("/v1/files")
    async def create_codex_file(
            payload: dict[str, object],
            request: Request,
            authorization: str | None = Header(default=None),
    ):
        require_identity(authorization)
        try:
            record = codex_files.codex_file_store.create(
                payload.get("file_name"),
                payload.get("file_size"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
        base_url = resolve_image_base_url(request).rstrip("/")
        return {
            "file_id": record.file_id,
            "upload_url": f"{base_url}/v1/files/{record.file_id}/upload/{record.upload_token}",
        }

    @router.put("/files/{file_id}/upload/{upload_token}")
    @router.put("/v1/files/{file_id}/upload/{upload_token}")
    async def upload_codex_file(file_id: str, upload_token: str, request: Request):
        record = codex_files.codex_file_store.get_for_upload(file_id, upload_token)
        if record is None:
            raise HTTPException(status_code=404, detail={"error": "file upload target not found"})
        declared_length = request.headers.get("content-length")
        if declared_length:
            try:
                if int(declared_length) > record.file_size:
                    codex_files.codex_file_store.discard_upload(record)
                    raise HTTPException(status_code=413, detail={"error": "file exceeds declared size"})
            except ValueError:
                raise HTTPException(status_code=400, detail={"error": "invalid content-length"})
        total = 0
        try:
            with record.path.open("wb") as output:
                async for chunk in request.stream():
                    total += len(chunk)
                    if total > record.file_size:
                        raise HTTPException(status_code=413, detail={"error": "file exceeds declared size"})
                    output.write(chunk)
        except HTTPException:
            codex_files.codex_file_store.discard_upload(record)
            raise
        except OSError as exc:
            codex_files.codex_file_store.discard_upload(record)
            raise HTTPException(status_code=500, detail={"error": "file upload could not be stored"}) from exc
        return Response(status_code=201)

    @router.post("/files/{file_id}/uploaded")
    @router.post("/v1/files/{file_id}/uploaded")
    async def finalize_codex_file(
            file_id: str,
            request: Request,
            authorization: str | None = Header(default=None),
    ):
        require_identity(authorization)
        record = codex_files.codex_file_store.get(file_id)
        if record is None:
            raise HTTPException(status_code=404, detail={"error": "file not found"})
        if record.uploaded:
            status = "success"
        else:
            try:
                uploaded_bytes = record.path.stat().st_size
            except OSError:
                return {"status": "retry"}
            record, error = codex_files.codex_file_store.mark_uploaded(file_id, uploaded_bytes)
            if error:
                return {"status": "error", "error_message": error}
            status = "success"
        base_url = resolve_image_base_url(request).rstrip("/")
        return {
            "status": status,
            "download_url": f"{base_url}/v1/files/{record.file_id}/download/{record.download_token}",
            "file_name": record.file_name,
            "mime_type": record.mime_type,
        }

    @router.get("/files/{file_id}/download/{download_token}")
    @router.get("/v1/files/{file_id}/download/{download_token}")
    async def download_codex_file(file_id: str, download_token: str):
        record = codex_files.codex_file_store.get_for_download(file_id, download_token)
        if record is None or not record.path.is_file():
            raise HTTPException(status_code=404, detail={"error": "file not found"})
        return FileResponse(record.path, media_type=record.mime_type, filename=record.file_name)

    @router.websocket("/responses")
    @router.websocket("/v1/responses")
    async def reject_responses_websocket(websocket: WebSocket):
        """Tell Codex to use its HTTP Responses transport instead of a missing WS bridge."""
        denial = Response(
            status_code=426,
            headers={"connection": "Upgrade", "upgrade": "websocket"},
        )
        try:
            await websocket.send_denial_response(denial)
        except (AttributeError, RuntimeError):
            # Older ASGI servers do not expose the denial-response extension.
            await websocket.close(code=1000, reason="Responses WebSocket is not supported")

    @router.post("/v1/messages")
    async def create_message(
            body: AnthropicMessageRequest,
            authorization: str | None = Header(default=None),
            x_api_key: str | None = Header(default=None, alias="x-api-key"),
            anthropic_version: str | None = Header(default=None, alias="anthropic-version"),
    ):
        identity = require_identity(authorization or (f"Bearer {x_api_key}" if x_api_key else None))
        payload = body.model_dump(mode="python")
        model = str(payload.get("model") or "auto")
        request_preview = request_text(payload.get("system"), payload.get("messages"), payload.get("tools"))
        call = LoggedCall(identity, "/v1/messages", model, "Messages", request_text=request_preview)
        await filter_or_log(call, request_preview)
        return await call.run(anthropic_v1_messages.handle, payload, sse="anthropic")

    @router.post("/v1/search")
    async def search(body: SearchRequest, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        call = LoggedCall(identity, "/v1/search", openai_search.MODEL, "搜索", request_text=body.prompt)
        await filter_or_log(call, body.prompt)
        return await call.run(openai_search.handle, body.model_dump(mode="python"))

    @router.post("/alpha/search")
    @router.post("/v1/alpha/search")
    async def codex_alpha_search(
            body: CodexSearchRequest,
            request: Request,
            authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        payload = body.model_dump(mode="python")
        prompt = codex_search.request_prompt(payload)
        request_preview = prompt or "empty Codex search command"
        call = LoggedCall(
            identity,
            request.url.path,
            body.model or "auto",
            "Codex Search",
            request_text=request_preview,
        )
        if prompt:
            await filter_or_log(call, prompt)
        return await call.run(codex_search.handle, payload, str(identity.get("id") or ""))

    @router.get("/v1/editable-file-tasks")
    async def list_editable_file_tasks(ids: str = "", authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        task_ids = [item.strip() for item in ids.split(",") if item.strip()]
        return await run_in_threadpool(editable_file_task_service.list_tasks, identity, task_ids)

    @router.get("/files/{file_path:path}")
    async def download_editable_file(file_path: str):
        try:
            path = await run_in_threadpool(editable_file_task_service.public_file_path, file_path)
        except Exception as exc:
            raise HTTPException(status_code=404, detail={"error": "file not found"}) from exc
        return FileResponse(path, filename=path.name)

    @router.post("/v1/ppt/generations")
    async def create_ppt_task(body: EditableFileTaskRequest, request: Request, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        await filter_or_log(LoggedCall(identity, "/v1/ppt/generations", "gpt-5-5-thinking", "PPT生成任务", request_text=body.prompt), body.prompt)
        return await run_in_threadpool(
            editable_file_task_service.submit_ppt,
            identity,
            client_task_id=body.client_task_id or "",
            prompt=body.prompt,
            base64_images=body.base64_images,
            base_url=resolve_image_base_url(request),
        )

    @router.post("/v1/psd/generations")
    async def create_psd_task(body: EditableFileTaskRequest, request: Request, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        await filter_or_log(LoggedCall(identity, "/v1/psd/generations", "gpt-5-5-thinking", "PSD生成任务", request_text=body.prompt), body.prompt)
        return await run_in_threadpool(
            editable_file_task_service.submit_psd,
            identity,
            client_task_id=body.client_task_id or "",
            prompt=body.prompt,
            base64_images=body.base64_images,
            base_url=resolve_image_base_url(request),
        )

    return router
