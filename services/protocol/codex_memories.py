from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from services.protocol.codex_tool_bridge import _truncate_utf8


MEMORY_TRACE_MAX_BYTES = 24 * 1024
MEMORY_SUMMARY_MAX_BYTES = 12 * 1024
MEMORY_MAX_TRACES = 64
MEMORY_TOTAL_INPUT_MAX_BYTES = 512 * 1024


class MemorySummarizeRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = "auto"
    traces: list[dict[str, Any]] = Field(default_factory=list)
    reasoning: object | None = None


def _trace_text(trace: dict[str, Any], index: int) -> str:
    trace_id = str(trace.get("id") or f"trace-{index}")
    metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
    source_path = str(metadata.get("source_path") or "")
    items = trace.get("items") if isinstance(trace.get("items"), list) else []
    records = [f"trace_id={trace_id}", f"source_path={source_path}"]
    for item_index, item in enumerate(items):
        encoded = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        records.append(f"item[{item_index}]={encoded}")
    return _truncate_utf8("\n".join(records), MEMORY_TRACE_MAX_BYTES)


def _summaries(trace: dict[str, Any], index: int) -> dict[str, str]:
    text = _trace_text(trace, index)
    trace_id = str(trace.get("id") or f"trace-{index}")
    # The endpoint is deliberately extractive: it never invents a memory and
    # remains usable when the account pool has no separate summarizer model.
    summary = _truncate_utf8(text, MEMORY_SUMMARY_MAX_BYTES)
    return {
        "trace_summary": summary,
        "memory_summary": f"Trace {trace_id}:\n{summary}",
    }


def handle(body: dict[str, Any]) -> dict[str, Any]:
    traces = body.get("traces")
    if not isinstance(traces, list):
        traces = []
    if len(traces) > MEMORY_MAX_TRACES:
        raise HTTPException(status_code=413, detail={"error": f"traces supports at most {MEMORY_MAX_TRACES} items"})
    total_bytes = 0
    valid_traces: list[dict[str, Any]] = []
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        total_bytes += len(json.dumps(trace, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        if total_bytes > MEMORY_TOTAL_INPUT_MAX_BYTES:
            raise HTTPException(status_code=413, detail={"error": "traces input exceeds the 512 KiB limit"})
        valid_traces.append(trace)
    output = [
        _summaries(trace, index)
        for index, trace in enumerate(valid_traces)
    ]
    return {"output": output}
