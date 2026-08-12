from __future__ import annotations

import mimetypes
import secrets
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from services.config import DATA_DIR


MAX_FILE_BYTES = 512 * 1024 * 1024
FILE_TTL_SECONDS = 60 * 60


@dataclass
class CodexFileRecord:
    file_id: str
    upload_token: str
    download_token: str
    path: Path
    file_name: str
    file_size: int
    mime_type: str
    uploaded_bytes: int = 0
    uploaded: bool = False
    upload_claimed: bool = False
    created_at: float = 0.0


class CodexFileStore:
    """Local implementation of Codex's create/upload/finalize file flow."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (DATA_DIR / "codex_files")
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._records: dict[str, CodexFileRecord] = {}

    @staticmethod
    def _safe_file_name(value: object) -> str:
        name = str(value or "file.bin").replace("\\", "/").rsplit("/", 1)[-1].strip()
        return name[:240] or "file.bin"

    def _cleanup_locked(self, now: float) -> None:
        expired = [
            file_id
            for file_id, record in self._records.items()
            if now - record.created_at > FILE_TTL_SECONDS
        ]
        for file_id in expired:
            record = self._records.pop(file_id, None)
            if record is not None:
                try:
                    record.path.unlink(missing_ok=True)
                except OSError:
                    pass

    def create(self, file_name: object, file_size: object) -> CodexFileRecord:
        try:
            size = int(file_size)
        except (TypeError, ValueError) as exc:
            raise ValueError("file_size must be an integer") from exc
        if size < 0 or size > MAX_FILE_BYTES:
            raise ValueError(f"file_size must be between 0 and {MAX_FILE_BYTES} bytes")
        now = time.time()
        file_id = f"file_{uuid.uuid4().hex}"
        record = CodexFileRecord(
            file_id=file_id,
            upload_token=secrets.token_urlsafe(32),
            download_token=secrets.token_urlsafe(32),
            path=self.root / f"{file_id}.bin",
            file_name=self._safe_file_name(file_name),
            file_size=size,
            mime_type=mimetypes.guess_type(self._safe_file_name(file_name))[0] or "application/octet-stream",
            created_at=now,
        )
        with self._lock:
            self._cleanup_locked(now)
            self._records[file_id] = record
        return record

    def get_for_upload(self, file_id: str, upload_token: str) -> CodexFileRecord | None:
        with self._lock:
            self._cleanup_locked(time.time())
            record = self._records.get(file_id)
            if (
                record is None
                or record.uploaded
                or record.upload_claimed
                or not secrets.compare_digest(record.upload_token, upload_token)
            ):
                return None
            record.upload_claimed = True
            return record

    def get_for_download(self, file_id: str, download_token: str) -> CodexFileRecord | None:
        with self._lock:
            self._cleanup_locked(time.time())
            record = self._records.get(file_id)
            if record is None or not record.uploaded:
                return None
            if not secrets.compare_digest(record.download_token, download_token):
                return None
            return record

    def get(self, file_id: str) -> CodexFileRecord | None:
        with self._lock:
            self._cleanup_locked(time.time())
            return self._records.get(file_id)

    def mark_uploaded(self, file_id: str, uploaded_bytes: int) -> tuple[CodexFileRecord | None, str]:
        with self._lock:
            self._cleanup_locked(time.time())
            record = self._records.get(file_id)
            if record is None:
                return None, "file not found"
            record.uploaded_bytes = uploaded_bytes
            record.upload_claimed = False
            if uploaded_bytes != record.file_size:
                return record, f"uploaded size {uploaded_bytes} does not match declared size {record.file_size}"
            record.uploaded = True
            return record, ""

    def discard_upload(self, record: CodexFileRecord) -> None:
        with self._lock:
            self._records.pop(record.file_id, None)
            try:
                record.path.unlink(missing_ok=True)
            except OSError:
                pass


codex_file_store = CodexFileStore()
