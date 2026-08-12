from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.ai import create_router
from services.protocol import codex_files


class CodexFileEndpointTests(unittest.TestCase):
    def test_upload_token_is_single_use_and_discard_removes_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = codex_files.CodexFileStore(Path(directory))
            record = store.create("x.txt", 1)
            self.assertIs(store.get_for_upload(record.file_id, record.upload_token), record)
            self.assertIsNone(store.get_for_upload(record.file_id, record.upload_token))
            self.assertIs(store.mark_uploaded(record.file_id, 1)[0], record)
            self.assertIsNone(store.get_for_upload(record.file_id, record.upload_token))
            store.discard_upload(record)
            self.assertIsNone(store.get(record.file_id))

    def test_expired_records_are_cleaned_by_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = codex_files.CodexFileStore(Path(directory))
            record = store.create("x.txt", 1)
            record.created_at -= codex_files.FILE_TTL_SECONDS + 1
            record.path.write_bytes(b"x")
            self.assertIsNone(store.get(record.file_id))
            self.assertFalse(record.path.exists())

    def test_codex_create_upload_finalize_download_flow(self) -> None:
        app = FastAPI()
        app.include_router(create_router())
        with tempfile.TemporaryDirectory() as directory:
            store = codex_files.CodexFileStore(Path(directory))
            with (
                mock.patch.object(codex_files, "codex_file_store", store),
                mock.patch("api.ai.require_identity", return_value={"id": "test", "role": "admin"}),
            ):
                with TestClient(app) as client:
                    created = client.post(
                        "/v1/files",
                        headers={"authorization": "Bearer test"},
                        json={"file_name": "..\\README.md", "file_size": 11, "use_case": "codex"},
                    )
                    self.assertEqual(created.status_code, 200)
                    payload = created.json()
                    self.assertTrue(payload["file_id"].startswith("file_"))
                    self.assertIn("/upload/", payload["upload_url"])

                    upload_path = payload["upload_url"].split("http://testserver", 1)[-1]
                    uploaded = client.put(
                        upload_path,
                        headers={"content-length": "11", "x-ms-blob-type": "BlockBlob"},
                        content=b"hello world",
                    )
                    self.assertEqual(uploaded.status_code, 201)

                    finalized = client.post(
                        f"/v1/files/{payload['file_id']}/uploaded",
                        headers={"authorization": "Bearer test"},
                        json={},
                    )
                    self.assertEqual(finalized.status_code, 200)
                    final_payload = finalized.json()
                    self.assertEqual(final_payload["status"], "success")
                    self.assertEqual(final_payload["file_name"], "README.md")

                    download_path = final_payload["download_url"].split("http://testserver", 1)[-1]
                    downloaded = client.get(download_path)
                    self.assertEqual(downloaded.status_code, 200)
                    self.assertEqual(downloaded.content, b"hello world")

    def test_codex_finalize_rejects_size_mismatch(self) -> None:
        app = FastAPI()
        app.include_router(create_router())
        with tempfile.TemporaryDirectory() as directory:
            store = codex_files.CodexFileStore(Path(directory))
            with (
                mock.patch.object(codex_files, "codex_file_store", store),
                mock.patch("api.ai.require_identity", return_value={"id": "test", "role": "admin"}),
            ):
                with TestClient(app) as client:
                    created = client.post(
                        "/v1/files",
                        headers={"authorization": "Bearer test"},
                        json={"file_name": "a.txt", "file_size": 5, "use_case": "codex"},
                    )
                    payload = created.json()
                    upload_path = payload["upload_url"].split("http://testserver", 1)[-1]
                    self.assertEqual(client.put(upload_path, content=b"no").status_code, 201)
                    finalized = client.post(
                        f"/v1/files/{payload['file_id']}/uploaded",
                        headers={"authorization": "Bearer test"},
                        json={},
                    )
                    self.assertEqual(finalized.status_code, 200)
                    self.assertEqual(finalized.json()["status"], "error")


if __name__ == "__main__":
    unittest.main()
