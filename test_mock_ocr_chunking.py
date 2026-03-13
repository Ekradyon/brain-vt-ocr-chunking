"""Pruebas mock para ocr_chunking.py (sin plpy)."""

import unittest
from unittest import mock

import fitz

import ocr_chunking


class FakeDB:
    """Fake DB client for local tests without PostgreSQL."""

    def __init__(self, pdf_bytes: bytes) -> None:
        self.pdf_bytes = pdf_bytes
        self.last_job_id = 100

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def fetch_item_by_oid(self, oid: int):
        return {
            "trabajo_id": 2,
            "item_id": 2,
            "nombre_archivo": "demo_contrato.pdf",
            "estado": "OK",
            "lo_oid": oid,
        }

    def read_large_object(self, oid: int) -> bytes:
        return self.pdf_bytes

    def ensure_queue(self, **kwargs):
        return {"id": 1, "nombre": kwargs.get("queue_name", "TEST"), "jobsPendientes": 0, "jobsProcesando": 0}

    def acquire_queue_slot(self, queue_name: str):
        return {"acquired": True, "queue": {"nombre": queue_name, "jobsProcesando": 1}}

    def release_queue_slot(self, queue_name: str):
        return {"released": True, "queue": {"nombre": queue_name, "jobsProcesando": 0}}

    def refresh_queue_stats(self, queue_name: str):
        return {"nombre": queue_name, "jobsPendientes": 0, "jobsProcesando": 0}

    def create_job(self, **kwargs):
        self.last_job_id += 1
        return self.last_job_id

    def update_job_state(self, **kwargs):
        return None

    def count_existing_embeddings(self, **kwargs):
        return 0

    def delete_existing_embeddings(self, **kwargs):
        return 0

    def insert_embeddings(self, rows):
        return len(rows)

    def get_job(self, job_id: int):
        return None


def make_pdf_bytes(text: str) -> bytes:
    """Creates a tiny in-memory PDF for tests."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


class TestOCRChunkingMock(unittest.TestCase):
    """Local tests for mock and mocked-real execution."""

    def test_mock_success(self):
        req = ocr_chunking.OCRChunkingRequest(
            oid=2299268,
            mock=ocr_chunking.MockOptions(enabled=True, latency_ms=0),
        )
        res = ocr_chunking.process_request(req)
        self.assertTrue(res.exitoso)
        self.assertEqual(res.status, "COMPLETED")
        self.assertEqual(res.data.get("mock_mode"), True)

    def test_mock_forced_failure(self):
        req = ocr_chunking.OCRChunkingRequest(
            oid=2299268,
            mock=ocr_chunking.MockOptions(enabled=True, fail_phase="CHUNKING", latency_ms=0),
        )
        res = ocr_chunking.process_request(req)
        self.assertFalse(res.exitoso)
        self.assertEqual(res.status, "FAILED")
        self.assertEqual((res.error or {}).get("code"), "MOCK_FORCED_FAILURE")

    def test_real_pipeline_with_fake_db(self):
        fake_pdf = make_pdf_bytes("Contrato de prueba ANH. Texto extraible para chunking.")
        fake_db = FakeDB(fake_pdf)

        req = ocr_chunking.OCRChunkingRequest(
            oid=2299268,
            queue=ocr_chunking.QueueOptions(enabled=False),
            extraction=ocr_chunking.ExtractionOptions(engine="pymupdf", page_mode="full"),
            cleaning=ocr_chunking.CleaningOptions(enabled=True),
            chunking=ocr_chunking.ChunkingOptions(strategy="simple", simple_chunk_size=300, simple_chunk_overlap=0),
            embedding=ocr_chunking.EmbeddingOptions(enabled=False, save_to_db=False),
            mock=ocr_chunking.MockOptions(enabled=False),
        )

        with mock.patch("ocr_chunking.PostgresSettings.from_env", return_value=ocr_chunking.PostgresSettings("localhost", 5432, "niledb", "postgres", "plexia")):
            with mock.patch("ocr_chunking.PostgresClient", return_value=fake_db):
                res = ocr_chunking.run_real_pipeline(req)

        self.assertTrue(res.exitoso)
        self.assertEqual(res.status, "COMPLETED")
        self.assertEqual(res.data.get("engine_used"), "pymupdf")
        self.assertGreaterEqual(int(res.data.get("chunks_count", 0)), 1)


if __name__ == "__main__":
    unittest.main()
