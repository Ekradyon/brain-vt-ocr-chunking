"""
OCR + chunking + embeddings orchestrator (FastAPI/OpenAPI, psycopg2, no plpy).

Main goals:
- Single service file at project root.
- Required input: oid (PostgreSQL Large Object OID).
- Full document processing or first/last N pages.
- Queue + overwrite policies with clear and traceable states.
- Docling OCR, semantic/simple chunking, embedding generation, DB persistence.
- Mock mode for local tests without DB/server.

Run API server:
  python ocr_chunking.py --host 0.0.0.0 --port 8000

Run local mock demo (no DB):
  python ocr_chunking.py --mock-local
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import fitz
import psycopg2
import torch
import torch.nn.functional as torch_functional
import uvicorn
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc import DocItemLabel, DoclingDocument
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from psycopg2.extras import RealDictCursor, execute_batch
from transformers import AutoModel, AutoTokenizer


SERVICE_NAME = "OCR Chunking Embeddings Orchestrator"
SERVICE_VERSION = "2.0.0"
TAG_CHUNKING = "Segmento Chunking"
TAG_EMBEDDING = "Segmento Embedding"
TAG_OCR = "Segmento Docling-OCR"
TAG_PIPELINE = "Segmento Pipeline"
TAG_HELPERS = "Segmento Helpers"

OPENAPI_TAGS = [
    {"name": TAG_CHUNKING, "description": "Metodos de chunking (texto a chunks)."},
    {"name": TAG_EMBEDDING, "description": "Metodos de generacion de embeddings."},
    {"name": TAG_OCR, "description": "Metodos de OCR y extraccion de texto."},
    {"name": TAG_PIPELINE, "description": "Metodos de orquestacion completa PipelineOCR."},
    {"name": TAG_HELPERS, "description": "Endpoints auxiliares: health, example-request, validate-db."},
]

DEFAULT_QUEUE_NAME = "BRAINVT_OCR_EMBEDDINGS_GPU"
DEFAULT_JOB_TYPE = "BRAINVT_OCR_EMBEDDINGS_GPU"
DEFAULT_CREATED_BY = 1101
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_PROBE_MAX_PAGES = 60
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"

PAGE_INDICATOR_PATTERN = re.compile(
    r"(?i)^\s*(?:pagina\s+\d+\s+de\s+\d+|page\s+\d+|\-\s*\d+\s*\-|\[\d+\])\s*$",
    re.MULTILINE,
)
ISOLATED_NUMBER_PATTERN = re.compile(r"^[\d\s\.,\-/]+$")
MULTI_EMPTY_LINES_PATTERN = re.compile(r"\n{3,}")

LOGGER = logging.getLogger("ocr_chunking")
if not LOGGER.handlers:
    logging.basicConfig(
        level=os.getenv("OCR_LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

_MODEL_CACHE: Dict[str, Tuple[Any, Any, str]] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}
_DOCLING_CONVERTER_CACHE: Dict[str, Any] = {}
_MODEL_LOCK = Lock()
_TOKENIZER_LOCK = Lock()
_DOCLING_LOCK = Lock()


def utc_now_iso() -> str:
    """Returns UTC timestamp as ISO string."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_str(value: Any, default: str = "") -> str:
    """Safe str conversion."""
    if value is None:
        return default
    try:
        return str(value)
    except Exception:
        return default


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Safe int conversion."""
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Safe float conversion."""
    try:
        return float(value)
    except Exception:
        return default


def safe_bool(value: Any, default: bool = False) -> bool:
    """Safe bool conversion."""
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(value)
    text = safe_str(value, "").strip().lower()
    if text in {"1", "true", "t", "yes", "y", "si", "s", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off", ""}:
        return False
    return bool(default)


def to_json_dict(value: Any, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Converts value to dict when possible."""
    if default is None:
        default = {}
    if isinstance(value, dict):
        return value
    if value is None:
        return dict(default)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return dict(default)
    return dict(default)


def to_json_safe(value: Any) -> Any:
    """Converts values to JSON-safe structures (datetime -> ISO, etc.)."""
    if isinstance(value, dict):
        return {safe_str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [to_json_safe(v) for v in value]
    if isinstance(value, set):
        return [to_json_safe(v) for v in sorted(value, key=lambda x: safe_str(x))]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return safe_str(value, "")
    return value


def json_dumps_safe(value: Any) -> str:
    """JSON dump that never fails on datetime/date/bytes."""
    return json.dumps(to_json_safe(value), ensure_ascii=False)


def pydantic_model_dump(model: BaseModel) -> Dict[str, Any]:
    """Pydantic v1/v2 compatibility."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def pydantic_model_dump_json(model: BaseModel, *, indent: int = 2) -> str:
    """Pydantic v1/v2 compatibility for JSON dump."""
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json(indent=indent)
    return model.json(indent=indent, ensure_ascii=False)


def vector_to_pg_literal(vector: List[float]) -> str:
    """Converts vector to pgvector literal."""
    return "[" + ",".join(f"{float(x):.10f}" for x in vector) + "]"


def mean_pooling(model_output: Any, attention_mask: Any) -> Any:
    """Mean pooling for sentence embeddings."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1),
        min=1e-9,
    )


class PipelineError(Exception):
    """Pipeline exception with phase and machine-friendly error code."""

    def __init__(self, phase: str, code: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.phase = phase
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize error."""
        return {
            "phase": self.phase,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class PhaseRecorder:
    """Trace recorder for all pipeline phases."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def push(self, phase: str, status: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Adds one phase event."""
        event = {
            "phase": phase,
            "status": status,
            "message": message,
            "details": details or {},
            "timestamp_utc": utc_now_iso(),
        }
        self.events.append(event)
        return event

    def as_list(self) -> List[Dict[str, Any]]:
        """Returns all events."""
        return list(self.events)


@dataclass
class PostgresSettings:
    """Postgres settings."""

    host: str
    port: int
    dbname: str
    user: str
    password: str

    @staticmethod
    def from_env() -> "PostgresSettings":
        """Loads DB settings from environment."""
        return PostgresSettings(
            host=os.getenv("OCR_DB_HOST", "localhost"),
            port=int(os.getenv("OCR_DB_PORT", "5432")),
            dbname=os.getenv("OCR_DB_NAME", "niledb"),
            user=os.getenv("OCR_DB_USER", "postgres"),
            password=os.getenv("OCR_DB_PASSWORD", "plexia"),
        )


def bogota_now_iso() -> str:
    """Returns current time in America/Bogota offset as ISO string."""
    bogota_tz = timezone(timedelta(hours=-5))
    return datetime.now(bogota_tz).replace(microsecond=0).isoformat()


def _validate_db_connection() -> Dict[str, Any]:
    """
    Valida conexión a Postgres.
    Retorna versión, metadatos y resultado de una consulta de prueba.
    """
    settings = PostgresSettings.from_env()
    try:
        conn = psycopg2.connect(
            host=settings.host,
            port=settings.port,
            dbname=settings.dbname,
            user=settings.user,
            password=settings.password,
            connect_timeout=5,
        )
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT current_database() AS current_database, current_user AS current_user, version() AS version;"
                )
                row = cur.fetchone()
                if not row:
                    return {
                        "status": "error",
                        "message": "No se obtuvo fila de metadatos.",
                        "postgres_version": None,
                        "metadata": None,
                        "test_query": None,
                        "error": "Empty result",
                    }
                version_str = row["version"] or ""
                version_short = version_str.split(",")[0].strip() if version_str else None
                metadata = {
                    "current_database": row["current_database"],
                    "current_user": row["current_user"],
                    "server_version_full": version_str,
                    "server_version_short": version_short,
                    "connection": {
                        "host": settings.host,
                        "port": settings.port,
                        "dbname": settings.dbname,
                        "user": settings.user,
                    },
                }
                cur.execute(
                    "SELECT 1 AS ping, current_timestamp AS server_time, "
                    "current_setting('server_version_num') AS version_num;"
                )
                test_row = cur.fetchone()
                test_query = dict(test_row) if test_row else None
            return {
                "status": "ok",
                "message": "Conexión exitosa.",
                "postgres_version": version_short,
                "metadata": metadata,
                "test_query": test_query,
                "error": None,
            }
        finally:
            conn.close()
    except Exception as exc:
        return {
            "status": "error",
            "message": "No se pudo conectar a Postgres.",
            "postgres_version": None,
            "metadata": {
                "connection": {
                    "host": settings.host,
                    "port": settings.port,
                    "dbname": settings.dbname,
                }
            },
            "test_query": None,
            "error": str(exc),
        }


class PostgresClient:
    """Postgres adapter for queue/job/embedding operations."""

    def __init__(self, settings: PostgresSettings) -> None:
        self.settings = settings
        self.conn = psycopg2.connect(
            host=settings.host,
            port=settings.port,
            dbname=settings.dbname,
            user=settings.user,
            password=settings.password,
        )
        self.conn.autocommit = False

    def __enter__(self) -> "PostgresClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc is not None:
            try:
                self.conn.rollback()
            except Exception:
                pass
        try:
            self.conn.close()
        except Exception:
            pass

    def query_one(self, sql: str, params: Tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
        """Executes SELECT and returns one dict row."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        return dict(row) if row is not None else None

    def query_all(self, sql: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
        """Executes SELECT and returns all dict rows."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def execute(self, sql: str, params: Tuple[Any, ...] = ()) -> int:
        """Executes write SQL and returns affected rows."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, params)
                affected = int(cur.rowcount)
            self.conn.commit()
            return affected
        except Exception:
            self.conn.rollback()
            raise

    def execute_returning_one(self, sql: str, params: Tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
        """Executes write SQL with RETURNING and returns one row."""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
            self.conn.commit()
            return dict(row) if row is not None else None
        except Exception:
            self.conn.rollback()
            raise

    def fetch_item_by_oid(self, oid: int) -> Optional[Dict[str, Any]]:
        """Gets latest item from Operaciones.ItemsIngestaSmb by loOid."""
        sql = """
        SELECT
          i."trabajoId"      AS trabajo_id,
          i.id               AS item_id,
          i."nombreArchivo"  AS nombre_archivo,
          i.estado           AS estado,
          i."loOid"          AS lo_oid,
          i."bytesEscritos"  AS bytes_escritos,
          i."tamanoBytes"    AS tamano_bytes,
          i."hashSha256"     AS sha256,
          i."cargadoEn"      AS cargado_en
        FROM "Operaciones"."ItemsIngestaSmb" i
        WHERE i."loOid" = %s
        ORDER BY i."trabajoId" DESC, i.id DESC
        LIMIT 1
        """
        return self.query_one(sql, (int(oid),))

    def read_large_object(self, oid: int) -> bytes:
        """Reads pg_largeobject bytes from loOid."""
        try:
            lobj = self.conn.lobject(oid=int(oid), mode="rb")
            data = lobj.read()
            lobj.close()
            self.conn.commit()
            return data
        except Exception:
            self.conn.rollback()
            raise

    def ensure_queue(
        self,
        queue_name: str,
        description: str,
        max_concurrency: int,
        timeout_seconds: int,
        retries_max: int,
        priority_default: int,
    ) -> Dict[str, Any]:
        """Creates or updates queue configuration."""
        sql = """
        INSERT INTO "Operaciones"."ColasProcesamiento"
        ("nombre","descripcion","maxConcurrencia","prioridadDefault","timeoutSegundos","reintentosMax",
         "jobsPendientes","jobsProcesando","activa","createdAt")
        VALUES (%s,%s,%s,%s,%s,%s,0,0,true,(CURRENT_TIMESTAMP AT TIME ZONE 'America/Bogota'))
        ON CONFLICT ("nombre") DO UPDATE
        SET "descripcion" = COALESCE(EXCLUDED."descripcion","Operaciones"."ColasProcesamiento"."descripcion"),
            "maxConcurrencia" = EXCLUDED."maxConcurrencia",
            "prioridadDefault" = EXCLUDED."prioridadDefault",
            "timeoutSegundos" = EXCLUDED."timeoutSegundos",
            "reintentosMax" = EXCLUDED."reintentosMax",
            "activa" = true
        RETURNING "id","nombre","maxConcurrencia","prioridadDefault","timeoutSegundos","reintentosMax",
                  "activa",COALESCE("jobsPendientes",0) AS "jobsPendientes",
                  COALESCE("jobsProcesando",0) AS "jobsProcesando"
        """
        row = self.execute_returning_one(
            sql,
            (
                queue_name,
                description,
                int(max_concurrency),
                int(priority_default),
                int(timeout_seconds),
                int(retries_max),
            ),
        )
        return row or {}

    def acquire_queue_slot(self, queue_name: str) -> Dict[str, Any]:
        """Acquires one queue slot if concurrency allows."""
        sql = """
        UPDATE "Operaciones"."ColasProcesamiento"
        SET "jobsProcesando" = COALESCE("jobsProcesando",0) + 1
        WHERE "nombre" = %s
          AND "activa" = true
          AND COALESCE("jobsProcesando",0) < COALESCE("maxConcurrencia",1)
        RETURNING "id","nombre","maxConcurrencia","prioridadDefault","timeoutSegundos","reintentosMax",
                  "activa",COALESCE("jobsPendientes",0) AS "jobsPendientes",
                  COALESCE("jobsProcesando",0) AS "jobsProcesando"
        """
        row = self.execute_returning_one(sql, (queue_name,))
        return {"acquired": bool(row), "queue": row}

    def release_queue_slot(self, queue_name: str) -> Dict[str, Any]:
        """Releases one queue slot."""
        sql = """
        UPDATE "Operaciones"."ColasProcesamiento"
        SET "jobsProcesando" = GREATEST(COALESCE("jobsProcesando",0) - 1, 0)
        WHERE "nombre" = %s
        RETURNING "id","nombre",COALESCE("jobsPendientes",0) AS "jobsPendientes",
                  COALESCE("jobsProcesando",0) AS "jobsProcesando"
        """
        row = self.execute_returning_one(sql, (queue_name,))
        return {"released": bool(row), "queue": row}

    def refresh_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """Refreshes queue counters from Operaciones.JobsProcesamiento."""
        sql = """
        WITH agg AS (
          SELECT
            count(*) FILTER (WHERE j."estado" = ('PENDIENTE'::"Operaciones"."EstadoJob"))::int AS pendientes,
            count(*) FILTER (WHERE j."estado" = ('EN_PROCESO'::"Operaciones"."EstadoJob"))::int AS procesando
          FROM "Operaciones"."JobsProcesamiento" j
          WHERE j."tipo" = %s
        )
        UPDATE "Operaciones"."ColasProcesamiento" c
        SET "jobsPendientes" = agg.pendientes,
            "jobsProcesando" = agg.procesando
        FROM agg
        WHERE c."nombre" = %s
        RETURNING c."id", c."nombre",
                  COALESCE(c."jobsPendientes",0) AS "jobsPendientes",
                  COALESCE(c."jobsProcesando",0) AS "jobsProcesando"
        """
        row = self.execute_returning_one(sql, (queue_name, queue_name))
        return row or {}

    def create_job(
        self,
        job_type: str,
        estado: str,
        prioridad: int,
        documento_id: Optional[int],
        parametros: Dict[str, Any],
        max_intentos: int,
    ) -> int:
        """Creates one job in Operaciones.JobsProcesamiento."""
        sql = """
        INSERT INTO "Operaciones"."JobsProcesamiento"
        ("tipo","estado","prioridad","documentoId","parametros","resultado","errorMensaje","intentos","maxIntentos",
         "programadoPara","inicio","fin","workerId","createdAt")
        VALUES (
            %s,
            CAST(%s AS "Operaciones"."EstadoJob"),
            %s,
            %s,
            %s,
            '{}'::text,
            NULL,
            0,
            %s,
            "BrainVtCommons"."getBogotaTime"(),
            CASE WHEN %s='EN_PROCESO' THEN "BrainVtCommons"."getBogotaTime"() ELSE NULL END,
            NULL,
            NULL,
            "BrainVtCommons"."getBogotaTime"()
        )
        RETURNING "id"
        """
        row = self.execute_returning_one(
            sql,
            (
                job_type,
                estado,
                int(prioridad),
                documento_id,
                json_dumps_safe(parametros),
                int(max_intentos),
                estado,
            ),
        )
        if row is None:
            raise RuntimeError("No fue posible crear job de procesamiento.")
        return int(row["id"])

    def update_job_state(
        self,
        job_id: int,
        estado: str,
        resultado: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        set_inicio_if_null: bool = False,
        set_fin: bool = False,
        clear_error: bool = False,
    ) -> None:
        """Updates job status, result payload and timestamps."""
        sets: List[str] = ['"estado" = CAST(%s AS "Operaciones"."EstadoJob")']
        params: List[Any] = [estado]

        if resultado is not None:
            sets.append('"resultado" = %s')
            params.append(json_dumps_safe(resultado))
        if error_message is not None:
            sets.append('"errorMensaje" = %s')
            params.append(error_message)
        elif clear_error:
            sets.append('"errorMensaje" = NULL')

        if set_inicio_if_null:
            sets.append('"inicio" = COALESCE("inicio","BrainVtCommons"."getBogotaTime"())')
        if set_fin:
            sets.append('"fin" = "BrainVtCommons"."getBogotaTime"()')

        sql = f'UPDATE "Operaciones"."JobsProcesamiento" SET {",".join(sets)} WHERE "id" = %s'
        params.append(int(job_id))
        self.execute(sql, tuple(params))

    def get_job(self, job_id: int) -> Optional[Dict[str, Any]]:
        """Gets one job row."""
        sql = """
        SELECT
          "id","tipo","estado","documentoId","prioridad","intentos","maxIntentos",
          "parametros","resultado","errorMensaje","programadoPara","inicio","fin","workerId","createdAt"
        FROM "Operaciones"."JobsProcesamiento"
        WHERE "id" = %s
        """
        return self.query_one(sql, (int(job_id),))

    def count_existing_embeddings(self, documento_id: Optional[int]) -> int:
        """Counts existing embeddings for one documentoId (overwrite a nivel documento)."""
        if documento_id is None:
            return 0
        row = self.query_one(
            'SELECT COUNT(*)::int AS c FROM "IaCore"."Embeddings" WHERE "documentoId" = %s',
            (int(documento_id),),
        )
        return int(row["c"]) if row else 0

    def delete_existing_embeddings(self, documento_id: Optional[int]) -> int:
        """Deletes existing embeddings for one documentoId (overwrite a nivel documento)."""
        if documento_id is None:
            return 0
        return self.execute(
            'DELETE FROM "IaCore"."Embeddings" WHERE "documentoId" = %s',
            (int(documento_id),),
        )

    def insert_embeddings(self, rows: List[Tuple[Any, ...]]) -> int:
        """Bulk inserts rows into IaCore.Embeddings."""
        if not rows:
            return 0
        sql = """
        INSERT INTO "IaCore"."Embeddings" (
            "chunkFin",
            "chunkIndex",
            "chunkInicio",
            "chunkTexto",
            "createdAt",
            "createdBy",
            "documentoId",
            "jobFileId",
            "metadata",
            "modelo",
            "tokens",
            "updatedAt",
            "vector"
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            (CURRENT_TIMESTAMP AT TIME ZONE 'America/Bogota'),
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            (CURRENT_TIMESTAMP AT TIME ZONE 'America/Bogota'),
            %s::vector
        )
        """
        try:
            with self.conn.cursor() as cur:
                execute_batch(cur, sql, rows, page_size=128)
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return len(rows)


class QueueOptions(BaseModel):
    """Queue options."""

    enabled: bool = Field(default=True, description="Enable queue controls.")
    max_concurrency: int = Field(default=2, ge=1, le=64)
    queue_when_busy: bool = Field(
        default=True,
        description="Si la cola esta ocupada, crea job PENDIENTE (ENQUEUED) en lugar de fallar.",
    )


class OverwriteOptions(BaseModel):
    """Overwrite options for duplicate embeddings (siempre a nivel documento)."""

    enabled: bool = Field(
        default=False,
        description="Si true, borra embeddings existentes del documento antes de insertar.",
    )


class ExtractionOptions(BaseModel):
    """Text/OCR extraction options."""

    engine: str = Field(default="auto", description="auto | docling | pymupdf")
    enable_pymupdf_fast_path: bool = Field(default=True)
    fast_path_confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    force_full_page_ocr: bool = Field(default=False)
    do_table_structure: bool = Field(default=True)
    images_scale: float = Field(default=1.5, ge=0.5, le=3.0)
    probe_max_pages: int = Field(default=DEFAULT_PROBE_MAX_PAGES, ge=1, le=5000)
    page_mode: str = Field(default="full", description="full | head_tail")
    head_pages: int = Field(default=5, ge=1, le=1000)
    tail_pages: int = Field(default=5, ge=1, le=1000)


class CleaningOptions(BaseModel):
    """Deterministic text-cleaning options."""

    enabled: bool = Field(default=True)
    remove_headers: bool = Field(default=True)
    remove_isolated_numbers: bool = Field(default=True)
    header_threshold: int = Field(default=3, ge=2, le=20)


class ChunkingOptions(BaseModel):
    """Chunking options."""

    strategy: str = Field(default="semantic", description="semantic | simple")
    max_chunks: int = Field(default=0, ge=0, description="0 means unlimited.")
    simple_chunk_size: int = Field(default=1500, ge=100, le=50000)
    simple_chunk_overlap: int = Field(default=200, ge=0, le=10000)
    min_text_chars: int = Field(default=50, ge=1, le=100000)
    enable_simple_fallback: bool = Field(
        default=False,
        description="Si true y semantic no genera chunks, usa simple chunking como fallback.",
    )


class EmbeddingOptions(BaseModel):
    """Embedding generation and persistence options."""

    enabled: bool = Field(default=True)
    model_name: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    batch_size: int = Field(
        default=8,
        ge=1,
        le=256,
        description="Cantidad de chunks procesados por lote al generar embeddings.",
    )
    save_to_db: bool = Field(default=True)
    return_vectors: bool = Field(default=False)
    require_inserted_rows: bool = Field(default=True)
    created_by_default: int = Field(default=DEFAULT_CREATED_BY)


class MockOptions(BaseModel):
    """Mock mode options."""

    enabled: bool = Field(default=False, description="Enable mock pipeline.")
    fail_phase: Optional[str] = Field(default=None, description="If set, mock fails at this phase.")
    latency_ms: int = Field(default=0, ge=0, le=120000)
    without_db: bool = Field(default=True)


class OCRChunkingRequest(BaseModel):
    """Input request model."""

    oid: int = Field(
        ...,
        description="Identificador unico del documento y OID del PDF (se usa el mismo valor para ambos).",
    )
    nombre_documento: Optional[str] = Field(
        default=None,
        description="Nombre del documento. Si se envia, no se intenta resolver nombre por OID.",
    )
    file_name: Optional[str] = Field(
        default=None,
        description="Alias opcional de nombre_documento para compatibilidad.",
    )
    job_filde_id: Optional[int] = Field(
        default=None,
        description="Id de archivo/job para trazabilidad (compatibilidad con nombre solicitado).",
    )
    usuario_proceso: Optional[str] = Field(
        default=None,
        description="Usuario funcional que ejecuta la solicitud.",
    )
    job_proceso: Optional[str] = Field(
        default=None,
        description="Nombre o identificador funcional del job de negocio.",
    )
    created_by: Optional[int] = Field(default=None)
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Metadata del documento. Recomendado: nombre_documento, metadata_documento"
            " (ej: paginas, idioma) y ruta_pdf."
        ),
    )
    queue: QueueOptions = Field(default_factory=QueueOptions)
    overwrite: OverwriteOptions = Field(default_factory=OverwriteOptions)
    extraction: ExtractionOptions = Field(default_factory=ExtractionOptions)
    cleaning: CleaningOptions = Field(default_factory=CleaningOptions)
    chunking: ChunkingOptions = Field(default_factory=ChunkingOptions)
    embedding: EmbeddingOptions = Field(default_factory=EmbeddingOptions)
    mock: MockOptions = Field(default_factory=MockOptions)

    model_config = {
        "json_schema_extra": {
            "example": {
                "oid": 2299268,
                "nombre_documento": "CTO_EyP_LLA_50_2013.pdf",
                "file_name": None,
                "job_filde_id": 4567,
                "usuario_proceso": "analista_anh",
                "job_proceso": "JOB_OCR_20260309_001",
                "created_by": 1101,
                "metadata": {
                    "nombre_documento": "CTO_EyP_LLA_50_2013.pdf",
                    "metadata_documento": {"paginas": 133, "idioma": "es"},
                    "ruta_pdf": "\\\\servidor\\share\\CTO_EyP_LLA_50_2013.pdf",
                },
                "queue": {
                    "enabled": True,
                    "max_concurrency": 2,
                    "queue_when_busy": True,
                },
                "overwrite": {"enabled": False},
                "extraction": {
                    "engine": "auto",
                    "enable_pymupdf_fast_path": True,
                    "fast_path_confidence_threshold": 0.85,
                    "page_mode": "head_tail",
                    "head_pages": 8,
                    "tail_pages": 8,
                },
                "cleaning": {"enabled": True, "remove_headers": True, "remove_isolated_numbers": True},
                "chunking": {"strategy": "semantic", "max_chunks": 0, "enable_simple_fallback": False},
                "embedding": {
                    "enabled": True,
                    "model_name": "intfloat/multilingual-e5-large-instruct",
                    "save_to_db": True,
                    "return_vectors": False,
                },
                "mock": {"enabled": False},
            }
        }
    }


class OCRChunkingResponse(BaseModel):
    """Output response model."""

    status: str = Field(description="COMPLETED | ENQUEUED | FAILED")
    exitoso: bool
    message: str
    error: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Error estructurado: phase, code, message, details.",
    )
    phases: List[Dict[str, Any]] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict)


class OCRChunkingBatchRequest(BaseModel):
    """Batch request model."""

    requests: List[OCRChunkingRequest] = Field(..., min_length=1, max_length=200)
    parallel_workers: int = Field(default=1, ge=1, le=16)


class OCRChunkingBatchResponse(BaseModel):
    """Batch response model."""

    status: str
    exitoso: bool
    message: str
    total: int
    completados: int
    fallidos: int
    resultados: List[OCRChunkingResponse] = Field(default_factory=list)


def load_tokenizer(model_name: str) -> Any:
    """Loads tokenizer with in-memory cache."""
    with _TOKENIZER_LOCK:
        cached = _TOKENIZER_CACHE.get(model_name)
        if cached is not None:
            return cached
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        _TOKENIZER_CACHE[model_name] = tokenizer
        return tokenizer


def load_embedding_model(model_name: str) -> Tuple[Any, Any, str]:
    """Loads tokenizer/model/device with in-memory cache."""
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(model_name)
        if cached is not None:
            return cached
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        _MODEL_CACHE[model_name] = (tokenizer, model, device)
        return tokenizer, model, device


def load_docling_converter(force_full_page_ocr: bool, do_table_structure: bool, images_scale: float) -> Any:
    """Loads Docling converter with option-key cache."""
    key = f"{int(force_full_page_ocr)}|{int(do_table_structure)}|{images_scale:.3f}"
    with _DOCLING_LOCK:
        cached = _DOCLING_CONVERTER_CACHE.get(key)
        if cached is not None:
            return cached
        options = PdfPipelineOptions()
        options.do_ocr = True
        options.do_table_structure = bool(do_table_structure)
        options.images_scale = float(images_scale)
        options.generate_page_images = False
        options.generate_picture_images = False
        if hasattr(options, "ocr_options") and options.ocr_options is not None:
            setattr(options.ocr_options, "force_full_page_ocr", bool(force_full_page_ocr))
        converter = DocumentConverter(format_options={"pdf": PdfFormatOption(pipeline_options=options)})
        _DOCLING_CONVERTER_CACHE[key] = converter
        return converter


def apply_page_selection(pdf_bytes: bytes, page_mode: str, head_pages: int, tail_pages: int) -> Tuple[bytes, Dict[str, Any]]:
    """Applies full/head_tail selection and returns selected PDF bytes."""
    mode = safe_str(page_mode, "full").strip().lower()
    if mode not in {"full", "head_tail"}:
        raise PipelineError("PAGE_SELECTION", "INVALID_PAGE_MODE", "page_mode invalido", {"page_mode": mode})
    if mode == "full":
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = int(doc.page_count)
        doc.close()
        return pdf_bytes, {"mode": "full", "total_pages": total_pages, "selected_pages": total_pages}

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = int(doc.page_count)
    head_n = max(1, int(head_pages))
    tail_n = max(1, int(tail_pages))
    front = list(range(min(head_n, total_pages)))
    back = list(range(max(total_pages - tail_n, 0), total_pages))
    selected_indexes = sorted(set(front + back))

    out_doc = fitz.open()
    for idx in selected_indexes:
        out_doc.insert_pdf(doc, from_page=idx, to_page=idx)
    selected_pdf = out_doc.tobytes(garbage=4, deflate=True)
    out_doc.close()
    doc.close()

    return selected_pdf, {
        "mode": "head_tail",
        "total_pages": total_pages,
        "selected_pages": len(selected_indexes),
        "head_pages": head_n,
        "tail_pages": tail_n,
        "selected_indexes_0_based": selected_indexes,
    }


def probe_pdf_extractability(pdf_bytes: bytes, max_pages: int) -> Dict[str, Any]:
    """Probes text extractability and image presence with PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = int(doc.page_count)
    sample_n = min(total_pages, max(1, int(max_pages)))

    pages_with_text = 0
    pages_with_images = 0
    total_chars = 0
    for idx in range(sample_n):
        page = doc.load_page(idx)
        page_text = page.get_text("text") or ""
        page_chars = len(page_text.strip())
        if page_chars > 20:
            pages_with_text += 1
        if len(page.get_images(full=True)) > 0:
            pages_with_images += 1
        total_chars += page_chars
    doc.close()

    text_page_ratio = float(pages_with_text) / float(sample_n) if sample_n > 0 else 0.0
    avg_chars_per_page = float(total_chars) / float(sample_n) if sample_n > 0 else 0.0
    density_score = min(1.0, avg_chars_per_page / 1000.0)
    image_ratio = float(pages_with_images) / float(sample_n) if sample_n > 0 else 0.0
    image_penalty = 0.15 if image_ratio > 0.5 else 0.0
    confidence = max(0.0, min(1.0, (0.6 * text_page_ratio) + (0.4 * density_score) - image_penalty))

    return {
        "total_pages": total_pages,
        "sample_pages": sample_n,
        "pages_with_text": pages_with_text,
        "pages_with_images": pages_with_images,
        "sample_total_chars": total_chars,
        "sample_avg_chars_per_page": round(avg_chars_per_page, 3),
        "text_page_ratio": round(text_page_ratio, 6),
        "image_ratio": round(image_ratio, 6),
        "extractable_confidence": round(confidence, 6),
        "is_text_extractable": confidence >= 0.85,
    }


def extract_text_pymupdf(pdf_bytes: bytes) -> Tuple[str, int]:
    """Extracts text with PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = int(doc.page_count)
    parts: List[str] = []
    for idx in range(page_count):
        page = doc.load_page(idx)
        parts.append(page.get_text("text") or "")
    doc.close()
    return "\n\n".join(parts).strip(), page_count


def extract_text_docling(pdf_bytes: bytes, extraction: ExtractionOptions) -> Tuple[str, int, Dict[str, Any]]:
    """Extracts OCR text with Docling."""
    converter = load_docling_converter(
        force_full_page_ocr=bool(extraction.force_full_page_ocr),
        do_table_structure=bool(extraction.do_table_structure),
        images_scale=float(extraction.images_scale),
    )

    temp = tempfile.NamedTemporaryFile(prefix="docling_ocr_", suffix=".pdf", delete=False)
    temp_path = temp.name
    temp.write(pdf_bytes)
    temp.flush()
    temp.close()
    try:
        result = converter.convert(temp_path)
        status = getattr(result, "status", None)
        status_str = safe_str(status, "")
        success = status == ConversionStatus.SUCCESS or status_str.endswith("SUCCESS")
        if not success:
            raise PipelineError(
                "TEXT_EXTRACTION",
                "DOCLING_CONVERSION_FAILED",
                "Docling no pudo convertir el PDF.",
                {"status": status_str},
            )

        document = getattr(result, "document", None)
        if document is None:
            raise PipelineError(
                "TEXT_EXTRACTION",
                "DOCLING_RESULT_EMPTY",
                "Docling no devolvio documento.",
            )
        text = safe_str(getattr(document, "export_to_markdown", lambda: "")(), "")
        if not text.strip():
            text = safe_str(getattr(document, "export_to_text", lambda: "")(), "")
        page_count = int(len(getattr(document, "pages", []) or []))
        return text.strip(), page_count, {"conversion_status": status_str, "conversion_mode": "docling_ocr"}
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


def clean_text(text: str, cleaning: CleaningOptions) -> Tuple[str, Dict[str, Any]]:
    """Applies deterministic cleanup controlled by flags."""
    original = safe_str(text, "")
    if not cleaning.enabled:
        return original, {"enabled": False, "chars_before": len(original), "chars_after": len(original)}

    cleaned = PAGE_INDICATOR_PATTERN.sub("", original)
    lines = cleaned.split("\n")

    if cleaning.remove_headers:
        counts: Dict[str, int] = {}
        for line in lines:
            normalized = line.strip()
            if 5 < len(normalized) < 150:
                counts[normalized] = counts.get(normalized, 0) + 1
        headers = {line for line, count in counts.items() if count >= int(cleaning.header_threshold)}
        seen: set[str] = set()
        filtered: List[str] = []
        for line in lines:
            normalized = line.strip()
            if normalized in headers:
                if normalized in seen:
                    continue
                seen.add(normalized)
            filtered.append(line)
        lines = filtered

    if cleaning.remove_isolated_numbers:
        lines = [line for line in lines if not ISOLATED_NUMBER_PATTERN.match(line.strip()) or not line.strip()]

    merged = "\n".join(lines)
    merged = MULTI_EMPTY_LINES_PATTERN.sub("\n\n", merged).strip()
    return merged, {
        "enabled": True,
        "chars_before": len(original),
        "chars_after": len(merged),
        "remove_headers": bool(cleaning.remove_headers),
        "remove_isolated_numbers": bool(cleaning.remove_isolated_numbers),
    }


def build_docling_document(text: str) -> Any:
    """Builds DoclingDocument from plain text using compatible methods."""
    try:
        doc = DoclingDocument(name="documento")
        if hasattr(doc, "add_text"):
            doc.add_text(text)
            return doc
    except Exception:
        pass
    try:
        doc = DoclingDocument(name="documento")
        if hasattr(doc, "add_text"):
            doc.add_text(label=DocItemLabel.PARAGRAPH, text=text)
            return doc
    except Exception:
        pass
    if hasattr(DoclingDocument, "from_markdown"):
        return DoclingDocument.from_markdown(text)
    if hasattr(DoclingDocument, "from_text"):
        return DoclingDocument.from_text(text)
    raise RuntimeError("No fue posible construir DoclingDocument.")


def simple_chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple character-based chunking."""
    content = safe_str(text, "")
    if not content.strip():
        return []
    size = max(1, int(chunk_size))
    ov = max(0, int(overlap))
    if ov >= size:
        ov = max(0, size - 1)
    chunks: List[str] = []
    pos = 0
    total = len(content)
    while pos < total:
        end = min(pos + size, total)
        piece = content[pos:end].strip()
        if piece:
            chunks.append(piece)
        if end >= total:
            break
        pos = end - ov
    return chunks


def semantic_chunk_text(text: str, tokenizer: Any) -> List[str]:
    """Semantic chunking with Docling HybridChunker."""
    document = build_docling_document(text)
    chunker = HybridChunker(tokenizer=tokenizer)
    raw_chunks = list(chunker.chunk(document))
    chunks: List[str] = []
    for chunk in raw_chunks:
        if hasattr(chunk, "text"):
            value = safe_str(getattr(chunk, "text"), "").strip()
        elif isinstance(chunk, dict):
            value = safe_str(chunk.get("text"), "").strip()
        else:
            value = safe_str(chunk, "").strip()
        if value:
            chunks.append(value)
    return chunks


def rebalance_chunks(chunks: List[str], target: Optional[int]) -> List[str]:
    """Rebalances chunk list to target count if needed."""
    if target is None:
        return chunks
    wanted = max(1, int(target))
    clean = [safe_str(x, "").strip() for x in chunks if safe_str(x, "").strip()]
    if len(clean) == wanted:
        return clean
    if len(clean) > wanted:
        return clean[:wanted]
    while len(clean) < wanted:
        longest_idx = max(range(len(clean)), key=lambda idx: len(clean[idx]))
        source = clean[longest_idx]
        if len(source) < 2:
            break
        split_at = max(1, len(source) // 2)
        left = source[:split_at].strip()
        right = source[split_at:].strip()
        if not left or not right:
            break
        clean = clean[:longest_idx] + [left, right] + clean[longest_idx + 1 :]
    return clean[:wanted]


def estimate_bounds(full_text: str, chunks: List[str]) -> List[Tuple[int, int]]:
    """Estimates chunk start/end offsets on source text."""
    text = safe_str(full_text, "")
    cursor = 0
    text_len = len(text)
    bounds: List[Tuple[int, int]] = []
    for chunk in chunks:
        if not chunk:
            bounds.append((cursor, cursor))
            continue
        start = text.find(chunk, cursor)
        if start < 0:
            start = text.find(chunk)
        if start < 0:
            start = min(cursor, text_len)
        end = min(text_len, start + len(chunk))
        cursor = max(cursor, end)
        bounds.append((int(start), int(end)))
    return bounds


def embed_chunks(
    chunks: List[str],
    tokenizer: Any,
    model: Any,
    device: str,
    max_length: int,
    batch_size: int,
) -> Tuple[List[List[float]], List[int]]:
    """Generates normalized embeddings in batches."""
    vectors: List[List[float]] = []
    tokens_per_chunk: List[int] = []
    step = max(1, int(batch_size))

    for idx in range(0, len(chunks), step):
        batch_text = chunks[idx : idx + step]
        encoded = tokenizer(
            batch_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max(16, int(max_length)),
        )
        attention_mask = encoded["attention_mask"]
        if device == "cuda":
            encoded = {k: v.to(device) for k, v in encoded.items()}
            attention_mask = attention_mask.to(device)
        with torch.no_grad():
            model_output = model(**encoded)
        pooled = mean_pooling(model_output, attention_mask)
        normalized = torch_functional.normalize(pooled, p=2, dim=1)

        vectors.extend(normalized.detach().cpu().tolist())
        token_counts = attention_mask.detach().cpu().sum(dim=1).tolist()
        tokens_per_chunk.extend([int(x) for x in token_counts])
    return vectors, tokens_per_chunk


def resolve_extraction_engine(extraction: ExtractionOptions, probe: Dict[str, Any]) -> str:
    """Decides extraction engine based on request and probe confidence."""
    requested = safe_str(extraction.engine, "auto").strip().lower()
    if requested not in {"auto", "docling", "pymupdf"}:
        raise PipelineError("TEXT_EXTRACTION", "INVALID_ENGINE", "extraction.engine invalido", {"engine": requested})
    if extraction.force_full_page_ocr:
        return "docling"
    if requested in {"docling", "pymupdf"}:
        return requested
    if not extraction.enable_pymupdf_fast_path:
        return "docling"

    confidence = float(probe.get("extractable_confidence", 0.0) or 0.0)
    threshold = float(extraction.fast_path_confidence_threshold)
    return "pymupdf" if confidence >= threshold else "docling"


def gpu_metrics() -> Dict[str, Any]:
    """Returns CUDA resource indicators."""
    if not torch.cuda.is_available():
        return {"cuda_available": False, "device_count": 0}
    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    total_mem = int(props.total_memory)
    allocated = int(torch.cuda.memory_allocated(device_index))
    reserved = int(torch.cuda.memory_reserved(device_index))
    free_estimate = max(total_mem - reserved, 0)
    return {
        "cuda_available": True,
        "device_count": int(torch.cuda.device_count()),
        "device_index": int(device_index),
        "device_name": props.name,
        "total_memory_bytes": total_mem,
        "allocated_memory_bytes": allocated,
        "reserved_memory_bytes": reserved,
        "free_estimate_bytes": free_estimate,
    }


def derive_embedding_max_length(chunking: ChunkingOptions) -> int:
    """Deriva max_length para embeddings a partir de la estrategia de chunking."""
    approx_from_chars = max(128, int(chunking.simple_chunk_size / 4))
    return min(2048, approx_from_chars)


def build_job_payload(
    request: OCRChunkingRequest,
    file_name: str,
    item_info: Optional[Dict[str, Any]],
    stage: str,
) -> Dict[str, Any]:
    """Builds compact payload for Operaciones.JobsProcesamiento."""
    return {
        "pipeline": "OCR_CHUNKING_SERVICE",
        "stage": stage,
        "oid": int(request.oid),
        "oid_documento": int(request.oid),
        "file_name": file_name,
        "usuario_proceso": request.usuario_proceso,
        "job_proceso": request.job_proceso,
        "job_filde_id": request.job_filde_id,
        "queue_name": DEFAULT_QUEUE_NAME,
        "overwrite_enabled": request.overwrite.enabled,
        "metadata": request.metadata,
        "resolved_item": to_json_safe(item_info or {}),
        "created_at_utc": utc_now_iso(),
    }


def update_job_progress(
    db: PostgresClient,
    job_id: int,
    recorder: PhaseRecorder,
    pipeline_status: str,
    current_phase: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Stores progressive status in job.resultado."""
    payload = {
        "pipeline_status": pipeline_status,
        "current_phase": current_phase,
        "phases": recorder.as_list(),
        "extra": extra or {},
        "updated_at_utc": utc_now_iso(),
    }
    db.update_job_state(
        job_id=job_id,
        estado="EN_PROCESO",
        resultado=payload,
        set_inicio_if_null=True,
    )


def run_mock_pipeline(request: OCRChunkingRequest) -> OCRChunkingResponse:
    """Runs deterministic mock pipeline (no DB usage)."""
    recorder = PhaseRecorder()
    started = time.monotonic()
    latency = max(0, int(request.mock.latency_ms))
    fail_phase = safe_str(request.mock.fail_phase, "").strip().upper() if request.mock.fail_phase else None

    def phase(name: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        recorder.push(name, "OK", message, details or {})
        if latency > 0:
            time.sleep(float(latency) / 1000.0)
        if fail_phase and fail_phase == name:
            raise PipelineError(name, "MOCK_FORCED_FAILURE", f"Falla mock forzada en fase {name}.")

    try:
        phase("REQUEST_VALIDATION", "Mock request validada.")
        phase("QUEUE", "Mock cola aceptada.", {"simulated": True})
        phase("LOAD_BINARY", "Mock bytes cargados.", {"bytes": 1024})
        phase("PAGE_SELECTION", "Mock seleccion de paginas.", {"mode": request.extraction.page_mode})
        phase("PROBE", "Mock probe realizado.", {"extractable_confidence": 0.93, "is_text_extractable": True})
        phase("TEXT_EXTRACTION", "Mock extraccion completada.", {"engine": "pymupdf"})
        phase("TEXT_CLEANING", "Mock limpieza completada.", {"chars_after": 5000})
        phase("CHUNKING", "Mock chunking completado.", {"chunks": 8})
        phase("EMBEDDINGS", "Mock embeddings completados.", {"vectors": 8})
        phase("PERSIST", "Mock persistencia completada.", {"inserted_rows": 8})

        elapsed_ms = int((time.monotonic() - started) * 1000)
        return OCRChunkingResponse(
            status="COMPLETED",
            exitoso=True,
            message="Mock pipeline completado.",
            phases=recorder.as_list(),
            data={
                "oid": int(request.oid),
                "job_id": None,
                "engine_used": "pymupdf",
                "chunks_count": 8,
                "inserted_rows": 8,
                "elapsed_ms": elapsed_ms,
                "mock_mode": True,
            },
        )
    except PipelineError as exc:
        recorder.push(exc.phase, "ERROR", exc.message, exc.details)
        return OCRChunkingResponse(
            status="FAILED",
            exitoso=False,
            message="Mock pipeline fallido.",
            error=exc.to_dict(),
            phases=recorder.as_list(),
            data={"oid": int(request.oid), "mock_mode": True},
        )


def run_real_pipeline(request: OCRChunkingRequest, stage: str = "pipeline") -> OCRChunkingResponse:
    """Runs selected stage (ocr/chunking/embedding/pipeline) against DB and model runtime."""
    stage_key = safe_str(stage, "pipeline").strip().lower()
    if stage_key not in {"ocr", "chunking", "embedding", "pipeline"}:
        raise PipelineError(
            "REQUEST_VALIDATION",
            "INVALID_STAGE",
            "Etapa invalida. Usa: ocr, chunking, embedding o pipeline.",
            {"stage": stage},
        )

    run_chunking_stage = stage_key in {"chunking", "embedding", "pipeline"}
    run_embedding_stage = stage_key in {"embedding", "pipeline"} and request.embedding.enabled
    run_persist_stage = stage_key in {"embedding", "pipeline"} and request.embedding.save_to_db

    started = time.monotonic()
    recorder = PhaseRecorder()
    queue_slot_acquired = False
    queue_name = DEFAULT_QUEUE_NAME
    job_type = DEFAULT_JOB_TYPE if stage_key == "pipeline" else f"{DEFAULT_JOB_TYPE}_{stage_key.upper()}"
    job_id: Optional[int] = None
    item_info: Optional[Dict[str, Any]] = None
    file_name = safe_str(request.nombre_documento or request.file_name, "").strip()
    documento_id = int(request.oid)

    recorder.push("REQUEST_VALIDATION", "OK", "Request recibida.", {"oid": int(request.oid), "stage": stage_key})

    if run_persist_stage and not request.embedding.enabled:
        err = PipelineError(
            "REQUEST_VALIDATION",
            "INVALID_EMBEDDING_CONFIG",
            "embedding.save_to_db requiere embedding.enabled=true.",
        )
        recorder.push(err.phase, "ERROR", err.message, err.details)
        return OCRChunkingResponse(
            status="FAILED",
            exitoso=False,
            message="Configuracion invalida.",
            error=err.to_dict(),
            phases=recorder.as_list(),
            data={"oid": int(request.oid), "stage": stage_key},
        )

    try:
        with PostgresClient(PostgresSettings.from_env()) as db:
            if file_name:
                recorder.push(
                    "LOAD_ITEM",
                    "SKIPPED",
                    "Nombre de documento recibido en request; no se consulta nombre por OID.",
                    {"file_name": file_name},
                )
            else:
                item_info = db.fetch_item_by_oid(int(request.oid))
                if item_info is not None:
                    file_name = safe_str(item_info.get("nombre_archivo"), "").strip()
                    recorder.push(
                        "LOAD_ITEM",
                        "OK",
                        "Item resuelto por OID.",
                        {
                            "item_id": safe_int(item_info.get("item_id"), None),
                            "file_name": file_name,
                            "estado_item": safe_str(item_info.get("estado"), ""),
                        },
                    )
                if not file_name:
                    file_name = f"documento_{int(request.oid)}.pdf"
                    recorder.push(
                        "LOAD_ITEM",
                        "WARN",
                        "No se encontro nombre por OID; se usa nombre por defecto.",
                        {"file_name": file_name},
                    )

            if request.queue.enabled:
                queue_info = db.ensure_queue(
                    queue_name=queue_name,
                    description=f"Servicio {stage_key} OCR/Chunking/Embeddings",
                    max_concurrency=request.queue.max_concurrency,
                    timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
                    retries_max=3,
                    priority_default=40,
                )
                recorder.push("QUEUE_SETUP", "OK", "Queue asegurada.", {"queue": queue_info})
            else:
                recorder.push("QUEUE_SETUP", "SKIPPED", "Queue deshabilitada.")

            payload = build_job_payload(request, file_name, item_info, stage_key)

            if request.queue.enabled:
                slot = db.acquire_queue_slot(queue_name)
                queue_slot_acquired = bool(slot.get("acquired"))
                if not queue_slot_acquired:
                    if request.queue.queue_when_busy:
                        job_id = db.create_job(
                            job_type=job_type,
                            estado="PENDIENTE",
                            prioridad=40,
                            documento_id=documento_id,
                            parametros=payload,
                            max_intentos=3,
                        )
                        db.refresh_queue_stats(queue_name)
                        recorder.push(
                            "QUEUE_ADMISSION",
                            "ENQUEUED",
                            "Cola ocupada; job en PENDIENTE.",
                            {"job_id": job_id, "queue_name": queue_name},
                        )
                        return OCRChunkingResponse(
                            status="ENQUEUED",
                            exitoso=True,
                            message="Cola ocupada, job encolado.",
                            phases=recorder.as_list(),
                            data={
                                "oid": int(request.oid),
                                "stage": stage_key,
                                "job_id": job_id,
                                "queue_name": queue_name,
                                "queue_state": slot.get("queue"),
                            },
                        )
                    raise PipelineError(
                        "QUEUE_ADMISSION",
                        "QUEUE_BUSY",
                        "No hay slots disponibles y queue_when_busy=false.",
                        {"queue_name": queue_name},
                    )

                recorder.push(
                    "QUEUE_ADMISSION",
                    "OK",
                    "Slot de cola adquirido.",
                    {"queue_name": queue_name, "queue_state": slot.get("queue")},
                )

            job_id = db.create_job(
                job_type=job_type,
                estado="EN_PROCESO",
                prioridad=40,
                documento_id=documento_id,
                parametros=payload,
                max_intentos=3,
            )
            update_job_progress(db, job_id, recorder, "RUNNING", "QUEUE_ADMISSION", {"job_id": job_id})

            try:
                pdf_bytes = db.read_large_object(int(request.oid))
            except Exception as exc:
                raise PipelineError(
                    "LOAD_BINARY",
                    "OID_READ_FAILED",
                    "No fue posible leer el binary del OID solicitado.",
                    {"oid": int(request.oid), "error": f"{type(exc).__name__}: {str(exc)}"},
                ) from exc
            if not pdf_bytes:
                raise PipelineError("LOAD_BINARY", "EMPTY_BINARY", "Large object vacio.")
            binary_sha256 = hashlib.sha256(pdf_bytes).hexdigest()
            recorder.push(
                "LOAD_BINARY",
                "OK",
                "PDF cargado desde large object.",
                {"oid": int(request.oid), "bytes": len(pdf_bytes), "sha256": binary_sha256},
            )
            update_job_progress(db, job_id, recorder, "RUNNING", "LOAD_BINARY")

            selected_pdf, page_info = apply_page_selection(
                pdf_bytes=pdf_bytes,
                page_mode=request.extraction.page_mode,
                head_pages=request.extraction.head_pages,
                tail_pages=request.extraction.tail_pages,
            )
            recorder.push("PAGE_SELECTION", "OK", "Seleccion de paginas aplicada.", page_info)
            update_job_progress(db, job_id, recorder, "RUNNING", "PAGE_SELECTION")

            probe = probe_pdf_extractability(selected_pdf, request.extraction.probe_max_pages)
            recorder.push("PROBE", "OK", "Probe de extraibilidad completado.", probe)
            update_job_progress(db, job_id, recorder, "RUNNING", "PROBE")

            engine = resolve_extraction_engine(request.extraction, probe)
            extraction_meta: Dict[str, Any] = {"engine": engine}
            if engine == "pymupdf":
                extracted_text, extracted_pages = extract_text_pymupdf(selected_pdf)
            else:
                extracted_text, extracted_pages, docling_meta = extract_text_docling(selected_pdf, request.extraction)
                extraction_meta.update(docling_meta)
            if not extracted_text.strip():
                raise PipelineError(
                    "TEXT_EXTRACTION",
                    "EMPTY_EXTRACTED_TEXT",
                    "No se pudo extraer texto del PDF.",
                    {"engine": engine},
                )
            extraction_meta.update(
                {"chars": len(extracted_text), "pages_processed": extracted_pages, "engine_used": engine}
            )
            recorder.push("TEXT_EXTRACTION", "OK", "Extraccion completada.", extraction_meta)
            update_job_progress(db, job_id, recorder, "RUNNING", "TEXT_EXTRACTION")

            cleaned_text, cleaning_meta = clean_text(extracted_text, request.cleaning)
            text_for_chunking = cleaned_text if request.cleaning.enabled else extracted_text
            if not text_for_chunking.strip():
                raise PipelineError("TEXT_CLEANING", "EMPTY_AFTER_CLEANING", "Texto vacio despues de limpieza.")
            recorder.push("TEXT_CLEANING", "OK", "Limpieza aplicada.", cleaning_meta)
            update_job_progress(db, job_id, recorder, "RUNNING", "TEXT_CLEANING")

            chunks: List[str] = []
            bounds: List[Tuple[int, int]] = []
            chunking_method = "none"
            if run_chunking_stage:
                chunk_strategy = safe_str(request.chunking.strategy, "semantic").strip().lower()
                if chunk_strategy not in {"semantic", "simple"}:
                    raise PipelineError(
                        "CHUNKING",
                        "INVALID_CHUNKING_STRATEGY",
                        "chunking.strategy debe ser semantic o simple.",
                        {"strategy": request.chunking.strategy},
                    )
                chunking_method = chunk_strategy
                if chunk_strategy == "semantic":
                    tokenizer_for_chunk = load_tokenizer(request.embedding.model_name)
                    chunks = semantic_chunk_text(text_for_chunking, tokenizer_for_chunk)
                    if not chunks and request.chunking.enable_simple_fallback:
                        chunks = simple_chunk_text(
                            text_for_chunking,
                            request.chunking.simple_chunk_size,
                            request.chunking.simple_chunk_overlap,
                        )
                        chunking_method = "simple_fallback"
                else:
                    chunks = simple_chunk_text(
                        text_for_chunking,
                        request.chunking.simple_chunk_size,
                        request.chunking.simple_chunk_overlap,
                    )

                if request.chunking.max_chunks > 0:
                    chunks = chunks[: int(request.chunking.max_chunks)]
                chunks = [chunk for chunk in chunks if chunk.strip()]
                if not chunks:
                    raise PipelineError("CHUNKING", "EMPTY_CHUNKS", "No se generaron chunks.")
                if len(text_for_chunking) < int(request.chunking.min_text_chars):
                    recorder.push(
                        "CHUNKING_WARNING",
                        "WARN",
                        "Texto menor al umbral min_text_chars; se continua con los chunks disponibles.",
                        {
                            "min_text_chars": int(request.chunking.min_text_chars),
                            "text_chars": len(text_for_chunking),
                        },
                    )
                bounds = estimate_bounds(text_for_chunking, chunks)
                recorder.push(
                    "CHUNKING",
                    "OK",
                    "Chunking completado.",
                    {"strategy": chunking_method, "chunks_count": len(chunks)},
                )
                update_job_progress(db, job_id, recorder, "RUNNING", "CHUNKING")
            else:
                recorder.push("CHUNKING", "SKIPPED", "Etapa no solicitada para este endpoint.")

            vectors: List[List[float]] = []
            tokens_per_chunk: List[int] = [0 for _ in chunks]
            embedding_device = "none"
            if run_embedding_stage:
                tokenizer, model, device = load_embedding_model(request.embedding.model_name)
                derived_max_length = derive_embedding_max_length(request.chunking)
                vectors, tokens_per_chunk = embed_chunks(
                    chunks=chunks,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    max_length=derived_max_length,
                    batch_size=request.embedding.batch_size,
                )
                embedding_device = device
                if len(vectors) != len(chunks):
                    raise PipelineError(
                        "EMBEDDINGS",
                        "EMBEDDING_COUNT_MISMATCH",
                        "La cantidad de embeddings no coincide con chunks.",
                        {"chunks": len(chunks), "vectors": len(vectors)},
                    )
                recorder.push(
                    "EMBEDDINGS",
                    "OK",
                    "Embeddings generados.",
                    {"enabled": True, "device": embedding_device, "vectors": len(vectors)},
                )
                update_job_progress(db, job_id, recorder, "RUNNING", "EMBEDDINGS")
            else:
                recorder.push("EMBEDDINGS", "SKIPPED", "Etapa no solicitada para este endpoint.")

            inserted_rows = 0
            if run_persist_stage:
                existing_count = db.count_existing_embeddings(documento_id=documento_id)
                deleted_count = 0
                if existing_count > 0 and not request.overwrite.enabled:
                    raise PipelineError(
                        "OVERWRITE_CHECK",
                        "DUPLICATE_EMBEDDINGS",
                        "Ya existen embeddings para el documento y overwrite.enabled=false.",
                        {"oid": documento_id, "existing_count": existing_count},
                    )
                if existing_count > 0 and request.overwrite.enabled:
                    deleted_count = db.delete_existing_embeddings(documento_id=documento_id)

                recorder.push(
                    "OVERWRITE_CHECK",
                    "OK",
                    "Validacion de duplicados completada.",
                    {
                        "existing_count": existing_count,
                        "overwrite_enabled": bool(request.overwrite.enabled),
                        "deleted_count": deleted_count,
                    },
                )
                update_job_progress(db, job_id, recorder, "RUNNING", "OVERWRITE_CHECK")

                created_by = request.created_by if request.created_by is not None else request.embedding.created_by_default
                row_values: List[Tuple[Any, ...]] = []
                for idx, chunk in enumerate(chunks):
                    chunk_inicio, chunk_fin = bounds[idx]
                    vector_literal = vector_to_pg_literal(vectors[idx])
                    metadata_row = {
                        "oid": int(request.oid),
                        "file_name": file_name,
                        "usuario_proceso": request.usuario_proceso,
                        "job_proceso": request.job_proceso,
                        "phase": "PERSIST",
                        "item_id": safe_int(item_info.get("item_id"), None) if item_info else None,
                        "probe": probe,
                        "engine_used": engine,
                        "page_selection": page_info,
                        "chunk_strategy": chunking_method,
                        "chunk_chars": len(chunk),
                        "request_metadata": request.metadata,
                    }
                    row_values.append(
                        (
                            int(chunk_fin),
                            int(idx + 1),
                            int(chunk_inicio),
                            chunk,
                            int(created_by),
                            documento_id,
                            request.job_filde_id,
                            json_dumps_safe(metadata_row),
                            request.embedding.model_name,
                            int(tokens_per_chunk[idx]) if idx < len(tokens_per_chunk) else 0,
                            vector_literal,
                        )
                    )

                inserted_rows = db.insert_embeddings(row_values)
                if request.embedding.require_inserted_rows and inserted_rows <= 0:
                    raise PipelineError(
                        "PERSIST",
                        "NO_ROWS_INSERTED",
                        "No se insertaron embeddings y require_inserted_rows=true.",
                    )
                recorder.push(
                    "PERSIST",
                    "OK",
                    "Persistencia completada.",
                    {"inserted_rows": inserted_rows, "save_to_db": True},
                )
                update_job_progress(db, job_id, recorder, "RUNNING", "PERSIST")
            else:
                recorder.push("PERSIST", "SKIPPED", "Persistencia no solicitada para este endpoint.")

            elapsed_ms = int((time.monotonic() - started) * 1000)
            result_data = {
                "oid": int(request.oid),
                "stage": stage_key,
                "job_id": job_id,
                "file_name": file_name,
                "item_id": safe_int(item_info.get("item_id"), None) if item_info else None,
                "oid_documento": documento_id,
                "job_filde_id": request.job_filde_id,
                "usuario_proceso": request.usuario_proceso,
                "job_proceso": request.job_proceso,
                "queue_name": queue_name if request.queue.enabled else None,
                "binary_sha256": binary_sha256,
                "engine_used": engine,
                "probe": probe,
                "page_selection": page_info,
                "cleaning": cleaning_meta,
                "chunks_count": len(chunks),
                "inserted_rows": inserted_rows,
                "gpu": gpu_metrics(),
                "elapsed_ms": elapsed_ms,
            }
            if stage_key == "ocr":
                result_data["ocr_text_chars"] = len(text_for_chunking)
                result_data["ocr_preview"] = text_for_chunking[:500]
            if stage_key == "chunking":
                result_data["chunks_preview"] = chunks[:5]
            if request.embedding.return_vectors and vectors:
                result_data["vectors"] = vectors

            final_payload = {
                "pipeline_status": "COMPLETED",
                "current_phase": "DONE",
                "phases": recorder.as_list(),
                "result": result_data,
                "updated_at_utc": utc_now_iso(),
            }
            db.update_job_state(
                job_id=job_id,
                estado="COMPLETADO",
                resultado=final_payload,
                error_message=None,
                clear_error=True,
                set_fin=True,
            )

            return OCRChunkingResponse(
                status="COMPLETED",
                exitoso=True,
                message=f"Proceso '{stage_key}' completado exitosamente.",
                phases=recorder.as_list(),
                data=result_data,
            )

    except PipelineError as exc:
        LOGGER.error("PipelineError phase=%s code=%s message=%s", exc.phase, exc.code, exc.message)
        recorder.push(exc.phase, "ERROR", exc.message, exc.details)
        error_payload = exc.to_dict()
        if job_id is not None:
            try:
                with PostgresClient(PostgresSettings.from_env()) as db_error:
                    db_error.update_job_state(
                        job_id=job_id,
                        estado="ERROR",
                        resultado={
                            "pipeline_status": "FAILED",
                            "current_phase": exc.phase,
                            "phases": recorder.as_list(),
                            "error": error_payload,
                            "updated_at_utc": utc_now_iso(),
                        },
                        error_message=exc.message,
                        set_fin=True,
                    )
            except Exception:
                LOGGER.exception("No fue posible actualizar job en estado ERROR.")

        return OCRChunkingResponse(
            status="FAILED",
            exitoso=False,
            message="Proceso fallido.",
            error=error_payload,
            phases=recorder.as_list(),
            data={"oid": int(request.oid), "stage": stage_key, "job_id": job_id, "file_name": file_name},
        )

    except Exception as exc:
        LOGGER.exception("Error inesperado en pipeline.")
        unknown = PipelineError(
            "UNHANDLED_EXCEPTION",
            "UNEXPECTED_ERROR",
            f"{type(exc).__name__}: {str(exc)}",
            {"traceback": traceback.format_exc()},
        )
        recorder.push(unknown.phase, "ERROR", unknown.message, unknown.details)
        if job_id is not None:
            try:
                with PostgresClient(PostgresSettings.from_env()) as db_error:
                    db_error.update_job_state(
                        job_id=job_id,
                        estado="ERROR",
                        resultado={
                            "pipeline_status": "FAILED",
                            "current_phase": unknown.phase,
                            "phases": recorder.as_list(),
                            "error": unknown.to_dict(),
                            "updated_at_utc": utc_now_iso(),
                        },
                        error_message=unknown.message,
                        set_fin=True,
                    )
            except Exception:
                LOGGER.exception("No fue posible actualizar job en estado ERROR.")

        return OCRChunkingResponse(
            status="FAILED",
            exitoso=False,
            message="Proceso fallido por error inesperado.",
            error=unknown.to_dict(),
            phases=recorder.as_list(),
            data={"oid": int(request.oid), "stage": stage_key, "job_id": job_id, "file_name": file_name},
        )

    finally:
        if request.queue.enabled and queue_slot_acquired:
            try:
                with PostgresClient(PostgresSettings.from_env()) as db_final:
                    db_final.release_queue_slot(queue_name)
                    db_final.refresh_queue_stats(queue_name)
            except Exception:
                LOGGER.exception("No fue posible liberar/actualizar queue.")
        elif request.queue.enabled:
            try:
                with PostgresClient(PostgresSettings.from_env()) as db_final:
                    db_final.refresh_queue_stats(queue_name)
            except Exception:
                LOGGER.exception("No fue posible refrescar queue stats.")


def process_request(request: OCRChunkingRequest, stage: str = "pipeline") -> OCRChunkingResponse:
    """Selects mock or real pipeline execution by stage."""
    if request.mock.enabled:
        return run_mock_pipeline(request)
    return run_real_pipeline(request, stage=stage)

def process_batch(request: OCRChunkingBatchRequest, stage: str = "pipeline") -> OCRChunkingBatchResponse:
    """Processes a batch of requests sequentially or in parallel."""
    total = len(request.requests)
    results: List[Optional[OCRChunkingResponse]] = [None] * total

    if request.parallel_workers <= 1:
        for idx, item in enumerate(request.requests):
            results[idx] = process_request(item, stage=stage)
    else:
        with ThreadPoolExecutor(max_workers=request.parallel_workers) as pool:
            future_map = {pool.submit(process_request, item, stage): idx for idx, item in enumerate(request.requests)}
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = OCRChunkingResponse(
                        status="FAILED",
                        exitoso=False,
                        message="Error inesperado en ejecucion batch.",
                        error={
                            "phase": "BATCH",
                            "code": "BATCH_UNHANDLED_EXCEPTION",
                            "message": f"{type(exc).__name__}: {str(exc)}",
                        },
                        phases=[],
                        data={"oid": int(request.requests[idx].oid)},
                    )

    final_results = [r for r in results if r is not None]
    completed = sum(1 for r in final_results if r.exitoso)
    failed = total - completed
    overall_ok = failed == 0
    return OCRChunkingBatchResponse(
        status="COMPLETED" if overall_ok else "PARTIAL",
        exitoso=overall_ok,
        message="Batch procesado." if overall_ok else "Batch procesado con errores.",
        total=total,
        completados=completed,
        fallidos=failed,
        resultados=final_results,  # type: ignore[arg-type]
    )


def sample_request() -> Dict[str, Any]:
    """Returns canonical sample request."""
    model = OCRChunkingRequest(
        oid=2299268,
        nombre_documento="CTO_EyP_LLA_50_2013.pdf",
        file_name=None,
        job_filde_id=4567,
        usuario_proceso="analista_anh",
        job_proceso="JOB_OCR_20260309_001",
        created_by=1101,
        metadata={
            "nombre_documento": "CTO_EyP_LLA_50_2013.pdf",
            "metadata_documento": {"paginas": 133, "idioma": "es"},
            "ruta_pdf": "\\\\servidor\\share\\CTO_EyP_LLA_50_2013.pdf",
        },
        queue=QueueOptions(enabled=True, max_concurrency=2, queue_when_busy=True),
        overwrite=OverwriteOptions(enabled=False),
        extraction=ExtractionOptions(
            engine="auto",
            enable_pymupdf_fast_path=True,
            fast_path_confidence_threshold=0.85,
            page_mode="head_tail",
            head_pages=8,
            tail_pages=8,
        ),
        cleaning=CleaningOptions(enabled=True, remove_headers=True, remove_isolated_numbers=True),
        chunking=ChunkingOptions(strategy="semantic", max_chunks=0, enable_simple_fallback=False),
        embedding=EmbeddingOptions(enabled=True, model_name=DEFAULT_EMBEDDING_MODEL, save_to_db=True, return_vectors=False),
        mock=MockOptions(enabled=False),
    )
    return pydantic_model_dump(model)


app = FastAPI(
    title=SERVICE_NAME,
    version=SERVICE_VERSION,
    openapi_tags=OPENAPI_TAGS,
    description=(
        "Servicio OpenAPI para OCR, chunking y embeddings.\n"
        "Rutas funcionales: /ocr-docling, /chunking-docling, /embedding-generation, /PipelineOCR.\n"
        "Entrada obligatoria: oid."
    ),
)


@app.get("/health", tags=[TAG_HELPERS])
def health() -> Dict[str, Any]:
    """Health endpoint."""
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "timestamp_utc": utc_now_iso(),
        "gpu": gpu_metrics(),
    }


@app.get("/example-request", tags=[TAG_HELPERS])
def example_request() -> Dict[str, Any]:
    """Returns request payload example."""
    return {"input": sample_request()}


@app.get("/validate-db", tags=[TAG_HELPERS])
def validate_db() -> Dict[str, Any]:
    """
    Valida conexión a Postgres a nivel API.
    Retorna versión, metadatos y una consulta de prueba.
    """
    result = _validate_db_connection()
    result["timestamp"] = bogota_now_iso()
    result["service"] = SERVICE_NAME
    return to_json_safe(result)


@app.post("/ocr-docling/process", response_model=OCRChunkingResponse, tags=[TAG_OCR])
def ocr_docling_process(payload: Dict[str, Any]) -> OCRChunkingResponse:
    """Ejecuta solo OCR + limpieza de texto."""
    input_payload = payload.get("input", payload)
    try:
        request = OCRChunkingRequest(**input_payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Payload invalido: {str(exc)}") from exc
    return process_request(request, stage="ocr")


@app.post("/ocr-docling/process-batch", response_model=OCRChunkingBatchResponse, tags=[TAG_OCR])
def ocr_docling_batch(payload: Dict[str, Any]) -> OCRChunkingBatchResponse:
    """Ejecuta OCR + limpieza para varios documentos."""
    input_payload = payload.get("input", payload)
    try:
        request = OCRChunkingBatchRequest(**input_payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Payload batch invalido: {str(exc)}") from exc
    return process_batch(request, stage="ocr")


@app.post("/chunking-docling/process", response_model=OCRChunkingResponse, tags=[TAG_CHUNKING])
def chunking_docling_process(payload: Dict[str, Any]) -> OCRChunkingResponse:
    """Ejecuta OCR + limpieza + chunking."""
    input_payload = payload.get("input", payload)
    try:
        request = OCRChunkingRequest(**input_payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Payload invalido: {str(exc)}") from exc
    return process_request(request, stage="chunking")


@app.post("/chunking-docling/process-batch", response_model=OCRChunkingBatchResponse, tags=[TAG_CHUNKING])
def chunking_docling_batch(payload: Dict[str, Any]) -> OCRChunkingBatchResponse:
    """Ejecuta OCR + limpieza + chunking para varios documentos."""
    input_payload = payload.get("input", payload)
    try:
        request = OCRChunkingBatchRequest(**input_payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Payload batch invalido: {str(exc)}") from exc
    return process_batch(request, stage="chunking")


@app.post("/embedding-generation/process", response_model=OCRChunkingResponse, tags=[TAG_EMBEDDING])
def embedding_generation_process(payload: Dict[str, Any]) -> OCRChunkingResponse:
    """Ejecuta OCR + limpieza + chunking + generacion de embeddings."""
    input_payload = payload.get("input", payload)
    try:
        request = OCRChunkingRequest(**input_payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Payload invalido: {str(exc)}") from exc
    return process_request(request, stage="embedding")


@app.post("/embedding-generation/process-batch", response_model=OCRChunkingBatchResponse, tags=[TAG_EMBEDDING])
def embedding_generation_batch(payload: Dict[str, Any]) -> OCRChunkingBatchResponse:
    """Ejecuta generacion de embeddings para varios documentos."""
    input_payload = payload.get("input", payload)
    try:
        request = OCRChunkingBatchRequest(**input_payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Payload batch invalido: {str(exc)}") from exc
    return process_batch(request, stage="embedding")


@app.post("/PipelineOCR/process", response_model=OCRChunkingResponse, tags=[TAG_PIPELINE])
def pipeline_ocr_process(payload: Dict[str, Any]) -> OCRChunkingResponse:
    """Orquesta todo el flujo: OCR, limpieza, chunking, embeddings e insercion."""
    input_payload = payload.get("input", payload)
    try:
        request = OCRChunkingRequest(**input_payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Payload invalido: {str(exc)}") from exc
    return process_request(request, stage="pipeline")


@app.post("/PipelineOCR/process-batch", response_model=OCRChunkingBatchResponse, tags=[TAG_PIPELINE])
def pipeline_ocr_batch(payload: Dict[str, Any]) -> OCRChunkingBatchResponse:
    """Orquesta flujo completo para varios documentos."""
    input_payload = payload.get("input", payload)
    try:
        request = OCRChunkingBatchRequest(**input_payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Payload batch invalido: {str(exc)}") from exc
    return process_batch(request, stage="pipeline")


def run_mock_local_demo(args: argparse.Namespace) -> None:
    """Runs local mock example from CLI and prints JSON response."""
    request = OCRChunkingRequest(
        oid=int(args.mock_oid),
        mock=MockOptions(
            enabled=True,
            fail_phase=args.mock_fail_phase,
            latency_ms=int(args.mock_latency_ms),
            without_db=True,
        ),
    )
    result = process_request(request)
    print(pydantic_model_dump_json(result, indent=2))


def parse_args() -> argparse.Namespace:
    """Parses CLI args."""
    parser = argparse.ArgumentParser(description=SERVICE_NAME)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--mock-local", action="store_true", help="Runs mock pipeline once and exits.")
    parser.add_argument("--mock-oid", type=int, default=2299268)
    parser.add_argument("--mock-fail-phase", default=None)
    parser.add_argument("--mock-latency-ms", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    if args.mock_local:
        run_mock_local_demo(args)
        return
    uvicorn.run(app, host=args.host, port=args.port, reload=bool(args.reload))


if __name__ == "__main__":
    main()

