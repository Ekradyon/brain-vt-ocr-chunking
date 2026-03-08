# funcionalidades_parametro

Descripcion operativa de cada parametro del payload `input`.

> Nota: cuando un parametro no se envia, se usa su default.

## Parametros principales

## `oid`
- Tipo: `int` (obligatorio)
- Si se envia: busca ese `loOid` en `Operaciones.ItemsIngestaSmb` y procesa el PDF.
- Si no se envia: error `422` (validacion).
- Si no existe en BD: `FAILED` con `OID_NOT_FOUND`.

## `file_name`
- Tipo: `string` (opcional)
- Si se envia: se usa para trazabilidad y metadata.
- Si no se envia: se intenta resolver desde `nombreArchivo` por `loOid`.

## `documento_id`
- Tipo: `int` (opcional)
- Si se envia: permite persistencia en `IaCore.Embeddings`.
- Si no se envia y `embedding.save_to_db=true` + `require_documento_id=true`: `FAILED` con `DOCUMENTO_ID_REQUIRED`.

## `job_filde_id`
- Tipo: `int` (opcional)
- Uso: trazabilidad de origen de job/archivo; se guarda en `jobFileId` en embeddings.
- Si no se envia: se inserta `NULL` en `jobFileId`.

## `created_by`
- Tipo: `int` (opcional)
- Si se envia: se usa como `createdBy` de embeddings.
- Si no se envia: usa `embedding.created_by_default`.

## `metadata`
- Tipo: `objeto` (opcional)
- Uso recomendado:
  - `nombre_documento`
  - `metadata_documento` (ej: `paginas`, `idioma`, `tipo`)
  - `ruta_pdf`
- Se agrega al metadata de salida y a los registros de embeddings.

## queue

## `queue.enabled`
- `true`: habilita control de cola (ensure, acquire, release, stats).
- `false`: ejecuta directo sin semaforo de cola.

## `queue.max_concurrency`
- Entero 1..64.
- Define la concurrencia maxima de la cola interna del servicio.

## `queue.queue_when_busy`
- `true`: si no hay slot disponible, retorna `ENQUEUED` y crea job `PENDIENTE`.
- `false`: si no hay slot disponible, retorna `FAILED` con `QUEUE_BUSY`.

## overwrite

## `overwrite.enabled`
- `true`: borra embeddings existentes del `documento_id` antes de insertar.
- `false`: si ya existen embeddings del documento, falla con `DUPLICATE_EMBEDDINGS`.

> Overwrite es solo a nivel documento (no hay scope).

## extraction

## `extraction.engine`
- `auto`: decide entre `pymupdf` y `docling` segun confianza.
- `pymupdf`: fuerza extraccion por PyMuPDF.
- `docling`: fuerza OCR Docling.

## `extraction.enable_pymupdf_fast_path`
- `true`: en `auto`, usa PyMuPDF cuando `extractable_confidence` supera umbral.
- `false`: en `auto`, usa Docling.

## `extraction.fast_path_confidence_threshold`
- Rango `0..1`.
- Umbral para considerar el PDF como texto extraible con alta confianza.

## `extraction.force_full_page_ocr`
- `true`: prioriza OCR completo por Docling.
- `false`: usa estrategia normal segun `engine`.

## `extraction.page_mode`
- `full`: procesa todo el documento.
- `head_tail`: procesa primeras N y ultimas N paginas.

## `extraction.head_pages`, `extraction.tail_pages`
- Usados solo cuando `page_mode=head_tail`.

## `extraction.probe_max_pages`
- Numero maximo de paginas para calcular confianza de extraibilidad.

## cleaning

## `cleaning.enabled`
- `true`: aplica limpieza previa al chunking.
- `false`: usa texto bruto extraido.

## `cleaning.remove_headers`
- `true`: elimina encabezados repetidos segun `header_threshold`.
- `false`: conserva encabezados.

## `cleaning.remove_isolated_numbers`
- `true`: elimina lineas con solo numeros/simbolos numericos.
- `false`: conserva lineas numericas sueltas.

## `cleaning.header_threshold`
- Frecuencia minima para considerar una linea como encabezado repetido.

## chunking

## `chunking.strategy`
- `semantic`: usa `HybridChunker` de Docling.
- `simple`: usa ventanas por caracteres.

## `chunking.max_chunks`
- `0`: sin limite.
- `>0`: limita el numero final de chunks.

## `chunking.simple_chunk_size`
- Tamano del chunk por caracteres para estrategia `simple`.
- Tambien se usa para derivar `max_length` interno de embeddings.

## `chunking.simple_chunk_overlap`
- Solape de caracteres entre chunks en estrategia `simple`.

## `chunking.min_text_chars`
- Umbral de advertencia; si texto queda por debajo, se emite warning y continua.

## `chunking.enable_simple_fallback`
- `true`: si semantic no produce chunks, intenta simple.
- `false`: si semantic no produce chunks, falla con `EMPTY_CHUNKS`.

## embedding

## `embedding.enabled`
- `true`: genera embeddings.
- `false`: no genera embeddings (util para endpoints OCR/Chunking).

## `embedding.model_name`
- Nombre del modelo HuggingFace usado para embeddings.

## `embedding.batch_size`
- Cantidad de chunks por lote al generar embeddings.
- `batch_size` alto: mas velocidad potencial, mayor uso de memoria GPU/CPU.
- `batch_size` bajo: menor uso de memoria, mayor tiempo total.

## `embedding.save_to_db`
- `true`: inserta embeddings en `IaCore.Embeddings`.
- `false`: no inserta (solo calcula si aplica).

## `embedding.return_vectors`
- `true`: incluye vectores en la respuesta HTTP.
- `false`: no devuelve vectores (respuesta mas liviana).

## `embedding.require_documento_id`
- `true`: exige `documento_id` para persistencia.
- `false`: no lo exige (no recomendado para insercion real).

## `embedding.require_inserted_rows`
- `true`: falla si no se inserta ninguna fila.
- `false`: permite continuar aunque no haya insercion.

## `embedding.created_by_default`
- Valor fallback para `createdBy` si `created_by` no se envio.

## mock

## `mock.enabled`
- `true`: ejecuta pipeline simulado sin DB real.
- `false`: ejecuta pipeline real.

## `mock.fail_phase`
- Si se define, fuerza error en la fase indicada (solo mock).

## `mock.latency_ms`
- Agrega latencia artificial por fase (solo mock).

## `mock.without_db`
- Indicador documental de ejecucion sin DB en modo mock.

## Estado y errores de salida

- `status`: `COMPLETED`, `ENQUEUED`, `FAILED`.
- `error` incluye:
  - `phase`
  - `code`
  - `message`
  - `details`

Codigos frecuentes:
- `OID_NOT_FOUND`
- `QUEUE_BUSY`
- `DOCUMENTO_ID_REQUIRED`
- `DUPLICATE_EMBEDDINGS`
- `EMPTY_EXTRACTED_TEXT`
- `EMPTY_CHUNKS`
- `NO_ROWS_INSERTED`
