# funcionalidades_parametro

Descripcion de que hace cada parametro en el payload `input`.

## Parametros principales

## `oid` (obligatorio)
- Tipo: `int`.
- Funcion: identificador LOID para reconstruir el PDF desde `pg_largeobject`.
- Regla: el servicio intenta resolver `documento_id` en `GestorDocumental.Documentos` por:
  1) `metadatosExtra.ocr.metadata.oid`
  2) fallback por `archivoNombre`.
- Si no logra resolver `documento_id`, persiste con `documentoId=null` (si el esquema lo permite) y deja traza en fases.
- Si no se envia: `422`.
- Nota operativa: cuando se resuelve `documento_id`, el OCR limpio se escribe en `GestorDocumental.Documentos.contenidoTexto`.
- Al finalizar OCR se actualiza `ocrAplicado=true` y `estado=EN_PROCESAMIENTO`.
- Al finalizar embeddings/chunking (etapas con embeddings) se actualiza `embeddingGenerado=true`, `estado=PROCESADO` y `metadatosExtra`.

## `nombre_documento`
- Tipo: `string`.
- Si se envia: se usa ese nombre y NO se busca nombre por OID en `ItemsIngestaSmb`.
- Si no se envia: se intenta obtener `nombreArchivo` por OID; si no existe, usa nombre por defecto.

## `file_name`
- Tipo: `string` (compatibilidad).
- Se usa como alias de `nombre_documento` si este no viene.

## `job_filde_id`
- Tipo: `int` (opcional).
- Se guarda como `jobFileId` en embeddings para trazabilidad.

## `usuario_proceso`
- Tipo: `string`.
- Usuario funcional que lanza el proceso; se guarda en metadata de salida/embedding.

## `job_proceso`
- Tipo: `string`.
- Nombre/identificador funcional del job de negocio; se guarda en metadata.

## `created_by`
- Tipo: `int`.
- Usuario tecnico para columna `createdBy` en embeddings.
- Si no se envia: usa `embedding.created_by_default`.

## `metadata`
- Tipo: `objeto`.
- Uso recomendado: metadatos adicionales de negocio.
- Nota: la metadata tecnica principal (OID, stats LO, resolucion de documento, engine, pages/chunks) la calcula el servicio.

## queue

## `queue.enabled`
- `true`: aplica cola y control de slots.
- `false`: corre sin semaforo de cola.

## `queue.max_concurrency`
- Entero de 1 a 64.
- Define la concurrencia maxima de la cola.

## `queue.queue_when_busy`
- `true`: si la cola esta ocupada, retorna `ENQUEUED`.
- `false`: si la cola esta ocupada, falla con `QUEUE_BUSY`.

## overwrite

## `overwrite.enabled`
- `true`: borra embeddings existentes del `documento_id` resuelto e inserta de nuevo.
- `false`: si existen embeddings del `documento_id` resuelto, falla con `DUPLICATE_EMBEDDINGS`.
- Si no hay `documento_id` resuelto: no hay validacion/borrado por documento y se registra warning.

> Overwrite es unicamente a nivel documento.

## extraction

## `extraction.engine`
- `auto`: decide PyMuPDF o Docling segun confianza.
- `pymupdf`: fuerza PyMuPDF.
- `docling`: fuerza OCR Docling.

## `extraction.enable_pymupdf_fast_path`
- `true`: habilita ruta rapida PyMuPDF en `auto`.
- `false`: en `auto`, usa Docling.

## `extraction.fast_path_confidence_threshold`
- Umbral `0..1` para decidir fast path.

## `extraction.force_full_page_ocr`
- `true`: prioriza OCR completo Docling.
- `false`: usa flujo normal.

## `extraction.page_mode`
- `full`: procesa todo el documento.
- `head_tail`: procesa primeras y ultimas paginas.

## `extraction.head_pages`, `extraction.tail_pages`
- Cantidad de paginas para `head_tail`.

## cleaning

## `cleaning.enabled`
- `true`: limpia texto antes de chunking.
- `false`: usa texto crudo.

## `cleaning.remove_headers`
- `true`: intenta remover encabezados repetidos.

## `cleaning.remove_isolated_numbers`
- `true`: elimina lineas solo numericas.

## `cleaning.remove_noisy_sentences`
- `true`: elimina oraciones OCR con exceso de tokens cortos sin valor semantico.

## `cleaning.noisy_min_alpha_tokens`, `cleaning.noisy_short_token_ratio`
- Controlan el umbral para detectar ruido OCR.

## `cleaning.header_threshold`
- Frecuencia para considerar encabezado repetido.

## chunking

## `chunking.strategy`
- `semantic` o `simple`.

## `chunking.max_chunks`
- `0`: sin limite.
- `>0`: limita total de chunks.

## `chunking.simple_chunk_size`
- Tamano por caracteres para estrategia simple.
- Tambien ayuda a derivar `max_length` interno para embeddings.

## `chunking.simple_chunk_overlap`
- Solape entre chunks en estrategia simple.

## `chunking.min_text_chars`
- Si el texto queda por debajo, emite warning y continua.

## `chunking.enable_simple_fallback`
- `true`: si semantic falla, usa simple.
- `false`: si semantic falla, responde `EMPTY_CHUNKS`.

## embedding

## `embedding.enabled`
- `true`: genera embeddings.
- `false`: no genera embeddings.

## `embedding.model_name`
- Modelo de embeddings.

## `embedding.batch_size`
- Numero de chunks por lote durante embedding.
- Alto: mas rapido potencialmente, mayor memoria.
- Bajo: menor memoria, mas tiempo.

## `embedding.save_to_db`
- `true`: inserta en `IaCore.Embeddings`.
- `false`: no inserta.

## `embedding.return_vectors`
- `true`: retorna vectores en respuesta.
- `false`: no retorna vectores.

## `embedding.require_inserted_rows`
- `true`: falla si no inserta filas.
- `false`: permite respuesta exitosa sin insercion.

## `embedding.created_by_default`
- Fallback de `created_by`.

## mock

## `mock.enabled`
- `true`: simula flujo sin DB real.
- `false`: flujo real.

## `mock.fail_phase`
- Fuerza error en la fase indicada (solo mock).

## `mock.latency_ms`
- Latencia artificial por fase (solo mock).

## Estado y errores de salida

- `status`: `COMPLETED`, `ENQUEUED`, `FAILED`.
- Errores HTTP: la API retorna `403` con detalle completo cuando hay falla de request o pipeline.
- `error`: `{ phase, code, message, details }`.
- En `data.ocr_confidence` se retorna confianza OCR completa (Docling o PyMuPDF probe).
- Codigos frecuentes:
  - `OID_READ_FAILED`
  - `QUEUE_BUSY`
  - `DUPLICATE_EMBEDDINGS`
  - `EMPTY_EXTRACTED_TEXT`
  - `EMPTY_CHUNKS`
  - `NO_ROWS_INSERTED`

