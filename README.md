# Databricks AI Functions — Emulators & Shims

This repo recreates Databricks' managed AI building blocks — the `ai_parse_document`
and `ai_prep_search` SQL/Python functions and the managed embedding endpoints that
power Vector Search — on workspaces where they don't exist natively, most notably
**GovCloud**. Each notebook packages the missing capability as a customer-managed
MLflow model behind a Model Serving endpoint, exposing the same interfaces so existing
parse → prep → index → query pipelines keep working unchanged.

## Why "emulators" and "shims"?

The two framings describe the same artifacts from different angles:

- **Shim** — a shim intercepts an expected API call and adapts it to an alternative
  backend. Native `ai_prep_search` / text-splitting workflows need a functional
  embedding endpoint behind the scenes; in GovCloud, managed endpoints like
  `databricks-bge-large-en` don't exist, so the shim would have nothing to target.
  Deploying an OSS embedding model to Model Serving is the prerequisite infrastructure
  that gives the shim an endpoint to talk to. If `ai_prep_search` is the adapter plug,
  the OSS Model Serving deployment is the generator you brought along to supply the
  electricity.

- **Emulator** — an emulator fully duplicates the behavior of a missing system. In
  commercial Databricks, managed embedding endpoints exist under `system.ai` / FMAPI
  as a zero-setup, pay-per-token service. Here, customer-managed Model Serving
  *emulates* that missing managed platform capability, exposing the same standard,
  OpenAI-compatible interface (`/serving-endpoints/{name}/invocations`).

The repo doesn't just emulate the high-level SQL/Python functions — it emulates the
**entire vertical stack**: the high-level functions *plus* the managed model endpoints
that power them.

## Notebooks

### `ai-parse-document-fmapi.ipynb` — `ai_parse_document` 2.0 emulator

Builds and deploys an MLflow pyfunc model that reproduces `ai_parse_document` 2.0 using
Databricks Foundation Model APIs.

- PDF pages are rasterized locally (`pypdfium2`) and sent as base64 images to a
  **vision endpoint**, which extracts OCR text spans with bounding boxes and region
  captions.
- A **reformat endpoint** groups and classifies the OCR spans into the
  `ai_parse_document` 2.0 element taxonomy (`text`, `table`, `figure`, `title`,
  `section_header`, …).
- **Deterministic finalization** replaces model-controlled IDs/pages/metadata with
  authoritative values and validates the full envelope against a strict Pydantic
  schema (`extra="forbid"`). If the LLM is unavailable, a schema-valid spans-only
  fallback is returned.
- Served behind a **CPU** Model Serving endpoint — all inference is remote FMAPI, so no
  GPU is needed. Output is the `ai_parse_document` 2.0 JSON string in a `response`
  column.
- Includes smoke tests via `requests` and via SQL `ai_query`.

### `ai-prep-search-fmapi.ipynb` — `ai_prep_search` emulator

Builds and deploys an MLflow pyfunc model that recreates `ai_prep_search` using FMAPI.

- Accepts structured `ai_parse_document` 2.0 JSON (or plain text/markdown), performs
  semantic chunking, and enriches each chunk with document context, table summaries,
  and related questions via a configurable LLM (`CHUNKING_MODEL`, default
  `databricks-gemini-3-8-flash`).
- Produces output compatible with Databricks AI / Vector Search indexes
  (`chunk_id`, `chunk_position`, `chunk_to_retrieve`, `chunk_to_embed`, `metadata`,
  `source_uri`).
- Served behind a **CPU** Model Serving endpoint.
- Includes smoke tests, helpers to flatten the JSON envelope (Pandas and Spark
  `variant_explode`) into a Delta table for Vector Search, an **end-to-end
  parse → prep → search** pipeline (Python SDK and batch SQL `ai_query`), and a
  **built-in vs custom** side-by-side comparison on the same document.

### `oss-embedding-embeddinggemma-2.ipynb` — managed embedding endpoint emulator

Deploys the open-source `google/embeddinggemma-2` embedding model to Model Serving so
Vector Search (and the prep-search shim) has an embedding backend to target.

- Downloads from Hugging Face Hub, registers to Unity Catalog via MLflow, and serves it.
- Logged with the `mlflow.sentence_transformers` flavor, producing embeddings in the
  standard format expected by Databricks Vector Search managed embedding indexes.
- Text-only config: 270M parameters (vision/audio encoders disabled), 768-dimensional
  embeddings, 8,192-token context, Apache 2.0 license.
- Served on a **GPU_SMALL** endpoint (1× NVIDIA T4). Loaded in float32 — float16 yields
  NaNs/degraded embeddings and the T4 has no native bfloat16 support. Cold start can
  take 10–15 minutes.

## Prerequisites

- A Databricks workspace with Model Serving and Unity Catalog, and Foundation Model API
  endpoints available for the vision, reformat, and chunking models.
- CLI or notebook access with privileges to register models (`USE CATALOG`,
  `USE SCHEMA`, `CREATE MODEL`/`CREATE TABLE`) and create serving endpoints.
- For the embedding notebook: a GPU_SMALL-capable workspace and (optionally) a Hugging
  Face token stored in a Databricks secret scope.

## Usage

1. Open a notebook in your Databricks workspace.
2. Update the configuration cell — set `CATALOG`, `SCHEMA`, and the relevant endpoint /
   model names (`VISION_ENDPOINT`, `REFORMAT_ENDPOINT`, `CHUNKING_MODEL`, etc.) to point
   at your own Unity Catalog location and preferred FMAPI models.
3. Run top to bottom to install dependencies, log and register the model, create the
   serving endpoint, and run the smoke tests.
4. Call the resulting endpoints from SQL (`ai_query`), the Python SDK
   (`serving_endpoints.query`), or any OpenAI-compatible client, and wire them into your
   parse → prep → index → query pipelines.

### GovCloud note

The intended environment is AWS GovCloud, where managed AI functions and FMAPI system
endpoints aren't available. On GovCloud you may need to set the S3 region before
registering models — the notebooks include a commented
`os.environ["AWS_DEFAULT_REGION"] = "us-gov-west-1"` line in the relevant config cells.
