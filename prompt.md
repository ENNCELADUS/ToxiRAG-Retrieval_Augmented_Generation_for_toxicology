Awesome — here’s a **from-top-to-bottom prompt pack** you can paste into a code LLM (GPT-5, Gemini 2.5 Pro, etc.) to generate the full project. Each step tells the model **what file(s) to create**, **requirements**, **APIs to expose**, and **acceptance criteria**. Use them **in order**. When a prompt says “Output: file content only,” ask the LLM to print just the file text (no commentary), then save it locally at the given path.

---

## 0) Master conventions (paste once at the start of your session)

**Prompt 0 — “Project Conventions & Style”**

> You are a senior Python engineer. We will build a production-grade RAG system for **AI-assisted animal experiment prediction** focused on **liver cancer** and **TCM compounds**.
> **Tech stack**: Python 3.11+, FastAPI, Qdrant, OpenAI (GPT-5) or Google (Gemini 2.5 Pro), Pydantic v2, pytest, pydantic-settings, uvicorn, httpx, python-dotenv, loguru, ragas (for evaluation).
> **Principles**:
>
> * Use **type hints** & **docstrings** everywhere.
> * Use **Pydantic models** for all external I/O schemas.
> * Separate **infra adapters** (LLM, embeddings, vector DB) from **domain logic**.
> * All modules must be **import-safe** (no side effects at import).
> * Use **structured logging** via loguru; never `print`.
> * Make files **idempotent** and **unit-testable**.
> * Prefer **pure functions**; keep state at the edges.
> * Return **JSON-serializable** outputs.
> * All configs read via **pydantic-settings** from `.env`.
> * Don’t hardcode API keys; use env vars.
> * When asked to “Output: file content only,” return the exact file body.
>
> Folder layout (we will create files step by step):
>
> ```
> ./
>   app/
>     __init__.py
>     config.py
>     logging.py
>     schemas.py
>     constants.py
>     utils_text.py
>   data/
>     samples/  # small example JSONs
>   ingest/
>     __init__.py
>     embeddings.py
>     vectorstore_qdrant.py
>     ingest_json.py
>   retriever/
>     __init__.py
>     retriever.py
>   llm/
>     __init__.py
>     openai_client.py
>     gemini_client.py
>     prompts.py
>     agentic_pipeline.py
>   api/
>     __init__.py
>     server.py
>   eval/
>     __init__.py
>     eval_ragas.py
>   tests/
>     test_ingest.py
>     test_retriever.py
>     test_pipeline.py
>   docker/
>     docker-compose.qdrant.yml
>   scripts/
>     run_ingest.sh
>     demo_query.sh
>   .env.example
>   pyproject.toml
>   requirements.txt
>   Makefile
>   README.md
> ```
>
> If any dependency is missing, add it to `requirements.txt`. We will now create files from top to bottom.

---

## 1) Project metadata, config, and README

**Prompt 1 — `pyproject.toml`**

> Create `./pyproject.toml` for a Python 3.11 project. Use `setuptools` build backend. Specify package `.` with src-layout disabled (root package under `./`). Include metadata (name, version, authors), and tool configs for `ruff` (lint), `pytest`, and `mypy` (strict optional).
> Output: file content only.

**Prompt 2 — `requirements.txt`**

> Create `./requirements.txt` with exact, recent stable pins for:
> `fastapi, uvicorn[standard], pydantic, pydantic-settings, qdrant-client, loguru, httpx, python-dotenv, tenacity, ragas, numpy, scipy, scikit-learn, rich, orjson, pytest, pytest-asyncio, typing-extensions`. Also add stubs as needed.
> Output: file content only.

**Prompt 3 — `.env.example`**

> Create `./.env.example` with:
>
> ```
> OPENAI_API_KEY=
> GOOGLE_API_KEY=
> LLM_PROVIDER=openai   # openai | gemini
> OPENAI_MODEL=gpt-5
> GEMINI_MODEL=gemini-2.5-pro
> EMBEDDING_PROVIDER=openai   # openai | gemini
> OPENAI_EMBED_MODEL=text-embedding-3-large
> GEMINI_EMBED_MODEL=text-embedding-004
> QDRANT_URL=http://localhost:6333
> QDRANT_API_KEY=
> COLLECTION_NAME=tcm_tox
> ```
>
> Output: file content only.

**Prompt 4 — `README.md`**

> Create `./README.md` explaining:
>
> * Project goal (TCM/liver-cancer toxicity prediction & experiment planning).
> * Architecture overview (ingest → index → retrieval → agentic LLM → API).
> * Quickstart: create venv, `pip install -r requirements.txt`, start Qdrant via docker compose, set `.env`, run `scripts/run_ingest.sh`, start API via `uvicorn api.server:app`.
> * How to switch providers (OpenAI/Gemini) via `.env`.
> * Data schema and sample JSONs in `data/samples/`.
>   Output: file content only.

---

## 2) Core app utilities

**Prompt 5 — `app/config.py`**

> Create `./app/config.py` using `pydantic-settings` to read all env vars from `.env` (fields from `.env.example`). Include computed properties like `is_openai`, `is_gemini`.
> Output: file content only.

**Prompt 6 — `app/logging.py`**

> Create `./app/logging.py` configuring loguru for JSON-ready formatting with request-id support. Provide `init_logging()` and `get_logger(name)` helpers.
> Output: file content only.

**Prompt 7 — `app/constants.py`**

> Create `./app/constants.py` holding fixed strings/enums: species defaults, supported routes (`PO`, `IV`, etc.), default chunk sizes, and retrieval constants (`TOP_K=12, MMR_LAMBDA=0.5`).
> Output: file content only.

**Prompt 8 — `app/schemas.py`**

> Create `./app/schemas.py` (Pydantic v2) defining:
>
> * `CompoundIngredient` (name\_cn, name\_en, smiles|None).
> * `CompoundRecord` (type, name, aliases, ingredients, mixture\_note).
> * `AnimalModel` (species, strain, sex, n\_per\_group, dose:{route,value\_g\_per\_kg,times\_per\_day,days}, env).
> * `ToxResults` (ld50|None, noael\_g\_per\_kg|None, clinical\_signs\:list, biochem\:dict, organs\:dict).
> * `PaperDoc` (paper\_id, title, year, access, compound\:CompoundRecord, animal\_model\:AnimalModel, tox\_results\:ToxResults, figures\:list, labels\:list, language\:str).
> * Retrieval outputs: `RetrievedChunk`, `RetrievedContext`.
> * API input: `PredictionRequest` containing compound name/SMILES, optional constraints (route/animal), and free-text context.
> * API output: `PredictionResponse` with structured `protocol_plan`, `predicted_outcomes`, `evidence_map`, `uncertainties`.
>   Output: file content only.

**Prompt 9 — `app/utils_text.py`**

> Create `./app/utils_text.py` utilities: safe JSON dumps (orjson), text normalization, simple tokenizer for chunking by sentences, and a unit conversion helper (mg/kg↔g/kg).
> Output: file content only.

---

## 3) Qdrant & embeddings & ingestion

**Prompt 10 — `docker/docker-compose.qdrant.yml`**

> Create a minimal compose to run Qdrant on `localhost:6333` with a named volume and no auth by default. Include commented lines for API key enablement.
> Output: file content only.

**Prompt 11 — `ingest/embeddings.py`**

> Create `./ingest/embeddings.py` that exposes:
>
> * `embed_texts(texts: list[str]) -> list[list[float]]` selecting provider from config.
> * OpenAI path: use `openai` client & `OPENAI_EMBED_MODEL`.
> * Gemini path: use `google.genai` client & `GEMINI_EMBED_MODEL`.
> * Implement **retry with tenacity**, chunking (max batch size 64), and **input length guard**.
>   Output: file content only.

**Prompt 12 — `ingest/vectorstore_qdrant.py`**

> Create `./ingest/vectorstore_qdrant.py`:
>
> * `QdrantStore` class with `create_or_recreate(collection_name, dim, distance="COSINE")`, `upsert(points)`, `search(query_vec, top_k, filters=None)`, `hybrid_search(query_vec, keywords, top_k)` and `mmr(postings, lambda_)`.
> * Wrap `qdrant_client` and expose typed payloads (`paper_id`, `section`, `text`, `meta`).
>   Output: file content only.

**Prompt 13 — `ingest/ingest_json.py`**

> Create an ingest script reading JSON files in `data/samples/` (matching `PaperDoc` schema). Build **chunk records** per logical section (animal\_model, tox\_results.biochem, organs, protocol). For each chunk:
>
> * Build textual representation;
> * Call `embed_texts`;
> * Upsert to Qdrant (payload stores `paper_id`, `title`, `year`, `labels`, `section`).
>   Provide a `main()` CLI with `--recreate` and `--collection` flags.
>   Output: file content only.

**Prompt 14 — `scripts/run_ingest.sh`**

> Bash script to run Qdrant (compose up), export `.env`, and run `python -m ingest.ingest_json --recreate`.
> Output: file content only.

---

## 4) Retrieval

**Prompt 15 — `retriever/retriever.py`**

> Create `./retriever/retriever.py` implementing:
>
> * `build_subqueries(compound_name, smiles, constraints) -> list[str]` that expands to related keys (“NOAEL”, “Kunming”, “灌胃”, etc.).
> * `retrieve_contexts(query, top_k=12) -> list[RetrievedChunk]` using hybrid search (vector + keywords) then MMR去冗余。
> * `format_contexts(chunks) -> str` to assemble evidence blocks with `(#i)` tags.
>   Add docstrings and unit-testable pure functions.
>   Output: file content only.

---

## 5) LLM clients & prompts

**Prompt 16 — `llm/openai_client.py`**

> Create `./llm/openai_client.py` with a thin wrapper:
>
> * `generate(text: str, temperature=0.2, max_tokens=1200) -> str` using **Responses API** style if available, otherwise fallback to Chat Completions gracefully.
> * Handle **rate limit** & **transient** errors via tenacity.
>   Output: file content only.

**Prompt 17 — `llm/gemini_client.py`**

> Create `./llm/gemini_client.py` with:
>
> * `generate(text: str, temperature=0.2, max_output_tokens=1200) -> str` using `google.genai`.
> * Same retry & error handling semantics as OpenAI client.
>   Output: file content only.

**Prompt 18 — `llm/prompts.py`**

> Create `./llm/prompts.py` exporting two template builders:
>
> * `build_protocol_prompt(compound, smiles, constraints, contexts_text) -> str`
> * `build_prediction_instruction() -> str`
>   The final prompt must request:
>
> 1. 实验设计（分组、n、剂量/频次/周期、终点），
> 2. 关键观测指标与采样时间点，
> 3. 结果预测（NOAEL/LD50范围、临床症状、器官与血生化的变化），
> 4. 证据映射（每条建议后的证据#编号），
> 5. 不确定性与下一步数据需求；
>    并要求 **严格输出 JSON**，字段：`protocol_plan, predicted_outcomes, evidence_map, uncertainties, confidence`.
>    Output: file content only.

---

## 6) Agentic pipeline (reason-then-answer)

**Prompt 19 — `llm/agentic_pipeline.py`**

> Create `./llm/agentic_pipeline.py` exposing:
>
> * `predict_experiment(compound_name:str, smiles:str|None, constraints:dict|None) -> dict`
>   Pipeline:
>
> 1. **Query Reformulation** → subqueries。
> 2. **Hybrid Retrieval** → top-k chunks → MMR。
> 3. **Prompt Compose**（system + user）→ JSON schema instructions。
> 4. 调用 LLM（根据配置走 openai 或 gemini）。
> 5. **JSON 解析**与字段校验（Pydantic）。
> 6. **Self-check**：若 `evidence_map` 未覆盖关键要点（剂量、物种、给药途径），追加一次“self-consistency”提示进行轻量重试。
>    Return Python `dict` matching `PredictionResponse`.
>    Output: file content only.

---

## 7) FastAPI service & demo scripts

**Prompt 20 — `api/server.py`**

> Create a FastAPI app:
>
> * `POST /v1/predict` with body `PredictionRequest`; returns `PredictionResponse`.
> * Health check `GET /healthz`.
> * On startup, init logging & config; lazy-init vector store.
> * Use `orjson` response class.
>   Output: file content only.

**Prompt 21 — `scripts/demo_query.sh`**

> Bash script: reads `.env`, sends a sample JSON (compound name+SMILES) to `http://127.0.0.1:8000/v1/predict` via `curl`, pretty-prints output with `jq`.
> Output: file content only.

---

## 8) RAG evaluation

**Prompt 22 — `eval/eval_ragas.py`**

> Implement RAGAS evaluation:
>
> * Load a small eval set from `data/samples/eval.jsonl` with `{"question","answer_gt","contexts_gt"}`.
> * For each question, run pipeline to get `answer_pred` and `contexts_used`.
> * Compute `faithfulness`, `answer_correctness`, `context_precision`, print a markdown summary table.
>   Output: file content only.

---

## 9) Tests & CI helpers

**Prompt 23 — `tests/test_ingest.py`**

> Unit tests for `ingest/ingest_json.py` to ensure:
>
> * JSON parsing → chunks produced;
> * Embedding function called with expected batch sizes;
> * Upsert called with correct payload keys (`paper_id, section, text, meta`).
>   Use monkeypatch to stub embeddings & vectorstore.
>   Output: file content only.

**Prompt 24 — `tests/test_retriever.py`**

> Tests for `retriever/retriever.py` on `build_subqueries`, `format_contexts`, `mmr` behavior (ensure diversity and reproducibility).
> Output: file content only.

**Prompt 25 — `tests/test_pipeline.py`**

> Tests for `predict_experiment` using a **fake LLM** (stub generate()) returning a minimal valid JSON. Assert schema validation and error handling on malformed JSON.
> Output: file content only.

**Prompt 26 — `Makefile`**

> Create a Makefile with targets: `install`, `qdrant`, `ingest`, `serve`, `test`, `format` (ruff), `typecheck` (mypy), `eval`.
> Output: file content only.

---

## 10) Sample data & scripts

**Prompt 27 — `data/samples/one_paper.json`**

> Create a **small sample** matching the 丹参通络解毒汤 schema（用简化数值即可）。
> Output: file content only.

**Prompt 28 — `data/samples/eval.jsonl`**

> Create 2–3 eval entries with simple `question`（如“对含丹参复方的急性毒性进行预测与实验流程建议”）、`answer_gt`（摘要式要点）、`contexts_gt`（包含应被检索到的字段关键字）。
> Output: file content only.

**Prompt 29 — `scripts/run_all.sh` (optional)**

> Bash: `make qdrant && make ingest && make serve` in sequence.
> Output: file content only.

---

## 11) Final polish

**Prompt 30 — `api` security & errors**

> Review `api/server.py`: add global exception handlers for `ValidationError`, `HTTPException`, unknown errors; ensure every response includes `request_id`. Add CORS (localhost).
> Output: updated file content only.

**Prompt 31 — Comments & docstrings pass**

> Walk through all Python files and add/normalize docstrings (Google style) and type hints. Ensure no `print` remains, only loguru.

---

### How to use this pack

1. Paste **Prompt 0** first to set conventions.
2. Then paste **Prompt 1 → 31** sequentially.
3. Save each returned file to the indicated path.
4. Run:

   ```
   cd .
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   docker compose -f docker/docker-compose.qdrant.yml up -d
   cp .env.example .env  # fill keys
   bash scripts/run_ingest.sh
   uvicorn api.server:app --reload
   bash scripts/demo_query.sh
   ```
5. Run tests & evaluation:

   ```
   pytest -q
   python -m eval.eval_ragas
   ```

If you’d like, I can also generate a **single “multi-file” prompt** that instructs the code LLM to output a tar/zip manifest with all files in one go.
