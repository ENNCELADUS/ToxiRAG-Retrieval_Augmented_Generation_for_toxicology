## ToxiRAG — RAG for Liver-Cancer Toxicity Prediction and Experiment Planning

### Project goal
- **Purpose**: Assist researchers in planning and predicting animal experiments for liver cancer toxicity studies, with support for Traditional Chinese Medicine (TCM) compounds and general toxicology corpora.
- **Outcome**: Retrieve the most relevant evidence, reason with an agentic LLM, and produce actionable experiment suggestions and safety considerations.

### Architecture overview
- **Ingest**: Load structured/unstructured files (e.g., Markdown/JSON) from `data/` and normalize into a unified schema.
- **Index**: Create embeddings and upsert into Qdrant collections.
- **Retrieval**: Hybrid or dense vector search to fetch top-k passages.
- **Agentic LLM**: Tool-using reasoning to critique, cross-check, and synthesize experimental guidance.
- **API (optional)**: FastAPI endpoints to expose retrieval + reasoning as a service.

Flow: `ingest → index → retrieval → agentic LLM` (API optional)

### What does `app/` do? Is it needed without a GUI?
- `app/` hosts core, shared application components that do not imply a UI:
  - `config.py`: pydantic-settings based configuration loaded from `.env`.
  - `logging.py`: loguru setup for structured logging.
  - `schemas.py`: Pydantic models for I/O and domain entities.
  - `constants.py`: project-wide constants and enums.
  - `utils_text.py`: pure text utility helpers reused across modules.
- Keep `app/` even for a CLI-only prototype. It centralizes config, typing, and utilities used by `ingest/`, `retriever/`, and `llm/`.
- The `api/` folder is OPTIONAL. Skip it if you do not need a server yet.

### Quickstart
- **Prerequisites**: Python 3.11+, Docker, Docker Compose

1) Create and activate a virtual environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) Start Qdrant (vector DB)
```bash
docker compose -f docker/docker-compose.qdrant.yml up -d
```

4) Configure environment
```bash
cp .env.example .env
# Fill in API keys and provider/model settings
```

5) Ingest data into Qdrant
```bash
bash scripts/run_ingest.sh
```

6) Start API server (optional)
```bash
uvicorn api.server:app --reload
```

CLI-only prototype: you can skip the API and invoke modules directly from your own scripts under `scripts/` or from an interactive session (e.g., use functions in `ingest/` to load/index and `retriever/` to query).

### Switch providers (OpenAI / Gemini)
Set in `.env`:
```env
# Provider (LLM + embeddings)
LLM_PROVIDER=openai      # openai | gemini
EMBEDDING_PROVIDER=openai # openai | gemini

# Models
OPENAI_MODEL=gpt-5
GEMINI_MODEL=gemini-2.5-pro
OPENAI_EMBED_MODEL=text-embedding-3-large
GEMINI_EMBED_MODEL=text-embedding-004

# API keys
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
```
- To use Gemini, set `LLM_PROVIDER=gemini` and `EMBEDDING_PROVIDER=gemini` and ensure `GOOGLE_API_KEY` is set.
- To use OpenAI, set `LLM_PROVIDER=openai` and `EMBEDDING_PROVIDER=openai` and ensure `OPENAI_API_KEY` is set.

### Data schema and samples
- **Location**: `data/samples/`
- **Document schema (suggested)**:
```json
{
  "id": "string",
  "title": "string",
  "abstract": "string",
  "content": "string",
  "compounds": ["string"],
  "indications": ["liver cancer", "..."],
  "tox_findings": ["hepatotoxicity", "..."],
  "species": ["mouse", "rat"],
  "source": "journal | database | report",
  "year": 2024,
  "url": "https://..."
}
```
- You can also store Markdown articles; the ingester extracts text and metadata for indexing.

### Notes
- Keep all credentials and model selections in `.env`.
- Tune `COLLECTION_NAME` and Qdrant params (`QDRANT_URL`, `QDRANT_API_KEY`) as needed.

### Reference
- High-level agentic RAG workflow adapted from patterns similar to the shared conversation: [Agentic RAG overview](https://chatgpt.com/share/68a59f5d-401c-800b-b8e3-232c5d837268). 