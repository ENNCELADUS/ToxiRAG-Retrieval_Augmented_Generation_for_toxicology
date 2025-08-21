# ToxiRAG — Implementation Plan

A concise, iterative plan to deliver a RAG-supported AI agent for liver cancer toxicity studies using paper-based Markdown knowledge.

## Goals
- Evidence-grounded answers using local Markdown KB (strict template)
- Hybrid retrieval (vector + keyword) over LanceDB
- Agentic reasoning via agno with OpenAI and Gemini
- Streamlit UI for ingestion, search, and analysis
- Unit tests for components + one end-to-end test

## Decisions (locked)
- Embedding model: `text-embedding-3-large`
- Default LLM: `gpt-5-nano` (Gemini `gemini-2.5-flash` optional)
- Primary vector store: LanceDB (stick to it; Qdrant optional later)
- Pilot scale: 500–1,000 Markdown docs
- Citation style: bracketed numerals with section tag, e.g., `[E2 · 实验分组与给药]`
- Unknown/omitted values: record as `未说明`
- Timepoints: accept `Day N` and `Day N+M` in raw; store canonical day `N+M`
- Units/normalization and metadata rules: see M1

## Non-goals (for now)
- Public API surface in `api/` (optional later)
- Docker + cloud deployment (later)

## Git Workflow
- **Beta branch**: All milestone work happens in `beta` branch
- **Commit after milestone**: Commit and push after each milestone passes tests
- **Commit format**: `feat: MX-MY complete - description` with detailed body
- **Test requirement**: All tests must pass before commit/push

## Milestones (with checklists)

### M0 — Environment, config, and scaffolding ✅ COMPLETED
- [x] Confirm conda env exists and activate: `conda activate toxirag`
- [x] Add `.env.example` with `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `OPENAI_EMBED_MODEL`, `LANCEDB_URI`, `COLLECTION_NAME`, `RETRIEVAL_TOP_K`
- [x] Add `scripts/` CLI entrypoints skeleton: `ingest_md.py`, `reindex.py`, `eval_run.py`
- [x] Wire up logging and settings (`pydantic-settings`)
- [x] **Git**: Create beta branch, commit M0 changes

Deliverables
- `.env.example` and `settings.py` (pydantic-settings)
- Script stubs under `scripts/`

Success criteria
- Streamlit boots and reads keys from env
- Settings validated; LanceDB URI defaults correctly

### M1 — Markdown parsing, normalization, and ingestion ✅ COMPLETED
- [x] Implement `ingest/markdown_schema.py` to parse the fixed template to structured dicts
- [x] Implement normalization utils (units, enums, table parsing) with rules:
  - Unknown/omitted → `未说明`
  - Timepoints → `Day N`; allow `Day N+M` in raw; store canonical day `N+M`
  - 肿瘤体积 (tumor volume): mm³; accept `mm^3`, `cm^3`, `mL`; convert `1 cm³ = 1000 mm³`, `1 mL = 1000 mm³`
  - If only diameters are given: record raw; do not compute unless paper defines formula; if defined, store `calc_method` (e.g., `V = (L×W²)/2`) and `value`
  - 体重 (body weight): g; if kg given → ×1000 to g
  - 器官质量 (organ mass): mg; if g given → ×1000 to mg
  - 剂量 (dose): mg/kg per administration; frequency normalized to `dose_frequency_norm` in {`qd`,`q2d`,`q3d`,`qod`,`qwk`,`bid`,`tid`,`未说明`}; store `daily_equiv_mg_per_kg` only if explicitly stated
  - 时间线: days since inoculation; convert week→days (×7) only if explicitly stated
  - 性别 (sex): {male, female, mixed, 未说明}
  - 品系 (strain): keep raw + `strain_norm` if exact match to {C57BL/6, BALB/c, KM, SD}
  - Persist metadata: `units_version = "v1.0"`, `calc_method`, `dose_frequency_norm`
- [x] Implement chunking strategy (by section + token-limit fallback)
- [x] Implement `ingest/ingest_local.py` to embed with OpenAI and upsert into LanceDB
- [x] Write unit tests for parsing, normalization, chunking, and upsert
- [x] **Git**: Commit M0-M1 to beta branch and push (39 tests passing)

Deliverables
- `ingest/markdown_schema.py`, `ingest/normalization.py`, `ingest/ingest_local.py`
- Tests: `tests/ingest/test_schema.py`, `test_normalization.py`, `test_ingest.py`

Success criteria
- Sample `.md` parses into structured dicts with normalized units and metadata
- Chunks upserted to LanceDB; embeddings created with `text-embedding-3-large`

### M2 — Vector store and hybrid retriever ✅ COMPLETED
- [x] Implement `retriever/retriever.py` with hybrid search (embedding + BM25/keywords)
- [x] Implement filters (by section/type), scoring, and deduplication
- [x] Implement evidence pack builder with bracketed numerals `[E1]`, section tags (e.g., `[E2 · 实验分组与给药]`), and metadata
- [x] Unit tests for retrieval ranking and evidence packing
- [x] **Git**: Commit M2 to beta branch and push (tests passing)

Deliverables
- `retriever/retriever.py` with hybrid scoring and evidence packing
- Tests: `tests/retriever/test_retrieve.py`, `test_evidence_pack.py`

Success criteria
- Top-K results relevant on sample queries; deduped and section-filterable
- Evidence packs render bracketed citations with section tags
- Handles pilot scale (500–1,000 docs) for ingest and retrieval within reasonable latency

### M3 — Agentic orchestration with agno ✅ COMPLETED
- [x] Implement `llm/agentic_pipeline.py` to: (a) decompose queries into sub-queries, (b) call retriever, (c) format evidence packs, (d) draft answers with reasoning tools
- [x] Add guardrails: answer only from evidence; cite sources; refuse if insufficient evidence
- [x] Unit tests with mocked LLM calls validating prompts and outputs
- [x] **Git**: Commit M3 to beta branch and push (tests passing)

Deliverables
- `llm/agentic_pipeline.py` with prompt templates and guardrails
- Tests: `tests/llm/test_agentic_pipeline.py`

Success criteria
- Responses include bracketed citations matching evidence packs
- Refusal behavior when evidence insufficient

### M4 — Streamlit app integration
- [ ] Connect ingestion button to `ingest_markdown_file`
- [ ] Connect Q&A to `retrieve_relevant_docs` and `create_agentic_response`
- [ ] Add configurable params (top_k, temperature, provider)
- [ ] Display evidence sources with stable IDs and sections
- [ ] Smoke test end-to-end locally
- [ ] **Git**: Commit M4 to beta branch and push (E2E test passing)

Deliverables
- Working UI flows for ingest and Q&A
- Evidence panel showing `[E# · Section]` and snippet previews

Success criteria
- End-to-end demo from upload → answer with citations
- UI reflects normalization (units, `未说明`) where displayed

### M5 — Evaluation and E2E test
- [ ] Add `eval/` golden questions + expected citations
- [ ] Implement RAGAS or custom checks for grounding and citation coverage
- [ ] Add `tests/e2e/test_rag_flow.py` covering ingest→retrieve→answer→citations
- [ ] Add coverage target (e.g., ≥70%) and CI-ready test command
- [ ] **Git**: Commit M5 to beta branch and push (all tests passing)

Deliverables
- `eval/` configs and gold answers
- `tests/e2e/test_rag_flow.py` and coverage setup

Success criteria
- E2E test green with correct citation formatting and grounding score ≥ target

### M6 — Quality, docs, and polish
- [ ] Add `readme` Quickstart, Troubleshooting, and FAQ for data format
- [ ] Provide `scripts` usage examples and sample `.md` in `data/samples/`
- [ ] Optional: implement `api/` FastAPI endpoints mirroring app flows
- [ ] **Git**: Final commit M6 to beta branch, merge to main

Deliverables
- Updated `README.md`, samples in `data/samples/`, script examples
- (Optional) `api/` endpoints documented

Success criteria
- New contributors can ingest and query within 10 minutes using README

## Scale target
- Support pilot corpus of 500–1,000 Markdown docs for ingest and retrieval

## Open items
- None at this time

## Success criteria (Definition of Done)
- All milestone checkboxes ticked
- Unit tests across `ingest/`, `retriever/`, `llm/` pass
- One `tests/e2e/test_rag_flow.py` passes: cited, evidence-grounded answer
- Streamlit demo works using a provided sample `.md`

## Quickstart (local)
```bash
conda activate toxirag
cp .env.example .env  # add keys
streamlit run app/main_app.py
```

## Test and eval
```bash
conda activate toxirag
pytest -q --maxfail=1 --disable-warnings
```

## Notes
- LanceDB path can default to `tmp/lancedb/agentic_rag_docs.lance` (already in repo structure) or `LANCEDB_URI` from `.env`
- Keep raw uploaded `.md` in `data/` for reproducibility


