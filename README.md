# ToxiRAG — Retrieval Augmented Generation for Toxicology

Evidence-grounded RAG system for toxicology research (Chinese/English) with strict normalization, hybrid retrieval over LanceDB, and agentic reasoning. Answers always include bracketed citations like `[E1 · 实验分组与给药]`.

## Getting Started (Quickstart)

### 1. Create and activate conda environment
```bash
# Create new conda environment with Python 3.11
conda create -n toxirag python=3.11 -y
conda activate toxirag
```

### 2. Install dependencies
```bash
# Install all required packages from requirements.txt
pip install -r requirements.txt
```

### 3. Configure API keys
```bash
# Copy environment template and add your API keys
cp .env.example .env
# Edit .env file and add your OPENAI_API_KEY and/or GOOGLE_API_KEY
```

### 4. Run the application
```bash
# Start the Streamlit UI
streamlit run app/main_app.py
```

### 5. Get started
- **Ingest sample docs**: Use the UI upload button or CLI commands below
- **Ask questions**: Type in Chinese or English; answers will include citations like `[E1 · 实验分组与给药]`
- **Try the sample**: Use `data/samples/mini_sample.md` for testing

## Project Structure

- `ingest/`: Markdown parsing, normalization, chunking
- `retriever/`: Hybrid search (vector + BM25), evidence packs
- `llm/`: Agentic orchestration with agno
- `config/`: Centralized settings via pydantic-settings
- `data/`: `summaries/` (real papers), `samples/` (toy example)
- `tests/`: Unit + e2e tests (bilingual, realistic data)

## Data Format (Strict Template)

Documents must follow the strict toxicology template used by the parser. Key rules:

- Unknown/missing values → `未说明`
- Units: tumor volume mm³; dose mg/kg; body weight g; organ mass mg
- Timeline uses Day N (+M) and is normalized to canonical days
- Citations in answers must be bracketed with section tag

See `data/samples/mini_sample.md` for a minimal, valid example.

## CLI Usage

### Ingest Markdown (.md) into LanceDB

```bash
conda activate toxirag
python scripts/ingest_md.py data/samples/mini_sample.md --verbose

# Or ingest a directory
python scripts/ingest_md.py data/summaries/ --verbose

# Dry run (parse + validate only, no DB writes)
python scripts/ingest_md.py data/samples/mini_sample.md --dry-run --verbose
```

### Evaluate against golden questions

```bash
conda activate toxirag
python scripts/eval_run.py --eval-config eval/config.yaml --verbose
```

### Reindex embeddings

```bash
conda activate toxirag
python scripts/reindex.py --verbose
```

## Troubleshooting

### Proxy Network Environments ✅ AUTO-FIXED
**ToxiRAG automatically handles proxy environments!** No manual setup needed.

The application includes built-in proxy bypass for OpenAI API access. When you start the app, it automatically:
- Clears proxy environment variables that interfere with OpenAI API  
- Sets `NO_PROXY` to include OpenAI domains (`api.openai.com`, `*.openai.com`)
- Shows "🔓 代理绕过已启用 (OpenAI API)" indicator in the sidebar

**Manual alternatives** (if needed):
```bash
# Option 1: Use the startup script
./run_toxirag.sh

# Option 2: Manual environment setup
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="api.openai.com,*.openai.com,openai.com,localhost,127.0.0.1"
streamlit run app/main_app.py
```

### Missing API keys
- Ensure `.env` has `OPENAI_API_KEY` and/or `GOOGLE_API_KEY`
- Streamlit/eval/ingest will fail if keys are missing (except `--dry-run`)

### LanceDB path issues
- Defaults via `LANCEDB_URI` in `.env` or project defaults in `config/settings.py`
- If you see "table not found", ingest first or point to the correct table

### Parsing errors
- Ensure headings match the template (e.g., `## 实验分组与给药`, `## 机制研究结果`)
- Use `未说明` for missing fields; keep bilingual text as-is

### Evidence/citation issues
- Answers are grounded only in retrieved chunks; ensure your docs were ingested
- Check that section tags exist so citations render like `[E1 · 机制研究结果]`

## FAQ

### Can I use English-only docs?
Yes, but keep the Chinese section headers so the parser recognizes sections.

### How do I add new papers?
Place normalized `.md` files under `data/summaries/` and run the ingest CLI.

### How are units normalized?
See `ingest/normalization.py`. Tumor volume → mm³, dose → mg/kg, etc.

### How are citations formatted?
The retriever builds evidence packs; the agent formats citations: `[E# · Section]`.

## Testing

```bash
conda activate toxirag
pytest --maxfail=1
```

## License

Apache-2.0 (or project default). See repository for details.


