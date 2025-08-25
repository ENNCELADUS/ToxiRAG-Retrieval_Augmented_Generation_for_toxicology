# ToxiRAG - Technical Architecture & User Manual

**A Specialized Retrieval-Augmented Generation System for Toxicology Research**

This document provides a comprehensive technical overview of ToxiRAG's architecture and complete usage instructions for both CLI and GUI interfaces.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Technical Architecture](#technical-architecture)
3. [Data Flow & Processing Pipeline](#data-flow--processing-pipeline)
4. [Module-by-Module Technical Details](#module-by-module-technical-details)
5. [CLI Usage Manual](#cli-usage-manual)
6. [GUI Usage Manual](#gui-usage-manual)
7. [Configuration & Environment](#configuration--environment)
8. [Performance & Scaling](#performance--scaling)
9. [Troubleshooting](#troubleshooting)
10. [Developer Reference](#developer-reference)

---

## System Overview

ToxiRAG is a domain-specific RAG system designed for toxicology researchers who need to query scientific literature with evidence-based answers. The system processes structured markdown documents following a strict toxicology template and provides hybrid search capabilities with proper citation tracking.

### Key Features

- **Domain-Specific Processing**: Specialized parsing and normalization for toxicology study data
- **Hybrid Search**: Combines vector similarity (OpenAI embeddings) with BM25 keyword matching  
- **Evidence-Based Responses**: All answers include bracketed citations linking to source sections
- **Bilingual Support**: Handles mixed Chinese/English toxicology content
- **Strict Data Normalization**: Standardizes units, measurements, and study parameters
- **Agentic Reasoning**: Uses LLM orchestration for query decomposition and response synthesis

### Technology Stack

- **Vector Database**: LanceDB for document embeddings and metadata storage
- **Embeddings**: OpenAI text-embedding-3-large (3072 dimensions)
- **LLMs**: GPT-5-nano (primary), Gemini 2.5-flash (secondary)
- **Framework**: agno for agentic orchestration
- **Backend**: Python with asyncio, pandas, sklearn
- **UI**: Streamlit for web interface
- **Testing**: pytest with comprehensive test coverage (61+ tests)

---

## Technical Architecture

### System Components Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ToxiRAG System Architecture                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   Input     │    │   Processing │    │     Storage      │   │
│  │   Layer     │───▶│    Layer     │───▶│     Layer        │   │
│  │             │    │              │    │                  │   │
│  │ • Markdown  │    │ • Parse      │    │ • LanceDB       │   │
│  │ • Upload UI │    │ • Normalize  │    │ • Embeddings    │   │
│  │ • CLI       │    │ • Chunk      │    │ • Metadata      │   │
│  └─────────────┘    │ • Embed      │    └──────────────────┘   │
│                     └──────────────┘                           │
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   Query     │    │   Retrieval  │    │     Response     │   │
│  │   Layer     │───▶│    Layer     │───▶│     Layer        │   │
│  │             │    │              │    │                  │   │
│  │ • User Q    │    │ • Vector     │    │ • LLM Reasoning │   │
│  │ • GUI/CLI   │    │ • BM25       │    │ • Citations     │   │
│  │ • Filters   │    │ • Hybrid     │    │ • Evidence Pack │   │
│  └─────────────┘    │ • Evidence   │    └──────────────────┘   │
│                     └──────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

1. **`ingest/`** - Data parsing, normalization, and chunking
2. **`retriever/`** - Hybrid search and evidence pack building
3. **`llm/`** - Agentic reasoning and LLM orchestration  
4. **`config/`** - Centralized settings and validation
5. **`utils/`** - Logging and shared utilities
6. **`app/`** - Streamlit web interface
7. **`scripts/`** - CLI tools for ingestion, reindexing, and evaluation

---

## Data Flow & Processing Pipeline

### 1. Document Ingestion Pipeline

```
Raw Markdown → Parse Structure → Normalize Data → Chunk Content → Generate Embeddings → Store in LanceDB
```

**Step-by-Step Process:**

1. **Markdown Parsing** (`ingest/markdown_schema.py`)
   - Parses strict toxicology template format
   - Extracts structured data: animal info, experimental groups, data tables, mechanisms
   - Handles bilingual content (Chinese/English)

2. **Data Normalization** (`ingest/normalization.py`)
   - **Units**: Tumor volume → mm³, Body weight → g, Organ mass → mg, Dose → mg/kg
   - **Missing Values**: Use `未说明` instead of null/empty
   - **Timeline**: Convert to canonical days since inoculation
   - **Frequencies**: Normalize to {qd, q2d, q3d, qod, qwk, bid, tid}
   - **Strains**: Map to standard codes (C57BL/6, BALB/c, KM, SD)

3. **Content Chunking** (`ingest/chunking.py`)
   - Section-based chunking with token-limit fallback
   - Default: 1000 tokens per chunk, 200 token overlap
   - Preserves section context and metadata
   - Generates citation IDs and section tags

4. **Embedding Generation**
   - Uses OpenAI text-embedding-3-large
   - Async processing for performance
   - 3072-dimensional vectors

5. **Vector Storage**
   - LanceDB with structured schema
   - Includes content, metadata, source pages, and embeddings
   - Deduplication based on content hashing

### 2. Query Processing Pipeline

```
User Query → Query Embedding → Vector Search + BM25 → Score Fusion → Evidence Pack → LLM Reasoning → Cited Response
```

**Step-by-Step Process:**

1. **Query Processing**
   - Generate embedding for user query
   - Prepare query for both vector and keyword search

2. **Hybrid Retrieval** (`retriever/retriever.py`)
   - **Vector Search**: Cosine similarity against document embeddings
   - **BM25 Scoring**: Keyword-based relevance using TF-IDF
   - **Score Fusion**: Weighted combination (default: 70% vector, 30% BM25)
   - **Filtering**: By section type, document title, minimum score threshold
   - **Deduplication**: Remove near-duplicate results using content similarity

3. **Evidence Pack Building**
   - Format citations: `[E1 · 实验分组与给药]`
   - Combine content with source page references
   - Maintain traceability to original documents

4. **Agentic Reasoning** (`llm/agentic_pipeline.py`)
   - Query decomposition for complex questions
   - Evidence evaluation and sufficiency checking
   - Response synthesis with proper citations
   - Refusal when evidence is insufficient

---

## Module-by-Module Technical Details

### `ingest/` - Data Ingestion Module

#### `markdown_schema.py` - Document Structure Parser

**Purpose**: Parse structured toxicology markdown into Python dataclasses

**Key Classes**:
- `ToxicologyDocument`: Complete document structure
- `AnimalInfo`: Experimental animal data with normalization
- `DataTable`: Generic table structure with metadata
- `ExperimentGroup`: Dosing and treatment information

**Key Features**:
- Handles multiple animal groups (mice_info_1-4, rat_info)
- Parses complex tables with headers and data
- Extracts source page references
- Normalizes dose frequencies using mapping dictionaries

**Example Usage**:
```python
from ingest.markdown_schema import MarkdownParser

parser = MarkdownParser()
doc = parser.parse_file(Path("data/summaries/肝癌.md"))
# Accesses structured data: doc.mice_info_1.strain, doc.experiment_groups.dose_mg_per_kg
```

#### `normalization.py` - Data Standardization

**Purpose**: Normalize toxicology measurements to standard units and formats

**Key Normalizers**:
- `TumorVolumeNormalizer`: Converts cm³, mL → mm³ with formula support
- `WeightNormalizer`: Standardizes to grams (kg×1000, mg÷1000)
- `DoseNormalizer`: Handles mg/kg dosing with frequency mapping
- `TimelineNormalizer`: Canonical day numbers (Day N+M → N+M)
- `StrainNormalizer`: Maps animal strains to standard codes
- `SexNormalizer`: Standardizes to {male, female, mixed, 未说明}

**Key Features**:
- Preserves original values alongside normalized ones
- Includes calculation methods for derived values
- Handles measurement uncertainties with `未说明`
- Version tracking for normalization rules (v1.0)

#### `chunking.py` - Content Segmentation

**Purpose**: Split documents into retrievable chunks while preserving context

**Chunking Strategy**:
1. **Section-Based**: Each major section becomes a chunk
2. **Token-Limit Fallback**: Large sections split at token boundaries
3. **Overlap Preservation**: 200-token overlap between adjacent chunks
4. **Metadata Enrichment**: Each chunk includes section type, citation info

**Section Types**:
- `title`: Paper title and source information
- `animal_info`: Experimental animal details
- `experiment_groups`: Treatment protocols
- `data_table`: Experimental results tables
- `mechanism`: Mechanism research findings
- `conclusion`: Study conclusions

### `retriever/` - Hybrid Search Module

#### `retriever.py` - Multi-Modal Search Engine

**Purpose**: Combine vector similarity and keyword matching for optimal relevance

**Core Components**:

1. **BM25Scorer**: Traditional information retrieval scoring
   - Parameters: k1=1.2, b=0.75 (tuned for scientific text)
   - Features: Bigram support, toxicology term preservation
   - Training: Fits on entire corpus for consistent IDF values

2. **ToxiRAGRetriever**: Main retrieval orchestrator
   - **Lazy Loading**: Database connections only when needed
   - **Async Operations**: Non-blocking embedding generation
   - **Score Normalization**: Converts distances to similarities
   - **Deduplication**: TF-IDF based content similarity detection

3. **EvidencePack**: Structured result formatting
   - Citation format: `[E1 · section_name]`
   - Source tracking: Page numbers and file paths
   - Metadata preservation: Scores, ranks, section types

### `llm/` - Agentic Orchestration Module

#### `agentic_pipeline.py` - LLM Reasoning Engine

**Purpose**: Orchestrate LLM interactions for query understanding and response generation

**Key Components**:

1. **QueryDecomposer**: Breaks complex questions into sub-queries
2. **EvidenceEvaluator**: Assesses sufficiency of retrieved evidence
3. **ResponseSynthesizer**: Generates answers with proper citations
4. **Guardrails**: Ensures evidence-based responses only

### `config/` - Configuration Management

#### `settings.py` - Centralized Configuration

**Purpose**: Type-safe configuration management using pydantic-settings

**Configuration Categories**:

1. **API Keys**: OpenAI, Google API credentials
2. **Model Settings**: Embedding models, LLM providers
3. **Database**: LanceDB URI, collection names
4. **Processing**: Chunk sizes, retrieval parameters
5. **Development**: Debug flags, logging levels

**Key Features**:
- **Environment Variable Integration**: Automatic `.env` file loading
- **Validation**: Pydantic validators for paths and settings
- **Type Safety**: Full type hints with IDE support
- **Default Values**: Sensible defaults for all parameters

---

## CLI Usage Manual

### Prerequisites

```bash
# Activate environment
conda activate toxirag

# Verify installation
python -c "import lancedb, agno; print('Dependencies OK')"

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
# OPENAI_API_KEY=your_key_here
# GOOGLE_API_KEY=your_key_here
```

### Document Ingestion - `scripts/ingest_md.py`

**Purpose**: Parse and ingest toxicology markdown documents into the vector database

#### Basic Usage

```bash
# Ingest single file
python scripts/ingest_md.py data/summaries/肝癌.md

# Ingest entire directory
python scripts/ingest_md.py data/summaries/

# Ingest with custom collection name
python scripts/ingest_md.py data/summaries/ --collection my_toxicology_docs
```

#### Advanced Options

```bash
# Full option example
python scripts/ingest_md.py data/summaries/ \
    --collection toxicology_docs \
    --chunk-size 1200 \
    --chunk-overlap 300 \
    --batch-size 10 \
    --embedding-model text-embedding-3-large \
    --force-reprocess \
    --verbose

# Dry run (validate without ingesting)
python scripts/ingest_md.py data/summaries/ --dry-run --verbose

# Process specific file with debugging
python scripts/ingest_md.py data/summaries/肝癌.md \
    --verbose \
    --log-level DEBUG
```

#### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `input_path` | Path to .md file or directory | **Required** |
| `--collection` | LanceDB collection name | `toxicology_docs` |
| `--chunk-size` | Maximum tokens per chunk | `1000` |
| `--chunk-overlap` | Overlap between chunks | `200` |
| `--batch-size` | Documents per batch | `5` |
| `--embedding-model` | OpenAI embedding model | `text-embedding-3-large` |
| `--force-reprocess` | Reprocess existing documents | `False` |
| `--dry-run` | Validate without ingesting | `False` |
| `--verbose` | Detailed logging output | `False` |
| `--log-level` | Logging level | `INFO` |

#### Processing Pipeline Details

1. **Document Validation**:
   - Checks markdown follows toxicology template
   - Validates required sections presence
   - Reports parsing errors with line numbers

2. **Data Processing**:
   - Parses structured content into dataclasses
   - Normalizes units and measurements
   - Generates chunks with metadata

3. **Embedding Generation**:
   - Async batch processing for performance
   - Retry logic for API failures
   - Progress tracking with ETA

4. **Database Storage**:
   - Deduplication based on content hash
   - Atomic transactions for consistency
   - Index optimization after ingestion

#### Troubleshooting Ingestion

**Common Issues**:

1. **API Key Missing**:
   ```bash
   Error: OpenAI API key not configured
   Solution: Set OPENAI_API_KEY in .env file
   ```

2. **Invalid Markdown Format**:
   ```bash
   Error: Failed to parse section '## 实验小鼠1信息'
   Solution: Check markdown follows exact template format
   ```

3. **Database Connection**:
   ```bash
   Error: Cannot connect to LanceDB at data/knowledge_base/lancedb
   Solution: Check directory permissions and disk space
   ```

### Database Reindexing - `scripts/reindex.py`

**Purpose**: Rebuild vector embeddings and search indices

#### When to Use

- **Embedding Model Changes**: Switching from text-embedding-3-large to newer models
- **Database Corruption**: Recovering from corrupted LanceDB files
- **Performance Issues**: Optimizing search after large ingestions
- **Schema Updates**: Migrating to new metadata structures

#### Usage Examples

```bash
# Basic reindexing
python scripts/reindex.py

# Reindex specific collection with new embedding model
python scripts/reindex.py \
    --collection toxicology_docs \
    --embedding-model text-embedding-3-small \
    --force

# Backup and reindex with verification
python scripts/reindex.py \
    --backup-dir data/backups/$(date +%Y%m%d_%H%M%S) \
    --verify-results \
    --verbose
```

#### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--collection` | Collection to reindex | `toxicology_docs` |
| `--embedding-model` | New embedding model | `text-embedding-3-large` |
| `--backup-dir` | Backup location before reindex | None |
| `--batch-size` | Documents per embedding batch | `10` |
| `--force` | Skip confirmation prompts | `False` |
| `--verify-results` | Validate after reindexing | `False` |
| `--verbose` | Detailed progress logging | `False` |

### Evaluation - `scripts/eval_run.py`

**Purpose**: Run evaluation tests against golden questions for quality assurance

#### Usage Examples

```bash
# Basic evaluation
python scripts/eval_run.py

# Custom evaluation config
python scripts/eval_run.py \
    --eval-config eval/custom_config.yaml \
    --output-dir eval/results/$(date +%Y%m%d_%H%M%S)

# Evaluation with specific LLM provider
python scripts/eval_run.py \
    --llm-provider gemini \
    --temperature 0.2 \
    --top-k 10 \
    --verbose
```

#### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--eval-config` | Evaluation configuration file | `eval/config.yaml` |
| `--collection` | LanceDB collection name | `toxicology_docs` |
| `--output-dir` | Results output directory | `eval/results` |
| `--llm-provider` | LLM provider (openai/gemini) | `openai` |
| `--temperature` | LLM temperature | `0.1` |
| `--top-k` | Number of results to retrieve | `5` |
| `--verbose` | Detailed evaluation logs | `False` |

#### Evaluation Metrics

1. **Citation Accuracy**: Percentage of responses with correct citations
2. **Evidence Grounding**: How well answers align with retrieved evidence
3. **Query Coverage**: Percentage of questions answered (vs. refused)
4. **Response Quality**: Semantic similarity to expected answers

---

## GUI Usage Manual

### Getting Started

#### 1. Launch the Application

```bash
# Start Streamlit app
conda activate toxirag
streamlit run app/main_app.py

# Access at: http://localhost:8501
```

#### 2. Initial Setup

The application will guide you through initial configuration:

1. **API Key Validation**: Check if OpenAI/Google keys are configured
2. **Database Status**: Verify LanceDB connection and existing documents
3. **Model Selection**: Choose embedding and LLM models

### Main Interface Overview

The Streamlit interface is organized into three main sections:

#### **📄 Document Ingestion Tab**

**Purpose**: Upload and process toxicology documents

**Features**:
- **File Upload**: Drag-and-drop or browse for .md files
- **Batch Processing**: Upload multiple files simultaneously
- **Progress Tracking**: Real-time processing status with progress bars
- **Validation**: Immediate feedback on document format compliance
- **Duplicate Detection**: Warns if content already exists in database

**Step-by-Step Usage**:

1. **Select Files**:
   ```
   Click "Browse files" or drag .md files into the upload area
   Supported: Single files or multiple file selection
   Format: Must follow strict toxicology template
   ```

2. **Configure Processing**:
   ```
   Collection Name: [toxicology_docs] (or custom)
   Chunk Size: [1000] tokens
   Chunk Overlap: [200] tokens
   Force Reprocess: □ (check to overwrite existing)
   ```

3. **Start Ingestion**:
   ```
   Click "🚀 Ingest Documents"
   Monitor progress bars for parsing, normalization, embedding
   View processing logs in expandable sections
   ```

4. **Review Results**:
   ```
   ✅ Documents processed: 3/3
   📊 Total chunks created: 47
   ⚠️ Warnings: 1 (duplicate content detected)
   💾 Database updated: toxicology_docs
   ```

#### **❓ Q&A Interface Tab**

**Purpose**: Query the knowledge base with evidence-based responses

**Features**:
- **Smart Query Input**: Auto-suggestions based on indexed content
- **Configurable Search**: Adjust retrieval parameters in real-time
- **Evidence Display**: Expandable sections showing source citations
- **Response Streaming**: Real-time answer generation with progress
- **Export Options**: Save conversations and citations

**Step-by-Step Usage**:

1. **Enter Query**:
   ```
   Query Input: "What is the mechanism of action for turmeric extract in liver cancer?"
   
   Suggested Queries:
   - "What are the dose-response relationships for..."
   - "Which animal models were used for..."
   - "What statistical significance was found for..."
   ```

2. **Configure Search Parameters**:
   ```
   Retrieval Settings:
   ├── Top K Results: [5] ────────── Number of evidence pieces
   ├── Vector Weight: [0.7] ──────── Semantic similarity weight
   ├── BM25 Weight: [0.3] ────────── Keyword matching weight
   ├── Temperature: [0.1] ────────── LLM randomness
   └── Max Tokens: [2000] ────────── Response length limit
   
   Filters:
   ├── Section Types: [mechanism, data_table, conclusion]
   ├── Document Titles: [All] or specific papers
   └── Min Score: [0.0] ───────────── Quality threshold
   ```

3. **Submit and Review**:
   ```
   Click "🔍 Search" or press Enter
   
   Processing Steps:
   ├── 🎯 Query Analysis ──────────── Decomposing complex questions
   ├── 🔍 Evidence Retrieval ──────── Hybrid search execution  
   ├── 📋 Evidence Evaluation ────── Sufficiency checking
   └── ✍️ Response Generation ────── LLM synthesis with citations
   ```

4. **Explore Evidence**:
   ```
   📋 Evidence Panel (5 results found):
   
   [E1 · 机制研究结果] Score: 0.89
   ╭─ Source: 姜黄素对肝癌细胞的作用机制研究
   ├─ Section: 机制研究结果  
   ├─ Page: p.15-18, Figure 3
   └─ Content: 姜黄素通过调节Wnt/β-catenin信号通路...
       [Click to expand full content]
   
   [E2 · 数据记录表格 - 肿瘤体积] Score: 0.82
   ╭─ Source: 姜黄素对肝癌细胞的作用机制研究
   ├─ Section: 数据记录表格
   ├─ Page: p.12, Table 2
   └─ Content: |组别|剂量|肿瘤体积变化率|统计学意义|...
       [Click to expand table data]
   ```

5. **Review Generated Response**:
   ```
   🤖 Response:
   
   Based on the available evidence, turmeric extract (curcumin) exhibits 
   anti-liver cancer effects through multiple mechanisms:
   
   **Primary Mechanism**: Wnt/β-catenin Pathway Regulation [E1 · 机制研究结果]
   Curcumin significantly downregulates β-catenin expression and inhibits 
   Wnt signaling pathway activation, leading to reduced tumor cell proliferation.
   
   **Supporting Evidence**: Dose-Response Data [E2 · 数据记录表格 - 肿瘤体积]
   Treatment with 100 mg/kg curcumin daily showed 67.3% tumor volume 
   reduction compared to control group (p<0.001).
   
   **Clinical Relevance**: [E3 · 研究结论]
   The mechanism suggests potential for clinical translation with 
   optimized dosing protocols.
   
   ---
   Evidence Confidence: High (5/5 sources)
   Citation Coverage: 100% (all claims cited)
   Response Quality: ⭐⭐⭐⭐⭐
   ```

#### **⚙️ Configuration Tab**

**Purpose**: System settings and advanced configuration

**Sections**:

1. **API Configuration**:
   ```
   OpenAI Settings:
   ├── API Key: [●●●●●●●●●●●●●●●●] ✅ Valid
   ├── Embedding Model: [text-embedding-3-large]
   └── Default Model: [gpt-5-nano]
   
   Google Settings:
   ├── API Key: [●●●●●●●●●●●●●●●●] ✅ Valid  
   └── Model: [gemini-2.5-flash]
   
   Active Provider: ○ OpenAI ● Gemini
   ```

2. **Database Configuration**:
   ```
   LanceDB Settings:
   ├── URI: [data/knowledge_base/lancedb]
   ├── Collection: [toxicology_docs]
   ├── Status: ✅ Connected (1,247 documents)
   └── Last Updated: 2024-01-15 14:30:22
   
   Actions:
   ├── [Backup Database] ──── Create timestamped backup
   ├── [Reindex Collection] ── Rebuild embeddings  
   └── [Clear Collection] ──── Remove all documents
   ```

3. **Processing Settings**:
   ```
   Document Processing:
   ├── Chunk Size: [1000] tokens
   ├── Chunk Overlap: [200] tokens
   ├── Units Version: [v1.0]
   └── Normalization: ✅ Enabled
   
   Retrieval Defaults:
   ├── Top K: [5] results
   ├── Vector Weight: [0.7]
   ├── BM25 Weight: [0.3]
   └── Min Score: [0.0]
   ```

### Advanced Features

#### **Query Suggestions & Auto-Complete**

The interface provides intelligent query suggestions based on:
- **Indexed Content**: Common terms from your document corpus
- **Section Types**: Queries targeting specific data types
- **Historical Queries**: Previously successful searches
- **Domain Templates**: Pre-built toxicology question patterns

Examples:
```
Dose-Response: "What is the dose-response relationship for [compound] in [model]?"
Mechanism: "What is the mechanism of action of [compound] against [target]?"
Comparison: "Compare the efficacy of [treatment A] vs [treatment B] in [context]"
Safety: "What are the toxicity profiles and safety margins for [compound]?"
```

#### **Filter Management**

**Section Type Filters**:
- `animal_info`: Experimental setup and animal details
- `experiment_groups`: Treatment protocols and dosing
- `data_table`: Quantitative results and measurements
- `mechanism`: Biological pathways and mode of action
- `pathology`: Histological and morphological findings
- `conclusion`: Summary findings and clinical implications

**Document Title Filters**:
- Search within specific papers or studies
- Useful for focused analysis or verification
- Supports partial matching and wildcards

#### **Export & Sharing**

**Conversation Export**:
```
Format Options:
├── 📄 PDF Report ───── Formatted document with citations
├── 📊 Excel Workbook ── Evidence tables with metadata
├── 📝 Markdown ─────── Plain text with citation links
└── 🔗 Shareable Link ── Temporary URL for collaboration
```

**Citation Export**:
- Standard academic formats (APA, MLA, Vancouver)
- BibTeX entries for LaTeX documents
- Mendeley/Zotero compatible formats
- Custom toxicology journal styles

---

## Configuration & Environment

### Environment Setup

#### Required Environment Variables

Create `.env` file in project root:

```bash
# API Keys (at least one required)
OPENAI_API_KEY=sk-your-openai-key-here
GOOGLE_API_KEY=your-google-api-key-here

# Model Configuration
OPENAI_EMBED_MODEL=text-embedding-3-large
DEFAULT_LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=gpt-5-nano
GEMINI_MODEL=gemini-2.5-flash

# Database Configuration  
LANCEDB_URI=data/knowledge_base/lancedb
COLLECTION_NAME=toxicology_docs

# Processing Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_TOP_K=5
RETRIEVAL_TEMPERATURE=0.1
MAX_TOKENS=2000

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/toxirag.log
DEBUG=false
ENVIRONMENT=development
```

#### Directory Structure

Ensure the following directories exist:

```
toxirag/
├── data/
│   ├── knowledge_base/
│   │   └── lancedb/           # Vector database files
│   ├── summaries/             # Input markdown documents
│   ├── samples/               # Sample/test documents
│   └── backups/              # Database backups
├── logs/                     # Application logs
├── eval/
│   ├── results/              # Evaluation outputs
│   └── config.yaml           # Evaluation configuration
└── .env                      # Environment variables
```

### Database Configuration

**LanceDB Setup**:

The system uses LanceDB with the following schema:

```python
Schema:
├── id: str                   # Unique chunk identifier
├── content: str              # Chunk text content  
├── embedding: vector(3072)   # OpenAI embedding
├── document_title: str       # Source document name
├── section_name: str         # Section header
├── section_type: str         # Section category
├── citation_id: str          # Citation reference (E1, E2, ...)
├── section_tag: str          # Display tag for citations
├── source_page: str          # Original page reference
├── file_path: str            # Source file location
├── metadata: str             # JSON-encoded additional data
└── created_at: timestamp     # Ingestion timestamp
```

---

## Performance & Scaling

### Performance Benchmarks

#### Ingestion Performance

**Single Document Processing**:
- Small document (5-10 sections): ~30-60 seconds
- Medium document (15-25 sections): ~2-4 minutes  
- Large document (30+ sections): ~5-8 minutes

**Batch Processing**:
- 10 documents: ~15-30 minutes
- 50 documents: ~1-2 hours
- 100+ documents: ~3-5 hours

**Bottlenecks**:
1. **OpenAI API Rate Limits**: 3,000 RPM for text-embedding-3-large
2. **Network Latency**: API round-trip times (50-200ms per request)
3. **Memory Usage**: Large documents require more RAM for processing

#### Query Performance

**Search Latency** (95th percentile):
- Vector search only: ~200-500ms
- Hybrid search: ~400-800ms  
- With LLM response: ~3-8 seconds

**Throughput**:
- Concurrent queries: 5-10 queries/second
- Database size impact: Linear scaling up to 100K documents
- Memory requirements: ~2-4GB for 10K documents

### Scaling Strategies

#### Performance Optimization

**Query Optimization**:
1. **Filter Early**: Apply section type and document filters before scoring
2. **Adjust Weights**: Tune vector/BM25 weights based on query type
3. **Limit Results**: Use appropriate top_k values (5-10 typically optimal)
4. **Cache Results**: Store frequent query results for faster response

---

## Troubleshooting

### Common Issues & Solutions

#### 1. Installation & Environment

**Issue**: `ModuleNotFoundError` when importing ToxiRAG modules
```bash
Error: ModuleNotFoundError: No module named 'ingest'
```

**Solutions**:
```bash
# Ensure conda environment is activated
conda activate toxirag

# Verify environment setup
conda list | grep -E "(lancedb|agno|streamlit)"

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### 2. API Key & Authentication

**Issue**: OpenAI API key validation failure
```bash
Error: Invalid OpenAI API key or insufficient permissions
```

**Solutions**:
```bash
# Check API key format (should start with sk-)
echo $OPENAI_API_KEY | head -c 10

# Test API key manually
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models

# Verify .env file loading
python -c "
from config.settings import settings
print(f'OpenAI key configured: {settings.has_openai_key()}')
"
```

#### 3. Database Connection Issues

**Issue**: LanceDB connection failures
```bash
Error: Cannot connect to LanceDB at data/knowledge_base/lancedb
```

**Solutions**:
```bash
# Check directory permissions
ls -la data/knowledge_base/
chmod 755 data/knowledge_base/lancedb/

# Verify disk space
df -h data/

# Test LanceDB directly
python -c "
import lancedb
db = lancedb.connect('data/knowledge_base/lancedb')
print(f'Tables: {db.table_names()}')
"
```

#### 4. Document Processing Errors

**Issue**: Markdown parsing failures
```bash
Error: Failed to parse section '## 实验小鼠1信息' at line 45
```

**Solutions**:
```bash
# Validate markdown format
python -c "
from ingest.markdown_schema import MarkdownParser
parser = MarkdownParser()
try:
    doc = parser.parse_file('problematic_file.md')
    print('Parsing successful')
except Exception as e:
    print(f'Parse error: {e}')
"

# Check required sections presence
grep -n "^## " your_file.md

# Compare with working sample
diff your_file.md data/samples/mini_sample.md
```

#### 5. Search & Retrieval Issues

**Issue**: No search results returned
```bash
Warning: No vector search results found for query
```

**Solutions**:
```bash
# Check database contents
python -c "
from retriever.retriever import ToxiRAGRetriever
retriever = ToxiRAGRetriever()
stats = retriever.get_corpus_stats()
print(f'Total chunks: {stats[\"total_chunks\"]}')
"

# Test with simpler query
python -c "
import asyncio
from retriever.retriever import search_documents

async def test():
    results = await search_documents('tumor', top_k=1)
    print(f'Results: {len(results.results)}')
    
asyncio.run(test())
"
```

#### 6. Performance Issues

**Issue**: Slow query responses
```bash
Issue: Queries taking >10 seconds to complete
```

**Solutions**:
```bash
# Reduce top_k for faster results
top_k = 3  # instead of 10

# Optimize database indices
python scripts/reindex.py --optimize-indices

# Monitor API latency
curl -w "@curl-format.txt" -s -o /dev/null \
     https://api.openai.com/v1/embeddings
```

---

## Developer Reference

### Key Classes

**`ToxicologyDocument`**:
```python
@dataclass
class ToxicologyDocument:
    """Complete parsed toxicology document."""
    title: str
    source_info: Optional[str]
    mice_info_1: Optional[AnimalInfo]
    experiment_groups: Optional[ExperimentGroup]
    data_tables: List[DataTable]
    mechanism: Optional[MechanismInfo]
    conclusion: Optional[StudyConclusion]
    keywords: List[str]
    units_version: str = "v1.0"
    file_path: Optional[str] = None
```

**`RetrievalResult`**:
```python
@dataclass  
class RetrievalResult:
    """Single retrieval result with scoring."""
    id: str
    content: str
    document_title: str
    section_type: str
    citation_id: str
    vector_score: float
    bm25_score: float
    combined_score: float
    rank: int
    metadata: Dict[str, Any]
```

**`EvidencePack`**:
```python
@dataclass
class EvidencePack:
    """Formatted evidence with citations."""
    query: str
    results: List[RetrievalResult]  
    evidence_text: str  # With [E1 · section] citations
    citations: List[Dict[str, Any]]
    total_results: int
    filters_applied: Dict[str, Any]
```

### Key Methods

**Document Processing**:
```python
# Parse markdown document
parser = MarkdownParser()
doc: ToxicologyDocument = parser.parse_file(file_path)

# Normalize data
normalizer = DataNormalizer()
normalized_data = normalizer.normalize_all_fields(raw_data)

# Chunk for retrieval
chunker = DocumentChunker()
chunks: List[DocumentChunk] = chunker.chunk_document(doc)
```

**Search Operations**:
```python
# Basic search
retriever = ToxiRAGRetriever()
results: List[RetrievalResult] = await retriever.search(
    query="tumor volume reduction",
    top_k=5,
    vector_weight=0.7,
    bm25_weight=0.3
)

# Build evidence pack
evidence_pack: EvidencePack = retriever.build_evidence_pack(
    query, results, filters_applied
)

# Convenience function
evidence_pack: EvidencePack = await search_documents(
    query="mechanism of action",
    section_types=["mechanism", "data_table"],
    top_k=10
)
```

**LLM Integration**:
```python
# Generate response
pipeline = AgenticPipeline()
response: str = await pipeline.create_agentic_response(
    query="How does curcumin affect liver cancer?",
    evidence_pack=evidence_pack,
    provider="openai",
    model="gpt-5-nano"
)
```

### Extension Points

#### Custom Normalizers

```python
# Add domain-specific normalizers
class CustomCompoundNormalizer:
    """Normalize chemical compound names."""
    
    COMPOUND_MAPPINGS = {
        "姜黄素": "curcumin",
        "白藜芦醇": "resveratrol",
        "槲皮素": "quercetin"
    }
    
    def normalize_compound(self, compound_str: str) -> Tuple[str, str]:
        """Return (original, standardized) compound names."""
        standardized = self.COMPOUND_MAPPINGS.get(compound_str, compound_str)
        return compound_str, standardized
```

#### Custom Search Filters

```python
# Add specialized filters
class ToxicologyFilters:
    @staticmethod
    def by_study_type(results: List[RetrievalResult], 
                     study_types: List[str]) -> List[RetrievalResult]:
        """Filter by toxicology study types."""
        study_keywords = {
            "acute_toxicity": ["急性毒性", "LD50", "acute"],
            "chronic_toxicity": ["慢性毒性", "长期", "chronic"],
            "carcinogenicity": ["致癌性", "癌变", "carcinogen"]
        }
        
        filtered = []
        for result in results:
            for study_type in study_types:
                keywords = study_keywords.get(study_type, [])
                if any(kw in result.content for kw in keywords):
                    filtered.append(result)
                    break
        
        return filtered
```

This comprehensive technical documentation and user manual provides complete guidance for both using and extending the ToxiRAG system. The modular architecture supports customization while maintaining the core evidence-based approach essential for toxicology research applications.

---

*End of Document - Last Updated: 2025-08-25*