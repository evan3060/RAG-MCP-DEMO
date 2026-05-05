# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
pytest tests/ -v

# Run unit tests only
pytest tests/unit/ -v

# Run integration tests only
pytest tests/integration/ -v

# Run single test file
pytest tests/unit/test_pipeline.py -v

# Run single test function
pytest tests/unit/test_pipeline.py::test_ask_without_index -v

# Run tests with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run tests excluding slow/API tests
pytest tests/ -v -m "not slow and not requires_api"

# Verify environment setup
python scripts/verify_setup.py

# Start MCP server (STDIO mode)
python -m src.mcp_server.server

# Start HTTP API server
python scripts/start_api_server.py

# Test models
python scripts/test_models.py

# Test full pipeline
python scripts/test_full_pipeline.py

# Run evaluation
python scripts/evaluate_with_ground_truth.py
```

## Architecture

### Three-Layer Design

**1. MCP/API Layer** (`src/mcp_server/server.py`, `src/api/server.py`)
- MCP server exposes `ingest_document`, `ask_question`, `search_knowledge` tools over STDIO/SSE
- FastAPI server wraps the MCP server for HTTP access, adds file upload (`/api/upload`), CORS, and serves the frontend SPA
- Both layers lazily initialize `RAGMCPServer` → `RAGPipeline`

**2. RAG Pipeline** (`src/rag/llamaindex/`)
- `RAGPipeline`: orchestrates index building (Chroma persistence), hybrid retrieval, and QA
- `SmartTextProcessor`: cleans text, detects doc type (general/technical), parses structure (headings/code/tables), and chunks with overlap
- `HybridRetriever`: combines vector search (top-20) + BM25 (top-20) → RRF fusion (top-10)

**3. Pluggable Components** (`src/rag/components/`)
- `embedders/` — `BaseEmbedder` → `SiliconFlowEmbedder` (extends LlamaIndex `BaseEmbedding`)
- `llms/` — `BaseLLM` → `SiliconFlowLLM`, `QianfanLLM`, `OpenAICompatibleLLM` + `LlamaIndexLLMAdapter`
- `rerankers/` — `BaseReranker` → `SiliconFlowReranker`
- `vector_stores/` — `BaseVectorStore` → `ChromaVectorStore`
- `loaders/` — `PDFLoader`, `DocxLoader`, `ExcelLoader`

### Key Data Flow

```
User → MCP/API → RAGPipeline → HybridRetriever → Chroma + BM25 → LLM → Answer + Sources
```

Document ingestion: File → Loader → SmartTextProcessor (clean + chunk) → Embedder → Chroma

### Configuration

Config loads from environment variables first (preferred), falling back to `config/default.yaml`. The config system supports `${VAR:-default}` expansion. Providers: `qianfan`, `siliconflow`, `openai`, `litellm`.

### Important Notes

- **nest_asyncio**: Required when running async RAG code inside FastAPI (applied in `src/api/server.py`)
- **MockEmbedding**: `Settings.embed_model = MockEmbedding(embed_dim=1024)` is set at module level in `pipeline.py` to suppress LlamaIndex's default OpenAI check
- **Chroma metadata**: ChromaDB only supports simple types (str, int, float) in metadata — lists are converted to comma-separated strings in `SmartTextProcessor._create_node`
- **BM25 lazy init**: `HybridRetriever` lazily initializes BM25 on first query; if docstore is empty, falls back to vector-only search
- **Test markers**: `unit`, `integration`, `slow`, `requires_api` — defined in `tests/conftest.py`
- **Venv**: Use `venv/bin/python` — Python 3.10+
