# Advanced RAG System

> Production-ready Retrieval-Augmented Generation for Enterprise Q&A

[![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python)](https://python.org)
[![Docker](https://img.shields.io/badge/docker-ready-blue?logo=docker)](https://docker.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-orange?logo=google)](https://ai.google.dev)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://github.com/gauthambinoy/advanced-rag-system/actions/workflows/ci.yml/badge.svg)](https://github.com/gauthambinoy/advanced-rag-system/actions/workflows/ci.yml)

A complete, production-grade RAG system that answers questions from a private document knowledge base. Features hybrid BM25 + FAISS retrieval, cross-encoder reranking, semantic caching, multi-format document ingestion, and a FastAPI REST interface with an interactive Streamlit UI.

---

## Architecture

```
User Query
    │
    ▼
[Semantic Cache] ──(hit)──────────────────────────────► Cached Answer
    │ (miss)
    ▼
[Query Preprocessing]
(normalisation)
    │
    ├──► [Dense Retrieval]   all-MiniLM-L6-v2 → FAISS index
    │
    ├──► [Sparse Retrieval]  BM25 keyword index
    │
    ▼
[Reciprocal Rank Fusion]
(hybrid result merging)
    │
    ▼
[Cross-Encoder Reranking]   ← optional (RAG_ENABLE_RERANKER=1)
    │
    ▼
[Context Formatting + Source Citation]
    │
    ▼
[Gemini 2.5 Flash LLM Generation]
    │
    ▼
Answer + [Source: file.pdf, Page: N]
```

---

## Features

- **Hybrid Retrieval** — BM25 sparse + FAISS dense search fused with Reciprocal Rank Fusion
- **Cross-Encoder Reranking** — Optional precision boost via `RAG_ENABLE_RERANKER=1`
- **Semantic Cache** — Near-instant responses for repeated queries (cosine similarity threshold 0.95)
- **Source Citations** — Every answer includes `[Source: file.pdf, Page: N]` references
- **Multi-format Ingestion** — PDF, DOCX, XLSX, TXT document loaders
- **Query Expansion** — Automatic sub-query generation for broader recall
- **Metadata Filtering** — 4x faster retrieval with document-level filters
- **Adaptive Ranking** — Dynamic re-ranking based on query type (+5–10% accuracy)
- **FastAPI REST API** — Production-ready `/query`, `/health`, `/stats` endpoints
- **Streamlit UI** — Interactive chat interface with source viewer
- **Docker Ready** — Single-command containerised deployment
- **Multi-provider LLM** — Gemini (default), OpenAI, OpenRouter fallback chain

---

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/gauthambinoy/advanced-rag-system.git
cd advanced-rag-system
cp .env.example .env          # add your GEMINI_API_KEY
docker build -t rag-system .
docker run -p 8000:8000 --env-file .env rag-system
```

### Manual Setup

```bash
git clone https://github.com/gauthambinoy/advanced-rag-system.git
cd advanced-rag-system

python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env            # add your GEMINI_API_KEY

# Place documents in data/ then build the index
python scripts/build_index.py

# Start the API
uvicorn app.api:app --reload --port 8000

# Or start the Streamlit UI (separate terminal)
streamlit run app/dashboard.py
```

---

## API Reference

### `POST /query`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the transformer architecture?", "top_k": 5}'
```

**Response:**
```json
{
  "question": "What is the transformer architecture?",
  "answer": "The Transformer is a neural network architecture... [Source: attention.pdf, Page: 2]",
  "sources": ["attention.pdf"],
  "num_chunks": 5,
  "relevance_scores": [0.44, 0.43, 0.40, 0.39, 0.38],
  "model": "gemini-2.5-flash",
  "provider": "gemini",
  "tokens_used": {"total_tokens": 650}
}
```

### `GET /health`
Returns `{"status": "healthy", "pipeline_ready": true}`

### `GET /stats`
Returns document count, embedding model, LLM config.

### `GET /docs`
Interactive Swagger UI.

---

## Configuration

| Variable | Description | Default |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API key | required |
| `OPENAI_API_KEY` | OpenAI fallback key | optional |
| `OPENROUTER_API_KEY` | OpenRouter fallback key | optional |
| `RAG_ENABLE_RERANKER` | Enable cross-encoder reranking | `0` |
| `RAG_DISABLE_EMBEDDINGS` | BM25-only mode (no GPU needed) | `0` |

---

## Project Structure

```
advanced-rag-system/
├── app/
│   ├── api.py              # FastAPI REST API
│   └── dashboard.py        # Streamlit chat UI
├── src/
│   ├── pipeline.py         # End-to-end RAG pipeline
│   ├── loaders/            # PDF, DOCX, XLSX document loaders
│   ├── preprocessing/      # Chunking, normalisation
│   ├── retrieval/          # FAISS, BM25, reranker, semantic cache
│   ├── generation/         # LLM interface, prompt templates
│   └── evaluation/         # Precision@K, MRR metrics
├── scripts/
│   └── build_index.py      # Index construction script
├── tests/                  # Test suite
├── data/                   # Place your documents here
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Retrieval Methods

| Method | Flag | Use case |
|---|---|---|
| Default hybrid | `"default"` | General purpose |
| Query expansion | `"expansion"` | Broad/ambiguous queries (+3% recall) |
| Metadata filtering | `"filtering"` | Known document subset (4x faster) |
| Adaptive ranking | `"adaptive"` | Mixed query types (+5–10% accuracy) |
| Progressive | `"progressive"` | High-precision 3-stage refinement |

---

## License

[MIT](LICENSE) © Gautham Binoy
