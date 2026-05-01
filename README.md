# Retriva — Hybrid RAG Engine

> Production-ready Retrieval-Augmented Generation for private document Q&A.

[![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-ff4b4b?logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ed?logo=docker)](https://docker.com)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-orange?logo=google)](https://ai.google.dev)
[![CI](https://github.com/gauthambinoy/retriva-hybrid-rag-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/gauthambinoy/retriva-hybrid-rag-engine/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Retriva is a portfolio-grade hybrid RAG system that answers questions over private documents using **BM25 + FAISS retrieval**, optional **cross-encoder reranking**, source-grounded prompts, a **FastAPI** API, and a lightweight **Streamlit** demo UI.

It is designed to show production AI engineering judgment: reproducible setup, saved indexes, health checks, Docker deployment, CI, evaluation reports, and honest notes about API keys/secrets.

---

## Demo

> Live demo: add the deployed URL here after following [DEPLOY.md](DEPLOY.md).
>
> Screenshot/GIF placeholder: add `docs/assets/retriva-demo.gif` or a screenshot once the API/UI is deployed. Do not commit API keys or private documents.

Local API docs are available at `http://localhost:8000/docs` after startup.

---

## What this project demonstrates

- **Hybrid retrieval** — sparse BM25 + dense FAISS search with reciprocal-rank-style merging.
- **Optional reranking** — cross-encoder reranker toggle via `RAG_ENABLE_RERANKER=1`.
- **Grounded generation** — Gemini answer generation with context-only instructions and source citations.
- **Semantic cache** — fast repeated-query responses when embeddings are enabled.
- **Multi-format ingestion** — PDF, DOCX, XLSX/XLS, and text-oriented preprocessing paths.
- **Production API** — FastAPI `/query`, `/health`, `/stats`, and OpenAPI `/docs`.
- **Interactive UI** — Streamlit question runner for demos and recruiter walkthroughs.
- **Operational readiness** — Dockerfile, GitHub Actions CI, health-check script, and deployment guide.
- **Evaluation artifacts** — retrieval/generation metrics under `outputs/evaluations/`.

---

## Architecture

```text
User Query
    │
    ▼
[FastAPI / Streamlit]
    │
    ▼
[Semantic Cache] ── hit ───────────────────────────────► Cached Answer
    │ miss
    ▼
[Query Normalization]
    │
    ├──► [Dense Retrieval]   SentenceTransformers → FAISS
    │
    ├──► [Sparse Retrieval]  BM25 keyword index
    │
    ▼
[Hybrid Result Fusion]
    │
    ▼
[Optional Cross-Encoder Reranking]
    │
    ▼
[Prompt Builder + Source Formatting]
    │
    ▼
[Gemini LLM]
    │
    ▼
Answer + citations + source metadata
```

### Request flow

1. Load saved indexes from `outputs/embeddings/` at API startup.
2. Retrieve candidate chunks from the private corpus.
3. Generate an answer constrained to retrieved context.
4. Return answer, source list, scores, model/provider metadata, and token metadata.

---

## Quick start

### 1) Clone and configure

```bash
git clone https://github.com/gauthambinoy/retriva-hybrid-rag-engine.git
cd retriva-hybrid-rag-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set:

```bash
GEMINI_API_KEY=your_google_ai_studio_key
```

Get a free Gemini key from Google AI Studio: <https://aistudio.google.com/app/apikey>.

### 2) Build or reuse the index

This repo includes saved demo indexes in `outputs/embeddings/`. If you change documents under `data/raw_documents/`, rebuild:

```bash
python scripts/build_index.py
```

For lightweight local/CI runs without dense embedding downloads:

```bash
RAG_DISABLE_EMBEDDINGS=1 python scripts/build_index.py
```

### 3) Run the API

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

Then open:

- API docs: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health>

### 4) Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the transformer architecture?","top_k":5,"min_score":0.3}'
```

### 5) Run the Streamlit UI

```bash
streamlit run app/dashboard.py
```

---

## Docker

```bash
cp .env.example .env   # add GEMINI_API_KEY
docker build -t retriva-rag .
docker run --env-file .env -p 8000:8000 retriva-rag
```

Verify:

```bash
curl http://localhost:8000/health
```

---

## API reference

### `POST /query`

Request:

```json
{
  "question": "What is the transformer architecture?",
  "top_k": 5,
  "min_score": 0.3
}
```

Response shape:

```json
{
  "question": "What is the transformer architecture?",
  "answer": "The Transformer is a neural network architecture... [C1]",
  "sources": ["Attention_is_all_you_need.pdf"],
  "num_chunks": 5,
  "relevance_scores": [0.44, 0.43, 0.40, 0.39, 0.38],
  "model": "gemini-2.5-flash",
  "provider": "gemini",
  "tokens_used": {"total_tokens": "N/A"}
}
```

### Other endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | API metadata |
| `GET /health` | Readiness check; returns 503 until the pipeline is loaded |
| `GET /stats` | Retriever/model stats |
| `GET /docs` | Swagger/OpenAPI UI |

---

## Configuration

| Variable | Required | Default | Description |
|---|---:|---|---|
| `GEMINI_API_KEY` | yes for generation | unset | Google Gemini API key. API starts without it, but `/query` returns a missing-key error. |
| `MODEL_NAME` | no | `gemini-2.5-flash` | Gemini model preference. |
| `RAG_ENABLE_RERANKER` | no | `0` | Enable cross-encoder reranking (`1`) for higher precision and more latency. |
| `RAG_DISABLE_EMBEDDINGS` | no | `0` | Use BM25-only mode; useful in CI or low-memory environments. |
| `CORS_ORIGINS` | no | `*` | Comma-separated allowed origins for browsers. Use exact domains in production. |
| `LOG_LEVEL` | no | `INFO` | Logging verbosity for deployments. |

---

## Testing and validation

```bash
# Unit/integration tests
python -m pytest tests/ -q

# API readiness without starting a public server
python scripts/health_check.py
```

CI runs dependency installation and `pytest` on pushes/PRs to `main` via `.github/workflows/ci.yml`.

---

## Project structure

```text
retriva-hybrid-rag-engine/
├── app/
│   ├── api.py                    # FastAPI app
│   └── dashboard.py              # Streamlit demo UI
├── src/
│   ├── pipeline.py               # End-to-end RAG orchestration
│   ├── loaders/                  # PDF, DOCX, XLS/XLSX loaders
│   ├── preprocessing/            # Normalization and chunking
│   ├── retrieval/                # FAISS, BM25, reranking, caching
│   ├── generation/               # Gemini interface and prompts
│   └── evaluation/               # Metrics utilities
├── scripts/
│   ├── build_index.py            # Build saved retrieval indexes
│   └── health_check.py           # Deployment readiness check
├── outputs/                      # Saved indexes and evaluation reports
├── data/raw_documents/           # Demo/private document corpus
├── .github/workflows/            # CI and deployment workflows
├── Dockerfile
├── DEPLOY.md
└── requirements.txt
```

---

## Deployment

See [DEPLOY.md](DEPLOY.md) for realistic free/low-cost deployment steps.

Recommended portfolio path:

1. Deploy the Docker API to Render or AWS App Runner.
2. Set `GEMINI_API_KEY` as a platform secret/environment variable.
3. Verify `/health` and `/docs`.
4. Add the live API docs URL and a GIF/screenshot to this README.
5. Pin the repo on GitHub with topics like `rag`, `fastapi`, `faiss`, `bm25`, `gemini`, `llmops`.

---

## License

[MIT](LICENSE) © Gautham Binoy
