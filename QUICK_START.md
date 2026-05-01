# Quick Start — Retriva Hybrid RAG Engine

Use this guide when you want to run the project locally for a recruiter demo.

## 1. Install

```bash
cd /home/gautham/retriva-hybrid-rag-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `GEMINI_API_KEY` in `.env`. Get a free key from <https://aistudio.google.com/app/apikey>.

## 2. Verify the saved index

```bash
python scripts/health_check.py
```

If documents changed, rebuild first:

```bash
python scripts/build_index.py
```

## 3. Start the FastAPI API

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

Open:

- Swagger docs: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health>

Ask a question:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the transformer architecture?","top_k":5,"min_score":0.3}'
```

## 4. Start the Streamlit UI

In another terminal:

```bash
cd /home/gautham/retriva-hybrid-rag-engine
source .venv/bin/activate
streamlit run app/dashboard.py
```

## 5. Run tests

```bash
python -m pytest tests/ -q
```

## 6. Portfolio deployment checklist

1. Follow `DEPLOY.md` to deploy the Docker API.
2. Add `GEMINI_API_KEY` as a secret/env var in the hosting platform.
3. Confirm `/health` returns healthy and `/docs` loads.
4. Add the live URL and a screenshot/GIF to `README.md`.
5. Pin the repository on GitHub.
