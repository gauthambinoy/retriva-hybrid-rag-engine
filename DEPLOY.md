# Deployment Guide

Retriva ships as a Dockerized FastAPI app. The saved demo indexes in `outputs/embeddings/` are copied into the image, so the API can start without rebuilding indexes on every deploy.

## Required secret

Set this as a hosting-platform environment variable/secret:

```bash
GEMINI_API_KEY=your_google_ai_studio_key
```

Get a free Gemini key from <https://aistudio.google.com/app/apikey>. Do not commit real keys.

Optional environment variables:

```bash
MODEL_NAME=gemini-2.5-flash
RAG_ENABLE_RERANKER=0
RAG_DISABLE_EMBEDDINGS=0
CORS_ORIGINS=*
LOG_LEVEL=INFO
```

## Local Docker smoke test

```bash
cp .env.example .env   # then edit GEMINI_API_KEY
docker build -t retriva-rag .
docker run --env-file .env -p 8000:8000 retriva-rag
curl http://localhost:8000/health
```

Open <http://localhost:8000/docs> for Swagger UI.

## Option A — Render Web Service (simple/free-to-start)

1. Push the repo to GitHub.
2. Render → **New** → **Web Service** → connect this repository.
3. Runtime: **Docker**.
4. Start command: leave blank; the Dockerfile starts `uvicorn` and respects Render's `PORT`.
5. Add environment variables:
   - `GEMINI_API_KEY`: your key
   - `MODEL_NAME`: `gemini-2.5-flash`
   - `RAG_ENABLE_RERANKER`: `0` for lower memory/latency
   - `CORS_ORIGINS`: your UI domain or `*` for API-only demos
6. Deploy.
7. Verify:

```bash
curl https://<render-service>.onrender.com/health
```

Notes:
- Render free instances may cold-start and may have memory limits. Keep reranking off for the first deploy.
- If you replace the document corpus, rebuild indexes locally and commit the updated `outputs/embeddings/` artifacts, or add a deploy build step that runs `python scripts/build_index.py`.

## Option B — Railway/Fly.io/other Docker hosts

Use the same Docker image settings:

- Exposed port: `8000` internally, or platform-provided `$PORT`.
- Health path: `/health`.
- Required secret: `GEMINI_API_KEY`.
- Start command is already in the Dockerfile.

Verify `/health` and `/docs` after deploy.

## Option C — AWS App Runner via GitHub Actions

This repo includes:

- `.github/workflows/ecr-push.yml` — builds and pushes the Docker image to ECR on `main`.
- `.github/workflows/deploy-apprunner.yml` — manually deploys/updates App Runner via CloudFormation.
- `aws/app-runner-service.yaml` — App Runner service template.

Prerequisites:

1. AWS account and ECR repository named `rag` in `eu-west-1`, or edit workflow env values.
2. GitHub repository variable `AWS_ROLE_TO_ASSUME` with an OIDC role ARN.
3. GitHub secret `GEMINI_API_KEY`.
4. OIDC role permissions for ECR push/read, CloudFormation deploy, App Runner create/update, and scoped `iam:PassRole`.

Deploy flow:

1. Push to `main` to run **Build and Push to ECR**.
2. GitHub Actions → **Deploy App Runner** → **Run workflow**.
3. Use latest image tag or provide a specific tag.
4. Verify the output service URL:

```bash
curl https://<apprunner-url>/health
curl https://<apprunner-url>/docs
```

## Health verification

Without starting a public server:

```bash
python scripts/health_check.py
```

This imports the FastAPI app, triggers startup, loads saved indexes, then checks `/health` and `/stats`.

## Portfolio launch checklist

1. Confirm CI is green on GitHub.
2. Deploy the Docker API and verify `/health`.
3. Add the live `/docs` URL to `README.md`.
4. Record a short GIF/screenshot of `/docs` or the Streamlit UI and add it under `docs/assets/`.
5. Pin the repo and add GitHub topics: `rag`, `llm`, `fastapi`, `faiss`, `bm25`, `gemini`, `docker`, `llmops`.
