#!/usr/bin/env python3
"""
Simple health check for the FastAPI service without starting a server.

Uses FastAPI's TestClient to trigger app startup, build/load the RAG pipeline,
then calls /health and /stats to verify readiness.

Exit codes:
- 0: Healthy
- 1: Not ready or error

Usage:
    python scripts/health_check.py

Environment:
    Ensure indices are present under outputs/ so pipeline can load.
"""
import os
import sys
from contextlib import suppress

# Add project root to sys.path so `app.api` resolves when run from anywhere
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient  # type: ignore

with suppress(Exception):
    # Avoid verbose tokenizer parallelism warnings
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from app.api import app
except Exception as e:
    print(f"❌ Failed to import API app: {e}")
    sys.exit(1)

try:
    with TestClient(app) as client:
        # Startup event runs automatically here
        r = client.get("/health")
        if r.status_code != 200:
            print(f"❌ /health returned {r.status_code}: {r.text}")
            sys.exit(1)
        data = r.json()
        if not data.get("pipeline_ready"):
            print("❌ Pipeline not ready")
            sys.exit(1)
        # Optional: stats endpoint
        r_stats = client.get("/stats")
        if r_stats.status_code == 200:
            stats = r_stats.json()
            total_vectors = (
                stats.get("retriever", {}).get("total_vectors")
                or stats.get("total_vectors")
                or stats.get("num_documents")
            )
            print("✓ Health OK | indexed items:", total_vectors)
        else:
            print("✓ Health OK (stats unavailable)")
        sys.exit(0)
except Exception as e:
    print(f"❌ Health check failed: {e}")
    sys.exit(1)
