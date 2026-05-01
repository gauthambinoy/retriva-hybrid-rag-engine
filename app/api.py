# ==============================================================================
# FILE: api.py
# PURPOSE: FastAPI application for RAG system deployment
# ==============================================================================

"""
REST API for RAG (Retrieval-Augmented Generation) system.

ENDPOINTS:
- GET  /           - API information
- GET  /health     - Health check
- POST /query      - Process RAG query
- GET  /stats      - System statistics

USAGE:
    # Run locally
    uvicorn app.api:app --reload --port 8000
    
    # Test
    curl -X POST http://localhost:8000/query \
      -H "Content-Type: application/json" \
      -d '{"question": "What is transformer architecture?"}'
"""

import os
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import create_pipeline_from_saved

# ==============================================================================
# CONFIGURATION
# ==============================================================================

APP_NAME = "Retriva API"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Retriva — Hybrid RAG Engine for Document Q&A"

# ==============================================================================
# REQUEST/RESPONSE MODELS
# ==============================================================================

class QueryRequest(BaseModel):
    """Request model for /query endpoint."""
    question: str = Field(..., description="User's question", min_length=3)
    top_k: Optional[int] = Field(5, description="Number of chunks to retrieve", ge=1, le=20)
    min_score: Optional[float] = Field(0.3, description="Minimum relevance score", ge=0.0, le=1.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the transformer architecture?",
                "top_k": 5,
                "min_score": 0.3
            }
        }


class QueryResponse(BaseModel):
    """Response model for /query endpoint."""
    question: str
    answer: str
    sources: list[str]
    num_chunks: int
    relevance_scores: list[float]
    model: str
    provider: str
    tokens_used: dict
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the transformer architecture?",
                "answer": "The Transformer is a neural network architecture...",
                "sources": ["Attention_is_all_you_need.pdf"],
                "num_chunks": 5,
                "relevance_scores": [0.44, 0.43, 0.40, 0.39, 0.38],
                "model": "gemini-2.5-flash",
                "provider": "gemini",
                "tokens_used": {"total_tokens": 650}
            }
        }


# ==============================================================================
# APPLICATION INITIALIZATION
# ==============================================================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)

def _cors_origins() -> list[str]:
    """Read CORS origins from env; defaults to demo-friendly wildcard."""
    raw = os.getenv("CORS_ORIGINS", "*")
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or ["*"]


# CORS middleware. Set CORS_ORIGINS to exact domains for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG pipeline instance
pipeline = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup."""
    global pipeline
    
    print(f"\n{'='*80}")
    print(f"STARTING {APP_NAME} v{APP_VERSION}")
    print(f"{'='*80}")
    
    try:
        if not os.getenv('GEMINI_API_KEY'):
            print("⚠ WARNING: GEMINI_API_KEY is not set")
            print("You can still start the API, but generation will fail until a key is provided.")
        
        # Load pipeline
        print("\nLoading RAG pipeline...")
        pipeline = create_pipeline_from_saved()
        
        print(f"\n{'='*80}")
        print(f"✓ {APP_NAME} READY")
        print(f"{'='*80}")
        print(f"\nAPI Documentation: http://localhost:8000/docs")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        raise


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/")
async def root():
    """API information."""
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "description": APP_DESCRIPTION,
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "POST /query": "Process RAG query",
            "GET /stats": "System statistics",
            "GET /docs": "Interactive API documentation"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    if pipeline is None or not pipeline.is_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "status": "healthy",
        "pipeline_ready": pipeline.is_ready
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a RAG query.
    
    Retrieves relevant document chunks and generates an answer using LLM.
    """
    if pipeline is None or not pipeline.is_ready:
        raise HTTPException(status_code=503, detail="RAG system not ready")
    
    try:
        # Process query
        result = pipeline.query(
            question=request.question,
            top_k=request.top_k,
            min_score=request.min_score,
            verbose=False
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/stats")
async def stats():
    """Get system statistics."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        return pipeline.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


# ==============================================================================
# MAIN (for local testing)
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("Starting RAG API server...")
    print("Set GEMINI_API_KEY for live answer generation.")
    print()
    
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
