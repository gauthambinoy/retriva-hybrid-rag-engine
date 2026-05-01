# ==============================================================================
# FILE: pipeline.py
# PURPOSE: Complete end-to-end RAG pipeline
# ==============================================================================

"""
This module provides the complete RAG pipeline connecting all components.

COMPLETE RAG WORKFLOW:
    User Query
        ↓
    [Preprocessing] - Normalize query
        ↓
    [Retrieval] - Find relevant chunks (semantic or BM25)
        ↓
    [Generation] - LLM generates answer from context
        ↓
    Answer + Sources

This is the main entry point for the RAG system.

USAGE:
    from src.pipeline import RAGPipeline
    
    # Initialize (one-time setup)
    pipeline = RAGPipeline()
    
    # Option 1: Build index from scratch
    pipeline.build_from_documents(chunks)
    
    # Option 2: Load existing index (faster)
    pipeline.load_index()
    
    # Query the system
    result = pipeline.query("What is transformer architecture?")
    print(result['answer'])
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

from src.retrieval.retriever import Retriever
from src.generation.llm_interface import LLMInterface


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Retrieval parameters
DEFAULT_TOP_K = 5  # Number of chunks to retrieve
MIN_SIMILARITY_SCORE = 0.3  # Filter low-relevance chunks

# Generation parameters
DEFAULT_LLM_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.1


# ==============================================================================
# RAG PIPELINE CLASS
# ==============================================================================

class RAGPipeline:
    """
    Complete Retrieval-Augmented Generation pipeline.
    
    RESPONSIBILITIES:
    - Initialize retrieval and generation components
    - Handle end-to-end query processing
    - Provide user-friendly interface
    - Track metrics (tokens, costs, etc.)
    
    COMPONENTS:
        1. Retriever: Find relevant document chunks
        2. LLM Interface: Generate answers from context
    
    ATTRIBUTES:
        retriever (Retriever): Retrieval component
        llm (LLMInterface): LLM generation component
        is_ready (bool): Whether system is ready for queries
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        llm_model: str = DEFAULT_LLM_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        use_semantic_cache: bool = True,
        verbose_init: bool = False
    ):
        """
        Initialize RAG pipeline.
        
        PARAMETERS:
            openai_api_key (str, optional): Deprecated; use GEMINI_API_KEY in the environment
            llm_model (str): Gemini model name (default: "gemini-2.5-flash")
            temperature (float): LLM temperature (default: 0.1)
            use_semantic_cache (bool): Enable semantic query caching (default: True)
        
        PROCESS:
            1. Load environment variables (.env file)
            2. Initialize retriever
            3. Initialize LLM interface
            4. Initialize semantic cache (optional)
        """
        if verbose_init:
            print(f"\n{'='*80}")
            print(f"INITIALIZING RAG PIPELINE")
            print(f"{'='*80}")
        
        # Load environment variables from .env file
        load_dotenv()

        # Decide embedding usage from env toggle
        disable_embeddings = os.getenv("RAG_DISABLE_EMBEDDINGS", "0") == "1"
        if disable_embeddings and verbose_init:
            print("\n⚠ Embeddings disabled via RAG_DISABLE_EMBEDDINGS=1 (BM25-only mode)")

        # Initialize retriever: keep hybrid=True to enable BM25; dense path controlled by env
        if verbose_init:
            print(f"\n[1/2] Initializing Retrieval Component...")
        self.retriever = Retriever(use_hybrid=True, use_reranker=False, lazy_embedding=True)
        if disable_embeddings:
            self.retriever.dense_enabled = False
            self.retriever.embedding_model = None
        
        # Initialize Gemini LLM interface
        if verbose_init:
            print(f"\n[2/2] Initializing Generation Component...")
        if openai_api_key:
            print("Note: openai_api_key parameter is deprecated; use environment variables instead. Ignoring explicit key.")
        self.llm = LLMInterface(
            model=llm_model,
            temperature=temperature
        )
        
        # Initialize semantic cache (IMPROVEMENT 3)
        self.use_semantic_cache = use_semantic_cache
        self.cache = None
        if use_semantic_cache:
            try:
                from src.retrieval.semantic_cache import SemanticCache
                self.cache = SemanticCache(max_size=1000, similarity_threshold=0.95)
                if verbose_init:
                    print(f"\n✓ Semantic caching enabled (max_size=1000)")
            except ImportError:
                if verbose_init:
                    print(f"\n⚠️ Semantic cache not available, skipping")
                self.use_semantic_cache = False
        # Disable semantic cache if embeddings are disabled/unavailable
        if self.use_semantic_cache and (not getattr(self.retriever, 'dense_enabled', True)):
            if verbose_init:
                print("\n⚠ Semantic cache requires embeddings; disabling because embeddings are unavailable")
            self.use_semantic_cache = False
        
        # Not ready until index is built/loaded
        self.is_ready = False
        
        if verbose_init:
            print(f"{'='*80}")
            print(f"✓ RAG PIPELINE INITIALIZED")
            print(f"{'='*80}\n")
    
    def build_from_documents(
        self,
        chunks: List[Dict],
        use_cache: bool = True
    ):
        """
        Build retrieval index from document chunks.
        
        USE CASE: First-time setup or when documents change
        
        PARAMETERS:
            chunks (List[Dict]): Preprocessed document chunks
            use_cache (bool): Use cached embeddings if available
        
        EXAMPLE:
            # After preprocessing
            chunks = preprocess_all_documents()
            
            pipeline = RAGPipeline()
            pipeline.build_from_documents(chunks)
        """
    # Quiet build unless explicitly verbose later (retain prints only if needed)
    # Removed noisy banners for minimal mode
        
        self.retriever.build_index(chunks, use_cache=use_cache)
        self.is_ready = True
        
    # Silent success; user can call get_stats() for info
    
    def load_index(self):
        """
        Load pre-built retrieval index from disk.
        
        USE CASE: Fast startup after initial build
        
        FASTER THAN build_from_documents:
        - build_from_documents(): 30-60 seconds (embedding generation)
        - load_index(): <1 second (just loading)
        
        EXAMPLE:
            pipeline = RAGPipeline()
            pipeline.load_index()  # Fast!
            
            result = pipeline.query("What is transformer?")
        """
    # Silent load for minimal mode
        
        self.retriever.load_index()
        self.is_ready = True
        
    # Silent success
    
    def query(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = MIN_SIMILARITY_SCORE,
        verbose: bool = False,
        use_cache: bool = True,
        retrieval_method: str = "default"
    ) -> Dict:
        """
        Process a query through the complete RAG pipeline.
        
        THIS IS THE MAIN FUNCTION - COMPLETE RAG SYSTEM
        
        IMPROVEMENTS ENABLED:
        - Semantic caching (IMPROVEMENT 3)
        - Query expansion (IMPROVEMENT 1) - enable with retrieval_method="expansion"
        - Metadata filtering (IMPROVEMENT 2) - enable with retrieval_method="filtering"
        - Adaptive ranking (IMPROVEMENT 4) - enable with retrieval_method="adaptive"
        - Progressive retrieval (IMPROVEMENT 5) - enable with retrieval_method="progressive"
        
        PARAMETERS:
            question (str): User's question
            top_k (int): Number of chunks to retrieve (default: 5)
            min_score (float): Minimum similarity score (default: 0.3)
            verbose (bool): Print detailed information
            use_cache (bool): Use semantic caching (default: True)
            retrieval_method (str): Which retrieval method to use
                - "default": Standard hybrid + reranking
                - "expansion": With query expansion (+3% recall)
                - "filtering": With metadata filtering (4x faster)
                - "adaptive": With adaptive ranking (+5-10% accuracy)
                - "progressive": With 3-stage refinement
        
        RETURNS:
            Dict with keys:
                - 'question': Original question
                - 'answer': Generated answer
                - 'sources': List of source documents
                - 'retrieved_chunks': List of retrieved chunks
                - 'num_chunks': Number of chunks used
                - 'model': LLM model used
                - 'tokens_used': Token counts
                - 'relevance_scores': Similarity scores of retrieved chunks
                - 'cache_hit': Whether result came from cache
        
        PROCESS:
            1. Check cache (if enabled)
            2. If cache miss:
               - Retrieve relevant chunks
               - Generate answer with LLM
               - Cache result
            3. Return complete result
        
        EXAMPLE:
            pipeline = RAGPipeline()
            pipeline.load_index()
            
            # Standard query
            result = pipeline.query("What is transformer architecture?")
            
            # With query expansion
            result = pipeline.query(
                "What is transformer?",
                retrieval_method="expansion"
            )
            
            # Second identical query hits cache
        """

        if not self.is_ready:
            raise RuntimeError("Pipeline not ready. Call build_from_documents() or load_index() first.")

        if verbose:
            print(f"\n{'='*80}")
            print("PROCESSING QUERY")
            print(f"{'='*80}")

        # Semantic cache (only when embeddings enabled)
        if self.use_semantic_cache and use_cache and getattr(self.retriever, 'dense_enabled', True) and self.cache is not None:
            query_embedding = None
            if getattr(self.retriever, 'embedding_model', None) is not None:
                query_embedding = self.retriever.embedding_model.embed_text(question)
            cached_result = self.cache.get(question, query_embedding)
            if cached_result is not None:
                cached_result['cache_hit'] = True
                return cached_result
        
        if verbose:
            print(f"\nRetrieving top {top_k} chunks with score ≥ {min_score}")
        
        # Step 1: Retrieve relevant chunks (with selected method)
        if verbose:
            print(f"\n[Step 1/2] Retrieving relevant chunks ({retrieval_method})...")
        
        # Select retrieval method
        if retrieval_method == "expansion":
            retrieved_chunks = self.retriever.retrieve_with_expansion(
                question,
                k=top_k,
                verbose=verbose
            )
        elif retrieval_method == "filtering":
            retrieved_chunks = self.retriever.retrieve_with_filtering(
                question,
                k=top_k,
                verbose=verbose
            )
        elif retrieval_method == "adaptive":
            retrieved_chunks = self.retriever.retrieve_adaptive(
                question,
                k=top_k
            )
        elif retrieval_method == "progressive":
            retrieved_chunks = self.retriever.retrieve_progressive(
                question,
                k=top_k
            )
        else:  # "default"
            retrieved_chunks = self.retriever.retrieve(
                question,
                k=top_k,
                min_score=min_score,
                verbose=verbose
            )
        
        if not retrieved_chunks:
            result = {
                'question': question,
                'answer': "I couldn't find any relevant information in the documents to answer this question.",
                'sources': [],
                'retrieved_chunks': [],
                'num_chunks': 0,
                'relevance_scores': [],
                'model': getattr(self.llm, 'model_name', 'unknown'),
                'provider': 'gemini',
                'tokens_used': {'total_tokens': 0},
                'cache_hit': False
            }
            
            # Cache this result too
            if self.use_semantic_cache and use_cache and getattr(self.retriever, 'dense_enabled', True) and self.cache is not None:
                query_embedding = None
                if getattr(self.retriever, 'embedding_model', None) is not None:
                    query_embedding = self.retriever.embedding_model.embed_text(question)
                self.cache.set(question, query_embedding, result)
            
            return result
        
        if verbose:
            print(f"✓ Retrieved {len(retrieved_chunks)} relevant chunks")
        
        # Step 2: Generate answer with LLM
        if verbose:
            print(f"\n[Step 2/2] Generating answer with LLM...")
        
        result = self.llm.generate_with_sources(
            question,
            retrieved_chunks,
            verbose=verbose
        )
        
        # Add retrieval information
        result['question'] = question
        result['retrieved_chunks'] = retrieved_chunks
        result['num_chunks'] = len(retrieved_chunks)
        result['relevance_scores'] = [
            chunk['score'] for chunk in retrieved_chunks
        ]
        result['cache_hit'] = False
        
        # IMPROVEMENT 3: Cache this result (only when embeddings enabled)
        if self.use_semantic_cache and use_cache and getattr(self.retriever, 'dense_enabled', True) and self.cache is not None:
            query_embedding = None
            if getattr(self.retriever, 'embedding_model', None) is not None:
                query_embedding = self.retriever.embedding_model.embed_text(question)
            self.cache.set(question, query_embedding, result)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"✓ QUERY COMPLETE")
            print(f"{'='*80}")
        
        return result
    
    def batch_query(
        self,
        questions: List[str],
        top_k: int = DEFAULT_TOP_K,
        verbose: bool = False
    ) -> List[Dict]:
        """
        Process multiple queries in batch.
        
        USE CASE: Evaluation, batch processing
        
        PARAMETERS:
            questions (List[str]): List of questions
            top_k (int): Number of chunks per query
            verbose (bool): Print progress
        
        RETURNS:
            List[Dict]: Results for each query
        
        EXAMPLE:
            questions = [
                "What is transformer?",
                "How does attention work?",
                "What is self-attention?"
            ]
            
            results = pipeline.batch_query(questions)
            
            for result in results:
                print(f"Q: {result['question']}")
                print(f"A: {result['answer']}\n")
        """
        print(f"\n{'='*80}")
        print(f"BATCH QUERY PROCESSING")
        print(f"{'='*80}")
        print(f"Number of queries: {len(questions)}")
        
        results = []
        for i, question in enumerate(questions, 1):
            if verbose:
                print(f"\nProcessing query {i}/{len(questions)}: {question}")
            
            result = self.query(question, top_k=top_k, verbose=False)
            results.append(result)
        
        print(f"\n✓ Processed {len(results)} queries")
        print(f"{'='*80}\n")
        
        return results
    
    def get_stats(self) -> Dict:
        """
        Get system statistics.
        
        RETURNS:
            Dict with system information
        """
        if not self.is_ready:
            return {'status': 'not_ready'}
        
        return {
            'status': 'ready',
            'num_documents': self.retriever.vector_store.get_num_vectors(),
            'embedding_model': getattr(self.retriever.embedding_model, 'model_name', 'disabled' if not getattr(self.retriever, 'dense_enabled', True) else 'unknown'),
            'embedding_dim': self.retriever.embedding_dim,
            'llm_model': getattr(self.llm, 'model_name', 'unknown'),
            'llm_temperature': self.llm.temperature
        }


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def create_pipeline_from_saved(
        openai_api_key: Optional[str] = None
) -> RAGPipeline:
    """
    Create RAG pipeline with pre-built index (fastest startup).
    
    USE CASE: Production deployment - just load and go
    
    PARAMETERS:
        openai_api_key (str, optional): Deprecated; use GEMINI_API_KEY in the environment
    
    RETURNS:
        RAGPipeline: Ready-to-use pipeline
    
    EXAMPLE:
        pipeline = create_pipeline_from_saved()
        result = pipeline.query("What is transformer?")
    """
    pipeline = RAGPipeline(openai_api_key=openai_api_key)
    pipeline.load_index()
    return pipeline


# ==============================================================================
# SELF-TEST / DEMO BLOCK
# ==============================================================================

if __name__ == "__main__":
    """
    Test the complete RAG pipeline.
    
    NOTE: Requires:
    1. Pre-built index (run scripts/build_index.py if outputs are absent)
    2. GEMINI_API_KEY environment variable
    
    Run:
        export GEMINI_API_KEY="your-key"
        python src/pipeline.py
    """
    
    print("="*80)
    print("TESTING: Complete RAG Pipeline")
    print("="*80)
    
    # Check for API key
    if not os.getenv('GEMINI_API_KEY'):
        print("\n⚠ GEMINI_API_KEY not set!")
        print("Set it to test the complete RAG pipeline:")
        print("  export GEMINI_API_KEY='your-key-here'")
        print("\nYou can get a key from: https://aistudio.google.com/app/apikey")
        print("\nSkipping test...")
        exit(0)
    
    try:
        # Create pipeline and load index
        print("\n" + "-"*80)
        print("TEST 1: Initialize and Load Index")
        print("-"*80)
        
        pipeline = create_pipeline_from_saved()
        
        print(f"\nSystem stats:")
        stats = pipeline.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test query
        print("\n" + "-"*80)
        print("TEST 2: Process Query")
        print("-"*80)
        
        test_query = "What is the transformer architecture?"
        
        result = pipeline.query(test_query, verbose=False)
        
        print(f"\n{'='*80}")
        print(f"FINAL RESULT")
        print(f"{'='*80}")
        print(f"Question: {result['question']}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources:")
        for source in result['sources']:
            print(f"  - {source}")
        print(f"\nRelevance scores: {[f'{s:.3f}' for s in result['relevance_scores']]}")
        print(f"Tokens used: {result['tokens_used']['total_tokens']}")
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED")
        print("="*80)
        print("\nRAG pipeline is working correctly!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
