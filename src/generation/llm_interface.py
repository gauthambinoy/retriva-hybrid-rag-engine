# ==============================================================================
# FILE: llm_interface.py
# PURPOSE: LLM integration for generation component of RAG
# ==============================================================================

"""
This module handles LLM API integration for answer generation.

WHAT THIS MODULE DOES:
- Takes retrieved context + user query
- Sends retrieved context to Gemini with a carefully crafted prompt
- Returns generated answer

THIS IS THE "G" IN RAG (Retrieval-Augmented Generation)

LLM CHOICE: Google Gemini (2.x / 1.5 series)
RATIONALE:
- Strong reasoning and instruction following
- Excellent cost/latency trade-off (Flash for speed)
- Reliable API with simple deployment

ALTERNATIVE OPTIONS (Trade-offs):
1. Anthropic Claude
   - Pros: Better at long context, more cautious (less hallucination)
   - Cons: More expensive, slower API
   - When to use: Legal/medical domains where accuracy critical

2. Local Models (Llama, Mistral)
   - Pros: No API costs, full control, data privacy
   - Cons: Requires GPU/compute, harder deployment, lower quality
   - When to use: Privacy-critical applications, high volume

3. Azure OpenAI (not used here)
    - Pros: Enterprise SLA, data residency options
    - Cons: Setup complexity
    - When to use: Enterprise constraints (out of scope here)

CHOSEN: Gemini 2.5 Flash (primary) / Gemini Pro as fallback model id
- Fast response times (~1-2 seconds)
- Low cost
- Multi-modal ready (future-proof)

PROMPT ENGINEERING STRATEGY:
- Clear instruction: "Answer based only on context"
- Context injection: Retrieved chunks provided
- Hallucination prevention: "If not in context, say so"
- Format control: Specify answer format
- Few-shot examples: (optional) Show good answers

USAGE:
    from src.generation.llm_interface import LLMInterface
    
    # Initialize (requires GEMINI_API_KEY environment variable for live generation)
    llm = LLMInterface()
    
    # Generate answer
    answer = llm.generate_answer(
        query="What is transformer architecture?",
        context="The Transformer is a neural network..."
    )
"""

import os
from typing import Optional, Dict, List, Tuple
from dotenv import load_dotenv
import requests
import google.generativeai as genai

load_dotenv()

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Model configuration
DEFAULT_MODEL = "gemini-2.5-flash"  # Default Gemini model

# Generation parameters
DEFAULT_TEMPERATURE = 0.1  # Low temperature = more deterministic
DEFAULT_MAX_TOKENS = 500  # Maximum answer length
DEFAULT_TOP_P = 0.95  # Nucleus sampling parameter


# ==============================================================================
# PROMPTS
# ==============================================================================

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on provided context.

INSTRUCTIONS:
1. Answer the question using ONLY the information in the provided context
2. If the answer is not in the context, explicitly state: "I cannot answer this based on the provided documents."
3. Be concise and accurate
4. If the context has conflicting information, mention it
5. Do not make up or infer information not present in the context
6. Add bracketed citations like [C1], [C2] whenever you use facts, where numbers correspond to Context i blocks below.

RESPONSE FORMAT:
- Direct answer first
- Then supporting evidence from context with citations [C#]
- Keep answers clear and structured
"""

USER_PROMPT_TEMPLATE = """Context from relevant documents:
{context}

Question: {query}

Answer the question based on the context above. If the answer is not in the context, say so clearly."""


# ==============================================================================
# LLM INTERFACE CLASS
# ==============================================================================

class LLMInterface:
    """Gemini-only LLM interface.

    Attempts a list of Gemini model identifiers; returns first successful response.
    Environment variables:
      GEMINI_API_KEY (required for generation)
      MODEL_NAME (optional override for default model)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = os.getenv("MODEL_NAME", model)
        self.gemini_key = api_key or os.getenv("GEMINI_API_KEY")

        print("\n" + "=" * 80)
        print("INITIALIZING LLM INTERFACE (Gemini only)")
        print("=" * 80)
        print(f"Model preference: {self.model_name}")
        print(f"Gemini key: {'set' if self.gemini_key else 'not set'}")
        print("✓ LLM interface initialized\n")

    def generate_answer(
        self,
        query: str,
        context: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict:
        """Generate answer using Gemini models only."""
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        user_prompt = USER_PROMPT_TEMPLATE.format(context=context, query=query)

        if not self.gemini_key:
            return {
                'answer': 'Error: GEMINI_API_KEY not configured',
                'model': self.model_name,
                'provider': 'gemini',
                'tokens_used': {'total_tokens': 0},
                'query': query,
                'error': 'missing_api_key'
            }

        gemini_candidates: List[str] = []
        if self.model_name.startswith("gemini"):
            gemini_candidates.append(self.model_name)
        for m in [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash-8b",
            "gemini-pro",
        ]:
            if m not in gemini_candidates:
                gemini_candidates.append(m)

        errors: List[str] = []
        genai.configure(api_key=self.gemini_key)

        for cand in gemini_candidates:
            try:
                ids_to_try = [cand]
                if not cand.startswith("models/"):
                    ids_to_try.append(f"models/{cand}")
                model_obj = None
                last_error = None
                for mid in ids_to_try:
                    try:
                        model_obj = genai.GenerativeModel(mid)
                        break
                    except Exception as ge_inner:
                        last_error = ge_inner
                        continue
                if model_obj is None:
                    raise last_error or Exception("model init failed")
                response = model_obj.generate_content([
                    SYSTEM_PROMPT,
                    user_prompt,
                ], generation_config=genai.types.GenerationConfig(temperature=temp))
                answer = (getattr(response, 'text', '') or '').strip()
                if answer:
                    return {
                        'answer': answer,
                        'model': cand,
                        'provider': 'gemini',
                        'tokens_used': {'total_tokens': 'N/A'},
                        'query': query,
                    }
                errors.append(f"empty response {cand}")
            except Exception as e:
                errors.append(f"{cand}: {e}")
                if verbose:
                    print(f"Gemini model {cand} failed: {e}")

        return {
            'answer': 'Error: all Gemini variants failed',
            'model': self.model_name,
            'provider': 'gemini',
            'tokens_used': {'total_tokens': 0},
            'query': query,
            'error': '; '.join(errors) if errors else 'unknown'
        }

    def generate_with_sources(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        verbose: bool = False,
    ) -> Dict:
        """Generate answer and attach source list."""
        context_parts: List[str] = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"Context {i} (Source: {chunk.get('source', 'unknown')}):\n{chunk.get('text','')}\n"
            )
        context = "\n".join(context_parts)
        result = self.generate_answer(query, context, verbose=verbose)
        sources = list({chunk.get('source', 'unknown') for chunk in retrieved_chunks})
        result['sources'] = sources
        result['chunks_used'] = len(retrieved_chunks)
        return result


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def estimate_cost(tokens_used: Dict, model: str = DEFAULT_MODEL) -> float:
    """
    Estimate API cost based on token usage.
    
    NOTE: This helper is retained for older evaluation utilities. Live generation
    uses Gemini and currently returns provider metadata rather than exact token
    accounting.
    
    PARAMETERS:
        tokens_used (Dict): Token counts from generate_answer
        model (str): Model name
    
    RETURNS:
        float: Estimated cost in USD
    """
    pricing = {
        'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
    }
    
    if model not in pricing:
        return 0.0
    
    input_cost = (tokens_used['prompt_tokens'] / 1000) * pricing[model]['input']
    output_cost = (tokens_used['completion_tokens'] / 1000) * pricing[model]['output']
    
    return input_cost + output_cost


# ==============================================================================
# SELF-TEST / DEMO BLOCK
# ==============================================================================

if __name__ == "__main__":
    """
    Test the LLM interface with mock context.
    
    NOTE: Requires GEMINI_API_KEY environment variable!
    
    Run:
        export GEMINI_API_KEY="your-key-here"
        python src/generation/llm_interface.py
    """
    
    print("="*80)
    print("TESTING: llm_interface.py (Google Gemini)")
    print("="*80)
    
    # Check for API key
    if not os.getenv('GEMINI_API_KEY'):
        print("\n⚠ GEMINI_API_KEY not set!")
        print("Set it to test the LLM interface:")
        print("  export GEMINI_API_KEY='your-key-here'")
        print("\nSkipping test...")
        exit(0)
    
    try:
        # Initialize LLM
        print("\n" + "-"*80)
        print("TEST 1: Initialize LLM Interface")
        print("-"*80)
        
        llm = LLMInterface(model="gemini-pro")
        
        # Test generation with mock context
        print("\n" + "-"*80)
        print("TEST 2: Generate Answer")
        print("-"*80)
        
        mock_context = """
        Context 1: The Transformer is a neural network architecture that relies entirely on self-attention mechanisms, dispensing with recurrence and convolutions entirely.
        
        Context 2: The architecture consists of an encoder and decoder, each with multiple layers. Each layer contains multi-head attention and feed-forward networks.
        """
        
        query = "What is the Transformer architecture?"
        
        result = llm.generate_answer(query, mock_context, verbose=True)
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
