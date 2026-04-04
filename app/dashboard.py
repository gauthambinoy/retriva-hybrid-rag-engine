"""Minimal RAG UI
-----------------
Shows all test questions (with categories) and a single Run button.
Only the generated answer is displayed. Terminal mirrors output.
"""

from io import StringIO
from contextlib import redirect_stdout
from pathlib import Path
import sys
import time

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import RAGPipeline
from src.queries.test_queries import get_all_queries


# Page config
st.set_page_config(page_title="Retriva", page_icon="🧠", layout="wide")


@st.cache_resource(show_spinner="Loading RAG pipeline…")
def load_pipeline():
    with redirect_stdout(StringIO()):
        p = RAGPipeline(verbose_init=False)
        try:
            p.load_index()
        except Exception:
            # If no index yet, leave as-is; pipeline can still build on demand
            pass
    return p


@st.cache_data(show_spinner=False)
def load_question_labels():
    """Return (queries, labels) where labels include category for display."""
    queries = get_all_queries()
    labels = [f"{q.get('category','?').upper()} | Q{q.get('id', i+1)} — {q.get('query','')}" for i, q in enumerate(queries)]
    return queries, labels


pipeline = load_pipeline()
queries, labels = load_question_labels()

st.title("🔍 Retriva — Hybrid RAG Engine")
st.caption("Select a question below and click Run. Only the answer is shown.")

# Single radio with all questions (category included in label)
selected_idx = st.radio("Questions", options=list(range(len(labels))), format_func=lambda i: labels[i], index=0)

run = st.button("Run", type="primary", use_container_width=True, disabled=(pipeline is None))

st.markdown("---")

if run and pipeline is not None:
    question = queries[selected_idx].get('query', '')
    with st.spinner("Generating answer…"):
        with redirect_stdout(StringIO()):
            start = time.time()
            result = pipeline.query(question, top_k=5, verbose=False)
            _ = time.time() - start
    answer = (result or {}).get('answer', '').strip() or "No answer generated."
    # Also print a concise, standardized line to the Streamlit server terminal
    try:
        chunks = (result or {}).get('retrieved_chunks', [])
        sources = [Path(c.get('source') or c.get('file_path','Unknown')).name for c in chunks]
        model = (result or {}).get('model', 'N/A')
        # Trim answer for terminal display
        snippet = answer[:180] + ("…" if len(answer) > 180 else "")
        print("\n[UI RUN] Q:", question)
        print("[UI RUN] A:", snippet)
        print("[UI RUN] Sources:", ", ".join(dict.fromkeys(sources)) or "None")
        print("[UI RUN] Model:", model)
    except Exception:
        pass
    st.subheader("Answer")
    st.write(answer)

else:
    st.info("Pick a question and click Run.")

# (All legacy/duplicate UI sections removed for minimal mode)

