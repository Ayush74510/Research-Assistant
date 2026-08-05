import os
import sys
import tempfile
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = PROJECT_ROOT / "src" / "data" / "sample"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.pipeline import RAGPipeline


st.set_page_config(page_title="RAG Research Assistant", page_icon="📄")
st.title("📄 RAG Research Assistant")
st.caption("Upload a research paper, or try a bundled sample, then ask questions about it.")

# --- Session state: one RAGPipeline instance per session ---
if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline()
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


# --- Step 1a: Try with a bundled sample paper ---
col1, col2 = st.columns([1, 2])

with col1:
    sample_available = os.path.isdir(SAMPLE_DIR) and any(
        f.endswith(".pdf") for f in os.listdir(SAMPLE_DIR)
    )
    if st.button("Try with a sample paper", disabled=not sample_available):
        with st.spinner("Indexing sample paper(s)..."):
            n_chunks = st.session_state.rag.load_sample_documents(SAMPLE_DIR)
            for f in os.listdir(SAMPLE_DIR):
                if f.endswith(".pdf") and f not in st.session_state.indexed_files:
                    st.session_state.indexed_files.append(f)
        st.success(f"Indexed sample paper(s) ({n_chunks} chunks)")

    if not sample_available:
        st.caption("No sample PDFs found in data/sample/")

with col2:
    uploaded_file = st.file_uploader("Or upload your own PDF", type=["pdf"])

    if uploaded_file is not None and uploaded_file.name not in st.session_state.indexed_files:
        with st.spinner(f"Reading and indexing {uploaded_file.name}..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            n_chunks = st.session_state.rag.add_document(tmp_path)
            st.session_state.indexed_files.append(uploaded_file.name)

            os.unlink(tmp_path)  # clean up temp file

        st.success(f"Indexed {uploaded_file.name} ({n_chunks} chunks)")


if st.session_state.indexed_files:
    st.caption(f"Indexed so far: {', '.join(st.session_state.indexed_files)}")


# --- Step 2: Ask a question ---
if st.session_state.indexed_files:
    question = st.text_input("Ask a question about the indexed document(s)")

    if question:
        with st.spinner("Retrieving and generating answer..."):
            answer, sources = st.session_state.rag.ask(question)

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("Sources used"):
            for doc in sources:
                st.markdown(f"**Page {doc.metadata.get('page', '?')}**")
                st.text(doc.page_content[:300] + "...")
else:
    st.info("Click 'Try with a sample paper' or upload a PDF above to get started.")