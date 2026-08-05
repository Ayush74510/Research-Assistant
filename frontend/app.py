"""
app.py

Streamlit frontend for the RAG Research Assistant.

Flow:
  1. User uploads a PDF
  2. It gets ingested + chunked + embedded + indexed on the spot
  3. User asks a question
  4. Top-k relevant chunks are retrieved and passed to Claude
  5. Answer is shown, with source page numbers cited

Run with:
    streamlit run frontend/app.py
"""

import os
import sys
import tempfile
import streamlit as st

# Make src/ modules importable
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "ingestion"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "embeddings"))

from ingestion import ingest
from vector_store import get_embedding_model, search
from langchain_community.vectorstores import FAISS

import anthropic


st.set_page_config(page_title="RAG Research Assistant", page_icon="📄")
st.title("📄 RAG Research Assistant")
st.caption("Upload a research paper, then ask questions about it.")

# --- Session state: holds the in-memory vector store for this session ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "filename" not in st.session_state:
    st.session_state.filename = None


# --- Step 1: Upload + ingest ---
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None and uploaded_file.name != st.session_state.filename:
    with st.spinner(f"Reading and indexing {uploaded_file.name}..."):
        # Save the upload to a temp file so PyPDFLoader can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Ingest: load + chunk
        chunks = ingest(tmp_path)

        # Embed + build a fresh in-memory index for this document
        embeddings = get_embedding_model()
        vector_store = FAISS.from_documents(chunks, embeddings)

        st.session_state.vector_store = vector_store
        st.session_state.filename = uploaded_file.name

        os.unlink(tmp_path)  # clean up temp file

    st.success(f"Indexed {uploaded_file.name} ({len(chunks)} chunks)")


# --- Step 2: Ask a question ---
if st.session_state.vector_store is not None:
    question = st.text_input("Ask a question about this paper")

    if question:
        with st.spinner("Retrieving relevant sections..."):
            results = search(st.session_state.vector_store, question, k=4)

        context = "\n\n---\n\n".join(
            f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in results
        )

        prompt = f"""Answer the question using only the context below.
If the answer isn't in the context, say so — don't make things up.
Cite page numbers where relevant.

Context:
{context}

Question: {question}"""

        with st.spinner("Generating answer..."):
            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.content[0].text

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("Sources used"):
            for doc in results:
                st.markdown(f"**Page {doc.metadata.get('page', '?')}**")
                st.text(doc.page_content[:300] + "...")
else:
    st.info("Upload a PDF above to get started.")