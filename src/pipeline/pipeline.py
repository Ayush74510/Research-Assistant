"""
pipeline.py

The orchestrator: ties ingestion -> embedding/vector store -> retrieval
-> generation into a single reusable RAGPipeline class.

This is what frontend/app.py and api/main.py should both import,
instead of duplicating ingestion/retrieval/generation logic inline.

Usage:
    from pipeline import RAGPipeline

    rag = RAGPipeline()
    rag.add_document("data/sample/paper.pdf")
    answer, sources = rag.ask("What is this paper about?")
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "ingestion"))
sys.path.append(os.path.join(os.path.dirname(__file__), "embeddings"))

from src.ingestion.ingestion import ingest
from src.embeddings.vector_store import get_embedding_model
from langchain_community.vectorstores import FAISS

import anthropic


DEFAULT_MODEL = "claude-sonnet-4-6"  # verify current model string before running
DEFAULT_TOP_K = 4

SYSTEM_PROMPT = """You are a research assistant. Answer the user's question
using only the provided context from the uploaded document(s).
If the answer isn't in the context, say so clearly — do not make things up.
Cite page numbers when you reference specific claims."""


class RAGPipeline:
    """
    A self-contained RAG pipeline: holds an in-memory vector store and
    exposes add_document() and ask() as the only two methods callers need.
    """

    def __init__(self, model: str = DEFAULT_MODEL, top_k: int = DEFAULT_TOP_K):
        self.model = model
        self.top_k = top_k
        self.embeddings = get_embedding_model()
        self.vector_store: FAISS | None = None
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.documents_indexed: list[str] = []

    def add_document(self, pdf_path: str) -> int:
        """
        Ingest a PDF and add it to the vector store.
        Creates a new store on first call, appends on subsequent calls.
        Returns the number of chunks added.
        """
        chunks = ingest(pdf_path)

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)

        self.documents_indexed.append(pdf_path)
        return len(chunks)

    def retrieve(self, question: str) -> list:
        """Return the top-k chunks most relevant to the question."""
        if self.vector_store is None:
            raise ValueError("No documents indexed yet — call add_document() first.")
        return self.vector_store.similarity_search(question, k=self.top_k)

    def generate(self, question: str, chunks: list) -> str:
        """Call Claude with the retrieved chunks as context."""
        context = "\n\n---\n\n".join(
            f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in chunks
        )

        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    def ask(self, question: str) -> tuple[str, list]:
        """
        Full pipeline: retrieve relevant chunks, generate an answer.
        Returns (answer, source_chunks) so callers can display citations.
        """
        chunks = self.retrieve(question)
        answer = self.generate(question, chunks)
        return answer, chunks


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pipeline.py <path-to-pdf>")
        sys.exit(1)

    rag = RAGPipeline()
    n_chunks = rag.add_document(sys.argv[1])
    print(f"Indexed {n_chunks} chunks from {sys.argv[1]}")

    question = "What is this paper about?"
    print(f"\nQ: {question}")
    answer, sources = rag.ask(question)

    print(f"\nA: {answer}")
    print("\nSources:")
    for doc in sources:
        print(f"  - Page {doc.metadata.get('page', '?')}")