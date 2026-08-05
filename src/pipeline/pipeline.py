import sys
from dotenv import load_dotenv
from groq import Groq
load_dotenv()
from src.ingestion.ingestion import ingest, ingest_folder
from src.embeddings.vector_store import get_embedding_model
from langchain_community.vectorstores import FAISS
from pathlib import Path


DEFAULT_MODEL = "llama-3.3-70b-versatile" 
DEFAULT_TOP_K = 4

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SYSTEM_PROMPT = """You are a research assistant. Answer the user's question
using only the provided context from the uploaded document(s).
If the answer isn't in the context, say so clearly — do not make things up.
Cite page numbers when you reference specific claims."""


class RAGPipeline:
    
    def __init__(self, model: str = DEFAULT_MODEL, top_k: int = DEFAULT_TOP_K):
        self.model = model
        self.top_k = top_k
        self.embeddings = get_embedding_model()
        self.vector_store: FAISS | None = None
        self.client = Groq() 
        self.documents_indexed: list[str] = []
        
    
    def add_chunks(self, chunks:list) -> None:
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)


    def add_document(self, pdf_path: str) -> int:
        chunks = ingest(pdf_path)

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)

        self.documents_indexed.append(pdf_path)
        return len(chunks)
    
    def load_sample_documents(self, sample_dir:str="src/data/sample") -> int:
        chunks = ingest_folder(sample_dir)
        self.add_chunks(chunks)
        self.documents_indexed.extend(
            p for p in [sample_dir] if p not in self.documents_indexed
        )
        return len(chunks)


    def retrieve(self, question: str) -> list:
        if self.vector_store is None:
            raise ValueError("No documents indexed yet — call add_document() first.")
        return self.vector_store.similarity_search(question, k=self.top_k)


    def generate(self, question: str, chunks: list) -> str:
        context = "\n\n---\n\n".join(
            f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in chunks
        )

        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.choices[0].message.content


    def ask(self, question: str) -> tuple[str, list]:
        chunks = self.retrieve(question)
        answer = self.generate(question, chunks)
        return answer, chunks


if __name__ == "__main__":
    
    rag = RAGPipeline()
    n_chunks = rag.add_document('src/data/sample/paper.pdf')
    print(f"Indexed {n_chunks} chunks")

    question = "What is this paper about?"
    print(f"\nQ: {question}")
    answer, sources = rag.ask(question)

    print(f"\nA: {answer}")
    print("\nSources:")
    for doc in sources:
        print(f"  - Page {doc.metadata.get('page', '?')}")