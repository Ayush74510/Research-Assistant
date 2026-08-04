import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(path: str) -> list:

    loader = PyPDFLoader(path)
    documents = loader.load()
    return documents


def chunk_documents(
    documents: list,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def ingest(path: str) -> list:
    print(f"Loading {path}...")
    documents = load_pdf(path)
    print(f"  {len(documents)} pages loaded")

    chunks = chunk_documents(documents)
    print(f"  {len(chunks)} chunks created")

    return chunks


if __name__ == "__main__":
    docs = load_pdf('src/data/paper.pdf')
    
    chunks = chunk_documents(docs)
    
    print(chunks)
