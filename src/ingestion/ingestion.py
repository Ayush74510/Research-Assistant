import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(path: str) -> list:

    loader = PyPDFLoader(path)
    documents = loader.load()
    return documents

def get_pdf_paths(path: str) -> list[str]:
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.pdf")))
    return [path]


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


def ingest_folder(folder_path: str) -> list:
    pdf_paths = get_pdf_paths(folder_path)
    if not pdf_paths:
        raise ValueError(f"No PDFs found in {folder_path}")
 
    all_chunks = []
    for pdf_path in pdf_paths:
        all_chunks.extend(ingest(pdf_path))
 
    return all_chunks


# if __name__ == "__main__":
#     chunk = ingest_folder('src/data')
    
#     print(chunk)