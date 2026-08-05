import os
from langchain_community.vectorstores import FAISS 
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

EMBEDDING_MODEL_NAME = 'BAAI/bge-small-en-v1.5'

INDEX_DIR = 'data/vector_store'

def get_embedding_model() -> HuggingFaceBgeEmbeddings:
    
    return HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL_NAME, encode_kwargs={'normalize_embeddings':True})


