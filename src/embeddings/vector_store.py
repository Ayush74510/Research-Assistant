import os
from langchain_community.vectorstores import FAISS 
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = 'BAAI/bge-small-en-v1.5'

INDEX_DIR = 'data/vector_store'

def get_embedding_model() -> HuggingFaceEmbeddings:
    
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, encode_kwargs={'normalize_embeddings':True})


