# RAG Research Assistant
Upload a research paper — or try a bundled sample — and ask it questions. Get grounded, page-cited answers instead of hallucinated summaries.

# Overview
Most "chat with your PDF" projects are a single notebook cell. This one is built as an actual retrieval-augmented generation system — ingestion, embedding, retrieval, and generation are separated into their own modules behind one orchestrator class (RAGPipeline), so each piece can be tested, swapped, or extended independently.

# Highlights:
📤 Upload any PDF, or click "Try with a sample paper" for an instant zero-setup demo
🔍 Semantic retrieval over chunked content using local sentence embeddings
🤖 Answers are generated only from retrieved context, with page-number citations — not the model's general knowledge
📚 Multi-document sessions — index several papers and ask questions across all of them
💸 Runs entirely on free tiers — local embeddings (Hugging Face) + free-tier LLM (Groq/Llama 3.3), no paid API key required
☁️ Deployable for free on Streamlit Community Cloud


# Architecture
┌───────────────────┐
   PDF (upload or   →    │     Ingestion      │  PyPDFLoader → RecursiveCharacterTextSplitter
   sample folder)        │                    │  (paragraph/sentence-aware chunking)
                         └─────────┬─────────┘
                                   ▼
                         ┌───────────────────┐
                         │     Embedding      │  BAAI/bge-small-en-v1.5
                         │   (local, free)    │  sentence-transformers, runs on CPU
                         └─────────┬─────────┘
                                   ▼
                         ┌───────────────────┐
                         │   Vector Store     │  FAISS (in-memory, per session)
                         └─────────┬─────────┘
                                   ▲
                                   │ top-k similarity search
   User question   →      ┌───────────────────┐
                         │     Retrieval      │
                         └─────────┬─────────┘
                                   ▼
                         ┌───────────────────┐
                         │    Generation      │  Llama 3.3 70B via Groq (free tier)
                         │                    │  grounded + page-cited answer
                         └───────────────────┘

Design notes:

Embeddings run locally (no API key, no cost) — only the final generation step calls out to Groq
Vector store is in-memory per session, rebuilt/appended to as documents are added — no persistent database to manage for a demo-scale project
RAGPipeline is the single entry point both the Streamlit UI and any future API layer use — no duplicated retrieval/generation logic


# Project Structure 
Research Assistant/
├── app.py                       # Streamlit UI — upload/sample flow + Q&A
├── src/
│   ├── pipeline/
│   │   └── pipeline.py          # RAGPipeline — orchestrates the full flow
│   ├── ingestion/
│   │   └── ingestion.py         # load + chunk PDFs (single file or folder)
│   ├── embeddings/
│   │   └── vector_store.py      # embedding model loader
│   └── data/
│       └── sample/              # bundled demo paper(s)
├── requirements.txt
├── .env.example
└── README.md


# Setup 
git clone https://github.com/Ayush74510/research-assistant.git
cd "Research Assistant"

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

# API keys (both free, no credit card)
KEY                         Get it from
GROQ_API_KEY                console.groq.com/keys
HF_TOKEN                    	huggingface.co/settings/tokens