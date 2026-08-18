<div align="center">

<!-- HEADER BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,14,20,24&height=200&section=header&text=RAG%20Research%20Assistant&fontSize=42&fontColor=ffffff&fontAlignY=38&desc=Chat%20with%20your%20research%20papers%20%E2%80%94%20grounded%2C%20cited%2C%20free&descSize=16&descAlignY=60&animation=fadeIn" width="100%"/>

<!-- BADGES -->
<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white"/>
  <img src="https://img.shields.io/badge/FAISS-Vector%20Store-00B4D8?style=for-the-badge&logo=meta&logoColor=white"/>
  <img src="https://img.shields.io/badge/Groq-Llama%203.3%2070B-F55036?style=for-the-badge&logo=groq&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
</p>

<p>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Cost-100%25%20Free%20Tier-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Deploy-Streamlit%20Cloud-FF4B4B?style=flat-square&logo=streamlit"/>
  <img src="https://img.shields.io/badge/Embeddings-Local%20%7C%20No%20API-blue?style=flat-square"/>
</p>

<br/>

> **Upload a research paper. Ask it anything. Get grounded, page-cited answers — not hallucinated summaries.**

<br/>

</div>

---

## ✨ What Makes This Different

> Most *"chat with PDF"* demos are a single notebook cell. This is a **production-style RAG system** where ingestion, embedding, retrieval, and generation are cleanly separated into their own modules — testable, swappable, and extensible.

<br/>

<table>
<tr>
<td width="50%">

### 🔑 Key Highlights

| Feature | Detail |
|---|---|
| 📤 **Upload or Sample** | Instant zero-setup demo with bundled paper |
| 🔍 **Semantic Retrieval** | Local sentence embeddings via HuggingFace |
| 🤖 **Grounded Answers** | LLM only uses retrieved context — no hallucinations |
| 📄 **Page Citations** | Every answer comes with source page numbers |
| 📚 **Multi-Document** | Index several papers, query across all of them |
| 💸 **100% Free Tier** | Local embeddings + Groq free API — no credit card |
| ☁️ **Cloud Deployable** | One-click deploy on Streamlit Community Cloud |

</td>
<td width="50%">

### 🧠 Tech Stack

| Layer | Technology |
|---|---|
| **UI** | Streamlit |
| **Ingestion** | LangChain `PyPDFLoader` |
| **Chunking** | `RecursiveCharacterTextSplitter` |
| **Embeddings** | `BAAI/bge-small-en-v1.5` (CPU, local) |
| **Vector Store** | FAISS (in-memory, per-session) |
| **LLM** | Llama 3.3 70B via Groq (free tier) |
| **Orchestrator** | `RAGPipeline` — single entry point |

</td>
</tr>
</table>

---

## 🏗️ Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │                 RAGPipeline                      │
                    │    (single orchestrator — UI & API share it)     │
                    └─────────────────────────────────────────────────┘
                                          │
          ┌───────────────────────────────┼───────────────────────────────┐
          ▼                               ▼                               ▼
  ┌───────────────┐              ┌────────────────┐              ┌────────────────┐
  │   Ingestion   │              │   Embedding    │              │   Generation   │
  │               │──────────►  │                │──────────►  │                │
  │ PyPDFLoader   │              │ BAAI/bge-small │              │  Llama 3.3 70B │
  │ + Chunker     │              │ (local, CPU)   │              │  via Groq API  │
  └───────────────┘              └────────┬───────┘              └───────▲────────┘
                                          │                              │
                                          ▼                              │
                                 ┌────────────────┐              ┌───────┴────────┐
                                 │  Vector Store  │──top-k──►   │   Retrieval    │
                                 │  FAISS         │  search      │   (semantic)   │
                                 │  (in-memory)   │◄── index ─── │                │
                                 └────────────────┘              └────────────────┘
                                                                          ▲
                                                                   User Question
```

> **Design principle:** Only the final generation step hits an external API (Groq). Everything else — loading, chunking, embedding, and retrieval — runs **locally and for free**.

---

## 📁 Project Structure

```
Research Assistant/
│
├── 📄 app.py                        ← Streamlit UI (upload flow + Q&A)
│
├── 📦 src/
│   ├── pipeline/
│   │   └── pipeline.py              ← RAGPipeline — orchestrates the full flow
│   ├── ingestion/
│   │   └── ingestion.py             ← Load & chunk PDFs (single file or folder)
│   ├── embeddings/
│   │   └── vector_store.py          ← Embedding model loader + FAISS wrapper
│   └── data/
│       └── sample/                  ← 📎 Drop your demo PDFs here
│
├── 📋 requirements.txt
├── 🔐 .env.example
└── 📖 README.md
```

---

## 🚀 Quick Start

### 1 · Clone & Install

```bash
git clone https://github.com/Ayush74510/research-assistant.git
cd "Research Assistant"

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### 2 · Get Your Free API Keys

Both keys are **100% free** — no credit card required.

| Key | Where to get it | Notes |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) | Free tier: 30 req/min |
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Read token is enough |

### 3 · Configure

```bash
# Copy the example env file
cp .env.example .env
```

Then open `.env` and fill in your keys:

```env
GROQ_API_KEY=gsk_...
HF_TOKEN=hf_...
```

### 4 · Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) — upload a PDF or click **"Try with a sample paper"** and start asking questions.

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Point it to `app.py`
4. Add `GROQ_API_KEY` and `HF_TOKEN` in **Secrets**
5. Deploy — done! 🎉

---

## 🧩 Design Decisions

<details>
<summary><b>Why local embeddings?</b></summary>

`BAAI/bge-small-en-v1.5` runs entirely on CPU via `sentence-transformers`. No embedding API key, no cost, no rate limits. Quality is excellent for retrieval tasks at this scale.

</details>

<details>
<summary><b>Why FAISS in-memory instead of a persistent DB?</b></summary>

For a demo-scale project, an in-memory vector store keeps the setup to a single `pip install`. The store is rebuilt/appended to as you add documents within a session — no Pinecone account, no Chroma server.

</details>

<details>
<summary><b>Why a RAGPipeline orchestrator class?</b></summary>

Both the Streamlit UI and any future REST API layer share the same `RAGPipeline` entry point. No duplicated retrieval/generation logic. Each internal module (ingestion, embeddings, retrieval) can be tested or swapped independently.

</details>

<details>
<summary><b>Why Groq + Llama 3.3 70B?</b></summary>

Groq's free tier gives ~30 requests/minute on one of the fastest inference APIs available. Llama 3.3 70B follows grounding instructions reliably — it stays within the retrieved context and doesn't hallucinate from its training data.

</details>

---

## 🗺️ Roadmap

- [ ] 🌐 REST API layer (`FastAPI`) alongside the Streamlit UI
- [ ] 💾 Persistent vector store (Chroma / Qdrant) for cross-session memory
- [ ] 📊 Evaluation harness (RAGAS metrics — faithfulness, answer relevancy)
- [ ] 🔄 Hybrid retrieval (BM25 sparse + dense vectors)
- [ ] 🗂️ Document management UI (list, delete, re-index)

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,14,20,24&height=100&section=footer" width="100%"/>

**Built with ❤️ by [Ayush](https://github.com/Ayush74510)**

*If this project helped you, consider giving it a ⭐ on GitHub!*

</div>