# 🔬 Research Intelligence Assistant
### HybridRAG — Vector Search + Knowledge Graph + Mistral AI

A production-deployed research assistant that helps AI/ML students discover relevant papers, identify research gaps, and find publication opportunities — powered by a hybrid retrieval architecture combining semantic vector search with knowledge graph traversal.

**Live Demo**: [HuggingFace Spaces](#) *(link after deployment)*

---

## 🎯 Problem Statement

4th-year AI/ML students preparing to publish research face a painful bottleneck: manually searching through hundreds of papers to understand the landscape, identify gaps, and find a viable research direction. This process takes weeks and is largely unguided.

This tool solves that. Type a topic and a question — get a synthesized, cited answer in under 30 seconds.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
ArXiv API (live paper fetch)
    │
    ▼
Text Chunking (title + abstract per paper)
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
FAISS Vector Store                  NetworkX Knowledge Graph
(SentenceTransformers embeddings)   (spaCy entity extraction)
Semantic similarity search          Graph traversal on concepts
    │                                      │
    └──────────────┬───────────────────────┘
                   ▼
         RRF Fusion Layer
         (Reciprocal Rank Fusion)
                   │
                   ▼
         Context Builder (Top-K papers)
                   │
                   ▼
         Mistral 7B via Ollama
                   │
                   ▼
         Research Answer + Gap Analysis
```

---

## 🧩 Components

| Component | Technology | Purpose |
|---|---|---|
| Paper ingestion | ArXiv API | Fetch live papers on any topic |
| Embedding | SentenceTransformers `all-MiniLM-L6-v2` | Convert text to semantic vectors |
| Vector store | FAISS | Fast semantic similarity search |
| Entity extraction | spaCy `en_core_web_sm` | Extract concepts and named entities |
| Knowledge graph | NetworkX | Model relationships between concepts |
| Fusion | Reciprocal Rank Fusion (RRF) | Merge and re-rank both retriever outputs |
| LLM | Mistral 7B via Ollama | Generate research-grade answers |
| Backend | FastAPI | REST API with topic-level caching |
| Frontend | Gradio | Clean web UI with example queries |

---

## ✨ Key Features

- **Live ArXiv integration** — no static dataset, always up to date
- **HybridRAG pipeline** — combines semantic vector search with knowledge graph traversal
- **Reciprocal Rank Fusion** — merges two ranked lists for higher precision than either alone
- **Research gap analysis** — LLM explicitly identifies open problems from fetched papers
- **Topic caching** — first query on a topic builds the index; follow-up questions are fast
- **Multimodal input ready** — architecture supports PDF, audio (Whisper), image (CLIP) extension

---

## 📁 Project Structure

```
hybrid-rag/
├── ingestion/
│   ├── arxiv_loader.py       # ArXiv API fetch + XML parsing
│   └── text_processor.py     # Chunk papers with metadata
├── retrieval/
│   ├── vector_store.py       # FAISS + SentenceTransformers
│   ├── graph_store.py        # NetworkX + spaCy knowledge graph
│   └── fusion.py             # Reciprocal Rank Fusion merger
├── orchestration/
│   └── graph_flow.py         # Full pipeline + Mistral LLM call
├── api/
│   └── main.py               # FastAPI backend with caching
├── ui/
│   └── app.py                # Gradio web interface
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed with Mistral pulled

```bash
ollama pull mistral
```

### Installation

```bash
git clone https://github.com/yourusername/hybrid-rag
cd hybrid-rag
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Run

```bash
# Terminal 1 — start the API
uvicorn api.main:app --reload

# Terminal 2 — start the UI
python ui/app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

---

## 📡 API Reference

### `POST /query`

```json
{
  "topic": "retrieval augmented generation",
  "question": "What are the open problems and where can I contribute?",
  "max_papers": 15
}
```

**Response:**
```json
{
  "answer": "...",
  "sources": [...],
  "topic": "retrieval augmented generation",
  "papers_analyzed": 15,
  "time_taken_seconds": 28.4
}
```

Interactive API docs at [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 💡 How RRF Fusion Works

Both retrievers return a ranked list of papers. RRF merges them using:

```
score(paper) = Σ 1 / (rank + 60)
```

Papers appearing high in **both** lists get a boosted combined score. Papers found by only one retriever score lower. This produces more reliable rankings than either retriever alone.

---

## 🧪 Example Queries

| Topic | Question |
|---|---|
| `retrieval augmented generation` | What are the open problems I can contribute to? |
| `large language model fine tuning` | What methods exist and what are their tradeoffs? |
| `graph neural networks` | What are recent advances and open research gaps? |
| `vision language models` | How do these work and what problems remain unsolved? |

---

## 📊 Performance

| Query type | Latency |
|---|---|
| First query on new topic | 15–30s (fetch + embed + build graph) |
| Follow-up on cached topic | 3–6s (LLM only) |

---

## 🔮 Future Work

- Add PDF upload support (LangChain + PyPDF)
- Audio ingestion via Whisper
- Image understanding via CLIP
- Citation graph visualization
- LangGraph orchestration for multi-step reasoning
- Docker deployment to HuggingFace Spaces

---

## 🛠️ Built With

`Python` `FastAPI` `Gradio` `FAISS` `SentenceTransformers` `NetworkX` `spaCy` `Mistral 7B` `Ollama` `ArXiv API`
