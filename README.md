# 📄 DocMind — Local AI Document Assistant

> A fully local, privacy-first document Q&A system with hybrid retrieval, re‑ranking, and an agentic mode — built with free, open‑source tools.

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![All Local](https://img.shields.io/badge/Runs%20Locally-Yes-brightgreen)]()
[![MPS](https://img.shields.io/badge/Apple%20Silicon-M1%20optimised-silver)]()

---

## 🚀 What DocMind Does

- **Chat with any PDF** — ask questions and get answers with cited sources.
- **Hybrid search** — combines dense embeddings (Chroma) and sparse BM25 for maximum retrieval accuracy.
- **Cross‑encoder re‑ranking** — boosts the most relevant chunks to the top.
- **Agent mode** — search the web, calculate, or query Wikipedia — all from a local LLM.
- **Human feedback loop** — thumbs‑up/down ratings stored in SQLite for future fine‑tuning.
- **Modular architecture** — clean separation of loading, retrieval, LLM, and UI.

---

## 🧰 Tech Stack (100% free & local)

| Component          | Technology |
|-------------------|------------|
| LLM                | Mistral 7B via [Ollama](https://ollama.com) (4‑bit quantised) |
| Embeddings         | `all-MiniLM-L6-v2` via HuggingFace, runs on MPS (Apple Silicon) |
| Vector store       | [ChromaDB](https://www.trychroma.com/) (persistent, embedded) |
| Sparse retrieval   | BM25 via `rank_bm25` |
| Re‑ranker          | Cross‑encoder `ms-marco-MiniLM-L-6-v2` |
| Agent tools        | DuckDuckGo Search, Wikipedia, Calculator |
| UI                 | [Streamlit](https://streamlit.io) |
| Feedback DB        | SQLite |
| Experiment tracking| MLflow (optional) |
| CI/CD              | GitHub Actions (ready to add) |

---

## 📁 Project Structure
doc-assistant/
├── app.py # Streamlit UI (thin presentation layer)
├── document_loader.py # PDF loading, chunking, vector store creation
├── llm.py # Ollama LLM factory
├── retrieval.py # Hybrid retriever + re‑ranker
├── agent.py # ReAct agent with tools (web, calc, wiki)
├── feedback_db.py # SQLite feedback storage
├── finetune_retriever.py # Script to fine‑tune embeddings on feedback
├── requirements.txt # Python dependencies
└── README.md



---

## 🖥️ Getting Started

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- At least 8 GB RAM (16 GB recommended)

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/docmind.git
cd docmind

conda create -n docmind python=3.11 -y
conda activate docmind
conda install pytorch::pytorch -c pytorch -y   # Apple Silicon native
pip install -r requirements.txt

ollama pull mistral:7b

streamlit run app.py

5. Open your browser

Navigate to http://localhost:8501 and upload a PDF (or use the included paper.pdf).


📜 License

MIT — feel free to use, modify, and share.

Built by Ayaz Ahmed as a portfolio project showcasing production‑grade local AI engineering.

