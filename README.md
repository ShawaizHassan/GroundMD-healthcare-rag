# 🏥 GroundMD – Healthcare RAG System

A **Retrieval-Augmented Generation (RAG)** system for clinical queries using medical guidelines.  
Combines **semantic search + reranking + LLM reasoning** to produce grounded, citeable answers.

---

## 🚀 Overview

GroundMD is an end-to-end AI system that:

- Retrieves relevant medical guideline chunks (ChromaDB)
- Reranks them using a cross-encoder
- Builds structured prompts
- Generates answers via an LLM (Ollama)
- Returns answers with **citations**

---

## 🧠 Architecture

```
User Query
   ↓
FastAPI Backend (/api/query)
   ↓
Chroma Retriever (Top-K = 10)
   ↓
Cross-Encoder Reranker (Top-K = 3)
   ↓
Prompt Builder
   ↓
Ollama LLM (phi3)
   ↓
Answer + Citations
   ↓
Streamlit Frontend
```

---

## ⚙️ Tech Stack

### Backend
- FastAPI
- ChromaDB
- Sentence Transformers
- Cross-Encoder (MS MARCO reranker)
- PyMuPDF

### LLM
- Ollama
- Model: `phi3`

### Frontend
- Streamlit

### Infra
- Docker
- Docker Compose

---

## 📁 Project Structure

```
.
├── backend/
│   ├── main.py
│   ├── routes/
│   ├── service.py
│   ├── models/
│   └── requirements.txt
│
├── frontend/
│   ├── app.py
│   ├── components/
│   └── requirements.txt
│
├── retriever/
├── ingestion/
├── generator/
├── vectorstores/
│
├── data/
│   └── chroma_store/
│
├── docker-compose.yml
```

---

## 🔧 Setup (Docker – Recommended)

### 1. Clone repo
```bash
git clone https://github.com/ShawaizHassan/GroundMD-healthcare-rag.git
cd GroundMD-healthcare-rag
```

### 2. Build & Run
```bash
docker compose up --build
```

---

## 🌐 Access

- Backend API:  
  ```
  http://localhost:8000/docs
  ```

- Frontend UI:  
  ```
  http://localhost:8501
  ```

---

## 🔍 Example Query

```
What are the recommended HbA1c targets for type 2 diabetes?
```

### Output
- Structured answer
- Source citations (file + page + relevance)

---

## 📊 Pipeline Details

### Retrieval
- Embedding model: `multi-qa-MiniLM-L6-cos-v1`
- Vector store: persistent ChromaDB

### Reranking
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Improves precision over raw similarity search

### Prompting
- Context-aware prompt built from top-k reranked chunks
- Enforces grounded answers + citations

### Generation
- LLM via Ollama
- Runs locally (no API dependency)

---

## 📦 Data

- Source: Medical guideline PDFs
- Chunking:
  - Size: `1000`
  - Overlap: `200`
- Metadata:
  - source file
  - page
  - disease name
  - chunk index

---

## 🧪 API

### POST `/api/query`

#### Request
```json
{
  "query": "What is HbA1c target?",
  "top_k": 3
}
```

#### Response
```json
{
  "answer": "...",
  "citations": [
    "[Source: file.pdf, page 108] Disease: diabetes, Relevance: 7.82"
  ],
  "status": "success"
}
```

---

## ⚠️ Known Limitations

- Requires pre-built ChromaDB store
- Initial model download (HF) can be slow
- CPU-only inference (default setup)
- LLM quality depends on prompt + context quality

---

## 🧠 Future Improvements

- RAG evaluation pipeline (accuracy, faithfulness)
- Better chunking strategies (semantic splitting)
- Hybrid search (BM25 + dense)
- Streaming responses
- GPU inference support
- Production-ready API layer

---

## 👨‍💻 Author

Ahmad Hassan  
AI/NLP | RAG Systems | LLM Evaluation

---

## ⚡ Project Reality

- End-to-end RAG system (not a notebook demo)
- Covers:
  - Data ingestion
  - Vector database
  - Reranking
  - Prompt engineering
  - API development
  - Frontend
  - Docker deployment

This is a **real system-level project**, not a toy example.
