from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from retrieval_rag import RetrievalRAG

app = FastAPI(title="Healthcare RAG API with Privacy Protection")

# Initialize RAG system
rag_system = RetrievalRAG()

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

class Citation(BaseModel):
    text: str
    source: str
    topic: str
    relevance_score: float

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    phi_detected: List[tuple]
    retrieved_docs_count: int
    status: str

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        result = rag_system.process_query(request.query, request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {
        "message": "Healthcare RAG API with Privacy Protection",
        "status": "healthy",
        "features": [
            "PHI detection and anonymization",
            "Vector search for medical documents",
            "RAG with Ollama (gemma3:4b)",
            "Citations with relevance scores"
        ]
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "gemma3:4b", "documents_loaded": len(rag_system.documents)}