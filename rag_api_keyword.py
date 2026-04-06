from fastapi import FastAPI
from pydantic import BaseModel
from rag_keyword import KeywordRAG

app = FastAPI()
rag = KeywordRAG()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/api/query")
async def query_endpoint(request: QueryRequest):
    result = rag.process_query(request.query, request.top_k)
    return result

@app.get("/")
def root():
    return {"message": "Healthcare RAG API (Keyword-based)", "status": "healthy"}