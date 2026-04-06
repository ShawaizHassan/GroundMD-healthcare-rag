from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from healthcare_rag import HealthcareRAG

app = FastAPI()
rag_system = HealthcareRAG()

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

@app.post("/api/query")
async def query_endpoint(request: QueryRequest):
    result = rag_system.process_query(request.query, request.top_k)
    return result

@app.get("/")
def root():
    return {"message": "Healthcare Privacy AI API", "status": "healthy"}