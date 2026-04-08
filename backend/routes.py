<<<<<<< HEAD
from fastapi import APIRouter, HTTPException
from backend.schemas import UserInput, OutputResponse
from backend.service import Service

service = Service()

router = APIRouter(prefix="/api", tags=["Backend"])

@router.get("/health")
def health():
    return {"status": "success"}

@router.post("/query", response_model=OutputResponse)
def query_handler(request: UserInput):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query can't be empty")
    result = service.process_query(request.query, request.top_k)
    return OutputResponse(**result)
=======
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class QueryRequest(BaseModel):
    query: str


@router.post("/query")
def query_handler(request: QueryRequest):
    return {
        "query": request.query,
        "answer": "Mock Answer",
        "sources": [],
        "status": "success"
    }
>>>>>>> origin/feature/graciella/retrieval-pipeline
