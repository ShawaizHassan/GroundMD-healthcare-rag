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