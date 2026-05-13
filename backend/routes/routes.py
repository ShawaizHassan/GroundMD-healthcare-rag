from fastapi import APIRouter
from pydantic import BaseModel
from backend.service import get_answerer

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

@router.post("/api/query")
async def query_endpoint(request: QueryRequest):
    answer = get_answerer(request.query)
    return {"answer": answer, "status": "success"}