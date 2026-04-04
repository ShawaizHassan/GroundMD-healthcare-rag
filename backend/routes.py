from fastapi import APIRouter
from backend.schemas import QueryRequest, QueryResponse
from backend.service import process_query

router = APIRouter(prefix="/api", tags=["Backend"])

@router.post("/query", response_model=QueryResponse)
def query_handler(request: QueryRequest):
    result = process_query(request.query)
    return QueryResponse(**result)