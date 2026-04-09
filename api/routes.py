from fastapi import APIRouter, HTTPException
from api.schema import UserInput, OutputResponse
from api.service import Services

router = APIRouter(prefix="/api", tags=["Healthcare AI Backend"])

# 1. Initialize ONCE here (outside the functions)
# This loads models into memory during startup, not during the request
service_instance = Services(llm=None)

@router.get("/health")
async def health():
    return {"status": "success"}

# Initialize ONCE at the top of the file
service = Services(llm=None)

@router.post("/query", response_model=OutputResponse)
def query_handler(request: UserInput):
    # Use the pre-loaded instance instead of creating a new one
    result = service.process_query(query=request.query, top_k=request.top_k)
    return OutputResponse(**result)