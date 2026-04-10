from fastapi import APIRouter, HTTPException
from backend.models.schemas import UserInput, OutputResponse
from backend.service import Services
from llm.ollama_client import OllamaLLM
import os

router = APIRouter(prefix="/backend", tags=["Healthcare AI Backend"])


@router.get("/health")
async def health():
    return {"status": "success"}


llm = OllamaLLM(
    model=os.getenv("OLLAMA_MODEL", "phi3"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
)

service = Services(llm=llm)


@router.post("/query", response_model=OutputResponse)
def query_handler(request: UserInput):
    result = service.process_query(query=request.query, top_k=request.top_k)

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["answer"])

    return OutputResponse(**result)