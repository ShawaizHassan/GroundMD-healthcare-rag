from fastapi import APIRouter, HTTPException
from api.models.schemas import UserInput, OutputResponse
from api.service import Services
from llm.ollama_client import OllamaLLM

router = APIRouter(prefix="/api", tags=["Healthcare AI Backend"])


@router.get("/health")
async def health():
    return {"status": "success"}


llm = OllamaLLM(
    model="phi3",
    base_url="http://127.0.0.1:11434"
)

service = Services(llm=llm)


@router.post("/query", response_model=OutputResponse)
def query_handler(request: UserInput):
    result = service.process_query(query=request.query, top_k=request.top_k)

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["answer"])

    return OutputResponse(**result)