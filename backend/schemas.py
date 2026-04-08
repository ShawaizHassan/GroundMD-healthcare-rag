from pydantic import BaseModel
from typing import List

class UserInput(BaseModel):
    query: str
    top_k: int

class OutputResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[str]
    status: str