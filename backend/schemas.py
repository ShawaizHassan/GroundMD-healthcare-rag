from pydantic import BaseModel
from typing import List

class UserInput(BaseModel):
    query: str
    top_k: int

class OutputResponse(BaseModel):
    answer: str
    sources: List[str]
    status: str