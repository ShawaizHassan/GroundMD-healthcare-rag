from pydantic import BaseModel, Field
from typing import List


class UserInput(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(default=3, ge=1, le=10)


class OutputResponse(BaseModel):
    answer: str
    citations: List[str]
    status: str