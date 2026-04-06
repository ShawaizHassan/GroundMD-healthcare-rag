from fastapi import FastAPI
from pydantic import BaseModel
from simple_healthcare import SimpleHealthcareAI

app = FastAPI()
ai = SimpleHealthcareAI()

class QueryRequest(BaseModel):
    query: str

@app.post("/api/query")
async def query_endpoint(request: QueryRequest):
    result = ai.process_query(request.query)
    return result

@app.get("/")
def root():
    return {"message": "Healthcare Privacy AI API", "status": "healthy"}