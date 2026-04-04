from fastapi import FastAPI
from backend.routes import router

app = FastAPI(title="Healthcare RAG Backend")

app.include_router(router)