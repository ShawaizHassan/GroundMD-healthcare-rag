from fastapi import FastAPI
from backend.routes.routes import router
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost",
    "http://localhost:8000",
    # Add any other origins (like a frontend dev server) here
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # for dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)