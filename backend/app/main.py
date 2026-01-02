import logging
from fastapi import FastAPI
from backend.app.core.logging_config import setup_logging
from backend.app.core.paths import LOGS_DIR
from backend.app.api import chat, health

# Setup logging first
setup_logging(LOGS_DIR / "app.log")

app = FastAPI(title="ML_RAG API")

# Include routers
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])

@app.get("/")
def read_root():
    return {"message": "Welcome to ML_RAG API. Docs at /docs"}
