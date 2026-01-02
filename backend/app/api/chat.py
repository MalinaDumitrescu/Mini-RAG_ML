from fastapi import APIRouter, HTTPException
from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.rag.pipeline import RAGPipeline
from backend.app.core.paths import INDEX_DIR
import logging

router = APIRouter()
logger = logging.getLogger("api.chat")

# Initialize pipeline once (lazy loading inside pipeline handles heavy models)
pipeline = RAGPipeline(INDEX_DIR)

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        # Run the full RAG pipeline
        result = pipeline.answer(request.message)
        
        # Extract sources text
        sources = [item["text"] for item in result.get("retrieved", [])]

        return ChatResponse(
            answer=result["answer"],
            sources=sources,
            judge_result=result.get("judge")
        )
    except Exception as e:
        logger.exception("Error in chat endpoint")
        raise HTTPException(status_code=500, detail=str(e))
