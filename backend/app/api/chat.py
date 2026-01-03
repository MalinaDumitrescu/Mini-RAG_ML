from fastapi import APIRouter, HTTPException
from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.rag.pipeline import RAGPipeline
from backend.app.core.paths import INDEX_DIR
import logging

router = APIRouter()
logger = logging.getLogger("api.chat")

pipeline = RAGPipeline(INDEX_DIR)

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        result = pipeline.answer(request.message)

        # richer sources
        sources = [
            {
                "chunk_id": item["chunk_id"],
                "faiss_score": item["faiss_score"],
                "score": item["score"],
                "metadata": item.get("metadata", {}),
                "text_preview": (item["text"][:300] + "...") if len(item["text"]) > 300 else item["text"],
            }
            for item in result.get("retrieved", [])
        ]

        return ChatResponse(
            answer=result["answer"],
            sources=sources,
            judge_result=result.get("judge")
        )
    except Exception as e:
        logger.exception("Error in chat endpoint")
        raise HTTPException(status_code=500, detail=str(e))
