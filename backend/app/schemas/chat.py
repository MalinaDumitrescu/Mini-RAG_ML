from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None  # e.g. [{"role": "user", "content": "..."}]

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []
    judge_result: Optional[Dict[str, Any]] = None
