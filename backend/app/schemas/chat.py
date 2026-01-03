from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None  # [{"role": "...", "content": "..."}]


class SourceItem(BaseModel):
    chunk_id: str
    faiss_score: float
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    text_preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = Field(default_factory=list)
    judge_result: Optional[Dict[str, Any]] = None
