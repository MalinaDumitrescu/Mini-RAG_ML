# backend/app/rag/corpus.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Document:
    doc_id: str
    text: str
    metadata: Dict[str, Any]


