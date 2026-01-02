# backend/app/rag/retrieval.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from backend.app.rag.embeddings import EmbeddingModel, EmbeddingModelConfig
from backend.app.rag.vector_store_faiss import FaissStore, load_faiss

logger = logging.getLogger("rag.retrieval")


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class Retriever:
    def __init__(self, index_dir: Path, emb_cfg: EmbeddingModelConfig | None = None):
        self.index_dir = index_dir
        self.store: FaissStore = load_faiss(index_dir / "faiss.index", index_dir / "docstore.json")

        id_map_path = index_dir / "id_map.json"
        if not id_map_path.exists():
            raise FileNotFoundError(f"Missing id_map.json at {id_map_path}")

        with id_map_path.open("r", encoding="utf-8") as f:
            self.id_map: Dict[str, str] = json.load(f)  # faiss_pos(str) -> chunk_id

        self.emb = EmbeddingModel(emb_cfg or EmbeddingModelConfig())

    def retrieve(self, question: str, top_k: int = 5) -> List[RetrievedChunk]:
        logger.info("Retrieving top_k=%d for question: %s", top_k, question)

        q_emb = self.emb.encode([question], batch_size=1, normalize=True)  # (1, dim)
        q_emb = q_emb.astype(np.float32, copy=False)

        scores, idxs = self.store.index.search(q_emb, top_k)

        results: List[RetrievedChunk] = []
        for faiss_pos, score in zip(idxs[0], scores[0]):
            if faiss_pos == -1:
                continue
            faiss_key = str(int(faiss_pos))
            chunk_id = self.id_map.get(faiss_key)
            if not chunk_id:
                continue

            entry = self.store.docstore.get(chunk_id)
            if not entry:
                continue

            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    score=float(score),
                    text=entry["text"],
                    metadata=entry.get("metadata", {}),
                )
            )

        logger.info("Retrieved %d chunks.", len(results))
        return results
