# backend/app/rag/vector_store_faiss.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import faiss
import numpy as np

logger = logging.getLogger("rag.vstore")

@dataclass
class FaissStore:
    index: faiss.Index
    docstore: Dict[str, Dict[str, Any]]  # chunk_id -> {"text":..., "metadata":...}

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    embeddings must be float32 and preferably normalized if using cosine similarity.
    We'll use inner product (IP). If normalized, IP == cosine similarity.
    """
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("FAISS index built. dim=%d, ntotal=%d", dim, index.ntotal)
    return index

def save_faiss(store: FaissStore, index_path: Path, docstore_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    docstore_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(store.index, str(index_path))
    with docstore_path.open("w", encoding="utf-8") as f:
        json.dump(store.docstore, f, ensure_ascii=False)

    logger.info("Saved FAISS index to %s", index_path)
    logger.info("Saved docstore to %s", docstore_path)

def load_faiss(index_path: Path, docstore_path: Path) -> FaissStore:
    index = faiss.read_index(str(index_path))
    with docstore_path.open("r", encoding="utf-8") as f:
        docstore = json.load(f)

    logger.info("Loaded FAISS index ntotal=%d from %s", index.ntotal, index_path)
    return FaissStore(index=index, docstore=docstore)

def search(store: FaissStore, query_emb: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Returns list of (chunk_id, score). Query must be shape (dim,) or (1, dim)
    """
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    query_emb = query_emb.astype(np.float32, copy=False)

    scores, idxs = store.index.search(query_emb, top_k)
    results: List[Tuple[str, float]] = []

    for rank, (i, s) in enumerate(zip(idxs[0], scores[0])):
        if i == -1:
            continue
        # FAISS stores vectors in the same order we added them.
        # We'll map i -> chunk_id by storing insertion order in docstore list later.
        results.append((str(i), float(s)))  # temp id; we map later in retrieval

    return results
