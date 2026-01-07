# backend/app/rag/vector_store_faiss.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import faiss
import numpy as np

logger = logging.getLogger("rag.vstore")


@dataclass
class FaissStore:
    index: faiss.Index
    docstore: Dict[str, Dict[str, Any]]  # chunk_id -> {"text":..., "metadata":...}


def _ensure_float32(x: np.ndarray) -> np.ndarray:
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    return x


def normalize_embeddings(x: np.ndarray) -> np.ndarray:
    """
    L2-normalize vectors in-place (FAISS expects float32).
    After normalization, inner product == cosine similarity.
    """
    x = _ensure_float32(x)
    faiss.normalize_L2(x)
    return x


def build_faiss_index(embeddings: np.ndarray, *, normalize: bool = True) -> faiss.Index:
    """
    Build an IndexFlatIP index.
    If embeddings are normalized, IndexFlatIP scores are cosine similarities.
    """
    emb = _ensure_float32(embeddings)

    if normalize:
        emb = normalize_embeddings(emb)

    if emb.ndim != 2:
        raise ValueError(f"Embeddings must be 2D (n, dim). Got shape={emb.shape}")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (cosine if normalized)
    index.add(emb)

    logger.info(
        "FAISS index built. type=%s dim=%d ntotal=%d normalized=%s",
        type(index).__name__,
        dim,
        index.ntotal,
        normalize,
    )
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
    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index at: {index_path}")
    if not docstore_path.exists():
        raise FileNotFoundError(f"Missing docstore at: {docstore_path}")

    index = faiss.read_index(str(index_path))
    with docstore_path.open("r", encoding="utf-8") as f:
        docstore = json.load(f)

    if not isinstance(index, faiss.Index):
        raise TypeError("Loaded object is not a FAISS Index.")

    logger.info(
        "Loaded FAISS index ntotal=%d from %s (type=%s, d=%s)",
        index.ntotal,
        index_path,
        type(index).__name__,
        getattr(index, "d", None),
    )

    return FaissStore(index=index, docstore=docstore)
