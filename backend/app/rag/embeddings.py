# backend/app/rag/embeddings.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from backend.app.core.paths import FINETUNED_DIR

logger = logging.getLogger("rag.embeddings")


@dataclass
class EmbeddingModelConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[str] = None  # "cuda" or "cpu"
    use_finetuned: bool = True


class EmbeddingModel:
    def __init__(self, cfg: EmbeddingModelConfig):
        if cfg.device is None:
            cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = cfg

        model_path = cfg.model_name

        if cfg.use_finetuned and FINETUNED_DIR.exists() and any(FINETUNED_DIR.iterdir()):
            logger.info("Found fine-tuned embedding model candidate at %s.", FINETUNED_DIR)
            try:
                # Verify it loads (on CPU for safety), then use it.
                _ = SentenceTransformer(str(FINETUNED_DIR), device="cpu")
                model_path = str(FINETUNED_DIR)
                logger.info("Verified fine-tuned embedding model. Using: %s", model_path)
            except Exception as e:
                logger.error("Failed to load fine-tuned model: %s. Reverting to base model.", e)
                model_path = cfg.model_name
        else:
            logger.info("Using base embedding model: %s", model_path)

        logger.info("Loading embedding model on %s", cfg.device)
        self.model = SentenceTransformer(model_path, device=cfg.device)

    def encode(self, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        logger.info("Encoding %d texts (batch_size=%d, normalize=%s)", len(texts), batch_size, normalize)
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        emb = emb.astype(np.float32, copy=False)
        logger.info("Embeddings shape: %s dtype=%s", emb.shape, emb.dtype)
        return emb
