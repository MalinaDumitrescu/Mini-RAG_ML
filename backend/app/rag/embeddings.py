# backend/app/rag/embeddings.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
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
        
        # Default to base model
        model_path = cfg.model_name
        
        # Check for fine-tuned model
        if cfg.use_finetuned and FINETUNED_DIR.exists() and any(FINETUNED_DIR.iterdir()):
            logger.info("Found fine-tuned model candidate at %s.", FINETUNED_DIR)
            try:
                # Try loading it to verify it's valid
                temp_model = SentenceTransformer(str(FINETUNED_DIR), device="cpu")
                del temp_model
                model_path = str(FINETUNED_DIR)
                logger.info("Verified fine-tuned model. Using it.")
            except Exception as e:
                logger.error("Failed to load fine-tuned model: %s. Reverting to base model.", e)
                model_path = cfg.model_name
        else:
            logger.info("Using base model: %s", model_path)

        logger.info("Loading embedding model on %s", cfg.device)
        self.model = SentenceTransformer(model_path, device=cfg.device)

    def encode(self, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        logger.info("Encoding %d texts (batch_size=%d, normalize=%s)", len(texts), batch_size, normalize)
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        emb = emb.astype(np.float32, copy=False)
        logger.info("Embeddings shape: %s dtype=%s", emb.shape, emb.dtype)
        return emb
