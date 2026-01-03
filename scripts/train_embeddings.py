# scripts/train_embeddings.py
from __future__ import annotations

import json
import logging
import shutil
import random
from pathlib import Path

import nltk
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import losses
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from torch.utils.data import DataLoader

from backend.app.core.paths import CORPUS_DIR, MODELS_DIR, LOGS_DIR
from backend.app.core.logging_config import setup_logging

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_embeddings() -> None:
    setup_logging(LOGS_DIR / "training.log", level=logging.INFO)
    logger = logging.getLogger("train_embeddings")
    
    # Set seed for reproducibility
    set_seed(42)

    # Ensure NLTK data is available
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        logger.info("Downloading NLTK punkt_tab data...")
        nltk.download('punkt_tab')

    # 1. Config
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size = 8
    
    # The "Sweet Spot" Configuration
    epochs = 1
    learning_rate = 1e-5    
    
    # Output path
    output_path = MODELS_DIR / "finetuned"
    
    # 2. Load Corpus
    chunks_path = CORPUS_DIR / "chunks.jsonl"
    if not chunks_path.exists():
        logger.error(f"Corpus not found at {chunks_path}. Run build_corpus.py first.")
        return

    logger.info("Loading corpus from %s...", chunks_path)
    train_sentences = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data.get("text", "").strip()
            if len(text) > 20:  # skip very short chunks
                train_sentences.append(text)

    logger.info("Loaded %d sentences for training.", len(train_sentences))

    if not train_sentences:
        logger.error("No sentences found.")
        return

    # 3. Initialize Model
    logger.info("Loading base model: %s", model_name)
    model = SentenceTransformer(model_name)

    # 4. Prepare Training Data (TSDAE)
    train_dataset = DenoisingAutoEncoderDataset(train_sentences)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 5. Loss Function
    train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name, tie_encoder_decoder=True)

    # 6. Train
    logger.info("Starting training for %d epochs (LR=%s)...", epochs, learning_rate)
    
    # Calculate warmup steps (10% of total steps)
    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * 0.1)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        weight_decay=0.01,       # Restored weight decay
        scheduler="warmupcosine", # Restored warmupcosine (this was the key!)
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        show_progress_bar=True
    )

    # 7. Save
    logger.info("Saving fine-tuned model to %s", output_path)
    
    if output_path.exists():
        try:
            shutil.rmtree(output_path)
        except Exception as e:
            logger.warning(f"Could not remove old directory: {e}")

    model.save(str(output_path))
    
    logger.info("Training complete.")

if __name__ == "__main__":
    train_embeddings()
