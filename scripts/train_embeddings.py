# scripts/train_embeddings.py
from __future__ import annotations

import json
import logging
import random
import shutil

import nltk
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from torch.utils.data import DataLoader

from backend.app.core.paths import CORPUS_DIR, LOGS_DIR, FINETUNED_DIR
from backend.app.core.logging_config import setup_logging


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_embeddings() -> None:
    setup_logging(LOGS_DIR / "training.log", level=logging.INFO)
    logger = logging.getLogger("train_embeddings")

    set_seed(42)

    # Ensure NLTK data is available
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        logger.info("Downloading NLTK punkt_tab data...")
        nltk.download("punkt_tab")

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size = 8

    epochs = 1
    learning_rate = 1e-5

    # IMPORTANT: Save where runtime expects it (FINETUNED_DIR)
    output_path = FINETUNED_DIR

    chunks_path = CORPUS_DIR / "chunks.jsonl"
    if not chunks_path.exists():
        logger.error("Corpus not found at %s. Run build_corpus.py first.", chunks_path)
        return

    logger.info("Loading corpus from %s...", chunks_path)
    train_sentences = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = (data.get("text") or "").strip()
            if len(text) > 20:
                train_sentences.append(text)

    logger.info("Loaded %d sentences for training.", len(train_sentences))
    if not train_sentences:
        logger.error("No sentences found.")
        return

    logger.info("Loading base model: %s", model_name)
    model = SentenceTransformer(model_name)

    train_dataset = DenoisingAutoEncoderDataset(train_sentences)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_loss = losses.DenoisingAutoEncoderLoss(
        model,
        decoder_name_or_path=model_name,
        tie_encoder_decoder=True,
    )

    logger.info("Starting training for %d epochs (LR=%s)...", epochs, learning_rate)
    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * 0.1)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        weight_decay=0.01,
        scheduler="warmupcosine",
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        show_progress_bar=True,
    )

    logger.info("Saving fine-tuned model to %s", output_path)

    if output_path.exists():
        try:
            shutil.rmtree(output_path)
        except Exception as e:
            logger.warning("Could not remove old directory: %s", e)

    output_path.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))

    logger.info("Training complete.")


if __name__ == "__main__":
    train_embeddings()
