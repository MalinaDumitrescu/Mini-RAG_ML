# scripts/download_models.py
from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.app.core.paths import LOGS_DIR, FINETUNED_DIR, MODELS_DIR
from backend.app.core.logging_config import setup_logging


def download_models() -> None:
    setup_logging(LOGS_DIR / "download_models.log", level=logging.INFO)
    logger = logging.getLogger("download_models")

    gen_model = "Qwen/Qwen2.5-1.5B-Instruct"
    judge_model = "Qwen/Qwen2.5-0.5B-Instruct"
    emb_model = "sentence-transformers/all-MiniLM-L6-v2"

    logger.info("Prefetching generator model: %s", gen_model)
    _ = AutoTokenizer.from_pretrained(gen_model, use_fast=True)
    _ = AutoModelForCausalLM.from_pretrained(gen_model, low_cpu_mem_usage=True)
    logger.info("Generator cached.")

    logger.info("Prefetching judge model: %s", judge_model)
    _ = AutoTokenizer.from_pretrained(judge_model, use_fast=True)
    _ = AutoModelForCausalLM.from_pretrained(judge_model, low_cpu_mem_usage=True)
    logger.info("Judge cached.")

    logger.info("Prefetching embedding model: %s", emb_model)
    _ = SentenceTransformer(emb_model, device="cpu")
    logger.info("Embeddings cached.")

    if FINETUNED_DIR.exists() and any(FINETUNED_DIR.iterdir()):
        logger.info("Found finetuned embeddings directory at %s", FINETUNED_DIR)
    else:
        logger.info("No finetuned embeddings found yet at %s (ok).", FINETUNED_DIR)

    lora_dir = MODELS_DIR / "llm_lora"
    if lora_dir.exists() and any(lora_dir.iterdir()):
        logger.info("Found LoRA adapter directory at %s", lora_dir)
    else:
        logger.info("No LoRA adapter found yet at %s (ok).", lora_dir)

    logger.info("Done prefetching.")


if __name__ == "__main__":
    download_models()
