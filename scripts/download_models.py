from __future__ import annotations

import logging
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.app.core.paths import LOGS_DIR, FINETUNED_DIR, MODELS_DIR
from backend.app.core.logging_config import setup_logging


def _exists_dir_with_file(dir_path: Path, filename: str) -> bool:
    return dir_path.exists() and (dir_path / filename).exists()


def download_models() -> None:
    setup_logging(LOGS_DIR / "download_models.log", level=logging.INFO)
    logger = logging.getLogger("download_models")

    gen_model = "Qwen/Qwen2.5-0.5B-Instruct"
    judge_model = "Qwen/Qwen2.5-0.5B-Instruct"
    emb_model = "sentence-transformers/all-MiniLM-L6-v2"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device available: %s", device)

    # -------- Generator --------
    logger.info("Prefetching GENERATOR model: %s", gen_model)
    _ = AutoTokenizer.from_pretrained(gen_model, use_fast=True)
    _ = AutoModelForCausalLM.from_pretrained(gen_model, low_cpu_mem_usage=True)
    logger.info("Generator cached.")

    # -------- Judge --------
    logger.info("Prefetching JUDGE model: %s", judge_model)
    _ = AutoTokenizer.from_pretrained(judge_model, use_fast=True)
    _ = AutoModelForCausalLM.from_pretrained(judge_model, low_cpu_mem_usage=True)
    logger.info("Judge cached.")

    # -------- Embeddings (base) --------
    logger.info("Prefetching EMBEDDING model: %s", emb_model)
    _ = SentenceTransformer(emb_model, device="cpu")
    logger.info("Embeddings cached.")

    if FINETUNED_DIR.exists() and any(FINETUNED_DIR.iterdir()):
        logger.info("Found finetuned embeddings directory at %s", FINETUNED_DIR)
        try:
            _ = SentenceTransformer(str(FINETUNED_DIR), device="cpu")
            logger.info("Finetuned embeddings load OK.")
        except Exception as e:
            logger.warning("Finetuned embeddings exist but failed to load: %s", e)
    else:
        logger.info("No finetuned embeddings found yet at %s (ok).", FINETUNED_DIR)

    lora_dir = MODELS_DIR / "llm_lora_qwen05b"
    adapter_cfg = lora_dir / "adapter_config.json"

    if adapter_cfg.exists():
        logger.info("Found LoRA adapter at %s", lora_dir)
    else:
        logger.info("No LoRA adapter found yet at %s (ok).", lora_dir)

    logger.info("Done prefetching.")


if __name__ == "__main__":
    download_models()
