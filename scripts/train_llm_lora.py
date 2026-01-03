# scripts/train_llm_lora.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from peft import LoraConfig, get_peft_model

from backend.app.core.paths import CORPUS_DIR, MODELS_DIR, LOGS_DIR
from backend.app.core.logging_config import setup_logging


@dataclass
class TrainCfg:
    # ✅ train the same model you use for inference
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    out_dir: Path = MODELS_DIR / "llm_lora_qwen05b"

    # ✅ conservative for 4GB VRAM
    max_samples: int = 500
    max_seq_len: int = 384

    epochs: int = 1
    batch_size: int = 1
    grad_accum: int = 16

    lr: float = 2e-4
    warmup_ratio: float = 0.05

    fp16: bool = True


SYSTEM = "You are a course assistant. Answer using ONLY the provided context."


def load_chunks() -> List[Dict]:
    path = CORPUS_DIR / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing corpus at {path}. Run build_corpus.py first.")
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def make_examples(chunks: List[Dict], max_samples: int) -> List[Dict]:
    examples = []
    for c in chunks:
        ctx = (c.get("text") or "").strip()
        if len(ctx) < 200:
            continue

        user = (
            f"CONTEXT:\n{ctx}\n\n"
            "TASK:\n"
            "Summarize the key points from the context in 3-6 bullet points. "
            "Do not add any facts not present in the context."
        )

        assistant = ctx[:900].strip()

        examples.append({"system": SYSTEM, "user": user, "assistant": assistant})

        if len(examples) >= max_samples:
            break

    return examples


def format_chat(tokenizer, system: str, user: str, assistant: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def main() -> None:
    setup_logging(LOGS_DIR / "llm_finetune.log", level=logging.INFO)
    logger = logging.getLogger("train_llm_lora")

    cfg = TrainCfg()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device=%s", device)

    logger.info("Loading chunks...")
    chunks = load_chunks()
    logger.info("Chunks=%d", len(chunks))

    examples = make_examples(chunks, cfg.max_samples)
    logger.info("Training examples=%d", len(examples))
    if not examples:
        logger.error("No training examples created. Check corpus/chunking.")
        return

    ds = Dataset.from_list(examples)

    logger.info("Loading tokenizer/model: %s", cfg.base_model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        low_cpu_mem_usage=True,
        dtype=torch.float16 if (device == "cuda" and cfg.fp16) else torch.float32,
    )

    # 4GB VRAM helpers
    if device == "cuda":
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    model.to(device)

    # LoRA config (Qwen attention projections)
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def tokenize_fn(batch):
        texts = [
            format_chat(tokenizer, s, u, a)
            for s, u, a in zip(batch["system"], batch["user"], batch["assistant"])
        ]
        tok = tokenizer(
            texts,
            truncation=True,
            max_length=cfg.max_seq_len,
            padding="max_length",
        )
        tok["labels"] = tok["input_ids"].copy()
        return tok

    ds_tok = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)

    args = TrainingArguments(
        output_dir=str(cfg.out_dir),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=(device == "cuda" and cfg.fp16),
        report_to="none",
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds_tok)

    logger.info("Starting LoRA fine-tuning...")
    trainer.train()

    logger.info("Saving LoRA adapter to %s", cfg.out_dir)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.out_dir))
    tokenizer.save_pretrained(str(cfg.out_dir))

    logger.info("Done.")


if __name__ == "__main__":
    main()
