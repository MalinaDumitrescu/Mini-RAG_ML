# backend/app/rag/llm.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 🔒 PEFT is optional (graceful fallback)
try:
    from peft import PeftModel
except Exception:
    PeftModel = None

from backend.app.core.paths import MODELS_DIR

logger = logging.getLogger("rag.llm")


@dataclass
class LLMConfig:
    # Generator model (fits 4GB VRAM well)
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    max_new_tokens: int = 256
    temperature: float = 0.2
    device: str = "cuda"  # falls back to CPU if unavailable

    max_input_tokens: Optional[int] = 2048

    # LoRA adapter directory (optional)
    lora_dir: Path = MODELS_DIR / "llm_lora_qwen05b"


class GeneratorLLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

        requested = (cfg.device or "cpu").lower()
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            requested = "cpu"
        self.device = requested

        logger.info("Loading GENERATOR LLM base: %s on %s", cfg.model_name, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )

        # VRAM helper
        if self.device == "cuda":
            self.model.config.use_cache = False

        self.model.to(self.device)
        self.model.eval()

        # 🔒 Robust LoRA adapter loading
        adapter_cfg = cfg.lora_dir / "adapter_config.json"
        if PeftModel is None:
            logger.info("PEFT not installed — running base model only.")
        elif adapter_cfg.exists():
            logger.info("Loading LoRA adapter from %s", cfg.lora_dir)
            try:
                self.model = PeftModel.from_pretrained(self.model, str(cfg.lora_dir))
                self.model.to(self.device)
                self.model.eval()
                logger.info("LoRA adapter loaded successfully.")
            except Exception:
                logger.exception("Failed to load LoRA adapter — continuing with base model.")
        else:
            logger.info("No LoRA adapter found at %s (running base model).", cfg.lora_dir)

        logger.info("Generator LLM ready (dtype=%s).", dtype)

    def _tokenize(self, text: str):
        if self.cfg.max_input_tokens is None:
            return self.tokenizer(text, return_tensors="pt")

        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=int(self.cfg.max_input_tokens),
        )

    def generate_chat(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenize(text_input)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        temp = float(self.cfg.temperature)
        do_sample = temp > 0.0

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=int(self.cfg.max_new_tokens),
                do_sample=do_sample,
                temperature=temp if do_sample else None,
                top_p=None if not do_sample else 0.95,
                top_k=None if not do_sample else 50,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def generate(self, prompt: str) -> str:
        inputs = self._tokenize(prompt)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        temp = float(self.cfg.temperature)
        do_sample = temp > 0.0

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=int(self.cfg.max_new_tokens),
                do_sample=do_sample,
                temperature=temp if do_sample else None,
                top_p=None if not do_sample else 0.95,
                top_k=None if not do_sample else 50,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        if text.startswith(prompt):
            return text[len(prompt):].strip()
        return text
