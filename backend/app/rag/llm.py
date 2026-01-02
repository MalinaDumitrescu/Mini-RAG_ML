# backend/app/rag/llm.py
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("rag.llm")


@dataclass
class LLMConfig:
    # smaller, stable local model (fits on RTX 3050 laptop)
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens: int = 256
    temperature: float = 0.2
    device: str = "cuda"  # fallback handled below


class GeneratorLLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

        device = cfg.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        logger.info("Loading LLM: %s on %s", cfg.model_name, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            dtype=dtype,
            device_map="cuda" if self.device == "cuda" else "cpu",
            low_cpu_mem_usage=True,
        )

        logger.info("LLM loaded successfully.")

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                do_sample=self.cfg.temperature > 0,
            )

        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Try to strip prompt echo if present
        if text.startswith(prompt):
            return text[len(prompt):].strip()
        return text.strip()
