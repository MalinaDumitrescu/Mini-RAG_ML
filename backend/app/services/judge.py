# backend/app/services/judge.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("rag.judge")


@dataclass
class JudgeConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_new_tokens: int = 512
    temperature: float = 0.0
    device: str = "cpu"


JUDGE_SYSTEM = """You are an expert judge for a RAG system.
Evaluate the assistant's answer based ONLY on the provided context.

You must output a single valid JSON object.
Do not output any text before or after the JSON.

Required JSON structure:
{
  "verdict": "pass" or "fail",
  "scores": {
    "correctness": <int 0-10>,
    "completeness": <int 0-10>,
    "hallucination_risk": <int 0-10>,
    "groundedness": <int 0-10>,
    "clarity": <int 0-10>
  },
  "reasons": ["<string>", ...],
  "missing_points": ["<string>", ...],
  "unsafe_or_policy": []
}

Example of a valid output:
{
  "verdict": "pass",
  "scores": {
    "correctness": 10,
    "completeness": 9,
    "hallucination_risk": 0,
    "groundedness": 10,
    "clarity": 10
  },
  "reasons": ["The answer accurately reflects the context."],
  "missing_points": [],
  "unsafe_or_policy": []
}
"""


class LLMJudge:
    def __init__(self, cfg: JudgeConfig | None = None):
        self.cfg = cfg or JudgeConfig()

        device = self.cfg.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.device = device
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        logger.info("Loading Judge LLM: %s on %s", self.cfg.model_name, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # IMPORTANT: if CPU, force CPU device_map
        device_map = "cuda" if self.device == "cuda" else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )

        logger.info("Judge LLM loaded successfully.")

    def _format_context(self, retrieved_texts: List[str]) -> str:
        blocks = []
        for i, t in enumerate(retrieved_texts, start=1):
            blocks.append(f"[CONTEXT {i}]\n{t}")
        return "\n\n".join(blocks)

    def judge(self, question: str, answer: str, retrieved_texts: List[str]) -> Dict[str, Any]:
        context = self._format_context(retrieved_texts)

        # Use chat template for Instruct models
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"QUESTION:\n{question}\n\nRETRIEVED CONTEXT:\n{context}\n\nASSISTANT ANSWER:\n{answer}\n\nProduce the JSON evaluation now."}
        ]
        
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text_input, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,            # deterministic
                temperature=None,           # avoid warnings
                top_p=None,
                top_k=None,
            )

        # Decode only the new tokens
        generated_ids = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Extract JSON if extra text exists
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

        try:
            data = json.loads(text)
            
            # Enforce schema structure
            required_top_level = ["verdict", "scores", "reasons", "missing_points", "unsafe_or_policy"]
            allowed_scores = {"correctness", "completeness", "hallucination_risk", "groundedness", "clarity"}
            
            # Ensure top-level keys exist
            for key in required_top_level:
                if key not in data:
                    if key == "scores":
                        data[key] = {}
                    elif key == "verdict":
                        data[key] = "fail"
                    else:
                        data[key] = []

            # Clean "scores" dictionary
            if isinstance(data["scores"], dict):
                scores = data["scores"]
                keys_to_move = []
                
                # Identify keys that shouldn't be in scores
                for k in scores.keys():
                    if k not in allowed_scores:
                        keys_to_move.append(k)
                
                # Move known lists to top level, delete others
                for k in keys_to_move:
                    if k in ["reasons", "missing_points", "unsafe_or_policy"]:
                        # Move content to top level if it's a list
                        if isinstance(scores[k], list):
                            if not isinstance(data[k], list):
                                data[k] = []
                            # Append unique items
                            existing = set(data[k])
                            for item in scores[k]:
                                if item not in existing:
                                    data[k].append(item)
                                    existing.add(item)
                    # Remove from scores
                    del scores[k]
            else:
                # If scores is not a dict, reset it
                data["scores"] = {k: 0 for k in allowed_scores}
            
            return data
        except Exception:
            logger.exception("Judge produced non-JSON output.")
            return {
                "verdict": "fail",
                "scores": {
                    "correctness": 0,
                    "completeness": 0,
                    "hallucination_risk": 10,
                    "groundedness": 0,
                    "clarity": 0,
                },
                "reasons": ["Judge output was not valid JSON."],
                "missing_points": [],
                "unsafe_or_policy": [],
                "raw_output": text[:2000],
            }
