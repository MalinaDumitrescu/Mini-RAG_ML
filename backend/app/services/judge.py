from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("rag.judge")


@dataclass
class JudgeConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_new_tokens: int = 512
    temperature: float = 0.0
    device: str = "cpu"
    json_max_chars: int = 6000


JUDGE_SYSTEM = """You are an expert judge for a RAG system.
Evaluate the assistant's answer based ONLY on the provided retrieved context.

You must output a single valid JSON object.
Do not output any text before or after the JSON.

Rules:
- If the assistant clearly refuses because the retrieved context is insufficient or off-topic,
  and it uses wording like: "I don't know based on the provided context." (or equivalent),
  then the verdict MUST be "pass".
  In that case:
  - supported_claims can be empty
  - unsupported_claims must be empty
  - hallucination_risk should be low
  - groundedness should be high (because it didn't invent facts)
- Otherwise, if the answer contains claims not supported by context, verdict must be "fail".
- Extract key claims from the answer. Classify each as supported/unsupported.
- Identify which chunk_ids are cited or should be cited.

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
  "supported_claims": ["<string>", ...],
  "unsupported_claims": ["<string>", ...],
  "cited_chunk_ids": ["<chunk_id>", ...],
  "reasons": ["<string>", ...],
  "missing_points": ["<string>", ...],
  "unsafe_or_policy": []
}
"""



def _extract_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


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

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

        logger.info("Judge LLM loaded successfully.")

    def _format_context(self, retrieved: List[Dict[str, str]]) -> str:
        # retrieved: [{"chunk_id": "...", "text": "..."}]
        blocks = []
        for i, item in enumerate(retrieved, start=1):
            cid = item.get("chunk_id", f"unknown_{i}")
            txt = item.get("text", "")
            blocks.append(f"[CONTEXT {i}] chunk_id={cid}\n{txt}")
        return "\n\n".join(blocks)

    def judge(
        self,
        question: str,
        answer: str,
        retrieved_chunks: List[Dict[str, str]],  # [{"chunk_id":..., "text":...}]
    ) -> Dict[str, Any]:
        context = self._format_context(retrieved_chunks)

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{question}\n\n"
                    f"RETRIEVED CONTEXT:\n{context}\n\n"
                    f"ASSISTANT ANSWER:\n{answer}\n\n"
                    "Produce the JSON evaluation now."
                ),
            },
        ]

        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text_input, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )

        generated_ids = out[0][inputs["input_ids"].shape[1] :]
        raw = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        json_str = _extract_json_object(raw) or raw
        json_str = json_str[: self.cfg.json_max_chars]

        try:
            data = json.loads(json_str)
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
                "supported_claims": [],
                "unsupported_claims": ["Judge output was not valid JSON."],
                "cited_chunk_ids": [],
                "reasons": ["Judge output was not valid JSON."],
                "missing_points": [],
                "unsafe_or_policy": [],
                "raw_output": json_str,
            }

        # --- Enforce schema / normalize ---
        required_top_level = [
            "verdict",
            "scores",
            "supported_claims",
            "unsupported_claims",
            "cited_chunk_ids",
            "reasons",
            "missing_points",
            "unsafe_or_policy",
        ]
        allowed_scores = {"correctness", "completeness", "hallucination_risk", "groundedness", "clarity"}

        for key in required_top_level:
            if key not in data:
                if key == "scores":
                    data[key] = {}
                elif key == "verdict":
                    data[key] = "fail"
                else:
                    data[key] = []

        if not isinstance(data["verdict"], str) or data["verdict"] not in ("pass", "fail"):
            data["verdict"] = "fail"

        if not isinstance(data["scores"], dict):
            data["scores"] = {}

        cleaned_scores = {}
        for k in allowed_scores:
            v = data["scores"].get(k, 0)
            try:
                v_int = int(v)
            except Exception:
                v_int = 0
            cleaned_scores[k] = max(0, min(10, v_int))
        data["scores"] = cleaned_scores

        # Lists: force list[str]
        list_fields = [
            "supported_claims",
            "unsupported_claims",
            "cited_chunk_ids",
            "reasons",
            "missing_points",
            "unsafe_or_policy",
        ]
        for k in list_fields:
            if not isinstance(data.get(k), list):
                data[k] = []
            data[k] = [str(x).strip() for x in data[k] if str(x).strip()]

        return data
