# backend/app/services/guardrails.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger("rag.guardrails")


@dataclass
class GuardrailsConfig:
    # simple keywords; pentru laborator e ok + explici în README
    max_question_chars: int = 2000
    # topic keywords for ML course
    topic_keywords = [
        "machine learning", "ml", "neural", "network", "backprop", "gradient",
        "regression", "classification", "svm", "knn", "decision tree",
        "random forest", "boosting", "xgboost", "overfitting", "underfitting",
        "cross validation", "loss", "optimizer", "embedding", "transformer",
        "dropout", "batch normalization", "cnn", "rnn", "lstm",
        "precision", "recall", "f1", "auc", "roc"
    ]
    # unsafe keywords (simple demo)
    unsafe_keywords = [
        "porn", "sex", "nude", "xxx", "child", "rape",
        "kill", "murder", "bomb", "terror", "suicide", "self-harm"
    ]


def _contains_any(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(k in t for k in keywords)


def check_input(question: str, cfg: GuardrailsConfig | None = None) -> Dict:
    cfg = cfg or GuardrailsConfig()

    if not question or not question.strip():
        return {"ok": False, "reason": "empty", "message": "Please ask a question."}

    if len(question) > cfg.max_question_chars:
        return {"ok": False, "reason": "too_long", "message": "Question too long."}

    if _contains_any(question, cfg.unsafe_keywords):
        return {
            "ok": False,
            "reason": "unsafe",
            "message": "I can’t help with that request.",
        }

    # Off-topic: dacă nu conține nimic ML-ish, refuză politicos
    if not _contains_any(question, cfg.topic_keywords):
        return {
            "ok": False,
            "reason": "off_topic",
            "message": "I can only answer questions related to machine learning topics covered by the provided course sources.",
        }

    return {"ok": True}


def check_output(answer: str, cfg: GuardrailsConfig | None = None) -> Dict:
    cfg = cfg or GuardrailsConfig()

    if _contains_any(answer, cfg.unsafe_keywords):
        return {
            "ok": False,
            "reason": "unsafe_output",
            "message": "Output blocked due to unsafe content.",
        }

    return {"ok": True}
