from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Pattern

logger = logging.getLogger("rag.guardrails")


@dataclass
class GuardrailsConfig:
    max_question_chars: int = 2000

    topic_keywords: List[str] = None

    unsafe_keywords: List[str] = None

    injection_patterns: List[Pattern] = None

    output_block_patterns: List[Pattern] = None

    def __post_init__(self):
        if self.topic_keywords is None:
            self.topic_keywords = [
                "machine learning", "ml", "deep learning", "neural", "network",
                "backprop", "gradient", "loss", "optimizer", "regularization",
                "overfitting", "underfitting", "bias", "variance",
                "regression", "classification", "svm", "knn", "decision tree",
                "random forest", "boosting", "xgboost", "cross validation",
                "precision", "recall", "f1", "auc", "roc",
                "clustering", "k-means", "silhouette", "pca", "embedding",
                "transformer", "attention", "rnn", "lstm", "gru", "cnn",
            ]

        if self.unsafe_keywords is None:
            self.unsafe_keywords = [
                r"\bporn\b", r"\bxxx\b", r"\bnude\b",
                r"\bsex\b", r"\bsexual\b",
                r"\bchild\s*sexual\b", r"\bpedo\b", r"\bpedophil",
                r"\brape\b",
                r"\bkill\b", r"\bmurder\b", r"\bmassacre\b",
                r"\bbomb\b", r"\bexplosive\b", r"\bterror\b",
                r"\bsuicide\b", r"\bself[-\s]?harm\b",
            ]

        if self.injection_patterns is None:
            self.injection_patterns = [
                re.compile(r"ignore (all|previous|earlier) instructions", re.IGNORECASE),
                re.compile(r"disregard (the )?system prompt", re.IGNORECASE),
                re.compile(r"reveal (the )?(system|developer) prompt", re.IGNORECASE),
                re.compile(r"print (the )?(system|developer) message", re.IGNORECASE),
                re.compile(r"jailbreak", re.IGNORECASE),
                re.compile(r"act as (a|an) ", re.IGNORECASE),
                re.compile(r"you are now ", re.IGNORECASE),
                re.compile(r"bypass (the )?guardrails", re.IGNORECASE),
            ]

        if self.output_block_patterns is None:
            self.output_block_patterns = [
                re.compile(r"system prompt", re.IGNORECASE),
                re.compile(r"developer prompt", re.IGNORECASE),
                re.compile(r"here is the system", re.IGNORECASE),
                re.compile(r"ignore (all|previous) instructions", re.IGNORECASE),
            ]


def _contains_any_substring(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in keywords)


def _matches_any_regex(text: str, patterns: List[str]) -> bool:
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False


def _matches_any_compiled(text: str, patterns: List[Pattern]) -> bool:
    for p in patterns:
        if p.search(text):
            return True
    return False


def check_input(question: str, cfg: GuardrailsConfig | None = None) -> Dict:
    cfg = cfg or GuardrailsConfig()

    q = (question or "").strip()
    if not q:
        return {"ok": False, "reason": "empty", "message": "Please ask a question."}

    if len(q) > cfg.max_question_chars:
        return {"ok": False, "reason": "too_long", "message": "Question too long."}

    if _matches_any_compiled(q, cfg.injection_patterns):
        logger.warning("Blocked potential prompt injection attempt.")
        return {
            "ok": False,
            "reason": "prompt_injection",
            "message": "Request blocked (prompt injection / jailbreak attempt).",
        }

    if _matches_any_regex(q, cfg.unsafe_keywords):
        return {
            "ok": False,
            "reason": "unsafe",
            "message": "I can’t help with that request.",
        }

    topic_hit = _contains_any_substring(q, cfg.topic_keywords)
    if not topic_hit:
        return {
            "ok": True,
            "warning": "maybe_off_topic",
            "message": "Question may be off-topic; attempting retrieval-based check.",
        }

    return {"ok": True}


def check_output(answer: str, cfg: GuardrailsConfig | None = None) -> Dict:
    cfg = cfg or GuardrailsConfig()
    a = (answer or "").strip()

    if _matches_any_regex(a, cfg.unsafe_keywords):
        return {
            "ok": False,
            "reason": "unsafe_output",
            "message": "Output blocked due to unsafe content.",
        }

    if _matches_any_compiled(a, cfg.injection_patterns) or _matches_any_compiled(a, cfg.output_block_patterns):
        return {
            "ok": False,
            "reason": "unsafe_output_prompt_leak_or_injection",
            "message": "Output blocked due to policy/security risk.",
        }

    return {"ok": True}
