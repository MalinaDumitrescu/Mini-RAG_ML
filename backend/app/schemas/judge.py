# backend/app/schemas/judge.py
from __future__ import annotations
from typing import Any, Dict, List, Literal
from pydantic import BaseModel

class JudgeScores(BaseModel):
    correctness: float
    completeness: float
    hallucination_risk: float
    groundedness: float
    clarity: float

class JudgeResult(BaseModel):
    verdict: Literal["pass", "fail"]
    scores: JudgeScores
    reasons: List[str]
    missing_points: List[str]
    unsafe_or_policy: List[str]
    raw_output: str | None = None
