# backend/app/rag/pipeline.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from backend.app.services.guardrails import check_input, check_output
from backend.app.services.judge import LLMJudge, JudgeConfig

from backend.app.rag.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from backend.app.rag.retrieval import Retriever, RetrievedChunk
from backend.app.rag.llm import GeneratorLLM, LLMConfig

logger = logging.getLogger("rag.pipeline")

# citation pattern expected by tests: [DOC::c000123]
CIT_RE = re.compile(r"\[[^\[\]]+::c\d{6}\]")


@dataclass
class RAGConfig:
    top_k: int = 5

    # FAISS cosine similarity threshold (general)
    min_faiss_score: float = 0.35

    # If guardrails suspects off-topic, require stronger retrieval confidence
    # (prevents accepting random questions even if corpus contains unrelated text)
    off_topic_faiss_score: float = 0.60

    # Refuse if retrieval is too weak
    refuse_on_weak_retrieval: bool = True


class RAGPipeline:
    def __init__(
        self,
        index_dir: Path,
        rag_cfg: Optional[RAGConfig] = None,
        llm_cfg: Optional[LLMConfig] = None,
    ):
        self.rag_cfg = rag_cfg or RAGConfig()
        self.retriever = Retriever(index_dir)
        self.llm = GeneratorLLM(llm_cfg or LLMConfig())

        # Lazy-loaded judge (CPU)
        self.judge: Optional[LLMJudge] = None

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"[{i}] chunk_id={c.chunk_id} (faiss={c.faiss_score:.4f})\n{c.text}"
            )
        return "\n\n".join(parts)

    def _retrieval_gate(self, chunks: List[RetrievedChunk]) -> tuple[bool, dict]:
        if not chunks:
            return False, {
                "ok": False,
                "reason": "no_chunks",
                "best_faiss_score": None,
                "threshold": self.rag_cfg.min_faiss_score,
            }

        faiss_scores = [float(c.faiss_score) for c in chunks]
        best = max(faiss_scores)
        ok = best >= self.rag_cfg.min_faiss_score

        return ok, {
            "ok": ok,
            "reason": "ok" if ok else "weak_retrieval",
            "best_faiss_score": float(best),
            "threshold": self.rag_cfg.min_faiss_score,
            "top_faiss_scores": sorted(faiss_scores, reverse=True)[:5],
        }

    def _ensure_citations(self, answer: str, chunks: List[RetrievedChunk]) -> str:
        """
        Enforce at least one chunk citation when retrieval is OK.
        """
        if CIT_RE.search(answer):
            return answer

        if not chunks:
            return answer

        cites = " ".join(f"[{c.chunk_id}]" for c in chunks[:2])
        return f"{answer}\n\nSources: {cites}"

    # ---------------------------------------------------------
    # Main pipeline
    # ---------------------------------------------------------

    def answer(self, question: str) -> dict:
        # 1) Guardrails — INPUT
        gr_in = check_input(question)
        if not gr_in.get("ok", False):
            return {
                "question": question,
                "answer": gr_in.get("message", "Request blocked."),
                "retrieved": [],
                "judge": None,
                "guardrails": {"input": gr_in, "output": None},
                "retrieval_gate": None,
            }

        maybe_off_topic = gr_in.get("warning") == "maybe_off_topic"

        # 2) Retrieve
        chunks = self.retriever.retrieve(question, top_k=self.rag_cfg.top_k)
        retrieval_ok, retrieval_meta = self._retrieval_gate(chunks)

        best_faiss = retrieval_meta.get("best_faiss_score") or 0.0
        logger.info(
            "Retrieved %d chunks. retrieval_ok=%s best_faiss=%s maybe_off_topic=%s",
            len(chunks),
            retrieval_ok,
            best_faiss,
            maybe_off_topic,
        )

        # ✅ Off-topic override:
        # If the question looks off-topic, only proceed if retrieval is VERY strong
        if maybe_off_topic and float(best_faiss) < float(self.rag_cfg.off_topic_faiss_score):
            return {
                "question": question,
                "answer": (
                    "I don't know based on the provided context. "
                    "This question seems off-topic for the ML course sources. "
                    "Please ask a machine learning question."
                ),
                "retrieved": [
                    {
                        "chunk_id": c.chunk_id,
                        "faiss_score": c.faiss_score,
                        "score": c.score,
                        "text": c.text,
                        "metadata": c.metadata,
                    }
                    for c in chunks
                ],
                "judge": None,
                "guardrails": {"input": gr_in, "output": None},
                "retrieval_gate": {
                    **retrieval_meta,
                    "off_topic_gate": {
                        "applied": True,
                        "maybe_off_topic": True,
                        "off_topic_threshold": self.rag_cfg.off_topic_faiss_score,
                    },
                },
            }

        # Existing weak retrieval refusal
        if self.rag_cfg.refuse_on_weak_retrieval and not retrieval_ok:
            return {
                "question": question,
                "answer": (
                    "I don’t have enough relevant information in the provided course sources "
                    "to answer this reliably."
                ),
                "retrieved": [
                    {
                        "chunk_id": c.chunk_id,
                        "faiss_score": c.faiss_score,
                        "score": c.score,
                        "text": c.text,
                        "metadata": c.metadata,
                    }
                    for c in chunks
                ],
                "judge": None,
                "guardrails": {"input": gr_in, "output": None},
                "retrieval_gate": retrieval_meta,
            }

        # 3) Generate
        context = self._format_context(chunks)
        user_text = USER_PROMPT_TEMPLATE.format(context=context, question=question)

        ans = self.llm.generate_chat(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_text,
        )

        # Enforce citations when retrieval is OK
        if retrieval_ok:
            ans = self._ensure_citations(ans, chunks)

        # 4) Guardrails — OUTPUT
        gr_out = check_output(ans)
        if not gr_out.get("ok", False):
            return {
                "question": question,
                "answer": gr_out.get("message", "Output blocked."),
                "retrieved": [],
                "judge": None,
                "guardrails": {"input": gr_in, "output": gr_out},
                "retrieval_gate": retrieval_meta,
            }

        # 5) Judge (CPU, lazy-loaded)
        if self.judge is None:
            logger.info("Lazy-loading Judge LLM on CPU...")
            self.judge = LLMJudge(JudgeConfig(device="cpu"))

        judge_result = self.judge.judge(
            question,
            ans,
            [{"chunk_id": c.chunk_id, "text": c.text} for c in chunks],
        )

        return {
            "question": question,
            "answer": ans,
            "retrieved": [
                {
                    "chunk_id": c.chunk_id,
                    "faiss_score": c.faiss_score,
                    "score": c.score,
                    "text": c.text,
                    "metadata": c.metadata,
                }
                for c in chunks
            ],
            "judge": judge_result,
            "guardrails": {"input": gr_in, "output": gr_out},
            "retrieval_gate": retrieval_meta,
        }
