# backend/app/rag/pipeline.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from backend.app.services.guardrails import check_input, check_output
from backend.app.services.judge import LLMJudge, JudgeConfig

from backend.app.rag.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from backend.app.rag.retrieval import Retriever, RetrievedChunk
from backend.app.rag.llm import GeneratorLLM, LLMConfig

logger = logging.getLogger("rag.pipeline")


@dataclass
class RAGConfig:
    top_k: int = 5

    # IMPORTANT:
    # - Your FAISS score is cosine similarity (IndexFlatIP with normalized embeddings).
    # - Your reranked score is "faiss_score + lexical bonuses".
    # Gate MUST use faiss_score, otherwise the bonuses inflate confidence.
    min_faiss_score: float = 0.35

    # If retrieval is empty or weak, refuse (instead of inventing)
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

        # Lazy-load judge to avoid GPU OOM and slow startup
        self.judge: Optional[LLMJudge] = None

    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Context shown to the generator.

        We include BOTH:
        - faiss_score: dense similarity (cosine)
        - score: reranked score (dense + lexical bonuses)

        The model should not overfit to scores; they're here mainly for debugging.
        """
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"[{i}] chunk_id={c.chunk_id} (faiss={c.faiss_score:.4f})\n{c.text}"
            )
        return "\n\n".join(parts)

    def _retrieval_gate(self, chunks: List[RetrievedChunk]) -> tuple[bool, dict]:
        """
        Gate based on FAISS cosine similarity only (NOT reranked score).
        Returns (ok, meta).
        """
        if not chunks:
            return False, {
                "ok": False,
                "reason": "no_chunks",
                "best_faiss_score": None,
                "threshold": self.rag_cfg.min_faiss_score,
                "score_semantics": "faiss_score ~= cosine similarity (normalized IndexFlatIP)",
            }

        faiss_scores = [c.faiss_score for c in chunks]
        best = max(faiss_scores)

        ok = best >= self.rag_cfg.min_faiss_score

        return ok, {
            "ok": ok,
            "reason": "ok" if ok else "weak_retrieval",
            "best_faiss_score": float(best),
            "threshold": self.rag_cfg.min_faiss_score,
            "top_faiss_scores": [float(x) for x in sorted(faiss_scores, reverse=True)[:5]],
            "score_semantics": "faiss_score ~= cosine similarity (normalized IndexFlatIP)",
        }

    def answer(self, question: str) -> dict:
        # 1) Guardrails (INPUT)
        gr_in = check_input(question)
        if not gr_in.get("ok", False):
            logger.warning("Guardrails input blocked: %s", gr_in.get("reason"))
            return {
                "question": question,
                "answer": gr_in.get("message", "Request blocked."),
                "retrieved": [],
                "judge": None,
                "guardrails": {"input": gr_in, "output": None},
                "retrieval_gate": None,
            }

        # 2) Retrieve (Retriever returns reranked order, but also includes faiss_score)
        chunks = self.retriever.retrieve(question, top_k=self.rag_cfg.top_k)
        retrieval_ok, retrieval_meta = self._retrieval_gate(chunks)

        logger.info(
            "Retrieved %d chunks. retrieval_ok=%s best_faiss=%s threshold=%s",
            len(chunks),
            retrieval_ok,
            retrieval_meta.get("best_faiss_score"),
            retrieval_meta.get("threshold"),
        )

        if self.rag_cfg.refuse_on_weak_retrieval and not retrieval_ok:
            # Refuse rather than hallucinate
            return {
                "question": question,
                "answer": (
                    "I don’t have enough relevant information in the provided course sources "
                    "to answer this reliably. Please rephrase the question or ask something "
                    "closer to the covered ML topics."
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

        context = self._format_context(chunks)

        # 3) Generate
        user_text = USER_PROMPT_TEMPLATE.format(context=context, question=question)
        logger.info("Generating answer with %d retrieved chunks.", len(chunks))
        # NOTE: this assumes your GeneratorLLM has generate_chat(system_prompt, user_prompt)
        ans = self.llm.generate_chat(system_prompt=SYSTEM_PROMPT, user_prompt=user_text)

        # 4) Guardrails (OUTPUT)
        gr_out = check_output(ans)
        if not gr_out.get("ok", False):
            logger.warning("Guardrails output blocked: %s", gr_out.get("reason"))
            return {
                "question": question,
                "answer": gr_out.get("message", "Output blocked."),
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
                "guardrails": {"input": gr_in, "output": gr_out},
                "retrieval_gate": retrieval_meta,
            }

        # 5) Judge (LLM-as-a-judge) — lazy load (CPU)
        if self.judge is None:
            logger.info("Lazy-loading Judge LLM on CPU...")
            self.judge = LLMJudge(JudgeConfig(device="cpu"))

        judge_result = self.judge.judge(
            question=question,
            answer=ans,
            retrieved_chunks=[{"chunk_id": c.chunk_id, "text": c.text} for c in chunks],
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
