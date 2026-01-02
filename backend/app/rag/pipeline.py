# backend/app/rag/pipeline.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from backend.app.services.guardrails import check_input, check_output
from backend.app.services.judge import LLMJudge

from backend.app.rag.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from backend.app.rag.retrieval import Retriever, RetrievedChunk
from backend.app.rag.llm import GeneratorLLM, LLMConfig

logger = logging.getLogger("rag.pipeline")


@dataclass
class RAGConfig:
    top_k: int = 5


class RAGPipeline:
    def __init__(
        self,
        index_dir: Path,
        rag_cfg: RAGConfig | None = None,
        llm_cfg: LLMConfig | None = None
    ):
        self.rag_cfg = rag_cfg or RAGConfig()
        self.retriever = Retriever(index_dir)
        self.llm = GeneratorLLM(llm_cfg or LLMConfig())

        # Lazy-load judge to avoid GPU OOM and slow startup
        self.judge: LLMJudge | None = None

    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(f"[{i}] (score={c.score:.4f}, id={c.chunk_id})\n{c.text}")
        return "\n\n".join(parts)

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
            }

        # 2) Retrieve
        chunks = self.retriever.retrieve(question, top_k=self.rag_cfg.top_k)
        context = self._format_context(chunks)

        # 3) Generate
        prompt = SYSTEM_PROMPT + "\n\n" + USER_PROMPT_TEMPLATE.format(context=context, question=question)
        logger.info("Generating answer with %d retrieved chunks.", len(chunks))
        ans = self.llm.generate(prompt)

        # 4) Guardrails (OUTPUT)
        gr_out = check_output(ans)
        if not gr_out.get("ok", False):
            logger.warning("Guardrails output blocked: %s", gr_out.get("reason"))
            return {
                "question": question,
                "answer": gr_out.get("message", "Output blocked."),
                "retrieved": [{"chunk_id": c.chunk_id, "score": c.score, "text": c.text} for c in chunks],
                "judge": None,
                "guardrails": {"input": gr_in, "output": gr_out},
            }

        # 5) Judge (LLM-as-a-judge) — lazy load (and should be configured to run on CPU)
        if self.judge is None:
            logger.info("Lazy-loading Judge LLM...")
            self.judge = LLMJudge()  # ensure JudgeConfig.device="cpu" in judge.py

        retrieved_texts = [c.text for c in chunks]
        judge_result = self.judge.judge(question=question, answer=ans, retrieved_texts=retrieved_texts)

        return {
            "question": question,
            "answer": ans,
            "retrieved": [{"chunk_id": c.chunk_id, "score": c.score, "text": c.text} for c in chunks],
            "judge": judge_result,
            "guardrails": {"input": gr_in, "output": gr_out},
        }
