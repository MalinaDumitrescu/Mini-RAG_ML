# backend/app/evaluation/run_judge_eval.py
from __future__ import annotations

import json
import logging
from pathlib import Path

from backend.app.core.logging_config import setup_logging
from backend.app.core.paths import EVAL_DIR, INDEX_DIR, LOGS_DIR
from backend.app.rag.pipeline import RAGPipeline

def run_evaluation() -> None:
    setup_logging(LOGS_DIR / "evaluation.log", level=logging.INFO)
    logger = logging.getLogger("eval")

    golden_set_path = Path(__file__).parent / "golden_set.json"
    if not golden_set_path.exists():
        logger.error("Golden set not found at %s", golden_set_path)
        return

    with golden_set_path.open("r", encoding="utf-8") as f:
        golden_set = json.load(f)

    logger.info("Loaded %d evaluation cases.", len(golden_set))

    # Initialize Pipeline
    rag = RAGPipeline(INDEX_DIR)

    results = []

    for i, case in enumerate(golden_set, 1):
        question = case.get("question")
        expected = case.get("expected_answer")

        logger.info("Evaluating [%d/%d]: %s", i, len(golden_set), question)

        # Run RAG
        output = rag.answer(question)

        # The pipeline already runs the Judge if configured.
        # We just capture the result.
        judge_res = output.get("judge")

        record = {
            "question": question,
            "expected": expected,
            "generated": output["answer"],
            "judge_verdict": judge_res.get("verdict") if judge_res else "N/A",
            "judge_scores": judge_res.get("scores") if judge_res else {},
            "retrieved_ids": [c["chunk_id"] for c in output["retrieved"]]
        }
        results.append(record)

    # Save Report
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    report_path = EVAL_DIR / "judge_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Evaluation complete. Report saved to %s", report_path)

    # Calculate aggregate metrics
    pass_count = sum(1 for r in results if r["judge_verdict"] == "pass")
    logger.info("Pass Rate: %d/%d (%.2f%%)", pass_count, len(results), (pass_count/len(results))*100)

if __name__ == "__main__":
    run_evaluation()
