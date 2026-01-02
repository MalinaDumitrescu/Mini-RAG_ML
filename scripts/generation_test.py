# scripts/05_test_generation.py
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import logging

from backend.app.core.logging_config import setup_logging
from backend.app.core.paths import INDEX_DIR, LOGS_DIR
from backend.app.rag.pipeline import RAGPipeline


def main() -> None:
    setup_logging(LOGS_DIR / "rag.log", level=logging.INFO)

    rag = RAGPipeline(INDEX_DIR)

    q = "What is backpropagation in neural networks?"
    result = rag.answer(q)

    print("\nQUESTION:", result["question"])
    print("\nANSWER:\n", result["answer"])
    print("\nRETRIEVED:", result["retrieved"])


if __name__ == "__main__":
    main()
