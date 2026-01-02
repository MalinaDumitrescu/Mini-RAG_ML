# scripts/04_test_retrieval.py
from __future__ import annotations

import logging

from backend.app.core.logging_config import setup_logging
from backend.app.core.paths import INDEX_DIR, LOGS_DIR
from backend.app.rag.retrieval import Retriever


def main() -> None:
    setup_logging(LOGS_DIR / "rag.log", level=logging.INFO)

    r = Retriever(INDEX_DIR)

    questions = [
        "What is backpropagation in neural networks?",
        "Explain the difference between dropout and batch normalization.",
        "What is an embedding and why is it useful for retrieval?",
    ]

    for q in questions:
        print("\n" + "=" * 80)
        print("Q:", q)
        hits = r.retrieve(q, top_k=5)

        for i, h in enumerate(hits, 1):
            preview = (h.text[:240] + "...").replace("\n", " ")
            print(f"{i}) score={h.score:.4f}  id={h.chunk_id}")
            print("   ", preview)


if __name__ == "__main__":
    main()
