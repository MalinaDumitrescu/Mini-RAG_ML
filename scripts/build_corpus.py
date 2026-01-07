from __future__ import annotations

import json
import logging

from backend.app.core.paths import RAW_DIR, CORPUS_DIR, LOGS_DIR
from backend.app.core.logging_config import setup_logging
from backend.app.rag.ingest import ingest_directory
from backend.app.rag.chunking import chunk_corpus

def main() -> None:
    setup_logging(LOGS_DIR / "rag.log", level=logging.INFO)

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    docs = ingest_directory(RAW_DIR)
    if not docs:
        print(f"No PDFs found in: {RAW_DIR}")
        return

    chunks = chunk_corpus(docs, chunk_words=400, overlap_words=80)

    out_path = CORPUS_DIR / "chunks.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "metadata": c.metadata,
            }, ensure_ascii=False) + "\n")

    print(f"Saved chunks to: {out_path} (total={len(chunks)})")

if __name__ == "__main__":
    main()
