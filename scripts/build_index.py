# scripts/build_index.py
from __future__ import annotations

import json
import logging

from backend.app.core.paths import CORPUS_DIR, INDEX_DIR, LOGS_DIR
from backend.app.core.logging_config import setup_logging
from backend.app.rag.embeddings import EmbeddingModel, EmbeddingModelConfig
from backend.app.rag.vector_store_faiss import FaissStore, build_faiss_index, save_faiss

def main() -> None:
    setup_logging(LOGS_DIR / "rag.log", level=logging.INFO)

    chunks_path = CORPUS_DIR / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"Missing chunks file: {chunks_path}. Run scripts/02_build_corpus.py first.")
        return

    chunks = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    print(f"Loaded chunks: {len(chunks)}")

    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    emb_model = EmbeddingModel(EmbeddingModelConfig())
    embeddings = emb_model.encode(texts, batch_size=64, normalize=True)

    index = build_faiss_index(embeddings)

    # docstore: chunk_id -> text + metadata
    docstore = {c["chunk_id"]: {"text": c["text"], "metadata": c["metadata"]} for c in chunks}

    # mapping from FAISS position -> chunk_id (critical)
    id_map = {str(i): chunk_ids[i] for i in range(len(chunk_ids))}

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with (INDEX_DIR / "id_map.json").open("w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False)

    store = FaissStore(index=index, docstore=docstore)
    save_faiss(store, INDEX_DIR / "faiss.index", INDEX_DIR / "docstore.json")

    print("Index build done.")

if __name__ == "__main__":
    main()
