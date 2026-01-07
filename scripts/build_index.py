from __future__ import annotations

import json
import logging
from datetime import datetime

from backend.app.core.paths import CORPUS_DIR, INDEX_DIR, LOGS_DIR
from backend.app.core.logging_config import setup_logging
from backend.app.rag.embeddings import EmbeddingModel, EmbeddingModelConfig
from backend.app.rag.vector_store_faiss import FaissStore, build_faiss_index, save_faiss


def main() -> None:
    setup_logging(LOGS_DIR / "rag.log", level=logging.INFO)
    logger = logging.getLogger("build_index")

    chunks_path = CORPUS_DIR / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"Missing chunks file: {chunks_path}. Run build_corpus.py first.")
        return

    chunks = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    print(f"Loaded chunks: {len(chunks)}")

    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    emb_cfg = EmbeddingModelConfig()
    emb_model = EmbeddingModel(emb_cfg)
    embeddings = emb_model.encode(texts, batch_size=64, normalize=True)

    index = build_faiss_index(embeddings)

    docstore = {c["chunk_id"]: {"text": c["text"], "metadata": c["metadata"]} for c in chunks}
    id_map = {str(i): chunk_ids[i] for i in range(len(chunk_ids))}

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with (INDEX_DIR / "id_map.json").open("w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False)

    store = FaissStore(index=index, docstore=docstore)
    save_faiss(store, INDEX_DIR / "faiss.index", INDEX_DIR / "docstore.json")

    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "num_chunks": len(chunks),
        "embedding_model_name": emb_cfg.model_name,
        "embedding_use_finetuned": emb_cfg.use_finetuned,
        "notes": "FAISS IndexFlatIP with normalized embeddings => cosine similarity.",
    }
    with (INDEX_DIR / "index_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("Index build done. Wrote index_meta.json.")


if __name__ == "__main__":
    main()
