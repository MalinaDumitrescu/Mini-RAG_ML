from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Iterator

from backend.app.rag.corpus import Document

logger = logging.getLogger("rag.chunking")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any]


def _words(text: str) -> List[str]:
    # simple & robust word split for PDF-extracted text
    return text.split()


def chunk_document(
    doc: Document,
    chunk_words: int = 400,
    overlap_words: int = 80,
) -> List[Chunk]:
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be < chunk_words")

    tokens = _words(doc.text)
    n = len(tokens)

    logger.info("Chunking doc=%s words=%d (chunk=%d, overlap=%d)", doc.doc_id, n, chunk_words, overlap_words)

    chunks: List[Chunk] = []
    step = chunk_words - overlap_words

    start = 0
    chunk_index = 0
    while start < n:
        end = min(start + chunk_words, n)
        chunk_tokens = tokens[start:end]
        chunk_text = " ".join(chunk_tokens).strip()

        if chunk_text:
            cid = f"{doc.doc_id}::c{chunk_index:06d}"
            meta = dict(doc.metadata)
            meta.update({
                "chunk_index": chunk_index,
                "start_word": start,
                "end_word": end,
                "chunk_words": len(chunk_tokens),
            })
            chunks.append(Chunk(
                chunk_id=cid,
                doc_id=doc.doc_id,
                text=chunk_text,
                metadata=meta
            ))

        chunk_index += 1
        start += step

    logger.info("Created %d chunks for doc=%s", len(chunks), doc.doc_id)
    return chunks


def chunk_corpus(
    docs: List[Document],
    chunk_words: int = 400,
    overlap_words: int = 80,
) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for d in docs:
        all_chunks.extend(chunk_document(d, chunk_words=chunk_words, overlap_words=overlap_words))
    logger.info("Chunking done. Total chunks: %d", len(all_chunks))
    return all_chunks
