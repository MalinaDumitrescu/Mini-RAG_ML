from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

from PyPDF2 import PdfReader

from backend.app.rag.corpus import Document

logger = logging.getLogger("rag.ingest")

_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def read_pdf_pages(pdf_path: Path, max_pages: Optional[int] = None) -> List[Document]:
    """
    Extract text per page -> returns one Document per page.
    This enables page-level traceability in chunk metadata.
    """
    logger.info("Reading PDF pages: %s", pdf_path)
    reader = PdfReader(str(pdf_path))

    n_pages = len(reader.pages)
    limit = min(n_pages, max_pages) if max_pages else n_pages

    docs: List[Document] = []
    base_doc_id = pdf_path.stem

    for i in range(limit):
        page = reader.pages[i]
        page_text = page.extract_text() or ""
        page_text = _clean_text(page_text)

        if not page_text:
            continue

        doc_id = f"{base_doc_id}::p{i+1:04d}"
        metadata = {
            "source_type": "pdf",
            "source_path": str(pdf_path),
            "filename": pdf_path.name,
            "num_pages_total": n_pages,
            "page_number": i + 1,
        }
        docs.append(Document(doc_id=doc_id, text=page_text, metadata=metadata))

    logger.info("Extracted %d page-docs from %s (%d/%d pages).", len(docs), pdf_path.name, limit, n_pages)
    return docs


def ingest_directory(raw_dir: Path, glob_pattern: str = "*.pdf", max_files: Optional[int] = None) -> List[Document]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    pdf_paths = sorted(raw_dir.glob(glob_pattern))
    if max_files:
        pdf_paths = pdf_paths[:max_files]

    logger.info("Found %d PDFs in %s", len(pdf_paths), raw_dir)

    docs: List[Document] = []
    for p in pdf_paths:
        try:
            docs.extend(read_pdf_pages(p))
        except Exception as e:
            logger.exception("Failed to read %s: %s", p, e)

    logger.info("Ingestion done. Loaded %d documents (page-level).", len(docs))
    return docs
