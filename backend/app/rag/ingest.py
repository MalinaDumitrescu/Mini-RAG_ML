# backend/app/rag/ingest.py
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
    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse repeated spaces/tabs
    text = _WHITESPACE_RE.sub(" ", text)
    # Collapse too many newlines
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    # Strip edges
    return text.strip()

def read_pdf(pdf_path: Path, max_pages: Optional[int] = None) -> Document:
    """
    Extract text from a single PDF.
    max_pages: if set, only reads the first N pages (useful for quick tests)
    """
    logger.info("Reading PDF: %s", pdf_path)
    reader = PdfReader(str(pdf_path))

    pages_text: List[str] = []
    n_pages = len(reader.pages)
    limit = min(n_pages, max_pages) if max_pages else n_pages

    for i in range(limit):
        page = reader.pages[i]
        page_text = page.extract_text() or ""
        page_text = _clean_text(page_text)
        if page_text:
            pages_text.append(page_text)

    full_text = "\n\n".join(pages_text).strip()
    doc_id = pdf_path.stem

    metadata = {
        "source_type": "pdf",
        "source_path": str(pdf_path),
        "filename": pdf_path.name,
        "num_pages_total": n_pages,
        "num_pages_read": limit,
    }

    logger.info("Extracted %d chars from %s (%d/%d pages).",
                len(full_text), pdf_path.name, limit, n_pages)

    return Document(doc_id=doc_id, text=full_text, metadata=metadata)

def ingest_directory(raw_dir: Path, glob_pattern: str = "*.pdf", max_files: Optional[int] = None) -> List[Document]:
    """
    Load all PDFs from a folder.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    pdf_paths = sorted(raw_dir.glob(glob_pattern))
    if max_files:
        pdf_paths = pdf_paths[:max_files]

    logger.info("Found %d PDFs in %s", len(pdf_paths), raw_dir)

    docs: List[Document] = []
    for p in pdf_paths:
        try:
            docs.append(read_pdf(p))
        except Exception as e:
            logger.exception("Failed to read %s: %s", p, e)

    logger.info("Ingestion done. Loaded %d documents.", len(docs))
    return docs
