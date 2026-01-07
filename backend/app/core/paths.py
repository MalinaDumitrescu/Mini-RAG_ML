from __future__ import annotations
from pathlib import Path

def project_root() -> Path:
    # .../mini-rag/backend/app/core/paths.py -> .../mini-rag
    return Path(__file__).resolve().parents[3]

ROOT = project_root()

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = ROOT / "artifacts"
CORPUS_DIR = ARTIFACTS_DIR / "corpus"
INDEX_DIR = ARTIFACTS_DIR / "index"
EVAL_DIR = ARTIFACTS_DIR / "eval"

MODELS_DIR = ROOT / "models"
HF_CACHE_DIR = MODELS_DIR / "hf_cache"
FINETUNED_DIR = MODELS_DIR / "finetuned"

LOGS_DIR = ROOT / "logs"
