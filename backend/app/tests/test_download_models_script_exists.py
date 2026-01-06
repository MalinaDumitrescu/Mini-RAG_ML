# backend/app/tests/test_download_models_script_exists.py
from pathlib import Path

def test_download_models_script_exists():
    assert Path("scripts/download_models.py").exists()
