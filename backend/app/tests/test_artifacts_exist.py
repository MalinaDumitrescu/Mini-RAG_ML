from pathlib import Path
import json

from backend.app.core.paths import CORPUS_DIR, INDEX_DIR

def test_corpus_exists():
    p = CORPUS_DIR / "chunks.jsonl"
    assert p.exists(), "chunks.jsonl missing. Run build_corpus.py"

def test_index_exists():
    assert (INDEX_DIR / "faiss.index").exists(), "faiss.index missing. Run build_index.py"
    assert (INDEX_DIR / "docstore.json").exists(), "docstore.json missing"
    assert (INDEX_DIR / "id_map.json").exists(), "id_map.json missing"

def test_docstore_and_id_map_consistency():
    docstore_path = INDEX_DIR / "docstore.json"
    id_map_path = INDEX_DIR / "id_map.json"

    docstore = json.loads(docstore_path.read_text(encoding="utf-8"))
    id_map = json.loads(id_map_path.read_text(encoding="utf-8"))

    # pick a few positions and ensure they map to chunk IDs that exist
    sample_keys = list(id_map.keys())[:10]
    assert sample_keys, "id_map.json empty"

    for k in sample_keys:
        cid = id_map[k]
        assert cid in docstore, f"chunk_id {cid} from id_map not found in docstore"
