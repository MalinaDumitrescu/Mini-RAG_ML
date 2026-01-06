import re
from backend.app.rag.pipeline import RAGPipeline
from backend.app.core.paths import INDEX_DIR

CIT_RE = re.compile(r"\[[^\[\]]+::c\d{6}\]")

def test_pipeline_smoke():
    pipe = RAGPipeline(INDEX_DIR)
    r = pipe.answer("Explain the difference between bias and variance.")
    assert "answer" in r
    assert "retrieved" in r
    assert isinstance(r["retrieved"], list)

def test_answer_has_citations_when_retrieval_ok():
    pipe = RAGPipeline(INDEX_DIR)
    r = pipe.answer("What is backpropagation?")
    # If retrieval gate refuses, skip citation requirement
    if r.get("retrieval_gate", {}).get("ok", True) is False:
        return
    assert CIT_RE.search(r["answer"]) is not None, "Answer missing chunk_id citations"
