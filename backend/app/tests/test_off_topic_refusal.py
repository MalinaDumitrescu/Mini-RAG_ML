from backend.app.rag.pipeline import RAGPipeline
from backend.app.core.paths import INDEX_DIR

def test_off_topic_question_is_refused_or_idk():
    pipe = RAGPipeline(INDEX_DIR)
    r = pipe.answer("What is the capital of France?")
    ans = (r["answer"] or "").lower()
    if r.get("retrieval_gate", {}).get("ok") is False:
        assert True
    else:
        assert "i don't know" in ans or "don’t know" in ans or "do not know" in ans
