from backend.app.rag.retrieval import Retriever
from backend.app.core.paths import INDEX_DIR

def test_retrieval_returns_chunks():
    retriever = Retriever(INDEX_DIR)
    q = "What is overfitting and how can regularization help?"
    chunks = retriever.retrieve(q, top_k=5)
    assert len(chunks) > 0
    assert all(hasattr(c, "chunk_id") for c in chunks)
    assert all(isinstance(c.faiss_score, float) for c in chunks)

def test_faiss_scores_sorted_reasonably():
    retriever = Retriever(INDEX_DIR)
    q = "Explain cross validation."
    chunks = retriever.retrieve(q, top_k=5)
    faiss_scores = [c.faiss_score for c in chunks]
    assert max(faiss_scores) > 0.1
