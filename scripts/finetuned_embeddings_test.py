# scripts/finetuned_embeddings_test.py
from __future__ import annotations
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.app.core.paths import FINETUNED_DIR

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test_embeddings")

    base_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    logger.info(f"Loading base model: {base_model_name}")
    base_model = SentenceTransformer(base_model_name)
    
    if not FINETUNED_DIR.exists():
        logger.error(f"Fine-tuned model not found at {FINETUNED_DIR}. Run scripts/train_embeddings.py first.")
        return
        
    logger.info(f"Loading fine-tuned model from: {FINETUNED_DIR}")
    finetuned_model = SentenceTransformer(str(FINETUNED_DIR))
    
    test_sentences = [
        "What is backpropagation?",
        "Machine learning models require training data.",
        "The loss function minimizes error."
    ]
    
    logger.info("Encoding test sentences...")
    base_emb = base_model.encode(test_sentences)
    ft_emb = finetuned_model.encode(test_sentences)
    
    # Calculate difference (Cosine Similarity between base and fine-tuned for same sentence)
    # If they are identical (similarity ~ 1.0), fine-tuning didn't change anything or wasn't loaded.
    
    print("\n=== Comparison (Base vs Fine-Tuned) ===")
    for i, text in enumerate(test_sentences):
        # Dot product of normalized vectors = Cosine Similarity
        # SentenceTransformers output is usually normalized if normalize_embeddings=True, 
        # but let's manually normalize to be safe for comparison.
        v1 = base_emb[i] / np.linalg.norm(base_emb[i])
        v2 = ft_emb[i] / np.linalg.norm(ft_emb[i])
        
        similarity = np.dot(v1, v2)
        diff = 1.0 - similarity
        
        print(f"Sentence: '{text}'")
        print(f"  Similarity: {similarity:.4f}")
        print(f"  Difference: {diff:.4f}")
        
        if diff < 0.0001:
            print("  [WARNING] Models appear identical for this sentence.")
        else:
            print("  [OK] Models are different.")
            
    print("\nDone.")

if __name__ == "__main__":
    main()
