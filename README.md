# MINI-RAG System – Introduction to Machine Learning 

This project implements a **minimalist Retrieval-Augmented Generation (RAG) system**.
The goal of the project is to build a language model that can answer **machine-learning-related questions** based strictly on a **given course dataset**, while preventing hallucinations, unsafe content, and off-topic usage.

The system runs fully **locally**, uses fine-tuned embeddings and a fine-tuned generator model, and includes **guardrails**, **LLM-as-a-judge**, and a **simple web-based GUI**.

---

## Project Overview

The system follows the standard RAG pipeline:

1. **Document ingestion**
2. **Chunking with overlap**
3. **Embedding fine-tuning**
4. **Vector indexing (FAISS)**
5. **Retrieval + reranking**
6. **Answer generation with citations**
7. **LLM-as-a-judge evaluation**
8. **Guardrails (input and output)**
9. **Chat-based GUI**

Only information contained in the provided course literature is used for answering questions.

---

## Dataset and Sources

The knowledge base is built from the official **literature references of the ML course**, provided as PDF files.
Additional machine-learning textbooks were also included.

Documents are read using **PyPDF2**, converted to text, and split into overlapping chunks to respect LLM token limits while preserving context.

All processed chunks are stored in:

```
artifacts/corpus/chunks.jsonl
```

---

## Embeddings and Fine-Tuning

* Base embedding model:
  `sentence-transformers/all-MiniLM-L6-v2`
* The embedding model is **fine-tuned on the course corpus**
* The same embeddings are used for:

  * document indexing
  * query representation

The fine-tuned model is stored locally and reused across runs.

---

## Retrieval and Vector Store

* Vector store: **FAISS (IndexFlatIP)**
* Embeddings are L2-normalized to enable cosine similarity
* Retrieval returns top-k candidates
* Optional reranking combines:

  * dense similarity
  * light lexical overlap
  * pair-coverage logic for comparison questions (e.g. “X vs Y”)

The index and metadata are stored in:

```
artifacts/index/
```

---

## Generator Model

* Base model:
  `Qwen/Qwen2.5-0.5B-Instruct`
* Fine-tuning method: **LoRA**
* Training data: selected chunks from the course corpus
* The LoRA adapter is saved locally and loaded at runtime

The generator is instructed to:

* use **only retrieved context**
* cite every claim with a chunk ID
* refuse when information is missing

---

## Guardrails

Guardrails are applied at two levels:

### Input Guardrails

* Block unsafe content (violence, weapons, etc.)
* Detect prompt-injection attempts
* Detect potentially off-topic questions

### Output Guardrails

* Prevent leaking system prompts
* Block unsafe or injected output

Off-topic questions are refused unless retrieval confidence is sufficiently high.

---

## LLM-as-a-Judge

A **second LLM** is used to evaluate generated answers.

The judge checks:

* factual correctness
* completeness
* grounding in the retrieved context
* hallucinations

If an answer is unsupported or incomplete, the judge returns a **FAIL** verdict.
Clear refusals such as *“I don’t know based on the provided context”* are explicitly allowed to **PASS**.

---

## Citations

Each answer includes **inline citations** in the format:

```
[DOCUMENT::c000123]
```

Citations refer to specific chunks used during retrieval and are enforced automatically if missing.

---

## Logging and Verbose Output

All steps in the pipeline log status information, including:

* model loading
* retrieval scores
* reranking decisions
* guardrail decisions
* judge evaluation

This makes the system behavior transparent and debuggable.

---

## GUI

A simple **chat-based GUI** is implemented using **React + Vite**.

Features:

* conversation history
* typing indicator
* dark/light mode
* backend connection via REST API
* 
<img width="1920" height="1080" alt="Screenshot (223)" src="https://github.com/user-attachments/assets/8b4c55b9-b5a9-46cb-bbb7-04389a63f654" />
<img width="1920" height="1080" alt="Screenshot (222)" src="https://github.com/user-attachments/assets/4a11515d-5e12-4921-8693-7a5f90a768ae" />

The frontend communicates with the backend through:

```
POST /api/v1/chat
```

---

## How to Run the Project

### Backend (Python)

Create and activate a virtual environment, then run:

rdinea corectă (și logică) e asta:

scripts/download_models.py
scripts/build_corpus.py
scripts/train_embeddings.py
scripts/build_index.py
scripts/train_llm_lora.py

Start the backend API:

```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8008
```

---

### Frontend (GUI)

From the frontend directory:

```bash
npm install
npm run dev
```

Open:

```
http://localhost:3000
```

---

## Testing

Unit and integration tests verify:

* presence and consistency of artifacts
* retrieval behavior
* guardrails
* citation enforcement
* API responses
* off-topic refusal logic

All tests pass successfully.

---

## Notes

* All models run locally after the initial download.
* The system strictly avoids using outside knowledge.
* Prompt design was iteratively refined to reduce hallucinations and improve judge scores.

---

## Conclusion

This project fulfills all of those:

* fine-tuned embeddings
* fine-tuned generator
* retrieval with chunking and overlap
* LLM-as-a-judge
* guardrails
* logging
* local execution
* interactive GUI

