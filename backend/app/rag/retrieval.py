from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.app.rag.embeddings import EmbeddingModel, EmbeddingModelConfig
from backend.app.rag.vector_store_faiss import FaissStore, load_faiss, normalize_embeddings

logger = logging.getLogger("rag.retrieval")


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    faiss_score: float
    text: str
    metadata: Dict[str, Any]


@dataclass
class RerankConfig:
    enabled: bool = True
    oversample_factor: int = 6       # top_k*6 candidates from FAISS
    alpha: float = 0.08              # lexical weight (small; FAISS is still main signal)

    # General coverage logic for "X vs Y" style questions
    ensure_pair_coverage: bool = True
    pair_bonus: float = 0.12         # bonus if chunk contains either X or Y
    min_token_len: int = 2
    max_query_terms: int = 12


class Retriever:
    _STOPWORDS = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are",
        "what", "why", "how", "explain", "difference", "between", "compare", "vs", "versus",
        "does", "do", "give", "tell", "define", "describe",
    }

    _PAIR_PATTERNS = [
        re.compile(
            r"\b(?P<a>[a-z0-9][a-z0-9\-\s]{0,60}?)\s+(?:vs\.?|versus)\s+(?P<b>[a-z0-9][a-z0-9\-\s]{0,60}?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bdifference\s+between\s+(?P<a>[a-z0-9][a-z0-9\-\s]{0,60}?)\s+and\s+(?P<b>[a-z0-9][a-z0-9\-\s]{0,60}?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bcompare\s+(?P<a>[a-z0-9][a-z0-9\-\s]{0,60}?)\s+(?:and|with)\s+(?P<b>[a-z0-9][a-z0-9\-\s]{0,60}?)\b",
            re.IGNORECASE,
        ),
    ]

    def __init__(
        self,
        index_dir: Path,
        emb_cfg: Optional[EmbeddingModelConfig] = None,
        rerank_cfg: Optional[RerankConfig] = None,
    ):
        self.index_dir = index_dir
        self.rerank_cfg = rerank_cfg or RerankConfig()

        self.store: FaissStore = load_faiss(
            index_dir / "faiss.index",
            index_dir / "docstore.json",
        )

        id_map_path = index_dir / "id_map.json"
        if not id_map_path.exists():
            raise FileNotFoundError(f"Missing id_map.json at {id_map_path}")

        with id_map_path.open("r", encoding="utf-8") as f:
            self.id_map: Dict[str, str] = json.load(f)

        self.emb = EmbeddingModel(emb_cfg or EmbeddingModelConfig())

        d = getattr(self.store.index, "d", None)
        logger.info("Retriever ready. FAISS dim=%s, docstore=%d chunks", d, len(self.store.docstore))

    def _embed_query(self, question: str) -> np.ndarray:
        q_emb = self.emb.encode([question], batch_size=1, normalize=True)
        q_emb = q_emb.astype(np.float32, copy=False)
        q_emb = normalize_embeddings(q_emb)
        return q_emb

    def _extract_query_terms(self, question: str) -> List[str]:
        q = (question or "").lower()
        tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", q)

        terms: List[str] = []
        for t in tokens:
            if len(t) < self.rerank_cfg.min_token_len:
                continue
            if t in self._STOPWORDS:
                continue
            terms.append(t)

        seen = set()
        uniq: List[str] = []
        for t in terms:
            if t not in seen:
                uniq.append(t)
                seen.add(t)

        return uniq[: self.rerank_cfg.max_query_terms]

    def _lexical_score(self, text: str, terms: List[str]) -> float:
        if not terms:
            return 0.0
        t = (text or "").lower()
        hits = 0
        for term in terms:
            if term in t:
                hits += 1
        return hits / len(terms)

    def _normalize_phrase(self, s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^\w\s\-]", "", s)
        return s.strip()

    def _extract_pair_terms(self, question: str) -> Optional[Tuple[str, str]]:
        """
        Detect pair-comparison questions and extract (X, Y).
        Returns None if not detected.

        We keep phrases modest length and normalize whitespace.
        """
        q = (question or "").strip()
        if not q:
            return None

        for pat in self._PAIR_PATTERNS:
            m = pat.search(q)
            if not m:
                continue
            a = self._normalize_phrase(m.group("a"))
            b = self._normalize_phrase(m.group("b"))

            if not a or not b:
                continue
            if len(a) < 2 or len(b) < 2:
                continue
            if a == b:
                continue

            a = a[:60].strip()
            b = b[:60].strip()

            return a, b

        return None

    def _phrase_hit(self, text: str, phrase: str) -> bool:
        """
        "Hit" if phrase appears as substring, or if all tokens of phrase appear.
        This handles cases like 'batch normalization' split across text.
        """
        if not phrase:
            return False
        t = (text or "").lower()
        p = phrase.lower()

        if p in t:
            return True

        toks = [x for x in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", p) if x and x not in self._STOPWORDS]
        if not toks:
            return False
        return all(tok in t for tok in toks)

    def retrieve_with_scores(self, question: str, top_k: int = 5) -> Tuple[List[RetrievedChunk], Dict[str, Any]]:
        q_preview = (question or "").strip().replace("\n", " ")
        if len(q_preview) > 120:
            q_preview = q_preview[:120] + "..."

        oversample = max(top_k, top_k * self.rerank_cfg.oversample_factor)

        logger.info(
            "Retrieving top_k=%d (oversample=%d, rerank=%s) for question: %s",
            top_k, oversample, self.rerank_cfg.enabled, q_preview
        )

        q_emb = self._embed_query(question)
        faiss_scores, idxs = self.store.index.search(q_emb, oversample)

        candidates: List[RetrievedChunk] = []
        for faiss_pos, fs in zip(idxs[0], faiss_scores[0]):
            if int(faiss_pos) == -1:
                continue
            faiss_key = str(int(faiss_pos))
            chunk_id = self.id_map.get(faiss_key)
            if not chunk_id:
                continue
            entry = self.store.docstore.get(chunk_id)
            if not entry:
                continue

            candidates.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    score=float(fs),
                    faiss_score=float(fs),
                    text=entry.get("text", ""),
                    metadata=entry.get("metadata", {}) or {},
                )
            )

        pair = self._extract_pair_terms(question) if self.rerank_cfg.ensure_pair_coverage else None

        if self.rerank_cfg.enabled and candidates:
            terms = self._extract_query_terms(question)

            scored: List[Tuple[RetrievedChunk, Dict[str, bool]]] = []
            for c in candidates:
                lex = self._lexical_score(c.text, terms)

                bonus = 0.0
                hits = {"a": False, "b": False, "both": False}

                if pair:
                    a, b = pair
                    ha = self._phrase_hit(c.text, a)
                    hb = self._phrase_hit(c.text, b)
                    hits = {"a": ha, "b": hb, "both": ha and hb}

                    if ha:
                        bonus += self.rerank_cfg.pair_bonus
                    if hb:
                        bonus += self.rerank_cfg.pair_bonus

                c.score = c.faiss_score + (self.rerank_cfg.alpha * lex) + bonus
                scored.append((c, hits))

            results: List[RetrievedChunk] = []
            used_ids = set()

            if pair and self.rerank_cfg.ensure_pair_coverage:
                a, b = pair
                both_pool = [c for (c, h) in scored if h["both"]]
                if both_pool:
                    both_pool.sort(key=lambda x: x.score, reverse=True)
                    results = both_pool[:top_k]
                else:
                    a_pool = [c for (c, h) in scored if h["a"]]
                    b_pool = [c for (c, h) in scored if h["b"]]
                    a_pool.sort(key=lambda x: x.score, reverse=True)
                    b_pool.sort(key=lambda x: x.score, reverse=True)

                    if a_pool:
                        results.append(a_pool[0])
                        used_ids.add(a_pool[0].chunk_id)

                    if b_pool:
                        best_b = next((x for x in b_pool if x.chunk_id not in used_ids), None)
                        if best_b:
                            results.append(best_b)
                            used_ids.add(best_b.chunk_id)

                    remaining = [c for (c, _h) in scored if c.chunk_id not in used_ids]
                    remaining.sort(key=lambda x: x.score, reverse=True)
                    results.extend(remaining[: max(0, top_k - len(results))])

            else:
                all_sorted = [c for (c, _h) in scored]
                all_sorted.sort(key=lambda x: x.score, reverse=True)
                results = all_sorted[:top_k]

        else:
            candidates.sort(key=lambda x: x.faiss_score, reverse=True)
            results = candidates[:top_k]

        top_scores = [r.score for r in results[: min(5, len(results))]]
        top_faiss = [r.faiss_score for r in results[: min(5, len(results))]]

        meta = {
            "retrieved_count": len(results),
            "best_score": float(results[0].score) if results else None,
            "top_scores": top_scores,
            "top_faiss_scores": top_faiss,
            "rerank": {
                "enabled": self.rerank_cfg.enabled,
                "oversample": oversample,
                "alpha": self.rerank_cfg.alpha,
                "ensure_pair_coverage": self.rerank_cfg.ensure_pair_coverage,
                "pair_bonus": self.rerank_cfg.pair_bonus,
            },
            "pair_detection": {
                "detected": bool(pair),
                "pair": list(pair) if pair else None,
            },
        }

        logger.info(
            "Retrieved %d chunks. best_score=%s top_scores=%s top_faiss=%s pair=%s",
            len(results), meta["best_score"], top_scores, top_faiss, meta["pair_detection"]["pair"]
        )

        return results, meta

    def retrieve(self, question: str, top_k: int = 5) -> List[RetrievedChunk]:
        chunks, _meta = self.retrieve_with_scores(question, top_k=top_k)
        return chunks
