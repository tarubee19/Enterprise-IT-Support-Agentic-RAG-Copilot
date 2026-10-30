"""Hybrid retrieval: BM25 over a local copy of the chunks + the dense Pinecone retriever, fused with
reciprocal-rank fusion (Cormack et al., 2009).

Why: on the labelled evaluation set (eval/README.md) pure keyword search misses paraphrased questions
and pure dense search misses exact identifiers (product names, error strings, ticket codes). Fusing the
two rankings recovers both, without any score calibration between them - RRF only uses ranks.

The BM25 side needs the chunk texts locally, so ingestion appends every chunk to
``data/chunks.jsonl`` (one JSON object per line: text + metadata). It is rebuilt lazily.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from langchain_core.documents import Document

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def rrf_fuse(rankings: list[list[Document]], k: int = 60, top_n: int = 4) -> list[Document]:
    """Reciprocal-rank fusion. Documents are identified by their text (the same chunk from both
    retrievers counts once)."""
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            docs.setdefault(key, doc)
    order = sorted(scores, key=lambda key: -scores[key])
    return [docs[key] for key in order[:top_n]]


class ChunkStore:
    """Append-only JSONL copy of the indexed chunks, with a BM25 index built on demand."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._bm25 = None
        self._docs: list[Document] = []

    def add(self, chunks: list[Document]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps({"text": c.page_content, "metadata": c.metadata}) + "\n")
            self._bm25 = None   # rebuild on next search

    def _load(self) -> None:
        from rank_bm25 import BM25Okapi
        docs = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    docs.append(Document(page_content=row["text"], metadata=row.get("metadata", {})))
        self._docs = docs
        self._bm25 = BM25Okapi([tokenize(d.page_content) for d in docs]) if docs else None

    def search(self, query: str, n: int) -> list[Document]:
        with self._lock:
            if self._bm25 is None:
                self._load()
            if self._bm25 is None:
                return []
            scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
        return [self._docs[i] for i in order if scores[i] > 0]


class HybridRetriever:
    """Same `.invoke(query) -> list[Document]` interface as a LangChain retriever."""

    def __init__(self, dense, store: ChunkStore, top_k: int = 4, candidates: int = 20):
        self.dense, self.store, self.top_k, self.candidates = dense, store, top_k, candidates

    def invoke(self, query: str) -> list[Document]:
        dense_docs = self.dense.invoke(query)
        sparse_docs = self.store.search(query, self.candidates)
        return rrf_fuse([dense_docs, sparse_docs], top_n=self.top_k)
