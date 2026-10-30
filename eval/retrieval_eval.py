"""Offline retrieval + refusal evaluation for the IT-support copilot.

Why this exists: the agent's answer quality is capped by retrieval. If the right passage is not in the
top-k chunks, the grader can only say "weak" and the agent falls back to the web. So we measure
retrieval directly, on a labelled question set, before spending any LLM calls.

Corpus  = the sample NovaRetail KB (2 files) + the public-domain 18F/TTS Handbook (~250 real
          internal-handbook pages: laptops, PIV badges, passwords, incidents, Slack, Zoom, ...).
          The handbook makes the test realistic: ~2,000 chunks, lots of near-duplicate topics.
Labels  = eval/questions.yaml: 76 in-scope questions (source file + exact answer span) and
          20 out-of-scope questions whose answer is in neither KB.
Metrics = Hit@1, Hit@4 (the app retrieves top_k=4), MRR@10, nDCG@10, split by lexical vs paraphrased
          questions, with 95% bootstrap intervals; refusal = ROC-AUC / accuracy of a top-1 score
          threshold that separates in-scope from out-of-scope questions (2-fold cross-validated).

Run:  python eval/fetch_corpus.py && python eval/retrieval_eval.py --out eval/results
      (dense models need `pip install sentence-transformers`; OpenAI embeddings need OPENAI_API_KEY)
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- corpus
def clean_markdown(text: str) -> str:
    text = re.sub(r"^---.*?---", "", text, count=1, flags=re.S)          # front matter
    text = re.sub(r"\{%.*?%\}", " ", text)                                 # Liquid tags
    return text


def load_corpus(handbook_dir: Path | None) -> list[dict]:
    docs = []
    for p in sorted((ROOT / "data" / "sample_kb").glob("*.md")):
        docs.append({"source": f"sample_kb/{p.name}", "text": p.read_text()})
    if handbook_dir and handbook_dir.exists():
        for p in sorted((handbook_dir / "pages").rglob("*.md")):
            docs.append({"source": str(p.relative_to(handbook_dir / "pages")), "text": clean_markdown(p.read_text())})
    return docs


def chunk(docs: list[dict], size: int, overlap: int, header_aware: bool = False) -> list[dict]:
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
    rec = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    out = []
    for d in docs:
        if header_aware:
            md = MarkdownHeaderTextSplitter([("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False)
            parts = []
            for sec in md.split_text(d["text"]):
                heading = " > ".join(v for k, v in sec.metadata.items())
                for piece in rec.split_text(sec.page_content):
                    parts.append((f"{heading}\n{piece}" if heading else piece))
        else:
            parts = rec.split_text(d["text"])
        for i, t in enumerate(parts):
            out.append({"source": d["source"], "i": i, "text": t, "norm": " ".join(t.split())})
    return out


# ----------------------------------------------------------------------------- retrievers
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(t: str) -> list[str]:
    return _TOKEN.findall(t.lower())


class BM25:
    name = "BM25"

    def __init__(self, chunks):
        from rank_bm25 import BM25Okapi
        self.bm = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def scores(self, queries):
        return np.vstack([self.bm.get_scores(tokenize(q)) for q in queries])


class TfIdf:
    name = "TF-IDF (word 1-2 grams)"

    def __init__(self, chunks):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.m = self.v.fit_transform([c["text"] for c in chunks])

    def scores(self, queries):
        return (self.v.transform(queries) @ self.m.T).toarray()


class Dense:
    """sentence-transformers bi-encoder. Query/passage prefixes follow each model card."""
    PREFIX = {"BAAI/bge-small-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
              "BAAI/bge-base-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
              "intfloat/e5-base-v2": ("query: ", "passage: "),
              "intfloat/e5-small-v2": ("query: ", "passage: "),
              "nomic-ai/nomic-embed-text-v1.5": ("search_query: ", "search_document: ")}

    def __init__(self, chunks, model_name):
        from sentence_transformers import SentenceTransformer
        self.name = model_name.split("/")[-1]
        self.model_name = model_name
        self.m = SentenceTransformer(model_name, trust_remote_code=True)
        self.qp, self.pp = self.PREFIX.get(model_name, ("", ""))
        t0 = time.time()
        self.emb = self.m.encode([self.pp + c["text"] for c in chunks], batch_size=64, normalize_embeddings=True,
                                 show_progress_bar=False)
        self.index_s = time.time() - t0

    def scores(self, queries):
        q = self.m.encode([self.qp + x for x in queries], normalize_embeddings=True, show_progress_bar=False)
        return q @ self.emb.T


class OpenAIDense:
    def __init__(self, chunks, model="text-embedding-3-small"):
        from openai import OpenAI
        self.name, self.client, self.model = model, OpenAI(), model
        self.emb = self._embed([c["text"] for c in chunks])

    def _embed(self, texts):
        out = []
        for i in range(0, len(texts), 256):
            r = self.client.embeddings.create(model=self.model, input=texts[i:i + 256])
            out += [d.embedding for d in r.data]
        e = np.array(out, dtype=np.float32)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def scores(self, queries):
        return self._embed(queries) @ self.emb.T


class Hybrid:
    """Reciprocal-rank fusion (Cormack et al. 2009, k=60) of a sparse and a dense ranking."""

    def __init__(self, a, b, k: int = 60):
        self.a, self.b, self.k = a, b, k
        self.name = f"Hybrid RRF ({a.name} + {b.name})"

    def scores(self, queries):
        out = []
        for sa, sb in zip(self.a.scores(queries), self.b.scores(queries)):
            r = np.zeros_like(sa)
            for s in (sa, sb):
                rank = np.empty(len(s))
                rank[np.argsort(-s)] = np.arange(1, len(s) + 1)
                r += 1.0 / (self.k + rank)
            out.append(r)
        return np.vstack(out)


class Reranked:
    """First stage retrieves `pool` candidates, a cross-encoder re-scores them."""

    def __init__(self, first, chunks, model_name="BAAI/bge-reranker-base", pool: int = 30):
        from sentence_transformers import CrossEncoder
        self.first, self.chunks, self.pool = first, chunks, pool
        self.ce = CrossEncoder(model_name)
        self.name = f"{first.name} + rerank ({model_name.split('/')[-1]}, top {pool})"

    def scores(self, queries):
        base = self.first.scores(queries)
        out = np.full_like(base, -1e9)
        for qi, q in enumerate(queries):
            cand = np.argsort(-base[qi])[: self.pool]
            s = self.ce.predict([(q, self.chunks[j]["text"]) for j in cand])
            out[qi, cand] = s
        return out


# ----------------------------------------------------------------------------- metrics
@dataclass
class Labels:
    questions: list[str]
    relevant: list[set[int]]   # chunk ids that contain the answer span
    style: list[str]


def make_labels(items: list[dict], chunks: list[dict]) -> Labels:
    rel = []
    for e in items:
        span = " ".join(e["span"].split())
        ids = {i for i, c in enumerate(chunks) if c["source"] == e["source"] and span in c["norm"]}
        if not ids:   # span cut by a chunk boundary: fall back to the chunk with the longest prefix overlap
            best, best_len = None, 0
            for i, c in enumerate(chunks):
                if c["source"] != e["source"]:
                    continue
                for L in range(len(span), 8, -4):
                    if span[:L] in c["norm"] or span[-L:] in c["norm"]:
                        if L > best_len:
                            best, best_len = i, L
                        break
            ids = {best} if best is not None else set()
        rel.append(ids)
    return Labels([e["q"] for e in items], rel, [e["style"] for e in items])


def rank_metrics(scores: np.ndarray, labels: Labels) -> dict:
    per = []
    for s, rel in zip(scores, labels.relevant):
        order = np.argsort(-s)[:10]
        hits = [int(j in rel) for j in order]
        first = next((r for r, h in enumerate(hits, 1) if h), None)
        dcg = sum(h / np.log2(r + 1) for r, h in enumerate(hits, 1))
        idcg = sum(1 / np.log2(r + 1) for r in range(1, min(len(rel), 10) + 1)) or 1
        per.append({"hit1": hits[0], "hit4": int(any(hits[:4])), "mrr": 1 / first if first else 0.0,
                    "ndcg": dcg / idcg})
    return per


def summarize(per: list[dict], labels: Labels, rng: np.random.Generator, n_boot: int = 2000) -> dict:
    out = {}
    arr = {k: np.array([p[k] for p in per], float) for k in per[0]}
    idx = rng.integers(0, len(per), (n_boot, len(per)))
    for k, v in arr.items():
        boots = v[idx].mean(1)
        out[k] = {"mean": float(v.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    for style in ("lexical", "paraphrase"):
        m = np.array([s == style for s in labels.style])
        out[f"hit4_{style}"] = float(arr["hit4"][m].mean())
    return out


def paired_test(a: np.ndarray, b: np.ndarray, rng, n: int = 10000) -> float:
    """Two-sided paired sign-flip permutation test on per-question differences."""
    d = b - a
    obs = abs(d.mean())
    if obs == 0:
        return 1.0
    flips = rng.choice([-1, 1], size=(n, len(d)))
    return float(((np.abs((flips * d).mean(1)) >= obs).sum() + 1) / (n + 1))


def refusal_eval(score_in: np.ndarray, score_out: np.ndarray, rng) -> dict:
    """Can the top-1 retrieval score tell answerable from unanswerable questions?"""
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(score_in)), np.zeros(len(score_out))]
    s = np.r_[score_in, score_out]
    auc = roc_auc_score(y, s)
    # 2-fold CV for the threshold: pick it on one half, measure on the other
    accs = []
    for _ in range(50):
        perm = rng.permutation(len(y))
        A, B = perm[: len(y) // 2], perm[len(y) // 2:]
        for tr, te in ((A, B), (B, A)):
            cands = np.unique(s[tr])
            best = max(cands, key=lambda t: ((s[tr] >= t) == y[tr]).mean())
            accs.append(((s[te] >= best) == y[te]).mean())
    return {"auc": float(auc), "cv_accuracy": float(np.mean(accs)), "majority_baseline": float(y.mean())}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--handbook", type=Path, default=ROOT / "eval" / "corpus" / "handbook")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results")
    ap.add_argument("--dense", nargs="*", default=[], help="sentence-transformers model ids")
    ap.add_argument("--rerank", default="", help="cross-encoder id to rerank the best hybrid")
    ap.add_argument("--openai", action="store_true")
    ap.add_argument("--chunk-sizes", nargs="*", type=int, default=[300, 600, 900, 1500])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    qs = yaml.safe_load((ROOT / "eval" / "questions.yaml").read_text())
    docs = load_corpus(a.handbook)
    results = {"corpus": {"documents": len(docs), "characters": sum(len(d["text"]) for d in docs)},
               "n_in_scope": len(qs["in_scope"]), "n_out_of_scope": len(qs["out_of_scope"]),
               "chunking": {}, "retrievers": {}, "refusal": {}, "paired_tests": {}}
    per_q = {}

    # --- 1. chunking ablation with BM25 (cheap, model-free) and the first dense model ------------
    for size in a.chunk_sizes:
        for header in (False, True):
            ch = chunk(docs, size, int(size * 0.13), header)
            lab = make_labels(qs["in_scope"], ch)
            row = {"chunks": len(ch)}
            for R in [BM25] + ([lambda c: Dense(c, a.dense[0])] if a.dense else []):
                r = R(ch)
                row[r.name] = summarize(rank_metrics(r.scores(lab.questions), lab), lab, rng)["hit4"]["mean"]
            results["chunking"][f"{size}{' +headers' if header else ''}"] = row
            print("chunking", size, header, row, flush=True)

    # --- 2. retriever comparison at the app's chunk size (900/120) -------------------------------
    ch = chunk(docs, 900, 120, False)
    results["corpus"]["chunks_at_900"] = len(ch)
    lab = make_labels(qs["in_scope"], ch)
    outq = [e["q"] for e in qs["out_of_scope"]]
    sparse = BM25(ch)
    retrievers = [sparse, TfIdf(ch)]
    dense_models = [Dense(ch, m) for m in a.dense]
    retrievers += dense_models
    if a.openai:
        retrievers.append(OpenAIDense(ch))
    for d in dense_models:
        retrievers.append(Hybrid(sparse, d))
    if a.rerank and dense_models:
        best_dense = dense_models[0]
        retrievers.append(Reranked(Hybrid(sparse, best_dense), ch, a.rerank))
    for r in retrievers:
        t0 = time.time()
        s_in = r.scores(lab.questions)
        lat = (time.time() - t0) / len(lab.questions)
        per = rank_metrics(s_in, lab)
        per_q[r.name] = per
        m = summarize(per, lab, rng)
        m["latency_ms_per_query"] = 1000 * lat
        results["retrievers"][r.name] = m
        s_out = r.scores(outq)
        results["refusal"][r.name] = refusal_eval(s_in.max(1), s_out.max(1), rng)
        print(f"{r.name:60s} hit@1={m['hit1']['mean']:.3f} hit@4={m['hit4']['mean']:.3f} "
              f"mrr={m['mrr']['mean']:.3f} para={m['hit4_paraphrase']:.3f} "
              f"refusal_auc={results['refusal'][r.name]['auc']:.3f}", flush=True)
        # keep the failures for the error analysis
        fails = [{"q": q, "gold": e["source"], "top1": ch[int(np.argmax(s))]["source"]}
                 for q, e, s, p in zip(lab.questions, qs["in_scope"], s_in, per) if not p["hit4"]]
        results["retrievers"][r.name]["misses_at_4"] = fails

    names = list(per_q)
    base = "BM25"
    for n in names:
        if n != base:
            results["paired_tests"][f"{n} vs {base} (hit@4)"] = paired_test(
                np.array([p["hit4"] for p in per_q[base]]), np.array([p["hit4"] for p in per_q[n]]), rng)
    (a.out / "retrieval_results.json").write_text(json.dumps(results, indent=2))
    print("saved", a.out / "retrieval_results.json")


if __name__ == "__main__":
    main()
