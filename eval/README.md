# Retrieval and refusal evaluation

The agent's answers can only be as good as what retrieval puts in front of the grader. If the right passage is
not in the top 4 chunks, the grader says "weak", the agent goes to the web, and a question the company knowledge
base could have answered gets an external answer. So the first thing to measure is retrieval, with labels,
before spending any LLM calls.

## Test set

* **Corpus (realistic size and noise):** the sample NovaRetail KB (2 files) plus the public-domain
  [18F / TTS Handbook](https://github.com/18F/handbook): 242 real internal-handbook pages (laptops, PIV badges,
  passwords, security incidents, Slack, Zoom, GitHub, travel, HR...). 1.48 M characters, 2,286 chunks at the app's
  900/120 setting. `python eval/fetch_corpus.py` pins it to a commit.
* **Labels:** `eval/questions.yaml`.
  * **76 answerable questions**, each with its source page and an exact answer span. A chunk counts as relevant only
    if it contains the span, so the labels do not depend on the chunk size.
  * **42 paraphrased** questions ("I left my work notebook computer on the train") and **34 lexical** ones
    ("How do I lock my MacBook?"). Real users paraphrase; keyword search mostly handles the lexical kind.
  * **20 unanswerable IT questions** (Cisco BGP, Intune, Jira...). These should *not* be answered from the KB.
* Every span was checked programmatically against the source text.
* Caveat: the questions were written for this project, from the documents. They are not real help-desk logs.

## Metrics

Hit@1, **Hit@4** (the app retrieves `top_k = 4`), MRR@10 and nDCG@10, each with a 95% bootstrap interval, and
Hit@4 split by lexical and paraphrased questions. Retrievers are compared with a paired sign-flip permutation test.
**Refusal:** can the top-1 retrieval score alone separate answerable from unanswerable questions? This is measured as
ROC-AUC and as the accuracy of a threshold chosen on one half of the questions and tested on the other (2-fold, 50 repeats).

## Results so far (sparse retrievers, run in this repo's sandbox)

| Retriever | Hit@1 | Hit@4 (95% CI) | MRR@10 | Hit@4 lexical | Hit@4 paraphrased | Refusal AUC |
|---|---|---|---|---|---|---|
| BM25 | 0.553 | 0.671 (0.566-0.776) | 0.604 | 0.912 | **0.476** | 0.72 |
| TF-IDF (1-2 grams) | 0.513 | 0.711 (0.618-0.816) | 0.613 | 0.882 | 0.571 | 0.72 |

What this already shows:

1. **Keyword search is fine for lexical questions (91%) and fails on paraphrases (48%).** That is the case for
   dense embeddings, and the reason the app uses them.
2. **A retrieval-score threshold cannot replace the LLM grader for refusals.** Its cross-validated accuracy
   (0.78) does not beat always answering "in scope" (0.79). Unanswerable IT questions share plenty of vocabulary
   with the handbook ("password", "laptop", "network"). This is evidence for the design decision to grade evidence
   with an LLM.
3. **Markdown-header-aware chunking** (each chunk prefixed with its section path) lifts BM25 Hit@4 from 0.671 to 0.684,
   a small change in the direction you would expect.

## Dense, hybrid and re-ranking (GitHub Actions)

The open embedding models (`bge-small`, `all-MiniLM-L6-v2`, `bge-base`, `e5-base`), hybrid BM25 + dense with
reciprocal-rank fusion, a `bge-reranker-base` cross-encoder, and the chunk-size ablation (300 / 600 / 900 / 1500,
with and without header prefixes) need Hugging Face downloads. They run in
`.github/workflows/retrieval-eval.yml`, which commits `eval/results/retrieval_results.json`.
Run locally: `pip install sentence-transformers rank-bm25 langchain-text-splitters pyyaml scikit-learn` and

```bash
python eval/fetch_corpus.py
python eval/retrieval_eval.py --out eval/results \
  --dense BAAI/bge-small-en-v1.5 sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-base-en-v1.5 intfloat/e5-base-v2 \
  --rerank BAAI/bge-reranker-base
```

## What changed in the app because of this

* **Hybrid retrieval** (`app/rag/hybrid.py`, `RETRIEVAL_MODE=hybrid`, the default). BM25 over a local copy of the
  chunks (`data/chunks.jsonl`, written at ingestion) is fused with the Pinecone results by reciprocal-rank fusion.
  Fusion uses only ranks, so the two score scales never need calibrating. It keeps exact-identifier matches
  (error codes, product names) that embeddings can blur. `RETRIEVAL_MODE=dense` restores the original behaviour.
  Unit tests: `tests/test_hybrid.py`.
* Refusals stay with the LLM grader, for the reason in point 2 above.
