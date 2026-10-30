"""Hybrid retrieval: fusion logic and the local BM25 store (no API keys needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document

from app.rag.hybrid import ChunkStore, HybridRetriever, rrf_fuse


def d(t):
    return Document(page_content=t, metadata={"source": t[:5]})


def test_rrf_rewards_agreement_between_retrievers():
    a = [d("vpn"), d("mfa"), d("printer")]
    b = [d("mfa"), d("laptop"), d("vpn")]
    fused = [x.page_content for x in rrf_fuse([a, b], top_n=3)]
    assert fused[:2] == ["mfa", "vpn"] or fused[:2] == ["vpn", "mfa"]
    assert "printer" not in fused[:2]


def test_rrf_deduplicates_the_same_chunk():
    fused = rrf_fuse([[d("same"), d("x")], [d("same")]], top_n=5)
    assert [x.page_content for x in fused].count("same") == 1


def test_chunk_store_finds_exact_identifiers(tmp_path):
    store = ChunkStore(tmp_path / "chunks.jsonl")
    store.add([d("Install the NovaSecure VPN client from the Software Portal"),
               d("Passwords must be at least 14 characters"),
               d("Error 0x80070005 means access denied during updates")])
    hits = store.search("what does 0x80070005 mean", 2)
    assert hits and "0x80070005" in hits[0].page_content


def test_hybrid_retriever_merges_dense_and_sparse(tmp_path):
    store = ChunkStore(tmp_path / "chunks.jsonl")
    store.add([d("ticket priority P2 means a major team is blocked"), d("lost laptop: report immediately"),
               d("MFA uses the authenticator app"), d("install software from the portal")])

    class FakeDense:
        def invoke(self, q):
            return [d("lost laptop: report immediately")]

    out = HybridRetriever(FakeDense(), store, top_k=2).invoke("what is P2 priority")
    texts = [x.page_content for x in out]
    assert "ticket priority P2 means a major team is blocked" in texts
