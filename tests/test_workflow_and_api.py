"""Routing logic and API guards, tested without any API keys or network calls."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

from app.rag import workflow
from app.api import routes


def state(**kw):
    base = {"source_used": "", "kb_grade": "", "web_grade": "", "retry_count": 0}
    base.update(kw)
    return base


def test_router_sends_it_questions_to_the_knowledge_base():
    assert workflow.route_after_router(state(source_used="kb")) == "retrieve_kb"
    assert workflow.route_after_router(state(source_used="direct")) == "direct_answer"


def test_good_kb_evidence_is_answered_without_web_search():
    assert workflow.after_kb(state(kb_grade="good")) == "generate_from_kb"
    assert workflow.after_kb(state(kb_grade="weak")) == "search_web"


def test_weak_web_evidence_retries_then_declines():
    limit = workflow.settings.max_retries
    assert workflow.after_web(state(web_grade="good")) == "generate_from_web"
    assert workflow.after_web(state(web_grade="weak", retry_count=0)) == ("rewrite_query" if limit > 0 else "insufficient")
    assert workflow.after_web(state(web_grade="weak", retry_count=limit)) == "insufficient"


def test_graph_has_all_ten_nodes():
    nodes = set(workflow.agent_graph.get_graph().nodes) - {"__start__", "__end__"}
    assert len(nodes) == 10


@pytest.fixture
def client(monkeypatch, tmp_path):
    from app.main import app
    monkeypatch.setattr(routes.settings, "upload_dir", str(tmp_path))
    return TestClient(app)


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_upload_refused_while_admin_key_is_default(client, monkeypatch):
    monkeypatch.setattr(routes.settings, "admin_api_key", "change-me")
    r = client.post("/api/ingest", files={"file": ("a.txt", b"hello")}, headers={"x-admin-key": "change-me"})
    assert r.status_code == 503


def test_upload_rejects_wrong_key_and_bad_type(client, monkeypatch):
    monkeypatch.setattr(routes.settings, "admin_api_key", "s3cret-key")
    assert client.post("/api/ingest", files={"file": ("a.txt", b"x")}, headers={"x-admin-key": "nope"}).status_code == 401
    assert client.post("/api/ingest", files={"file": ("a.exe", b"x")}, headers={"x-admin-key": "s3cret-key"}).status_code == 400


def test_chat_hides_internal_errors(client, monkeypatch):
    def boom(_):
        raise RuntimeError("OPENAI_API_KEY is missing: secret detail")
    monkeypatch.setattr(routes, "ask", boom)
    r = client.post("/api/chat", json={"question": "How do I reset my password?"})
    assert r.status_code == 500 and "secret" not in r.text
