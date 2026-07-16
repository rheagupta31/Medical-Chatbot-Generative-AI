import pytest
from fastapi.testclient import TestClient

import app as app_module


class FakeChain:
    """Stands in for MedicalRagChain so tests never hit Pinecone/OpenAI/Gemini."""

    def __init__(self, settings=None):
        pass

    def ask(self, session_id: str, question: str) -> dict:
        return {
            "answer": f"[fake answer for: {question}]",
            "disclaimer": "This is not medical advice.",
            "sources": ["Medical_book.pdf"],
        }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "MedicalRagChain", FakeChain)
    with TestClient(app_module.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_rejects_empty_message(client):
    response = client.post("/api/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_returns_answer_and_disclaimer(client):
    response = client.post("/api/chat", json={"message": "What is Acne?"})
    assert response.status_code == 200
    body = response.json()
    assert "acne" in body["answer"].lower() or "Acne" in body["answer"]
    assert body["disclaimer"]
    assert "session_id" in body


def test_chat_surfaces_rate_limit_as_429(monkeypatch):
    class RateLimitedChain(FakeChain):
        def ask(self, session_id, question):
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")

    monkeypatch.setattr(app_module, "MedicalRagChain", RateLimitedChain)
    with TestClient(app_module.app) as test_client:
        response = test_client.post("/api/chat", json={"message": "What is Acne?"})
    assert response.status_code == 429
    assert "rate limit" in response.json()["detail"].lower()


def test_chat_reuses_session_id(client):
    first = client.post("/api/chat", json={"message": "Hello"}).json()
    second = client.post(
        "/api/chat", json={"message": "Follow up", "session_id": first["session_id"]}
    ).json()
    assert second["session_id"] == first["session_id"]
