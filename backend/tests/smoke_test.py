"""Run: python tests/smoke_test.py (from backend/).
Uses a temporary SQLite DB and generated PDFs; no API key or network is required.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import fitz
from fastapi.testclient import TestClient


with tempfile.TemporaryDirectory(prefix="researchpilot-test-") as temp_dir:
    os.environ["DATA_DIR"] = str(Path(temp_dir) / "data")
    os.environ["DB_PATH"] = str(Path(temp_dir) / "data" / "researchpilot.sqlite3")
    os.environ.pop("LLM_API_KEY", None)
    os.environ.pop("DEEPSEEK_API_KEY", None)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    # Permit local smoke tests in minimal environments that have not installed
    # the OpenAI SDK; this stub is never used for LLM-backed test cases.
    try:
        import openai  # noqa: F401
    except ImportError:
        import types
        openai_stub = types.ModuleType("openai")
        class OpenAI:  # noqa: N801
            def __init__(self, *args, **kwargs):
                self.chat = None
        openai_stub.OpenAI = OpenAI
        sys.modules["openai"] = openai_stub

    import main  # noqa: E402  (environment must be set before import)

    def make_pdf(path: Path, body: str) -> None:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), body, fontsize=10)
        doc.save(path)
        doc.close()

    with TestClient(main.app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200 and health.json()["status"] == "ok"

        registration = client.post("/api/auth/register", json={"username": "demo-user", "password": "demo-password-123"})
        assert registration.status_code == 200, registration.text
        login = client.post("/api/auth/login", json={"username": "demo-user", "password": "demo-password-123"})
        assert login.status_code == 200, login.text
        auth = {"Authorization": f"Bearer {login.json()['token']}"}
        second_registration = client.post("/api/auth/register", json={"username": "other-user", "password": "another-password-123"})
        assert second_registration.status_code == 200
        second_login = client.post("/api/auth/login", json={"username": "other-user", "password": "another-password-123"})
        other_auth = {"Authorization": f"Bearer {second_login.json()['token']}"}

        p1 = Path(temp_dir) / "1706.03762v7.pdf"
        p2 = Path(temp_dir) / "1706.03762v1.pdf"
        make_pdf(p1, "Attention Is All You Need. The Transformer architecture uses attention. Future work will explore longer context windows.")
        make_pdf(p2, "Attention Is All You Need. The Transformer architecture uses attention and residual connections. Future work will explore multilingual models.")

        with p1.open("rb") as handle:
            up1 = client.post("/api/papers/upload", headers=auth, files={"file": (p1.name, handle, "application/pdf")})
        assert up1.status_code == 200, up1.text
        paper_id = up1.json()["paper_id"]

        with p2.open("rb") as handle:
            up2 = client.post("/api/papers/upload", headers=auth, files={"file": (p2.name, handle, "application/pdf")})
        assert up2.status_code == 200, up2.text
        paper2_id = up2.json()["paper_id"]
        assert up2.json()["duplicates"], "same arXiv ID should be flagged as a potential version duplicate"

        # ???????????????????????????
        assert up2.json().get("pending_duplicate") is True
        assert up2.json().get("chunk_count") == 0

        papers_pending = client.get("/api/papers", headers=auth)
        assert papers_pending.status_code == 200
        assert len(papers_pending.json()) == 1, "pending duplicate must be hidden from the library"

        pending_read = client.get(f"/api/papers/{paper2_id}", headers=auth)
        assert pending_read.status_code == 409, "pending duplicate must not be readable before resolution"

        # ??????????????????????
        resolution = client.post(
            f"/api/papers/{paper2_id}/resolve-duplicate",
            headers=auth,
            json={"candidate_id": paper_id, "action": "keep_both"},
        )
        assert resolution.status_code == 200, resolution.text

        papers = client.get("/api/papers", headers=auth)
        assert papers.status_code == 200 and len(papers.json()) == 2

        # ????????????????
        finalized_read = client.get(f"/api/papers/{paper2_id}", headers=auth)
        assert finalized_read.status_code == 200, finalized_read.text

        paged = client.get(f"/api/papers/{paper_id}?start_page=1&page_size=1", headers=auth)
        assert paged.status_code == 200 and len(paged.json()["pages"]) == 1
        cross_user = client.get(f"/api/papers/{paper_id}", headers=other_auth)
        assert cross_user.status_code == 404, "a different user must not be able to read this paper"

        search = client.post("/api/qa/search", headers=auth, json={"question": "Transformer attention", "paper_ids": [paper_id]})
        assert search.status_code == 200 and search.json()["sources"]
        abstain = client.post("/api/qa/search", headers=auth, json={"question": "antidisestablishmentarianism xyzzy", "paper_ids": [paper_id]})
        assert abstain.status_code == 200 and abstain.json()["sources"] == []

        qa_stream = client.post("/api/qa/stream", headers=auth, json={"paper_id": paper_id, "question": "What is the core method?"})
        assert qa_stream.status_code == 200 and "data:" in qa_stream.text

        compare = client.post("/api/qa/compare", headers=auth, json={"paper_ids": [paper_id, paper2_id], "question": "Compare their future work."})
        assert compare.status_code == 200 and "data:" in compare.text

        graph = client.post("/api/graph/build", headers=auth)
        assert graph.status_code == 200 and len(graph.json()["nodes"]) == 2
        future = client.post("/api/futurework", headers=auth)
        assert future.status_code == 200
        review = client.post("/api/research/review", headers=auth, json={"topic": "Transformer attention"})
        assert review.status_code == 200 and review.json()["sources"]

        agent = client.post("/api/agent/chat", headers=auth, json={"message": "帮我查一下库里有多少篇论文？"})
        assert agent.status_code == 200 and '"type": "action"' in agent.text and '"type": "text"' in agent.text
        agent_review = client.post("/api/agent/chat", headers=auth, json={"message": "请生成一份 Transformer 的文献综述"})
        assert agent_review.status_code == 200 and "generate_literature_review" in agent_review.text

        print("ResearchPilot smoke test: PASS")
        print("Validated: health, auth, PDF upload, duplicate detection, pagination, retrieval, QA SSE, compare SSE, graph, Future Work, literature review, Agent tool call.")
