import importlib
import sys
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_app(tmp_path, monkeypatch):
    """使用独立临时数据库，不触碰正式论文库。"""
    backend_dir = str(Path(__file__).resolve().parents[1])
    monkeypatch.syspath_prepend(backend_dir)

    old_main = sys.modules.pop("main", None)

    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv(
        "DB_PATH",
        str(tmp_path / "data" / "researchpilot.sqlite3"),
    )

    # 测试不使用外部 LLM 或 embedding 服务。
    for key in (
        "LLM_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "EMBEDDING_API_KEY",
        "LOCAL_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)

    try:
        importlib.invalidate_caches()
        main = importlib.import_module("main")
        main.init_db()
        yield main
    finally:
        sys.modules.pop("main", None)
        if old_main is not None:
            sys.modules["main"] = old_main


def make_pdf(path: Path, content: str):
    doc = fitz.open()
    try:
        page = doc.new_page()
        page.insert_text((72, 72), content, fontsize=11)
        doc.save(path)
    finally:
        doc.close()


def upload_pdf(client, auth, tmp_path, filename, content):
    path = tmp_path / filename
    make_pdf(path, content)

    with path.open("rb") as handle:
        response = client.post(
            "/api/papers/upload",
            headers=auth,
            files={"file": (filename, handle, "application/pdf")},
        )

    assert response.status_code == 200, response.text
    return response.json()


def resolve(client, auth, paper_id, candidate_id, action):
    return client.post(
        f"/api/papers/{paper_id}/resolve-duplicate",
        headers=auth,
        json={
            "candidate_id": candidate_id,
            "action": action,
        },
    )


def test_all_duplicate_resolution_branches(
    test_app,
    tmp_path,
    monkeypatch,
):
    main = test_app

    # raise_server_exceptions=False：将故意注入的异常转换为 HTTP 500，
    # 以便检查接口失败后的数据库状态。
    with TestClient(main.app, raise_server_exceptions=False) as client:

        registration = client.post(
            "/api/auth/register",
            json={
                "username": "duplicate-test-user",
                "password": "duplicate-test-password-123",
            },
        )
        assert registration.status_code == 200, registration.text

        login = client.post(
            "/api/auth/login",
            json={
                "username": "duplicate-test-user",
                "password": "duplicate-test-password-123",
            },
        )
        assert login.status_code == 200, login.text
        auth = {"Authorization": f"Bearer {login.json()['token']}"}

        # -------------------------------------------------
        # 1. 建立原始论文
        # -------------------------------------------------
        original = upload_pdf(
            client,
            auth,
            tmp_path,
            "1706.03762v7.pdf",
            "Attention Is All You Need. The Transformer architecture uses attention. Future work will explore longer context windows.",
        )
        original_id = original["paper_id"]

        original_read = client.get(
            f"/api/papers/{original_id}",
            headers=auth,
        )
        assert original_read.status_code == 200

        # -------------------------------------------------
        # 2. keep_both：两个版本都保留
        # -------------------------------------------------
        both = upload_pdf(
            client,
            auth,
            tmp_path,
            "1706.03762v1.pdf",
            "Attention Is All You Need. The Transformer architecture uses attention. Future work will explore longer context windows. Revised version: residual connections and multilingual models.",
        )
        both_id = both["paper_id"]

        assert both.get("pending_duplicate") is True
        assert both.get("chunk_count") == 0

        both_result = resolve(
            client,
            auth,
            both_id,
            original_id,
            "keep_both",
        )
        assert both_result.status_code == 200, both_result.text

        assert client.get(
            f"/api/papers/{original_id}",
            headers=auth,
        ).status_code == 200

        assert client.get(
            f"/api/papers/{both_id}",
            headers=auth,
        ).status_code == 200

        # -------------------------------------------------
        # 3. keep_new：新版本建立索引成功后才删除旧版本
        # -------------------------------------------------
        new_version = upload_pdf(
            client,
            auth,
            tmp_path,
            "1706.03762v2.pdf",
            "Attention Is All You Need. The Transformer architecture uses attention. Future work will explore longer context windows. Version three: new experimental results and analysis.",
        )
        new_id = new_version["paper_id"]

        assert new_version.get("pending_duplicate") is True

        # 记录旧版本 PDF 的实际路径。
        with main.db() as conn:
            row = conn.execute(
                "SELECT file_path FROM papers WHERE id=?",
                (original_id,),
            ).fetchone()
        assert row is not None
        old_pdf_path = Path(row["file_path"])

        keep_new_result = resolve(
            client,
            auth,
            new_id,
            original_id,
            "keep_new",
        )
        assert keep_new_result.status_code == 200, keep_new_result.text

        assert client.get(
            f"/api/papers/{new_id}",
            headers=auth,
        ).status_code == 200

        assert client.get(
            f"/api/papers/{original_id}",
            headers=auth,
        ).status_code == 404

        assert not old_pdf_path.exists()

        # keep_both 版本仍然存在。
        assert client.get(
            f"/api/papers/{both_id}",
            headers=auth,
        ).status_code == 200

        # -------------------------------------------------
        # 4. keep_existing：保留旧版本，不执行索引
        # -------------------------------------------------
        rejected = upload_pdf(
            client,
            auth,
            tmp_path,
            "1706.03762v3.pdf",
            "Attention Is All You Need. The Transformer architecture uses attention. Future work will explore longer context windows. Version four: this version will be rejected.",
        )
        rejected_id = rejected["paper_id"]

        assert rejected.get("pending_duplicate") is True

        with main.db() as conn:
            row = conn.execute(
                "SELECT file_path FROM papers WHERE id=?",
                (rejected_id,),
            ).fetchone()
        assert row is not None
        rejected_pdf_path = Path(row["file_path"])

        # 监控向量准备函数，确认 keep_existing 不触发索引。
        original_prepare = main._prepare_chunk_vectors
        prepare_calls = []

        def tracked_prepare(chunks):
            prepare_calls.append(len(chunks))
            return original_prepare(chunks)

        monkeypatch.setattr(
            main,
            "_prepare_chunk_vectors",
            tracked_prepare,
        )

        calls_before = len(prepare_calls)

        keep_existing_result = resolve(
            client,
            auth,
            rejected_id,
            new_id,
            "keep_existing",
        )
        assert keep_existing_result.status_code == 200, (
            keep_existing_result.text
        )

        assert len(prepare_calls) == calls_before, (
            "keep_existing 不应该调用向量准备函数"
        )

        assert client.get(
            f"/api/papers/{new_id}",
            headers=auth,
        ).status_code == 200

        assert client.get(
            f"/api/papers/{rejected_id}",
            headers=auth,
        ).status_code == 404

        assert not rejected_pdf_path.exists()

        # -------------------------------------------------
        # 5. 最终化失败：旧版本和待确认新版本都必须保留
        # -------------------------------------------------
        failed = upload_pdf(
            client,
            auth,
            tmp_path,
            "1706.03762v4.pdf",
            "Attention Is All You Need. The Transformer architecture uses attention. Future work will explore longer context windows. Version five: this upload tests finalization rollback.",
        )
        failed_id = failed["paper_id"]

        assert failed.get("pending_duplicate") is True

        with main.db() as conn:
            row = conn.execute(
                "SELECT file_path FROM papers WHERE id=?",
                (failed_id,),
            ).fetchone()
        assert row is not None
        failed_pdf_path = Path(row["file_path"])

        def fail_index(*args, **kwargs):
            raise RuntimeError("TEST_FINALIZE_INDEX_FAILURE")

        # 注入索引写入故障，而不是 embedding 故障。
        # embedding 异常在当前实现中会降级为 BM25，不会中止最终化。
        monkeypatch.setattr(main, "_index_chunks", fail_index)

        failed_result = resolve(
            client,
            auth,
            failed_id,
            new_id,
            "keep_new",
        )

        assert failed_result.status_code == 500, failed_result.text

        # 原版本不得被删除。
        assert client.get(
            f"/api/papers/{new_id}",
            headers=auth,
        ).status_code == 200

        # 失败的新版本仍然存在，但保持待确认状态。
        assert client.get(
            f"/api/papers/{failed_id}",
            headers=auth,
        ).status_code == 409

        assert failed_pdf_path.exists()

        with main.db() as conn:
            pending_row = conn.execute(
                "SELECT metadata_json FROM papers WHERE id=?",
                (failed_id,),
            ).fetchone()

            page_count = conn.execute(
                "SELECT COUNT(*) FROM pages WHERE paper_id=?",
                (failed_id,),
            ).fetchone()[0]

            chunk_count = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE paper_id=?",
                (failed_id,),
            ).fetchone()[0]

        assert pending_row is not None
        import json
        metadata = json.loads(pending_row["metadata_json"] or "{}")
        assert metadata.get("index_status") == "pending_duplicate"
        assert page_count > 0
        assert chunk_count == 0

        print("Deferred duplicate resolution tests: PASS")
        print("Validated: keep_both, keep_new, keep_existing, finalization rollback.")

def test_pending_paper_residual_chunks_are_excluded_from_retrieval(
    test_app,
    tmp_path,
):
    """?? pending ?????? chunks?????????"""
    main = test_app

    with TestClient(main.app, raise_server_exceptions=False) as client:
        registration = client.post(
            "/api/auth/register",
            json={
                "username": "pending-filter-test-user",
                "password": "pending-filter-test-password-123",
            },
        )
        assert registration.status_code == 200, registration.text

        login = client.post(
            "/api/auth/login",
            json={
                "username": "pending-filter-test-user",
                "password": "pending-filter-test-password-123",
            },
        )
        assert login.status_code == 200, login.text
        auth = {"Authorization": f"Bearer {login.json()['token']}"}

        original = upload_pdf(
            client,
            auth,
            tmp_path,
            "1706.03762v7.pdf",
            "Attention Is All You Need. The Transformer architecture uses attention. Future work will explore longer context windows.",
        )
        original_id = original["paper_id"]

        pending = upload_pdf(
            client,
            auth,
            tmp_path,
            "1706.03762v1.pdf",
            "Attention Is All You Need. The Transformer architecture uses attention. Future work will explore longer context windows. Revised version adds residual connections and multilingual models.",
        )
        pending_id = pending["paper_id"]

        assert pending.get("pending_duplicate") is True
        assert pending.get("chunk_count") == 0

        # ??????????????? ID?
        with main.db() as conn:
            user_row = conn.execute(
                "SELECT id FROM users WHERE username=?",
                ("pending-filter-test-user",),
            ).fetchone()
            assert user_row is not None
            user_id = user_row["id"]

            # ???????? pending ???????? chunk?
            conn.execute(
                """
                INSERT INTO chunks
                    (id, paper_id, page_no, section, text,
                     embedding_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    "residual-" + pending_id,
                    pending_id,
                    1,
                    "TEST_RESIDUAL",
                    "THIS PENDING PAPER MUST NEVER APPEAR IN RETRIEVAL",
                    None,
                ),
            )

        # ????????????????????????
        with main.db() as conn:
            residual_count = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE paper_id=?",
                (pending_id,),
            ).fetchone()[0]
        assert residual_count == 1

        # ?????? chunk ???????? pending ???
        loaded = main._load_user_chunks(user_id, [pending_id])
        assert loaded == [], (
            "?????_load_user_chunks ??? pending ????? chunk"
        )

        # ??????????????? pending ???
        results = main.retrieve(
            user_id,
            "Transformer architecture attention",
            paper_ids=[pending_id],
        )
        assert results == [], (
            "?????retrieve ??? pending ???????"
        )

        # ?????????????????????????????
        original_chunks = main._load_user_chunks(user_id, [original_id])
        assert original_chunks, (
            "???????????????????"
        )

        print("Pending retrieval isolation test: PASS")
        print("Validated: residual chunks are excluded; active papers remain searchable.")



def test_distinct_papers_are_not_marked_as_duplicates(
    test_app,
    tmp_path,
):
    """Two unrelated papers must not be classified as duplicates."""
    main = test_app

    with TestClient(main.app) as client:
        registration = client.post(
            "/api/auth/register",
            json={
                "username": "distinct-dedup-test-user",
                "password": "distinct-dedup-test-password-123",
            },
        )
        assert registration.status_code == 200, registration.text

        login = client.post(
            "/api/auth/login",
            json={
                "username": "distinct-dedup-test-user",
                "password": "distinct-dedup-test-password-123",
            },
        )
        assert login.status_code == 200, login.text
        auth = {"Authorization": f"Bearer {login.json()['token']}"}

        first = upload_pdf(
            client,
            auth,
            tmp_path,
            "spectroscopy-paper.pdf",
            (
                "Spectroscopy photons wavelength absorption fluorescence "
                "quantum yield chromophore molecular orbital emission "
                "detector calibration."
            ),
        )
        assert first.get("duplicates", []) == []
        assert not first.get("pending_duplicate", False)

        second = upload_pdf(
            client,
            auth,
            tmp_path,
            "botanical-paper.pdf",
            (
                "Botanical taxonomy chlorophyll stomata xylem phloem "
                "germination mycorrhiza pollination seed dispersal "
                "rhizosphere."
            ),
        )

        assert second.get("duplicates", []) == [], (
            f"Unrelated paper was marked as duplicate: {second.get('duplicates')}"
        )
        assert not second.get("pending_duplicate", False), (
            "Unrelated paper was incorrectly deferred"
        )

        with main.db() as conn:
            paper_count = conn.execute(
                "SELECT COUNT(*) FROM papers"
            ).fetchone()[0]

        assert paper_count == 2




def _delete_test_auth(client, username):
    registration = client.post(
        "/api/auth/register",
        json={
            "username": username,
            "password": "delete-test-password-123",
        },
    )
    assert registration.status_code == 200, registration.text

    login = client.post(
        "/api/auth/login",
        json={
            "username": username,
            "password": "delete-test-password-123",
        },
    )
    assert login.status_code == 200, login.text
    return {"Authorization": f"Bearer {login.json()['token']}"}


def test_normal_paper_delete_cleans_database_and_pdf(
    test_app, tmp_path
):
    main = test_app

    with TestClient(main.app, raise_server_exceptions=False) as client:
        auth = _delete_test_auth(client, "paper-delete-owner")

        uploaded = upload_pdf(
            client,
            auth,
            tmp_path,
            "delete-acceptance.pdf",
            (
                "This is a paper deletion acceptance test. "
                "It contains enough meaningful text to create "
                "a searchable chunk and verify cascade deletion. "
            ) * 5,
        )
        paper_id = uploaded["paper_id"]

        with main.db() as conn:
            paper = conn.execute(
                "SELECT file_path FROM papers WHERE id=?",
                (paper_id,),
            ).fetchone()
            assert paper is not None

            pdf_path = Path(paper["file_path"])
            page_count = conn.execute(
                "SELECT COUNT(*) FROM pages WHERE paper_id=?",
                (paper_id,),
            ).fetchone()[0]
            chunk_count = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE paper_id=?",
                (paper_id,),
            ).fetchone()[0]

        assert pdf_path.is_file()
        assert page_count > 0
        assert chunk_count > 0

        response = client.delete(
            f"/api/papers/{paper_id}",
            headers=auth,
        )
        assert response.status_code == 200, response.text

        assert client.get(
            f"/api/papers/{paper_id}",
            headers=auth,
        ).status_code == 404

        with main.db() as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM papers WHERE id=?",
                (paper_id,),
            ).fetchone()[0] == 0
            assert conn.execute(
                "SELECT COUNT(*) FROM pages WHERE paper_id=?",
                (paper_id,),
            ).fetchone()[0] == 0
            assert conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE paper_id=?",
                (paper_id,),
            ).fetchone()[0] == 0

        assert not pdf_path.exists()


def test_paper_delete_rejects_other_user_and_missing_paper(
    test_app, tmp_path
):
    main = test_app

    with TestClient(main.app, raise_server_exceptions=False) as client:
        owner_auth = _delete_test_auth(client, "paper-delete-owner-2")
        other_auth = _delete_test_auth(client, "paper-delete-other")

        uploaded = upload_pdf(
            client,
            owner_auth,
            tmp_path,
            "owner-paper.pdf",
            (
                "This paper belongs to the first user. "
                "It must remain available when another user "
                "attempts to delete it. "
            ) * 5,
        )
        paper_id = uploaded["paper_id"]

        with main.db() as conn:
            paper = conn.execute(
                "SELECT file_path FROM papers WHERE id=?",
                (paper_id,),
            ).fetchone()
        pdf_path = Path(paper["file_path"])
        assert pdf_path.is_file()

        forbidden = client.delete(
            f"/api/papers/{paper_id}",
            headers=other_auth,
        )
        assert forbidden.status_code == 404

        assert pdf_path.is_file()
        assert client.get(
            f"/api/papers/{paper_id}",
            headers=owner_auth,
        ).status_code == 200

        missing = client.delete(
            "/api/papers/paper-that-does-not-exist",
            headers=owner_auth,
        )
        assert missing.status_code == 404

        # ?????????????
        cleanup = client.delete(
            f"/api/papers/{paper_id}",
            headers=owner_auth,
        )
        assert cleanup.status_code == 200, cleanup.text
        assert not pdf_path.exists()




def test_paper_delete_pdf_unlink_failure_diagnostic(
    test_app, tmp_path, monkeypatch
):
    main = test_app

    with TestClient(main.app, raise_server_exceptions=False) as client:
        auth = _delete_test_auth(client, "paper-delete-unlink-failure")

        uploaded = upload_pdf(
            client,
            auth,
            tmp_path,
            "unlink-failure.pdf",
            (
                "This document tests what happens when deleting "
                "the stored PDF file fails. "
            ) * 5,
        )
        paper_id = uploaded["paper_id"]

        with main.db() as conn:
            paper = conn.execute(
                "SELECT file_path FROM papers WHERE id=?",
                (paper_id,),
            ).fetchone()

        pdf_path = Path(paper["file_path"])
        assert pdf_path.is_file()

        original_unlink = Path.unlink

        def failing_unlink(path, *args, **kwargs):
            if path == pdf_path:
                raise PermissionError("Simulated PDF deletion failure")
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", failing_unlink)

        response = client.delete(
            f"/api/papers/{paper_id}",
            headers=auth,
        )

        # ??????????????????????????500?
        assert response.status_code == 200

        with main.db() as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM papers WHERE id=?",
                (paper_id,),
            ).fetchone()[0] == 0
            assert conn.execute(
                "SELECT COUNT(*) FROM pages WHERE paper_id=?",
                (paper_id,),
            ).fetchone()[0] == 0
            assert conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE paper_id=?",
                (paper_id,),
            ).fetchone()[0] == 0

        assert pdf_path.is_file()

        # ?????????????????? PDF?
        monkeypatch.undo()
        pdf_path.unlink(missing_ok=True)
