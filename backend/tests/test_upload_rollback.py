import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture
def rollback_main(tmp_path, monkeypatch):
    """
    使用临时数据目录初始化独立的 ResearchPilot 环境。
    测试结束后恢复 main 模块，避免污染其他测试。
    """
    backend_dir = str(Path(__file__).resolve().parents[1])
    monkeypatch.syspath_prepend(backend_dir)

    old_main = sys.modules.pop("main", None)

    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("DB_PATH", raising=False)

    try:
        importlib.invalidate_caches()
        main = importlib.import_module("main")
        main.init_db()

        user_id = "rollback-test-user"

        with main.db() as conn:
            conn.execute(
                """
                INSERT INTO users(
                    id, username, password_salt,
                    password_hash, created_at
                )
                VALUES(?,?,?,?,?)
                """,
                (
                    user_id,
                    "rollback-test-user",
                    "test-salt",
                    "test-password-hash",
                    "2026-01-01T00:00:00+00:00",
                ),
            )

        yield main

    finally:
        sys.modules.pop("main", None)

        if old_main is not None:
            sys.modules["main"] = old_main


def test_upload_failure_rolls_back_database_and_removes_pdf(
    rollback_main,
    tmp_path,
    monkeypatch,
):
    """
    模拟 chunks 索引阶段失败，验证：
    1. 原始异常被保留；
    2. papers 数据库记录回滚；
    3. pages 数据库记录回滚；
    4. chunks 数据库记录回滚；
    5. 已保存的 PDF 文件被删除。
    """
    import fitz

    main = rollback_main
    user_id = "rollback-test-user"

    # 创建一个真实的、可读取的测试 PDF。
    pdf_path = tmp_path / "rollback-test.pdf"

    doc = fitz.open()
    try:
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            (
                "Transformer attention mechanism. "
                "This document is created for upload rollback testing. "
                "It contains searchable text and enough content "
                "to exercise the PDF extraction pipeline."
            ),
            fontsize=11,
        )
        doc.save(pdf_path)
    finally:
        doc.close()

    content = pdf_path.read_bytes()

    # 注入故障：让 chunks 索引阶段主动抛出异常。
    def fail_index(*args, **kwargs):
        raise RuntimeError("TEST_INJECTED_INDEX_FAILURE")

    monkeypatch.setattr(main, "_index_chunks", fail_index)

    # 必须保留原始异常，不能吞掉或替换异常。
    with pytest.raises(
        RuntimeError,
        match="TEST_INJECTED_INDEX_FAILURE",
    ):
        main._upload_bytes(
            {"id": user_id},
            "rollback-test.pdf",
            content,
        )

    # 检查数据库中是否残留任何上传记录。
    with main.db() as conn:
        paper_rows = conn.execute(
            "SELECT COUNT(*) FROM papers"
        ).fetchone()[0]

        page_rows = conn.execute(
            "SELECT COUNT(*) FROM pages"
        ).fetchone()[0]

        chunk_rows = conn.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()[0]

    # 检查刚才上传的 PDF 是否已经清理。
    remaining_pdfs = list(main.PAPER_DIR.glob("*.pdf"))

    assert paper_rows == 0, (
        f"数据库 papers 表残留 {paper_rows} 条记录"
    )
    assert page_rows == 0, (
        f"数据库 pages 表残留 {page_rows} 条记录"
    )
    assert chunk_rows == 0, (
        f"数据库 chunks 表残留 {chunk_rows} 条记录"
    )
    assert len(remaining_pdfs) == 0, (
        f"上传失败后仍残留 PDF 文件：{remaining_pdfs}"
    )
