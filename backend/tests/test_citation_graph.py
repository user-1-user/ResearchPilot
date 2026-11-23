import importlib
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def graph_main(tmp_path, monkeypatch):
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
        yield main
    finally:
        sys.modules.pop("main", None)
        if old_main is not None:
            sys.modules["main"] = old_main


def _insert_papers(main, papers):
    with main.db() as conn:
        conn.execute(
            """
            INSERT INTO users
            (id, username, password_salt, password_hash, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("graph-test-user", "graph-test-user", "salt", "hash", "2026-01-01"),
        )

        for paper in papers:
            conn.execute(
                """
                INSERT INTO papers
                (id, user_id, filename, file_path, sha256,
                 full_text, page_count, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper["id"],
                    "graph-test-user",
                    paper["filename"],
                    paper["filename"],
                    paper["id"] + "-sha",
                    paper.get("full_text", ""),
                    1,
                    json.dumps(paper.get("metadata", {})),
                    "2026-01-01",
                ),
            )


def test_graph_matches_reference_doi(graph_main):
    main = graph_main
    _insert_papers(main, [
        {
            "id": "source",
            "filename": "source.pdf",
            "metadata": {
                "title": "Source Paper",
                "reference_dois": ["10.1000/target.paper"],
            },
        },
        {
            "id": "target",
            "filename": "target.pdf",
            "metadata": {
                "title": "Target Paper",
                "doi": "10.1000/target.paper",
            },
        },
    ])

    result = main._build_graph_data("graph-test-user")
    assert any(
        edge["source"] == "source" and edge["target"] == "target"
        for edge in result["edges"]
    )


def test_graph_matches_reference_title(graph_main):
    main = graph_main
    _insert_papers(main, [
        {
            "id": "source",
            "filename": "source.pdf",
            "metadata": {
                "title": "Source Paper",
                "references_preview": [
                    "[1] A Detailed Study of Transformer Retrieval Architectures"
                ],
            },
        },
        {
            "id": "target",
            "filename": "target.pdf",
            "metadata": {
                "title": "A Detailed Study of Transformer Retrieval Architectures",
            },
        },
    ])

    result = main._build_graph_data("graph-test-user")
    assert any(
        edge["source"] == "source" and edge["target"] == "target"
        for edge in result["edges"]
    )


def test_graph_ignores_doi_mentioned_only_in_body(graph_main):
    main = graph_main
    _insert_papers(main, [
        {
            "id": "source",
            "filename": "source.pdf",
            "full_text": "The introduction mentions DOI 10.1000/target.paper.",
            "metadata": {"title": "Source Paper"},
        },
        {
            "id": "target",
            "filename": "target.pdf",
            "metadata": {
                "title": "Target Paper",
                "doi": "10.1000/target.paper",
            },
        },
    ])

    result = main._build_graph_data("graph-test-user")
    assert not result["edges"]


def test_graph_ignores_title_mentioned_only_in_body(graph_main):
    main = graph_main
    _insert_papers(main, [
        {
            "id": "source",
            "filename": "source.pdf",
            "full_text": (
                "This introduction discusses "
                "A Detailed Study of Transformer Retrieval Architectures."
            ),
            "metadata": {"title": "Source Paper"},
        },
        {
            "id": "target",
            "filename": "target.pdf",
            "metadata": {
                "title": "A Detailed Study of Transformer Retrieval Architectures",
            },
        },
    ])

    result = main._build_graph_data("graph-test-user")
    assert not result["edges"]


def test_graph_matches_title_split_across_multiple_lines(graph_main):
    main = graph_main
    _insert_papers(main, [
        {
            "id": "source",
            "filename": "source.pdf",
            "metadata": {
                "title": "Source Paper",
                "references_preview": [
                    "Smith, J. A Detailed Study of Transformer Retrieval Systems for",
                    "language models",
                    "across scientific literature. Journal, 2020.",
                ],
            },
        },
        {
            "id": "target",
            "filename": "target.pdf",
            "metadata": {
                "title": "A Detailed Study of Transformer Retrieval Systems for Language Models",
            },
        },
    ])

    result = main._build_graph_data("graph-test-user")
    assert any(
        edge["source"] == "source" and edge["target"] == "target"
        for edge in result["edges"]
    )


def test_graph_does_not_join_separate_reference_entries(graph_main):
    main = graph_main
    _insert_papers(main, [
        {
            "id": "source",
            "filename": "source.pdf",
            "metadata": {
                "title": "Source Paper",
                "references_preview": [
                    "A Detailed Survey Discusses",
                    "Smith, J. Study of Transformer Retrieval Architectures.",
                ],
            },
        },
        {
            "id": "target",
            "filename": "target.pdf",
            "metadata": {
                "title": (
                    "A Detailed Survey Discusses Smith Study of "
                    "Transformer Retrieval Architectures"
                ),
            },
        },
    ])

    result = main._build_graph_data("graph-test-user")
    assert not any(
        edge["source"] == "source" and edge["target"] == "target"
        for edge in result["edges"]
    )
