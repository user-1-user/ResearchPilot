import importlib
import os
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException


@pytest.fixture(scope="module")
def researchpilot(tmp_path_factory):
    """???????????????????"""

    backend_dir = str(Path(__file__).resolve().parents[1])
    added_to_path = backend_dir not in sys.path

    if added_to_path:
        sys.path.insert(0, backend_dir)

    env_keys = [
        "PYTHON_DOTENV_DISABLED",
        "LLM_API_KEY",
        "DEEPSEEK_API_KEY",
        "DATA_DIR",
        "DB_PATH",
    ]

    old_env = {
        key: os.environ.get(key)
        for key in env_keys
    }

    old_main = sys.modules.pop("main", None)

    try:
        temp_dir = tmp_path_factory.mktemp("agent-validation")

        os.environ["PYTHON_DOTENV_DISABLED"] = "1"
        os.environ.pop("LLM_API_KEY", None)
        os.environ.pop("DEEPSEEK_API_KEY", None)
        os.environ["DATA_DIR"] = str(temp_dir / "data")
        os.environ.pop("DB_PATH", None)

        main = importlib.import_module("main")
        main.init_db()

        # ????????
        with main.db() as conn:
            for uid, username in [
                ("user-a", "test-user-a"),
                ("user-b", "test-user-b"),
            ]:
                conn.execute(
                    """
                    INSERT INTO users(
                        id, username, password_salt,
                        password_hash, created_at
                    )
                    VALUES(?,?,?,?,?)
                    """,
                    (
                        uid,
                        username,
                        "00",
                        "00",
                        main._iso(main._now()),
                    ),
                )

            # ???? user-b ?????
            conn.execute(
                """
                INSERT INTO papers(
                    id, user_id, filename, file_path,
                    sha256, full_text, page_count,
                    metadata_json, created_at
                )
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    "foreign-paper",
                    "user-b",
                    "private.pdf",
                    "private.pdf",
                    "test-sha256",
                    "test content",
                    1,
                    "{}",
                    main._iso(main._now()),
                ),
            )

        yield main

    finally:
        sys.modules.pop("main", None)

        if old_main is not None:
            sys.modules["main"] = old_main

        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        if added_to_path and backend_dir in sys.path:
            sys.path.remove(backend_dir)


def test_valid_tool_arguments(researchpilot):
    """???????????"""

    result = researchpilot._run_tool(
        "user-a",
        "library_stats",
        {},
    )

    assert result["paper_count"] == 0


@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected_status"),
    [
        pytest.param(
            "search_library",
            {"query": "   "},
            422,
            id="empty-search-query",
        ),
        pytest.param(
            "search_library",
            {
                "query": "Transformer attention",
                "unexpected": "value",
            },
            422,
            id="unexpected-extra-argument",
        ),
        pytest.param(
            "validate_research_gap",
            {"idea": "AI"},
            422,
            id="short-research-idea",
        ),
        pytest.param(
            "compare_library",
            {
                "query": "Transformer attention",
                "paper_ids": [
                    f"paper-{i}" for i in range(9)
                ],
            },
            422,
            id="too-many-paper-ids",
        ),
        pytest.param(
            "download_arxiv_paper",
            {"arxiv_id": "not-a-valid-id"},
            422,
            id="invalid-arxiv-id",
        ),
        pytest.param(
            "search_library",
            ["Transformer"],
            422,
            id="non-object-arguments",
        ),
        pytest.param(
            "unknown_tool",
            {},
            400,
            id="unknown-tool",
        ),
        pytest.param(
            "compare_library",
            {
                "query": "Transformer attention",
                "paper_ids": ["foreign-paper"],
            },
            404,
            id="cross-user-paper-access",
        ),
    ],
)
def test_invalid_tool_arguments(
    researchpilot,
    tool_name,
    arguments,
    expected_status,
):
    """????????????"""

    with pytest.raises(HTTPException) as exc_info:
        researchpilot._run_tool(
            "user-a",
            tool_name,
            arguments,
        )

    assert exc_info.value.status_code == expected_status
