from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def researchpilot(tmp_path_factory):
    backend_dir = str(Path(__file__).resolve().parents[1])
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    env_keys = [
        "PYTHON_DOTENV_DISABLED",
        "LLM_API_KEY",
        "DEEPSEEK_API_KEY",
        "DATA_DIR",
        "DB_PATH",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    old_main = sys.modules.pop("main", None)

    try:
        temp_dir = tmp_path_factory.mktemp("futurework-page")
        os.environ["PYTHON_DOTENV_DISABLED"] = "1"
        os.environ.pop("LLM_API_KEY", None)
        os.environ.pop("DEEPSEEK_API_KEY", None)
        os.environ["DATA_DIR"] = str(temp_dir / "data")
        os.environ.pop("DB_PATH", None)

        main = importlib.import_module("main")
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


def test_each_quote_matches_its_own_page(researchpilot):
    pages = [
        {"page_no": 1, "text": "Introduction and background."},
        {"page_no": 2, "text": "Future work includes improving positional encoding."},
        {"page_no": 3, "text": "We plan to extend the Transformer architecture."},
    ]

    assert researchpilot._find_future_work_page(
        "improving positional encoding", pages
    ) == 2

    assert researchpilot._find_future_work_page(
        "extend the Transformer architecture", pages
    ) == 3


def test_quote_matches_across_line_breaks_and_spaces(researchpilot):
    pages = [
        {"page_no": 4, "text": "Future work includes improving\npositional   encoding methods."},
    ]

    assert researchpilot._find_future_work_page(
        "improving positional encoding", pages
    ) == 4


def test_unmatched_quote_returns_none(researchpilot):
    pages = [
        {"page_no": 1, "text": "Introduction."},
        {"page_no": 2, "text": "Experimental results."},
    ]

    assert researchpilot._find_future_work_page(
        "A completely unrelated sentence.", pages
    ) is None


def test_empty_quote_returns_none(researchpilot):
    assert researchpilot._find_future_work_page(
        "   ", [{"page_no": 1, "text": "Some text."}]
    ) is None


def test_unicode_ligatures_match_normal_text(researchpilot):
    pages = [
        {"page_no": 5, "text": "The result is reﬂecting a sufﬁcient improvement."},
    ]

    assert researchpilot._find_future_work_page(
        "reflecting a sufficient improvement", pages
    ) == 5


def test_hyphenated_line_break_matches_complete_word(researchpilot):
    pages = [
        {
            "page_no": 6,
            "text": "The homoge-\nneous state limits the archi-\ntectures.",
        },
    ]

    assert researchpilot._find_future_work_page(
        "homogeneous state limits the architectures", pages
    ) == 6


def test_soft_hyphen_is_ignored(researchpilot):
    pages = [
        {"page_no": 7, "text": "This is a suffi\u00adcient result."},
    ]

    assert researchpilot._find_future_work_page(
        "This is a sufficient result", pages
    ) == 7


def test_ligature_and_line_break_combined(researchpilot):
    pages = [
        {
            "page_no": 8,
            "text": "The reﬂecting behavior of the homoge-\nneous system.",
        },
    ]

    assert researchpilot._find_future_work_page(
        "The reflecting behavior of the homogeneous system", pages
    ) == 8