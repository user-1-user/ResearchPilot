import importlib
import sys
from pathlib import Path

import fitz
import pytest


@pytest.fixture
def pdf_main(tmp_path, monkeypatch):
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
        yield main
    finally:
        sys.modules.pop("main", None)
        if old_main is not None:
            sys.modules["main"] = old_main


def test_pdf_extraction_preserves_metadata(pdf_main, tmp_path):
    main = pdf_main
    pdf_path = tmp_path / "2401.12345v2.pdf"

    doc = fitz.open()
    try:
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            "ResearchPilot PDF Extraction Regression Test",
            fontsize=14,
        )
        page.insert_text(
            (72, 105),
            "DOI: 10.1000/main.paper123",
            fontsize=11,
        )
        page.insert_text(
            (72, 135),
            "This paper studies transformer models and retrieval systems. "
            "The experiments evaluate model quality and retrieval accuracy "
            "across multiple datasets.",
            fontsize=10,
        )
        page.insert_text(
            (72, 190),
            "Figure 1: Overview of the proposed retrieval architecture.",
            fontsize=10,
        )
        page.insert_text(
            (72, 250),
            "References",
            fontsize=12,
        )
        page.insert_text(
            (72, 275),
            "[1] Example Author. A study of retrieval systems. "
            "Journal of Testing, 2024. DOI: 10.1000/reference.paper456",
            fontsize=10,
        )
        doc.set_metadata({"title": "ResearchPilot Extraction Test"})
        doc.save(pdf_path)
    finally:
        doc.close()

    pages, metadata = main._extract_pdf(
        pdf_path,
        original_filename=pdf_path.name,
    )

    assert len(pages) == 1
    assert "transformer models" in pages[0]["text"].lower()

    assert metadata["title"] == "ResearchPilot Extraction Test"
    assert metadata["doi"].lower() == "10.1000/main.paper123"
    assert metadata["arxiv_id"] == "2401.12345"
    assert metadata["arxiv_version"] == 2

    assert metadata["reference_dois"] == [
        "10.1000/reference.paper456"
    ]
    assert metadata["references_preview"]
    assert any(
        "Example Author" in item
        for item in metadata["references_preview"]
    )
    assert any(
        "Figure 1:" in item
        for item in metadata["figure_captions"]
    )
    assert metadata["ocr_pages"] == []

def test_pdf_extraction_preserves_cross_page_references(pdf_main, tmp_path):
    main = pdf_main
    pdf_path = tmp_path / "2402.12345v1.pdf"

    doc = fitz.open()
    try:
        page1 = doc.new_page()
        page1.insert_text(
            (72, 72),
            "Cross Page Reference Regression Test",
            fontsize=14,
        )
        page1.insert_text(
            (72, 110),
            "DOI: 10.1000/cross.page123",
            fontsize=11,
        )
        page1.insert_text(
            (72, 250),
            "References",
            fontsize=12,
        )

        page2 = doc.new_page()
        page2.insert_text(
            (72, 72),
            "[1] First Author. Cross-page reference example. "
            "Testing Journal, 2024. DOI: 10.1000/cross.ref456",
            fontsize=10,
        )
        page2.insert_text(
            (72, 110),
            "Figure 2: Cross-page extraction test result.",
            fontsize=10,
        )
        doc.save(pdf_path)
    finally:
        doc.close()

    pages, metadata = main._extract_pdf(
        pdf_path,
        original_filename=pdf_path.name,
    )

    assert len(pages) == 2
    assert metadata["doi"].lower() == "10.1000/cross.page123"
    assert metadata["arxiv_id"] == "2402.12345"
    assert metadata["arxiv_version"] == 1
    assert "10.1000/cross.ref456" in metadata["reference_dois"]
    assert any(
        "First Author" in item
        for item in metadata["references_preview"]
    )
    assert any(
        "Figure 2:" in item
        for item in metadata["figure_captions"]
    )

@pytest.mark.parametrize(
    "text_a,text_b,expected",
    [
        ("transformer retrieval model", "transformer retrieval model", 1.0),
        ("alpha beta gamma delta", "alpha beta gamma epsilon", 0.0),
        ("apple banana", "quantum physics", 0.0),
        ("", "transformer retrieval model", 0.0),
        ("研究论文检索系统", "研究论文检索系统", 1.0),
    ],
)
def test_shingle_similarity_preserves_results(
    pdf_main, text_a, text_b, expected
):
    actual = pdf_main._shingle_similarity(text_a, text_b)
    assert actual == pytest.approx(expected)


def test_pdf_extraction_joins_multiline_title(pdf_main, tmp_path):
    main = pdf_main
    pdf_path = tmp_path / "multiline_title.pdf"

    doc = fitz.open()
    try:
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            "BERT: Pre-training of Deep Bidirectional Transformers for",
            fontsize=14,
        )
        page.insert_text(
            (72, 92),
            "Language Understanding",
            fontsize=14,
        )
        page.insert_text(
            (72, 125),
            "Jacob Devlin",
            fontsize=12,
        )
        page.insert_text(
            (72, 155),
            "Abstract",
            fontsize=12,
        )
        doc.save(pdf_path)
    finally:
        doc.close()

    _, metadata = main._extract_pdf(
        pdf_path,
        original_filename=pdf_path.name,
    )

    assert metadata["title"] == (
        "BERT: Pre-training of Deep Bidirectional Transformers for "
        "Language Understanding"
    )
