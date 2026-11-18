import unittest
from unittest.mock import patch

from main import (
    validate_evidence_sources,
    validate_answer_citations,
    validate_answer_semantic_support,
)


def make_source(evidence_id="E1", quote="A supported statement"):
    return {
        "evidence_id": evidence_id,
        "chunk_id": "chunk-1",
        "paper_id": "paper-1",
        "filename": "paper.pdf",
        "page": 1,
        "section": "Abstract",
        "quote": quote,
        "score": 0.5,
        "rerank_score": 1.2,
    }


class EvidenceValidationTests(unittest.TestCase):
    def test_valid_source_passes(self):
        result = validate_evidence_sources([make_source()])
        self.assertTrue(result["valid"])
        self.assertEqual(result["evidence_ids"], ["E1"])

    def test_empty_sources_fail(self):
        self.assertFalse(validate_evidence_sources([])["valid"])

    def test_missing_field_fails(self):
        source = make_source()
        del source["quote"]
        result = validate_evidence_sources([source])
        self.assertFalse(result["valid"])
        self.assertTrue(result["errors"])

    def test_duplicate_ids_fail(self):
        result = validate_evidence_sources([
            make_source("E1"),
            make_source("E1"),
        ])
        self.assertFalse(result["valid"])
        self.assertTrue(any("Duplicate" in e for e in result["errors"]))

    def test_empty_quote_fails(self):
        result = validate_evidence_sources([make_source(quote="  ")])
        self.assertFalse(result["valid"])

    def test_valid_answer_citation_passes(self):
        result = validate_answer_citations(
            "The paper reports this result [E1].",
            ["E1"],
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["citation_count"], 1)

    def test_unknown_answer_citation_fails(self):
        result = validate_answer_citations(
            "The paper reports this result [E99].",
            ["E1"],
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["unknown_evidence_ids"], ["E99"])

    def test_no_citation_is_warning(self):
        result = validate_answer_citations("No citations here.", ["E1"])
        self.assertTrue(result["valid"])
        self.assertTrue(result["warnings"])


    def _semantic_check(self, verdict, evidence_ids=None):
        response = {
            "claims": [{
                "claim": "The Transformer uses positional encodings.",
                "verdict": verdict,
                "evidence_ids": evidence_ids if evidence_ids is not None else ["E1"],
                "reason": "The evidence supports this claim.",
            }]
        }
        with patch("main.llm_json", return_value=response):
            return validate_answer_semantic_support(
                "The Transformer uses positional encodings [E1].",
                [make_source(quote="The model uses positional encodings.")],
            )

    def test_semantic_supported_passes(self):
        result = self._semantic_check("supported")
        self.assertTrue(result["valid"])
        self.assertTrue(result["semantic_support_checked"])

    def test_semantic_partially_supported_warns(self):
        result = self._semantic_check("partially_supported")
        self.assertFalse(result["valid"])
        self.assertEqual(result["claims"][0]["verdict"], "partially_supported")

    def test_semantic_unsupported_warns(self):
        result = self._semantic_check("unsupported")
        self.assertFalse(result["valid"])

    def test_semantic_insufficient_evidence_warns(self):
        result = self._semantic_check("insufficient_evidence")
        self.assertFalse(result["valid"])

    def test_semantic_supported_without_evidence_fails(self):
        result = self._semantic_check("supported", [])
        self.assertFalse(result["valid"])
        self.assertEqual(
            result["claims"][0]["verdict"],
            "insufficient_evidence",
        )

    def test_semantic_unknown_evidence_id_fails(self):
        result = self._semantic_check("supported", ["E99"])
        self.assertFalse(result["valid"])
        self.assertTrue(result["errors"])
        self.assertEqual(result["claims"][0]["verdict"], "unsupported")


if __name__ == "__main__":
    unittest.main()
