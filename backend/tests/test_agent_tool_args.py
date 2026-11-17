import unittest
from unittest.mock import patch

from fastapi import HTTPException
from main import _run_tool


class AgentToolArgsTests(unittest.TestCase):

    @patch("main._arxiv_search")
    def test_search_arxiv_rejects_unsupported_arguments(self, mock_search):
        with self.assertRaises(HTTPException) as context:
            _run_tool(
                "test-user",
                "search_arxiv",
                {
                    "query": "transformer",
                    "max_results": 20,
                },
            )

        self.assertEqual(context.exception.status_code, 422)
        mock_search.assert_not_called()

    @patch("main._generate_literature_review")
    def test_literature_review_rejects_unsupported_arguments(self, mock_review):
        with self.assertRaises(HTTPException) as context:
            _run_tool(
                "test-user",
                "generate_literature_review",
                {
                    "topic": "Transformer",
                    "include_arxiv": True,
                    "use_local_library": True,
                },
            )

        self.assertEqual(context.exception.status_code, 422)
        mock_review.assert_not_called()

    def test_required_idea_missing_still_fails(self):
        with self.assertRaises(HTTPException) as context:
            _run_tool(
                "test-user",
                "generate_framework",
                {},
            )

        self.assertEqual(context.exception.status_code, 422)

    def test_invalid_idea_type_still_fails(self):
        with self.assertRaises(HTTPException) as context:
            _run_tool(
                "test-user",
                "generate_framework",
                {"idea": ["not", "a", "string"]},
            )

        self.assertEqual(context.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()