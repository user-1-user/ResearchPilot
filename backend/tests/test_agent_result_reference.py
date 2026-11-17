import unittest
from fastapi import HTTPException
from unittest.mock import patch

import main


class AgentResultReferenceTests(unittest.TestCase):

    def setUp(self):
        self.observations = [{
            "tool": "search_arxiv",
            "result": {
                "items": [
                    {"id": "2301.12345v2", "title": "Test paper"},
                    {"id": "2401.00001", "title": "Another paper"},
                ]
            },
        }]

    def test_resolves_first_arxiv_result(self):
        result = main._resolve_agent_step_args(
            "download_arxiv_paper",
            {"arxiv_id": {"$from_step": 1, "path": "items.0.id"}},
            self.observations,
            2,
        )
        self.assertEqual(result["arxiv_id"], "2301.12345v2")

    def test_resolves_selected_arxiv_result(self):
        result = main._resolve_agent_step_args(
            "download_arxiv_paper",
            {"arxiv_id": {"$from_step": 1, "path": "items.1.id"}},
            self.observations,
            2,
        )
        self.assertEqual(result["arxiv_id"], "2401.00001")

    def test_rejects_non_search_source(self):
        observations = [{"tool": "library_stats", "result": {}}]
        with self.assertRaises(HTTPException):
            main._resolve_agent_step_args(
                "download_arxiv_paper",
                {"arxiv_id": {"$from_step": 1, "path": "items.0.id"}},
                observations,
                2,
            )

    def test_rejects_failed_previous_step(self):
        observations = [{"tool": "search_arxiv", "error": "network error"}]
        with self.assertRaises(HTTPException):
            main._resolve_agent_step_args(
                "download_arxiv_paper",
                {"arxiv_id": {"$from_step": 1, "path": "items.0.id"}},
                observations,
                2,
            )

    def test_rejects_empty_search_results(self):
        observations = [{
            "tool": "search_arxiv",
            "result": {"items": []},
        }]
        with self.assertRaises(HTTPException):
            main._resolve_agent_step_args(
                "download_arxiv_paper",
                {"arxiv_id": {"$from_step": 1, "path": "items.0.id"}},
                observations,
                2,
            )

    def test_rejects_arbitrary_path(self):
        with self.assertRaises(HTTPException):
            main._resolve_agent_step_args(
                "download_arxiv_paper",
                {"arxiv_id": {"$from_step": 1, "path": "items.0.title"}},
                self.observations,
                2,
            )

    def test_rejects_non_immediate_step(self):
        with self.assertRaises(HTTPException):
            main._resolve_agent_step_args(
                "download_arxiv_paper",
                {"arxiv_id": {"$from_step": 1, "path": "items.0.id"}},
                self.observations,
                3,
            )

    def test_rejects_invalid_id_from_search_result(self):
        observations = [{
            "tool": "search_arxiv",
            "result": {"items": [{"id": "not-an-arxiv-id"}]},
        }]
        with self.assertRaises(HTTPException):
            main._resolve_agent_step_args(
                "download_arxiv_paper",
                {"arxiv_id": {"$from_step": 1, "path": "items.0.id"}},
                observations,
                2,
            )

    def test_reference_is_not_allowed_for_other_tools(self):
        with self.assertRaises(HTTPException):
            main._resolve_agent_step_args(
                "search_library",
                {"query": {"$from_step": 1, "path": "items.0.id"}},
                self.observations,
                2,
            )


if __name__ == "__main__":
    unittest.main()
