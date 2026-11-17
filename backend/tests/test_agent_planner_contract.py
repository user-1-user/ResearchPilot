import unittest
from unittest.mock import patch
from main import _planner


class AgentPlannerContractTests(unittest.TestCase):

    @patch("main.llm_json")
    @patch("main.llm_client", new=object())
    def test_prompt_contains_real_tool_schemas(self, mock_llm_json):
        mock_llm_json.return_value = {
            "goal": "review",
            "steps": [{
                "tool": "generate_literature_review",
                "args": {"topic": "Transformer"},
            }],
        }

        result = _planner("Generate a literature review", [])
        self.assertEqual(len(result["steps"]), 1)

        prompt = mock_llm_json.call_args.args[0]
        self.assertIn("arguments_schema", prompt)
        self.assertIn('"generate_literature_review"', prompt)
        self.assertIn('"topic"', prompt)
        self.assertIn('"search_arxiv"', prompt)
        self.assertIn('"arxiv_id"', prompt)
        self.assertIn("Never invent unsupported argument fields", prompt)

    @patch("main.llm_json")
    @patch("main.llm_client", new=object())
    def test_prompt_documents_review_capability(self, mock_llm_json):
        mock_llm_json.return_value = {
            "goal": "review",
            "steps": [{
                "tool": "generate_literature_review",
                "args": {"topic": "Transformer"},
            }],
        }

        _planner("Generate a literature review", [])
        prompt = mock_llm_json.call_args.args[0]

        self.assertIn("only accepts topic", prompt)
        self.assertNotIn('"include_arxiv"', prompt)
        self.assertNotIn('"use_local_evidence"', prompt)
        self.assertNotIn('"focus"', prompt)


    @patch("main.llm_json")
    @patch("main.llm_client", new=object())
    def test_search_then_download_creates_two_step_plan(self, mock_llm_json):
        request = (
            "\u8bf7\u641c\u7d22\u8fd1\u4e00\u5e74\u4e0e Transformer "
            "Positional Encoding \u76f8\u5173\u7684 arxiv \u8bba\u6587\uff0c"
            "\u5e76\u4e0b\u8f7d\u7b2c\u4e00\u7bc7"
        )

        result = _planner(request, [])

        self.assertEqual(
            [step["tool"] for step in result["steps"]],
            ["search_arxiv", "download_arxiv_paper"],
        )
        self.assertEqual(
            result["steps"][0]["args"]["sort_by"],
            "latest",
        )
        self.assertEqual(
            result["steps"][1]["args"]["arxiv_id"],
            {
                "$from_step": 1,
                "path": "items.0.id",
            },
        )
        mock_llm_json.assert_not_called()

    @patch("main.llm_client", new=None)
    def test_download_without_id_requests_clarification(self):
        result = _planner("\u8bf7\u4e0b\u8f7d arxiv paper", [])
        self.assertEqual(result["steps"], [])
        self.assertIn("arXiv ID", result["clarification"])

    @patch("main.llm_client", new=None)
    def test_download_with_valid_id_creates_tool_step(self):
        result = _planner("\u8bf7\u4e0b\u8f7d arxiv paper 2301.12345v2", [])
        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual(
            result["steps"][0]["args"]["arxiv_id"],
            "2301.12345v2",
        )

    @patch("main.llm_json")
    @patch("main.llm_client", new=object())
    def test_llm_download_without_id_is_blocked(self, mock_llm_json):
        mock_llm_json.return_value = {
            "goal": "download paper",
            "steps": [{
                "tool": "download_arxiv_paper",
                "args": {},
            }],
        }

        result = _planner("\u8bf7\u4e0b\u8f7d arxiv paper", [])
        self.assertEqual(result["steps"], [])
        self.assertIn("arXiv ID", result["clarification"])


    @patch("main.llm_json")
    @patch("main.llm_client", new=object())
    def test_explicit_download_without_id_bypasses_llm(self, mock_llm_json):
        mock_llm_json.return_value = {
            "goal": "search",
            "steps": [{"tool": "search_arxiv", "args": {"query": "machine learning"}}],
        }
        request = "\u8bf7\u4e0b\u8f7d arxiv paper"
        result = _planner(request, [])

        self.assertEqual(result["steps"], [])
        self.assertIn("arXiv ID", result["clarification"])
        mock_llm_json.assert_not_called()

    @patch("main.llm_json")
    @patch("main.llm_client", new=object())
    def test_explicit_download_with_id_bypasses_llm(self, mock_llm_json):
        mock_llm_json.return_value = {
            "goal": "search",
            "steps": [{"tool": "search_arxiv", "args": {"query": "machine learning"}}],
        }
        request = "\u8bf7\u4e0b\u8f7d arxiv paper 2301.12345v2"
        result = _planner(request, [])

        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual(result["steps"][0]["tool"], "download_arxiv_paper")
        self.assertEqual(result["steps"][0]["args"]["arxiv_id"], "2301.12345v2")
        mock_llm_json.assert_not_called()


if __name__ == "__main__":
    unittest.main()
