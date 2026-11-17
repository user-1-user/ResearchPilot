import asyncio
import json
import unittest
from unittest.mock import patch, call

import main


class AgentResultReferenceIntegrationTests(unittest.TestCase):

    def collect_stream(self, response):
        async def collect():
            chunks = []
            async for chunk in response.body_iterator:
                if isinstance(chunk, bytes):
                    chunk = chunk.decode("utf-8")
                chunks.append(chunk)
            return "".join(chunks)

        return asyncio.run(collect())

    def run_agent(self, plan, tool_results):
        with (
            patch.object(main, "_planner", return_value=plan),
            patch.object(main, "_run_tool", side_effect=tool_results) as mock_tool,
            patch.object(main, "llm_client", None),
            patch.object(
                main,
                "_sse",
                side_effect=lambda event: json.dumps(
                    event, ensure_ascii=False
                ) + "\n",
            ),
        ):
            response = main.agent_chat(
                {"message": "请搜索 Transformer 论文并下载第一篇 arXiv 论文"},
                user={"id": "integration-test-user"},
            )
            output = self.collect_stream(response)

        return output, mock_tool

    def test_search_result_id_is_passed_to_download_tool(self):
        plan = {
            "goal": "Search and download Transformer papers",
            "steps": [
                {
                    "tool": "search_arxiv",
                    "args": {"query": "Transformer"},
                },
                {
                    "tool": "download_arxiv_paper",
                    "args": {
                        "arxiv_id": {
                            "$from_step": 1,
                            "path": "items.0.id",
                        }
                    },
                },
            ],
        }

        search_result = {
            "items": [
                {
                    "id": "2301.12345v2",
                    "title": "Test Transformer Paper",
                },
                {
                    "id": "2401.00001",
                    "title": "Another Paper",
                },
            ]
        }

        download_result = {
            "paper_id": "mock-paper-id",
            "filename": "2301.12345v2.pdf",
        }

        output, mock_tool = self.run_agent(
            plan,
            [search_result, download_result],
        )

        self.assertEqual(mock_tool.call_count, 2)

        self.assertEqual(
            mock_tool.call_args_list,
            [
                call(
                    "integration-test-user",
                    "search_arxiv",
                    {"query": "Transformer"},
                ),
                call(
                    "integration-test-user",
                    "download_arxiv_paper",
                    {"arxiv_id": "2301.12345v2"},
                ),
            ],
        )

        self.assertIn("search_arxiv", output)
        self.assertIn("download_arxiv_paper", output)
        self.assertIn("2301.12345v2", output)
        self.assertIn("mock-paper-id", output)

    def test_failed_search_blocks_referenced_download(self):
        plan = {
            "goal": "Download a paper after a failed search",
            "steps": [
                {
                    "tool": "search_arxiv",
                    "args": {"query": "Transformer"},
                },
                {
                    "tool": "download_arxiv_paper",
                    "args": {
                        "arxiv_id": {
                            "$from_step": 1,
                            "path": "items.0.id",
                        }
                    },
                },
            ],
        }

        output, mock_tool = self.run_agent(
            plan,
            [RuntimeError("simulated arXiv search failure")],
        )

        # A failed search must prevent the dependent download step.
        self.assertEqual(mock_tool.call_count, 1)

        self.assertEqual(
            mock_tool.call_args_list,
            [
                call(
                    "integration-test-user",
                    "search_arxiv",
                    {"query": "Transformer"},
                )
            ],
        )

        self.assertIn("simulated arXiv search failure", output)
        self.assertIn("download_arxiv_paper", output)
        self.assertIn("422", output)

    def test_second_search_result_id_is_passed_to_download_tool(self):
        plan = {
            "goal": "Search and download Transformer papers",
            "steps": [
                {
                    "tool": "search_arxiv",
                    "args": {"query": "Transformer"},
                },
                {
                    "tool": "download_arxiv_paper",
                    "args": {
                        "arxiv_id": {
                            "$from_step": 1,
                            "path": "items.1.id",
                        }
                    },
                },
            ],
        }

        search_result = {
            "items": [
                {"id": "2301.12345v2"},
                {"id": "2401.00001"},
            ]
        }

        output, mock_tool = self.run_agent(
            plan,
            [search_result, {"paper_id": "mock-second-paper"}],
        )

        self.assertEqual(mock_tool.call_count, 2)
        self.assertEqual(
            mock_tool.call_args_list[1],
            call(
                "integration-test-user",
                "download_arxiv_paper",
                {"arxiv_id": "2401.00001"},
            ),
        )

        self.assertIn("2401.00001", output)


if __name__ == "__main__":
    unittest.main()
