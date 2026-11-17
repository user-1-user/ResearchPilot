import asyncio
import json
import unittest
from unittest.mock import patch, call

import main


class AgentMultiStepTests(unittest.TestCase):

    async def collect_stream(self, response):
        chunks = []
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8")
            chunks.append(chunk)
        return "".join(chunks)

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
                {"message": "执行一个多步骤科研任务"},
                user={"id": "test-user"},
            )
            output = asyncio.run(self.collect_stream(response))

        return output, mock_tool

    def test_two_tools_execute_in_order(self):
        plan = {
            "goal": "search then analyze",
            "steps": [
                {
                    "tool": "search_library",
                    "args": {"query": "Transformer"},
                },
                {
                    "tool": "generate_framework",
                    "args": {"idea": "Transformer research idea"},
                },
            ],
        }

        output, mock_tool = self.run_agent(
            plan,
            [
                {"sources": [{"filename": "paper-a.pdf"}]},
                {"framework": "Research framework"},
            ],
        )

        self.assertEqual(
            mock_tool.call_args_list,
            [
                call(
                    "test-user",
                    "search_library",
                    {"query": "Transformer"},
                ),
                call(
                    "test-user",
                    "generate_framework",
                    {"idea": "Transformer research idea"},
                ),
            ],
        )

        self.assertIn("步骤 1", output)
        self.assertIn("步骤 2", output)
        self.assertIn("paper-a.pdf", output)
        self.assertIn("Research framework", output)

    def test_failed_step_does_not_block_next_step(self):
        plan = {
            "goal": "test error handling",
            "steps": [
                {
                    "tool": "search_library",
                    "args": {"query": "Transformer"},
                },
                {
                    "tool": "library_stats",
                    "args": {},
                },
            ],
        }

        output, mock_tool = self.run_agent(
            plan,
            [
                RuntimeError("simulated search failure"),
                {"paper_count": 3},
            ],
        )

        self.assertEqual(mock_tool.call_count, 2)
        self.assertIn("simulated search failure", output)
        self.assertIn("paper_count", output)
        self.assertIn("3", output)

    def test_all_successful_results_are_in_final_output(self):
        plan = {
            "goal": "collect results",
            "steps": [
                {"tool": "library_stats", "args": {}},
                {"tool": "extract_future_work", "args": {}},
            ],
        }

        output, mock_tool = self.run_agent(
            plan,
            [
                {"paper_count": 5},
                {"future_work": ["improve retrieval"]},
            ],
        )

        self.assertEqual(mock_tool.call_count, 2)
        self.assertIn("paper_count", output)
        self.assertIn("future_work", output)
        self.assertIn("improve retrieval", output)


if __name__ == "__main__":
    unittest.main()
