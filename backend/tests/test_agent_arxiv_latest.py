import unittest
from unittest.mock import patch

from main import _planner, _run_tool


class AgentArxivLatestTests(unittest.TestCase):

    @patch("main._arxiv_search")
    def test_latest_filters_and_sorts_top_five(self, mock_search):
        mock_search.return_value = [
            {
                "id": f"rag-{i}",
                "title": "Retrieval Augmented Generation paper",
                "summary": "Retrieval augmented generation research",
                "published": f"2025-01-{i:02d}T00:00:00Z",
            }
            for i in range(1, 7)
        ] + [{
            "id": "irrelevant",
            "title": "White Dwarf Stars",
            "summary": "Astronomy and stellar evolution",
            "published": "2026-01-01T00:00:00Z",
        }]

        result = _run_tool(
            "test-user",
            "search_arxiv",
            {
                "query": "retrieval augmented generation RAG",
                "sort_by": "latest",
            },
        )

        self.assertEqual(
            [call.args for call in mock_search.call_args_list],
            [
                (
                    'ti:"retrieval-augmented generation" OR abs:"retrieval-augmented generation"',
                    100,
                    "submittedDate",
                ),
                (
                    'ti:"retrieval augmented generation" OR abs:"retrieval augmented generation"',
                    100,
                    "submittedDate",
                ),
                (
                    'ti:RAG AND (abs:retrieval OR abs:augmented OR abs:generation)',
                    100,
                    "submittedDate",
                ),
            ],
        )
        self.assertEqual(len(result["items"]), 5)
        self.assertEqual(
            [x["id"] for x in result["items"]],
            ["rag-6", "rag-5", "rag-4", "rag-3", "rag-2"],
        )
        self.assertEqual(result["sort_by"], "latest")
        self.assertEqual(result["candidate_count"], 7)


    @patch("main._arxiv_search")
    def test_rag_latest_rejects_unrelated_keyword_overlap(self, mock_search):
        mock_search.return_value = [
            {
                "id": "unrelated-1",
                "title": "ArchSpace: LLM Architecture Research",
                "summary": "We study retrieval methods and generation of architectures.",
                "published": "2026-11-12T00:00:00Z",
            },
            {
                "id": "unrelated-2",
                "title": "A Paragraph-Based Language Model",
                "summary": "A study of language generation.",
                "published": "2026-11-11T00:00:00Z",
            },
            {
                "id": "rag-acronym",
                "title": "RAG: Retrieval Methods for Knowledge Systems",
                "summary": "A retrieval augmented approach.",
                "published": "2026-11-10T00:00:00Z",
            },
            {
                "id": "rag-phrase",
                "title": "Efficient Retrieval-Augmented Generation",
                "summary": "A new framework.",
                "published": "2026-11-09T00:00:00Z",
            },
        ]

        result = _run_tool(
            "test-user",
            "search_arxiv",
            {
                "query": "retrieval augmented generation RAG",
                "sort_by": "latest",
            },
        )

        self.assertEqual(
            [item["id"] for item in result["items"]],
            ["rag-acronym", "rag-phrase"],
        )
        self.assertEqual(result["candidate_count"], 4)
        self.assertEqual(result["filtered_count"], 2)
        self.assertEqual(mock_search.call_count, 3)

    @patch("main._arxiv_search")
    def test_relevance_mode_preserves_old_behavior(self, mock_search):
        mock_search.return_value = [{"title": "test"}]

        result = _run_tool(
            "test-user",
            "search_arxiv",
            {"query": "transformer"},
        )

        mock_search.assert_called_once_with("transformer", 6)
        self.assertEqual(result["sort_by"], "relevance")

    @patch("main.llm_json")
    @patch("main.llm_client", new=object())
    def test_planner_preserves_latest_intent(self, mock_llm_json):
        mock_llm_json.return_value = {
            "goal": "search",
            "steps": [{
                "tool": "search_arxiv",
                "args": {
                    "query": "retrieval augmented generation RAG",
                    "sort_by": "relevance",
                },
            }],
        }

        result = _planner(
            "\u5e2e\u6211\u641c\u7d22 RAG \u7684\u6700\u65b0\u8bba\u6587",
            [],
        )

        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual(
            result["steps"][0]["args"]["sort_by"],
            "latest",
        )


    @patch("main._arxiv_search")
    def test_rag_latest_distinguishes_core_research_from_background(
        self, mock_search
    ):
        mock_search.return_value = [
            {
                "id": "background-only",
                "title": "AstroMemBench: Memory Systems for Spacecraft Agents",
                "summary": (
                    "Retrieval-augmented generation has attracted attention. "
                    "We propose a memory architecture for spacecraft agents."
                ),
                "published": "2026-11-12T00:00:00Z",
            },
            {
                "id": "abstract-core",
                "title": "Evidence-Guided Adaptive Graph Retrieval",
                "summary": (
                    "We propose EAGR, a graph retrieval method for "
                    "retrieval-augmented generation in multi-hop question answering."
                ),
                "published": "2026-11-11T00:00:00Z",
            },
            {
                "id": "title-core",
                "title": "RAG-TRACE: Diagnosing Retrieval-Augmented Generation",
                "summary": "We introduce a diagnostic framework.",
                "published": "2026-11-10T00:00:00Z",
            },
            {
                "id": "unrelated",
                "title": "A New Optimization Method",
                "summary": "We propose an optimizer for neural networks.",
                "published": "2026-11-09T00:00:00Z",
            },
        ]

        result = _run_tool(
            "test-user",
            "search_arxiv",
            {
                "query": "retrieval augmented generation RAG",
                "sort_by": "latest",
            },
        )

        ids = [item["id"] for item in result["items"]]

        self.assertEqual(
            ids,
            ["abstract-core", "title-core"],
        )
        self.assertEqual(result["candidate_count"], 4)
        self.assertEqual(result["filtered_count"], 2)


    @patch("main._arxiv_search")
    def test_rag_latest_rejects_repeated_background_mentions(self, mock_search):
        mock_search.return_value = [
            {
                "id": "background-repeated",
                "title": "DISCERN: Persistent LLM-Agent Memory",
                "summary": (
                    "Retrieval-augmented generation is widely used. "
                    "Retrieval-augmented generation can retrieve relevant passages. "
                    "We propose a persistent memory system for enterprise agents."
                ),
                "published": "2026-11-12T00:00:00Z",
            },
            {
                "id": "core-rag",
                "title": "A New Retrieval Framework",
                "summary": (
                    "We propose a method for retrieval-augmented generation "
                    "that improves evidence selection."
                ),
                "published": "2026-11-11T00:00:00Z",
            },
        ]

        result = _run_tool(
            "test-user",
            "search_arxiv",
            {
                "query": "retrieval augmented generation RAG",
                "sort_by": "latest",
            },
        )

        self.assertEqual(
            [item["id"] for item in result["items"]],
            ["core-rag"],
        )
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["filtered_count"], 1)


    @patch("main._arxiv_search")
    def test_rag_latest_does_not_treat_generic_nouns_as_contributions(self, mock_search):
        mock_search.return_value = [
            {
                "id": "background-system",
                "title": "Citation Behavior in Language Models",
                "summary": (
                    "Retrieval-augmented generation systems attach citations "
                    "to attribute claims to their sources. "
                    "These systems are widely used in language models."
                ),
                "published": "2026-11-12T00:00:00Z",
            },
            {
                "id": "core-method",
                "title": "Evidence Selection for Language Models",
                "summary": (
                    "We propose a new method for retrieval-augmented "
                    "generation that improves evidence selection."
                ),
                "published": "2026-11-11T00:00:00Z",
            },
        ]

        result = _run_tool(
            "test-user",
            "search_arxiv",
            {
                "query": "retrieval augmented generation RAG",
                "sort_by": "latest",
            },
        )

        self.assertEqual(
            [item["id"] for item in result["items"]],
            ["core-method"],
        )
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["filtered_count"], 1)


    @patch("main._arxiv_search")
    def test_rag_latest_rejects_rag_baseline_and_component_mentions(
        self, mock_search
    ):
        mock_search.return_value = [
            {
                "id": "physics-baseline",
                "title": "Chain of Physics: Supporting Fault Diagnosis",
                "summary": (
                    "Compared with direct prompting and conventional "
                    "Retrieval-Augmented Generation (RAG), CoP improves "
                    "diagnostic accuracy across all evaluated models."
                ),
                "published": "2026-11-12T00:00:00Z",
            },
            {
                "id": "mobility-component",
                "title": "Mobility-Aware Task Offloading in Edge Networks",
                "summary": (
                    "To solve the former, a retrieval-augmented generation "
                    "(RAG)-assisted algorithm is proposed."
                ),
                "published": "2026-11-11T00:00:00Z",
            },
            {
                "id": "vla-component",
                "title": "Efficient VLA Inference via Retrieval-Based Speculation",
                "summary": (
                    "We propose a hardware-software framework. "
                    "Specifically, it introduces a lightweight "
                    "retrieval-augmented generation (RAG)-based draft model."
                ),
                "published": "2026-11-10T00:00:00Z",
            },
            {
                "id": "nebula-background",
                "title": "Nebula: Distributed Position-Independent Caching",
                "summary": (
                    "Modern LLM workflows, such as retrieval-augmented "
                    "generation (RAG) and LLM agents, build long prompts. "
                    "We present Nebula, a distributed caching framework."
                ),
                "published": "2026-11-09T00:00:00Z",
            },
            {
                "id": "abstract-core",
                "title": "Evidence-Guided Adaptive Graph Retrieval",
                "summary": (
                    "We propose EAGR, a graph retrieval method for "
                    "retrieval-augmented generation in multi-hop "
                    "question answering."
                ),
                "published": "2026-11-08T00:00:00Z",
            },
            {
                "id": "title-core",
                "title": "RAG-TRACE: Diagnosing Retrieval-Augmented Generation",
                "summary": "We introduce a diagnostic framework.",
                "published": "2026-11-07T00:00:00Z",
            },
        ]

        result = _run_tool(
            "test-user",
            "search_arxiv",
            {
                "query": "retrieval augmented generation RAG",
                "sort_by": "latest",
            },
        )

        self.assertEqual(
            [item["id"] for item in result["items"]],
            ["abstract-core", "title-core"],
        )
        self.assertEqual(result["candidate_count"], 6)
        self.assertEqual(result["filtered_count"], 2)


if __name__ == "__main__":
    unittest.main()
