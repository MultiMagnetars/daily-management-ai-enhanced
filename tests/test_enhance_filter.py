import ast
import os
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

from daily_arxiv.daily_arxiv.journal_rankings import (
    select_journal_quality_and_exploration,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
AI_DIR = ROOT_DIR / "ai"

# Load only the pure filtering/selection helpers so these tests do not require
# optional LLM/runtime dependencies.
source_path = AI_DIR / "enhance.py"
source_tree = ast.parse(source_path.read_text(encoding="utf-8"))
helper_nodes = [
    node
    for node in source_tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name in {
        "parse_filter_keywords",
        "filter_papers_by_keywords",
        "parse_max_ai_papers",
        "apply_ai_paper_cap",
    }
]
helper_namespace = {
    "Dict": Dict,
    "List": List,
    "MAX_AI_PAPERS_PER_RUN": 20,
    "Optional": Optional,
    "sys": sys,
    "Tuple": Tuple,
    "select_journal_quality_and_exploration": select_journal_quality_and_exploration,
}
exec(
    compile(
        ast.Module(body=helper_nodes, type_ignores=[]),
        str(source_path),
        "exec",
    ),
    helper_namespace,
)
parse_filter_keywords = helper_namespace["parse_filter_keywords"]
filter_papers_by_keywords = helper_namespace["filter_papers_by_keywords"]
parse_max_ai_papers = helper_namespace["parse_max_ai_papers"]
apply_ai_paper_cap = helper_namespace["apply_ai_paper_cap"]


def paper(paper_id, title="", summary=""):
    return {
        "id": paper_id,
        "title": title,
        "summary": summary,
    }


class KeywordFilterTests(unittest.TestCase):
    def test_case_insensitive_match(self):
        papers = [paper("1", title="NEUTRON STAR OBSERVATIONS")]
        keywords = parse_filter_keywords("neutron star")

        filtered, counts = filter_papers_by_keywords(papers, keywords)

        self.assertEqual(["1"], [item["id"] for item in filtered])
        self.assertEqual({"neutron star": 1}, counts)

    def test_title_match(self):
        papers = [paper("1", title="Magnetar bursts")]
        keywords = parse_filter_keywords("magnetar")

        filtered, counts = filter_papers_by_keywords(papers, keywords)

        self.assertEqual(["1"], [item["id"] for item in filtered])
        self.assertEqual(1, counts["magnetar"])

    def test_summary_match(self):
        papers = [paper("1", summary="We measure a black hole accretion disk.")]
        keywords = parse_filter_keywords("black hole")

        filtered, counts = filter_papers_by_keywords(papers, keywords)

        self.assertEqual(["1"], [item["id"] for item in filtered])
        self.assertEqual(1, counts["black hole"])

    def test_space_phrase_and_hyphen_phrase(self):
        papers = [
            paper("1", summary="A black hole candidate is studied."),
            paper("2", summary="We evaluate a self-supervised model."),
        ]
        keywords = parse_filter_keywords("black hole,self-supervised")

        filtered, counts = filter_papers_by_keywords(papers, keywords)

        self.assertEqual(["1", "2"], [item["id"] for item in filtered])
        self.assertEqual(1, counts["black hole"])
        self.assertEqual(1, counts["self-supervised"])

    def test_or_match_and_per_keyword_counts(self):
        papers = [
            paper("1", title="Black hole and neutron star systems"),
            paper("2", summary="A magnetar observation"),
            paper("3", title="Unrelated paper"),
        ]
        keywords = parse_filter_keywords("black hole,neutron star,magnetar")

        filtered, counts = filter_papers_by_keywords(papers, keywords)

        self.assertEqual(["1", "2"], [item["id"] for item in filtered])
        self.assertEqual(1, counts["black hole"])
        self.assertEqual(1, counts["neutron star"])
        self.assertEqual(1, counts["magnetar"])

    def test_empty_filter_keywords_passes_all(self):
        papers = [
            paper("1", title="First paper"),
            paper("2", summary="Second paper"),
        ]

        for raw_value in (None, "", "   ", ", ,"):
            with self.subTest(raw_value=raw_value):
                environment = (
                    {}
                    if raw_value is None
                    else {"FILTER_KEYWORDS": raw_value}
                )
                with patch.dict(os.environ, environment, clear=True):
                    configured_value = os.environ.get("FILTER_KEYWORDS")
                keywords = parse_filter_keywords(configured_value)
                filtered, counts = filter_papers_by_keywords(papers, keywords)

                self.assertEqual(["1", "2"], [item["id"] for item in filtered])
                self.assertEqual({}, counts)

    def test_zero_matches_returns_empty_result(self):
        papers = [
            paper("1", title="Magnetar paper"),
            paper("2", summary="Neutron star paper"),
        ]
        keywords = parse_filter_keywords("black hole")

        filtered, counts = filter_papers_by_keywords(papers, keywords)

        self.assertEqual([], filtered)
        self.assertEqual({"black hole": 0}, counts)

    def test_management_second_layer_terms_match_title_and_abstract(self):
        papers = [
            paper(
                "spillover",
                title="Information Spillover and Disclosure Spillover",
            ),
            paper("peer", summary="Peer firm disclosure changes the information environment."),
            paper("externality", summary="Information externality affects capital allocation."),
            paper("mda", title="MD&A Disclosure Quality"),
            paper("ai", title="Medical Artificial Intelligence Applications"),
            paper("noise", title="A topic outside the configured vocabulary"),
        ]
        keywords = parse_filter_keywords(
            "information spillover,disclosure spillover,peer firm disclosure,"
            "information externality,MD&A disclosure,artificial intelligence"
        )

        filtered, counts = filter_papers_by_keywords(papers, keywords)

        self.assertEqual(
            ["spillover", "peer", "externality", "mda", "ai"],
            [item["id"] for item in filtered],
        )
        self.assertEqual(1, counts["information spillover"])
        self.assertEqual(1, counts["disclosure spillover"])
        self.assertEqual(1, counts["peer firm disclosure"])
        self.assertEqual(1, counts["information externality"])
        self.assertEqual(1, counts["MD&A disclosure"])
        self.assertEqual(1, counts["artificial intelligence"])

    def test_ai_cap_parser_falls_back_for_unset_invalid_and_nonpositive_values(self):
        for raw_value in (None, "", "   ", "not-an-int", "0", "-3"):
            with self.subTest(raw_value=raw_value):
                self.assertEqual(20, parse_max_ai_papers(raw_value))
        self.assertEqual(7, parse_max_ai_papers(" 7 "))

    def test_ai_cap_preserves_order_and_returns_deferred_records(self):
        papers = [paper(str(index), title=f"Paper {index}") for index in range(30)]

        selected, deferred = apply_ai_paper_cap(papers, 20)

        self.assertEqual([str(index) for index in range(20)], [item["id"] for item in selected])
        self.assertEqual([str(index) for index in range(20, 30)], [item["id"] for item in deferred])

    def test_ai_cap_keeps_all_records_when_limit_is_large(self):
        papers = [paper("1"), paper("2")]
        selected, deferred = apply_ai_paper_cap(papers, 20)
        self.assertEqual(["1", "2"], [item["id"] for item in selected])
        self.assertEqual(["U", "U"], [item["journal_tier"] for item in selected])
        self.assertEqual([], deferred)

    def test_ai_cap_is_after_filter_and_before_ai_processing(self):
        source = (ROOT_DIR / "ai" / "enhance.py").read_text(encoding="utf-8")
        main_source = source[source.index("def main():") :]
        filter_position = main_source.index("filter_papers_by_keywords(data, keywords)")
        cap_position = main_source.index("apply_ai_paper_cap(annotated_filtered_data, max_ai_papers)")
        ai_position = main_source.index("process_all_items(")

        self.assertLess(filter_position, cap_position)
        self.assertLess(cap_position, ai_position)
        self.assertIn("MAX_AI_PAPERS_PER_RUN", source)
        self.assertIn("keyword_matched_count", source)
        self.assertIn("ai_selected_count", source)
        self.assertIn("ai_deferred_count", source)


if __name__ == "__main__":
    unittest.main()
