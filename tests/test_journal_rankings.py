import unittest
from unittest.mock import patch

from daily_arxiv.daily_arxiv.journal_rankings import (
    TIER_PRIORITY,
    annotate_paper_journal_rank,
    classify_journal,
    journal_tier_distribution,
    normalize_journal_name,
    select_journal_quality_and_exploration,
    selection_slots,
)
from daily_arxiv.daily_arxiv.openalex_client import openalex_work_to_paper


def paper(paper_id, journal="", *, work_type="article", title=None):
    return {
        "id": paper_id,
        "title": title or paper_id,
        "summary": "management evidence",
        "journal": journal,
        "source_name": journal,
        "work_type": work_type,
    }


class JournalRankingTests(unittest.TestCase):
    def test_exact_curated_tier_matches(self):
        expected = {
            "Review of Accounting Studies": "S",
            "Journal of Accounting Research": "S",
            "Journal of Financial Economics": "S",
            "MIS Quarterly": "S",
            "Management Science": "S",
            "Journal of Corporate Finance": "A",
            "Journal of Supply Chain Management": "A",
            "Journal of Business Ethics": "B",
            "The British Accounting Review": "B",
            "International Journal of Production Economics": "B",
        }
        for journal, tier in expected.items():
            with self.subTest(journal=journal):
                self.assertEqual(tier, classify_journal(journal))

    def test_unknown_formal_missing_and_preprint(self):
        self.assertEqual("C", classify_journal("A Journal Not In The Curated Table"))
        self.assertEqual("U", classify_journal("", work_type="article"))
        self.assertEqual("U", classify_journal("Management Science", work_type="preprint"))

    def test_normalization_supports_safe_presentation_variants(self):
        self.assertEqual(
            normalize_journal_name("The Accounting, Organizations & Society"),
            normalize_journal_name("Accounting, Organizations and Society"),
        )
        self.assertEqual(
            "journal of finance",
            normalize_journal_name("  THE Journal of Finance "),
        )
        self.assertEqual("mis quarterly", normalize_journal_name("MIS QUARTERLY"))

    def test_normalization_does_not_do_substring_matching(self):
        self.assertEqual("C", classify_journal("International Journal of Finance and Economics"))
        self.assertEqual("C", classify_journal("International Journal of Management Science"))
        self.assertEqual("C", classify_journal("Accounting Review and Policy"))

    def test_identifier_priority_is_explicit_and_extensible(self):
        with patch.dict(
            "daily_arxiv.daily_arxiv.journal_rankings.ISSN_L_TO_TIER",
            {"1234-5678": "A"},
            clear=True,
        ), patch.dict(
            "daily_arxiv.daily_arxiv.journal_rankings.SOURCE_ID_TO_TIER",
            {"S123": "B"},
            clear=True,
        ):
            self.assertEqual("A", classify_journal("Unknown", issn_l="1234-5678"))
            self.assertEqual("B", classify_journal("Unknown", source_id="https://openalex.org/S123"))
            self.assertEqual("S", classify_journal("Management Science", issn_l="missing"))

    def test_priority_values_and_annotation_metadata(self):
        self.assertEqual({"S": 4, "A": 3, "B": 2, "C": 1, "U": 0}, TIER_PRIORITY)
        annotated = annotate_paper_journal_rank(paper("1", "Review of Accounting Studies"))
        self.assertEqual("S", annotated["journal_tier"])
        self.assertEqual(4, annotated["journal_priority"])
        self.assertEqual(["priority-S"], annotated["journal_rank_sources"])
        preprint = annotate_paper_journal_rank(paper("2", "Management Science", work_type="preprint"))
        self.assertEqual("U", preprint["journal_tier"])
        self.assertEqual(["Preprint"], preprint["journal_rank_sources"])
        self.assertEqual([], annotate_paper_journal_rank(paper("3", "Unknown"))["journal_rank_sources"])

    def test_tier_distribution_has_all_buckets(self):
        values = [
            paper("s", "Management Science"),
            paper("a", "Journal of Corporate Finance"),
            paper("b", "Journal of Business Ethics"),
            paper("c", "Unknown Formal Journal"),
            paper("u", "", work_type="preprint"),
        ]
        annotated = [annotate_paper_journal_rank(value) for value in values]
        self.assertEqual({"S": 1, "A": 1, "B": 1, "C": 1, "U": 1}, journal_tier_distribution(annotated))

    def test_selection_slots_for_all_caps(self):
        self.assertEqual((18, 2), selection_slots(20))
        self.assertEqual((8, 2), selection_slots(10))
        self.assertEqual((1, 1), selection_slots(2))
        self.assertEqual((1, 0), selection_slots(1))

    def test_quality_sort_is_stable_within_tier(self):
        values = [
            paper("u0", "Unknown"),
            paper("s0", "Management Science"),
            paper("a0", "Journal of Corporate Finance"),
            paper("a1", "Journal of Supply Chain Management"),
            paper("b0", "Journal of Business Ethics"),
            paper("s1", "MIS Quarterly"),
        ]
        selected, deferred = select_journal_quality_and_exploration(values, 4)
        self.assertEqual(["s0", "s1", "u0", "a0"], [item["id"] for item in selected])
        self.assertEqual(["journal_quality", "journal_quality", "exploration", "exploration"], [item["selection_reason"] for item in selected])
        self.assertEqual(["a1", "b0"], [item["id"] for item in deferred])

    def test_limit_20_is_18_quality_plus_2_original_order_exploration(self):
        values = [paper(f"u{i}", "Unknown Formal Journal") for i in range(30)]
        values[25] = paper("s-late", "Management Science")
        values[27] = paper("a-late", "Journal of Corporate Finance")
        selected, deferred = select_journal_quality_and_exploration(values, 20)
        self.assertEqual(20, len(selected))
        self.assertEqual(18, sum(item["selection_reason"] == "journal_quality" for item in selected))
        self.assertEqual(2, sum(item["selection_reason"] == "exploration" for item in selected))
        self.assertEqual(["s-late", "a-late"], [item["id"] for item in selected[:2]])
        self.assertEqual(["u16", "u17"], [item["id"] for item in selected[-2:]])
        self.assertEqual(10, len(deferred))

    def test_limit_10_is_8_quality_plus_2_exploration(self):
        values = [paper(f"u{i}", "Unknown Formal Journal") for i in range(15)]
        values[10] = paper("s", "Management Science")
        values[11] = paper("a", "Journal of Corporate Finance")
        selected, deferred = select_journal_quality_and_exploration(values, 10)
        self.assertEqual(10, len(selected))
        self.assertEqual(8, sum(item["selection_reason"] == "journal_quality" for item in selected))
        self.assertEqual(2, sum(item["selection_reason"] == "exploration" for item in selected))
        self.assertEqual(5, len(deferred))

    def test_limit_2_and_limit_1(self):
        values = [paper("u0", "Unknown"), paper("s", "Management Science"), paper("a", "Journal of Corporate Finance")]
        selected, deferred = select_journal_quality_and_exploration(values, 2)
        self.assertEqual(["s", "u0"], [item["id"] for item in selected])
        self.assertEqual(["journal_quality", "exploration"], [item["selection_reason"] for item in selected])
        self.assertEqual(["a"], [item["id"] for item in deferred])
        selected, deferred = select_journal_quality_and_exploration(values, 1)
        self.assertEqual(["s"], [item["id"] for item in selected])
        self.assertEqual(["u0", "a"], [item["id"] for item in deferred])

    def test_candidates_at_or_below_cap_remain_in_input_order(self):
        values = [paper("u", "Unknown"), paper("s", "Management Science")]
        selected, deferred = select_journal_quality_and_exploration(values, 20)
        self.assertEqual(["u", "s"], [item["id"] for item in selected])
        self.assertEqual([], deferred)

    def test_selection_has_no_duplicate_ids(self):
        values = [paper(f"u{i}", "Unknown") for i in range(8)]
        values += [paper("same", "Management Science"), paper("same", "Management Science")]
        selected, _ = select_journal_quality_and_exploration(values, 5)
        ids = [item["id"] for item in selected]
        self.assertEqual(len(ids), len(set(ids)))

    def test_deferred_papers_are_not_marked_as_processed(self):
        values = [paper(f"u{i}", "Unknown") for i in range(5)]
        selected, deferred = select_journal_quality_and_exploration(values, 3)
        self.assertEqual([], [item for item in deferred if "AI" in item])
        self.assertTrue(all("selection_reason" not in item for item in deferred))
        self.assertEqual(3, len(selected))

    def test_openalex_source_metadata_maps_safely_without_extra_api_shape(self):
        work = {
            "id": "https://openalex.org/W999",
            "title": "Management source metadata",
            "abstract_inverted_index": {"Management": [0], "evidence": [1]},
            "primary_location": {
                "landing_page_url": "https://publisher.example/article",
                "source": {
                    "id": "https://openalex.org/S999",
                    "display_name": "Management Science",
                    "issn_l": "1234-5678",
                    "issn": ["1234-5678", "8765-4321"],
                },
            },
            "publication_date": "2026-08-24",
            "type": "article",
        }
        paper_value = openalex_work_to_paper(work)
        self.assertEqual("S999", paper_value["source_id"])
        self.assertEqual("1234-5678", paper_value["issn_l"])
        self.assertEqual(["1234-5678", "8765-4321"], paper_value["issn"])
        missing = openalex_work_to_paper({"id": "W1000", "title": "No source"})
        self.assertEqual("", missing["source_id"])
        self.assertEqual("", missing["issn_l"])
        self.assertEqual([], missing["issn"])


if __name__ == "__main__":
    unittest.main()
