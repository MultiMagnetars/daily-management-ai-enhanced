"""Curated journal prestige metadata and bounded AI selection helpers.

The tier list is an internal reading-priority configuration.  It is applied
only after the existing FILTER_KEYWORDS admission check and never affects
OpenAlex retrieval, deduplication, or same-day accumulation.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


S_TIER_JOURNALS = frozenset(
    {
        # Accounting
        "The Accounting Review",
        "Accounting, Organizations and Society",
        "Contemporary Accounting Research",
        "Journal of Accounting and Economics",
        "Journal of Accounting Research",
        "Review of Accounting Studies",
        # Finance
        "Journal of Finance",
        "The Journal of Finance",
        "Journal of Financial Economics",
        "Review of Financial Studies",
        "The Review of Financial Studies",
        "Journal of Financial and Quantitative Analysis",
        "Review of Finance",
        # Management / strategy / organization
        "Academy of Management Annals",
        "Academy of Management Journal",
        "Academy of Management Review",
        "Administrative Science Quarterly",
        "Journal of Management",
        "Journal of Management Studies",
        "Organization Science",
        "Strategic Management Journal",
        "Journal of International Business Studies",
        "Research Policy",
        # Information systems / digital
        "Information Systems Research",
        "MIS Quarterly",
        "Journal of Management Information Systems",
        # Operations / supply chain
        "Management Science",
        "Operations Research",
        "Journal of Operations Management",
        "Manufacturing & Service Operations Management",
        "Manufacturing and Service Operations Management",
        "Production and Operations Management",
        # Economics / interdisciplinary
        "American Economic Review",
        "Econometrica",
        "Journal of Political Economy",
        "The Quarterly Journal of Economics",
        "Quarterly Journal of Economics",
        "Review of Economic Studies",
        "The Review of Economic Studies",
    }
)

A_TIER_JOURNALS = frozenset(
    {
        # Finance
        "Journal of Corporate Finance",
        "Journal of Financial Intermediation",
        "Journal of Money, Credit and Banking",
        # Information systems
        "Journal of the Association for Information Systems",
        "European Journal of Information Systems",
        "Information Systems Journal",
        "Journal of Information Technology",
        "Journal of Strategic Information Systems",
        # Operations / supply chain
        "International Journal of Operations & Production Management",
        "International Journal of Operations and Production Management",
        "Journal of Supply Chain Management",
        "European Journal of Operational Research",
        # Management / strategy
        "British Journal of Management",
        "Academy of Management Perspectives",
        "Journal of World Business",
        "Organizational Research Methods",
        "Human Relations",
        "The Leadership Quarterly",
        "Leadership Quarterly",
        "Organization Studies",
        "Global Strategy Journal",
        # Innovation
        "Journal of Product Innovation Management",
    }
)

B_TIER_JOURNALS = frozenset(
    {
        # Accounting / auditing
        "Abacus",
        "Accounting and Business Research",
        "Accounting Forum",
        "Accounting Horizons",
        "Accounting, Auditing & Accountability Journal",
        "Accounting Auditing & Accountability Journal",
        "Auditing: A Journal of Practice & Theory",
        "Behavioral Research in Accounting",
        "The British Accounting Review",
        "British Accounting Review",
        "European Accounting Review",
        "Journal of Accounting and Public Policy",
        "Journal of Accounting, Auditing & Finance",
        "Journal of Accounting Auditing & Finance",
        "Journal of Business Finance & Accounting",
        "Journal of Business Finance and Accounting",
        # Finance
        "Journal of Banking & Finance",
        "Journal of Banking and Finance",
        "Financial Management",
        "International Review of Financial Analysis",
        "Journal of Financial Stability",
        "European Financial Management",
        "Journal of Empirical Finance",
        # ESG / governance
        "Journal of Business Ethics",
        "Corporate Governance: An International Review",
        "Business Strategy and the Environment",
        "Business & Society",
        "Business and Society",
        "Organization & Environment",
        "Organization and Environment",
        # Information systems / digital
        "Decision Support Systems",
        "Information & Management",
        "Information and Management",
        "Information and Organization",
        "Information Systems Frontiers",
        "Internet Research",
        "International Journal of Electronic Commerce",
        "Government Information Quarterly",
        # Supply chain / operations
        "International Journal of Production Economics",
        "International Journal of Production Research",
        "Journal of Business Logistics",
        "Journal of Purchasing and Supply Management",
        "Supply Chain Management: An International Journal",
        "Production Planning & Control",
        "Production Planning and Control",
        "Transportation Research Part E: Logistics and Transportation Review",
        # General management
        "Journal of Business Research",
        "International Journal of Management Reviews",
        "International Business Review",
        "Asia Pacific Journal of Management",
        # Green / energy economics
        "Energy Economics",
    }
)

TIER_PRIORITY = {"S": 4, "A": 3, "B": 2, "C": 1, "U": 0}

# These maps are intentionally empty in V1.  The lookup path is explicit so
# verified ISSN-L or OpenAlex source-ID mappings can be added later without
# changing the selection algorithm or duplicating the title table.
ISSN_L_TO_TIER: dict[str, str] = {}
SOURCE_ID_TO_TIER: dict[str, str] = {}


def normalize_journal_name(name: Any) -> str:
    """Normalize only harmless presentation variants for exact matching."""

    if not isinstance(name, str):
        return ""
    text = name.casefold().strip().replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = " ".join(text.split())
    if text.startswith("the "):
        text = text[4:]
    return text


def _normalize_issn(value: Any) -> str:
    return value.strip().casefold() if isinstance(value, str) else ""


def normalize_source_id(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip().rstrip("/")
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return text if re.fullmatch(r"(?i)S\d+", text) else ""


def _title_tier_map() -> dict[str, str]:
    result: dict[str, str] = {}
    for tier, names in (("S", S_TIER_JOURNALS), ("A", A_TIER_JOURNALS), ("B", B_TIER_JOURNALS)):
        for name in names:
            result.setdefault(normalize_journal_name(name), tier)
    return result


JOURNAL_NAME_TO_TIER = _title_tier_map()


def _paper_metadata(paper_or_name: Any, *, work_type: Any = "", issn_l: Any = "", source_id: Any = ""):
    if isinstance(paper_or_name, Mapping):
        paper = paper_or_name
        return (
            paper.get("journal") or paper.get("source_name") or "",
            paper.get("work_type") or "",
            paper.get("issn_l") or "",
            paper.get("source_id") or "",
        )
    return paper_or_name or "", work_type, issn_l, source_id


def _classify_with_source(
    paper_or_name: Any,
    *,
    work_type: Any = "",
    issn_l: Any = "",
    source_id: Any = "",
) -> tuple[str, str]:
    journal_name, actual_work_type, actual_issn_l, actual_source_id = _paper_metadata(
        paper_or_name,
        work_type=work_type,
        issn_l=issn_l,
        source_id=source_id,
    )
    if str(actual_work_type).casefold().strip() == "preprint":
        return "U", "preprint"

    normalized_issn_l = _normalize_issn(actual_issn_l)
    if normalized_issn_l and normalized_issn_l in ISSN_L_TO_TIER:
        return ISSN_L_TO_TIER[normalized_issn_l], "ISSN-L"

    normalized_source_id = normalize_source_id(actual_source_id)
    if normalized_source_id and normalized_source_id in SOURCE_ID_TO_TIER:
        return SOURCE_ID_TO_TIER[normalized_source_id], "source_id"

    normalized_name = normalize_journal_name(journal_name)
    if normalized_name in JOURNAL_NAME_TO_TIER:
        return JOURNAL_NAME_TO_TIER[normalized_name], "journal_name"
    if normalized_name:
        return "C", "formal_journal"
    return "U", "missing_journal"


def classify_journal(
    paper_or_name: Any,
    *,
    work_type: Any = "",
    issn_l: Any = "",
    source_id: Any = "",
) -> str:
    """Return one internal tier using exact normalized matching only."""

    return _classify_with_source(
        paper_or_name,
        work_type=work_type,
        issn_l=issn_l,
        source_id=source_id,
    )[0]


def annotate_paper_journal_rank(paper: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with journal tier metadata and no inferred official badge."""

    annotated = dict(paper)
    tier, match_source = _classify_with_source(paper)
    if tier == "U" and match_source == "preprint":
        rank_sources = ["Preprint"]
    elif tier in {"S", "A", "B"}:
        # Internal priority labels are deliberately not frontend badges.
        rank_sources = [f"priority-{tier}"]
    else:
        rank_sources = []
    annotated.update(
        {
            "journal_tier": tier,
            "journal_priority": TIER_PRIORITY[tier],
            "journal_rank_sources": rank_sources,
        }
    )
    return annotated


def journal_tier_distribution(papers: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(paper.get("journal_tier") or "U") for paper in papers)
    return {tier: counts.get(tier, 0) for tier in ("S", "A", "B", "C", "U")}


def selection_slots(limit: int) -> tuple[int, int]:
    """Return quality and exploration slots for a positive AI cap."""

    if limit <= 0:
        return 0, 0
    if limit == 1:
        return 1, 0
    if limit == 2:
        return 1, 1
    exploration = min(2, limit)
    return limit - exploration, exploration


def _paper_identity(paper: Mapping[str, Any]) -> str:
    for key in ("id", "openalex_id", "doi"):
        value = paper.get(key)
        if isinstance(value, str) and value.strip():
            return f"{key}:{value.strip().casefold()}"
    title = str(paper.get("title") or "").casefold().strip()
    authors = paper.get("authors")
    if isinstance(authors, list):
        authors = "|".join(str(author).casefold().strip() for author in authors)
    else:
        authors = str(authors or "").casefold().strip()
    published = str(paper.get("published") or "").strip()
    return f"fallback:{title}|{authors}|{published}"


def select_journal_quality_and_exploration(
    papers: Sequence[Mapping[str, Any]],
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select stable quality-first papers plus original-order exploration papers."""

    annotated = [annotate_paper_journal_rank(paper) for paper in papers]
    if limit <= 0:
        return [], annotated

    if len(annotated) <= limit:
        selected = annotated
        for rank, paper in enumerate(selected, start=1):
            paper["selection_reason"] = "journal_quality"
            paper["selection_rank"] = rank
        return selected, []

    quality_slots, exploration_slots = selection_slots(limit)
    ranked_indices = sorted(
        range(len(annotated)),
        key=lambda index: (-int(annotated[index].get("journal_priority") or 0), index),
    )
    selected_indices: list[int] = []
    selected_identities: set[str] = set()
    for index in ranked_indices:
        identity = _paper_identity(annotated[index])
        if identity in selected_identities:
            continue
        selected_indices.append(index)
        selected_identities.add(identity)
        if len(selected_indices) >= quality_slots:
            break

    quality_indices = set(selected_indices)
    quality_indices_ordered = list(selected_indices)
    exploration_indices: list[int] = []
    if exploration_slots:
        for index, paper in enumerate(annotated):
            if index in quality_indices:
                continue
            identity = _paper_identity(paper)
            if identity in selected_identities:
                continue
            selected_indices.append(index)
            exploration_indices.append(index)
            selected_identities.add(identity)
            if len(exploration_indices) >= exploration_slots:
                break

    selected_indices = quality_indices_ordered + exploration_indices
    selected = [annotated[index] for index in selected_indices]
    for rank, paper in enumerate(selected, start=1):
        paper["selection_rank"] = rank
        paper["selection_reason"] = "journal_quality" if selected_indices[rank - 1] in quality_indices else "exploration"
    deferred = [
        paper for index, paper in enumerate(annotated)
        if index not in set(selected_indices)
    ]
    return selected, deferred


# Short aliases make the intended production call easy to discover.
annotate_journal_rank = annotate_paper_journal_rank
select_papers_for_ai = select_journal_quality_and_exploration
