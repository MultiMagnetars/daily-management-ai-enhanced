import ast
import re
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
AI_DIR = ROOT_DIR / "ai"
EXPECTED_FIELDS = [
    "abstract_translation",
    "tldr",
    "motivation",
    "method",
    "result",
    "conclusion",
]


class ManagementPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = (AI_DIR / "system.txt").read_text(encoding="utf-8")
        cls.template = (AI_DIR / "template.txt").read_text(encoding="utf-8")
        cls.structure = (AI_DIR / "structure.py").read_text(encoding="utf-8")
        cls.enhance = (AI_DIR / "enhance.py").read_text(encoding="utf-8")
        cls.active_prompt = (cls.system + "\n" + cls.template).lower()

    def test_schema_is_exactly_six_management_fields(self):
        tree = ast.parse(self.structure)
        structure_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Structure"
        )
        fields = [
            node.target.id
            for node in structure_class.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
        ]
        self.assertEqual(EXPECTED_FIELDS, fields)
        for extra in ("research_question", "theory", "data_sample", "mechanism", "heterogeneity", "contribution"):
            self.assertNotIn(extra, fields)

    def test_management_persona_and_domains_are_active(self):
        for phrase in (
            "professional academic paper analyst",
            "accounting",
            "corporate finance",
            "corporate governance",
            "esg",
            "capital markets",
            "digital transformation",
            "fintech",
            "supply-chain",
        ):
            self.assertIn(phrase, self.active_prompt)
        self.assertIn("management, accounting, corporate finance", self.active_prompt)

    def test_active_prompt_has_no_astronomy_persona(self):
        forbidden = (
            "observational radio astronomy",
            "pulsar",
            "magnetar",
            "frb",
            "single-pulse",
            "pulse-profile",
            "spin-down",
            "polarization",
            "toa",
            "dm",
            "rm",
            "telescope",
        )
        for term in forbidden:
            if len(term) <= 3:
                self.assertIsNone(
                    re.search(r"\b" + re.escape(term) + r"\b", self.active_prompt),
                    term,
                )
            else:
                self.assertNotIn(term, self.active_prompt, term)

    def test_title_abstract_only_evidence_and_hallucination_guards(self):
        for phrase in (
            "only paper content available",
            "journal ranking",
            "a-share",
            "fixed effects",
            "difference-in-differences",
            "instrumental variables",
            "agency theory",
            "stakeholder theory",
            "association, correlation, and prediction are not causation",
            "摘要未说明",
        ):
            self.assertIn(phrase.lower(), self.active_prompt)

    def test_methods_are_not_negated_when_unmentioned(self):
        for phrase in (
            "describe what methods are stated, not what methods were not used",
            "did not use",
            "does not use",
            "does not adopt",
            "is not based on",
            "do not mechanically write that a paper did not use",
            "bibliometric analysis and qualitative content analysis",
            "do not add a negative comparison with regression or fixed effects",
            "未提及”不等于“未使用",
        ):
            self.assertIn(phrase.lower(), self.active_prompt)

    def test_contribution_does_not_become_unstated_literature_gap(self):
        for phrase in (
            "do not infer a prior-literature deficiency",
            "extends, enriches, contributes to, or broadens",
            "only state a literature gap when",
            "prior studies have overlooked",
            "little is known",
            "research remains limited",
        ):
            self.assertIn(phrase.lower(), self.active_prompt)

    def test_moderation_is_distinct_from_mechanism(self):
        for phrase in (
            "a moderating effect is not automatically a mechanism",
            "label it “调节效应：...”",
            "do not place it under “作用机制：...”",
            "mechanism, channel, mediation, or transmission",
            "mechanism, moderation, and heterogeneity are distinct concepts",
            "a subgroup difference is not automatically a moderating effect",
            "heterogeneous effects",
            "do not substitute one category for another",
        ):
            self.assertIn(phrase.lower(), self.active_prompt)
        self.assertIn("result: str", self.structure)

    def test_field_mappings_and_missing_sections_are_explicit(self):
        for phrase in (
            "研究问题：",
            "理论/研究背景：",
            "理论框架：摘要未说明。",
            "数据与样本：",
            "核心变量/研究对象：",
            "研究方法：",
            "主要结论：",
            "作用机制：",
            "异质性：",
            "经济后果/进一步结果：",
            "理论/文献贡献：",
            "实践/政策启示：",
            "摘要未提供可翻译内容。",
        ):
            self.assertIn(phrase, self.active_prompt)

    def test_translation_and_terminology_rules_are_active(self):
        for phrase in (
            "completely, sentence by sentence",
            "original order",
            "do not summarize",
            "may, might, suggests",
            "corporate governance 公司治理",
            "earnings management 盈余管理",
            "audit quality 审计质量",
            "environmental governance 环境治理",
            "环境、社会与治理（esg）",
            "企业社会责任（csr）",
            "管理层讨论与分析（md&a）",
        ):
            self.assertIn(phrase.lower(), self.active_prompt)

    def test_non_empirical_and_economic_result_handling(self):
        for phrase in (
            "systematic review",
            "bibliometric",
            "theoretical",
            "conceptual",
            "survey",
            "experiment",
            "case",
            "methodological",
            "经济后果/进一步结果",
            "do not invent recommendations",
        ):
            self.assertIn(phrase.lower(), self.active_prompt)

    def test_fallbacks_keep_exact_management_six_fields(self):
        for phrase in (
            '"abstract_translation": ""',
            '"tldr": "摘要分析暂不可用"',
            '"motivation": "研究问题与理论背景分析暂不可用"',
            '"method": "数据、样本与研究方法分析暂不可用"',
            '"result": "研究结果分析暂不可用"',
            '"conclusion": "研究贡献与启示分析暂不可用"',
        ):
            self.assertIn(phrase, self.enhance)
        for old_phrase in (
            "Summary generation failed",
            "Motivation analysis unavailable",
            "Method extraction failed",
            "Result analysis unavailable",
            "Conclusion extraction failed",
            "Processing failed",
        ):
            self.assertNotIn(old_phrase, self.enhance)

    def test_prompt_keeps_only_supported_template_variables(self):
        self.assertEqual(self.template.count("{title}"), 1)
        self.assertEqual(self.template.count("{content}"), 1)
        self.assertNotIn("{research", self.template)
        self.assertNotIn("{theory", self.template)


if __name__ == "__main__":
    unittest.main()
