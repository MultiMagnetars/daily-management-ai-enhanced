import json
import subprocess
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


class ManagementFrontendOpenAlexTests(unittest.TestCase):
    def test_management_branding_and_legacy_frontend_residue(self):
        index = (ROOT_DIR / "index.html").read_text(encoding="utf-8").lower()
        settings = (ROOT_DIR / "settings.html").read_text(encoding="utf-8").lower()
        statistic = (ROOT_DIR / "statistic.html").read_text(encoding="utf-8").lower()
        app = (ROOT_DIR / "js" / "app.js").read_text(encoding="utf-8").lower()
        statistic_js = (ROOT_DIR / "js" / "statistic.js").read_text(encoding="utf-8").lower()

        for text in (index, settings, statistic):
            self.assertIn("daily management ai enhanced", text)
            self.assertNotRegex(text, r"daily arxiv|arxiv papers|astronomy|\bpulsar\b|\bmagnetar\b|\bfrb\b")
        for text in (app, statistic_js):
            self.assertNotRegex(text, r"arxiv|astronomy|\bpulsar\b|\bmagnetar\b|\bfrb\b|radio astronomy")

        self.assertIn("daily-management-ai-enhanced", (ROOT_DIR / "js" / "data-config.js").read_text())
        app_source = (ROOT_DIR / "js" / "app.js").read_text(encoding="utf-8")
        for old_expression in ("replace('abs'", 'replace("abs"', "arxiv.org/abs", "arxiv.org/pdf"):
            self.assertNotIn(old_expression, app_source)
        self.assertNotIn("replace('abs'", (ROOT_DIR / "js" / "statistic.js").read_text(encoding="utf-8"))

    def test_settings_storage_and_six_field_contract_remain(self):
        settings_html = (ROOT_DIR / "settings.html").read_text(encoding="utf-8")
        settings_js = (ROOT_DIR / "js" / "settings.js").read_text(encoding="utf-8")
        structure = (ROOT_DIR / "ai" / "structure.py").read_text(encoding="utf-8")
        for label in ("Interested Keywords", "Interested Authors"):
            self.assertIn(label, settings_html)
        for key in ("preferredKeywords", "preferredAuthors"):
            self.assertIn(f'localStorage.getItem(\'{key}\')', settings_js)
            self.assertIn(f'localStorage.setItem(\'{key}\'', settings_js)
        for field in (
            "abstract_translation",
            "tldr",
            "motivation",
            "method",
            "result",
            "conclusion",
        ):
            self.assertIn(field, structure)

    def test_openalex_link_fixtures_and_modal_labels(self):
        node_script = r"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync('js/app.js', 'utf8');
const elements = {};
function element(id) {
  if (!elements[id]) {
    elements[id] = {
      innerHTML: '', textContent: '', href: '', style: {}, scrollTop: 0,
      classList: { add() {}, remove() {}, contains() { return false; } }
    };
  }
  return elements[id];
}
const document = {
  addEventListener() {},
  body: { style: {} },
  getElementById(id) { return element(id); },
  querySelector() { return element('query'); },
  querySelectorAll() { return []; }
};
const context = {
  console, document, window: {}, Date, URL, URLSearchParams,
  setTimeout, clearTimeout, Math
};
vm.createContext(context);
vm.runInContext(source, context);
function assert(condition, message) { if (!condition) throw new Error(message); }

const landingAndPdf = {
  id: 'openalex:W123', title: 'Management <study>',
  authors: ['A. Author'], categories: ['management'],
  summary: 'Original abstract with evidence.', published: '2026-08-23',
  doi: '10.1000/example.1', journal: 'Management <Review>', source: 'openalex',
  links: {
    landing: 'https://publisher.example/article/1',
    pdf: 'https://publisher.example/article/1.pdf',
    doi: 'https://doi.org/10.1000/example.1',
    openalex: 'https://openalex.org/W123'
  },
  AI: {
    abstract_translation: '中文摘要', tldr: '核心结论',
    motivation: '研究问题与理论背景', method: '数据、样本与研究方法',
    result: '主要结果、机制与异质性', conclusion: '研究贡献与启示'
  }
};
assert(context.getPaperLandingUrl(landingAndPdf) === 'https://publisher.example/article/1', 'landing priority failed');
assert(context.getPaperPdfUrl(landingAndPdf) === 'https://publisher.example/article/1.pdf', 'explicit PDF failed');
assert(context.getPaperDoiUrl(landingAndPdf) === 'https://doi.org/10.1000/example.1', 'DOI mapping failed');
assert(context.getPaperOpenAlexUrl(landingAndPdf) === 'https://openalex.org/W123', 'OpenAlex mapping failed');
assert(!context.getPaperLandingUrl({ id: 'openalex:W404', links: {} }).includes('arxiv'), 'ID created arXiv URL');

const explicitArxivPdf = {
  links: { landing: 'https://publisher.example/article/2', pdf: 'https://arxiv.org/pdf/2608.00001' }
};
assert(context.getPaperPdfUrl(explicitArxivPdf) === 'https://arxiv.org/pdf/2608.00001', 'explicit arXiv PDF was rejected');
assert(context.getPaperPdfUrl({ links: { landing: 'https://publisher.example/article/3', pdf: 'https://publisher.example/article/3/' } }) === '', 'landing/PDF anomaly was not suppressed');
assert(context.getPaperPdfUrl({ links: { doi: 'https://doi.org/10.1000/example.2', pdf: 'https://doi.org/10.1000/example.2/' } }) === '', 'DOI/PDF anomaly was not suppressed');
assert(context.getPaperPdfUrl({ links: { landing: 'https://publisher.example/article/4' } }) === '', 'missing PDF was not suppressed');
assert(context.getPaperLandingUrl({ links: { doi: '10.1000/example.3' } }) === 'https://doi.org/10.1000/example.3', 'raw DOI landing fallback failed');

const parsed = context.parseJsonlData(JSON.stringify(landingAndPdf), '2026-08-24');
const paper = parsed.management[0];
assert(paper.url === landingAndPdf.links.landing, 'canonical landing URL was not preserved');
assert(paper.pdf === '', 'PDF field unexpectedly replaced canonical metadata');
assert(paper.journal === landingAndPdf.journal, 'journal was not preserved');
assert(paper.published === landingAndPdf.published, 'publication date was not preserved');
assert(paper.source === 'openalex', 'source was not preserved');
assert(paper.links.openalex === landingAndPdf.links.openalex, 'OpenAlex URL was not preserved');

const missingMetadata = { id: 'openalex:W404', title: 'No metadata', categories: ['management'], summary: '' };
const missing = context.parseJsonlData(JSON.stringify(missingMetadata), '2026-08-24').management[0];
assert(missing.authors === '' && missing.journal === '' && missing.source_name === '', 'missing metadata was rendered as a label');
context.showPaperDetails(paper, 1);
let html = element('modalBody').innerHTML;
assert(html.includes('核心结论'), 'management TL;DR label missing');
for (const label of ['研究问题与理论背景', '数据、样本与研究方法', '主要结果、机制与异质性', '研究贡献与启示', '中文摘要', '英文摘要']) {
  assert(html.includes(label), `management label missing: ${label}`);
}
assert(html.includes('&lt;Review&gt;') && !html.includes('<Review>'), 'metadata HTML was not escaped');
assert(element('paperLink').href === landingAndPdf.links.landing, 'landing button href failed');
assert(element('pdfLink').href === landingAndPdf.links.pdf, 'PDF button href failed');
assert(element('doiLink').href === landingAndPdf.links.doi, 'DOI button href failed');
assert(element('openAlexLink').href === landingAndPdf.links.openalex, 'OpenAlex button href failed');
assert(html.includes('iframe src="https://publisher.example/article/1.pdf"'), 'PDF preview did not use explicit PDF');

const noPdf = Object.assign({}, paper, { links: Object.assign({}, paper.links, { pdf: '' }), pdf: '' });
context.showPaperDetails(noPdf, 1);
assert(!element('modalBody').innerHTML.includes('<iframe'), 'missing PDF still created preview');
assert(element('pdfLink').style.display === 'none', 'missing PDF button was not hidden');
assert(context.highlightMatches('<img src=x onerror=alert(1)>', [], 'keyword-highlight').includes('&lt;img'), 'XSS escaping failed');
assert(context.renderJournalBadges({}) === '', 'old data without ranking metadata failed');
const badges = context.renderJournalBadges({ journal_rank_sources: ['UTD24', 'FT50', 'AJG4', 'AJG3', 'Preprint', 'priority-S', '<img>'] });
for (const badge of ['UTD24', 'FT50', 'AJG4', 'AJG3', 'Preprint']) {
  assert(badges.includes(`>${badge}<`), `verified badge missing: ${badge}`);
}
assert(!badges.includes('priority-S') && !badges.includes('<img>'), 'internal or unsafe badge leaked');
"""
        completed = subprocess.run(
            ["node", "-e", node_script],
            cwd=ROOT_DIR,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)

    def test_mathjax_and_management_labels_are_present(self):
        index = (ROOT_DIR / "index.html").read_text(encoding="utf-8")
        app = (ROOT_DIR / "js" / "app.js").read_text(encoding="utf-8")
        for label in (
            "核心结论",
            "研究问题与理论背景",
            "数据、样本与研究方法",
            "主要结果、机制与异质性",
            "研究贡献与启示",
            "中文摘要",
            "英文摘要",
        ):
            self.assertIn(label, app)
        self.assertIn("window.MathJax =", index)
        self.assertIn("tex-mml-chtml.js", index)
        self.assertIn("journal_rank_sources", app)
        self.assertNotIn("S_TIER_JOURNALS", app)
        self.assertNotIn("A_TIER_JOURNALS", app)
        self.assertNotIn("B_TIER_JOURNALS", app)


if __name__ == "__main__":
    unittest.main()
