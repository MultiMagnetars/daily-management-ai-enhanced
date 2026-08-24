import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from daily_arxiv.daily_arxiv.openalex_client import (
    DEFAULT_MAX_PAGES,
    DEFAULT_OPENALEX_SEARCH_TERMS,
    DEFAULT_OPENALEX_WORK_TYPES,
    DEFAULT_PER_PAGE,
    OPENALEX_API_URL,
    OpenAlexClient,
    OpenAlexFetchResult,
    build_openalex_search_query,
    main as openalex_main,
    matches_openalex_search_terms,
    normalize_openalex_id,
    openalex_work_to_paper,
    parse_openalex_search_terms,
    reconstruct_abstract,
    status_payload,
)
from daily_arxiv.daily_arxiv.source_merge import history_keys


ROOT_DIR = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT_DIR / ".github" / "workflows" / "run.yml"


class FakeResponse:
    def __init__(self, status_code, payload=None, headers=None, json_error=None):
        self.status_code = status_code
        self.payload = payload
        self.headers = headers or {}
        self.json_error = json_error

    def json(self):
        if self.json_error:
            raise self.json_error
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def abstract_index():
    return {
        "Corporate": [0],
        "governance": [1],
        "improves": [2],
        "financial": [3],
        "reporting": [4],
    }


def openalex_work(
    *,
    work_id="W123456789",
    title="Corporate governance improves financial reporting",
    abstract=None,
    doi="https://doi.org/10.1234/Example.1",
    landing="https://publisher.example/landing",
    primary_pdf="https://publisher.example/primary.pdf",
    best_pdf="https://publisher.example/best.pdf",
):
    return {
        "id": f"https://openalex.org/{work_id}",
        "title": title,
        "display_name": title,
        "abstract_inverted_index": abstract if abstract is not None else abstract_index(),
        "authorships": [
            {"author": {"display_name": "Alice Author"}},
            {"author": {"display_name": "Bob Researcher"}},
        ],
        "doi": doi,
        "primary_location": {
            "landing_page_url": landing,
            "pdf_url": primary_pdf,
            "source": {"display_name": "Management Science"},
        },
        "best_oa_location": {"pdf_url": best_pdf},
        "publication_date": "2026-08-22",
        "type": "article",
        "language": "en",
        "primary_topic": {"display_name": "Corporate governance"},
        "topics": [{"display_name": "Corporate governance"}, {"display_name": "Finance"}],
    }


def page(works, next_cursor=None):
    return FakeResponse(
        200,
        {
            "meta": {"next_cursor": next_cursor},
            "results": works,
        },
    )


class OpenAlexIngestTests(unittest.TestCase):
    def test_default_search_terms_are_management_terms(self):
        expected = (
            "accounting",
            "auditing",
            "financial reporting",
            "earnings management",
            "corporate finance",
            "corporate governance",
            "capital market",
            "ESG",
            "corporate social responsibility",
            "sustainable finance",
            "green finance",
            "climate finance",
            "corporate disclosure",
            "environmental disclosure",
            "institutional investor",
            "digital transformation",
            "digital economy",
            "fintech",
            "data asset",
            "supply chain",
            "supply chain resilience",
            "green innovation",
            "environmental governance",
            "information spillover",
            "disclosure spillover",
            "peer disclosure",
            "information externality",
        )
        self.assertEqual(expected, DEFAULT_OPENALEX_SEARCH_TERMS)
        self.assertEqual(DEFAULT_OPENALEX_SEARCH_TERMS, parse_openalex_search_terms(None))
        self.assertIn("information spillover", DEFAULT_OPENALEX_SEARCH_TERMS)
        self.assertIn("disclosure spillover", DEFAULT_OPENALEX_SEARCH_TERMS)
        self.assertIn("peer disclosure", DEFAULT_OPENALEX_SEARCH_TERMS)
        self.assertIn("information externality", DEFAULT_OPENALEX_SEARCH_TERMS)
        self.assertNotIn("sustainability", DEFAULT_OPENALEX_SEARCH_TERMS)
        self.assertNotIn("artificial intelligence", DEFAULT_OPENALEX_SEARCH_TERMS)

    def test_new_disclosure_terms_are_quoted_in_openalex_query(self):
        query = build_openalex_search_query(
            (
                "environmental disclosure",
                "information spillover",
                "peer disclosure",
                "information externality",
            )
        )
        self.assertEqual(
            '"environmental disclosure" OR "information spillover" OR '
            '"peer disclosure" OR "information externality"',
            query,
        )

    def test_empty_env_falls_back_to_defaults(self):
        self.assertEqual(DEFAULT_OPENALEX_SEARCH_TERMS, parse_openalex_search_terms(""))
        self.assertEqual(DEFAULT_OPENALEX_SEARCH_TERMS, parse_openalex_search_terms(" , , "))

    def test_terms_trim_empty_and_case_insensitive_dedup_preserves_first_spelling(self):
        self.assertEqual(
            ("ESG",),
            parse_openalex_search_terms(" ESG ")[:1],
        )
        self.assertEqual(
            ("Accounting", "financial reporting"),
            parse_openalex_search_terms(" Accounting, accounting, , financial reporting "),
        )

    def test_query_uses_or_and_quotes_phrases(self):
        query = build_openalex_search_query(("accounting", "financial reporting", "ESG"))
        self.assertEqual('accounting OR "financial reporting" OR ESG', query)

    def test_query_builder_does_not_encode_transport_parameters(self):
        query = build_openalex_search_query(("corporate finance",))
        self.assertNotIn("%20", query)
        self.assertEqual('"corporate finance"', query)

    def test_date_filter_per_page_and_cursor_are_sent_via_params(self):
        session = FakeSession([page([])])
        client = OpenAlexClient(
            api_key="secret-key",
            search_terms=("accounting",),
            session=session,
        )
        result = client.fetch("2026-08-20", "2026-08-23")
        self.assertEqual("ok", result.status)
        url, kwargs = session.calls[0]
        self.assertEqual(OPENALEX_API_URL, url)
        self.assertEqual("*", kwargs["params"]["cursor"])
        self.assertEqual(DEFAULT_PER_PAGE, kwargs["params"]["per-page"])
        self.assertEqual(
            "from_publication_date:2026-08-20,to_publication_date:2026-08-23,"
            "has_abstract:true,type:article|preprint|review|report",
            kwargs["params"]["filter"],
        )
        self.assertEqual("secret-key", kwargs["params"]["api_key"])

    def test_production_page_cap_and_research_work_types(self):
        self.assertEqual(2, DEFAULT_MAX_PAGES)
        self.assertEqual(
            ("article", "preprint", "review", "report"),
            DEFAULT_OPENALEX_WORK_TYPES,
        )
        for excluded in (
            "software",
            "dataset",
            "editorial",
            "peer-review",
            "dissertation",
            "book",
            "book-chapter",
            "other",
            "conference-paper",
        ):
            self.assertNotIn(excluded, DEFAULT_OPENALEX_WORK_TYPES)

    def test_cursor_pagination_until_next_cursor_is_empty(self):
        session = FakeSession(
            [
                page([openalex_work(work_id="W1")], next_cursor="cursor-2"),
                page([openalex_work(work_id="W2")], next_cursor=None),
            ]
        )
        result = OpenAlexClient(api_key="key", session=session).fetch(
            "2026-08-20", "2026-08-23"
        )
        self.assertEqual("ok", result.status)
        self.assertEqual(2, result.pages)
        self.assertEqual(2, result.works_received)
        self.assertEqual("cursor-2", session.calls[1][1]["params"]["cursor"])

    def test_max_pages_is_bounded_and_marked_truncated(self):
        session = FakeSession(
            [
                page([openalex_work(work_id="W1")], next_cursor="cursor-2"),
                page([openalex_work(work_id="W2")], next_cursor="cursor-3"),
            ]
        )
        result = OpenAlexClient(
            api_key="key", session=session, max_pages=2, sleep_fn=lambda _: None
        ).fetch("2026-08-20", "2026-08-23")
        self.assertEqual("truncated", result.status)
        self.assertTrue(result.truncated)
        self.assertEqual(2, result.pages)
        self.assertEqual(2, len(session.calls))

    def test_reconstruct_abstract_orders_positions_and_supports_repeated_words(self):
        self.assertEqual(
            "Corporate governance governance improves",
            reconstruct_abstract(
                {"Corporate": [0], "governance": [1, 2], "improves": [3]}
            ),
        )

    def test_malformed_abstract_returns_empty(self):
        for value in (None, {}, {"word": "not-a-list"}, {"word": [-1]}, {1: [0]}):
            with self.subTest(value=value):
                self.assertEqual("", reconstruct_abstract(value))

    def test_openalex_id_normalization(self):
        self.assertEqual("W123", normalize_openalex_id("W123"))
        self.assertEqual("W123", normalize_openalex_id("openalex:W123"))
        self.assertEqual("W123", normalize_openalex_id("https://openalex.org/W123"))
        self.assertIsNone(normalize_openalex_id("https://example.org/W123"))
        self.assertIsNone(normalize_openalex_id("not-an-openalex-id"))

    def test_canonical_mapping_extracts_authors_doi_journal_and_metadata(self):
        paper = openalex_work_to_paper(openalex_work())
        self.assertEqual("openalex:W123456789", paper["id"])
        self.assertEqual("openalex", paper["source"])
        self.assertEqual(["Alice Author", "Bob Researcher"], paper["authors"])
        self.assertEqual("10.1234/example.1", paper["doi"])
        self.assertEqual("Management Science", paper["journal"])
        self.assertEqual(["management"], paper["categories"])
        self.assertEqual("W123456789", paper["openalex_id"])
        self.assertEqual("Corporate governance", paper["primary_topic"])

    def test_landing_url_is_preferred_for_abs(self):
        paper = openalex_work_to_paper(openalex_work(landing="https://example.org/landing"))
        self.assertEqual("https://example.org/landing", paper["abs"])
        self.assertEqual("https://example.org/landing", paper["links"]["landing"])

    def test_doi_then_openalex_url_are_abs_fallbacks(self):
        no_landing = openalex_work(landing="", primary_pdf="", best_pdf="")
        paper = openalex_work_to_paper(no_landing)
        self.assertEqual("https://doi.org/10.1234/example.1", paper["abs"])
        no_doi = openalex_work(landing="", doi="", primary_pdf="", best_pdf="")
        paper = openalex_work_to_paper(no_doi)
        self.assertEqual("https://openalex.org/W123456789", paper["abs"])

    def test_pdf_prefers_best_oa_then_primary_then_empty(self):
        paper = openalex_work_to_paper(openalex_work())
        self.assertEqual("https://publisher.example/best.pdf", paper["pdf"])
        paper = openalex_work_to_paper(openalex_work(best_pdf=""))
        self.assertEqual("https://publisher.example/primary.pdf", paper["pdf"])
        paper = openalex_work_to_paper(
            openalex_work(best_pdf="", primary_pdf="", landing="")
        )
        self.assertEqual("", paper["pdf"])

    def test_openalex_mapping_never_fabricates_arxiv_links(self):
        paper = openalex_work_to_paper(openalex_work(landing="", best_pdf="", primary_pdf=""))
        self.assertNotIn("arxiv.org", paper["abs"])
        self.assertNotIn("arxiv.org", paper["pdf"])

    def test_local_candidate_check_is_case_insensitive_or_substring(self):
        self.assertTrue(
            matches_openalex_search_terms(
                "A Study of ESG Controls", "An unrelated abstract", ("accounting", "esg")
            )
        )
        self.assertFalse(
            matches_openalex_search_terms("A Study", "No configured phrase here", ("ESG",))
        )

    def test_fulltext_only_false_positive_is_rejected_locally(self):
        work = openalex_work(
            title="A study of operations",
            abstract={"The": [0], "paper": [1], "is": [2], "useful": [3]},
        )
        result = OpenAlexClient(
            api_key="key", search_terms=("accounting",), session=FakeSession([page([work])])
        ).fetch_candidates("2026-08-20", "2026-08-23")
        self.assertEqual("ok", result.status)
        self.assertEqual([], result.candidates)
        self.assertEqual(1, result.local_term_rejected_count)

    def test_missing_abstract_is_not_an_ai_candidate(self):
        work = openalex_work(abstract=None)
        work["abstract_inverted_index"] = None
        result = OpenAlexClient(
            api_key="key", search_terms=("accounting",), session=FakeSession([page([work])])
        ).fetch_candidates("2026-08-20", "2026-08-23")
        self.assertEqual(1, result.missing_abstract_count)
        self.assertEqual([], result.candidates)

    def test_same_doi_is_compatible_with_existing_history_keys(self):
        first = openalex_work_to_paper(openalex_work(work_id="W1"))
        second = openalex_work_to_paper(openalex_work(work_id="W2"))
        self.assertIn("doi:10.1234/example.1", history_keys(first))
        self.assertTrue(history_keys(first).intersection(history_keys(second)))

    def test_429_retries_and_respects_retry_after(self):
        delays = []
        session = FakeSession([FakeResponse(429, headers={"Retry-After": "7"}), page([])])
        result = OpenAlexClient(
            api_key="key", session=session, retries=1, sleep_fn=delays.append
        ).fetch("2026-08-20", "2026-08-23")
        self.assertEqual("ok", result.status)
        self.assertEqual([7.0], delays)
        self.assertEqual(2, len(session.calls))

    def test_5xx_retries_with_exponential_backoff(self):
        delays = []
        session = FakeSession([FakeResponse(503), page([])])
        result = OpenAlexClient(
            api_key="key", session=session, retries=1, sleep_fn=delays.append
        ).fetch("2026-08-20", "2026-08-23")
        self.assertEqual("ok", result.status)
        self.assertEqual([1.0], delays)

    def test_timeout_retries(self):
        delays = []
        session = FakeSession([TimeoutError("timeout"), page([])])
        result = OpenAlexClient(
            api_key="key", session=session, retries=1, sleep_fn=delays.append
        ).fetch("2026-08-20", "2026-08-23")
        self.assertEqual("ok", result.status)
        self.assertEqual([1.0], delays)

    def test_client_error_fails_closed_without_retry(self):
        session = FakeSession([FakeResponse(403)])
        result = OpenAlexClient(
            api_key="key", session=session, retries=3, sleep_fn=lambda _: None
        ).fetch("2026-08-20", "2026-08-23")
        self.assertEqual("error", result.status)
        self.assertIn("HTTP 403", result.error)
        self.assertEqual(1, len(session.calls))

    def test_malformed_json_fails_closed(self):
        session = FakeSession([FakeResponse(200, json_error=ValueError("bad json"))])
        result = OpenAlexClient(api_key="key", session=session).fetch(
            "2026-08-20", "2026-08-23"
        )
        self.assertEqual("error", result.status)
        self.assertEqual("malformed JSON response", result.error)

    def test_malformed_response_shape_fails_closed(self):
        session = FakeSession([FakeResponse(200, {"results": "not-a-list", "meta": {}})])
        result = OpenAlexClient(api_key="key", session=session).fetch(
            "2026-08-20", "2026-08-23"
        )
        self.assertEqual("error", result.status)
        self.assertIn("malformed OpenAlex response shape", result.error)

    def test_missing_api_key_warning_does_not_expose_a_value(self):
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            OpenAlexClient(api_key=None, session=FakeSession([]))
        self.assertIn("OpenAlex API key not configured; anonymous quota will be used", stream.getvalue())
        self.assertNotIn("api_key=", stream.getvalue())

    def test_client_reads_api_key_and_terms_from_environment(self):
        session = FakeSession([page([])])
        with patch.dict(
            os.environ,
            {
                "OPENALEX_API_KEY": "env-secret",
                "OPENALEX_SEARCH_TERMS": "Accounting, ESG",
            },
            clear=False,
        ):
            OpenAlexClient(session=session).fetch("2026-08-20", "2026-08-23")
        params = session.calls[0][1]["params"]
        self.assertEqual("env-secret", params["api_key"])
        self.assertEqual('Accounting OR ESG', params["search"])

    def test_status_json_has_required_fields(self):
        result = OpenAlexFetchResult(
            status="truncated",
            start_date="2026-08-20",
            end_date="2026-08-23",
            pages=10,
            works=[openalex_work()],
            candidates=[{"id": "openalex:W1"}],
            missing_abstract_count=2,
            local_term_rejected_count=3,
        )
        payload = status_payload(result, ("accounting", "ESG"))
        self.assertEqual(
            {
                "status",
                "start_date",
                "end_date",
                "terms_count",
                "pages",
                "works_received",
                "missing_abstract_count",
                "local_term_rejected_count",
                "candidate_count",
            },
            set(payload),
        )
        self.assertEqual(1, payload["works_received"])
        self.assertEqual(1, payload["candidate_count"])

    def test_cli_writes_canonical_jsonl_and_status_without_network(self):
        candidate = {"id": "openalex:W1", "source": "openalex"}
        fake_result = OpenAlexFetchResult(
            status="ok",
            start_date="2026-08-20",
            end_date="2026-08-23",
            pages=1,
            works=[openalex_work()],
            candidates=[candidate],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "papers.jsonl"
            status = Path(temp_dir) / "status.json"
            with patch(
                "daily_arxiv.daily_arxiv.openalex_client.OpenAlexClient"
            ) as client_class:
                client_class.return_value.fetch_candidates.return_value = fake_result
                with patch.dict(
                    os.environ,
                    {"OPENALEX_SEARCH_TERMS": "Accounting, accounting, ESG"},
                    clear=False,
                ):
                    exit_code = openalex_main(
                        [
                            "--start-date",
                            "2026-08-20",
                            "--end-date",
                            "2026-08-23",
                            "--output",
                            str(output),
                            "--status-file",
                            str(status),
                            "--max-pages",
                            "5",
                        ]
                    )
            self.assertEqual(0, exit_code)
            self.assertEqual(5, client_class.call_args.kwargs["max_pages"])
            self.assertEqual(candidate, json.loads(output.read_text(encoding="utf-8")))
            self.assertEqual("ok", json.loads(status.read_text(encoding="utf-8"))["status"])

    def test_cli_truncated_result_is_successful_bounded_exit(self):
        fake_result = OpenAlexFetchResult(
            status="truncated",
            start_date="2026-08-20",
            end_date="2026-08-23",
            pages=2,
            works=[openalex_work()],
            candidates=[{"id": "openalex:W1", "source": "openalex"}],
            truncated=True,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "papers.jsonl"
            status = Path(temp_dir) / "status.json"
            with patch(
                "daily_arxiv.daily_arxiv.openalex_client.OpenAlexClient"
            ) as client_class:
                client_class.return_value.fetch_candidates.return_value = fake_result
                exit_code = openalex_main(
                    [
                        "--start-date",
                        "2026-08-20",
                        "--end-date",
                        "2026-08-23",
                        "--output",
                        str(output),
                        "--status-file",
                        str(status),
                        "--max-pages",
                        "2",
                    ]
                )
            self.assertEqual(0, exit_code)
            self.assertEqual("truncated", json.loads(status.read_text(encoding="utf-8"))["status"])

    def test_normal_workflow_uses_openalex_only_and_keeps_downstream(self):
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        build_block = workflow.split("  build:", 1)[1]
        self.assertIn("Fetch OpenAlex candidates", build_block)
        self.assertIn("openalex_client", build_block)
        self.assertIn("OPENALEX_SEARCH_TERMS", build_block)
        self.assertIn("OPENALEX_API_KEY", build_block)
        self.assertNotIn("Crawl arXiv papers", build_block)
        self.assertNotIn("scrapy crawl arxiv", build_block)
        self.assertNotIn("Fetch and merge SciX candidates", build_block)
        self.assertNotIn("scix_client", build_block)
        self.assertNotIn("source_merge", build_block)
        self.assertIn("id: history_step", build_block)
        self.assertIn("id: dedup_check", build_block)
        self.assertIn("id: same_day_step", build_block)
        self.assertIn("same_day_merge subtract", build_block)
        self.assertIn("FILTER_KEYWORDS", build_block)
        self.assertIn("same_day_merge merge", build_block)

    def test_workflow_keeps_manual_dispatch_and_no_schedule(self):
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("validation_mode:", workflow)
        self.assertNotIn("schedule:", workflow)
        self.assertNotIn('cron: "30 17 * * *"', workflow)

    def test_workflow_keeps_same_day_and_filter_wiring_unchanged(self):
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        self.assertIn('python -m daily_arxiv.daily_arxiv.same_day_merge subtract', workflow)
        self.assertIn('python -m daily_arxiv.daily_arxiv.same_day_merge merge', workflow)
        self.assertIn('--history-language "$language"', workflow)
        self.assertIn('export FILTER_KEYWORDS="${{ vars.FILTER_KEYWORDS }}"', workflow)
        self.assertIn('MAX_AI_PAPERS_PER_RUN: ${{ vars.MAX_AI_PAPERS_PER_RUN }}', workflow)
        self.assertIn('python enhance.py --data ../data/${today}_new.jsonl', workflow)


if __name__ == "__main__":
    unittest.main()
