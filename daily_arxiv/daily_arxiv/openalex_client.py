#!/usr/bin/env python3
"""OpenAlex Works client and canonical paper adapter.

The client is deliberately independent from the existing arXiv/SciX code.
It fetches a bounded date window, performs a broad OpenAlex search, then
applies a local title-plus-abstract check before producing the paper shape
used by the existing deduplication and AI pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    import requests
except ImportError:  # pragma: no cover - requests is already used by SciX
    requests = None  # type: ignore[assignment]

try:
    from .source_merge import normalize_doi, write_jsonl
except ImportError:  # pragma: no cover - direct script execution fallback
    from source_merge import normalize_doi, write_jsonl


OPENALEX_API_BASE_URL = "https://api.openalex.org"
OPENALEX_API_URL = f"{OPENALEX_API_BASE_URL}/works"
DEFAULT_PER_PAGE = 100
DEFAULT_MAX_PAGES = 2
DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIES = 3
DEFAULT_OPENALEX_WORK_TYPES: tuple[str, ...] = (
    "article",
    "preprint",
    "review",
    "report",
)

DEFAULT_OPENALEX_SEARCH_TERMS: tuple[str, ...] = (
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

# Short aliases make the configuration and tests easy to discover while the
# explicit name documents that this is the production OpenAlex default.
DEFAULT_SEARCH_TERMS = DEFAULT_OPENALEX_SEARCH_TERMS
OPENALEX_SEARCH_TERMS_ENV = "OPENALEX_SEARCH_TERMS"
OPENALEX_API_KEY_ENV = "OPENALEX_API_KEY"


def parse_openalex_search_terms(raw_terms: str | None) -> tuple[str, ...]:
    """Parse comma-separated terms, preserving first-seen spelling.

    Empty values and case-insensitive duplicates are ignored.  Missing or
    effectively empty configuration falls back to the management defaults.
    """

    if raw_terms is None:
        return DEFAULT_OPENALEX_SEARCH_TERMS

    terms: list[str] = []
    seen: set[str] = set()
    for raw_term in raw_terms.split(","):
        term = raw_term.strip()
        normalized = term.casefold()
        if term and normalized not in seen:
            terms.append(term)
            seen.add(normalized)

    return tuple(terms) if terms else DEFAULT_OPENALEX_SEARCH_TERMS


def _quote_search_term(term: str) -> str:
    """Quote phrases for OpenAlex search while leaving simple terms plain."""

    if re.search(r"\s", term) or '"' in term:
        escaped = term.replace('"', r'\"')
        return f'"{escaped}"'
    return term


def build_openalex_search_query(terms: Sequence[str]) -> str:
    """Build an OR query; transport encoding is left to ``requests``."""

    return " OR ".join(_quote_search_term(term.strip()) for term in terms if term.strip())


def build_openalex_query(terms: Sequence[str]) -> str:
    """Backward-friendly alias for the independent query builder."""

    return build_openalex_search_query(terms)


def reconstruct_abstract(index: Any) -> str:
    """Reconstruct OpenAlex ``abstract_inverted_index`` into plain text.

    A malformed index is treated as unavailable.  Multiple positions for the
    same word are supported, and equal positions retain input order.
    """

    if not isinstance(index, Mapping) or not index:
        return ""

    positioned_words: list[tuple[int, int, str]] = []
    order = 0
    for word, positions in index.items():
        if not isinstance(word, str) or not word.strip():
            return ""
        if not isinstance(positions, (list, tuple)):
            return ""
        for position in positions:
            if isinstance(position, bool) or not isinstance(position, int) or position < 0:
                return ""
            positioned_words.append((position, order, word))
            order += 1

    if not positioned_words:
        return ""
    positioned_words.sort(key=lambda item: (item[0], item[1]))
    return " ".join(word for _, _, word in positioned_words).strip()


def normalize_openalex_id(value: Any) -> str | None:
    """Return the canonical ``W...`` identifier from common OpenAlex forms."""

    if not isinstance(value, str):
        return None
    text = value.strip().rstrip("/")
    lower_text = text.casefold()
    for prefix in (
        "https://openalex.org/",
        "http://openalex.org/",
        "openalex:",
    ):
        if lower_text.startswith(prefix):
            text = text[len(prefix) :]
            break
    if not re.fullmatch(r"(?i)W\d+", text):
        return None
    return "W" + text[1:]


def _clean_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    return value if value.startswith(("https://", "http://")) else ""


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _authors_from_work(work: Mapping[str, Any]) -> list[str]:
    authorships = work.get("authorships")
    if not isinstance(authorships, list):
        return []

    authors: list[str] = []
    for authorship in authorships:
        if not isinstance(authorship, Mapping):
            continue
        author = authorship.get("author")
        if not isinstance(author, Mapping):
            continue
        name = _text(author.get("display_name"))
        if name:
            authors.append(name)
    return authors


def _topic_names(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    names: list[str] = []
    for topic in value:
        if not isinstance(topic, Mapping):
            continue
        name = _text(topic.get("display_name"))
        if name:
            names.append(name)
    return names


def openalex_work_to_paper(
    work: Mapping[str, Any],
    *,
    summary: str | None = None,
) -> dict[str, Any]:
    """Map one OpenAlex Work to the existing canonical paper shape."""

    openalex_id = normalize_openalex_id(work.get("id"))
    if not openalex_id:
        raise ValueError("OpenAlex work has no valid id")

    work_url = _clean_url(work.get("id")) or f"{OPENALEX_API_BASE_URL}/{openalex_id}"
    primary_location = work.get("primary_location")
    if not isinstance(primary_location, Mapping):
        primary_location = {}
    primary_source = primary_location.get("source")
    if not isinstance(primary_source, Mapping):
        primary_source = {}

    landing_url = _clean_url(primary_location.get("landing_page_url"))
    doi = normalize_doi(work.get("doi")) or ""
    doi_url = f"https://doi.org/{doi}" if doi else ""
    abstract_url = landing_url or doi_url or work_url

    best_oa_location = work.get("best_oa_location")
    if not isinstance(best_oa_location, Mapping):
        best_oa_location = {}
    pdf_url = _clean_url(best_oa_location.get("pdf_url"))
    if not pdf_url:
        pdf_url = _clean_url(primary_location.get("pdf_url"))

    topic_names = _topic_names(work.get("topics"))
    primary_topic = work.get("primary_topic")
    if isinstance(primary_topic, Mapping):
        primary_topic_name = _text(primary_topic.get("display_name"))
    else:
        primary_topic_name = _text(primary_topic)

    title = _text(work.get("title")) or _text(work.get("display_name"))
    reconstructed = reconstruct_abstract(work.get("abstract_inverted_index"))
    paper_summary = reconstructed if summary is None else _text(summary)

    return {
        "id": f"openalex:{openalex_id}",
        "title": title,
        "summary": paper_summary,
        "authors": _authors_from_work(work),
        "categories": ["management"],
        "abs": abstract_url,
        "pdf": pdf_url,
        "comment": "",
        "source": "openalex",
        "doi": doi,
        "journal": _text(primary_source.get("display_name")),
        "published": _text(work.get("publication_date")),
        "openalex_id": openalex_id,
        "work_type": _text(work.get("type")),
        "language": _text(work.get("language")),
        "primary_topic": primary_topic_name,
        "topics": topic_names,
        "source_name": _text(primary_source.get("display_name")),
        "links": {
            "landing": landing_url,
            "pdf": pdf_url,
            "doi": doi_url,
            "openalex": work_url,
        },
    }


def matches_openalex_search_terms(
    title: Any,
    abstract: Any,
    terms: Sequence[str],
) -> bool:
    """Apply the local case-insensitive OR candidate check."""

    searchable = f"{_text(title)}\n{_text(abstract)}".casefold()
    return any(term.strip().casefold() in searchable for term in terms if term.strip())


@dataclass
class OpenAlexFetchResult:
    works: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    start_date: str = ""
    end_date: str = ""
    pages: int = 0
    truncated: bool = False
    error: str = ""
    candidates: list[dict[str, Any]] = field(default_factory=list)
    missing_abstract_count: int = 0
    local_term_rejected_count: int = 0

    @property
    def works_received(self) -> int:
        return len(self.works)

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)


class OpenAlexClient:
    """Bounded, injectable OpenAlex Works API client."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        search_terms: Sequence[str] | None = None,
        session: Any | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_pages: int = DEFAULT_MAX_PAGES,
        per_page: int = DEFAULT_PER_PAGE,
        retries: int = DEFAULT_RETRIES,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_pages < 1:
            raise ValueError("max_pages must be at least 1")
        if per_page < 1:
            raise ValueError("per_page must be at least 1")
        if retries < 0:
            raise ValueError("retries must not be negative")

        if api_key is None:
            api_key = os.environ.get(OPENALEX_API_KEY_ENV)
        if search_terms is None:
            search_terms = parse_openalex_search_terms(
                os.environ.get(OPENALEX_SEARCH_TERMS_ENV)
            )

        self.api_key = (api_key or "").strip()
        self.search_terms = tuple(search_terms or DEFAULT_OPENALEX_SEARCH_TERMS)
        self.timeout = timeout
        self.max_pages = max_pages
        self.per_page = per_page
        self.retries = retries
        self.sleep_fn = sleep_fn
        if session is not None:
            self.session = session
        elif requests is not None:
            self.session = requests.Session()
        else:  # pragma: no cover - project dependencies include requests
            self.session = None

        if not self.api_key:
            print(
                "OpenAlex API key not configured; anonymous quota will be used",
                file=sys.stderr,
            )

    def _retry_delay(self, response: Any, attempt: int) -> float:
        headers = getattr(response, "headers", {})
        retry_after = headers.get("Retry-After") if isinstance(headers, Mapping) else None
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            return float(2**attempt)

    def _get_page(self, params: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        if self.session is None:
            return None, "requests transport is unavailable"

        for attempt in range(self.retries + 1):
            try:
                response = self.session.get(
                    OPENALEX_API_URL,
                    params=dict(params),
                    timeout=self.timeout,
                )
            except Exception as exc:  # network boundary; never expose request data
                retryable = isinstance(exc, (TimeoutError, OSError))
                if requests is not None:
                    retryable = retryable or isinstance(
                        exc,
                        (
                            requests.exceptions.Timeout,
                            requests.exceptions.RequestException,
                        ),
                    )
                if retryable and attempt < self.retries:
                    self.sleep_fn(float(2**attempt))
                    continue
                kind = "timeout" if isinstance(exc, (TimeoutError,)) else "network error"
                return None, f"{kind}: {type(exc).__name__}"

            status_code = getattr(response, "status_code", None)
            if status_code == 429 or isinstance(status_code, int) and 500 <= status_code <= 599:
                if attempt < self.retries:
                    self.sleep_fn(self._retry_delay(response, attempt))
                    continue
                return None, f"HTTP {status_code} after retries"

            if not isinstance(status_code, int) or not 200 <= status_code < 300:
                return None, f"HTTP {status_code}"

            try:
                payload = response.json()
            except (ValueError, json.JSONDecodeError):
                return None, "malformed JSON response"
            except Exception:
                return None, "malformed JSON response"

            if not isinstance(payload, Mapping):
                return None, "malformed JSON response"
            return dict(payload), None

        return None, "request failed"  # pragma: no cover

    def _request_params(self, start_date: str, end_date: str, cursor: str) -> dict[str, Any]:
        params: dict[str, Any] = {
            "search": build_openalex_search_query(self.search_terms),
            "filter": (
                f"from_publication_date:{start_date},"
                f"to_publication_date:{end_date},has_abstract:true,"
                f"type:{'|'.join(DEFAULT_OPENALEX_WORK_TYPES)}"
            ),
            "per-page": self.per_page,
            "cursor": cursor,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def fetch(self, start_date: str, end_date: str) -> OpenAlexFetchResult:
        """Fetch bounded raw OpenAlex works for an explicit date window."""

        result = OpenAlexFetchResult(start_date=start_date, end_date=end_date)
        cursor = "*"
        seen_cursors: set[str] = set()

        while True:
            if cursor in seen_cursors:
                result.status = "error"
                result.error = "cursor pagination repeated a cursor"
                return result
            seen_cursors.add(cursor)

            payload, error = self._get_page(self._request_params(start_date, end_date, cursor))
            if error:
                result.status = "error"
                result.error = error
                return result

            assert payload is not None
            results = payload.get("results")
            meta = payload.get("meta")
            if not isinstance(results, list) or not isinstance(meta, Mapping):
                result.status = "error"
                result.error = "malformed OpenAlex response shape"
                return result
            if any(not isinstance(work, Mapping) for work in results):
                result.status = "error"
                result.error = "malformed OpenAlex work record"
                return result

            result.pages += 1
            result.works.extend(dict(work) for work in results)
            if not results:
                break

            next_cursor = meta.get("next_cursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                break
            if result.pages >= self.max_pages:
                result.status = "truncated"
                result.truncated = True
                if self.max_pages == DEFAULT_MAX_PAGES:
                    print(
                        f"OpenAlex production page cap reached: {self.max_pages}; "
                        "status=truncated",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"OpenAlex pagination reached max_pages={self.max_pages}; "
                        "status=truncated",
                        file=sys.stderr,
                    )
                break
            cursor = next_cursor

        return result

    def fetch_candidates(self, start_date: str, end_date: str) -> OpenAlexFetchResult:
        """Fetch works and apply the local title-plus-abstract candidate check."""

        result = self.fetch(start_date, end_date)
        if result.status == "error":
            return result

        for work in result.works:
            summary = reconstruct_abstract(work.get("abstract_inverted_index"))
            if not summary:
                result.missing_abstract_count += 1
                continue
            if not matches_openalex_search_terms(work.get("title") or work.get("display_name"), summary, self.search_terms):
                result.local_term_rejected_count += 1
                continue
            try:
                result.candidates.append(openalex_work_to_paper(work, summary=summary))
            except ValueError:
                result.local_term_rejected_count += 1

        return result


def status_payload(result: OpenAlexFetchResult, terms: Sequence[str]) -> dict[str, Any]:
    """Return the stable status JSON shape written beside production data."""

    return {
        "status": result.status,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "terms_count": len(terms),
        "pages": result.pages,
        "works_received": result.works_received,
        "missing_abstract_count": result.missing_abstract_count,
        "local_term_rejected_count": result.local_term_rejected_count,
        "candidate_count": result.candidate_count,
    }


def _write_status(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch OpenAlex management candidates")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", required=True, help="Canonical candidate JSONL path")
    parser.add_argument("--status-file", required=True, help="OpenAlex status JSON path")
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    terms = parse_openalex_search_terms(os.environ.get(OPENALEX_SEARCH_TERMS_ENV))
    print(f"OpenAlex search terms ({len(terms)}): {', '.join(terms)}")
    print(f"OpenAlex date window: {args.start_date} to {args.end_date}")

    client = OpenAlexClient(
        api_key=os.environ.get(OPENALEX_API_KEY_ENV),
        search_terms=terms,
        max_pages=args.max_pages,
        timeout=args.timeout,
        retries=args.retries,
    )
    result = client.fetch_candidates(args.start_date, args.end_date)
    payload = status_payload(result, terms)
    _write_status(args.status_file, payload)

    if result.status == "error":
        write_jsonl([], args.output)
        print(f"OpenAlex error: {result.error}", file=sys.stderr)
    else:
        write_jsonl(result.candidates, args.output)

    print(f"OpenAlex status: {result.status}")
    print(f"OpenAlex pages fetched: {result.pages}")
    print(f"OpenAlex works received: {result.works_received}")
    print(f"OpenAlex missing abstract: {result.missing_abstract_count}")
    print(
        "OpenAlex rejected by local title+abstract candidate check: "
        f"{result.local_term_rejected_count}"
    )
    print(f"OpenAlex canonical candidates: {result.candidate_count}")
    return 1 if result.status == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
