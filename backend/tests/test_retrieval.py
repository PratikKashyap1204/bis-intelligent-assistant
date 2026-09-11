"""
Tests for the Milestone 1 retrieval layer.

Synthetic sample data only (IS 9999 / example.invalid) — same pattern as
test_ingestion_integration.py. Automated tests do not touch the live
bis_db pilot rows and do not call official BIS websites.
"""

from __future__ import annotations

from app.services.bis_ingestion import (
    BISIngestionService,
    RawClause,
    RawDocument,
    RawStandard,
)
from app.services.retrieval import (
    BISRetrievalService,
    KeywordRetrievalBackend,
    RelevanceInfo,
    RetrievalResult,
    SearchFilters,
    _score_fields,
    tokenize_query,
)


SAMPLE_STANDARD = RawStandard(
    is_number="IS 9999",
    title="Household Electric Appliance Safety Requirements",
    year=2099,
    status="ACTIVE",
    scope="Safety requirements for household electric appliances.",
    source_url="https://example.invalid/sample-standard",
)

SAMPLE_DOCUMENT = RawDocument(
    title="Sample Household Appliance Circular",
    document_type="OTHER",
    source_url="https://example.invalid/sample-document.pdf",
    standard_is_number="IS 9999",
    content_hash="cafebabe" * 8,
    language="en",
)

SAMPLE_CLAUSES = [
    RawClause(
        clause_number="4.1",
        title="Earthing",
        content="Earthing requirements for household electric appliances.",
        page_number=3,
        language="en",
        clause_type="CLAUSE",
    ),
    RawClause(
        clause_number="5",
        title="Marking",
        content="Packaging labels shall be printed in English only.",
        page_number=4,
        language="en",
        clause_type="CLAUSE",
    ),
    RawClause(
        clause_number="Annex A",
        title="List of appliances",
        content="Toasters and kettles are listed here.",
        page_number=8,
        language="en",
        clause_type="ANNEX",
    ),
]


def _seed(db_session):
    ingestion = BISIngestionService(db_session)
    standard, _ = ingestion.ingest_standard(SAMPLE_STANDARD)
    document, _, _ = ingestion.ingest_document(
        SAMPLE_DOCUMENT, SAMPLE_CLAUSES, standard=standard
    )
    db_session.commit()
    return standard, document


# ---------------------------------------------------------------------------
# Pure unit tests (no database)
# ---------------------------------------------------------------------------

def test_tokenize_query_drops_stopwords_keeps_is_number_tokens():
    assert tokenize_query("What are the requirements for household electric appliances?") == [
        "requirements",
        "household",
        "electric",
        "appliances",
    ]
    assert tokenize_query("IS 302") == ["is", "302"]
    assert tokenize_query("") == []
    assert tokenize_query("   ") == []


# ---------------------------------------------------------------------------
# Milestone 5: keyword scoring fixes discovered via the retrieval
# evaluation (data/eval/retrieval_eval_dataset.json, questions q19/q21/
# q23-26) — see app/services/retrieval.py's _STOPWORDS/"who" and
# _LOW_SIGNAL_SCORING_TERMS/"is" comments for the full rationale.
# ---------------------------------------------------------------------------


def test_tokenize_query_drops_who_as_stopword():
    # Real evaluation failure: "Who won the FIFA World Cup in 2022?" (an
    # out-of-scope question) matched real "Any person WHO contravenes..."
    # clauses purely via the token "who".
    assert tokenize_query("Who is the certifying authority?") == ["is", "certifying", "authority"]
    assert "who" not in tokenize_query("Who won the FIFA World Cup in 2022?")


def test_score_fields_ignores_bare_is_token():
    # "is" must still be a token (see IS-302-style query tests above) but
    # must not, on its own, cause a field to be counted as matched —
    # otherwise every is_number ("IS ...") "matches" every query that
    # merely contains the word "is" (i.e. almost all English questions).
    score, matched = _score_fields({"is_number": "IS 9999"}, ["is"])
    assert score == 0.0
    assert matched == []

    # A real, higher-signal term still scores normally alongside "is".
    score, matched = _score_fields({"is_number": "IS 9999"}, ["is", "9999"])
    assert score > 0.0
    assert "is_number" in matched


def test_bare_is_token_does_not_falsely_match_unrelated_standard(db_session):
    _seed(db_session)
    service = BISRetrievalService(db_session)

    # Genuinely out-of-scope question that happens to contain "is" (as
    # almost every English question does) must not match the seeded
    # IS 9999 standard purely because of that.
    results = service.search("What is the boiling point of water?")

    standard_hits = [r for r in results if r.standard_id is not None and r.clause_id is None]
    assert standard_hits == []


# ---------------------------------------------------------------------------
# Integration tests (TEST_DATABASE_URL / db_session)
# ---------------------------------------------------------------------------

def test_search_known_standard(db_session):
    _seed(db_session)
    service = BISRetrievalService(db_session)

    results = service.search("IS 9999")

    assert results
    standard_hits = [r for r in results if r.clause_id is None and r.document_id is None]
    assert standard_hits
    hit = standard_hits[0]
    assert hit.standard_number == "IS 9999"
    assert hit.standard_title == SAMPLE_STANDARD.title
    assert hit.standard_id is not None
    assert hit.relevance.method == "keyword"
    assert "is_number" in hit.relevance.matched_fields


def test_search_known_clause(db_session):
    _seed(db_session)
    service = BISRetrievalService(db_session)

    results = service.search("Earthing requirements")

    clause_hits = [r for r in results if r.clause_id is not None]
    assert clause_hits
    hit = clause_hits[0]
    assert hit.clause_number == "4.1"
    assert hit.clause_type == "CLAUSE"
    assert hit.clause_text is not None
    assert "Earthing" in hit.clause_text
    assert hit.page_number == 3


def test_filter_by_standard_number_and_year(db_session):
    _seed(db_session)
    service = BISRetrievalService(db_session)

    matching = service.search("household", is_number="IS 9999", year=2099)
    assert matching

    wrong_year = service.search("household", is_number="IS 9999", year=1999)
    assert wrong_year == []

    wrong_number = service.search("household", is_number="IS 0000")
    assert wrong_number == []


def test_filter_by_clause_type_and_language(db_session):
    _seed(db_session)
    service = BISRetrievalService(db_session)

    annex_hits = service.search("toasters", clause_type="ANNEX", language="en")
    assert annex_hits
    assert all(r.clause_type == "ANNEX" for r in annex_hits if r.clause_id is not None)

    hi_hits = service.search("toasters", language="hi")
    assert hi_hits == []


def test_retrieval_returns_correct_provenance(db_session):
    standard, document = _seed(db_session)
    service = BISRetrievalService(db_session)

    results = service.search("Earthing")
    clause_hits = [r for r in results if r.clause_id is not None]
    assert clause_hits
    hit = clause_hits[0]

    assert hit.source_url == SAMPLE_DOCUMENT.source_url
    assert hit.standard_id == standard.id
    assert hit.document_id == document.id
    assert hit.standard_number == "IS 9999"
    assert hit.document_title == SAMPLE_DOCUMENT.title


def test_irrelevant_query_returns_no_results(db_session):
    _seed(db_session)
    service = BISRetrievalService(db_session)

    assert service.search("unrelatedxyz123") == []
    assert service.search("") == []
    assert service.search("   ") == []


def test_clause_results_preserve_standard_document_clause_chain(db_session):
    standard, document = _seed(db_session)
    service = BISRetrievalService(db_session)

    results = service.search("household electric")
    clause_hits = [r for r in results if r.clause_id is not None]
    assert clause_hits

    for hit in clause_hits:
        assert hit.standard_id == standard.id
        assert hit.standard_number == standard.is_number
        assert hit.standard_title == standard.title
        assert hit.document_id == document.id
        assert hit.document_title == document.title
        assert hit.clause_id is not None
        assert hit.clause_number is not None
        assert hit.source_url == document.source_url


def test_retrieval_backend_is_swappable_for_future_vector_search(db_session):
    """
    Milestone 2 will inject a vector backend with this same search()
    contract. This test proves BISRetrievalService does not hard-code
    keyword matching.
    """
    canned = RetrievalResult(
        standard_id=42,
        standard_number="IS 0001",
        clause_id=7,
        clause_text="vector-placeholder",
        relevance=RelevanceInfo(score=0.99, method="vector", matched_fields=["embedding"]),
    )

    class StubVectorBackend:
        def search(self, query, filters, limit):
            assert isinstance(filters, SearchFilters)
            assert query == "semantic query"
            return [canned]

    service = BISRetrievalService(db_session, backend=StubVectorBackend())
    results = service.search("semantic query")

    assert len(results) == 1
    assert results[0] is canned
    assert results[0].relevance.method == "vector"
    assert not isinstance(service.backend, KeywordRetrievalBackend)
