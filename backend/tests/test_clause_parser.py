"""
Unit tests for app.services.clause_parser.

All inputs here are synthetic sample text (never real BIS content) —
chosen only to exercise the parser's structural rules.
"""

from app.services.clause_parser import (
    CLAUSE_TYPE_ANNEX,
    CLAUSE_TYPE_CLAUSE,
    CLAUSE_TYPE_PREAMBLE,
    PREAMBLE_CLAUSE_NUMBER,
    parse_pages_to_clauses,
)


def test_parses_top_level_numbered_clauses():
    pages = [(1, "1. This is the first paragraph.\n2. This is the second paragraph.")]

    clauses = parse_pages_to_clauses(pages)

    numbers = [c.clause_number for c in clauses]
    assert numbers == ["1", "2"]
    assert clauses[0].clause_type == CLAUSE_TYPE_CLAUSE
    assert "first paragraph" in clauses[0].content


def test_parses_nested_numbered_clauses():
    pages = [(1, "4.1 Sub-clause text.\n4.1.2 Deeper sub-clause text.")]

    clauses = parse_pages_to_clauses(pages)

    assert clauses[0].clause_number == "4.1"
    assert clauses[1].clause_number == "4.1.2"


def test_multiline_content_is_accumulated_under_one_clause():
    pages = [
        (
            1,
            "3. Heading line\n"
            "This is a continuation line.\n"
            "This is another continuation line.",
        )
    ]

    clauses = parse_pages_to_clauses(pages)

    assert len(clauses) == 1
    assert "Heading line" in clauses[0].content
    assert "continuation line" in clauses[0].content


def test_parses_annexure_sections_with_title_on_heading_line():
    # Title text on the SAME line as the heading is captured as `title`.
    pages = [(5, "Annexure \u2013 I List of items covered\nItem A\nItem B")]

    clauses = parse_pages_to_clauses(pages)

    assert len(clauses) == 1
    assert clauses[0].clause_number == "Annexure I"
    assert clauses[0].clause_type == CLAUSE_TYPE_ANNEX
    assert clauses[0].title == "List of items covered"
    assert "Item A" in clauses[0].content


def test_parses_annexure_heading_alone_then_body_as_content():
    # Real BIS circulars often put the heading alone on its own line,
    # with the description starting on the next line — that description
    # belongs in `content`, not `title`.
    pages = [(5, "Annexure \u2013 I\nList of items covered\nItem A\nItem B")]

    clauses = parse_pages_to_clauses(pages)

    assert len(clauses) == 1
    assert clauses[0].clause_number == "Annexure I"
    assert clauses[0].clause_type == CLAUSE_TYPE_ANNEX
    assert clauses[0].title is None
    assert "List of items covered" in clauses[0].content
    assert "Item A" in clauses[0].content


def test_parses_annex_without_ure_suffix():
    pages = [(2, "Annex A\nSome annex content here.")]

    clauses = parse_pages_to_clauses(pages)

    assert clauses[0].clause_number == "Annex A" or clauses[0].clause_number == "Annexure A"
    assert clauses[0].clause_type == CLAUSE_TYPE_ANNEX


def test_page_numbers_are_preserved_across_pages():
    pages = [(1, "1. First clause"), (7, "2. Second clause on a later page")]

    clauses = parse_pages_to_clauses(pages)

    assert clauses[0].page_number == 1
    assert clauses[1].page_number == 7


def test_unstructured_text_does_not_fabricate_clause_numbers():
    """
    Per the ingestion rules: if no numbered/annex heading is found, the
    parser must NOT invent clause boundaries — everything falls under the
    PREAMBLE pseudo-clause instead.
    """
    pages = [(1, "Just some free-flowing text.\nNo numbering here at all.")]

    clauses = parse_pages_to_clauses(pages)

    assert len(clauses) == 1
    assert clauses[0].clause_number == PREAMBLE_CLAUSE_NUMBER
    assert clauses[0].clause_type == CLAUSE_TYPE_PREAMBLE


def test_language_is_stamped_on_every_clause_when_provided():
    pages = [(1, "1. First\n2. Second")]

    clauses = parse_pages_to_clauses(pages, language="en")

    assert all(c.language == "en" for c in clauses)


def test_empty_pages_produce_no_clauses():
    pages = [(1, ""), (2, "   ")]

    clauses = parse_pages_to_clauses(pages)

    assert clauses == []
