"""Unit tests for app.services.normalizer."""

from app.services.normalizer import normalize_text


def test_empty_string_returns_empty():
    assert normalize_text("") == ""


def test_collapses_repeated_spaces():
    assert normalize_text("hello    world") == "hello world"


def test_collapses_repeated_blank_lines():
    text = "line one\n\n\n\n\nline two"
    result = normalize_text(text)
    assert "\n\n\n" not in result
    assert "line one" in result
    assert "line two" in result


def test_strips_trailing_whitespace_on_lines():
    text = "first line   \nsecond line\t\nthird"
    result = normalize_text(text)
    assert "   \n" not in result
    assert "\t\n" not in result


def test_normalizes_windows_line_endings():
    text = "line one\r\nline two\r\n"
    result = normalize_text(text)
    assert "\r" not in result


def test_preserves_meaningful_words_and_order():
    text = "  The   quick  brown  fox  \n\n\n\n  jumps over the lazy dog  "
    result = normalize_text(text)
    assert "The quick brown fox" in result
    assert "jumps over the lazy dog" in result
    # word order preserved
    assert result.index("quick") < result.index("jumps")


def test_is_idempotent():
    text = "some   text\n\n\n\nwith   artifacts   "
    once = normalize_text(text)
    twice = normalize_text(once)
    assert once == twice
