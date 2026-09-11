"""Unit tests for app.services.normalizer."""

from app.services.normalizer import (
    normalize_text,
    strip_devanagari_lines,
    strip_gazette_boilerplate_lines,
)


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


# ---------------------------------------------------------------------------
# Milestone 5: strip_devanagari_lines / strip_gazette_boilerplate_lines.
#
# Excerpts below are taken verbatim from two real, freely-published BIS
# Gazette QCO notifications (Resin Treated Compressed Wood Laminates QCO
# 2024, https://www.bis.gov.in/wp-content/uploads/2024/03/Resin-Treated-Compressed-Wood-Laminates-QCO-2024.pdf
# and Self-Contained Drinking Water Cooler QCO 2024,
# https://bis.gov.in/wp-content/uploads/2024/05/Self-Contained-Drinking-Water-Cooler-QCO-2024.pdf)
# — real cases observed while implementing Milestone 5 corpus expansion,
# not synthetic/invented examples.
# ---------------------------------------------------------------------------


def test_strip_devanagari_lines_drops_hindi_lines_keeps_english():
    text = (
        "बिते दक लघु उद्यमों के जलए\n"
        "MINISTRY OF COMMERCE AND INDUSTRY\n"
        "(Department For Promotion of Industry and Internal Trade)\n"
    )
    result = strip_devanagari_lines(text)
    assert "बिते" not in result
    assert "MINISTRY OF COMMERCE AND INDUSTRY" in result
    assert "(Department For Promotion of Industry and Internal Trade)" in result


def test_strip_devanagari_lines_on_pure_english_text_is_a_no_op():
    text = "1. Short title and commencement.\nThis Order may be called the Order, 2024."
    assert strip_devanagari_lines(text) == text


def test_strip_devanagari_lines_empty_string():
    assert strip_devanagari_lines("") == ""


def test_strip_gazette_boilerplate_removes_known_running_head_and_footer_lines():
    text = (
        "CG-DxLx-xEG-I0D5H0x3x2x0 24-252600\n"
        "xxxGIDExxx\n"
        "EXTRAORDINARY\n"
        "PART II—Section 3—Sub-section (ii)\n"
        "PUBLISHED BY AUTHORITY\n"
        "No. 973] NEW DELHI, TUESDAY, MARCH 5, 2024/PHALGUNA 15, 1945\n"
        "1490 GI/2024 (1)\n"
        "2 THE GAZETTE OF INDIA : EXTRAORDINARY [PART II—SEC. 3(ii)]\n"
        "1. Short title and commencement. - (1) This Order may be called the Order.\n"
    )
    result = strip_gazette_boilerplate_lines(text)
    assert "xxxGIDExxx" not in result
    assert "PUBLISHED BY AUTHORITY" not in result
    assert "1490 GI/2024 (1)" not in result
    assert "THE GAZETTE OF INDIA" not in result
    # Real clause content must survive untouched.
    assert "1. Short title and commencement. - (1) This Order may be called the Order." in result


def test_strip_gazette_boilerplate_does_not_touch_ordinary_content_lines():
    text = (
        "3. Certification and enforcing authority.- The Bureau shall be the "
        "certifying and enforcing authority for the goods or articles specified "
        "in column (1) of the Table."
    )
    assert strip_gazette_boilerplate_lines(text) == text


def test_strip_gazette_boilerplate_empty_string():
    assert strip_gazette_boilerplate_lines("") == ""
