"""
Unit tests for app.services.pdf_extractor.

Uses a minimal, hand-built PDF (constructed in-memory with correctly
computed xref offsets — see `_build_minimal_pdf` below) instead of a
committed binary fixture or an extra PDF-writing dependency. This keeps
the test suite fast, deterministic, and dependency-free.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.pdf_extractor import PdfExtractionError, extract_pages


def _build_minimal_pdf(page_texts: list[str]) -> bytes:
    """
    Build a minimal, valid, multi-page PDF containing the given text,
    one page per string. Uses the standard Helvetica font (no embedding
    needed) and computes exact byte offsets for the xref table.
    """

    def esc(s: str) -> str:
        return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")

    num_pages = len(page_texts)
    font_obj_num = 3 + 2 * num_pages
    page_obj_nums = [3 + i for i in range(num_pages)]
    content_obj_nums = [3 + num_pages + i for i in range(num_pages)]

    # objects[0] is unused (PDF object numbers are 1-indexed)
    objects: list[bytes] = [b""]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")  # 1: Catalog
    kids = " ".join(f"{n} 0 R" for n in page_obj_nums)
    objects.append(  # 2: Pages
        f"<< /Type /Pages /Kids [{kids}] /Count {num_pages} >>".encode()
    )
    for i in range(num_pages):  # 3..: Page objects
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R "
                f"/Resources << /Font << /F1 {font_obj_num} 0 R >> >> "
                f"/MediaBox [0 0 612 792] /Contents {content_obj_nums[i]} 0 R >>"
            ).encode()
        )
    for text in page_texts:  # content stream objects
        y = 700
        lines = []
        for line in text.split("\n"):
            lines.append(f"BT /F1 12 Tf 50 {y} Td ({esc(line)}) Tj ET")
            y -= 20
        stream_data = "\n".join(lines).encode()
        objects.append(
            f"<< /Length {len(stream_data)} >>\nstream\n".encode()
            + stream_data
            + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")  # font

    buf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i in range(1, len(objects)):
        offsets.append(len(buf))
        buf += f"{i} 0 obj\n".encode()
        buf += objects[i]
        buf += b"\nendobj\n"

    xref_offset = len(buf)
    total = len(objects)
    buf += f"xref\n0 {total}\n".encode()
    buf += b"0000000000 65535 f \n"
    for i in range(1, total):
        buf += f"{offsets[i]:010d} 00000 n \n".encode()
    buf += (
        f"trailer\n<< /Size {total} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF"
    ).encode()

    return bytes(buf)


def _write_pdf(tmp_path: Path, page_texts: list[str], name: str = "sample.pdf") -> Path:
    pdf_path = tmp_path / name
    pdf_path.write_bytes(_build_minimal_pdf(page_texts))
    return pdf_path


def test_extract_pages_returns_text_for_each_page(tmp_path):
    pdf_path = _write_pdf(tmp_path, ["Hello World", "Second page text"])

    pages = extract_pages(pdf_path)

    assert len(pages) == 2
    assert "Hello World" in pages[0][1]
    assert "Second page text" in pages[1][1]


def test_extract_pages_preserves_1_indexed_page_numbers(tmp_path):
    pdf_path = _write_pdf(tmp_path, ["Page A", "Page B", "Page C"])

    pages = extract_pages(pdf_path)

    assert [p[0] for p in pages] == [1, 2, 3]


def test_extract_pages_preserves_reading_order(tmp_path):
    texts = [f"Content of page {i}" for i in range(1, 5)]
    pdf_path = _write_pdf(tmp_path, texts)

    pages = extract_pages(pdf_path)

    for i, (page_number, text) in enumerate(pages, start=1):
        assert page_number == i
        assert f"Content of page {i}" in text


def test_extract_pages_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.pdf"

    with pytest.raises(PdfExtractionError):
        extract_pages(missing)


def test_extract_pages_never_drops_a_page(tmp_path):
    """Even if a page's text is empty, the page itself must still be returned."""
    pdf_path = _write_pdf(tmp_path, ["Some text", ""])

    pages = extract_pages(pdf_path)

    assert len(pages) == 2
    assert pages[1][0] == 2  # second page present, even though it has no text
