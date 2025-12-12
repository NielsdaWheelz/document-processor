"""
Unit tests for pdf_text module.

Tests native PDF text extraction using pypdf.
"""

from pathlib import Path

import pytest

from app.pdf_text import extract_text_per_page, is_pdf


# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestExtractTextPerPage:
    """Tests for PDF text extraction."""

    def test_blank_pdf_has_no_text_layer(self) -> None:
        """Test that a blank PDF is detected as having no text layer."""
        blank_pdf = FIXTURES_DIR / "blank.pdf"
        assert blank_pdf.exists(), "blank.pdf fixture missing"

        result = extract_text_per_page(blank_pdf)

        assert result.has_text_layer is False
        assert result.page_count == 1
        # All pages should be empty/whitespace
        assert all(not page.strip() for page in result.pages)
        assert result.error is None

    def test_text_pdf_has_text_layer(self) -> None:
        """Test that a PDF with text is detected correctly."""
        text_pdf = FIXTURES_DIR / "sample_text.pdf"
        assert text_pdf.exists(), "sample_text.pdf fixture missing"

        result = extract_text_per_page(text_pdf)

        assert result.has_text_layer is True
        assert result.page_count == 1
        assert len(result.pages) == 1
        assert "Hello World" in result.pages[0]
        assert result.error is None

    def test_two_page_pdf_extracts_both_pages(self) -> None:
        """Test that multi-page PDFs have all pages extracted."""
        two_page_pdf = FIXTURES_DIR / "two_pages.pdf"
        assert two_page_pdf.exists(), "two_pages.pdf fixture missing"

        result = extract_text_per_page(two_page_pdf)

        assert result.has_text_layer is True
        assert result.page_count == 2
        assert len(result.pages) == 2
        # Internal list is 0-indexed
        assert "Page One" in result.pages[0]
        assert "Page Two" in result.pages[1]

    def test_invalid_pdf_returns_parse_error(self, tmp_path: Path) -> None:
        """Test that invalid PDF data returns error result."""
        invalid_pdf = tmp_path / "invalid.pdf"
        invalid_pdf.write_bytes(b"not a valid pdf")

        result = extract_text_per_page(invalid_pdf)

        assert result.has_text_layer is False
        assert result.page_count == 0
        assert len(result.pages) == 0
        assert result.error is not None
        assert "parse_error" in result.error

    def test_nonexistent_file_returns_error(self, tmp_path: Path) -> None:
        """Test that nonexistent file returns error."""
        nonexistent = tmp_path / "does_not_exist.pdf"

        result = extract_text_per_page(nonexistent)

        assert result.has_text_layer is False
        assert result.error is not None

    def test_extraction_is_deterministic(self) -> None:
        """Test that text extraction is deterministic."""
        text_pdf = FIXTURES_DIR / "sample_text.pdf"

        result1 = extract_text_per_page(text_pdf)
        result2 = extract_text_per_page(text_pdf)

        assert result1.pages == result2.pages
        assert result1.has_text_layer == result2.has_text_layer
        assert result1.page_count == result2.page_count


class TestIsPdf:
    """Tests for PDF detection."""

    def test_detects_valid_pdf(self) -> None:
        """Test that valid PDF is detected."""
        pdf_file = FIXTURES_DIR / "sample_text.pdf"
        assert is_pdf(pdf_file) is True

    def test_rejects_non_pdf(self, tmp_path: Path) -> None:
        """Test that non-PDF files are rejected."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_bytes(b"plain text content")

        assert is_pdf(txt_file) is False

    def test_rejects_nonexistent(self, tmp_path: Path) -> None:
        """Test that nonexistent files return False."""
        nonexistent = tmp_path / "nope.pdf"
        assert is_pdf(nonexistent) is False


class TestEmptyTextPdfSetsUnreadable:
    """Acceptance test: empty-text PDF sets has_text_layer=false."""

    def test_blank_pdf_extraction_result(self) -> None:
        """
        Given a PDF with at least one page but no extractable text,
        extract_text_per_page returns has_text_layer=False.
        """
        blank_pdf = FIXTURES_DIR / "blank.pdf"
        result = extract_text_per_page(blank_pdf)

        # Per spec: if all pages yield empty/whitespace only
        # => has_text_layer=false
        assert result.has_text_layer is False
        assert result.page_count >= 1  # Has at least one page

