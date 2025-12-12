"""
Native PDF text extraction.

Uses pypdf for read-only PDF text extraction.
No OCR, no image extraction - native text layer only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


@dataclass
class PdfExtractionResult:
    """Result of PDF text extraction."""

    pages: list[str]  # per-page text, 0-indexed internally
    page_count: int
    has_text_layer: bool
    error: str | None = None


def extract_text_per_page(filepath: Path) -> PdfExtractionResult:
    """
    Extract text from each page of a PDF.

    Uses pypdf for native text extraction. Does not perform OCR.
    If all pages yield empty/whitespace-only text, has_text_layer is False.

    Args:
        filepath: Path to the PDF file.

    Returns:
        PdfExtractionResult with per-page text (0-indexed), page count,
        and has_text_layer flag.

    Note:
        - Internal list is 0-indexed; callers should convert to 1-indexed
          for LayoutPageText.page field.
        - Text is returned as-is from pypdf; no normalization applied.
    """
    try:
        reader = PdfReader(filepath)
    except PdfReadError as e:
        return PdfExtractionResult(
            pages=[],
            page_count=0,
            has_text_layer=False,
            error=f"parse_error: {e}",
        )
    except Exception as e:
        # Catch any other parsing errors
        return PdfExtractionResult(
            pages=[],
            page_count=0,
            has_text_layer=False,
            error=f"parse_error: {e}",
        )

    page_count = len(reader.pages)
    pages: list[str] = []

    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            # If a single page fails, treat it as empty
            text = ""
        pages.append(text)

    # Determine if there's a usable text layer
    # A text layer exists if any page has non-whitespace text
    has_text = any(page_text.strip() for page_text in pages)

    return PdfExtractionResult(
        pages=pages,
        page_count=page_count,
        has_text_layer=has_text,
        error=None,
    )


def is_pdf(filepath: Path) -> bool:
    """
    Check if a file is a PDF by reading its header.

    Args:
        filepath: Path to the file.

    Returns:
        True if the file starts with PDF magic bytes, False otherwise.
    """
    try:
        with open(filepath, "rb") as f:
            header = f.read(8)
        return header.startswith(b"%PDF")
    except OSError:
        return False

