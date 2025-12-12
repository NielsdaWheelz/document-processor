"""
Document ingestion for pipeline runs.

Handles:
- Deterministic doc_id assignment (doc_001, doc_002, ...)
- File copying to runs/<run_id>/input/input_docs/
- SHA256 computation
- MIME type detection (using mimetypes + PDF header sniff)
"""

from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING

from app.models import DocIndexItem

if TYPE_CHECKING:
    from app.runfs import RunPaths


# PDF magic bytes
_PDF_MAGIC = b"%PDF"


def _detect_mime_type(filepath: Path) -> str:
    """
    Detect MIME type of a file.

    Uses mimetypes module for extension-based detection,
    with fallback to PDF header sniff for .pdf files or unknowns.

    Args:
        filepath: Path to the file.

    Returns:
        MIME type string (e.g., "application/pdf", "text/plain").
    """
    # Try extension-based detection first
    mime_type, _ = mimetypes.guess_type(str(filepath))

    if mime_type:
        return mime_type

    # Fallback: check for PDF magic bytes
    try:
        with open(filepath, "rb") as f:
            header = f.read(8)
        if header.startswith(_PDF_MAGIC):
            return "application/pdf"
    except OSError:
        pass

    # Default fallback
    return "application/octet-stream"


def _compute_sha256(filepath: Path) -> str:
    """
    Compute SHA256 hash of a file.

    Args:
        filepath: Path to the file.

    Returns:
        Lowercase hex digest of SHA256 hash.
    """
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _format_doc_id(index: int) -> str:
    """
    Format a document ID from a 0-based index.

    Args:
        index: 0-based index of the document.

    Returns:
        Document ID string (e.g., "doc_001", "doc_002").
    """
    return f"doc_{index + 1:03d}"


def ingest_documents(
    run_paths: RunPaths,
    input_files: list[tuple[str, bytes]],
) -> list[DocIndexItem]:
    """
    Ingest input documents into a run directory.

    Copies files to runs/<run_id>/input/input_docs/, computes SHA256,
    and detects MIME types. Returns DocIndexItem list with placeholder
    values for has_text_layer and unreadable_reason (to be filled by
    pdf_text extraction).

    Args:
        run_paths: The RunPaths for this run.
        input_files: List of (filename, content) tuples in upload order.

    Returns:
        List of DocIndexItem with doc_id, filename, mime_type, sha256.
        has_text_layer defaults to True, pages to None, unreadable_reason to None.
        These placeholders are updated by the text extraction step.
    """
    from app.runfs import _sanitize_filename

    input_docs_dir = run_paths.input_docs_dir()
    items: list[DocIndexItem] = []

    for idx, (filename, content) in enumerate(input_files):
        doc_id = _format_doc_id(idx)
        safe_name = _sanitize_filename(filename)

        # Write file to input_docs directory
        dest_path = input_docs_dir / safe_name
        if not dest_path.exists():
            dest_path.write_bytes(content)

        # Compute sha256 and detect mime type from the written file
        sha256 = _compute_sha256(dest_path)
        mime_type = _detect_mime_type(dest_path)

        # Create DocIndexItem with placeholders for text layer info
        item = DocIndexItem(
            doc_id=doc_id,
            filename=safe_name,
            mime_type=mime_type,
            pages=None,
            has_text_layer=True,  # placeholder, updated by pdf_text
            unreadable_reason=None,
            sha256=sha256,
        )
        items.append(item)

    return items


def get_input_doc_paths(run_paths: RunPaths) -> list[Path]:
    """
    Get paths to all input documents in a run.

    Args:
        run_paths: The RunPaths for this run.

    Returns:
        List of Paths to input documents, sorted by filename.
    """
    input_docs_dir = run_paths.input_docs_dir()
    if not input_docs_dir.exists():
        return []
    return sorted(input_docs_dir.iterdir())

