"""
Unit tests for ingest module.

Tests document ingestion, SHA256 computation, and MIME type detection.
"""

import hashlib
from pathlib import Path

import pytest

from app.ingest import (
    _compute_sha256,
    _detect_mime_type,
    _format_doc_id,
    ingest_documents,
)
from app.runfs import create_run


class TestComputeSha256:
    """Tests for SHA256 computation."""

    def test_sha256_matches_hashlib(self, tmp_path: Path) -> None:
        """Test that computed SHA256 matches hashlib.sha256 exactly."""
        test_bytes = b"Hello, World! This is test content for SHA256."
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(test_bytes)

        # Compute using our function
        computed = _compute_sha256(test_file)

        # Compute using hashlib directly
        expected = hashlib.sha256(test_bytes).hexdigest()

        assert computed == expected

    def test_sha256_is_lowercase_hex(self, tmp_path: Path) -> None:
        """Test that SHA256 is returned as lowercase hex."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test")

        result = _compute_sha256(test_file)

        # Should be 64 lowercase hex chars
        assert len(result) == 64
        assert result == result.lower()
        assert all(c in "0123456789abcdef" for c in result)

    def test_sha256_stable_across_calls(self, tmp_path: Path) -> None:
        """Test that SHA256 is deterministic."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"deterministic content")

        result1 = _compute_sha256(test_file)
        result2 = _compute_sha256(test_file)

        assert result1 == result2

    def test_sha256_different_for_different_content(self, tmp_path: Path) -> None:
        """Test that different content produces different hashes."""
        file1 = tmp_path / "file1.bin"
        file2 = tmp_path / "file2.bin"
        file1.write_bytes(b"content A")
        file2.write_bytes(b"content B")

        hash1 = _compute_sha256(file1)
        hash2 = _compute_sha256(file2)

        assert hash1 != hash2


class TestDetectMimeType:
    """Tests for MIME type detection."""

    def test_detects_pdf_by_extension(self, tmp_path: Path) -> None:
        """Test that .pdf extension is detected."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 content")

        mime = _detect_mime_type(pdf_file)
        assert mime == "application/pdf"

    def test_detects_pdf_by_magic_bytes(self, tmp_path: Path) -> None:
        """Test that PDF is detected by magic bytes even without extension."""
        pdf_file = tmp_path / "unknown_file"
        pdf_file.write_bytes(b"%PDF-1.4 some content here")

        mime = _detect_mime_type(pdf_file)
        assert mime == "application/pdf"

    def test_detects_text_file(self, tmp_path: Path) -> None:
        """Test that .txt extension is detected."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_bytes(b"plain text content")

        mime = _detect_mime_type(txt_file)
        assert mime == "text/plain"

    def test_fallback_for_unknown(self, tmp_path: Path) -> None:
        """Test that unknown files get octet-stream mime type."""
        unknown_file = tmp_path / "unknown"
        unknown_file.write_bytes(b"\x00\x01\x02\x03")

        mime = _detect_mime_type(unknown_file)
        assert mime == "application/octet-stream"


class TestFormatDocId:
    """Tests for doc_id formatting."""

    def test_format_first_doc(self) -> None:
        """Test that index 0 produces doc_001."""
        assert _format_doc_id(0) == "doc_001"

    def test_format_tenth_doc(self) -> None:
        """Test that index 9 produces doc_010."""
        assert _format_doc_id(9) == "doc_010"

    def test_format_hundredth_doc(self) -> None:
        """Test that index 99 produces doc_100."""
        assert _format_doc_id(99) == "doc_100"

    def test_format_preserves_order(self) -> None:
        """Test that doc_ids are lexicographically ordered."""
        ids = [_format_doc_id(i) for i in range(100)]
        assert ids == sorted(ids)


class TestIngestDocuments:
    """Tests for document ingestion."""

    def test_ingest_single_file(self, tmp_path: Path) -> None:
        """Test ingesting a single file."""
        run = create_run(run_id="test-ingest", base_dir=tmp_path)
        content = b"test pdf content"
        input_files = [("test.pdf", content)]

        items = ingest_documents(run, input_files)

        assert len(items) == 1
        item = items[0]
        assert item.doc_id == "doc_001"
        assert item.filename == "test.pdf"
        assert item.sha256 == hashlib.sha256(content).hexdigest()

    def test_ingest_multiple_files_ordered(self, tmp_path: Path) -> None:
        """Test that multiple files get sequential doc_ids in upload order."""
        run = create_run(run_id="test-multi", base_dir=tmp_path)
        input_files = [
            ("first.pdf", b"first"),
            ("second.pdf", b"second"),
            ("third.pdf", b"third"),
        ]

        items = ingest_documents(run, input_files)

        assert len(items) == 3
        assert items[0].doc_id == "doc_001"
        assert items[0].filename == "first.pdf"
        assert items[1].doc_id == "doc_002"
        assert items[1].filename == "second.pdf"
        assert items[2].doc_id == "doc_003"
        assert items[2].filename == "third.pdf"

    def test_ingest_copies_files_to_input_docs(self, tmp_path: Path) -> None:
        """Test that files are copied to input_docs directory."""
        run = create_run(run_id="test-copy", base_dir=tmp_path)
        content = b"%PDF-1.4 test content"
        input_files = [("doc.pdf", content)]

        ingest_documents(run, input_files)

        # File should exist in input_docs
        copied_file = run.input_docs_dir() / "doc.pdf"
        assert copied_file.exists()
        assert copied_file.read_bytes() == content

    def test_ingest_sha256_stable(self, tmp_path: Path) -> None:
        """Test that SHA256 matches hashlib.sha256(file_bytes).hexdigest() exactly."""
        run = create_run(run_id="test-sha", base_dir=tmp_path)
        content = b"specific test content for sha256 verification"
        input_files = [("test.pdf", content)]

        items = ingest_documents(run, input_files)

        expected_sha = hashlib.sha256(content).hexdigest()
        assert items[0].sha256 == expected_sha

    def test_ingest_detects_mime_type(self, tmp_path: Path) -> None:
        """Test that MIME type is detected correctly."""
        run = create_run(run_id="test-mime", base_dir=tmp_path)
        input_files = [
            ("doc.pdf", b"%PDF-1.4 content"),
            ("notes.txt", b"plain text"),
        ]

        items = ingest_documents(run, input_files)

        assert items[0].mime_type == "application/pdf"
        assert items[1].mime_type == "text/plain"

    def test_ingest_placeholder_values(self, tmp_path: Path) -> None:
        """Test that has_text_layer defaults to True as placeholder."""
        run = create_run(run_id="test-placeholder", base_dir=tmp_path)
        input_files = [("doc.pdf", b"%PDF-1.4")]

        items = ingest_documents(run, input_files)

        # Placeholders before text extraction
        assert items[0].has_text_layer is True
        assert items[0].unreadable_reason is None
        assert items[0].pages is None

