"""
Unit tests for layout_builder module.

Tests LayoutDoc building and doc_index/layout artifact emission.
"""

from pathlib import Path

import pytest

from app.ingest import ingest_documents
from app.layout_builder import (
    build_doc_index_and_layout,
    build_layout_doc,
    run_ingest_and_extract,
    update_doc_index_with_extraction,
)
from app.models import DocIndexItem, LayoutDoc, LayoutPageText
from app.pdf_text import PdfExtractionResult, extract_text_per_page
from app.runfs import create_run, read_json
from app.trace import TraceLogger


# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestBuildLayoutDoc:
    """Tests for building LayoutDoc from extraction results."""

    def test_builds_layout_with_1_indexed_pages(self) -> None:
        """Test that LayoutDoc pages are 1-indexed."""
        extraction = PdfExtractionResult(
            pages=["Page one text", "Page two text"],
            page_count=2,
            has_text_layer=True,
        )

        layout = build_layout_doc("doc_001", extraction)

        assert layout.doc_id == "doc_001"
        assert len(layout.pages) == 2
        # Pages should be 1-indexed
        assert layout.pages[0].page == 1
        assert layout.pages[1].page == 2

    def test_preserves_full_text(self) -> None:
        """Test that full_text is preserved (no normalization of internal content)."""
        extraction = PdfExtractionResult(
            pages=["Text with\nmultiple\nlines"],
            page_count=1,
            has_text_layer=True,
        )

        layout = build_layout_doc("doc_001", extraction)

        # Internal newlines should be preserved
        assert "\n" in layout.pages[0].full_text
        assert "Text with\nmultiple\nlines" == layout.pages[0].full_text

    def test_strips_trailing_whitespace(self) -> None:
        """Test that trailing whitespace is stripped."""
        extraction = PdfExtractionResult(
            pages=["Text with trailing space   \n\n"],
            page_count=1,
            has_text_layer=True,
        )

        layout = build_layout_doc("doc_001", extraction)

        # Trailing whitespace stripped
        assert not layout.pages[0].full_text.endswith(" ")
        assert not layout.pages[0].full_text.endswith("\n")

    def test_empty_spans(self) -> None:
        """Test that spans are empty by default."""
        extraction = PdfExtractionResult(
            pages=["Some text"],
            page_count=1,
            has_text_layer=True,
        )

        layout = build_layout_doc("doc_001", extraction)

        assert layout.pages[0].spans == []

    def test_empty_extraction_produces_empty_layout(self) -> None:
        """Test that empty extraction produces layout with no pages."""
        extraction = PdfExtractionResult(
            pages=[],
            page_count=0,
            has_text_layer=False,
            error="parse_error: something went wrong",
        )

        layout = build_layout_doc("doc_001", extraction)

        assert layout.doc_id == "doc_001"
        assert len(layout.pages) == 0


class TestUpdateDocIndexWithExtraction:
    """Tests for updating DocIndexItem with extraction results."""

    def test_update_with_text_layer(self) -> None:
        """Test update when PDF has text layer."""
        item = DocIndexItem(
            doc_id="doc_001",
            filename="test.pdf",
            mime_type="application/pdf",
            pages=None,
            has_text_layer=True,  # placeholder
            unreadable_reason=None,
            sha256="abc123",
        )
        extraction = PdfExtractionResult(
            pages=["Text"],
            page_count=1,
            has_text_layer=True,
        )

        updated = update_doc_index_with_extraction(item, extraction)

        assert updated.has_text_layer is True
        assert updated.unreadable_reason is None
        assert updated.pages == 1

    def test_update_with_no_text_layer(self) -> None:
        """Test update when PDF has no text layer."""
        item = DocIndexItem(
            doc_id="doc_001",
            filename="blank.pdf",
            mime_type="application/pdf",
            pages=None,
            has_text_layer=True,  # placeholder
            unreadable_reason=None,
            sha256="abc123",
        )
        extraction = PdfExtractionResult(
            pages=[""],
            page_count=1,
            has_text_layer=False,
        )

        updated = update_doc_index_with_extraction(item, extraction)

        assert updated.has_text_layer is False
        assert updated.unreadable_reason == "no_text_layer"
        assert updated.pages == 1

    def test_update_with_parse_error(self) -> None:
        """Test update when PDF parsing fails."""
        item = DocIndexItem(
            doc_id="doc_001",
            filename="corrupted.pdf",
            mime_type="application/pdf",
            pages=None,
            has_text_layer=True,
            unreadable_reason=None,
            sha256="abc123",
        )
        extraction = PdfExtractionResult(
            pages=[],
            page_count=0,
            has_text_layer=False,
            error="parse_error: corrupted file",
        )

        updated = update_doc_index_with_extraction(item, extraction)

        assert updated.has_text_layer is False
        assert updated.unreadable_reason == "parse_error"
        assert updated.pages is None


class TestLayoutPagesAre1Indexed:
    """Acceptance test: layout pages are 1-indexed."""

    def test_layout_from_fixture(self) -> None:
        """
        Given a PDF with known text on page 1,
        LayoutDoc.pages[0].page == 1 and contains expected text.
        """
        two_page_pdf = FIXTURES_DIR / "two_pages.pdf"
        extraction = extract_text_per_page(two_page_pdf)
        layout = build_layout_doc("doc_001", extraction)

        # First element should be page 1
        assert layout.pages[0].page == 1
        assert "Page One" in layout.pages[0].full_text

        # Second element should be page 2
        assert layout.pages[1].page == 2
        assert "Page Two" in layout.pages[1].full_text


class TestRunIngestAndExtract:
    """Integration tests for the full ingest + extract pipeline."""

    def test_full_pipeline_with_text_pdf(self, tmp_path: Path) -> None:
        """Test full pipeline produces correct doc_index and layout."""
        # Read fixture
        text_pdf = FIXTURES_DIR / "sample_text.pdf"
        content = text_pdf.read_bytes()

        # Create run
        run = create_run(run_id="test-pipeline", base_dir=tmp_path)
        trace = TraceLogger(run)

        # Run pipeline
        doc_index, layouts = run_ingest_and_extract(
            run,
            [("sample.pdf", content)],
            trace,
        )

        # Check doc_index
        assert len(doc_index) == 1
        assert doc_index[0].doc_id == "doc_001"
        assert doc_index[0].has_text_layer is True
        assert doc_index[0].unreadable_reason is None
        assert doc_index[0].pages == 1

        # Check layouts
        assert len(layouts) == 1
        assert layouts[0].doc_id == "doc_001"
        assert len(layouts[0].pages) == 1
        assert layouts[0].pages[0].page == 1
        assert "Hello World" in layouts[0].pages[0].full_text

    def test_full_pipeline_with_blank_pdf(self, tmp_path: Path) -> None:
        """Test that blank PDF sets has_text_layer=false and unreadable_reason."""
        blank_pdf = FIXTURES_DIR / "blank.pdf"
        content = blank_pdf.read_bytes()

        run = create_run(run_id="test-blank", base_dir=tmp_path)
        trace = TraceLogger(run)

        doc_index, layouts = run_ingest_and_extract(
            run,
            [("blank.pdf", content)],
            trace,
        )

        # doc_index should indicate no text layer
        assert len(doc_index) == 1
        assert doc_index[0].has_text_layer is False
        assert doc_index[0].unreadable_reason == "no_text_layer"

        # layout should have empty/whitespace pages
        assert len(layouts) == 1
        assert len(layouts[0].pages) == 1

    def test_artifacts_written(self, tmp_path: Path) -> None:
        """Test that doc_index.json and layout.json are written atomically."""
        text_pdf = FIXTURES_DIR / "sample_text.pdf"
        content = text_pdf.read_bytes()

        run = create_run(run_id="test-artifacts", base_dir=tmp_path)
        trace = TraceLogger(run)

        run_ingest_and_extract(run, [("doc.pdf", content)], trace)

        # Check artifacts exist
        doc_index_path = run.artifact_path("doc_index.json")
        layout_path = run.artifact_path("layout.json")
        assert doc_index_path.exists()
        assert layout_path.exists()

        # Verify content
        doc_index_data = read_json(doc_index_path)
        layout_data = read_json(layout_path)

        assert len(doc_index_data) == 1
        assert doc_index_data[0]["doc_id"] == "doc_001"

        assert len(layout_data) == 1
        assert layout_data[0]["doc_id"] == "doc_001"

    def test_trace_steps_recorded(self, tmp_path: Path) -> None:
        """Test that trace steps are recorded."""
        text_pdf = FIXTURES_DIR / "sample_text.pdf"
        content = text_pdf.read_bytes()

        run = create_run(run_id="test-trace", base_dir=tmp_path)
        trace = TraceLogger(run)

        run_ingest_and_extract(run, [("doc.pdf", content)], trace)

        # Read trace file
        trace_path = run.trace_jsonl_path()
        assert trace_path.exists()

        import json
        events = [json.loads(line) for line in trace_path.read_text().strip().split("\n")]

        # Should have ingest, extract_text, write_artifacts steps
        steps = [e["step"] for e in events]
        assert "ingest" in steps
        assert "extract_text" in steps
        assert "write_artifacts" in steps

    def test_multiple_docs_in_order(self, tmp_path: Path) -> None:
        """Test that multiple docs get sequential doc_ids."""
        text_pdf = FIXTURES_DIR / "sample_text.pdf"
        blank_pdf = FIXTURES_DIR / "blank.pdf"
        text_content = text_pdf.read_bytes()
        blank_content = blank_pdf.read_bytes()

        run = create_run(run_id="test-multi", base_dir=tmp_path)
        trace = TraceLogger(run)

        doc_index, layouts = run_ingest_and_extract(
            run,
            [
                ("first.pdf", text_content),
                ("second.pdf", blank_content),
            ],
            trace,
        )

        # Check ordering
        assert doc_index[0].doc_id == "doc_001"
        assert doc_index[0].filename == "first.pdf"
        assert doc_index[0].has_text_layer is True

        assert doc_index[1].doc_id == "doc_002"
        assert doc_index[1].filename == "second.pdf"
        assert doc_index[1].has_text_layer is False

