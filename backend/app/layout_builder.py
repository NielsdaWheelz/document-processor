"""
Layout document builder.

Builds LayoutDoc objects from extracted PDF text.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from app.models import DocIndexItem, LayoutDoc, LayoutPageText, UnreadableReason
from app.pdf_text import PdfExtractionResult, extract_text_per_page, is_pdf
from app.runfs import RunPaths, write_json_atomic
from app.trace import TraceLogger, trace_step

if TYPE_CHECKING:
    pass


def build_layout_doc(doc_id: str, extraction: PdfExtractionResult) -> LayoutDoc:
    """
    Build a LayoutDoc from PDF extraction results.

    Args:
        doc_id: The document ID.
        extraction: The extraction result from pdf_text.

    Returns:
        LayoutDoc with 1-indexed pages and full_text per page.
        Spans are empty (not extracting bboxes in this implementation).
    """
    pages: list[LayoutPageText] = []

    for idx, page_text in enumerate(extraction.pages):
        # Convert 0-indexed to 1-indexed
        page_num = idx + 1

        # Strip trailing whitespace but preserve internal newlines
        full_text = page_text.rstrip()

        pages.append(
            LayoutPageText(
                page=page_num,
                full_text=full_text,
                spans=[],  # Not extracting spans/bboxes
            )
        )

    return LayoutDoc(doc_id=doc_id, pages=pages)


def update_doc_index_with_extraction(
    doc_item: DocIndexItem,
    extraction: PdfExtractionResult,
) -> DocIndexItem:
    """
    Update a DocIndexItem with extraction results.

    Sets has_text_layer, pages count, and unreadable_reason based on
    the extraction results.

    Args:
        doc_item: The original DocIndexItem (may have placeholder values).
        extraction: The extraction result from pdf_text.

    Returns:
        Updated DocIndexItem with extraction metadata.
    """
    unreadable_reason: UnreadableReason | None = None

    if extraction.error:
        has_text_layer = False
        unreadable_reason = "parse_error"
        pages = None
    elif not extraction.has_text_layer:
        has_text_layer = False
        unreadable_reason = "no_text_layer"
        pages = extraction.page_count
    else:
        has_text_layer = True
        unreadable_reason = None
        pages = extraction.page_count

    return DocIndexItem(
        doc_id=doc_item.doc_id,
        filename=doc_item.filename,
        mime_type=doc_item.mime_type,
        pages=pages,
        has_text_layer=has_text_layer,
        unreadable_reason=unreadable_reason,
        sha256=doc_item.sha256,
    )


def build_doc_index_and_layout(
    run_paths: RunPaths,
    doc_index_items: list[DocIndexItem],
    trace: TraceLogger,
) -> tuple[list[DocIndexItem], list[LayoutDoc]]:
    """
    Build doc_index and layout artifacts from ingested documents.

    For each document:
    - If PDF: extract text per page, update doc_index with text layer info
    - If not PDF: mark as non-PDF with no text extraction

    Args:
        run_paths: The RunPaths for this run.
        doc_index_items: Initial DocIndexItem list from ingest.
        trace: TraceLogger for recording step events.

    Returns:
        Tuple of (updated doc_index, layout_docs).
        Also writes doc_index.json and layout.json to artifacts directory.
    """
    input_docs_dir = run_paths.input_docs_dir()
    updated_items: list[DocIndexItem] = []
    layout_docs: list[LayoutDoc] = []

    # Step 1: Extract text
    with trace_step(
        trace,
        step="extract_text",
        inputs_ref=[str(input_docs_dir)],
        outputs_ref=[str(run_paths.artifact_path("layout.json"))],
    ):
        for doc_item in doc_index_items:
            filepath = input_docs_dir / doc_item.filename

            if doc_item.mime_type == "application/pdf" and is_pdf(filepath):
                extraction = extract_text_per_page(filepath)
                updated_item = update_doc_index_with_extraction(doc_item, extraction)
                layout_doc = build_layout_doc(doc_item.doc_id, extraction)
            else:
                # Non-PDF files: no text extraction
                updated_item = DocIndexItem(
                    doc_id=doc_item.doc_id,
                    filename=doc_item.filename,
                    mime_type=doc_item.mime_type,
                    pages=None,
                    has_text_layer=False,
                    unreadable_reason="no_text_layer",
                    sha256=doc_item.sha256,
                )
                layout_doc = LayoutDoc(doc_id=doc_item.doc_id, pages=[])

            updated_items.append(updated_item)
            layout_docs.append(layout_doc)

    # Step 2: Write artifacts
    with trace_step(
        trace,
        step="write_artifacts",
        inputs_ref=[],
        outputs_ref=[
            str(run_paths.artifact_path("doc_index.json")),
            str(run_paths.artifact_path("layout.json")),
        ],
    ):
        # Serialize to JSON-compatible dicts
        doc_index_data = [item.model_dump() for item in updated_items]
        layout_data = [doc.model_dump() for doc in layout_docs]

        write_json_atomic(run_paths.artifact_path("doc_index.json"), doc_index_data)
        write_json_atomic(run_paths.artifact_path("layout.json"), layout_data)

    return updated_items, layout_docs


def run_ingest_and_extract(
    run_paths: RunPaths,
    input_files: list[tuple[str, bytes]],
    trace: TraceLogger,
) -> tuple[list[DocIndexItem], list[LayoutDoc]]:
    """
    Run the full ingest + text extraction pipeline.

    This is the main entry point for pr-04 that pr-08's pipeline can call.
    Performs:
    1. Ingest: copy files, compute sha256, assign doc_ids
    2. Extract: native PDF text extraction
    3. Build: create doc_index and layout artifacts

    Args:
        run_paths: The RunPaths for this run.
        input_files: List of (filename, content) tuples in upload order.
        trace: TraceLogger for recording step events.

    Returns:
        Tuple of (doc_index, layout_docs).
    """
    from app.ingest import ingest_documents

    # Step 1: Ingest
    with trace_step(
        trace,
        step="ingest",
        inputs_ref=[f"{name}" for name, _ in input_files],
        outputs_ref=[str(run_paths.input_docs_dir())],
    ):
        doc_index_items = ingest_documents(run_paths, input_files)

    # Step 2-3: Extract + build + write artifacts
    return build_doc_index_and_layout(run_paths, doc_index_items, trace)

