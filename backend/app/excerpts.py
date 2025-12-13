"""
Excerpt building for LLM candidate extraction.

Builds capped, deterministic document excerpts for LLM context windows.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.models import FieldSpec, LayoutDoc
from app.routing import FIELD_ALIASES


class DocExcerpt(BaseModel):
    """A capped excerpt from a document for LLM context."""

    doc_id: str
    page: int
    text: str


def _get_field_keywords(field: FieldSpec) -> set[str]:
    """
    Get keywords to search for in document text for a field.

    Includes: field.key, field.label (if present), and aliases.
    All lowercased for matching.
    """
    keywords: set[str] = {field.key.lower()}

    if field.label:
        keywords.add(field.label.lower())

    aliases = FIELD_ALIASES.get(field.key, [])
    for alias in aliases:
        keywords.add(alias.lower())

    return keywords


def _page_contains_keyword(page_text: str, keywords: set[str]) -> bool:
    """Check if page text contains any of the keywords."""
    text_lower = page_text.lower()
    return any(kw in text_lower for kw in keywords)


def build_excerpts_for_field(
    field: FieldSpec,
    routed_docs: list[LayoutDoc],
    *,
    max_total_chars: int,
    max_chars_per_doc: int,
    max_pages_per_doc: int,
) -> list[DocExcerpt]:
    """
    Build deterministic excerpts for a field from routed documents.

    Capping rules (locked):
    - Consider docs in routing order (routed_docs order)
    - For each doc, consider pages in ascending page order
    - Pick up to max_pages_per_doc pages whose text contains any keyword
      from field query (key + label + aliases); if none match, take first page
    - Truncate text to max_chars_per_doc per doc
    - Enforce max_total_chars overall by truncating the last excerpt
    - Output order is stable across runs

    Args:
        field: The field spec to build excerpts for.
        routed_docs: Documents routed to this field, in routing order.
        max_total_chars: Maximum total characters across all excerpts.
        max_chars_per_doc: Maximum characters per document.
        max_pages_per_doc: Maximum pages to include per document.

    Returns:
        List of DocExcerpt in deterministic order.
    """
    keywords = _get_field_keywords(field)
    excerpts: list[DocExcerpt] = []
    total_chars = 0

    for doc in routed_docs:
        if total_chars >= max_total_chars:
            break

        # Sort pages by page number ascending
        sorted_pages = sorted(doc.pages, key=lambda p: p.page)

        # Find pages containing keywords
        matching_pages = [
            p for p in sorted_pages
            if _page_contains_keyword(p.full_text, keywords)
        ]

        # If no matches, take first page
        if not matching_pages and sorted_pages:
            matching_pages = [sorted_pages[0]]

        # Limit to max_pages_per_doc
        selected_pages = matching_pages[:max_pages_per_doc]

        # Build excerpts for this doc, respecting per-doc char limit
        doc_chars_used = 0

        for page in selected_pages:
            if total_chars >= max_total_chars:
                break

            if doc_chars_used >= max_chars_per_doc:
                break

            # Calculate how much text we can take
            remaining_doc_chars = max_chars_per_doc - doc_chars_used
            remaining_total_chars = max_total_chars - total_chars
            max_for_this_page = min(remaining_doc_chars, remaining_total_chars)

            text = page.full_text[:max_for_this_page]

            if text:  # Only add non-empty excerpts
                excerpts.append(DocExcerpt(
                    doc_id=doc.doc_id,
                    page=page.page,
                    text=text,
                ))
                doc_chars_used += len(text)
                total_chars += len(text)

    return excerpts

