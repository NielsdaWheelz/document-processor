"""
Routing module for the document-processor pipeline.

Routes each resolved field to the top-k most relevant readable documents
using a deterministic token-overlap similarity function.

Produces artifacts/routing.json containing list[RoutingEntry].
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from app.models import (
    DocIndexItem,
    FieldSpec,
    LayoutDoc,
    ResolvedSchema,
    RoutingEntry,
)
from app.runfs import RunPaths, write_json_atomic
from app.trace import TraceLogger, trace_step

if TYPE_CHECKING:
    pass

# Maximum characters to consider from each doc's text
N_CHARS: int = 20000

# Field alias map for query construction
# Duplicated from schema_resolver to avoid import cycles
FIELD_ALIASES: dict[str, list[str]] = {
    "full_name": ["full_name", "name", "patient_name"],
    "dob": ["dob", "date_of_birth", "birthdate"],
    "phone": ["phone", "mobile", "telephone"],
    "address": ["address", "street"],
    "insurance_member_id": ["insurance_member_id", "member_id", "policy", "insurance_id"],
    "allergies": ["allergies", "allergy"],
    "medications": ["medications", "meds"],
}


def tokenize(text: str) -> set[str]:
    """
    Tokenize text deterministically.

    Algorithm:
    - Lowercase the text
    - Replace any non-alphanumeric character with space
    - Split on whitespace
    - Drop empty tokens
    - Keep only tokens with length >= 2

    Args:
        text: The input text to tokenize.

    Returns:
        A set of unique tokens.
    """
    # Lowercase
    text = text.lower()

    # Replace non-alphanumeric with space
    text = re.sub(r"[^a-z0-9]", " ", text)

    # Split on whitespace and filter
    tokens = text.split()

    # Keep tokens with length >= 2
    return {t for t in tokens if len(t) >= 2}


def score_query_doc(query: str, doc_text: str) -> float:
    """
    Compute similarity score between a field query and document text.

    Formula:
        score = |tokens(query) ∩ tokens(doc)| / max(1, |tokens(query)|)

    Args:
        query: The field query string.
        doc_text: The document text to compare against.

    Returns:
        A float score in [0, 1].
    """
    query_tokens = tokenize(query)
    doc_tokens = tokenize(doc_text)

    if not query_tokens:
        return 0.0

    intersection_size = len(query_tokens & doc_tokens)
    return intersection_size / len(query_tokens)


def _build_field_query(field: FieldSpec) -> str:
    """
    Build a query string for a field.

    Includes:
    - field.key
    - field.label (if present)
    - alias terms for that key

    Args:
        field: The FieldSpec to build a query for.

    Returns:
        A space-joined query string.
    """
    parts: list[str] = [field.key]

    if field.label:
        parts.append(field.label)

    # Add aliases for this key
    aliases = FIELD_ALIASES.get(field.key, [])
    parts.extend(aliases)

    return " ".join(parts)


def _build_doc_text(doc_id: str, layout_docs: list[LayoutDoc]) -> str:
    """
    Build the text representation for a document.

    Concatenates full_text of all pages in order (page 1..n),
    capped at N_CHARS characters.

    Args:
        doc_id: The document ID to find.
        layout_docs: The list of LayoutDoc objects.

    Returns:
        The concatenated text, capped at N_CHARS.
    """
    # Find the layout doc for this doc_id
    layout_doc = None
    for ld in layout_docs:
        if ld.doc_id == doc_id:
            layout_doc = ld
            break

    if layout_doc is None:
        return ""

    # Sort pages by page number and concatenate
    sorted_pages = sorted(layout_doc.pages, key=lambda p: p.page)
    full_text = "".join(p.full_text for p in sorted_pages)

    # Cap at N_CHARS
    return full_text[:N_CHARS]


def _is_readable(doc: DocIndexItem) -> bool:
    """
    Check if a document is readable.

    A doc is readable if:
    - has_text_layer == True
    - unreadable_reason is None

    Args:
        doc: The DocIndexItem to check.

    Returns:
        True if readable, False otherwise.
    """
    return doc.has_text_layer and doc.unreadable_reason is None


def route_docs(
    schema: ResolvedSchema,
    doc_index: list[DocIndexItem],
    layout: list[LayoutDoc],
    *,
    top_k: int,
) -> list[RoutingEntry]:
    """
    Route each resolved field to the top-k most relevant readable documents.

    Args:
        schema: The resolved schema containing fields to route.
        doc_index: The document index with metadata.
        layout: The layout documents with text content.
        top_k: The maximum number of documents to route to per field.

    Returns:
        A list of RoutingEntry, one per resolved field, ordered by field key ascending.
    """
    # Filter to readable docs only
    readable_docs = [d for d in doc_index if _is_readable(d)]

    # Pre-compute doc texts for readable docs
    doc_texts: dict[str, str] = {}
    for doc in readable_docs:
        doc_texts[doc.doc_id] = _build_doc_text(doc.doc_id, layout)

    routing_entries: list[RoutingEntry] = []

    # Process fields in sorted order by key
    sorted_fields = sorted(schema.resolved_fields, key=lambda f: f.key)

    for field in sorted_fields:
        query = _build_field_query(field)

        if not readable_docs:
            # No readable docs: empty routing entry
            routing_entries.append(RoutingEntry(
                field=field.key,
                doc_ids=[],
                scores={},
            ))
            continue

        # Score each readable doc
        scored_docs: list[tuple[str, float]] = []
        for doc in readable_docs:
            doc_text = doc_texts.get(doc.doc_id, "")
            score = score_query_doc(query, doc_text)
            scored_docs.append((doc.doc_id, score))

        # Sort by: descending score, then ascending doc_id (tie-breaker)
        scored_docs.sort(key=lambda x: (-x[1], x[0]))

        # Take top_k
        top_docs = scored_docs[:top_k]

        # Build doc_ids list (ordered best to worst) and scores dict
        doc_ids = [doc_id for doc_id, _ in top_docs]
        scores = {doc_id: score for doc_id, score in top_docs}

        routing_entries.append(RoutingEntry(
            field=field.key,
            doc_ids=doc_ids,
            scores=scores,
        ))

    return routing_entries


def write_routing_artifact(run_paths: RunPaths, routing: list[RoutingEntry]) -> None:
    """
    Write the routing artifact to artifacts/routing.json.

    Args:
        run_paths: The RunPaths for this run.
        routing: The list of RoutingEntry to write.
    """
    path = run_paths.artifact_path("routing.json")
    data = [entry.model_dump() for entry in routing]
    write_json_atomic(path, data)


def run_routing(
    run_paths: RunPaths,
    schema: ResolvedSchema,
    doc_index: list[DocIndexItem],
    layout: list[LayoutDoc],
    top_k: int,
    trace: TraceLogger,
) -> list[RoutingEntry]:
    """
    Execute the routing step with tracing.

    Wraps route_docs in a trace step named 'route_docs', writes the artifact,
    and handles status (ok/warn/error).

    Args:
        run_paths: The RunPaths for this run.
        schema: The resolved schema.
        doc_index: The document index.
        layout: The layout documents.
        top_k: The maximum number of documents per field.
        trace: The TraceLogger for this run.

    Returns:
        The list of RoutingEntry produced.
    """
    # Determine if there are any readable docs (for status)
    readable_count = sum(1 for d in doc_index if _is_readable(d))
    has_readable_docs = readable_count > 0

    # Determine input/output refs
    inputs_ref = [
        f"runs/{run_paths.run_id}/artifacts/schema.json",
        f"runs/{run_paths.run_id}/artifacts/doc_index.json",
        f"runs/{run_paths.run_id}/artifacts/layout.json",
    ]
    outputs_ref = [f"runs/{run_paths.run_id}/artifacts/routing.json"]

    routing: list[RoutingEntry] = []

    # We need custom handling for the warn status, so we wrap manually
    # instead of using trace_step directly with an exception
    import time
    from datetime import datetime, timezone

    start_time = time.perf_counter()
    status = "ok"
    error_info: dict[str, str] | None = None

    try:
        routing = route_docs(schema, doc_index, layout, top_k=top_k)
        write_routing_artifact(run_paths, routing)

        # Set status to warn if no readable docs
        if not has_readable_docs:
            status = "warn"

    except Exception as exc:
        status = "error"
        error_info = {
            "kind": exc.__class__.__name__,
            "message": str(exc),
        }
        raise
    finally:
        end_time = time.perf_counter()
        duration_ms = int((end_time - start_time) * 1000)

        ts = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")

        event: dict[str, object] = {
            "ts": ts,
            "run_id": trace.run_id,
            "step": "route_docs",
            "status": status,
            "duration_ms": duration_ms,
            "inputs_ref": inputs_ref,
            "outputs_ref": outputs_ref,
        }

        if error_info is not None:
            event["error"] = error_info

        trace.append(event)

    return routing

