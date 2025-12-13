"""
Candidate extraction orchestrator.

Coordinates heuristic extraction and optional LLM calls to produce
candidates.json for each run. Enforces evidence-first invariant and
deterministic hallucination checking.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from app.excerpts import build_excerpts_for_field
from app.heuristics import (
    extract_digits,
    heuristic_candidates_for_field,
    normalize_date,
    normalize_text,
)
from app.llm_client import LLMClient, LLMInvalidJSONError
from app.models import (
    Candidate,
    FieldSpec,
    LayoutDoc,
    ResolvedSchema,
    RoutingEntry,
    RunOptions,
)
from app.runfs import RunPaths, read_json, write_json_atomic
from app.trace import TraceLogger

if TYPE_CHECKING:
    pass


# --- Constants ---

# Autofill threshold for deciding whether to call LLM
AUTOFILL_THRESHOLD = 0.75

# Excerpt capping defaults
DEFAULT_MAX_TOTAL_CHARS = 8000
DEFAULT_MAX_CHARS_PER_DOC = 4000
DEFAULT_MAX_PAGES_PER_DOC = 3


# --- Evidence checking (deterministic hallucination check) ---


def _normalize_for_evidence_check(text: str) -> str:
    """
    Normalize text for evidence comparison.

    Rules:
    - Lowercase
    - Collapse whitespace
    - Strip punctuation except hyphens
    """
    return normalize_text(text)


def _evidence_supports_string(normalized_value: str, evidence_texts: list[str]) -> bool:
    """Check if normalized_value is substring of any normalized evidence text."""
    norm_value = _normalize_for_evidence_check(normalized_value)

    for ev_text in evidence_texts:
        norm_ev = _normalize_for_evidence_check(ev_text)
        if norm_value in norm_ev:
            return True

    return False


def _evidence_supports_date(normalized_value: str, evidence_texts: list[str]) -> bool:
    """
    Check if normalized date matches date-like substring in evidence.

    The normalized_value should be in YYYY-MM-DD format.
    """
    # Parse the normalized value
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", normalized_value)
    if not match:
        return False

    year, month, day = match.groups()

    # Check each evidence text for matching date patterns
    for ev_text in evidence_texts:
        # Look for various date representations in evidence

        # YYYY-MM-DD or YYYY/MM/DD
        if re.search(rf"{year}[-/]{month}[-/]{day}", ev_text):
            return True
        if re.search(rf"{year}[-/]{int(month)}[-/]{int(day)}", ev_text):
            return True

        # MM/DD/YYYY or MM-DD-YYYY
        if re.search(rf"{month}[-/]{day}[-/]{year}", ev_text):
            return True
        if re.search(rf"{int(month)}[-/]{int(day)}[-/]{year}", ev_text):
            return True

        # Month name formats (simplified check)
        # Just check that year, month num, and day are all present
        ev_lower = ev_text.lower()
        if year in ev_text and day.lstrip("0") in ev_text:
            # Month names
            month_names = {
                "01": ["jan", "january"],
                "02": ["feb", "february"],
                "03": ["mar", "march"],
                "04": ["apr", "april"],
                "05": ["may"],
                "06": ["jun", "june"],
                "07": ["jul", "july"],
                "08": ["aug", "august"],
                "09": ["sep", "september"],
                "10": ["oct", "october"],
                "11": ["nov", "november"],
                "12": ["dec", "december"],
            }
            month_strs = month_names.get(month, [])
            for m_str in month_strs:
                if m_str in ev_lower:
                    return True

    return False


def _evidence_supports_phone(normalized_value: str, evidence_texts: list[str]) -> bool:
    """
    Check if normalized phone digits match evidence digits.

    Allows separators in evidence (dashes, spaces, parens, dots).
    """
    norm_digits = extract_digits(normalized_value)

    # Need at least 10 digits
    if len(norm_digits) < 10:
        return False

    for ev_text in evidence_texts:
        ev_digits = extract_digits(ev_text)

        # Check if normalized digits are contained in evidence digits
        # Account for country code differences
        if norm_digits in ev_digits:
            return True

        # Also check without leading country code
        if norm_digits.startswith("1") and norm_digits[1:] in ev_digits:
            return True

        # And check if evidence has country code we added
        if len(norm_digits) == 11 and norm_digits.startswith("1"):
            base_digits = norm_digits[1:]
            if base_digits in ev_digits:
                return True

    return False


def _evidence_supports_list(normalized_value: str, evidence_texts: list[str]) -> bool:
    """
    Check if list items are all present in evidence.

    If normalized_value represents a list (comma/semicolon separated),
    check each item. Otherwise, use substring rule.
    """
    # Check if it looks like a list (contains comma or semicolon)
    if "," in normalized_value or ";" in normalized_value:
        # Split into items
        items = re.split(r"[,;]", normalized_value)
        items = [item.strip() for item in items if item.strip()]

        if not items:
            return False

        # All items must be found in evidence
        combined_evidence = " ".join(evidence_texts)
        norm_evidence = _normalize_for_evidence_check(combined_evidence)

        for item in items:
            norm_item = _normalize_for_evidence_check(item)
            if norm_item and norm_item not in norm_evidence:
                return False

        return True
    else:
        # Use substring rule
        return _evidence_supports_string(normalized_value, evidence_texts)


def evidence_supports_value(field: FieldSpec, candidate: Candidate) -> bool:
    """
    Check if candidate's evidence supports its value (deterministic hallucination check).

    Rules (from L2 spec):
    - Evidence list must be non-empty with doc_id/page/quoted_text
    - String fields: normalized_value substring in normalized(quoted_text)
    - DOB: normalized parsed date matches date-like substring in evidence
    - Phone: normalized digits match evidence digits (allowing separators)
    - List fields: every item must appear as substring in evidence

    Args:
        field: The field specification.
        candidate: The candidate to check.

    Returns:
        True if evidence supports the value, False otherwise.
    """
    # Check evidence is non-empty
    if not candidate.evidence:
        return False

    # Check each evidence has required fields
    evidence_texts: list[str] = []
    for ev in candidate.evidence:
        if not ev.doc_id or not ev.page or not ev.quoted_text:
            return False
        evidence_texts.append(ev.quoted_text)

    if not evidence_texts:
        return False

    # Field-specific checks
    if field.type == "date":
        return _evidence_supports_date(candidate.normalized_value, evidence_texts)
    elif field.type == "phone":
        return _evidence_supports_phone(candidate.normalized_value, evidence_texts)
    elif field.type == "string_or_list":
        return _evidence_supports_list(candidate.normalized_value, evidence_texts)
    else:  # string
        return _evidence_supports_string(candidate.normalized_value, evidence_texts)


def _compute_provisional_confidence(candidate: Candidate) -> float:
    """
    Compute provisional confidence for LLM decision.

    Per spec: 0.45*anchor_match + 0.25*doc_relevance + 0.30*validator
    But doc_relevance and validator are 0 in PR-06, so: 0.45*anchor_match only.
    """
    return 0.45 * candidate.scores.anchor_match


# --- Artifact loading ---


def _load_schema(run_paths: RunPaths) -> ResolvedSchema:
    """Load schema.json artifact."""
    path = run_paths.artifact_path("schema.json")
    data = read_json(path)
    return ResolvedSchema.model_validate(data)


def _load_layout(run_paths: RunPaths) -> list[LayoutDoc]:
    """Load layout.json artifact."""
    path = run_paths.artifact_path("layout.json")
    data = read_json(path)
    return [LayoutDoc.model_validate(item) for item in data]


def _load_routing(run_paths: RunPaths) -> list[RoutingEntry]:
    """Load routing.json artifact."""
    path = run_paths.artifact_path("routing.json")
    data = read_json(path)
    return [RoutingEntry.model_validate(item) for item in data]


def _get_routed_docs(
    field_key: str,
    routing: list[RoutingEntry],
    layout: list[LayoutDoc],
) -> list[LayoutDoc]:
    """Get layout docs routed to a field, in routing order."""
    # Find routing entry for this field
    entry = None
    for r in routing:
        if r.field == field_key:
            entry = r
            break

    if entry is None or not entry.doc_ids:
        return []

    # Build doc_id -> LayoutDoc map
    layout_map = {doc.doc_id: doc for doc in layout}

    # Return docs in routing order
    result: list[LayoutDoc] = []
    for doc_id in entry.doc_ids:
        if doc_id in layout_map:
            result.append(layout_map[doc_id])

    return result


# --- Main orchestration ---


def _sort_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """
    Sort candidates deterministically.

    Order:
    - Primary: field asc
    - Secondary: from_method (heuristic before llm)
    - Tertiary: normalized_value asc
    """
    method_order = {"heuristic": 0, "llm": 1}

    return sorted(
        candidates,
        key=lambda c: (
            c.field,
            method_order.get(c.from_method, 2),
            c.normalized_value,
        ),
    )


def extract_candidates_for_run(
    run_id: str,
    *,
    run_options: RunOptions,
    llm_client: LLMClient,
    base_dir: str | None = None,
) -> list[Candidate]:
    """
    Extract candidates for a run.

    Reads schema/layout/routing artifacts, generates candidates per field
    with heuristic + optional LLM, applies evidence enforcement, and
    writes artifacts/candidates.json.

    Args:
        run_id: The run identifier.
        run_options: Run configuration options.
        llm_client: LLM client for extraction calls.
        base_dir: Optional base directory for runs (for testing).

    Returns:
        The list of all candidates (including rejected ones).
    """
    from app.runfs import create_run, get_runs_base_dir
    from pathlib import Path

    # Get run paths
    if base_dir:
        run_paths = create_run(run_id=run_id, base_dir=Path(base_dir))
    else:
        run_paths = create_run(run_id=run_id, base_dir=get_runs_base_dir())

    # Initialize trace logger
    trace = TraceLogger(run_paths)

    # Overall timing
    overall_start = time.perf_counter()
    all_candidates: list[Candidate] = []
    field_stats: dict[str, dict[str, Any]] = {}

    try:
        # Load artifacts
        schema = _load_schema(run_paths)
        layout = _load_layout(run_paths)
        routing = _load_routing(run_paths)

        # Process each resolved field (respecting max_fields)
        fields_to_process = schema.resolved_fields[:run_options.max_fields]

        for field in fields_to_process:
            field_start = time.perf_counter()
            field_candidates: list[Candidate] = []
            llm_used = False

            # Get routed docs for this field
            routed_docs = _get_routed_docs(field.key, routing, layout)

            if not routed_docs:
                # No docs routed - skip to next field
                field_stats[field.key] = {
                    "heuristic_count": 0,
                    "llm_used": False,
                    "accepted_count": 0,
                    "rejected_count": 0,
                }
                continue

            # --- Heuristic pass ---
            heuristic_start = time.perf_counter()
            heuristic_candidates = heuristic_candidates_for_field(field, routed_docs)
            heuristic_duration = int((time.perf_counter() - heuristic_start) * 1000)

            # Apply evidence check to heuristic candidates
            accepted_heuristic: list[Candidate] = []
            for candidate in heuristic_candidates:
                if not evidence_supports_value(field, candidate):
                    # Mark as rejected
                    candidate.rejected_reasons.append("unsupported_by_evidence")
                else:
                    accepted_heuristic.append(candidate)
                field_candidates.append(candidate)

            # Log heuristic step
            trace.append({
                "ts": datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
                "run_id": run_id,
                "step": f"field:{field.key}:heuristic",
                "status": "ok",
                "duration_ms": heuristic_duration,
                "inputs_ref": [],
                "outputs_ref": [],
            })

            # --- Decide whether to call LLM ---
            should_call_llm = False

            if not accepted_heuristic:
                # No acceptable candidates from heuristic
                should_call_llm = True
            else:
                # Check best heuristic confidence
                best_confidence = max(
                    _compute_provisional_confidence(c) for c in accepted_heuristic
                )
                if best_confidence < AUTOFILL_THRESHOLD:
                    should_call_llm = True

            # --- LLM pass (if needed) ---
            if should_call_llm:
                llm_used = True
                llm_start = time.perf_counter()
                llm_error: dict[str, str] | None = None

                try:
                    # Build excerpts for LLM
                    excerpts = build_excerpts_for_field(
                        field,
                        routed_docs,
                        max_total_chars=DEFAULT_MAX_TOTAL_CHARS,
                        max_chars_per_doc=DEFAULT_MAX_CHARS_PER_DOC,
                        max_pages_per_doc=DEFAULT_MAX_PAGES_PER_DOC,
                    )

                    if excerpts:
                        llm_candidates = llm_client.extract_candidates(
                            field,
                            excerpts,
                            run_options=run_options,
                        )

                        # Apply evidence check to LLM candidates
                        for candidate in llm_candidates:
                            if not evidence_supports_value(field, candidate):
                                candidate.rejected_reasons.append("unsupported_by_evidence")
                            field_candidates.append(candidate)

                except LLMInvalidJSONError as e:
                    llm_error = {
                        "kind": "LLMInvalidJSONError",
                        "message": str(e),
                    }

                llm_duration = int((time.perf_counter() - llm_start) * 1000)

                # Log LLM step
                llm_event: dict[str, Any] = {
                    "ts": datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
                    "run_id": run_id,
                    "step": f"field:{field.key}:llm",
                    "status": "error" if llm_error else "ok",
                    "duration_ms": llm_duration,
                    "inputs_ref": [],
                    "outputs_ref": [],
                }
                if llm_error:
                    llm_event["error"] = llm_error
                trace.append(llm_event)

            # Collect stats
            accepted_count = sum(1 for c in field_candidates if not c.rejected_reasons)
            rejected_count = sum(1 for c in field_candidates if c.rejected_reasons)

            field_stats[field.key] = {
                "heuristic_count": len(heuristic_candidates),
                "llm_used": llm_used,
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
            }

            all_candidates.extend(field_candidates)

        # Sort candidates deterministically
        all_candidates = _sort_candidates(all_candidates)

        # Write candidates.json
        candidates_data = [c.model_dump() for c in all_candidates]
        write_json_atomic(run_paths.artifact_path("candidates.json"), candidates_data)

        overall_duration = int((time.perf_counter() - overall_start) * 1000)

        # Log overall step
        trace.append({
            "ts": datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
            "run_id": run_id,
            "step": "extract_candidates",
            "status": "ok",
            "duration_ms": overall_duration,
            "inputs_ref": [
                f"runs/{run_id}/artifacts/schema.json",
                f"runs/{run_id}/artifacts/layout.json",
                f"runs/{run_id}/artifacts/routing.json",
            ],
            "outputs_ref": [f"runs/{run_id}/artifacts/candidates.json"],
        })

    except Exception as exc:
        overall_duration = int((time.perf_counter() - overall_start) * 1000)

        # Log error
        trace.append({
            "ts": datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
            "run_id": run_id,
            "step": "extract_candidates",
            "status": "error",
            "duration_ms": overall_duration,
            "inputs_ref": [],
            "outputs_ref": [],
            "error": {
                "kind": exc.__class__.__name__,
                "message": str(exc),
            },
        })
        raise

    return all_candidates

