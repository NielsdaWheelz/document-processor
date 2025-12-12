"""
Schema resolution for the document-processor pipeline.

Resolves the schema for a run in strict precedence:
1. User-provided schema_json (highest priority)
2. First fillable PDF in target_docs with AcroForm fields
3. Fallback to V1 supported field set

Produces artifacts/schema.json containing ResolvedSchema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.models import (
    FieldSpec,
    FieldType,
    ResolvedSchema,
    RunOptions,
    SchemaSource,
    SUPPORTED_FIELD_KEYS,
)
from app.runfs import RunPaths, write_json_atomic
from app.trace import TraceLogger, trace_step

if TYPE_CHECKING:
    pass


# Fixed field ordering for fallback and max_fields capping
FIELD_ORDER: list[str] = [
    "full_name",
    "dob",
    "phone",
    "address",
    "insurance_member_id",
    "allergies",
    "medications",
]

# Default field types for V1 supported fields
FIELD_TYPES: dict[str, FieldType] = {
    "full_name": "string",
    "dob": "date",
    "phone": "phone",
    "address": "string",
    "insurance_member_id": "string",
    "allergies": "string_or_list",
    "medications": "string_or_list",
}

# Alias map for AcroForm field matching (fixed)
# Each key maps to a list of substrings that indicate that field
FIELD_ALIASES: dict[str, list[str]] = {
    "full_name": ["full_name", "name", "patient_name"],
    "dob": ["dob", "date_of_birth", "birthdate"],
    "phone": ["phone", "mobile", "telephone"],
    "address": ["address", "street"],
    "insurance_member_id": ["insurance_member_id", "member_id", "policy", "insurance_id"],
    "allergies": ["allergies", "allergy"],
    "medications": ["medications", "meds"],
}


class SchemaWarning:
    """Represents a warning during schema resolution."""

    def __init__(self, kind: str, message: str, details: dict[str, Any] | None = None):
        self.kind = kind
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind, "message": self.message}
        if self.details:
            result["details"] = self.details
        return result


def _order_and_cap_fields(
    fields: list[FieldSpec],
    max_fields: int,
) -> list[FieldSpec]:
    """
    Order fields by the fixed FIELD_ORDER and cap to max_fields.

    Args:
        fields: List of FieldSpec to order and cap.
        max_fields: Maximum number of fields to return.

    Returns:
        Ordered and capped list of FieldSpec.
    """
    # Create a map for O(1) lookup
    field_map = {f.key: f for f in fields}

    # Order by FIELD_ORDER, keeping only fields that exist
    ordered = []
    for key in FIELD_ORDER:
        if key in field_map:
            ordered.append(field_map[key])

    # Cap to max_fields
    return ordered[:max_fields]


def _make_field_spec(key: str, label: str | None = None) -> FieldSpec:
    """
    Create a FieldSpec for a supported key with the correct type.

    Args:
        key: The field key (must be in SUPPORTED_FIELD_KEYS).
        label: Optional human-readable label.

    Returns:
        A FieldSpec with the correct type for the key.
    """
    return FieldSpec(
        key=key,
        label=label,
        type=FIELD_TYPES[key],
    )


def parse_user_schema(
    schema_json_bytes: bytes,
    options: RunOptions,
    warnings: list[SchemaWarning],
) -> ResolvedSchema | None:
    """
    Parse user-provided schema JSON bytes into a ResolvedSchema.

    Expected format:
    {
        "fields": [
            {"key": "full_name", "label": "Patient Name", "type": "string"},
            ...
        ]
    }

    Args:
        schema_json_bytes: Raw bytes of the schema JSON.
        options: RunOptions for max_fields cap.
        warnings: List to append any warnings to.

    Returns:
        ResolvedSchema if parsing succeeds, None if invalid.
    """
    try:
        data = json.loads(schema_json_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        warnings.append(SchemaWarning(
            kind="user_schema_invalid",
            message=f"Failed to parse schema JSON: {e}",
        ))
        return None

    # Validate expected shape
    if not isinstance(data, dict) or "fields" not in data:
        warnings.append(SchemaWarning(
            kind="user_schema_invalid",
            message="Schema JSON must be an object with a 'fields' array",
        ))
        return None

    fields_data = data.get("fields")
    if not isinstance(fields_data, list):
        warnings.append(SchemaWarning(
            kind="user_schema_invalid",
            message="'fields' must be an array",
        ))
        return None

    resolved_fields: list[FieldSpec] = []
    unsupported_fields: list[str] = []

    for item in fields_data:
        if not isinstance(item, dict):
            continue

        key = item.get("key")
        if not isinstance(key, str):
            continue

        # Check if key is supported
        if key not in SUPPORTED_FIELD_KEYS:
            unsupported_fields.append(key)
            continue

        # Extract label and type
        label = item.get("label")
        if label is not None and not isinstance(label, str):
            label = None

        field_type = item.get("type")
        # Use the correct type for the key (from FIELD_TYPES), not user-provided type
        # This ensures type consistency
        resolved_fields.append(_make_field_spec(key, label))

    # Order and cap fields
    resolved_fields = _order_and_cap_fields(resolved_fields, options.max_fields)

    return ResolvedSchema(
        schema_source="user_schema",
        resolved_fields=resolved_fields,
        unsupported_fields=unsupported_fields,
    )


def _match_acroform_field_to_key(
    field_name: str,
    warnings: list[SchemaWarning],
) -> str | None:
    """
    Match an AcroForm field name to a V1 supported key using alias matching.

    Matching algorithm:
    1. Lowercase the field name
    2. For each supported key, check if any of its aliases appear as substrings
    3. If exactly 1 key matches -> return it
    4. If 0 matches -> return None
    5. If >1 matches -> ambiguous, add warning, return None

    Args:
        field_name: The AcroForm field name to match.
        warnings: List to append any warnings to.

    Returns:
        The matched key, or None if no match or ambiguous.
    """
    normalized_name = field_name.lower()

    matched_keys: list[str] = []

    for key, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            if alias in normalized_name:
                matched_keys.append(key)
                break  # Only count each key once

    if len(matched_keys) == 1:
        return matched_keys[0]
    elif len(matched_keys) > 1:
        warnings.append(SchemaWarning(
            kind="acroform_ambiguous_field",
            message=f"Field '{field_name}' matches multiple keys: {matched_keys}",
            details={"field_name": field_name, "matched_keys": matched_keys},
        ))
        return None
    else:
        return None


def _extract_acroform_fields(pdf_path: Path) -> list[str] | None:
    """
    Extract AcroForm field names from a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of field names if the PDF has AcroForm fields, None otherwise.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        # pypdf not available
        return None

    try:
        reader = PdfReader(pdf_path)

        # Check if PDF has AcroForm
        if reader.get_fields() is None:
            return None

        fields = reader.get_fields()
        if not fields:
            return None

        return list(fields.keys())
    except Exception:
        # Any error reading the PDF
        return None


def resolve_from_acroform(
    target_pdf_path: Path,
    options: RunOptions,
    warnings: list[SchemaWarning],
) -> ResolvedSchema | None:
    """
    Resolve schema from AcroForm fields in a PDF (read-only).

    Args:
        target_pdf_path: Path to the target PDF.
        options: RunOptions for max_fields cap.
        warnings: List to append any warnings to.

    Returns:
        ResolvedSchema if the PDF has mappable AcroForm fields, None otherwise.
    """
    try:
        field_names = _extract_acroform_fields(target_pdf_path)
    except Exception as e:
        warnings.append(SchemaWarning(
            kind="acroform_read_error",
            message=f"Failed to read AcroForm fields from {target_pdf_path.name}: {e}",
        ))
        return None

    if field_names is None:
        return None

    # Map AcroForm fields to V1 keys
    matched_keys: set[str] = set()
    for field_name in field_names:
        key = _match_acroform_field_to_key(field_name, warnings)
        if key is not None:
            matched_keys.add(key)

    if not matched_keys:
        return None

    # Create FieldSpecs for matched keys
    resolved_fields = [_make_field_spec(key) for key in matched_keys]

    # Order and cap fields
    resolved_fields = _order_and_cap_fields(resolved_fields, options.max_fields)

    return ResolvedSchema(
        schema_source="fillable_pdf",
        resolved_fields=resolved_fields,
        unsupported_fields=[],
    )


def _find_fillable_pdfs(target_docs_dir: Path) -> list[Path]:
    """
    Find PDF files in the target_docs directory.

    Args:
        target_docs_dir: Path to the target_docs directory.

    Returns:
        List of PDF file paths, sorted by name for deterministic order.
    """
    if not target_docs_dir.exists():
        return []

    pdfs = list(target_docs_dir.glob("*.pdf")) + list(target_docs_dir.glob("*.PDF"))
    # Sort for deterministic order
    return sorted(pdfs, key=lambda p: p.name.lower())


def _resolve_fallback_v1(options: RunOptions) -> ResolvedSchema:
    """
    Create fallback V1 schema with all supported fields.

    Args:
        options: RunOptions for max_fields cap.

    Returns:
        ResolvedSchema with fallback_v1 source.
    """
    resolved_fields = [_make_field_spec(key) for key in FIELD_ORDER]

    # Cap to max_fields
    resolved_fields = resolved_fields[:options.max_fields]

    return ResolvedSchema(
        schema_source="fallback_v1",
        resolved_fields=resolved_fields,
        unsupported_fields=[],
    )


def resolve_schema(
    run_paths: RunPaths,
    schema_json_bytes: bytes | None,
    options: RunOptions,
    trace: TraceLogger,
) -> ResolvedSchema:
    """
    Resolve schema for a run in strict precedence order.

    Precedence:
    1. User-provided schema_json (if valid)
    2. First fillable PDF in target_docs with mappable AcroForm fields
    3. Fallback to V1 supported field set

    Writes the resolved schema to artifacts/schema.json.

    Args:
        run_paths: RunPaths for this run.
        schema_json_bytes: Optional user-provided schema JSON bytes.
        options: RunOptions for max_fields cap.
        trace: TraceLogger for logging events.

    Returns:
        The resolved ResolvedSchema.
    """
    warnings: list[SchemaWarning] = []
    resolved: ResolvedSchema | None = None

    with trace_step(
        trace,
        step="resolve_schema",
        inputs_ref=["schema_json"] if schema_json_bytes else [],
        outputs_ref=[f"runs/{run_paths.run_id}/artifacts/schema.json"],
    ):
        # 1. Try user schema (highest precedence)
        if schema_json_bytes is not None:
            resolved = parse_user_schema(schema_json_bytes, options, warnings)

        # 2. Try fillable PDF in target_docs
        if resolved is None:
            target_docs_dir = run_paths.target_docs_dir()
            for pdf_path in _find_fillable_pdfs(target_docs_dir):
                result = resolve_from_acroform(pdf_path, options, warnings)
                if result is not None:
                    resolved = result
                    break

        # 3. Fallback to V1
        if resolved is None:
            resolved = _resolve_fallback_v1(options)

        # Log warnings to trace
        for warning in warnings:
            trace.append({
                "ts": _utc_iso_timestamp(),
                "run_id": trace.run_id,
                "step": "resolve_schema",
                "level": "warn",
                **warning.to_dict(),
            })

        # Write artifact
        schema_path = run_paths.artifact_path("schema.json")
        write_json_atomic(schema_path, resolved.model_dump())

    return resolved


def _utc_iso_timestamp() -> str:
    """Get current UTC timestamp in ISO-8601 format."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
