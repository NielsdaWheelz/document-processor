"""
Unit tests for schema resolution.

Tests cover:
1. User schema precedence over fillable PDF
2. Invalid user schema falls through to next source
3. AcroForm alias single match
4. AcroForm ambiguity skip + trace warning
5. Fallback ordering + max_fields cap

Uses real PDF fixtures from tests/fixtures/ (copied from waive/ sample data).
"""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from pypdf import PdfReader

from app.models import FieldSpec, ResolvedSchema, RunOptions
from app.runfs import RunPaths, create_run, read_json
from app.schema_resolver import (
    FIELD_ALIASES,
    FIELD_ORDER,
    FIELD_TYPES,
    _extract_acroform_fields,
    _match_acroform_field_to_key,
    _order_and_cap_fields,
    parse_user_schema,
    resolve_from_acroform,
    resolve_schema,
    SchemaWarning,
)
from app.trace import TraceLogger


# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------

# Path to sample PDFs in waive/ directory
FIXTURES_DIR = Path(__file__).parent.parent.parent / "waive"


def make_user_schema_bytes(fields: list[dict[str, Any]]) -> bytes:
    """Create user schema JSON bytes."""
    return json.dumps({"fields": fields}).encode("utf-8")


def copy_fixture(name: str, dest_dir: Path) -> Path:
    """Copy a fixture file to destination directory."""
    src = FIXTURES_DIR / name
    if not src.exists():
        pytest.skip(f"{name} fixture not found")
    dest = dest_dir / name
    shutil.copy(src, dest)
    return dest


@pytest.fixture
def tmp_run(tmp_path: Path) -> RunPaths:
    """Create a temporary run for testing."""
    return create_run(run_id="test-run", base_dir=tmp_path)


@pytest.fixture
def trace(tmp_run: RunPaths) -> TraceLogger:
    """Create a trace logger for testing."""
    return TraceLogger(tmp_run)


# ---------------------------------------------------------------------------
# Test: User schema precedence
# ---------------------------------------------------------------------------


class TestUserSchemaPrecedence:
    """Tests for user schema taking precedence over other sources."""

    def test_user_schema_wins_over_fillable_pdf(
        self, tmp_run: RunPaths, trace: TraceLogger
    ):
        """
        Given schema_json with fields and a fillable PDF present,
        expect schema_source='user_schema' and resolved_fields from schema_json.
        """
        # Create user schema
        user_schema = make_user_schema_bytes([
            {"key": "allergies", "label": "Allergies", "type": "string_or_list"},
            {"key": "medications", "label": "Meds", "type": "string_or_list"},
        ])

        # Copy real fillable PDF to target_docs
        copy_fixture("form_fillable.pdf", tmp_run.target_docs_dir())

        # Resolve schema
        options = RunOptions()
        result = resolve_schema(tmp_run, user_schema, options, trace)

        # Assert user schema won (not fillable_pdf)
        assert result.schema_source == "user_schema"
        field_keys = [f.key for f in result.resolved_fields]
        assert field_keys == ["allergies", "medications"]  # In FIELD_ORDER

    def test_user_schema_filters_unsupported_fields(
        self, tmp_run: RunPaths, trace: TraceLogger
    ):
        """User schema with unsupported fields records them in unsupported_fields."""
        user_schema = make_user_schema_bytes([
            {"key": "full_name", "label": "Name", "type": "string"},
            {"key": "unknown_field", "label": "Unknown", "type": "string"},
            {"key": "another_bad", "label": "Bad", "type": "string"},
        ])

        options = RunOptions()
        result = resolve_schema(tmp_run, user_schema, options, trace)

        assert result.schema_source == "user_schema"
        assert len(result.resolved_fields) == 1
        assert result.resolved_fields[0].key == "full_name"
        assert "unknown_field" in result.unsupported_fields
        assert "another_bad" in result.unsupported_fields

    def test_user_schema_respects_max_fields(
        self, tmp_run: RunPaths, trace: TraceLogger
    ):
        """User schema with more fields than max_fields is capped."""
        user_schema = make_user_schema_bytes([
            {"key": "full_name", "label": "Name", "type": "string"},
            {"key": "dob", "label": "DOB", "type": "date"},
            {"key": "phone", "label": "Phone", "type": "phone"},
            {"key": "address", "label": "Address", "type": "string"},
        ])

        options = RunOptions(max_fields=2)
        result = resolve_schema(tmp_run, user_schema, options, trace)

        assert result.schema_source == "user_schema"
        assert len(result.resolved_fields) == 2
        # Should be capped to first 2 in FIELD_ORDER
        field_keys = [f.key for f in result.resolved_fields]
        assert field_keys == ["full_name", "dob"]


# ---------------------------------------------------------------------------
# Test: Invalid user schema falls through
# ---------------------------------------------------------------------------


class TestUserSchemaInvalid:
    """Tests for invalid user schema falling through to next source."""

    def test_malformed_json_falls_through(
        self, tmp_run: RunPaths, trace: TraceLogger
    ):
        """Invalid JSON falls through to fillable PDF source."""
        invalid_json = b"{ not valid json }"

        # Copy real fillable PDF
        copy_fixture("form_fillable.pdf", tmp_run.target_docs_dir())

        options = RunOptions()
        result = resolve_schema(tmp_run, invalid_json, options, trace)

        # Should fall through to fillable_pdf
        assert result.schema_source == "fillable_pdf"
        field_keys = {f.key for f in result.resolved_fields}
        assert "full_name" in field_keys  # form_fillable.pdf has Name field

        # Check trace has warning
        trace_content = tmp_run.trace_jsonl_path().read_text()
        assert "user_schema_invalid" in trace_content

    def test_missing_fields_key_falls_through(
        self, tmp_run: RunPaths, trace: TraceLogger
    ):
        """JSON without 'fields' key falls through."""
        invalid_schema = json.dumps({"something": "else"}).encode()

        # Copy real fillable PDF
        copy_fixture("form_fillable.pdf", tmp_run.target_docs_dir())

        options = RunOptions()
        result = resolve_schema(tmp_run, invalid_schema, options, trace)

        assert result.schema_source == "fillable_pdf"

    def test_fields_not_array_falls_through(
        self, tmp_run: RunPaths, trace: TraceLogger
    ):
        """JSON with 'fields' not an array falls through."""
        invalid_schema = json.dumps({"fields": "not an array"}).encode()

        copy_fixture("form_fillable.pdf", tmp_run.target_docs_dir())

        options = RunOptions()
        result = resolve_schema(tmp_run, invalid_schema, options, trace)

        assert result.schema_source == "fillable_pdf"

    def test_invalid_falls_through_to_fallback_when_no_pdf(
        self, tmp_run: RunPaths, trace: TraceLogger
    ):
        """Invalid user schema falls through to fallback when no fillable PDF."""
        invalid_json = b"not json at all"

        options = RunOptions()
        result = resolve_schema(tmp_run, invalid_json, options, trace)

        assert result.schema_source == "fallback_v1"


# ---------------------------------------------------------------------------
# Test: AcroForm alias single match
# ---------------------------------------------------------------------------


class TestAcroFormAliasSingleMatch:
    """Tests for AcroForm field name matching via aliases."""

    def test_exact_alias_match(self):
        """Field name containing exactly one alias matches that key."""
        warnings: list[SchemaWarning] = []

        # 'patient_name' contains 'patient_name' alias for full_name
        result = _match_acroform_field_to_key("patient_name", warnings)
        assert result == "full_name"
        assert len(warnings) == 0

    def test_substring_alias_match(self):
        """Field name containing an alias as substring matches."""
        warnings: list[SchemaWarning] = []

        # 'user_phone_number' contains 'phone' alias
        result = _match_acroform_field_to_key("user_phone_number", warnings)
        assert result == "phone"
        assert len(warnings) == 0

    def test_case_insensitive_match(self):
        """Matching is case-insensitive."""
        warnings: list[SchemaWarning] = []

        result = _match_acroform_field_to_key("DATE_OF_BIRTH", warnings)
        assert result == "dob"
        assert len(warnings) == 0

    def test_no_match_returns_none(self):
        """Field name matching no aliases returns None."""
        warnings: list[SchemaWarning] = []

        result = _match_acroform_field_to_key("random_field_xyz", warnings)
        assert result is None
        assert len(warnings) == 0

    def test_each_supported_key_has_aliases(self):
        """Each supported key in FIELD_ORDER has aliases defined."""
        for key in FIELD_ORDER:
            assert key in FIELD_ALIASES
            assert len(FIELD_ALIASES[key]) > 0

    def test_resolve_from_acroform_with_real_pdf(self, tmp_run: RunPaths):
        """Real fillable PDF resolves to expected V1 keys."""
        pdf_path = copy_fixture("form_fillable.pdf", tmp_run.target_docs_dir())

        warnings: list[SchemaWarning] = []
        options = RunOptions()
        result = resolve_from_acroform(pdf_path, options, warnings)

        assert result is not None
        assert result.schema_source == "fillable_pdf"

        field_keys = {f.key for f in result.resolved_fields}
        # form_fillable.pdf has: First Name, Address, Date_of_birth_*, phone*
        assert "full_name" in field_keys
        assert "address" in field_keys
        assert "dob" in field_keys
        assert "phone" in field_keys


# ---------------------------------------------------------------------------
# Test: AcroForm ambiguity skip + trace warn
# ---------------------------------------------------------------------------


class TestAcroFormAmbiguity:
    """Tests for ambiguous AcroForm field name handling."""

    def test_ambiguous_field_skipped_and_warned(self):
        """Field matching multiple keys is skipped with warning."""
        warnings: list[SchemaWarning] = []

        # 'patient_name_dob' contains both 'name' (full_name) and 'dob' (dob)
        result = _match_acroform_field_to_key("patient_name_dob", warnings)

        assert result is None
        assert len(warnings) == 1
        assert warnings[0].kind == "acroform_ambiguous_field"
        assert "patient_name_dob" in warnings[0].message
        assert "full_name" in warnings[0].details["matched_keys"]
        assert "dob" in warnings[0].details["matched_keys"]

    def test_multiple_ambiguous_patterns(self):
        """Various ambiguous field name patterns are detected."""
        # name + dob
        warnings: list[SchemaWarning] = []
        assert _match_acroform_field_to_key("name_birthdate", warnings) is None
        assert len(warnings) == 1

        # name + address (both contain common substrings)
        warnings = []
        assert _match_acroform_field_to_key("patient_name_street_address", warnings) is None
        assert len(warnings) == 1

        # phone + address
        warnings = []
        assert _match_acroform_field_to_key("phone_address_field", warnings) is None
        assert len(warnings) == 1


# ---------------------------------------------------------------------------
# Test: Fallback ordering + max_fields cap
# ---------------------------------------------------------------------------


class TestFallbackOrdering:
    """Tests for fallback V1 schema ordering and max_fields cap."""

    def test_fallback_has_correct_ordering(
        self, tmp_run: RunPaths, trace: TraceLogger
    ):
        """Fallback schema has fields in FIELD_ORDER."""
        options = RunOptions(max_fields=7)
        result = resolve_schema(tmp_run, None, options, trace)

        assert result.schema_source == "fallback_v1"
        field_keys = [f.key for f in result.resolved_fields]
        assert field_keys == FIELD_ORDER

    def test_fallback_respects_max_fields_cap(
        self, tmp_run: RunPaths, trace: TraceLogger
    ):
        """Fallback schema is capped to max_fields."""
        options = RunOptions(max_fields=3)
        result = resolve_schema(tmp_run, None, options, trace)

        assert result.schema_source == "fallback_v1"
        assert len(result.resolved_fields) == 3
        field_keys = [f.key for f in result.resolved_fields]
        # First 3 in FIELD_ORDER
        assert field_keys == ["full_name", "dob", "phone"]

    def test_fallback_max_fields_1(
        self, tmp_run: RunPaths, trace: TraceLogger
    ):
        """Fallback with max_fields=1 returns only first field."""
        options = RunOptions(max_fields=1)
        result = resolve_schema(tmp_run, None, options, trace)

        assert len(result.resolved_fields) == 1
        assert result.resolved_fields[0].key == "full_name"

    def test_fallback_field_types_are_correct(
        self, tmp_run: RunPaths, trace: TraceLogger
    ):
        """Fallback fields have correct types from FIELD_TYPES."""
        options = RunOptions()
        result = resolve_schema(tmp_run, None, options, trace)

        for field in result.resolved_fields:
            assert field.type == FIELD_TYPES[field.key]


# ---------------------------------------------------------------------------
# Test: Artifact writing
# ---------------------------------------------------------------------------


class TestArtifactWriting:
    """Tests for schema.json artifact writing."""

    def test_schema_json_artifact_written(
        self, tmp_run: RunPaths, trace: TraceLogger
    ):
        """resolve_schema writes schema.json artifact."""
        options = RunOptions()
        resolve_schema(tmp_run, None, options, trace)

        schema_path = tmp_run.artifact_path("schema.json")
        assert schema_path.exists()

    def test_schema_json_validates_as_resolved_schema(
        self, tmp_run: RunPaths, trace: TraceLogger
    ):
        """Written schema.json validates as ResolvedSchema model."""
        options = RunOptions()
        resolve_schema(tmp_run, None, options, trace)

        schema_path = tmp_run.artifact_path("schema.json")
        data = read_json(schema_path)

        # Should not raise
        schema = ResolvedSchema(**data)
        assert schema.schema_source in ["user_schema", "fillable_pdf", "fallback_v1"]

    def test_user_schema_artifact_contains_correct_data(
        self, tmp_run: RunPaths, trace: TraceLogger
    ):
        """User schema artifact contains the expected structure."""
        user_schema = make_user_schema_bytes([
            {"key": "allergies", "label": "Allergies", "type": "string_or_list"},
            {"key": "medications", "label": "Meds", "type": "string_or_list"},
        ])

        options = RunOptions()
        resolve_schema(tmp_run, user_schema, options, trace)

        schema_path = tmp_run.artifact_path("schema.json")
        data = read_json(schema_path)

        assert data["schema_source"] == "user_schema"
        assert len(data["resolved_fields"]) == 2


# ---------------------------------------------------------------------------
# Test: Order and cap helper
# ---------------------------------------------------------------------------


class TestOrderAndCapFields:
    """Tests for the _order_and_cap_fields helper."""

    def test_orders_by_field_order(self):
        """Fields are reordered according to FIELD_ORDER."""
        # Create fields in wrong order
        fields = [
            FieldSpec(key="medications", label=None, type="string_or_list"),
            FieldSpec(key="full_name", label=None, type="string"),
            FieldSpec(key="dob", label=None, type="date"),
        ]

        result = _order_and_cap_fields(fields, max_fields=10)
        keys = [f.key for f in result]

        assert keys == ["full_name", "dob", "medications"]

    def test_caps_to_max_fields(self):
        """Result is capped to max_fields."""
        fields = [
            FieldSpec(key="full_name", label=None, type="string"),
            FieldSpec(key="dob", label=None, type="date"),
            FieldSpec(key="phone", label=None, type="phone"),
            FieldSpec(key="address", label=None, type="string"),
        ]

        result = _order_and_cap_fields(fields, max_fields=2)
        assert len(result) == 2

    def test_deduplicates_by_key(self):
        """Duplicate keys result in single entry (uses dict)."""
        fields = [
            FieldSpec(key="full_name", label="Name 1", type="string"),
            FieldSpec(key="full_name", label="Name 2", type="string"),
        ]

        result = _order_and_cap_fields(fields, max_fields=10)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Test: parse_user_schema directly
# ---------------------------------------------------------------------------


class TestParseUserSchema:
    """Direct tests for parse_user_schema function."""

    def test_valid_schema_returns_resolved(self):
        """Valid user schema bytes return ResolvedSchema."""
        schema_bytes = make_user_schema_bytes([
            {"key": "full_name", "label": "Name", "type": "string"},
        ])
        warnings: list[SchemaWarning] = []
        options = RunOptions()

        result = parse_user_schema(schema_bytes, options, warnings)

        assert result is not None
        assert result.schema_source == "user_schema"
        assert len(result.resolved_fields) == 1

    def test_invalid_json_returns_none_with_warning(self):
        """Invalid JSON returns None and adds warning."""
        schema_bytes = b"not json"
        warnings: list[SchemaWarning] = []
        options = RunOptions()

        result = parse_user_schema(schema_bytes, options, warnings)

        assert result is None
        assert len(warnings) == 1
        assert warnings[0].kind == "user_schema_invalid"

    def test_invalid_utf8_returns_none(self):
        """Invalid UTF-8 bytes return None."""
        schema_bytes = b"\xff\xfe"
        warnings: list[SchemaWarning] = []
        options = RunOptions()

        result = parse_user_schema(schema_bytes, options, warnings)

        assert result is None
        assert len(warnings) == 1

    def test_empty_fields_returns_empty_resolved(self):
        """Empty fields array returns ResolvedSchema with no fields."""
        schema_bytes = make_user_schema_bytes([])
        warnings: list[SchemaWarning] = []
        options = RunOptions()

        result = parse_user_schema(schema_bytes, options, warnings)

        assert result is not None
        assert len(result.resolved_fields) == 0

    def test_fields_without_key_skipped(self):
        """Field entries without 'key' are skipped."""
        schema_bytes = json.dumps({
            "fields": [
                {"label": "No Key"},
                {"key": "full_name", "label": "Name", "type": "string"},
            ]
        }).encode()
        warnings: list[SchemaWarning] = []
        options = RunOptions()

        result = parse_user_schema(schema_bytes, options, warnings)

        assert result is not None
        assert len(result.resolved_fields) == 1


# ---------------------------------------------------------------------------
# Test: Non-fillable PDF handling (using real fixtures)
# ---------------------------------------------------------------------------


class TestNonFillablePdf:
    """Tests for handling PDFs without AcroForm fields."""

    def test_form_scanned_is_not_fillable(self):
        """form_scanned.pdf should not have AcroForm fields."""
        pdf_path = FIXTURES_DIR / "form_scanned.pdf"
        if not pdf_path.exists():
            pytest.skip("form_scanned.pdf fixture not found")

        reader = PdfReader(pdf_path)
        fields = reader.get_fields()

        assert fields is None or len(fields) == 0

    def test_non_fillable_pdf_returns_none(self, tmp_run: RunPaths):
        """PDF without AcroForm fields returns None from resolve_from_acroform."""
        pdf_path = copy_fixture("form_scanned.pdf", tmp_run.target_docs_dir())

        warnings: list[SchemaWarning] = []
        options = RunOptions()
        result = resolve_from_acroform(pdf_path, options, warnings)

        assert result is None

    def test_non_fillable_pdf_falls_through_to_fallback(
        self, tmp_run: RunPaths, trace: TraceLogger
    ):
        """Non-fillable PDF in target_docs falls through to fallback."""
        copy_fixture("form_scanned.pdf", tmp_run.target_docs_dir())

        options = RunOptions()
        result = resolve_schema(tmp_run, None, options, trace)

        assert result.schema_source == "fallback_v1"


# ---------------------------------------------------------------------------
# Test: Multiple PDFs - first fillable wins (using real fixtures)
# ---------------------------------------------------------------------------


class TestMultiplePdfs:
    """Tests for handling multiple PDFs in target_docs."""

    def test_fillable_pdf_beats_scanned_pdf(
        self, tmp_run: RunPaths, trace: TraceLogger
    ):
        """First fillable PDF is used even if non-fillable is alphabetically first."""
        fillable_path = FIXTURES_DIR / "form_fillable.pdf"
        scanned_path = FIXTURES_DIR / "form_scanned.pdf"
        if not fillable_path.exists() or not scanned_path.exists():
            pytest.skip("PDF fixtures not found")

        # Copy both PDFs - scanned first alphabetically
        target_dir = tmp_run.target_docs_dir()
        shutil.copy(scanned_path, target_dir / "a_scanned.pdf")  # First alphabetically
        shutil.copy(fillable_path, target_dir / "b_fillable.pdf")  # Second

        options = RunOptions()
        result = resolve_schema(tmp_run, None, options, trace)

        # Should find the fillable one even though scanned is first
        assert result.schema_source == "fillable_pdf"
        field_keys = {f.key for f in result.resolved_fields}
        assert "full_name" in field_keys


# ---------------------------------------------------------------------------
# Test: Trace step logging
# ---------------------------------------------------------------------------


class TestTraceLogging:
    """Tests for trace step logging during schema resolution."""

    def test_resolve_schema_step_logged(
        self, tmp_run: RunPaths, trace: TraceLogger
    ):
        """resolve_schema logs a trace step."""
        options = RunOptions()
        resolve_schema(tmp_run, None, options, trace)

        trace_path = tmp_run.trace_jsonl_path()
        assert trace_path.exists()

        content = trace_path.read_text()
        lines = [json.loads(line) for line in content.strip().split("\n") if line]

        # Find the resolve_schema step
        step_events = [e for e in lines if e.get("step") == "resolve_schema"]
        assert len(step_events) >= 1

    def test_trace_includes_outputs_ref(
        self, tmp_run: RunPaths, trace: TraceLogger
    ):
        """Trace step includes outputs_ref to schema.json."""
        options = RunOptions()
        resolve_schema(tmp_run, None, options, trace)

        trace_path = tmp_run.trace_jsonl_path()
        content = trace_path.read_text()
        lines = [json.loads(line) for line in content.strip().split("\n") if line]

        step_event = next(
            (e for e in lines if e.get("step") == "resolve_schema"), None
        )
        assert step_event is not None
        assert "schema.json" in str(step_event.get("outputs_ref", []))


# ---------------------------------------------------------------------------
# Test: Real PDF fixtures (additional coverage)
# ---------------------------------------------------------------------------


class TestRealPdfFixtures:
    """Additional tests using real PDF fixtures."""

    def test_form_fillable_pdf_field_extraction(self):
        """form_fillable.pdf should have expected AcroForm fields."""
        pdf_path = FIXTURES_DIR / "form_fillable.pdf"
        if not pdf_path.exists():
            pytest.skip("form_fillable.pdf fixture not found")

        reader = PdfReader(pdf_path)
        fields = reader.get_fields()

        assert fields is not None
        # Real PDF has hierarchical field names like "APS1.First Name"
        field_names = list(fields.keys())
        assert len(field_names) > 0
        # Should have name, address, date_of_birth, phone fields
        assert any("Name" in f for f in field_names)
        assert any("Address" in f for f in field_names)
        assert any("Date_of_birth" in f for f in field_names)
        assert any("phone" in f.lower() for f in field_names)

    def test_lab_result_is_not_fillable(self):
        """lab_result.pdf should not have AcroForm fields."""
        pdf_path = FIXTURES_DIR / "lab_result.pdf"
        if not pdf_path.exists():
            pytest.skip("lab_result.pdf fixture not found")

        reader = PdfReader(pdf_path)
        fields = reader.get_fields()

        assert fields is None or len(fields) == 0

    def test_extract_acroform_fields_returns_field_names(self):
        """_extract_acroform_fields returns list of field names from real PDF."""
        pdf_path = FIXTURES_DIR / "form_fillable.pdf"
        if not pdf_path.exists():
            pytest.skip("form_fillable.pdf fixture not found")

        fields = _extract_acroform_fields(pdf_path)

        assert fields is not None
        assert len(fields) > 0
        assert any("Name" in f for f in fields)
