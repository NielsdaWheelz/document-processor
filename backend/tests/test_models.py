"""
Unit tests for pydantic models.

Tests model instantiation, validation rules, and invariant enforcement.
"""

import pytest
from pydantic import ValidationError

from app.models import (
    SUPPORTED_FIELD_KEYS,
    Candidate,
    CandidateScores,
    DocIndexItem,
    Evidence,
    FieldSpec,
    FinalField,
    FinalResult,
    LayoutDoc,
    LayoutPageText,
    LayoutSpan,
    ResolvedSchema,
    RoutingEntry,
    RunOptions,
)


class TestRunOptions:
    """Tests for RunOptions model."""

    def test_defaults(self):
        """Test that defaults are applied correctly."""
        opts = RunOptions()
        assert opts.top_k_docs == 3
        assert opts.llm_provider == "anthropic"
        assert opts.llm_model == "claude-sonnet-4-20250514"
        assert opts.max_llm_tokens == 1200
        assert opts.max_fields == 7

    def test_custom_values(self):
        """Test custom values are accepted."""
        opts = RunOptions(
            top_k_docs=5,
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            max_llm_tokens=2000,
            max_fields=10,
        )
        assert opts.top_k_docs == 5
        assert opts.llm_provider == "openai"
        assert opts.llm_model == "gpt-4o-mini"

    def test_invalid_provider(self):
        """Test that invalid provider is rejected."""
        with pytest.raises(ValidationError):
            RunOptions(llm_provider="invalid")


class TestFieldSpec:
    """Tests for FieldSpec model."""

    def test_valid_field_spec(self):
        """Test valid field spec creation."""
        spec = FieldSpec(key="full_name", label="Patient Name", type="string")
        assert spec.key == "full_name"
        assert spec.label == "Patient Name"
        assert spec.type == "string"

    def test_all_supported_keys(self):
        """Test all supported keys are valid."""
        types = {
            "full_name": "string",
            "dob": "date",
            "phone": "phone",
            "address": "string",
            "insurance_member_id": "string",
            "allergies": "string_or_list",
            "medications": "string_or_list",
        }
        for key, field_type in types.items():
            spec = FieldSpec(key=key, type=field_type)
            assert spec.key == key

    def test_unsupported_key_rejected(self):
        """Test that unsupported field keys are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            FieldSpec(key="unknown_field", type="string")
        assert "Unsupported field key" in str(exc_info.value)

    def test_invalid_type_rejected(self):
        """Test that invalid field types are rejected."""
        with pytest.raises(ValidationError):
            FieldSpec(key="full_name", type="invalid_type")


class TestResolvedSchema:
    """Tests for ResolvedSchema model."""

    def test_valid_schema(self):
        """Test valid schema creation."""
        schema = ResolvedSchema(
            schema_source="user_schema",
            resolved_fields=[
                FieldSpec(key="full_name", type="string"),
                FieldSpec(key="dob", type="date"),
            ],
        )
        assert schema.schema_source == "user_schema"
        assert len(schema.resolved_fields) == 2
        assert schema.unsupported_fields == []

    def test_schema_source_literal_validation(self):
        """Test that schema_source only allows valid literals."""
        # Valid sources
        for source in ["user_schema", "fillable_pdf", "fallback_v1"]:
            schema = ResolvedSchema(schema_source=source, resolved_fields=[])
            assert schema.schema_source == source

        # Invalid source
        with pytest.raises(ValidationError):
            ResolvedSchema(schema_source="invalid_source", resolved_fields=[])

    def test_unsupported_fields_tracking(self):
        """Test unsupported fields are tracked."""
        schema = ResolvedSchema(
            schema_source="user_schema",
            resolved_fields=[FieldSpec(key="full_name", type="string")],
            unsupported_fields=["ssn", "email"],
        )
        assert schema.unsupported_fields == ["ssn", "email"]


class TestDocIndexItem:
    """Tests for DocIndexItem model."""

    def test_valid_doc_index_item(self):
        """Test valid DocIndexItem creation."""
        item = DocIndexItem(
            doc_id="doc_001",
            filename="medical_records.pdf",
            mime_type="application/pdf",
            pages=5,
            has_text_layer=True,
            sha256="abc123def456",
        )
        assert item.doc_id == "doc_001"
        assert item.unreadable_reason is None

    def test_unreadable_document(self):
        """Test unreadable document with reason."""
        item = DocIndexItem(
            doc_id="doc_002",
            filename="scanned.pdf",
            mime_type="application/pdf",
            pages=3,
            has_text_layer=False,
            unreadable_reason="no_text_layer",
            sha256="xyz789",
        )
        assert item.has_text_layer is False
        assert item.unreadable_reason == "no_text_layer"

    def test_invalid_unreadable_reason(self):
        """Test invalid unreadable reason is rejected."""
        with pytest.raises(ValidationError):
            DocIndexItem(
                doc_id="doc_001",
                filename="test.pdf",
                mime_type="application/pdf",
                has_text_layer=False,
                unreadable_reason="invalid_reason",
                sha256="abc",
            )


class TestLayoutModels:
    """Tests for layout-related models."""

    def test_layout_span(self):
        """Test LayoutSpan creation."""
        span = LayoutSpan(text="Patient Name: John Doe")
        assert span.text == "Patient Name: John Doe"
        assert span.bbox is None

        span_with_bbox = LayoutSpan(text="test", bbox=[0.0, 0.0, 100.0, 20.0])
        assert span_with_bbox.bbox == [0.0, 0.0, 100.0, 20.0]

    def test_layout_page_text_valid(self):
        """Test valid LayoutPageText creation."""
        page = LayoutPageText(page=1, full_text="Page content here")
        assert page.page == 1
        assert page.spans == []

    def test_layout_page_text_page_zero_rejected(self):
        """Test that page=0 is rejected (pages are 1-indexed)."""
        with pytest.raises(ValidationError) as exc_info:
            LayoutPageText(page=0, full_text="content")
        assert "Page must be >= 1" in str(exc_info.value)

    def test_layout_page_text_negative_page_rejected(self):
        """Test that negative page numbers are rejected."""
        with pytest.raises(ValidationError):
            LayoutPageText(page=-1, full_text="content")

    def test_layout_doc(self):
        """Test LayoutDoc creation."""
        doc = LayoutDoc(
            doc_id="doc_001",
            pages=[
                LayoutPageText(page=1, full_text="Page 1 content"),
                LayoutPageText(page=2, full_text="Page 2 content"),
            ],
        )
        assert doc.doc_id == "doc_001"
        assert len(doc.pages) == 2


class TestRoutingEntry:
    """Tests for RoutingEntry model."""

    def test_valid_routing_entry(self):
        """Test valid RoutingEntry creation."""
        entry = RoutingEntry(
            field="full_name",
            doc_ids=["doc_001", "doc_002"],
            scores={"doc_001": 0.9, "doc_002": 0.7},
        )
        assert entry.field == "full_name"
        assert len(entry.doc_ids) == 2

    def test_unsupported_field_rejected(self):
        """Test that unsupported field is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RoutingEntry(field="unknown", doc_ids=[], scores={})
        assert "Unsupported field key" in str(exc_info.value)


class TestEvidence:
    """Tests for Evidence model."""

    def test_valid_evidence(self):
        """Test valid Evidence creation."""
        evidence = Evidence(
            doc_id="doc_001",
            page=1,
            quoted_text="Patient Name: John Doe",
        )
        assert evidence.doc_id == "doc_001"
        assert evidence.page == 1
        assert evidence.bbox is None

    def test_evidence_with_bbox(self):
        """Test Evidence with bounding box."""
        evidence = Evidence(
            doc_id="doc_001",
            page=1,
            quoted_text="DOB: 1990-01-15",
            bbox=[10.0, 20.0, 100.0, 40.0],
        )
        assert evidence.bbox == [10.0, 20.0, 100.0, 40.0]

    def test_empty_doc_id_rejected(self):
        """Test that empty doc_id is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Evidence(doc_id="", page=1, quoted_text="text")
        assert "doc_id must not be empty" in str(exc_info.value)

    def test_whitespace_doc_id_rejected(self):
        """Test that whitespace-only doc_id is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Evidence(doc_id="   ", page=1, quoted_text="text")
        assert "doc_id must not be empty" in str(exc_info.value)

    def test_page_zero_rejected(self):
        """Test that page=0 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Evidence(doc_id="doc_001", page=0, quoted_text="text")
        assert "Page must be >= 1" in str(exc_info.value)

    def test_negative_page_rejected(self):
        """Test that negative page is rejected."""
        with pytest.raises(ValidationError):
            Evidence(doc_id="doc_001", page=-1, quoted_text="text")

    def test_empty_quoted_text_rejected(self):
        """Test that empty quoted_text is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Evidence(doc_id="doc_001", page=1, quoted_text="")
        assert "quoted_text must not be empty" in str(exc_info.value)

    def test_whitespace_quoted_text_rejected(self):
        """Test that whitespace-only quoted_text is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Evidence(doc_id="doc_001", page=1, quoted_text="   \n\t  ")
        assert "quoted_text must not be empty" in str(exc_info.value)


class TestCandidateScores:
    """Tests for CandidateScores model."""

    def test_valid_scores(self):
        """Test valid score creation."""
        scores = CandidateScores(
            anchor_match=1.0,
            validator=0.8,
            doc_relevance=0.5,
        )
        assert scores.anchor_match == 1.0
        assert scores.cross_doc_agreement == 0.0
        assert scores.contradiction_penalty == 0.0

    def test_all_scores_at_bounds(self):
        """Test scores at boundary values."""
        scores = CandidateScores(
            anchor_match=0.0,
            validator=1.0,
            doc_relevance=0.0,
            cross_doc_agreement=1.0,
            contradiction_penalty=1.0,
        )
        assert scores.anchor_match == 0.0
        assert scores.contradiction_penalty == 1.0

    def test_score_below_zero_rejected(self):
        """Test that scores below 0 are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            CandidateScores(anchor_match=-0.1, validator=0.5, doc_relevance=0.5)
        assert "must be in [0, 1]" in str(exc_info.value)

    def test_score_above_one_rejected(self):
        """Test that scores above 1 are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            CandidateScores(anchor_match=0.5, validator=1.1, doc_relevance=0.5)
        assert "must be in [0, 1]" in str(exc_info.value)

    def test_all_score_fields_validated(self):
        """Test that all score fields have range validation."""
        score_fields = [
            "anchor_match",
            "validator",
            "doc_relevance",
            "cross_doc_agreement",
            "contradiction_penalty",
        ]
        for field in score_fields:
            with pytest.raises(ValidationError):
                kwargs = {
                    "anchor_match": 0.5,
                    "validator": 0.5,
                    "doc_relevance": 0.5,
                }
                kwargs[field] = 1.5  # Out of range
                CandidateScores(**kwargs)


class TestCandidate:
    """Tests for Candidate model."""

    def _make_evidence(self) -> Evidence:
        return Evidence(doc_id="doc_001", page=1, quoted_text="John Doe")

    def _make_scores(self) -> CandidateScores:
        return CandidateScores(anchor_match=1.0, validator=1.0, doc_relevance=0.8)

    def test_valid_candidate(self):
        """Test valid Candidate creation."""
        candidate = Candidate(
            field="full_name",
            raw_value="John Doe",
            normalized_value="John Doe",
            evidence=[self._make_evidence()],
            from_method="heuristic",
            scores=self._make_scores(),
        )
        assert candidate.field == "full_name"
        assert candidate.is_accepted is True

    def test_candidate_with_rejection(self):
        """Test candidate with rejection reasons."""
        candidate = Candidate(
            field="full_name",
            raw_value="John Doe",
            normalized_value="John Doe",
            evidence=[self._make_evidence()],
            from_method="llm",
            rejected_reasons=["unsupported_by_evidence"],
            scores=self._make_scores(),
        )
        assert candidate.is_accepted is False

    def test_empty_evidence_rejected(self):
        """Test that empty evidence list is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Candidate(
                field="full_name",
                raw_value="John Doe",
                normalized_value="John Doe",
                evidence=[],
                from_method="heuristic",
                scores=self._make_scores(),
            )
        assert "evidence must not be empty" in str(exc_info.value)

    def test_unsupported_field_rejected(self):
        """Test that unsupported field is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Candidate(
                field="ssn",
                raw_value="123-45-6789",
                normalized_value="123456789",
                evidence=[self._make_evidence()],
                from_method="heuristic",
                scores=self._make_scores(),
            )
        assert "Unsupported field key" in str(exc_info.value)

    def test_invalid_from_method_rejected(self):
        """Test that invalid from_method is rejected."""
        with pytest.raises(ValidationError):
            Candidate(
                field="full_name",
                raw_value="John",
                normalized_value="John",
                evidence=[self._make_evidence()],
                from_method="regex",  # Invalid
                scores=self._make_scores(),
            )


class TestFinalField:
    """Tests for FinalField model."""

    def test_filled_field(self):
        """Test filled field creation."""
        field = FinalField(
            field="full_name",
            status="filled",
            value="John Doe",
            normalized_value="John Doe",
            confidence=0.95,
            rationale=["High anchor match", "Validator passed"],
            evidence=[Evidence(doc_id="doc_001", page=1, quoted_text="John Doe")],
        )
        assert field.status == "filled"
        assert field.confidence == 0.95

    def test_missing_field(self):
        """Test missing field creation."""
        field = FinalField(
            field="insurance_member_id",
            status="missing",
            value=None,
            normalized_value=None,
            confidence=0.0,
            rationale=["No candidates found"],
        )
        assert field.status == "missing"
        assert field.value is None

    def test_needs_review_field(self):
        """Test needs_review field creation."""
        field = FinalField(
            field="dob",
            status="needs_review",
            value="1990-01-15",
            normalized_value="1990-01-15",
            confidence=0.65,
            rationale=["Confidence below threshold"],
        )
        assert field.status == "needs_review"

    def test_invalid_status_rejected(self):
        """Test that invalid status is rejected."""
        with pytest.raises(ValidationError):
            FinalField(
                field="full_name",
                status="approved",  # Invalid
                confidence=0.8,
                rationale=[],
            )

    def test_status_enum_validation(self):
        """Test that only valid status literals are accepted."""
        valid_statuses = ["filled", "needs_review", "missing"]
        for status in valid_statuses:
            field = FinalField(field="full_name", status=status, confidence=0.5, rationale=[])
            assert field.status == status

        invalid_statuses = ["done", "pending", "error", "complete"]
        for status in invalid_statuses:
            with pytest.raises(ValidationError):
                FinalField(field="full_name", status=status, confidence=0.5, rationale=[])

    def test_confidence_range_validation(self):
        """Test confidence must be in [0, 1]."""
        with pytest.raises(ValidationError):
            FinalField(field="full_name", status="filled", confidence=1.5, rationale=[])

        with pytest.raises(ValidationError):
            FinalField(field="full_name", status="filled", confidence=-0.1, rationale=[])


class TestFinalResult:
    """Tests for FinalResult model."""

    def test_valid_final_result(self):
        """Test valid FinalResult creation."""
        result = FinalResult(
            run_id="2025-12-12T11-32-01Z_ab12cd",
            schema_source="user_schema",
            fields={
                "full_name": FinalField(
                    field="full_name",
                    status="filled",
                    value="John Doe",
                    normalized_value="John Doe",
                    confidence=0.95,
                    rationale=["Match found"],
                )
            },
        )
        assert result.run_id == "2025-12-12T11-32-01Z_ab12cd"
        assert "full_name" in result.fields

    def test_schema_source_literal_validation(self):
        """Test FinalResult.schema_source only allows valid literals."""
        valid_sources = ["user_schema", "fillable_pdf", "fallback_v1"]
        for source in valid_sources:
            result = FinalResult(run_id="test", schema_source=source, fields={})
            assert result.schema_source == source

        with pytest.raises(ValidationError):
            FinalResult(run_id="test", schema_source="invalid", fields={})

    def test_empty_fields(self):
        """Test FinalResult with no fields (all missing scenario)."""
        result = FinalResult(
            run_id="test_run",
            schema_source="fallback_v1",
            fields={},
        )
        assert result.fields == {}


class TestSupportedFieldKeys:
    """Tests for the SUPPORTED_FIELD_KEYS constant."""

    def test_all_expected_keys_present(self):
        """Test that all expected keys are in the set."""
        expected = {
            "full_name",
            "dob",
            "phone",
            "address",
            "insurance_member_id",
            "allergies",
            "medications",
        }
        assert SUPPORTED_FIELD_KEYS == expected

    def test_key_count(self):
        """Test the number of supported keys."""
        assert len(SUPPORTED_FIELD_KEYS) == 7
