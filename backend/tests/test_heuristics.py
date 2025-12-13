"""
Unit tests for heuristics module.

Tests per-field extractors, normalization, and evidence anchoring.
"""

import pytest

from app.heuristics import (
    heuristic_candidates_for_field,
    normalize_date,
    normalize_phone,
    normalize_text,
)
from app.models import (
    FieldSpec,
    LayoutDoc,
    LayoutPageText,
)


# --- Helper factories ---


def make_field_spec(key: str, label: str | None = None) -> FieldSpec:
    """Create a minimal FieldSpec."""
    type_map = {
        "full_name": "string",
        "dob": "date",
        "phone": "phone",
        "address": "string",
        "insurance_member_id": "string",
        "allergies": "string_or_list",
        "medications": "string_or_list",
    }
    return FieldSpec(key=key, label=label, type=type_map[key])


def make_layout_doc(doc_id: str, pages_text: list[str]) -> LayoutDoc:
    """Create a minimal LayoutDoc with pages."""
    pages = [
        LayoutPageText(page=i + 1, full_text=text)
        for i, text in enumerate(pages_text)
    ]
    return LayoutDoc(doc_id=doc_id, pages=pages)


# --- Test normalization helpers ---


class TestNormalizeText:
    """Tests for normalize_text function."""

    def test_lowercase(self) -> None:
        assert normalize_text("HELLO WORLD") == "hello world"

    def test_collapse_whitespace(self) -> None:
        assert normalize_text("hello   world") == "hello world"
        assert normalize_text("hello\n\nworld") == "hello world"
        assert normalize_text("hello\t\tworld") == "hello world"

    def test_strip_punctuation_except_hyphens(self) -> None:
        assert normalize_text("hello, world!") == "hello world"
        assert normalize_text("test-value") == "test-value"
        assert normalize_text("hello...world") == "helloworld"

    def test_strip_leading_trailing(self) -> None:
        assert normalize_text("  hello  ") == "hello"

    def test_empty_string(self) -> None:
        assert normalize_text("") == ""


class TestNormalizeDate:
    """Tests for normalize_date function."""

    def test_yyyy_mm_dd(self) -> None:
        assert normalize_date("2024-01-15") == "2024-01-15"
        assert normalize_date("2024/01/15") == "2024-01-15"

    def test_mm_dd_yyyy(self) -> None:
        assert normalize_date("01/15/2024") == "2024-01-15"
        assert normalize_date("1/5/2024") == "2024-01-05"

    def test_month_name_formats(self) -> None:
        assert normalize_date("January 15, 2024") == "2024-01-15"
        assert normalize_date("Jan 15 2024") == "2024-01-15"
        assert normalize_date("15 January 2024") == "2024-01-15"

    def test_invalid_date(self) -> None:
        assert normalize_date("not a date") is None
        assert normalize_date("hello world") is None

    def test_padding(self) -> None:
        """Test that single-digit months/days are zero-padded."""
        assert normalize_date("1/5/2024") == "2024-01-05"


class TestNormalizePhone:
    """Tests for normalize_phone function."""

    def test_ten_digits_adds_country_code(self) -> None:
        digits, assumed = normalize_phone("5551234567")
        assert digits == "15551234567"
        assert assumed is True

    def test_formatted_phone(self) -> None:
        digits, assumed = normalize_phone("(555) 123-4567")
        assert digits == "15551234567"
        assert assumed is True

    def test_already_has_country_code(self) -> None:
        digits, assumed = normalize_phone("15551234567")
        assert digits == "15551234567"
        assert assumed is False

    def test_with_plus_one(self) -> None:
        digits, assumed = normalize_phone("+1 555-123-4567")
        assert digits == "15551234567"
        assert assumed is False

    def test_strips_separators(self) -> None:
        digits, _ = normalize_phone("555.123.4567")
        assert digits == "15551234567"


# --- Test DOB extraction ---


class TestDOBExtraction:
    """Tests for DOB candidate extraction."""

    def test_extracts_dob_with_keyword_anchor(self) -> None:
        """DOB extraction from 'DOB: 1978-10-05' includes evidence."""
        field = make_field_spec("dob")
        doc = make_layout_doc("doc_001", ["Patient DOB: 1978-10-05"])

        candidates = heuristic_candidates_for_field(field, [doc])

        assert len(candidates) >= 1

        # Find the candidate with the expected date
        matching = [c for c in candidates if c.normalized_value == "1978-10-05"]
        assert len(matching) == 1

        candidate = matching[0]
        assert candidate.field == "dob"
        assert candidate.normalized_value == "1978-10-05"
        assert candidate.from_method == "heuristic"

        # Check evidence
        assert len(candidate.evidence) == 1
        ev = candidate.evidence[0]
        assert ev.doc_id == "doc_001"
        assert ev.page == 1
        assert "DOB: 1978-10-05" in ev.quoted_text

        # Check anchor match (should be 1.0 since "dob" keyword is near)
        assert candidate.scores.anchor_match == 1.0

    def test_extracts_dob_mm_dd_yyyy_format(self) -> None:
        """Test extraction of MM/DD/YYYY format."""
        field = make_field_spec("dob")
        doc = make_layout_doc("doc_001", ["Date of Birth: 10/05/1978"])

        candidates = heuristic_candidates_for_field(field, [doc])

        matching = [c for c in candidates if c.normalized_value == "1978-10-05"]
        assert len(matching) == 1
        assert matching[0].scores.anchor_match == 1.0

    def test_dob_without_anchor_has_lower_score(self) -> None:
        """DOB found without nearby keyword gets anchor_match=0."""
        field = make_field_spec("dob")
        doc = make_layout_doc("doc_001", ["Some text 2024-01-15 more text"])

        candidates = heuristic_candidates_for_field(field, [doc])

        # Should still find the date
        assert len(candidates) >= 1
        candidate = candidates[0]
        assert candidate.scores.anchor_match == 0.0

    def test_evidence_page_number_correct(self) -> None:
        """Evidence includes correct page number (1-indexed)."""
        field = make_field_spec("dob")
        doc = make_layout_doc("doc_001", [
            "Page 1 content",
            "DOB: 1990-05-15",
        ])

        candidates = heuristic_candidates_for_field(field, [doc])

        matching = [c for c in candidates if c.normalized_value == "1990-05-15"]
        assert len(matching) == 1
        assert matching[0].evidence[0].page == 2


# --- Test phone extraction ---


class TestPhoneExtraction:
    """Tests for phone candidate extraction."""

    def test_extracts_phone_with_normalization(self) -> None:
        """Phone extraction from '(555) 123-4567' normalizes digits."""
        field = make_field_spec("phone")
        doc = make_layout_doc("doc_001", ["Phone: (555) 123-4567"])

        candidates = heuristic_candidates_for_field(field, [doc])

        assert len(candidates) >= 1
        candidate = candidates[0]

        # Should be normalized to 11 digits with country code
        assert candidate.normalized_value == "15551234567"
        assert candidate.from_method == "heuristic"

        # Should have evidence
        assert len(candidate.evidence) == 1
        assert "Phone:" in candidate.evidence[0].quoted_text

    def test_phone_default_country_flag(self) -> None:
        """If +1 applied, includes validator flag 'default_country_assumed'."""
        field = make_field_spec("phone")
        doc = make_layout_doc("doc_001", ["Contact: 555-123-4567"])

        candidates = heuristic_candidates_for_field(field, [doc])

        # Find candidate (may have multiple patterns matching)
        assert len(candidates) >= 1
        candidate = candidates[0]

        assert "default_country_assumed" in candidate.validators

    def test_phone_with_country_code_no_flag(self) -> None:
        """Phone with explicit 11-digit number doesn't get flag."""
        field = make_field_spec("phone")
        # Use 11 consecutive digits starting with 1
        doc = make_layout_doc("doc_001", ["Tel: 15551234567"])

        candidates = heuristic_candidates_for_field(field, [doc])

        assert len(candidates) >= 1
        # At least one candidate should not have the flag (already has country code)
        has_without_flag = any(
            "default_country_assumed" not in c.validators
            for c in candidates
        )
        assert has_without_flag

    def test_phone_anchor_match(self) -> None:
        """Phone near keyword gets anchor_match=1."""
        field = make_field_spec("phone")
        doc = make_layout_doc("doc_001", ["Mobile: 555-123-4567"])

        candidates = heuristic_candidates_for_field(field, [doc])

        assert len(candidates) >= 1
        assert candidates[0].scores.anchor_match == 1.0


# --- Test insurance_member_id extraction ---


class TestInsuranceIDExtraction:
    """Tests for insurance member ID extraction."""

    def test_extracts_id_near_keywords(self) -> None:
        """Insurance ID extraction near keywords extracts candidate and evidence."""
        field = make_field_spec("insurance_member_id")
        doc = make_layout_doc("doc_001", ["Member ID: ABC123456789"])

        candidates = heuristic_candidates_for_field(field, [doc])

        assert len(candidates) >= 1
        candidate = candidates[0]

        assert candidate.field == "insurance_member_id"
        assert "abc123456789" in candidate.normalized_value.lower()
        assert candidate.from_method == "heuristic"
        assert candidate.scores.anchor_match == 1.0

        # Evidence anchors
        assert len(candidate.evidence) == 1
        assert "Member ID" in candidate.evidence[0].quoted_text

    def test_extracts_policy_number(self) -> None:
        """Also extracts IDs near 'policy' keyword."""
        field = make_field_spec("insurance_member_id")
        doc = make_layout_doc("doc_001", ["Policy: XYZ987654"])

        candidates = heuristic_candidates_for_field(field, [doc])

        assert len(candidates) >= 1
        # Should find the policy number
        matching = [c for c in candidates if "xyz987654" in c.normalized_value.lower()]
        assert len(matching) >= 1

    def test_skips_common_words(self) -> None:
        """Should not extract common words like 'member' as IDs."""
        field = make_field_spec("insurance_member_id")
        doc = make_layout_doc("doc_001", ["Insurance member information"])

        candidates = heuristic_candidates_for_field(field, [doc])

        # Should not have 'member' as a candidate
        for c in candidates:
            assert c.normalized_value.lower() != "member"


# --- Test name extraction ---


class TestNameExtraction:
    """Tests for full_name extraction."""

    def test_extracts_name_from_label(self) -> None:
        """Extracts name from 'Name: John Doe' pattern."""
        field = make_field_spec("full_name")
        doc = make_layout_doc("doc_001", ["Patient Name: John Smith"])

        candidates = heuristic_candidates_for_field(field, [doc])

        assert len(candidates) >= 1
        candidate = candidates[0]

        assert "john smith" in candidate.normalized_value
        assert candidate.scores.anchor_match == 1.0

    def test_extracts_patient_name(self) -> None:
        """Extracts name from 'Patient: ...' pattern."""
        field = make_field_spec("full_name")
        doc = make_layout_doc("doc_001", ["Patient: Jane Doe"])

        candidates = heuristic_candidates_for_field(field, [doc])

        assert len(candidates) >= 1
        assert "jane doe" in candidates[0].normalized_value


# --- Test address extraction ---


class TestAddressExtraction:
    """Tests for address extraction."""

    def test_extracts_address(self) -> None:
        """Extracts address from 'Address: ...' pattern."""
        field = make_field_spec("address")
        doc = make_layout_doc("doc_001", ["Address: 123 Main St, City, ST 12345"])

        candidates = heuristic_candidates_for_field(field, [doc])

        assert len(candidates) >= 1
        assert "123 main st" in candidates[0].normalized_value


# --- Test list field extraction ---


class TestAllergiesExtraction:
    """Tests for allergies extraction."""

    def test_extracts_allergies(self) -> None:
        """Extracts allergies list."""
        field = make_field_spec("allergies")
        doc = make_layout_doc("doc_001", ["Allergies: Penicillin, Sulfa, Peanuts"])

        candidates = heuristic_candidates_for_field(field, [doc])

        assert len(candidates) >= 1
        assert "penicillin" in candidates[0].normalized_value


class TestMedicationsExtraction:
    """Tests for medications extraction."""

    def test_extracts_medications(self) -> None:
        """Extracts medications list."""
        field = make_field_spec("medications")
        doc = make_layout_doc("doc_001", ["Medications: Aspirin 81mg, Lisinopril 10mg"])

        candidates = heuristic_candidates_for_field(field, [doc])

        assert len(candidates) >= 1
        assert "aspirin" in candidates[0].normalized_value


# --- Test multiple documents ---


class TestMultipleDocuments:
    """Tests for extraction across multiple documents."""

    def test_extracts_from_multiple_docs(self) -> None:
        """Candidates come from multiple routed documents."""
        field = make_field_spec("dob")
        docs = [
            make_layout_doc("doc_001", ["DOB: 1980-01-15"]),
            make_layout_doc("doc_002", ["Date of Birth: 02/20/1990"]),
        ]

        candidates = heuristic_candidates_for_field(field, docs)

        # Should have candidates from both docs
        doc_ids = {c.evidence[0].doc_id for c in candidates}
        assert "doc_001" in doc_ids
        assert "doc_002" in doc_ids

    def test_deduplicates_same_value(self) -> None:
        """Same normalized value from same doc is deduplicated."""
        field = make_field_spec("dob")
        doc = make_layout_doc("doc_001", ["DOB: 1980-01-15\nDOB: 1980-01-15"])

        candidates = heuristic_candidates_for_field(field, [doc])

        # Should only have one candidate for this date
        values = [c.normalized_value for c in candidates]
        assert values.count("1980-01-15") == 1

