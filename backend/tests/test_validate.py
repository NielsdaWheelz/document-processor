"""
Unit tests for validation helpers.
"""

import pytest

from app.models import (
    CandidateScores,
    DocIndexItem,
    Evidence,
    FieldSpec,
    ResolvedSchema,
)
from app.validate import (
    JSONParseException,
    ValidationException,
    load_json_bytes,
    validate_data,
)


class TestLoadJsonBytes:
    """Tests for load_json_bytes function."""

    def test_valid_json(self):
        """Test parsing valid JSON."""
        data = b'{"key": "value", "number": 42}'
        result = load_json_bytes(data)
        assert result == {"key": "value", "number": 42}

    def test_valid_json_array(self):
        """Test parsing valid JSON array."""
        data = b'[1, 2, 3]'
        result = load_json_bytes(data)
        assert result == [1, 2, 3]

    def test_valid_json_string(self):
        """Test parsing JSON string."""
        data = b'"hello"'
        result = load_json_bytes(data)
        assert result == "hello"

    def test_unicode_content(self):
        """Test parsing JSON with unicode."""
        data = '{"name": "José García"}'.encode("utf-8")
        result = load_json_bytes(data)
        assert result["name"] == "José García"

    def test_invalid_json(self):
        """Test invalid JSON raises exception."""
        data = b'{"key": value}'  # Missing quotes around value
        with pytest.raises(JSONParseException) as exc_info:
            load_json_bytes(data)
        assert "Invalid JSON" in str(exc_info.value)

    def test_invalid_utf8(self):
        """Test invalid UTF-8 raises exception."""
        data = b'\xff\xfe'  # Invalid UTF-8
        with pytest.raises(JSONParseException) as exc_info:
            load_json_bytes(data)
        assert "Invalid UTF-8" in str(exc_info.value)

    def test_empty_bytes(self):
        """Test empty bytes raises exception."""
        with pytest.raises(JSONParseException):
            load_json_bytes(b'')

    def test_truncated_json(self):
        """Test truncated JSON raises exception."""
        data = b'{"key": "val'
        with pytest.raises(JSONParseException):
            load_json_bytes(data)


class TestValidateDataWithModels:
    """Tests for validate_data with pydantic models."""

    def test_valid_model(self):
        """Test validating against a pydantic model."""
        data = {
            "schema_source": "user_schema",
            "resolved_fields": [{"key": "full_name", "type": "string"}],
        }
        result = validate_data(ResolvedSchema, data)
        assert isinstance(result, ResolvedSchema)
        assert result.schema_source == "user_schema"

    def test_invalid_model_data(self):
        """Test invalid data raises ValidationException."""
        data = {
            "schema_source": "invalid_source",  # Invalid literal
            "resolved_fields": [],
        }
        with pytest.raises(ValidationException) as exc_info:
            validate_data(ResolvedSchema, data)
        assert "Validation failed" in str(exc_info.value)
        assert exc_info.value.errors is not None

    def test_missing_required_field(self):
        """Test missing required field raises exception."""
        data = {"schema_source": "user_schema"}  # Missing resolved_fields
        with pytest.raises(ValidationException):
            validate_data(ResolvedSchema, data)

    def test_nested_validation(self):
        """Test validation of nested models."""
        data = {
            "anchor_match": 0.5,
            "validator": 0.8,
            "doc_relevance": 1.5,  # Out of range
        }
        with pytest.raises(ValidationException) as exc_info:
            validate_data(CandidateScores, data)
        assert "must be in [0, 1]" in str(exc_info.value)


class TestValidateDataWithListTypes:
    """Tests for validate_data with list types."""

    def test_valid_list(self):
        """Test validating a list of models."""
        data = [
            {
                "doc_id": "doc_001",
                "filename": "test.pdf",
                "mime_type": "application/pdf",
                "pages": 5,
                "has_text_layer": True,
                "sha256": "abc123",
            },
            {
                "doc_id": "doc_002",
                "filename": "test2.pdf",
                "mime_type": "application/pdf",
                "pages": 3,
                "has_text_layer": False,
                "unreadable_reason": "no_text_layer",
                "sha256": "def456",
            },
        ]
        result = validate_data(list[DocIndexItem], data)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(item, DocIndexItem) for item in result)

    def test_empty_list(self):
        """Test validating an empty list."""
        result = validate_data(list[DocIndexItem], [])
        assert result == []

    def test_invalid_list_item(self):
        """Test that invalid items in list raise exception."""
        data = [
            {
                "doc_id": "doc_001",
                "filename": "test.pdf",
                "mime_type": "application/pdf",
                "has_text_layer": True,
                "sha256": "abc",
            },
            {
                "doc_id": "",  # Empty, should fail
                "filename": "test2.pdf",
                "mime_type": "application/pdf",
                "has_text_layer": True,
                "sha256": "def",
            },
        ]
        # Note: pydantic allows empty strings for doc_id in DocIndexItem
        # The validation would pass since DocIndexItem doesn't validate doc_id emptiness
        # Let's use Evidence instead which does validate
        data = [
            {"doc_id": "doc_001", "page": 1, "quoted_text": "text"},
            {"doc_id": "", "page": 1, "quoted_text": "text"},  # Empty doc_id
        ]
        with pytest.raises(ValidationException):
            validate_data(list[Evidence], data)

    def test_non_list_when_list_expected(self):
        """Test that non-list raises exception when list expected."""
        data = {"doc_id": "doc_001"}  # Dict instead of list
        with pytest.raises(ValidationException):
            validate_data(list[DocIndexItem], data)


class TestValidateDataEdgeCases:
    """Edge case tests for validate_data."""

    def test_extra_fields_ignored(self):
        """Test that extra fields are handled (pydantic default behavior)."""
        data = {
            "key": "full_name",
            "type": "string",
            "extra_field": "ignored",
        }
        result = validate_data(FieldSpec, data)
        assert result.key == "full_name"
        assert not hasattr(result, "extra_field")

    def test_none_value_handling(self):
        """Test handling of None values."""
        data = {
            "doc_id": "doc_001",
            "filename": "test.pdf",
            "mime_type": "application/pdf",
            "pages": None,
            "has_text_layer": True,
            "sha256": "abc",
        }
        result = validate_data(DocIndexItem, data)
        assert result.pages is None

    def test_type_coercion(self):
        """Test pydantic type coercion."""
        data = {
            "anchor_match": "0.5",  # String that can be coerced to float
            "validator": 0.8,
            "doc_relevance": 0.7,
        }
        result = validate_data(CandidateScores, data)
        assert result.anchor_match == 0.5
        assert isinstance(result.anchor_match, float)


class TestValidationExceptionDetails:
    """Tests for ValidationException error details."""

    def test_error_details_available(self):
        """Test that error details are available."""
        data = {
            "schema_source": "invalid",
            "resolved_fields": [],
        }
        with pytest.raises(ValidationException) as exc_info:
            validate_data(ResolvedSchema, data)
        assert exc_info.value.errors is not None
        assert len(exc_info.value.errors) > 0

    def test_multiple_errors(self):
        """Test handling multiple validation errors."""
        data = {
            "anchor_match": -0.5,  # Invalid
            "validator": 1.5,  # Invalid
            "doc_relevance": 0.5,
        }
        with pytest.raises(ValidationException) as exc_info:
            validate_data(CandidateScores, data)
        # Should have at least one error
        assert len(exc_info.value.errors) >= 1
