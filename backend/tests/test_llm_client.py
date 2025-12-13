"""
Unit tests for LLM client module.

Tests retry logic, JSON parsing, and candidate validation without network calls.
"""

import json

import pytest

from app.excerpts import DocExcerpt
from app.llm_client import (
    ApiLLMClient,
    FakeLLMClient,
    LLMInvalidJSONError,
    _parse_llm_response,
)
from app.models import (
    Candidate,
    CandidateScores,
    Evidence,
    FieldSpec,
    RunOptions,
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


def make_excerpt(doc_id: str, page: int, text: str) -> DocExcerpt:
    """Create a DocExcerpt."""
    return DocExcerpt(doc_id=doc_id, page=page, text=text)


def make_valid_json_response(
    raw_value: str,
    normalized_value: str,
    doc_id: str,
    page: int,
    quoted_text: str,
) -> str:
    """Create a valid JSON response string."""
    return json.dumps([{
        "raw_value": raw_value,
        "normalized_value": normalized_value,
        "evidence": [{
            "doc_id": doc_id,
            "page": page,
            "quoted_text": quoted_text,
        }],
    }])


# --- Test _parse_llm_response ---


class TestParseLLMResponse:
    """Tests for _parse_llm_response function."""

    def test_parses_valid_json(self) -> None:
        """Parses valid JSON into Candidate objects."""
        field = make_field_spec("full_name")
        response = make_valid_json_response(
            raw_value="John Doe",
            normalized_value="john doe",
            doc_id="doc_001",
            page=1,
            quoted_text="Patient Name: John Doe",
        )

        candidates = _parse_llm_response(field, response)

        assert len(candidates) == 1
        c = candidates[0]
        assert c.field == "full_name"
        assert c.raw_value == "John Doe"
        assert c.from_method == "llm"
        assert len(c.evidence) == 1

    def test_handles_markdown_code_blocks(self) -> None:
        """Handles responses wrapped in markdown code blocks."""
        field = make_field_spec("dob")
        json_content = make_valid_json_response(
            raw_value="1990-01-15",
            normalized_value="1990-01-15",
            doc_id="doc_001",
            page=1,
            quoted_text="DOB: 1990-01-15",
        )
        response = f"```json\n{json_content}\n```"

        candidates = _parse_llm_response(field, response)

        assert len(candidates) == 1
        assert candidates[0].normalized_value == "1990-01-15"

    def test_empty_array_returns_empty_list(self) -> None:
        """Empty JSON array returns empty candidate list."""
        field = make_field_spec("full_name")

        candidates = _parse_llm_response(field, "[]")

        assert candidates == []

    def test_invalid_json_raises(self) -> None:
        """Invalid JSON raises ValueError."""
        field = make_field_spec("full_name")

        with pytest.raises(ValueError, match="Invalid JSON"):
            _parse_llm_response(field, "not json at all")

    def test_non_array_raises(self) -> None:
        """Non-array JSON raises ValueError."""
        field = make_field_spec("full_name")

        with pytest.raises(ValueError, match="must be a JSON array"):
            _parse_llm_response(field, '{"raw_value": "test"}')

    def test_skips_items_without_evidence(self) -> None:
        """Items without evidence are skipped."""
        field = make_field_spec("full_name")
        response = json.dumps([{
            "raw_value": "John Doe",
            "normalized_value": "john doe",
            "evidence": [],  # Empty evidence
        }])

        candidates = _parse_llm_response(field, response)

        assert candidates == []

    def test_skips_items_with_invalid_evidence(self) -> None:
        """Items with invalid evidence are skipped."""
        field = make_field_spec("full_name")
        response = json.dumps([{
            "raw_value": "John Doe",
            "normalized_value": "john doe",
            "evidence": [{
                "doc_id": "",  # Invalid: empty
                "page": 1,
                "quoted_text": "test",
            }],
        }])

        candidates = _parse_llm_response(field, response)

        assert candidates == []

    def test_phone_adds_validator_flag(self) -> None:
        """Phone candidates get default_country_assumed flag when appropriate."""
        field = make_field_spec("phone")
        response = json.dumps([{
            "raw_value": "555-123-4567",  # 10 digits -> needs +1
            "normalized_value": "5551234567",
            "evidence": [{
                "doc_id": "doc_001",
                "page": 1,
                "quoted_text": "Phone: 555-123-4567",
            }],
        }])

        candidates = _parse_llm_response(field, response)

        assert len(candidates) == 1
        assert "default_country_assumed" in candidates[0].validators


# --- Test ApiLLMClient retry logic ---


class TestApiLLMClientRetry:
    """Tests for ApiLLMClient retry behavior."""

    def test_invalid_json_triggers_exactly_one_retry_then_success(self) -> None:
        """Invalid JSON triggers exactly one retry, then success."""
        client = ApiLLMClient(
            provider="anthropic",
            api_key="fake-key",
            model="test-model",
        )

        field = make_field_spec("full_name")
        excerpts = [make_excerpt("doc_001", 1, "Patient Name: John Doe")]
        run_options = RunOptions()

        # Track call count
        call_count = 0
        responses = [
            "not valid json",  # First call: invalid
            make_valid_json_response(  # Second call: valid
                "John Doe", "john doe", "doc_001", 1, "Patient Name: John Doe"
            ),
        ]

        def mock_call_api(messages: list, max_tokens: int) -> str:
            nonlocal call_count
            result = responses[call_count]
            call_count += 1
            return result

        # Monkeypatch the _call_api method
        client._call_api = mock_call_api  # type: ignore

        candidates = client.extract_candidates(field, excerpts, run_options=run_options)

        # Should have made exactly 2 calls
        assert call_count == 2

        # Should have gotten candidates from retry
        assert len(candidates) == 1
        assert candidates[0].raw_value == "John Doe"

    def test_invalid_json_twice_raises_llm_invalid_json_error(self) -> None:
        """Invalid JSON twice raises LLMInvalidJSONError."""
        client = ApiLLMClient(
            provider="openai",
            api_key="fake-key",
            model="test-model",
        )

        field = make_field_spec("dob")
        excerpts = [make_excerpt("doc_001", 1, "DOB: 1990-01-15")]
        run_options = RunOptions()

        call_count = 0

        def mock_call_api(messages: list, max_tokens: int) -> str:
            nonlocal call_count
            call_count += 1
            return "still not valid json"

        client._call_api = mock_call_api  # type: ignore

        with pytest.raises(LLMInvalidJSONError) as exc_info:
            client.extract_candidates(field, excerpts, run_options=run_options)

        # Should have made exactly 2 calls (original + 1 retry)
        assert call_count == 2
        assert exc_info.value.field == "dob"

    def test_valid_json_first_try_no_retry(self) -> None:
        """Valid JSON on first try doesn't trigger retry."""
        client = ApiLLMClient(
            provider="anthropic",
            api_key="fake-key",
            model="test-model",
        )

        field = make_field_spec("phone")
        excerpts = [make_excerpt("doc_001", 1, "Phone: 555-123-4567")]
        run_options = RunOptions()

        call_count = 0

        def mock_call_api(messages: list, max_tokens: int) -> str:
            nonlocal call_count
            call_count += 1
            return make_valid_json_response(
                "555-123-4567", "15551234567", "doc_001", 1, "Phone: 555-123-4567"
            )

        client._call_api = mock_call_api  # type: ignore

        candidates = client.extract_candidates(field, excerpts, run_options=run_options)

        # Should have made exactly 1 call
        assert call_count == 1
        assert len(candidates) == 1

    def test_empty_excerpts_returns_empty(self) -> None:
        """Empty excerpts list returns empty candidates without API call."""
        client = ApiLLMClient(
            provider="anthropic",
            api_key="fake-key",
            model="test-model",
        )

        field = make_field_spec("full_name")
        run_options = RunOptions()

        call_count = 0

        def mock_call_api(messages: list, max_tokens: int) -> str:
            nonlocal call_count
            call_count += 1
            return "[]"

        client._call_api = mock_call_api  # type: ignore

        candidates = client.extract_candidates(field, [], run_options=run_options)

        assert candidates == []
        assert call_count == 0  # No API call made


# --- Test FakeLLMClient ---


class TestFakeLLMClient:
    """Tests for FakeLLMClient test helper."""

    def test_returns_configured_responses(self) -> None:
        """Returns configured response sequence."""
        client = FakeLLMClient()

        field = make_field_spec("full_name")
        excerpts = [make_excerpt("doc_001", 1, "text")]
        run_options = RunOptions()

        candidate = Candidate(
            field="full_name",
            raw_value="Test",
            normalized_value="test",
            evidence=[Evidence(doc_id="doc_001", page=1, quoted_text="Test")],
            from_method="llm",
            scores=CandidateScores(anchor_match=1.0, validator=0.0, doc_relevance=0.0),
        )

        client.set_responses([[candidate]])

        result = client.extract_candidates(field, excerpts, run_options=run_options)

        assert len(result) == 1
        assert result[0].raw_value == "Test"

    def test_raises_configured_exception(self) -> None:
        """Raises configured exception."""
        client = FakeLLMClient()

        field = make_field_spec("full_name")
        excerpts = [make_excerpt("doc_001", 1, "text")]
        run_options = RunOptions()

        client.set_responses([LLMInvalidJSONError("full_name", "test error")])

        with pytest.raises(LLMInvalidJSONError):
            client.extract_candidates(field, excerpts, run_options=run_options)

    def test_tracks_calls(self) -> None:
        """Tracks all calls made."""
        client = FakeLLMClient()

        field1 = make_field_spec("full_name")
        field2 = make_field_spec("dob")
        excerpts = [make_excerpt("doc_001", 1, "text")]
        run_options = RunOptions()

        client.set_responses([[], []])

        client.extract_candidates(field1, excerpts, run_options=run_options)
        client.extract_candidates(field2, excerpts, run_options=run_options)

        calls = client.get_calls()
        assert len(calls) == 2
        assert calls[0][0].key == "full_name"
        assert calls[1][0].key == "dob"

    def test_empty_responses_returns_empty(self) -> None:
        """No configured responses returns empty list."""
        client = FakeLLMClient()

        field = make_field_spec("full_name")
        excerpts = [make_excerpt("doc_001", 1, "text")]
        run_options = RunOptions()

        result = client.extract_candidates(field, excerpts, run_options=run_options)

        assert result == []


# --- Test output candidate missing evidence (client responsibility check) ---


class TestCandidateMissingEvidence:
    """Tests that client returns candidates even with missing evidence.

    Per spec: "output candidate missing evidence → should still be returned by client,
    but rejected later (don't mix responsibilities)"
    """

    def test_client_returns_candidate_evidence_check_is_separate(self) -> None:
        """Client returns candidates; evidence check is done by orchestrator."""
        # The _parse_llm_response function skips items without valid evidence
        # But if the LLM returns evidence, even if hallucinated, the client
        # returns it. The evidence_supports_value check is in extract_candidates.

        # This test verifies the boundary: client parses what LLM gives,
        # orchestrator validates evidence.

        field = make_field_spec("full_name")

        # LLM returns candidate with evidence (whether valid is orchestrator's job)
        response = make_valid_json_response(
            raw_value="John Doe",
            normalized_value="john doe",
            doc_id="doc_001",
            page=1,
            quoted_text="Some completely unrelated text",  # Hallucinated evidence
        )

        candidates = _parse_llm_response(field, response)

        # Client returns the candidate - it has evidence fields populated
        assert len(candidates) == 1
        assert candidates[0].evidence[0].quoted_text == "Some completely unrelated text"

        # The evidence_supports_value check (in extract_candidates.py) would
        # reject this, but that's not the client's responsibility

