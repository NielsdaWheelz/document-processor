"""
LLM client abstraction for candidate extraction.

Provides a protocol for LLM interactions and a concrete implementation
for OpenAI/Anthropic providers with strict JSON parsing and retry logic.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Protocol

from pydantic import ValidationError

from app.excerpts import DocExcerpt
from app.heuristics import normalize_date, normalize_phone, normalize_text
from app.models import (
    Candidate,
    CandidateScores,
    Evidence,
    FieldSpec,
    RunOptions,
)


class LLMInvalidJSONError(Exception):
    """Raised when LLM returns invalid JSON after retry."""

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        self.message = message
        super().__init__(f"Invalid JSON from LLM for field {field}: {message}")


class LLMClient(Protocol):
    """Protocol for LLM candidate extraction."""

    def extract_candidates(
        self,
        field: FieldSpec,
        excerpts: list[DocExcerpt],
        *,
        run_options: RunOptions,
    ) -> list[Candidate]:
        """
        Extract candidates for a field using LLM.

        Args:
            field: The field specification to extract.
            excerpts: Document excerpts to analyze.
            run_options: Run configuration options.

        Returns:
            List of Candidate objects.

        Raises:
            LLMInvalidJSONError: If LLM returns invalid JSON after retry.
        """
        ...


def _build_extraction_prompt(
    field: FieldSpec,
    excerpts: list[DocExcerpt],
) -> str:
    """Build the extraction prompt for the LLM."""
    # Build context from excerpts
    context_parts: list[str] = []
    for excerpt in excerpts:
        context_parts.append(
            f"[Document: {excerpt.doc_id}, Page: {excerpt.page}]\n{excerpt.text}"
        )

    context = "\n\n".join(context_parts)

    # Field-specific instructions
    type_instructions = {
        "date": "Return the date in YYYY-MM-DD format.",
        "phone": "Return the phone number with digits only (include country code if present).",
        "string": "Return the value as a string.",
        "string_or_list": "Return as a string. If multiple values, separate with commas.",
    }

    type_hint = type_instructions.get(field.type, "Return the value as a string.")

    prompt = f"""Extract the value for field "{field.key}" from the following document excerpts.

Field: {field.key}
Type: {field.type}
{f'Label: {field.label}' if field.label else ''}

{type_hint}

IMPORTANT:
- You MUST include evidence showing where you found the value.
- The quoted_text must be an EXACT quote from the document.
- If you cannot find the field, return an empty array [].

Return ONLY valid JSON in this exact format:
[
  {{
    "raw_value": "the value as found in the document",
    "normalized_value": "the normalized/cleaned value",
    "evidence": [
      {{
        "doc_id": "document id where found",
        "page": page_number,
        "quoted_text": "exact quote from document containing the value"
      }}
    ]
  }}
]

Document excerpts:
{context}

Return ONLY the JSON array, no other text."""

    return prompt


def _build_retry_prompt(field: FieldSpec, original_error: str) -> str:
    """Build the retry prompt after invalid JSON."""
    return f"""Your previous response was not valid JSON. Error: {original_error}

Return ONLY a valid JSON array matching this schema for field "{field.key}":

[
  {{
    "raw_value": "string",
    "normalized_value": "string",
    "evidence": [
      {{
        "doc_id": "string",
        "page": number,
        "quoted_text": "string"
      }}
    ]
  }}
]

If no value found, return: []

Return ONLY the JSON array, nothing else."""


def _normalize_value_for_field(field: FieldSpec, raw_value: str) -> str:
    """Apply field-specific normalization."""
    if field.type == "date":
        normalized = normalize_date(raw_value)
        return normalized if normalized else normalize_text(raw_value)
    elif field.type == "phone":
        normalized, _ = normalize_phone(raw_value)
        return normalized
    else:
        return normalize_text(raw_value)


def _parse_llm_response(
    field: FieldSpec,
    response_text: str,
) -> list[Candidate]:
    """
    Parse LLM response into Candidate objects.

    Args:
        field: The field specification.
        response_text: Raw LLM response text.

    Returns:
        List of Candidate objects.

    Raises:
        ValueError: If parsing fails.
    """
    # Try to extract JSON from response (handle markdown code blocks)
    text = response_text.strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        # Find the end of the code block
        lines = text.split("\n")
        # Skip first line (```json or ```)
        lines = lines[1:]
        # Find closing ```
        for i, line in enumerate(lines):
            if line.strip() == "```":
                lines = lines[:i]
                break
        text = "\n".join(lines)

    # Parse JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    if not isinstance(data, list):
        raise ValueError("Response must be a JSON array")

    candidates: list[Candidate] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        raw_value = item.get("raw_value", "")
        if not raw_value:
            continue

        # Get or compute normalized value
        normalized_value = item.get("normalized_value")
        if not normalized_value:
            normalized_value = _normalize_value_for_field(field, raw_value)

        # Parse evidence
        evidence_list = item.get("evidence", [])
        if not evidence_list:
            continue

        evidence_objs: list[Evidence] = []
        for ev in evidence_list:
            if not isinstance(ev, dict):
                continue

            doc_id = ev.get("doc_id", "")
            page = ev.get("page", 0)
            quoted_text = ev.get("quoted_text", "")

            if not doc_id or not page or not quoted_text:
                continue

            try:
                evidence_objs.append(Evidence(
                    doc_id=str(doc_id),
                    page=int(page),
                    quoted_text=str(quoted_text),
                ))
            except (ValidationError, ValueError):
                continue

        if not evidence_objs:
            continue

        # Determine if default country was assumed for phone
        validators: list[str] = []
        if field.type == "phone":
            _, default_country = normalize_phone(raw_value)
            if default_country:
                validators.append("default_country_assumed")

        try:
            candidate = Candidate(
                field=field.key,
                raw_value=str(raw_value),
                normalized_value=str(normalized_value),
                evidence=evidence_objs,
                from_method="llm",
                validators=validators,
                rejected_reasons=[],
                scores=CandidateScores(
                    anchor_match=1.0,  # LLM claims match
                    validator=0.0,
                    doc_relevance=0.0,
                ),
            )
            candidates.append(candidate)
        except ValidationError:
            continue

    return candidates


class ApiLLMClient:
    """
    Concrete LLM client implementation for OpenAI/Anthropic.

    Supports strict JSON parsing with exactly one retry on malformed JSON.
    """

    def __init__(
        self,
        provider: Literal["anthropic", "openai"],
        api_key: str,
        model: str,
    ) -> None:
        """
        Initialize the LLM client.

        Args:
            provider: The LLM provider ("anthropic" or "openai").
            api_key: API key for the provider.
            model: Model identifier to use.
        """
        self.provider = provider
        self.api_key = api_key
        self.model = model

    def _call_api(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> str:
        """
        Make an API call to the LLM provider.

        This method is designed to be monkeypatched for testing.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            max_tokens: Maximum tokens in response.

        Returns:
            The response text from the LLM.

        Raises:
            RuntimeError: If the API call fails.
        """
        # TODO: Implement actual API calls using httpx or urllib
        # For now, this raises to indicate it needs implementation
        # Tests will monkeypatch this method
        raise NotImplementedError(
            "API calls not implemented. Use a mock/fake for testing."
        )

    def extract_candidates(
        self,
        field: FieldSpec,
        excerpts: list[DocExcerpt],
        *,
        run_options: RunOptions,
    ) -> list[Candidate]:
        """
        Extract candidates using the LLM.

        Implements the retry contract: exactly one retry on malformed JSON.

        Args:
            field: The field specification to extract.
            excerpts: Document excerpts to analyze.
            run_options: Run configuration options.

        Returns:
            List of Candidate objects.

        Raises:
            LLMInvalidJSONError: If LLM returns invalid JSON after retry.
        """
        if not excerpts:
            return []

        # Build initial prompt
        prompt = _build_extraction_prompt(field, excerpts)

        messages = [{"role": "user", "content": prompt}]

        # First attempt
        try:
            response = self._call_api(messages, run_options.max_llm_tokens)
            return _parse_llm_response(field, response)
        except ValueError as e:
            first_error = str(e)

        # Retry with stricter prompt
        retry_prompt = _build_retry_prompt(field, first_error)
        messages.append({"role": "assistant", "content": response if 'response' in dir() else ""})
        messages.append({"role": "user", "content": retry_prompt})

        try:
            response = self._call_api(messages, run_options.max_llm_tokens)
            return _parse_llm_response(field, response)
        except ValueError as e:
            raise LLMInvalidJSONError(field.key, str(e))


class FakeLLMClient:
    """
    Fake LLM client for testing.

    Can be configured to return specific responses or simulate failures.
    """

    def __init__(self) -> None:
        """Initialize with empty response queue."""
        self._responses: list[list[Candidate] | Exception] = []
        self._calls: list[tuple[FieldSpec, list[DocExcerpt]]] = []

    def set_responses(self, responses: list[list[Candidate] | Exception]) -> None:
        """Set the sequence of responses to return."""
        self._responses = list(responses)

    def get_calls(self) -> list[tuple[FieldSpec, list[DocExcerpt]]]:
        """Get the list of calls made to this client."""
        return self._calls

    def extract_candidates(
        self,
        field: FieldSpec,
        excerpts: list[DocExcerpt],
        *,
        run_options: RunOptions,
    ) -> list[Candidate]:
        """
        Return the next configured response.

        Args:
            field: The field specification.
            excerpts: Document excerpts.
            run_options: Run options.

        Returns:
            The next configured response.

        Raises:
            Exception: If configured to raise.
            IndexError: If no more responses configured.
        """
        self._calls.append((field, excerpts))

        if not self._responses:
            return []

        response = self._responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response

