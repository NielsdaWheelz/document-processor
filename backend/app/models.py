"""
Pydantic models for the document-processor pipeline.

These models define the contract for all artifacts and domain types
used in the evidence-first form population system.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# V1 supported field keys - the authoritative list
SUPPORTED_FIELD_KEYS: set[str] = {
    "full_name",
    "dob",
    "phone",
    "address",
    "insurance_member_id",
    "allergies",
    "medications",
}

# Field type literal
FieldType = Literal["string", "date", "phone", "string_or_list"]

# Schema source literal
SchemaSource = Literal["user_schema", "fillable_pdf", "fallback_v1"]

# Field status literal
FieldStatus = Literal["filled", "needs_review", "missing"]

# Extraction method literal
ExtractionMethod = Literal["heuristic", "llm"]

# Unreadable reason literal
UnreadableReason = Literal["no_text_layer", "parse_error"]

# LLM provider literal
LLMProvider = Literal["anthropic", "openai"]


class RunOptions(BaseModel):
    """Options for configuring a pipeline run."""

    top_k_docs: int = 3
    llm_provider: LLMProvider = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"
    max_llm_tokens: int = 1200
    max_fields: int = 7


class FieldSpec(BaseModel):
    """Specification for a single field in the schema."""

    key: str
    label: str | None = None
    type: FieldType

    @field_validator("key")
    @classmethod
    def key_must_be_supported(cls, v: str) -> str:
        if v not in SUPPORTED_FIELD_KEYS:
            raise ValueError(f"Unsupported field key: {v}. Must be one of: {SUPPORTED_FIELD_KEYS}")
        return v


class ResolvedSchema(BaseModel):
    """The resolved schema for a run, including source and any unsupported fields."""

    schema_source: SchemaSource
    resolved_fields: list[FieldSpec]
    unsupported_fields: list[str] = Field(default_factory=list)


class DocIndexItem(BaseModel):
    """Metadata for a single document in the index."""

    doc_id: str
    filename: str
    mime_type: str
    pages: int | None = None
    has_text_layer: bool
    unreadable_reason: UnreadableReason | None = None
    sha256: str


class LayoutSpan(BaseModel):
    """A span of text with optional bounding box."""

    text: str
    bbox: list[float] | None = None


class LayoutPageText(BaseModel):
    """Text content for a single page."""

    page: int
    full_text: str
    spans: list[LayoutSpan] = Field(default_factory=list)

    @field_validator("page")
    @classmethod
    def page_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"Page must be >= 1, got {v}")
        return v


class LayoutDoc(BaseModel):
    """Layout information for a single document."""

    doc_id: str
    pages: list[LayoutPageText]


class RoutingEntry(BaseModel):
    """Routing decision for a single field."""

    field: str
    doc_ids: list[str]
    scores: dict[str, float]

    @field_validator("field")
    @classmethod
    def field_must_be_supported(cls, v: str) -> str:
        if v not in SUPPORTED_FIELD_KEYS:
            raise ValueError(f"Unsupported field key: {v}. Must be one of: {SUPPORTED_FIELD_KEYS}")
        return v


class Evidence(BaseModel):
    """Evidence supporting a candidate value."""

    doc_id: str
    page: int
    quoted_text: str
    bbox: list[float] | None = None

    @field_validator("doc_id")
    @classmethod
    def doc_id_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("doc_id must not be empty")
        return v

    @field_validator("page")
    @classmethod
    def page_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"Page must be >= 1, got {v}")
        return v

    @field_validator("quoted_text")
    @classmethod
    def quoted_text_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("quoted_text must not be empty")
        return v


def _validate_score_range(v: float, field_name: str) -> float:
    """Validate that a score is in [0, 1]."""
    if v < 0.0 or v > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {v}")
    return v


class CandidateScores(BaseModel):
    """Scores for a candidate extraction."""

    anchor_match: float
    validator: float
    doc_relevance: float
    cross_doc_agreement: float = 0.0
    contradiction_penalty: float = 0.0

    @field_validator("anchor_match")
    @classmethod
    def anchor_match_in_range(cls, v: float) -> float:
        return _validate_score_range(v, "anchor_match")

    @field_validator("validator")
    @classmethod
    def validator_in_range(cls, v: float) -> float:
        return _validate_score_range(v, "validator")

    @field_validator("doc_relevance")
    @classmethod
    def doc_relevance_in_range(cls, v: float) -> float:
        return _validate_score_range(v, "doc_relevance")

    @field_validator("cross_doc_agreement")
    @classmethod
    def cross_doc_agreement_in_range(cls, v: float) -> float:
        return _validate_score_range(v, "cross_doc_agreement")

    @field_validator("contradiction_penalty")
    @classmethod
    def contradiction_penalty_in_range(cls, v: float) -> float:
        return _validate_score_range(v, "contradiction_penalty")


class Candidate(BaseModel):
    """A candidate value for a field."""

    field: str
    raw_value: str
    normalized_value: str
    evidence: list[Evidence]
    from_method: ExtractionMethod
    validators: list[str] = Field(default_factory=list)
    rejected_reasons: list[str] = Field(default_factory=list)
    scores: CandidateScores

    @field_validator("field")
    @classmethod
    def field_must_be_supported(cls, v: str) -> str:
        if v not in SUPPORTED_FIELD_KEYS:
            raise ValueError(f"Unsupported field key: {v}. Must be one of: {SUPPORTED_FIELD_KEYS}")
        return v

    @field_validator("evidence")
    @classmethod
    def evidence_must_not_be_empty(cls, v: list[Evidence]) -> list[Evidence]:
        if not v:
            raise ValueError("evidence must not be empty - candidates must include evidence")
        return v

    @property
    def is_accepted(self) -> bool:
        """A candidate is accepted iff rejected_reasons is empty."""
        return len(self.rejected_reasons) == 0


class FinalField(BaseModel):
    """The final resolved value for a field."""

    field: str
    status: FieldStatus
    value: str | None = None
    normalized_value: str | None = None
    confidence: float
    rationale: list[str]
    evidence: list[Evidence] = Field(default_factory=list)
    alternatives: list[Candidate] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        return _validate_score_range(v, "confidence")


class FinalResult(BaseModel):
    """The final result of a pipeline run."""

    run_id: str
    schema_source: SchemaSource
    fields: dict[str, FinalField]
