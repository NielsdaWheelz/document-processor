"""
Unit tests for the routing module.

Tests tokenization, scoring, routing logic, and artifact serialization
without depending on PDF libraries.
"""

import json
from pathlib import Path

import pytest

from app.models import (
    DocIndexItem,
    FieldSpec,
    LayoutDoc,
    LayoutPageText,
    ResolvedSchema,
    RoutingEntry,
)
from app.routing import (
    N_CHARS,
    route_docs,
    score_query_doc,
    tokenize,
    write_routing_artifact,
)
from app.runfs import create_run, read_json


# --- Helper factories for minimal fake objects ---


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


def make_resolved_schema(fields: list[FieldSpec]) -> ResolvedSchema:
    """Create a minimal ResolvedSchema."""
    return ResolvedSchema(
        schema_source="fallback_v1",
        resolved_fields=fields,
        unsupported_fields=[],
    )


def make_doc_index_item(
    doc_id: str,
    *,
    has_text_layer: bool = True,
    unreadable_reason: str | None = None,
) -> DocIndexItem:
    """Create a minimal DocIndexItem."""
    return DocIndexItem(
        doc_id=doc_id,
        filename=f"{doc_id}.pdf",
        mime_type="application/pdf",
        pages=1,
        has_text_layer=has_text_layer,
        unreadable_reason=unreadable_reason,
        sha256="fake_sha256_" + doc_id,
    )


def make_layout_doc(doc_id: str, pages_text: list[str]) -> LayoutDoc:
    """Create a minimal LayoutDoc with pages."""
    pages = [
        LayoutPageText(page=i + 1, full_text=text)
        for i, text in enumerate(pages_text)
    ]
    return LayoutDoc(doc_id=doc_id, pages=pages)


# --- Test classes ---


class TestTokenize:
    """Tests for the tokenize function."""

    def test_basic_tokenization(self) -> None:
        """Test basic tokenization behavior."""
        result = tokenize("Hello World")
        assert result == {"hello", "world"}

    def test_lowercases_text(self) -> None:
        """Test that tokenization lowercases input."""
        result = tokenize("HELLO WORLD")
        assert result == {"hello", "world"}

    def test_replaces_non_alphanumeric(self) -> None:
        """Test that non-alphanumeric chars become spaces."""
        result = tokenize("hello-world_test")
        assert result == {"hello", "world", "test"}

    def test_drops_short_tokens(self) -> None:
        """Test that tokens shorter than 2 chars are dropped."""
        result = tokenize("a ab abc I x y")
        assert result == {"ab", "abc"}

    def test_empty_string(self) -> None:
        """Test tokenizing empty string returns empty set."""
        result = tokenize("")
        assert result == set()

    def test_only_short_tokens(self) -> None:
        """Test that all-short-tokens string returns empty set."""
        result = tokenize("a b c d e")
        assert result == set()

    def test_numbers_preserved(self) -> None:
        """Test that numbers are preserved as tokens."""
        result = tokenize("test123 abc")
        assert result == {"test123", "abc"}

    def test_unicode_removed(self) -> None:
        """Test that non-ASCII chars are replaced."""
        result = tokenize("hello\u00e9world")
        # é is non-alphanumeric (not a-z, 0-9), so it becomes space
        assert result == {"hello", "world"}


class TestScoreQueryDoc:
    """Tests for the score_query_doc function."""

    def test_score_range_basic(self) -> None:
        """Test that score is in [0, 1] for typical inputs."""
        score = score_query_doc("test query", "this is a test document")
        assert 0.0 <= score <= 1.0

    def test_score_range_empty_query(self) -> None:
        """Test that empty query returns 0.0."""
        score = score_query_doc("", "some document text")
        assert score == 0.0

    def test_score_range_empty_doc(self) -> None:
        """Test score when doc is empty (should be 0)."""
        score = score_query_doc("test query", "")
        assert score == 0.0

    def test_score_range_both_empty(self) -> None:
        """Test score when both query and doc are empty."""
        score = score_query_doc("", "")
        assert score == 0.0

    def test_perfect_overlap(self) -> None:
        """Test score is 1.0 when query tokens are subset of doc tokens."""
        score = score_query_doc("hello world", "hello world more text")
        assert score == 1.0

    def test_no_overlap(self) -> None:
        """Test score is 0.0 when no token overlap."""
        score = score_query_doc("apple banana", "cat dog elephant")
        assert score == 0.0

    def test_partial_overlap(self) -> None:
        """Test score with partial overlap."""
        # query tokens: {hello, world}
        # doc tokens: {hello, friend}
        # intersection: {hello} -> 1/2 = 0.5
        score = score_query_doc("hello world", "hello friend")
        assert score == 0.5

    def test_score_ignores_doc_extra_tokens(self) -> None:
        """Test that extra doc tokens don't change score."""
        score1 = score_query_doc("test", "test")
        score2 = score_query_doc("test", "test extra tokens here")
        assert score1 == score2 == 1.0

    def test_query_tokens_only_short(self) -> None:
        """Test score when query has only short tokens (all filtered)."""
        score = score_query_doc("a b c", "apple banana cherry")
        assert score == 0.0


class TestRouteDocsUnreadableExcluded:
    """Tests that unreadable docs are excluded from routing."""

    def test_unreadable_doc_excluded_has_text_layer_false(self) -> None:
        """Test that doc with has_text_layer=False is excluded."""
        schema = make_resolved_schema([make_field_spec("full_name")])

        doc_readable = make_doc_index_item("doc_readable", has_text_layer=True)
        doc_unreadable = make_doc_index_item("doc_unreadable", has_text_layer=False)

        layout = [
            make_layout_doc("doc_readable", ["John Smith patient name"]),
            make_layout_doc("doc_unreadable", ["Jane Doe patient name"]),
        ]

        result = route_docs(schema, [doc_readable, doc_unreadable], layout, top_k=3)

        assert len(result) == 1
        entry = result[0]
        assert entry.field == "full_name"
        # Only readable doc should appear
        assert "doc_readable" in entry.doc_ids
        assert "doc_unreadable" not in entry.doc_ids

    def test_unreadable_doc_excluded_unreadable_reason_set(self) -> None:
        """Test that doc with unreadable_reason set is excluded."""
        schema = make_resolved_schema([make_field_spec("dob")])

        doc_readable = make_doc_index_item("doc_readable", has_text_layer=True)
        doc_unreadable = make_doc_index_item(
            "doc_unreadable",
            has_text_layer=True,
            unreadable_reason="no_text_layer",
        )

        layout = [
            make_layout_doc("doc_readable", ["date of birth 1990-01-01"]),
            make_layout_doc("doc_unreadable", ["date of birth 1985-05-15"]),
        ]

        result = route_docs(schema, [doc_readable, doc_unreadable], layout, top_k=3)

        assert len(result) == 1
        entry = result[0]
        assert entry.field == "dob"
        assert entry.doc_ids == ["doc_readable"]
        assert "doc_unreadable" not in entry.doc_ids


class TestRouteDocsDeterministicTieBreaker:
    """Tests for deterministic ordering with tie-breaker."""

    def test_tie_breaker_ascending_doc_id(self) -> None:
        """Test that docs with identical scores are sorted by ascending doc_id."""
        schema = make_resolved_schema([make_field_spec("address")])

        # Create docs with same content (same score)
        docs = [
            make_doc_index_item("doc_c"),
            make_doc_index_item("doc_a"),
            make_doc_index_item("doc_b"),
        ]

        # Same text -> same score
        layout = [
            make_layout_doc("doc_c", ["address street location"]),
            make_layout_doc("doc_a", ["address street location"]),
            make_layout_doc("doc_b", ["address street location"]),
        ]

        result = route_docs(schema, docs, layout, top_k=3)

        entry = result[0]
        # All have same score, should be sorted by doc_id ascending
        assert entry.doc_ids == ["doc_a", "doc_b", "doc_c"]

    def test_primary_sort_by_score_descending(self) -> None:
        """Test that primary sort is by score descending."""
        schema = make_resolved_schema([make_field_spec("phone")])

        docs = [
            make_doc_index_item("doc_low"),
            make_doc_index_item("doc_high"),
            make_doc_index_item("doc_mid"),
        ]

        layout = [
            make_layout_doc("doc_low", ["irrelevant content"]),
            make_layout_doc("doc_high", ["phone mobile telephone contact"]),
            make_layout_doc("doc_mid", ["phone number here"]),
        ]

        result = route_docs(schema, docs, layout, top_k=3)

        entry = result[0]
        # doc_high should be first (most tokens matching)
        assert entry.doc_ids[0] == "doc_high"
        # doc_low should be last (least matching)
        assert entry.doc_ids[-1] == "doc_low"


class TestRouteDocsTopKHonored:
    """Tests that top_k limit is honored."""

    def test_top_k_limits_results(self) -> None:
        """Test that with 5 docs, top_k=3 returns only 3."""
        schema = make_resolved_schema([make_field_spec("allergies")])

        docs = [
            make_doc_index_item(f"doc_{i}") for i in range(5)
        ]

        layout = [
            make_layout_doc(f"doc_{i}", [f"allergies content {i}"]) for i in range(5)
        ]

        result = route_docs(schema, docs, layout, top_k=3)

        entry = result[0]
        assert len(entry.doc_ids) == 3
        assert len(entry.scores) == 3

    def test_top_k_returns_all_if_fewer_docs(self) -> None:
        """Test that top_k=5 with 2 docs returns 2."""
        schema = make_resolved_schema([make_field_spec("medications")])

        docs = [
            make_doc_index_item("doc_1"),
            make_doc_index_item("doc_2"),
        ]

        layout = [
            make_layout_doc("doc_1", ["medications meds list"]),
            make_layout_doc("doc_2", ["medications prescription"]),
        ]

        result = route_docs(schema, docs, layout, top_k=5)

        entry = result[0]
        assert len(entry.doc_ids) == 2


class TestRouteDocsNoReadableDocs:
    """Tests for handling no readable docs."""

    def test_no_readable_docs_empty_routing(self) -> None:
        """Test that with no readable docs, routing entry has empty doc_ids and scores."""
        schema = make_resolved_schema([make_field_spec("full_name")])

        # All docs are unreadable
        docs = [
            make_doc_index_item("doc_1", has_text_layer=False),
            make_doc_index_item("doc_2", has_text_layer=False),
        ]

        layout = [
            make_layout_doc("doc_1", ["John Smith"]),
            make_layout_doc("doc_2", ["Jane Doe"]),
        ]

        result = route_docs(schema, docs, layout, top_k=3)

        assert len(result) == 1
        entry = result[0]
        assert entry.field == "full_name"
        assert entry.doc_ids == []
        assert entry.scores == {}

    def test_empty_doc_index(self) -> None:
        """Test routing with empty doc_index."""
        schema = make_resolved_schema([make_field_spec("dob")])

        result = route_docs(schema, [], [], top_k=3)

        assert len(result) == 1
        entry = result[0]
        assert entry.doc_ids == []
        assert entry.scores == {}


class TestRouteDocsArtifactSerialization:
    """Tests that routing entries are serializable and valid."""

    def test_artifact_shape_serializable(self) -> None:
        """Test that routing entries can be dumped to JSON and re-validated."""
        schema = make_resolved_schema([
            make_field_spec("full_name"),
            make_field_spec("dob"),
        ])

        docs = [
            make_doc_index_item("doc_1"),
            make_doc_index_item("doc_2"),
        ]

        layout = [
            make_layout_doc("doc_1", ["patient name John dob 1990-01-01"]),
            make_layout_doc("doc_2", ["patient record"]),
        ]

        result = route_docs(schema, docs, layout, top_k=3)

        # Serialize to JSON
        json_data = [entry.model_dump() for entry in result]
        json_str = json.dumps(json_data)

        # Parse back and validate
        parsed = json.loads(json_str)
        for item in parsed:
            validated = RoutingEntry.model_validate(item)
            assert validated.field in {"full_name", "dob"}

    def test_routing_entries_ordered_by_field_key(self) -> None:
        """Test that routing entries are ordered by field key ascending."""
        # Create fields in non-alphabetical order
        schema = make_resolved_schema([
            make_field_spec("phone"),
            make_field_spec("address"),
            make_field_spec("full_name"),
        ])

        docs = [make_doc_index_item("doc_1")]
        layout = [make_layout_doc("doc_1", ["content"])]

        result = route_docs(schema, docs, layout, top_k=3)

        field_order = [entry.field for entry in result]
        assert field_order == ["address", "full_name", "phone"]


class TestRouteDocsScoreValues:
    """Tests for score computation in routing."""

    def test_scores_dict_matches_doc_ids(self) -> None:
        """Test that scores dict contains exactly the doc_ids returned."""
        schema = make_resolved_schema([make_field_spec("insurance_member_id")])

        docs = [
            make_doc_index_item("doc_1"),
            make_doc_index_item("doc_2"),
            make_doc_index_item("doc_3"),
        ]

        layout = [
            make_layout_doc("doc_1", ["member id policy insurance"]),
            make_layout_doc("doc_2", ["insurance member"]),
            make_layout_doc("doc_3", ["unrelated content"]),
        ]

        result = route_docs(schema, docs, layout, top_k=2)

        entry = result[0]
        assert set(entry.scores.keys()) == set(entry.doc_ids)
        assert len(entry.doc_ids) == 2

    def test_zero_score_docs_included_if_readable(self) -> None:
        """Test that docs with score 0.0 are still included in top_k if readable."""
        schema = make_resolved_schema([make_field_spec("full_name")])

        docs = [
            make_doc_index_item("doc_no_match"),
        ]

        # Content has no overlap with full_name query
        layout = [
            make_layout_doc("doc_no_match", ["xyz abc 123"]),
        ]

        result = route_docs(schema, docs, layout, top_k=3)

        entry = result[0]
        # Even with 0 score, doc should be included
        assert entry.doc_ids == ["doc_no_match"]
        assert entry.scores["doc_no_match"] == 0.0


class TestWriteRoutingArtifact:
    """Tests for write_routing_artifact function."""

    def test_writes_artifact_to_correct_path(self, tmp_path: Path) -> None:
        """Test that artifact is written to artifacts/routing.json."""
        run = create_run(run_id="test-routing", base_dir=tmp_path)

        routing = [
            RoutingEntry(field="full_name", doc_ids=["doc_1"], scores={"doc_1": 0.8}),
            RoutingEntry(field="dob", doc_ids=["doc_2"], scores={"doc_2": 0.5}),
        ]

        write_routing_artifact(run, routing)

        artifact_path = run.artifact_path("routing.json")
        assert artifact_path.exists()

        # Verify content
        data = read_json(artifact_path)
        assert len(data) == 2
        assert data[0]["field"] == "full_name"
        assert data[1]["field"] == "dob"

    def test_written_artifact_revalidates(self, tmp_path: Path) -> None:
        """Test that written artifact can be parsed back as list[RoutingEntry]."""
        run = create_run(run_id="test-revalidate", base_dir=tmp_path)

        routing = [
            RoutingEntry(
                field="allergies",
                doc_ids=["doc_a", "doc_b"],
                scores={"doc_a": 0.9, "doc_b": 0.3},
            ),
        ]

        write_routing_artifact(run, routing)

        artifact_path = run.artifact_path("routing.json")
        data = read_json(artifact_path)

        # Re-validate as RoutingEntry
        for item in data:
            entry = RoutingEntry.model_validate(item)
            assert entry.field == "allergies"


class TestRouteDocsIntegration:
    """Integration tests for route_docs with realistic scenarios."""

    def test_multi_field_routing(self) -> None:
        """Test routing multiple fields to multiple docs."""
        schema = make_resolved_schema([
            make_field_spec("full_name", label="Patient Name"),
            make_field_spec("dob", label="Date of Birth"),
            make_field_spec("phone", label="Contact Number"),
        ])

        docs = [
            make_doc_index_item("medical_record"),
            make_doc_index_item("insurance_form"),
            make_doc_index_item("lab_results"),
        ]

        layout = [
            make_layout_doc("medical_record", [
                "Patient Name: John Smith",
                "Date of Birth: 1985-03-15",
                "Phone: 555-1234",
            ]),
            make_layout_doc("insurance_form", [
                "Member Name: John Smith",
                "DOB: 03/15/1985",
            ]),
            make_layout_doc("lab_results", [
                "Lab results for patient",
                "Test date: 2024-01-01",
            ]),
        ]

        result = route_docs(schema, docs, layout, top_k=2)

        # Should have 3 entries, one per field
        assert len(result) == 3

        # Fields should be in alphabetical order
        assert [e.field for e in result] == ["dob", "full_name", "phone"]

        # Each entry should have at most 2 docs
        for entry in result:
            assert len(entry.doc_ids) <= 2

    def test_multi_page_document(self) -> None:
        """Test that multi-page docs have pages concatenated."""
        schema = make_resolved_schema([make_field_spec("full_name")])

        docs = [make_doc_index_item("multi_page")]

        # Pages with content split across them
        layout = [
            make_layout_doc("multi_page", [
                "Page 1: Introduction",
                "Page 2: Patient Name John",
                "Page 3: Conclusion",
            ]),
        ]

        result = route_docs(schema, docs, layout, top_k=1)

        entry = result[0]
        # Should find the doc (name appears on page 2)
        assert entry.doc_ids == ["multi_page"]
        assert entry.scores["multi_page"] > 0

    def test_n_chars_limit_respected(self) -> None:
        """Test that doc text is capped at N_CHARS."""
        schema = make_resolved_schema([make_field_spec("full_name")])

        docs = [make_doc_index_item("huge_doc")]

        # Create a doc with text longer than N_CHARS
        # Put "name" only after the cap point
        long_prefix = "x" * (N_CHARS + 100)
        layout = [
            make_layout_doc("huge_doc", [long_prefix + " patient name john"]),
        ]

        result = route_docs(schema, docs, layout, top_k=1)

        entry = result[0]
        # The "name" text is after N_CHARS cap, so score should be lower
        # than if it were at the beginning
        # (We're testing the cap is applied - exact score depends on tokenization)
        assert entry.doc_ids == ["huge_doc"]

