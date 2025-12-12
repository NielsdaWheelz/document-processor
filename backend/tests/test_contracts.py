"""
Unit tests for artifact contract mapping.
"""

import pytest

from app.contracts import ARTIFACT_MODEL_MAP, LIST_ARTIFACTS, ArtifactName
from app.models import (
    Candidate,
    DocIndexItem,
    FinalResult,
    LayoutDoc,
    ResolvedSchema,
    RoutingEntry,
)


class TestArtifactName:
    """Tests for ArtifactName enum."""

    def test_all_artifact_names(self):
        """Test all expected artifact names exist."""
        expected = {"schema", "doc_index", "layout", "routing", "candidates", "final"}
        actual = {a.value for a in ArtifactName}
        assert actual == expected

    def test_enum_values(self):
        """Test enum values are correct strings."""
        assert ArtifactName.SCHEMA.value == "schema"
        assert ArtifactName.DOC_INDEX.value == "doc_index"
        assert ArtifactName.LAYOUT.value == "layout"
        assert ArtifactName.ROUTING.value == "routing"
        assert ArtifactName.CANDIDATES.value == "candidates"
        assert ArtifactName.FINAL.value == "final"

    def test_enum_is_str(self):
        """Test that ArtifactName is a string enum."""
        assert isinstance(ArtifactName.SCHEMA, str)
        assert ArtifactName.SCHEMA == "schema"


class TestArtifactModelMap:
    """Tests for ARTIFACT_MODEL_MAP."""

    def test_all_artifacts_mapped(self):
        """Test all artifact names have a mapping."""
        for artifact in ArtifactName:
            assert artifact in ARTIFACT_MODEL_MAP

    def test_correct_model_mapping(self):
        """Test each artifact maps to the correct model."""
        assert ARTIFACT_MODEL_MAP[ArtifactName.SCHEMA] is ResolvedSchema
        assert ARTIFACT_MODEL_MAP[ArtifactName.DOC_INDEX] is DocIndexItem
        assert ARTIFACT_MODEL_MAP[ArtifactName.LAYOUT] is LayoutDoc
        assert ARTIFACT_MODEL_MAP[ArtifactName.ROUTING] is RoutingEntry
        assert ARTIFACT_MODEL_MAP[ArtifactName.CANDIDATES] is Candidate
        assert ARTIFACT_MODEL_MAP[ArtifactName.FINAL] is FinalResult

    def test_map_completeness(self):
        """Test map has exactly the expected number of entries."""
        assert len(ARTIFACT_MODEL_MAP) == len(ArtifactName)


class TestListArtifacts:
    """Tests for LIST_ARTIFACTS set."""

    def test_list_artifacts_correct(self):
        """Test the correct artifacts are marked as lists."""
        expected_lists = {
            ArtifactName.DOC_INDEX,
            ArtifactName.LAYOUT,
            ArtifactName.ROUTING,
            ArtifactName.CANDIDATES,
        }
        assert LIST_ARTIFACTS == expected_lists

    def test_non_list_artifacts(self):
        """Test non-list artifacts are not in LIST_ARTIFACTS."""
        assert ArtifactName.SCHEMA not in LIST_ARTIFACTS
        assert ArtifactName.FINAL not in LIST_ARTIFACTS

    def test_list_artifacts_count(self):
        """Test the number of list artifacts."""
        assert len(LIST_ARTIFACTS) == 4
