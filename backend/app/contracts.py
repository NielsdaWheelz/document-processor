"""
Artifact contract mapping.

Maps artifact names to their corresponding pydantic models/types.
"""

from enum import Enum
from typing import Any

from app.models import (
    Candidate,
    DocIndexItem,
    FinalResult,
    LayoutDoc,
    ResolvedSchema,
    RoutingEntry,
)


class ArtifactName(str, Enum):
    """Names of artifacts produced by the pipeline."""

    SCHEMA = "schema"
    DOC_INDEX = "doc_index"
    LAYOUT = "layout"
    ROUTING = "routing"
    CANDIDATES = "candidates"
    FINAL = "final"


# Mapping from artifact name to its expected type.
# For list types, we use the inner type - validation happens in validate.py
ARTIFACT_MODEL_MAP: dict[ArtifactName, type[Any]] = {
    ArtifactName.SCHEMA: ResolvedSchema,
    ArtifactName.DOC_INDEX: DocIndexItem,  # list[DocIndexItem]
    ArtifactName.LAYOUT: LayoutDoc,  # list[LayoutDoc]
    ArtifactName.ROUTING: RoutingEntry,  # list[RoutingEntry]
    ArtifactName.CANDIDATES: Candidate,  # list[Candidate]
    ArtifactName.FINAL: FinalResult,
}

# Artifacts that are stored as lists
LIST_ARTIFACTS: set[ArtifactName] = {
    ArtifactName.DOC_INDEX,
    ArtifactName.LAYOUT,
    ArtifactName.ROUTING,
    ArtifactName.CANDIDATES,
}
