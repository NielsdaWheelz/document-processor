"""
Heuristic candidate extraction for V1 fields.

Provides deterministic, evidence-based extraction using regex patterns
and keyword proximity. Each extracted candidate includes Evidence with
doc_id, page, and exact quoted_text.
"""

from __future__ import annotations

import re
from typing import Callable

from app.models import (
    Candidate,
    CandidateScores,
    Evidence,
    FieldSpec,
    LayoutDoc,
)


# --- Normalization helpers (deterministic) ---


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    Rules:
    - Lowercase
    - Collapse whitespace (multiple spaces/tabs/newlines -> single space)
    - Strip punctuation except hyphens (for dates/phones)
    - Strip leading/trailing whitespace
    """
    # Lowercase
    text = text.lower()

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    # Strip punctuation except hyphens
    text = re.sub(r"[^\w\s\-]", "", text)

    # Strip leading/trailing whitespace
    return text.strip()


def normalize_date(raw: str) -> str | None:
    """
    Normalize a date string to YYYY-MM-DD format.

    Handles common formats:
    - MM/DD/YYYY, MM-DD-YYYY
    - YYYY-MM-DD, YYYY/MM/DD
    - Month DD, YYYY (e.g., January 15, 1990)
    - DD Month YYYY

    Returns None if parsing fails.
    """
    # Try YYYY-MM-DD or YYYY/MM/DD
    match = re.match(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", raw)
    if match:
        year, month, day = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    # Try MM/DD/YYYY or MM-DD-YYYY
    match = re.match(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", raw)
    if match:
        month, day, year = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    # Try Month DD, YYYY or Month DD YYYY
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    match = re.match(
        r"([a-zA-Z]+)\s*(\d{1,2}),?\s*(\d{4})",
        raw,
        re.IGNORECASE,
    )
    if match:
        month_str, day, year = match.groups()
        month_num = months.get(month_str.lower())
        if month_num:
            return f"{year}-{month_num:02d}-{int(day):02d}"

    # Try DD Month YYYY
    match = re.match(
        r"(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})",
        raw,
        re.IGNORECASE,
    )
    if match:
        day, month_str, year = match.groups()
        month_num = months.get(month_str.lower())
        if month_num:
            return f"{year}-{month_num:02d}-{int(day):02d}"

    return None


def normalize_phone(raw: str) -> tuple[str, bool]:
    """
    Normalize a phone number to digits only.

    Returns (normalized_digits, default_country_assumed).
    If the phone has fewer than 11 digits, assumes +1 country code.
    """
    # Extract digits only
    digits = re.sub(r"\D", "", raw)

    # Check if country code is present
    default_country_assumed = False

    if len(digits) == 10:
        # Assume US +1
        digits = "1" + digits
        default_country_assumed = True
    elif len(digits) == 11 and digits.startswith("1"):
        # Already has country code
        pass
    # Other lengths: keep as-is (may be invalid, validator will catch)

    return digits, default_country_assumed


def extract_digits(text: str) -> str:
    """Extract only digits from text."""
    return re.sub(r"\D", "", text)


# --- Pattern-based extractors ---


# Date patterns
DATE_PATTERNS = [
    # YYYY-MM-DD or YYYY/MM/DD
    r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
    # MM/DD/YYYY or MM-DD-YYYY
    r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b",
    # Month DD, YYYY
    r"\b([A-Za-z]+\s+\d{1,2},?\s+\d{4})\b",
    # DD Month YYYY
    r"\b(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b",
]

# Phone patterns
PHONE_PATTERNS = [
    # (XXX) XXX-XXXX
    r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
    # XXX-XXX-XXXX
    r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b",
    # +1 variants
    r"\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
    # 11 consecutive digits (with country code)
    r"\b1\d{10}\b",
    # 10 consecutive digits
    r"\b\d{10}\b",
]

# Insurance ID patterns (alphanumeric runs near keywords)
INSURANCE_KEYWORDS = ["member", "policy", "id", "insurance", "subscriber", "group"]
INSURANCE_ID_PATTERN = r"\b[A-Za-z0-9]{4,20}\b"


def _find_line_containing(text: str, match_start: int) -> str:
    """Find the full line containing a match position."""
    # Find line start
    line_start = text.rfind("\n", 0, match_start)
    if line_start == -1:
        line_start = 0
    else:
        line_start += 1

    # Find line end
    line_end = text.find("\n", match_start)
    if line_end == -1:
        line_end = len(text)

    return text[line_start:line_end].strip()


def _extract_dob_candidates(
    field: FieldSpec,
    doc: LayoutDoc,
) -> list[Candidate]:
    """Extract DOB candidates from a document."""
    candidates: list[Candidate] = []
    seen_values: set[str] = set()

    # Keywords that suggest DOB context
    dob_keywords = ["dob", "date of birth", "birthdate", "birth date", "born"]

    for page in doc.pages:
        text = page.full_text
        text_lower = text.lower()

        for pattern in DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                raw_value = match.group(0).strip()
                normalized = normalize_date(raw_value)

                if normalized is None:
                    continue

                if normalized in seen_values:
                    continue

                # Check if near a DOB keyword for anchor scoring
                match_pos = match.start()
                context_start = max(0, match_pos - 50)
                context = text_lower[context_start:match_pos]
                has_anchor = any(kw in context for kw in dob_keywords)

                # Get the line as evidence
                quoted_text = _find_line_containing(text, match_pos)

                if not quoted_text.strip():
                    continue

                seen_values.add(normalized)

                candidates.append(Candidate(
                    field=field.key,
                    raw_value=raw_value,
                    normalized_value=normalized,
                    evidence=[Evidence(
                        doc_id=doc.doc_id,
                        page=page.page,
                        quoted_text=quoted_text,
                    )],
                    from_method="heuristic",
                    validators=[],
                    rejected_reasons=[],
                    scores=CandidateScores(
                        anchor_match=1.0 if has_anchor else 0.0,
                        validator=0.0,
                        doc_relevance=0.0,
                    ),
                ))

    return candidates


def _extract_phone_candidates(
    field: FieldSpec,
    doc: LayoutDoc,
) -> list[Candidate]:
    """Extract phone candidates from a document."""
    candidates: list[Candidate] = []
    seen_values: set[str] = set()

    phone_keywords = ["phone", "mobile", "telephone", "tel", "cell", "contact"]

    for page in doc.pages:
        text = page.full_text
        text_lower = text.lower()

        for pattern in PHONE_PATTERNS:
            for match in re.finditer(pattern, text):
                raw_value = match.group(0).strip()
                normalized, default_country = normalize_phone(raw_value)

                # Must have at least 10 digits
                if len(normalized) < 10:
                    continue

                if normalized in seen_values:
                    continue

                # Check anchor
                match_pos = match.start()
                context_start = max(0, match_pos - 50)
                context = text_lower[context_start:match_pos]
                has_anchor = any(kw in context for kw in phone_keywords)

                quoted_text = _find_line_containing(text, match_pos)

                if not quoted_text.strip():
                    continue

                seen_values.add(normalized)

                validators: list[str] = []
                if default_country:
                    validators.append("default_country_assumed")

                candidates.append(Candidate(
                    field=field.key,
                    raw_value=raw_value,
                    normalized_value=normalized,
                    evidence=[Evidence(
                        doc_id=doc.doc_id,
                        page=page.page,
                        quoted_text=quoted_text,
                    )],
                    from_method="heuristic",
                    validators=validators,
                    rejected_reasons=[],
                    scores=CandidateScores(
                        anchor_match=1.0 if has_anchor else 0.0,
                        validator=0.0,
                        doc_relevance=0.0,
                    ),
                ))

    return candidates


def _extract_insurance_id_candidates(
    field: FieldSpec,
    doc: LayoutDoc,
) -> list[Candidate]:
    """Extract insurance member ID candidates from a document."""
    candidates: list[Candidate] = []
    seen_values: set[str] = set()

    for page in doc.pages:
        text = page.full_text
        text_lower = text.lower()

        # Find lines containing insurance keywords
        lines = text.split("\n")
        line_start = 0

        for line in lines:
            line_lower = line.lower()
            has_keyword = any(kw in line_lower for kw in INSURANCE_KEYWORDS)

            if has_keyword:
                # Look for alphanumeric IDs in this line
                for match in re.finditer(INSURANCE_ID_PATTERN, line):
                    raw_value = match.group(0)

                    # Skip if it looks like a date or common word
                    if normalize_date(raw_value) is not None:
                        continue
                    if raw_value.lower() in {"member", "policy", "insurance", "group", "subscriber"}:
                        continue

                    normalized = normalize_text(raw_value)

                    if normalized in seen_values:
                        continue

                    if not normalized:
                        continue

                    seen_values.add(normalized)

                    candidates.append(Candidate(
                        field=field.key,
                        raw_value=raw_value,
                        normalized_value=normalized,
                        evidence=[Evidence(
                            doc_id=doc.doc_id,
                            page=page.page,
                            quoted_text=line.strip(),
                        )],
                        from_method="heuristic",
                        validators=[],
                        rejected_reasons=[],
                        scores=CandidateScores(
                            anchor_match=1.0,  # Found near keyword
                            validator=0.0,
                            doc_relevance=0.0,
                        ),
                    ))

            line_start += len(line) + 1

    return candidates


def _extract_name_candidates(
    field: FieldSpec,
    doc: LayoutDoc,
) -> list[Candidate]:
    """Extract full name candidates from a document."""
    candidates: list[Candidate] = []
    seen_values: set[str] = set()

    name_patterns = [
        r"(?:patient\s+)?name\s*:\s*(.+)",
        r"full\s+name\s*:\s*(.+)",
        r"patient\s*:\s*(.+)",
    ]

    for page in doc.pages:
        text = page.full_text

        for line in text.split("\n"):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            for pattern in name_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    raw_value = match.group(1).strip()

                    # Clean up: remove trailing punctuation, limit length
                    raw_value = re.sub(r"[,;:\.\|]+$", "", raw_value).strip()

                    if len(raw_value) < 2 or len(raw_value) > 100:
                        continue

                    # Skip if mostly digits
                    digit_ratio = sum(c.isdigit() for c in raw_value) / len(raw_value)
                    if digit_ratio > 0.5:
                        continue

                    normalized = normalize_text(raw_value)

                    if normalized in seen_values:
                        continue

                    if not normalized:
                        continue

                    seen_values.add(normalized)

                    candidates.append(Candidate(
                        field=field.key,
                        raw_value=raw_value,
                        normalized_value=normalized,
                        evidence=[Evidence(
                            doc_id=doc.doc_id,
                            page=page.page,
                            quoted_text=line_stripped,
                        )],
                        from_method="heuristic",
                        validators=[],
                        rejected_reasons=[],
                        scores=CandidateScores(
                            anchor_match=1.0,
                            validator=0.0,
                            doc_relevance=0.0,
                        ),
                    ))

    return candidates


def _extract_address_candidates(
    field: FieldSpec,
    doc: LayoutDoc,
) -> list[Candidate]:
    """Extract address candidates from a document."""
    candidates: list[Candidate] = []
    seen_values: set[str] = set()

    address_patterns = [
        r"address\s*:\s*(.+)",
        r"street\s*:\s*(.+)",
        r"mailing\s+address\s*:\s*(.+)",
        r"home\s+address\s*:\s*(.+)",
    ]

    for page in doc.pages:
        text = page.full_text

        for line in text.split("\n"):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            for pattern in address_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    raw_value = match.group(1).strip()

                    if len(raw_value) < 5:
                        continue

                    normalized = normalize_text(raw_value)

                    if normalized in seen_values:
                        continue

                    if not normalized:
                        continue

                    seen_values.add(normalized)

                    candidates.append(Candidate(
                        field=field.key,
                        raw_value=raw_value,
                        normalized_value=normalized,
                        evidence=[Evidence(
                            doc_id=doc.doc_id,
                            page=page.page,
                            quoted_text=line_stripped,
                        )],
                        from_method="heuristic",
                        validators=[],
                        rejected_reasons=[],
                        scores=CandidateScores(
                            anchor_match=1.0,
                            validator=0.0,
                            doc_relevance=0.0,
                        ),
                    ))

    return candidates


def _extract_list_field_candidates(
    field: FieldSpec,
    doc: LayoutDoc,
    keywords: list[str],
) -> list[Candidate]:
    """Extract list-type field candidates (allergies, medications)."""
    candidates: list[Candidate] = []
    seen_values: set[str] = set()

    # Build pattern from keywords
    keyword_pattern = "|".join(re.escape(kw) for kw in keywords)
    pattern = rf"(?:{keyword_pattern})\s*:\s*(.+)"

    for page in doc.pages:
        text = page.full_text

        for line in text.split("\n"):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                raw_value = match.group(1).strip()

                if len(raw_value) < 2:
                    continue

                # Keep as string (spec allows string or list, we use string)
                normalized = normalize_text(raw_value)

                if normalized in seen_values:
                    continue

                if not normalized:
                    continue

                seen_values.add(normalized)

                candidates.append(Candidate(
                    field=field.key,
                    raw_value=raw_value,
                    normalized_value=normalized,
                    evidence=[Evidence(
                        doc_id=doc.doc_id,
                        page=page.page,
                        quoted_text=line_stripped,
                    )],
                    from_method="heuristic",
                    validators=[],
                    rejected_reasons=[],
                    scores=CandidateScores(
                        anchor_match=1.0,
                        validator=0.0,
                        doc_relevance=0.0,
                    ),
                ))

    return candidates


def _extract_allergies_candidates(
    field: FieldSpec,
    doc: LayoutDoc,
) -> list[Candidate]:
    """Extract allergies candidates."""
    return _extract_list_field_candidates(
        field, doc,
        keywords=["allergies", "allergy", "allergic to", "known allergies"],
    )


def _extract_medications_candidates(
    field: FieldSpec,
    doc: LayoutDoc,
) -> list[Candidate]:
    """Extract medications candidates."""
    return _extract_list_field_candidates(
        field, doc,
        keywords=["medications", "meds", "current medications", "prescriptions", "rx"],
    )


# Extractor dispatch table
_EXTRACTORS: dict[str, Callable[[FieldSpec, LayoutDoc], list[Candidate]]] = {
    "dob": _extract_dob_candidates,
    "phone": _extract_phone_candidates,
    "insurance_member_id": _extract_insurance_id_candidates,
    "full_name": _extract_name_candidates,
    "address": _extract_address_candidates,
    "allergies": _extract_allergies_candidates,
    "medications": _extract_medications_candidates,
}


def heuristic_candidates_for_field(
    field: FieldSpec,
    routed_docs: list[LayoutDoc],
) -> list[Candidate]:
    """
    Single-pass heuristic scan for a field across routed documents.

    Each candidate includes Evidence with doc_id, page, and quoted_text.
    Results are deterministic and include from_method="heuristic".

    Args:
        field: The field specification to extract.
        routed_docs: Documents routed to this field.

    Returns:
        List of Candidate objects, possibly empty.
    """
    extractor = _EXTRACTORS.get(field.key)

    if extractor is None:
        # Unknown field type - return empty
        return []

    all_candidates: list[Candidate] = []

    for doc in routed_docs:
        candidates = extractor(field, doc)
        all_candidates.extend(candidates)

    return all_candidates

