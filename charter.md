# L0: Demo Charter — Evidence-First Form Population

## Purpose

Build a one-day demo of an evidence-first document understanding pipeline that extracts patient intake fields from a bundle of input documents and (optionally) populates a target form. The demo's "senior" signal is traceability + fallbacks + review, not model novelty.

## Goals

1. **Accept target docs** (forms / schemas) + input docs (patient bundle) and produce:
   - `final.json` with field values + confidence + structured rationale + evidence citations
   - `populated.pdf` if the target is fillable, otherwise `population_report.json` (see non-goals)

2. **Handle target schema sources** with explicit resolution:

   **Schema resolution order (highest → lowest authority):**
   1. User-provided `schema.json`
   2. Fillable PDF AcroForm fields
   3. **Fallback:** V1 default field set (if none of the above succeed)

   If higher-priority schema exists, lower-priority sources are ignored for field definition (but may still provide evidence text).

   **Schema resolution never stops the run.** If no schema can be resolved, use the V1 default field set.

   **V1 only supports the V1 field set.** Schema resolution determines which of those fields are present and provides field labels/aliases—it does not define arbitrary new fields.

3. **Implement best-effort extraction** with:
   - Routing to likely docs
   - Extraction attempts (heuristics → model) with explicit ceiling:

     **Per field:**
     - Attempt 1: heuristic / regex candidate scan
     - Attempt 2: single LLM extraction call with evidence constraint
     - No iterative refinement loops

   - Contradiction detection → `needs_review`

4. **Ship a review UI:**
   - Per-field confirm / override / mark missing
   - Evidence view (page + highlight) read-only

5. **Ship reproducible run artifacts** + debug trace under `runs/`

6. **Ship a mini-eval harness:**
   - 5 manual fixtures
   - Metrics: evidence coverage, normalization correctness, `needs_review` rate, hallucination rate

## Non-Goals

- Security/privacy/compliance
- Anti-fraud / tamper detection
- OCR (V1 uses native PDF text extraction only)
- Database, accounts, auth, sharing, multi-tenant
- Generalized "any form ever" accuracy claims
- Editing evidence or manual annotation tools
- **Overlay PDF generation** (no visual overlays on non-fillable forms; use `population_report.json` instead)
- **Inferred schema from layout/labels** (stretch goal, not implemented in one-day demo)

## Hard Constraints and Invariants

### Evidence-First Invariant (No Hallucinations-by-Default)

- No value may be emitted unless it has evidence:
  - `doc_id` + `page` + `quoted_text` required
  - `bbox` required when available, optional otherwise
- Any model output lacking evidence is discarded (field becomes `missing`/`needs_review`)

**Hallucination check (deterministic):**

Accept candidate if ANY of the following holds:
1. `normalized_value` is a substring of `normalized_quoted_text` (after normalizing whitespace, punctuation, case), OR
2. For phone/date: field-specific regex matcher succeeds after normalization (e.g., `\d{3}-\d{3}-\d{4}` matches evidence digits)

**Listy fields (allergies, medications):**
- Each item in the list must be individually anchored in `quoted_text`
- If any item lacks anchoring, the entire field becomes `needs_review`
- No fuzzy token overlap; keep it deterministic and explainable

Normalization applied before comparison:
- Collapse whitespace
- Strip punctuation (except hyphens in dates/phones)
- Lowercase

If none of the above holds, the candidate is rejected as unsupported by evidence. **Evidence `quoted_text` is always required**—this check is not freeform.

### Three Field States (Deterministic)

| State | Definition |
|-------|------------|
| `filled` | Value present, validators pass, not flagged for contradiction |
| `needs_review` | Value exists but low confidence, validator warnings, or contradiction detected |
| `missing` | No acceptable candidate |

### Contradiction Policy (Simple + Strict)

- If two candidates with different normalized values both exceed a quality threshold, field becomes `needs_review` even if a "winner" is chosen.
- Still choose a winner for convenience, but require user confirmation.
- `final.json` always contains a selected winner (if any), even when `status = needs_review`. Review requirement is signaled via status, not absence of value.

**Contradiction handling (locked):**
- Do NOT re-query model to "resolve" contradiction
- Do NOT ask follow-up questions
- Just flag `needs_review` and show top 2 candidates in UI

### Failure Semantics

- Best-effort per-field: failures do not stop the run
- Schema resolution never stops the run (fallback to V1 field set)
- **LLM failures degrade per-field, not globally:**
  - If LLM call fails for a field, mark field as `missing` with rationale `["llm_unavailable"]`
  - Continue processing other fields
  - Do not stop the entire run unless there is no path forward
- Stop-run only on catastrophic errors:
  - No input docs provided
  - Cannot read target doc at all
  - All LLM calls fail AND heuristics found nothing (i.e., literally no path forward)
- **Extraction always runs** even if target schema fails (schema fallback applies)
- **Population gating:**
  - Population runs only if target is a fillable PDF
  - If target is not fillable, emit `population_report.json` with extracted values + `population_skipped: true` + reason
  - Pipeline can still extract even if target schema fails; PDF population is skipped with a report

### Reasoning Policy

- Structured rationale only, no freeform chain-of-thought prose.
- Rationale is a small list of factors, e.g.:
  ```json
  ["literal_anchor", "validator_passed", "best_doc_relevance", "cross_doc_agreement"]
  ```

### Text Extraction Constraint

- **V1: Native PDF text extraction only.** No OCR.
- Text extraction is treated as a black box that produces `{page, text, optional bbox}` units.
- No assumptions are made about reading order, table structure, or semantic blocks.
- If a PDF has no extractable text layer, the document is marked `text_extraction_failed: true` in `doc_index.json` and fields routed to it become `missing` with rationale `["no_text_layer"]`.

## User-Facing Model

### Inputs

- **target_docs**: list of either:
  - A fillable PDF form
  - A user schema JSON describing fields to extract/fill
- **input_docs**: list of PDFs containing patient info (must have extractable text layer; no images in V1)

### Fields (V1 Scope)

The system resolves a target schema from the user target_docs.

Then it tries to find those fields in the input_docs.

**V1 default field set** (used as fallback when no schema can be resolved from user input or fillable PDF):

| Field | Type |
|-------|------|
| `full_name` | string |
| `dob` | date |
| `phone` | phone |
| `address` | string |
| `insurance_member_id` | string |
| `allergies` | string_or_list |
| `medications` | string_or_list |

If the resolved schema contains fields outside this list, they are ignored in V1 (not processed). However, they are **reported** in `schema.json`:
- `unsupported_fields: [...]` — list of field keys that were requested but not supported
- Shown in UI as "not processed in v1"

### Review Actions

- Confirm value
- Override value (manual)
- Mark missing

Users cannot edit evidence.

## Technical Decisions (Locked)

| Component | Choice |
|-----------|--------|
| Frontend | Vite + React + TypeScript |
| Backend | FastAPI + Pydantic |
| Storage | Filesystem only, under `runs/` (no DB) |
| Text Extraction | Native PDF text extraction only (no OCR in V1) |
| LLM Provider | OpenAI (default) |

### LLM Provider Configuration (Locked)

- **Default provider:** OpenAI
- **Required env vars:**
  - `LLM_PROVIDER` — `openai` (default) or `anthropic` (stretch)
  - `LLM_MODEL` — model ID (e.g., `gpt-4o`, `gpt-4o-mini`)
  - `LLM_API_KEY` — API key for the provider
- **Response format:** JSON-only responses validated by Pydantic schemas
- **Retry policy:** Retry once on schema validation failure or transient error (5xx, timeout)
- **Max tokens per call:** 4096 (configurable via `LLM_MAX_TOKENS`)
- **Cost ceiling:** No hard limit in V1, but log estimated cost per run in trace

### Text Extraction Configuration

- **V1:** Native PDF text extraction using Python PDF libraries (e.g., `pdfplumber`, `PyMuPDF`)
- **No OCR.** Image-only PDFs or scanned documents will have no extractable text.
- **bbox:** Available when the PDF library provides character/word positions; omit if not available.

## Candidate Object Contract

Each candidate in `candidates.json` has this shape:

```json
{
  "field": "full_name",
  "raw_value": "John A. Smith",
  "normalized_value": "john a smith",
  "evidence": [
    {
      "doc_id": "doc_001",
      "page": 1,
      "quoted_text": "Patient Name: John A. Smith",
      "bbox": [100, 200, 300, 220]
    }
  ],
  "scores": {
    "anchor_match": 0.9,
    "validator": 1.0,
    "doc_relevance": 0.8
  },
  "validators": ["not_empty", "no_digits"],
  "rejected_reasons": []
}
```

**"Diff" in UI** means side-by-side display of top candidates; no fancy textual diff algorithms.

## Confidence Contract (Deterministic)

Confidence is computed from explicit signals (no self-reported confidence):

- Evidence literal anchor present
- Validator pass
- Doc relevance score
- Cross-doc agreement
- Contradiction penalty

**Thresholds (locked):**

| Parameter | Value |
|-----------|-------|
| Confidence range | `[0, 1]` |
| Auto-fill threshold | `0.75` |
| Contradiction penalty | `≥ 0.3` |

## Run Artifacts (Must Exist)

All runs persisted under `runs/<run_id>/`:

```
runs/<run_id>/
  input/
    target_docs/...
    input_docs/...
    request.json
  artifacts/
    schema.json              # resolved field schema (see structure below)
    doc_index.json           # docs + derived metadata
    layout.json              # extracted text/blocks/tokens where available
    routing.json             # field -> candidate docs
    candidates.json          # per-field candidate set
    final.json               # winners + evidence + confidence + rationale + status
    populated.pdf            # fillable PDFs only
    population_report.json   # non-fillable targets OR population skipped
  trace/
    trace.jsonl              # step logs
  eval/
    metrics.json             # if run is an eval
```

**`schema.json` structure:**
```json
{
  "schema_source": "user_schema|fillable_pdf|fallback_v1",
  "resolved_fields": [...],
  "unsupported_fields": ["field_x", "field_y"]
}
```

**Durability:** Runs persist for process lifetime / container disk only. No durability guarantees across deploys or restarts.

## Trace Contract (Must Exist)

Each trace event is one JSON object per line:

- `ts`, `run_id`, `step`, `status`, `duration_ms`
- `inputs_ref`, `outputs_ref` (paths into artifacts)
- `error` (string + kind) when present
- `model_calls` summary (provider/model, tokens, cost estimate, latency)

### Artifact Write Semantics (Locked)

- **Atomicity:** Every artifact write is atomic (write to temp file, then rename). Partial runs never corrupt outputs.
- **Idempotency:** Every step is idempotent when re-run with the same `run_id`. If an artifact already exists and matches expected output, skip the write. This enables safe re-runs and debugging.

## Mini-Eval (Required)

**Fixtures:** 5 manual fixtures with:
- Target schema/form + input bundle + ground truth for V1 fields

**Metrics:**

| Metric | Definition |
|--------|------------|
| Exact match | After normalization (dob/phone) |
| Evidence coverage rate | % filled fields with required evidence |
| `needs_review` rate | % fields flagged for review |
| Hallucination rate | % filled fields where `normalized_value` is not substring of `normalized_quoted_text` (or regex-equivalent for phone/date) |

## Success Criteria

From a fresh checkout: one command starts backend+frontend and completes a run that:

- [ ] Produces `final.json` with ≥5 fields filled
- [ ] Every filled field has evidence
- [ ] UI displays candidates + evidence + status and allows confirm/override
- [ ] Exports populated PDF for fillable forms (or `population_report.json` if not)
