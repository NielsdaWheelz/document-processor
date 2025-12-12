# L1: Slice Plan (One-Day Demo)

---

## Slice 0 — Repo Scaffold + Golden "Hello Run"

**Goal:** A fresh checkout can run end-to-end and write a run folder, even if outputs are stubby.

**Depends on:** None

**Deliverable / Success Criteria:**
- `make dev` (or one command) starts backend + frontend
- Backend can accept a request and creates `runs/<run_id>/input/request.json`, `trace/trace.jsonl`
- UI can upload `target_docs` + `input_docs` and trigger a run (even if results are placeholder)

**Risk spike:** None (pure scaffolding)

---

## Slice 1a — PDF Reality Check (Blocking Spike)

**Goal:** Prove PDF filling works before writing pipeline code.

**Duration:** 15–30 min

**Depends on:** Slice 0

### Steps

1. Load one sample fillable PDF
2. List AcroForm fields
3. Write values to fields
4. Re-open PDF and verify fields persisted

### Decision Gate

- **If this succeeds:** Proceed to Slice 1 with fillable PDF support
- **If this fails:** Population downgraded to `population_report.json` only, immediately. Do not touch PDF writing again. If Slice 1a fails, no PDF-writing libraries may be added in later slices.

---

## Slice 1 — Pipeline Core + Reproducible Artifacts (The Body)

**Goal:** Implement the deterministic pipeline and artifact contract; produce `final.json` with evidence-first semantics for V1 fields.

**Depends on:** Slice 1a

**User-visible outcome:** Run completes and UI shows extracted fields + statuses.

### Success Criteria (Observable)

- `runs/<run_id>/artifacts/schema.json` produced via priority:
  1. User schema input
  2. Fillable PDF fields (AcroForm)
  3. Fallback V1 field set
- `layout.json`, `routing.json`, `candidates.json`, `final.json` are produced
- `final.json` contains at least 5 of the V1 fields when data exists
- `final.json` includes for each field:
  - Selected winner with evidence
  - Status (`filled` | `needs_review` | `missing`)
  - When `needs_review = true`: top 2 rejected candidates with rejection reasons
- Invariant enforced: no value without `doc_id` + `page` + `quoted_text` evidence
- Contradiction detection sets `needs_review` deterministically
- Trace timeline exists with per-step durations + error capture

### Pipeline Steps (Ordered)

1. **Ingest:** Copy inputs into `runs/<run_id>/input/…`
2. **Schema resolution:** Build `schema.json` from `target_docs` / user schema
   - Priority: user schema → fillable PDF fields → fallback V1 field set
   - No layout inference. No label mining. No fuzzy guessing.
3. **Doc text extraction:**
   - PDFs: extract native text layer only
   - If no text layer exists → mark document unreadable in `doc_index.json` (do not drop from index)
   - Routing may still select unreadable docs; extraction must short-circuit with rationale `["no_text_layer"]` → field becomes `missing`
   - No OCR in v1
   - Emit `layout.json` and `doc_index.json`
4. **Routing:** For each supported field, pick top-k docs (cheap similarity) → `routing.json`
5. **Candidate extraction (per field):**
   - Attempt A: Heuristic candidates (regex/validators)
   - Attempt B: Model extraction constrained to routed docs, must cite evidence
   - Emit `candidates.json`
6. **Validate + normalize + score candidates**
7. **Select winner + detect contradictions**
8. **Emit `final.json`**

### Extraction Ceiling (Invariant)

For each field, per run:
- At most one heuristic pass
- At most one LLM call
- No retries, no refinement loops, no re-prompts
- If both fail → `missing`
- Overrides do not trigger re-extraction (they only replace the selected value)

---

## Slice 2 — Review UI + Confirm/Override + Population Output

**Goal:** Make the demo feel real: human review loop + PDF population output.

**Depends on:** Slice 1

**User-visible outcome:** User can confirm/override per field and export a populated form or report.

### Success Criteria

- **UI shows:**
  - Field table with status, confidence, winner
  - Candidate drawer: top candidates + evidence (page + highlight, read-only)
  - Run log timeline (trace events: step, duration, warnings)
- **User actions persisted as override artifact:**
  - `artifacts/overrides.json` (confirm/override/missing)
  - `artifacts/final_effective.json` (final after overrides)
- **Population:**
  - If fillable (and Slice 1a passed): `populated.pdf` with supported fields filled
  - Else: `population_report.json` only
- Re-render uses `final_effective.json`, not raw model output

---

## Slice 3 — Mini Eval Harness (Only If Time Remains)

**Goal:** Prove engineering instincts with replayable fixtures.

**Depends on:** Slice 1 (Slice 2 optional)

### Success Criteria

- `eval/fixtures/<fixture_id>/` contains `target` + `bundle` + `ground_truth.json` (5 fixtures)
- `python -m eval.run_all` runs all fixtures and writes:
  - `runs/<run_id>/eval/metrics.json` per fixture
  - Aggregate `eval/report.json`
- **Metrics:**
  - Evidence coverage (% filled fields with required evidence)
  - Normalization correctness (dob/phone exact after normalization)
  - `needs_review` rate

---

## One-Day Ordering (Ruthless)

| Slice | Duration | Focus |
|-------|----------|-------|
| Slice 0 | 20–40 min | Wiring + run folder + trace stub |
| Slice 1a | 15–30 min | PDF spike → go/no-go |
| Slice 1 | 4–5 hrs | Schema → text extraction → extraction → `final.json` |
| Slice 2 | 2–3 hrs | UI review + overrides + populate or report |
| Slice 3 | 30–60 min | 5 fixtures + report (only if ahead) |

---

## Explicit Deferrals (To Prevent Scope Creep)

- OCR (all forms: document OCR, image OCR, poor-text recovery)
- Image document support
- Layout inference / label mining
- Multi-field generalization beyond V1 list
- Multi-run history UI
- Document annotation tools
- Any "planner agent" or graph runtime
- OCR tuning, table reconstruction, handwriting
- PDF overlay generation
