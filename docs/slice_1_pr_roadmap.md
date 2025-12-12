pr graph

pr-01 (contracts) is the root.
then in parallel:
	•	pr-02 run fs + trace
	•	pr-03 schema resolution
	•	pr-04 pdf ingest + text extraction + doc_index
then:
	•	pr-05 routing
	•	pr-06 candidate extraction (heuristics + llm adapter)
	•	pr-07 scoring + contradiction + final.json
finally:
	•	pr-08 api endpoints wiring + artifact serving + “hello run”

you can collapse pr-08 into pr-02 if you want fewer PRs, but this is the clean parallel cut.

⸻

pr-01: contracts + schema validation

goal: establish all pydantic models + json schema constraints as the single source of truth.

changes
	•	backend/app/models/*.py (or domain.py): all pydantic models exactly as in l2
	•	backend/app/contracts.py: artifact name enum + mapping artifact→model
	•	backend/app/validate.py: validate_json(model, data) helpers

acceptance tests
	•	unit: instantiate/validate minimal examples for each model
	•	unit: ResolvedSchema rejects unsupported field keys
	•	unit: Evidence requires doc_id/page/quoted_text
	•	unit: FinalResult.schema_source literal validation

constraints
	•	no filesystem writes
	•	no pdf libs
	•	no fastapi routes

prompt pack (execute pr-01)
	•	implement models + validators + tests
	•	all new/changed behavior must be unit tested
	•	keep modules small; no pipeline code

⸻

pr-02: run folder + atomic artifact writer + trace logger

goal: implement run storage contract with atomic writes and append-only trace.

depends: pr-01

changes
	•	backend/app/runfs.py:
	•	create_run(run_id|auto) -> RunPaths
	•	write_json_atomic(path, data)
	•	copy_inputs_once(...)
	•	backend/app/trace.py:
	•	TraceLogger.append(event)
	•	context manager trace_step(step_name) captures duration_ms, status, errors
	•	tests for atomic write + idempotency semantics

acceptance tests
	•	unit: write_json_atomic writes .tmp then renames
	•	unit: trace append-only (two writes => two lines)
	•	unit: re-run with same run_id does not recopy inputs if present

constraints
	•	no pdf parsing
	•	no llm calls
	•	no fastapi routes

prompt pack
	•	implement runfs + trace with strict file layout under runs/<run_id>/
	•	ensure portability (relative paths)
	•	add tests that run fast

⸻

pr-03: schema resolution

goal: produce schema.json from: user schema → fillable pdf fields → fallback v1, plus unsupported_fields reporting.

depends: pr-01, pr-02 (for writing artifacts + trace)

changes
	•	backend/app/schema_resolver.py:
	•	resolve_schema(request) -> ResolvedSchema
	•	parse_user_schema(filebytes) -> ResolvedSchema
	•	resolve_from_acroform(target_pdf) -> ResolvedSchema (read-only)
	•	alias matching rule (as revised)
	•	write artifacts/schema.json

acceptance tests
	•	unit: user_schema precedence over pdf
	•	unit: acroform alias resolves exactly one key; ambiguous -> skipped + trace warn
	•	unit: fallback produces v1 default fields capped by max_fields ordering rule

constraints
	•	do not add pdf writing libs; read-only extraction ok
	•	no routing, no extraction from input_docs

prompt pack
	•	implement resolver and tests; log trace events for schema warnings

⸻

pr-04: ingest + doc_index + native pdf text extraction → layout.json

goal: ingest input docs, compute sha256, detect text layer, extract per-page text into layout.json.

depends: pr-01, pr-02

changes
	•	backend/app/ingest.py: assign doc_ids, copy files, compute sha256, mime sniff
	•	backend/app/pdf_text.py: native text extraction per page (library choice allowed but must be stable)
	•	backend/app/layout_builder.py: produce LayoutDoc[]
	•	emit doc_index.json and layout.json

acceptance tests
	•	unit: sha256 stable
	•	unit: empty-text pdf sets has_text_layer=false and unreadable_reason="no_text_layer"
	•	unit: layout pages are 1-indexed

constraints
	•	no OCR
	•	no routing or LLM
	•	keep extraction deterministic

prompt pack
	•	pick a single pdf lib and use it consistently
	•	prioritize correctness over fancy spans/bboxes (spans optional)

⸻

pr-05: routing

goal: produce routing.json mapping each field→top_k docs among readable docs using locked similarity.

depends: pr-01, pr-04, pr-03 (needs schema + layout + doc_index)

changes
	•	backend/app/routing.py:
	•	tokenization
	•	score function locked
	•	route_docs(fields, docs) -> list[RoutingEntry]
	•	emit routing.json

acceptance tests
	•	unit: score in [0,1]
	•	unit: unreadable docs excluded
	•	unit: deterministic ordering tie-breaker (doc_id)

constraints
	•	no LLM
	•	no candidate extraction

prompt pack
	•	implement routing + tests + artifact write + trace step

⸻

pr-06: candidate extraction (heuristics + llm adapter) → candidates.json

goal: for each field, generate candidates via heuristic pass, then optional single LLM call with strict JSON validation + one retry only on malformed JSON.

depends: pr-01, pr-04, pr-05, pr-02

changes
	•	backend/app/heuristics.py: per-field extractors that return Candidate(s) with Evidence
	•	backend/app/llm_client.py: provider abstraction (anthropic/openai), strict JSON mode, 1 retry on schema failure
	•	backend/app/extract_candidates.py: orchestrates per-field, enforces ceiling, writes candidates.json

acceptance tests
	•	unit: heuristic dob/phone normalization + evidence quoted_text anchors
	•	unit: llm invalid json triggers exactly 1 retry then fails with llm_invalid_json
	•	unit: evidence enforcement rejects unsupported_by_evidence
	•	unit: excerpt capping rules (max chars)

constraints
	•	do not select winners
	•	do not compute final confidence yet

prompt pack
	•	implement heuristics for all v1 fields minimally
	•	implement llm adapter with mockable interface; tests must not hit network

⸻

pr-07: scoring + contradiction + final.json

goal: validate/normalize, compute confidence deterministically, detect contradictions, select winner, emit final.json.

depends: pr-01, pr-06, pr-03

changes
	•	backend/app/validate_norm.py: validators + normalization (pure)
	•	backend/app/scoring.py: base_confidence, agreement, contradiction penalty
	•	backend/app/select.py: winner + status + alternatives (max 2) with acceptance semantics (your revised is_accepted or equivalent)
	•	emit final.json

acceptance tests
	•	unit: confidence math exact
	•	unit: contradiction triggers needs_review and penalty applied
	•	unit: low confidence (<0.75) => needs_review
	•	unit: missing when no acceptable candidates

constraints
	•	no LLM calls here
	•	no pdf operations

prompt pack
	•	pure functions + deterministic tests; no I/O besides artifact write via runfs

⸻

pr-08: fastapi wiring + artifact serving + “hello run”

goal: implement /api/runs and /api/runs/{run_id}/artifacts/{name} and wire the pipeline end-to-end.

depends: pr-02..pr-07

changes
	•	backend/app/main.py fastapi app
	•	backend/app/api.py routes
	•	backend/app/pipeline.py: run_pipeline(run_id, inputs, options) -> FinalResult
	•	minimal CORS for vite dev
	•	response includes run_id + artifact paths; status completed/failed

acceptance tests
	•	integration: POST with no input_docs => 400 no_input_docs
	•	integration: POST with small sample => returns run_id and schema/final endpoints readable
	•	integration: artifact name validation

constraints
	•	do not build UI here
	•	do not add DB

prompt pack
	•	add minimal integration tests using fastapi TestClient
	•	keep route logic thin; pipeline module owns orchestration

⸻

how to run in parallel (realistically)
	•	person/claude session A: pr-02
	•	session B: pr-03
	•	session C: pr-04
after those merge, pr-05/pr-06/pr-07 can be split, then pr-08.
