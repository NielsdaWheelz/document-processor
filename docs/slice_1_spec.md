l2: slice 1 contract — pipeline core + reproducible artifacts

goal

implement the deterministic, evidence-first pipeline that produces schema.json, doc_index.json, layout.json, routing.json, candidates.json, and final.json under runs/<run_id>/artifacts/, with a complete trace/trace.jsonl. no ui requirements in this slice beyond “backend returns run_id and final.json path.”

scope

in:
	•	schema resolution: user schema → fillable pdf fields → fallback v1 field set
	•	native pdf text extraction only (no ocr)
	•	routing: cheap similarity to pick top-k docs per field
	•	per-field candidate extraction: one heuristic pass + at most one llm call
	•	validation + normalization + deterministic confidence
	•	contradiction detection → needs_review
	•	artifact persistence + trace timeline

out:
	•	pdf population writing (slice 2)
	•	override/confirm ui state (slice 2)
	•	layout/label inference for non-fillable forms
	•	image support / ocr
	•	eval harness (slice 3)

hard invariants
	1.	no value without evidence

	•	any emitted winner with value != null must include at least one evidence item with:
	•	doc_id, page, quoted_text (required)
	•	bbox optional
	•	any model/heuristic output lacking evidence is rejected.

	2.	extraction ceiling

	•	per field, per run:
	•	≤1 heuristic scan pass
	•	≤1 llm extraction call
	•	0 refinement loops / 0 re-prompts / 0 retries (except 1 retry for malformed json; see llm contract)

	3.	three field statuses

	•	filled | needs_review | missing only.

	4.	best-effort per field

	•	failures for one field never stop other fields.
	•	stop-run only if:
	•	no input docs provided, or
	•	cannot create run folder / write artifacts
	•	all other cases complete with missing fields (e.g. all docs unreadable → all fields missing, not a run failure)

	5.	slice 1a pdf spike gate (carry-forward constraint)

	•	if pdf write spike failed, slice 1 must not add pdf writing libs and must not attempt population.

v1 supported fields

slice 1 processes only these field keys (others are reported as unsupported in schema.json):
	•	full_name: string
	•	dob: date (normalized YYYY-MM-DD)
	•	phone: phone (normalized e164-ish; if country unknown, +1 default assumed, triggers needs_review)
	•	address: string
	•	insurance_member_id: string
	•	allergies: string_or_list
	•	medications: string_or_list

filesystem contract

all outputs go under runs/<run_id>/:

runs/<run_id>/
  input/
    request.json
    target_docs/...
    input_docs/...
  artifacts/
    schema.json
    doc_index.json
    layout.json
    routing.json
    candidates.json
    final.json
  trace/
    trace.jsonl

atomic writes

every artifact write must be atomic:
	•	write to *.tmp then rename to final path.

idempotency

if a run is re-executed with the same run_id, steps may overwrite artifacts, but must preserve:
	•	trace append-only (new events appended)
	•	input/ unchanged (do not re-copy if already present)

api contract (slice 1 only)

POST /api/runs

starts a run.

request (multipart form-data)
	•	target_docs: files[] (optional)
	•	input_docs: files[] (required, ≥1)
	•	schema_json: file (optional) — a json file containing UserSchema (preferred over inferring)
	•	options: json field (optional) — RunOptions

response 200

{
  "run_id": "2025-12-12T11-32-01Z_ab12cd",
  "status": "completed|failed",
  "artifacts": {
    "schema": "runs/<run_id>/artifacts/schema.json",
    "final": "runs/<run_id>/artifacts/final.json"
  }
}

error responses
	•	400 no_input_docs
	•	500 run_failed (with message)

GET /api/runs/{run_id}/artifacts/{name}

serves a specific artifact by name (name ∈ schema|doc_index|layout|routing|candidates|final).

response 200
	•	json content of artifact

errors
	•	404 artifact_not_found
	•	400 invalid_artifact_name

core domain types (pydantic models)

RunOptions

class RunOptions(BaseModel):
    top_k_docs: int = 3
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    llm_model: str | None = None  # default: "claude-sonnet-4-20250514" if anthropic, "gpt-4o-mini" if openai
    max_llm_tokens: int = 1200
    max_fields: int = 7  # process at most 7 supported fields

# llm_model resolution: if None, use provider default:
#   anthropic -> "claude-sonnet-4-20250514"
#   openai -> "gpt-4o-mini"

FieldType

"string" | "date" | "phone" | "string_or_list"

FieldSpec

class FieldSpec(BaseModel):
    key: str  # one of v1 supported keys
    label: str | None  # human label if known
    type: FieldType

ResolvedSchema

class ResolvedSchema(BaseModel):
    schema_source: Literal["user_schema", "fillable_pdf", "fallback_v1"]
    resolved_fields: list[FieldSpec]
    unsupported_fields: list[str] = []

DocIndexItem

class DocIndexItem(BaseModel):
    doc_id: str
    filename: str
    mime_type: str
    pages: int | None
    has_text_layer: bool
    unreadable_reason: Literal["no_text_layer", "parse_error"] | None = None
    sha256: str

LayoutPageText

(note: bbox optional; we only promise text + page.)

class LayoutSpan(BaseModel):
    text: str
    bbox: list[float] | None = None  # [x1,y1,x2,y2] page coords if available

class LayoutPageText(BaseModel):
    page: int  # 1-indexed
    full_text: str
    spans: list[LayoutSpan] = []  # optional, may be empty

LayoutDoc

class LayoutDoc(BaseModel):
    doc_id: str
    pages: list[LayoutPageText]

RoutingEntry

class RoutingEntry(BaseModel):
    field: str
    doc_ids: list[str]  # ordered best->worst, length<=top_k_docs
    scores: dict[str, float]  # doc_id -> relevance score

Evidence

class Evidence(BaseModel):
    doc_id: str
    page: int
    quoted_text: str
    bbox: list[float] | None = None

CandidateScores

class CandidateScores(BaseModel):
    anchor_match: float  # 0..1
    validator: float     # 0..1
    doc_relevance: float # 0..1
    cross_doc_agreement: float = 0.0
    contradiction_penalty: float = 0.0

Candidate

class Candidate(BaseModel):
    field: str
    raw_value: str
    normalized_value: str
    evidence: list[Evidence]
    from_method: Literal["heuristic", "llm"]
    validators: list[str] = []
    rejected_reasons: list[str] = []  # non-empty means rejected
    scores: CandidateScores

# acceptance rule: candidate is accepted iff rejected_reasons == []

FinalField

class FinalField(BaseModel):
    field: str
    status: Literal["filled", "needs_review", "missing"]
    value: str | None
    normalized_value: str | None
    confidence: float
    rationale: list[str]  # structured only
    evidence: list[Evidence] = []
    alternatives: list[Candidate] = []  # top non-winner candidates (accepted or rejected), max 2

FinalResult

class FinalResult(BaseModel):
    run_id: str
    schema_source: Literal["user_schema", "fillable_pdf", "fallback_v1"]
    fields: dict[str, FinalField]  # key = field key

schema resolution rules

input precedence
	1.	if schema_json is provided: parse UserSchema → ResolvedSchema(schema_source="user_schema")
	2.	else if target_docs includes a fillable pdf with acroform fields: generate ResolvedSchema(schema_source="fillable_pdf")
	3.	else: fallback to v1 default set (schema_source="fallback_v1")

user schema format (UserSchema)

{
  "fields": [
    { "key": "full_name", "label": "Patient Name", "type": "string" },
    ...
  ]
}

	•	only v1 supported keys are accepted into resolved_fields
	•	any other keys go to unsupported_fields

fillable pdf → field mapping

for slice 1, acroform fields determine presence, not semantic meaning:
	•	alias map is fixed in code:
	•	full_name: ["full_name", "name", "patient_name"]
	•	dob: ["dob", "date_of_birth", "birthdate"]
	•	phone: ["phone", "mobile", "telephone"]
	•	address: ["address", "street"]
	•	insurance_member_id: ["insurance_member_id", "member_id", "policy", "insurance_id"]
	•	allergies: ["allergies", "allergy"]
	•	medications: ["medications", "meds"]

	•	matching algorithm (locked):
	1.	normalize acroform field name: lowercase, replace _ and - with space
	2.	for each supported key, check if field name contains the exact key (e.g. "dob") → direct match
	3.	if no direct match, check aliases: field name contains exactly one alias from exactly one key → use that key
	4.	if field name matches aliases from multiple keys → ambiguous, skip field, record warning in trace
	5.	if no match → skip field

	•	ambiguity example: field "patient_name_dob" matches both full_name and dob aliases → ambiguous, skip

text extraction rules (pdf native only)
	•	for each pdf:
	•	attempt native text extraction per page
	•	if all pages yield empty/whitespace only: has_text_layer=false, unreadable_reason="no_text_layer"
	•	layout.json stores full_text per page and may omit spans/bboxes.

routing rules
	•	represent each doc as one string = concat first N chars of all pages (N fixed at 20000)
	•	represent each field as a query string = field.key + field.label + alias terms
	•	compute similarity (locked algorithm):
	•	tokenize: lowercase, split on whitespace/punctuation, discard tokens < 2 chars
	•	score = |tokens(field_query) ∩ tokens(doc)| / |tokens(field_query)|
	•	clamp to 0..1
	•	pick top_k docs for each field among readable docs.
	•	if no readable docs: routing still emits empty list; field will be missing.

candidate extraction rules

heuristic pass (attempt a)

per field, scan routed docs’ text for patterns:
	•	dob: date regex candidates; normalize to YYYY-MM-DD
	•	phone: phone regex candidates; normalize
	•	insurance_member_id: alnum runs near keywords “member”, “policy”, “id”
	•	full_name: “name:” lines; also near “patient”
	•	address: lines near “address:”
	•	allergies / medications: lines near keywords; split on commas/semicolons

heuristic candidates must include evidence:
	•	evidence quoted_text is the exact line/snippet from which value was derived
	•	page must be identified (page-local search)

llm pass (attempt b)

if heuristic produced no acceptable candidates OR confidence below auto-fill threshold, make one llm call.

llm input
	•	field spec
	•	routed doc excerpts (page-local snippets only; cap total chars)
	•	explicit instruction: return Candidate json with evidence quoting exact text

llm output contract
	•	must validate as Candidate (or list of Candidate; implementer choice but must be recorded in l3)
	•	any candidate whose evidence fails deterministic hallucination check is rejected.

llm retry contract (locked)
	•	if llm output fails Candidate pydantic validation (malformed json or missing fields):
	•	retry once with prompt: "Your response was invalid JSON. Return ONLY valid JSON matching this schema: {Candidate schema}"
	•	if second attempt also fails: mark field as missing with rejected_reasons=["llm_invalid_json"], continue to next field
	•	no other retries permitted (rate limits, timeouts, etc. → field missing with appropriate reason)

deterministic hallucination check (locked)

candidate is acceptable if:
	•	evidence quoted_text exists AND
	•	field-specific matcher passes:

matchers:
	•	string fields (full_name, address, insurance_member_id):
	•	normalized_value substring in normalized(quoted_text)
	•	dob:
	•	parsed normalized date matches any date-like substring in evidence after normalization
	•	phone:
	•	normalized digits match evidence digits (allow separators)
	•	listy:
	•	if value is list: every item must appear as substring in evidence (after normalization)
	•	if value is string: substring rule

if fails → candidate rejected with rejected_reasons += ["unsupported_by_evidence"]

validation + normalization rules
	•	validators produce warnings but do not mutate value silently.
	•	normalization functions are deterministic and pure.

required validators:
	•	full_name: not empty, contains letters, not mostly digits
	•	dob: valid date, not in future, age < 120
	•	phone: ≥10 digits (after normalization); if default country (+1) assumed → add "default_country_assumed" to validators and force needs_review
	•	address: not empty
	•	insurance_member_id: not empty, length bounds (e.g. 4..32)
	•	listy: not empty

validator scoring:
	•	pass = 1.0, warn = 0.6, fail = 0.0 (locked)

scoring + confidence (deterministic)

compute per candidate:
	•	anchor_match:
	•	1.0 if literal substring match (or regex-equivalent for dob/phone), else 0.0 (locked)
	•	doc_relevance: from routing score (0..1)
	•	validator: as above

base_confidence:

0.45*anchor_match + 0.30*validator + 0.25*doc_relevance

cross-doc agreement:
	•	if ≥2 candidates across different docs share same normalized_value: +0.10 (cap at 0.10)

contradiction:
	•	if there exist two different normalized values with base_confidence >= 0.60:
	•	set contradiction_penalty = 0.30 on winner
	•	field status becomes needs_review

final confidence:
	•	clamp(base_confidence + agreement_bonus - contradiction_penalty, 0, 1)

auto-fill threshold:
	•	0.75 (from l0)

winner selection rules
	•	choose candidate with max final_confidence
	•	if no acceptable candidates: field missing
	•	if winner exists but final_confidence < 0.75: field needs_review unless missing
	•	if contradiction triggered: field needs_review regardless of confidence

alternatives:
	•	include up to top 2 non-winner candidates by final_confidence (may be accepted or rejected)
	•	check rejected_reasons to distinguish: empty = accepted runner-up, non-empty = rejected

artifacts schemas

each artifact file must validate against its corresponding pydantic model(s):
	•	schema.json: ResolvedSchema
	•	doc_index.json: list[DocIndexItem]
	•	layout.json: list[LayoutDoc]
	•	routing.json: list[RoutingEntry]
	•	candidates.json: list[Candidate] — sorted by (field asc, final_confidence desc)
	•	final.json: FinalResult

trace contract

write trace/trace.jsonl, one json per line:

{
  "ts": "2025-12-12T11:40:12.123Z",
  "run_id": "...",
  "step": "extract_text",
  "status": "ok|warn|error",
  "duration_ms": 123,
  "inputs_ref": ["runs/<run_id>/input/input_docs/doc_001.pdf"],
  "outputs_ref": ["runs/<run_id>/artifacts/layout.json"],
  "error": { "kind": "parse_error", "message": "..." },
  "model_calls": [
    { "provider": "anthropic", "model": "...", "input_tokens": 123, "output_tokens": 45, "latency_ms": 900 }
  ]
}

required steps (must appear at least once per run):
	•	ingest
	•	resolve_schema
	•	extract_text
	•	route_docs
	•	extract_candidates
	•	score_select
	•	write_final

acceptance scenarios (slice-level)
	1.	happy path (heuristic-only works)

	•	given: pdf bundle contains “Patient Name: …” and “DOB: …”
	•	when: run
	•	then:
	•	final.json has filled for at least full_name and dob
	•	evidence exists
	•	no llm call in trace for those fields

	2.	llm used, evidence enforced

	•	given: insurance id present only in messy paragraph
	•	when: run
	•	then:
	•	candidate from llm exists with quoted evidence
	•	if evidence lacks literal support → field missing or needs_review with rejection reason

	3.	contradiction triggers needs_review

	•	given: two docs contain different DOBs
	•	when: run
	•	then:
	•	winner chosen
	•	status is needs_review
	•	alternatives include the other dob candidate

	4.	no text layer doc (all docs unreadable)

	•	given: single pdf with no extractable text (no other docs)
	•	when: run
	•	then:
	•	doc_index marks has_text_layer=false, unreadable_reason="no_text_layer"
	•	routing entries have doc_ids=[] (no readable docs to route to)
	•	all fields become missing with rationale including "no_readable_docs"

	5.	no input docs

	•	given: empty input_docs
	•	when: POST /api/runs
	•	then: 400 no_input_docs
