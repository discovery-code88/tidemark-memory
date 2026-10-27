use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnswerRequest {
    #[serde(default)]
    pub reasoning_mode: Option<crate::support::AnswerReasoningMode>,
    pub conversation_id: Option<String>,
    pub query: String,
    pub draft_answer: Option<String>,
    #[serde(default)]
    pub response_mode: Option<String>,
    #[serde(default)]
    pub answer_nodes: Option<usize>,
    #[serde(default)]
    pub detail_level: Option<String>,
    #[serde(default)]
    pub source_limit: Option<usize>,
    #[serde(default)]
    pub document_limit: Option<usize>,
    /// Explicit consent to send the selected evidence to a configured answer provider.
    #[serde(default)]
    pub allow_external_provider: Option<bool>,
    #[serde(default)]
    pub selected_memory_ids: Option<Vec<u64>>,
    /// The user's reply budget per model call, in tokens (server default
    /// `MEMORY_ANSWER_LLM_MAX_TOKENS`; an operator may set
    /// `MEMORY_ANSWER_LLM_MAX_TOKENS_CEILING`, unset by default).
    #[serde(default)]
    pub reply_max_tokens: Option<u32>,
    /// How many times a reply cut off at the budget may be continued (server
    /// default `MEMORY_ANSWER_LLM_MAX_CONTINUATIONS`; an operator may set
    /// `MEMORY_ANSWER_LLM_MAX_CONTINUATIONS_CEILING`, unset by default).
    #[serde(default)]
    pub max_continuations: Option<u32>,
    /// The user's choices for how model-assisted answers are built and
    /// checked; anything left out falls back to the server settings
    /// (`MEMORY_ANSWER_*`) and then to the defaults.
    #[serde(default)]
    pub answer_options: Option<AnswerOptions>,
    #[serde(skip)]
    pub(crate) selection_binding: Option<AnswerSelectionBinding>,
}

/// Per-request answer options. Every field is optional.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnswerOptions {
    /// Allow labelled inferences ("Inference: …"). Server: `MEMORY_ANSWER_INFERENCE`.
    #[serde(default)]
    pub inference: Option<bool>,
    /// Allow answers in the model's own words. Server: `MEMORY_ANSWER_REWORDED`.
    #[serde(default)]
    pub reworded_answers: Option<bool>,
    /// What happens to a model-worded statement the check cannot confirm:
    /// "label" (as "Inference: …", default), "drop", or "allow" (deliver it
    /// as an answer). Server: `MEMORY_ANSWER_UNCONFIRMED`.
    #[serde(default)]
    pub unconfirmed: Option<String>,
    /// Withhold answers whose premise the model judges false (trick
    /// questions). Server: `MEMORY_ANSWER_PREMISE_CHECK`.
    #[serde(default)]
    pub premise_check: Option<bool>,
    /// Only sentences said by or about the person asked may answer.
    /// Server: `MEMORY_ANSWER_SPEAKER_RULE`.
    #[serde(default)]
    pub speaker_rule: Option<bool>,
    /// Most sentences one answer may cite. Server: `MEMORY_ANSWER_MAX_SENTENCES`.
    #[serde(default)]
    pub max_sentences: Option<usize>,
    /// Most words of a word-for-word short answer (0 = no limit).
    /// Server: `MEMORY_ANSWER_VERBATIM_SPAN_MAX_WORDS`.
    #[serde(default)]
    pub verbatim_span_max_words: Option<usize>,
    /// Bring a turn's most relevant longer sentence along with a short pick.
    /// Server: `MEMORY_ANSWER_SMALL_TURN_COMPLETION`.
    #[serde(default)]
    pub turn_completion: Option<bool>,
    /// Picks of at most this many words count as short.
    /// Server: `MEMORY_ANSWER_TURN_COMPLETION_MAX_WORDS`.
    #[serde(default)]
    pub turn_completion_max_words: Option<usize>,
    /// A second look at unpicked sentences for list questions.
    /// Server: `MEMORY_ANSWER_SMALL_FIND_MORE`.
    #[serde(default)]
    pub find_more: Option<bool>,
    /// "json", "transcript", or "compact" (the transcript's sentences with
    /// short IDs, one line per turn, question last). Server:
    /// `MEMORY_ANSWER_SMALL_CONTEXT_FORMAT`.
    #[serde(default)]
    pub context_format: Option<String>,
    /// "first" or "last" (default): where the question stands relative to a
    /// compact context. Last lets calls about one conversation share a cached
    /// prefix; first puts the question before the evidence.
    /// Server: `MEMORY_ANSWER_QUESTION_POSITION`.
    #[serde(default)]
    pub question_position: Option<String>,
    /// "unstated" (default): a question whose asked fact the conversation does
    /// not state about the named person is rejected as a false premise.
    /// "presupposition": rejected only when it presupposes something the
    /// conversation contradicts, attributes to the wrong person or never
    /// mentions happening; an answer the conversation only implies goes to the
    /// labelled-inference path instead. Server: `MEMORY_ANSWER_PREMISE_SCOPE`.
    #[serde(default)]
    pub premise_scope: Option<String>,
    /// "premises" (default): an inference may use only the quoted sentences.
    /// "general": it may also use widely known general knowledge (calendars
    /// and holidays, geography, well-known products and works); it stays
    /// labelled "Inference:" and independently checked.
    /// Server: `MEMORY_ANSWER_INFERENCE_KNOWLEDGE`.
    #[serde(default)]
    pub inference_knowledge: Option<String>,
    /// Checks of reworded answers and inferences also see the session date
    /// line of every quoted sentence, so a relative time ("yesterday") can be
    /// dated. Server: `MEMORY_ANSWER_CHECK_SESSION_DATES`.
    #[serde(default)]
    pub check_session_dates: Option<bool>,
    /// A question the question plan classifies as speculative (its
    /// `inference_type`; see `MEMORY_ANSWER_QUESTION_CLASSIFICATION`) is
    /// marked as such for the model; if the model still rejects its premise,
    /// one follow-up call asks for its best inference.
    /// Server: `MEMORY_ANSWER_INFERENCE_WORDING`.
    #[serde(default)]
    pub inference_wording: Option<bool>,
    /// A premise rejection must cite the sentences showing the asked fact
    /// belongs to someone else; code verifies the citation (sent sentence,
    /// another speaker than the person asked about, shared topic). A rejection
    /// without such proof means "not stated", and one follow-up call asks for
    /// an answer (a labelled inference if only implied).
    /// Server: `MEMORY_ANSWER_PREMISE_PROOF`.
    #[serde(default)]
    pub premise_proof: Option<bool>,
    /// Typed tier: how many times to ask when the model rejects the premise
    /// (default 2: one more ask, abstaining only if both reject; 1 = once).
    /// Server: `MEMORY_ANSWER_PREMISE_ASKS`.
    #[serde(default)]
    pub premise_asks: Option<usize>,
    /// Typed tier: after an "unmentioned" verdict, the next ask says the
    /// answer may only be implied; cited replies are checked as labelled
    /// inferences only. Server: `MEMORY_ANSWER_INFER_UNMENTIONED` (default off).
    #[serde(default)]
    pub infer_unmentioned: Option<bool>,
    /// Typed answers also see the asked person's profile (facts collected
    /// from the whole conversation, each citing its sentences; built by
    /// `/memory/profile/conversation/build`). A cited profile fact is replaced
    /// by its original sentences. Server: `MEMORY_ANSWER_PERSON_PROFILES`.
    #[serde(default)]
    pub person_profiles: Option<bool>,
    /// Tell the model which sessions fall in a date the question names.
    /// Server: `MEMORY_ANSWER_QUESTION_PERIOD_HINT`.
    #[serde(default)]
    pub question_period_hint: Option<bool>,
    /// Date an undated picked sentence by another sentence of its turn.
    /// Server: `MEMORY_ANSWER_SAME_TURN_TIME`.
    #[serde(default)]
    pub same_turn_time: Option<bool>,
    /// Fewest distinct instances for an "at least N" count.
    /// Server: `MEMORY_ANSWER_COUNT_MIN_INSTANCES`.
    #[serde(default)]
    pub count_min_instances: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnswerShapeReceipt {
    pub receipt_format: String,
    pub response_mode: String,
    pub detail_level: String,
    pub answer_nodes_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_answer_nodes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_answer_nodes: Option<usize>,
    pub source_limit_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_source_limit: Option<usize>,
    pub effective_source_limit: usize,
    pub document_limit_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_document_limit: Option<usize>,
    pub effective_document_limit: usize,
    pub eligible: usize,
    pub selected: usize,
    pub omitted: usize,
    pub deferred: usize,
    pub blocked: usize,
    pub failed: usize,
    pub safety_envelope: serde_json::Value,
}

#[derive(Debug, Clone)]
pub(crate) struct AnswerSelectionBinding {
    pub source: String,
    pub workspace_id: Option<String>,
    pub workspace_revision: Option<u64>,
    pub projection_kind: Option<String>,
    pub selected_object_ids: Vec<String>,
    pub permission_digest: Option<String>,
    pub selection_digest: String,
    pub authority_backend: String,
    pub memory_bindings: Vec<AnswerSelectedMemoryBinding>,
}

#[derive(Debug, Clone)]
pub(crate) struct AnswerSelectedMemoryBinding {
    pub memory_id: u64,
    pub memory_version: u32,
    pub source_record_digest: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnswerSelectionReceipt {
    pub receipt_format: String,
    pub enforcement: String,
    pub source: String,
    pub scope_space_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace_revision: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub projection_kind: Option<String>,
    pub selected_object_ids: Vec<String>,
    pub requested_memory_ids: Vec<u64>,
    pub eligible_memory_ids: Vec<u64>,
    pub response_evidence_ids: Vec<u64>,
    pub excluded_unselected_memory_count: usize,
    pub exact_scope_enforced: bool,
    pub full_space_renderers_bypassed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub permission_digest: Option<String>,
    pub selection_digest: String,
    pub authority_backend: String,
    pub action_authority: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReflectRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub query: String,
    pub draft_answer: Option<String>,
    #[serde(default)]
    pub response_mode: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CoverageReport {
    pub claim: String,
    pub covered: bool,
    /// Backward-compatible record-level evidence IDs.
    pub evidence_ids: Vec<u64>,
    /// The exact projected facts used for this claim. A record can contain
    /// several unrelated statements, so memory IDs alone are not sufficient
    /// for provenance or evaluation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_atom_ids: Vec<String>,
    /// Source-native locators for the exact atoms above.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_source_anchors: Vec<AnswerEvidenceSourceAnchor>,
    /// Bounded, machine-readable explanation of how complementary evidence
    /// was combined. Empty means a directly observed single-atom claim.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub derivation: Vec<String>,
    /// One of: verified, inferred, unverified, or unsupported.
    #[serde(default)]
    pub verification_status: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnswerClaim {
    pub claim_id: String,
    pub text: String,
    /// Backward-compatible record-level evidence IDs.
    pub evidence_ids: Vec<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_atom_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_source_anchors: Vec<AnswerEvidenceSourceAnchor>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub derivation: Vec<String>,
    pub support_status: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnswerQueryPlanReceipt {
    pub plan_format: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    /// Named people that the answer must keep distinct.
    pub people: Vec<String>,
    /// Proper names and named constraints that are not grammatical actors.
    pub named_entities: Vec<String>,
    /// The value shape the final answer must provide.
    pub answer_target: String,
    /// Semantic answer shape used to select and verify complementary evidence.
    pub answer_shape: String,
    /// Whether the result must be an action distinct from the primary activity
    /// named in the question.
    pub requires_complementary_action: bool,
    /// Whether the question asks for a personal best, record, or other
    /// achievement-extremum event rather than a literal phrase match.
    pub achievement_extremum: bool,
    pub temporal_scope: String,
    /// Whether the user asks for the provenance or identity of a shared
    /// image, video, audio item, or other native media object.
    pub media_provenance_requested: bool,
    /// Whether the answer must preserve an event jointly involving every
    /// named participant rather than treating one participant as incidental.
    pub co_participant_event_requested: bool,
    /// Whether the answer is a recommendation inferred from separately
    /// grounded preferences or capabilities rather than a direct observation.
    pub recommendation_requested: bool,
    /// Whether a named owner's household dependent is the subject of a direct
    /// preference list, rather than the owner being the preference subject.
    pub household_dependent_preference: bool,
    /// Topic constraints that an inferred recommendation must explicitly use.
    pub recommendation_topic_terms: Vec<String>,
    /// Requested precision for a temporal answer: unspecified, year, month,
    /// week, or day.
    pub temporal_granularity: String,
    /// Explicit bounded time window that evidence must satisfy, when present.
    pub temporal_constraint: String,
    pub time_terms: Vec<String>,
    /// Canonical structural scopes requested by the user, such as `chapter:315`
    /// or `section:7`; they are distinct from quantities and answer objects.
    pub structural_selectors: Vec<String>,
    pub action_terms: Vec<String>,
    /// Requested answer objects that evidence must bind to the action.
    pub object_terms: Vec<String>,
    /// The requested class of list entries, such as movies or reports.
    pub answer_class_terms: Vec<String>,
    /// Relational context that may be supported by separate evidence atoms.
    pub qualifier_terms: Vec<String>,
    /// True when a relation clause constrains which participant may answer.
    pub participant_relationship_qualifier: bool,

    pub cardinality: String,
    pub quantity_terms: Vec<String>,
    pub list_scope: String,
    pub inference_type: String,
    pub negation_terms: Vec<String>,
    pub condition_terms: Vec<String>,
    /// Query values that must survive generation and claim verification.
    pub protected_terms: Vec<String>,
    pub requested_modalities: Vec<String>,
    pub required_slots: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnswerEvidenceSourceAnchor {
    /// Stable, source-native locator for the selected atom when one exists.
    pub anchor_format: String,
    /// Examples: `dialogue_turn`, `text_span`, or a domain-specific source locator.
    pub kind: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnswerEvidenceAtomReceipt {
    pub atom_id: String,
    pub memory_id: u64,
    pub evidence_type: String,
    /// One of: original, context, or expansion_only.
    pub grounding_role: String,
    pub modality: String,
    pub source: String,
    /// Kept separate from the record-level citation so callers can validate a
    /// specific source fact without exposing it in ordinary answer text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_anchor: Option<AnswerEvidenceSourceAnchor>,
    /// The turn speaker remains distinct from the person the fact is about.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    /// Explicit or first-person fact owners inferred from this atom. Mentioning
    /// a name alone never places it here.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fact_subjects: Vec<String>,
    /// People directly addressed by this atom, such as "Thanks, Caroline".
    /// Addressed people are retrieval hints, not fact owners.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub addressed_people: Vec<String>,
    /// People whose possessive relation is directly expressed in the atom.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub possessive_owners: Vec<String>,
    /// Reproducible subject attribution reasons, for example
    /// `speaker_utterance` or `explicit_named_subject`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub subject_binding: Vec<String>,
    pub score: f32,
    pub supports_slots: Vec<String>,
    /// Reranker probability that this atom's turn answers the question.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_relevance: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnswerEvidenceRejectionReceipt {
    pub atom_id: String,
    pub memory_id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_anchor: Option<AnswerEvidenceSourceAnchor>,
    /// Machine-readable reason why the atom was not allowed into the final
    /// proof plan. This is diagnostic only; it never means the source itself
    /// is false or should be deleted.
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnswerEvidencePlanReceipt {
    pub plan_format: String,
    /// Declares how source-native locators are exposed on selected evidence
    /// atoms so evaluators do not infer provenance from rendered text.
    pub source_anchor_format: String,
    pub candidate_pool_count: usize,
    pub reranked_candidate_count: usize,
    pub rerank_strategy: String,
    pub grounding_policy: String,
    pub document_candidates: usize,
    pub event_candidates: usize,
    pub statement_candidates: usize,
    pub native_media_candidates: usize,
    pub system_metadata_candidates: usize,
    pub original_evidence_candidates: usize,
    pub context_evidence_candidates: usize,
    pub expansion_only_candidates: usize,
    /// Original evidence atoms selected for claim planning before contextual
    /// session metadata is attached.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub selected_answer_atom_ids: Vec<String>,
    /// Selected original atoms that are proof-only supports for another answer
    /// atom. They stay visible to Claim planning but do not create an extra
    /// display node by themselves.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub selected_bridge_atom_ids: Vec<String>,
    pub selected_atom_count: usize,
    pub selected_atoms: Vec<AnswerEvidenceAtomReceipt>,
    /// Backend and models that scored semantic relevance, when enabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_scorer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_scored_turns: Option<usize>,
    /// Candidate records entering Evidence Plan, in retrieval order.
    pub candidate_memory_ids: Vec<u64>,
    /// Records selected by Evidence Plan for claim planning and rendering.
    pub selected_memory_ids: Vec<u64>,
    /// Candidate records rejected before claim planning.
    pub dropped_memory_ids: Vec<u64>,
    /// Bounded atom-level rejection log. This distinguishes an attribution
    /// conflict, missing typed slot, proof-budget stop, and unlinked partial
    /// evidence instead of collapsing all failures into reranking.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rejected_atoms: Vec<AnswerEvidenceRejectionReceipt>,
    pub selection_limit: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnswerSlotCoverageReceipt {
    pub slot: String,
    /// One of: covered, partial, or missing.
    pub status: String,
    pub evidence_ids: Vec<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnswerClaimPlanReceipt {
    pub claim_id: String,
    pub claim_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    pub required_slots: Vec<String>,
    /// Backward-compatible record-level evidence IDs.
    pub evidence_ids: Vec<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_atom_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_source_anchors: Vec<AnswerEvidenceSourceAnchor>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub derivation: Vec<String>,
    /// One of: planned, supported, inferred, unsupported, or not_generated.
    pub support_status: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnswerOperatorPlanReceipt {
    pub plan_format: String,
    pub planned_operators: Vec<String>,
    pub attempted_operators: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_operator: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnswerPipelineStageReceipt {
    pub stage: String,
    /// One of: completed, partial, failed, not_applicable, or unjudged.
    pub status: String,
    pub input_count: usize,
    pub output_count: usize,
    pub evidence_ids: Vec<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnswerPipelineReceipt {
    pub receipt_format: String,
    pub pipeline_version: String,
    pub query_plan: AnswerQueryPlanReceipt,
    pub evidence_plan: AnswerEvidencePlanReceipt,
    pub slot_coverage: Vec<AnswerSlotCoverageReceipt>,
    pub claim_plan: Vec<AnswerClaimPlanReceipt>,
    pub operator_plan: AnswerOperatorPlanReceipt,
    pub stage_trace: Vec<AnswerPipelineStageReceipt>,
    /// The sole evidence set allowed to back claims, response evidence, and citations.
    pub final_evidence_ids: Vec<u64>,
    pub answer_generator: String,
    pub legacy_fallback_used: bool,
    /// One of: none, retrieval_miss, slot_rerank_drop, claim_planning_error,
    /// citation_filter_mismatch, answer_expression_error, or insufficient_evidence.
    pub failure_class: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnswerEvidenceProjectionStatusRequest {
    #[serde(default)]
    pub conversation_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnswerEvidenceProjectionStatusResponse {
    pub status_format: String,
    pub backend: String,
    pub projector_version: String,
    pub conversation_id: Option<String>,
    pub total_records: usize,
    pub projected_records: usize,
    pub current_records: usize,
    pub missing_records: usize,
    pub stale_records: usize,
    pub atom_count: usize,
    pub completion_percent: f64,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnswerEvidenceProjectionRepairRequest {
    #[serde(default)]
    pub conversation_id: Option<String>,
    #[serde(default)]
    pub after_memory_id: Option<u64>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub dry_run: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnswerEvidenceProjectionRepairResponse {
    pub repair_format: String,
    pub backend: String,
    pub projector_version: String,
    pub conversation_id: Option<String>,
    pub dry_run: bool,
    pub selected: usize,
    pub repaired: usize,
    pub skipped: usize,
    pub atoms_written: usize,
    pub candidate_memory_ids: Vec<u64>,
    pub next_after_memory_id: Option<u64>,
    pub remaining_records: usize,
    pub complete: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SemanticAnswerNote {
    pub note_type: String,
    pub message: String,
    pub entity_id: Option<String>,
    pub attribute_key: Option<String>,
    pub scope: Option<String>,
    pub candidate_values: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnswerResponse {
    /// Backward-compatible alias for `direct_answer`.
    pub answer: String,
    pub direct_answer: String,
    pub answer_mode: String,
    /// One of: answered, partial, uncertain, or no_evidence.
    pub answer_status: String,
    #[serde(default)]
    pub claims: Vec<AnswerClaim>,
    pub confidence: String,
    pub evidence: Vec<RetrievedItem>,
    #[serde(default)]
    pub citations: Vec<AnswerCitation>,
    #[serde(default)]
    pub memories: Vec<RetrievedItem>,
    #[serde(default)]
    pub memory_count: usize,
    pub coverage: Vec<CoverageReport>,
    pub semantic_notes: Vec<SemanticAnswerNote>,
    #[serde(default)]
    pub memory_used: usize,
    #[serde(default)]
    pub memory_ignored: usize,
    #[serde(default)]
    pub conflict_count: usize,
    #[serde(default)]
    pub trace_id: String,
    #[serde(default)]
    pub runtime_enforcement: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection_receipt: Option<AnswerSelectionReceipt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer_shape: Option<AnswerShapeReceipt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer_pipeline: Option<AnswerPipelineReceipt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub performance: Option<RequestPerformanceReceipt>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SummaryResponse {
    pub summary_id: Option<u64>,
    pub summary: Option<String>,
    pub anchors: Vec<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PruneResponse {
    pub pruned: usize,
    pub pruned_ids: Vec<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecayResponse {
    pub decayed: usize,
    pub decayed_ids: Vec<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BackendResponse {
    pub backend: String,
    /// Semantic answer/retrieval models: `loading`, `ready`, or `off`.
    #[serde(default)]
    pub semantic_models: String,
    pub writable_fact_source: String,
    pub writable_fact_source_count: usize,
    pub cache_role: String,
    pub projection_role: String,
    pub rebuild_source: String,
    pub durable_write_failure_behavior: String,
    pub embedding: String,
    pub loaded_memories: usize,
    pub layer_counts: HashMap<String, usize>,
    pub embedding_provider: String,
    pub embedding_model: String,
    pub embedding_status_counts: HashMap<String, usize>,
    pub embedding_provider_counts: HashMap<String, usize>,
    pub embedding_model_counts: HashMap<String, usize>,
    pub taste_arbitration_provider: String,
    pub taste_arbitration_model: Option<String>,
    pub taste_arbitration_status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntelligenceLayerStatus {
    pub layer: String,
    pub status: String,
    pub active_by_default: bool,
    pub bundled: bool,
    pub configured: bool,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub capability: String,
    pub limitation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntelligenceStatusResponse {
    pub status_format: String,
    pub generated_at: DateTime<Utc>,
    pub active_default_layer: String,
    pub bundled_foundation_model_weights: bool,
    pub bundled_general_llm: bool,
    pub bundled_model_claim: String,
    pub optional_external_llm_configured: bool,
    pub optional_local_llm_configured: bool,
    pub layers: Vec<IntelligenceLayerStatus>,
    pub trace_policy: String,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReindexRequest {
    pub conversation_id: Option<String>,
    pub limit: Option<usize>,
    pub dry_run: Option<bool>,
    pub only_outdated_embeddings: Option<bool>,
    pub target_provider: Option<String>,
    pub target_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalVectorBackendSmokeRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub query: String,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalVectorBackendSmokeHit {
    pub memory_id: u64,
    pub score: f32,
    pub memory_type: Option<String>,
    pub evidence_preview: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalVectorBackendSmokeResponse {
    pub smoke_format: String,
    pub backend: String,
    pub status: String,
    pub space_id: String,
    pub embedding_provider: String,
    pub embedding_model: String,
    pub dims: usize,
    pub indexed_count: usize,
    pub compressed_bytes_estimate: usize,
    pub stable_ids: Vec<u64>,
    pub top_hits: Vec<LocalVectorBackendSmokeHit>,
    pub allowlist_hit_count: usize,
    pub removed_memory_id: Option<u64>,
    pub removed_hit_count_after_delete: usize,
    pub online_update_supported_in_proof: bool,
    pub persisted: bool,
    pub persist_path: Option<String>,
    pub reloaded_count: usize,
    pub reloaded_top_hit_ids: Vec<u64>,
    pub persistence_roundtrip_ok: bool,
    pub no_cloud_mode: bool,
    pub limitations: Vec<String>,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalVectorBackendBenchmarkRequest {
    #[serde(default)]
    pub memory_count: Option<usize>,
    #[serde(default)]
    pub dims: Option<usize>,
    #[serde(default)]
    pub query_count: Option<usize>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub allowlist_stride: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalVectorBackendBenchmarkResponse {
    pub benchmark_format: String,
    pub backend: String,
    pub status: String,
    pub memory_count: usize,
    pub dims: usize,
    pub query_count: usize,
    pub top_k: usize,
    pub allowlist_stride: usize,
    pub compressed_bytes_estimate: usize,
    pub build_ms: u64,
    pub total_search_ms: u64,
    pub p50_search_ms: f64,
    pub p95_search_ms: f64,
    pub max_search_ms: f64,
    pub recall_top1: f64,
    pub allowlist_selectivity: f64,
    pub no_cloud_mode: bool,
    pub reason_codes: Vec<String>,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReindexResponse {
    pub dry_run: bool,
    pub selected: usize,
    pub reindexed: usize,
    pub skipped: usize,
    pub backend: String,
    pub embedding: String,
    pub target_provider: String,
    pub target_model: String,
}

pub fn api_error(
    status: StatusCode,
    error: &str,
    error_code: &str,
    message: impl Into<String>,
    details: serde_json::Value,
) -> (StatusCode, Json<ApiErrorResponse>) {
    (
        status,
        Json(ApiErrorResponse {
            error: error.to_string(),
            error_code: error_code.to_string(),
            message: message.into(),
            details,
        }),
    )
}
