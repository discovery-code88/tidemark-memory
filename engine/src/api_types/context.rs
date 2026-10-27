use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    pub conversation_id: Option<String>,
    pub query: String,
    pub top_k: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecallRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub query: String,
    pub top_k: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextBuildRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub task: String,
    pub project: Option<String>,
    pub repo_path: Option<String>,
    pub memory_mode: Option<String>,
    pub max_tokens: Option<usize>,
    pub max_latency_ms: Option<u64>,
    pub top_k: Option<usize>,
    pub recall_every_turns: Option<u64>,
    #[serde(default)]
    pub user_id: Option<String>,
    #[serde(default, alias = "client_id")]
    pub client_installation_id: Option<String>,
    #[serde(default)]
    pub task_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextAgentBootstrapRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub task: String,
    pub project: Option<String>,
    pub repo_path: Option<String>,
    pub memory_mode: Option<String>,
    pub max_tokens: Option<usize>,
    pub max_latency_ms: Option<u64>,
    pub top_k: Option<usize>,
    pub recall_every_turns: Option<u64>,
    #[serde(default)]
    pub user_id: Option<String>,
    #[serde(default, alias = "client_id")]
    pub client_installation_id: Option<String>,
    #[serde(default)]
    pub task_type: Option<String>,
    pub include_project_overview: Option<bool>,
    pub include_write_proposals: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextManagedTurnRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub task: String,
    pub project: Option<String>,
    pub repo_path: Option<String>,
    pub memory_mode: Option<String>,
    pub max_tokens: Option<usize>,
    pub max_latency_ms: Option<u64>,
    pub top_k: Option<usize>,
    pub recall_every_turns: Option<u64>,
    #[serde(default)]
    pub user_id: Option<String>,
    #[serde(default, alias = "client_id")]
    pub client_installation_id: Option<String>,
    #[serde(default)]
    pub task_type: Option<String>,
    pub include_project_overview: Option<bool>,
    pub draft_answer: Option<String>,
    pub max_write_candidates: Option<usize>,
    #[serde(default)]
    pub strict_mode: Option<bool>,
    #[serde(default)]
    pub auto_receipt: Option<bool>,
    #[serde(default)]
    pub client_name: Option<String>,
    #[serde(default)]
    pub client_version: Option<String>,
    #[serde(default)]
    pub actor: Option<String>,
    #[serde(default)]
    pub reasoning_policy: Option<MemoryReasoningPolicy>,
    #[serde(default, alias = "reasoning")]
    pub reasoning_level: Option<String>,
    #[serde(default, alias = "reasoning_visibility")]
    pub explanation_visibility: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReasoningPolicy {
    pub policy_format: String,
    pub level: String,
    pub explanation_visibility: String,
    pub tool_aggressiveness: String,
    pub retrieval_breadth: String,
    pub verification_strictness: String,
    pub concise_rationale: bool,
    #[serde(default)]
    pub model_effort: String,
    #[serde(default)]
    pub tool_call_budget: u32,
    #[serde(default)]
    pub retrieval_top_k_hint: u32,
    #[serde(default)]
    pub verification_passes: u32,
    #[serde(default)]
    pub runtime_budget: serde_json::Value,
    pub source: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReasoningPolicyRequest {
    #[serde(default, alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub level: Option<String>,
    #[serde(default, alias = "visibility")]
    pub explanation_visibility: Option<String>,
    #[serde(default)]
    pub task_type: Option<String>,
    #[serde(default)]
    pub user_id: Option<String>,
    #[serde(default, alias = "client_id")]
    pub client_installation_id: Option<String>,
    #[serde(default)]
    pub persist_default: Option<bool>,
    #[serde(default)]
    pub scope_type: Option<String>,
    #[serde(default)]
    pub scope_id: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReasoningPolicyResponse {
    pub reasoning_policy_format: String,
    pub generated_at: DateTime<Utc>,
    pub space_id: String,
    pub policy: MemoryReasoningPolicy,
    pub gateway_fields: serde_json::Value,
    pub trace_group: String,
    pub matched_default: Option<serde_json::Value>,
    pub default_scope: Option<serde_json::Value>,
    pub persisted_default: bool,
    pub runtime_budget: serde_json::Value,
    pub actions: Vec<serde_json::Value>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecallPolicyEnvelope {
    pub envelope_format: String,
    pub mode: String,
    pub source: String,
    pub outcome: String,
    pub reason: String,
    pub requires_user_decision: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_choice: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextManagedTurnContinueRequest {
    pub turn_id: String,
    pub recall_choice: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextManagedTurnResponse {
    pub managed_turn_format: String,
    pub generated_at: DateTime<Utc>,
    pub space_id: String,
    pub task: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    pub recall_policy: MemoryRecallPolicyEnvelope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_policy: Option<MemoryReasoningPolicy>,
    pub bootstrap: serde_json::Value,
    pub answer: serde_json::Value,
    pub context_trace_url: String,
    pub answer_trace_url: String,
    pub response_footer: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_summary: Option<AnswerTraceSummary>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub file_evidence: Vec<MemoryFileEvidence>,
    pub write_proposals: MemoryContextProposeResponse,
    pub approval_required: bool,
    pub gateway_runtime: GatewayRuntimeEnvelope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gateway_receipt: Option<MemoryContextTurnReceiptResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayRuntimeEnvelope {
    pub runtime_format: String,
    pub strict_mode: bool,
    pub enforcement: String,
    pub entrypoint: String,
    pub conformance_status: String,
    pub receipt_required: bool,
    pub receipt_id: Option<String>,
    pub receipt_valid: Option<bool>,
    pub bypass_visible: bool,
    pub approval_gate: MemoryContextDurableWriteGate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayRuntimeStatusRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub client_name: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayRuntimeEntrypointStatus {
    pub entrypoint: String,
    pub tool_or_endpoint: String,
    pub enforcement: String,
    pub client_scope: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayRuntimeStepStatus {
    pub step: String,
    pub enforcement: String,
    pub source: String,
    pub evidence_field: String,
    pub failure_status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayRuntimeConformanceSummary {
    pub receipt_count: usize,
    pub valid_receipt_count: usize,
    pub invalid_receipt_count: usize,
    pub client_names: Vec<String>,
    pub latest_receipt_id: Option<String>,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayRuntimeClientRequirement {
    pub requirement: String,
    pub enforcement: String,
    pub client_action: String,
    pub evidence_field: String,
    pub failure_status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayRuntimeClientWrapper {
    pub wrapper_format: String,
    pub status: String,
    pub python_method: String,
    pub typescript_method: String,
    pub required_entrypoint: String,
    pub validation_endpoint: String,
    pub fail_closed: bool,
    pub required_turn_options: Vec<String>,
    pub required_visibility: Vec<String>,
    pub rejects: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayRuntimeThirdPartyGuard {
    pub guard_format: String,
    pub status: String,
    pub enforcement_boundary: String,
    pub bypass_policy: String,
    pub required_entrypoint: String,
    pub runtime_status_endpoint: String,
    pub guard_validation_endpoint: String,
    pub receipt_endpoint: String,
    pub validation_endpoint: String,
    pub required_client_runtime: bool,
    pub can_block_external_direct_calls: bool,
    pub client_runtime_wrapper: GatewayRuntimeClientWrapper,
    pub requirements: Vec<GatewayRuntimeClientRequirement>,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayRuntimeStatusResponse {
    pub runtime_status_format: String,
    pub generated_at: DateTime<Utc>,
    pub space_id: String,
    pub enforcement_model: String,
    pub owned_runtime_required: bool,
    pub strict_clients_require_receipts: bool,
    pub direct_bypass_conformance_status: String,
    pub third_party_client_policy: String,
    pub entrypoints: Vec<GatewayRuntimeEntrypointStatus>,
    pub enforced_steps: Vec<GatewayRuntimeStepStatus>,
    pub approval_gate: MemoryContextDurableWriteGate,
    pub rollback_endpoints: Vec<String>,
    pub receipt_endpoint: String,
    pub receipts_endpoint: String,
    pub third_party_runtime_guard: GatewayRuntimeThirdPartyGuard,
    pub ui_review: String,
    pub conformance: GatewayRuntimeConformanceSummary,
    pub recent_receipts: Vec<MemoryContextTurnReceiptItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayRuntimeGuardValidateRequest {
    #[serde(default, alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub client_name: Option<String>,
    #[serde(default)]
    pub attempted_entrypoint: Option<String>,
    #[serde(default)]
    pub attempted_direct_endpoint: Option<String>,
    #[serde(default)]
    pub project_like: Option<bool>,
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub receipt_id: Option<String>,
    #[serde(default)]
    pub displayed_trace: Option<bool>,
    #[serde(default)]
    pub displayed_response_footer: Option<bool>,
    #[serde(default)]
    pub opened_approval_ui: Option<bool>,
    #[serde(default)]
    pub rollback_visible: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayRuntimeGuardValidateResponse {
    pub guard_validation_format: String,
    pub generated_at: DateTime<Utc>,
    pub space_id: String,
    pub client_name: String,
    pub project_like: bool,
    pub attempted_entrypoint: String,
    pub decision: String,
    pub conformance_status: String,
    pub receipt_valid: Option<bool>,
    pub matched_receipt_id: Option<String>,
    pub matched_request_id: Option<String>,
    pub missing_requirements: Vec<String>,
    pub failure_statuses: Vec<String>,
    pub warnings: Vec<String>,
    pub required_entrypoint: String,
    pub runtime_status_endpoint: String,
    pub receipt_endpoint: String,
    pub validation_endpoint: String,
    pub client_runtime_can_block: bool,
    pub server_can_block_unintegrated_direct_calls: bool,
    pub user_copy: String,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextAgentBootstrapResponse {
    pub bootstrap_format: String,
    pub generated_at: DateTime<Utc>,
    pub space_id: String,
    pub task: String,
    pub decision: MemoryContextDecideResponse,
    pub working_context: MemoryContextBuildResponse,
    pub project_context: Option<MemoryContextProjectResponse>,
    pub taste_profile: Vec<MemoryContextEvidence>,
    pub write_proposals: Option<MemoryContextProposeResponse>,
    pub request_id: String,
    pub trace_url: String,
    pub injectable_prompt: String,
    pub client_instructions: Vec<String>,
    pub execution_contract: MemoryContextAgentExecutionContract,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextAgentExecutionContract {
    pub contract_format: String,
    pub contract_id: String,
    pub enforcement_level: String,
    pub managed_turn_tool: String,
    pub turn_sequence: Vec<MemoryContextAgentTurnStep>,
    pub required_client_operations: Vec<MemoryContextAgentClientOperation>,
    pub display_artifacts: Vec<MemoryContextAgentDisplayArtifact>,
    pub required_recovered_state: Vec<String>,
    pub response_obligations: Vec<String>,
    pub approval_obligations: Vec<String>,
    pub required_answer_fields: Vec<String>,
    pub correction_triggers: Vec<String>,
    pub validation_endpoint: String,
    pub validation_payload_template: serde_json::Value,
    pub durable_write_gate: MemoryContextDurableWriteGate,
    pub next_actions: Vec<MemoryContextAgentNextAction>,
    pub trace: MemoryContextAgentTraceContract,
    pub pending_write_candidate_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextAgentTurnStep {
    pub step: String,
    pub status: String,
    pub source: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextAgentClientOperation {
    pub operation: String,
    pub timing: String,
    pub required: bool,
    pub source_field: String,
    pub tool_or_endpoint: Option<String>,
    pub evidence_field: String,
    pub failure_mode: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextAgentDisplayArtifact {
    pub artifact: String,
    pub source_field: String,
    pub required: bool,
    pub display_hint: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextAgentTraceContract {
    pub request_id: String,
    pub trace_url: String,
    pub must_show_to_user: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextDurableWriteGate {
    pub default_policy: String,
    pub approval_required: bool,
    pub commit_endpoint: String,
    pub forbidden_without: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextAgentNextAction {
    pub action: String,
    pub endpoint: Option<String>,
    pub required: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextTurnContractValidateRequest {
    pub request_id: String,
    pub injected_context: Option<bool>,
    pub displayed_trace: Option<bool>,
    pub displayed_trace_url: Option<String>,
    pub displayed_response_footer: Option<bool>,
    pub proposed_memory_updates: Option<bool>,
    pub opened_approval_ui: Option<bool>,
    pub committed_without_approval: Option<bool>,
    pub approval_state: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextTurnContractValidationCheck {
    pub check: String,
    pub status: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextTurnContractValidateResponse {
    pub validation_format: String,
    pub request_id: String,
    pub trace_url: String,
    pub valid: bool,
    pub checks: Vec<MemoryContextTurnContractValidationCheck>,
    pub required_next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextTurnReceiptRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub client_name: Option<String>,
    #[serde(default)]
    pub client_version: Option<String>,
    #[serde(default)]
    pub actor: Option<String>,
    #[serde(default)]
    pub response_id: Option<String>,
    #[serde(default)]
    pub answer_trace_url: Option<String>,
    #[serde(default)]
    pub rendered_answer_preview: Option<String>,
    #[serde(flatten)]
    pub validation: MemoryContextTurnContractValidateRequest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextTurnReceiptItem {
    pub receipt_id: String,
    pub request_id: String,
    pub space_id: String,
    pub client_name: String,
    pub client_version: Option<String>,
    pub actor: String,
    pub response_id: Option<String>,
    pub valid: bool,
    pub trace_url: String,
    pub answer_trace_url: Option<String>,
    pub rendered_answer_preview: Option<String>,
    pub validation: MemoryContextTurnContractValidateResponse,
    pub created_at: DateTime<Utc>,
    pub durable_memory_id: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextTurnReceiptResponse {
    pub receipt_format: String,
    pub receipt: MemoryContextTurnReceiptItem,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextTurnReceiptsRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub client_name: Option<String>,
    #[serde(default)]
    pub valid: Option<bool>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextTurnReceiptsResponse {
    pub receipt_store_format: String,
    pub space_id: String,
    pub items: Vec<MemoryContextTurnReceiptItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextTasteProfileRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub task: Option<String>,
    pub top_k: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextTasteProfileNode {
    pub memory_id: u64,
    pub memory_type: String,
    pub node_kind: String,
    pub summary: String,
    pub status: String,
    pub strength: f32,
    pub confidence: f32,
    pub evidence_count: u64,
    pub counter_evidence_count: u64,
    pub requires_verification: bool,
    pub evidence: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextTasteProfileCluster {
    pub cluster_id: String,
    pub label: String,
    pub node_count: usize,
    pub evidence_count: u64,
    pub counter_evidence_count: u64,
    pub aggregate_strength: f32,
    pub aggregate_confidence: f32,
    pub statuses: Vec<String>,
    pub representative_summaries: Vec<String>,
    pub explanation: String,
    pub review_risk: String,
    pub recommended_action: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextTasteProfileResponse {
    pub profile_format: String,
    pub generated_at: DateTime<Utc>,
    pub space_id: String,
    pub task: String,
    pub nodes: Vec<MemoryContextTasteProfileNode>,
    pub clusters: Vec<MemoryContextTasteProfileCluster>,
    pub graph: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextPreferenceReviewItem {
    pub memory_id: u64,
    pub memory_type: String,
    pub summary: String,
    pub status: String,
    pub scope: String,
    pub decay: String,
    pub strength: f32,
    pub confidence: f32,
    pub confidence_level: String,
    pub confidence_explanation: String,
    pub evidence_count: u64,
    pub counter_evidence_count: u64,
    pub requires_verification: bool,
    pub recommended_action: String,
    pub evidence: String,
    pub context_match_level: String,
    pub context_relevance: f32,
    pub context_terms: Vec<String>,
    pub context_explanation: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub actions: Vec<MemoryContextPreferenceReviewAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextPreferenceReviewAction {
    pub action: String,
    pub label: String,
    pub description: String,
    pub effect_preview: String,
    pub approval_note: String,
    pub next_step: String,
    pub endpoint: String,
    pub method: String,
    pub payload_template: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextPreferenceReviewResponse {
    pub review_format: String,
    pub generated_at: DateTime<Utc>,
    pub space_id: String,
    pub task: String,
    pub item_count: usize,
    pub grounded_count: usize,
    pub inferred_count: usize,
    pub rejected_count: usize,
    pub requires_review_count: usize,
    pub items: Vec<MemoryContextPreferenceReviewItem>,
    pub next_actions: Vec<String>,
    pub user_copy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextMergeDiffItem {
    pub field: String,
    pub previous_value: serde_json::Value,
    pub candidate_value: serde_json::Value,
    pub merged_value: serde_json::Value,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextMergeRiskAssessment {
    pub risk_level: String,
    pub recommendation_score: f32,
    pub requires_arbitration: bool,
    pub merge_strategy: String,
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextMergeSuggestion {
    pub memory_id: u64,
    pub memory_type: String,
    pub node_kind: String,
    pub summary: String,
    pub status: String,
    pub evidence_count: u64,
    pub counter_evidence_count: u64,
    pub confidence: f32,
    pub similarity: f32,
    pub reason: String,
    pub risk: MemoryContextMergeRiskAssessment,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diff: Vec<MemoryContextMergeDiffItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextRepairWorkflow {
    pub workflow_format: String,
    pub repair_action: String,
    pub label: String,
    pub status: String,
    pub steps: Vec<String>,
    pub approval_endpoint: String,
    pub approval_payload_template: serde_json::Value,
    pub verification_steps: Vec<String>,
    pub receipt_endpoint: String,
    pub receipt_payload_template: serde_json::Value,
    pub rollback_endpoint: String,
    pub rollback_payload_template: serde_json::Value,
    pub user_copy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextWriteCandidate {
    pub candidate_id: String,
    pub layer: MemoryLayer,
    pub memory_type: String,
    pub content: String,
    pub source: String,
    pub confidence: f32,
    pub freshness: f32,
    pub importance: f32,
    pub reason: String,
    pub requires_approval: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repair_action: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub repair_reason_codes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repair_user_copy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repair_workflow: Option<MemoryContextRepairWorkflow>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supersedes_memory_ids: Vec<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub merge_suggestions: Vec<MemoryContextMergeSuggestion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextProposeRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub task: String,
    pub project: Option<String>,
    pub repo_path: Option<String>,
    pub request_id: Option<String>,
    pub max_candidates: Option<usize>,
    pub correction_text: Option<String>,
    pub correction_of_memory_id: Option<u64>,
    pub taste_feedback: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextProposeResponse {
    pub proposal_format: String,
    pub generated_at: DateTime<Utc>,
    pub space_id: String,
    pub requires_human_approval: bool,
    pub candidates: Vec<MemoryContextWriteCandidate>,
    #[serde(default)]
    pub proposal_history: MemoryContextProposalHistorySummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalMemoryProposeRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub note_text: String,
    #[serde(default)]
    pub note_id: Option<String>,
    #[serde(default)]
    pub patient_id: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub actor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalEntity {
    pub label: String,
    pub text: String,
    pub confidence: f32,
    pub redaction_status: String,
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalMemoryFileBridge {
    pub target_path: String,
    pub redacted_note_summary: String,
    pub entity_table_markdown: String,
    pub approval_required: bool,
    pub next_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalMemoryProposeResponse {
    pub proposal_format: String,
    pub generated_at: DateTime<Utc>,
    pub space_id: String,
    pub note_id: String,
    pub extractor: String,
    pub synthetic_data_only: bool,
    pub no_cloud_mode: bool,
    pub entities: Vec<ClinicalEntity>,
    pub blocked_entities: Vec<ClinicalEntity>,
    pub redacted_note: String,
    pub candidates: Vec<MemoryContextWriteCandidate>,
    pub memory_file_bridge: ClinicalMemoryFileBridge,
    pub approval_endpoint: String,
    pub reason_codes: Vec<String>,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryContextProposalHistorySummary {
    #[serde(default)]
    pub history_format: String,
    #[serde(default)]
    pub endpoint: String,
    #[serde(default)]
    pub method: String,
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub candidate_ids: Vec<String>,
    #[serde(default)]
    pub request_payload: serde_json::Value,
    #[serde(default)]
    pub candidate_payloads: Vec<serde_json::Value>,
    #[serde(default)]
    pub user_copy: String,
    #[serde(default)]
    pub next_step: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayBtwRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub note: String,
    #[serde(default)]
    pub task: Option<String>,
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub max_candidates: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayBtwResponse {
    pub btw_format: String,
    pub generated_at: DateTime<Utc>,
    pub space_id: String,
    pub origin_layer: String,
    pub classification: String,
    pub message: String,
    pub proposal: MemoryContextProposeResponse,
    pub commit_endpoint: String,
    pub trace_group: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayInsightRequest {
    #[serde(default, alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub reflection: Option<String>,
    #[serde(default)]
    pub task: Option<String>,
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub max_candidates: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayInsightResponse {
    pub insight_format: String,
    pub generated_at: DateTime<Utc>,
    pub space_id: String,
    pub origin_layer: String,
    pub insight_type: String,
    pub message: String,
    pub source_evidence: Vec<String>,
    pub proposal: MemoryContextProposeResponse,
    pub commit_endpoint: String,
    pub trace_group: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextCommitRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub candidates: Vec<MemoryContextWriteCandidate>,
    pub approved_candidate_ids: Option<Vec<String>>,
    pub approve_all: Option<bool>,
    pub auto_commit_policy: Option<String>,
    pub request_id: Option<String>,
    pub actor: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextApprovalFollowup {
    pub summary: String,
    pub changed_memory_ids: Vec<u64>,
    pub verify_next_answer: String,
    pub rollback_guidance: String,
    pub user_copy: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextCommitResponse {
    pub committed_count: usize,
    pub skipped_count: usize,
    pub approval_required_count: usize,
    pub blocked_count: usize,
    pub memory_ids: Vec<u64>,
    pub skipped_candidate_ids: Vec<String>,
    pub approval_required_candidate_ids: Vec<String>,
    pub blocked_candidate_ids: Vec<String>,
    pub policy_decisions: Vec<MemoryContextPolicyDecision>,
    pub rollback_endpoint: String,
    pub approval_followup: MemoryContextApprovalFollowup,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextMergeRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub candidate: MemoryContextWriteCandidate,
    pub target_memory_id: u64,
    pub approve_merge: Option<bool>,
    pub request_id: Option<String>,
    pub actor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextMergeBatchItem {
    pub candidate: MemoryContextWriteCandidate,
    pub target_memory_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextMergeBatchRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub merges: Vec<MemoryContextMergeBatchItem>,
    pub approve_all_merges: Option<bool>,
    pub request_id: Option<String>,
    pub actor: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextMergeBatchFailure {
    pub candidate_id: String,
    pub target_memory_id: u64,
    pub error_code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextMergeResponse {
    pub merge_format: String,
    pub merged: bool,
    pub space_id: String,
    pub target_memory_id: u64,
    pub candidate_id: String,
    pub memory_id: u64,
    pub decision_id: String,
    pub actor: String,
    pub reason: String,
    pub risk: MemoryContextMergeRiskAssessment,
    pub rollback_endpoint: String,
    pub diff: Vec<MemoryContextMergeDiffItem>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextMergeBatchResponse {
    pub merge_format: String,
    pub space_id: String,
    pub requested_count: usize,
    pub merged_count: usize,
    pub failed_count: usize,
    pub merged: Vec<MemoryContextMergeResponse>,
    pub failed: Vec<MemoryContextMergeBatchFailure>,
    pub rollback_endpoint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextPolicyDecision {
    pub decision_id: String,
    pub candidate_id: String,
    pub request_id: Option<String>,
    pub policy: String,
    pub decision: String,
    pub risk_level: String,
    pub reason_codes: Vec<String>,
    pub actor: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub candidate: MemoryContextWriteCandidate,
    pub committed_memory_id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merge_target_memory_id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merge_previous_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merge_new_content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub merge_diff: Vec<MemoryContextMergeDiffItem>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub preference_file_rollbacks: Vec<MemoryContextPreferenceFileRollback>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intelligence_audit: Option<MemoryContextPolicyIntelligenceAudit>,
    pub rollback_id: Option<String>,
    pub rollback_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextPolicyIntelligenceAudit {
    pub audit_format: String,
    pub policy_engine: String,
    pub active_layer: String,
    pub bundled_foundation_model_weights: bool,
    pub bundled_general_llm: bool,
    pub optional_local_llm_used: bool,
    pub optional_external_llm_used: bool,
    pub decision_basis: Vec<String>,
    pub user_copy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextPreferenceFileRollback {
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision_id: Option<String>,
    pub rollback_endpoint: String,
    pub rollback_payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningWorkflowReviewRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningWorkflowReviewItem {
    pub memory_id: u64,
    pub memory_type: String,
    pub status: String,
    pub topic: Option<String>,
    pub ability_boundary: Option<String>,
    pub stuck_point: Option<String>,
    pub attempt: Option<String>,
    pub next_executable_step: Option<String>,
    pub concept_relation: Option<String>,
    pub evidence_preview: String,
    pub source: String,
    pub trust: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningWorkflowCoverageItem {
    pub surface: String,
    pub status: String,
    pub count: usize,
    pub description: String,
    pub next_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningWorkflowReviewResponse {
    pub review_format: String,
    pub space_id: String,
    pub status: String,
    pub total_learning_memory_count: usize,
    pub problem_card_count: usize,
    pub learning_loop_count: usize,
    pub concept_link_count: usize,
    pub style_preference_count: usize,
    pub coverage: Vec<LearningWorkflowCoverageItem>,
    pub problem_cards: Vec<LearningWorkflowReviewItem>,
    pub learning_loops: Vec<LearningWorkflowReviewItem>,
    pub concept_links: Vec<LearningWorkflowReviewItem>,
    pub style_preferences: Vec<LearningWorkflowReviewItem>,
    pub proposal_endpoint: String,
    pub approval_endpoint: String,
    pub gateway_endpoint: String,
    pub trace_endpoint_template: String,
    pub memory_file_targets: Vec<String>,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConceptMapReviewRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConceptMapAction {
    pub action: String,
    pub label: String,
    pub description: String,
    pub endpoint: String,
    pub method: String,
    pub payload_template: serde_json::Value,
    pub effect_preview: String,
    pub next_step: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConceptMapEdge {
    pub memory_id: u64,
    pub source_concept: String,
    pub target_concept: String,
    pub relation: String,
    pub direction: String,
    pub status: String,
    pub evidence_preview: String,
    pub actions: Vec<LearningConceptMapAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConceptMapNode {
    pub concept: String,
    pub incoming_count: usize,
    pub outgoing_count: usize,
    pub evidence_memory_ids: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConceptMapReviewResponse {
    pub review_format: String,
    pub space_id: String,
    pub status: String,
    pub node_count: usize,
    pub edge_count: usize,
    pub nodes: Vec<LearningConceptMapNode>,
    pub edges: Vec<LearningConceptMapEdge>,
    pub memory_file_target: String,
    pub proposal_endpoint: String,
    pub review_action_endpoint: String,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextRollbackRequest {
    pub decision_id: String,
    pub reason: Option<String>,
    pub actor: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextRollbackResponse {
    pub rollback_format: String,
    pub rolled_back: bool,
    pub decision_id: String,
    pub rollback_id: String,
    pub memory_id: u64,
    pub rollback_action: String,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub preference_file_rollbacks: Vec<MemoryFileRollbackResponse>,
    pub policy_decision: MemoryContextPolicyDecision,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextBatchRollbackRequest {
    pub decision_ids: Vec<String>,
    pub reason: Option<String>,
    pub actor: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextBatchRollbackFailure {
    pub decision_id: String,
    pub error_code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextBatchRollbackResponse {
    pub rollback_format: String,
    pub requested_count: usize,
    pub rolled_back_count: usize,
    pub failed_count: usize,
    pub rolled_back: Vec<MemoryContextRollbackResponse>,
    pub failed: Vec<MemoryContextBatchRollbackFailure>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedRollbackRequest {
    #[serde(alias = "type")]
    pub rollback_type: String,
    #[serde(default, alias = "id")]
    pub target_id: Option<String>,
    #[serde(default)]
    pub decision_id: Option<String>,
    #[serde(default)]
    pub application_id: Option<String>,
    #[serde(default, alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub revision_id: Option<String>,
    #[serde(default)]
    pub snapshot_id: Option<String>,
    #[serde(default)]
    pub checkpoint_id: Option<String>,
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub actor: Option<String>,
    #[serde(default)]
    pub apply: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedRollbackResponse {
    pub rollback_format: String,
    pub rollback_type: String,
    pub mode: String,
    pub status: String,
    pub rolled_back: bool,
    pub target: serde_json::Value,
    pub preview: serde_json::Value,
    pub receipt: serde_json::Value,
    pub trace: serde_json::Value,
    pub policy_feedback: serde_json::Value,
    pub next_actions: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextTasteGraphEditRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub action: String,
    pub approve_edit: Option<bool>,
    pub source_memory_id: Option<u64>,
    pub target_memory_id: Option<u64>,
    pub edge_memory_id: Option<u64>,
    pub summary: Option<String>,
    pub evidence: Option<String>,
    pub status: Option<String>,
    pub strength: Option<f32>,
    pub confidence: Option<f32>,
    #[serde(default)]
    pub split_summaries: Vec<String>,
    pub actor: Option<String>,
    pub provenance: Option<String>,
    pub causality: Option<String>,
    pub updated_at_lamport: Option<u64>,
    pub updated_by: Option<String>,
    #[serde(default)]
    pub protected_scopes: Vec<MemoryProtectedScopeInput>,
    #[serde(default)]
    pub override_protection: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextTasteGraphEditResponse {
    pub edit_format: String,
    pub applied: bool,
    pub apply_allowed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub apply_blocked_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub protected_matches: Vec<MemoryProtectedScopeMatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_receipt_preview: Option<MemoryTransformJobReceiptPreview>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lifecycle_contract: Option<MemoryLifecycleContract>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub why_surfaced_contract: Option<MemoryWhySurfacedContract>,
    pub action: String,
    pub space_id: String,
    pub created_memory_ids: Vec<u64>,
    pub affected_memory_ids: Vec<u64>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextTasteGraphCorrectionRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub memory_id: Option<u64>,
    #[serde(default)]
    pub source_memory_id: Option<u64>,
    #[serde(default)]
    pub edge_memory_id: Option<u64>,
    pub correction_text: String,
    #[serde(default)]
    pub approve_edit: Option<bool>,
    #[serde(default)]
    pub actor: Option<String>,
    #[serde(default)]
    pub split_summaries: Vec<String>,
    #[serde(default)]
    pub protected_scopes: Vec<MemoryProtectedScopeInput>,
    #[serde(default)]
    pub override_protection: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextTasteGraphCorrectionResponse {
    pub correction_format: String,
    pub applied: bool,
    pub apply_allowed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub apply_blocked_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub protected_matches: Vec<MemoryProtectedScopeMatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_receipt_preview: Option<MemoryTransformJobReceiptPreview>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lifecycle_contract: Option<MemoryLifecycleContract>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub why_surfaced_contract: Option<MemoryWhySurfacedContract>,
    pub action: String,
    pub space_id: String,
    pub target_memory_id: u64,
    pub target_kind: String,
    pub correction_text: String,
    pub user_copy: String,
    pub edit_request: MemoryContextTasteGraphEditRequest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edit_result: Option<MemoryContextTasteGraphEditResponse>,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextTasteArbitrationCheckRequest {
    #[serde(default, alias = "conversation_id")]
    pub space_id: Option<String>,
    pub target_summary: Option<String>,
    pub candidate_summary: Option<String>,
    pub target_node: Option<serde_json::Value>,
    pub candidate_node: Option<serde_json::Value>,
    #[serde(default)]
    pub explicit_provider_approval: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextTasteArbitrationCheckResponse {
    pub arbitration_format: String,
    pub provider: String,
    pub model: Option<String>,
    pub status: String,
    pub used_external_llm: bool,
    pub check_status: String,
    pub result: serde_json::Value,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextTasteGraphProvenanceRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub edge_memory_id: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextTasteGraphProvenanceEntry {
    pub at: String,
    pub actor: String,
    pub action: String,
    pub source_memory_id: Option<u64>,
    pub target_memory_id: Option<u64>,
    pub summary: String,
    pub evidence: String,
    pub provenance: String,
    pub causality: String,
    pub status: String,
    pub version: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextTasteGraphProvenanceResponse {
    pub provenance_format: String,
    pub space_id: String,
    pub edge_memory_id: u64,
    pub current: MemoryContextTasteGraphProvenanceEntry,
    pub history: Vec<MemoryContextTasteGraphProvenanceEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextPolicyAuditRequest {
    pub decision_id: Option<String>,
    pub request_id: Option<String>,
    pub decision: Option<String>,
    pub candidate_id: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextPolicyAuditResponse {
    pub audit_format: String,
    pub generated_at: DateTime<Utc>,
    pub items: Vec<MemoryContextPolicyDecision>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextRepairHistoryRequest {
    pub decision_id: Option<String>,
    pub request_id: Option<String>,
    pub candidate_id: Option<String>,
    #[serde(default)]
    pub include_blocked: Option<bool>,
    #[serde(default)]
    pub include_rolled_back: Option<bool>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextRepairHistoryItem {
    pub decision_id: String,
    pub candidate_id: String,
    pub request_id: Option<String>,
    pub status: String,
    pub decision: String,
    pub risk_level: String,
    pub actor: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub memory_type: String,
    pub source: String,
    pub summary: String,
    pub committed_memory_id: Option<u64>,
    pub rollback_id: Option<String>,
    pub rollback_reason: Option<String>,
    pub rollback_available: bool,
    pub rollback_endpoint: String,
    pub receipt_endpoint: String,
    pub receipt_payload: serde_json::Value,
    pub receipt_note: String,
    pub repair_action: Option<String>,
    pub repair_user_copy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repair_workflow: Option<MemoryContextRepairWorkflow>,
    pub next_action: String,
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextRepairHistoryResponse {
    pub history_format: String,
    pub generated_at: DateTime<Utc>,
    pub total_count: usize,
    pub committed_count: usize,
    pub rolled_back_count: usize,
    pub approval_required_count: usize,
    pub blocked_count: usize,
    pub items: Vec<MemoryContextRepairHistoryItem>,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextDecideRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub task: String,
    pub project: Option<String>,
    pub repo_path: Option<String>,
    pub memory_mode: Option<String>,
    pub max_latency_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recall_every_turns: Option<u64>,
    #[serde(default)]
    pub user_id: Option<String>,
    #[serde(default, alias = "client_id")]
    pub client_installation_id: Option<String>,
    #[serde(default)]
    pub task_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextDecideResponse {
    pub memory_mode: String,
    pub memory_status: String,
    pub decision: String,
    pub reason: String,
    pub cache_available: bool,
    pub max_latency_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recommended_cadence: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drift_warning: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallPolicyReviewRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub task: Option<String>,
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub repo_path: Option<String>,
    #[serde(default)]
    pub memory_mode: Option<String>,
    #[serde(default)]
    pub max_latency_ms: Option<u64>,
    #[serde(default)]
    pub user_id: Option<String>,
    #[serde(default, alias = "client_id")]
    pub client_installation_id: Option<String>,
    #[serde(default)]
    pub task_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallPolicyReviewSetting {
    pub scope_type: String,
    pub scope_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recall_every_turns: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_mode: Option<String>,
    pub matched: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precedence: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RecallPolicyTuningSignals {
    pub signal_format: String,
    pub cached_turns_count: u64,
    pub correction_count: u64,
    pub failure_memory_count: u64,
    pub task_failure_signal_count: u64,
    pub task_failure_risk_level: String,
    pub task_failure_terms: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correction_ratio_percent: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recommended_cadence: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recommended_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub personalized_recommendation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recommendation_explanation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drift_warning: Option<String>,
    pub sensitive_task_detected: bool,
    pub explicit_policy_matched: bool,
    pub matched_policy_count: usize,
    pub personalized_policy_matched: bool,
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallPolicyReviewAction {
    pub action: String,
    pub label: String,
    pub description: String,
    pub effect_preview: String,
    pub approval_note: String,
    pub next_step: String,
    pub endpoint: String,
    pub method: String,
    pub payload_template: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallPolicyReviewResponse {
    pub review_format: String,
    pub generated_at: DateTime<Utc>,
    pub space_id: String,
    pub task: String,
    pub task_type: String,
    pub sensitive_task: bool,
    pub effective_mode: String,
    pub effective_recall_every_turns: u64,
    pub decision: MemoryContextDecideResponse,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recommended_cadence: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drift_warning: Option<String>,
    pub tuning_signals: RecallPolicyTuningSignals,
    pub matched_settings: Vec<RecallPolicyReviewSetting>,
    pub sensitive_guidance: Vec<String>,
    pub next_actions: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub actions: Vec<RecallPolicyReviewAction>,
    pub suggested_commands: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextEvidence {
    pub memory_id: u64,
    pub layer: MemoryLayer,
    pub memory_type: String,
    pub score: f32,
    pub source: String,
    pub created_at: DateTime<Utc>,
    pub reason: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTraceItem {
    pub memory_id: u64,
    pub memory_type: String,
    pub layer: MemoryLayer,
    pub reason: String,
    pub used_in_context: bool,
    pub ignored_reason: Option<String>,
    pub evidence: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub freshness: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub importance: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_verified_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub captured: Option<TraceCapturedContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceCapturedContext {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_app: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub redaction_status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_used: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approved_memory_file_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approved_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deleted_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lifecycle: Vec<AmbientCaptureLifecycleEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnswerTraceSummary {
    pub summary_format: String,
    pub request_id: String,
    pub trace_url: String,
    pub recall: TraceRecallSummary,
    pub counts: TraceEvidenceCounts,
    pub groups: Vec<TraceEvidenceGroup>,
    pub correction_actions: Vec<TraceCorrectionAction>,
    #[serde(default)]
    pub repair_history: TraceRepairHistorySummary,
    #[serde(default)]
    pub footer: serde_json::Value,
    #[serde(default)]
    pub confidence_explanation: serde_json::Value,
    #[serde(default)]
    pub evidence_comparison: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TraceRepairHistorySummary {
    #[serde(default)]
    pub history_format: String,
    #[serde(default)]
    pub endpoint: String,
    #[serde(default)]
    pub method: String,
    #[serde(default)]
    pub request_id: String,
    #[serde(default)]
    pub status_scope: Vec<String>,
    #[serde(default)]
    pub payload_template: serde_json::Value,
    #[serde(default)]
    pub user_copy: String,
    #[serde(default)]
    pub next_step: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRecallSummary {
    pub mode: String,
    pub outcome: String,
    pub source: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TraceEvidenceCounts {
    pub memories_used: usize,
    pub files_used: usize,
    pub preferences_used: usize,
    pub ignored: usize,
    pub stale: usize,
    pub conflicts: usize,
    pub policy_skipped: usize,
    pub cached: usize,
    pub captured: usize,
    pub proposed_updates: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEvidenceGroup {
    pub group: String,
    pub label: String,
    pub items: Vec<TraceEvidenceSummaryItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEvidenceSummaryItem {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub section: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub redaction_status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub captured: Option<TraceCapturedContext>,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence_explanation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub why_surfaced: Option<serde_json::Value>,
    pub used_in_context: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ignored_reason: Option<String>,
    pub evidence: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recommendation: Option<TraceRepairRecommendation>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub actions: Vec<TraceCorrectionAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRepairRecommendation {
    pub recommended_action: String,
    pub risk: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<u8>,
    pub confidence: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence_explanation: Option<String>,
    pub why: String,
    pub next_step: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reason_codes: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supporting_signals: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub caveats: Vec<String>,
    #[serde(default)]
    pub evidence_comparison: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceCorrectionAction {
    pub action: String,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effect_preview: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_note: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_step: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    #[serde(default)]
    pub payload_template: serde_json::Value,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceActionRequest {
    #[serde(default, alias = "conversation_id")]
    pub space_id: Option<String>,
    pub request_id: String,
    pub action: String,
    #[serde(default)]
    pub memory_id: Option<u64>,
    #[serde(default)]
    pub replacement_memory_id: Option<u64>,
    #[serde(default)]
    pub superseded_by: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub correction_text: Option<String>,
    #[serde(default)]
    pub apply_edit: Option<bool>,
    #[serde(default)]
    pub taste_feedback: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub actor: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct TraceActionResponse {
    pub action_format: String,
    pub request_id: String,
    pub action: String,
    pub applied: bool,
    pub outcome: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub review_result: Option<MemoryContextReviewActionResponse>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proposals: Option<MemoryContextProposeResponse>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edit_file: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceActionBatchRequest {
    pub actions: Vec<TraceActionRequest>,
    #[serde(default)]
    pub continue_on_error: Option<bool>,
    #[serde(default)]
    pub actor: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct TraceActionBatchSuccess {
    pub index: usize,
    pub request_id: String,
    pub action: String,
    pub response: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct TraceActionBatchFailure {
    pub index: usize,
    pub request_id: String,
    pub action: String,
    pub status: u16,
    pub error_code: String,
    pub message: String,
    pub retryable: bool,
    pub recovery_hint: String,
    pub retry_action: TraceActionRequest,
}

#[derive(Debug, Serialize)]
pub struct TraceActionBatchResponse {
    pub action_format: String,
    pub requested_count: usize,
    pub succeeded_count: usize,
    pub failed_count: usize,
    pub retryable_count: usize,
    pub manual_repair_count: usize,
    pub stopped_on_error: bool,
    pub results: Vec<TraceActionBatchSuccess>,
    pub failures: Vec<TraceActionBatchFailure>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_batch: Option<TraceActionBatchRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retryable_batch: Option<TraceActionBatchRequest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileEvidence {
    pub path: String,
    pub section: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stale: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    pub reason: String,
    pub used_in_context: bool,
    pub confidence: String,
    pub evidence: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub captured: Option<TraceCapturedContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileWriteRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub path: String,
    pub content: String,
    #[serde(default)]
    pub confidence: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileReadRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileSearchRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub query: String,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileListRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileDeleteRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub path: String,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub actor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileStaleCheckRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub paths: Option<Vec<String>>,
    #[serde(default)]
    pub max_age_days: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileReindexRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub paths: Option<Vec<String>>,
    #[serde(default)]
    pub include_sections: Option<bool>,
    #[serde(default)]
    pub max_age_days: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileBrowseRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub max_age_days: Option<i64>,
    #[serde(default)]
    pub include_custom: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileInitProfileRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub overwrite: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileRollbackRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub path: String,
    #[serde(default)]
    pub revision_id: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub actor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileRollbackPreviewRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub path: String,
    #[serde(default)]
    pub revision_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileSectionEditRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub path: String,
    #[serde(default)]
    pub section_id: Option<String>,
    #[serde(default)]
    pub heading: Option<String>,
    pub replacement_text: String,
    #[serde(default)]
    pub expected_current_text: Option<String>,
    #[serde(default)]
    pub apply: Option<bool>,
    #[serde(default)]
    pub force: Option<bool>,
    #[serde(default)]
    pub actor: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub protected_scopes: Vec<MemoryProtectedScopeInput>,
    #[serde(default)]
    pub override_protection: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileEditRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub path: String,
    pub content: String,
    #[serde(default)]
    pub expected_current_text: Option<String>,
    #[serde(default)]
    pub apply: Option<bool>,
    #[serde(default)]
    pub force: Option<bool>,
    #[serde(default)]
    pub actor: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub protected_scopes: Vec<MemoryProtectedScopeInput>,
    #[serde(default)]
    pub override_protection: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileDiffApprovalMetadata {
    pub approval_format: String,
    pub proposal_id: String,
    pub approval_required: bool,
    pub approval_state: String,
    pub approval_endpoint: String,
    pub approval_payload: serde_json::Value,
    pub approval_reason_codes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receipt_endpoint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receipt_payload: Option<serde_json::Value>,
    pub user_message: String,
    pub next_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileDiffProposalApplyRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub proposal_id: String,
    #[serde(default)]
    pub force: Option<bool>,
    #[serde(default)]
    pub actor: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub protected_scopes: Vec<MemoryProtectedScopeInput>,
    #[serde(default)]
    pub override_protection: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileDiffProposalReceiptRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub proposal_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileDiffProposalReceiptResponse {
    pub receipt_format: String,
    pub proposal_id: String,
    pub space_id: String,
    pub path: String,
    pub status: String,
    pub applied: bool,
    pub created_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<DateTime<Utc>>,
    pub approval_endpoint: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub applied_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub applied_by: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollback_revision_id: Option<String>,
    pub audit_endpoint: String,
    pub audit_payload: serde_json::Value,
    pub user_message: String,
    pub next_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileSectionMergeRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub path: String,
    #[serde(default)]
    pub section_id: Option<String>,
    #[serde(default)]
    pub heading: Option<String>,
    pub base_text: String,
    pub proposed_text: String,
    #[serde(default)]
    pub merged_text: Option<String>,
    #[serde(default)]
    pub resolution_strategy: Option<String>,
    #[serde(default)]
    pub apply: Option<bool>,
    #[serde(default)]
    pub actor: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub protected_scopes: Vec<MemoryProtectedScopeInput>,
    #[serde(default)]
    pub override_protection: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileRecord {
    pub path: String,
    pub bytes: u64,
    pub updated_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileWriteResponse {
    pub file_format: String,
    pub space_id: String,
    pub path: String,
    pub bytes: u64,
    pub updated_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_revision_id: Option<String>,
    pub rollback_endpoint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileReadResponse {
    pub file_format: String,
    pub space_id: String,
    pub file: MemoryFileRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileListResponse {
    pub file_format: String,
    pub space_id: String,
    pub files: Vec<MemoryFileRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileDeleteResponse {
    pub file_format: String,
    pub space_id: String,
    pub path: String,
    pub deleted: bool,
    pub revision_id: String,
    pub rollback_endpoint: String,
    pub updated_at: DateTime<Utc>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileStaleCheckItem {
    pub path: String,
    pub exists: bool,
    pub stale: bool,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub age_days: Option<i64>,
    pub recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileStaleCheckResponse {
    pub file_format: String,
    pub space_id: String,
    pub max_age_days: i64,
    pub checked_count: usize,
    pub stale_count: usize,
    pub missing_count: usize,
    pub items: Vec<MemoryFileStaleCheckItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileReindexItem {
    pub path: String,
    pub exists: bool,
    pub indexed: bool,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub age_days: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub headings: Vec<String>,
    pub stale: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileReindexResponse {
    pub file_format: String,
    pub generated_at: DateTime<Utc>,
    pub space_id: String,
    pub index_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector_index: Option<MemoryFileVectorIndexMetadata>,
    pub max_age_days: i64,
    pub checked_count: usize,
    pub indexed_count: usize,
    pub missing_count: usize,
    pub stale_count: usize,
    pub items: Vec<MemoryFileReindexItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileVectorIndexMetadata {
    pub vector_index_format: String,
    pub backend: String,
    pub status: String,
    pub index_path: String,
    pub mapping_path: String,
    pub indexed_count: usize,
    pub dims: usize,
    pub compressed_bytes_estimate: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stable_ids: Vec<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reason_codes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileBrowseItem {
    pub path: String,
    pub role: String,
    pub canonical: bool,
    pub exists: bool,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub age_days: Option<i64>,
    pub stale: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub headings: Vec<String>,
    pub recommended_action: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollback_revision_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileReviewAction {
    pub action: String,
    pub label: String,
    pub description: String,
    pub endpoint: String,
    pub method: String,
    pub payload_template: serde_json::Value,
    pub review_scope: String,
    pub effect_preview: String,
    pub next_step: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileVaultConflictReview {
    pub review_format: String,
    pub status: String,
    pub conflict_count: usize,
    pub conflict_paths: Vec<String>,
    #[serde(default)]
    pub vector_mapping_status: String,
    #[serde(default)]
    pub vector_mapping_conflict_count: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vector_mapping_conflict_paths: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector_mapping_path: Option<String>,
    #[serde(default)]
    pub indexed_vector_count: usize,
    pub user_copy: String,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileBrowseReviewSummary {
    pub review_format: String,
    pub ready_count: usize,
    pub needs_review_count: usize,
    pub missing_count: usize,
    pub stale_count: usize,
    pub custom_count: usize,
    pub rollback_available_count: usize,
    pub multi_review_available: bool,
    pub user_copy: String,
    pub review_actions: Vec<MemoryFileReviewAction>,
    pub vault_conflict_review: MemoryFileVaultConflictReview,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileBrowseResponse {
    pub file_format: String,
    pub generated_at: DateTime<Utc>,
    pub space_id: String,
    pub max_age_days: i64,
    pub total_count: usize,
    pub canonical_count: usize,
    pub present_count: usize,
    pub missing_count: usize,
    pub stale_count: usize,
    pub custom_count: usize,
    pub files: Vec<MemoryFileBrowseItem>,
    pub next_actions: Vec<String>,
    pub review_summary: MemoryFileBrowseReviewSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileInitProfileFile {
    pub path: String,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<MemoryFileWriteResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileInitProfileResponse {
    pub file_format: String,
    pub space_id: String,
    pub created_count: usize,
    pub overwritten_count: usize,
    pub skipped_count: usize,
    pub files: Vec<MemoryFileInitProfileFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileSearchResponse {
    pub file_format: String,
    pub space_id: String,
    pub query: String,
    pub matches: Vec<MemoryFileEvidence>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileRollbackResponse {
    pub file_format: String,
    pub space_id: String,
    pub path: String,
    pub rolled_back: bool,
    pub revision_id: String,
    pub bytes: u64,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileDiffLine {
    pub kind: String,
    pub line: usize,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileVisualDiff {
    pub diff_format: String,
    pub view: String,
    pub left_label: String,
    pub right_label: String,
    pub added_lines: usize,
    pub removed_lines: usize,
    pub changed_lines: usize,
    pub unchanged_lines: usize,
    pub hunks: Vec<MemoryFileDiffHunk>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileDiffHunk {
    pub hunk_id: String,
    pub left_start_line: usize,
    pub right_start_line: usize,
    pub left_line_count: usize,
    pub right_line_count: usize,
    pub status: String,
    pub rows: Vec<MemoryFileDiffRow>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileDiffRow {
    pub row_id: String,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub left_line: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub right_line: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub left_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub right_text: Option<String>,
    pub inline_changed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileSectionDiff {
    pub section_id: String,
    pub heading: String,
    pub current_start_line: usize,
    pub rollback_start_line: usize,
    pub current_text: String,
    pub rollback_text: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileDiffHistoryMetadata {
    pub history_format: String,
    pub scope: String,
    pub diff_kind: String,
    pub space_id: String,
    pub path: String,
    pub applied: bool,
    pub apply_endpoint: String,
    pub apply_payload: serde_json::Value,
    pub rollback_available: bool,
    pub rollback_endpoint: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollback_payload: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollback_revision_id: Option<String>,
    pub audit_endpoint: String,
    pub audit_payload: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receipt_endpoint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receipt_payload: Option<serde_json::Value>,
    pub user_message: String,
    pub next_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileRollbackConflict {
    pub detected: bool,
    pub reason: String,
    pub current_matches_revision_replacement: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileEditConflict {
    pub detected: bool,
    pub reason: String,
    pub expected_matches_current: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileMergeConflict {
    pub detected: bool,
    pub reason: String,
    pub base_matches_current: bool,
    pub proposed_matches_current: bool,
    pub requires_manual_resolution: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileRollbackPreviewResponse {
    pub file_format: String,
    pub space_id: String,
    pub path: String,
    pub revision_id: String,
    pub revision_created_at: DateTime<Utc>,
    pub current_content: String,
    pub rollback_content: String,
    pub diff: Vec<MemoryFileDiffLine>,
    pub visual_diff: MemoryFileVisualDiff,
    pub sections: Vec<MemoryFileSectionDiff>,
    pub conflict: MemoryFileRollbackConflict,
    pub apply_endpoint: String,
    pub diff_history: MemoryFileDiffHistoryMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileSectionEditResponse {
    pub file_format: String,
    pub space_id: String,
    pub path: String,
    pub section_id: String,
    pub heading: String,
    pub current_content: String,
    pub proposed_content: String,
    pub diff: Vec<MemoryFileDiffLine>,
    pub visual_diff: MemoryFileVisualDiff,
    pub sections: Vec<MemoryFileSectionDiff>,
    pub conflict: MemoryFileEditConflict,
    pub applied: bool,
    pub apply_endpoint: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diff_approval: Option<MemoryFileDiffApprovalMetadata>,
    pub diff_history: MemoryFileDiffHistoryMetadata,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub protected_matches: Vec<MemoryProtectedScopeMatch>,
    pub apply_allowed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub apply_blocked_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_receipt_preview: Option<MemoryTransformJobReceiptPreview>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lifecycle_contract: Option<MemoryLifecycleContract>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub why_surfaced_contract: Option<MemoryWhySurfacedContract>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub write_result: Option<MemoryFileWriteResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileEditResponse {
    pub file_format: String,
    pub space_id: String,
    pub path: String,
    pub current_content: String,
    pub proposed_content: String,
    pub diff: Vec<MemoryFileDiffLine>,
    pub visual_diff: MemoryFileVisualDiff,
    pub sections: Vec<MemoryFileSectionDiff>,
    pub conflict: MemoryFileEditConflict,
    pub applied: bool,
    pub apply_endpoint: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diff_approval: Option<MemoryFileDiffApprovalMetadata>,
    pub diff_history: MemoryFileDiffHistoryMetadata,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub protected_matches: Vec<MemoryProtectedScopeMatch>,
    pub apply_allowed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub apply_blocked_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_receipt_preview: Option<MemoryTransformJobReceiptPreview>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lifecycle_contract: Option<MemoryLifecycleContract>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub why_surfaced_contract: Option<MemoryWhySurfacedContract>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub write_result: Option<MemoryFileWriteResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFileSectionMergeResponse {
    pub file_format: String,
    pub space_id: String,
    pub path: String,
    pub section_id: String,
    pub heading: String,
    pub base_text: String,
    pub current_text: String,
    pub proposed_text: String,
    pub merged_text: String,
    pub resolution_strategy: String,
    pub conflict: MemoryFileMergeConflict,
    pub current_content: String,
    pub merged_content: String,
    pub diff: Vec<MemoryFileDiffLine>,
    pub visual_diff: MemoryFileVisualDiff,
    pub sections: Vec<MemoryFileSectionDiff>,
    pub applied: bool,
    pub apply_endpoint: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diff_approval: Option<MemoryFileDiffApprovalMetadata>,
    pub diff_history: MemoryFileDiffHistoryMetadata,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub protected_matches: Vec<MemoryProtectedScopeMatch>,
    pub apply_allowed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub apply_blocked_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_receipt_preview: Option<MemoryTransformJobReceiptPreview>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lifecycle_contract: Option<MemoryLifecycleContract>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub why_surfaced_contract: Option<MemoryWhySurfacedContract>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub write_result: Option<MemoryFileWriteResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCaptureSourceUpsertRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub source_id: String,
    pub source_type: String,
    pub display_name: Option<String>,
    pub enabled: bool,
    #[serde(default)]
    pub retention_days: Option<u32>,
    #[serde(default)]
    pub redaction_policy: Option<String>,
    #[serde(default)]
    pub allowed_scopes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCaptureEventRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub source_id: String,
    pub source_app: String,
    pub title: Option<String>,
    pub url: Option<String>,
    pub path: Option<String>,
    pub thread_id: Option<String>,
    pub timestamp: Option<DateTime<Utc>>,
    pub content: String,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub redaction_status: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCaptureRedactionCheck {
    pub check: String,
    pub redacted: bool,
    pub count: usize,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCaptureLifecycleEntry {
    pub action: String,
    pub actor: String,
    pub reason: String,
    pub at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_file_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCaptureListRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub source_id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCaptureApproveRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub event_ids: Vec<u64>,
    #[serde(default)]
    pub target_path: Option<String>,
    #[serde(default)]
    pub summary_override: Option<String>,
    #[serde(default)]
    pub actor: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCaptureProposalApplyRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub proposal_id: String,
    #[serde(default)]
    pub force: Option<bool>,
    #[serde(default)]
    pub actor: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCaptureProposalReceiptRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub proposal_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCapturePreviewRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub event_ids: Vec<u64>,
    #[serde(default)]
    pub target_path: Option<String>,
    #[serde(default)]
    pub summary_override: Option<String>,
    #[serde(default)]
    pub actor: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCaptureDeleteRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub event_ids: Vec<u64>,
    #[serde(default)]
    pub actor: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCaptureRetentionPruneRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub source_id: Option<String>,
    #[serde(default)]
    pub dry_run: Option<bool>,
    #[serde(default)]
    pub actor: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCaptureSourceRecord {
    pub source_format: String,
    pub source_id: String,
    pub source_type: String,
    pub display_name: Option<String>,
    pub enabled: bool,
    pub retention_days: Option<u32>,
    pub redaction_policy: Option<String>,
    pub allowed_scopes: Vec<String>,
    pub updated_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub durable_memory_id: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCaptureConnectorStatusRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCaptureConnectorStatusItem {
    pub connector_id: String,
    pub display_name: String,
    pub source_type: String,
    pub status: String,
    pub configured_source_count: usize,
    pub enabled_source_count: usize,
    pub buffered_event_count: usize,
    pub approved_event_count: usize,
    pub deleted_event_count: usize,
    pub retention_policy: String,
    pub privacy_policy: String,
    pub redaction_policy: String,
    pub approval_required: bool,
    pub trace_provenance: bool,
    pub memory_file_bridge: bool,
    pub risk_level: String,
    pub setup_requirements: Vec<String>,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCaptureConnectorStatusResponse {
    pub catalog_format: String,
    pub space_id: String,
    pub default_capture_mode: String,
    pub durable_write_policy: String,
    pub default_retention_policy: String,
    pub privacy_contract: Vec<String>,
    pub endpoints: Vec<String>,
    pub connectors: Vec<AmbientCaptureConnectorStatusItem>,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCaptureEventRecord {
    pub event_format: String,
    pub event_id: String,
    pub source_id: String,
    pub status: String,
    pub captured: TraceCapturedContext,
    pub content: String,
    pub created_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub redaction_checks: Vec<AmbientCaptureRedactionCheck>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lifecycle: Vec<AmbientCaptureLifecycleEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approved_memory_file_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approved_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deleted_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub durable_memory_id: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCaptureSourceResponse {
    pub capture_format: String,
    pub space_id: String,
    pub source: AmbientCaptureSourceRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCaptureEventResponse {
    pub capture_format: String,
    pub space_id: String,
    pub event: AmbientCaptureEventRecord,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub redaction_checks: Vec<AmbientCaptureRedactionCheck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCaptureListResponse {
    pub capture_format: String,
    pub space_id: String,
    pub sources: Vec<AmbientCaptureSourceRecord>,
    pub events: Vec<AmbientCaptureEventRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCaptureApproveResponse {
    pub capture_format: String,
    pub space_id: String,
    pub approved_count: usize,
    pub target_path: String,
    pub file: MemoryFileWriteResponse,
    pub events: Vec<AmbientCaptureEventRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCaptureProposalReceiptResponse {
    pub receipt_format: String,
    pub proposal_id: String,
    pub space_id: String,
    pub status: String,
    pub target_path: String,
    pub event_ids: Vec<u64>,
    pub created_at: DateTime<Utc>,
    pub approval_endpoint: String,
    pub approval_payload: serde_json::Value,
    pub receipt_endpoint: String,
    pub receipt_payload: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub applied_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub applied_by: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_reason: Option<String>,
    pub approved_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<MemoryFileWriteResponse>,
    pub events: Vec<AmbientCaptureEventRecord>,
    pub user_message: String,
    pub next_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCaptureDiffLine {
    pub kind: String,
    pub line: usize,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCapturePreviewResponse {
    pub capture_format: String,
    pub space_id: String,
    pub approval_required: bool,
    pub event_count: usize,
    pub target_path: String,
    pub current_content: String,
    pub proposed_content: String,
    pub diff: Vec<AmbientCaptureDiffLine>,
    pub redaction_checks: Vec<AmbientCaptureRedactionCheck>,
    pub events: Vec<AmbientCaptureEventRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval: Option<MemoryFileDiffApprovalMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCaptureDeleteResponse {
    pub capture_format: String,
    pub space_id: String,
    pub deleted_count: usize,
    pub events: Vec<AmbientCaptureEventRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCaptureRetentionPruneResponse {
    pub capture_format: String,
    pub space_id: String,
    pub dry_run: bool,
    pub scanned_count: usize,
    pub expired_count: usize,
    pub deleted_count: usize,
    pub retention_policy: Vec<String>,
    pub events: Vec<AmbientCaptureEventRecord>,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileReviewItem {
    pub memory_id: u64,
    pub memory_type: String,
    pub dimension: String,
    pub summary: String,
    pub evidence: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_modalities: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub modality_signals: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub modality_signal_records: Vec<serde_json::Value>,
    pub source: String,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileEvidenceInput {
    pub modality: String,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub uri: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub transcript: Option<String>,
    #[serde(default)]
    pub ocr_text: Option<String>,
    #[serde(default)]
    pub caption: Option<String>,
    #[serde(default)]
    pub alt_text: Option<String>,
    #[serde(default)]
    pub observed_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalProfileExtractRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub source_id: Option<String>,
    #[serde(default)]
    pub subject: Option<String>,
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default)]
    pub profile_kind: Option<String>,
    #[serde(default)]
    pub observations: String,
    #[serde(default)]
    pub dimensions: Vec<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub trust: Option<f32>,
    #[serde(default)]
    pub dry_run: Option<bool>,
    #[serde(default)]
    pub evidence_items: Vec<ProfileEvidenceInput>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalProfileExtractResponse {
    pub profile_format: String,
    pub space_id: String,
    pub source_id: String,
    pub subject: String,
    pub domain: String,
    pub profile_kind: String,
    pub dry_run: bool,
    pub record_count: usize,
    pub memory_ids: Vec<u64>,
    pub records: Vec<ProfileReviewItem>,
    pub review_endpoint: String,
    pub answer_endpoint: String,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalProfileReviewRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub source_id: Option<String>,
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default)]
    pub profile_kind: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalProfileReviewResponse {
    pub review_format: String,
    pub space_id: String,
    pub source_id: Option<String>,
    pub domain: Option<String>,
    pub profile_kind: Option<String>,
    pub total_count: usize,
    pub universal_profiles: Vec<ProfileReviewItem>,
    pub answer_endpoint: String,
    pub correction_guidance: String,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferenceTuningCapabilitiesResponse {
    pub capabilities_format: String,
    pub status: String,
    pub profile_type: String,
    pub domain: String,
    pub profile_kind: String,
    pub extract_endpoint: String,
    pub enhance_endpoint: String,
    pub review_endpoint: String,
    pub correct_endpoint: String,
    pub answer_endpoint: String,
    pub supported_input_modalities: Vec<String>,
    pub supported_signal_kinds: Vec<String>,
    pub supported_output_forms: Vec<String>,
    pub default_dimensions: Vec<String>,
    pub guardrails: Vec<String>,
    pub known_limits: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTransformCapabilitiesResponse {
    pub capabilities_format: String,
    pub status: String,
    pub reference_sources: Vec<String>,
    pub preview_endpoint: String,
    pub answer_endpoint: String,
    pub trace_endpoint_template: String,
    pub supported_mutation_surfaces: Vec<String>,
    pub diff_fields: Vec<String>,
    pub protected_scope_selectors: Vec<String>,
    pub bounded_job_receipt_fields: Vec<String>,
    pub lifecycle_flags: Vec<String>,
    pub why_surfaced_factors: Vec<String>,
    pub user_actions: Vec<String>,
    pub guardrails: Vec<String>,
    pub onboarding_health_checks: Vec<String>,
    pub implemented_now: Vec<String>,
    pub remaining_work: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProtectedScopeInput {
    pub scope_type: String,
    pub selector: String,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub action: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTransformPreviewRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub mutation_intent: Option<String>,
    #[serde(default)]
    pub target_kind: Option<String>,
    #[serde(default)]
    pub target_id: Option<String>,
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default)]
    pub source_refs: Vec<String>,
    #[serde(default)]
    pub before_record: serde_json::Value,
    #[serde(default)]
    pub after_record: serde_json::Value,
    #[serde(default)]
    pub changed_fields: Vec<String>,
    #[serde(default)]
    pub protected_scopes: Vec<MemoryProtectedScopeInput>,
    #[serde(default)]
    pub override_protection: Option<bool>,
    #[serde(default)]
    pub job_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProtectedScopeMatch {
    pub scope_type: String,
    pub selector: String,
    pub matched: bool,
    pub reason: String,
    pub action: String,
    pub matched_fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTransformDiffField {
    pub field: String,
    pub before_preview: String,
    pub after_preview: String,
    pub change_kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTransformJobReceiptPreview {
    pub receipt_format: String,
    pub job_id: String,
    pub phase: String,
    pub accepted_count: usize,
    pub dropped_count: usize,
    pub skipped_protected_count: usize,
    pub timeout_policy: String,
    pub backpressure_policy: String,
    pub recursion_guard: String,
    pub partial_failure_records: Vec<serde_json::Value>,
    pub resume_token: String,
    pub cancel_endpoint: String,
    pub rollback_plan: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLifecycleContract {
    pub lifecycle_format: String,
    pub flags: Vec<String>,
    pub default_flags: Vec<String>,
    pub hold_endpoint: String,
    pub archive_endpoint: String,
    pub repair_endpoint: String,
    pub trace_endpoint_template: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryWhySurfacedContract {
    pub explanation_format: String,
    pub factors: Vec<String>,
    pub citation_requirement: String,
    pub trace_requirement: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTransformPreviewResponse {
    pub preview_format: String,
    pub space_id: String,
    pub dry_run: bool,
    pub mutation_intent: String,
    pub target_kind: String,
    pub target_id: Option<String>,
    pub domain: String,
    pub source_refs: Vec<String>,
    pub before_record: serde_json::Value,
    pub after_record: serde_json::Value,
    pub changed_fields: Vec<String>,
    pub diff_summary: String,
    pub diff_fields: Vec<MemoryTransformDiffField>,
    pub protected_matches: Vec<MemoryProtectedScopeMatch>,
    pub apply_allowed: bool,
    pub apply_blocked_reason: Option<String>,
    pub affected_answer_surfaces: Vec<String>,
    pub candidate_repair: serde_json::Value,
    pub job_receipt_preview: MemoryTransformJobReceiptPreview,
    pub lifecycle_contract: MemoryLifecycleContract,
    pub why_surfaced_contract: MemoryWhySurfacedContract,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WritingStyleProfileExtractRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub evidence_items: Vec<ProfileEvidenceInput>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub source_id: Option<String>,
    #[serde(default)]
    pub document_type: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub trust: Option<f32>,
    #[serde(default)]
    pub dry_run: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WritingStyleProfileExtractResponse {
    pub profile_format: String,
    pub space_id: String,
    pub source_id: String,
    pub document_type: String,
    pub dry_run: bool,
    pub record_count: usize,
    pub memory_ids: Vec<u64>,
    pub records: Vec<ProfileReviewItem>,
    pub review_endpoint: String,
    pub answer_endpoint: String,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WritingStyleProfileReviewRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub source_id: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WritingStyleProfileReviewResponse {
    pub review_format: String,
    pub space_id: String,
    pub source_id: Option<String>,
    pub total_count: usize,
    pub style_profiles: Vec<ProfileReviewItem>,
    pub answer_endpoint: String,
    pub correction_guidance: String,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignStyleProfileExtractRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub source_id: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub artifact_type: Option<String>,
    #[serde(default)]
    pub notes: String,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub trust: Option<f32>,
    #[serde(default)]
    pub dry_run: Option<bool>,
    #[serde(default)]
    pub evidence_items: Vec<ProfileEvidenceInput>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignStyleProfileExtractResponse {
    pub profile_format: String,
    pub space_id: String,
    pub source_id: String,
    pub artifact_type: String,
    pub dry_run: bool,
    pub record_count: usize,
    pub memory_ids: Vec<u64>,
    pub records: Vec<ProfileReviewItem>,
    pub review_endpoint: String,
    pub answer_endpoint: String,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignStyleProfileReviewRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub source_id: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignStyleProfileReviewResponse {
    pub review_format: String,
    pub space_id: String,
    pub source_id: Option<String>,
    pub total_count: usize,
    pub design_profiles: Vec<ProfileReviewItem>,
    pub answer_endpoint: String,
    pub correction_guidance: String,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationalHabitProfileExtractRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub mission_id: Option<String>,
    #[serde(default)]
    pub operator: Option<String>,
    #[serde(default)]
    pub system_type: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub trust: Option<f32>,
    #[serde(default)]
    pub dry_run: Option<bool>,
    #[serde(default)]
    pub mission_goal: Option<String>,
    #[serde(default)]
    pub sop_markdown: Option<String>,
    #[serde(default)]
    pub constraints: Vec<String>,
    #[serde(default)]
    pub telemetry: Vec<PhysicalMissionTelemetryRecord>,
    #[serde(default)]
    pub incidents: Vec<PhysicalMissionIncidentRecord>,
    #[serde(default)]
    pub after_action_notes: Vec<String>,
    #[serde(default)]
    pub evidence_items: Vec<ProfileEvidenceInput>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationalHabitProfileExtractResponse {
    pub profile_format: String,
    pub space_id: String,
    pub mission_id: Option<String>,
    pub operator: Option<String>,
    pub dry_run: bool,
    pub record_count: usize,
    pub memory_ids: Vec<u64>,
    pub records: Vec<ProfileReviewItem>,
    pub review_endpoint: String,
    pub answer_endpoint: String,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationalHabitProfileReviewRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub mission_id: Option<String>,
    #[serde(default)]
    pub operator: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationalHabitProfileReviewResponse {
    pub review_format: String,
    pub space_id: String,
    pub mission_id: Option<String>,
    pub operator: Option<String>,
    pub total_count: usize,
    pub habit_profiles: Vec<ProfileReviewItem>,
    pub answer_endpoint: String,
    pub correction_guidance: String,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileCorrectRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub activation_scope: Option<String>,
    pub memory_id: u64,
    #[serde(default)]
    pub dimension: Option<String>,
    pub corrected_summary: String,
    #[serde(default)]
    pub evidence: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub trust: Option<f32>,
    #[serde(default)]
    pub dry_run: Option<bool>,
    #[serde(default)]
    pub protected_scopes: Vec<MemoryProtectedScopeInput>,
    #[serde(default)]
    pub override_protection: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileCorrectResponse {
    pub correction_format: String,
    pub space_id: String,
    pub dry_run: bool,
    pub applied: bool,
    pub profile_type: String,
    pub source_memory_id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replacement_memory_id: Option<u64>,
    pub affected_memory_ids: Vec<u64>,
    pub old_preview: String,
    pub new_preview: String,
    pub proposed_record: serde_json::Value,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub protected_matches: Vec<MemoryProtectedScopeMatch>,
    pub apply_allowed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub apply_blocked_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_receipt_preview: Option<MemoryTransformJobReceiptPreview>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lifecycle_contract: Option<MemoryLifecycleContract>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub why_surfaced_contract: Option<MemoryWhySurfacedContract>,
    pub correction_endpoint: String,
    pub review_endpoint: String,
    pub answer_endpoint: String,
    pub rollback_guidance: String,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalMissionChecklistItem {
    pub title: String,
    #[serde(default)]
    pub required: Option<bool>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub evidence: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalMissionTelemetryRecord {
    pub signal: String,
    pub value: String,
    #[serde(default)]
    pub unit: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub observed_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub risk_level: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalMissionIncidentRecord {
    pub summary: String,
    #[serde(default)]
    pub severity: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub observed_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub stale: Option<bool>,
    #[serde(default)]
    pub recommended_action: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalMissionIngestRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub mission_id: String,
    pub mission_goal: String,
    #[serde(default)]
    pub system_type: Option<String>,
    #[serde(default)]
    pub operator: Option<String>,
    #[serde(default)]
    pub risk_level: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub sop_markdown: Option<String>,
    #[serde(default)]
    pub constraints: Vec<String>,
    #[serde(default)]
    pub no_go_zones: Vec<String>,
    #[serde(default)]
    pub checklist: Vec<PhysicalMissionChecklistItem>,
    #[serde(default)]
    pub telemetry: Vec<PhysicalMissionTelemetryRecord>,
    #[serde(default)]
    pub incidents: Vec<PhysicalMissionIncidentRecord>,
    #[serde(default)]
    pub evidence_items: Vec<ProfileEvidenceInput>,
    #[serde(default)]
    pub retention_days: Option<u32>,
    #[serde(default)]
    pub include_operational_habit_profile: Option<bool>,
    #[serde(default)]
    pub protected_scopes: Vec<MemoryProtectedScopeInput>,
    #[serde(default)]
    pub override_protection: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalMissionIngestResponse {
    pub mission_format: String,
    pub space_id: String,
    pub mission_id: String,
    pub apply_allowed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub apply_blocked_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub protected_matches: Vec<MemoryProtectedScopeMatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_receipt_preview: Option<MemoryTransformJobReceiptPreview>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lifecycle_contract: Option<MemoryLifecycleContract>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub why_surfaced_contract: Option<MemoryWhySurfacedContract>,
    pub mission_file_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checklist_file_path: Option<String>,
    pub memory_ids: Vec<u64>,
    pub telemetry_count: usize,
    pub incident_count: usize,
    pub constraint_count: usize,
    pub operational_habit_records: usize,
    pub high_risk_gate: bool,
    pub approval_required_for_actions: bool,
    pub authority_projection: PhysicalMissionAuthorityProjection,
    pub policy_reasons: Vec<String>,
    pub trace_categories: Vec<String>,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalMissionAuthorityProjection {
    pub projection_format: String,
    pub availability: String,
    pub evidence_classification: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub intent_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub action_proposal_ids: Vec<String>,
    pub external_side_effects_authorized: bool,
    pub notes: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalAdapterStatusItem {
    pub adapter_id: String,
    pub label: String,
    pub status: String,
    pub configured: bool,
    pub enabled: bool,
    pub risk_level: String,
    pub setup_requirements: Vec<String>,
    pub input_examples: Vec<String>,
    pub mission_ingest_mapping: Vec<String>,
    pub policy: Vec<String>,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalAdapterStatusResponse {
    pub status_format: String,
    pub generated_at: DateTime<Utc>,
    pub default_mode: String,
    pub configured_count: usize,
    pub enabled_count: usize,
    pub approval_required_for_actions: bool,
    pub adapters: Vec<PhysicalAdapterStatusItem>,
    pub privacy_and_safety_contract: Vec<String>,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalAdapterIngestRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub adapter_id: String,
    pub mission_id: String,
    pub mission_goal: String,
    #[serde(default)]
    pub operator: Option<String>,
    #[serde(default)]
    pub risk_level: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub events: Vec<serde_json::Value>,
    #[serde(default)]
    pub retention_days: Option<u32>,
    #[serde(default)]
    pub protected_scopes: Vec<MemoryProtectedScopeInput>,
    #[serde(default)]
    pub override_protection: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalAdapterIngestResponse {
    pub adapter_ingest_format: String,
    pub adapter_id: String,
    pub mode: String,
    pub normalized_event_count: usize,
    pub telemetry_count: usize,
    pub incident_count: usize,
    pub constraint_count: usize,
    pub checklist_count: usize,
    pub sop_section_count: usize,
    pub warnings: Vec<String>,
    pub mission_ingest: PhysicalMissionIngestResponse,
    pub policy_reasons: Vec<String>,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalAdapterPollRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub adapter_id: String,
    pub mission_id: String,
    pub mission_goal: String,
    #[serde(default)]
    pub operator: Option<String>,
    #[serde(default)]
    pub risk_level: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub endpoint_url: Option<String>,
    #[serde(default)]
    pub subscription_topic: Option<String>,
    #[serde(default)]
    pub events: Vec<serde_json::Value>,
    #[serde(default)]
    pub dry_run: Option<bool>,
    #[serde(default)]
    pub explicit_provider_approval: Option<bool>,
    #[serde(default)]
    pub retention_days: Option<u32>,
    #[serde(default)]
    pub protected_scopes: Vec<MemoryProtectedScopeInput>,
    #[serde(default)]
    pub override_protection: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalAdapterPollResponse {
    pub poll_format: String,
    pub adapter_id: String,
    pub connector_mode: String,
    pub source_kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subscription_topic: Option<String>,
    pub fetched_event_count: usize,
    pub dry_run: bool,
    pub read_only: bool,
    pub subscription_state: String,
    pub ingest: PhysicalAdapterIngestResponse,
    pub warnings: Vec<String>,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalAdapterCollectorRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub collector_id: String,
    pub adapter_id: String,
    pub mission_id: String,
    pub mission_goal: String,
    #[serde(default)]
    pub operator: Option<String>,
    #[serde(default)]
    pub risk_level: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub connector_kind: Option<String>,
    #[serde(default)]
    pub endpoint_url: Option<String>,
    #[serde(default)]
    pub subscription_topic: Option<String>,
    #[serde(default)]
    pub events: Vec<serde_json::Value>,
    #[serde(default)]
    pub dry_run: Option<bool>,
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub interval_seconds: Option<u32>,
    #[serde(default)]
    pub retention_days: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalAdapterCollectorResponse {
    pub collector_format: String,
    pub collector_id: String,
    pub adapter_id: String,
    pub enabled: bool,
    pub connector_kind: String,
    pub interval_seconds: u32,
    pub collector_file_path: String,
    pub run_endpoint: String,
    pub run_payload: serde_json::Value,
    pub approval_required_for_actions: bool,
    pub safety_contract: Vec<String>,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalAdapterCollectorRunResponse {
    pub collector_run_format: String,
    pub collector_id: String,
    pub configured: PhysicalAdapterCollectorResponse,
    pub poll: PhysicalAdapterPollResponse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalOperatorReviewRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub mission_id: String,
    #[serde(default)]
    pub include_collectors: Option<bool>,
    #[serde(default)]
    pub include_file_content: Option<bool>,
    #[serde(default)]
    pub max_files: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalOperatorReviewFile {
    pub path: String,
    pub bytes: u64,
    pub updated_at: DateTime<Utc>,
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_preview: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalOperatorReviewAction {
    pub action_id: String,
    pub label: String,
    pub description: String,
    pub endpoint: String,
    pub method: String,
    pub payload_template: serde_json::Value,
    pub next_step: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalOperatorReviewResponse {
    pub review_format: String,
    pub space_id: String,
    pub mission_id: String,
    pub mission_file_prefix: String,
    pub mission_files: Vec<PhysicalOperatorReviewFile>,
    pub collector_files: Vec<PhysicalOperatorReviewFile>,
    pub high_risk_gate_visible: bool,
    pub approval_required_for_actions: bool,
    pub review_checklist: Vec<String>,
    pub actions: Vec<PhysicalOperatorReviewAction>,
    pub warnings: Vec<String>,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalOperatorMultiReviewRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub mission_ids: Vec<String>,
    #[serde(default)]
    pub include_collectors: Option<bool>,
    #[serde(default)]
    pub include_file_content: Option<bool>,
    #[serde(default)]
    pub max_files_per_mission: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalOperatorMultiReviewMissionSummary {
    pub mission_id: String,
    pub mission_file_count: usize,
    pub collector_file_count: usize,
    pub high_risk_gate_visible: bool,
    pub warning_count: usize,
    pub status: String,
    pub next_step: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalOperatorMultiReviewResponse {
    pub review_format: String,
    pub space_id: String,
    pub mission_count: usize,
    pub mission_summaries: Vec<PhysicalOperatorMultiReviewMissionSummary>,
    pub reviews: Vec<PhysicalOperatorReviewResponse>,
    pub high_risk_mission_count: usize,
    pub missing_evidence_mission_count: usize,
    pub fleet_checklist: Vec<String>,
    pub actions: Vec<PhysicalOperatorReviewAction>,
    pub warnings: Vec<String>,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalOperatorMissionBoardRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub status_filter: Option<String>,
    #[serde(default)]
    pub mission_id_contains: Option<String>,
    #[serde(default)]
    pub include_collectors: Option<bool>,
    #[serde(default)]
    pub include_file_content: Option<bool>,
    #[serde(default)]
    pub max_missions: Option<usize>,
    #[serde(default)]
    pub max_files_per_mission: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalOperatorMissionBoardResponse {
    pub board_format: String,
    pub space_id: String,
    pub discovered_mission_ids: Vec<String>,
    pub filtered_mission_ids: Vec<String>,
    pub status_filter: Option<String>,
    pub mission_id_contains: Option<String>,
    pub multi_review: PhysicalOperatorMultiReviewResponse,
    pub board_sections: Vec<String>,
    pub actions: Vec<PhysicalOperatorReviewAction>,
    pub warnings: Vec<String>,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalOperatorMissionDetailRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub mission_id: String,
    #[serde(default)]
    pub include_collectors: Option<bool>,
    #[serde(default)]
    pub include_file_content: Option<bool>,
    #[serde(default)]
    pub max_files: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalOperatorMissionDetailResponse {
    pub detail_format: String,
    pub space_id: String,
    pub mission_id: String,
    pub status: String,
    pub review: PhysicalOperatorReviewResponse,
    pub detail_sections: Vec<String>,
    pub actions: Vec<PhysicalOperatorReviewAction>,
    pub warnings: Vec<String>,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTraceResponse {
    pub request_id: String,
    pub trace_format: String,
    pub generated_at: DateTime<Utc>,
    pub memory_mode: String,
    pub memory_status: String,
    pub decision_reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recall_policy: Option<MemoryRecallPolicyEnvelope>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recall_outcome: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_policy: Option<MemoryReasoningPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector_retrieval: Option<MemoryTraceVectorRetrieval>,
    pub memories_searched: Vec<MemoryTraceItem>,
    pub memories_selected: Vec<MemoryTraceItem>,
    pub memories_injected: Vec<MemoryTraceItem>,
    pub memories_ignored: Vec<MemoryTraceItem>,
    pub conflicts_or_stale_records: Vec<MemoryTraceItem>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub file_evidence: Vec<MemoryFileEvidence>,
    pub final_context: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explanation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer_pipeline: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextBuildResponse {
    pub request_id: String,
    pub context_format: String,
    pub memory_mode: String,
    pub memory_status: String,
    pub decision_reason: String,
    pub recall_policy: MemoryRecallPolicyEnvelope,
    pub summary: String,
    pub markdown: String,
    pub current_goal: String,
    pub constraints: Vec<String>,
    pub recently_completed: Vec<String>,
    pub open_risks: Vec<String>,
    pub recommended_next_steps: Vec<String>,
    pub evidence_ids: Vec<u64>,
    pub memory_evidence: Vec<MemoryContextEvidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recommended_cadence: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drift_warning: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextProjectRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub project: Option<String>,
    pub repo_path: Option<String>,
    pub task: Option<String>,
    pub top_k: Option<usize>,
    pub per_bucket: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextProjectResponse {
    pub context_format: String,
    pub generated_at: DateTime<Utc>,
    pub space_id: String,
    pub project: Option<String>,
    pub repo_path: Option<String>,
    pub current_goal: String,
    pub decisions: Vec<MemoryContextEvidence>,
    pub constraints: Vec<MemoryContextEvidence>,
    pub open_loops: Vec<MemoryContextEvidence>,
    pub recent_incidents: Vec<MemoryContextEvidence>,
    pub key_artifacts: Vec<MemoryContextEvidence>,
    pub stale_or_conflicting: Vec<MemoryContextEvidence>,
    pub evidence_ids: Vec<u64>,
    pub markdown: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextResumeRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub project: Option<String>,
    pub repo_path: Option<String>,
    pub task: Option<String>,
    pub context_package: Option<MemoryCapsuleContextPackage>,
    pub bundle: Option<MemoryContainerBundle>,
    pub mip_bundle: Option<MemoryInterchangeBundle>,
    pub top_k: Option<usize>,
    pub per_bucket: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextResumeResponse {
    pub context_format: String,
    pub generated_at: DateTime<Utc>,
    pub source: String,
    pub space_id: Option<String>,
    pub project: Option<String>,
    pub repo_path: Option<String>,
    pub resume_prompt: String,
    pub working_context: String,
    pub context_package: Option<MemoryCapsuleContextPackage>,
    pub project_context: Option<MemoryContextProjectResponse>,
    pub evidence_ids: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextReviewRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub project: Option<String>,
    pub repo_path: Option<String>,
    pub task: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextReviewItem {
    pub memory_id: u64,
    pub layer: MemoryLayer,
    pub memory_type: String,
    pub created_at: DateTime<Utc>,
    pub issue_type: String,
    pub severity: String,
    pub priority: u8,
    pub reason: String,
    pub recommendation_reason_codes: Vec<String>,
    pub recommended_action: String,
    pub recommendation_why: String,
    pub recommendation_next_step: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supporting_signals: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub caveats: Vec<String>,
    pub action_endpoint: String,
    pub canonical_key: Option<String>,
    pub semantic_status: Option<String>,
    pub conflict_set_id: Option<String>,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextReviewResponse {
    pub context_format: String,
    pub generated_at: DateTime<Utc>,
    pub space_id: Option<String>,
    pub project: Option<String>,
    pub stale_records: Vec<MemoryContextReviewItem>,
    pub conflict_records: Vec<MemoryContextReviewItem>,
    pub superseded_records: Vec<MemoryContextReviewItem>,
    pub recommended_actions: Vec<String>,
    pub action_plan: Vec<String>,
    pub evidence_ids: Vec<u64>,
    pub markdown: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextReviewActionRequest {
    pub memory_id: u64,
    pub action: String,
    pub replacement_memory_id: Option<u64>,
    pub superseded_by: Option<String>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextReviewActionResponse {
    pub action_format: String,
    pub applied: bool,
    pub action: String,
    pub memory_id: u64,
    pub affected_memory_ids: Vec<u64>,
    pub item: Option<MemoryBrowserItem>,
    pub message: String,
}
