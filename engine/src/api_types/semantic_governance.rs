use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalSemanticQueryRequest {
    pub conversation_id: Option<String>,
    pub entity_id: Option<String>,
    pub attribute_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalSemanticItem {
    pub canonical_key: String,
    pub conversation_id: Option<String>,
    pub semantic_node: SemanticNode,
    pub source_memory_ids: Vec<u64>,
    pub audit_entries: Vec<CanonicalSemanticAuditEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CanonicalSemanticResponse {
    pub items: Vec<CanonicalSemanticItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalSemanticMutationRequest {
    pub conversation_id: String,
    pub entity_id: String,
    pub attribute_key: String,
    pub action: String,
    pub winner_semantic_id: Option<String>,
    pub replacement_value: Option<String>,
    pub reason_code: Option<String>,
    pub evidence_bundle: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CanonicalSemanticMutationResponse {
    pub canonical_key: String,
    pub applied: bool,
    pub action: String,
    pub semantic_node: Option<SemanticNode>,
    pub updated_memory_ids: Vec<u64>,
    pub audit_entry: Option<CanonicalSemanticAuditEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalSemanticAuditEntry {
    pub canonical_key: String,
    pub conversation_id: String,
    pub action: String,
    pub before_value: Option<String>,
    pub after_value: Option<String>,
    pub reason: String,
    pub reason_code: String,
    pub evidence_bundle: Vec<String>,
    pub affected_memory_ids: Vec<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SemanticSleepCycleResponse {
    pub conversation_id: String,
    pub job_id: String,
    pub job_status: String,
    pub conversation_count: usize,
    pub window_start: Option<DateTime<Utc>>,
    pub window_end: Option<DateTime<Utc>>,
    pub scanned_canonical_nodes: usize,
    pub repaired_nodes: usize,
    pub resolved_conflicts: usize,
    pub deprecated_nodes: usize,
    pub audit_entries_created: usize,
    pub cluster_summaries_created: usize,
    pub cluster_summaries: Vec<ConsolidationClusterSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConsolidationJobMetrics {
    pub repaired_nodes: usize,
    pub resolved_conflicts: usize,
    pub deprecated_nodes: usize,
    pub audit_entries_created: usize,
    pub cluster_summaries_created: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConsolidationClusterSummary {
    pub cluster_kind: String,
    pub cluster_key: String,
    pub item_count: usize,
    pub anchor_ids: Vec<u64>,
    pub summary_memory_id: Option<u64>,
    #[serde(default)]
    pub conversation_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConsolidationJob {
    pub job_id: String,
    pub job_type: String,
    pub scope: String,
    pub status: String,
    pub priority: u32,
    pub trigger_reason: String,
    pub scheduled_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub conversation_id: Option<String>,
    pub conversation_count: usize,
    pub window_start: Option<DateTime<Utc>>,
    pub window_end: Option<DateTime<Utc>>,
    pub result_summary: Option<String>,
    pub result_metrics: Option<ConsolidationJobMetrics>,
    #[serde(default)]
    pub result_clusters: Vec<ConsolidationClusterSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationRunRequest {
    pub conversation_id: Option<String>,
    pub scope: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConsolidationJobsResponse {
    pub items: Vec<ConsolidationJob>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GovernanceMultimodalCounts {
    pub total: usize,
    pub documents: usize,
    pub images: usize,
    pub audios: usize,
    pub videos: usize,
    pub ocr_texts: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct GovernanceDashboardResponse {
    pub conversation_id: Option<String>,
    pub sort_by: String,
    pub limit: usize,
    pub has_more_jobs: bool,
    pub has_more_candidates: bool,
    pub has_more_conflicts: bool,
    pub disputed_semantic_count: usize,
    pub high_priority_jobs: usize,
    pub recent_audit_entries: Vec<CanonicalSemanticAuditEntry>,
    pub top_disputed_semantic_keys: Vec<String>,
    pub top_high_priority_jobs: Vec<ConsolidationJob>,
    pub recommended_action_counts: HashMap<String, usize>,
    pub layer_governance_counts: HashMap<String, usize>,
    pub multimodal_counts: GovernanceMultimodalCounts,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceDashboardRequest {
    pub conversation_id: Option<String>,
    pub only_blocked_candidates: Option<bool>,
    pub min_job_priority: Option<u32>,
    pub limit: Option<usize>,
    pub sort_by: Option<String>,
    #[serde(default)]
    pub trace_group: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GovernanceConflictViewResponse {
    pub conversation_id: Option<String>,
    pub limit: usize,
    pub sort_by: String,
    pub has_more: bool,
    pub items: Vec<String>,
}



#[derive(Debug, Clone, Serialize)]
pub struct GovernanceJobViewResponse {
    pub conversation_id: Option<String>,
    pub limit: usize,
    pub sort_by: String,
    pub has_more: bool,
    pub items: Vec<ConsolidationJob>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GovernanceJobWindowSummary {
    pub job_count: usize,
    pub conversation_count: usize,
    pub scopes: Vec<String>,
    pub window_start: Option<DateTime<Utc>>,
    pub window_end: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GovernanceContainerActivitySummary {
    pub total_operations: usize,
    pub exports: usize,
    pub imports: usize,
    pub affected_memory_count: usize,
    pub affected_activation_memory_count: usize,
    pub last_recorded_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GovernanceConnectorActivitySummary {
    pub total_runs: usize,
    pub manual_runs: usize,
    pub scheduled_runs: usize,
    pub webhook_runs: usize,
    pub scheduled_connectors: usize,
    pub affected_memory_count: usize,
    pub last_recorded_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GovernanceConnectorSecuritySummary {
    pub protected_conversations: usize,
    pub replay_protection_enabled: bool,
    pub timestamp_skew_seconds: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct GovernanceConnectorRetryPostureSummary {
    pub failing_schedules: usize,
    pub backoff_schedules: usize,
    pub max_consecutive_failures: u32,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GovernanceAlert {
    pub code: String,
    pub severity: String,
    pub count: usize,
    pub message: String,
    pub trace_group: String,
    pub category: String,
    pub recommended_filter: String,
    pub details: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct GovernanceAlertsResponse {
    pub conversation_id: Option<String>,
    pub generated_at: DateTime<Utc>,
    pub summary: String,
    pub job_window: Option<GovernanceJobWindowSummary>,
    pub container_activity: Option<GovernanceContainerActivitySummary>,
    pub connector_activity: Option<GovernanceConnectorActivitySummary>,
    pub connector_security: Option<GovernanceConnectorSecuritySummary>,
    pub connector_retry_posture: Option<GovernanceConnectorRetryPostureSummary>,
    pub multimodal_counts: GovernanceMultimodalCounts,
    pub items: Vec<GovernanceAlert>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GovernanceSnapshotResponse {
    pub generated_at: DateTime<Utc>,
    pub summary: String,
    pub job_window: Option<GovernanceJobWindowSummary>,
    pub container_activity: Option<GovernanceContainerActivitySummary>,
    pub connector_activity: Option<GovernanceConnectorActivitySummary>,
    pub connector_security: Option<GovernanceConnectorSecuritySummary>,
    pub connector_retry_posture: Option<GovernanceConnectorRetryPostureSummary>,
    pub multimodal_counts: GovernanceMultimodalCounts,
    pub dashboard: GovernanceDashboardResponse,
    pub conflicts: GovernanceConflictViewResponse,
    pub jobs: GovernanceJobViewResponse,
    pub alerts: GovernanceAlertsResponse,
}























#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcedureActivateRequest {
    pub conversation_id: String,
    pub query: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcedureImportSkillFileRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub path: String,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub trust: Option<f32>,
    #[serde(default)]
    pub activate_query: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProcedureImportSkillFileResponse {
    pub import_format: String,
    pub generated_at: DateTime<Utc>,
    pub space_id: String,
    pub path: String,
    pub accepted: bool,
    pub memory_id: Option<u64>,
    pub procedure_schema: Option<ProcedureSchema>,
    pub validation_warnings: Vec<String>,
    pub activation: Option<ProcedureActivationResponse>,
    pub ingest: IngestResponse,
    pub rollback_endpoint: String,
    pub rollback_payload: serde_json::Value,
    pub trace_group: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcedureImportRollbackRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub procedure_id: u64,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub actor: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProcedureImportRollbackResponse {
    pub rollback_format: String,
    pub generated_at: DateTime<Utc>,
    pub space_id: String,
    pub procedure_id: u64,
    pub path: Option<String>,
    pub rolled_back: bool,
    pub reason: String,
    pub activation_after_rollback: Option<ProcedureActivationResponse>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcedureSyncSkillFilesRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub force: Option<bool>,
    #[serde(default)]
    pub trust: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProcedureSyncSkillFileItem {
    pub path: String,
    pub status: String,
    pub previous_procedure_id: Option<u64>,
    pub memory_id: Option<u64>,
    pub validation_warnings: Vec<String>,
    pub rollback_endpoint: Option<String>,
    pub rollback_payload: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProcedureSyncSkillFilesResponse {
    pub sync_format: String,
    pub generated_at: DateTime<Utc>,
    pub space_id: String,
    pub scanned_count: usize,
    pub imported_count: usize,
    pub updated_count: usize,
    pub unchanged_count: usize,
    pub failed_count: usize,
    pub items: Vec<ProcedureSyncSkillFileItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProcedureStepStatus {
    pub step: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProcedureActivationResponse {
    pub procedure_id: u64,
    pub procedure_content: String,
    pub source: Option<String>,
    pub skill_file_path: Option<String>,
    pub activation_eligible: bool,
    pub safety_warnings: Vec<String>,
    pub preconditions_met: bool,
    pub confidence: f32,
    pub trigger_conditions: Vec<String>,
    pub steps: Vec<ProcedureStepStatus>,
    pub rollback_steps: Vec<String>,
    pub success_criteria: Vec<String>,
    pub semantic_preconditions: Vec<String>,
    pub semantic_recovery_patterns: Vec<String>,
    pub semantic_failure_motifs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcedureExecutionRequest {
    pub conversation_id: String,
    pub procedure_id: u64,
    pub steps: Vec<ProcedureStepStatus>,
    pub failure_reason: Option<String>,
    pub success_evidence: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProcedureExecutionResponse {
    pub procedure_id: u64,
    pub execution_status: String,
    pub execution_memory_id: Option<u64>,
    pub event_id: Option<u64>,
    pub steps: Vec<ProcedureStepStatus>,
    pub failure_reason: Option<String>,
    pub success_evidence: Option<String>,
    pub semantic_recovery_guidance: Vec<String>,
    pub semantic_failure_motifs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcedureStatusRequest {
    pub conversation_id: String,
    pub procedure_id: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProcedureStatusResponse {
    pub procedure_id: u64,
    pub current_step_index: Option<usize>,
    pub retry_count: u32,
    pub outcome: String,
    pub failure_reason: Option<String>,
    pub success_evidence: Option<String>,
    pub steps: Vec<ProcedureStepStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcedureConsolidateRequest {
    pub conversation_id: String,
    pub procedure_id: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProcedureConsolidateResponse {
    pub consolidated: bool,
    pub procedure_id: Option<u64>,
    pub version: Option<u32>,
    pub synthesized_trigger_conditions: Vec<String>,
    pub synthesized_rollback_steps: Vec<String>,
    pub synthesized_success_criteria: Vec<String>,
    pub synthesized_failure_motifs: Vec<String>,
    pub synthesized_fact_ids: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcedureEvaluateRequest {
    pub conversation_id: String,
    pub procedure_id: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProcedureEvaluateResponse {
    pub procedure_id: u64,
    pub version: u32,
    pub execution_count: u32,
    pub completed_count: u32,
    pub failed_count: u32,
    pub success_rate: f32,
    pub consecutive_failures: u32,
    pub health_score: f32,
    pub upgrade_recommended: bool,
    pub deprecate_recommended: bool,
    pub activation_eligible: bool,
    pub preferred: bool,
    pub replacement_procedure_id: Option<u64>,
}
