use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorPullRequest {
    pub conversation_id: String,
    pub connector_type: String,
    pub dry_run: Option<bool>,
    pub payload: serde_json::Value,
    pub trust: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConnectorPreviewItem {
    pub external_id: String,
    pub title: Option<String>,
    pub normalized_source: String,
    pub layer: MemoryLayer,
    pub content_preview: String,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConnectorItemError {
    pub item_index: usize,
    pub error: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConnectorPullResponse {
    pub connector_type: String,
    pub trigger: String,
    pub dry_run: bool,
    pub imported: usize,
    pub failed_items: usize,
    pub preview_items: Vec<ConnectorPreviewItem>,
    pub item_errors: Vec<ConnectorItemError>,
    pub memory_ids: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorWebhookSecretUpsertRequest {
    pub conversation_id: String,
    pub secret: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConnectorWebhookSecretUpsertResponse {
    pub conversation_id: String,
    pub configured: bool,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorScheduleUpsertRequest {
    pub schedule_id: String,
    pub conversation_id: String,
    pub connector_type: String,
    pub interval_minutes: u64,
    pub payload: serde_json::Value,
    pub trust: Option<f32>,
    pub enabled: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConnectorScheduleRecord {
    pub schedule_id: String,
    pub conversation_id: String,
    pub connector_type: String,
    pub interval_minutes: u64,
    pub enabled: bool,
    pub state: String,
    pub updated_at: DateTime<Utc>,
    pub last_run_at: Option<DateTime<Utc>>,
    pub last_success_at: Option<DateTime<Utc>>,
    pub last_failure_at: Option<DateTime<Utc>>,
    pub next_run_at: Option<DateTime<Utc>>,
    pub consecutive_failures: u32,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConnectorScheduleUpsertResponse {
    pub schedule: ConnectorScheduleRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConnectorScheduleRunRequest {
    pub conversation_id: Option<String>,
    pub connector_type: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConnectorScheduleRunResponse {
    pub executed_schedules: usize,
    pub failed_schedules: usize,
    pub total_imported: usize,
    pub runs: Vec<ConnectorRunRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConnectorSchedulesRequest {
    pub conversation_id: Option<String>,
    pub connector_type: Option<String>,
    pub state: Option<String>,
    pub only_failing: Option<bool>,
    pub only_backoff: Option<bool>,
    pub enabled: Option<bool>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConnectorSchedulesSummary {
    pub total_schedules: usize,
    pub enabled_schedules: usize,
    pub ready_schedules: usize,
    pub healthy_schedules: usize,
    pub disabled_schedules: usize,
    pub failing_schedules: usize,
    pub backoff_schedules: usize,
    pub last_success_at: Option<DateTime<Utc>>,
    pub last_failure_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConnectorSchedulesResponse {
    pub summary: ConnectorSchedulesSummary,
    pub items: Vec<ConnectorScheduleRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConnectorRunsRequest {
    pub conversation_id: Option<String>,
    pub connector_type: Option<String>,
    pub trigger: Option<String>,
    pub outcome: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConnectorRunRecord {
    pub run_id: String,
    pub connector_type: String,
    pub trigger: String,
    pub conversation_id: String,
    pub dry_run: bool,
    pub outcome: String,
    pub error: Option<String>,
    pub imported: usize,
    pub failed_items: usize,
    pub preview_count: usize,
    pub memory_ids: Vec<u64>,
    pub created_at: DateTime<Utc>,
    pub schedule_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConnectorRunsSummary {
    pub total_runs: usize,
    pub applied_runs: usize,
    pub preview_runs: usize,
    pub failed_runs: usize,
    pub total_imported: usize,
    pub total_failed_items: usize,
    pub total_preview_items: usize,
    pub multi_item_runs: usize,
    pub last_run_at: Option<DateTime<Utc>>,
    pub last_failure_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConnectorRunsResponse {
    pub summary: ConnectorRunsSummary,
    pub items: Vec<ConnectorRunRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConnectorStatusRequest {
    pub conversation_id: Option<String>,
    pub connector_type: Option<String>,
    pub only_failing: Option<bool>,
    pub only_backoff: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConnectorStatusItem {
    pub connector_type: String,
    pub configured: bool,
    pub active_schedule_count: usize,
    pub last_outcome: Option<String>,
    pub last_error: Option<String>,
    pub total_runs: usize,
    pub total_imported: usize,
    pub dry_runs: usize,
    pub manual_runs: usize,
    pub scheduled_runs: usize,
    pub webhook_runs: usize,
    pub last_run_at: Option<DateTime<Utc>>,
    pub schedule_count: usize,
    pub failing_schedule_count: usize,
    pub backoff_schedule_count: usize,
    pub last_success_at: Option<DateTime<Utc>>,
    pub last_failure_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConnectorRetryPostureSummary {
    pub failing_schedules: usize,
    pub backoff_schedules: usize,
    pub max_consecutive_failures: u32,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConnectorCatalogSummary {
    pub total_connectors: usize,
    pub configured_connectors: usize,
    pub active_connectors: usize,
    pub failing_connectors: usize,
    pub backoff_connectors: usize,
    pub applied_last_outcome_connectors: usize,
    pub preview_last_outcome_connectors: usize,
    pub failed_last_outcome_connectors: usize,
    pub last_run_at: Option<DateTime<Utc>>,
    pub last_failure_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConnectorStatusResponse {
    pub total_runs: usize,
    pub total_imported: usize,
    pub catalog_summary: ConnectorCatalogSummary,
    pub retry_posture: Option<ConnectorRetryPostureSummary>,
    pub connectors: Vec<ConnectorStatusItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConnectorOperationsReportRequest {
    pub conversation_id: Option<String>,
    pub connector_type: Option<String>,
    pub recent_limit: Option<usize>,
    pub finding_severity: Option<String>,
    pub finding_code: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConnectorOperationsReportResponse {
    pub status: ConnectorStatusResponse,
    pub runs_summary: ConnectorRunsSummary,
    pub schedules_summary: ConnectorSchedulesSummary,
    pub findings_summary: ConnectorOperationsFindingsSummary,
    pub findings: Vec<OperatorFindingSummary>,
    pub recent_failed_runs: Vec<ConnectorRunRecord>,
    pub recent_backoff_schedules: Vec<ConnectorScheduleRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConnectorOperationsFindingsSummary {
    pub total_findings: usize,
    pub high_findings: usize,
    pub medium_findings: usize,
    pub low_findings: usize,
    pub connector_failing_schedule_findings: usize,
    pub connector_failed_run_findings: usize,
}
