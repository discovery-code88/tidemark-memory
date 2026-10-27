use super::*;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExecutionBudget {
    #[serde(default)]
    pub deadline_ms: Option<u64>,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    #[serde(default)]
    pub max_cost_microusd: Option<u64>,
    #[serde(default)]
    pub max_compute_ms: Option<u64>,
    #[serde(default)]
    pub max_energy_millijoules: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestStageTiming {
    pub stage: String,
    pub queue_class: String,
    pub component: String,
    pub started_offset_ms: u64,
    pub duration_ms: u64,
    pub outcome: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct WorkDispositionCounts {
    pub eligible: usize,
    pub selected: usize,
    pub omitted: usize,
    pub deferred: usize,
    pub blocked: usize,
    pub failed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultMilestoneReceipt {
    pub first_useful_result_ms: u64,
    pub complete_result_ms: u64,
    pub background_enrichment_status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background_job_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueClassReceipt {
    pub queue_class: String,
    pub priority: u8,
    pub running_limit: usize,
    pub queued_limit: usize,
    pub queued: usize,
    pub running: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceBudgetReceipt {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deadline_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_limit_microusd: Option<u64>,
    pub cost_used_microusd: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compute_limit_ms: Option<u64>,
    pub compute_used_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy_limit_millijoules: Option<u64>,
    pub energy_used_millijoules: u64,
    pub energy_measurement: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentDegradationReceipt {
    pub component: String,
    pub observed_status: String,
    pub disposition: String,
    pub fallback_from: String,
    pub fallback_to: String,
    pub reason_code: String,
    pub retry_count: usize,
    pub user_visible: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderFallbackReceipt {
    pub provider_kind: String,
    pub requested_provider: String,
    pub selected_provider: String,
    pub reason_code: String,
    pub quality_tier: String,
    pub estimated_cost_microusd: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuationReceipt {
    pub available: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestPerformanceReceipt {
    pub receipt_format: String,
    pub build_id: String,
    pub request_id: String,
    pub operation: String,
    pub workload_shape: String,
    pub cache_state: String,
    pub stages: Vec<RequestStageTiming>,
    pub milestones: ResultMilestoneReceipt,
    pub queues: Vec<QueueClassReceipt>,
    pub counts: WorkDispositionCounts,
    pub budget: ResourceBudgetReceipt,
    pub degradations: Vec<ComponentDegradationReceipt>,
    pub provider_fallbacks: Vec<ProviderFallbackReceipt>,
    pub continuation: ContinuationReceipt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedTaskSubmitRequest {
    pub operation: String,
    pub queue_class: String,
    #[serde(default)]
    pub duration_ms: Option<u64>,
    #[serde(default)]
    pub fail_attempts: Option<usize>,
    #[serde(default)]
    pub retry_safe: Option<bool>,
    #[serde(default)]
    pub max_retries: Option<usize>,
    #[serde(default)]
    pub budget: ExecutionBudget,
    #[serde(default)]
    pub counts: WorkDispositionCounts,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedTaskRecord {
    pub task_format: String,
    pub job_id: String,
    pub operation: String,
    pub queue_class: String,
    pub status: String,
    pub stage: String,
    pub attempt: usize,
    pub max_attempts: usize,
    pub retry_safe: bool,
    pub submitted_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub cancel_requested: bool,
    pub counts: WorkDispositionCounts,
    pub budget: ResourceBudgetReceipt,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuation_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_job_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedTaskSubmissionResponse {
    pub accepted: bool,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    pub queue_class: String,
    pub counts: WorkDispositionCounts,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedTaskActionRequest {
    pub job_id: String,
    #[serde(default)]
    pub continuation_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedTaskListRequest {
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub queue_class: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedTaskListResponse {
    pub list_format: String,
    pub items: Vec<ManagedTaskRecord>,
    pub queues: Vec<QueueClassReceipt>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealthInput {
    pub component: String,
    pub status: String,
    #[serde(default)]
    pub required: Option<bool>,
    #[serde(default)]
    pub retry_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationPlanRequest {
    pub operation: String,
    pub components: Vec<ComponentHealthInput>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationPlanResponse {
    pub plan_format: String,
    pub operation: String,
    pub overall_status: String,
    pub receipts: Vec<ComponentDegradationReceipt>,
    pub counts: WorkDispositionCounts,
}
