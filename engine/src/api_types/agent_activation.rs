use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentScopedQueryRequest {
    pub conversation_id: Option<String>,
    pub agent_id: String,
    pub shared_blocks: Option<Vec<String>>,
    pub query: String,
    pub top_k: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentRuntimeListRequest {
    pub conversation_id: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRuntimeSummary {
    pub conversation_id: String,
    pub agent_id: String,
    pub role: String,
    pub status: String,
    pub lifecycle_state: String,
    pub registered_at: DateTime<Utc>,
    pub last_seen_at: DateTime<Utc>,
    pub last_ingest_at: Option<DateTime<Utc>>,
    pub last_query_at: Option<DateTime<Utc>>,
    pub ingest_count: usize,
    pub query_count: usize,
    pub scoped_memory_count: usize,
    pub supervisor_agent_id: Option<String>,
    pub worker_count: usize,
    pub shared_blocks: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentRuntimeListResponse {
    pub items: Vec<AgentRuntimeSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRuntimeUpsertRequest {
    pub conversation_id: String,
    pub agent_id: String,
    pub role: Option<String>,
    pub supervisor_agent_id: Option<String>,
    pub shared_blocks: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentRuntimeUpsertResponse {
    pub agent: AgentRuntimeSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRuntimeControlRequest {
    pub conversation_id: String,
    pub agent_id: String,
    pub command: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentRuntimeControlResponse {
    pub applied: bool,
    pub command: String,
    pub reason: String,
    pub agent: AgentRuntimeSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTaskRecord {
    pub task_id: String,
    pub conversation_id: String,
    pub supervisor_agent_id: String,
    pub worker_agent_id: String,
    pub lifecycle_contract: String,
    pub authority_state: String,
    pub external_side_effects_authorized: bool,
    pub status: String,
    pub content: String,
    pub result: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTaskDispatchRequest {
    pub conversation_id: String,
    pub task_id: String,
    pub supervisor_agent_id: String,
    pub worker_agent_id: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentTaskDispatchResponse {
    pub task: AgentTaskRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentTaskListRequest {
    pub conversation_id: Option<String>,
    pub agent_id: Option<String>,
    pub status: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentTaskListResponse {
    pub items: Vec<AgentTaskRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTaskUpdateRequest {
    pub conversation_id: String,
    pub task_id: String,
    pub agent_id: String,
    pub status: String,
    pub result: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentTaskUpdateResponse {
    pub task: AgentTaskRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRuntimeInspectRequest {
    pub conversation_id: String,
    pub agent_id: String,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentRuntimeInspectResponse {
    pub agent: AgentRuntimeSummary,
    pub workers: Vec<AgentRuntimeSummary>,
    pub tasks: Vec<AgentTaskRecord>,
    pub recent_memories: Vec<RetrievedItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessageSendRequest {
    pub conversation_id: String,
    pub from_agent_id: String,
    pub to_agent_id: String,
    pub content: String,
    pub source: Option<String>,
    pub trust: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentMessageSendResponse {
    pub delivered: bool,
    pub from_agent_id: String,
    pub to_agent_id: String,
    pub memory_id: Option<u64>,
    pub source: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessageInboxRequest {
    pub conversation_id: String,
    pub agent_id: String,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentMessageInboxResponse {
    pub items: Vec<RetrievedItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationPromoteRequest {
    pub conversation_id: String,
    pub memory_ids: Option<Vec<u64>>,
    pub query: Option<String>,
    pub top_k: Option<usize>,
    pub pin: Option<bool>,
    pub source: Option<String>,
    pub expires_in_seconds: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ActivationPromoteResponse {
    pub promoted_count: usize,
    pub created_memory_ids: Vec<u64>,
    pub items: Vec<RetrievedItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationUpdateRequest {
    pub conversation_id: String,
    pub activation_memory_ids: Vec<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ActivationUpdateResponse {
    pub updated_count: usize,
    pub activation_memory_ids: Vec<u64>,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationQueryRequest {
    pub conversation_id: String,
    pub include_cleared: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ActivationQueryResponse {
    pub conversation_id: String,
    pub items: Vec<RetrievedItem>,
}
