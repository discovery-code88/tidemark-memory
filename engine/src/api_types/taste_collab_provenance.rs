use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextTasteGraphLayoutSaveRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub positions: Vec<MemoryGraphLayoutPosition>,
    pub mode: Option<String>,
    pub width: Option<f32>,
    pub height: Option<f32>,
    pub actor: Option<String>,
    pub revision_note: Option<String>,
    pub base_revision: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextTasteGraphLayoutSaveResponse {
    pub applied: bool,
    pub space_id: String,
    pub position_count: usize,
    pub layout_memory_id: u64,
    pub revision: u32,
    pub actor: String,
    pub updated_at: DateTime<Utc>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextTasteGraphLayoutLoadRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub mode: Option<String>,
    pub width: Option<f32>,
    pub height: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextTasteGraphLayoutLoadResponse {
    pub space_id: String,
    pub positions: Vec<MemoryGraphLayoutPosition>,
    pub layout_memory_id: Option<u64>,
    pub revision: Option<u32>,
    pub actor: Option<String>,
    pub updated_at: Option<DateTime<Utc>>,
    pub revision_note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextTasteGraphTraverseRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub source_node_id: String,
    pub target_node_id: String,
    pub max_depth: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextTasteGraphTraverseResponse {
    pub path_found: bool,
    pub path_node_ids: Vec<String>,
    pub traversed_edges: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextTasteGraphAlgorithmsRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageRankResult {
    pub node_id: String,
    pub label: String,
    pub rank: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleResult {
    pub path: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringResult {
    pub component_index: usize,
    pub node_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContextTasteGraphAlgorithmsResponse {
    pub pagerank: Vec<PageRankResult>,
    pub cycles: Vec<CycleResult>,
    pub clusters: Vec<ClusteringResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CollaborationMessage {
    Join {
        username: String,
        color: String,
    },
    Leave {
        username: String,
    },
    CursorMove {
        username: String,
        x: f64,
        y: f64,
        active_node_id: Option<String>,
    },
    Presence {
        users: Vec<CollaborationUser>,
    },
    Patch {
        patch_type: String, // e.g. "node_updated", "node_merged", "node_split", "layout_saved"
        payload: serde_json::Value,
    },
    Operation {
        operation_id: String,
        username: String,
        base_revision: Option<u64>,
        revision: u64,
        op: String,
        payload: serde_json::Value,
        conflict: bool,
    },
    ReplayOps {
        base_revision: u64,
        operations: Vec<ReplayOpItem>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayOpItem {
    pub op: String,
    pub payload: serde_json::Value,
    #[serde(default)]
    pub parent_revisions: Option<Vec<u64>>,
    #[serde(default)]
    pub base_revision: Option<u64>,
    #[serde(default)]
    pub vector_clock: Option<std::collections::HashMap<String, u64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationEnvelope {
    pub space_id: String,
    pub client_id: String,
    pub last_seen_server_revision: u64,
    pub operations: Vec<ReplayOpItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationSyncRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub client_id: String,
    pub last_seen_server_revision: u64,
    #[serde(default)]
    pub client_operations: Vec<ReplayOpItem>,
    #[serde(default)]
    pub known_revisions: Option<Vec<u64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationRemoteOperation {
    pub operation_id: String,
    pub username: String,
    pub base_revision: Option<u64>,
    pub revision: u64,
    pub op: String,
    pub payload: serde_json::Value,
    pub conflict: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationReplayRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub username: String,
    pub base_revision: u64,
    pub operations: Vec<ReplayOpItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationReplayResponse {
    pub applied_count: usize,
    pub conflicts_count: usize,
    pub server_revision: u64,
    #[serde(default)]
    pub applied_operations: Vec<CollaborationAppliedOperation>,
    #[serde(default)]
    pub conflicts: Vec<CollaborationConflict>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationSyncResponse {
    pub sync_format: String,
    pub space_id: String,
    pub client_id: String,
    pub server_revision_before_sync: u64,
    pub server_revision_after_sync: u64,
    pub remote_operations: Vec<CollaborationRemoteOperation>,
    pub applied_count: usize,
    pub conflicts_count: usize,
    pub applied_operations: Vec<CollaborationAppliedOperation>,
    pub conflicts: Vec<CollaborationConflict>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationOperationContractRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub include_examples: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationOperationContractItem {
    pub op: String,
    pub target_scope: String,
    pub required_fields: Vec<String>,
    pub automatic_transform: bool,
    pub compactable: bool,
    pub automatic_rebase: bool,
    pub conflict_review_required_when: Vec<String>,
    pub merge_semantics: Vec<String>,
    pub unsupported_when: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub example_payload: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationOperationContractResponse {
    pub contract_format: String,
    pub generated_at: DateTime<Utc>,
    pub space_id: String,
    pub model: String,
    pub operation_scope: String,
    pub supported_operations: Vec<CollaborationOperationContractItem>,
    pub unsupported_policy: String,
    pub safe_rebase_policy: String,
    pub review_required_policy: String,
    pub endpoints: Vec<String>,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationCompactRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub client_id: String,
    pub since_revision: u64,
    #[serde(default)]
    pub known_revisions: Option<Vec<u64>>,
    #[serde(default)]
    pub include_payloads: Option<bool>,
    #[serde(default)]
    pub persist_package: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationCompactedOperation {
    pub compact_id: String,
    pub target_key: String,
    pub op: String,
    pub first_revision: u64,
    pub last_revision: u64,
    pub operation_count: usize,
    pub superseded_operation_ids: Vec<String>,
    pub kept_operation_id: String,
    pub conflict_risk: String,
    pub reason: String,
    pub payload: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationCompactResponse {
    pub compact_format: String,
    pub package_id: String,
    pub space_id: String,
    pub client_id: String,
    pub since_revision: u64,
    pub server_revision: u64,
    pub persisted: bool,
    pub durable_memory_id: Option<u64>,
    pub remote_operation_count: usize,
    pub compacted_operation_count: usize,
    pub dropped_operation_count: usize,
    pub unsupported_operation_count: usize,
    pub compacted_operations: Vec<CollaborationCompactedOperation>,
    pub unsupported_operations: Vec<CollaborationRemoteOperation>,
    pub required_user_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationCompactPackagesRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub package_id: Option<String>,
    #[serde(default)]
    pub client_id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationCompactPackageItem {
    pub package_id: String,
    pub space_id: String,
    pub client_id: String,
    pub status: String,
    pub since_revision: u64,
    pub server_revision: u64,
    pub remote_operation_count: usize,
    pub compacted_operation_count: usize,
    pub dropped_operation_count: usize,
    pub unsupported_operation_count: usize,
    pub conflict_risk: String,
    pub compacted_operations: Vec<CollaborationCompactedOperation>,
    pub unsupported_operations: Vec<CollaborationRemoteOperation>,
    pub required_user_actions: Vec<String>,
    pub package_event_id: String,
    pub status_event_id: Option<String>,
    pub durable_memory_id: Option<u64>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub status_note: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CollaborationCompactPackagesResponse {
    pub package_store_format: String,
    pub space_id: String,
    pub items: Vec<CollaborationCompactPackageItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationCompactPackageActionRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub package_id: String,
    pub action: String,
    #[serde(default)]
    pub actor: Option<String>,
    #[serde(default)]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CollaborationCompactPackageActionResponse {
    pub package_store_format: String,
    pub space_id: String,
    pub package_id: String,
    pub status: String,
    pub event: CollaborationEvent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationOfflineBranchCreateRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub branch_id: Option<String>,
    pub client_id: String,
    pub base_revision: u64,
    #[serde(default)]
    pub head_revision: Option<u64>,
    #[serde(default)]
    pub package_ids: Vec<String>,
    #[serde(default)]
    pub base_snapshot: Option<serde_json::Value>,
    #[serde(default)]
    pub local_snapshot: Option<serde_json::Value>,
    #[serde(default)]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationOfflineBranchesRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub branch_id: Option<String>,
    #[serde(default)]
    pub client_id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationOfflineBranchItem {
    pub branch_id: String,
    pub space_id: String,
    pub client_id: String,
    pub status: String,
    pub base_revision: u64,
    pub head_revision: Option<u64>,
    pub package_ids: Vec<String>,
    pub base_snapshot: Option<serde_json::Value>,
    pub local_snapshot: Option<serde_json::Value>,
    pub latest_server_revision: Option<u64>,
    pub branch_event_id: String,
    pub status_event_id: Option<String>,
    pub durable_memory_id: Option<u64>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub status_note: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CollaborationOfflineBranchesResponse {
    pub branch_store_format: String,
    pub space_id: String,
    pub items: Vec<CollaborationOfflineBranchItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationOfflineBranchActionRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub branch_id: String,
    pub action: String,
    #[serde(default)]
    pub actor: Option<String>,
    #[serde(default)]
    pub package_ids: Vec<String>,
    #[serde(default)]
    pub server_revision: Option<u64>,
    #[serde(default)]
    pub merge_plan: Option<serde_json::Value>,
    #[serde(default)]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CollaborationOfflineBranchActionResponse {
    pub branch_store_format: String,
    pub space_id: String,
    pub branch_id: String,
    pub status: String,
    pub event: CollaborationEvent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationAppliedOperation {
    pub operation_id: String,
    pub username: String,
    pub op: String,
    pub base_revision: u64,
    pub revision: u64,
    pub transformed: bool,
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationConflict {
    pub operation_id: String,
    pub op: String,
    pub base_revision: u64,
    pub server_revision_before_apply: u64,
    pub reason: String,
    pub transformed_payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationMergePlanRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub client_id: String,
    pub base_revision: u64,
    #[serde(default)]
    pub client_operations: Vec<ReplayOpItem>,
    #[serde(default)]
    pub base_snapshot: Option<serde_json::Value>,
    #[serde(default)]
    pub local_snapshot: Option<serde_json::Value>,
    #[serde(default)]
    pub server_snapshot: Option<serde_json::Value>,
    #[serde(default)]
    pub strategy: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationMergePlanStep {
    pub step: String,
    pub status: String,
    pub reason: String,
    pub operation: Option<CollaborationAppliedOperation>,
    pub conflict: Option<CollaborationConflict>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationMergePlanResponse {
    pub merge_plan_format: String,
    pub space_id: String,
    pub client_id: String,
    pub base_revision: u64,
    pub server_revision: u64,
    pub strategy: String,
    pub auto_mergeable: bool,
    pub transformed_operations: Vec<CollaborationAppliedOperation>,
    pub conflicts: Vec<CollaborationConflict>,
    pub steps: Vec<CollaborationMergePlanStep>,
    pub merged_snapshot: Option<serde_json::Value>,
    pub replay_order: Vec<String>,
    pub required_user_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationMergePlanApplyRequest {
    #[serde(default, alias = "conversation_id")]
    pub space_id: Option<String>,
    pub client_id: String,
    #[serde(default)]
    pub actor: Option<String>,
    pub auto_mergeable: bool,
    #[serde(default)]
    pub transformed_operations: Vec<CollaborationAppliedOperation>,
    #[serde(default)]
    pub conflicts: Vec<CollaborationConflict>,
    #[serde(default)]
    pub protected_scopes: Vec<MemoryProtectedScopeInput>,
    #[serde(default)]
    pub override_protection: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationMergePlanApplyResponse {
    pub apply_format: String,
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
    pub space_id: String,
    pub client_id: String,
    pub status: String,
    pub blocked: bool,
    pub reason: String,
    pub applied_count: usize,
    pub applied_operations: Vec<CollaborationAppliedOperation>,
    pub event_ids: Vec<String>,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationUser {
    pub username: String,
    pub color: String,
    pub x: f64,
    pub y: f64,
    pub active_node_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationEvent {
    pub event_id: String,
    pub space_id: String,
    pub event_type: String,
    pub actor: String,
    pub revision: Option<u64>,
    pub payload: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub durable_memory_id: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationEventsRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CollaborationEventsResponse {
    pub event_format: String,
    pub space_id: String,
    pub items: Vec<CollaborationEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceNode {
    pub id: String,
    pub node_type: String, // e.g., "source", "process", "decision", "memory"
    pub label: String,
    pub description: Option<String>,
    pub metadata: serde_json::Value,
    pub space_id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(default)]
    pub branch_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceEdge {
    pub id: String,
    pub from_node_id: String,
    pub to_node_id: String,
    pub relation_type: String, // e.g., "derived_from", "generated_by", "attributed_to"
    pub metadata: serde_json::Value,
    pub space_id: String,
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub branch_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceNodeCreateRequest {
    pub id: String,
    pub node_type: String,
    pub label: String,
    pub description: Option<String>,
    pub metadata: Option<serde_json::Value>,
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub branch_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ProvenanceNodeUpdateRequest {
    pub label: Option<String>,
    pub description: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceEdgeCreateRequest {
    pub id: String,
    pub from_node_id: String,
    pub to_node_id: String,
    pub relation_type: String,
    pub metadata: Option<serde_json::Value>,
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub branch_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceGraphQueryRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub node_type: Option<String>,
    pub relation_type: Option<String>,
    #[serde(default)]
    pub branch_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProvenanceGraphQueryResponse {
    pub space_id: String,
    pub nodes: Vec<ProvenanceNode>,
    pub edges: Vec<ProvenanceEdge>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProvenanceNodeUpsertResponse {
    pub provenance_graph_format: String,
    pub space_id: String,
    pub memory_id: u64,
    pub node: ProvenanceNode,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProvenanceEdgeUpsertResponse {
    pub provenance_graph_format: String,
    pub space_id: String,
    pub memory_id: u64,
    pub edge: ProvenanceEdge,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceBranchCreateRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub source_branch_id: String,
    pub new_branch_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceBranchMergeRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub source_branch_id: String,
    pub target_branch_id: String,
    #[serde(default)]
    pub preview_only: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceBranchMergeReplayStep {
    pub step: String,
    pub element_type: String,
    pub element_id: String,
    pub action: String,
    pub status: String,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proposed: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceBranchSemanticConflict {
    pub conflict_type: String,
    pub element_type: String,
    pub element_id: String,
    pub source_branch_id: String,
    pub target_branch_id: String,
    pub fields: Vec<String>,
    pub severity: String,
    pub user_copy: String,
    pub recommended_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceBranchMergeResponse {
    pub merge_preview_format: String,
    pub space_id: String,
    pub source_branch_id: String,
    pub target_branch_id: String,
    pub preview_only: bool,
    pub applied: bool,
    pub merged_nodes_count: usize,
    pub merged_edges_count: usize,
    pub conflicts: Vec<String>,
    pub semantic_conflicts: Vec<ProvenanceBranchSemanticConflict>,
    pub replay_steps: Vec<ProvenanceBranchMergeReplayStep>,
    pub transaction_boundary: String,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceBranchCommitRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub branch_id: Option<String>,
    pub base_revision: Option<u64>,
    pub actor: Option<String>,
    pub message: Option<String>,
    #[serde(default)]
    pub nodes: Vec<ProvenanceNodeCreateRequest>,
    #[serde(default)]
    pub edges: Vec<ProvenanceEdgeCreateRequest>,
    #[serde(default)]
    pub delete_node_ids: Vec<String>,
    #[serde(default)]
    pub delete_edge_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceBranchCommitResponse {
    pub provenance_commit_format: String,
    pub space_id: String,
    pub branch_id: String,
    pub commit_id: String,
    pub base_revision: u64,
    pub revision: u64,
    pub node_count: usize,
    pub edge_count: usize,
    pub deleted_node_count: usize,
    pub deleted_edge_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceBranchLogRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub branch_id: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceBranchLogEntry {
    pub commit_id: String,
    pub branch_id: String,
    pub revision: u64,
    pub actor: String,
    pub message: String,
    pub node_count: usize,
    pub edge_count: usize,
    pub deleted_node_count: usize,
    pub deleted_edge_count: usize,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceBranchLogResponse {
    pub provenance_log_format: String,
    pub space_id: String,
    pub branch_id: String,
    pub commits: Vec<ProvenanceBranchLogEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceComplexQueryRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub branch_id: Option<String>,
    pub start_node_id: Option<String>,
    pub max_hops: Option<usize>,
    pub direction: Option<String>,
    pub relation_types: Option<Vec<String>>,
    pub node_types: Option<Vec<String>>,
    pub metadata_filters: Option<std::collections::BTreeMap<String, String>>,
    pub label_regex: Option<String>,
    pub query: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceComplexQueryResponse {
    pub provenance_query_format: String,
    pub query_language: String,
    pub space_id: String,
    pub branch_id: String,
    pub start_node_id: String,
    pub direction: String,
    pub max_hops: usize,
    pub relation_types: Vec<String>,
    pub node_types: Vec<String>,
    pub metadata_filters: std::collections::BTreeMap<String, String>,
    pub label_regex: Option<String>,
    pub parsed_query: serde_json::Value,
    pub path_count: usize,
    pub nodes: Vec<ProvenanceNode>,
    pub edges: Vec<ProvenanceEdge>,
    pub explanation: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenancePlaybackRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub branch_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenancePlaybackEvent {
    pub timestamp: DateTime<Utc>,
    pub action: String,
    pub element_id: String,
    pub element_type: String,
    pub details: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenancePlaybackResponse {
    pub space_id: String,
    pub events: Vec<ProvenancePlaybackEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceRevisionStateRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub branch_id: Option<String>,
    pub revision: Option<u64>,
    pub include_tombstones: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceRevisionStateResponse {
    pub provenance_state_format: String,
    pub space_id: String,
    pub branch_id: String,
    pub revision: u64,
    pub latest_revision: u64,
    pub nodes: Vec<ProvenanceNode>,
    pub edges: Vec<ProvenanceEdge>,
    pub tombstones: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceRevisionDiffRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub branch_id: Option<String>,
    pub from_revision: u64,
    pub to_revision: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceRevisionDiffResponse {
    pub provenance_diff_format: String,
    pub space_id: String,
    pub branch_id: String,
    pub from_revision: u64,
    pub to_revision: u64,
    pub added_nodes: Vec<ProvenanceNode>,
    pub removed_nodes: Vec<ProvenanceNode>,
    pub changed_nodes: Vec<serde_json::Value>,
    pub added_edges: Vec<ProvenanceEdge>,
    pub removed_edges: Vec<ProvenanceEdge>,
    pub changed_edges: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CadenceSettingsUpsertRequest {
    pub scope_type: String,
    pub scope_id: String,
    pub recall_every_turns: Option<u64>,
    pub memory_mode: Option<String>,
}
