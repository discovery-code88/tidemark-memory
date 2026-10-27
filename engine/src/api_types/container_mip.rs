use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortableMemoryRecord {
    pub id: u64,
    pub conversation_id: String,
    pub layer: MemoryLayer,
    pub content: String,
    pub tokens: Vec<String>,
    pub source: String,
    pub trust: f32,
    pub created_at: DateTime<Utc>,
    pub anchor_ids: Vec<u64>,
    pub entity: Option<String>,
    pub fact_key: Option<String>,
    pub fact_value: Option<String>,
    #[serde(default)]
    pub memory_type: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub freshness: Option<f32>,
    #[serde(default)]
    pub importance: Option<f32>,
    #[serde(default)]
    pub last_verified_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub supersedes: Vec<String>,
    #[serde(default)]
    pub superseded_by: Vec<String>,
    pub version: u32,
    pub event_metadata: Option<EventMetadata>,
    pub procedure_schema: Option<ProcedureSchema>,
    pub procedure_execution_state: Option<ProcedureExecutionState>,
    pub procedure_governance: Option<ProcedureGovernance>,
    pub episodic_metadata: Option<EpisodicMetadata>,
    pub meta_memory_state: Option<MetaMemoryState>,
    pub activation_metadata: Option<ActivationMetadata>,
    pub semantic_node: Option<SemanticNode>,
    pub semantic_conflict: Option<SemanticConflictSet>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContainerBundle {
    pub bundle_version: String,
    pub exported_at: DateTime<Utc>,
    pub conversation_id: Option<String>,
    #[serde(default)]
    pub context_package: MemoryCapsuleContextPackage,
    #[serde(default)]
    pub memory_files: Vec<MemoryFileRecord>,
    pub memories: Vec<PortableMemoryRecord>,
    pub canonical_semantic_state: Vec<CanonicalSemanticItem>,
    pub consolidation_jobs: Vec<ConsolidationJob>,
    pub canonical_semantic_audit: Vec<CanonicalSemanticAuditEntry>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryCapsuleContextPackage {
    #[serde(rename = "profile.md")]
    pub profile_md: String,
    #[serde(rename = "working_context.md")]
    pub working_context_md: String,
    #[serde(rename = "decisions.jsonl")]
    pub decisions_jsonl: String,
    #[serde(rename = "constraints.jsonl")]
    pub constraints_jsonl: String,
    #[serde(rename = "timeline.jsonl")]
    pub timeline_jsonl: String,
    #[serde(rename = "open_loops.jsonl")]
    pub open_loops_jsonl: String,
    #[serde(rename = "incidents.jsonl")]
    pub incidents_jsonl: String,
    #[serde(rename = "artifacts.jsonl")]
    pub artifacts_jsonl: String,
    #[serde(rename = "memory.index.json")]
    pub memory_index_json: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInterchangeProducer {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInterchangeCapabilities {
    pub layers: Vec<String>,
    pub features: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInterchangeDeviceProfile {
    pub device_id: Option<String>,
    pub device_class: Option<String>,
    pub sync_scope: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInterchangeModelProfile {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub embedding_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInterchangePortability {
    pub device_profile: MemoryInterchangeDeviceProfile,
    pub model_profile: MemoryInterchangeModelProfile,
    pub exported_for_cross_device: bool,
    pub exported_for_cross_model: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInterchangeCompatibility {
    pub supported_protocol_versions: Vec<String>,
    pub required_features: Vec<String>,
    pub optional_features: Vec<String>,
    pub compatibility_notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInterchangeForeignMergePolicy {
    pub policy: String,
    pub identity_strategy: String,
    pub conflict_strategy: String,
    pub provenance_strategy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInterchangeIntegrity {
    pub algorithm: String,
    pub content_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInterchangeBundle {
    pub protocol_version: String,
    pub exported_at: DateTime<Utc>,
    pub conversation_id: Option<String>,
    pub producer: MemoryInterchangeProducer,
    pub capabilities: MemoryInterchangeCapabilities,
    pub merge_policy: String,
    pub integrity: MemoryInterchangeIntegrity,
    pub portability: Option<MemoryInterchangePortability>,
    pub compatibility: Option<MemoryInterchangeCompatibility>,
    pub foreign_merge: Option<MemoryInterchangeForeignMergePolicy>,
    #[serde(default)]
    pub context_package: MemoryCapsuleContextPackage,
    #[serde(default)]
    pub memory_files: Vec<MemoryFileRecord>,
    pub memories: Vec<PortableMemoryRecord>,
    pub canonical_semantic_state: Vec<CanonicalSemanticItem>,
    pub consolidation_jobs: Vec<ConsolidationJob>,
    pub canonical_semantic_audit: Vec<CanonicalSemanticAuditEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContainerExportRequest {
    pub conversation_id: Option<String>,
    pub include_jobs: Option<bool>,
    pub include_audit: Option<bool>,
    pub format: Option<String>,
    pub protocol_version: Option<String>,
    pub merge_policy: Option<String>,
    pub integrity_algorithm: Option<String>,
    pub device_profile: Option<MemoryInterchangeDeviceProfile>,
    pub model_profile: Option<MemoryInterchangeModelProfile>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContainerExportResponse {
    pub capsule_name: String,
    pub capsule_format: String,
    pub format: String,
    pub bundle_version: String,
    #[serde(default)]
    pub snapshot_id: String,
    pub exported_at: DateTime<Utc>,
    pub conversation_id: Option<String>,
    #[serde(default)]
    pub snapshot_preview: serde_json::Value,
    #[serde(default)]
    pub included_policy_summary: serde_json::Value,
    #[serde(default)]
    pub included_receipt_summary: serde_json::Value,
    #[serde(default)]
    pub pending_proposal_summary: serde_json::Value,
    #[serde(default)]
    pub restore_guidance: serde_json::Value,
    #[serde(default)]
    pub restore_conflict_review: serde_json::Value,
    pub context_package: MemoryCapsuleContextPackage,
    pub memory_files: Vec<MemoryFileRecord>,
    pub memories: Vec<PortableMemoryRecord>,
    pub canonical_semantic_state: Vec<CanonicalSemanticItem>,
    pub consolidation_jobs: Vec<ConsolidationJob>,
    pub canonical_semantic_audit: Vec<CanonicalSemanticAuditEntry>,
    pub mip_bundle: Option<MemoryInterchangeBundle>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContainerImportRequest {
    pub target_conversation_id: Option<String>,
    pub format: Option<String>,
    pub bundle: Option<MemoryContainerBundle>,
    pub mip_bundle: Option<MemoryInterchangeBundle>,
    #[serde(default)]
    pub protected_scopes: Vec<MemoryProtectedScopeInput>,
    #[serde(default)]
    pub override_protection: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContainerImportResponse {
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
    pub capsule_name: String,
    pub capsule_format: String,
    pub imported_count: usize,
    pub imported_memory_ids: Vec<u64>,
    pub imported_memory_file_count: usize,
    pub imported_memory_files: Vec<String>,
    pub target_conversation_id: Option<String>,
    pub canonical_semantic_items: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryInterchangeCapabilitiesResponse {
    pub protocol_versions: Vec<String>,
    pub producer_name: String,
    pub producer_version: String,
    pub supported_layers: Vec<String>,
    pub supported_features: Vec<String>,
    pub default_merge_policy: String,
    pub merge_policies: Vec<String>,
    pub integrity_algorithms: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInterchangeExportRequest {
    pub conversation_id: Option<String>,
    pub include_jobs: Option<bool>,
    pub include_audit: Option<bool>,
    pub protocol_version: Option<String>,
    pub merge_policy: Option<String>,
    pub integrity_algorithm: Option<String>,
    pub device_profile: Option<MemoryInterchangeDeviceProfile>,
    pub model_profile: Option<MemoryInterchangeModelProfile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInterchangeImportRequest {
    pub target_conversation_id: Option<String>,
    pub bundle: MemoryInterchangeBundle,
    #[serde(default)]
    pub protected_scopes: Vec<MemoryProtectedScopeInput>,
    #[serde(default)]
    pub override_protection: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInterchangeValidateRequest {
    pub bundle: MemoryInterchangeBundle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInterchangeNegotiationRequest {
    pub protocol_version: Option<String>,
    pub capabilities: Option<MemoryInterchangeCapabilities>,
    pub merge_policy: Option<String>,
    pub integrity_algorithm: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryInterchangeValidateResponse {
    pub compatible: bool,
    pub protocol_version: String,
    pub producer_name: String,
    pub producer_version: String,
    pub merge_policy: String,
    pub memory_count: usize,
    pub canonical_semantic_items: usize,
    pub supported_layers: Vec<String>,
    pub supported_features: Vec<String>,
    pub integrity_algorithm: String,
    pub integrity_verified: bool,
    pub compatibility_notes: Vec<String>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryInterchangeNegotiationResponse {
    pub compatible: bool,
    pub requested_protocol_version: Option<String>,
    pub negotiated_protocol_version: String,
    pub accepted_layers: Vec<String>,
    pub accepted_features: Vec<String>,
    pub unsupported_layers: Vec<String>,
    pub unsupported_features: Vec<String>,
    pub accepted_merge_policy: String,
    pub accepted_integrity_algorithm: String,
    pub compatibility_notes: Vec<String>,
    pub errors: Vec<String>,
}
