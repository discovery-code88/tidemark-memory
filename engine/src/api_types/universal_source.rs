use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceAdapterBoundary {
    pub boundary_format: String,
    pub adapter_id: String,
    pub adapter_version: String,
    pub input_form: String,
    pub capabilities: Vec<String>,
    pub limitations: Vec<String>,
    pub native_bytes_consumed: bool,
    pub semantic_verification: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalNativeAnchorInput {
    pub kind: String,
    #[serde(default)]
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalSourceIdentityInput {
    pub source_id: String,
    pub source_uri: String,
    pub source_kind: String,
    pub source_version: String,
    pub revision: u64,
    #[serde(default)]
    pub expected_previous_revision: Option<u64>,
    #[serde(default)]
    pub folder_id: Option<String>,
    #[serde(default)]
    pub document_id: Option<String>,
    #[serde(default)]
    pub work_id: Option<String>,
    #[serde(default)]
    pub section_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalIngestItemRequest {
    pub item_id: String,
    pub source: UniversalSourceIdentityInput,
    pub modality: String,
    pub evidence_form: String,
    pub adapter_id: String,
    pub truth_state: String,
    pub native_anchor: UniversalNativeAnchorInput,
    pub normalized_evidence: serde_json::Value,
    #[serde(default)]
    pub display_text: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub layer: Option<MemoryLayer>,
    #[serde(default)]
    pub trust: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalSourceRelationInput {
    pub from_source_id: String,
    pub to_source_id: String,
    pub relation_kind: String,
    #[serde(default = "default_relation_enabled")]
    pub enabled: bool,
}

fn default_relation_enabled() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalIngestJobSubmitRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: String,
    pub items: Vec<UniversalIngestItemRequest>,
    #[serde(default)]
    pub auto_connect_same_folder: Option<bool>,
    #[serde(default)]
    pub cross_folder_relations: Vec<UniversalSourceRelationInput>,
    #[serde(default)]
    pub max_retries: Option<usize>,
    #[serde(default)]
    pub actor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UniversalIngestProgress {
    pub total_items: usize,
    pub processed_items: usize,
    pub progress_percent: f32,
    pub current_stage: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalIngestFailedItem {
    pub item_id: String,
    pub status: String,
    pub error_code: String,
    pub error_message: String,
    pub retryable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalIngestJobRecord {
    pub job_format: String,
    pub job_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_job_id: Option<String>,
    pub tenant_id: String,
    pub workspace_id: String,
    pub conversation_id: String,
    pub status: String,
    pub stage: String,
    pub attempt: usize,
    pub max_attempts: usize,
    pub cancel_requested: bool,
    pub progress: UniversalIngestProgress,
    pub counts: WorkDispositionCounts,
    #[serde(default)]
    pub failed_items: Vec<UniversalIngestFailedItem>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuation_token: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalIngestItemRecord {
    pub job_id: String,
    pub item_id: String,
    pub status: String,
    pub stage: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_revision_id: Option<String>,
    pub requested_revision: u64,
    pub source_version: String,
    pub input_digest: String,
    pub item_request: UniversalIngestItemRequest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_result: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    pub retryable: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalIngestJobSubmissionResponse {
    pub accepted: bool,
    pub job_id: String,
    pub status: String,
    pub status_endpoint: String,
    pub cancel_endpoint: String,
    pub retry_endpoint: String,
    pub continue_endpoint: String,
    pub counts: WorkDispositionCounts,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalUploadFileInput {
    pub file_id: String,
    pub expected_bytes: u64,
    pub chunk_size: usize,
    pub expected_chunks: usize,
    pub mime_type: String,
    pub item_template: UniversalIngestItemRequest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalUploadStartRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: String,
    pub files: Vec<UniversalUploadFileInput>,
    #[serde(default)]
    pub auto_connect_same_folder: Option<bool>,
    #[serde(default)]
    pub cross_folder_relations: Vec<UniversalSourceRelationInput>,
    #[serde(default)]
    pub max_retries: Option<usize>,
    #[serde(default)]
    pub actor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalUploadFileRecord {
    pub job_id: String,
    pub file_id: String,
    pub status: String,
    pub expected_bytes: u64,
    pub chunk_size: usize,
    pub expected_chunks: usize,
    pub received_bytes: u64,
    pub received_chunks: usize,
    pub mime_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
    pub item_template: UniversalIngestItemRequest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalUploadStartResponse {
    pub accepted: bool,
    pub job_id: String,
    pub status: String,
    pub stage: String,
    pub max_chunk_bytes: usize,
    pub files: Vec<UniversalUploadFileRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalUploadChunkRequest {
    pub job_id: String,
    pub file_id: String,
    pub chunk_index: usize,
    pub byte_offset: u64,
    pub content_hex: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalUploadChunkResponse {
    pub accepted: bool,
    pub job_id: String,
    pub file: UniversalUploadFileRecord,
    pub progress: UniversalIngestProgress,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalUploadFinishRequest {
    pub job_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UniversalIngestJobListRequest {
    #[serde(default)]
    pub job_id: Option<String>,
    #[serde(default, alias = "conversation_id")]
    pub space_id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub include_items: Option<bool>,
    #[serde(default)]
    pub item_limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalIngestJobView {
    pub job: UniversalIngestJobRecord,
    #[serde(default)]
    pub source_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub items: Vec<UniversalIngestItemRecord>,
    pub items_truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalIngestJobListResponse {
    pub list_format: String,
    pub items: Vec<UniversalIngestJobView>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalIngestJobActionRequest {
    pub job_id: String,
    #[serde(default)]
    pub continuation_token: Option<String>,
    #[serde(default)]
    pub replacement_items: Vec<UniversalIngestItemRequest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalSourceDescriptor {
    pub descriptor_format: String,
    pub source_id: String,
    pub tenant_id: String,
    pub workspace_id: String,
    pub conversation_id: String,
    pub external_source_id: String,
    pub source_uri: String,
    pub source_kind: String,
    pub modality: String,
    pub evidence_form: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub folder_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub work_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub section_id: Option<String>,
    pub revision: u64,
    pub source_version: String,
    pub source_revision_id: String,
    pub content_hash: String,
    pub processing_state: String,
    pub adapter: SourceAdapterBoundary,
    pub truth_state: String,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UniversalSourceListRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: String,
    #[serde(default)]
    pub folder_id: Option<String>,
    #[serde(default)]
    pub source_ids: Vec<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalSourceListResponse {
    pub list_format: String,
    pub sources: Vec<UniversalSourceDescriptor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalSourceRelationRecord {
    pub relation_format: String,
    pub relation_id: String,
    pub tenant_id: String,
    pub workspace_id: String,
    pub conversation_id: String,
    pub from_source_id: String,
    pub to_source_id: String,
    pub relation_kind: String,
    pub origin: String,
    pub enabled: bool,
    pub revision: u64,
    pub created_by: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UniversalSourceRelationListRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: String,
    #[serde(default)]
    pub source_id: Option<String>,
    #[serde(default)]
    pub include_disabled: Option<bool>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalSourceRelationListResponse {
    pub list_format: String,
    pub relations: Vec<UniversalSourceRelationRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalSourceRelationActionRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: String,
    pub relation_id: String,
    pub enabled: bool,
    pub expected_revision: u64,
    #[serde(default)]
    pub actor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalSourceRelationCreateRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: String,
    pub from_source_id: String,
    pub to_source_id: String,
    pub relation_kind: String,
    #[serde(default = "default_relation_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub actor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnswerCitation {
    pub citation_format: String,
    pub citation_id: String,
    pub uri: String,
    pub memory_id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence_memory_id: Option<u64>,
    #[serde(default)]
    pub lineage_depth: usize,
    pub citation_role: String,
    pub source_identity: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_revision_id: Option<String>,
    pub source_version: String,
    pub source_revision: u64,
    pub modality: String,
    pub native_anchor: serde_json::Value,
    pub processing_state: String,
    pub adapter: SourceAdapterBoundary,
    pub truth_state: String,
    pub resolution_status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationResolveRequest {
    #[serde(default)]
    pub citation_id: Option<String>,
    #[serde(default)]
    pub uri: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalArtifactReadRequest {
    pub citation_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UniversalArtifactReconstructionRequest {
    #[serde(default)]
    pub citation_id: Option<String>,
    #[serde(default)]
    pub source_id: Option<String>,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub byte_start: Option<u64>,
    #[serde(default)]
    pub byte_end: Option<u64>,
    #[serde(default)]
    pub chapter_start: Option<u32>,
    #[serde(default)]
    pub chapter_end: Option<u32>,
    #[serde(default)]
    pub chapter_start_ordinal: Option<usize>,
    #[serde(default)]
    pub chapter_end_ordinal: Option<usize>,
    #[serde(default)]
    pub include_chapter_index: Option<bool>,
    #[serde(default)]
    pub download: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalArtifactChapterRange {
    pub ordinal: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chapter_number: Option<u32>,
    pub title: String,
    pub byte_start: u64,
    pub byte_end: u64,
    pub byte_length: u64,
    pub selected: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalArtifactReconstructionPlanResponse {
    pub reconstruction_format: String,
    pub source_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub citation_id: Option<String>,
    pub file_id: String,
    pub mime_type: String,
    pub source_content_sha256: String,
    pub artifact_bytes: u64,
    pub scope: String,
    pub byte_start: u64,
    pub byte_end: u64,
    pub selected_bytes: u64,
    pub chapter_index_status: String,
    pub chapter_count: usize,
    #[serde(default)]
    pub selected_chapter_ordinals: Vec<usize>,
    #[serde(default)]
    pub chapters: Vec<UniversalArtifactChapterRange>,
    pub complete: bool,
    pub integrity_status: String,
    pub stream_endpoint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationResolveResponse {
    pub resolution_format: String,
    pub citation: AnswerCitation,
    pub source: UniversalSourceDescriptor,
    pub anchor_resolved: bool,
    pub current_source_revision: bool,
    pub lifecycle_state: String,
}
