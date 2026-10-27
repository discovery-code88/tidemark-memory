use super::*;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryListRequest {
    pub workspace_id: Option<String>,
    pub conversation_id: Option<String>,
    pub layer: Option<MemoryLayer>,
    pub text: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryFolderSummary {
    pub workspace_id: String,
    pub display_name: Option<String>,
    pub tenant_id: String,
    pub conversation_count: usize,
    pub memory_count: usize,
    pub conversations: Vec<WorkspaceConversationSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryFolderSuggestion {
    pub workspace_id: String,
    pub display_name: String,
    pub reason: String,
    pub confidence: f32,
    pub conversation_ids: Vec<String>,
    pub memory_count: usize,
    pub matched_terms: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryFactVersionItem {
    pub id: u64,
    pub content: String,
    pub source: String,
    pub trust: f32,
    pub created_at: DateTime<Utc>,
    pub version: u32,
    pub fact_value: Option<String>,
    pub status: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryBrowserItem {
    pub id: u64,
    pub workspace_id: String,
    pub conversation_id: String,
    pub topic_key: String,
    pub topic_group: String,
    pub layer: MemoryLayer,
    pub content: String,
    pub source: String,
    pub trust: f32,
    pub created_at: DateTime<Utc>,
    pub version: u32,
    pub entity: Option<String>,
    pub fact_key: Option<String>,
    pub fact_value: Option<String>,
    pub memory_type: String,
    pub confidence: Option<f32>,
    pub freshness: Option<f32>,
    pub importance: Option<f32>,
    pub last_verified_at: Option<DateTime<Utc>>,
    pub supersedes: Vec<String>,
    pub superseded_by: Vec<String>,
    pub canonical_semantic_id: Option<String>,
    pub semantic_node: Option<SemanticNode>,
    pub semantic_conflict: Option<SemanticConflictSet>,
    pub meta_memory_state: Option<MetaMemoryState>,
    pub fact_timeline: Vec<MemoryFactVersionItem>,
    pub conflict_detected: bool,
    pub conflict_label: Option<String>,
    pub review_status: Option<String>,
    pub review_label: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryListResponse {
    pub generated_at: DateTime<Utc>,
    pub total: usize,
    pub limit: usize,
    pub offset: usize,
    pub has_more: bool,
    pub layer_counts: HashMap<String, usize>,
    pub folders: Vec<MemoryFolderSummary>,
    pub folder_suggestions: Vec<MemoryFolderSuggestion>,
    pub items: Vec<MemoryBrowserItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryAutoOrganizeRequest {
    pub min_confidence: Option<f32>,
    pub dry_run: Option<bool>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryAutoOrganizeResponse {
    pub dry_run: bool,
    pub min_confidence: f32,
    pub applied_count: usize,
    pub skipped_count: usize,
    pub suggestions: Vec<MemoryFolderSuggestion>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryUpdateRequest {
    pub memory_id: u64,
    pub content: Option<String>,
    pub source: Option<String>,
    pub trust: Option<f32>,
    pub entity: Option<String>,
    pub fact_key: Option<String>,
    pub fact_value: Option<String>,
    pub memory_type: Option<String>,
    pub confidence: Option<f32>,
    pub freshness: Option<f32>,
    pub importance: Option<f32>,
    pub last_verified_at: Option<DateTime<Utc>>,
    pub supersedes: Option<Vec<String>>,
    pub superseded_by: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryUpdateResponse {
    pub updated: bool,
    pub item: Option<MemoryBrowserItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryDeleteRequest {
    pub memory_id: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryDeleteResponse {
    pub deleted: bool,
    pub memory_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryBatchUpdateRequest {
    pub memory_ids: Vec<u64>,
    pub delete: Option<bool>,
    pub merge_conflicts: Option<bool>,
    pub conflict_winner_strategy: Option<String>,
    pub source: Option<String>,
    pub trust: Option<f32>,
    pub entity: Option<String>,
    pub fact_key: Option<String>,
    pub fact_value: Option<String>,
    pub workspace_id: Option<String>,
    pub workspace_display_name: Option<String>,
    #[serde(default)]
    pub protected_scopes: Vec<MemoryProtectedScopeInput>,
    #[serde(default)]
    pub override_protection: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryBatchUpdateResponse {
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
    pub updated_count: usize,
    pub deleted_count: usize,
    pub merged_conflict_count: usize,
    pub moved_conversation_count: usize,
    pub workspace_id: Option<String>,
    pub conversation_ids: Vec<String>,
    pub items: Vec<MemoryBrowserItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryFolderLlmNameRequest {
    pub memory_ids: Vec<u64>,
    pub language: Option<String>,
    pub provider_url: Option<String>,
    pub model: Option<String>,
    #[serde(default)]
    pub explicit_provider_approval: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryFolderLlmNameResponse {
    pub workspace_id: String,
    pub display_name: String,
    pub reason: String,
    pub provider_url: String,
    pub model: String,
    pub used_external_llm: bool,
    pub sample_memory_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NovelImportExtractRequest {
    pub title: Option<String>,
    pub conversation_id: String,
    pub file_name: Option<String>,
    pub text: String,
    pub import_archive: Option<bool>,
    pub extract_world_book: Option<bool>,
    pub dry_run: Option<bool>,
    pub include_open_questions: Option<bool>,
    pub include_style_profile: Option<bool>,
    #[serde(default)]
    pub chapter_summary_steps: Option<usize>,
    #[serde(default)]
    pub auto_connect_folder: Option<bool>,
    #[serde(default)]
    pub linked_folders: Vec<String>,
    pub chunk_chars: Option<usize>,
    pub max_chunks: Option<usize>,
    pub trust: Option<f32>,
    pub provider_url: Option<String>,
    pub model: Option<String>,
    #[serde(default)]
    pub explicit_provider_approval: Option<bool>,
    #[serde(default)]
    pub protected_scopes: Vec<MemoryProtectedScopeInput>,
    #[serde(default)]
    pub override_protection: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NovelImportExtractResponse {
    pub title: String,
    pub conversation_id: String,
    pub dry_run: bool,
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
    pub archive_records: usize,
    pub chapter_summary_records: usize,
    pub chapter_summary_steps: usize,
    pub max_chapter_summary_steps: usize,
    pub style_profile_records: usize,
    pub document_connection_records: usize,
    pub extracted_records: usize,
    pub final_structured_records: usize,
    pub deduplicated_count: usize,
    pub arbitrated_count: usize,
    pub accepted_count: usize,
    pub rejected_count: usize,
    pub failed_count: usize,
    pub memory_ids: Vec<u64>,
    pub errors: Vec<String>,
    pub used_external_llm: bool,
    pub model: Option<String>,
}

pub type LongDocumentImportExtractRequest = NovelImportExtractRequest;
pub type LongDocumentImportExtractResponse = NovelImportExtractResponse;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NovelReadingModelReviewRequest {
    pub conversation_id: String,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NovelReadingModelItem {
    pub memory_id: u64,
    pub label: String,
    pub entity: Option<String>,
    pub fact_key: Option<String>,
    pub fact_value: Option<String>,
    pub source: String,
    pub trust: f32,
    pub evidence_preview: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NovelReadingModelCoverageItem {
    pub surface: String,
    pub status: String,
    pub count: usize,
    pub description: String,
    pub next_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NovelReadingModelReviewResponse {
    pub review_format: String,
    pub conversation_id: String,
    pub title: Option<String>,
    pub status: String,
    pub total_memory_count: usize,
    pub archive_record_count: usize,
    pub chapter_summary_count: usize,
    pub character_count: usize,
    pub location_count: usize,
    pub world_rule_count: usize,
    pub timeline_event_count: usize,
    pub relationship_count: usize,
    pub open_question_count: usize,
    pub style_profile_count: usize,
    pub coverage: Vec<NovelReadingModelCoverageItem>,
    pub chapter_summaries: Vec<NovelReadingModelItem>,
    pub characters: Vec<NovelReadingModelItem>,
    pub locations: Vec<NovelReadingModelItem>,
    pub timeline_events: Vec<NovelReadingModelItem>,
    pub relationships: Vec<NovelReadingModelItem>,
    pub open_questions: Vec<NovelReadingModelItem>,
    pub style_profiles: Vec<NovelReadingModelItem>,
    pub traceable_answer_endpoint: String,
    pub correction_endpoint: String,
    pub import_endpoint: String,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NovelReadingModelCorrectRequest {
    pub conversation_id: String,
    pub memory_id: u64,
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub activation_scope: Option<String>,
    #[serde(default)]
    pub surface: Option<String>,
    #[serde(default)]
    pub label: Option<String>,
    pub corrected_summary: String,
    #[serde(default)]
    pub storylines: Vec<String>,
    #[serde(default)]
    pub key_events: Vec<String>,
    #[serde(default)]
    pub evidence: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub source_note: Option<String>,
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
pub struct NovelReadingModelCorrectResponse {
    pub correction_format: String,
    pub conversation_id: String,
    pub dry_run: bool,
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
    pub surface: String,
    pub source_memory_id: u64,
    pub replacement_memory_id: Option<u64>,
    pub affected_memory_ids: Vec<u64>,
    pub old_preview: String,
    pub new_preview: String,
    pub proposed_record: IngestRequest,
    pub correction_endpoint: String,
    pub review_endpoint: String,
    pub traceable_answer_endpoint: String,
    pub rollback_guidance: String,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NovelChapterSummaryRefreshRequest {
    pub conversation_id: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub section: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub trust: Option<f32>,
    #[serde(default)]
    pub dry_run: Option<bool>,
    #[serde(default)]
    pub chapter_summary_steps: Option<usize>,
    #[serde(default)]
    pub protected_scopes: Vec<MemoryProtectedScopeInput>,
    #[serde(default)]
    pub override_protection: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NovelChapterSummaryRefreshPreview {
    pub section: String,
    pub source_chunk_count: usize,
    pub source_summary_memory_ids: Vec<u64>,
    pub replacement_preview: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NovelChapterSummaryRefreshResponse {
    pub refresh_format: String,
    pub conversation_id: String,
    pub dry_run: bool,
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
    pub title: String,
    pub archive_sections: usize,
    pub existing_summary_records: usize,
    pub regenerated_records: usize,
    pub superseded_count: usize,
    pub replacement_memory_ids: Vec<u64>,
    pub superseded_memory_ids: Vec<u64>,
    pub previews: Vec<NovelChapterSummaryRefreshPreview>,
    pub refresh_endpoint: String,
    pub review_endpoint: String,
    pub traceable_answer_endpoint: String,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NovelChapterSourceLinkValidateRequest {
    pub conversation_id: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub section: Option<String>,
    #[serde(default)]
    pub trust: Option<f32>,
    #[serde(default)]
    pub dry_run: Option<bool>,
    #[serde(default)]
    pub chapter_summary_steps: Option<usize>,
    #[serde(default)]
    pub protected_scopes: Vec<MemoryProtectedScopeInput>,
    #[serde(default)]
    pub override_protection: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NovelChapterSourceLinkIssue {
    pub section: String,
    pub summary_memory_id: u64,
    pub status: String,
    pub source_chunk_count: usize,
    pub source_archive_memory_ids: Vec<u64>,
    pub has_explicit_source_ids: bool,
    pub incomplete_summary: bool,
    pub repair_action: String,
    pub repaired: bool,
    pub replacement_memory_id: Option<u64>,
    pub summary_preview: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NovelChapterSourceLinkValidateResponse {
    pub validation_format: String,
    pub conversation_id: String,
    pub dry_run: bool,
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
    pub title: String,
    pub checked_summary_records: usize,
    pub archive_sections: usize,
    pub ok_count: usize,
    pub missing_archive_count: usize,
    pub incomplete_summary_count: usize,
    pub missing_explicit_source_ids_count: usize,
    pub regenerated_count: usize,
    pub superseded_count: usize,
    pub replacement_memory_ids: Vec<u64>,
    pub superseded_memory_ids: Vec<u64>,
    pub issues: Vec<NovelChapterSourceLinkIssue>,
    pub validate_endpoint: String,
    pub refresh_endpoint: String,
    pub traceable_answer_endpoint: String,
    pub next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NovelImportJobRecord {
    pub job_id: String,
    pub title: String,
    pub conversation_id: String,
    pub status: String,
    pub total_chunks: usize,
    pub processed_chunks: usize,
    pub progress_percent: f32,
    pub response: Option<NovelImportExtractResponse>,
    pub errors: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NovelImportJobSubmitResponse {
    pub job_id: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NovelImportJobListRequest {
    pub job_id: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NovelImportJobListResponse {
    pub items: Vec<NovelImportJobRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NovelImportJobActionRequest {
    pub job_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NovelUploadStartRequest {
    pub title: Option<String>,
    pub conversation_id: String,
    pub file_name: Option<String>,
    pub total_chunks: usize,
    pub import_archive: Option<bool>,
    pub extract_world_book: Option<bool>,
    pub dry_run: Option<bool>,
    pub include_open_questions: Option<bool>,
    pub include_style_profile: Option<bool>,
    #[serde(default)]
    pub chapter_summary_steps: Option<usize>,
    #[serde(default)]
    pub auto_connect_folder: Option<bool>,
    #[serde(default)]
    pub linked_folders: Vec<String>,
    pub chunk_chars: Option<usize>,
    pub trust: Option<f32>,
    pub provider_url: Option<String>,
    pub model: Option<String>,
    #[serde(default)]
    pub explicit_provider_approval: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NovelUploadStartResponse {
    pub upload_id: String,
    pub total_chunks: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NovelUploadChunkRequest {
    pub upload_id: String,
    pub chunk_index: usize,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct NovelUploadChunkResponse {
    pub upload_id: String,
    pub received_chunks: usize,
    pub total_chunks: usize,
    pub complete: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NovelUploadFinishRequest {
    pub upload_id: String,
}
