use super::*;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MultimodalKind {
    Document,
    Image,
    Audio,
    Video,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DocumentFormat {
    Text,
    Markdown,
    Pdf,
    #[serde(other)]
    Unsupported,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePrivacyRedaction {
    pub target: String,
    pub action: String,
    pub label: Option<String>,
    pub x: Option<f32>,
    pub y: Option<f32>,
    pub width: Option<f32>,
    pub height: Option<f32>,
    pub start_seconds: Option<f64>,
    pub end_seconds: Option<f64>,
    pub start_page: Option<u32>,
    pub end_page: Option<u32>,
    pub start_char: Option<u64>,
    pub end_char: Option<u64>,
    pub field_path: Option<String>,
    pub precision_meters: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePrivacyPolicy {
    pub purpose: String,
    pub lawful_basis: String,
    pub retention_days: u32,
    pub local_only: bool,
    #[serde(default)]
    pub sensitivity_classes: Vec<String>,
    #[serde(default)]
    pub source_redacted: bool,
    #[serde(default)]
    pub allow_raw_sensitive: bool,
    pub consent_reference: Option<String>,
    #[serde(default)]
    pub redactions: Vec<NativePrivacyRedaction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalIngestRequest {
    pub conversation_id: String,
    pub kind: MultimodalKind,
    pub layer: Option<MemoryLayer>,
    pub source: Option<String>,
    pub trust: Option<f32>,
    pub content: Option<String>,
    pub title: Option<String>,
    pub uri: Option<String>,
    pub document_format: Option<DocumentFormat>,
    pub caption: Option<String>,
    pub mime_type: Option<String>,
    pub alt_text: Option<String>,
    #[serde(default)]
    pub privacy: Option<NativePrivacyPolicy>,
    #[serde(default)]
    pub auto_connect_folder: Option<bool>,
    #[serde(default)]
    pub linked_folders: Vec<String>,
    #[serde(default)]
    pub protected_scopes: Vec<MemoryProtectedScopeInput>,
    #[serde(default)]
    pub override_protection: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalBatchIngestRequest {
    pub items: Vec<MultimodalIngestRequest>,
    #[serde(default)]
    pub protected_scopes: Vec<MemoryProtectedScopeInput>,
    #[serde(default)]
    pub override_protection: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MultimodalBatchGovernance {
    pub max_items: usize,
    pub max_total_content_chars: usize,
    pub requested_items: usize,
    pub total_content_chars: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MultimodalDerivedItem {
    pub layer: MemoryLayer,
    pub normalized_source: String,
    pub content_preview: String,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct MultimodalIngestResponse {
    pub accepted: bool,
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
    pub kind: MultimodalKind,
    pub normalized_source: String,
    pub created_memory_ids: Vec<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub privacy_receipt_memory_id: Option<u64>,
    pub source_connection_records: usize,
    pub derived_items: Vec<MultimodalDerivedItem>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MultimodalBatchIngestResponse {
    pub accepted: bool,
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
    pub imported_items: usize,
    pub created_memory_ids: Vec<u64>,
    pub source_connection_records: usize,
    pub items: Vec<MultimodalIngestResponse>,
    pub governance: MultimodalBatchGovernance,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalExtractRequest {
    pub conversation_id: String,
    pub kind: MultimodalKind,
    pub layer: Option<MemoryLayer>,
    pub trust: Option<f32>,
    pub uri: String,
    pub title: Option<String>,
    pub mime_type: Option<String>,
    pub size_bytes: Option<usize>,
    pub duration_seconds: Option<u64>,
    #[serde(default)]
    pub privacy: Option<NativePrivacyPolicy>,
    #[serde(default)]
    pub protected_scopes: Vec<MemoryProtectedScopeInput>,
    #[serde(default)]
    pub override_protection: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalExtractionRunsRequest {
    pub conversation_id: Option<String>,
    pub kind: Option<MultimodalKind>,
    pub status: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalExtractionRunRecord {
    pub extraction_id: String,
    pub conversation_id: String,
    pub kind: MultimodalKind,
    pub uri: String,
    pub title: Option<String>,
    pub mime_type: Option<String>,
    pub status: String,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub size_bytes: Option<usize>,
    pub duration_seconds: Option<u64>,
    pub transcript_preview: String,
    pub normalized_source: Option<String>,
    pub created_memory_ids: Vec<u64>,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MultimodalExtractResponse {
    pub accepted: bool,
    pub extraction_id: String,
    pub status: String,
    pub kind: MultimodalKind,
    pub normalized_source: String,
    pub created_memory_ids: Vec<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub privacy_receipt_memory_id: Option<u64>,
    pub transcript_preview: String,
    pub resource_controls: serde_json::Value,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MultimodalExtractionRunsResponse {
    pub total: usize,
    pub items: Vec<MultimodalExtractionRunRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OcrUploadResponse {
    pub accepted: bool,
    pub image_source: String,
    pub ocr_source: String,
    pub created_memory_ids: Vec<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub privacy_receipt_memory_id: Option<u64>,
    pub ocr_text_preview: String,
    pub deleted_temp_file: bool,
    pub reason: String,
}
