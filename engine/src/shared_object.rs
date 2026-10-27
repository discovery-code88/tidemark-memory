use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};

static CORE_OBJECT_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

fn empty_object() -> serde_json::Value {
    serde_json::json!({})
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct CoreScope {
    pub tenant_id: String,
    pub workspace_id: String,
    #[serde(default)]
    pub conversation_id: Option<String>,
}

impl CoreScope {
    pub(crate) fn normalized(&self) -> Result<Self, String> {
        let tenant_id = normalized_required_id(&self.tenant_id, "tenant_id")?;
        let workspace_id = normalized_required_id(&self.workspace_id, "workspace_id")?;
        let conversation_id = self
            .conversation_id
            .as_deref()
            .map(|value| normalized_required_id(value, "conversation_id"))
            .transpose()?;
        Ok(Self {
            tenant_id,
            workspace_id,
            conversation_id,
        })
    }

    pub(crate) fn key(&self) -> String {
        format!(
            "{}:{}:{}",
            self.tenant_id,
            self.workspace_id,
            self.conversation_id.as_deref().unwrap_or("_")
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CoreObjectWrite {
    #[serde(default)]
    pub object_id: Option<String>,
    pub kind: String,
    pub scope: CoreScope,
    #[serde(default)]
    pub source_id: Option<String>,
    #[serde(default)]
    pub source_revision_id: Option<String>,
    #[serde(default)]
    pub parent_object_id: Option<String>,
    #[serde(default)]
    pub target_object_id: Option<String>,
    #[serde(default)]
    pub external_target: Option<String>,
    #[serde(default)]
    pub predicate: Option<String>,
    #[serde(default)]
    pub relation_direction: Option<String>,
    #[serde(default)]
    pub native_anchor: Option<serde_json::Value>,
    #[serde(default)]
    pub modality: Option<String>,
    #[serde(default)]
    pub profile_kind: Option<String>,
    #[serde(default)]
    pub subject_object_id: Option<String>,
    #[serde(default)]
    pub truth_class: Option<String>,
    #[serde(default)]
    pub lifecycle_state: Option<String>,
    #[serde(default)]
    pub idempotency_key: Option<String>,
    #[serde(default)]
    pub supersedes_object_id: Option<String>,
    #[serde(default)]
    pub expected_revision: Option<i32>,
    #[serde(default)]
    pub payload_schema_version: Option<i32>,
    #[serde(default = "empty_object")]
    pub payload: serde_json::Value,
    #[serde(default = "empty_object")]
    pub provenance: serde_json::Value,
    #[serde(default)]
    pub created_by: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct NormalizedCoreObjectWrite {
    pub object_id: String,
    pub kind: String,
    pub scope: CoreScope,
    pub source_id: Option<String>,
    pub source_revision_id: Option<String>,
    pub parent_object_id: Option<String>,
    pub target_object_id: Option<String>,
    pub external_target: Option<String>,
    pub predicate: Option<String>,
    pub relation_direction: Option<String>,
    pub native_anchor: Option<serde_json::Value>,
    pub modality: Option<String>,
    pub profile_kind: Option<String>,
    pub subject_object_id: Option<String>,
    pub truth_class: String,
    pub lifecycle_state: String,
    pub idempotency_key: Option<String>,
    pub supersedes_object_id: Option<String>,
    pub expected_revision: Option<i32>,
    pub payload_schema_version: i32,
    pub payload: serde_json::Value,
    pub provenance: serde_json::Value,
    pub created_by: String,
}

impl CoreObjectWrite {
    pub(crate) fn normalized(self) -> Result<NormalizedCoreObjectWrite, String> {
        let kind = normalize_core_kind(&self.kind)?;
        let scope = self.scope.normalized()?;
        validate_payload(&self.payload)?;
        if !self.provenance.is_object() {
            return Err("provenance must be a JSON object".to_string());
        }
        let object_id = match self.object_id {
            Some(value) => normalized_required_id(&value, "object_id")?,
            None => generated_core_object_id(&kind, &scope),
        };
        let source_id = normalize_optional_id(self.source_id, "source_id")?;
        let source_revision_id =
            normalize_optional_id(self.source_revision_id, "source_revision_id")?;
        let parent_object_id = normalize_optional_id(self.parent_object_id, "parent_object_id")?;
        let target_object_id = normalize_optional_id(self.target_object_id, "target_object_id")?;
        let subject_object_id = normalize_optional_id(self.subject_object_id, "subject_object_id")?;
        let supersedes_object_id =
            normalize_optional_id(self.supersedes_object_id, "supersedes_object_id")?;
        let external_target =
            normalize_optional_text(self.external_target, "external_target", 1024)?;
        let predicate = normalize_optional_text(self.predicate, "predicate", 160)?;
        let relation_direction =
            normalize_optional_text(self.relation_direction, "relation_direction", 32)?;
        let modality = normalize_optional_text(self.modality, "modality", 80)?;
        let profile_kind = normalize_optional_text(self.profile_kind, "profile_kind", 80)?;
        let idempotency_key =
            normalize_optional_text(self.idempotency_key, "idempotency_key", 240)?;
        let created_by = normalize_optional_text(self.created_by, "created_by", 160)?
            .unwrap_or_else(|| "api".to_string());
        let truth_class = normalize_truth_class(self.truth_class.as_deref().unwrap_or("observed"))?;
        let lifecycle_state =
            normalize_lifecycle_state(self.lifecycle_state.as_deref().unwrap_or("active"))?;
        let payload_schema_version = self.payload_schema_version.unwrap_or(1);
        if payload_schema_version <= 0 {
            return Err("payload_schema_version must be positive".to_string());
        }
        if let Some(expected_revision) = self.expected_revision {
            if expected_revision <= 0 {
                return Err("expected_revision must be positive".to_string());
            }
        }

        validate_kind_requirements(
            &kind,
            source_id.as_deref(),
            source_revision_id.as_deref(),
            parent_object_id.as_deref(),
            target_object_id.as_deref(),
            external_target.as_deref(),
            predicate.as_deref(),
            self.native_anchor.as_ref(),
            modality.as_deref(),
            profile_kind.as_deref(),
            subject_object_id.as_deref(),
            &self.payload,
            &truth_class,
        )?;

        Ok(NormalizedCoreObjectWrite {
            object_id,
            kind,
            scope,
            source_id,
            source_revision_id,
            parent_object_id,
            target_object_id,
            external_target,
            predicate,
            relation_direction,
            native_anchor: self.native_anchor,
            modality,
            profile_kind,
            subject_object_id,
            truth_class,
            lifecycle_state,
            idempotency_key,
            supersedes_object_id,
            expected_revision: self.expected_revision,
            payload_schema_version,
            payload: self.payload,
            provenance: self.provenance,
            created_by,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct CoreObjectRecord {
    pub object_id: String,
    pub kind: String,
    pub scope: CoreScope,
    pub source_id: Option<String>,
    pub source_revision_id: Option<String>,
    pub truth_class: String,
    pub lifecycle_state: String,
    pub revision: i32,
    pub payload_schema_version: i32,
    pub payload: serde_json::Value,
    pub provenance: serde_json::Value,
    pub supersedes_object_id: Option<String>,
    pub created_by: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct CoreObjectListRequest {
    #[serde(default)]
    pub tenant_id: Option<String>,
    #[serde(default)]
    pub workspace_id: Option<String>,
    #[serde(default)]
    pub conversation_id: Option<String>,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub source_id: Option<String>,
    #[serde(default)]
    pub object_id: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct CoreObjectListResponse {
    pub objects: Vec<CoreObjectRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProjectionOutboxProcessRequest {
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub max_attempts: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ProjectionOutboxProcessResponse {
    pub claimed: usize,
    pub completed: usize,
    pub retried: usize,
    pub dead_lettered: usize,
    pub superseded: usize,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProjectionOutboxListRequest {
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ProjectionOutboxJob {
    pub outbox_id: String,
    pub projection_kind: String,
    pub object_id: String,
    pub source_revision_id: Option<String>,
    pub memory_id: Option<u64>,
    pub content_hash: String,
    pub payload: serde_json::Value,
    pub status: String,
    pub attempt_count: u32,
    pub next_attempt_at: DateTime<Utc>,
    pub last_error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ProjectionOutboxListResponse {
    pub jobs: Vec<ProjectionOutboxJob>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CoreArtifactStoreRequest {
    pub scope: CoreScope,
    pub mime_type: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub content_utf8: Option<String>,
    #[serde(default)]
    pub content_hex: Option<String>,
    #[serde(default)]
    pub storage_uri: Option<String>,
    #[serde(default)]
    pub content_sha256: Option<String>,
    #[serde(default)]
    pub retention_class: Option<String>,
    #[serde(default)]
    pub created_by: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct CoreArtifactStoreResponse {
    pub artifact_manifest: CoreObjectRecord,
    pub storage_uri: String,
    pub content_sha256: String,
    pub byte_length: u64,
    pub storage_backend: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CoreArtifactReadRequest {
    pub scope: CoreScope,
    pub artifact_object_id: String,
    #[serde(default)]
    pub max_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct CoreArtifactReadResponse {
    pub artifact_format: String,
    pub artifact_object_id: String,
    pub mime_type: String,
    pub content_sha256: String,
    pub byte_length: u64,
    pub content_utf8: Option<String>,
    pub content_hex: Option<String>,
    pub storage_backend: String,
    pub authority: String,
}

#[derive(Debug, Clone)]
pub(crate) struct StoredArtifact {
    pub storage_uri: String,
    pub content_sha256: String,
    pub byte_length: u64,
    pub storage_backend: String,
}

pub(crate) fn stable_digest(value: impl AsRef<[u8]>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_ref());
    hex::encode(hasher.finalize())
}

pub(crate) fn content_hash(value: impl AsRef<[u8]>) -> String {
    format!("sha256:{}", stable_digest(value))
}

pub(crate) fn generated_core_object_id(kind: &str, scope: &CoreScope) -> String {
    let counter = CORE_OBJECT_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let seed = format!(
        "{kind}:{}:{}:{counter}",
        scope.key(),
        Utc::now().timestamp_nanos_opt().unwrap_or_default()
    );
    format!("{kind}_{}", &stable_digest(seed)[..32])
}

pub(crate) fn deterministic_core_object_id(prefix: &str, value: &str) -> String {
    format!("{prefix}_{}", &stable_digest(value)[..32])
}

pub(crate) fn normalize_core_kind(value: &str) -> Result<String, String> {
    let normalized = value.trim().to_ascii_lowercase();
    if !matches!(
        normalized.as_str(),
        "source"
            | "source_revision"
            | "artifact_manifest"
            | "observation"
            | "entity"
            | "episode"
            | "episode_member"
            | "relation"
            | "relation_evidence"
            | "profile"
            | "preference"
            | "operational_habit"
            | "task"
            | "action"
            | "outcome"
            | "feedback"
            | "correction"
            | "policy_receipt"
            | "memory_record"
    ) {
        return Err(format!("unsupported core object kind `{value}`"));
    }
    Ok(normalized)
}

pub(crate) fn normalize_truth_class(value: &str) -> Result<String, String> {
    let normalized = value.trim().to_ascii_lowercase();
    if !matches!(
        normalized.as_str(),
        "observed"
            | "measured"
            | "reported"
            | "computed"
            | "inferred"
            | "generated"
            | "predicted"
            | "simulated"
            // Retained only for CORE-004 compatibility. New action intent and
            // authority state live in the CORE-005 action ledger.
            | "proposed"
    ) {
        return Err(format!("unsupported truth_class `{value}`"));
    }
    Ok(normalized)
}

pub(crate) fn normalize_lifecycle_state(value: &str) -> Result<String, String> {
    let normalized = value.trim().to_ascii_lowercase();
    if !matches!(
        normalized.as_str(),
        "candidate" | "active" | "held" | "superseded" | "deleted" | "archived"
    ) {
        return Err(format!("unsupported lifecycle_state `{value}`"));
    }
    Ok(normalized)
}

pub(crate) fn infer_native_modality(source: &str, content: &str) -> &'static str {
    let source = source.to_ascii_lowercase();
    let content = content.to_ascii_lowercase();
    if source.contains("multimodal:image") || source.contains("screenshot") {
        "image"
    } else if source.contains("multimodal:audio") {
        "audio"
    } else if source.contains("multimodal:video") {
        "video"
    } else if source.contains("multimodal:document") || source.contains("multimodal:ocr") {
        "document"
    } else {
        "text"
    }
}

fn normalized_required_id(value: &str, field: &str) -> Result<String, String> {
    let value = value.trim();
    if value.is_empty() || value.len() > 160 {
        return Err(format!("{field} must contain 1-160 characters"));
    }
    if !value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | ':' | '/'))
    {
        return Err(format!("{field} contains unsupported characters"));
    }
    Ok(value.to_string())
}

fn normalize_optional_id(value: Option<String>, field: &str) -> Result<Option<String>, String> {
    value
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .map(|value| normalized_required_id(value, field))
        .transpose()
}

fn normalize_optional_text(
    value: Option<String>,
    field: &str,
    max_len: usize,
) -> Result<Option<String>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    let value = value.trim();
    if value.is_empty() {
        return Ok(None);
    }
    if value.len() > max_len {
        return Err(format!("{field} exceeds {max_len} characters"));
    }
    Ok(Some(value.to_string()))
}

fn validate_payload(payload: &serde_json::Value) -> Result<(), String> {
    let Some(object) = payload.as_object() else {
        return Err("payload must be a JSON object".to_string());
    };
    for forbidden in [
        "tenant_id",
        "workspace_id",
        "conversation_id",
        "scope",
        "source_id",
        "source_revision_id",
        "supersedes_object_id",
        "correction_of_object_id",
        "revision",
        "truth_class",
        "lifecycle_state",
    ] {
        if object.contains_key(forbidden) {
            return Err(format!(
                "payload cannot carry authoritative `{forbidden}`; use the typed core field"
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_kind_requirements(
    kind: &str,
    source_id: Option<&str>,
    source_revision_id: Option<&str>,
    parent_object_id: Option<&str>,
    target_object_id: Option<&str>,
    external_target: Option<&str>,
    predicate: Option<&str>,
    native_anchor: Option<&serde_json::Value>,
    modality: Option<&str>,
    profile_kind: Option<&str>,
    subject_object_id: Option<&str>,
    payload: &serde_json::Value,
    truth_class: &str,
) -> Result<(), String> {
    match kind {
        "source" => {
            if payload
                .get("source_uri")
                .and_then(serde_json::Value::as_str)
                .is_none()
            {
                return Err("source payload requires non-empty `source_uri`".to_string());
            }
        }
        "source_revision" => {
            if source_id.is_none() {
                return Err("source_revision requires source_id".to_string());
            }
            if payload
                .get("content_hash")
                .and_then(serde_json::Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .is_none()
            {
                return Err("source_revision payload requires immutable `content_hash`".to_string());
            }
            if !matches!(truth_class, "observed" | "reported") {
                return Err("source_revision truth_class must be observed or reported".to_string());
            }
        }
        "artifact_manifest" => {
            for field in ["storage_uri", "content_hash", "byte_length", "mime_type"] {
                if payload.get(field).is_none() {
                    return Err(format!("artifact_manifest payload requires `{field}`"));
                }
            }
        }
        "observation" => {
            if source_revision_id.is_none() || native_anchor.is_none() || modality.is_none() {
                return Err(
                    "observation requires source_revision_id, native_anchor, and modality"
                        .to_string(),
                );
            }
        }
        "entity" => {
            if payload
                .get("scoped_identity")
                .and_then(serde_json::Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .is_none()
            {
                return Err("entity payload requires `scoped_identity`".to_string());
            }
        }
        "episode_member" => {
            if parent_object_id.is_none() || target_object_id.is_none() {
                return Err(
                    "episode_member requires parent_object_id and target_object_id".to_string(),
                );
            }
        }
        "relation" => {
            if source_id.is_none() && source_revision_id.is_none() {
                return Err(
                    "relation requires source_id or source_revision_id provenance".to_string(),
                );
            }
            if target_object_id.is_none() && external_target.is_none() {
                return Err("relation requires target_object_id or external_target".to_string());
            }
            if predicate.is_none() {
                return Err("relation requires predicate".to_string());
            }
            if parent_object_id.is_none() {
                return Err("relation requires parent_object_id as source endpoint".to_string());
            }
        }
        "relation_evidence" => {
            if parent_object_id.is_none()
                || (source_revision_id.is_none() && target_object_id.is_none())
            {
                return Err(
                    "relation_evidence requires parent_object_id and source_revision_id or target_object_id"
                        .to_string(),
                );
            }
        }
        "profile" | "preference" | "operational_habit" => {
            if profile_kind.is_none() {
                return Err(
                    "profile/preference/operational_habit requires profile_kind".to_string()
                );
            }
            if subject_object_id.is_none() {
                return Err(
                    "profile/preference/operational_habit requires subject_object_id".to_string(),
                );
            }
        }
        "task" | "outcome" => {
            if payload
                .get("status")
                .and_then(serde_json::Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .is_none()
            {
                return Err(format!("{kind} payload requires `status`"));
            }
        }
        "action" => {
            let status = payload
                .get("status")
                .and_then(serde_json::Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| "action payload requires `status`".to_string())?;
            if !status.trim().eq_ignore_ascii_case("proposed") {
                return Err(
                    "action status authority is ledger-owned; generic core action writes may only create `proposed` declarations"
                        .to_string(),
                );
            }
        }
        "feedback" | "correction" => {
            if target_object_id.is_none() {
                return Err("feedback/correction requires target_object_id".to_string());
            }
        }
        "policy_receipt" => {
            if payload
                .get("decision")
                .and_then(serde_json::Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .is_none()
            {
                return Err("policy_receipt payload requires `decision`".to_string());
            }
        }
        "memory_record" | "episode" => {}
        _ => return Err(format!("unsupported core object kind `{kind}`")),
    }
    Ok(())
}
