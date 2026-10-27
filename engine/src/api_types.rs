use crate::semantic::{SemanticConflictSet, SemanticNode};
use crate::{
    ActivationMetadata, EpisodicMetadata, EventMetadata, MemoryLayer,
    MetaMemoryState, ProcedureExecutionState, ProcedureGovernance, ProcedureSchema,
};
use axum::{http::StatusCode, Json};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone, Serialize)]
pub struct MetricsResponse {
    pub total_requests: u64,
    pub unauthorized_requests: u64,
    pub forbidden_requests: u64,
    pub rate_limited_requests: u64,
    pub timeout_requests: u64,
    pub server_error_requests: u64,
    pub db_unavailable_events: u64,
    pub requests_per_second_60s: f64,
    pub p95_latency_ms: u64,
    pub embedding_status_ready: u64,
    pub embedding_status_missing: u64,
    pub readiness_ready: u64,
    pub readiness_degraded_checks: u64,
    pub shutdown_active: u64,
    pub postgres_configured: u64,
    pub migration_files_up: u64,
    pub migration_files_down: u64,
    pub workspace_registry_items: u64,
    pub control_plane_policy_assignments: u64,
    pub control_plane_denied_events: u64,
    pub connector_failing_schedules: u64,
    pub connector_backoff_schedules: u64,
    pub backup_restore_ok: u64,
    pub backup_restore_degraded: u64,
    pub backup_restore_not_configured: u64,
    pub backup_restore_findings: u64,
    pub security_posture_ok: u64,
    pub security_posture_degraded: u64,
    pub security_posture_not_configured: u64,
    pub security_posture_findings: u64,
    pub ingest_requests: u64,
    pub ingest_accepted: u64,
    pub ingest_rejected: u64,
    pub query_requests: u64,
    pub query_result_items: u64,
    pub answer_requests: u64,
    pub answer_evidence_items: u64,
    pub conflicts_detected: u64,
    pub context_decide_requests: u64,
    pub context_build_requests: u64,
    pub context_status_skipped: u64,
    pub context_status_cached: u64,
    pub context_status_fresh: u64,
    pub context_mode_off: u64,
    pub context_mode_cached: u64,
    pub context_mode_auto: u64,
    pub context_mode_fresh: u64,
    pub context_mode_full: u64,
    pub context_mode_ask: u64,
    pub context_p95_latency_ms: u64,
    pub memory_type_fact: u64,
    pub memory_type_decision: u64,
    pub memory_type_constraint: u64,
    pub memory_type_todo: u64,
    pub memory_type_incident: u64,
    pub memory_type_preference: u64,
    pub memory_type_artifact: u64,
    pub memory_quality_low_confidence: u64,
    pub memory_quality_stale: u64,
    pub memory_quality_superseded: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiErrorResponse {
    pub error: String,
    pub error_code: String,
    pub message: String,
    pub details: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestRequest {
    pub conversation_id: String,
    pub layer: MemoryLayer,
    pub content: String,
    pub source: Option<String>,
    pub trust: Option<f32>,
    pub entity: Option<String>,
    pub fact_key: Option<String>,
    pub fact_value: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRetainRequest {
    #[serde(alias = "conversation_id")]
    pub space_id: Option<String>,
    pub layer: MemoryLayer,
    pub content: String,
    pub source: Option<String>,
    pub trust: Option<f32>,
    pub entity: Option<String>,
    pub fact_key: Option<String>,
    pub fact_value: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentScopedIngestRequest {
    pub conversation_id: String,
    pub layer: MemoryLayer,
    pub content: String,
    pub source: Option<String>,
    pub trust: Option<f32>,
    pub entity: Option<String>,
    pub fact_key: Option<String>,
    pub fact_value: Option<String>,
    pub agent_id: Option<String>,
    pub shared_block: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTraceVectorRetrieval {
    pub diagnostics_format: String,
    pub backend: String,
    pub enabled: bool,
    pub candidate_count: usize,
    pub indexed_count: usize,
    pub returned_count: usize,
    pub filtered_count: usize,
    pub persisted: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub persist_path: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reason_codes: Vec<String>,
    pub user_copy: String,
    pub next_action: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct IngestResponse {
    pub accepted: bool,
    pub reason: String,
    pub memory_id: Option<u64>,
    pub write_score: f32,
    pub conflict_detected: bool,
    pub active_version: Option<u32>,
    pub error_code: Option<String>,
    pub persistence_error: Option<String>,
    pub rolled_back: bool,
}

mod admin_control_plane;
mod agent_activation;
mod answer_admin;
mod container_mip;
mod multimodal;
mod connector;
mod context;
mod memory_browser;
mod memory_graph;
mod performance;
mod query_meta;
mod semantic_governance;
mod taste_collab_provenance;
mod universal_source;

pub use admin_control_plane::*;
pub use agent_activation::*;
pub use answer_admin::*;
pub use container_mip::*;
pub use multimodal::*;
pub use connector::*;
pub use context::*;
pub use memory_browser::*;
pub use memory_graph::*;
pub use performance::*;
pub use query_meta::*;
pub use semantic_governance::*;
pub use taste_collab_provenance::*;
pub use universal_source::*;
