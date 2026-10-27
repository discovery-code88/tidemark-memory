//! Small pieces the memory core shares with its host application: helpers
//! copied from the host, a slim application context, and no-op stand-ins for
//! host services the engine does not carry (database persistence, admin
//! audit). Access control and durability belong to the host service; the
//! engine itself keeps memories in memory.

use crate::*;
use crate::api_types::*;
use std::sync::OnceLock;
use tokio::sync::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::Duration as StdDuration;
use axum::response::Response;





#[derive(Debug, Clone)]
pub struct AuthContext {
    pub api_key: String,
    pub session_id: Option<String>,
    pub principal_type: String,
    pub principal_id: String,
}

pub fn read_env_f32_range(key: &str, default: f32, min: f32, max: f32) -> f32 {
    env::var(key)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .map(|v| v.clamp(min, max))
        .unwrap_or(default)
}

#[derive(Debug, Clone)]
pub struct WriteGateThresholds {
    pub working: f32,
    pub session_summary: f32,
    pub fact: f32,
    pub procedure: f32,
    pub archive: f32,
}

impl WriteGateThresholds {
    pub fn from_env() -> Self {
        let mut thresholds = env::var("WRITE_GATE_THRESHOLDS_JSON")
            .ok()
            .map(|raw| Self::try_from_json(&raw))
            .transpose()
            .map_err(|e| {
                eprintln!("invalid WRITE_GATE_THRESHOLDS_JSON, fallback to defaults: {e}");
                e
            })
            .ok()
            .flatten()
            .unwrap_or_default();

        thresholds.working = read_threshold_env("WRITE_GATE_THRESHOLD_WORKING", thresholds.working);
        thresholds.session_summary = read_threshold_env(
            "WRITE_GATE_THRESHOLD_SESSION_SUMMARY",
            thresholds.session_summary,
        );
        thresholds.fact = read_threshold_env("WRITE_GATE_THRESHOLD_FACT", thresholds.fact);
        thresholds.procedure =
            read_threshold_env("WRITE_GATE_THRESHOLD_PROCEDURE", thresholds.procedure);
        thresholds.archive = read_threshold_env("WRITE_GATE_THRESHOLD_ARCHIVE", thresholds.archive);

        thresholds
    }

    pub fn try_from_json(raw: &str) -> Result<Self, serde_json::Error> {
        let mut thresholds = Self::default();
        let overrides: WriteGateThresholdOverrides = serde_json::from_str(raw)?;
        thresholds.apply_overrides(overrides);
        Ok(thresholds)
    }

    pub fn apply_overrides(&mut self, overrides: WriteGateThresholdOverrides) {
        if let Some(v) = overrides.working {
            self.working = v.clamp(0.0, 1.0);
        }
        if let Some(v) = overrides.session_summary {
            self.session_summary = v.clamp(0.0, 1.0);
        }
        if let Some(v) = overrides.fact {
            self.fact = v.clamp(0.0, 1.0);
        }
        if let Some(v) = overrides.procedure {
            self.procedure = v.clamp(0.0, 1.0);
        }
        if let Some(v) = overrides.archive {
            self.archive = v.clamp(0.0, 1.0);
        }
    }

    pub fn for_layer(&self, layer: &MemoryLayer) -> f32 {
        match layer {
            MemoryLayer::Event => self.working,
            MemoryLayer::Episodic => self.session_summary,
            MemoryLayer::Activation => self.fact,
            MemoryLayer::Parametric => self.fact,
            MemoryLayer::Foresight => self.fact,
            MemoryLayer::SelfModel => self.fact,
            MemoryLayer::Autobiographical => self.fact,
            MemoryLayer::Working => self.working,
            MemoryLayer::SessionSummary => self.session_summary,
            MemoryLayer::Fact => self.fact,
            MemoryLayer::Procedure => self.procedure,
            MemoryLayer::Archive => self.archive,
        }
    }
}

pub static ID_GEN: AtomicU64 = AtomicU64::new(1);

pub fn clean_optional_string(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Shared state of the host application, reduced to what the memory core uses.
pub struct AppContext {
    pub engine: Arc<tokio::sync::RwLock<MemoryEngine>>,
    /// Always `None`: the engine keeps memories in memory; the host persists.
    pub pg: Option<PgStore>,
    pub backend: String,
    pub embedding_mode: String,
    pub embedder: crate::embedding::EmbeddingClient,
    pub stats: RuntimeStats,
    pub shutting_down: Arc<AtomicBool>,
    pub sleep_cycle_interval: StdDuration,
}

pub type AppState = Arc<AppContext>;

#[derive(Clone, Default)]
pub struct RuntimeStats {
    pub answer_requests: Arc<AtomicU64>,
    pub answer_evidence_items: Arc<AtomicU64>,
}

impl RuntimeStats {
    pub fn record_answer(&self, evidence_count: usize) {
        self.answer_requests.fetch_add(1, AtomicOrdering::Relaxed);
        self.answer_evidence_items.fetch_add(evidence_count as u64, AtomicOrdering::Relaxed);
    }
}

/// Stand-in for the host's database store. It is never constructed (the
/// engine runs without a database), so these methods only satisfy the
/// compiler for code paths guarded by `if let Some(pg) = &state.pg`.
pub struct PgStore {
    _never: std::convert::Infallible,
}

impl PgStore {
    pub async fn delete_memories(&self, _ids: &[u64]) -> Result<u64, String> { match self._never {} }
    pub async fn ensure_answer_evidence_projections_bounded(&self, _memories: &[MemoryRecord], _rebuild_limit: usize) -> Result<(usize, usize), String> { match self._never {} }
    pub async fn insert_canonical_semantic_audit(&self, _entry: &CanonicalSemanticAuditEntry) -> Result<(), String> { match self._never {} }
    pub async fn persist_answer_reasoning_proof(&self, _conversation_id: &str, _tenant_id: &str, _workspace_id: &str,
        _proof: &crate::answer_reasoning_provider::VerifiedReasoningProjection, _memories: &[MemoryRecord]) -> Result<bool, String> { match self._never {} }
    pub async fn search_answer_supplemental_ids(&self, _query: &str, _conversation_id: &str, _selected: Option<&[u64]>, _limit: usize) -> Result<Vec<u64>, String> { match self._never {} }
    pub async fn universal_citations_for_memory_ids(&self, _memory_ids: &[u64]) -> Result<Vec<AnswerCitation>, String> { match self._never {} }
    pub async fn upsert_consolidation_job(&self, _job: &ConsolidationJob) -> Result<(), String> { match self._never {} }
}

// Persistence is the host's job (Tidemark keeps a durable write log).
pub async fn maybe_persist_memory(_state: &AppContext, _memory_id: u64) -> Result<(), String> { Ok(()) }
pub async fn maybe_persist_memories(_state: &AppContext, _memory_ids: &[u64]) -> Result<(), String> { Ok(()) }
pub async fn rollback_memory(_state: &AppContext, _memory_id: u64) -> bool { true }
pub async fn rollback_memories(_state: &AppContext, _memory_ids: &[u64]) -> bool { true }

// Admin write policy and audit are the host's job; the engine allows writes.
#[allow(clippy::too_many_arguments)]
pub async fn audit_runtime_write_denial_for_conversation(_state: &AppState, _auth: Option<&AuthContext>, _action: &str,
    _resource_type: &str, _resource_id: &str, _tenant_id: Option<String>, _workspace_id: Option<String>,
    _conversation_id: Option<String>, _reason: String, _error_code: &str) {}
pub async fn audit_runtime_write_applied_for_conversation(_state: &AppState, _auth: Option<&AuthContext>, _action: &str,
    _resource_type: &str, _resource_id: &str, _conversation_id: Option<&str>, _metadata: serde_json::Value) {}
pub fn enforce_admin_runtime_write_for_conversation(_state: &AppContext, _engine: &MemoryEngine, _auth: Option<&AuthContext>,
    _conversation_id: Option<&str>, _action_label: &str) -> Option<Response> { None }

impl MemoryEngine {
    pub fn workspace_for_conversation(&self, conversation_id: &str) -> String {
        self.conversation_workspaces.get(conversation_id).cloned().unwrap_or_else(|| "default".to_string())
    }
}


pub fn read_threshold_env(key: &str, default: f32) -> f32 {
    env::var(key)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(default)
}

#[derive(Debug, Clone, Deserialize)]
pub struct WriteGateThresholdOverrides {
    pub working: Option<f32>,
    pub session_summary: Option<f32>,
    pub fact: Option<f32>,
    pub procedure: Option<f32>,
    pub archive: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct MemoryQualityPatch {
    pub memory_type: Option<String>,
    pub confidence: Option<f32>,
    pub freshness: Option<f32>,
    pub importance: Option<f32>,
    pub last_verified_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub supersedes: Vec<String>,
    #[serde(default)]
    pub superseded_by: Vec<String>,
}





pub async fn embedding_scope_for_conversation(
    state: &AppContext,
    conversation_id: Option<&str>,
    explicit_provider_approval: bool,
) -> crate::embedding::EmbeddingScope {
    let engine = state.engine.read().await;
    let workspace_id = engine.workspace_for_conversation(conversation_id.unwrap_or("default"));
    let tenant_id = engine.workspace_tenant_id(&workspace_id);
    crate::embedding::EmbeddingScope::new(tenant_id, workspace_id, explicit_provider_approval)
}

pub fn execution_budget_from_headers(
    headers: &HeaderMap,
    default_timeout_ms: u64,
) -> (ExecutionBudget, String) {
    fn header_u64(headers: &HeaderMap, name: &str) -> Option<u64> {
        headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.trim().parse::<u64>().ok())
    }
    let workload_shape = headers
        .get("x-umem-workload-shape")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("auto")
        .chars()
        .take(64)
        .collect::<String>();
    (
        ExecutionBudget {
            deadline_ms: header_u64(headers, "x-umem-deadline-ms"),
            timeout_ms: header_u64(headers, "x-umem-timeout-ms").or(Some(default_timeout_ms)),
            max_cost_microusd: header_u64(headers, "x-umem-max-cost-microusd"),
            max_compute_ms: header_u64(headers, "x-umem-max-compute-ms"),
            max_energy_millijoules: header_u64(headers, "x-umem-max-energy-millijoules"),
        },
        workload_shape,
    )
}

// Retrieval feedback is persisted by the host when it has a database.
pub async fn persist_retrieval_feedback(_state: &AppContext, _memory_ids: &[u64]) {}

impl PgStore {
    pub async fn upsert_canonical_semantic_state(&self, _canonical_key: &str, _conversation_id: &str, _semantic_node: &SemanticNode) -> Result<(), String> { match self._never {} }
}

impl MemoryEngine {
    pub fn workspace_tenant_id(&self, _workspace_id: &str) -> String {
        "default".to_string()
    }

    /// Consolidation priority of a conversation from its memories' meta-memory
    /// state (conflict pressure, utility, consolidation hints).
    pub fn consolidation_priority_for_conversation(&self, conversation_id: &str) -> (u32, String) {
        let relevant = self.memories.iter().filter(|m| m.conversation_id == conversation_id)
            .filter_map(|m| m.meta_memory_state.as_ref()).collect::<Vec<_>>();
        if relevant.is_empty() {
            return (10, "no_meta_memory_state".to_string());
        }
        let max_conflict = relevant.iter().map(|m| m.conflict_pressure).fold(0.0_f32, f32::max);
        let max_utility = relevant.iter().map(|m| m.utility_score).fold(0.0_f32, f32::max);
        let priority = 10 + (max_conflict * 50.0) as u32 + (max_utility * 20.0) as u32;
        (priority.min(100), format!("conflict={max_conflict:.2},utility={max_utility:.2}"))
    }
}

/// How much reasoning an answer may use (shared with the HTTP client API).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnswerReasoningMode {
    #[default]
    Balanced,
    Deep,
}

impl AnswerReasoningMode {
    pub fn timeout_ms(self) -> u64 {
        match self {
            Self::Balanced => 20_000,
            Self::Deep => 60_000,
        }
    }
    pub fn call_limit(self) -> usize {
        match self {
            Self::Balanced => 3,
            Self::Deep => 5,
        }
    }
}




pub fn memory_trace_store() -> Arc<RwLock<HashMap<String, MemoryTraceResponse>>> {
    MEMORY_TRACE_STORE
        .get_or_init(|| Arc::new(RwLock::new(HashMap::new())))
        .clone()
}

/// Answer traces carry source text. Recheck after waiting for the cache lock,
/// before inserting any of that text. Never await an engine lock while holding
/// the cache write lock: governance/deletion paths may acquire them oppositely.
pub async fn store_current_answer_trace(
    state: &AppState, trace: MemoryTraceResponse, req: &AnswerRequest,
    sources: &[RetrievedItem], execution: &crate::answer_execution::AnswerExecution,
) -> Result<Option<&'static str>, &'static str> {
    let store=memory_trace_store();
    let mut cache=store.write().await;
    let engine=state.engine.try_read().map_err(|_|"answer_state_busy")?;
    if !execution.authority_is_current(&engine) { return Err("answer_authority_changed"); }
    if !crate::semantic_handlers::local_checkpoint_is_current(&engine,req,sources) {
        return Err("answer_source_snapshot_changed");
    }
    cache.insert(trace.request_id.clone(),trace.clone());
    if cache.len()>1000 {
        let mut keys=cache.keys().cloned().collect::<Vec<_>>();keys.sort();
        for key in keys.into_iter().take(cache.len().saturating_sub(1000)) { cache.remove(&key); }
    }
    drop(cache);
    // Keep the current engine source lease through the bounded PG attempt.
    let warning=if let Some(pg)=&state.pg {
        let deadline=execution.deadline.min(tokio::time::Instant::now()+std::time::Duration::from_millis(250));
        match tokio::time::timeout_at(deadline,pg.upsert_memory_trace(&trace)).await {
            Ok(Ok(()))=>None,
            Ok(Err(_))=>Some("trace_postgres_persist_failed"),
            Err(_)=>Some("trace_postgres_persist_timeout"),
        }
    } else { None };
    drop(engine);
    Ok(warning)
}

pub struct RequestPerformanceBuilder {
    request_id: String,
    operation: String,
    workload_shape: String,
    started: Instant,
    stages: Vec<RequestStageTiming>,
    first_useful_ms: Option<u64>,
    background_submission: Option<ManagedTaskSubmissionResponse>,
    counts: WorkDispositionCounts,
    budget: ExecutionBudget,
    degradations: Vec<ComponentDegradationReceipt>,
    provider_fallbacks: Vec<ProviderFallbackReceipt>,
}

impl RequestPerformanceBuilder {
    pub fn new(operation: &str, workload_shape: &str, budget: ExecutionBudget) -> Self {
        Self {
            request_id: next_performance_request_id(),
            operation: operation.to_string(),
            workload_shape: workload_shape.to_string(),
            started: Instant::now(),
            stages: Vec::new(),
            first_useful_ms: None,
            background_submission: None,
            counts: WorkDispositionCounts::default(),
            budget,
            degradations: Vec::new(),
            provider_fallbacks: Vec::new(),
        }
    }

    pub fn stage_start(&self) -> Instant {
        Instant::now()
    }

    pub fn record_stage(
        &mut self,
        stage: &str,
        queue_class: &str,
        component: &str,
        stage_started: Instant,
        outcome: &str,
        reason: Option<String>,
    ) {
        let duration_ms = stage_started.elapsed().as_millis() as u64;
        let started_offset_ms = stage_started
            .saturating_duration_since(self.started)
            .as_millis() as u64;
        self.stages.push(RequestStageTiming {
            stage: stage.to_string(),
            queue_class: queue_class.to_string(),
            component: component.to_string(),
            started_offset_ms,
            duration_ms,
            outcome: outcome.to_string(),
            reason,
        });
    }

    pub fn mark_first_useful(&mut self) {
        if self.first_useful_ms.is_none() {
            self.first_useful_ms = Some(self.started.elapsed().as_millis() as u64);
        }
    }

    pub fn set_counts(&mut self, counts: WorkDispositionCounts) {
        self.counts = counts;
    }

    pub fn set_background_submission(&mut self, submission: ManagedTaskSubmissionResponse) {
        if !submission.accepted {
            self.counts.deferred = self
                .counts
                .deferred
                .saturating_add(submission.counts.deferred);
            self.counts.blocked = self
                .counts
                .blocked
                .saturating_add(submission.counts.blocked);
        }
        self.background_submission = Some(submission);
    }

    pub fn record_degradation(&mut self, receipt: ComponentDegradationReceipt) {
        self.degradations.push(receipt);
    }

    pub fn record_provider_fallback(&mut self, receipt: ProviderFallbackReceipt) {
        self.provider_fallbacks.push(receipt);
    }

    pub fn finish(self, cache_state: &str) -> RequestPerformanceReceipt {
        let complete_result_ms = self.started.elapsed().as_millis() as u64;
        let first_useful_result_ms = self.first_useful_ms.unwrap_or(complete_result_ms);
        let background_job_id = self
            .background_submission
            .as_ref()
            .and_then(|submission| submission.job_id.clone());
        let background_enrichment_status = self
            .background_submission
            .as_ref()
            .map(|submission| submission.status.clone())
            .unwrap_or_else(|| "not_requested".to_string());
        let continuation = if self.counts.deferred > 0
            || self.counts.omitted > 0
            || self.counts.blocked > 0
            || self.counts.failed > 0
        {
            let token = continuation_token(&self.request_id, 1);
            ContinuationReceipt {
                available: true,
                token: Some(token),
                reason: "work_remains_after_current_result".to_string(),
            }
        } else {
            ContinuationReceipt {
                available: false,
                token: None,
                reason: "no_deferred_or_omitted_work".to_string(),
            }
        };
        RequestPerformanceReceipt {
            receipt_format: "request_performance_receipt_v1".to_string(),
            build_id: env::var("UNLIMITED_MEMORY_BUILD_ID")
                .unwrap_or_else(|_| format!("{}-unbound", env!("CARGO_PKG_VERSION"))),
            request_id: self.request_id,
            operation: self.operation,
            workload_shape: self.workload_shape,
            cache_state: cache_state.to_string(),
            stages: self.stages,
            milestones: ResultMilestoneReceipt {
                first_useful_result_ms,
                complete_result_ms,
                background_enrichment_status,
                background_job_id,
            },
            queues: managed_task_queues(),
            counts: self.counts,
            budget: ResourceBudgetReceipt {
                deadline_ms: self.budget.deadline_ms,
                timeout_ms: self.budget.timeout_ms,
                cost_limit_microusd: self.budget.max_cost_microusd,
                cost_used_microusd: self
                    .provider_fallbacks
                    .iter()
                    .map(|receipt| receipt.estimated_cost_microusd)
                    .sum(),
                compute_limit_ms: self.budget.max_compute_ms,
                compute_used_ms: complete_result_ms,
                energy_limit_millijoules: self.budget.max_energy_millijoules,
                energy_used_millijoules: complete_result_ms,
                energy_measurement: "estimated_cpu_wall_time_v1".to_string(),
            },
            degradations: self.degradations,
            provider_fallbacks: self.provider_fallbacks,
            continuation,
        }
    }
}

impl Default for WriteGateThresholds {
    fn default() -> Self {
        Self {
            working: 0.25,
            session_summary: 0.35,
            fact: 0.30,
            procedure: 0.35,
            archive: 0.45,
        }
    }
}


/// Memory type of host-written memory-file snapshots (excluded from some summaries).
pub const MEMORY_FILE_SNAPSHOT_TYPE: &str = "memory_file_snapshot";

/// Database-backed candidate retrieval of the host. Without a database the
/// host returns `None` and answers use the engine's in-memory retrieval; the
/// engine always runs without one.
pub async fn retrieve_answer_candidates(_state: AppState, _req: QueryRequest, _context: &serde_json::Value,
    _auth: Option<&AuthContext>) -> Option<QueryResponse> {
    None
}


impl PgStore {
    pub async fn upsert_memory(&self, _m: &MemoryRecord, _embedding_vec: &[f32], _embedding_provider: &str, _embedding_model: &str) -> Result<(), String> { match self._never {} }
    pub async fn upsert_memory_trace(&self, _trace: &MemoryTraceResponse) -> Result<(), String> { match self._never {} }
}

impl MemoryEngine {
    /// Re-evaluates every memory's meta-memory state (salience, stability,
    /// utility, conflict pressure).
    pub fn rebuild_meta_memory_state(&mut self) {
        for memory in &mut self.memories {
            memory.meta_memory_state = Some(MetaMemoryController::evaluate_memory(memory));
        }
    }

    /// Re-evaluates the meta-memory state of one conversation's memories.
    pub fn rebuild_meta_memory_state_for(&mut self, conversation_id: &str) {
        for memory in self.memories.iter_mut().filter(|m| m.conversation_id == conversation_id) {
            memory.meta_memory_state = Some(MetaMemoryController::evaluate_memory(memory));
        }
    }

    pub fn workspace_registry_state_for(&self, workspace_id: &str) -> Option<&WorkspaceRegistryState> {
        self.workspace_registry.get(workspace_id)
    }

    pub fn conversation_matches_workspace(&self, conversation_id: &str, workspace_id: Option<&str>) -> bool {
        workspace_id.is_none_or(|w| self.workspace_for_conversation(conversation_id) == w)
    }
}


pub static MEMORY_TRACE_STORE: OnceLock<Arc<RwLock<HashMap<String, MemoryTraceResponse>>>> = OnceLock::new();


/// What ingestion records about each stored item: a normalized event
/// summary (intent, result, confidence, value weight, evidence tags).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PerceptionEvent {
    pub event_type: String,
    pub source_type: String,
    pub normalizer_id: String,
    pub actor: Option<String>,
    pub target: Option<String>,
    pub intent: String,
    pub result: String,
    pub confidence: f32,
    pub value_weight: f32,
    pub evidence_tags: Vec<String>,
    pub normalized_summary: String,
    pub raw_payload: serde_json::Value,
}


pub fn semantic_status_label(status: &SemanticStatus) -> &'static str {
    match status {
        SemanticStatus::Active => "active",
        SemanticStatus::Superseded => "superseded",
        SemanticStatus::Disputed => "disputed",
        SemanticStatus::Deprecated => "deprecated",
    }
}

pub fn strict_deployment_profile_name(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "production" | "commercial-pilot"
    )
}

pub fn read_env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

pub fn read_env_i64(key: &str, default: i64) -> i64 {
    env::var(key)
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

#[derive(Debug, Clone, Default)]
pub struct AdminScopePolicy {
    pub role: String,
    pub tenant_ids: HashSet<String>,
    pub workspace_ids: HashSet<String>,
}





pub fn next_performance_request_id() -> String {
    format!(
        "perf-{}-{}",
        Utc::now().timestamp_millis(),
        PERFORMANCE_REQUEST_ID_GEN.fetch_add(1, AtomicOrdering::Relaxed)
    )
}



pub fn continuation_token(job_id: &str, attempt: usize) -> String {
    let digest = sha256_hex(&format!("core006-continuation:{job_id}:{attempt}"));
    format!("continue-{}", &digest[..32])
}

pub fn component_degradation_receipt(
    component: &str,
    status: &str,
    required: bool,
    retry_count: usize,
) -> Result<ComponentDegradationReceipt, String> {
    let component = component.trim().to_ascii_lowercase();
    let status = status.trim().to_ascii_lowercase();
    if !matches!(
        component.as_str(),
        "provider" | "graph_projection" | "object_storage" | "database" | "device"
    ) {
        return Err("unsupported_component".to_string());
    }
    if !matches!(
        status.as_str(),
        "healthy" | "slow" | "unavailable" | "not_configured"
    ) {
        return Err("unsupported_component_status".to_string());
    }
    let (disposition, fallback_from, fallback_to, reason_code) =
        match (component.as_str(), status.as_str(), required) {
            (_, "healthy", _) => (
                "selected",
                component.as_str(),
                component.as_str(),
                "healthy",
            ),
            ("provider", _, true) => (
                "blocked",
                "requested_provider",
                "none",
                "required_provider_unavailable",
            ),
            ("provider", _, false) => (
                "degraded",
                "requested_provider",
                "local_deterministic_provider",
                "provider_fallback",
            ),
            ("graph_projection", _, true) => (
                "blocked",
                "graph_projection",
                "none",
                "required_projection_unavailable",
            ),
            ("graph_projection", _, false) => (
                "degraded",
                "graph_projection",
                "authoritative_database_bounded_traversal",
                "projection_fallback",
            ),
            ("object_storage", _, true) => (
                "blocked",
                "object_storage",
                "none",
                "required_object_unavailable",
            ),
            ("object_storage", _, false) => (
                "deferred",
                "object_storage",
                "metadata_only_result",
                "object_enrichment_deferred",
            ),
            ("database", _, true) => (
                "blocked",
                "database",
                "none",
                "authoritative_database_unavailable",
            ),
            ("database", _, false) => (
                "degraded",
                "database",
                "read_only_warm_memory",
                "database_read_fallback",
            ),
            ("device", _, _) => (
                "blocked",
                "live_device",
                "no_action_cached_observation",
                "device_action_fail_closed",
            ),
            _ => (
                "degraded",
                component.as_str(),
                "bounded_local_path",
                "component_degraded",
            ),
        };
    Ok(ComponentDegradationReceipt {
        component: component.clone(),
        observed_status: status,
        disposition: disposition.to_string(),
        fallback_from: fallback_from.to_string(),
        fallback_to: fallback_to.to_string(),
        reason_code: reason_code.to_string(),
        retry_count,
        user_visible: true,
    })
}

pub fn next_memory_trace_id() -> String {
    format!(
        "trace_{}",
        MEMORY_TRACE_ID_GEN.fetch_add(1, AtomicOrdering::Relaxed)
    )
}

pub fn multimodal_query_score(memory: &MemoryRecord, query: &str) -> f32 {
    let query = query.to_lowercase();
    let wants_document = query.contains("文档")
        || query.contains("pdf")
        || query.contains("markdown")
        || query.contains("文件")
        || query.contains("adr");
    let wants_image = query.contains("图片")
        || query.contains("截图")
        || query.contains("图像")
        || query.contains("image")
        || query.contains("白板");
    let wants_audio = query.contains("录音")
        || query.contains("音频")
        || query.contains("通话")
        || query.contains("会议录音")
        || query.contains("播客");
    let wants_video = query.contains("视频")
        || query.contains("录像")
        || query.contains("演示视频")
        || query.contains("demo video")
        || query.contains("录屏");

    let mut score = 0.0;
    if memory.source.starts_with("multimodal:document:") && wants_document {
        score += 0.22;
        if query.contains("pdf") && memory.source.starts_with("multimodal:document:pdf:") {
            score += 0.08;
        }
        if query.contains("markdown") && memory.source.starts_with("multimodal:document:markdown:")
        {
            score += 0.08;
        }
    }
    if memory.source.starts_with("multimodal:image:") && wants_image {
        score += 0.24;
        if query.contains("截图") || query.contains("白板") {
            score += 0.08;
        }
    }
    if memory.source.starts_with("multimodal:audio:") && wants_audio {
        score += 0.24;
        if query.contains("录音") || query.contains("会议录音") {
            score += 0.08;
        }
    }
    if memory.source.starts_with("multimodal:video:") && wants_video {
        score += 0.24;
        if query.contains("视频") || query.contains("录屏") {
            score += 0.08;
        }
    }
    if memory.source.starts_with("multimodal:ocr:")
        && (query.contains("截图里的字")
            || query.contains("图里的字")
            || query.contains("图片里的字")
            || query.contains("ocr")
            || query.contains("识别文字"))
    {
        score += 0.28;
    }
    score
}

pub fn build_multimodal_answer_notes(
    items: &[RetrievedItem],
    query: &str,
) -> Vec<SemanticAnswerNote> {
    let query = query.to_lowercase();
    let wants_document = query.contains("文档")
        || query.contains("pdf")
        || query.contains("markdown")
        || query.contains("文件")
        || query.contains("adr");
    let wants_image = query.contains("图片")
        || query.contains("截图")
        || query.contains("image")
        || query.contains("白板")
        || query.contains("看板");
    let wants_ocr = query.contains("截图里的字")
        || query.contains("图里的字")
        || query.contains("图片里的字")
        || query.contains("ocr")
        || query.contains("识别文字")
        || query.contains("提取文字");
    let wants_audio = query.contains("录音")
        || query.contains("音频")
        || query.contains("通话")
        || query.contains("会议录音")
        || query.contains("播客");
    let wants_video = query.contains("视频")
        || query.contains("录像")
        || query.contains("演示视频")
        || query.contains("demo video")
        || query.contains("录屏");

    let mut notes = Vec::new();
    let mut seen = HashSet::new();
    for item in items {
        if wants_document && item.source.starts_with("multimodal:document:") {
            let source_ref = item.source.splitn(4, ':').nth(3).unwrap_or("document");
            let message = format!(
                "[文档记忆提示] 已检索到文档来源 {source_ref}：{}。",
                preview_text(&item.content)
            );
            if seen.insert(message.clone()) {
                notes.push(SemanticAnswerNote {
                    note_type: "multimodal_document".to_string(),
                    message,
                    entity_id: None,
                    attribute_key: None,
                    scope: None,
                    candidate_values: Vec::new(),
                });
            }
        }

        if wants_image && item.source.starts_with("multimodal:image:") {
            let source_ref = item.source.splitn(4, ':').nth(3).unwrap_or("image");
            let message = format!(
                "[图片记忆提示] 已检索到图片来源 {source_ref}：{}。",
                preview_text(&item.content)
            );
            if seen.insert(message.clone()) {
                notes.push(SemanticAnswerNote {
                    note_type: "multimodal_image".to_string(),
                    message,
                    entity_id: None,
                    attribute_key: None,
                    scope: None,
                    candidate_values: Vec::new(),
                });
            }
        }

        if wants_ocr && item.source.starts_with("multimodal:ocr:") {
            let source_ref = item.source.splitn(4, ':').nth(3).unwrap_or("ocr");
            let message = format!(
                "[OCR 文字提示] 已提取 OCR 文字来源 {source_ref}：{}。",
                preview_text(&item.content)
            );
            if seen.insert(message.clone()) {
                notes.push(SemanticAnswerNote {
                    note_type: "multimodal_ocr".to_string(),
                    message,
                    entity_id: None,
                    attribute_key: None,
                    scope: None,
                    candidate_values: Vec::new(),
                });
            }
        }

        if wants_audio && item.source.starts_with("multimodal:audio:") {
            let source_ref = item.source.splitn(4, ':').nth(3).unwrap_or("audio");
            let message = format!(
                "[音频记忆提示] 已检索到音频来源 {source_ref}：{}。",
                preview_text(&item.content)
            );
            if seen.insert(message.clone()) {
                notes.push(SemanticAnswerNote {
                    note_type: "multimodal_audio".to_string(),
                    message,
                    entity_id: None,
                    attribute_key: None,
                    scope: None,
                    candidate_values: Vec::new(),
                });
            }
        }

        if wants_video && item.source.starts_with("multimodal:video:") {
            let source_ref = item.source.splitn(4, ':').nth(3).unwrap_or("video");
            let message = format!(
                "[视频记忆提示] 已检索到视频来源 {source_ref}：{}。",
                preview_text(&item.content)
            );
            if seen.insert(message.clone()) {
                notes.push(SemanticAnswerNote {
                    note_type: "multimodal_video".to_string(),
                    message,
                    entity_id: None,
                    attribute_key: None,
                    scope: None,
                    candidate_values: Vec::new(),
                });
            }
        }
    }

    notes
}

pub fn kind_to_slug(value: &str) -> &'static str {
    match value {
        "小说" => "novel",
        "项目" => "project",
        "应用" => "app",
        _ => "topic",
    }
}

pub async fn append_system_memory(
    state: &AppState,
    conversation_id: &str,
    layer: MemoryLayer,
    source: &str,
    memory_type: &str,
    content: serde_json::Value,
    trust: f32,
) -> Result<u64, String> {
    let memory_id = append_system_memory_unpersisted(
        state,
        conversation_id,
        layer,
        source,
        memory_type,
        content,
        trust,
    )
    .await?;
    maybe_persist_memory(state.as_ref(), memory_id).await?;
    Ok(memory_id)
}

pub fn advance_memory_id_generator_floor(next_id_floor: u64) {
    let mut current = ID_GEN.load(AtomicOrdering::Relaxed);
    while current < next_id_floor {
        match ID_GEN.compare_exchange(
            current,
            next_id_floor,
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
        ) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

/// Long-document summaries for answers come from the host's import pipeline;
/// the engine has none to refresh.
pub async fn maybe_auto_refresh_long_document_summary_for_answer(_state: &AppState, _auth: Option<&AuthContext>,
    _req: &AnswerRequest) -> Option<SemanticAnswerNote> {
    None
}

/// Admin scopes are enforced by the host; inside the engine no caller has one.
pub fn admin_scope_policy(_state: &AppContext, _engine: &MemoryEngine, _auth: Option<&AuthContext>) -> Option<AdminScopePolicy> {
    None
}

#[allow(clippy::too_many_arguments)]
pub async fn admin_scope_required_with_audit(_state: &AppState, _auth: Option<&AuthContext>, _action: &str,
    _resource_type: &str, _resource_id: &str, _tenant_id: Option<String>, _workspace_id: Option<String>,
    _conversation_id: Option<String>, message: impl Into<String>) -> Response {
    use axum::response::IntoResponse;
    (axum::http::StatusCode::FORBIDDEN, axum::Json(serde_json::json!({"error": "admin_scope_required", "message": message.into()}))).into_response()
}


pub fn preview_text(content: &str) -> String {
    let text = content.trim();
    if text.chars().count() <= 80 {
        text.to_string()
    } else {
        format!("{}...", text.chars().take(80).collect::<String>())
    }
}

pub fn sha256_hex(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

pub static PERFORMANCE_REQUEST_ID_GEN: AtomicU64 = AtomicU64::new(1);

pub static MEMORY_TRACE_ID_GEN: AtomicU64 = AtomicU64::new(1);

pub async fn append_system_memory_unpersisted(
    state: &AppState,
    conversation_id: &str,
    layer: MemoryLayer,
    source: &str,
    memory_type: &str,
    content: serde_json::Value,
    trust: f32,
) -> Result<u64, String> {
    let content = content.to_string();
    let memory_id = match &state.pg {
        Some(pg) => pg
            .reserve_id()
            .await
            .map_err(|error| format!("reserve_memory_id_failed:{error}"))?,
        None => ID_GEN.fetch_add(1, AtomicOrdering::Relaxed),
    };
    append_memory_record_unpersisted(
        state,
        memory_id,
        conversation_id,
        layer,
        source,
        memory_type,
        content,
        trust,
        None,
        None,
        None,
    )
    .await
}

/// Memory files live in the host's file store; the engine has none.
pub async fn read_memory_file_content_for_space(_space_id: &str, path: &str) -> Result<(String, String), Response> {
    use axum::response::IntoResponse;
    Err((axum::http::StatusCode::NOT_FOUND, axum::Json(serde_json::json!({"error": "memory_file_not_found", "path": path}))).into_response())
}

pub async fn collect_memory_file_records_for_space(_space_id: &str, _include_content: bool) -> Vec<MemoryFileRecord> {
    Vec::new()
}

/// Managed task queues belong to the host's job runner; the engine has none.
pub fn managed_task_queues() -> Vec<QueueClassReceipt> {
    Vec::new()
}


pub async fn append_memory_record_unpersisted(
    state: &AppState,
    memory_id: u64,
    conversation_id: &str,
    layer: MemoryLayer,
    source: &str,
    memory_type: &str,
    content: String,
    trust: f32,
    entity: Option<String>,
    fact_key: Option<String>,
    fact_value: Option<String>,
) -> Result<u64, String> {
    let record = MemoryRecord {
        id: memory_id,
        conversation_id: conversation_id.to_string(),
        layer,
        tokens: MemoryEngine::tokenize(&content),
        content,
        source: source.to_string(),
        trust: trust.clamp(0.0, 1.0),
        created_at: Utc::now(),
        anchor_ids: Vec::new(),
        entity,
        fact_key,
        fact_value,
        memory_type: Some(memory_type.to_string()),
        confidence: Some(trust.clamp(0.0, 1.0)),
        freshness: Some(1.0),
        importance: Some(0.5),
        last_verified_at: Some(Utc::now()),
        supersedes: Vec::new(),
        superseded_by: Vec::new(),
        version: 1,
        event_metadata: None,
        procedure_schema: None,
        procedure_execution_state: None,
        procedure_governance: None,
        episodic_metadata: None,
        meta_memory_state: None,
        activation_metadata: None,
        retrieval_hits: 0,
        last_retrieved_at: None,
        embedding_provider: None,
        embedding_model: None,
        embedding_status: None,
        embedded_at: None,
        semantic_node: None,
        semantic_conflict: None,
    };
    {
        let mut engine = state.engine.write().await;
        if engine.memories.iter().any(|memory| memory.id == memory_id) {
            return Err(format!(
                "memory identity {memory_id} is already present in the local projection"
            ));
        }
        engine.memories.push(record);
        engine.invalidate_cache_for_conversation(Some(conversation_id));
    }
    Ok(memory_id)
}

impl PgStore {
    pub async fn reserve_id(&self) -> Result<u64, String> { match self._never {} }
}
