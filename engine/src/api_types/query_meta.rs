use super::*;

#[derive(Debug, Clone, Serialize)]
pub struct RetrievedItem {
    pub id: u64,
    pub layer: MemoryLayer,
    pub memory_type: String,
    pub content: String,
    pub score: f32,
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity: Option<String>,
    pub created_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub freshness: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub importance: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_verified_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supersedes: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
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
    pub canonical_semantic_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueryResponse {
    pub cached: bool,
    pub items: Vec<RetrievedItem>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector_retrieval: Option<MemoryTraceVectorRetrieval>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub performance: Option<RequestPerformanceReceipt>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetaMemoryEvaluatedItem {
    pub memory_id: u64,
    pub layer: MemoryLayer,
    pub meta_memory_state: MetaMemoryState,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetaMemoryEvaluateResponse {
    pub items: Vec<MetaMemoryEvaluatedItem>,
    pub recommended_action_counts: HashMap<String, usize>,
    pub layer_counts: HashMap<String, usize>,
}
