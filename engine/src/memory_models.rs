use super::*;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MemoryLayer {
    Event,
    Episodic,
    Activation,
    Parametric,
    Foresight,
    SelfModel,
    Autobiographical,
    Working,
    SessionSummary,
    Fact,
    Procedure,
    Archive,
}

impl MemoryLayer {
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryLayer::Event => "event",
            MemoryLayer::Episodic => "episodic",
            MemoryLayer::Activation => "activation",
            MemoryLayer::Parametric => "parametric",
            MemoryLayer::Foresight => "foresight",
            MemoryLayer::SelfModel => "self_model",
            MemoryLayer::Autobiographical => "autobiographical",
            MemoryLayer::Working => "working",
            MemoryLayer::SessionSummary => "session_summary",
            MemoryLayer::Fact => "fact",
            MemoryLayer::Procedure => "procedure",
            MemoryLayer::Archive => "archive",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ActivationMetadata {
    pub status: String,
    pub source: String,
    pub target_memory_ids: Vec<u64>,
    pub expires_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct EventMetadata {
    pub origin_layer: MemoryLayer,
    pub intent: String,
    pub result: String,
    pub memory_action: String,
    pub confidence: f32,
    pub value_weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ProcedureSchema {
    pub trigger_conditions: Vec<String>,
    pub steps: Vec<String>,
    pub rollback_steps: Vec<String>,
    pub success_criteria: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ProcedureExecutionState {
    pub procedure_id: u64,
    pub current_step_index: Option<usize>,
    pub retry_count: u32,
    pub outcome: String,
    pub failure_reason: Option<String>,
    pub success_evidence: Option<String>,
    pub steps: Vec<ProcedureStepStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ProcedureGovernance {
    pub preferred: bool,
    pub auto_deprecated: bool,
    pub last_health_score: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct EpisodicMetadata {
    pub actors: Vec<String>,
    pub actions: Vec<String>,
    pub constraints: Vec<String>,
    pub result: Option<String>,
    pub outcome: Option<String>,
    pub cause_links: Vec<String>,
    pub recovery_links: Vec<String>,
    pub failure_type: Option<String>,
    pub actor_action_pairs: Vec<String>,
    pub action_result_pairs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MemoryRecord {
    pub(crate) id: u64,
    pub(crate) conversation_id: String,
    pub(crate) layer: MemoryLayer,
    pub(crate) content: String,
    pub(crate) tokens: Vec<String>,
    pub(crate) source: String,
    pub(crate) trust: f32,
    pub(crate) created_at: DateTime<Utc>,
    pub(crate) anchor_ids: Vec<u64>,
    pub(crate) entity: Option<String>,
    pub(crate) fact_key: Option<String>,
    pub(crate) fact_value: Option<String>,
    pub(crate) memory_type: Option<String>,
    pub(crate) confidence: Option<f32>,
    pub(crate) freshness: Option<f32>,
    pub(crate) importance: Option<f32>,
    pub(crate) last_verified_at: Option<DateTime<Utc>>,
    pub(crate) supersedes: Vec<String>,
    pub(crate) superseded_by: Vec<String>,
    pub(crate) version: u32,
    pub(crate) event_metadata: Option<EventMetadata>,
    pub(crate) procedure_schema: Option<ProcedureSchema>,
    pub(crate) procedure_execution_state: Option<ProcedureExecutionState>,
    pub(crate) procedure_governance: Option<ProcedureGovernance>,
    pub(crate) episodic_metadata: Option<EpisodicMetadata>,
    pub(crate) meta_memory_state: Option<MetaMemoryState>,
    pub(crate) activation_metadata: Option<ActivationMetadata>,
    pub(crate) retrieval_hits: u32,
    pub(crate) last_retrieved_at: Option<DateTime<Utc>>,
    pub(crate) embedding_provider: Option<String>,
    pub(crate) embedding_model: Option<String>,
    pub(crate) embedding_status: Option<String>,
    pub(crate) embedded_at: Option<DateTime<Utc>>,
    pub(crate) semantic_node: Option<SemanticNode>,
    pub(crate) semantic_conflict: Option<SemanticConflictSet>,
}

impl From<MemoryRecord> for PortableMemoryRecord {
    fn from(memory: MemoryRecord) -> Self {
        Self {
            id: memory.id,
            conversation_id: memory.conversation_id,
            layer: memory.layer,
            content: memory.content,
            tokens: memory.tokens,
            source: memory.source,
            trust: memory.trust,
            created_at: memory.created_at,
            anchor_ids: memory.anchor_ids,
            entity: memory.entity,
            fact_key: memory.fact_key,
            fact_value: memory.fact_value,
            memory_type: memory.memory_type,
            confidence: memory.confidence,
            freshness: memory.freshness,
            importance: memory.importance,
            last_verified_at: memory.last_verified_at,
            supersedes: memory.supersedes,
            superseded_by: memory.superseded_by,
            version: memory.version,
            event_metadata: memory.event_metadata,
            procedure_schema: memory.procedure_schema,
            procedure_execution_state: memory.procedure_execution_state,
            procedure_governance: memory.procedure_governance,
            episodic_metadata: memory.episodic_metadata,
            meta_memory_state: memory.meta_memory_state,
            activation_metadata: memory.activation_metadata,
            semantic_node: memory.semantic_node,
            semantic_conflict: memory.semantic_conflict,
        }
    }
}

impl From<PortableMemoryRecord> for MemoryRecord {
    fn from(memory: PortableMemoryRecord) -> Self {
        Self {
            id: memory.id,
            conversation_id: memory.conversation_id,
            layer: memory.layer,
            content: memory.content,
            tokens: memory.tokens,
            source: memory.source,
            trust: memory.trust,
            created_at: memory.created_at,
            anchor_ids: memory.anchor_ids,
            entity: memory.entity,
            fact_key: memory.fact_key,
            fact_value: memory.fact_value,
            memory_type: memory.memory_type,
            confidence: memory.confidence,
            freshness: memory.freshness,
            importance: memory.importance,
            last_verified_at: memory.last_verified_at,
            supersedes: memory.supersedes,
            superseded_by: memory.superseded_by,
            version: memory.version,
            event_metadata: memory.event_metadata,
            procedure_schema: memory.procedure_schema,
            procedure_execution_state: memory.procedure_execution_state,
            procedure_governance: memory.procedure_governance,
            episodic_metadata: memory.episodic_metadata,
            meta_memory_state: memory.meta_memory_state,
            activation_metadata: memory.activation_metadata,
            retrieval_hits: 0,
            last_retrieved_at: None,
            embedding_provider: None,
            embedding_model: None,
            embedding_status: None,
            embedded_at: None,
            semantic_node: memory.semantic_node,
            semantic_conflict: memory.semantic_conflict,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CacheEntry {
    pub(crate) query: String,
    pub(crate) ids: Vec<u64>,
    pub(crate) expires_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub(crate) struct ProcedureExecutionResult {
    pub(crate) response: ProcedureExecutionResponse,
    pub(crate) created_ids: Vec<u64>,
}
