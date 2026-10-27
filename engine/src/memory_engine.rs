use super::*;



#[derive(Debug, Clone)]
pub(crate) struct MemoryEngine {
    pub(crate) memories: Vec<MemoryRecord>,
    pub(crate) shared_memory_visibility: HashMap<u64, MemoryVisibility>,
    pub(crate) agent_runtime: HashMap<String, AgentRuntimeState>,
    pub(crate) agent_tasks: HashMap<String, AgentTaskRecord>,
    pub(crate) workspace_registry: HashMap<String, WorkspaceRegistryState>,
    pub(crate) conversation_workspaces: HashMap<String, String>,
    pub(crate) admin_principals: HashMap<String, AdminPrincipalState>,
    pub(crate) admin_role_bindings: HashMap<String, AdminRoleBindingState>,
    pub(crate) control_plane_policy_assignments: HashMap<String, ControlPlanePolicyAssignmentState>,
    pub(crate) canonical_semantic_state: HashMap<String, SemanticNode>,
    pub(crate) canonical_semantic_audit: Vec<CanonicalSemanticAuditEntry>,
    pub(crate) container_audit_log: Vec<ContainerAuditEntry>,
    pub(crate) control_plane_audit_log: Vec<ControlPlaneAuditEntry>,
    pub(crate) connector_runs: Vec<ConnectorRunRecord>,
    pub(crate) multimodal_extraction_runs: Vec<MultimodalExtractionRunRecord>,
    pub(crate) connector_schedules: HashMap<String, ConnectorScheduleState>,
    pub(crate) connector_webhook_secrets: HashMap<String, String>,
    pub(crate) connector_webhook_replay_cache: HashMap<String, DateTime<Utc>>,
    pub(crate) consolidation_jobs: Vec<ConsolidationJob>,
    pub(crate) cache: HashMap<String, CacheEntry>,
    pub(crate) candidate_cache_revision: u64,
    pub(crate) user_preferences: HashSet<String>,
    pub(crate) write_gate_thresholds: WriteGateThresholds,
    pub(crate) auto_consolidation: AutoConsolidationConfig,
    pub(crate) memory_decay: MemoryDecayConfig,
}

#[derive(Debug, Clone)]
pub(crate) struct MemoryVisibility {
    pub(crate) agent_id: Option<String>,
    pub(crate) shared_block: Option<String>,
}

pub(crate) struct QueryRuntimeSnapshot {
    pub(crate) eligible_count: usize,
    pub(crate) has_live_activation: bool,
    pub(crate) cached_ids: Option<Vec<u64>>,
    pub(crate) current_project: Option<String>,
    pub(crate) current_task: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct ConnectorScheduleState {
    pub(crate) record: ConnectorScheduleRecord,
    pub(crate) payload: serde_json::Value,
    pub(crate) trust: Option<f32>,
}

#[derive(Debug, Clone)]
pub(crate) struct ScopedQueryContext {
    pub(crate) agent_id: String,
    pub(crate) shared_blocks: HashSet<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct AgentRuntimeState {
    pub(crate) conversation_id: String,
    pub(crate) agent_id: String,
    pub(crate) role: String,
    pub(crate) lifecycle_state: String,
    pub(crate) registered_at: DateTime<Utc>,
    pub(crate) last_seen_at: DateTime<Utc>,
    pub(crate) last_ingest_at: Option<DateTime<Utc>>,
    pub(crate) last_query_at: Option<DateTime<Utc>>,
    pub(crate) ingest_count: usize,
    pub(crate) query_count: usize,
    pub(crate) shared_blocks: HashSet<String>,
    pub(crate) supervisor_agent_id: Option<String>,
    pub(crate) worker_agent_ids: HashSet<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct WorkspaceRegistryState {
    pub(crate) workspace_id: String,
    pub(crate) tenant_id: String,
    pub(crate) display_name: Option<String>,
    pub(crate) isolation_policy: String,
    pub(crate) compliance_profile: String,
    pub(crate) lifecycle_status: String,
    pub(crate) updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub(crate) struct AdminPrincipalState {
    pub(crate) principal_id: String,
    pub(crate) display_name: Option<String>,
    pub(crate) key_fingerprint: String,
    pub(crate) status: String,
    pub(crate) principal_type: String,
    pub(crate) created_at: DateTime<Utc>,
    pub(crate) updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub(crate) struct AdminRoleBindingState {
    pub(crate) binding_id: String,
    pub(crate) principal_id: String,
    pub(crate) role: String,
    pub(crate) status: String,
    pub(crate) tenant_ids: HashSet<String>,
    pub(crate) workspace_ids: HashSet<String>,
    pub(crate) created_at: DateTime<Utc>,
    pub(crate) updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub(crate) struct ControlPlanePolicyAssignmentState {
    pub(crate) assignment_id: String,
    pub(crate) tenant_id: Option<String>,
    pub(crate) workspace_id: Option<String>,
    pub(crate) policy_key: String,
    pub(crate) policy_value: String,
    pub(crate) status: String,
    pub(crate) created_at: DateTime<Utc>,
    pub(crate) updated_at: DateTime<Utc>,
}





#[derive(Debug, Clone)]
pub(crate) struct AutoConsolidationConfig {
    pub(crate) min_high_level_memories: usize,
    pub(crate) min_value_weight: f32,
}

#[derive(Debug, Clone)]
pub(crate) struct MemoryDecayConfig {
    pub(crate) deferred_max_age_hours: i64,
    pub(crate) deferred_max_value_weight: f32,
}

impl Default for AutoConsolidationConfig {
    fn default() -> Self {
        Self {
            min_high_level_memories: 2,
            min_value_weight: 0.85,
        }
    }
}

impl Default for MemoryDecayConfig {
    fn default() -> Self {
        Self {
            deferred_max_age_hours: 24,
            deferred_max_value_weight: 0.5,
        }
    }
}

impl AutoConsolidationConfig {
    pub(crate) fn from_env() -> Self {
        Self {
            min_high_level_memories: read_env_usize(
                "AUTO_CONSOLIDATION_MIN_HIGH_LEVEL_MEMORIES",
                Self::default().min_high_level_memories,
            ),
            min_value_weight: env::var("AUTO_CONSOLIDATION_MIN_VALUE_WEIGHT")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .map(|v| v.clamp(0.0, 1.0))
                .unwrap_or(Self::default().min_value_weight),
        }
    }
}

impl MemoryEngine {
    pub(crate) const DEFAULT_TENANT_ID: &'static str = "default";
    pub(crate) const DEFAULT_WORKSPACE_ISOLATION_POLICY: &'static str = "shared";
    pub(crate) const DEFAULT_WORKSPACE_COMPLIANCE_PROFILE: &'static str = "standard";

    pub(crate) fn batch_window_for_conversation(
        &self,
        conversation_id: &str,
    ) -> (usize, Option<DateTime<Utc>>, Option<DateTime<Utc>>) {
        let scoped = self
            .memories
            .iter()
            .filter(|memory| memory.conversation_id == conversation_id)
            .collect::<Vec<_>>();
        let window_start = scoped.iter().map(|memory| memory.created_at).min();
        let window_end = scoped.iter().map(|memory| memory.created_at).max();
        let conversation_count = usize::from(!scoped.is_empty());
        (conversation_count, window_start, window_end)
    }

    pub(crate) fn batch_window_for_global(
        &self,
    ) -> (usize, Option<DateTime<Utc>>, Option<DateTime<Utc>>) {
        let scoped = self
            .memories
            .iter()
            .filter(|memory| memory.conversation_id != "global")
            .collect::<Vec<_>>();
        let window_start = scoped.iter().map(|memory| memory.created_at).min();
        let window_end = scoped.iter().map(|memory| memory.created_at).max();
        let conversation_count = scoped
            .iter()
            .map(|memory| memory.conversation_id.as_str())
            .collect::<HashSet<_>>()
            .len();
        (conversation_count, window_start, window_end)
    }

    

    pub(crate) fn with_write_gate_thresholds(write_gate_thresholds: WriteGateThresholds) -> Self {
        let prefs = ["偏好", "风格", "习惯", "喜欢", "不喜欢"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        Self {
            memories: Vec::new(),
            shared_memory_visibility: HashMap::new(),
            agent_runtime: HashMap::new(),
            agent_tasks: HashMap::new(),
            workspace_registry: HashMap::new(),
            conversation_workspaces: HashMap::new(),
            admin_principals: HashMap::new(),
            admin_role_bindings: HashMap::new(),
            control_plane_policy_assignments: HashMap::new(),
            canonical_semantic_state: HashMap::new(),
            canonical_semantic_audit: Vec::new(),
            container_audit_log: Vec::new(),
            control_plane_audit_log: Vec::new(),
            connector_runs: Vec::new(),
            multimodal_extraction_runs: Vec::new(),
            connector_schedules: HashMap::new(),
            connector_webhook_secrets: HashMap::new(),
            connector_webhook_replay_cache: HashMap::new(),
            consolidation_jobs: Vec::new(),
            cache: HashMap::new(),
            candidate_cache_revision: 0,
            user_preferences: prefs,
            write_gate_thresholds,
            auto_consolidation: AutoConsolidationConfig::default(),
            memory_decay: MemoryDecayConfig::default(),
        }
    }

    pub(crate) fn with_runtime_config(
        write_gate_thresholds: WriteGateThresholds,
        auto_consolidation: AutoConsolidationConfig,
    ) -> Self {
        let mut engine = Self::with_write_gate_thresholds(write_gate_thresholds);
        engine.auto_consolidation = auto_consolidation;
        engine
    }

    /// Builds the bounded state needed after retrieval has already selected
    /// answer candidates. This avoids cloning unrelated durable runtime state
    /// for an ordinary answer request.
    pub(crate) fn answer_snapshot_for_memory_ids(&self, ids: &[u64]) -> Self {
        let selected_ids = ids.iter().copied().collect::<HashSet<_>>();
        let mut snapshot = Self::with_write_gate_thresholds(self.write_gate_thresholds.clone());
        snapshot.memories = self
            .memories
            .iter()
            .filter(|memory| selected_ids.contains(&memory.id))
            .cloned()
            .collect();

        snapshot.shared_memory_visibility = self
            .shared_memory_visibility
            .iter()
            .filter(|(memory_id, _)| selected_ids.contains(memory_id))
            .map(|(memory_id, visibility)| (*memory_id, visibility.clone()))
            .collect();
        let semantic_keys = snapshot
            .memories
            .iter()
            .filter_map(|memory| memory.semantic_node.as_ref())
            .map(|node| Self::canonical_semantic_key(&node.entity_id, &node.attribute_key))
            .collect::<HashSet<_>>();
        snapshot.canonical_semantic_state = self
            .canonical_semantic_state
            .iter()
            .filter(|(key, _)| semantic_keys.contains(*key))
            .map(|(key, node)| (key.clone(), node.clone()))
            .collect();
        snapshot.user_preferences = self.user_preferences.clone();
        snapshot.auto_consolidation = self.auto_consolidation.clone();
        snapshot.memory_decay = self.memory_decay.clone();
        snapshot
    }

    /// Scope the snapshot before cloning record bodies. IDs are nominations,
    /// not authority to copy a record from outside the requested conversation.
    pub(crate) fn answer_snapshot_for_scope(&self, conversation: Option<&str>, ids: Option<&[u64]>) -> Self {
        let nominated = ids.map(|ids| ids.iter().copied().collect::<HashSet<_>>());
        let selected = self.memories.iter()
            .filter(|record| conversation.is_none_or(|scope| record.conversation_id == scope))
            .filter(|record| nominated.as_ref().is_none_or(|ids| ids.contains(&record.id)))
            .map(|record| record.id).collect::<Vec<_>>();
        let mut snapshot = self.answer_snapshot_for_memory_ids(&selected);
        let conversations = snapshot.memories.iter().map(|record| record.conversation_id.as_str()).collect::<HashSet<_>>();
        snapshot.conversation_workspaces = self.conversation_workspaces.iter()
            .filter(|(conversation, _)| conversations.contains(conversation.as_str()))
            .map(|(conversation, workspace)| (conversation.clone(),workspace.clone())).collect();
        let workspaces = conversations.iter().map(|conversation| self.workspace_for_conversation(conversation)).collect::<HashSet<_>>();
        snapshot.workspace_registry = self.workspace_registry.iter().filter(|(workspace, _)| workspaces.contains(*workspace))
            .map(|(workspace, registry)| (workspace.clone(),registry.clone())).collect();
        snapshot
    }

    pub(crate) fn tokenize(s: &str) -> Vec<String> {
        s.to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|t| !t.is_empty())
            .map(|t| t.to_string())
            .collect()
    }

    pub(crate) fn activation_status_rank(status: &str) -> i32 {
        match status {
            "pinned" => 2,
            "active" => 1,
            _ => 0,
        }
    }

    pub(crate) fn activation_metadata_is_live(metadata: &ActivationMetadata) -> bool {
        if metadata.status == "cleared" {
            return false;
        }
        if metadata.status == "pinned" {
            return true;
        }
        metadata
            .expires_at
            .map(|expires_at| expires_at > Utc::now())
            .unwrap_or(true)
    }

    pub(crate) fn activation_memory_is_live(memory: &MemoryRecord) -> bool {
        memory.layer == MemoryLayer::Activation
            && memory
                .activation_metadata
                .as_ref()
                .map(Self::activation_metadata_is_live)
                .unwrap_or(false)
    }

    pub(crate) fn activation_score_bonus(memory: &MemoryRecord) -> f32 {
        if !Self::activation_memory_is_live(memory) {
            return 0.0;
        }
        match memory
            .activation_metadata
            .as_ref()
            .map(|metadata| metadata.status.as_str())
        {
            Some("pinned") => 0.85,
            Some("active") => 0.70,
            _ => 0.0,
        }
    }
}


#[cfg(test)]
mod answer_snapshot_tests {
    use super::*;

}
