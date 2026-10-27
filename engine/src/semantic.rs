use serde::{Deserialize, Serialize};

// ─── Semantic Scope ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticScope {
    Universal,
    Conversation,
    Project,
    Task,
}

// ─── Semantic Status ───────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticStatus {
    Active,
    Superseded,
    Disputed,
    Deprecated,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticKind {
    StableFact,
    ProcedurePrecondition,
    ProcedureRecoveryPattern,
    ProcedureFailureMotif,
    ProcedureSuccessPattern,
}

// ─── Semantic Node ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticNode {
    pub semantic_id: String,
    pub entity_id: String,
    pub entity_type: String,
    pub attribute_key: String,
    pub attribute_value: String,
    pub kind: SemanticKind,
    pub confidence: f32,
    pub scope: SemanticScope,
    pub status: SemanticStatus,
    pub applicable_project: Option<String>,
    pub applicable_task: Option<String>,
    pub provenance_anchor_ids: Vec<u64>,
    pub conflict_set_id: Option<String>,
    pub version: u32,
}

// ─── Conflict Structures ──────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticConflictMember {
    pub semantic_id: String,
    pub attribute_value: String,
    pub confidence: f32,
    pub provenance_anchor_ids: Vec<u64>,
    pub version: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticConflictSet {
    pub conflict_set_id: String,
    pub entity_id: String,
    pub attribute_key: String,
    pub members: Vec<SemanticConflictMember>,
    pub resolved: bool,
    pub winner_semantic_id: Option<String>,
}

// ─── Consolidation Helpers ─────────────────────────────────────────

/// Build a semantic node candidate from a fact memory record.
pub fn maybe_build_semantic_node(
    entity_id: &str,
    entity_type: &str,
    attribute_key: &str,
    attribute_value: &str,
    trust: f32,
    anchor_id: u64,
    scope: SemanticScope,
) -> SemanticNode {
    let semantic_id = format!("sem:{}:{}:{}", entity_id, attribute_key, anchor_id);
    let kind = infer_semantic_kind(attribute_key, attribute_value);
    let (applicable_project, applicable_task) =
        infer_semantic_applicability(attribute_value, &scope);
    SemanticNode {
        semantic_id,
        entity_id: entity_id.to_string(),
        entity_type: entity_type.to_string(),
        attribute_key: attribute_key.to_string(),
        attribute_value: attribute_value.to_string(),
        kind,
        confidence: trust.clamp(0.0, 1.0),
        scope,
        status: SemanticStatus::Active,
        applicable_project,
        applicable_task,
        provenance_anchor_ids: vec![anchor_id],
        conflict_set_id: None,
        version: 1,
    }
}

/// Merge confirming evidence into an existing semantic node.
/// Returns the updated node with increased confidence and additional provenance.
pub fn merge_semantic_confirmation(
    existing: &SemanticNode,
    new_anchor_id: u64,
    new_trust: f32,
) -> SemanticNode {
    let mut merged = existing.clone();
    // Confidence grows when confirming evidence arrives, with diminishing returns.
    let boost = new_trust * 0.1;
    merged.confidence = (merged.confidence + boost).clamp(0.0, 1.0);
    if !merged.provenance_anchor_ids.contains(&new_anchor_id) {
        merged.provenance_anchor_ids.push(new_anchor_id);
    }
    merged.version += 1;
    merged
}

/// Build a conflict set when contradictory facts arrive for the same entity + attribute.
pub fn build_semantic_conflict_set(
    entity_id: &str,
    attribute_key: &str,
    existing_node: &SemanticNode,
    contradicting_value: &str,
    contradicting_trust: f32,
    contradicting_anchor_id: u64,
) -> (SemanticConflictSet, SemanticNode, SemanticNode) {
    let conflict_set_id = format!(
        "conflict:{}:{}:{}",
        entity_id, attribute_key, contradicting_anchor_id
    );

    let member_a = SemanticConflictMember {
        semantic_id: existing_node.semantic_id.clone(),
        attribute_value: existing_node.attribute_value.clone(),
        confidence: existing_node.confidence,
        provenance_anchor_ids: existing_node.provenance_anchor_ids.clone(),
        version: existing_node.version,
    };

    let new_semantic_id = format!(
        "sem:{}:{}:{}",
        entity_id, attribute_key, contradicting_anchor_id
    );
    let member_b = SemanticConflictMember {
        semantic_id: new_semantic_id.clone(),
        attribute_value: contradicting_value.to_string(),
        confidence: contradicting_trust.clamp(0.0, 1.0),
        provenance_anchor_ids: vec![contradicting_anchor_id],
        version: 1,
    };

    let conflict_set = SemanticConflictSet {
        conflict_set_id: conflict_set_id.clone(),
        entity_id: entity_id.to_string(),
        attribute_key: attribute_key.to_string(),
        members: vec![member_a, member_b],
        resolved: false,
        winner_semantic_id: None,
    };

    // Mark existing node as disputed
    let mut disputed_existing = existing_node.clone();
    disputed_existing.status = SemanticStatus::Disputed;
    disputed_existing.conflict_set_id = Some(conflict_set_id.clone());

    // Create new disputed node
    let disputed_new = SemanticNode {
        semantic_id: new_semantic_id,
        entity_id: entity_id.to_string(),
        entity_type: existing_node.entity_type.clone(),
        attribute_key: attribute_key.to_string(),
        attribute_value: contradicting_value.to_string(),
        kind: existing_node.kind.clone(),
        confidence: contradicting_trust.clamp(0.0, 1.0),
        scope: existing_node.scope.clone(),
        status: SemanticStatus::Disputed,
        applicable_project: existing_node.applicable_project.clone(),
        applicable_task: existing_node.applicable_task.clone(),
        provenance_anchor_ids: vec![contradicting_anchor_id],
        conflict_set_id: Some(conflict_set_id),
        version: 1,
    };

    (conflict_set, disputed_existing, disputed_new)
}

// ─── Conflict State Transitions ────────────────────────────────────

/// Attempt to resolve a conflict set by selecting the stronger-evidence winner.
///
/// Rules (V1 simple):
/// - Higher version can win only if trust is not materially lower (within 0.15).
/// - If evidence is still ambiguous (both members have similar confidence and version),
///   keep disputed state.
pub fn try_resolve_conflict(conflict: &mut SemanticConflictSet) -> Option<String> {
    if conflict.members.len() < 2 || conflict.resolved {
        return conflict.winner_semantic_id.clone();
    }

    // Find the member with highest confidence, then version as tiebreaker
    let mut sorted = conflict.members.clone();
    sorted.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.version.cmp(&a.version))
    });

    let best = &sorted[0];
    let second = &sorted[1];

    // If the confidence gap is too small and version gap is also small, keep disputed
    let confidence_gap = best.confidence - second.confidence;
    let version_gap = best.version as i32 - second.version as i32;

    if confidence_gap < 0.10 && version_gap.abs() <= 1 {
        // Ambiguous — stay disputed
        return None;
    }

    // Higher version wins only if trust is not materially lower
    if version_gap > 0 && confidence_gap >= -0.15 {
        conflict.resolved = true;
        conflict.winner_semantic_id = Some(best.semantic_id.clone());
        return Some(best.semantic_id.clone());
    }

    // Higher confidence wins
    if confidence_gap >= 0.10 {
        conflict.resolved = true;
        conflict.winner_semantic_id = Some(best.semantic_id.clone());
        return Some(best.semantic_id.clone());
    }

    None
}

/// Apply resolution result to semantic nodes.
/// The winner becomes Active; losers become Superseded.
pub fn apply_conflict_resolution(
    winner_id: &str,
    existing_node: &mut SemanticNode,
    new_node: &mut SemanticNode,
) {
    if existing_node.semantic_id == winner_id {
        existing_node.status = SemanticStatus::Active;
        existing_node.conflict_set_id = None;
        new_node.status = SemanticStatus::Superseded;
    } else if new_node.semantic_id == winner_id {
        new_node.status = SemanticStatus::Active;
        new_node.conflict_set_id = None;
        existing_node.status = SemanticStatus::Superseded;
    }
}

pub fn semantic_status_score_adjustment(node: Option<&SemanticNode>) -> f32 {
    match node.map(|semantic| &semantic.status) {
        Some(SemanticStatus::Active) => 0.15,
        Some(SemanticStatus::Disputed) => -0.05,
        Some(SemanticStatus::Superseded) => -0.20,
        Some(SemanticStatus::Deprecated) => -0.30,
        None => 0.0,
    }
}

pub fn infer_semantic_scope(entity_id: &str, attribute_key: &str, content: &str) -> SemanticScope {
    let entity = entity_id.to_lowercase();
    let key = attribute_key.to_lowercase();
    let body = content.to_lowercase();

    if matches!(key.as_str(), "current_role" | "identity" | "long_term_goal")
        || body.contains("长期")
        || body.contains("总是")
        || body.contains("通用")
    {
        return SemanticScope::Universal;
    }

    if key.contains("project") || body.contains("项目") || body.contains("里程碑") {
        return SemanticScope::Project;
    }

    if body.contains("任务")
        || body.contains("本次")
        || body.contains("当前步骤")
        || body.contains("这个问题")
    {
        return SemanticScope::Task;
    }

    if matches!(entity.as_str(), "agent" | "self" | "assistant") {
        return SemanticScope::Universal;
    }

    SemanticScope::Conversation
}

pub fn semantic_scope_score_adjustment(node: Option<&SemanticNode>, query: &str) -> f32 {
    let Some(node) = node else {
        return 0.0;
    };

    let query = query.to_lowercase();
    let wants_project = query.contains("项目") || query.contains("project");
    let wants_task = query.contains("任务") || query.contains("步骤") || query.contains("当前问题");
    let wants_global = query.contains("长期")
        || query.contains("身份")
        || query.contains("角色")
        || query.contains("总是")
        || query.contains("什么是");

    match node.scope {
        SemanticScope::Project if wants_project => 0.08,
        SemanticScope::Task if wants_task => 0.08,
        SemanticScope::Universal if wants_global => 0.08,
        SemanticScope::Conversation if !(wants_project || wants_task || wants_global) => 0.03,
        SemanticScope::Conversation => -0.02,
        _ => 0.0,
    }
}

pub fn semantic_scope_allows_query(
    node: Option<&SemanticNode>,
    query: &str,
    current_project: Option<&str>,
    current_task: Option<&str>,
) -> bool {
    let Some(node) = node else {
        return true;
    };

    let query = query.to_lowercase();
    let wants_project = query.contains("项目") || query.contains("project");
    let wants_task = query.contains("任务") || query.contains("步骤") || query.contains("当前问题");
    let wants_global = query.contains("长期")
        || query.contains("身份")
        || query.contains("角色")
        || query.contains("总是")
        || query.contains("你是谁")
        || query.contains("who are you");

    let project_match = node
        .applicable_project
        .as_deref()
        .map(|project| {
            current_project
                .map(|current| current.eq_ignore_ascii_case(project))
                .unwrap_or(false)
                || query.contains(&project.to_lowercase())
        })
        .unwrap_or(false);
    let task_match = node
        .applicable_task
        .as_deref()
        .map(|task| {
            current_task
                .map(|current| current.eq_ignore_ascii_case(task))
                .unwrap_or(false)
                || query.contains(&task.to_lowercase())
        })
        .unwrap_or(false);

    match node.scope {
        SemanticScope::Task => {
            if wants_global || wants_project {
                return false;
            }
            node.applicable_task.is_none() || wants_task || task_match
        }
        SemanticScope::Project => {
            if wants_global || wants_task {
                return false;
            }
            node.applicable_project.is_none() || wants_project || project_match
        }
        // Universal knowledge is a valid fallback in narrower contexts. Scope
        // and applicability scoring still rank matching project/task evidence
        // ahead of it.
        SemanticScope::Universal => true,
        SemanticScope::Conversation => true,
    }
}

pub fn semantic_kind_score_adjustment(node: Option<&SemanticNode>, query: &str) -> f32 {
    let Some(node) = node else {
        return 0.0;
    };
    let query = query.to_lowercase();
    let wants_precondition =
        query.contains("前置") || query.contains("准备") || query.contains("先检查");
    let wants_recovery = query.contains("恢复") || query.contains("重试");
    let wants_failure = query.contains("失败原因")
        || query.contains("为什么失败")
        || query.contains("失败模式")
        || query.contains("风险")
        || query.contains("容易失败");

    match node.kind {
        SemanticKind::ProcedurePrecondition if wants_precondition => 0.06,
        SemanticKind::ProcedureRecoveryPattern if wants_recovery => 0.06,
        SemanticKind::ProcedureFailureMotif if wants_failure => 0.06,
        SemanticKind::StableFact if !(wants_precondition || wants_recovery || wants_failure) => {
            0.02
        }
        _ => 0.0,
    }
}

pub fn semantic_applicability_score_adjustment(
    node: Option<&SemanticNode>,
    query: &str,
    current_project: Option<&str>,
    current_task: Option<&str>,
) -> f32 {
    let Some(node) = node else {
        return 0.0;
    };

    let query = query.to_lowercase();
    let project_match = node
        .applicable_project
        .as_deref()
        .map(|project| {
            let project_lower = project.to_lowercase();
            current_project
                .map(|current| current.eq_ignore_ascii_case(project))
                .unwrap_or(false)
                || query.contains(&project_lower)
        })
        .unwrap_or(false);
    let task_match = node
        .applicable_task
        .as_deref()
        .map(|task| {
            let task_lower = task.to_lowercase();
            current_task
                .map(|current| current.eq_ignore_ascii_case(task))
                .unwrap_or(false)
                || query.contains(&task_lower)
        })
        .unwrap_or(false);

    let mentions_project = query.contains("项目") || query.contains("project");
    let mentions_task =
        query.contains("任务") || query.contains("步骤") || query.contains("当前问题");

    let mut score = 0.0;
    if project_match {
        score += 0.07;
    } else if node.applicable_project.is_some() && mentions_project {
        score -= 0.03;
    }
    if task_match {
        score += 0.07;
    } else if node.applicable_task.is_some() && mentions_task {
        score -= 0.03;
    }
    score
}

pub fn infer_semantic_kind(attribute_key: &str, attribute_value: &str) -> SemanticKind {
    let key = attribute_key.to_lowercase();
    let value = attribute_value.to_lowercase();
    if key.ends_with("_recovery_pattern") || value.contains("重试") || value.contains("恢复模式")
    {
        SemanticKind::ProcedureRecoveryPattern
    } else if key.ends_with("_failure_motif")
        || value.contains("失败模式")
        || value.contains("常见失败")
    {
        SemanticKind::ProcedureFailureMotif
    } else if key.ends_with("_precondition") || value.contains("前置条件") {
        SemanticKind::ProcedurePrecondition
    } else if key.ends_with("_success_pattern") || value.contains("已验证证据") {
        SemanticKind::ProcedureSuccessPattern
    } else {
        SemanticKind::StableFact
    }
}

pub fn infer_semantic_applicability(
    attribute_value: &str,
    scope: &SemanticScope,
) -> (Option<String>, Option<String>) {
    let value = attribute_value.trim();
    match scope {
        SemanticScope::Project => (Some(value.to_string()), None),
        SemanticScope::Task => (None, Some(value.to_string())),
        _ => (None, None),
    }
}

pub fn semantic_scope_label(scope: &SemanticScope) -> &'static str {
    match scope {
        SemanticScope::Universal => "全局",
        SemanticScope::Conversation => "会话",
        SemanticScope::Project => "项目",
        SemanticScope::Task => "任务",
    }
}




#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_node_creation() {
        let node = maybe_build_semantic_node(
            "role",
            "character",
            "weapon",
            "剑",
            0.9,
            100,
            SemanticScope::Conversation,
        );
        assert_eq!(node.entity_id, "role");
        assert_eq!(node.attribute_key, "weapon");
        assert_eq!(node.attribute_value, "剑");
        assert_eq!(node.kind, SemanticKind::StableFact);
        assert_eq!(node.confidence, 0.9);
        assert_eq!(node.status, SemanticStatus::Active);
        assert_eq!(node.provenance_anchor_ids, vec![100]);
        assert!(node.conflict_set_id.is_none());
        assert_eq!(node.version, 1);
    }

    #[test]
    fn semantic_merge_confirmation_increases_confidence() {
        let node = maybe_build_semantic_node(
            "role",
            "character",
            "weapon",
            "剑",
            0.7,
            100,
            SemanticScope::Conversation,
        );
        let merged = merge_semantic_confirmation(&node, 200, 0.9);
        assert!(merged.confidence > node.confidence);
        assert_eq!(merged.provenance_anchor_ids, vec![100, 200]);
        assert_eq!(merged.version, 2);
        assert_eq!(merged.status, SemanticStatus::Active);
    }

    #[test]
    fn semantic_conflict_set_from_contradictory_facts() {
        let node = maybe_build_semantic_node(
            "role",
            "character",
            "weapon",
            "剑",
            0.9,
            100,
            SemanticScope::Conversation,
        );
        let (conflict, existing, new) =
            build_semantic_conflict_set("role", "weapon", &node, "弓", 0.85, 200);
        assert_eq!(conflict.members.len(), 2);
        assert!(!conflict.resolved);
        assert!(conflict.winner_semantic_id.is_none());
        assert_eq!(existing.status, SemanticStatus::Disputed);
        assert_eq!(new.status, SemanticStatus::Disputed);
        assert_eq!(new.attribute_value, "弓");
        assert!(existing.conflict_set_id.is_some());
        assert_eq!(existing.conflict_set_id, new.conflict_set_id);
    }

    #[test]
    fn semantic_conflict_resolution_by_confidence() {
        let node = maybe_build_semantic_node(
            "role",
            "character",
            "weapon",
            "剑",
            0.9,
            100,
            SemanticScope::Conversation,
        );
        let (mut conflict, mut existing, mut new) =
            build_semantic_conflict_set("role", "weapon", &node, "弓", 0.5, 200);
        // Large confidence gap → should resolve
        let winner = try_resolve_conflict(&mut conflict);
        assert!(winner.is_some());
        assert!(conflict.resolved);

        apply_conflict_resolution(winner.as_ref().unwrap(), &mut existing, &mut new);
        // The higher confidence node (剑, 0.9) should win
        assert_eq!(existing.status, SemanticStatus::Active);
        assert_eq!(new.status, SemanticStatus::Superseded);
    }

    #[test]
    fn semantic_conflict_stays_disputed_when_ambiguous() {
        let node = maybe_build_semantic_node(
            "role",
            "character",
            "weapon",
            "剑",
            0.85,
            100,
            SemanticScope::Conversation,
        );
        let (mut conflict, existing, new) =
            build_semantic_conflict_set("role", "weapon", &node, "弓", 0.82, 200);
        // Small confidence gap + same version → ambiguous
        let winner = try_resolve_conflict(&mut conflict);
        assert!(winner.is_none());
        assert!(!conflict.resolved);
        assert_eq!(existing.status, SemanticStatus::Disputed);
        assert_eq!(new.status, SemanticStatus::Disputed);
    }

    #[test]
    fn infer_scope_distinguishes_universal_project_and_task() {
        assert_eq!(
            infer_semantic_scope("agent", "current_role", "agent 当前角色是架构师"),
            SemanticScope::Universal
        );
        assert_eq!(
            infer_semantic_scope("agent", "active_project", "当前项目是语义记忆改造"),
            SemanticScope::Project
        );
        assert_eq!(
            infer_semantic_scope("build", "next_step", "当前任务需要先补测试再改代码"),
            SemanticScope::Task
        );
    }

    #[test]
    fn infer_kind_distinguishes_internalized_procedure_knowledge() {
        assert_eq!(
            infer_semantic_kind(
                "debug_strategy_precondition",
                "执行前检查日志链路的前置条件"
            ),
            SemanticKind::ProcedurePrecondition
        );
        assert_eq!(
            infer_semantic_kind(
                "debug_strategy_recovery_pattern",
                "若 再收集日志 失败，先补齐前置条件再重试"
            ),
            SemanticKind::ProcedureRecoveryPattern
        );
        assert_eq!(
            infer_semantic_kind("debug_strategy_failure_motif", "日志缺失"),
            SemanticKind::ProcedureFailureMotif
        );
        assert_eq!(
            infer_semantic_kind("weapon", "剑"),
            SemanticKind::StableFact
        );
    }
}
