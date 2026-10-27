use crate::{
    api_types::IngestRequest,
    support::PerceptionEvent,
    EpisodicMetadata, EventMetadata, MemoryLayer, MemoryRecord, ProcedureSchema,
};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryIntent {
    SelfModel,
    Autobiographical,
    Foresight,
    Episodic,
    Procedural,
    Semantic,
    General,
}

#[derive(Debug, Clone)]
pub struct MemorySchedule {
    pub preferred_layers: Vec<MemoryLayer>,
}

pub fn detect_query_intent(query: &str) -> QueryIntent {
    let q = query.to_lowercase();
    let episodic_failure_recall = contains_any(
        &q,
        &[
            "失败模式",
            "类似失败",
            "相似失败",
            "之前失败",
            "过去失败",
            "为什么失败",
        ],
    ) || ((q.contains("类似")
        || q.contains("相似")
        || q.contains("之前")
        || q.contains("过去"))
        && (q.contains("失败") || q.contains("问题") || q.contains("出错")));

    if contains_any(
        &q,
        &[
            "接下来要做什么",
            "接着要做什么",
            "下一步做什么",
            "计划是什么",
            "打算做什么",
            "明天要做什么",
            "下周要做什么",
            "截止什么时候",
            "答应要做什么",
        ],
    ) {
        QueryIntent::Foresight
    } else if contains_any(&q, &["执行前", "准备什么"]) {
        QueryIntent::Procedural
    } else if contains_any(
        &q,
        &[
            "正在推进什么项目",
            "推进什么项目",
            "关键承诺",
            "长期目标是什么",
            "现在在做什么",
            "当前项目是什么",
        ],
    ) {
        QueryIntent::Autobiographical
    } else if contains_any(
        &q,
        &[
            "你是谁",
            "我是谁",
            "身份",
            "角色",
            "长期目标",
            "承诺",
            "当前项目",
        ],
    ) {
        QueryIntent::SelfModel
    } else if episodic_failure_recall {
        QueryIntent::Episodic
    } else if contains_any(
        &q,
        &[
            "发生了什么",
            "刚才",
            "经历",
            "过程",
            "episode",
            "timeline",
            "回顾",
        ],
    ) {
        QueryIntent::Episodic
    } else if contains_any(
        &q,
        &[
            "怎么做",
            "怎么写",
            "步骤",
            "流程",
            "如何处理",
            "如何定位",
            "调试",
            "排查",
        ],
    ) {
        QueryIntent::Procedural
    } else if contains_any(
        &q,
        &[
            "属性",
            "特征",
            "是什么",
            "什么是",
            "entity",
            "attribute",
            "信息",
            "知识",
            "武器",
            "角色",
            "设定",
            "稳定事实",
        ],
    ) {
        QueryIntent::Semantic
    } else {
        QueryIntent::General
    }
}

pub fn layer_boost(intent: QueryIntent, layer: &MemoryLayer) -> f32 {
    match intent {
        QueryIntent::SelfModel => match layer {
            MemoryLayer::SelfModel => 0.35,
            MemoryLayer::Fact => 0.10,
            MemoryLayer::Parametric => 0.06,
            MemoryLayer::Event => -0.15,
            MemoryLayer::Episodic => -0.10,
            MemoryLayer::Working => -0.10,
            _ => 0.0,
        },
        QueryIntent::Autobiographical => match layer {
            MemoryLayer::Autobiographical => 0.55,
            MemoryLayer::Foresight => 0.18,
            MemoryLayer::SelfModel => 0.15,
            MemoryLayer::Fact => 0.08,
            MemoryLayer::Working => -0.08,
            MemoryLayer::Event => -0.10,
            _ => 0.0,
        },
        QueryIntent::Foresight => match layer {
            MemoryLayer::Foresight => 0.45,
            MemoryLayer::Autobiographical => 0.10,
            MemoryLayer::SessionSummary => 0.06,
            _ => 0.0,
        },
        QueryIntent::Episodic => match layer {
            MemoryLayer::Episodic => 0.20,
            MemoryLayer::Event => 0.10,
            MemoryLayer::SessionSummary => 0.08,
            _ => 0.0,
        },
        QueryIntent::Procedural => match layer {
            MemoryLayer::Procedure => 0.35,
            MemoryLayer::Parametric => 0.12,
            MemoryLayer::SessionSummary => 0.08,
            MemoryLayer::Fact => 0.05,
            _ => 0.0,
        },
        QueryIntent::General => match layer {
            MemoryLayer::Fact | MemoryLayer::Procedure => 0.05,
            MemoryLayer::Parametric => 0.10,
            _ => 0.0,
        },
        QueryIntent::Semantic => match layer {
            MemoryLayer::Parametric => 0.30,
            MemoryLayer::Fact => 0.24,
            MemoryLayer::SelfModel => 0.08,
            _ => 0.0,
        },
    }
}

pub fn schedule_for_intent(intent: QueryIntent) -> Option<MemorySchedule> {
    let preferred_layers = match intent {
        QueryIntent::SelfModel => vec![
            MemoryLayer::SelfModel,
            MemoryLayer::Autobiographical,
            MemoryLayer::Fact,
        ],
        QueryIntent::Autobiographical => vec![
            MemoryLayer::Autobiographical,
            MemoryLayer::Foresight,
            MemoryLayer::SelfModel,
            MemoryLayer::Fact,
        ],
        QueryIntent::Foresight => vec![
            MemoryLayer::Foresight,
            MemoryLayer::Autobiographical,
            MemoryLayer::SessionSummary,
        ],
        QueryIntent::Episodic => vec![
            MemoryLayer::Episodic,
            MemoryLayer::Event,
            MemoryLayer::SessionSummary,
        ],
        QueryIntent::Procedural => vec![
            MemoryLayer::Procedure,
            MemoryLayer::Parametric,
            MemoryLayer::SessionSummary,
            MemoryLayer::Fact,
        ],
        QueryIntent::General => return None,
        QueryIntent::Semantic => vec![
            MemoryLayer::Parametric,
            MemoryLayer::Fact,
            MemoryLayer::SelfModel,
            MemoryLayer::Autobiographical,
        ],
    };

    Some(MemorySchedule { preferred_layers })
}

pub fn schedule_rank_boost(preferred_layers: &[MemoryLayer], layer: &MemoryLayer) -> f32 {
    preferred_layers
        .iter()
        .position(|candidate| candidate == layer)
        .map(|idx| match idx {
            0 => 0.18,
            1 => 0.10,
            2 => 0.04,
            _ => 0.0,
        })
        .unwrap_or(0.0)
}

pub fn memory_fits_schedule(
    intent: QueryIntent,
    memory: &MemoryRecord,
    preferred_layers: &[MemoryLayer],
) -> bool {
    if !preferred_layers.iter().any(|layer| layer == &memory.layer) {
        return false;
    }

    match intent {
        QueryIntent::Episodic if memory.layer == MemoryLayer::Event => memory
            .event_metadata
            .as_ref()
            .map(|metadata| metadata.origin_layer == MemoryLayer::Working)
            .unwrap_or(false),
        _ => true,
    }
}

pub fn should_exclude_from_retrieval(memory: &MemoryRecord) -> bool {
    if memory
        .event_metadata
        .as_ref()
        .map(|metadata| metadata.memory_action == "ignore")
        .unwrap_or(false)
    {
        return true;
    }

    if memory.layer == MemoryLayer::Archive {
        let low_trust = memory.trust < 0.6;
        let low_signal = memory.content.contains("临时噪声")
            || memory.content.contains("流水记录")
            || memory.content.contains("低价值");
        if low_trust && low_signal {
            return true;
        }
    }

    false
}

pub fn extract_procedure_schema(req: &IngestRequest) -> Option<ProcedureSchema> {
    if req.layer != MemoryLayer::Procedure {
        return None;
    }

    let mut trigger_conditions = Vec::new();
    let mut steps = Vec::new();
    let mut rollback_steps = Vec::new();
    let mut success_criteria = Vec::new();

    for raw_line in req.content.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(value) = strip_prefixes(line, &["触发条件:", "触发:", "when:"]) {
            trigger_conditions.push(value.to_string());
            continue;
        }
        if let Some(value) = strip_prefixes(line, &["步骤:", "step:", "- "]) {
            steps.push(value.to_string());
            continue;
        }
        if let Some(value) = strip_prefixes(line, &["失败回滚:", "回滚:", "rollback:"]) {
            rollback_steps.push(value.to_string());
            continue;
        }
        if let Some(value) = strip_prefixes(line, &["成功判据:", "成功标准:", "success:"]) {
            success_criteria.push(value.to_string());
        }
    }

    if trigger_conditions.is_empty()
        && steps.is_empty()
        && rollback_steps.is_empty()
        && success_criteria.is_empty()
    {
        return None;
    }

    Some(ProcedureSchema {
        trigger_conditions,
        steps,
        rollback_steps,
        success_criteria,
    })
}

pub fn build_event_record(
    id: u64,
    req: &IngestRequest,
    tokens: Vec<String>,
    created_at: DateTime<Utc>,
) -> MemoryRecord {
    let source = req.source.clone().unwrap_or_else(|| "user".to_string());
    let entity = req.entity.clone().unwrap_or_else(|| "unknown".to_string());
    let content = format!(
        "event: layer={:?}; source={}; entity={}; content={}",
        req.layer,
        source,
        entity,
        req.content.trim()
    );
    let perception = normalize_perception_event(req);
    let intent = perception.intent.clone();
    let value_weight = perception.value_weight;
    let memory_action = infer_memory_action(req, value_weight);

    MemoryRecord {
        id,
        conversation_id: req.conversation_id.clone(),
        layer: MemoryLayer::Event,
        content,
        tokens,
        source: "event_ingestion".to_string(),
        trust: req.trust.unwrap_or(0.8).clamp(0.0, 1.0),
        created_at,
        anchor_ids: Vec::new(),
        entity: req.entity.clone(),
        fact_key: None,
        fact_value: None,
        memory_type: None,
        confidence: None,
        freshness: None,
        importance: None,
        last_verified_at: None,
        supersedes: Vec::new(),
        superseded_by: Vec::new(),
        version: 1,
        event_metadata: Some(EventMetadata {
            origin_layer: req.layer.clone(),
            intent,
            result: perception.result,
            memory_action,
            confidence: perception.confidence,
            value_weight,
        }),
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
    }
}

pub fn normalize_perception_event(req: &IngestRequest) -> PerceptionEvent {
    let source_type = req
        .source
        .clone()
        .unwrap_or_else(|| "conversation".to_string());
    let event_type = match req.layer {
        MemoryLayer::Fact => "fact_ingest",
        MemoryLayer::Parametric => "parametric_ingest",
        MemoryLayer::Foresight => "foresight_ingest",
        MemoryLayer::Procedure => "procedure_ingest",
        MemoryLayer::Working => "working_update",
        MemoryLayer::Event => "event_ingest",
        _ => "memory_ingest",
    }
    .to_string();
    let actor = req.entity.clone();
    let target = req.fact_key.clone();
    let intent = infer_event_intent(req);
    let value_weight = infer_value_weight(req, &intent);
    let evidence_tags = infer_perception_evidence_tags(
        &source_type,
        &req.content,
        &serde_json::json!({
            "fact_key": req.fact_key,
            "entity": req.entity,
        }),
    );
    let normalized_summary = format!("{}:{}", source_type, intent);
    PerceptionEvent {
        event_type,
        source_type,
        normalizer_id: "ingest_default_v1".to_string(),
        actor,
        target,
        intent,
        result: "stored".to_string(),
        confidence: req.trust.unwrap_or(0.8).clamp(0.0, 1.0),
        value_weight,
        evidence_tags,
        normalized_summary,
        raw_payload: serde_json::json!({
            "content": req.content,
            "entity": req.entity,
            "fact_key": req.fact_key,
            "fact_value": req.fact_value,
            "layer": req.layer,
        }),
    }
}

fn infer_perception_evidence_tags(
    source_type: &str,
    content: &str,
    raw_payload: &serde_json::Value,
) -> Vec<String> {
    let mut tags = Vec::new();
    match source_type {
        "tool" => {
            tags.push("tool_output".to_string());
            if raw_payload.get("tool").is_some() {
                tags.push("tool_name_present".to_string());
            }
        }
        "file" => {
            tags.push("file_change".to_string());
            if raw_payload.get("path").is_some() {
                tags.push("file_path_present".to_string());
            }
        }
        "runtime" => {
            tags.push("runtime_signal".to_string());
        }
        "conversation" => {
            tags.push("conversation_act".to_string());
        }
        _ => tags.push("generic_signal".to_string()),
    }

    if content.contains("失败") || content.contains("错误") {
        tags.push("failure_signal".to_string());
    }
    if content.contains("成功") || content.contains("完成") {
        tags.push("success_signal".to_string());
    }
    tags
}

fn infer_event_intent(req: &IngestRequest) -> String {
    match req.layer {
        MemoryLayer::Fact => "fact_update",
        MemoryLayer::Activation => "activation_capture",
        MemoryLayer::Parametric => "parametric_capture",
        MemoryLayer::Foresight => "foresight_capture",
        MemoryLayer::Procedure => "procedure_capture",
        MemoryLayer::Working => "working_memory_capture",
        MemoryLayer::SessionSummary => "session_summary_capture",
        MemoryLayer::Archive => "archive_capture",
        MemoryLayer::SelfModel => "self_model_capture",
        MemoryLayer::Autobiographical => "autobiographical_capture",
        MemoryLayer::Episodic => "episodic_capture",
        MemoryLayer::Event => "event_capture",
    }
    .to_string()
}

fn infer_value_weight(req: &IngestRequest, intent: &str) -> f32 {
    let base: f32 = match req.layer {
        MemoryLayer::Fact
        | MemoryLayer::Activation
        | MemoryLayer::Parametric
        | MemoryLayer::Foresight
        | MemoryLayer::SelfModel
        | MemoryLayer::Autobiographical => 0.8,
        MemoryLayer::Procedure | MemoryLayer::Episodic | MemoryLayer::SessionSummary => 0.65,
        MemoryLayer::Working | MemoryLayer::Event => 0.45,
        MemoryLayer::Archive => 0.3,
    };
    let intent_bonus: f32 = if intent.contains("fact") || intent.contains("self_model") {
        0.1
    } else {
        0.0
    };
    (base + intent_bonus).clamp(0.0, 1.0)
}

fn infer_memory_action(req: &IngestRequest, value_weight: f32) -> String {
    let action = match req.layer {
        MemoryLayer::Fact
        | MemoryLayer::Activation
        | MemoryLayer::Parametric
        | MemoryLayer::Foresight
        | MemoryLayer::SelfModel
        | MemoryLayer::Autobiographical => {
            if value_weight >= 0.85 {
                "consolidate"
            } else {
                "retain"
            }
        }
        MemoryLayer::Procedure | MemoryLayer::Episodic | MemoryLayer::SessionSummary => "retain",
        MemoryLayer::Working => "defer",
        MemoryLayer::Archive => {
            if value_weight < 0.4 {
                "ignore"
            } else {
                "defer"
            }
        }
        MemoryLayer::Event => "retain",
    };
    action.to_string()
}

pub fn maybe_build_episode(
    conversation_id: &str,
    memories: &[&MemoryRecord],
    id: u64,
    created_at: DateTime<Utc>,
) -> Option<MemoryRecord> {
    let recent_context = memories
        .iter()
        .filter(|m| {
            m.conversation_id == conversation_id
                && matches!(m.layer, MemoryLayer::Working | MemoryLayer::Event)
        })
        .rev()
        .take(4)
        .map(|m| (*m).clone())
        .collect::<Vec<_>>();

    let working_count = recent_context
        .iter()
        .filter(|m| m.layer == MemoryLayer::Working)
        .count();
    if working_count < 2 {
        return None;
    }

    let mut anchor_ids = recent_context.iter().map(|m| m.id).collect::<Vec<_>>();
    anchor_ids.sort_unstable();

    if memories.iter().any(|m| {
        m.conversation_id == conversation_id
            && m.layer == MemoryLayer::Episodic
            && normalized_anchor_ids(&m.anchor_ids) == anchor_ids
    }) {
        return None;
    }

    let ordered = recent_context.into_iter().rev().collect::<Vec<_>>();
    let summary = ordered
        .iter()
        .map(|m| {
            let prefix = match m.layer {
                MemoryLayer::Event => "event",
                MemoryLayer::Working => "working",
                _ => "memory",
            };
            format!("{}: {}", prefix, m.content.trim())
        })
        .collect::<Vec<_>>()
        .join(" -> ");
    let content = format!("episode: {}", summary);
    let episodic_metadata = build_episodic_metadata(&ordered);

    Some(MemoryRecord {
        id,
        conversation_id: conversation_id.to_string(),
        layer: MemoryLayer::Episodic,
        content: content.clone(),
        tokens: tokenize_like_engine(&content),
        source: "consolidation_engine".to_string(),
        trust: 0.78,
        created_at,
        anchor_ids,
        entity: None,
        fact_key: None,
        fact_value: None,
        memory_type: None,
        confidence: None,
        freshness: None,
        importance: None,
        last_verified_at: None,
        supersedes: Vec::new(),
        superseded_by: Vec::new(),
        version: 1,
        event_metadata: None,
        procedure_schema: None,
        procedure_execution_state: None,
        procedure_governance: None,
        episodic_metadata: Some(episodic_metadata),
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
    })
}

pub fn build_episodic_metadata(ordered: &[MemoryRecord]) -> EpisodicMetadata {
    let actors = ordered
        .iter()
        .filter_map(|m| m.entity.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let actions = ordered
        .iter()
        .map(|m| match m.layer {
            MemoryLayer::Event => "event_capture".to_string(),
            MemoryLayer::Working => "working_step".to_string(),
            _ => "memory_step".to_string(),
        })
        .collect::<Vec<_>>();
    let constraints = ordered
        .iter()
        .filter(|m| {
            m.content.contains("约束") || m.content.contains("限制") || m.content.contains("必须")
        })
        .map(|m| m.content.clone())
        .collect::<Vec<_>>();
    let outcome = ordered
        .iter()
        .rev()
        .find_map(|m| {
            m.procedure_execution_state
                .as_ref()
                .map(|state| state.outcome.clone())
        })
        .or_else(|| {
            ordered
                .iter()
                .rev()
                .find_map(|m| m.event_metadata.as_ref().map(|meta| meta.result.clone()))
        });
    let failure_type = ordered.iter().rev().find_map(|m| {
        m.procedure_execution_state
            .as_ref()
            .and_then(|state| state.failure_reason.clone())
    });
    let result = ordered
        .iter()
        .rev()
        .find_map(|m| {
            m.procedure_execution_state.as_ref().and_then(|state| {
                state
                    .success_evidence
                    .clone()
                    .or(state.failure_reason.clone())
            })
        })
        .or_else(|| ordered.last().map(|m| m.content.clone()));
    let cause_links = ordered
        .windows(2)
        .map(|pair| format!("{} -> {}", pair[0].id, pair[1].id))
        .collect::<Vec<_>>();
    let recovery_links = ordered
        .iter()
        .filter_map(|m| {
            m.procedure_execution_state.as_ref().and_then(|state| {
                if state.outcome == "failed" {
                    Some(
                        state
                            .steps
                            .iter()
                            .filter(|step| step.status == "failed")
                            .map(|step| format!("recover:{}", step.step))
                            .collect::<Vec<_>>(),
                    )
                } else {
                    None
                }
            })
        })
        .flatten()
        .collect::<Vec<_>>();
    let actor_action_pairs = ordered
        .iter()
        .map(|m| {
            format!(
                "{}::{}",
                m.entity
                    .clone()
                    .unwrap_or_else(|| "unknown_actor".to_string()),
                match m.layer {
                    MemoryLayer::Event => "event_capture",
                    MemoryLayer::Working => "working_step",
                    _ => "memory_step",
                }
            )
        })
        .collect::<Vec<_>>();
    let action_result_pairs = ordered
        .iter()
        .map(|m| {
            let result = m
                .procedure_execution_state
                .as_ref()
                .map(|s| s.outcome.clone())
                .or_else(|| m.event_metadata.as_ref().map(|e| e.result.clone()))
                .unwrap_or_else(|| "observed".to_string());
            format!(
                "{}::{}",
                match m.layer {
                    MemoryLayer::Event => "event_capture",
                    MemoryLayer::Working => "working_step",
                    _ => "memory_step",
                },
                result
            )
        })
        .collect::<Vec<_>>();
    EpisodicMetadata {
        actors,
        actions,
        constraints,
        result,
        outcome,
        cause_links,
        recovery_links,
        failure_type,
        actor_action_pairs,
        action_result_pairs,
    }
}

pub fn maybe_build_self_model(
    req: &IngestRequest,
    memories: &[&MemoryRecord],
    id: u64,
    created_at: DateTime<Utc>,
) -> Option<MemoryRecord> {
    if req.layer != MemoryLayer::Fact {
        return None;
    }

    let entity = req.entity.as_deref()?.to_lowercase();
    let fact_key = req.fact_key.as_deref()?.to_lowercase();
    let fact_value = req.fact_value.as_deref()?.to_string();

    if !matches!(entity.as_str(), "agent" | "self" | "assistant") {
        return None;
    }

    if !matches!(
        fact_key.as_str(),
        "current_role" | "long_term_goal" | "commitment" | "active_project" | "identity"
    ) {
        return None;
    }

    if memories.iter().any(|m| {
        m.conversation_id == req.conversation_id
            && m.layer == MemoryLayer::SelfModel
            && m.fact_key.as_deref() == Some(fact_key.as_str())
            && m.fact_value.as_deref() == Some(fact_value.as_str())
    }) {
        return None;
    }

    let version = memories
        .iter()
        .filter(|m| {
            m.conversation_id == req.conversation_id
                && m.layer == MemoryLayer::SelfModel
                && m.fact_key.as_deref() == Some(fact_key.as_str())
        })
        .map(|m| m.version)
        .max()
        .unwrap_or(0)
        + 1;

    let content = format!("自我模型：{}", req.content.trim());
    Some(MemoryRecord {
        id,
        conversation_id: req.conversation_id.clone(),
        layer: MemoryLayer::SelfModel,
        content: content.clone(),
        tokens: tokenize_like_engine(&content),
        source: "self_model_manager".to_string(),
        trust: req.trust.unwrap_or(0.9).clamp(0.0, 1.0),
        created_at,
        anchor_ids: Vec::new(),
        entity: Some(entity),
        fact_key: Some(fact_key),
        fact_value: Some(fact_value),
        memory_type: None,
        confidence: None,
        freshness: None,
        importance: None,
        last_verified_at: None,
        supersedes: Vec::new(),
        superseded_by: Vec::new(),
        version,
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
    })
}

pub fn maybe_build_autobiographical(
    req: &IngestRequest,
    memories: &[&MemoryRecord],
    id: u64,
    created_at: DateTime<Utc>,
) -> Option<MemoryRecord> {
    if req.layer != MemoryLayer::Fact {
        return None;
    }

    let entity = req.entity.as_deref()?.to_lowercase();
    if !matches!(entity.as_str(), "agent" | "self" | "assistant") {
        return None;
    }

    let relevant = memories
        .iter()
        .filter(|m| {
            m.conversation_id == req.conversation_id
                && m.layer == MemoryLayer::SelfModel
                && matches!(
                    m.fact_key.as_deref(),
                    Some("current_role")
                        | Some("long_term_goal")
                        | Some("commitment")
                        | Some("active_project")
                        | Some("identity")
                )
        })
        .collect::<Vec<_>>();

    if relevant.is_empty() {
        return None;
    }

    let latest_value = |fact_key: &str| {
        relevant
            .iter()
            .filter(|m| m.fact_key.as_deref() == Some(fact_key))
            .max_by_key(|m| m.version)
            .and_then(|m| m.fact_value.clone())
    };

    let project = latest_value("active_project");
    let goal = latest_value("long_term_goal");
    let commitment = latest_value("commitment");
    let role = latest_value("current_role").or_else(|| latest_value("identity"));

    if project.is_none() && goal.is_none() && commitment.is_none() && role.is_none() {
        return None;
    }

    let content = format!(
        "自传记忆：我当前的角色是{}；我正在推进的项目是{}；我的长期目标是{}；我的关键承诺是{}",
        role.unwrap_or_else(|| "未知".to_string()),
        project.unwrap_or_else(|| "未定义".to_string()),
        goal.unwrap_or_else(|| "未定义".to_string()),
        commitment.unwrap_or_else(|| "未定义".to_string()),
    );

    if memories.iter().any(|m| {
        m.conversation_id == req.conversation_id
            && m.layer == MemoryLayer::Autobiographical
            && m.content == content
    }) {
        return None;
    }

    let version = memories
        .iter()
        .filter(|m| {
            m.conversation_id == req.conversation_id && m.layer == MemoryLayer::Autobiographical
        })
        .map(|m| m.version)
        .max()
        .unwrap_or(0)
        + 1;

    Some(MemoryRecord {
        id,
        conversation_id: req.conversation_id.clone(),
        layer: MemoryLayer::Autobiographical,
        content: content.clone(),
        tokens: tokenize_like_engine(&content),
        source: "autobiographical_manager".to_string(),
        trust: req.trust.unwrap_or(0.9).clamp(0.0, 1.0),
        created_at,
        anchor_ids: relevant.iter().map(|m| m.id).collect(),
        entity: Some(entity),
        fact_key: Some("autobiographical_snapshot".to_string()),
        fact_value: Some(content),
        memory_type: None,
        confidence: None,
        freshness: None,
        importance: None,
        last_verified_at: None,
        supersedes: Vec::new(),
        superseded_by: Vec::new(),
        version,
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
    })
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

fn strip_prefixes<'a>(line: &'a str, prefixes: &[&str]) -> Option<&'a str> {
    prefixes.iter().find_map(|prefix| {
        line.strip_prefix(prefix)
            .map(str::trim)
            .filter(|value| !value.is_empty())
    })
}

fn normalized_anchor_ids(ids: &[u64]) -> Vec<u64> {
    let mut sorted = ids.to_vec();
    sorted.sort_unstable();
    sorted
}

fn tokenize_like_engine(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}
