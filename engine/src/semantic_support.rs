use super::*;

pub(crate) fn truncate(s: &str, max: usize) -> String {
    let char_count = s.chars().count();
    if char_count <= max {
        s.to_string()
    } else {
        format!("{}...", s.chars().take(max).collect::<String>())
    }
}

pub(crate) fn procedure_activation_confidence(
    memory: &MemoryRecord,
    schema: &ProcedureSchema,
    query: &str,
) -> f32 {
    let normalized_query = normalize_procedure_phrase(query);
    let content_score = if procedure_phrase_matches(&normalized_query, &memory.content) {
        0.35
    } else {
        0.0
    };
    let trigger_score = schema
        .trigger_conditions
        .iter()
        .filter(|condition| procedure_phrase_matches(&normalized_query, condition))
        .count() as f32
        * 0.45;
    let key_score = memory
        .fact_value
        .as_deref()
        .filter(|value| procedure_phrase_matches(&normalized_query, value))
        .map(|_| 0.20)
        .unwrap_or(0.0);

    (content_score + trigger_score + key_score).min(1.0)
}

pub(crate) fn procedure_phrase_matches(query: &str, candidate: &str) -> bool {
    let query = normalize_procedure_phrase(query);
    let candidate = normalize_procedure_phrase(candidate);
    !query.is_empty()
        && !candidate.is_empty()
        && (query.contains(&candidate)
            || candidate.contains(&query)
            || query
                .split_whitespace()
                .all(|segment| candidate.contains(segment))
            || candidate
                .split_whitespace()
                .all(|segment| query.contains(segment)))
}

pub(crate) fn normalize_procedure_phrase(value: &str) -> String {
    value
        .trim()
        .replace("触发条件", "")
        .replace("成功判据", "")
        .replace("成功标准", "")
        .replace("失败回滚", "")
        .replace("步骤", "")
        .replace("当需要", "")
        .replace("如何", "")
        .replace("怎么", "")
        .replace("给我", "")
        .replace([':', '：', '，', '。', '\n', '\t'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn normalize_step_status(value: &str) -> String {
    match value.trim() {
        "completed" => "completed",
        "failed" => "failed",
        "in_progress" => "in_progress",
        _ => "pending",
    }
    .to_string()
}

pub(crate) fn render_procedure_content(schema: &ProcedureSchema) -> String {
    let mut lines = Vec::new();
    lines.extend(
        schema
            .trigger_conditions
            .iter()
            .map(|item| format!("触发条件: {}", item)),
    );
    lines.extend(schema.steps.iter().map(|item| format!("步骤: {}", item)));
    lines.extend(
        schema
            .rollback_steps
            .iter()
            .map(|item| format!("失败回滚: {}", item)),
    );
    lines.extend(
        schema
            .success_criteria
            .iter()
            .map(|item| format!("成功判据: {}", item)),
    );
    lines.join("\n")
}

pub(crate) fn procedure_health_bonus(evaluation: &ProcedureEvaluateResponse) -> f32 {
    evaluation.health_score * 0.25 - (evaluation.consecutive_failures as f32 * 0.03)
}

pub(crate) fn extract_claims(answer: &str) -> Vec<String> {
    answer
        .split(['。', '.', '!', '?'])
        .map(str::trim)
        .filter(|s| s.len() > 8)
        .filter(|s| {
            !s.starts_with("Trace:")
                && !s.starts_with("Trace：")
                && !s.starts_with("[")
                && !s.contains("/memory/trace/")
                && !s.contains("系统反馈：当前不能可靠回答")
                && !s.contains("没有可用的 active `novel_chapter_summary`")
                && !s.starts_with("不能用写作风格")
                && !s.starts_with("原因：系统只找到")
                && !s.starts_with("原因：系统没有找到")
                && !s.starts_with("当前可用记录：")
                && !s.starts_with("需要的系统动作：")
                && !s.contains("archive 原文块")
        })
        .map(|s| s.to_string())
        .collect()
}

pub(crate) fn build_activation_answer_notes(items: &[RetrievedItem]) -> Vec<SemanticAnswerNote> {
    let mut notes = Vec::new();
    let mut seen = HashSet::new();

    for item in items {
        if item.layer != MemoryLayer::Activation {
            continue;
        }
        let Some(metadata) = item.activation_metadata.as_ref() else {
            continue;
        };
        if metadata.status == "cleared" {
            continue;
        }
        let label = if metadata.status == "pinned" {
            "固定激活工作集"
        } else {
            "激活工作集"
        };
        let message = format!("[{label}] {}", truncate(&item.content, 96));
        if seen.insert(message.clone()) {
            notes.push(SemanticAnswerNote {
                note_type: "activation".to_string(),
                message,
                entity_id: item
                    .semantic_node
                    .as_ref()
                    .map(|node| node.entity_id.clone()),
                attribute_key: item
                    .semantic_node
                    .as_ref()
                    .map(|node| node.attribute_key.clone()),
                scope: None,
                candidate_values: Vec::new(),
            });
        }
    }

    notes
}

pub(crate) fn build_parametric_answer_notes(
    items: &[RetrievedItem],
    query: &str,
) -> Vec<SemanticAnswerNote> {
    let q = query.to_lowercase();
    let wants_parametric = [
        "参数",
        "阈值",
        "配置",
        "默认",
        "policy",
        "config",
        "threshold",
    ]
    .iter()
    .any(|pattern| q.contains(pattern));
    if !wants_parametric {
        return Vec::new();
    }

    let mut notes = Vec::new();
    let mut seen = HashSet::new();
    for item in items {
        if item.layer != MemoryLayer::Parametric {
            continue;
        }
        let message = format!("[参数记忆] {}", truncate(&item.content, 96));
        if seen.insert(message.clone()) {
            notes.push(SemanticAnswerNote {
                note_type: "parametric".to_string(),
                message,
                entity_id: item
                    .semantic_node
                    .as_ref()
                    .map(|node| node.entity_id.clone()),
                attribute_key: item
                    .semantic_node
                    .as_ref()
                    .map(|node| node.attribute_key.clone()),
                scope: None,
                candidate_values: Vec::new(),
            });
        }
    }
    notes
}

pub(crate) fn build_semantic_answer_notes(items: &[RetrievedItem]) -> Vec<SemanticAnswerNote> {
    let mut notes = Vec::new();
    let mut seen = HashSet::new();

    for item in items {
        let Some(node) = item.semantic_node.as_ref() else {
            continue;
        };

        let note = match node.status {
            SemanticStatus::Disputed => {
                let candidate_values = item
                    .semantic_conflict
                    .as_ref()
                    .map(|conflict| {
                        conflict
                            .members
                            .iter()
                            .map(|member| member.attribute_value.clone())
                            .collect::<Vec<_>>()
                            .join(" / ")
                    })
                    .filter(|variants| !variants.is_empty())
                    .unwrap_or_else(|| node.attribute_value.clone());
                let values = candidate_values
                    .split(" / ")
                    .map(|value| value.to_string())
                    .collect::<Vec<_>>();
                SemanticAnswerNote {
                    note_type: "conflict".to_string(),
                    message: format!(
                        "[语义冲突提示] {}.{} 存在冲突候选值：{}。",
                        node.entity_id, node.attribute_key, candidate_values
                    ),
                    entity_id: Some(node.entity_id.clone()),
                    attribute_key: Some(node.attribute_key.clone()),
                    scope: Some(semantic_scope_label(&node.scope).to_string()),
                    candidate_values: values,
                }
            }
            SemanticStatus::Active => match node.scope {
                semantic::SemanticScope::Project | semantic::SemanticScope::Task => {
                    let message = format!(
                            "[适用范围提示] 当前证据更偏{}级知识：{}.{} = {}。",
                            semantic_scope_label(&node.scope),
                            node.entity_id,
                            node.attribute_key,
                            node.attribute_value
                        );
                    SemanticAnswerNote {
                        note_type: "scope".to_string(),
                        message,
                        entity_id: Some(node.entity_id.clone()),
                        attribute_key: Some(node.attribute_key.clone()),
                        scope: Some(semantic_scope_label(&node.scope).to_string()),
                        candidate_values: vec![node.attribute_value.clone()],
                    }
                }
                _ => continue,
            },
            _ => continue,
        };

        if seen.insert(note.message.clone()) {
            notes.push(note);
        }
    }

    notes
}

pub(crate) fn build_experience_answer_notes(
    items: &[RetrievedItem],
    query: &str,
) -> Vec<SemanticAnswerNote> {
    let query = query.to_lowercase();
    let wants_cause = query.contains("为什么") || query.contains("导致") || query.contains("原因");
    let wants_recovery =
        query.contains("恢复") || query.contains("补救") || query.contains("怎么修");
    let wants_similarity = query.contains("类似")
        || query.contains("相似")
        || query.contains("之前")
        || query.contains("过去")
        || query.contains("同样")
        || query.contains("失败模式");

    let mut notes = Vec::new();
    let mut seen = HashSet::new();

    for item in items {
        let Some(meta) = item.episodic_metadata.as_ref() else {
            continue;
        };

        if wants_cause && !meta.cause_links.is_empty() {
            let message = format!(
                "[失败原因提示] 相关经验显示：{}。",
                meta.cause_links.join("；")
            );
            if seen.insert(message.clone()) {
                notes.push(SemanticAnswerNote {
                    note_type: "failure_reason".to_string(),
                    message,
                    entity_id: None,
                    attribute_key: None,
                    scope: None,
                    candidate_values: meta.failure_type.clone().into_iter().collect::<Vec<_>>(),
                });
            }
        }

        if wants_recovery && !meta.recovery_links.is_empty() {
            let message = format!(
                "[恢复建议提示] 相关经验显示：{}。",
                meta.recovery_links.join("；")
            );
            if seen.insert(message.clone()) {
                notes.push(SemanticAnswerNote {
                    note_type: "recovery".to_string(),
                    message,
                    entity_id: None,
                    attribute_key: None,
                    scope: None,
                    candidate_values: meta.failure_type.clone().into_iter().collect::<Vec<_>>(),
                });
            }
        }

        if wants_similarity {
            let failure_label = meta
                .failure_type
                .clone()
                .or_else(|| meta.constraints.first().cloned())
                .unwrap_or_else(|| "未知问题".to_string());
            let result_label = meta
                .result
                .clone()
                .or_else(|| meta.outcome.clone())
                .unwrap_or_else(|| "结果未知".to_string());
            let message = format!(
                "[相似经验提示] 过去出现过相似问题：{}；当时结果：{}。",
                failure_label, result_label
            );
            if seen.insert(message.clone()) {
                notes.push(SemanticAnswerNote {
                    note_type: "similar_experience".to_string(),
                    message,
                    entity_id: None,
                    attribute_key: None,
                    scope: None,
                    candidate_values: vec![failure_label],
                });
            }
        }
    }

    notes
}
