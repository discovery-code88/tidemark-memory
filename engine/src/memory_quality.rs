use super::*;

#[derive(Debug, Clone)]
pub(crate) struct MemoryQualityTuning {
    pub(crate) stale_days: i64,
    pub(crate) importance_weight: f32,
    pub(crate) confidence_weight: f32,
    pub(crate) freshness_weight: f32,
    pub(crate) supersedes_bonus: f32,
    pub(crate) superseded_penalty: f32,
    pub(crate) boost_min: f32,
    pub(crate) boost_max: f32,
}

impl MemoryQualityTuning {
    pub(crate) fn from_env() -> Self {
        let boost_min = read_env_f32_range("MEMORY_QUALITY_BOOST_MIN", -0.20, -1.0, 1.0);
        let boost_max = read_env_f32_range("MEMORY_QUALITY_BOOST_MAX", 0.30, -1.0, 1.0);
        Self {
            stale_days: read_env_i64("MEMORY_QUALITY_STALE_DAYS", 30),
            importance_weight: read_env_f32_range(
                "MEMORY_QUALITY_IMPORTANCE_WEIGHT",
                0.08,
                0.0,
                1.0,
            ),
            confidence_weight: read_env_f32_range(
                "MEMORY_QUALITY_CONFIDENCE_WEIGHT",
                0.04,
                0.0,
                1.0,
            ),
            freshness_weight: read_env_f32_range("MEMORY_QUALITY_FRESHNESS_WEIGHT", 0.04, 0.0, 1.0),
            supersedes_bonus: read_env_f32_range("MEMORY_QUALITY_SUPERSEDES_BONUS", 0.03, 0.0, 1.0),
            superseded_penalty: read_env_f32_range(
                "MEMORY_QUALITY_SUPERSEDED_PENALTY",
                0.15,
                0.0,
                1.0,
            ),
            boost_min: boost_min.min(boost_max),
            boost_max: boost_max.max(boost_min),
        }
    }
}

pub(crate) static MEMORY_QUALITY_TUNING: OnceLock<MemoryQualityTuning> = OnceLock::new();

pub(crate) fn memory_quality_tuning() -> &'static MemoryQualityTuning {
    MEMORY_QUALITY_TUNING.get_or_init(MemoryQualityTuning::from_env)
}

pub(crate) fn collect_embedding_stats(
    memories: &[MemoryRecord],
) -> (
    HashMap<String, usize>,
    HashMap<String, usize>,
    HashMap<String, usize>,
) {
    let mut embedding_status_counts = HashMap::new();
    let mut embedding_provider_counts = HashMap::new();
    let mut embedding_model_counts = HashMap::new();

    for memory in memories {
        *embedding_status_counts
            .entry(
                memory
                    .embedding_status
                    .clone()
                    .unwrap_or_else(|| "missing".to_string()),
            )
            .or_insert(0) += 1;
        *embedding_provider_counts
            .entry(
                memory
                    .embedding_provider
                    .clone()
                    .unwrap_or_else(|| "missing".to_string()),
            )
            .or_insert(0) += 1;
        *embedding_model_counts
            .entry(
                memory
                    .embedding_model
                    .clone()
                    .unwrap_or_else(|| "missing".to_string()),
            )
            .or_insert(0) += 1;
    }

    (
        embedding_status_counts,
        embedding_provider_counts,
        embedding_model_counts,
    )
}

pub(crate) fn apply_memory_quality_metrics(
    snapshot: &mut MetricsResponse,
    memories: &[MemoryRecord],
) {
    let tuning = memory_quality_tuning();
    let stale_cutoff = Utc::now() - Duration::days(tuning.stale_days);
    for memory in memories {
        match memory_quality_type(memory) {
            "fact" => snapshot.memory_type_fact += 1,
            "decision" => snapshot.memory_type_decision += 1,
            "constraint" => snapshot.memory_type_constraint += 1,
            "todo" => snapshot.memory_type_todo += 1,
            "incident" => snapshot.memory_type_incident += 1,
            "preference" => snapshot.memory_type_preference += 1,
            "artifact" => snapshot.memory_type_artifact += 1,
            _ => {}
        }
        let confidence = memory
            .confidence
            .or_else(|| memory_quality_number(&memory.content, "confidence"))
            .unwrap_or(memory.trust);
        if confidence < 0.5 {
            snapshot.memory_quality_low_confidence += 1;
        }
        if memory
            .last_verified_at
            .map(|last_verified_at| last_verified_at < stale_cutoff)
            .unwrap_or(false)
        {
            snapshot.memory_quality_stale += 1;
        }
        if !memory.superseded_by.is_empty()
            || memory_quality_text_value(&memory.content, &["superseded_by"]).is_some()
        {
            snapshot.memory_quality_superseded += 1;
        }
    }
}

pub(crate) fn collect_layer_stats(memories: &[MemoryRecord]) -> HashMap<String, usize> {
    let mut counts = HashMap::new();

    for memory in memories {
        let key = match memory.layer {
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
        };
        *counts.entry(key.to_string()).or_insert(0) += 1;
    }

    counts
}

pub(crate) fn memory_layer_label(layer: &MemoryLayer) -> &'static str {
    match layer {
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

pub(crate) const MEMORY_QUALITY_TYPES: [&str; 51] = [
    "fact",
    "decision",
    "constraint",
    "todo",
    "incident",
    "preference",
    "artifact",
    "relationship",
    "profile",
    "habit",
    "conclusion",
    "taste_preference",
    "taste_anti_pattern",
    "taste_dial",
    "taste_theme",
    "taste_association",
    "taste_boundary",
    "taste_uncertainty",
    "taste_near_miss",
    "taste_rejected_association",
    "learning_problem_card",
    "learning_loop",
    "learning_concept_link",
    "learning_style_preference",
    "universal_profile",
    "novel_chapter_summary",
    "novel_style_profile",
    "long_document_memory",
    "long_document_character",
    "long_document_location",
    "long_document_timeline_event",
    "long_document_relationship",
    "long_document_world_rule",
    "long_document_open_question",
    "long_document_archive_chunk",
    "writing_style_profile",
    "design_style_profile",
    "operational_habit_profile",
    "native_privacy_receipt",
    "universal_source_text",
    "universal_source_document",
    "universal_source_image",
    "universal_source_audio",
    "universal_source_video",
    "universal_source_dom",
    "universal_source_code",
    "universal_source_3d",
    "universal_source_telemetry",
    "universal_source_map",
    "universal_source_trajectory",
    "universal_source_mixed",
];

pub(crate) fn memory_quality_type(memory: &MemoryRecord) -> &'static str {
    if let Some(value) = memory
        .memory_type
        .as_deref()
        .and_then(normalize_memory_type)
    {
        return value;
    }
    if let Some(value) = memory_quality_text_value(&memory.content, &["memory_type", "type"]) {
        if let Some(supported) = normalize_memory_type(&value) {
            return supported;
        }
    }
    let normalized = memory.content.to_lowercase();
    if memory.entity.is_some() && memory.fact_key.is_some() && memory.fact_value.is_some() {
        "fact"
    } else if matches!(memory.layer, MemoryLayer::Procedure) {
        "decision"
    } else if contains_any(
        &normalized,
        &[
            "constraint",
            "must",
            "do not",
            "don't",
            "compatibility",
            "禁止",
            "不要",
            "必须",
            "约束",
            "兼容",
        ],
    ) {
        "constraint"
    } else if contains_any(
        &normalized,
        &[
            "todo",
            "next",
            "open",
            "unfinished",
            "待办",
            "下一步",
            "未完成",
        ],
    ) {
        "todo"
    } else if contains_any(
        &normalized,
        &[
            "incident",
            "failure",
            "root cause",
            "rollback",
            "故障",
            "失败",
            "根因",
            "回滚",
        ],
    ) {
        "incident"
    } else if contains_any(
        &normalized,
        &[
            "taste_preference",
            "taste_anti_pattern",
            "taste_dial",
            "taste_theme",
            "taste_association",
            "taste_boundary",
            "taste_uncertainty",
            "taste_near_miss",
            "taste_rejected_association",
            "taste_node",
            "taste_node_format",
            "taste profile",
            "preference",
            "prefer",
            "likes",
            "品位",
            "偏好",
            "喜欢",
        ],
    ) {
        "preference"
    } else if contains_any(
        &normalized,
        &[
            "file:", "endpoint", "script", "doc", "docker", "文件", "脚本", "文档", "接口",
        ],
    ) {
        "artifact"
    } else {
        match memory.layer {
            MemoryLayer::Fact
            | MemoryLayer::Parametric
            | MemoryLayer::SelfModel
            | MemoryLayer::Autobiographical => "fact",
            MemoryLayer::Procedure => "decision",
            _ => "fact",
        }
    }
}

pub(crate) fn memory_quality_score_boost(memory: &MemoryRecord, query: &str) -> f32 {
    let tuning = memory_quality_tuning();
    let memory_type = memory_quality_type(memory);
    let normalized = query.to_lowercase();
    if contains_any(
        &normalized,
        &[
            "文档", "pdf", "markdown", "文件", "图片", "截图", "图像", "image", "录音", "音频",
            "通话", "视频", "录像", "录屏", "ocr",
        ],
    ) {
        return 0.0;
    }
    let project_context = contains_any(
        &normalized,
        &[
            "current goal",
            "constraint",
            "decision",
            "todo",
            "incident",
            "risk",
            "next step",
            "learning",
            "learn",
            "concept",
            "concept map",
            "ability boundary",
            "local concept",
            "project",
            "repo",
            "当前目标",
            "约束",
            "决策",
            "待办",
            "风险",
            "学习",
            "概念",
            "概念地图",
            "能力边界",
            "局部概念",
            "项目",
            "chapter",
            "section",
            "章节",
        ],
    );
    let asks_for_summary_or_section = memory_query_asks_for_summary_or_section(&normalized);
    let summary_record_boost =
        if asks_for_summary_or_section && memory_is_compact_summary(memory, memory_type) {
            0.42
        } else if asks_for_summary_or_section && memory.layer == MemoryLayer::Archive {
            -0.12
        } else {
            0.0
        };
    let type_boost = match memory_type {
        "constraint" if contains_any(&normalized, &["constraint", "rule", "约束", "规则"]) => {
            0.14
        }
        "decision" if contains_any(&normalized, &["decision", "decide", "决策", "决定"]) => {
            0.12
        }
        "todo" if contains_any(&normalized, &["todo", "next", "open", "待办", "下一步"]) => {
            0.12
        }
        "incident"
            if contains_any(
                &normalized,
                &["incident", "failure", "risk", "故障", "失败"],
            ) =>
        {
            0.12
        }
        "preference"
        | "taste_preference"
        | "taste_anti_pattern"
        | "taste_dial"
        | "taste_theme"
        | "taste_association"
        | "taste_boundary"
        | "taste_uncertainty"
        | "taste_near_miss"
        | "taste_rejected_association"
            if contains_any(
                &normalized,
                &[
                    "preference",
                    "prefer",
                    "taste",
                    "style",
                    "喜欢",
                    "偏好",
                    "品位",
                ],
            ) =>
        {
            0.14
        }
        "artifact" if contains_any(&normalized, &["file", "endpoint", "script", "doc", "文件"]) => {
            0.10
        }
        "learning_problem_card"
        | "learning_loop"
        | "learning_concept_link"
        | "learning_style_preference"
            if contains_any(
                &normalized,
                &[
                    "learning",
                    "learn",
                    "concept",
                    "concept map",
                    "progressive disclosure",
                    "problem",
                    "ability boundary",
                    "stuck",
                    "reflection",
                    "review",
                    "学习",
                    "概念",
                    "概念地图",
                    "渐进披露",
                    "问题",
                    "能力边界",
                    "卡住",
                    "复盘",
                    "局部概念",
                ],
            ) =>
        {
            0.24
        }
        "novel_chapter_summary" if asks_for_summary_or_section => 0.18,
        "universal_profile"
            if contains_any(
                &normalized,
                &[
                    "universal profile",
                    "profile",
                    "pattern",
                    "style",
                    "habit",
                    "behavior",
                    "preference",
                    "preference tuning",
                    "fine-tuning",
                    "fine tuning",
                    "match user preferences",
                    "user preference adaptation",
                    "taste",
                    "workflow",
                    "modality",
                    "evidence",
                    "boundless",
                    "general",
                    "通用画像",
                    "画像",
                    "模式",
                    "风格",
                    "习惯",
                    "偏好",
                    "偏好调优",
                    "偏好微调",
                    "匹配用户偏好",
                    "不限",
                    "万能",
                ],
            ) =>
        {
            0.24
        }
        "novel_style_profile" | "writing_style_profile"
            if contains_any(
                &normalized,
                &[
                    "style", "voice", "diction", "rhythm", "tone", "imagery", "dialogue", "pacing",
                    "writing", "风格", "文风", "语气", "节奏", "意象", "对白",
                ],
            ) =>
        {
            0.24
        }
        "design_style_profile"
            if contains_any(
                &normalized,
                &[
                    "design style",
                    "web design",
                    "ui design",
                    "visual style",
                    "layout",
                    "typography",
                    "color",
                    "palette",
                    "spacing",
                    "component",
                    "responsive",
                    "accessibility",
                    "网页设计",
                    "界面设计",
                    "视觉风格",
                    "布局",
                    "配色",
                    "排版",
                    "组件",
                ],
            ) =>
        {
            0.24
        }
        "operational_habit_profile"
            if contains_any(
                &normalized,
                &[
                    "operational habit",
                    "operator habit",
                    "workflow habit",
                    "physical ai",
                    "operator briefing",
                    "mission briefing",
                    "approval style",
                    "risk posture",
                    "操作习惯",
                    "运行习惯",
                    "物理",
                    "操作员",
                    "审批",
                ],
            ) =>
        {
            0.24
        }
        "fact" if contains_any(&normalized, &["fact", "what", "事实", "什么"]) => 0.08,
        "constraint" | "decision" if project_context => 0.10,
        "todo" | "incident" if project_context => 0.09,
        "artifact" if project_context => 0.06,
        "fact"
        | "preference"
        | "taste_preference"
        | "taste_anti_pattern"
        | "taste_dial"
        | "taste_theme"
        | "taste_association"
        | "taste_boundary"
        | "taste_uncertainty"
        | "taste_near_miss"
        | "taste_rejected_association"
        | "learning_problem_card"
        | "learning_loop"
        | "learning_concept_link"
        | "learning_style_preference"
        | "universal_profile"
        | "novel_chapter_summary"
        | "novel_style_profile"
        | "writing_style_profile"
        | "design_style_profile"
        | "operational_habit_profile"
            if project_context =>
        {
            0.08
        }
        _ => 0.0,
    };
    let importance = memory
        .importance
        .or_else(|| memory_quality_number(&memory.content, "importance"))
        .unwrap_or(0.0);
    let confidence = memory
        .confidence
        .or_else(|| memory_quality_number(&memory.content, "confidence"))
        .unwrap_or(memory.trust)
        .clamp(0.0, 1.0);
    let freshness = memory
        .freshness
        .or_else(|| memory_quality_number(&memory.content, "freshness"))
        .unwrap_or(0.0);
    let superseded_penalty = if !memory.superseded_by.is_empty()
        || memory_quality_text_value(&memory.content, &["superseded_by"]).is_some()
    {
        tuning.superseded_penalty
    } else {
        0.0
    };
    let supersedes_bonus = if !memory.supersedes.is_empty()
        || memory_quality_text_value(&memory.content, &["supersedes"]).is_some()
    {
        tuning.supersedes_bonus
    } else {
        0.0
    };
    (summary_record_boost
        + type_boost
        + importance.clamp(0.0, 1.0) * tuning.importance_weight
        + confidence * tuning.confidence_weight
        + freshness.clamp(0.0, 1.0) * tuning.freshness_weight
        + supersedes_bonus
        - superseded_penalty)
        .clamp(tuning.boost_min, tuning.boost_max)
}

fn memory_query_asks_for_summary_or_section(normalized_query: &str) -> bool {
    contains_any(
        normalized_query,
        &[
            "chapter",
            "section",
            "summary",
            "summarize",
            "key events",
            "what happened",
            "what happens",
            "recap",
            "overview",
            "brief",
            "发生了什么",
            "怎么回事",
            "第",
            "章",
            "节",
            "章节",
            "小节",
            "概要",
            "总结",
            "概括",
            "关键事件",
            "回顾",
            "简报",
        ],
    )
}

fn memory_is_compact_summary(memory: &MemoryRecord, memory_type: &str) -> bool {
    memory.layer == MemoryLayer::SessionSummary
        || memory_type.ends_with("_summary")
        || memory_type.contains("profile")
        || memory.content.contains("Chapter summary:")
        || memory.content.contains("Summary:")
        || memory.content.contains("Key events:")
}

pub(crate) fn normalize_memory_type(value: &str) -> Option<&'static str> {
    let normalized = value.trim().to_lowercase();
    MEMORY_QUALITY_TYPES
        .iter()
        .copied()
        .find(|supported| *supported == normalized)
}

pub(crate) fn clean_memory_type(value: Option<&String>) -> Option<String> {
    value.and_then(|value| normalize_memory_type(value).map(str::to_string))
}

pub(crate) fn clean_quality_score(value: Option<f32>) -> Option<f32> {
    value.map(|value| value.clamp(0.0, 1.0))
}

pub(crate) fn clean_quality_refs(values: &[String]) -> Vec<String> {
    values
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect()
}

pub(crate) fn memory_quality_number(content: &str, key: &str) -> Option<f32> {
    memory_quality_text_value(content, &[key]).and_then(|value| value.parse::<f32>().ok())
}

pub(crate) fn memory_quality_text_value(content: &str, keys: &[&str]) -> Option<String> {
    content.lines().take(16).find_map(|line| {
        let trimmed = line
            .trim()
            .trim_start_matches('-')
            .trim_start_matches('*')
            .trim();
        let (key, value) = trimmed.split_once(':')?;
        let normalized_key = key.trim().to_lowercase();
        if !keys.iter().any(|candidate| normalized_key == *candidate) {
            return None;
        }
        let value = value
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .trim_matches('`')
            .to_lowercase();
        if value.is_empty() {
            None
        } else {
            Some(value)
        }
    })
}

pub(crate) fn memory_quality_datetime_value(content: &str, key: &str) -> Option<DateTime<Utc>> {
    memory_quality_text_value(content, &[key]).and_then(|value| {
        let normalized = if value.ends_with('z') {
            format!("{}Z", value.trim_end_matches('z'))
        } else {
            value
        };
        DateTime::parse_from_rfc3339(&normalized)
            .ok()
            .map(|parsed| parsed.with_timezone(&Utc))
    })
}

pub(crate) fn is_legacy_quality_metadata_line(line: &str) -> bool {
    let trimmed = line
        .trim()
        .trim_start_matches('-')
        .trim_start_matches('*')
        .trim();
    let Some((key, _)) = trimmed.split_once(':') else {
        return false;
    };
    matches!(
        key.trim().to_ascii_lowercase().as_str(),
        "confidence"
            | "freshness"
            | "importance"
            | "last_verified_at"
            | "supersedes"
            | "superseded_by"
    )
}

pub(crate) fn structured_json_payload(content: &str) -> Option<&str> {
    let trimmed = content.trim();
    if serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
        return Some(trimmed);
    }

    let mut offset = 0;
    let mut saw_quality_metadata = false;
    for segment in content.split_inclusive('\n') {
        let line = segment.trim();
        if line.starts_with('{') || line.starts_with('[') {
            let candidate = content.get(offset..)?.trim();
            return (saw_quality_metadata
                && serde_json::from_str::<serde_json::Value>(candidate).is_ok())
            .then_some(candidate);
        }
        if line.is_empty() {
            offset += segment.len();
            continue;
        }
        if !is_legacy_quality_metadata_line(line) {
            return None;
        }
        saw_quality_metadata = true;
        offset += segment.len();
    }
    None
}

pub(crate) fn parse_structured_memory_content<T>(content: &str) -> Option<T>
where
    T: serde::de::DeserializeOwned,
{
    serde_json::from_str(structured_json_payload(content)?).ok()
}

pub(crate) fn set_quality_line(content: &str, key: &str, value: &str) -> String {
    // Quality state for structured records lives in typed columns. Preserve
    // their native payload and remove any legacy text prefix on first touch.
    if let Some(payload) = structured_json_payload(content) {
        return payload.to_string();
    }

    let mut replaced = false;
    let mut lines = content
        .lines()
        .map(|line| {
            let trimmed = line
                .trim()
                .trim_start_matches('-')
                .trim_start_matches('*')
                .trim();
            let matches_key = trimmed
                .split_once(':')
                .map(|(existing_key, _)| existing_key.trim().eq_ignore_ascii_case(key))
                .unwrap_or(false);
            if matches_key {
                replaced = true;
                format!("{key}: {value}")
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>();
    if !replaced {
        lines.insert(0, format!("{key}: {value}"));
    }
    lines.join("\n")
}




#[cfg(test)]
mod structured_memory_content_tests {
    use super::*;

    #[test]
    fn quality_updates_should_preserve_native_json_payloads() {
        let payload = r#"{"receipt_id":"turn_receipt_1","valid":true}"#;
        assert_eq!(
            set_quality_line(payload, "superseded_by", "memory_2"),
            payload
        );
    }

    #[test]
    fn structured_parser_should_recover_and_clean_legacy_quality_prefixes() {
        let polluted =
            "superseded_by: \nfreshness: 1.0\n{\"receipt_id\":\"turn_receipt_1\",\"valid\":true}";
        let parsed =
            parse_structured_memory_content::<serde_json::Value>(polluted).expect("legacy payload");
        assert_eq!(parsed["receipt_id"], "turn_receipt_1");
        assert_eq!(
            set_quality_line(polluted, "last_verified_at", "2026-07-30T00:00:00Z"),
            r#"{"receipt_id":"turn_receipt_1","valid":true}"#
        );
    }

    #[test]
    fn structured_parser_should_not_extract_json_from_arbitrary_text() {
        assert!(parse_structured_memory_content::<serde_json::Value>(
            "ordinary narrative\n{\"not\":\"a structured memory\"}"
        )
        .is_none());
    }
}
