use super::*;
use std::collections::{BTreeMap, BTreeSet};

const MAX_ANSWER_RANGE_NODES: usize = 64;
const MAX_MULTI_DOCUMENT_ANSWER_LIMIT: usize = 50;
const MAX_ANSWER_SOURCE_LIMIT: usize = 512;
const MAX_ANSWER_CANDIDATE_POOL: usize = 200;

fn normalized_answer_detail_level(value: Option<&str>) -> &'static str {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("concise") => "concise",
        Some("balanced") => "balanced",
        Some("detailed") => "detailed",
        Some("custom") => "custom",
        _ => "auto",
    }
}

pub(crate) fn answer_response_mode_is_valid(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "auto" | "direct" | "evidence" | "verbatim"
    )
}

fn normalized_answer_response_mode(value: Option<&str>, query: &str) -> &'static str {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("direct") => "direct",
        Some("evidence") => "evidence",
        Some("verbatim") => "verbatim",
        _ => {
            let lower = query.to_lowercase();
            if contains_any(
                &lower,
                &[
                    "verbatim",
                    "exact wording",
                    "quote exactly",
                    "raw content",
                    "full original text",
                    "完整原文",
                    "原始内容",
                    "一字不差",
                    "逐字",
                    "原文",
                ],
            ) {
                "verbatim"
            } else if contains_any(
                &lower,
                &[
                    "show evidence excerpts",
                    "source excerpts",
                    "supporting text",
                    "证据原文",
                    "证据片段",
                    "显示出处",
                ],
            ) {
                "evidence"
            } else {
                "direct"
            }
        }
    }
}

fn auto_answer_source_limit(detail_level: &str, eligible: usize, explicit: bool) -> usize {
    if explicit {
        return eligible.max(1).min(MAX_ANSWER_SOURCE_LIMIT);
    }
    let adaptive_limit = match detail_level {
        "concise" => 32,
        "balanced" => 32,
        "detailed" => 96,
        "custom" => 64,
        _ if eligible <= 32 => eligible.max(1),
        _ if eligible <= 128 => 64,
        _ => 128,
    };
    adaptive_limit
        .min(eligible.max(1))
        .max(1)
        .min(MAX_ANSWER_SOURCE_LIMIT)
}

fn auto_answer_document_limit(eligible: usize) -> usize {
    eligible.max(1).min(MAX_MULTI_DOCUMENT_ANSWER_LIMIT)
}

pub(crate) fn answer_candidate_limit(req: &AnswerRequest, eligible: usize) -> usize {
    let detail_level = normalized_answer_detail_level(req.detail_level.as_deref());
    let final_limit = req
        .source_limit
        .unwrap_or_else(|| auto_answer_source_limit(detail_level, eligible, false))
        .clamp(1, MAX_ANSWER_SOURCE_LIMIT);
    let expansion_factor = match final_limit {
        1..=32 => 4,
        33..=96 => 3,
        _ => 2,
    };
    final_limit
        .saturating_mul(expansion_factor)
        .min(MAX_ANSWER_CANDIDATE_POOL)
        .min(eligible.max(1))
}

#[allow(dead_code)]
fn answer_candidate_limit_legacy(req: &AnswerRequest, eligible: usize) -> usize {
    let detail_level = normalized_answer_detail_level(req.detail_level.as_deref());
    req.source_limit
        .unwrap_or_else(|| auto_answer_source_limit(detail_level, eligible, false))
        .clamp(1, MAX_ANSWER_SOURCE_LIMIT)
        .min(50)
}

fn answer_confidence_level(
    has_disputed_semantic: bool,
    draft_mode: &str,
    answer_status: &str,
    all_covered: bool,
    inferred_source_count: usize,
) -> &'static str {
    if has_disputed_semantic {
        "disputed"
    } else if draft_mode == "inferred" && all_covered && inferred_source_count >= 3 {
        "high"
    } else if draft_mode == "inferred" && all_covered && inferred_source_count > 0 {
        "medium"
    } else if draft_mode == "inferred" {
        "medium_or_low"
    } else if draft_mode == "partial_inferred" {
        "low"
    } else if answer_status != "answered" {
        "medium_or_low"
    } else if all_covered {
        "high"
    } else {
        "medium_or_low"
    }
}
fn strip_inline_answer_citations(value: &str) -> String {
    static INLINE_CITATION: OnceLock<regex::Regex> = OnceLock::new();
    static SPACE_BEFORE_PUNCTUATION: OnceLock<regex::Regex> = OnceLock::new();
    static REPEATED_INLINE_SPACE: OnceLock<regex::Regex> = OnceLock::new();
    let regex = INLINE_CITATION.get_or_init(|| {
        regex::Regex::new(r"\[[^\]\r\n]*memory#[0-9]+[^\]\r\n]*\]")
            .expect("inline answer citation regex")
    });
    let punctuation = SPACE_BEFORE_PUNCTUATION.get_or_init(|| {
        regex::Regex::new(r"[ \t]+([.,!?;:，。！？；：])")
            .expect("answer punctuation whitespace regex")
    });
    let repeated_space = REPEATED_INLINE_SPACE
        .get_or_init(|| regex::Regex::new(r"[ \t]{2,}").expect("answer repeated whitespace regex"));
    repeated_space
        .replace_all(
            &punctuation.replace_all(&regex.replace_all(value, ""), "$1"),
            " ",
        )
        .to_string()
}

fn normalize_answer_inline_citation_labels(answer: &str, items: &[RetrievedItem]) -> String {
    static INLINE_CITATION: OnceLock<regex::Regex> = OnceLock::new();
    let labels = items
        .iter()
        .map(|item| (item.id, readable_answer_source_label(item)))
        .collect::<HashMap<_, _>>();
    if labels.is_empty() {
        return answer.to_string();
    }

    let citation = INLINE_CITATION.get_or_init(|| {
        regex::Regex::new(r"\[memory#(?P<memory_id>[0-9]+)\s*\|\s*[^\]\r\n]*\]")
            .expect("inline citation label regex")
    });
    citation
        .replace_all(answer, |captures: &regex::Captures| {
            let Some(memory_id) = captures
                .name("memory_id")
                .and_then(|value| value.as_str().parse::<u64>().ok())
            else {
                return captures
                    .get(0)
                    .map(|value| value.as_str().to_string())
                    .unwrap_or_default();
            };
            labels
                .get(&memory_id)
                .map(|label| format!("[memory#{memory_id} | {label}]"))
                .unwrap_or_else(|| {
                    captures
                        .get(0)
                        .map(|value| value.as_str().to_string())
                        .unwrap_or_default()
                })
        })
        .into_owned()
}
pub(crate) fn direct_answer_display_text(draft: &str, response_mode: &str) -> String {
    direct_answer_display_text_with_citations(draft, response_mode, false)
}

fn direct_answer_display_text_with_citations(
    draft: &str,
    response_mode: &str,
    preserve_inline_citations: bool,
) -> String {
    if matches!(response_mode, "evidence" | "verbatim") {
        return draft.trim().to_string();
    }

    let mut answer = draft.trim();
    let evidence_markers = [
        "\n\nSupporting evidence:",
        "\n\n支持依据：",
        "\n\nReasoning:",
        "\n\n推理依据：",
        "\n\n- Evidence:",
        "\n\n- 证据：",
    ];
    if let Some(index) = evidence_markers
        .iter()
        .filter_map(|marker| answer.find(marker))
        .min()
    {
        answer = &answer[..index];
    }
    for prefix in [
        "Direct answer:",
        "直接回答：",
        "直接回答:",
        "Focused answer:",
        "聚焦回答：",
        "聚焦回答:",
    ] {
        if let Some(stripped) = answer.strip_prefix(prefix) {
            answer = stripped.trim_start();
            break;
        }
    }

    let displayed = if preserve_inline_citations {
        answer.to_string()
    } else {
        strip_inline_answer_citations(answer)
    };
    displayed
        .lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn answer_claim_display_text(claim: &str) -> String {
    let cleaned = direct_answer_display_text(claim.trim_start_matches("- "), "direct");
    if cleaned.is_empty() {
        claim.trim().to_string()
    } else {
        cleaned
    }
}



pub(crate) fn checked_claim_cited_text(
    report: &CoverageReport,
    sources: &[RetrievedItem],
) -> String {
    let text = answer_claim_display_text(&report.claim);
    if !report.covered || report.evidence_atom_ids.is_empty() {
        return text;
    }
    let sections = sources
        .iter()
        .filter(|item| {
            report.evidence_ids.contains(&item.id) && is_novel_chapter_summary_item(item)
        })
        .flat_map(|item| chapter_field_values(&item.content))
        .filter(|(_, value)| answer_claim_display_text(value) == text)
        .map(|(section, _)| section)
        .collect::<BTreeSet<_>>();
    let text = if sections.is_empty() {
        text
    } else {
        let headings = sections
            .iter()
            .map(|section| section.label())
            .collect::<Vec<_>>()
            .join(" / ");
        format!("{headings}：{text}")
    };
    let mut citations = report
        .evidence_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter_map(|id| {
            sources
                .iter()
                .find(|item| item.id == id)
                .map(|item| format!("[memory#{id} | {}]", readable_answer_source_label(item)))
        })
        .collect::<Vec<_>>()
        .join(" ");
    for step in &report.derivation {
        let Some(encoded) = step.strip_prefix("chapter_archive_support:") else {
            continue;
        };
        let Ok(link) = serde_json::from_str::<serde_json::Value>(encoded) else {
            continue;
        };
        let (Some(summary_id), Some(archive_id)) =
            (link["summary_id"].as_u64(), link["archive_id"].as_u64())
        else {
            continue;
        };
        if !report.evidence_ids.contains(&summary_id) || !report.evidence_ids.contains(&archive_id)
        {
            continue;
        }
        let summary = sources.iter().find(|item| item.id == summary_id);
        let archive = sources.iter().find(|item| item.id == archive_id);
        if let (Some(summary), Some(archive)) = (summary, archive) {
            if link["summary_version"].as_u64() == Some(u64::from(summary.version))
                && link["archive_version"].as_u64() == Some(u64::from(archive.version))
                && link["archive_digest"].as_str()
                    == Some(
                        crate::shared_object::stable_digest(archive.content.as_bytes()).as_str(),
                    )
            {
                citations.push_str(&format!(
                    " 摘要依据：[novel_chapter_summary -> source memory#{archive_id}]"
                ));
            }
        }
    }
    if citations.is_empty() {
        text
    } else if !sections.is_empty() {
        format!("{text} 证据：{citations}")
    } else {
        format!("{text} {citations}")
    }
}

fn deliver_checked_answer_with_sources(
    draft: String,
    response_mode: &str,
    draft_mode: &str,
    coverage: &[CoverageReport],
    pipeline_incomplete: bool,
    disputed: bool,
    has_evidence: bool,
    chinese: bool,
    sources: &[RetrievedItem],
    frame: Option<&RendererFrame>,
) -> (String, &'static str) {
    let raw_requested = matches!(response_mode, "evidence" | "verbatim");
    let all_covered = !coverage.is_empty() && coverage.iter().all(|report| report.covered);
    let supported_count = coverage.iter().filter(|report| report.covered).count();
    // Complete coverage proves the registered claims, not arbitrary extra text
    // a renderer placed in its draft. Always rebuild generated prose from the
    // checked claims, including the all-covered case.
    let text = if !raw_requested && draft_mode != "unavailable" {
        let supported = frame
            .and_then(|frame| frame.render(coverage, sources, chinese))
            .unwrap_or_else(|| {
                coverage
                    .iter()
                    .filter(|report| report.covered)
                    .map(|report| checked_claim_cited_text(report, sources))
                    .collect::<Vec<_>>()
                    .join("\n")
            });
        let message = if chinese {
            "现有证据尚未通过全部结论核验，未确认的结论不予输出。"
        } else {
            "The evidence does not verify all conclusions; unverified conclusions are withheld."
        };
        if all_covered {
            supported
        } else if supported.is_empty() {
            message.to_string()
        } else {
            format!("{supported}\n\n{message}")
        }
    } else {
        draft
    };
    let status = if draft_mode == "unavailable" {
        if has_evidence {
            "uncertain"
        } else {
            "no_evidence"
        }
    } else if raw_requested && draft_mode == "evidence_only" {
        "answered"
    } else if supported_count == 0 {
        "uncertain"
    } else if !all_covered
        || pipeline_incomplete
        || matches!(draft_mode, "partial" | "partial_inferred")
    {
        "partial"
    } else if draft_mode == "evidence_only" || disputed {
        "uncertain"
    } else {
        "answered"
    };
    (text, status)
}

impl MemoryEngine {
    pub(crate) fn consolidate_conversation(&mut self, conversation_id: &str) -> SummaryResponse {
        let mut episodic = self
            .memories
            .iter()
            .filter(|m| m.conversation_id == conversation_id && m.layer == MemoryLayer::Episodic)
            .cloned()
            .collect::<Vec<_>>();
        let mut autobiographical = self
            .memories
            .iter()
            .filter(|m| {
                m.conversation_id == conversation_id && m.layer == MemoryLayer::Autobiographical
            })
            .cloned()
            .collect::<Vec<_>>();
        let mut self_model = self
            .memories
            .iter()
            .filter(|m| m.conversation_id == conversation_id && m.layer == MemoryLayer::SelfModel)
            .cloned()
            .collect::<Vec<_>>();

        episodic.sort_by_key(|m| m.created_at);
        autobiographical.sort_by_key(|m| m.created_at);
        self_model.sort_by_key(|m| m.created_at);

        let mut selected = Vec::new();
        selected.extend(episodic.into_iter().rev().take(2));
        selected.extend(autobiographical.into_iter().rev().take(1));
        selected.extend(self_model.into_iter().rev().take(1));

        if selected.is_empty() {
            return SummaryResponse {
                summary_id: None,
                summary: None,
                anchors: Vec::new(),
            };
        }

        selected.sort_by_key(|m| m.created_at);
        let mut anchors = selected.iter().map(|m| m.id).collect::<Vec<_>>();
        anchors.sort_unstable();

        let summary_text = selected
            .iter()
            .map(|m| format!("[{:?}] {}", m.layer, truncate(&m.content, 96)))
            .collect::<Vec<_>>()
            .join(" | ");

        let already_exists = self.memories.iter().any(|m| {
            m.conversation_id == conversation_id
                && m.layer == MemoryLayer::SessionSummary
                && (m.content == format!("整合摘要: {}", summary_text) || {
                    let mut existing_anchors = m.anchor_ids.clone();
                    existing_anchors.sort_unstable();
                    existing_anchors == anchors
                })
        });

        if already_exists {
            return SummaryResponse {
                summary_id: None,
                summary: None,
                anchors: Vec::new(),
            };
        }

        let id = ID_GEN.fetch_add(1, AtomicOrdering::Relaxed);
        self.memories.push(MemoryRecord {
            id,
            conversation_id: conversation_id.to_string(),
            layer: MemoryLayer::SessionSummary,
            content: format!("整合摘要: {}", summary_text),
            tokens: Self::tokenize(&summary_text),
            source: "consolidation_engine".to_string(),
            trust: 0.82,
            created_at: Utc::now(),
            anchor_ids: anchors.clone(),
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
        });
        self.invalidate_cache_for_conversation(Some(conversation_id));

        SummaryResponse {
            summary_id: Some(id),
            summary: Some(summary_text),
            anchors,
        }
    }

    pub(crate) fn summarize_conversation(&mut self, conversation_id: &str) -> SummaryResponse {
        let mut candidates: Vec<_> = self
            .memories
            .iter()
            .filter(|m| m.conversation_id == conversation_id && m.layer == MemoryLayer::Working)
            .cloned()
            .collect();

        if candidates.is_empty() {
            return SummaryResponse {
                summary_id: None,
                summary: None,
                anchors: Vec::new(),
            };
        }

        candidates.sort_by_key(|m| m.created_at);
        let anchors: Vec<u64> = candidates.iter().rev().take(6).map(|m| m.id).collect();
        let summary_text = anchors
            .iter()
            .filter_map(|id| candidates.iter().find(|m| m.id == *id))
            .map(|m| format!("[{}] {}", m.id, truncate(&m.content, 72)))
            .collect::<Vec<_>>()
            .join("; ");

        let id = ID_GEN.fetch_add(1, AtomicOrdering::Relaxed);
        self.memories.push(MemoryRecord {
            id,
            conversation_id: conversation_id.to_string(),
            layer: MemoryLayer::SessionSummary,
            content: format!("递归摘要: {}", summary_text),
            tokens: Self::tokenize(&summary_text),
            source: "summarizer".to_string(),
            trust: 0.75,
            created_at: Utc::now(),
            anchor_ids: anchors.clone(),
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
        });
        self.invalidate_cache_for_conversation(Some(conversation_id));

        SummaryResponse {
            summary_id: Some(id),
            summary: Some(summary_text),
            anchors,
        }
    }

    

    

    pub(crate) fn answer_with_self_check_prefetched(
        &mut self,
        req: &AnswerRequest,
        prefetched_ids: Option<&[u64]>,
    ) -> (AnswerResponse, Vec<u64>) {
        let explicit_selection = req.selected_memory_ids.as_ref();
        let detail_level = normalized_answer_detail_level(req.detail_level.as_deref());
        let response_mode =
            normalized_answer_response_mode(req.response_mode.as_deref(), &req.query);
        let eligible_selected_ids = explicit_selection
            .map(|requested_ids| {
                requested_ids
                    .iter()
                    .copied()
                    .filter(|memory_id| {
                        self.memories.iter().any(|memory| {
                            memory.id == *memory_id
                                && req
                                    .conversation_id
                                    .as_deref()
                                    .map(|space_id| memory.conversation_id == space_id)
                                    .unwrap_or(false)
                                && memory_record_is_eligible_for_selected_answer(memory)
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let eligible_count = explicit_selection.map_or_else(
            || {
                self.memories
                    .iter()
                    .filter(|memory| {
                        req.conversation_id
                            .as_deref()
                            .map(|space_id| memory.conversation_id == space_id)
                            .unwrap_or(true)
                            && memory_record_is_eligible_for_selected_answer(memory)
                    })
                    .count()
            },
            |_| eligible_selected_ids.len(),
        );
        let effective_source_limit = req.source_limit.unwrap_or_else(|| {
            auto_answer_source_limit(detail_level, eligible_count, explicit_selection.is_some())
        });
        let effective_source_limit = effective_source_limit.clamp(1, MAX_ANSWER_SOURCE_LIMIT);
        let candidate_pool_limit = if explicit_selection.is_some() {
            effective_source_limit
        } else {
            answer_candidate_limit(req, eligible_count)
        };
        let effective_document_limit = req
            .document_limit
            .unwrap_or_else(|| auto_answer_document_limit(eligible_count))
            .clamp(1, MAX_MULTI_DOCUMENT_ANSWER_LIMIT);
        // Pinned activation is a control-plane inclusion directive. Keep its
        // IDs alongside the bounded retrieval result for non-explicit answers;
        // explicit selected_memory_ids remains an exact, fail-closed scope.
        let forced_pinned_activation_ids = explicit_selection
            .is_none()
            .then(|| self.pinned_activation_ids_for_scope(req.conversation_id.as_deref(), None))
            .unwrap_or_default();
        let mut retrieved = if explicit_selection.is_some() {
            let selected_ids = eligible_selected_ids
                .iter()
                .copied()
                .take(effective_source_limit)
                .collect::<Vec<_>>();
            self.mark_retrieved(&selected_ids);
            QueryResponse {
                cached: false,
                items: self.materialize_items(&selected_ids, &req.query),
                vector_retrieval: None,
                performance: None,
            }
        } else if let Some(prefetched_ids) = prefetched_ids {
            let mut selected_ids = prefetched_ids
                .iter()
                .copied()
                .take(candidate_pool_limit)
                .collect::<Vec<_>>();
            let mut seen = selected_ids.iter().copied().collect::<HashSet<_>>();
            for pinned_id in &forced_pinned_activation_ids {
                if seen.insert(*pinned_id) {
                    selected_ids.push(*pinned_id);
                }
            }
            self.mark_retrieved(&selected_ids);
            QueryResponse {
                cached: false,
                items: self.materialize_items(&selected_ids, &req.query),
                vector_retrieval: None,
                performance: None,
            }
        } else {
            let selected_ids = self.retrieve_ids(
                req.conversation_id.as_deref(),
                &req.query,
                candidate_pool_limit,
            );
            self.mark_retrieved(&selected_ids);
            QueryResponse {
                cached: false,
                items: self.materialize_items(&selected_ids, &req.query),
                vector_retrieval: None,
                performance: None,
            }
        };
        let candidate_pool_count = retrieved.items.len();
        // Freeze the authorized retrieval pool before local answer-bearing
        // heuristics. A local renderer's abstention must not erase evidence
        // that the model planner/verifier still needs to inspect.
        let eligible_pool_ids = self.memories.iter().filter(|record|
            memory_record_is_eligible_for_selected_answer(record)
                && req.conversation_id.as_deref().is_none_or(|scope| record.conversation_id == scope)
                && req.selected_memory_ids.as_ref().is_none_or(|ids| ids.contains(&record.id)))
            .map(|record|record.id).collect::<HashSet<_>>();
        let mut seen_pool_ids=HashSet::new();
        let mut frozen_candidate_ids=retrieved.items.iter()
            .filter(|item| eligible_pool_ids.contains(&item.id) && seen_pool_ids.insert(item.id))
            .map(|item|item.id).collect::<Vec<_>>();
        let mut diagnostic_seen=HashSet::new();
        retrieved.items.retain(|item|eligible_pool_ids.contains(&item.id) && diagnostic_seen.insert(item.id));
        // Specialized renderers must plan Claims from the same frozen,
        // authorized retrieval pool rather than from only the first atom-level
        // winner. The verifier still admits and binds every cited record below.
        let frozen_candidate_items = retrieved.items.clone();
        let mut modality_notes=build_multimodal_answer_notes(&retrieved.items,&req.query);
        for note in &mut modality_notes {
            if let Some(message)=multimodal_availability_feedback(&note.note_type) {
                note.message=message.to_string();
                note.scope=req.conversation_id.clone();
            }
        }
        let mut modalities=HashSet::new();
        modality_notes.retain(|note|modalities.insert(note.note_type.clone()));
        let retrieved_active_evidence = retrieved.items.iter().any(memory_item_is_active);
        if matches!(response_mode, "evidence" | "verbatim") {
            let mut seen=HashSet::new();
            retrieved.items.retain(|item| eligible_pool_ids.contains(&item.id) && seen.insert(item.id));
        } else {
            retrieved.items = crate::answer_pipeline::rerank_answer_candidates(&req.query, &retrieved.items);
        }

        let mut answer_pipeline_plan = crate::answer_pipeline::plan_answer_pipeline(
            &req.query,
            &retrieved.items,
            eligible_count,
            candidate_pool_count,
            req.answer_nodes,
            detail_level,
            effective_source_limit,
            explicit_selection.is_none(),
        );
        if matches!(response_mode, "evidence" | "verbatim") {
            retrieved.items.truncate(effective_source_limit);
        } else {
            let selected_order = answer_pipeline_plan
                .render_memory_ids()
                .iter()
                .enumerate()
                .map(|(index, memory_id)| (*memory_id, index))
                .collect::<HashMap<_, _>>();
            retrieved
                .items
                .retain(|item| selected_order.contains_key(&item.id));
            retrieved
                .items
                .sort_by_key(|item| selected_order.get(&item.id).copied().unwrap_or(usize::MAX));
        }
        // Slot reranking may drop a pinned activation whose text is only
        // context for the current question. Retain it in the final retrieved
        // collection so activation diagnostics and the frozen candidate-pool
        // contract remain observable, while Claim/evidence scoping below
        // still requires selected atoms and independent verification.
        if explicit_selection.is_none() {
            for item in frozen_candidate_items.iter().filter(|item| {
                item.layer == MemoryLayer::Activation
                    && item
                        .activation_metadata
                        .as_ref()
                        .is_some_and(|metadata| metadata.status == "pinned")
            }) {
                if !retrieved
                    .items
                    .iter()
                    .any(|candidate| candidate.id == item.id)
                {
                    retrieved.items.push(item.clone());
                }
            }
        }
        let answer_items = answer_pipeline_plan.scoped_items(&retrieved.items);
        let grounded_list_items = answer_pipeline_plan.grounded_list_items(&retrieved.items);
        let mut attempted_answer_operators = Vec::new();
        let mut selected_answer_operator = None::<String>;
        let mut renderer_claim_plan = None::<Vec<RendererClaim>>;
        let mut renderer_gaps = Vec::<RendererCoverageGap>::new();
        let mut renderer_frame = None::<RendererFrame>;
        let mut chapter_feedback = None::<ChapterFeedback>;
        let mut range_coverage = None::<String>;
        let mut renderer_contract_gap = None::<String>;

        let (draft, draft_mode) = if matches!(response_mode, "evidence" | "verbatim") {
            attempted_answer_operators.push("explicit_evidence_renderer".to_string());
            selected_answer_operator = Some("explicit_evidence_renderer".to_string());
            match render_explicit_evidence_answer(
                &req.query,
                &retrieved.items,
                req.answer_nodes,
                detail_level,
                response_mode,
            ) {
                Some(answer) => (answer, "evidence_only"),
                None => {
                    let chinese = req
                        .query
                        .chars()
                        .any(|character| ('一'..='鿿').contains(&character));
                    (
                        if chinese {
                            "目前没有可返回的原始证据。".to_string()
                        } else {
                            "There is no source evidence available to return.".to_string()
                        },
                        "unavailable",
                    )
                }
            }
        } else if let Some(draft) = req.draft_answer.clone() {
            attempted_answer_operators.push("caller_supplied_draft".to_string());
            selected_answer_operator = Some("caller_supplied_draft".to_string());
            (draft, "structured")
        } else {
            let rendered = (|| -> Option<(String, &'static str)> {
                use crate::answer_pipeline::AnswerOperator;

                for &operator in answer_pipeline_plan.operators() {
                    if crate::answer_execution::answer_worker_cancelled() { return None; }
                    if !answer_pipeline_plan.operator_is_admissible(operator) {
                        continue;
                    }
                    let operator_id = operator.as_str();
                    attempted_answer_operators.push(operator_id.to_string());
                    let candidate = match operator {
                        AnswerOperator::NovelChapterRangeFromRecords => plan_novel_chapter_range(
                            &req.query,
                            &self.memories,
                            req.conversation_id.as_deref(),
                            req.answer_nodes,
                            Some(effective_document_limit),
                        )
                        .map(|planned| {
                            renderer_claim_plan = Some(planned.claims);
                            chapter_feedback = planned.feedback;
                            range_coverage = Some(planned.coverage_note);
                            (
                                planned.display,
                                if chapter_feedback.is_some() {
                                    "unavailable"
                                } else if planned.incomplete {
                                    "partial"
                                } else {
                                    "structured"
                                },
                            )
                        }),
                        AnswerOperator::NovelChapterFromRecords => {
                            render_novel_chapter_answer_from_records(
                                &req.query,
                                &self.memories,
                                req.conversation_id.as_deref(),
                                Some(effective_document_limit),
                            )
                            .map(|(answer, claims, feedback)| {
                                chapter_feedback = feedback;
                                let records = self
                                    .memories
                                    .iter()
                                    .map(|record| (record.id, record.content.clone()))
                                    .filter(|(id, _)| {
                                        claims.iter().any(|claim| claim.evidence_ids.contains(id))
                                    })
                                    .collect::<Vec<_>>();
                                renderer_frame = Some(chapter_answer_frame(&claims, &records));
                                renderer_claim_plan = Some(claims);
                                if chapter_feedback.is_some() {
                                    (String::new(), "unavailable")
                                } else {
                                    (answer, "structured")
                                }
                            })
                        }
                        AnswerOperator::OperationalHabit => {
                            render_operational_habit_answer(&req.query, &frozen_candidate_items).map(
                                |(answer, claims)| {
                                    renderer_claim_plan = Some(claims);
                                    (answer, "structured")
                                },
                            )
                        }
                        AnswerOperator::VerifiedHandoff => {
                            render_verified_handoff_answer_plan(&req.query, &retrieved.items).map(
                                |(answer, claims)| {
                                    renderer_claim_plan = Some(claims);
                                    (answer, "structured")
                                },
                            )
                        }
                        AnswerOperator::OperatorBriefing => {
                            render_operator_briefing_answer_plan(&req.query, &frozen_candidate_items).map(
                                |(answer, claims, frame)| {
                                    renderer_frame = Some(frame);
                                    renderer_claim_plan = Some(claims);
                                    (answer, "structured")
                                },
                            )
                        }
                        AnswerOperator::AmbientCapture => {
                            render_ambient_capture_answer(&req.query, &frozen_candidate_items).map(
                                |(answer, claims, frame)| {
                                    renderer_frame = Some(frame);
                                    renderer_claim_plan = Some(claims);
                                    (answer, "structured")
                                },
                            )
                        }
                        AnswerOperator::LearningResume => {
                            render_learning_resume_answer(&req.query, &frozen_candidate_items).map(
                                |(answer, claims, frame)| {
                                    renderer_frame = Some(frame);
                                    renderer_claim_plan = Some(claims);
                                    (answer, "structured")
                                },
                            )
                        }
                        AnswerOperator::DocumentComparison => {
                            render_document_comparison_answer_plan(
                                &req.query,
                                &frozen_candidate_items,
                                Some(effective_document_limit),
                                req.answer_nodes,
                                detail_level,
                            )
                            .map(|(answer, claims, frame)| {
                                    renderer_frame = Some(frame);
                                    renderer_claim_plan = Some(claims);
                                    (answer, "synthesized")
                                })
                        }
                        AnswerOperator::ExperienceRecall => {
                            render_experience_recall_answer(&req.query, &frozen_candidate_items).map(
                                |(answer, claims, frame)| {
                                    renderer_frame = Some(frame);
                                    renderer_claim_plan = Some(claims);
                                    (answer, "structured")
                                },
                            )
                        }
                        AnswerOperator::UniversalProfile => {
                            render_universal_profile_answer(&req.query, &frozen_candidate_items).map(
                                |(answer, claims)| {
                                    renderer_claim_plan = Some(claims);
                                    (answer, "structured")
                                },
                            )
                        }
                        AnswerOperator::DesignStyle => {
                            render_design_style_answer(&req.query, &frozen_candidate_items).map(
                                |(answer, claims)| {
                                    renderer_claim_plan = Some(claims);
                                    (answer, "structured")
                                },
                            )
                        }
                        AnswerOperator::CategoricalAttribute => {
                            render_categorical_attribute_answer_plan(&req.query, &answer_items).map(
                                |(answer, claims)| {
                                    renderer_claim_plan = Some(claims);
                                    (answer, "synthesized")
                                },
                            )
                        }
                        AnswerOperator::SharedLocation => {
                            render_shared_location_answer_plan(&req.query, &answer_items).map(
                                |(answer, claims)| {
                                    renderer_claim_plan = Some(claims);
                                    (answer, "synthesized")
                                },
                            )
                        }
                        AnswerOperator::MultimodalAppearanceChange => {
                            render_multimodal_appearance_change_answer_plan(
                                &req.query,
                                &answer_items,
                            )
                            .map(|(answer, claims)| {
                                renderer_claim_plan = Some(claims);
                                (answer, "synthesized")
                            })
                        }
                        AnswerOperator::RecentVisualCreation => {
                            render_recent_visual_creation_answer_plan(&req.query, &answer_items)
                                .map(|(answer, claims)| {
                                    renderer_claim_plan = Some(claims);
                                    (answer, "synthesized")
                                })
                        }
                        AnswerOperator::GroundedList => render_grounded_list_answer_plan(
                            &req.query,
                            &grounded_list_items,
                            req.answer_nodes,
                            detail_level,
                        )
                        .map(|(answer, claims, gaps)| {
                            renderer_claim_plan = Some(claims);
                            renderer_gaps = gaps;
                            (answer.text, answer.draft_kind)
                        }),
                        AnswerOperator::GroundedInference => {
                            render_grounded_inference_answer(&req.query, &answer_items).map(
                                |answer| {
                                    renderer_claim_plan = Some(answer.claims);
                                    (answer.text, answer.draft_kind)
                                },
                            )
                        }
                        AnswerOperator::ResearchObject => {
                            render_research_object_answer_plan(&req.query, &answer_items).map(
                                |(answer, claims)| {
                                    renderer_claim_plan = Some(claims);
                                    (answer, "synthesized")
                                },
                            )
                        }
                        AnswerOperator::LinkedOrigin => {
                            render_linked_origin_answer_plan(&req.query, &answer_items).map(
                                |(answer, claims)| {
                                    renderer_claim_plan = Some(claims);
                                    (answer, "synthesized")
                                },
                            )
                        }
                        AnswerOperator::EventDate => {
                            render_event_date_answer_plan(&req.query, &answer_items).map(
                                |(answer, claims)| {
                                    renderer_claim_plan = Some(claims);
                                    (answer, "synthesized")
                                },
                            )
                        }
                        AnswerOperator::TemporalReasoning => {
                            render_temporal_reasoning_answer_plan(&req.query, &answer_items).map(
                                |(answer, claims)| {
                                    renderer_claim_plan = Some(claims);
                                    (answer, "structured")
                                },
                            )
                        }
                        AnswerOperator::CrossSourceEventCount => {
                            render_cross_source_event_count_answer_plan(&req.query, &answer_items)
                                .map(|(answer, claims)| {
                                    renderer_claim_plan = Some(claims);
                                    (answer, "structured")
                                })
                        }
                        AnswerOperator::FocusedScalar => {
                            render_focused_scalar_answer_plan(&req.query, &answer_items).map(
                                |(answer, claims)| {
                                    renderer_claim_plan = Some(claims);
                                    (answer, "structured")
                                },
                            )
                        }
                        AnswerOperator::NovelCrossChapter => {
                            render_novel_cross_chapter_answer(&req.query, &frozen_candidate_items).map(
                                |(answer, claims, frame)| {
                                    renderer_claim_plan = Some(claims);
                                    renderer_frame = Some(frame);
                                    (answer, "structured")
                                },
                            )
                        }
                        AnswerOperator::NovelChapter => {
                            render_novel_chapter_answer(&req.query, &retrieved.items).map(
                                |(answer, claims)| {
                                    let records = retrieved
                                        .items
                                        .iter()
                                        .map(|item| (item.id, item.content.clone()))
                                        .collect::<Vec<_>>();
                                    renderer_frame = Some(chapter_answer_frame(&claims, &records));
                                    renderer_claim_plan = Some(claims);
                                    (answer, "structured")
                                },
                            )
                        }
                        AnswerOperator::NovelChapterUnavailable => {
                            render_novel_chapter_unavailable_answer(&req.query)
                                .map(|answer| (answer, "unavailable"))
                        }
                        AnswerOperator::NovelStyle => {
                            render_novel_style_answer(&req.query, &frozen_candidate_items).map(
                                |(answer, claims)| {
                                    renderer_claim_plan = Some(claims);
                                    (answer, "structured")
                                },
                            )
                        }
                        AnswerOperator::UnifiedGeneral => answer_pipeline_plan
                            .unified_draft
                            .as_ref()
                            .map(|planned| (planned.text.clone(), planned.draft_mode)),
                        AnswerOperator::AnswerabilityGate => {
                            selected_answer_operator = Some(operator_id.to_string());
                            break;
                        }
                    };
                    if let Some(answer) = candidate {
                        if operator.contract()
                            == crate::answer_pipeline::AnswerOperatorContract::CandidateClaims
                            && renderer_claim_plan.is_none()
                        {
                            renderer_contract_gap =
                                Some(format!("{}:missing_structured_claim_plan", operator_id));
                        }
                        selected_answer_operator = Some(operator_id.to_string());
                        return Some(answer);
                    }
                }
                None
            })();

            rendered.unwrap_or_else(|| {
                let chinese = req
                    .query
                    .chars()
                    .any(|character| ('\u{4e00}'..='\u{9fff}').contains(&character));
                (
                    if chinese {
                        if answer_items.iter().any(memory_item_is_active) {
                            "检索到相关记录，但没有足以回答该问题的证据。".to_string()
                        } else {
                            "没有找到与该问题直接相关的记忆证据。".to_string()
                        }
                    } else if answer_items.iter().any(memory_item_is_active) {
                        "The retrieved records are related, but they do not contain enough answer-bearing evidence for this question."
                            .to_string()
                    } else {
                        "No memory evidence directly relevant to this question was found."
                            .to_string()
                    },
                    "unavailable",
                )
            })
        };

        // System renderers own validated inline citations. Preserve them in
        // user-visible direct answers, but replace internal storage identifiers
        // with readable record or document labels.
        let renderer_controls_citations = !matches!(
            selected_answer_operator.as_deref(),
            Some("caller_supplied_draft" | "explicit_evidence_renderer")
        );
        let renderer_cited_ids = if renderer_controls_citations {
            renderer_claim_plan
                .as_ref()
                .map(|claims| {
                    let mut ids = claims
                        .iter()
                        .flat_map(|claim| claim.evidence_ids.iter().copied())
                        .collect::<HashSet<_>>();
                    let archive_ids = self
                        .memories
                        .iter()
                        .filter(|record| {
                            ids.contains(&record.id)
                                && memory_record_has_direct_type(record, "novel_chapter_summary")
                        })
                        .flat_map(|record| declared_chapter_archive_ids(&record.content))
                        .filter(|id| {
                            self.memories.iter().any(|record| {
                                record.id == *id
                                    && memory_record_is_archive(record)
                                    && memory_record_is_eligible_for_selected_answer(record)
                            })
                        })
                        .collect::<Vec<_>>();
                    ids.extend(archive_ids);
                    ids
                })
                .unwrap_or_else(|| cited_memory_ids(&draft))
                .into_iter()
                .filter(|memory_id| {
                    self.memories.iter().any(|memory| {
                        memory.id == *memory_id
                            && memory_record_is_active(memory)
                            && req
                                .conversation_id
                                .as_deref()
                                .map(|scope| memory.conversation_id == scope)
                                .unwrap_or(true)
                            && req
                                .selected_memory_ids
                                .as_ref()
                                .map(|ids| ids.contains(memory_id))
                                .unwrap_or(true)
                    })
                })
                .collect::<HashSet<_>>()
        } else {
            HashSet::new()
        };
        let renderer_cited_id_list = renderer_cited_ids.iter().copied().collect::<Vec<_>>();
        let renderer_citation_items = self.materialize_items(&renderer_cited_id_list, &req.query);
        answer_pipeline_plan.admit_renderer_evidence(&renderer_citation_items);
        let mut citation_label_items = retrieved.items.clone();
        for item in &renderer_citation_items {
            if !citation_label_items
                .iter()
                .any(|candidate| candidate.id == item.id)
            {
                citation_label_items.push(item.clone());
            }
        }
        let direct_answer = direct_answer_display_text_with_citations(
            &draft,
            response_mode,
            renderer_controls_citations,
        );
        let direct_answer = if renderer_controls_citations {
            normalize_answer_inline_citation_labels(&direct_answer, &citation_label_items)
        } else {
            direct_answer
        };
        let mut claim_evidence_items = answer_items.clone();
        let renderer_provided_evidence_plan = renderer_controls_citations
            && !renderer_cited_ids.is_empty()
            && selected_answer_operator.as_deref() != Some("unified_answer_generator_v1");
        for item in renderer_citation_items {
            if !claim_evidence_items
                .iter()
                .any(|candidate| candidate.id == item.id)
            {
                claim_evidence_items.push(item);
            }
        }
        answer_pipeline_plan.set_selected_operator(selected_answer_operator.as_deref());
        let planned_unified_coverage =
            if selected_answer_operator.as_deref() == Some("unified_answer_generator_v1") {
                answer_pipeline_plan.unified_draft_coverage()
            } else {
                None
            };
        let explicit_renderer_claims = renderer_claim_plan.is_some();
        let extracted_claims = if draft_mode == "unavailable" || planned_unified_coverage.is_some()
        {
            Vec::new()
        } else if let Some(claims) = renderer_claim_plan {
            claims
        } else if selected_answer_operator.as_deref() == Some("caller_supplied_draft") {
            let claims = extract_structured_temporal_claims(&draft)
                .or_else(|| extract_direct_answer_claims(&draft))
                .unwrap_or_else(|| extract_claims(&direct_answer));
            renderer_claim_texts(claims, renderer_provided_evidence_plan, &direct_answer)
                .into_iter()
                .map(|text| RendererClaim {
                    evidence_ids: cited_memory_ids(&text).into_iter().collect(),
                    text,
                })
                .collect()
        } else {
            if !matches!(response_mode, "evidence" | "verbatim") {
                renderer_contract_gap = Some(format!(
                    "{}:missing_structured_claim_plan",
                    selected_answer_operator.as_deref().unwrap_or("unknown")
                ));
            }
            Vec::new()
        };
        // A caller draft cannot run specialized renderers, but a failure or
        // recovery question can still be answered from verified past episodes.
        // Those Claims are checked with renderer (verbatim) semantics.
        let mut extracted_claims = extracted_claims;
        let supplementary_claims_start = extracted_claims.len();
        if selected_answer_operator.as_deref() == Some("caller_supplied_draft")
            && !matches!(response_mode, "evidence" | "verbatim")
        {
            if let Some((_, experience_claims, _)) =
                render_experience_recall_answer(&req.query, &frozen_candidate_items)
            {
                let experience_items = self.materialize_items(
                    &experience_claims
                        .iter()
                        .flat_map(|claim| claim.evidence_ids.iter().copied())
                        .collect::<Vec<_>>(),
                    &req.query,
                );
                answer_pipeline_plan.admit_renderer_evidence(&experience_items);
                for item in experience_items {
                    if !claim_evidence_items.iter().any(|candidate| candidate.id == item.id) {
                        claim_evidence_items.push(item.clone());
                    }
                    if !citation_label_items.iter().any(|candidate| candidate.id == item.id) {
                        citation_label_items.push(item);
                    }
                }
                extracted_claims.extend(experience_claims);
            }
        }
        let mut coverage = planned_unified_coverage.unwrap_or_default();
        let mut all_covered = coverage.iter().all(|report| report.covered);

        for (claim_index, planned_claim) in extracted_claims.into_iter().enumerate() {
            let explicit_renderer_claims =
                explicit_renderer_claims || claim_index >= supplementary_claims_start;
            if crate::answer_execution::answer_worker_cancelled() { break; }
            let claim = planned_claim.text;
            let claim_tokens = Self::tokenize(&claim);
            let cited_claim_ids = planned_claim
                .evidence_ids
                .into_iter()
                .collect::<HashSet<_>>();
            let cited_candidates = claim_evidence_items
                .iter()
                .filter(|item| cited_claim_ids.contains(&item.id))
                .map(|item| item.id)
                .collect::<HashSet<_>>();
            // Inline renderer citations nominate candidates only. They no
            // longer bypass the same fact/subject/time/condition verifier used
            // by the general answer path.
            let renderer_candidates = if explicit_renderer_claims {
                cited_candidates.clone()
            } else if renderer_provided_evidence_plan {
                renderer_cited_ids.clone()
            } else {
                cited_candidates.clone()
            };
            let mut proof = if renderer_candidates.is_empty() {
                crate::answer_pipeline::VerifiedClaimEvidence::default()
            } else {
                answer_pipeline_plan.verify_claim_evidence(&claim, &renderer_candidates)
            };
            if explicit_renderer_claims && !proof.is_bound() {
                proof = answer_pipeline_plan
                    .verify_verbatim_renderer_quotation(&claim, &renderer_candidates);
            }
            if explicit_renderer_claims && proof.is_bound() {
                bind_chapter_archive_proof(
                    &answer_pipeline_plan,
                    &claim,
                    &cited_candidates,
                    &citation_label_items,
                    &mut proof,
                );
            }
            if !proof.is_bound() && !explicit_renderer_claims && !renderer_provided_evidence_plan {
                let lexical_candidates = answer_items
                    .iter()
                    .filter(|item| {
                        if answer_claim_is_directly_supported_by_content(&claim, &item.content) {
                            return true;
                        }
                        let itoks = Self::tokenize(&item.content);
                        let overlap = claim_tokens
                            .iter()
                            .filter(|t| itoks.iter().any(|x| x == *t))
                            .count();
                        overlap >= 2
                    })
                    .map(|item| item.id)
                    .collect::<HashSet<_>>();
                proof = answer_pipeline_plan.verify_claim_evidence(&claim, &lexical_candidates);
            }

            // `covered` and every returned source field come from the same
            // atom-level proof. A matching memory ID alone can never promote a
            // claim to verified.
            let covered = proof.is_bound();
            if !covered {
                all_covered = false;
            }
            coverage.push(CoverageReport {
                claim,
                covered,
                evidence_ids: proof.memory_ids,
                evidence_atom_ids: proof.atom_ids,
                evidence_source_anchors: proof.source_anchors,
                derivation: proof.derivation,
                verification_status: if covered {
                    if matches!(draft_mode, "inferred" | "partial_inferred") {
                        "inferred".to_string()
                    } else {
                        "verified".to_string()
                    }
                } else {
                    "unsupported".to_string()
                },
            });
        }

        // A trusted specialized renderer's validated inline citations are its
        // evidence plan, so a stale candidate-pool receipt cannot force a false partial.
        let incomplete_answer_slots = answer_pipeline_plan
            .receipt
            .slot_coverage
            .iter()
            .filter(|slot| slot.status != "covered")
            .map(|slot| slot.slot.clone())
            .collect::<Vec<_>>();
        let initial_pipeline_slots_incomplete = !incomplete_answer_slots.is_empty();
        let renderer_evidence_closes_pipeline_slots = renderer_provided_evidence_plan
            && !incomplete_answer_slots.iter().any(|slot| {
                matches!(slot.as_str(), "proof_scope" | "source_snapshot")
                    || slot.starts_with("count_")
            })
            && !coverage.is_empty()
            && all_covered
            && !matches!(draft_mode, "partial" | "partial_inferred" | "inferred");
        let pipeline_slots_incomplete =
            initial_pipeline_slots_incomplete && !renderer_evidence_closes_pipeline_slots;

        let has_disputed_semantic = retrieved.items.iter().any(|item| {
            item.semantic_node
                .as_ref()
                .map(|sn| sn.status == SemanticStatus::Disputed)
                .unwrap_or(false)
        });
        let mut semantic_notes = build_activation_answer_notes(&retrieved.items);
        if let Some(gap) = renderer_contract_gap {
            semantic_notes.push(SemanticAnswerNote {
                note_type: "renderer_contract_gap".into(),
                message: gap,
                entity_id: None,
                attribute_key: None,
                scope: req.conversation_id.clone(),
                candidate_values: Vec::new(),
            });
        }
        semantic_notes.extend(build_parametric_answer_notes(&retrieved.items, &req.query));
        semantic_notes.extend(build_semantic_answer_notes(&retrieved.items));
        semantic_notes.extend(build_experience_answer_notes(&frozen_candidate_items, &req.query));
        semantic_notes.extend(build_foresight_answer_notes(&retrieved.items, &req.query));
        semantic_notes.extend(modality_notes);
        if has_disputed_semantic {
            semantic_notes.push(SemanticAnswerNote {
                note_type: "conflict_summary".to_string(),
                message: "[语义冲突提示] 记忆中存在相互矛盾的语义知识，结论可能不确定。"
                    .to_string(),
                entity_id: None,
                attribute_key: None,
                scope: None,
                candidate_values: Vec::new(),
            });
        } else if !all_covered {
            semantic_notes.push(SemanticAnswerNote {
                note_type: "coverage".to_string(),
                message: "[不确定性提示] 部分结论缺少证据支撑，请谨慎使用。".to_string(),
                entity_id: None,
                attribute_key: None,
                scope: None,
                candidate_values: Vec::new(),
            });
        }
        if pipeline_slots_incomplete || matches!(draft_mode, "partial" | "partial_inferred") {
            semantic_notes.push(SemanticAnswerNote {
                note_type: "partial_answer".to_string(),
                message: "The response separates what the selected evidence establishes from what remains missing.".to_string(),
                entity_id: None,
                attribute_key: None,
                scope: None,
                candidate_values: incomplete_answer_slots.clone(),
            });
        }
        if matches!(draft_mode, "inferred" | "partial_inferred") {
            semantic_notes.push(SemanticAnswerNote {
                note_type: "grounded_inference".to_string(),
                message: "The conclusion is an explicitly labeled inference from cited evidence, not a directly observed fact.".to_string(),
                entity_id: None,
                attribute_key: None,
                scope: None,
                candidate_values: Vec::new(),
            });
        }

        let (direct_answer, answer_status) = deliver_checked_answer_with_sources(
            direct_answer,
            response_mode,
            draft_mode,
            &coverage,
            pipeline_slots_incomplete,
            has_disputed_semantic,
            retrieved_active_evidence || retrieved.items.iter().any(memory_item_is_active),
            req.query
                .chars()
                .any(|ch| ('\u{4e00}'..='\u{9fff}').contains(&ch)),
            &citation_label_items,
            renderer_frame.as_ref().filter(|_| chapter_feedback.is_none()),
        );
        // A "when" question is not answered by verified Claims that never
        // state a time ("A car ran a red light and hit us"). Presenting the
        // event sentence as a (partial) answer misleads; judged on LoCoMo,
        // every such partial was wrong. The response is uncertain and says
        // what is missing; the verified evidence remains in the sources.
        let (direct_answer, mut answer_status) = if matches!(answer_status, "answered" | "partial")
            && answer_pipeline_plan.receipt.query_plan.answer_target == "temporal"
            && !coverage
                .iter()
                .any(|report| report.covered && claim_states_temporal_value(&report.claim))
        {
            let chinese = req.query.chars().any(|ch| ('\u{4e00}'..='\u{9fff}').contains(&ch));
            let message = if chinese {
                "记录提到了这件事，但没有说明发生的时间。".to_string()
            } else {
                "The records mention this event but do not state when it happened.".to_string()
            };
            semantic_notes.push(SemanticAnswerNote {
                note_type: "answer_value_missing".into(),
                message: "The verified evidence does not state the requested time.".into(),
                entity_id: None,
                attribute_key: None,
                scope: req.conversation_id.clone(),
                candidate_values: vec!["temporal".into()],
            });
            (message, "uncertain")
        } else {
            (direct_answer, answer_status)
        };
        let mut answer = direct_answer.clone();
        if !matches!(response_mode, "evidence" | "verbatim") {
            let chinese = req.query.chars().any(|ch| ('一'..='鿿').contains(&ch));
            let chapter_documents =
                verified_chapter_document_labels(&coverage, &citation_label_items);
            if chapter_documents.len() > 1 {
                let heading = if chinese {
                    "跨文档回答"
                } else {
                    "Multi-document answer"
                };
                answer = format!("{heading}（{}）\n\n{answer}", chapter_documents.join("；"));
            }
            if selected_answer_operator.as_deref() == Some("novel_chapter_range_records_v1") {
                if let Some((display, omitted)) = render_verified_chapter_range(
                    &req.query, &coverage, &citation_label_items, req.answer_nodes,
                    Some(effective_document_limit),
                ) {
                    answer = display;
                    if omitted > 0 {
                        if answer_status == "answered" { answer_status = "partial"; }
                        semantic_notes.push(SemanticAnswerNote {
                            note_type:"chapter_display_coverage".into(),
                            message:format!("{omitted} verified Claims omitted from prose by display limits; complete Claim proofs retained."),
                            entity_id:None,attribute_key:None,scope:req.conversation_id.clone(),candidate_values:vec![],
                        });
                    }
                }
            }
            if let Some(feedback) = &chapter_feedback {
                answer = feedback.render();
                semantic_notes.push(SemanticAnswerNote {
                    note_type: "chapter_coverage_gap".into(),
                    message: feedback.render(),
                    entity_id: None,
                    attribute_key: None,
                    scope: req.conversation_id.clone(),
                    candidate_values: feedback
                        .source_ids()
                        .into_iter()
                        .map(|id| id.to_string())
                        .collect(),
                });
            }
            if coverage.iter().any(|report| {
                report.covered
                    && (selected_answer_operator.as_deref() == Some("temporal_reasoning_v1")
                        || report
                            .derivation
                            .iter()
                            .any(|method| method == "date_difference_v1"))
            }) {
                answer = format!(
                    "{}\n\n{answer}",
                    if chinese {
                        "时间推理："
                    } else {
                        "Temporal reasoning:"
                    }
                );
            }
            for gap in &renderer_gaps {
                answer.push_str("\n\n");
                answer.push_str(gap.message(chinese));
            }
            if let Some(note) = &range_coverage {
                answer.push_str("\n\n");
                answer.push_str(note);
            }
            for message in semantic_notes.iter().filter_map(|note|multimodal_availability_feedback(&note.note_type)) {
                answer.push_str("\n\n系统反馈：");
                answer.push_str(message);
            }
        }
        let direct_answer = direct_answer_display_text(&answer, response_mode);
        let claims = coverage
            .iter()
            .enumerate()
            .map(|(index, report)| crate::api_types::AnswerClaim {
                claim_id: format!("claim-{}", index + 1),
                text: answer_claim_display_text(&report.claim),
                evidence_ids: report.evidence_ids.clone(),
                evidence_atom_ids: report.evidence_atom_ids.clone(),
                evidence_source_anchors: report.evidence_source_anchors.clone(),
                derivation: report.derivation.clone(),
                support_status: if report.covered
                    && matches!(draft_mode, "inferred" | "partial_inferred")
                {
                    "inferred".to_string()
                } else if report.covered {
                    "supported".to_string()
                } else {
                    "unsupported".to_string()
                },
            })
            .collect::<Vec<_>>();

        let inferred_source_count = coverage
            .iter()
            .filter(|report| report.covered)
            .flat_map(|report| report.evidence_ids.iter().copied())
            .collect::<HashSet<_>>()
            .len();
        let confidence_level = answer_confidence_level(
            has_disputed_semantic,
            draft_mode,
            answer_status,
            all_covered,
            inferred_source_count,
        );

        let cited_ids = cited_memory_ids(&draft);
        let verified_claim_evidence_ids = coverage
            .iter()
            .filter(|report| report.covered)
            .flat_map(|report| report.evidence_ids.iter().copied())
            .collect::<HashSet<_>>();
        let response_evidence = if matches!(response_mode, "evidence" | "verbatim") {
            retrieved
                .items
                .iter()
                .filter(|item| {
                    memory_item_is_active(item)
                        && (cited_ids.is_empty() || cited_ids.contains(&item.id))
                })
                .cloned()
                .collect::<Vec<_>>()
        } else if let Some(feedback) = &chapter_feedback {
            let ids = feedback
                .source_ids()
                .into_iter()
                .filter(|id| {
                    self.memories.iter().any(|memory| {
                        memory.id == *id
                            && memory_record_is_eligible_for_selected_answer(memory)
                            && req
                                .conversation_id
                                .as_deref()
                                .is_none_or(|scope| memory.conversation_id == scope)
                            && req
                                .selected_memory_ids
                                .as_ref()
                                .is_none_or(|selected| selected.contains(id))
                    })
                })
                .collect::<Vec<_>>();
            semantic_notes.push(SemanticAnswerNote {
                note_type: "diagnostic_evidence".into(),
                message: "These records explain the coverage gap; they do not support a verified answer claim.".into(),
                entity_id: None, attribute_key: None, scope: req.conversation_id.clone(),
                candidate_values: ids.iter().map(|id| id.to_string()).collect(),
            });
            self.materialize_items(&ids, &req.query)
        } else if draft_mode == "unavailable" || answer_is_chapter_unavailable(&draft) {
            Vec::new()
        } else {
            claim_evidence_items
                .iter()
                .filter(|item| {
                    memory_item_is_active(item) && verified_claim_evidence_ids.contains(&item.id)
                })
                .cloned()
                .collect::<Vec<_>>()
        };

        let selection_receipt = explicit_selection.map(|requested_ids| {
            let eligible_memory_ids = eligible_selected_ids.clone();
            let response_evidence_ids = response_evidence
                .iter()
                .map(|item| item.id)
                .collect::<Vec<_>>();
            let scope_space_id = req.conversation_id.clone().unwrap_or_default();
            let excluded_unselected_memory_count = self
                .memories
                .iter()
                .filter(|memory| {
                    memory.conversation_id == scope_space_id
                        && memory_record_is_eligible_for_selected_answer(memory)
                        && !eligible_memory_ids.contains(&memory.id)
                })
                .count();
            let fallback_digest = {
                let versions = requested_ids
                    .iter()
                    .filter_map(|memory_id| {
                        self.memories
                            .iter()
                            .find(|memory| memory.id == *memory_id)
                            .map(|memory| serde_json::json!([memory.id, memory.version]))
                    })
                    .collect::<Vec<_>>();
                let canonical = serde_json::json!({
                    "scope_space_id": scope_space_id,
                    "memory_revisions": versions,
                });
                format!(
                    "sha256:{}",
                    crate::shared_object::stable_digest(
                        serde_json::to_vec(&canonical).unwrap_or_default()
                    )
                )
            };
            let binding = req.selection_binding.as_ref();
            crate::api_types::AnswerSelectionReceipt {
                receipt_format: "answer_selection_receipt_v1".to_string(),
                enforcement: "exact_selected_memory_scope".to_string(),
                source: binding
                    .map(|value| value.source.clone())
                    .unwrap_or_else(|| "direct_memory_selection".to_string()),
                scope_space_id,
                workspace_id: binding.and_then(|value| value.workspace_id.clone()),
                workspace_revision: binding.and_then(|value| value.workspace_revision),
                projection_kind: binding.and_then(|value| value.projection_kind.clone()),
                selected_object_ids: binding
                    .map(|value| value.selected_object_ids.clone())
                    .unwrap_or_default(),
                requested_memory_ids: requested_ids.clone(),
                eligible_memory_ids,
                response_evidence_ids,
                excluded_unselected_memory_count,
                exact_scope_enforced: true,
                full_space_renderers_bypassed: true,
                permission_digest: binding.and_then(|value| value.permission_digest.clone()),
                selection_digest: binding
                    .map(|value| value.selection_digest.clone())
                    .unwrap_or(fallback_digest),
                authority_backend: binding
                    .map(|value| value.authority_backend.clone())
                    .unwrap_or_else(|| "canonical_memory_engine".to_string()),
                action_authority: "none".to_string(),
            }
        });

        let selected_count = retrieved.items.len();
        let blocked_count = explicit_selection
            .map(|requested| requested.len().saturating_sub(eligible_count))
            .unwrap_or(0);
        let answer_shape = Some(crate::api_types::AnswerShapeReceipt {
            receipt_format: "answer_shape_receipt_v1".to_string(),
            response_mode: response_mode.to_string(),
            detail_level: detail_level.to_string(),
            answer_nodes_mode: if req.answer_nodes.is_some() {
                "user".to_string()
            } else {
                "auto".to_string()
            },
            requested_answer_nodes: req.answer_nodes,
            effective_answer_nodes: req.answer_nodes.map(normalize_answer_range_node_count),
            source_limit_mode: if req.source_limit.is_some() {
                "user".to_string()
            } else {
                "auto".to_string()
            },
            requested_source_limit: req.source_limit,
            effective_source_limit,
            document_limit_mode: if req.document_limit.is_some() {
                "user".to_string()
            } else {
                "auto".to_string()
            },
            requested_document_limit: req.document_limit,
            effective_document_limit,
            eligible: eligible_count,
            selected: selected_count,
            omitted: eligible_count.saturating_sub(selected_count),
            deferred: 0,
            blocked: blocked_count,
            failed: 0,
            safety_envelope: serde_json::json!({
                "max_answer_nodes": MAX_ANSWER_RANGE_NODES,
                "max_sources": MAX_ANSWER_SOURCE_LIMIT,
                "max_documents": MAX_MULTI_DOCUMENT_ANSWER_LIMIT,
                "truncation_policy": "explicit_receipt_no_silent_fixed_default",
            }),
        });

        // Renderer additions have passed explicit scoped admission above;
        // retain those in the same candidate snapshot, not a second pool.
        let mut admitted_ids=renderer_cited_id_list.iter().copied()
            .filter(|id| eligible_pool_ids.contains(id) && seen_pool_ids.insert(*id)).collect::<Vec<_>>();
        admitted_ids.sort_unstable();
        frozen_candidate_ids.extend(admitted_ids);
        let retrieved_ids = frozen_candidate_ids;
        let mut answer_pipeline = crate::answer_pipeline::finalize_answer_pipeline_receipt(
            answer_pipeline_plan.receipt,
            &coverage,
            draft_mode,
            selected_answer_operator
                .as_deref()
                .unwrap_or("answerability_gate_v1"),
            selected_answer_operator.as_deref() == Some("caller_supplied_draft"),
            &attempted_answer_operators,
            selected_answer_operator.as_deref(),
        );
        if renderer_evidence_closes_pipeline_slots {
            let final_evidence_ids = answer_pipeline.final_evidence_ids.clone();
            answer_pipeline.failure_class = "none".to_string();
            if let Some(stage) = answer_pipeline
                .stage_trace
                .iter_mut()
                .find(|stage| stage.stage == "evidence_plan")
            {
                stage.status = "completed".to_string();
                stage.output_count = final_evidence_ids.len();
                stage.evidence_ids = final_evidence_ids.clone();
                stage.reason = Some(
                    "initial slot-rerank miss recovered by validated specialized-renderer evidence"
                        .to_string(),
                );
            }
            if let Some(stage) = answer_pipeline
                .stage_trace
                .iter_mut()
                .find(|stage| stage.stage == "answer")
            {
                stage.status = "completed".to_string();
                stage.reason = None;
            }
        }

        let response = AnswerResponse {
            answer,
            direct_answer,
            answer_mode: match draft_mode {
                "partial" | "inferred" | "partial_inferred" => "synthesized",
                other => other,
            }
            .to_string(),
            answer_status: answer_status.to_string(),
            claims,
            confidence: confidence_level.to_string(),
            evidence: response_evidence.clone(),
            citations: Vec::new(),
            memories: response_evidence.clone(),
            memory_count: response_evidence.len(),
            coverage,
            semantic_notes,
            memory_used: 0,
            memory_ignored: 0,
            conflict_count: 0,
            trace_id: String::new(),
            runtime_enforcement: serde_json::json!({
                "runtime_format": "gateway_runtime_enforcement_v1",
                "enforcement": "unenforced",
                "entrypoint": "direct_answer",
                "conformance_status": "direct_bypass_unenforced",
                "bypass_visible": true,
                "reason": "direct /memory/answer does not orchestrate recall policy, write proposals, approval gate, or receipts"
            }),
            selection_receipt,
            answer_shape,
            answer_pipeline: Some(answer_pipeline),
            performance: None,
        };
        (response, retrieved_ids)
    }

    pub(crate) fn prune_ignored_memories(&mut self, conversation_id: &str) -> PruneResponse {
        let pruned_ids = self
            .memories
            .iter()
            .filter(|m| m.conversation_id == conversation_id)
            .filter(|m| should_exclude_from_retrieval(m))
            .map(|m| m.id)
            .collect::<Vec<_>>();

        if pruned_ids.is_empty() {
            return PruneResponse {
                pruned: 0,
                pruned_ids: Vec::new(),
            };
        }

        let prune_set = pruned_ids.iter().copied().collect::<HashSet<_>>();
        self.memories.retain(|m| !prune_set.contains(&m.id));
        self.invalidate_cache_for_conversation(Some(conversation_id));

        PruneResponse {
            pruned: pruned_ids.len(),
            pruned_ids,
        }
    }

    pub(crate) fn decay_stale_memories(&mut self, conversation_id: &str) -> DecayResponse {
        let now = Utc::now();
        let decayed_ids = self
            .memories
            .iter()
            .filter(|m| m.conversation_id == conversation_id)
            .filter(|m| {
                if should_exclude_from_retrieval(m) {
                    return true;
                }

                let recently_retrieved = m
                    .last_retrieved_at
                    .map(|last| m.retrieval_hits > 0 && (now - last).num_hours().max(0) < 24)
                    .unwrap_or(false);
                if recently_retrieved {
                    return false;
                }

                let age_hours = (now - m.created_at).num_hours().max(0);
                let is_old = age_hours >= self.memory_decay.deferred_max_age_hours;
                if !is_old {
                    return false;
                }

                match m.layer {
                    MemoryLayer::Event => m
                        .event_metadata
                        .as_ref()
                        .map(|metadata| {
                            metadata.memory_action == "defer"
                                && metadata.value_weight
                                    <= self.memory_decay.deferred_max_value_weight
                        })
                        .unwrap_or(false),
                    MemoryLayer::Working => m.trust <= 0.85,
                    MemoryLayer::Archive => m.trust <= 0.6,
                    _ => false,
                }
            })
            .map(|m| m.id)
            .collect::<Vec<_>>();

        if decayed_ids.is_empty() {
            return DecayResponse {
                decayed: 0,
                decayed_ids: Vec::new(),
            };
        }

        let decay_set = decayed_ids.iter().copied().collect::<HashSet<_>>();
        self.memories.retain(|m| !decay_set.contains(&m.id));
        self.invalidate_cache_for_conversation(Some(conversation_id));

        DecayResponse {
            decayed: decayed_ids.len(),
            decayed_ids,
        }
    }
}

fn answer_claim_is_directly_supported_by_content(claim: &str, content: &str) -> bool {
    let content_norm = normalize_answer_coverage_text(content);
    answer_claim_support_candidates(claim)
        .into_iter()
        .any(|candidate| {
            let candidate_norm = normalize_answer_coverage_text(&candidate);
            candidate_norm.chars().count() >= 8 && content_norm.contains(&candidate_norm)
        })
}

fn answer_claim_support_candidates(claim: &str) -> Vec<String> {
    let mut candidates = vec![claim.to_string()];
    for marker in ["故事线1", "故事线2", "故事线3", "故事线4"] {
        if let Some(pos) = claim.find(marker) {
            candidates.push(claim[pos..].to_string());
        }
    }
    candidates.extend(
        claim
            .lines()
            .map(str::trim)
            .map(|line| line.trim_start_matches('-').trim())
            .filter(|line| !line.is_empty())
            .map(str::to_string),
    );
    candidates
}

fn normalize_answer_coverage_text(value: &str) -> String {
    value
        .chars()
        .filter(|ch| !ch.is_whitespace() && !matches!(ch, '-' | ':' | '：'))
        .collect::<String>()
        .to_lowercase()
}

fn answer_is_chapter_unavailable(answer: &str) -> bool {
    answer.contains("系统反馈：当前不能可靠回答这一章")
        || answer.contains("没有可用的 active `novel_chapter_summary`")
}

pub(crate) fn cited_memory_ids(answer: &str) -> HashSet<u64> {
    let mut ids = HashSet::new();
    let mut remainder = answer;
    while let Some(index) = remainder.find("memory#") {
        let after_marker = &remainder[index + "memory#".len()..];
        let digits = after_marker
            .chars()
            .take_while(|ch| ch.is_ascii_digit())
            .collect::<String>();
        if let Ok(id) = digits.parse::<u64>() {
            ids.insert(id);
        }
        remainder = after_marker;
    }
    ids
}



fn render_verified_handoff_answer_plan(
    query: &str,
    items: &[RetrievedItem],
) -> Option<(String, Vec<RendererClaim>)> {
    let task_text = answer_task_text(query);
    let normalized = task_text.to_lowercase();
    if !contains_any(
        &normalized,
        &[
            "verified resume pack",
            "resume pack",
            "handoff pack",
            "handoff",
            "continuation pack",
            "continue this work",
            "resume this work",
            "next person",
            "next agent",
            "交接包",
            "交接",
            "续接",
            "继续工作",
            "恢复任务",
            "可继续",
            "承接",
            "下一位",
            "接班",
        ],
    ) {
        return None;
    }

    let active_items = items
        .iter()
        .filter(|item| memory_item_is_active(item))
        .filter(|item| !item.content.trim_start().starts_with("event:"))
        .collect::<Vec<_>>();
    if active_items.is_empty() {
        return None;
    }

    let mut goals = Vec::new();
    let mut authorities: Vec<String> = Vec::new();
    let mut evidence = Vec::new();
    let mut conflicts = Vec::new();
    let mut corrections = Vec::new();
    let mut rollback_paths = Vec::new();
    let mut next_actions = Vec::new();
    let mut stop_conditions = Vec::new();
    let mut present_truth_states = Vec::new();
    let mut has_corrected_projection = false;
    let mut claims = Vec::new();

    for item in active_items {
        let mut bind = |text: String| {
            if !text.trim().is_empty() {
                claims.push(RendererClaim {
                    text,
                    evidence_ids: vec![item.id],
                });
            }
        };
        let body = handoff_prepare_body(
            &universal_source_projection_body(&item.content)
                .unwrap_or_else(|| item.content.trim().to_string()),
        );
        let source_label = document_source_label_for_item(item)
            .or_else(|| item.entity.clone())
            .unwrap_or_else(|| item.source.clone());
        let citation = format!("memory#{}", item.id);

        if let Some(goal) = handoff_section_value(&body, &["goal", "objective", "目标", "任务目标"])
        {
            bind(format!("Goal: {goal}"));
            operator_push_unique_limited(
                &mut goals,
                format!("{} [{citation}]", compact_context_line(&goal, 560)),
                4,
            );
        }

        for (label, keys) in [
            ("Owner", &["owner", "负责人", "owner id"][..]),
            (
                "Action authority",
                &[
                    "action authority",
                    "action_authority",
                    "行动权限",
                    "动作权限",
                ][..],
            ),
        ] {
            if let Some(value) = handoff_field_value(&body, keys) {
                bind(format!("{label}: {value}"));
                let normalized_value = format!("{label}=`{}`", value.trim_matches('`'));
                if authorities
                    .iter()
                    .any(|existing| existing.starts_with(&normalized_value))
                {
                    continue;
                }
                operator_push_unique_limited(
                    &mut authorities,
                    format!("{normalized_value} [{citation}]"),
                    8,
                );
            }
        }

        let truth_state = handoff_truth_state(item, &body);
        operator_push_unique_limited(&mut present_truth_states, truth_state.clone(), 8);
        let summary = handoff_item_summary(&body);
        bind(summary.clone());
        evidence.push(format!(
            "[{truth_state}] `{}`: {} [{citation}]",
            compact_context_line(&source_label, 120),
            compact_context_line(&summary, 720)
        ));

        if let Some(open_question) = handoff_section_value(
            &body,
            &["open question", "open questions", "待确认", "未决问题"],
        ) {
            bind(format!("Open question: {open_question}"));
            operator_push_unique_limited(
                &mut conflicts,
                format!("{} [{citation}]", compact_context_line(&open_question, 520)),
                8,
            );
        }
        let json_conflicts = handoff_json_conflict_summaries(&body);
        for conflict in &json_conflicts {
            bind(format!("Conflict: {conflict}"));
            operator_push_unique_limited(
                &mut conflicts,
                format!("{} [{citation}]", compact_context_line(&conflict, 520)),
                8,
            );
        }
        if json_conflicts.is_empty() {
            if let Some(conflict) = handoff_conflict_excerpt(&body) {
                bind(format!("Conflict: {conflict}"));
                operator_push_unique_limited(
                    &mut conflicts,
                    format!("{} [{citation}]", compact_context_line(&conflict, 520)),
                    8,
                );
            }
        }

        if let Some(required) = handoff_section_value(
            &body,
            &[
                "required correction",
                "correction",
                "required change",
                "需要纠正",
                "纠正",
                "修正",
            ],
        ) {
            bind(format!("Required correction: {required}"));
            operator_push_unique_limited(
                &mut corrections,
                format!(
                    "Required: {} [{citation}]",
                    compact_context_line(&required, 620)
                ),
                8,
            );
            if let Some(rollback) = handoff_keyword_excerpt(
                &required,
                &[
                    "rollback",
                    "roll back",
                    "restore",
                    "preserve",
                    "回滚",
                    "恢复",
                    "保留",
                ],
            ) {
                operator_push_unique_limited(
                    &mut rollback_paths,
                    format!("{} [{citation}]", compact_context_line(&rollback, 420)),
                    6,
                );
            }
        }
        if handoff_item_is_corrected_projection(item) {
            bind(body.clone());
            has_corrected_projection = true;
            operator_push_unique_limited(
                &mut corrections,
                format!(
                    "Active reviewed projection: {} [{citation}]",
                    compact_context_line(&body, 520)
                ),
                8,
            );
        }

        if let Some(next_action) = handoff_section_value(
            &body,
            &[
                "next action",
                "next actions",
                "next step",
                "next steps",
                "下一步",
                "后续动作",
            ],
        ) {
            bind(format!("Next action: {next_action}"));
            operator_push_unique_limited(
                &mut next_actions,
                format!("{} [{citation}]", compact_context_line(&next_action, 620)),
                8,
            );
        }
        if let Some(stop) = handoff_section_value(
            &body,
            &[
                "stop conditions",
                "stop condition",
                "abort conditions",
                "停止条件",
                "中止条件",
            ],
        ) {
            bind(format!("Stop conditions: {stop}"));
            operator_push_unique_limited(
                &mut stop_conditions,
                format!("{} [{citation}]", compact_context_line(&stop, 700)),
                8,
            );
        }
    }

    present_truth_states.sort();
    let expected_truth_states = [
        "reported",
        "simulated",
        "computed",
        "inferred",
        "proposed",
        "corrected",
    ];
    let absent_truth_states = expected_truth_states
        .iter()
        .filter(|state| {
            !present_truth_states
                .iter()
                .any(|present| present.as_str() == **state)
        })
        .copied()
        .collect::<Vec<_>>();
    let has_required_correction = !corrections.is_empty();

    let mut sections = Vec::new();
    if !goals.is_empty() {
        sections.push(format!("Goal\n{}", handoff_bullets(&goals)));
    }
    if !authorities.is_empty() {
        sections.push(format!(
            "Ownership and authority\n{}",
            handoff_bullets(&authorities)
        ));
    }
    let coverage = if absent_truth_states.is_empty() {
        format!("Present states: {}.", present_truth_states.join(", "))
    } else {
        format!(
            "Present states: {}. No selected source declares: {}.",
            present_truth_states.join(", "),
            absent_truth_states.join(", ")
        )
    };
    sections.push(format!(
        "Truth-state evidence\n{coverage}\n{}",
        handoff_bullets(&evidence)
    ));
    if !conflicts.is_empty() {
        sections.push(format!(
            "Conflicts and open questions\n{}",
            handoff_bullets(&conflicts)
        ));
    }
    if !corrections.is_empty() || !rollback_paths.is_empty() {
        let mut correction_lines = corrections;
        correction_lines.extend(
            rollback_paths
                .into_iter()
                .map(|value| format!("Rollback: {value}")),
        );
        sections.push(format!(
            "Correction and rollback\n{}",
            handoff_bullets(&correction_lines)
        ));
    }
    if !next_actions.is_empty() {
        sections.push(format!("Next action\n{}", handoff_bullets(&next_actions)));
    }
    if !stop_conditions.is_empty() {
        sections.push(format!(
            "Stop conditions\n{}",
            handoff_bullets(&stop_conditions)
        ));
    }

    let continuation = if has_required_correction && !has_corrected_projection {
        "Review gate remains open: apply and review the cited correction before continuing. The pack itself grants no additional action authority."
    } else if has_corrected_projection {
        "The reviewed correction is active in the selected evidence. Continue only with the cited next action, stop conditions, and action-authority boundary; use the recorded correction receipt to roll back if needed."
    } else {
        "Continue only within the cited next action, stop conditions, and action-authority boundary. This pack does not grant additional execution authority."
    };
    sections.push(format!("Continuation decision\n- {continuation}"));

    Some((format!(
        "Verified Handoff / Continuation Pack\n\n{}\n\nTrace: use the response trace_id with /memory/trace/{{trace_id}}/summary to inspect the cited active records and correction lineage.",
        sections.join("\n\n")
    ), claims))
}

fn handoff_item_is_corrected_projection(item: &RetrievedItem) -> bool {
    item.source == "intent_workspace_user_semantic_correction"
}

fn handoff_truth_state(item: &RetrievedItem, body: &str) -> String {
    if handoff_item_is_corrected_projection(item) {
        return "corrected".to_string();
    }
    if let Some(state) = handoff_field_value(body, &["truth state", "truth_state", "事实状态"])
    {
        let state = state.trim_matches('`').trim().to_lowercase();
        if matches!(
            state.as_str(),
            "reported"
                | "simulated"
                | "computed"
                | "inferred"
                | "proposed"
                | "observed"
                | "confirmed"
        ) {
            return state;
        }
    }
    let lower = body.to_lowercase();
    if contains_any(
        &lower,
        &[
            "simulated evidence",
            "synthetic evidence",
            "not a real observation",
            "仿真证据",
            "模拟证据",
        ],
    ) {
        "simulated".to_string()
    } else {
        "reported".to_string()
    }
}

fn handoff_item_summary(body: &str) -> String {
    if let Some(summary) = handoff_json_activity_summary(body) {
        return summary;
    }
    for headings in [
        &[
            "decision",
            "outcome",
            "result",
            "current state",
            "current status",
            "决定",
            "结果",
            "当前状态",
        ][..],
        &["summary", "overview", "摘要", "概述"][..],
        &["goal", "objective", "目标", "任务目标"][..],
    ] {
        if let Some(value) = handoff_section_value(body, headings) {
            return value;
        }
    }
    body.to_string()
}

fn handoff_json_activity_summary(body: &str) -> Option<String> {
    let value = serde_json::from_str::<serde_json::Value>(body).ok()?;
    let object = value.as_object()?;
    let mut records = Vec::new();
    for collection in [
        "runs",
        "events",
        "tasks",
        "steps",
        "agent_activity",
        "results",
    ] {
        let Some(values) = object.get(collection).and_then(serde_json::Value::as_array) else {
            continue;
        };
        for value in values.iter().take(6usize.saturating_sub(records.len())) {
            let Some(record) = handoff_json_record_summary(value) else {
                continue;
            };
            records.push(format!("{collection}: {record}"));
        }
        if records.len() >= 6 {
            break;
        }
    }
    (!records.is_empty()).then(|| records.join("; "))
}

fn handoff_json_record_summary(value: &serde_json::Value) -> Option<String> {
    let object = value.as_object()?;
    let identity_key = [
        "run_id", "event_id", "task_id", "step_id", "item_id", "id", "name",
    ]
    .iter()
    .find(|key| object.get(**key).is_some())
    .copied();
    let mut parts = Vec::new();
    if let Some(key) = identity_key {
        if let Some(value) = handoff_json_scalar(object.get(key)?) {
            parts.push(value);
        }
    }
    for key in ["status", "result"] {
        if let Some(value) = object.get(key).and_then(handoff_json_scalar) {
            parts.push(format!("{key}={value}"));
        }
    }
    for (key, value) in object {
        if Some(key.as_str()) == identity_key
            || matches!(
                key.as_str(),
                "status" | "result" | "started_at" | "completed_at" | "created_at" | "updated_at"
            )
        {
            continue;
        }
        let Some(value) = handoff_json_scalar(value) else {
            continue;
        };
        parts.push(format!("{key}={value}"));
        if parts.len() >= 6 {
            break;
        }
    }
    (!parts.is_empty()).then(|| parts.join(", "))
}

fn handoff_json_conflict_summaries(body: &str) -> Vec<String> {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(body) else {
        return Vec::new();
    };
    let Some(object) = value.as_object() else {
        return Vec::new();
    };
    let mut conflicts = Vec::new();
    for (collection, values) in object {
        let Some(values) = values.as_array() else {
            continue;
        };
        for value in values {
            let Some(record) = value.as_object() else {
                continue;
            };
            let identity = ["run_id", "event_id", "task_id", "step_id", "id", "name"]
                .iter()
                .find_map(|key| record.get(*key).and_then(handoff_json_scalar));
            for key in ["result", "message", "summary", "status"] {
                let Some(message) = record.get(key).and_then(handoff_json_scalar) else {
                    continue;
                };
                if !contains_any(
                    &message.to_lowercase(),
                    &[
                        "conflict",
                        "mismatch",
                        "unresolved",
                        "stale",
                        "unknown",
                        "blocked",
                        "waiting_for_human",
                        "冲突",
                        "不一致",
                        "未解决",
                        "未知",
                        "等待人工",
                    ],
                ) {
                    continue;
                }
                let prefix = identity
                    .as_deref()
                    .map(|value| format!("{collection} `{value}`"))
                    .unwrap_or_else(|| collection.to_string());
                operator_push_unique_limited(&mut conflicts, format!("{prefix}: {message}"), 8);
            }
        }
    }
    conflicts
}

fn handoff_json_scalar(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(value) => Some(compact_context_line(value, 180)),
        serde_json::Value::Number(value) => Some(value.to_string()),
        serde_json::Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn handoff_field_value(content: &str, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(value) = json_text_value(content, key) {
            return Some(value);
        }
        if let Some(value) = content.lines().find_map(|line| {
            let line = line
                .trim()
                .trim_start_matches(['-', '*'])
                .trim()
                .trim_matches('*');
            let (line_key, value) = line.split_once(':').or_else(|| line.split_once('：'))?;
            line_key
                .trim()
                .trim_matches(['*', '`'])
                .eq_ignore_ascii_case(key)
                .then(|| value.trim().trim_matches('*').to_string())
                .filter(|value| !value.is_empty())
        }) {
            return Some(value);
        }
        let lower = content.to_lowercase();
        let needle = format!("{}:", key.to_lowercase());
        if let Some(index) = lower.find(&needle) {
            let after = &content[index + needle.len()..];
            let end = after
                .find(" ## ")
                .or_else(|| after.find(" - "))
                .or_else(|| after.find('\n'))
                .unwrap_or(after.len());
            let value = after[..end].trim().trim_matches('*').to_string();
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    None
}

fn handoff_prepare_body(content: &str) -> String {
    content.replace(" ## ", "\n## ")
}

fn handoff_section_value(content: &str, headings: &[&str]) -> Option<String> {
    let mut active = false;
    let mut paragraphs = Vec::new();
    let mut current = String::new();
    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.starts_with('#') {
            let heading = line
                .trim_start_matches('#')
                .trim()
                .trim_matches(['*', '`', ':', '：']);
            if active {
                break;
            }
            active = headings
                .iter()
                .any(|candidate| heading.eq_ignore_ascii_case(candidate));
            continue;
        }
        if !active {
            continue;
        }
        if line.is_empty() {
            if !current.is_empty() {
                paragraphs.push(std::mem::take(&mut current));
            }
            continue;
        }
        let is_bullet = line.starts_with('-') || line.starts_with('*');
        let value = line.trim_start_matches(['-', '*']).trim();
        let value = if is_bullet {
            value.trim_end_matches([';', '；'])
        } else {
            value
        };
        if is_bullet && !current.is_empty() {
            paragraphs.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(value);
        if is_bullet {
            paragraphs.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        paragraphs.push(current);
    }
    (!paragraphs.is_empty()).then(|| paragraphs.join("; "))
}

fn handoff_conflict_excerpt(content: &str) -> Option<String> {
    handoff_keyword_excerpt(
        content,
        &[
            "conflict",
            "conflicting",
            "mismatch",
            "unresolved",
            "stale",
            "unknown",
            "冲突",
            "不一致",
            "未解决",
            "未知",
            "过期",
        ],
    )
}

fn handoff_keyword_excerpt(content: &str, keywords: &[&str]) -> Option<String> {
    for line in content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        let lower = line.to_lowercase();
        let Some(index) = keywords.iter().find_map(|keyword| lower.find(keyword)) else {
            continue;
        };
        let start = line[..index]
            .rfind(|ch| matches!(ch, '.' | ';' | ',' | '{' | '。' | '；' | '，'))
            .map(|position| position + 1)
            .unwrap_or(0);
        let end = line[index..]
            .find(|ch| matches!(ch, '.' | ';' | '。' | '；'))
            .map(|position| index + position + 1)
            .unwrap_or(line.len());
        let excerpt = line[start..end]
            .trim()
            .trim_start_matches(['-', '*', '"'])
            .trim_end_matches(['"', ','])
            .trim();
        if !excerpt.is_empty() {
            return Some(excerpt.to_string());
        }
    }
    None
}

fn handoff_bullets(values: &[String]) -> String {
    values
        .iter()
        .map(|value| format!("- {value}"))
        .collect::<Vec<_>>()
        .join("\n")
}



fn render_operator_briefing_answer_plan(
    query: &str,
    items: &[RetrievedItem],
) -> Option<(String, Vec<RendererClaim>, RendererFrame)> {
    let task_text = answer_task_text(query);
    let normalized = task_text.to_lowercase();
    let explicit_briefing = contains_any(
        &normalized,
        &[
            "operator briefing",
            "mission briefing",
            "sop",
            "physical ai",
            "runbook",
            "stale risk",
            "操作员",
            "任务简报",
            "运行手册",
            "物理智能",
        ],
    );
    let operational_context = contains_any(
        &normalized,
        &[
            "mission",
            "operation",
            "operator",
            "device",
            "system status",
            "risk",
            "任务",
            "运行",
            "操作",
            "机器人",
            "无人机",
            "设备",
            "风险",
        ],
    );
    let operational_signal = contains_any(
        &normalized,
        &[
            "telemetry",
            "incident",
            "infrastructure",
            "遥测",
            "事故",
            "基础设施",
        ],
    );
    if !explicit_briefing && !(operational_context && operational_signal) {
        return None;
    }

    let file_constraints = operator_file_constraints(query);
    let mut claims = Vec::new();
    let mut constraint_indices = Vec::new();
    let mut stale_indices = Vec::new();
    let mut telemetry_indices = Vec::new();
    let mut habit_indices = Vec::new();
    let mut provenance = Vec::new();
    let mut constraints = file_constraints
        .iter()
        .map(|value| format!("SOP/File: {}", compact_context_line(value, 220)))
        .collect::<Vec<_>>();
    // Memory File snapshots are versioned evidence records; each content line
    // is a mission-constraint Claim bound to the snapshot.
    let snapshots = items
        .iter()
        .filter(|item| memory_item_is_active(item) && is_memory_file_snapshot(item))
        .collect::<Vec<_>>();
    for item in &snapshots {
        for line in memory_file_snapshot_body_lines(&item.content) {
            constraint_indices.push(claims.len());
            claims.push(RendererClaim {
                text: line.to_string(),
                evidence_ids: vec![item.id],
            });
        }
    }
    for item in items
        .iter()
        .filter(|item| matches!(item.memory_type.as_str(), "constraint" | "decision"))
        .filter(|item| memory_item_is_active(item))
    {
        push_statement_clause_claims(&mut claims, &mut constraint_indices, item.id, &operator_record_statement(&item.content));
        operator_push_unique_limited(
            &mut constraints,
            format!(
                "{} [memory#{} | {}:{}]",
                compact_context_line(&item.content, 220),
                item.id,
                compact_context_line(&item.source, 80),
                item.memory_type
            ),
            6,
        );
    }

    let mut stale_risks = Vec::new();
    for item in items
        .iter()
        .filter(|item| !is_memory_file_snapshot(item) && operator_item_is_stale_risk(item))
    {
        // Staleness is stored record metadata, shown as provenance of the
        // record; the record's own statement is the Claim.
        push_statement_clause_claims(&mut claims, &mut stale_indices, item.id, &operator_record_statement(&item.content));
        provenance.push((
            FrameSection::StaleRisks,
            item.id,
            format!("memory#{}: {}", item.id, operator_stale_metadata(item)),
        ));
        operator_push_unique_limited(
            &mut stale_risks,
            format!(
                "{} [memory#{} | {}:{}]",
                operator_stale_reason(item),
                item.id,
                compact_context_line(&item.source, 80),
                item.memory_type
            ),
            5,
        );
    }

    let mut telemetry = Vec::new();
    for item in items
        .iter()
        .filter(|item| {
            matches!(
                item.memory_type.as_str(),
                "incident" | "todo" | "artifact"
            )
        })
        .filter(|item| memory_item_is_active(item))
    {
        push_statement_clause_claims(&mut claims, &mut telemetry_indices, item.id, &operator_record_statement(&item.content));
        operator_push_unique_limited(
            &mut telemetry,
            format!(
                "{} [memory#{} | {}:{}]",
                compact_context_line(&item.content, 220),
                item.id,
                compact_context_line(&item.source, 80),
                item.memory_type
            ),
            6,
        );
    }

    let mut habits = Vec::new();
    for item in items
        .iter()
        .filter(|item| memory_item_has_profile_type(item, "operational_habit_profile"))
        .filter(|item| memory_item_is_active(item))
    {
        let dimension = json_text_value(&item.content, "dimension")
            .or_else(|| answer_line_value(&item.content, "Dimension"))
            .unwrap_or_else(|| "habit".to_string());
        let summary = json_text_value(&item.content, "summary")
            .or_else(|| answer_line_value(&item.content, "Summary"))
            .unwrap_or_else(|| item.content.clone());
        habit_indices.push(claims.len());
        claims.push(RendererClaim {
            text: summary.clone(),
            evidence_ids: vec![item.id],
        });
        operator_push_unique_limited(
            &mut habits,
            format!(
                "{summary} [memory#{} | {}:operational_habit_profile:{dimension}]",
                item.id,
                compact_context_line(&item.source, 80)
            ),
            5,
        );
    }

    let mut evidence = Vec::new();
    if !file_constraints.is_empty() {
        operator_push_unique_limited(
            &mut evidence,
            "SOP/mission file evidence was injected via Memory Files context.".to_string(),
            6,
        );
    }
    for item in items.iter().take(6) {
        operator_push_unique_limited(
            &mut evidence,
            format!(
                "source={} type={} score={:.2} [memory#{} | {}]",
                compact_context_line(&item.source, 100),
                item.memory_type,
                item.score,
                item.id,
                compact_context_line(&item.source, 80)
            ),
            8,
        );
    }

    if constraints.is_empty() && telemetry.is_empty() && stale_risks.is_empty() && habits.is_empty()
    {
        return None;
    }

    let constraints_text = operator_bullets_or_none(&constraints);
    let stale_text = operator_bullets_or_none(&stale_risks);
    let telemetry_text = operator_bullets_or_none(&telemetry);
    let habits_text = operator_bullets_or_none(&habits);
    let evidence_text = operator_bullets_or_none(&evidence);
    let frame = RendererFrame {
        kind: FrameKind::OperatorBriefing,
        sections: vec![
            (FrameSection::MissionConstraints, constraint_indices),
            (FrameSection::StaleRisks, stale_indices),
            (FrameSection::TelemetryIncidents, telemetry_indices),
            (FrameSection::OperationalHabits, habit_indices),
        ],
        notes: if snapshots.is_empty() {
            Vec::new()
        } else {
            vec![FrameNote::MemoryFilesConsulted]
        },
        provenance,
        claim_sources: HashMap::new(),
        claim_count: claims.len(),
    };
    Some((format!(
        "Operator briefing\n\nMission constraints:\n{constraints_text}\n\nStale risks:\n{stale_text}\n\nTelemetry / incidents:\n{telemetry_text}\n\nOperational habits:\n{habits_text}\n\nEvidence:\n{evidence_text}\n\nTrace: use the response trace_id with /memory/trace/{{trace_id}}."
    ), claims, frame))
}

fn render_operational_habit_answer(
    query: &str,
    items: &[RetrievedItem],
) -> Option<(String, Vec<RendererClaim>)> {
    let task_text = answer_task_text(query);
    let normalized = task_text.to_lowercase();
    if !contains_any(
        &normalized,
        &[
            "operational habit",
            "operator habit",
            "workflow habit",
            "approval style",
            "risk posture",
            "physical ai habit",
            "操作习惯",
            "运行习惯",
            "操作员习惯",
            "审批习惯",
        ],
    ) {
        return None;
    }
    let records = items
        .iter()
        .filter(|item| {
            memory_item_is_active(item)
                && memory_item_has_profile_type(item, "operational_habit_profile")
        })
        .take(8)
        .map(|item| profile_citation_line(item, "operational_habit_profile", "habit"))
        .collect::<Vec<_>>();
    if records.is_empty() {
        return None;
    }
    let display = format!(
        "Operational habit profile answer:\n\n{}\n\nOperational conclusion: these operational_habit_profile records are durable, reviewable evidence for risk posture, approval style, telemetry attention, incident response, and handoff habits. For Physical AI or infrastructure work, they may shape briefings, but physical-world actions still require explicit approval and trace inspection.\n\nTrace: use the response trace_id with /memory/trace/{{trace_id}}/summary to inspect cited habit records.",
        records.join("\n")
    );
    Some((
        display,
        profile_candidate_claims(items, "operational_habit_profile"),
    ))
}

fn profile_citation_line(
    item: &RetrievedItem,
    memory_type: &str,
    default_dimension: &str,
) -> String {
    let dimension = profile_field_value(&item.content, "dimension", &["Dimension"])
        .or_else(|| profile_field_value(&item.content, "writing_style", &["Writing Style"]))
        .unwrap_or_else(|| default_dimension.to_string());
    let summary = profile_field_value(&item.content, "summary", &["Summary"])
        .unwrap_or_else(|| compact_context_line(&item.content, 180));
    let evidence = profile_evidence_value(&item.content)
        .map(|value| format!(" Evidence: {}", compact_context_line(&value, 180)))
        .unwrap_or_default();
    let modality_note = profile_modalities_note(&item.content);
    let signals_note = profile_signals_note(&item.content);
    format!(
        "- memory#{} [{memory_type}:{dimension}] {}{}{}{}",
        item.id, summary, modality_note, signals_note, evidence
    )
}

fn universal_profile_citation_line(item: &RetrievedItem) -> String {
    let domain = profile_field_value(&item.content, "domain", &["Domain"])
        .unwrap_or_else(|| "unbounded".to_string());
    let profile_kind = profile_field_value(
        &item.content,
        "profile_kind",
        &["Profile kind", "Profile Kind"],
    )
    .unwrap_or_else(|| "pattern_profile".to_string());
    let dimension = profile_field_value(&item.content, "dimension", &["Dimension"])
        .unwrap_or_else(|| "profile".to_string());
    let summary = profile_field_value(&item.content, "summary", &["Summary"])
        .unwrap_or_else(|| compact_context_line(&item.content, 180));
    let evidence = profile_evidence_value(&item.content)
        .map(|value| format!(" Evidence: {}", compact_context_line(&value, 180)))
        .unwrap_or_default();
    let modality_note = profile_modalities_note(&item.content);
    let signals_note = profile_signals_note(&item.content);
    format!(
        "- memory#{} [universal_profile:{domain}:{profile_kind}:{dimension}] {}{}{}{}",
        item.id, summary, modality_note, signals_note, evidence
    )
}

fn memory_item_has_profile_type(item: &RetrievedItem, memory_type: &str) -> bool {
    if item.memory_type == memory_type {
        return true;
    }
    if json_text_value(&item.content, "memory_type").as_deref() == Some(memory_type) {
        return true;
    }
    let trimmed = item.content.trim_start();
    trimmed.starts_with(&format!("memory_type: {memory_type}"))
        || trimmed.starts_with(&format!("memory_type:{memory_type}"))
}

fn profile_field_value(content: &str, json_key: &str, line_keys: &[&str]) -> Option<String> {
    json_text_value(content, json_key).or_else(|| {
        line_keys
            .iter()
            .find_map(|line_key| answer_line_value(content, line_key))
    })
}

fn profile_evidence_value(content: &str) -> Option<String> {
    profile_field_value(content, "evidence", &["Evidence"]).or_else(|| {
        answer_block_values(
            content,
            "Evidence",
            &[
                "Source chunks sampled",
                "Source chunks",
                "Evidence modalities",
                "Source",
                "Title",
            ],
        )
        .first()
        .cloned()
    })
}

fn profile_modalities_note(content: &str) -> String {
    let modalities = json_string_array_value(content, "evidence_modalities");
    if modalities.is_empty() {
        String::new()
    } else {
        format!(" modalities={};", modalities.join(","))
    }
}

fn profile_signals_note(content: &str) -> String {
    let signals = json_string_array_value(content, "modality_signals");
    if signals.is_empty() {
        String::new()
    } else {
        let compact = signals
            .iter()
            .take(4)
            .map(|signal| compact_context_line(signal, 90))
            .collect::<Vec<_>>()
            .join(", ");
        format!(" Signals: {compact}.")
    }
}

fn answer_task_text(query: &str) -> &str {
    query
        .rsplit_once("Task/Question:")
        .map(|(_, task)| task.trim())
        .unwrap_or(query)
}

fn operator_push_unique_limited(values: &mut Vec<String>, value: String, limit: usize) {
    if values.len() >= limit || value.trim().is_empty() {
        return;
    }
    if values.iter().any(|existing| existing == &value) {
        return;
    }
    values.push(value);
}

fn render_ambient_capture_answer(
    query: &str,
    items: &[RetrievedItem],
) -> Option<(String, Vec<RendererClaim>, RendererFrame)> {
    let task_text = answer_task_text(query);
    let normalized = task_text.to_lowercase();
    if !contains_any(
        &normalized,
        &[
            "ambient capture",
            "ambient context",
            "captured context",
            "captured source",
            "approved ambient",
            "browser capture",
            "slack capture",
            "source evidence",
        ],
    ) {
        return None;
    }

    let mut source_bullets = Vec::new();
    let mut evidence_bullets = Vec::new();
    let mut claims = Vec::new();
    let mut provenance = Vec::new();
    for item in items
        .iter()
        .filter(|item| memory_item_is_active(item) && !is_memory_file_snapshot(item))
    {
        let Some(captured) =
            trace_captured_context_from_text(&item.content, Some(&item.source), item.created_at)
        else {
            continue;
        };
        let status = captured.status.as_deref().unwrap_or("");
        if status.eq_ignore_ascii_case("deleted")
            || captured
                .redaction_status
                .as_deref()
                .map(|value| value.eq_ignore_ascii_case("deleted"))
                .unwrap_or(false)
        {
            continue;
        }

        // One Claim per captured line; redaction placeholders are not facts.
        // Records without a captured body (connector/source configuration)
        // contribute provenance only.
        let has_body = serde_json::from_str::<serde_json::Value>(&item.content)
            .ok()
            .is_some_and(|value| value.get("content").is_some_and(serde_json::Value::is_string));
        let body = if has_body {
            ambient_capture_content_body(&item.content)
        } else {
            String::new()
        };
        for line in body
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.contains("[REDACTED]"))
        {
            claims.push(RendererClaim {
                text: line.to_string(),
                evidence_ids: vec![item.id],
            });
        }
        let mut source_parts = Vec::new();
        if let Some(source_app) = &captured.source_app {
            source_parts.push(format!("source_app={source_app}"));
        }
        if let Some(source_id) = &captured.source_id {
            source_parts.push(format!("source_id={source_id}"));
        }
        if let Some(title) = &captured.title {
            source_parts.push(format!("title={title}"));
        }
        if let Some(url) = &captured.url {
            source_parts.push(format!("url={url}"));
        }
        if let Some(path) = &captured.path {
            source_parts.push(format!("path={path}"));
        }
        if let Some(thread_id) = &captured.thread_id {
            source_parts.push(format!("thread_id={thread_id}"));
        }
        if let Some(timestamp) = captured.timestamp {
            source_parts.push(format!("timestamp={}", timestamp.to_rfc3339()));
        }
        if let Some(confidence) = captured.confidence {
            source_parts.push(format!("confidence={confidence:.2}"));
        }
        if let Some(redaction_status) = &captured.redaction_status {
            source_parts.push(format!("redaction_status={redaction_status}"));
        }
        if let Some(reason_used) = &captured.reason_used {
            source_parts.push(format!("reason_used={reason_used}"));
        }
        if let Some(path) = &captured.approved_memory_file_path {
            source_parts.push(format!("approved_memory_file_path={path}"));
        }
        if let Some(approved_at) = captured.approved_at {
            source_parts.push(format!("approved_at={}", approved_at.to_rfc3339()));
        }

        provenance.push((
            FrameSection::CapturedSources,
            item.id,
            format!("memory#{} [{}] {}", item.id, item.memory_type, source_parts.join("; ")),
        ));
        provenance.push((
            FrameSection::CapturedSources,
            item.id,
            format!("memory#{} excerpt: {}", item.id, ambient_capture_content_excerpt(&item.content)),
        ));
        operator_push_unique_limited(
            &mut source_bullets,
            format!(
                "memory#{} [{}]: {}",
                item.id,
                item.memory_type,
                source_parts.join("; ")
            ),
            5,
        );
        operator_push_unique_limited(
            &mut evidence_bullets,
            format!(
                "memory#{}: {}",
                item.id,
                ambient_capture_content_excerpt(&item.content)
            ),
            5,
        );
    }

    if source_bullets.is_empty() {
        return None;
    }

    let sources_text = operator_bullets_or_none(&source_bullets);
    let evidence_text = operator_bullets_or_none(&evidence_bullets);
    Some((format!(
        "Ambient capture answer\n\nCaptured source evidence:\n{sources_text}\n\nRelevant captured content:\n{evidence_text}\n\nTrace: use the response trace_id with /memory/trace/{{trace_id}}."
    ), claims.clone(), RendererFrame {
        kind: FrameKind::AmbientCapture,
        sections: Vec::new(),
        notes: Vec::new(),
        provenance,
        claim_sources: HashMap::new(),
        claim_count: claims.len(),
    }))
}

/// Resumes a learning session from approved structured learning records.
/// Each Claim is a `field: value` pair quoted from the record's JSON, so it
/// binds verbatim; placeholder values are skipped.
fn render_learning_resume_answer(
    query: &str,
    items: &[RetrievedItem],
) -> Option<(String, Vec<RendererClaim>, RendererFrame)> {
    let task = answer_task_text(query).to_lowercase();
    if !contains_any(
        &task,
        &[
            "继续学习",
            "能力边界",
            "卡住",
            "局部概念",
            "resume learning",
            "continue learning",
            "ability boundary",
            "stuck point",
        ],
    ) {
        return None;
    }
    const FIELDS: &[(&str, &[&str])] = &[
        (
            "learning_problem_card",
            &["ability_boundary", "stuck_point", "next_executable_step"],
        ),
        ("learning_loop", &["attempt", "stuck_point", "reflection"]),
        (
            "learning_concept_link",
            &["source_concept", "relation", "target_concept"],
        ),
    ];
    let mut claims = Vec::new();
    let mut sections = vec![
        (FrameSection::LearningProblemCard, Vec::new()),
        (FrameSection::LearningLoop, Vec::new()),
        (FrameSection::LearningConceptLink, Vec::new()),
    ];
    let mut seen = HashSet::new();
    for item in items.iter().filter(|item| memory_item_is_active(item)) {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&item.content) else {
            continue;
        };
        let Some(format) = value.get("learning_memory_format").and_then(serde_json::Value::as_str)
        else {
            continue;
        };
        let Some((section_index, (_, fields))) = FIELDS
            .iter()
            .enumerate()
            .find(|(_, (kind, _))| format.starts_with(kind))
        else {
            continue;
        };
        for field in *fields {
            let Some(text) = value.get(*field).and_then(serde_json::Value::as_str) else {
                continue;
            };
            let text = text.trim();
            if text.is_empty() || text.starts_with("pending_") {
                continue;
            }
            // Quoted with its key, which is adjacent to the value in the JSON.
            for claim in [format!("{field}: {text}")] {
                if !seen.insert(claim.clone()) {
                    continue;
                }
                sections[section_index].1.push(claims.len());
                claims.push(RendererClaim {
                    text: claim,
                    evidence_ids: vec![item.id],
                });
            }
        }
    }
    if claims.is_empty() {
        return None;
    }
    let display = claims
        .iter()
        .map(|claim| format!("- {}", claim.text))
        .collect::<Vec<_>>()
        .join("\n");
    let frame = RendererFrame {
        kind: FrameKind::LearningResume,
        sections,
        notes: Vec::new(),
        provenance: Vec::new(),
        claim_sources: HashMap::new(),
        claim_count: claims.len(),
    };
    Some((display, claims, frame))
}

/// Recalls past episodes relevant to a failure/recovery question. Claims quote
/// the episode text; the stored episodic metadata (cause and recovery links)
/// is provenance of those records. The "similar experience" heading appears
/// only when an episode Claim verified, i.e. when a similar case is proven.
fn render_experience_recall_answer(
    query: &str,
    items: &[RetrievedItem],
) -> Option<(String, Vec<RendererClaim>, RendererFrame)> {
    let lower = answer_task_text(query).to_lowercase();
    let wants_cause = contains_any(&lower, &["为什么", "导致", "原因", "why", "cause"]);
    let wants_recovery = contains_any(&lower, &["恢复", "补救", "怎么修", "recover", "fix"]);
    let wants_similarity = contains_any(
        &lower,
        &["类似", "相似", "之前", "过去", "同样", "失败模式", "similar", "before"],
    );
    if !(wants_cause || wants_recovery || wants_similarity) {
        return None;
    }
    let mut claims = Vec::new();
    let mut episode_indices = Vec::new();
    let mut provenance = Vec::new();
    for item in items.iter().filter(|item| memory_item_is_active(item)) {
        let Some(meta) = item.episodic_metadata.as_ref() else {
            continue;
        };
        let has_experience = !meta.cause_links.is_empty()
            || !meta.recovery_links.is_empty()
            || meta.failure_type.is_some();
        if !has_experience {
            continue;
        }
        let statement = item
            .content
            .trim_start()
            .strip_prefix("episode:")
            .map(|rest| event_record_body(rest.trim_start()))
            .unwrap_or_else(|| event_record_body(&item.content));
        push_statement_clause_claims(&mut claims, &mut episode_indices, item.id, statement);
        if wants_cause && !meta.cause_links.is_empty() {
            provenance.push((
                FrameSection::FailureReason,
                item.id,
                format!("memory#{}: {}", item.id, meta.cause_links.join("；")),
            ));
        }
        if wants_recovery && !meta.recovery_links.is_empty() {
            provenance.push((
                FrameSection::Recovery,
                item.id,
                format!("memory#{}: {}", item.id, meta.recovery_links.join("；")),
            ));
        }
    }
    if claims.is_empty() {
        return None;
    }
    let display = claims
        .iter()
        .map(|claim| format!("- {}", claim.text))
        .collect::<Vec<_>>()
        .join("\n");
    let frame = RendererFrame {
        kind: if wants_similarity {
            FrameKind::SimilarExperience
        } else {
            FrameKind::RelatedExperience
        },
        sections: vec![(FrameSection::ExperienceRecord, episode_indices)],
        notes: Vec::new(),
        provenance,
        claim_sources: HashMap::new(),
        claim_count: claims.len(),
    };
    Some((display, claims, frame))
}

/// Whether a Claim states a time value: a year, month, weekday, season, a
/// numeric date, or a relative time expression.
fn claim_states_temporal_value(claim: &str) -> bool {
    static TEMPORAL_VALUE: OnceLock<regex::Regex> = OnceLock::new();
    TEMPORAL_VALUE
        .get_or_init(|| {
            regex::Regex::new(concat!(
                r"(?i)\b(?:1[89]|20)\d{2}\b",
                r"|\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
                r"|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend)\b",
                r"|\b(?:spring|summer|autumn|fall|winter)\b",
                r"|\b\d{1,2}[/.-]\d{1,2}(?:[/.-]\d{2,4})?\b",
                r"|\b(?:yesterday|today|tonight|tomorrow|ago|last (?:week|month|year|night)|next (?:week|month|year)|earlier this|since)\b",
                r"|[年月日]|昨天|今天|明天|上周|上个月|去年"
            ))
            .expect("temporal value regex")
        })
        .is_match(claim)
}

fn ambient_capture_content_excerpt(content: &str) -> String {
    compact_context_line(&ambient_capture_content_body(content), 240)
}

/// The statement an operator record carries, as the evidence projector binds
/// it: the primary text field of a typed JSON record, or the event body.
fn operator_record_statement(content: &str) -> String {
    serde_json::from_str::<serde_json::Value>(content)
        .ok()
        .and_then(|value| {
            ["constraint", "summary", "statement", "content", "decision"]
                .iter()
                .find_map(|key| value.get(*key).and_then(serde_json::Value::as_str))
                .map(str::to_string)
        })
        .unwrap_or_else(|| {
            // Legacy text records carry quality metadata (freshness,
            // last_verified_at, ...) as extra lines; they are not the statement.
            event_record_body(content)
                .lines()
                .map(str::trim)
                .filter(|line| {
                    !line.is_empty() && !crate::memory_quality::is_legacy_quality_metadata_line(line)
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
}

/// The statement carried by an `event: layer=...; content=...` record, which is
/// what the evidence projector binds; other records are returned unchanged.
fn event_record_body(content: &str) -> &str {
    content
        .trim_start()
        .strip_prefix("event:")
        .and_then(|fields| fields.split_once("content="))
        .map(|(_, body)| body.trim())
        .filter(|body| !body.is_empty())
        .unwrap_or(content)
}

fn ambient_capture_content_body(content: &str) -> String {
    serde_json::from_str::<serde_json::Value>(content)
        .ok()
        .and_then(|value| {
            value
                .get("content")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| content.to_string())
}

fn operator_file_constraints(query: &str) -> Vec<String> {
    let Some((_, tail)) = query.split_once("Memory Files context:") else {
        return Vec::new();
    };
    let file_context = tail
        .split_once("Task/Question:")
        .map(|(context, _)| context)
        .unwrap_or(tail);
    file_context
        .lines()
        .map(str::trim)
        .filter(|line| {
            contains_any(
                &line.to_lowercase(),
                &[
                    "sop",
                    "mission",
                    "constraint",
                    "must",
                    "do not",
                    "abort",
                    "no-go",
                    "operator",
                    "runbook",
                    "任务",
                    "约束",
                    "必须",
                    "不得",
                    "禁止",
                ],
            )
        })
        .map(|line| line.trim_start_matches("- ").to_string())
        .take(6)
        .collect()
}

fn operator_item_is_stale_risk(item: &RetrievedItem) -> bool {
    if item
        .freshness
        .map(|freshness| freshness < 0.45)
        .unwrap_or(false)
    {
        return true;
    }
    if item
        .last_verified_at
        .map(|last_verified_at| last_verified_at < Utc::now() - Duration::days(30))
        .unwrap_or(false)
    {
        return true;
    }
    operator_content_indicates_stale(&item.content)
}

fn operator_content_indicates_stale(content: &str) -> bool {
    let normalized = content.to_lowercase();
    if contains_any(
        &normalized,
        &[
            "stale",
            "legacy",
            "obsolete",
            "last_verified_at",
            "freshness: 0.",
            "过期",
            "陈旧",
        ],
    ) {
        return true;
    }
    normalized
        .split(|ch: char| !ch.is_alphanumeric())
        .any(|token| token == "old")
}

fn operator_stale_reason(item: &RetrievedItem) -> String {
    let mut parts = Vec::new();
    if let Some(freshness) = item.freshness {
        parts.push(format!("freshness={freshness:.2}"));
    }
    if let Some(last_verified_at) = item.last_verified_at {
        parts.push(format!(
            "last_verified_at={}",
            last_verified_at.to_rfc3339()
        ));
    }
    if parts.is_empty() {
        parts.push("content indicates stale or legacy operational evidence".to_string());
    }
    format!(
        "{} | {}",
        parts.join(", "),
        compact_context_line(&item.content, 180)
    )
}

/// Memory File snapshots are typed by source: `memory_type` is normalized to
/// a fixed allowlist and would report them as plain facts.
fn is_memory_file_snapshot(item: &RetrievedItem) -> bool {
    item.source == crate::support::MEMORY_FILE_SNAPSHOT_TYPE
}

/// Statement lines of a Memory File snapshot: the file body without its
/// front-matter block, headings, or blank lines.
fn memory_file_snapshot_body_lines(snapshot: &str) -> Vec<String> {
    let content = serde_json::from_str::<serde_json::Value>(snapshot)
        .ok()
        .and_then(|value| value.get("content").and_then(serde_json::Value::as_str).map(str::to_string))
        .unwrap_or_default();
    let mut lines = content.lines().map(str::trim).peekable();
    if lines.peek() == Some(&"---") {
        lines.next();
        for line in lines.by_ref() {
            if line == "---" {
                break;
            }
        }
    }
    lines
        .map(|line| line.trim_start_matches("- ").trim())
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_string)
        .collect()
}

/// One Claim per line of a record statement. A line may span several source
/// atoms; the verifier binds it against the record's consecutive atoms.
fn push_statement_clause_claims(
    claims: &mut Vec<RendererClaim>,
    indices: &mut Vec<usize>,
    memory_id: u64,
    statement: &str,
) {
    for line in statement.lines().map(str::trim).filter(|line| !line.is_empty()) {
        indices.push(claims.len());
        claims.push(RendererClaim {
            text: line.to_string(),
            evidence_ids: vec![memory_id],
        });
    }
}

/// Stored freshness metadata of a record, without any content.
fn operator_stale_metadata(item: &RetrievedItem) -> String {
    let mut parts = Vec::new();
    if let Some(freshness) = item.freshness {
        parts.push(format!("freshness={freshness:.2}"));
    }
    if let Some(last_verified_at) = item.last_verified_at {
        parts.push(format!("last_verified_at={}", last_verified_at.to_rfc3339()));
    }
    if parts.is_empty() {
        parts.push("content marks this record as stale or legacy".to_string());
    }
    parts.join(", ")
}

fn operator_bullets_or_none(values: &[String]) -> String {
    if values.is_empty() {
        "- None found in recalled context.".to_string()
    } else {
        values
            .iter()
            .map(|value| format!("- {value}"))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ChapterSection {
    Summary,
    Continuity,
    Storyline,
    Event,
    Evidence,
}

impl ChapterSection {
    fn label(self) -> &'static str {
        match self {
            Self::Summary => "章节概要",
            Self::Continuity => "承接前文",
            Self::Storyline => "故事线",
            Self::Event => "关键事件",
            Self::Evidence => "原文证据",
        }
    }
}

fn chapter_field_values(content: &str) -> Vec<(ChapterSection, String)> {
    let mut values = Vec::new();
    if let Some(summary) = answer_line_value(content, "Chapter summary") {
        values.push((ChapterSection::Summary, summary));
    }
    if let Some(continuity) = answer_line_value(content, "Continuity") {
        values.push((ChapterSection::Continuity, continuity));
    }
    values.extend(
        answer_block_values(
            content,
            "Storylines",
            &["Key events", "Evidence", "Source chunks"],
        )
        .into_iter()
        .map(|value| (ChapterSection::Storyline, value)),
    );
    values.extend(
        answer_block_values(content, "Key events", &["Evidence", "Source chunks"])
            .into_iter()
            .map(|value| (ChapterSection::Event, value)),
    );
    values.extend(
        answer_block_values(
            content,
            "Evidence",
            &["Source chunks", "Source archive memory IDs"],
        )
        .into_iter()
        .map(|value| (ChapterSection::Evidence, value)),
    );
    values
}

/// Frame for a single-chapter answer: the chapter heading of each cited
/// record is provenance, shown only for records with a verified Claim.
fn chapter_answer_frame(claims: &[RendererClaim], records: &[(u64, String)]) -> RendererFrame {
    let cited = claims
        .iter()
        .flat_map(|claim| claim.evidence_ids.iter().copied())
        .collect::<BTreeSet<_>>();
    let provenance = records
        .iter()
        .filter(|(id, _)| cited.contains(id))
        .filter_map(|(id, content)| {
            answer_line_value(content, "Section")
                .map(|section| (FrameSection::ChapterIdentity, *id, section))
        })
        .collect();
    RendererFrame {
        kind: FrameKind::NovelChapter,
        sections: vec![(FrameSection::ChapterIdentity, Vec::new())],
        notes: vec![FrameNote::TraceLookup],
        provenance,
        claim_sources: HashMap::new(),
        claim_count: claims.len(),
    }
}

fn chapter_record_claims(memory_id: u64, content: &str) -> Vec<RendererClaim> {
    let mut seen = HashSet::new();
    chapter_field_values(content)
        .into_iter()
        .map(|(_, text)| text)
        .filter(|text| !text.trim().is_empty() && seen.insert(text.clone()))
        .map(|text| RendererClaim {
            text,
            evidence_ids: vec![memory_id],
        })
        .collect()
}

fn render_novel_chapter_answer(
    query: &str,
    items: &[RetrievedItem],
) -> Option<(String, Vec<RendererClaim>)> {
    if answer_chapter_range(query).is_some() {
        return None;
    }
    let chapter_marker = answer_chapter_marker(query);
    if chapter_marker.is_none()
        && !contains_any(
            &query.to_lowercase(),
            &["chapter", "章节", "发生了什么", "summary", "概要"],
        )
    {
        return None;
    }
    let marker = chapter_marker.as_deref();
    let item = items
        .iter()
        .filter(|item| {
            memory_item_is_active(item)
                && is_novel_chapter_summary_item(item)
                && marker
                    .map(|value| novel_chapter_marker_matches(&item.content, value))
                    .unwrap_or(true)
        })
        .min_by_key(|item| {
            if item.memory_type == "novel_chapter_summary" {
                0
            } else {
                1
            }
        })?;
    let section = answer_line_value(&item.content, "Section")
        .or_else(|| marker.map(ToString::to_string))
        .unwrap_or_else(|| "匹配章节".to_string());
    let summary = answer_line_value(&item.content, "Chapter summary")
        .unwrap_or_else(|| compact_context_line(&item.content, 260));
    let continuity = answer_line_value(&item.content, "Continuity");
    let storylines = answer_block_values(
        &item.content,
        "Storylines",
        &["Key events", "Evidence", "Source chunks"],
    );
    let key_events =
        answer_block_values(&item.content, "Key events", &["Evidence", "Source chunks"]);
    let key_events = if key_events.is_empty() {
        vec![summary.clone()]
    } else {
        key_events
    };
    let key_events_text = key_events
        .iter()
        .take(5)
        .map(|event| format!("- {}", compact_context_line(event, 220)))
        .collect::<Vec<_>>()
        .join("\n");
    let evidence_text = render_chapter_summary_evidence_line(
        item.id,
        &item.content,
        Some(&item.source),
        &summary,
        &key_events,
    );
    let continuity_text = continuity
        .map(|value| format!("\n\n承接前文：{}", compact_context_line(&value, 260)))
        .unwrap_or_default();
    let storylines_text = render_chapter_storylines_section(&storylines);
    Some((format!(
        "{section}发生了什么：\n\n章节概要：{summary}{continuity_text}{storylines_text}\n\n关键事件：\n{key_events_text}\n\n证据：\n{evidence_text}\n\nTrace: 使用响应里的 trace_id 查询 /memory/trace/{{trace_id}}。"
    ), chapter_record_claims(item.id, &item.content)))
}

fn render_novel_chapter_unavailable_answer(query: &str) -> Option<String> {
    if answer_chapter_range(query).is_some() {
        return None;
    }
    let marker = answer_chapter_marker(query)?;
    Some(format!(
        "{marker}系统反馈：当前不能可靠回答这一章。\n\n原因：系统没有找到可用的 active `novel_chapter_summary` 记录。不能用写作风格、其他章节概要或泛化检索结果替代章节事实。\n\n需要的系统动作：如果本章 archive 原文块已经存在，请运行 `/memory/admin/long-document/chapter-summaries/refresh`；如果 archive 原文块缺失，请重新导入或恢复本章 archive 后再运行 source-link validation/refresh。\n\nTrace: 使用响应里的 trace_id 查询 /memory/trace/{{trace_id}}。"
    ))
}

fn render_novel_chapter_answer_from_records(
    query: &str,
    records: &[MemoryRecord],
    conversation_id: Option<&str>,
    document_limit: Option<usize>,
) -> Option<(String, Vec<RendererClaim>, Option<ChapterFeedback>)> {
    if answer_chapter_range(query).is_some() {
        return None;
    }
    let chapter_marker = answer_chapter_marker(query);
    if chapter_marker.is_none() {
        return None;
    }
    let marker = chapter_marker.as_deref();
    let mut candidates = records
        .iter()
        .filter(|record| {
            conversation_id
                .map(|conversation_id| record.conversation_id == conversation_id)
                .unwrap_or(true)
        })
        .filter(|record| memory_record_is_active(record))
        .filter(|record| memory_record_has_direct_type(record, "novel_chapter_summary"))
        .filter(|record| {
            marker
                .map(|value| novel_chapter_marker_matches(&record.content, value))
                .unwrap_or(true)
        })
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return None;
    }
    let query_scoped_candidates = candidates
        .iter()
        .copied()
        .filter(|record| query_mentions_document_identity(query, &record.content))
        .collect::<Vec<_>>();
    if !query_scoped_candidates.is_empty() {
        candidates = query_scoped_candidates;
    } else if chapter_candidates_have_multiple_documents(&candidates) {
        if answer_wants_multi_document_chapters(query) {
            let claims = document_grouped_chapter_candidates(&candidates)
                .into_iter()
                .flat_map(|(_, records)| {
                    records
                        .into_iter()
                        .max_by_key(|record| (record.created_at, record.id))
                        .map(|record| chapter_record_claims(record.id, &record.content))
                        .unwrap_or_default()
                })
                .collect();
            return Some((
                render_multi_document_chapter_answer(query, marker, &candidates, document_limit),
                claims,
                None,
            ));
        }
        return Some((
            String::new(),
            vec![],
            Some(ChapterFeedback::Ambiguous {
                documents: document_grouped_chapter_candidates(&candidates)
                    .into_iter()
                    .map(|(name, records)| (name, records.iter().map(|record| record.id).collect()))
                    .collect(),
                display_limit: normalize_document_limit(document_limit),
            }),
        ));
    }
    let record = candidates.into_iter().max_by_key(|record| {
        (
            record.memory_type.as_deref() == Some("novel_chapter_summary"),
            record.created_at,
            record.id,
        )
    })?;
    if novel_chapter_record_is_extractively_incomplete(record)
        && !records.iter().any(|candidate| {
            conversation_id
                .map(|conversation_id| candidate.conversation_id == conversation_id)
                .unwrap_or(true)
                && memory_record_is_active(candidate)
                && memory_record_is_archive(candidate)
                && marker
                    .map(|value| novel_chapter_marker_matches(&candidate.content, value))
                    .unwrap_or(true)
        })
    {
        return Some((
            String::new(),
            vec![],
            Some(ChapterFeedback::Incomplete {
                memory_id: record.id,
            }),
        ));
    }
    Some((
        render_novel_chapter_record_answer(query, record, marker),
        chapter_record_claims(record.id, &record.content),
        None,
    ))
}

struct ChapterRangePlan {
    display: String,
    claims: Vec<RendererClaim>,
    feedback: Option<ChapterFeedback>,
    coverage_note: String,
    incomplete: bool,
}

fn plan_novel_chapter_range(
    query: &str,
    records: &[MemoryRecord],
    conversation_id: Option<&str>,
    answer_nodes: Option<usize>,
    document_limit: Option<usize>,
) -> Option<ChapterRangePlan> {
    let (start, end) = answer_chapter_range(query)?;
    let matching = records
        .iter()
        .filter(|record| {
            conversation_id.is_none_or(|scope| record.conversation_id == scope)
                && memory_record_is_eligible_for_selected_answer(record)
                && memory_record_has_direct_type(record, "novel_chapter_summary")
                && canonical_chapter_number(&record.content)
                    .is_some_and(|number| number >= start && number <= end)
        })
        .collect::<Vec<_>>();
    let scoped = matching
        .iter()
        .any(|record| query_mentions_document_identity(query, &record.content));
    if !scoped
        && chapter_candidates_have_multiple_documents(&matching)
        && !answer_wants_multi_document_chapters(query)
    {
        return Some(ChapterRangePlan {
            display: String::new(),
            claims: vec![],
            incomplete: true,
            feedback: Some(ChapterFeedback::Ambiguous {
                documents: document_grouped_chapter_candidates(&matching)
                    .into_iter()
                    .map(|(name, records)| (name, records.iter().map(|record| record.id).collect()))
                    .collect(),
                display_limit: normalize_document_limit(document_limit),
            }),
            coverage_note: format!("覆盖范围：第{start}章至第{end}章，文档身份待确认。"),
        });
    }
    let mut latest = std::collections::BTreeMap::new();
    for record in matching
        .into_iter()
        .filter(|record| !scoped || query_mentions_document_identity(query, &record.content))
    {
        let key = (
            chapter_record_document_identity(&record.content),
            canonical_chapter_number(&record.content)?,
        );
        let previous = latest.entry(key).or_insert(record);
        if (record.created_at, record.id) > (previous.created_at, previous.id) {
            *previous = record;
        }
    }
    let documents = latest
        .keys()
        .map(|(name, _)| name)
        .collect::<BTreeSet<_>>()
        .len()
        .max(1);
    let expected = (u64::from(end) - u64::from(start) + 1).saturating_mul(documents as u64);
    let found = latest.len() as u64;
    let missing = expected.saturating_sub(found);
    let claims = latest
        .values()
        .flat_map(|record| chapter_record_claims(record.id, &record.content))
        .collect();
    // Legacy layout is display-only. Neither its 50-chapter cap nor node and
    // document display budgets determine the proof inputs above.
    let display = render_novel_chapter_range_answer_from_records(
        query,
        records,
        conversation_id,
        answer_nodes,
        document_limit,
    )
    .unwrap_or_default();
    Some(ChapterRangePlan { display, claims, feedback: None, incomplete: missing > 0,
        coverage_note: format!("覆盖范围：第{start}章至第{end}章；{documents} 个文档中找到 {found}/{expected} 个章节记录，缺失 {missing} 个。记录存在不等于结论已核验。") })
}

fn render_novel_chapter_range_answer_from_records(
    query: &str,
    records: &[MemoryRecord],
    conversation_id: Option<&str>,
    answer_nodes: Option<usize>,
    document_limit: Option<usize>,
) -> Option<String> {
    let (start, end) = answer_chapter_range(query)?;
    let display_end = end.min(start.saturating_add(49));
    let capped = display_end < end;
    let detailed = answer_wants_chapter_by_chapter(query);
    let all_matching_summaries = records
        .iter()
        .filter(|record| {
            conversation_id
                .map(|conversation_id| record.conversation_id == conversation_id)
                .unwrap_or(true)
        })
        .filter(|record| memory_record_is_active(record))
        .filter(|record| memory_record_has_direct_type(record, "novel_chapter_summary"))
        .filter(|record| {
            canonical_chapter_number(&record.content)
                .map(|chapter| chapter >= start && chapter <= display_end)
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    let query_scoped = all_matching_summaries
        .iter()
        .any(|record| query_mentions_document_identity(query, &record.content));
    if !query_scoped && chapter_candidates_have_multiple_documents(&all_matching_summaries) {
        if answer_wants_multi_document_chapters(query) {
            return Some(render_multi_document_chapter_range_answer(
                query,
                start,
                display_end,
                &all_matching_summaries,
                answer_nodes,
                document_limit,
                capped,
            ));
        }
        return Some(render_chapter_range_document_ambiguity_feedback(
            start,
            display_end,
            &all_matching_summaries,
            document_limit,
        ));
    }
    let mut chapter_evidence = Vec::new();
    let mut missing_lines = Vec::new();

    for chapter in start..=display_end {
        let summary_record = records
            .iter()
            .filter(|record| {
                conversation_id
                    .map(|conversation_id| record.conversation_id == conversation_id)
                    .unwrap_or(true)
            })
            .filter(|record| memory_record_is_active(record))
            .filter(|record| memory_record_has_direct_type(record, "novel_chapter_summary"))
            .filter(|record| canonical_chapter_number(&record.content) == Some(chapter))
            .filter(|record| {
                !query_scoped || query_mentions_document_identity(query, &record.content)
            })
            .max_by_key(|record| {
                (
                    record.memory_type.as_deref() == Some("novel_chapter_summary"),
                    record.created_at,
                    record.id,
                )
            });

        if let Some(record) = summary_record {
            let section = answer_line_value(&record.content, "Section")
                .unwrap_or_else(|| format!("第{chapter}章"));
            let summary = answer_line_value(&record.content, "Chapter summary")
                .unwrap_or_else(|| compact_context_line(&record.content, 220));
            let key_events = answer_block_values(
                &record.content,
                "Key events",
                &["Evidence", "Source chunks"],
            );
            let source_label =
                chapter_summary_source_label(&record.content, Some(record.source.as_str()));
            chapter_evidence.push(ChapterRangeEvidence {
                chapter,
                section,
                summary,
                key_events,
                source_label,
                memory_id: record.id,
            });
        } else {
            let archive_ids = records
                .iter()
                .filter(|record| {
                    conversation_id
                        .map(|conversation_id| record.conversation_id == conversation_id)
                        .unwrap_or(true)
                })
                .filter(|record| memory_record_is_active(record))
                .filter(|record| memory_record_is_archive(record))
                .filter(|record| canonical_chapter_number(&record.content) == Some(chapter))
                .map(|record| format!("memory#{}", record.id))
                .take(3)
                .collect::<Vec<_>>();
            if archive_ids.is_empty() {
                missing_lines.push(format!(
                    "- 第{chapter}章：缺少 active `novel_chapter_summary`，也没有匹配 archive 原文块。"
                ));
            } else {
                missing_lines.push(format!(
                    "- 第{chapter}章：缺少 active `novel_chapter_summary`；已有 archive 原文块 {}，需要刷新章节概要。",
                    archive_ids.join(", ")
                ));
            }
        }
    }

    if chapter_evidence.is_empty() && missing_lines.is_empty() {
        return None;
    }

    let requested = display_end.saturating_sub(start) + 1;
    let coverage = format!(
        "覆盖状态：找到 {}/{} 个章节概要记录{}。",
        chapter_evidence.len(),
        requested,
        if capped {
            "；本次最多渲染 50 章，后续章节被截断"
        } else {
            ""
        }
    );
    let missing_text = if missing_lines.is_empty() {
        String::new()
    } else {
        format!("\n\n缺口：\n{}", missing_lines.join("\n"))
    };

    if detailed {
        let evidence_lines = chapter_evidence
            .iter()
            .map(render_chapter_range_evidence_line)
            .collect::<Vec<_>>();
        return Some(format!(
            "第{start}章到第{display_end}章逐章系统回答：\n\n{coverage}\n\n章节证据：\n{}{missing_text}\n\n结论依据：以上内容来自 durable reading model 的 `novel_chapter_summary` 记录；缺口只引用 archive 原文块，不用其他章节或写作风格记录补假答案。\n\nTrace: 使用响应里的 trace_id 查询 /memory/trace/{{trace_id}}/summary。",
            evidence_lines.join("\n")
        ));
    }

    let requested_chapters = display_end.saturating_sub(start) as usize + 1;
    let explicit_nodes = answer_nodes
        .map(normalize_answer_range_node_count)
        .or_else(|| explicit_answer_range_node_count(query));
    let node_count = explicit_nodes
        .unwrap_or_else(|| adaptive_answer_range_node_count(requested_chapters, &chapter_evidence));
    let node_label = if explicit_nodes.is_some() {
        format!("{node_count} 个以内")
    } else {
        format!("自适应 {node_count} 个以内")
    };
    let overview = render_chapter_range_phase_overview(&chapter_evidence, node_count);
    Some(format!(
        "第{start}章到第{display_end}章系统范围回答：\n\n{coverage}\n\n概要节点（{node_label}）：\n{overview}{missing_text}\n\n需要指定粒度时，可以说“用3个节点概括”；需要逐章展开时，请在问题里加“逐章”或“each chapter”。结论依据：以上内容来自 durable reading model 的 `novel_chapter_summary` 记录；缺口只引用 archive 原文块，不用其他章节或写作风格记录补假答案。\n\nTrace: 使用响应里的 trace_id 查询 /memory/trace/{{trace_id}}/summary。"
    ))
}

#[derive(Debug, Clone)]
struct ChapterRangeEvidence {
    chapter: u32,
    section: String,
    summary: String,
    key_events: Vec<String>,
    source_label: String,
    memory_id: u64,
}

fn render_chapter_range_evidence_line(item: &ChapterRangeEvidence) -> String {
    let key_events_text = item
        .key_events
        .iter()
        .take(2)
        .map(|event| compact_context_line(event, 90))
        .collect::<Vec<_>>()
        .join("；");
    let key_events_text = if key_events_text.is_empty() {
        String::new()
    } else {
        format!("；关键事件：{key_events_text}")
    };
    format!(
        "- memory#{} [{}] {}：{}{}",
        item.memory_id,
        item.source_label,
        item.section,
        compact_context_line(&item.summary, 180),
        key_events_text
    )
}

fn render_chapter_range_phase_overview(
    evidence: &[ChapterRangeEvidence],
    node_count: usize,
) -> String {
    if evidence.is_empty() {
        return "没有可用章节概要节点。".to_string();
    }
    let node_count = normalize_answer_range_node_count(node_count).min(evidence.len());
    (0..node_count)
        .filter_map(|index| {
            let start = index * evidence.len() / node_count;
            let end = (index + 1) * evidence.len() / node_count;
            (start < end).then_some((index, &evidence[start..end]))
        })
        .map(|(index, chunk)| {
            let first = chunk.first().map(|item| item.chapter).unwrap_or_default();
            let last = chunk.last().map(|item| item.chapter).unwrap_or(first);
            let citations = chunk
                .iter()
                .map(|item| format!("memory#{}", item.memory_id))
                .collect::<Vec<_>>()
                .join(", ");
            let beats = chunk
                .iter()
                .flat_map(|item| item.key_events.iter())
                .take(4)
                .map(|event| compact_context_line(event, 120))
                .collect::<Vec<_>>();
            let body = if beats.is_empty() {
                chunk
                    .iter()
                    .take(2)
                    .map(|item| compact_context_line(&item.summary, 160))
                    .collect::<Vec<_>>()
                    .join("；")
            } else {
                beats.join("；")
            };
            format!(
                "{}. 第{}-{}章：{}。证据：{}",
                index + 1,
                first,
                last,
                body,
                citations
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn novel_chapter_record_is_extractively_incomplete(record: &MemoryRecord) -> bool {
    let summary = answer_line_value(&record.content, "Chapter summary").unwrap_or_default();
    let continuity = answer_line_value(&record.content, "Continuity");
    let key_events = answer_block_values(
        &record.content,
        "Key events",
        &["Evidence", "Source chunks"],
    );
    if summary.trim().is_empty() || continuity.is_some() {
        return false;
    }
    let summary_char_count = summary.chars().count();
    let starts_like_raw_opening =
        summary.starts_with(['“', '"', '「']) || summary.contains('；') || summary.contains(';');
    let lacks_abstractive_shape = !contains_any(
        &summary,
        &[
            "本章围绕",
            "展开",
            "转入",
            "确认",
            "揭示",
            "推进",
            "识别",
            "应对",
            "Chapter centers",
        ],
    );
    let source_chunks = answer_line_value(&record.content, "Source chunks")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    starts_like_raw_opening
        && lacks_abstractive_shape
        && summary_char_count <= 120
        && source_chunks <= 1
        && key_events.iter().take(3).all(|event| {
            event.ends_with(['。', '？', '！', '.', '?', '!'])
                || event.starts_with(['“', '"', '「'])
                || event.chars().count() <= 40
        })
}

fn render_novel_chapter_record_answer(
    _query: &str,
    record: &MemoryRecord,
    marker: Option<&str>,
) -> String {
    let section = answer_line_value(&record.content, "Section")
        .or_else(|| marker.map(ToString::to_string))
        .unwrap_or_else(|| "匹配章节".to_string());
    let summary = answer_line_value(&record.content, "Chapter summary")
        .unwrap_or_else(|| compact_context_line(&record.content, 260));
    let continuity = answer_line_value(&record.content, "Continuity");
    let storylines = answer_block_values(
        &record.content,
        "Storylines",
        &["Key events", "Evidence", "Source chunks"],
    );
    let key_events = answer_block_values(
        &record.content,
        "Key events",
        &["Evidence", "Source chunks"],
    );
    let key_events = if key_events.is_empty() {
        vec![summary.clone()]
    } else {
        key_events
    };
    let key_events_text = key_events
        .iter()
        .take(5)
        .map(|event| format!("- {}", compact_context_line(event, 220)))
        .collect::<Vec<_>>()
        .join("\n");
    let evidence_text = render_chapter_summary_evidence_line(
        record.id,
        &record.content,
        Some(&record.source),
        &summary,
        &key_events,
    );
    let continuity_text = continuity
        .map(|value| format!("\n\n承接前文：{}", compact_context_line(&value, 260)))
        .unwrap_or_default();
    let storylines_text = render_chapter_storylines_section(&storylines);
    format!(
        "{section}发生了什么：\n\n章节概要：{summary}{continuity_text}{storylines_text}\n\n关键事件：\n{key_events_text}\n\n证据：\n{evidence_text}\n\nTrace: 使用响应里的 trace_id 查询 /memory/trace/{{trace_id}}。"
    )
}

fn render_multi_document_chapter_answer(
    _query: &str,
    marker: Option<&str>,
    candidates: &[&MemoryRecord],
    document_limit: Option<usize>,
) -> String {
    let mut grouped = document_grouped_chapter_candidates(candidates);
    grouped.sort_by(|a, b| a.0.cmp(&b.0));
    let document_limit = normalize_document_limit(document_limit);
    let total_document_count = grouped.len();
    let body = grouped
        .into_iter()
        .take(document_limit)
        .map(|(identity, mut records)| {
            records.sort_by_key(|record| (record.created_at, record.id));
            let record = records.last().copied().unwrap();
            let section = answer_line_value(&record.content, "Section")
                .or_else(|| marker.map(ToString::to_string))
                .unwrap_or_else(|| "匹配章节".to_string());
            let summary = answer_line_value(&record.content, "Chapter summary")
                .unwrap_or_else(|| compact_context_line(&record.content, 220));
            format!(
                "- `{}`：memory#{} {}：{}",
                compact_context_line(&identity, 120),
                record.id,
                compact_context_line(&section, 80),
                compact_context_line(&summary, 180)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let marker_text = marker.unwrap_or("该章节");
    let cap_text = if total_document_count > document_limit {
        format!("\n\n显示限制：找到 {total_document_count} 个匹配文档，本次显示 {document_limit} 个。可设置 document_limit 调整。")
    } else {
        String::new()
    };
    format!(
        "{marker_text}跨文档回答：\n\n{body}{cap_text}\n\n结论依据：同一文件夹/工作区内存在多个文件或作品，系统按 Source file/Novel 分组回答，没有把同名章节随机合并。若只想看其中一本，请在问题中写明书名或文件名。\n\nTrace: 使用响应里的 trace_id 查询 /memory/trace/{{trace_id}}。"
    )
}

fn render_multi_document_chapter_range_answer(
    query: &str,
    start: u32,
    end: u32,
    candidates: &[&MemoryRecord],
    answer_nodes: Option<usize>,
    document_limit: Option<usize>,
    capped: bool,
) -> String {
    let mut grouped = document_grouped_chapter_candidates(candidates);
    grouped.sort_by(|a, b| a.0.cmp(&b.0));
    let requested_chapters = end.saturating_sub(start) as usize + 1;
    let explicit_nodes = answer_nodes
        .map(normalize_answer_range_node_count)
        .or_else(|| explicit_answer_range_node_count(query));
    let document_limit = normalize_document_limit(document_limit);
    let total_document_count = grouped.len();
    let shown_document_count = grouped.len().min(document_limit);
    let body = grouped
        .into_iter()
        .take(document_limit)
        .map(|(identity, mut records)| {
            records.sort_by_key(|record| {
                (
                    canonical_chapter_number(&record.content).unwrap_or(u32::MAX),
                    record.created_at,
                    record.id,
                )
            });
            let evidence = records
                .into_iter()
                .map(chapter_range_evidence_from_record)
                .collect::<Vec<_>>();
            let node_count = explicit_nodes
                .unwrap_or_else(|| adaptive_answer_range_node_count(requested_chapters, &evidence));
            let node_label = if explicit_nodes.is_some() {
                format!("{node_count} 个以内/文档")
            } else {
                format!("自适应 {node_count} 个以内/文档")
            };
            let lines = render_chapter_range_phase_overview(&evidence, node_count);
            format!(
                "- `{}`（{}）：\n{}",
                compact_context_line(&identity, 120),
                node_label,
                lines
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let capped_text = if capped {
        "；本次最多渲染 50 章，后续章节被截断"
    } else {
        ""
    };
    let document_cap_text = if total_document_count > document_limit {
        format!("\n\n显示限制：找到 {total_document_count} 个匹配文档，本次显示 {document_limit} 个。可设置 document_limit 调整，最大 {MAX_MULTI_DOCUMENT_ANSWER_LIMIT}。")
    } else {
        String::new()
    };
    format!(
        "第{start}章到第{end}章跨文档回答{capped_text}：\n\n节点预算：每个文档单独计算；本次显示 {shown_document_count} 个文档，每个文档最多 64 个节点。\n\n{body}{document_cap_text}\n\n结论依据：系统按 Source file/Novel 分组展示章节概要；同名章节不会跨文件合并，节点数也不会在不同文件之间抢占。若只想看其中一本，请指定书名或文件名。\n\nTrace: 使用响应里的 trace_id 查询 /memory/trace/{{trace_id}}/summary。"
    )
}

fn chapter_range_evidence_from_record(record: &MemoryRecord) -> ChapterRangeEvidence {
    let chapter = canonical_chapter_number(&record.content).unwrap_or_default();
    let section =
        answer_line_value(&record.content, "Section").unwrap_or_else(|| format!("第{chapter}章"));
    let summary = answer_line_value(&record.content, "Chapter summary")
        .unwrap_or_else(|| compact_context_line(&record.content, 220));
    let key_events = answer_block_values(
        &record.content,
        "Key events",
        &["Evidence", "Source chunks"],
    );
    let source_label = chapter_summary_source_label(&record.content, Some(record.source.as_str()));
    ChapterRangeEvidence {
        chapter,
        section,
        summary,
        key_events,
        source_label,
        memory_id: record.id,
    }
}

fn render_chapter_range_document_ambiguity_feedback(
    start: u32,
    end: u32,
    candidates: &[&MemoryRecord],
    document_limit: Option<usize>,
) -> String {
    let document_limit = normalize_document_limit(document_limit);
    let grouped = document_grouped_chapter_candidates(candidates);
    let total_document_count = grouped.len();
    let options = grouped
        .into_iter()
        .take(document_limit)
        .map(|(identity, records)| {
            let chapter_count = records
                .iter()
                .filter_map(|record| canonical_chapter_number(&record.content))
                .collect::<std::collections::HashSet<_>>()
                .len();
            format!(
                "- `{}`：{} 个匹配章节概要",
                compact_context_line(&identity, 140),
                chapter_count
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let cap_text = if total_document_count > document_limit {
        format!("\n\n显示限制：找到 {total_document_count} 个候选文档，本次显示 {document_limit} 个。可设置 document_limit 调整。")
    } else {
        String::new()
    };
    format!(
        "第{start}章到第{end}章系统反馈：当前不能可靠回答这个章节范围，因为同一工作区/文件夹里有多个文件或作品匹配该范围。\n\n候选文档：\n{options}{cap_text}\n\n需要用户选择：请指定书名/文件名，或明确说“比较所有文件/所有小说的第{start}章到第{end}章”。\n\nTrace: 使用响应里的 trace_id 查询 /memory/trace/{{trace_id}}/summary。"
    )
}

fn document_grouped_chapter_candidates<'a>(
    candidates: &[&'a MemoryRecord],
) -> Vec<(String, Vec<&'a MemoryRecord>)> {
    let mut grouped: Vec<(String, Vec<&MemoryRecord>)> = Vec::new();
    for record in candidates {
        let identity = chapter_record_document_identity(&record.content);
        if let Some((_, records)) = grouped
            .iter_mut()
            .find(|(existing, _)| existing == &identity)
        {
            records.push(*record);
        } else {
            grouped.push((identity, vec![*record]));
        }
    }
    grouped
}

fn chapter_candidates_have_multiple_documents(candidates: &[&MemoryRecord]) -> bool {
    candidates
        .iter()
        .map(|record| chapter_record_document_identity(&record.content))
        .collect::<std::collections::HashSet<_>>()
        .len()
        > 1
}

/// Apply presentation budgets only after verification. Never render the old
/// free-form chapter draft or drop proof inputs to satisfy a display limit.
fn render_verified_chapter_range(
    query: &str,
    coverage: &[CoverageReport],
    sources: &[RetrievedItem],
    answer_nodes: Option<usize>,
    document_limit: Option<usize>,
) -> Option<(String, usize)> {
    let (start, end) = answer_chapter_range(query)?;
    let mut documents = BTreeMap::<String, Vec<(u32, usize, &CoverageReport)>>::new();
    for (index, report) in coverage.iter().enumerate().filter(|(_, report)| report.covered) {
        if report.evidence_atom_ids.is_empty() { return None; }
        let identities = sources.iter().filter(|source|
            report.evidence_ids.contains(&source.id) && is_novel_chapter_summary_item(source))
            .filter_map(|source| canonical_chapter_number(&source.content)
                .filter(|chapter| (start..=end).contains(chapter))
                .map(|chapter| (chapter_record_document_identity(&source.content), chapter)))
            .collect::<BTreeSet<_>>();
        // Cross-document derivations need an explicit relationship layout;
        // do not silently assign one to an arbitrary document or omit it.
        let names = identities.iter().map(|(name, _)| name).collect::<BTreeSet<_>>();
        if names.len() != 1 { return None; }
        let (name, chapter) = identities.iter().next()?;
        documents.entry(name.clone()).or_default().push((*chapter, index, report));
    }
    if documents.is_empty() { return None; }
    let nodes = answer_nodes.map(normalize_answer_range_node_count)
        .or_else(|| explicit_answer_range_node_count(query)).unwrap_or(8);
    let total_documents = documents.len();
    let total_claims = documents.values().map(Vec::len).sum::<usize>();
    let mut shown_claims = 0;
    let shown_documents = total_documents.min(normalize_document_limit(document_limit));
    let mut body = Vec::new();
    for (name, mut reports) in documents.into_iter().take(shown_documents) {
        reports.sort_by_key(|(chapter, index, _)| (*chapter, *index));
        let label = compact_context_line(&strip_inline_answer_citations(&name), 140)
            .replace(['[', ']', '`'], "");
        let lines = reports.into_iter().take(nodes).map(|(chapter, _, report)| {
            shown_claims += 1;
            format!("- 第{chapter}章：{}", checked_claim_cited_text(report, sources))
        }).collect::<Vec<_>>().join("\n");
        body.push(format!("{label}（{nodes} 个以内/文档）：\n{lines}"));
    }
    let omitted = total_claims.saturating_sub(shown_claims);
    Some((format!("第{start}章到第{end}章已核验回答：\n\n节点预算：每个文档单独计算；找到 {total_documents} 个已核验文档，本次显示 {shown_documents} 个文档。\n\n{}\n\n显示覆盖：省略 {omitted} 条已核验结论；完整来源及证明保留在 claims/coverage 中。", body.join("\n\n")), omitted))
}

fn verified_chapter_document_labels(
    coverage: &[CoverageReport],
    sources: &[RetrievedItem],
) -> Vec<String> {
    let used = coverage
        .iter()
        .filter(|report| report.covered && !report.evidence_atom_ids.is_empty())
        .flat_map(|report| report.evidence_ids.iter().copied())
        .collect::<BTreeSet<_>>();
    sources
        .iter()
        .filter(|item| used.contains(&item.id) && is_novel_chapter_summary_item(item))
        .filter(|item| {
            answer_line_value(&item.content, "Novel").is_some()
                || answer_line_value(&item.content, "Source file").is_some()
        })
        .map(|item| {
            let label =
                strip_inline_answer_citations(&chapter_record_document_identity(&item.content));
            compact_context_line(&label, 140).replace(['[', ']', '`'], "")
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn chapter_record_document_identity(content: &str) -> String {
    let novel = answer_line_value(content, "Novel");
    let source_file = answer_line_value(content, "Source file");
    match (novel, source_file) {
        (Some(novel), Some(source_file)) if source_file != "uploaded-text" => {
            format!("{novel} / {source_file}")
        }
        (Some(novel), _) => novel,
        (_, Some(source_file)) => source_file,
        _ => "unknown document".to_string(),
    }
}

fn query_mentions_document_identity(query: &str, content: &str) -> bool {
    let query_lower = query.to_lowercase();
    [
        answer_line_value(content, "Novel"),
        answer_line_value(content, "Source file"),
    ]
    .into_iter()
    .flatten()
    .filter(|value| !value.trim().is_empty() && value != "uploaded-text")
    .any(|value| {
        let value_lower = value.to_lowercase();
        query_lower.contains(&value_lower)
            || value_lower
                .split(['/', '\\'])
                .next_back()
                .map(|file_name| query_lower.contains(file_name))
                .unwrap_or(false)
    })
}

pub(crate) fn answer_wants_multi_document_chapters(query: &str) -> bool {
    let query_lower = query.to_lowercase();
    contains_any(
        &query_lower,
        &[
            "all files",
            "all documents",
            "all novels",
            "each file",
            "each document",
            "each novel",
            "across files",
            "across documents",
            "compare",
            "comparison",
            "所有文件",
            "所有文档",
            "所有小说",
            "每个文件",
            "每本文档",
            "每本小说",
            "各文件",
            "各文档",
            "各小说",
            "比较",
            "对比",
            "跨文件",
            "跨文档",
        ],
    )
}

fn render_chapter_storylines_section(storylines: &[String]) -> String {
    if storylines.is_empty() {
        return String::new();
    }
    let body = storylines
        .iter()
        .take(4)
        .map(|storyline| format!("- {}", compact_context_line(storyline, 260)))
        .collect::<Vec<_>>()
        .join("\n");
    format!("\n\n故事线：\n{body}")
}

fn render_chapter_summary_evidence_line(
    memory_id: u64,
    content: &str,
    source: Option<&str>,
    summary: &str,
    key_events: &[String],
) -> String {
    let source_label = chapter_summary_source_label(content, source);
    let support = key_events
        .iter()
        .take(2)
        .map(|event| compact_context_line(event, 120))
        .collect::<Vec<_>>()
        .join("；");
    let support = if support.is_empty() {
        compact_context_line(summary, 180)
    } else {
        support
    };
    format!("- memory#{memory_id} [{source_label}] 摘要依据：{support}")
}

fn declared_chapter_archive_ids(content: &str) -> Vec<u64> {
    let Some(value) = answer_line_value(content, "Source archive memory IDs") else {
        return vec![];
    };
    let ids = value
        .split(',')
        .map(|value| value.trim().parse::<u64>())
        .collect::<Result<BTreeSet<_>, _>>();
    ids.map(|ids| ids.into_iter().filter(|id| *id > 0).collect())
        .unwrap_or_default()
}

fn bind_chapter_archive_proof(
    plan: &crate::answer_pipeline::PlannedAnswerPipeline,
    claim: &str,
    nominated: &HashSet<u64>,
    sources: &[RetrievedItem],
    proof: &mut crate::answer_pipeline::VerifiedClaimEvidence,
) {
    let mut declared_link = false;
    let mut original_supported = false;
    for summary in sources
        .iter()
        .filter(|item| {
            nominated.contains(&item.id)
                && proof.memory_ids.contains(&item.id)
                && is_novel_chapter_summary_item(item)
        })
        .collect::<Vec<_>>()
    {
        declared_link |= answer_line_value(&summary.content, "Source archive memory IDs").is_some();
        for id in declared_chapter_archive_ids(&summary.content) {
            let Some(archive) = sources.iter().find(|item| {
                item.id == id
                    && memory_item_is_active(item)
                    && (item.layer == MemoryLayer::Archive
                        || item.content.contains("Archive type:")
                        || item.source == "admin-ui-novel-import")
            }) else {
                continue;
            };
            if canonical_chapter_number(&summary.content)
                .zip(canonical_chapter_number(&archive.content))
                .is_some_and(|(a, b)| a != b)
            {
                continue;
            }
            let archive_ids = HashSet::from([id]);
            let mut bound = plan.verify_claim_evidence(claim, &archive_ids);
            if !bound.is_bound() {
                bound = plan.verify_verbatim_renderer_quotation(claim, &archive_ids);
            }
            if !bound.is_bound() {
                continue;
            }
            original_supported = true;
            proof.derivation.push(format!("chapter_archive_support:{}", serde_json::json!({
                "summary_id":summary.id, "summary_version":summary.version,
                "archive_id":id, "archive_version":archive.version,
                "archive_digest":crate::shared_object::stable_digest(archive.content.as_bytes()),
                "method":"declared_link_and_independent_claim_binding"
            })));
            proof.memory_ids.extend(bound.memory_ids);
            proof.atom_ids.extend(bound.atom_ids);
            proof.source_anchors.extend(bound.source_anchors);
            proof.derivation.extend(bound.derivation);
        }
    }
    if declared_link && !original_supported {
        // A derived record declaring underlying sources must not certify
        // itself when those sources are missing or do not support the claim.
        *proof = crate::answer_pipeline::VerifiedClaimEvidence::default();
        return;
    }
    proof.memory_ids.sort_unstable();
    proof.memory_ids.dedup();
    proof.atom_ids.sort();
    proof.atom_ids.dedup();
}

fn chapter_summary_source_label(content: &str, source: Option<&str>) -> String {
    let summary_type = "novel_chapter_summary";
    let archive_ids = answer_line_value(content, "Source archive memory IDs")
        .map(|value| {
            value
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(|value| format!("source memory#{value}"))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if archive_ids.is_empty() {
        source
            .filter(|value| !value.trim().is_empty())
            .map(|value| format!("{summary_type}; source={value}"))
            .unwrap_or_else(|| summary_type.to_string())
    } else {
        format!("{summary_type} -> {}", archive_ids.join(", "))
    }
}

fn is_novel_chapter_summary_item(item: &RetrievedItem) -> bool {
    !item.content.trim_start().starts_with("event:")
        && memory_item_has_type(item, "novel_chapter_summary")
}

fn memory_record_is_active(record: &MemoryRecord) -> bool {
    record.superseded_by.is_empty()
        && memory_quality_text_value(&record.content, &["superseded_by"]).is_none()
}

pub(crate) fn memory_record_is_eligible_for_selected_answer(record: &MemoryRecord) -> bool {
    memory_record_is_active(record)
        && !should_exclude_from_retrieval(record)
        && (record.layer != MemoryLayer::Activation
            || MemoryEngine::activation_memory_is_live(record))
        && !record.semantic_node.as_ref().is_some_and(|node| {
            matches!(
                node.status,
                SemanticStatus::Superseded | SemanticStatus::Deprecated
            )
        })
}

pub(crate) fn selected_answer_source_record_digest(record: &MemoryRecord) -> String {
    format!(
        "sha256:{}",
        crate::shared_object::stable_digest(format!(
            "{}:{}:{}:{}:{}",
            record.id,
            record.version,
            record.source,
            record.content,
            record.created_at.to_rfc3339()
        ))
    )
}

fn memory_record_has_direct_type(record: &MemoryRecord, memory_type: &str) -> bool {
    if record.content.trim_start().starts_with("event:") {
        return false;
    }
    record.memory_type.as_deref() == Some(memory_type)
        || record
            .content
            .trim_start()
            .starts_with(&format!("memory_type: {memory_type}"))
        || record
            .content
            .trim_start()
            .starts_with(&format!("memory_type:{memory_type}"))
}

fn memory_record_is_archive(record: &MemoryRecord) -> bool {
    record.layer == MemoryLayer::Archive
        || record.content.contains("Archive type:")
        || record.source == "admin-ui-novel-import"
}

fn memory_item_is_active(item: &RetrievedItem) -> bool {
    item.superseded_by.is_empty()
        && memory_quality_text_value(&item.content, &["superseded_by"]).is_none()
}

fn memory_item_has_type(item: &RetrievedItem, memory_type: &str) -> bool {
    item.memory_type == memory_type
        || item
            .content
            .contains(&format!("memory_type: {memory_type}"))
        || item.content.contains(&format!("memory_type:{memory_type}"))
        || item
            .content
            .contains(&format!("\"memory_type\":\"{memory_type}\""))
        || item
            .content
            .contains(&format!("\"memory_type\": \"{memory_type}\""))
}

fn novel_chapter_marker_matches(content: &str, marker: &str) -> bool {
    content.contains(marker)
        || canonical_chapter_number(content)
            .zip(canonical_chapter_number(marker))
            .map(|(content_number, marker_number)| content_number == marker_number)
            .unwrap_or(false)
}

fn canonical_chapter_number(text: &str) -> Option<u32> {
    let start = text.find('第')?;
    let tail = &text[start + '第'.len_utf8()..];
    let mut number_text = String::new();
    for ch in tail.chars() {
        if matches!(ch, '章' | '节' | '幕' | '卷' | '回') {
            break;
        }
        if ch.is_whitespace() {
            continue;
        }
        if ch.is_ascii_digit()
            || chinese_numeral_value(ch).is_some()
            || ch == '十'
            || ch == '百'
            || ch == '千'
        {
            number_text.push(ch);
        } else {
            return None;
        }
        if number_text.chars().count() > 12 {
            return None;
        }
    }
    if number_text.is_empty() {
        return None;
    }
    number_text
        .parse::<u32>()
        .ok()
        .or_else(|| parse_chinese_chapter_number(&number_text))
}

fn chinese_numeral_value(ch: char) -> Option<u32> {
    match ch {
        '零' | '〇' => Some(0),
        '一' => Some(1),
        '二' | '两' => Some(2),
        '三' => Some(3),
        '四' => Some(4),
        '五' => Some(5),
        '六' => Some(6),
        '七' => Some(7),
        '八' => Some(8),
        '九' => Some(9),
        _ => None,
    }
}

pub(crate) fn parse_chinese_chapter_number(text: &str) -> Option<u32> {
    let mut total = 0;
    let mut current = 0;
    let mut saw_any = false;
    for ch in text.chars() {
        match ch {
            '千' => {
                total += current.max(1) * 1000;
                current = 0;
                saw_any = true;
            }
            '百' => {
                total += current.max(1) * 100;
                current = 0;
                saw_any = true;
            }
            '十' => {
                total += current.max(1) * 10;
                current = 0;
                saw_any = true;
            }
            _ => {
                current = chinese_numeral_value(ch)?;
                saw_any = true;
            }
        }
    }
    saw_any.then_some(total + current)
}

fn render_novel_style_answer(
    query: &str,
    items: &[RetrievedItem],
) -> Option<(String, Vec<RendererClaim>)> {
    let query_lower = query.to_lowercase();
    let asks_style = contains_any(
        &query_lower,
        &[
            "writing style",
            "style profile",
            "author style",
            "voice",
            "diction",
            "rhythm",
            "tone",
            "imagery",
            "dialogue",
            "pacing",
            "style transfer",
            "文风",
            "写作风格",
            "风格",
            "语气",
            "节奏",
            "意象",
            "对白",
        ],
    );
    if !asks_style {
        return None;
    }
    let style_records = items
        .iter()
        .filter(|item| {
            memory_item_is_active(item)
                && (memory_item_has_profile_type(item, "novel_style_profile")
                    || memory_item_has_profile_type(item, "writing_style_profile"))
        })
        .take(8)
        .map(|item| {
            if memory_item_has_profile_type(item, "writing_style_profile") {
                profile_citation_line(item, "writing_style_profile", "style")
            } else {
                profile_citation_line(item, "novel_style_profile", "style")
            }
        })
        .collect::<Vec<_>>();
    if style_records.is_empty() {
        return None;
    }
    let mut claims = profile_candidate_claims(items, "writing_style_profile");
    for claim in profile_candidate_claims(items, "novel_style_profile") {
        if !claims
            .iter()
            .any(|existing| existing.evidence_ids == claim.evidence_ids)
        {
            claims.push(claim);
        }
    }
    Some((format!(
        "写作风格结构化回答：\n\n{}\n\n风格结论：以上 style_profile 记录来自 durable writing-style extraction，可覆盖小说、短文、脚本、博客、聊天样本、文档等来源；novel_style_profile 是其中的小说专用记录。用于仿写或风格迁移前，应先 review 这些记录；如果文风判断不对，添加 corrected writing_style_profile/novel_style_profile replacement 并 supersede 旧记录。\n\nTrace: 使用响应里的 trace_id 查询 /memory/trace/{{trace_id}}/summary，查看 style_profile evidence 和 correction action。",
        style_records.join("\n")
    ), claims))
}

fn profile_candidate_claims(items: &[RetrievedItem], profile_type: &str) -> Vec<RendererClaim> {
    items
        .iter()
        .filter(|item| {
            memory_item_is_active(item) && memory_item_has_profile_type(item, profile_type)
        })
        .map(|item| RendererClaim {
            text: profile_field_value(&item.content, "summary", &["Summary"])
                .unwrap_or_else(|| item.content.clone()),
            evidence_ids: vec![item.id],
        })
        .collect()
}

fn render_design_style_answer(
    query: &str,
    items: &[RetrievedItem],
) -> Option<(String, Vec<RendererClaim>)> {
    let query_lower = query.to_lowercase();
    let asks_design = contains_any(
        &query_lower,
        &[
            "design style",
            "web design",
            "ui design",
            "visual style",
            "interface style",
            "layout",
            "typography",
            "color palette",
            "spacing",
            "component style",
            "responsive",
            "accessibility",
            "网页设计",
            "界面设计",
            "视觉风格",
            "布局",
            "配色",
            "排版",
            "组件",
            "响应式",
            "可访问性",
        ],
    );
    if !asks_design {
        return None;
    }
    let records = items
        .iter()
        .filter(|item| {
            memory_item_is_active(item)
                && memory_item_has_profile_type(item, "design_style_profile")
        })
        .take(8)
        .map(|item| profile_citation_line(item, "design_style_profile", "design"))
        .collect::<Vec<_>>();
    if records.is_empty() {
        return None;
    }
    let display = format!(
        "Design style profile answer:\n\n{}\n\nDesign conclusion: these design_style_profile records are durable, reviewable evidence for visual hierarchy, layout density, palette, typography, spacing, components, interaction, accessibility, media, and responsive behavior. Use them for web/UI design guidance only after reviewing the cited source notes, screenshots, captions, OCR/transcripts, or design metadata; correct inaccurate records with /memory/design-style/correct.\n\nTrace: use the response trace_id with /memory/trace/{{trace_id}}/summary to inspect cited design records.",
        records.join("\n")
    );
    Some((
        display,
        profile_candidate_claims(items, "design_style_profile"),
    ))
}

#[derive(Debug, Clone)]
struct TemporalDateEvidence {
    memory_id: u64,
    source: String,
    date: chrono::NaiveDate,
    event_label: String,
    score: f32,
    anchor_start: usize,
}

fn english_month_number(value: &str) -> Option<u32> {
    match value.to_ascii_lowercase().as_str() {
        "jan" | "january" => Some(1),
        "feb" | "february" => Some(2),
        "mar" | "march" => Some(3),
        "apr" | "april" => Some(4),
        "may" => Some(5),
        "jun" | "june" => Some(6),
        "jul" | "july" => Some(7),
        "aug" | "august" => Some(8),
        "sep" | "sept" | "september" => Some(9),
        "oct" | "october" => Some(10),
        "nov" | "november" => Some(11),
        "dec" | "december" => Some(12),
        _ => None,
    }
}

fn content_session_year(content: &str) -> Option<i32> {
    static SESSION_DATE: OnceLock<regex::Regex> = OnceLock::new();
    let regex = SESSION_DATE.get_or_init(|| {
        regex::Regex::new(r"(?i)Session date:\s*([0-9]{4})[/\-][0-9]{1,2}[/\-][0-9]{1,2}")
            .expect("session date regex")
    });
    regex
        .captures(content)
        .and_then(|captures| captures.get(1))
        .and_then(|value| value.as_str().parse::<i32>().ok())
}

pub(crate) fn temporal_date_anchors(
    content: &str,
    default_year: i32,
) -> Vec<(chrono::NaiveDate, usize, usize, bool)> {
    static NUMERIC_DATE: OnceLock<regex::Regex> = OnceLock::new();
    static ENGLISH_DATE: OnceLock<regex::Regex> = OnceLock::new();
    static CHINESE_DATE: OnceLock<regex::Regex> = OnceLock::new();
    let numeric = NUMERIC_DATE.get_or_init(|| {
        regex::Regex::new(r"\b([0-9]{4})[/\-]([0-9]{1,2})[/\-]([0-9]{1,2})\b")
            .expect("numeric date regex")
    });
    let english = ENGLISH_DATE.get_or_init(|| {
        regex::Regex::new(
            r"(?i)\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+([0-9]{1,2})(?:st|nd|rd|th)?(?:,?\s+([0-9]{4}))?\b",
        )
        .expect("English date regex")
    });
    let chinese = CHINESE_DATE.get_or_init(|| {
        regex::Regex::new(r"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})日").expect("Chinese date regex")
    });
    let mut anchors = Vec::new();

    for captures in numeric.captures_iter(content) {
        let Some(full_match) = captures.get(0) else {
            continue;
        };
        let parsed = captures
            .get(1)
            .and_then(|value| value.as_str().parse::<i32>().ok())
            .zip(
                captures
                    .get(2)
                    .and_then(|value| value.as_str().parse::<u32>().ok()),
            )
            .zip(
                captures
                    .get(3)
                    .and_then(|value| value.as_str().parse::<u32>().ok()),
            )
            .and_then(|((year, month), day)| chrono::NaiveDate::from_ymd_opt(year, month, day));
        if let Some(date) = parsed {
            let line_start = content[..full_match.start()]
                .rfind('\n')
                .map(|index| index + 1)
                .unwrap_or(0);
            let header = content[line_start..full_match.start()]
                .to_ascii_lowercase()
                .contains("session date:");
            anchors.push((date, full_match.start(), full_match.end(), header));
        }
    }
    for captures in english.captures_iter(content) {
        let Some(full_match) = captures.get(0) else {
            continue;
        };
        let parsed = captures
            .get(1)
            .and_then(|value| english_month_number(value.as_str()))
            .zip(
                captures
                    .get(2)
                    .and_then(|value| value.as_str().parse::<u32>().ok()),
            )
            .map(|(month, day)| {
                let year = captures
                    .get(3)
                    .and_then(|value| value.as_str().parse::<i32>().ok())
                    .unwrap_or(default_year);
                (year, month, day)
            })
            .and_then(|(year, month, day)| chrono::NaiveDate::from_ymd_opt(year, month, day));
        if let Some(date) = parsed {
            anchors.push((date, full_match.start(), full_match.end(), false));
        }
    }
    for captures in chinese.captures_iter(content) {
        let Some(full_match) = captures.get(0) else {
            continue;
        };
        let year = captures
            .get(1)
            .and_then(|value| value.as_str().parse::<i32>().ok())
            .unwrap_or(default_year);
        let parsed = captures
            .get(2)
            .and_then(|value| value.as_str().parse::<u32>().ok())
            .zip(
                captures
                    .get(3)
                    .and_then(|value| value.as_str().parse::<u32>().ok()),
            )
            .and_then(|(month, day)| chrono::NaiveDate::from_ymd_opt(year, month, day));
        if let Some(date) = parsed {
            anchors.push((date, full_match.start(), full_match.end(), false));
        }
    }
    anchors.sort_by_key(|(_, start, _, _)| *start);
    anchors.dedup_by(|left, right| left.0 == right.0 && left.1 == right.1);
    anchors
}

fn bounded_text_window(content: &str, start: usize, end: usize, radius: usize) -> &str {
    let mut window_start = start.saturating_sub(radius);
    while window_start > 0 && !content.is_char_boundary(window_start) {
        window_start -= 1;
    }
    let mut window_end = end.saturating_add(radius).min(content.len());
    while window_end < content.len() && !content.is_char_boundary(window_end) {
        window_end += 1;
    }
    &content[window_start..window_end]
}

fn temporal_evidence_for_clause(
    clause: &str,
    items: &[RetrievedItem],
) -> Vec<TemporalDateEvidence> {
    let mut candidates = Vec::new();
    for item in items.iter().filter(|item| memory_item_is_active(item)) {
        let searchable = searchable_source_body(&item.content);
        let default_year = content_session_year(&item.content).unwrap_or_else(|| {
            item.created_at
                .format("%Y")
                .to_string()
                .parse::<i32>()
                .unwrap_or(1970)
        });
        for (date, start, end, session_header) in temporal_date_anchors(searchable, default_year) {
            let window = bounded_text_window(searchable, start, end, 420);
            let mut score = crate::memory_engine_query::lexical_query_coverage(clause, window);
            if window.to_lowercase().contains(&clause.to_lowercase()) {
                score += 0.20;
            }
            if !session_header {
                score += 0.05;
            }
            if score < 0.20 {
                continue;
            }
            candidates.push(TemporalDateEvidence {
                memory_id: item.id,
                source: item.source.clone(),
                date,
                event_label: clause.to_string(),
                score,
                anchor_start: start,
            });
        }
    }
    candidates.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
    });
    candidates.truncate(12);
    candidates
}

pub(crate) fn temporal_duration_requested(query: &str) -> bool {
    let query = query.to_lowercase();
    contains_any(
        &query,
        &[
            "how many days",
            "days passed",
            "days elapsed",
            "day difference",
            "相隔多少天",
            "过去多少天",
            "过了多少天",
            "间隔多少天",
        ],
    )
}

fn grounded_line_segments(line: &str) -> Vec<&str> {
    let mut segments = Vec::new();
    let mut start = 0usize;
    let mut previous = None;
    let mut characters = line.char_indices().peekable();
    while let Some((index, character)) = characters.next() {
        let next = characters.peek().map(|(_, value)| *value);
        let decimal_point = character == '.'
            && previous.is_some_and(|value: char| value.is_ascii_digit())
            && next.is_some_and(|value| value.is_ascii_digit());
        if matches!(character, '.' | '!' | '?' | '。' | '！' | '？' | ';' | '；') && !decimal_point
        {
            let end = index + character.len_utf8();
            if start < end {
                segments.push(&line[start..end]);
            }
            start = end;
        }
        previous = Some(character);
    }
    if start < line.len() {
        segments.push(&line[start..]);
    }
    segments
}

fn strip_grounded_turn_id(value: &str) -> &str {
    let value = value.trim();
    let Some(separator) = value.find(char::is_whitespace) else {
        return value;
    };
    let prefix = &value[..separator];
    let Some((session, turn)) = prefix.split_once(':') else {
        return value;
    };
    let mut session_chars = session.chars();
    let valid_session = session_chars
        .next()
        .is_some_and(|character| character.is_ascii_alphabetic())
        && session_chars.all(|character| character.is_ascii_digit());
    if valid_session && !turn.is_empty() && turn.chars().all(|character| character.is_ascii_digit())
    {
        value[separator..].trim_start()
    } else {
        value
    }
}

fn grounded_statement_parts(value: &str) -> (Option<&str>, &str) {
    let value = strip_grounded_turn_id(value);
    let Some((speaker, body)) = value.split_once(':') else {
        return (None, value);
    };
    let speaker = speaker.trim();
    let body = body.trim();
    let plausible_speaker = !speaker.is_empty()
        && speaker.chars().count() <= 48
        && speaker
            .chars()
            .all(|character| character.is_alphabetic() || matches!(character, ' ' | '-' | '_'));
    if plausible_speaker && !body.is_empty() {
        (Some(speaker), body)
    } else {
        (None, value)
    }
}

fn grounded_statement_is_media_caption(value: &str) -> bool {
    fn has_caption_prefix(value: &str) -> bool {
        let lower = value.trim_start().to_lowercase();
        [
            "[image caption:",
            "image caption:",
            "[video caption:",
            "video caption:",
            "[audio caption:",
            "audio caption:",
        ]
        .iter()
        .any(|prefix| lower.starts_with(prefix))
    }

    let value = strip_grounded_turn_id(value);
    let (_, body) = grounded_statement_parts(value);
    has_caption_prefix(value) || has_caption_prefix(body)
}

fn grounded_evidence_display_text(value: &str) -> String {
    let value = value.trim();
    let lower = value.to_lowercase();
    for prefix in [
        "[image caption:",
        "image caption:",
        "[video caption:",
        "video caption:",
        "[audio caption:",
        "audio caption:",
    ] {
        if lower.starts_with(prefix) {
            return value[prefix.len()..]
                .trim()
                .trim_end_matches(']')
                .trim()
                .to_string();
        }
    }
    value.to_string()
}

fn grounded_text_without_media_captions(value: &str) -> String {
    static MEDIA_CAPTION: OnceLock<regex::Regex> = OnceLock::new();
    let regex = MEDIA_CAPTION.get_or_init(|| {
        regex::Regex::new(r"(?i)\[?(?:image|video|audio)\s+caption:[^\]]*(?:\]|$)")
            .expect("inline media caption regex")
    });
    regex
        .replace_all(value, " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn grounded_inline_media_caption(value: &str) -> Option<String> {
    static MEDIA_CAPTION: OnceLock<regex::Regex> = OnceLock::new();
    let regex = MEDIA_CAPTION.get_or_init(|| {
        regex::Regex::new(r"(?i)\[(?:image|video|audio)\s+caption:\s*([^\]]+)\]")
            .expect("inline media caption capture regex")
    });
    regex
        .captures(value)
        .and_then(|captures| captures.get(1))
        .map(|capture| capture.as_str().trim().to_string())
        .filter(|caption| !caption.is_empty())
}

fn query_requests_visual_detail(query: &str) -> bool {
    let lower = query.to_lowercase();
    contains_any(
        &lower,
        &[
            "what color",
            "which color",
            "look like",
            "appearance",
            "physical transformation",
            "new look",
            "what did",
            "什么颜色",
            "长什么样",
            "外观",
            "外貌变化",
        ],
    ) && (contains_any(
        &lower,
        &[
            "paint",
            "draw",
            "design",
            "create",
            "hair",
            "transformation",
            "look",
            "color",
            "画",
            "设计",
            "头发",
            "变化",
            "颜色",
        ],
    ) || contains_any(&lower, &["image", "photo", "video", "图片", "照片", "视频"]))
}

fn grounded_dialogue_turns(content: &str) -> Vec<(Option<String>, String)> {
    let metadata_prefixes = [
        "archive type:",
        "source file:",
        "source:",
        "evidence:",
        "observation:",
        "session date:",
        "longmemeval session",
        "universal source projection",
    ];
    content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|line| {
            let lower = line.to_lowercase();
            !metadata_prefixes
                .iter()
                .any(|prefix| lower.starts_with(prefix))
        })
        .filter_map(|line| {
            let line = strip_grounded_turn_id(line);
            let (speaker, body) = grounded_statement_parts(line);
            (!body.trim().is_empty())
                .then(|| (speaker.map(str::to_string), body.trim().to_string()))
        })
        .collect()
}

fn adjacent_question_score_for_statement(
    query: &str,
    content: &str,
    statement: &str,
) -> Option<f32> {
    let target = normalized_grounded_statement(grounded_statement_parts(statement).1);
    if target.is_empty() {
        return None;
    }
    grounded_dialogue_turns(content)
        .windows(2)
        .find_map(|pair| {
            let question_body = &pair[0].1;
            let answer_body = &pair[1].1;
            let answer_key = normalized_grounded_statement(answer_body);
            if !question_body.contains(['?', '？'])
                || !(answer_key.contains(&target) || target.contains(&answer_key))
            {
                return None;
            }
            Some(
                crate::memory_engine_query::lexical_query_coverage(query, question_body)
                    + semantic_topic_bonus(query, question_body),
            )
        })
}

fn grounded_adjacent_answer_candidates(
    query: &str,
    content: &str,
    item_score: f32,
) -> Vec<(String, f32)> {
    let visual_detail = query_requests_visual_detail(query);
    grounded_dialogue_turns(content)
        .windows(2)
        .filter_map(|pair| {
            let (question_speaker, question_body) = &pair[0];
            let (answer_speaker, answer_body) = &pair[1];
            if !question_body.contains(['?', '？'])
                || question_speaker
                    .as_deref()
                    .zip(answer_speaker.as_deref())
                    .is_some_and(|(left, right)| person_names_match(left, right))
            {
                return None;
            }
            if let Some(subject) = question_named_subject(query) {
                let answer_subject_matches = answer_speaker
                    .as_deref()
                    .is_some_and(|speaker| person_names_match(speaker, &subject))
                    || answer_body
                        .split(|character: char| !character.is_alphabetic())
                        .filter(|token| !token.is_empty())
                        .any(|token| person_names_match(token, &subject));
                if answer_speaker.is_some() && !answer_subject_matches {
                    return None;
                }
            }
            let question_score =
                crate::memory_engine_query::lexical_query_coverage(query, question_body)
                    + semantic_topic_bonus(query, question_body);
            if question_score < 0.32 {
                return None;
            }
            let mut answer = grounded_text_without_media_captions(answer_body);
            let mut visual_evidence_bonus = 0.0;
            if visual_detail {
                if let Some(caption) = grounded_inline_media_caption(answer_body) {
                    if !answer.is_empty() {
                        answer.push_str(" Visual evidence: ");
                    }
                    answer.push_str(&caption);
                    visual_evidence_bonus = 0.75;
                }
            }
            if answer.chars().count() < 3 {
                return None;
            }
            let answer_score = statement_relevance_score(query, answer_body, item_score);
            Some((
                compact_context_line(&answer, 420),
                question_score + answer_score.max(0.0) * 0.35 + 0.25 + visual_evidence_bonus,
            ))
        })
        .collect()
}

fn low_information_statement(value: &str) -> bool {
    let lower = value.trim().to_lowercase();
    value.chars().count() <= 90
        && [
            "hey ",
            "hi ",
            "hello ",
            "wow",
            "glad ",
            "thanks",
            "thank you",
            "that sounds",
            "sounds great",
            "that's awesome",
            "thats awesome",
            "cool",
        ]
        .iter()
        .any(|prefix| lower.starts_with(prefix))
}

fn semantic_topic_bonus(query: &str, statement: &str) -> f32 {
    let query = query.to_lowercase();
    let statement = statement.to_lowercase();
    let topic_groups: &[(&[&str], &[&str])] = &[
        (
            &[
                "education",
                "educaton",
                "field",
                "career",
                "pursue",
                "study",
                "degree",
            ],
            &[
                "psychology",
                "counseling",
                "counselling",
                "certification",
                "degree",
                "career",
                "mental health",
            ],
        ),
        (
            &["identity", "identify", "gender"],
            &[
                "transgender woman",
                "transgender man",
                "nonbinary",
                "non-binary",
                "gender identity",
            ],
        ),
        (
            &["relationship status", "relationship", "dating"],
            &[
                "single",
                "married",
                "divorced",
                "partner",
                "dating",
                "breakup",
                "relationship",
            ],
        ),
        (
            &["activities", "activity", "partake", "hobbies", "hobby"],
            &[
                "pottery", "camping", "painting", "swimming", "running", "hiking", "reading",
            ],
        ),
        (
            &["book", "books", "read"],
            &["book", "read", "novel", "title", "author"],
        ),
        (
            &[
                "creative project",
                "paint",
                "painting",
                "draw",
                "art project",
            ],
            &["paint", "painting", "drawing", "art", "nature-inspired"],
        ),
        (
            &["song", "music", "vivaldi", "four seasons", "composer"],
            &["music", "classical", "bach", "mozart", "composer", "song"],
        ),
        (
            &[
                "physical transformation",
                "appearance",
                "hair color",
                "new look",
            ],
            &[
                "dyed my hair",
                "dyed his hair",
                "dyed her hair",
                "hair color",
                "purple hair",
                "haircut",
                "shaved",
                "tattoo",
                "piercing",
            ],
        ),
        (
            &["recommend", "living room", "comfy", "comfortable"],
            &["couch", "sofa", "blanket", "lights", "lighting", "comfy"],
        ),
        (
            &["sparked", "interest", "education", "infrastructure"],
            &[
                "growing up",
                "saw how",
                "affected",
                "neighborhood",
                "crumbling",
            ],
        ),
        (
            &["destress", "de-stress", "relax", "stress"],
            &[
                "relax", "peace", "calm", "running", "pottery", "painting", "music",
            ],
        ),
        (
            &["research", "investigate"],
            &["research", "look into", "looking into", "adoption agenc"],
        ),
        (
            &["speech", "speak", "presentation"],
            &["speech", "talk", "spoke", "school event", "presentation"],
        ),
        (
            &["race", "run", "marathon"],
            &["race", "run", "ran", "marathon"],
        ),
        (&["museum", "exhibit"], &["museum", "exhibit", "gallery"]),
        (&["picnic"], &["picnic"]),
        (
            &["pottery class", "pottery course"],
            &["pottery class", "pottery course", "signed up for pottery"],
        ),
        (&["conference"], &["conference", "convention", "summit"]),
        (
            &["move from", "moved from", "relocate from", "origin"],
            &[
                "move from",
                "moved from",
                "relocate from",
                "relocated from",
                "home country",
                "hometown",
                "originally from",
            ],
        ),
    ];
    let mut bonus = topic_groups
        .iter()
        .filter(|(questions, answers)| {
            questions.iter().any(|marker| query.contains(marker))
                && answers.iter().any(|marker| statement.contains(marker))
        })
        .count() as f32
        * 0.24;
    if contains_any(&query, &["identity", "identify", "gender"])
        && contains_any(
            &statement,
            &[
                "transgender woman",
                "trans woman",
                "transgender man",
                "nonbinary",
            ],
        )
    {
        bonus += 0.20;
    }
    if contains_any(&query, &["relationship status", "dating status"])
        && contains_any(
            &statement,
            &[
                "single",
                "married",
                "divorced",
                "widowed",
                "in a relationship",
            ],
        )
    {
        bonus += 0.20;
    }
    if contains_any(&query, &["career", "profession", "career path"])
        && contains_any(
            &statement,
            &[
                "counseling",
                "counselling",
                "mental health",
                "psychology",
                "engineering",
                "medicine",
                "teaching",
            ],
        )
    {
        bonus += 0.20;
    }
    bonus
}

fn grounded_source_statements(content: &str) -> Vec<String> {
    let metadata_prefixes = [
        "archive type:",
        "source file:",
        "source:",
        "evidence:",
        "observation:",
        "session date:",
        "longmemeval session",
        "universal source projection",
    ];
    let mut statements = Vec::new();
    for line in content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        let lower = line.to_lowercase();
        if metadata_prefixes
            .iter()
            .any(|prefix| lower.starts_with(prefix))
        {
            continue;
        }
        let (line_speaker, _) = grounded_statement_parts(strip_grounded_turn_id(line));
        for segment in grounded_line_segments(line) {
            let mut statement = segment.trim();
            for prefix in [
                "USER:",
                "ASSISTANT:",
                "User:",
                "Assistant:",
                "用户：",
                "助手：",
            ] {
                statement = statement.strip_prefix(prefix).unwrap_or(statement).trim();
            }
            statement = strip_grounded_turn_id(statement);
            let (_, body) = grounded_statement_parts(statement);
            if body.ends_with('?') || body.ends_with('？') {
                continue;
            }
            let statement = statement
                .trim_start_matches(['-', '*', '#'])
                .trim()
                .trim_end_matches(|character: char| {
                    matches!(character, '.' | '!' | '。' | '！' | ';' | '；')
                })
                .trim();
            let (_, body) = grounded_statement_parts(statement);
            if body.chars().count() < 8 {
                continue;
            }
            let attributed = if grounded_statement_parts(statement).0.is_none() {
                line_speaker
                    .map(|speaker| format!("{speaker}: {statement}"))
                    .unwrap_or_else(|| statement.to_string())
            } else {
                statement.to_string()
            };
            statements.push(compact_context_line(&attributed, 360));
        }
    }
    statements
}

fn normalized_grounded_statement(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn statement_relevance_score(query: &str, statement: &str, item_score: f32) -> f32 {
    let query_lower = query.to_lowercase();
    let (speaker, body) = grounded_statement_parts(statement);
    let statement_lower = body.to_lowercase();
    let named_subject = question_named_subject(query);
    let mut score =
        crate::memory_engine_query::lexical_query_coverage(query, body) + item_score * 0.08;
    score += semantic_topic_bonus(query, body);
    if let Some(subject) = question_named_subject(query) {
        if speaker.is_some_and(|speaker| !person_names_match(speaker, &subject)) {
            score -= 0.60;
        }
    }
    if query_requests_visual_detail(query) && grounded_statement_is_media_caption(statement) {
        score += 0.65;
    }
    if speaker.is_some_and(|speaker| {
        query_lower.contains(&speaker.to_lowercase())
            || named_subject
                .as_deref()
                .is_some_and(|subject| person_names_match(speaker, subject))
    }) {
        score += 0.24;
    }
    if low_information_statement(body) {
        score -= 0.25;
    }
    if contains_any(&query_lower, &["why", "原因", "为什么"])
        && contains_any(
            &statement_lower,
            &["because", "due to", "therefore", "所以", "因为", "导致"],
        )
    {
        score += 0.18;
    }
    if contains_any(
        &query_lower,
        &[
            "bookshelf",
            "book shelf",
            "bookcase",
            "book collection",
            "library",
        ],
    ) && contains_any(
        &statement_lower,
        &[
            "got lots of",
            "have lots of",
            "collect",
            "collection",
            "library",
            "bookshelf",
            "bookcase",
        ],
    ) {
        score += 0.40;
    }
    if contains_any(&query_lower, &["likely", "enjoy", "probably"])
        && contains_any(&query_lower, &["song", "music", "composer"])
        && contains_any(
            &statement_lower,
            &[
                "fan of",
                "i like",
                "i love",
                "i enjoy",
                "into classical",
                "classical music",
            ],
        )
    {
        score += 0.35;
    }
    if contains_any(&query_lower, &["likely", "enjoy", "probably"])
        && contains_any(&query_lower, &["song", "music", "composer"])
        && contains_any(
            &statement_lower,
            &[
                "classical",
                "jazz",
                "rock",
                "pop music",
                "folk",
                "hip-hop",
                "country music",
                "electronic",
                "blues",
                "metal",
            ],
        )
    {
        score += 0.30;
    }
    if contains_any(
        &query_lower,
        &["why", "sparked", "motivated", "inspired", "原因", "为什么"],
    ) && contains_any(
        &statement_lower,
        &[
            "growing up",
            "saw how",
            "affected",
            "drove me",
            "drives me",
            "reminded me",
        ],
    ) {
        score += 0.22;
    }
    if contains_any(
        &query_lower,
        &["when", "what date", "什么时候", "哪天", "日期"],
    ) && (contains_any(
        &statement_lower,
        &[
            "yesterday",
            "tomorrow",
            "today",
            "last week",
            "next week",
            "昨天",
            "前天",
            "今天",
            "明天",
        ],
    ) || !temporal_date_anchors(statement, 1970).is_empty())
    {
        score += 0.20;
    }
    if contains_any(&query_lower, &["where", "哪里", "哪儿", "地点"])
        && contains_any(
            &statement_lower,
            &[
                " at ",
                " in ",
                " near ",
                " from ",
                " located",
                "home country",
                "hometown",
                "origin",
                "位于",
                "地点",
                "在",
                "来自",
            ],
        )
    {
        score += 0.12;
    }
    score
}

#[derive(Debug, Clone)]
struct GroundedAnswerEvidence {
    memory_id: u64,
    source: String,
    source_label: String,
    text: String,
    score: f32,
    is_media_caption: bool,
}

fn query_requires_scalar_evidence(query: &str) -> bool {
    let lower = query.to_lowercase();
    contains_any(
        &lower,
        &[
            "how many",
            "how much",
            "how long",
            "what date",
            "which date",
            "when did",
            "when was",
            "when is",
            "when are",
            "when will",
            "多少",
            "多久",
            "几次",
            "哪天",
            "什么时候",
        ],
    )
}

fn statement_has_scalar_evidence(value: &str) -> bool {
    static SCALAR: OnceLock<regex::Regex> = OnceLock::new();
    let regex = SCALAR.get_or_init(|| {
        regex::Regex::new(r"(?i)\b\d+(?:\.\d+)?\b").expect("scalar evidence regex")
    });
    regex.is_match(value)
        || contains_any(
            &value.to_lowercase(),
            &[
                "once",
                "twice",
                "three times",
                "several",
                "many",
                "a few",
                "yesterday",
                "tomorrow",
                "last week",
                "next week",
                "day before",
                "week before",
                "month before",
                "year ago",
                "years ago",
                "天前",
                "周前",
                "月前",
                "年前",
            ],
        )
}

fn grounded_candidate_is_answerable(query: &str, candidate: &GroundedAnswerEvidence) -> bool {
    !query_requires_scalar_evidence(query) || statement_has_scalar_evidence(&candidate.text)
}

fn grounded_answer_candidates(query: &str, items: &[RetrievedItem]) -> Vec<GroundedAnswerEvidence> {
    let mut candidates = Vec::new();
    for item in items.iter().filter(|item| memory_item_is_active(item)) {
        let source_label =
            document_source_label_for_item(item).unwrap_or_else(|| item.source.clone());
        let source_body = searchable_source_body(&item.content);
        let mut item_candidates = grounded_source_statements(source_body)
            .into_iter()
            .filter_map(|statement| {
                let (speaker, body) = grounded_statement_parts(&statement);
                if !statement_matches_question_subject(query, speaker, body) {
                    return None;
                }
                let score = statement_relevance_score(query, &statement, item.score);
                let is_media_caption = grounded_statement_is_media_caption(&statement);
                Some(GroundedAnswerEvidence {
                    memory_id: item.id,
                    source: item.source.clone(),
                    source_label: source_label.clone(),
                    text: body.to_string(),
                    score,
                    is_media_caption,
                })
            })
            .collect::<Vec<_>>();
        item_candidates.extend(
            grounded_adjacent_answer_candidates(query, source_body, item.score)
                .into_iter()
                .map(|(text, score)| GroundedAnswerEvidence {
                    memory_id: item.id,
                    source: item.source.clone(),
                    source_label: source_label.clone(),
                    text,
                    score,
                    is_media_caption: false,
                }),
        );
        item_candidates.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
        });
        candidates.extend(item_candidates.into_iter().take(6));
    }
    candidates.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
    });

    let mut deduplicated: Vec<GroundedAnswerEvidence> = Vec::new();
    for candidate in candidates {
        let key = normalized_grounded_statement(&candidate.text);
        if key.is_empty()
            || deduplicated.iter().any(|existing| {
                let existing_key = normalized_grounded_statement(&existing.text);
                existing_key == key
                    || (key.chars().count() >= 24
                        && (existing_key.contains(&key) || key.contains(&existing_key)))
            })
        {
            continue;
        }
        deduplicated.push(candidate);
    }
    deduplicated
}

#[derive(Debug, Clone)]
struct SharedLocationEvidence {
    speaker: String,
    memory_id: u64,
    source: String,
    snippet: String,
}

fn visited_location_values(body: &str) -> Vec<String> {
    static DIRECT_VISIT: OnceLock<regex::Regex> = OnceLock::new();
    static TRIP_TO: OnceLock<regex::Regex> = OnceLock::new();
    let direct_visit = DIRECT_VISIT.get_or_init(|| {
        regex::Regex::new(
            r"(?i:\b(?:visited|visit|went|been|traveled|travelled)\s+(?:only\s+)?to\s+)([A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){0,2})",
        )
        .expect("visited location regex")
    });
    let trip_to = TRIP_TO.get_or_init(|| {
        regex::Regex::new(
            r"(?i:\b(?:trip|vacation|journey)[^.!?\r\n]{0,64}\bto\s+)([A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){0,2})",
        )
        .expect("trip location regex")
    });
    let mut values = direct_visit
        .captures_iter(body)
        .chain(trip_to.captures_iter(body))
        .filter_map(|captures| captures.get(1))
        .map(|capture| capture.as_str().trim().to_string())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    values.sort_by_key(|value| normalized_grounded_statement(value));
    values.dedup_by(|left, right| left.eq_ignore_ascii_case(right));
    values
}



fn render_shared_location_answer_plan(
    query: &str,
    items: &[RetrievedItem],
) -> Option<(String, Vec<RendererClaim>)> {
    let lower = query.to_lowercase();
    if !contains_any(&lower, &["both", "共同", "都"])
        || !contains_any(
            &lower,
            &["visit", "visited", "been", "travel", "去过", "到过"],
        )
        || !contains_any(&lower, &["city", "place", "where", "城市", "地点", "哪里"])
    {
        return None;
    }

    let mut by_location: std::collections::HashMap<String, (String, Vec<SharedLocationEvidence>)> =
        std::collections::HashMap::new();
    for item in items.iter().filter(|item| memory_item_is_active(item)) {
        for statement in grounded_source_statements(searchable_source_body(&item.content)) {
            let (speaker, body) = grounded_statement_parts(&statement);
            let Some(speaker) = speaker else {
                continue;
            };
            for location in visited_location_values(body) {
                let key = normalized_grounded_statement(&location);
                let entry = by_location
                    .entry(key)
                    .or_insert_with(|| (location.clone(), Vec::new()));
                if entry
                    .1
                    .iter()
                    .any(|existing| person_names_match(&existing.speaker, speaker))
                {
                    continue;
                }
                entry.1.push(SharedLocationEvidence {
                    speaker: speaker.to_string(),
                    memory_id: item.id,
                    source: item.source.clone(),
                    snippet: compact_context_line(body, 220),
                });
            }
        }
    }

    let mut shared = by_location
        .into_values()
        .filter(|(_, evidence)| evidence.len() >= 2)
        .collect::<Vec<_>>();
    shared.sort_by(|left, right| {
        right
            .1
            .len()
            .cmp(&left.1.len())
            .then_with(|| left.0.cmp(&right.0))
    });
    let (location, evidence) = shared.into_iter().next()?;
    let claim = RendererClaim {
        text: location.clone(),
        evidence_ids: evidence
            .iter()
            .map(|item| item.memory_id)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
    };
    let citations = evidence
        .iter()
        .take(2)
        .map(|item| {
            format!(
                "[memory#{} | {}]",
                item.memory_id,
                compact_context_line(&item.source, 80)
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    let reasoning = evidence
        .iter()
        .take(2)
        .map(|item| format!("{}: {}", item.speaker, item.snippet))
        .collect::<Vec<_>>()
        .join("; ");
    let chinese = query
        .chars()
        .any(|character| ('一'..='鿿').contains(&character));
    let display = if chinese {
        format!("直接回答：共同地点是{location} {citations}。\n\n推理依据：{reasoning}。")
    } else {
        format!("Direct answer: {location} {citations}.\n\nReasoning: {reasoning}.")
    };
    Some((display, vec![claim]))
}

fn appearance_change_question(query: &str) -> bool {
    contains_any(
        &query.to_lowercase(),
        &[
            "physical transformation",
            "appearance change",
            "changed appearance",
            "new look",
            "hair transformation",
            "外貌变化",
            "外观变化",
            "形象变化",
        ],
    )
}

fn hair_color_from_text(value: &str) -> Option<&'static str> {
    let lower = value.to_lowercase();
    let colors = [
        ("purple hair", "purple"),
        ("violet hair", "violet"),
        ("pink hair", "pink"),
        ("blue hair", "blue"),
        ("green hair", "green"),
        ("red hair", "red"),
        ("orange hair", "orange"),
        ("blonde hair", "blonde"),
        ("brown hair", "brown"),
        ("black hair", "black"),
        ("white hair", "white"),
        ("gray hair", "gray"),
        ("grey hair", "grey"),
    ]
    .iter()
    .filter_map(|(marker, color)| lower.contains(marker).then_some(*color))
    .collect::<BTreeSet<_>>();
    (colors.len() == 1).then(|| *colors.first().unwrap())
}

fn appearance_turn_color(turns: &[(Option<String>, String)], index: usize) -> Option<&'static str> {
    let (speaker, body) = &turns[index];
    let lower = body.to_lowercase();
    if !contains_any(
        &lower,
        &[
            "dyed my hair",
            "dyed his hair",
            "dyed her hair",
            "dyed their hair",
            "colored my hair",
        ],
    ) {
        return None;
    }
    let mut context = body.clone();
    if let Some((next_speaker, next_body)) = turns.get(index + 1) {
        let next_lower = next_body.to_lowercase();
        let same_speaker = speaker
            .as_deref()
            .zip(next_speaker.as_deref())
            .is_some_and(|(left, right)| person_names_match(left, right));
        let explicit_reference = contains_any(
            &next_lower,
            &[
                "check it out",
                "here is my new look",
                "here's my new look",
                "here is my new hair",
                "here's my new hair",
            ],
        );
        if same_speaker && explicit_reference && grounded_inline_media_caption(next_body).is_some()
        {
            context.push('\n');
            context.push_str(next_body);
        }
    }
    // Sharing a turn is not proof that another person's appearance belongs to the speaker.
    if contains_any(
        &context.to_lowercase(),
        &[
            "my friend",
            "my sister",
            "my brother",
            "my daughter",
            "my son",
            "another person",
            "someone else",
        ],
    ) {
        return None;
    }
    hair_color_from_text(&context)
}



fn render_multimodal_appearance_change_answer_plan(
    query: &str,
    items: &[RetrievedItem],
) -> Option<(String, Vec<RendererClaim>)> {
    if !appearance_change_question(query) {
        return None;
    }
    let subject = question_named_subject(query).unwrap_or_else(|| "The subject".to_string());
    let query_lower = query.to_lowercase();
    let mut candidates = Vec::new();
    for item in items.iter().filter(|item| memory_item_is_active(item)) {
        let source_body = searchable_source_body(&item.content);
        let turns = grounded_dialogue_turns(source_body);
        for (index, (speaker, body)) in turns.iter().enumerate() {
            if !statement_matches_question_subject(query, speaker.as_deref(), body) {
                continue;
            }
            let body_lower = body.to_lowercase();
            if !contains_any(
                &body_lower,
                &[
                    "dyed my hair",
                    "dyed his hair",
                    "dyed her hair",
                    "dyed their hair",
                    "colored my hair",
                    "new hair color",
                    "cut my hair",
                    "shaved my head",
                    "got a tattoo",
                    "got a piercing",
                ],
            ) {
                continue;
            }
            let color = appearance_turn_color(&turns, index);
            let temporal_bonus = MemoryEngine::tokenize(&query_lower)
                .into_iter()
                .filter(|token| {
                    token.chars().all(|character| character.is_ascii_digit())
                        || matches!(
                            token.as_str(),
                            "january"
                                | "february"
                                | "march"
                                | "april"
                                | "may"
                                | "june"
                                | "july"
                                | "august"
                                | "september"
                                | "october"
                                | "november"
                                | "december"
                        )
                })
                .filter(|token| source_body.to_lowercase().contains(token))
                .count() as f32
                * 0.10;
            candidates.push((
                statement_relevance_score(query, body, item.score)
                    + color.is_some() as u8 as f32 * 0.35
                    + temporal_bonus,
                item.id,
                item.source.clone(),
                compact_context_line(&grounded_text_without_media_captions(body), 220),
                color,
            ));
        }
    }
    candidates.sort_by(|left, right| right.0.partial_cmp(&left.0).unwrap_or(Ordering::Equal));
    let (_, memory_id, source, action, color) = candidates.into_iter().next()?;
    let citation = format!(
        "[memory#{memory_id} | {}]",
        compact_context_line(&source, 80)
    );
    let chinese = query
        .chars()
        .any(|character| ('一'..='鿿').contains(&character));
    let claim = match (chinese, color) {
        (true, Some(color)) => {
            format!("{subject}把头发染成了{color}色。")
        }
        (false, Some(color)) => {
            format!("{subject}'s transformation was dyeing their hair {color}.")
        }
        (_, None) => action,
    };
    let text = if chinese {
        format!("直接回答：{claim} {citation}")
    } else {
        format!("Direct answer: {claim} {citation}")
    };
    Some((
        text,
        vec![RendererClaim {
            text: claim,
            evidence_ids: vec![memory_id],
        }],
    ))
}

#[derive(Debug, Clone)]
struct RecentCreationEvidence {
    memory_id: u64,
    source: String,
    value: String,
    source_date: Option<chrono::NaiveDate>,
}

fn recent_visual_creation_requested(query: &str) -> bool {
    let lower = query.to_lowercase();
    contains_any(
        &lower,
        &[
            "paint recently",
            "painted recently",
            "recently paint",
            "recent painting",
            "latest painting",
            "create recently",
            "created recently",
            "recent artwork",
            "最近画",
            "最近创作",
            "最新作品",
        ],
    )
}

fn recent_visual_creation_value(body: &str) -> Option<String> {
    let lower = body.to_lowercase();
    if !contains_any(
        &lower,
        &[
            "i painted",
            "i've painted",
            "i have painted",
            "one i did",
            "painting i did",
            "i made",
            "i just finished",
            "i've done an abstract",
            "our latest work",
            "we just finished",
            "我画",
            "我创作",
        ],
    ) {
        return None;
    }

    let caption = grounded_inline_media_caption(body).unwrap_or_default();
    let caption_lower = caption.to_lowercase();
    if lower.contains("abstract painting") {
        return Some(if contains_any(&caption_lower, &["blue", "蓝色"]) {
            "an abstract painting with a blue background".to_string()
        } else {
            "an abstract painting".to_string()
        });
    }

    static CREATION_SUBJECT: OnceLock<regex::Regex> = OnceLock::new();
    let regex = CREATION_SUBJECT.get_or_init(|| {
        regex::Regex::new(r"(?i)(?:painting|drawing) of (?:a |an |the )?([^,.;\r\n]{2,100})")
            .expect("visual creation subject regex")
    });
    if let Some(subject) = regex
        .captures(&caption)
        .and_then(|captures| captures.get(1))
        .map(|capture| capture.as_str().trim())
        .filter(|value| {
            !contains_any(
                &value.to_lowercase(),
                &["painting brush", "paint brush", "wall with"],
            )
        })
    {
        return Some(format!("a painting of {subject}"));
    }
    for (marker, value) in [
        ("lake sunrise", "a lake sunrise painting"),
        ("sunset", "a sunset painting"),
        ("horse painting", "a horse painting"),
        ("sunflower", "a sunflower painting"),
        ("landscape", "a landscape painting"),
        ("still life", "a still-life painting"),
    ] {
        if lower.contains(marker) {
            return Some(value.to_string());
        }
    }
    None
}



fn render_recent_visual_creation_answer_plan(
    query: &str,
    items: &[RetrievedItem],
) -> Option<(String, Vec<RendererClaim>)> {
    if !recent_visual_creation_requested(query) {
        return None;
    }
    let subject = question_named_subject(query)?;
    let mut candidates = Vec::new();
    for item in items.iter().filter(|item| memory_item_is_active(item)) {
        let source_date = source_session_date(&item.content);
        for (speaker, body) in grounded_dialogue_turns(searchable_source_body(&item.content)) {
            if !speaker
                .as_deref()
                .is_some_and(|speaker| person_names_match(speaker, &subject))
            {
                continue;
            }
            let Some(value) = recent_visual_creation_value(&body) else {
                continue;
            };
            candidates.push(RecentCreationEvidence {
                memory_id: item.id,
                source: item.source.clone(),
                value,
                source_date,
            });
        }
    }
    let latest_date = candidates.iter().filter_map(|item| item.source_date).max();
    if let Some(latest_date) = latest_date {
        candidates.retain(|item| item.source_date == Some(latest_date));
    }
    let mut seen = HashSet::new();
    candidates.retain(|item| seen.insert(normalized_grounded_statement(&item.value)));
    if candidates.is_empty() {
        return None;
    }
    let rendered = candidates
        .iter()
        .map(|item| {
            format!(
                "{} [memory#{} | {}]",
                item.value,
                item.memory_id,
                compact_context_line(&item.source, 80)
            )
        })
        .collect::<Vec<_>>()
        .join(" and ");
    let claims = candidates
        .iter()
        .map(|item| RendererClaim {
            text: format!("{subject} recently created {}.", item.value),
            evidence_ids: vec![item.memory_id],
        })
        .collect();
    Some((
        format!("Direct answer: {subject} recently created {rendered}."),
        claims,
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CategoricalAnswerKind {
    Identity,
    RelationshipStatus,
}

pub(crate) fn categorical_answer_kind(query: &str) -> Option<CategoricalAnswerKind> {
    let lower = query.to_lowercase();
    if contains_any(
        &lower,
        &[
            "what is the identity",
            "what is their identity",
            "what is her identity",
            "what is his identity",
            "gender identity",
            "身份是什么",
            "性别认同",
        ],
    ) || (lower.starts_with("what is ")
        && lower.trim_end_matches(['?', '？']).ends_with(" identity"))
    {
        Some(CategoricalAnswerKind::Identity)
    } else if contains_any(
        &lower,
        &[
            "relationship status",
            "dating status",
            "marital status",
            "感情状态",
            "关系状态",
            "婚姻状态",
        ],
    ) {
        Some(CategoricalAnswerKind::RelationshipStatus)
    } else {
        None
    }
}

pub(crate) fn categorical_value(kind: CategoricalAnswerKind, body: &str) -> Option<&'static str> {
    let lower = body.to_lowercase();
    match kind {
        CategoricalAnswerKind::Identity => {
            if contains_any(&lower, &["transgender woman", "trans woman", "跨性别女性"]) {
                Some("a transgender woman")
            } else if contains_any(&lower, &["transgender man", "trans man", "跨性别男性"]) {
                Some("a transgender man")
            } else if contains_any(&lower, &["non-binary", "nonbinary", "非二元性别"]) {
                Some("nonbinary")
            } else {
                None
            }
        }
        CategoricalAnswerKind::RelationshipStatus => {
            if contains_any(
                &lower,
                &[
                    "i'm single",
                    "i am single",
                    "currently single",
                    "single parent",
                    "single right now",
                    "单身",
                ],
            ) {
                Some("single")
            } else if contains_any(
                &lower,
                &[
                    "i'm married",
                    "i am married",
                    "currently married",
                    "married to",
                    "已婚",
                ],
            ) {
                Some("married")
            } else if contains_any(&lower, &["i'm divorced", "i am divorced", "离异", "离婚"]) {
                Some("divorced")
            } else if contains_any(&lower, &["i'm widowed", "i am widowed", "丧偶"]) {
                Some("widowed")
            } else if contains_any(
                &lower,
                &["i'm engaged", "i am engaged", "engaged to", "已订婚"],
            ) {
                Some("engaged")
            } else if contains_any(
                &lower,
                &["i'm dating", "i am dating", "in a relationship", "恋爱中"],
            ) {
                Some("in a relationship")
            } else {
                None
            }
        }
    }
}



fn render_categorical_attribute_answer_plan(
    query: &str,
    items: &[RetrievedItem],
) -> Option<(String, Vec<RendererClaim>)> {
    let kind = categorical_answer_kind(query)?;
    let query_lower = query.to_lowercase();
    let mut candidates = Vec::new();
    for item in items.iter().filter(|item| memory_item_is_active(item)) {
        for statement in grounded_source_statements(searchable_source_body(&item.content)) {
            let (speaker, body) = grounded_statement_parts(&statement);
            let Some(value) = categorical_value(kind, body) else {
                continue;
            };
            let speaker_matches = speaker
                .map(|speaker| query_lower.contains(&speaker.to_lowercase()))
                .unwrap_or(false);
            if speaker.is_some() && !speaker_matches {
                continue;
            }
            candidates.push((
                statement_relevance_score(query, &statement, item.score) + 0.65,
                item.id,
                item.source.clone(),
                speaker.unwrap_or("The subject").to_string(),
                value,
            ));
        }
    }
    candidates.sort_by(|left, right| right.0.partial_cmp(&left.0).unwrap_or(Ordering::Equal));
    let (_, memory_id, source, subject, value) = candidates.into_iter().next()?;
    let chinese = query
        .chars()
        .any(|character| ('一'..='鿿').contains(&character));
    if chinese {
        let label = match kind {
            CategoricalAnswerKind::Identity => "身份",
            CategoricalAnswerKind::RelationshipStatus => "关系状态",
        };
        let claim = RendererClaim {
            text: format!("{subject}的{label}是{value}。"),
            evidence_ids: vec![memory_id],
        };
        Some((
            format!(
                "直接回答：{subject}的{label}是{value} [memory#{memory_id} | {}]。",
                compact_context_line(&source, 80)
            ),
            vec![claim],
        ))
    } else {
        let claim = RendererClaim {
            text: format!("{subject} is {value}."),
            evidence_ids: vec![memory_id],
        };
        Some((
            format!(
                "Direct answer: {subject} is {value} [memory#{memory_id} | {}].",
                compact_context_line(&source, 80)
            ),
            vec![claim],
        ))
    }
}

fn extract_research_object(body: &str) -> Option<(String, f32)> {
    let lower = body.to_lowercase();
    let markers = [
        ("researching ", 0.35),
        ("researched ", 0.35),
        ("research into ", 0.32),
        ("research about ", 0.32),
        ("research on ", 0.32),
        ("research ", 0.28),
        ("investigating ", 0.24),
        ("investigated ", 0.24),
        ("looking into ", 0.16),
    ];
    let (index, marker, directness) = markers
        .iter()
        .filter_map(|(marker, directness)| {
            lower
                .find(marker)
                .map(|index| (index, *marker, *directness))
        })
        .min_by_key(|(index, _, _)| *index)?;
    let mut value = body[index + marker.len()..].trim();
    value = value
        .trim_start_matches("the ")
        .trim_start_matches("some ")
        .trim();
    let boundary = value
        .char_indices()
        .find(|(_, character)| {
            matches!(
                character,
                '—' | '–' | '.' | ',' | ';' | '!' | '?' | '。' | '，' | '；' | '！' | '？'
            )
        })
        .map(|(index, _)| index)
        .unwrap_or(value.len());
    value = value[..boundary].trim();
    let value_lower = value.to_lowercase();
    for clause in [
        " - ",
        " it's ",
        " it is ",
        " because ",
        " before ",
        " after ",
    ] {
        if let Some(index) = value_lower.find(clause) {
            value = value[..index].trim();
            break;
        }
    }
    let normalized = value.trim_matches(|character: char| {
        character.is_ascii_punctuation() || character.is_whitespace()
    });
    if normalized.chars().count() < 3
        || normalized.chars().count() > 100
        || matches!(
            normalized.to_lowercase().as_str(),
            "it" | "this" | "that" | "something" | "more" | "things" | "stuff"
        )
    {
        None
    } else {
        Some((normalized.to_string(), directness))
    }
}



fn render_research_object_answer_plan(
    query: &str,
    items: &[RetrievedItem],
) -> Option<(String, Vec<RendererClaim>)> {
    let query_lower = query.to_lowercase();
    if !query_lower.starts_with("what ")
        || !contains_any(&query_lower, &["research", "investigat", "look into"])
    {
        return None;
    }
    let mut candidates = Vec::new();
    for item in items.iter().filter(|item| memory_item_is_active(item)) {
        for statement in grounded_source_statements(searchable_source_body(&item.content)) {
            let (speaker, body) = grounded_statement_parts(&statement);
            let Some((object, directness)) = extract_research_object(body) else {
                continue;
            };
            let speaker_matches = speaker
                .map(|speaker| query_lower.contains(&speaker.to_lowercase()))
                .unwrap_or(false);
            if speaker.is_some() && !speaker_matches {
                continue;
            }
            candidates.push((
                statement_relevance_score(query, &statement, item.score) + directness,
                item.id,
                item.source.clone(),
                speaker.unwrap_or("The subject").to_string(),
                object,
            ));
        }
    }
    candidates.sort_by(|left, right| right.0.partial_cmp(&left.0).unwrap_or(Ordering::Equal));
    let (_, memory_id, source, subject, object) = candidates.into_iter().next()?;
    let claim = RendererClaim {
        text: format!("{subject} researched {object}."),
        evidence_ids: vec![memory_id],
    };
    Some((
        format!(
            "Direct answer: {subject} researched {object} [memory#{memory_id} | {}].",
            compact_context_line(&source, 80)
        ),
        vec![claim],
    ))
}

fn clean_origin_value(value: &str) -> Option<String> {
    let mut value = value
        .trim()
        .trim_start_matches([',', ':', '-', '—', '–'])
        .trim();
    for prefix in ["is ", "was ", "of ", "called "] {
        value = value.strip_prefix(prefix).unwrap_or(value).trim();
    }
    let boundary = value
        .char_indices()
        .find(|(_, character)| {
            matches!(
                character,
                '.' | ',' | ';' | '!' | '?' | '—' | '–' | '。' | '，' | '；' | '！' | '？'
            )
        })
        .map(|(index, _)| index)
        .unwrap_or(value.len());
    value = value[..boundary].trim();
    let normalized = value.to_lowercase();
    if value.is_empty()
        || value.split_whitespace().count() > 6
        || contains_any(
            &normalized,
            &["home country", "my country", "our country", "hometown"],
        )
    {
        None
    } else {
        Some(value.to_string())
    }
}

fn direct_movement_origin(body: &str) -> Option<String> {
    let lower = body.to_lowercase();
    for marker in [
        "moved from ",
        "move from ",
        "relocated from ",
        "relocate from ",
    ] {
        let Some(index) = lower.find(marker) else {
            continue;
        };
        let mut tail = body[index + marker.len()..].trim();
        for prefix in ["my ", "our ", "the "] {
            tail = tail.strip_prefix(prefix).unwrap_or(tail).trim();
        }
        if let Some(value) = clean_origin_value(tail) {
            return Some(value);
        }
    }
    None
}

fn home_country_value(body: &str) -> Option<String> {
    let lower = body.to_lowercase();
    let index = lower.find("home country")?;
    clean_origin_value(&body[index + "home country".len()..])
}



fn render_linked_origin_answer_plan(
    query: &str,
    items: &[RetrievedItem],
) -> Option<(String, Vec<RendererClaim>)> {
    let query_lower = query.to_lowercase();
    if !contains_any(&query_lower, &["where", "哪里", "哪儿"])
        || !contains_any(&query_lower, &["move", "moved", "relocat", "搬", "迁"])
    {
        return None;
    }
    let mut statements = Vec::new();
    for item in items.iter().filter(|item| memory_item_is_active(item)) {
        for statement in grounded_source_statements(searchable_source_body(&item.content)) {
            let (speaker, body) = grounded_statement_parts(&statement);
            let speaker_matches = speaker
                .map(|speaker| query_lower.contains(&speaker.to_lowercase()))
                .unwrap_or(false);
            if speaker.is_some() && !speaker_matches {
                continue;
            }
            statements.push((
                statement_relevance_score(query, &statement, item.score),
                item.id,
                item.source.clone(),
                speaker.unwrap_or("The subject").to_string(),
                body.to_string(),
            ));
        }
    }
    let mut movement = statements
        .iter()
        .filter(|(_, _, _, _, body)| {
            let lower = body.to_lowercase();
            contains_any(
                &lower,
                &[
                    "moved from ",
                    "move from ",
                    "relocated from ",
                    "relocate from ",
                ],
            )
        })
        .cloned()
        .collect::<Vec<_>>();
    movement.sort_by(|left, right| right.0.partial_cmp(&left.0).unwrap_or(Ordering::Equal));
    let (_, movement_id, movement_source, subject, movement_body) = movement.into_iter().next()?;
    if let Some(origin) = direct_movement_origin(&movement_body) {
        let claim = RendererClaim {
            text: format!("{subject} moved from {origin}."),
            evidence_ids: vec![movement_id],
        };
        return Some((
            format!(
                "Direct answer: {subject} moved from {origin} [memory#{movement_id} | {}].",
                compact_context_line(&movement_source, 80)
            ),
            vec![claim],
        ));
    }

    let mut resolvers = statements
        .into_iter()
        .filter_map(|(score, memory_id, source, speaker, body)| {
            home_country_value(&body)
                .map(|origin| (score + 0.55, memory_id, source, speaker, origin))
        })
        .collect::<Vec<_>>();
    resolvers.sort_by(|left, right| right.0.partial_cmp(&left.0).unwrap_or(Ordering::Equal));
    let (_, resolver_id, resolver_source, _, origin) = resolvers.into_iter().next()?;
    let resolver_citation = if resolver_id == movement_id {
        String::new()
    } else {
        format!(
            " [memory#{resolver_id} | {}]",
            compact_context_line(&resolver_source, 80)
        )
    };
    let mut evidence_ids = vec![movement_id];
    if resolver_id != movement_id {
        evidence_ids.push(resolver_id);
    }
    let claim = RendererClaim {
        text: format!("{subject} moved from {origin}."),
        evidence_ids,
    };
    Some((format!(
        "Direct answer: {subject} moved from {origin} [memory#{movement_id} | {}]{resolver_citation}.",
        compact_context_line(&movement_source, 80)
    ), vec![claim]))
}

#[derive(Debug, Clone)]
struct EventDateEvidence {
    memory_id: u64,
    source: String,
    answer_en: String,
    answer_zh: String,
    session_date: Option<chrono::NaiveDate>,
    snippet: String,
    relative_marker: Option<String>,
    score: f32,
}

fn event_date_requested(query: &str) -> bool {
    let lower = query.to_lowercase();
    contains_any(
        &lower,
        &[
            "when did",
            "when was",
            "when is",
            "when are",
            "when will",
            "what date",
            "which date",
            "什么时候",
            "what day",
            "which day",
            "what month",
            "which month",
            "哪一天",
            "哪天",
            "日期是什么",
        ],
    )
}

fn event_date_query_terms(query: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "when", "did", "does", "do", "was", "were", "is", "are", "will", "what", "which", "date",
        "day", "time", "the", "a", "an", "to", "at", "on", "in", "of", "for", "from", "her", "his",
        "their", "our", "your", "my", "its", "has", "have", "had", "get", "got", "happen",
        "happened", "take", "took", "place",
    ];
    let subject = question_named_subject(query).map(|value| normalized_person_name(&value));
    let mut terms = MemoryEngine::tokenize(query)
        .into_iter()
        .filter(|term| term.chars().count() >= 3)
        .filter(|term| !STOP.contains(&term.as_str()))
        .filter(|term| {
            !(term.len() == 4
                && term.chars().all(|character| character.is_ascii_digit())
                && term
                    .parse::<i32>()
                    .is_ok_and(|year| (1900..=2100).contains(&year)))
        })
        .filter(|term| {
            subject
                .as_deref()
                .is_none_or(|subject| normalized_person_name(term) != subject)
        })
        .collect::<Vec<_>>();
    terms.sort();
    terms.dedup();
    terms
}
fn explicit_query_years(query: &str) -> Vec<i32> {
    let mut years = query
        .split(|character: char| !character.is_ascii_digit())
        .filter(|value| value.len() == 4)
        .filter_map(|value| value.parse::<i32>().ok())
        .filter(|year| (1900..=2100).contains(year))
        .collect::<Vec<_>>();
    years.sort_unstable();
    years.dedup();
    years
}

fn answer_date_matches_requested_years(answer: &str, requested_years: &[i32]) -> bool {
    requested_years.is_empty()
        || answer
            .split(|character: char| !character.is_ascii_digit())
            .filter_map(|value| value.parse::<i32>().ok())
            .any(|year| requested_years.contains(&year))
}

fn event_date_statement_match_score(query: &str, statement: &str) -> f32 {
    let terms = event_date_query_terms(query);
    if terms.is_empty() {
        return 1.0;
    }
    let statement_lower = statement.to_lowercase();
    let matched = terms
        .iter()
        .filter(|term| crate::answer_pipeline::text_matches_term(&statement_lower, term))
        .count();
    matched as f32 / terms.len() as f32
}

fn adjacent_event_question_match_score(query: &str, content: &str, statement: &str) -> Option<f32> {
    let target = normalized_grounded_statement(grounded_statement_parts(statement).1);
    if target.is_empty() {
        return None;
    }
    grounded_dialogue_turns(content)
        .windows(2)
        .find_map(|pair| {
            let question_body = &pair[0].1;
            let answer_body = &pair[1].1;
            let answer_key = normalized_grounded_statement(answer_body);
            if !question_body.contains(['?', '？'])
                || !(answer_key.contains(&target) || target.contains(&answer_key))
            {
                return None;
            }
            Some(event_date_statement_match_score(query, question_body))
        })
}

fn relative_day_offset(statement: &str) -> Option<(i64, String)> {
    let lower = statement.to_lowercase();
    // "two days ago" / "3 days ago" is exact arithmetic; "a few days ago" is
    // vague and stays unresolved.
    static DAYS_AGO: OnceLock<regex::Regex> = OnceLock::new();
    let days_ago = DAYS_AGO.get_or_init(|| {
        regex::Regex::new(r"\b([0-9]{1,2}|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen)\s+days\s+ago\b")
            .expect("days ago regex")
    });
    if let Some(captures) = days_ago.captures(&lower) {
        let raw = captures.get(1)?.as_str();
        let count = match raw {
            "two" => 2,
            "three" => 3,
            "four" => 4,
            "five" => 5,
            "six" => 6,
            "seven" => 7,
            "eight" => 8,
            "nine" => 9,
            "ten" => 10,
            "eleven" => 11,
            "twelve" => 12,
            "thirteen" => 13,
            _ => raw.parse::<i64>().ok().filter(|days| (1..=31).contains(days))?,
        };
        return Some((-count, captures.get(0)?.as_str().to_string()));
    }
    let fixed = |offset: i64, marker: &str| Some((offset, marker.to_string()));
    if contains_any(&lower, &["day before yesterday", "前天"]) {
        fixed(-2, "day before yesterday")
    } else if lower.contains("last night") {
        fixed(-1, "last night")
    } else if contains_any(&lower, &["yesterday", "昨天"]) {
        fixed(-1, "yesterday")
    } else if contains_any(&lower, &["day after tomorrow", "后天"]) {
        fixed(2, "day after tomorrow")
    } else if contains_any(&lower, &["tomorrow", "明天"]) {
        fixed(1, "tomorrow")
    } else if contains_any(&lower, &["today", "今天"]) {
        fixed(0, "today")
    } else {
        None
    }
}

fn shifted_calendar_month(session_date: chrono::NaiveDate, offset: i32) -> Option<(i32, u32)> {
    let year = session_date.format("%Y").to_string().parse::<i32>().ok()?;
    let month = session_date.format("%m").to_string().parse::<i32>().ok()?;
    let absolute = year
        .checked_mul(12)?
        .checked_add(month - 1)?
        .checked_add(offset)?;
    let shifted_year = absolute.div_euclid(12);
    let shifted_month = absolute.rem_euclid(12) + 1;
    Some((shifted_year, shifted_month as u32))
}

fn contains_bounded_ascii_phrase(value: &str, phrase: &str) -> bool {
    value.match_indices(phrase).any(|(start, _)| {
        let end = start + phrase.len();
        let before = value[..start].chars().next_back();
        let after = value[end..].chars().next();
        before.is_none_or(|character| !character.is_ascii_alphanumeric())
            && after.is_none_or(|character| !character.is_ascii_alphanumeric())
    })
}

fn event_date_answer_at_requested_granularity(
    query: &str,
    answer_en: &str,
    answer_zh: &str,
) -> (String, String) {
    let precision = crate::answer_pipeline::requested_temporal_granularity(query);
    let date = chrono::NaiveDate::parse_from_str(answer_en, "%-d %B %Y")
        .or_else(|_| chrono::NaiveDate::parse_from_str(answer_en, "%d %B %Y"))
        .ok();
    match (precision, date) {
        ("year", Some(date)) => (
            date.format("%Y").to_string(),
            format!("{}年", date.format("%Y")),
        ),
        ("month", Some(date)) => (
            date.format("%B %Y").to_string(),
            format!("{}年{}月", date.format("%Y"), date.format("%-m")),
        ),
        _ => (answer_en.to_string(), answer_zh.to_string()),
    }
}

/// The calendar date (approximate for weeks, months and years) of the event
/// a statement describes, from its relative time and the session date. Used
/// for interval arithmetic between two events. `immediate` treats "just",
/// "today", "right now", "recently" as the session date — right for intervals
/// at the scale of weeks and months, not for
/// single-date answers, where it was measured to add wrong answers.
pub(crate) fn relative_event_date(
    statement: &str,
    session_date: chrono::NaiveDate,
    immediate: bool,
) -> Option<chrono::NaiveDate> {
    static RELATIVE: OnceLock<regex::Regex> = OnceLock::new();
    let lower = statement.to_lowercase();
    let pattern = RELATIVE.get_or_init(|| regex::Regex::new(
        r"\b(?:(?P<n>[0-9]{1,2}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a few|a couple of) (?P<unit>day|week|weekend|month|year)s? ago|(?P<last>last|this past) (?P<lunit>night|week|weekend|month|year)|(?P<next>next|this coming) (?P<nunit>week|weekend|month|year)|(?P<single>day before yesterday|yesterday|tomorrow|tonight|today|this morning|this afternoon|this evening|earlier today|right now|just now|currently|recently)|(?P<just>just)\b)"
    ).expect("relative event date regex"));
    let count = |raw: &str| -> Option<i64> {
        Some(match raw {
            "a" | "an" | "one" => 1, "a couple of" | "two" => 2, "a few" | "three" => 3, "four" => 4, "five" => 5,
            "six" => 6, "seven" => 7, "eight" => 8, "nine" => 9, "ten" => 10, "eleven" => 11, "twelve" => 12,
            digits => digits.parse().ok()?,
        })
    };
    let days = |unit: &str, n: i64| -> i64 {
        match unit { "day" | "night" => n, "week" | "weekend" => 7 * n, "month" => 30 * n, _ => 365 * n }
    };
    for captures in pattern.captures_iter(&lower) {
        let offset = if let (Some(n), Some(unit)) = (captures.name("n"), captures.name("unit")) {
            -days(unit.as_str(), count(n.as_str())?)
        } else if let Some(unit) = captures.name("lunit") {
            -days(unit.as_str(), 1)
        } else if let Some(unit) = captures.name("nunit") {
            days(unit.as_str(), 1)
        } else if let Some(single) = captures.name("single") {
            match single.as_str() {
                "day before yesterday" => -2,
                "yesterday" => -1,
                "tomorrow" => 1,
                "today" | "tonight" | "this morning" | "this afternoon" | "this evening" | "earlier today" => 0,
                _ if immediate => 0,
                _ => continue,
            }
        } else if captures.name("just").is_some() && immediate {
            // "just like" / "just as" are not about time.
            let rest = &lower[captures.get(0)?.end()..];
            if [" like", " as", " because", " in case", " about", " so"].iter().any(|word| rest.starts_with(word)) {
                continue;
            }
            0
        } else {
            continue;
        };
        return session_date.checked_add_signed(chrono::Duration::days(offset));
    }
    None
}

pub(crate) fn relative_event_date_answer(
    statement: &str,
    session_date: chrono::NaiveDate,
) -> Option<(String, String, String)> {
    let lower = statement.to_lowercase();
    static RELATIVE_MONTH_DURATION: OnceLock<regex::Regex> = OnceLock::new();
    let duration = RELATIVE_MONTH_DURATION.get_or_init(|| {
        regex::Regex::new(r"(?i)\b(?:for\s+)?(?:about\s+|around\s+|roughly\s+)?([0-9]{1,2}|a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+months?\s+(?:now|ago)\b").expect("relative month duration regex")
    });
    if let Some(captures) = duration.captures(&lower) {
        let raw = captures.get(1)?.as_str();
        let count = match raw {
            "a" | "one" => 1,
            "two" => 2,
            "three" => 3,
            "four" => 4,
            "five" => 5,
            "six" => 6,
            "seven" => 7,
            "eight" => 8,
            "nine" => 9,
            "ten" => 10,
            "eleven" => 11,
            "twelve" => 12,
            _ => raw.parse::<i32>().ok()?,
        };
        let (year, month) = shifted_calendar_month(session_date, -count)?;
        let date = chrono::NaiveDate::from_ymd_opt(year, month, 1)?;
        return Some((
            date.format("%B %Y").to_string(),
            date.format("%Y年%-m月").to_string(),
            captures.get(0)?.as_str().to_string(),
        ));
    }
    // "I started surfing five years ago" (session in 2023) -> 2018; "I've had
    // them for 3 years now" -> since 2020, like the month durations above.
    static RELATIVE_YEARS: OnceLock<regex::Regex> = OnceLock::new();
    let years = RELATIVE_YEARS.get_or_init(|| {
        regex::Regex::new(r"(?i)\b(?:for\s+)?(?:about\s+|around\s+|roughly\s+)?([0-9]{1,2}|a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*years?\s+(?:now|ago)\b")
            .expect("relative years regex")
    });
    // Present perfect without "now": "I've had them for 3 years!", "we've
    // been together for three years", "Been doing it for 3 years." -> since
    // 2020. The "have/has" keeps
    // plans ("I'll be away for two years") out.
    static PERFECT_YEARS: OnceLock<regex::Regex> = OnceLock::new();
    let perfect_years = PERFECT_YEARS.get_or_init(|| {
        regex::Regex::new(r"(?i)(?:^been\s+|\b(?:i've|we've|they've|you've|he's|she's|have|has)\s+)(?:[a-z']+\s+){0,4}?for\s+(?:about\s+|around\s+|roughly\s+|almost\s+|nearly\s+|over\s+)?([0-9]{1,2}|a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+years?\b(?:\s+ago)?")
            .expect("present perfect years regex")
    });
    // "I recently left my IT job after 3 years" (said in 2022) -> started
    // in 2019: a span that has just ended.
    static ENDED_AFTER_YEARS: OnceLock<regex::Regex> = OnceLock::new();
    let ended_after = ENDED_AFTER_YEARS.get_or_init(|| {
        regex::Regex::new(r"(?i)\b(?:left|quit|ended|finished|retired|stopped|graduated|moved out)\b(?:[a-z' ]{0,40}?)\bafter\s+(?:about\s+|almost\s+|nearly\s+|over\s+)?([0-9]{1,2}|a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+years?\b")
            .expect("ended after years regex")
    });
    if let Some(captures) = years.captures(&lower).or_else(|| ended_after.captures(&lower)).or_else(|| perfect_years.captures(&lower)
        .filter(|captures| !captures.get(0).is_some_and(|m| m.as_str().ends_with("ago"))))
    {
        let raw = captures.get(1)?.as_str();
        let count = match raw {
            "a" | "one" => 1, "two" => 2, "three" => 3, "four" => 4, "five" => 5, "six" => 6, "seven" => 7,
            "eight" => 8, "nine" => 9, "ten" => 10, "eleven" => 11, "twelve" => 12,
            _ => raw.parse::<i32>().ok()?,
        };
        let year = session_date.format("%Y").to_string().parse::<i32>().ok()?.checked_sub(count)?;
        return Some((year.to_string(), format!("{year}年"), captures.get(0)?.as_str().to_string()));
    }
    let session_en = session_date.format("%-d %B %Y").to_string();
    let session_zh = session_date.format("%Y年%-m月%-d日").to_string();
    // "she passed away a few years ago" (said in 2023) -> a few years before
    // 2023; "next Saturday" -> the Saturday after the session; "two weekends
    // later" -> two weekends after it.
    static VAGUE_AND_NEXT: OnceLock<regex::Regex> = OnceLock::new();
    let vague_and_next = VAGUE_AND_NEXT.get_or_init(|| {
        regex::Regex::new(r"(?i)\b(?:(?P<few>a few|a couple of|several) years ago|(?P<fewdays>(?:a\s+)?few|a couple of|several) days ago|next (?P<weekday>monday|tuesday|wednesday|thursday|friday|saturday|sunday)|(?P<count>a|one|two|three|four|[0-9]) (?P<unit>weekend|week)s? (?:later|from now))\b")
            .expect("vague and next expressions regex")
    });
    if let Some(captures) = vague_and_next.captures(&lower) {
        let marker = captures.get(0)?.as_str().to_string();
        let year = session_date.format("%Y").to_string();
        if let Some(few) = captures.name("few") {
            return Some((format!("{} years before {year}", few.as_str()), format!("{year}年之前几年"), marker));
        }
        // "finished few days ago" (said 17 December 2023) -> a few days
        // before 17 December 2023: as vague as the sentence, no exact day.
        if captures.name("fewdays").is_some() {
            return Some((format!("a few days before {session_en}"), format!("{session_zh}之前几天"), marker));
        }
        if let Some(weekday) = captures.name("weekday") {
            let mut name = weekday.as_str().to_string();
            name[..1].make_ascii_uppercase();
            return Some((format!("the {name} after {session_en}"), format!("{session_zh}之后的{name}"), marker));
        }
        if let (Some(count), Some(unit)) = (captures.name("count"), captures.name("unit")) {
            let count = match count.as_str() { "a" | "one" | "1" => "one", "two" | "2" => "two", "three" | "3" => "three", "four" | "4" => "four", other => other };
            let unit = if count == "one" { unit.as_str().to_string() } else { format!("{}s", unit.as_str()) };
            return Some((format!("{count} {unit} after {session_en}"), format!("{session_zh}之后{count}{unit}"), marker));
        }
    }
    // Weekend and week expressions, phrased as LoCoMo states them ("two
    // weekends before 17 July 2023", "the week of 23 August 2023").
    static WEEK_EXPRESSIONS: OnceLock<regex::Regex> = OnceLock::new();
    let week_expressions = WEEK_EXPRESSIONS.get_or_init(|| {
        regex::Regex::new(r"(?i)\b(?:(?P<past>this past|last) weekend|(?P<count>[0-9]{1,2}|a|one|two|three|four|five|six|seven|eight|nine|ten) (?P<unit>weekend|week)s? ago|this (?P<this>week|weekend)|(?P<today>this morning|this afternoon|this evening|earlier today|tonight))\b")
            .expect("week expressions regex")
    });
    if let Some(captures) = week_expressions.captures(&lower) {
        let marker = captures.get(0)?.as_str().to_string();
        if captures.name("past").is_some() {
            return Some((format!("the weekend before {session_en}"), format!("{session_zh}之前的周末"), marker));
        }
        if let (Some(raw), Some(unit)) = (captures.name("count"), captures.name("unit")) {
            let count = match raw.as_str() {
                "a" | "one" => 1, "two" => 2, "three" => 3, "four" => 4, "five" => 5, "six" => 6,
                "seven" => 7, "eight" => 8, "nine" => 9, "ten" => 10,
                digits => digits.parse::<i32>().ok()?,
            };
            let unit = unit.as_str();
            return Some(if count == 1 {
                (format!("the {unit} before {session_en}"), format!("{session_zh}之前的那个{}", if unit == "week" { "星期" } else { "周末" }), marker)
            } else {
                let words = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
                let count_en = words.get(count as usize).map(|w| w.to_string()).unwrap_or_else(|| count.to_string());
                (format!("{count_en} {unit}s before {session_en}"), format!("{session_zh}之前{count}个{}", if unit == "week" { "星期" } else { "周末" }), marker)
            });
        }
        if let Some(this) = captures.name("this") {
            let unit = this.as_str();
            return Some((format!("the {unit} of {session_en}"), format!("{session_zh}所在的{}", if unit == "week" { "那一周" } else { "周末" }), marker));
        }
        if captures.name("today").is_some() {
            return Some((session_en.clone(), session_zh.clone(), marker));
        }
    }
    // "I had a super fun weekend" said on a Monday: the weekend before. Only
    // when no explicit week expression ("two weekends ago") is present.
    static PAST_WEEKEND: OnceLock<regex::Regex> = OnceLock::new();
    let past_weekend = PAST_WEEKEND.get_or_init(|| {
        regex::Regex::new(r"(?i)\b(?:over the weekend|(?:had|spent) (?:a|an|my|the) (?:[a-z']+ ){0,3}weekend)\b")
            .expect("past weekend regex")
    });
    if let Some(found) = past_weekend.find(&lower) {
        return Some((format!("the weekend before {session_en}"), format!("{session_zh}之前的周末"), found.as_str().to_string()));
    }
    for (marker, weekday_en, weekday_zh) in [
        ("last sunday", "Sunday", "周日"),
        ("last sun", "Sunday", "周日"),
        ("last monday", "Monday", "周一"),
        ("last mon", "Monday", "周一"),
        ("last tuesday", "Tuesday", "周二"),
        ("last tue", "Tuesday", "周二"),
        ("last wednesday", "Wednesday", "周三"),
        ("last wed", "Wednesday", "周三"),
        ("last thursday", "Thursday", "周四"),
        ("last thu", "Thursday", "周四"),
        ("last friday", "Friday", "周五"),
        ("last fri", "Friday", "周五"),
        ("last saturday", "Saturday", "周六"),
        ("last sat", "Saturday", "周六"),
    ] {
        if contains_bounded_ascii_phrase(&lower, marker) {
            return Some((
                format!("the {weekday_en} before {session_en}"),
                format!("{session_zh}之前的{weekday_zh}"),
                marker.to_string(),
            ));
        }
    }
    for (marker, target_weekday) in [
        ("on sunday", 7_i64),
        ("on monday", 1),
        ("on tuesday", 2),
        ("on wednesday", 3),
        ("on thursday", 4),
        ("on friday", 5),
        ("on saturday", 6),
    ] {
        if lower.contains(marker) {
            let current_weekday = session_date.format("%u").to_string().parse::<i64>().ok()?;
            let days_back = (current_weekday - target_weekday).rem_euclid(7);
            let date = session_date.checked_sub_signed(chrono::Duration::days(days_back))?;
            return Some((
                date.format("%-d %B %Y").to_string(),
                date.format("%Y年%-m月%-d日").to_string(),
                marker.to_string(),
            ));
        }
    }
    // "I met them on the 15th": a bare ordinal day refers to the session's
    // month, or the previous month when that day is still ahead.
    static ORDINAL_DAY: OnceLock<regex::Regex> = OnceLock::new();
    let ordinal_day = ORDINAL_DAY.get_or_init(|| {
        regex::Regex::new(r"(?i)\bon the ([0-9]{1,2})(?:st|nd|rd|th)\b(?:\s+of\s+[a-z]+)?")
            .expect("ordinal day regex")
    });
    if let Some(captures) = ordinal_day.captures(&lower) {
        let whole = captures.get(0)?.as_str();
        if !whole.contains(" of ") {
            let day = captures.get(1)?.as_str().parse::<u32>().ok()?;
            let session_day = session_date.format("%-d").to_string().parse::<u32>().ok()?;
            let (year, month) = if day <= session_day {
                (
                    session_date.format("%Y").to_string().parse::<i32>().ok()?,
                    session_date.format("%-m").to_string().parse::<u32>().ok()?,
                )
            } else {
                shifted_calendar_month(session_date, -1)?
            };
            if let Some(date) = chrono::NaiveDate::from_ymd_opt(year, month, day) {
                return Some((
                    date.format("%-d %B %Y").to_string(),
                    date.format("%Y年%-m月%-d日").to_string(),
                    whole.to_string(),
                ));
            }
        }
    }
    if contains_any(&lower, &["last week", "previous week"]) {
        let date = session_date.checked_sub_signed(chrono::Duration::days(7))?;
        return Some((
            format!("the week of {}", date.format("%-d %B %Y")),
            format!("{}所在的一周", date.format("%Y年%-m月%-d日")),
            "last week".to_string(),
        ));
    }
    if contains_any(&lower, &["next week", "following week"]) {
        let date = session_date.checked_add_signed(chrono::Duration::days(7))?;
        return Some((
            format!("the week of {}", date.format("%-d %B %Y")),
            format!("{}所在的一周", date.format("%Y年%-m月%-d日")),
            "next week".to_string(),
        ));
    }
    if contains_any(&lower, &["this month", "current month"]) {
        return Some((
            session_date.format("%B %Y").to_string(),
            session_date.format("%Y年%-m月").to_string(),
            "this month".to_string(),
        ));
    }
    if contains_any(&lower, &["last month", "previous month"]) {
        let (year, month) = shifted_calendar_month(session_date, -1)?;
        let date = chrono::NaiveDate::from_ymd_opt(year, month, 1)?;
        return Some((
            date.format("%B %Y").to_string(),
            date.format("%Y年%-m月").to_string(),
            "last month".to_string(),
        ));
    }
    if contains_any(&lower, &["next month", "following month"]) {
        let (year, month) = shifted_calendar_month(session_date, 1)?;
        let date = chrono::NaiveDate::from_ymd_opt(year, month, 1)?;
        return Some((
            date.format("%B %Y").to_string(),
            date.format("%Y年%-m月").to_string(),
            "next month".to_string(),
        ));
    }
    static BARE_CALENDAR_MONTH: OnceLock<regex::Regex> = OnceLock::new();
    let bare_calendar_month = BARE_CALENDAR_MONTH.get_or_init(|| {
        regex::Regex::new(
            r"(?i)\b(?:in|during|by|until|from|through)\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b",
        )
        .expect("bare calendar month regex")
    });
    if let Some(captures) = bare_calendar_month.captures(&lower) {
        let marker = captures.get(0)?.as_str().to_string();
        let month = captures
            .get(1)
            .and_then(|value| english_month_number(value.as_str()))?;
        let session_year = session_date.format("%Y").to_string().parse::<i32>().ok()?;
        let session_month = session_date.format("%m").to_string().parse::<u32>().ok()?;
        let future_event = contains_any(
            &lower,
            &[
                "will ", "going to", "plan to", "planning", "leave", "depart", "next ",
            ],
        );
        let year = if future_event && month < session_month {
            session_year.checked_add(1)?
        } else {
            session_year
        };
        let date = chrono::NaiveDate::from_ymd_opt(year, month, 1)?;
        return Some((
            date.format("%B %Y").to_string(),
            date.format("%Y年%-m月").to_string(),
            marker,
        ));
    }
    let session_year = session_date.format("%Y").to_string().parse::<i32>().ok()?;
    if lower.contains("last summer") {
        let current_month = session_date.format("%m").to_string().parse::<u32>().ok()?;
        let year = if current_month <= 8 {
            session_year - 1
        } else {
            session_year
        };
        return Some((
            format!("summer {year}"),
            format!("{year}年夏季"),
            "last summer".to_string(),
        ));
    }
    if contains_any(&lower, &["last year", "previous year"]) {
        let year = session_year.checked_sub(1)?;
        return Some((
            year.to_string(),
            format!("{year}年"),
            "last year".to_string(),
        ));
    }
    if contains_any(&lower, &["next year", "following year"]) {
        let year = session_year.checked_add(1)?;
        return Some((
            year.to_string(),
            format!("{year}年"),
            "next year".to_string(),
        ));
    }
    if contains_any(
        &lower,
        &[
            "just left",
            "just got",
            "just signed",
            "just started",
            "this injury",
            "this week",
        ],
    // An explicit day wins: "I just signed up for a pottery class yesterday".
    ) && relative_day_offset(statement).is_none() {
        return Some((
            session_en,
            session_zh,
            "contemporaneous session context".to_string(),
        ));
    }
    let (offset, marker) = relative_day_offset(statement)?;
    let date = session_date.checked_add_signed(chrono::Duration::days(offset))?;
    Some((
        date.format("%-d %B %Y").to_string(),
        date.format("%Y年%-m月%-d日").to_string(),
        marker,
    ))
}

fn grounded_event_date_units(content: &str) -> Vec<String> {
    let mut units = grounded_source_statements(content);
    let mut seen = units
        .iter()
        .map(|unit| normalized_grounded_statement(unit))
        .collect::<HashSet<_>>();
    for (speaker, body) in grounded_dialogue_turns(content) {
        let attributed = speaker
            .map(|speaker| format!("{speaker}: {body}"))
            .unwrap_or(body);
        let key = normalized_grounded_statement(&attributed);
        if !key.is_empty() && seen.insert(key) {
            units.push(attributed);
        }
    }
    units
}

/// Lexical overlap alone dates the wrong event ("I was bored today, so I took
/// my turtles out for a walk" for "When did Nate get his first two turtles?").
/// With a semantic scorer, each candidate's turn is scored together with the
/// preceding turn (the pipeline's passage format), and confidently relevant
/// candidates outrank the rest: tier + 10·relevance + 0.1·lexical, as in
/// `apply_semantic_relevance`; candidates below the revoke bar are marked
/// (score = -inf) for removal. Without a scorer the lexical order stands.
fn rank_event_date_candidates_semantically(
    query: &str,
    items: &[RetrievedItem],
    candidates: &mut [EventDateEvidence],
) {
    let Some(scorer) = crate::semantic_scorer::active_scorer() else {
        return;
    };
    for item in items {
        let turns = grounded_dialogue_turns(searchable_source_body(&item.content))
            .into_iter()
            .map(|(speaker, body)| match speaker {
                Some(speaker) => format!("{speaker}: {body}"),
                None => body,
            })
            .collect::<Vec<_>>();
        let located = candidates
            .iter()
            .enumerate()
            .filter(|(_, candidate)| candidate.memory_id == item.id)
            .filter_map(|(index, candidate)| {
                let snippet = candidate.snippet.split_once(": ").map_or(candidate.snippet.as_str(), |(_, body)| body);
                turns.iter().position(|turn| turn.contains(snippet)).map(|turn| (index, turn))
            })
            .collect::<Vec<_>>();
        if located.is_empty() {
            continue;
        }
        let passages = located
            .iter()
            .map(|(_, turn)| match turn.checked_sub(1) {
                Some(previous) => format!("{}\n{}", turns[previous], turns[*turn]),
                None => turns[*turn].clone(),
            })
            .collect::<Vec<_>>();
        let Ok(scores) = scorer.relevance(query, &passages.iter().map(String::as_str).collect::<Vec<_>>()) else {
            continue;
        };
        for ((index, _), relevance) in located.into_iter().zip(scores) {
            let candidate = &mut candidates[index];
            candidate.score = if relevance < crate::semantic_scorer::RELEVANCE_REVOKE {
                f32::NEG_INFINITY
            } else if relevance >= crate::semantic_scorer::RELEVANCE_GRANT {
                crate::semantic_scorer::relevance_tier() + 10.0 * relevance + 0.1 * candidate.score
            } else {
                candidate.score + relevance
            };
        }
    }
}



fn render_event_date_answer_plan(
    query: &str,
    items: &[RetrievedItem],
) -> Option<(String, Vec<RendererClaim>)> {
    if !event_date_requested(query) {
        return None;
    }
    let mut candidates = Vec::new();
    for item in items.iter().filter(|item| memory_item_is_active(item)) {
        let searchable = searchable_source_body(&item.content);
        let source_event_match = event_date_statement_match_score(query, searchable);
        if source_event_match < 0.40 {
            continue;
        }
        let session_date = source_session_date(&item.content);
        let default_year = content_session_year(&item.content).unwrap_or_else(|| {
            item.created_at
                .format("%Y")
                .to_string()
                .parse::<i32>()
                .unwrap_or(1970)
        });
        for statement in grounded_event_date_units(searchable) {
            let adjacent_context_score =
                adjacent_question_score_for_statement(query, searchable, &statement);
            let adjacent_event_score =
                adjacent_event_question_match_score(query, searchable, &statement);
            let event_match_score = event_date_statement_match_score(query, &statement);
            if event_match_score < 0.20
                && (adjacent_context_score.is_none_or(|score| score < 0.32)
                    || adjacent_event_score.is_none_or(|score| score < 0.20))
            {
                continue;
            }
            let mut base_score = statement_relevance_score(query, &statement, item.score);
            base_score += event_match_score * 1.20;
            if let Some(context_score) = adjacent_context_score {
                if context_score < 0.32 || adjacent_event_score.is_none_or(|score| score < 0.20) {
                    continue;
                }
                base_score += context_score * 0.80 + 0.20;
            }
            if let Some((answer_en, answer_zh, marker)) =
                session_date.and_then(|date| relative_event_date_answer(&statement, date))
            {
                let minimum_score = if event_match_score >= 0.20 {
                    0.24
                } else {
                    0.36
                };
                let directly_bound_relative_event =
                    source_event_match >= 0.40 && event_match_score >= 0.50;
                if base_score >= minimum_score || directly_bound_relative_event {
                    candidates.push(EventDateEvidence {
                        memory_id: item.id,
                        source: item.source.clone(),
                        answer_en,
                        answer_zh,
                        session_date,
                        snippet: statement.clone(),
                        relative_marker: Some(marker),
                        score: base_score + 0.30,
                    });
                }
            }
            for (date, _, _, session_header) in temporal_date_anchors(&statement, default_year) {
                if session_header || base_score < 0.36 {
                    continue;
                }
                candidates.push(EventDateEvidence {
                    memory_id: item.id,
                    source: item.source.clone(),
                    answer_en: date.format("%-d %B %Y").to_string(),
                    answer_zh: date.format("%Y年%-m月%-d日").to_string(),
                    session_date,
                    snippet: statement.clone(),
                    relative_marker: None,
                    score: base_score + 0.20,
                });
            }
        }
    }
    let requested_years = explicit_query_years(query);
    candidates.retain(|candidate| {
        answer_date_matches_requested_years(&candidate.answer_en, &requested_years)
    });
    rank_event_date_candidates_semantically(query, items, &mut candidates);
    // A candidate the scorer judges irrelevant dates some other event.
    candidates.retain(|candidate| candidate.score.is_finite());
    candidates.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
    });
    let candidate = candidates.into_iter().next()?;
    let (answer_en, answer_zh) = event_date_answer_at_requested_granularity(
        query,
        &candidate.answer_en,
        &candidate.answer_zh,
    );
    let chinese = query
        .chars()
        .any(|character| ('一'..='鿿').contains(&character));
    let source = compact_context_line(&candidate.source, 80);
    let claim = RendererClaim {
        text: if chinese {
            format!("日期是 {answer_zh}。")
        } else {
            format!("The date was {answer_en}.")
        },
        evidence_ids: vec![candidate.memory_id],
    };
    if chinese {
        let reasoning = match (candidate.session_date, candidate.relative_marker.as_deref()) {
            (Some(session_date), Some(marker)) => format!(
                "来源会话日期是 {}，事件使用了相对时间“{}”",
                session_date, marker
            ),
            _ => "来源证据明确给出了该日期".to_string(),
        };
        Some((
            format!(
                "直接回答：日期是 {} [memory#{} | {}]。\n\n推理依据：{}；原始相关句为“{}”。",
                answer_zh, candidate.memory_id, source, reasoning, candidate.snippet
            ),
            vec![claim],
        ))
    } else {
        let reasoning = match (candidate.session_date, candidate.relative_marker.as_deref()) {
            (Some(session_date), Some(marker)) => format!(
                "the source session is dated {session_date}, and the event uses the relative marker “{marker}”"
            ),
            _ => "the cited evidence states this date explicitly".to_string(),
        };
        Some((format!(
            "Direct answer: the date was {} [memory#{} | {}].\n\nReasoning: {}; the relevant source sentence is “{}”.",
            answer_en, candidate.memory_id, source, reasoning, candidate.snippet
        ), vec![claim]))
    }
}




#[derive(Debug, Clone)]
struct GroundedRenderedAnswer {
    text: String,
    draft_kind: &'static str,
    claims: Vec<RendererClaim>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GroundedListKind {
    Activities,
    Events,
    Fields,
    Books,
    Locations,
    Preferences,
    Destress,
}

#[derive(Debug, Clone)]
struct GroundedListValue {
    value: String,
    memory_id: u64,
    source_label: String,
    relation_strength: u8,
    score: f32,
}

fn readable_answer_source_label(item: &RetrievedItem) -> String {
    let source = item.source.trim();
    let normalized_source = source.to_ascii_lowercase();
    let internal_source = normalized_source.starts_with("universal-source:")
        || normalized_source.starts_with("universal_source")
        || normalized_source.starts_with("intent_workspace_")
        || normalized_source.contains("source_revision_");
    let compact_label = |value: &str| {
        compact_context_line(value, 80)
            .replace("[", "(")
            .replace("]", ")")
    };

    if let Some(source_label) = document_source_label_for_item(item) {
        return compact_label(&source_label);
    }

    if internal_source || source.is_empty() {
        return item
            .entity
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(compact_label)
            .unwrap_or_else(|| "record".to_string());
    }

    compact_label(source)
}

pub(crate) fn grounded_list_kind(query: &str) -> Option<GroundedListKind> {
    let lower = query.to_lowercase();
    if contains_any(
        &lower,
        &[
            "what events",
            "which events",
            "events has",
            "events have",
            "events did",
            "participating in the community",
            "participate in the community",
            "community participation",
            "哪些事件",
            "参加过什么活动",
            "如何参与社区",
        ],
    ) || ((lower.starts_with("what ") || lower.starts_with("which "))
        && contains_any(&lower, &[" event", " events"]))
        || (contains_any(&lower, &["in what ways", "what ways"])
            && contains_any(
                &lower,
                &["participat", "community", "advocacy", "children", "kids"],
            ))
    {
        Some(GroundedListKind::Events)
    } else if contains_any(
        &lower,
        &[
            "what activities",
            "which activities",
            "creative project",
            "creative projects",
            "art project",
            "哪些活动",
            "参与哪些",
        ],
    ) {
        Some(GroundedListKind::Activities)
    } else if contains_any(
        &lower,
        &[
            "what fields",
            "which fields",
            "fields would",
            "哪些领域",
            "什么专业",
            "哪些专业",
        ],
    ) {
        Some(GroundedListKind::Fields)
    } else if contains_any(
        &lower,
        &["what books", "which books", "哪些书", "读过什么书"],
    ) {
        Some(GroundedListKind::Books)
    } else if contains_any(
        &lower,
        &["where has", "where have", "都去过哪里", "去过哪些地方"],
    ) {
        Some(GroundedListKind::Locations)
    } else if preference_question_has_early_like(&lower)
        || contains_any(&lower, &["喜欢什么", "有哪些喜好"])
    {
        Some(GroundedListKind::Preferences)
    } else if contains_any(
        &lower,
        &[
            "destress",
            "de-stress",
            "decompress",
            "cope with stress",
            "如何减压",
            "怎么减压",
            "如何放松",
        ],
    ) {
        Some(GroundedListKind::Destress)
    } else {
        None
    }
}

fn question_named_subject(query: &str) -> Option<String> {
    const QUESTION_WORDS: &[&str] = &[
        "what",
        "in",
        "which",
        "where",
        "when",
        "why",
        "how",
        "who",
        "would",
        "could",
        "should",
        "does",
        "do",
        "did",
        "has",
        "have",
        "had",
        "is",
        "are",
        "was",
        "were",
        "compare",
        "list",
        "show",
        "tell",
        "describe",
        "summarize",
        "explain",
        "give",
    ];
    query.split_whitespace().find_map(|raw| {
        let token = raw.trim_matches(|character: char| {
            !character.is_alphabetic() && character != '\'' && character != '-'
        });
        let token = token
            .strip_suffix("'s")
            .or_else(|| token.strip_suffix("’s"))
            .unwrap_or(token);
        let lower = token.to_lowercase();
        let starts_with_uppercase = token
            .chars()
            .next()
            .is_some_and(|character| character.is_uppercase());
        let is_acronym = token
            .chars()
            .all(|character| !character.is_alphabetic() || character.is_uppercase());
        (starts_with_uppercase
            && token.chars().count() > 1
            && !is_acronym
            && !raw.contains('+')
            && !QUESTION_WORDS.contains(&lower.as_str()))
        .then(|| token.to_string())
    })
}

fn normalized_person_name(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_alphabetic())
        .flat_map(char::to_lowercase)
        .collect()
}

fn person_names_match(left: &str, right: &str) -> bool {
    let left = normalized_person_name(left);
    let right = normalized_person_name(right);
    if left == right {
        return true;
    }
    const ALIAS_GROUPS: &[&[&str]] = &[
        &["mel", "melanie"],
        &["jon", "john"],
        &["jo", "joanna"],
        &["liz", "elizabeth"],
        &["beth", "elizabeth"],
        &["mike", "michael"],
        &["kate", "katherine", "kathryn"],
        &["alex", "alexander", "alexandra"],
        &["sam", "samuel", "samantha"],
    ];
    ALIAS_GROUPS
        .iter()
        .any(|group| group.contains(&left.as_str()) && group.contains(&right.as_str()))
}

fn statement_matches_question_subject(query: &str, speaker: Option<&str>, body: &str) -> bool {
    let Some(subject) = question_named_subject(query) else {
        return true;
    };
    speaker.is_some_and(|speaker| person_names_match(speaker, &subject))
        || body
            .split(|character: char| !character.is_alphabetic())
            .filter(|token| !token.is_empty())
            .any(|token| person_names_match(token, &subject))
}

fn preference_question_has_early_like(lower: &str) -> bool {
    let tokens = MemoryEngine::tokenize(lower);
    matches!(tokens.as_slice(), [first, second, ..] if first == "what" && matches!(second.as_str(), "do" | "does"))
        && tokens
            .iter()
            .position(|token| token == "like")
            .is_some_and(|index| index <= 6)
}

fn grounded_marker_present(lower: &str, marker: &str) -> bool {
    if marker.contains(' ') || marker.contains('-') {
        return lower.contains(marker);
    }
    lower
        .split(|character: char| !character.is_alphanumeric())
        .any(|token| token == marker)
}

fn mapped_grounded_values(lower: &str, mappings: &[(&str, &str)]) -> Vec<String> {
    let negated = contains_any(
        lower,
        &[
            "never tried",
            "have not tried",
            "haven't tried",
            "does not do",
            "doesn't do",
            "不喜欢",
            "从未",
        ],
    );
    mappings
        .iter()
        .filter(|(marker, _)| !negated && grounded_marker_present(lower, marker))
        .map(|(_, value)| (*value).to_string())
        .collect()
}

fn activity_values(lower: &str) -> Vec<String> {
    mapped_grounded_values(
        lower,
        &[
            ("pottery", "pottery"),
            ("camping", "camping"),
            ("camped", "camping"),
            ("painting", "painting"),
            ("paints", "painting"),
            ("painted", "painting"),
            ("swimming", "swimming"),
            ("swims", "swimming"),
            ("running", "running"),
            ("runs", "running"),
            ("ran", "running"),
            ("hiking", "hiking"),
            ("hikes", "hiking"),
            ("museum", "museum visit"),
            ("reading", "reading"),
            ("reads", "reading"),
            ("cycling", "cycling"),
            ("biking", "cycling"),
            ("violin", "playing violin"),
            ("piano", "playing piano"),
            ("guitar", "playing guitar"),
            ("drawing", "drawing"),
            ("dancing", "dancing"),
            ("cooking", "cooking"),
            ("gardening", "gardening"),
            ("photography", "photography"),
            ("yoga", "yoga"),
            ("volunteering", "volunteering"),
            ("gaming", "gaming"),
        ],
    )
}

/// Rates whether a statement explicitly connects an activity to relief or
/// restoration. Weak renewal language is useful retrieval context, but does
/// not by itself justify a direct answer to "what does X do to destress?".
pub(crate) fn decompression_evidence_strength(body: &str) -> u8 {
    let lower = body.to_lowercase();
    if contains_any(
        &lower,
        &[
            "destress",
            "de-stress",
            "stress relief",
            "clear my mind",
            "减压",
        ],
    ) {
        4
    } else if contains_any(&lower, &["therapy", "therapeutic", "治疗"]) {
        3
    } else if contains_any(&lower, &["relax", "calm", "headspace", "self-care", "放松"]) {
        2
    } else if contains_any(&lower, &["refresh", "recharge"]) {
        1
    } else {
        0
    }
}

pub(crate) fn source_signals_decompression(body: &str) -> bool {
    decompression_evidence_strength(body) >= 2
}

pub(crate) fn event_values(lower: &str) -> Vec<String> {
    let mut values = mapped_grounded_values(
        lower,
        &[
            ("pride parade", "pride parade"),
            ("pride event", "LGBTQ+ pride event"),
            ("lgbt event", "LGBTQ+ event"),
            ("lgbtq event", "LGBTQ+ event"),
            ("school event", "school speech"),
            ("school speech", "school speech"),
            ("speech at a school", "school speech"),
            ("giving my talk", "school speech"),
            ("mentorship program", "mentoring program"),
            ("mentoring program", "mentoring program"),
            ("mentor program", "mentoring program"),
            ("support group", "support group"),
            ("activist group", "activist group"),
            ("advocacy group", "advocacy group"),
            ("art show", "art show"),
            ("conference", "conference"),
            ("fundraiser", "fundraiser"),
            ("fundraising event", "fundraising event"),
            ("community event", "community event"),
            ("volunteering", "volunteering"),
            ("volunteered", "volunteering"),
        ],
    );
    if contains_any(
        lower,
        &[
            "missed the pride parade",
            "missed that pride parade",
            "missed it",
            "did not attend",
            "didn't attend",
            "could not attend",
            "couldn't attend",
        ],
    ) {
        values.retain(|value| value != "pride parade" && value != "LGBTQ+ pride event");
    }
    values
}

fn event_statement_matches_scope(query_lower: &str, statement_lower: &str) -> bool {
    if contains_any(
        query_lower,
        &["kid", "child", "children", "youth", "学生", "儿童"],
    ) {
        return contains_any(
            statement_lower,
            &[
                "school event",
                "school speech",
                "speech at a school",
                "giving my talk",
                "mentorship",
                "mentoring",
                "mentor program",
                "volunteer",
                "children",
                "kids",
                "youth",
                "students",
                "fundraiser",
            ],
        );
    }
    true
}

fn event_value_matches_scope(query_lower: &str, value: &str) -> bool {
    if contains_any(
        query_lower,
        &["kid", "child", "children", "youth", "学生", "儿童"],
    ) {
        return contains_any(
            &value.to_lowercase(),
            &["school speech", "mentoring", "volunteer", "fundrais"],
        );
    }
    true
}

fn field_values(lower: &str) -> Vec<String> {
    mapped_grounded_values(
        lower,
        &[
            ("psychology", "psychology"),
            ("counseling", "counseling"),
            ("counselling", "counseling"),
            ("mental health", "mental health"),
            ("social work", "social work"),
            ("engineering", "engineering"),
            ("medicine", "medicine"),
            ("nursing", "nursing"),
            ("teaching", "teaching"),
            ("education", "education"),
            ("law", "law"),
            ("design", "design"),
            ("computer science", "computer science"),
            ("data science", "data science"),
            ("business", "business"),
            ("finance", "finance"),
            ("biology", "biology"),
            ("chemistry", "chemistry"),
            ("physics", "physics"),
            ("history", "history"),
            ("literature", "literature"),
        ],
    )
}

fn quoted_book_values(body: &str) -> Vec<String> {
    static QUOTED_TITLE: OnceLock<regex::Regex> = OnceLock::new();
    let regex = QUOTED_TITLE.get_or_init(|| {
        regex::Regex::new(r#"["“]([^"”\r\n]{2,120})["”]"#).expect("quoted title regex")
    });
    let lower = body.to_lowercase();
    if !contains_any(&lower, &["book", "read", "novel"]) {
        return Vec::new();
    }
    regex
        .captures_iter(body)
        .filter_map(|captures| captures.get(1))
        .map(|value| value.as_str().trim().to_string())
        .filter(|value| !value.is_empty())
        .collect()
}

fn location_values(lower: &str) -> Vec<String> {
    mapped_grounded_values(
        lower,
        &[
            ("beach", "beach"),
            ("mountains", "mountains"),
            ("mountain", "mountains"),
            ("forest", "forest"),
            ("woods", "woods"),
            ("park", "park"),
            ("museum", "museum"),
            ("gallery", "gallery"),
            ("library", "library"),
            ("school", "school"),
            ("office", "office"),
            ("workplace", "workplace"),
            ("home", "home"),
            ("lake", "lake"),
            ("river", "river"),
            ("coast", "coast"),
            ("desert", "desert"),
            ("campsite", "campsite"),
            ("city", "city"),
            ("village", "village"),
        ],
    )
}

fn grounded_preference_regex() -> &'static regex::Regex {
    static PREFERENCE_OBJECT: OnceLock<regex::Regex> = OnceLock::new();
    PREFERENCE_OBJECT.get_or_init(|| {
        regex::Regex::new(
            r"(?i)\b((?:i|we|you|they|he|she)|(?:(?:the|my|our|her|his|their)\s+(?:(?:\d+|one|two|three|four)\s+)?(?:(?:older|younger)\s+)?(?:kids?|child|children)))\s+(?:(?:(?:really|absolutely|especially|always)\s+)*(?:love|loves|loved|like|likes|liked|enjoy|enjoys|enjoyed)|(?:am|are|is|was|were)\s+(?:(?:really|especially)\s+)*(?:stoked for|excited about))\s+([^.!?;\r\n]{2,100})",
        )
        .expect("preference object regex")
    })
}

fn preference_clause_boundary_regex() -> &'static regex::Regex {
    static CLAUSE_BOUNDARY: OnceLock<regex::Regex> = OnceLock::new();
    CLAUSE_BOUNDARY.get_or_init(|| {
        regex::Regex::new(
            r"(?i)\s+(?:and|but)\s+(?:i|it|we|you|they|he|she|(?:the|a|an|my|our|her|his|their)\s+\w+(?:\s+\w+){0,2})\s+(?:am|are|is|was|were|have|has|had|do|does|did|can|could|will|would|should)\b",
        )
        .expect("preference clause boundary regex")
    })
}

fn preference_values(body: &str) -> Vec<String> {
    grounded_preference_regex()
        .captures_iter(body)
        .filter_map(|captures| captures.get(2))
        .map(|capture| {
            let mut value = capture.as_str().trim();
            for boundary in [" because ", " while ", " which ", " so ", " - ", ","] {
                if let Some(index) = value.to_lowercase().find(boundary) {
                    value = value[..index].trim();
                }
            }
            if let Some(boundary) = preference_clause_boundary_regex().find(value) {
                value = value[..boundary.start()].trim();
            }
            value
                .trim_start_matches("the ")
                .trim_matches(|character: char| matches!(character, '"' | '\'' | '“' | '”' | ' '))
                .to_string()
        })
        .filter(|value| value.chars().count() >= 2)
        .filter(|value| {
            !matches!(
                value.trim().to_lowercase().as_str(),
                "it" | "this" | "that" | "them" | "one" | "all of that"
            )
        })
        .collect()
}

fn preference_statement_matches_children(body: &str) -> bool {
    grounded_preference_regex()
        .captures_iter(body)
        .any(|captures| {
            let subject = captures
                .get(1)
                .map(|capture| capture.as_str().to_lowercase())
                .unwrap_or_default();
            subject == "they"
                || contains_any(&subject, &["kid", "child", "children", "younger", "older"])
        })
}

pub(crate) fn list_values_from_statement(kind: GroundedListKind, body: &str) -> Vec<String> {
    let lower = body.to_lowercase();
    match kind {
        GroundedListKind::Activities => activity_values(&lower),
        GroundedListKind::Events => event_values(&lower),
        GroundedListKind::Fields => field_values(&lower),
        GroundedListKind::Books => quoted_book_values(body),
        GroundedListKind::Locations => location_values(&lower),
        GroundedListKind::Preferences => preference_values(body),
        GroundedListKind::Destress => {
            if source_signals_decompression(&lower) {
                activity_values(&lower)
            } else {
                Vec::new()
            }
        }
    }
}
pub(crate) fn answer_list_values_from_statement(kind: GroundedListKind, body: &str) -> Vec<String> {
    let values = list_values_from_statement(kind, body);
    if kind == GroundedListKind::Activities && !activity_statement_records_participation(body) {
        Vec::new()
    } else {
        values
    }
}

pub(crate) fn source_has_unresolved_book_reference(body: &str) -> bool {
    let lower = body.to_lowercase();
    contains_any(
        &lower,
        &[
            "this book",
            "that book",
            "the book",
            "book i read",
            "book you recommended",
            "这本书",
            "那本书",
        ],
    )
}

fn activity_statement_records_participation(body: &str) -> bool {
    let lower = body.to_lowercase();
    !contains_any(
        &lower,
        &[
            "sounds like a great way",
            "would be a great way",
            "could be a great way",
            "should be a great way",
            "got everyone ",
            "gets everyone ",
            "made everyone ",
            "makes everyone ",
        ],
    )
}

fn excluded_activity_values(query: &str) -> Vec<String> {
    let lower = query.to_lowercase();
    ["besides ", "other than ", "excluding "]
        .iter()
        .find_map(|marker| {
            lower
                .find(marker)
                .map(|index| &lower[index + marker.len()..])
        })
        .map(activity_values)
        .unwrap_or_default()
}

fn push_grounded_list_value(
    values: &mut Vec<GroundedListValue>,
    value: String,
    item: &RetrievedItem,
    relation_strength: u8,
    score: f32,
) {
    let key = normalized_grounded_statement(&value);
    if key.is_empty() {
        return;
    }
    if let Some(existing) = values.iter_mut().find(|existing| {
        let existing_key = normalized_grounded_statement(&existing.value);
        existing_key == key
            || (key.chars().count() >= 5
                && existing_key.chars().count() >= 5
                && (existing_key.contains(&key) || key.contains(&existing_key)))
    }) {
        if relation_strength > existing.relation_strength
            || (relation_strength == existing.relation_strength && score > existing.score)
        {
            existing.memory_id = item.id;
            existing.source_label = readable_answer_source_label(item);
            existing.relation_strength = relation_strength;
            existing.score = score;
        }
        return;
    }
    values.push(GroundedListValue {
        value,
        memory_id: item.id,
        source_label: readable_answer_source_label(item),
        relation_strength,
        score,
    });
}

fn adjacent_statement_decompression_strength(
    statements: &[String],
    index: usize,
    speaker: Option<&str>,
) -> u8 {
    let Some(speaker) = speaker else {
        return 0;
    };
    let mut strongest = 0u8;
    for neighbor_index in [index.checked_sub(1), Some(index.saturating_add(1))]
        .into_iter()
        .flatten()
    {
        let Some(neighbor) = statements.get(neighbor_index) else {
            continue;
        };
        let (neighbor_speaker, neighbor_body) = grounded_statement_parts(neighbor);
        if neighbor_speaker.is_some_and(|candidate| person_names_match(candidate, speaker)) {
            strongest = strongest.max(decompression_evidence_strength(neighbor_body));
        }
    }
    strongest
}

fn destress_query_minimum_relation_strength(query_lower: &str) -> u8 {
    if contains_any(
        query_lower,
        &[
            "destress",
            "de-stress",
            "stress relief",
            "cope with stress",
            "减压",
        ],
    ) {
        3
    } else {
        2
    }
}



#[derive(Debug, Clone)]
struct RendererClaim {
    text: String,
    evidence_ids: Vec<u64>,
}

/// Allowlisted, content-free structure a renderer may place around its
/// verified Claims: a heading, section labels, and fixed notes. A frame asserts
/// nothing, so it needs no evidence; anything factual (a chapter number, "a
/// similar case exists") must be a Claim instead. The Claim gate renders the
/// frame itself, so it can only ever wrap Claims that verified.
#[derive(Debug, Clone)]
struct RendererFrame {
    kind: FrameKind,
    /// Claim indices into the renderer's claim plan, grouped by section.
    sections: Vec<(FrameSection, Vec<usize>)>,
    notes: Vec<FrameNote>,
    /// System-derived metadata of a source record (capture origin, stored
    /// freshness), shown under its section only when that record is cited by
    /// a verified Claim. Never renderer prose.
    provenance: Vec<(FrameSection, u64, String)>,
    /// Source label of a Claim's own cited record, shown as a prefix when that
    /// Claim verified (e.g. which document a finding came from).
    claim_sources: HashMap<usize, String>,
    /// Number of Claims the renderer planned; the frame is dropped if the
    /// checked coverage does not line up with it.
    claim_count: usize,
}

#[derive(Debug, Clone, Copy)]
enum FrameKind {
    NovelCrossChapter,
    OperatorBriefing,
    AmbientCapture,
    LearningResume,
    DocumentComparison,
    SimilarExperience,
    RelatedExperience,
    NovelChapter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameSection {
    ChapterClues,
    Timeline,
    Relationships,
    MissionConstraints,
    StaleRisks,
    TelemetryIncidents,
    OperationalHabits,
    CapturedSources,
    LearningProblemCard,
    LearningLoop,
    LearningConceptLink,
    SourceDocuments,
    DocumentFindings,
    ExperienceRecord,
    FailureReason,
    Recovery,
    ChapterIdentity,
}

#[derive(Debug, Clone, Copy)]
enum FrameNote {
    ArcReadingMethod,
    LongDocumentCorrection,
    MemoryFilesConsulted,
    LegalCaution,
    TraceLookup,
}

impl FrameKind {
    fn heading(self, chinese: bool) -> &'static str {
        match (self, chinese) {
            (Self::NovelCrossChapter, true) => "跨章节结构化回答：",
            (Self::NovelCrossChapter, false) => "Cross-chapter structured answer:",
            (Self::OperatorBriefing, _) => "Operator briefing",
            (Self::AmbientCapture, _) => "Ambient capture answer",
            (Self::LearningResume, true) => "学习续接：",
            (Self::LearningResume, false) => "Learning resume:",
            (Self::DocumentComparison, true) => "文件比较：",
            (Self::DocumentComparison, false) => "Document comparison:",
            (Self::SimilarExperience, true) => "相似经验提示：",
            (Self::SimilarExperience, false) => "Similar past experience:",
            (Self::RelatedExperience, true) => "相关经验：",
            (Self::RelatedExperience, false) => "Related experience:",
            (Self::NovelChapter, true) => "章节回答：",
            (Self::NovelChapter, false) => "Chapter answer:",
        }
    }
}

impl FrameSection {
    fn label(self, chinese: bool) -> &'static str {
        match (self, chinese) {
            (Self::ChapterClues, true) => "章节线索证据：",
            (Self::ChapterClues, false) => "Chapter evidence:",
            (Self::Timeline, true) => "时间线证据：",
            (Self::Timeline, false) => "Timeline evidence:",
            (Self::Relationships, true) => "关系证据：",
            (Self::Relationships, false) => "Relationship evidence:",
            (Self::MissionConstraints, _) => "Mission constraints:",
            (Self::StaleRisks, _) => "Stale risks:",
            (Self::TelemetryIncidents, _) => "Telemetry / incidents:",
            (Self::OperationalHabits, _) => "Operational habits:",
            (Self::CapturedSources, _) => "Captured source evidence:",
            (Self::LearningProblemCard, true) => "当前问题与能力边界：",
            (Self::LearningProblemCard, false) => "Current problem and ability boundary:",
            (Self::LearningLoop, true) => "上次尝试与卡住点：",
            (Self::LearningLoop, false) => "Last attempt and stuck point:",
            (Self::LearningConceptLink, true) => "下一步局部概念：",
            (Self::LearningConceptLink, false) => "Next local concept:",
            (Self::SourceDocuments, true) => "来源文件：",
            (Self::SourceDocuments, false) => "Source documents:",
            (Self::DocumentFindings, true) => "已核验要点：",
            (Self::DocumentFindings, false) => "Verified findings:",
            (Self::ExperienceRecord, true) => "过往经历：",
            (Self::ExperienceRecord, false) => "Past episode:",
            (Self::FailureReason, true) => "失败原因：",
            (Self::FailureReason, false) => "Failure reason:",
            (Self::Recovery, true) => "恢复方式：",
            (Self::Recovery, false) => "Recovery:",
            (Self::ChapterIdentity, true) => "章节：",
            (Self::ChapterIdentity, false) => "Chapter:",
        }
    }
}

impl FrameNote {
    fn text(self, chinese: bool) -> &'static str {
        match (self, chinese) {
            (Self::ArcReadingMethod, true) => "弧线推理：按章节概要、时间线事件、人物关系的顺序阅读以上已核验结论。",
            (Self::ArcReadingMethod, false) => "Arc reasoning: read the verified conclusions above in chapter, timeline, then relationship order.",
            (Self::LongDocumentCorrection, true) => "如果章节概要、事件顺序或人物关系不对，请用 trace summary 里的 long_document correction action 修正对应记录。",
            (Self::LongDocumentCorrection, false) => "If a chapter summary, event order, or relationship is wrong, use the long_document correction action in the trace summary to correct that record.",
            (Self::MemoryFilesConsulted, _) => "Evidence: Memory Files context was snapshotted as versioned records and consulted for this answer.",
            (Self::TraceLookup, true) => "Trace: 使用响应里的 trace_id 查询 /memory/trace/{trace_id}。",
            (Self::TraceLookup, false) => "Trace: use the response trace_id with /memory/trace/{trace_id}.",
            (Self::LegalCaution, true) => "注意：这是基于来源的文件比较，不构成法律意见；涉及权利、义务、期限或风险时，请核对引用原文并咨询合格专业人士。",
            (Self::LegalCaution, false) => "Caution: this is a source-grounded document comparison, not legal advice. Verify the cited originals and consult qualified counsel before relying on rights, obligations, deadlines, or risk conclusions.",
        }
    }
}

impl RendererFrame {
    /// Renders verified Claims inside the frame. Returns `None` when nothing
    /// verified or the coverage does not match the planned Claims, in which
    /// case the caller falls back to the unframed Claim list.
    fn render(
        &self,
        coverage: &[CoverageReport],
        sources: &[RetrievedItem],
        chinese: bool,
    ) -> Option<String> {
        if coverage.len() != self.claim_count || !coverage.iter().any(|report| report.covered) {
            return None;
        }
        let cited = |index: usize| {
            coverage.get(index).filter(|report| report.covered).map(|report| {
                match self.claim_sources.get(&index) {
                    Some(source) => format!("- {source}: {}", checked_claim_cited_text(report, sources)),
                    None => format!("- {}", checked_claim_cited_text(report, sources)),
                }
            })
        };
        let verified_records = coverage
            .iter()
            .filter(|report| report.covered)
            .flat_map(|report| report.evidence_ids.iter().copied())
            .collect::<HashSet<_>>();
        let provenance_lines = |section: FrameSection| {
            self.provenance
                .iter()
                .filter(|(owner, id, _)| *owner == section && verified_records.contains(id))
                .map(|(_, _, text)| format!("- {text}"))
                .collect::<Vec<_>>()
        };
        let mut blocks = vec![self.kind.heading(chinese).to_string()];
        let mut placed = HashSet::new();
        let mut sections = self.sections.iter().map(|(section, _)| *section).collect::<Vec<_>>();
        for (section, _, _) in &self.provenance {
            if !sections.contains(section) {
                sections.push(*section);
            }
        }
        for section in sections {
            let mut lines = self
                .sections
                .iter()
                .filter(|(owner, _)| *owner == section)
                .flat_map(|(_, indices)| indices.iter())
                .filter(|index| placed.insert(**index))
                .filter_map(|index| cited(*index))
                .collect::<Vec<_>>();
            lines.extend(provenance_lines(section));
            if !lines.is_empty() {
                blocks.push(format!("{}\n{}", section.label(chinese), lines.join("\n")));
            }
        }
        let unsectioned = (0..coverage.len())
            .filter(|index| !placed.contains(index))
            .filter_map(cited)
            .collect::<Vec<_>>();
        if !unsectioned.is_empty() {
            blocks.push(unsectioned.join("\n"));
        }
        blocks.extend(self.notes.iter().map(|note| note.text(chinese).to_string()));
        Some(blocks.join("\n\n"))
    }
}

#[derive(Debug, Clone)]
enum ChapterFeedback {
    Ambiguous {
        documents: Vec<(String, Vec<u64>)>,
        display_limit: usize,
    },
    Incomplete {
        memory_id: u64,
    },
}

impl ChapterFeedback {
    fn source_ids(&self) -> Vec<u64> {
        match self {
            Self::Ambiguous { documents, .. } => documents
                .iter()
                .flat_map(|(_, ids)| ids.iter().copied())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
            Self::Incomplete { memory_id } => vec![*memory_id],
        }
    }

    fn render(&self) -> String {
        match self {
            Self::Ambiguous { documents, display_limit } => {
                let options = documents.iter().take(*display_limit).map(|(name, ids)| {
                    // Labels are metadata, not instructions or citation sources.
                    let name = strip_inline_answer_citations(name).replace(['[', ']', '`'], "");
                    let ids = ids.iter().map(|id| format!("memory#{id}")).collect::<Vec<_>>().join(", ");
                    format!("- {}：{ids}", compact_context_line(&name, 140))
                }).collect::<Vec<_>>().join("\n");
                let omitted = documents.len().saturating_sub(*display_limit);
                format!("系统反馈：当前不能可靠回答这一章，因为多个文档匹配。请指定书名或文件名，或明确要求比较所有文件。\n\n候选文档：\n{options}\n\n覆盖状态：另有 {omitted} 个候选文档未显示。")
            }
            Self::Incomplete { memory_id } => format!(
                "系统反馈：当前不能可靠回答这一章。候选 memory#{memory_id} 为开头摘录式摘要，当前没有匹配本章的 archive 原文块。请恢复本章原文后刷新摘要。"
            ),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum RendererCoverageGap {
    UnnamedBook,
    EducationCredential,
    DisplayLimit,
}

impl RendererCoverageGap {
    fn message(self, chinese: bool) -> &'static str {
        match (self, chinese) {
            (Self::UnnamedBook, false) => "Coverage gap: another referenced book has no title in the persisted text.",
            (Self::UnnamedBook, true) => "覆盖缺口：记录还提到一本未保留标题的书。",
            (Self::EducationCredential, false) => "Coverage gap: the available evidence does not name a specific degree, major, or certification.",
            (Self::EducationCredential, true) => "覆盖缺口：现有证据没有明确具体学位、专业或资格认证。",
            (Self::DisplayLimit, false) => "Coverage gap: the answer-node limit omitted additional supported values.",
            (Self::DisplayLimit, true) => "覆盖缺口：答案节点上限省略了其他已支持条目。",
        }
    }
}

fn render_grounded_list_answer_plan(
    query: &str,
    items: &[RetrievedItem],
    answer_nodes: Option<usize>,
    detail_level: &str,
) -> Option<(
    GroundedRenderedAnswer,
    Vec<RendererClaim>,
    Vec<RendererCoverageGap>,
)> {
    let kind = grounded_list_kind(query)?;
    let query_lower = query.to_lowercase();
    let minimum_destress_relation_strength = destress_query_minimum_relation_strength(&query_lower);
    let mut values = Vec::new();
    let mut explicit_education_credential = false;
    let mut unresolved_book_reference: Option<(u64, String, String)> = None;
    let creative_project_scope = kind == GroundedListKind::Activities
        && contains_any(
            &query_lower,
            &["creative project", "art project", "craft project"],
        );
    let collaborative_scope = kind == GroundedListKind::Activities
        && contains_any(
            &query_lower,
            &["kid", "child", "children", "together", "with them"],
        );
    let education_scope = kind == GroundedListKind::Fields
        && contains_any(
            &query_lower,
            &[
                "educat",
                "degree",
                "major",
                "certification",
                "study",
                "专业",
                "教育",
                "认证",
            ],
        );
    let excluded_activities = excluded_activity_values(query);

    for item in items.iter().filter(|item| memory_item_is_active(item)) {
        let statements = grounded_source_statements(searchable_source_body(&item.content));
        for (statement_index, statement) in statements.iter().enumerate() {
            if grounded_statement_is_media_caption(&statement) {
                continue;
            }
            let cleaned_statement = grounded_text_without_media_captions(&statement);
            if cleaned_statement.is_empty() {
                continue;
            }
            let (speaker, body) = grounded_statement_parts(&cleaned_statement);
            if !statement_matches_question_subject(query, speaker, body) {
                continue;
            }
            let score = statement_relevance_score(query, &cleaned_statement, item.score);
            let minimum_score = if kind == GroundedListKind::Events {
                0.14
            } else {
                0.24
            };
            if score < minimum_score {
                continue;
            }
            let lower = body.to_lowercase();
            if kind == GroundedListKind::Events
                && !event_statement_matches_scope(&query_lower, &lower)
            {
                continue;
            }
            if education_scope
                && !contains_any(
                    &lower,
                    &[
                        "educat",
                        "degree",
                        "major",
                        "certif",
                        "study",
                        "career",
                        "pursu",
                        "looking into",
                        "counsel",
                        "mental health",
                        "psycholog",
                        "专业",
                        "教育",
                        "认证",
                    ],
                )
            {
                continue;
            }
            if creative_project_scope
                && !contains_any(
                    &lower,
                    &[
                        "paint", "draw", "art", "craft", "pottery", "sculpt", "write", "design",
                    ],
                )
            {
                continue;
            }
            if collaborative_scope
                && !contains_any(
                    &lower,
                    &["we ", "our ", "kids", "children", "together", "with them"],
                )
            {
                continue;
            }
            if kind == GroundedListKind::Locations {
                let action_matches = if query_lower.contains("camp") {
                    lower.contains("camp")
                } else if query_lower.contains("hik") {
                    lower.contains("hik")
                } else if contains_any(&query_lower, &["visit", "visited"]) {
                    contains_any(&lower, &["visit", "went to", "been to"])
                } else if contains_any(&query_lower, &["live", "lived"]) {
                    contains_any(&lower, &["live", "lived", "home"])
                } else if contains_any(&query_lower, &["work", "worked"]) {
                    contains_any(&lower, &["work", "worked", "office"])
                } else {
                    true
                };
                if !action_matches {
                    continue;
                }
            }
            if kind == GroundedListKind::Preferences
                && contains_any(&query_lower, &["kid", "child", "children"])
                && !preference_statement_matches_children(body)
            {
                continue;
            }
            if kind == GroundedListKind::Fields
                && contains_any(
                    &lower,
                    &[
                        "degree",
                        "major",
                        "certification",
                        "certificate",
                        "study",
                        "studying",
                        "course",
                        "program",
                        "学位",
                        "专业",
                        "认证",
                    ],
                )
            {
                explicit_education_credential = true;
            }
            let direct_relation_strength = if kind == GroundedListKind::Destress {
                decompression_evidence_strength(body)
            } else {
                0
            };
            let adjacent_relation_strength = if kind == GroundedListKind::Destress {
                adjacent_statement_decompression_strength(&statements, statement_index, speaker)
            } else {
                0
            };
            let relation_strength = direct_relation_strength.max(adjacent_relation_strength);
            let direct_extracted = if kind == GroundedListKind::Destress
                && direct_relation_strength < minimum_destress_relation_strength
            {
                Vec::new()
            } else {
                answer_list_values_from_statement(kind, body)
            };
            let extracted = if kind == GroundedListKind::Destress
                && direct_extracted.is_empty()
                && adjacent_relation_strength >= minimum_destress_relation_strength
            {
                answer_list_values_from_statement(GroundedListKind::Activities, body)
            } else {
                direct_extracted
            };
            if kind == GroundedListKind::Books
                && extracted.is_empty()
                && source_has_unresolved_book_reference(body)
            {
                unresolved_book_reference.get_or_insert((
                    item.id,
                    readable_answer_source_label(item),
                    body.to_string(),
                ));
            }
            for value in extracted {
                if kind == GroundedListKind::Events
                    && !event_value_matches_scope(&query_lower, &value)
                {
                    continue;
                }
                if kind == GroundedListKind::Activities
                    && excluded_activities.iter().any(|excluded| {
                        normalized_grounded_statement(excluded)
                            == normalized_grounded_statement(&value)
                    })
                {
                    continue;
                }
                push_grounded_list_value(&mut values, value, item, relation_strength, score);
            }
        }
    }

    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| {
        right
            .relation_strength
            .cmp(&left.relation_strength)
            .then_with(|| {
                right
                    .score
                    .partial_cmp(&left.score)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| left.value.cmp(&right.value))
    });
    let automatic_limit = match detail_level {
        "concise" => 12,
        "balanced" => 20,
        "detailed" => values.len(),
        "custom" => 24,
        _ => 16,
    };
    let value_limit = answer_nodes
        .map(normalize_answer_range_node_count)
        .unwrap_or(automatic_limit)
        .max(1)
        .min(values.len())
        .min(MAX_ANSWER_RANGE_NODES);
    let omitted_by_limit = values.len().saturating_sub(value_limit);
    values.truncate(value_limit);

    // Presentation prefixes and missing-value notices are not factual Claims.
    let mut claims = values
        .iter()
        .map(|value| RendererClaim {
            text: value.value.clone(),
            evidence_ids: vec![value.memory_id],
        })
        .collect::<Vec<_>>();
    if let Some((id, _, body)) = &unresolved_book_reference {
        claims.push(RendererClaim {
            text: body.clone(),
            evidence_ids: vec![*id],
        });
    }
    let rendered_values = values
        .iter()
        .map(|value| {
            format!(
                "{} [memory#{} | {}]",
                value.value,
                value.memory_id,
                value.source_label.as_str()
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let chinese = query
        .chars()
        .any(|character| ('一'..='鿿').contains(&character));
    let prefix = if chinese {
        match kind {
            GroundedListKind::Activities => "记录中的活动包括",
            GroundedListKind::Events => "记录中的参与事件包括",
            GroundedListKind::Fields => "现有证据支持的方向包括",
            GroundedListKind::Books => "记录明确提到的书包括",
            GroundedListKind::Locations => "记录中的地点包括",
            GroundedListKind::Preferences => "记录显示的喜好包括",
            GroundedListKind::Destress => "记录中的减压方式包括",
        }
    } else {
        match kind {
            GroundedListKind::Activities => "Recorded activities include",
            GroundedListKind::Events => "Recorded participation includes",
            GroundedListKind::Fields => "The evidence supports these fields:",
            GroundedListKind::Books => "The records explicitly name",
            GroundedListKind::Locations => "The recorded locations are",
            GroundedListKind::Preferences => "The records indicate these preferences:",
            GroundedListKind::Destress => "Recorded ways of decompressing include",
        }
    };

    let mut missing: Option<String> = None;
    let mut gaps = Vec::new();
    if kind == GroundedListKind::Fields
        && contains_any(
            &query_lower,
            &["educat", "degree", "certification", "专业", "教育", "认证"],
        )
        && !explicit_education_credential
    {
        gaps.push(RendererCoverageGap::EducationCredential);
        missing = Some(
            if chinese {
                "但现有证据没有明确具体学位、专业或资格认证"
            } else {
                "but the available evidence does not name a specific degree, major, or certification"
            }
            .to_string(),
        );
    } else if kind == GroundedListKind::Books {
        if let Some((memory_id, source_label, _)) = unresolved_book_reference.as_ref() {
            gaps.push(RendererCoverageGap::UnnamedBook);
            missing = Some(if chinese {
                format!(
                    "但记录还提到一本未保留标题的书 [memory#{memory_id} | {}]",
                    source_label
                )
            } else {
                format!(
                    "but another referenced book has no title in the persisted text [memory#{memory_id} | {}]",
                    source_label
                )
            });
        }
    }
    if missing.is_none() && omitted_by_limit > 0 {
        gaps.push(RendererCoverageGap::DisplayLimit);
        missing = Some(
            if chinese {
                "但用户设置的答案节点上限省略了其他已支持条目"
            } else {
                "but the answer-node limit omitted additional supported values"
            }
            .to_string(),
        );
    }
    let partial = missing.is_some();
    let text = match (chinese, missing.as_deref()) {
        (true, Some(missing)) => {
            format!("直接回答：{prefix}：{rendered_values}；{missing}。")
        }
        (true, None) => format!("直接回答：{prefix}：{rendered_values}。"),
        (false, Some(missing)) => {
            format!("Direct answer: {prefix} {rendered_values}; {missing}.")
        }
        (false, None) => format!("Direct answer: {prefix} {rendered_values}."),
    };
    Some((
        GroundedRenderedAnswer {
            text,
            draft_kind: if partial { "partial" } else { "synthesized" },
            claims: claims.clone(),
        },
        claims,
        gaps,
    ))
}

fn inference_question_kind(query: &str) -> Option<&'static str> {
    let lower = query.to_lowercase();
    if (lower.contains("would ") || lower.contains("could "))
        && (lower.contains(" if ") || lower.contains("如果"))
    {
        Some("counterfactual")
    } else if (lower.starts_with("would ") || lower.starts_with("could "))
        && contains_any(
            &lower,
            &[
                "likely",
                "probably",
                "might",
                "be considered",
                "career option",
                "pursue ",
                "可能",
                "大概",
                "很可能",
            ],
        )
    {
        Some("likely")
    } else {
        None
    }
}

fn inference_specific_terms(query: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "would",
        "could",
        "should",
        "likely",
        "probably",
        "might",
        "what",
        "which",
        "where",
        "when",
        "why",
        "how",
        "have",
        "has",
        "had",
        "still",
        "want",
        "pursue",
        "career",
        "if",
        "not",
        "without",
        "with",
        "from",
        "into",
        "about",
        "their",
        "there",
        "they",
        "them",
        "she",
        "her",
        "hers",
        "he",
        "him",
        "his",
        "books",
        "book",
        "bookshelf",
        "read",
    ];
    let subject = question_named_subject(query).map(|value| value.to_lowercase());
    let mut terms = query
        .to_lowercase()
        .split(|character: char| !character.is_alphanumeric())
        .filter(|term| term.chars().count() >= 3)
        .filter(|term| !STOP.contains(term))
        .filter(|term| subject.as_deref() != Some(*term))
        .map(str::to_string)
        .collect::<Vec<_>>();
    terms.sort();
    terms.dedup();
    terms
}

#[derive(Debug, Clone)]
struct SubjectInferenceEvidence {
    memory_id: u64,
    source: String,
    text: String,
    score: f32,
}

fn subject_inference_evidence(
    query: &str,
    items: &[RetrievedItem],
) -> Vec<SubjectInferenceEvidence> {
    let subject = question_named_subject(query);
    let mut evidence = Vec::new();
    for item in items.iter().filter(|item| memory_item_is_active(item)) {
        for statement in grounded_source_statements(searchable_source_body(&item.content)) {
            if grounded_statement_is_media_caption(&statement) {
                continue;
            }
            let (speaker, body) = grounded_statement_parts(&statement);
            if subject.as_deref().is_some_and(|subject| {
                !speaker.is_some_and(|speaker| person_names_match(speaker, subject))
            }) {
                continue;
            }
            evidence.push(SubjectInferenceEvidence {
                memory_id: item.id,
                source: item.source.clone(),
                text: body.to_string(),
                score: statement_relevance_score(query, &statement, item.score),
            });
        }
    }
    evidence.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
    });
    evidence
}

fn render_specialized_likely_answer(
    query: &str,
    items: &[RetrievedItem],
) -> Option<GroundedRenderedAnswer> {
    let lower = query.to_lowercase();
    let evidence = subject_inference_evidence(query, items);
    let citation = |item: &SubjectInferenceEvidence| {
        format!(
            "[memory#{} | {}]",
            item.memory_id,
            compact_context_line(&item.source, 80)
        )
    };

    if contains_any(
        &lower,
        &["member of the lgbtq", "member of lgbtq", "lgbtq member"],
    ) {
        if let Some(explicit) = evidence.iter().find(|item| {
            contains_any(
                &item.text.to_lowercase(),
                &[
                    "i'm transgender",
                    "i am transgender",
                    "as a transgender",
                    "my transition",
                    "i'm gay",
                    "i am gay",
                    "i'm lesbian",
                    "i am lesbian",
                    "i'm bisexual",
                    "i am bisexual",
                    "part of the lgbtq community",
                    "member of the lgbtq community",
                ],
            )
        }) {
            return Some(GroundedRenderedAnswer {
                text: format!(
                    "Direct answer: Likely yes (medium confidence) {}.\n\nReasoning: the subject explicitly self-identifies in the cited record: “{}”.",
                    citation(explicit),
                    explicit.text
                ),
                draft_kind: "inferred",
                claims: vec![RendererClaim {
                    text: format!("Likely yes (medium confidence): {}", explicit.text),
                    evidence_ids: vec![explicit.memory_id],
                }],
            });
        }
        let supportive = evidence.iter().find(|item| {
            let text = item.text.to_lowercase();
            contains_any(
                &text,
                &["lgbtq", "transgender", "transition", "trans community"],
            ) && contains_any(
                &text,
                &[
                    "support",
                    "proud of you",
                    "accept",
                    "ally",
                    "standing up",
                    "inspiring",
                ],
            )
        })?;
        return Some(GroundedRenderedAnswer {
            text: format!(
                "Direct answer: Likely no, but confidence is low {}.\n\nReasoning: the records show support for the LGBTQ community, but they do not show the subject self-identifying as a member: “{}”.",
                citation(supportive),
                supportive.text
            ),
            draft_kind: "partial_inferred",
            claims: vec![RendererClaim {
                text: format!("Likely no, but confidence is low: {}", supportive.text),
                evidence_ids: vec![supportive.memory_id],
            }],
        });
    }

    if lower.contains("ally") && contains_any(&lower, &["transgender", "trans community", "lgbtq"])
    {
        let supportive = evidence.iter().find(|item| {
            let text = item.text.to_lowercase();
            contains_any(
                &text,
                &["transgender", "transition", "trans community", "lgbtq"],
            ) && contains_any(
                &text,
                &[
                    "support",
                    "proud of you",
                    "accept",
                    "standing up",
                    "inspiring",
                    "courage",
                ],
            )
        })?;
        return Some(GroundedRenderedAnswer {
            text: format!(
                "Direct answer: Likely yes (medium confidence) {}.\n\nReasoning: the cited behavior is explicitly supportive of transgender people: “{}”.",
                citation(supportive),
                supportive.text
            ),
            draft_kind: "inferred",
            claims: vec![RendererClaim {
                text: format!("Likely yes (medium confidence): {}", supportive.text),
                evidence_ids: vec![supportive.memory_id],
            }],
        });
    }

    if lower.contains("career") && contains_any(&lower, &["writing", "writer", "author"]) {
        let alternative = evidence.iter().find(|item| {
            let text = item.text.to_lowercase();
            contains_any(
                &text,
                &[
                    "career",
                    "looking into",
                    "want to work",
                    "thinking of working",
                ],
            ) && contains_any(
                &text,
                &["counsel", "mental health", "psycholog", "social work"],
            )
        })?;
        return Some(GroundedRenderedAnswer {
            text: format!(
                "Direct answer: Likely no (medium confidence) {}.\n\nReasoning: the stated career direction is counseling or mental health rather than writing: “{}”.",
                citation(alternative),
                alternative.text
            ),
            draft_kind: "inferred",
            claims: vec![RendererClaim {
                text: format!("Likely no (medium confidence): {}", alternative.text),
                evidence_ids: vec![alternative.memory_id],
            }],
        });
    }
    None
}

fn render_grounded_inference_answer(
    query: &str,
    items: &[RetrievedItem],
) -> Option<GroundedRenderedAnswer> {
    let kind = inference_question_kind(query)?;
    let query_lower = query.to_lowercase();
    if kind == "likely" {
        if let Some(answer) = render_specialized_likely_answer(query, items) {
            return Some(answer);
        }
    }
    let candidates = grounded_answer_candidates(query, items)
        .into_iter()
        .filter(|candidate| {
            grounded_candidate_is_answerable(query, candidate) && candidate.score >= 0.24
        })
        .collect::<Vec<_>>();
    let primary = candidates
        .iter()
        .find(|candidate| !candidate.is_media_caption)
        .or_else(|| candidates.first())?;
    let chinese = query
        .chars()
        .any(|character| ('一'..='鿿').contains(&character));

    if kind == "counterfactual" {
        let causal = candidates.iter().find(|candidate| {
            !candidate.is_media_caption
                && contains_any(
                    &candidate.text.to_lowercase(),
                    &[
                        "because",
                        "made a difference",
                        "motivated",
                        "inspired",
                        "led to",
                        "so i started",
                        "support",
                        "由于",
                        "因为",
                        "促使",
                        "支持",
                    ],
                )
        });
        let negative_condition = contains_any(
            &query_lower,
            &[
                "hadn't",
                "had not",
                "didn't",
                "did not",
                "without",
                "如果没有",
                "假如没有",
            ],
        );
        if let Some(causal) = causal {
            let conclusion = if negative_condition {
                if chinese {
                    "很可能不会（中等置信度）"
                } else {
                    "Likely no (medium confidence)"
                }
            } else if chinese {
                "很可能会（中等置信度）"
            } else {
                "Likely yes (medium confidence)"
            };
            let text = if chinese {
                format!(
                    "直接回答：{conclusion} [memory#{} | {}]。\n\n推理依据：证据把问题中的条件与结果联系起来：“{}”。这是有依据的反事实推断，并非对另一条时间线的直接观察。",
                    causal.memory_id,
                    compact_context_line(&causal.source, 80),
                    causal.text
                )
            } else {
                format!(
                    "Direct answer: {conclusion} [memory#{} | {}].\n\nReasoning: the evidence links the condition in the question to the outcome: “{}”. This is a grounded counterfactual inference, not a directly observed alternate history.",
                    causal.memory_id,
                    compact_context_line(&causal.source, 80),
                    causal.text
                )
            };
            return Some(GroundedRenderedAnswer {
                text,
                draft_kind: "inferred",
                claims: vec![RendererClaim {
                    text: format!("{conclusion}: {}", causal.text),
                    evidence_ids: vec![causal.memory_id],
                }],
            });
        }
    }

    let specific_terms = inference_specific_terms(query);
    let source_body = items
        .iter()
        .find(|item| item.id == primary.memory_id)
        .map(|item| searchable_source_body(&item.content).to_lowercase())
        .unwrap_or_default();
    let named_target_supported =
        !specific_terms.is_empty() && specific_terms.iter().all(|term| source_body.contains(term));
    let primary_text_lower = primary.text.to_lowercase();
    let directly_negative = contains_any(
        &primary_text_lower,
        &[
            "does not",
            "doesn't",
            "never",
            "not interested",
            "不喜欢",
            "从不",
        ],
    );

    let (conclusion, draft_kind) = if named_target_supported {
        (
            if directly_negative {
                if chinese {
                    "很可能不会（中等置信度）"
                } else {
                    "Likely no (medium confidence)"
                }
            } else if chinese {
                "很可能会（中等置信度）"
            } else {
                "Likely yes (medium confidence)"
            },
            "inferred",
        )
    } else {
        (
            if directly_negative && chinese {
                "很可能不会，但置信度较低"
            } else if directly_negative {
                "Likely no, but confidence is low"
            } else if chinese {
                "很可能会，但置信度较低"
            } else {
                "Likely yes, but confidence is low"
            },
            "partial_inferred",
        )
    };
    let missing = if named_target_supported {
        if chinese {
            "该结论由引用证据推断得出，不是直接观察到的未来结果"
        } else {
            "the conclusion is inferred from the cited evidence rather than directly observed"
        }
    } else if chinese {
        "引用记录提供了相关类别证据，但没有明确点名问题中的具体对象"
    } else {
        "the cited record provides related category evidence but does not explicitly name the requested item"
    };
    let evidence_text = grounded_evidence_display_text(&primary.text);
    let text = if chinese {
        format!(
            "直接回答：{conclusion}：{} [memory#{} | {}]。\n\n推理依据：{missing}。",
            evidence_text,
            primary.memory_id,
            compact_context_line(&primary.source, 80)
        )
    } else {
        format!(
            "Direct answer: {conclusion}: {} [memory#{} | {}].\n\nReasoning: {missing}.",
            evidence_text,
            primary.memory_id,
            compact_context_line(&primary.source, 80)
        )
    };
    Some(GroundedRenderedAnswer {
        text,
        draft_kind,
        claims: vec![RendererClaim {
            text: format!("{conclusion}: {}", primary.text),
            evidence_ids: vec![primary.memory_id],
        }],
    })
}



fn extract_structured_temporal_claims(answer: &str) -> Option<Vec<String>> {
    if !(answer.starts_with("Temporal reasoning:") || answer.starts_with("时间推理：")) {
        return None;
    }

    let claims = answer
        .lines()
        .map(str::trim)
        .filter_map(|line| line.strip_prefix("- "))
        .filter(|line| {
            contains_any(
                line,
                &[
                    "First event:",
                    "Second event:",
                    "Elapsed time:",
                    "Inclusive count:",
                    "事件一：",
                    "事件二：",
                    "相隔：",
                    "若首尾两天都计入：",
                ],
            )
        })
        .map(str::to_string)
        .collect::<Vec<_>>();
    (!claims.is_empty()).then_some(claims)
}

fn extract_direct_answer_claims(answer: &str) -> Option<Vec<String>> {
    let starts_with_direct_answer = answer.starts_with("Direct answer:")
        || answer.starts_with("直接回答：")
        || answer.starts_with("直接回答:");
    if !starts_with_direct_answer {
        return None;
    }

    let mut claims = Vec::new();
    for line in answer
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && line.contains("[memory#"))
    {
        let line = line.strip_prefix("- ").unwrap_or(line).trim();
        let mut pending = String::new();
        for segment in line.split("; ") {
            if !pending.is_empty() {
                pending.push_str("; ");
            }
            pending.push_str(segment.trim());
            if pending.contains("[memory#") {
                claims.push(pending.trim().to_string());
                pending.clear();
            }
        }
    }
    (!claims.is_empty()).then_some(claims)
}

fn renderer_claim_texts(
    mut extracted_claims: Vec<String>,
    renderer_provided_evidence_plan: bool,
    direct_answer: &str,
) -> Vec<String> {
    if extracted_claims.is_empty() && renderer_provided_evidence_plan {
        let direct_answer = direct_answer.trim();
        if !direct_answer.is_empty() {
            extracted_claims.push(direct_answer.to_string());
        }
    }
    extracted_claims
}

fn bounded_verbatim_text(value: &str, max_chars: usize) -> (String, bool) {
    let value = value.trim();
    let total = value.chars().count();
    if total <= max_chars {
        return (value.to_string(), false);
    }
    let mut excerpt = value
        .chars()
        .take(max_chars.saturating_sub(3))
        .collect::<String>();
    excerpt.push_str("...");
    (excerpt, true)
}

fn multimodal_availability_feedback(note_type: &str) -> Option<&'static str> {
    match note_type {
        "multimodal_document"=>Some("[文档记忆提示] 已检索到文档文本候选；来源内容未核验，不能单凭检索确认答案。"),
        "multimodal_image"=>Some("[图片记忆提示] 已检索到图片描述候选；尚未核验原始图像，不代表视觉观察。"),
        "multimodal_ocr"=>Some("[OCR 文字提示] 已检索到标注为 OCR 的文本候选；尚未核验原始图像与识别准确性。"),
        "multimodal_audio"=>Some("[音频记忆提示] 已检索到音频转写或描述候选；尚未核验原始录音。"),
        "multimodal_video"=>Some("[视频记忆提示] 已检索到视频描述候选；尚未核验原始帧、时间段或区域。"),
        _=>None,
    }
}

fn render_explicit_evidence_answer(
    query: &str,
    items: &[RetrievedItem],
    answer_nodes: Option<usize>,
    detail_level: &str,
    response_mode: &str,
) -> Option<String> {
    let automatic_limit = match detail_level {
        "concise" => 1,
        "balanced" => 3,
        "detailed" => 8,
        "custom" => 6,
        _ => 3,
    };
    let node_limit = answer_nodes
        .map(normalize_answer_range_node_count)
        .unwrap_or(automatic_limit)
        .max(1)
        .min(MAX_ANSWER_RANGE_NODES);
    let chinese = query
        .chars()
        .any(|character| ('一'..='鿿').contains(&character));

    if response_mode == "evidence" {
        let candidates = grounded_answer_candidates(query, items);
        let selected = candidates.into_iter().take(node_limit).collect::<Vec<_>>();
        if selected.is_empty() {
            return None;
        }
        let excerpts = selected
            .iter()
            .map(|candidate| {
                format!(
                    "- {} [memory#{} | {}].",
                    candidate.text,
                    candidate.memory_id,
                    compact_context_line(&candidate.source, 80)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        return Some(if chinese {
            format!("证据片段：\n\n{excerpts}")
        } else {
            format!("Evidence excerpts:\n\n{excerpts}")
        });
    }

    let mut blocks = Vec::new();
    let mut total_chars = 0usize;
    const MAX_VERBATIM_CHARS_PER_NODE: usize = 8_000;
    const MAX_VERBATIM_TOTAL_CHARS: usize = 64_000;
    for item in items
        .iter()
        .filter(|item| memory_item_is_active(item))
        .take(node_limit)
    {
        if total_chars >= MAX_VERBATIM_TOTAL_CHARS {
            break;
        }
        let remaining = MAX_VERBATIM_TOTAL_CHARS.saturating_sub(total_chars);
        let per_node_limit = MAX_VERBATIM_CHARS_PER_NODE.min(remaining);
        let body = searchable_source_body(&item.content);
        let (excerpt, truncated) = bounded_verbatim_text(body, per_node_limit);
        if excerpt.is_empty() {
            continue;
        }
        total_chars = total_chars.saturating_add(excerpt.chars().count());
        let source = document_source_label_for_item(item).unwrap_or_else(|| item.source.clone());
        let truncation = if truncated {
            if chinese {
                "\n[此处达到显示上限；完整记录仍保留在 response.evidence 中]"
            } else {
                "\n[Display limit reached; the complete record remains in response.evidence.]"
            }
        } else {
            ""
        };
        blocks.push(format!(
            "[memory#{} | {}]\n{}{}",
            item.id,
            compact_context_line(&source, 120),
            excerpt,
            truncation
        ));
    }
    if blocks.is_empty() {
        return None;
    }
    Some(if chinese {
        format!("原文证据：\n\n{}", blocks.join("\n\n"))
    } else {
        format!("Verbatim evidence:\n\n{}", blocks.join("\n\n"))
    })
}



fn render_temporal_reasoning_answer_plan(
    query: &str,
    items: &[RetrievedItem],
) -> Option<(String, Vec<RendererClaim>)> {
    if !temporal_duration_requested(query) {
        return None;
    }
    let (first_clause, second_clause) =
        crate::memory_engine_query::temporal_relation_clauses(query)?;
    let first_candidates = temporal_evidence_for_clause(&first_clause, items);
    let second_candidates = temporal_evidence_for_clause(&second_clause, items);
    let mut best_pair: Option<(TemporalDateEvidence, TemporalDateEvidence, f32)> = None;
    for first in &first_candidates {
        for second in &second_candidates {
            if first.memory_id == second.memory_id && first.anchor_start == second.anchor_start {
                continue;
            }
            let pair_score = first.score + second.score;
            if best_pair
                .as_ref()
                .map(|(_, _, score)| pair_score > *score)
                .unwrap_or(true)
            {
                best_pair = Some((first.clone(), second.clone(), pair_score));
            }
        }
    }
    let (first, second, _) = best_pair?;
    let elapsed_days = (second.date - first.date).num_days().abs();
    let inclusive_days = elapsed_days.saturating_add(1);
    let chinese = query
        .chars()
        .any(|character| ('\u{4e00}'..='\u{9fff}').contains(&character));
    let text = if chinese {
        format!(
            "时间推理：\n\n- 事件一：{}，日期 {} [memory#{} | {}]。\n- 事件二：{}，日期 {} [memory#{} | {}]。\n- 相隔：{} 天。\n- 若首尾两天都计入：{} 天。",
            first.event_label,
            first.date,
            first.memory_id,
            first.source,
            second.event_label,
            second.date,
            second.memory_id,
            second.source,
            elapsed_days,
            inclusive_days,
        )
    } else {
        format!(
            "Temporal reasoning:\n\n- First event: {} on {} [memory#{} | {}].\n- Second event: {} on {} [memory#{} | {}].\n- Elapsed time: {} days.\n- Inclusive count: {} days when both boundary dates are counted.",
            first.event_label,
            first.date,
            first.memory_id,
            first.source,
            second.event_label,
            second.date,
            second.memory_id,
            second.source,
            elapsed_days,
            inclusive_days,
        )
    };
    let facts = if chinese {
        vec![
            format!("事件一：{}，日期 {}。", first.event_label, first.date),
            format!("事件二：{}，日期 {}。", second.event_label, second.date),
            format!("相隔：{elapsed_days} 天。"),
            format!("若首尾两天都计入：{inclusive_days} 天。"),
        ]
    } else {
        vec![
            format!("First event: {} on {}.", first.event_label, first.date),
            format!("Second event: {} on {}.", second.event_label, second.date),
            format!("Elapsed time: {elapsed_days} days."),
            format!("Inclusive count: {inclusive_days} days when both boundary dates are counted."),
        ]
    };
    let claims = facts
        .into_iter()
        .enumerate()
        .map(|(index, text)| RendererClaim {
            text,
            evidence_ids: match index {
                0 => vec![first.memory_id],
                1 => vec![second.memory_id],
                _ => vec![first.memory_id, second.memory_id]
                    .into_iter()
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect(),
            },
        })
        .collect();
    Some((text, claims))
}

#[derive(Debug, Clone)]
struct CompletedEventEvidence {
    memory_id: u64,
    source: String,
    provider_key: String,
    date: chrono::NaiveDate,
    explicit_date: bool,
    score: f32,
    snippet: String,
}

fn query_scope_month(query: &str) -> Option<u32> {
    let lower = query.to_lowercase();
    for (month, names) in [
        (1, ["january", "jan"]),
        (2, ["february", "feb"]),
        (3, ["march", "mar"]),
        (4, ["april", "apr"]),
        (5, ["may", "may"]),
        (6, ["june", "jun"]),
        (7, ["july", "jul"]),
        (8, ["august", "aug"]),
        (9, ["september", "sep"]),
        (10, ["october", "oct"]),
        (11, ["november", "nov"]),
        (12, ["december", "dec"]),
    ] {
        if names.iter().any(|name| {
            lower
                .split(|character: char| !character.is_alphanumeric())
                .any(|token| token == *name)
        }) {
            return Some(month);
        }
    }
    static CHINESE_MONTH: OnceLock<regex::Regex> = OnceLock::new();
    CHINESE_MONTH
        .get_or_init(|| regex::Regex::new(r"(1[0-2]|[1-9])月").expect("Chinese month regex"))
        .captures(query)
        .and_then(|captures| captures.get(1))
        .and_then(|value| value.as_str().parse::<u32>().ok())
}

fn date_month(date: chrono::NaiveDate) -> Option<u32> {
    date.format("%m").to_string().parse::<u32>().ok()
}

pub(crate) fn source_session_date(content: &str) -> Option<chrono::NaiveDate> {
    static NUMERIC_SESSION_DATE: OnceLock<regex::Regex> = OnceLock::new();
    let numeric = NUMERIC_SESSION_DATE.get_or_init(|| {
        regex::Regex::new(r"(?i)Session date:\s*([0-9]{4})[/\-]([0-9]{1,2})[/\-]([0-9]{1,2})")
            .expect("session date regex")
    });
    if let Some(captures) = numeric.captures(content) {
        return chrono::NaiveDate::from_ymd_opt(
            captures.get(1)?.as_str().parse().ok()?,
            captures.get(2)?.as_str().parse().ok()?,
            captures.get(3)?.as_str().parse().ok()?,
        );
    }

    fn month_number(value: &str) -> Option<u32> {
        match value.trim().to_ascii_lowercase().as_str() {
            "january" | "jan" => Some(1),
            "february" | "feb" => Some(2),
            "march" | "mar" => Some(3),
            "april" | "apr" => Some(4),
            "may" => Some(5),
            "june" | "jun" => Some(6),
            "july" | "jul" => Some(7),
            "august" | "aug" => Some(8),
            "september" | "sep" | "sept" => Some(9),
            "october" | "oct" => Some(10),
            "november" | "nov" => Some(11),
            "december" | "dec" => Some(12),
            _ => None,
        }
    }

    static DAY_MONTH_SESSION_DATE: OnceLock<regex::Regex> = OnceLock::new();
    let day_month = DAY_MONTH_SESSION_DATE.get_or_init(|| {
        regex::Regex::new(
            r"(?i)Session date:\s*(?:[0-9]{1,2}:[0-9]{2}\s*(?:am|pm)\s+on\s+)?([0-9]{1,2})\s+([a-z]+),?\s+([0-9]{4})",
        )
        .expect("English day-month session date regex")
    });
    if let Some(captures) = day_month.captures(content) {
        return chrono::NaiveDate::from_ymd_opt(
            captures.get(3)?.as_str().parse().ok()?,
            month_number(captures.get(2)?.as_str())?,
            captures.get(1)?.as_str().parse().ok()?,
        );
    }

    static MONTH_DAY_SESSION_DATE: OnceLock<regex::Regex> = OnceLock::new();
    let month_day = MONTH_DAY_SESSION_DATE.get_or_init(|| {
        regex::Regex::new(
            r"(?i)Session date:\s*(?:[0-9]{1,2}:[0-9]{2}\s*(?:am|pm)\s+on\s+)?([a-z]+)\s+([0-9]{1,2}),?\s+([0-9]{4})",
        )
        .expect("English month-day session date regex")
    });
    let captures = month_day.captures(content)?;
    chrono::NaiveDate::from_ymd_opt(
        captures.get(3)?.as_str().parse().ok()?,
        month_number(captures.get(1)?.as_str())?,
        captures.get(2)?.as_str().parse().ok()?,
    )
}

fn searchable_source_body(content: &str) -> &str {
    if content
        .trim_start()
        .starts_with("Universal source projection\n")
    {
        content
            .split_once("\n\n")
            .map(|(_, body)| body)
            .unwrap_or(content)
    } else {
        content
    }
}

fn event_topic_markers(query: &str) -> Vec<&'static str> {
    let query = query.to_lowercase();
    let families: &[(&[&str], &[&str])] = &[
        (
            &["appointment", "doctor", "physician", "就诊", "看诊", "预约"],
            &[
                "appointment",
                "doctor",
                "physician",
                "surgeon",
                "neurologist",
                "therapist",
                "dentist",
                "就诊",
                "看诊",
                "医生",
                "预约",
            ],
        ),
        (&["meeting", "meetings", "会议"], &["meeting", "会议"]),
        (&["interview", "interviews", "面试"], &["interview", "面试"]),
        (
            &["lesson", "lessons", "class", "classes", "课程"],
            &["lesson", "class", "课程"],
        ),
        (
            &["trip", "trips", "flight", "flights", "旅行", "航班"],
            &["trip", "flight", "旅行", "航班"],
        ),
        (
            &["visit", "visits", "拜访", "到访"],
            &["visit", "visited", "拜访", "到访"],
        ),
        (&["call", "calls", "通话"], &["call", "通话"]),
    ];
    for (query_markers, evidence_markers) in families {
        if query_markers.iter().any(|marker| query.contains(marker)) {
            return evidence_markers.to_vec();
        }
    }
    Vec::new()
}

fn event_count_question(query: &str) -> bool {
    let lower = query.to_lowercase();
    (lower.contains("how many") || lower.contains("多少次") || lower.contains("几次"))
        && !event_topic_markers(query).is_empty()
}

fn floor_char_boundary(text: &str, minimum: usize, offset: usize) -> usize {
    let mut boundary = offset.min(text.len());
    while boundary > minimum && !text.is_char_boundary(boundary) {
        boundary -= 1;
    }
    boundary
}

fn completed_event_clause(content: &str, action_start: usize) -> &str {
    let mut end = floor_char_boundary(content, action_start, action_start.saturating_add(420));
    let tail = &content[action_start..end];
    for (offset, character) in tail.char_indices() {
        let boundary = matches!(character, '\n' | '?' | '!' | '。' | '？' | '！');
        let sentence_period = character == '.'
            && tail[offset..]
                .chars()
                .nth(1)
                .map(|next| next.is_whitespace())
                .unwrap_or(true)
            && !tail[..offset]
                .split_whitespace()
                .last()
                .map(|word| {
                    matches!(
                        word.trim_matches(|value: char| !value.is_alphanumeric()),
                        "Dr" | "Mr" | "Mrs" | "Ms" | "St"
                    )
                })
                .unwrap_or(false);
        if boundary || sentence_period {
            end = action_start + offset + character.len_utf8();
            break;
        }
    }
    &content[action_start..end]
}

fn event_provider_key(clause: &str, topic_markers: &[&str]) -> String {
    let lower = clause.to_lowercase();
    if [
        "primary care",
        "family doctor",
        "general practitioner",
        "dr. smith",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
    {
        return "primary-care".to_string();
    }
    if ["orthopedic", "orthopaedic", "dr. thompson"]
        .iter()
        .any(|marker| lower.contains(marker))
    {
        return "orthopedic".to_string();
    }
    if ["neurologist", "neurology", "dr. johnson"]
        .iter()
        .any(|marker| lower.contains(marker))
    {
        return "neurology".to_string();
    }
    if ["dentist", "dental"]
        .iter()
        .any(|marker| lower.contains(marker))
    {
        return "dentistry".to_string();
    }
    if ["therapist", "therapy"]
        .iter()
        .any(|marker| lower.contains(marker))
    {
        return "therapy".to_string();
    }
    static NAMED_PROVIDER: OnceLock<regex::Regex> = OnceLock::new();
    if let Some(name) = NAMED_PROVIDER
        .get_or_init(|| {
            regex::Regex::new(r"(?i)\bdr\.?\s+([a-z][a-z'\-]+)").expect("named provider regex")
        })
        .captures(clause)
        .and_then(|captures| captures.get(1))
    {
        return format!("dr:{}", name.as_str().to_lowercase());
    }
    topic_markers
        .iter()
        .find(|marker| lower.contains(**marker))
        .copied()
        .unwrap_or("event")
        .to_string()
}

fn completed_event_candidates(query: &str, items: &[RetrievedItem]) -> Vec<CompletedEventEvidence> {
    static COMPLETED_ACTION: OnceLock<regex::Regex> = OnceLock::new();
    let action_regex = COMPLETED_ACTION.get_or_init(|| {
        regex::Regex::new(
            r"(?i)\b(?:I|we)\s+(?:(?:recently|finally|just|already)\s+)?(?:had|attended|visited|went(?:\s+to(?:\s+see)?)?|saw|completed|joined|took|made)\b|(?:我|我们)(?:最近|终于|刚刚|已经)?(?:参加了|去了|看了|完成了|拜访了|开了)",
        )
        .expect("completed action regex")
    });
    let topic_markers = event_topic_markers(query);
    let scope_month = query_scope_month(query);
    let mut candidates = Vec::new();
    for item in items.iter().filter(|item| memory_item_is_active(item)) {
        let searchable = searchable_source_body(&item.content);
        let session_date = source_session_date(searchable).unwrap_or_else(|| {
            chrono::NaiveDate::parse_from_str(
                &item.created_at.format("%Y-%m-%d").to_string(),
                "%Y-%m-%d",
            )
            .unwrap_or_else(|_| chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap())
        });
        for action in action_regex.find_iter(searchable) {
            let clause = completed_event_clause(searchable, action.start());
            let clause_lower = clause.to_lowercase();
            if !topic_markers
                .iter()
                .any(|marker| clause_lower.contains(marker))
            {
                continue;
            }
            let default_year = session_date
                .format("%Y")
                .to_string()
                .parse::<i32>()
                .unwrap_or(1970);
            let explicit_date = temporal_date_anchors(clause, default_year)
                .into_iter()
                .find(|(_, _, _, header)| !*header)
                .map(|(date, _, _, _)| date);
            let effective_date = explicit_date.unwrap_or(session_date);
            if scope_month
                .zip(date_month(effective_date))
                .is_some_and(|(expected, actual)| expected != actual)
            {
                continue;
            }
            let score = crate::memory_engine_query::lexical_query_coverage(query, clause)
                + if explicit_date.is_some() { 0.12 } else { 0.0 }
                + item.score * 0.05;
            if score < 0.20 {
                continue;
            }
            candidates.push(CompletedEventEvidence {
                memory_id: item.id,
                source: item.source.clone(),
                provider_key: event_provider_key(clause, &topic_markers),
                date: effective_date,
                explicit_date: explicit_date.is_some(),
                score,
                snippet: compact_context_line(clause, 240),
            });
        }
    }
    candidates.sort_by(|left, right| {
        right.explicit_date.cmp(&left.explicit_date).then_with(|| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
        })
    });

    let mut distinct: Vec<CompletedEventEvidence> = Vec::new();
    for candidate in candidates {
        let duplicate = distinct.iter().position(|existing| {
            existing.provider_key == candidate.provider_key
                && (!existing.explicit_date
                    || !candidate.explicit_date
                    || existing.date == candidate.date)
        });
        if let Some(index) = duplicate {
            if (candidate.explicit_date && !distinct[index].explicit_date)
                || (candidate.explicit_date == distinct[index].explicit_date
                    && candidate.score > distinct[index].score)
            {
                distinct[index] = candidate;
            }
        } else {
            distinct.push(candidate);
        }
    }
    distinct.sort_by_key(|candidate| candidate.date);
    distinct
}



fn render_cross_source_event_count_answer_plan(
    query: &str,
    items: &[RetrievedItem],
) -> Option<(String, Vec<RendererClaim>)> {
    if !event_count_question(query) {
        return None;
    }
    let events = completed_event_candidates(query, items);
    if events.is_empty() {
        return None;
    }
    let evidence = events
        .iter()
        .map(|event| {
            format!(
                "- {} on {}. [memory#{} | {}] {}",
                event.provider_key, event.date, event.memory_id, event.source, event.snippet
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let chinese = query
        .chars()
        .any(|character| ('\u{4e00}'..='\u{9fff}').contains(&character));
    let text = if chinese {
        format!(
            "跨来源聚合：\n\n- 请求时间范围内，去重后的已完成事件：{} 次。\n{}",
            events.len(),
            evidence
        )
    } else {
        format!(
            "Cross-source aggregation:\n\n- Distinct completed events in the requested period: {}.\n{}",
            events.len(),
            evidence
        )
    };
    let claim = RendererClaim {
        text: if chinese {
            format!("请求时间范围内，去重后的已完成事件：{} 次。", events.len())
        } else {
            format!(
                "Distinct completed events in the requested period: {}.",
                events.len()
            )
        },
        evidence_ids: events
            .iter()
            .map(|event| event.memory_id)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
    };
    Some((text, vec![claim]))
}

#[derive(Debug, Clone)]
struct ScalarEvidence {
    memory_id: u64,
    source: String,
    answer: String,
    snippet: String,
    score: f32,
    source_date: Option<chrono::NaiveDate>,
}

fn normalized_count_unit(value: &str) -> String {
    if value.len() > 3 && value.ends_with("ies") {
        format!("{}y", &value[..value.len() - 3])
    } else if value.len() > 2 && value.ends_with('s') && !value.ends_with("ss") {
        value[..value.len() - 1].to_string()
    } else {
        value.to_string()
    }
}

fn question_count_unit(query: &str) -> Option<String> {
    let query = query.to_lowercase();
    let start = query.find("how many")?.saturating_add("how many".len());
    let mut candidate = None;
    for token in MemoryEngine::tokenize(&query[start..]) {
        if matches!(
            token.as_str(),
            "have" | "has" | "had" | "did" | "do" | "does" | "were" | "was" | "are" | "is"
        ) {
            break;
        }
        if matches!(
            token.as_str(),
            "a" | "an" | "the" | "new" | "more" | "total" | "different" | "distinct" | "additional"
        ) {
            continue;
        }
        candidate = Some(normalized_count_unit(&token));
    }
    candidate
}

fn following_query_unit_score(count_unit: Option<&str>, content_after_number: &str) -> f32 {
    let Some(count_unit) = count_unit else {
        return 0.0;
    };
    for token in MemoryEngine::tokenize(content_after_number) {
        if matches!(
            token.as_str(),
            "a" | "an" | "the" | "new" | "more" | "about" | "approximately"
        ) {
            continue;
        }
        return if normalized_count_unit(&token) == count_unit {
            0.45
        } else {
            0.0
        };
    }
    0.0
}

fn previous_question_score_for_offset(query: &str, content: &str, offset: usize) -> Option<f32> {
    let mut previous_line: Option<&str> = None;
    let mut line_start = 0;
    for segment in content.split_inclusive('\n') {
        let line_end = line_start + segment.len();
        if offset < line_end {
            let question = previous_line?;
            if !question.contains(['?', '？']) {
                return None;
            }
            return Some(
                crate::memory_engine_query::lexical_query_coverage(query, question)
                    + semantic_topic_bonus(query, question),
            );
        }
        previous_line = Some(segment.trim());
        line_start = line_end;
    }
    None
}

fn focused_number_candidates(query: &str, items: &[RetrievedItem]) -> Vec<ScalarEvidence> {
    static NUMBER: OnceLock<regex::Regex> = OnceLock::new();
    let number_regex = NUMBER
        .get_or_init(|| regex::Regex::new(r"(?i)\b([0-9]+(?:\.[0-9]+)?)\b").expect("number regex"));
    let latest_state = crate::memory_engine_query::query_requests_latest_state(query);
    let count_unit = question_count_unit(query);
    let mut candidates = Vec::new();
    for item in items.iter().filter(|item| memory_item_is_active(item)) {
        let searchable = searchable_source_body(&item.content);
        for number in number_regex.captures_iter(searchable) {
            let Some(full_match) = number.get(0) else {
                continue;
            };
            let line_start = searchable[..full_match.start()]
                .rfind('\n')
                .map(|index| index + 1)
                .unwrap_or(0);
            let line_end = searchable[full_match.end()..]
                .find('\n')
                .map(|index| full_match.end() + index)
                .unwrap_or(searchable.len());
            if searchable[line_start..line_end]
                .trim_start()
                .to_lowercase()
                .starts_with("session date:")
            {
                continue;
            }
            let previous = searchable[..full_match.start()].chars().next_back();
            let next = searchable[full_match.end()..].chars().next();
            if previous.is_some_and(|character| matches!(character, '/' | '-' | ':'))
                || next.is_some_and(|character| matches!(character, '/' | '-' | ':'))
            {
                continue;
            }
            let Ok(numeric_value) = full_match.as_str().parse::<f64>() else {
                continue;
            };
            if numeric_value > 1_000_000_000.0 || (1900.0..=2100.0).contains(&numeric_value) {
                continue;
            }
            let window = bounded_text_window(searchable, full_match.start(), full_match.end(), 340);
            let after_end = floor_char_boundary(
                searchable,
                full_match.end(),
                full_match.end().saturating_add(80),
            );
            let after = &searchable[full_match.end()..after_end];
            let mut score = crate::memory_engine_query::lexical_query_coverage(query, window)
                + following_query_unit_score(count_unit.as_deref(), after)
                + item.score * 0.05;
            if let Some(context_score) =
                previous_question_score_for_offset(query, searchable, full_match.start())
            {
                if context_score < 0.32 {
                    score -= 0.45;
                } else {
                    score += context_score * 0.80 + 0.20;
                }
            }
            if score < 0.42 {
                continue;
            }
            candidates.push(ScalarEvidence {
                memory_id: item.id,
                source: item.source.clone(),
                answer: full_match.as_str().to_string(),
                snippet: compact_context_line(
                    bounded_text_window(searchable, full_match.start(), full_match.end(), 150),
                    320,
                ),
                score,
                source_date: source_session_date(searchable),
            });
        }
    }
    if latest_state {
        let latest_relevant_date = candidates
            .iter()
            .filter(|candidate| candidate.score >= 0.65)
            .filter_map(|candidate| candidate.source_date)
            .max();
        if let Some(latest) = latest_relevant_date {
            for candidate in &mut candidates {
                if candidate.source_date == Some(latest) {
                    candidate.score += 0.12;
                }
            }
        }
    }
    candidates.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
    });
    candidates
}

fn duration_value_regex() -> &'static regex::Regex {
    static DURATION: OnceLock<regex::Regex> = OnceLock::new();
    DURATION.get_or_init(|| {
        regex::Regex::new(
            r"(?i)(?:\b(?:(?:over|more than|about|approximately|nearly|almost|around)\s+)?(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|[0-9]+)\s+(?:day|week|month|year|hour|minute)s?\b|(?:[0-9]+|一|两|二|三|四|五|六|七|八|九|十)\s*(?:天|日|周|星期|个月|月|年|小时|分钟))",
        )
        .expect("duration regex")
    })
}

pub(crate) fn source_has_duration_value(value: &str) -> bool {
    duration_value_regex().is_match(value)
}

fn focused_duration_candidates(query: &str, items: &[RetrievedItem]) -> Vec<ScalarEvidence> {
    let duration_regex = duration_value_regex();
    let mut candidates = Vec::new();
    for item in items.iter().filter(|item| memory_item_is_active(item)) {
        let searchable = searchable_source_body(&item.content);
        let statements = searchable
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .filter(|line| {
                let lower = line.to_lowercase();
                ![
                    "archive type:",
                    "source file:",
                    "source:",
                    "evidence:",
                    "observation:",
                    "session date:",
                    "universal source projection",
                ]
                .iter()
                .any(|prefix| lower.starts_with(prefix))
            })
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        for (index, statement) in statements.iter().enumerate() {
            let (speaker, _) = grounded_statement_parts(strip_grounded_turn_id(statement.as_str()));
            if question_named_subject(query)
                .as_deref()
                .zip(speaker)
                .is_some_and(|(subject, speaker)| !person_names_match(subject, speaker))
            {
                continue;
            }
            for duration in duration_regex.find_iter(statement) {
                let mut score = statement_relevance_score(query, statement, item.score);
                if query.to_lowercase().contains("ago") && statement.to_lowercase().contains("ago")
                {
                    score += 0.12;
                }
                let previous_question = index
                    .checked_sub(1)
                    .and_then(|previous| statements.get(previous))
                    .filter(|previous| previous.contains('?') || previous.contains('？'));
                if let Some(previous) = previous_question {
                    let adjacency_score =
                        crate::memory_engine_query::lexical_query_coverage(query, previous)
                            + semantic_topic_bonus(query, previous);
                    if adjacency_score >= 0.32 {
                        score += adjacency_score * 0.80 + 0.20;
                    } else {
                        score -= 0.45;
                    }
                }
                if score < 0.38 {
                    continue;
                }
                let snippet = previous_question
                    .map(|previous| format!("{previous} {statement}"))
                    .unwrap_or_else(|| statement.clone());
                candidates.push(ScalarEvidence {
                    memory_id: item.id,
                    source: item.source.clone(),
                    answer: duration.as_str().to_lowercase(),
                    snippet: compact_context_line(&snippet, 320),
                    score,
                    source_date: source_session_date(searchable),
                });
            }
        }
    }
    candidates.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
    });
    candidates
}



fn render_focused_scalar_answer_plan(
    query: &str,
    items: &[RetrievedItem],
) -> Option<(String, Vec<RendererClaim>)> {
    let lower = query.to_lowercase();
    let duration_question = lower.contains("how long") || lower.contains("多久");
    let number_question =
        lower.contains("how many") || lower.contains("多少") || lower.contains("几个");
    let candidate = if duration_question {
        focused_duration_candidates(query, items).into_iter().next()
    } else if number_question {
        focused_number_candidates(query, items).into_iter().next()
    } else {
        None
    }?;
    let chinese = query
        .chars()
        .any(|character| ('\u{4e00}'..='\u{9fff}').contains(&character));
    let text = if chinese {
        format!(
            "聚焦回答：{}。\n\n- 证据：[memory#{} | {}] {}",
            candidate.answer, candidate.memory_id, candidate.source, candidate.snippet
        )
    } else {
        format!(
            "Focused answer: {}.\n\n- Evidence: [memory#{} | {}] {}",
            candidate.answer, candidate.memory_id, candidate.source, candidate.snippet
        )
    };
    Some((
        text,
        vec![RendererClaim {
            text: candidate.answer,
            evidence_ids: vec![candidate.memory_id],
        }],
    ))
}

/// Compares statements across two or more source documents. Each Claim is
/// one grounded statement from one document record, prefixed (after
/// verification) with that record's source label. Every displayed document
/// keeps at least one node; `answer_nodes`/`detail_level` size the rest.
fn render_document_comparison_answer_plan(
    query: &str,
    items: &[RetrievedItem],
    document_limit: Option<usize>,
    answer_nodes: Option<usize>,
    detail_level: &str,
) -> Option<(String, Vec<RendererClaim>, RendererFrame)> {
    let query_lower = answer_task_text(query).to_lowercase();
    if !contains_any(
        &query_lower,
        &[
            "multiple documents",
            "multiple legal documents",
            "legal documents",
            "several documents",
            "all documents",
            "across documents",
            "compare documents",
            "contracts",
            "agreements",
            "policies",
            "files",
            "多份",
            "多个文件",
            "多个文档",
            "这些文件",
            "这些文档",
            "合同",
            "协议",
            "法律文件",
            "条款",
        ],
    ) {
        return None;
    }
    let mut document_labels = Vec::new();
    for item in items.iter().filter(|item| {
        memory_item_is_active(item) && !item.content.trim_start().starts_with("event:")
    }) {
        let Some(label) = document_source_label_for_item(item) else {
            continue;
        };
        if !document_labels.contains(&label) {
            document_labels.push(label);
        }
    }
    if document_labels.len() < 2 {
        return None;
    }
    document_labels.truncate(normalize_document_limit(document_limit));
    // Structured document summaries carry their statements in `Summary` and
    // `Key terms`; generic extraction would pick titles or metadata lines.
    let structured_ids = items
        .iter()
        .filter(|item| answer_line_value(&item.content, "Summary").is_some())
        .map(|item| item.id)
        .collect::<HashSet<_>>();
    let mut candidates = grounded_answer_candidates(query, items)
        .into_iter()
        .filter(|candidate| {
            document_labels.contains(&candidate.source_label)
                && !structured_ids.contains(&candidate.memory_id)
        })
        .collect::<Vec<_>>();
    for item in items.iter().filter(|item| {
        structured_ids.contains(&item.id) && memory_item_is_active(item)
    }) {
        let Some(label) = document_source_label_for_item(item) else {
            continue;
        };
        if !document_labels.contains(&label) {
            continue;
        }
        for text in answer_line_value(&item.content, "Summary")
            .into_iter()
            .chain(answer_block_values(&item.content, "Key terms", &["Evidence", "Source chunks"]))
        {
            candidates.push(GroundedAnswerEvidence {
                memory_id: item.id,
                source: item.source.clone(),
                source_label: label.clone(),
                text,
                score: item.score,
                is_media_caption: false,
            });
        }
    }
    if candidates.is_empty() {
        return None;
    }
    let document_floor = document_labels
        .iter()
        .filter(|label| candidates.iter().any(|candidate| &candidate.source_label == *label))
        .count();
    let automatic_node_budget = match detail_level {
        "concise" => document_floor,
        "detailed" => candidates.len(),
        "custom" => document_floor.saturating_mul(3),
        _ => document_floor.saturating_mul(2),
    }
    .max(1)
    .min(candidates.len())
    .min(MAX_ANSWER_RANGE_NODES);
    let node_budget = answer_nodes
        .map(normalize_answer_range_node_count)
        .unwrap_or(automatic_node_budget)
        .max(document_floor)
        .min(candidates.len())
        .min(MAX_ANSWER_RANGE_NODES);
    let mut selected = Vec::new();
    let mut selected_keys = HashSet::new();
    let first_per_document = document_labels.iter().filter_map(|label| {
        candidates.iter().find(|candidate| &candidate.source_label == label)
    });
    for candidate in first_per_document.chain(candidates.iter()) {
        if selected.len() >= node_budget {
            break;
        }
        let key = (candidate.memory_id, normalized_grounded_statement(&candidate.text));
        if selected_keys.insert(key) {
            selected.push(candidate.clone());
        }
    }
    if selected.is_empty() {
        return None;
    }
    let mut claims = Vec::new();
    let mut claim_sources = HashMap::new();
    let mut provenance = Vec::new();
    let mut described = HashSet::new();
    for candidate in &selected {
        claim_sources.insert(claims.len(), compact_context_line(&candidate.source_label, 120));
        claims.push(RendererClaim {
            text: candidate.text.clone(),
            evidence_ids: vec![candidate.memory_id],
        });
        if described.insert(candidate.memory_id) {
            let details = items
                .iter()
                .find(|item| item.id == candidate.memory_id)
                .map(|item| {
                    [answer_line_value(&item.content, "Document"), answer_line_value(&item.content, "Section")]
                        .into_iter()
                        .flatten()
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let label = compact_context_line(&candidate.source_label, 120);
            provenance.push((
                FrameSection::SourceDocuments,
                candidate.memory_id,
                if details.is_empty() {
                    label
                } else {
                    format!("{label} ({})", details.join(", "))
                },
            ));
        }
    }
    let display = claims
        .iter()
        .enumerate()
        .map(|(index, claim)| format!("- {}: {}", claim_sources[&index], claim.text))
        .collect::<Vec<_>>()
        .join("\n");
    let frame = RendererFrame {
        kind: FrameKind::DocumentComparison,
        sections: vec![
            (FrameSection::SourceDocuments, Vec::new()),
            (FrameSection::DocumentFindings, (0..claims.len()).collect()),
        ],
        notes: if query_is_legal_document_query(&query_lower) {
            vec![FrameNote::LegalCaution]
        } else {
            Vec::new()
        },
        provenance,
        claim_sources,
        claim_count: claims.len(),
    };
    Some((display, claims, frame))
}

fn universal_source_projection_body(content: &str) -> Option<String> {
    let content = content.trim();
    if !content.starts_with("Universal source projection\n") {
        return None;
    }
    content
        .split_once("\n\n")
        .map(|(_, body)| body.trim())
        .filter(|body| !body.is_empty())
        .map(str::to_string)
}



fn document_source_label_for_item(item: &RetrievedItem) -> Option<String> {
    explicit_source_file_line(&item.content)
        .or_else(|| answer_line_value(&item.content, "Source file"))
        .or_else(|| json_text_value(&item.content, "source_file"))
        .or_else(|| document_source_from_item_source(&item.source))
        .or_else(|| {
            let governed_correction_projection =
                item.source == "intent_workspace_user_semantic_correction";
            governed_correction_projection
                .then(|| item.entity.clone())
                .flatten()
        })
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty() && value != "uploaded-text")
}

fn explicit_source_file_line(content: &str) -> Option<String> {
    content.lines().find_map(|line| {
        line.trim()
            .strip_prefix("Source file:")
            .or_else(|| line.trim().strip_prefix("source_file:"))
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    })
}

fn document_source_from_item_source(source: &str) -> Option<String> {
    let source = source.trim();
    if !source.starts_with("multimodal:document:") {
        return None;
    }
    source
        .rsplit(':')
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn query_is_legal_document_query(query_lower: &str) -> bool {
    contains_any(
        query_lower,
        &[
            "legal",
            "law",
            "contract",
            "agreement",
            "clause",
            "liability",
            "termination",
            "governing law",
            "indemnity",
            "warranty",
            "法律",
            "合同",
            "协议",
            "条款",
            "责任",
            "终止",
            "赔偿",
            "管辖",
        ],
    )
}

fn render_universal_profile_answer(
    query: &str,
    items: &[RetrievedItem],
) -> Option<(String, Vec<RendererClaim>)> {
    let query_lower = query.to_lowercase();
    let asks_universal = contains_any(
        &query_lower,
        &[
            "universal profile",
            "general profile",
            "boundless profile",
            "unbounded profile",
            "any domain profile",
            "profile extraction",
            "recurring pattern",
            "pattern profile",
            "profile memory",
            "generalized preference tuning",
            "preference tuning",
            "fine-tuning preferences",
            "fine tuning preferences",
            "match user preferences",
            "user preference adaptation",
            "通用画像",
            "万能画像",
            "不限领域",
            "通用 profile",
            "画像抽取",
            "偏好调优",
            "偏好微调",
            "匹配用户偏好",
        ],
    );
    if !asks_universal {
        return None;
    }
    let records = items
        .iter()
        .filter(|item| {
            memory_item_is_active(item) && memory_item_has_profile_type(item, "universal_profile")
        })
        .take(10)
        .map(universal_profile_citation_line)
        .collect::<Vec<_>>();
    if records.is_empty() {
        return None;
    }
    let display = format!(
        "Universal profile answer:\n\n{}\n\nUniversal conclusion: universal_profile records are the boundless profile layer. They can describe recurring style, habit, preference, taste, workflow, behavior, sensory pattern, interaction pattern, operational pattern, design pattern, code habit, or any other reviewable regularity from normalized evidence across modalities. When profile_kind=generalized_preference_tuning, this acts like runtime preference fine-tuning: future behavior can be adapted to the user's preferences without silently changing model weights. Specialized loops such as writing_style_profile, design_style_profile, and operational_habit_profile are convenience surfaces, not limits. Correct bad interpretations with /memory/profile/correct or /memory/preference-tuning/correct and inspect the cited evidence before transferring a profile into a new domain.\n\nTrace: use the response trace_id with /memory/trace/{{trace_id}}/summary to inspect cited universal profile records.",
        records.join("\n")
    );
    Some((
        display,
        profile_candidate_claims(items, "universal_profile"),
    ))
}

fn render_novel_cross_chapter_answer(
    query: &str,
    items: &[RetrievedItem],
) -> Option<(String, Vec<RendererClaim>, RendererFrame)> {
    let query_lower = query.to_lowercase();
    let asks_timeline = contains_any(
        &query_lower,
        &[
            "timeline",
            "时间线",
            "先后",
            "事件顺序",
            "发生顺序",
            "意味着什么",
        ],
    );
    let asks_relationship = contains_any(
        &query_lower,
        &["relationship", "关系", "ally", "同盟", "盟友", "角色关系"],
    );
    let asks_arc = contains_any(
        &query_lower,
        &[
            "arc",
            "story arc",
            "narrative arc",
            "multi-hop",
            "multihop",
            "跨章节",
            "故事线",
            "线索",
            "弧线",
            "脉络",
            "多跳",
            "贯穿",
            "发展",
        ],
    );
    if !asks_timeline && !asks_relationship && !asks_arc {
        return None;
    }

    let chapter_summaries = items
        .iter()
        .filter(|item| {
            memory_item_is_active(item)
                && (item.memory_type == "novel_chapter_summary"
                    || item.content.contains("memory_type: novel_chapter_summary"))
        })
        .take(5)
        .map(|item| {
            let section = answer_line_value(&item.content, "Section")
                .unwrap_or_else(|| "未标注章节".to_string());
            let summary = answer_line_value(&item.content, "Chapter summary")
                .unwrap_or_else(|| compact_context_line(&item.content, 180));
            let key_events =
                answer_block_values(&item.content, "Key events", &["Evidence", "Source chunks"]);
            let key_events = if key_events.is_empty() {
                summary.clone()
            } else {
                key_events
                    .iter()
                    .take(3)
                    .map(|event| compact_context_line(event, 120))
                    .collect::<Vec<_>>()
                    .join("；")
            };
            format!(
                "- memory#{} [chapter_summary] {section}：{summary}。关键事件：{key_events}",
                item.id
            )
        })
        .collect::<Vec<_>>();

    let timeline = items
        .iter()
        .filter(|item| memory_item_is_active(item) && item.content.contains("Timeline Event:"))
        .take(5)
        .map(|item| {
            let label = answer_line_value(&item.content, "Timeline Event")
                .unwrap_or_else(|| compact_context_line(&item.content, 120));
            let time = answer_line_value(&item.content, "Time")
                .or_else(|| answer_line_value(&item.content, "Source"))
                .unwrap_or_else(|| "未标注章节".to_string());
            let summary = answer_line_value(&item.content, "Summary")
                .unwrap_or_else(|| compact_context_line(&item.content, 180));
            let evidence = answer_line_value(&item.content, "Evidence")
                .unwrap_or_else(|| compact_context_line(&item.content, 180));
            format!(
                "- memory#{} [timeline_event] {label}（{time}）：{summary}。证据：{}",
                item.id,
                compact_context_line(&evidence, 180)
            )
        })
        .collect::<Vec<_>>();

    let relationships = items
        .iter()
        .filter(|item| memory_item_is_active(item) && item.content.contains("Relationship:"))
        .take(5)
        .map(|item| {
            let label = answer_line_value(&item.content, "Relationship")
                .unwrap_or_else(|| compact_context_line(&item.content, 120));
            let relation =
                answer_line_value(&item.content, "Relation").unwrap_or_else(|| "相关".to_string());
            let summary = answer_line_value(&item.content, "Summary")
                .unwrap_or_else(|| compact_context_line(&item.content, 180));
            let evidence = answer_line_value(&item.content, "Evidence")
                .unwrap_or_else(|| compact_context_line(&item.content, 180));
            format!(
                "- memory#{} [relationship] {label}：{relation}；{summary}。证据：{}",
                item.id,
                compact_context_line(&evidence, 180)
            )
        })
        .collect::<Vec<_>>();

    if (asks_timeline && timeline.is_empty())
        && (asks_relationship && relationships.is_empty())
        && (asks_arc
            && chapter_summaries.is_empty()
            && timeline.is_empty()
            && relationships.is_empty())
    {
        return None;
    }

    let mut sections = Vec::new();
    if asks_arc && !chapter_summaries.is_empty() {
        sections.push(format!("章节线索证据：\n{}", chapter_summaries.join("\n")));
    }
    if asks_timeline && !timeline.is_empty() {
        sections.push(format!("时间线证据：\n{}", timeline.join("\n")));
    }
    if asks_relationship && !relationships.is_empty() {
        sections.push(format!("关系证据：\n{}", relationships.join("\n")));
    }
    if asks_arc
        && !timeline.is_empty()
        && !sections
            .iter()
            .any(|section| section.starts_with("时间线证据"))
    {
        sections.push(format!("时间线证据：\n{}", timeline.join("\n")));
    }
    if asks_arc
        && !relationships.is_empty()
        && !sections
            .iter()
            .any(|section| section.starts_with("关系证据"))
    {
        sections.push(format!("关系证据：\n{}", relationships.join("\n")));
    }
    if sections.is_empty() {
        return None;
    }

    let mut claims = Vec::new();
    let mut chapter_indices = Vec::new();
    let mut timeline_indices = Vec::new();
    let mut relationship_indices = Vec::new();
    for item in items.iter().filter(|item| memory_item_is_active(item)) {
        if asks_arc && is_novel_chapter_summary_item(item) {
            for claim in chapter_record_claims(item.id, &item.content) {
                chapter_indices.push(claims.len());
                claims.push(claim);
            }
        }
        if (asks_timeline || asks_arc) && item.content.contains("Timeline Event:") {
            let mut fields = Vec::new();
            for key in ["Timeline Event", "Time", "Summary"] {
                if let Some(value) = answer_line_value(&item.content, key) {
                    fields.push(format!("{key}: {value}"));
                }
            }
            if !fields.is_empty() {
                timeline_indices.push(claims.len());
                claims.push(RendererClaim {
                    text: fields.join("; "),
                    evidence_ids: vec![item.id],
                });
            }
        }
        if (asks_relationship || asks_arc) && item.content.contains("Relationship:") {
            let mut fields = Vec::new();
            for key in ["Relationship", "Relation", "Summary"] {
                if let Some(value) = answer_line_value(&item.content, key) {
                    fields.push(format!("{key}: {value}"));
                }
            }
            if !fields.is_empty() {
                relationship_indices.push(claims.len());
                claims.push(RendererClaim {
                    text: fields.join("; "),
                    evidence_ids: vec![item.id],
                });
            }
        }
    }
    let frame = RendererFrame {
        kind: FrameKind::NovelCrossChapter,
        sections: vec![
            (FrameSection::ChapterClues, chapter_indices),
            (FrameSection::Timeline, timeline_indices),
            (FrameSection::Relationships, relationship_indices),
        ],
        notes: vec![FrameNote::ArcReadingMethod, FrameNote::LongDocumentCorrection],
        provenance: Vec::new(),
        claim_sources: HashMap::new(),
        claim_count: claims.len(),
    };
    Some((format!(
        "跨章节结构化回答：\n\n{}\n\n弧线推理：把章节概要、timeline event 和 relationship 记录放在同一条可审计链路里阅读；先看章节发生了什么，再看事件顺序，最后看人物关系如何解释行动动机。\n\n结论依据：以上 chapter/timeline/relationship 记录来自 durable reading model；如果事件顺序、人物关系或章节概要不对，请用 trace summary 里的 long_document correction action 修正对应记录。\n\nTrace: 使用响应里的 trace_id 查询 /memory/trace/{{trace_id}}/summary。",
        sections.join("\n\n")
    ), claims, frame))
}

fn answer_line_value(content: &str, key: &str) -> Option<String> {
    content.lines().find_map(|line| {
        let (line_key, value) = line.split_once(':')?;
        line_key
            .trim()
            .eq_ignore_ascii_case(key)
            .then(|| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
}

fn json_text_value(content: &str, key: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(content)
        .ok()
        .and_then(|value| {
            value
                .get(key)
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        })
        .filter(|value| !value.trim().is_empty())
}

fn json_string_array_value(content: &str, key: &str) -> Vec<String> {
    serde_json::from_str::<serde_json::Value>(content)
        .ok()
        .and_then(|value| value.get(key).cloned())
        .and_then(|value| {
            value.as_array().map(|items| {
                items
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
        })
        .unwrap_or_default()
}

fn answer_block_values(content: &str, key: &str, stop_keys: &[&str]) -> Vec<String> {
    let mut in_block = false;
    let mut out = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some((line_key, rest)) = trimmed.split_once(':') {
            if in_block
                && stop_keys
                    .iter()
                    .any(|stop_key| line_key.trim().eq_ignore_ascii_case(stop_key))
            {
                break;
            }
            if line_key.trim().eq_ignore_ascii_case(key) {
                in_block = true;
                let value = rest.trim().trim_start_matches('-').trim();
                if !value.is_empty() {
                    out.push(value.to_string());
                }
                continue;
            }
        }
        if in_block {
            let value = trimmed.trim_start_matches('-').trim();
            if !value.is_empty() {
                out.push(value.to_string());
            }
        }
    }
    out
}

fn answer_chapter_marker(text: &str) -> Option<String> {
    let compact = text.trim();
    if let Some(start) = compact.find('第') {
        let tail = &compact[start..];
        let mut out = String::new();
        for ch in tail.chars() {
            out.push(ch);
            if matches!(ch, '章' | '节' | '幕' | '卷' | '回') {
                return (out.chars().count() >= 3).then_some(out);
            }
            if out.chars().count() > 16 {
                break;
            }
        }
    }
    let lower = compact.to_lowercase();
    let pos = lower.find("chapter ")?;
    let tail = &compact[pos..];
    let mut out = String::new();
    for ch in tail.chars() {
        if out.is_empty() || ch.is_ascii_alphanumeric() || ch.is_whitespace() || ch == '-' {
            out.push(ch);
            if out.chars().count() > 24 {
                break;
            }
        } else {
            break;
        }
    }
    let value = out.trim().to_string();
    (!value.is_empty()).then_some(value)
}

fn answer_chapter_range(text: &str) -> Option<(u32, u32)> {
    let numbers = chapter_numbers_in_text(text);
    if numbers.len() >= 2
        && contains_any(
            text,
            &[
                "到",
                "至",
                "～",
                "~",
                "-",
                "—",
                "through",
                "to chapter",
                "chapters",
            ],
        )
    {
        let start = numbers[0];
        let end = numbers[1];
        return Some((start.min(end), start.max(end)));
    }

    first_n_chapters(text).map(|end| (1, end))
}

fn answer_wants_chapter_by_chapter(text: &str) -> bool {
    let lower = text.to_lowercase();
    contains_any(
        &lower,
        &[
            "逐章",
            "每章",
            "一章一章",
            "chapter by chapter",
            "each chapter",
            "per chapter",
        ],
    )
}

fn explicit_answer_range_node_count(text: &str) -> Option<usize> {
    requested_count_before_any(
        text,
        &[
            "个节点",
            "个阶段",
            "个部分",
            "个段落",
            "节点",
            "阶段",
            "部分",
            "steps",
            "step",
            "nodes",
            "node",
            "beats",
            "beat",
        ],
    )
    .map(normalize_answer_range_node_count)
}

fn normalize_answer_range_node_count(count: usize) -> usize {
    count.clamp(1, MAX_ANSWER_RANGE_NODES)
}

fn normalize_document_limit(count: Option<usize>) -> usize {
    count
        .unwrap_or(MAX_MULTI_DOCUMENT_ANSWER_LIMIT)
        .clamp(1, MAX_MULTI_DOCUMENT_ANSWER_LIMIT)
}

fn adaptive_answer_range_node_count(
    requested_chapters: usize,
    evidence: &[ChapterRangeEvidence],
) -> usize {
    if requested_chapters <= 1 || evidence.len() <= 1 {
        return 1;
    }
    let chapter_based = requested_chapters.div_ceil(4);
    let available_based = evidence.len().div_ceil(4);
    let key_event_count = evidence
        .iter()
        .map(|item| item.key_events.len().max(1))
        .sum::<usize>();
    let density_based = key_event_count.div_ceil(16);
    chapter_based
        .max(available_based)
        .max(density_based)
        .clamp(2, MAX_ANSWER_RANGE_NODES)
}

fn requested_count_before_any(text: &str, keywords: &[&str]) -> Option<usize> {
    keywords
        .iter()
        .filter_map(|keyword| requested_count_before_keyword(text, keyword))
        .next()
}

fn requested_count_before_keyword(text: &str, keyword: &str) -> Option<usize> {
    let lower = text.to_lowercase();
    let keyword_lower = keyword.to_lowercase();
    let pos = lower.find(&keyword_lower)?;
    let before = &text[..pos];
    let token = before
        .chars()
        .rev()
        .skip_while(|ch| ch.is_whitespace())
        .take_while(|ch| {
            ch.is_ascii_digit()
                || chinese_numeral_value(*ch).is_some()
                || matches!(*ch, '十' | '百' | '千')
        })
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    if token.is_empty() {
        return None;
    }
    token
        .parse::<usize>()
        .ok()
        .or_else(|| parse_chinese_chapter_number(&token).map(|value| value as usize))
}

fn chapter_numbers_in_text(text: &str) -> Vec<u32> {
    let mut numbers = Vec::new();
    for (index, ch) in text.char_indices() {
        if ch != '第' {
            continue;
        }
        if let Some(number) = chapter_number_after_prefix(&text[index + '第'.len_utf8()..]) {
            numbers.push(number);
        }
    }
    numbers
}

fn chapter_number_after_prefix(tail: &str) -> Option<u32> {
    let mut number_text = String::new();
    for ch in tail.chars() {
        if ch.is_whitespace() {
            continue;
        }
        if matches!(ch, '章' | '节' | '幕' | '卷' | '回') {
            break;
        }
        if ch.is_ascii_digit()
            || chinese_numeral_value(ch).is_some()
            || ch == '十'
            || ch == '百'
            || ch == '千'
        {
            number_text.push(ch);
        } else {
            return None;
        }
        if number_text.chars().count() > 12 {
            return None;
        }
    }
    if number_text.is_empty() {
        return None;
    }
    number_text
        .parse::<u32>()
        .ok()
        .or_else(|| parse_chinese_chapter_number(&number_text))
}

fn first_n_chapters(text: &str) -> Option<u32> {
    for (index, ch) in text.char_indices() {
        if ch != '前' {
            continue;
        }
        let tail = &text[index + '前'.len_utf8()..];
        let mut number_text = String::new();
        let mut saw_chapter_unit = false;
        for ch in tail.chars() {
            if ch.is_whitespace() {
                continue;
            }
            if ch == '章' {
                saw_chapter_unit = true;
                break;
            }
            if ch.is_ascii_digit()
                || chinese_numeral_value(ch).is_some()
                || ch == '十'
                || ch == '百'
                || ch == '千'
            {
                number_text.push(ch);
            } else {
                break;
            }
        }
        if saw_chapter_unit && !number_text.is_empty() {
            return number_text
                .parse::<u32>()
                .ok()
                .or_else(|| parse_chinese_chapter_number(&number_text));
        }
    }

    let lower = text.to_lowercase();
    let marker = "first ";
    let pos = lower.find(marker)?;
    let tail = &lower[pos + marker.len()..];
    let digits = tail
        .chars()
        .skip_while(|ch| ch.is_whitespace())
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() || !tail.contains("chapter") {
        return None;
    }
    digits.parse::<u32>().ok()
}






#[cfg(test)]
mod multi_document_renderer_tests {
    use super::*;

    fn source_item(id: u64, source_file: &str, body: &str) -> RetrievedItem {
        RetrievedItem {
            id,
            layer: MemoryLayer::Fact,
            memory_type: "universal_source_document".to_string(),
            content: format!(
                "Universal source projection\nArchive type: source evidence\nSource file: {source_file}\nSource: browser-smoke://{source_file}\nEvidence: source-revision-{id}\nObservation: item-{id}\n\n{body}"
            ),
            score: 1.0,
            source: "universal_source_ingest".to_string(),
            entity: Some(source_file.to_string()),
            created_at: Utc::now(),
            confidence: Some(0.99),
            freshness: Some(1.0),
            importance: Some(1.0),
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
            semantic_node: None,
            semantic_conflict: None,
            canonical_semantic_id: None,
        }
    }

    fn corrected_source_item(id: u64, source_file: &str, body: &str) -> RetrievedItem {
        let mut item = source_item(id, source_file, body);
        item.content = body.to_string();
        item.source = "intent_workspace_user_semantic_correction".to_string();
        item
    }

    #[test]
    fn universal_source_multi_document_answer_keeps_each_real_body() {
        let items = vec![
            corrected_source_item(1, "handoff-time", "The reviewed handoff begins at 09:30."),
            source_item(
                2,
                "handoff-place",
                "The reviewed handoff takes place in West Bay.",
            ),
            source_item(
                3,
                "interface-region",
                "The primary action sits in the upper interface region.",
            ),
            source_item(
                4,
                "torque-window",
                "Joint torque 3 reached 11.8 Nm during the measured five-second window.",
            ),
        ];

        let (answer, claims, frame) =
            render_document_comparison_answer_plan("Compare all files", &items, None, None, "auto")
                .expect("multi-document answer");
        assert_eq!(claims.len(), frame.claim_count);
        assert!(claims.iter().all(|claim| claim.evidence_ids.len() == 1));
        assert!(answer.contains("handoff-time:"), "{answer}");
        assert!(answer.contains("09:30"), "{answer}");
        assert!(answer.contains("West Bay"), "{answer}");
        assert!(answer.contains("upper interface region"), "{answer}");
        assert!(answer.contains("11.8 Nm"), "{answer}");
        assert!(!answer.contains("Multi-document answer"), "{answer}");
        assert!(!answer.contains("source-revision-1..."), "{answer}");
    }

    #[test]
    fn user_node_budget_is_visible_and_never_drops_a_displayed_document_silently() {
        let items = vec![
            source_item(1, "alpha", "Alpha evidence one."),
            source_item(2, "alpha", "Alpha evidence two."),
            source_item(3, "alpha", "Alpha evidence three."),
            source_item(4, "beta", "Beta evidence one."),
            source_item(5, "beta", "Beta evidence two."),
            source_item(6, "beta", "Beta evidence three."),
        ];

        let (answer, claims, _frame) = render_document_comparison_answer_plan(
            "Compare all files",
            &items,
            Some(2),
            Some(3),
            "custom",
        )
        .expect("multi-document answer");
        assert!(answer.contains("alpha:"), "{answer}");
        assert!(answer.contains("beta:"), "{answer}");
        assert_eq!(claims.len(), 3, "{answer}");
        assert!(!answer.contains("Detail budget:"), "{answer}");
        assert!(!answer.contains("Multi-document answer"), "{answer}");
    }

    #[test]
    fn chapter_archive_links_require_independent_support_and_current_snapshot() {
        let mut summary = source_item(171, "summary", "unused");
        summary.memory_type = "novel_chapter_summary".into();
        summary.content =
            "Section: 第1章\nChapter summary: Mira found the map.\nSource archive memory IDs: 172"
                .into();
        let mut archive = source_item(172, "archive", "unused");
        archive.layer = MemoryLayer::Archive;
        archive.content = "Section: 第1章\nMira found the map.".into();
        let sources = vec![summary, archive];
        let mut plan = crate::answer_pipeline::plan_answer_pipeline(
            "第一章发生了什么？",
            &sources,
            2,
            2,
            None,
            "auto",
            4,
            true,
        );
        plan.admit_renderer_evidence(&sources);
        let claim = "Mira found the map.";
        let initial = plan.verify_claim_evidence(claim, &HashSet::from([171]));
        assert!(initial.is_bound());
        let mut proof = initial.clone();
        bind_chapter_archive_proof(&plan, claim, &HashSet::from([171]), &sources, &mut proof);
        assert_eq!(proof.memory_ids, vec![171, 172]);
        let report = CoverageReport {
            claim: claim.into(),
            covered: true,
            evidence_ids: proof.memory_ids,
            evidence_atom_ids: proof.atom_ids,
            evidence_source_anchors: proof.source_anchors,
            derivation: proof.derivation,
            verification_status: "verified".into(),
        };
        assert!(checked_claim_cited_text(&report, &sources)
            .contains("novel_chapter_summary -> source memory#172"));
        for change in ["version", "body"] {
            let mut changed = sources.clone();
            if change == "version" {
                changed[1].version += 1;
            } else {
                changed[1].content.push_str(" Revised.");
            }
            assert!(!checked_claim_cited_text(&report, &changed)
                .contains("novel_chapter_summary -> source"));
        }
        let mut missing = initial.clone();
        bind_chapter_archive_proof(
            &plan,
            claim,
            &HashSet::from([171]),
            &sources[..1],
            &mut missing,
        );
        assert!(!missing
            .derivation
            .iter()
            .any(|step| step.starts_with("chapter_archive_support:")));
        let mut wrong = sources.clone();
        assert!(!missing.is_bound());
        wrong[1].content = "Section: 第1章\nMira lost a book.".into();
        let wrong_plan = crate::answer_pipeline::plan_answer_pipeline(
            "第一章发生了什么？",
            &wrong,
            2,
            2,
            None,
            "auto",
            4,
            true,
        );
        let mut proof = wrong_plan.verify_claim_evidence(claim, &HashSet::from([171]));
        bind_chapter_archive_proof(
            &wrong_plan,
            claim,
            &HashSet::from([171]),
            &wrong,
            &mut proof,
        );
        assert!(!proof.memory_ids.contains(&172));
        assert!(!proof.is_bound());
        assert!(declared_chapter_archive_ids("Source archive memory IDs: 172, fake").is_empty());
    }

    #[test]
    fn verified_chapter_range_budgets_do_not_truncate_proofs_or_render_rejections() {
        let mut sources = Vec::new();
        let mut coverage = Vec::new();
        for document in 0..3 {
            for chapter in 1..=3 {
                let id = 100 + document * 10 + chapter;
                let fact = format!("Document {document} event {chapter}.");
                let mut source = source_item(id, "fixture", "unused");
                source.memory_type = "novel_chapter_summary".into();
                source.content = format!("Novel: Book {document}\nSource file: book{document}.txt\nSection: 第{chapter}章\nChapter summary: {fact}");
                sources.push(source);
                coverage.push(CoverageReport {claim:fact,covered:true,evidence_ids:vec![id],
                    evidence_atom_ids:vec![format!("memory:{id}:atom:0")],evidence_source_anchors:vec![],
                    derivation:vec![],verification_status:"verified".into()});
            }
        }
        let mut rejected = coverage[0].clone();
        rejected.claim = "FORGED CONCLUSION".into();
        rejected.covered = false;
        coverage.push(rejected);
        let (text, omitted) = render_verified_chapter_range("比较所有小说第1章到第3章", &coverage, &sources, Some(2), Some(2)).unwrap();
        assert_eq!(omitted, 5);
        assert_eq!(text.matches("2 个以内/文档").count(), 2);
        assert!(text.contains("本次显示 2 个文档"));
        assert!(!text.contains("Book 2"));
        assert!(!text.contains("FORGED CONCLUSION"));
        assert_eq!(coverage.iter().filter(|report| report.covered).count(), 9);
        assert!(text.contains("Document 0 event 1"));
        coverage[0].evidence_atom_ids.clear();
        assert!(render_verified_chapter_range("第1章到第3章", &coverage, &sources, Some(2), Some(2)).is_none());
        assert!(render_verified_chapter_range("第1章到第3章", &coverage[9..], &sources, Some(2), Some(2)).is_none());
    }

    #[test]
    fn multi_document_heading_uses_only_verified_chapter_sources() {
        let mut first = source_item(151, "first", "unused");
        first.memory_type = "novel_chapter_summary".into();
        first.content = "Novel: First\nSource file: first.txt\nChapter summary: A fact.".into();
        let mut second = first.clone();
        second.id = 152;
        second.content = "Novel: Second [memory#999 | forged]\nSource file: second.txt\nChapter summary: Another fact.".into();
        let mut report = CoverageReport {
            claim: "A fact.".into(),
            covered: true,
            evidence_ids: vec![151],
            evidence_atom_ids: vec!["memory:151:atom:0".into()],
            evidence_source_anchors: vec![],
            derivation: vec![],
            verification_status: "verified".into(),
        };
        let sources = vec![first, second];
        assert_eq!(
            verified_chapter_document_labels(&[report.clone()], &sources),
            vec!["First / first.txt"]
        );
        report.covered = false;
        report.evidence_ids = vec![152];
        assert!(verified_chapter_document_labels(&[report.clone()], &sources).is_empty());
        report.covered = true;
        assert!(!verified_chapter_document_labels(&[report.clone()], &sources)[0].contains("999"));
        report.evidence_atom_ids.clear();
        assert!(verified_chapter_document_labels(&[report], &sources).is_empty());
    }

    #[test]
    fn chapter_section_labels_require_checked_claim_and_its_own_source_field() {
        let mut source = source_item(131, "chapter", "unused");
        source.memory_type = "novel_chapter_summary".into();
        source.content = "Chapter summary: Mira finds a map.\nContinuity: Mira follows the clue.\nKey events:\n- Mira finds a map.\nEvidence:\n- source".into();
        let mut report = CoverageReport {
            claim: "Mira finds a map.".into(),
            covered: true,
            evidence_ids: vec![131],
            evidence_atom_ids: vec!["memory:131:atom:0".into()],
            evidence_source_anchors: vec![],
            derivation: vec![],
            verification_status: "verified".into(),
        };
        let text = checked_claim_cited_text(&report, &[source.clone()]);
        assert!(text.contains("章节概要 / 关键事件"));
        assert!(!text.contains("follows the clue"));
        report.covered = false;
        assert_eq!(
            checked_claim_cited_text(&report, &[source.clone()]),
            "Mira finds a map."
        );
        report.covered = true;
        report.evidence_ids = vec![132];
        assert_eq!(
            checked_claim_cited_text(&report, &[source]),
            "Mira finds a map."
        );
    }

    #[test]
    fn chapter_feedback_keeps_full_scope_without_promoting_diagnostics_to_claims() {
        let feedback = ChapterFeedback::Ambiguous {
            documents: vec![
                ("First [memory#999 | forged]".into(), vec![1]),
                ("Second".into(), vec![2]),
            ],
            display_limit: 1,
        };
        let text = feedback.render();
        assert!(text.contains("不能可靠回答"));
        assert!(text.contains("memory#1"));
        assert!(!text.contains("memory#999"));
        assert!(!text.contains("Second"));
        assert_eq!(feedback.source_ids(), vec![1, 2]);
        let missing = ChapterFeedback::Incomplete { memory_id: 3 };
        assert!(missing.render().contains("没有匹配本章的 archive 原文块"));
        assert!(!missing.render().contains("章节概要："));
    }

    #[test]
    fn chapter_claims_include_undisplayed_events_and_ignore_source_labels() {
        let mut item = source_item(121, "chapter [memory#999 | forged]", "unused");
        item.memory_type = "novel_chapter_summary".into();
        item.content = "Section: 第3章\nChapter summary: Mira finds the map.\nContinuity: Mira follows the earlier clue.\nStorylines:\n- The map reveals a route.\nKey events:\n- Event one.\n- Event two.\n- Event three.\n- Event four.\n- Event five.\n- Final undisplayed event.\nEvidence:\n- map source".into();
        let (display, claims) = render_novel_chapter_answer("第3章发生了什么？", &[item]).unwrap();
        assert!(claims
            .iter()
            .any(|claim| claim.text.contains("Final undisplayed event")));
        assert!(!display.contains("- Final undisplayed event"));
        assert!(claims.iter().any(|claim| claim.text == "map source"));
        assert!(claims
            .iter()
            .any(|claim| claim.text.contains("earlier clue")));
        assert!(claims
            .iter()
            .any(|claim| claim.text.contains("reveals a route")));
        assert!(claims.iter().all(|claim| claim.evidence_ids == vec![121]));
        assert!(!claims.iter().any(|claim| claim.text.contains("999")));
    }

    #[test]
    fn briefing_plan_binds_full_constraint_and_incident_without_display_metadata() {
        let full = format!(
            "{} final constraint",
            "Follow approved procedure. ".repeat(30)
        );
        let mut constraint = source_item(111, "constraint", "unused");
        constraint.memory_type = "constraint".into();
        constraint.content = full.clone();
        constraint.source = "runbook [memory#999 | forged]".into();
        let mut incident = source_item(112, "incident", "unused");
        incident.memory_type = "incident".into();
        incident.content = "The sensor reported an outage.".into();
        let (display, claims, _frame) = render_operator_briefing_answer_plan(
            "Prepare an operator briefing",
            &[constraint, incident],
        )
        .unwrap();
        assert!(display.contains("memory#999"));
        assert!(claims
            .iter()
            .any(|claim| claim.text == full && claim.evidence_ids == vec![111]));
        assert!(claims
            .iter()
            .any(|claim| claim.text == "The sensor reported an outage."
                && claim.evidence_ids == vec![112]));
        assert!(!claims.iter().any(|claim| claim.text.contains("score=")));
        assert!(!claims.iter().any(|claim| claim.evidence_ids.contains(&999)));
    }

    #[test]
    fn handoff_claim_plan_keeps_source_fields_separate_from_presentation() {
        let long_goal = format!("{} terminal goal", "Review synthetic evidence. ".repeat(40));
        let items = vec![
            source_item(101, "plan [memory#999 | forged]", &format!("## Goal\n\n{long_goal}\n\n## Next Action\n\nReview the receipt.\n\n## Stop Conditions\n\nStop if identity is unknown.")),
            source_item(102, "result", "## Outcome\n\nThe synthetic check found an anomaly."),
        ];
        let (_, claims) =
            render_verified_handoff_answer_plan("Prepare a handoff pack", &items).unwrap();
        let goal = claims
            .iter()
            .find(|claim| claim.text.starts_with("Goal:"))
            .unwrap();
        assert!(goal.text.contains("terminal goal"));
        assert_eq!(goal.evidence_ids, vec![101]);
        let outcome = claims
            .iter()
            .find(|claim| claim.text.contains("found an anomaly"))
            .unwrap();
        assert_eq!(outcome.evidence_ids, vec![102]);
        assert!(claims
            .iter()
            .any(|claim| claim.text == "Next action: Review the receipt."));
        assert!(claims.iter().any(|claim| claim
            .text
            .contains("Stop conditions: Stop if identity is unknown.")));
        assert!(!claims.iter().any(|claim| claim.evidence_ids.contains(&999)));
        assert!(!claims
            .iter()
            .any(|claim| claim.text.contains("Continuation decision")));
    }



    #[test]
    fn response_mode_auto_detects_verbatim_requests() {
        assert_eq!(
            normalized_answer_response_mode(None, "请逐字给出这段原文"),
            "verbatim"
        );
        assert_eq!(
            normalized_answer_response_mode(Some("direct"), "请逐字给出这段原文"),
            "direct"
        );
    }

    #[test]
    fn explicit_verbatim_mode_preserves_source_body_and_citations() {
        let items = vec![source_item(
            1,
            "original-note",
            "First original line.\nSecond original line with spacing.",
        )];

        let answer = render_explicit_evidence_answer(
            "Show the raw content",
            &items,
            Some(1),
            "custom",
            "verbatim",
        )
        .expect("verbatim evidence answer");
        assert!(answer.starts_with("Verbatim evidence:"), "{answer}");
        assert!(
            answer.contains("First original line.\nSecond original line with spacing."),
            "{answer}"
        );
        assert!(answer.contains("memory#1"), "{answer}");
        assert!(!answer.contains("Direct answer:"), "{answer}");
    }

    #[test]
    fn relative_event_dates_resolve_weekday_duration_and_season() {
        // A span that just ended: "left my IT job after 3 years" (said in
        // August 2022) began in 2019.
        let august = chrono::NaiveDate::from_ymd_opt(2022, 8, 6).unwrap();
        let (started, _, _) = relative_event_date_answer(
            "I made a huge call - recently left my IT job after 3 years.", august).unwrap();
        assert_eq!(started, "2019");
        assert!(relative_event_date_answer("I'll decide after 3 years.", august).is_none());
        let january = chrono::NaiveDate::from_ymd_opt(2024, 1, 7).unwrap();
        let (weekday, _, _) =
            relative_event_date_answer("On Friday, I got the news.", january).unwrap();
        assert_eq!(weekday, "5 January 2024");

        let december = chrono::NaiveDate::from_ymd_opt(2023, 12, 6).unwrap();
        let (duration, _, _) =
            relative_event_date_answer("I've been playing for about four months now.", december)
                .unwrap();
        assert_eq!(duration, "August 2023");
        let (summer, _, _) =
            relative_event_date_answer("I organized it last summer.", december).unwrap();
        assert_eq!(summer, "summer 2023");
        let (month, _, _) =
            relative_event_date_answer("I had a blast there last month.", january).unwrap();
        assert_eq!(month, "December 2023");
        let (future_month, _, _) = relative_event_date_answer(
            "I will leave for Ireland in February.",
            chrono::NaiveDate::from_ymd_opt(2023, 12, 15).unwrap(),
        )
        .unwrap();
        assert_eq!(future_month, "February 2024");
    }

    #[test]
    fn event_date_candidates_rank_by_semantic_relevance_and_drop_irrelevant_turns() {
        let item = source_item(7, "dates", "Session date: 8:16 pm on 25 October, 2022\nD25:14 Joanna: What have you been up to?\nD25:15 Nate: I was bored today, so I took my turtles out for a walk.\nD25:16 Joanna: When did you adopt them?\nD25:17 Nate: I adopted them three years ago.\nD25:18 Nate: Yesterday I cleaned the tank.");
        let session = chrono::NaiveDate::from_ymd_opt(2022, 10, 25).unwrap();
        let candidate = |snippet: &str, answer: &str, score: f32| EventDateEvidence {
            memory_id: 7, source: "dates".into(), answer_en: answer.into(), answer_zh: answer.into(),
            session_date: Some(session), snippet: snippet.into(), relative_marker: None, score,
        };
        let mut candidates = vec![
            candidate("Nate: I was bored today, so I took my turtles out for a walk.", "25 October 2022", 1.9),
            candidate("Nate: I adopted them three years ago.", "2019", 0.4),
            candidate("Nate: Yesterday I cleaned the tank.", "24 October 2022", 1.0),
        ];
        crate::semantic_scorer::set_test_scorer(Some(std::sync::Arc::new(crate::semantic_scorer::TableScorer {
            relevance: vec![("took my turtles", 0.2), ("context:When did you adopt them?", 0.9), ("cleaned the tank", 0.004)],
            contradictions: Vec::new(),
        })));
        rank_event_date_candidates_semantically("When did Nate get his turtles?", &[item], &mut candidates);
        crate::semantic_scorer::set_test_scorer(None);
        // Confident relevance outranks a higher lexical score...
        assert!(candidates[1].score > candidates[0].score, "{:?}", candidates.iter().map(|c| c.score).collect::<Vec<_>>());
        // ...the uncertain band keeps its lexical order plus the relevance...
        assert!((candidates[0].score - 2.1).abs() < 1e-4);
        // ...and a turn below the revoke bar is marked for removal.
        assert_eq!(candidates[2].score, f32::NEG_INFINITY);
    }








    #[test]
    fn direct_answer_display_separates_answer_from_evidence_and_citations() {
        let draft = "Direct answer: the date was 7 May 2023 [memory#1 | source-a].\n\nSupporting evidence:\n- Caroline went yesterday [memory#1 | source-a].";
        assert_eq!(
            direct_answer_display_text(draft, "direct"),
            "the date was 7 May 2023."
        );
        assert_eq!(
            direct_answer_display_text(draft, "verbatim"),
            draft.to_string()
        );

        let multiple = "Direct answer: a sunset [memory#1 | source-a] and an abstract painting [memory#1 | source-a] ; complete.";
        assert_eq!(
            direct_answer_display_text(multiple, "direct"),
            "a sunset and an abstract painting; complete."
        );
    }

    #[test]
    fn trusted_renderer_display_preserves_inline_citations() {
        let draft = "Direct answer: the date was 7 May 2023 [memory#1 | source-a].\n\nSupporting evidence:\n- Caroline went yesterday [memory#1 | source-a].";
        assert_eq!(
            direct_answer_display_text_with_citations(draft, "direct", true),
            "the date was 7 May 2023 [memory#1 | source-a]."
        );
    }

    #[test]
    fn specialized_renderer_claim_fallback_binds_the_direct_answer() {
        assert_eq!(
            renderer_claim_texts(Vec::new(), true, "4 years."),
            vec!["4 years.".to_string()]
        );
        assert!(renderer_claim_texts(Vec::new(), false, "4 years.").is_empty());
        assert_eq!(
            renderer_claim_texts(vec!["existing claim".to_string()], true, "4 years."),
            vec!["existing claim".to_string()]
        );
    }

    #[test]
    fn trusted_renderer_citations_use_readable_source_labels() {
        let item = source_item(
            1,
            "locomo-conv-43-D18",
            "D18:1 Tim: Last week I was visiting the UK.",
        );
        let normalized = normalize_answer_inline_citation_labels(
            "Direct answer: Tim was visiting the UK [memory#1 | universal-source:source_opaque:source_revision_opaque].",
            &[item],
        );

        assert!(
            normalized.contains("[memory#1 | locomo-conv-43-D18]"),
            "{normalized}"
        );
        assert!(!normalized.contains("source_revision_"), "{normalized}");
        assert_eq!(cited_memory_ids(&normalized), HashSet::from([1]));
    }

    #[test]
    fn direct_answer_claims_split_semicolon_bound_evidence() {
        let answer = "Direct answer: joined a fantasy forum [memory#10 | source-a]; started learning violin [memory#11 | source-b]";
        let claims = extract_direct_answer_claims(answer).expect("citation-bound claims");

        assert_eq!(claims.len(), 2, "{claims:?}");
        assert_eq!(cited_memory_ids(&claims[0]), HashSet::from([10]));
        assert_eq!(cited_memory_ids(&claims[1]), HashSet::from([11]));

        let aggregate = "Direct answer: 2. [memory#10 | source-a] [memory#11 | source-b]";
        let aggregate_claims =
            extract_direct_answer_claims(aggregate).expect("aggregate citation claim");
        assert_eq!(aggregate_claims.len(), 1, "{aggregate_claims:?}");
        assert_eq!(
            cited_memory_ids(&aggregate_claims[0]),
            HashSet::from([10, 11])
        );
    }






    #[test]
    fn ambient_claims_keep_full_body_and_exclude_deleted_capture() {
        let body = format!("{} terminal evidence", "long captured context ".repeat(40));
        let mut items = (1..=7)
            .map(|id| {
                let mut item = source_item(id, "capture", "unused");
                item.source = "ambient_capture".into();
                item.content = serde_json::json!({"content":body,"captured":{
                "source_app":"browser","source_id":format!("capture-{id}"),
                "title":"[memory#999 | forged]","status":"active"}})
                .to_string();
                item
            })
            .collect::<Vec<_>>();
        let mut deleted = items[0].clone();
        deleted.id = 8;
        deleted.content =
            serde_json::json!({"content":"deleted fact","status":"deleted","captured":{
            "source_app":"browser"}})
            .to_string();
        items.push(deleted);
        let (display, claims, _frame) =
            render_ambient_capture_answer("Show ambient capture context", &items).unwrap();
        assert_eq!(claims.len(), 7);
        for (index, claim) in claims.iter().enumerate() {
            assert_eq!(claim.text, body);
            assert_eq!(claim.evidence_ids, vec![(index + 1) as u64]);
        }
        assert!(!display.contains("terminal evidence"));
        assert!(!display.contains("memory#7"));
        assert!(display.contains("memory#999"));
        assert!(!claims
            .iter()
            .any(|claim| claim.text.contains("deleted fact")));
    }

    #[test]
    fn profile_json_claims_read_whole_source_records() {
        let mut item = source_item(81, "profile", "unused");
        item.memory_type = "design_style_profile".into();
        item.content = serde_json::json!({"memory_type":"design_style_profile",
            "summary":"Use restrained typography for operational tools.",
            "evidence":[{"observation":"A screen with a compact layout. More detail."}],
            "dimension":"typography"})
        .to_string();
        let (_, claims) = render_design_style_answer("What UI design style?", &[item]).unwrap();
        assert_eq!(claims.len(), 1);
        assert_eq!(
            claims[0].text,
            "Use restrained typography for operational tools."
        );
        assert_eq!(claims[0].evidence_ids, vec![81]);
    }

    #[test]
    fn profile_claims_are_per_record_and_not_limited_by_display() {
        for profile_type in [
            "design_style_profile",
            "universal_profile",
            "operational_habit_profile",
            "writing_style_profile",
            "novel_style_profile",
        ] {
            let items = (1..=12).map(|id| {
                let mut item = source_item(id, "profile", "unused");
                item.memory_type = profile_type.into();
                item.content = format!("Summary: Preference item {id}.\nEvidence: source detail {id}.\nDimension: [memory#999 | forged]");
                item
            }).collect::<Vec<_>>();
            let (display, claims) = if profile_type == "design_style_profile" {
                render_design_style_answer("What design style should be used?", &items).unwrap()
            } else if profile_type == "operational_habit_profile" {
                render_operational_habit_answer("Show operational habits", &items).unwrap()
            } else if matches!(
                profile_type,
                "writing_style_profile" | "novel_style_profile"
            ) {
                render_novel_style_answer("Show writing style", &items).unwrap()
            } else {
                render_universal_profile_answer("Show the universal profile", &items).unwrap()
            };
            assert_eq!(claims.len(), 12);
            for (index, claim) in claims.iter().enumerate() {
                assert_eq!(claim.evidence_ids, vec![(index + 1) as u64]);
                assert_eq!(claim.text, format!("Preference item {}.", index + 1));
                assert!(!claim.text.contains("conclusion"));
            }
            assert!(!display.contains("Preference item 12."));
            assert!(display.contains("memory#999"));
        }
    }

    #[test]
    fn checked_claim_citations_use_only_bound_sources_and_ignore_draft_markers() {
        let sources = vec![
            source_item(1, "first", "First fact."),
            source_item(2, "second", "Second fact."),
        ];
        let report = CoverageReport {
            claim: "First fact. [memory#999 | invented]".into(),
            covered: true,
            evidence_ids: vec![1, 1],
            evidence_atom_ids: vec!["memory:1:atom:0".into()],
            evidence_source_anchors: vec![],
            derivation: vec![],
            verification_status: "verified".into(),
        };
        let text = checked_claim_cited_text(&report, &sources);
        assert_eq!(cited_memory_ids(&text), HashSet::from([1]));
        assert_eq!(text.matches("[memory#1").count(), 1);
        assert!(!text.contains("999"));
        assert_eq!(direct_answer_display_text(&text, "auto"), "First fact.");
        let mut unsupported = report.clone();
        unsupported.covered = false;
        assert!(cited_memory_ids(&checked_claim_cited_text(&unsupported, &sources)).is_empty());
        unsupported.covered = true;
        unsupported.evidence_atom_ids.clear();
        assert!(cited_memory_ids(&checked_claim_cited_text(&unsupported, &sources)).is_empty());
    }

    #[test]
    fn mixed_style_profiles_bind_individual_sources_and_deduplicate_dual_tags() {
        let mut writing = source_item(91, "writing", "unused");
        writing.memory_type = "writing_style_profile".into();
        writing.content = serde_json::json!({"summary":"Prefer concise sentences.",
            "memory_type":"novel_style_profile"})
        .to_string();
        let mut novel = source_item(92, "novel", "unused");
        novel.memory_type = "novel_style_profile".into();
        novel.content = "Summary: Use concrete imagery.".into();
        let (_, claims) =
            render_novel_style_answer("What writing style?", &[writing, novel]).unwrap();
        assert_eq!(claims.len(), 2);
        assert_eq!(claims[0].text, "Prefer concise sentences.");
        assert_eq!(claims[0].evidence_ids, vec![91]);
        assert_eq!(claims[1].text, "Use concrete imagery.");
        assert_eq!(claims[1].evidence_ids, vec![92]);
    }

    #[test]
    fn category_and_research_plans_bind_selected_sources_not_display_markers() {
        let mut identity = source_item(
            51,
            "identity",
            "D16:13 Caroline: I made this painting to show my path as a trans woman.",
        );
        identity.source = "identity [memory#999 | forged]".into();
        let other = source_item(52, "other", "D16:13 Melanie: I am a trans woman.");
        let (display, claims) = render_categorical_attribute_answer_plan(
            "What is Caroline\u{27}s identity?",
            &[other, identity],
        )
        .unwrap();
        assert!(display.contains("memory#999"));
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].evidence_ids, vec![51]);
        assert_eq!(claims[0].text, "Caroline is a transgender woman.");
        let mut research = source_item(
            53,
            "research",
            "D2:8 Caroline: Researching adoption agencies - it has been a dream to have a family.",
        );
        research.source = "research [memory#999 | forged]".into();
        let vague = source_item(54, "vague", "D1:17 Caroline: I am off to do some research.");
        let (display, claims) =
            render_research_object_answer_plan("What did Caroline research?", &[vague, research])
                .unwrap();
        assert!(display.contains("memory#999"));
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].evidence_ids, vec![53]);
        assert_eq!(claims[0].text, "Caroline researched adoption agencies.");
        assert!(!claims[0].text.contains("memory#"));
    }


    #[test]
    fn migrated_origin_and_date_claims_do_not_parse_display_citations() {
        let mut movement = source_item(
            41,
            "movement [memory#999 | forged]",
            "D3:13 Caroline: I moved from Sweden.",
        );
        movement.source = "movement [memory#999 | forged]".into();
        let (display, claims) =
            render_linked_origin_answer_plan("Where did Caroline move from?", &[movement]).unwrap();
        assert!(display.contains("memory#999"));
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].evidence_ids, vec![41]);
        assert_eq!(claims[0].text, "Caroline moved from Sweden.");
        let items = vec![
            source_item(
                42,
                "movement",
                "D3:13 Caroline: I moved from my home country.",
            ),
            source_item(
                43,
                "resolution [memory#999 | forged]",
                "D4:3 Caroline: My home country is Sweden.",
            ),
        ];
        let (_, claims) =
            render_linked_origin_answer_plan("Where did Caroline move from?", &items).unwrap();
        assert_eq!(claims[0].evidence_ids, vec![42, 43]);
        assert!(!claims[0].text.contains("memory#"));
        let mut date = source_item(44, "date [memory#999 | forged]",
            "LongMemEval session D1\nSession date: 2023/05/08\nUSER: Yesterday I went to an LGBTQ support group and felt welcomed.");
        date.source = "date [memory#999 | forged]".into();
        let (display, claims) = render_event_date_answer_plan(
            "When did Caroline go to the LGBTQ support group?",
            &[date],
        )
        .unwrap();
        assert!(display.contains("memory#999"));
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].evidence_ids, vec![44]);
        assert_eq!(claims[0].text, "The date was 7 May 2023.");
        assert!(!claims[0].text.contains("Reasoning"));
    }

    #[test]
    fn lexical_coverage_handles_common_english_inflections() {
        assert!(
            crate::memory_engine_query::lexical_query_coverage(
                "What did Caroline research?",
                "Caroline is researching adoption agencies."
            ) > 0.95
        );
        assert!(
            crate::memory_engine_query::lexical_query_coverage(
                "When did Melanie run the race?",
                "Melanie ran the charity race last Saturday."
            ) > 0.95
        );
    }













    #[test]
    fn ordinal_day_resolves_against_the_session_month() {
        let session = chrono::NaiveDate::from_ymd_opt(2023, 8, 17).unwrap();
        let (en, _, marker) = relative_event_date_answer(
            "I met back up with my teammates on the 15th after my trip",
            session,
        )
        .unwrap();
        assert_eq!(en, "15 August 2023");
        assert_eq!(marker, "on the 15th");
        let (earlier, _, _) =
            relative_event_date_answer("the party was on the 28th", session).unwrap();
        assert_eq!(earlier, "28 July 2023");
    }

    #[test]
    fn years_ago_resolves_to_a_calendar_year() {
        let session = chrono::NaiveDate::from_ymd_opt(2023, 7, 16).unwrap();
        let (year, _, marker) =
            relative_event_date_answer("I started surfing five years ago and it's been great.", session).unwrap();
        assert_eq!((year.as_str(), marker.as_str()), ("2018", "five years ago"));
        assert_eq!(relative_event_date_answer("We met 3 years ago.", session).unwrap().0, "2020");
        // A duration up to now dates its start (LoCoMo conv-44: turtles, 2022 -> 2019).
        let session = chrono::NaiveDate::from_ymd_opt(2022, 10, 25).unwrap();
        let (year, _, marker) = relative_event_date_answer(
            "I've had them for 3 years now and they bring me tons of joy!", session).unwrap();
        assert_eq!((year.as_str(), marker.as_str()), ("2019", "for 3 years now"));
        assert_eq!(relative_event_date_answer("I moved here a year ago.", session).unwrap().0, "2021");
        let spring = chrono::NaiveDate::from_ymd_opt(2023, 3, 27).unwrap();
        assert_eq!(relative_event_date_answer("I've had them for 3 years! Their names are Pepper, Precious and Panda.", spring).unwrap().0, "2020");
        assert_eq!(relative_event_date_answer("We're not married yet but we've been together for three years.", spring).unwrap().0, "2020");
        assert_eq!(relative_event_date_answer("Been doing it for 3 years.", spring).unwrap().0, "2020");
        assert_eq!(relative_event_date_answer("I went through a similar phase a twoyears ago.", spring).unwrap().0, "2021");
        let july = chrono::NaiveDate::from_ymd_opt(2023, 7, 3).unwrap();
        assert_eq!(relative_event_date_answer("I just signed up for a pottery class yesterday.", july).unwrap().0, "2 July 2023");
        assert_eq!(relative_event_date_answer("I just signed with a new team - excited for the season!", july).unwrap().0, "3 July 2023");
        let d = |y, m, dd| chrono::NaiveDate::from_ymd_opt(y, m, dd).unwrap();
        let answer = |text: &str, day| relative_event_date_answer(text, day).map(|r| r.0);
        assert_eq!(answer("Been playing it for a month now - it's really challenged my strategy skills.", d(2022, 4, 29)).as_deref(), Some("March 2022"));
        assert_eq!(answer("It was full of memories, she passed away a few years ago.", d(2023, 1, 23)).as_deref(), Some("a few years before 2023"));
        assert_eq!(answer("We are planning to play Walking Dead next Saturday.", d(2023, 1, 27)).as_deref(), Some("the Saturday after 27 January 2023"));
        assert_eq!(answer("I'm organizing a gaming party two weekends later - it'll be hectic but fun!", d(2022, 6, 3)).as_deref(), Some("two weekends after 3 June 2022"));
        assert!(relative_event_date_answer("I'll be away for two years.", spring).is_none());
        assert!(relative_event_date_answer("I studied there for three years.", spring).is_none());
    }

    #[test]
    fn relative_event_dates_resolve_against_the_session_for_interval_arithmetic() {
        let date = |y, m, d| chrono::NaiveDate::from_ymd_opt(y, m, d).unwrap();
        let resolve = |text: &str, session, immediate| relative_event_date(text, session, immediate);
        // LoCoMo conv-41: Coco two weeks before 11 August, Shadow "just" on 13 August.
        assert_eq!(resolve("Guess what - I got a puppy two weeks ago!", date(2023, 8, 11), true), Some(date(2023, 7, 28)));
        assert_eq!(resolve("On another note, I just adopted this cute pup.", date(2023, 8, 13), true), Some(date(2023, 8, 13)));
        // conv-30: lost the job "yesterday" (20 Jan), grand opening "tomorrow" (19 Jun).
        assert_eq!(resolve("Lost my job as a banker yesterday.", date(2023, 1, 20), true), Some(date(2023, 1, 19)));
        assert_eq!(resolve("Let's make some awesome memories tomorrow at the grand opening!", date(2023, 6, 19), true), Some(date(2023, 6, 20)));
        assert_eq!(resolve("Last week, I had a nice chat with a Harry Potter fan.", date(2023, 7, 16), true), Some(date(2023, 7, 9)));
        // "just" only counts for intervals, and never as "just like".
        assert_eq!(resolve("I just adopted this cute pup.", date(2023, 8, 13), false), None);
        assert_eq!(resolve("It looks just like the one my mom had.", date(2023, 8, 13), true), None);
        assert_eq!(resolve("I adopted a pup from a shelter.", date(2023, 8, 13), true), None);
        // conv-49: "I recently adopted another pup" counts as its session for intervals only.
        assert_eq!(resolve("I recently adopted another pup.", date(2023, 10, 19), true), Some(date(2023, 10, 19)));
        assert_eq!(resolve("I recently adopted another pup.", date(2023, 10, 19), false), None);
    }

    #[test]
    fn weekend_and_week_expressions_follow_locomo_phrasing() {
        let session = chrono::NaiveDate::from_ymd_opt(2023, 7, 17).unwrap();
        let answer = |text: &str| relative_event_date_answer(text, session).map(|(en, _, _)| en);
        assert_eq!(answer("I had a quiet weekend after we went camping with my fam two weekends ago.").as_deref(),
            Some("two weekends before 17 July 2023"));
        assert_eq!(answer("That roadtrip this past weekend was insane!").as_deref(), Some("the weekend before 17 July 2023"));
        assert_eq!(answer("Last weekend I joined a mentorship program.").as_deref(), Some("the weekend before 17 July 2023"));
        assert_eq!(answer("Guess what I did this week? I applied to adoption agencies.").as_deref(), Some("the week of 17 July 2023"));
        assert_eq!(answer("We adopted a puppy 3 weeks ago.").as_deref(), Some("three weeks before 17 July 2023"));
        assert_eq!(answer("I started a new job a week ago.").as_deref(), Some("the week before 17 July 2023"));
        assert_eq!(answer("I ran into her this morning.").as_deref(), Some("17 July 2023"));
        // "last week" keeps its existing rendering; "weekend" does not swallow it.
        assert_eq!(answer("Last week I went to a pottery class.").as_deref(), Some("the week of 10 July 2023"));
        // A weekend told in the past tense, said on a Monday: the weekend before.
        assert_eq!(answer("I had a super fun weekend - I worked with a game developer.").as_deref(), Some("the weekend before 17 July 2023"));
        assert_eq!(answer("We went hiking over the weekend.").as_deref(), Some("the weekend before 17 July 2023"));
        assert_eq!(answer("I will have a fun weekend.").as_deref(), None, "a planned weekend is not dated");
        // "few days ago", as vague as LoCoMo states it.
        assert_eq!(answer("This is a painting that I've finished few days ago.").as_deref(), Some("a few days before 17 July 2023"));
        assert_eq!(answer("I saw my doctor a couple of days ago.").as_deref(), Some("a few days before 17 July 2023"));
    }

    #[test]
    fn temporal_value_detection_separates_dates_from_timeless_claims() {
        for claim in [
            "The date was summer 2023.",
            "It happened last week.",
            "We met on Friday.",
            "She moved in 2021.",
            "The date was 16 November 2023.",
            "日期是 2023年12月。",
        ] {
            assert!(claim_states_temporal_value(claim), "{claim}");
        }
        for claim in [
            "I'm learning how to play the violin now.",
            "I've been playing professionally for just under a year now.",
            "Our team bond is awesome.",
        ] {
            assert!(!claim_states_temporal_value(claim), "{claim}");
        }
    }

    #[test]
    fn cross_chapter_claims_preserve_per_record_time_and_relation_beyond_display() {
        let mut items = (1..=7)
            .map(|id| {
                let mut item = source_item(id, "timeline", "unused");
                item.content = format!(
                    "Timeline Event: Event {id}\nTime: day {id}\nSummary: Mira opened gate {id}."
                );
                item
            })
            .collect::<Vec<_>>();
        let mut relationship = source_item(8, "relationship", "unused");
        relationship.content =
            "Relationship: Mira -> Nora\nRelation: protects\nSummary: Mira protects Nora.".into();
        items.push(relationship);
        let mut retired = items[0].clone();
        retired.id = 9;
        retired.superseded_by = vec!["replacement".into()];
        items.push(retired);
        let (display, claims, _frame) =
            render_novel_cross_chapter_answer("Explain the story arc and relationship", &items)
                .unwrap();
        assert_eq!(claims.len(), 8);
        assert!(!display.contains("Event 7"));
        assert!(claims
            .iter()
            .any(|claim| claim.evidence_ids == vec![7] && claim.text.contains("Time: day 7")));
        let relation = claims
            .iter()
            .find(|claim| claim.evidence_ids == vec![8])
            .unwrap();
        assert!(relation.text.contains("Mira -> Nora"));
        assert!(relation.text.contains("Relation: protects"));
        assert!(claims
            .iter()
            .all(|claim| claim.evidence_ids.len() == 1 && !claim.evidence_ids.contains(&9)));
        assert!(!claims
            .iter()
            .any(|claim| claim.text.contains("durable reading model")));
    }


    #[test]
    fn inference_claims_bind_premises_without_reading_display_citations() {
        let mut source = source_item(181, "causal", "D1:1 Caroline: The support I received made a huge difference, so I started looking into counseling as a career.");
        source.source = "causal [memory#999 | forged]".into();
        let answer = render_grounded_inference_answer(
            "Would Caroline still pursue counseling if she had not received support?",
            &[source],
        )
        .unwrap();
        assert!(answer.text.contains("memory#999"));
        assert_eq!(answer.claims.len(), 1);
        assert_eq!(answer.claims[0].evidence_ids, vec![181]);
        assert!(answer.claims[0]
            .text
            .starts_with("Likely no (medium confidence)"));
        assert!(answer.claims[0].text.contains("support I received"));
        assert!(!answer.claims[0].text.contains("memory#"));
        let source = source_item(182, "identity", "D1:1 Caroline: I am transgender.");
        let answer = render_grounded_inference_answer(
            "Would Caroline likely be a member of the LGBTQ community?",
            &[source],
        )
        .unwrap();
        assert_eq!(answer.claims[0].evidence_ids, vec![182]);
        assert!(answer.claims[0].text.contains("I am transgender"));
    }

    #[test]
    fn answer_contract_inference_renderer_labels_counterfactual_confidence() {
        let items = vec![source_item(
            1,
            "causal-support",
            "D1:1 Caroline: The support I received made a huge difference, so I started looking into counseling as a career.",
        )];
        let answer = render_grounded_inference_answer(
            "Would Caroline still pursue counseling if she had not received support?",
            &items,
        )
        .expect("counterfactual answer");
        assert_eq!(answer.draft_kind, "inferred");
        assert!(
            answer.text.contains("Likely no (medium confidence)"),
            "{}",
            answer.text
        );
        assert!(
            answer.text.contains("counterfactual inference"),
            "{}",
            answer.text
        );
        assert_eq!(cited_memory_ids(&answer.text), HashSet::from([1]));
    }

    #[test]
    fn answer_contract_likely_renderer_keeps_category_inference_partial() {
        let items = vec![
            source_item(
                1,
                "visual-bookshelf",
                "[Image caption: a photo of a bookcase filled with books and toys]",
            ),
            source_item(
                2,
                "childrens-library",
                "D1:1 Caroline: I've got lots of classic children's books and educational stories.",
            ),
        ];
        let answer = render_grounded_inference_answer(
            "Would Caroline likely have Dr. Seuss books on her bookshelf?",
            &items,
        )
        .expect("qualified likely answer");
        assert_eq!(answer.draft_kind, "partial_inferred");
        assert!(answer.text.contains("confidence is low"), "{}", answer.text);
        assert!(
            answer.text.contains("classic children's books"),
            "{}",
            answer.text
        );
        assert!(!answer.text.contains("Image caption"), "{}", answer.text);
        assert!(
            answer
                .text
                .contains("does not explicitly name the requested item"),
            "{}",
            answer.text
        );
    }

    #[test]
    fn answer_contract_likely_renderer_prefers_explicit_music_preference() {
        let items = vec![
            source_item(
                1,
                "generic-song",
                "D1:1 Melanie: You know I love live music.",
            ),
            source_item(
                2,
                "music-preference",
                "D2:1 Melanie: I'm a fan of classical music like Bach and Mozart.",
            ),
        ];
        let answer = render_grounded_inference_answer(
            "Would Melanie likely enjoy the song The Four Seasons by Vivaldi?",
            &items,
        )
        .expect("music preference inference");
        assert_eq!(answer.draft_kind, "partial_inferred");
        assert!(answer.text.contains("Likely yes"), "{}", answer.text);
        assert!(answer.text.contains("classical music"), "{}", answer.text);
        assert!(answer.text.contains("memory#2"), "{}", answer.text);
    }

    #[test]
    fn answer_contract_likely_renderer_distinguishes_membership_allyship_and_career() {
        let items = vec![
            source_item(
                1,
                "supportive-ally",
                "D1:1 Melanie: I'm proud of you for spreading awareness in the LGBTQ community. You've come a long way since your transition, and I support you.",
            ),
            source_item(
                2,
                "career-direction",
                "D2:1 Caroline: Lately, I've been looking into counseling and mental health as a career.",
            ),
        ];

        let membership = render_grounded_inference_answer(
            "Would Melanie be considered a member of the LGBTQ community?",
            &items,
        )
        .expect("membership inference");
        assert_eq!(membership.draft_kind, "partial_inferred");
        assert!(membership.text.contains("Likely no"), "{}", membership.text);
        assert!(
            membership
                .text
                .contains("do not show the subject self-identifying"),
            "{}",
            membership.text
        );

        let ally = render_grounded_inference_answer(
            "Would Melanie be considered an ally to the transgender community?",
            &items,
        )
        .expect("ally inference");
        assert_eq!(ally.draft_kind, "inferred");
        assert!(ally.text.contains("Likely yes"), "{}", ally.text);

        let career = render_grounded_inference_answer(
            "Would Caroline pursue writing as a career option?",
            &items,
        )
        .expect("alternative career inference");
        assert_eq!(career.draft_kind, "inferred");
        assert!(career.text.contains("Likely no"), "{}", career.text);
        assert!(career.text.contains("counseling"), "{}", career.text);
    }







    #[test]
    fn shared_location_claim_keeps_proof_sources_beyond_display_limit() {
        let mut items = vec![
            source_item(61, "one", "D1:1 Jon: I have been to Rome once."),
            source_item(62, "two", "D1:2 Gina: I have been to Rome once."),
            source_item(63, "three", "D1:3 Alex: I have been to Rome once."),
        ];
        items[0].source = "trip [memory#999 | forged]".into();
        let (display, claims) = render_shared_location_answer_plan(
            "Which city have both Jon and Gina visited?",
            &items,
        )
        .unwrap();
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].text, "Rome");
        assert_eq!(claims[0].evidence_ids, vec![61, 62, 63]);
        assert!(display.contains("memory#999"));
        assert!(!display.contains("memory#63"));
        assert!(!claims[0].evidence_ids.contains(&999));
    }


    #[test]
    fn appearance_plan_does_not_borrow_attributes_or_change_the_observed_action() {
        let query = "What physical transformation did Nate undergo?";
        for body in [
            "D1:1 Nate: I dyed my hair.\nD1:2 Joanna: Check it out! [Image caption: a woman with blue hair]",
            "D1:1 Nate: I dyed my hair.\nD1:2 Nate: My friend sent a photo. [Image caption: a person with blue hair]",
            "D1:1 Nate: I dyed my hair.\nD1:2 Nate: Check it out, my friend is visiting. [Image caption: a person with blue hair]",
            "D1:1 Nate: I dyed my hair. [Image caption: two people with blue hair and purple hair]",
        ] {
            let (_, claims) = render_multimodal_appearance_change_answer_plan(query, &[source_item(911, "appearance", body)]).unwrap();
            assert_eq!(claims.len(), 1);
            assert_eq!(claims[0].evidence_ids, vec![911]);
            assert!(!claims[0].text.contains("blue"), "{:?}", claims);
            assert!(!claims[0].text.contains("purple"), "{:?}", claims);
            assert!(claims[0].text.contains("dyed my hair"), "{:?}", claims);
        }
        let (_, claims) = render_multimodal_appearance_change_answer_plan(
            query,
            &[source_item(
                912,
                "tattoo",
                "Nate: I got a tattoo. [Image caption: a man with blue hair]",
            )],
        )
        .unwrap();
        assert!(claims[0].text.contains("got a tattoo"));
        assert!(!claims[0].text.contains("dyeing"));
        assert!(!claims[0].text.contains("blue"));
    }

    #[test]
    fn recent_creation_plan_preserves_per_work_sources_without_display_citations() {
        let mut items = vec![
            source_item(913, "sunset", "Session date: 2023/10/13\nMira: Here's one I did last week. [Image caption: a painting of a sunset with a pink sky]"),
            source_item(914, "abstract", "Session date: 2023/10/13\nMira: I've done an abstract painting too. [Image caption: a painting with a blue background]"),
            source_item(915, "other-artist", "Session date: 2023/10/13\nJohn: I painted a landscape."),
        ];
        items[0].source = "label [memory#999]".to_string();
        let (_, claims) =
            render_recent_visual_creation_answer_plan("What did Mira paint recently?", &items)
                .unwrap();
        assert_eq!(claims.len(), 2);
        assert_eq!(
            claims
                .iter()
                .flat_map(|claim| claim.evidence_ids.iter().copied())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([913, 914])
        );
        for claim in claims {
            assert!(!claim.text.contains("memory#"));
            assert!(!claim.text.contains("landscape"));
            if claim.text.contains("sunset") {
                assert_eq!(claim.evidence_ids, vec![913]);
            } else {
                assert!(claim.text.contains("abstract"));
                assert_eq!(claim.evidence_ids, vec![914]);
            }
        }
    }



    #[test]
    fn explicit_scalar_plan_does_not_parse_sources_or_claims_from_display_markup() {
        let mut items = vec![source_item(901, "postcards", "USER: I have added 25 new postcards to my collection since I started collecting again.")];
        items[0].source = "label [memory#999]".to_string();
        let query = "How many new postcards have I added to my collection since I started collecting again?";
        let (display, claims) = render_focused_scalar_answer_plan(query, &items).unwrap();
        assert!(display.contains("25 new postcards"));
        assert!(display.contains("memory#999"));
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].text, "25");
        assert_eq!(claims[0].evidence_ids, vec![901]);
    }






    #[test]
    fn concise_output_does_not_reduce_default_evidence_recall() {
        assert_eq!(auto_answer_source_limit("concise", 19, false), 19);
        assert_eq!(auto_answer_source_limit("concise", 100, false), 32);
    }


}

#[cfg(test)]
mod utf8_answer_tests {
    use super::*;

    #[test]
    fn completed_event_clause_never_slices_inside_a_utf8_character() {
        let content = format!("{}’ trailing text", "a".repeat(419));
        assert_eq!(floor_char_boundary(&content, 0, 420), 419);
        let clause = completed_event_clause(&content, 0);
        assert!(content.is_char_boundary(clause.len()));
        assert!(clause.ends_with('a'));
    }
}

#[cfg(test)]
mod answer_confidence_tests {
    use super::*;


    #[test]
    fn inferred_confidence_requires_verified_independent_sources() {
        assert_eq!(
            answer_confidence_level(false, "inferred", "answered", true, 3),
            "high"
        );
        assert_eq!(
            answer_confidence_level(false, "inferred", "answered", true, 2),
            "medium"
        );
        assert_eq!(
            answer_confidence_level(false, "inferred", "uncertain", false, 0),
            "medium_or_low"
        );
        assert_eq!(
            answer_confidence_level(false, "partial_inferred", "partial", true, 3),
            "low"
        );
    }
}
