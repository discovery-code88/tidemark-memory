use super::*;

#[derive(Debug, Clone)]
pub(crate) struct JointRetrievalCandidate {
    pub(crate) id: u64,
    pub(crate) score: f32,
    pub(crate) lexical_score: f32,
    pub(crate) source_projection: bool,
    pub(crate) source_group: Option<String>,
    pub(crate) first_relation_score: f32,
    pub(crate) second_relation_score: f32,
}

fn retrieval_stopword(token: &str) -> bool {
    matches!(
        token,
        "a" | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "be"
            | "been"
            | "between"
            | "by"
            | "can"
            | "did"
            | "do"
            | "does"
            | "for"
            | "from"
            | "had"
            | "has"
            | "have"
            | "how"
            | "i"
            | "in"
            | "is"
            | "it"
            | "many"
            | "me"
            | "my"
            | "of"
            | "on"
            | "or"
            | "our"
            | "since"
            | "some"
            | "that"
            | "the"
            | "this"
            | "to"
            | "was"
            | "were"
            | "what"
            | "when"
            | "which"
            | "with"
            | "would"
    )
}

fn meaningful_retrieval_tokens(text: &str) -> Vec<String> {
    let mut tokens = MemoryEngine::tokenize(text)
        .into_iter()
        .filter(|token| token.chars().count() > 1)
        .filter(|token| !retrieval_stopword(token))
        .collect::<Vec<_>>();
    tokens.sort();
    tokens.dedup();
    tokens
}

fn retrieval_token_forms(token: &str) -> Vec<String> {
    let token = token.to_lowercase();
    let mut forms = vec![token.clone()];
    if !token.is_ascii() {
        return forms;
    }
    let mut push = |value: String| {
        if value.chars().count() > 1 && !forms.contains(&value) {
            forms.push(value);
        }
    };
    if let Some(root) = match token.as_str() {
        "went" | "gone" => Some("go"),
        "ran" => Some("run"),
        "gave" | "given" => Some("give"),
        "met" => Some("meet"),
        "took" | "taken" => Some("take"),
        "thought" => Some("think"),
        _ => None,
    } {
        push(root.to_string());
    }
    if token.len() > 4 && token.ends_with("ies") {
        push(format!("{}y", &token[..token.len() - 3]));
    }
    if token.len() > 5 && token.ends_with("ing") {
        let mut root = token[..token.len() - 3].to_string();
        if root.len() > 2 {
            let mut chars = root.chars().rev();
            if chars.next() == chars.next() {
                root.pop();
            }
        }
        push(root.clone());
        push(format!("{root}e"));
    }
    if token.len() > 4 && token.ends_with("ed") {
        let mut root = token[..token.len() - 2].to_string();
        if root.len() > 2 {
            let mut chars = root.chars().rev();
            if chars.next() == chars.next() {
                root.pop();
            }
        }
        push(root.clone());
        push(format!("{root}e"));
    }
    if token.len() > 3 && token.ends_with('s') && !token.ends_with("ss") {
        push(token[..token.len() - 1].to_string());
    }
    forms
}

pub(crate) fn lexical_query_coverage(query: &str, content: &str) -> f32 {
    let query = query.trim().to_lowercase();
    let content = content.to_lowercase();
    if query.is_empty() || content.is_empty() {
        return 0.0;
    }
    if content.contains(&query) {
        return 1.0;
    }
    let terms = meaningful_retrieval_tokens(&query);
    if terms.is_empty() {
        return 0.0;
    }
    let content_tokens = MemoryEngine::tokenize(&content)
        .into_iter()
        .flat_map(|token| retrieval_token_forms(&token))
        .collect::<HashSet<_>>();
    terms
        .iter()
        .filter(|term| {
            retrieval_token_forms(term)
                .iter()
                .any(|form| content_tokens.contains(form))
        })
        .count() as f32
        / terms.len() as f32
}

fn clean_relation_clause(value: &str) -> String {
    value
        .trim()
        .trim_matches(|character: char| {
            character.is_ascii_punctuation()
                || matches!(character, '，' | '。' | '？' | '！' | '：' | '；')
        })
        .trim()
        .to_string()
}

pub(crate) fn temporal_relation_clauses(query: &str) -> Option<(String, String)> {
    let lower = query.to_lowercase();
    if let Some(between_start) = lower.find("between ") {
        let value_start = between_start + "between ".len();
        let tail = &query[value_start..];
        let tail_lower = &lower[value_start..];
        if let Some(and_offset) = tail_lower.find(" and ") {
            let first = clean_relation_clause(&tail[..and_offset]);
            let second = clean_relation_clause(&tail[and_offset + " and ".len()..]);
            if !first.is_empty() && !second.is_empty() {
                return Some((first, second));
            }
        }
    }
    if let Some(from_start) = lower.find("from ") {
        let value_start = from_start + "from ".len();
        let tail = &query[value_start..];
        let tail_lower = &lower[value_start..];
        if let Some(to_offset) = tail_lower.find(" to ") {
            let first = clean_relation_clause(&tail[..to_offset]);
            let second = clean_relation_clause(&tail[to_offset + " to ".len()..]);
            if !first.is_empty() && !second.is_empty() {
                return Some((first, second));
            }
        }
    }

    if let Some(from_start) = query.find('从') {
        let tail = &query[from_start + '从'.len_utf8()..];
        if let Some(to_offset) = tail.find('到') {
            let first = clean_relation_clause(&tail[..to_offset]);
            let second = clean_relation_clause(&tail[to_offset + '到'.len_utf8()..]);
            if !first.is_empty() && !second.is_empty() {
                return Some((first, second));
            }
        }
    }
    let between_end = query.find("之间")?;
    let before = &query[..between_end];
    for separator in ['与', '和'] {
        if let Some(separator_at) = before.rfind(separator) {
            let first = clean_relation_clause(&before[..separator_at]);
            let second =
                clean_relation_clause(&before[separator_at + separator.len_utf8()..between_end]);
            if !first.is_empty() && !second.is_empty() {
                return Some((first, second));
            }
        }
    }
    None
}

pub(crate) fn query_requests_latest_state(query: &str) -> bool {
    let query = query.to_lowercase();
    [
        " latest ",
        " current ",
        " now ",
        " since ",
        "most recent",
        "up to date",
        "目前",
        "当前",
        "现在",
        "最新",
        "以来",
        "至今",
    ]
    .iter()
    .any(|marker| format!(" {query} ").contains(marker))
}

fn push_unique_candidate(
    selected: &mut Vec<JointRetrievalCandidate>,
    selected_ids: &mut HashSet<u64>,
    candidate: &JointRetrievalCandidate,
    top_k: usize,
) {
    if selected.len() < top_k && selected_ids.insert(candidate.id) {
        selected.push(candidate.clone());
    }
}

pub(crate) fn select_joint_retrieval_candidates(
    mut candidates: Vec<JointRetrievalCandidate>,
    top_k: usize,
    preserve_relation_sides: bool,
) -> Vec<JointRetrievalCandidate> {
    if top_k == 0 || candidates.is_empty() {
        return Vec::new();
    }
    candidates.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
    });

    let mut seen_source_groups = HashSet::new();
    let candidates = candidates
        .into_iter()
        .filter(|candidate| {
            if !candidate.source_projection {
                return true;
            }
            candidate
                .source_group
                .as_ref()
                .map(|group| seen_source_groups.insert(group.clone()))
                .unwrap_or(true)
        })
        .collect::<Vec<_>>();
    let best_score = candidates
        .first()
        .map(|candidate| candidate.score)
        .unwrap_or(0.0);
    let mut selected = Vec::new();
    let mut selected_ids = HashSet::new();

    if preserve_relation_sides && top_k >= 2 {
        if let Some(candidate) = candidates.iter().max_by(|left, right| {
            left.first_relation_score
                .partial_cmp(&right.first_relation_score)
                .unwrap_or(Ordering::Equal)
        }) {
            if candidate.first_relation_score >= 0.15 {
                push_unique_candidate(&mut selected, &mut selected_ids, candidate, top_k);
            }
        }
        if let Some(candidate) = candidates.iter().max_by(|left, right| {
            left.second_relation_score
                .partial_cmp(&right.second_relation_score)
                .unwrap_or(Ordering::Equal)
        }) {
            if candidate.second_relation_score >= 0.15 {
                push_unique_candidate(&mut selected, &mut selected_ids, candidate, top_k);
            }
        }
    }

    if top_k >= 2 {
        let source = candidates.iter().find(|candidate| {
            candidate.source_projection
                && (candidate.lexical_score >= 0.15 || candidate.score >= best_score * 0.70)
        });
        let durable = candidates.iter().find(|candidate| {
            !candidate.source_projection
                && (candidate.lexical_score >= 0.15 || candidate.score >= best_score * 0.70)
        });
        if let (Some(source), Some(durable)) = (source, durable) {
            push_unique_candidate(&mut selected, &mut selected_ids, source, top_k);
            push_unique_candidate(&mut selected, &mut selected_ids, durable, top_k);
        }
    }

    for candidate in &candidates {
        push_unique_candidate(&mut selected, &mut selected_ids, candidate, top_k);
        if selected.len() == top_k {
            break;
        }
    }
    selected.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
    });
    selected
}

impl MemoryEngine {
    pub(crate) fn retrieve_ids(
        &self,
        conversation_id: Option<&str>,
        query: &str,
        top_k: usize,
    ) -> Vec<u64> {
        self.retrieve_ids_with_scope(conversation_id, query, top_k, None)
    }

    pub(crate) fn retrieve_ids_with_scope(
        &self,
        conversation_id: Option<&str>,
        query: &str,
        top_k: usize,
        scope: Option<&ScopedQueryContext>,
    ) -> Vec<u64> {
        let q_tokens = Self::tokenize(query);
        if crate::answer_execution::answer_worker_cancelled() { return Vec::new(); }
        let temporal_clauses = temporal_relation_clauses(query);
        let intent = detect_query_intent(query);
        let chapter_query = exact_chapter_marker(query);
        let (current_project, current_task) = self.current_applicability_context(conversation_id);
        let mut active_fact_ids: HashSet<u64> = HashSet::new();
        let mut latest_facts: HashMap<(String, String, String), &MemoryRecord> = HashMap::new();
        for m in &self.memories {
            if crate::answer_execution::answer_worker_cancelled() { return Vec::new(); }
            if m.layer != MemoryLayer::Fact {
                continue;
            }
            let (Some(entity), Some(key)) = (&m.entity, &m.fact_key) else {
                continue;
            };
            let conversation_key = m.conversation_id.clone();
            let dedupe_value = if key.ends_with("_precondition")
                || key.ends_with("_recovery_pattern")
                || key.ends_with("_failure_motif")
                || key.ends_with("_success_pattern")
            {
                m.fact_value.clone().unwrap_or_default()
            } else {
                String::new()
            };
            let k = (
                conversation_key,
                entity.clone(),
                format!("{}::{}", key, dedupe_value),
            );
            let replace = match latest_facts.get(&k) {
                Some(current) => {
                    let candidate_active = m.superseded_by.is_empty()
                        && memory_quality_text_value(&m.content, &["superseded_by"]).is_none();
                    let current_active = current.superseded_by.is_empty()
                        && memory_quality_text_value(&current.content, &["superseded_by"])
                            .is_none();
                    (candidate_active && !current_active)
                        || (candidate_active == current_active
                            && ((m.version > current.version)
                                || (m.version == current.version && m.trust > current.trust)
                                || (m.version == current.version
                                    && (m.trust - current.trust).abs() < f32::EPSILON
                                    && m.created_at > current.created_at)))
                }
                None => true,
            };
            if replace {
                latest_facts.insert(k, m);
            }
        }
        for m in latest_facts.values() {
            active_fact_ids.insert(m.id);
        }

        let is_eligible = |m: &MemoryRecord| {
            if should_exclude_from_retrieval(m) {
                return false;
            }
            if m.layer == MemoryLayer::Activation && !Self::activation_memory_is_live(m) {
                return false;
            }
            if !self.memory_visible_to_scope(m.id, scope) {
                return false;
            }
            if conversation_id
                .map(|cid| m.conversation_id != cid)
                .unwrap_or(false)
            {
                return false;
            }
            if m.layer != MemoryLayer::Fact {
                return true;
            }
            if m.entity.is_none() || m.fact_key.is_none() {
                return true;
            }
            active_fact_ids.contains(&m.id)
        };

        let scheduled_layers = schedule_for_intent(intent)
            .filter(|schedule| {
                self.memories.iter().any(|m| {
                    is_eligible(m) && memory_fits_schedule(intent, m, &schedule.preferred_layers)
                })
            })
            .map(|schedule| schedule.preferred_layers);

        // Semantic channel: best-chunk embedding similarity, normalized over
        // the eligible records of this query. Absent when no embedding model
        // is configured or too many records still await indexing.
        let semantic_scores = crate::semantic_retrieval::global_index().and_then(|index| {
            let records = self
                .memories
                .iter()
                .filter(|m| is_eligible(m))
                .map(|m| (m.id, m.content.as_str()))
                .collect::<Vec<_>>();
            index.scores(query, &records)
        });
        let semantic_weight = crate::semantic_retrieval::retrieval_weight();
        let scored = self
            .memories
            .iter()
            .take_while(|_| !crate::answer_execution::answer_worker_cancelled())
            .filter(|m| is_eligible(m))
            .filter(|m| {
                scheduled_layers
                    .as_ref()
                    .map(|layers| {
                        m.layer == MemoryLayer::Activation
                            || memory_fits_schedule(intent, m, layers)
                    })
                    .unwrap_or(true)
            })
            .filter(|m| {
                semantic_scope_allows_query(
                    m.semantic_node.as_ref(),
                    query,
                    current_project.as_deref(),
                    current_task.as_deref(),
                )
            })
            .map(|m| {
                let vector = Self::cosine_tf(&q_tokens, &m.tokens);
                let keyword = Self::jaccard(&q_tokens, &m.tokens);
                let lexical_score = lexical_query_coverage(query, &m.content);
                let time = Self::time_decay(m.created_at);
                let trust = m.trust;

                let mut score = 0.45 * vector + 0.30 * keyword + 0.15 * time + 0.10 * trust;
                score += 0.30 * lexical_score;
                score += semantic_weight
                    * semantic_scores
                        .as_ref()
                        .and_then(|scores| scores.get(&m.id))
                        .copied()
                        .unwrap_or(0.0);
                if m.content.to_lowercase().contains(&query.to_lowercase()) {
                    score += 0.08;
                }
                score += layer_boost(intent, &m.layer);
                score += Self::activation_score_bonus(m);
                if let Some(layers) = scheduled_layers.as_ref() {
                    score += schedule_rank_boost(layers, &m.layer);
                }
                // Semantic-aware retrieval boost
                score += semantic_status_score_adjustment(m.semantic_node.as_ref());
                score += semantic_scope_score_adjustment(m.semantic_node.as_ref(), query);
                score += semantic_kind_score_adjustment(m.semantic_node.as_ref(), query);
                score += semantic_applicability_score_adjustment(
                    m.semantic_node.as_ref(),
                    query,
                    current_project.as_deref(),
                    current_task.as_deref(),
                );
                score += self.episodic_query_score(m, query);
                score += multimodal_query_score(m, query);
                score += Self::parametric_query_score(m, query);
                score += memory_quality_score_boost(m, query);
                if chapter_query
                    .as_deref()
                    .map(|marker| novel_memory_matches_chapter(&m.content, marker))
                    .unwrap_or(false)
                {
                    score += 0.65;
                }
                let (first_relation_score, second_relation_score) = temporal_clauses
                    .as_ref()
                    .map(|(first, second)| {
                        (
                            lexical_query_coverage(first, &m.content),
                            lexical_query_coverage(second, &m.content),
                        )
                    })
                    .unwrap_or((0.0, 0.0));
                score += 0.14 * first_relation_score.max(second_relation_score);
                let source_projection = m
                    .memory_type
                    .as_deref()
                    .is_some_and(|memory_type| memory_type.starts_with("universal_source_"));
                JointRetrievalCandidate {
                    id: m.id,
                    score,
                    lexical_score,
                    source_projection,
                    source_group: source_projection
                        .then(|| m.entity.clone().unwrap_or_else(|| m.source.clone())),
                    first_relation_score,
                    second_relation_score,
                }
            })
            .collect::<Vec<_>>();

        let mut selected_ids =
            select_joint_retrieval_candidates(scored, top_k, temporal_clauses.is_some())
                .into_iter()
                .map(|candidate| candidate.id)
                .collect::<Vec<_>>();

        // A pinned activation is an explicit control-plane instruction for the
        // conversation, not merely another lexical hit. Once the ordinary
        // authorization, conversation scope, liveness, and retrieval
        // eligibility checks above pass, it must remain in the frozen answer
        // candidate pool even when the question wording does not mention it.
        // This deliberately does not run for an explicit answer selection:
        // selected_memory_ids is an exact, fail-closed scope.
        if top_k > 0 {
            let mut seen = selected_ids.iter().copied().collect::<HashSet<_>>();
            for pinned_id in self.pinned_activation_ids_for_scope(conversation_id, scope) {
                if seen.insert(pinned_id) {
                    selected_ids.push(pinned_id);
                }
            }
        }
        selected_ids
    }

    /// Return pinned activation records that are safe to force into a query's
    /// candidate pool. Query relevance is intentionally not part of this
    /// predicate: pinning is the relevance decision. Visibility and normal
    /// answer eligibility remain mandatory so this helper cannot widen scope
    /// or revive cleared/superseded/deprecated records.
    pub(crate) fn pinned_activation_ids_for_scope(
        &self,
        conversation_id: Option<&str>,
        scope: Option<&ScopedQueryContext>,
    ) -> Vec<u64> {
        let mut ids = self
            .memories
            .iter()
            .filter(|memory| {
                conversation_id
                    .map(|conversation| memory.conversation_id == conversation)
                    .unwrap_or(true)
                    && memory.layer == MemoryLayer::Activation
                    && memory
                        .activation_metadata
                        .as_ref()
                        .is_some_and(|metadata| metadata.status == "pinned")
                    && Self::activation_memory_is_live(memory)
                    && crate::memory_engine_summary::memory_record_is_eligible_for_selected_answer(
                        memory,
                    )
                    && self.memory_visible_to_scope(memory.id, scope)
            })
            .map(|memory| memory.id)
            .collect::<Vec<_>>();
        ids.sort_unstable();
        ids
    }

    pub(crate) fn memory_visible_to_scope(
        &self,
        memory_id: u64,
        scope: Option<&ScopedQueryContext>,
    ) -> bool {
        let Some(scope) = scope else {
            return true;
        };
        let Some(visibility) = self.shared_memory_visibility.get(&memory_id) else {
            return true;
        };
        visibility.agent_id.as_deref() == Some(scope.agent_id.as_str())
            || visibility
                .shared_block
                .as_ref()
                .map(|block| scope.shared_blocks.contains(block))
                .unwrap_or(false)
    }

    pub(crate) fn ingest_with_visibility(
        &mut self,
        req: IngestRequest,
        visibility: MemoryVisibility,
    ) -> (IngestResponse, Vec<u64>) {
        self.ingest_with_visibility_and_quality(req, visibility, MemoryQualityPatch::default())
    }

    pub(crate) fn ingest_with_visibility_and_quality(
        &mut self,
        req: IngestRequest,
        visibility: MemoryVisibility,
        quality: MemoryQualityPatch,
    ) -> (IngestResponse, Vec<u64>) {
        let conversation_id = req.conversation_id.clone();
        let before_len = self.memories.len();
        let response = self.ingest_with_reserved_id_and_quality(req, None, quality);
        let created_ids = self
            .memories
            .iter()
            .skip(before_len)
            .map(|memory| memory.id)
            .collect::<Vec<_>>();
        if response.accepted {
            for memory_id in &created_ids {
                self.shared_memory_visibility
                    .insert(*memory_id, visibility.clone());
            }
            self.invalidate_cache_for_conversation(Some(conversation_id.as_str()));
        }
        (response, created_ids)
    }
}

fn novel_memory_matches_chapter(content: &str, marker: &str) -> bool {
    content
        .to_lowercase()
        .contains(&marker.to_ascii_lowercase())
}

fn exact_chapter_marker(text: &str) -> Option<String> {
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
