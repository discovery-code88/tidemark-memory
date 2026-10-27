use super::*;

impl MemoryEngine {
    pub(crate) fn episodic_query_score(&self, memory: &MemoryRecord, query: &str) -> f32 {
        let Some(meta) = memory.episodic_metadata.as_ref() else {
            return 0.0;
        };
        let query = query.to_lowercase();
        let wants_failure =
            query.contains("失败") || query.contains("出错") || query.contains("问题");
        let wants_cause =
            query.contains("为什么") || query.contains("导致") || query.contains("原因");
        let wants_recovery =
            query.contains("恢复") || query.contains("补救") || query.contains("怎么修");
        let wants_similarity = query.contains("类似")
            || query.contains("相似")
            || query.contains("之前")
            || query.contains("同样");
        let wants_actor_action =
            query.contains("谁") || query.contains("执行") || query.contains("做了什么");

        let mut score = 0.0;
        if wants_failure && meta.failure_type.is_some() {
            score += 0.12;
        }
        if wants_cause && !meta.cause_links.is_empty() {
            score += 0.10;
        }
        if wants_recovery && !meta.recovery_links.is_empty() {
            score += 0.10;
        }
        if wants_similarity {
            if let Some(failure_type) = meta.failure_type.as_ref() {
                if query.contains(&failure_type.to_lowercase()) {
                    score += 0.12;
                }
            }
            if meta
                .action_result_pairs
                .iter()
                .any(|pair| query.contains(&pair.to_lowercase()) || pair.contains("failed"))
            {
                score += 0.08;
            }
        }
        if wants_actor_action && !meta.actor_action_pairs.is_empty() {
            score += 0.06;
        }
        score += self.semantic_failure_motif_bridge_score(memory, &query, meta);
        score
    }

    pub(crate) fn parametric_query_score(memory: &MemoryRecord, query: &str) -> f32 {
        if memory.layer != MemoryLayer::Parametric {
            return 0.0;
        }
        let q = query.to_lowercase();
        let wants_parametric = [
            "参数",
            "阈值",
            "配置",
            "默认",
            "policy",
            "config",
            "threshold",
            "setting",
        ]
        .iter()
        .any(|pattern| q.contains(pattern));
        if !wants_parametric {
            return 0.0;
        }

        let mut score = 0.18;
        if memory
            .fact_key
            .as_ref()
            .map(|key| q.contains(&key.to_lowercase()))
            .unwrap_or(false)
        {
            score += 0.18;
        }
        if memory
            .fact_value
            .as_ref()
            .map(|value| q.contains(&value.to_lowercase()))
            .unwrap_or(false)
        {
            score += 0.10;
        }
        if q.contains("当前") || q.contains("默认") {
            score += 0.06;
        }
        score
    }

    pub(crate) fn semantic_failure_motif_bridge_score(
        &self,
        memory: &MemoryRecord,
        query: &str,
        meta: &EpisodicMetadata,
    ) -> f32 {
        let Some(failure_type) = meta.failure_type.as_ref() else {
            return 0.0;
        };
        let query_mentions_failure_mode = query.contains("失败模式")
            || query.contains("类似")
            || query.contains("相似")
            || query.contains("问题")
            || query.contains("脆弱");
        if !query_mentions_failure_mode {
            return 0.0;
        }

        let failure_type_lower = failure_type.to_lowercase();
        let related_text = meta
            .cause_links
            .iter()
            .chain(meta.recovery_links.iter())
            .map(|item| item.to_lowercase())
            .collect::<Vec<_>>();

        self.memories
            .iter()
            .filter(|candidate| {
                candidate.conversation_id == memory.conversation_id
                    && candidate.layer == MemoryLayer::Fact
                    && candidate
                        .fact_key
                        .as_deref()
                        .map(|key| key.ends_with("_failure_motif"))
                        .unwrap_or(false)
            })
            .filter_map(|candidate| {
                let motif = candidate.fact_value.as_ref()?.to_lowercase();
                let motif_matches_episode = failure_type_lower.contains(&motif)
                    || motif.contains(&failure_type_lower)
                    || Self::char_bigram_overlap(&failure_type_lower, &motif) >= 0.2
                    || related_text.iter().any(|item| {
                        item.contains(&motif) || Self::char_bigram_overlap(item, &motif) >= 0.2
                    });
                if !motif_matches_episode {
                    return None;
                }

                let query_overlap = Self::char_bigram_overlap(query, &motif);
                let generic_log_bridge = query.contains("日志") && motif.contains("日志");
                if query_overlap <= 0.0 && !generic_log_bridge {
                    return None;
                }
                Some(0.14 + query_overlap * 0.20 + if generic_log_bridge { 0.06 } else { 0.0 })
            })
            .fold(0.0_f32, f32::max)
    }

    pub(crate) fn refresh_episodic_metadata_for_anchor(
        &mut self,
        conversation_id: &str,
        anchor_memory_id: u64,
    ) {
        let episodic_ids = self
            .memories
            .iter()
            .filter(|memory| {
                memory.conversation_id == conversation_id
                    && memory.layer == MemoryLayer::Episodic
                    && memory.anchor_ids.contains(&anchor_memory_id)
            })
            .map(|memory| memory.id)
            .collect::<Vec<_>>();

        for episodic_id in episodic_ids {
            let anchors = self
                .memories
                .iter()
                .find(|memory| memory.id == episodic_id)
                .map(|memory| memory.anchor_ids.clone())
                .unwrap_or_default();
            let ordered = anchors
                .iter()
                .filter_map(|anchor_id| self.memories.iter().find(|memory| memory.id == *anchor_id))
                .cloned()
                .collect::<Vec<_>>();
            let metadata = build_episodic_metadata(&ordered);
            if let Some(episodic) = self
                .memories
                .iter_mut()
                .find(|memory| memory.id == episodic_id)
            {
                episodic.episodic_metadata = Some(metadata);
            }
        }
    }

    pub(crate) fn maybe_auto_consolidate(&mut self, conversation_id: &str) {
        let latest_summary_at = self
            .memories
            .iter()
            .filter(|m| {
                m.conversation_id == conversation_id && m.layer == MemoryLayer::SessionSummary
            })
            .map(|m| m.created_at)
            .max();

        let new_high_level_count = self
            .memories
            .iter()
            .filter(|m| m.conversation_id == conversation_id)
            .filter(|m| {
                matches!(
                    m.layer,
                    MemoryLayer::Episodic | MemoryLayer::SelfModel | MemoryLayer::Autobiographical
                )
            })
            .filter(|m| {
                latest_summary_at
                    .map(|cutoff| m.created_at > cutoff)
                    .unwrap_or(true)
            })
            .count();

        let has_high_value_event = self
            .memories
            .iter()
            .filter(|m| m.conversation_id == conversation_id && m.layer == MemoryLayer::Event)
            .filter(|m| {
                latest_summary_at
                    .map(|cutoff| m.created_at > cutoff)
                    .unwrap_or(true)
            })
            .any(|m| {
                m.event_metadata
                    .as_ref()
                    .map(|metadata| {
                        metadata.memory_action == "consolidate"
                            && metadata.value_weight >= self.auto_consolidation.min_value_weight
                    })
                    .unwrap_or(false)
            });

        if new_high_level_count >= self.auto_consolidation.min_high_level_memories
            || has_high_value_event
        {
            let _ = self.consolidate_conversation(conversation_id);
        }
    }

    pub(crate) fn query_runtime_snapshot(
        &self,
        conversation_id: Option<&str>,
        cache_key: &str,
    ) -> QueryRuntimeSnapshot {
        let mut eligible_count = 0;
        let mut has_live_activation = false;
        let mut active_project: Option<&MemoryRecord> = None;
        let mut current_task: Option<&MemoryRecord> = None;
        for memory in &self.memories {
            if conversation_id
                .map(|cid| memory.conversation_id != cid)
                .unwrap_or(false)
            {
                continue;
            }
            eligible_count += 1;
            has_live_activation |= Self::activation_memory_is_live(memory);
            if memory.layer != MemoryLayer::Fact {
                continue;
            }
            let slot = match memory.fact_key.as_deref() {
                Some("active_project") => &mut active_project,
                Some("next_step") => &mut current_task,
                _ => continue,
            };
            let replace = slot
                .map(|current| {
                    memory.version > current.version
                        || (memory.version == current.version
                            && memory.created_at > current.created_at)
                })
                .unwrap_or(true);
            if replace {
                *slot = Some(memory);
            }
        }
        QueryRuntimeSnapshot {
            eligible_count,
            has_live_activation,
            cached_ids: self.get_cached_ids(cache_key),
            current_project: active_project.and_then(|memory| memory.fact_value.clone()),
            current_task: current_task.and_then(|memory| memory.fact_value.clone()),
        }
    }

    pub(crate) fn current_applicability_context(
        &self,
        conversation_id: Option<&str>,
    ) -> (Option<String>, Option<String>) {
        let mut active_project: Option<&MemoryRecord> = None;
        let mut next_step: Option<&MemoryRecord> = None;
        for memory in &self.memories {
            if conversation_id
                .map(|cid| memory.conversation_id != cid)
                .unwrap_or(false)
                || memory.layer != MemoryLayer::Fact
            {
                continue;
            }
            let slot = match memory.fact_key.as_deref() {
                Some("active_project") => &mut active_project,
                Some("next_step") => &mut next_step,
                _ => continue,
            };
            let replace = slot
                .map(|current| {
                    memory.version > current.version
                        || (memory.version == current.version
                            && memory.created_at > current.created_at)
                })
                .unwrap_or(true);
            if replace {
                *slot = Some(memory);
            }
        }
        (
            active_project.and_then(|memory| memory.fact_value.clone()),
            next_step.and_then(|memory| memory.fact_value.clone()),
        )
    }

    pub(crate) fn mark_retrieved(&mut self, ids: &[u64]) -> Vec<(u64, u32, Option<DateTime<Utc>>)> {
        if ids.is_empty() {
            return Vec::new();
        }

        let now = Utc::now();
        let id_set = ids.iter().copied().collect::<HashSet<_>>();
        let mut updates = Vec::new();
        for memory in &mut self.memories {
            if id_set.contains(&memory.id) {
                memory.retrieval_hits = memory.retrieval_hits.saturating_add(1);
                memory.last_retrieved_at = Some(now);
                updates.push((memory.id, memory.retrieval_hits, memory.last_retrieved_at));
            }
        }
        updates
    }

    pub(crate) fn normalized_top_k(top_k: Option<usize>, max: usize) -> usize {
        top_k.unwrap_or(8).max(1).min(max)
    }

    pub(crate) fn cache_scope(conversation_id: Option<&str>) -> String {
        conversation_id.unwrap_or("*").to_string()
    }

    pub(crate) fn cache_key(conversation_id: Option<&str>, query: &str, top_k: usize) -> String {
        format!(
            "conv={}::top_k={}::query={}",
            Self::cache_scope(conversation_id),
            top_k,
            query
        )
    }

    pub(crate) fn invalidate_cache_for_conversation(&mut self, conversation_id: Option<&str>) {
        self.candidate_cache_revision = self.candidate_cache_revision.wrapping_add(1);
        let global_prefix = format!("conv={}::", Self::cache_scope(None));
        let conversation_prefix = format!("conv={}::", Self::cache_scope(conversation_id));
        self.cache.retain(|key, _| {
            !key.starts_with(&global_prefix) && !key.starts_with(&conversation_prefix)
        });
    }

    pub(crate) fn invalidate_all_cache(&mut self) {
        self.candidate_cache_revision = self.candidate_cache_revision.wrapping_add(1);
        self.cache.clear();
    }

    pub(crate) fn retrieve(&mut self, req: &QueryRequest) -> QueryResponse {
        let top_k = Self::normalized_top_k(req.top_k, 30);
        let key = Self::cache_key(req.conversation_id.as_deref(), &req.query, top_k);
        if let Some(cached) = self.cache.get(&key) {
            if cached.expires_at > Utc::now() {
                let ids = cached.ids.clone();
                self.mark_retrieved(&ids);
                return QueryResponse {
                    cached: true,
                    items: self.materialize_items(&ids, &req.query),
                    vector_retrieval: None,
                    performance: None,
                };
            }
        }

        let ids = self.retrieve_ids(req.conversation_id.as_deref(), &req.query, top_k);
        self.mark_retrieved(&ids);
        self.cache.insert(
            key,
            CacheEntry {
                query: req.query.clone(),
                ids: ids.clone(),
                expires_at: Utc::now() + Duration::minutes(5),
            },
        );

        QueryResponse {
            cached: false,
            items: self.materialize_items(&ids, &req.query),
            vector_retrieval: None,
            performance: None,
        }
    }

    pub(crate) fn list_memories(&self, req: &MemoryListRequest) -> MemoryListResponse {
        let limit = req.limit.unwrap_or(100).clamp(1, 500);
        let offset = req.offset.unwrap_or(0);
        let text_filter = req.text.as_deref().map(|value| value.trim().to_lowercase());
        let text_filter = text_filter.as_deref().filter(|value| !value.is_empty());

        let mut layer_counts = HashMap::new();
        let mut filtered = self
            .memories
            .iter()
            .filter(|memory| {
                req.conversation_id
                    .as_deref()
                    .map(|conversation_id| memory.conversation_id == conversation_id)
                    .unwrap_or(true)
            })
            .filter(|memory| {
                self.conversation_matches_workspace(
                    &memory.conversation_id,
                    req.workspace_id.as_deref(),
                )
            })
            .filter(|memory| {
                req.layer
                    .as_ref()
                    .map(|layer| memory.layer == *layer)
                    .unwrap_or(true)
            })
            .filter(|memory| {
                let Some(text) = text_filter else {
                    return true;
                };
                memory.content.to_lowercase().contains(text)
                    || memory
                        .entity
                        .as_deref()
                        .map(|value| value.to_lowercase().contains(text))
                        .unwrap_or(false)
                    || memory
                        .fact_key
                        .as_deref()
                        .map(|value| value.to_lowercase().contains(text))
                        .unwrap_or(false)
                    || memory
                        .fact_value
                        .as_deref()
                        .map(|value| value.to_lowercase().contains(text))
                        .unwrap_or(false)
            })
            .collect::<Vec<_>>();

        for memory in &filtered {
            *layer_counts
                .entry(memory_layer_label(&memory.layer).to_string())
                .or_insert(0) += 1;
        }

        filtered.sort_by(|a, b| {
            b.created_at
                .cmp(&a.created_at)
                .then_with(|| b.id.cmp(&a.id))
        });

        let total = filtered.len();
        let items = filtered
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(|memory| self.memory_browser_item(memory))
            .collect::<Vec<_>>();

        MemoryListResponse {
            generated_at: Utc::now(),
            total,
            limit,
            offset,
            has_more: offset.saturating_add(limit) < total,
            layer_counts,
            folders: self.memory_folder_summaries(),
            folder_suggestions: self.memory_folder_suggestions(),
            items,
        }
    }

    pub(crate) fn memory_browser_item(&self, memory: &MemoryRecord) -> MemoryBrowserItem {
        let (topic_key, topic_group) = Self::memory_topic_group(memory);
        let (review_status, review_label) = Self::memory_review_status(memory);
        let conflict_label = Self::memory_conflict_label(memory);
        MemoryBrowserItem {
            id: memory.id,
            workspace_id: self.workspace_for_conversation(&memory.conversation_id),
            conversation_id: memory.conversation_id.clone(),
            topic_key,
            topic_group,
            layer: memory.layer.clone(),
            content: memory.content.clone(),
            source: memory.source.clone(),
            trust: memory.trust,
            created_at: memory.created_at,
            version: memory.version,
            entity: memory.entity.clone(),
            fact_key: memory.fact_key.clone(),
            fact_value: memory.fact_value.clone(),
            memory_type: memory_quality_type(memory).to_string(),
            confidence: memory
                .confidence
                .or_else(|| memory_quality_number(&memory.content, "confidence")),
            freshness: memory
                .freshness
                .or_else(|| memory_quality_number(&memory.content, "freshness")),
            importance: memory
                .importance
                .or_else(|| memory_quality_number(&memory.content, "importance")),
            last_verified_at: memory.last_verified_at,
            supersedes: memory.supersedes.clone(),
            superseded_by: memory.superseded_by.clone(),
            canonical_semantic_id: self
                .canonical_semantic_for_memory(memory)
                .map(|node| node.semantic_id.clone()),
            semantic_node: memory.semantic_node.clone(),
            semantic_conflict: memory.semantic_conflict.clone(),
            meta_memory_state: memory.meta_memory_state.clone(),
            fact_timeline: self.fact_timeline_for_memory(memory),
            conflict_detected: conflict_label.is_some(),
            conflict_label,
            review_status,
            review_label,
        }
    }

    pub(crate) fn memory_topic_group(memory: &MemoryRecord) -> (String, String) {
        if let Some(label) = Self::extract_chapter_label(&memory.content) {
            let key = label
                .to_lowercase()
                .replace([' ', ':', '/', '\\'], "-")
                .chars()
                .filter(|ch| {
                    ch.is_ascii_alphanumeric()
                        || *ch == '-'
                        || ('\u{4e00}'..='\u{9fff}').contains(ch)
                })
                .collect::<String>();
            return (format!("chapter:{key}"), label);
        }

        if let (Some(entity), Some(fact_key)) = (&memory.entity, &memory.fact_key) {
            return (
                format!("fact:{}:{}", entity.to_lowercase(), fact_key.to_lowercase()),
                format!("{entity} / {fact_key}"),
            );
        }

        if let Some(node) = &memory.semantic_node {
            let kind = format!("{:?}", node.kind);
            return (
                format!("semantic:{}", kind.to_lowercase()),
                format!("语义 / {kind}"),
            );
        }

        (
            format!("layer:{}", memory_layer_label(&memory.layer)),
            format!("{} memory", memory_layer_label(&memory.layer)),
        )
    }

    pub(crate) fn extract_chapter_label(content: &str) -> Option<String> {
        let compact = content.trim();
        let lower = compact.to_lowercase();
        if let Some(pos) = lower.find("chapter") {
            let tail = compact[pos..].chars().take(28).collect::<String>();
            return Some(
                tail.trim_matches(|ch: char| ch.is_whitespace() || ch == ':' || ch == '：')
                    .to_string(),
            );
        }

        if compact.contains('第')
            && ['章', '节', '幕', '卷']
                .iter()
                .any(|marker| compact.contains(*marker))
        {
            if let Some(start) = compact.find('第') {
                let tail = compact[start..].chars().take(24).collect::<String>();
                let end = tail
                    .char_indices()
                    .find(|(_, ch)| matches!(ch, '。' | '，' | ',' | '.' | '\n'))
                    .map(|(idx, _)| idx)
                    .unwrap_or_else(|| tail.len());
                let label = tail[..end].trim().to_string();
                if label.chars().count() >= 3 {
                    return Some(label);
                }
            }
        }
        None
    }

    pub(crate) fn memory_conflict_label(memory: &MemoryRecord) -> Option<String> {
        if Self::memory_review_status(memory)
            .0
            .as_deref()
            .map(|status| status.starts_with("resolved"))
            .unwrap_or(false)
        {
            return None;
        }
        if memory
            .semantic_conflict
            .as_ref()
            .map(|conflict| !conflict.resolved)
            .unwrap_or(false)
        {
            return Some("同一事实存在多个候选值".to_string());
        }
        if memory
            .semantic_node
            .as_ref()
            .map(|node| node.status == SemanticStatus::Disputed)
            .unwrap_or(false)
        {
            return Some("语义节点处于冲突状态".to_string());
        }
        if memory
            .meta_memory_state
            .as_ref()
            .map(|state| state.conflict_pressure > 0.0)
            .unwrap_or(false)
        {
            return Some("这条记忆有冲突压力".to_string());
        }
        None
    }

    pub(crate) fn memory_review_status(memory: &MemoryRecord) -> (Option<String>, Option<String>) {
        let source = memory.source.to_lowercase();
        let content = memory.content.to_lowercase();
        let resolved =
            source.contains("review_status=resolved") || content.contains("resolved arbitration");
        let archived =
            source.contains("archive_status=archived") || source.contains("arbitration-archived");
        if resolved && archived {
            return (
                Some("resolved_archived".to_string()),
                Some("冲突已解决 / 已归档".to_string()),
            );
        }
        if resolved {
            return (Some("resolved".to_string()), Some("冲突已解决".to_string()));
        }
        if archived {
            return (Some("archived".to_string()), Some("已归档".to_string()));
        }
        (None, None)
    }

    pub(crate) fn fact_timeline_for_memory(
        &self,
        memory: &MemoryRecord,
    ) -> Vec<MemoryFactVersionItem> {
        let (Some(entity), Some(fact_key)) = (&memory.entity, &memory.fact_key) else {
            return Vec::new();
        };
        let mut timeline = self
            .memories
            .iter()
            .filter(|candidate| {
                candidate.conversation_id == memory.conversation_id
                    && candidate.entity.as_deref() == Some(entity)
                    && candidate.fact_key.as_deref() == Some(fact_key)
            })
            .map(|candidate| MemoryFactVersionItem {
                id: candidate.id,
                content: candidate.content.clone(),
                source: candidate.source.clone(),
                trust: candidate.trust,
                created_at: candidate.created_at,
                version: candidate.version,
                fact_value: candidate.fact_value.clone(),
                status: candidate
                    .semantic_node
                    .as_ref()
                    .map(|node| semantic_status_label(&node.status).to_string()),
            })
            .collect::<Vec<_>>();
        if timeline.len() <= 1 {
            if let Some(conflict) = &memory.semantic_conflict {
                for member in &conflict.members {
                    let member_id = member
                        .provenance_anchor_ids
                        .first()
                        .copied()
                        .unwrap_or(memory.id);
                    if timeline.iter().any(|item| item.id == member_id) {
                        continue;
                    }
                    timeline.push(MemoryFactVersionItem {
                        id: member_id,
                        content: format!(
                            "{} / {} = {}",
                            conflict.entity_id, conflict.attribute_key, member.attribute_value
                        ),
                        source: "semantic_conflict".to_string(),
                        trust: member.confidence,
                        created_at: memory.created_at,
                        version: member.version,
                        fact_value: Some(member.attribute_value.clone()),
                        status: Some("disputed".to_string()),
                    });
                }
            }
        }
        timeline.sort_by(|a, b| {
            b.version
                .cmp(&a.version)
                .then_with(|| b.created_at.cmp(&a.created_at))
                .then_with(|| b.id.cmp(&a.id))
        });
        timeline
    }

    pub(crate) fn update_memory(&mut self, req: &MemoryUpdateRequest) -> Option<MemoryBrowserItem> {
        let pos = self
            .memories
            .iter()
            .position(|memory| memory.id == req.memory_id)?;
        let conversation_id = self.memories[pos].conversation_id.clone();
        {
            let memory = &mut self.memories[pos];
            if let Some(content) = req
                .content
                .as_ref()
                .map(|value| value.trim())
                .filter(|value| !value.is_empty())
            {
                memory.content = content.to_string();
                memory.tokens = Self::tokenize(&memory.content);
            }
            if let Some(source) = req
                .source
                .as_ref()
                .map(|value| value.trim())
                .filter(|value| !value.is_empty())
            {
                memory.source = source.to_string();
            }
            if let Some(trust) = req.trust {
                memory.trust = trust.clamp(0.0, 1.0);
            }
            if let Some(entity) = &req.entity {
                memory.entity = clean_optional_string(entity);
            }
            if let Some(fact_key) = &req.fact_key {
                memory.fact_key = clean_optional_string(fact_key);
            }
            if let Some(fact_value) = &req.fact_value {
                memory.fact_value = clean_optional_string(fact_value);
            }
            if let Some(memory_type) = &req.memory_type {
                memory.memory_type = normalize_memory_type(memory_type).map(str::to_string);
            }
            if let Some(confidence) = req.confidence {
                memory.confidence = Some(confidence.clamp(0.0, 1.0));
            }
            if let Some(freshness) = req.freshness {
                memory.freshness = Some(freshness.clamp(0.0, 1.0));
            }
            if let Some(importance) = req.importance {
                memory.importance = Some(importance.clamp(0.0, 1.0));
            }
            if req.last_verified_at.is_some() {
                memory.last_verified_at = req.last_verified_at;
            }
            if let Some(supersedes) = &req.supersedes {
                memory.supersedes = clean_quality_refs(supersedes);
            }
            if let Some(superseded_by) = &req.superseded_by {
                memory.superseded_by = clean_quality_refs(superseded_by);
            }
            memory.version = memory.version.saturating_add(1);
            memory.embedding_status = None;
            memory.embedded_at = None;
        }

        self.rebuild_canonical_semantic_state();
        self.rebuild_meta_memory_state();
        self.invalidate_cache_for_conversation(Some(&conversation_id));
        self.memories
            .iter()
            .find(|memory| memory.id == req.memory_id)
            .map(|memory| self.memory_browser_item(memory))
    }

    pub(crate) fn apply_context_review_action(
        &mut self,
        req: &MemoryContextReviewActionRequest,
    ) -> Result<MemoryContextReviewActionResponse, String> {
        let action = req.action.trim().to_ascii_lowercase();
        let now = Utc::now();
        let reason = req
            .reason
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("context review action");
        let pos = self
            .memories
            .iter()
            .position(|memory| memory.id == req.memory_id)
            .ok_or_else(|| "memory id not found".to_string())?;
        let conversation_id = self.memories[pos].conversation_id.clone();
        let mut affected = vec![req.memory_id];

        match action.as_str() {
            "reverify" | "mark_stale_resolved" => {
                let memory = &mut self.memories[pos];
                memory.last_verified_at = Some(now);
                memory.freshness = Some(1.0);
                memory.confidence =
                    Some(memory.confidence.unwrap_or(memory.trust).max(0.8).min(1.0));
                memory.content =
                    set_quality_line(&memory.content, "last_verified_at", &now.to_rfc3339());
                memory.content = set_quality_line(&memory.content, "freshness", "1.0");
                memory.tokens = Self::tokenize(&memory.content);
                memory.version = memory.version.saturating_add(1);
                memory.embedding_status = None;
                memory.embedded_at = None;
            }
            "supersede" => {
                let replacement_ref = req
                    .replacement_memory_id
                    .map(|id| id.to_string())
                    .or_else(|| {
                        req.superseded_by
                            .as_deref()
                            .map(str::trim)
                            .filter(|value| !value.is_empty())
                            .map(str::to_string)
                    })
                    .ok_or_else(|| {
                        "supersede requires replacement_memory_id or superseded_by".to_string()
                    })?;
                {
                    let memory = &mut self.memories[pos];
                    if !memory
                        .superseded_by
                        .iter()
                        .any(|value| value == &replacement_ref)
                    {
                        memory.superseded_by.push(replacement_ref.clone());
                    }
                    memory.content =
                        set_quality_line(&memory.content, "superseded_by", &replacement_ref);
                    memory.tokens = Self::tokenize(&memory.content);
                    if let Some(conflict) = memory.semantic_conflict.as_mut() {
                        conflict.resolved = true;
                        conflict.winner_semantic_id = req
                            .replacement_memory_id
                            .map(|id| id.to_string())
                            .or_else(|| Some(replacement_ref.clone()));
                    }
                    if let Some(node) = memory.semantic_node.as_mut() {
                        node.status = SemanticStatus::Superseded;
                        node.conflict_set_id = None;
                    }
                    memory.version = memory.version.saturating_add(1);
                    memory.embedding_status = None;
                    memory.embedded_at = None;
                }
                if let Some(replacement_id) = req.replacement_memory_id {
                    if let Some(replacement) = self
                        .memories
                        .iter_mut()
                        .find(|memory| memory.id == replacement_id)
                    {
                        let source_ref = req.memory_id.to_string();
                        if !replacement
                            .supersedes
                            .iter()
                            .any(|value| value == &source_ref)
                        {
                            replacement.supersedes.push(source_ref);
                            replacement.version = replacement.version.saturating_add(1);
                            replacement.embedding_status = None;
                            replacement.embedded_at = None;
                        }
                        affected.push(replacement_id);
                    }
                }
            }
            "restore_superseded" => {
                let replacement_ref = req
                    .replacement_memory_id
                    .map(|id| id.to_string())
                    .or_else(|| {
                        req.superseded_by
                            .as_deref()
                            .map(str::trim)
                            .filter(|value| !value.is_empty())
                            .map(str::to_string)
                    })
                    .ok_or_else(|| {
                        "restore_superseded requires replacement_memory_id or superseded_by"
                            .to_string()
                    })?;
                {
                    let memory = &mut self.memories[pos];
                    if !memory
                        .superseded_by
                        .iter()
                        .any(|value| value == &replacement_ref)
                    {
                        return Err(
                            "restore_superseded replacement does not supersede this memory"
                                .to_string(),
                        );
                    }
                    memory
                        .superseded_by
                        .retain(|value| value != &replacement_ref);
                    memory.content = set_quality_line(
                        &memory.content,
                        "superseded_by",
                        &memory.superseded_by.join(","),
                    );
                    memory.tokens = Self::tokenize(&memory.content);
                    if memory.superseded_by.is_empty() {
                        if let Some(conflict) = memory.semantic_conflict.as_mut() {
                            conflict.resolved = false;
                            conflict.winner_semantic_id = None;
                        }
                        if let Some(node) = memory.semantic_node.as_mut() {
                            node.status = SemanticStatus::Active;
                            node.conflict_set_id = None;
                        }
                    }
                    memory.version = memory.version.saturating_add(1);
                    memory.embedding_status = None;
                    memory.embedded_at = None;
                }
                if let Some(replacement_id) = req.replacement_memory_id {
                    if let Some(replacement) = self
                        .memories
                        .iter_mut()
                        .find(|memory| memory.id == replacement_id)
                    {
                        let source_ref = req.memory_id.to_string();
                        replacement.supersedes.retain(|value| value != &source_ref);
                        replacement.version = replacement.version.saturating_add(1);
                        replacement.embedding_status = None;
                        replacement.embedded_at = None;
                        affected.push(replacement_id);
                    }
                }
            }
            "ignore" => {
                let memory = &mut self.memories[pos];
                memory.event_metadata = Some(EventMetadata {
                    origin_layer: memory.layer.clone(),
                    intent: "context_review".to_string(),
                    result: reason.to_string(),
                    memory_action: "ignore".to_string(),
                    confidence: 1.0,
                    value_weight: 0.0,
                });
                memory.freshness = Some(0.0);
                memory.importance = Some(0.0);
                memory.version = memory.version.saturating_add(1);
                memory.embedding_status = None;
                memory.embedded_at = None;
            }
            _ => {
                return Err(format!(
                    "unsupported context review action `{}`",
                    req.action.trim()
                ));
            }
        }

        affected.sort_unstable();
        affected.dedup();
        self.rebuild_canonical_semantic_state();
        self.rebuild_meta_memory_state();
        self.invalidate_cache_for_conversation(Some(&conversation_id));
        let item = self
            .memories
            .iter()
            .find(|memory| memory.id == req.memory_id)
            .map(|memory| self.memory_browser_item(memory));
        Ok(MemoryContextReviewActionResponse {
            action_format: "memory_review_action_v1".to_string(),
            applied: true,
            action,
            memory_id: req.memory_id,
            affected_memory_ids: affected,
            item,
            message: format!("context review action applied: {reason}"),
        })
    }

    pub(crate) fn delete_memory(&mut self, memory_id: u64) -> bool {
        let conversation_id = self
            .memories
            .iter()
            .find(|memory| memory.id == memory_id)
            .map(|memory| memory.conversation_id.clone());
        let before = self.memories.len();
        self.memories.retain(|memory| memory.id != memory_id);
        self.shared_memory_visibility.remove(&memory_id);
        if before == self.memories.len() {
            return false;
        }
        self.rebuild_canonical_semantic_state();
        self.rebuild_meta_memory_state();
        if let Some(conversation_id) = conversation_id.as_deref() {
            self.invalidate_cache_for_conversation(Some(conversation_id));
        } else {
            self.invalidate_all_cache();
        }
        true
    }

    pub(crate) fn assign_conversations_to_workspace(
        &mut self,
        workspace_id: &str,
        display_name: Option<String>,
        conversation_ids: &[String],
    ) -> Vec<String> {
        let workspace_id = workspace_id.trim();
        if workspace_id.is_empty() {
            return Vec::new();
        }
        let updated_at = Utc::now();
        let existing = self.workspace_registry.get(workspace_id).cloned();
        let tenant_id = existing
            .as_ref()
            .map(|item| item.tenant_id.clone())
            .unwrap_or_else(|| Self::DEFAULT_TENANT_ID.to_string());
        let isolation_policy = existing
            .as_ref()
            .map(|item| item.isolation_policy.clone())
            .unwrap_or_else(|| Self::DEFAULT_WORKSPACE_ISOLATION_POLICY.to_string());
        let compliance_profile = existing
            .as_ref()
            .map(|item| item.compliance_profile.clone())
            .unwrap_or_else(|| Self::DEFAULT_WORKSPACE_COMPLIANCE_PROFILE.to_string());
        let lifecycle_status = existing
            .as_ref()
            .map(|item| item.lifecycle_status.clone())
            .unwrap_or_else(|| "active".to_string());
        let display_name = display_name
            .and_then(|value| clean_optional_string(&value))
            .or_else(|| existing.and_then(|item| item.display_name))
            .or_else(|| Some(Self::display_name_from_conversation(workspace_id)));
        self.workspace_registry.insert(
            workspace_id.to_string(),
            WorkspaceRegistryState {
                workspace_id: workspace_id.to_string(),
                tenant_id,
                display_name,
                isolation_policy,
                compliance_profile,
                lifecycle_status,
                updated_at,
            },
        );

        let mut moved = HashSet::new();
        for conversation_id in conversation_ids {
            let conversation_id = conversation_id.trim();
            if conversation_id.is_empty() {
                continue;
            }
            self.conversation_workspaces
                .insert(conversation_id.to_string(), workspace_id.to_string());
            moved.insert(conversation_id.to_string());
        }
        let mut moved = moved.into_iter().collect::<Vec<_>>();
        moved.sort();
        moved
    }

    pub(crate) fn batch_update_memories(
        &mut self,
        req: &MemoryBatchUpdateRequest,
    ) -> MemoryBatchUpdateResponse {
        let ids = req.memory_ids.iter().copied().collect::<HashSet<_>>();
        let mut conversations = self
            .memories
            .iter()
            .filter(|memory| ids.contains(&memory.id))
            .map(|memory| memory.conversation_id.clone())
            .collect::<HashSet<_>>();
        let mut conversation_ids = conversations.iter().cloned().collect::<Vec<_>>();
        conversation_ids.sort();

        if req.delete.unwrap_or(false) {
            let before = self.memories.len();
            self.memories.retain(|memory| !ids.contains(&memory.id));
            for id in &ids {
                self.shared_memory_visibility.remove(id);
            }
            let deleted_count = before.saturating_sub(self.memories.len());
            if deleted_count > 0 {
                self.rebuild_canonical_semantic_state();
                self.rebuild_meta_memory_state();
            }
            for conversation_id in &conversation_ids {
                self.invalidate_cache_for_conversation(Some(conversation_id));
            }
            return MemoryBatchUpdateResponse {
                apply_allowed: true,
                apply_blocked_reason: None,
                protected_matches: Vec::new(),
                job_receipt_preview: None,
                lifecycle_contract: None,
                why_surfaced_contract: None,
                updated_count: 0,
                deleted_count,
                merged_conflict_count: 0,
                moved_conversation_count: 0,
                workspace_id: None,
                conversation_ids,
                items: Vec::new(),
            };
        }

        let mut updated_ids = Vec::new();
        for memory in &mut self.memories {
            if !ids.contains(&memory.id) {
                continue;
            }
            conversations.insert(memory.conversation_id.clone());
            let mut changed = false;
            if let Some(source) = req
                .source
                .as_ref()
                .map(|value| value.trim())
                .filter(|value| !value.is_empty())
            {
                memory.source = source.to_string();
                changed = true;
            }
            if let Some(trust) = req.trust {
                memory.trust = trust.clamp(0.0, 1.0);
                changed = true;
            }
            if let Some(entity) = &req.entity {
                memory.entity = clean_optional_string(entity);
                changed = true;
            }
            if let Some(fact_key) = &req.fact_key {
                memory.fact_key = clean_optional_string(fact_key);
                changed = true;
            }
            if let Some(fact_value) = &req.fact_value {
                memory.fact_value = clean_optional_string(fact_value);
                changed = true;
            }
            if changed {
                memory.version = memory.version.saturating_add(1);
                memory.embedding_status = None;
                memory.embedded_at = None;
                updated_ids.push(memory.id);
            }
        }

        conversation_ids = conversations.into_iter().collect::<Vec<_>>();
        conversation_ids.sort();
        let merged_conflict_count = if req.merge_conflicts.unwrap_or(false) {
            let strategy = req
                .conflict_winner_strategy
                .as_deref()
                .unwrap_or("highest_confidence");
            self.merge_selected_conflicts(&ids, strategy)
        } else {
            0
        };
        let moved_conversations = req
            .workspace_id
            .as_deref()
            .map(|workspace_id| {
                self.assign_conversations_to_workspace(
                    workspace_id,
                    req.workspace_display_name.clone(),
                    &conversation_ids,
                )
            })
            .unwrap_or_default();

        if !updated_ids.is_empty() || merged_conflict_count > 0 {
            self.rebuild_canonical_semantic_state();
            self.rebuild_meta_memory_state();
        }
        for conversation_id in &conversation_ids {
            self.invalidate_cache_for_conversation(Some(conversation_id));
        }

        let items = self
            .memories
            .iter()
            .filter(|memory| ids.contains(&memory.id))
            .map(|memory| self.memory_browser_item(memory))
            .collect::<Vec<_>>();
        MemoryBatchUpdateResponse {
            apply_allowed: true,
            apply_blocked_reason: None,
            protected_matches: Vec::new(),
            job_receipt_preview: None,
            lifecycle_contract: None,
            why_surfaced_contract: None,
            updated_count: updated_ids.len(),
            deleted_count: 0,
            merged_conflict_count,
            moved_conversation_count: moved_conversations.len(),
            workspace_id: req.workspace_id.clone(),
            conversation_ids,
            items,
        }
    }

    pub(crate) fn merge_selected_conflicts(&mut self, ids: &HashSet<u64>, strategy: &str) -> usize {
        let mut conflict_winners = HashMap::<String, String>::new();
        for memory in self
            .memories
            .iter()
            .filter(|memory| ids.contains(&memory.id))
        {
            let Some(conflict) = memory.semantic_conflict.as_ref() else {
                continue;
            };
            if conflict.resolved {
                continue;
            }
            let winner = match strategy {
                "current_card" => memory
                    .semantic_node
                    .as_ref()
                    .map(|node| node.semantic_id.clone())
                    .or_else(|| {
                        conflict
                            .members
                            .iter()
                            .max_by(|a, b| {
                                a.confidence
                                    .partial_cmp(&b.confidence)
                                    .unwrap_or(Ordering::Equal)
                            })
                            .map(|member| member.semantic_id.clone())
                    }),
                _ => conflict
                    .members
                    .iter()
                    .max_by(|a, b| {
                        a.confidence
                            .partial_cmp(&b.confidence)
                            .unwrap_or(Ordering::Equal)
                    })
                    .map(|member| member.semantic_id.clone()),
            };
            if let Some(winner) = winner {
                conflict_winners.insert(conflict.conflict_set_id.clone(), winner);
            }
        }

        let mut resolved = HashSet::new();
        for memory in &mut self.memories {
            let Some(conflict) = memory.semantic_conflict.as_mut() else {
                continue;
            };
            let Some(winner_id) = conflict_winners.get(&conflict.conflict_set_id) else {
                continue;
            };
            conflict.resolved = true;
            conflict.winner_semantic_id = Some(winner_id.clone());
            if let Some(node) = memory.semantic_node.as_mut() {
                if node.semantic_id == *winner_id {
                    node.status = SemanticStatus::Active;
                    node.conflict_set_id = None;
                } else if node.conflict_set_id.as_deref() == Some(&conflict.conflict_set_id) {
                    node.status = SemanticStatus::Superseded;
                }
            }
            resolved.insert(conflict.conflict_set_id.clone());
        }
        resolved.len()
    }

    pub(crate) fn memory_folder_suggestions(&self) -> Vec<MemoryFolderSuggestion> {
        #[derive(Default)]
        struct DraftSuggestion {
            display_name: String,
            reason: String,
            conversations: HashSet<String>,
            memory_count: usize,
            matched_terms: HashSet<String>,
            score: f32,
        }

        let mut suggestions: HashMap<String, DraftSuggestion> = HashMap::new();
        for memory in &self.memories {
            let current_workspace = self.workspace_for_conversation(&memory.conversation_id);
            if current_workspace != "default" {
                continue;
            }

            let mut candidates = Self::folder_candidates_for_memory(memory);
            if candidates.is_empty() {
                candidates.push((
                    Self::workspace_id_from_conversation(&memory.conversation_id),
                    Self::display_name_from_conversation(&memory.conversation_id),
                    "conversation naming pattern".to_string(),
                    vec![memory.conversation_id.clone()],
                    0.55,
                ));
            }

            for (workspace_id, display_name, reason, terms, score) in candidates {
                let entry = suggestions.entry(workspace_id).or_insert(DraftSuggestion {
                    display_name,
                    reason,
                    ..Default::default()
                });
                entry.conversations.insert(memory.conversation_id.clone());
                entry.memory_count += 1;
                entry.score = entry.score.max(score);
                for term in terms {
                    entry.matched_terms.insert(term);
                }
            }
        }

        let mut items = suggestions
            .into_iter()
            .filter(|(_, item)| item.memory_count >= 1)
            .map(|(workspace_id, item)| {
                let mut conversation_ids = item.conversations.into_iter().collect::<Vec<_>>();
                conversation_ids.sort();
                let mut matched_terms = item.matched_terms.into_iter().collect::<Vec<_>>();
                matched_terms.sort();
                MemoryFolderSuggestion {
                    workspace_id,
                    display_name: item.display_name,
                    reason: item.reason,
                    confidence: (item.score + (item.memory_count as f32).min(20.0) * 0.01)
                        .min(0.95),
                    conversation_ids,
                    memory_count: item.memory_count,
                    matched_terms,
                }
            })
            .collect::<Vec<_>>();
        items.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(Ordering::Equal)
                .then_with(|| b.memory_count.cmp(&a.memory_count))
                .then_with(|| a.workspace_id.cmp(&b.workspace_id))
        });
        items.truncate(12);
        items
    }

    pub(crate) fn folder_candidates_for_memory(
        memory: &MemoryRecord,
    ) -> Vec<(String, String, String, Vec<String>, f32)> {
        let mut matches = Self::ai_named_folder_candidates(memory);
        let mut text = format!(
            "{} {} {} {} {}",
            memory.conversation_id,
            memory.content,
            memory.entity.clone().unwrap_or_default(),
            memory.fact_key.clone().unwrap_or_default(),
            memory.fact_value.clone().unwrap_or_default()
        )
        .to_lowercase();
        text = text.replace(['_', '-', ':', '/', '.'], " ");

        let categories: [(&str, &str, &str, &[&str]); 6] = [
            (
                "novel-worldbuilding",
                "小说 / 世界设定",
                "story, character, plot, or worldbuilding terms",
                &[
                    "小说",
                    "角色",
                    "剧情",
                    "世界观",
                    "设定",
                    "人物",
                    "章节",
                    "lore",
                    "novel",
                    "character",
                ],
            ),
            (
                "app-design",
                "应用 / 产品设计",
                "product, UI, backend, API, or design terms",
                &[
                    "应用", "产品", "界面", "后端", "前端", "设计", "功能", "api", "ui", "backend",
                    "frontend", "app",
                ],
            ),
            (
                "software-project",
                "软件 / 工程项目",
                "code, deployment, bug, or system terms",
                &[
                    "代码",
                    "项目",
                    "部署",
                    "测试",
                    "错误",
                    "接口",
                    "数据库",
                    "rust",
                    "python",
                    "bug",
                    "deploy",
                ],
            ),
            (
                "user-preferences",
                "用户偏好",
                "preference or user profile terms",
                &[
                    "用户喜欢",
                    "偏好",
                    "不喜欢",
                    "风格",
                    "preference",
                    "profile",
                ],
            ),
            (
                "operations-runbook",
                "运维 / 故障手册",
                "operations, monitoring, incident, or runbook terms",
                &[
                    "运维", "监控", "告警", "回滚", "故障", "日志", "slo", "metrics", "runbook",
                ],
            ),
            (
                "research-notes",
                "研究 / 资料笔记",
                "research, docs, notes, or references",
                &[
                    "资料", "研究", "文档", "论文", "链接", "总结", "notes", "docs", "research",
                ],
            ),
        ];

        for (workspace_id, display_name, reason, terms) in categories {
            let matched = terms
                .iter()
                .filter(|term| text.contains(&term.to_lowercase()))
                .map(|term| (*term).to_string())
                .collect::<Vec<_>>();
            if !matched.is_empty() {
                matches.push((
                    workspace_id.to_string(),
                    display_name.to_string(),
                    reason.to_string(),
                    matched,
                    0.72,
                ));
            }
        }
        matches
    }

    pub(crate) fn ai_named_folder_candidates(
        memory: &MemoryRecord,
    ) -> Vec<(String, String, String, Vec<String>, f32)> {
        let source = format!(
            "{} {} {} {} {}",
            memory.conversation_id,
            memory.content,
            memory.entity.clone().unwrap_or_default(),
            memory.fact_key.clone().unwrap_or_default(),
            memory.fact_value.clone().unwrap_or_default()
        );
        let mut out = Vec::new();

        if let Some(title) = Self::extract_bracketed_title(&source) {
            let kind = if Self::looks_like_story_memory(&source) {
                "小说"
            } else if Self::looks_like_product_memory(&source) {
                "项目"
            } else {
                "主题"
            };
            out.push((
                format!(
                    "{}-{}",
                    kind_to_slug(kind),
                    Self::slugify_workspace_id(&title)
                ),
                format!("{kind} / {title}"),
                "AI naming: detected explicit title enclosed by book-name brackets".to_string(),
                vec![title],
                0.90,
            ));
        }

        if let Some(named) = Self::extract_named_subject(&source) {
            let label = if Self::looks_like_product_memory(&source) {
                format!("应用 / {named}")
            } else if Self::looks_like_story_memory(&source) {
                format!("小说 / {named}")
            } else {
                format!("主题 / {named}")
            };
            out.push((
                format!("topic-{}", Self::slugify_workspace_id(&named)),
                label,
                "AI naming: inferred recurring named subject from text".to_string(),
                vec![named],
                0.82,
            ));
        }

        if let Some(chapter_label) = Self::extract_chapter_label(&source) {
            out.push((
                format!("chapter-{}", Self::slugify_workspace_id(&chapter_label)),
                format!("章节 / {chapter_label}"),
                "AI naming: detected chapter or section marker".to_string(),
                vec![chapter_label],
                0.76,
            ));
        }

        if let (Some(entity), Some(fact_key)) = (&memory.entity, &memory.fact_key) {
            let subject = format!("{entity} {fact_key}");
            out.push((
                format!(
                    "semantic-{}-{}",
                    Self::slugify_workspace_id(entity),
                    Self::slugify_workspace_id(fact_key)
                ),
                format!("语义 / {entity} / {fact_key}"),
                "AI naming: grouped by semantic entity and attribute".to_string(),
                vec![subject],
                0.74,
            ));
        }

        out
    }

    pub(crate) fn extract_bracketed_title(text: &str) -> Option<String> {
        let start = text.find('《')?;
        let rest = &text[start + '《'.len_utf8()..];
        let end = rest.find('》')?;
        let title = rest[..end].trim();
        (title.chars().count() >= 2).then(|| title.to_string())
    }

    pub(crate) fn extract_named_subject(text: &str) -> Option<String> {
        let markers = [
            "项目名是",
            "项目叫",
            "应用名是",
            "应用叫",
            "小说名是",
            "小说叫",
            "作品名是",
            "作品叫",
            "世界观是",
            "主题是",
        ];
        for marker in markers {
            if let Some(pos) = text.find(marker) {
                let tail = &text[pos + marker.len()..];
                let value = tail
                    .trim()
                    .trim_start_matches(['：', ':', '"', '\'', '“'])
                    .chars()
                    .take_while(|ch| {
                        !matches!(ch, '。' | '，' | ',' | '.' | '\n' | '"' | '\'' | '”')
                    })
                    .collect::<String>()
                    .trim()
                    .to_string();
                if value.chars().count() >= 2 {
                    return Some(value);
                }
            }
        }
        None
    }

    pub(crate) fn looks_like_story_memory(text: &str) -> bool {
        [
            "小说",
            "角色",
            "剧情",
            "世界观",
            "章节",
            "人物",
            "lore",
            "character",
        ]
        .iter()
        .any(|term| text.to_lowercase().contains(&term.to_lowercase()))
    }

    pub(crate) fn looks_like_product_memory(text: &str) -> bool {
        [
            "应用", "产品", "界面", "功能", "后端", "前端", "api", "ui", "app",
        ]
        .iter()
        .any(|term| text.to_lowercase().contains(&term.to_lowercase()))
    }

    pub(crate) fn slugify_workspace_id(value: &str) -> String {
        let mut slug = value
            .to_lowercase()
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || ('\u{4e00}'..='\u{9fff}').contains(&ch) {
                    ch
                } else {
                    '-'
                }
            })
            .collect::<String>();
        while slug.contains("--") {
            slug = slug.replace("--", "-");
        }
        let slug = slug.trim_matches('-').to_string();
        if slug.is_empty() {
            "untitled".to_string()
        } else {
            slug.chars().take(48).collect()
        }
    }

    pub(crate) fn workspace_id_from_conversation(conversation_id: &str) -> String {
        let parts = conversation_id
            .split(|ch: char| !ch.is_ascii_alphanumeric())
            .filter(|part| !part.is_empty())
            .take(2)
            .collect::<Vec<_>>();
        if parts.is_empty() {
            "default-drafts".to_string()
        } else {
            parts.join("-").to_lowercase()
        }
    }

    pub(crate) fn display_name_from_conversation(conversation_id: &str) -> String {
        let id = Self::workspace_id_from_conversation(conversation_id);
        id.split('-')
            .map(|part| {
                let mut chars = part.chars();
                match chars.next() {
                    Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str()),
                    None => String::new(),
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub(crate) fn memory_folder_summaries(&self) -> Vec<MemoryFolderSummary> {
        let mut conversations: HashMap<String, HashMap<String, (usize, Option<DateTime<Utc>>)>> =
            HashMap::new();

        for memory in &self.memories {
            let workspace_id = self.workspace_for_conversation(&memory.conversation_id);
            let entry = conversations
                .entry(workspace_id)
                .or_default()
                .entry(memory.conversation_id.clone())
                .or_insert((0, None));
            entry.0 += 1;
            if entry
                .1
                .map(|current| memory.created_at > current)
                .unwrap_or(true)
            {
                entry.1 = Some(memory.created_at);
            }
        }

        for workspace_id in self.workspace_registry.keys() {
            conversations.entry(workspace_id.clone()).or_default();
        }

        let mut folders = conversations
            .into_iter()
            .map(|(workspace_id, conversation_map)| {
                let mut conversation_items = conversation_map
                    .into_iter()
                    .map(|(conversation_id, (memory_count, last_memory_at))| {
                        WorkspaceConversationSummary {
                            conversation_id,
                            memory_count,
                            last_memory_at,
                        }
                    })
                    .collect::<Vec<_>>();
                conversation_items.sort_by(|a, b| {
                    b.last_memory_at
                        .cmp(&a.last_memory_at)
                        .then_with(|| a.conversation_id.cmp(&b.conversation_id))
                });
                let registry = self.workspace_registry_state_for(&workspace_id);
                MemoryFolderSummary {
                    workspace_id: workspace_id.clone(),
                    display_name: registry.and_then(|item| item.display_name.clone()),
                    tenant_id: registry
                        .map(|item| item.tenant_id.clone())
                        .unwrap_or_else(|| Self::DEFAULT_TENANT_ID.to_string()),
                    conversation_count: conversation_items.len(),
                    memory_count: conversation_items
                        .iter()
                        .map(|conversation| conversation.memory_count)
                        .sum(),
                    conversations: conversation_items,
                }
            })
            .collect::<Vec<_>>();
        folders.sort_by(|a, b| {
            b.memory_count
                .cmp(&a.memory_count)
                .then_with(|| a.workspace_id.cmp(&b.workspace_id))
        });
        folders
    }

    pub(crate) fn get_cached_ids(&self, key: &str) -> Option<Vec<u64>> {
        self.cache
            .get(key)
            .filter(|c| c.expires_at > Utc::now())
            .map(|c| c.ids.clone())
    }

    pub(crate) fn put_cached_ids(&mut self, key: String, ids: Vec<u64>) {
        self.cache.insert(
            key,
            CacheEntry {
                query: String::new(),
                ids,
                expires_at: Utc::now() + Duration::minutes(5),
            },
        );
    }

    pub(crate) fn materialize_items(&self, ids: &[u64], query: &str) -> Vec<RetrievedItem> {
        self.materialize_items_internal(ids, query, None)
    }

    fn materialize_items_internal(
        &self,
        ids: &[u64],
        query: &str,
        score_overrides: Option<&HashMap<u64, f32>>,
    ) -> Vec<RetrievedItem> {
        let q_tokens = Self::tokenize(query);
        let selected_ids = ids.iter().copied().collect::<HashSet<_>>();
        let selected = self
            .memories
            .iter()
            .filter(|memory| selected_ids.contains(&memory.id))
            .map(|memory| (memory.id, memory))
            .collect::<HashMap<_, _>>();
        ids.iter()
            .filter_map(|id| selected.get(id).copied())
            .map(|m| RetrievedItem {
                canonical_semantic_id: self
                    .canonical_semantic_for_memory(m)
                    .map(|node| node.semantic_id.clone()),
                id: m.id,
                layer: m.layer.clone(),
                memory_type: memory_quality_type(m).to_string(),
                content: m.content.clone(),
                score: score_overrides
                    .and_then(|scores| scores.get(&m.id).copied())
                    .unwrap_or_else(|| {
                        0.5 * Self::cosine_tf(&q_tokens, &m.tokens)
                            + 0.3 * Self::jaccard(&q_tokens, &m.tokens)
                            + 0.2 * Self::time_decay(m.created_at)
                    })
                    + memory_quality_score_boost(m, query),
                source: m.source.clone(),
                entity: m.entity.clone(),
                created_at: m.created_at,
                confidence: m
                    .confidence
                    .or_else(|| memory_quality_number(&m.content, "confidence")),
                freshness: m
                    .freshness
                    .or_else(|| memory_quality_number(&m.content, "freshness")),
                importance: m
                    .importance
                    .or_else(|| memory_quality_number(&m.content, "importance")),
                last_verified_at: m.last_verified_at,
                supersedes: m.supersedes.clone(),
                superseded_by: m.superseded_by.clone(),
                version: m.version,
                event_metadata: m.event_metadata.clone(),
                procedure_schema: m.procedure_schema.clone(),
                procedure_execution_state: m.procedure_execution_state.clone(),
                procedure_governance: m.procedure_governance.clone(),
                episodic_metadata: m.episodic_metadata.clone(),
                meta_memory_state: m.meta_memory_state.clone(),
                activation_metadata: m.activation_metadata.clone(),
                semantic_node: m.semantic_node.clone(),
                semantic_conflict: m.semantic_conflict.clone(),
            })
            .collect()
    }

    pub(crate) fn materialize_items_with_scores(
        &self,
        ids: &[u64],
        query: &str,
        score_overrides: &HashMap<u64, f32>,
    ) -> Vec<RetrievedItem> {
        self.materialize_items_internal(ids, query, Some(score_overrides))
    }

    pub(crate) fn query_activation_memories(
        &self,
        conversation_id: &str,
        include_cleared: bool,
    ) -> ActivationQueryResponse {
        let mut memories = self
            .memories
            .iter()
            .filter(|memory| {
                memory.conversation_id == conversation_id
                    && memory.layer == MemoryLayer::Activation
                    && (include_cleared
                        || memory
                            .activation_metadata
                            .as_ref()
                            .map(Self::activation_metadata_is_live)
                            .unwrap_or(false))
            })
            .collect::<Vec<_>>();
        memories.sort_by(|a, b| {
            let a_rank = a
                .activation_metadata
                .as_ref()
                .map(|metadata| Self::activation_status_rank(&metadata.status))
                .unwrap_or_default();
            let b_rank = b
                .activation_metadata
                .as_ref()
                .map(|metadata| Self::activation_status_rank(&metadata.status))
                .unwrap_or_default();
            b_rank
                .cmp(&a_rank)
                .then_with(|| b.created_at.cmp(&a.created_at))
        });
        let ids = memories.iter().map(|memory| memory.id).collect::<Vec<_>>();
        ActivationQueryResponse {
            conversation_id: conversation_id.to_string(),
            items: self.materialize_items(&ids, "activation working set"),
        }
    }

    pub(crate) fn promote_activation_memories(
        &mut self,
        conversation_id: &str,
        memory_ids: Option<Vec<u64>>,
        query: Option<&str>,
        top_k: Option<usize>,
        pin: bool,
        source: Option<&str>,
        expires_in_seconds: Option<i64>,
        reserved_ids: Option<&[u64]>,
    ) -> ActivationPromoteResponse {
        let selected_ids = if let Some(memory_ids) = memory_ids {
            memory_ids
        } else if let Some(query) = query {
            self.retrieve_ids(
                Some(conversation_id),
                query,
                Self::normalized_top_k(top_k, 5),
            )
        } else {
            Vec::new()
        };

        let selected = self
            .memories
            .iter()
            .filter(|memory| memory.conversation_id == conversation_id)
            .filter(|memory| selected_ids.iter().any(|id| *id == memory.id))
            .cloned()
            .collect::<Vec<_>>();

        let expires_at = if pin {
            None
        } else {
            expires_in_seconds
                .filter(|seconds| *seconds > 0)
                .map(|seconds| Utc::now() + Duration::seconds(seconds))
        };

        let created = selected
            .into_iter()
            .enumerate()
            .map(|(idx, source_memory)| {
                let id = reserved_ids
                    .and_then(|ids| ids.get(idx).copied())
                    .unwrap_or_else(|| ID_GEN.fetch_add(1, AtomicOrdering::Relaxed));
                MemoryRecord {
                    id,
                    conversation_id: conversation_id.to_string(),
                    layer: MemoryLayer::Activation,
                    content: source_memory.content.clone(),
                    tokens: source_memory.tokens.clone(),
                    source: format!(
                        "activation:{}",
                        source.unwrap_or(source_memory.source.as_str())
                    ),
                    trust: source_memory.trust,
                    created_at: Utc::now(),
                    anchor_ids: vec![source_memory.id],
                    entity: source_memory.entity.clone(),
                    fact_key: source_memory.fact_key.clone(),
                    fact_value: source_memory.fact_value.clone(),
                    memory_type: None,
                    confidence: None,
                    freshness: None,
                    importance: None,
                    last_verified_at: None,
                    supersedes: Vec::new(),
                    superseded_by: Vec::new(),
                    version: 1,
                    event_metadata: source_memory.event_metadata.clone(),
                    procedure_schema: source_memory.procedure_schema.clone(),
                    procedure_execution_state: source_memory.procedure_execution_state.clone(),
                    procedure_governance: source_memory.procedure_governance.clone(),
                    episodic_metadata: source_memory.episodic_metadata.clone(),
                    meta_memory_state: source_memory.meta_memory_state.clone(),
                    activation_metadata: Some(ActivationMetadata {
                        status: if pin {
                            "pinned".to_string()
                        } else {
                            "active".to_string()
                        },
                        source: source.unwrap_or("manual").to_string(),
                        target_memory_ids: vec![source_memory.id],
                        expires_at,
                    }),
                    retrieval_hits: 0,
                    last_retrieved_at: None,
                    embedding_provider: None,
                    embedding_model: None,
                    embedding_status: None,
                    embedded_at: None,
                    semantic_node: source_memory.semantic_node.clone(),
                    semantic_conflict: source_memory.semantic_conflict.clone(),
                }
            })
            .collect::<Vec<_>>();

        let created_ids = created.iter().map(|memory| memory.id).collect::<Vec<_>>();
        self.memories.extend(created);
        self.invalidate_cache_for_conversation(Some(conversation_id));
        self.rebuild_canonical_semantic_state();
        self.rebuild_meta_memory_state();

        ActivationPromoteResponse {
            promoted_count: created_ids.len(),
            created_memory_ids: created_ids.clone(),
            items: self.materialize_items(&created_ids, "activation promotion"),
        }
    }

    pub(crate) fn update_activation_status(
        &mut self,
        conversation_id: &str,
        activation_memory_ids: &[u64],
        status: &str,
    ) -> ActivationUpdateResponse {
        let mut updated = Vec::new();
        for memory in self.memories.iter_mut().filter(|memory| {
            memory.conversation_id == conversation_id
                && memory.layer == MemoryLayer::Activation
                && activation_memory_ids.iter().any(|id| *id == memory.id)
        }) {
            if let Some(metadata) = memory.activation_metadata.as_mut() {
                metadata.status = status.to_string();
                if status == "pinned" {
                    metadata.expires_at = None;
                }
                if status == "cleared" {
                    metadata.expires_at = None;
                }
                updated.push(memory.id);
            }
        }
        self.invalidate_cache_for_conversation(Some(conversation_id));
        self.rebuild_meta_memory_state();

        ActivationUpdateResponse {
            updated_count: updated.len(),
            activation_memory_ids: updated,
            status: status.to_string(),
        }
    }
}
