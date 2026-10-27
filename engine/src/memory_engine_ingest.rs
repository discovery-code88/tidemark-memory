use super::*;

impl MemoryEngine {
    pub(crate) fn jaccard(a: &[String], b: &[String]) -> f32 {
        let sa: HashSet<_> = a.iter().collect();
        let sb: HashSet<_> = b.iter().collect();
        let inter = sa.intersection(&sb).count() as f32;
        let union = sa.union(&sb).count() as f32;
        if union == 0.0 {
            0.0
        } else {
            inter / union
        }
    }

    /// The largest Jaccard similarity between `a` and any of `others`, equal to
    /// folding `jaccard` with `max` but without building two sets per pair:
    /// `a`'s set is built once, one scratch set is reused, and token lists too
    /// short to beat the best so far are skipped (J(a, b) <= |b| / |a|).
    pub(crate) fn max_jaccard<'a>(a: &[String], others: impl Iterator<Item = &'a [String]>) -> f32 {
        type Fnv = std::hash::BuildHasherDefault<FnvHasher>;
        let sa: HashSet<&str, Fnv> = a.iter().map(String::as_str).collect();
        let mut seen: HashSet<&'a str, Fnv> = HashSet::default();
        let mut best = 0.0_f32;
        for b in others {
            if !sa.is_empty() && (b.len() as f32) < best * sa.len() as f32 {
                continue;
            }
            seen.clear();
            let mut inter = 0_usize;
            for t in b {
                if seen.insert(t.as_str()) && sa.contains(t.as_str()) {
                    inter += 1;
                }
            }
            let union = sa.len() + seen.len() - inter;
            if union > 0 {
                best = best.max(inter as f32 / union as f32);
            }
            if best >= 1.0 {
                break;
            }
        }
        best
    }

    pub(crate) fn cosine_tf(a: &[String], b: &[String]) -> f32 {
        let mut ma: HashMap<&String, f32> = HashMap::new();
        let mut mb: HashMap<&String, f32> = HashMap::new();
        for t in a {
            *ma.entry(t).or_insert(0.0) += 1.0;
        }
        for t in b {
            *mb.entry(t).or_insert(0.0) += 1.0;
        }
        let mut dot = 0.0;
        for (k, va) in &ma {
            if let Some(vb) = mb.get(k) {
                dot += va * vb;
            }
        }
        let na = ma.values().map(|v| v * v).sum::<f32>().sqrt();
        let nb = mb.values().map(|v| v * v).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    }

    pub(crate) fn char_bigram_overlap(a: &str, b: &str) -> f32 {
        pub(crate) fn bigrams(input: &str) -> HashSet<String> {
            let chars = input.chars().collect::<Vec<_>>();
            if chars.len() < 2 {
                return chars.into_iter().map(|c| c.to_string()).collect();
            }
            chars
                .windows(2)
                .map(|window| window.iter().collect::<String>())
                .collect()
        }

        let a = a.trim();
        let b = b.trim();
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        if a.contains(b) || b.contains(a) {
            return 1.0;
        }

        let ga = bigrams(a);
        let gb = bigrams(b);
        let inter = ga.intersection(&gb).count() as f32;
        let union = ga.union(&gb).count() as f32;
        if union == 0.0 {
            0.0
        } else {
            inter / union
        }
    }

    pub(crate) fn time_decay(created_at: DateTime<Utc>) -> f32 {
        let age_hours = (Utc::now() - created_at).num_hours().max(0) as f32;
        let half_life = 24.0 * 7.0;
        (0.5_f32).powf(age_hours / half_life)
    }

    pub(crate) fn write_score(&self, req: &IngestRequest, tokens: &[String]) -> f32 {
        let novelty = 1.0
            - Self::max_jaccard(
                tokens,
                self.memories
                    .iter()
                    .filter(|m| m.conversation_id == req.conversation_id && m.layer == req.layer)
                    .map(|m| m.tokens.as_slice()),
            );

        let reusable_markers = [
            "规则",
            "流程",
            "步骤",
            "偏好",
            "设定",
            "事实",
            "总是",
            "不要",
            "Character",
            "Location",
            "World Rule",
            "Timeline Event",
            "Relationship",
            "Evidence",
            "Summary",
            "Novel",
            "Archive type",
            "Source file",
            "Index terms",
            "memory_type: novel_chapter_summary",
            "Chapter summary",
            "Key events",
            "Source chunks",
            "章节",
            "关键事件",
            "learning_memory_format",
            "learning",
            "learn",
            "concept",
            "concept map",
            "problem",
            "attempt",
            "stuck",
            "reflection",
            "review",
            "学习",
            "概念",
            "概念地图",
            "问题",
            "尝试",
            "卡住",
            "卡点",
            "复盘",
            "能力边界",
            "局部概念",
        ];
        let reusable_hits = reusable_markers
            .iter()
            .filter(|k| req.content.contains(*k))
            .count() as f32;
        let reusable = (reusable_hits / 3.0).min(1.0);

        let pref_overlap = tokens
            .iter()
            .filter(|t| self.user_preferences.contains(*t))
            .count() as f32;
        let pref_score = (pref_overlap / 2.0).min(1.0);

        let structured_fact_bonus =
            if req.entity.is_some() && req.fact_key.is_some() && req.fact_value.is_some() {
                0.12
            } else {
                0.0
            };
        let provenance_bonus =
            if req.content.contains("Evidence:") && req.content.contains("Source:") {
                0.08
            } else {
                0.0
            };
        let archive_provenance_bonus =
            if req.content.contains("Archive type:") && req.content.contains("Source file:") {
                0.08
            } else {
                0.0
            };

        (0.40 * novelty
            + 0.30 * reusable
            + 0.10 * pref_score
            + structured_fact_bonus
            + provenance_bonus
            + archive_provenance_bonus)
            .min(1.0)
    }

    pub(crate) fn ingest(&mut self, req: IngestRequest) -> IngestResponse {
        self.ingest_with_reserved_id(req, None)
    }

    pub(crate) fn ingest_with_reserved_id(
        &mut self,
        req: IngestRequest,
        reserved_id: Option<u64>,
    ) -> IngestResponse {
        self.ingest_with_reserved_id_and_quality(req, reserved_id, MemoryQualityPatch::default())
    }

    pub(crate) fn ingest_with_reserved_id_and_quality(
        &mut self,
        req: IngestRequest,
        reserved_id: Option<u64>,
        quality: MemoryQualityPatch,
    ) -> IngestResponse {
        let conversation_id = req.conversation_id.clone();
        let compact_len = req.content.trim().chars().count();
        if compact_len < 6 {
            return IngestResponse {
                accepted: false,
                reason: "blocked_by_write_gate_low_signal".to_string(),
                memory_id: None,
                write_score: 0.0,
                conflict_detected: false,
                active_version: None,
                error_code: None,
                persistence_error: None,
                rolled_back: false,
            };
        }

        let tokens = Self::tokenize(&req.content);
        let score = self.write_score(&req, &tokens);
        let threshold = self.write_gate_thresholds.for_layer(&req.layer);
        let write_strategy = MetaMemoryController::write_strategy(compact_len, score, threshold);

        if write_strategy.primary_action == MetaMemoryAction::RejectLowSignal {
            return IngestResponse {
                accepted: false,
                reason: "blocked_by_write_gate".to_string(),
                memory_id: None,
                write_score: score,
                conflict_detected: false,
                active_version: None,
                error_code: None,
                persistence_error: None,
                rolled_back: false,
            };
        }

        let mut conflict = false;
        let mut version = 1_u32;
        if req.layer == MemoryLayer::Fact {
            if let (Some(entity), Some(key), Some(value)) =
                (&req.entity, &req.fact_key, &req.fact_value)
            {
                if let Some(last) = self
                    .memories
                    .iter()
                    .filter(|m| {
                        m.conversation_id == conversation_id
                            && m.layer == MemoryLayer::Fact
                            && m.entity.as_deref() == Some(entity.as_str())
                            && m.fact_key.as_deref() == Some(key.as_str())
                    })
                    .max_by(|a, b| {
                        a.version
                            .cmp(&b.version)
                            .then_with(|| a.created_at.cmp(&b.created_at))
                    })
                {
                    version = last.version + 1;
                    let conflict_strategy = MetaMemoryController::fact_conflict_strategy(
                        last.fact_value.as_deref(),
                        value,
                    );
                    conflict = conflict_strategy.primary_action == MetaMemoryAction::DetectConflict;
                }
            }
        }

        let cache_conversation_id = conversation_id.clone();
        let id = reserved_id.unwrap_or_else(|| ID_GEN.fetch_add(1, AtomicOrdering::Relaxed));
        if reserved_id.is_some() {
            advance_memory_id_generator_floor(id.saturating_add(1));
        }
        let created_at = Utc::now();
        let procedure_schema = extract_procedure_schema(&req);
        let is_procedure = req.layer == MemoryLayer::Procedure;
        let primary_layer_is_fact = req.layer == MemoryLayer::Fact;
        let req_content_clone = req.content.clone();
        let req_entity_clone = req.entity.clone();
        let req_fact_key_clone = req.fact_key.clone();
        let req_fact_value_clone = req.fact_value.clone();
        let req_trust_val = req.trust.unwrap_or(0.8).clamp(0.0, 1.0);
        let memory_type = clean_memory_type(quality.memory_type.as_ref());
        let confidence = clean_quality_score(quality.confidence);
        let freshness = clean_quality_score(quality.freshness);
        let importance = clean_quality_score(quality.importance);
        let last_verified_at = quality.last_verified_at;
        let supersedes = clean_quality_refs(&quality.supersedes);
        let superseded_by = clean_quality_refs(&quality.superseded_by);
        self.memories.push(MemoryRecord {
            id,
            conversation_id,
            layer: req.layer,
            content: req.content,
            tokens,
            source: req.source.unwrap_or_else(|| "user".to_string()),
            trust: req.trust.unwrap_or(0.8).clamp(0.0, 1.0),
            created_at,
            anchor_ids: Vec::new(),
            entity: req.entity,
            fact_key: req.fact_key,
            fact_value: req.fact_value,
            memory_type,
            confidence,
            freshness,
            importance,
            last_verified_at,
            supersedes,
            superseded_by,
            version,
            event_metadata: None,
            procedure_schema,
            procedure_execution_state: None,
            procedure_governance: if is_procedure {
                Some(ProcedureGovernance {
                    preferred: true,
                    auto_deprecated: false,
                    last_health_score: None,
                })
            } else {
                None
            },
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

        // ── Semantic consolidation (V1): after accepted Fact ingest ──
        if primary_layer_is_fact {
            if let (Some(ref entity), Some(ref fk), Some(ref fv)) = (
                &req_entity_clone,
                &req_fact_key_clone,
                &req_fact_value_clone,
            ) {
                let semantic_scope = infer_semantic_scope(entity, fk, &req_content_clone);
                // Find existing semantic node for same entity + attribute
                let existing_semantic = self
                    .memories
                    .iter()
                    .filter(|m| {
                        m.id != id
                            && m.layer == MemoryLayer::Fact
                            && m.semantic_node.is_some()
                            && m.semantic_node
                                .as_ref()
                                .map(|sn| {
                                    sn.entity_id == *entity
                                        && sn.attribute_key == *fk
                                        && sn.scope == semantic_scope
                                        && sn.status != SemanticStatus::Superseded
                                })
                                .unwrap_or(false)
                    })
                    .max_by_key(|m| m.version)
                    .and_then(|m| m.semantic_node.clone());

                let trust_val = req_trust_val;
                if let Some(existing) = existing_semantic {
                    if existing.attribute_value == *fv {
                        // Confirming evidence → merge
                        let merged = merge_semantic_confirmation(&existing, id, trust_val);
                        if let Some(mem) = self.memories.iter_mut().find(|m| m.id == id) {
                            mem.semantic_node = Some(merged);
                        }
                    } else {
                        // Contradiction → build conflict set
                        let (mut conflict_set, mut disputed_existing, mut disputed_new) =
                            build_semantic_conflict_set(entity, fk, &existing, fv, trust_val, id);
                        if let Some(winner_id) = try_resolve_conflict(&mut conflict_set) {
                            apply_conflict_resolution(
                                &winner_id,
                                &mut disputed_existing,
                                &mut disputed_new,
                            );
                        }
                        // Mark the new memory
                        if let Some(mem) = self.memories.iter_mut().find(|m| m.id == id) {
                            mem.semantic_node = Some(disputed_new);
                            mem.semantic_conflict = Some(conflict_set.clone());
                        }
                        // Mark existing memory as disputed too
                        let existing_sem_id = existing.semantic_id.clone();
                        if let Some(old_mem) = self.memories.iter_mut().find(|m| {
                            m.semantic_node
                                .as_ref()
                                .map(|sn| sn.semantic_id == existing_sem_id)
                                .unwrap_or(false)
                        }) {
                            old_mem.semantic_node = Some(disputed_existing);
                            old_mem.semantic_conflict = Some(conflict_set);
                        }
                    }
                } else {
                    // First time → create semantic node
                    let new_node = maybe_build_semantic_node(
                        entity,
                        "entity",
                        fk,
                        fv,
                        trust_val,
                        id,
                        semantic_scope,
                    );
                    if let Some(mem) = self.memories.iter_mut().find(|m| m.id == id) {
                        mem.semantic_node = Some(new_node);
                    }
                }
            }
        }
        let primary = self
            .memories
            .iter()
            .find(|m| m.id == id)
            .cloned()
            .expect("ingested memory should exist");
        let event_req = IngestRequest {
            conversation_id: primary.conversation_id.clone(),
            layer: primary.layer.clone(),
            content: primary.content.clone(),
            source: Some(primary.source.clone()),
            trust: Some(primary.trust),
            entity: primary.entity.clone(),
            fact_key: primary.fact_key.clone(),
            fact_value: primary.fact_value.clone(),
        };
        // The episode, self-model and autobiographical builders only read
        // memories of this conversation; give them just those (copying every
        // memory on each write made ingestion quadratic in the total stored).
        let scoped = self
            .memories
            .iter()
            .filter(|m| m.conversation_id == primary.conversation_id)
            .collect::<Vec<&MemoryRecord>>();
        let mut generated = vec![build_event_record(
            ID_GEN.fetch_add(1, AtomicOrdering::Relaxed),
            &event_req,
            primary.tokens.clone(),
            created_at,
        )];
        if let Some(episode) = maybe_build_episode(
            &primary.conversation_id,
            &scoped,
            ID_GEN.fetch_add(1, AtomicOrdering::Relaxed),
            created_at,
        ) {
            generated.push(episode);
        }
        if let Some(self_model) = maybe_build_self_model(
            &event_req,
            &scoped,
            ID_GEN.fetch_add(1, AtomicOrdering::Relaxed),
            created_at,
        ) {
            generated.push(self_model);
        }
        if let Some(autobiographical) = maybe_build_autobiographical(
            &event_req,
            &scoped
                .iter()
                .copied()
                .chain(generated.iter())
                .collect::<Vec<&MemoryRecord>>(),
            ID_GEN.fetch_add(1, AtomicOrdering::Relaxed),
            created_at,
        ) {
            generated.push(autobiographical);
        }
        self.memories.extend(generated);
        self.maybe_auto_consolidate(&primary.conversation_id);
        self.invalidate_cache_for_conversation(Some(cache_conversation_id.as_str()));
        let canonical_strategy = MetaMemoryController::canonical_refresh_strategy();
        if canonical_strategy.refresh_canonical_semantic {
            self.rebuild_canonical_semantic_state();
        }
        // A write only changes memories of its own conversation, and each
        // memory's meta-memory state depends on that memory alone.
        self.rebuild_meta_memory_state_for(&primary.conversation_id);

        IngestResponse {
            accepted: true,
            reason: "accepted".to_string(),
            memory_id: Some(id),
            write_score: score,
            conflict_detected: conflict,
            active_version: Some(version),
            error_code: None,
            persistence_error: None,
            rolled_back: false,
        }
    }
}

/// FNV-1a: a fast hasher for short token strings (no HashDoS concern here).
#[derive(Default)]
pub(crate) struct FnvHasher(u64);

impl std::hash::Hasher for FnvHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        let mut h = if self.0 == 0 { 0xcbf2_9ce4_8422_2325 } else { self.0 };
        for b in bytes {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        self.0 = h;
    }
}

#[cfg(test)]
mod max_jaccard_tests {
    use super::*;

    #[test]
    fn max_jaccard_equals_folded_jaccard() {
        let words = ["kite", "beach", "red", "sun", "kite", "dog", "sea", "sand", "red", "wind"];
        let lists = (0..60)
            .map(|i: usize| (0..(i * 7) % 13).map(|k| words[(i * 3 + k * k) % words.len()].to_string()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        for a in &lists {
            let folded = lists.iter().map(|b| MemoryEngine::jaccard(a, b)).fold(0.0_f32, f32::max);
            let fast = MemoryEngine::max_jaccard(a, lists.iter().map(|b| b.as_slice()));
            assert!((folded - fast).abs() < 1e-6, "{a:?}: {folded} vs {fast}");
        }
        assert_eq!(MemoryEngine::max_jaccard(&[], std::iter::empty()), 0.0);
    }
}
