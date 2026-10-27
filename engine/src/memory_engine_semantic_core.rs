use super::*;

impl MemoryEngine {
    pub(crate) fn canonical_semantic_key(entity_id: &str, attribute_key: &str) -> String {
        format!("{}::{}", entity_id, attribute_key)
    }

    pub(crate) fn rebuild_canonical_semantic_state(&mut self) {
        self.canonical_semantic_state.clear();
        for memory in &self.memories {
            let Some(node) = memory.semantic_node.as_ref() else {
                continue;
            };
            let key = Self::canonical_semantic_key(&node.entity_id, &node.attribute_key);
            let replace = match self.canonical_semantic_state.get(&key) {
                Some(existing) => {
                    let existing_rank = semantic_status_score_adjustment(Some(existing));
                    let candidate_rank = semantic_status_score_adjustment(Some(node));
                    candidate_rank > existing_rank
                        || (candidate_rank - existing_rank).abs() < f32::EPSILON
                            && (node.confidence > existing.confidence
                                || (node.confidence - existing.confidence).abs() < f32::EPSILON
                                    && node.version > existing.version)
                }
                None => true,
            };
            if replace {
                self.canonical_semantic_state.insert(key, node.clone());
            }
        }
    }

    pub(crate) fn canonical_semantic_for_memory(
        &self,
        memory: &MemoryRecord,
    ) -> Option<&SemanticNode> {
        let node = memory.semantic_node.as_ref()?;
        let key = Self::canonical_semantic_key(&node.entity_id, &node.attribute_key);
        self.canonical_semantic_state.get(&key)
    }

    pub(crate) fn query_canonical_semantic(
        &self,
        conversation_id: Option<&str>,
        entity_id: Option<&str>,
        attribute_key: Option<&str>,
    ) -> CanonicalSemanticResponse {
        let items = self
            .canonical_semantic_state
            .iter()
            .filter(|(_, node)| {
                entity_id
                    .map(|entity| node.entity_id == entity)
                    .unwrap_or(true)
                    && attribute_key
                        .map(|attribute| node.attribute_key == attribute)
                        .unwrap_or(true)
            })
            .map(|(canonical_key, node)| {
                let source_memory_ids = self
                    .memories
                    .iter()
                    .filter(|memory| {
                        conversation_id
                            .map(|cid| memory.conversation_id == cid)
                            .unwrap_or(true)
                    })
                    .filter_map(|memory| {
                        memory.semantic_node.as_ref().and_then(|candidate| {
                            (candidate.entity_id == node.entity_id
                                && candidate.attribute_key == node.attribute_key)
                                .then_some(memory.id)
                        })
                    })
                    .collect::<Vec<_>>();
                let item_conversation = conversation_id.map(|id| id.to_string()).or_else(|| {
                    self.memories
                        .iter()
                        .find(|memory| source_memory_ids.iter().any(|id| *id == memory.id))
                        .map(|memory| memory.conversation_id.clone())
                });
                CanonicalSemanticItem {
                    canonical_key: canonical_key.clone(),
                    conversation_id: item_conversation,
                    semantic_node: node.clone(),
                    source_memory_ids,
                    audit_entries: self
                        .canonical_semantic_audit
                        .iter()
                        .filter(|entry| {
                            entry.canonical_key == *canonical_key
                                && conversation_id
                                    .map(|cid| entry.conversation_id == cid)
                                    .unwrap_or(true)
                        })
                        .cloned()
                        .collect(),
                }
            })
            .collect::<Vec<_>>();

        CanonicalSemanticResponse { items }
    }

    pub(crate) fn semantic_temporal_bounds(
        &self,
        conversation_id: Option<&str>,
        node: &SemanticNode,
    ) -> (Option<DateTime<Utc>>, Option<DateTime<Utc>>) {
        let related_times = self
            .memories
            .iter()
            .filter(|memory| {
                conversation_id
                    .map(|cid| memory.conversation_id == cid)
                    .unwrap_or(true)
            })
            .filter_map(|memory| {
                memory.semantic_node.as_ref().and_then(|candidate| {
                    (candidate.entity_id == node.entity_id
                        && candidate.attribute_key == node.attribute_key)
                        .then_some(memory.created_at)
                })
            })
            .collect::<Vec<_>>();
        let valid_at = related_times.iter().min().copied();
        let invalid_at = if matches!(
            node.status,
            SemanticStatus::Superseded | SemanticStatus::Deprecated
        ) {
            related_times.iter().max().copied()
        } else {
            None
        };
        (valid_at, invalid_at)
    }

    pub(crate) fn mutate_canonical_semantic(
        &mut self,
        req: &CanonicalSemanticMutationRequest,
    ) -> CanonicalSemanticMutationResponse {
        let canonical_key = Self::canonical_semantic_key(&req.entity_id, &req.attribute_key);
        let mut updated_memory_ids = Vec::new();
        let before_value = self
            .canonical_semantic_state
            .get(&canonical_key)
            .map(|node| node.attribute_value.clone());

        let matching_ids = self
            .memories
            .iter()
            .filter(|memory| memory.conversation_id == req.conversation_id)
            .filter_map(|memory| {
                memory.semantic_node.as_ref().and_then(|node| {
                    (node.entity_id == req.entity_id && node.attribute_key == req.attribute_key)
                        .then_some(memory.id)
                })
            })
            .collect::<Vec<_>>();
        let mut reconciled_values = self
            .memories
            .iter()
            .filter(|memory| matching_ids.contains(&memory.id))
            .filter_map(|memory| {
                memory
                    .semantic_node
                    .as_ref()
                    .map(|node| node.attribute_value.clone())
            })
            .collect::<Vec<_>>();
        reconciled_values.sort();
        reconciled_values.dedup();

        match req.action.as_str() {
            "deprecate" => {
                for memory in self
                    .memories
                    .iter_mut()
                    .filter(|memory| matching_ids.contains(&memory.id))
                {
                    if let Some(node) = memory.semantic_node.as_mut() {
                        node.status = SemanticStatus::Deprecated;
                        updated_memory_ids.push(memory.id);
                    }
                }
            }
            "repair" => {
                let Some(replacement) = req.replacement_value.as_ref() else {
                    return CanonicalSemanticMutationResponse {
                        canonical_key,
                        applied: false,
                        action: req.action.clone(),
                        semantic_node: None,
                        updated_memory_ids,
                        audit_entry: None,
                    };
                };
                for memory in self
                    .memories
                    .iter_mut()
                    .filter(|memory| matching_ids.contains(&memory.id))
                {
                    if let Some(node) = memory.semantic_node.as_mut() {
                        node.attribute_value = replacement.clone();
                        node.status = SemanticStatus::Active;
                        memory.fact_value = Some(replacement.clone());
                        updated_memory_ids.push(memory.id);
                    }
                }
            }
            "resolve" => {
                let Some(winner_id) = req.winner_semantic_id.as_deref() else {
                    return CanonicalSemanticMutationResponse {
                        canonical_key,
                        applied: false,
                        action: req.action.clone(),
                        semantic_node: None,
                        updated_memory_ids,
                        audit_entry: None,
                    };
                };
                for memory in self
                    .memories
                    .iter_mut()
                    .filter(|memory| matching_ids.contains(&memory.id))
                {
                    if let Some(node) = memory.semantic_node.as_mut() {
                        if node.semantic_id == winner_id {
                            node.status = SemanticStatus::Active;
                            node.conflict_set_id = None;
                        } else {
                            node.status = SemanticStatus::Superseded;
                        }
                        if let Some(conflict) = memory.semantic_conflict.as_mut() {
                            conflict.resolved = true;
                            conflict.winner_semantic_id = Some(winner_id.to_string());
                        }
                        updated_memory_ids.push(memory.id);
                    }
                }
            }
            _ => {
                return CanonicalSemanticMutationResponse {
                    canonical_key,
                    applied: false,
                    action: req.action.clone(),
                    semantic_node: None,
                    updated_memory_ids,
                    audit_entry: None,
                };
            }
        }

        self.rebuild_canonical_semantic_state();
        let semantic_node = self.canonical_semantic_state.get(&canonical_key).cloned();
        let reason_code = req.reason_code.clone().unwrap_or_else(|| {
            if req.action == "repair" && reconciled_values.len() > 1 {
                "canonical_repair_conflict_reconciled".to_string()
            } else {
                format!("canonical_{}", req.action)
            }
        });
        let evidence_bundle = req.evidence_bundle.clone().unwrap_or_else(|| {
            let mut evidence = Vec::new();
            if let Some(before) = before_value.as_ref() {
                evidence.push(format!("before={before}"));
            }
            if !reconciled_values.is_empty() {
                evidence.push(format!("reconciled_values={}", reconciled_values.join("|")));
            }
            if let Some(node) = semantic_node.as_ref() {
                evidence.push(format!("after={}", node.attribute_value));
            }
            evidence.push(format!("updated_memories={}", updated_memory_ids.len()));
            evidence
        });
        let audit_entry = semantic_node
            .as_ref()
            .map(|node| CanonicalSemanticAuditEntry {
                canonical_key: canonical_key.clone(),
                conversation_id: req.conversation_id.clone(),
                action: req.action.clone(),
                before_value: before_value.clone(),
                after_value: Some(node.attribute_value.clone()),
                reason: if req.action == "repair" && !reconciled_values.is_empty() {
                    format!(
                        "canonical semantic repair reconciled [{}] into {}",
                        reconciled_values.join(", "),
                        node.attribute_value
                    )
                } else {
                    format!("canonical semantic {} applied", req.action)
                },
                reason_code,
                evidence_bundle,
                affected_memory_ids: updated_memory_ids.clone(),
            });
        if let Some(entry) = audit_entry.clone() {
            self.canonical_semantic_audit.push(entry);
        }

        CanonicalSemanticMutationResponse {
            canonical_key,
            applied: !updated_memory_ids.is_empty(),
            action: req.action.clone(),
            semantic_node,
            updated_memory_ids,
            audit_entry,
        }
    }

    pub(crate) fn run_semantic_sleep_cycle(
        &mut self,
        conversation_id: &str,
    ) -> SemanticSleepCycleResponse {
        let (conversation_count, window_start, window_end) =
            self.batch_window_for_conversation(conversation_id);
        let job_id = format!(
            "sleep:{}:{}",
            conversation_id,
            Utc::now().timestamp_millis()
        );
        let (priority, trigger_reason) =
            self.consolidation_priority_for_conversation(conversation_id);
        self.consolidation_jobs.push(ConsolidationJob {
            job_id: job_id.clone(),
            job_type: "semantic_sleep_cycle".to_string(),
            scope: "conversation".to_string(),
            status: "running".to_string(),
            priority,
            trigger_reason,
            scheduled_at: Utc::now(),
            completed_at: None,
            conversation_id: Some(conversation_id.to_string()),
            conversation_count,
            window_start,
            window_end,
            result_summary: None,
            result_metrics: None,
            result_clusters: Vec::new(),
        });
        let mut repaired_nodes = 0usize;
        let mut resolved_conflicts = 0usize;
        let mut deprecated_nodes = 0usize;
        let mut audit_entries_created = 0usize;
        let mut cluster_summaries_created = 0usize;
        let mut cluster_summaries = Vec::new();

        let canonical_targets = self
            .canonical_semantic_state
            .iter()
            .filter(|(_, node)| {
                self.memories.iter().any(|memory| {
                    memory.conversation_id == conversation_id
                        && memory
                            .semantic_node
                            .as_ref()
                            .map(|candidate| {
                                candidate.entity_id == node.entity_id
                                    && candidate.attribute_key == node.attribute_key
                            })
                            .unwrap_or(false)
                })
            })
            .map(|(key, node)| (key.clone(), node.clone()))
            .collect::<Vec<_>>();

        for (canonical_key, node) in &canonical_targets {
            if node.status == SemanticStatus::Disputed && node.confidence < 0.5 {
                let result = self.mutate_canonical_semantic(&CanonicalSemanticMutationRequest {
                    conversation_id: conversation_id.to_string(),
                    entity_id: node.entity_id.clone(),
                    attribute_key: node.attribute_key.clone(),
                    action: "deprecate".to_string(),
                    winner_semantic_id: None,
                    replacement_value: None,
                    reason_code: Some("sleep_cycle_low_conflict_confidence".to_string()),
                    evidence_bundle: Some(vec![format!("confidence={:.2}", node.confidence)]),
                });
                if result.applied {
                    deprecated_nodes += 1;
                    if result.audit_entry.is_some() {
                        audit_entries_created += 1;
                    }
                }
                continue;
            }

            if node.status == SemanticStatus::Disputed && node.confidence >= 0.5 {
                let has_explicit_conflict_metadata = self
                    .memories
                    .iter()
                    .filter(|memory| memory.conversation_id == conversation_id)
                    .filter(|memory| {
                        memory
                            .semantic_node
                            .as_ref()
                            .map(|candidate| {
                                Self::canonical_semantic_key(
                                    &candidate.entity_id,
                                    &candidate.attribute_key,
                                ) == *canonical_key
                            })
                            .unwrap_or(false)
                    })
                    .any(|memory| {
                        memory
                            .semantic_conflict
                            .as_ref()
                            .map(|conflict| !conflict.resolved)
                            .unwrap_or(false)
                    });
                let mut by_value: HashMap<String, (usize, f32, String, f32, u32)> = HashMap::new();
                for candidate in self
                    .memories
                    .iter()
                    .filter(|memory| memory.conversation_id == conversation_id)
                    .filter_map(|memory| memory.semantic_node.as_ref())
                    .filter(|candidate| {
                        Self::canonical_semantic_key(&candidate.entity_id, &candidate.attribute_key)
                            == *canonical_key
                            && candidate.status != SemanticStatus::Superseded
                            && candidate.status != SemanticStatus::Deprecated
                    })
                {
                    let entry = by_value
                        .entry(candidate.attribute_value.clone())
                        .or_insert((
                            0,
                            0.0,
                            candidate.semantic_id.clone(),
                            candidate.confidence,
                            candidate.version,
                        ));
                    entry.0 += 1;
                    entry.1 += candidate.confidence;
                    if candidate.confidence > entry.3
                        || ((candidate.confidence - entry.3).abs() < f32::EPSILON
                            && candidate.version > entry.4)
                    {
                        entry.2 = candidate.semantic_id.clone();
                        entry.3 = candidate.confidence;
                        entry.4 = candidate.version;
                    }
                }
                let mut ranked_resolution = by_value.into_iter().collect::<Vec<_>>();
                ranked_resolution.sort_by(|a, b| {
                    b.1 .0
                        .cmp(&a.1 .0)
                        .then_with(|| b.1 .1.partial_cmp(&a.1 .1).unwrap_or(Ordering::Equal))
                });
                if has_explicit_conflict_metadata && ranked_resolution.len() >= 2 {
                    let (winner_value, (winner_count, winner_sum, winner_id, _, _)) =
                        &ranked_resolution[0];
                    let (_, (runner_count, runner_sum, _, _, _)) = &ranked_resolution[1];
                    let has_decisive_support = winner_count > runner_count
                        || (winner_count == runner_count && winner_sum - runner_sum >= 0.10);
                    if has_decisive_support {
                        let evidence_bundle = ranked_resolution
                            .iter()
                            .map(|(value, (count, sum, _, _, _))| {
                                format!("{value}:count={count},confidence_sum={sum:.2}")
                            })
                            .collect::<Vec<_>>();
                        let result =
                            self.mutate_canonical_semantic(&CanonicalSemanticMutationRequest {
                                conversation_id: conversation_id.to_string(),
                                entity_id: node.entity_id.clone(),
                                attribute_key: node.attribute_key.clone(),
                                action: "resolve".to_string(),
                                winner_semantic_id: Some(winner_id.clone()),
                                replacement_value: None,
                                reason_code: Some(
                                    "sleep_cycle_conflict_resolution_sweep".to_string(),
                                ),
                                evidence_bundle: Some({
                                    let mut evidence = evidence_bundle;
                                    evidence.push(format!("winner_value={winner_value}"));
                                    evidence.push(format!("winner_semantic_id={winner_id}"));
                                    evidence
                                }),
                            });
                        if result.applied {
                            repaired_nodes += 1;
                            resolved_conflicts += 1;
                            if result.audit_entry.is_some() {
                                audit_entries_created += 1;
                            }
                            continue;
                        }
                    }
                }

                let supporting = self
                    .memories
                    .iter()
                    .filter(|memory| memory.conversation_id == conversation_id)
                    .filter_map(|memory| memory.semantic_node.as_ref())
                    .find(|candidate| {
                        Self::canonical_semantic_key(&candidate.entity_id, &candidate.attribute_key)
                            == *canonical_key
                            && candidate.status == SemanticStatus::Active
                    })
                    .map(|candidate| candidate.attribute_value.clone());
                let repeated_support = if supporting.is_none() {
                    let mut by_value: HashMap<String, (usize, f32)> = HashMap::new();
                    for candidate in self
                        .memories
                        .iter()
                        .filter(|memory| memory.conversation_id == conversation_id)
                        .filter_map(|memory| memory.semantic_node.as_ref())
                        .filter(|candidate| {
                            Self::canonical_semantic_key(
                                &candidate.entity_id,
                                &candidate.attribute_key,
                            ) == *canonical_key
                                && candidate.status != SemanticStatus::Superseded
                                && candidate.status != SemanticStatus::Deprecated
                        })
                    {
                        let entry = by_value
                            .entry(candidate.attribute_value.clone())
                            .or_insert((0, 0.0));
                        entry.0 += 1;
                        entry.1 += candidate.confidence;
                    }
                    let mut ranked = by_value.into_iter().collect::<Vec<_>>();
                    ranked.sort_by(|a, b| {
                        b.1 .0
                            .cmp(&a.1 .0)
                            .then_with(|| b.1 .1.partial_cmp(&a.1 .1).unwrap_or(Ordering::Equal))
                    });
                    match ranked.as_slice() {
                        [(value, (count, _)), (_, (next_count, _)), ..]
                            if *count >= 2 && count > next_count =>
                        {
                            Some(value.clone())
                        }
                        [(value, (count, _))] if *count >= 2 => Some(value.clone()),
                        _ => None,
                    }
                } else {
                    None
                };
                let has_active_support = supporting.is_some();
                if let Some(replacement) = supporting.or(repeated_support) {
                    let reason_code = if has_active_support {
                        "sleep_cycle_supporting_active_value"
                    } else {
                        "sleep_cycle_repeated_supporting_value"
                    };
                    let result =
                        self.mutate_canonical_semantic(&CanonicalSemanticMutationRequest {
                            conversation_id: conversation_id.to_string(),
                            entity_id: node.entity_id.clone(),
                            attribute_key: node.attribute_key.clone(),
                            action: "repair".to_string(),
                            winner_semantic_id: None,
                            replacement_value: Some(replacement),
                            reason_code: Some(reason_code.to_string()),
                            evidence_bundle: Some(vec![format!(
                                "supporting_replacement={}",
                                canonical_key
                            )]),
                        });
                    if result.applied {
                        repaired_nodes += 1;
                        if result.audit_entry.is_some() {
                            audit_entries_created += 1;
                        }
                    }
                }
            }
        }

        let episodic_groups = self
            .memories
            .iter()
            .filter(|memory| {
                memory.conversation_id == conversation_id && memory.layer == MemoryLayer::Episodic
            })
            .filter_map(|memory| {
                memory.episodic_metadata.as_ref().and_then(|meta| {
                    meta.failure_type
                        .as_ref()
                        .map(|failure| (memory.id, failure.clone()))
                })
            })
            .fold(
                HashMap::<String, Vec<u64>>::new(),
                |mut acc, (id, failure)| {
                    acc.entry(failure).or_default().push(id);
                    acc
                },
            );

        for (failure_type, anchor_ids) in episodic_groups {
            if anchor_ids.len() < 3 {
                continue;
            }
            let mut sorted_anchor_ids = anchor_ids.clone();
            sorted_anchor_ids.sort_unstable();
            let content = format!(
                "睡眠聚类摘要：重复失败模式 [{}] 在会话 {} 中出现了 {} 次，需要优先治理。",
                failure_type,
                conversation_id,
                sorted_anchor_ids.len()
            );
            let already_exists = self.memories.iter().any(|memory| {
                memory.conversation_id == conversation_id
                    && memory.layer == MemoryLayer::Archive
                    && memory.source == "sleep_cycle_clusterer"
                    && {
                        let mut existing = memory.anchor_ids.clone();
                        existing.sort_unstable();
                        existing == sorted_anchor_ids
                    }
            });
            if already_exists {
                continue;
            }
            let summary_memory_id = ID_GEN.fetch_add(1, AtomicOrdering::Relaxed);
            self.memories.push(MemoryRecord {
                id: summary_memory_id,
                conversation_id: conversation_id.to_string(),
                layer: MemoryLayer::Archive,
                content: content.clone(),
                tokens: Self::tokenize(&content),
                source: "sleep_cycle_clusterer".to_string(),
                trust: 0.9,
                created_at: Utc::now(),
                anchor_ids: sorted_anchor_ids.clone(),
                entity: None,
                fact_key: None,
                fact_value: Some(failure_type.clone()),
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
            cluster_summaries_created += 1;
            cluster_summaries.push(ConsolidationClusterSummary {
                cluster_kind: "episodic_failure".to_string(),
                cluster_key: failure_type,
                item_count: sorted_anchor_ids.len(),
                anchor_ids: sorted_anchor_ids,
                summary_memory_id: Some(summary_memory_id),
                conversation_ids: vec![conversation_id.to_string()],
            });
        }

        self.rebuild_meta_memory_state();

        let response = SemanticSleepCycleResponse {
            conversation_id: conversation_id.to_string(),
            job_id: job_id.clone(),
            job_status: "completed".to_string(),
            conversation_count,
            window_start,
            window_end,
            scanned_canonical_nodes: canonical_targets.len(),
            repaired_nodes,
            resolved_conflicts,
            deprecated_nodes,
            audit_entries_created,
            cluster_summaries_created,
            cluster_summaries,
        };
        if let Some(job) = self
            .consolidation_jobs
            .iter_mut()
            .find(|job| job.job_id == job_id)
        {
            job.status = "completed".to_string();
            job.completed_at = Some(Utc::now());
            job.result_summary = Some(format!(
                "repaired={}, resolved_conflicts={}, deprecated={}, audit_entries={}, clustered={}",
                response.repaired_nodes,
                response.resolved_conflicts,
                response.deprecated_nodes,
                response.audit_entries_created,
                response.cluster_summaries_created
            ));
            job.result_metrics = Some(ConsolidationJobMetrics {
                repaired_nodes: response.repaired_nodes,
                resolved_conflicts: response.resolved_conflicts,
                deprecated_nodes: response.deprecated_nodes,
                audit_entries_created: response.audit_entries_created,
                cluster_summaries_created: response.cluster_summaries_created,
            });
            job.result_clusters = response.cluster_summaries.clone();
        }
        response
    }

    pub(crate) fn run_global_sleep_cycle(&mut self) -> SemanticSleepCycleResponse {
        let (conversation_count, window_start, window_end) = self.batch_window_for_global();
        let job_id = format!("sleep:global:{}", Utc::now().timestamp_millis());
        let conversation_ids = self
            .memories
            .iter()
            .filter(|memory| memory.conversation_id != "global")
            .map(|memory| memory.conversation_id.clone())
            .collect::<HashSet<_>>();
        let (priority, trigger_reason) = if conversation_ids.is_empty() {
            (10, "global_window_idle".to_string())
        } else {
            let max_priority = conversation_ids
                .iter()
                .map(|conversation_id| {
                    self.consolidation_priority_for_conversation(conversation_id)
                        .0
                })
                .max()
                .unwrap_or(10);
            (
                max_priority.max(10),
                format!(
                    "global_window:{} conversations eligible for clustering",
                    conversation_ids.len()
                ),
            )
        };
        self.consolidation_jobs.push(ConsolidationJob {
            job_id: job_id.clone(),
            job_type: "semantic_sleep_cycle".to_string(),
            scope: "global".to_string(),
            status: "running".to_string(),
            priority,
            trigger_reason,
            scheduled_at: Utc::now(),
            completed_at: None,
            conversation_id: Some("global".to_string()),
            conversation_count,
            window_start,
            window_end,
            result_summary: None,
            result_metrics: None,
            result_clusters: Vec::new(),
        });

        let mut cluster_summaries_created = 0usize;
        let mut cluster_summaries = Vec::new();

        let episodic_groups = self
            .memories
            .iter()
            .filter(|memory| {
                memory.conversation_id != "global" && memory.layer == MemoryLayer::Episodic
            })
            .filter_map(|memory| {
                memory.episodic_metadata.as_ref().and_then(|meta| {
                    meta.failure_type
                        .as_ref()
                        .map(|failure| (memory.id, memory.conversation_id.clone(), failure.clone()))
                })
            })
            .fold(
                HashMap::<String, Vec<(u64, String)>>::new(),
                |mut acc, (id, conversation_id, failure)| {
                    acc.entry(failure).or_default().push((id, conversation_id));
                    acc
                },
            );

        for (failure_type, members) in episodic_groups {
            if members.len() < 3 {
                continue;
            }
            let mut sorted_anchor_ids = members.iter().map(|(id, _)| *id).collect::<Vec<_>>();
            sorted_anchor_ids.sort_unstable();
            let mut conversation_ids = members
                .iter()
                .map(|(_, conversation_id)| conversation_id.clone())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            conversation_ids.sort();
            if conversation_ids.len() < 2 {
                continue;
            }
            let content = format!(
                "全局睡眠聚类摘要：重复失败模式 [{}] 横跨 {} 个会话出现了 {} 次，需要优先做跨会话修复。",
                failure_type,
                conversation_ids.len(),
                sorted_anchor_ids.len()
            );
            let already_exists = self.memories.iter().any(|memory| {
                memory.conversation_id == "global"
                    && memory.layer == MemoryLayer::Archive
                    && memory.source == "sleep_cycle_clusterer_global"
                    && {
                        let mut existing = memory.anchor_ids.clone();
                        existing.sort_unstable();
                        existing == sorted_anchor_ids
                    }
            });
            if already_exists {
                continue;
            }
            let summary_memory_id = ID_GEN.fetch_add(1, AtomicOrdering::Relaxed);
            self.memories.push(MemoryRecord {
                id: summary_memory_id,
                conversation_id: "global".to_string(),
                layer: MemoryLayer::Archive,
                content: content.clone(),
                tokens: Self::tokenize(&content),
                source: "sleep_cycle_clusterer_global".to_string(),
                trust: 0.95,
                created_at: Utc::now(),
                anchor_ids: sorted_anchor_ids.clone(),
                entity: None,
                fact_key: Some("cross_conversation_failure".to_string()),
                fact_value: Some(failure_type.clone()),
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
            cluster_summaries_created += 1;
            cluster_summaries.push(ConsolidationClusterSummary {
                cluster_kind: "cross_conversation_failure".to_string(),
                cluster_key: failure_type,
                item_count: sorted_anchor_ids.len(),
                anchor_ids: sorted_anchor_ids,
                summary_memory_id: Some(summary_memory_id),
                conversation_ids,
            });
        }

        self.rebuild_meta_memory_state();

        let response = SemanticSleepCycleResponse {
            conversation_id: "global".to_string(),
            job_id: job_id.clone(),
            job_status: "completed".to_string(),
            conversation_count,
            window_start,
            window_end,
            scanned_canonical_nodes: 0,
            repaired_nodes: 0,
            resolved_conflicts: 0,
            deprecated_nodes: 0,
            audit_entries_created: 0,
            cluster_summaries_created,
            cluster_summaries,
        };
        if let Some(job) = self
            .consolidation_jobs
            .iter_mut()
            .find(|job| job.job_id == job_id)
        {
            job.status = "completed".to_string();
            job.completed_at = Some(Utc::now());
            job.result_summary = Some(format!(
                "repaired={}, resolved_conflicts={}, deprecated={}, audit_entries={}, clustered={}",
                response.repaired_nodes,
                response.resolved_conflicts,
                response.deprecated_nodes,
                response.audit_entries_created,
                response.cluster_summaries_created
            ));
            job.result_metrics = Some(ConsolidationJobMetrics {
                repaired_nodes: response.repaired_nodes,
                resolved_conflicts: response.resolved_conflicts,
                deprecated_nodes: response.deprecated_nodes,
                audit_entries_created: response.audit_entries_created,
                cluster_summaries_created: response.cluster_summaries_created,
            });
            job.result_clusters = response.cluster_summaries.clone();
        }
        response
    }
}
