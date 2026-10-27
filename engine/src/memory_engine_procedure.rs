use super::*;

impl MemoryEngine {
    pub(crate) fn activate_procedure(
        &mut self,
        conversation_id: &str,
        query: &str,
    ) -> Option<ProcedureActivationResponse> {
        let (current_project, current_task) =
            self.current_applicability_context(Some(conversation_id));
        let top_match = self
            .memories
            .iter()
            .filter(|memory| {
                memory.conversation_id == conversation_id
                    && memory.layer == MemoryLayer::Procedure
                    && memory.procedure_schema.is_some()
            })
            .filter_map(|memory| {
                let schema = memory.procedure_schema.as_ref()?;
                let related_fact_key = memory.fact_key.clone();
                let related_precondition_key = related_fact_key
                    .as_ref()
                    .map(|key| format!("{}_precondition", key));
                let related_failure_motif_key = related_fact_key
                    .as_ref()
                    .map(|key| format!("{}_failure_motif", key));
                let semantic_preconditions = self
                    .memories
                    .iter()
                    .filter(|candidate| {
                        candidate.conversation_id == conversation_id
                            && candidate.layer == MemoryLayer::Fact
                            && candidate.entity == memory.entity
                            && candidate.fact_key.as_deref() == related_precondition_key.as_deref()
                    })
                    .filter_map(|candidate| candidate.fact_value.clone())
                    .collect::<Vec<_>>();
                let semantic_recovery_patterns = self
                    .memories
                    .iter()
                    .filter(|candidate| {
                        candidate.conversation_id == conversation_id
                            && candidate.layer == MemoryLayer::Fact
                            && candidate.entity == memory.entity
                            && candidate.fact_key.as_deref()
                                == memory
                                    .fact_key
                                    .as_ref()
                                    .map(|key| format!("{}_recovery_pattern", key))
                                    .as_deref()
                    })
                    .filter_map(|candidate| candidate.fact_value.clone())
                    .collect::<Vec<_>>();
                let semantic_failure_motifs = self
                    .memories
                    .iter()
                    .filter(|candidate| {
                        candidate.conversation_id == conversation_id
                            && candidate.layer == MemoryLayer::Fact
                            && candidate.entity == memory.entity
                            && candidate.fact_key.as_deref() == related_failure_motif_key.as_deref()
                    })
                    .filter_map(|candidate| candidate.fact_value.clone())
                    .collect::<Vec<_>>();
                let semantic_precondition_match = semantic_preconditions
                    .iter()
                    .any(|condition| procedure_phrase_matches(query, condition));
                let semantic_recovery_match = semantic_recovery_patterns
                    .iter()
                    .any(|pattern| procedure_phrase_matches(query, pattern));
                let confidence = procedure_activation_confidence(memory, schema, query)
                    + if semantic_precondition_match {
                        0.22
                    } else {
                        0.0
                    }
                    + if semantic_recovery_match { 0.10 } else { 0.0 };
                if confidence <= 0.0 {
                    return None;
                }
                let evaluation = self.evaluate_procedure(conversation_id, memory.id)?;
                if !evaluation.activation_eligible {
                    return None;
                }
                let mut safety_warnings = self
                    .memories
                    .iter()
                    .filter(|candidate| {
                        candidate.conversation_id == conversation_id
                            && candidate.layer == MemoryLayer::Fact
                            && candidate.entity == memory.entity
                            && (candidate.fact_key == related_fact_key
                                || candidate.fact_key == related_precondition_key)
                    })
                    .filter_map(|candidate| {
                        candidate.semantic_node.as_ref().and_then(|node| {
                            (node.status == SemanticStatus::Disputed).then_some(format!(
                                "相关语义冲突: {} 目前存在争议值 [{}]",
                                node.attribute_key, node.attribute_value
                            ))
                        })
                    })
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>();
                safety_warnings.sort();
                let activation_eligible = safety_warnings.is_empty();
                let safety_penalty = if activation_eligible { 0.0 } else { 0.35 };
                let preconditions_met = (schema.trigger_conditions.is_empty()
                    && semantic_preconditions.is_empty())
                    || schema
                        .trigger_conditions
                        .iter()
                        .any(|condition| procedure_phrase_matches(query, condition))
                    || semantic_preconditions
                        .iter()
                        .any(|condition| procedure_phrase_matches(query, condition));
                let semantic_bonus = (semantic_preconditions.len() as f32 * 0.04)
                    + (semantic_recovery_patterns.len() as f32 * 0.02)
                    + (semantic_failure_motifs.len() as f32 * 0.01);
                let cached_preferred = false;
                let adjusted_confidence = (confidence - safety_penalty).max(0.0);
                let ranking_score = adjusted_confidence
                    + procedure_health_bonus(&evaluation)
                    + (memory.version as f32 * 0.01)
                    + semantic_bonus
                    + if cached_preferred { 0.30 } else { 0.0 }
                    + semantic_applicability_score_adjustment(
                        memory.semantic_node.as_ref(),
                        query,
                        current_project.as_deref(),
                        current_task.as_deref(),
                    );
                Some((
                    memory.id,
                    adjusted_confidence,
                    activation_eligible,
                    safety_warnings,
                    preconditions_met,
                    ranking_score,
                    semantic_preconditions,
                    semantic_recovery_patterns,
                    semantic_failure_motifs,
                ))
            })
            .max_by(|a, b| a.5.partial_cmp(&b.5).unwrap_or(Ordering::Equal))?;

        self.mark_retrieved(&[top_match.0]);

        let memory = self
            .memories
            .iter()
            .find(|memory| memory.id == top_match.0)?;
        let schema = memory.procedure_schema.as_ref()?;
        Some(ProcedureActivationResponse {
            procedure_id: memory.id,
            procedure_content: memory.content.clone(),
            source: Some(memory.source.clone()),
            skill_file_path: memory
                .source
                .strip_prefix("skill_file_import:")
                .map(str::to_string)
                .or_else(|| {
                    memory
                        .fact_value
                        .clone()
                        .filter(|value| value.starts_with("skills/"))
                }),
            activation_eligible: top_match.2,
            safety_warnings: top_match.3.clone(),
            preconditions_met: top_match.4,
            confidence: top_match.1,
            trigger_conditions: schema.trigger_conditions.clone(),
            steps: schema
                .steps
                .iter()
                .map(|step| ProcedureStepStatus {
                    step: step.clone(),
                    status: "pending".to_string(),
                })
                .collect(),
            rollback_steps: schema.rollback_steps.clone(),
            success_criteria: schema.success_criteria.clone(),
            semantic_preconditions: top_match.6.clone(),
            semantic_recovery_patterns: top_match.7.clone(),
            semantic_failure_motifs: top_match.8.clone(),
        })
    }

    pub(crate) fn execute_procedure(
        &mut self,
        req: &ProcedureExecutionRequest,
    ) -> Option<ProcedureExecutionResult> {
        let procedure = self
            .memories
            .iter()
            .find(|memory| {
                memory.conversation_id == req.conversation_id
                    && memory.id == req.procedure_id
                    && memory.layer == MemoryLayer::Procedure
                    && memory.procedure_schema.is_some()
            })
            .cloned()?;
        let schema = procedure.procedure_schema.clone()?;
        let semantic_recovery_guidance = procedure
            .fact_key
            .as_ref()
            .map(|fact_key| format!("{}_recovery_pattern", fact_key))
            .map(|recovery_key| {
                self.memories
                    .iter()
                    .filter(|memory| {
                        memory.conversation_id == req.conversation_id
                            && memory.layer == MemoryLayer::Fact
                            && memory.entity == procedure.entity
                            && memory.fact_key.as_deref() == Some(recovery_key.as_str())
                    })
                    .filter_map(|memory| memory.fact_value.clone())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let semantic_failure_motifs = procedure
            .fact_key
            .as_ref()
            .map(|fact_key| format!("{}_failure_motif", fact_key))
            .map(|motif_key| {
                self.memories
                    .iter()
                    .filter(|memory| {
                        memory.conversation_id == req.conversation_id
                            && memory.layer == MemoryLayer::Fact
                            && memory.entity == procedure.entity
                            && memory.fact_key.as_deref() == Some(motif_key.as_str())
                    })
                    .filter_map(|memory| memory.fact_value.clone())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let step_status_map = req
            .steps
            .iter()
            .map(|step| (step.step.clone(), normalize_step_status(&step.status)))
            .collect::<HashMap<_, _>>();
        let steps = schema
            .steps
            .iter()
            .map(|step| ProcedureStepStatus {
                step: step.clone(),
                status: step_status_map
                    .get(step)
                    .cloned()
                    .unwrap_or_else(|| "pending".to_string()),
            })
            .collect::<Vec<_>>();

        let execution_status = if steps.iter().any(|step| step.status == "failed") {
            "failed"
        } else if !steps.is_empty() && steps.iter().all(|step| step.status == "completed") {
            "completed"
        } else if steps
            .iter()
            .any(|step| matches!(step.status.as_str(), "completed" | "in_progress"))
        {
            "in_progress"
        } else {
            "pending"
        }
        .to_string();
        let current_step_index = steps
            .iter()
            .position(|step| step.status == "in_progress")
            .or_else(|| steps.iter().position(|step| step.status == "failed"))
            .or_else(|| steps.iter().position(|step| step.status == "pending"));
        let retry_count = self
            .latest_procedure_execution_state(&req.conversation_id, req.procedure_id)
            .map(|state| {
                if execution_status == "failed" {
                    state.retry_count.saturating_add(1)
                } else {
                    state.retry_count
                }
            })
            .unwrap_or_else(|| if execution_status == "failed" { 1 } else { 0 });
        let execution_state = ProcedureExecutionState {
            procedure_id: req.procedure_id,
            current_step_index,
            retry_count,
            outcome: execution_status.clone(),
            failure_reason: req.failure_reason.clone(),
            success_evidence: req.success_evidence.clone(),
            steps: steps.clone(),
        };

        let summary = format!(
            "procedure_execution 规则流程记录: procedure_id={}; status={}; current_step_index={:?}; retry_count={}; steps={}; failure_reason={}; success_evidence={}; semantic_recovery_guidance={}; semantic_failure_motifs={}",
            req.procedure_id,
            execution_status,
            current_step_index,
            retry_count,
            steps
                .iter()
                .map(|step| format!("{}[{}]", step.step, step.status))
                .collect::<Vec<_>>()
                .join(", "),
            req.failure_reason.clone().unwrap_or_else(|| "none".to_string()),
            req.success_evidence
                .clone()
                .unwrap_or_else(|| "none".to_string()),
            if semantic_recovery_guidance.is_empty() {
                "none".to_string()
            } else {
                semantic_recovery_guidance.join(" | ")
            },
            if semantic_failure_motifs.is_empty() {
                "none".to_string()
            } else {
                semantic_failure_motifs.join(" | ")
            },
        );

        let before_len = self.memories.len();
        let ingest = self.ingest(IngestRequest {
            conversation_id: req.conversation_id.clone(),
            layer: MemoryLayer::Working,
            content: summary,
            source: Some("procedure_executor".to_string()),
            trust: Some(0.95),
            entity: procedure.entity.clone(),
            fact_key: Some("procedure_execution".to_string()),
            fact_value: procedure.fact_value.clone(),
        });
        if !ingest.accepted {
            return None;
        }

        let created_ids = self
            .memories
            .iter()
            .skip(before_len)
            .map(|memory| memory.id)
            .collect::<Vec<_>>();
        if let Some(execution_memory_id) = ingest.memory_id {
            if let Some(memory) = self
                .memories
                .iter_mut()
                .find(|memory| memory.id == execution_memory_id)
            {
                memory.procedure_execution_state = Some(execution_state.clone());
            }
            self.refresh_episodic_metadata_for_anchor(&req.conversation_id, execution_memory_id);
        }
        let execution_strategy = MetaMemoryController::procedure_execution_strategy();
        if execution_strategy.apply_procedure_feedback {
            self.apply_procedure_feedback_to_semantic(
                &req.conversation_id,
                &procedure,
                &execution_state,
                ingest.memory_id,
            );
        }
        let event_id = created_ids.iter().copied().find(|id| {
            self.memories
                .iter()
                .find(|memory| memory.id == *id)
                .map(|memory| memory.layer == MemoryLayer::Event)
                .unwrap_or(false)
        });

        Some(ProcedureExecutionResult {
            response: ProcedureExecutionResponse {
                procedure_id: req.procedure_id,
                execution_status,
                execution_memory_id: ingest.memory_id,
                event_id,
                steps,
                failure_reason: req.failure_reason.clone(),
                success_evidence: req.success_evidence.clone(),
                semantic_recovery_guidance,
                semantic_failure_motifs,
            },
            created_ids,
        })
        .map(|result| {
            if execution_strategy.refresh_procedure_governance {
                self.refresh_procedure_governance(&req.conversation_id, req.procedure_id);
            }
            result
        })
    }

    pub(crate) fn latest_procedure_execution_state(
        &self,
        conversation_id: &str,
        procedure_id: u64,
    ) -> Option<ProcedureExecutionState> {
        self.memories
            .iter()
            .filter(|memory| {
                memory.conversation_id == conversation_id
                    && memory.layer == MemoryLayer::Working
                    && memory
                        .procedure_execution_state
                        .as_ref()
                        .map(|state| state.procedure_id == procedure_id)
                        .unwrap_or(false)
            })
            .max_by_key(|memory| memory.created_at)
            .and_then(|memory| memory.procedure_execution_state.clone())
    }

    pub(crate) fn apply_procedure_feedback_to_semantic(
        &mut self,
        conversation_id: &str,
        procedure: &MemoryRecord,
        execution_state: &ProcedureExecutionState,
        evidence_memory_id: Option<u64>,
    ) {
        let Some(entity) = procedure.entity.as_deref() else {
            return;
        };
        let Some(fact_key) = procedure.fact_key.as_deref() else {
            return;
        };

        let confidence_delta = match execution_state.outcome.as_str() {
            "completed" => {
                0.05 + if execution_state
                    .success_evidence
                    .as_ref()
                    .map(|s| !s.trim().is_empty())
                    .unwrap_or(false)
                {
                    0.03
                } else {
                    0.0
                }
            }
            "failed" => -(0.06 + execution_state.retry_count.saturating_sub(1) as f32 * 0.02),
            "in_progress" => 0.01,
            _ => 0.0,
        };

        if confidence_delta == 0.0 {
            return;
        }

        for memory in self.memories.iter_mut().filter(|memory| {
            memory.conversation_id == conversation_id
                && memory.layer == MemoryLayer::Fact
                && memory.entity.as_deref() == Some(entity)
                && memory.fact_key.as_deref() == Some(fact_key)
                && memory.semantic_node.is_some()
                && match (
                    procedure.fact_value.as_deref(),
                    memory.fact_value.as_deref(),
                ) {
                    (Some(expected), Some(actual)) => expected == actual,
                    (None, _) => true,
                    _ => false,
                }
        }) {
            if let Some(node) = memory.semantic_node.as_mut() {
                node.confidence = (node.confidence + confidence_delta).clamp(0.0, 1.0);
                match execution_state.outcome.as_str() {
                    "failed" if execution_state.retry_count >= 2 || node.confidence <= 0.45 => {
                        node.status = SemanticStatus::Disputed;
                    }
                    "completed"
                        if node.status == SemanticStatus::Disputed
                            && memory.semantic_conflict.is_none()
                            && node.confidence >= 0.55 =>
                    {
                        node.status = SemanticStatus::Active;
                    }
                    _ => {}
                }
                if let Some(anchor_id) = evidence_memory_id {
                    if !node.provenance_anchor_ids.iter().any(|id| *id == anchor_id) {
                        node.provenance_anchor_ids.push(anchor_id);
                    }
                }
            }
        }
    }

    pub(crate) fn procedure_failure_motif_count(
        &self,
        conversation_id: &str,
        procedure: &MemoryRecord,
    ) -> usize {
        let Some(entity) = procedure.entity.as_deref() else {
            return 0;
        };
        let Some(fact_key) = procedure.fact_key.as_deref() else {
            return 0;
        };
        let failure_key = format!("{}_failure_motif", fact_key);
        self.memories
            .iter()
            .filter(|memory| {
                memory.conversation_id == conversation_id
                    && memory.layer == MemoryLayer::Fact
                    && memory.entity.as_deref() == Some(entity)
                    && memory.fact_key.as_deref() == Some(failure_key.as_str())
            })
            .filter_map(|memory| memory.fact_value.clone())
            .collect::<HashSet<_>>()
            .len()
    }

    pub(crate) fn has_newer_procedure_version(
        &self,
        conversation_id: &str,
        procedure: &MemoryRecord,
    ) -> bool {
        self.memories.iter().any(|memory| {
            memory.conversation_id == conversation_id
                && memory.layer == MemoryLayer::Procedure
                && memory.fact_key == procedure.fact_key
                && memory.id != procedure.id
                && memory.version > procedure.version
        })
    }

    pub(crate) fn apply_replacement_feedback_to_semantic(
        &mut self,
        conversation_id: &str,
        procedure: &MemoryRecord,
        synthesized_failure_motif_count: usize,
        evidence_memory_ids: &[u64],
    ) {
        if synthesized_failure_motif_count == 0 {
            return;
        }
        let Some(entity) = procedure.entity.as_deref() else {
            return;
        };
        let Some(fact_key) = procedure.fact_key.as_deref() else {
            return;
        };

        let confidence_penalty = (0.04 + synthesized_failure_motif_count as f32 * 0.02).min(0.14);
        for memory in self.memories.iter_mut().filter(|memory| {
            memory.conversation_id == conversation_id
                && memory.layer == MemoryLayer::Fact
                && memory.entity.as_deref() == Some(entity)
                && memory.fact_key.as_deref() == Some(fact_key)
                && match (
                    procedure.fact_value.as_deref(),
                    memory.fact_value.as_deref(),
                ) {
                    (Some(expected), Some(actual)) => expected == actual,
                    (None, _) => true,
                    _ => false,
                }
        }) {
            if let Some(node) = memory.semantic_node.as_mut() {
                node.confidence = (node.confidence - confidence_penalty).clamp(0.0, 1.0);
                if node.confidence <= 0.45 || synthesized_failure_motif_count >= 2 {
                    node.status = SemanticStatus::Disputed;
                }
                for anchor_id in evidence_memory_ids {
                    if !node.provenance_anchor_ids.iter().any(|id| id == anchor_id) {
                        node.provenance_anchor_ids.push(*anchor_id);
                    }
                }
            }
        }
    }

    pub(crate) fn refresh_procedure_governance(
        &mut self,
        conversation_id: &str,
        procedure_id: u64,
    ) {
        let Some(target) = self.memories.iter().find(|memory| {
            memory.conversation_id == conversation_id
                && memory.id == procedure_id
                && memory.layer == MemoryLayer::Procedure
        }) else {
            return;
        };
        let Some(fact_key) = target.fact_key.clone() else {
            return;
        };

        let related_ids = self
            .memories
            .iter()
            .filter(|memory| {
                memory.conversation_id == conversation_id
                    && memory.layer == MemoryLayer::Procedure
                    && memory.fact_key.as_deref() == Some(fact_key.as_str())
            })
            .map(|memory| memory.id)
            .collect::<Vec<_>>();
        if related_ids.is_empty() {
            return;
        }

        let mut evaluations = related_ids
            .iter()
            .filter_map(|id| self.base_procedure_evaluation(conversation_id, *id))
            .collect::<Vec<_>>();
        evaluations.sort_by(|a, b| {
            b.health_score
                .partial_cmp(&a.health_score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| b.version.cmp(&a.version))
        });
        let preferred_id = evaluations
            .first()
            .map(|evaluation| evaluation.procedure_id);
        let preferred_health = evaluations
            .first()
            .map(|evaluation| evaluation.health_score);
        let failure_motif_pressure = self
            .memories
            .iter()
            .filter(|memory| {
                memory.conversation_id == conversation_id
                    && memory.layer == MemoryLayer::Procedure
                    && memory.fact_key.as_deref() == Some(fact_key.as_str())
            })
            .map(|memory| {
                (
                    memory.id,
                    self.procedure_failure_motif_count(conversation_id, memory),
                )
            })
            .collect::<HashMap<_, _>>();

        for memory in self.memories.iter_mut().filter(|memory| {
            memory.conversation_id == conversation_id
                && memory.layer == MemoryLayer::Procedure
                && memory.fact_key.as_deref() == Some(fact_key.as_str())
        }) {
            let evaluation = evaluations
                .iter()
                .find(|evaluation| evaluation.procedure_id == memory.id);
            let preferred = Some(memory.id) == preferred_id;
            let motif_pressure = failure_motif_pressure.get(&memory.id).copied().unwrap_or(0);
            let auto_deprecated = evaluation
                .map(|evaluation| {
                    (evaluation.deprecate_recommended && !preferred)
                        || (!preferred
                            && motif_pressure > 0
                            && preferred_health
                                .map(|score| score > evaluation.health_score)
                                .unwrap_or(false))
                })
                .unwrap_or(false);
            let last_health_score = evaluation.map(|evaluation| evaluation.health_score);
            memory.procedure_governance = Some(ProcedureGovernance {
                preferred,
                auto_deprecated,
                last_health_score,
            });
        }
    }

    pub(crate) fn evaluate_procedure(
        &self,
        conversation_id: &str,
        procedure_id: u64,
    ) -> Option<ProcedureEvaluateResponse> {
        let mut evaluation = self.base_procedure_evaluation(conversation_id, procedure_id)?;
        let replacement = self
            .memories
            .iter()
            .filter(|memory| {
                memory.conversation_id == conversation_id
                    && memory.layer == MemoryLayer::Procedure
                    && memory.fact_key
                        == self
                            .memories
                            .iter()
                            .find(|m| m.id == procedure_id)
                            .and_then(|m| m.fact_key.clone())
                    && memory.id != procedure_id
            })
            .filter_map(|memory| self.base_procedure_evaluation(conversation_id, memory.id))
            .filter(|candidate| candidate.health_score > evaluation.health_score)
            .max_by(|a, b| {
                a.health_score
                    .partial_cmp(&b.health_score)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.version.cmp(&b.version))
            });
        let governance = self
            .memories
            .iter()
            .find(|memory| memory.id == procedure_id)
            .and_then(|memory| memory.procedure_governance.as_ref());
        evaluation.preferred = governance.map(|g| g.preferred).unwrap_or(false);
        let failure_motif_pressure = self
            .memories
            .iter()
            .find(|memory| memory.id == procedure_id)
            .map(|memory| self.procedure_failure_motif_count(conversation_id, memory))
            .unwrap_or(0);
        evaluation.activation_eligible = if evaluation.deprecate_recommended {
            false
        } else {
            ((!governance.map(|g| g.auto_deprecated).unwrap_or(false)) || replacement.is_none())
                && !(replacement.is_some() && failure_motif_pressure > 0 && !evaluation.preferred)
        };
        evaluation.replacement_procedure_id = replacement.map(|item| item.procedure_id);
        Some(evaluation)
    }

    pub(crate) fn base_procedure_evaluation(
        &self,
        conversation_id: &str,
        procedure_id: u64,
    ) -> Option<ProcedureEvaluateResponse> {
        let procedure = self.memories.iter().find(|memory| {
            memory.conversation_id == conversation_id
                && memory.id == procedure_id
                && memory.layer == MemoryLayer::Procedure
        })?;
        let execution_states = self
            .memories
            .iter()
            .filter(|memory| {
                memory.conversation_id == conversation_id
                    && memory
                        .procedure_execution_state
                        .as_ref()
                        .map(|state| state.procedure_id == procedure_id)
                        .unwrap_or(false)
            })
            .filter_map(|memory| memory.procedure_execution_state.as_ref())
            .collect::<Vec<_>>();

        let execution_count = execution_states.len() as u32;
        let completed_count = execution_states
            .iter()
            .filter(|state| state.outcome == "completed")
            .count() as u32;
        let failed_count = execution_states
            .iter()
            .filter(|state| state.outcome == "failed")
            .count() as u32;
        let success_rate = if execution_count == 0 {
            0.0
        } else {
            completed_count as f32 / execution_count as f32
        };
        let consecutive_failures = execution_states
            .iter()
            .rev()
            .take_while(|state| state.outcome == "failed")
            .count() as u32;
        let failure_motif_penalty = if self.has_newer_procedure_version(conversation_id, procedure)
        {
            (self.procedure_failure_motif_count(conversation_id, procedure) as f32 * 0.05).min(0.15)
        } else {
            0.0
        };
        let health_score = (success_rate * 0.7 + if procedure.version > 1 { 0.1 } else { 0.0 }
            - (consecutive_failures as f32 * 0.08)
            - if failed_count > completed_count {
                0.1
            } else {
                0.0
            }
            - failure_motif_penalty)
            .clamp(0.0, 1.0);

        Some(ProcedureEvaluateResponse {
            procedure_id,
            version: procedure.version,
            execution_count,
            completed_count,
            failed_count,
            success_rate,
            consecutive_failures,
            health_score,
            upgrade_recommended: failed_count >= 2 && success_rate < 0.5,
            deprecate_recommended: failed_count >= 2
                && completed_count == 0
                && consecutive_failures >= 2,
            activation_eligible: true,
            preferred: false,
            replacement_procedure_id: None,
        })
    }

    pub(crate) fn consolidate_procedure(
        &mut self,
        conversation_id: &str,
        procedure_id: u64,
    ) -> Option<ProcedureConsolidateResponse> {
        let procedure = self
            .memories
            .iter()
            .find(|memory| {
                memory.conversation_id == conversation_id
                    && memory.id == procedure_id
                    && memory.layer == MemoryLayer::Procedure
                    && memory.procedure_schema.is_some()
            })
            .cloned()?;
        let base_schema = procedure.procedure_schema.clone()?;
        let execution_states = self
            .memories
            .iter()
            .filter(|memory| {
                memory.conversation_id == conversation_id
                    && memory
                        .procedure_execution_state
                        .as_ref()
                        .map(|state| state.procedure_id == procedure_id)
                        .unwrap_or(false)
            })
            .filter_map(|memory| memory.procedure_execution_state.clone())
            .collect::<Vec<_>>();

        if execution_states.len() < 2 {
            return Some(ProcedureConsolidateResponse {
                consolidated: false,
                procedure_id: None,
                version: None,
                synthesized_trigger_conditions: Vec::new(),
                synthesized_rollback_steps: Vec::new(),
                synthesized_success_criteria: Vec::new(),
                synthesized_failure_motifs: Vec::new(),
                synthesized_fact_ids: Vec::new(),
            });
        }

        let mut failed_step_counts: HashMap<String, u32> = HashMap::new();
        let mut failure_reason_counts: HashMap<String, u32> = HashMap::new();
        let mut success_evidence = Vec::new();
        for state in &execution_states {
            for step in &state.steps {
                if step.status == "failed" {
                    *failed_step_counts.entry(step.step.clone()).or_insert(0) += 1;
                }
            }
            if let Some(reason) = state.failure_reason.as_ref() {
                let trimmed = reason.trim();
                if !trimmed.is_empty() {
                    *failure_reason_counts
                        .entry(trimmed.to_string())
                        .or_insert(0) += 1;
                }
            }
            if let Some(evidence) = state.success_evidence.as_ref() {
                if !evidence.trim().is_empty() {
                    success_evidence.push(evidence.trim().to_string());
                }
            }
        }

        let synthesized_trigger_conditions = failed_step_counts
            .iter()
            .filter(|(_, count)| **count >= 2)
            .map(|(step, _)| format!("执行前检查 {} 的前置条件", step))
            .filter(|item| {
                !base_schema
                    .trigger_conditions
                    .iter()
                    .any(|existing| existing == item)
            })
            .collect::<Vec<_>>();
        let synthesized_rollback_steps = failed_step_counts
            .iter()
            .filter(|(_, count)| **count >= 2)
            .map(|(step, _)| format!("若 {} 失败，先补齐前置条件再重试", step))
            .filter(|item| {
                !base_schema
                    .rollback_steps
                    .iter()
                    .any(|existing| existing == item)
            })
            .collect::<Vec<_>>();
        let synthesized_success_criteria = success_evidence
            .first()
            .map(|evidence| format!("复用已验证证据: {}", truncate(evidence, 48)))
            .into_iter()
            .filter(|item| {
                !base_schema
                    .success_criteria
                    .iter()
                    .any(|existing| existing == item)
            })
            .collect::<Vec<_>>();
        let synthesized_failure_motifs = failure_reason_counts
            .iter()
            .filter(|(_, count)| **count >= 2)
            .map(|(reason, _)| reason.clone())
            .filter(|item| {
                !self.memories.iter().any(|memory| {
                    memory.conversation_id == conversation_id
                        && memory.layer == MemoryLayer::Fact
                        && memory.entity == procedure.entity
                        && memory.fact_key.as_deref()
                            == procedure
                                .fact_key
                                .as_ref()
                                .map(|key| format!("{}_failure_motif", key))
                                .as_deref()
                        && memory.fact_value.as_deref() == Some(item.as_str())
                })
            })
            .collect::<Vec<_>>();

        if synthesized_trigger_conditions.is_empty()
            && synthesized_rollback_steps.is_empty()
            && synthesized_success_criteria.is_empty()
            && synthesized_failure_motifs.is_empty()
        {
            return Some(ProcedureConsolidateResponse {
                consolidated: false,
                procedure_id: None,
                version: None,
                synthesized_trigger_conditions,
                synthesized_rollback_steps,
                synthesized_success_criteria,
                synthesized_failure_motifs,
                synthesized_fact_ids: Vec::new(),
            });
        }

        let version = self
            .memories
            .iter()
            .filter(|memory| {
                memory.conversation_id == conversation_id
                    && memory.layer == MemoryLayer::Procedure
                    && memory.fact_key == procedure.fact_key
            })
            .map(|memory| memory.version)
            .max()
            .unwrap_or(1)
            + 1;

        let mut next_schema = base_schema.clone();
        next_schema
            .trigger_conditions
            .extend(synthesized_trigger_conditions.iter().cloned());
        next_schema
            .rollback_steps
            .extend(synthesized_rollback_steps.iter().cloned());
        next_schema
            .success_criteria
            .extend(synthesized_success_criteria.iter().cloned());

        let content = render_procedure_content(&next_schema);
        let id = ID_GEN.fetch_add(1, AtomicOrdering::Relaxed);
        let created_at = Utc::now();
        self.memories.push(MemoryRecord {
            id,
            conversation_id: conversation_id.to_string(),
            layer: MemoryLayer::Procedure,
            content: content.clone(),
            tokens: Self::tokenize(&content),
            source: "procedure_consolidator".to_string(),
            trust: 0.9,
            created_at,
            anchor_ids: execution_states
                .iter()
                .filter_map(|state| {
                    self.memories
                        .iter()
                        .find(|memory| {
                            memory.conversation_id == conversation_id
                                && memory
                                    .procedure_execution_state
                                    .as_ref()
                                    .map(|s| {
                                        s.procedure_id == state.procedure_id
                                            && s.retry_count == state.retry_count
                                            && s.outcome == state.outcome
                                    })
                                    .unwrap_or(false)
                        })
                        .map(|memory| memory.id)
                })
                .collect(),
            entity: procedure.entity.clone(),
            fact_key: procedure.fact_key.clone(),
            fact_value: procedure.fact_value.clone(),
            memory_type: None,
            confidence: None,
            freshness: None,
            importance: None,
            last_verified_at: None,
            supersedes: Vec::new(),
            superseded_by: Vec::new(),
            version,
            event_metadata: None,
            procedure_schema: Some(next_schema),
            procedure_execution_state: None,
            procedure_governance: Some(ProcedureGovernance {
                preferred: false,
                auto_deprecated: false,
                last_health_score: None,
            }),
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

        let mut synthesized_fact_ids = Vec::new();
        for synthesized in &synthesized_trigger_conditions {
            let fact_id = ID_GEN.fetch_add(1, AtomicOrdering::Relaxed);
            let content = format!(
                "过程知识内化：流程 {} 学到新的前置条件：{}",
                procedure
                    .fact_key
                    .clone()
                    .unwrap_or_else(|| "unknown_procedure".to_string()),
                synthesized
            );
            let req = IngestRequest {
                conversation_id: conversation_id.to_string(),
                layer: MemoryLayer::Fact,
                content,
                source: Some("procedure_semantic_internalizer".to_string()),
                trust: Some(0.88),
                entity: procedure.entity.clone(),
                fact_key: procedure
                    .fact_key
                    .as_ref()
                    .map(|key| format!("{}_precondition", key)),
                fact_value: Some(synthesized.clone()),
            };
            let resp = self.ingest_with_reserved_id(req, Some(fact_id));
            if resp.accepted {
                synthesized_fact_ids.push(fact_id);
            }
        }
        for synthesized in &synthesized_rollback_steps {
            let fact_id = ID_GEN.fetch_add(1, AtomicOrdering::Relaxed);
            let content = format!(
                "过程知识内化：流程 {} 学到新的恢复模式：{}",
                procedure
                    .fact_key
                    .clone()
                    .unwrap_or_else(|| "unknown_procedure".to_string()),
                synthesized
            );
            let req = IngestRequest {
                conversation_id: conversation_id.to_string(),
                layer: MemoryLayer::Fact,
                content,
                source: Some("procedure_semantic_internalizer".to_string()),
                trust: Some(0.86),
                entity: procedure.entity.clone(),
                fact_key: procedure
                    .fact_key
                    .as_ref()
                    .map(|key| format!("{}_recovery_pattern", key)),
                fact_value: Some(synthesized.clone()),
            };
            let resp = self.ingest_with_reserved_id(req, Some(fact_id));
            if resp.accepted {
                synthesized_fact_ids.push(fact_id);
            }
        }
        for synthesized in &synthesized_failure_motifs {
            let fact_id = ID_GEN.fetch_add(1, AtomicOrdering::Relaxed);
            let content = format!(
                "过程知识内化：流程 {} 暴露出重复失败模式：{}",
                procedure
                    .fact_key
                    .clone()
                    .unwrap_or_else(|| "unknown_procedure".to_string()),
                synthesized
            );
            let req = IngestRequest {
                conversation_id: conversation_id.to_string(),
                layer: MemoryLayer::Fact,
                content,
                source: Some("procedure_semantic_internalizer".to_string()),
                trust: Some(0.84),
                entity: procedure.entity.clone(),
                fact_key: procedure
                    .fact_key
                    .as_ref()
                    .map(|key| format!("{}_failure_motif", key)),
                fact_value: Some(synthesized.clone()),
            };
            let resp = self.ingest_with_reserved_id(req, Some(fact_id));
            if resp.accepted {
                synthesized_fact_ids.push(fact_id);
            }
        }
        self.apply_replacement_feedback_to_semantic(
            conversation_id,
            &procedure,
            synthesized_failure_motifs.len(),
            &synthesized_fact_ids,
        );
        self.invalidate_cache_for_conversation(Some(conversation_id));

        let consolidation_strategy = MetaMemoryController::procedure_consolidation_strategy();
        Some(ProcedureConsolidateResponse {
            consolidated: true,
            procedure_id: Some(id),
            version: Some(version),
            synthesized_trigger_conditions,
            synthesized_rollback_steps,
            synthesized_success_criteria,
            synthesized_failure_motifs,
            synthesized_fact_ids,
        })
        .map(|response| {
            if consolidation_strategy.refresh_procedure_governance {
                self.refresh_procedure_governance(conversation_id, procedure_id);
                if let Some(new_id) = response.procedure_id {
                    self.refresh_procedure_governance(conversation_id, new_id);
                }
            }
            response
        })
    }
}
