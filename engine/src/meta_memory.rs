use crate::semantic::SemanticStatus;
use crate::{MemoryLayer, MemoryRecord};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetaMemoryState {
    pub salience_score: f32,
    pub stability_score: f32,
    pub utility_score: f32,
    pub conflict_pressure: f32,
    pub recommended_action: String,
    pub action_priority: Vec<String>,
    pub downstream_job_hints: Vec<String>,
    pub last_decision_at: DateTime<Utc>,
    pub last_decision_reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MetaMemoryAction {
    AcceptWrite,
    RejectLowSignal,
    DetectConflict,
    RefreshCanonicalSemantic,
    UpdateProcedureGovernance,
    ApplyProcedureSemanticFeedback,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetaMemoryStrategy {
    pub(crate) primary_action: MetaMemoryAction,
    pub(crate) refresh_canonical_semantic: bool,
    pub(crate) refresh_procedure_governance: bool,
    pub(crate) apply_procedure_feedback: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct MetaMemoryController;

#[derive(Debug, Clone, PartialEq, Eq)]
enum MetaMemoryRecommendedAction {
    Retain,
    Defer,
    Consolidate,
    Forget,
    Repair,
    Deprecate,
}

impl MetaMemoryController {
    pub(crate) fn write_strategy(
        compact_len: usize,
        score: f32,
        threshold: f32,
    ) -> MetaMemoryStrategy {
        let primary_action = if compact_len < 6 || score < threshold {
            MetaMemoryAction::RejectLowSignal
        } else {
            MetaMemoryAction::AcceptWrite
        };
        MetaMemoryStrategy {
            primary_action,
            refresh_canonical_semantic: false,
            refresh_procedure_governance: false,
            apply_procedure_feedback: false,
        }
    }

    pub(crate) fn fact_conflict_strategy(
        previous_value: Option<&str>,
        next_value: &str,
    ) -> MetaMemoryStrategy {
        let primary_action = if previous_value == Some(next_value) || previous_value.is_none() {
            MetaMemoryAction::AcceptWrite
        } else {
            MetaMemoryAction::DetectConflict
        };
        MetaMemoryStrategy {
            primary_action,
            refresh_canonical_semantic: false,
            refresh_procedure_governance: false,
            apply_procedure_feedback: false,
        }
    }

    pub(crate) fn canonical_refresh_strategy() -> MetaMemoryStrategy {
        MetaMemoryStrategy {
            primary_action: MetaMemoryAction::RefreshCanonicalSemantic,
            refresh_canonical_semantic: true,
            refresh_procedure_governance: false,
            apply_procedure_feedback: false,
        }
    }

    pub(crate) fn procedure_execution_strategy() -> MetaMemoryStrategy {
        MetaMemoryStrategy {
            primary_action: MetaMemoryAction::ApplyProcedureSemanticFeedback,
            refresh_canonical_semantic: false,
            refresh_procedure_governance: true,
            apply_procedure_feedback: true,
        }
    }

    pub(crate) fn procedure_consolidation_strategy() -> MetaMemoryStrategy {
        MetaMemoryStrategy {
            primary_action: MetaMemoryAction::UpdateProcedureGovernance,
            refresh_canonical_semantic: false,
            refresh_procedure_governance: true,
            apply_procedure_feedback: false,
        }
    }

    pub(crate) fn evaluate_memory(memory: &MemoryRecord) -> MetaMemoryState {
        let layer_salience_bonus = match memory.layer {
            MemoryLayer::SessionSummary => 0.08,
            MemoryLayer::Foresight => 0.06,
            MemoryLayer::SelfModel | MemoryLayer::Autobiographical => 0.05,
            _ => 0.0,
        };
        let layer_utility_bonus = match memory.layer {
            MemoryLayer::Foresight => 0.30,
            MemoryLayer::SelfModel | MemoryLayer::Autobiographical => 0.35,
            MemoryLayer::SessionSummary => 0.20,
            MemoryLayer::Episodic => 0.15,
            _ => 0.0,
        };
        let salience_score = (memory.trust * 0.45
            + if memory.layer == MemoryLayer::Procedure {
                0.25
            } else {
                0.0
            }
            + if memory.semantic_node.is_some() {
                0.2
            } else {
                0.0
            }
            + if memory.episodic_metadata.is_some() {
                0.1
            } else {
                0.0
            }
            + layer_salience_bonus)
            .clamp(0.0, 1.0);

        let stability_score = memory
            .semantic_node
            .as_ref()
            .map(|node| node.confidence)
            .or_else(|| {
                memory.procedure_execution_state.as_ref().map(|state| {
                    if state.outcome == "completed" {
                        0.9
                    } else {
                        0.35
                    }
                })
            })
            .unwrap_or(memory.trust)
            .clamp(0.0, 1.0);

        let utility_score = ((memory.retrieval_hits as f32 / 5.0).min(1.0) * 0.5
            + if memory.procedure_governance.is_some() {
                0.25
            } else {
                0.0
            }
            + if memory.fact_key.is_some() { 0.15 } else { 0.0 }
            + if memory.source == "procedure_semantic_internalizer" {
                0.1
            } else {
                0.0
            }
            + layer_utility_bonus)
            .clamp(0.0, 1.0);

        let conflict_pressure = (if memory.semantic_conflict.is_some() {
            0.6_f32
        } else {
            0.0_f32
        } + memory
            .procedure_execution_state
            .as_ref()
            .map(|state| {
                if state.outcome == "failed" {
                    0.25_f32
                } else {
                    0.0_f32
                }
            })
            .unwrap_or(0.0_f32)
            + memory
                .semantic_node
                .as_ref()
                .map(|node| {
                    if node.status == SemanticStatus::Disputed {
                        0.25_f32
                    } else {
                        0.0_f32
                    }
                })
                .unwrap_or(0.0_f32))
        .clamp(0.0, 1.0);

        let recommended_action = if conflict_pressure >= 0.75 {
            MetaMemoryRecommendedAction::Repair
        } else if stability_score >= 0.88 && utility_score >= 0.45 {
            MetaMemoryRecommendedAction::Consolidate
        } else if salience_score < 0.28 && utility_score < 0.2 {
            MetaMemoryRecommendedAction::Forget
        } else if memory
            .semantic_node
            .as_ref()
            .map(|n| n.status == SemanticStatus::Deprecated)
            .unwrap_or(false)
        {
            MetaMemoryRecommendedAction::Deprecate
        } else if salience_score < 0.4 {
            MetaMemoryRecommendedAction::Defer
        } else {
            MetaMemoryRecommendedAction::Retain
        };

        let mut action_priority = Vec::new();
        if conflict_pressure >= 0.75 {
            action_priority.push("repair".to_string());
        }
        if stability_score >= 0.88 && utility_score >= 0.45 {
            action_priority.push("consolidate".to_string());
        }
        if salience_score < 0.4 {
            action_priority.push("defer".to_string());
        }
        if memory
            .semantic_node
            .as_ref()
            .map(|n| n.status == SemanticStatus::Deprecated)
            .unwrap_or(false)
        {
            action_priority.push("deprecate".to_string());
        }
        if salience_score < 0.28 && utility_score < 0.2 {
            action_priority.push("forget".to_string());
        }
        if action_priority.is_empty() {
            action_priority.push("retain".to_string());
        }

        let mut downstream_job_hints = Vec::new();
        if action_priority.iter().any(|a| a == "repair") {
            downstream_job_hints.push("semantic_sleep_cycle:high_priority".to_string());
        }
        if action_priority.iter().any(|a| a == "consolidate") {
            downstream_job_hints.push("semantic_sleep_cycle:consolidate_candidate".to_string());
        }
        if utility_score >= 0.35
            || (memory.layer == MemoryLayer::Episodic && memory.episodic_metadata.is_some())
        {
            downstream_job_hints.push("semantic_sleep_cycle:review_candidate".to_string());
        }

        MetaMemoryState {
            salience_score,
            stability_score,
            utility_score,
            conflict_pressure,
            recommended_action: match recommended_action {
                MetaMemoryRecommendedAction::Retain => "retain",
                MetaMemoryRecommendedAction::Defer => "defer",
                MetaMemoryRecommendedAction::Consolidate => "consolidate",
                MetaMemoryRecommendedAction::Forget => "forget",
                MetaMemoryRecommendedAction::Repair => "repair",
                MetaMemoryRecommendedAction::Deprecate => "deprecate",
            }
            .to_string(),
            action_priority,
            downstream_job_hints,
            last_decision_at: Utc::now(),
            last_decision_reason: format!(
                "salience={:.2}, stability={:.2}, utility={:.2}, conflict={:.2}",
                salience_score, stability_score, utility_score, conflict_pressure
            ),
        }
    }
}
